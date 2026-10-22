// FP32 KDA replay variants adapted from the ggml-derived vec/block helpers.
// Upstream a17e94d229fc7fed1567a799baebd11b771fe5af, MIT license.
// Copyright (c) 2023-2026 The ggml authors.
// Full license: ../../llm_generated_docs/THIRD_PARTY_NOTICES.md.
#pragma once
#include "kda.h"
#include "vec.h"
#include <algorithm>
#include <vector>

namespace kda_probe {
inline float row(size_t d, float* state, const float* key, const float* decay,
                 float beta, float value, const float* query) {
    size_t i=0; float predicted=0;
#if defined(__aarch64__)
    float32x4_t sum[4]={vdupq_n_f32(0),vdupq_n_f32(0),vdupq_n_f32(0),vdupq_n_f32(0)};
    for(;i+16<=d;i+=16)for(int j=0;j<4;++j) {
        const size_t at=i+4*j;
        const auto x=vmulq_f32(vld1q_f32(state+at),vld1q_f32(decay+at));
        vst1q_f32(state+at,x);sum[j]=vfmaq_f32(sum[j],x,vld1q_f32(key+at));
    }
    predicted=vaddvq_f32(vaddq_f32(vaddq_f32(sum[0],sum[2]),vaddq_f32(sum[1],sum[3])));
#endif
    for(;i<d;++i){state[i]*=decay[i];predicted+=state[i]*key[i];}
    const float delta=(value-predicted)*beta;
    float output=0; i=0;
#if defined(__aarch64__)
    for(auto& v:sum)v=vdupq_n_f32(0);
    for(;i+16<=d;i+=16)for(int j=0;j<4;++j) {
        const size_t at=i+4*j;
        const auto x=vfmaq_n_f32(vld1q_f32(state+at),vld1q_f32(key+at),delta);
        vst1q_f32(state+at,x);sum[j]=vfmaq_f32(sum[j],x,vld1q_f32(query+at));
    }
    output=vaddvq_f32(vaddq_f32(vaddq_f32(sum[0],sum[2]),vaddq_f32(sum[1],sum[3])));
#endif
    for(;i<d;++i){state[i]+=delta*key[i];output+=state[i]*query[i];}
    return output;
}

#if defined(__aarch64__)
template<int Rows>
inline void block(float* state,const float* key,const float* decay,float beta,
                  const float* value,const float* query,float* output,float scaling) {
    float32x4_t sums[Rows][4];
    for(auto& row:sums)for(auto& v:row)v=vdupq_n_f32(0);
    for(int i=0;i<128;i+=16)for(int j=0;j<4;++j) {
        const int at=i+4*j;
        const auto k=vld1q_f32(key+at),g=vld1q_f32(decay+at);
        for(int r=0;r<Rows;++r) {
            auto* s=state+r*128+at;
            const auto x=vmulq_f32(vld1q_f32(s),g);
            vst1q_f32(s,x);sums[r][j]=vfmaq_f32(sums[r][j],x,k);
        }
    }
    float delta[Rows];
    for(int r=0;r<Rows;++r) {
        const float predicted=vaddvq_f32(vaddq_f32(vaddq_f32(sums[r][0],sums[r][2]),vaddq_f32(sums[r][1],sums[r][3])));
        delta[r]=(value[r]-predicted)*beta;
        for(auto& v:sums[r])v=vdupq_n_f32(0);
    }
    for(int i=0;i<128;i+=16)for(int j=0;j<4;++j) {
        const int at=i+4*j;
        const auto k=vld1q_f32(key+at),q=vld1q_f32(query+at);
        for(int r=0;r<Rows;++r) {
            auto* s=state+r*128+at;
            const auto x=vfmaq_n_f32(vld1q_f32(s),k,delta[r]);
            vst1q_f32(s,x);sums[r][j]=vfmaq_f32(sums[r][j],x,q);
        }
    }
    for(int r=0;r<Rows;++r)
        output[r]=vaddvq_f32(vaddq_f32(vaddq_f32(sums[r][0],sums[r][2]),vaddq_f32(sums[r][1],sums[r][3])))*scaling;
}
#endif

template<int Block, int Tokens>
void run(const rknpu2_kda_params* p) {
    const size_t d=p->dimension,width=p->heads*d;
    const float scaling=1.f/std::sqrt(float(d));
#pragma omp parallel num_threads(p->threads)
    {
        std::vector<float> decay(Tokens*d);
#pragma omp for schedule(static)
        for(size_t h=0;h<p->heads;++h) {
            float* state=p->state+h*d*d;
            for(size_t first=0;first<p->rows;first+=Tokens) {
                const size_t count=std::min(size_t(Tokens),p->rows-first);
                for(size_t t=0;t<count;++t)for(size_t k=0;k<d;++k)
                    decay[t*d+k]=std::exp(p->log_decay[(first+t)*width+h*d+k]);
                size_t v=0;
#if defined(__aarch64__)
                if constexpr(Block>1)if(d==128) {
                    for(;v+Block<=d;v+=Block)for(size_t t=0;t<count;++t) {
                        const size_t at=(first+t)*width+h*d;
                        block<Block>(state+v*d,p->key+at,decay.data()+t*d,
                            p->beta[(first+t)*p->heads+h],p->value+at+v,p->query+at,
                            p->output+at+v,scaling);
                    }
                }
#endif
                for(;v<d;++v)for(size_t t=0;t<count;++t) {
                    const size_t at=(first+t)*width+h*d;
                    p->output[at+v]=row(d,state+v*d,p->key+at,decay.data()+t*d,
                        p->beta[(first+t)*p->heads+h],p->value[at+v],p->query+at)*scaling;
                }
            }
        }
    }
}
} // namespace kda_probe
