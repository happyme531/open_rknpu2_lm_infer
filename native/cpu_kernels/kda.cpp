// Adapted from ggml_compute_forward_gated_delta_net_one_chunk, llama.cpp
// a17e94d229fc7fed1567a799baebd11b771fe5af, MIT license.
// Copyright (c) 2023-2026 The ggml authors.
// Full license: ../../llm_generated_docs/THIRD_PARTY_NOTICES.md.
#include "kda.h"
#include "vec.h"
#include "block.h"
#include <vector>

extern "C" void rknpu2_kda_run(const rknpu2_kda_params * p) {
    using namespace rknpu2_cpu_kernels;
    const size_t d = p->dimension, width = p->heads * d;
    const float scaling = 1.0f / std::sqrt(static_cast<float>(d));
    #pragma omp parallel num_threads(p->threads)
    {
        std::vector<float> decay(d);
        #pragma omp for schedule(static)
        for (size_t h = 0; h < p->heads; ++h) {
            float * state = p->state + h*d*d;
            for (size_t t = 0; t < p->rows; ++t) {
                const size_t at = t*width + h*d;
                const float * key = p->key + at;
                const float * query = p->query + at;
                for (size_t k = 0; k < d; ++k) decay[k] = std::exp(p->log_decay[at+k]);
                const float beta = p->beta[t*p->heads+h];
                size_t v = 0;
#if defined(__aarch64__)
                if (d == 128) {
                    for (; v + 4 <= d; v += 4)
                        update_block<4, true>(state + v*d, key, decay.data(), beta,
                            p->value + at + v, query, p->output + at + v, scaling);
                }
#endif
                for (; v < d; ++v)
                    p->output[at+v] = update_row<true>(d, state + v*d, key,
                        decay.data(), beta, p->value[at+v], query) * scaling;
            }
        }
    }
}
