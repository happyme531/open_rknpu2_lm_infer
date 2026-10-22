// Adapted from ggml_compute_forward_gated_delta_net_one_chunk, llama.cpp
// a17e94d229fc7fed1567a799baebd11b771fe5af, MIT license.
// Copyright (c) 2023-2026 The ggml authors.
// Full license: ../../llm_generated_docs/THIRD_PARTY_NOTICES.md.
#pragma once
#include "../../native/cpu_kernels/kda.h"
#include "../../native/cpu_kernels/vec.h"
#include <vector>

extern "C" void kda_reference(const rknpu2_kda_params * p) {
    using namespace rknpu2_cpu_kernels;
    const size_t d = p->dimension, width = p->heads * d;
    const float scaling = 1.0f / std::sqrt(static_cast<float>(d));
    #pragma omp parallel num_threads(p->threads)
    {
        std::vector<float> decay(d), delta(d);
        #pragma omp for schedule(static)
        for (size_t h = 0; h < p->heads; ++h) {
            float * state = p->state + h*d*d;
            for (size_t t = 0; t < p->rows; ++t) {
                const size_t at = t*width + h*d;
                const float * key = p->key + at;
                const float * query = p->query + at;
                for (size_t k = 0; k < d; ++k) decay[k] = std::exp(p->log_decay[at+k]);
                for (size_t v = 0; v < d; ++v) {
                    float * row = state + v*d;
                    for (size_t k = 0; k < d; ++k) row[k] *= decay[k];
                    delta[v] = (p->value[at+v] - dot(d, row, key)) * p->beta[t*p->heads+h];
                }
                for (size_t v = 0; v < d; ++v) {
                    float * row = state + v*d;
                    mad(d, row, key, delta[v]);
                    p->output[at+v] = dot(d, row, query) * scaling;
                }
            }
        }
    }
}
