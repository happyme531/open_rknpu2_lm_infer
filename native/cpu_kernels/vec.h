// Adapted from llama.cpp ggml-cpu/vec.cpp, vec.h and simd-mappings.h.
// Upstream a17e94d229fc7fed1567a799baebd11b771fe5af, MIT license.
// Copyright (c) 2023-2026 The ggml authors.
// Full license: ../../llm_generated_docs/THIRD_PARTY_NOTICES.md.
#pragma once
#include <cmath>
#include <cstddef>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace rknpu2_cpu_kernels {
inline float dot(size_t n, const float * x, const float * y) {
    size_t i = 0;
    float result = 0;
#if defined(__aarch64__)
    float32x4_t sum[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    for (; i + 16 <= n; i += 16) {
        for (int j = 0; j < 4; ++j) {
            sum[j] = vfmaq_f32(sum[j], vld1q_f32(x + i + 4*j), vld1q_f32(y + i + 4*j));
        }
    }
    result = vaddvq_f32(vaddq_f32(vaddq_f32(sum[0], sum[2]), vaddq_f32(sum[1], sum[3])));
#endif
    for (; i < n; ++i) result += x[i] * y[i];
    return result;
}

inline void scale(size_t n, float * x, float decay) {
    for (size_t i = 0; i < n; ++i) x[i] *= decay;
}

inline void mad(size_t n, float * x, const float * key, float delta) {
    size_t i = 0;
#if defined(__aarch64__)
    for (; i + 16 <= n; i += 16) {
        for (int j = 0; j < 4; ++j) {
            vst1q_f32(x + i + 4*j, vfmaq_n_f32(vld1q_f32(x + i + 4*j), vld1q_f32(key + i + 4*j), delta));
        }
    }
#endif
    for (; i < n; ++i) x[i] += delta * key[i];
}

inline float decay_at(float decay, size_t) { return decay; }
inline float decay_at(const float * decay, size_t i) { return decay[i]; }
#if defined(__aarch64__)
inline float decay_lanes(float decay, size_t) { return decay; }
inline float32x4_t decay_lanes(const float * decay, size_t i) { return vld1q_f32(decay + i); }
inline float32x4_t apply_decay(float32x4_t state, float decay) { return vmulq_n_f32(state, decay); }
inline float32x4_t apply_decay(float32x4_t state, float32x4_t decay) { return vmulq_f32(state, decay); }
#endif

// The same arithmetic and reduction as ggml, merging decay+dot and mad+dot
// while each state row is in registers. Decay is scalar for GDN and per-key
// channel for KDA. No FP16 state or relaxed FP flags.
template<bool output, class Decay>
inline float update_row(size_t n, float * state, const float * key,
                        Decay decay, float beta, float value, const float * query) {
    size_t i = 0;
    float predicted = 0;
#if defined(__aarch64__)
    float32x4_t sum[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    for (; i + 16 <= n; i += 16) {
        for (int j = 0; j < 4; ++j) {
            const size_t at = i + 4*j;
            const float32x4_t s = apply_decay(vld1q_f32(state + at), decay_lanes(decay, at));
            vst1q_f32(state + at, s);
            sum[j] = vfmaq_f32(sum[j], s, vld1q_f32(key + at));
        }
    }
    predicted = vaddvq_f32(vaddq_f32(vaddq_f32(sum[0], sum[2]), vaddq_f32(sum[1], sum[3])));
#endif
    for (; i < n; ++i) {
        state[i] *= decay_at(decay, i);
        predicted += state[i] * key[i];
    }
    const float delta = (value - predicted) * beta;
    float result = 0;
    i = 0;
#if defined(__aarch64__)
    for (auto & v : sum) v = vdupq_n_f32(0);
    for (; i + 16 <= n; i += 16) {
        for (int j = 0; j < 4; ++j) {
            const size_t at = i + 4*j;
            const float32x4_t s = vfmaq_n_f32(vld1q_f32(state + at), vld1q_f32(key + at), delta);
            vst1q_f32(state + at, s);
            if constexpr (output) sum[j] = vfmaq_f32(sum[j], s, vld1q_f32(query + at));
        }
    }
    if constexpr (output) result = vaddvq_f32(vaddq_f32(vaddq_f32(sum[0], sum[2]), vaddq_f32(sum[1], sum[3])));
#endif
    for (; i < n; ++i) {
        state[i] += delta * key[i];
        if constexpr (output) result += state[i] * query[i];
    }
    return result;
}
} // namespace rknpu2_cpu_kernels
