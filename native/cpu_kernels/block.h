#pragma once
#include "vec.h"

namespace rknpu2_cpu_kernels {
#if defined(__aarch64__)
// Value rows share each key/query load, with ggml's four accumulators per row.
template<int rows, bool output, class Decay>
inline void update_block(float * state, const float * key, Decay decay,
                          float beta, const float * value, const float * query,
                          float * out, float scaling) {
    float32x4_t sums[rows][4];
    for (auto & row : sums) for (auto & sum : row) sum = vdupq_n_f32(0);
    for (int i = 0; i < 128; i += 16) {
        for (int j = 0; j < 4; ++j) {
            const int at = i + 4*j;
            const float32x4_t k = vld1q_f32(key + at);
            const auto factor = decay_lanes(decay, at);
            for (int r = 0; r < rows; ++r) {
                float * s = state + r*128 + at;
                const float32x4_t v = apply_decay(vld1q_f32(s), factor);
                vst1q_f32(s, v);
                sums[r][j] = vfmaq_f32(sums[r][j], v, k);
            }
        }
    }
    float delta[rows];
    for (int r = 0; r < rows; ++r) {
        const float predicted = vaddvq_f32(vaddq_f32(vaddq_f32(sums[r][0], sums[r][2]), vaddq_f32(sums[r][1], sums[r][3])));
        delta[r] = (value[r] - predicted) * beta;
        for (auto & sum : sums[r]) sum = vdupq_n_f32(0);
    }
    for (int i = 0; i < 128; i += 16) {
        for (int j = 0; j < 4; ++j) {
            const int at = i + 4*j;
            const float32x4_t k = vld1q_f32(key + at);
            float32x4_t q;
            if constexpr (output) q = vld1q_f32(query + at);
            for (int r = 0; r < rows; ++r) {
                float * s = state + r*128 + at;
                const float32x4_t v = vfmaq_n_f32(vld1q_f32(s), k, delta[r]);
                vst1q_f32(s, v);
                if constexpr (output) sums[r][j] = vfmaq_f32(sums[r][j], v, q);
            }
        }
    }
    if constexpr (output) {
        for (int r = 0; r < rows; ++r) {
            out[r] = vaddvq_f32(vaddq_f32(vaddq_f32(sums[r][0], sums[r][2]), vaddq_f32(sums[r][1], sums[r][3]))) * scaling;
        }
    }
}
#endif
} // namespace rknpu2_cpu_kernels
