#include "rk_npu_swiglu_fp32_neon.h"
#include <cmath>
#include <cstddef>
#include <cstring>

// Private frontend ABI. Rust supplies equal-length, nonoverlapping slices.
// NOTE: shares the operator library's FP32 approximation and signed-zero
// negative tail. There is no runtime FP64 or scalar precision fallback.
extern "C" void rknpu2_cpu_swiglu(const float* gate, const float* up,
                                  float* output, size_t count) {
#if defined(__aarch64__)
    size_t i = 0;
    for (; i + 16 <= count; i += 16)
        rknpu2_matmul_open::detail::swiglu_fp32::apply16<false>(gate+i, up+i, output+i);
    if (i < count) {
        alignas(16) float g[16]{}, u[16]{}, y[16];
        const size_t bytes = (count-i)*sizeof(float);
        std::memcpy(g, gate+i, bytes);
        std::memcpy(u, up+i, bytes);
        rknpu2_matmul_open::detail::swiglu_fp32::apply16<false>(g, u, y);
        std::memcpy(output+i, y, bytes);
    }
#else
    for (size_t i = 0; i < count; ++i)
        output[i] = (gate[i]/(1.f+std::exp(-gate[i])))*up[i];
#endif
}
