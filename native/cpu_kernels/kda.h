#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Token-major Q/K/V and per-key-channel log decay; state [head,value,key].
// Q/K are L2 normalized by the caller. beta is already sigmoid activated.
struct rknpu2_kda_params {
    size_t rows, heads, dimension, threads;
    float * state;
    const float * query;
    const float * key;
    const float * value;
    const float * log_decay;
    const float * beta;
    float * output;
};
void rknpu2_kda_run(const struct rknpu2_kda_params * params);
#ifdef __cplusplus
}
#endif
