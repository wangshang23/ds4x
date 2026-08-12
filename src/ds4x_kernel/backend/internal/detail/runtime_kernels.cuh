#ifndef DS4X_BACKEND_RUNTIME_KERNELS_CUH
#define DS4X_BACKEND_RUNTIME_KERNELS_CUH

#include "../backend_common.cuh"



/* mmq pipeline helpers, ported from the fork: SwiGLU + clamp + router
 * weight in the (token, slot, feature) layout ds4_mmq_*_moe writes, and
 * the guarded per-token slot sum. */
__global__ static void moe_mmq_swiglu_weighted_clamp_kernel(
        float *mid_out,
        const float *gate_buf, const float *up_buf,
        const float *weights,
        uint32_t expert_mid_dim,
        uint32_t n_tokens,
        uint32_t n_expert_used,
        float clamp) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_expert_used * expert_mid_dim;
    if (gid >= n) return;
    uint64_t slot_pair = gid / expert_mid_dim;
    uint32_t tok = (uint32_t)(slot_pair / n_expert_used);
    uint32_t slot = (uint32_t)(slot_pair - (uint64_t)tok * n_expert_used);
    float g = gate_buf[gid];
    float u = up_buf[gid];
    if (!isfinite(g)) g = 0.0f;
    if (!isfinite(u)) u = 0.0f;
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }
    const float w = weights[(uint64_t)tok * n_expert_used + slot];
    const float s = g / (1.0f + expf(-g));
    mid_out[gid] = s * u * w;
}



__global__ static void moe_mmq_sum_kernel(float *out, const float *down,
        const int32_t *selected, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_tokens, uint32_t guard_nonfinite) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) {
        if (selected && selected[(uint64_t)tok * n_expert + e] < 0) continue;
        const float v = down[((uint64_t)tok * n_expert + e) * out_dim + row];
        if (!guard_nonfinite || isfinite(v)) acc += v;
    }
    out[gid] = acc;
}

#endif  // DS4X_BACKEND_RUNTIME_KERNELS_CUH
