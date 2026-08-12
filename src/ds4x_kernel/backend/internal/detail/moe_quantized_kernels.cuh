#ifndef DS4X_BACKEND_MOE_QUANTIZED_KERNELS_CUH
#define DS4X_BACKEND_MOE_QUANTIZED_KERNELS_CUH

#include "../backend_common.cuh"

/* Vector-load variant of dev_dot_q4_K_q8_K_block: loads the whole 144-byte
 * Q4_K block with nine 16B loads (requires a 16B-aligned tensor base; block
 * stride 144 and row strides are 16B multiples), then computes the exact same
 * integer sums and float finish. Same values in the same order, so results
 * are bit-identical; the wide loads just improve DRAM/memory-level
 * parallelism for the bandwidth-bound decode matvecs. */
__device__ __forceinline__ static void dev_dot_q4_K_q8_K_block_vec(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y,
        float *out_acc) {
    const uint4 hdr = *(const uint4 *)x; /* d, dmin, scales[12] */
    uint4 qv[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) qv[i] = ((const uint4 *)(x->qs))[i];
    const uint16_t xd_u = (uint16_t)(hdr.x & 0xffffu);
    const uint16_t xmin_u = (uint16_t)(hdr.x >> 16u);
    uint8_t scales[12];
    scales[0] = (uint8_t)(hdr.y);
    scales[1] = (uint8_t)(hdr.y >> 8);
    scales[2] = (uint8_t)(hdr.y >> 16);
    scales[3] = (uint8_t)(hdr.y >> 24);
    scales[4] = (uint8_t)(hdr.z);
    scales[5] = (uint8_t)(hdr.z >> 8);
    scales[6] = (uint8_t)(hdr.z >> 16);
    scales[7] = (uint8_t)(hdr.z >> 24);
    scales[8] = (uint8_t)(hdr.w);
    scales[9] = (uint8_t)(hdr.w >> 8);
    scales[10] = (uint8_t)(hdr.w >> 16);
    scales[11] = (uint8_t)(hdr.w >> 24);
    const float xd = dev_f16_to_f32(xd_u);
    const float xmin = dev_f16_to_f32(xmin_u);
    int isum = 0;
    int summs = 0;
    const int32_t *qw = (const int32_t *)qv;
#pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t word_off = (j >> 1u) * 8u;
        const int shift = (j & 1u) ? 4 : 0;
        int32_t sum = 0;
#pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            const int32_t v = (qw[word_off + i] >> shift) & 0x0f0f0f0f;
            sum = __dp4a(v, *(const int32_t *)(y->qs + j * 32u + i * 4u), sum);
        }
        isum += (int)sc * sum;
    }
    *out_acc += y->d * xd * (float)isum - y->d * xmin * (float)summs;
}



__device__ static void dev_dot_q4_K_q8_K_block8(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        for (uint32_t p = 0; p < n; p++) {
            summs[p] += (int)m * (int)(ys[p]->bsums[2u * j] + ys[p]->bsums[2u * j + 1u]);
            isum[p] += (int)sc * dev_dot_q4_32(x->qs + byte_off, ys[p]->qs + j * 32u, shift);
        }
    }

    for (uint32_t p = 0; p < n; p++) {
        acc[p] += ys[p]->d * xd * (float)isum[p] - ys[p]->d * xmin * (float)summs[p];
    }
}



__device__ static float dev_dot_q2_K_q8_K_block(const cuda_block_q2_K *x, const cuda_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum = 0;
    int is = 0;
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}



__device__ static void dev_dot_q2_K_q8_K_block4(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}



__device__ static void dev_dot_q2_K_q8_K_block8(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}



__device__ static float half_warp_sum_f32(float v, uint32_t lane16) {
    uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 16);
    }
    (void)lane16;
    return v;
}



__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 8);
    }
    (void)lane8;
    return v;
}



__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}



/* Decode-only dual quantizer.  The Q8_0 half mirrors
 * quantize_q8_0_f32_kernel's 32-thread reduction and expression order, while
 * the Q8_K half remains byte-for-byte the ordinary routed-MoE quantizer. */
__global__ static void q8_K_q8_0_quantize_kernel(
        cuda_block_q8_K *out,
        int8_t *q8_0,
        float *q8_0_scale,
        const float *x,
        uint32_t in_dim,
        uint32_t n_rows) {
    const uint32_t b = blockIdx.x;
    const uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim +
                      (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out +
        (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;

    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            abs_part[tid] = fmaxf(abs_part[tid], abs_part[tid + stride]);
        }
        __syncthreads();
    }
    const uint32_t q8_blocks = in_dim / 32u;
    const uint32_t q8_block = b * 8u + warp;
    const float d = abs_part[warp * 32u] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (lane == 0u) {
        q8_0_scale[(uint64_t)row * q8_blocks + q8_block] = d;
    }
    int qv = (int)lrintf(v * id);
    qv = qv > 127 ? 127 : (qv < -128 ? -128 : qv);
    q8_0[((uint64_t)row * q8_blocks + q8_block) * 32u + lane] =
        (int8_t)qv;
    __syncthreads();

    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    const float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int kv = (int)lrintf(iscale_s * xr[tid]);
        if (kv > 127) kv = 127;
        if (kv < -128) kv = -128;
        yb->qs[tid] = (int8_t)kv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}



__device__ __forceinline__ static bool moe_owned_local_expert(
        int32_t expert,
        uint32_t expert_base,
        uint32_t expert_count,
        uint32_t *local_expert) {
    if (expert < 0) return false;
    const uint32_t e = (uint32_t)expert;
    if (e < expert_base || e - expert_base >= expert_count) return false;
    if (local_expert) *local_expert = e - expert_base;
    return true;
}



__global__ static void mxfp4_prepare_owned_assignments_kernel(
        int32_t *local_ids,
        float *local_weights,
        const int32_t *selected,
        const float *weights,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t slot = threadIdx.x;
    if (slot >= 6u) return;
    uint32_t local_expert = 0;
    const bool owned = moe_owned_local_expert(
            selected[slot], expert_base, expert_count, &local_expert);
    local_ids[slot] = owned ? (int32_t)local_expert : -1;
    local_weights[slot] = owned ? weights[slot] : 0.0f;
}



/* Quantize only selected slots owned by this expert-parallel rank.  Rows keep
 * their original slot index so the final rank-local reduction can visit slots
 * in canonical order without compaction or a host synchronization. */
__global__ static void q8_K_quantize_owned_kernel(
        cuda_block_q8_K *out,
        const float *x,
        const int32_t *selected,
        uint32_t in_dim,
        uint32_t n_rows,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t b = blockIdx.x;
    const uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    if (!moe_owned_local_expert(selected[row], expert_base, expert_count, NULL)) return;

    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    const uint32_t tid = threadIdx.x;
    const float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    const float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}



__global__ static void moe_filter_owned_pairs_kernel(
        int32_t *selected,
        float *weights,
        uint64_t pair_count,
        uint32_t n_total_expert,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint64_t pair = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pair_count) return;
    const int32_t expert_i = selected[pair];
    if (expert_i >= 0 && (uint32_t)expert_i < n_total_expert &&
        (uint32_t)expert_i >= expert_base &&
        (uint32_t)expert_i - expert_base < expert_count) {
        selected[pair] = expert_i - (int32_t)expert_base;
    } else {
        selected[pair] = -1;
        weights[pair] = 0.0f;
    }
}



__global__ static void q8_K_quantize_sidecar_kernel(
        cuda_block_q8_K *out,
        const float *x,
        const float *amax_sidecar,
        uint32_t in_dim,
        uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    const uint32_t blocks = in_dim / CUDA_QK_K;
    if (row >= n_rows || b >= blocks) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    const float *sc = amax_sidecar + ((uint64_t)row * blocks + b) * 32u;
    cuda_block_q8_K *yb = out + (uint64_t)row * blocks + b;
    __shared__ float abs_part[32];
    __shared__ float val_part[32];
    __shared__ float iscale_s;
    const uint32_t tid = threadIdx.x;
    if (tid < 32u) {
        const float v = sc[tid];
        abs_part[tid] = fabsf(v);
        val_part[tid] = v;
    }
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    const float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0u) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16u) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0u) iscale_s = -127.0f / val_part[0];
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16u) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16u + (uint32_t)i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0u) yb->d = 1.0f / iscale_s;
}


__global__ static void moe_gate_up_mid_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < MOE_DECODE_ROW_TILES; rr++) {
        uint32_t row = blockIdx.x * MOE_DECODE_ROWS_PER_BLOCK + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}



__global__ static void moe_gate_up_mid_decode_lut_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < MOE_DECODE_ROW_TILES; rr++) {
        uint32_t row = blockIdx.x * MOE_DECODE_ROWS_PER_BLOCK + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block_lut(gr + b, xqb + b, s_iq2_grid, s_iq2_signs);
            up += dev_dot_iq2_xxs_q8_K_block_lut(ur + b, xqb + b, s_iq2_grid, s_iq2_signs);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}



__global__ static void moe_gate_up_mid_decode_lut_owned_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t expert_base,
        uint32_t expert_count,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t expert = 0u;
    if (!moe_owned_local_expert(selected[pair], expert_base, expert_count,
                                &expert)) return;
    const cuda_block_q8_K *xqb = xq;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < MOE_DECODE_ROW_TILES; rr++) {
        uint32_t row = blockIdx.x * MOE_DECODE_ROWS_PER_BLOCK + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block_lut(gr + b, xqb + b, s_iq2_grid, s_iq2_signs);
            up += dev_dot_iq2_xxs_q8_K_block_lut(ur + b, xqb + b, s_iq2_grid, s_iq2_signs);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair];
        }
    }
}



__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) return;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}



__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t n_total_expert) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[n_total_expert] = sum;
    }
}



__global__ static void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) return;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}



__global__ static void moe_build_expert_tile_offsets_kernel(
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        const uint32_t *counts,
        uint32_t block_m,
        uint32_t n_total_expert) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            tile_offsets[e] = sum;
            sum += (counts[e] + block_m - 1u) / block_m;
        }
        tile_offsets[n_total_expert] = sum;
        *tile_total = sum;
    }
}



__global__ static void moe_build_expert_tiles_kernel(
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const uint32_t *tile_offsets,
        const uint32_t *counts,
        uint32_t block_m,
        uint32_t n_total_expert) {
    uint32_t e = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= n_total_expert) return;
    uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
    uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) {
        tile_experts[off + t] = e;
        tile_starts[off + t] = t * block_m;
    }
}



/* Decode-sized routed batches spend more host time launching metadata kernels
 * than doing the <= 96 pair / 128 expert setup. Build both tile lists in one
 * deterministic block; the expensive expert kernels remain unchanged. */
__global__ static void moe_prepare_sorted_tiles_small_kernel(
        uint32_t *counts,
        uint32_t *offsets,
        uint32_t *cursors,
        uint32_t *sorted_pairs,
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        uint32_t *tile16_offsets,
        uint32_t *tile16_total,
        uint32_t *tile16_experts,
        uint32_t *tile16_starts,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert,
        uint32_t block_m,
        bool build_tile16) {
    if (blockIdx.x != 0) return;
    const uint32_t tid = threadIdx.x;
    __shared__ uint32_t local_counts[128];
    __shared__ int32_t local_selected[96];

    for (uint32_t e = tid; e < n_total_expert; e += blockDim.x) {
        local_counts[e] = 0u;
    }
    for (uint32_t pair = tid; pair < pair_count; pair += blockDim.x) {
        local_selected[pair] = selected[pair];
    }
    __syncthreads();

    for (uint32_t pair = tid; pair < pair_count; pair += blockDim.x) {
        const int32_t expert_i = local_selected[pair];
        if (expert_i >= 0 && (uint32_t)expert_i < n_total_expert) {
            atomicAdd(local_counts + (uint32_t)expert_i, 1u);
        }
    }
    __syncthreads();

    for (uint32_t e = tid; e < n_total_expert; e += blockDim.x) {
        counts[e] = local_counts[e];
    }
    if (tid == 0u) {
        uint32_t pair_sum = 0u;
        uint32_t tile_sum = 0u;
        uint32_t tile16_sum = 0u;
        for (uint32_t e = 0; e < n_total_expert; e++) {
            const uint32_t count = local_counts[e];
            offsets[e] = pair_sum;
            pair_sum += count;
            cursors[e] = pair_sum;
            tile_offsets[e] = tile_sum;
            tile_sum += (count + block_m - 1u) / block_m;
            if (build_tile16) {
                tile16_offsets[e] = tile16_sum;
                tile16_sum += (count + 15u) / 16u;
            }
        }
        offsets[n_total_expert] = pair_sum;
        tile_offsets[n_total_expert] = tile_sum;
        *tile_total = tile_sum;
        if (build_tile16) {
            tile16_offsets[n_total_expert] = tile16_sum;
            *tile16_total = tile16_sum;
        }
    }
    __syncthreads();

    for (uint32_t pair = tid; pair < pair_count; pair += blockDim.x) {
        const int32_t expert_i = local_selected[pair];
        if (expert_i >= 0 && (uint32_t)expert_i < n_total_expert) {
            uint32_t rank = 0u;
            for (uint32_t prev = 0; prev < pair; prev++) {
                rank += local_selected[prev] == expert_i;
            }
            sorted_pairs[offsets[(uint32_t)expert_i] + rank] = pair;
        }
    }
    for (uint32_t e = tid; e < n_total_expert; e += blockDim.x) {
        const uint32_t count = local_counts[e];
        const uint32_t ntiles = (count + block_m - 1u) / block_m;
        const uint32_t tile_off = tile_offsets[e];
        for (uint32_t t = 0; t < ntiles; t++) {
            tile_experts[tile_off + t] = e;
            tile_starts[tile_off + t] = t * block_m;
        }
        if (build_tile16) {
            const uint32_t ntiles16 = (count + 15u) / 16u;
            const uint32_t tile16_off = tile16_offsets[e];
            for (uint32_t t = 0; t < ntiles16; t++) {
                tile16_experts[tile16_off + t] = e;
                tile16_starts[tile16_off + t] = t * 16u;
            }
        }
    }
}



__global__ static void moe_gate_up_mid_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}



__global__ static void moe_gate_up_mid_expert_tile4_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block4(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, gate);
        dev_dot_iq2_xxs_q8_K_block4(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}



__global__ static void moe_gate_up_mid_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                            s_iq2_grid, s_iq2_signs);
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                            s_iq2_grid, s_iq2_signs);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}



__global__ static void moe_gate_up_mid_expert_tile8_row2048_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}



template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}



__global__ static void moe_gate_up_mid_sorted_p2_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t pair_count,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= expert_mid_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}



__global__ static void moe_down_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}



__global__ static void moe_gate_up_mid_decode_q4K_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < MOE_DECODE_ROW_TILES; rr++) {
        uint32_t row = blockIdx.x * MOE_DECODE_ROWS_PER_BLOCK + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}



__global__ static void moe_gate_up_mid_decode_q4K_hwarp16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 16u) {
        gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
        up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
    }
    gate = half_warp_sum_f32(gate, lane);
    up = half_warp_sum_f32(up, lane);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        if (write_aux) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                       weights[(uint64_t)tok * n_expert + slot];
    }
}



__global__ static void moe_gate_up_mid_decode_q4K_hwarp16_row8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t group = threadIdx.x >> 4u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    if (group >= 8u) return;
    uint32_t row = blockIdx.x * 8u + group;
    if (row >= expert_mid_dim) return;
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 16u) {
        gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
        up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
    }
    gate = half_warp_sum_f32(gate, lane);
    up = half_warp_sum_f32(up, lane);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        if (write_aux) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                       weights[(uint64_t)tok * n_expert + slot];
    }
}



__global__ static void moe_gate_up_mid_decode_q4K_warp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
        up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        if (write_aux) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                       weights[(uint64_t)tok * n_expert + slot];
    }
}



__global__ static void moe_gate_up_mid_decode_q4K_warp32_noaux_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        /* Word-wise cooperative staging copy (same bytes, all lanes busy). */
        const uint32_t words = xq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
        uint32_t *dst = (uint32_t *)sxq;
        const uint32_t *srcw = (const uint32_t *)xqb;
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) dst[i] = srcw[i];
        __syncthreads();
        xqb = sxq;
    }
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate = 0.0f;
    float up = 0.0f;
    const bool vec_ok = ((((uintptr_t)gate_base | (uintptr_t)up_base |
                           gate_row_bytes | gate_expert_bytes) & 15u) == 0u);
    if (vec_ok) {
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            dev_dot_q4_K_q8_K_block_vec(gr + b, xqb + b, &gate);
            dev_dot_q4_K_q8_K_block_vec(ur + b, xqb + b, &up);
        }
    } else {
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
        }
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                       weights[(uint64_t)tok * n_expert + slot];
    }
}



__global__ static void moe_gate_up_mid_decode_q4K_owned_warp32_noaux_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t expert_base,
        uint32_t expert_count,
        float clamp) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t row = blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t expert = 0u;
    if (!moe_owned_local_expert(selected[pair], expert_base, expert_count,
                                &expert)) return;
    const cuda_block_q8_K *xqb = xq;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        const uint32_t words = xq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
        uint32_t *dst = (uint32_t *)sxq;
        const uint32_t *srcw = (const uint32_t *)xqb;
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) dst[i] = srcw[i];
        __syncthreads();
        xqb = sxq;
    }
    const bool vec_ok = ((((uintptr_t)gate_base | (uintptr_t)up_base |
                           gate_row_bytes | gate_expert_bytes) & 15u) == 0u);
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate = 0.0f;
    float up = 0.0f;
    if (vec_ok) {
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            dev_dot_q4_K_q8_K_block_vec(gr + b, xqb + b, &gate);
            dev_dot_q4_K_q8_K_block_vec(ur + b, xqb + b, &up);
        }
    } else {
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
        }
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair];
    }
}



__global__ static void moe_gate_up_mid_decode_q4K_warp32_noaux_sidecar_kernel(
        float *mid_out,
        float *amax_sidecar,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t row = blockIdx.x * 8u + warp;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ float tile_vals[8];
    __shared__ float tile_abs[8];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }

    float midv = 0.0f;
    const bool valid = row < expert_mid_dim;
    if (valid) {
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
        }
        gate = warp_sum_f32(gate);
        up = warp_sum_f32(up);
        if (lane == 0u) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            midv = (gate / (1.0f + expf(-gate))) * up *
                   weights[(uint64_t)tok * n_expert + slot];
            mid_out[off] = midv;
        }
    }
    if (lane == 0u) {
        tile_vals[warp] = midv;
        tile_abs[warp] = valid ? fabsf(midv) : 0.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        float best_abs = tile_abs[0];
        float best_val = tile_vals[0];
        #pragma unroll
        for (uint32_t i = 1u; i < 8u; i++) {
            if (tile_abs[i] > best_abs) {
                best_abs = tile_abs[i];
                best_val = tile_vals[i];
            }
        }
        const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
        const uint32_t qblock = blockIdx.x / 32u;
        const uint32_t tile = blockIdx.x & 31u;
        if (qblock < midq_blocks) {
            amax_sidecar[((uint64_t)pair * midq_blocks + qblock) * 32u + tile] = best_val;
        }
    }
}



__global__ static void moe_gate_up_mid_decode_q4K_warp32_row16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 16u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }
    const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
        up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        if (write_aux) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                       weights[(uint64_t)tok * n_expert + slot];
    }
}



__global__ static void moe_gate_up_midq_decode_q4K_qwarp32_kernel(
        float *mid_out,
        cuda_block_q8_K *midq,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t qblock = blockIdx.x;
    const uint32_t pair = blockIdx.y;
    const uint32_t tok = pair / n_expert;
    const uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ float vals[CUDA_QK_K];
    __shared__ float abs_part[CUDA_QK_K];
    __shared__ float val_part[CUDA_QK_K];
    __shared__ float iscale_s;

    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        __syncthreads();
        xqb = sxq;
    }

    const float w = weights[(uint64_t)tok * n_expert + slot];
    #pragma unroll
    for (uint32_t rr = 0; rr < 8u; rr++) {
        const uint32_t row_in_block = row_lane + rr * 32u;
        const uint32_t row = qblock * CUDA_QK_K + row_in_block;
        float midv = 0.0f;
        if (row < expert_mid_dim) {
            const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
            const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
            float gate = 0.0f;
            float up = 0.0f;
            for (uint32_t b = lane; b < xq_blocks; b += 8u) {
                gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
                up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
            }
            gate = quarter_warp_sum_f32(gate, lane);
            up = quarter_warp_sum_f32(up, lane);
            if (lane == 0u) {
                if (clamp > 1.0e-6f) {
                    if (gate > clamp) gate = clamp;
                    if (up > clamp) up = clamp;
                    if (up < -clamp) up = -clamp;
                }
                midv = (gate / (1.0f + expf(-gate))) * up * w;
                const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
                mid_out[off] = midv;
            }
        }
        if (lane == 0u) vals[row_in_block] = midv;
    }
    __syncthreads();

    cuda_block_q8_K *yb = midq + (uint64_t)pair * (expert_mid_dim / CUDA_QK_K) + qblock;
    const uint32_t tid = threadIdx.x;
    const float v = vals[tid];
    abs_part[tid] = fabsf(v);
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    const float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0u) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16u) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0u) {
        iscale_s = -127.0f / val_part[0];
    }
    __syncthreads();
    int qv = (int)lrintf(iscale_s * v);
    if (qv > 127) qv = 127;
    if (qv < -128) qv = -128;
    yb->qs[tid] = (int8_t)qv;
    __syncthreads();
    if (tid < CUDA_QK_K / 16u) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16u + (uint32_t)i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0u) yb->d = 1.0f / iscale_s;
}



template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_q4_K_q8_K_block8(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                      xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                      xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                      xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate);
            dev_dot_q4_K_q8_K_block8(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                      xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                      xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                      xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}



__global__ static void moe_down_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}



__global__ static void moe_down_owned_slots_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t slot = blockIdx.y;
    if (row >= out_dim || slot >= 6u) return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        return;
    }
    const cuda_block_q2_K *wr =
        (const cuda_block_q2_K *)(down_base +
            (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)slot * out_dim + row] = acc;
}




/* Map one of two packed operands for a three-slot reduction group. The only
 * multi-slot operand is the peer-owned prefix (slots 0+1 within the group),
 * which can be pre-added exactly because the reference reduction starts from
 * +0. Every other peer slot remains a distinct operand in original order. */
__device__ __forceinline__ static int moe_owned_packed_component(
        const int32_t *selected,
        uint32_t group,
        uint32_t component,
        uint32_t expert_base,
        uint32_t expert_count,
        bool *prefix_pair) {
    const uint32_t slot0 = group * 3u;
    uint32_t mask = 0u;
    #pragma unroll
    for (uint32_t i = 0; i < 3u; i++) {
        if (moe_owned_local_expert(selected[slot0 + i], expert_base,
                                   expert_count, NULL)) {
            mask |= 1u << i;
        }
    }
    *prefix_pair = false;
    if ((mask & 3u) == 3u) {
        if (component == 0u) {
            *prefix_pair = true;
            return (int)slot0;
        }
        return (mask & 4u) != 0u ? (int)(slot0 + 2u) : -1;
    }
    uint32_t ordinal = 0u;
    #pragma unroll
    for (uint32_t i = 0; i < 3u; i++) {
        if ((mask & (1u << i)) == 0u) continue;
        if (ordinal++ == component) return (int)(slot0 + i);
    }
    return -1;
}



/* One thread owns an output column and loads all source slots before writing.
 * This permits packed_out == slots for the non-peer-copy fallback. */
__global__ static void moe_down_owned_pack_f32_slots_kernel(
        float *packed_out,
        const float *slots,
        const int32_t *selected,
        uint32_t out_dim,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t col =
        (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (col >= out_dim) return;

    float slotv[6];
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        slotv[slot] = slots[(uint64_t)slot * out_dim + col];
    }

    float packed[4];
    #pragma unroll
    for (uint32_t packed_slot = 0; packed_slot < 4u; packed_slot++) {
        bool prefix_pair = false;
        const int first_slot = moe_owned_packed_component(
                selected, packed_slot / 2u, packed_slot & 1u,
                expert_base, expert_count, &prefix_pair);
        if (first_slot < 0) {
            packed[packed_slot] = 0.0f;
        } else if (prefix_pair) {
            float value = __fadd_rn(0.0f, slotv[first_slot]);
            packed[packed_slot] = __fadd_rn(value, slotv[first_slot + 1]);
        } else {
            packed[packed_slot] = slotv[first_slot];
        }
    }

    #pragma unroll
    for (uint32_t packed_slot = 0; packed_slot < 4u; packed_slot++) {
        packed_out[(uint64_t)packed_slot * out_dim + col] = packed[packed_slot];
    }
}



__global__ static void moe_down_owned_copy_f32_slots_kernel(
        float *dst,
        const float *src,
        uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = src[i];
}



__global__ static void moe_down_owned_packed_qwarp32_kernel(
        float *packed_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t packed_slot = blockIdx.y;
    if (row >= out_dim || packed_slot >= 4u) return;
    bool prefix_pair = false;
    const int first_slot = moe_owned_packed_component(
            selected, packed_slot / 2u, packed_slot & 1u,
            expert_base, expert_count, &prefix_pair);
    if (first_slot < 0) {
        if (lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = 0.0f;
        return;
    }

    float packed = 0.0f;
    const uint32_t n_slots = prefix_pair ? 2u : 1u;
    #pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
        if (i >= n_slots) break;
        const uint32_t slot = (uint32_t)first_slot + i;
        uint32_t expert = 0;
        if (!moe_owned_local_expert(selected[slot], expert_base,
                                    expert_count, &expert)) {
            continue;
        }
        const cuda_block_q2_K *wr =
            (const cuda_block_q2_K *)(down_base +
                (uint64_t)expert * down_expert_bytes +
                (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0u) {
            packed = prefix_pair ? __fadd_rn(packed, acc) : acc;
        }
    }
    if (lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = packed;
}



__global__ static void moe_down_sum3_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 3u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}



__global__ static void moe_down_q4K_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    const bool vec_ok = ((((uintptr_t)down_base | down_row_bytes | down_expert_bytes) & 15u) == 0u);
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        if (vec_ok) {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b, &acc);
        } else {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}



__global__ static void moe_down_q4K_owned_slots_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t slot = blockIdx.y;
    if (row >= out_dim || slot >= 6u) return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        return;
    }
    const cuda_block_q4_K *wr =
        (const cuda_block_q4_K *)(down_base +
            (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
    const bool vec_ok = ((((uintptr_t)down_base | down_row_bytes |
                           down_expert_bytes) & 15u) == 0u);
    float acc = 0.0f;
    if (vec_ok) {
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b, &acc);
        }
    } else {
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)slot * out_dim + row] = acc;
}



__global__ static void moe_down_q4K_owned_packed_qwarp32_kernel(
        float *packed_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t packed_slot = blockIdx.y;
    if (row >= out_dim || packed_slot >= 4u) return;
    bool prefix_pair = false;
    const int first_slot = moe_owned_packed_component(
            selected, packed_slot / 2u, packed_slot & 1u,
            expert_base, expert_count, &prefix_pair);
    if (first_slot < 0) {
        if (lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = 0.0f;
        return;
    }

    const bool vec_ok = ((((uintptr_t)down_base | down_row_bytes |
                           down_expert_bytes) & 15u) == 0u);
    float packed = 0.0f;
    const uint32_t n_slots = prefix_pair ? 2u : 1u;
    #pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
        if (i >= n_slots) break;
        const uint32_t slot = (uint32_t)first_slot + i;
        uint32_t expert = 0;
        if (!moe_owned_local_expert(selected[slot], expert_base,
                                    expert_count, &expert)) {
            continue;
        }
        const cuda_block_q4_K *wr =
            (const cuda_block_q4_K *)(down_base +
                (uint64_t)expert * down_expert_bytes +
                (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        if (vec_ok) {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b, &acc);
            }
        } else {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
            }
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0u) {
            packed = prefix_pair ? __fadd_rn(packed, acc) : acc;
        }
    }
    if (lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = packed;
}



__global__ static void moe_owned_slots_combine_fixed3_kernel(
        float *out,
        const float *home_slots,
        const float *peer_slots,
        const int32_t *selected,
        uint32_t out_dim,
        uint32_t expert_split) {
    const uint32_t col =
        (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t row = blockIdx.y;
    if (col >= out_dim) return;
    out += (uint64_t)row * out_dim;
    home_slots += (uint64_t)row * 6u * out_dim;
    peer_slots += (uint64_t)row * 6u * out_dim;
    selected += (uint64_t)row * 6u;
    float slotv[6];
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        const int32_t expert = selected[slot];
        if (expert < 0 ||
            (uint32_t)expert >= 2u * expert_split) {
            slotv[slot] = 0.0f;
        } else {
            const bool on_home = (uint32_t)expert < expert_split;
            const float *src = on_home ? home_slots : peer_slots;
            slotv[slot] = src[(uint64_t)slot * out_dim + col];
        }
    }
    float home = __fadd_rn(0.0f, slotv[0]);
    home = __fadd_rn(home, slotv[1]);
    home = __fadd_rn(home, slotv[2]);
    float peer = __fadd_rn(0.0f, slotv[3]);
    peer = __fadd_rn(peer, slotv[4]);
    peer = __fadd_rn(peer, slotv[5]);
    out[col] = __fadd_rn(home, peer);
}



__device__ static float moe_owned_packed_combine_row(
        const float *home_slots,
        const float *peer_packed,
        const int32_t *selected,
        uint32_t row,
        uint32_t out_dim,
        uint32_t expert_split) {
    float groups[2];
    #pragma unroll
    for (uint32_t group = 0; group < 2u; group++) {
        const uint32_t slot0 = group * 3u;
        uint32_t peer_mask = 0u;
        uint32_t valid_mask = 0u;
        #pragma unroll
        for (uint32_t i = 0; i < 3u; i++) {
            const int32_t expert = selected[slot0 + i];
            if (expert >= 0 && (uint32_t)expert < 2u * expert_split) {
                valid_mask |= 1u << i;
            }
            if (expert >= 0 && (uint32_t)expert >= expert_split &&
                (uint32_t)expert < 2u * expert_split) {
                peer_mask |= 1u << i;
            }
        }
        const float *packed = peer_packed +
            (uint64_t)group * 2u * out_dim + row;
        float acc;
        if ((peer_mask & 3u) == 3u) {
            /* packed[0] is already (+0 + slot0) + slot1. */
            acc = packed[0];
            float slot2 = 0.0f;
            if ((peer_mask & 4u) != 0u) {
                slot2 = packed[out_dim];
            } else if ((valid_mask & 4u) != 0u) {
                slot2 = home_slots[(uint64_t)(slot0 + 2u) * out_dim + row];
            }
            acc = __fadd_rn(acc, slot2);
        } else {
            acc = 0.0f;
            uint32_t peer_operand = 0u;
            #pragma unroll
            for (uint32_t i = 0; i < 3u; i++) {
                float value;
                if ((peer_mask & (1u << i)) != 0u) {
                    value = packed[(uint64_t)peer_operand * out_dim];
                    peer_operand++;
                } else if ((valid_mask & (1u << i)) != 0u) {
                    value = home_slots[(uint64_t)(slot0 + i) * out_dim + row];
                } else {
                    value = 0.0f;
                }
                acc = __fadd_rn(acc, value);
            }
        }
        groups[group] = acc;
    }
    return __fadd_rn(groups[0], groups[1]);
}



__global__ static void moe_owned_packed_combine_fixed3_kernel(
        float *out,
        const float *home_slots,
        const float *peer_packed,
        const int32_t *selected,
        uint32_t out_dim,
        uint32_t expert_split) {
    const uint32_t row =
        (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= out_dim) return;
    out[row] = moe_owned_packed_combine_row(
            home_slots, peer_packed, selected, row, out_dim, expert_split);
}



__global__ static void moe_down_q4K_sum3_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    const bool vec_ok = ((((uintptr_t)down_base | down_row_bytes | down_expert_bytes) & 15u) == 0u);
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 3u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        if (vec_ok) {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b, &acc);
        } else {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}



__global__ static void moe_down_q4K_sum3_slotwarp_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t slot = lane >> 3u;
    const uint32_t qlane = lane & 7u;
    const uint32_t row = blockIdx.x * 8u + warp;
    if (row >= out_dim) return;

    float acc = 0.0f;
    if (slot < 3u) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr =
            (const cuda_block_q4_K *)(down_base +
                                      (uint64_t)(uint32_t)expert_i * down_expert_bytes +
                                      (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        for (uint32_t b = qlane; b < midq_blocks; b += 8u) {
            acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, qlane);
    }

    const float s1 = __shfl_sync(0xffffffffu, acc, 8);
    const float s2 = __shfl_sync(0xffffffffu, acc, 16);
    if (lane == 0u) {
        const float s0 = acc;
        out[row] = (s0 + s1) + s2;
    }
}



/* Q4_K prefill (n_tokens > 1) down kernel. Mirrors moe_down_qwarp32_kernel
 * geometry exactly; only the weight block type and dot helper differ. The
 * pair = blockIdx.y indexing means the same grid shape (out_dim/32, n_tokens*n_expert)
 * used by the IQ2 path applies here. The downstream moe_sum_kernel is
 * weight-type-agnostic and sums these per-pair outputs into the final output. */
__global__ static void moe_down_q4K_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}



template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_expert_tile8_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q4_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                      xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                      xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                      xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
        }
    }
}




/* INT8 tensor-core (m8n8k16) exact MoE prefill tile kernels.
 *
 * Each warp computes an 8-token x 8-row tile. The Q4_K x Q8_K superblock dot
 * keeps its integer sums (order-invariant, exact) but computes the 32-wide
 * group dots on tensor cores; every output element keeps 8 float slot
 * accumulators (slot[b & 7] += term_b, b ascending) and reduces them with the
 * exact quarter_warp_sum_f32 grouping, so results are bit-identical to the
 * scalar expert-tile kernels (fuzz-verified). Requires sm_75+, 16B-aligned
 * expert tensors, and the staged activation-block counts (<=16 gate/up,
 * <=8 down). Rollback: DS4_CUDA_MOE_NO_Q4_MMA=1. */
__device__ __forceinline__ static void mma_m8n8k16_s8(int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {
#if __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
#else
    (void)a; (void)b; (void)c0; (void)c1;
#endif
}



template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q4K_tile8_mma_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint32_t s_pair[8];
    __shared__ uint32_t s_tok[8];
    __shared__ uint32_t s_slot[8];
    __shared__ uint32_t s_np;
    if (threadIdx.x == 0) {
        uint32_t np = 0;
        for (; np < 8u; np++) {
            uint32_t local_pair = local_start + np;
            if (local_pair >= counts[expert]) break;
            uint32_t pr = sorted_pairs[offsets[expert] + local_pair];
            s_pair[np] = pr;
            s_tok[np] = pr / n_expert;
            s_slot[np] = pr - s_tok[np] * n_expert;
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u); i += blockDim.x) {
            const uint32_t words_per_tok = xq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
            uint32_t p = i / words_per_tok;
            uint32_t w = i - p * words_per_tok;
            ((uint32_t *)sxq[p])[w] = ((const uint32_t *)(xq + (uint64_t)s_tok[p] * xq_blocks))[w];
        }
        __syncthreads();
    }
    const uint32_t mtok = lane >> 2u;      /* token row of this thread's C elems */
    const uint32_t n0 = (lane & 3u) * 2u;  /* first C column (weight row) */
    /* 8 warps x 8 rows = 64 rows per pass */
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN + rr * 64u + warp * 8u;
        if (row0 >= expert_mid_dim) continue;
        const char *grow = gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row0 * gate_row_bytes;
        const char *urow = up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row0 * gate_row_bytes;
        /* per-element slot accumulators (2 elements x 8 slots) */
        float sg0[8] = {0,0,0,0,0,0,0,0}, sg1[8] = {0,0,0,0,0,0,0,0};
        float su0[8] = {0,0,0,0,0,0,0,0}, su1[8] = {0,0,0,0,0,0,0,0};
        for (uint32_t b = 0; b < xq_blocks; b++) {
            /* headers for this thread's two C columns */
            const uint4 ghdr0 = *(const uint4 *)((const cuda_block_q4_K *)(grow + (uint64_t)n0 * gate_row_bytes) + b);
            const uint4 ghdr1 = *(const uint4 *)((const cuda_block_q4_K *)(grow + (uint64_t)(n0 + 1u) * gate_row_bytes) + b);
            const uint4 uhdr0 = *(const uint4 *)((const cuda_block_q4_K *)(urow + (uint64_t)n0 * gate_row_bytes) + b);
            const uint4 uhdr1 = *(const uint4 *)((const cuda_block_q4_K *)(urow + (uint64_t)(n0 + 1u) * gate_row_bytes) + b);
            /* B-fragment source rows for loads: n_load = lane>>2.
             * Batch all global loads for this superblock upfront so the
             * memory system sees independent requests instead of a
             * load->mma dependency chain. */
            const uint32_t *gqw = (const uint32_t *)(((const cuda_block_q4_K *)(grow + (uint64_t)(lane >> 2u) * gate_row_bytes) + b)->qs);
            const uint32_t *uqw = (const uint32_t *)(((const cuda_block_q4_K *)(urow + (uint64_t)(lane >> 2u) * gate_row_bytes) + b)->qs);
            const int8_t *aqs = sxq[mtok][b].qs;
            uint32_t gw8[8], uw8[8];
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) {
                gw8[k] = gqw[k * 4u + (lane & 3u)];
                uw8[k] = uqw[k * 4u + (lane & 3u)];
            }
            int gi0 = 0, gi1 = 0, ui0 = 0, ui1 = 0;
            int gs0 = 0, gs1 = 0, us0 = 0, us1 = 0;
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const int shift = (j & 1u) ? 4 : 0;
                /* dot32 via two chained k16 mmas, per matrix */
                int32_t gc0 = 0, gc1 = 0, uc0 = 0, uc1 = 0;
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    const uint32_t koff = h * 16u + (lane & 3u) * 4u;
                    const uint32_t a = *(const uint32_t *)(aqs + j * 32u + koff);
                    const uint32_t gw = (gw8[(j >> 1u) * 2u + h] >> shift) & 0x0f0f0f0fu;
                    const uint32_t uw = (uw8[(j >> 1u) * 2u + h] >> shift) & 0x0f0f0f0fu;
                    mma_m8n8k16_s8(gc0, gc1, a, gw);
                    mma_m8n8k16_s8(uc0, uc1, a, uw);
                }
                /* integer scale application for this thread's two columns */
                uint8_t sc, m;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&ghdr0.y, &sc, &m);
                gi0 += (int)sc * gc0;
                const int bs = (int)sxq[mtok][b].bsums[2u * j] + (int)sxq[mtok][b].bsums[2u * j + 1u];
                gs0 += (int)m * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&ghdr1.y, &sc, &m);
                gi1 += (int)sc * gc1;
                gs1 += (int)m * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&uhdr0.y, &sc, &m);
                ui0 += (int)sc * uc0;
                us0 += (int)m * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&uhdr1.y, &sc, &m);
                ui1 += (int)sc * uc1;
                us1 += (int)m * bs;
            }
            /* float finish, exact dev_dot_q4_K_q8_K_block8 expression */
            const float yd = sxq[mtok][b].d;
            const uint32_t sl = b & 7u;
            sg0[sl] += yd * dev_f16_to_f32((uint16_t)(ghdr0.x & 0xffffu)) * (float)gi0 -
                       yd * dev_f16_to_f32((uint16_t)(ghdr0.x >> 16u)) * (float)gs0;
            sg1[sl] += yd * dev_f16_to_f32((uint16_t)(ghdr1.x & 0xffffu)) * (float)gi1 -
                       yd * dev_f16_to_f32((uint16_t)(ghdr1.x >> 16u)) * (float)gs1;
            su0[sl] += yd * dev_f16_to_f32((uint16_t)(uhdr0.x & 0xffffu)) * (float)ui0 -
                       yd * dev_f16_to_f32((uint16_t)(uhdr0.x >> 16u)) * (float)us0;
            su1[sl] += yd * dev_f16_to_f32((uint16_t)(uhdr1.x & 0xffffu)) * (float)ui1 -
                       yd * dev_f16_to_f32((uint16_t)(uhdr1.x >> 16u)) * (float)us1;
        }
        /* quarter_warp_sum_f32 order: ((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7)) */
        const uint32_t p = mtok;
        if (p < np) {
            const uint32_t rowa = row0 + n0;
            const uint32_t rowb = row0 + n0 + 1u;
            float gate2[2], up2[2];
            {
                float a0 = sg0[0] + sg0[4], a1 = sg0[1] + sg0[5], a2 = sg0[2] + sg0[6], a3 = sg0[3] + sg0[7];
                gate2[0] = (a0 + a2) + (a1 + a3);
                a0 = sg1[0] + sg1[4]; a1 = sg1[1] + sg1[5]; a2 = sg1[2] + sg1[6]; a3 = sg1[3] + sg1[7];
                gate2[1] = (a0 + a2) + (a1 + a3);
                a0 = su0[0] + su0[4]; a1 = su0[1] + su0[5]; a2 = su0[2] + su0[6]; a3 = su0[3] + su0[7];
                up2[0] = (a0 + a2) + (a1 + a3);
                a0 = su1[0] + su1[4]; a1 = su1[1] + su1[5]; a2 = su1[2] + su1[6]; a3 = su1[3] + su1[7];
                up2[1] = (a0 + a2) + (a1 + a3);
            }
#pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                const uint32_t row = e ? rowb : rowa;
                if (row >= expert_mid_dim) continue;
                float gate = gate2[e];
                float up = up2[e];
                if (clamp > 1.0e-6f) {
                    if (gate > clamp) gate = clamp;
                    if (up > clamp) up = clamp;
                    if (up < -clamp) up = -clamp;
                }
                const uint64_t off = (uint64_t)s_pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate;
                    up_out[off] = up;
                }
                mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)s_tok[p] * n_expert + s_slot[p]];
            }
        }
    }
}



template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_tile8_mma_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    __shared__ uint32_t s_pair[8];
    __shared__ uint32_t s_np;
    if (threadIdx.x == 0) {
        uint32_t np = 0;
        for (; np < 8u; np++) {
            uint32_t local_pair = local_start + np;
            if (local_pair >= counts[expert]) break;
            s_pair[np] = sorted_pairs[offsets[expert] + local_pair];
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    if (midq_blocks <= 8u) {
        const uint32_t words_per_tok = midq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
        for (uint32_t i = threadIdx.x; i < np * words_per_tok; i += blockDim.x) {
            uint32_t p = i / words_per_tok;
            uint32_t w = i - p * words_per_tok;
            ((uint32_t *)sxq[p])[w] = ((const uint32_t *)(midq + (uint64_t)s_pair[p] * midq_blocks))[w];
        }
        __syncthreads();
    }
    const uint32_t mtok = lane >> 2u;
    const uint32_t n0 = (lane & 3u) * 2u;
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN + rr * 64u + warp * 8u;
        if (row0 >= out_dim) continue;
        const char *wrow = down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row0 * down_row_bytes;
        float s0[8] = {0,0,0,0,0,0,0,0}, s1[8] = {0,0,0,0,0,0,0,0};
        for (uint32_t b = 0; b < midq_blocks; b++) {
            const uint4 hdr0 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)n0 * down_row_bytes) + b);
            const uint4 hdr1 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)(n0 + 1u) * down_row_bytes) + b);
            const uint32_t *wqw = (const uint32_t *)(((const cuda_block_q4_K *)(wrow + (uint64_t)(lane >> 2u) * down_row_bytes) + b)->qs);
            const int8_t *aqs = sxq[mtok][b].qs;
            uint32_t w8[8];
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) w8[k] = wqw[k * 4u + (lane & 3u)];
            int i0 = 0, i1 = 0, m0 = 0, m1 = 0;
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const int shift = (j & 1u) ? 4 : 0;
                int32_t c0 = 0, c1 = 0;
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    const uint32_t koff = h * 16u + (lane & 3u) * 4u;
                    const uint32_t a = *(const uint32_t *)(aqs + j * 32u + koff);
                    const uint32_t w = (w8[(j >> 1u) * 2u + h] >> shift) & 0x0f0f0f0fu;
                    mma_m8n8k16_s8(c0, c1, a, w);
                }
                uint8_t sc, m;
                const int bs = (int)sxq[mtok][b].bsums[2u * j] + (int)sxq[mtok][b].bsums[2u * j + 1u];
                dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr0.y, &sc, &m);
                i0 += (int)sc * c0;
                m0 += (int)m * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr1.y, &sc, &m);
                i1 += (int)sc * c1;
                m1 += (int)m * bs;
            }
            const float yd = sxq[mtok][b].d;
            const uint32_t sl = b & 7u;
            s0[sl] += yd * dev_f16_to_f32((uint16_t)(hdr0.x & 0xffffu)) * (float)i0 -
                      yd * dev_f16_to_f32((uint16_t)(hdr0.x >> 16u)) * (float)m0;
            s1[sl] += yd * dev_f16_to_f32((uint16_t)(hdr1.x & 0xffffu)) * (float)i1 -
                      yd * dev_f16_to_f32((uint16_t)(hdr1.x >> 16u)) * (float)m1;
        }
        const uint32_t p = mtok;
        if (p < np) {
            float a0 = s0[0] + s0[4], a1 = s0[1] + s0[5], a2 = s0[2] + s0[6], a3 = s0[3] + s0[7];
            const float r0 = (a0 + a2) + (a1 + a3);
            a0 = s1[0] + s1[4]; a1 = s1[1] + s1[5]; a2 = s1[2] + s1[6]; a3 = s1[3] + s1[7];
            const float r1 = (a0 + a2) + (a1 + a3);
            if (row0 + n0 < out_dim) down_out[(uint64_t)s_pair[p] * out_dim + row0 + n0] = r0;
            if (row0 + n0 + 1u < out_dim) down_out[(uint64_t)s_pair[p] * out_dim + row0 + n0 + 1u] = r1;
        }
    }
}



/* 16-pair MoE expert tile kernels on sm_80+ m16n8k32 INT8 tensor cores.
 *
 * Same per-output math and reduction order as the 8-pair expert tile
 * kernels (slot[b & 7] += term_b with b ascending, then the exact
 * quarter_warp_sum_f32 grouping), so results are bit-identical; grouping
 * 16 pairs per tile just halves how often each expert's weights are
 * streamed from DRAM. Gate and up run as two passes over the superblocks
 * to keep register pressure at the 8-pair kernel's level. */

__device__ __forceinline__ static void mma16_m16n8k32_s8(
        int32_t &c0, int32_t &c1, int32_t &c2, int32_t &c3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
                 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#else
    (void)a0;(void)a1;(void)a2;(void)a3;(void)b0;(void)b1;(void)c0;(void)c1;(void)c2;(void)c3;
#endif
}

#endif  // DS4X_BACKEND_MOE_QUANTIZED_KERNELS_CUH
