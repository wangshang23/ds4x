#ifndef DS4X_BACKEND_STORAGE_KERNELS_CUH
#define DS4X_BACKEND_STORAGE_KERNELS_CUH

#include "../backend_common.cuh"



__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}



__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}



__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}



__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    out[i] = row[i % n_embd];
}



__global__ static void repeat_hc_rows_kernel(float *out, const float *rows, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (i >= n) return;

    uint64_t hc_row = (uint64_t)n_hc * n_embd;
    uint64_t tok = i / hc_row;
    uint64_t embd = i % n_embd;
    out[i] = rows[tok * n_embd + embd];
}



__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}



__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}



__device__ static float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}



__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}



__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}



__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}



__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}



__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a) {
    if (use_dp4a && n == 32u) return dot_i8x32_dp4a(a, b);
    int32_t dot = 0;
    for (uint64_t i = 0; i < n; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
}



__global__ static void quantize_q8_0_f32_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t in_dim,
        uint64_t blocks) {
    uint64_t b = blockIdx.x;
    uint64_t tok = blockIdx.y;
    if (b >= blocks) return;
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const float *xr = x + tok * in_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) vals[threadIdx.x] = fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[tok * blocks + b] = d;
    int8_t *dst = xq + (tok * blocks + b) * 32;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}



__global__ static void quantize_q8_0_group_slice_rows_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t group_dim,
        uint64_t blocks,
        uint32_t n_groups_total,
        uint32_t group0,
        uint32_t group_cnt) {
    const uint64_t b = blockIdx.x;
    const uint64_t packed_row = blockIdx.y;
    if (b >= blocks) return;
    const uint64_t token = packed_row / group_cnt;
    const uint64_t group = group0 + packed_row - token * group_cnt;
    const uint64_t i0 = b * 32u;
    const uint64_t bn = group_dim - i0 < 32u ? group_dim - i0 : 32u;
    const float *xr = x +
        (token * n_groups_total + group) * group_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            vals[threadIdx.x] =
                fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0u) xscale[packed_row * blocks + b] = d;
    int8_t *dst = xq + (packed_row * blocks + b) * 32u;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}



__global__ static void matmul_q8_0_preq_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}



__global__ static void matmul_q8_0_preq_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32u;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}



__global__ static void matmul_q8_0_pair_preq_warp8_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned char *wr0 = row < out0_dim ? w0 + row * blocks * 34 : NULL;
    const unsigned char *wr1 = row < out1_dim ? w1 + row * blocks * 34 : NULL;
    const int8_t *xqr = xq + tok * blocks * 32u;
    const float *xsr = xscale + tok * blocks;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const int8_t *xqb = xqr + b * 32;
        const float xs = xsr[b];
        if (wr0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc0 += __half2float(*scale_h) * xs * (float)dot;
        }
        if (wr1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc1 += __half2float(*scale_h) * xs * (float)dot;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        if (row < out0_dim) out0[tok * out0_dim + row] = acc0;
        if (row < out1_dim) out1[tok * out1_dim + row] = acc1;
    }
}






__global__ static void matmul_q8_0_pair_preq_batch_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint64_t tok = (uint64_t)blockIdx.y;
    if (tok >= n_tok) return;
    const int has0 = row < out0_dim;
    const int has1 = row < out1_dim;
    if (!has0 && !has1) return;

    const unsigned char *wr0 = has0 ? w0 + row * blocks * 34u : NULL;
    const unsigned char *wr1 = has1 ? w1 + row * blocks * 34u : NULL;
    const int8_t *xqr = xq + tok * blocks * 32u;
    const float *xsr = xscale + tok * blocks;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const int8_t *xqb = xqr + b * 32u;
        const float xs = xsr[b];
        if (has0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34u);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34u + 2u);
            const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc0 += __half2float(*scale_h) * xs * (float)dot;
        }
        if (has1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34u);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34u + 2u);
            const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc1 += __half2float(*scale_h) * xs * (float)dot;
        }
    }

    __shared__ float partial0[256];
    __shared__ float partial1[256];
    partial0[threadIdx.x] = acc0;
    partial1[threadIdx.x] = acc1;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial0[threadIdx.x] += partial0[threadIdx.x + stride];
            partial1[threadIdx.x] += partial1[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (has0) out0[tok * out0_dim + row] = partial0[0];
        if (has1) out1[tok * out1_dim + row] = partial1[0];
    }
}



__global__ static void matmul_q8_0_pair_preq_batch_tok2_exact_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint64_t tok0 = (uint64_t)blockIdx.y * 2u;
    if (tok0 >= n_tok) return;
    const int has0 = row < out0_dim;
    const int has1 = row < out1_dim;
    if (!has0 && !has1) return;
    const int valid1 = tok0 + 1u < n_tok;

    const unsigned char *wr0 = has0 ? w0 + row * blocks * 34u : NULL;
    const unsigned char *wr1 = has1 ? w1 + row * blocks * 34u : NULL;
    const int8_t *xqr0 = xq + tok0 * blocks * 32u;
    const int8_t *xqr1 = valid1 ? xqr0 + blocks * 32u : xqr0;
    const float *xsr0 = xscale + tok0 * blocks;
    const float *xsr1 = valid1 ? xsr0 + blocks : xsr0;
    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const int8_t *xqb0 = xqr0 + b * 32u;
        const int8_t *xqb1 = xqr1 + b * 32u;
        const float xs0 = xsr0[b];
        const float xs1 = valid1 ? xsr1[b] : 0.0f;
        if (has0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34u);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34u + 2u);
            const int dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            int dot1 = 0;
            if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
            const float ws = __half2float(*scale_h);
            acc00 += ws * xs0 * (float)dot0;
            if (valid1) acc01 += ws * xs1 * (float)dot1;
        }
        if (has1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34u);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34u + 2u);
            const int dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            int dot1 = 0;
            if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
            const float ws = __half2float(*scale_h);
            acc10 += ws * xs0 * (float)dot0;
            if (valid1) acc11 += ws * xs1 * (float)dot1;
        }
    }

    __shared__ float partial00[256];
    __shared__ float partial01[256];
    __shared__ float partial10[256];
    __shared__ float partial11[256];
    partial00[threadIdx.x] = acc00;
    partial01[threadIdx.x] = acc01;
    partial10[threadIdx.x] = acc10;
    partial11[threadIdx.x] = acc11;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial00[threadIdx.x] += partial00[threadIdx.x + stride];
            partial01[threadIdx.x] += partial01[threadIdx.x + stride];
            partial10[threadIdx.x] += partial10[threadIdx.x + stride];
            partial11[threadIdx.x] += partial11[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (has0) {
            out0[tok0 * out0_dim + row] = partial00[0];
            if (valid1) out0[(tok0 + 1u) * out0_dim + row] = partial01[0];
        }
        if (has1) {
            out1[tok0 * out1_dim + row] = partial10[0];
            if (valid1) out1[(tok0 + 1u) * out1_dim + row] = partial11[0];
        }
    }
}



__global__ static void matmul_q8_0_hc_expand_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *block_add2,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add,
        int has_add2,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) {
            float add_v = block_add[d];
            if (has_add2) add_v += block_add2[d];
            block_v += add_v;
        }
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}



__global__ static void matmul_q8_0_kslice_hc_expand_add_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t slice_dim,
        uint64_t out_dim,
        uint64_t full_blocks,
        uint64_t block_start,
        uint64_t slice_blocks,
        uint32_t n_embd,
        uint32_t n_hc,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * full_blocks * 34u + block_start * 34u;
    float acc = 0.0f;
    for (uint64_t b = lane; b < slice_blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = slice_dim - i0 < 32u ? slice_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb = xq + b * 32u;
        const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        const float block_v = acc + block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}



__global__ static void matmul_q8_0_preq_batch_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}



__global__ static void matmul_q8_0_preq_batch_warp8_tok2_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const unsigned char *wr = w + row * blocks * 34u;
    const int8_t *xqr0 = xq;
    const int8_t *xqr1 = xq + blocks * 32u;
    const float *xsr0 = xscale;
    const float *xsr1 = xscale + blocks;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb0 = xqr0 + b * 32u;
        const int8_t *xqb1 = xqr1 + b * 32u;
        int dot0 = 0;
        int dot1 = 0;
        if (use_dp4a && bn == 32u) {
#pragma unroll
            for (uint32_t i = 0; i < 32u; i += 4u) {
                const int32_t w4 = load_i8x4_i32_unaligned(qs + i);
                dot0 = __dp4a(w4, load_i8x4_i32_aligned(xqb0 + i), dot0);
                dot1 = __dp4a(w4, load_i8x4_i32_aligned(xqb1 + i), dot1);
            }
        } else {
            dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
        }
        const float ws = __half2float(*scale_h);
        acc0 += ws * xsr0[b] * (float)dot0;
        acc1 += ws * xsr1[b] * (float)dot1;
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        out[row] = acc0;
        out[out_dim + row] = acc1;
    }
}



__global__ static void matmul_q8_0_preq_batch_warp8_tok4_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok0 = (uint64_t)blockIdx.y * 4u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok0 >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr0 = xq + tok0 * blocks * 32;
    const int8_t *xqr1 = xqr0 + blocks * 32;
    const int8_t *xqr2 = xqr1 + blocks * 32;
    const int8_t *xqr3 = xqr2 + blocks * 32;
    const float *xsr0 = xscale + tok0 * blocks;
    const float *xsr1 = xsr0 + blocks;
    const float *xsr2 = xsr1 + blocks;
    const float *xsr3 = xsr2 + blocks;
    const int valid1 = tok0 + 1u < n_tok;
    const int valid2 = tok0 + 2u < n_tok;
    const int valid3 = tok0 + 3u < n_tok;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb0 = xqr0 + b * 32;
        const int8_t *xqb1 = xqr1 + b * 32;
        const int8_t *xqb2 = xqr2 + b * 32;
        const int8_t *xqb3 = xqr3 + b * 32;
        int dot0 = 0;
        int dot1 = 0;
        int dot2 = 0;
        int dot3 = 0;
        if (use_dp4a && bn == 32u) {
#pragma unroll
            for (uint32_t i = 0; i < 32u; i += 4u) {
                const int32_t w4 = load_i8x4_i32_unaligned(qs + i);
                dot0 = __dp4a(w4, load_i8x4_i32_aligned(xqb0 + i), dot0);
                if (valid1) dot1 = __dp4a(w4, load_i8x4_i32_aligned(xqb1 + i), dot1);
                if (valid2) dot2 = __dp4a(w4, load_i8x4_i32_aligned(xqb2 + i), dot2);
                if (valid3) dot3 = __dp4a(w4, load_i8x4_i32_aligned(xqb3 + i), dot3);
            }
        } else {
            dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
            if (valid2) dot2 = dot_i8_block(qs, xqb2, bn, use_dp4a);
            if (valid3) dot3 = dot_i8_block(qs, xqb3, bn, use_dp4a);
        }
        const float ws = __half2float(*scale_h);
        acc0 += ws * xsr0[b] * (float)dot0;
        if (valid1) acc1 += ws * xsr1[b] * (float)dot1;
        if (valid2) acc2 += ws * xsr2[b] * (float)dot2;
        if (valid3) acc3 += ws * xsr3[b] * (float)dot3;
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    acc2 = warp_sum_f32(acc2);
    acc3 = warp_sum_f32(acc3);
    if (lane == 0) {
        out[tok0 * out_dim + row] = acc0;
        if (valid1) out[(tok0 + 1u) * out_dim + row] = acc1;
        if (valid2) out[(tok0 + 2u) * out_dim + row] = acc2;
        if (valid3) out[(tok0 + 3u) * out_dim + row] = acc3;
    }
}



__global__ static void matmul_q8_0_preq_batch_warp8_tok8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok0 = (uint64_t)blockIdx.y * 8u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok0 >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const uint64_t xq_stride = blocks * 32u;
    const int8_t *xqr0 = xq + tok0 * xq_stride;
    const int valid1 = tok0 + 1u < n_tok;
    const int valid2 = tok0 + 2u < n_tok;
    const int valid3 = tok0 + 3u < n_tok;
    const int valid4 = tok0 + 4u < n_tok;
    const int valid5 = tok0 + 5u < n_tok;
    const int valid6 = tok0 + 6u < n_tok;
    const int valid7 = tok0 + 7u < n_tok;
    const int8_t *xqr1 = valid1 ? xqr0 + xq_stride : xqr0;
    const int8_t *xqr2 = valid2 ? xqr1 + xq_stride : xqr0;
    const int8_t *xqr3 = valid3 ? xqr2 + xq_stride : xqr0;
    const int8_t *xqr4 = valid4 ? xqr3 + xq_stride : xqr0;
    const int8_t *xqr5 = valid5 ? xqr4 + xq_stride : xqr0;
    const int8_t *xqr6 = valid6 ? xqr5 + xq_stride : xqr0;
    const int8_t *xqr7 = valid7 ? xqr6 + xq_stride : xqr0;
    const float *xsr0 = xscale + tok0 * blocks;
    const float *xsr1 = valid1 ? xsr0 + blocks : xsr0;
    const float *xsr2 = valid2 ? xsr1 + blocks : xsr0;
    const float *xsr3 = valid3 ? xsr2 + blocks : xsr0;
    const float *xsr4 = valid4 ? xsr3 + blocks : xsr0;
    const float *xsr5 = valid5 ? xsr4 + blocks : xsr0;
    const float *xsr6 = valid6 ? xsr5 + blocks : xsr0;
    const float *xsr7 = valid7 ? xsr6 + blocks : xsr0;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb0 = xqr0 + b * 32;
        const int8_t *xqb1 = xqr1 + b * 32;
        const int8_t *xqb2 = xqr2 + b * 32;
        const int8_t *xqb3 = xqr3 + b * 32;
        const int8_t *xqb4 = xqr4 + b * 32;
        const int8_t *xqb5 = xqr5 + b * 32;
        const int8_t *xqb6 = xqr6 + b * 32;
        const int8_t *xqb7 = xqr7 + b * 32;
        int dot0 = 0;
        int dot1 = 0;
        int dot2 = 0;
        int dot3 = 0;
        int dot4 = 0;
        int dot5 = 0;
        int dot6 = 0;
        int dot7 = 0;
        if (use_dp4a && bn == 32u) {
#pragma unroll
            for (uint32_t i = 0; i < 32u; i += 4u) {
                const int32_t w4 = load_i8x4_i32_unaligned(qs + i);
                dot0 = __dp4a(w4, load_i8x4_i32_aligned(xqb0 + i), dot0);
                if (valid1) dot1 = __dp4a(w4, load_i8x4_i32_aligned(xqb1 + i), dot1);
                if (valid2) dot2 = __dp4a(w4, load_i8x4_i32_aligned(xqb2 + i), dot2);
                if (valid3) dot3 = __dp4a(w4, load_i8x4_i32_aligned(xqb3 + i), dot3);
                if (valid4) dot4 = __dp4a(w4, load_i8x4_i32_aligned(xqb4 + i), dot4);
                if (valid5) dot5 = __dp4a(w4, load_i8x4_i32_aligned(xqb5 + i), dot5);
                if (valid6) dot6 = __dp4a(w4, load_i8x4_i32_aligned(xqb6 + i), dot6);
                if (valid7) dot7 = __dp4a(w4, load_i8x4_i32_aligned(xqb7 + i), dot7);
            }
        } else {
            dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
            if (valid2) dot2 = dot_i8_block(qs, xqb2, bn, use_dp4a);
            if (valid3) dot3 = dot_i8_block(qs, xqb3, bn, use_dp4a);
            if (valid4) dot4 = dot_i8_block(qs, xqb4, bn, use_dp4a);
            if (valid5) dot5 = dot_i8_block(qs, xqb5, bn, use_dp4a);
            if (valid6) dot6 = dot_i8_block(qs, xqb6, bn, use_dp4a);
            if (valid7) dot7 = dot_i8_block(qs, xqb7, bn, use_dp4a);
        }
        const float ws = __half2float(*scale_h);
        acc0 += ws * xsr0[b] * (float)dot0;
        if (valid1) acc1 += ws * xsr1[b] * (float)dot1;
        if (valid2) acc2 += ws * xsr2[b] * (float)dot2;
        if (valid3) acc3 += ws * xsr3[b] * (float)dot3;
        if (valid4) acc4 += ws * xsr4[b] * (float)dot4;
        if (valid5) acc5 += ws * xsr5[b] * (float)dot5;
        if (valid6) acc6 += ws * xsr6[b] * (float)dot6;
        if (valid7) acc7 += ws * xsr7[b] * (float)dot7;
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    acc2 = warp_sum_f32(acc2);
    acc3 = warp_sum_f32(acc3);
    acc4 = warp_sum_f32(acc4);
    acc5 = warp_sum_f32(acc5);
    acc6 = warp_sum_f32(acc6);
    acc7 = warp_sum_f32(acc7);
    if (lane == 0) {
        out[tok0 * out_dim + row] = acc0;
        if (valid1) out[(tok0 + 1u) * out_dim + row] = acc1;
        if (valid2) out[(tok0 + 2u) * out_dim + row] = acc2;
        if (valid3) out[(tok0 + 3u) * out_dim + row] = acc3;
        if (valid4) out[(tok0 + 4u) * out_dim + row] = acc4;
        if (valid5) out[(tok0 + 5u) * out_dim + row] = acc5;
        if (valid6) out[(tok0 + 6u) * out_dim + row] = acc6;
        if (valid7) out[(tok0 + 7u) * out_dim + row] = acc7;
    }
}



__global__ static void matmul_q8_0_preq_batch_tok2_exact_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint64_t tok0 = (uint64_t)blockIdx.y * 2u;
    if (row >= out_dim || tok0 >= n_tok) return;
    const int valid1 = tok0 + 1u < n_tok;
    const unsigned char *wr = w + row * blocks * 34u;
    const int8_t *xqr0 = xq + tok0 * blocks * 32u;
    const int8_t *xqr1 = valid1 ? xqr0 + blocks * 32u : xqr0;
    const float *xsr0 = xscale + tok0 * blocks;
    const float *xsr1 = valid1 ? xsr0 + blocks : xsr0;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb0 = xqr0 + b * 32u;
        const int8_t *xqb1 = xqr1 + b * 32u;
        const int dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
        int dot1 = 0;
        if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
        const float ws = __half2float(*scale_h);
        acc0 += ws * xsr0[b] * (float)dot0;
        if (valid1) acc1 += ws * xsr1[b] * (float)dot1;
    }

    __shared__ float partial0[256];
    __shared__ float partial1[256];
    partial0[threadIdx.x] = acc0;
    partial1[threadIdx.x] = acc1;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial0[threadIdx.x] += partial0[threadIdx.x + stride];
            partial1[threadIdx.x] += partial1[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[tok0 * out_dim + row] = partial0[0];
        if (valid1) out[(tok0 + 1u) * out_dim + row] = partial1[0];
    }
}




/* ---- INT8 tensor-core exact Q8_0 batch matmul --------------------------
 * Bit-identical replacement for the exact tok2/warp8-family batched Q8_0
 * kernels. Each output element's reduction is the reference's strided
 * halving tree over T slots (T = reduction width: 32 for the warp kernels,
 * cuda_q8_exact_threads(blocks) for the exact kernels; slots >= blocks hold
 * +0.0f). The kernel decomposes that tree as: 32 streams at stride T/32
 * whose 32 terms per outer step j combine via an adjacent-pairwise static
 * register stack taken in bit-reversed stream order (== the top five strided
 * tree levels), plus per-(j&3) sequential accumulators and a fixed tail for
 * the remaining levels. Fuzz-verified bitwise against both reference
 * kernels across shapes, including blocks < T and ragged out_dim/n_tok.
 * Rollback: DS4_CUDA_NO_Q8_MMA=1. */
__device__ __forceinline__ static uint32_t ldu32_unaligned(const uint8_t *p) {
    const uintptr_t addr = (uintptr_t)p;
    const uint32_t *base = (const uint32_t *)(addr & ~(uintptr_t)3);
    const uint32_t lo = base[0];
    const uint32_t hi = base[1];
    return __funnelshift_r(lo, hi, (uint32_t)(addr & 3u) * 8u);
}



__device__ __forceinline__ static void mma_m16n8k32_s8(
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



__device__ __forceinline__ static uint32_t bitrev5(uint32_t i) {
    return ((i & 1u) << 4) | ((i & 2u) << 2) | (i & 4u) | ((i & 8u) >> 2) | ((i & 16u) >> 4);
}



template <uint32_t T>
__global__ static void matmul_q8_0_mma_exact_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        uint64_t a_stride_blocks, /* activation row stride in blocks (>= blocks) */
        uint64_t out_stride) {    /* output token stride in floats (>= out_dim) */
    extern __shared__ unsigned char q8mma_sh[];
    __half *sh_ws = (__half *)q8mma_sh;                    /* 64 rows x blocks */
    float *sh_xs = (float *)(q8mma_sh + 64u * blocks * 2u); /* 16 toks x blocks */
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint64_t row_base = (uint64_t)blockIdx.x * 64u;
    const uint64_t tok_base = (uint64_t)blockIdx.y * 16u;

    /* stage weight scales (64 rows) and activation scales (16 tokens) */
    for (uint32_t idx = threadIdx.x; idx < 64u * (uint32_t)blocks; idx += blockDim.x) {
        const uint32_t rl = idx / (uint32_t)blocks;
        const uint32_t b = idx - rl * (uint32_t)blocks;
        uint64_t row = row_base + rl;
        if (row >= out_dim) row = out_dim - 1u;
        sh_ws[idx] = *(const __half *)(w + row * blocks * 34u + (uint64_t)b * 34u);
    }
    for (uint32_t idx = threadIdx.x; idx < 16u * (uint32_t)blocks; idx += blockDim.x) {
        const uint32_t tl = idx / (uint32_t)blocks;
        const uint32_t b = idx - tl * (uint32_t)blocks;
        const uint64_t tok = tok_base + tl;
        sh_xs[idx] = tok < n_tok ? xscale[tok * a_stride_blocks + b] : 0.0f;
    }
    __syncthreads();

    const uint64_t row0 = row_base + (uint64_t)warp * 8u;
    /* thread's C elements: rows n0,n0+1; tokens mt0, mt0+8 */
    const uint32_t n0 = (lane & 3u) * 2u;
    const uint32_t mt0 = lane >> 2u;
    const uint64_t tokA = tok_base + mt0;
    const uint64_t tokB = tok_base + mt0 + 8u;
    /* A source rows for loads (fragment layout): rows lane>>2 and (lane>>2)+8 */
    const uint64_t a_tok_lo = tok_base + (lane >> 2u);
    const uint64_t a_tok_hi = a_tok_lo + 8u;
    const int8_t *aq_lo = xq + (a_tok_lo < n_tok ? a_tok_lo : 0u) * a_stride_blocks * 32u;
    const int8_t *aq_hi = xq + (a_tok_hi < n_tok ? a_tok_hi : 0u) * a_stride_blocks * 32u;
    const bool a_lo_ok = a_tok_lo < n_tok;
    const bool a_hi_ok = a_tok_hi < n_tok;
    /* B source row for loads: row lane>>2 within the warp tile */
    uint64_t b_row = row0 + (lane >> 2u);
    if (b_row >= out_dim) b_row = out_dim - 1u;
    const unsigned char *b_wr = w + b_row * blocks * 34u;

    /* per-element (4) x per-(j&3) accumulators */
    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    const uint32_t stride = T / 32u;
    const uint32_t rl_ws0 = warp * 8u + n0;      /* local row for ws of element cols */
    const uint32_t tl_xsA = mt0;                 /* local token rows for xs */
    const uint32_t tl_xsB = mt0 + 8u;

    for (uint32_t j = 0; j < stride; j++) {
        /* adjacent-pairwise static stack over 32 terms in bitrev5 m order */
        float s0e0 = 0, s1e0 = 0, s2e0 = 0, s3e0 = 0, s4e0 = 0;
        float s0e1 = 0, s1e1 = 0, s2e1 = 0, s3e1 = 0, s4e1 = 0;
        float s0e2 = 0, s1e2 = 0, s2e2 = 0, s3e2 = 0, s4e2 = 0;
        float s0e3 = 0, s1e3 = 0, s2e3 = 0, s3e3 = 0, s4e3 = 0;
#pragma unroll
        for (uint32_t i = 0; i < 32u; i++) {
            const uint32_t m = bitrev5(i);
            const uint32_t s = j + m * stride; /* slot index */
            float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
            /* slot s sums blocks {s + k*T} sequentially (multi-term when
             * blocks > T, exactly like the per-lane strided walk). */
            for (uint32_t b = s; b < blocks; b += T) {
                const uint32_t koff = (lane & 3u) * 4u;
                const int8_t *ablk_lo = aq_lo + b * 32u;
                const int8_t *ablk_hi = aq_hi + b * 32u;
                const uint32_t a0 = a_lo_ok ? *(const uint32_t *)(ablk_lo + koff) : 0u;
                const uint32_t a1 = a_hi_ok ? *(const uint32_t *)(ablk_hi + koff) : 0u;
                const uint32_t a2 = a_lo_ok ? *(const uint32_t *)(ablk_lo + 16u + koff) : 0u;
                const uint32_t a3 = a_hi_ok ? *(const uint32_t *)(ablk_hi + 16u + koff) : 0u;
                const uint8_t *bq = (const uint8_t *)(b_wr + (uint64_t)b * 34u + 2u);
                const uint32_t b0 = ldu32_unaligned(bq + koff);
                const uint32_t b1 = ldu32_unaligned(bq + 16u + koff);
                int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                mma_m16n8k32_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
                /* term = ws * xs * dot, same expression as reference */
                const float ws0 = __half2float(sh_ws[rl_ws0 * (uint32_t)blocks + b]);
                const float ws1 = __half2float(sh_ws[(rl_ws0 + 1u) * (uint32_t)blocks + b]);
                const float xsA = sh_xs[tl_xsA * (uint32_t)blocks + b];
                const float xsB = sh_xs[tl_xsB * (uint32_t)blocks + b];
                t0 += ws0 * xsA * (float)c0;
                t1 += ws1 * xsA * (float)c1;
                t2 += ws0 * xsB * (float)c2;
                t3 += ws1 * xsB * (float)c3;
            }
            /* static adjacent stack push (compile-time resolved) */
            if ((i & 1u) == 0u) { s0e0 = t0; s0e1 = t1; s0e2 = t2; s0e3 = t3; }
            else {
                t0 = s0e0 + t0; t1 = s0e1 + t1; t2 = s0e2 + t2; t3 = s0e3 + t3;
                if ((i & 2u) == 0u) { s1e0 = t0; s1e1 = t1; s1e2 = t2; s1e3 = t3; }
                else {
                    t0 = s1e0 + t0; t1 = s1e1 + t1; t2 = s1e2 + t2; t3 = s1e3 + t3;
                    if ((i & 4u) == 0u) { s2e0 = t0; s2e1 = t1; s2e2 = t2; s2e3 = t3; }
                    else {
                        t0 = s2e0 + t0; t1 = s2e1 + t1; t2 = s2e2 + t2; t3 = s2e3 + t3;
                        if ((i & 8u) == 0u) { s3e0 = t0; s3e1 = t1; s3e2 = t2; s3e3 = t3; }
                        else {
                            t0 = s3e0 + t0; t1 = s3e1 + t1; t2 = s3e2 + t2; t3 = s3e3 + t3;
                            if ((i & 16u) == 0u) { s4e0 = t0; s4e1 = t1; s4e2 = t2; s4e3 = t3; }
                            else {
                                t0 = s4e0 + t0; t1 = s4e1 + t1; t2 = s4e2 + t2; t3 = s4e3 + t3;
                                /* i == 31: t is the finished x_j */
                                switch (j & 3u) {
                                case 0u: acc00 += t0; acc10 += t1; acc20 += t2; acc30 += t3; break;
                                case 1u: acc01 += t0; acc11 += t1; acc21 += t2; acc31 += t3; break;
                                case 2u: acc02 += t0; acc12 += t1; acc22 += t2; acc32 += t3; break;
                                default: acc03 += t0; acc13 += t1; acc23 += t2; acc33 += t3; break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    /* tail combine per T */
    float r0, r1, r2, r3;
    if (T == 32u) {
        r0 = acc00; r1 = acc10; r2 = acc20; r3 = acc30;
    } else if (T == 64u) {
        r0 = acc00 + acc01; r1 = acc10 + acc11; r2 = acc20 + acc21; r3 = acc30 + acc31;
    } else {
        r0 = (acc00 + acc02) + (acc01 + acc03);
        r1 = (acc10 + acc12) + (acc11 + acc13);
        r2 = (acc20 + acc22) + (acc21 + acc23);
        r3 = (acc30 + acc32) + (acc31 + acc33);
    }
    /* writes */
    const uint64_t rowa = row0 + n0;
    const uint64_t rowb = rowa + 1u;
    if (tokA < n_tok) {
        if (rowa < out_dim) out[tokA * out_stride + rowa] = r0;
        if (rowb < out_dim) out[tokA * out_stride + rowb] = r1;
    }
    if (tokB < n_tok) {
        if (rowa < out_dim) out[tokB * out_stride + rowa] = r2;
        if (rowb < out_dim) out[tokB * out_stride + rowb] = r3;
    }
}




__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = __hmul(scale, __float2half((float)q));
}



__global__ static void grouped_q8_0_a_preq_warp8_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;

    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * blocks * 32;
    const float *xsr = xscale + xrow * blocks;
    float acc = 0.0f;

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}



__global__ static void grouped_q8_0_a_preq_warp8_tok2_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a) {
    const uint32_t tid_in_tok = threadIdx.x & 255u;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (tid_in_tok >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y * 2u + (threadIdx.x >> 8u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;

    float acc = 0.0f;
    if (row < low_dim && tok < n_tokens) {
        const uint64_t group = row / rank;
        const uint64_t row_in_group = row - group * rank;
        const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34u;
        const uint64_t xrow = tok * (uint64_t)n_groups + group;
        const int8_t *xqr = xq + xrow * blocks * 32u;
        const float *xsr = xscale + xrow * blocks;

        for (uint64_t b = lane; b < blocks; b += 32u) {
            const uint64_t i0 = b * 32u;
            const uint64_t bn = group_dim - i0 < 32u ? group_dim - i0 : 32u;
            const __half *scale_h = (const __half *)(wr + b * 34u);
            const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
            const int8_t *xqb = xqr + b * 32u;
            const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc += __half2float(*scale_h) * xsr[b] * (float)dot;
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0 && row < low_dim && tok < n_tokens) {
        low[tok * low_dim + row] = acc;
    }
}



__global__ static void rms_norm_plain_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale;
    }
}

#endif  // DS4X_BACKEND_STORAGE_KERNELS_CUH
