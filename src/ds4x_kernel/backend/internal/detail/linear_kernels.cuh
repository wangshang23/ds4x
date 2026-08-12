#ifndef DS4X_BACKEND_LINEAR_KERNELS_CUH
#define DS4X_BACKEND_LINEAR_KERNELS_CUH

#include "../backend_common.cuh"

/* Latency-optimized RMS norm for the common n==4096 decode shape: one global
 * read pass with register-batched loads, same per-thread accumulation order
 * and shared-memory tree as rms_norm_plain_kernel (bit-identical, fuzz
 * checked). */
__global__ static void rms_norm_plain_fast4096_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float v[16];
#pragma unroll
    for (uint32_t j = 0; j < 16u; j++) v[j] = xr[threadIdx.x + j * 256u];
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < 16u; j++) sum += v[j] * v[j];
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
#pragma unroll
    for (uint32_t j = 0; j < 16u; j++) orow[threadIdx.x + j * 256u] = v[j] * scale;
}



/* Batched-load RMS norm for larger rows (n multiple of 2048, e.g. the 16384
 * HC-concatenated decode rows). Two passes like the reference kernel, but
 * eight independent loads are issued per accumulation group; the per-thread
 * accumulation order (ascending i with stride 256) is unchanged, so results
 * are bit-identical. */
__global__ static void rms_norm_plain_batch8_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
#pragma unroll 1
    for (uint32_t i = threadIdx.x; i < n; i += 2048u) {
        const float v0 = xr[i];
        const float v1 = xr[i + 256u];
        const float v2 = xr[i + 512u];
        const float v3 = xr[i + 768u];
        const float v4 = xr[i + 1024u];
        const float v5 = xr[i + 1280u];
        const float v6 = xr[i + 1536u];
        const float v7 = xr[i + 1792u];
        sum += v0 * v0;
        sum += v1 * v1;
        sum += v2 * v2;
        sum += v3 * v3;
        sum += v4 * v4;
        sum += v5 * v5;
        sum += v6 * v6;
        sum += v7 * v7;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
#pragma unroll 1
    for (uint32_t i = threadIdx.x; i < n; i += 2048u) {
        const float v0 = xr[i];
        const float v1 = xr[i + 256u];
        const float v2 = xr[i + 512u];
        const float v3 = xr[i + 768u];
        const float v4 = xr[i + 1024u];
        const float v5 = xr[i + 1280u];
        const float v6 = xr[i + 1536u];
        const float v7 = xr[i + 1792u];
        orow[i] = v0 * scale;
        orow[i + 256u] = v1 * scale;
        orow[i + 512u] = v2 * scale;
        orow[i + 768u] = v3 * scale;
        orow[i + 1024u] = v4 * scale;
        orow[i + 1280u] = v5 * scale;
        orow[i + 1536u] = v6 * scale;
        orow[i + 1792u] = v7 * scale;
    }
}



__global__ static void rms_norm_weight_kernel(float *out, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
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
        orow[i] = xr[i] * scale * w[i];
    }
}



__global__ static void dsv4_qkv_rms_norm_rows_kernel(
        float *q_out,
        const float *q,
        const float *q_w,
        uint32_t q_n,
        float *kv_out,
        const float *kv,
        const float *kv_w,
        uint32_t kv_n,
        uint32_t rows,
        float eps) {
    const uint32_t row = blockIdx.x;
    const uint32_t which = blockIdx.y;
    if (row >= rows || which > 1u) return;
    const uint32_t n = which == 0u ? q_n : kv_n;
    const float *xr = (which == 0u ? q : kv) + (uint64_t)row * n;
    float *orow = (which == 0u ? q_out : kv_out) + (uint64_t)row * n;
    const float *w = which == 0u ? q_w : kv_w;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}



__global__ static void head_rms_norm_kernel(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
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
    float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) xr[i] *= scale;
}



__global__ static void dsv4_qkv_rms_norm_rows_kv_rope_kernel(
        float *q_out,
        const float *q,
        const float *q_w,
        uint32_t q_n,
        float *kv_out,
        const float *kv,
        const float *kv_w,
        uint32_t kv_n,
        uint32_t rows,
        uint32_t kv_n_head,
        uint32_t kv_head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    const uint32_t row = blockIdx.x;
    const uint32_t which = blockIdx.y;
    if (row >= rows || which > 1u) return;
    const uint32_t n = which == 0u ? q_n : kv_n;
    const float *xr = (which == 0u ? q : kv) + (uint64_t)row * n;
    float *orow = (which == 0u ? q_out : kv_out) + (uint64_t)row * n;
    const float *w = which == 0u ? q_w : kv_w;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    if (which == 0u) {
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            orow[i] = xr[i] * scale * w[i];
        }
        return;
    }

    const uint32_t n_nope = kv_head_dim - n_rot;
    for (uint32_t h = 0; h < kv_n_head; h++) {
        const uint32_t head_base = h * kv_head_dim;
        for (uint32_t d = threadIdx.x; d < n_nope; d += blockDim.x) {
            const uint32_t i = head_base + d;
            orow[i] = xr[i] * scale * w[i];
        }
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    const uint32_t pairs_per_head = n_rot / 2u;
    const uint32_t total_pairs = kv_n_head * pairs_per_head;
    for (uint32_t p = threadIdx.x; p < total_pairs; p += blockDim.x) {
        const uint32_t h = p / pairs_per_head;
        const uint32_t pair = p - h * pairs_per_head;
        const uint32_t d = n_nope + pair * 2u;
        const uint32_t i0 = h * kv_head_dim + d;
        const uint32_t i = pair * 2u;
        float theta_extrap = (float)(pos0 + row) * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        if (inverse) s = -s;
        const float x0 = xr[i0] * scale * w[i0];
        const float x1 = xr[i0 + 1u] * scale * w[i0 + 1u];
        orow[i0] = x0 * c - x1 * s;
        orow[i0 + 1u] = x0 * s + x1 * c;
    }
}



__global__ static void head_rms_norm_rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    uint32_t t = row / n_head;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
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
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t i = threadIdx.x; i < n_nope; i += blockDim.x) {
        xr[i] *= scale;
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x) {
        uint32_t i = pair * 2u;
        float theta_extrap = (float)(pos0 + t) * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        if (inverse) s = -s;
        float *tail = xr + n_nope;
        float x0 = tail[i] * scale;
        float x1 = tail[i + 1] * scale;
        tail[i] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
}



__device__ static float rope_yarn_ramp_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}



__global__ static void rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t pos_stride,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    if (gid >= pairs) return;
    uint32_t pair = gid % (n_rot / 2);
    uint32_t tmp = gid / (n_rot / 2);
    uint32_t h = tmp % n_head;
    uint32_t t = tmp / n_head;
    uint32_t n_nope = head_dim - n_rot;
    uint32_t i = pair * 2;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }

    float theta_extrap = (float)(pos0 + t * pos_stride) * powf(freq_base, -((float)i) / (float)n_rot);
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;
    if (inverse) s = -s;

    float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
    float x0 = tail[i];
    float x1 = tail[i + 1];
    tail[i] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}



__global__ static void rope_tail_decode_rows_kernel(
        float *x,
        cuda_attention_decode_row_table rows,
        uint32_t n_rows,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pairs = n_rows * n_head * (n_rot / 2u);
    if (gid >= pairs) return;
    const uint32_t pair = gid % (n_rot / 2u);
    const uint32_t tmp = gid / (n_rot / 2u);
    const uint32_t h = tmp % n_head;
    const uint32_t row = tmp / n_head;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t i = pair * 2u;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        const float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot *
                       logf((float)n_ctx_orig /
                            (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot *
                      logf((float)n_ctx_orig /
                           (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1u), corr1);
    }

    const float theta_extrap = (float)rows.row[row].pos *
        powf(freq_base, -((float)i) / (float)n_rot);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float ramp_mix =
            rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;
    if (inverse) s = -s;

    float *tail = x + ((uint64_t)row * n_head + h) * head_dim + n_nope;
    const float x0 = tail[i];
    const float x1 = tail[i + 1u];
    tail[i] = x0 * c - x1 * s;
    tail[i + 1u] = x0 * s + x1 * c;
}



__device__ static float dsv4_e4m3fn_value_dev(int i) {
    int exp = (i >> 3) & 15;
    int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}



__device__ static float dsv4_e4m3fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * dsv4_e4m3fn_value_dev(best);
}



__device__ static unsigned char dsv4_e4m3fn_encode_dev(float x) {
    const bool negative = signbit(x);
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        const float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) &&
                                     ((best & 1) != 0))) {
            best++;
        }
    }
    return (unsigned char)best | (negative ? 0x80u : 0u);
}



__device__ __forceinline__ static half spark_kv_decode_half(
        const unsigned char *row, uint32_t dim) {
    if (dim < DS4_SPARK_KV_NOPE_DIM) {
        const unsigned char code = row[dim];
        const int exponent = (int)row[576u + (dim >> 6u)] - 127;
        const float scale = ldexpf(1.0f, exponent);
        const float mag = dsv4_e4m3fn_value_dev(code & 0x7fu);
        const float value = (code & 0x80u ? -mag : mag) * scale;
        return __float2half(value);
    }
    const uint32_t off = 448u + 2u * (dim - DS4_SPARK_KV_NOPE_DIM);
    __half_raw bits;
    bits.x = (unsigned short)((uint32_t)row[off] |
                              ((uint32_t)row[off + 1u] << 8u));
    return (__half)bits;
}



__device__ __forceinline__ static float spark_kv_decode(
        const unsigned char *row, uint32_t dim) {
    return __half2float(spark_kv_decode_half(row, dim));
}



__device__ static float dsv4_e2m1fn_value_dev(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}



__device__ static float dsv4_e2m1fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_value_dev(best);
}



__device__ static uint32_t dsv4_e2m1fn_encode_dev(float x) {
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff ||
            (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return (uint32_t)best | (signbit(x) ? 8u : 0u);
}



__device__ static float model_scalar_dev(const void *base, uint64_t offset, uint32_t type, uint64_t idx) {
    const char *p = (const char *)base + offset;
    if (type == 1u) return __half2float(((const __half *)p)[idx]);
    return ((const float *)p)[idx];
}



__device__ static void fp8_kv_quantize_row(
        float    *xr,
        uint32_t  head_dim,
        uint32_t  n_rot,
        float    *scratch) {
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
            xr[off + tid] = q;
        }
        __syncthreads();
    }
}



__global__ static void fp8_kv_quantize_kernel(
        float    *x,
        uint32_t  n_tok,
        uint32_t  head_dim,
        uint32_t  n_rot) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok) return;
    __shared__ float scratch[64];
    fp8_kv_quantize_row(
            x + (uint64_t)row * head_dim, head_dim, n_rot, scratch);
}



__device__ static void spark_pack_kv_row(
        float *src, unsigned char *dst, bool quantize_in_place) {
    __shared__ float scratch[64];
    const uint32_t tid = threadIdx.x;
    for (uint32_t off = 0; off < DS4_SPARK_KV_NOPE_DIM; off += 64u) {
        const float v = src[off + tid];
        scratch[tid] = fabsf(v);
        __syncthreads();
        for (uint32_t stride = 32u; stride > 0u; stride >>= 1u) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        const int exponent = (int)ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f));
        const float scale = ldexpf(1.0f, exponent);
        const float normalized = fminf(448.0f, fmaxf(-448.0f, v / scale));
        const unsigned char code = dsv4_e4m3fn_encode_dev(normalized);
        dst[off + tid] = code;
        if (quantize_in_place) {
            const float mag = dsv4_e4m3fn_value_dev(code & 0x7fu);
            src[off + tid] = (code & 0x80u ? -mag : mag) * scale;
        }
        if (tid == 0u) dst[576u + (off >> 6u)] = (unsigned char)(exponent + 127);
        __syncthreads();
    }
    const __half_raw tail = (__half_raw)__float2half(src[448u + tid]);
    dst[448u + 2u * tid] = (unsigned char)(tail.x & 0xffu);
    dst[449u + 2u * tid] = (unsigned char)(tail.x >> 8u);
}



__global__ static void spark_pack_kv_rows_kernel(
        unsigned char *dst, uint64_t dst_row,
        float *src, uint32_t src_row, uint32_t rows,
        bool quantize_in_place) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || threadIdx.x >= 64u) return;
    spark_pack_kv_row(src + (uint64_t)(src_row + row) * 512u,
                      dst + (dst_row + row) * DS4_SPARK_KV_ROW_BYTES,
                      quantize_in_place);
}



__global__ static void spark_pack_kv_ring_rows_kernel(
        unsigned char *dst, uint32_t raw_cap, uint32_t pos0,
        float *src, uint32_t rows) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || threadIdx.x >= 64u) return;
    spark_pack_kv_row(src + (uint64_t)row * 512u,
                      dst + (uint64_t)((pos0 + row) % raw_cap) *
                          DS4_SPARK_KV_ROW_BYTES,
                      false);
}



__global__ static void spark_pack_kv_decode_rows_kernel(
        float *src, cuda_attention_decode_row_table rows,
        uint32_t n_rows) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows || threadIdx.x >= 64u) return;
    const ds4_gpu_attention_decode_row dsc = rows.row[row];
    spark_pack_kv_row(src + (uint64_t)row * 512u,
                      (unsigned char *)(uintptr_t)dsc.raw_kv +
                          (uint64_t)dsc.raw_start * DS4_SPARK_KV_ROW_BYTES,
                      true);
}



__global__ static void spark_pack_index_rows_kernel(
        unsigned char *dst, uint64_t dst_row,
        const float *src, uint32_t src_row, uint32_t rows) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= rows || threadIdx.x >= 32u) return;
    const float *xr = src + (uint64_t)(src_row + row) * 128u;
    unsigned char *out = dst + (dst_row + row) * DS4_SPARK_INDEX_ROW_BYTES;
    float values[4];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        values[i] = xr[lane * 4u + i];
        amax = fmaxf(amax, fabsf(values[i]));
    }
#pragma unroll
    for (uint32_t off = 4u; off != 0u; off >>= 1u) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off, 8));
    }
    uint32_t exponent = 127u;
    if (amax > 0.0f) {
        const uint32_t bits = __float_as_uint(amax / 6.0f);
        exponent = (bits >> 23u) & 0xffu;
        if ((bits & 0x7fffffu) != 0u) exponent++;
        exponent = min(max(exponent, 1u), 254u);
    }
    const float reciprocal = __uint_as_float((254u - exponent) << 23u);
    uint32_t packed = 0u;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        packed |= dsv4_e2m1fn_encode_dev(values[i] * reciprocal) << (4 * i);
    }
    ((uint16_t *)out)[lane] = (uint16_t)packed;
    const uint32_t e1 = __shfl_sync(0xffffffffu, exponent, 8);
    const uint32_t e2 = __shfl_sync(0xffffffffu, exponent, 16);
    const uint32_t e3 = __shfl_sync(0xffffffffu, exponent, 24);
    if (lane == 0u) {
        *(uint32_t *)(out + 64u) = exponent | (e1 << 8u) | (e2 << 16u) | (e3 << 24u);
    }
}



__global__ static void spark_zero_kv_rows_kernel(
        unsigned char *dst, uint32_t rows) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    unsigned char *out = dst + (uint64_t)row * DS4_SPARK_KV_ROW_BYTES;
    for (uint32_t i = threadIdx.x; i < DS4_SPARK_KV_ROW_BYTES; i += blockDim.x) {
        out[i] = i >= 576u ? 127u : 0u;
    }
}



__global__ static void spark_zero_index_rows_kernel(
        unsigned char *dst, uint32_t rows) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    unsigned char *out = dst + (uint64_t)row * DS4_SPARK_INDEX_ROW_BYTES;
    for (uint32_t i = threadIdx.x; i < DS4_SPARK_INDEX_ROW_BYTES; i += blockDim.x) {
        out[i] = i >= 64u ? 127u : 0u;
    }
}



__device__ __forceinline__ static float spark_index_decode(
        const unsigned char *packed, uint32_t dim) {
    const unsigned char byte = packed[dim >> 1u];
    const uint32_t code = (dim & 1u) ? (byte >> 4u) : (byte & 0x0fu);
    const int exponent = (int)packed[64u + (dim >> 5u)] - 127;
    const float mag = dsv4_e2m1fn_value_dev(code & 7u);
    return (code & 8u ? -mag : mag) * ldexpf(1.0f, exponent);
}



__global__ static void spark_unpack_index_rows_kernel(
        float *dst, const unsigned char *src, uint32_t rows) {
    const uint32_t row = blockIdx.x;
    const uint32_t dim = threadIdx.x;
    if (row >= rows || dim >= 128u) return;
    const unsigned char *packed = src +
        (uint64_t)row * DS4_SPARK_INDEX_ROW_BYTES;
    dst[(uint64_t)row * 128u + dim] = spark_index_decode(packed, dim);
}



__global__ static void indexer_hadamard_fp4_kernel(float *x, uint32_t n_rows, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant_dev(fminf(6.0f, fmaxf(-6.0f, v / scale))) * scale;
}



__global__ static void spark_fill_iota_i32_kernel(int32_t *dst, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int32_t)i;
}



__global__ static void attention_prefill_raw_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t raw_count = t + 1 < window ? t + 1 : window;
    uint32_t raw_start = t + 1 - raw_count;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[256];
    __shared__ float partial[128];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    if (threadIdx.x == 0) {
        float den = expf(sinks[h] - max_s);
        for (uint32_t r = 0; r < raw_count; r++) {
            scores[r] = expf(scores[r] - max_s);
            den += scores[r];
        }
        denom = den;
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        }
        oh[d] = acc / denom;
    }
}



__global__ static void attention_prefill_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
    uint32_t raw_count = t + 1u - raw_start;
    uint32_t visible_comp = (t + 1u) / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    __shared__ float scores[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    uint32_t n_score = raw_count + visible_comp;

    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kvrow = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
        float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
        float s = -INFINITY;
        if (add > -1.0e20f) {
            const float *kvrow = comp_kv + (uint64_t)c * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            s = dot * scale + add;
        }
        scores[raw_count + c] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
        oh[d] = acc / denom;
    }
}



__global__ static void attention_prefill_raw_softmax_kernel(
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        bool valid = k <= t && (window == 0 || t - k < window);
        float s = valid ? row[k] : -INFINITY;
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}



__global__ static void attention_prefill_mixed_softmax_kernel(
        float *scores,
        const float *sinks,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || ratio == 0) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    const uint32_t visible_comp = (t + 1u) / ratio;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float s = -INFINITY;
        if (k < n_tokens) {
            if (k <= t && (window == 0 || t - k < window)) s = row[k];
        } else {
            uint32_t c = k - n_tokens;
            if (c < n_comp && c < visible_comp) {
                float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                if (add > -1.0e20f) s = row[k] + add;
            }
        }
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}



__global__ static void attention_prefill_pack_mixed_kv_kernel(
        float *dst,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)(n_tokens + n_comp) * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t r = gid / head_dim;
    dst[gid] = r < n_tokens ? raw_kv[(uint64_t)r * head_dim + d]
                             : comp_kv[(uint64_t)(r - n_tokens) * head_dim + d];
}



__global__ static void attention_prefill_unpack_heads_kernel(
        float *heads,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint64_t q = gid / head_dim;
    uint32_t h = q % n_head;
    uint32_t t = q / n_head;
    heads[gid] = tmp[((uint64_t)h * n_tokens + t) * head_dim + d];
}



__global__ static void attention_pack_group_heads_f16_kernel(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * group_dim;
    if (gid >= n) return;
    uint32_t d = gid % group_dim;
    uint64_t q = gid / group_dim;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    dst[gid] = __float2half(heads[((uint64_t)t * n_groups + g) * group_dim + d]);
}



__global__ static void attention_unpack_group_low_kernel(
        float *low,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t rank) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * rank;
    if (gid >= n) return;
    uint32_t r = gid % rank;
    uint64_t q = gid / rank;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    uint32_t low_dim = n_groups * rank;
    low[(uint64_t)t * low_dim + (uint64_t)g * rank + r] = tmp[gid];
}



__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t score_lanes_single) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    const uint32_t score_threads = blockDim.x > 256u ? 256u : blockDim.x;
    const bool score_thread = threadIdx.x < score_threads;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    if (score_thread) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += score_threads) {
            raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
        }
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    if (score_thread) {
        if (visible_comp == 0 || (n_tokens == 1u && score_lanes_single == 0u)) {
            for (uint32_t r = threadIdx.x; r < raw_count; r += score_threads) {
                const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                scores[r] = dot * scale;
                local_max = fmaxf(local_max, scores[r]);
            }
            for (uint32_t c = threadIdx.x; c < visible_comp; c += score_threads) {
                float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                float s = -INFINITY;
                if (add > -1.0e20f) {
                    const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                    s = dot * scale + add;
                }
                scores[raw_count + c] = s;
                local_max = fmaxf(local_max, s);
            }
        } else if (n_tokens == 1u && score_lanes_single == 4u) {
            uint32_t qlane = threadIdx.x & 3u;
            uint32_t qgroup = threadIdx.x >> 2u;
            for (uint32_t row0 = 0; row0 < n_score; row0 += 64u) {
                uint32_t row = row0 + qgroup;
                if (row < n_score) {
                    float add = 0.0f;
                    const float *kvrow = NULL;
                    if (row < raw_count) {
                        kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                    } else {
                        uint32_t c = row - raw_count;
                        add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                        if (add > -1.0e20f) kvrow = comp_kv + (uint64_t)c * head_dim;
                    }
                    float s = -INFINITY;
                    if (kvrow) {
                        float dot = 0.0f;
                        for (uint32_t d = qlane; d < head_dim; d += 4u) dot += qh[d] * kvrow[d];
                        const uint32_t mask = 0xfu << (threadIdx.x & 28u);
                        dot += __shfl_down_sync(mask, dot, 2, 4);
                        dot += __shfl_down_sync(mask, dot, 1, 4);
                        s = dot * scale + add;
                    }
                    if (qlane == 0) scores[row] = s;
                }
            }
            __syncthreads();
            for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
                local_max = fmaxf(local_max, scores[i]);
            }
        } else {
            uint32_t qlane = threadIdx.x & 7u;
            uint32_t qgroup = threadIdx.x >> 3u;
            for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
                uint32_t row = row0 + qgroup;
                if (row < n_score) {
                    float add = 0.0f;
                    const float *kvrow = NULL;
                    if (row < raw_count) {
                        kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                    } else {
                        uint32_t c = row - raw_count;
                        add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                        if (add > -1.0e20f) kvrow = comp_kv + (uint64_t)c * head_dim;
                    }
                    float s = -INFINITY;
                    if (kvrow) {
                        float dot = 0.0f;
                        for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                        const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                        for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                            dot += __shfl_down_sync(mask, dot, off, 8);
                        }
                        s = dot * scale + add;
                    }
                    if (qlane == 0) scores[row] = s;
                }
            }
            __syncthreads();
            for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
                local_max = fmaxf(local_max, scores[i]);
            }
        }
    }
    if (score_thread) partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            scores[i] = expf(scores[i] - max_s);
            den_local += scores[i];
        }
    }
    if (score_thread) partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x >= 512u) {
        uint32_t d = threadIdx.x;
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc += kv[d] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc += kv[d] * s;
        }
        oh[d] = acc / denom;
    } else if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
            oh[d] = acc / denom;
        }
    }
}



__global__ static void attention_decode_score_split_scores_kernel(
        float *score_out,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S) {
    const uint32_t h = blockIdx.y;
    const uint32_t j = blockIdx.z;
    if (h >= n_head || j >= S) return;
    const bool single_all = (ratio == 0u);
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + 1u - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;

    uint32_t raw_count = 0;
    uint32_t raw_first_idx = 0;
    if (n_raw != 0) {
        const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
        if (single_all) {
            raw_count = n_raw > 256u ? 256u : n_raw;
        } else if (qpos >= first_raw_pos) {
            uint32_t lo = first_raw_pos;
            if (window != 0 && qpos + 1u > window) {
                const uint32_t wlo = qpos + 1u - window;
                if (wlo > lo) lo = wlo;
            }
            const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
            if (hi >= lo) {
                raw_first_idx = lo - first_raw_pos;
                raw_count = hi - lo + 1u;
                if (raw_count > 256u) raw_count = 256u;
            }
        }
    }
    const uint32_t n_score = raw_count + visible_comp;
    if (n_score == 0u) return;

    const uint32_t qbase = n_score / S;
    const uint32_t rem = n_score % S;
    const uint32_t g0 = j * qbase + (j < rem ? j : rem);
    const uint32_t cnt = qbase + (j < rem ? 1u : 0u);
    const uint32_t g1 = g0 + cnt;
    const float *qh = q + (uint64_t)h * head_dim;
    float *row_scores = score_out + (uint64_t)h * n_score;
    const float scale = rsqrtf((float)head_dim);

    for (uint32_t g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
        float s = -INFINITY;
        if (g < raw_count) {
            const uint32_t raw_row =
                (raw_start + raw_first_idx + g) % raw_cap;
            const float *kvrow = raw_kv + (uint64_t)raw_row * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            s = dot * scale;
        } else {
            const uint32_t cidx = g - raw_count;
            const float add = use_comp_mask ? comp_mask[(uint64_t)cidx] : 0.0f;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)cidx * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                s = dot * scale + add;
            }
        }
        row_scores[g] = s;
    }
}



__device__ __forceinline__ float ds4_dot_scalar_ldg(
        const float *a,
        const float *b,
        uint32_t n) {
    float dot = 0.0f;
    for (uint32_t d = 0; d < n; d++) dot += __ldg(a + d) * __ldg(b + d);
    return dot;
}



__global__ static void attention_decode_score_split_scores_ldg_kernel(
        float *score_out,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S) {
    const uint32_t h = blockIdx.y;
    const uint32_t j = blockIdx.z;
    if (h >= n_head || j >= S) return;
    const bool single_all = (ratio == 0u);
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + 1u - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;

    uint32_t raw_count = 0;
    uint32_t raw_first_idx = 0;
    if (n_raw != 0) {
        const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
        if (single_all) {
            raw_count = n_raw > 256u ? 256u : n_raw;
        } else if (qpos >= first_raw_pos) {
            uint32_t lo = first_raw_pos;
            if (window != 0 && qpos + 1u > window) {
                const uint32_t wlo = qpos + 1u - window;
                if (wlo > lo) lo = wlo;
            }
            const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
            if (hi >= lo) {
                raw_first_idx = lo - first_raw_pos;
                raw_count = hi - lo + 1u;
                if (raw_count > 256u) raw_count = 256u;
            }
        }
    }
    const uint32_t n_score = raw_count + visible_comp;
    if (n_score == 0u) return;

    const uint32_t qbase = n_score / S;
    const uint32_t rem = n_score % S;
    const uint32_t g0 = j * qbase + (j < rem ? j : rem);
    const uint32_t cnt = qbase + (j < rem ? 1u : 0u);
    const uint32_t g1 = g0 + cnt;
    const float *qh = q + (uint64_t)h * head_dim;
    float *row_scores = score_out + (uint64_t)h * n_score;
    const float scale = rsqrtf((float)head_dim);

    for (uint32_t g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
        float s = -INFINITY;
        if (g < raw_count) {
            const uint32_t raw_row =
                (raw_start + raw_first_idx + g) % raw_cap;
            const float *kvrow = raw_kv + (uint64_t)raw_row * head_dim;
            const float dot = ds4_dot_scalar_ldg(qh, kvrow, head_dim);
            s = dot * scale;
        } else {
            const uint32_t cidx = g - raw_count;
            const float add = use_comp_mask ? comp_mask[(uint64_t)cidx] : 0.0f;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)cidx * head_dim;
                const float dot = ds4_dot_scalar_ldg(qh, kvrow, head_dim);
                s = dot * scale + add;
            }
        }
        row_scores[g] = s;
    }
}

#endif  // DS4X_BACKEND_LINEAR_KERNELS_CUH
