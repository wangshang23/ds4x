#include "../internal/backend_internal.cuh"

/* Attention Decode implementation. */
/* Head-tiled exact score kernel for head_dim==512.
 *
 * The reference score kernel assigns one (head, row-chunk) per block and lets
 * every thread walk one KV row with a scalar sequential dot. Because MQA
 * shares the same KV rows across all 64 heads, that reference layout re-reads
 * every KV row once per head, and the per-thread row walk is fully
 * uncoalesced (threads stride 2KB apart), which multiplies L2 traffic again.
 *
 * This kernel keeps the per-score arithmetic bit-identical (same ascending-d
 * scalar accumulation `dot += q[d] * kv[d]`, same `dot * scale [+ add]`
 * epilogue, same masked-row/raw-window classification) but stages a 16-row KV
 * tile and a 16-head Q tile in shared memory with coalesced global loads, so
 * each KV row is read from L2 once per 16 heads instead of once per head.
 * Scores are independent outputs, so retiling the (head, row) space cannot
 * change any output bit as long as each individual dot keeps its order. */

__global__ static void attention_decode_score_split_scores_tile512_kernel(
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
        uint32_t head_dim) {
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

    extern __shared__ float score_tile_shared[];
    float *sh_q = score_tile_shared; /* 16 x 516 */
    float *sh_kv = sh_q + DS4_SCORE_TILE_HEADS * DS4_SCORE_TILE_STRIDE; /* 16 x 516 */
    __shared__ float sh_add[DS4_SCORE_TILE_ROWS];

    const uint32_t g_base = blockIdx.x * DS4_SCORE_TILE_ROWS;
    const uint32_t h_base = blockIdx.y * DS4_SCORE_TILE_HEADS;
    if (g_base >= n_score || h_base >= n_head) return;

    /* Cooperative Q tile load: 16 heads x 512 floats, float4 coalesced. */
    {
        const float4 *q4 = (const float4 *)(q + (uint64_t)h_base * 512u);
        const uint32_t tile_heads =
            n_head - h_base < DS4_SCORE_TILE_HEADS ? n_head - h_base : DS4_SCORE_TILE_HEADS;
        for (uint32_t idx = threadIdx.x; idx < tile_heads * 128u; idx += blockDim.x) {
            const uint32_t hh = idx >> 7u;      /* head within tile */
            const uint32_t dd = idx & 127u;     /* float4 within row */
            const float4 v = q4[hh * 128u + dd];
            float *dst = sh_q + hh * DS4_SCORE_TILE_STRIDE + dd * 4u;
            dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
        }
    }
    /* Row classification + mask staging (thread per row). */
    if (threadIdx.x < DS4_SCORE_TILE_ROWS) {
        const uint32_t g = g_base + threadIdx.x;
        float add = -INFINITY;
        if (g < n_score) {
            if (g < raw_count) {
                add = 0.0f; /* raw rows are always visible */
            } else {
                const uint32_t cidx = g - raw_count;
                add = use_comp_mask ? comp_mask[(uint64_t)cidx] : 0.0f;
            }
        }
        sh_add[threadIdx.x] = add;
    }
    __syncthreads();
    /* Cooperative KV tile load: two rows at a time, float4 coalesced.
     * Masked rows (add <= -1e20) are skipped; their scores never read KV. */
    {
        const uint32_t rows_per_pass = blockDim.x >> 7u; /* 128 threads per row */
        const uint32_t rr0 = threadIdx.x >> 7u;
        const uint32_t dd = threadIdx.x & 127u;
        for (uint32_t r = rr0; r < DS4_SCORE_TILE_ROWS; r += rows_per_pass) {
            const uint32_t g = g_base + r;
            if (g >= n_score) continue;
            const bool visible = g < raw_count || sh_add[r] > -1.0e20f;
            if (!visible) continue;
            const float4 *src;
            if (g < raw_count) {
                const uint32_t raw_row = (raw_start + raw_first_idx + g) % raw_cap;
                src = (const float4 *)(raw_kv + (uint64_t)raw_row * 512u);
            } else {
                const uint32_t cidx = g - raw_count;
                src = (const float4 *)(comp_kv + (uint64_t)cidx * 512u);
            }
            const float4 v = src[dd];
            float *dst = sh_kv + r * DS4_SCORE_TILE_STRIDE + dd * 4u;
            dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
        }
    }
    __syncthreads();

    /* One score per thread: r = tid&15 (consecutive threads, coalesced score
     * writes), h = tid>>4. The dot keeps the reference kernel's exact scalar
     * ascending-d accumulation. */
    const uint32_t r = threadIdx.x & (DS4_SCORE_TILE_ROWS - 1u);
    const uint32_t h = h_base + (threadIdx.x >> 4u);
    const uint32_t g = g_base + r;
    if (h >= n_head || g >= n_score) return;
    const float scale = rsqrtf((float)head_dim);
    float *row_scores = score_out + (uint64_t)h * n_score;
    const float *qh = sh_q + (uint64_t)(threadIdx.x >> 4u) * DS4_SCORE_TILE_STRIDE;
    const float *kvrow = sh_kv + (uint64_t)r * DS4_SCORE_TILE_STRIDE;
    float s = -INFINITY;
    const bool need_dot = g < raw_count || sh_add[r] > -1.0e20f;
    if (need_dot) {
        /* The reference kernel's runtime-trip loop compiles to one sequential
         * FFMA chain. Keep exactly that accumulation order here: batched loads
         * for latency hiding, but a single explicit ascending fma chain. */
        float dot = 0.0f;
#pragma unroll 1
        for (uint32_t d = 0; d < 512u; d += 8u) {
            const float a0 = qh[d + 0u], a1 = qh[d + 1u];
            const float a2 = qh[d + 2u], a3 = qh[d + 3u];
            const float a4 = qh[d + 4u], a5 = qh[d + 5u];
            const float a6 = qh[d + 6u], a7 = qh[d + 7u];
            const float b0 = kvrow[d + 0u], b1 = kvrow[d + 1u];
            const float b2 = kvrow[d + 2u], b3 = kvrow[d + 3u];
            const float b4 = kvrow[d + 4u], b5 = kvrow[d + 5u];
            const float b6 = kvrow[d + 6u], b7 = kvrow[d + 7u];
            dot = __fmaf_rn(a0, b0, dot);
            dot = __fmaf_rn(a1, b1, dot);
            dot = __fmaf_rn(a2, b2, dot);
            dot = __fmaf_rn(a3, b3, dot);
            dot = __fmaf_rn(a4, b4, dot);
            dot = __fmaf_rn(a5, b5, dot);
            dot = __fmaf_rn(a6, b6, dot);
            dot = __fmaf_rn(a7, b7, dot);
        }
        if (g < raw_count) {
            s = dot * scale;
        } else {
            /* The reference expression `dot * scale + add` contracts to one
             * FFMA; keep that exact contraction explicit. */
            s = __fmaf_rn(dot, scale, sh_add[r]);
        }
    }
    row_scores[g] = s;
}

__device__ float ds4_dot512_float4_ordered(
        const float *a,
        const float *b) {
    const float4 *a4 = (const float4 *)a;
    const float4 *b4 = (const float4 *)b;
    float dot = 0.0f;
#pragma unroll 1
    for (uint32_t i = 0; i < 128u; i++) {
        const float4 av = a4[i];
        const float4 bv = b4[i];
        dot = __fadd_rn(dot, __fmul_rn(av.x, bv.x));
        dot = __fadd_rn(dot, __fmul_rn(av.y, bv.y));
        dot = __fadd_rn(dot, __fmul_rn(av.z, bv.z));
        dot = __fadd_rn(dot, __fmul_rn(av.w, bv.w));
    }
    return dot;
}

__device__ float ds4_dot512_float4_plain(
        const float *a,
        const float *b) {
    const float4 *a4 = (const float4 *)a;
    const float4 *b4 = (const float4 *)b;
    float dot = 0.0f;
#pragma unroll 1
    for (uint32_t i = 0; i < 128u; i++) {
        const float4 av = a4[i];
        const float4 bv = b4[i];
        dot += av.x * bv.x;
        dot += av.y * bv.y;
        dot += av.z * bv.z;
        dot += av.w * bv.w;
    }
    return dot;
}

__global__ static void attention_decode_score_split_scores_vec4_kernel(
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
        uint32_t S) {
    const uint32_t h = blockIdx.y;
    const uint32_t j = blockIdx.z;
    if (h >= n_head || j >= S) return;
    const uint32_t head_dim = 512u;
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
            const float dot = ds4_dot512_float4_ordered(qh, kvrow);
            s = dot * scale;
        } else {
            const uint32_t cidx = g - raw_count;
            const float add = use_comp_mask ? comp_mask[(uint64_t)cidx] : 0.0f;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)cidx * head_dim;
                const float dot = ds4_dot512_float4_ordered(qh, kvrow);
                s = dot * scale + add;
            }
        }
        row_scores[g] = s;
    }
}

__global__ static void attention_decode_score_split_scores_vec4_plain_kernel(
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
        uint32_t S) {
    const uint32_t h = blockIdx.y;
    const uint32_t j = blockIdx.z;
    if (h >= n_head || j >= S) return;
    const uint32_t head_dim = 512u;
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
            const float dot = ds4_dot512_float4_plain(qh, kvrow);
            s = dot * scale;
        } else {
            const uint32_t cidx = g - raw_count;
            const float add = use_comp_mask ? comp_mask[(uint64_t)cidx] : 0.0f;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)cidx * head_dim;
                const float dot = ds4_dot512_float4_plain(qh, kvrow);
                s = dot * scale + add;
            }
        }
        row_scores[g] = s;
    }
}

__global__ static void attention_decode_score_split_finalize_kernel(
        float *heads,
        const float *sinks,
        const float *score_in,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t h = blockIdx.y;
    if (h >= n_head) return;
    const bool single_all = (ratio == 0u);
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + 1u - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;

    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;

    const uint32_t score_threads = blockDim.x > 256u ? 256u : blockDim.x;
    const bool score_thread = threadIdx.x < score_threads;
    if (threadIdx.x == 0) {
        raw_count_s = 0;
        raw_first_idx_s = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count_s = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx_s = lo - first_raw_pos;
                    raw_count_s = hi - lo + 1u;
                    if (raw_count_s > 256u) raw_count_s = 256u;
                }
            }
        }
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    if (score_thread) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += score_threads) {
            raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
        }
    }
    __syncthreads();
    const uint32_t n_score = raw_count + visible_comp;
    const float *row_scores = score_in + (uint64_t)h * n_score;
    float local_max = sinks[h];
    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            const float s = row_scores[i];
            scores[i] = s;
            local_max = fmaxf(local_max, s);
        }
    }
    if (score_thread) partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] =
                fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
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
    float *oh = heads + (uint64_t)h * head_dim;
    if (head_dim == 512u && blockDim.x >= 512u) {
        const uint32_t d = threadIdx.x;
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc += kv[d] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            const float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc += kv[d] * s;
        }
        oh[d] = acc / denom;
    } else if (head_dim == 512u && blockDim.x == 256u) {
        const uint32_t d0 = threadIdx.x;
        const uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            const float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) {
                acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
            }
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_decode_score_split_finalize_dim2_kernel(
        float *heads,
        const float *sinks,
        const float *score_in,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t dim_half = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (h >= n_head || head_dim != 512u || dim_half >= 2u) return;
    const bool single_all = (ratio == 0u);
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + 1u - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;

    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;

    const uint32_t score_threads = 256u;
    if (threadIdx.x == 0) {
        raw_count_s = 0;
        raw_first_idx_s = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count_s = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx_s = lo - first_raw_pos;
                    raw_count_s = hi - lo + 1u;
                    if (raw_count_s > 256u) raw_count_s = 256u;
                }
            }
        }
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    for (uint32_t r = threadIdx.x; r < raw_count; r += score_threads) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t n_score = raw_count + visible_comp;
    const float *row_scores = score_in + (uint64_t)h * n_score;
    float local_max = sinks[h];
    for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
        const float s = row_scores[i];
        scores[i] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] =
                fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();

    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();

    const uint32_t d = dim_half * 256u + threadIdx.x;
    float acc = 0.0f;
    for (uint32_t r = 0; r < raw_count; r++) {
        const float s = scores[r];
        const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
        acc += kv[d] * s;
    }
    for (uint32_t c = 0; c < visible_comp; c++) {
        const float s = scores[raw_count + c];
        const float *kv = comp_kv + (uint64_t)c * head_dim;
        acc += kv[d] * s;
    }
    heads[(uint64_t)h * head_dim + d] = acc / denom;
}

void attention_decode_score_split_graph_destroy_one(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= DS4_MAX_GPUS) return;
    cuda_score_split_graph_cache *c = &g_score_split_graph[logical_tier];
    if (c->exec) (void)cudaGraphExecDestroy(c->exec);
    if (c->graph) (void)cudaGraphDestroy(c->graph);
    memset(c, 0, sizeof(*c));
}

static int attention_decode_score_split_graph_launch(
        int logical_tier,
        float *heads,
        const float *sinks,
        float *scores,
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
        uint32_t final_threads,
        uint32_t S,
        const cuda_attention_inv_rope_params *inv_rope) {
    if (logical_tier < 0 || logical_tier >= DS4_MAX_GPUS) return 0;
    cuda_score_split_graph_cache *c = &g_score_split_graph[logical_tier];
    const bool graph_inv_rope =
        inv_rope &&
        head_dim == 512u &&
        inv_rope->n_rot != 0u &&
        inv_rope->n_rot <= head_dim &&
        (inv_rope->n_rot & 1u) == 0u;
    const bool shape_match =
        c->valid &&
        c->n_head == n_head &&
        c->head_dim == head_dim &&
        c->S == S &&
        c->final_threads == final_threads &&
        c->fuses_inv_rope == (graph_inv_rope ? 1 : 0) &&
        (!graph_inv_rope || c->n_rot == inv_rope->n_rot);
    if (c->valid && !shape_match) {
        attention_decode_score_split_graph_destroy_one(logical_tier);
        c = &g_score_split_graph[logical_tier];
    }

    dim3 score_grid(1, n_head, S);
    dim3 final_grid(1, n_head, 1);
    dim3 score_block(256, 1, 1);
    dim3 final_block(final_threads, 1, 1);

    void *score_args[] = {
        &scores, &q, &raw_kv, &comp_kv, &comp_mask, &use_comp_mask,
        &pos0, &n_raw, &raw_cap, &raw_start, &n_comp, &window, &ratio,
        &n_head, &head_dim, &S
    };
    cudaKernelNodeParams score_params;
    memset(&score_params, 0, sizeof(score_params));
    score_params.func = (void *)attention_decode_score_split_scores_kernel;
    score_params.gridDim = score_grid;
    score_params.blockDim = score_block;
    score_params.sharedMemBytes = 0;
    score_params.kernelParams = score_args;
    score_params.extra = NULL;

    void *final_args[] = {
        &heads, &sinks, &scores, &raw_kv, &comp_kv, &pos0, &n_raw,
        &raw_cap, &raw_start, &n_comp, &window, &ratio, &n_head, &head_dim
    };
    cudaKernelNodeParams final_params;
    memset(&final_params, 0, sizeof(final_params));
    final_params.func = (void *)attention_decode_score_split_finalize_kernel;
    final_params.gridDim = final_grid;
    final_params.blockDim = final_block;
    final_params.sharedMemBytes = 0;
    final_params.kernelParams = final_args;
    final_params.extra = NULL;

    uint32_t rope_n_tok = 1u;
    uint32_t rope_pos_stride = 1u;
    int rope_inverse = 1;
    uint32_t rope_n_rot = graph_inv_rope ? inv_rope->n_rot : 0u;
    uint32_t rope_pos0 = graph_inv_rope ? inv_rope->pos0 : 0u;
    uint32_t rope_n_ctx_orig = graph_inv_rope ? inv_rope->n_ctx_orig : 0u;
    float rope_freq_base = graph_inv_rope ? inv_rope->freq_base : 0.0f;
    float rope_freq_scale = graph_inv_rope ? inv_rope->freq_scale : 0.0f;
    float rope_ext_factor = graph_inv_rope ? inv_rope->ext_factor : 0.0f;
    float rope_attn_factor = graph_inv_rope ? inv_rope->attn_factor : 0.0f;
    float rope_beta_fast = graph_inv_rope ? inv_rope->beta_fast : 0.0f;
    float rope_beta_slow = graph_inv_rope ? inv_rope->beta_slow : 0.0f;
    void *rope_args[] = {
        &heads, &rope_n_tok, &n_head, &head_dim, &rope_n_rot,
        &rope_pos0, &rope_pos_stride, &rope_n_ctx_orig, &rope_inverse,
        &rope_freq_base, &rope_freq_scale, &rope_ext_factor,
        &rope_attn_factor, &rope_beta_fast, &rope_beta_slow
    };
    cudaKernelNodeParams rope_params;
    memset(&rope_params, 0, sizeof(rope_params));
    if (graph_inv_rope) {
        const uint32_t pairs = n_head * (rope_n_rot / 2u);
        rope_params.func = (void *)rope_tail_kernel;
        rope_params.gridDim = dim3((pairs + 255u) / 256u, 1, 1);
        rope_params.blockDim = dim3(256, 1, 1);
        rope_params.sharedMemBytes = 0;
        rope_params.kernelParams = rope_args;
        rope_params.extra = NULL;
    }

    if (!c->valid) {
        cudaError_t err = cudaGraphCreate(&c->graph, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: attention score-split graph create failed: %s\n",
                    cudaGetErrorString(err));
            attention_decode_score_split_graph_destroy_one(logical_tier);
            return -1;
        }
        err = cudaGraphAddKernelNode(&c->score_node, c->graph, NULL, 0,
                                     &score_params);
        if (err == cudaSuccess) {
            err = cudaGraphAddKernelNode(&c->final_node, c->graph,
                                         &c->score_node, 1, &final_params);
        }
        if (err == cudaSuccess && graph_inv_rope) {
            err = cudaGraphAddKernelNode(&c->rope_node, c->graph,
                                         &c->final_node, 1, &rope_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphInstantiate(&c->exec, c->graph, NULL, NULL, 0);
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: attention score-split graph instantiate failed: %s\n",
                    cudaGetErrorString(err));
            attention_decode_score_split_graph_destroy_one(logical_tier);
            return -1;
        }
        c->n_head = n_head;
        c->head_dim = head_dim;
        c->S = S;
        c->final_threads = final_threads;
        c->n_rot = graph_inv_rope ? inv_rope->n_rot : 0u;
        c->fuses_inv_rope = graph_inv_rope ? 1 : 0;
        c->valid = 1;
    } else {
        cudaError_t err =
            cudaGraphExecKernelNodeSetParams(c->exec, c->score_node,
                                             &score_params);
        if (err == cudaSuccess) {
            err = cudaGraphExecKernelNodeSetParams(c->exec, c->final_node,
                                                   &final_params);
        }
        if (err == cudaSuccess && graph_inv_rope) {
            err = cudaGraphExecKernelNodeSetParams(c->exec, c->rope_node,
                                                   &rope_params);
        }
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: attention score-split graph update failed: %s\n",
                    cudaGetErrorString(err));
            attention_decode_score_split_graph_destroy_one(logical_tier);
            return -1;
        }
    }

    cudaError_t err = cudaGraphLaunch(c->exec, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: attention score-split graph launch failed: %s\n",
                cudaGetErrorString(err));
        attention_decode_score_split_graph_destroy_one(logical_tier);
        return -1;
    }
    return 1;
}

int attention_decode_score_split_launch(
        int logical_tier,
        float *heads,
        const float *sinks,
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
        uint32_t final_threads,
        const cuda_attention_inv_rope_params *inv_rope) {
    if (cuda_env_flag_enabled("DS4_CUDA_NO_EXACT_SCORE_SPLIT_DECODE", 0)) return 0;
    const int explicit_exact =
        cuda_env_flag_enabled("DS4_CUDA_EXACT_SCORE_SPLIT_DECODE", 0);
    if (!cuda_env_flag_enabled("DS4_CUDA_EXACT_SCORE_SPLIT_DECODE", 1)) return 0;
    if (!explicit_exact && cuda_splitkv_decode_requested()) return 0;
    if (g_cuda_decode_score4 || g_cuda_decode_score8) return 0;
    if (head_dim == 0u || n_head == 0u) return 0;
    const bool single_all = (ratio == 0u);
    const uint32_t qpos = pos0;
    const uint32_t first_raw_pos = pos0 + 1u - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    uint32_t raw_count = 0;
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
                raw_count = hi - lo + 1u;
                if (raw_count > 256u) raw_count = 256u;
            }
        }
    }
    const uint32_t n_score = raw_count + visible_comp;
    if (n_score == 0u || n_score > DS4_CUDA_ATTENTION_SCORE_CAP) return 0;
    /* With the head-tiled score kernel the exact score-split path beats the
     * one-block mixed kernel even for short score counts, so the gate that
     * used to protect short contexts (512) now defaults to 1. */
    const uint32_t min_score = cuda_parse_u32_env_clamped(
            "DS4_CUDA_EXACT_SCORE_SPLIT_MIN_SCORE", 1u, 0u,
            DS4_CUDA_ATTENTION_SCORE_CAP, NULL);
    if (n_score < min_score) return 0;
    uint32_t chunk = cuda_parse_u32_env_clamped(
            "DS4_CUDA_EXACT_SCORE_SPLIT_CHUNK", DS4_CUDA_SPLITKV_CHUNK,
            1u, DS4_CUDA_ATTENTION_SCORE_CAP, NULL);
    uint32_t s_floor = cuda_parse_u32_env_clamped(
            "DS4_CUDA_EXACT_SCORE_SPLIT_S_FLOOR", 6u,
            1u, DS4_CUDA_SPLITKV_S_MAX, NULL);
    uint32_t s_max = cuda_parse_u32_env_clamped(
            "DS4_CUDA_EXACT_SCORE_SPLIT_S_MAX", DS4_CUDA_SPLITKV_S_MAX,
            1u, DS4_CUDA_SPLITKV_S_MAX, NULL);
    int exact_present = 0;
    uint32_t S = cuda_parse_u32_env_clamped(
            "DS4_CUDA_EXACT_SCORE_SPLIT_S", 0u, 1u,
            DS4_CUDA_SPLITKV_S_MAX, &exact_present);
    if (!exact_present) {
        S = (n_score + chunk - 1u) / chunk;
        if (S < s_floor) S = s_floor < n_score ? s_floor : n_score;
        if (S > s_max) S = s_max;
    }
    if (S > n_score) S = n_score;
    if (S <= 1u) return 0;
    const bool graph_inv_rope =
        g_cuda_exact_score_split_fuse_inv_rope &&
        inv_rope &&
        head_dim == 512u &&
        final_threads >= 512u &&
        inv_rope->n_rot != 0u &&
        inv_rope->n_rot <= 512u &&
        (inv_rope->n_rot & 1u) == 0u;

    const uint64_t score_count = (uint64_t)n_head * n_score;
    float *scores = (float *)cuda_tmp_alloc_on(logical_tier,
                                               score_count * sizeof(float),
                                               "attention exact score split");
    if (!scores) return 0;
    const bool use_ldg_scores = g_cuda_exact_score_split_ldg;
    const bool use_vec4_plain_scores =
        !use_ldg_scores &&
        g_cuda_exact_score_split_vec4_plain &&
        head_dim == 512u;
    const bool use_vec4_scores =
        !use_ldg_scores &&
        !use_vec4_plain_scores &&
        (g_cuda_exact_score_split_vec4 || g_decode_score_vec4) &&
        head_dim == 512u;
    const bool use_dim2_finalize =
        g_cuda_exact_score_split_dim2 &&
        head_dim == 512u &&
        final_threads >= 512u &&
        !graph_inv_rope;
    if ((g_cuda_exact_score_split_graph || graph_inv_rope) &&
        !use_dim2_finalize &&
        !use_ldg_scores &&
        !use_vec4_plain_scores &&
        !use_vec4_scores)
    {
        int rc = attention_decode_score_split_graph_launch(
            logical_tier, heads, sinks, scores, q, raw_kv, comp_kv, comp_mask,
            use_comp_mask, pos0, n_raw, raw_cap, raw_start, n_comp, window,
            ratio, n_head, head_dim, final_threads, S,
            graph_inv_rope ? inv_rope : NULL);
        if (rc == 1) return 1;
        if (rc < 0) return -1;
    }
    if (graph_inv_rope) return 0;
    static int score_tile_disabled = -1;
    if (score_tile_disabled < 0) {
        score_tile_disabled = getenv("DS4_CUDA_NO_SCORE_TILE") != NULL ? 1 : 0;
    }
    if (!score_tile_disabled &&
        head_dim == 512u &&
        !use_ldg_scores &&
        !use_vec4_plain_scores &&
        !use_vec4_scores) {
        /* cudaFuncSetAttribute() applies to the current device only, so opt in
         * to >48KB dynamic shared memory once per device. */
        static int tile_shmem_ready[DS4_MAX_GPUS] = {0};
        const size_t tile_shmem =
            (size_t)(DS4_SCORE_TILE_HEADS + DS4_SCORE_TILE_ROWS) *
            DS4_SCORE_TILE_STRIDE * sizeof(float);
        int tile_dev = 0;
        cudaGetDevice(&tile_dev);
        if (tile_dev >= 0 && tile_dev < DS4_MAX_GPUS &&
            !tile_shmem_ready[tile_dev]) {
            if (!cuda_ok(cudaFuncSetAttribute(
                             attention_decode_score_split_scores_tile512_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)tile_shmem),
                         "attention score tile shared-memory opt-in")) {
                score_tile_disabled = 1;
            }
            tile_shmem_ready[tile_dev] = 1;
        }
        if (score_tile_disabled) {
            return 0; /* retry via the generic path on the next call */
        }
        dim3 tile_grid((n_score + DS4_SCORE_TILE_ROWS - 1u) / DS4_SCORE_TILE_ROWS,
                       (n_head + DS4_SCORE_TILE_HEADS - 1u) / DS4_SCORE_TILE_HEADS,
                       1);
        attention_decode_score_split_scores_tile512_kernel<<<tile_grid, 256, tile_shmem>>>(
                scores, q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention exact score split tile launch")) return -1;
    } else {
    dim3 score_grid(1, n_head, S);
    if (use_ldg_scores) {
        attention_decode_score_split_scores_ldg_kernel<<<score_grid, 256>>>(
                scores, q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim, S);
    } else if (use_vec4_plain_scores) {
        attention_decode_score_split_scores_vec4_plain_kernel<<<score_grid, 256>>>(
                scores, q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, S);
    } else if (use_vec4_scores) {
        attention_decode_score_split_scores_vec4_kernel<<<score_grid, 256>>>(
                scores, q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, S);
    } else {
        attention_decode_score_split_scores_kernel<<<score_grid, 256>>>(
                scores, q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim, S);
    }
    if (!cuda_ok(cudaGetLastError(), "attention exact score split scores launch")) return -1;
    }
    if (use_dim2_finalize) {
        dim3 final_grid(2, n_head, 1);
        attention_decode_score_split_finalize_dim2_kernel<<<final_grid, 256>>>(
                heads, sinks, scores, raw_kv, comp_kv,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention exact score split dim2 finalize launch")) return -1;
    } else {
        dim3 final_grid(1, n_head, 1);
        attention_decode_score_split_finalize_kernel<<<final_grid, final_threads>>>(
                heads, sinks, scores, raw_kv, comp_kv,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention exact score split finalize launch")) return -1;
    }
    return 1;
}

__device__ void tt_ldmatrix_x4(uint32_t (&r)[4], const void *p) {
    tt_ldmatrix_x4_addr(r, tt_smem_addr(p));
}

__device__ void tt_ldmatrix_x2(uint32_t (&r)[2], const void *p) {
    tt_ldmatrix_x2_addr(r, tt_smem_addr(p));
}

__device__ void tt_ldmatrix_x2_trans(uint32_t (&r)[2], const void *p) {
    tt_ldmatrix_x2_trans_addr(r, tt_smem_addr(p));
}
int ds4_cuda_attn_tokentile_arch_ok(void) {
    int device = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    return prop.major >= 8;
}
