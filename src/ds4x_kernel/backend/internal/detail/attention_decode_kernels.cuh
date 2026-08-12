#ifndef DS4X_BACKEND_ATTENTION_DECODE_KERNELS_CUH
#define DS4X_BACKEND_ATTENTION_DECODE_KERNELS_CUH

#include "../backend_common.cuh"



/* Multi-session form of the exact tiled score kernel. Each z-slice selects a
 * private KV table entry, while every individual score keeps the same scalar
 * ascending-d FMA chain as the one-session kernel. */
__global__ static void attention_decode_score_split_scores_tile512_rows_kernel(
        float *score_out,
        const float *q,
        cuda_attention_decode_row_table rows,
        uint32_t n_rows,
        uint32_t score_stride,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.z;
    if (row >= n_rows) return;
    const ds4_gpu_attention_decode_row dsc = rows.row[row];
    if (dsc.indexed) return;

    const float *raw_kv = (const float *)(uintptr_t)dsc.raw_kv;
    const float *comp_kv = (const float *)(uintptr_t)dsc.comp_kv;
    const bool single_all = dsc.ratio == 0u;
    const uint32_t qpos = dsc.pos;
    const uint32_t first_raw_pos = dsc.pos + 1u - dsc.n_raw;
    uint32_t visible_comp = single_all
        ? dsc.n_comp
        : (dsc.n_comp ? (qpos + 1u) / dsc.ratio : 0u);
    if (visible_comp > dsc.n_comp) visible_comp = dsc.n_comp;

    uint32_t raw_count = 0u;
    uint32_t raw_first_idx = 0u;
    if (dsc.n_raw != 0u) {
        const uint32_t raw_last_pos = first_raw_pos + dsc.n_raw - 1u;
        if (single_all) {
            raw_count = dsc.n_raw > 256u ? 256u : dsc.n_raw;
        } else if (qpos >= first_raw_pos) {
            uint32_t lo = first_raw_pos;
            if (dsc.window != 0u && qpos + 1u > dsc.window) {
                const uint32_t wlo = qpos + 1u - dsc.window;
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

    extern __shared__ float score_tile_shared[];
    float *sh_q = score_tile_shared;
    float *sh_kv = sh_q + DS4_SCORE_TILE_HEADS * DS4_SCORE_TILE_STRIDE;

    const uint32_t g_base = blockIdx.x * DS4_SCORE_TILE_ROWS;
    const uint32_t h_base = blockIdx.y * DS4_SCORE_TILE_HEADS;
    if (g_base >= n_score || h_base >= n_head) return;

    {
        const float4 *q4 = (const float4 *)(
            q + ((uint64_t)row * n_head + h_base) * head_dim);
        const uint32_t tile_heads =
            n_head - h_base < DS4_SCORE_TILE_HEADS
                ? n_head - h_base : DS4_SCORE_TILE_HEADS;
        for (uint32_t idx = threadIdx.x;
             idx < tile_heads * 128u;
             idx += blockDim.x) {
            const uint32_t hh = idx >> 7u;
            const uint32_t dd = idx & 127u;
            const float4 v = q4[hh * 128u + dd];
            float *dst = sh_q + hh * DS4_SCORE_TILE_STRIDE + dd * 4u;
            dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
        }
    }
    __syncthreads();
    {
        const uint32_t rows_per_pass = blockDim.x >> 7u;
        const uint32_t rr0 = threadIdx.x >> 7u;
        const uint32_t dd = threadIdx.x & 127u;
        for (uint32_t r = rr0; r < DS4_SCORE_TILE_ROWS; r += rows_per_pass) {
            const uint32_t g = g_base + r;
            if (g >= n_score) continue;
            const float4 *src;
            if (g < raw_count) {
                const uint32_t raw_row =
                    (dsc.raw_start + raw_first_idx + g) % dsc.raw_cap;
                src = (const float4 *)(raw_kv + (uint64_t)raw_row * head_dim);
            } else {
                src = (const float4 *)(comp_kv +
                    (uint64_t)(g - raw_count) * head_dim);
            }
            const float4 v = src[dd];
            float *dst = sh_kv + r * DS4_SCORE_TILE_STRIDE + dd * 4u;
            dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
        }
    }
    __syncthreads();

    const uint32_t r = threadIdx.x & (DS4_SCORE_TILE_ROWS - 1u);
    const uint32_t h = h_base + (threadIdx.x >> 4u);
    const uint32_t g = g_base + r;
    if (h >= n_head || g >= n_score) return;
    const float scale = rsqrtf((float)head_dim);
    float *row_scores = score_out +
        ((uint64_t)row * n_head + h) * score_stride;
    const float *qh = sh_q +
        (uint64_t)(threadIdx.x >> 4u) * DS4_SCORE_TILE_STRIDE;
    const float *kvrow = sh_kv + (uint64_t)r * DS4_SCORE_TILE_STRIDE;
    float dot = 0.0f;
#pragma unroll 1
    for (uint32_t dd = 0; dd < 512u; dd += 8u) {
        const float a0 = qh[dd + 0u], a1 = qh[dd + 1u];
        const float a2 = qh[dd + 2u], a3 = qh[dd + 3u];
        const float a4 = qh[dd + 4u], a5 = qh[dd + 5u];
        const float a6 = qh[dd + 6u], a7 = qh[dd + 7u];
        const float b0 = kvrow[dd + 0u], b1 = kvrow[dd + 1u];
        const float b2 = kvrow[dd + 2u], b3 = kvrow[dd + 3u];
        const float b4 = kvrow[dd + 4u], b5 = kvrow[dd + 5u];
        const float b6 = kvrow[dd + 6u], b7 = kvrow[dd + 7u];
        dot = __fmaf_rn(a0, b0, dot);
        dot = __fmaf_rn(a1, b1, dot);
        dot = __fmaf_rn(a2, b2, dot);
        dot = __fmaf_rn(a3, b3, dot);
        dot = __fmaf_rn(a4, b4, dot);
        dot = __fmaf_rn(a5, b5, dot);
        dot = __fmaf_rn(a6, b6, dot);
        dot = __fmaf_rn(a7, b7, dot);
    }
    row_scores[g] = g < raw_count
        ? dot * scale
        : __fmaf_rn(dot, scale, 0.0f);
}



__global__ static void attention_decode_score_split_finalize_rows_kernel(
        float *heads,
        const float *sinks,
        const float *score_in,
        cuda_attention_decode_row_table rows,
        uint32_t n_rows,
        uint32_t score_stride,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (row >= n_rows || h >= n_head) return;
    const ds4_gpu_attention_decode_row dsc = rows.row[row];
    if (dsc.indexed) return;
    const float *raw_kv = (const float *)(uintptr_t)dsc.raw_kv;
    const float *comp_kv = (const float *)(uintptr_t)dsc.comp_kv;
    const bool single_all = dsc.ratio == 0u;
    const uint32_t qpos = dsc.pos;
    const uint32_t first_raw_pos = dsc.pos + 1u - dsc.n_raw;
    uint32_t visible_comp = single_all
        ? dsc.n_comp
        : (dsc.n_comp ? (qpos + 1u) / dsc.ratio : 0u);
    if (visible_comp > dsc.n_comp) visible_comp = dsc.n_comp;

    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;

    const uint32_t score_threads = blockDim.x > 256u ? 256u : blockDim.x;
    const bool score_thread = threadIdx.x < score_threads;
    if (threadIdx.x == 0u) {
        raw_count_s = 0u;
        raw_first_idx_s = 0u;
        if (dsc.n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + dsc.n_raw - 1u;
            if (single_all) {
                raw_count_s = dsc.n_raw > 256u ? 256u : dsc.n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (dsc.window != 0u && qpos + 1u > dsc.window) {
                    const uint32_t wlo = qpos + 1u - dsc.window;
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
            raw_rows[r] =
                (dsc.raw_start + raw_first_idx + r) % dsc.raw_cap;
        }
    }
    __syncthreads();
    const uint32_t n_score = raw_count + visible_comp;
    const float *row_scores = score_in +
        ((uint64_t)row * n_head + h) * score_stride;
    float local_max = sinks[h];
    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            const float s = row_scores[i];
            scores[i] = s;
            local_max = fmaxf(local_max, s);
        }
        partial[threadIdx.x] = local_max;
    }
    __syncthreads();
    for (uint32_t stride = score_threads >> 1u;
         stride > 0u;
         stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] =
                fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            scores[i] = expf(scores[i] - max_s);
            den_local += scores[i];
        }
        partial[threadIdx.x] = den_local;
    }
    __syncthreads();
    for (uint32_t stride = score_threads >> 1u;
         stride > 0u;
         stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        denom = partial[0] + expf(sinks[h] - max_s);
    }
    __syncthreads();

    float *oh = heads + ((uint64_t)row * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x >= 512u) {
        const uint32_t dim = threadIdx.x;
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc += kv[dim] * scores[r];
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc += kv[dim] * scores[raw_count + c];
        }
        oh[dim] = acc / denom;
    } else {
        for (uint32_t dim = threadIdx.x;
             dim < head_dim;
             dim += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) {
                acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + dim] *
                       scores[r];
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                acc += comp_kv[(uint64_t)c * head_dim + dim] *
                       scores[raw_count + c];
            }
            oh[dim] = acc / denom;
        }
    }
}



__global__ static void attention_decode_global_softmax_kernel(
        float *score_inout,
        float *denom_out,
        const float *sinks,
        uint32_t n_score,
        uint32_t n_head) {
    const uint32_t h = blockIdx.x;
    if (h >= n_head || n_score == 0u || n_score > DS4_CUDA_ATTENTION_SCORE_CAP) return;
    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom_s;
    const uint32_t score_threads = blockDim.x > 256u ? 256u : blockDim.x;
    const bool score_thread = threadIdx.x < score_threads;
    float *row_scores = score_inout + (uint64_t)h * n_score;

    float local_max = sinks[h];
    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            const float s = row_scores[i];
            scores[i] = s;
            local_max = fmaxf(local_max, s);
        }
        partial[threadIdx.x] = local_max;
    }
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
            const float e = expf(scores[i] - max_s);
            scores[i] = e;
            den_local += e;
        }
        partial[threadIdx.x] = den_local;
    }
    __syncthreads();
    for (uint32_t stride = score_threads >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom_s = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();

    if (score_thread) {
        for (uint32_t i = threadIdx.x; i < n_score; i += score_threads) {
            row_scores[i] = scores[i];
        }
    }
    if (threadIdx.x == 0) denom_out[h] = denom_s;
}



__global__ static void attention_decode_split_value_kernel(
        float *partials,
        const float *score_exp,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t raw_count,
        uint32_t raw_first_idx,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_score,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S) {
    const uint32_t h = blockIdx.y;
    const uint32_t j = blockIdx.z;
    if (h >= n_head || j >= S || n_score == 0u) return;
    const uint32_t qbase = n_score / S;
    const uint32_t rem = n_score % S;
    const uint32_t g0 = j * qbase + (j < rem ? j : rem);
    const uint32_t cnt = qbase + (j < rem ? 1u : 0u);
    const uint32_t g1 = g0 + cnt;
    const float *row_scores = score_exp + (uint64_t)h * n_score;
    float *pout = partials + ((uint64_t)h * S + j) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t g = g0; g < g1; g++) {
            const float s = row_scores[g];
            if (g < raw_count) {
                const uint32_t raw_row =
                    (raw_start + raw_first_idx + g) % raw_cap;
                acc += raw_kv[(uint64_t)raw_row * head_dim + d] * s;
            } else {
                const uint32_t c = g - raw_count;
                acc += comp_kv[(uint64_t)c * head_dim + d] * s;
            }
        }
        pout[d] = acc;
    }
}



__global__ static void attention_decode_split_value_combine_kernel(
        float *heads,
        const float *partials,
        const float *denom,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S) {
    const uint32_t h = blockIdx.y;
    if (h >= n_head) return;
    const float *base = partials + (uint64_t)h * S * head_dim;
    const float den = denom[h];
    float *oh = heads + (uint64_t)h * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < S; j++) {
            acc += base[(uint64_t)j * head_dim + d];
        }
        oh[d] = acc / den;
    }
}



/* ---- perf-02 split-KV / flash-decode (opt-in, default OFF) ----------------
 *
 * attention_decode_splitkv_kernel computes a partial online-softmax over a
 * contiguous chunk of the flattened logical row set [0, n_score) used by
 * attention_decode_mixed_kernel (raw rows first, then compressed rows, same
 * ascending ordering). Each block handles (t = blockIdx.x, h = blockIdx.y,
 * chunk = blockIdx.z) and writes a partial (m_j, l_j, acc_j[head_dim]) WITHOUT
 * the sink term. attention_decode_splitkv_combine_kernel merges the S partials
 * per (t,h), folds the sink once, and writes the final normalized head output.
 *
 * The math is the standard flash-attention online-softmax rescale and is
 * algebraically identical to attention_decode_mixed_kernel; it is NOT
 * guaranteed bit-identical in FP32 (different expf inputs + add/mul grouping),
 * hence default-OFF behind DS4_CUDA_SPLITKV_DECODE and the S==1 dispatch to the
 * old kernel as the bit-exact anchor (handled in the launch helper).
 *
 * Partials scratch layout (per logical tier), contiguous floats:
 *   stride = head_dim + 2
 *   base(t,h,j) = ((t*n_head + h)*S + j) * stride
 *     [0]               = m_j   (chunk running max; -INF if empty/all-masked)
 *     [1]               = l_j   (chunk denominator sum exp(s - m_j))
 *     [2 .. 2+head_dim) = acc_j[head_dim] (chunk weighted value sum)
 */
__global__ static void attention_decode_splitkv_kernel(
        float *partials,
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
        uint32_t S) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    uint32_t j = blockIdx.z;
    if (t >= n_tokens || h >= n_head || j >= S) return;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    /* scores buffer holds only this chunk's rows. The launch helper guarantees
     * cnt <= DS4_CUDA_SPLITKV_SCORE_CAP, including env-tuned split counts. */
    __shared__ float scores[DS4_CUDA_SPLITKV_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float m_s;
    __shared__ float l_s;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
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
    uint32_t n_score = raw_count + visible_comp;
    /* even split of [0, n_score) across S chunks: first (n_score % S) chunks
     * get base+1, identical deterministic partition for every block. */
    uint32_t qbase = n_score / S;
    uint32_t rem = n_score % S;
    uint32_t g0 = j * qbase + (j < rem ? j : rem);
    uint32_t cnt = qbase + (j < rem ? 1u : 0u);
    uint32_t g1 = g0 + cnt;          /* exclusive end of this chunk */
    /* Map raw rows that fall in this chunk into shared raw_rows[]. The chunk's
     * raw portion is [raw_lo, raw_hi). cnt <= CHUNK and raw rows <= 256, so
     * the slice fits raw_rows[256]. */
    uint32_t raw_lo = g0 < raw_count ? g0 : raw_count;
    uint32_t raw_hi = g1 < raw_count ? g1 : raw_count;
    for (uint32_t r = raw_lo + threadIdx.x; r < raw_hi; r += blockDim.x) {
        raw_rows[r - raw_lo] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();
    float *pout = partials + (((uint64_t)t * n_head + h) * S + j) * (head_dim + 2u);
    /* Pass 1: scores for this chunk's rows into shared scores[0..cnt). */
    float local_max = -INFINITY;
    for (uint32_t i = threadIdx.x; i < cnt; i += blockDim.x) {
        uint32_t g = g0 + i;
        float s;
        if (g < raw_count) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[g - raw_lo] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            s = dot * scale;
        } else {
            uint32_t c = g - raw_count;
            float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            s = -INFINITY;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                s = dot * scale + add;
            }
        }
        scores[i] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) m_s = partial[0];
    __syncthreads();
    float chunk_max = m_s;
    /* All-masked / empty-chunk guard: never evaluate exp(-INF - -INF) -> NaN.
     * Write zero partial (m=-INF, l=0, acc=0) and return. */
    if (!isfinite(chunk_max)) {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) pout[2u + d] = 0.0f;
        if (threadIdx.x == 0) { pout[0] = -INFINITY; pout[1] = 0.0f; }
        return;
    }
    /* Pass 2: exponentiate in place and reduce denominator. */
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < cnt; i += blockDim.x) {
        float e = expf(scores[i] - chunk_max);
        scores[i] = e;
        den_local += e;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) l_s = partial[0];
    __syncthreads();
    /* Pass 3: weighted value accumulation over this chunk's rows (ascending g),
     * preserving raw-then-comp ordering to match the reference accumulation. */
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t i = 0; i < cnt; i++) {
            uint32_t g = g0 + i;
            float s = scores[i];
            const float *kv = (g < raw_count)
                    ? raw_kv + (uint64_t)raw_rows[g - raw_lo] * head_dim
                    : comp_kv + (uint64_t)(g - raw_count) * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        pout[2u + d0] = acc0;
        pout[2u + d1] = acc1;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t i = 0; i < cnt; i++) {
                uint32_t g = g0 + i;
                float s = scores[i];
                const float *kv = (g < raw_count)
                        ? raw_kv + (uint64_t)raw_rows[g - raw_lo] * head_dim
                        : comp_kv + (uint64_t)(g - raw_count) * head_dim;
                acc += kv[d] * s;
            }
            pout[2u + d] = acc;
        }
    }
    if (threadIdx.x == 0) {
        pout[0] = chunk_max;
        pout[1] = l_s;
    }
}



__global__ static void attention_decode_splitkv_combine_kernel(
        float *heads,
        const float *sinks,
        const float *partials,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const float *base = partials + (((uint64_t)t * n_head + h) * S) * (head_dim + 2u);
    uint32_t stride = head_dim + 2u;
    __shared__ float M_s;
    __shared__ float L_s;
    if (threadIdx.x == 0) {
        /* Global max M = max(sink, max_j m_j); sink placed first to match the
         * reference (sink seeds local_max). */
        float M = sinks[h];
        for (uint32_t jj = 0; jj < S; jj++) {
            float m_j = base[(uint64_t)jj * stride];
            M = fmaxf(M, m_j);   /* -INF partials never raise M */
        }
        M_s = M;
        /* L = Σ_j exp(m_j - M) * l_j + exp(sink - M); sink term added last to
         * mirror the reference's denom = Σ scores + expf(sink - max). Chunks
         * with l_j == 0 / m_j == -INF contribute exactly 0 (guarded to avoid
         * exp(-INF - finite) * 0 edge cases). */
        float L = 0.0f;
        for (uint32_t jj = 0; jj < S; jj++) {
            float m_j = base[(uint64_t)jj * stride];
            float l_j = base[(uint64_t)jj * stride + 1u];
            if (l_j != 0.0f && isfinite(m_j)) L += expf(m_j - M) * l_j;
        }
        L += expf(sinks[h] - M);
        L_s = L;
    }
    __syncthreads();
    float M = M_s;
    float L = L_s;
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float A = 0.0f;
        for (uint32_t jj = 0; jj < S; jj++) {
            float m_j = base[(uint64_t)jj * stride];
            float l_j = base[(uint64_t)jj * stride + 1u];
            if (l_j != 0.0f && isfinite(m_j)) {
                A += expf(m_j - M) * base[(uint64_t)jj * stride + 2u + d];
            }
        }
        oh[d] = A / L;
    }
}



__device__ __forceinline__ void attention_compact_topk_stable(
        uint32_t *comp_rows,
        uint32_t *comp_count,
        uint32_t *warp_offsets,
        const int32_t *topk,
        uint32_t top_k,
        uint32_t visible_comp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t n_warp = blockDim.x >> 5u;
    if (threadIdx.x == 0u) *comp_count = 0u;
    __syncthreads();

    for (uint32_t base = 0u; base < 512u; base += blockDim.x) {
        const uint32_t i = base + threadIdx.x;
        const int32_t c = i < top_k ? topk[i] : -1;
        const bool valid = c >= 0 && (uint32_t)c < visible_comp;
        const uint32_t mask = __ballot_sync(0xffffffffu, valid);
        if (lane == 0u) warp_offsets[warp] = __popc(mask);
        __syncthreads();
        if (threadIdx.x == 0u) {
            uint32_t out = *comp_count;
            for (uint32_t w = 0u; w < n_warp; w++) {
                const uint32_t count = warp_offsets[w];
                warp_offsets[w] = out;
                out += count;
            }
            *comp_count = out;
        }
        __syncthreads();
        if (valid) {
            const uint32_t lanes_before = lane == 0u
                ? 0u : ((1u << lane) - 1u);
            const uint32_t slot = warp_offsets[warp] +
                                  __popc(mask & lanes_before);
            if (slot < 512u) comp_rows[slot] = (uint32_t)c;
        }
        __syncthreads();
    }
}



__global__ static void attention_indexed_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[768];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    __shared__ uint32_t comp_warp_offsets[8];
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
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
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    attention_compact_topk_stable(
        comp_rows, &comp_count, comp_warp_offsets,
        topk + (uint64_t)t * top_k, top_k, visible_comp);
    uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                const float *kvrow = row < raw_count
                    ? raw_kv + (uint64_t)raw_rows[row] * head_dim
                    : comp_kv + (uint64_t)comp_rows[row - raw_count] * head_dim;
                float dot = 0.0f;
                for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(mask, dot, off, 8);
                }
                if (qlane == 0) scores[row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
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
    if (head_dim == 512u && blockDim.x == 256u) {
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
        for (uint32_t c = 0; c < comp_count; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)comp_rows[c] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t s = 0; s < comp_count; s++) acc += comp_kv[(uint64_t)comp_rows[s] * head_dim + d] * scores[raw_count + s];
            oh[d] = acc / denom;
        }
    }
}



__global__ static void attention_indexed_mixed_decode_rows_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        cuda_attention_decode_row_table rows,
        uint32_t n_rows,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (row >= n_rows || h >= n_head) return;
    const ds4_gpu_attention_decode_row dsc = rows.row[row];
    if (!dsc.indexed) return;
    const float *raw_kv = (const float *)(uintptr_t)dsc.raw_kv;
    const float *comp_kv = (const float *)(uintptr_t)dsc.comp_kv;
    const int32_t *topk = (const int32_t *)(uintptr_t)dsc.topk;
    const uint32_t qpos = dsc.pos;
    const uint32_t first_raw_pos = dsc.pos + 1u - dsc.n_raw;
    uint32_t visible_comp = dsc.n_comp;
    if (dsc.ratio != 0u) {
        visible_comp = (qpos + 1u) / dsc.ratio;
        if (visible_comp > dsc.n_comp) visible_comp = dsc.n_comp;
    }
    const float *qh = q + ((uint64_t)row * n_head + h) * head_dim;
    __shared__ float scores[768];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    __shared__ uint32_t comp_warp_offsets[8];
    const float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0u) {
        raw_count = 0u;
        raw_first_idx = 0u;
        if (dsc.n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + dsc.n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (dsc.window != 0u && qpos + 1u > dsc.window) {
                    const uint32_t wlo = qpos + 1u - dsc.window;
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
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] =
            (dsc.raw_start + raw_first_idx + r) % dsc.raw_cap;
    }
    attention_compact_topk_stable(
        comp_rows, &comp_count, comp_warp_offsets,
        topk, dsc.top_k, visible_comp);
    const uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0u) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t dim = 0; dim < head_dim; dim++) {
                dot += qh[dim] * kvrow[dim];
            }
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        const uint32_t qlane = threadIdx.x & 7u;
        const uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            const uint32_t score_row = row0 + qgroup;
            if (score_row < n_score) {
                const float *kvrow = score_row < raw_count
                    ? raw_kv + (uint64_t)raw_rows[score_row] * head_dim
                    : comp_kv +
                        (uint64_t)comp_rows[score_row - raw_count] * head_dim;
                float dot = 0.0f;
                for (uint32_t dim = qlane; dim < head_dim; dim += 8u) {
                    dot += qh[dim] * kvrow[dim];
                }
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(mask, dot, off, 8);
                }
                if (qlane == 0u) scores[score_row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u;
         stride > 0u;
         stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] =
                fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u;
         stride > 0u;
         stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        denom = partial[0] + expf(sinks[h] - max_s);
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)row * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
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
        for (uint32_t c = 0; c < comp_count; c++) {
            const float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)comp_rows[c] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t dim = threadIdx.x;
             dim < head_dim;
             dim += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) {
                acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + dim] *
                       scores[r];
            }
            for (uint32_t c = 0; c < comp_count; c++) {
                acc += comp_kv[(uint64_t)comp_rows[c] * head_dim + dim] *
                       scores[raw_count + c];
            }
            oh[dim] = acc / denom;
        }
    }
}



__global__ static void attention_indexed_mixed_heads8_rb4_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    __shared__ float4 kv_shared[4 * 128];
    __shared__ float scores[8 * 768];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
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
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < top_k && comp_count < 512u; i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp) comp_rows[comp_count++] = (uint32_t)c;
        }
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float dot = dot4_f32(q0, kv4[lane +  0u]) +
                            dot4_f32(q1, kv4[lane + 32u]) +
                            dot4_f32(q2, kv4[lane + 64u]) +
                            dot4_f32(q3, kv4[lane + 96u]);
                dot = warp_sum_f32(dot);
                if (lane == 0) scores[warp * 768u + row0 + rr] = dot * scale;
            }
        }
        __syncthreads();
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    if (valid_head) {
        const float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) max_s = fmaxf(max_s, score_row[i]);
        max_s = warp_max_f32(max_s);
        max_s = __shfl_sync(0xffffffffu, max_s, 0);
    }
    float den = 0.0f;
    if (valid_head) {
        float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) {
            float p = expf(score_row[i] - max_s);
            score_row[i] = p;
            den += p;
        }
        den = warp_sum_f32(den);
        den += expf(sinks[head] - max_s);
        den = __shfl_sync(0xffffffffu, den, 0);
    }

    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;
    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            const float *score_row = scores + warp * 768u;
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float p = den == 0.0f ? 0.0f : score_row[row0 + rr] / den;
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                o0.x += k0.x * p; o0.y += k0.y * p; o0.z += k0.z * p; o0.w += k0.w * p;
                o1.x += k1.x * p; o1.y += k1.y * p; o1.z += k1.z * p; o1.w += k1.w * p;
                o2.x += k2.x * p; o2.y += k2.y * p; o2.z += k2.z * p; o2.w += k2.w * p;
                o3.x += k3.x * p; o3.y += k3.y * p; o3.z += k3.z * p; o3.w += k3.w * p;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}



template <uint32_t ROWS_PER_STAGE, uint32_t HEADS_PER_GROUP>
__global__ static void __launch_bounds__(512, 2)
attention_indexed_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * HEADS_PER_GROUP + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ float4 kv_shared[ROWS_PER_STAGE * 128];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
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
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    uint32_t comp_count = top_k < visible_comp ? top_k : visible_comp;
    if (comp_count > 512u) comp_count = 512u;
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = n_score - row0 < ROWS_PER_STAGE ? n_score - row0 : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const uint32_t comp_idx = sr < raw_count
                ? 0u
                : (uint32_t)topk[(uint64_t)t * top_k + (sr - raw_count)];
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_idx * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}



__global__ static void attention_static_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ float4 kv_shared[4 * 128];

    const uint32_t raw_count = window != 0u && t + 1u > window ? window : t + 1u;
    const uint32_t raw_start = t + 1u - raw_count;
    uint32_t comp_count = 0;
    if (n_comp != 0u && ratio != 0u) {
        comp_count = (t + 1u) / ratio;
        if (comp_count > n_comp) comp_count = n_comp;
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)(raw_start + sr) * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}



__device__ static float tt_warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}



__device__ static float tt_warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}



__device__ __forceinline__ uint32_t tt_lane_id(void) {
    return threadIdx.x & 31u;
}



__device__ __forceinline__ uint32_t tt_warp_id(void) {
    return threadIdx.x >> 5u;
}



__device__ __forceinline__ int tt_mma_c_i(uint32_t lane, int l) {
    return ((l >> 1) << 3) + (int)(lane >> 2);
}



__device__ __forceinline__ int tt_mma_c_j(uint32_t lane, int l) {
    return (int)((lane & 3u) << 1) + (l & 1);
}



__device__ __forceinline__ unsigned tt_smem_addr(const void *p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}



__device__ __forceinline__ uint32_t tt_ring_off_bytes(uint32_t row, uint32_t c) {
    return (row * kTTRingChunksPerRow + (c ^ (row & 7u))) * kTTRingChunkBytes;
}



__device__ __forceinline__ void tt_ldmatrix_x4_addr(uint32_t (&r)[4], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = r[2] = r[3] = 0;
#endif
}



__device__ __forceinline__ void tt_ldmatrix_x2_addr(uint32_t (&r)[2], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = 0;
#endif
}



__device__ __forceinline__ void tt_ldmatrix_x2_trans_addr(uint32_t (&r)[2], unsigned a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
#else
    (void)a;
    r[0] = r[1] = 0;
#endif
}



__device__ __forceinline__ void tt_mma_m16n8k16_f16_f32(
        float *d,
        const uint32_t (&a)[4],
        const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t d0 = __float_as_uint(d[0]);
    uint32_t d1 = __float_as_uint(d[1]);
    uint32_t d2 = __float_as_uint(d[2]);
    uint32_t d3 = __float_as_uint(d[3]);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    d[0] = __uint_as_float(d0);
    d[1] = __uint_as_float(d1);
    d[2] = __uint_as_float(d2);
    d[3] = __uint_as_float(d3);
#else
    (void)a;
    (void)b;
#endif
}



__device__ __forceinline__ unsigned char *tt_align16(unsigned char *p) {
    uintptr_t x = reinterpret_cast<uintptr_t>(p);
    x = (x + 15u) & ~uintptr_t(15u);
    return reinterpret_cast<unsigned char *>(x);
}



__device__ __forceinline__ void tt_zero_16B(void *dst) {
    *reinterpret_cast<int4 *>(dst) = make_int4(0, 0, 0, 0);
}



__device__ __forceinline__ void tt_zero_8B(void *dst) {
    *reinterpret_cast<int2 *>(dst) = make_int2(0, 0);
}



__device__ __forceinline__ void tt_cp_async_16B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = tt_smem_addr(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem), "l"(src));
    } else {
        tt_zero_16B(dst);
    }
#else
    (void)src;
    (void)pred;
    tt_zero_16B(dst);
#endif
}



__device__ __forceinline__ void tt_cp_async_8B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = tt_smem_addr(dst);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                     :: "r"(smem), "l"(src));
    } else {
        tt_zero_8B(dst);
    }
#else
    (void)src;
    (void)pred;
    tt_zero_8B(dst);
#endif
}



__device__ __forceinline__ void tt_cp_async_commit(void) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}



template <int KeepGroups>
__device__ __forceinline__ void tt_cp_async_wait_group(void) {
    static_assert(KeepGroups >= 0 && KeepGroups <= 7, "bad cp.async wait_group depth");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(KeepGroups));
#endif
}



__device__ __forceinline__ uint32_t tt_score_partial_slot(
        uint32_t kq,
        uint32_t m,
        uint32_t r) {
    return (kq + r + m) & 3u;
}



template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_store_score_partial(
        float4 * __restrict__ partials,
        uint32_t kq,
        uint32_t m,
        uint32_t r,
        float v) {
    float *dst = &partials[m * TT_STAGE_ROWS + r].x;
    dst[tt_score_partial_slot(kq, m, r)] = v;
}



__device__ __forceinline__ float4 tt_load_score_partial_record(
        const float4 * __restrict__ p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float x, y, z, w;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                 : "r"(tt_smem_addr(p)));
    return make_float4(x, y, z, w);
#else
    return *p;
#endif
}



__device__ __forceinline__ void tt_issue_cp_async_row(
        half * __restrict__ dst,
        uint32_t rr,
        uint32_t lane16,
        const half * __restrict__ src,
        bool live) {
    const char *src_b = reinterpret_cast<const char *>(src);
    char *dst_b = reinterpret_cast<char *>(dst);

#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t chunk = lane16 + i * 16u;
        char *db = dst_b + tt_ring_off_bytes(rr, chunk);
        tt_cp_async_16B(db, live ? static_cast<const void *>(src_b + chunk * 16u)
                                 : static_cast<const void *>(db),
                        live);
    }
}



__device__ __forceinline__ uint32_t tt_stage_raw_rows(
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count) {
    uint32_t raw_rows = 0;
    if (row0 < raw_union_count) {
        const uint32_t raw_left = raw_union_count - row0;
        raw_rows = raw_left < nr ? raw_left : nr;
    }
    return raw_rows;
}



__global__ static void __launch_bounds__(512, 1) attention_tokentile_union_build_kernel(
        int2 *records,
        uint32_t *counts,
        const int32_t *topk,
        const int32_t *positions,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t rec_stride) {
    extern __shared__ uint32_t bitmap[];
    __shared__ uint32_t scan[513];
    __shared__ uint32_t running_s;

    const uint32_t tid = threadIdx.x;
    const uint32_t tile_base = blockIdx.x * kTTTileTokens;
    if (tile_base >= n_tokens) {
        if (tid == 0u) counts[blockIdx.x] = 0u;
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens ? n_tokens - tile_base : kTTTileTokens;
    const uint32_t bitmap_words = (n_comp + 1u) >> 1u;

    for (uint32_t w = tid; w < bitmap_words; w += blockDim.x) {
        bitmap[w] = 0u;
    }
    if (tid == 0u) running_s = 0u;
    __syncthreads();

    const uint32_t total_slots = tile_count * top_k;
    for (uint32_t idx = tid; idx < total_slots; idx += blockDim.x) {
        const uint32_t tok = idx / top_k;
        const uint32_t i = idx - tok * top_k;
        const uint32_t t = tile_base + tok;
        const uint32_t qpos = positions ? (uint32_t)positions[t] : pos0 + t;
        uint32_t visible = ratio ? (positions ? qpos / ratio : (qpos + 1u) / ratio) : n_comp;
        if (visible > n_comp) visible = n_comp;
        const int32_t c = topk[(uint64_t)t * top_k + i];
        if (c >= 0 && (uint32_t)c < visible) {
            const uint32_t cu = (uint32_t)c;
            const uint32_t bits = ((uint32_t)(1u << tok)) << ((cu & 1u) * 16u);
            atomicOr(&bitmap[cu >> 1u], bits);
        }
    }
    __syncthreads();

    for (uint32_t base = 0u; base < n_comp; base += blockDim.x) {
        const uint32_t id = base + tid;
        uint32_t mask = 0u;
        if (id < n_comp) {
            const uint32_t word = bitmap[id >> 1u];
            mask = (id & 1u) ? (word >> 16u) : (word & 0xffffu);
        }
        const uint32_t pred = mask != 0u ? 1u : 0u;
        if (tid == 0u) scan[0] = 0u;
        scan[tid + 1u] = pred;
        __syncthreads();
        for (uint32_t off = 1u; off < blockDim.x; off <<= 1u) {
            uint32_t v = 0u;
            if (tid >= off) v = scan[tid + 1u - off];
            __syncthreads();
            scan[tid + 1u] += v;
            __syncthreads();
        }
        const uint32_t rank = scan[tid];
        const uint32_t total = scan[blockDim.x];
        const uint32_t running = running_s;
        if (pred) {
            records[(uint64_t)blockIdx.x * rec_stride + running + rank] =
                make_int2((int)id, (int)mask);
        }
        __syncthreads();
        if (tid == 0u) running_s = running + total;
        __syncthreads();
    }

    if (tid == 0u) counts[blockIdx.x] = running_s;
}



__global__ static void __launch_bounds__(256, 1) attention_tokentile_raw_mirror_kernel(
        half *dst,
        const float *raw_kv,
        const int32_t *seq_id,
        uint32_t tt_run_pos0,
        uint32_t n_tokens,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t first_raw_pos,
        uint32_t raw_row_min,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t d0 = threadIdx.x << 1u;
    if (d0 >= head_dim) return;
    half *dst_row = dst + (uint64_t)row * head_dim;
    if (row < raw_row_min) {
        dst_row[d0] = __float2half(0.0f);
        if (d0 + 1u < head_dim) dst_row[d0 + 1u] = __float2half(0.0f);
        return;
    }

    const int64_t p = (int64_t)tt_run_pos0 - (int64_t)(kTTRawWindow - 1u) + (int64_t)row;
    uint32_t slot = 0u;
    if (seq_id) {
        slot = (uint32_t)seq_id[0] * raw_cap + (uint32_t)((uint64_t)p % raw_cap);
    } else {
        const uint32_t rel = (uint32_t)(p - (int64_t)first_raw_pos);
        slot = (raw_start + rel) % raw_cap;
    }
    const float *src = raw_kv + (uint64_t)slot * head_dim;
    dst_row[d0] = __float2half(src[d0]);
    if (d0 + 1u < head_dim) dst_row[d0 + 1u] = __float2half(src[d0 + 1u]);
}



__global__ static void __launch_bounds__(256, 1) attention_tokentile_comp_mirror_kernel(
        half *dst,
        const float *comp_kv,
        uint32_t n_comp,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t c4 = threadIdx.x;
    if (row >= n_comp || c4 >= (head_dim >> 2u)) return;
    const float4 v = ((const float4 *)(comp_kv + (uint64_t)row * head_dim))[c4];
    half *out = dst + (uint64_t)row * head_dim + (c4 << 2u);
    out[0] = __float2half(v.x);
    out[1] = __float2half(v.y);
    out[2] = __float2half(v.z);
    out[3] = __float2half(v.w);
}



/* Packed prefill feeds HMMA directly through FP16 mirrors. Persistent rows
 * remain in the 583-byte cache format; no transient F32 history is created. */
__global__ static void __launch_bounds__(256, 1)
attention_tokentile_raw_packed_mirror_kernel(
        half *dst,
        const unsigned char *raw_kv,
        uint32_t tt_run_pos0,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t first_raw_pos,
        uint32_t raw_row_min,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t d0 = threadIdx.x << 1u;
    if (d0 >= head_dim) return;
    half *dst_row = dst + (uint64_t)row * head_dim;
    if (row < raw_row_min) {
        dst_row[d0] = __float2half(0.0f);
        if (d0 + 1u < head_dim) dst_row[d0 + 1u] = __float2half(0.0f);
        return;
    }

    const int64_t p = (int64_t)tt_run_pos0 -
                      (int64_t)(kTTRawWindow - 1u) + (int64_t)row;
    const uint32_t rel = (uint32_t)(p - (int64_t)first_raw_pos);
    const uint32_t slot = (raw_start + rel) % raw_cap;
    const unsigned char *src = raw_kv +
        (uint64_t)slot * DS4_SPARK_KV_ROW_BYTES;
    dst_row[d0] = spark_kv_decode_half(src, d0);
    if (d0 + 1u < head_dim) {
        dst_row[d0 + 1u] = spark_kv_decode_half(src, d0 + 1u);
    }
}



__global__ static void __launch_bounds__(256, 1)
attention_tokentile_comp_packed_mirror_kernel(
        half *dst,
        const unsigned char *comp_kv,
        uint32_t n_comp,
        uint32_t head_dim) {
    const uint32_t row = blockIdx.x;
    const uint32_t d0 = threadIdx.x << 1u;
    if (row >= n_comp || d0 >= head_dim) return;
    const unsigned char *src = comp_kv +
        (uint64_t)row * DS4_SPARK_KV_ROW_BYTES;
    half *out = dst + (uint64_t)row * head_dim;
    out[d0] = spark_kv_decode_half(src, d0);
    if (d0 + 1u < head_dim) out[d0 + 1u] = spark_kv_decode_half(src, d0 + 1u);
}



/* Non-indexed layers select the causal compressed range [0, visible), so
 * their per-tile records can be built directly without a bitmap or sort. */
__global__ static void __launch_bounds__(512, 1)
attention_tokentile_dense_build_kernel(
        int2 *records,
        uint32_t *counts,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t rec_stride) {
    __shared__ uint32_t visible_s[kTTTileTokens];
    const uint32_t tid = threadIdx.x;
    const uint32_t tile_base = blockIdx.x * kTTTileTokens;
    if (tile_base >= n_tokens) {
        if (tid == 0u) counts[blockIdx.x] = 0u;
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens
            ? n_tokens - tile_base : kTTTileTokens;
    if (tid < kTTTileTokens) {
        uint32_t visible = 0u;
        if (tid < tile_count && n_comp != 0u && ratio != 0u) {
            visible = (pos0 + tile_base + tid + 1u) / ratio;
            if (visible > n_comp) visible = n_comp;
        }
        visible_s[tid] = visible;
    }
    __syncthreads();

    uint32_t vmax = 0u;
    for (uint32_t i = 0; i < tile_count; i++) {
        if (visible_s[i] > vmax) vmax = visible_s[i];
    }
    for (uint32_t c = tid; c < vmax; c += blockDim.x) {
        uint32_t mask = 0u;
        for (uint32_t i = 0; i < tile_count; i++) {
            if (c < visible_s[i]) mask |= 1u << i;
        }
        records[(uint64_t)blockIdx.x * rec_stride + c] =
            make_int2((int)c, (int)mask);
    }
    if (tid == 0u) counts[blockIdx.x] = vmax;
}



template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_issue_record_stage_cp_async(
        int2 * __restrict__ rec_plane,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        const int2 * __restrict__ union_records_tile) {
    static_assert(TT_STAGE_ROWS == 32u, "token-tile record issue is fixed at R32");
    const uint32_t raw_rows = tt_stage_raw_rows(row0, nr, raw_union_count);
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t lane16 = lane & 15u;
    const uint32_t rr = warp * 2u + (lane >> 4u);
    const bool active = rr < TT_STAGE_ROWS;
    const bool comp_live = active && rr >= raw_rows && rr < nr;
    if (comp_live && lane16 == 0u) {
        int2 *dst = rec_plane + rr;
        const uint32_t ci = row0 + rr - raw_union_count;
        tt_cp_async_8B(dst, union_records_tile + ci, true);
    }
}



template <uint32_t TT_STAGE_ROWS, bool USE_SMEM_RECORDS>
__device__ __forceinline__ void tt_issue_kv_stage_cp_async(
        half * __restrict__ dst,
        const int2 * __restrict__ rec_plane,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        const int2 * __restrict__ union_records_tile,
        uint32_t tile_base,
        const half * __restrict__ raw_kv,
        const half * __restrict__ comp_kv,
        uint32_t tid) {
    constexpr uint32_t kCp16PerRow = (kTTHeadDim * sizeof(half)) / 16u;
    static_assert(kCp16PerRow == 64u, "expected 64 cp.async chunks per f16 KV row");
    static_assert(TT_STAGE_ROWS == 32u, "token-tile KV issue is fixed at R32");
    (void)tid;
    const uint32_t raw_rows = tt_stage_raw_rows(row0, nr, raw_union_count);
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t lane16 = lane & 15u;
    const uint32_t rr = warp * 2u + (lane >> 4u);
    const bool active = rr < TT_STAGE_ROWS;

    if (active && rr < raw_rows) {
        const uint32_t sr = row0 + rr;
        const half *src = raw_kv + (uint64_t)(tile_base + sr) * kTTHeadDim;
        tt_issue_cp_async_row(dst, rr, lane16, src, true);
    }

    if (active && rr >= raw_rows && rr < nr) {
        uint32_t comp_id = 0u;
        if (USE_SMEM_RECORDS) {
            comp_id = (uint32_t)rec_plane[rr].x;
        } else {
            const uint32_t ci = row0 + rr - raw_union_count;
            comp_id = (uint32_t)union_records_tile[ci].x;
        }
        const half *src = comp_kv + (uint64_t)comp_id * kTTHeadDim;
        tt_issue_cp_async_row(dst, rr, lane16, src, true);
    }

    if (active && rr >= nr) {
        tt_issue_cp_async_row(dst, rr, lane16, NULL, false);
    }
}



__device__ __forceinline__ void tt_load_score_q_frag(
        uint32_t (&q_frag)[kTTScoreKStepsPerQuarter][4],
        const float * __restrict__ q,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t tile_base,
        uint32_t head_base) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kScoreWarps = kMtiles * kTTScoreKQuarters;
    const uint32_t warp = tt_warp_id();
    if (warp >= kScoreWarps) {
        return;
    }

    const uint32_t mtile = warp >> 2u;
    const uint32_t kq = warp & 3u;
    const uint32_t k_base = kq * kTTScoreKSliceDim;
    const uint32_t lane = tt_lane_id();
    const uint32_t a_group = lane >> 2u;
    const uint32_t a_col_pair = (lane & 3u) << 1u;
#pragma unroll
    for (uint32_t kt = 0; kt < kTTScoreKStepsPerQuarter; ++kt) {
        const uint32_t k0 = k_base + kt * 16u;
#pragma unroll
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint32_t m = mtile * 16u + a_group + ((r & 1u) ? 8u : 0u);
            const uint32_t tok = m / kTTG;
            const uint32_t h = m - tok * kTTG;
            const uint32_t gt = tile_base + tok;
            const uint32_t gh = head_base + h;
            const uint32_t d = k0 + a_col_pair + ((r & 2u) ? 8u : 0u);
            float x0 = 0.0f;
            float x1 = 0.0f;
            if (gt < n_tokens && gh < n_head) {
                const float *q_row = q + ((uint64_t)gt * n_head + gh) * kTTHeadDim;
                x0 = q_row[d];
                x1 = q_row[d + 1u];
            }
            const half2 packed = __floats2half2_rn(x0, x1);
            q_frag[kt][r] =
                (uint32_t)__half_as_ushort(__low2half(packed)) |
                ((uint32_t)__half_as_ushort(__high2half(packed)) << 16);
        }
    }
}



template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_hmma_score_stage(
        float4 * __restrict__ partial_scores,
        const uint32_t (&q_frag)[kTTScoreKStepsPerQuarter][4],
        const half * __restrict__ kv_cur,
        uint32_t nr,
        float score_scale) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kScoreWarps = kMtiles * kTTScoreKQuarters;
    constexpr uint32_t kNtiles = TT_STAGE_ROWS / 8u;
    const uint32_t warp = tt_warp_id();
    if (warp >= kScoreWarps) {
        return;
    }

    const uint32_t mtile = warp >> 2u;
    const uint32_t kq = warp & 3u;
    const uint32_t lane = tt_lane_id();
    const unsigned kv_smem = tt_smem_addr(kv_cur);
    const uint32_t score_row_lane = lane & 7u;
    const uint32_t score_chunk_lane = (lane >> 3u) & 1u;
    const uint32_t score_chunk_base = kq * (kTTScoreKSliceDim / 8u) + score_chunk_lane;
#pragma unroll
    for (uint32_t ntile = 0; ntile < kNtiles; ++ntile) {
        const uint32_t row_base = ntile * 8u;
        if (row_base < nr) {
            float s_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const uint32_t score_row = row_base + score_row_lane;
#pragma unroll
            for (uint32_t kt = 0; kt < kTTScoreKStepsPerQuarter; ++kt) {
                uint32_t b[2];
                tt_ldmatrix_x2_addr(
                    b,
                    kv_smem + tt_ring_off_bytes(score_row, score_chunk_base + kt * 2u));
                tt_mma_m16n8k16_f16_f32(s_frag, q_frag[kt], b);
            }
            const uint32_t m0 = mtile * 16u + (lane >> 2u);
            const uint32_t r0 = row_base + ((lane & 3u) << 1u);
            const uint32_t r1 = r0 + 1u;
            const uint32_t m1 = m0 + 8u;
            if (r0 < nr) {
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m0, r0, s_frag[0] * score_scale);
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m1, r0, s_frag[2] * score_scale);
            }
            if (r1 < nr) {
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m0, r1, s_frag[1] * score_scale);
                tt_store_score_partial<TT_STAGE_ROWS>(
                    partial_scores, kq, m1, r1, s_frag[3] * score_scale);
            }
        }
    }
}



template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_softmax_stage(
        half * __restrict__ probs,
        float * __restrict__ stage_rescale,
        float * __restrict__ max_s,
        float * __restrict__ sum_s,
        const float4 * __restrict__ scores,
        const int2 * __restrict__ records,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        uint32_t tile_count,
        uint32_t tile_base,
        uint32_t raw_row_min) {
    constexpr uint32_t kMPerWarp = kTTM / kTTWarps;
    constexpr uint32_t kProbStride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    static_assert(kTTM % kTTWarps == 0u, "softmax maps integral m rows per warp");
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    const uint32_t sr = row0 + lane;
    const bool lane_live = lane < TT_STAGE_ROWS && lane < nr;
    const bool raw_slot = lane_live && sr < raw_union_count;
    uint16_t comp_mask = 0u;
    if (lane_live && !raw_slot) {
        comp_mask = (uint16_t)records[lane].y;
    }
    const uint32_t prob_lane_base = warp * kProbStride + lane;

#pragma unroll
    for (uint32_t mi = 0; mi < kMPerWarp; ++mi) {
        const uint32_t m = warp + mi * kTTWarps;
        const uint32_t tok = m / kTTG;
        const bool valid_token = tok < tile_count;
        const uint32_t score_idx = m * TT_STAGE_ROWS + lane;
        const uint32_t prob_idx = prob_lane_base + mi * kTTWarps * kProbStride;
        float score = -INFINITY;
        if (lane_live && valid_token) {
            const bool selected = raw_slot
                ? (((uint32_t)(sr - tok) < kTTRawWindow) && (tile_base + sr >= raw_row_min))
                : ((comp_mask & (uint16_t)(1u << tok)) != 0u);
            if (selected) {
                const float4 parts = tt_load_score_partial_record(scores + score_idx);
                const float s01 = parts.x + parts.y;
                const float s23 = parts.z + parts.w;
                score = s01 + s23;
            }
        }

        const float stage_m = tt_warp_max_f32(score);
        const float old_m = max_s[m];
        const float new_m = fmaxf(old_m, stage_m);
        float old_scale = 1.0f;
        if (new_m != -INFINITY) {
            old_scale = (old_m == -INFINITY) ? 0.0f : expf(old_m - new_m);
        }
        const float row_scale = (score == -INFINITY || new_m == -INFINITY)
            ? 0.0f
            : expf(score - new_m);
        const float stage_sum = tt_warp_sum_f32(row_scale);

        if (lane < TT_STAGE_ROWS) {
            probs[prob_idx] = __float2half(lane < nr ? row_scale : 0.0f);
        }
        if (lane == 0u) {
            max_s[m] = new_m;
            sum_s[m] = sum_s[m] * old_scale + stage_sum;
            stage_rescale[m] = old_scale;
        }
    }
}



template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_pv_mma_stage(
        float (&o_acc)[2u * kTTTileTokens * kTTG],
        const half * __restrict__ probs,
        const float * __restrict__ stage_rescale,
        const half * __restrict__ kv_cur) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kPvWarpBase = 8u;
    constexpr uint32_t kPvNTiles = 8u;
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    if (warp < kPvWarpBase) {
        return;
    }
    const uint32_t pv_warp = warp - kPvWarpBase;

#pragma unroll
    for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
        const float *scale0 = stage_rescale + mtile * 16u + (lane >> 2u);
        const float *scale1 = scale0 + 8u;
        const float rs0 = *scale0;
        const float rs1 = *scale1;
#pragma unroll
        for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
            const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2);
            o_acc[idx + 0u] *= rs0;
            o_acc[idx + 1u] *= rs0;
            o_acc[idx + 2u] *= rs1;
            o_acc[idx + 3u] *= rs1;
        }
    }

    constexpr uint32_t kProbStride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    constexpr unsigned kPvAStepBytes = 16u * sizeof(half);
    constexpr unsigned kPvMtileBytes = 16u * kProbStride * sizeof(half);
    const unsigned probs_lane_base =
        tt_smem_addr(probs) +
        (unsigned)(((lane & 15u) * (kProbStride / 2u) +
                    (lane >> 4u) * 4u) * sizeof(uint32_t));
    const unsigned kv_smem = tt_smem_addr(kv_cur);
    const uint32_t pv_row_lane = lane & 15u;
    const uint32_t pv_chunk_base = pv_warp * kPvNTiles;
#pragma unroll
    for (uint32_t kt = 0; kt < TT_STAGE_ROWS / 16u; ++kt) {
        const unsigned probs_kt_base = probs_lane_base + (unsigned)(kt * kPvAStepBytes);
        const uint32_t pv_row = kt * 16u + pv_row_lane;
#pragma unroll
        for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
            uint32_t a[4];
            tt_ldmatrix_x4_addr(a, probs_kt_base + (unsigned)(mtile * kPvMtileBytes));
#pragma unroll
            for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
                uint32_t b[2];
                const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2);
                tt_ldmatrix_x2_trans_addr(
                    b,
                    kv_smem + tt_ring_off_bytes(pv_row, pv_chunk_base + ntile));
                tt_mma_m16n8k16_f16_f32(o_acc + idx, a, b);
            }
        }
    }
}



__device__ __forceinline__ void tt_pv_mma_epilogue(
        const float (&o_acc)[2u * kTTTileTokens * kTTG],
        const float * __restrict__ final_scale,
        float * __restrict__ heads,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t tile_base,
        uint32_t head_base) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kPvWarpBase = 8u;
    constexpr uint32_t kPvNTiles = 8u;
    const uint32_t lane = tt_lane_id();
    const uint32_t warp = tt_warp_id();
    if (warp < kPvWarpBase) {
        return;
    }
    const uint32_t pv_warp = warp - kPvWarpBase;

#pragma unroll
    for (uint32_t mtile = 0; mtile < kMtiles; ++mtile) {
#pragma unroll
        for (uint32_t ntile = 0; ntile < kPvNTiles; ++ntile) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const uint32_t idx = ((mtile * kPvNTiles + ntile) << 2) + (uint32_t)l;
                const uint32_t m = mtile * 16u + (uint32_t)tt_mma_c_i(lane, l);
                const uint32_t tok = m / kTTG;
                const uint32_t h = m - tok * kTTG;
                const uint32_t gt = tile_base + tok;
                const uint32_t gh = head_base + h;
                const uint32_t d =
                    pv_warp * 64u + ntile * 8u + (uint32_t)tt_mma_c_j(lane, l);
                if (gt < n_tokens && gh < n_head) {
                    heads[((uint64_t)gt * n_head + gh) * kTTHeadDim + d] =
                        o_acc[idx] * final_scale[m];
                }
            }
        }
    }
}



__global__ static void __launch_bounds__(512, 1) attention_tokentile_hmma_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const half *raw_kv,
        const half *comp_kv,
        const int2 *union_records,
        const uint32_t *union_counts,
        uint32_t rec_stride,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t raw_row_min) {
    constexpr uint32_t kKvElems = tt_TokentileLayout<kTTStageRows>::ring_plane_elems;
    constexpr uint32_t kProbStride = tt_TokentileLayout<kTTStageRows>::prob_stride;
    const uint32_t tid = threadIdx.x;
    const uint32_t tile_idx = blockIdx.x;
    const uint32_t head_group = blockIdx.y;
    const uint32_t tile_base = tile_idx * kTTTileTokens;
    const uint32_t head_base = head_group * kTTG;
    if (tile_base >= n_tokens) {
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens ? n_tokens - tile_base : kTTTileTokens;
    const uint32_t raw_union_count = tile_count + kTTRawWindow - 1u;
    const uint32_t comp_union_count = union_counts[tile_idx];
    const uint32_t n_score = raw_union_count + comp_union_count;
    const uint64_t union_tile_off = (uint64_t)tile_idx * rec_stride;
    const int2 * __restrict__ union_records_tile = union_records + union_tile_off;
    const float score_scale = rsqrtf((float)kTTHeadDim);

    extern __shared__ unsigned char smem[];
    unsigned char *p = tt_align16(smem);
    half *kv_h = reinterpret_cast<half *>(p);
    p = tt_align16(p + 2u * tt_TokentileLayout<kTTStageRows>::ring_plane_bytes);
    half *probs = reinterpret_cast<half *>(p);
    p = tt_align16(p + 2u * kTTM * kProbStride * sizeof(half));
    float4 *score_scratch = reinterpret_cast<float4 *>(p);
    p = tt_align16(p + kTTM * kTTStageRows * sizeof(float4));
    float *max_s = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *sum_s = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *stage_rescale = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *final_scale = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    int2 *rec_ring = reinterpret_cast<int2 *>(p);
    p = tt_align16(p + kTTRecordRingPlanes * kTTStageRows * sizeof(int2));

    for (uint32_t m = tid; m < kTTM; m += blockDim.x) {
        max_s[m] = -INFINITY;
        sum_s[m] = 0.0f;
        stage_rescale[m] = 1.0f;
        final_scale[m] = 0.0f;
    }
    __syncthreads();

    union tt_TokentileRoleRegs {
        uint32_t score_q_frag[kTTScoreKStepsPerQuarter][4];
        float o_acc[2u * kTTTileTokens * kTTG];
    };
    tt_TokentileRoleRegs role_regs;
    tt_load_score_q_frag(
        role_regs.score_q_frag, q, n_tokens, n_head, tile_base, head_base);
    if (tt_warp_id() >= 8u) {
#pragma unroll
        for (uint32_t i = 0; i < 2u * kTTM; ++i) {
            role_regs.o_acc[i] = 0.0f;
        }
    }

    uint32_t cur = 0u;
    uint32_t free = 1u;
    if (n_score != 0u) {
        const uint32_t nr0 = n_score < kTTStageRows ? n_score : kTTStageRows;
        tt_issue_kv_stage_cp_async<kTTStageRows, false>(
            kv_h + cur * kKvElems,
            rec_ring,
            0u,
            nr0,
            raw_union_count,
            union_records_tile,
            tile_base,
            raw_kv,
            comp_kv,
            tid);
        tt_issue_record_stage_cp_async<kTTStageRows>(
            rec_ring,
            0u,
            nr0,
            raw_union_count,
            union_records_tile);
        if (kTTStageRows < n_score) {
            const uint32_t nr1 =
                n_score - kTTStageRows < kTTStageRows ? n_score - kTTStageRows : kTTStageRows;
            tt_issue_record_stage_cp_async<kTTStageRows>(
                rec_ring + kTTStageRows,
                kTTStageRows,
                nr1,
                raw_union_count,
                union_records_tile);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
    }
    __syncthreads();

    uint32_t prob_cur = 0u;
    for (uint32_t row0 = 0u; row0 < n_score; row0 += kTTStageRows) {
        const uint32_t nr = n_score - row0 < kTTStageRows ? n_score - row0 : kTTStageRows;
        half *kv_cur = kv_h + cur * kKvElems;
        half *kv_free = kv_h + free * kKvElems;

        tt_hmma_score_stage<kTTStageRows>(
            score_scratch, role_regs.score_q_frag, kv_cur, nr, score_scale);
        if (row0 != 0u) {
            const half *kv_prev = kv_h + (cur ^ 1u) * kKvElems;
            const half *probs_prev = probs + (prob_cur ^ 1u) * kTTM * kProbStride;
            tt_pv_mma_stage<kTTStageRows>(
                role_regs.o_acc, probs_prev, stage_rescale, kv_prev);
        }
        __syncthreads();

        const uint32_t next_row0 = row0 + kTTStageRows;
        const bool has_next = next_row0 < n_score;
        if (has_next) {
            const uint32_t next_nr =
                n_score - next_row0 < kTTStageRows ? n_score - next_row0 : kTTStageRows;
            tt_issue_kv_stage_cp_async<kTTStageRows, true>(
                kv_free,
                rec_ring + (((row0 / kTTStageRows) + 1u) & 3u) * kTTStageRows,
                next_row0,
                next_nr,
                raw_union_count,
                union_records_tile,
                tile_base,
                raw_kv,
                comp_kv,
                tid);
            const uint32_t prefetch_row0 = next_row0 + kTTStageRows;
            if (prefetch_row0 < n_score) {
                const uint32_t prefetch_nr =
                    n_score - prefetch_row0 < kTTStageRows
                        ? n_score - prefetch_row0
                        : kTTStageRows;
                tt_issue_record_stage_cp_async<kTTStageRows>(
                    rec_ring + (((row0 / kTTStageRows) + 2u) & 3u) * kTTStageRows,
                    prefetch_row0,
                    prefetch_nr,
                    raw_union_count,
                    union_records_tile);
            }
            tt_cp_async_commit();
        }

        half *probs_cur = probs + prob_cur * kTTM * kProbStride;
        tt_softmax_stage<kTTStageRows>(
            probs_cur,
            stage_rescale,
            max_s,
            sum_s,
            score_scratch,
            rec_ring + ((row0 / kTTStageRows) & 3u) * kTTStageRows,
            row0,
            nr,
            raw_union_count,
            tile_count,
            tile_base,
            raw_row_min);
        tt_cp_async_wait_group<0>();
        __syncthreads();
        cur ^= 1u;
        free ^= 1u;
        prob_cur ^= 1u;
    }

    if (n_score != 0u) {
        const half *kv_prev = kv_h + (cur ^ 1u) * kKvElems;
        const half *probs_prev = probs + (prob_cur ^ 1u) * kTTM * kProbStride;
        tt_pv_mma_stage<kTTStageRows>(
            role_regs.o_acc, probs_prev, stage_rescale, kv_prev);
    }
    __syncthreads();

    for (uint32_t m = tid; m < kTTM; m += blockDim.x) {
        const uint32_t tok = m / kTTG;
        const uint32_t h = m - tok * kTTG;
        const uint32_t gt = tile_base + tok;
        const uint32_t gh = head_base + h;
        if (gt < n_tokens && gh < n_head) {
            const float sink = sinks[gh];
            const float old_m = max_s[m];
            const float new_m = fmaxf(old_m, sink);
            const float old_scale = old_m == -INFINITY ? 0.0f : expf(old_m - new_m);
            const float sink_scale = expf(sink - new_m);
            const float den = sum_s[m] * old_scale + sink_scale;
            final_scale[m] = den == 0.0f ? 0.0f : old_scale / den;
        } else {
            final_scale[m] = 0.0f;
        }
    }
    __syncthreads();

    tt_pv_mma_epilogue(
        role_regs.o_acc, final_scale, heads, n_tokens, n_head, tile_base, head_base);
}



__global__ static void __launch_bounds__(256, 4)
attention_decode_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;
    __shared__ float4 kv_shared[4 * 128];

    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t comp_count = 0;
    if (n_comp != 0u) {
        if (n_tokens == 1u && ratio == 0u) {
            comp_count = n_comp;
        } else if (ratio != 0u) {
            comp_count = (qpos + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    if (threadIdx.x == 0) {
        uint32_t raw_count = 0;
        uint32_t raw_first_idx = 0;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
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
        raw_count_s = raw_count;
        raw_first_idx_s = raw_first_idx;
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}



__device__ static void hc4_split_one(float *out, const float *mix, const float *scale, const float *base, uint32_t sinkhorn_iters, float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; col++) {
        float s = epsv;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = epsv;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = epsv;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[8 + i] = c[i];
}



__global__ static void hc_split_sinkhorn_kernel(float *out, const float *mix, const float *scale, const float *base, uint32_t n_rows, uint32_t sinkhorn_iters, float epsv) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    hc4_split_one(out + (uint64_t)row * 24, mix + (uint64_t)row * 24, scale, base, sinkhorn_iters, epsv);
}



__global__ static void hc_weighted_sum_kernel(float *out, const float *x, const float *w, uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens, uint32_t weight_stride_f32) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_tokens;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint32_t t = gid / n_embd;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += x[(uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d] *
               w[(uint64_t)t * weight_stride_f32 + h];
    }
    out[(uint64_t)t * n_embd + d] = acc;
}



__global__ static void hc_expand_kernel(
        float *out_hc,
        const float *block_out,
        const float *block_add,
        const float *block_add2,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add,
        int has_add2) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) {
        float add_v = block_add[(uint64_t)t * n_embd + d];
        if (has_add2) add_v += block_add2[(uint64_t)t * n_embd + d];
        block_v += add_v;
    }
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}



__global__ static void hc_split_weighted_sum_fused_kernel(
        float *out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv) {
    uint32_t t = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
    }
}



__global__ static void hc_split_weighted_sum_norm_fused_kernel(
        float *out,
        float *norm_out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        const float *norm_w,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
        sum += acc * acc;
    }

    __shared__ float partial[256];
    partial[d] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (d < stride) partial[d] += partial[d + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)n_embd + norm_eps);
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        const float v = out[(uint64_t)t * n_embd + col];
        norm_out[(uint64_t)t * n_embd + col] = v * norm_scale * norm_w[col];
    }
}



__global__ static void output_hc_weights_kernel(
        float *out,
        const float *pre,
        const float *scale,
        const float *base,
        uint32_t n_hc,
        uint32_t n_tokens,
        float epsv) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_tokens * n_hc;
    if (gid >= n) return;
    uint32_t h = gid % n_hc;
    float z = pre[gid] * scale[0] + base[h];
    out[gid] = 1.0f / (1.0f + expf(-z)) + epsv;
}



__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}



__global__ static void compressor_store_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens) {
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    uint32_t t = gid / width;
    uint32_t j = gid - (uint64_t)t * width;
    uint32_t pos_mod = (pos0 + t) % ratio;
    uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    state_kv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
    state_score[(uint64_t)dst_row * width + j] =
        sc[(uint64_t)t * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)pos_mod * width + j);
}



__global__ static void compressor_set_rows_kernel(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t src0,
        uint32_t dst0,
        uint32_t rows) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)rows * width;
    if (gid >= n) return;
    uint32_t r = gid / width;
    uint32_t j = gid - (uint64_t)r * width;
    uint32_t src = src0 + r;
    uint32_t dst = dst0 + r;
    uint32_t phase = (pos0 + src) % ratio;
    state_kv[(uint64_t)dst * width + j] = kv[(uint64_t)src * width + j];
    state_score[(uint64_t)dst * width + j] =
        sc[(uint64_t)src * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)phase * width + j);
}



__global__ static void compressor_prefill_pool_kernel(
        float *comp,
        const float *kv,
        const float *sc,
        const float *state_kv,
        const float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_comp,
        uint32_t replay) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = blockIdx.y;
    if (d >= head_dim || c >= n_comp) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        if (replay && c == 0) {
            for (uint32_t r = 0; r < 4; r++) {
                vals[n_cand] = state_kv[(uint64_t)r * width + d];
                scores[n_cand] = state_score[(uint64_t)r * width + d];
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        } else if (c > 0) {
            uint32_t base = (c - 1u) * ratio;
            for (uint32_t r = 0; r < 4; r++) {
                uint32_t t = base + r;
                float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
                vals[n_cand] = kv[(uint64_t)t * width + d];
                scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        }
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < 4; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + head_dim + d);
            vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
            scores[n_cand] = sc[(uint64_t)t * width + head_dim + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < ratio; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
            vals[n_cand] = kv[(uint64_t)t * width + d];
            scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    comp[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
}



__global__ static void compressor_update_pool_kernel(
        float *row,
        const float *state_kv,
        const float *state_score,
        uint32_t head_dim,
        uint32_t ratio) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
            scores[n_cand] = state_score[(uint64_t)(ratio + r) * width + head_dim + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    row[d] = den != 0.0f ? acc / den : 0.0f;
}



__global__ static void compressor_shift_ratio4_kernel(float *state_kv, float *state_score, uint32_t width) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t half = 4ull * width;
    if (i >= half) return;
    float v = state_kv[half + i];
    float s = state_score[half + i];
    state_kv[i] = v;
    state_score[i] = s;
    state_kv[half + i] = v;
    state_score[half + i] = s;
}



__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}



__global__ static void router_select_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *log = logits + (uint64_t)t * 256;
    float *prob = probs + (uint64_t)t * 256;
    int32_t *sel = selected + (uint64_t)t * 6;
    float *w = weights + (uint64_t)t * 6;

    for (int i = 0; i < 256; i++) prob[i] = sqrtf(softplus_dev(log[i]));

    if (hash_mode) {
        int32_t tok = tokens ? tokens[t] : token_scalar;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * 6;
        for (int i = 0; i < 6; i++) sel[i] = row[i];
    } else {
        for (int i = 0; i < 6; i++) sel[i] = -1;
        for (int i = 0; i < 256; i++) {
            float score = prob[i] + (has_bias ? bias[i] : 0.0f);
            for (int j = 0; j < 6; j++) {
                if (sel[j] < 0 || score > prob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (int k = 5; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = i;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < 6; i++) {
        int e = sel[i];
        float v = (e >= 0 && e < 256) ? prob[e] : 0.0f;
        w[i] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int i = 0; i < 6; i++) w[i] = w[i] / sum * 1.5f;
}



__global__ static void router_select_parallel_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
    uint32_t t = blockIdx.x;
    uint32_t i = threadIdx.x;
    if (t >= n_tokens || i >= 256u) return;
    const float *log = logits + (uint64_t)t * 256;
    float *prob = probs + (uint64_t)t * 256;
    int32_t *sel = selected + (uint64_t)t * 6;
    float *w = weights + (uint64_t)t * 6;
    __shared__ float sprob[256];

    const float p = sqrtf(softplus_dev(log[i]));
    sprob[i] = p;
    prob[i] = p;
    __syncthreads();

    if (i != 0) return;
    if (hash_mode) {
        int32_t tok = tokens ? tokens[t] : token_scalar;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * 6;
        for (int j = 0; j < 6; j++) sel[j] = row[j];
    } else {
        for (int j = 0; j < 6; j++) sel[j] = -1;
        for (int e = 0; e < 256; e++) {
            float score = sprob[e] + (has_bias ? bias[e] : 0.0f);
            for (int j = 0; j < 6; j++) {
                if (sel[j] < 0 || score > sprob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (int k = 5; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = e;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int j = 0; j < 6; j++) {
        int e = sel[j];
        float v = (e >= 0 && e < 256) ? sprob[e] : 0.0f;
        w[j] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int j = 0; j < 6; j++) w[j] = w[j] / sum * 1.5f;
}



__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}



__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * 256u;
    float *prob = probs + (uint64_t)t * 256u;
    int32_t *sel = selected + (uint64_t)t * 6u;
    float *w = weights + (uint64_t)t * 6u;
    __shared__ float sprob[4][256];
    float local_prob[8];
    float local_score[8];

    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = sqrtf(softplus_dev(log[e]));
        local_prob[j] = p;
        local_score[j] = p + (has_bias ? bias[e] : 0.0f);
        sprob[row_in_block][e] = p;
        prob[e] = p;
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0) {
            int32_t tok = tokens ? tokens[t] : token_scalar;
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * 6u;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) {
                const int32_t e = row[j];
                sel[j] = e;
                const float v = (e >= 0 && e < 256) ? sprob[row_in_block][(uint32_t)e] : 0.0f;
                w[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
        }
        return;
    }

    float out_prob[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t out_idx[6] = {0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t k = 0; k < 6u; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) {
            sel[j] = (int32_t)out_idx[j];
            w[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
    }
}



__global__ static void swiglu_kernel(float *out, const float *gate, const float *up, uint32_t n, float clamp, float weight) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    float s = g / (1.0f + expf(-g));
    out[i] = s * u * weight;
}



__global__ static void add_kernel(float *out, const float *a, const float *b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}



__global__ static void directional_steering_project_kernel(
        float       *x,
        const float *directions,
        uint32_t     layer,
        uint32_t     width,
        uint32_t     rows,
        float        scale) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || width == 0) return;

    float *xr = x + (uint64_t)row * width;
    const float *dir = directions + (uint64_t)layer * width;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        sum += xr[i] * dir[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    const float coeff = scale * partial[0];
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        xr[i] -= coeff * dir[i];
    }
}



__global__ static void zero_kernel(float *out, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.0f;
}





/* Slow, transparent B1 oracle for the persistent 68-byte indexer ABI. It is
 * only selected by the A/B regression; production uses block-scaled MMA. */
__global__ static void spark_indexer_score_one_reference_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const unsigned char *index_comp,
        uint32_t n_comp,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal) {
    const uint32_t c = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || tid >= 128u) return;
    if (causal) {
        const uint32_t visible = ratio ? (pos0 + 1u) / ratio : n_comp;
        if (c >= visible) {
            if (tid == 0) scores[c] = -INFINITY;
            return;
        }
    }

    __shared__ float krow[128];
    __shared__ float partial[4];
    const unsigned char *packed = index_comp +
        (uint64_t)c * DS4_SPARK_INDEX_ROW_BYTES;
    krow[tid] = spark_index_decode(packed, tid);
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum_f32(dot);
        if (lane == 0) {
            partial[warp] = fmaxf(dot, 0.0f) * weights[h] * scale;
        }
        __syncthreads();
        if (tid == 0) {
            total += partial[0] + partial[1] + partial[2] + partial[3];
        }
        __syncthreads();
    }
    if (tid == 0) scores[c] = total;
}



#endif  // DS4X_BACKEND_ATTENTION_DECODE_KERNELS_CUH
