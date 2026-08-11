#ifdef DS4_CUDA_HAVE_MXF4
#define DS4_MXF4_INDEXER_HEADS 64u
#define DS4_MXF4_INDEXER_DIM 128u

__device__ static inline uint32_t indexer_mxf4_nearest_level(float v) {
    if (v <= 0.25f) return 0u;
    if (v <= 0.75f) return 1u;
    if (v <= 1.25f) return 2u;
    if (v <= 1.75f) return 3u;
    if (v <= 2.5f) return 4u;
    if (v <= 3.5f) return 5u;
    if (v <= 5.0f) return 6u;
    return 7u;
}

/* Encode rows that have already passed the model's E2M1 QAT. The packed
 * exponent word is consumed directly by the SM121 block-scaled MMA. */
__global__ static void indexer_mxf4_encode_rows_kernel(
        const float * __restrict__ src,
        unsigned char * __restrict__ codes,
        uint32_t * __restrict__ scales,
        uint32_t rows) {
    const uint32_t row =
        (blockIdx.x * blockDim.x + threadIdx.x) >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= rows) return;

    const float *xr = src + (uint64_t)row * DS4_MXF4_INDEXER_DIM;
    float values[4];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        values[i] = xr[lane * 4u + i];
        amax = fmaxf(amax, fabsf(values[i]));
    }
#pragma unroll
    for (uint32_t off = 4u; off != 0u; off >>= 1u) {
        amax = fmaxf(amax,
                     __shfl_xor_sync(0xffffffffu, amax, off, 8));
    }

    uint32_t exponent = 127u;
    if (amax > 0.0f) {
        const uint32_t bits = __float_as_uint(amax / 6.0f);
        exponent = (bits >> 23u) & 0xffu;
        if ((bits & 0x7fffffu) != 0u) exponent++;
        exponent = min(max(exponent, 1u), 254u);
    }
    const float reciprocal =
        __uint_as_float((254u - exponent) << 23u);
    uint32_t packed = 0u;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t level =
            indexer_mxf4_nearest_level(fabsf(values[i]) * reciprocal);
        packed |= (level | (values[i] < 0.0f ? 8u : 0u)) << (4 * i);
    }
    ((uint16_t *)codes)[(uint64_t)row * 32u + lane] = (uint16_t)packed;

    const uint32_t e1 = __shfl_sync(0xffffffffu, exponent, 8);
    const uint32_t e2 = __shfl_sync(0xffffffffu, exponent, 16);
    const uint32_t e3 = __shfl_sync(0xffffffffu, exponent, 24);
    if (lane == 0u) {
        scales[row] = exponent | (e1 << 8u) | (e2 << 16u) | (e3 << 24u);
    }
}

__device__ static inline void indexer_mxf4_mma(
        float out[4],
        const uint32_t a[4],
        uint32_t b0,
        uint32_t b1,
        uint32_t scale_a,
        uint32_t scale_b) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {0, 0}, {%11}, {0, 0};\n"
        : "+f"(out[0]), "+f"(out[1]), "+f"(out[2]), "+f"(out[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b));
}

__device__ __forceinline__ static void indexer_cp_async_u32(
        uint32_t *dst,
        const void *src,
        bool valid) {
    const unsigned smem = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                 :: "r"(smem), "l"(src), "r"(valid ? 4u : 0u));
}

__device__ __forceinline__ static void indexer_cp_commit(void) {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ static void indexer_cp_wait(void) {
    asm volatile("cp.async.wait_group 0;\n");
}

/* CUDA SM121 implementation adapted from entrpi/batched-serving's v3
 * scorer and Marco Palaferri's original 16-token staging geometry. */
__global__ static void __launch_bounds__(512) indexer_scores_mxf4_kernel(
        float *scores,
        const unsigned char * __restrict__ q_codes,
        const uint32_t * __restrict__ q_scales,
        const float * __restrict__ weights,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t ratio,
        uint32_t causal,
        float scale,
        const unsigned char * __restrict__ k_codes,
        const uint32_t * __restrict__ k_scales,
        uint32_t k_row_bytes,
        uint32_t k_scale_offset) {
    constexpr uint32_t HEAD_GROUP = 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t lane_in_group = lane & 3u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tile_c = blockIdx.x * 256u;

    const uint32_t last_token = min(tile_t + 16u, n_tokens);
    const uint32_t max_visible = causal
        ? (last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp) : 0u)
        : n_comp;
    if (tile_c >= max_visible) {
        for (uint32_t i = tid; i < 16u * 256u; i += 512u) {
            const uint32_t row = i >> 8u;
            const uint32_t col = i & 255u;
            const uint32_t token = tile_t + row;
            const uint32_t comp = tile_c + col;
            if (token < n_tokens && comp < n_comp) {
                scores[(uint64_t)token * n_comp + comp] = -INFINITY;
            }
        }
        return;
    }

    __shared__ __align__(16) uint32_t q_shared[2][16][8][32];
    __shared__ uint32_t scale_shared[2][16][16];
    __shared__ float weight_shared[2][16][16];

    uint32_t b_frag[2][4];
    uint32_t scale_b[2][2];
#pragma unroll
    for (uint32_t cg = 0; cg < 2u; cg++) {
        const uint32_t col =
            tile_c + (cg * 16u + warp) * 8u + (lane >> 2u);
        if (col < n_comp) {
            const uint32_t *words =
                (const uint32_t *)(k_codes + (uint64_t)col * k_row_bytes);
            b_frag[cg][0] = words[lane_in_group];
            b_frag[cg][1] = words[4u + lane_in_group];
            b_frag[cg][2] = words[8u + lane_in_group];
            b_frag[cg][3] = words[12u + lane_in_group];
            const uint32_t packed_scale = k_scales
                ? k_scales[col]
                : *(const uint32_t *)(k_codes +
                    (uint64_t)col * k_row_bytes + k_scale_offset);
            scale_b[cg][0] = packed_scale & 0xffffu;
            scale_b[cg][1] = packed_scale >> 16u;
        } else {
            b_frag[cg][0] = b_frag[cg][1] = 0u;
            b_frag[cg][2] = b_frag[cg][3] = 0u;
            scale_b[cg][0] = scale_b[cg][1] = 127u | (127u << 8u);
        }
    }

    auto stage = [&](uint32_t head0, uint32_t buffer) {
        for (uint32_t i = tid; i < HEAD_GROUP * 16u * 17u; i += 512u) {
            const uint32_t hh = i / (16u * 17u);
            const uint32_t rem = i - hh * (16u * 17u);
            const uint32_t row = rem / 17u;
            const uint32_t word = rem - row * 17u;
            const uint32_t token = tile_t + row;
            const bool valid = token < n_tokens;
            const uint64_t qrow =
                (uint64_t)token * DS4_MXF4_INDEXER_HEADS + head0 + hh;
            const void *src;
            uint32_t *dst;
            if (word < 16u) {
                src = valid
                    ? (const void *)(q_codes + qrow * 64u + word * 4u)
                    : (const void *)q_codes;
                dst = &q_shared[buffer][hh][row >> 1u]
                    [(row & 1u) * 16u + (word & 3u) * 4u + (word >> 2u)];
            } else {
                src = valid ? (const void *)(q_scales + qrow)
                            : (const void *)q_scales;
                dst = &scale_shared[buffer][hh][row];
            }
            indexer_cp_async_u32(dst, src, valid);
        }
        for (uint32_t i = tid; i < HEAD_GROUP * 16u; i += 512u) {
            const uint32_t hh = i >> 4u;
            const uint32_t row = i & 15u;
            const uint32_t token = tile_t + row;
            const bool valid = token < n_tokens;
            const void *src = valid
                ? (const void *)(weights +
                    (uint64_t)token * DS4_MXF4_INDEXER_HEADS + head0 + hh)
                : (const void *)weights;
            indexer_cp_async_u32(
                (uint32_t *)&weight_shared[buffer][hh][row], src, valid);
        }
    };

    stage(0u, 0u);
    indexer_cp_commit();
    indexer_cp_wait();
    __syncthreads();

    float accum[2][4];
#pragma unroll
    for (uint32_t cg = 0; cg < 2u; cg++) {
        accum[cg][0] = accum[cg][1] = 0.0f;
        accum[cg][2] = accum[cg][3] = 0.0f;
    }

    const uint32_t row0 = lane >> 2u;
    const uint32_t scale_row = (lane & 1u) * 8u + row0;
    for (uint32_t group = 0;
         group < DS4_MXF4_INDEXER_HEADS / HEAD_GROUP;
         group++) {
        const uint32_t current = group & 1u;
        if (group + 1u < DS4_MXF4_INDEXER_HEADS / HEAD_GROUP) {
            stage((group + 1u) * HEAD_GROUP, current ^ 1u);
            indexer_cp_commit();
        }
#pragma unroll
        for (uint32_t hh = 0; hh < HEAD_GROUP; hh++) {
            const uint4 a0 = *reinterpret_cast<const uint4 *>(
                &q_shared[current][hh][row0 >> 1u]
                    [(row0 & 1u) * 16u + lane_in_group * 4u]);
            const uint4 a1 = *reinterpret_cast<const uint4 *>(
                &q_shared[current][hh][(row0 + 8u) >> 1u]
                    [(row0 & 1u) * 16u + lane_in_group * 4u]);
            const uint32_t packed_scale =
                scale_shared[current][hh][scale_row];
            const uint32_t scale_a0 = packed_scale & 0xffffu;
            const uint32_t scale_a1 = packed_scale >> 16u;
            const float w0 = weight_shared[current][hh][row0];
            const float w1 = weight_shared[current][hh][row0 + 8u];
            const uint32_t a_lo[4] = {a0.x, a1.x, a0.y, a1.y};
            const uint32_t a_hi[4] = {a0.z, a1.z, a0.w, a1.w};
#pragma unroll
            for (uint32_t cg = 0; cg < 2u; cg++) {
                float product[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                indexer_mxf4_mma(product, a_lo,
                                 b_frag[cg][0], b_frag[cg][1],
                                 scale_a0, scale_b[cg][0]);
                indexer_mxf4_mma(product, a_hi,
                                 b_frag[cg][2], b_frag[cg][3],
                                 scale_a1, scale_b[cg][1]);
                accum[cg][0] += fmaxf(product[0], 0.0f) * w0;
                accum[cg][1] += fmaxf(product[1], 0.0f) * w0;
                accum[cg][2] += fmaxf(product[2], 0.0f) * w1;
                accum[cg][3] += fmaxf(product[3], 0.0f) * w1;
            }
        }
        if (group + 1u < DS4_MXF4_INDEXER_HEADS / HEAD_GROUP) {
            indexer_cp_wait();
            __syncthreads();
        }
    }

    const uint32_t token0 = tile_t + row0;
    const uint32_t token1 = token0 + 8u;
    const uint32_t visible0 = causal ? (pos0 + token0 + 1u) / ratio : n_comp;
    const uint32_t visible1 = causal ? (pos0 + token1 + 1u) / ratio : n_comp;
#pragma unroll
    for (uint32_t cg = 0; cg < 2u; cg++) {
        const uint32_t col0 = tile_c +
            (cg * 16u + warp) * 8u + lane_in_group * 2u;
#pragma unroll
        for (uint32_t j = 0; j < 2u; j++) {
            const uint32_t col = col0 + j;
            if (col >= n_comp) continue;
            if (token0 < n_tokens) {
                scores[(uint64_t)token0 * n_comp + col] =
                    col >= visible0 ? -INFINITY : accum[cg][j] * scale;
            }
            if (token1 < n_tokens) {
                scores[(uint64_t)token1 * n_comp + col] =
                    col >= visible1 ? -INFINITY : accum[cg][2u + j] * scale;
            }
        }
    }
}
#endif

#ifdef DS4_CUDA_HAVE_MXF4
static void *indexer_mxf4_scratch_alloc(uint64_t bytes) {
    if (g_indexer_mxf4_scratch_bytes >= bytes) {
        return g_indexer_mxf4_scratch;
    }

    int device = -1;
    if (!cuda_ok(cudaGetDevice(&device),
                 "get device for indexer mxf4 scratch")) {
        return NULL;
    }
    if (g_indexer_mxf4_scratch) {
        if (!cuda_ok(cudaDeviceSynchronize(),
                     "synchronize indexer mxf4 scratch growth")) {
            return NULL;
        }
        (void)cudaFree(g_indexer_mxf4_scratch);
        g_indexer_mxf4_scratch = NULL;
        g_indexer_mxf4_scratch_bytes = 0;
        g_indexer_mxf4_scratch_device = -1;
    }
    if (!cuda_ok(cudaMalloc(&g_indexer_mxf4_scratch, (size_t)bytes),
                 "allocate indexer mxf4 scratch")) {
        g_indexer_mxf4_scratch = NULL;
        return NULL;
    }
    g_indexer_mxf4_scratch_bytes = bytes;
    g_indexer_mxf4_scratch_device = device;
    return g_indexer_mxf4_scratch;
}
#endif

static int indexer_scores_spark_try(
        ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        uint32_t causal) {
#ifndef DS4_CUDA_HAVE_MXF4
    (void)scores; (void)q; (void)weights; (void)index_comp;
    (void)n_comp; (void)n_tokens; (void)pos0; (void)n_head;
    (void)head_dim; (void)ratio; (void)scale; (void)causal;
    return 0;
#else
    if (g_n_gpus != 1 || ds4_tensor_device_idx(scores) != 0 ||
        n_head != DS4_MXF4_INDEXER_HEADS ||
        head_dim != DS4_MXF4_INDEXER_DIM || n_comp == 0u ||
        n_tokens == 0u || (causal && ratio == 0u) ||
        index_comp->bytes < (uint64_t)n_comp * DS4_SPARK_INDEX_ROW_BYTES) {
        return 0;
    }
    if (n_tokens == 1u &&
        getenv("DS4_CUDA_SPARK_INDEXER_REFERENCE") != NULL) {
        spark_indexer_score_one_reference_kernel<<<n_comp, 128>>>(
            (float *)scores->ptr, (const float *)q->ptr,
            (const float *)weights->ptr,
            (const unsigned char *)index_comp->ptr,
            n_comp, pos0, ratio, scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(),
                       "Spark packed indexer reference launch") ? 1 : -1;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(0, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        return 0;
    }
    const uint64_t q_rows64 = (uint64_t)n_tokens * DS4_MXF4_INDEXER_HEADS;
    if (q_rows64 > UINT32_MAX) return 0;
    const uint32_t q_rows = (uint32_t)q_rows64;
    const uint64_t q_codes_bytes = (q_rows64 * 64u + 255u) & ~255ull;
    const uint64_t q_scales_bytes = (q_rows64 * 4u + 255u) & ~255ull;
    unsigned char *scratch = (unsigned char *)
        indexer_mxf4_scratch_alloc(q_codes_bytes + q_scales_bytes);
    if (!scratch) return 0;
    unsigned char *q_codes = scratch;
    uint32_t *q_scales = (uint32_t *)(scratch + q_codes_bytes);
    indexer_mxf4_encode_rows_kernel<<<
        (uint32_t)(((uint64_t)q_rows * 32u + 255u) / 256u), 256>>>(
            (const float *)q->ptr, q_codes, q_scales, q_rows);
    if (!cuda_ok(cudaGetLastError(), "Spark indexer Q encode launch")) return -1;
    dim3 grid((n_comp + 255u) / 256u, (n_tokens + 15u) / 16u, 1u);
    indexer_scores_mxf4_kernel<<<grid, 512>>>(
        (float *)scores->ptr, q_codes, q_scales,
        (const float *)weights->ptr, n_comp, n_tokens, pos0, ratio,
        causal, scale, (const unsigned char *)index_comp->ptr, NULL,
        DS4_SPARK_INDEX_ROW_BYTES, 64u);
    return cuda_ok(cudaGetLastError(), "Spark packed indexer score launch") ? 1 : -1;
#endif
}

/* Return 1 when the SM121 path handled the call, 0 to use the established
 * scorer, and -1 on a launch failure. */
static int indexer_scores_mxf4_try(
        ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale) {
#ifndef DS4_CUDA_HAVE_MXF4
    (void)scores; (void)q; (void)weights; (void)index_comp;
    (void)n_comp; (void)n_tokens; (void)pos0; (void)n_head;
    (void)head_dim; (void)ratio; (void)scale;
    return 0;
#else
    if (g_quality_mode || getenv("DS4_CUDA_NO_INDEXER_MXF4") != NULL ||
        g_n_gpus != 1 || ds4_tensor_device_idx(scores) != 0 ||
        n_head != DS4_MXF4_INDEXER_HEADS ||
        head_dim != DS4_MXF4_INDEXER_DIM || ratio != 4u ||
        n_comp < 1024u || n_tokens < 32u) {
        return 0;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(0, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        return 0;
    }

    const uint64_t q_rows64 =
        (uint64_t)n_tokens * DS4_MXF4_INDEXER_HEADS;
    if (q_rows64 > UINT32_MAX) return 0;
    const uint32_t q_rows = (uint32_t)q_rows64;
    const uint64_t k_codes_bytes = ((uint64_t)n_comp * 64u + 255u) & ~255ull;
    const uint64_t k_scales_bytes = ((uint64_t)n_comp * 4u + 255u) & ~255ull;
    const uint64_t q_codes_bytes = (q_rows64 * 64u + 255u) & ~255ull;
    const uint64_t q_scales_bytes = (q_rows64 * 4u + 255u) & ~255ull;
    const uint64_t scratch_bytes = k_codes_bytes + k_scales_bytes +
                                   q_codes_bytes + q_scales_bytes;
    unsigned char *scratch = (unsigned char *)
        indexer_mxf4_scratch_alloc(scratch_bytes);
    if (!scratch) return 0;

    unsigned char *k_codes = scratch;
    uint32_t *k_scales = (uint32_t *)(scratch + k_codes_bytes);
    unsigned char *q_codes = scratch + k_codes_bytes + k_scales_bytes;
    uint32_t *q_scales = (uint32_t *)(q_codes + q_codes_bytes);

    indexer_mxf4_encode_rows_kernel<<<
        (uint32_t)(((uint64_t)n_comp * 32u + 255u) / 256u), 256>>>(
            (const float *)index_comp->ptr, k_codes, k_scales, n_comp);
    if (!cuda_ok(cudaGetLastError(), "indexer mxf4 K encode launch")) {
        return -1;
    }
    indexer_mxf4_encode_rows_kernel<<<
        (uint32_t)(((uint64_t)q_rows * 32u + 255u) / 256u), 256>>>(
            (const float *)q->ptr, q_codes, q_scales, q_rows);
    if (!cuda_ok(cudaGetLastError(), "indexer mxf4 Q encode launch")) {
        return -1;
    }

    dim3 grid((n_comp + 255u) / 256u,
              (n_tokens + 15u) / 16u, 1u);
    indexer_scores_mxf4_kernel<<<grid, 512>>>(
        (float *)scores->ptr, q_codes, q_scales,
        (const float *)weights->ptr, n_comp, n_tokens, pos0, ratio,
        1u, scale, k_codes, k_scales, 64u, 0u);
    if (!cuda_ok(cudaGetLastError(), "indexer mxf4 score launch")) {
        return -1;
    }
    return 1;
#endif
}

__global__ static void indexer_topk_kernel(uint32_t *selected, const float *scores, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *row = scores + (uint64_t)t * n_comp;
    uint32_t *sel = selected + (uint64_t)t * top_k;
    for (uint32_t k = 0; k < top_k; k++) sel[k] = 0;
    for (uint32_t c = 0; c < n_comp; c++) {
        float v = row[c];
        for (uint32_t k = 0; k < top_k; k++) {
            if ((k >= c) || v > row[sel[k]]) {
                for (uint32_t j = top_k - 1; j > k; j--) sel[j] = sel[j - 1];
                sel[k] = c;
                break;
            }
        }
    }
}

__device__ __forceinline__ static bool topk_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__device__ __forceinline__ static void top2_insert_candidate(
        float     v,
        uint32_t  i,
        float    *v0,
        uint32_t *i0,
        float    *v1,
        uint32_t *i1) {
    if (i == *i0 || i == *i1) return;
    if (topk_score_better(v, i, *v0, *i0)) {
        *v1 = *v0;
        *i1 = *i0;
        *v0 = v;
        *i0 = i;
    } else if (topk_score_better(v, i, *v1, *i1)) {
        *v1 = v;
        *i1 = i;
    }
}

/* DSpark markov chain step: out = argmax_i(logits[i] + dot(w2[i], w1[prev]))
 * over the vocab, entirely on-device (logits row stays resident; the chain
 * loop only reads back 4 bytes per draft). w1/w2 are q8_0 with 272-byte rows
 * (8 blocks of 32). Single block; ~35 MB w2 read per step. */
__global__ static void dspark_markov_argmax_kernel(
        unsigned long long *out_key,
        const float *logits,
        const unsigned char *w1_row,
        const unsigned char *w2,
        uint32_t vocab,
        uint32_t rank_blocks) {
    __shared__ float state[256];
    const uint32_t tid = threadIdx.x;
    if (tid < rank_blocks * 32u) {
        const uint32_t b = tid >> 5, k = tid & 31u;
        const unsigned char *blk = w1_row + (uint64_t)b * 34u;
        const float d = __half2float(*(const __half *)blk);
        state[tid] = d * (float)((const int8_t *)(blk + 2))[k];
    }
    __syncthreads();

    float best_v = -INFINITY;
    uint32_t best_i = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + tid; i < vocab;
         i += gridDim.x * blockDim.x) {
        const unsigned char *row = w2 + (uint64_t)i * rank_blocks * 34u;
        float acc = 0.0f;
        for (uint32_t b = 0; b < rank_blocks; b++) {
            const unsigned char *blk = row + (uint64_t)b * 34u;
            const float d = __half2float(*(const __half *)blk);
            const int8_t *q = (const int8_t *)(blk + 2);
            float s = 0.0f;
            #pragma unroll
            for (uint32_t k = 0; k < 32u; k++) s += (float)q[k] * state[b * 32u + k];
            acc += d * s;
        }
        const float v = logits[i] + acc;
        if (topk_score_better(v, i, best_v, best_i)) {
            best_v = v;
            best_i = i;
        }
    }

    __shared__ float vals[256];
    __shared__ uint32_t idxs[256];
    vals[tid] = best_v;
    idxs[tid] = best_i;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            if (topk_score_better(vals[tid + stride], idxs[tid + stride],
                                  vals[tid], idxs[tid])) {
                vals[tid] = vals[tid + stride];
                idxs[tid] = idxs[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
        /* Monotonic float key; ~idx in the low bits makes ties resolve to
         * the smaller index under atomicMax (matches topk_score_better). */
        const unsigned int f = __float_as_uint(vals[0]);
        const unsigned int fkey = (f & 0x80000000u) ? ~f : (f | 0x80000000u);
        const unsigned long long key =
            ((unsigned long long)fkey << 32) | (unsigned int)(~idxs[0]);
        atomicMax(out_key, key);
    }
}

__global__ static void indexer_top1_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 1024u) return;

    const float *row = scores + (uint64_t)t * n_comp;
    float best_v = -INFINITY;
    uint32_t best_i = 0;
    for (uint32_t i = tid; i < n_comp; i += 1024u) {
        const float v = row[i];
        if (topk_score_better(v, i, best_v, best_i)) {
            best_v = v;
            best_i = i;
        }
    }

    __shared__ float vals[1024];
    __shared__ uint32_t idxs[1024];
    vals[tid] = best_v;
    idxs[tid] = best_i;
    __syncthreads();

    for (uint32_t stride = 512u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float ov = vals[tid + stride];
            const uint32_t oi = idxs[tid + stride];
            if (topk_score_better(ov, oi, vals[tid], idxs[tid])) {
                vals[tid] = ov;
                idxs[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0u) selected[t] = idxs[0];
}

__global__ static void indexer_top1_value_kernel(
        uint32_t *selected,
        float *values,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t index_offset) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 1024u) return;

    const float *row = scores + (uint64_t)t * n_comp;
    float best_v = -INFINITY;
    uint32_t best_i = 0;
    for (uint32_t i = tid; i < n_comp; i += 1024u) {
        const float v = row[i];
        const uint32_t gi = index_offset + i;
        const uint32_t best_gi = index_offset + best_i;
        if (topk_score_better(v, gi, best_v, best_gi)) {
            best_v = v;
            best_i = i;
        }
    }

    __shared__ float vals[1024];
    __shared__ uint32_t idxs[1024];
    vals[tid] = best_v;
    idxs[tid] = best_i;
    __syncthreads();

    for (uint32_t stride = 512u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float ov = vals[tid + stride];
            const uint32_t oi = idxs[tid + stride];
            const uint32_t ogi = index_offset + oi;
            const uint32_t gi = index_offset + idxs[tid];
            if (topk_score_better(ov, ogi, vals[tid], gi)) {
                vals[tid] = ov;
                idxs[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0u) {
        selected[t] = index_offset + idxs[0];
        values[t] = vals[0];
    }
}

__global__ static void indexer_top2_value_kernel(
        uint32_t *selected,
        float *values,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t index_offset) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 1024u) return;

    const float *row = scores + (uint64_t)t * n_comp;
    float best0_v = -INFINITY;
    float best1_v = -INFINITY;
    uint32_t best0_i = UINT32_MAX;
    uint32_t best1_i = UINT32_MAX;
    for (uint32_t i = tid; i < n_comp; i += 1024u) {
        const uint32_t gi = index_offset + i;
        top2_insert_candidate(row[i], gi,
                              &best0_v, &best0_i,
                              &best1_v, &best1_i);
    }

    __shared__ float vals0[1024];
    __shared__ float vals1[1024];
    __shared__ uint32_t idxs0[1024];
    __shared__ uint32_t idxs1[1024];
    vals0[tid] = best0_v;
    vals1[tid] = best1_v;
    idxs0[tid] = best0_i;
    idxs1[tid] = best1_i;
    __syncthreads();

    for (uint32_t stride = 512u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            top2_insert_candidate(vals0[tid + stride], idxs0[tid + stride],
                                  &vals0[tid], &idxs0[tid],
                                  &vals1[tid], &idxs1[tid]);
            top2_insert_candidate(vals1[tid + stride], idxs1[tid + stride],
                                  &vals0[tid], &idxs0[tid],
                                  &vals1[tid], &idxs1[tid]);
        }
        __syncthreads();
    }

    if (tid == 0u) {
        selected[(uint64_t)t * 2u + 0u] = idxs0[0];
        selected[(uint64_t)t * 2u + 1u] = idxs1[0];
        values[(uint64_t)t * 2u + 0u] = vals0[0];
        values[(uint64_t)t * 2u + 1u] = vals1[0];
    }
}

__device__ __forceinline__ static uint32_t topk_float_ordered_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ static uint64_t topk_pack_key(float v, uint32_t idx) {
    return ((uint64_t)topk_float_ordered_key(v) << 32u) | (uint64_t)(0xffffffffu - idx);
}

__global__ static void indexer_topk_8192_cub_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    constexpr uint32_t BLOCK_THREADS = 512u;
    constexpr uint32_t ITEMS_PER_THREAD = 16u;
    using BlockSort = cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
    extern __shared__ __align__(16) unsigned char sort_smem[];
    typename BlockSort::TempStorage &sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage *>(sort_smem);

    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= BLOCK_THREADS) return;

    const float *row = scores + (uint64_t)t * n_comp;
    uint64_t keys[ITEMS_PER_THREAD];
#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < n_comp) {
            keys[item] = topk_pack_key(row[i], i);
        } else {
            keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
        }
    }

    BlockSort(sort_storage).SortDescending(keys);

#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < top_k) {
            selected[(uint64_t)t * top_k + i] = 0xffffffffu - (uint32_t)keys[item];
        }
    }
}

__global__ static void indexer_topk_1024_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 1024u) return;
    __shared__ float vals[1024];
    __shared__ uint32_t idxs[1024];

    const float *row = scores + (uint64_t)t * n_comp;
    if (tid < n_comp) {
        vals[tid] = row[tid];
        idxs[tid] = tid;
    } else {
        vals[tid] = -INFINITY;
        idxs[tid] = UINT32_MAX;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= 1024u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            uint32_t other = tid ^ j;
            if (other > tid && other < 1024u) {
                const float av = vals[tid];
                const float bv = vals[other];
                const uint32_t ai = idxs[tid];
                const uint32_t bi = idxs[other];
                const bool desc_half = (tid & k) == 0u;
                const bool swap = desc_half
                    ? topk_score_better(bv, bi, av, ai)
                    : topk_score_better(av, ai, bv, bi);
                if (swap) {
                    vals[tid] = bv;
                    idxs[tid] = bi;
                    vals[other] = av;
                    idxs[other] = ai;
                }
            }
            __syncthreads();
        }
    }

    if (tid < top_k) selected[(uint64_t)t * top_k + tid] = idxs[tid];
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_u16_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint16_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = (uint16_t)i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT16_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = (uint16_t)bi;
                        vals[other] = av;
                        idxs[other] = (uint16_t)ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_chunk_pow2_kernel(
        uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t chunk = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t chunk_start = chunk * SORT_N;
    if (chunk_start >= n_comp) return;
    const uint32_t chunk_n = n_comp - chunk_start < SORT_N ? n_comp - chunk_start : SORT_N;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < chunk_n) {
            vals[i] = row[chunk_start + i];
            idxs[i] = chunk_start + i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *out = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        out[i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_merge_pow2_kernel(
        uint32_t *selected,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_count,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_tree_merge_pow2_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_sets,
        uint32_t merge_group,
        uint32_t candidate_stride,
        uint32_t out_stride) {
    uint32_t t = blockIdx.x;
    uint32_t group = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t set0 = group * merge_group;
    if (set0 >= n_sets) return;
    uint32_t set_count = n_sets - set0;
    if (set_count > merge_group) set_count = merge_group;
    const uint32_t candidate_count = set_count * top_k;

    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride + set0 * top_k;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *dst = out + (uint64_t)t * out_stride + group * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        dst[i] = idxs[i];
    }
}

/* Exact streaming top-512 for wide prefill rows. The threshold is the
 * 512th-best key seen so far, so it can only discard candidates that cannot
 * belong to the final top set. Packing supplies the same value/index total
 * order as the existing bitonic and CUB tiers. */
__global__ static void __launch_bounds__(512) indexer_topk_stream512_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    constexpr uint32_t STREAM_THREADS = 512u;
    constexpr uint32_t STREAM_ITEMS = 4u;
    constexpr uint32_t STREAM_CAP = STREAM_THREADS * STREAM_ITEMS;
    using StreamSort = cub::BlockRadixSort<uint64_t, STREAM_THREADS, STREAM_ITEMS>;
    __shared__ uint64_t buf[STREAM_CAP];
    __shared__ typename StreamSort::TempStorage sort_tmp;
    __shared__ uint32_t s_cnt;
    __shared__ uint64_t s_thr;

    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    const float *row = scores + (uint64_t)t * n_comp;
    if (tid == 0u) {
        s_cnt = 0u;
        s_thr = 0u;
    }
    __syncthreads();

    const uint32_t start =
        (uint32_t)(((uint64_t)(t + 1u) * 0x9E3779B9u) % n_comp);
    const uint32_t tile = blockDim.x;
    for (uint32_t base = 0; base < n_comp; base += tile) {
        const uint64_t thr = s_thr;
        const uint32_t i = base + tid;
        uint64_t key = 0u;
        bool take = false;
        if (i < n_comp) {
            uint32_t c = start + i;
            if (c >= n_comp) c -= n_comp;
            key = topk_pack_key(row[c], c);
            take = key > thr;
        }
        const uint32_t ballot = __ballot_sync(0xffffffffu, take);
        if (take) {
            const uint32_t lane = tid & 31u;
            const uint32_t rank = __popc(ballot & ((1u << lane) - 1u));
            uint32_t pos = 0u;
            if (rank == 0u) pos = atomicAdd(&s_cnt, __popc(ballot));
            pos = __shfl_sync(ballot, pos, __ffs(ballot) - 1);
            buf[pos + rank] = key;
        }
        __syncthreads();

        if (s_cnt > STREAM_CAP - tile) {
            const uint32_t cnt = s_cnt;
            uint64_t keys[STREAM_ITEMS];
#pragma unroll
            for (uint32_t k = 0; k < STREAM_ITEMS; k++) {
                const uint32_t j = tid * STREAM_ITEMS + k;
                keys[k] = j < cnt ? buf[j] : 0u;
            }
            __syncthreads();
            StreamSort(sort_tmp).SortDescending(keys);
            __syncthreads();
            if (tid < top_k / STREAM_ITEMS) {
#pragma unroll
                for (uint32_t k = 0; k < STREAM_ITEMS; k++) {
                    buf[tid * STREAM_ITEMS + k] = keys[k];
                }
            }
            if (tid == top_k / STREAM_ITEMS - 1u) {
                s_thr = keys[STREAM_ITEMS - 1u];
            }
            if (tid == 0u) s_cnt = top_k;
            __syncthreads();
        }
    }

    const uint32_t cnt = s_cnt;
    uint64_t keys[STREAM_ITEMS];
#pragma unroll
    for (uint32_t k = 0; k < STREAM_ITEMS; k++) {
        const uint32_t i = tid * STREAM_ITEMS + k;
        keys[k] = i < cnt ? buf[i] : 0u;
    }
    __syncthreads();
    StreamSort(sort_tmp).SortDescending(keys);
    __syncthreads();
    if (tid < top_k / STREAM_ITEMS) {
#pragma unroll
        for (uint32_t k = 0; k < STREAM_ITEMS; k++) {
            selected[(uint64_t)t * top_k + tid * STREAM_ITEMS + k] =
                0xffffffffu - (uint32_t)keys[k];
        }
    }
}

__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 512u) return;
    __shared__ int32_t rows[512];

    const int32_t *src_row = src + (uint64_t)t * 512u;
    int32_t *dst_row = dst + (uint64_t)t * 512u;
    rows[tid] = src_row[tid];
    __syncthreads();

    for (uint32_t k = 2u; k <= 512u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid && other < 512u) {
                const int32_t a = rows[tid];
                const int32_t b = rows[other];
                const bool up = (tid & k) == 0u;
                if ((up && a > b) || (!up && a < b)) {
                    rows[tid] = b;
                    rows[other] = a;
                }
            }
            __syncthreads();
        }
    }

    dst_row[tid] = rows[tid];
}

__global__ static void topk_mask_kernel(float *mask, const uint32_t *topk, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_comp;
    if (gid >= n) return;
    uint32_t t = gid / n_comp;
    uint32_t c = gid - (uint64_t)t * n_comp;
    float v = -INFINITY;
    for (uint32_t k = 0; k < top_k; k++) {
        if (topk[(uint64_t)t * top_k + k] == c) {
            v = 0.0f;
            break;
        }
    }
    mask[gid] = v;
}

extern "C" int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    (void)n_vocab;
    if (!out_hc || !model_map || weight_offset >= model_size) return 0;
    uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const int logical_tier = ds4_tensor_device_idx(out_hc);
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes, logical_tier, "token_embd");
    if (!wptr) return 0;
    uint32_t n = n_embd * n_hc;
    embed_token_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr, (const unsigned short *)wptr, token, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed token launch");
}

extern "C" int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens_t || !model_map ||
        weight_offset > model_size ||
        (uint64_t)n_vocab * n_embd * sizeof(uint16_t) > model_size - weight_offset ||
        tokens_t->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
        out_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out_hc);
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset,
                                                (uint64_t)n_vocab * n_embd * sizeof(uint16_t),
                                                logical_tier,
                                                "token_embd");
    if (!wptr) return 0;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    embed_tokens_hc_kernel<<<(n + 255) / 256, 256>>>(
        (float *)out_hc->ptr,
        (const int32_t *)tokens_t->ptr,
        (const __half *)wptr,
        n_vocab, n_tokens, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed tokens launch");
}

static int indexer_scores_launch(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        uint32_t                causal) {
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp->bytes < (uint64_t)n_comp * DS4_SPARK_INDEX_ROW_BYTES ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) {
        return 0;
    }
    if (causal && ratio == 0) return 0;
    const int spark = indexer_scores_spark_try(
        scores, q, weights, index_comp, n_comp, n_tokens, pos0,
        n_head, head_dim, ratio, scale, causal);
    if (spark != 0) return spark > 0;
#ifdef DS4_CUDA_SPARK_ONLY
    return 0;
#endif
    const int mxf4 = indexer_scores_mxf4_try(
        scores, q, weights, index_comp, n_comp, n_tokens, pos0,
        n_head, head_dim, ratio, scale);
    if (mxf4 != 0) return mxf4 > 0;
    /* The tensor-core scorer overtakes the direct B1 kernel once the
     * compressed history is long enough; keep the exact path for quality
     * mode and short contexts where launch/staging overhead dominates. */
    if (n_tokens == 1u && head_dim == 128u && n_head == 64u &&
        (g_quality_mode || n_comp < 8192u ||
         getenv("DS4_CUDA_NO_INDEXER_WMMA") != NULL) &&
        getenv("DS4_CUDA_NO_INDEXER_DIRECT_ONE") == NULL) {
        indexer_score_one_direct_kernel<<<n_comp, 128>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, pos0, ratio,
                                                         scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "indexer score one direct launch");
    }
    if (!g_quality_mode && head_dim == 128u && n_head == 64u &&
        getenv("DS4_CUDA_NO_INDEXER_WMMA") == NULL) {
        if (getenv("DS4_CUDA_NO_INDEXER_WMMA128") == NULL) {
            dim3 grid((n_comp + 127u) / 128u, (n_tokens + 31u) / 32u, 1);
            indexer_scores_wmma128_kernel<<<grid, 256>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, n_tokens, pos0, n_head,
                                                         head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma128 launch");
        } else if (getenv("DS4_CUDA_NO_INDEXER_WMMA64") == NULL) {
            dim3 grid((n_comp + 63u) / 64u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma64_kernel<<<grid, 128>>>((float *)scores->ptr,
                                                        (const float *)q->ptr,
                                                        (const float *)weights->ptr,
                                                        (const float *)index_comp->ptr,
                                                        n_comp, n_tokens, pos0, n_head,
                                                        head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma64 launch");
        } else if (getenv("DS4_CUDA_NO_INDEXER_WMMA32") == NULL) {
            dim3 grid((n_comp + 31u) / 32u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma32_kernel<<<grid, 64>>>((float *)scores->ptr,
                                                       (const float *)q->ptr,
                                                       (const float *)weights->ptr,
                                                       (const float *)index_comp->ptr,
                                                       n_comp, n_tokens, pos0, n_head,
                                                       head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma32 launch");
        } else {
            dim3 grid((n_comp + 15u) / 16u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma_kernel<<<grid, 32>>>((float *)scores->ptr,
                                                     (const float *)q->ptr,
                                                     (const float *)weights->ptr,
                                                     (const float *)index_comp->ptr,
                                                     n_comp, n_tokens, pos0, n_head,
                                                     head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma launch");
        }
    }
    dim3 grid(n_comp, n_tokens, 1);
    indexer_scores_kernel<<<grid, 256>>>((float *)scores->ptr,
                                         (const float *)q->ptr,
                                         (const float *)weights->ptr,
                                         (const float *)index_comp->ptr,
                                         n_comp, n_tokens, pos0, n_head,
                                         head_dim, ratio, scale, causal ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "indexer scores launch");
}

extern "C" int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, 1, 0,
                                 n_head, head_dim, 1, scale, 0);
}

extern "C" int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, 0,
                                 n_head, head_dim, ratio, scale, 1);
}

extern "C" int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, pos0,
                                 n_head, head_dim, ratio, scale, 1);
}

extern "C" int ds4_gpu_dspark_markov_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits_row,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              w1_offset,
        uint64_t              w2_offset,
        uint32_t              prev_token,
        uint32_t              vocab,
        uint32_t              rank) {
    if (!out_idx || !logits_row || !model_map || vocab == 0 ||
        rank == 0 || (rank & 31u) != 0u || rank > 256u ||
        out_idx->bytes < sizeof(unsigned long long) ||
        logits_row->bytes < (uint64_t)vocab * sizeof(float)) {
        return 0;
    }
    const uint32_t rank_blocks = rank / 32u;
    const uint64_t row_bytes = (uint64_t)rank_blocks * 34u;
    if (w1_offset > model_size ||
        (uint64_t)prev_token * row_bytes + row_bytes > model_size - w1_offset ||
        w2_offset > model_size ||
        (uint64_t)vocab * row_bytes > model_size - w2_offset) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(logits_row);
    const unsigned char *w1_row = (const unsigned char *)cuda_resolve_weight_ptr(
            model_map, w1_offset + (uint64_t)prev_token * row_bytes,
            row_bytes, logical_tier, "markov_w1_row");
    const unsigned char *w2 = (const unsigned char *)cuda_resolve_weight_ptr(
            model_map, w2_offset, (uint64_t)vocab * row_bytes,
            logical_tier, "markov_w2");
    if (!w1_row || !w2) return 0;
    int dev_save = 0;
    if (cudaGetDevice(&dev_save) != cudaSuccess) return 0;
    if (logical_tier != dev_save && cudaSetDevice(logical_tier) != cudaSuccess) {
        return 0;
    }
    int rc = cudaMemsetAsync(out_idx->ptr, 0,
                             sizeof(unsigned long long)) == cudaSuccess;
    if (rc) {
        dspark_markov_argmax_kernel<<<128, 256>>>(
                (unsigned long long *)out_idx->ptr,
                (const float *)logits_row->ptr,
                w1_row, w2, vocab, rank_blocks);
        rc = cuda_ok(cudaGetLastError(), "dspark markov argmax launch");
    }
    if (logical_tier != dev_save) (void)cudaSetDevice(dev_save);
    return rc;
}

extern "C" int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    if (top_k == 1u && !g_cuda_no_top1) {
        indexer_top1_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                (const float *)scores->ptr,
                                                n_comp,
                                                n_tokens);
        return cuda_ok(cudaGetLastError(), "indexer top1 launch");
    }
    if (top_k == 2048u && n_comp <= 4096u &&
        getenv("DS4_CUDA_NO_TOPK2048_WIDE") == NULL) {
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>(
                (uint32_t *)selected->ptr,
                (const float *)scores->ptr,
                n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 2048-wide launch");
    }
    if (top_k == 2048u && n_comp > 4096u &&
        getenv("DS4_CUDA_NO_TOPK2048_WIDE") == NULL) {
        const uint32_t chunk_n = 4096u;
        const uint32_t merge_group = 2u;
        const uint32_t n_chunks = (n_comp + chunk_n - 1u) / chunk_n;
        const uint32_t candidate_stride = n_chunks * top_k;
        uint32_t n_sets = n_chunks;
        uint64_t scratch_u32_per_token = candidate_stride;
        while (n_sets > merge_group) {
            n_sets = (n_sets + merge_group - 1u) / merge_group;
            scratch_u32_per_token += (uint64_t)n_sets * top_k;
        }
        if (scratch_u32_per_token >
            UINT64_MAX / n_tokens / sizeof(uint32_t)) {
            return 0;
        }
        int exec_tier = ds4_tensor_device_idx(selected);
        int current_device = -1;
        if (cudaGetDevice(&current_device) == cudaSuccess) {
            for (int t = 0; t < g_n_gpus; t++) {
                if (g_gpu[t].device_id == current_device) {
                    exec_tier = t;
                    break;
                }
            }
        }
        const uint64_t tmp_bytes =
            (uint64_t)n_tokens * scratch_u32_per_token * sizeof(uint32_t);
        uint32_t *scratch = (uint32_t *)cuda_tmp_alloc_on(
                exec_tier, tmp_bytes, "indexer topk 2048-wide tree");
        if (!scratch) return 0;

        uint32_t *cur = scratch;
        n_sets = n_chunks;
        uint32_t cur_stride = candidate_stride;
        dim3 grid_chunks(n_tokens, n_chunks, 1);
        indexer_topk_chunk_pow2_kernel<4096><<<grid_chunks, 1024>>>(
                cur, (const float *)scores->ptr,
                n_comp, n_tokens, top_k, candidate_stride);
        if (!cuda_ok(cudaGetLastError(),
                     "indexer topk 2048-wide chunk launch")) {
            return 0;
        }

        while (n_sets > merge_group) {
            const uint32_t next_sets =
                (n_sets + merge_group - 1u) / merge_group;
            const uint32_t next_stride = next_sets * top_k;
            uint32_t *next = cur + (uint64_t)n_tokens * cur_stride;
            dim3 grid_merge(n_tokens, next_sets, 1);
            indexer_topk_tree_merge_pow2_kernel<4096><<<grid_merge, 1024>>>(
                    next, cur, (const float *)scores->ptr,
                    n_comp, n_tokens, top_k, n_sets, merge_group,
                    cur_stride, next_stride);
            if (!cuda_ok(cudaGetLastError(),
                         "indexer topk 2048-wide merge launch")) {
                return 0;
            }
            cur = next;
            n_sets = next_sets;
            cur_stride = next_stride;
        }

        indexer_topk_merge_pow2_kernel<4096><<<n_tokens, 1024>>>(
                (uint32_t *)selected->ptr,
                cur, (const float *)scores->ptr,
                n_comp, n_tokens, top_k, n_sets * top_k, cur_stride);
        return cuda_ok(cudaGetLastError(),
                       "indexer topk 2048-wide final launch");
    }
    if (top_k == 512u && n_comp <= 1024u &&
        getenv("DS4_CUDA_NO_TOPK1024") == NULL) {
        indexer_topk_1024_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 1024 launch");
    }
    if (top_k == 512u && n_comp <= 2048u &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL) {
        indexer_topk_pow2_kernel<2048><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 2048 launch");
    }
    if (top_k == 512u && n_comp <= 4096u &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL) {
        if (n_comp == 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 4096 cub launch");
                }
            }
        }
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096 launch");
    }
    if (top_k == 512u && n_comp <= 8192u &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL &&
        getenv("DS4_CUDA_NO_TOPK8192") == NULL) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 8192 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192 launch");
    }
    if (top_k == 512u && n_tokens >= 32u &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL &&
        getenv("DS4_CUDA_NO_TOPK_STREAM") == NULL) {
        indexer_topk_stream512_kernel<<<n_tokens, 512>>>(
                (uint32_t *)selected->ptr,
                (const float *)scores->ptr,
                n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk stream launch");
    }
    if (top_k == 512u && getenv("DS4_CUDA_NO_TOPK2048") == NULL &&
        getenv("DS4_CUDA_NO_TOPK_CHUNKED") == NULL) {
        const uint32_t chunk_n = 4096u;
        const uint32_t n_chunks = (n_comp + chunk_n - 1u) / chunk_n;
        const uint32_t candidate_stride = n_chunks * top_k;
        uint32_t n_sets = n_chunks;
        uint64_t scratch_u32_per_token = candidate_stride;
        while (n_sets > DS4_CUDA_TOPK_MERGE_GROUP) {
            n_sets = (n_sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
            scratch_u32_per_token += (uint64_t)n_sets * top_k;
        }
        if (scratch_u32_per_token > UINT64_MAX / n_tokens / sizeof(uint32_t)) return 0;
        const uint64_t tmp_bytes = (uint64_t)n_tokens * scratch_u32_per_token * sizeof(uint32_t);
        const int logical_tier = ds4_tensor_device_idx(selected);
        uint32_t *scratch = (uint32_t *)cuda_tmp_alloc_on(logical_tier, tmp_bytes, "indexer topk tree");
        if (!scratch) return 0;

        uint32_t *cur = scratch;
        n_sets = n_chunks;
        uint32_t cur_stride = candidate_stride;
        dim3 grid_chunks(n_tokens, n_chunks, 1);
        indexer_topk_chunk_pow2_kernel<4096><<<grid_chunks, 1024>>>(cur,
                                                                    (const float *)scores->ptr,
                                                                    n_comp,
                                                                    n_tokens,
                                                                    top_k,
                                                                    candidate_stride);
        if (!cuda_ok(cudaGetLastError(), "indexer topk chunk launch")) return 0;

        while (n_sets > DS4_CUDA_TOPK_MERGE_GROUP) {
            const uint32_t next_sets = (n_sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
            const uint32_t next_stride = next_sets * top_k;
            uint32_t *next = cur + (uint64_t)n_tokens * cur_stride;
            dim3 grid_merge(n_tokens, next_sets, 1);
            indexer_topk_tree_merge_pow2_kernel<4096><<<grid_merge, 1024>>>(
                    next,
                    cur,
                    (const float *)scores->ptr,
                    n_comp,
                    n_tokens,
                    top_k,
                    n_sets,
                    DS4_CUDA_TOPK_MERGE_GROUP,
                    cur_stride,
                    next_stride);
            if (!cuda_ok(cudaGetLastError(), "indexer topk tree merge launch")) return 0;
            cur = next;
            n_sets = next_sets;
            cur_stride = next_stride;
        }

        indexer_topk_merge_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                                 cur,
                                                                 (const float *)scores->ptr,
                                                                 n_comp,
                                                                 n_tokens,
                                                                 top_k,
                                                                 n_sets * top_k,
                                                                 cur_stride);
        return cuda_ok(cudaGetLastError(), "indexer topk tree final launch");
    }
    indexer_topk_kernel<<<n_tokens, 1>>>((uint32_t *)selected->ptr,
                                         (const float *)scores->ptr,
                                         n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "indexer topk launch");
}

extern "C" int ds4_gpu_indexer_top1_value_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                index_offset) {
    if (!selected || !values || !scores || n_comp == 0 || n_tokens == 0 ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * sizeof(uint32_t) ||
        values->bytes < (uint64_t)n_tokens * sizeof(float)) {
        return 0;
    }
    indexer_top1_value_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                  (float *)values->ptr,
                                                  (const float *)scores->ptr,
                                                  n_comp,
                                                  n_tokens,
                                                  index_offset);
    return cuda_ok(cudaGetLastError(), "indexer top1 value launch");
}

extern "C" int ds4_gpu_indexer_top2_value_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                index_offset) {
    if (!selected || !values || !scores || n_comp < 2u || n_tokens == 0 ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * 2u * sizeof(uint32_t) ||
        values->bytes < (uint64_t)n_tokens * 2u * sizeof(float)) {
        return 0;
    }
    indexer_top2_value_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                  (float *)values->ptr,
                                                  (const float *)scores->ptr,
                                                  n_comp,
                                                  n_tokens,
                                                  index_offset);
    return cuda_ok(cudaGetLastError(), "indexer top2 value launch");
}

extern "C" int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_tokens * n_comp;
    uint64_t nk = (uint64_t)n_tokens * top_k;
    uint64_t blocks = ((n > nk ? n : nk) + 255) / 256;
    topk_mask_kernel<<<blocks, 256>>>((float *)mask->ptr,
                                      (const uint32_t *)topk->ptr,
                                      n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "topk mask launch");
}
