/* Fused decode-path QKV norm + compact-KV store, one block per
 * (token, part): part 0 rms-norms q into q_out, part 1 rms-norms
 * kv_raw[:kv_lora_dim] into the kv_lora ring, part 2 copies the
 * UNROTATED rope tail into the k_rope ring (roped at attention read). */
__global__ static void glm_qkv_norm_store_compact_kv_kernel(
        float *q_dst,
        const float *q_src,
        const float *q_w,
        uint32_t q_n,
        const float *kv_raw,
        const float *kv_w,
        char *kv_lora_cache,
        char *k_rope_cache,
        uint32_t pos0,
        uint32_t cache_cap,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_rope,
        int cache_f16,
        float eps) {
    const uint32_t token = blockIdx.x;
    const uint32_t part = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    const uint32_t pos = pos0 + token;

    if (part == 2u) {
        if (pos >= cache_cap) return;
        const float *src = kv_raw + (uint64_t)token * kv_raw_dim + kv_lora_dim;
        if (cache_f16) {
            __half *dst = (__half *)k_rope_cache + (uint64_t)pos * qk_rope;
            for (uint32_t i = tid; i < qk_rope; i += nth) {
                dst[i] = __float2half(src[i]);
            }
        } else {
            float *dst = (float *)k_rope_cache + (uint64_t)pos * qk_rope;
            for (uint32_t i = tid; i < qk_rope; i += nth) {
                dst[i] = src[i];
            }
        }
        return;
    }

    const bool kv_task = part != 0u;
    const uint32_t n = kv_task ? kv_lora_dim : q_n;
    const float *x = kv_task ? kv_raw + (uint64_t)token * kv_raw_dim
                             : q_src + (uint64_t)token * q_n;
    const float *w = kv_task ? kv_w : q_w;

    __shared__ float sh[32];
    float sumf = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float v = x[i];
        sumf += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        sumf += __shfl_xor_sync(0xffffffffu, sumf, off);
    }
    if ((tid & 31u) == 0u) sh[tid >> 5] = sumf;
    __syncthreads();
    if (tid < 32u) {
        sumf = (tid < (nth + 31u) / 32u) ? sh[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            sumf += __shfl_xor_sync(0xffffffffu, sumf, off);
        }
        if (tid == 0u) sh[0] = sumf;
    }
    __syncthreads();
    const float scale = rsqrtf(sh[0] / (float)n + eps);

    if (!kv_task) {
        float *y = q_dst + (uint64_t)token * q_n;
        for (uint32_t i = tid; i < n; i += nth) {
            y[i] = (x[i] * scale) * w[i];
        }
        return;
    }

    if (pos >= cache_cap) return;
    if (cache_f16) {
        __half *dst = (__half *)kv_lora_cache + (uint64_t)pos * kv_lora_dim;
        for (uint32_t i = tid; i < kv_lora_dim; i += nth) {
            dst[i] = __float2half((x[i] * scale) * w[i]);
        }
    } else {
        float *dst = (float *)kv_lora_cache + (uint64_t)pos * kv_lora_dim;
        for (uint32_t i = tid; i < kv_lora_dim; i += nth) {
            dst[i] = (x[i] * scale) * w[i];
        }
    }
}

extern "C" int ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_raw,
        uint64_t              kv_weight_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16,
        float                 eps) {
    if (!q_out || !q || !kv_lora_cache || !k_rope_cache || !kv_raw ||
        !model_map || n_tokens == 0 || q_n == 0 || kv_lora_dim == 0 ||
        qk_rope == 0 || kv_raw_dim < kv_lora_dim + qk_rope ||
        q->bytes < (uint64_t)n_tokens * q_n * sizeof(float) ||
        q_out->bytes < (uint64_t)n_tokens * q_n * sizeof(float) ||
        kv_raw->bytes < (uint64_t)n_tokens * kv_raw_dim * sizeof(float)) {
        return 0;
    }
    if (q_weight_offset > model_size ||
        (uint64_t)q_n * sizeof(float) > model_size - q_weight_offset ||
        kv_weight_offset > model_size ||
        (uint64_t)kv_lora_dim * sizeof(float) > model_size - kv_weight_offset) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const float *q_w = (const float *)cuda_resolve_weight_ptr(
            model_map, q_weight_offset, (uint64_t)q_n * sizeof(float),
            logical_tier, "glm_q_norm");
    const float *kv_w = (const float *)cuda_resolve_weight_ptr(
            model_map, kv_weight_offset,
            (uint64_t)kv_lora_dim * sizeof(float),
            logical_tier, "glm_kv_norm");
    if (!q_w || !kv_w) return 0;
    dim3 grid(n_tokens, 3, 1);
    glm_qkv_norm_store_compact_kv_kernel<<<grid, 256>>>(
            (float *)q_out->ptr,
            (const float *)q->ptr,
            q_w,
            q_n,
            (const float *)kv_raw->ptr,
            kv_w,
            (char *)kv_lora_cache->ptr,
            (char *)k_rope_cache->ptr,
            pos0, cache_cap, kv_raw_dim, kv_lora_dim, qk_rope,
            cache_f16 ? 1 : 0, eps);
    return cuda_ok(cudaGetLastError(), "glm qkv norm store compact kv");
}

/* In-place interleaved-pair yarn rope on a [n_tokens][n_head][head_dim]
 * f32 tensor, rotating rot_dim dims starting at rot_offset. Shared by the
 * attention q tail (offset = head_dim - rot_dim) and the DSA indexer
 * (offset = 0). Grid (n_head, n_tokens). */
__global__ static void glm_rope_tail_offset_kernel(
        float *x,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t rot_dim,
        uint32_t rot_offset,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    if (rot_dim == 0u || (rot_dim & 1u) != 0u ||
        rot_offset > head_dim || rot_dim > head_dim - rot_offset) return;

    const uint32_t pos = pos0 + token;
    float *row = x + ((uint64_t)token * n_head + head) * head_dim + rot_offset;

    float corr_dims[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f) {
        corr_dims[0] = fmaxf(0.0f,
            floorf(glm_rope_yarn_corr_factor_dev((int)rot_dim,
                    (int)n_ctx_orig, beta_fast, freq_base)));
        corr_dims[1] = fminf((float)rot_dim - 1.0f,
            ceilf(glm_rope_yarn_corr_factor_dev((int)rot_dim,
                    (int)n_ctx_orig, beta_slow, freq_base)));
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)rot_dim;
    for (uint32_t i = tid * 2u; i < rot_dim; i += nth * 2u) {
        const float theta =
            theta_base * powf(freq_base, inv_ndims * (float)i);
        float ct, st;
        glm_rope_yarn_dev(theta, freq_scale, corr_dims, (int)i,
                          ext_factor, attn_factor, &ct, &st);
        const float x0 = row[i];
        const float x1 = row[i + 1u];
        row[i] = x0 * ct - x1 * st;
        row[i + 1u] = x0 * st + x1 * ct;
    }
}

static int glm_rope_tail_offset_launch(
        ds4_gpu_tensor *x,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim, uint32_t rot_offset, uint32_t pos0,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, const char *what) {
    if (!x || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        rot_dim == 0 || (rot_dim & 1u) != 0u || rot_offset > head_dim ||
        rot_dim > head_dim - rot_offset ||
        x->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    dim3 grid(n_head, n_tokens, 1);
    glm_rope_tail_offset_kernel<<<grid, 32>>>(
            (float *)x->ptr, n_head, head_dim, rot_dim, rot_offset,
            pos0, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), what);
}

extern "C" int ds4_gpu_glm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow) {
    if (rot_dim > head_dim) return 0;
    return glm_rope_tail_offset_launch(x, n_tokens, n_head, head_dim,
            rot_dim, head_dim - rot_dim, pos0, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, "glm rope tail");
}

extern "C" int ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor\n");
    return 0;
}

/* Scalar-correct GLM routed MoE (q2_K experts): per (token, slot) block
 * quantizes nothing - dots q2_K rows against a q8_K-quantized activation
 * staged in shared memory. Grid: (n_tokens, n_expert). Mid buffer holds
 * silu(gate)*up per slot; out accumulates expert_weight-scaled down rows.
 */
__global__ static void glm_routed_moe_batch_q2K_gateup_kernel(
        float *mid,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t n_tokens,
        uint32_t mid_token_stride) {
    const uint32_t tok = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (tok >= n_tokens || slot >= n_expert) return;
    const int32_t expert = selected[(uint64_t)tok * n_expert + slot];
    if (expert < 0) return;
    const cuda_block_q8_K *xrow = xq + (uint64_t)tok * xq_blocks;
    float *mrow = mid + (uint64_t)tok * mid_token_stride +
                  (uint64_t)slot * expert_mid_dim;
    for (uint32_t r = threadIdx.x; r < expert_mid_dim; r += blockDim.x) {
        const cuda_block_q2_K *gr = (const cuda_block_q2_K *)(gate_base +
            (uint64_t)expert * gate_expert_bytes + (uint64_t)r * gate_row_bytes);
        const cuda_block_q2_K *ur = (const cuda_block_q2_K *)(up_base +
            (uint64_t)expert * up_expert_bytes + (uint64_t)r * up_row_bytes);
        float g = 0.0f, u = 0.0f;
        for (uint32_t b = 0; b < xq_blocks; b++) {
            g += dev_dot_q2_K_q8_K_block(gr + b, xrow + b);
            u += dev_dot_q2_K_q8_K_block(ur + b, xrow + b);
        }
        mrow[r] = (g / (1.0f + expf(-g))) * u;   /* silu(g)*u */
    }
}

__global__ static void glm_routed_moe_batch_q2K_down_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        const float *weights,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint32_t tok = blockIdx.y;
    if (tok >= n_tokens) return;
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= out_dim) return;
    float acc = 0.0f;
    for (uint32_t slot = 0; slot < n_expert; slot++) {
        const int32_t expert = selected[(uint64_t)tok * n_expert + slot];
        if (expert < 0) continue;
        const float w = weights[(uint64_t)tok * n_expert + slot];
        const cuda_block_q2_K *dr = (const cuda_block_q2_K *)(down_base +
            (uint64_t)expert * down_expert_bytes + (uint64_t)r * down_row_bytes);
        const cuda_block_q8_K *mrow = midq +
            ((uint64_t)tok * n_expert + slot) * midq_blocks;
        float s = 0.0f;
        for (uint32_t b = 0; b < midq_blocks; b++) {
            s += dev_dot_q2_K_q8_K_block(dr + b, mrow + b);
        }
        acc += w * s;
    }
    out[(uint64_t)tok * out_dim + r] = acc;
}

/* Warp-per-row routed MoE (q2_K x q8_K). Each block stages the token's
 * q8_K activation row in shared memory; one warp produces one mid row
 * (gate dot + up dot + silu*mul fused). Grid:
 * (expert_mid_dim/warps, n_expert, n_tokens). */
__global__ static void glm_routed_moe_gateup_warp_kernel(
        float *mid,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint32_t tok = blockIdx.z;
    const uint32_t slot = blockIdx.y;
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (tok >= n_tokens || slot >= n_expert) return;
    const int32_t expert = selected[(uint64_t)tok * n_expert + slot];

    extern __shared__ unsigned int glm_moe_sh_u32[];
    {
        const unsigned int *src =
            (const unsigned int *)(xq + (uint64_t)tok * xq_blocks);
        const uint32_t words = xq_blocks * (uint32_t)sizeof(cuda_block_q8_K) / 4u;
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) {
            glm_moe_sh_u32[i] = src[i];
        }
    }
    __syncthreads();
    if (expert < 0) return;
    const cuda_block_q8_K *xrow = (const cuda_block_q8_K *)glm_moe_sh_u32;

    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= expert_mid_dim) return;
    const char *gr = gate_base + (uint64_t)expert * gate_expert_bytes +
                     (uint64_t)r * gate_row_bytes;
    const char *ur = up_base + (uint64_t)expert * up_expert_bytes +
                     (uint64_t)r * up_row_bytes;
    float g = 0.0f, u = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        g += dev_dot_q2_K_q8_K_block(
                (const cuda_block_q2_K *)(gr + (uint64_t)b * 84u), xrow + b);
        u += dev_dot_q2_K_q8_K_block(
                (const cuda_block_q2_K *)(ur + (uint64_t)b * 84u), xrow + b);
    }
    for (int off = 16; off > 0; off >>= 1) {
        g += __shfl_down_sync(0xffffffffu, g, off);
        u += __shfl_down_sync(0xffffffffu, u, off);
    }
    if (lane == 0u) {
        mid[((uint64_t)tok * n_expert + slot) * expert_mid_dim + r] =
            (g / (1.0f + expf(-g))) * u;
    }
}

/* Exact two-token gate/up with adjacent-token expert reuse. Token 0 owns an
 * expert present in both rows; token 1 only launches work for unmatched
 * experts. Each token keeps the native lane assignment and warp reduction. */
__global__ static void glm_routed_moe_gateup_tok2_reuse_kernel(
        float *mid,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert) {
    const uint32_t owner = blockIdx.y;
    const uint32_t tok = owner / n_expert;
    const uint32_t slot = owner - tok * n_expert;
    const uint32_t warps = blockDim.x >> 5u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    if (tok >= 2u || slot >= n_expert) return;

    const int32_t expert = selected[owner];
    if (expert < 0) return;
    int32_t mate_slot = -1;
    for (uint32_t s = 0; s < n_expert; s++) {
        if (selected[(uint64_t)(1u - tok) * n_expert + s] == expert) {
            mate_slot = (int32_t)s;
            break;
        }
    }
    if (tok == 1u && mate_slot >= 0) return;

    const uint32_t np = tok == 0u && mate_slot >= 0 ? 2u : 1u;
    const uint32_t pair0 = owner;
    const uint32_t pair1 = n_expert + (uint32_t)mate_slot;
    extern __shared__ unsigned int glm_moe_tok2_sh_u32[];
    const uint32_t words_per_row =
        xq_blocks * (uint32_t)sizeof(cuda_block_q8_K) / 4u;
    const unsigned int *src0 = (const unsigned int *)(
        xq + (uint64_t)tok * xq_blocks);
    for (uint32_t i = threadIdx.x; i < words_per_row;
         i += blockDim.x) {
        glm_moe_tok2_sh_u32[i] = src0[i];
    }
    if (np == 2u) {
        const unsigned int *src1 =
            (const unsigned int *)(xq + xq_blocks);
        for (uint32_t i = threadIdx.x; i < words_per_row;
             i += blockDim.x) {
            glm_moe_tok2_sh_u32[words_per_row + i] = src1[i];
        }
    }
    __syncthreads();

    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= expert_mid_dim) return;
    const char *gr = gate_base + (uint64_t)expert * gate_expert_bytes +
                     (uint64_t)r * gate_row_bytes;
    const char *ur = up_base + (uint64_t)expert * up_expert_bytes +
                     (uint64_t)r * up_row_bytes;
    const cuda_block_q8_K *x0 =
        (const cuda_block_q8_K *)glm_moe_tok2_sh_u32;
    const cuda_block_q8_K *x1 = np == 2u
        ? x0 + xq_blocks
        : NULL;
    float g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        dev_dot_q2_K_q8_K_block8(
                (const cuda_block_q2_K *)(gr + (uint64_t)b * 84u),
                x0 + b, np == 2u ? x1 + b : NULL,
                NULL, NULL, NULL, NULL, NULL, NULL, np, g);
        dev_dot_q2_K_q8_K_block8(
                (const cuda_block_q2_K *)(ur + (uint64_t)b * 84u),
                x0 + b, np == 2u ? x1 + b : NULL,
                NULL, NULL, NULL, NULL, NULL, NULL, np, u);
    }
    for (uint32_t p = 0; p < np; p++) {
        for (int off = 16; off > 0; off >>= 1) {
            g[p] += __shfl_down_sync(0xffffffffu, g[p], off);
            u[p] += __shfl_down_sync(0xffffffffu, u[p], off);
        }
        if (lane == 0u) {
            const uint32_t pair = p == 0u ? pair0 : pair1;
            mid[(uint64_t)pair * expert_mid_dim + r] =
                (g[p] / (1.0f + expf(-g[p]))) * u[p];
        }
    }
}

/* Warp-per-output-row down projection: stages all n_expert quantized mid
 * rows for the token in shared memory, each warp accumulates one out row
 * across every selected expert. Grid: (out_dim/warps, n_tokens). */
__global__ static void glm_routed_moe_down_warp_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        const float *weights,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint32_t tok = blockIdx.y;
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (tok >= n_tokens) return;

    extern __shared__ unsigned int glm_moe_sh_u32[];
    {
        const unsigned int *src = (const unsigned int *)
                (midq + (uint64_t)tok * n_expert * midq_blocks);
        const uint32_t words = n_expert * midq_blocks *
                               (uint32_t)sizeof(cuda_block_q8_K) / 4u;
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) {
            glm_moe_sh_u32[i] = src[i];
        }
    }
    __syncthreads();
    const cuda_block_q8_K *msh = (const cuda_block_q8_K *)glm_moe_sh_u32;

    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= out_dim) return;
    const uint32_t units = n_expert * midq_blocks;
    float acc = 0.0f;
    for (uint32_t idx = lane; idx < units; idx += 32u) {
        const uint32_t slot = idx / midq_blocks;
        const uint32_t b = idx - slot * midq_blocks;
        const int32_t expert = selected[(uint64_t)tok * n_expert + slot];
        if (expert < 0) continue;
        const float w = weights[(uint64_t)tok * n_expert + slot];
        const cuda_block_q2_K *dr = (const cuda_block_q2_K *)(down_base +
            (uint64_t)expert * down_expert_bytes + (uint64_t)r * down_row_bytes);
        acc += w * dev_dot_q2_K_q8_K_block(dr + b, msh + slot * midq_blocks + b);
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0u) out[(uint64_t)tok * out_dim + r] = acc;
}

/* Expert-major routed MoE for prefill: build per-expert token lists,
 * then walk rows expert-by-expert so weights stream once per layer and
 * activations hit L2. pair = tok * n_expert + slot indexes selected/
 * weights/mid rows directly. */
__global__ static void glm_moe_expert_map_kernel(
        int32_t *counts,
        int32_t *lists,
        const int32_t *selected,
        uint32_t n_pairs,
        uint32_t n_total_expert,
        uint32_t cap,
        uint32_t pair_base) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    const uint32_t pair = pair_base + p;
    const int32_t e = selected[pair];
    if (e < 0 || (uint32_t)e >= n_total_expert) return;
    const int32_t idx = atomicAdd(&counts[e], 1);
    lists[(uint64_t)e * cap + idx] = (int32_t)pair;
}

__global__ static void glm_moe_build_expert_tiles8_kernel(
        uint32_t *tile_total,
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const int32_t *counts,
        uint32_t n_total_expert) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    uint32_t total = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        const uint32_t count = counts[e] > 0 ? (uint32_t)counts[e] : 0u;
        const uint32_t ntiles = (count + 7u) / 8u;
        for (uint32_t t = 0; t < ntiles; t++) {
            tile_experts[total] = e;
            tile_starts[total] = t * 8u;
            total++;
        }
    }
    *tile_total = total;
}

/* Expert-tiled Q2_K gate/up for GLM prefill. One warp keeps the same
 * block-to-lane assignment and reduction tree as the token-major W32
 * kernel, but evaluates eight pairs against each loaded expert row. */
__global__ static void glm_routed_moe_gateup_expert_tile8_kernel(
        float *mid,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *counts,
        const int32_t *lists,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t cap) {
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t warps = blockDim.x >> 5u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    const uint32_t count = counts[expert] > 0 ? (uint32_t)counts[expert] : 0u;

    __shared__ uint32_t pair[8];
    __shared__ uint32_t tok[8];
    __shared__ uint32_t np;
    if (threadIdx.x == 0u) {
        uint32_t n = count - local_start;
        if (n > 8u) n = 8u;
        np = n;
        for (uint32_t p = 0; p < n; p++) {
            const uint32_t pr =
                (uint32_t)lists[(uint64_t)expert * cap + local_start + p];
            pair[p] = pr;
            tok[p] = pr / n_expert;
        }
    }
    __syncthreads();

    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= expert_mid_dim) return;
    const char *gr = gate_base + (uint64_t)expert * gate_expert_bytes +
                     (uint64_t)r * gate_row_bytes;
    const char *ur = up_base + (uint64_t)expert * up_expert_bytes +
                     (uint64_t)r * up_row_bytes;
    float g[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float u[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        const cuda_block_q8_K *x0 = np > 0u ? xq + (uint64_t)tok[0] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x1 = np > 1u ? xq + (uint64_t)tok[1] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x2 = np > 2u ? xq + (uint64_t)tok[2] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x3 = np > 3u ? xq + (uint64_t)tok[3] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x4 = np > 4u ? xq + (uint64_t)tok[4] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x5 = np > 5u ? xq + (uint64_t)tok[5] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x6 = np > 6u ? xq + (uint64_t)tok[6] * xq_blocks + b : NULL;
        const cuda_block_q8_K *x7 = np > 7u ? xq + (uint64_t)tok[7] * xq_blocks + b : NULL;
        dev_dot_q2_K_q8_K_block8(
                (const cuda_block_q2_K *)(gr + (uint64_t)b * 84u),
                x0, x1, x2, x3, x4, x5, x6, x7, np, g);
        dev_dot_q2_K_q8_K_block8(
                (const cuda_block_q2_K *)(ur + (uint64_t)b * 84u),
                x0, x1, x2, x3, x4, x5, x6, x7, np, u);
    }
    for (uint32_t p = 0; p < np; p++) {
        for (int off = 16; off > 0; off >>= 1) {
            g[p] += __shfl_down_sync(0xffffffffu, g[p], off);
            u[p] += __shfl_down_sync(0xffffffffu, u[p], off);
        }
        if (lane == 0u) {
            mid[(uint64_t)pair[p] * expert_mid_dim + r] =
                (g[p] / (1.0f + expf(-g[p]))) * u[p];
        }
    }
}

__global__ static void glm_routed_moe_gateup_expert_kernel(
        float *mid,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *counts,
        const int32_t *lists,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t up_expert_bytes,
        uint64_t up_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t cap) {
    const uint32_t e = blockIdx.y;
    const int32_t nt = counts[e];
    if (nt == 0) return;
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= expert_mid_dim) return;
    const char *gr = gate_base + (uint64_t)e * gate_expert_bytes +
                     (uint64_t)r * gate_row_bytes;
    const char *ur = up_base + (uint64_t)e * up_expert_bytes +
                     (uint64_t)r * up_row_bytes;
    const int32_t *lst = lists + (uint64_t)e * cap;
    for (int32_t i = 0; i < nt; i++) {
        const uint32_t pair = (uint32_t)lst[i];
        const cuda_block_q8_K *xrow = xq + (uint64_t)(pair / n_expert) * xq_blocks;
        float g = 0.0f, u = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            g += dev_dot_q2_K_q8_K_block(
                    (const cuda_block_q2_K *)(gr + (uint64_t)b * 84u), xrow + b);
            u += dev_dot_q2_K_q8_K_block(
                    (const cuda_block_q2_K *)(ur + (uint64_t)b * 84u), xrow + b);
        }
        for (int off = 16; off > 0; off >>= 1) {
            g += __shfl_down_sync(0xffffffffu, g, off);
            u += __shfl_down_sync(0xffffffffu, u, off);
        }
        if (lane == 0u) {
            mid[(uint64_t)pair * expert_mid_dim + r] =
                (g / (1.0f + expf(-g))) * u;
        }
    }
}

__global__ static void glm_routed_moe_down_expert_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *counts,
        const int32_t *lists,
        const float *weights,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t cap) {
    const uint32_t e = blockIdx.y;
    const int32_t nt = counts[e];
    if (nt == 0) return;
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= out_dim) return;
    const char *dr = down_base + (uint64_t)e * down_expert_bytes +
                     (uint64_t)r * down_row_bytes;
    const int32_t *lst = lists + (uint64_t)e * cap;
    for (int32_t i = 0; i < nt; i++) {
        const uint32_t pair = (uint32_t)lst[i];
        const cuda_block_q8_K *mrow = midq + (uint64_t)pair * midq_blocks;
        float s = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 32u) {
            s += dev_dot_q2_K_q8_K_block(
                    (const cuda_block_q2_K *)(dr + (uint64_t)b * 84u), mrow + b);
        }
        for (int off = 16; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffffffffu, s, off);
        }
        if (lane == 0u) {
            atomicAdd(&out[(uint64_t)(pair / n_expert) * out_dim + r],
                      weights[pair] * s);
        }
    }
}

/* Expert-tiled down projection with an exact token-major reduction. The
 * first kernel reuses each Q2_K row across eight routed pairs, but materializes
 * each block dot. The second kernel applies the router weight and consumes the
 * dots with the same lane assignment and warp tree as the native kernel. */
__global__ static void glm_routed_moe_down_expert_tile8_terms_kernel(
        float *terms,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *counts,
        const int32_t *lists,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t cap,
        uint32_t pair_base) {
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    const uint32_t count = counts[expert] > 0 ? (uint32_t)counts[expert] : 0u;

    __shared__ uint32_t pair[8];
    __shared__ uint32_t np;
    __shared__ cuda_block_q8_K mq[8][8];
    if (threadIdx.x == 0u) {
        uint32_t n = count - local_start;
        if (n > 8u) n = 8u;
        np = n;
        for (uint32_t p = 0; p < n; p++) {
            pair[p] = (uint32_t)lists[
                (uint64_t)expert * cap + local_start + p];
        }
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
        const uint32_t p = i / midq_blocks;
        const uint32_t b = i - p * midq_blocks;
        mq[p][b] = midq[(uint64_t)pair[p] * midq_blocks + b];
    }
    __syncthreads();
    if (row >= out_dim || lane >= midq_blocks) return;

    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base +
        (uint64_t)expert * down_expert_bytes +
        (uint64_t)row * down_row_bytes);
    for (uint32_t p = 0; p < np; p++) {
        const uint32_t pr = pair[p];
        const float dot = dev_dot_q2_K_q8_K_block(wr + lane, &mq[p][lane]);
        terms[((uint64_t)(pr - pair_base) * out_dim + row) *
              midq_blocks + lane] =
            dot;
    }
}

__global__ static void glm_routed_moe_down_terms_reduce_kernel(
        float *out,
        const float *terms,
        const int32_t *selected,
        const float *weights,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint32_t tok = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + warp;
    if (tok >= n_tokens || row >= out_dim) return;
    const uint32_t units = n_expert * midq_blocks;
    float acc = 0.0f;
    for (uint32_t idx = lane; idx < units; idx += 32u) {
        const uint32_t slot = idx / midq_blocks;
        const uint32_t b = idx - slot * midq_blocks;
        const uint32_t pr = tok * n_expert + slot;
        if (selected[pr] >= 0) {
            acc += weights[pr] *
                terms[((uint64_t)pr * out_dim + row) * midq_blocks + b];
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = acc;
}

static int glm_routed_moe_finish_batch(
        ds4_gpu_tensor *out,
        float *out_work,
        uint64_t out_bytes,
        const char *what) {
    if (!cuda_ok(cudaGetLastError(), what)) return 0;
    if (out_work == (float *)out->ptr) return 1;
    return cuda_ok(cudaMemcpyAsync(out->ptr, out_work, out_bytes,
                                   cudaMemcpyDeviceToDevice, 0),
                   "glm routed moe local output copy");
}

extern "C" int ds4_gpu_glm_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride) {
    (void)layer_index; (void)n_total_expert;
    if (!out || !mid || !x || !selected || !weights || !model_map ||
        n_tokens == 0 || n_expert == 0 ||
        (expert_in_dim & 255u) != 0u || (expert_mid_dim & 255u) != 0u) {
        return 0;
    }
    if (gate_type != 10u || up_type != 10u || down_type != 10u) {
        fprintf(stderr, "ds4: glm routed moe: unsupported types %u/%u/%u\n",
                gate_type, up_type, down_type);
        return 0;
    }
    if (mid_token_stride != n_expert * expert_mid_dim) {
        fprintf(stderr,
            "ds4: glm routed moe: mid stride %u != %u (packed rows expected)\n",
            mid_token_stride, n_expert * expert_mid_dim);
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const char *gw = (const char *)cuda_resolve_weight_ptr(model_map,
            gate_offset, (uint64_t)256 * gate_expert_bytes, logical_tier,
            "glm_gate_exps");
    const char *uw = (const char *)cuda_resolve_weight_ptr(model_map,
            up_offset, (uint64_t)256 * up_expert_bytes, logical_tier,
            "glm_up_exps");
    const char *dw = (const char *)cuda_resolve_weight_ptr(model_map,
            down_offset, (uint64_t)256 * down_expert_bytes, logical_tier,
            "glm_down_exps");
    if (!gw || !uw || !dw) return 0;

    /* Stage 1: quantize x rows to q8_K (existing kernel). */
    const uint32_t xq_blocks = expert_in_dim / 256u;
    const uint32_t midq_blocks = expert_mid_dim / 256u;
    static ds4_gpu_tensor *xq_scratch[DS4_MAX_GPUS] = {0};
    static ds4_gpu_tensor *midq_scratch[DS4_MAX_GPUS] = {0};
    int dev = logical_tier;
    const int scratch_tier = getenv("DS4_GLM_MOE_SCRATCH_TIER0") ? 0 : dev;
    const uint64_t xq_bytes = (uint64_t)n_tokens * xq_blocks *
                              sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = (uint64_t)n_tokens * n_expert * midq_blocks *
                                sizeof(cuda_block_q8_K);
    if (!xq_scratch[dev] || xq_scratch[dev]->bytes < xq_bytes) {
        if (xq_scratch[dev]) ds4_gpu_tensor_free(xq_scratch[dev]);
        xq_scratch[dev] = ds4_gpu_tensor_alloc_ptr_on(scratch_tier, xq_bytes);
    }
    if (!midq_scratch[dev] || midq_scratch[dev]->bytes < midq_bytes) {
        if (midq_scratch[dev]) ds4_gpu_tensor_free(midq_scratch[dev]);
        midq_scratch[dev] = ds4_gpu_tensor_alloc_ptr_on(scratch_tier, midq_bytes);
    }
    if (!xq_scratch[dev] || !midq_scratch[dev]) return 0;

    static ds4_gpu_tensor *mid_local[DS4_MAX_GPUS] = {0};
    static ds4_gpu_tensor *out_local[DS4_MAX_GPUS] = {0};
    const uint64_t mid_work_bytes =
        (uint64_t)n_tokens * mid_token_stride * sizeof(float);
    const uint64_t out_work_bytes =
        (uint64_t)n_tokens * out_dim * sizeof(float);
    float *mid_work = (float *)mid->ptr;
    float *out_work = (float *)out->ptr;
    const bool use_local_batch_io =
        n_tokens >= 128u && !getenv("DS4_GLM_MOE_NO_LOCAL_BATCH_IO");
    if (use_local_batch_io && ds4_tensor_device_idx(mid) != dev) {
        if (!mid_local[dev] || mid_local[dev]->bytes < mid_work_bytes) {
            if (mid_local[dev]) ds4_gpu_tensor_free(mid_local[dev]);
            mid_local[dev] =
                ds4_gpu_tensor_alloc_ptr_on(dev, mid_work_bytes);
        }
        if (!mid_local[dev]) return 0;
        mid_work = (float *)mid_local[dev]->ptr;
    }
    if (use_local_batch_io && ds4_tensor_device_idx(out) != dev) {
        if (!out_local[dev] || out_local[dev]->bytes < out_work_bytes) {
            if (out_local[dev]) ds4_gpu_tensor_free(out_local[dev]);
            out_local[dev] =
                ds4_gpu_tensor_alloc_ptr_on(dev, out_work_bytes);
        }
        if (!out_local[dev]) return 0;
        out_work = (float *)out_local[dev]->ptr;
    }
    {
        dim3 gq(xq_blocks, n_tokens, 1);
        q8_K_quantize_kernel<<<gq, 256>>>(
                (cuda_block_q8_K *)xq_scratch[dev]->ptr,
                (const float *)x->ptr, expert_in_dim, n_tokens);
    }

    static ds4_gpu_tensor *map_scratch[DS4_MAX_GPUS] = {0};
    static ds4_gpu_tensor *down_terms_scratch[DS4_MAX_GPUS] = {0};
    const bool use_expert_tile8 =
        n_tokens >= 128u && !getenv("DS4_GLM_MOE_NO_EXPERT_TILE8");
    const bool use_expert_major =
        n_tokens >= 16u && getenv("DS4_GLM_MOE_EXPERT_MAJOR");
    if (use_expert_tile8 || use_expert_major) {
        const uint32_t cap = n_tokens;
        const uint32_t n_pairs = n_tokens * n_expert;
        const uint64_t counts_bytes = 256u * sizeof(int32_t);
        const uint64_t lists_off = (counts_bytes + 255u) & ~255ull;
        const uint64_t lists_bytes =
            (uint64_t)256u * cap * sizeof(int32_t);
        const uint32_t tile_capacity =
            (n_pairs + 7u) / 8u + 256u;
        const uint64_t tile_total_off =
            (lists_off + lists_bytes + 255u) & ~255ull;
        const uint64_t tile_experts_off =
            (tile_total_off + sizeof(uint32_t) + 255u) & ~255ull;
        const uint64_t tile_starts_off =
            tile_experts_off + (uint64_t)tile_capacity * sizeof(uint32_t);
        const uint64_t map_bytes = use_expert_tile8
            ? tile_starts_off + (uint64_t)tile_capacity * sizeof(uint32_t)
            : lists_off + lists_bytes;
        if (!map_scratch[dev] || map_scratch[dev]->bytes < map_bytes) {
            if (map_scratch[dev]) ds4_gpu_tensor_free(map_scratch[dev]);
            map_scratch[dev] = ds4_gpu_tensor_alloc_ptr_on(dev, map_bytes);
        }
        if (map_scratch[dev]) {
            int32_t *counts = (int32_t *)map_scratch[dev]->ptr;
            int32_t *lists = (int32_t *)((char *)map_scratch[dev]->ptr + lists_off);
            cudaMemsetAsync(counts, 0, counts_bytes);
            glm_moe_expert_map_kernel<<<(n_pairs + 255u) / 256u, 256>>>(
                    counts, lists, (const int32_t *)selected->ptr,
                    n_pairs, 256u, cap, 0u);
            if (use_expert_tile8) {
                uint32_t *tile_total = (uint32_t *)(
                        (char *)map_scratch[dev]->ptr + tile_total_off);
                uint32_t *tile_experts = (uint32_t *)(
                        (char *)map_scratch[dev]->ptr + tile_experts_off);
                uint32_t *tile_starts = (uint32_t *)(
                        (char *)map_scratch[dev]->ptr + tile_starts_off);
                glm_moe_build_expert_tiles8_kernel<<<1, 1>>>(
                        tile_total, tile_experts, tile_starts,
                        counts, 256u);
                dim3 ge1((expert_mid_dim + 7u) / 8u,
                          tile_capacity, 1);
                glm_routed_moe_gateup_expert_tile8_kernel<<<ge1, 256>>>(
                        mid_work, gw, uw,
                        (const cuda_block_q8_K *)xq_scratch[dev]->ptr,
                        counts, lists,
                        tile_total, tile_experts, tile_starts,
                        gate_expert_bytes, gate_row_bytes,
                        up_expert_bytes, up_row_bytes,
                        xq_blocks, expert_mid_dim, n_expert, cap);
                q8_K_quantize_kernel<<<
                        dim3(midq_blocks, n_tokens * n_expert, 1), 256>>>(
                        (cuda_block_q8_K *)midq_scratch[dev]->ptr,
                        mid_work,
                        expert_mid_dim, n_tokens * n_expert);
                if (getenv("DS4_GLM_MOE_NO_DOWN_TILE8_EXACT") == NULL) {
                    const uint32_t max_chunk_tokens = 512u;
                    const uint32_t scratch_tokens =
                        n_tokens < max_chunk_tokens ? n_tokens : max_chunk_tokens;
                    const uint64_t term_count =
                        (uint64_t)scratch_tokens * n_expert *
                        out_dim * midq_blocks;
                    const uint64_t term_bytes = term_count * sizeof(float);
                    if (!down_terms_scratch[dev] ||
                        down_terms_scratch[dev]->bytes < term_bytes) {
                        if (down_terms_scratch[dev]) {
                            ds4_gpu_tensor_free(down_terms_scratch[dev]);
                        }
                        down_terms_scratch[dev] =
                            ds4_gpu_tensor_alloc_ptr_on(dev, term_bytes);
                    }
                    if (down_terms_scratch[dev]) {
                        for (uint32_t token0 = 0; token0 < n_tokens;
                             token0 += max_chunk_tokens) {
                            uint32_t chunk_tokens = n_tokens - token0;
                            if (chunk_tokens > max_chunk_tokens) {
                                chunk_tokens = max_chunk_tokens;
                            }
                            const uint32_t pair_base = token0 * n_expert;
                            const uint32_t chunk_pairs =
                                chunk_tokens * n_expert;
                            uint32_t chunk_tile_capacity = tile_capacity;
                            if (n_tokens > max_chunk_tokens) {
                                chunk_tile_capacity =
                                    (chunk_pairs + 7u) / 8u + 256u;
                                cudaMemsetAsync(counts, 0, counts_bytes);
                                glm_moe_expert_map_kernel<<<
                                        (chunk_pairs + 255u) / 256u, 256>>>(
                                        counts, lists,
                                        (const int32_t *)selected->ptr,
                                        chunk_pairs, 256u, cap, pair_base);
                                glm_moe_build_expert_tiles8_kernel<<<1, 1>>>(
                                        tile_total, tile_experts, tile_starts,
                                        counts, 256u);
                            }
                            dim3 gd1((out_dim + 31u) / 32u,
                                     chunk_tile_capacity, 1);
                            glm_routed_moe_down_expert_tile8_terms_kernel<<<
                                    gd1, 256>>>(
                                    (float *)down_terms_scratch[dev]->ptr,
                                    dw,
                                    (const cuda_block_q8_K *)midq_scratch[dev]->ptr,
                                    counts, lists,
                                    tile_total, tile_experts, tile_starts,
                                    down_expert_bytes, down_row_bytes,
                                    midq_blocks, out_dim, n_expert, cap,
                                    pair_base);
                            dim3 gd2((out_dim + 7u) / 8u,
                                     chunk_tokens, 1);
                            glm_routed_moe_down_terms_reduce_kernel<<<
                                    gd2, 256>>>(
                                    out_work + (uint64_t)token0 * out_dim,
                                    (const float *)down_terms_scratch[dev]->ptr,
                                    (const int32_t *)selected->ptr + pair_base,
                                    (const float *)weights->ptr + pair_base,
                                    midq_blocks, out_dim, n_expert,
                                    chunk_tokens);
                        }
                        return glm_routed_moe_finish_batch(
                                out, out_work, out_work_bytes,
                                "glm routed moe exact down tile8");
                    }
                }
                const uint32_t warps = 8u;
                dim3 ge2((out_dim + warps - 1u) / warps,
                          n_tokens, 1);
                const uint32_t sh2 = n_expert * midq_blocks *
                                     (uint32_t)sizeof(cuda_block_q8_K);
                glm_routed_moe_down_warp_kernel<<<
                        ge2, warps * 32u, sh2>>>(
                        out_work, dw,
                        (const cuda_block_q8_K *)midq_scratch[dev]->ptr,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, n_tokens);
                return glm_routed_moe_finish_batch(
                        out, out_work, out_work_bytes,
                        "glm routed moe expert tile8");
            }
            dim3 ge1((expert_mid_dim + 7u) / 8u, 256u, 1);
            glm_routed_moe_gateup_expert_kernel<<<ge1, 256>>>(
                    mid_work, gw, uw,
                    (const cuda_block_q8_K *)xq_scratch[dev]->ptr,
                    counts, lists,
                    gate_expert_bytes, gate_row_bytes,
                    up_expert_bytes, up_row_bytes,
                    xq_blocks, expert_mid_dim, n_expert, cap);
            q8_K_quantize_kernel<<<dim3(midq_blocks, n_tokens * n_expert, 1), 256>>>(
                    (cuda_block_q8_K *)midq_scratch[dev]->ptr,
                    mid_work, expert_mid_dim, n_tokens * n_expert);
            cudaMemsetAsync(out_work, 0,
                            (uint64_t)n_tokens * out_dim * sizeof(float));
            dim3 ge2((out_dim + 7u) / 8u, 256u, 1);
            glm_routed_moe_down_expert_kernel<<<ge2, 256>>>(
                    out_work, dw,
                    (const cuda_block_q8_K *)midq_scratch[dev]->ptr,
                    counts, lists, (const float *)weights->ptr,
                    down_expert_bytes, down_row_bytes,
                    midq_blocks, out_dim, n_expert, cap);
            return glm_routed_moe_finish_batch(
                    out, out_work, out_work_bytes,
                    "glm routed moe expert-major");
        }
    }
    if (n_tokens == 2u &&
        g_glm_mtp_verify_mode &&
        getenv("DS4_GLM_MTP_NO_MOE_TOK2") == NULL) {
        const uint32_t warps = 8u;
        dim3 g1((expert_mid_dim + warps - 1u) / warps,
                2u * n_expert, 1u);
        const uint32_t sh1 = 2u * xq_blocks *
                             (uint32_t)sizeof(cuda_block_q8_K);
        glm_routed_moe_gateup_tok2_reuse_kernel<<<
                g1, warps * 32u, sh1>>>(
                mid_work, gw, uw,
                (const cuda_block_q8_K *)xq_scratch[dev]->ptr,
                (const int32_t *)selected->ptr,
                gate_expert_bytes, gate_row_bytes,
                up_expert_bytes, up_row_bytes,
                xq_blocks, expert_mid_dim, n_expert);
    } else if (getenv("DS4_GLM_MOE_SCALAR")) {
        dim3 g1(n_tokens, n_expert, 1);
        glm_routed_moe_batch_q2K_gateup_kernel<<<g1, 128>>>(
                mid_work, gw, uw,
                (const cuda_block_q8_K *)xq_scratch[dev]->ptr,
                (const int32_t *)selected->ptr,
                gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes,
                xq_blocks, expert_mid_dim, n_expert, n_tokens, mid_token_stride);
    } else {
        const uint32_t warps = 8u;
        dim3 g1((expert_mid_dim + warps - 1u) / warps, n_expert, n_tokens);
        const uint32_t sh1 = xq_blocks * (uint32_t)sizeof(cuda_block_q8_K);
        glm_routed_moe_gateup_warp_kernel<<<g1, warps * 32u, sh1>>>(
                mid_work, gw, uw,
                (const cuda_block_q8_K *)xq_scratch[dev]->ptr,
                (const int32_t *)selected->ptr,
                gate_expert_bytes, gate_row_bytes, up_expert_bytes, up_row_bytes,
                xq_blocks, expert_mid_dim, n_expert, n_tokens);
    }

    {
        dim3 gq(midq_blocks, n_tokens * n_expert, 1);
        q8_K_quantize_kernel<<<gq, 256>>>(
                (cuda_block_q8_K *)midq_scratch[dev]->ptr,
                mid_work, expert_mid_dim, n_tokens * n_expert);
    }

    if (getenv("DS4_GLM_MOE_SCALAR")) {
        dim3 g2((out_dim + 127u) / 128u, n_tokens, 1);
        glm_routed_moe_batch_q2K_down_kernel<<<g2, 128>>>(
                out_work, dw,
                (const cuda_block_q8_K *)midq_scratch[dev]->ptr,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                down_expert_bytes, down_row_bytes, midq_blocks, out_dim,
                n_expert, n_tokens);
    } else {
        const uint32_t warps = 8u;
        dim3 g2((out_dim + warps - 1u) / warps, n_tokens, 1);
        const uint32_t sh2 = n_expert * midq_blocks *
                             (uint32_t)sizeof(cuda_block_q8_K);
        glm_routed_moe_down_warp_kernel<<<g2, warps * 32u, sh2>>>(
                out_work, dw,
                (const cuda_block_q8_K *)midq_scratch[dev]->ptr,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                down_expert_bytes, down_row_bytes, midq_blocks, out_dim,
                n_expert, n_tokens);
    }
    return glm_routed_moe_finish_batch(
            out, out_work, out_work_bytes,
            "glm routed moe batch launch");
}

extern "C" int ds4_gpu_glm_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        bool                    force_resident) {
    (void)force_resident;
    return ds4_gpu_glm_routed_moe_batch_tensor(out, mid,
            model_map, model_size,
            gate_offset, up_offset, down_offset,
            gate_type, up_type, down_type,
            gate_expert_bytes, gate_row_bytes,
            up_expert_bytes, up_row_bytes,
            down_expert_bytes, down_row_bytes,
            expert_in_dim, expert_mid_dim, out_dim,
            selected, weights, n_total_expert, n_expert, layer_index,
            x, 1, n_expert * expert_mid_dim);
}

