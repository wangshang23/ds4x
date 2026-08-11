/* Parallel router select: 256 threads compute sigmoid probs, then top-k
 * via k rounds of shared-memory argmax over probs+bias (value desc, index
 * asc tie-break — matches the CPU topk_desc). One block per token. */
__global__ static void glm_router_select_parallel_kernel(
        int32_t *selected,
        float *weights_out,
        float *probs_out,
        const float *bias,
        const float *logits,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float expert_weight_scale,
        uint32_t n_tokens) {
    const uint32_t tok = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (tok >= n_tokens) return;
    const float *lg = logits + (uint64_t)tok * n_expert;
    float *probs = probs_out + (uint64_t)tok * n_expert;
    int32_t *sel = selected + (uint64_t)tok * n_expert_used;
    float *w = weights_out + (uint64_t)tok * n_expert_used;

    __shared__ float sh_v[256];
    __shared__ int sh_i[256];
    __shared__ float sh_sel_v[256];
    __shared__ float sh_sum;

    float my_v = -1e30f;
    if (tid < n_expert) {
        const float p = 1.0f / (1.0f + expf(-lg[tid]));
        probs[tid] = p;
        my_v = p + bias[tid];
    }
    if (tid == 0u) sh_sum = 0.0f;
    sh_sel_v[tid] = my_v;
    __syncthreads();

    for (uint32_t k2 = 0; k2 < n_expert_used; k2++) {
        sh_v[tid] = sh_sel_v[tid];
        sh_i[tid] = (int)tid;
        __syncthreads();
        for (uint32_t step = 128u; step > 0u; step >>= 1u) {
            if (tid < step) {
                const float ov = sh_v[tid + step];
                const int oi = sh_i[tid + step];
                if (ov > sh_v[tid] || (ov == sh_v[tid] && oi < sh_i[tid])) {
                    sh_v[tid] = ov;
                    sh_i[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0u) {
            const int best = sh_i[0];
            sel[k2] = best;
            const float p = probs[best];
            w[k2] = p;
            sh_sum += p;
            sh_sel_v[best] = -1e30f;
        }
        __syncthreads();
    }
    if (tid == 0u) {
        float sum = sh_sum;
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        for (uint32_t k2 = 0; k2 < n_expert_used; k2++) {
            w[k2] = w[k2] / sum * expert_weight_scale;
        }
    }
}

__global__ static void glm_router_select_batch_kernel(
        int32_t *selected,
        float *weights_out,
        float *probs_out,
        const float *bias,
        const float *logits,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float expert_weight_scale,
        uint32_t n_tokens) {
    const uint32_t tok = blockIdx.x;
    if (tok >= n_tokens || threadIdx.x != 0u) return;
    const float *lg = logits + (uint64_t)tok * n_expert;
    float *probs = probs_out + (uint64_t)tok * n_expert;
    int32_t *sel = selected + (uint64_t)tok * n_expert_used;
    float *w = weights_out + (uint64_t)tok * n_expert_used;

    for (uint32_t i = 0; i < n_expert; i++) {
        const float p = 1.0f / (1.0f + expf(-lg[i]));
        probs[i] = p;
    }
    /* top-k over probs+bias, ties by smaller index (matches CPU topk_desc) */
    bool taken[384];
    for (uint32_t i = 0; i < n_expert; i++) taken[i] = false;
    float sum = 0.0f;
    for (uint32_t k2 = 0; k2 < n_expert_used; k2++) {
        int best = -1; float bv = -1e30f;
        for (uint32_t i = 0; i < n_expert; i++) {
            if (taken[i]) continue;
            const float v = probs[i] + bias[i];
            if (v > bv) { bv = v; best = (int)i; }
        }
        taken[best] = true;
        sel[k2] = best;
        w[k2] = probs[best];
        sum += probs[best];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t k2 = 0; k2 < n_expert_used; k2++) {
        w[k2] = w[k2] / sum * expert_weight_scale;
    }
}

extern "C" int ds4_gpu_glm_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens) {
    if (!selected || !weights || !probs || !logits || !model_map ||
        n_expert == 0 || n_expert > 384u || n_expert_used == 0 ||
        n_tokens == 0) {
        return 0;
    }
    const uint64_t bb = (uint64_t)n_expert * sizeof(float);
    if (bias_offset > model_size || bb > model_size - bias_offset ||
        logits->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * n_expert * sizeof(float)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const float *bias = (const float *)cuda_resolve_weight_ptr(
            model_map, bias_offset, bb, logical_tier, "glm_exp_probs_b");
    if (!bias) return 0;
    if (n_expert <= 256u && !getenv("DS4_GLM_ROUTER_SCALAR")) {
        glm_router_select_parallel_kernel<<<n_tokens, 256>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                (float *)probs->ptr,
                bias,
                (const float *)logits->ptr,
                n_expert, n_expert_used, expert_weight_scale, n_tokens);
    } else glm_router_select_batch_kernel<<<n_tokens, 32>>>(
            (int32_t *)selected->ptr,
            (float *)weights->ptr,
            (float *)probs->ptr,
            bias,
            (const float *)logits->ptr,
            n_expert, n_expert_used, expert_weight_scale, n_tokens);
    return cuda_ok(cudaGetLastError(), "glm router select batch launch");
}

extern "C" int ds4_gpu_glm_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale) {
    return ds4_gpu_glm_router_select_batch_tensor(selected, weights, probs,
                                                  model_map, model_size,
                                                  bias_offset, logits,
                                                  n_expert, n_expert_used,
                                                  expert_weight_scale, 1u);
}

__global__ static void glm_store_compact_kv_kernel(
        char *kv_lora_cache,
        char *k_rope_cache,
        const float *kv_norm,
        const float *kv_raw,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        uint32_t qk_rope,
        uint32_t cache_f16) {
    const uint32_t token = blockIdx.x;
    const uint32_t part = blockIdx.y;
    if (token >= n_tokens || part > 1u) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    if (part == 0u) {
        const float *src = kv_norm + (uint64_t)token * kv_lora_dim;
        if (cache_f16) {
            __half *dst = (__half *)(kv_lora_cache +
                (uint64_t)pos * kv_lora_dim * sizeof(__half));
            for (uint32_t i = tid; i < kv_lora_dim; i += nth)
                dst[i] = __float2half(src[i]);
        } else {
            float *dst = (float *)(kv_lora_cache +
                (uint64_t)pos * kv_lora_dim * sizeof(float));
            for (uint32_t i = tid; i < kv_lora_dim; i += nth)
                dst[i] = src[i];
        }
    } else {
        const float *src = kv_raw +
            (uint64_t)token * kv_raw_dim + kv_lora_dim;
        if (cache_f16) {
            __half *dst = (__half *)(k_rope_cache +
                (uint64_t)pos * qk_rope * sizeof(__half));
            for (uint32_t i = tid; i < qk_rope; i += nth)
                dst[i] = __float2half(src[i]);
        } else {
            float *dst = (float *)(k_rope_cache +
                (uint64_t)pos * qk_rope * sizeof(float));
            for (uint32_t i = tid; i < qk_rope; i += nth)
                dst[i] = src[i];
        }
    }
}

extern "C" int ds4_gpu_glm_store_compact_kv_tensor(
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_norm,
        const ds4_gpu_tensor *kv_raw,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16) {
    if (!kv_lora_cache || !k_rope_cache || !kv_norm || !kv_raw ||
        n_tokens == 0 || kv_lora_dim == 0 || qk_rope == 0 ||
        kv_lora_dim + qk_rope > kv_raw_dim + qk_rope) {
        return 0;
    }
    const uint64_t es = cache_f16 ? sizeof(__half) : sizeof(float);
    if (kv_norm->bytes < (uint64_t)n_tokens * kv_lora_dim * sizeof(float) ||
        kv_raw->bytes < (uint64_t)n_tokens * kv_raw_dim * sizeof(float) ||
        kv_lora_cache->bytes < (uint64_t)cache_cap * kv_lora_dim * es ||
        k_rope_cache->bytes < (uint64_t)cache_cap * qk_rope * es) {
        return 0;
    }
    dim3 grid(n_tokens, 2, 1);
    glm_store_compact_kv_kernel<<<grid, 128>>>(
            (char *)kv_lora_cache->ptr,
            (char *)k_rope_cache->ptr,
            (const float *)kv_norm->ptr,
            (const float *)kv_raw->ptr,
            pos0, n_tokens, cache_cap, kv_raw_dim, kv_lora_dim, qk_rope,
            cache_f16 ? 1u : 0u);
    return cuda_ok(cudaGetLastError(), "glm store compact kv launch");
}







__global__ static void glm_store_indexer_k_kernel(
        char *cache,
        const float *raw_k,
        const float *w,
        const float *b,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_cap,
        uint32_t head_dim,
        uint32_t rot_dim,
        uint32_t n_ctx_orig,
        float eps,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        uint32_t cache_f16) {
    const uint32_t token = blockIdx.x;
    if (token >= n_tokens) return;
    const uint32_t pos = pos0 + token;
    if (pos >= cache_cap) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    const float *src = raw_k + (uint64_t)token * head_dim;

    __shared__ float scratch[256];
    float sum = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += nth) sum += src[i];
    scratch[tid] = sum;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    const float mean = scratch[0] / (float)head_dim;
    __syncthreads();
    float ss = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += nth) {
        const float d = src[i] - mean;
        ss += d * d;
    }
    scratch[tid] = ss;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / (float)head_dim + eps);

    float corr_dims[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f) {
        corr_dims[0] = fmaxf(0.0f,
            floorf(glm_rope_yarn_corr_factor_dev((int)rot_dim, (int)n_ctx_orig,
                                                 beta_fast, freq_base)));
        corr_dims[1] = fminf((float)rot_dim - 1.0f,
            ceilf(glm_rope_yarn_corr_factor_dev((int)rot_dim, (int)n_ctx_orig,
                                                beta_slow, freq_base)));
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)rot_dim;

    for (uint32_t i = tid; i < head_dim; i += nth) {
        float v0, v1; bool pair = false;
        if (i < rot_dim) {
            if ((i & 1u) != 0u) continue;
            const float theta = theta_base * powf(freq_base, inv_ndims * (float)i);
            float ct, st;
            glm_rope_yarn_dev(theta, freq_scale, corr_dims, (int)i,
                              ext_factor, attn_factor, &ct, &st);
            const float x0 = (src[i] - mean) * inv * w[i] + b[i];
            const float x1 = (src[i + 1u] - mean) * inv * w[i + 1u] + b[i + 1u];
            v0 = x0 * ct - x1 * st;
            v1 = x0 * st + x1 * ct;
            pair = true;
        } else {
            v0 = (src[i] - mean) * inv * w[i] + b[i];
        }
        if (cache_f16) {
            __half *dst = (__half *)(cache + (uint64_t)pos * head_dim * sizeof(__half));
            dst[i] = __float2half(v0);
            if (pair) dst[i + 1u] = __float2half(v1);
        } else {
            float *dst = (float *)(cache + (uint64_t)pos * head_dim * sizeof(float));
            dst[i] = v0;
            if (pair) dst[i + 1u] = v1;
        }
    }
}

extern "C" int ds4_gpu_glm_store_indexer_k_tensor(
        ds4_gpu_tensor       *indexer_key_cache,
        const ds4_gpu_tensor *raw_k,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              bias_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              n_ctx_orig,
        float                 eps,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    if (!indexer_key_cache || !raw_k || !model_map || n_tokens == 0 ||
        head_dim == 0 || head_dim > 256u || (rot_dim & 1u) != 0u) {
        return 0;
    }
    const uint64_t wb = (uint64_t)head_dim * sizeof(float);
    if (weight_offset > model_size || wb > model_size - weight_offset ||
        bias_offset > model_size || wb > model_size - bias_offset ||
        raw_k->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        indexer_key_cache->bytes <
            (uint64_t)cache_cap * head_dim *
                (cache_f16 ? sizeof(__half) : sizeof(float))) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const float *w = (const float *)cuda_resolve_weight_ptr(
            model_map, weight_offset, wb, logical_tier, "glm_indexer_k_norm");
    const float *b = (const float *)cuda_resolve_weight_ptr(
            model_map, bias_offset, wb, logical_tier, "glm_indexer_k_norm_b");
    if (!w || !b) return 0;
    glm_store_indexer_k_kernel<<<n_tokens, 128>>>(
            (char *)indexer_key_cache->ptr,
            (const float *)raw_k->ptr,
            w, b, pos0, n_tokens, cache_cap, head_dim, rot_dim, n_ctx_orig,
            eps, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, cache_f16 ? 1u : 0u);
    return cuda_ok(cudaGetLastError(), "glm store indexer k launch");
}

extern "C" int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table || !selected || n_selected == 0 ||
        selected->bytes < (uint64_t)n_selected * sizeof(int32_t)) {
        return 0;
    }
    std::vector<int32_t> ids;
    try {
        ids.resize(n_selected);
    } catch (...) {
        return 0;
    }
    if (!cuda_ok(cudaMemcpy(ids.data(), selected->ptr,
                            (size_t)n_selected * sizeof(int32_t),
                            cudaMemcpyDeviceToHost),
                 "GLM streaming selected-id read")) {
        return 0;
    }
    return cuda_stream_selected_cache_begin_load(table, ids.data(), n_selected);
}

__global__ static void glm_value_project_q8_0_batch_heads_kernel(
        float *heads,
        const char *weight,
        const float *lora,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t value_dim,
        uint64_t row_bytes) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (head >= n_head || token >= n_tokens) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    extern __shared__ float xsh[];
    const float *src = lora + (uint64_t)token * n_head * kv_lora_dim +
                       (uint64_t)head * kv_lora_dim;
    float *out = heads + (uint64_t)token * n_head * value_dim +
                 (uint64_t)head * value_dim;
    for (uint32_t j = tid; j < kv_lora_dim; j += nth) xsh[j] = src[j];
    __syncthreads();
    for (uint32_t d = tid; d < value_dim; d += nth) {
        const char *row = weight + ((uint64_t)head * value_dim + d) * row_bytes;
        out[d] = glm_q8_0_dot_row_dev(row, xsh, kv_lora_dim);
    }
}

/* Reuse each Q8 row across a token tile while retaining the scalar kernel's
 * block/k accumulation order independently for every output token. */
template <uint32_t token_tile>
__global__ static void glm_value_project_q8_0_batch_heads_tiled_kernel(
        float *heads,
        const char *weight,
        const float *lora,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t value_dim,
        uint64_t row_bytes) {
    const uint32_t head = blockIdx.x;
    const uint32_t token0 = blockIdx.y * token_tile;
    if (head >= n_head || token0 >= n_tokens) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    extern __shared__ float xsh[];

#pragma unroll
    for (uint32_t t = 0; t < token_tile; t++) {
        const uint32_t token = token0 + t;
        if (token >= n_tokens) break;
        const float *src = lora + (uint64_t)token * n_head * kv_lora_dim +
                           (uint64_t)head * kv_lora_dim;
        for (uint32_t j = tid; j < kv_lora_dim; j += nth) {
            xsh[(uint64_t)t * kv_lora_dim + j] = src[j];
        }
    }
    __syncthreads();

    for (uint32_t od = tid; od < value_dim; od += nth) {
        const char *row = weight +
            ((uint64_t)head * value_dim + od) * row_bytes;
        float acc[token_tile] = { 0.0f };
        const uint32_t nb = kv_lora_dim >> 5;
        for (uint32_t b = 0; b < nb; b++) {
            const char *blk = row + (uint64_t)b * 34u;
            const float d = __half2float(*(const __half *)blk);
            const int8_t *q = (const int8_t *)(blk + 2);
            float s[token_tile] = { 0.0f };
#pragma unroll 8
            for (uint32_t k = 0; k < 32u; k++) {
                const float w = (float)q[k];
#pragma unroll
                for (uint32_t t = 0; t < token_tile; t++) {
                    if (token0 + t < n_tokens) {
                        s[t] += w * xsh[(uint64_t)t * kv_lora_dim +
                                        b * 32u + k];
                    }
                }
            }
#pragma unroll
            for (uint32_t t = 0; t < token_tile; t++) {
                if (token0 + t < n_tokens) acc[t] += d * s[t];
            }
        }
#pragma unroll
        for (uint32_t t = 0; t < token_tile; t++) {
            const uint32_t token = token0 + t;
            if (token < n_tokens) {
                heads[(uint64_t)token * n_head * value_dim +
                      (uint64_t)head * value_dim + od] = acc[t];
            }
        }
    }
}

extern "C" int ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim) {
    if (!heads || !lora || !model_map || n_tokens == 0 || n_head == 0 ||
        kv_lora_dim == 0 || (kv_lora_dim & 31u) != 0u || value_dim == 0) {
        return 0;
    }
    if (weight_type != 8u) {
        fprintf(stderr, "ds4: glm value project: unsupported type %u\n",
                weight_type);
        return 0;
    }
    const uint64_t row_bytes = ((uint64_t)kv_lora_dim / 32u) * 34u;
    const uint64_t wbytes = (uint64_t)n_head * value_dim * row_bytes;
    if (weight_offset > model_size || wbytes > model_size - weight_offset ||
        lora->bytes < (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * value_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const char *w = (const char *)cuda_resolve_weight_ptr(
            model_map, weight_offset, wbytes, logical_tier, "glm_v_b");
    if (!w) return 0;
    if (n_tokens >= 16u && getenv("DS4_GLM_VALUE_NO_TILE16") == NULL) {
        dim3 grid(n_head, (n_tokens + 15u) / 16u, 1);
        const size_t shmem = 16ull * kv_lora_dim * sizeof(float);
        glm_value_project_q8_0_batch_heads_tiled_kernel<16><<<grid, 128, shmem>>>(
                (float *)heads->ptr, w, (const float *)lora->ptr,
                n_tokens, n_head, kv_lora_dim, value_dim, row_bytes);
        return cuda_ok(cudaGetLastError(), "glm value project tile16 launch");
    }
    dim3 grid(n_head, n_tokens, 1);
    const size_t shmem = (size_t)kv_lora_dim * sizeof(float);
    glm_value_project_q8_0_batch_heads_kernel<<<grid, 128, shmem>>>(
            (float *)heads->ptr, w, (const float *)lora->ptr,
            n_tokens, n_head, kv_lora_dim, value_dim, row_bytes);
    return cuda_ok(cudaGetLastError(), "glm value project launch");
}

/* Decode-time (n_tok small) quant matvec. The Metal "mpp/model-view"
 * variant is a bandwidth-tuned matvec; on CUDA the generic quant matmul
 * already dispatches per type, so delegate. Revisit in the perf pass. */
extern "C" int ds4_gpu_matmul_quant_tensor(ds4_gpu_tensor *out,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok);

extern "C" int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    return ds4_gpu_matmul_quant_tensor(out, model_map, model_size,
                                       weight_offset, weight_type,
                                       in_dim, out_dim, x, n_tok);
}

extern "C" int ds4_gpu_matmul_quant_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_matmul_quant_rows_scalar_tensor\n");
    return 0;
}

extern "C" int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    switch (weight_type) {
    case 8u:   /* Q8_0 */
        return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                          weight_offset, in_dim, out_dim,
                                          x, n_tok);
    case 1u:   /* F16 */
        return ds4_gpu_matmul_f16_tensor(out, model_map, model_size,
                                         weight_offset, in_dim, out_dim,
                                         x, n_tok);
    default:
        fprintf(stderr, "ds4: matmul_quant: unsupported type %u\n",
                weight_type);
        return 0;
    }
}

extern "C" uint64_t ds4_gpu_recommended_working_set_size(void) {
    /* GLM graph memory guard: on this backend the model weights are
     * distributed across all devices by the multi-tier placement, so the
     * relevant budget is the aggregate VRAM. */
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return 0;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return (uint64_t)total_b * (uint64_t)n;
}

extern "C" int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected, uint32_t n_selected) {
    (void)selected;
    (void)n_selected;
    return 1;
}

extern "C" void ds4_gpu_set_glm_streaming_prefill_full_layer(bool enabled) {
    (void)enabled;   /* SSD streaming is not used on the CUDA backend */
}

extern "C" void ds4_gpu_set_glm_mtp_verify_mode(bool enabled) {
    g_glm_mtp_verify_mode = enabled;
}

extern "C" int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) {
        return 0;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size || sizes[i] == 0 ||
            sizes[i] > model_size - offsets[i]) {
            return 0;
        }
    }
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    if (getenv("DS4_CUDA_COPY_MODEL_CHUNKED") != NULL) {
        for (uint32_t i = 0; i < count; i++) {
            (void)cuda_model_prefetch_range(model_map, model_size,
                                            offsets[i], sizes[i]);
        }
    }
    return 1;
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok, float clamp);

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    return ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(gate, up, mid,
            model_map, model_size, gate_offset, up_offset,
            in_dim, out_dim, x, 1, clamp);
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    (void)gate; (void)up; (void)mid; (void)model_map; (void)model_size;
    (void)gate_offset; (void)up_offset; (void)in_dim; (void)out_dim;
    (void)x; (void)n_tok; (void)clamp;
    return 0;
}

/* Fused single-token shared-expert gate+up+swiglu: one warp per output
 * row computes both q8_0 dots against a shared-staged f32 x and writes
 * silu(gate)*up directly. Falls back to the split path for n_tok > 1. */
__global__ static void glm_shared_gate_up_swiglu_one_kernel(
        float *mid,
        const char *gw,
        const char *uw,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim,
        float clamp) {
    extern __shared__ float glm_sgu_sh[];
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        glm_sgu_sh[i] = x[i];
    }
    __syncthreads();
    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= out_dim) return;
    const uint32_t nblk = in_dim >> 5;
    const uint64_t row_bytes = (uint64_t)nblk * 34u;
    const char *grow = gw + (uint64_t)r * row_bytes;
    const char *urow = uw + (uint64_t)r * row_bytes;
    float g = 0.0f, u = 0.0f;
    for (uint32_t blk = lane; blk < nblk; blk += 32u) {
        const char *gb = grow + (uint64_t)blk * 34u;
        const char *ub = urow + (uint64_t)blk * 34u;
        const float gd = __half2float(*(const __half *)gb);
        const float ud = __half2float(*(const __half *)ub);
        const int8_t *gq = (const int8_t *)(gb + 2);
        const int8_t *uq = (const int8_t *)(ub + 2);
        const float *xs = glm_sgu_sh + blk * 32u;
        float gs = 0.0f, us = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < 32; k++) {
            gs += (float)gq[k] * xs[k];
            us += (float)uq[k] * xs[k];
        }
        g += gd * gs;
        u += ud * us;
    }
    for (int off = 16; off > 0; off >>= 1) {
        g += __shfl_down_sync(0xffffffffu, g, off);
        u += __shfl_down_sync(0xffffffffu, u, off);
    }
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        mid[r] = (g / (1.0f + expf(-g))) * u;
    }
}

/* Two-token verifier variant of the decode kernel above. Each warp loads a
 * gate/up weight row once, while each token keeps the decode kernel's block
 * order and warp reduction tree independently. */
__global__ static void glm_shared_gate_up_swiglu_tok2_exact_kernel(
        float *mid,
        const char *gw,
        const char *uw,
        const float *x,
        uint32_t in_dim,
        uint32_t out_dim,
        float clamp) {
    extern __shared__ float glm_sgu2_sh[];
    float *x0 = glm_sgu2_sh;
    float *x1 = glm_sgu2_sh + in_dim;
    const uint32_t warps = blockDim.x >> 5;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        x0[i] = x[i];
        x1[i] = x[in_dim + i];
    }
    __syncthreads();
    const uint32_t r = blockIdx.x * warps + warp;
    if (r >= out_dim) return;
    const uint32_t nblk = in_dim >> 5;
    const uint64_t row_bytes = (uint64_t)nblk * 34u;
    const char *grow = gw + (uint64_t)r * row_bytes;
    const char *urow = uw + (uint64_t)r * row_bytes;
    float g0 = 0.0f, u0 = 0.0f;
    float g1 = 0.0f, u1 = 0.0f;
    for (uint32_t blk = lane; blk < nblk; blk += 32u) {
        const char *gb = grow + (uint64_t)blk * 34u;
        const char *ub = urow + (uint64_t)blk * 34u;
        const float gd = __half2float(*(const __half *)gb);
        const float ud = __half2float(*(const __half *)ub);
        const int8_t *gq = (const int8_t *)(gb + 2);
        const int8_t *uq = (const int8_t *)(ub + 2);
        const float *xs0 = x0 + blk * 32u;
        const float *xs1 = x1 + blk * 32u;
        float gs0 = 0.0f, us0 = 0.0f;
        float gs1 = 0.0f, us1 = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < 32; k++) {
            const float gk = (float)gq[k];
            const float uk = (float)uq[k];
            gs0 += gk * xs0[k];
            us0 += uk * xs0[k];
            gs1 += gk * xs1[k];
            us1 += uk * xs1[k];
        }
        g0 += gd * gs0;
        u0 += ud * us0;
        g1 += gd * gs1;
        u1 += ud * us1;
    }
    for (int off = 16; off > 0; off >>= 1) {
        g0 += __shfl_down_sync(0xffffffffu, g0, off);
        u0 += __shfl_down_sync(0xffffffffu, u0, off);
        g1 += __shfl_down_sync(0xffffffffu, g1, off);
        u1 += __shfl_down_sync(0xffffffffu, u1, off);
    }
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            if (g0 > clamp) g0 = clamp;
            if (u0 > clamp) u0 = clamp;
            if (u0 < -clamp) u0 = -clamp;
            if (g1 > clamp) g1 = clamp;
            if (u1 > clamp) u1 = clamp;
            if (u1 < -clamp) u1 = -clamp;
        }
        mid[r] = (g0 / (1.0f + expf(-g0))) * u0;
        mid[out_dim + r] = (g1 / (1.0f + expf(-g1))) * u1;
    }
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp) {
    if (!gate || !up || !mid || !x || n_tok == 0) return 0;
    if (n_tok == 2 && (in_dim & 31u) == 0u &&
        g_glm_mtp_verify_mode &&
        getenv("DS4_GLM_MTP_NO_SHARED_TOK2") == NULL &&
        mid->bytes >= 2u * out_dim * sizeof(float) &&
        x->bytes >= 2u * in_dim * sizeof(float)) {
        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        const uint64_t wb = out_dim * row_bytes;
        if (gate_offset <= model_size && wb <= model_size - gate_offset &&
            up_offset <= model_size && wb <= model_size - up_offset) {
            const int logical_tier = cuda_current_tier();
            const char *gw = cuda_resolve_weight_ptr(model_map, gate_offset,
                    wb, logical_tier, "glm_shared_gate");
            const char *uw = cuda_resolve_weight_ptr(model_map, up_offset,
                    wb, logical_tier, "glm_shared_up");
            if (gw && uw) {
                const uint32_t warps = 8u;
                const uint32_t sh = 2u * (uint32_t)in_dim * sizeof(float);
                glm_shared_gate_up_swiglu_tok2_exact_kernel
                        <<<(unsigned)((out_dim + warps - 1u) / warps),
                           warps * 32u, sh>>>(
                        (float *)mid->ptr, gw, uw, (const float *)x->ptr,
                        (uint32_t)in_dim, (uint32_t)out_dim, clamp);
                return cuda_ok(cudaGetLastError(),
                               "glm shared swiglu tok2 exact");
            }
        }
    }
    if (n_tok == 1 && (in_dim & 31u) == 0u &&
        !getenv("DS4_GLM_SHARED_SPLIT") &&
        mid->bytes >= out_dim * sizeof(float) &&
        x->bytes >= in_dim * sizeof(float)) {
        const uint64_t row_bytes = (in_dim / 32u) * 34u;
        const uint64_t wb = out_dim * row_bytes;
        if (gate_offset <= model_size && wb <= model_size - gate_offset &&
            up_offset <= model_size && wb <= model_size - up_offset) {
            const int logical_tier = cuda_current_tier();
            const char *gw = cuda_resolve_weight_ptr(model_map, gate_offset,
                    wb, logical_tier, "glm_shared_gate");
            const char *uw = cuda_resolve_weight_ptr(model_map, up_offset,
                    wb, logical_tier, "glm_shared_up");
            if (gw && uw) {
                const uint32_t warps = 8u;
                const uint32_t sh = (uint32_t)in_dim * sizeof(float);
                glm_shared_gate_up_swiglu_one_kernel
                        <<<(unsigned)((out_dim + warps - 1u) / warps),
                           warps * 32u, sh>>>(
                        (float *)mid->ptr, gw, uw, (const float *)x->ptr,
                        (uint32_t)in_dim, (uint32_t)out_dim, clamp);
                return cuda_ok(cudaGetLastError(), "glm shared swiglu one");
            }
        }
    }
    if (!ds4_gpu_matmul_q8_0_tensor(gate, model_map, model_size, gate_offset,
                                    in_dim, out_dim, x, n_tok) ||
        !ds4_gpu_matmul_q8_0_tensor(up, model_map, model_size, up_offset,
                                    in_dim, out_dim, x, n_tok)) {
        return 0;
    }
    return ds4_gpu_swiglu_tensor(mid, gate, up,
                                 (uint32_t)(out_dim * n_tok), clamp, 1.0f);
}

extern "C" int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    static ds4_gpu_tensor *gu_scratch[DS4_MAX_GPUS][2] = {{0}};
    const int dev = cuda_current_tier();
    const uint64_t need = out_dim * sizeof(float);
    for (int i = 0; i < 2; i++) {
        if (!gu_scratch[dev][i] || gu_scratch[dev][i]->bytes < need) {
            if (gu_scratch[dev][i]) ds4_gpu_tensor_free(gu_scratch[dev][i]);
            gu_scratch[dev][i] = ds4_gpu_tensor_alloc(need);
        }
        if (!gu_scratch[dev][i]) return 0;
    }
    return ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
            gu_scratch[dev][0], gu_scratch[dev][1], mid,
            model_map, model_size, gate_offset, up_offset,
            in_dim, out_dim, x, 1, clamp);
}

extern "C" int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value) {
    if (event_value) *event_value = 1;
    return cuda_ok(cudaDeviceSynchronize(), "selected readback signal");
}

extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    return cuda_stream_selected_cache_begin_load(table, selected_ids,
                                                 n_selected);
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    return 0;
}

extern "C" int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count) {
    if (!dst || !src) return 0;
    if (count == 0) return 1;
    if (count > UINT64_MAX / sizeof(float) ||
        count > UINT64_MAX / sizeof(__half)) {
        return 0;
    }
    const uint64_t src_bytes = count * sizeof(float);
    const uint64_t dst_bytes = count * sizeof(__half);
    if (src_offset > src->bytes || src_bytes > src->bytes - src_offset ||
        dst_offset > dst->bytes || dst_bytes > dst->bytes - dst_offset ||
        ds4_tensor_device_idx(dst) != ds4_tensor_device_idx(src)) {
        return 0;
    }
    const int tier = ds4_tensor_device_idx(dst);
    if (ds4_gpu_set_current_device(tier) != 0) return 0;
    const uint64_t blocks = (count + 255u) / 256u;
    if (blocks > UINT32_MAX) return 0;
    f32_to_f16_kernel<<<(unsigned)blocks, 256>>>(
            (__half *)((char *)dst->ptr + dst_offset),
            (const float *)((const char *)src->ptr + src_offset),
            count);
    return cuda_ok(cudaGetLastError(), "tensor f32-to-f16 copy launch");
}

extern "C" int ds4_gpu_tensor_read_after_selected_event(const ds4_gpu_tensor *tensor,
                                             uint64_t offset,
                                             void *data,
                                             uint64_t bytes,
                                             uint64_t event_value,
                                             const char *label) {
    (void)event_value;
    if (!tensor || !data || offset > tensor->bytes ||
        bytes > tensor->bytes - offset) {
        return 0;
    }
    if (!cuda_ok(cudaDeviceSynchronize(),
                 label ? label : "selected readback wait")) {
        return 0;
    }
    return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset,
                              (size_t)bytes, cudaMemcpyDeviceToHost),
                   "selected tensor read");
}

extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                               const ds4_gpu_tensor *out_t,
                               ds4_gpu_tensor *in_t,
                               uint64_t bytes) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_tp_big_gate_encode\n");
    return 0;
}

extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_tp_gate_encode\n");
    return 0;
}

extern "C" void ds4_gpu_tp_set_attn_head_split(int enabled) {
    (void)enabled;   /* Mac network-TP head split: no-op on CUDA */
}

extern "C" int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label) {
    (void)event_value;
    return cuda_ok(cudaDeviceSynchronize(),
                   label ? label : "selected readback wait");
}

