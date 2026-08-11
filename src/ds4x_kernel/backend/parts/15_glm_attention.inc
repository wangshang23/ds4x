/* ===== GLM 5.2 stubs (to be implemented; fail loudly) ===== */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
__global__ static void add3_kernel(float *out, const float *a,
                                   const float *b, const float *c,
                                   uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i] + c[i];
}

extern "C" int ds4_gpu_add3_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *c,
        uint32_t                n) {
    if (!out || !a || !b || !c || n == 0 ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float) ||
        c->bytes < (uint64_t)n * sizeof(float)) {
        return 0;
    }
    add3_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr, (const float *)a->ptr,
            (const float *)b->ptr, (const float *)c->ptr, n);
    return cuda_ok(cudaGetLastError(), "add3 launch");
}

/* Fused decode residual: sum_out = a + b; norm_out = rmsnorm(sum) * w.
 * Single row, one block (two-pass over n with a shared reduction). */
__global__ static void glm_add_rms_norm_weight_kernel(
        float *norm_out,
        float *sum_out,
        const float *a,
        const float *b,
        const float *w,
        uint32_t n,
        float eps) {
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    __shared__ float sh[32];
    float sumsq = 0.0f;
    for (uint32_t i = tid; i < n; i += nth) {
        const float v = a[i] + b[i];
        sum_out[i] = v;
        sumsq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        sumsq += __shfl_xor_sync(0xffffffffu, sumsq, off);
    }
    if ((tid & 31u) == 0u) sh[tid >> 5] = sumsq;
    __syncthreads();
    if (tid < 32u) {
        sumsq = (tid < (nth + 31u) / 32u) ? sh[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            sumsq += __shfl_xor_sync(0xffffffffu, sumsq, off);
        }
        if (tid == 0u) sh[0] = sumsq;
    }
    __syncthreads();
    const float scale = rsqrtf(sh[0] / (float)n + eps);
    for (uint32_t i = tid; i < n; i += nth) {
        norm_out[i] = (sum_out[i] * scale) * w[i];
    }
}

extern "C" int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps) {
    if (!norm_out || !sum_out || !a || !b || !model_map || n == 0 ||
        norm_out->bytes < (uint64_t)n * sizeof(float) ||
        sum_out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float) ||
        weight_offset > model_size ||
        (uint64_t)n * sizeof(float) > model_size - weight_offset) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const float *w = (const float *)cuda_resolve_weight_ptr(
            model_map, weight_offset, (uint64_t)n * sizeof(float),
            logical_tier, "rms_weight");
    if (!w) return 0;
    glm_add_rms_norm_weight_kernel<<<1, 1024>>>(
            (float *)norm_out->ptr, (float *)sum_out->ptr,
            (const float *)a->ptr, (const float *)b->ptr, w, n, eps);
    return cuda_ok(cudaGetLastError(), "add rms norm weight");
}

extern "C" bool ds4_gpu_commands_active(void) {
    return false;
}

__global__ static void glm_embed_token_q8_0_kernel(
        float *out,
        const unsigned char *w,
        uint32_t token,
        uint32_t n_embd) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) return;
    const uint64_t row_blocks = n_embd / 32u;
    const unsigned char *blk =
        w + ((uint64_t)token * row_blocks + (d >> 5)) * 34u;
    const float scale = __half2float(*(const __half *)blk);
    out[d] = scale * (float)((const int8_t *)(blk + 2))[d & 31u];
}

extern "C" int ds4_gpu_embed_token_quant_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          weight_type,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd) {
    if (!out || !model_map || n_embd == 0 || (n_embd & 31u) != 0u ||
        token >= n_vocab) {
        return 0;
    }
    if (weight_type != 8u) {   /* DS4_TENSOR_Q8_0 */
        fprintf(stderr, "ds4: embed_token_quant: unsupported type %u\n",
                weight_type);
        return 0;
    }
    const uint64_t row_bytes = ((uint64_t)n_embd / 32u) * 34u;
    if (weight_offset > model_size ||
        (uint64_t)n_vocab * row_bytes > model_size - weight_offset ||
        out->bytes < (uint64_t)n_embd * sizeof(float)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const unsigned char *w = (const unsigned char *)cuda_resolve_weight_ptr(
            model_map, weight_offset, (uint64_t)n_vocab * row_bytes,
            logical_tier, "glm_token_embd");
    if (!w) return 0;
    glm_embed_token_q8_0_kernel<<<(n_embd + 255) / 256, 256>>>(
            (float *)out->ptr, w, token, n_embd);
    return cuda_ok(cudaGetLastError(), "glm embed token launch");
}

__global__ static void glm_embed_tokens_q8_0_kernel(
        float *out,
        const int32_t *tokens,
        const unsigned char *w,
        uint32_t n_tokens,
        uint32_t n_embd) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_embd;
    if (gid >= n) return;
    uint32_t t = gid / n_embd;
    uint32_t d = gid - (uint64_t)t * n_embd;
    int32_t tok = tokens[t];
    const uint64_t row_blocks = n_embd / 32u;
    const unsigned char *blk = w + ((uint64_t)tok * row_blocks + (d >> 5)) * 34u;
    const float scale = __half2float(*(const __half *)blk);
    out[gid] = scale * (float)((const int8_t *)(blk + 2))[d & 31u];
}

extern "C" int ds4_gpu_embed_tokens_quant_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd) {
    if (!out || !tokens || !model_map || n_tokens == 0 || n_embd == 0 ||
        (n_embd & 31u) != 0u) {
        return 0;
    }
    if (weight_type != 8u) {   /* DS4_TENSOR_Q8_0 */
        fprintf(stderr, "ds4: embed_tokens_quant: unsupported type %u\n",
                weight_type);
        return 0;
    }
    const uint64_t row_bytes = ((uint64_t)n_embd / 32u) * 34u;
    if (weight_offset > model_size ||
        (uint64_t)n_vocab * row_bytes > model_size - weight_offset ||
        out->bytes < (uint64_t)n_tokens * n_embd * sizeof(float) ||
        tokens->bytes < (uint64_t)n_tokens * sizeof(int32_t)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const unsigned char *w = (const unsigned char *)cuda_resolve_weight_ptr(
            model_map, weight_offset, (uint64_t)n_vocab * row_bytes,
            logical_tier, "glm_token_embd");
    if (!w) return 0;
    uint64_t n = (uint64_t)n_tokens * n_embd;
    glm_embed_tokens_q8_0_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr,
            (const int32_t *)tokens->ptr,
            w, n_tokens, n_embd);
    return cuda_ok(cudaGetLastError(), "glm embed tokens launch");
}

extern "C" int ds4_gpu_flush_encoder(void) {
    /* Metal encoder flush: CUDA kernels are already queued in stream
     * order, nothing to split. */
    return 1;
}

extern "C" int ds4_gpu_glm_attention_flash_staged_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_attention_flash_staged_tensor\n");
    return 0;
}

extern "C" int ds4_gpu_glm_attention_flash_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_attention_flash_tensor\n");
    return 0;
}

extern "C" int ds4_gpu_glm_attention_full_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_attention_full_tensor\n");
    return 0;
}

template <typename CT>
__device__ __forceinline__ static float2 glm_cache_rope_pair_f16_dev(
        const CT *rope_cache, uint64_t rope_base, uint32_t r,
        uint32_t row, uint32_t qk_rope, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, const float corr_dims[2]) {
    const float theta_base = (float)row;
    const float inv_ndims = -1.0f / (float)qk_rope;
    const float theta = theta_base * powf(freq_base, inv_ndims * (float)r);
    float ct, st;
    glm_rope_yarn_dev(theta, freq_scale, corr_dims, (int)r,
                      ext_factor, attn_factor, &ct, &st);
    const float x0 = (float)rope_cache[rope_base + r];
    const float x1 = (float)rope_cache[rope_base + r + 1u];
    return make_float2(x0 * ct - x1 * st, x0 * st + x1 * ct);
}

/* Scalar-correct MLA attention: one warp per head, grid
 * (ceil(n_head/8), n_tokens). The row loop handles either a contiguous
 * causal range or an explicit selected-row list and mirrors the Metal online
 * softmax so numerics stay comparable. */
template <typename CT, bool selected_rows>
__global__ static void glm_attention_lora_causal_kernel(
        float *lora_out,
        const float *q,
        const float *qk_low,
        const CT *kv_lora_cache,
        const CT *k_rope_cache,
        const uint32_t *selected,
        uint32_t cache_cap,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_selected,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float scale) {
    const uint32_t token = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t head = blockIdx.x * 8u + warp;
    if (token >= n_tokens || head >= n_head || kv_lora_dim != 512u ||
        qk_rope != 64u) {
        return;
    }
    const uint32_t visible = selected_rows
        ? n_selected : min(n_selected, pos0 + token + 1u);
    if (visible == 0u) return;

    const uint32_t qk_dim = qk_nope + qk_rope;
    const float *qh = q + (uint64_t)token * n_head * qk_dim +
                      (uint64_t)head * qk_dim;
    const float4 *low4 = (const float4 *)(qk_low +
        (uint64_t)token * n_head * kv_lora_dim +
        (uint64_t)head * kv_lora_dim);

    float4 low0 = low4[lane];
    float4 low1 = low4[lane + 32u];
    float4 low2 = low4[lane + 64u];
    float4 low3 = low4[lane + 96u];
    float4 qrope = make_float4(0.f, 0.f, 0.f, 0.f);
    const uint32_t rope_vecs = qk_rope >> 2;   /* 16 */
    if (lane < rope_vecs) {
        qrope = *((const float4 *)(qh + qk_nope + lane * 4u));
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f) {
        corr_dims[0] = fmaxf(0.0f,
            floorf(glm_rope_yarn_corr_factor_dev((int)qk_rope, (int)n_ctx_orig,
                                                 beta_fast, freq_base)));
        corr_dims[1] = fminf((float)qk_rope - 1.0f,
            ceilf(glm_rope_yarn_corr_factor_dev((int)qk_rope, (int)n_ctx_orig,
                                                beta_slow, freq_base)));
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = make_float4(0.f,0.f,0.f,0.f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t ri = 0u; ri < visible; ri++) {
        const uint32_t row = selected_rows
            ? selected[(uint64_t)token * n_selected + ri] : ri;
        if (row >= cache_cap) continue;
        const CT *kvrow = kv_lora_cache + (uint64_t)row * kv_lora_dim;
        float partial = 0.0f;
        {
            const float4 k0 = make_float4(
                (float)(kvrow[lane*4u+0u]), (float)(kvrow[lane*4u+1u]),
                (float)(kvrow[lane*4u+2u]), (float)(kvrow[lane*4u+3u]));
            const float4 k1 = make_float4(
                (float)(kvrow[(lane+32u)*4u+0u]), (float)(kvrow[(lane+32u)*4u+1u]),
                (float)(kvrow[(lane+32u)*4u+2u]), (float)(kvrow[(lane+32u)*4u+3u]));
            const float4 k2 = make_float4(
                (float)(kvrow[(lane+64u)*4u+0u]), (float)(kvrow[(lane+64u)*4u+1u]),
                (float)(kvrow[(lane+64u)*4u+2u]), (float)(kvrow[(lane+64u)*4u+3u]));
            const float4 k3 = make_float4(
                (float)(kvrow[(lane+96u)*4u+0u]), (float)(kvrow[(lane+96u)*4u+1u]),
                (float)(kvrow[(lane+96u)*4u+2u]), (float)(kvrow[(lane+96u)*4u+3u]));
            partial += low0.x*k0.x + low0.y*k0.y + low0.z*k0.z + low0.w*k0.w;
            partial += low1.x*k1.x + low1.y*k1.y + low1.z*k1.z + low1.w*k1.w;
            partial += low2.x*k2.x + low2.y*k2.y + low2.z*k2.z + low2.w*k2.w;
            partial += low3.x*k3.x + low3.y*k3.y + low3.z*k3.z + low3.w*k3.w;
            if (lane < rope_vecs) {
                const uint64_t rope_base = (uint64_t)row * qk_rope;
                const uint32_t r = lane * 4u;
                const float2 y0 = glm_cache_rope_pair_f16_dev(
                        k_rope_cache, rope_base, r, row, qk_rope, freq_base,
                        freq_scale, ext_factor, attn_factor, corr_dims);
                const float2 y1 = glm_cache_rope_pair_f16_dev(
                        k_rope_cache, rope_base, r + 2u, row, qk_rope,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        corr_dims);
                partial += qrope.x*y0.x + qrope.y*y0.y +
                           qrope.z*y1.x + qrope.w*y1.y;
            }
            for (uint32_t off = 16u; off > 0u; off >>= 1u) {
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            }
            const float score = partial * scale;
            const float new_m = fmaxf(M, score);
            const float old_scale = expf(M - new_m);
            const float row_scale = expf(score - new_m);
            o0.x = o0.x*old_scale + k0.x*row_scale; o0.y = o0.y*old_scale + k0.y*row_scale;
            o0.z = o0.z*old_scale + k0.z*row_scale; o0.w = o0.w*old_scale + k0.w*row_scale;
            o1.x = o1.x*old_scale + k1.x*row_scale; o1.y = o1.y*old_scale + k1.y*row_scale;
            o1.z = o1.z*old_scale + k1.z*row_scale; o1.w = o1.w*old_scale + k1.w*row_scale;
            o2.x = o2.x*old_scale + k2.x*row_scale; o2.y = o2.y*old_scale + k2.y*row_scale;
            o2.z = o2.z*old_scale + k2.z*row_scale; o2.w = o2.w*old_scale + k2.w*row_scale;
            o3.x = o3.x*old_scale + k3.x*row_scale; o3.y = o3.y*old_scale + k3.y*row_scale;
            o3.z = o3.z*old_scale + k3.z*row_scale; o3.w = o3.w*old_scale + k3.w*row_scale;
            S = S*old_scale + row_scale;
            M = new_m;
        }
    }

    const float inv_s = S > 0.0f ? 1.0f / S : 0.0f;
    float4 *out4 = (float4 *)(lora_out +
        ((uint64_t)token * n_head + head) * kv_lora_dim);
    o0.x*=inv_s; o0.y*=inv_s; o0.z*=inv_s; o0.w*=inv_s;
    o1.x*=inv_s; o1.y*=inv_s; o1.z*=inv_s; o1.w*=inv_s;
    o2.x*=inv_s; o2.y*=inv_s; o2.z*=inv_s; o2.w*=inv_s;
    o3.x*=inv_s; o3.y*=inv_s; o3.z*=inv_s; o3.w*=inv_s;
    out4[lane] = o0;
    out4[lane + 32u] = o1;
    out4[lane + 64u] = o2;
    out4[lane + 96u] = o3;
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    if (!lora_out || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        n_tokens == 0 || n_head == 0 || kv_lora_dim != 512u ||
        qk_rope != 64u) {
        fprintf(stderr, "ds4: glm attn lora causal: unsupported config "
                "(n_tok=%u head=%u lora=%u rope=%u f16=%d)\n",
                n_tokens, n_head, kv_lora_dim, qk_rope, (int)cache_f16);
        return 0;
    }
    const float scale = 1.0f / sqrtf((float)(qk_nope + qk_rope));
    dim3 grid((n_head + 7u) / 8u, n_tokens, 1);
    if (cache_f16) {
        glm_attention_lora_causal_kernel<__half, false><<<grid, 256>>>(
                (float *)lora_out->ptr,
                (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const __half *)kv_lora_cache->ptr,
                (const __half *)k_rope_cache->ptr,
                NULL, cache_cap,
                n_tokens, pos0, n_selected, n_head, kv_lora_dim, qk_nope,
                qk_rope, n_ctx_orig, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow, scale);
    } else {
        glm_attention_lora_causal_kernel<float, false><<<grid, 256>>>(
                (float *)lora_out->ptr,
                (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const float *)kv_lora_cache->ptr,
                (const float *)k_rope_cache->ptr,
                NULL, cache_cap,
                n_tokens, pos0, n_selected, n_head, kv_lora_dim, qk_nope,
                qk_rope, n_ctx_orig, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow, scale);
    }
    return cuda_ok(cudaGetLastError(), "glm attn lora causal launch");
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    const uint64_t cache_elem = cache_f16 ? sizeof(__half) : sizeof(float);
    const uint64_t qk_dim = (uint64_t)qk_nope + qk_rope;
    if (!lora_out || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        !selected || n_tokens == 0u || n_selected == 0u || n_head == 0u ||
        kv_lora_dim != 512u || qk_rope != 64u || cache_cap == 0u ||
        selected->bytes < (uint64_t)n_tokens * n_selected * sizeof(uint32_t) ||
        q->bytes < (uint64_t)n_tokens * n_head * qk_dim * sizeof(float) ||
        qk_low->bytes < (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float) ||
        kv_lora_cache->bytes < (uint64_t)cache_cap * kv_lora_dim * cache_elem ||
        k_rope_cache->bytes < (uint64_t)cache_cap * qk_rope * cache_elem ||
        lora_out->bytes < (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float)) {
        fprintf(stderr, "ds4: glm attn lora selected: unsupported config "
                "(n_tok=%u selected=%u head=%u lora=%u rope=%u f16=%d)\n",
                n_tokens, n_selected, n_head, kv_lora_dim, qk_rope,
                (int)cache_f16);
        return 0;
    }
    const float scale = 1.0f / sqrtf((float)(qk_nope + qk_rope));
    dim3 grid((n_head + 7u) / 8u, n_tokens, 1);
    if (cache_f16) {
        glm_attention_lora_causal_kernel<__half, true><<<grid, 256>>>(
                (float *)lora_out->ptr,
                (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const __half *)kv_lora_cache->ptr,
                (const __half *)k_rope_cache->ptr,
                (const uint32_t *)selected->ptr, cache_cap,
                n_tokens, 0u, n_selected, n_head, kv_lora_dim, qk_nope,
                qk_rope, n_ctx_orig, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow, scale);
    } else {
        glm_attention_lora_causal_kernel<float, true><<<grid, 256>>>(
                (float *)lora_out->ptr,
                (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const float *)kv_lora_cache->ptr,
                (const float *)k_rope_cache->ptr,
                (const uint32_t *)selected->ptr, cache_cap,
                n_tokens, 0u, n_selected, n_head, kv_lora_dim, qk_nope,
                qk_rope, n_ctx_orig, freq_base, freq_scale, ext_factor,
                attn_factor, beta_fast, beta_slow, scale);
    }
    return cuda_ok(cudaGetLastError(), "glm attn lora selected launch");
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_attention_indexed_batch_typed_tensor\n");
    return 0;
}

extern "C" int ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor\n");
    return 0;
}

__device__ __forceinline__ static float glm_q8_0_dot_row_dev(
        const char *row, const float *x, uint32_t n_cols) {
    float acc = 0.0f;
    const uint32_t nb = n_cols >> 5;
    for (uint32_t b = 0; b < nb; b++) {
        const char *blk = row + (uint64_t)b * 34u;
        const float d = __half2float(*(const __half *)blk);
        const int8_t *q = (const int8_t *)(blk + 2);
        float s = 0.0f;
        #pragma unroll 8
        for (uint32_t k = 0; k < 32u; k++) s += (float)q[k] * x[b * 32u + k];
        acc += d * s;
    }
    return acc;
}

template <typename CT>
__device__ __forceinline__ static float2 glm_cache_value_pair_dev(
        const CT *p) {
    return make_float2((float)p[0], (float)p[1]);
}

template <>
__device__ __forceinline__ float2 glm_cache_value_pair_dev<__half>(
        const __half *p) {
    return __half22float2(*(const __half2 *)p);
}

template <>
__device__ __forceinline__ float2 glm_cache_value_pair_dev<float>(
        const float *p) {
    return *(const float2 *)p;
}

/* Exact staged decode attention. The original fused kernel owns one block per
 * head, which leaves more than half of an L40S idle. These stages preserve the
 * fused kernel's arithmetic order for every score, softmax lane, lora output,
 * and value-projection row while exposing independent rows/dimensions as
 * separate blocks. */
template <typename CT, bool RANGE_TOK2 = false>
__global__ static void glm_attention_decode_weights_staged_kernel(
        float *weights,
        float *denom,
        const float *q,
        const float *qk_low,
        const CT *kv_lora_cache,
        const CT *k_rope_cache,
        const uint32_t *selected,
        uint32_t n_selected,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        float scale,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        bool score_vec2) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = RANGE_TOK2 ? blockIdx.y : 0u;
    const uint32_t row_count = n_selected + (RANGE_TOK2 ? token : 0u);
    const uint32_t score_stride = n_selected + (RANGE_TOK2 ? 1u : 0u);
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    if (head >= n_head || row_count == 0u) return;
    const uint32_t qk_dim = qk_nope + qk_rope;
    extern __shared__ float glm_dec_stage_sh[];
    float *red = glm_dec_stage_sh;
    float *scores = glm_dec_stage_sh + 256u;
    const float *qh = q +
        ((uint64_t)token * n_head + head) * qk_dim;
    const float *low = qk_low +
        ((uint64_t)token * n_head + head) * kv_lora_dim;

    float corr_dims[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f) {
        corr_dims[0] = fmaxf(0.0f,
            floorf(glm_rope_yarn_corr_factor_dev((int)qk_rope,
                    (int)n_ctx_orig, beta_fast, freq_base)));
        corr_dims[1] = fminf((float)qk_rope - 1.0f,
            ceilf(glm_rope_yarn_corr_factor_dev((int)qk_rope,
                    (int)n_ctx_orig, beta_slow, freq_base)));
    }

    float local_max = -FLT_MAX;
    for (uint32_t s = tid; s < row_count; s += nth) {
        const uint32_t row = RANGE_TOK2 ? s : selected[s];
        float score = -FLT_MAX;
        if (row < cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * kv_lora_dim;
            if (score_vec2) {
                for (uint32_t j = 0; j < kv_lora_dim; j += 2u) {
                    const float2 x = *(const float2 *)(low + j);
                    const float2 y = glm_cache_value_pair_dev(
                            kv_lora_cache + lora_base + j);
                    dotv += x.x * y.x;
                    dotv += x.y * y.y;
                }
            } else {
                for (uint32_t j = 0; j < kv_lora_dim; j++) {
                    dotv += low[j] * (float)kv_lora_cache[lora_base + j];
                }
            }
            const uint64_t rope_base = (uint64_t)row * qk_rope;
            for (uint32_t r = 0; r < qk_rope; r += 2u) {
                const float2 y = glm_cache_rope_pair_f16_dev<CT>(
                        k_rope_cache, rope_base, r, row, qk_rope,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        corr_dims);
                dotv += qh[qk_nope + r] * y.x +
                        qh[qk_nope + r + 1u] * y.y;
            }
            score = dotv * scale;
        }
        scores[s] = score;
        local_max = fmaxf(local_max, score);
    }
    red[tid] = local_max;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = fmaxf(red[tid], red[tid + step]);
        __syncthreads();
    }
    const float max_score = red[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t s = tid; s < row_count; s += nth) {
        const float w = expf(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }
    const uint64_t head_index = (uint64_t)token * n_head + head;
    if (tid == 0u) denom[head_index] = fmaxf(red[0], 1.0e-20f);
    float *head_weights = weights + head_index * score_stride;
    for (uint32_t s = tid; s < row_count; s += nth) {
        head_weights[s] = scores[s];
    }
}

template <typename CT, bool RANGE_TOK2 = false>
__global__ static void glm_attention_decode_lora_staged_kernel(
        float *lora_sum,
        const float *scores,
        const float *denom,
        const CT *kv_lora_cache,
        const uint32_t *selected,
        uint32_t n_selected,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t kv_lora_dim) {
    const uint32_t head = blockIdx.y;
    const uint32_t token = RANGE_TOK2 ? blockIdx.z : 0u;
    const uint32_t row_count = n_selected + (RANGE_TOK2 ? token : 0u);
    const uint32_t score_stride = n_selected + (RANGE_TOK2 ? 1u : 0u);
    const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t j = pair * 2u;
    if (head >= n_head || j >= kv_lora_dim) return;
    const uint64_t head_index = (uint64_t)token * n_head + head;
    const float *head_scores = scores + head_index * score_stride;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint32_t s = 0; s < row_count; s++) {
        const uint32_t row = RANGE_TOK2 ? s : selected[s];
        if (row < cache_cap) {
            const float2 v = glm_cache_value_pair_dev(
                    kv_lora_cache + (uint64_t)row * kv_lora_dim + j);
            const float w = head_scores[s];
            acc0 += w * v.x;
            acc1 += w * v.y;
        }
    }
    float *out = lora_sum + head_index * kv_lora_dim + j;
    out[0] = acc0 / denom[head_index];
    out[1] = acc1 / denom[head_index];
}

template <bool TOK2 = false>
__global__ static void glm_attention_decode_value_staged_kernel(
        float *heads,
        const float *lora_sum,
        const char *value_weight,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t value_dim,
        uint32_t value_row_bytes) {
    const uint32_t head = blockIdx.y;
    const uint32_t token = TOK2 ? blockIdx.z : 0u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t nwarps = blockDim.x >> 5;
    const uint32_t out_warp = blockIdx.x * nwarps + warp;
    const uint32_t total_warps = gridDim.x * nwarps;
    if (head >= n_head) return;
    const float *low = lora_sum +
        ((uint64_t)token * n_head + head) * kv_lora_dim;
    float *out = heads +
        ((uint64_t)token * n_head + head) * value_dim;
    const uint32_t nblk = kv_lora_dim >> 5;
    for (uint32_t d = out_warp; d < value_dim; d += total_warps) {
        const char *row = value_weight +
            ((uint64_t)head * value_dim + d) * value_row_bytes;
        float acc = 0.0f;
        for (uint32_t blk = lane >> 1; blk < nblk; blk += 16u) {
            const char *b = row + (uint64_t)blk * 34u;
            const float dscale = __half2float(*(const __half *)b);
            const int8_t *q = (const int8_t *)(b + 2) + (lane & 1u) * 16u;
            const float *xs = low + blk * 32u + (lane & 1u) * 16u;
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < 16; k++) s += (float)q[k] * xs[k];
            acc += dscale * s;
        }
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0u) out[d] = acc;
    }
}

/* Single-token indexed MLA decode attention, one block per head.
 * Fuses score (qk_low . kv_lora + q_rope . rope(k_rope@row)), softmax over
 * the indexer-selected rows, the weighted kv_lora sum, and the per-head
 * value projection (q8_0). Dynamic shared: red[256] + scores[n_selected] +
 * lora_sum[kv_lora_dim]. */
template <typename CT, bool RANGE_TOK2 = false>
__global__ static void glm_attention_indexed_decode_kernel(
        float *heads,
        const float *q,
        const float *qk_low,
        const CT *kv_lora_cache,
        const CT *k_rope_cache,
        const char *value_weight,
        const uint32_t *selected,
        uint32_t n_selected,
        uint32_t cache_cap,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_rope,
        uint32_t value_dim,
        uint32_t value_row_bytes,
        bool lora_vec2,
        bool score_vec2,
        float scale,
        uint32_t n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = RANGE_TOK2 ? blockIdx.y : 0u;
    const uint32_t row_count = n_selected + (RANGE_TOK2 ? token : 0u);
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    if (head >= n_head || row_count == 0u) return;
    const uint32_t qk_dim = qk_nope + qk_rope;

    extern __shared__ float glm_dec_sh[];
    float *red = glm_dec_sh;
    float *scores = glm_dec_sh + 256u;
    float *lora_sum = scores + row_count;

    const float *qh = q +
        ((uint64_t)token * n_head + head) * qk_dim;
    const float *low = qk_low +
        ((uint64_t)token * n_head + head) * kv_lora_dim;

    float corr_dims[2] = {0.0f, 0.0f};
    if (ext_factor != 0.0f) {
        corr_dims[0] = fmaxf(0.0f,
            floorf(glm_rope_yarn_corr_factor_dev((int)qk_rope,
                    (int)n_ctx_orig, beta_fast, freq_base)));
        corr_dims[1] = fminf((float)qk_rope - 1.0f,
            ceilf(glm_rope_yarn_corr_factor_dev((int)qk_rope,
                    (int)n_ctx_orig, beta_slow, freq_base)));
    }

    float local_max = -FLT_MAX;
    for (uint32_t s = tid; s < row_count; s += nth) {
        const uint32_t row = RANGE_TOK2 ? s : selected[s];
        float score = -FLT_MAX;
        if (row < cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * kv_lora_dim;
            if (score_vec2) {
                for (uint32_t j = 0; j < kv_lora_dim; j += 2u) {
                    const float2 x = *(const float2 *)(low + j);
                    const float2 y = glm_cache_value_pair_dev(
                            kv_lora_cache + lora_base + j);
                    dotv += x.x * y.x;
                    dotv += x.y * y.y;
                }
            } else {
                for (uint32_t j = 0; j < kv_lora_dim; j++) {
                    dotv += low[j] * (float)kv_lora_cache[lora_base + j];
                }
            }
            const uint64_t rope_base = (uint64_t)row * qk_rope;
            for (uint32_t r = 0; r < qk_rope; r += 2u) {
                const float2 y = glm_cache_rope_pair_f16_dev<CT>(
                        k_rope_cache, rope_base, r, row, qk_rope,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        corr_dims);
                dotv += qh[qk_nope + r] * y.x + qh[qk_nope + r + 1u] * y.y;
            }
            score = dotv * scale;
        }
        scores[s] = score;
        local_max = fmaxf(local_max, score);
    }
    red[tid] = local_max;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = fmaxf(red[tid], red[tid + step]);
        __syncthreads();
    }
    const float max_score = red[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t s = tid; s < row_count; s += nth) {
        const float w = expf(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }
    const float denom = fmaxf(red[0], 1.0e-20f);
    __syncthreads();

    if (lora_vec2) {
        for (uint32_t j = tid * 2u; j < kv_lora_dim; j += nth * 2u) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint32_t s = 0; s < row_count; s++) {
                const uint32_t row = RANGE_TOK2 ? s : selected[s];
                if (row < cache_cap) {
                    const float2 v = glm_cache_value_pair_dev(
                            kv_lora_cache + (uint64_t)row * kv_lora_dim + j);
                    const float w = scores[s];
                    acc0 += w * v.x;
                    acc1 += w * v.y;
                }
            }
            lora_sum[j] = acc0 / denom;
            lora_sum[j + 1u] = acc1 / denom;
        }
    } else {
        for (uint32_t j = tid; j < kv_lora_dim; j += nth) {
            float acc = 0.0f;
            for (uint32_t s = 0; s < row_count; s++) {
                const uint32_t row = RANGE_TOK2 ? s : selected[s];
                if (row < cache_cap) {
                    acc += scores[s] *
                           (float)kv_lora_cache[(uint64_t)row * kv_lora_dim + j];
                }
            }
            lora_sum[j] = acc / denom;
        }
    }
    __syncthreads();

    float *out = heads +
        ((uint64_t)token * n_head + head) * value_dim;
    /* Warp-cooperative value projection: one warp per output dim, two
     * lanes per q8_0 block (16 cols each). */
    const uint32_t nwarps = nth >> 5;
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t nblk = kv_lora_dim >> 5;
    for (uint32_t d = warp; d < value_dim; d += nwarps) {
        const char *row = value_weight +
            ((uint64_t)head * value_dim + d) * value_row_bytes;
        float acc = 0.0f;
        for (uint32_t blk = lane >> 1; blk < nblk; blk += 16u) {
            const char *b = row + (uint64_t)blk * 34u;
            const float dscale = __half2float(*(const __half *)b);
            const int8_t *q = (const int8_t *)(b + 2) + (lane & 1u) * 16u;
            const float *xs = lora_sum + blk * 32u + (lane & 1u) * 16u;
            float s = 0.0f;
            #pragma unroll
            for (int k = 0; k < 16; k++) s += (float)q[k] * xs[k];
            acc += dscale * s;
        }
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0u) out[d] = acc;
    }
}

extern "C" int ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    const uint32_t qk_dim = qk_nope + qk_rope;
    if (!heads || !q || !qk_low || !kv_lora_cache || !k_rope_cache ||
        !model_map || !selected ||
        n_selected == 0 || cache_cap == 0 || n_selected > cache_cap ||
        n_head == 0 || kv_lora_dim == 0 || (kv_lora_dim & 31u) != 0u ||
        qk_nope == 0 || qk_rope == 0 || (qk_rope & 1u) != 0u ||
        value_dim == 0) {
        return 0;
    }
    if (value_weight_type != 8u) {   /* DS4_TENSOR_Q8_0 */
        fprintf(stderr,
            "ds4: glm indexed decode attention: unsupported value type %u\n",
            value_weight_type);
        return 0;
    }
    const uint64_t value_row_bytes = ((uint64_t)kv_lora_dim / 32u) * 34u;
    const uint64_t value_weight_bytes =
        (uint64_t)n_head * value_dim * value_row_bytes;
    if (value_weight_offset > model_size ||
        value_weight_bytes > model_size - value_weight_offset) {
        return 0;
    }
    const uint64_t cache_elem = cache_f16 ? 2u : 4u;
    if (heads->bytes < (uint64_t)n_head * value_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * qk_dim * sizeof(float) ||
        qk_low->bytes < (uint64_t)n_head * kv_lora_dim * sizeof(float) ||
        kv_lora_cache->bytes < (uint64_t)cache_cap * kv_lora_dim * cache_elem ||
        k_rope_cache->bytes < (uint64_t)cache_cap * qk_rope * cache_elem ||
        selected->bytes < (uint64_t)n_selected * sizeof(uint32_t)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const char *vw = cuda_resolve_weight_ptr(model_map, value_weight_offset,
            value_weight_bytes, logical_tier, "glm_v_b_decode");
    if (!vw) return 0;
    const float scale = 1.0f / sqrtf((float)qk_dim);
    const bool score_vec2 = getenv("DS4_GLM_ATTN_NO_SCORE_VEC2") == NULL;
    const bool range_tok2 =
        g_glm_mtp_verify_mode &&
        getenv("DS4_GLM_MTP_NO_ATTN_TOK2") == NULL &&
        n_selected < cache_cap &&
        heads->bytes >= 2u * (uint64_t)n_head * value_dim * sizeof(float) &&
        q->bytes >= 2u * (uint64_t)n_head * qk_dim * sizeof(float) &&
        qk_low->bytes >=
            2u * (uint64_t)n_head * kv_lora_dim * sizeof(float);
    if (range_tok2 && n_selected < 512u) {
        const bool lora_vec2 =
            getenv("DS4_GLM_ATTN_NO_LORA_VEC2") == NULL;
        const uint32_t shmem =
            (256u + n_selected + 1u + kv_lora_dim) *
            (uint32_t)sizeof(float);
        const dim3 grid(n_head, 2u, 1u);
        if (cache_f16) {
            glm_attention_indexed_decode_kernel<__half, true>
                    <<<grid, 256, shmem>>>(
                    (float *)heads->ptr, (const float *)q->ptr,
                    (const float *)qk_low->ptr,
                    (const __half *)kv_lora_cache->ptr,
                    (const __half *)k_rope_cache->ptr,
                    vw, (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim,
                    qk_nope, qk_rope, value_dim,
                    (uint32_t)value_row_bytes,
                    lora_vec2, score_vec2, scale, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    beta_fast, beta_slow);
        } else {
            glm_attention_indexed_decode_kernel<float, true>
                    <<<grid, 256, shmem>>>(
                    (float *)heads->ptr, (const float *)q->ptr,
                    (const float *)qk_low->ptr,
                    (const float *)kv_lora_cache->ptr,
                    (const float *)k_rope_cache->ptr,
                    vw, (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim,
                    qk_nope, qk_rope, value_dim,
                    (uint32_t)value_row_bytes,
                    lora_vec2, score_vec2, scale, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor,
                    beta_fast, beta_slow);
        }
        return cuda_ok(cudaGetLastError(),
                       "glm indexed decode attention tok2 range");
    }
    if (n_selected >= 512u &&
        getenv("DS4_GLM_ATTN_NO_STAGED_DECODE") == NULL) {
        const uint32_t token_count = range_tok2 ? 2u : 1u;
        const uint32_t score_stride = n_selected + (range_tok2 ? 1u : 0u);
        const uint64_t head_count = (uint64_t)token_count * n_head;
        if (head_count > UINT64_MAX / score_stride ||
            head_count > UINT64_MAX / kv_lora_dim) {
            return 0;
        }
        const uint64_t score_count = head_count * score_stride;
        const uint64_t lora_count = head_count * kv_lora_dim;
        if (score_count > UINT64_MAX - head_count - lora_count ||
            score_count + head_count + lora_count >
                UINT64_MAX / sizeof(float)) {
            return 0;
        }
        const uint64_t scratch_bytes =
            (score_count + head_count + lora_count) * sizeof(float);
        float *scratch = (float *)cuda_tmp_alloc_on(
                ds4_tensor_device_idx(heads), scratch_bytes,
                "glm staged decode attention");
        if (!scratch) return 0;
        float *softmax_denom = scratch + score_count;
        float *lora_sum = softmax_denom + head_count;
        const uint32_t weight_shmem =
            (256u + score_stride) * (uint32_t)sizeof(float);
        const dim3 weight_grid(n_head, token_count, 1u);
        if (cache_f16 && range_tok2) {
            glm_attention_decode_weights_staged_kernel<__half, true>
                    <<<weight_grid, 256, weight_shmem>>>(
                    scratch, softmax_denom, (const float *)q->ptr,
                    (const float *)qk_low->ptr,
                    (const __half *)kv_lora_cache->ptr,
                    (const __half *)k_rope_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim,
                    qk_nope, qk_rope, scale, n_ctx_orig, freq_base,
                    freq_scale, ext_factor, attn_factor, beta_fast,
                    beta_slow, score_vec2);
        } else if (cache_f16) {
            glm_attention_decode_weights_staged_kernel<__half>
                    <<<weight_grid, 256, weight_shmem>>>(
                    scratch, softmax_denom, (const float *)q->ptr,
                    (const float *)qk_low->ptr,
                    (const __half *)kv_lora_cache->ptr,
                    (const __half *)k_rope_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim,
                    qk_nope, qk_rope, scale, n_ctx_orig, freq_base,
                    freq_scale, ext_factor, attn_factor, beta_fast,
                    beta_slow, score_vec2);
        } else if (range_tok2) {
            glm_attention_decode_weights_staged_kernel<float, true>
                    <<<weight_grid, 256, weight_shmem>>>(
                    scratch, softmax_denom, (const float *)q->ptr,
                    (const float *)qk_low->ptr,
                    (const float *)kv_lora_cache->ptr,
                    (const float *)k_rope_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim,
                    qk_nope, qk_rope, scale, n_ctx_orig, freq_base,
                    freq_scale, ext_factor, attn_factor, beta_fast,
                    beta_slow, score_vec2);
        } else {
            glm_attention_decode_weights_staged_kernel<float>
                    <<<weight_grid, 256, weight_shmem>>>(
                    scratch, softmax_denom, (const float *)q->ptr,
                    (const float *)qk_low->ptr,
                    (const float *)kv_lora_cache->ptr,
                    (const float *)k_rope_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim,
                    qk_nope, qk_rope, scale, n_ctx_orig, freq_base,
                    freq_scale, ext_factor, attn_factor, beta_fast,
                    beta_slow, score_vec2);
        }
        if (!cuda_ok(cudaGetLastError(),
                     "glm staged decode weights launch")) {
            return 0;
        }
        dim3 lora_grid((kv_lora_dim / 2u + 63u) / 64u,
                       n_head, token_count);
        if (cache_f16 && range_tok2) {
            glm_attention_decode_lora_staged_kernel<__half, true>
                    <<<lora_grid, 64>>>(
                    lora_sum, scratch, softmax_denom,
                    (const __half *)kv_lora_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim);
        } else if (cache_f16) {
            glm_attention_decode_lora_staged_kernel<__half>
                    <<<lora_grid, 64>>>(
                    lora_sum, scratch, softmax_denom,
                    (const __half *)kv_lora_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim);
        } else if (range_tok2) {
            glm_attention_decode_lora_staged_kernel<float, true>
                    <<<lora_grid, 64>>>(
                    lora_sum, scratch, softmax_denom,
                    (const float *)kv_lora_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim);
        } else {
            glm_attention_decode_lora_staged_kernel<float>
                    <<<lora_grid, 64>>>(
                    lora_sum, scratch, softmax_denom,
                    (const float *)kv_lora_cache->ptr,
                    (const uint32_t *)selected->ptr,
                    n_selected, cache_cap, n_head, kv_lora_dim);
        }
        if (!cuda_ok(cudaGetLastError(),
                     "glm staged decode lora launch")) {
            return 0;
        }
        dim3 value_grid((value_dim + 127u) / 128u,
                        n_head, token_count);
        if (range_tok2) {
            glm_attention_decode_value_staged_kernel<true>
                    <<<value_grid, 128>>>(
                    (float *)heads->ptr, lora_sum, vw,
                    n_head, kv_lora_dim, value_dim,
                    (uint32_t)value_row_bytes);
        } else {
            glm_attention_decode_value_staged_kernel<false>
                    <<<value_grid, 128>>>(
                    (float *)heads->ptr, lora_sum, vw,
                    n_head, kv_lora_dim, value_dim,
                    (uint32_t)value_row_bytes);
        }
        return cuda_ok(cudaGetLastError(),
                       "glm staged decode value launch");
    }
    const bool lora_vec2 = getenv("DS4_GLM_ATTN_NO_LORA_VEC2") == NULL;
    const uint32_t shmem =
        (256u + n_selected + kv_lora_dim) * (uint32_t)sizeof(float);
    if (cache_f16) {
        glm_attention_indexed_decode_kernel<__half><<<n_head, 256, shmem>>>(
                (float *)heads->ptr, (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const __half *)kv_lora_cache->ptr,
                (const __half *)k_rope_cache->ptr,
                vw, (const uint32_t *)selected->ptr,
                n_selected, cache_cap, n_head, kv_lora_dim,
                qk_nope, qk_rope, value_dim, (uint32_t)value_row_bytes,
                lora_vec2, score_vec2, scale, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    } else {
        glm_attention_indexed_decode_kernel<float><<<n_head, 256, shmem>>>(
                (float *)heads->ptr, (const float *)q->ptr,
                (const float *)qk_low->ptr,
                (const float *)kv_lora_cache->ptr,
                (const float *)k_rope_cache->ptr,
                vw, (const uint32_t *)selected->ptr,
                n_selected, cache_cap, n_head, kv_lora_dim,
                qk_nope, qk_rope, value_dim, (uint32_t)value_row_bytes,
                lora_vec2, score_vec2, scale, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    }
    return cuda_ok(cudaGetLastError(), "glm indexed decode attention");
}

extern "C" int ds4_gpu_glm_build_kv_cache_flash_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_build_kv_cache_flash_tensor\n");
    return 0;
}

extern "C" int ds4_gpu_glm_build_kv_cache_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_build_kv_cache_tensor\n");
    return 0;
}

__global__ static void glm_fill_selected_range_batch_kernel(
        uint32_t *selected,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_selected,
        uint32_t pad_row) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = n_tokens * n_selected;
    if (gid >= total || n_selected == 0u) return;
    const uint32_t token = gid / n_selected;
    const uint32_t slot = gid - token * n_selected;
    const uint32_t visible = pos0 + token + 1u;
    selected[gid] = slot < visible ? slot : pad_row;
}

extern "C" int ds4_gpu_glm_fill_selected_range_batch_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_tokens,
        uint32_t        pos0,
        uint32_t        n_selected,
        uint32_t        pad_row) {
    if (!selected || n_tokens == 0 || n_selected == 0 ||
        selected->bytes < (uint64_t)n_tokens * n_selected * sizeof(uint32_t)) {
        return 0;
    }
    const uint64_t total = (uint64_t)n_tokens * n_selected;
    glm_fill_selected_range_batch_kernel<<<(unsigned)((total + 255) / 256), 256>>>(
            (uint32_t *)selected->ptr, n_tokens, pos0, n_selected, pad_row);
    return cuda_ok(cudaGetLastError(), "glm fill selected batch launch");
}

__global__ static void glm_fill_selected_range_kernel(
        uint32_t *selected, uint32_t n_selected) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n_selected) selected[gid] = gid;
}

extern "C" int ds4_gpu_glm_fill_selected_range_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_selected) {
    if (!selected || n_selected == 0 ||
        selected->bytes < (uint64_t)n_selected * sizeof(uint32_t)) {
        return 0;
    }
    glm_fill_selected_range_kernel<<<(n_selected + 255) / 256, 256>>>(
            (uint32_t *)selected->ptr, n_selected);
    return cuda_ok(cudaGetLastError(), "glm fill selected launch");
}

static int glm_rope_tail_offset_launch(
        ds4_gpu_tensor *x,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim, uint32_t rot_offset, uint32_t pos0,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, const char *what);

extern "C" int ds4_gpu_glm_indexer_rope_tail_tensor(
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
    return glm_rope_tail_offset_launch(x, n_tokens, n_head, head_dim,
            rot_dim, 0, pos0, n_ctx_orig,
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, "glm indexer rope tail");
}

template <typename CT>
__global__ static void glm_indexer_scores_f32_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const CT *indexer_key_cache,
        uint32_t n_rows,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        float scale,
        bool causal) {
    const uint32_t row = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (row >= n_rows || token >= n_tokens || tid >= 128u) return;
    if (causal && row >= min(n_rows, pos0 + token + 1u)) {
        if (tid == 0u) scores[(uint64_t)token * n_rows + row] = -INFINITY;
        return;
    }

    __shared__ float partial[128];
    float total = 0.0f;
    const CT *krow = indexer_key_cache + (uint64_t)row * head_dim;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q +
            ((uint64_t)token * n_head + h) * head_dim;
        float dot = tid < head_dim ? qh[tid] * (float)krow[tid] : 0.0f;
        partial[tid] = dot;
        __syncthreads();
        for (uint32_t stride = 64u; stride > 0u; stride >>= 1u) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0u) {
            total += fmaxf(partial[0], 0.0f) *
                weights[(uint64_t)token * n_head + h];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        scores[(uint64_t)token * n_rows + row] = total * scale;
    }
}

/* 16-token x 128-row indexer tile. Q and cached K are staged as fp16,
 * matching the model's compact-cache precision; each head's MMA result and
 * the weighted head reduction remain fp32. */
template <typename CT>
__global__ static void glm_indexer_scores_wmma128_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const CT *indexer_key_cache,
        uint32_t n_rows,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        float scale,
        bool causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t row0 = blockIdx.x * 128u;
    const uint32_t token0 = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 256u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(token0 + 16u, n_tokens);
        const uint32_t max_visible = last_token > token0
            ? min(pos0 + last_token, n_rows) : 0u;
        if (row0 >= max_visible) {
            for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
                const uint32_t token = token0 + (i >> 7u);
                const uint32_t row = row0 + (i & 127u);
                if (token < n_tokens && row < n_rows) {
                    scores[(uint64_t)token * n_rows + row] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half q_sh[16 * 128];
    __shared__ __half k_sh[128 * 128];
    __shared__ float dot_sh[8 * 16 * 16];
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f};

    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
        const uint32_t r = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t row = row0 + r;
        const float v = row < n_rows
            ? (float)indexer_key_cache[(uint64_t)row * head_dim + d]
            : 0.0f;
        k_sh[d + r * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
            const uint32_t tr = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = token0 + tr;
            const float v = token < n_tokens
                ? q[((uint64_t)token * n_head + h) * head_dim + d]
                : 0.0f;
            q_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16,
                       __half, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16,
                       __half, wmma::col_major> k_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> dot_frag;
        wmma::fill_fragment(dot_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(q_frag, q_sh + k0, 128);
            wmma::load_matrix_sync(k_frag,
                                   k_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(dot_frag, q_frag, k_frag, dot_frag);
        }
        wmma::store_matrix_sync(dot_sh + warp * 16u * 16u,
                                dot_frag, 16, wmma::mem_row_major);
        __syncthreads();

        const uint32_t local0 = tid & 255u;
        const uint32_t token = token0 + (local0 >> 4u);
        const float w = token < n_tokens
            ? weights[(uint64_t)token * n_head + h] : 0.0f;
        uint32_t slot = 0;
        for (uint32_t i = tid; i < 8u * 16u * 16u;
             i += 256u, slot++) {
            const uint32_t row = row0 + (i >> 8u) * 16u + (i & 15u);
            if (token < n_tokens && row < n_rows) {
                acc[slot] += fmaxf(dot_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    uint32_t slot = 0;
    for (uint32_t i = tid; i < 8u * 16u * 16u;
         i += 256u, slot++) {
        const uint32_t local = i & 255u;
        const uint32_t token = token0 + (local >> 4u);
        const uint32_t row = row0 + (i >> 8u) * 16u + (local & 15u);
        if (token < n_tokens && row < n_rows) {
            float out = acc[slot] * scale;
            if (causal && row >= pos0 + token + 1u) out = -INFINITY;
            scores[(uint64_t)token * n_rows + row] = out;
        }
    }
#endif
}

static int glm_indexer_scores_launch(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16,
        bool                  causal) {
    const uint64_t cache_elem = cache_f16 ? sizeof(__half) : sizeof(float);
    if (!scores || !q || !weights || !indexer_key_cache || n_rows == 0u ||
        n_tokens == 0u || n_head == 0u || head_dim != 128u ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        indexer_key_cache->bytes < (uint64_t)n_rows * head_dim * cache_elem ||
        scores->bytes < (uint64_t)n_tokens * n_rows * sizeof(float)) {
        return 0;
    }
    if (!g_quality_mode) {
        dim3 grid((n_rows + 127u) / 128u,
                  (n_tokens + 15u) / 16u, 1);
        if (cache_f16) {
            glm_indexer_scores_wmma128_kernel<__half><<<grid, 256>>>(
                    (float *)scores->ptr, (const float *)q->ptr,
                    (const float *)weights->ptr,
                    (const __half *)indexer_key_cache->ptr,
                    n_rows, n_tokens, pos0, n_head, head_dim, scale, causal);
        } else {
            glm_indexer_scores_wmma128_kernel<float><<<grid, 256>>>(
                    (float *)scores->ptr, (const float *)q->ptr,
                    (const float *)weights->ptr,
                    (const float *)indexer_key_cache->ptr,
                    n_rows, n_tokens, pos0, n_head, head_dim, scale, causal);
        }
        return cuda_ok(cudaGetLastError(), "glm indexer scores wmma launch");
    }

    dim3 grid(n_rows, n_tokens, 1);
    if (cache_f16) {
        glm_indexer_scores_f32_kernel<__half><<<grid, 128>>>(
                (float *)scores->ptr, (const float *)q->ptr,
                (const float *)weights->ptr,
                (const __half *)indexer_key_cache->ptr,
                n_rows, n_tokens, pos0, n_head, head_dim, scale, causal);
    } else {
        glm_indexer_scores_f32_kernel<float><<<grid, 128>>>(
                (float *)scores->ptr, (const float *)q->ptr,
                (const float *)weights->ptr,
                (const float *)indexer_key_cache->ptr,
                n_rows, n_tokens, pos0, n_head, head_dim, scale, causal);
    }
    return cuda_ok(cudaGetLastError(), "glm indexer scores f32 launch");
}

extern "C" int ds4_gpu_glm_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16) {
    return glm_indexer_scores_launch(scores, q, weights, indexer_key_cache,
                                     n_rows, 1u, 0u, n_head, head_dim,
                                     scale, cache_f16, false);
}

extern "C" int ds4_gpu_glm_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16) {
    return glm_indexer_scores_launch(scores, q, weights, indexer_key_cache,
                                     n_rows, n_tokens, pos0, n_head, head_dim,
                                     scale, cache_f16, true);
}

extern "C" int ds4_gpu_glm_k_b_project_typed_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head) {
    fprintf(stderr, "ds4: CUDA stub called: ds4_gpu_glm_k_b_project_typed_tensor\n");
    return 0;
}

__global__ static void glm_kv_lora_rms_norm_kernel(
        float *dst,
        const float *src,
        const float *w,
        uint32_t n_tokens,
        uint32_t kv_raw_dim,
        uint32_t kv_lora_dim,
        float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= n_tokens) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    const float *x = src + (uint64_t)row * kv_raw_dim;
    float *out = dst + (uint64_t)row * kv_lora_dim;
    __shared__ float scratch[256];
    float ss = 0.0f;
    for (uint32_t i = tid; i < kv_lora_dim; i += nth) {
        const float v = x[i];
        ss += v * v;
    }
    scratch[tid] = ss;
    __syncthreads();
    for (uint32_t step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / (float)kv_lora_dim + eps);
    for (uint32_t i = tid; i < kv_lora_dim; i += nth) {
        out[i] = x[i] * inv * w[i];
    }
}

extern "C" int ds4_gpu_glm_kv_lora_rms_norm_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        float                 eps) {
    if (!dst || !src || !model_map || n_tokens == 0 ||
        kv_lora_dim == 0 || kv_lora_dim > kv_raw_dim) {
        return 0;
    }
    const uint64_t wb = (uint64_t)kv_lora_dim * sizeof(float);
    if (weight_offset > model_size || wb > model_size - weight_offset ||
        src->bytes < (uint64_t)n_tokens * kv_raw_dim * sizeof(float) ||
        dst->bytes < (uint64_t)n_tokens * kv_lora_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const float *w = (const float *)cuda_resolve_weight_ptr(
            model_map, weight_offset, wb, logical_tier, "glm_kv_lora_norm");
    if (!w) return 0;
    glm_kv_lora_rms_norm_kernel<<<n_tokens, 256>>>(
            (float *)dst->ptr, (const float *)src->ptr, w,
            n_tokens, kv_raw_dim, kv_lora_dim, eps);
    return cuda_ok(cudaGetLastError(), "glm kv lora rms norm launch");
}



__global__ static void glm_qk_lowrank_q8_0_batch_kernel(
        float *qk_low,
        const char *weight,
        const float *q,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t kv_lora_dim,
        uint32_t qk_nope,
        uint32_t qk_dim,
        uint64_t row_bytes) {
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (head >= n_head || token >= n_tokens) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t nth = blockDim.x;
    const float *qh = q + (uint64_t)token * n_head * qk_dim +
                      (uint64_t)head * qk_dim;
    float *out = qk_low + (uint64_t)token * n_head * kv_lora_dim +
                 (uint64_t)head * kv_lora_dim;
    for (uint32_t j = tid; j < kv_lora_dim; j += nth) {
        const char *row = weight +
            ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        out[j] = glm_q8_0_dot_row_dev(row, qh, qk_nope);
    }
}

extern "C" int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    if (!qk_low || !q || !model_map || n_tokens == 0 || n_head == 0 ||
        kv_lora_dim == 0 || qk_nope == 0 || (qk_nope & 31u) != 0u) {
        return 0;
    }
    if (weight_type != 8u) {
        fprintf(stderr, "ds4: glm qk_lowrank: unsupported type %u\n",
                weight_type);
        return 0;
    }
    const uint64_t row_bytes = ((uint64_t)qk_nope / 32u) * 34u;
    const uint64_t wbytes = (uint64_t)n_head * kv_lora_dim * row_bytes;
    if (weight_offset > model_size || wbytes > model_size - weight_offset ||
        q->bytes < (uint64_t)n_tokens * n_head * qk_dim * sizeof(float) ||
        qk_low->bytes <
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = cuda_current_tier();
    const char *w = (const char *)cuda_resolve_weight_ptr(
            model_map, weight_offset, wbytes, logical_tier, "glm_k_b_qk");
    if (!w) return 0;
    if (g_q8_dequant_gemm_enabled && g_cublas_ready && n_tokens >= 128u) {
        /* Per-head strided-batched GEMM over a dequantized k_b: the
         * per-(token,head) warp kernel was ~65ms/layer at 820 tokens.
         * Scratch (executing device): [w_f16][q_f16][out_f32]. */
        const uint64_t wh_bytes =
            (uint64_t)n_head * kv_lora_dim * qk_nope * sizeof(__half);
        const uint64_t xh_off = (wh_bytes + 255u) & ~255ull;
        const uint64_t xh_bytes =
            (uint64_t)n_tokens * n_head * qk_dim * sizeof(__half);
        const uint64_t oo_off = (xh_off + xh_bytes + 255u) & ~255ull;
        const uint64_t oo_bytes =
            (uint64_t)n_tokens * n_head * kv_lora_dim * sizeof(float);
        void *tmp = cuda_tmp_alloc_on(logical_tier, oo_off + oo_bytes,
                                      "glm qk_low gemm");
        if (tmp) {
            __half *wh = (__half *)tmp;
            __half *xh = (__half *)((char *)tmp + xh_off);
            float *oo = (float *)((char *)tmp + oo_off);
            const uint64_t total_blocks =
                (uint64_t)n_head * kv_lora_dim * (qk_nope / 32u);
            q8_0_dequant_f16_kernel<<<(unsigned)((total_blocks * 2u + 255u) / 256u), 256>>>(
                    wh, (const unsigned char *)w, total_blocks,
                    qk_nope / 32u, qk_nope);
            const uint64_t xn = (uint64_t)n_tokens * n_head * qk_dim;
            f32_to_f16_kernel<<<(xn + 255u) / 256u, 256>>>(
                    xh, (const float *)q->ptr, xn);
            if (cuda_ok(cudaGetLastError(), "glm qk_low gemm staging")) {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                cublasStatus_t st = cublasGemmStridedBatchedEx(
                        cuda_cublas_for_tier(logical_tier),
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        (int)kv_lora_dim, (int)n_tokens, (int)qk_nope,
                        &alpha,
                        wh, CUDA_R_16F, (int)qk_nope,
                        (long long)((uint64_t)kv_lora_dim * qk_nope),
                        xh, CUDA_R_16F, (int)(n_head * qk_dim),
                        (long long)qk_dim,
                        &beta,
                        oo, CUDA_R_32F, (int)(n_head * kv_lora_dim),
                        (long long)kv_lora_dim,
                        (int)n_head,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
                if (st == CUBLAS_STATUS_SUCCESS &&
                    cuda_ok(cudaMemcpyAsync(qk_low->ptr, oo, oo_bytes,
                            cudaMemcpyDeviceToDevice, 0),
                            "glm qk_low gemm out copy")) {
                    return 1;
                }
                fprintf(stderr,
                        "ds4: glm qk_low gemm failed (status %d); native path\n",
                        (int)st);
            }
        }
    }
    dim3 grid(n_head, n_tokens, 1);
    glm_qk_lowrank_q8_0_batch_kernel<<<grid, 128>>>(
            (float *)qk_low->ptr, w, (const float *)q->ptr,
            n_tokens, n_head, kv_lora_dim, qk_nope, qk_dim, row_bytes);
    return cuda_ok(cudaGetLastError(), "glm qk lowrank batch launch");
}

extern "C" int ds4_gpu_glm_qk_lowrank_typed_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim) {
    return ds4_gpu_glm_qk_lowrank_typed_batch_tensor(qk_low, q, model_map,
                                                     model_size,
                                                     weight_offset,
                                                     weight_type, 1u, n_head,
                                                     kv_lora_dim, qk_nope,
                                                     qk_dim);
}

