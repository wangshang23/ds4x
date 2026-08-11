/* Non-causal batch attention over a raw KV ring for the DSpark draft block.
 * Every query row attends over all n_raw visible rows plus the per-head sink,
 * with the same exact one-block max/denominator/value accumulation order as
 * the reference decode attention (scores in shared, sequential value pass). */
__global__ static void attention_noncausal_raw_batch_heads_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_head,
        uint32_t head_dim) {
    const uint32_t tok = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (tok >= n_tokens || h >= n_head) return;
    extern __shared__ float sh_scores[]; /* n_raw floats */
    const float *qh = q + ((uint64_t)tok * n_head + h) * head_dim;
    const float scale = rsqrtf((float)head_dim);
    for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
        const uint32_t row = (raw_start + r) % raw_cap;
        const float *kv = raw_kv + (uint64_t)row * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        sh_scores[r] = dot * scale;
    }
    __syncthreads();
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
        local_max = fmaxf(local_max, sh_scores[r]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t r = threadIdx.x; r < n_raw; r += blockDim.x) {
        sh_scores[r] = expf(sh_scores[r] - max_s);
        den_local += sh_scores[r];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)tok * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = (raw_start + r) % raw_cap;
            acc += raw_kv[(uint64_t)row * head_dim + d] * sh_scores[r];
        }
        oh[d] = acc / denom;
    }
}

extern "C" int ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !model_map ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || n_head == 0 || head_dim == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), logical_tier,
            "dspark_attn_sinks");
    if (!sinks) return 0;
    const size_t shmem = (size_t)n_raw * sizeof(float);
    if (shmem > 32768) return 0; /* draft blocks are tiny; guard anyway */
    dim3 grid(n_tokens, n_head, 1);
    attention_noncausal_raw_batch_heads_kernel<<<grid, 256, shmem>>>(
            (float *)heads->ptr,
            sinks,
            (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            n_tokens, n_raw, raw_cap, raw_start, n_head, head_dim);
    if (!cuda_ok(cudaGetLastError(), "attention noncausal raw batch heads launch")) return 0;
    static int verify_left = -1;
    if (verify_left < 0) {
        verify_left = getenv("DS4_DSPARK_VERIFY_NONCAUSAL") != NULL ? 3 : 0;
    }
    if (verify_left > 0) {
        verify_left--;
        (void)cudaDeviceSynchronize();
        const uint64_t qn = (uint64_t)n_tokens * n_head * head_dim;
        const uint64_t kn = (uint64_t)raw_cap * head_dim;
        std::vector<float> hq(qn), hkv(kn), hout(qn), hsink(n_head);
        (void)cudaMemcpy(hq.data(), q->ptr, qn * 4, cudaMemcpyDeviceToHost);
        (void)cudaMemcpy(hkv.data(), raw_kv->ptr, kn * 4, cudaMemcpyDeviceToHost);
        (void)cudaMemcpy(hout.data(), heads->ptr, qn * 4, cudaMemcpyDeviceToHost);
        (void)cudaMemcpy(hsink.data(), sinks, (uint64_t)n_head * 4, cudaMemcpyDeviceToHost);
        double max_abs = 0.0, max_rel = 0.0;
        const double scale = 1.0 / sqrt((double)head_dim);
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t h = 0; h < n_head; h++) {
                std::vector<double> sc(n_raw);
                double mx = (double)hsink[h];
                for (uint32_t r = 0; r < n_raw; r++) {
                    const uint32_t row = (raw_start + r) % raw_cap;
                    double dot = 0.0;
                    for (uint32_t d = 0; d < head_dim; d++) {
                        dot += (double)hq[((uint64_t)t * n_head + h) * head_dim + d] *
                               (double)hkv[(uint64_t)row * head_dim + d];
                    }
                    sc[r] = dot * scale;
                    if (sc[r] > mx) mx = sc[r];
                }
                double den = exp((double)hsink[h] - mx);
                for (uint32_t r = 0; r < n_raw; r++) den += exp(sc[r] - mx);
                for (uint32_t d = 0; d < head_dim; d++) {
                    double acc = 0.0;
                    for (uint32_t r = 0; r < n_raw; r++) {
                        const uint32_t row = (raw_start + r) % raw_cap;
                        acc += exp(sc[r] - mx) * (double)hkv[(uint64_t)row * head_dim + d];
                    }
                    const double ref = acc / den;
                    const double got = (double)hout[((uint64_t)t * n_head + h) * head_dim + d];
                    const double ad = fabs(ref - got);
                    if (ad > max_abs) max_abs = ad;
                    if (fabs(ref) > 1e-3 && ad / fabs(ref) > max_rel) max_rel = ad / fabs(ref);
                }
            }
        }
        fprintf(stderr,
                "ds4: DSpark noncausal verify n_tok=%u n_raw=%u start=%u cap=%u "
                "max_abs=%.3e max_rel=%.3e\n",
                n_tokens, n_raw, raw_start, raw_cap, max_abs, max_rel);
    }
    return 1;
}

extern "C" int ds4_gpu_repeat_hc_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *rows, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    uint64_t rows_elems = 0;
    uint64_t out_elems = 0;
    if (!out || !rows || n_tokens == 0 || n_embd == 0 || n_hc == 0 ||
        (uint64_t)n_tokens > UINT64_MAX / n_embd ||
        (rows_elems = (uint64_t)n_tokens * n_embd) > UINT64_MAX / n_hc ||
        (out_elems = rows_elems * n_hc) > UINT64_MAX / sizeof(float) ||
        rows_elems > UINT64_MAX / sizeof(float) ||
        rows->bytes < rows_elems * sizeof(float) ||
        out->bytes < out_elems * sizeof(float)) {
        return 0;
    }
    const uint64_t blocks = (out_elems + 255u) / 256u;
    if (blocks > UINT32_MAX) return 0;
    repeat_hc_rows_kernel<<<(unsigned)blocks, 256>>>((float *)out->ptr, (const float *)rows->ptr, n_tokens, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "repeat_hc_rows launch");
}

extern "C" int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    if (n == 4096u) {
        rms_norm_plain_fast4096_kernel<<<1, 256, 0, cuda_decode_stream()>>>((float *)out->ptr, (const float *)x->ptr, n, 1, eps);
    } else if ((n & 2047u) == 0u) {
        rms_norm_plain_batch8_kernel<<<1, 256, 0, cuda_decode_stream()>>>((float *)out->ptr, (const float *)x->ptr, n, 1, eps);
    } else {
        rms_norm_plain_kernel<<<1, 256, 0, cuda_decode_stream()>>>((float *)out->ptr, (const float *)x->ptr, n, 1, eps);
    }
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    if (n == 4096u) {
        rms_norm_plain_fast4096_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, n, rows, eps);
    } else if ((n & 2047u) == 0u) {
        rms_norm_plain_batch8_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, n, rows, eps);
    } else {
        rms_norm_plain_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, n, rows, eps);
    }
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), logical_tier, "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<1, 256, 0, cuda_decode_stream()>>>((float *)out->ptr, (const float *)x->ptr, w, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), logical_tier, "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<rows, 256>>>((float *)out->ptr, (const float *)x->ptr, w, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    if (!g_cuda_disable_qkv_rms_fused) {
        if (!q_out || !q || !kv_out || !kv || !model_map ||
            q_weight_offset > model_size ||
            kv_weight_offset > model_size ||
            model_size - q_weight_offset < (uint64_t)q_n * sizeof(float) ||
            model_size - kv_weight_offset < (uint64_t)kv_n * sizeof(float) ||
            q_out->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            q->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            kv_out->bytes < (uint64_t)kv_n * rows * sizeof(float) ||
            kv->bytes < (uint64_t)kv_n * rows * sizeof(float)) {
            return 0;
        }
        const int logical_tier = ds4_tensor_device_idx(q_out);
        const float *q_w = (const float *)cuda_resolve_weight_ptr(model_map,
                q_weight_offset, (uint64_t)q_n * sizeof(float), logical_tier, "q_rms_weight");
        const float *kv_w = (const float *)cuda_resolve_weight_ptr(model_map,
                kv_weight_offset, (uint64_t)kv_n * sizeof(float), logical_tier, "kv_rms_weight");
        if (!q_w || !kv_w) return 0;
        dim3 grid(rows, 2u, 1u);
        dsv4_qkv_rms_norm_rows_kernel<<<grid, 256>>>(
                (float *)q_out->ptr,
                (const float *)q->ptr,
                q_w,
                q_n,
                (float *)kv_out->ptr,
                (const float *)kv->ptr,
                kv_w,
                kv_n,
                rows,
                eps);
        return cuda_ok(cudaGetLastError(), "dsv4 qkv rms norm rows launch");
    }
    return ds4_gpu_rms_norm_weight_rows_tensor(q_out, q, model_map, model_size,
                                                 q_weight_offset, q_n, rows, eps) &&
           ds4_gpu_rms_norm_weight_rows_tensor(kv_out, kv, model_map, model_size,
                                                 kv_weight_offset, kv_n, rows, eps);
}

extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        uint32_t                kv_n_head,
        uint32_t                kv_head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        bool                    inverse,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   eps) {
    if (g_cuda_disable_qkv_rms_fused) return 0;
    if (!q_out || !q || !kv_out || !kv || !model_map ||
        q_weight_offset > model_size ||
        kv_weight_offset > model_size ||
        kv_n_head == 0 || kv_head_dim == 0 ||
        n_rot > kv_head_dim || (n_rot & 1u) ||
        kv_n != kv_n_head * kv_head_dim ||
        model_size - q_weight_offset < (uint64_t)q_n * sizeof(float) ||
        model_size - kv_weight_offset < (uint64_t)kv_n * sizeof(float) ||
        q_out->bytes < (uint64_t)q_n * rows * sizeof(float) ||
        q->bytes < (uint64_t)q_n * rows * sizeof(float) ||
        kv_out->bytes < (uint64_t)kv_n * rows * sizeof(float) ||
        kv->bytes < (uint64_t)kv_n * rows * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(q_out);
    const float *q_w = (const float *)cuda_resolve_weight_ptr(model_map,
            q_weight_offset, (uint64_t)q_n * sizeof(float), logical_tier, "q_rms_weight");
    const float *kv_w = (const float *)cuda_resolve_weight_ptr(model_map,
            kv_weight_offset, (uint64_t)kv_n * sizeof(float), logical_tier, "kv_rms_weight");
    if (!q_w || !kv_w) return 0;
    dim3 grid(rows, 2u, 1u);
    dsv4_qkv_rms_norm_rows_kv_rope_kernel<<<grid, 256>>>(
            (float *)q_out->ptr,
            (const float *)q->ptr,
            q_w,
            q_n,
            (float *)kv_out->ptr,
            (const float *)kv->ptr,
            kv_w,
            kv_n,
            rows,
            kv_n_head,
            kv_head_dim,
            n_rot,
            pos0,
            n_ctx_orig,
            inverse ? 1 : 0,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            eps);
    return cuda_ok(cudaGetLastError(), "dsv4 qkv rms norm kv rope launch");
}

extern "C" int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    if (!x || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm launch");
}
extern "C" int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps) {
    if (!x || n_rot > head_dim || (n_rot & 1u) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_rope_tail_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm_rope_tail launch");
}
extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_rot > head_dim || x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    fp8_kv_quantize_kernel<<<n_tok, 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}
extern "C" int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim) {
    if (!x || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    indexer_hadamard_fp4_kernel<<<n_rows, 128>>>((float *)x->ptr, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "indexer_hadamard_fp4 launch");
}
extern "C" int ds4_gpu_spark_pack_kv_rows_tensor(
        ds4_gpu_tensor *dst, uint64_t dst_row,
        const ds4_gpu_tensor *src, uint32_t src_row, uint32_t rows) {
    if (!dst || !src || rows == 0u || dst->device_id != src->device_id ||
        dst->bytes < (dst_row + rows) * DS4_SPARK_KV_ROW_BYTES ||
        src->bytes < (uint64_t)(src_row + rows) * 512u * sizeof(float)) {
        return 0;
    }
    spark_pack_kv_rows_kernel<<<rows, 64>>>(
            (unsigned char *)dst->ptr, dst_row,
            (float *)src->ptr, src_row, rows, false);
    return cuda_ok(cudaGetLastError(), "pack Spark KV rows launch");
}
extern "C" int ds4_gpu_spark_pack_index_rows_tensor(
        ds4_gpu_tensor *dst, uint64_t dst_row,
        const ds4_gpu_tensor *src, uint32_t src_row, uint32_t rows) {
    if (!dst || !src || rows == 0u || dst->device_id != src->device_id ||
        dst->bytes < (dst_row + rows) * DS4_SPARK_INDEX_ROW_BYTES ||
        src->bytes < (uint64_t)(src_row + rows) * 128u * sizeof(float)) {
        return 0;
    }
    spark_pack_index_rows_kernel<<<rows, 32>>>(
            (unsigned char *)dst->ptr, dst_row,
            (const float *)src->ptr, src_row, rows);
    return cuda_ok(cudaGetLastError(), "pack Spark index rows launch");
}
extern "C" int ds4_gpu_spark_zero_kv_rows_tensor(
        ds4_gpu_tensor *dst, uint32_t rows) {
    if (!dst || rows == 0u ||
        dst->bytes < (uint64_t)rows * DS4_SPARK_KV_ROW_BYTES) return 0;
    spark_zero_kv_rows_kernel<<<rows, 128>>>((unsigned char *)dst->ptr, rows);
    return cuda_ok(cudaGetLastError(), "zero Spark KV rows launch");
}
extern "C" int ds4_gpu_spark_zero_index_rows_tensor(
        ds4_gpu_tensor *dst, uint32_t rows) {
    if (!dst || rows == 0u ||
        dst->bytes < (uint64_t)rows * DS4_SPARK_INDEX_ROW_BYTES) return 0;
    spark_zero_index_rows_kernel<<<rows, 64>>>((unsigned char *)dst->ptr, rows);
    return cuda_ok(cudaGetLastError(), "zero Spark index rows launch");
}
extern "C" int ds4_gpu_spark_unpack_kv_rows_tensor(
        ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint32_t rows) {
    if (!dst || !src || rows == 0u || dst->device_id != src->device_id ||
        dst->bytes < (uint64_t)rows * 512u * sizeof(float) ||
        src->bytes < (uint64_t)rows * DS4_SPARK_KV_ROW_BYTES) return 0;
    spark_unpack_kv_rows_kernel<<<rows, 256>>>(
        (float *)dst->ptr, (const unsigned char *)src->ptr,
        NULL, rows, rows, 0u, 0u);
    return cuda_ok(cudaGetLastError(), "unpack Spark KV rows launch");
}
extern "C" int ds4_gpu_spark_unpack_index_rows_tensor(
        ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint32_t rows) {
    if (!dst || !src || rows == 0u || dst->device_id != src->device_id ||
        dst->bytes < (uint64_t)rows * 128u * sizeof(float) ||
        src->bytes < (uint64_t)rows * DS4_SPARK_INDEX_ROW_BYTES) return 0;
    spark_unpack_index_rows_kernel<<<rows, 128>>>(
        (float *)dst->ptr, (const unsigned char *)src->ptr, rows);
    return cuda_ok(cudaGetLastError(), "unpack Spark index rows launch");
}
extern "C" int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!x || n_rot > head_dim || (n_rot & 1) || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    rope_tail_kernel<<<(pairs + 255) / 256, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, 1, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "rope_tail launch");
}
extern "C" int ds4_gpu_rope_tail_decode_rows_tensor(
        ds4_gpu_tensor                       *x,
        const ds4_gpu_attention_decode_row   *rows,
        uint32_t                              n_rows,
        uint32_t                              n_head,
        uint32_t                              head_dim,
        uint32_t                              n_rot,
        uint32_t                              n_ctx_orig,
        bool                                  inverse,
        float                                 freq_base,
        float                                 freq_scale,
        float                                 ext_factor,
        float                                 attn_factor,
        float                                 beta_fast,
        float                                 beta_slow) {
    if (!x || !rows || n_rows == 0u ||
        n_rows > DS4_GPU_ATTENTION_DECODE_BATCH_MAX || n_head == 0u ||
        n_rot == 0u || n_rot > head_dim || (n_rot & 1u) != 0u ||
        x->bytes < (uint64_t)n_rows * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    cuda_attention_decode_row_table table;
    memset(&table, 0, sizeof(table));
    for (uint32_t i = 0; i < n_rows; i++) table.row[i].pos = rows[i].pos;
    const uint32_t pairs = n_rows * n_head * (n_rot / 2u);
    rope_tail_decode_rows_kernel<<<(pairs + 255u) / 256u, 256>>>(
            (float *)x->ptr, table, n_rows, n_head, head_dim, n_rot,
            n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "rope tail decode rows launch");
}
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim);
extern "C" int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    if (!kv || !raw_cache || raw_cap == 0u || n_rot > head_dim ||
        kv->device_id != raw_cache->device_id ||
        kv->bytes < (uint64_t)head_dim * sizeof(float) ||
        raw_cache->bytes < (uint64_t)raw_cap * DS4_SPARK_KV_ROW_BYTES) {
        return 0;
    }
    cuda_attention_decode_row_table table;
    memset(&table, 0, sizeof(table));
    table.row[0].raw_kv = (uint64_t)(uintptr_t)raw_cache->ptr;
    table.row[0].raw_cap = raw_cap;
    table.row[0].raw_start = raw_row % raw_cap;
    spark_pack_kv_decode_rows_kernel<<<1, 64>>>(
            (float *)kv->ptr, table, 1u);
    return cuda_ok(cudaGetLastError(), "fp8 KV quantize/store launch");
}

extern "C" int ds4_gpu_kv_fp8_store_raw_decode_rows_tensor(
        ds4_gpu_tensor        *kv,
        ds4_gpu_tensor *const *raw_caches,
        const uint32_t        *raw_caps,
        const uint32_t        *raw_rows,
        uint32_t               n_rows,
        uint32_t               head_dim,
        uint32_t               n_rot) {
    if (!kv || !raw_caches || !raw_caps || !raw_rows || n_rows == 0u ||
        n_rows > DS4_GPU_ATTENTION_DECODE_BATCH_MAX || n_rot > head_dim ||
        kv->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    cuda_attention_decode_row_table table;
    memset(&table, 0, sizeof(table));
    for (uint32_t i = 0; i < n_rows; i++) {
        const ds4_gpu_tensor *raw = raw_caches[i];
        if (!raw || raw_caps[i] == 0u || raw_rows[i] >= raw_caps[i] ||
            raw->device_id != kv->device_id ||
            raw->bytes < (uint64_t)raw_caps[i] * DS4_SPARK_KV_ROW_BYTES) {
            return 0;
        }
        table.row[i].raw_kv = (uint64_t)(uintptr_t)raw->ptr;
        table.row[i].raw_cap = raw_caps[i];
        table.row[i].raw_start = raw_rows[i];
    }
    spark_pack_kv_decode_rows_kernel<<<n_rows, 64>>>(
            (float *)kv->ptr, table, n_rows);
    return cuda_ok(cudaGetLastError(), "fp8 KV quantize/store rows launch");
}
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        head_dim != 512u ||
        raw_cache->bytes < (uint64_t)raw_cap * DS4_SPARK_KV_ROW_BYTES ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    spark_pack_kv_rows_kernel<<<1, 64>>>(
            (unsigned char *)raw_cache->ptr, row % raw_cap,
            (float *)kv->ptr, 0, 1, false);
    return cuda_ok(cudaGetLastError(), "store_raw_kv launch");
}
extern "C" int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        head_dim != 512u ||
        raw_cache->bytes < (uint64_t)raw_cap * DS4_SPARK_KV_ROW_BYTES ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) return 0;
    spark_pack_kv_ring_rows_kernel<<<n_tokens, 64>>>(
            (unsigned char *)raw_cache->ptr, raw_cap, pos0,
            (float *)kv->ptr, n_tokens);
    return cuda_ok(cudaGetLastError(), "store_raw_kv_batch launch");
}
extern "C" int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(state_kv);
    const char *ape = cuda_resolve_weight_ptr(model_map, ape_offset, ape_bytes, logical_tier, "compressor_ape");
    if (!ape) return 0;
    uint64_t n = (uint64_t)n_tokens * width;
    compressor_store_kernel<<<(n + 255) / 256, 256>>>(
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (float *)state_kv->ptr,
            (float *)state_score->ptr,
            ape,
            0,
            ape_type,
            head_dim,
            ratio,
            pos0,
            n_tokens);
    return cuda_ok(cudaGetLastError(), "compressor store launch");
}

extern "C" int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        bool                    state_already_stored,
        bool                    decode_one_token,
        bool                    defer_finalize) {
    (void)decode_one_token;
    (void)defer_finalize;
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv_cur->bytes < kv_bytes || sc_cur->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (emit && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    if (!state_already_stored) {
        if (!ds4_gpu_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,
                                                     model_map, model_size, ape_offset, ape_type,
                                                     head_dim, ratio, pos, 1)) {
            return 0;
        }
    }
    if (!emit) return 1;
    ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
            comp_cache,
            (uint64_t)comp_row * head_dim * sizeof(float),
            (uint64_t)head_dim * sizeof(float));
    if (!comp_row_view) return 0;
    compressor_update_pool_kernel<<<(head_dim + 255) / 256, 256>>>(
            (float *)comp_row_view->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            head_dim,
            ratio);
    int ok = cuda_ok(cudaGetLastError(), "compressor update pool launch");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,
                                                       model_map, model_size, norm_offset,
                                                       head_dim, 1, rms_eps);
    if (ok) ok = ds4_gpu_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot,
                                            pos + 1u - ratio, n_ctx_orig, false,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            beta_fast, beta_slow);
    ds4_gpu_tensor_free(comp_row_view);
    if (ok && ratio == 4u) {
        uint64_t half = 4ull * width;
        compressor_shift_ratio4_kernel<<<(half + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr, width);
        ok = cuda_ok(cudaGetLastError(), "compressor ratio4 shift launch");
    }
    return ok;
}
extern "C" int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (n_comp && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(state_kv);
    const char *ape = cuda_resolve_weight_ptr(model_map, ape_offset, ape_bytes, logical_tier, "compressor_ape");
    if (!ape) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;

    if (ratio == 4u) {
        if (cutoff >= ratio) {
            uint32_t prev_start = cutoff - ratio;
            uint64_t n = (uint64_t)ratio * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    prev_start, 0, ratio);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill prev state launch")) return 0;
        }
        if (rem != 0) {
            uint64_t n = (uint64_t)rem * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    cutoff, ratio, rem);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
        }
    } else if (rem != 0) {
        uint64_t n = (uint64_t)rem * width;
        compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr,
                (const float *)kv->ptr, (const float *)sc->ptr,
                ape, 0, ape_type, width, ratio, pos0,
                cutoff, 0, rem);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
    }
    if (n_comp != 0) {
        dim3 grid((head_dim + 255) / 256, n_comp, 1);
        compressor_prefill_pool_kernel<<<grid, 256>>>(
                (float *)comp_cache->ptr,
                (const float *)kv->ptr,
                (const float *)sc->ptr,
                (const float *)state_kv->ptr,
                (const float *)state_score->ptr,
                ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 0);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill pool launch")) return 0;
        if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                                   model_map, model_size, norm_offset,
                                                   head_dim, n_comp, rms_eps)) return 0;
        if (n_rot != 0) {
            const uint32_t pairs = n_comp * (n_rot / 2u);
            rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                    (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                    pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rope launch")) return 0;
        }
        if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;
    }
    return 1;
}
extern "C" int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        comp_cache->bytes < comp_bytes) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(comp_cache);
    const char *ape = cuda_resolve_weight_ptr(model_map, ape_offset, ape_bytes, logical_tier, "compressor_ape");
    if (!ape) return 0;
    dim3 grid((head_dim + 255) / 256, n_comp, 1);
    compressor_prefill_pool_kernel<<<grid, 256>>>(
            (float *)comp_cache->ptr,
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 1);
    if (!cuda_ok(cudaGetLastError(), "compressor replay pool launch")) return 0;
    if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                               model_map, model_size, norm_offset,
                                               head_dim, n_comp, rms_eps)) return 0;
    if (n_rot != 0) {
        const uint32_t pairs = n_comp * (n_rot / 2u);
        rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        if (!cuda_ok(cudaGetLastError(), "compressor replay rope launch")) return 0;
    }
    if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor replay state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor replay state score fill launch")) return 0;
    uint32_t prev_start = n_tokens - ratio;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv->ptr, (const float *)sc->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            prev_start, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor replay state launch");
}
extern "C" int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(state_kv);
    const char *ape = cuda_resolve_weight_ptr(model_map, ape_offset, ape_bytes, logical_tier, "compressor_ape");
    if (!ape) return 0;
    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv_tail->ptr, (const float *)sc_tail->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            0, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor state set launch");
}

/* perf-02 split-KV / flash-decode launch helper (opt-in, default OFF).
 * Returns 1 if the split path handled the launch, 0 if the caller should fall
 * through to the existing attention_decode_mixed_kernel path.
 *
 * Engages only for the single-token decode shape (n_tokens==1) and only when
 * DS4_CUDA_SPLITKV_DECODE is set. S==1 is NOT handled here: the caller dispatches
 * the old kernel as the bit-exact anchor when S would be 1.
 */
static int attention_decode_splitkv_launch(
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
        uint32_t head_dim) {
    /* n_tokens is fixed at 1 for the split path; compute the EXACT logical row
     * count the kernel will use (raw_count + visible_comp) so S is sized to the
     * real work. raw_count MUST apply the same window logic as the kernel /
     * reference, otherwise a true-S==1 case (e.g. ratio=1, window=1, n_raw>=2,
     * n_comp=0) could be over-estimated to S>1 and engage split-KV instead of
     * the bit-exact old-kernel anchor. The count is head-independent. */
    const bool single_all = (ratio == 0u);
    uint32_t qpos = pos0;            /* t==0, n_tokens==1 */
    uint32_t first_raw_pos = pos0 + 1u - n_raw;
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
    uint32_t n_score = raw_count + visible_comp;
    if (n_score == 0u) return 0;     /* nothing to do; let old path handle it */
    const int manual_splitkv = cuda_env_flag_enabled("DS4_CUDA_SPLITKV_DECODE", 0);
    const uint32_t scoped_min_score =
        (g_decode_fast_attention && !manual_splitkv) ? 512u : 0u;
    uint32_t min_score = cuda_parse_u32_env_clamped("DS4_CUDA_SPLITKV_MIN_SCORE",
                                                    scoped_min_score, 0u,
                                                    DS4_CUDA_ATTENTION_SCORE_CAP,
                                                    NULL);
    if (n_score < min_score) return 0;
    /* S = clamp(ceil(n_score / CHUNK), 1, S_MAX); raise to S_FLOOR for short
     * context to fill more SMs, but never exceed n_score (no empty chunks).
     * Optional tuning knobs are guarded by min_needed so every block's chunk
     * still fits the fixed shared score buffer. */
    const uint32_t split_cap = DS4_CUDA_SPLITKV_SCORE_CAP;
    const uint32_t min_needed = (n_score + split_cap - 1u) / split_cap;
    uint32_t chunk = cuda_parse_u32_env_clamped("DS4_CUDA_SPLITKV_CHUNK",
                                                DS4_CUDA_SPLITKV_CHUNK,
                                                1u, split_cap, NULL);
    uint32_t s_floor = cuda_parse_u32_env_clamped("DS4_CUDA_SPLITKV_S_FLOOR",
                                                  DS4_CUDA_SPLITKV_S_FLOOR,
                                                  1u, DS4_CUDA_SPLITKV_S_MAX, NULL);
    uint32_t s_max = cuda_parse_u32_env_clamped("DS4_CUDA_SPLITKV_S_MAX",
                                                DS4_CUDA_SPLITKV_S_MAX,
                                                1u, DS4_CUDA_SPLITKV_S_MAX, NULL);
    int exact_present = 0;
    uint32_t S = cuda_parse_u32_env_clamped("DS4_CUDA_SPLITKV_S",
                                            0u, 1u, DS4_CUDA_SPLITKV_S_MAX,
                                            &exact_present);
    if (!exact_present) {
        S = (n_score + chunk - 1u) / chunk;
        if (S < s_floor) S = s_floor < n_score ? s_floor : n_score;
        if (S > s_max) S = s_max;
    }
    if (S < min_needed) S = min_needed;
    if (S > n_score) S = n_score;
    if (S <= 1u) return 0;           /* S==1: caller uses the old kernel anchor */
    if (cuda_env_flag_enabled("DS4_CUDA_SPLITKV_GLOBAL_SOFTMAX", 0)) {
        const uint64_t score_count = (uint64_t)n_head * n_score;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t denom_offset = (score_bytes + 255u) & ~255ull;
        const uint64_t denom_bytes = (uint64_t)n_head * sizeof(float);
        const uint64_t partial_offset = (denom_offset + denom_bytes + 255u) & ~255ull;
        const uint64_t partial_bytes = (uint64_t)n_head * S * head_dim * sizeof(float);
        void *tmp = cuda_tmp_alloc_on(logical_tier,
                                      partial_offset + partial_bytes,
                                      "attention splitkv global softmax");
        if (!tmp) return 0;
        float *scores = (float *)tmp;
        float *denom = (float *)((char *)tmp + denom_offset);
        float *partials = (float *)((char *)tmp + partial_offset);
        dim3 score_grid(1, n_head, S);
        attention_decode_score_split_scores_kernel<<<score_grid, 256>>>(
                scores, q, raw_kv, comp_kv, comp_mask, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim, S);
        if (!cuda_ok(cudaGetLastError(), "attention splitkv global score launch")) return -1;
        attention_decode_global_softmax_kernel<<<n_head, 256>>>(
                scores, denom, sinks, n_score, n_head);
        if (!cuda_ok(cudaGetLastError(), "attention splitkv global softmax launch")) return -1;
        dim3 value_grid(1, n_head, S);
        attention_decode_split_value_kernel<<<value_grid, 256>>>(
                partials, scores, raw_kv, comp_kv, raw_count, raw_first_idx,
                raw_cap, raw_start, n_score, n_head, head_dim, S);
        if (!cuda_ok(cudaGetLastError(), "attention splitkv global value launch")) return -1;
        dim3 combine_grid(1, n_head, 1);
        attention_decode_split_value_combine_kernel<<<combine_grid, 256>>>(
                heads, partials, denom, n_head, head_dim, S);
        if (!cuda_ok(cudaGetLastError(), "attention splitkv global combine launch")) return -1;
        return 1;
    }
    /* Partials scratch: n_head * S * (head_dim + 2) floats (n_tokens==1). */
    uint64_t stride = (uint64_t)head_dim + 2u;
    uint64_t count = (uint64_t)n_head * S * stride;
    float *partials = (float *)cuda_tmp_alloc_on(logical_tier, count * sizeof(float),
                                                 "attention splitkv partials");
    if (!partials) return 0;
    dim3 split_grid(1, n_head, S);
    attention_decode_splitkv_kernel<<<split_grid, 256>>>(partials,
                                                         q,
                                                         raw_kv,
                                                         comp_kv,
                                                         comp_mask,
                                                         use_comp_mask,
                                                         1, pos0, n_raw, raw_cap, raw_start,
                                                         n_comp, window, ratio, n_head, head_dim, S);
    if (!cuda_ok(cudaGetLastError(), "attention splitkv partial launch")) return -1;
    dim3 combine_grid(1, n_head, 1);
    attention_decode_splitkv_combine_kernel<<<combine_grid, 256>>>(heads,
                                                                   sinks,
                                                                   partials,
                                                                   1, n_head, head_dim, S);
    if (!cuda_ok(cudaGetLastError(), "attention splitkv combine launch")) return -1;
    return 1;
}

template <bool INDEXED>
__global__ static void __launch_bounds__(256, 2)
spark_attention_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const unsigned char *raw_kv,
        const unsigned char *comp_kv,
        const int32_t *topk,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t n_head) {
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t head = blockIdx.x * 8u + warp;
    const bool valid_head = head < n_head;
    __shared__ float kv_row[512];

    float out[16] = {0.0f};
    float max_score = valid_head ? sinks[head] : -INFINITY;
    float denom = valid_head ? 1.0f : 0.0f;
    const float *qh = valid_head ? q + (uint64_t)head * 512u : q;
    const float attn_scale = 0.04419417382415922f; /* 1/sqrt(512) */
    const uint32_t total_rows = n_raw + (INDEXED ? top_k : n_comp);

    for (uint32_t r = 0; r < total_rows; r++) {
        const bool is_raw = r < n_raw;
        uint32_t row = 0u;
        const unsigned char *base = raw_kv;
        if (is_raw) {
            row = (raw_start + r) % raw_cap;
        } else {
            base = comp_kv;
            row = INDEXED ? (uint32_t)topk[r - n_raw] : (r - n_raw);
            if (row >= n_comp) row = 0u;
        }
        const unsigned char *packed = base + (uint64_t)row * DS4_SPARK_KV_ROW_BYTES;
        kv_row[tid] = spark_kv_decode(packed, tid);
        kv_row[tid + 256u] = spark_kv_decode(packed, tid + 256u);
        __syncthreads();

        float dot = 0.0f;
        if (valid_head) {
#pragma unroll
            for (uint32_t i = 0; i < 16u; i++) {
                const uint32_t d = lane + i * 32u;
                dot += qh[d] * kv_row[d];
            }
            for (uint32_t off = 16u; off > 0u; off >>= 1u) {
                dot += __shfl_down_sync(0xffffffffu, dot, off);
            }
            const float score = __shfl_sync(0xffffffffu, dot, 0) * attn_scale;
            const float next_max = fmaxf(max_score, score);
            const float old_scale = expf(max_score - next_max);
            const float row_scale = expf(score - next_max);
            denom = denom * old_scale + row_scale;
#pragma unroll
            for (uint32_t i = 0; i < 16u; i++) {
                out[i] = out[i] * old_scale + kv_row[lane + i * 32u] * row_scale;
            }
            max_score = next_max;
        }
        __syncthreads();
    }

    if (valid_head) {
        float *dst = heads + (uint64_t)head * 512u;
#pragma unroll
        for (uint32_t i = 0; i < 16u; i++) {
            dst[lane + i * 32u] = out[i] / denom;
        }
    }
}

__global__ static void spark_unpack_kv_rows_kernel(
        float *dst,
        const unsigned char *src,
        const int32_t *rows,
        uint32_t n_rows,
        uint32_t src_cap,
        uint32_t src_start,
        uint32_t ring) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    uint32_t src_row = rows ? (uint32_t)rows[row] : row;
    if (ring) src_row = (src_start + row) % src_cap;
    const unsigned char *packed = src +
        (uint64_t)src_row * DS4_SPARK_KV_ROW_BYTES;
    dst[(uint64_t)row * 512u + threadIdx.x] =
        spark_kv_decode(packed, threadIdx.x);
    dst[(uint64_t)row * 512u + threadIdx.x + 256u] =
        spark_kv_decode(packed, threadIdx.x + 256u);
}

static int spark_attention_exact_rows_launch(
        int logical_tier,
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_f32,
        const float *comp_f32,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head) {
    if (!heads || !sinks || !q || !raw_f32 || n_tokens == 0u ||
        n_raw == 0u || n_head == 0u || (n_comp != 0u && !comp_f32) ||
        (topk && (top_k == 0u || top_k > 512u || ratio == 0u))) {
        return 0;
    }

    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    const uint32_t last_qpos = pos0 + n_tokens - 1u;
    uint32_t last_raw_lo = first_raw_pos;
    if (window != 0u && last_qpos + 1u > window) {
        const uint32_t window_lo = last_qpos + 1u - window;
        if (window_lo > last_raw_lo) last_raw_lo = window_lo;
    }
    const uint32_t max_raw_count = last_qpos - last_raw_lo + 1u;
    uint32_t max_visible_comp = n_comp;
    if (ratio != 0u) {
        max_visible_comp = (last_qpos + 1u) / ratio;
        if (max_visible_comp > n_comp) max_visible_comp = n_comp;
    }
    const uint32_t max_dense_comp = topk && max_visible_comp > top_k
        ? top_k : max_visible_comp;
    const uint32_t score_stride = max_raw_count + max_dense_comp;
    if (score_stride <= 1u ||
        score_stride > DS4_CUDA_ATTENTION_SCORE_CAP) {
        return 0;
    }

    const uint32_t wave_cap = DS4_GPU_ATTENTION_DECODE_BATCH_MAX;
    const uint64_t score_count =
        (uint64_t)wave_cap * n_head * score_stride;
    float *scores = (float *)cuda_tmp_alloc_on(
        logical_tier, score_count * sizeof(float),
        "Spark packed prefill exact rows");
    if (!scores) return 0;

    const size_t tile_shmem =
        (size_t)(DS4_SCORE_TILE_HEADS + DS4_SCORE_TILE_ROWS) *
        DS4_SCORE_TILE_STRIDE * sizeof(float);
    static int tile_shmem_ready[DS4_MAX_GPUS] = {0};
    int physical_device = 0;
    if (cudaGetDevice(&physical_device) != cudaSuccess ||
        physical_device < 0 || physical_device >= DS4_MAX_GPUS) {
        return 0;
    }
    if (!tile_shmem_ready[physical_device]) {
        if (!cuda_ok(cudaFuncSetAttribute(
                attention_decode_score_split_scores_tile512_rows_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)tile_shmem),
                "Spark packed prefill score rows shared-memory opt-in")) {
            return 0;
        }
        tile_shmem_ready[physical_device] = 1;
    }

    const uint64_t q_stride = (uint64_t)n_head * 512u;
    for (uint32_t wave0 = 0; wave0 < n_tokens; wave0 += wave_cap) {
        const uint32_t wave_rows = n_tokens - wave0 < wave_cap
            ? n_tokens - wave0 : wave_cap;
        cuda_attention_decode_row_table table;
        memset(&table, 0, sizeof(table));
        uint32_t wave_max_dense_score = 0u;
        bool have_dense = false;
        bool have_indexed = false;

        for (uint32_t i = 0; i < wave_rows; i++) {
            const uint32_t t = wave0 + i;
            const uint32_t qpos = pos0 + t;
            uint32_t raw_lo = first_raw_pos;
            if (window != 0u && qpos + 1u > window) {
                const uint32_t window_lo = qpos + 1u - window;
                if (window_lo > raw_lo) raw_lo = window_lo;
            }
            const uint32_t raw_count = qpos - raw_lo + 1u;
            uint32_t visible_comp = n_comp;
            if (ratio != 0u) {
                visible_comp = (qpos + 1u) / ratio;
                if (visible_comp > n_comp) visible_comp = n_comp;
            }

            ds4_gpu_attention_decode_row *row = &table.row[i];
            row->raw_kv = (uint64_t)(uintptr_t)(
                raw_f32 + (uint64_t)(raw_lo - first_raw_pos) * 512u);
            row->comp_kv = (uint64_t)(uintptr_t)(
                visible_comp ? comp_f32 : raw_f32);
            row->n_raw = raw_count;
            row->raw_cap = raw_count;
            row->raw_start = 0u;
            row->n_comp = visible_comp;

            if (topk && visible_comp > top_k) {
                row->topk = (uint64_t)(uintptr_t)(
                    topk + (uint64_t)t * top_k);
                row->pos = qpos;
                row->top_k = top_k;
                row->window = window;
                row->ratio = ratio;
                row->indexed = 1u;
                have_indexed = true;
            } else {
                row->pos = raw_count - 1u;
                row->ratio = 0u;
                const uint32_t n_score = raw_count + visible_comp;
                if (n_score <= 1u ||
                    n_score > DS4_CUDA_ATTENTION_SCORE_CAP) {
                    return 0;
                }
                if (n_score > wave_max_dense_score) {
                    wave_max_dense_score = n_score;
                }
                have_dense = true;
            }
        }

        float *wave_heads = heads + (uint64_t)wave0 * q_stride;
        const float *wave_q = q + (uint64_t)wave0 * q_stride;
        if (have_dense) {
            dim3 score_grid(
                (wave_max_dense_score + DS4_SCORE_TILE_ROWS - 1u) /
                    DS4_SCORE_TILE_ROWS,
                (n_head + DS4_SCORE_TILE_HEADS - 1u) /
                    DS4_SCORE_TILE_HEADS,
                wave_rows);
            attention_decode_score_split_scores_tile512_rows_kernel
                <<<score_grid, 256, tile_shmem>>>(
                    scores, wave_q, table, wave_rows,
                    score_stride, n_head, 512u);
            if (!cuda_ok(cudaGetLastError(),
                         "Spark packed prefill exact score rows launch")) {
                return -1;
            }
            dim3 final_grid(wave_rows, n_head, 1u);
            attention_decode_score_split_finalize_rows_kernel
                <<<final_grid, 512>>>(
                    wave_heads, sinks, scores, table, wave_rows,
                    score_stride, n_head, 512u);
            if (!cuda_ok(cudaGetLastError(),
                         "Spark packed prefill exact finalize rows launch")) {
                return -1;
            }
        }
        if (have_indexed) {
            dim3 indexed_grid(wave_rows, n_head, 1u);
            attention_indexed_mixed_decode_rows_kernel<<<indexed_grid, 256>>>(
                wave_heads, sinks, wave_q, table, wave_rows, n_head, 512u);
            if (!cuda_ok(cudaGetLastError(),
                         "Spark packed prefill exact indexed rows launch")) {
                return -1;
            }
        }
    }
    return 1;
}

static int spark_attention_unpack_launch(
        int logical_tier,
        float *heads,
        const float *sinks,
        const float *q,
        const unsigned char *raw_kv,
        const unsigned char *comp_kv,
        const int32_t *topk,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t selected_comp,
        uint32_t n_head) {
    const uint32_t comp_rows = topk ? selected_comp : n_comp;
    const uint64_t raw_bytes = (uint64_t)n_raw * 512u * sizeof(float);
    const uint64_t comp_offset = (raw_bytes + 255u) & ~255ull;
    const uint64_t comp_bytes = (uint64_t)comp_rows * 512u * sizeof(float);
    unsigned char *scratch = (unsigned char *)tt_scratch_ensure(
        comp_offset + (comp_bytes ? comp_bytes : 256u),
        "allocate Spark attention unpack scratch");
    if (!scratch) return 0;
    float *raw_f32 = (float *)scratch;
    float *comp_f32 = (float *)(scratch + comp_offset);
    spark_unpack_kv_rows_kernel<<<n_raw, 256>>>(
        raw_f32, raw_kv, NULL, n_raw, raw_cap, raw_start, 1u);
    if (!cuda_ok(cudaGetLastError(), "unpack Spark raw KV launch")) return -1;
    if (comp_rows != 0u) {
        spark_unpack_kv_rows_kernel<<<comp_rows, 256>>>(
            comp_f32, comp_kv, topk, comp_rows, n_comp, 0u, 0u);
        if (!cuda_ok(cudaGetLastError(), "unpack Spark compressed KV launch")) return -1;
    }
    const int rc = attention_decode_score_split_launch(
        logical_tier, heads, sinks, q, raw_f32,
        comp_rows ? comp_f32 : raw_f32,
        NULL, 0u, 0u, n_raw, n_raw, 0u, comp_rows,
        0u, 0u, n_head, 512u, 512u, NULL);
    return rc;
}

static int spark_indexed_attention_unpack_launch(
        float *heads,
        const float *sinks,
        const float *q,
        const unsigned char *raw_kv,
        const unsigned char *comp_kv,
        const int32_t *topk,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head) {
    const uint64_t raw_bytes = (uint64_t)n_raw * 512u * sizeof(float);
    const uint64_t comp_offset = (raw_bytes + 255u) & ~255ull;
    const uint64_t comp_bytes = (uint64_t)top_k * 512u * sizeof(float);
    const uint64_t topk_offset = (comp_offset + comp_bytes + 255u) & ~255ull;
    const uint64_t topk_bytes = (uint64_t)top_k * sizeof(int32_t);
    unsigned char *scratch = (unsigned char *)tt_scratch_ensure(
        topk_offset + topk_bytes, "allocate Spark indexed unpack scratch");
    if (!scratch) return 0;
    float *raw_f32 = (float *)scratch;
    float *comp_f32 = (float *)(scratch + comp_offset);
    int32_t *compact_topk = (int32_t *)(scratch + topk_offset);
    spark_unpack_kv_rows_kernel<<<n_raw, 256>>>(
        raw_f32, raw_kv, NULL, n_raw, raw_cap, raw_start, 1u);
    spark_unpack_kv_rows_kernel<<<top_k, 256>>>(
        comp_f32, comp_kv, topk, top_k, n_comp, 0u, 0u);
    spark_fill_iota_i32_kernel<<<(top_k + 255u) / 256u, 256>>>(
        compact_topk, top_k);
    if (!cuda_ok(cudaGetLastError(), "prepare Spark indexed attention")) return -1;
    dim3 grid(1u, n_head, 1u);
    attention_indexed_mixed_kernel<<<grid, 256>>>(
        heads, sinks, q, raw_f32, comp_f32, compact_topk,
        1u, pos0, n_raw, n_raw, 0u, top_k, top_k,
        window, ratio, n_head, 512u);
    return cuda_ok(cudaGetLastError(), "Spark indexed attention launch") ? 1 : -1;
}

