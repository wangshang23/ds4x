/* Default packed prefill decodes persistent rows directly into FP16 token-tile
 * mirrors for HMMA. Unsupported shapes and exact mode retain the reusable F32
 * fallback; neither path changes the persistent cache ABI. */
static int spark_attention_packed_dense_tokentile_launch(
        float *heads,
        const float *sinks,
        const float *q,
        const unsigned char *raw_kv,
        const unsigned char *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head) {
    if (g_n_gpus != 1 || n_tokens < 128u || n_head != 64u ||
        window != kTTRawWindow || ratio == 0u || n_comp > 32768u ||
        n_raw < n_tokens || (uint64_t)n_raw > (uint64_t)pos0 + n_tokens ||
        g_cuda_no_window_attention || !ds4_cuda_attn_tokentile_arch_ok()) {
        return 0;
    }
    cudaStream_t stream = cuda_decode_stream();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        (void)cudaGetLastError();
        return 0;
    }

    const uint32_t n_tiles =
        (n_tokens + kTTTileTokens - 1u) / kTTTileTokens;
    const uint32_t rec_stride = n_comp;
    const uint32_t n_mirror_rows = n_tokens + kTTRawWindow - 1u;
    const uint32_t raw_before = n_raw - n_tokens;
    const uint32_t available_before = raw_before < pos0 ? raw_before : pos0;
    const uint32_t available_clamped =
        available_before < (kTTRawWindow - 1u)
            ? available_before : (kTTRawWindow - 1u);
    const uint32_t raw_row_min =
        (kTTRawWindow - 1u) - available_clamped;
    const uint32_t first_raw_pos = (uint32_t)(
        (uint64_t)pos0 + n_tokens - n_raw);

    uint64_t off = 0;
    const uint64_t records_off = off;
    off = tt_align256_u64(
        off + (uint64_t)n_tiles * (rec_stride ? rec_stride : 1u) *
              sizeof(int2));
    const uint64_t counts_off = off;
    off = tt_align256_u64(off + (uint64_t)n_tiles * sizeof(uint32_t));
    const uint64_t raw_mirror_off = off;
    off = tt_align256_u64(
        off + (uint64_t)n_mirror_rows * kTTHeadDim * sizeof(half));
    const uint64_t comp_mirror_off = off;
    off = tt_align256_u64(
        off + (uint64_t)n_comp * kTTHeadDim * sizeof(half));

    unsigned char *scratch = (unsigned char *)tt_scratch_ensure(
        off, "allocate packed dense token-tile attention scratch");
    if (!scratch) return 0;
    int2 *records = (int2 *)(scratch + records_off);
    uint32_t *counts = (uint32_t *)(scratch + counts_off);
    half *raw_mirror = (half *)(scratch + raw_mirror_off);
    half *comp_mirror = (half *)(scratch + comp_mirror_off);
    if (!cuda_ok(cudaFuncSetAttribute(
            attention_tokentile_hmma_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)tt_TokentileSmemBudget<kTTStageRows, kTTG>::total),
            "set packed dense token-tile HMMA shared-memory limit")) {
        return -1;
    }

    attention_tokentile_dense_build_kernel
        <<<n_tiles, kTTThreads, 0, stream>>>(
            records, counts, pos0, n_tokens, ratio, n_comp, rec_stride);
    if (!cuda_ok(cudaGetLastError(),
                 "launch packed dense token-tile record builder")) {
        return -1;
    }
    attention_tokentile_raw_packed_mirror_kernel
        <<<n_mirror_rows, 256, 0, stream>>>(
            raw_mirror, raw_kv, pos0, raw_cap, raw_start, first_raw_pos,
            raw_row_min, kTTHeadDim);
    if (!cuda_ok(cudaGetLastError(),
                 "launch packed dense token-tile raw FP16 mirror")) {
        return -1;
    }
    if (n_comp != 0u) {
        attention_tokentile_comp_packed_mirror_kernel
            <<<n_comp, 256, 0, stream>>>(
                comp_mirror, comp_kv, n_comp, kTTHeadDim);
        if (!cuda_ok(cudaGetLastError(),
                     "launch packed dense token-tile comp FP16 mirror")) {
            return -1;
        }
    }
    const dim3 grid(n_tiles, n_head / kTTG, 1u);
    attention_tokentile_hmma_kernel
        <<<grid, kTTThreads,
           tt_TokentileSmemBudget<kTTStageRows, kTTG>::total, stream>>>(
            heads, sinks, q, raw_mirror, comp_mirror, records, counts,
            rec_stride, n_tokens, n_head, raw_row_min);
    return cuda_ok(cudaGetLastError(),
                   "launch packed dense token-tile HMMA attention")
        ? 1 : -1;
}

static int spark_attention_packed_indexed_tokentile_launch(
        float *heads,
        const float *sinks,
        const float *q,
        const unsigned char *raw_kv,
        const unsigned char *comp_kv,
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
        uint32_t n_head) {
    if (g_n_gpus != 1 || n_tokens < 128u || n_head != 64u ||
        top_k != 512u || window != kTTRawWindow || ratio == 0u ||
        n_comp > 32768u || n_raw < n_tokens ||
        (uint64_t)n_raw > (uint64_t)pos0 + n_tokens ||
        g_cuda_no_window_attention || !ds4_cuda_attn_tokentile_arch_ok()) {
        return 0;
    }
    cudaStream_t stream = cuda_decode_stream();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        (void)cudaGetLastError();
        return 0;
    }

    const uint32_t n_tiles =
        (n_tokens + kTTTileTokens - 1u) / kTTTileTokens;
    const uint32_t rec_stride =
        (kTTTileTokens * top_k) < n_comp
            ? (kTTTileTokens * top_k) : n_comp;
    const uint32_t n_mirror_rows = n_tokens + kTTRawWindow - 1u;
    const uint32_t raw_before = n_raw - n_tokens;
    const uint32_t available_before = raw_before < pos0 ? raw_before : pos0;
    const uint32_t available_clamped =
        available_before < (kTTRawWindow - 1u)
            ? available_before : (kTTRawWindow - 1u);
    const uint32_t raw_row_min =
        (kTTRawWindow - 1u) - available_clamped;
    const uint32_t first_raw_pos = (uint32_t)(
        (uint64_t)pos0 + n_tokens - n_raw);

    uint64_t off = 0;
    const uint64_t records_off = off;
    off = tt_align256_u64(
        off + (uint64_t)n_tiles * rec_stride * sizeof(int2));
    const uint64_t counts_off = off;
    off = tt_align256_u64(off + (uint64_t)n_tiles * sizeof(uint32_t));
    const uint64_t raw_mirror_off = off;
    off = tt_align256_u64(
        off + (uint64_t)n_mirror_rows * kTTHeadDim * sizeof(half));
    const uint64_t comp_mirror_off = off;
    off = tt_align256_u64(
        off + (uint64_t)n_comp * kTTHeadDim * sizeof(half));

    unsigned char *scratch = (unsigned char *)tt_scratch_ensure(
        off, "allocate packed indexed token-tile attention scratch");
    if (!scratch) return 0;
    int2 *records = (int2 *)(scratch + records_off);
    uint32_t *counts = (uint32_t *)(scratch + counts_off);
    half *raw_mirror = (half *)(scratch + raw_mirror_off);
    half *comp_mirror = (half *)(scratch + comp_mirror_off);
    const uint32_t bitmap_words = (n_comp + 1u) >> 1u;
    const size_t bitmap_smem = (size_t)bitmap_words * sizeof(uint32_t);
    static int packed_union_smem_attr_set = 0;
    if (!packed_union_smem_attr_set) {
        const int max_bitmap_smem =
            (int)(((32768u + 1u) >> 1u) * sizeof(uint32_t));
        if (!cuda_ok(cudaFuncSetAttribute(
                attention_tokentile_union_build_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                max_bitmap_smem),
                "set packed token-tile union shared-memory limit")) {
            return -1;
        }
        packed_union_smem_attr_set = 1;
    }
    if (!cuda_ok(cudaFuncSetAttribute(
            attention_tokentile_hmma_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)tt_TokentileSmemBudget<kTTStageRows, kTTG>::total),
            "set packed indexed token-tile HMMA shared-memory limit")) {
        return -1;
    }

    attention_tokentile_union_build_kernel
        <<<n_tiles, kTTThreads, bitmap_smem, stream>>>(
            records, counts, topk, NULL, pos0, n_tokens, top_k, ratio,
            n_comp, rec_stride);
    if (!cuda_ok(cudaGetLastError(),
                 "launch packed token-tile union builder")) {
        return -1;
    }
    attention_tokentile_raw_packed_mirror_kernel
        <<<n_mirror_rows, 256, 0, stream>>>(
            raw_mirror, raw_kv, pos0, raw_cap, raw_start, first_raw_pos,
            raw_row_min, kTTHeadDim);
    if (!cuda_ok(cudaGetLastError(),
                 "launch packed indexed token-tile raw FP16 mirror")) {
        return -1;
    }
    attention_tokentile_comp_packed_mirror_kernel
        <<<n_comp, 256, 0, stream>>>(
            comp_mirror, comp_kv, n_comp, kTTHeadDim);
    if (!cuda_ok(cudaGetLastError(),
                 "launch packed indexed token-tile comp FP16 mirror")) {
        return -1;
    }
    const dim3 grid(n_tiles, n_head / kTTG, 1u);
    attention_tokentile_hmma_kernel
        <<<grid, kTTThreads,
           tt_TokentileSmemBudget<kTTStageRows, kTTG>::total, stream>>>(
            heads, sinks, q, raw_mirror, comp_mirror, records, counts,
            rec_stride, n_tokens, n_head, raw_row_min);
    return cuda_ok(cudaGetLastError(),
                   "launch packed indexed token-tile HMMA attention")
        ? 1 : -1;
}

static int spark_attention_unpack_batch_launch(
        int logical_tier,
        float *heads,
        const float *sinks,
        const float *q,
        const unsigned char *raw_kv,
        const unsigned char *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head) {
    if (n_tokens <= 1u || n_raw == 0u || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0u && !comp_kv)) {
        return 0;
    }
    if (getenv("DS4_CUDA_SPARK_PREFILL_EXACT") == NULL) {
        const int tt = spark_attention_packed_dense_tokentile_launch(
            heads, sinks, q, raw_kv, comp_kv, n_tokens, pos0,
            n_raw, raw_cap, raw_start, n_comp, window, ratio, n_head);
        if (tt != 0) return tt;
    }
    const uint64_t raw_bytes = (uint64_t)n_raw * 512u * sizeof(float);
    const uint64_t comp_offset = tt_align256_u64(raw_bytes);
    const uint64_t comp_bytes = (uint64_t)n_comp * 512u * sizeof(float);
    unsigned char *scratch = (unsigned char *)tt_scratch_ensure(
        comp_offset + (comp_bytes ? comp_bytes : 256u),
        "allocate Spark packed prefill scratch");
    if (!scratch) return 0;
    float *raw_f32 = (float *)scratch;
    float *comp_f32 = (float *)(scratch + comp_offset);
    spark_unpack_kv_rows_kernel<<<n_raw, 256>>>(
        raw_f32, raw_kv, NULL, n_raw, raw_cap, raw_start, 1u);
    if (!cuda_ok(cudaGetLastError(), "unpack Spark prefill raw KV launch")) {
        return -1;
    }
    if (n_comp != 0u) {
        spark_unpack_kv_rows_kernel<<<n_comp, 256>>>(
            comp_f32, comp_kv, NULL, n_comp, n_comp, 0u, 0u);
        if (!cuda_ok(cudaGetLastError(),
                     "unpack Spark prefill compressed KV launch")) {
            return -1;
        }
    }
    if (getenv("DS4_CUDA_SPARK_PREFILL_EXACT") == NULL) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1u);
        attention_decode_mixed_heads8_online_kernel<<<grid, 256>>>(
            heads, sinks, q, raw_f32, n_comp ? comp_f32 : raw_f32,
            n_tokens, pos0, n_raw, n_raw, 0u, n_comp, window, ratio,
            n_head, 512u);
        return cuda_ok(cudaGetLastError(),
                       "Spark packed online prefill attention launch")
            ? 1 : -1;
    }
    return spark_attention_exact_rows_launch(
        logical_tier, heads, sinks, q, raw_f32,
        n_comp ? comp_f32 : raw_f32, NULL,
        n_tokens, pos0, n_raw, n_comp, 0u, window, ratio, n_head);
}

static int spark_indexed_attention_unpack_batch_launch(
        int logical_tier,
        float *heads,
        const float *sinks,
        const float *q,
        const unsigned char *raw_kv,
        const unsigned char *comp_kv,
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
        uint32_t n_head) {
    if (n_tokens <= 1u || n_raw == 0u || raw_cap < n_raw ||
        raw_start >= raw_cap || n_comp == 0u || top_k == 0u ||
        top_k > 512u || !comp_kv || !topk) {
        return 0;
    }
    if (getenv("DS4_CUDA_SPARK_PREFILL_EXACT") == NULL) {
        const int tt = spark_attention_packed_indexed_tokentile_launch(
            heads, sinks, q, raw_kv, comp_kv, topk, n_tokens, pos0,
            n_raw, raw_cap, raw_start, n_comp, top_k, window, ratio,
            n_head);
        if (tt != 0) return tt;
    }
    const uint64_t raw_bytes = (uint64_t)n_raw * 512u * sizeof(float);
    const uint64_t comp_offset = tt_align256_u64(raw_bytes);
    const uint64_t comp_bytes = (uint64_t)n_comp * 512u * sizeof(float);
    unsigned char *scratch = (unsigned char *)tt_scratch_ensure(
        comp_offset + comp_bytes,
        "allocate Spark packed indexed prefill scratch");
    if (!scratch) return 0;
    float *raw_f32 = (float *)scratch;
    float *comp_f32 = (float *)(scratch + comp_offset);
    spark_unpack_kv_rows_kernel<<<n_raw, 256>>>(
        raw_f32, raw_kv, NULL, n_raw, raw_cap, raw_start, 1u);
    if (!cuda_ok(cudaGetLastError(),
                 "unpack Spark indexed prefill raw KV launch")) {
        return -1;
    }
    spark_unpack_kv_rows_kernel<<<n_comp, 256>>>(
        comp_f32, comp_kv, NULL, n_comp, n_comp, 0u, 0u);
    if (!cuda_ok(cudaGetLastError(),
                 "unpack Spark indexed prefill compressed KV launch")) {
        return -1;
    }
    if (getenv("DS4_CUDA_SPARK_PREFILL_EXACT") == NULL) {
        dim3 grid(n_tokens, (n_head + 15u) / 16u, 1u);
        attention_indexed_mixed_heads8_online_kernel<8, 16>
            <<<grid, 512>>>(
                heads, sinks, q, raw_f32, comp_f32, topk,
                n_tokens, pos0, n_raw, n_raw, 0u, n_comp, top_k,
                window, ratio, n_head, 512u);
        return cuda_ok(cudaGetLastError(),
                       "Spark packed online indexed prefill attention launch")
            ? 1 : -1;
    }
    return spark_attention_exact_rows_launch(
        logical_tier, heads, sinks, q, raw_f32, comp_f32, topk,
        n_tokens, pos0, n_raw, n_comp, top_k, window, ratio, n_head);
}

extern "C" int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_format,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * DS4_SPARK_KV_ROW_BYTES ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * DS4_SPARK_KV_ROW_BYTES) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
        model_map, sinks_offset, (uint64_t)n_head * sizeof(float), logical_tier, "attn_sinks");
    if (!sinks) return 0;
    if (comp_kv_format == DS4_GPU_CACHE_SPARK_KV && head_dim == 512u &&
        !use_mask) {
        const int rc = spark_attention_unpack_launch(
            logical_tier, (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const unsigned char *)raw_kv->ptr,
            n_comp ? (const unsigned char *)comp_kv->ptr
                   : (const unsigned char *)raw_kv->ptr,
            NULL, n_raw, raw_cap, raw_start, n_comp, 0u, n_head);
        if (rc == 1) return cuda_ok(cudaGetLastError(), "Spark packed attention launch");
        return 0;
    }
    if (comp_kv_format != DS4_GPU_CACHE_F32) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_mask && head_dim == 512u &&
            !g_cuda_no_window_attention) {
            const uint32_t synthetic_pos0 = n_raw - 1u;
            dim3 online_grid(1, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              1,
                                                                              synthetic_pos0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              0,
                                                                              0,
                                                                              n_head,
                                                                              head_dim);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    if (!use_mask && head_dim == 512u &&
        g_cuda_decode_heads8_online &&
        !g_cuda_no_window_attention) {
        const uint32_t synthetic_pos0 = n_raw - 1u;
        dim3 online_grid(1, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                          sinks,
                                                                          (const float *)q->ptr,
                                                                          (const float *)raw_kv->ptr,
                                                                          n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                          1,
                                                                          synthetic_pos0,
                                                                          n_raw,
                                                                          raw_cap,
                                                                          raw_start,
                                                                          n_comp,
                                                                          0,
                                                                          0,
                                                                          n_head,
                                                                          head_dim);
        return cuda_ok(cudaGetLastError(), "attention decode heads8 online launch");
    }
    const uint32_t score_lanes =
        g_cuda_decode_score4 ? 4u : (g_cuda_decode_score8 ? 8u : 0u);
    const uint32_t threads =
        head_dim == 512u && score_lanes == 0u &&
        !g_cuda_no_decode_value512 ? 512u : 256u;
    int score_split_rc = attention_decode_score_split_launch(
            logical_tier, (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
            use_mask ? (const float *)comp_mask->ptr : NULL, use_mask,
            0, n_raw, raw_cap, raw_start, n_comp, 0, 0,
            n_head, head_dim, threads, NULL);
    if (score_split_rc == 1) {
        return cuda_ok(cudaGetLastError(), "attention exact score split launch");
    }
    if (score_split_rc < 0) return 0;
    /* perf-02 split-KV opt-in (default OFF). n_tokens==1 here by construction.
     * S==1 / disabled / unhandled -> rc 0, fall through to the old kernel. */
    if (cuda_splitkv_decode_requested()) {
        int rc = attention_decode_splitkv_launch(
                logical_tier, (float *)heads->ptr, sinks, (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                use_mask ? (const float *)comp_mask->ptr : NULL, use_mask,
                0, n_raw, raw_cap, raw_start, n_comp, 0, 0, n_head, head_dim);
        if (rc == 1) return cuda_ok(cudaGetLastError(), "attention decode splitkv launch");
        if (rc < 0) return 0;
    }
    dim3 grid(1, n_head, 1);
    attention_decode_mixed_kernel<<<grid, threads>>>((float *)heads->ptr,
                                                     sinks,
                                                     (const float *)q->ptr,
                                                     (const float *)raw_kv->ptr,
                                                     n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                     use_mask ? (const float *)comp_mask->ptr : NULL,
                                                     use_mask,
                                                     1, 0, n_raw, raw_cap, raw_start, n_comp,
                                                     0, 0, n_head, head_dim,
                                                     score_lanes);
    return cuda_ok(cudaGetLastError(), "attention decode launch");
}

extern "C" int ds4_gpu_attention_decode_heads_rope_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        int                    *fused_inv_rope) {
    if (fused_inv_rope) *fused_inv_rope = 0;
    if (!g_cuda_exact_score_split_fuse_inv_rope ||
        n_rot == 0u || n_rot > head_dim || (n_rot & 1u) ||
        head_dim != 512u) {
        return ds4_gpu_attention_decode_heads_tensor(
                heads, model_map, model_size, sinks_offset, q, raw_kv,
                n_raw, raw_cap, raw_start, comp_kv, comp_kv_f16, n_comp,
                comp_mask, use_mask, n_head, head_dim);
    }
    if (!use_mask && g_cuda_decode_heads8_online && !g_cuda_no_window_attention) {
        return ds4_gpu_attention_decode_heads_tensor(
                heads, model_map, model_size, sinks_offset, q, raw_kv,
                n_raw, raw_cap, raw_start, comp_kv, comp_kv_f16, n_comp,
                comp_mask, use_mask, n_head, head_dim);
    }
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !model_map || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float)) ||
        !cuda_attention_score_buffer_fits(n_comp)) {
        return ds4_gpu_attention_decode_heads_tensor(
                heads, model_map, model_size, sinks_offset, q, raw_kv,
                n_raw, raw_cap, raw_start, comp_kv, comp_kv_f16, n_comp,
                comp_mask, use_mask, n_head, head_dim);
    }
    const uint32_t score_lanes =
        g_cuda_decode_score4 ? 4u : (g_cuda_decode_score8 ? 8u : 0u);
    const uint32_t threads =
        score_lanes == 0u && !g_cuda_no_decode_value512 ? 512u : 256u;
    if (threads < 512u) {
        return ds4_gpu_attention_decode_heads_tensor(
                heads, model_map, model_size, sinks_offset, q, raw_kv,
                n_raw, raw_cap, raw_start, comp_kv, comp_kv_f16, n_comp,
                comp_mask, use_mask, n_head, head_dim);
    }
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float),
            logical_tier, "attn_sinks");
    if (!sinks) return 0;
    cuda_attention_inv_rope_params rope;
    rope.n_rot = n_rot;
    rope.pos0 = pos0;
    rope.n_ctx_orig = n_ctx_orig;
    rope.freq_base = freq_base;
    rope.freq_scale = freq_scale;
    rope.ext_factor = ext_factor;
    rope.attn_factor = attn_factor;
    rope.beta_fast = beta_fast;
    rope.beta_slow = beta_slow;
    int score_split_rc = attention_decode_score_split_launch(
            logical_tier, (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const float *)raw_kv->ptr,
            n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
            use_mask ? (const float *)comp_mask->ptr : NULL, use_mask,
            0, n_raw, raw_cap, raw_start, n_comp, 0, 0,
            n_head, head_dim, threads, &rope);
    if (score_split_rc == 1) {
        if (fused_inv_rope) *fused_inv_rope = 1;
        return cuda_ok(cudaGetLastError(),
                       "attention exact score split fused inv rope launch");
    }
    if (score_split_rc < 0) return 0;
    return ds4_gpu_attention_decode_heads_tensor(
            heads, model_map, model_size, sinks_offset, q, raw_kv,
            n_raw, raw_cap, raw_start, comp_kv, comp_kv_f16, n_comp,
            comp_mask, use_mask, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_decode_rows_rope_tensor(
        ds4_gpu_tensor                       *heads,
        const void                           *model_map,
        uint64_t                              model_size,
        uint64_t                              sinks_offset,
        const ds4_gpu_tensor                 *q,
        const ds4_gpu_attention_decode_row   *rows,
        uint32_t                              n_rows,
        uint32_t                              n_head,
        uint32_t                              head_dim,
        uint32_t                              n_rot,
        uint32_t                              n_ctx_orig,
        float                                 freq_base,
        float                                 freq_scale,
        float                                 ext_factor,
        float                                 attn_factor,
        float                                 beta_fast,
        float                                 beta_slow) {
    if (!heads || !q || !rows || !model_map || n_rows < 2u ||
        n_rows > DS4_GPU_ATTENTION_DECODE_BATCH_MAX || n_head == 0u ||
        head_dim != 512u || n_rot == 0u || n_rot > head_dim ||
        (n_rot & 1u) != 0u || sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_rows * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_rows * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(heads);
    if (logical_tier < 0 || logical_tier >= g_n_gpus ||
        ds4_tensor_device_idx(q) != logical_tier) {
        return 0;
    }

    /* This first grouped path mirrors the promoted default decode exactly.
     * Alternative score kernels and graph/split-KV experiments retain the
     * one-session dispatcher until they gain equivalent row-table variants. */
    if (cuda_env_flag_enabled("DS4_CUDA_NO_EXACT_SCORE_SPLIT_DECODE", 0) ||
        !cuda_env_flag_enabled("DS4_CUDA_EXACT_SCORE_SPLIT_DECODE", 1) ||
        cuda_splitkv_decode_requested() ||
        g_cuda_decode_heads8_online || g_cuda_decode_score4 ||
        g_cuda_decode_score8 || g_cuda_no_decode_value512 ||
        g_cuda_exact_score_split_graph || g_cuda_exact_score_split_ldg ||
        g_cuda_exact_score_split_vec4 ||
        g_cuda_exact_score_split_vec4_plain ||
        g_cuda_exact_score_split_dim2 ||
        g_cuda_exact_score_split_fuse_inv_rope ||
        getenv("DS4_CUDA_NO_SCORE_TILE") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_MIN_SCORE") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_CHUNK") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_S_FLOOR") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_S_MAX") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_S") != NULL) {
        return 0;
    }

    cuda_attention_decode_row_table table;
    memset(&table, 0, sizeof(table));
    uint32_t max_dense_score = 0u;
    bool have_dense = false;
    bool have_indexed = false;
    for (uint32_t i = 0; i < n_rows; i++) {
        const ds4_gpu_attention_decode_row r = rows[i];
        if (r.raw_kv == 0u || r.n_raw == 0u || r.raw_cap < r.n_raw ||
            r.raw_start >= r.raw_cap || (r.n_comp != 0u && r.comp_kv == 0u)) {
            return 0;
        }
        if (r.indexed) {
            if (r.comp_kv == 0u || r.topk == 0u || r.n_comp == 0u ||
                r.top_k == 0u || r.top_k > 512u || r.ratio == 0u) {
                return 0;
            }
            have_indexed = true;
        } else {
            const uint32_t raw_count = r.n_raw > 256u ? 256u : r.n_raw;
            const uint32_t n_score = raw_count + r.n_comp;
            /* n_score==1 takes the legacy one-block kernel and is not a
             * score-split shape. Decode after any nonempty prompt is >1. */
            if (n_score <= 1u || n_score > DS4_CUDA_ATTENTION_SCORE_CAP) {
                return 0;
            }
            if (n_score > max_dense_score) max_dense_score = n_score;
            have_dense = true;
        }
        table.row[i] = r;
    }

    const float *sinks = (const float *)cuda_resolve_weight_ptr(
        model_map, sinks_offset, (uint64_t)n_head * sizeof(float),
        logical_tier, "attn_sinks_rows");
    if (!sinks) return 0;

    if (have_dense) {
        if ((uint64_t)n_rows > UINT64_MAX / n_head ||
            (uint64_t)n_rows * n_head > UINT64_MAX / max_dense_score) {
            return 0;
        }
        const uint64_t score_count =
            (uint64_t)n_rows * n_head * max_dense_score;
        float *scores = (float *)cuda_tmp_alloc_on(
            logical_tier, score_count * sizeof(float),
            "attention exact decode rows");
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
                    "attention score rows shared-memory opt-in")) {
                return 0;
            }
            tile_shmem_ready[physical_device] = 1;
        }
        dim3 score_grid(
            (max_dense_score + DS4_SCORE_TILE_ROWS - 1u) /
                DS4_SCORE_TILE_ROWS,
            (n_head + DS4_SCORE_TILE_HEADS - 1u) /
                DS4_SCORE_TILE_HEADS,
            n_rows);
        attention_decode_score_split_scores_tile512_rows_kernel
            <<<score_grid, 256, tile_shmem>>>(
                scores, (const float *)q->ptr, table, n_rows,
                max_dense_score, n_head, head_dim);
        if (!cuda_ok(cudaGetLastError(),
                     "attention exact score rows launch")) {
            return 0;
        }
        dim3 final_grid(n_rows, n_head, 1u);
        attention_decode_score_split_finalize_rows_kernel
            <<<final_grid, 512>>>(
                (float *)heads->ptr, sinks, scores, table, n_rows,
                max_dense_score, n_head, head_dim);
        if (!cuda_ok(cudaGetLastError(),
                     "attention exact finalize rows launch")) {
            return 0;
        }
    }
    if (have_indexed) {
        dim3 indexed_grid(n_rows, n_head, 1u);
        attention_indexed_mixed_decode_rows_kernel<<<indexed_grid, 256>>>(
            (float *)heads->ptr, sinks, (const float *)q->ptr, table,
            n_rows, n_head, head_dim);
        if (!cuda_ok(cudaGetLastError(),
                     "attention indexed decode rows launch")) {
            return 0;
        }
    }

    const uint32_t pairs = n_rows * n_head * (n_rot / 2u);
    rope_tail_decode_rows_kernel<<<(pairs + 255u) / 256u, 256>>>(
        (float *)heads->ptr, table, n_rows, n_head, head_dim, n_rot,
        n_ctx_orig, 1, freq_base, freq_scale, ext_factor, attn_factor,
        beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(),
                   "attention decode rows inverse rope launch");
}

extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !model_map || sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        window > 256) return 0;
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), logical_tier, "attn_sinks");
    if (!sinks) return 0;
    if (n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   0,
                                                                   window,
                                                                   1,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION") == NULL) {
        const uint32_t n_keys = n_tokens;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = (score_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc_on(logical_tier, tmp_bytes, "attention raw cublas");
        if (!tmp) return 0;
        float *scores = tmp;
        float *out_tmp = (float *)((char *)tmp + out_offset);
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(cuda_cublas_for_tier(logical_tier),
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      (const float *)raw_kv->ptr,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention raw score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_raw_softmax_kernel<<<sgrid, 256>>>(scores, sinks, n_tokens, window, n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention raw softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(cuda_cublas_for_tier(logical_tier),
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       (const float *)raw_kv->ptr,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention raw value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_raw_kernel<<<grid, 128>>>((float *)heads->ptr,
                                                sinks,
                                                (const float *)q->ptr,
                                                (const float *)raw_kv->ptr,
                                                n_tokens, window, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention_prefill_raw launch");
}
static int attention_decode_batch_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    if (n_comp != 0 && ratio == 0) return 0;
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), logical_tier, "attn_sinks");
    if (!sinks) return 0;
    if (g_n_gpus == 1 && !use_comp_mask &&
        n_tokens >= 128u && head_dim == kTTHeadDim && n_head == 64u &&
        window == kTTRawWindow && ratio != 0u && n_comp <= 32768u &&
        n_raw >= n_tokens &&
        (uint64_t)n_raw <= (uint64_t)pos0 + n_tokens &&
        !g_cuda_no_window_attention && ds4_cuda_attn_tokentile_arch_ok()) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        cudaStream_t stream = cuda_decode_stream();
        if (cudaStreamIsCapturing(stream, &capture) == cudaSuccess &&
            capture == cudaStreamCaptureStatusNone) {
            const uint32_t n_tiles =
                (n_tokens + kTTTileTokens - 1u) / kTTTileTokens;
            const uint32_t rec_stride = n_comp;
            const uint32_t n_mirror_rows = n_tokens + kTTRawWindow - 1u;
            const uint32_t raw_before = n_raw - n_tokens;
            const uint32_t available_before = raw_before < pos0 ? raw_before : pos0;
            const uint32_t available_clamped =
                available_before < (kTTRawWindow - 1u)
                    ? available_before : (kTTRawWindow - 1u);
            const uint32_t raw_row_min =
                (kTTRawWindow - 1u) - available_clamped;
            const uint32_t first_raw_pos = (uint32_t)(
                (uint64_t)pos0 + n_tokens - n_raw);

            uint64_t off = 0;
            const uint64_t records_off = off;
            off = tt_align256_u64(
                off + (uint64_t)n_tiles * (rec_stride ? rec_stride : 1u) *
                      sizeof(int2));
            const uint64_t counts_off = off;
            off = tt_align256_u64(
                off + (uint64_t)n_tiles * sizeof(uint32_t));
            const uint64_t raw_mirror_off = off;
            off = tt_align256_u64(
                off + (uint64_t)n_mirror_rows * kTTHeadDim * sizeof(half));
            const uint64_t comp_mirror_off = off;
            off = tt_align256_u64(
                off + (uint64_t)n_comp * kTTHeadDim * sizeof(half));

            unsigned char *scratch = (unsigned char *)tt_scratch_ensure(
                off, "allocate dense token-tile attention scratch");
            if (scratch) {
                int2 *records = (int2 *)(scratch + records_off);
                uint32_t *counts = (uint32_t *)(scratch + counts_off);
                half *raw_mirror = (half *)(scratch + raw_mirror_off);
                half *comp_mirror = (half *)(scratch + comp_mirror_off);
                if (!cuda_ok(cudaFuncSetAttribute(
                    attention_tokentile_hmma_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)tt_TokentileSmemBudget<kTTStageRows, kTTG>::total),
                    "set dense token-tile HMMA shared-memory limit")) {
                    return 0;
                }

                attention_tokentile_dense_build_kernel
                    <<<n_tiles, kTTThreads, 0, stream>>>(
                        records, counts, pos0, n_tokens, ratio, n_comp,
                        rec_stride);
                if (!cuda_ok(cudaGetLastError(),
                             "launch dense token-tile record builder")) return 0;
                attention_tokentile_raw_mirror_kernel
                    <<<n_mirror_rows, 256, 0, stream>>>(
                        raw_mirror, (const float *)raw_kv->ptr, NULL, pos0,
                        n_tokens, raw_cap, raw_start, first_raw_pos,
                        raw_row_min, head_dim);
                if (!cuda_ok(cudaGetLastError(),
                             "launch dense token-tile raw mirror")) return 0;
                if (n_comp != 0u) {
                    attention_tokentile_comp_mirror_kernel
                        <<<n_comp, 256, 0, stream>>>(
                            comp_mirror, (const float *)comp_kv->ptr,
                            n_comp, head_dim);
                    if (!cuda_ok(cudaGetLastError(),
                                 "launch dense token-tile compressed mirror")) {
                        return 0;
                    }
                }
                const dim3 grid(n_tiles, n_head / kTTG, 1);
                attention_tokentile_hmma_kernel
                    <<<grid, kTTThreads,
                       tt_TokentileSmemBudget<kTTStageRows, kTTG>::total,
                       stream>>>(
                        (float *)heads->ptr, sinks, (const float *)q->ptr,
                        raw_mirror, comp_mirror, records, counts, rec_stride,
                        n_tokens, n_head, raw_row_min);
                return cuda_ok(cudaGetLastError(),
                               "launch dense token-tile HMMA attention");
            }
        } else {
            (void)cudaGetLastError();
        }
    }
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_comp_mask && head_dim == 512u &&
            !g_cuda_no_window_attention) {
            dim3 online_grid(n_tokens, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              window,
                                                                              ratio,
                                                                              n_head,
                                                                              head_dim);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        !g_cuda_no_window_attention &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   raw_cap,
                                                                   raw_start,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention decode window launch");
    }
    if (!use_comp_mask && n_tokens == 1u && head_dim == 512 &&
        g_cuda_decode_heads8_online &&
        !g_cuda_no_window_attention) {
        dim3 grid(1, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   raw_cap,
                                                                   raw_start,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention decode heads8 online batch launch");
    }
    const uint32_t score_lanes =
        g_cuda_decode_score4 ? 4u : (g_cuda_decode_score8 ? 8u : 0u);
    const uint32_t threads =
        n_tokens == 1u && head_dim == 512u && score_lanes == 0u &&
        !g_cuda_no_decode_value512 ? 512u : 256u;
    if (n_tokens == 1u) {
        int score_split_rc = attention_decode_score_split_launch(
                logical_tier, (float *)heads->ptr, sinks, (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio,
                n_head, head_dim, threads, NULL);
        if (score_split_rc == 1) {
            return cuda_ok(cudaGetLastError(), "attention exact score split batch launch");
        }
        if (score_split_rc < 0) return 0;
    }
    /* perf-02 split-KV opt-in (default OFF). Single-token decode only; multi-
     * token batch shapes already fill the grid and fall through unchanged.
     * S==1 / disabled / unhandled -> rc 0, fall through to the old kernel. */
    if (n_tokens == 1u && cuda_splitkv_decode_requested()) {
        int rc = attention_decode_splitkv_launch(
                logical_tier, (float *)heads->ptr, sinks, (const float *)q->ptr,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL, use_comp_mask,
                pos0, n_raw, raw_cap, raw_start, n_comp, window, ratio, n_head, head_dim);
        if (rc == 1) return cuda_ok(cudaGetLastError(), "attention decode splitkv batch launch");
        if (rc < 0) return 0;
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_decode_mixed_kernel<<<grid, threads>>>((float *)heads->ptr,
                                                     sinks,
                                                     (const float *)q->ptr,
                                                     (const float *)raw_kv->ptr,
                                                     n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                     use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                     use_comp_mask, n_tokens, pos0, n_raw, raw_cap,
                                                     raw_start, n_comp, window, ratio, n_head, head_dim,
                                                     score_lanes);
    return cuda_ok(cudaGetLastError(), "attention decode batch launch");
}

extern "C" int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
#ifdef DS4_CUDA_SPARK_ONLY
    if (n_tokens > 1u && head_dim == 512u && heads && q && raw_kv && model_map &&
        n_raw != 0u && raw_cap >= n_raw && raw_start < raw_cap &&
        sinks_offset <= model_size &&
        (uint64_t)n_head * sizeof(float) <= model_size - sinks_offset &&
        heads->bytes >= (uint64_t)n_tokens * n_head * head_dim * sizeof(float) &&
        q->bytes >= (uint64_t)n_tokens * n_head * head_dim * sizeof(float) &&
        raw_kv->bytes >= (uint64_t)raw_cap * DS4_SPARK_KV_ROW_BYTES) {
        const int logical_tier = ds4_tensor_device_idx(heads);
        const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float),
            logical_tier, "attn_sinks");
        if (!sinks) return 0;
        const int rc = spark_attention_unpack_batch_launch(
            logical_tier,
            (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const unsigned char *)raw_kv->ptr, NULL,
            n_tokens, pos0, n_raw, raw_cap, raw_start, 0u,
            window, 1u, n_head);
        return rc == 1;
    }
#endif
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, NULL, 0, NULL, 0, n_tokens, pos0,
                                      n_raw, raw_cap, raw_start, 0, window, 1,
                                      n_head, head_dim);
}

extern "C" int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_format,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_format == DS4_GPU_CACHE_SPARK_KV && n_tokens > 1u &&
        head_dim == 512u && !use_comp_mask && heads && q && raw_kv &&
        model_map && n_raw != 0u && raw_cap >= n_raw && raw_start < raw_cap &&
        (n_comp == 0u || comp_kv) && sinks_offset <= model_size &&
        (uint64_t)n_head * sizeof(float) <= model_size - sinks_offset &&
        heads->bytes >= (uint64_t)n_tokens * n_head * head_dim * sizeof(float) &&
        q->bytes >= (uint64_t)n_tokens * n_head * head_dim * sizeof(float) &&
        raw_kv->bytes >= (uint64_t)raw_cap * DS4_SPARK_KV_ROW_BYTES &&
        (n_comp == 0u || comp_kv->bytes >=
            (uint64_t)n_comp * DS4_SPARK_KV_ROW_BYTES)) {
        const int logical_tier = ds4_tensor_device_idx(heads);
        const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float),
            logical_tier, "attn_sinks");
        if (!sinks) return 0;
        const int rc = spark_attention_unpack_batch_launch(
            logical_tier,
            (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const unsigned char *)raw_kv->ptr,
            n_comp ? (const unsigned char *)comp_kv->ptr : NULL,
            n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp,
            window, ratio, n_head);
        return rc == 1;
    }
    if (comp_kv_format != DS4_GPU_CACHE_F32) return 0;
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, comp_kv, comp_kv_format, comp_mask, use_comp_mask,
                                      n_tokens, pos0, n_raw, raw_cap, raw_start,
                                      n_comp, window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_format,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !comp_kv || !topk || !model_map ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * DS4_SPARK_KV_ROW_BYTES ||
        comp_kv->bytes < (uint64_t)n_comp * DS4_SPARK_KV_ROW_BYTES ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) {
        return 0;
    }
    if (top_k > 512u) return 0;
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
        model_map, sinks_offset, (uint64_t)n_head * sizeof(float), logical_tier, "attn_sinks");
    if (!sinks) return 0;
    const int32_t *topk_ptr = (const int32_t *)topk->ptr;
    if (comp_kv_format == DS4_GPU_CACHE_SPARK_KV && n_tokens == 1u &&
        head_dim == 512u) {
        const int rc = spark_indexed_attention_unpack_launch(
            (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const unsigned char *)raw_kv->ptr,
            (const unsigned char *)comp_kv->ptr,
            topk_ptr, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
            window, ratio, n_head);
        if (rc == 1) {
            return cuda_ok(cudaGetLastError(), "Spark packed indexed attention launch");
        }
        return 0;
    }
    if (comp_kv_format == DS4_GPU_CACHE_SPARK_KV && n_tokens > 1u &&
        head_dim == 512u) {
        const int rc = spark_indexed_attention_unpack_batch_launch(
            logical_tier,
            (float *)heads->ptr, sinks, (const float *)q->ptr,
            (const unsigned char *)raw_kv->ptr,
            (const unsigned char *)comp_kv->ptr, topk_ptr,
            n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
            window, ratio, n_head);
        return rc == 1;
    }
    if (comp_kv_format != DS4_GPU_CACHE_F32) return 0;
    if (g_n_gpus == 1 && n_tokens >= 128u && head_dim == kTTHeadDim &&
        n_head == 64u && top_k == 512u && window == kTTRawWindow &&
        ratio != 0u && n_comp <= 32768u &&
        n_raw >= n_tokens &&
        (uint64_t)n_raw <= (uint64_t)pos0 + n_tokens &&
        ds4_cuda_attn_tokentile_arch_ok()) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        cudaStream_t stream = cuda_decode_stream();
        if (cudaStreamIsCapturing(stream, &capture) == cudaSuccess &&
            capture == cudaStreamCaptureStatusNone) {
            const uint32_t n_tiles =
                (n_tokens + kTTTileTokens - 1u) / kTTTileTokens;
            const uint32_t rec_stride =
                (kTTTileTokens * top_k) < n_comp
                    ? (kTTTileTokens * top_k) : n_comp;
            const uint32_t n_mirror_rows = n_tokens + kTTRawWindow - 1u;
            const uint32_t raw_before = n_raw - n_tokens;
            const uint32_t available_before = raw_before < pos0 ? raw_before : pos0;
            const uint32_t available_clamped =
                available_before < (kTTRawWindow - 1u)
                    ? available_before : (kTTRawWindow - 1u);
            const uint32_t raw_row_min =
                (kTTRawWindow - 1u) - available_clamped;
            const uint32_t first_raw_pos = (uint32_t)(
                (uint64_t)pos0 + n_tokens - n_raw);

            uint64_t off = 0;
            const uint64_t records_off = off;
            off = tt_align256_u64(
                off + (uint64_t)n_tiles * rec_stride * sizeof(int2));
            const uint64_t counts_off = off;
            off = tt_align256_u64(
                off + (uint64_t)n_tiles * sizeof(uint32_t));
            const uint64_t raw_mirror_off = off;
            off = tt_align256_u64(
                off + (uint64_t)n_mirror_rows * kTTHeadDim * sizeof(half));
            const uint64_t comp_mirror_off = off;
            off = tt_align256_u64(
                off + (uint64_t)n_comp * kTTHeadDim * sizeof(half));

            unsigned char *scratch = (unsigned char *)tt_scratch_ensure(
                off, "allocate token-tile attention scratch");
            if (scratch) {
                int2 *records = (int2 *)(scratch + records_off);
                uint32_t *counts = (uint32_t *)(scratch + counts_off);
                half *raw_mirror = (half *)(scratch + raw_mirror_off);
                half *comp_mirror = (half *)(scratch + comp_mirror_off);
                const uint32_t bitmap_words = (n_comp + 1u) >> 1u;
                const size_t bitmap_smem =
                    (size_t)bitmap_words * sizeof(uint32_t);
                /* The launch limit includes the kernel's static scan array,
                 * so opt in to the largest bitmap admitted by this path even
                 * when the dynamic bitmap alone is just under 48 KiB. */
                static int union_smem_attr_set = 0;
                if (!union_smem_attr_set) {
                    const int max_bitmap_smem =
                        (int)(((32768u + 1u) >> 1u) * sizeof(uint32_t));
                    if (!cuda_ok(cudaFuncSetAttribute(
                        attention_tokentile_union_build_kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        max_bitmap_smem),
                        "set token-tile union shared-memory limit")) {
                        return 0;
                    }
                    union_smem_attr_set = 1;
                }
                if (!cuda_ok(cudaFuncSetAttribute(
                    attention_tokentile_hmma_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)tt_TokentileSmemBudget<kTTStageRows, kTTG>::total),
                    "set token-tile HMMA shared-memory limit")) {
                    return 0;
                }

                attention_tokentile_union_build_kernel
                    <<<n_tiles, kTTThreads, bitmap_smem, stream>>>(
                        records, counts, topk_ptr, NULL, pos0, n_tokens,
                        top_k, ratio, n_comp, rec_stride);
                if (!cuda_ok(cudaGetLastError(),
                             "launch token-tile union builder")) return 0;
                attention_tokentile_raw_mirror_kernel
                    <<<n_mirror_rows, 256, 0, stream>>>(
                        raw_mirror, (const float *)raw_kv->ptr, NULL, pos0,
                        n_tokens, raw_cap, raw_start, first_raw_pos,
                        raw_row_min, head_dim);
                if (!cuda_ok(cudaGetLastError(),
                             "launch token-tile raw mirror")) return 0;
                attention_tokentile_comp_mirror_kernel
                    <<<n_comp, 256, 0, stream>>>(
                        comp_mirror, (const float *)comp_kv->ptr,
                        n_comp, head_dim);
                if (!cuda_ok(cudaGetLastError(),
                             "launch token-tile compressed mirror")) return 0;
                const dim3 grid(n_tiles, n_head / kTTG, 1);
                attention_tokentile_hmma_kernel
                    <<<grid, kTTThreads,
                       tt_TokentileSmemBudget<kTTStageRows, kTTG>::total,
                       stream>>>(
                        (float *)heads->ptr, sinks, (const float *)q->ptr,
                        raw_mirror, comp_mirror, records, counts, rec_stride,
                        n_tokens, n_head, raw_row_min);
                return cuda_ok(cudaGetLastError(),
                               "launch token-tile HMMA attention");
            }
        } else {
            (void)cudaGetLastError();
        }
    }
    if (n_tokens > 1u && top_k == 512u &&
        getenv("DS4_CUDA_NO_INDEXED_TOPK_SORT") == NULL) {
        const uint64_t sort_bytes = (uint64_t)n_tokens * top_k * sizeof(int32_t);
        int32_t *sorted = (int32_t *)cuda_tmp_alloc_on(logical_tier, sort_bytes, "indexed attention topk sort");
        if (!sorted) return 0;
        indexed_topk_sort_512_asc_kernel<<<n_tokens, 512>>>(sorted, topk_ptr, n_tokens);
        if (!cuda_ok(cudaGetLastError(), "indexed attention topk sort launch")) return 0;
        topk_ptr = sorted;
    }
    if (n_tokens > 1 && head_dim == 512 && top_k <= 512u &&
        getenv("DS4_CUDA_NO_INDEXED_HEADS8") == NULL) {
        if (getenv("DS4_CUDA_INDEXED_TWOPASS") == NULL) {
            dim3 grid(n_tokens, (n_head + 15u) / 16u, 1);
            attention_indexed_mixed_heads8_online_kernel<8, 16><<<grid, 512>>>((float *)heads->ptr,
                                                                               sinks,
                                                                               (const float *)q->ptr,
                                                                               (const float *)raw_kv->ptr,
                                                                               (const float *)comp_kv->ptr,
                                                                               topk_ptr,
                                                                               n_tokens,
                                                                               pos0,
                                                                               n_raw,
                                                                               raw_cap,
                                                                               raw_start,
                                                                               n_comp,
                                                                               top_k,
                                                                               window,
                                                                               ratio,
                                                                               n_head,
                                                                               head_dim);
            return cuda_ok(cudaGetLastError(), "attention indexed online launch");
        }
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_indexed_mixed_heads8_rb4_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                 sinks,
                                                                 (const float *)q->ptr,
                                                                 (const float *)raw_kv->ptr,
                                                                 (const float *)comp_kv->ptr,
                                                                 topk_ptr,
                                                                 n_tokens,
                                                                 pos0,
                                                                 n_raw,
                                                                 raw_cap,
                                                                 raw_start,
                                                                 n_comp,
                                                                 top_k,
                                                                 window,
                                                                 ratio,
                                                                 n_head,
                                                                 head_dim);
        return cuda_ok(cudaGetLastError(), "attention indexed heads8 launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_indexed_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  (const float *)comp_kv->ptr,
                                                  topk_ptr,
                                                  n_tokens,
                                                  pos0,
                                                  n_raw,
                                                  raw_cap,
                                                  raw_start,
                                                  n_comp,
                                                  top_k,
                                                  window,
                                                  ratio,
                                                  n_head,
                                                  head_dim);
    return cuda_ok(cudaGetLastError(), "attention indexed mixed launch");
}

static int attention_prefill_mixed_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 || ratio == 0 ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(heads);
    const float *sinks = (const float *)cuda_resolve_weight_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), logical_tier, "attn_sinks");
    if (!sinks) return 0;
    if (!use_comp_mask && g_n_gpus == 1 && n_tokens >= 128u &&
        head_dim == kTTHeadDim && n_head == 64u &&
        window == kTTRawWindow && ratio != 0u && n_comp <= 32768u &&
        !g_cuda_no_window_attention && ds4_cuda_attn_tokentile_arch_ok()) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(cuda_decode_stream(), &capture) == cudaSuccess &&
            capture == cudaStreamCaptureStatusNone) {
            return attention_decode_batch_launch(
                heads, model_map, model_size, sinks_offset, q, raw_kv,
                comp_kv, 0, NULL, 0, n_tokens, 0, n_tokens, n_tokens, 0,
                n_comp, window, ratio, n_head, head_dim);
        }
        (void)cudaGetLastError();
    }
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION") == NULL) {
        const uint32_t n_keys = n_tokens + n_comp;
        const uint64_t kv_count = (uint64_t)n_keys * head_dim;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t kv_bytes = kv_count * sizeof(float);
        const uint64_t score_offset = (kv_bytes + 255u) & ~255ull;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = score_offset + ((score_bytes + 255u) & ~255ull);
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc_on(logical_tier, tmp_bytes, "attention mixed cublas");
        if (!tmp) return 0;
        float *kv = tmp;
        float *scores = (float *)((char *)tmp + score_offset);
        float *out_tmp = (float *)((char *)tmp + out_offset);
        attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256>>>(
                kv,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                n_tokens,
                n_comp,
                head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention mixed kv pack launch")) return 0;
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(cuda_cublas_for_tier(logical_tier),
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      kv,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention mixed score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_mixed_softmax_kernel<<<sgrid, 256>>>(
                scores,
                sinks,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                use_comp_mask,
                n_tokens,
                n_comp,
                window,
                ratio,
                n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention mixed softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(cuda_cublas_for_tier(logical_tier),
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       kv,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention mixed value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                  use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                  use_comp_mask, n_tokens, n_comp, window, ratio,
                                                  n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention prefill mixed launch");
}

extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, NULL, 0, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, comp_mask, 1, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}
extern "C" int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    (void)group_tmp;
    (void)low_tmp;
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t blocks_b = (low_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    const uint64_t out_b_bytes = out_dim * blocks_b * 34;
    if (out_a_offset > model_size || out_b_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        out_b_bytes > model_size - out_b_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out);
    const int physical_device =
        (g_n_gpus > 1 && logical_tier >= 0 && logical_tier < g_n_gpus)
            ? g_gpu[logical_tier].device_id : 0;
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_resolve_weight_ptr(model_map, out_a_offset, out_a_bytes, logical_tier, "attn_out_a"));
    const unsigned char *out_b = reinterpret_cast<const unsigned char *>(
            cuda_resolve_weight_ptr(model_map, out_b_offset, out_b_bytes, logical_tier, "attn_out_b"));
    if (!out_a || !out_b) return 0;

    const uint32_t profile = getenv("DS4_CUDA_ATTN_OUTPUT_PROFILE") != NULL;
    cudaEvent_t prof_ev[3] = {NULL, NULL, NULL};
    if (profile) {
        for (uint32_t i = 0; i < 3u; i++) {
            if (cudaEventCreate(&prof_ev[i]) != cudaSuccess) {
                for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(prof_ev[j]);
                memset(prof_ev, 0, sizeof(prof_ev));
                break;
            }
        }
        if (prof_ev[0]) (void)cudaEventRecord(prof_ev[0], 0);
    }

    const __half *out_a_f16 = NULL;
    uint32_t out_a_cublas_min_tokens = 2u;
    const char *out_a_min_env = getenv("DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN");
    if (out_a_min_env && out_a_min_env[0]) {
        char *endp = NULL;
        long v = strtol(out_a_min_env, &endp, 10);
        if (endp != out_a_min_env && v > 1 && v < 4096) out_a_cublas_min_tokens = (uint32_t)v;
    }
    if (!g_quality_mode &&
        g_cublas_ready &&
        n_tokens >= out_a_cublas_min_tokens &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A") == NULL) {
        out_a_f16 = cuda_q8_f16_ptr(model_map, out_a_offset, out_a_bytes, group_dim, low_dim, physical_device, "attn_output_a");
    }
    if (out_a_f16) {
        const uint64_t heads_h_count = (uint64_t)n_groups * n_tokens * group_dim;
        const uint64_t low_tmp_count = (uint64_t)n_groups * n_tokens * rank;
        const uint64_t heads_h_bytes = heads_h_count * sizeof(__half);
        const uint64_t low_tmp_offset = (heads_h_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = low_tmp_offset + low_tmp_count * sizeof(float);
        void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "attention output a cublas");
        if (!tmp) return 0;
        __half *heads_h = (__half *)tmp;
        float *low_packed = (float *)((char *)tmp + low_tmp_offset);
        attention_pack_group_heads_f16_kernel<<<(heads_h_count + 255) / 256, 256, 0, cuda_decode_stream()>>>(
                heads_h,
                (const float *)heads->ptr,
                n_tokens,
                n_groups,
                group_dim);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a pack launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmStridedBatchedEx(cuda_cublas_for_tier(logical_tier),
                                                       CUBLAS_OP_T,
                                                       CUBLAS_OP_N,
                                                       (int)rank,
                                                       (int)n_tokens,
                                                       (int)group_dim,
                                                       &alpha,
                                                       out_a_f16,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)rank * group_dim,
                                                       heads_h,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)n_tokens * group_dim,
                                                       &beta,
                                                       low_packed,
                                                       CUDA_R_32F,
                                                       (int)rank,
                                                       (long long)rank * n_tokens,
                                                       (int)n_groups,
                                                       CUDA_R_32F,
                                                       CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "attention output a gemm")) return 0;
        attention_unpack_group_low_kernel<<<(low_tmp_count + 255) / 256, 256, 0, cuda_decode_stream()>>>(
                (float *)low->ptr,
                low_packed,
                n_tokens,
                n_groups,
                rank);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a unpack launch")) return 0;
    } else {
        const uint64_t x_rows = (uint64_t)n_tokens * n_groups;
        const uint64_t xq_bytes = x_rows * blocks_a * 32u;
        const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
        const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
        void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "attention output a q8 prequant");
        if (!tmp) return 0;
        int8_t *xq = (int8_t *)tmp;
        float *xscale = (float *)((char *)tmp + scale_offset);
        const int use_dp4a = cuda_q8_use_dp4a();
        dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
        quantize_q8_0_f32_kernel<<<qgrid, 32, 0, cuda_decode_stream()>>>(xq,
                                                xscale,
                                                (const float *)heads->ptr,
                                                group_dim,
                                                blocks_a);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a prequant launch")) return 0;
        int grouped_mma_done = 0;
        if (n_tokens >= 8u) {
            /* One mma launch per group: T=32 matches the warp tree of the
             * grouped reference kernels (multi-term slots for blocks > 32). */
            grouped_mma_done = 1;
            for (uint32_t g = 0; g < n_groups && grouped_mma_done == 1; g++) {
                const int rc = cuda_q8_mma_try_launch(
                        (float *)low->ptr + (uint64_t)g * rank,
                        reinterpret_cast<const unsigned char *>(out_a) +
                            (uint64_t)g * rank * blocks_a * 34u,
                        xq + (uint64_t)g * blocks_a * 32u,
                        xscale + (uint64_t)g * blocks_a,
                        group_dim, rank, n_tokens, blocks_a,
                        (uint64_t)n_groups * blocks_a, low_dim, 32u);
                if (rc < 0) return 0;
                if (rc == 0) grouped_mma_done = 0;
            }
        }
        if (grouped_mma_done) {
            /* handled */
        } else if (getenv("DS4_CUDA_NO_ATTN_A_TOK2") == NULL && n_tokens >= 2u) {
            dim3 grid_a(((unsigned)low_dim + 7u) / 8u, ((unsigned)n_tokens + 1u) / 2u, 1);
            grouped_q8_0_a_preq_warp8_tok2_kernel<<<grid_a, 512, 0, cuda_decode_stream()>>>((float *)low->ptr,
                                                                   out_a,
                                                                   xq,
                                                                   xscale,
                                                                   group_dim,
                                                                   rank,
                                                                   n_groups,
                                                                   n_tokens,
                                                                   blocks_a,
                                                                   use_dp4a);
        } else {
            dim3 grid_a(((unsigned)low_dim + 7u) / 8u, (unsigned)n_tokens, 1);
            grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256, 0, cuda_decode_stream()>>>((float *)low->ptr,
                                                              out_a,
                                                              xq,
                                                              xscale,
                                                              group_dim,
                                                              rank,
                                                              n_groups,
                                                              n_tokens,
                                                              blocks_a,
                                                              use_dp4a);
        }
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a preq launch")) return 0;
    }

    if (prof_ev[1]) (void)cudaEventRecord(prof_ev[1], 0);
    (void)out_b;
    int ok = cuda_matmul_q8_0_tensor_labeled(out,
                                             model_map,
                                             model_size,
                                             out_b_offset,
                                             low_dim,
                                             out_dim,
                                             low,
                                             n_tokens,
                                             "attn_output_b");
    if (prof_ev[2]) {
        (void)cudaEventRecord(prof_ev[2], 0);
        if (cudaEventSynchronize(prof_ev[2]) == cudaSuccess) {
            float ms_a = 0.0f, ms_b = 0.0f, ms_total = 0.0f;
            (void)cudaEventElapsedTime(&ms_a, prof_ev[0], prof_ev[1]);
            (void)cudaEventElapsedTime(&ms_b, prof_ev[1], prof_ev[2]);
            (void)cudaEventElapsedTime(&ms_total, prof_ev[0], prof_ev[2]);
            fprintf(stderr,
                    "ds4: CUDA attention output profile tokens=%u groups=%u group_dim=%llu rank=%llu low=%llu out=%llu A=%.3f B=%.3f total=%.3f ms\n",
                    n_tokens,
                    n_groups,
                    (unsigned long long)group_dim,
                    (unsigned long long)rank,
                    (unsigned long long)low_dim,
                    (unsigned long long)out_dim,
                    ms_a,
                    ms_b,
                    ms_total);
        }
        for (uint32_t i = 0; i < 3u; i++) (void)cudaEventDestroy(prof_ev[i]);
    }
    return ok;
}
extern "C" int ds4_gpu_attention_output_low_q8_rows_exact_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads,
        uint32_t                n_rows) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 ||
        n_groups_total == 0 || group_cnt == 0 ||
        group0 > n_groups_total || group_cnt > n_groups_total - group0 ||
        n_rows == 0 || (uint64_t)n_rows * group_cnt > 65535u) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)group_cnt * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t row_a_bytes = blocks_a * 34u;
    const uint64_t a_offset =
        out_a_offset + (uint64_t)group0 * rank * row_a_bytes;
    const uint64_t out_a_bytes = low_dim * row_a_bytes;
    if (a_offset < out_a_offset || a_offset > model_size ||
        out_a_bytes > model_size - a_offset ||
        heads->bytes < (uint64_t)n_rows * n_groups_total * group_dim *
            sizeof(float) ||
        low->bytes < (uint64_t)n_rows * low_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(low);
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_resolve_weight_ptr(model_map, a_offset, out_a_bytes,
                                    logical_tier, "attn_out_a_rows"));
    if (!out_a) return 0;

    const uint64_t x_rows = (uint64_t)n_rows * group_cnt;
    const uint64_t xq_bytes = x_rows * blocks_a * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "attention output low q8 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
    quantize_q8_0_group_slice_rows_kernel<<<qgrid, 32, 0, cuda_decode_stream()>>>(
            xq,
            xscale,
            (const float *)heads->ptr,
            group_dim,
            blocks_a,
            n_groups_total,
            group0,
            group_cnt);
    if (!cuda_ok(cudaGetLastError(),
                 "attention_output_low_q8 rows prequant launch")) return 0;
    dim3 grid_a(((unsigned)low_dim + 7u) / 8u, n_rows, 1u);
    grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256, 0, cuda_decode_stream()>>>((float *)low->ptr,
                                                      out_a,
                                                      xq,
                                                      xscale,
                                                      group_dim,
                                                      rank,
                                                      group_cnt,
                                                      n_rows,
                                                      blocks_a,
                                                      use_dp4a);
    return cuda_ok(cudaGetLastError(),
                   "attention_output_low_q8 rows launch");
}

extern "C" int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    return ds4_gpu_attention_output_low_q8_rows_exact_tensor(
            low, model_map, model_size, out_a_offset, group_dim, rank,
            n_groups, 0u, n_groups, heads, 1u);
}

extern "C" int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads) {
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups_total == 0 ||
        group_cnt == 0 || group0 > n_groups_total ||
        group_cnt > n_groups_total - group0 || out_dim == 0) {
        return 0;
    }
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    const uint64_t row_a_bytes = blocks_a * 34u;
    const uint64_t low_dim_total = (uint64_t)n_groups_total * rank;
    const uint64_t k_off = (uint64_t)group0 * rank;
    const uint64_t k_cnt = (uint64_t)group_cnt * rank;
    if ((k_off % 32u) != 0 || (k_cnt % 32u) != 0) return 0;
    if (heads->bytes < (uint64_t)(group0 + group_cnt) * group_dim * sizeof(float) ||
        low->bytes < k_cnt * sizeof(float) ||
        out->bytes < out_dim * sizeof(float)) {
        return 0;
    }

    ds4_gpu_tensor heads_slice = *heads;
    heads_slice.ptr = (char *)heads->ptr + (uint64_t)group0 * group_dim * sizeof(float);
    heads_slice.bytes = (uint64_t)group_cnt * group_dim * sizeof(float);
    heads_slice.owner = 0;

    const uint64_t a_off = out_a_offset + (uint64_t)group0 * rank * row_a_bytes;
    return ds4_gpu_attention_output_low_q8_tensor(low,
                                                  model_map,
                                                  model_size,
                                                  a_off,
                                                  group_dim,
                                                  rank,
                                                  group_cnt,
                                                  &heads_slice) &&
           ds4_gpu_matmul_q8_0_kslice_rows_tensor(out,
                                             model_map,
                                             model_size,
                                             out_b_offset,
                                             low_dim_total,
                                             out_dim,
                                             k_off,
                                             k_cnt,
                                             low,
                                             1);
}
extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!out || !gate || !up ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * sizeof(float) ||
        up->bytes < (uint64_t)n * sizeof(float)) return 0;
    swiglu_kernel<<<(n + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
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
    if (getenv("DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR") == NULL) {
        return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                                 model_map, model_size,
                                                 gate_offset, up_offset,
                                                 in_dim, out_dim, out_dim,
                                                 x, 1) &&
               ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
    }
    return ds4_gpu_matmul_q8_0_tensor(gate, model_map, model_size,
                                        gate_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_matmul_q8_0_tensor(up, model_map, model_size,
                                        up_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
}

extern "C" int ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *prequant,
        uint32_t                expert_split,
        bool                    home_rank) {
    if (!mid || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        x->bytes < in_dim * sizeof(float) ||
        mid->bytes < out_dim * sizeof(float) ||
        (selected && (selected->bytes < 6u * sizeof(int32_t) ||
                      expert_split == 0u))) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (gate_offset > model_size || up_offset > model_size ||
        out_dim > UINT64_MAX / (blocks * 34u)) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (weight_bytes > model_size - gate_offset ||
        weight_bytes > model_size - up_offset) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(x);
    if (logical_tier < 0 || logical_tier >= g_n_gpus) return 0;
    if (selected && ds4_tensor_device_idx(selected) != logical_tier) return 0;
    if (prequant && ds4_tensor_device_idx(prequant) != logical_tier) return 0;
    const int mid_tier = ds4_tensor_device_idx(mid);
    if (mid_tier != logical_tier && !g_gpu_peer_ok[logical_tier][mid_tier]) {
        return 0;
    }
    const char *gate_w = cuda_resolve_weight_ptr(
            model_map, gate_offset, weight_bytes, logical_tier,
            "shared_mid_gate_exact");
    const char *up_w = cuda_resolve_weight_ptr(
            model_map, up_offset, weight_bytes, logical_tier,
            "shared_mid_up_exact");
    if (!gate_w || !up_w) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    int8_t *xq;
    float *xscale;
    if (prequant) {
        if (prequant->bytes < tmp_bytes) return 0;
        xq = (int8_t *)prequant->ptr;
        xscale = (float *)((char *)prequant->ptr + scale_offset);
    } else {
        void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes,
                                      "shared mid q8 exact prequant");
        if (!tmp) return 0;
        xq = (int8_t *)tmp;
        xscale = (float *)((char *)tmp + scale_offset);
        quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32, 0, cuda_decode_stream()>>>(
                xq, xscale, (const float *)x->ptr, in_dim, blocks);
        if (!cuda_ok(cudaGetLastError(),
                     "shared mid q8 exact quantize launch")) {
            return 0;
        }
    }
    shared_mid_q8_0_preq_warp8_exact_kernel<<<
            ((unsigned)out_dim + 7u) / 8u, 256, 0, cuda_decode_stream()>>>(
            (float *)mid->ptr,
            (const unsigned char *)gate_w,
            (const unsigned char *)up_w,
            xq,
            xscale,
            in_dim,
            out_dim,
            blocks,
            clamp,
            selected ? (const int32_t *)selected->ptr : NULL,
            expert_split,
            home_rank,
            cuda_q8_use_dp4a());
    return cuda_ok(cudaGetLastError(), "shared mid q8 exact launch");
}
extern "C" int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n) {
    if (!out || !a || !b ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float)) return 0;
    add_kernel<<<(n + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "add launch");
}

extern "C" int ds4_gpu_add_xdev_tensor(ds4_gpu_tensor *out,
                                        const ds4_gpu_tensor *local,
                                        const ds4_gpu_tensor *remote,
                                        ds4_gpu_tensor *remote_tmp,
                                        uint32_t n) {
    if (!out || !local || !remote ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        local->bytes < (uint64_t)n * sizeof(float) ||
        remote->bytes < (uint64_t)n * sizeof(float)) return 0;
    if (n == 0) return 1;

    const int od = ds4_tensor_device_idx(out);
    const int ld = ds4_tensor_device_idx(local);
    const int rd = ds4_tensor_device_idx(remote);
    if (od != ld) return 0;

    const ds4_gpu_tensor *rhs = remote;
    if (rd != od) {
        if (!remote_tmp ||
            remote_tmp->bytes < (uint64_t)n * sizeof(float) ||
            ds4_tensor_device_idx(remote_tmp) != od) return 0;
        if (!ds4_gpu_tensor_copy_xdev(remote_tmp, remote,
                                      (uint64_t)n * sizeof(float))) return 0;
        rhs = remote_tmp;
    }

    int ok = 0;
    WITH_DEVICE(g_gpu[od].device_id) {
        cudaStream_t s = (cudaStream_t)g_gpu[od].stream;
        add_kernel<<<(n + 255u) / 256u, 256, 0, s>>>(
                (float *)out->ptr,
                (const float *)local->ptr,
                (const float *)rhs->ptr,
                n);
        ok = cuda_ok(cudaGetLastError(), "xdev add launch");
        cudaEvent_t e = (cudaEvent_t)g_gpu[od].boundary_event;
        if (ok) ok = cuda_ok(cudaEventRecord(e, s), "xdev add event record");
        if (ok) ok = cuda_ok(cudaStreamWaitEvent(0, e, 0), "xdev add default wait");
        if (ok && g_xdev_sync_debug) {
            ok = cuda_ok(cudaStreamSynchronize(s), "xdev add sync");
        }
    }
    return ok;
}
extern "C" int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;
    const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
    const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
    if (x->bytes < x_bytes || directions->bytes < dir_bytes) return 0;

    uint32_t nth = 256u;
    while (nth > width && nth > 1u) nth >>= 1;
    directional_steering_project_kernel<<<rows, nth>>>(
            (float *)x->ptr,
            (const float *)directions->ptr,
            layer,
            width,
            rows,
            scale);
    return cuda_ok(cudaGetLastError(), "directional steering launch");
}
extern "C" int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits) {
    if (!selected || !weights || !probs || !logits || !model_map || n_expert_groups > 1u || n_group_used > 0u) return 0;
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    int32_t tok = (int32_t)token;
    int ok = 1;
    const float *bias = NULL;
    const int32_t *hash = NULL;
    const int logical_tier = ds4_tensor_device_idx(selected);
    if (ok && has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) ok = 0;
        else bias = (const float *)cuda_resolve_weight_ptr(model_map, bias_offset, 256u * sizeof(float), logical_tier, "router_bias");
        if (!bias) ok = 0;
    }
    if (ok && hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) ok = 0;
        else hash = (const int32_t *)cuda_resolve_weight_ptr(model_map, hash_offset, hash_bytes, logical_tier, "router_hash");
        if (!hash) ok = 0;
    }
    if (ok) {
        if (getenv("DS4_CUDA_NO_WARP_ROUTER_SELECT") == NULL &&
            getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
            dim3 block(32, 4, 1);
            router_select_warp_topk_kernel<<<1, block, 0, cuda_decode_stream()>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                         bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                         has_bias && !hash_mode, hash_mode);
        } else if (getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
            router_select_parallel_kernel<<<1, 256, 0, cuda_decode_stream()>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                      bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                      has_bias && !hash_mode, hash_mode);
        } else {
            router_select_kernel<<<1, 1, 0, cuda_decode_stream()>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                          bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                          has_bias && !hash_mode, hash_mode);
        }
        ok = cuda_ok(cudaGetLastError(), "router_select launch");
    }
    return ok;
}
extern "C" int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens) {
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !model_map || n_tokens == 0 ||
        n_expert_groups > 1u || n_group_used > 0u ||
        logits->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * 6u * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * 6u * sizeof(float)) {
        return 0;
    }
    const float *bias = NULL;
    const int32_t *hash = NULL;
    const int logical_tier = ds4_tensor_device_idx(selected);
    if (has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) return 0;
        bias = (const float *)cuda_resolve_weight_ptr(model_map, bias_offset, 256u * sizeof(float), logical_tier, "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) return 0;
        hash = (const int32_t *)cuda_resolve_weight_ptr(model_map, hash_offset, hash_bytes, logical_tier, "router_hash");
        if (!hash) return 0;
    }
    if (getenv("DS4_CUDA_NO_WARP_ROUTER_SELECT") == NULL &&
        getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
        dim3 block(32, 4, 1);
        router_select_warp_topk_kernel<<<(n_tokens + 3u) / 4u, block>>>((int32_t *)selected->ptr,
                                                                        (float *)weights->ptr,
                                                                        (float *)probs->ptr,
                                                                        bias,
                                                                        hash,
                                                                        (const float *)logits->ptr,
                                                                        (const int32_t *)tokens->ptr,
                                                                        0,
                                                                        hash_rows,
                                                                        n_tokens,
                                                                        has_bias && !hash_mode,
                                                                        hash_mode);
    } else if (getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
        router_select_parallel_kernel<<<n_tokens, 256>>>((int32_t *)selected->ptr,
                                                         (float *)weights->ptr,
                                                         (float *)probs->ptr,
                                                         bias,
                                                         hash,
                                                         (const float *)logits->ptr,
                                                         (const int32_t *)tokens->ptr,
                                                         0,
                                                         hash_rows,
                                                         n_tokens,
                                                         has_bias && !hash_mode,
                                                         hash_mode);
    } else {
        router_select_kernel<<<n_tokens, 1>>>((int32_t *)selected->ptr,
                                              (float *)weights->ptr,
                                              (float *)probs->ptr,
                                              bias,
                                              hash,
                                              (const float *)logits->ptr,
                                              (const int32_t *)tokens->ptr,
                                              0,
                                              hash_rows,
                                              n_tokens,
                                              has_bias && !hash_mode,
                                              hash_mode);
    }
    return cuda_ok(cudaGetLastError(), "router_select launch");
}

__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid, uint32_t sign, const int8_t *q8, int32_t acc) {
    const uint32_t signs = dev_unpack_iq2_signs(sign);
    const int32_t sm0 = __vcmpne4(signs & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(signs & 0x80402010u, 0);
    const int32_t g0 = __vsub4((int32_t)(uint32_t)grid ^ sm0, sm0);
    const int32_t g1 = __vsub4((int32_t)(uint32_t)(grid >> 32) ^ sm1, sm1);
    acc = __dp4a(g0, *(const int32_t *)(q8 + 0), acc);
    acc = __dp4a(g1, *(const int32_t *)(q8 + 4), acc);
    return acc;
}

__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static int32_t dev_dot_iq2_pair_16(uint8_t grid0, uint32_t sign0, uint8_t grid1, uint32_t sign1, const int8_t *q8) {
    int32_t sum = 0;
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid0], cuda_ksigns_iq2xs[sign0], q8, sum);
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid1], cuda_ksigns_iq2xs[sign1], q8 + 8, sum);
    return sum;
}

__device__ __forceinline__ static void dev_iq2_i8x8_lut(
        const uint64_t *grid,
        const uint8_t *signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1) {
    const uint32_t s = dev_unpack_iq2_signs(signs[sign_idx]);
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
}

__device__ static float dev_dot_iq2_xxs_q8_K_block_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

__device__ static float dev_dot_iq2_xxs_q8_K_block(const cuda_block_iq2_xxs *x, const cuda_block_q8_K *y) {
    const float d = dev_f16_to_f32(x->d) * y->d;
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        int32_t sumi = 0;
        sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8);
        q8 += 16;
        sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8);
        q8 += 16;
        bsum += sumi * (int32_t)ls;
    }
    return 0.125f * d * (float)bsum;
}

__device__ static void dev_dot_iq2_xxs_q8_K_block8_deq_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8],
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        for (uint32_t p = 0; p < n; p++) {
            const int8_t *q = q8[p] + ib32 * 32;
            int32_t sumi = 0;
            sumi = __dp4a(w[0], *(const int32_t *)(q + 0),  sumi);
            sumi = __dp4a(w[1], *(const int32_t *)(q + 4),  sumi);
            sumi = __dp4a(w[2], *(const int32_t *)(q + 8),  sumi);
            sumi = __dp4a(w[3], *(const int32_t *)(q + 12), sumi);
            sumi = __dp4a(w[4], *(const int32_t *)(q + 16), sumi);
            sumi = __dp4a(w[5], *(const int32_t *)(q + 20), sumi);
            sumi = __dp4a(w[6], *(const int32_t *)(q + 24), sumi);
            sumi = __dp4a(w[7], *(const int32_t *)(q + 28), sumi);
            bsum[p] += sumi * ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_dot_iq2_xxs_q8_K_block4(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[4] = {0, 0, 0, 0};
    const int8_t *q8[4] = {
        y0 ? y0->qs : NULL,
        y1 ? y1->qs : NULL,
        y2 ? y2->qs : NULL,
        y3 ? y3->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static DS4_CUDA_UNUSED void dev_dot_iq2_xxs_q8_K_block8(
        const cuda_block_iq2_xxs *x,
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
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *x, const cuda_block_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum = 0;
    int summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}
