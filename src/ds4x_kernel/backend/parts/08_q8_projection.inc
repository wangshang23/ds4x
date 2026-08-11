/* GLM opt-in: batched q8_0 matmuls with blocks > 32 may run as a
 * streaming dequant-to-f16 GEMM (exact-q8 native kernels only cover
 * blocks <= 32). Never enabled on DS4 paths, keeping them byte-stable. */
static int g_q8_dequant_gemm_enabled = 0;
extern "C" void ds4_gpu_enable_q8_dequant_gemm(void) {
    g_q8_dequant_gemm_enabled = 1;
}

__global__ static void q8_0_dequant_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t total_blocks,
        uint32_t blocks_per_row,
        uint32_t in_dim) {
    /* Two threads per q8_0 block; each converts 16 values with half2
     * stores so a warp writes 512B contiguously per block pair. */
    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t b = tid >> 1;
    if (b >= total_blocks) return;
    const uint32_t half_idx = (uint32_t)tid & 1u;
    const unsigned char *blk = w + b * 34u;
    const float d = __half2float(*(const __half *)blk);
    const int8_t *q = (const int8_t *)(blk + 2) + half_idx * 16u;
    const uint64_t row = b / blocks_per_row;
    const uint32_t col = (uint32_t)(b - row * blocks_per_row) * 32u +
                         half_idx * 16u;
    __half2 *dst = (__half2 *)(out + row * in_dim + col);
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        dst[k] = __floats2half2_rn(d * (float)q[2 * k],
                                   d * (float)q[2 * k + 1]);
    }
}

static int cuda_matmul_q8_0_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map) return 0;
    uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    uint64_t weight_bytes = out_dim * blocks * 34;
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const int physical_device =
        (g_n_gpus > 1 && logical_tier >= 0 && logical_tier < g_n_gpus)
            ? g_gpu[logical_tier].device_id : 0;
    const uint64_t aligned_bytes =
        (in_dim % 1024u) == 0 && (out_dim % 128u) == 0
            ? ds4_mmq_q8_0_aligned_bytes((int)out_dim, (int)in_dim)
            : 0;
    const char *aligned = aligned_bytes && cuda_aligned_q8_enabled()
        ? cuda_derived_weight_ptr(
              model_map, weight_offset, weight_bytes,
              CUDA_DERIVED_Q8_0_ALIGNED_DENSE,
              in_dim, out_dim, 1u, aligned_bytes)
        : NULL;
    if (aligned && n_tok == 1u) {
        const int rc = ds4_mmq_q8_0_aligned_dense_vec(
            aligned, (const float *)x->ptr, (float *)out->ptr,
            (int)out_dim, 1, (int)in_dim, cuda_decode_stream());
        if (rc == 0) return 1;
    }
    if (aligned && n_tok >= 512u && out_dim >= 2048u &&
        in_dim <= 4096u && cuda_use_mmq()) {
        const char *d2r = getenv("DS4_MMQ_DENSE_D2R");
        if (!d2r || strcmp(d2r, "0") != 0) {
            const int rc = ds4_mmq_q8_0_dense_d2r(
                aligned, (const float *)x->ptr, (float *)out->ptr,
                (int)out_dim, (int)n_tok, (int)in_dim, (cudaStream_t)0);
            if (rc == 0) {
                static int logged = 0;
                if (!logged) {
                    logged = 1;
                    fprintf(stderr,
                            "ds4: dense Q8 prefill using aligned D2R\n");
                }
                return 1;
            }
            fprintf(stderr,
                    "ds4: aligned dense Q8 D2R returned %d "
                    "(label='%s' M=%llu N=%llu K=%llu); falling back\n",
                    rc, label ? label : "",
                    (unsigned long long)out_dim,
                    (unsigned long long)n_tok,
                    (unsigned long long)in_dim);
        }
    }
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes, logical_tier, "q8_0");
    if (!wptr) return 0;
    /* mmq fused-dequant-matmul prefill tier (ported from the Entrpi/ds4
     * fork).  Layout-compatible drop-in for the cuBLAS+dequant pipeline
     * below: mmq's [out_dim, n_tok] column-major output flattens to
     * [n_tok, out_dim] row-major, exactly what ds4 stores in out->ptr,
     * and the Q8_0 weight is already in mmq's expected row-major block
     * layout (the GGUF on-disk format).  K must be a multiple of 256;
     * every Q8_0 weight in V4 Flash satisfies this, odd shapes fall
     * through to the legacy paths. */
    if (n_tok > 1 && (in_dim % 256u) == 0 && cuda_use_mmq()) {
        int rc = ds4_mmq_q8_0_dense(wptr, (const float *)x->ptr, (float *)out->ptr,
                                    (int)out_dim, (int)n_tok, (int)in_dim,
                                    (cudaStream_t)0);
        if (rc == 0) return 1;
        fprintf(stderr, "ds4: ds4_mmq_q8_0_dense returned %d (label='%s' in=%llu out=%llu n_tok=%llu); falling back\n",
                rc, label ? label : "", (unsigned long long)in_dim,
                (unsigned long long)out_dim, (unsigned long long)n_tok);
    }
    if (g_cublas_ready && n_tok > 1) {
        const float *w_f32 = cuda_q8_f32_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, physical_device, label);
        if (w_f32) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasSgemm(cuda_cublas_for_tier(logical_tier),
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            (int)out_dim,
                                            (int)n_tok,
                                            (int)in_dim,
                                            &alpha,
                                            w_f32,
                                            (int)in_dim,
                                            (const float *)x->ptr,
                                            (int)in_dim,
                                            &beta,
                                            (float *)out->ptr,
                                            (int)out_dim);
            return cublas_ok(st, "q8 fp32 matmul");
        }
        const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, physical_device, label);
        if (w_f16) {
            const uint64_t xh_count = n_tok * in_dim;
            __half *xh = (__half *)cuda_tmp_alloc_on(logical_tier, xh_count * sizeof(__half), "q8 f16 gemm activations");
            if (!xh) return 0;
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256, 0, cuda_decode_stream()>>>(xh, (const float *)x->ptr, xh_count);
            if (!cuda_ok(cudaGetLastError(), "q8 f16 activation convert launch")) return 0;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(cuda_cublas_for_tier(logical_tier),
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             (int)out_dim,
                                             (int)n_tok,
                                             (int)in_dim,
                                             &alpha,
                                             w_f16,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             xh,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             &beta,
                                             out->ptr,
                                             CUDA_R_32F,
                                             (int)out_dim,
                                             CUDA_R_32F,
                                             CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) return 1;
            fprintf(stderr, "ds4: cuBLAS q8 f16 matmul failed: status %d\n", (int)st);
            cuda_q8_f16_cache_disable_after_failure("cuBLAS f16 matmul failure",
                                                    in_dim * out_dim * sizeof(__half));
            /* The F16 expansion cache is only an optimization.  If cuBLAS
             * rejects the cached path under memory pressure, retry the same
             * operation through the native Q8 kernels below. */
        }
    }
    if (g_q8_dequant_gemm_enabled && g_cublas_ready &&
        n_tok >= 128u && blocks > 32u && (in_dim & 31u) == 0u) {
        /* Streaming dequant + f16 GEMM: the exact-q8 batched kernels only
         * cover blocks <= 32 (DS4 TP shard widths); the per-token fallback
         * re-reads the full weight per token (~30x the bytes at GLM dims).
         * Scratch layout: [w_f16][x_f16] in one arena grab. */
        const uint64_t wh_bytes = in_dim * out_dim * sizeof(__half);
        const uint64_t xh_off = (wh_bytes + 255u) & ~255ull;
        const uint64_t oo_off =
            (xh_off + n_tok * in_dim * sizeof(__half) + 255u) & ~255ull;
        const uint64_t gemm_tmp = oo_off + n_tok * out_dim * sizeof(float);
        /* Scratch must live on the EXECUTING device: logical_tier is the
         * out tensor's tier (0 for GLM graph buffers), and a GEMM reading
         * its staged weights across PCIe costs ~20ms instead of ~0.1ms. */
        int exec_tier = logical_tier;
        {
            int cur_dev = -1;
            if (cudaGetDevice(&cur_dev) == cudaSuccess) {
                for (int t = 0; t < g_n_gpus; t++) {
                    if (g_gpu[t].device_id == cur_dev) { exec_tier = t; break; }
                }
            }
        }
        void *tmp16 = cuda_tmp_alloc_on(exec_tier, gemm_tmp, "q8 dequant gemm");
        if (tmp16) {
            __half *wh = (__half *)tmp16;
            __half *xh = (__half *)((char *)tmp16 + xh_off);
            /* GEMM into device-local scratch, then one bulk D2D to the
             * (possibly peer-mapped) out tensor: scattered peer stores
             * from GEMM kernels run at <1GB/s over PCIe. */
            float *olocal = (float *)((char *)tmp16 + oo_off);
            const uint64_t total_blocks = out_dim * blocks;
            q8_0_dequant_f16_kernel<<<(unsigned)((total_blocks * 2u + 255u) / 256u), 256, 0, cuda_decode_stream()>>>(
                    wh, reinterpret_cast<const unsigned char *>(wptr),
                    total_blocks, (uint32_t)blocks, (uint32_t)in_dim);
            const uint64_t xh_count = n_tok * in_dim;
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256, 0, cuda_decode_stream()>>>(
                    xh, (const float *)x->ptr, xh_count);
            if (cuda_ok(cudaGetLastError(), "q8 dequant gemm staging")) {
                const int gemm_trace = getenv("DS4_GLM_GEMM_TRACE") != NULL;
                cudaEvent_t ev0, ev1, ev2;
                if (gemm_trace) {
                    cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventCreate(&ev2);
                    cudaEventRecord(ev0);
                }
                const float alpha = 1.0f;
                const float beta = 0.0f;
                if (gemm_trace) cudaEventRecord(ev1);
                cublasStatus_t st = cublasGemmEx(
                        cuda_cublas_for_tier(logical_tier),
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        (int)out_dim, (int)n_tok, (int)in_dim,
                        &alpha,
                        wh, CUDA_R_16F, (int)in_dim,
                        xh, CUDA_R_16F, (int)in_dim,
                        &beta,
                        olocal, CUDA_R_32F, (int)out_dim,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
                if (st == CUBLAS_STATUS_SUCCESS) {
                    if (!cuda_ok(cudaMemcpyAsync(out->ptr, olocal,
                                 n_tok * out_dim * sizeof(float),
                                 cudaMemcpyDeviceToDevice, 0),
                                 "q8 dequant gemm out copy")) {
                        st = CUBLAS_STATUS_INTERNAL_ERROR;
                    }
                }
                if (gemm_trace) {
                    cudaEventRecord(ev2);
                    cudaEventSynchronize(ev2);
                    float stage_ms = 0, gemm_ms = 0;
                    cudaEventElapsedTime(&stage_ms, ev0, ev1);
                    cudaEventElapsedTime(&gemm_ms, ev1, ev2);
                    fprintf(stderr,
                        "ds4: gemm trace in=%llu out=%llu n=%llu stage(before)=%.2f gemm=%.2f ms\n",
                        (unsigned long long)in_dim, (unsigned long long)out_dim,
                        (unsigned long long)n_tok, stage_ms, gemm_ms);
                    cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaEventDestroy(ev2);
                }
                if (st == CUBLAS_STATUS_SUCCESS) return 1;
                fprintf(stderr,
                        "ds4: q8 dequant gemm failed: status %d; using native path\n",
                        (int)st);
            }
        }
    }
    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "q8_0 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32, 0, cuda_decode_stream()>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 quantize launch")) return 0;
    if (n_tok == 1) {
        matmul_q8_0_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256, 0, cuda_decode_stream()>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 warp launch");
    }
    const bool force_decode_warp =
        n_tok == 2u && g_glm_mtp_verify_mode;
    if (n_tok > 1u && !force_decode_warp) {
        /* T matches the reduction width of whichever reference kernel would
         * have run: warp tree (32) for blocks <= 32, exact-thread tree
         * otherwise. */
        const uint32_t mma_T = blocks <= 32u ? 32u : cuda_q8_exact_threads(blocks);
        const int mma_rc = cuda_q8_mma_try_launch(
                (float *)out->ptr, reinterpret_cast<const unsigned char *>(wptr),
                xq, xscale, in_dim, out_dim, n_tok, blocks, blocks, out_dim, mma_T);
        if (mma_rc) return mma_rc > 0;
    }
    if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL &&
        getenv("DS4_CUDA_NO_Q8_BATCH_TOK8") == NULL &&
        blocks <= 32u &&
        n_tok >= 8u) {
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, ((unsigned)n_tok + 7u) / 8u, 1);
        matmul_q8_0_preq_batch_warp8_tok8_kernel<<<bgrid, 256, 0, cuda_decode_stream()>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 batch tok8 warp launch");
    }
    if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL &&
        getenv("DS4_CUDA_NO_Q8_BATCH_TOK4") == NULL &&
        blocks <= 32u &&
        n_tok >= 4u) {
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, ((unsigned)n_tok + 3u) / 4u, 1);
        matmul_q8_0_preq_batch_warp8_tok4_kernel<<<bgrid, 256, 0, cuda_decode_stream()>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 batch tok4 warp launch");
    }
    if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL &&
        (blocks <= 32u || force_decode_warp)) {
        if (force_decode_warp &&
            getenv("DS4_CUDA_GLM_VERIFY_NO_Q8_TOK2") == NULL) {
            matmul_q8_0_preq_batch_warp8_tok2_kernel
                    <<<((unsigned)out_dim + 7u) / 8u, 256, 0, cuda_decode_stream()>>>(
                    (float *)out->ptr,
                    reinterpret_cast<const unsigned char *>(wptr),
                    xq,
                    xscale,
                    in_dim,
                    out_dim,
                    blocks,
                    use_dp4a);
            return cuda_ok(cudaGetLastError(),
                           "matmul_q8_0 batch tok2 warp launch");
        }
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_q8_0_preq_batch_warp8_kernel<<<bgrid, 256, 0, cuda_decode_stream()>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 batch warp launch");
    }
    const unsigned exact_threads = cuda_q8_exact_threads(blocks);
    if (getenv("DS4_CUDA_NO_Q8_BATCH_EXACT_TOK2") == NULL &&
        n_tok >= 2u) {
        dim3 bgrid((unsigned)out_dim, ((unsigned)n_tok + 1u) / 2u, 1);
        matmul_q8_0_preq_batch_tok2_exact_kernel<<<bgrid, exact_threads, 0, cuda_decode_stream()>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 exact tok2 launch");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_q8_0_preq_kernel<<<grid, exact_threads, 0, cuda_decode_stream()>>>((float *)out->ptr,
                                                     reinterpret_cast<const unsigned char *>(wptr),
                                                     xq,
                                                     xscale,
                                                     in_dim, out_dim, n_tok, blocks,
                                                     use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 launch");
}

extern "C" int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_q8_0_tensor_labeled(out, model_map, model_size, weight_offset,
                                           in_dim, out_dim, x, n_tok, "q8_0");
}

extern "C" int ds4_gpu_matmul_q8_0_top1_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint32_t                index_offset) {
    if (!selected || !values || !x || !model_map ||
        in_dim == 0 || out_dim == 0 || out_dim > UINT32_MAX) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34u)) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        selected->bytes < sizeof(uint32_t) ||
        values->bytes < sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(selected);
    const char *wptr = cuda_resolve_weight_ptr(model_map,
                                               weight_offset,
                                               weight_bytes,
                                               logical_tier,
                                               "q8_0_top1");
    if (!wptr) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t key_offset =
        (scale_offset + blocks * sizeof(float) + 7u) & ~7ull;
    const uint64_t tmp_bytes = key_offset + sizeof(unsigned long long);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "q8_0 top1 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    unsigned long long *best_key =
        (unsigned long long *)((char *)tmp + key_offset);
    const int use_dp4a = cuda_q8_use_dp4a();

    if (!cuda_ok(cudaMemsetAsync(best_key, 0, sizeof(*best_key)),
                 "matmul_q8_0_top1 clear")) {
        return 0;
    }
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(
            xq,
            xscale,
            (const float *)x->ptr,
            in_dim,
            blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_top1 quantize launch")) return 0;
    matmul_q8_0_top1_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
            best_key,
            reinterpret_cast<const unsigned char *>(wptr),
            xq,
            xscale,
            in_dim,
            out_dim,
            blocks,
            index_offset,
            use_dp4a);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_top1 launch")) return 0;
    matmul_q8_0_top1_unpack_kernel<<<1, 1>>>(
            (uint32_t *)selected->ptr,
            (float *)values->ptr,
            best_key);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_top1 unpack launch");
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_rows_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t in_start,
        uint64_t in_count,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 ||
        in_count == 0 || n_tok == 0 || n_tok > 65535u) return 0;
    if ((in_start % 32u) != 0 || (in_count % 32u) != 0 ||
        in_start > in_dim || in_count > in_dim - in_start) return 0;
    const uint64_t full_blocks = (in_dim + 31u) / 32u;
    const uint64_t block_start = in_start / 32u;
    const uint64_t slice_blocks = in_count / 32u;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (full_blocks * 34u)) return 0;
    const uint64_t weight_bytes = out_dim * full_blocks * 34u;
    if (in_count > UINT64_MAX / n_tok || out_dim > UINT64_MAX / n_tok) {
        return 0;
    }
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < n_tok * in_count * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const unsigned char *wptr = reinterpret_cast<const unsigned char *>(
            cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes,
                                    logical_tier, "q8_0_kslice"));
    if (!wptr) return 0;

    const uint64_t xq_bytes = n_tok * slice_blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes =
        scale_offset + n_tok * slice_blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "q8_0 kslice prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    const dim3 qgrid((unsigned)slice_blocks, (unsigned)n_tok, 1u);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
            xq,
            xscale,
            (const float *)x->ptr,
            in_count,
            slice_blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_kslice quantize launch")) return 0;
    const dim3 grid(((unsigned)out_dim + 7u) / 8u,
                    (unsigned)n_tok, 1u);
    matmul_q8_0_kslice_preq_warp8_kernel<<<grid, 256>>>(
            (float *)out->ptr,
            wptr,
            xq,
            xscale,
            in_count,
            out_dim,
            full_blocks,
            block_start,
            slice_blocks,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_kslice launch");
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        uint64_t                in_start,
        uint64_t                in_count,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !block_out || !x || !block_add || !residual_hc || !split ||
        !model_map || in_dim == 0 || out_dim == 0 || in_count == 0 ||
        n_embd == 0 || n_hc == 0 || out_dim != (uint64_t)n_embd) {
        return 0;
    }
    if ((in_start % 32u) != 0 || (in_count % 32u) != 0 ||
        in_start > in_dim || in_count > in_dim - in_start) return 0;
    const uint64_t full_blocks = (in_dim + 31u) / 32u;
    const uint64_t block_start = in_start / 32u;
    const uint64_t slice_blocks = in_count / 32u;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (full_blocks * 34u)) return 0;
    const uint64_t weight_bytes = out_dim * full_blocks * 34u;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes =
        (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_count * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        block_add->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out_hc);
    const unsigned char *wptr = reinterpret_cast<const unsigned char *>(
            cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes,
                                    logical_tier, "q8_0_kslice_hc_expand_add"));
    if (!wptr) return 0;

    const uint64_t xq_bytes = slice_blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + slice_blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "q8_0 kslice hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<(unsigned)slice_blocks, 32>>>(
            xq,
            xscale,
            (const float *)x->ptr,
            in_count,
            slice_blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_kslice_hc_expand_add quantize launch")) return 0;
    matmul_q8_0_kslice_hc_expand_add_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
            (float *)out_hc->ptr,
            (float *)block_out->ptr,
            (const float *)block_add->ptr,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            wptr,
            xq,
            xscale,
            in_count,
            out_dim,
            full_blocks,
            block_start,
            slice_blocks,
            n_embd,
            n_hc,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_kslice_hc_expand_add launch");
}

extern "C" int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out0_dim > UINT64_MAX / (blocks * 34) ||
        out1_dim > UINT64_MAX / (blocks * 34)) {
        return 0;
    }
    const uint64_t weight0_bytes = out0_dim * blocks * 34;
    const uint64_t weight1_bytes = out1_dim * blocks * 34;
    if (weight0_bytes > model_size - weight0_offset ||
        weight1_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) ||
        out1->bytes < out1_dim * sizeof(float)) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out0);
    const char *w0 = cuda_resolve_weight_ptr(model_map, weight0_offset, weight0_bytes, logical_tier, "q8_0_pair0");
    const char *w1 = cuda_resolve_weight_ptr(model_map, weight1_offset, weight1_bytes, logical_tier, "q8_0_pair1");
    if (!w0 || !w1) return 0;

    const bool force_decode_warp =
        n_tok == 2u && g_glm_mtp_verify_mode;
    if (n_tok != 1 && !force_decode_warp && !g_q8_cache_suppressed &&
        getenv("DS4_CUDA_Q8_PAIR_BATCH") == NULL) {
        return cuda_matmul_q8_0_tensor_labeled(out0, model_map, model_size, weight0_offset,
                                               in_dim, out0_dim, x, n_tok, "q8_0_pair0") &&
               cuda_matmul_q8_0_tensor_labeled(out1, model_map, model_size, weight1_offset,
                                               in_dim, out1_dim, x, n_tok, "q8_0_pair1");
    }

    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "q8_0 pair prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32, 0, cuda_decode_stream()>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair quantize launch")) return 0;
    if (n_tok != 1) {
        if (force_decode_warp &&
            getenv("DS4_CUDA_GLM_VERIFY_NO_Q8_TOK2") == NULL) {
            matmul_q8_0_preq_batch_warp8_tok2_kernel
                    <<<((unsigned)out0_dim + 7u) / 8u, 256, 0, cuda_decode_stream()>>>(
                    (float *)out0->ptr,
                    reinterpret_cast<const unsigned char *>(w0),
                    xq, xscale, in_dim, out0_dim, blocks, use_dp4a);
            if (!cuda_ok(cudaGetLastError(),
                         "matmul_q8_0 pair0 tok2 warp launch")) {
                return 0;
            }
            matmul_q8_0_preq_batch_warp8_tok2_kernel
                    <<<((unsigned)out1_dim + 7u) / 8u, 256, 0, cuda_decode_stream()>>>(
                    (float *)out1->ptr,
                    reinterpret_cast<const unsigned char *>(w1),
                    xq, xscale, in_dim, out1_dim, blocks, use_dp4a);
            return cuda_ok(cudaGetLastError(),
                           "matmul_q8_0 pair1 tok2 warp launch");
        }
        const uint32_t mma_T = blocks <= 32u ? 32u : cuda_q8_exact_threads(blocks);
        int mma_rc = cuda_q8_mma_try_launch(
                (float *)out0->ptr, reinterpret_cast<const unsigned char *>(w0),
                xq, xscale, in_dim, out0_dim, n_tok, blocks, blocks, out0_dim, mma_T);
        if (mma_rc < 0) return 0;
        if (mma_rc > 0) {
            mma_rc = cuda_q8_mma_try_launch(
                    (float *)out1->ptr, reinterpret_cast<const unsigned char *>(w1),
                    xq, xscale, in_dim, out1_dim, n_tok, blocks, blocks, out1_dim, mma_T);
            if (mma_rc > 0) return 1;
            return 0;
        }
        if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL &&
            getenv("DS4_CUDA_NO_Q8_BATCH_TOK8") == NULL &&
            blocks <= 32u &&
            n_tok >= 8u) {
            dim3 grid0(((unsigned)out0_dim + 7u) / 8u, ((unsigned)n_tok + 7u) / 8u, 1);
            matmul_q8_0_preq_batch_warp8_tok8_kernel<<<grid0, 256, 0, cuda_decode_stream()>>>(
                    (float *)out0->ptr,
                    reinterpret_cast<const unsigned char *>(w0),
                    xq,
                    xscale,
                    in_dim,
                    out0_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair0 batch tok8 launch")) return 0;
            dim3 grid1(((unsigned)out1_dim + 7u) / 8u, ((unsigned)n_tok + 7u) / 8u, 1);
            matmul_q8_0_preq_batch_warp8_tok8_kernel<<<grid1, 256, 0, cuda_decode_stream()>>>(
                    (float *)out1->ptr,
                    reinterpret_cast<const unsigned char *>(w1),
                    xq,
                    xscale,
                    in_dim,
                    out1_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair1 batch tok8 launch");
        }
        if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL &&
            getenv("DS4_CUDA_NO_Q8_BATCH_TOK4") == NULL &&
            blocks <= 32u &&
            n_tok >= 4u) {
            dim3 grid0(((unsigned)out0_dim + 7u) / 8u, ((unsigned)n_tok + 3u) / 4u, 1);
            matmul_q8_0_preq_batch_warp8_tok4_kernel<<<grid0, 256, 0, cuda_decode_stream()>>>(
                    (float *)out0->ptr,
                    reinterpret_cast<const unsigned char *>(w0),
                    xq,
                    xscale,
                    in_dim,
                    out0_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair0 batch tok4 launch")) return 0;
            dim3 grid1(((unsigned)out1_dim + 7u) / 8u, ((unsigned)n_tok + 3u) / 4u, 1);
            matmul_q8_0_preq_batch_warp8_tok4_kernel<<<grid1, 256, 0, cuda_decode_stream()>>>(
                    (float *)out1->ptr,
                    reinterpret_cast<const unsigned char *>(w1),
                    xq,
                    xscale,
                    in_dim,
                    out1_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair1 batch tok4 launch");
        }
        if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL &&
            blocks <= 32u) {
            dim3 grid0(((unsigned)out0_dim + 7u) / 8u, (unsigned)n_tok, 1);
            matmul_q8_0_preq_batch_warp8_kernel<<<grid0, 256, 0, cuda_decode_stream()>>>(
                    (float *)out0->ptr,
                    reinterpret_cast<const unsigned char *>(w0),
                    xq,
                    xscale,
                    in_dim,
                    out0_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair0 batch warp launch")) return 0;
            dim3 grid1(((unsigned)out1_dim + 7u) / 8u, (unsigned)n_tok, 1);
            matmul_q8_0_preq_batch_warp8_kernel<<<grid1, 256, 0, cuda_decode_stream()>>>(
                    (float *)out1->ptr,
                    reinterpret_cast<const unsigned char *>(w1),
                    xq,
                    xscale,
                    in_dim,
                    out1_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair1 batch warp launch");
        }
        if (getenv("DS4_CUDA_NO_Q8_PAIR_BATCH_EXACT") == NULL) {
            const uint64_t max_out_dim = out0_dim > out1_dim ? out0_dim : out1_dim;
            const unsigned exact_threads = cuda_q8_exact_threads(blocks);
            if (getenv("DS4_CUDA_NO_Q8_PAIR_BATCH_EXACT_TOK2") == NULL &&
                n_tok >= 2u) {
                dim3 grid((unsigned)max_out_dim, ((unsigned)n_tok + 1u) / 2u, 1);
                matmul_q8_0_pair_preq_batch_tok2_exact_kernel<<<grid, exact_threads, 0, cuda_decode_stream()>>>(
                        (float *)out0->ptr,
                        (float *)out1->ptr,
                        reinterpret_cast<const unsigned char *>(w0),
                        reinterpret_cast<const unsigned char *>(w1),
                        xq,
                        xscale,
                        in_dim,
                        out0_dim,
                        out1_dim,
                        n_tok,
                        blocks,
                        use_dp4a);
                return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair exact tok2 launch");
            }
            dim3 grid((unsigned)max_out_dim, (unsigned)n_tok, 1);
            matmul_q8_0_pair_preq_batch_kernel<<<grid, exact_threads, 0, cuda_decode_stream()>>>(
                    (float *)out0->ptr,
                    (float *)out1->ptr,
                    reinterpret_cast<const unsigned char *>(w0),
                    reinterpret_cast<const unsigned char *>(w1),
                    xq,
                    xscale,
                    in_dim,
                    out0_dim,
                    out1_dim,
                    n_tok,
                    blocks,
                    use_dp4a);
            return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair batch exact launch");
        }
        const unsigned exact_threads = cuda_q8_exact_threads(blocks);
        dim3 grid0((unsigned)out0_dim, (unsigned)n_tok, 1);
        matmul_q8_0_preq_kernel<<<grid0, exact_threads, 0, cuda_decode_stream()>>>((float *)out0->ptr,
                                                          reinterpret_cast<const unsigned char *>(w0),
                                                          xq,
                                                          xscale,
                                                          in_dim, out0_dim, n_tok, blocks,
                                                          use_dp4a);
        if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair0 batch launch")) return 0;
        dim3 grid1((unsigned)out1_dim, (unsigned)n_tok, 1);
        matmul_q8_0_preq_kernel<<<grid1, exact_threads, 0, cuda_decode_stream()>>>((float *)out1->ptr,
                                                          reinterpret_cast<const unsigned char *>(w1),
                                                          xq,
                                                          xscale,
                                                          in_dim, out1_dim, n_tok, blocks,
                                                          use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair1 batch launch");
    }
    const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
    matmul_q8_0_pair_preq_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256, 0, cuda_decode_stream()>>>(
            (float *)out0->ptr,
            (float *)out1->ptr,
            reinterpret_cast<const unsigned char *>(w0),
            reinterpret_cast<const unsigned char *>(w1),
            xq,
            xscale,
            in_dim,
            out0_dim,
            out1_dim,
            blocks,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair warp launch");
}

extern "C" int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint32_t                n_rows) {
    if (!out || !x || !model_map || in_dim == 0u || out_dim == 0u ||
        n_rows == 0u ||
        x->bytes < (uint64_t)n_rows * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_rows * out_dim * sizeof(float)) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (weight_offset > model_size ||
        out_dim > UINT64_MAX / (blocks * 34u)) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    if (weight_bytes > model_size - weight_offset) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    if (logical_tier < 0 || logical_tier >= g_n_gpus ||
        ds4_tensor_device_idx(x) != logical_tier) {
        return 0;
    }
    const char *wptr = cuda_resolve_weight_ptr(
            model_map, weight_offset, weight_bytes, logical_tier,
            "q8_0 decode rows exact");
    if (!wptr) return 0;

    const uint64_t xq_bytes = (uint64_t)n_rows * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes =
        scale_offset + (uint64_t)n_rows * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(
            logical_tier, tmp_bytes, "q8_0 decode rows exact prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    dim3 qgrid((unsigned)blocks, n_rows, 1u);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(
            xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(),
                 "q8_0 decode rows exact quantize launch")) {
        return 0;
    }
    dim3 grid(((unsigned)out_dim + 7u) / 8u, n_rows, 1u);
    matmul_q8_0_preq_warp8_kernel<<<grid, 256>>>(
            (float *)out->ptr,
            reinterpret_cast<const unsigned char *>(wptr),
            xq, xscale, in_dim, out_dim, blocks, cuda_q8_use_dp4a());
    return cuda_ok(cudaGetLastError(),
                   "q8_0 decode rows exact warp launch");
}

extern "C" int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ds4_gpu_tensor *x,
        uint32_t                n_rows) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0u ||
        out0_dim == 0u || out1_dim == 0u || n_rows == 0u ||
        x->bytes < (uint64_t)n_rows * in_dim * sizeof(float) ||
        out0->bytes < (uint64_t)n_rows * out0_dim * sizeof(float) ||
        out1->bytes < (uint64_t)n_rows * out1_dim * sizeof(float)) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out0_dim > UINT64_MAX / (blocks * 34u) ||
        out1_dim > UINT64_MAX / (blocks * 34u)) {
        return 0;
    }
    const uint64_t weight0_bytes = out0_dim * blocks * 34u;
    const uint64_t weight1_bytes = out1_dim * blocks * 34u;
    if (weight0_bytes > model_size - weight0_offset ||
        weight1_bytes > model_size - weight1_offset) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out0);
    if (logical_tier < 0 || logical_tier >= g_n_gpus ||
        ds4_tensor_device_idx(out1) != logical_tier ||
        ds4_tensor_device_idx(x) != logical_tier) {
        return 0;
    }
    const char *w0 = cuda_resolve_weight_ptr(
            model_map, weight0_offset, weight0_bytes, logical_tier,
            "q8_0 pair decode rows exact gate");
    const char *w1 = cuda_resolve_weight_ptr(
            model_map, weight1_offset, weight1_bytes, logical_tier,
            "q8_0 pair decode rows exact up");
    if (!w0 || !w1) return 0;

    const uint64_t xq_bytes = (uint64_t)n_rows * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes =
        scale_offset + (uint64_t)n_rows * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(
            logical_tier, tmp_bytes, "q8_0 pair decode rows exact prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    dim3 qgrid((unsigned)blocks, n_rows, 1u);
    quantize_q8_0_f32_kernel<<<qgrid, 32, 0, cuda_decode_stream()>>>(
            xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(),
                 "q8_0 pair decode rows exact quantize launch")) {
        return 0;
    }
    const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
    dim3 grid(((unsigned)max_out + 7u) / 8u, n_rows, 1u);
    matmul_q8_0_pair_preq_warp8_kernel<<<grid, 256, 0, cuda_decode_stream()>>>(
            (float *)out0->ptr,
            (float *)out1->ptr,
            reinterpret_cast<const unsigned char *>(w0),
            reinterpret_cast<const unsigned char *>(w1),
            xq, xscale, in_dim, out0_dim, out1_dim, blocks,
            cuda_q8_use_dp4a());
    return cuda_ok(cudaGetLastError(),
                   "q8_0 pair decode rows exact warp launch");
}

static int cuda_matmul_q8_0_hc_expand_tensor_labeled(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *block_add2,
        const ds4_gpu_tensor *owned_home_slots,
        const ds4_gpu_tensor *owned_peer_packed,
        const ds4_gpu_tensor *owned_selected,
        uint32_t                owned_expert_split,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes ||
        (block_add && block_add->bytes < out_dim * sizeof(float)) ||
        (block_add2 && block_add2->bytes < out_dim * sizeof(float)) ||
        ((owned_home_slots || owned_peer_packed || owned_selected) &&
         (!owned_home_slots || !owned_peer_packed || !owned_selected ||
          owned_expert_split == 0u ||
          owned_home_slots->bytes < 6u * out_dim * sizeof(float) ||
          owned_peer_packed->bytes < 4u * out_dim * sizeof(float) ||
          owned_selected->bytes < 6u * sizeof(int32_t)))) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out_hc);
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes, logical_tier, label ? label : "q8_0_hc_expand");
    if (!wptr) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc_on(logical_tier, tmp_bytes, "q8_0 hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32, 0, cuda_decode_stream()>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand quantize launch")) return 0;
    matmul_q8_0_hc_expand_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256, 0, cuda_decode_stream()>>>(
            (float *)out_hc->ptr,
            (float *)block_out->ptr,
            block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
            block_add2 ? (const float *)block_add2->ptr : (const float *)block_out->ptr,
            owned_home_slots ? (const float *)owned_home_slots->ptr : NULL,
            owned_peer_packed ? (const float *)owned_peer_packed->ptr : NULL,
            owned_selected ? (const int32_t *)owned_selected->ptr : NULL,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            reinterpret_cast<const unsigned char *>(wptr),
            xq,
            xscale,
            in_dim,
            out_dim,
            n_embd,
            n_hc,
            blocks,
            block_add ? 1 : 0,
            block_add2 ? 1 : 0,
            owned_home_slots ? 1 : 0,
            owned_expert_split,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand launch");
}

extern "C" int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes, logical_tier, "f16");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
    const int serial_f16 = getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL;
    const int router_shape = in_dim == 4096u && out_dim == 256u && n_tok == 1u;
    const int serial_router =
        !serial_f16 &&
        router_shape &&
        getenv("DS4_CUDA_SERIAL_ROUTER") != NULL;
    const int ordered_router =
        !serial_f16 &&
        !serial_router &&
        n_tok == 1u &&
        getenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") == NULL;
    const int small_out_one_token =
        !serial_f16 &&
        !serial_router &&
        !g_quality_mode &&
        n_tok == 1u &&
        out_dim <= 32u &&
        in_dim >= 8192u &&
        getenv("DS4_CUDA_F16_SMALL_OUT") != NULL &&
        getenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") == NULL &&
        getenv("DS4_CUDA_NO_F16_SMALL_OUT") == NULL;
    if (small_out_one_token) {
        const int status = ds4x_launch_fp16_small_output_one(
                (float *)out->ptr,
                (const uint16_t *)w,
                (const float *)x->ptr,
                in_dim,
                out_dim,
                (void *)cuda_decode_stream());
        return cuda_ok((cudaError_t)status, "matmul_f16_small_output_one launch");
    }
    const int small_out_batch =
        !serial_f16 &&
        !serial_router &&
        n_tok > 1u &&
        out_dim <= 32u &&
        in_dim >= 4096u &&
        (g_quality_mode || getenv("DS4_CUDA_F16_SMALL_BATCH") != NULL) &&
        getenv("DS4_CUDA_NO_F16_SMALL_BATCH") == NULL;
    if (small_out_batch) {
        const int status = ds4x_launch_fp16_small_output_batch(
                (float *)out->ptr,
                (const uint16_t *)w,
                (const float *)x->ptr,
                in_dim,
                out_dim,
                n_tok,
                (void *)cuda_decode_stream());
        return cuda_ok((cudaError_t)status, "matmul_f16_small_output_batch launch");
    }
    const int cublas_one_token =
        n_tok == 1u &&
        getenv("DS4_CUDA_NO_F16_CUBLAS_ONE") == NULL &&
        (!g_quality_mode || getenv("DS4_CUDA_F16_CUBLAS_ONE") != NULL);
    const int cublas_batch =
        n_tok > 1u && getenv("DS4_CUDA_NO_F16_CUBLAS_BATCH") == NULL;
    if (!serial_f16 && g_cublas_ready && (cublas_batch || cublas_one_token)) {
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc_on(logical_tier, xh_count * sizeof(__half), "f16 gemm activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255) / 256, 256, 0, cuda_decode_stream()>>>(xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 activation convert launch")) return 0;
        /* On GB10, CUTLASS's explicit mma.sync kernel wins for the large
         * indexer q_b projection while losing badly on latency-sized
         * compressor/router GEMMs. Keep the measured auto policy narrow;
         * the environment override exists for reproducible A/B work. */
        static const int cutlass_mode = [] {
            const char *backend = getenv("DS4_CUDA_F16_BACKEND");
            return backend && strcmp(backend, "cutlass") == 0 ? 2 :
                   backend && strcmp(backend, "cublas") == 0 ? 0 : 1;
        }();
        const int cutlass_auto_shape =
            in_dim == 1024u && out_dim == 8192u && n_tok >= 2048u;
        if (cutlass_mode == 2 || (cutlass_mode == 1 && cutlass_auto_shape)) {
            const int rc = ds4x_cutlass_fp16_gemm(
                    (float *)out->ptr,
                    (const uint16_t *)xh,
                    (const uint16_t *)w,
                    n_tok,
                    in_dim,
                    out_dim,
                    (void *)cuda_decode_stream());
            if (rc == 1) return 1;
            if (rc < 0) {
                static int logged = 0;
                if (!logged) {
                    logged = 1;
                    fprintf(stderr,
                            "ds4: CUTLASS FP16 GEMM failed (%s); falling back to cuBLAS\n",
                            ds4x_cutlass_status_string(rc));
                }
            }
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(cuda_cublas_for_tier(logical_tier),
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         (int)out_dim,
                                         (int)n_tok,
                                         (int)in_dim,
                                         &alpha,
                                         w,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         xh,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         &beta,
                                         out->ptr,
                                         CUDA_R_32F,
                                         (int)out_dim,
                                         CUDA_R_32F,
                                         CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "f16 matmul");
    }
    const int mode = serial_f16 || serial_router ? DS4X_FP16_PROJECTION_SERIAL :
                     ordered_router ? DS4X_FP16_PROJECTION_ORDERED :
                                      DS4X_FP16_PROJECTION_PARALLEL;
    const int status = ds4x_launch_fp16_projection(
            (float *)out->ptr,
            (const uint16_t *)w,
            (const float *)x->ptr,
            in_dim,
            out_dim,
            n_tok,
            mode,
            (void *)cuda_decode_stream());
    const char *label = serial_router ? "matmul_f16_router_serial launch" :
                        mode == DS4X_FP16_PROJECTION_SERIAL ? "matmul_f16_serial launch" :
                        mode == DS4X_FP16_PROJECTION_ORDERED ? "matmul_f16_ordered launch" :
                                                             "matmul_f16_parallel launch";
    return cuda_ok((cudaError_t)status, label);
}

extern "C" int ds4_gpu_matmul_f16_rms_fold_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        float norm_eps) {
    if (!out || !x || !model_map || !g_cublas_ready || n_tok <= 1u ||
        (in_dim & 2047u) != 0u || in_dim > UINT32_MAX ||
        n_tok > UINT32_MAX || getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL ||
        getenv("DS4_CUDA_NO_F16_CUBLAS_BATCH") != NULL ||
        weight_offset > model_size || out_dim > UINT64_MAX / in_dim) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) {
        return 0;
    }

    const int logical_tier = ds4_tensor_device_idx(out);
    const char *wptr = cuda_resolve_weight_ptr(
        model_map, weight_offset, weight_bytes, logical_tier, "f16 rms fold");
    if (!wptr) return 0;
    const uint64_t xh_count = n_tok * in_dim;
    __half *xh = (__half *)cuda_tmp_alloc_on(
        logical_tier, xh_count * sizeof(__half), "f16 rms-fold activations");
    if (!xh) return 0;

    rms_norm_plain_f16_batch8_kernel<<<
        (unsigned)n_tok, 256, 0, cuda_decode_stream()>>>(
            xh, (const float *)x->ptr, (uint32_t)in_dim,
            (uint32_t)n_tok, norm_eps);
    if (!cuda_ok(cudaGetLastError(), "f16 rms-fold activation launch")) {
        return 0;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t st = cublasGemmEx(
        cuda_cublas_for_tier(logical_tier),
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)out_dim, (int)n_tok, (int)in_dim,
        &alpha,
        (const __half *)wptr, CUDA_R_16F, (int)in_dim,
        xh, CUDA_R_16F, (int)in_dim,
        &beta,
        out->ptr, CUDA_R_32F, (int)out_dim,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    return cublas_ok(st, "f16 rms-fold matmul");
}

extern "C" int ds4_gpu_matmul_f16_router_rows_exact_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        const ds4_gpu_tensor *x,
        uint32_t n_rows) {
    const uint64_t in_dim = 4096u;
    const uint64_t out_dim = 256u;
    if (!out || !x || !model_map || n_rows == 0u ||
        weight_offset > model_size) {
        return 0;
    }
    const uint64_t weight_bytes = in_dim * out_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < (uint64_t)n_rows * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_rows * out_dim * sizeof(float)) {
        return 0;
    }
    if (n_rows == 1u) {
        return ds4_gpu_matmul_f16_tensor(
                out, model_map, model_size, weight_offset,
                in_dim, out_dim, x, 1);
    }
    const int logical_tier = ds4_tensor_device_idx(out);
    if (ds4_tensor_device_idx(x) != logical_tier || !g_cublas_ready) return 0;
    const __half *w = (const __half *)cuda_resolve_weight_ptr(
            model_map, weight_offset, weight_bytes, logical_tier,
            "f16_router_rows_exact");
    if (!w) return 0;

    const uint64_t xh_count = (uint64_t)n_rows * in_dim;
    __half *xh = (__half *)cuda_tmp_alloc_on(
            logical_tier, xh_count * sizeof(__half),
            "f16 exact router batch activations");
    if (!xh) return 0;
    f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(
            xh, (const float *)x->ptr, xh_count);
    if (!cuda_ok(cudaGetLastError(),
                 "f16 exact router activation convert launch")) {
        return 0;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    /* Larger batchCount values let cuBLAS select a different reduction and
     * change logits. Four-row calls match the one-row decode bit for bit on
     * this projection, while still replacing most per-session launches. */
    uint32_t row = 0;
    for (; row + 4u <= n_rows; row += 4u) {
        cublasStatus_t st = cublasGemmStridedBatchedEx(
                cuda_cublas_for_tier(logical_tier),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                (int)out_dim,
                1,
                (int)in_dim,
                &alpha,
                w,
                CUDA_R_16F,
                (int)in_dim,
                0,
                xh + (uint64_t)row * in_dim,
                CUDA_R_16F,
                (int)in_dim,
                (long long int)in_dim,
                &beta,
                (float *)out->ptr + (uint64_t)row * out_dim,
                CUDA_R_32F,
                (int)out_dim,
                (long long int)out_dim,
                4,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "f16 exact router row batch")) return 0;
    }
    for (; row < n_rows; row++) {
        ds4_gpu_tensor out_row = *out;
        ds4_gpu_tensor x_row = *x;
        out_row.ptr = (float *)out->ptr + (uint64_t)row * out_dim;
        out_row.bytes = out_dim * sizeof(float);
        x_row.ptr = (float *)x->ptr + (uint64_t)row * in_dim;
        x_row.bytes = in_dim * sizeof(float);
        if (!ds4_gpu_matmul_f16_tensor(
                    &out_row, model_map, model_size, weight_offset,
                    in_dim, out_dim, &x_row, 1)) {
            return 0;
        }
    }
    return 1;
}

extern "C" int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) {
        return 0;
    }
    if (getenv("DS4_CUDA_NO_F16_PAIR_MATMUL") != NULL ||
        getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL ||
        getenv("DS4_CUDA_SERIAL_ROUTER") != NULL ||
        getenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") != NULL) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out_dim > UINT64_MAX / in_dim ||
        n_tok > UINT64_MAX / in_dim ||
        n_tok > UINT64_MAX / out_dim) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    const uint64_t x_bytes = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    if (weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        x->bytes < x_bytes ||
        out0->bytes < out_bytes ||
        out1->bytes < out_bytes) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out0);
    if (ds4_tensor_device_idx(out1) != logical_tier) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    const __half *w0 = (const __half *)cuda_resolve_weight_ptr(model_map, weight0_offset, weight_bytes, logical_tier, "f16_pair0");
    const __half *w1 = (const __half *)cuda_resolve_weight_ptr(model_map, weight1_offset, weight_bytes, logical_tier, "f16_pair1");
    if (!w0 || !w1) return 0;
    if (n_tok > 1) {
        const bool small_out_batch_requested =
            out_dim <= 32u &&
            in_dim >= 4096u &&
            (g_quality_mode || getenv("DS4_CUDA_F16_SMALL_BATCH") != NULL) &&
            getenv("DS4_CUDA_NO_F16_SMALL_BATCH") == NULL;
        if (!small_out_batch_requested &&
            g_cublas_ready &&
            getenv("DS4_CUDA_NO_F16_CUBLAS_BATCH") == NULL) {
            const uint64_t xh_count = n_tok * in_dim;
            __half *xh = (__half *)cuda_tmp_alloc_on(logical_tier,
                                                     xh_count * sizeof(__half),
                                                     "f16 pair gemm activations");
            if (!xh) return 0;
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(
                    xh, (const float *)x->ptr, xh_count);
            if (!cuda_ok(cudaGetLastError(), "f16 pair activation convert launch")) return 0;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(cuda_cublas_for_tier(logical_tier),
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             (int)out_dim,
                                             (int)n_tok,
                                             (int)in_dim,
                                             &alpha,
                                             w0,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             xh,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             &beta,
                                             out0->ptr,
                                             CUDA_R_32F,
                                             (int)out_dim,
                                             CUDA_R_32F,
                                             CUBLAS_GEMM_DEFAULT);
            if (!cublas_ok(st, "f16 pair matmul0")) return 0;
            st = cublasGemmEx(cuda_cublas_for_tier(logical_tier),
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              (int)out_dim,
                              (int)n_tok,
                              (int)in_dim,
                              &alpha,
                              w1,
                              CUDA_R_16F,
                              (int)in_dim,
                              xh,
                              CUDA_R_16F,
                              (int)in_dim,
                              &beta,
                              out1->ptr,
                              CUDA_R_32F,
                              (int)out_dim,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT);
            return cublas_ok(st, "f16 pair matmul1");
        }
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    const int status = ds4x_launch_fp16_pair_ordered(
        (float *)out0->ptr,
        (float *)out1->ptr,
        (const uint16_t *)w0,
        (const uint16_t *)w1,
        (const float *)x->ptr,
        in_dim,
        out_dim,
        out_dim,
        NULL);
    return cuda_ok((cudaError_t)status, "matmul_f16_pair_ordered launch");
}

extern "C" int ds4_gpu_matmul_f16_pair_compressor_store_tensor(
        ds4_gpu_tensor *out_kv,
        ds4_gpu_tensor *out_score,
        ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_kv_offset,
        uint64_t weight_score_offset,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint64_t in_dim,
        uint32_t width,
        const ds4_gpu_tensor *x,
        uint32_t ratio,
        uint32_t pos) {
    (void)out_kv;
    (void)out_score;
    (void)state_kv;
    (void)state_score;
    (void)model_map;
    (void)model_size;
    (void)weight_kv_offset;
    (void)weight_score_offset;
    (void)ape_offset;
    (void)ape_type;
    (void)in_dim;
    (void)width;
    (void)x;
    (void)ratio;
    (void)pos;
    return 0;
}

extern "C" int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_elems = out_dim * in_dim;
    if (weight_elems > UINT64_MAX / sizeof(float)) return 0;
    uint64_t weight_bytes = weight_elems * sizeof(float);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const char *wptr = cuda_resolve_weight_ptr(model_map, weight_offset, weight_bytes, logical_tier, "f32");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    if (g_cublas_ready && n_tok > 1) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(cuda_cublas_for_tier(logical_tier),
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        (int)out_dim,
                                        (int)n_tok,
                                        (int)in_dim,
                                        &alpha,
                                        w,
                                        (int)in_dim,
                                        (const float *)x->ptr,
                                        (int)in_dim,
                                        &beta,
                                        (float *)out->ptr,
                                        (int)out_dim);
        return cublas_ok(st, "f32 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_f32_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f32 launch");
}

extern "C" int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !row || n_embd == 0 || n_hc == 0 ||
        row->bytes < (uint64_t)n_embd * sizeof(float) ||
        out->bytes < (uint64_t)n_embd * n_hc * sizeof(float)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_embd * n_hc;
    repeat_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)row->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "repeat_hc launch");
}
