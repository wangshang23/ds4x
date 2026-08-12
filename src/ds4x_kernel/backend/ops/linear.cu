#include "../internal/backend_internal.cuh"

/* Linear implementation. */


/* Emit the same normalized values as rms_norm_plain_batch8_kernel directly
 * as FP16 activations for the following cuBLAS projection. */
__global__ static void rms_norm_plain_f16_batch8_kernel(
        __half *out,
        const float *x,
        uint32_t n,
        uint32_t rows,
        float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    __half *orow = out + (uint64_t)row * n;
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
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
#pragma unroll 1
    for (uint32_t i = threadIdx.x; i < n; i += 2048u) {
        orow[i] = __float2half(xr[i] * scale);
        orow[i + 256u] = __float2half(xr[i + 256u] * scale);
        orow[i + 512u] = __float2half(xr[i + 512u] * scale);
        orow[i + 768u] = __float2half(xr[i + 768u] * scale);
        orow[i + 1024u] = __float2half(xr[i + 1024u] * scale);
        orow[i + 1280u] = __float2half(xr[i + 1280u] * scale);
        orow[i + 1536u] = __float2half(xr[i + 1536u] * scale);
        orow[i + 1792u] = __float2half(xr[i + 1792u] * scale);
    }
}

int cuda_matmul_q8_0_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map) return 0;
    uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    uint64_t weight_bytes = out_dim * blocks * 34;
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const int physical_device = 0;
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
    if (n_tok > 1u) {
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
        blocks <= 32u) {
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

    if (n_tok != 1 && !g_q8_cache_suppressed &&
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

int cuda_matmul_q8_0_hc_expand_tensor_labeled(
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
        (block_add2 && block_add2->bytes < out_dim * sizeof(float))) {
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
            in_dim == 1024u && out_dim == 8192u && n_tok >= 128u;
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
