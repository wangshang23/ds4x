extern "C" int ds4_mmq_q8_0_dense(

#pragma once
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_Q8_0>("ds4_mmq_q8_0_dense", W, X, out, M, N, K, stream);
}

// Dense Q8_0 D2R entry: same activation quantize + scratch treatment as
// ds4_mmq_dense_impl (incl. the S1.1a zero for the never-written tail), then
// the D2R kernel on the kind-5 aligned artifact instead of mul_mat_q_case.
// No out-memset / trailing sanitize: the D2R epilogue writes every element
// through an isfinite guard.  Caller (ds4_cuda.cu) resolves W_aligned and
// gates on shape (M%128, K%1024, K<=4096) + n_tok.
extern "C" int ds4_mmq_q8_0_dense_d2r(
        const void * W_aligned, const float * X_f32, float * out_f32,
        int M, int N, int K, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q8_0_dense_d2r";
    if (!W_aligned || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || (M % 128) != 0 || N <= 0 || K <= 0 || (K % 1024) != 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_q8_0_dense_d2r_available(cc)) {
        return -1;
    }
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    ds4_pool_set_stream(stream);

    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    // Slack: the guarded last col tile reads up to 128 blocks past N*K/128.
    const int64_t slack_blocks = std::max<int64_t>(get_mmq_x_max_host(cc), 128);
    const size_t nbytes_src1_q8_1 =
        (int64_t)N * ne10_padded * sizeof(block_q8_1) / QK8_1 +
        slack_blocks * sizeof(block_q8_1_mmq);

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);
    ybuf_memset(src1_q8_1.get(), nbytes_src1_q8_1, stream);

    quantize_mmq_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1.get(),
        GGML_TYPE_Q8_0, /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
        /*ne0=*/ne10_padded, /*ne1=*/(int64_t)N, /*ne2=*/1, /*ne3=*/1,
        stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }
    return ds4_mmq_q8_0_dense_d2r_launch(W_aligned, src1_q8_1.get(), out_f32,
                                         M, N, K, stream);
}

extern "C" int ds4_mmq_q2_K_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_Q2_K>("ds4_mmq_q2_K_dense", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_iq2_xxs_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_IQ2_XXS>("ds4_mmq_iq2_xxs_dense", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_q4_K_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_Q4_K>("ds4_mmq_q4_K_dense", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_mxfp4_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_MXFP4>("ds4_mmq_mxfp4_dense", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_nvfp4_dense(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_impl<GGML_TYPE_NVFP4>("ds4_mmq_nvfp4_dense", W, X, out, M, N, K, stream);
}

// ----------------------------------------------------------------------------
// MoE matmul implementation, shared across all three quant types.
//
// Mirrors upstream mmq.cu:163-222 (the ids != nullptr branch).  Caller
// provides:
//   - per-expert weights stacked contiguously
//   - per-token activations [n_tokens, K]
//   - routing table ids[t, s] = expert id
// The wrapper invokes:
//   1. ggml_cuda_launch_mm_ids_helper to build (ids_src1, ids_dst,
//      expert_bounds) - permutations that sort assignments by expert.
//   2. quantize_mmq_q8_1_cuda with ids_src1 - gathers and quantizes the
//      activation into the expert-major flat layout.
//   3. mul_mat_q_case<type> with ids_dst + expert_bounds - the matmul.
// ----------------------------------------------------------------------------

namespace {

template <ggml_type type>
int ds4_mmq_moe_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream,
        /* ds4 (P4 Inc3): optional aligned-SoA artifact; when non-null the mmq
         * kernel loads tiles from it directly and W is ignored (see mmq_args). */
        const char    * x_soa      = NULL,
        int64_t         soa_blocks = 0,
        /* ds4 (P3): false skips the whole-buffer nonfinite pass; only valid
         * when every consumer sanitizes at read (the routed-MoE swiglu/sum
         * kernels do). */
        bool            sanitize_out = true) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_k_tile_supported<type>(tag, K, cc)) return -1;

    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);  /* task #22: pool ops must be stream-ordered with the kernels (see ds4_mmq_dense_impl) */

    const int64_t ne_get_rows  = (int64_t)n_tokens * n_expert_used;
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = 1;             // src1 rows per channel (one per token)
    const int64_t ne12         = n_tokens;      // src1 channels (= tokens)
    const int64_t blck         = ggml_blck_size(type);
    const int64_t s01          = (int64_t)K / blck;
    const int64_t s02          = (int64_t)M * s01;   // per-expert weight stride in blocks

    // 1. Build the expert-major work map.
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx->pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx->pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx->pool(), n_experts + 1);

    // Task #22 root-cause fix: mm_ids_helper COMPACTS - it only writes ids_src1
    // entries for in-range router ids and drops invalid ones (the router's NaN
    // path emits -1 by design), so with any dropped id the tail of ids_src1
    // stays unwritten pool memory.  quantize_mmq_q8_1's grid covers all
    // ne_get_rows rows and gathers x rows via ids_src1[i1] unconditionally
    // (quantize.cu:304), so a stale/garbage tail entry becomes a wild OOB read
    // (the intermittent batched-draft illegal access; B200 memcheck-convicted).
    // Zero both id maps so unwritten tail slots gather/scatter row 0 instead:
    // those lanes' output is never consumed (the mmq write-back loop is
    // expert_bounds-bounded), the cost is a few KB of memset on-stream.
    cudaMemsetAsync(ids_src1.get(), 0, ne_get_rows * sizeof(int32_t), stream);
    cudaMemsetAsync(ids_dst.get(),  0, ne_get_rows * sizeof(int32_t), stream);

    // si1 = stride between tokens in the ids tensor, in elements. Our ids is
    // contiguous [n_tokens, n_expert_used] so si1 = n_expert_used.
    // sis1 = stride between src1 channels in row-units. With ne11=1, sis1=1
    //        means each "channel" of src1 is one row of K floats.
    const int si1  = n_expert_used;
    const int sis1 = 1;

    // The smem mm_ids_helper uses n_tokens * 4 bytes of dynamic shared memory;
    // the down matmul reaches here with n_tokens = assignments (6x the forward
    // width), so 8192-row prefill chunks pass 48384 "tokens" > cap.  P5: past
    // the cap the launcher dispatches the bit-identical two-pass global
    // variant instead (mmid.cu mm_ids_helper_global) — refusing here used to
    // throw the WHOLE MoE block (including gate/up mmq work) onto the legacy
    // expert-tile fallback, the W8192 prefill cliff.  DS4_MMID_LARGE=0
    // restores the refusal.
    if ((size_t)n_tokens * 4u > ggml_cuda_info().devices[dev].smpbo && !ds4_mmid_large_enabled()) {
        fprintf(stderr, "%s: n_tokens=%d exceeds mm_ids_helper shared-mem cap; falling back\n",
                tag, n_tokens);
        return -1;
    }

    ggml_cuda_launch_mm_ids_helper(
        ids, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, n_tokens, n_expert_used, /*nchannels_y=*/(int)ne11, si1, sis1, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    // 2. Gather + quantize activations. Native Blackwell MXFP4 consumes FP4;
    //    all other MMQ kernels consume Q8_1.
    const bool use_native_fp4 =
        type == GGML_TYPE_MXFP4 && blackwell_mma_available(cc);
    const size_t y_block_size = use_native_fp4
        ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4
        ? QK_FP4_MMQ : 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * y_block_size / y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx->pool(), nbytes_src1_q8_1);

    // S1.1a fix (same as the dense path): the mmq Y buffer is over-allocated for the
    // kernel's tail-tile reads and ne_get_rows columns need not fill the final mmq
    // column tile, but quantize only writes the valid columns.  The mmq kernel
    // (mmq.cuh:3528) unconditionally loads the full tile, reading the never-written
    // tail from stale pool memory -> allocator-perturbation-dependent garbage in the
    // (write_back-masked) tail lanes -> non-deterministic batched-forward output.
    // Zero it so the masked-out tail is a deterministic zero.
    ybuf_memset(src1_q8_1.get(), nbytes_src1_q8_1, stream);

    // src1 logical [K, ne11=1, ne12=n_tokens, ne13=1] - K innermost, then
    // one row per channel, channels = tokens.
    const int64_t s11_src = (int64_t)K;                                 // stride between rows of a channel
    const int64_t s12_src = (int64_t)K * ne11;                          // stride between channels = K*1
    const int64_t s13_src = (int64_t)K * ne11 * ne12;                   // stride between samples

    if (use_native_fp4) {
        quantize_mmq_fp4_cuda(
            X_f32, ids_src1.get(), (void *)src1_q8_1.get(),
            type, /*ne00=*/K, s11_src, s12_src, s13_src,
            /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows, /*ne2=*/1, /*ne3=*/1,
            stream);
    } else {
        quantize_mmq_q8_1_cuda(
            X_f32, ids_src1.get(), (void *)src1_q8_1.get(),
            type, /*ne00=*/K, s11_src, s12_src, s13_src,
            /*ne0=*/ne10_padded, /*ne1=*/ne_get_rows, /*ne2=*/1, /*ne3=*/1,
            stream);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: MMQ activation quantize failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }

    // 3. Build mmq_args for the MoE path.
    //
    // dst layout convention matches upstream's MoE branch
    // (mmq.cu:215-220): dst is interpreted as [M, n_expert_used, n_tokens]
    // with M innermost and n_expert_used as the second dim that mmq writes
    // through ids_dst.  s1 = M (the column stride in the flat dst buffer
    // mmq writes into).  The output is column-major: out[col*M + row].
    const int64_t s1            = (int64_t)M;
    // stride_channel_y per upstream: ne11 * ne10_padded * sizeof(block_q8_1)
    //                                     / (QK8_1 * sizeof(int))
    // In MoE mode the kernel zeroes out the channel-stride contribution to
    // offset_y after reading expert_bounds, so the value is permissive -
    // but we set it consistently with upstream.
    const int64_t s12_mmq = ne11 * ne10_padded * y_block_size /
                            (y_values_per_block * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);

    if (out_memset_enabled()) {
        cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)ne_get_rows * sizeof(float), stream);
    }

    if (type == GGML_TYPE_Q2_K && x_soa != nullptr && d2r_enabled() &&
        K % 256 == 0 && M % 2 == 0 && ne_get_rows >= d2r_min_cols()) {
        static int d2r_avail_cc = -1;
        static int d2r_avail = 0;
        if (d2r_avail_cc != cc) {
            d2r_avail_cc = cc;
            d2r_avail = ds4_mmq_q2_K_moe_d2r_available(cc) ? 1 : 0;
        }
        if (d2r_avail) {
            const size_t d2r_work_bytes =
                ds4_mmq_q2_K_moe_d2r_scratch_bytes(ne_get_rows, n_experts);
            if (d2r_work_bytes != 0) {
                ggml_cuda_pool_alloc<char> d2r_work(ctx->pool(), d2r_work_bytes);
                const int d2r_rc = ds4_mmq_q2_K_moe_d2r_launch(
                    x_soa, soa_blocks, src1_q8_1.get(), ids_dst.get(), expert_bounds.get(),
                    out_f32, M, K, ne_get_rows, n_experts, d2r_work.get(), d2r_work_bytes,
                    stream);
                if (d2r_rc == 0) {
                    return 0;
                }
            }
        }
    }

    const mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1.get(),
        /*ids_dst=*/ids_dst.get(),
        /*expert_bounds=*/expert_bounds.get(),
        /*dst=*/out_f32,
        /*ncols_x=*/ne00,
        /*nrows_x=*/(int64_t)M,
        /*ncols_dst=*/ne_get_rows,
        /*stride_row_x=*/s01,
        /*ncols_y=*/ne_get_rows,
        /*nrows_dst=*/s1,
        /*nchannels_x=*/(int64_t)n_experts,
        /*nchannels_y=*/(int64_t)n_experts,
        /*stride_channel_x=*/s02,
        /*stride_channel_y=*/s12_mmq,
        /*stride_channel_dst=*/(int64_t)0,
        /*nsamples_x=*/1,
        /*nsamples_y=*/1,
        /*stride_sample_x=*/0,
        /*stride_sample_y=*/s13_mmq,
        /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/ne_get_rows,
        /*x_soa=*/x_soa,
        /*soa_blocks=*/soa_blocks,
    };

    mul_mat_q_case<type>(*ctx, args, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case (moe) launch failed: %s\n", tag, cudaGetErrorString(err));
        return -4;
    }
    if (sanitize_out) {
        ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)ne_get_rows, stream);
    }
    return 0;
}

struct ds4_mmq_fused_down {
    const void  * W;
    const char  * W_soa;
    int64_t       soa_blocks;
    const float * router_weights;
    float       * mid_f32;
    float       * out;
    int           out_dim;
    float         clamp;
    bool          direct_gateup_q8;
    void        * input_q8_scratch;
    size_t        input_q8_scratch_bytes;
    void        * q8_scratch;
    size_t        q8_scratch_bytes;
    void        * work_scratch;
    size_t        work_scratch_bytes;
    /* flat-pool p5c: producer-emitted token-compact q8 of X (see the
     * fused_direct_soa doc in ds4_mmq.h); NULL = quantize internally. */
    const void  * input_q8_ext;
    size_t        input_q8_ext_bytes;
};
