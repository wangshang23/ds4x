static bool ds4_mmq_take_scratch(
        void *base, size_t capacity, size_t *offset,
        size_t bytes, size_t alignment, void **result) {
    if (!base || !offset || !result || alignment == 0 ||
        (alignment & (alignment - 1)) != 0) {
        return false;
    }
    if (*offset > capacity) return false;
    const uintptr_t address = (uintptr_t)base + *offset;
    const size_t padding = (size_t)(-(uintptr_t)address) & (alignment - 1);
    if (padding > capacity - *offset) return false;
    const size_t aligned = *offset + padding;
    if (bytes > capacity - aligned) return false;
    *result = (char *)base + aligned;
    *offset = aligned + bytes;
    return true;
}

static bool ds4_mmq_scratch_overlaps(
        const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    const uintptr_t a_addr = (uintptr_t)a;
    const uintptr_t b_addr = (uintptr_t)b;
    return a_addr <= b_addr
        ? b_addr - a_addr < a_bytes
        : a_addr - b_addr < b_bytes;
}

// Produce the weighted SwiGLU rows in their canonical pair-major order. The
// proven upstream quantizer below gathers them through the already available
// ids_dst map, so gate/up and down share one expert-major schedule without a
// second mm_ids_helper.
static __global__ void ds4_swiglu_weighted_f32(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        const float * __restrict__ router_weights,
        float * __restrict__ mid,
        uint64_t n,
        int K,
        float clamp) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t pair = i / (uint64_t)K;
    float g = isfinite(gate[i]) ? gate[i] : 0.0f;
    float u = isfinite(up[i]) ? up[i] : 0.0f;
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    mid[i] = (g / (1.0f + expf(-g))) * u * router_weights[pair];
}

// Paired MoE: one helper + one quantize covers both weights.  See the
// header comment on ds4_mmq_iq2_xxs_moe_pair for motivation.  Internal
// structure mirrors ds4_mmq_moe_impl above; the only differences are the
// two W pointers, the two output pointers, and the second mul_mat_q_case
// launch with a fresh (x, dst) pair.
template <ggml_type type, bool profile_fused_prefill = false>
int ds4_mmq_moe_pair_impl(
        const char    * tag,
        const void    * W_a,
        const void    * W_b,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_a,
        float         * out_b,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream,
        /* ds4 (P4 Inc3): optional aligned-SoA artifacts for W_a / W_b (same
         * shape, so one block count); see ds4_mmq_moe_impl. */
        const char    * xa_soa     = NULL,
        const char    * xb_soa     = NULL,
        int64_t         soa_blocks = 0,
        /* ds4 (P3): see ds4_mmq_moe_impl. */
        bool            sanitize_out = true,
        const ds4_mmq_fused_down *fused_down = nullptr) {

    const bool direct_gateup_q8 =
        fused_down != nullptr && fused_down->direct_gateup_q8;
    if (!W_a || !W_b || !X_f32 || !ids ||
        (!direct_gateup_q8 && (!out_a || !out_b))) {
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
    if (fused_down &&
        (type != GGML_TYPE_IQ2_XXS || !fused_down->W ||
         !fused_down->router_weights ||
         (!direct_gateup_q8 && !fused_down->mid_f32) ||
         (direct_gateup_q8 &&
          (!xa_soa || !xb_soa || !fused_down->W_soa ||
           !fused_down->input_q8_scratch ||
           fused_down->input_q8_scratch_bytes == 0 ||
           !fused_down->q8_scratch || fused_down->q8_scratch_bytes == 0 ||
           !fused_down->work_scratch || fused_down->work_scratch_bytes == 0)) ||
         !fused_down->out || fused_down->out_dim <= 0 || M % 256 != 0)) {
        fprintf(stderr, "%s: invalid fused Q2_K down configuration\n", tag);
        return -1;
    }

    const bool nvtx_prefill = profile_fused_prefill &&
                              fused_down != nullptr &&
                              n_tokens >= 1024 &&
                              ds4_mmq_nvtx_requested();
    ds4_mmq_nvtx_scope fused_scope(
            "ds4/prefill/moe/mmq_fused",
            ds4_mmq_nvtx_payload((uint32_t)n_tokens, (uint32_t)n_expert_used),
            nvtx_prefill);

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
    const int64_t ne11         = 1;
    const int64_t ne12         = n_tokens;
    const int64_t blck         = ggml_blck_size(type);
    const int64_t s01          = (int64_t)K / blck;
    const int64_t s02          = (int64_t)M * s01;

    ggml_cuda_pool_alloc<int32_t> ids_src1_alloc;
    ggml_cuda_pool_alloc<int32_t> ids_dst_alloc;
    ggml_cuda_pool_alloc<int32_t> expert_bounds_alloc;
    int32_t *ids_src1 = nullptr;
    int32_t *ids_dst = nullptr;
    int32_t *expert_bounds = nullptr;
    void *direct_work = nullptr;
    size_t direct_work_bytes = 0;

    const bool use_native_fp4 =
        type == GGML_TYPE_MXFP4 && blackwell_mma_available(cc);
    const size_t y_block_size = use_native_fp4
        ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4
        ? QK_FP4_MMQ : 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne_get_rows * ne10_padded * y_block_size / y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
    size_t direct_down_q8_bytes = 0;
    if (direct_gateup_q8) {
        const int64_t down_ne10_padded = GGML_PAD((int64_t)M, MATRIX_ROW_PADDING);
        direct_down_q8_bytes =
            (size_t)ne_get_rows * (size_t)down_ne10_padded *
                sizeof(block_q8_1) / QK8_1 +
            (size_t)get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
        const size_t gateup_work_bytes =
            ds4_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(
                ne_get_rows, n_experts);
        const size_t down_work_bytes =
            ds4_mmq_q2_K_moe_d2r_scratch_bytes(ne_get_rows, n_experts);
        if (fused_down->input_q8_scratch_bytes < nbytes_src1_q8_1) return -91;
        if (fused_down->q8_scratch_bytes < direct_down_q8_bytes) return -92;
        if (gateup_work_bytes == 0 || down_work_bytes == 0) return -93;
        if (!d2r_enabled() || !d2r_iq2_enabled()) return -94;
        if (ne_get_rows < d2r_min_cols()) return -95;
        if (!ds4_mmq_iq2_xxs_moe_d2r_available(cc) ||
            !ds4_mmq_q2_K_moe_d2r_available(cc)) {
            return -96;
        }

        size_t offset = 0;
        void *ids_src1_raw = nullptr;
        void *ids_dst_raw = nullptr;
        void *expert_bounds_raw = nullptr;
        direct_work_bytes = gateup_work_bytes > down_work_bytes
            ? gateup_work_bytes : down_work_bytes;
        if (!ds4_mmq_take_scratch(
                fused_down->work_scratch, fused_down->work_scratch_bytes,
                &offset, (size_t)ne_get_rows * sizeof(int32_t), 256,
                &ids_src1_raw) ||
            !ds4_mmq_take_scratch(
                fused_down->work_scratch, fused_down->work_scratch_bytes,
                &offset, (size_t)ne_get_rows * sizeof(int32_t), 256,
                &ids_dst_raw) ||
            !ds4_mmq_take_scratch(
                fused_down->work_scratch, fused_down->work_scratch_bytes,
                &offset, (size_t)(n_experts + 1) * sizeof(int32_t), 256,
                &expert_bounds_raw) ||
            !ds4_mmq_take_scratch(
                fused_down->work_scratch, fused_down->work_scratch_bytes,
                &offset, direct_work_bytes, 256, &direct_work)) {
            return -97;
        }
        ids_src1 = (int32_t *)ids_src1_raw;
        ids_dst = (int32_t *)ids_dst_raw;
        expert_bounds = (int32_t *)expert_bounds_raw;
    } else {
        ids_src1 = ids_src1_alloc.alloc(ctx->pool(), ne_get_rows);
        ids_dst = ids_dst_alloc.alloc(ctx->pool(), ne_get_rows);
        expert_bounds = expert_bounds_alloc.alloc(ctx->pool(), n_experts + 1);
    }

    const int si1  = n_expert_used;
    const int sis1 = 1;

    // Same cap guard as ds4_mmq_moe_impl (see comment there): past the smem
    // cap the launcher takes the bit-identical global variant (P5); only
    // refuse with DS4_MMID_LARGE=0.
    if ((size_t)n_tokens * 4u > ggml_cuda_info().devices[dev].smpbo && !ds4_mmid_large_enabled()) {
        fprintf(stderr, "%s: n_tokens=%d exceeds mm_ids_helper shared-mem cap; falling back\n",
                tag, n_tokens);
        return -1;
    }

    cudaError_t err = cudaSuccess;
    {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/expert_map",
                ds4_mmq_nvtx_payload((uint32_t)n_tokens, (uint32_t)n_experts),
                nvtx_prefill);
        // Task #22 root-cause fix (same as ds4_mmq_moe_impl): zero the id maps
        // so entries dropped by mm_ids_helper never expose stale pool memory.
        cudaMemsetAsync(ids_src1, 0, ne_get_rows * sizeof(int32_t), stream);
        cudaMemsetAsync(ids_dst,  0, ne_get_rows * sizeof(int32_t), stream);
        ggml_cuda_launch_mm_ids_helper(
            ids, ids_src1, ids_dst, expert_bounds,
            n_experts, n_tokens, n_expert_used, /*nchannels_y=*/(int)ne11,
            si1, sis1, stream);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mm_ids_helper failed: %s\n", tag, cudaGetErrorString(err));
            return -2;
        }
    }

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);
    /* The fused target-prefill path receives a true top-k assignment: one
     * token cannot select the same expert twice, so no expert bucket can
     * exceed n_tokens rows. Keep the conservative gathered-row bound for all
     * generic MMQ callers, including DSpark/MTP. */
    const int64_t routed_ncols_max = fused_down
        ? (int64_t)n_tokens
        : ne_get_rows;

    /* The materialized path stream-frees gate/up Q8_1 before allocating the
     * down Q8_1. The direct path needs both simultaneously, but writes down
     * Q8_1 into caller-owned gate scratch instead of growing the CUDA pool. */
    {
    ggml_cuda_pool_alloc<char> src1_q8_1_alloc;
    char *src1_q8_1 = direct_gateup_q8
        ? (char *)fused_down->input_q8_scratch
        : src1_q8_1_alloc.alloc(ctx->pool(), nbytes_src1_q8_1);

    // S1.1a fix (same as the dense/moe paths): zero the over-allocated mmq Y buffer
    // so the kernel's unconditional masked-out tail-tile read (mmq.cuh:3528) returns
    // a deterministic zero instead of allocator-perturbation-dependent stale memory.
    const int64_t s11_src = (int64_t)K;
    const int64_t s12_src = (int64_t)K * ne11;
    const int64_t s13_src = (int64_t)K * ne11 * ne12;
    /* p5b: token-compact input quantize on the direct fused path (the
     * materialized paths below keep the slot-gathered form: their pair/mmq
     * consumers address Y by assignment slot).
     * p5c: a producer-emitted token-compact buffer replaces even the
     * compact quantize — same layout by construction (ib = kseg*n_tokens
     * + row), so the p5b indirection consumes it unchanged. */
    const bool moe_yind = direct_gateup_q8 && moe_yind_enabled();
    const void *input_q8_ext = nullptr;
    if (moe_yind && fused_down && fused_down->input_q8_ext) {
        const size_t ext_need =
            (size_t)n_tokens * (size_t)ne10_padded * sizeof(block_q8_1) / QK8_1;
        if (fused_down->input_q8_ext_bytes >= ext_need) {
            input_q8_ext = fused_down->input_q8_ext;
        }
    }
    if (input_q8_ext) {
        src1_q8_1 = (char *)const_cast<void *>(input_q8_ext);
        static bool ext_logged = false;
        if (!ext_logged) {
            ext_logged = true;
            fprintf(stderr, "ds4: moe gateup consuming producer q8 "
                    "(flat-pool p5c, first n_tokens=%d)\n", n_tokens);
        }
    }
    const int64_t quant_rows = moe_yind ? (int64_t)n_tokens : ne_get_rows;
    if (!input_q8_ext) {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/input_quant_q8_1",
                ds4_mmq_nvtx_payload((uint32_t)quant_rows, (uint32_t)K),
                nvtx_prefill);
        ybuf_memset(src1_q8_1, nbytes_src1_q8_1, stream);
        if (use_native_fp4) {
            quantize_mmq_fp4_cuda(
                X_f32, moe_yind ? nullptr : ids_src1, (void *)src1_q8_1,
                type, /*ne00=*/K, s11_src, s12_src, s13_src,
                /*ne0=*/ne10_padded, /*ne1=*/quant_rows, /*ne2=*/1, /*ne3=*/1,
                stream);
        } else {
            quantize_mmq_q8_1_cuda(
                X_f32, moe_yind ? nullptr : ids_src1, (void *)src1_q8_1,
                type, /*ne00=*/K, s11_src, s12_src, s13_src,
                /*ne0=*/ne10_padded, /*ne1=*/quant_rows, /*ne2=*/1, /*ne3=*/1,
                stream);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: MMQ activation quantize failed: %s\n",
                    tag, cudaGetErrorString(err));
            return -3;
        }
    }

    if (direct_gateup_q8) {
        if (moe_yind) {
            static bool yind_logged = false;
            if (!yind_logged) {
                yind_logged = true;
                fprintf(stderr, "ds4: moe gateup y-indirect q8 staging engage "
                        "(flat-pool p5b, first n_tokens=%d n_assign=%lld)\n",
                        n_tokens, (long long)ne_get_rows);
            }
        }
        if (moe_yind && moe_yind_verify_enabled()) {
            /* In-situ byte-diff (p5a VERIFY pattern): run the slot-gathered
             * reference quantize into a temp buffer and compare every
             * assignment slot's blocks against the token-compact buffer
             * through ids_src1.  Instrument only - synchronous. */
            const size_t blk = sizeof(block_q8_1_mmq);
            const int nkseg = (int)(ne10_padded / (4 * QK8_1));
            const size_t ref_payload = (size_t)nkseg * (size_t)ne_get_rows * blk;
            const size_t cmp_payload = (size_t)nkseg * (size_t)n_tokens * blk;
            char *ref = nullptr;
            if (cudaMalloc((void **)&ref, ref_payload) == cudaSuccess) {
                quantize_mmq_q8_1_cuda(
                    X_f32, ids_src1, (void *)ref,
                    type, K, s11_src, s12_src, s13_src,
                    ne10_padded, ne_get_rows, 1, 1, stream);
                char *h_ref = (char *)malloc(ref_payload);
                char *h_cmp = (char *)malloc(cmp_payload);
                int32_t *h_ids = (int32_t *)malloc((size_t)ne_get_rows * sizeof(int32_t));
                if (h_ref && h_cmp && h_ids) {
                    cudaMemcpyAsync(h_ref, ref, ref_payload, cudaMemcpyDeviceToHost, stream);
                    cudaMemcpyAsync(h_cmp, src1_q8_1, cmp_payload, cudaMemcpyDeviceToHost, stream);
                    cudaMemcpyAsync(h_ids, ids_src1, (size_t)ne_get_rows * sizeof(int32_t),
                                    cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    long long bad = 0;
                    long long first_slot = -1, first_kseg = -1;
                    for (int ks = 0; ks < nkseg; ++ks) {
                        for (int64_t slot = 0; slot < ne_get_rows; ++slot) {
                            const char *a = h_ref + ((size_t)ks * (size_t)ne_get_rows + (size_t)slot) * blk;
                            const char *b = h_cmp + ((size_t)ks * (size_t)n_tokens + (size_t)h_ids[slot]) * blk;
                            if (memcmp(a, b, blk) != 0) {
                                if (first_slot < 0) { first_slot = slot; first_kseg = ks; }
                                ++bad;
                            }
                        }
                    }
                    fprintf(stderr, "ds4: moe yind VERIFY n_tokens=%d n_assign=%lld ksegs=%d "
                            "bad=%lld/%lld first_slot=%lld first_kseg=%lld\n",
                            n_tokens, (long long)ne_get_rows, nkseg,
                            bad, (long long)nkseg * (long long)ne_get_rows,
                            first_slot, first_kseg);
                }
                free(h_ref); free(h_cmp); free(h_ids);
                cudaFree(ref);
            }
        }
        ybuf_memset(fused_down->q8_scratch, direct_down_q8_bytes, stream);
        const size_t gateup_work_bytes =
            ds4_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(
                ne_get_rows, n_experts);
        if (gateup_work_bytes == 0) {
            return -9;
        }
        {
            ds4_mmq_nvtx_scope stage(
                    "ds4/prefill/moe/iq2_gate_up_swiglu_q8_d2r",
                    ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                    nvtx_prefill);
            const int d2r_rc = ds4_mmq_iq2_xxs_moe_d2r_fused_launch(
                    xa_soa, xb_soa, soa_blocks,
                    src1_q8_1,
                    moe_yind ? ids_src1 : nullptr,
                    moe_yind ? n_tokens : 0,
                    ids_dst, expert_bounds,
                    fused_down->router_weights, fused_down->q8_scratch,
                    M, K, ne_get_rows, n_experts, fused_down->clamp,
                    direct_work, gateup_work_bytes, stream);
            if (d2r_rc != 0) {
                return -10;
            }
        }

        if (out_memset_enabled()) {
            cudaMemsetAsync(fused_down->out, 0,
                    (size_t)fused_down->out_dim * (size_t)ne_get_rows * sizeof(float),
                    stream);
        }
        const size_t down_work_bytes =
            ds4_mmq_q2_K_moe_d2r_scratch_bytes(ne_get_rows, n_experts);
        if (down_work_bytes == 0) {
            return -11;
        }
        {
            ds4_mmq_nvtx_scope stage(
                    "ds4/prefill/moe/q2_down_d2r",
                    ds4_mmq_nvtx_payload((uint32_t)ne_get_rows,
                                         (uint32_t)fused_down->out_dim),
                    nvtx_prefill);
            const int down_rc = ds4_mmq_q2_K_moe_d2r_launch(
                    fused_down->W_soa,
                    fused_down->soa_blocks,
                    fused_down->q8_scratch,
                    ids_dst, expert_bounds,
                    fused_down->out,
                    fused_down->out_dim, M, ne_get_rows, n_experts,
                    direct_work, down_work_bytes, stream);
            if (down_rc != 0) {
                return -12;
            }
        }
        return 0;
    }

    const int64_t s1      = (int64_t)M;
    const int64_t s12_mmq = ne11 * ne10_padded * y_block_size /
                            (y_values_per_block * sizeof(int));
    const int64_t s13_mmq = ne12 * s12_mmq;

    if (out_memset_enabled()) {
        cudaMemsetAsync(out_a, 0, (size_t)M * (size_t)ne_get_rows * sizeof(float), stream);
        cudaMemsetAsync(out_b, 0, (size_t)M * (size_t)ne_get_rows * sizeof(float), stream);
    }

    bool gate_up_done = false;
    if (type == GGML_TYPE_IQ2_XXS && xa_soa != nullptr && xb_soa != nullptr &&
        d2r_enabled() && d2r_iq2_enabled() && K % 256 == 0 &&
        ne_get_rows >= d2r_min_cols()) {
        static int d2r_iq2_avail_cc = -1;
        static int d2r_iq2_avail = 0;
        if (d2r_iq2_avail_cc != cc) {
            d2r_iq2_avail_cc = cc;
            d2r_iq2_avail = ds4_mmq_iq2_xxs_moe_d2r_available(cc) ? 1 : 0;
        }
        if (d2r_iq2_avail) {
            const size_t d2r_work_bytes =
                ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(ne_get_rows, n_experts);
            if (d2r_work_bytes != 0) {
                ggml_cuda_pool_alloc<char> d2r_work(ctx->pool(), d2r_work_bytes);
                ds4_mmq_nvtx_scope stage(
                        "ds4/prefill/moe/iq2_gate_up_d2r",
                        ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                        nvtx_prefill);
                const int d2r_rc = ds4_mmq_iq2_xxs_moe_d2r_pair_launch(
                        xa_soa, xb_soa, soa_blocks, src1_q8_1, ids_dst,
                        expert_bounds, out_a, out_b, M, K, ne_get_rows, n_experts,
                        d2r_work.get(), d2r_work_bytes, stream);
                if (d2r_rc == 0) {
                    gate_up_done = true;
                }
            }
        }
    }

    if (!gate_up_done) {
    mmq_args args = {
        /*x=*/(const char *)W_a,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1,
        /*ids_dst=*/ids_dst,
        /*expert_bounds=*/expert_bounds,
        /*dst=*/out_a,
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
        /*ncols_max=*/routed_ncols_max,
        /*x_soa=*/xa_soa,
        /*soa_blocks=*/soa_blocks,
    };

    {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/iq2_gate",
                ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                nvtx_prefill);
        mul_mat_q_case<type>(*ctx, args, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_q_case (pair a) launch failed: %s\n", tag, cudaGetErrorString(err));
            return -4;
        }
    }

    // Second matmul over the same activation buffer and same routing map.
    args.x     = (const char *)W_b;
    args.dst   = out_b;
    args.x_soa = xb_soa;
    {
        ds4_mmq_nvtx_scope stage(
                "ds4/prefill/moe/iq2_up",
                ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                nvtx_prefill);
        mul_mat_q_case<type>(*ctx, args, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_q_case (pair b) launch failed: %s\n", tag, cudaGetErrorString(err));
            return -5;
        }
    }
    }
    }

    if (fused_down) {
        const int64_t down_ne10_padded = GGML_PAD((int64_t)M, MATRIX_ROW_PADDING);
        const size_t logical_q8_bytes =
            (size_t)ne_get_rows * (size_t)down_ne10_padded * sizeof(block_q8_1) / QK8_1;
        const size_t tail_q8_bytes =
            (size_t)get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> down_q8_1(
            ctx->pool(), logical_q8_bytes + tail_q8_bytes);

        const uint64_t mid_values = (uint64_t)ne_get_rows * (uint64_t)M;
        {
            ds4_mmq_nvtx_scope stage(
                    "ds4/prefill/moe/swiglu_down_quant",
                    ds4_mmq_nvtx_payload((uint32_t)ne_get_rows, (uint32_t)M),
                    nvtx_prefill);
            ybuf_memset(down_q8_1.get(), logical_q8_bytes + tail_q8_bytes, stream);
            ds4_swiglu_weighted_f32<<<
                (uint32_t)((mid_values + 255u) / 256u), 256, 0, stream>>>(
                    out_a, out_b, fused_down->router_weights,
                    fused_down->mid_f32, mid_values, M, fused_down->clamp);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: weighted SwiGLU launch failed: %s\n",
                        tag, cudaGetErrorString(err));
                return -6;
            }

            quantize_mmq_q8_1_cuda(
                fused_down->mid_f32, ids_dst, (void *)down_q8_1.get(),
                GGML_TYPE_Q2_K, /*ne00=*/M, /*s01=*/M,
                /*s02=*/(int64_t)M, /*s03=*/(int64_t)M * ne_get_rows,
                /*ne0=*/down_ne10_padded, /*ne1=*/ne_get_rows,
                /*ne2=*/1, /*ne3=*/1, stream);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: down quantize_mmq_q8_1_cuda failed: %s\n",
                        tag, cudaGetErrorString(err));
                return -7;
            }
        }

        if (out_memset_enabled()) {
            cudaMemsetAsync(fused_down->out, 0,
                    (size_t)fused_down->out_dim * (size_t)ne_get_rows * sizeof(float),
                    stream);
        }
        const int64_t down_s01 = (int64_t)M / ggml_blck_size(GGML_TYPE_Q2_K);
        const int64_t down_s02 = (int64_t)fused_down->out_dim * down_s01;
        const int64_t down_s12 =
            down_ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const mmq_args down_args = {
            /*x=*/(const char *)fused_down->W,
            /*type_x=*/GGML_TYPE_Q2_K,
            /*y=*/(const int *)down_q8_1.get(),
            /*ids_dst=*/ids_dst,
            /*expert_bounds=*/expert_bounds,
            /*dst=*/fused_down->out,
            /*ncols_x=*/(int64_t)M,
            /*nrows_x=*/(int64_t)fused_down->out_dim,
            /*ncols_dst=*/ne_get_rows,
            /*stride_row_x=*/down_s01,
            /*ncols_y=*/ne_get_rows,
            /*nrows_dst=*/(int64_t)fused_down->out_dim,
            /*nchannels_x=*/(int64_t)n_experts,
            /*nchannels_y=*/(int64_t)n_experts,
            /*stride_channel_x=*/down_s02,
            /*stride_channel_y=*/down_s12,
            /*stride_channel_dst=*/(int64_t)0,
            /*nsamples_x=*/1,
            /*nsamples_y=*/1,
            /*stride_sample_x=*/0,
            /*stride_sample_y=*/ne_get_rows * down_s12,
            /*stride_sample_dst=*/0,
            /*use_stream_k=*/use_stream_k,
            /*ncols_max=*/routed_ncols_max,
            /*x_soa=*/fused_down->W_soa,
            /*soa_blocks=*/fused_down->soa_blocks,
        };
        bool down_done = false;
        if (fused_down->W_soa != nullptr && d2r_enabled() &&
            ne_get_rows >= d2r_min_cols() &&
            ds4_mmq_q2_K_moe_d2r_available(cc)) {
            const size_t work_bytes =
                ds4_mmq_q2_K_moe_d2r_scratch_bytes(ne_get_rows, n_experts);
            if (work_bytes != 0u) {
                ggml_cuda_pool_alloc<char> work(ctx->pool(), work_bytes);
                ds4_mmq_nvtx_scope stage(
                        "ds4/prefill/moe/q2_down_d2r",
                        ds4_mmq_nvtx_payload((uint32_t)ne_get_rows,
                                             (uint32_t)fused_down->out_dim),
                        nvtx_prefill);
                down_done = ds4_mmq_q2_K_moe_d2r_launch(
                        fused_down->W_soa,
                        fused_down->soa_blocks,
                        down_q8_1.get(),
                        ids_dst,
                        expert_bounds,
                        fused_down->out,
                        fused_down->out_dim,
                        M,
                        ne_get_rows,
                        n_experts,
                        work.get(),
                        work_bytes,
                        stream) == 0;
            }
        }
        if (!down_done) {
            ds4_mmq_nvtx_scope stage(
                    "ds4/prefill/moe/q2_down",
                    ds4_mmq_nvtx_payload((uint32_t)ne_get_rows,
                                         (uint32_t)fused_down->out_dim),
                    nvtx_prefill);
            mul_mat_q_case<GGML_TYPE_Q2_K>(*ctx, down_args, stream);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "%s: fused Q2_K down launch failed: %s\n",
                        tag, cudaGetErrorString(err));
                return -8;
            }
        }
    }
    if (sanitize_out) {
        ds4_mmq_sanitize_f32(out_a, (uint64_t)M * (uint64_t)ne_get_rows, stream);
        ds4_mmq_sanitize_f32(out_b, (uint64_t)M * (uint64_t)ne_get_rows, stream);
    }
    return 0;
}

} // anonymous namespace

