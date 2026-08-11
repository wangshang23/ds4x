bool ds4_mmq_q2_K_moe_d2r_available(int cc) {
    static int cached_cc = -1;
    static int cached = 0;
    if (cached_cc != cc) {
        cached_cc = cc;
        cached = (GGML_CUDA_CC_IS_NVIDIA(cc) &&
                  ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE) ? 1 : 0;
    }
    return cached != 0;
}

bool ds4_mmq_iq2_xxs_moe_d2r_available(int cc) {
    static int cached_cc = -1;
    static int cached = 0;
    if (cached_cc != cc) {
        cached_cc = cc;
        cached = (GGML_CUDA_CC_IS_NVIDIA(cc) &&
                  ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE) ? 1 : 0;
    }
    return cached != 0;
}

size_t ds4_mmq_q2_K_moe_d2r_scratch_bytes(int64_t ncols_max, int n_experts) {
    const int64_t capacity = d2r_work_capacity(ncols_max, n_experts);
    if (capacity <= 0 || capacity > (int64_t)(INT_MAX - 1)) {
        return 0;
    }
    return (size_t)capacity * sizeof(int) + sizeof(int);
}

size_t ds4_mmq_iq2_xxs_moe_d2r_pair_scratch_bytes(int64_t ncols_max, int n_experts) {
    const int64_t capacity = d2r_work_capacity(ncols_max, n_experts);
    if (capacity <= 0 || capacity > (int64_t)(INT_MAX - 1)) {
        return 0;
    }
    return (size_t)capacity * sizeof(int) + sizeof(int);
}

size_t ds4_mmq_iq2_xxs_moe_d2r_fused_scratch_bytes(
        int64_t ncols_max, int n_experts) {
    const int64_t capacity =
        d2r_work_capacity_for_tile(ncols_max, n_experts, kFusedNTile);
    if (capacity <= 0 || capacity > (int64_t)(INT_MAX - 1)) {
        return 0;
    }
    return (size_t)capacity * sizeof(int) + sizeof(int);
}

int ds4_mmq_q2_K_moe_d2r_launch(const void *W_soa,
                                int64_t soa_blocks,
                                const void *q8,
                                const int32_t *ids_dst,
                                const int32_t *expert_bounds,
                                float *out,
                                int M,
                                int K,
                                int64_t ne_get_rows,
                                int n_experts,
                                void *worklist_scratch,
                                size_t worklist_scratch_bytes,
                                cudaStream_t stream) {
    const char *tag = "ds4_mmq_q2_K_moe_d2r_launch";
    const int dev = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_q2_K_moe_d2r_available(cc)) {
        return 1;
    }
    if (!W_soa || !q8 || !ids_dst || !expert_bounds || !out || !worklist_scratch ||
        M <= 0 || (M & 1) != 0 || K <= 0 || K % 256 != 0 || ne_get_rows <= 0 ||
        ne_get_rows > INT_MAX || n_experts <= 0) {
        return -1;
    }

    const int64_t expected_soa_blocks = (int64_t)n_experts * (int64_t)(M >> 1) * (int64_t)(K >> 8);
    if (soa_blocks < expected_soa_blocks) {
        return -1;
    }

    const int64_t capacity64 = d2r_work_capacity(ne_get_rows, n_experts);
    if (capacity64 <= 0 || capacity64 > (int64_t)(INT_MAX - 1)) {
        return -1;
    }
    const size_t needed = (size_t)capacity64 * sizeof(int) + sizeof(int);
    if (worklist_scratch_bytes < needed) {
        return -1;
    }

    int *work = (int *)worklist_scratch;
    int *n_items = work + capacity64;

    /* Single guarded launch: at the serving chunk profile (2048-token
     * chunks -> ~48 assignments per expert bucket) nearly EVERY expert is a
     * ragged tail, so the tail-specialized 8/16/32/64 decomposition turns
     * one dense launch into five mostly-sparse ones and the relaunch +
     * occupancy-ramp tax dominates (reslice6: 64% of down time in tail
     * launches, +1.2 s/180k vs this shape).  The guarded 64-wide path
     * inside the kernel handles ragged tiles exactly as it always did. */
    d2r_build_worklist_kernel<kNTile><<<1, kThreads, 0, stream>>>(
        expert_bounds, work, n_items, n_experts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: worklist builder launch failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    if (d2r_stats_enabled()) {
        d2r_print_fill_stats(tag, expert_bounds, n_experts, ne_get_rows, stream);
    }

    /* Row tiles ride blockIdx.x (fastest) and the expert-ordered worklist
     * rides blockIdx.y, so consecutive CTAs share one col-tile's q8 window
     * and consecutive worklist items share one expert's weight slab -- the
     * proto's expert-major L2 schedule.  The flat col-tile-fastest order
     * re-read each q8 window through ~94 MB of intervening traffic (all
     * misses, lts hit 55% vs proto 84%). */
    const dim3 block(32, kWarps, 1);
    const unsigned row_tiles = (unsigned)((M + kMTile - 1) / kMTile);
    const dim3 grid(row_tiles, (unsigned)capacity64, 1);
    down_q2k_d2r_kernel<64, kNTile><<<grid, block, 0, stream>>>(
        W_soa, (const block_q8_1_mmq *)q8, ids_dst, expert_bounds,
        work, n_items, out,
        M, K, (int)ne_get_rows, n_experts);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: main kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

int ds4_mmq_iq2_xxs_moe_d2r_pair_launch(const void *gate_soa,
                                         const void *up_soa,
                                         int64_t soa_blocks,
                                         const void *q8,
                                         const int32_t *ids_dst,
                                         const int32_t *expert_bounds,
                                         float *out_gate,
                                         float *out_up,
                                         int M,
                                         int K,
                                         int64_t ne_get_rows,
                                         int n_experts,
                                         void *worklist_scratch,
                                         size_t worklist_scratch_bytes,
                                         cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_moe_d2r_pair_launch";
    const int dev = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_iq2_xxs_moe_d2r_available(cc)) {
        return 1;
    }
    if (!gate_soa || !up_soa || !q8 || !ids_dst || !expert_bounds || !out_gate || !out_up ||
        !worklist_scratch || M <= 0 || K <= 0 || K % 256 != 0 || ne_get_rows <= 0 ||
        ne_get_rows > INT_MAX || n_experts <= 0) {
        return -1;
    }

    const int64_t expected_soa_blocks =
        (int64_t)n_experts * (int64_t)M * (int64_t)(K >> 8);
    if (soa_blocks < expected_soa_blocks) {
        return -1;
    }

    const int64_t capacity64 = d2r_work_capacity(ne_get_rows, n_experts);
    if (capacity64 <= 0 || capacity64 > (int64_t)(INT_MAX - 1)) {
        return -1;
    }
    const size_t needed = (size_t)capacity64 * sizeof(int) + sizeof(int);
    if (worklist_scratch_bytes < needed) {
        return -1;
    }

    int *work = (int *)worklist_scratch;
    int *n_items = work + capacity64;

    d2r_build_worklist_kernel<kNTile><<<1, kThreads, 0, stream>>>(
        expert_bounds, work, n_items, n_experts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: worklist builder launch failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    /* Same expert-major schedule as the down launch (see comment there). */
    const dim3 grid((unsigned)((M + kMTile - 1) / kMTile), (unsigned)capacity64, 2);
    const dim3 block(32, kWarps, 1);
    gateup_iq2_d2r_pair_kernel<<<grid, block, 0, stream>>>(
        gate_soa, up_soa, (const block_q8_1_mmq *)q8, ids_dst, expert_bounds, work, n_items,
        out_gate, out_up, M, K, (int)ne_get_rows, n_experts);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: main kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

int ds4_mmq_iq2_xxs_moe_d2r_fused_launch(
        const void *gate_soa,
        const void *up_soa,
        int64_t soa_blocks,
        const void *input_q8,
        const int32_t *ids_src,
        int n_tokens,
        const int32_t *ids_dst,
        const int32_t *expert_bounds,
        const float *router_weights,
        void *down_q8,
        int M,
        int K,
        int64_t ne_get_rows,
        int n_experts,
        float clamp,
        void *worklist_scratch,
        size_t worklist_scratch_bytes,
        cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_moe_d2r_fused_launch";
    const int dev = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_iq2_xxs_moe_d2r_available(cc)) {
        return 1;
    }
    if (!gate_soa || !up_soa || !input_q8 || !ids_dst || !expert_bounds ||
        !router_weights || !down_q8 || !worklist_scratch ||
        M <= 0 || M % kMTile != 0 || K <= 0 || K % 256 != 0 ||
        ne_get_rows <= 0 || ne_get_rows > INT_MAX || n_experts <= 0) {
        return -1;
    }
    /* p5b: ids_src selects the token-compact input layout; the compact row
     * count must cover every map entry (entries are token indices). */
    if (ids_src && (n_tokens <= 0 || (int64_t)n_tokens > ne_get_rows)) {
        return -1;
    }

    const int64_t expected_soa_blocks =
        (int64_t)n_experts * (int64_t)M * (int64_t)(K >> 8);
    if (soa_blocks < expected_soa_blocks) {
        return -1;
    }

    const int64_t capacity32 =
        d2r_work_capacity_for_tile(ne_get_rows, n_experts, kFusedNTile);
    if (capacity32 <= 0 || capacity32 > (int64_t)(INT_MAX - 1)) {
        return -1;
    }
    const size_t needed = (size_t)capacity32 * sizeof(int) + sizeof(int);
    if (worklist_scratch_bytes < needed) {
        return -1;
    }

    int *work = (int *)worklist_scratch;
    int *n_items = work + capacity32;

    /* Single guarded launch (see the down launch comment): the upstream
     * fork's tail-specialized 8/16/32 decomposition assumes big expert
     * buckets (its 8k-token chunks -> ~192 cols/expert, full tiles dominate).
     * At OUR 2048-token serving chunks the mean bucket is ~48 cols, every
     * expert carries a ragged tail, and the three extra launches cost more
     * than the narrow-tile waste they avoid (reslice6: 22.6 of 47.5 s in
     * tail launches).  The kernel's per-CTA full_tile guard handles ragged
     * columns in the one launch. */
    d2r_build_worklist_kernel<kFusedNTile><<<1, kThreads, 0, stream>>>(
        expert_bounds, work, n_items, n_experts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: worklist builder launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    if (d2r_stats_enabled()) {
        d2r_print_fill_stats(tag, expert_bounds, n_experts, ne_get_rows, stream);
    }

    const dim3 block(32, kWarps, 1);
    const dim3 grid((unsigned)(M / kMTile), (unsigned)capacity32, 1);
    gateup_iq2_swiglu_q8_d2r_kernel<32, kFusedNTile><<<grid, block, 0, stream>>>(
        gate_soa, up_soa,
        (const block_q8_1_mmq *)input_q8,
        ids_src, ids_dst, expert_bounds, router_weights, work, n_items,
        (block_q8_1_mmq *)down_q8,
        M, K, (int)ne_get_rows, ids_src ? n_tokens : 0, n_experts, clamp);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: main kernel launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

