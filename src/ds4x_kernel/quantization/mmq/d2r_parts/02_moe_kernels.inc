template <bool FullTile, int NFrag,
          typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void down_q2k_d2r_mainloop(
        float (&acc)[NFrag][TileC::ne],
        block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv) {
    static_assert(NFrag == 1 || NFrag == 2 || NFrag == 4 || NFrag == 8,
                  "unsupported Q2 D2R tile width");
    int guarded_nf_live = 0;
    if constexpr (!FullTile) {
        const int guarded_col_count = s_inv.col_count;
        guarded_nf_live = (guarded_col_count + 7) >> 3;
    }

#pragma unroll
    for (int pf = 0; pf < kStages; ++pf) {
        if (pf < s_inv.k128_iters) {
            if constexpr (FullTile) {
                static_assert(NFrag == kNFrag,
                              "full Q2 D2R path is the tuned 64-column kernel");
                issue_q8_prefetch_fast(s_q8, s_inv, d2r_q8_stage(pf), pf);
            } else if constexpr (NFrag == kNFrag) {
                issue_q8_prefetch<false>(s_q8, s_inv, d2r_q8_stage(pf), pf, d2r_tid());
            } else {
                issue_fused_q8_prefetch<false>(s_q8, s_inv, d2r_q8_stage(pf), pf);
            }
        }
    }

#pragma unroll
    for (int pf = 0; pf < kRawStages; ++pf) {
        if (pf < s_inv.nb) {
            if constexpr (FullTile) {
                issue_q2k_raw_prefetch_fast(s_raw, s_inv, d2r_raw_stage(pf), pf);
            } else {
                issue_q2k_raw_prefetch<false>(
                    s_raw, s_inv, d2r_raw_stage(pf), pf, d2r_warp(), d2r_lane());
            }
        }
    }

    for (int k128_iter = 0;; ++k128_iter) {
        if (k128_iter >= s_inv.k128_iters) {
            break;
        }
        if ((k128_iter & 1) == 0) {
            int keep_raw = 0;
            if constexpr (kRawStages > 1) {
                keep_raw = ((k128_iter >> 1) + 1 < s_inv.nb) ? 1 : 0;
            }
            cp_async_wait_keep(keep_raw);
        }
        __syncthreads();
        if constexpr (FullTile) {
            publish_q8_fix_f32(s_q8_fix, s_q8, d2r_q8_stage(k128_iter), d2r_tid());
        } else {
            publish_q8_fix_f32_guarded(
                s_q8_fix, s_q8, d2r_q8_stage(k128_iter), d2r_tid(), guarded_nf_live << 3);
        }
        __syncthreads();

        if constexpr (FullTile) {
            mma_fold_k128_k64_fast<TileA, TileB, TileC>(
                acc, s_raw, s_q8, s_q8_fix, k128_iter);
        } else {
            const int fold_warp = d2r_warp();
            const int fold_group = d2r_group();
            const int fold_warp_row0 = s_inv.cta_row0 + (fold_warp << 4);
            const int fold_M = s_inv.M;
            const int fold_col_lo = s_inv.col_lo;
            const int fold_col_hi = fold_col_lo + s_inv.col_count;
            const bool fold_row0_ok = (fold_warp_row0 + fold_group) < fold_M;
            const bool fold_row1_ok = (fold_warp_row0 + fold_group + 8) < fold_M;
            const int raw_stage = d2r_raw_stage(k128_iter >> 1);
            const int raw_half = k128_iter & 1;
            const int parity = fold_group & 1;
            mma_fold_k128_k64<TileA, TileB, TileC, false>(
                acc, s_raw, s_q8, s_q8_fix, raw_stage,
                fold_row0_ok, fold_row1_ok,
                parity, raw_half, fold_warp, fold_group, d2r_tig(),
                d2r_q8_stage(k128_iter), fold_warp_row0, fold_M, fold_col_lo, fold_col_hi,
                guarded_nf_live);
        }

        __syncthreads();
        const int pf_iter = k128_iter + kStages;
        if (pf_iter < s_inv.k128_iters) {
            if constexpr (FullTile) {
                issue_q8_prefetch_fast(s_q8, s_inv, d2r_q8_stage(pf_iter), pf_iter);
            } else if constexpr (NFrag == kNFrag) {
                issue_q8_prefetch<false>(
                    s_q8, s_inv, d2r_q8_stage(pf_iter), pf_iter, d2r_tid());
            } else {
                issue_fused_q8_prefetch<false>(
                    s_q8, s_inv, d2r_q8_stage(pf_iter), pf_iter);
            }
        }
        if ((k128_iter & 1) != 0) {
            const int raw_pf = (k128_iter >> 1) + kRawStages;
            if (raw_pf < s_inv.nb) {
                if constexpr (FullTile) {
                    issue_q2k_raw_prefetch_fast(s_raw, s_inv, d2r_raw_stage(raw_pf), raw_pf);
                } else {
                    issue_q2k_raw_prefetch<false>(
                        s_raw, s_inv, d2r_raw_stage(raw_pf), raw_pf, d2r_warp(), d2r_lane());
                }
            }
        }
    }
}


template <int TileN>
__global__ __launch_bounds__(kThreads, 2)
void d2r_build_worklist_kernel(const int32_t * __restrict__ expert_bounds,
                               int * __restrict__ work,
                               int * __restrict__ n_items_out,
                               int n_experts) {
    static_assert(TileN > 0, "worklist tile width must be positive");
    __shared__ int scan[kThreads];
    __shared__ int running;
    __shared__ int chunk_base;

    const int tid = (int)threadIdx.x;
    if (tid == 0) {
        running = 0;
    }
    __syncthreads();

    for (int base = 0; base < n_experts; base += kThreads) {
        const int expert = base + tid;
        int tiles = 0;
        if (expert < n_experts) {
            const int count = expert_bounds[expert + 1] - expert_bounds[expert];
            tiles = count > 0 ? ((count + TileN - 1) / TileN) : 0;
        }
        scan[tid] = tiles;
        __syncthreads();

#pragma unroll
        for (int offset = 1; offset < kThreads; offset <<= 1) {
            const int add = tid >= offset ? scan[tid - offset] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }

        if (tid == 0) {
            chunk_base = running;
        }
        __syncthreads();

        if (expert < n_experts && tiles > 0) {
            const int exclusive = tid == 0 ? 0 : scan[tid - 1];
            const int out_base = chunk_base + exclusive;
            for (int jt = 0; jt < tiles; ++jt) {
                work[out_base + jt] = (expert << 16) | jt;
            }
        }
        __syncthreads();

        if (tid == 0) {
            running += scan[kThreads - 1];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *n_items_out = running;
    }
}

template <int TileN, int BaseTileN>
__global__ __launch_bounds__(kThreads, 2)
void down_q2k_d2r_kernel(const void * __restrict__ W_soa,
                         const block_q8_1_mmq * __restrict__ q8,
                         const int32_t * __restrict__ ids_dst,
                         const int32_t * __restrict__ expert_bounds,
                         const int * __restrict__ work,
                         const int * __restrict__ n_items_ptr,
                         float * __restrict__ out,
                         int M, int K, int n_assign, int E) {
#if defined(TURING_MMA_AVAILABLE)
    const int n_items = *n_items_ptr;
    if ((int)blockIdx.y >= n_items) {
        return;
    }

    const int packed = work[blockIdx.y];
    const int expert = packed >> 16;
    const int jt = packed & 0xFFFF;
    if (expert >= E) {
        return;
    }
    static_assert(TileN == 8 || TileN == 16 || TileN == 32 || TileN == 64,
                  "unsupported Q2 D2R tile width");
    static_assert(BaseTileN == kNTile,
                  "Q2 D2R tail must follow the 64-column full tiles");
    constexpr int NFrag = TileN / 8;

    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    __shared__ __align__(16) block_q8_1_mmq s_q8[kStages][NFrag][8];
    __shared__ __align__(16) Q8ColFixF32 s_q8_fix[kStages][NFrag][8];
    // CFG1/NT64 static shared: q8 = 2*(8*8*144 + 8*8*40) = 23,552 B;
    // raw = 8 warps*2*(2*8*9*8 qs + 2*8*16 scales + 8*8 dm) = 23,552 B;
    // invariant table <= 128 B; total stays below the 48 KiB launch target. The
    // qs stride pads one uint2 per row-pair half so each lane's four raw uint2
    // reads rotate banks.
    __shared__ __align__(16) Q2KRawWarpStage s_raw[kWarps][kRawStages];
    __shared__ __align__(16) volatile SmemInvariants s_inv;
    /* Scatter indices staged up front so the epilogue's dependent STG chain
     * never waits on an ids_dst LDG (68% of this kernel's long-scoreboard
     * stalls sat on that load, cmd2rncu15b PC sampling). */
    __shared__ int s_out_cols[TileN];

    const int col_lo = expert_bounds[expert] + jt * BaseTileN;
    const int col_hi_full = expert_bounds[expert + 1];
    const int col_tile_hi =
        (col_hi_full < col_lo + TileN) ? col_hi_full : (col_lo + TileN);
    if (col_lo >= col_tile_hi) {
        return;
    }

    const int cta_row0 = (int)blockIdx.x * kMTile;
    const int warp_row0 = cta_row0 + d2r_warp() * 16;
    const int nb = K >> 8;

    if (d2r_tid() == 0) {
        const uint64_t npair = (uint64_t)E * (uint64_t)(M >> 1) * (uint64_t)nb;
        const uint64_t dm_bytes = (npair * 8ull + 63ull) & ~63ull;
        const uint64_t sc_bytes = (npair * 32ull + 63ull) & ~63ull;
        s_inv.w_base = (const char *)W_soa;
        s_inv.q8_tile_base = (const char *)q8 + (uint64_t)col_lo * sizeof(block_q8_1_mmq);
        s_inv.out = out;
        s_inv.sc_off_bytes = (uint32_t)dm_bytes;
        s_inv.qs_off_bytes = (uint32_t)(dm_bytes + sc_bytes);
        s_inv.q8_k128_stride_bytes = (uint32_t)((uint64_t)n_assign * sizeof(block_q8_1_mmq));
        s_inv.nb = nb;
        s_inv.k128_iters = K >> 7;
        s_inv.M = M;
        s_inv.cta_row0 = cta_row0;
        s_inv.col_lo = col_lo;
        s_inv.col_count = col_tile_hi - col_lo;
    }
    if (d2r_lane() == 0) {
        const uint64_t expert_row_pair = (uint64_t)expert * (uint64_t)(M >> 1);
        const uint64_t warp_pair0_base =
            (expert_row_pair + (uint64_t)(warp_row0 >> 1)) * (uint64_t)nb;
        s_inv.warp_pair0_blk[d2r_warp()] = (uint32_t)warp_pair0_base;
    }
    if (d2r_tid() < col_tile_hi - col_lo) {
        s_out_cols[d2r_tid()] = ids_dst[col_lo + d2r_tid()];
    }
    __syncthreads();

    float acc[NFrag][tile_C::ne] = {};

    if constexpr (TileN == kNTile) {
        const bool full_warp_tile =
            (warp_row0 + 15 < M) && (col_lo + TileN <= col_hi_full);
        if (full_warp_tile) {
            down_q2k_d2r_mainloop<true, NFrag, tile_A, tile_B, tile_C>(
                acc, s_q8, s_q8_fix, s_raw, s_inv);
        } else {
            down_q2k_d2r_mainloop<false, NFrag, tile_A, tile_B, tile_C>(
                acc, s_q8, s_q8_fix, s_raw, s_inv);
        }
    } else {
        down_q2k_d2r_mainloop<false, NFrag, tile_A, tile_B, tile_C>(
            acc, s_q8, s_q8_fix, s_raw, s_inv);
    }

    const int out_col_lo = s_inv.col_lo;
    const int out_col_hi = out_col_lo + s_inv.col_count;
    const int out_warp_row0 = s_inv.cta_row0 + (d2r_warp() << 4);
    const int out_M = s_inv.M;
    float *out_base = s_inv.out;
#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        const int col_frag0 = out_col_lo + nf * 8;
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = out_warp_row0 + tile_C::get_i(l);
            const int col = col_frag0 + tile_C::get_j(l);
            if (row < out_M && col < out_col_hi) {
                const int out_col = s_out_cols[col - out_col_lo];
                out_base[(uint64_t)out_col * (uint64_t)out_M + (uint64_t)row] = acc[nf][l];
            }
        }
    }
#else
    (void)W_soa;
    (void)q8;
    (void)ids_dst;
    (void)expert_bounds;
    (void)work;
    (void)n_items_ptr;
    (void)out;
    (void)M;
    (void)K;
    (void)n_assign;
    (void)E;
#endif
}

__global__ __launch_bounds__(kThreads, 2)
void gateup_iq2_d2r_pair_kernel(const void * __restrict__ gate_soa,
                                const void * __restrict__ up_soa,
                                const block_q8_1_mmq * __restrict__ q8,
                                const int32_t * __restrict__ ids_dst,
                                const int32_t * __restrict__ expert_bounds,
                                const int * __restrict__ work,
                                const int * __restrict__ n_items_ptr,
                                float * __restrict__ out_gate,
                                float * __restrict__ out_up,
                                int M, int K, int n_assign, int E) {
#if defined(TURING_MMA_AVAILABLE)
    const int n_items = *n_items_ptr;
    if ((int)blockIdx.y >= n_items) {
        return;
    }

    const int packed = work[blockIdx.y];
    const int expert = packed >> 16;
    const int jt = packed & 0xFFFF;
    const int leg = (int)blockIdx.z;
    if (expert >= E || leg >= 2) {
        return;
    }

    const int col_lo = expert_bounds[expert] + jt * kNTile;
    const int col_hi_full = expert_bounds[expert + 1];
    const int col_tile_hi = (col_hi_full < col_lo + kNTile) ? col_hi_full : (col_lo + kNTile);
    if (col_lo >= col_tile_hi) {
        return;
    }

    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    __shared__ __align__(16) block_q8_1_mmq s_q8[kStages][kNFrag][8];
    __shared__ __align__(16) IQ2RawWarpStage s_raw[kWarps][kRawStages];
    __shared__ __align__(16) uint2 s_grid[256];
    __shared__ __align__(16) volatile SmemInvariants s_inv;
    /* Same scatter-index staging as down_q2k_d2r_kernel (see comment there). */
    __shared__ int s_out_cols[kNTile];

    const void *W_soa = leg == 0 ? gate_soa : up_soa;
    float *out = leg == 0 ? out_gate : out_up;
    const int cta_row0 = (int)blockIdx.x * kMTile;
    const int warp_row0 = cta_row0 + d2r_warp() * 16;
    const int nb = K >> 8;

    const bool full_warp_tile = (warp_row0 + 15 < M) && (col_lo + kNTile <= col_hi_full);

    for (int i = d2r_tid(); i < 256; i += kThreads) {
        s_grid[i] = reinterpret_cast<const uint2 *>(iq2xxs_grid)[i];
    }
    if (d2r_tid() == 0) {
        const uint64_t nblk = (uint64_t)E * (uint64_t)M * (uint64_t)nb;
        const uint64_t dq_bytes = (nblk * 2ull + 63ull) & ~63ull;
        s_inv.w_base = (const char *)W_soa;
        s_inv.iq2_dq_base = reinterpret_cast<const half *>(W_soa);
        s_inv.iq2_qs_base =
            reinterpret_cast<const uint2 *>(reinterpret_cast<const char *>(W_soa) + dq_bytes);
        s_inv.q8_tile_base = reinterpret_cast<const char *>(q8) + (uint64_t)col_lo * sizeof(block_q8_1_mmq);
        s_inv.out = out;
        s_inv.sc_off_bytes = 0;
        s_inv.qs_off_bytes = 0;
        s_inv.q8_k128_stride_bytes = (uint32_t)((uint64_t)n_assign * sizeof(block_q8_1_mmq));
        s_inv.nb = nb;
        s_inv.k128_iters = K >> 7;
        s_inv.M = M;
        s_inv.cta_row0 = cta_row0;
        s_inv.col_lo = col_lo;
        s_inv.col_count = col_tile_hi - col_lo;
    }
    if (d2r_lane() == 0) {
        const uint64_t expert_row = (uint64_t)expert * (uint64_t)M;
        const uint64_t warp_row0_base = (expert_row + (uint64_t)warp_row0) * (uint64_t)nb;
        s_inv.warp_row0_blk[d2r_warp()] = (uint32_t)warp_row0_base;
    }
    if (d2r_tid() < col_tile_hi - col_lo) {
        s_out_cols[d2r_tid()] = ids_dst[col_lo + d2r_tid()];
    }
    __syncthreads();

    float acc[kNFrag][tile_C::ne] = {};

    if (full_warp_tile) {
        iq2_d2r_mainloop<true, tile_A, tile_B, tile_C>(
            acc, s_q8, s_raw, s_grid, s_inv);
    } else {
        iq2_d2r_mainloop<false, tile_A, tile_B, tile_C>(
            acc, s_q8, s_raw, s_grid, s_inv);
    }

    const int out_col_lo = s_inv.col_lo;
    const int out_col_hi = out_col_lo + s_inv.col_count;
    const int out_warp_row0 = s_inv.cta_row0 + (d2r_warp() << 4);
    const int out_M = s_inv.M;
    float *out_base = s_inv.out;
#pragma unroll
    for (int nf = 0; nf < kNFrag; ++nf) {
        const int col_frag0 = out_col_lo + nf * 8;
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = out_warp_row0 + tile_C::get_i(l);
            const int col = col_frag0 + tile_C::get_j(l);
            if (row < out_M && col < out_col_hi) {
                const int out_col = s_out_cols[col - out_col_lo];
                out_base[(uint64_t)out_col * (uint64_t)out_M + (uint64_t)row] = acc[nf][l];
            }
        }
    }
#else
    (void)gate_soa;
    (void)up_soa;
    (void)q8;
    (void)ids_dst;
    (void)expert_bounds;
    (void)work;
    (void)n_items_ptr;
    (void)out_gate;
    (void)out_up;
    (void)M;
    (void)K;
    (void)n_assign;
    (void)E;
#endif
}

template <int TileN, int BaseTileN>
__global__ __launch_bounds__(kThreads, 2)
void gateup_iq2_swiglu_q8_d2r_kernel(
        const void * __restrict__ gate_soa,
        const void * __restrict__ up_soa,
        const block_q8_1_mmq * __restrict__ input_q8,
        const int32_t * __restrict__ ids_src,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const float * __restrict__ router_weights,
        const int * __restrict__ work,
        const int * __restrict__ n_items_ptr,
        block_q8_1_mmq * __restrict__ down_q8,
        int M, int K, int n_assign, int n_tokens, int E, float clamp) {
#if defined(TURING_MMA_AVAILABLE)
    const int n_items = *n_items_ptr;
    if ((int)blockIdx.y >= n_items) {
        return;
    }

    const int packed = work[blockIdx.y];
    const int expert = packed >> 16;
    const int jt = packed & 0xFFFF;
    if (expert >= E) {
        return;
    }

    static_assert(TileN == 8 || TileN == 16 || TileN == 32,
                  "unsupported fused D2R tile width");
    static_assert(BaseTileN == kFusedNTile,
                  "fused D2R tail must follow the 32-column full tiles");
    constexpr int NFrag = TileN / 8;
    const int col_lo = expert_bounds[expert] + jt * BaseTileN;
    const int col_hi_full = expert_bounds[expert + 1];
    const int col_tile_hi =
        (col_hi_full < col_lo + TileN) ? col_hi_full : (col_lo + TileN);
    if (col_lo >= col_tile_hi) {
        return;
    }

    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    __shared__ FusedGateUpSmem<TileN> s_smem;
    __shared__ float s_route_weight[TileN];

    FusedGateUpComputeSmem<TileN> &s = s_smem.compute;
    const int tid = d2r_tid();
    const int cta_row0 = (int)blockIdx.x * kMTile;
    const int warp_row0 = cta_row0 + d2r_warp() * 16;
    const int nb = K >> 8;
    const int col_count = col_tile_hi - col_lo;
    const bool full_tile =
        cta_row0 + kMTile <= M && col_lo + TileN <= col_hi_full;

    for (int i = tid; i < 256; i += kThreads) {
        s.grid[i] = reinterpret_cast<const uint2 *>(iq2xxs_grid)[i];
    }
    if (tid == 0) {
        const void *weights[2] = {gate_soa, up_soa};
        const uint64_t nblk = (uint64_t)E * (uint64_t)M * (uint64_t)nb;
        const uint64_t dq_bytes = (nblk * 2ull + 63ull) & ~63ull;
#pragma unroll
        for (int leg = 0; leg < 2; ++leg) {
            s.inv[leg].w_base = (const char *)weights[leg];
            s.inv[leg].iq2_dq_base = reinterpret_cast<const half *>(weights[leg]);
            s.inv[leg].iq2_qs_base = reinterpret_cast<const uint2 *>(
                reinterpret_cast<const char *>(weights[leg]) + dq_bytes);
            /* p5b: with ids_src the input Q8 buffer is token-compact
             * (quantized once per token, n_tokens row stride) and the
             * column -> row map rides the per-thread y_off below; the
             * slot-gathered form keeps the column offset in the base. */
            s.inv[leg].q8_tile_base = reinterpret_cast<const char *>(input_q8) +
                                      (ids_src ? 0ull
                                               : (uint64_t)col_lo * sizeof(block_q8_1_mmq));
            s.inv[leg].out = nullptr;
            s.inv[leg].sc_off_bytes = 0;
            s.inv[leg].qs_off_bytes = 0;
            s.inv[leg].q8_k128_stride_bytes =
                (uint32_t)((uint64_t)(ids_src ? n_tokens : n_assign) *
                           sizeof(block_q8_1_mmq));
            s.inv[leg].nb = nb;
            s.inv[leg].k128_iters = K >> 7;
            s.inv[leg].M = M;
            s.inv[leg].cta_row0 = cta_row0;
            s.inv[leg].col_lo = col_lo;
            s.inv[leg].col_count = col_count;
        }
    }
    if (d2r_lane() == 0) {
        const uint64_t expert_row = (uint64_t)expert * (uint64_t)M;
        const uint64_t warp_row0_base =
            (expert_row + (uint64_t)warp_row0) * (uint64_t)nb;
        s.inv[0].warp_row0_blk[d2r_warp()] = (uint32_t)warp_row0_base;
        s.inv[1].warp_row0_blk[d2r_warp()] = (uint32_t)warp_row0_base;
    }
    if (tid < col_count) {
        const int pair = ids_dst[col_lo + tid];
        s_route_weight[tid] = router_weights[pair];
    }
    /* p5b: per-thread Y row byte offset (see issue_fused_q8_prefetch_one).
     * Out-of-segment columns are cp.async-predicated off, so their map
     * entry is never read; clamp to 0 to keep the ids_src load in range. */
    const int y_col_local = tid & (TileN - 1);
    uint32_t y_off;
    if (ids_src) {
        y_off = (y_col_local < col_count)
            ? (uint32_t)ids_src[col_lo + y_col_local] *
              (uint32_t)sizeof(block_q8_1_mmq)
            : 0u;
    } else {
        y_off = (uint32_t)y_col_local * (uint32_t)sizeof(block_q8_1_mmq);
    }
    __syncthreads();

    float gate_acc[NFrag][tile_C::ne] = {};
    float up_acc[NFrag][tile_C::ne] = {};
    if (full_tile) {
        iq2_gateup_fused_mainloop<true, TileN, tile_A, tile_B, tile_C>(
            gate_acc, up_acc, s, y_off);
    } else {
        iq2_gateup_fused_mainloop<false, TileN, tile_A, tile_B, tile_C>(
            gate_acc, up_acc, s, y_off);
    }

    /* The MMA rings are dead now. Reuse the same shared allocation as a
     * column-major float tile so the quantization reduction matches the
     * established Q8_1 D2S6 kernel: one warp owns one 128-row column. */
    __syncthreads();
#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = d2r_warp() * 16 + tile_C::get_i(l);
            const int col = nf * 8 + tile_C::get_j(l);
            if (cta_row0 + row < M && col < col_count) {
                float gate = isfinite(gate_acc[nf][l]) ? gate_acc[nf][l] : 0.0f;
                float up = isfinite(up_acc[nf][l]) ? up_acc[nf][l] : 0.0f;
                if (clamp > 1.0e-6f) {
                    gate = fminf(gate, clamp);
                    up = fminf(fmaxf(up, -clamp), clamp);
                }
                const float swiglu = (gate / (1.0f + expf(-gate))) * up;
                s_smem.mid[col][row] = swiglu * s_route_weight[col];
            }
        }
    }
    __syncthreads();

    const int lane = d2r_lane();
    for (int col = d2r_warp(); col < col_count; col += kWarps) {
        const float4 xi = *reinterpret_cast<const float4 *>(
            &s_smem.mid[col][lane * 4]);
        float amax = fabsf(xi.x);
        amax = fmaxf(amax, fabsf(xi.y));
        amax = fmaxf(amax, fabsf(xi.z));
        amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));
        }

        float sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, 32);
        }

        const float d_inv = 127.0f / amax;
        char4 q;
        q.x = roundf(xi.x * d_inv);
        q.y = roundf(xi.y * d_inv);
        q.z = roundf(xi.z * d_inv);
        q.w = roundf(xi.w * d_inv);

        block_q8_1_mmq &out_block =
            down_q8[(uint64_t)(cta_row0 / kMTile) * (uint64_t)n_assign +
                    (uint64_t)(col_lo + col)];
        reinterpret_cast<char4 *>(out_block.qs)[lane] = q;
        if ((lane & 3) == 0 && lane < 24) {
            out_block.d2s6[2 + lane / 4] = __float2half(sum);
        }
        if ((lane & 15) == 0) {
            out_block.d2s6[lane / 16] = __float2half(1.0f / d_inv);
        }
    }
#else
    (void)gate_soa;
    (void)up_soa;
    (void)input_q8;
    (void)ids_src;
    (void)ids_dst;
    (void)expert_bounds;
    (void)router_weights;
    (void)work;
    (void)n_items_ptr;
    (void)down_q8;
    (void)M;
    (void)K;
    (void)n_assign;
    (void)n_tokens;
    (void)E;
    (void)clamp;
#endif
}

static int64_t d2r_work_capacity_for_tile(
        int64_t ncols_max, int n_experts, int tile_n) {
    if (ncols_max <= 0 || n_experts <= 0 || tile_n <= 0) {
        return 0;
    }
    return (ncols_max + tile_n - 1) / tile_n + (int64_t)n_experts;
}

static int64_t d2r_work_capacity(int64_t ncols_max, int n_experts) {
    return d2r_work_capacity_for_tile(ncols_max, n_experts, kNTile);
}

} // namespace

