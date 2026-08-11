template <int T, typename TileA>
__device__ __forceinline__ void make_A_tile(TileA &A, const Q2KWeightHalf &w) {
    const uint8_t sc00 = q2k_scale_from_packs_t<T, 0>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint8_t sc01 = q2k_scale_from_packs_t<T, 1>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint8_t sc10 = q2k_scale_from_packs_t<T, 0>(w.sc_r1_lo4, w.sc_r1_hi4);
    const uint8_t sc11 = q2k_scale_from_packs_t<T, 1>(w.sc_r1_lo4, w.sc_r1_hi4);
    A.x[0] = (int)q2k_decode_scaled_reg_t<T>(w.q_r0_c0, sc00);
    A.x[1] = (int)q2k_decode_scaled_reg_t<T>(w.q_r1_c0, sc10);
    A.x[2] = (int)q2k_decode_scaled_reg_t<T>(w.q_r0_c1, sc01);
    A.x[3] = (int)q2k_decode_scaled_reg_t<T>(w.q_r1_c1, sc11);
}

template <int T>
__device__ __forceinline__ uint16_t min_pack_for_t(uint32_t pack_lo4, uint32_t pack_hi4) {
    const uint8_t sc0 = q2k_scale_from_packs_t<T, 0>(pack_lo4, pack_hi4);
    const uint8_t sc1 = q2k_scale_from_packs_t<T, 1>(pack_lo4, pack_hi4);
    return (uint16_t)((uint16_t)(sc0 >> 4) | ((uint16_t)(sc1 >> 4) << 8));
}

template <int T0, int T1>
__device__ __forceinline__ uint32_t min_pack4_for_pair(uint32_t pack_lo4, uint32_t pack_hi4) {
    const uint8_t sc0 = q2k_scale_from_packs_t<T0, 0>(pack_lo4, pack_hi4);
    const uint8_t sc1 = q2k_scale_from_packs_t<T0, 1>(pack_lo4, pack_hi4);
    const uint8_t sc2 = q2k_scale_from_packs_t<T1, 0>(pack_lo4, pack_hi4);
    const uint8_t sc3 = q2k_scale_from_packs_t<T1, 1>(pack_lo4, pack_hi4);
    return (uint32_t)(sc0 >> 4) |
           ((uint32_t)(sc1 >> 4) << 8) |
           ((uint32_t)(sc2 >> 4) << 16) |
           ((uint32_t)(sc3 >> 4) << 24);
}

template <int NFrag, typename TileB>
__device__ __forceinline__ void load_B_tile(
        TileB &B,
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        int stage, int nf, int k_in_q8) {
    const int *base = reinterpret_cast<const int *>(&s_q8[stage][nf][0].qs[k_in_q8]);
    ggml_cuda_mma::load_ldmatrix(B, base, sizeof(block_q8_1_mmq) / sizeof(int));
}

struct Q8D4K64PairF32 {
    float2 c0;
    float2 c1;
};

template <int T0, int NFrag>
__device__ __forceinline__ Q8D4K64PairF32 load_q8_d4_k64_pair(
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        int stage, int nf, int c0, int c1) {
    static_assert((T0 & 1) == 0, "expected even k32 base for k64 d4 load");
    constexpr int d4_base = T0 & 2;
    Q8D4K64PairF32 d;
    d.c0 = *reinterpret_cast<const float2 *>(&s_q8[stage][nf][c0].d4[d4_base]);
    d.c1 = *reinterpret_cast<const float2 *>(&s_q8[stage][nf][c1].d4[d4_base]);
    return d;
}

template <int T>
__device__ __forceinline__ float q8_d4_k64_slot(const float2 &d) {
    if constexpr ((T & 1) == 0) {
        return d.x;
    } else {
        return d.y;
    }
}

__device__ __forceinline__ bool iq2_raw_row_valid(int warp_row0, int row, int M) {
    return warp_row0 + row < M;
}

template <bool FullTile, int Iter>
__device__ __forceinline__ void issue_iq2_raw_codes_iter(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ qs_base,
        uint32_t warp_row0_blk, int nb, int raw_stage, int k256_iter,
        int warp_row0, int M, int warp, int lane) {
    if constexpr (Iter < kIQ2RawQCodeTrips) {
        const int t = lane + Iter * 32;
        if (t < kIQ2RawQCodeChunks) {
            const int row = t >> 3;
            const int pair = t & 7;
            const bool row_valid = FullTile ? true : iq2_raw_row_valid(warp_row0, row, M);
            const bool valid = k256_iter < nb && row_valid;
            const uint64_t blk = valid ? ((uint64_t)warp_row0_blk + (uint64_t)row * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[warp][raw_stage].qs[row][pair];
            const void *src = qs_base + blk * 8ull + (uint64_t)pair;
            cp_async_8B(dst, src, valid);
        }
        issue_iq2_raw_codes_iter<FullTile, Iter + 1>(
            s_raw, qs_base, warp_row0_blk, nb, raw_stage, k256_iter,
            warp_row0, M, warp, lane);
    }
}

template <bool FullTile>
__device__ __forceinline__ void issue_iq2_raw_prefetch(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv,
        int raw_stage, int k256_iter, int warp, int lane, half dq_val) {
    const uint2 *qs_base = s_inv.iq2_qs_base;
    const uint32_t warp_row0_blk = s_inv.warp_row0_blk[warp];
    const int nb = s_inv.nb;
    int warp_row0 = 0;
    int M = 0;
    if constexpr (!FullTile) {
        warp_row0 = s_inv.cta_row0 + (warp << 4);
        M = s_inv.M;
    }
    issue_iq2_raw_codes_iter<FullTile, 0>(
        s_raw, qs_base, warp_row0_blk, nb, raw_stage, k256_iter,
        warp_row0, M, warp, lane);
    if (lane < kIQ2RawRowsPerWarp) {
        s_raw[warp][raw_stage].dq[lane] = dq_val;
    }
    cp_async_commit();
}

/* The per-block dq halves cannot ride the cp.async ring (2-byte elements,
 * nb-strided rows), so they go global->register->smem.  Issuing the LDG here
 * and passing the value into issue_iq2_raw_prefetch* one k128 iteration later
 * hides the load latency behind a fold; fused LDG.U16->STS.U16 was 52% of the
 * kernel's long-scoreboard stalls (cmd2rncu15b PC sampling). */
template <bool FullTile>
__device__ __forceinline__ half iq2_raw_dq_preload(
        const volatile SmemInvariants &s_inv, int k256_iter) {
    const int lane = d2r_lane();
    half dq = __float2half(0.0f);
    if (lane < kIQ2RawRowsPerWarp) {
        bool valid = k256_iter < s_inv.nb;
        if constexpr (!FullTile) {
            const int warp_row0 = s_inv.cta_row0 + (d2r_warp() << 4);
            valid = valid && iq2_raw_row_valid(warp_row0, lane, s_inv.M);
        }
        if (valid) {
            const uint64_t blk = (uint64_t)s_inv.warp_row0_blk[d2r_warp()] +
                                 (uint64_t)lane * (uint64_t)s_inv.nb + (uint64_t)k256_iter;
            dq = s_inv.iq2_dq_base[blk];
        }
    }
    return dq;
}

template <int Iter>
__device__ __forceinline__ void issue_iq2_raw_codes_iter_fast(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ qs_base,
        uint32_t warp_row0_blk, int nb, int raw_stage, int k256_iter) {
    if constexpr (Iter < kIQ2RawQCodeTrips) {
        const int lane = d2r_lane();
        const int t = lane + Iter * 32;
        if (t < kIQ2RawQCodeChunks) {
            const int row = t >> 3;
            const int pair = t & 7;
            const bool valid = k256_iter < nb;
            const uint64_t blk = valid ? ((uint64_t)warp_row0_blk + (uint64_t)row * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[d2r_warp()][raw_stage].qs[row][pair];
            const void *src = qs_base + blk * 8ull + (uint64_t)pair;
            cp_async_8B(dst, src, valid);
        }
        issue_iq2_raw_codes_iter_fast<Iter + 1>(
            s_raw, qs_base, warp_row0_blk, nb, raw_stage, k256_iter);
    }
}

__device__ __forceinline__ void issue_iq2_raw_prefetch_fast(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv,
        int raw_stage, int k256_iter, half dq_val) {
    const int warp = d2r_warp();
    const int lane = d2r_lane();
    const uint32_t warp_row0_blk = s_inv.warp_row0_blk[warp];
    issue_iq2_raw_codes_iter_fast<0>(
        s_raw, s_inv.iq2_qs_base, warp_row0_blk, s_inv.nb, raw_stage, k256_iter);
    if (lane < kIQ2RawRowsPerWarp) {
        s_raw[warp][raw_stage].dq[lane] = dq_val;
    }
    cp_async_commit();
}

__device__ __forceinline__ uint32_t iq2_decode_signed_half(
        uint2 code, const uint2 * __restrict__ s_grid, int chunk) {
    const int group = chunk >> 1;
    const int hi = chunk & 1;
    const uint8_t aux = (uint8_t)(code.x >> (8 * group));
    const uint2 grid_pos = s_grid[aux];
    const uint32_t signs8 = unpack_ksigns((uint8_t)(code.y >> (7 * group)));
    const uint32_t sel = hi ? 0x80402010u : 0x08040201u;
    const uint32_t s = __vcmpne4(signs8 & sel, 0);
    const uint32_t grid_half = hi ? grid_pos.y : grid_pos.x;
    return __vsub4(grid_half ^ s, s);
}

template <int T, typename TileA>
__device__ __forceinline__ void make_iq2_A_tile(
        TileA &A, float &dA0, float &dA1,
        const IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ s_grid,
        int warp, int raw_stage, bool row0_ok, bool row1_ok,
        int group, int tig) {
    const IQ2RawWarpStage &raw = s_raw[warp][raw_stage];
    constexpr int pair = T;
    const int row0 = group;
    const int row1 = group + 8;
    const uint2 code0 = row0_ok ? raw.qs[row0][pair] : make_uint2(0, 0);
    const uint2 code1 = row1_ok ? raw.qs[row1][pair] : make_uint2(0, 0);

    A.x[0] = row0_ok ? (int)iq2_decode_signed_half(code0, s_grid, tig) : 0;
    A.x[1] = row1_ok ? (int)iq2_decode_signed_half(code1, s_grid, tig) : 0;
    A.x[2] = row0_ok ? (int)iq2_decode_signed_half(code0, s_grid, tig + 4) : 0;
    A.x[3] = row1_ok ? (int)iq2_decode_signed_half(code1, s_grid, tig + 4) : 0;

    const float d0 = row0_ok ? __half2float(raw.dq[row0]) : 0.0f;
    const float d1 = row1_ok ? __half2float(raw.dq[row1]) : 0.0f;
    const int ls0 = (int)(code0.y >> 27) | 1;
    const int ls1 = (int)(code1.y >> 27) | 1;
    dA0 = d0 * (float)ls0 * 0.125f;
    dA1 = d1 * (float)ls1 * 0.125f;
}

template <int NFrag>
__device__ __forceinline__ void fold_iq2_fragment_fast(
        float (&acc)[NFrag][4],
        const ggml_cuda_mma::tile<16, 8, int> &C,
        int nf, float dA0, float dA1, float dB0, float dB1) {
    const float s00 = dA0 * dB0;
    const float s01 = dA0 * dB1;
    const float s10 = dA1 * dB0;
    const float s11 = dA1 * dB1;
    acc[nf][0] = fmaf((float)C.x[0], s00, acc[nf][0]);
    acc[nf][1] = fmaf((float)C.x[1], s01, acc[nf][1]);
    acc[nf][2] = fmaf((float)C.x[2], s10, acc[nf][2]);
    acc[nf][3] = fmaf((float)C.x[3], s11, acc[nf][3]);
}

template <int NFrag>
__device__ __forceinline__ void fold_iq2_fragment_guarded(
        float (&acc)[NFrag][4],
        const ggml_cuda_mma::tile<16, 8, int> &C,
        int nf, float dA0, float dA1, float dB0, float dB1,
        bool row0_ok, bool row1_ok, bool col0_ok, bool col1_ok) {
    const float s00 = dA0 * dB0;
    const float s01 = dA0 * dB1;
    const float s10 = dA1 * dB0;
    const float s11 = dA1 * dB1;
    if (row0_ok && col0_ok) acc[nf][0] = fmaf((float)C.x[0], s00, acc[nf][0]);
    if (row0_ok && col1_ok) acc[nf][1] = fmaf((float)C.x[1], s01, acc[nf][1]);
    if (row1_ok && col0_ok) acc[nf][2] = fmaf((float)C.x[2], s10, acc[nf][2]);
    if (row1_ok && col1_ok) acc[nf][3] = fmaf((float)C.x[3], s11, acc[nf][3]);
}

template <bool FullTile, typename TileA, typename TileB, typename TileC,
          int T0, int T1, int NFrag>
__device__ __forceinline__ void mma_fold_iq2_k32_pair_t(
        float (&acc)[NFrag][TileC::ne],
        const IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ s_grid,
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        int raw_stage, int q8_stage, bool raw_row0_ok, bool raw_row1_ok,
        int warp, int group, int tig, const volatile SmemInvariants &s_inv) {
    static_assert(T1 == T0 + 1, "expected adjacent k32 pair");
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    TileA A0;
    TileA A1;
    float dA00;
    float dA01;
    float dA10;
    float dA11;
    make_iq2_A_tile<T0>(A0, dA00, dA01, s_raw, s_grid, warp, raw_stage,
                        raw_row0_ok, raw_row1_ok, group, tig);
    make_iq2_A_tile<T1>(A1, dA10, dA11, s_raw, s_grid, warp, raw_stage,
                        raw_row0_ok, raw_row1_ok, group, tig);

    constexpr int k_in_q8_0 = (T0 & 3) * 32;
    constexpr int k_in_q8_1 = (T1 & 3) * 32;
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);
    int col_count = 0;
    int nf_live = NFrag;
    if constexpr (!FullTile) {
        col_count = s_inv.col_count;
        nf_live = (col_count + 7) >> 3;
    }

#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        if constexpr (!FullTile) {
            if (nf >= nf_live) {
                break;
            }
        }
        const int col_local = nf * 8;
        bool col0_ok = true;
        bool col1_ok = true;
        if constexpr (!FullTile) {
            col0_ok = (col_local + c0) < col_count;
            col1_ok = (col_local + c1) < col_count;
        }
        const Q8D4K64PairF32 dB = load_q8_d4_k64_pair<T0>(s_q8, q8_stage, nf, c0, c1);

        TileB B0;
        TileC C0;
        load_B_tile(B0, s_q8, q8_stage, nf, k_in_q8_0);
        ggml_cuda_mma::mma(C0, A0, B0);
        if constexpr (FullTile) {
            fold_iq2_fragment_fast(acc, C0, nf, dA00, dA01,
                                   q8_d4_k64_slot<T0>(dB.c0), q8_d4_k64_slot<T0>(dB.c1));
        } else {
            fold_iq2_fragment_guarded(acc, C0, nf, dA00, dA01,
                                      q8_d4_k64_slot<T0>(dB.c0), q8_d4_k64_slot<T0>(dB.c1),
                                      raw_row0_ok, raw_row1_ok, col0_ok, col1_ok);
        }

        TileB B1;
        TileC C1;
        load_B_tile(B1, s_q8, q8_stage, nf, k_in_q8_1);
        ggml_cuda_mma::mma(C1, A1, B1);
        if constexpr (FullTile) {
            fold_iq2_fragment_fast(acc, C1, nf, dA10, dA11,
                                   q8_d4_k64_slot<T1>(dB.c0), q8_d4_k64_slot<T1>(dB.c1));
        } else {
            fold_iq2_fragment_guarded(acc, C1, nf, dA10, dA11,
                                      q8_d4_k64_slot<T1>(dB.c0), q8_d4_k64_slot<T1>(dB.c1),
                                      raw_row0_ok, raw_row1_ok, col0_ok, col1_ok);
        }
    }
}

template <bool FullTile, typename TileA, typename TileB, typename TileC,
          int NFrag>
__device__ __forceinline__ void mma_fold_iq2_k128(
        float (&acc)[NFrag][TileC::ne],
        const IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ s_grid,
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        int k128_iter, const volatile SmemInvariants &s_inv) {
    const int warp = d2r_warp();
    const int group = d2r_group();
    int warp_row0 = 0;
    if constexpr (!FullTile) {
        warp_row0 = s_inv.cta_row0 + (warp << 4);
    }
    bool row0_ok = true;
    bool row1_ok = true;
    if constexpr (!FullTile) {
        row0_ok = (warp_row0 + group) < s_inv.M;
        row1_ok = (warp_row0 + group + 8) < s_inv.M;
    }
    const int tig = d2r_tig();
    const int raw_stage = d2r_raw_stage(k128_iter >> 1);
    const int q8_stage = d2r_q8_stage(k128_iter);
    const int half_pair_base = (k128_iter & 1) ? 4 : 0;

    if (half_pair_base == 0) {
        mma_fold_iq2_k32_pair_t<FullTile, TileA, TileB, TileC, 0, 1>(
            acc, s_raw, s_grid, s_q8, raw_stage, q8_stage,
            row0_ok, row1_ok, warp, group, tig, s_inv);
        mma_fold_iq2_k32_pair_t<FullTile, TileA, TileB, TileC, 2, 3>(
            acc, s_raw, s_grid, s_q8, raw_stage, q8_stage,
            row0_ok, row1_ok, warp, group, tig, s_inv);
    } else {
        mma_fold_iq2_k32_pair_t<FullTile, TileA, TileB, TileC, 4, 5>(
            acc, s_raw, s_grid, s_q8, raw_stage, q8_stage,
            row0_ok, row1_ok, warp, group, tig, s_inv);
        mma_fold_iq2_k32_pair_t<FullTile, TileA, TileB, TileC, 6, 7>(
            acc, s_raw, s_grid, s_q8, raw_stage, q8_stage,
            row0_ok, row1_ok, warp, group, tig, s_inv);
    }
}

template <bool FullTile, typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void iq2_d2r_mainloop(
        float (&acc)[kNFrag][TileC::ne],
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const uint2 * __restrict__ s_grid,
        const volatile SmemInvariants &s_inv) {
#pragma unroll
    for (int pf = 0; pf < kStages; ++pf) {
        if (pf < s_inv.k128_iters) {
            if constexpr (FullTile) {
                issue_q8_prefetch_fast(s_q8, s_inv, d2r_q8_stage(pf), pf);
            } else {
                issue_q8_prefetch<false>(s_q8, s_inv, d2r_q8_stage(pf), pf, d2r_tid());
            }
        }
    }

#pragma unroll
    for (int pf = 0; pf < kRawStages; ++pf) {
        if (pf < s_inv.nb) {
            const half dq0 = iq2_raw_dq_preload<FullTile>(s_inv, pf);
            if constexpr (FullTile) {
                issue_iq2_raw_prefetch_fast(s_raw, s_inv, d2r_raw_stage(pf), pf, dq0);
            } else {
                issue_iq2_raw_prefetch<false>(
                    s_raw, s_inv, d2r_raw_stage(pf), pf, d2r_warp(), d2r_lane(), dq0);
            }
        }
    }

    half dq_pend = __float2half(0.0f);
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
        if ((k128_iter & 1) == 0) {
            /* dq LDG for the raw prefetch issued at the NEXT (odd) iteration:
             * a full fold of distance between the load and its smem store. */
            const int raw_pf = (k128_iter >> 1) + kRawStages;
            if (raw_pf < s_inv.nb) {
                dq_pend = iq2_raw_dq_preload<FullTile>(s_inv, raw_pf);
            }
        }

        if constexpr (FullTile) {
            mma_fold_iq2_k128<true, TileA, TileB, TileC>(
                acc, s_raw, s_grid, s_q8, k128_iter, s_inv);
        } else {
            mma_fold_iq2_k128<false, TileA, TileB, TileC>(
                acc, s_raw, s_grid, s_q8, k128_iter, s_inv);
        }

        __syncthreads();
        const int pf_iter = k128_iter + kStages;
        if (pf_iter < s_inv.k128_iters) {
            if constexpr (FullTile) {
                issue_q8_prefetch_fast(s_q8, s_inv, d2r_q8_stage(pf_iter), pf_iter);
            } else {
                issue_q8_prefetch<false>(s_q8, s_inv, d2r_q8_stage(pf_iter), pf_iter, d2r_tid());
            }
        }
        if ((k128_iter & 1) != 0) {
            const int raw_pf = (k128_iter >> 1) + kRawStages;
            if (raw_pf < s_inv.nb) {
                if constexpr (FullTile) {
                    issue_iq2_raw_prefetch_fast(s_raw, s_inv, d2r_raw_stage(raw_pf), raw_pf, dq_pend);
                } else {
                    issue_iq2_raw_prefetch<false>(
                        s_raw, s_inv, d2r_raw_stage(raw_pf), raw_pf, d2r_warp(), d2r_lane(), dq_pend);
                }
            }
        }
    }
}

template <bool FullTile>
__device__ __forceinline__ void issue_fused_iq2_raw_prefetch(
        IQ2RawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv,
        int raw_stage, int k256_iter) {
    const half dq = iq2_raw_dq_preload<FullTile>(s_inv, k256_iter);
    if constexpr (FullTile) {
        issue_iq2_raw_prefetch_fast(s_raw, s_inv, raw_stage, k256_iter, dq);
    } else {
        issue_iq2_raw_prefetch<false>(
            s_raw, s_inv, raw_stage, k256_iter,
            d2r_warp(), d2r_lane(), dq);
    }
}

template <bool FullTile, int TileN,
          typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void iq2_gateup_fused_mainloop(
        float (&gate_acc)[TileN / 8][TileC::ne],
        float (&up_acc)[TileN / 8][TileC::ne],
        FusedGateUpComputeSmem<TileN> &s,
        uint32_t y_off) {
    const int k128_iters = s.inv[0].k128_iters;
    const int k256_iters = s.inv[0].nb;

    issue_fused_q8_prefetch<FullTile>(s.q8, s.inv[0], y_off, 0, 0);
    issue_fused_q8_prefetch<FullTile>(s.q8, s.inv[0], y_off, 1, 1);
    issue_fused_iq2_raw_prefetch<FullTile>(s.raw[0], s.inv[0], 0, 0);
    issue_fused_iq2_raw_prefetch<FullTile>(s.raw[1], s.inv[1], 0, 0);
    cp_async_wait_keep(0);
    __syncthreads();

    for (int k256_iter = 0; k256_iter < k256_iters; ++k256_iter) {
        const int next_k256 = k256_iter + 1;
        if (next_k256 < k256_iters) {
            const int next_stage = d2r_raw_stage(next_k256);
            issue_fused_iq2_raw_prefetch<FullTile>(
                s.raw[0], s.inv[0], next_stage, next_k256);
            issue_fused_iq2_raw_prefetch<FullTile>(
                s.raw[1], s.inv[1], next_stage, next_k256);
        }

        const int even_k128 = 2 * k256_iter;
        mma_fold_iq2_k128<FullTile, TileA, TileB, TileC>(
            gate_acc, s.raw[0], s.grid, s.q8, even_k128, s.inv[0]);
        mma_fold_iq2_k128<FullTile, TileA, TileB, TileC>(
            up_acc, s.raw[1], s.grid, s.q8, even_k128, s.inv[1]);

        __syncthreads();
        const int next_even_k128 = even_k128 + 2;
        if (next_even_k128 < k128_iters) {
            issue_fused_q8_prefetch<FullTile>(
                s.q8, s.inv[0], y_off, d2r_q8_stage(next_even_k128), next_even_k128);
        }

        const int odd_k128 = even_k128 + 1;
        mma_fold_iq2_k128<FullTile, TileA, TileB, TileC>(
            gate_acc, s.raw[0], s.grid, s.q8, odd_k128, s.inv[0]);
        mma_fold_iq2_k128<FullTile, TileA, TileB, TileC>(
            up_acc, s.raw[1], s.grid, s.q8, odd_k128, s.inv[1]);

        __syncthreads();
        const int next_odd_k128 = odd_k128 + 2;
        if (next_odd_k128 < k128_iters) {
            issue_fused_q8_prefetch<FullTile>(
                s.q8, s.inv[0], y_off, d2r_q8_stage(next_odd_k128), next_odd_k128);
        }

        if (next_k256 < k256_iters) {
            cp_async_wait_keep(0);
            __syncthreads();
        }
    }
}

[[maybe_unused]] __device__ __forceinline__ void fold_element_k32(
        float &acc, int c, float2 dm, uint16_t min_pack, const Q8Fix2 &fix) {
    acc += (float)c * dm.x * fix.d8;
    acc -= dm.y * (float)(min_pack & 0xFFu) * fix.sum0;
    acc -= dm.y * (float)((min_pack >> 8) & 0xFFu) * fix.sum1;
}

[[maybe_unused]] __device__ __forceinline__ void fold_element_k64(
        float &acc, int c, float2 dm, uint32_t min_pack4, const Q8Fix4 &fix) {
    acc += (float)c * dm.x * fix.d8;
    acc -= dm.y * (float)(min_pack4 & 0xFFu) * fix.sum0;
    acc -= dm.y * (float)((min_pack4 >> 8) & 0xFFu) * fix.sum1;
    acc -= dm.y * (float)((min_pack4 >> 16) & 0xFFu) * fix.sum2;
    acc -= dm.y * (float)((min_pack4 >> 24) & 0xFFu) * fix.sum3;
}

[[maybe_unused]] __device__ __forceinline__ float q2k_bias_k32(
        float dmin, uint16_t min_pack, const Q8Fix2 &fix) {
    const float min0 = (float)(min_pack & 0xFFu);
    const float min1 = (float)((min_pack >> 8) & 0xFFu);
    return dmin * (min0 * fix.sum0 + min1 * fix.sum1);
}

__device__ __forceinline__ float q2k_minsum4_f32(
        uint32_t min_pack4, float sum0, float sum1, float sum2, float sum3) {
    float s = fmaf((float)(min_pack4 & 0xFFu), sum0, 0.0f);
    s = fmaf((float)((min_pack4 >> 8) & 0xFFu), sum1, s);
    s = fmaf((float)((min_pack4 >> 16) & 0xFFu), sum2, s);
    return fmaf((float)((min_pack4 >> 24) & 0xFFu), sum3, s);
}

template <int T0, int T1, int NFrag>
__device__ __forceinline__ void fold_k64_col_fast(
        float &acc0, float &acc1, int c0, int c1,
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int c, float2 dm0, float2 dm1,
        uint32_t min0, uint32_t min1) {
    static_assert((T0 == 0 && T1 == 1) || (T0 == 2 && T1 == 3), "expected adjacent k64 q8 window");
    constexpr int d8_slot = (T0 >= 2) ? 1 : 0;
    constexpr int sub128 = 2 * T0;
    const Q8ColFixF32 &sf = s_q8_fix[stage][nf][c];
    const float d8 = sf.d8[d8_slot];
    const float sum0 = sf.sum[sub128 + 0];
    const float sum1 = sf.sum[sub128 + 1];
    const float sum2 = sf.sum[sub128 + 2];
    const float sum3 = sf.sum[sub128 + 3];

    const float minsum0 = q2k_minsum4_f32(min0, sum0, sum1, sum2, sum3);
    acc0 = fmaf(-dm0.y, minsum0, fmaf((float)c0, dm0.x * d8, acc0));

    const float minsum1 = q2k_minsum4_f32(min1, sum0, sum1, sum2, sum3);
    acc1 = fmaf(-dm1.y, minsum1, fmaf((float)c1, dm1.x * d8, acc1));
}

template <int T0, int T1, int NFrag>
__device__ __forceinline__ void fold_k64_col_guarded(
        float &acc0, float &acc1, int c0, int c1,
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int c, float2 dm0, float2 dm1,
        uint32_t min0, uint32_t min1, bool row0_ok, bool row1_ok, bool col_ok) {
    static_assert((T0 == 0 && T1 == 1) || (T0 == 2 && T1 == 3), "expected adjacent k64 q8 window");
    if (!col_ok || (!row0_ok && !row1_ok)) {
        return;
    }
    constexpr int d8_slot = (T0 >= 2) ? 1 : 0;
    constexpr int sub128 = 2 * T0;
    const Q8ColFixF32 &sf = s_q8_fix[stage][nf][c];
    const float d8 = sf.d8[d8_slot];
    const float sum0 = sf.sum[sub128 + 0];
    const float sum1 = sf.sum[sub128 + 1];
    const float sum2 = sf.sum[sub128 + 2];
    const float sum3 = sf.sum[sub128 + 3];

    if (row0_ok) {
        const float minsum0 = q2k_minsum4_f32(min0, sum0, sum1, sum2, sum3);
        acc0 = fmaf(-dm0.y, minsum0, fmaf((float)c0, dm0.x * d8, acc0));
    }
    if (row1_ok) {
        const float minsum1 = q2k_minsum4_f32(min1, sum0, sum1, sum2, sum3);
        acc1 = fmaf(-dm1.y, minsum1, fmaf((float)c1, dm1.x * d8, acc1));
    }
}

template <int NFrag>
__device__ __forceinline__ void fold_fragment_k32_fast(
        float (&acc)[NFrag][4],
        const ggml_cuda_mma::tile<16, 8, int> &C,
        const Q8Fix2 &fix0, const Q8Fix2 &fix1, int nf,
        float2 dm0, float2 dm1, uint16_t min0, uint16_t min1) {
    const float d8c0 = fix0.d8;
    const float d8c1 = fix1.d8;
    const float scale00 = dm0.x * d8c0;
    const float scale01 = dm0.x * d8c1;
    const float scale10 = dm1.x * d8c0;
    const float scale11 = dm1.x * d8c1;
    const float bias00 = q2k_bias_k32(dm0.y, min0, fix0);
    const float bias01 = q2k_bias_k32(dm0.y, min0, fix1);
    const float bias10 = q2k_bias_k32(dm1.y, min1, fix0);
    const float bias11 = q2k_bias_k32(dm1.y, min1, fix1);

    acc[nf][0] = fmaf((float)C.x[0], scale00, acc[nf][0]) - bias00;
    acc[nf][1] = fmaf((float)C.x[1], scale01, acc[nf][1]) - bias01;
    acc[nf][2] = fmaf((float)C.x[2], scale10, acc[nf][2]) - bias10;
    acc[nf][3] = fmaf((float)C.x[3], scale11, acc[nf][3]) - bias11;
}

template <int NFrag>
__device__ __forceinline__ void fold_fragment_k32_guarded(
        float (&acc)[NFrag][4],
        const ggml_cuda_mma::tile<16, 8, int> &C,
        const Q8Fix2 &fix0, const Q8Fix2 &fix1, int nf,
        float2 dm0, float2 dm1, uint16_t min0, uint16_t min1,
        bool row0_ok, bool row1_ok, bool col0_ok, bool col1_ok) {
    const float d8c0 = fix0.d8;
    const float d8c1 = fix1.d8;
    const float scale00 = dm0.x * d8c0;
    const float scale01 = dm0.x * d8c1;
    const float scale10 = dm1.x * d8c0;
    const float scale11 = dm1.x * d8c1;
    const float bias00 = q2k_bias_k32(dm0.y, min0, fix0);
    const float bias01 = q2k_bias_k32(dm0.y, min0, fix1);
    const float bias10 = q2k_bias_k32(dm1.y, min1, fix0);
    const float bias11 = q2k_bias_k32(dm1.y, min1, fix1);

    if (row0_ok && col0_ok) acc[nf][0] = fmaf((float)C.x[0], scale00, acc[nf][0]) - bias00;
    if (row0_ok && col1_ok) acc[nf][1] = fmaf((float)C.x[1], scale01, acc[nf][1]) - bias01;
    if (row1_ok && col0_ok) acc[nf][2] = fmaf((float)C.x[2], scale10, acc[nf][2]) - bias10;
    if (row1_ok && col1_ok) acc[nf][3] = fmaf((float)C.x[3], scale11, acc[nf][3]) - bias11;
}

template <int T0, int T1, int NFrag>
__device__ __forceinline__ void fold_fragment_k64_fast(
        float (&acc)[NFrag][4],
        const ggml_cuda_mma::tile<16, 8, int> &C,
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int c0, int c1, float2 dm0, float2 dm1,
        uint32_t min0, uint32_t min1) {
    fold_k64_col_fast<T0, T1>(
        acc[nf][0], acc[nf][2], C.x[0], C.x[2],
        s_q8_fix, stage, nf, c0, dm0, dm1, min0, min1);
    fold_k64_col_fast<T0, T1>(
        acc[nf][1], acc[nf][3], C.x[1], C.x[3],
        s_q8_fix, stage, nf, c1, dm0, dm1, min0, min1);
}

template <int T0, int T1, int NFrag>
__device__ __forceinline__ void fold_fragment_k64_guarded(
        float (&acc)[NFrag][4],
        const ggml_cuda_mma::tile<16, 8, int> &C,
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int c0, int c1, float2 dm0, float2 dm1,
        uint32_t min0, uint32_t min1,
        bool row0_ok, bool row1_ok, bool col0_ok, bool col1_ok) {
    if (col0_ok && (row0_ok || row1_ok)) {
        fold_k64_col_guarded<T0, T1>(
            acc[nf][0], acc[nf][2], C.x[0], C.x[2],
            s_q8_fix, stage, nf, c0, dm0, dm1, min0, min1,
            row0_ok, row1_ok, col0_ok);
    }
    if (col1_ok && (row0_ok || row1_ok)) {
        fold_k64_col_guarded<T0, T1>(
            acc[nf][1], acc[nf][3], C.x[1], C.x[3],
            s_q8_fix, stage, nf, c1, dm0, dm1, min0, min1,
            row0_ok, row1_ok, col1_ok);
    }
}

template <typename TileC, int NFrag>
__device__ __forceinline__ void fold_fragment_k32(
        float (&acc)[NFrag][TileC::ne],
        const TileC &C,
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int k_in_q8, int warp_row0, int M,
        int col_lo, int col_hi, uint32_t dm0_bits, uint32_t dm1_bits,
        uint16_t min0, uint16_t min1) {
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);
    const int col0 = col_lo + nf * 8 + c0;
    const int col1 = col_lo + nf * 8 + c1;
    const bool col0_ok = col0 < col_hi;
    const bool col1_ok = col1 < col_hi;
    const Q8Fix2 fix0 = load_q8_fix2(s_q8_fix, stage, nf, c0, k_in_q8);
    const Q8Fix2 fix1 = load_q8_fix2(s_q8_fix, stage, nf, c1, k_in_q8);

    const float2 dm0 = half2_bits_to_float2(dm0_bits);
    const float2 dm1 = half2_bits_to_float2(dm1_bits);
    const int row0 = warp_row0 + TileC::get_i(0);
    const int row1 = warp_row0 + TileC::get_i(2);
    const bool row0_ok = row0 < M;
    const bool row1_ok = row1 < M;

    if (row0_ok && col0_ok) {
        fold_element_k32(acc[nf][0], C.x[0], dm0, min0, fix0);
    }
    if (row0_ok && col1_ok) {
        fold_element_k32(acc[nf][1], C.x[1], dm0, min0, fix1);
    }
    if (row1_ok && col0_ok) {
        fold_element_k32(acc[nf][2], C.x[2], dm1, min1, fix0);
    }
    if (row1_ok && col1_ok) {
        fold_element_k32(acc[nf][3], C.x[3], dm1, min1, fix1);
    }
}

template <typename TileC, int NFrag>
__device__ __forceinline__ void fold_fragment_k64(
        float (&acc)[NFrag][TileC::ne],
        const TileC &C,
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int k_in_q8_pair, int warp_row0, int M,
        int col_lo, int col_hi, uint32_t dm0_bits, uint32_t dm1_bits,
        uint32_t min0, uint32_t min1) {
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);
    const int col0 = col_lo + nf * 8 + c0;
    const int col1 = col_lo + nf * 8 + c1;
    const bool col0_ok = col0 < col_hi;
    const bool col1_ok = col1 < col_hi;
    const Q8Fix4 fix0 = load_q8_fix4(s_q8_fix, stage, nf, c0, k_in_q8_pair);
    const Q8Fix4 fix1 = load_q8_fix4(s_q8_fix, stage, nf, c1, k_in_q8_pair);

    const float2 dm0 = half2_bits_to_float2(dm0_bits);
    const float2 dm1 = half2_bits_to_float2(dm1_bits);
    const int row0 = warp_row0 + TileC::get_i(0);
    const int row1 = warp_row0 + TileC::get_i(2);
    const bool row0_ok = row0 < M;
    const bool row1_ok = row1 < M;

    if (row0_ok && col0_ok) {
        fold_element_k64(acc[nf][0], C.x[0], dm0, min0, fix0);
    }
    if (row0_ok && col1_ok) {
        fold_element_k64(acc[nf][1], C.x[1], dm0, min0, fix1);
    }
    if (row1_ok && col0_ok) {
        fold_element_k64(acc[nf][2], C.x[2], dm1, min1, fix0);
    }
    if (row1_ok && col1_ok) {
        fold_element_k64(acc[nf][3], C.x[3], dm1, min1, fix1);
    }
}

template <typename TileA, typename TileB, typename TileC, bool FullTile, int NFrag>
__device__ __forceinline__ void mma_fold_k128_k32(
        float (&acc)[NFrag][TileC::ne],
        const Q2KWeightHalf &w,
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int warp_row0, int M, int col_lo, int col_hi, int nf_live) {
    TileA A0;
    TileA A1;
    TileA A2;
    TileA A3;
    make_A_tile<0>(A0, w);
    make_A_tile<1>(A1, w);
    make_A_tile<2>(A2, w);
    make_A_tile<3>(A3, w);

    const uint32_t min0_p01 = min_pack4_for_pair<0, 1>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint32_t min1_p01 = min_pack4_for_pair<0, 1>(w.sc_r1_lo4, w.sc_r1_hi4);
    const uint32_t min0_p23 = min_pack4_for_pair<2, 3>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint32_t min1_p23 = min_pack4_for_pair<2, 3>(w.sc_r1_lo4, w.sc_r1_hi4);
    const float2 dm0 = half2_bits_to_float2(w.dm_r0);
    const float2 dm1 = half2_bits_to_float2(w.dm_r1);
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);
    const bool row0_ok = FullTile ? true : ((warp_row0 + TileC::get_i(0)) < M);
    const bool row1_ok = FullTile ? true : ((warp_row0 + TileC::get_i(2)) < M);

#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        if constexpr (!FullTile) {
            if (nf >= nf_live) {
                break;
            }
        }
        const int col_frag = col_lo + nf * 8;
        const bool col0_ok = FullTile ? true : ((col_frag + c0) < col_hi);
        const bool col1_ok = FullTile ? true : ((col_frag + c1) < col_hi);

        TileB B0;
        TileC C0;
        const Q8Fix2 fix00 = load_q8_fix2(s_q8_fix, stage, nf, c0, 0);
        const Q8Fix2 fix01 = load_q8_fix2(s_q8_fix, stage, nf, c1, 0);
        load_B_tile(B0, s_q8, stage, nf, 0);
        ggml_cuda_mma::mma(C0, A0, B0);
        if constexpr (FullTile) {
            fold_fragment_k32_fast(acc, C0, fix00, fix01, nf, dm0, dm1,
                                   (uint16_t)min0_p01, (uint16_t)min1_p01);
        } else {
            fold_fragment_k32_guarded(acc, C0, fix00, fix01, nf, dm0, dm1,
                                      (uint16_t)min0_p01, (uint16_t)min1_p01,
                                      row0_ok, row1_ok, col0_ok, col1_ok);
        }

        TileB B1;
        TileC C1;
        const Q8Fix2 fix10 = load_q8_fix2(s_q8_fix, stage, nf, c0, 32);
        const Q8Fix2 fix11 = load_q8_fix2(s_q8_fix, stage, nf, c1, 32);
        load_B_tile(B1, s_q8, stage, nf, 32);
        ggml_cuda_mma::mma(C1, A1, B1);
        if constexpr (FullTile) {
            fold_fragment_k32_fast(acc, C1, fix10, fix11, nf, dm0, dm1,
                                   (uint16_t)(min0_p01 >> 16), (uint16_t)(min1_p01 >> 16));
        } else {
            fold_fragment_k32_guarded(acc, C1, fix10, fix11, nf, dm0, dm1,
                                      (uint16_t)(min0_p01 >> 16), (uint16_t)(min1_p01 >> 16),
                                      row0_ok, row1_ok, col0_ok, col1_ok);
        }

        TileB B2;
        TileC C2;
        const Q8Fix2 fix20 = load_q8_fix2(s_q8_fix, stage, nf, c0, 64);
        const Q8Fix2 fix21 = load_q8_fix2(s_q8_fix, stage, nf, c1, 64);
        load_B_tile(B2, s_q8, stage, nf, 64);
        ggml_cuda_mma::mma(C2, A2, B2);
        if constexpr (FullTile) {
            fold_fragment_k32_fast(acc, C2, fix20, fix21, nf, dm0, dm1,
                                   (uint16_t)min0_p23, (uint16_t)min1_p23);
        } else {
            fold_fragment_k32_guarded(acc, C2, fix20, fix21, nf, dm0, dm1,
                                      (uint16_t)min0_p23, (uint16_t)min1_p23,
                                      row0_ok, row1_ok, col0_ok, col1_ok);
        }

        TileB B3;
        TileC C3;
        const Q8Fix2 fix30 = load_q8_fix2(s_q8_fix, stage, nf, c0, 96);
        const Q8Fix2 fix31 = load_q8_fix2(s_q8_fix, stage, nf, c1, 96);
        load_B_tile(B3, s_q8, stage, nf, 96);
        ggml_cuda_mma::mma(C3, A3, B3);
        if constexpr (FullTile) {
            fold_fragment_k32_fast(acc, C3, fix30, fix31, nf, dm0, dm1,
                                   (uint16_t)(min0_p23 >> 16), (uint16_t)(min1_p23 >> 16));
        } else {
            fold_fragment_k32_guarded(acc, C3, fix30, fix31, nf, dm0, dm1,
                                      (uint16_t)(min0_p23 >> 16), (uint16_t)(min1_p23 >> 16),
                                      row0_ok, row1_ok, col0_ok, col1_ok);
        }
    }
}

template <typename TileA, typename TileB, typename TileC, bool FullTile, int NFrag>
__device__ __forceinline__ void mma_fold_k128_k64(
        float (&acc)[NFrag][TileC::ne],
        const Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int raw_stage, bool raw_row0_ok, bool raw_row1_ok, int parity, int raw_half,
        int warp, int group, int tig, int stage, int warp_row0, int M, int col_lo, int col_hi,
        int nf_live) {
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);
    const bool row0_ok = FullTile ? true : ((warp_row0 + TileC::get_i(0)) < M);
    const bool row1_ok = FullTile ? true : ((warp_row0 + TileC::get_i(2)) < M);

    {
        TileA A0;
        TileA A1;
        uint32_t min0;
        uint32_t min1;
        float2 dm0;
        float2 dm1;
        {
            Q2KWeightHalf w;
            load_q2k_weight_half_raw(
                w, s_raw, warp, raw_stage, raw_row0_ok, raw_row1_ok,
                parity, raw_half, group, tig);
            make_A_tile<0>(A0, w);
            make_A_tile<1>(A1, w);
            min0 = min_pack4_for_pair<0, 1>(w.sc_r0_lo4, w.sc_r0_hi4);
            min1 = min_pack4_for_pair<0, 1>(w.sc_r1_lo4, w.sc_r1_hi4);
            dm0 = half2_bits_to_float2(w.dm_r0);
            dm1 = half2_bits_to_float2(w.dm_r1);
        }
#pragma unroll
        for (int nf = 0; nf < NFrag; ++nf) {
            if constexpr (!FullTile) {
                if (nf >= nf_live) {
                    break;
                }
            }
            const int col_frag = col_lo + nf * 8;
            const bool col0_ok = FullTile ? true : ((col_frag + c0) < col_hi);
            const bool col1_ok = FullTile ? true : ((col_frag + c1) < col_hi);
            TileC C01;
            TileB B;
            load_B_tile(B, s_q8, stage, nf, 0);
            ggml_cuda_mma::mma(C01, A0, B);
            load_B_tile(B, s_q8, stage, nf, 32);
            ggml_cuda_mma::mma(C01, A1, B);
            if constexpr (FullTile) {
                fold_fragment_k64_fast<0, 1>(
                    acc, C01, s_q8_fix, stage, nf, c0, c1, dm0, dm1, min0, min1);
            } else {
                fold_fragment_k64_guarded<0, 1>(
                    acc, C01, s_q8_fix, stage, nf, c0, c1, dm0, dm1, min0, min1,
                    row0_ok, row1_ok, col0_ok, col1_ok);
            }
        }
    }

    {
        TileA A2;
        TileA A3;
        uint32_t min0;
        uint32_t min1;
        float2 dm0;
        float2 dm1;
        {
            Q2KWeightHalf w;
            load_q2k_weight_half_raw(
                w, s_raw, warp, raw_stage, raw_row0_ok, raw_row1_ok,
                parity, raw_half, group, tig);
            make_A_tile<2>(A2, w);
            make_A_tile<3>(A3, w);
            min0 = min_pack4_for_pair<2, 3>(w.sc_r0_lo4, w.sc_r0_hi4);
            min1 = min_pack4_for_pair<2, 3>(w.sc_r1_lo4, w.sc_r1_hi4);
            dm0 = half2_bits_to_float2(w.dm_r0);
            dm1 = half2_bits_to_float2(w.dm_r1);
        }
#pragma unroll
        for (int nf = 0; nf < NFrag; ++nf) {
            if constexpr (!FullTile) {
                if (nf >= nf_live) {
                    break;
                }
            }
            const int col_frag = col_lo + nf * 8;
            const bool col0_ok = FullTile ? true : ((col_frag + c0) < col_hi);
            const bool col1_ok = FullTile ? true : ((col_frag + c1) < col_hi);
            TileC C23;
            TileB B;
            load_B_tile(B, s_q8, stage, nf, 64);
            ggml_cuda_mma::mma(C23, A2, B);
            load_B_tile(B, s_q8, stage, nf, 96);
            ggml_cuda_mma::mma(C23, A3, B);
            if constexpr (FullTile) {
                fold_fragment_k64_fast<2, 3>(
                    acc, C23, s_q8_fix, stage, nf, c0, c1, dm0, dm1, min0, min1);
            } else {
                fold_fragment_k64_guarded<2, 3>(
                    acc, C23, s_q8_fix, stage, nf, c0, c1, dm0, dm1, min0, min1,
                    row0_ok, row1_ok, col0_ok, col1_ok);
            }
        }
    }
}

template <typename TileA, typename TileB, typename TileC, int NFrag>
__device__ __forceinline__ void mma_fold_k128_k32_fast(
        float (&acc)[NFrag][TileC::ne],
        const Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int k128_iter) {
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    Q2KWeightHalf w;
    load_q2k_weight_half_raw_fast(w, s_raw, k128_iter);

    TileA A0;
    TileA A1;
    TileA A2;
    TileA A3;
    make_A_tile<0>(A0, w);
    make_A_tile<1>(A1, w);
    make_A_tile<2>(A2, w);
    make_A_tile<3>(A3, w);

    const uint32_t min0_p01 = min_pack4_for_pair<0, 1>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint32_t min1_p01 = min_pack4_for_pair<0, 1>(w.sc_r1_lo4, w.sc_r1_hi4);
    const uint32_t min0_p23 = min_pack4_for_pair<2, 3>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint32_t min1_p23 = min_pack4_for_pair<2, 3>(w.sc_r1_lo4, w.sc_r1_hi4);
    const float2 dm0 = half2_bits_to_float2(w.dm_r0);
    const float2 dm1 = half2_bits_to_float2(w.dm_r1);
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);

#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        TileB B0;
        TileC C0;
        const Q8Fix2 fix00 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c0, 0);
        const Q8Fix2 fix01 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c1, 0);
        load_B_tile(B0, s_q8, d2r_q8_stage(k128_iter), nf, 0);
        ggml_cuda_mma::mma(C0, A0, B0);
        fold_fragment_k32_fast(acc, C0, fix00, fix01, nf, dm0, dm1,
                               (uint16_t)min0_p01, (uint16_t)min1_p01);

        TileB B1;
        TileC C1;
        const Q8Fix2 fix10 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c0, 32);
        const Q8Fix2 fix11 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c1, 32);
        load_B_tile(B1, s_q8, d2r_q8_stage(k128_iter), nf, 32);
        ggml_cuda_mma::mma(C1, A1, B1);
        fold_fragment_k32_fast(acc, C1, fix10, fix11, nf, dm0, dm1,
                               (uint16_t)(min0_p01 >> 16), (uint16_t)(min1_p01 >> 16));

        TileB B2;
        TileC C2;
        const Q8Fix2 fix20 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c0, 64);
        const Q8Fix2 fix21 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c1, 64);
        load_B_tile(B2, s_q8, d2r_q8_stage(k128_iter), nf, 64);
        ggml_cuda_mma::mma(C2, A2, B2);
        fold_fragment_k32_fast(acc, C2, fix20, fix21, nf, dm0, dm1,
                               (uint16_t)min0_p23, (uint16_t)min1_p23);

        TileB B3;
        TileC C3;
        const Q8Fix2 fix30 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c0, 96);
        const Q8Fix2 fix31 = load_q8_fix2(s_q8_fix, d2r_q8_stage(k128_iter), nf, c1, 96);
        load_B_tile(B3, s_q8, d2r_q8_stage(k128_iter), nf, 96);
        ggml_cuda_mma::mma(C3, A3, B3);
        fold_fragment_k32_fast(acc, C3, fix30, fix31, nf, dm0, dm1,
                               (uint16_t)(min0_p23 >> 16), (uint16_t)(min1_p23 >> 16));
    }
}

template <typename TileA, typename TileB, typename TileC, int NFrag>
__device__ __forceinline__ void mma_fold_k128_k64_fast(
        float (&acc)[NFrag][TileC::ne],
        const Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int k128_iter) {
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);

    {
        TileA A0;
        TileA A1;
        uint32_t min0;
        uint32_t min1;
        float2 dm0;
        float2 dm1;
        {
            Q2KWeightHalf w;
            load_q2k_weight_half_raw_fast(w, s_raw, k128_iter);
            make_A_tile<0>(A0, w);
            make_A_tile<1>(A1, w);
            min0 = min_pack4_for_pair<0, 1>(w.sc_r0_lo4, w.sc_r0_hi4);
            min1 = min_pack4_for_pair<0, 1>(w.sc_r1_lo4, w.sc_r1_hi4);
            dm0 = half2_bits_to_float2(w.dm_r0);
            dm1 = half2_bits_to_float2(w.dm_r1);
        }
#pragma unroll
        for (int nf = 0; nf < NFrag; ++nf) {
            TileC C01;
            TileB B;
            load_B_tile(B, s_q8, d2r_q8_stage(k128_iter), nf, 0);
            ggml_cuda_mma::mma(C01, A0, B);
            load_B_tile(B, s_q8, d2r_q8_stage(k128_iter), nf, 32);
            ggml_cuda_mma::mma(C01, A1, B);
            fold_fragment_k64_fast<0, 1>(
                acc, C01, s_q8_fix, d2r_q8_stage(k128_iter), nf, c0, c1, dm0, dm1, min0, min1);
        }
    }

    {
        TileA A2;
        TileA A3;
        uint32_t min0;
        uint32_t min1;
        float2 dm0;
        float2 dm1;
        {
            Q2KWeightHalf w;
            load_q2k_weight_half_raw_fast(w, s_raw, k128_iter);
            make_A_tile<2>(A2, w);
            make_A_tile<3>(A3, w);
            min0 = min_pack4_for_pair<2, 3>(w.sc_r0_lo4, w.sc_r0_hi4);
            min1 = min_pack4_for_pair<2, 3>(w.sc_r1_lo4, w.sc_r1_hi4);
            dm0 = half2_bits_to_float2(w.dm_r0);
            dm1 = half2_bits_to_float2(w.dm_r1);
        }
#pragma unroll
        for (int nf = 0; nf < NFrag; ++nf) {
            TileC C23;
            TileB B;
            load_B_tile(B, s_q8, d2r_q8_stage(k128_iter), nf, 64);
            ggml_cuda_mma::mma(C23, A2, B);
            load_B_tile(B, s_q8, d2r_q8_stage(k128_iter), nf, 96);
            ggml_cuda_mma::mma(C23, A3, B);
            fold_fragment_k64_fast<2, 3>(
                acc, C23, s_q8_fix, d2r_q8_stage(k128_iter), nf, c0, c1, dm0, dm1, min0, min1);
        }
    }
}

template <typename TileA, typename TileB, typename TileC, int T, int NFrag>
__device__ __forceinline__ void mma_fold_k32_t(
        float (&acc)[NFrag][TileC::ne],
        const Q2KWeightHalf &w,
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int k_in_q8, int warp_row0, int M, int col_lo, int col_hi) {
    TileA A;
    make_A_tile<T>(A, w);
    const uint16_t min0 = min_pack_for_t<T>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint16_t min1 = min_pack_for_t<T>(w.sc_r1_lo4, w.sc_r1_hi4);

#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        TileB B;
        TileC C;
        load_B_tile(B, s_q8, stage, nf, k_in_q8);
        ggml_cuda_mma::mma(C, A, B);
        fold_fragment_k32(acc, C, s_q8_fix, stage, nf, k_in_q8,
                          warp_row0, M, col_lo, col_hi, w.dm_r0, w.dm_r1,
                          min0, min1);
    }
}

template <typename TileA, typename TileB, typename TileC, int T0, int T1, int NFrag>
__device__ __forceinline__ void mma_fold_k64_pair_t(
        float (&acc)[NFrag][TileC::ne],
        const Q2KWeightHalf &w,
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int k_in_q8_pair, int warp_row0, int M, int col_lo, int col_hi) {
    TileA A0;
    TileA A1;
    make_A_tile<T0>(A0, w);
    make_A_tile<T1>(A1, w);
    const uint32_t min0 = min_pack4_for_pair<T0, T1>(w.sc_r0_lo4, w.sc_r0_hi4);
    const uint32_t min1 = min_pack4_for_pair<T0, T1>(w.sc_r1_lo4, w.sc_r1_hi4);

#pragma unroll
    for (int nf = 0; nf < NFrag; ++nf) {
        TileC C;
        TileB B0;
        TileB B1;
        load_B_tile(B0, s_q8, stage, nf, k_in_q8_pair);
        ggml_cuda_mma::mma(C, A0, B0);
        load_B_tile(B1, s_q8, stage, nf, k_in_q8_pair + 32);
        ggml_cuda_mma::mma(C, A1, B1);
        fold_fragment_k64(acc, C, s_q8_fix, stage, nf, k_in_q8_pair,
                          warp_row0, M, col_lo, col_hi, w.dm_r0, w.dm_r1,
                          min0, min1);
    }
}

