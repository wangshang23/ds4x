// SPDX-License-Identifier: MIT
// ds4_mmq_d2r.cu - gated D2R Q2_K MoE down-GEMM production path.
//
// The fused gate/up+SwiGLU+Q8 prefill pipeline (FusedGateUpSmem,
// iq2_gateup_fused_mainloop, gateup_iq2_swiglu_q8_d2r_kernel and its
// launcher) is Portions Copyright (c) 2026 Marco Palaferri (MIT), adapted
// from xangel82/DS4-GB10-GX10-DSpark-CUDA commit 910501e (v0.5 inc-9; the
// launch decomposition was re-derived for this fork's serving profile).

#include "ds4_mmq_d2r.cuh"

#include <type_traits>

#include "common.cuh"
#include "mmq.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

// Debug-only fill telemetry for the partial-tile lever (DS4_MMQ_D2R_STATS=1):
// per-launch column/tile fill summary from expert_bounds.  Synchronizes the
// stream - never enable on perf legs.
static bool d2r_stats_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_MMQ_D2R_STATS");
        cached = (env && env[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}

constexpr int kMTile      = 128;
constexpr int kNTile      = 64;
constexpr int kWarps      = 8;
constexpr int kThreads    = 32 * kWarps;
constexpr int kStages     = 2;
constexpr int kNFrag      = kNTile / 8;
constexpr int kRawStages  = 2;  // k256 raw slots; NT=64 stays under 48 KiB.
constexpr int kRawPairsPerWarp = 8;
constexpr int kRawHalves = 2;
constexpr int kRawQSWordsPerHalf = 8;
constexpr int kRawQSPairStride = 9;  // 8 uint2 payload + 1 uint2 pad to rotate banks.
constexpr int kRawQSChunks = kRawHalves * kRawPairsPerWarp * kRawQSWordsPerHalf;
constexpr int kRawSCChunks = kRawHalves * kRawPairsPerWarp;
constexpr int kRawDMChunks = kRawPairsPerWarp;
constexpr int kRawCopyChunks = kRawQSChunks + kRawSCChunks + kRawDMChunks;
constexpr int kIQ2RawRowsPerWarp = 16;
constexpr int kIQ2RawPairsPerRow = 8;
constexpr int kIQ2RawQCodeChunks = kIQ2RawRowsPerWarp * kIQ2RawPairsPerRow;
constexpr int kIQ2RawQCodeTrips = (kIQ2RawQCodeChunks + 31) / 32;
constexpr int kFusedNTile = 32;
constexpr int kQ8PrefetchItems = kNFrag * 8 * 9;
constexpr int kQ8PrefetchTrips = (kQ8PrefetchItems + kThreads - 1) / kThreads;
constexpr int kRawCopyTrips = (kRawCopyChunks + 31) / 32;

static_assert(kNTile == 64, "D2R production path is CFG1 NT64 only");

static void d2r_print_fill_stats(const char *tag, const int32_t *expert_bounds_dev,
                                 int n_experts, int64_t ne_get_rows,
                                 cudaStream_t stream) {
    enum { kMaxExperts = 1024 };
    static int32_t bounds[kMaxExperts + 1];
    if (n_experts > kMaxExperts) return;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return;
    if (cudaMemcpy(bounds, expert_bounds_dev, (size_t)(n_experts + 1) * sizeof(int32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return;
    int live = 0, tiles = 0, full = 0;
    long long cols = 0;
    int tail_hist[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // tail fill 1-8, 9-16, ..., 57-64
    for (int e = 0; e < n_experts; e++) {
        const int count = bounds[e + 1] - bounds[e];
        if (count <= 0) continue;
        live++;
        cols += count;
        tiles += (count + kNTile - 1) / kNTile;
        full += count / kNTile;
        const int tail = count % kNTile;
        if (tail > 0) tail_hist[(tail - 1) / 8]++;
    }
    fprintf(stderr,
            "ds4: D2R fill %s: rows=%lld cols=%lld live=%d/%d tiles=%d full=%d "
            "tail_hist[1-8..57-64]=[%d %d %d %d %d %d %d %d]\n",
            tag, (long long)ne_get_rows, cols, live, n_experts, tiles, full,
            tail_hist[0], tail_hist[1], tail_hist[2], tail_hist[3],
            tail_hist[4], tail_hist[5], tail_hist[6], tail_hist[7]);
}
static_assert(kStages == 2, "D2R raw-ring schedule expects exactly two q8 stages");
static_assert(kThreads == 256, "D2R CTA is fixed at 256 threads");
static_assert(kQ8PrefetchTrips == 3, "unexpected q8 issue trip count");
static_assert(kRawCopyTrips == 5, "unexpected raw-ring issue trip count");
static_assert(kIQ2RawQCodeTrips == 4, "unexpected IQ2 raw-ring issue trip count");

struct alignas(16) SmemInvariants {
    const char *w_base;
    const half *iq2_dq_base;
    const uint2 *iq2_qs_base;
    const char *q8_tile_base;
    float *out;
    uint32_t sc_off_bytes;
    uint32_t qs_off_bytes;
    uint32_t q8_k128_stride_bytes;
    int nb;
    int k128_iters;
    int M;
    int cta_row0;
    int col_lo;
    int col_count;
    union {
        uint32_t warp_pair0_blk[kWarps];
        uint32_t warp_row0_blk[kWarps];
    };
};

static_assert(sizeof(SmemInvariants) <= 128, "shared invariant table must stay small");

[[maybe_unused]] __device__ __forceinline__ uint32_t q2k_dm_bits(
        const uint2 * __restrict__ dm2, uint64_t pblk, int parity) {
    const uint2 d = dm2[pblk];
    return parity ? d.y : d.x;
}

__device__ __forceinline__ float2 half2_bits_to_float2(uint32_t bits) {
    half2 h;
    *reinterpret_cast<uint32_t *>(&h) = bits;
    return __half22float2(h);
}

[[maybe_unused]] __device__ __forceinline__ uint8_t q2k_scale_byte(
        const int4 * __restrict__ sc4, uint64_t pblk, int parity, int sub16) {
    const int w = sub16 >> 2;
    const int4 s = sc4[pblk * 2ull + (uint64_t)(w >> 1)];
    const uint32_t sw = parity ? ((w & 1) ? (uint32_t)s.w : (uint32_t)s.z)
                               : ((w & 1) ? (uint32_t)s.y : (uint32_t)s.x);
    return (uint8_t)((sw >> (8 * (sub16 & 3))) & 0xFFu);
}

[[maybe_unused]] __device__ __forceinline__ void q2k_scale_packs_for_half(
        const int4 * __restrict__ sc4, uint64_t pblk, int parity, int half,
        uint32_t &pack_lo4, uint32_t &pack_hi4) {
    const int4 s = sc4[pblk * 2ull + (uint64_t)half];
    if (parity) {
        pack_lo4 = (uint32_t)s.z;
        pack_hi4 = (uint32_t)s.w;
    } else {
        pack_lo4 = (uint32_t)s.x;
        pack_hi4 = (uint32_t)s.y;
    }
}

__device__ __forceinline__ void q2k_scale_packs_from_int4(
        int4 s, int parity, uint32_t &pack_lo4, uint32_t &pack_hi4) {
    if (parity) {
        pack_lo4 = (uint32_t)s.z;
        pack_hi4 = (uint32_t)s.w;
    } else {
        pack_lo4 = (uint32_t)s.x;
        pack_hi4 = (uint32_t)s.y;
    }
}

[[maybe_unused]] __device__ __forceinline__ uint8_t q2k_scale_from_packs(
        uint32_t pack_lo4, uint32_t pack_hi4, int t, int sub_in_k32) {
    const int sub = 2 * t + sub_in_k32;
    const uint32_t pack = (sub < 4) ? pack_lo4 : pack_hi4;
    return (uint8_t)((pack >> (8 * (sub & 3))) & 0xFFu);
}

template <int T, int SubInK32>
__device__ __forceinline__ uint8_t q2k_scale_from_packs_t(uint32_t pack_lo4, uint32_t pack_hi4) {
    constexpr int sub = 2 * T + SubInK32;
    const uint32_t pack = (sub < 4) ? pack_lo4 : pack_hi4;
    return (uint8_t)((pack >> (8 * (sub & 3))) & 0xFFu);
}

[[maybe_unused]] __device__ __forceinline__ uint32_t q2k_decode_scaled_reg(
        uint32_t word, int t, uint8_t scale_byte) {
    const uint32_t q = (word >> (2 * t)) & 0x03030303u;
    return q * (uint32_t)(scale_byte & 0x0Fu);
}

template <int T>
__device__ __forceinline__ uint32_t q2k_decode_scaled_reg_t(uint32_t word, uint8_t scale_byte) {
    const uint32_t q = (word >> (2 * T)) & 0x03030303u;
    return q * (uint32_t)(scale_byte & 0x0Fu);
}

__device__ __forceinline__ int sum_i8x4(uint32_t v) {
    return (int)(int8_t)(v >>  0) +
           (int)(int8_t)(v >>  8) +
           (int)(int8_t)(v >> 16) +
           (int)(int8_t)(v >> 24);
}

__device__ __forceinline__ int q8_sum16_words(const block_q8_1_mmq &b, int k0) {
    const uint32_t *p = reinterpret_cast<const uint32_t *>(b.qs + k0);
    return sum_i8x4(p[0]) + sum_i8x4(p[1]) + sum_i8x4(p[2]) + sum_i8x4(p[3]);
}

__device__ __forceinline__ void zero_16B(void *dst) {
    int4 z = make_int4(0, 0, 0, 0);
    *reinterpret_cast<int4 *>(dst) = z;
}

__device__ __forceinline__ void zero_8B(void *dst) {
    uint2 z = make_uint2(0, 0);
    *reinterpret_cast<uint2 *>(dst) = z;
}

__device__ __forceinline__ void cp_async_16B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(smem), "l"(src));
    } else {
        zero_16B(dst);
    }
#else
    if (pred) {
        *reinterpret_cast<int4 *>(dst) = *reinterpret_cast<const int4 *>(src);
    } else {
        zero_16B(dst);
    }
#endif
}

__device__ __forceinline__ void cp_async_8B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
                     :: "r"(smem), "l"(src));
    } else {
        zero_8B(dst);
    }
#else
    if (pred) {
        *reinterpret_cast<uint2 *>(dst) = *reinterpret_cast<const uint2 *>(src);
    } else {
        zero_8B(dst);
    }
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}

template <int KeepGroups>
__device__ __forceinline__ void cp_async_wait_group() {
    static_assert(KeepGroups >= 0 && KeepGroups <= 7, "bad cp.async wait_group depth");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(KeepGroups));
#endif
}

__device__ __forceinline__ void cp_async_wait_keep(int keep_groups) {
    switch (keep_groups) {
        case 0: cp_async_wait_group<0>(); break;
        case 1: cp_async_wait_group<1>(); break;
        case 2: cp_async_wait_group<2>(); break;
        case 3: cp_async_wait_group<3>(); break;
        default: cp_async_wait_group<4>(); break;
    }
}

[[maybe_unused]] __device__ __forceinline__ int cp_async_keep_for_tile(
        int tile_iter, int k_iters) {
    int keep = kStages - 1;
    const int newer = k_iters - tile_iter - 1;
    if (keep > newer) {
        keep = newer;
    }
    return keep > 0 ? keep : 0;
}

__device__ __forceinline__ int d2r_lane() {
#if defined(__CUDA_ARCH__)
    uint32_t lane;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(lane));
    return (int)lane;
#else
    return (int)threadIdx.x;
#endif
}

__device__ __forceinline__ int d2r_warp() {
#if defined(__CUDA_ARCH__)
    uint32_t warp;
    asm volatile("mov.u32 %0, %%tid.y;" : "=r"(warp));
    return (int)warp;
#else
    return (int)threadIdx.y;
#endif
}

__device__ __forceinline__ int d2r_tid() {
    return (d2r_warp() << 5) | d2r_lane();
}

__device__ __forceinline__ int d2r_group() {
    return d2r_lane() >> 2;
}

__device__ __forceinline__ int d2r_tig() {
    return d2r_lane() & 3;
}

__device__ __forceinline__ int d2r_q8_stage(int k128_iter) {
    return k128_iter & (kStages - 1);
}

__device__ __forceinline__ int d2r_raw_stage(int k256_iter) {
    return k256_iter & (kRawStages - 1);
}

template <bool FullTile>
__device__ __forceinline__ void issue_q8_prefetch_one(
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        const char * __restrict__ q8_iter_base,
        int col_count, int stage, int t) {
    constexpr int cols = kNFrag * 8;
    static_assert(cols == 64, "NT64 q8 prefetch mapping expects 64 columns");
    const int col_local = t & (cols - 1);
    const int chunk = t >> 6;
    const int nf = col_local >> 3;
    const int c = col_local & 7;
    const bool valid = FullTile ? true : (col_local < col_count);
    void *dst = (char *)&s_q8[stage][nf][c] + chunk * 16;
    const void *src = q8_iter_base + (uint64_t)col_local * sizeof(block_q8_1_mmq) + chunk * 16;
    cp_async_16B(dst, src, valid);
}

template <bool FullTile, int Iter>
__device__ __forceinline__ void issue_q8_prefetch_unrolled(
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        const char * __restrict__ q8_iter_base,
        int col_count, int stage, int tid) {
    if constexpr (Iter < kQ8PrefetchTrips) {
        const int t = tid + Iter * kThreads;
        if constexpr ((Iter + 1) * kThreads <= kQ8PrefetchItems) {
            issue_q8_prefetch_one<FullTile>(s_q8, q8_iter_base, col_count, stage, t);
        } else {
            if (t < kQ8PrefetchItems) {
                issue_q8_prefetch_one<FullTile>(s_q8, q8_iter_base, col_count, stage, t);
            }
        }
        issue_q8_prefetch_unrolled<FullTile, Iter + 1>(
            s_q8, q8_iter_base, col_count, stage, tid);
    }
}

template <bool FullTile>
__device__ __forceinline__ void issue_q8_prefetch(
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        const volatile SmemInvariants &s_inv,
        int stage, int k128_iter, int tid) {
    const char *q8_tile_base = s_inv.q8_tile_base;
    const uint32_t k128_stride = s_inv.q8_k128_stride_bytes;
    int col_count = kNTile;
    if constexpr (!FullTile) {
        col_count = s_inv.col_count;
    }
    const char *q8_iter_base = q8_tile_base + (uint64_t)k128_iter * (uint64_t)k128_stride;
    issue_q8_prefetch_unrolled<FullTile, 0>(s_q8, q8_iter_base, col_count, stage, tid);
    cp_async_commit();
}

__device__ __forceinline__ void issue_q8_prefetch_one_fast(
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        const char * __restrict__ q8_iter_base,
        int stage, int t) {
    constexpr int cols = kNFrag * 8;
    static_assert(cols == 64, "NT64 q8 prefetch mapping expects 64 columns");
    const int col_local = t & (cols - 1);
    const int c = col_local & 7;
    const int nf = col_local >> 3;
    const int chunk = t >> 6;
    void *dst = (char *)&s_q8[stage][nf][c] + chunk * 16;
    const void *src = q8_iter_base + (uint64_t)col_local * sizeof(block_q8_1_mmq) + chunk * 16;
    cp_async_16B(dst, src, true);
}

template <int Iter>
__device__ __forceinline__ void issue_q8_prefetch_fast_unrolled(
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        const char * __restrict__ q8_iter_base,
        int stage, int tid) {
    if constexpr (Iter < kQ8PrefetchTrips) {
        const int t = tid + Iter * kThreads;
        if constexpr ((Iter + 1) * kThreads <= kQ8PrefetchItems) {
            issue_q8_prefetch_one_fast(s_q8, q8_iter_base, stage, t);
        } else {
            if (t < kQ8PrefetchItems) {
                issue_q8_prefetch_one_fast(s_q8, q8_iter_base, stage, t);
            }
        }
        issue_q8_prefetch_fast_unrolled<Iter + 1>(s_q8, q8_iter_base, stage, tid);
    }
}

__device__ __forceinline__ void issue_q8_prefetch_fast(
        block_q8_1_mmq (&s_q8)[kStages][kNFrag][8],
        const volatile SmemInvariants &s_inv,
        int stage, int k128_iter) {
    const char *q8_iter_base =
        s_inv.q8_tile_base + (uint64_t)k128_iter * (uint64_t)s_inv.q8_k128_stride_bytes;
    issue_q8_prefetch_fast_unrolled<0>(s_q8, q8_iter_base, stage, d2r_tid());
    cp_async_commit();
}

/* flat-pool p5b: the column's Y row byte offset arrives as a per-thread
 * scalar (y_off) instead of being derived from the column slot.  col_local
 * is trip-invariant across the unrolled cp.async issue trips (kThreads is a
 * multiple of every fused TileN), so one register carries the offset for
 * the whole mainloop.  Identity staging (Y row == assignment slot) passes
 * col_local * sizeof(block_q8_1_mmq) and compiles to the pre-p5b address
 * math; indirect staging passes ids_src[col] * sizeof(block_q8_1_mmq)
 * against a token-compact buffer. */
template <bool FullTile, int NFrag>
__device__ __forceinline__ void issue_fused_q8_prefetch_one(
        block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const char * __restrict__ q8_iter_base,
        uint32_t y_off, int col_count, int stage, int t) {
    constexpr int TileN = NFrag * 8;
    static_assert(TileN == 8 || TileN == 16 || TileN == 32,
                  "unsupported fused D2R tile width");
    static_assert(kThreads % TileN == 0,
                  "fused q8 y_off expects trip-invariant col_local");
    const int col_local = t & (TileN - 1);
    const int chunk = t / TileN;
    const int nf = col_local >> 3;
    const int c = col_local & 7;
    const bool valid = FullTile ? true : (col_local < col_count);
    void *dst = (char *)&s_q8[stage][nf][c] + chunk * 16;
    const void *src = q8_iter_base + (uint64_t)y_off + chunk * 16;
    cp_async_16B(dst, src, valid);
}

template <bool FullTile, int NFrag, int Iter>
__device__ __forceinline__ void issue_fused_q8_prefetch_unrolled(
        block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const char * __restrict__ q8_iter_base,
        uint32_t y_off, int col_count, int stage, int tid) {
    constexpr int Items = NFrag * 8 * 9;
    constexpr int Trips = (Items + kThreads - 1) / kThreads;
    if constexpr (Iter < Trips) {
        const int t = tid + Iter * kThreads;
        if constexpr ((Iter + 1) * kThreads <= Items) {
            issue_fused_q8_prefetch_one<FullTile, NFrag>(
                s_q8, q8_iter_base, y_off, col_count, stage, t);
        } else if (t < Items) {
            issue_fused_q8_prefetch_one<FullTile, NFrag>(
                s_q8, q8_iter_base, y_off, col_count, stage, t);
        }
        issue_fused_q8_prefetch_unrolled<FullTile, NFrag, Iter + 1>(
            s_q8, q8_iter_base, y_off, col_count, stage, tid);
    }
}

template <bool FullTile, int NFrag>
__device__ __forceinline__ void issue_fused_q8_prefetch(
        block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const volatile SmemInvariants &s_inv,
        uint32_t y_off, int stage, int k128_iter) {
    constexpr int TileN = NFrag * 8;
    const char *q8_iter_base =
        s_inv.q8_tile_base +
        (uint64_t)k128_iter * (uint64_t)s_inv.q8_k128_stride_bytes;
    const int col_count = FullTile ? TileN : s_inv.col_count;
    issue_fused_q8_prefetch_unrolled<FullTile, NFrag, 0>(
        s_q8, q8_iter_base, y_off, col_count, stage, d2r_tid());
    cp_async_commit();
}

/* Identity-staged entry (Q2_K down tail tiles keep slot == Y row). */
template <bool FullTile, int NFrag>
__device__ __forceinline__ void issue_fused_q8_prefetch(
        block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        const volatile SmemInvariants &s_inv,
        int stage, int k128_iter) {
    constexpr int TileN = NFrag * 8;
    const uint32_t y_off = (uint32_t)(d2r_tid() & (TileN - 1)) *
                           (uint32_t)sizeof(block_q8_1_mmq);
    issue_fused_q8_prefetch<FullTile, NFrag>(s_q8, s_inv, y_off, stage, k128_iter);
}

struct Q8ColFixF32 {
    float d8[2];
    float sum[8];
};

template <int NFrag>
__device__ __forceinline__ void publish_q8_fix_f32(
        Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        int stage, int tid) {
    if (tid < NFrag * 8) {
        const int col_local = tid;
        const int nf = col_local >> 3;
        const int c = col_local & 7;
        const block_q8_1_mmq &qb = s_q8[stage][nf][c];
        const uint4 d2s6 = *reinterpret_cast<const uint4 *>(qb.d2s6);
        const float2 d01 = half2_bits_to_float2(d2s6.x);
        const float2 s01 = half2_bits_to_float2(d2s6.y);
        const float2 s23 = half2_bits_to_float2(d2s6.z);
        const float2 s45 = half2_bits_to_float2(d2s6.w);
        Q8ColFixF32 &f = s_q8_fix[stage][nf][c];
        f.d8[0] = d01.x;
        f.d8[1] = d01.y;
        f.sum[0] = s01.x;
        f.sum[1] = s01.y;
        f.sum[2] = s23.x;
        f.sum[3] = s23.y;
        f.sum[4] = s45.x;
        f.sum[5] = s45.y;
        f.sum[6] = d01.y * (float)q8_sum16_words(qb, 96);
        f.sum[7] = d01.y * (float)q8_sum16_words(qb, 112);
    }
}

template <int NFrag>
__device__ __forceinline__ void publish_q8_fix_f32_guarded(
        Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        const block_q8_1_mmq (&s_q8)[kStages][NFrag][8],
        int stage, int tid, int col_count) {
    if (tid < col_count) {
        const int col_local = tid;
        const int nf = col_local >> 3;
        const int c = col_local & 7;
        const block_q8_1_mmq &qb = s_q8[stage][nf][c];
        const uint4 d2s6 = *reinterpret_cast<const uint4 *>(qb.d2s6);
        const float2 d01 = half2_bits_to_float2(d2s6.x);
        const float2 s01 = half2_bits_to_float2(d2s6.y);
        const float2 s23 = half2_bits_to_float2(d2s6.z);
        const float2 s45 = half2_bits_to_float2(d2s6.w);
        Q8ColFixF32 &f = s_q8_fix[stage][nf][c];
        f.d8[0] = d01.x;
        f.d8[1] = d01.y;
        f.sum[0] = s01.x;
        f.sum[1] = s01.y;
        f.sum[2] = s23.x;
        f.sum[3] = s23.y;
        f.sum[4] = s45.x;
        f.sum[5] = s45.y;
        f.sum[6] = d01.y * (float)q8_sum16_words(qb, 96);
        f.sum[7] = d01.y * (float)q8_sum16_words(qb, 112);
    }
}

struct Q8Fix2 {
    float d8;
    float sum0;
    float sum1;
};

struct Q8Fix4 {
    float d8;
    float sum0;
    float sum1;
    float sum2;
    float sum3;
};

template <int NFrag>
__device__ __forceinline__ Q8Fix2 load_q8_fix2(
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int c, int k_in_q8) {
    const Q8ColFixF32 &sf = s_q8_fix[stage][nf][c];
    const int d8_slot = (k_in_q8 >= 64) ? 1 : 0;
    const int sub128 = k_in_q8 >> 4;
    Q8Fix2 f;
    f.d8 = sf.d8[d8_slot];
    f.sum0 = sf.sum[sub128 + 0];
    f.sum1 = sf.sum[sub128 + 1];
    return f;
}

template <int NFrag>
__device__ __forceinline__ Q8Fix4 load_q8_fix4(
        const Q8ColFixF32 (&s_q8_fix)[kStages][NFrag][8],
        int stage, int nf, int c, int k_in_q8_pair) {
    const Q8ColFixF32 &sf = s_q8_fix[stage][nf][c];
    const int d8_slot = (k_in_q8_pair >= 64) ? 1 : 0;
    const int sub128 = k_in_q8_pair >> 4;
    Q8Fix4 f;
    f.d8 = sf.d8[d8_slot];
    f.sum0 = sf.sum[sub128 + 0];
    f.sum1 = sf.sum[sub128 + 1];
    f.sum2 = sf.sum[sub128 + 2];
    f.sum3 = sf.sum[sub128 + 3];
    return f;
}

struct Q2KWeightHalf {
    uint32_t q_r0_c0;
    uint32_t q_r1_c0;
    uint32_t q_r0_c1;
    uint32_t q_r1_c1;
    uint32_t sc_r0_lo4;
    uint32_t sc_r0_hi4;
    uint32_t sc_r1_lo4;
    uint32_t sc_r1_hi4;
    uint32_t dm_r0;
    uint32_t dm_r1;
};

struct alignas(16) Q2KRawWarpStage {
    uint2 qs[kRawHalves][kRawPairsPerWarp][kRawQSPairStride];
    int4 sc[kRawHalves][kRawPairsPerWarp];
    uint2 dm[kRawPairsPerWarp];
};

static_assert(sizeof(Q2KRawWarpStage) ==
              kRawHalves * kRawPairsPerWarp * kRawQSPairStride * sizeof(uint2) +
              kRawHalves * kRawPairsPerWarp * sizeof(int4) +
              kRawPairsPerWarp * sizeof(uint2),
              "unexpected Q2K raw ring stage size");

struct alignas(16) IQ2RawWarpStage {
    uint2 qs[kIQ2RawRowsPerWarp][kIQ2RawPairsPerRow];
    half dq[kIQ2RawRowsPerWarp];
};

static_assert(sizeof(IQ2RawWarpStage) ==
              kIQ2RawRowsPerWarp * kIQ2RawPairsPerRow * sizeof(uint2) +
              kIQ2RawRowsPerWarp * sizeof(half),
              "unexpected IQ2 raw ring stage size");

template <int TileN>
struct alignas(16) FusedGateUpComputeSmem {
    static_assert(TileN == 8 || TileN == 16 || TileN == 32,
                  "unsupported fused D2R tile width");
    block_q8_1_mmq q8[kStages][TileN / 8][8];
    IQ2RawWarpStage raw[2][kWarps][kRawStages];
    uint2 grid[256];
    volatile SmemInvariants inv[2];
};

template <int TileN>
union alignas(16) FusedGateUpSmem {
    FusedGateUpComputeSmem<TileN> compute;
    float mid[TileN][kMTile];
};

static_assert(sizeof(FusedGateUpComputeSmem<kFusedNTile>) <= 48ull * 1024ull,
              "fused IQ2 gate/up shared memory exceeds 48 KiB");
static_assert(sizeof(FusedGateUpSmem<kFusedNTile>) ==
                  sizeof(FusedGateUpComputeSmem<kFusedNTile>),
              "fused post-MMA staging unexpectedly grows shared memory");
static_assert(2ull * (sizeof(FusedGateUpSmem<kFusedNTile>) +
                      kFusedNTile * sizeof(float)) <= 90ull * 1024ull,
              "fused IQ2 gate/up no longer permits two CTAs per GB10 SM");

constexpr size_t kSmemQ8StageBytes = (size_t)kNFrag * 8 * sizeof(block_q8_1_mmq);
constexpr size_t kSmemTailStageBytes = (size_t)kNFrag * 8 * sizeof(Q8ColFixF32);
constexpr size_t kSmemRawBytes = (size_t)kWarps * kRawStages * sizeof(Q2KRawWarpStage);
constexpr size_t kSmemInvBytes = sizeof(SmemInvariants);
constexpr size_t kSmemStaticBytes = (size_t)kStages * (kSmemQ8StageBytes + kSmemTailStageBytes) +
                                    kSmemRawBytes + kSmemInvBytes;
static_assert(kSmemStaticBytes <= 48ull * 1024ull, "D2R static shared memory exceeds 48 KiB");
constexpr size_t kSmemIQ2RawBytes = (size_t)kWarps * kRawStages * sizeof(IQ2RawWarpStage);
constexpr size_t kSmemIQ2GridBytes = 256u * sizeof(uint2);
constexpr size_t kSmemIQ2StaticBytes = (size_t)kStages * kSmemQ8StageBytes +
                                       kSmemIQ2RawBytes + kSmemIQ2GridBytes + kSmemInvBytes;
static_assert(kSmemIQ2StaticBytes <= 48ull * 1024ull,
              "IQ2 D2R static shared memory exceeds 48 KiB");


__device__ __forceinline__ uint32_t q2k_select_parity(uint2 v, int parity) {
    return parity ? v.y : v.x;
}

__device__ __forceinline__ bool q2k_raw_pair_valid(int warp_row0, int pair, int M) {
    return warp_row0 + 2 * pair < M;
}

template <bool FullTile, int Iter>
__device__ __forceinline__ void issue_q2k_raw_prefetch_iter(
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const char * __restrict__ w_base,
        uint32_t sc_off_bytes, uint32_t qs_off_bytes,
        uint32_t warp_pair0_blk, int nb, int raw_stage, int k256_iter,
        int warp_row0, int M, int warp, int lane) {
    static_assert(kRawCopyTrips == 5, "raw prefetch iter specialization expects five trips");
    if constexpr (Iter < 4) {
        constexpr int chunks_per_half = kRawPairsPerWarp * kRawQSWordsPerHalf;
        constexpr int half = (Iter * 32) / chunks_per_half;
        const int t = lane + Iter * 32;
        const int rem = t - half * chunks_per_half;
        const int pair = rem / kRawQSWordsPerHalf;
        const int word = rem & (kRawQSWordsPerHalf - 1);
        const bool row_valid = FullTile ? true : q2k_raw_pair_valid(warp_row0, pair, M);
        const bool valid = k256_iter < nb && row_valid;
        const uint64_t pblk = valid ? ((uint64_t)warp_pair0_blk + (uint64_t)pair * (uint64_t)nb +
                                       (uint64_t)k256_iter) : 0ull;
        void *dst = &s_raw[warp][raw_stage].qs[half][pair][word];
        const void *src = w_base + (uint64_t)qs_off_bytes +
                          pblk * 16ull * sizeof(uint2) +
                          (uint64_t)half * 8ull * sizeof(uint2) +
                          (uint64_t)word * sizeof(uint2);
        cp_async_8B(dst, src, valid);
    } else {
        if (lane < kRawSCChunks) {
            const int half = lane / kRawPairsPerWarp;
            const int pair = lane - half * kRawPairsPerWarp;
            const bool row_valid = FullTile ? true : q2k_raw_pair_valid(warp_row0, pair, M);
            const bool valid = k256_iter < nb && row_valid;
            const uint64_t pblk = valid ? ((uint64_t)warp_pair0_blk + (uint64_t)pair * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[warp][raw_stage].sc[half][pair];
            const void *src = w_base + (uint64_t)sc_off_bytes +
                              pblk * 2ull * sizeof(int4) +
                              (uint64_t)half * sizeof(int4);
            cp_async_16B(dst, src, valid);
        } else if (lane < kRawSCChunks + kRawDMChunks) {
            const int pair = lane - kRawSCChunks;
            const bool row_valid = FullTile ? true : q2k_raw_pair_valid(warp_row0, pair, M);
            const bool valid = k256_iter < nb && row_valid;
            const uint64_t pblk = valid ? ((uint64_t)warp_pair0_blk + (uint64_t)pair * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[warp][raw_stage].dm[pair];
            const void *src = w_base + pblk * sizeof(uint2);
            cp_async_8B(dst, src, valid);
        }
    }
}

template <bool FullTile, int Iter>
__device__ __forceinline__ void issue_q2k_raw_prefetch_unrolled(
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const char * __restrict__ w_base,
        uint32_t sc_off_bytes, uint32_t qs_off_bytes,
        uint32_t warp_pair0_blk, int nb, int raw_stage, int k256_iter,
        int warp_row0, int M, int warp, int lane) {
    if constexpr (Iter < kRawCopyTrips) {
        issue_q2k_raw_prefetch_iter<FullTile, Iter>(
            s_raw, w_base, sc_off_bytes, qs_off_bytes, warp_pair0_blk, nb, raw_stage, k256_iter,
            warp_row0, M, warp, lane);
        issue_q2k_raw_prefetch_unrolled<FullTile, Iter + 1>(
            s_raw, w_base, sc_off_bytes, qs_off_bytes, warp_pair0_blk, nb, raw_stage, k256_iter,
            warp_row0, M, warp, lane);
    }
}

template <bool FullTile>
__device__ __forceinline__ void issue_q2k_raw_prefetch(
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv,
        int raw_stage, int k256_iter, int warp, int lane) {
    const char *w_base = s_inv.w_base;
    const uint32_t sc_off_bytes = s_inv.sc_off_bytes;
    const uint32_t qs_off_bytes = s_inv.qs_off_bytes;
    const uint32_t warp_pair0_blk = s_inv.warp_pair0_blk[warp];
    const int nb = s_inv.nb;
    int warp_row0 = 0;
    int M = 0;
    if constexpr (!FullTile) {
        warp_row0 = s_inv.cta_row0 + (warp << 4);
        M = s_inv.M;
    }
    issue_q2k_raw_prefetch_unrolled<FullTile, 0>(
        s_raw, w_base, sc_off_bytes, qs_off_bytes, warp_pair0_blk, nb, raw_stage, k256_iter,
        warp_row0, M, warp, lane);
    cp_async_commit();
}

template <int Iter>
__device__ __forceinline__ void issue_q2k_raw_prefetch_iter_fast(
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const char * __restrict__ w_base,
        uint32_t sc_off_bytes, uint32_t qs_off_bytes,
        uint32_t warp_pair0_blk, int nb, int raw_stage, int k256_iter) {
    static_assert(kRawCopyTrips == 5, "raw prefetch iter specialization expects five trips");
    const int warp = d2r_warp();
    const int lane = d2r_lane();
    if constexpr (Iter < 4) {
        constexpr int chunks_per_half = kRawPairsPerWarp * kRawQSWordsPerHalf;
        constexpr int half = (Iter * 32) / chunks_per_half;
        const int t = lane + Iter * 32;
        const int rem = t - half * chunks_per_half;
        const int pair = rem / kRawQSWordsPerHalf;
        const int word = rem & (kRawQSWordsPerHalf - 1);
        const bool valid = k256_iter < nb;
        const uint64_t pblk = valid ? ((uint64_t)warp_pair0_blk + (uint64_t)pair * (uint64_t)nb +
                                       (uint64_t)k256_iter) : 0ull;
        void *dst = &s_raw[warp][raw_stage].qs[half][pair][word];
        const void *src = w_base + (uint64_t)qs_off_bytes +
                          pblk * 16ull * sizeof(uint2) +
                          (uint64_t)half * 8ull * sizeof(uint2) +
                          (uint64_t)word * sizeof(uint2);
        cp_async_8B(dst, src, valid);
    } else {
        if (lane < kRawSCChunks) {
            const int half = lane / kRawPairsPerWarp;
            const int pair = lane - half * kRawPairsPerWarp;
            const bool valid = k256_iter < nb;
            const uint64_t pblk = valid ? ((uint64_t)warp_pair0_blk + (uint64_t)pair * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[warp][raw_stage].sc[half][pair];
            const void *src = w_base + (uint64_t)sc_off_bytes +
                              pblk * 2ull * sizeof(int4) +
                              (uint64_t)half * sizeof(int4);
            cp_async_16B(dst, src, valid);
        } else if (lane < kRawSCChunks + kRawDMChunks) {
            const int pair = lane - kRawSCChunks;
            const bool valid = k256_iter < nb;
            const uint64_t pblk = valid ? ((uint64_t)warp_pair0_blk + (uint64_t)pair * (uint64_t)nb +
                                           (uint64_t)k256_iter) : 0ull;
            void *dst = &s_raw[warp][raw_stage].dm[pair];
            const void *src = w_base + pblk * sizeof(uint2);
            cp_async_8B(dst, src, valid);
        }
    }
}

template <int Iter>
__device__ __forceinline__ void issue_q2k_raw_prefetch_fast_unrolled(
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const char * __restrict__ w_base,
        uint32_t sc_off_bytes, uint32_t qs_off_bytes,
        uint32_t warp_pair0_blk, int nb, int raw_stage, int k256_iter) {
    if constexpr (Iter < kRawCopyTrips) {
        issue_q2k_raw_prefetch_iter_fast<Iter>(
            s_raw, w_base, sc_off_bytes, qs_off_bytes, warp_pair0_blk, nb, raw_stage, k256_iter);
        issue_q2k_raw_prefetch_fast_unrolled<Iter + 1>(
            s_raw, w_base, sc_off_bytes, qs_off_bytes, warp_pair0_blk, nb, raw_stage, k256_iter);
    }
}

__device__ __forceinline__ void issue_q2k_raw_prefetch_fast(
        Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        const volatile SmemInvariants &s_inv,
        int raw_stage, int k256_iter) {
    const int warp = d2r_warp();
    issue_q2k_raw_prefetch_fast_unrolled<0>(
        s_raw, s_inv.w_base, s_inv.sc_off_bytes, s_inv.qs_off_bytes,
        s_inv.warp_pair0_blk[warp], s_inv.nb, raw_stage, k256_iter);
    cp_async_commit();
}

__device__ __forceinline__ void load_q2k_weight_half_raw(
        Q2KWeightHalf &w,
        const Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        int warp, int raw_stage, bool row0_ok, bool row1_ok,
        int parity, int half, int group, int tig) {
    const Q2KRawWarpStage &raw = s_raw[warp][raw_stage];
    const int pair0 = group >> 1;
    const int pair1 = pair0 + 4;
    const int p_base = half * 8;
    const int p0 = p_base + tig;
    const int p1 = p_base + tig + 4;

    if (row0_ok) {
        const uint2 q0 = raw.qs[half][pair0][p0 - p_base];
        const uint2 q1 = raw.qs[half][pair0][p1 - p_base];
        w.q_r0_c0 = q2k_select_parity(q0, parity);
        w.q_r0_c1 = q2k_select_parity(q1, parity);
        q2k_scale_packs_from_int4(raw.sc[half][pair0], parity, w.sc_r0_lo4, w.sc_r0_hi4);
        w.dm_r0 = q2k_select_parity(raw.dm[pair0], parity);
    } else {
        w.q_r0_c0 = 0;
        w.q_r0_c1 = 0;
        w.sc_r0_lo4 = 0;
        w.sc_r0_hi4 = 0;
        w.dm_r0 = 0;
    }
    if (row1_ok) {
        const uint2 q0 = raw.qs[half][pair1][p0 - p_base];
        const uint2 q1 = raw.qs[half][pair1][p1 - p_base];
        w.q_r1_c0 = q2k_select_parity(q0, parity);
        w.q_r1_c1 = q2k_select_parity(q1, parity);
        q2k_scale_packs_from_int4(raw.sc[half][pair1], parity, w.sc_r1_lo4, w.sc_r1_hi4);
        w.dm_r1 = q2k_select_parity(raw.dm[pair1], parity);
    } else {
        w.q_r1_c0 = 0;
        w.q_r1_c1 = 0;
        w.sc_r1_lo4 = 0;
        w.sc_r1_hi4 = 0;
        w.dm_r1 = 0;
    }
}

__device__ __forceinline__ void load_q2k_weight_half_raw_fast(
        Q2KWeightHalf &w,
        const Q2KRawWarpStage (&s_raw)[kWarps][kRawStages],
        int k128_iter) {
    const int lane = d2r_lane();
    const int group = lane >> 2;
    load_q2k_weight_half_raw(
        w, s_raw, d2r_warp(), d2r_raw_stage(k128_iter >> 1),
        true, true, group & 1, k128_iter & 1, group, lane & 3);
}

