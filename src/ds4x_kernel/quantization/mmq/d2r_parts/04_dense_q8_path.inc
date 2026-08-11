// ============================================================================
// Dense Q8_0 D2R (2026-07-09; proto_gemm_dense_q8_d2r.cu passes 0-5).
//
// Reads the kind-5 aligned artifact (CUDA_DERIVED_Q8_0_ALIGNED_DENSE, weight
// server --repack-q8-aligned, ADDITIVE) in place: [half dq[nblk]][pad to 64B]
// [int8 qs[nblk*32]], nblk = M*K/32, raw (row-major) block order.  m128n128
// CTA, 16 warps (8 row x 2 col, one CTA/SM), 3-stage rings at k64 cadence
// with a single barrier per iter (wait -> sync -> issue), split q8 staging
// (d4 headers @k128 + 64 B qs halves @k64, both 80 B padded rows), t-phased
// mma.m16n8k32.s8 with per-k32 float fold.
//
// Proto laws baked in (measured, do not "simplify" back):
//  - row-major weight reads, NOT contiguous-stage tiling (L2 slice camping,
//    -27% at identical instruction count);
//  - t-phased A loads, NOT both-up-front (-27%: hard A dependency at the k64
//    head);
//  - 16-warp CTA: the 2x8-warp NT128 shape spills (acc = 64 regs) and the
//    spill reloads become the top stall;
//  - invariants in plain registers (uniform-RF promotable), no smem table.
//
// vs production mul_mat_q<8,128,0> per-launch (cmttmixnsys15.sqlite): down
// [4096x2048] 3.2x, gate/up [2048x4096] 1.37x, q_b [32768x1024] 1.17x f32.
// o_proj [4096x8192] measured SLOWER (0.81x) - mmq is strong at deep K; the
// dispatch in ds4_cuda.cu keeps K=8192 on mmq.
// ============================================================================

namespace {
namespace dq8 {

constexpr int kDqMTile   = 128;
constexpr int kDqNTile   = 128;
constexpr int kDqRowWarps = 8;
constexpr int kDqColWarps = 2;
constexpr int kDqWarps   = kDqRowWarps * kDqColWarps;
constexpr int kDqThreads = 32 * kDqWarps;
constexpr int kDqStages  = 3;
constexpr int kDqRowPad  = 80;  // 64 B payload + 16 B: 20-int ldmatrix stride rotates banks
constexpr int kDqNFrag   = kDqNTile / 8;
constexpr int kDqNFragPerWarp = kDqNFrag / kDqColWarps;

constexpr size_t kDqSmemWQBytes  = (size_t)kDqStages * kDqMTile * kDqRowPad;
constexpr size_t kDqSmemQ8QBytes = (size_t)kDqStages * kDqNTile * kDqRowPad;
constexpr size_t kDqSmemQ8HBytes = (size_t)kDqStages * kDqNTile * 16;
constexpr size_t kDqSmemWDBytes  = (size_t)kDqStages * kDqMTile * sizeof(uint32_t);
constexpr size_t kDqSmemWQOff  = 0;
constexpr size_t kDqSmemQ8QOff = kDqSmemWQOff + kDqSmemWQBytes;
constexpr size_t kDqSmemQ8HOff = kDqSmemQ8QOff + kDqSmemQ8QBytes;
constexpr size_t kDqSmemWDOff  = kDqSmemQ8HOff + kDqSmemQ8HBytes;
constexpr size_t kDqSmemTotalBytes = kDqSmemWDOff + kDqSmemWDBytes;
static_assert(kDqSmemTotalBytes <= 99ull * 1024ull, "dense D2R dynamic smem exceeds sm limit");

struct DenseQ8Params {
    const char *wd_row0;       // dq plane at cta_row0 (row stride nb*2 = K/16)
    const char *wq_row0;       // qs plane at cta_row0 (row stride K)
    const char *q8_tile;       // q8 blocks at col_lo
    uint32_t q8_k128_stride_bytes;
    uint32_t wq_row_stride;    // K bytes
    int k64_iters;
    int k128_iters;
};

__device__ __forceinline__ void dq8_cp_async_4B(void *dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (pred) {
        const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                     :: "r"(smem), "l"(src));
    } else {
        *reinterpret_cast<uint32_t *>(dst) = 0u;
    }
#else
    if (pred) {
        *reinterpret_cast<uint32_t *>(dst) = *reinterpret_cast<const uint32_t *>(src);
    } else {
        *reinterpret_cast<uint32_t *>(dst) = 0u;
    }
#endif
}

__device__ __forceinline__ void dq8_issue_q8_qs(
        int8_t (&s_q8q)[kDqStages][kDqNTile][kDqRowPad],
        const DenseQ8Params &p, int k64_iter) {
    const bool pred = k64_iter < p.k64_iters;
    const char *base = p.q8_tile +
                       (uint64_t)(k64_iter >> 1) * p.q8_k128_stride_bytes +
                       16u + (uint64_t)(k64_iter & 1) * 64u;
    const int buf = k64_iter % kDqStages;
    const int chunk = d2r_tid();
    const int col = chunk >> 2;
    const int h = chunk & 3;
    void *dst = &s_q8q[buf][col][h * 16];
    const void *src = base + (uint64_t)col * sizeof(block_q8_1_mmq) + (uint64_t)h * 16u;
    cp_async_16B(dst, src, pred);
}

__device__ __forceinline__ void dq8_issue_q8_hdr(
        float (&s_q8h)[kDqStages][kDqNTile][4],
        const DenseQ8Params &p, int k128_iter) {
    if (d2r_tid() >= kDqNTile) {
        return;
    }
    const bool pred = k128_iter < p.k128_iters;
    const char *base = p.q8_tile + (uint64_t)k128_iter * p.q8_k128_stride_bytes;
    const int col = d2r_tid();
    void *dst = &s_q8h[k128_iter % kDqStages][col][0];
    const void *src = base + (uint64_t)col * sizeof(block_q8_1_mmq);
    cp_async_16B(dst, src, pred);
}

__device__ __forceinline__ void dq8_issue_w(
        int8_t (&s_wq)[kDqStages][kDqMTile][kDqRowPad],
        uint32_t (&s_wd)[kDqStages][kDqMTile],
        const DenseQ8Params &p, int k64_iter) {
    const bool pred = k64_iter < p.k64_iters;
    const int wst = k64_iter % kDqStages;
    const char *wq = p.wq_row0 + (uint64_t)k64_iter * 64u;
    const char *wd = p.wd_row0 + (uint64_t)k64_iter * 4u;
    const uint32_t wq_row_stride = p.wq_row_stride;
    const uint32_t wd_row_stride = p.wq_row_stride >> 4;  // nb*2 = K/16
    const int chunk = d2r_tid();
    const int row = chunk >> 2;
    const int h = chunk & 3;
    void *dst = &s_wq[wst][row][h * 16];
    const void *src = wq + (uint64_t)row * wq_row_stride + (uint64_t)h * 16u;
    cp_async_16B(dst, src, pred);
    if (d2r_tid() < kDqMTile) {
        void *ddst = &s_wd[wst][d2r_tid()];
        const void *dsrc = wd + (uint64_t)d2r_tid() * wd_row_stride;
        dq8_cp_async_4B(ddst, dsrc, pred);
    }
}

template <typename TileA, typename TileB, typename TileC>
__device__ __forceinline__ void dq8_mainloop(
        float (&acc)[kDqNFragPerWarp][4],
        float (&s_q8h)[kDqStages][kDqNTile][4],
        int8_t (&s_q8q)[kDqStages][kDqNTile][kDqRowPad],
        int8_t (&s_wq)[kDqStages][kDqMTile][kDqRowPad],
        uint32_t (&s_wd)[kDqStages][kDqMTile],
        const DenseQ8Params &p) {
    static_assert(TileC::ne == 4, "expected m16n8 s32 accumulator fragment");
    const int k64_iters = p.k64_iters;
    const int lane = d2r_lane();
    const int group = lane >> 2;
    const int wrow = (d2r_warp() >> 1) * 16;
    const int nf0 = (d2r_warp() & 1) * kDqNFragPerWarp;
    const int c0 = TileC::get_j(0);
    const int c1 = TileC::get_j(1);

    dq8_issue_w(s_wq, s_wd, p, 0);
    dq8_issue_q8_qs(s_q8q, p, 0);
    dq8_issue_q8_hdr(s_q8h, p, 0);
    cp_async_commit();
    dq8_issue_w(s_wq, s_wd, p, 1);
    dq8_issue_q8_qs(s_q8q, p, 1);
    dq8_issue_q8_hdr(s_q8h, p, 1);
    cp_async_commit();

    for (int i = 0; i < k64_iters; ++i) {
        // wait -> barrier -> issue: the barrier both publishes stage i CTA-wide
        // and proves all warps finished stage i-1, whose buffer ((i-1)%3 ==
        // (i+2)%3) the issues below overwrite.
        cp_async_wait_group<1>();
        __syncthreads();
        dq8_issue_w(s_wq, s_wd, p, i + 2);
        dq8_issue_q8_qs(s_q8q, p, i + 2);
        if ((i & 1) == 0) {
            dq8_issue_q8_hdr(s_q8h, p, (i >> 1) + 2);
        }
        cp_async_commit();

        const int wst = i % kDqStages;
        const int qbuf = i % kDqStages;
        const int hbuf = (i >> 1) % kDqStages;
        const uint32_t dw0_bits = s_wd[wst][wrow + group];
        const uint32_t dw1_bits = s_wd[wst][wrow + group + 8];
        const float2 dw0 = __half22float2(*reinterpret_cast<const half2 *>(&dw0_bits));
        const float2 dw1 = __half22float2(*reinterpret_cast<const half2 *>(&dw1_bits));

#pragma unroll
        for (int t = 0; t < 2; ++t) {
            TileA A;
            ggml_cuda_mma::load_ldmatrix(
                A, reinterpret_cast<const int *>(&s_wq[wst][wrow][t * 32]),
                kDqRowPad / (int)sizeof(int));
            const int tq8 = (i & 1) * 2 + t;
            const float dwr0 = t ? dw0.y : dw0.x;
            const float dwr1 = t ? dw1.y : dw1.x;
#pragma unroll
            for (int nf = 0; nf < kDqNFragPerWarp; ++nf) {
                TileB B;
                TileC C;
                ggml_cuda_mma::load_ldmatrix(
                    B, reinterpret_cast<const int *>(&s_q8q[qbuf][(nf0 + nf) * 8][t * 32]),
                    kDqRowPad / (int)sizeof(int));
                ggml_cuda_mma::mma(C, A, B);
                const float da0 = s_q8h[hbuf][(nf0 + nf) * 8 + c0][tq8];
                const float da1 = s_q8h[hbuf][(nf0 + nf) * 8 + c1][tq8];
                acc[nf][0] = fmaf((float)C.x[0], dwr0 * da0, acc[nf][0]);
                acc[nf][1] = fmaf((float)C.x[1], dwr0 * da1, acc[nf][1]);
                acc[nf][2] = fmaf((float)C.x[2], dwr1 * da0, acc[nf][2]);
                acc[nf][3] = fmaf((float)C.x[3], dwr1 * da1, acc[nf][3]);
            }
        }
    }
}

__global__ __launch_bounds__(kDqThreads, 1)
void dense_q8_d2r_kernel(const char * __restrict__ wd_plane,
                         const char * __restrict__ wq_plane,
                         const block_q8_1_mmq * __restrict__ q8,
                         float * __restrict__ out,
                         int M, int N, int K, int group_m) {
#if defined(TURING_MMA_AVAILABLE)
    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile<8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, int>;

    extern __shared__ __align__(16) char dq8_smem[];
    auto &s_wq  = *reinterpret_cast<int8_t (*)[kDqStages][kDqMTile][kDqRowPad]>(dq8_smem + kDqSmemWQOff);
    auto &s_q8q = *reinterpret_cast<int8_t (*)[kDqStages][kDqNTile][kDqRowPad]>(dq8_smem + kDqSmemQ8QOff);
    auto &s_q8h = *reinterpret_cast<float (*)[kDqStages][kDqNTile][4]>(dq8_smem + kDqSmemQ8HOff);
    auto &s_wd  = *reinterpret_cast<uint32_t (*)[kDqStages][kDqMTile]>(dq8_smem + kDqSmemWDOff);

    // Grouped supertile order: group_m row-tiles stay L2-hot while col tiles
    // stream past them.
    const int num_m = M / kDqMTile;
    const int num_n = (N + kDqNTile - 1) / kDqNTile;
    const int width = group_m * num_n;
    const int g = (int)blockIdx.x / width;
    const int rem = (int)blockIdx.x - g * width;
    int gsize = num_m - g * group_m;
    if (gsize > group_m) {
        gsize = group_m;
    }
    const int pid_m = g * group_m + rem % gsize;
    const int pid_n = rem / gsize;

    const int cta_row0 = pid_m * kDqMTile;
    const int col_lo = pid_n * kDqNTile;

    DenseQ8Params p;
    p.wd_row0 = wd_plane + (uint64_t)cta_row0 * (uint64_t)(K >> 4);
    p.wq_row0 = wq_plane + (uint64_t)cta_row0 * (uint64_t)K;
    p.q8_tile = reinterpret_cast<const char *>(q8) +
                (uint64_t)col_lo * sizeof(block_q8_1_mmq);
    p.q8_k128_stride_bytes = (uint32_t)((uint64_t)N * sizeof(block_q8_1_mmq));
    p.wq_row_stride = (uint32_t)K;
    p.k64_iters = K >> 6;
    p.k128_iters = K >> 7;

    float acc[kDqNFragPerWarp][tile_C::ne] = {};

    dq8_mainloop<tile_A, tile_B, tile_C>(acc, s_q8h, s_q8q, s_wq, s_wd, p);

    // Column-major out [N][M]; rows always in range (M % 128 == 0 validated).
    // The isfinite guard preserves the sanitize contract of the mmq path.
    const int out_col_lo = col_lo + (d2r_warp() & 1) * (kDqNFragPerWarp * 8);
    const int out_row0 = cta_row0 + ((d2r_warp() >> 1) << 4);
#pragma unroll
    for (int nf = 0; nf < kDqNFragPerWarp; ++nf) {
        const int col_frag0 = out_col_lo + nf * 8;
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = out_row0 + tile_C::get_i(l);
            const int col = col_frag0 + tile_C::get_j(l);
            if (col < N) {
                const float v = isfinite(acc[nf][l]) ? acc[nf][l] : 0.0f;
                out[(uint64_t)col * (uint64_t)M + (uint64_t)row] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(wd_plane, wq_plane, q8, out, M, N, K, group_m);
    NO_DEVICE_CODE;
#endif
}

} // namespace dq8
} // anonymous namespace

bool ds4_mmq_q8_0_dense_d2r_available(int cc) {
    static int cached_cc = -1;
    static int cached = 0;
    if (cached_cc != cc) {
        cached_cc = cc;
        cached = (GGML_CUDA_CC_IS_NVIDIA(cc) &&
                  ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE) ? 1 : 0;
    }
    return cached != 0;
}

// W_aligned = kind-5 artifact base; q8 = block_q8_1_mmq activation buffer
// (D4 layout, [k128][col] with stride N, over-allocated by >= 128 blocks for
// the guarded last col tile - the mmq Y buffer's own mmq_x_max slack covers
// this).  out is column-major [N][M] f32, every element written.
int ds4_mmq_q8_0_dense_d2r_launch(
        const void *W_aligned, const void *q8, float *out,
        int M, int N, int K, cudaStream_t stream) {
    using namespace dq8;
    if (!W_aligned || !q8 || !out ||
        M <= 0 || (M % kDqMTile) != 0 || N <= 0 || K <= 0 ||
        (K % 1024) != 0 || (K >> 6) < kDqStages) {
        fprintf(stderr, "ds4_mmq_q8_0_dense_d2r_launch: bad args M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }
    static int smem_opted = 0;
    if (!smem_opted) {
        const cudaError_t aerr = cudaFuncSetAttribute(
            dense_q8_d2r_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)kDqSmemTotalBytes);
        if (aerr != cudaSuccess) {
            fprintf(stderr, "ds4_mmq_q8_0_dense_d2r_launch: smem opt-in failed: %s\n",
                    cudaGetErrorString(aerr));
            return -2;
        }
        smem_opted = 1;
    }
    const uint64_t nblk = (uint64_t)M * (uint64_t)(K / 32);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    const char *wd_plane = (const char *)W_aligned;
    const char *wq_plane = (const char *)W_aligned + dq_bytes;
    // Grouped-order sweep verdicts from the proto: shallow K prefers
    // ungrouped (down/q_b group_m=1), K=4096 prefers 4.
    const int group_m = (K <= 2048) ? 1 : 4;
    const int num_m = M / kDqMTile;
    const int num_n = (N + kDqNTile - 1) / kDqNTile;
    const dim3 grid((unsigned)(num_m * num_n), 1, 1);
    const dim3 block(32, kDqWarps, 1);
    dense_q8_d2r_kernel<<<grid, block, kDqSmemTotalBytes, stream>>>(
        wd_plane, wq_plane, (const block_q8_1_mmq *)q8, out, M, N, K, group_m);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4_mmq_q8_0_dense_d2r_launch: launch failed: %s\n",
                cudaGetErrorString(err));
        return -3;
    }
    return 0;
}
