namespace {

#pragma once

template <ggml_type type>
int ds4_mmq_moe_vec_impl(
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
        cudaStream_t    stream) {

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
    // mmvq's per-arch batch cap. ncols_dst as computed below is
    // max(n_tokens, n_expert_used) depending on which dim we route into.
    // We follow upstream's convention: ne_y = n_tokens, ne_dst = n_expert_used.
    // So ncols_dst = n_tokens and nchannels_dst = n_expert_used.
    // FD Inc2a: n_tokens beyond the per-launch column cap no longer rejects;
    // the launch loop below splits the column dim into capped chunks.

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    // Route the pool's cudaMallocAsync / cudaFreeAsync through the same
    // stream the caller uses for kernel launches.  Required for Step 8
    // (CUDA Graph capture): pool allocations on a different stream than
    // the capture stream would invalidate the capture.
    ds4_pool_set_stream(stream);

    // 1. Quantize X into CANONICAL Q8_1 (NOT the MMQ-interleaved variant).
    //    Layout: [ne13=1, ne12=n_tokens, ne11=1, ne10_padded blocks].
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    // Step 7 task #29: experimental persistent Q8_1 scratch.  Avoids
    // pool_alloc (cudaMallocAsync) graph nodes whose pointer baked at
    // capture time may not match the address resolved at replay.  When
    // disabled (default) or when the persistent buffer is too small,
    // fall back to the pool path.  See ds4_mmq_init for setup.
    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr &&
        g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    // s11 = stride between rows of an src1 channel in source-float units.
    //       Logical src1 [K, ne11=1, ne12=n_tokens, ne13=1] - K innermost.
    // s12 = stride between channels = K * ne11 = K.
    // s13 = stride between samples = K * ne11 * ne12 = K * n_tokens.
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    // 2. mmvq stride setup. Mirror upstream's ggml_cuda_mul_mat_vec_q
    //    dispatch (mmvq.cu:1101-1136).
    //
    //    For MoE (ids != nullptr): per the dispatch math at line 1121-1130,
    //      ncols_dst          = ne2  = n_tokens
    //      nchannels_y        = ne11 = 1
    //      nchannels_dst      = ne1  = n_expert_used
    //      stride_col_y       = s12  = ne11 * (ne10_padded / QK8_1)
    //      stride_col_dst     = s2   = n_expert_used * M (token stride in dst)
    //      stride_channel_y   = s11  = ne10_padded / QK8_1
    //      stride_channel_dst = s1   = M (channel/slot stride in dst)
    //      ids_stride         = stride between rows of ids[] tensor
    //
    //    FD Inc2a stride fix: stride_col_dst was previously M, same as the
    //    channel stride.  That was invisible while every caller degenerated
    //    one dim (gate/up at n_tokens=1: col index always 0; down at
    //    n_expert_used=1: channel index always 0, and 1 * M == M keeps it
    //    bit-identical here).  At n_tokens >= 2 with n_expert_used > 1 the
    //    multi-token MoE kernel writes dst[chan*s1 + col*s2 + row], and
    //    equal strides collide (token=0,slot=1) with (token=1,slot=0).
    //    s2 = n_expert_used * M yields the row-major
    //    [token * n_expert_used + slot, M] layout the swiglu consumer
    //    expects.
    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;            // weight row stride in blocks
    const int64_t s02_chan  = (int64_t)M * s01_row;         // expert-stack stride
    const int64_t s11_y     = ne10_padded / QK8_1;          // src1 channel stride in blocks
    const int64_t s12_y     = (int64_t)1 * s11_y;           // ne11 * s11
    const int64_t s1_dst    = (int64_t)M;                   // dst channel (slot) stride
    const int64_t s2_dst    = (int64_t)n_expert_used * M;   // dst col (token) stride

    // ids_stride: stride between rows of the ids tensor in int32 elements.
    // Caller passes ids[t * n_expert_used + s], so stride between tokens
    // is n_expert_used.
    const int ids_stride = n_expert_used;

    ggml_cuda_mm_fusion_args_device fusion = {};

    cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)n_tokens * (size_t)n_expert_used * sizeof(float), stream);

    // FD Inc2a: one mmvq launch serves at most col_cap columns -- the moe
    // kernel runs one warp per column (block.y = ncols_dst) under
    // __launch_bounds__ baked per COMPILED arch + type
    // (get_mmvq_mmid_max_batch_for_device).  The runtime device cc can
    // exceed the compiled arch (CUDA_ARCH= builds run default-arch PTX on
    // newer GPUs), so the host cap MUST be looked up at the compiled arch:
    // asking the runtime cc says 8 where the compiled bounds say 7 (e.g.
    // Q2_K builds at turing_plus -> 7*warp_size threads) and the launch
    // dies with cudaErrorInvalidValue.  Wider batches run as
    // ceil(n_tokens / col_cap) launches; every per-column stride (vy, ids,
    // dst) is uniform, so a chunk is plain pointer offsets.  The single
    // quantize above already covers all columns.
    const int cc      = ggml_cuda_info().devices[dev].cc;
    const int col_cap = get_mmvq_mmid_max_batch(type, ggml_cuda_highest_compiled_arch(cc));

    for (int c0 = 0; c0 < n_tokens; c0 += col_cap) {
        const int ncols = (n_tokens - c0 < col_cap) ? (n_tokens - c0) : col_cap;
        mul_mat_vec_q_switch_type(
            /*vx=*/W, /*type_x=*/type,
            /*vy=*/(const void *)(src1_q8_1_ptr + (size_t)c0 * s12_y * sizeof(block_q8_1)),
            /*ids=*/ids + (size_t)c0 * ids_stride, /*fusion=*/fusion,
            /*dst=*/out_f32 + (int64_t)c0 * s2_dst,
            /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/ncols,
            /*stride_row_x=*/(int)s01_row,
            /*stride_col_y=*/(int)s12_y,
            /*stride_col_dst=*/(int)s2_dst,
            /*nchannels_x=*/n_experts,
            /*nchannels_y=*/1,
            /*nchannels_dst=*/n_expert_used,
            /*stride_channel_x=*/(int)s02_chan,
            /*stride_channel_y=*/(int)s11_y,
            /*stride_channel_dst=*/(int)s1_dst,
            /*nsamples_x=*/1, /*nsamples_dst=*/1,
            /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
            /*ids_stride=*/ids_stride, stream);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_vec_q_switch_type launch failed: %s (cols %d..%d cap %d)\n",
                    tag, cudaGetErrorString(err), c0, c0 + ncols - 1, col_cap);
            return -3;
        }
    }

    ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)n_tokens * (uint64_t)n_expert_used, stream);
    return 0;
}

// ---------------------------------------------------------------------------
// Aligned-SoA IQ2_XXS decode matvec (megakernel program M1-Inc1).
//
// Layout contract (see ds4_mmq.h): W_aligned = [__half dq[nblk]][pad to 64B]
// [uint2 qs[nblk*8]], nblk = n_experts * M * (K/256), block linear order equal
// to the raw tensor byte order.  Per-pair integer math is bit-identical to
// vec_dot_iq2_xxs_q8_1 (vecdotq.cuh); only the float accumulation order
// differs (per-warp-row here vs per-mmvq-tile there).  Proven +12% over the
// raw-layout vec path at the production decode shape
// in the MMQ parity suite.
__global__ void iq2_xxs_aligned_moe_vec_kernel(
        float             *out,        // [n_tokens*n_expert_used, M]
        const uint2       *qs,         // 64B-aligned code pairs
        const __half      *dq,         // block scales
        const block_q8_1  *x8,         // [n_tokens][nyb] canonical Q8_1 activations
        const int32_t     *ids,        // [n_tokens*n_expert_used] expert ids
        int                M,
        int                nb,         // IQ2_XXS blocks per row = K/256
        int                nyb,        // Q8_1 blocks per activation row
        int                n_expert_used)
{
    const int row  = blockIdx.x;
    const int slot = blockIdx.y;       // flat assignment = token*n_expert_used+slot
    const int lane = threadIdx.x;      // 32 lanes: lane covers (block b, pair p)
    // The router's NaN path emits -1 expert ids by design (same guard as
    // mul_mat_vec_q_moe): clamp the pointer math to expert 0, skip the dot
    // loop, write a clean 0.
    const int32_t id_raw = ids[slot];
    const bool invalid_id = id_raw < 0;
    const long long rbase = ((long long)(invalid_id ? 0 : id_raw) * M + row) * nb;
    x8 += (long long)(slot / n_expert_used) * nyb;

    float acc = 0.0f;
    // 32 lanes cover 4 blocks x 8 pairs per pass.
    for (int b0 = 0; !invalid_id && b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        const int p = lane & 7;
        const uint2 cw   = qs[(rbase + b) * 8 + p];   // aligned 8B load
        const uint32_t q2 = cw.x, aux32 = cw.y;
        const uint8_t *aux8 = (const uint8_t *)&q2;

        int sumi = 0;
        const int q8i = (b * 256 + p * 32) / 32;   // q8_1 block covering these 32 values
        const int *u = (const int *)x8[q8i].qs;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
            sumi = ggml_cuda_dp4a(grid0, u[k0 + 0], sumi);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = ggml_cuda_dp4a(grid1, u[k0 + 1], sumi);
        }
        const int ls = aux32 >> 27 | 1;
        sumi = sumi * ls / 8;
        const float d = __half2float(dq[rbase + b]) * __low2float(x8[q8i].ds);
        acc += d * (float)sumi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[(long long)slot * M + row] = acc;
}

// M1-Inc2 variant P: one launch covers gate and up (blockIdx.z selects the
// weight stream); nonfinite accs are zeroed in-kernel so no sanitize pass is
// needed.  Same per-warp math as iq2_xxs_aligned_moe_vec_kernel.
__global__ void iq2_xxs_aligned_moe_pair_vec_kernel(
        float             *out_gate,   // [n_tokens*n_expert_used, M]
        float             *out_up,     // [n_tokens*n_expert_used, M]
        const uint2       *qs_gate,
        const __half      *dq_gate,
        const uint2       *qs_up,
        const __half      *dq_up,
        const block_q8_1  *x8,         // [n_tokens][nyb]
        const int32_t     *ids,        // [n_tokens*n_expert_used]
        int                M,
        int                nb,
        int                nyb,
        int                n_expert_used)
{
    const int row  = blockIdx.x;
    const int slot = blockIdx.y;       // flat assignment = token*n_expert_used+slot
    const int lane = threadIdx.x;
    const uint2  *qs = blockIdx.z ? qs_up : qs_gate;
    const __half *dq = blockIdx.z ? dq_up : dq_gate;
    float        *out = blockIdx.z ? out_up : out_gate;
    const int32_t id_raw = ids[slot];
    const bool invalid_id = id_raw < 0;
    const long long rbase = ((long long)(invalid_id ? 0 : id_raw) * M + row) * nb;
    x8 += (long long)(slot / n_expert_used) * nyb;

    float acc = 0.0f;
    for (int b0 = 0; !invalid_id && b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        const int p = lane & 7;
        const uint2 cw   = qs[(rbase + b) * 8 + p];
        const uint32_t q2 = cw.x, aux32 = cw.y;
        const uint8_t *aux8 = (const uint8_t *)&q2;

        int sumi = 0;
        const int q8i = (b * 256 + p * 32) / 32;
        const int *u = (const int *)x8[q8i].qs;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
            sumi = ggml_cuda_dp4a(grid0, u[k0 + 0], sumi);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = ggml_cuda_dp4a(grid1, u[k0 + 1], sumi);
        }
        const int ls = aux32 >> 27 | 1;
        sumi = sumi * ls / 8;
        const float d = __half2float(dq[rbase + b]) * __low2float(x8[q8i].ds);
        acc += d * (float)sumi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) {
        if (!isfinite(acc)) acc = 0.0f;
        out[(long long)slot * M + row] = acc;
    }
}

// M1-Inc2 variant F: gate and up accumulated in the same warp (interleaved so
// each q8 activation block is loaded once), clamp/SwiGLU/router-weight
// epilogue folded in (semantics copied from
// ds4_mmq_moe_gate_up_mid_q8_1_qwarp32_kernel) -> mid directly.  Replaces
// quantize+gate+up+sanitize+swiglu with quantize+one launch.
__global__ void iq2_xxs_aligned_moe_gate_up_mid_kernel(
        float             *mid,        // [n_tokens*n_expert_used, M]
        const uint2       *qs_gate,
        const __half      *dq_gate,
        const uint2       *qs_up,
        const __half      *dq_up,
        const block_q8_1  *x8,         // [n_tokens][nyb]
        const int32_t     *ids,        // [n_tokens*n_expert_used]
        const float       *weights,    // [n_tokens*n_expert_used] router weights
        int                M,
        int                nb,
        int                nyb,
        int                n_expert_used,
        float              clamp)
{
    const int row  = blockIdx.x;
    const int slot = blockIdx.y;       // flat assignment = token*n_expert_used+slot
    const int lane = threadIdx.x;
    const int32_t id_raw = ids[slot];
    const bool invalid_id = id_raw < 0;
    const long long rbase = ((long long)(invalid_id ? 0 : id_raw) * M + row) * nb;
    x8 += (long long)(slot / n_expert_used) * nyb;

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int b0 = 0; !invalid_id && b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        const int p = lane & 7;
        const int q8i = (b * 256 + p * 32) / 32;
        const int *u = (const int *)x8[q8i].qs;
        const float d8 = __low2float(x8[q8i].ds);

        const uint2 cwg = qs_gate[(rbase + b) * 8 + p];
        const uint2 cwu = qs_up[(rbase + b) * 8 + p];
        const uint8_t *aux8g = (const uint8_t *)&cwg.x;
        const uint8_t *aux8u = (const uint8_t *)&cwu.x;

        int sumi_g = 0;
        int sumi_u = 0;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            {
                const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8g[k0 / 2]];
                const uint32_t signs = unpack_ksigns(cwg.y >> (7 * k0 / 2));
                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
                sumi_g = ggml_cuda_dp4a(grid0, u[k0 + 0], sumi_g);
                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
                sumi_g = ggml_cuda_dp4a(grid1, u[k0 + 1], sumi_g);
            }
            {
                const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8u[k0 / 2]];
                const uint32_t signs = unpack_ksigns(cwu.y >> (7 * k0 / 2));
                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                const int grid0  = __vsub4(grid_pos.x ^ signs0, signs0);
                sumi_u = ggml_cuda_dp4a(grid0, u[k0 + 0], sumi_u);
                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                const int grid1  = __vsub4(grid_pos.y ^ signs1, signs1);
                sumi_u = ggml_cuda_dp4a(grid1, u[k0 + 1], sumi_u);
            }
        }
        const int ls_g = cwg.y >> 27 | 1;
        const int ls_u = cwu.y >> 27 | 1;
        acc_g += __half2float(dq_gate[rbase + b]) * d8 * (float)(sumi_g * ls_g / 8);
        acc_u += __half2float(dq_up[rbase + b])   * d8 * (float)(sumi_u * ls_u / 8);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc_g += __shfl_down_sync(0xffffffffu, acc_g, off);
        acc_u += __shfl_down_sync(0xffffffffu, acc_u, off);
    }
    if (lane == 0) {
        float gate = acc_g;
        float up = acc_u;
        if (!isfinite(gate)) gate = 0.0f;
        if (!isfinite(up)) up = 0.0f;
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const float silu = gate / (1.0f + expf(-gate));
        mid[(long long)slot * M + row] = silu * up * weights[slot];
    }
}

// v0.4 V6: expert-overlap dedup for the gate_up mid kernel at DSpark
// verify widths. A live census measured
// a mean of 18.2 DISTINCT experts per 30 assignment slots at w5 (~40%
// overlap across the verify tokens); the per-slot kernel above re-reads
// every duplicate's weights from DRAM.  First-owner dedup keeps the grid
// at (M, n_slots) -- capture-safe (the decision replays from LIVE ids
// content inside baked MoE graphs), sort-free, no host id knowledge:
// each CTA exits unless it is the first slot bearing its expert id, and
// otherwise accumulates ALL matching slots (<= n_tokens; top-k is
// without replacement) as extra q8_1 columns.  Weight bytes and the iq2
// grid/sign decode collapse to distinct experts.  Per-slot int dots are
// exact and float folds stay block-major, so outputs are bitwise identical to
// the per-slot kernel on every leg, including invalid-id
// sign-zeros; timing D=18 1.53x, D=12 2.03x, D=30 0.93x -- the
// no-overlap tail is ~9% of live launches and priced).
template <int MAXM>
__global__ void iq2_xxs_aligned_moe_gate_up_mid_dedup_kernel(
        float             *mid,
        const uint2       *qs_gate,
        const __half      *dq_gate,
        const uint2       *qs_up,
        const __half      *dq_up,
        const block_q8_1  *x8,
        const int32_t     *ids,
        const float       *weights,
        int                M,
        int                nb,
        int                nyb,
        int                n_expert_used,
        int                n_slots,
        float              clamp)
{
    const int row  = blockIdx.x;
    const int slot = blockIdx.y;
    const int lane = threadIdx.x;
    const int32_t id_raw = ids[slot];

    if (id_raw < 0) {
        /* The per-slot kernel's zero path runs the epilogue with acc 0:
         * (+0)*(+0)*w = sign(w)*0 -- keep the sign bitwise. */
        if (lane == 0) mid[(long long)slot * M + row] = 0.0f * weights[slot];
        return;
    }
    for (int j = 0; j < slot; j++)
        if (ids[j] == id_raw) return;

    int msl[MAXM];
    const block_q8_1 *xcol[MAXM];
    int nm = 0;
    for (int j = slot; j < n_slots && nm < MAXM; j++)
        if (ids[j] == id_raw) {
            msl[nm] = j;
            xcol[nm] = x8 + (long long)(j / n_expert_used) * nyb;
            nm++;
        }

    const long long rbase = ((long long)id_raw * M + row) * nb;

    float acc_g[MAXM];
    float acc_u[MAXM];
#pragma unroll
    for (int m = 0; m < MAXM; m++) { acc_g[m] = 0.0f; acc_u[m] = 0.0f; }

    for (int b0 = 0; b0 < nb; b0 += 4) {
        const int b = b0 + (lane >> 3);
        const int p = lane & 7;
        const int q8i = (b * 256 + p * 32) / 32;

        const uint2 cwg = qs_gate[(rbase + b) * 8 + p];
        const uint2 cwu = qs_up[(rbase + b) * 8 + p];
        const uint8_t *aux8g = (const uint8_t *)&cwg.x;
        const uint8_t *aux8u = (const uint8_t *)&cwu.x;

        int sumi_g[MAXM];
        int sumi_u[MAXM];
#pragma unroll
        for (int m = 0; m < MAXM; m++) { sumi_g[m] = 0; sumi_u[m] = 0; }

#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            int g0g, g1g, g0u, g1u;
            {
                const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8g[k0 / 2]];
                const uint32_t signs = unpack_ksigns(cwg.y >> (7 * k0 / 2));
                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                g0g = __vsub4(grid_pos.x ^ signs0, signs0);
                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                g1g = __vsub4(grid_pos.y ^ signs1, signs1);
            }
            {
                const uint2 grid_pos = ((const uint2 *)iq2xxs_grid)[aux8u[k0 / 2]];
                const uint32_t signs = unpack_ksigns(cwu.y >> (7 * k0 / 2));
                const int signs0 = __vcmpne4(signs & 0x08040201, 0);
                g0u = __vsub4(grid_pos.x ^ signs0, signs0);
                const int signs1 = __vcmpne4(signs & 0x80402010, 0);
                g1u = __vsub4(grid_pos.y ^ signs1, signs1);
            }
#pragma unroll
            for (int m = 0; m < MAXM; m++) {
                if (m < nm) {
                    const int *u = (const int *)xcol[m][q8i].qs;
                    sumi_g[m] = ggml_cuda_dp4a(g0g, u[k0 + 0], sumi_g[m]);
                    sumi_g[m] = ggml_cuda_dp4a(g1g, u[k0 + 1], sumi_g[m]);
                    sumi_u[m] = ggml_cuda_dp4a(g0u, u[k0 + 0], sumi_u[m]);
                    sumi_u[m] = ggml_cuda_dp4a(g1u, u[k0 + 1], sumi_u[m]);
                }
            }
        }
        const int ls_g = cwg.y >> 27 | 1;
        const int ls_u = cwu.y >> 27 | 1;
        const float dg = __half2float(dq_gate[rbase + b]);
        const float du = __half2float(dq_up[rbase + b]);
#pragma unroll
        for (int m = 0; m < MAXM; m++) {
            if (m < nm) {
                const float d8 = __low2float(xcol[m][q8i].ds);
                acc_g[m] += dg * d8 * (float)(sumi_g[m] * ls_g / 8);
                acc_u[m] += du * d8 * (float)(sumi_u[m] * ls_u / 8);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MAXM; m++) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc_g[m] += __shfl_down_sync(0xffffffffu, acc_g[m], off);
            acc_u[m] += __shfl_down_sync(0xffffffffu, acc_u[m], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MAXM; m++) {
            if (m < nm) {
                float gate = acc_g[m];
                float up = acc_u[m];
                if (!isfinite(gate)) gate = 0.0f;
                if (!isfinite(up)) up = 0.0f;
                if (clamp > 1.0e-6f) {
                    if (gate > clamp) gate = clamp;
                    if (up > clamp) up = clamp;
                    if (up < -clamp) up = -clamp;
                }
                const float silu = gate / (1.0f + expf(-gate));
                mid[(long long)msl[m] * M + row] = silu * up * weights[msl[m]];
            }
        }
    }
}

template <ggml_type type>
int ds4_mmq_moe_pair_raw_vec_impl(
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
        cudaStream_t    stream) {

    if (!W_a || !W_b || !X_f32 || !ids || !out_a || !out_b) {
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
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);

    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;
    const int64_t s02_chan  = (int64_t)M * s01_row;
    const int64_t s11_y     = ne10_padded / QK8_1;
    const int64_t s12_y     = (int64_t)1 * s11_y;
    const int64_t s1_dst    = (int64_t)M;
    const int64_t s2_dst    = (int64_t)n_expert_used * M;
    const int ids_stride    = n_expert_used;
    const int cc            = ggml_cuda_info().devices[dev].cc;
    const int col_cap       = get_mmvq_mmid_max_batch(type, ggml_cuda_highest_compiled_arch(cc));
    ggml_cuda_mm_fusion_args_device fusion = {};

    const size_t out_bytes = (size_t)M * (size_t)n_tokens * (size_t)n_expert_used * sizeof(float);
    cudaMemsetAsync(out_a, 0, out_bytes, stream);
    cudaMemsetAsync(out_b, 0, out_bytes, stream);

    for (int c0 = 0; c0 < n_tokens; c0 += col_cap) {
        const int ncols = (n_tokens - c0 < col_cap) ? (n_tokens - c0) : col_cap;
        const void *vy = (const void *)(src1_q8_1_ptr + (size_t)c0 * s12_y * sizeof(block_q8_1));
        const int32_t *ids_chunk = ids + (size_t)c0 * ids_stride;
        float *out_a_chunk = out_a + (int64_t)c0 * s2_dst;
        float *out_b_chunk = out_b + (int64_t)c0 * s2_dst;

        mul_mat_vec_q_switch_type(
            /*vx=*/W_a, /*type_x=*/type,
            /*vy=*/vy, /*ids=*/ids_chunk, /*fusion=*/fusion,
            /*dst=*/out_a_chunk,
            /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/ncols,
            /*stride_row_x=*/(int)s01_row,
            /*stride_col_y=*/(int)s12_y,
            /*stride_col_dst=*/(int)s2_dst,
            /*nchannels_x=*/n_experts,
            /*nchannels_y=*/1,
            /*nchannels_dst=*/n_expert_used,
            /*stride_channel_x=*/(int)s02_chan,
            /*stride_channel_y=*/(int)s11_y,
            /*stride_channel_dst=*/(int)s1_dst,
            /*nsamples_x=*/1, /*nsamples_dst=*/1,
            /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
            /*ids_stride=*/ids_stride, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_vec_q_switch_type (a) failed: %s (cols %d..%d cap %d)\n",
                    tag, cudaGetErrorString(err), c0, c0 + ncols - 1, col_cap);
            return -3;
        }

        mul_mat_vec_q_switch_type(
            /*vx=*/W_b, /*type_x=*/type,
            /*vy=*/vy, /*ids=*/ids_chunk, /*fusion=*/fusion,
            /*dst=*/out_b_chunk,
            /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/ncols,
            /*stride_row_x=*/(int)s01_row,
            /*stride_col_y=*/(int)s12_y,
            /*stride_col_dst=*/(int)s2_dst,
            /*nchannels_x=*/n_experts,
            /*nchannels_y=*/1,
            /*nchannels_dst=*/n_expert_used,
            /*stride_channel_x=*/(int)s02_chan,
            /*stride_channel_y=*/(int)s11_y,
            /*stride_channel_dst=*/(int)s1_dst,
            /*nsamples_x=*/1, /*nsamples_dst=*/1,
            /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
            /*ids_stride=*/ids_stride, stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: mul_mat_vec_q_switch_type (b) failed: %s (cols %d..%d cap %d)\n",
                    tag, cudaGetErrorString(err), c0, c0 + ncols - 1, col_cap);
            return -4;
        }
    }

    const uint64_t out_count = (uint64_t)M * (uint64_t)n_tokens * (uint64_t)n_expert_used;
    ds4_mmq_sanitize_f32(out_a, out_count, stream);
    ds4_mmq_sanitize_f32(out_b, out_count, stream);
    return 0;
}

template <ggml_type type>
int ds4_mmq_moe_pair_vec_impl(
        const char    * tag,
        const void    * W_a,
        const void    * W_b,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_silu,
        int             M,
        int             K,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W_a || !W_b || !X_f32 || !ids || !out_silu) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d nexp=%d nused=%d\n",
                tag, M, K, n_experts, n_expert_used);
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
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    // Route the pool's cudaMallocAsync through the caller-supplied stream
    // for Step 8 / CUDA Graph compatibility.  See ds4_mmq_moe_vec_impl.
    ds4_pool_set_stream(stream);

    const int n_tokens = 1;  // fusion only supported at ncols_dst=1.

    // Quantize X (single token) into canonical Q8_1.
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr &&
        g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;
    const int64_t s02_chan  = (int64_t)M * s01_row;
    const int64_t s11_y     = ne10_padded / QK8_1;
    const int64_t s12_y     = (int64_t)1 * s11_y;
    const int64_t s1_dst    = (int64_t)M;
    const int ids_stride    = n_expert_used;

    // Configure fusion: gate=W_b (up weights), glu_op=SWIGLU.
    // mmvq's kernel will compute, for each (channel_dst, row):
    //   a = vec_dot(W_a, x); b = vec_dot(W_b, x);
    //   dst = silu(a) * b
    ggml_cuda_mm_fusion_args_device fusion = {};
    fusion.gate   = W_b;
    fusion.glu_op = GGML_GLU_OP_SWIGLU;

    cudaMemsetAsync(out_silu, 0, (size_t)M * (size_t)n_expert_used * sizeof(float), stream);

    mul_mat_vec_q_switch_type(
        /*vx=*/W_a, /*type_x=*/type,
        /*vy=*/(const void *)src1_q8_1_ptr,
        /*ids=*/ids, /*fusion=*/fusion,
        /*dst=*/out_silu,
        /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/n_tokens,
        /*stride_row_x=*/(int)s01_row,
        /*stride_col_y=*/(int)s12_y,
        /*stride_col_dst=*/(int)s1_dst,
        /*nchannels_x=*/n_experts,
        /*nchannels_y=*/1,
        /*nchannels_dst=*/n_expert_used,
        /*stride_channel_x=*/(int)s02_chan,
        /*stride_channel_y=*/(int)s11_y,
        /*stride_channel_dst=*/(int)s1_dst,
        /*nsamples_x=*/1, /*nsamples_dst=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
        /*ids_stride=*/ids_stride, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_vec_q_switch_type (fused) launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    ds4_mmq_sanitize_f32(out_silu, (uint64_t)M * (uint64_t)n_expert_used, stream);
    return 0;
}

template <ggml_type type>
int ds4_mmq_dense_vec_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        cudaStream_t  stream) {

    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (N > MMVQ_MAX_BATCH_SIZE) {
        fprintf(stderr, "%s: N=%d exceeds MMVQ_MAX_BATCH_SIZE=%d\n",
                tag, N, MMVQ_MAX_BATCH_SIZE);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    // Route the pool's cudaMallocAsync through the caller-supplied stream
    // for Step 8 / CUDA Graph compatibility.  See ds4_mmq_moe_vec_impl.
    ds4_pool_set_stream(stream);

    // Dense: no MoE, ids=null. Layout [K, N, 1, 1] for src1.
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)N * ne10_padded *
                                sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr &&
        g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    // Dense src1 layout: K innermost, N next; ne11=N, ne12=1, ne13=1.
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K * N, /*s13=*/(int64_t)K * N,
        /*ne0=*/ne10_padded, /*ne1=*/N, /*ne2=*/1, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    // Dense (no ids): per upstream dispatch (mmvq.cu:1121-1127),
    //   ncols_dst          = ne1  = N
    //   nchannels_y        = ne12 = 1
    //   nchannels_dst      = ne2  = 1
    //   stride_col_y       = s11  = ne10_padded / QK8_1
    //   stride_channel_y   = s12  = N * (ne10_padded / QK8_1)
    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;
    const int64_t s11_y     = ne10_padded / QK8_1;
    const int64_t s12_y     = (int64_t)N * s11_y;
    const int64_t s1_dst    = (int64_t)M;

    ggml_cuda_mm_fusion_args_device fusion = {};

    cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)N * sizeof(float), stream);

    mul_mat_vec_q_switch_type(
        /*vx=*/W, /*type_x=*/type,
        /*vy=*/(const void *)src1_q8_1_ptr,
        /*ids=*/nullptr, /*fusion=*/fusion,
        /*dst=*/out_f32,
        /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/N,
        /*stride_row_x=*/(int)s01_row,
        /*stride_col_y=*/(int)s11_y,
        /*stride_col_dst=*/(int)s1_dst,
        /*nchannels_x=*/1,
        /*nchannels_y=*/1,
        /*nchannels_dst=*/1,
        /*stride_channel_x=*/0,
        /*stride_channel_y=*/(int)s12_y,
        /*stride_channel_dst=*/0,
        /*nsamples_x=*/1, /*nsamples_dst=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
        /*ids_stride=*/0, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_vec_q_switch_type (dense) launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }
    ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)N, stream);
    return 0;
}

template <ggml_type type> struct ds4_mmq_vdr_mmvq_value;
template <> struct ds4_mmq_vdr_mmvq_value<GGML_TYPE_IQ2_XXS> { static constexpr int value = VDR_IQ2_XXS_Q8_1_MMVQ; };
template <> struct ds4_mmq_vdr_mmvq_value<GGML_TYPE_Q2_K>    { static constexpr int value = VDR_Q2_K_Q8_1_MMVQ; };
template <> struct ds4_mmq_vdr_mmvq_value<GGML_TYPE_Q4_K>    { static constexpr int value = VDR_Q4_K_Q8_1_MMVQ; };
template <> struct ds4_mmq_vdr_mmvq_value<GGML_TYPE_MXFP4>   { static constexpr int value = VDR_MXFP4_Q8_1_MMVQ; };
