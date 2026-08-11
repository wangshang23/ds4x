extern "C" int ds4_mmq_iq2_xxs_aligned_derepack(
        const void * W_aligned, void * raw_out,
        int M, int K, int n_experts, cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_aligned_derepack";
    if (!W_aligned || !raw_out) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_experts <= 0 || K % 256 != 0) return -1;
    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    const uint64_t n_threads = nblk * 8ull;
    iq2_xxs_aligned_derepack_kernel<<<(unsigned)((n_threads + 255ull) / 256ull), 256, 0, stream>>>(
        (unsigned char *)raw_out,
        (const uint2 *)((const char *)W_aligned + dq_bytes),
        (const __half *)W_aligned,
        nblk);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Aligned-SoA Q8_0 dense decode matvec (megakernel program M1-Inc3).
//
// block_q8_0 is 34 bytes ([half d][int8 qs[32]]), so the raw code stream is
// only 2-byte aligned — the same misalignment class proto_iq2_aligned proved
// costly.  Artifact layout (weight server --repack-q8-aligned, derived kind
// DERIVED_Q8_0_ALIGNED_DENSE): [__half dq[nblk]][pad to 64B][int8 qs[nblk*32]]
// with nblk = M * (K/32), block order equal to the raw tensor byte order.
// Unlike the IQ2 expert repack, the raw spans stay SERVED (dense tensors are
// ~6 GiB total, affordable to duplicate), so every other consumer is
// unchanged.  proto_q8_aligned.cu A/B (GB10, L2-defeating rotation, double-ref
// parity): attn_q_b 217->235, mid 2048x4096 172->218, out_a 8192x4096
// 199->230, head 224->243 GB/s; the warp-per-row accumulation is also ~1000x
// closer to the double reference than the mmvq tile order at K>=4096.
__global__ void q8_0_aligned_dense_vec_kernel(
        float             *out,        // [M]
        const int4        *qs,         // aligned codes, 2 int4 per block
        const __half      *dq,         // block scales
        const block_q8_1  *x8,         // [K/32] canonical Q8_1 activation
        int                M,
        int                nb)         // blocks per row = K/32
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const long long rbase = (long long)row * nb;

    float acc = 0.0f;
    for (int b0 = 0; b0 < nb; b0 += 32) {
        const int b = b0 + lane;
        const int4 w0 = qs[(rbase + b) * 2 + 0];   // aligned 16B loads
        const int4 w1 = qs[(rbase + b) * 2 + 1];
        const int *u = (const int *)x8[b].qs;
        int sumi = 0;
        sumi = ggml_cuda_dp4a(w0.x, u[0], sumi);
        sumi = ggml_cuda_dp4a(w0.y, u[1], sumi);
        sumi = ggml_cuda_dp4a(w0.z, u[2], sumi);
        sumi = ggml_cuda_dp4a(w0.w, u[3], sumi);
        sumi = ggml_cuda_dp4a(w1.x, u[4], sumi);
        sumi = ggml_cuda_dp4a(w1.y, u[5], sumi);
        sumi = ggml_cuda_dp4a(w1.z, u[6], sumi);
        sumi = ggml_cuda_dp4a(w1.w, u[7], sumi);
        acc += __half2float(dq[rbase + b]) * __low2float(x8[b].ds) * (float)sumi;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) out[row] = acc;
}

// Verify-width variant (v0.4 dense chase, proto_q8_aligned_nc): same aligned
// weight stream read ONCE per row, NC output columns accumulated per lane
// against col-strided q8_1 activations (which L1/L2-broadcast across rows).
// Bytes identical to the N=1 kernel, so it holds the aligned tier's rate at
// the spec-verify widths where the raw-block mmvq fallback ran 90-200 GB/s
// (proto: +17..+87% per shape, family within 4% of the weight-bytes floor).
// out is column-major [NC][M], the engine's [n_tok, out_dim] flattening.
template <int NC>
__global__ void q8_0_aligned_dense_vec_nc_kernel(
        float             *out,        // [NC * M]
        const int4        *qs,         // aligned codes, 2 int4 per block
        const __half      *dq,         // block scales
        const block_q8_1  *x8,         // [NC * nb], col stride nb
        int                M,
        int                nb)         // blocks per row = K/32
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const long long rbase = (long long)row * nb;

    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) acc[c] = 0.0f;

    for (int b0 = 0; b0 < nb; b0 += 32) {
        const int b = b0 + lane;
        const int4 w0 = qs[(rbase + b) * 2 + 0];   // aligned 16B, read once
        const int4 w1 = qs[(rbase + b) * 2 + 1];
        const float dw = __half2float(dq[rbase + b]);
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const block_q8_1 *xb = &x8[(size_t)c * nb + b];
            const int *u = (const int *)xb->qs;
            int sumi = 0;
            sumi = ggml_cuda_dp4a(w0.x, u[0], sumi);
            sumi = ggml_cuda_dp4a(w0.y, u[1], sumi);
            sumi = ggml_cuda_dp4a(w0.z, u[2], sumi);
            sumi = ggml_cuda_dp4a(w0.w, u[3], sumi);
            sumi = ggml_cuda_dp4a(w1.x, u[4], sumi);
            sumi = ggml_cuda_dp4a(w1.y, u[5], sumi);
            sumi = ggml_cuda_dp4a(w1.z, u[6], sumi);
            sumi = ggml_cuda_dp4a(w1.w, u[7], sumi);
            acc[c] += dw * __low2float(xb->ds) * (float)sumi;
        }
    }
#pragma unroll
    for (int c = 0; c < NC; c++) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[c] += __shfl_down_sync(0xffffffffu, acc[c], off);
        if (lane == 0) out[(size_t)c * M + row] = acc[c];
    }
}

extern "C" uint64_t ds4_mmq_q8_0_aligned_bytes(int M, int K) {
    if (M <= 0 || K <= 0 || K % 1024 != 0) return 0;
    const uint64_t nblk = (uint64_t)M * (uint64_t)(K / 32);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    return dq_bytes + nblk * 32u;
}

extern "C" int ds4_mmq_q8_0_aligned_dense_vec(
        const void * W_aligned, const float * X_f32, float * out_f32,
        int M, int N, int K, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q8_0_aligned_dense_vec";
    if (!W_aligned || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    // K % 1024: the kernel's 32-blocks-per-pass loop needs nb % 32 == 0.
    // N covers the decode/verify-width envelope (mmvq batch bound); K % 1024
    // also guarantees ne10_padded == K, so the q8_1 col stride is exactly nb.
    if (N < 1 || N > 8 || M <= 0 || K <= 0 || K % 1024 != 0) return -1;

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    ds4_pool_set_stream(stream);
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)N * ne10_padded * sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> q8_pool;
    // M2-Inc2a: producer-emitted q8_1 codes (qr_norm from the qkv-rms
    // kernel) -- take them and skip the quantize prelude.  Single-column
    // producers only; verify widths always quantize.
    char *x8 = N == 1 ? ds4_mmq_folded_q81(X_f32, K, 1, ne10_padded) : NULL;
    cudaError_t err;
    if (!x8) {
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        x8 = (char *)g_q81_scratch_ptr;
    } else {
        q8_pool.alloc(ctx->pool(), nbytes_q8_1);
        x8 = q8_pool.get();
    }
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)x8,
        GGML_TYPE_Q8_0, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K * N, /*s13=*/(int64_t)K * N,
        /*ne0=*/ne10_padded, /*ne1=*/N, /*ne2=*/1, /*ne3=*/1,
        stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }
    }

    const uint64_t nblk = (uint64_t)M * (uint64_t)(K / 32);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    const int4   *qsp = (const int4 *)((const char *)W_aligned + dq_bytes);
    const __half *dqp = (const __half *)W_aligned;
    const block_q8_1 *x8p = (const block_q8_1 *)x8;
    switch (N) {
    case 1:
        q8_0_aligned_dense_vec_kernel<<<(unsigned)M, 32, 0, stream>>>(
            out_f32, qsp, dqp, x8p, M, K / 32);
        break;
    case 2: q8_0_aligned_dense_vec_nc_kernel<2><<<(unsigned)M, 32, 0, stream>>>(out_f32, qsp, dqp, x8p, M, K / 32); break;
    case 3: q8_0_aligned_dense_vec_nc_kernel<3><<<(unsigned)M, 32, 0, stream>>>(out_f32, qsp, dqp, x8p, M, K / 32); break;
    case 4: q8_0_aligned_dense_vec_nc_kernel<4><<<(unsigned)M, 32, 0, stream>>>(out_f32, qsp, dqp, x8p, M, K / 32); break;
    case 5: q8_0_aligned_dense_vec_nc_kernel<5><<<(unsigned)M, 32, 0, stream>>>(out_f32, qsp, dqp, x8p, M, K / 32); break;
    case 6: q8_0_aligned_dense_vec_nc_kernel<6><<<(unsigned)M, 32, 0, stream>>>(out_f32, qsp, dqp, x8p, M, K / 32); break;
    case 7: q8_0_aligned_dense_vec_nc_kernel<7><<<(unsigned)M, 32, 0, stream>>>(out_f32, qsp, dqp, x8p, M, K / 32); break;
    case 8: q8_0_aligned_dense_vec_nc_kernel<8><<<(unsigned)M, 32, 0, stream>>>(out_f32, qsp, dqp, x8p, M, K / 32); break;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Aligned row-pair-SoA Q2_K routed-expert decode matvec (megakernel program
// M2, moe-down increment).  The production down leg runs
// mul_mat_vec_q_moe<GGML_TYPE_Q2_K, 2> over raw 84-byte block_q2_K stacks at
// ~190 GB/s (per-lane loads: one 4B qs int, four scale BYTES, one 4B half2 --
// 12 load instructions per lane-iteration).  W_aligned is a repacked copy of
// the SAME bytes keyed to that kernel's rows_per_block == 2: for the row pair
// (2p, 2p+1) of an expert, each lane-iteration needs exactly one 8B qs load
// (both rows' int), one 16B scales-window load (both rows' 8B half), one 8B
// dm load.  Layout contract (shared with the weight server
// --repack-q2k-aligned, DERIVED_Q2_K_ALIGNED_MOE, and ds4_mmq.h):
//
//   npair = n_experts * (M/2) * (K/256)      pair-blocks, expert-major then
//                                            row-pair then block (raw order)
//   [ uint2 dm2[npair] ]        {row0 half2(d,dmin), row1 half2}
//   [ pad to 64B ]
//   [ int4  sc4[npair*2] ]      half h: {row0 scales[8h..8h+3], row0 [8h+4..
//                               8h+7], row1 [8h..8h+3], row1 [8h+4..8h+7]}
//   [ pad to 64B ]
//   [ uint2 qs2[npair*16] ]     iqs: {row0 qs int[iqs], row1 qs int[iqs]}
//
// Lane mapping, scale-byte values, q8 side and the float accumulation order
// are copied verbatim from mul_mat_vec_q_moe/vec_dot_q2_K_q8_1 -> outputs are
// bit-identical to the raw path (proto_m2_q2k.cu: 240/240 parity + graph
// capture/replay, and 214 GB/s vs 154 raw on the same rotating rig).
// ---------------------------------------------------------------------------

// Same float chain as vec_dot_q2_K_q8_1_impl_mmvq; the four scale bytes come
// from the two pre-loaded 32-bit window words (byte lo+2i of the 8B window ==
// scales[scale_offset + 2i] of the raw block).
static __device__ __forceinline__ float q2_k_vec_dot_windowed(
        const int v, const int * __restrict__ u, const uint32_t w0, const uint32_t w1,
        const int lo, const half2 dm2, const float * __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int bidx = lo + 2*i;
        const uint32_t w = (bidx < 4) ? w0 : w1;
        const int sc = (int)((w >> ((bidx & 3) * 8)) & 0xFFu);

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF));

        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0);
    }
    const float2 dm2f = __half22float2(dm2);
    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

// Twin of mul_mat_vec_q_moe<GGML_TYPE_Q2_K, 2> at the down-leg call shape
// (nchannels_dst == 1, ids_stride == 1): grid (M/2, 1), block (32, ncols_dst),
// warp per assignment column.  Keeps the -1 router-id guard (task #23).
__launch_bounds__(8*32, 1)   /* MMVQ_MAX_BATCH_SIZE (mmvq.cuh) * warp; not included here */
__global__ static void q2_k_aligned_moe_vec_kernel(
        const uint2 * __restrict__ dm2_soa,
        const int4  * __restrict__ sc4_soa,
        const uint2 * __restrict__ qs2_soa,
        const block_q8_1 * __restrict__ vy, const int32_t * __restrict__ ids,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t nrows_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t ncols_dst) {
    constexpr int qi  = 16;   // QI2_K
    constexpr int vdr = 1;    // VDR_Q2_K_Q8_1_MMVQ
    constexpr int warp_size = 32;

    const uint32_t token_idx = threadIdx.y;
    const int      row0      = 2*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / QK_K;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;   // 2

    if (token_idx >= ncols_dst) {
        return;
    }

    const int32_t  id_raw     = ids[token_idx];
    const bool     invalid_id = id_raw < 0;
    const uint32_t channel_x  = invalid_id ? 0u : (uint32_t)id_raw;

    const block_q8_1 * y = vy + token_idx*stride_col_y;
    const size_t pair_base = ((size_t)channel_x * (nrows_x/2u) + (size_t)blockIdx.x)
                           * (size_t)blocks_per_row_x;

    float tmp[2] = {0.0f, 0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (QK_K/QK8_1);
        const int iqs = vdr * (threadIdx.x % (qi/vdr));

        const int bq8_offset = QR2_K * (iqs / QI8_1);
        const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);
        const int whalf = iqs / QI8_1;
        const int lo    = scale_offset - 8*whalf;
        const block_q8_1 * bq8_1 = &y[kby];

        int    u[QR2_K];
        float d8[QR2_K];
#pragma unroll
        for (int i = 0; i < QR2_K; ++i) {
            u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
            d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
        }

        const size_t pblk = pair_base + (size_t)kbx;
        const uint2 v2  = qs2_soa[pblk*16u + (unsigned)iqs];
        const uint2 dmw = dm2_soa[pblk];
        const int4  scw = sc4_soa[pblk*2u + (unsigned)whalf];
        const half2 dm0 = *(const half2 *)&dmw.x;
        const half2 dm1 = *(const half2 *)&dmw.y;

        tmp[0] += q2_k_vec_dot_windowed((int)v2.x, u, (uint32_t)scw.x, (uint32_t)scw.y, lo, dm0, d8);
        tmp[1] += q2_k_vec_dot_windowed((int)v2.y, u, (uint32_t)scw.z, (uint32_t)scw.w, lo, dm1, d8);
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    if (threadIdx.x < 2 && uint32_t(row0 + threadIdx.x) < nrows_x) {
        dst[token_idx*stride_col_dst + row0 + threadIdx.x] = tmp[threadIdx.x];
    }
}

// Exact inverse of the weight-server repack (repack_q2_k_aligned_kernel,
// tools/ds4_weight_server.cu): pair-SoA -> raw block_q2_K byte stream.  One
// thread per (raw block, qs int); p < 4 additionally restores a scales word,
// p == 0 the dm word.
__global__ static void q2_k_aligned_derepack_kernel(
        unsigned char *raw_out,
        const uint2   * __restrict__ dm2_soa,
        const int4    * __restrict__ sc4_soa,
        const uint2   * __restrict__ qs2_soa,
        uint64_t nblk,       // raw blocks total
        uint32_t nb_row,     // blocks per row = K/256
        uint32_t nrows) {    // rows per expert = M
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk * 16ull) return;
    const uint64_t g = i >> 4;
    const uint32_t p = (uint32_t)(i & 15u);
    const uint32_t b = (uint32_t)(g % nb_row);
    const uint32_t r = (uint32_t)((g / nb_row) % nrows);
    const uint64_t e = g / ((uint64_t)nb_row * nrows);
    const uint64_t pblk = ((uint64_t)e * (nrows/2u) + r/2u) * nb_row + b;
    const uint32_t parity = r & 1u;
    unsigned char *dst = raw_out + g * 84ull;
    const uint2 q = qs2_soa[pblk*16u + p];
    const uint32_t qw = parity ? q.y : q.x;
    memcpy(dst + 16u + (uint64_t)p * 4u, &qw, 4u);
    if (p < 4u) {
        const int4 s = sc4_soa[pblk*2u + (p >> 1)];
        const uint32_t sw = parity ? ((p & 1u) ? (uint32_t)s.w : (uint32_t)s.z)
                                   : ((p & 1u) ? (uint32_t)s.y : (uint32_t)s.x);
        memcpy(dst + ((p >> 1) * 8u + (p & 1u) * 4u), &sw, 4u);
    }
    if (p == 0u) {
        const uint2 d = dm2_soa[pblk];
        const uint32_t dw = parity ? d.y : d.x;
        memcpy(dst + 80u, &dw, 4u);
    }
}

extern "C" uint64_t ds4_mmq_q2_k_aligned_bytes(int M, int K, int n_experts) {
    if (M <= 0 || K <= 0 || n_experts <= 0 || K % 256 != 0 || M % 2 != 0) return 0;
    const uint64_t npair = (uint64_t)n_experts * (uint64_t)(M/2) * (uint64_t)(K / 256);
    const uint64_t dm_bytes = (npair * 8u + 63u) & ~63ull;
    const uint64_t sc_bytes = (npair * 32u + 63u) & ~63ull;
    return dm_bytes + sc_bytes + npair * 128u;
}

extern "C" int ds4_mmq_q2_K_aligned_moe_vec(
        const void * W_aligned, const float * X_f32, const int32_t * ids,
        float * out_f32, int M, int K, int n_tokens, int n_experts,
        int n_expert_used, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q2_K_aligned_moe_vec";
    if (!W_aligned || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    /* Down-leg call shape only: each (token, slot) assignment arrives as its
     * own "token" with one expert (n_expert_used == 1, ids_stride == 1). */
    if (n_expert_used != 1 || n_tokens < 1 || M <= 0 || M % 2 != 0 || K <= 0 ||
        K % 256 != 0 || n_experts <= 0) {
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    ds4_pool_set_stream(stream);

    /* Quantize verbatim from ds4_mmq_moe_vec_impl<GGML_TYPE_Q2_K> so the q8_1
     * codes feeding the twin are bit-identical to the raw path's. */
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded * sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_aligned_q81_scratch_ptr &&
        g_aligned_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_aligned_q81_scratch_ptr;
    } else if (g_q81_scratch_enabled && g_q81_scratch_ptr &&
               g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        GGML_TYPE_Q2_K, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t s12_y  = ne10_padded / QK8_1;
    const int64_t s2_dst = (int64_t)M;   /* n_expert_used == 1 */

    cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)n_tokens * sizeof(float), stream);

    const uint64_t npair = (uint64_t)n_experts * (uint64_t)(M/2) * (uint64_t)(K / 256);
    const uint64_t dm_bytes = (npair * 8u + 63u) & ~63ull;
    const uint64_t sc_bytes = (npair * 32u + 63u) & ~63ull;
    const uint2 *dm2 = (const uint2 *)W_aligned;
    const int4  *sc4 = (const int4 *)((const char *)W_aligned + dm_bytes);
    const uint2 *qs2 = (const uint2 *)((const char *)W_aligned + dm_bytes + sc_bytes);

    const int col_cap = 8;   /* MMVQ_MAX_BATCH_SIZE; matches __launch_bounds__ */
    for (int c0 = 0; c0 < n_tokens; c0 += col_cap) {
        const int ncols = (n_tokens - c0 < col_cap) ? (n_tokens - c0) : col_cap;
        dim3 grid((unsigned)(M/2), 1);
        dim3 block(32, (unsigned)ncols);
        q2_k_aligned_moe_vec_kernel<<<grid, block, 0, stream>>>(
            dm2, sc4, qs2,
            (const block_q8_1 *)(src1_q8_1_ptr + (size_t)c0 * s12_y * sizeof(block_q8_1)),
            ids + c0,
            out_f32 + (int64_t)c0 * s2_dst,
            (uint32_t)K, (uint32_t)M,
            (uint32_t)s12_y, (uint32_t)s2_dst,
            (uint32_t)ncols);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: kernel launch failed: %s (cols %d..%d)\n",
                    tag, cudaGetErrorString(err), c0, c0 + ncols - 1);
            return -3;
        }
    }

    ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)n_tokens, stream);
    return 0;
}

extern "C" int ds4_mmq_q2_K_aligned_derepack(
        const void * W_aligned, void * raw_out,
        int M, int K, int n_experts, cudaStream_t stream) {
    const char *tag = "ds4_mmq_q2_K_aligned_derepack";
    if (!W_aligned || !raw_out) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || M % 2 != 0 || K <= 0 || K % 256 != 0 || n_experts <= 0) return -1;
    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t npair = nblk / 2u;
    const uint64_t dm_bytes = (npair * 8u + 63u) & ~63ull;
    const uint64_t sc_bytes = (npair * 32u + 63u) & ~63ull;
    const uint64_t n_threads = nblk * 16ull;
    q2_k_aligned_derepack_kernel<<<(unsigned)((n_threads + 255ull) / 256ull), 256, 0, stream>>>(
        (unsigned char *)raw_out,
        (const uint2 *)W_aligned,
        (const int4 *)((const char *)W_aligned + dm_bytes),
        (const uint2 *)((const char *)W_aligned + dm_bytes + sc_bytes),
        nblk, (uint32_t)(K / 256), (uint32_t)M);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" uint64_t ds4_mmq_iq2_xxs_aligned_bytes(int M, int K, int n_experts) {
    if (M <= 0 || K <= 0 || n_experts <= 0 || K % 256 != 0) return 0;
    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    return dq_bytes + nblk * 64u;
}

// Shared single-token canonical-Q8_1 quantize for the aligned IQ2_XXS
// entries.  Returns the device pointer (persistent scratch when enabled,
// pool otherwise) or nullptr on failure; *pool must outlive the launches.
static char *iq2_aligned_quantize_xn(
        const char *tag, const float *X_f32, int K, int n_tokens,
        ggml_cuda_pool_alloc<char> *pool, cudaStream_t stream) {
    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return nullptr;
    }
    ds4_pool_set_stream(stream);
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded * sizeof(block_q8_1) / QK8_1;
    // M2-Inc2a: producer-emitted q8_1 codes (ffn_norm from the fused HC
    // stage) -- take them and skip the quantize prelude.
    char *folded = ds4_mmq_folded_q81(X_f32, K, n_tokens, ne10_padded);
    if (folded) {
        // C3-Inc4 fold twin selftest (DS4_Q8_FOLD_SELFTEST=<call budget>,
        // eager legs only -- syncs the stream): the taken sidecar must be
        // byte-identical to the fresh quantize this prelude would have run.
        // Do NOT combine with DS4_HC_STAGE_BATCH_PARITY (the probe rewrites
        // norm_out after the sidecar was emitted).
        static int fold_st = -1;
        if (fold_st < 0) {
            const char *st = getenv("DS4_Q8_FOLD_SELFTEST");
            fold_st = st && *st ? atoi(st) : 0;
            if (st && *st && fold_st <= 1) fold_st = 512;
        }
        cudaStreamCaptureStatus fold_cs = cudaStreamCaptureStatusNone;
        if (fold_st > 0) (void)cudaStreamIsCapturing(stream, &fold_cs);
        if (fold_st > 0 && fold_cs == cudaStreamCaptureStatusNone &&
            nbytes_q8_1 <= 16384u) {
            fold_st--;
            static char h[2][16384];
            pool->alloc(ctx->pool(), nbytes_q8_1);
            char *fresh = pool->get();
            quantize_row_q8_1_cuda(
                X_f32, /*ids=*/nullptr, (void *)fresh,
                GGML_TYPE_IQ2_XXS, /*ne00=*/K,
                /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
                /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
                stream);
            if (cudaGetLastError() == cudaSuccess &&
                cudaStreamSynchronize(stream) == cudaSuccess &&
                cudaMemcpy(h[0], folded, nbytes_q8_1, cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(h[1], fresh, nbytes_q8_1, cudaMemcpyDeviceToHost) == cudaSuccess) {
                fprintf(stderr, "ds4: Q8F-SELFTEST(q81 moe) K=%d %s\n", K,
                        memcmp(h[0], h[1], nbytes_q8_1) == 0 ? "PASS" : "FAIL");
            } else {
                fprintf(stderr, "ds4: Q8F-SELFTEST(q81 moe) SKIP (setup failed)\n");
            }
        }
        return folded;
    }
    char *ptr = nullptr;
    if (g_aligned_q81_scratch_ptr &&
        g_aligned_q81_scratch_bytes >= nbytes_q8_1) {
        ptr = (char *)g_aligned_q81_scratch_ptr;
    } else if (g_q81_scratch_enabled && g_q81_scratch_ptr &&
               g_q81_scratch_bytes >= nbytes_q8_1) {
        ptr = (char *)g_q81_scratch_ptr;
    } else {
        pool->alloc(ctx->pool(), nbytes_q8_1);
        ptr = pool->get();
    }
    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)ptr,
        GGML_TYPE_IQ2_XXS, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

extern "C" int ds4_mmq_iq2_xxs_aligned_moe_pair_vec(
        const void * W_gate_aligned, const void * W_up_aligned,
        const float * X_f32, const int32_t * ids,
        float * gate_out, float * up_out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_aligned_moe_pair_vec";
    if (!W_gate_aligned || !W_up_aligned || !X_f32 || !ids || !gate_out || !up_out) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (n_tokens < 1 || n_tokens > 16 || M <= 0 || K <= 0 || n_experts <= 0 ||
        n_expert_used <= 0 || n_expert_used > n_experts || K % 1024 != 0) {
        return -1;
    }
    ggml_cuda_pool_alloc<char> q8_pool;
    char *x8 = iq2_aligned_quantize_xn(tag, X_f32, K, n_tokens, &q8_pool, stream);
    if (!x8) return -2;

    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    dim3 grid((unsigned)M, (unsigned)(n_tokens * n_expert_used), 2);
    iq2_xxs_aligned_moe_pair_vec_kernel<<<grid, 32, 0, stream>>>(
        gate_out, up_out,
        (const uint2 *)((const char *)W_gate_aligned + dq_bytes),
        (const __half *)W_gate_aligned,
        (const uint2 *)((const char *)W_up_aligned + dq_bytes),
        (const __half *)W_up_aligned,
        (const block_q8_1 *)x8, ids, M, K / 256, K / 32, n_expert_used);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" int ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec(
        const void * W_gate_aligned, const void * W_up_aligned,
        const float * X_f32, const int32_t * ids, const float * weights,
        float * mid_f32,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec";
    if (!W_gate_aligned || !W_up_aligned || !X_f32 || !ids || !weights || !mid_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (n_tokens < 1 || n_tokens > 16 || M <= 0 || K <= 0 || n_experts <= 0 ||
        n_expert_used <= 0 || n_expert_used > n_experts || K % 1024 != 0) {
        return -1;
    }
    ggml_cuda_pool_alloc<char> q8_pool;
    char *x8 = iq2_aligned_quantize_xn(tag, X_f32, K, n_tokens, &q8_pool, stream);
    if (!x8) return -2;

    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    dim3 grid((unsigned)M, (unsigned)(n_tokens * n_expert_used), 1);
    const uint2  *qs_g = (const uint2 *)((const char *)W_gate_aligned + dq_bytes);
    const __half *dq_g = (const __half *)W_gate_aligned;
    const uint2  *qs_u = (const uint2 *)((const char *)W_up_aligned + dq_bytes);
    const __half *dq_u = (const __half *)W_up_aligned;
    /* v0.4 V6: verify widths dedup expert overlap (see the dedup kernel's
     * header comment).  n_tokens==1 has no cross-token overlap and keeps
     * the per-slot kernel; widths beyond the verify envelope likewise.
     * DS4_CUDA_NO_MOE_DEDUP restores the per-slot kernel (diagnostic). */
    static int moe_dedup_en = -1;
    if (moe_dedup_en < 0) moe_dedup_en = getenv("DS4_CUDA_NO_MOE_DEDUP") == NULL;
    if (moe_dedup_en && n_tokens >= 2 && n_tokens <= 8) {
        const int n_slots = n_tokens * n_expert_used;
        switch (n_tokens) {
#define DS4_GATEUP_DEDUP_CASE(NT) \
        case NT: \
            iq2_xxs_aligned_moe_gate_up_mid_dedup_kernel<NT><<<grid, 32, 0, stream>>>( \
                mid_f32, qs_g, dq_g, qs_u, dq_u, \
                (const block_q8_1 *)x8, ids, weights, M, K / 256, K / 32, \
                n_expert_used, n_slots, clamp); \
            break;
        DS4_GATEUP_DEDUP_CASE(2)
        DS4_GATEUP_DEDUP_CASE(3)
        DS4_GATEUP_DEDUP_CASE(4)
        DS4_GATEUP_DEDUP_CASE(5)
        DS4_GATEUP_DEDUP_CASE(6)
        DS4_GATEUP_DEDUP_CASE(7)
        DS4_GATEUP_DEDUP_CASE(8)
#undef DS4_GATEUP_DEDUP_CASE
        }
    } else {
        iq2_xxs_aligned_moe_gate_up_mid_kernel<<<grid, 32, 0, stream>>>(
            mid_f32, qs_g, dq_g, qs_u, dq_u,
            (const block_q8_1 *)x8, ids, weights, M, K / 256, K / 32, n_expert_used, clamp);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" int ds4_mmq_iq2_xxs_aligned_moe_vec(
        const void * W_aligned, const float * X_f32, const int32_t * ids, float * out_f32,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    const char *tag = "ds4_mmq_iq2_xxs_aligned_moe_vec";
    if (!W_aligned || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    // n_tokens 1..16 (the vec-tier envelope; each warp reads one activation
    // row selected by assignment/n_expert_used).
    // K % 1024: the lane->(block,pair) mapping covers 4 blocks per pass.
    if (n_tokens < 1 || n_tokens > 16 || M <= 0 || K <= 0 || n_experts <= 0 ||
        n_expert_used <= 0 || n_expert_used > n_experts || K % 1024 != 0) {
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    ds4_pool_set_stream(stream);

    // Quantize X into canonical Q8_1, exactly as ds4_mmq_moe_vec_impl does, so
    // the aligned path shares its activation numerics (and its persistent
    // scratch when enabled).
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = (size_t)n_tokens * ne10_padded * sizeof(block_q8_1) / QK8_1;
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
        GGML_TYPE_IQ2_XXS, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    const uint64_t nblk = (uint64_t)n_experts * (uint64_t)M * (uint64_t)(K / 256);
    const uint64_t dq_bytes = (nblk * 2u + 63u) & ~63ull;
    const __half *dq = (const __half *)W_aligned;
    const uint2  *qs = (const uint2 *)((const char *)W_aligned + dq_bytes);

    dim3 grid((unsigned)M, (unsigned)(n_tokens * n_expert_used), 1);
    iq2_xxs_aligned_moe_vec_kernel<<<grid, 32, 0, stream>>>(
        out_f32, qs, dq, (const block_q8_1 *)src1_q8_1_ptr, ids, M, K / 256,
        K / 32, n_expert_used);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: kernel launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }

    ds4_mmq_sanitize_f32(out_f32, (uint64_t)n_tokens * (uint64_t)M * (uint64_t)n_expert_used, stream);
    return 0;
}

extern "C" int ds4_mmq_q2_K_moe_down_sum6_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_down_sum6_vec_impl<GGML_TYPE_Q2_K>(
        "ds4_mmq_q2_K_moe_down_sum6_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_down_sum6_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_down_sum6_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_down_sum6_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_mxfp4_moe_down_sum6_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_down_sum6_vec_impl<GGML_TYPE_MXFP4>(
        "ds4_mmq_mxfp4_moe_down_sum6_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_gate_up_mid_vec(
        const void * W_gate, const void * W_up,
        const float * X, const int32_t * ids, const float * weights, float * mid,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    return ds4_mmq_moe_gate_up_mid_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_gate_up_mid_vec", W_gate, W_up, X, ids, weights, mid,
        M, K, n_tokens, n_experts, n_expert_used, clamp, stream);
}

extern "C" int ds4_mmq_q4_K_moe_gate_up_mid_vec(
        const void * W_gate, const void * W_up,
        const float * X, const int32_t * ids, const float * weights, float * mid,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    return ds4_mmq_moe_gate_up_mid_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_gate_up_mid_vec", W_gate, W_up, X, ids, weights, mid,
        M, K, n_tokens, n_experts, n_expert_used, clamp, stream);
}

extern "C" int ds4_mmq_mxfp4_moe_gate_up_mid_vec(
        const void * W_gate, const void * W_up,
        const float * X, const int32_t * ids, const float * weights, float * mid,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    return ds4_mmq_moe_gate_up_mid_vec_impl<GGML_TYPE_MXFP4>(
        "ds4_mmq_mxfp4_moe_gate_up_mid_vec", W_gate, W_up, X, ids, weights, mid,
        M, K, n_tokens, n_experts, n_expert_used, clamp, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_pair_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_silu,
        int M, int K, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair_vec", W_a, W_b, X, ids, out_silu,
        M, K, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_pair_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_silu,
        int M, int K, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_pair_vec", W_a, W_b, X, ids, out_silu,
        M, K, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_pair_raw_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_raw_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair_raw_vec", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_pair_raw_vec(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_raw_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_pair_raw_vec", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q8_0_dense_vec(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_vec_impl<GGML_TYPE_Q8_0>(
        "ds4_mmq_q8_0_dense_vec", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_mxfp4_dense_vec(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_vec_impl<GGML_TYPE_MXFP4>(
        "ds4_mmq_mxfp4_dense_vec", W, X, out, M, N, K, stream);
}

extern "C" int ds4_mmq_nvfp4_dense_vec(
        const void * W, const float * X, float * out,
        int M, int N, int K, cudaStream_t stream) {
    return ds4_mmq_dense_vec_impl<GGML_TYPE_NVFP4>(
        "ds4_mmq_nvfp4_dense_vec", W, X, out, M, N, K, stream);
}

// Explicit instantiations. One per quant type the public API exposes.
// Each instantiation drags in the load_tiles_<type> + vec_dot_<type>_*
// device functions from mmq.cuh, so the .o objects below contain everything
// needed to link against the public C entries.
template void mul_mat_q_case<GGML_TYPE_Q8_0>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q2_K>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ2_XXS>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q4_K>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_MXFP4>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_NVFP4>(
    ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
