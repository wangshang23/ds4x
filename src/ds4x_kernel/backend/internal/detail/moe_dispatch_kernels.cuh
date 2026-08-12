#ifndef DS4X_BACKEND_MOE_DISPATCH_KERNELS_CUH
#define DS4X_BACKEND_MOE_DISPATCH_KERNELS_CUH

#include "../backend_common.cuh"

/* One matrix pass over all superblocks for this thread's 4 C elements
 * (tokens mtokA/mtokB x rows n0/n0+1). Returns the quarter-tree-reduced
 * values in r[4] with the exact reference ordering. */
__device__ __forceinline__ static void moe_tile16_mma_pass(
        const char *wrow,            /* row0 base of this matrix */
        uint64_t row_bytes,
        const cuda_block_q8_K (*sxq)[16],
        uint32_t xq_blocks,
        uint32_t lane,
        float r[4]) {
    const uint32_t mtokA = lane >> 2u;
    const uint32_t mtokB = mtokA + 8u;
    const uint32_t n0 = (lane & 3u) * 2u;
    float s0[8] = {0,0,0,0,0,0,0,0};
    float s1[8] = {0,0,0,0,0,0,0,0};
    float s2[8] = {0,0,0,0,0,0,0,0};
    float s3[8] = {0,0,0,0,0,0,0,0};
    for (uint32_t b = 0; b < xq_blocks; b++) {
        const uint4 hdr0 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)n0 * row_bytes) + b);
        const uint4 hdr1 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)(n0 + 1u) * row_bytes) + b);
        const uint32_t *qw = (const uint32_t *)(((const cuda_block_q4_K *)(wrow + (uint64_t)(lane >> 2u) * row_bytes) + b)->qs);
        uint32_t w8[8];
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) w8[k] = qw[k * 4u + (lane & 3u)];
        const int8_t *aqsA = sxq[mtokA][b].qs;
        const int8_t *aqsB = sxq[mtokB][b].qs;
        int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
        int m0 = 0, m1 = 0, m2 = 0, m3 = 0;
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const int shift = (j & 1u) ? 4 : 0;
            const uint32_t koff = (lane & 3u) * 4u;
            const uint32_t a0 = *(const uint32_t *)(aqsA + j * 32u + koff);
            const uint32_t a1 = *(const uint32_t *)(aqsB + j * 32u + koff);
            const uint32_t a2 = *(const uint32_t *)(aqsA + j * 32u + 16u + koff);
            const uint32_t a3 = *(const uint32_t *)(aqsB + j * 32u + 16u + koff);
            const uint32_t b0 = (w8[(j >> 1u) * 2u + 0u] >> shift) & 0x0f0f0f0fu;
            const uint32_t b1 = (w8[(j >> 1u) * 2u + 1u] >> shift) & 0x0f0f0f0fu;
            int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            mma16_m16n8k32_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
            uint8_t sc0, sm0, sc1, sm1;
            dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr0.y, &sc0, &sm0);
            dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr1.y, &sc1, &sm1);
            const int bsA = (int)sxq[mtokA][b].bsums[2u * j] + (int)sxq[mtokA][b].bsums[2u * j + 1u];
            const int bsB = (int)sxq[mtokB][b].bsums[2u * j] + (int)sxq[mtokB][b].bsums[2u * j + 1u];
            i0 += (int)sc0 * c0;
            i1 += (int)sc1 * c1;
            i2 += (int)sc0 * c2;
            i3 += (int)sc1 * c3;
            m0 += (int)sm0 * bsA;
            m1 += (int)sm1 * bsA;
            m2 += (int)sm0 * bsB;
            m3 += (int)sm1 * bsB;
        }
        const float ydA = sxq[mtokA][b].d;
        const float ydB = sxq[mtokB][b].d;
        const float xd0 = dev_f16_to_f32((uint16_t)(hdr0.x & 0xffffu));
        const float xmin0 = dev_f16_to_f32((uint16_t)(hdr0.x >> 16u));
        const float xd1 = dev_f16_to_f32((uint16_t)(hdr1.x & 0xffffu));
        const float xmin1 = dev_f16_to_f32((uint16_t)(hdr1.x >> 16u));
        const uint32_t sl = b & 7u;
        s0[sl] += ydA * xd0 * (float)i0 - ydA * xmin0 * (float)m0;
        s1[sl] += ydA * xd1 * (float)i1 - ydA * xmin1 * (float)m1;
        s2[sl] += ydB * xd0 * (float)i2 - ydB * xmin0 * (float)m2;
        s3[sl] += ydB * xd1 * (float)i3 - ydB * xmin1 * (float)m3;
    }
    {
        float a0 = s0[0] + s0[4], a1 = s0[1] + s0[5], a2 = s0[2] + s0[6], a3 = s0[3] + s0[7];
        r[0] = (a0 + a2) + (a1 + a3);
        a0 = s1[0] + s1[4]; a1 = s1[1] + s1[5]; a2 = s1[2] + s1[6]; a3 = s1[3] + s1[7];
        r[1] = (a0 + a2) + (a1 + a3);
        a0 = s2[0] + s2[4]; a1 = s2[1] + s2[5]; a2 = s2[2] + s2[6]; a3 = s2[3] + s2[7];
        r[2] = (a0 + a2) + (a1 + a3);
        a0 = s3[0] + s3[4]; a1 = s3[1] + s3[5]; a2 = s3[2] + s3[6]; a3 = s3[3] + s3[7];
        r[3] = (a0 + a2) + (a1 + a3);
    }
}



template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q4K_tile16_mma_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    extern __shared__ unsigned char t16_sh[];
    cuda_block_q8_K (*sxq)[16] = (cuda_block_q8_K (*)[16])t16_sh; /* [16][16] */
    __shared__ uint32_t s_pair[16];
    __shared__ uint32_t s_tok[16];
    __shared__ uint32_t s_slot[16];
    __shared__ uint32_t s_np;
    if (threadIdx.x == 0) {
        uint32_t np = 0;
        for (; np < 16u; np++) {
            uint32_t local_pair = local_start + np;
            if (local_pair >= counts[expert]) break;
            uint32_t pr = sorted_pairs[offsets[expert] + local_pair];
            s_pair[np] = pr;
            s_tok[np] = pr / n_expert;
            s_slot[np] = pr - s_tok[np] * n_expert;
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    if (xq_blocks <= 16u) {
        const uint32_t words_per_tok = xq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
        for (uint32_t i = threadIdx.x; i < np * words_per_tok; i += blockDim.x) {
            uint32_t p = i / words_per_tok;
            uint32_t w = i - p * words_per_tok;
            ((uint32_t *)sxq[p])[w] = ((const uint32_t *)(xq + (uint64_t)s_tok[p] * xq_blocks))[w];
        }
        /* zero-fill missing pairs so the A fragments are defined */
        const uint32_t total_words = 16u * words_per_tok;
        for (uint32_t i = threadIdx.x + np * words_per_tok; i < total_words; i += blockDim.x) {
            ((uint32_t *)t16_sh)[i] = 0u;
        }
        __syncthreads();
    }
    const uint32_t mtokA = lane >> 2u;
    const uint32_t mtokB = mtokA + 8u;
    const uint32_t n0 = (lane & 3u) * 2u;
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN + rr * 64u + warp * 8u;
        if (row0 >= expert_mid_dim) continue;
        const char *grow = gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row0 * gate_row_bytes;
        const char *urow = up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row0 * gate_row_bytes;
        float gr[4], ur[4];
        moe_tile16_mma_pass(grow, gate_row_bytes, (const cuda_block_q8_K (*)[16])sxq, xq_blocks, lane, gr);
        moe_tile16_mma_pass(urow, gate_row_bytes, (const cuda_block_q8_K (*)[16])sxq, xq_blocks, lane, ur);
#pragma unroll
        for (uint32_t e = 0; e < 4u; e++) {
            const uint32_t p = (e < 2u) ? mtokA : mtokB;
            const uint32_t row = row0 + n0 + (e & 1u);
            if (p >= np || row >= expert_mid_dim) continue;
            float gate = gr[e];
            float up = ur[e];
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)s_pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up *
                           weights[(uint64_t)s_tok[p] * n_expert + s_slot[p]];
        }
    }
}



template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_tile16_mma_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    extern __shared__ unsigned char t16_sh[];
    __shared__ uint32_t s_pair[16];
    __shared__ uint32_t s_np;
    if (threadIdx.x == 0) {
        uint32_t np = 0;
        for (; np < 16u; np++) {
            uint32_t local_pair = local_start + np;
            if (local_pair >= counts[expert]) break;
            s_pair[np] = sorted_pairs[offsets[expert] + local_pair];
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    if (midq_blocks <= 16u) {
        const uint32_t words_per_tok = midq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
        for (uint32_t i = threadIdx.x; i < np * words_per_tok; i += blockDim.x) {
            uint32_t p = i / words_per_tok;
            uint32_t w = i - p * words_per_tok;
            ((uint32_t *)t16_sh)[i] = ((const uint32_t *)(midq + (uint64_t)s_pair[p] * midq_blocks))[w];
        }
        const uint32_t total_words = 16u * words_per_tok;
        for (uint32_t i = threadIdx.x + np * words_per_tok; i < total_words; i += blockDim.x) {
            ((uint32_t *)t16_sh)[i] = 0u;
        }
        __syncthreads();
    }
    const uint32_t mtokA = lane >> 2u;
    const uint32_t mtokB = mtokA + 8u;
    const uint32_t n0 = (lane & 3u) * 2u;
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN + rr * 64u + warp * 8u;
        if (row0 >= out_dim) continue;
        const char *wrow = down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row0 * down_row_bytes;
        float s0[8] = {0,0,0,0,0,0,0,0};
        float s1[8] = {0,0,0,0,0,0,0,0};
        float s2[8] = {0,0,0,0,0,0,0,0};
        float s3[8] = {0,0,0,0,0,0,0,0};
        for (uint32_t b = 0; b < midq_blocks; b++) {
            const uint4 hdr0 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)n0 * down_row_bytes) + b);
            const uint4 hdr1 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)(n0 + 1u) * down_row_bytes) + b);
            const uint32_t *qw = (const uint32_t *)(((const cuda_block_q4_K *)(wrow + (uint64_t)(lane >> 2u) * down_row_bytes) + b)->qs);
            uint32_t w8[8];
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) w8[k] = qw[k * 4u + (lane & 3u)];
            /* activation rows: midq_blocks stride within the staged region */
            const int8_t *aqsA = ((const cuda_block_q8_K *)t16_sh + (uint64_t)mtokA * midq_blocks + b)->qs;
            const int8_t *aqsB = ((const cuda_block_q8_K *)t16_sh + (uint64_t)mtokB * midq_blocks + b)->qs;
            const cuda_block_q8_K *blkA = (const cuda_block_q8_K *)t16_sh + (uint64_t)mtokA * midq_blocks + b;
            const cuda_block_q8_K *blkB = (const cuda_block_q8_K *)t16_sh + (uint64_t)mtokB * midq_blocks + b;
            int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
            int m0 = 0, m1 = 0, m2 = 0, m3 = 0;
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const int shift = (j & 1u) ? 4 : 0;
                const uint32_t koff = (lane & 3u) * 4u;
                const uint32_t a0 = *(const uint32_t *)(aqsA + j * 32u + koff);
                const uint32_t a1 = *(const uint32_t *)(aqsB + j * 32u + koff);
                const uint32_t a2 = *(const uint32_t *)(aqsA + j * 32u + 16u + koff);
                const uint32_t a3 = *(const uint32_t *)(aqsB + j * 32u + 16u + koff);
                const uint32_t b0 = (w8[(j >> 1u) * 2u + 0u] >> shift) & 0x0f0f0f0fu;
                const uint32_t b1 = (w8[(j >> 1u) * 2u + 1u] >> shift) & 0x0f0f0f0fu;
                int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                mma16_m16n8k32_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
                uint8_t sc0, sm0, sc1, sm1;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr0.y, &sc0, &sm0);
                dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr1.y, &sc1, &sm1);
                const int bsA = (int)blkA->bsums[2u * j] + (int)blkA->bsums[2u * j + 1u];
                const int bsB = (int)blkB->bsums[2u * j] + (int)blkB->bsums[2u * j + 1u];
                i0 += (int)sc0 * c0;
                i1 += (int)sc1 * c1;
                i2 += (int)sc0 * c2;
                i3 += (int)sc1 * c3;
                m0 += (int)sm0 * bsA;
                m1 += (int)sm1 * bsA;
                m2 += (int)sm0 * bsB;
                m3 += (int)sm1 * bsB;
            }
            const float ydA = blkA->d;
            const float ydB = blkB->d;
            const float xd0 = dev_f16_to_f32((uint16_t)(hdr0.x & 0xffffu));
            const float xmin0 = dev_f16_to_f32((uint16_t)(hdr0.x >> 16u));
            const float xd1 = dev_f16_to_f32((uint16_t)(hdr1.x & 0xffffu));
            const float xmin1 = dev_f16_to_f32((uint16_t)(hdr1.x >> 16u));
            const uint32_t sl = b & 7u;
            s0[sl] += ydA * xd0 * (float)i0 - ydA * xmin0 * (float)m0;
            s1[sl] += ydA * xd1 * (float)i1 - ydA * xmin1 * (float)m1;
            s2[sl] += ydB * xd0 * (float)i2 - ydB * xmin0 * (float)m2;
            s3[sl] += ydB * xd1 * (float)i3 - ydB * xmin1 * (float)m3;
        }
        float rr4[4];
        {
            float a0 = s0[0] + s0[4], a1 = s0[1] + s0[5], a2 = s0[2] + s0[6], a3 = s0[3] + s0[7];
            rr4[0] = (a0 + a2) + (a1 + a3);
            a0 = s1[0] + s1[4]; a1 = s1[1] + s1[5]; a2 = s1[2] + s1[6]; a3 = s1[3] + s1[7];
            rr4[1] = (a0 + a2) + (a1 + a3);
            a0 = s2[0] + s2[4]; a1 = s2[1] + s2[5]; a2 = s2[2] + s2[6]; a3 = s2[3] + s2[7];
            rr4[2] = (a0 + a2) + (a1 + a3);
            a0 = s3[0] + s3[4]; a1 = s3[1] + s3[5]; a2 = s3[2] + s3[6]; a3 = s3[3] + s3[7];
            rr4[3] = (a0 + a2) + (a1 + a3);
        }
#pragma unroll
        for (uint32_t e = 0; e < 4u; e++) {
            const uint32_t p = (e < 2u) ? mtokA : mtokB;
            const uint32_t row = row0 + n0 + (e & 1u);
            if (p >= np || row >= out_dim) continue;
            down_out[(uint64_t)s_pair[p] * out_dim + row] = rr4[e];
        }
    }
}



template <uint32_t ROW_SPAN>
__global__ static void moe_down_expert_tile16_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

#endif  // DS4X_BACKEND_MOE_DISPATCH_KERNELS_CUH
