#include "../internal/backend_internal.cuh"

/* Moe Dispatch implementation. */


int cuda_q4_mma_tile16_shmem_ok(int which_down) {
    /* Opt the tile16 kernels into >48KB dynamic shared memory, per device. */
    static int ready[DS4_MAX_GPUS][2];
    static int failed = 0;
    if (failed) return 0;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_MAX_GPUS) return 0;
    if (ready[dev][which_down]) return 1;
    cudaFuncAttributes fn_attr;
    cudaError_t err = which_down
        ? cudaFuncGetAttributes(&fn_attr, moe_down_q4K_tile16_mma_kernel<512>)
        : cudaFuncGetAttributes(&fn_attr, moe_gate_up_mid_q4K_tile16_mma_kernel<512>);
    if (err != cudaSuccess || fn_attr.binaryVersion < 80) {
        failed = 1;
        return 0;
    }
    const int bytes = (int)(16u * 16u * sizeof(cuda_block_q8_K));
    if (which_down) {
        err = cudaFuncSetAttribute(moe_down_q4K_tile16_mma_kernel<512>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (err == cudaSuccess)
            err = cudaFuncSetAttribute(moe_down_q4K_tile16_mma_kernel<1024>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (err == cudaSuccess)
            err = cudaFuncSetAttribute(moe_down_q4K_tile16_mma_kernel<2048>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    } else {
        err = cudaFuncSetAttribute(moe_gate_up_mid_q4K_tile16_mma_kernel<512>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (err == cudaSuccess)
            err = cudaFuncSetAttribute(moe_gate_up_mid_q4K_tile16_mma_kernel<1024>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (err == cudaSuccess)
            err = cudaFuncSetAttribute(moe_gate_up_mid_q4K_tile16_mma_kernel<2048>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    }
    if (err != cudaSuccess) {
        failed = 1;
        return 0;
    }
    ready[dev][which_down] = 1;
    return 1;
}




__global__ static void moe_down_sorted_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_CUDA_UNUSED void moe_down_expert_tile8_kernel(
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
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
    }
}

__global__ static void moe_down_expert_tile4_row32_kernel(
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
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][8];
    uint32_t pair[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
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
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block4(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, acc);
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

__global__ static void moe_down_expert_tile8_row32_kernel(
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
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
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
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
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

__global__ static void moe_down_expert_tile16_row32_kernel(
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
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
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
    if (row >= out_dim) return;
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

__global__ static void moe_down_expert_tile16_row2048_kernel(
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
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
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

__global__ static void moe_down_sorted_p2_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t pair_count) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= out_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_sum_kernel(float *out, const float *down, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += down[((uint64_t)tok * n_expert + e) * out_dim + row];
    out[gid] = acc;
}

__global__ static void moe_sum_owned_kernel(
        float *out,
        const float *down,
        const int32_t *selected,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t n_tokens) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    const uint32_t tok = (uint32_t)(gid / out_dim);
    const uint32_t row = (uint32_t)(gid - (uint64_t)tok * out_dim);
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        if (slot >= n_expert) break;
        const uint64_t pair = (uint64_t)tok * n_expert + slot;
        const float value = selected[pair] >= 0
            ? down[pair * out_dim + row] : 0.0f;
        acc = __fadd_rn(acc, value);
    }
    out[gid] = acc;
}

__device__ static float dev_iq2_xxs_dot_f32(const cuda_block_iq2_xxs *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_iq2_xxs *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const uint16_t *q2 = xb->qs;
        const float *xf = x + (uint64_t)b * CUDA_QK_K;
        for (uint32_t ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
            const uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            q2 += 4;
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            const uint8_t grids[4] = {
                (uint8_t)(aux_g & 0xffu),
                (uint8_t)((aux_g >> 8) & 0xffu),
                (uint8_t)((aux_g >> 16) & 0xffu),
                (uint8_t)((aux_g >> 24) & 0xffu),
            };
            for (uint32_t half = 0; half < 2; half++) {
                for (uint32_t g = 0; g < 2; g++) {
                    const uint32_t gi = half * 2 + g;
                    const uint64_t grid = cuda_iq2xxs_grid[grids[gi]];
                    const uint8_t signs = cuda_ksigns_iq2xs[(aux_s >> (14u * half + 7u * g)) & 127u];
                    for (uint32_t i = 0; i < 8; i++) {
                        float w = (float)((grid >> (8u * i)) & 0xffu);
                        if (signs & (1u << i)) w = -w;
                        acc += dl * w * xf[ib32 * 32u + half * 16u + g * 8u + i];
                    }
                }
            }
        }
    }
    return acc;
}

__device__ static float dev_q2_K_dot_f32(const cuda_block_q2_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_q2_K *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const float dmin = dev_f16_to_f32(xb->dmin);
        for (uint32_t il = 0; il < 16; il++) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = xb->scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4);
            const uint8_t *q = xb->qs + 32u * chunk + 16u * pair;
            const float *xf = x + (uint64_t)b * CUDA_QK_K + chunk * 128u + ((il % 8u) / 2u) * 32u + pair * 16u;
            for (uint32_t i = 0; i < 16; i++) {
                const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                acc += w * xf[i];
            }
        }
    }
    return acc;
}

__global__ static void moe_gate_up_mid_f32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / CUDA_QK_K;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dev_iq2_xxs_dot_f32(gr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
        up += dev_iq2_xxs_dot_f32(ur + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_down_f32_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t nb = expert_mid_dim / CUDA_QK_K;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const float *xr = mid + (uint64_t)pair * expert_mid_dim;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) acc += dev_q2_K_dot_f32(wr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}

static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t layer_index,
        uint32_t n_tokens,
        int allow_streaming,
        int owned_filtered) {
    (void)allow_streaming;
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        gate->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        up->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        mid->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const int q4k_path = (gate_type == 12u && down_type == 12u);
    const int iq2_path = (gate_type == 16u && down_type == 10u);
    const int mxfp4_path = (gate_type == 39u && down_type == 39u);
    if (!q4k_path && !iq2_path && !mxfp4_path) return 0;

    /* The aligned artifacts replace the raw expert tensors on integrated
     * CUDA systems.  Route both prefill and decode before resolving a raw
     * pointer, otherwise the fallback cache would duplicate tens of GiB. */
    if (iq2_path && !owned_filtered &&
        cuda_aligned_iq2_enabled() && cuda_aligned_q2k_enabled()) {
        const uint64_t gate_total = (uint64_t)n_total_expert * gate_expert_bytes;
        const uint64_t down_total = (uint64_t)n_total_expert * down_expert_bytes;
        const uint64_t gate_aligned_bytes = ds4_mmq_iq2_xxs_aligned_bytes(
            (int)expert_mid_dim, (int)expert_in_dim, (int)n_total_expert);
        const uint64_t down_aligned_bytes = ds4_mmq_q2_k_aligned_bytes(
            (int)out_dim, (int)expert_mid_dim, (int)n_total_expert);
        const char *gate_aligned = cuda_derived_weight_ptr(
            model_map, gate_offset, gate_total,
            CUDA_DERIVED_IQ2_XXS_ALIGNED_MOE,
            expert_in_dim, expert_mid_dim, n_total_expert,
            gate_aligned_bytes);
        const char *up_aligned = cuda_derived_weight_ptr(
            model_map, up_offset, gate_total,
            CUDA_DERIVED_IQ2_XXS_ALIGNED_MOE,
            expert_in_dim, expert_mid_dim, n_total_expert,
            gate_aligned_bytes);
        const char *down_aligned = cuda_derived_weight_ptr(
            model_map, down_offset, down_total,
            CUDA_DERIVED_Q2_K_ALIGNED_MOE,
            expert_mid_dim, out_dim, n_total_expert,
            down_aligned_bytes);
        if (gate_aligned && up_aligned && down_aligned) {
            const cudaStream_t aligned_stream =
                n_tokens == 1u ? cuda_decode_stream() : (cudaStream_t)0;
            int rc;
            if (n_tokens == 1u) {
                rc = ds4_mmq_iq2_xxs_aligned_moe_gate_up_mid_vec(
                    gate_aligned, up_aligned,
                    (const float *)x->ptr,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    (float *)mid->ptr,
                    (int)expert_mid_dim, (int)expert_in_dim,
                    (int)n_tokens, (int)n_total_expert, (int)n_expert,
                    clamp, aligned_stream);
                if (rc == 0) {
                    const uint32_t assignments = n_tokens * n_expert;
                    rc = ds4_mmq_q2_K_aligned_moe_vec(
                        down_aligned, (const float *)mid->ptr,
                        (const int32_t *)selected->ptr,
                        (float *)down->ptr,
                        (int)out_dim, (int)expert_mid_dim,
                        (int)assignments, (int)n_total_expert,
                        /*n_expert_used=*/1,
                        aligned_stream);
                }
            } else {
                rc = ds4_mmq_iq2_xxs_q2_K_moe_fused_soa(
                    gate_aligned, up_aligned, down_aligned,
                    (const float *)x->ptr,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    (float *)gate->ptr, (float *)up->ptr,
                    (float *)mid->ptr, (float *)down->ptr,
                    (int)expert_mid_dim, (int)expert_in_dim, (int)out_dim,
                    (int)n_tokens, (int)n_total_expert, (int)n_expert,
                    clamp, aligned_stream);
            }
            if (rc == 0) {
                const uint64_t n = (uint64_t)n_tokens * out_dim;
                moe_mmq_sum_kernel<<<
                    (uint32_t)((n + 255u) / 256u), 256, 0, aligned_stream>>>(
                    (float *)out->ptr, (const float *)down->ptr,
                    NULL, out_dim, n_expert, n_tokens,
                    /*guard_nonfinite=*/1);
                if (cuda_ok(cudaGetLastError(), "aligned moe sum launch")) {
                    static int logged = 0;
                    if (!logged) {
                        logged = 1;
                        fprintf(stderr,
                                "ds4: routed MoE using aligned CUDA artifacts\n");
                    }
                    return 1;
                }
                rc = -1;
            }
            fprintf(stderr,
                    "ds4: aligned routed-MoE returned %d "
                    "(layer=%u n_tokens=%u)\n",
                    rc, layer_index, n_tokens);
            if (cuda_model_map_replaces_complete(model_map)) return 0;
        }
    }

    /* Native MXFP4 routed experts use the vendored MMVQ decode kernels and
     * MMQ matrix kernels. On Blackwell the latter dispatch to FP4 MMA. */
    if (mxfp4_path) {
        if (!cuda_use_mxfp4_mmq()) {
            fprintf(stderr, "ds4: CUDA MXFP4 requires the MMQ backend\n");
            return 0;
        }
        const uint64_t gate_total =
            (uint64_t)n_total_expert * gate_expert_bytes;
        const uint64_t down_total =
            (uint64_t)n_total_expert * down_expert_bytes;
        if (gate_total > model_size - gate_offset ||
            gate_total > model_size - up_offset ||
            down_total > model_size - down_offset) {
            return 0;
        }

        const int logical_tier = ds4_tensor_device_idx(out);
        const uint64_t slot_count = (uint64_t)n_tokens * n_expert;
        const ds4_gpu_tensor *mx_selected = selected;
        const uint32_t weight_experts = n_total_expert;
        const char *gate_w = cuda_resolve_weight_ptr(
                model_map, gate_offset, gate_total, logical_tier,
                "mxfp4 moe gate");
        const char *up_w = cuda_resolve_weight_ptr(
                model_map, up_offset, gate_total, logical_tier,
                "mxfp4 moe up");
        const char *down_w = cuda_resolve_weight_ptr(
                model_map, down_offset, down_total, logical_tier,
                "mxfp4 moe down");
        if (!gate_w || !up_w || !down_w || weight_experts == 0u) return 0;

        const cudaStream_t stream =
            n_tokens == 1u ? cuda_decode_stream() : (cudaStream_t)0;
        int rc = -1;
        if (n_tokens == 1u && n_expert == 6u) {
            rc = ds4_mmq_mxfp4_moe_gate_up_mid_vec(
                gate_w, up_w, (const float *)x->ptr,
                (const int32_t *)mx_selected->ptr,
                (const float *)weights->ptr, (float *)mid->ptr,
                (int)expert_mid_dim, (int)expert_in_dim,
                (int)n_tokens, (int)weight_experts, (int)n_expert,
                clamp, stream);
            if (rc == 0) {
                rc = ds4_mmq_mxfp4_moe_down_sum6_vec(
                    down_w, (const float *)mid->ptr,
                    (const int32_t *)mx_selected->ptr, (float *)out->ptr,
                    (int)out_dim, (int)expert_mid_dim,
                    (int)n_tokens, (int)weight_experts, (int)n_expert,
                    stream);
            }
        } else {
            rc = ds4_mmq_mxfp4_moe_pair(
                gate_w, up_w, (const float *)x->ptr,
                (const int32_t *)mx_selected->ptr,
                (float *)gate->ptr, (float *)up->ptr,
                (int)expert_mid_dim, (int)expert_in_dim,
                (int)n_tokens, (int)weight_experts, (int)n_expert,
                stream);
            if (rc == 0) {
                const uint64_t mid_floats =
                    slot_count * expert_mid_dim;
                moe_mmq_swiglu_weighted_clamp_kernel<<<
                    (uint32_t)((mid_floats + 255u) / 256u), 256, 0, stream>>>(
                    (float *)mid->ptr,
                    (const float *)gate->ptr, (const float *)up->ptr,
                    (const float *)weights->ptr,
                    expert_mid_dim, n_tokens, n_expert, clamp);
                rc = cuda_ok(cudaGetLastError(),
                             "mxfp4 moe swiglu launch") ? 0 : -1;
            }
            if (rc == 0) {
                rc = ds4_mmq_mxfp4_moe(
                    down_w, (const float *)mid->ptr,
                    (const int32_t *)mx_selected->ptr,
                    (float *)down->ptr,
                    (int)out_dim, (int)expert_mid_dim,
                    (int)slot_count, (int)weight_experts,
                    /*n_expert_used=*/1, stream);
            }
            if (rc == 0) {
                const uint64_t n = (uint64_t)n_tokens * out_dim;
                moe_mmq_sum_kernel<<<
                    (uint32_t)((n + 255u) / 256u), 256, 0, stream>>>(
                    (float *)out->ptr, (const float *)down->ptr,
                    owned_filtered
                        ? (const int32_t *)mx_selected->ptr
                        : NULL,
                    out_dim, n_expert, n_tokens, /*guard_nonfinite=*/1);
                rc = cuda_ok(cudaGetLastError(),
                             "mxfp4 moe sum launch") ? 0 : -1;
            }
        }
        if (rc == 0) return 1;
        fprintf(stderr,
                "ds4: CUDA MXFP4 routed-MoE returned %d "
                "(layer=%u n_tokens=%u)\n",
                rc, layer_index, n_tokens);
        return 0;
    }
    /* mmq routed-MoE prefill tier (ported from the Entrpi/ds4 fork).
     * IQ2_XXS gate/up pair (one shared activation quantize + routing
     * pass) -> SwiGLU + clamp + router weight -> Q2_K down, treating
     * each (token, slot) assignment as its own single-expert row ->
     * guarded slot sum.  Buffers gate/up/mid/down are already sized to
     * [n_tokens, n_expert, *] by the validation above.  Any entry
     * failure falls through to the legacy sorted-pairs path (the
     * buffers are scratch there too). */
    if (iq2_path && n_tokens > 1u && !owned_filtered && cuda_use_mmq()) {
        const uint64_t gate_total = (uint64_t)n_total_expert * gate_expert_bytes;
        const uint64_t down_total = (uint64_t)n_total_expert * down_expert_bytes;
        const int mmq_tier = ds4_tensor_device_idx(out);
        const char *gate_w = cuda_resolve_weight_ptr(model_map, gate_offset, gate_total, mmq_tier, "moe gate mmq");
        const char *up_w = gate_w ? cuda_resolve_weight_ptr(model_map, up_offset, gate_total, mmq_tier, "moe up mmq") : NULL;
        const char *down_w = up_w ? cuda_resolve_weight_ptr(model_map, down_offset, down_total, mmq_tier, "moe down mmq") : NULL;
        if (down_w) {
            const uint64_t n_assignments = (uint64_t)n_tokens * n_expert;
            int rc = ds4_mmq_iq2_xxs_moe_pair(
                    gate_w, up_w, (const float *)x->ptr,
                    (const int32_t *)selected->ptr,
                    (float *)gate->ptr, (float *)up->ptr,
                    (int)expert_mid_dim, (int)expert_in_dim,
                    (int)n_tokens, (int)n_total_expert, (int)n_expert,
                    (cudaStream_t)0);
            if (rc == 0) {
                const uint64_t mid_floats = n_assignments * expert_mid_dim;
                moe_mmq_swiglu_weighted_clamp_kernel<<<(uint32_t)((mid_floats + 255) / 256), 256>>>(
                        (float *)mid->ptr,
                        (const float *)gate->ptr, (const float *)up->ptr,
                        (const float *)weights->ptr,
                        expert_mid_dim, n_tokens, n_expert, clamp);
                rc = cuda_ok(cudaGetLastError(), "mmq moe swiglu launch") ? 0 : -1;
            }
            if (rc == 0) {
                rc = ds4_mmq_q2_K_moe(
                        down_w, (const float *)mid->ptr,
                        (const int32_t *)selected->ptr,
                        (float *)down->ptr,
                        (int)out_dim, (int)expert_mid_dim,
                        (int)n_assignments, (int)n_total_expert,
                        /*n_expert_used=*/1,
                        (cudaStream_t)0);
            }
            if (rc == 0) {
                const uint64_t n = (uint64_t)n_tokens * out_dim;
                moe_mmq_sum_kernel<<<(uint32_t)((n + 255) / 256), 256>>>(
                        (float *)out->ptr, (const float *)down->ptr,
                        NULL, out_dim, n_expert, n_tokens,
                        /*guard_nonfinite=*/1);
                if (cuda_ok(cudaGetLastError(), "mmq moe sum launch")) return 1;
                rc = -1;
            }
            fprintf(stderr, "ds4: mmq routed-MoE tier rc=%d (layer=%u n_tokens=%u); falling back\n",
                    rc, layer_index, n_tokens);
        }
    }
    /* Q4_K routed-MoE dispatch:
     *   n_tokens == 1 and n_expert == 6:
     *                  use_direct_down_sum + moe_gate_up_mid_decode_q4K_qwarp32
     *                  + moe_down_q4K_sum6_qwarp32.
     *   n_tokens == 1 and n_expert == 3:
     *                  use the same direct path with moe_down_q4K_sum3_qwarp32.
     *                  Decode TP relies on this for splitting the six selected
     *                  experts into two groups.
     *   n_tokens == 1 and other n_expert:
     *                  use the same per-pair gate/up kernel plus the generic
     *                  q4K down + sum path.
     *   n_tokens >  1: default sorted-pairs expert-tile path groups token/expert
     *                  pairs by expert and uses Q4_K tile8 gate/up + down kernels
     *                  (`DS4_CUDA_MOE_NO_Q4_SORTED=1` restores the older
     *                  token-indexed decode-style prefill kernels). */
    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out);
    const char *gate_w = cuda_resolve_weight_ptr(
            model_map, gate_offset, gate_bytes, logical_tier, "moe_gate");
    const char *up_w = cuda_resolve_weight_ptr(
            model_map, up_offset, gate_bytes, logical_tier, "moe_up");
    const char *down_w = cuda_resolve_weight_ptr(
            model_map, down_offset, down_bytes, logical_tier, "moe_down");
    if (!gate_w || !up_w || !down_w) return 0;

    int ok = 1;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint64_t xq_count = (uint64_t)n_tokens * xq_blocks;
    const uint64_t midq_count = (uint64_t)n_tokens * n_expert * midq_blocks;
    const uint64_t xq_bytes = xq_count * sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = midq_count * sizeof(cuda_block_q8_K);
    if (down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;
        const uint32_t profile_moe = getenv("DS4_CUDA_MOE_PROFILE") != NULL;
        cudaEvent_t prof_ev[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
        if (profile_moe) {
            for (uint32_t i = 0; i < 7u; i++) {
                if (cudaEventCreate(&prof_ev[i]) != cudaSuccess) {
                    for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(prof_ev[j]);
                    memset(prof_ev, 0, sizeof(prof_ev));
                    break;
                }
            }
            if (prof_ev[0]) (void)cudaEventRecord(prof_ev[0], 0);
        }
        const uint32_t pair_count = n_tokens * n_expert;
        const uint32_t use_q4_sorted_pairs =
            q4k_path && n_tokens > 1u &&
            (owned_filtered ||
             (getenv("DS4_CUDA_MOE_NO_Q4_SORTED") == NULL &&
              getenv("DS4_CUDA_MOE_NO_EXPERT_TILES") == NULL &&
              getenv("DS4_CUDA_MOE_TILE4") == NULL));
        const uint32_t use_sorted_pairs =
            n_tokens > 1u &&
            (owned_filtered || !q4k_path || use_q4_sorted_pairs);
        const uint32_t use_expert_tiles =
            use_sorted_pairs &&
            (owned_filtered || getenv("DS4_CUDA_MOE_NO_EXPERT_TILES") == NULL);
        const uint32_t q4_owned_batch =
            owned_filtered && q4k_path && n_tokens >= 4u && n_tokens <= 16u;
        /* Small batches (DSpark stage chain / verify, n<=8) leave most of an
         * 8-slot expert tile empty (1-2 rows per expert): tile4 halves the
         * wasted dot-slots and measures ~2x faster there. Q4 TP batches use
         * tile8 because its exact tensor-core row-span kernels recover more
         * than the empty slots cost. Large prefill also keeps tile8. */
        const uint32_t expert_tile_m =
            getenv("DS4_CUDA_MOE_TILE4") ? 4u :
            (getenv("DS4_CUDA_MOE_TILE8") ? 8u :
             (q4_owned_batch || n_tokens > 8u ? 8u : 4u));
        const uint32_t write_gate_up = getenv("DS4_CUDA_MOE_WRITE_GATE_UP") != NULL;
        const uint32_t use_p2_sorted =
            use_sorted_pairs && !owned_filtered &&
            getenv("DS4_CUDA_MOE_NO_P2") == NULL;
        const uint32_t use_atomic_down = !q4k_path && use_expert_tiles &&
            (getenv("DS4_CUDA_MOE_ATOMIC_DOWN") != NULL ||
             (n_tokens >= 128u && getenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN") == NULL));
        const uint32_t use_owned_sparse_buffers = owned_filtered &&
            getenv("DS4_CUDA_MOE_NO_OWNED_SPARSE_BUFFERS") == NULL;
        const uint32_t use_gate_row2048 = use_expert_tiles && expert_tile_m == 8u &&
            (getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW128") != NULL ||
             ((q4_owned_batch || n_tokens >= 128u) &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW128") == NULL));
        const uint32_t use_q4_mma_tiles16 = q4k_path && use_expert_tiles &&
            expert_tile_m == 8u && cuda_q4_mma_ok() &&
            getenv("DS4_CUDA_MOE_NO_Q4_MMA_TILE16") == NULL;
        const uint32_t use_down_tile16 = !q4k_path && use_atomic_down && expert_tile_m == 8u &&
            n_tokens >= 128u && getenv("DS4_CUDA_MOE_NO_DOWN_TILE16") == NULL;
        const uint32_t use_small_sorted_prep =
            owned_filtered && q4k_path && n_tokens <= 16u && pair_count <= 96u &&
            n_total_expert <= 128u && use_sorted_pairs && use_expert_tiles &&
            getenv("DS4_CUDA_MOE_NO_SMALL_SORTED_PREP") == NULL;
        const uint32_t force_q4_down_rowspan =
            getenv("DS4_CUDA_MOE_DOWN_ROW512") != NULL ||
            getenv("DS4_CUDA_MOE_DOWN_ROW1024") != NULL ||
            getenv("DS4_CUDA_MOE_DOWN_ROW2048") != NULL;
        const uint32_t use_q4_down_rowspan =
            q4k_path && use_expert_tiles && expert_tile_m == 8u &&
            (q4_owned_batch || n_tokens >= 128u || force_q4_down_rowspan) &&
            getenv("DS4_CUDA_MOE_NO_Q4_DOWN_ROWSPAN") == NULL;
        const uint32_t use_decode_lut_gate =
            n_tokens == 1u && xq_blocks <= 16u &&
            getenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") == NULL;
        const uint32_t gate_row_span =
            getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ? 2048u :
            getenv("DS4_CUDA_MOE_GATE_ROW1024") != NULL ? 1024u : 512u;
        const uint32_t down_row_span =
            getenv("DS4_CUDA_MOE_DOWN_ROW512") != NULL ? 512u :
            getenv("DS4_CUDA_MOE_DOWN_ROW2048") != NULL ? 2048u :
            getenv("DS4_CUDA_MOE_DOWN_ROW1024") != NULL ? 1024u : 512u;
        const uint32_t use_down_row2048 = !q4k_path && use_atomic_down && expert_tile_m == 8u &&
            (getenv("DS4_CUDA_MOE_DOWN_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW128") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW64") != NULL ||
             (use_down_tile16 &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW128") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW64") == NULL));
        const uint32_t use_direct_down_sum =
            n_tokens == 1u && (n_expert == 6u || n_expert == 3u) &&
            getenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6") == NULL;
        const uint32_t use_direct_midq =
            q4k_path && use_direct_down_sum && !write_gate_up &&
            getenv("DS4_CUDA_MOE_DIRECT_MIDQ") != NULL &&
            getenv("DS4_CUDA_MOE_NO_DIRECT_MIDQ") == NULL;
        const uint32_t use_q4_gate_h16r8 =
            q4k_path && !use_direct_midq &&
            getenv("DS4_CUDA_MOE_Q4_GATE_H16R8") != NULL &&
            getenv("DS4_CUDA_MOE_NO_Q4_GATE_H16R8") == NULL;
        const uint32_t use_q4_gate_h16 =
            q4k_path && !use_direct_midq && !use_q4_gate_h16r8 &&
            getenv("DS4_CUDA_MOE_Q4_GATE_H16") != NULL &&
            getenv("DS4_CUDA_MOE_NO_Q4_GATE_H16") == NULL;
        const uint32_t use_q4_gate_w32r16 =
            q4k_path && !use_direct_midq && !use_q4_gate_h16r8 && !use_q4_gate_h16 &&
            getenv("DS4_CUDA_MOE_Q4_GATE_W32R16") != NULL &&
            getenv("DS4_CUDA_MOE_NO_Q4_GATE_W32R16") == NULL;
        const uint32_t use_q4_gate_w32 =
            q4k_path && !use_direct_midq && !use_q4_gate_h16r8 && !use_q4_gate_h16 &&
            !use_q4_gate_w32r16 &&
            getenv("DS4_CUDA_MOE_NO_Q4_GATE_W32") == NULL;
        const uint32_t use_q4_gate_w32_noaux =
            use_q4_gate_w32 && !write_gate_up &&
            getenv("DS4_CUDA_MOE_NO_Q4_GATE_W32_NOAUX") == NULL;
        const uint32_t use_q4_down_slot3 =
            q4k_path && use_direct_down_sum && n_expert == 3u &&
            getenv("DS4_CUDA_MOE_Q4_DOWN_SLOT3") != NULL &&
            getenv("DS4_CUDA_MOE_NO_Q4_DOWN_SLOT3") == NULL;
        const uint32_t use_q4_midq_sidecar =
            q4k_path && use_direct_down_sum && use_q4_gate_w32_noaux &&
            !use_direct_midq && !write_gate_up &&
            (expert_mid_dim % CUDA_QK_K) == 0u &&
            getenv("DS4_CUDA_MOE_MIDQ_SIDECAR") != NULL &&
            getenv("DS4_CUDA_MOE_NO_MIDQ_SIDECAR") == NULL;
        float *midq_sidecar = use_q4_midq_sidecar ? (float *)up->ptr : NULL;
        if (g_cuda_moe_decode_graph &&
            !owned_filtered &&
            !profile_moe &&
            q4k_path &&
            n_tokens == 1u &&
            use_direct_down_sum &&
            use_q4_gate_w32 &&
            !use_q4_gate_w32r16 &&
            !use_q4_down_slot3 &&
            !use_direct_midq &&
            !use_q4_midq_sidecar &&
            (n_expert == 3u || n_expert == 6u)) {
            int grc = routed_moe_decode_q4_graph_launch(
                    logical_tier,
                    (float *)out->ptr,
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    down_w,
                    xq,
                    midq,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    down_expert_bytes,
                    down_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    out_dim,
                    n_expert,
                    write_gate_up,
                    clamp,
                    (const float *)x->ptr);
            if (grc == 1) return 1;
            if (grc < 0) return 0;
        }
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        dim3 xq_grid(xq_blocks, n_tokens, 1);
        q8_K_quantize_kernel<<<xq_grid, 256, 0, cuda_decode_stream()>>>(xq, (const float *)x->ptr, expert_in_dim, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe x quantize launch");
        if (prof_ev[1]) (void)cudaEventRecord(prof_ev[1], 0);
        if (ok && use_sorted_pairs) {
            const uint64_t counts_bytes = (uint64_t)n_total_expert * sizeof(uint32_t);
            const uint64_t offsets_bytes = ((uint64_t)n_total_expert + 1ull) * sizeof(uint32_t);
            const uint64_t cursors_bytes = (uint64_t)n_total_expert * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = (pair_count + expert_tile_m - 1u) / expert_tile_m + n_total_expert;
            tile16_capacity = (use_down_tile16 || use_q4_mma_tiles16) ? ((pair_count + 15u) / 16u + n_total_expert) : 0u;
            const uint64_t tile_offsets_bytes = ((uint64_t)n_total_expert + 1ull) * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = (use_down_tile16 || use_q4_mma_tiles16) ? (((uint64_t)n_total_expert + 1ull) * sizeof(uint32_t)) : 0u;
            const uint64_t tile16_total_bytes = (use_down_tile16 || use_q4_mma_tiles16) ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t scratch_bytes = tile16_starts_off + tile16_starts_bytes;
            uint8_t *scratch = (uint8_t *)cuda_tmp_alloc_on(logical_tier, scratch_bytes,
                                                             "routed_moe sorted pairs");
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = (use_down_tile16 || use_q4_mma_tiles16) ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = (use_down_tile16 || use_q4_mma_tiles16) ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = (use_down_tile16 || use_q4_mma_tiles16) ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = (use_down_tile16 || use_q4_mma_tiles16) ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                if (use_small_sorted_prep) {
                    moe_prepare_sorted_tiles_small_kernel<<<1, 128, 0, cuda_decode_stream()>>>(
                        counts, offsets, cursors, sorted_pairs,
                        tile_offsets, tile_total, tile_experts, tile_starts,
                        tile16_offsets, tile16_total, tile16_experts, tile16_starts,
                        (const int32_t *)selected->ptr, pair_count, n_total_expert,
                        expert_tile_m, use_down_tile16 || use_q4_mma_tiles16);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe small sorted setup launch");
                } else {
                    ok = cuda_ok(cudaMemset(counts, 0, counts_bytes),
                                 "routed_moe sorted counts clear");
                }
                if (ok && !use_small_sorted_prep) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256, 0, cuda_decode_stream()>>>(
                        counts,
                        (const int32_t *)selected->ptr,
                        pair_count,
                        n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted count launch");
                }
                if (ok && !use_small_sorted_prep) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1, 0, cuda_decode_stream()>>>(offsets, cursors, counts, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok && !use_small_sorted_prep) {
                    moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256, 0, cuda_decode_stream()>>>(
                        sorted_pairs,
                        cursors,
                        (const int32_t *)selected->ptr,
                        pair_count,
                        n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok && use_expert_tiles && !use_small_sorted_prep) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1, 0, cuda_decode_stream()>>>(tile_offsets, tile_total, counts, expert_tile_m, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_expert_tiles && !use_small_sorted_prep) {
                    moe_build_expert_tiles_kernel<<<(n_total_expert + 255u) / 256u, 256, 0, cuda_decode_stream()>>>(
                        tile_experts, tile_starts, tile_offsets, counts, expert_tile_m, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && !use_small_sorted_prep &&
                    (use_down_tile16 || use_q4_mma_tiles16)) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1, 0, cuda_decode_stream()>>>(tile16_offsets, tile16_total, counts, 16u, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && !use_small_sorted_prep &&
                    (use_down_tile16 || use_q4_mma_tiles16)) {
                    moe_build_expert_tiles_kernel<<<(n_total_expert + 255u) / 256u, 256, 0, cuda_decode_stream()>>>(
                        tile16_experts, tile16_starts, tile16_offsets, counts, 16u, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        if (prof_ev[2]) (void)cudaEventRecord(prof_ev[2], 0);
        if (ok && owned_filtered && use_sorted_pairs &&
            !use_owned_sparse_buffers) {
            const uint64_t mid_bytes =
                (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float);
            ok = cuda_ok(cudaMemset(mid->ptr, 0, (size_t)mid_bytes),
                         "owned routed_moe mid clear");
        }
        if (ok) {
            dim3 mgrid((expert_mid_dim + 31u) / 32u, n_tokens * n_expert, 1);
            if (ok && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (q4k_path) {
                    const int use_q4_mma = cuda_q4_mma_ok() &&
                        ((((uintptr_t)gate_w | (uintptr_t)up_w |
                           gate_row_bytes | gate_expert_bytes) & 15u) == 0u) &&
                        xq_blocks <= 16u && (expert_mid_dim & 7u) == 0u;
                    const int use_q4_mma_t16 = use_q4_mma && use_q4_mma_tiles16 &&
                        tile16_total && tile16_experts && tile16_starts &&
                        xq_blocks == 16u && cuda_q4_mma_tile16_shmem_ok(0);
                    if (use_q4_mma_t16 && use_gate_row2048) {
                        const unsigned t16cap = (unsigned)((pair_count + 15u) / 16u + n_total_expert);
                        const size_t t16sh = 16u * 16u * sizeof(cuda_block_q8_K);
                        if (gate_row_span == 512u) {
                            dim3 tgrid((expert_mid_dim + 511u) / 512u, t16cap, 1);
                            moe_gate_up_mid_q4K_tile16_mma_kernel<512><<<tgrid, 256, t16sh, cuda_decode_stream()>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile16_total, tile16_experts, tile16_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else if (gate_row_span == 1024u) {
                            dim3 tgrid((expert_mid_dim + 1023u) / 1024u, t16cap, 1);
                            moe_gate_up_mid_q4K_tile16_mma_kernel<1024><<<tgrid, 256, t16sh, cuda_decode_stream()>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile16_total, tile16_experts, tile16_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else {
                            dim3 tgrid((expert_mid_dim + 2047u) / 2048u, t16cap, 1);
                            moe_gate_up_mid_q4K_tile16_mma_kernel<2048><<<tgrid, 256, t16sh, cuda_decode_stream()>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile16_total, tile16_experts, tile16_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        }
                    } else if (use_q4_mma && use_gate_row2048) {
                        if (gate_row_span == 512u) {
                            dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_tile8_mma_kernel<512><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else if (gate_row_span == 1024u) {
                            dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_tile8_mma_kernel<1024><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else {
                            dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_tile8_mma_kernel<2048><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        }
                    } else if (use_gate_row2048) {
                        if (gate_row_span == 512u) {
                            dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<512><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else if (gate_row_span == 1024u) {
                            dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<1024><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        } else {
                            dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                            moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<2048><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                                gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                                gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                                write_gate_up, clamp);
                        }
                    } else {
                        dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                        moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel<32><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256, 0, cuda_decode_stream()>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile8_row32_kernel<<<tgrid, 256, 0, cuda_decode_stream()>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile4_row32_kernel<<<tgrid, 256, 0, cuda_decode_stream()>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                }
            } else if (ok && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256, 0, cuda_decode_stream()>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && sorted_pairs) {
                moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256, 0, cuda_decode_stream()>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    clamp);
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + MOE_DECODE_ROWS_PER_BLOCK - 1u) / MOE_DECODE_ROWS_PER_BLOCK, n_tokens * n_expert, 1);
                if (q4k_path) {
                    /* Q4_K gate/up: the decode kernel is token-indexed via
                     * pair = blockIdx.y; tok = pair / n_expert, so the same
                     * launch covers both n_tokens == 1 (decode) and n_tokens > 1
                     * (prefill). q4k_path is steered here by use_sorted_pairs = 0
                     * cascading the IQ2 sorted/expert-tile branches off. */
                    if (use_direct_midq) {
                        dim3 mqgrid(midq_blocks, n_tokens * n_expert, 1);
                        moe_gate_up_midq_decode_q4K_qwarp32_kernel<<<mqgrid, 256, 0, cuda_decode_stream()>>>(
                            (float *)mid->ptr,
                            midq,
                            gate_w,
                            up_w,
                            xq,
                            (const int32_t *)selected->ptr,
                            (const float *)weights->ptr,
                            gate_expert_bytes,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            clamp);
                    } else if (use_q4_gate_h16r8) {
                        dim3 h8grid((expert_mid_dim + 7u) / 8u, n_tokens * n_expert, 1);
                        moe_gate_up_mid_decode_q4K_hwarp16_row8_kernel<<<h8grid, 256, 0, cuda_decode_stream()>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            gate_w,
                            up_w,
                            xq,
                            (const int32_t *)selected->ptr,
                            (const float *)weights->ptr,
                            gate_expert_bytes,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            write_gate_up,
                            clamp);
                    } else if (use_q4_gate_w32r16) {
                        dim3 w16grid((expert_mid_dim + 15u) / 16u, n_tokens * n_expert, 1);
                        moe_gate_up_mid_decode_q4K_warp32_row16_kernel<<<w16grid, 512, 0, cuda_decode_stream()>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            gate_w,
                            up_w,
                            xq,
                            (const int32_t *)selected->ptr,
                            (const float *)weights->ptr,
                            gate_expert_bytes,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            write_gate_up,
                            clamp);
                    } else if (use_q4_gate_w32) {
                        dim3 wgrid((expert_mid_dim + 7u) / 8u, n_tokens * n_expert, 1);
                        if (use_q4_gate_w32_noaux) {
                            if (use_q4_midq_sidecar) {
                                moe_gate_up_mid_decode_q4K_warp32_noaux_sidecar_kernel<<<wgrid, 256, 0, cuda_decode_stream()>>>(
                                    (float *)mid->ptr,
                                    midq_sidecar,
                                    gate_w,
                                    up_w,
                                    xq,
                                    (const int32_t *)selected->ptr,
                                    (const float *)weights->ptr,
                                    gate_expert_bytes,
                                    gate_row_bytes,
                                    xq_blocks,
                                    expert_mid_dim,
                                    n_expert,
                                    clamp);
                            } else {
                                moe_gate_up_mid_decode_q4K_warp32_noaux_kernel<<<wgrid, 256, 0, cuda_decode_stream()>>>(
                                    (float *)mid->ptr,
                                    gate_w,
                                    up_w,
                                    xq,
                                    (const int32_t *)selected->ptr,
                                    (const float *)weights->ptr,
                                    gate_expert_bytes,
                                    gate_row_bytes,
                                    xq_blocks,
                                    expert_mid_dim,
                                    n_expert,
                                    clamp);
                            }
                        } else {
                            moe_gate_up_mid_decode_q4K_warp32_kernel<<<wgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)gate->ptr,
                                (float *)up->ptr,
                                (float *)mid->ptr,
                                gate_w,
                                up_w,
                                xq,
                                (const int32_t *)selected->ptr,
                                (const float *)weights->ptr,
                                gate_expert_bytes,
                                gate_row_bytes,
                                xq_blocks,
                                expert_mid_dim,
                                n_expert,
                                write_gate_up,
                                clamp);
                        }
                    } else if (use_q4_gate_h16) {
                        dim3 hgrid((expert_mid_dim + 15u) / 16u, n_tokens * n_expert, 1);
                        moe_gate_up_mid_decode_q4K_hwarp16_kernel<<<hgrid, 256, 0, cuda_decode_stream()>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            gate_w,
                            up_w,
                            xq,
                            (const int32_t *)selected->ptr,
                            (const float *)weights->ptr,
                            gate_expert_bytes,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            write_gate_up,
                            clamp);
                    } else {
                        moe_gate_up_mid_decode_q4K_qwarp32_kernel<<<qgrid, 256, 0, cuda_decode_stream()>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            gate_w,
                            up_w,
                            xq,
                            (const int32_t *)selected->ptr,
                            (const float *)weights->ptr,
                            gate_expert_bytes,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            write_gate_up,
                            clamp);
                    }
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256, 0, cuda_decode_stream()>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256, 0, cuda_decode_stream()>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
        }
        if (prof_ev[3]) (void)cudaEventRecord(prof_ev[3], 0);
        if (ok && !use_direct_midq) {
            dim3 midq_grid(midq_blocks, n_tokens * n_expert, 1);
            if (use_q4_midq_sidecar) {
                q8_K_quantize_sidecar_kernel<<<midq_grid, 256, 0, cuda_decode_stream()>>>(
                        midq,
                        (const float *)mid->ptr,
                        midq_sidecar,
                        expert_mid_dim,
                        n_tokens * n_expert);
                ok = cuda_ok(cudaGetLastError(), "routed_moe mid sidecar quantize launch");
            } else if (use_owned_sparse_buffers) {
                q8_K_quantize_owned_kernel<<<midq_grid, 256, 0, cuda_decode_stream()>>>(
                        midq,
                        (const float *)mid->ptr,
                        (const int32_t *)selected->ptr,
                        expert_mid_dim,
                        n_tokens * n_expert,
                        0u,
                        n_total_expert);
                ok = cuda_ok(cudaGetLastError(),
                             "owned routed_moe active mid quantize launch");
            } else {
                q8_K_quantize_kernel<<<midq_grid, 256, 0, cuda_decode_stream()>>>(midq, (const float *)mid->ptr, expert_mid_dim, n_tokens * n_expert);
                ok = cuda_ok(cudaGetLastError(), "routed_moe mid quantize launch");
            }
        }
        if (prof_ev[4]) (void)cudaEventRecord(prof_ev[4], 0);
        if (ok && owned_filtered && use_sorted_pairs && !use_atomic_down &&
            !use_owned_sparse_buffers) {
            const uint64_t down_clear_bytes =
                (uint64_t)n_tokens * n_expert * out_dim * sizeof(float);
            ok = cuda_ok(cudaMemset(down->ptr, 0, (size_t)down_clear_bytes),
                         "owned routed_moe down clear");
        }
        if (ok) {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens * n_expert, 1);
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            if (use_direct_down_sum) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (q4k_path) {
                    if (n_expert == 6u) {
                        moe_down_q4K_sum6_qwarp32_kernel<<<sgrid, 256, 0, cuda_decode_stream()>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim);
                    } else {
                        if (use_q4_down_slot3) {
                            dim3 swgrid((out_dim + 7u) / 8u, 1, 1);
                            moe_down_q4K_sum3_slotwarp_kernel<<<swgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)out->ptr,
                                down_w,
                                midq,
                                (const int32_t *)selected->ptr,
                                down_expert_bytes,
                                down_row_bytes,
                                midq_blocks,
                                out_dim);
                        } else {
                            moe_down_q4K_sum3_qwarp32_kernel<<<sgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)out->ptr,
                                down_w,
                                midq,
                                (const int32_t *)selected->ptr,
                                down_expert_bytes,
                                down_row_bytes,
                                midq_blocks,
                                out_dim);
                        }
                    }
                } else {
                    if (n_expert == 6u) {
                        moe_down_sum6_qwarp32_kernel<<<sgrid, 256, 0, cuda_decode_stream()>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim);
                    } else {
                        moe_down_sum3_qwarp32_kernel<<<sgrid, 256, 0, cuda_decode_stream()>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim);
                    }
                }
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256, 0, cuda_decode_stream()>>>((float *)out->ptr, n);
                ok = cuda_ok(cudaGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_direct_down_sum) {
                /* The direct decode kernel writes the final token row. */
            } else if (sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (q4k_path) {
                    const int use_q4_down_mma = cuda_q4_mma_ok() &&
                        ((((uintptr_t)down_w | down_row_bytes | down_expert_bytes) & 15u) == 0u) &&
                        midq_blocks <= 8u && (out_dim & 7u) == 0u;
                    const int use_q4_down_t16 = use_q4_down_mma && use_q4_mma_tiles16 &&
                        tile16_total && tile16_experts && tile16_starts &&
                        midq_blocks <= 16u && cuda_q4_mma_tile16_shmem_ok(1);
                    if (use_q4_down_t16 && use_q4_down_rowspan) {
                        const unsigned t16cap = (unsigned)((pair_count + 15u) / 16u + n_total_expert);
                        const size_t dt16sh = 16u * (size_t)midq_blocks * sizeof(cuda_block_q8_K);
                        if (down_row_span == 512u) {
                            dim3 tgrid((out_dim + 511u) / 512u, t16cap, 1);
                            moe_down_q4K_tile16_mma_kernel<512><<<tgrid, 256, dt16sh, cuda_decode_stream()>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile16_total, tile16_experts, tile16_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        } else if (down_row_span == 1024u) {
                            dim3 tgrid((out_dim + 1023u) / 1024u, t16cap, 1);
                            moe_down_q4K_tile16_mma_kernel<1024><<<tgrid, 256, dt16sh, cuda_decode_stream()>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile16_total, tile16_experts, tile16_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        } else {
                            dim3 tgrid((out_dim + 2047u) / 2048u, t16cap, 1);
                            moe_down_q4K_tile16_mma_kernel<2048><<<tgrid, 256, dt16sh, cuda_decode_stream()>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                tile16_total, tile16_experts, tile16_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        }
                    } else if (use_q4_down_mma && use_q4_down_rowspan) {
                        if (down_row_span == 512u) {
                            dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                            moe_down_q4K_tile8_mma_kernel<512><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        } else if (down_row_span == 1024u) {
                            dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                            moe_down_q4K_tile8_mma_kernel<1024><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        } else {
                            dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                            moe_down_q4K_tile8_mma_kernel<2048><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        }
                    } else if (use_q4_down_rowspan) {
                        if (down_row_span == 512u) {
                            dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                            moe_down_q4K_expert_tile8_rowspan_kernel<512><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        } else if (down_row_span == 1024u) {
                            dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                            moe_down_q4K_expert_tile8_rowspan_kernel<1024><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        } else {
                            dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                            moe_down_q4K_expert_tile8_rowspan_kernel<2048><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                                (float *)down->ptr,
                                down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                                down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                                midq_blocks, out_dim, n_expert);
                        }
                    } else {
                        dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                        moe_down_q4K_expert_tile8_rowspan_kernel<32><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                            (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert);
                    }
                } else if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256, 0, cuda_decode_stream()>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256, 0, cuda_decode_stream()>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256, 0, cuda_decode_stream()>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256, 0, cuda_decode_stream()>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256, 0, cuda_decode_stream()>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256, 0, cuda_decode_stream()>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (sorted_pairs) {
                moe_down_sorted_qwarp32_kernel<<<dgrid, 256, 0, cuda_decode_stream()>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            } else if (q4k_path) {
                /* Q4_K prefill down. New kernel mirrors moe_down_qwarp32_kernel
                 * grid/geometry, swapping the weight block type to cuda_block_q4_K
                 * and the dot helper to dev_dot_q4_K_q8_K_block. Writes per-pair
                 * outputs into down->ptr; moe_sum_kernel below sums them across
                 * experts into out->ptr. */
                moe_down_q4K_qwarp32_kernel<<<dgrid, 256, 0, cuda_decode_stream()>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            } else {
                moe_down_qwarp32_kernel<<<dgrid, 256, 0, cuda_decode_stream()>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
        }
        if (prof_ev[5]) (void)cudaEventRecord(prof_ev[5], 0);
        if (ok && !use_atomic_down && !use_direct_down_sum) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            if (use_owned_sparse_buffers) {
                moe_sum_owned_kernel<<<(n + 255) / 256, 256, 0, cuda_decode_stream()>>>(
                        (float *)out->ptr,
                        (const float *)down->ptr,
                        (const int32_t *)selected->ptr,
                        out_dim,
                        n_expert,
                        n_tokens);
            } else {
                moe_sum_kernel<<<(n + 255) / 256, 256, 0, cuda_decode_stream()>>>(
                        (float *)out->ptr,
                        (const float *)down->ptr,
                        out_dim,
                        n_expert,
                        n_tokens);
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        if (prof_ev[6]) {
            (void)cudaEventRecord(prof_ev[6], 0);
            if (cudaEventSynchronize(prof_ev[6]) == cudaSuccess) {
                float ms_xq = 0.0f, ms_sort = 0.0f, ms_gate = 0.0f, ms_midq = 0.0f, ms_down = 0.0f, ms_sum = 0.0f, ms_total = 0.0f;
                (void)cudaEventElapsedTime(&ms_xq, prof_ev[0], prof_ev[1]);
                (void)cudaEventElapsedTime(&ms_sort, prof_ev[1], prof_ev[2]);
                (void)cudaEventElapsedTime(&ms_gate, prof_ev[2], prof_ev[3]);
                (void)cudaEventElapsedTime(&ms_midq, prof_ev[3], prof_ev[4]);
                (void)cudaEventElapsedTime(&ms_down, prof_ev[4], prof_ev[5]);
                (void)cudaEventElapsedTime(&ms_sum, prof_ev[5], prof_ev[6]);
                (void)cudaEventElapsedTime(&ms_total, prof_ev[0], prof_ev[6]);
                fprintf(stderr,
                        "ds4: CUDA MoE profile tokens=%u pairs=%u xq=%.3f sort=%.3f gateup=%.3f midq=%.3f down=%.3f sum=%.3f total=%.3f ms\n",
                        n_tokens, pair_count, ms_xq, ms_sort, ms_gate, ms_midq, ms_down, ms_sum, ms_total);
            }
            for (uint32_t i = 0; i < 7u; i++) (void)cudaEventDestroy(prof_ev[i]);
        }
        return ok;
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, n_tokens * n_expert, 1);
        moe_gate_up_mid_f32_kernel<<<mgrid, 256, 0, cuda_decode_stream()>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            (float *)mid->ptr,
            gate_w,
            up_w,
            (const float *)x->ptr,
            (const int32_t *)selected->ptr,
            (const float *)weights->ptr,
            gate_expert_bytes,
            gate_row_bytes,
            expert_in_dim,
            expert_mid_dim,
            n_expert,
            clamp);
        ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, n_tokens * n_expert, 1);
        moe_down_f32_kernel<<<dgrid, 256, 0, cuda_decode_stream()>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            (const int32_t *)selected->ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
    }
    if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
    }
    return ok;
}

extern "C" int ds4_gpu_routed_moe_one_owned_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t resident_expert_base,
        uint32_t resident_expert_count,
        float clamp,
        const ds4_gpu_tensor *x,
        ds4_gpu_tensor *down_output,
        bool pack_fixed3,
        ds4_gpu_tensor *shared_prequant) {
    if (!out || !gate || !up || !mid || !down || !model_map ||
        !selected || !weights || !x || n_expert != 6u ||
        n_total_expert == 0u || resident_expert_count == 0u ||
        gate_expert_bytes == 0u || gate_row_bytes == 0u ||
        down_expert_bytes == 0u || down_row_bytes == 0u ||
        resident_expert_base >= n_total_expert ||
        resident_expert_count > n_total_expert - resident_expert_base ||
        expert_in_dim % CUDA_QK_K != 0u ||
        expert_mid_dim % CUDA_QK_K != 0u ||
        selected->bytes < 6u * sizeof(int32_t) ||
        weights->bytes < 6u * sizeof(float) ||
        x->bytes < (uint64_t)expert_in_dim * sizeof(float) ||
        mid->bytes < 6ull * expert_mid_dim * sizeof(float) ||
        out->bytes < (uint64_t)out_dim * sizeof(float)) {
        return 0;
    }
    if (pack_fixed3 && resident_expert_base == 0u) return 0;
    const bool q4k_path = gate_type == 12u && down_type == 12u;
    const bool mxfp4_path = gate_type == 39u && down_type == 39u;
    if (!q4k_path && !mxfp4_path &&
        (gate_type != 16u || down_type != 10u)) {
        return 0;
    }
    if (q4k_path && getenv("DS4_CUDA_MOE_WRITE_GATE_UP") != NULL) {
        fprintf(stderr, "ds4: CUDA owned Q4 decode does not support gate/up auxiliary output\n");
        return 0;
    }
    if (!q4k_path && !mxfp4_path &&
        getenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") != NULL) {
        fprintf(stderr, "ds4: CUDA owned IQ2 decode requires the LUT gate path\n");
        return 0;
    }
    const bool write_aux =
        !q4k_path && !mxfp4_path &&
        getenv("DS4_CUDA_MOE_WRITE_GATE_UP") != NULL;

    if (resident_expert_base > UINT64_MAX / gate_expert_bytes ||
        resident_expert_count > UINT64_MAX / gate_expert_bytes ||
        resident_expert_base > UINT64_MAX / down_expert_bytes ||
        resident_expert_count > UINT64_MAX / down_expert_bytes) {
        return 0;
    }
    const uint64_t gate_shift = (uint64_t)resident_expert_base * gate_expert_bytes;
    const uint64_t down_shift = (uint64_t)resident_expert_base * down_expert_bytes;
    const uint64_t gate_bytes = (uint64_t)resident_expert_count * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)resident_expert_count * down_expert_bytes;
    if (gate_offset > model_size || gate_shift > model_size - gate_offset ||
        gate_bytes > model_size - gate_offset - gate_shift ||
        up_offset > model_size || gate_shift > model_size - up_offset ||
        gate_bytes > model_size - up_offset - gate_shift ||
        down_offset > model_size || down_shift > model_size - down_offset ||
        down_bytes > model_size - down_offset - down_shift) {
        return 0;
    }

    const int logical_tier = ds4_tensor_device_idx(out);
    const char *gate_w = (const char *)cuda_resolve_weight_ptr(
            model_map, gate_offset + gate_shift, gate_bytes,
            logical_tier, "moe_owned_gate");
    const char *up_w = (const char *)cuda_resolve_weight_ptr(
            model_map, up_offset + gate_shift, gate_bytes,
            logical_tier, "moe_owned_up");
    const char *down_w = (const char *)cuda_resolve_weight_ptr(
            model_map, down_offset + down_shift, down_bytes,
            logical_tier, "moe_owned_down");
    if (!gate_w || !up_w || !down_w) return 0;

    const uint64_t down_output_bytes =
        (uint64_t)(pack_fixed3 ? 4u : 6u) * out_dim * sizeof(float);
    if (mxfp4_path) {
        const uint64_t slot_bytes = 6ull * out_dim * sizeof(float);
        if (!cuda_use_mxfp4_mmq() ||
            gate->bytes < 6u * sizeof(int32_t) ||
            up->bytes < 6u * sizeof(float) ||
            down->bytes < slot_bytes ||
            (down_output && down_output->bytes < down_output_bytes)) {
            return 0;
        }

        cudaStream_t stream = cuda_decode_stream();
        int32_t *local_ids = (int32_t *)gate->ptr;
        float *local_weights = (float *)up->ptr;
        mxfp4_prepare_owned_assignments_kernel<<<1, 32, 0, stream>>>(
                local_ids,
                local_weights,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                resident_expert_base,
                resident_expert_count);
        if (!cuda_ok(cudaGetLastError(),
                     "owned MXFP4 assignment prepare launch")) {
            return 0;
        }
        if (ds4_mmq_mxfp4_moe_gate_up_mid_vec(
                    gate_w,
                    up_w,
                    (const float *)x->ptr,
                    local_ids,
                    local_weights,
                    (float *)mid->ptr,
                    expert_mid_dim,
                    expert_in_dim,
                    1,
                    resident_expert_count,
                    6,
                    clamp,
                    stream) != 0) {
            return 0;
        }

        if (ds4_mmq_mxfp4_moe_vec(
                    down_w,
                    (const float *)mid->ptr,
                    local_ids,
                    (float *)down->ptr,
                    out_dim,
                    expert_mid_dim,
                    6,
                    resident_expert_count,
                    1,
                    stream) != 0) {
            return 0;
        }
        if (pack_fixed3) {
            float *packed_dst = down_output
                ? (float *)down_output->ptr
                : (float *)down->ptr;
            moe_down_owned_pack_f32_slots_kernel<<<
                    (out_dim + 255u) / 256u, 256, 0, stream>>>(
                    packed_dst,
                    (const float *)down->ptr,
                    (const int32_t *)selected->ptr,
                    out_dim,
                    resident_expert_base,
                    resident_expert_count);
            if (!cuda_ok(cudaGetLastError(),
                         "owned MXFP4 down pack launch")) {
                return 0;
            }
        } else if (down_output) {
            const uint64_t slot_count = 6ull * out_dim;
            moe_down_owned_copy_f32_slots_kernel<<<
                    (slot_count + 255u) / 256u, 256, 0, stream>>>(
                    (float *)down_output->ptr,
                    (const float *)down->ptr,
                    slot_count);
            if (!cuda_ok(cudaGetLastError(),
                         "owned MXFP4 down peer copy launch")) {
                return 0;
            }
        }
        return 1;
    }

    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint64_t xq_bytes = (uint64_t)xq_blocks * sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = 6ull * midq_blocks * sizeof(cuda_block_q8_K);
    const uint64_t aux_bytes = 6ull * expert_mid_dim * sizeof(float);
    const uint64_t shared_q8_blocks = expert_in_dim / 32u;
    const uint64_t shared_q8_bytes = shared_q8_blocks * 32u;
    const uint64_t shared_scale_offset =
        (shared_q8_bytes + 15u) & ~15ull;
    const uint64_t shared_prequant_bytes =
        shared_scale_offset + shared_q8_blocks * sizeof(float);
    if (down->bytes < xq_bytes || down->bytes < down_output_bytes ||
        (down_output && down_output->bytes < down_output_bytes) ||
        gate->bytes < midq_bytes ||
        (shared_prequant &&
         (shared_prequant->bytes < shared_prequant_bytes ||
          ds4_tensor_device_idx(shared_prequant) != logical_tier)) ||
        (write_aux && (gate->bytes < aux_bytes || up->bytes < aux_bytes))) {
        return 0;
    }
    float *down_dst = (float *)(down_output ? down_output->ptr : down->ptr);
    cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
    cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;

    dim3 xq_grid(xq_blocks, 1, 1);
    if (shared_prequant) {
        int8_t *shared_xq = (int8_t *)shared_prequant->ptr;
        float *shared_scale = (float *)((char *)shared_prequant->ptr +
                                        shared_scale_offset);
        q8_K_q8_0_quantize_kernel<<<xq_grid, 256>>>(
                xq, shared_xq, shared_scale, (const float *)x->ptr,
                expert_in_dim, 1u);
    } else {
        q8_K_quantize_kernel<<<xq_grid, 256>>>(
                xq, (const float *)x->ptr, expert_in_dim, 1u);
    }
    if (!cuda_ok(cudaGetLastError(), "owned routed_moe x quantize launch")) return 0;

    if (q4k_path) {
        dim3 gate_grid((expert_mid_dim + 7u) / 8u, 6u, 1u);
        moe_gate_up_mid_decode_q4K_owned_warp32_noaux_kernel<<<gate_grid, 256>>>(
                (float *)mid->ptr,
                gate_w,
                up_w,
                xq,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                xq_blocks,
                expert_mid_dim,
                6u,
                resident_expert_base,
                resident_expert_count,
                clamp);
    } else {
        dim3 gate_grid((expert_mid_dim + 31u) / 32u, 6u, 1u);
        moe_gate_up_mid_decode_lut_owned_qwarp32_kernel<<<gate_grid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                xq,
                (const int32_t *)selected->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                xq_blocks,
                expert_mid_dim,
                6u,
                resident_expert_base,
                resident_expert_count,
                write_aux,
                clamp);
    }
    if (!cuda_ok(cudaGetLastError(), "owned routed_moe gate/up launch")) return 0;

    dim3 midq_grid(midq_blocks, 6u, 1u);
    q8_K_quantize_owned_kernel<<<midq_grid, 256>>>(
            midq,
            (const float *)mid->ptr,
            (const int32_t *)selected->ptr,
            expert_mid_dim,
            6u,
            resident_expert_base,
            resident_expert_count);
    if (!cuda_ok(cudaGetLastError(), "owned routed_moe mid quantize launch")) return 0;

    dim3 down_grid((out_dim + 31u) / 32u, pack_fixed3 ? 4u : 6u, 1u);
    if (q4k_path && pack_fixed3) {
        moe_down_q4K_owned_packed_qwarp32_kernel<<<down_grid, 256>>>(
                down_dst,
                down_w,
                midq,
                (const int32_t *)selected->ptr,
                down_expert_bytes,
                down_row_bytes,
                midq_blocks,
                out_dim,
                resident_expert_base,
                resident_expert_count);
    } else if (q4k_path) {
        moe_down_q4K_owned_slots_qwarp32_kernel<<<down_grid, 256>>>(
                down_dst,
                down_w,
                midq,
                (const int32_t *)selected->ptr,
                down_expert_bytes,
                down_row_bytes,
                midq_blocks,
                out_dim,
                resident_expert_base,
                resident_expert_count);
    } else if (pack_fixed3) {
        moe_down_owned_packed_qwarp32_kernel<<<down_grid, 256>>>(
                down_dst,
                down_w,
                midq,
                (const int32_t *)selected->ptr,
                down_expert_bytes,
                down_row_bytes,
                midq_blocks,
                out_dim,
                resident_expert_base,
                resident_expert_count);
    } else {
        moe_down_owned_slots_qwarp32_kernel<<<down_grid, 256>>>(
                down_dst,
                down_w,
                midq,
                (const int32_t *)selected->ptr,
                down_expert_bytes,
                down_row_bytes,
                midq_blocks,
                out_dim,
                resident_expert_base,
                resident_expert_count);
    }
    return cuda_ok(cudaGetLastError(), "owned routed_moe down launch");
}

extern "C" int ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_slots,
        const ds4_gpu_tensor *selected,
        uint32_t out_dim,
        uint32_t expert_split,
        uint32_t rows) {
    if (!out || !home_slots || !peer_slots || !selected || out_dim == 0u ||
        rows == 0u || rows > 65535u) {
        return 0;
    }
    const uint64_t row_elems = (uint64_t)rows * out_dim;
    if (row_elems > UINT64_MAX / (6u * sizeof(float))) return 0;
    const uint64_t out_bytes = row_elems * sizeof(float);
    const uint64_t slots_bytes = row_elems * 6u * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)rows * 6u * sizeof(int32_t);
    if (out->bytes < out_bytes || home_slots->bytes < slots_bytes ||
        peer_slots->bytes < slots_bytes || selected->bytes < selected_bytes) {
        return 0;
    }
    const dim3 grid((out_dim + 255u) / 256u, rows, 1u);
    moe_owned_slots_combine_fixed3_kernel<<<grid, 256>>>(
            (float *)out->ptr,
            (const float *)home_slots->ptr,
            (const float *)peer_slots->ptr,
            (const int32_t *)selected->ptr,
            out_dim,
            expert_split);
    return cuda_ok(cudaGetLastError(),
                   "owned routed_moe slot rows combine launch");
}

extern "C" int ds4_gpu_routed_moe_owned_slots_combine_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_slots,
        const ds4_gpu_tensor *selected,
        uint32_t out_dim,
        uint32_t expert_split) {
    return ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(
            out, home_slots, peer_slots, selected,
            out_dim, expert_split, 1u);
}

extern "C" int ds4_gpu_routed_moe_owned_packed_combine_tensor(
        ds4_gpu_tensor *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_packed,
        const ds4_gpu_tensor *selected,
        uint32_t out_dim,
        uint32_t expert_split) {
    const uint64_t home_bytes = 6ull * out_dim * sizeof(float);
    const uint64_t peer_bytes = 4ull * out_dim * sizeof(float);
    if (!out || !home_slots || !peer_packed || !selected || out_dim == 0u ||
        out->bytes < (uint64_t)out_dim * sizeof(float) ||
        home_slots->bytes < home_bytes || peer_packed->bytes < peer_bytes ||
        selected->bytes < 6u * sizeof(int32_t)) {
        return 0;
    }
    moe_owned_packed_combine_fixed3_kernel<<<
            (out_dim + 255u) / 256u, 256>>>(
            (float *)out->ptr,
            (const float *)home_slots->ptr,
            (const float *)peer_packed->ptr,
            (const int32_t *)selected->ptr,
            out_dim,
            expert_split);
    return cuda_ok(cudaGetLastError(),
                   "owned routed_moe packed combine launch");
}

extern "C" int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *add_in,
        uint32_t layer_index,
        bool force_resident) {
    if (add_in) {
        if (!ds4_gpu_add_tensor(out, out, add_in,
                                (uint32_t)(out->bytes / sizeof(float)))) return 0;
    }
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x,
                             layer_index, 1, force_resident ? 0 : 1, 0);
}
extern "C" int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16, bool force_resident) {
    (void)force_resident;
    if (mid_is_f16) *mid_is_f16 = false;
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x,
                             layer_index, n_tokens, 1, 0);
}

extern "C" int ds4_gpu_routed_moe_batch_owned_tensor(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        ds4_gpu_tensor *selected,
        ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t resident_expert_base,
        uint32_t resident_expert_count,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t layer_index,
        uint32_t n_tokens,
        bool *mid_is_f16) {
    if (mid_is_f16) *mid_is_f16 = false;
    if (!selected || !weights || n_tokens == 0u || n_expert == 0u ||
        n_total_expert == 0u || resident_expert_count == 0u ||
        gate_expert_bytes == 0u || gate_row_bytes == 0u ||
        down_expert_bytes == 0u || down_row_bytes == 0u ||
        resident_expert_base >= n_total_expert ||
        resident_expert_count > n_total_expert - resident_expert_base ||
        resident_expert_base > UINT64_MAX / gate_expert_bytes ||
        resident_expert_base > UINT64_MAX / down_expert_bytes) {
        return 0;
    }
    const uint64_t pair_count = (uint64_t)n_tokens * n_expert;
    if (pair_count > UINT32_MAX ||
        selected->bytes < pair_count * sizeof(int32_t) ||
        weights->bytes < pair_count * sizeof(float)) {
        return 0;
    }
    const uint64_t gate_shift =
        (uint64_t)resident_expert_base * gate_expert_bytes;
    const uint64_t down_shift =
        (uint64_t)resident_expert_base * down_expert_bytes;
    if (gate_offset > model_size || gate_shift > model_size - gate_offset ||
        up_offset > model_size || gate_shift > model_size - up_offset ||
        down_offset > model_size || down_shift > model_size - down_offset) {
        return 0;
    }
    moe_filter_owned_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
            (int32_t *)selected->ptr,
            (float *)weights->ptr,
            pair_count,
            n_total_expert,
            resident_expert_base,
            resident_expert_count);
    if (!cuda_ok(cudaGetLastError(), "owned routed_moe pair filter launch")) return 0;

    return routed_moe_launch(
            out, gate, up, mid, down, model_map, model_size,
            gate_offset + gate_shift,
            up_offset + gate_shift,
            down_offset + down_shift,
            gate_type, down_type,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            expert_in_dim, expert_mid_dim, out_dim,
            selected, weights, resident_expert_count, n_expert,
            clamp, x, layer_index, n_tokens, 0, 1);
}
extern "C" int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    if (!out || !mix || !model_map || n_hc != 4) return 0;
    const uint64_t mix_bytes = 24ull * sizeof(float);
    if (scale_offset > model_size || model_size - scale_offset < 3ull * sizeof(float) ||
        base_offset > model_size || model_size - base_offset < mix_bytes ||
        mix->bytes < mix_bytes || out->bytes < mix_bytes) return 0;
    const int logical_tier = ds4_tensor_device_idx(out);
    const float *scale = (const float *)cuda_resolve_weight_ptr(model_map, scale_offset, 3ull * sizeof(float), logical_tier, "hc_scale");
    const float *base = (const float *)cuda_resolve_weight_ptr(model_map, base_offset, mix_bytes, logical_tier, "hc_base");
    if (!scale || !base) return 0;
    uint32_t n_rows = (uint32_t)(mix->bytes / mix_bytes);
    if (out->bytes / mix_bytes < n_rows) n_rows = (uint32_t)(out->bytes / mix_bytes);
    hc_split_sinkhorn_kernel<<<(n_rows + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)mix->ptr,
        scale,
        base,
        n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc_split_sinkhorn launch");
}
extern "C" int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)weights->ptr,
        n_embd, n_hc, n_tokens, n_hc);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum launch");
}
extern "C" int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    uint32_t stride = (uint32_t)(2u * n_hc + n_hc * n_hc);
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)split->ptr,
        n_embd, n_hc, n_tokens, stride);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum_split launch");
}
extern "C" int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset) {
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        return 0;
    }
    const int logical_tier = ds4_tensor_device_idx(out);
    const float *scale = (const float *)cuda_resolve_weight_ptr(model_map, scale_offset, 3ull * sizeof(float), logical_tier, "hc_scale");
    const float *base = (const float *)cuda_resolve_weight_ptr(model_map, base_offset, mix_bytes, logical_tier, "hc_base");
    if (!scale || !base) return 0;
    hc_split_weighted_sum_fused_kernel<<<(uint32_t)n_rows, 256, 0, cuda_decode_stream()>>>(
            (float *)out->ptr,
            (float *)split->ptr,
            (const float *)mix->ptr,
            (const float *)residual_hc->ptr,
            scale,
            base,
            n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc split weighted sum launch");
}
extern "C" int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (getenv("DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED") == NULL) {
        if (!out || !norm_out || !split || !mix || !residual_hc || !model_map ||
            n_embd == 0 || n_hc != 4) {
            return 0;
        }
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        const uint64_t mix_bytes = mix_hc * sizeof(float);
        const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
            norm_out->bytes < out->bytes ||
            scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset ||
            norm_weight_offset > model_size ||
            (uint64_t)n_embd * sizeof(float) > model_size - norm_weight_offset) {
            return 0;
        }
        uint64_t n_rows = out->bytes / out_row_bytes;
        if (n_rows == 1) {
            if (mix->bytes < n_rows * mix_bytes ||
                split->bytes < n_rows * mix_bytes ||
                residual_hc->bytes < n_rows * residual_row_bytes) {
                return 0;
            }
            const int logical_tier = ds4_tensor_device_idx(out);
            const float *scale = (const float *)cuda_resolve_weight_ptr(model_map, scale_offset,
                    3ull * sizeof(float), logical_tier, "hc_scale");
            const float *base = (const float *)cuda_resolve_weight_ptr(model_map, base_offset,
                    mix_bytes, logical_tier, "hc_base");
            const float *norm_w = (const float *)cuda_resolve_weight_ptr(model_map, norm_weight_offset,
                    (uint64_t)n_embd * sizeof(float), logical_tier, "hc_norm_weight");
            if (!scale || !base || !norm_w) return 0;
            hc_split_weighted_sum_norm_fused_kernel<<<(uint32_t)n_rows, 256, 0, cuda_decode_stream()>>>(
                    (float *)out->ptr,
                    (float *)norm_out->ptr,
                    (float *)split->ptr,
                    (const float *)mix->ptr,
                    (const float *)residual_hc->ptr,
                    scale,
                    base,
                    norm_w,
                    n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps);
            return cuda_ok(cudaGetLastError(), "hc split weighted sum norm launch");
        }
    }
    /* Multi-row fallback: norm EVERY row (rms_norm_weight_tensor is the
     * single-row entry and would leave rows 1..n-1 of norm_out untouched). */
    if (!out || n_embd == 0) return 0;
    return ds4_gpu_hc_split_weighted_sum_tensor(out, split, mix, residual_hc,
                                                  model_map, model_size,
                                                  scale_offset, base_offset,
                                                  n_embd, n_hc,
                                                  sinkhorn_iters, eps) &&
           ds4_gpu_rms_norm_weight_rows_tensor(
                   norm_out, out, model_map, model_size,
                   norm_weight_offset, n_embd,
                   (uint32_t)(out->bytes /
                              ((uint64_t)n_embd * sizeof(float))),
                   norm_eps);
}
extern "C" int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!out || !pre || !model_map || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (row_bytes == 0 || out->bytes < row_bytes || out->bytes % row_bytes != 0 ||
        pre->bytes < out->bytes ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || row_bytes > model_size - base_offset) {
        return 0;
    }
    const uint64_t n_tokens = out->bytes / row_bytes;
    const int logical_tier = ds4_tensor_device_idx(out);
    const float *scale = (const float *)cuda_resolve_weight_ptr(model_map, scale_offset, sizeof(float), logical_tier, "output_hc_scale");
    const float *base = (const float *)cuda_resolve_weight_ptr(model_map, base_offset, row_bytes, logical_tier, "output_hc_base");
    if (!scale || !base) return 0;
    uint64_t n = n_tokens * n_hc;
    output_hc_weights_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr,
            (const float *)pre->ptr,
            scale,
            base,
            n_hc,
            (uint32_t)n_tokens,
            eps);
    return cuda_ok(cudaGetLastError(), "output hc weights launch");
}
extern "C" int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !post || !comb || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    (const float *)post->ptr,
                                                    (const float *)comb->ptr,
                                                    n_embd, n_hc, n_tokens,
                                                    n_hc, n_hc * n_hc, 0, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand launch");
}
extern "C" int ds4_gpu_hc_expand_add_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !block_add || !residual_hc || !post || !comb ||
        n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_add->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    (const float *)post->ptr,
                                                    (const float *)comb->ptr,
                                                    n_embd, n_hc, n_tokens,
                                                    n_hc, n_hc * n_hc, 1, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand_add launch");
}
extern "C" int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 0, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand_split launch");
}
extern "C" int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !block_add || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256, 0, cuda_decode_stream()>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_add->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 1, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand_add_split launch");
}

extern "C" int ds4_gpu_hc_expand_add2_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *block_add2, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !block_add || !block_add2 ||
        !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_add->ptr,
                                                    (const float *)block_add2->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 1, 1);
    return cuda_ok(cudaGetLastError(), "hc_expand_add2_split launch");
}

extern "C" int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (getenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") == NULL) {
        return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, shared_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        shared_mid,
                                                        routed_out,
                                                        NULL,
                                                        NULL, NULL, NULL, 0,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "shared_down_hc_expand");
    }
    return ds4_gpu_matmul_q8_0_tensor(shared_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim,
                                        shared_mid, 1) &&
           ds4_gpu_hc_expand_add_split_tensor(out_hc, shared_out, routed_out,
	                                                residual_hc, split, n_embd, n_hc);
}

extern "C" int ds4_gpu_shared_down_hc_expand_add_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *routed_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (getenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") == NULL) {
        return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, shared_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        shared_mid,
                                                        routed_out,
                                                        routed_add,
                                                        NULL, NULL, NULL, 0,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "shared_down_hc_expand_add");
    }
    return ds4_gpu_matmul_q8_0_tensor(shared_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim,
                                        shared_mid, 1) &&
           ds4_gpu_hc_expand_add2_split_tensor(out_hc, shared_out, routed_out,
                                                routed_add, residual_hc, split,
                                                n_embd, n_hc);
}

extern "C" int ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor(
        ds4_gpu_tensor *out_hc,
        ds4_gpu_tensor *shared_out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_packed,
        const ds4_gpu_tensor *selected,
        uint32_t expert_split,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t n_embd,
        uint32_t n_hc) {
    if (getenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") != NULL) return 0;
    return cuda_matmul_q8_0_hc_expand_tensor_labeled(
            out_hc,
            shared_out,
            model_map,
            model_size,
            weight_offset,
            in_dim,
            out_dim,
            shared_mid,
            NULL,
            NULL,
            home_slots,
            peer_packed,
            selected,
            expert_split,
            residual_hc,
            split,
            n_embd,
            n_hc,
            "shared_down_hc_expand_owned");
}

extern "C" int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (getenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") == NULL) {
        return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, block_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        x,
                                                        NULL,
                                                        NULL,
                                                        NULL, NULL, NULL, 0,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "q8_hc_expand");
    }
    return ds4_gpu_matmul_q8_0_tensor(block_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual_hc,
                                            split, n_embd, n_hc);
}

#if 0
/* --gpu-vram auto probe. Defined here (in the .cu unit) so the
 * C-side parser (ds4_gpu_args.c) does not need to include
 * <cuda_runtime.h>. Returns 0 on success, nonzero on error
 * (errbuf populated). See ds4_gpu_args.h.
 *
 * Side-effect-light: changes cudaSetDevice during probing; callers
 * that care about the active device should reset it themselves
 * before continuing. (The mgpu init path resets it anyway.) */
extern "C" int ds4_gpu_args_probe_auto_cuda(const int      *device_filter,
                                              int             filter_len,
                                              ds4_gpu_config *out,
                                              size_t          safety_margin_bytes,
                                              char           *errbuf,
                                              size_t          errbuflen) {
    if (!out) {
        if (errbuf && errbuflen) snprintf(errbuf, errbuflen, "internal: NULL out");
        return 1;
    }
    int visible = 0;
    cudaError_t rc = cudaGetDeviceCount(&visible);
    if (rc != cudaSuccess || visible <= 0) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen,
                     "cudaGetDeviceCount failed: %s",
                     rc == cudaSuccess ? "no devices" : cudaGetErrorString(rc));
        }
        return 1;
    }
    /* Build the device list: either the explicit filter or 0..visible-1. */
    int devs[DS4_MAX_GPUS];
    int n_dev = 0;
    if (device_filter && filter_len > 0) {
        if (filter_len > DS4_MAX_GPUS) {
            if (errbuf && errbuflen) {
                snprintf(errbuf, errbuflen,
                         "--gpu-devices filter has %d entries (max %d)",
                         filter_len, DS4_MAX_GPUS);
            }
            return 1;
        }
        for (int i = 0; i < filter_len; i++) {
            int d = device_filter[i];
            if (d < 0 || d >= visible) {
                if (errbuf && errbuflen) {
                    snprintf(errbuf, errbuflen,
                             "--gpu-devices: device %d not in 0..%d",
                             d, visible - 1);
                }
                return 1;
            }
            devs[n_dev++] = d;
        }
    } else {
        int cap = visible < DS4_MAX_GPUS ? visible : DS4_MAX_GPUS;
        for (int i = 0; i < cap; i++) devs[n_dev++] = i;
    }
    out->n_gpus = n_dev;
    out->safety_margin_bytes = safety_margin_bytes;
    for (int i = 0; i < n_dev; i++) {
        int d = devs[i];
        rc = cudaSetDevice(d);
        if (rc != cudaSuccess) {
            if (errbuf && errbuflen) {
                snprintf(errbuf, errbuflen,
                         "cudaSetDevice(%d) failed: %s",
                         d, cudaGetErrorString(rc));
            }
            return 1;
        }
        size_t free_b = 0, total_b = 0;
        rc = cudaMemGetInfo(&free_b, &total_b);
        if (rc != cudaSuccess) {
            if (errbuf && errbuflen) {
                snprintf(errbuf, errbuflen,
                         "cudaMemGetInfo on device %d failed: %s",
                         d, cudaGetErrorString(rc));
            }
            return 1;
        }
        /* Auto-mode reserve. Auto-probe is the only place we override
         * the user's stated budget, so this is where the conservative-
         * on-the-user's-behalf reserve belongs. The
         * engine path (engine_classify_multi_tier) still subtracts the
         * user-supplied safety_margin_bytes + the cuBLAS workspace from
         * whatever budget we hand back; that math is unchanged and
         * applies on top of the reserve we trim here.
         *
         * Reserve = max(2 GiB, 5 % of free). Why these numbers:
         *   - 2 GiB floor covers runtime scratch / Q8 dequant caches /
         *     MTP optional state on small GPUs (8-12 GB cards) where
         *     5 % is < 1 GiB and not enough headroom.
         *   - 5 % of free scales the reserve up on larger cards where
         *     workspace + KV growth needs proportionally more room.
         * Explicit --gpu-vram 47,37 budgets do not go through this
         * probe and are unaffected. */
        const size_t reserve_floor = (size_t)2ull * 1024ull * 1024ull * 1024ull;
        const size_t reserve_pct = free_b / 20u;
        const size_t reserve = reserve_floor > reserve_pct ? reserve_floor : reserve_pct;
        const size_t budget = free_b > reserve ? (free_b - reserve) : 0;
        (void)safety_margin_bytes;
        out->device_indices[i] = d;
        out->vram_bytes[i] = budget;
    }
    return 0;
}
#endif

#if 0


static int cuda_stream_selected_ensure_bytes(
        char **ptr, uint64_t *capacity, uint64_t bytes, const char *label) {
    if (*ptr && *capacity >= bytes) return 1;
    if (*ptr) {
        (void)cudaFree(*ptr);
        *ptr = NULL;
        *capacity = 0;
    }
    if (bytes == 0 || bytes > (uint64_t)SIZE_MAX) return 0;
    cudaError_t err = cudaMalloc((void **)ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA streaming %s allocation failed for %.2f MiB: %s\n",
                label, (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    *capacity = bytes;
    return 1;
}

static int cuda_stream_selected_ensure_i32(uint64_t count) {
    if (count == 0 || count > UINT64_MAX / sizeof(int32_t)) return 0;
    const uint64_t bytes = count * sizeof(int32_t);
    return cuda_stream_selected_ensure_bytes(
            (char **)&g_stream_selected_cache.slot_selected_ptr,
            &g_stream_selected_cache.slot_selected_capacity,
            bytes,
            "selected-id remap");
}

static int cuda_stream_selected_ranges_valid(
        const ds4_gpu_stream_expert_table *table) {
    if (!table || !table->model_map || table->model_size == 0 ||
        table->n_total_expert == 0 || table->gate_expert_bytes == 0 ||
        table->down_expert_bytes == 0) {
        return 0;
    }
    if ((uint64_t)table->n_total_expert >
            UINT64_MAX / table->gate_expert_bytes ||
        (uint64_t)table->n_total_expert >
            UINT64_MAX / table->down_expert_bytes) {
        return 0;
    }
    const uint64_t gate_bytes =
        (uint64_t)table->n_total_expert * table->gate_expert_bytes;
    const uint64_t down_bytes =
        (uint64_t)table->n_total_expert * table->down_expert_bytes;
    return table->gate_offset <= table->model_size &&
           gate_bytes <= table->model_size - table->gate_offset &&
           table->up_offset <= table->model_size &&
           gate_bytes <= table->model_size - table->up_offset &&
           table->down_offset <= table->model_size &&
           down_bytes <= table->model_size - table->down_offset;
}

static int cuda_stream_selected_cache_begin_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *selected_ids,
        uint32_t slot_count) {
    cuda_stream_selected_cache_invalidate();
    if (!g_ssd_streaming_mode) return 1;
    if (!cuda_stream_selected_ranges_valid(table) || !selected_ids ||
        slot_count == 0) {
        return 0;
    }
    if (g_n_gpus != 1) {
        fprintf(stderr,
                "ds4: CUDA SSD streaming requires single-GPU placement\n");
        return 0;
    }

    std::vector<int32_t> expert_to_slot;
    std::vector<int32_t> compact_ids;
    std::vector<int32_t> slot_ids;
    try {
        expert_to_slot.assign(table->n_total_expert, -1);
        compact_ids.reserve(slot_count < table->n_total_expert ?
                            slot_count : table->n_total_expert);
        slot_ids.resize(slot_count);
    } catch (...) {
        return 0;
    }
    for (uint32_t i = 0; i < slot_count; i++) {
        const int32_t expert = selected_ids[i];
        if (expert < 0 || (uint32_t)expert >= table->n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming expert id %d is outside 0..%u at layer %u\n",
                    expert, table->n_total_expert, table->layer);
            return 0;
        }
        int32_t compact = expert_to_slot[(uint32_t)expert];
        if (compact < 0) {
            compact = (int32_t)compact_ids.size();
            expert_to_slot[(uint32_t)expert] = compact;
            compact_ids.push_back(expert);
        }
        slot_ids[i] = compact;
    }
    if (compact_ids.empty() || compact_ids.size() > UINT32_MAX) return 0;
    const uint64_t compact_count = compact_ids.size();
    if (compact_count > UINT64_MAX / table->gate_expert_bytes ||
        compact_count > UINT64_MAX / table->down_expert_bytes) {
        return 0;
    }
    const uint64_t gate_bytes = compact_count * table->gate_expert_bytes;
    const uint64_t down_bytes = compact_count * table->down_expert_bytes;
    const int logical_tier = 0;
    if (g_stream_selected_cache.logical_tier != logical_tier &&
        (g_stream_selected_cache.gate_ptr ||
         g_stream_selected_cache.up_ptr ||
         g_stream_selected_cache.down_ptr ||
         g_stream_selected_cache.slot_selected_ptr)) {
        cuda_stream_selected_cache_release();
    }
    if (ds4_gpu_set_current_device(logical_tier) != 0 ||
        !cuda_stream_selected_ensure_bytes(
                &g_stream_selected_cache.gate_ptr,
                &g_stream_selected_cache.gate_capacity,
                gate_bytes, "gate experts") ||
        !cuda_stream_selected_ensure_bytes(
                &g_stream_selected_cache.up_ptr,
                &g_stream_selected_cache.up_capacity,
                gate_bytes, "up experts") ||
        !cuda_stream_selected_ensure_bytes(
                &g_stream_selected_cache.down_ptr,
                &g_stream_selected_cache.down_capacity,
                down_bytes, "down experts") ||
        !cuda_stream_selected_ensure_i32(slot_count)) {
        cuda_stream_selected_cache_invalidate();
        return 0;
    }

    for (uint32_t i = 0; i < compact_ids.size(); i++) {
        const uint64_t expert = (uint32_t)compact_ids[i];
        const uint64_t gate_src =
            table->gate_offset + expert * table->gate_expert_bytes;
        const uint64_t up_src =
            table->up_offset + expert * table->gate_expert_bytes;
        const uint64_t down_src =
            table->down_offset + expert * table->down_expert_bytes;
        const uint64_t gate_dst = (uint64_t)i * table->gate_expert_bytes;
        const uint64_t down_dst = (uint64_t)i * table->down_expert_bytes;
        if (!cuda_model_copy_to_device_streamed(
                    g_stream_selected_cache.gate_ptr + gate_dst,
                    table->model_map, table->model_size,
                    gate_src, table->gate_expert_bytes,
                    "stream gate expert copy") ||
            !cuda_model_copy_to_device_streamed(
                    g_stream_selected_cache.up_ptr + gate_dst,
                    table->model_map, table->model_size,
                    up_src, table->gate_expert_bytes,
                    "stream up expert copy") ||
            !cuda_model_copy_to_device_streamed(
                    g_stream_selected_cache.down_ptr + down_dst,
                    table->model_map, table->model_size,
                    down_src, table->down_expert_bytes,
                    "stream down expert copy")) {
            cuda_stream_selected_cache_invalidate();
            return 0;
        }
    }
    if (!cuda_ok(cudaMemcpy(g_stream_selected_cache.slot_selected_ptr,
                            slot_ids.data(),
                            (size_t)slot_count * sizeof(int32_t),
                            cudaMemcpyHostToDevice),
                 "stream selected-id remap copy")) {
        cuda_stream_selected_cache_invalidate();
        return 0;
    }

    g_stream_selected_cache.logical_tier = logical_tier;
    g_stream_selected_cache.model_map = table->model_map;
    g_stream_selected_cache.layer = table->layer;
    g_stream_selected_cache.n_total_expert = table->n_total_expert;
    g_stream_selected_cache.slot_count = slot_count;
    g_stream_selected_cache.compact_count = (uint32_t)compact_count;
    g_stream_selected_cache.gate_offset = table->gate_offset;
    g_stream_selected_cache.up_offset = table->up_offset;
    g_stream_selected_cache.down_offset = table->down_offset;
    g_stream_selected_cache.gate_expert_bytes = table->gate_expert_bytes;
    g_stream_selected_cache.down_expert_bytes = table->down_expert_bytes;
    g_stream_selected_cache.slot_selected_tensor.ptr =
        g_stream_selected_cache.slot_selected_ptr;
    g_stream_selected_cache.slot_selected_tensor.bytes =
        (uint64_t)slot_count * sizeof(int32_t);
    g_stream_selected_cache.slot_selected_tensor.owner = 0;
    g_stream_selected_cache.slot_selected_tensor.device_id = logical_tier;
    g_stream_selected_cache.valid = 1;
    return 1;
}
#endif
