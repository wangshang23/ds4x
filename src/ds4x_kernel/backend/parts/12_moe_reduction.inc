/* Map one of two packed operands for a three-slot reduction group. The only
 * multi-slot operand is the peer-owned prefix (slots 0+1 within the group),
 * which can be pre-added exactly because the reference reduction starts from
 * +0. Every other peer slot remains a distinct operand in original order. */
__device__ __forceinline__ static int moe_owned_packed_component(
        const int32_t *selected,
        uint32_t group,
        uint32_t component,
        uint32_t expert_base,
        uint32_t expert_count,
        bool *prefix_pair) {
    const uint32_t slot0 = group * 3u;
    uint32_t mask = 0u;
    #pragma unroll
    for (uint32_t i = 0; i < 3u; i++) {
        if (moe_owned_local_expert(selected[slot0 + i], expert_base,
                                   expert_count, NULL)) {
            mask |= 1u << i;
        }
    }
    *prefix_pair = false;
    if ((mask & 3u) == 3u) {
        if (component == 0u) {
            *prefix_pair = true;
            return (int)slot0;
        }
        return (mask & 4u) != 0u ? (int)(slot0 + 2u) : -1;
    }
    uint32_t ordinal = 0u;
    #pragma unroll
    for (uint32_t i = 0; i < 3u; i++) {
        if ((mask & (1u << i)) == 0u) continue;
        if (ordinal++ == component) return (int)(slot0 + i);
    }
    return -1;
}

/* One thread owns an output column and loads all source slots before writing.
 * This permits packed_out == slots for the non-peer-copy fallback. */
__global__ static void moe_down_owned_pack_f32_slots_kernel(
        float *packed_out,
        const float *slots,
        const int32_t *selected,
        uint32_t out_dim,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t col =
        (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (col >= out_dim) return;

    float slotv[6];
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        slotv[slot] = slots[(uint64_t)slot * out_dim + col];
    }

    float packed[4];
    #pragma unroll
    for (uint32_t packed_slot = 0; packed_slot < 4u; packed_slot++) {
        bool prefix_pair = false;
        const int first_slot = moe_owned_packed_component(
                selected, packed_slot / 2u, packed_slot & 1u,
                expert_base, expert_count, &prefix_pair);
        if (first_slot < 0) {
            packed[packed_slot] = 0.0f;
        } else if (prefix_pair) {
            float value = __fadd_rn(0.0f, slotv[first_slot]);
            packed[packed_slot] = __fadd_rn(value, slotv[first_slot + 1]);
        } else {
            packed[packed_slot] = slotv[first_slot];
        }
    }

    #pragma unroll
    for (uint32_t packed_slot = 0; packed_slot < 4u; packed_slot++) {
        packed_out[(uint64_t)packed_slot * out_dim + col] = packed[packed_slot];
    }
}

__global__ static void moe_down_owned_copy_f32_slots_kernel(
        float *dst,
        const float *src,
        uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = src[i];
}

__global__ static void moe_down_owned_packed_qwarp32_kernel(
        float *packed_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t packed_slot = blockIdx.y;
    if (row >= out_dim || packed_slot >= 4u) return;
    bool prefix_pair = false;
    const int first_slot = moe_owned_packed_component(
            selected, packed_slot / 2u, packed_slot & 1u,
            expert_base, expert_count, &prefix_pair);
    if (first_slot < 0) {
        if (lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = 0.0f;
        return;
    }

    float packed = 0.0f;
    const uint32_t n_slots = prefix_pair ? 2u : 1u;
    #pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
        if (i >= n_slots) break;
        const uint32_t slot = (uint32_t)first_slot + i;
        uint32_t expert = 0;
        if (!moe_owned_local_expert(selected[slot], expert_base,
                                    expert_count, &expert)) {
            continue;
        }
        const cuda_block_q2_K *wr =
            (const cuda_block_q2_K *)(down_base +
                (uint64_t)expert * down_expert_bytes +
                (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0u) {
            packed = prefix_pair ? __fadd_rn(packed, acc) : acc;
        }
    }
    if (lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = packed;
}

__global__ static void moe_down_sum3_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 3u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    const bool vec_ok = ((((uintptr_t)down_base | down_row_bytes | down_expert_bytes) & 15u) == 0u);
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        if (vec_ok) {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b, &acc);
        } else {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_owned_slots_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t slot = blockIdx.y;
    if (row >= out_dim || slot >= 6u) return;
    uint32_t expert = 0;
    if (!moe_owned_local_expert(selected[slot], expert_base,
                                expert_count, &expert)) {
        return;
    }
    const cuda_block_q4_K *wr =
        (const cuda_block_q4_K *)(down_base +
            (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
    const bool vec_ok = ((((uintptr_t)down_base | down_row_bytes |
                           down_expert_bytes) & 15u) == 0u);
    float acc = 0.0f;
    if (vec_ok) {
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b, &acc);
        }
    } else {
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
    }
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)slot * out_dim + row] = acc;
}

__global__ static void moe_down_q4K_owned_packed_qwarp32_kernel(
        float *packed_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t expert_base,
        uint32_t expert_count) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t packed_slot = blockIdx.y;
    if (row >= out_dim || packed_slot >= 4u) return;
    bool prefix_pair = false;
    const int first_slot = moe_owned_packed_component(
            selected, packed_slot / 2u, packed_slot & 1u,
            expert_base, expert_count, &prefix_pair);
    if (first_slot < 0) {
        if (lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = 0.0f;
        return;
    }

    const bool vec_ok = ((((uintptr_t)down_base | down_row_bytes |
                           down_expert_bytes) & 15u) == 0u);
    float packed = 0.0f;
    const uint32_t n_slots = prefix_pair ? 2u : 1u;
    #pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
        if (i >= n_slots) break;
        const uint32_t slot = (uint32_t)first_slot + i;
        uint32_t expert = 0;
        if (!moe_owned_local_expert(selected[slot], expert_base,
                                    expert_count, &expert)) {
            continue;
        }
        const cuda_block_q4_K *wr =
            (const cuda_block_q4_K *)(down_base +
                (uint64_t)expert * down_expert_bytes +
                (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        if (vec_ok) {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b, &acc);
            }
        } else {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) {
                acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
            }
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0u) {
            packed = prefix_pair ? __fadd_rn(packed, acc) : acc;
        }
    }
    if (lane == 0u) packed_out[(uint64_t)packed_slot * out_dim + row] = packed;
}

__global__ static void moe_owned_slots_combine_fixed3_kernel(
        float *out,
        const float *home_slots,
        const float *peer_slots,
        const int32_t *selected,
        uint32_t out_dim,
        uint32_t expert_split) {
    const uint32_t col =
        (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t row = blockIdx.y;
    if (col >= out_dim) return;
    out += (uint64_t)row * out_dim;
    home_slots += (uint64_t)row * 6u * out_dim;
    peer_slots += (uint64_t)row * 6u * out_dim;
    selected += (uint64_t)row * 6u;
    float slotv[6];
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        const int32_t expert = selected[slot];
        if (expert < 0 ||
            (uint32_t)expert >= 2u * expert_split) {
            slotv[slot] = 0.0f;
        } else {
            const bool on_home = (uint32_t)expert < expert_split;
            const float *src = on_home ? home_slots : peer_slots;
            slotv[slot] = src[(uint64_t)slot * out_dim + col];
        }
    }
    float home = __fadd_rn(0.0f, slotv[0]);
    home = __fadd_rn(home, slotv[1]);
    home = __fadd_rn(home, slotv[2]);
    float peer = __fadd_rn(0.0f, slotv[3]);
    peer = __fadd_rn(peer, slotv[4]);
    peer = __fadd_rn(peer, slotv[5]);
    out[col] = __fadd_rn(home, peer);
}

__device__ static float moe_owned_packed_combine_row(
        const float *home_slots,
        const float *peer_packed,
        const int32_t *selected,
        uint32_t row,
        uint32_t out_dim,
        uint32_t expert_split) {
    float groups[2];
    #pragma unroll
    for (uint32_t group = 0; group < 2u; group++) {
        const uint32_t slot0 = group * 3u;
        uint32_t peer_mask = 0u;
        uint32_t valid_mask = 0u;
        #pragma unroll
        for (uint32_t i = 0; i < 3u; i++) {
            const int32_t expert = selected[slot0 + i];
            if (expert >= 0 && (uint32_t)expert < 2u * expert_split) {
                valid_mask |= 1u << i;
            }
            if (expert >= 0 && (uint32_t)expert >= expert_split &&
                (uint32_t)expert < 2u * expert_split) {
                peer_mask |= 1u << i;
            }
        }
        const float *packed = peer_packed +
            (uint64_t)group * 2u * out_dim + row;
        float acc;
        if ((peer_mask & 3u) == 3u) {
            /* packed[0] is already (+0 + slot0) + slot1. */
            acc = packed[0];
            float slot2 = 0.0f;
            if ((peer_mask & 4u) != 0u) {
                slot2 = packed[out_dim];
            } else if ((valid_mask & 4u) != 0u) {
                slot2 = home_slots[(uint64_t)(slot0 + 2u) * out_dim + row];
            }
            acc = __fadd_rn(acc, slot2);
        } else {
            acc = 0.0f;
            uint32_t peer_operand = 0u;
            #pragma unroll
            for (uint32_t i = 0; i < 3u; i++) {
                float value;
                if ((peer_mask & (1u << i)) != 0u) {
                    value = packed[(uint64_t)peer_operand * out_dim];
                    peer_operand++;
                } else if ((valid_mask & (1u << i)) != 0u) {
                    value = home_slots[(uint64_t)(slot0 + i) * out_dim + row];
                } else {
                    value = 0.0f;
                }
                acc = __fadd_rn(acc, value);
            }
        }
        groups[group] = acc;
    }
    return __fadd_rn(groups[0], groups[1]);
}

__global__ static void moe_owned_packed_combine_fixed3_kernel(
        float *out,
        const float *home_slots,
        const float *peer_packed,
        const int32_t *selected,
        uint32_t out_dim,
        uint32_t expert_split) {
    const uint32_t row =
        (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= out_dim) return;
    out[row] = moe_owned_packed_combine_row(
            home_slots, peer_packed, selected, row, out_dim, expert_split);
}

__global__ static void moe_down_q4K_sum3_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    const bool vec_ok = ((((uintptr_t)down_base | down_row_bytes | down_expert_bytes) & 15u) == 0u);
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 3u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        if (vec_ok) {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b, &acc);
        } else {
            for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_sum3_slotwarp_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t slot = lane >> 3u;
    const uint32_t qlane = lane & 7u;
    const uint32_t row = blockIdx.x * 8u + warp;
    if (row >= out_dim) return;

    float acc = 0.0f;
    if (slot < 3u) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr =
            (const cuda_block_q4_K *)(down_base +
                                      (uint64_t)(uint32_t)expert_i * down_expert_bytes +
                                      (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        for (uint32_t b = qlane; b < midq_blocks; b += 8u) {
            acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        }
        acc = quarter_warp_sum_f32(acc, qlane);
    }

    const float s1 = __shfl_sync(0xffffffffu, acc, 8);
    const float s2 = __shfl_sync(0xffffffffu, acc, 16);
    if (lane == 0u) {
        const float s0 = acc;
        out[row] = (s0 + s1) + s2;
    }
}

static void routed_moe_decode_graph_destroy_one(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= DS4_MAX_GPUS) return;
    cuda_moe_decode_graph_cache *c = &g_moe_decode_graph[logical_tier];
    if (c->exec) (void)cudaGraphExecDestroy(c->exec);
    if (c->graph) (void)cudaGraphDestroy(c->graph);
    memset(c, 0, sizeof(*c));
}

static int routed_moe_decode_q4_graph_launch(
        int logical_tier,
        float *out,
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_w,
        const char *up_w,
        const char *down_w,
        cuda_block_q8_K *xq,
        cuda_block_q8_K *midq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp,
        const float *x) {
    if (logical_tier < 0 || logical_tier >= DS4_MAX_GPUS) return 0;
    if (n_expert != 3u && n_expert != 6u) return 0;
    uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    if (xq_blocks == 0u || midq_blocks == 0u) return 0;

    cuda_moe_decode_graph_cache *c = &g_moe_decode_graph[logical_tier];
    const bool shape_match =
        c->valid &&
        c->n_expert == n_expert &&
        c->expert_in_dim == expert_in_dim &&
        c->expert_mid_dim == expert_mid_dim &&
        c->out_dim == out_dim;
    if (c->valid && !shape_match) {
        routed_moe_decode_graph_destroy_one(logical_tier);
        c = &g_moe_decode_graph[logical_tier];
    }

    uint32_t x_rows = 1u;
    uint32_t mid_rows = n_expert;
    dim3 xq_grid(xq_blocks, 1, 1);
    dim3 gate_grid((expert_mid_dim + 7u) / 8u, n_expert, 1);
    dim3 midq_grid(midq_blocks, n_expert, 1);
    dim3 down_grid((out_dim + 31u) / 32u, 1, 1);
    dim3 block(256, 1, 1);

    void *xq_args[] = { &xq, &x, &expert_in_dim, &x_rows };
    cudaKernelNodeParams xq_params;
    memset(&xq_params, 0, sizeof(xq_params));
    xq_params.func = (void *)q8_K_quantize_kernel;
    xq_params.gridDim = xq_grid;
    xq_params.blockDim = block;
    xq_params.kernelParams = xq_args;

    void *gate_args[] = {
        &gate_out, &up_out, &mid_out, &gate_w, &up_w, &xq, &selected,
        &weights, &gate_expert_bytes, &gate_row_bytes, &xq_blocks,
        &expert_mid_dim, &n_expert, &write_aux, &clamp
    };
    cudaKernelNodeParams gate_params;
    memset(&gate_params, 0, sizeof(gate_params));
    gate_params.func = (void *)moe_gate_up_mid_decode_q4K_warp32_kernel;
    gate_params.gridDim = gate_grid;
    gate_params.blockDim = block;
    gate_params.kernelParams = gate_args;

    void *midq_args[] = { &midq, &mid_out, &expert_mid_dim, &mid_rows };
    cudaKernelNodeParams midq_params;
    memset(&midq_params, 0, sizeof(midq_params));
    midq_params.func = (void *)q8_K_quantize_kernel;
    midq_params.gridDim = midq_grid;
    midq_params.blockDim = block;
    midq_params.kernelParams = midq_args;

    void *down_args[] = {
        &out, &down_w, &midq, &selected, &down_expert_bytes,
        &down_row_bytes, &midq_blocks, &out_dim
    };
    cudaKernelNodeParams down_params;
    memset(&down_params, 0, sizeof(down_params));
    down_params.func = n_expert == 6u
        ? (void *)moe_down_q4K_sum6_qwarp32_kernel
        : (void *)moe_down_q4K_sum3_qwarp32_kernel;
    down_params.gridDim = down_grid;
    down_params.blockDim = block;
    down_params.kernelParams = down_args;

    if (!c->valid) {
        cudaError_t err = cudaGraphCreate(&c->graph, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: routed MoE decode graph create failed: %s\n",
                    cudaGetErrorString(err));
            routed_moe_decode_graph_destroy_one(logical_tier);
            return -1;
        }
        err = cudaGraphAddKernelNode(&c->xq_node, c->graph, NULL, 0,
                                     &xq_params);
        if (err == cudaSuccess) {
            err = cudaGraphAddKernelNode(&c->gate_node, c->graph,
                                         &c->xq_node, 1, &gate_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphAddKernelNode(&c->midq_node, c->graph,
                                         &c->gate_node, 1, &midq_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphAddKernelNode(&c->down_node, c->graph,
                                         &c->midq_node, 1, &down_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphInstantiate(&c->exec, c->graph, NULL, NULL, 0);
        }
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: routed MoE decode graph instantiate failed: %s\n",
                    cudaGetErrorString(err));
            routed_moe_decode_graph_destroy_one(logical_tier);
            return -1;
        }
        c->n_expert = n_expert;
        c->expert_in_dim = expert_in_dim;
        c->expert_mid_dim = expert_mid_dim;
        c->out_dim = out_dim;
        c->valid = 1;
    } else {
        cudaError_t err =
            cudaGraphExecKernelNodeSetParams(c->exec, c->xq_node,
                                             &xq_params);
        if (err == cudaSuccess) {
            err = cudaGraphExecKernelNodeSetParams(c->exec, c->gate_node,
                                                   &gate_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphExecKernelNodeSetParams(c->exec, c->midq_node,
                                                   &midq_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphExecKernelNodeSetParams(c->exec, c->down_node,
                                                   &down_params);
        }
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: routed MoE decode graph update failed: %s\n",
                    cudaGetErrorString(err));
            routed_moe_decode_graph_destroy_one(logical_tier);
            return -1;
        }
    }

    cudaError_t err = cudaGraphLaunch(c->exec, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: routed MoE decode graph launch failed: %s\n",
                cudaGetErrorString(err));
        routed_moe_decode_graph_destroy_one(logical_tier);
        return -1;
    }
    return 1;
}

/* Q4_K prefill (n_tokens > 1) down kernel. Mirrors moe_down_qwarp32_kernel
 * geometry exactly; only the weight block type and dot helper differ. The
 * pair = blockIdx.y indexing means the same grid shape (out_dim/32, n_tokens*n_expert)
 * used by the IQ2 path applies here. The downstream moe_sum_kernel is
 * weight-type-agnostic and sums these per-pair outputs into the final output. */
__global__ static void moe_down_q4K_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_expert_tile8_rowspan_kernel(
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
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
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
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q4_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                      xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                      xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                      xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
        }
    }
}


/* INT8 tensor-core (m8n8k16) exact MoE prefill tile kernels.
 *
 * Each warp computes an 8-token x 8-row tile. The Q4_K x Q8_K superblock dot
 * keeps its integer sums (order-invariant, exact) but computes the 32-wide
 * group dots on tensor cores; every output element keeps 8 float slot
 * accumulators (slot[b & 7] += term_b, b ascending) and reduces them with the
 * exact quarter_warp_sum_f32 grouping, so results are bit-identical to the
 * scalar expert-tile kernels (fuzz-verified). Requires sm_75+, 16B-aligned
 * expert tensors, and the staged activation-block counts (<=16 gate/up,
 * <=8 down). Rollback: DS4_CUDA_MOE_NO_Q4_MMA=1. */
__device__ __forceinline__ static void mma_m8n8k16_s8(int32_t &c0, int32_t &c1, uint32_t a, uint32_t b) {
#if __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
#else
    (void)a; (void)b; (void)c0; (void)c1;
#endif
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q4K_tile8_mma_kernel(
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
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint32_t s_pair[8];
    __shared__ uint32_t s_tok[8];
    __shared__ uint32_t s_slot[8];
    __shared__ uint32_t s_np;
    if (threadIdx.x == 0) {
        uint32_t np = 0;
        for (; np < 8u; np++) {
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
        for (uint32_t i = threadIdx.x; i < np * xq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u); i += blockDim.x) {
            const uint32_t words_per_tok = xq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
            uint32_t p = i / words_per_tok;
            uint32_t w = i - p * words_per_tok;
            ((uint32_t *)sxq[p])[w] = ((const uint32_t *)(xq + (uint64_t)s_tok[p] * xq_blocks))[w];
        }
        __syncthreads();
    }
    const uint32_t mtok = lane >> 2u;      /* token row of this thread's C elems */
    const uint32_t n0 = (lane & 3u) * 2u;  /* first C column (weight row) */
    /* 8 warps x 8 rows = 64 rows per pass */
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN + rr * 64u + warp * 8u;
        if (row0 >= expert_mid_dim) continue;
        const char *grow = gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row0 * gate_row_bytes;
        const char *urow = up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row0 * gate_row_bytes;
        /* per-element slot accumulators (2 elements x 8 slots) */
        float sg0[8] = {0,0,0,0,0,0,0,0}, sg1[8] = {0,0,0,0,0,0,0,0};
        float su0[8] = {0,0,0,0,0,0,0,0}, su1[8] = {0,0,0,0,0,0,0,0};
        for (uint32_t b = 0; b < xq_blocks; b++) {
            /* headers for this thread's two C columns */
            const uint4 ghdr0 = *(const uint4 *)((const cuda_block_q4_K *)(grow + (uint64_t)n0 * gate_row_bytes) + b);
            const uint4 ghdr1 = *(const uint4 *)((const cuda_block_q4_K *)(grow + (uint64_t)(n0 + 1u) * gate_row_bytes) + b);
            const uint4 uhdr0 = *(const uint4 *)((const cuda_block_q4_K *)(urow + (uint64_t)n0 * gate_row_bytes) + b);
            const uint4 uhdr1 = *(const uint4 *)((const cuda_block_q4_K *)(urow + (uint64_t)(n0 + 1u) * gate_row_bytes) + b);
            /* B-fragment source rows for loads: n_load = lane>>2.
             * Batch all global loads for this superblock upfront so the
             * memory system sees independent requests instead of a
             * load->mma dependency chain. */
            const uint32_t *gqw = (const uint32_t *)(((const cuda_block_q4_K *)(grow + (uint64_t)(lane >> 2u) * gate_row_bytes) + b)->qs);
            const uint32_t *uqw = (const uint32_t *)(((const cuda_block_q4_K *)(urow + (uint64_t)(lane >> 2u) * gate_row_bytes) + b)->qs);
            const int8_t *aqs = sxq[mtok][b].qs;
            uint32_t gw8[8], uw8[8];
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) {
                gw8[k] = gqw[k * 4u + (lane & 3u)];
                uw8[k] = uqw[k * 4u + (lane & 3u)];
            }
            int gi0 = 0, gi1 = 0, ui0 = 0, ui1 = 0;
            int gs0 = 0, gs1 = 0, us0 = 0, us1 = 0;
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const int shift = (j & 1u) ? 4 : 0;
                /* dot32 via two chained k16 mmas, per matrix */
                int32_t gc0 = 0, gc1 = 0, uc0 = 0, uc1 = 0;
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    const uint32_t koff = h * 16u + (lane & 3u) * 4u;
                    const uint32_t a = *(const uint32_t *)(aqs + j * 32u + koff);
                    const uint32_t gw = (gw8[(j >> 1u) * 2u + h] >> shift) & 0x0f0f0f0fu;
                    const uint32_t uw = (uw8[(j >> 1u) * 2u + h] >> shift) & 0x0f0f0f0fu;
                    mma_m8n8k16_s8(gc0, gc1, a, gw);
                    mma_m8n8k16_s8(uc0, uc1, a, uw);
                }
                /* integer scale application for this thread's two columns */
                uint8_t sc, m;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&ghdr0.y, &sc, &m);
                gi0 += (int)sc * gc0;
                const int bs = (int)sxq[mtok][b].bsums[2u * j] + (int)sxq[mtok][b].bsums[2u * j + 1u];
                gs0 += (int)m * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&ghdr1.y, &sc, &m);
                gi1 += (int)sc * gc1;
                gs1 += (int)m * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&uhdr0.y, &sc, &m);
                ui0 += (int)sc * uc0;
                us0 += (int)m * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&uhdr1.y, &sc, &m);
                ui1 += (int)sc * uc1;
                us1 += (int)m * bs;
            }
            /* float finish, exact dev_dot_q4_K_q8_K_block8 expression */
            const float yd = sxq[mtok][b].d;
            const uint32_t sl = b & 7u;
            sg0[sl] += yd * dev_f16_to_f32((uint16_t)(ghdr0.x & 0xffffu)) * (float)gi0 -
                       yd * dev_f16_to_f32((uint16_t)(ghdr0.x >> 16u)) * (float)gs0;
            sg1[sl] += yd * dev_f16_to_f32((uint16_t)(ghdr1.x & 0xffffu)) * (float)gi1 -
                       yd * dev_f16_to_f32((uint16_t)(ghdr1.x >> 16u)) * (float)gs1;
            su0[sl] += yd * dev_f16_to_f32((uint16_t)(uhdr0.x & 0xffffu)) * (float)ui0 -
                       yd * dev_f16_to_f32((uint16_t)(uhdr0.x >> 16u)) * (float)us0;
            su1[sl] += yd * dev_f16_to_f32((uint16_t)(uhdr1.x & 0xffffu)) * (float)ui1 -
                       yd * dev_f16_to_f32((uint16_t)(uhdr1.x >> 16u)) * (float)us1;
        }
        /* quarter_warp_sum_f32 order: ((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7)) */
        const uint32_t p = mtok;
        if (p < np) {
            const uint32_t rowa = row0 + n0;
            const uint32_t rowb = row0 + n0 + 1u;
            float gate2[2], up2[2];
            {
                float a0 = sg0[0] + sg0[4], a1 = sg0[1] + sg0[5], a2 = sg0[2] + sg0[6], a3 = sg0[3] + sg0[7];
                gate2[0] = (a0 + a2) + (a1 + a3);
                a0 = sg1[0] + sg1[4]; a1 = sg1[1] + sg1[5]; a2 = sg1[2] + sg1[6]; a3 = sg1[3] + sg1[7];
                gate2[1] = (a0 + a2) + (a1 + a3);
                a0 = su0[0] + su0[4]; a1 = su0[1] + su0[5]; a2 = su0[2] + su0[6]; a3 = su0[3] + su0[7];
                up2[0] = (a0 + a2) + (a1 + a3);
                a0 = su1[0] + su1[4]; a1 = su1[1] + su1[5]; a2 = su1[2] + su1[6]; a3 = su1[3] + su1[7];
                up2[1] = (a0 + a2) + (a1 + a3);
            }
#pragma unroll
            for (uint32_t e = 0; e < 2u; e++) {
                const uint32_t row = e ? rowb : rowa;
                if (row >= expert_mid_dim) continue;
                float gate = gate2[e];
                float up = up2[e];
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
                mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)s_tok[p] * n_expert + s_slot[p]];
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_tile8_mma_kernel(
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
    __shared__ cuda_block_q8_K sxq[8][8];
    __shared__ uint32_t s_pair[8];
    __shared__ uint32_t s_np;
    if (threadIdx.x == 0) {
        uint32_t np = 0;
        for (; np < 8u; np++) {
            uint32_t local_pair = local_start + np;
            if (local_pair >= counts[expert]) break;
            s_pair[np] = sorted_pairs[offsets[expert] + local_pair];
        }
        s_np = np;
    }
    __syncthreads();
    const uint32_t np = s_np;
    if (midq_blocks <= 8u) {
        const uint32_t words_per_tok = midq_blocks * (uint32_t)(sizeof(cuda_block_q8_K) / 4u);
        for (uint32_t i = threadIdx.x; i < np * words_per_tok; i += blockDim.x) {
            uint32_t p = i / words_per_tok;
            uint32_t w = i - p * words_per_tok;
            ((uint32_t *)sxq[p])[w] = ((const uint32_t *)(midq + (uint64_t)s_pair[p] * midq_blocks))[w];
        }
        __syncthreads();
    }
    const uint32_t mtok = lane >> 2u;
    const uint32_t n0 = (lane & 3u) * 2u;
    for (uint32_t rr = 0; rr < ROW_SPAN / 64u; rr++) {
        const uint32_t row0 = blockIdx.x * ROW_SPAN + rr * 64u + warp * 8u;
        if (row0 >= out_dim) continue;
        const char *wrow = down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row0 * down_row_bytes;
        float s0[8] = {0,0,0,0,0,0,0,0}, s1[8] = {0,0,0,0,0,0,0,0};
        for (uint32_t b = 0; b < midq_blocks; b++) {
            const uint4 hdr0 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)n0 * down_row_bytes) + b);
            const uint4 hdr1 = *(const uint4 *)((const cuda_block_q4_K *)(wrow + (uint64_t)(n0 + 1u) * down_row_bytes) + b);
            const uint32_t *wqw = (const uint32_t *)(((const cuda_block_q4_K *)(wrow + (uint64_t)(lane >> 2u) * down_row_bytes) + b)->qs);
            const int8_t *aqs = sxq[mtok][b].qs;
            uint32_t w8[8];
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) w8[k] = wqw[k * 4u + (lane & 3u)];
            int i0 = 0, i1 = 0, m0 = 0, m1 = 0;
#pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const int shift = (j & 1u) ? 4 : 0;
                int32_t c0 = 0, c1 = 0;
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    const uint32_t koff = h * 16u + (lane & 3u) * 4u;
                    const uint32_t a = *(const uint32_t *)(aqs + j * 32u + koff);
                    const uint32_t w = (w8[(j >> 1u) * 2u + h] >> shift) & 0x0f0f0f0fu;
                    mma_m8n8k16_s8(c0, c1, a, w);
                }
                uint8_t sc, m;
                const int bs = (int)sxq[mtok][b].bsums[2u * j] + (int)sxq[mtok][b].bsums[2u * j + 1u];
                dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr0.y, &sc, &m);
                i0 += (int)sc * c0;
                m0 += (int)m * bs;
                dev_q4_K_get_scale_min(j, (const uint8_t *)&hdr1.y, &sc, &m);
                i1 += (int)sc * c1;
                m1 += (int)m * bs;
            }
            const float yd = sxq[mtok][b].d;
            const uint32_t sl = b & 7u;
            s0[sl] += yd * dev_f16_to_f32((uint16_t)(hdr0.x & 0xffffu)) * (float)i0 -
                      yd * dev_f16_to_f32((uint16_t)(hdr0.x >> 16u)) * (float)m0;
            s1[sl] += yd * dev_f16_to_f32((uint16_t)(hdr1.x & 0xffffu)) * (float)i1 -
                      yd * dev_f16_to_f32((uint16_t)(hdr1.x >> 16u)) * (float)m1;
        }
        const uint32_t p = mtok;
        if (p < np) {
            float a0 = s0[0] + s0[4], a1 = s0[1] + s0[5], a2 = s0[2] + s0[6], a3 = s0[3] + s0[7];
            const float r0 = (a0 + a2) + (a1 + a3);
            a0 = s1[0] + s1[4]; a1 = s1[1] + s1[5]; a2 = s1[2] + s1[6]; a3 = s1[3] + s1[7];
            const float r1 = (a0 + a2) + (a1 + a3);
            if (row0 + n0 < out_dim) down_out[(uint64_t)s_pair[p] * out_dim + row0 + n0] = r0;
            if (row0 + n0 + 1u < out_dim) down_out[(uint64_t)s_pair[p] * out_dim + row0 + n0 + 1u] = r1;
        }
    }
}

/* 16-pair MoE expert tile kernels on sm_80+ m16n8k32 INT8 tensor cores.
 *
 * Same per-output math and reduction order as the 8-pair expert tile
 * kernels (slot[b & 7] += term_b with b ascending, then the exact
 * quarter_warp_sum_f32 grouping), so results are bit-identical; grouping
 * 16 pairs per tile just halves how often each expert's weights are
 * streamed from DRAM. Gate and up run as two passes over the superblocks
 * to keep register pressure at the 8-pair kernel's level. */

__device__ __forceinline__ static void mma16_m16n8k32_s8(
        int32_t &c0, int32_t &c1, int32_t &c2, int32_t &c3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
                 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#else
    (void)a0;(void)a1;(void)a2;(void)a3;(void)b0;(void)b1;(void)c0;(void)c1;(void)c2;(void)c3;
#endif
}

