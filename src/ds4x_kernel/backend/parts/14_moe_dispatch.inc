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
typedef struct ds4_gpu_stream_expert_table {
    const void *model_map;
    uint64_t    model_size;
    uint32_t    layer;
    uint32_t    n_total_expert;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;
} ds4_gpu_stream_expert_table;

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

__device__ __forceinline__ static float glm_rope_yarn_corr_factor_dev(
        int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2.0f * (float)M_PI)) /
           (2.0f * logf(base));
}
__device__ __forceinline__ static float glm_rope_yarn_ramp_dev(
        float low, float high, int i0) {
    const float y = (i0 / 2 - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}
__device__ __forceinline__ static void glm_rope_yarn_dev(
        float theta_extrap, float freq_scale, const float corr_dims[2],
        int i0, float ext_factor, float mscale,
        float *cos_theta, float *sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = glm_rope_yarn_ramp_dev(corr_dims[0], corr_dims[1], i0) *
                         ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static int cuda_current_tier(void) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    return dev;
}
