#ifndef DS4_NO_GPU
/* Batch the routed part of one decode layer across independent sessions.  The
 * router and KV-bearing attention path have already run in each session.  We
 * gather only normalized rows and routing metadata, execute the existing
 * expert-sorted multi-row kernels on both EP owners, then reduce each row in
 * the exact decode slot grouping before scattering it back. */
static bool metal_graph_encode_routed_session_batch(
        ds4_decode_item *items,
        int count,
        const ds4_model *model,
        const ds4_layer_weights *layer,
        uint32_t il,
        uint32_t row_base,
        bool gather_rows,
        bool scatter_rows) {
    if (!items || count < 2 || !model || !layer ||
        DS4_N_EXPERT_USED != 6u) {
        return false;
    }

    ds4_gpu_graph *g = &items[0].session->graph;
    const int home_tier = g->active_tier;
    const int partner_tier = g->cuda_tp_decode
        ? metal_graph_cuda_tp_partner_tier(home_tier) : -1;
    if (!g->cuda_tp_ep || partner_tier < 0 ||
        (uint64_t)row_base + (uint32_t)count > g->prefill_cap ||
        !g_gpu_peer_ok[partner_tier][home_tier]) {
        return false;
    }

    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    const uint64_t norm_bytes = (uint64_t)count * DS4_N_EMBD * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)count * DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t weights_bytes =
        (uint64_t)count * DS4_N_EXPERT_USED * sizeof(float);
    const uint64_t slots_bytes =
        (uint64_t)count * DS4_N_EXPERT_USED * routed_out_dim * sizeof(float);
    const uint64_t output_bytes =
        (uint64_t)count * routed_out_dim * sizeof(float);
    const uint64_t norm_offset =
        (uint64_t)row_base * DS4_N_EMBD * sizeof(float);
    const uint64_t selected_offset =
        (uint64_t)row_base * DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t weights_offset =
        (uint64_t)row_base * DS4_N_EXPERT_USED * sizeof(float);
    const uint64_t slots_offset =
        (uint64_t)row_base * DS4_N_EXPERT_USED * routed_out_dim *
        sizeof(float);
    const uint64_t output_offset =
        (uint64_t)row_base * routed_out_dim * sizeof(float);

    ds4_gpu_tensor local_norm, local_selected, local_weights, original_selected;
    ds4_gpu_tensor local_out, local_down;
    ds4_gpu_tensor peer_norm, peer_selected, peer_weights, peer_out, peer_down;
    bool ok =
        metal_graph_borrow_tensor_view(&local_norm,
                metal_graph_batch_ffn_norm(g), norm_offset, norm_bytes) &&
        metal_graph_borrow_tensor_view(&local_selected,
                metal_graph_batch_router_selected(g), selected_offset,
                selected_bytes) &&
        metal_graph_borrow_tensor_view(&local_weights,
                metal_graph_batch_router_weights(g), weights_offset,
                weights_bytes) &&
        metal_graph_borrow_tensor_view(&original_selected,
                metal_graph_batch_router_logits(g), selected_offset,
                selected_bytes) &&
        metal_graph_borrow_tensor_view(&local_out,
                metal_graph_batch_routed_out(g), output_offset, output_bytes) &&
        metal_graph_borrow_tensor_view(&local_down,
                metal_graph_batch_routed_down(g), slots_offset, slots_bytes) &&
        metal_graph_borrow_tensor_view(&peer_norm,
                g->batch_ffn_norm_by_tier[partner_tier], norm_offset,
                norm_bytes) &&
        metal_graph_borrow_tensor_view(&peer_selected,
                g->batch_router_selected_by_tier[partner_tier],
                selected_offset, selected_bytes) &&
        metal_graph_borrow_tensor_view(&peer_weights,
                g->batch_router_weights_by_tier[partner_tier], weights_offset,
                weights_bytes) &&
        metal_graph_borrow_tensor_view(&peer_out,
                g->batch_routed_out_by_tier[partner_tier], output_offset,
                output_bytes) &&
        metal_graph_borrow_tensor_view(&peer_down,
                g->batch_routed_down_by_tier[partner_tier], slots_offset,
                slots_bytes);

    for (int i = 0; gather_rows && ok && i < count; i++) {
        ds4_gpu_graph *srcg = &items[i].session->graph;
        if (srcg->active_tier != home_tier) {
            ok = false;
            break;
        }
        ds4_gpu_tensor norm_row, weights_row, original_row;
        ok = metal_graph_borrow_tensor_view(
                     &norm_row, &local_norm,
                     (uint64_t)i * DS4_N_EMBD * sizeof(float),
                     (uint64_t)DS4_N_EMBD * sizeof(float)) &&
             metal_graph_borrow_tensor_view(
                     &weights_row, &local_weights,
                     (uint64_t)i * DS4_N_EXPERT_USED * sizeof(float),
                     (uint64_t)DS4_N_EXPERT_USED * sizeof(float)) &&
             metal_graph_borrow_tensor_view(
                     &original_row, &original_selected,
                     (uint64_t)i * DS4_N_EXPERT_USED * sizeof(int32_t),
                     (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
        if (ok) {
            ok = ds4_gpu_tensor_copy_xdev3_default_dst(
                    &norm_row, metal_graph_ffn_norm(srcg), norm_row.bytes,
                    &original_row, metal_graph_router_selected(srcg), original_row.bytes,
                    &weights_row, metal_graph_router_weights(srcg), weights_row.bytes) != 0;
        }
    }
    if (ok && gather_rows) {
        ok = ds4_gpu_tensor_copy_xdev_default(
                &local_selected, &original_selected, selected_bytes) != 0;
    } else if (ok) {
        ok = ds4_gpu_tensor_copy_xdev_default(
                &original_selected, &local_selected, selected_bytes) != 0;
    }

    if (ok) {
        ok = ds4_gpu_tensor_copy_xdev3_default_dst(
                &peer_norm, &local_norm, norm_bytes,
                &peer_selected, &local_selected, selected_bytes,
                &peer_weights, &local_weights, weights_bytes) != 0;
    }

    bool peer_mid_is_f16 = false;
    if (ok) ok = ds4_gpu_set_current_device(partner_tier) == 0;
    if (ok) {
        ok = ds4_gpu_routed_moe_batch_owned_tensor(
                &peer_out,
                g->batch_routed_gate_by_tier[partner_tier],
                g->batch_routed_up_by_tier[partner_tier],
                g->batch_routed_mid_by_tier[partner_tier],
                &peer_down,
                model->map, model->size,
                layer->ffn_gate_exps->abs_offset,
                layer->ffn_up_exps->abs_offset,
                layer->ffn_down_exps->abs_offset,
                layer->ffn_gate_exps->type,
                layer->ffn_down_exps->type,
                gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes,
                (uint32_t)expert_in_dim,
                (uint32_t)down_in_dim,
                (uint32_t)routed_out_dim,
                &peer_selected, &peer_weights,
                DS4_N_EXPERT, DS4_N_EXPERT_USED,
                DS4_N_EXPERT / 2u,
                DS4_N_EXPERT - DS4_N_EXPERT / 2u,
                DS4_SWIGLU_CLAMP_EXP,
                &peer_norm, il, (uint32_t)count,
                &peer_mid_is_f16) != 0;
    }

    if (ds4_gpu_set_current_device(home_tier) != 0) ok = false;
    if (ok) {
        ok = ds4_gpu_routed_moe_batch_owned_tensor(
                &local_out,
                metal_graph_batch_routed_gate(g),
                metal_graph_batch_routed_up(g),
                metal_graph_batch_routed_mid(g),
                &local_down,
                model->map, model->size,
                layer->ffn_gate_exps->abs_offset,
                layer->ffn_up_exps->abs_offset,
                layer->ffn_down_exps->abs_offset,
                layer->ffn_gate_exps->type,
                layer->ffn_down_exps->type,
                gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes,
                (uint32_t)expert_in_dim,
                (uint32_t)down_in_dim,
                (uint32_t)routed_out_dim,
                &local_selected, &local_weights,
                DS4_N_EXPERT, DS4_N_EXPERT_USED,
                0, DS4_N_EXPERT / 2u,
                DS4_SWIGLU_CLAMP_EXP,
                &local_norm, il, (uint32_t)count,
                &g->batch_routed_mid_is_f16) != 0;
    }
    if (ok) {
        ok = ds4_gpu_tensor_wait_xdev_default(&peer_down, home_tier) != 0;
    }

    const bool combine_rows = metal_graph_tp_env_flag(
            "DS4_CUDA_SESSION_BATCH_MOE_COMBINE_ROWS", true);
    if (ok && combine_rows) {
        ok = ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(
                &local_out, &local_down, &peer_down, &original_selected,
                (uint32_t)routed_out_dim, DS4_N_EXPERT / 2u,
                (uint32_t)count) != 0;
    }
    for (int i = 0;
         ok && (!combine_rows || scatter_rows) && i < count;
         i++) {
        const uint64_t slot_stride =
            (uint64_t)DS4_N_EXPERT_USED * routed_out_dim * sizeof(float);
        const uint64_t out_stride = routed_out_dim * sizeof(float);
        ds4_gpu_tensor out_row, home_slots, peer_slots, selected_row;
        ok = metal_graph_borrow_tensor_view(
                     &out_row, &local_out, (uint64_t)i * out_stride, out_stride) &&
             metal_graph_borrow_tensor_view(
                     &home_slots, &local_down,
                     (uint64_t)i * slot_stride, slot_stride) &&
             metal_graph_borrow_tensor_view(
                     &peer_slots, &peer_down,
                     (uint64_t)i * slot_stride, slot_stride) &&
             metal_graph_borrow_tensor_view(
                     &selected_row, &original_selected,
                     (uint64_t)i * DS4_N_EXPERT_USED * sizeof(int32_t),
                     (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
        if (ok && !combine_rows) {
            ok = ds4_gpu_routed_moe_owned_slots_combine_tensor(
                    &out_row, &home_slots, &peer_slots, &selected_row,
                    (uint32_t)routed_out_dim, DS4_N_EXPERT / 2u) != 0;
        }
        if (ok && scatter_rows) {
            ok = ds4_gpu_tensor_copy_xdev_default(
                    metal_graph_routed_out(&items[i].session->graph),
                    &out_row, out_stride) != 0;
        }
    }
    return ok;
}

/* Run one owned-expert dispatch over prefill rows followed by exact decode
 * rows. The expert kernels see the larger combined batch, but reduction is
 * path-specific: prefill keeps the existing two-owner partial sum and decode
 * keeps its fixed six-slot order in the shared row workspace. */
static bool metal_graph_encode_mixed_routed_rows(
        ds4_gpu_graph *g,
        ds4_decode_item *decode_items,
        int decode_count,
        const ds4_model *model,
        const ds4_layer_weights *layer,
        uint32_t il,
        uint32_t prefill_rows) {
    if (!g || decode_count < 0 ||
        (decode_count > 0 && !decode_items) || !model || !layer ||
        prefill_rows == 0 || DS4_N_EXPERT_USED != 6u) {
        return false;
    }

    const uint32_t decode_rows = (uint32_t)decode_count;
    const uint32_t total_rows = prefill_rows + decode_rows;
    const int home_tier = g->active_tier;
    const int partner_tier = g->cuda_tp_decode
        ? metal_graph_cuda_tp_partner_tier(home_tier) : -1;
    if (total_rows < prefill_rows || total_rows > g->prefill_cap ||
        !g->cuda_tp_ep || partner_tier < 0 ||
        !g_gpu_peer_ok[partner_tier][home_tier] ||
        !metal_graph_ensure_batch_ffn_out_on(g, home_tier)) {
        return false;
    }
    for (int i = 0; i < decode_count; i++) {
        if (!decode_items[i].session ||
            decode_items[i].session->graph.active_tier != home_tier) {
            return false;
        }
    }

    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    const uint64_t norm_bytes =
        (uint64_t)total_rows * DS4_N_EMBD * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)total_rows * DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t weights_bytes =
        (uint64_t)total_rows * DS4_N_EXPERT_USED * sizeof(float);
    const uint64_t slots_bytes =
        (uint64_t)total_rows * DS4_N_EXPERT_USED * routed_out_dim * sizeof(float);
    const uint64_t output_bytes =
        (uint64_t)total_rows * routed_out_dim * sizeof(float);
    const uint64_t prefill_output_bytes =
        (uint64_t)prefill_rows * routed_out_dim * sizeof(float);

    ds4_gpu_tensor local_norm, local_selected, local_weights, original_selected;
    ds4_gpu_tensor local_out, local_down, prefill_peer_out;
    ds4_gpu_tensor peer_norm, peer_selected, peer_weights, peer_out, peer_down;
    ds4_gpu_tensor peer_prefill_out;
    bool ok =
        metal_graph_borrow_tensor_view(&local_norm,
                metal_graph_batch_ffn_norm(g), 0, norm_bytes) &&
        metal_graph_borrow_tensor_view(&local_selected,
                metal_graph_batch_router_selected(g), 0, selected_bytes) &&
        metal_graph_borrow_tensor_view(&local_weights,
                metal_graph_batch_router_weights(g), 0, weights_bytes) &&
        metal_graph_borrow_tensor_view(&original_selected,
                metal_graph_batch_router_logits(g), 0, selected_bytes) &&
        metal_graph_borrow_tensor_view(&local_out,
                metal_graph_batch_routed_out(g), 0, output_bytes) &&
        metal_graph_borrow_tensor_view(&local_down,
                metal_graph_batch_routed_down(g), 0, slots_bytes) &&
        metal_graph_borrow_tensor_view(&prefill_peer_out,
                g->batch_ffn_out_by_tier[home_tier], 0,
                prefill_output_bytes) &&
        metal_graph_borrow_tensor_view(&peer_norm,
                g->batch_ffn_norm_by_tier[partner_tier], 0, norm_bytes) &&
        metal_graph_borrow_tensor_view(&peer_selected,
                g->batch_router_selected_by_tier[partner_tier], 0,
                selected_bytes) &&
        metal_graph_borrow_tensor_view(&peer_weights,
                g->batch_router_weights_by_tier[partner_tier], 0,
                weights_bytes) &&
        metal_graph_borrow_tensor_view(&peer_out,
                g->batch_routed_out_by_tier[partner_tier], 0, output_bytes) &&
        metal_graph_borrow_tensor_view(&peer_down,
                g->batch_routed_down_by_tier[partner_tier], 0, slots_bytes) &&
        metal_graph_borrow_tensor_view(&peer_prefill_out,
                &peer_out, 0, prefill_output_bytes);

    if (ok) {
        ok = ds4_gpu_tensor_copy_xdev_default(
                &original_selected, &local_selected, selected_bytes) != 0 &&
             ds4_gpu_tensor_copy_xdev3_default_dst(
                &peer_norm, &local_norm, norm_bytes,
                &peer_selected, &local_selected, selected_bytes,
                &peer_weights, &local_weights, weights_bytes) != 0;
    }

    bool peer_mid_is_f16 = false;
    if (ok) ok = ds4_gpu_set_current_device(partner_tier) == 0;
    if (ok) {
        ok = ds4_gpu_routed_moe_batch_owned_tensor(
                &peer_out,
                g->batch_routed_gate_by_tier[partner_tier],
                g->batch_routed_up_by_tier[partner_tier],
                g->batch_routed_mid_by_tier[partner_tier],
                &peer_down,
                model->map, model->size,
                layer->ffn_gate_exps->abs_offset,
                layer->ffn_up_exps->abs_offset,
                layer->ffn_down_exps->abs_offset,
                layer->ffn_gate_exps->type,
                layer->ffn_down_exps->type,
                gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes,
                (uint32_t)expert_in_dim,
                (uint32_t)down_in_dim,
                (uint32_t)routed_out_dim,
                &peer_selected, &peer_weights,
                DS4_N_EXPERT, DS4_N_EXPERT_USED,
                DS4_N_EXPERT / 2u,
                DS4_N_EXPERT - DS4_N_EXPERT / 2u,
                DS4_SWIGLU_CLAMP_EXP,
                &peer_norm, il, total_rows,
                &peer_mid_is_f16) != 0;
    }

    if (ds4_gpu_set_current_device(home_tier) != 0) ok = false;
    if (ok) {
        ok = ds4_gpu_routed_moe_batch_owned_tensor(
                &local_out,
                metal_graph_batch_routed_gate(g),
                metal_graph_batch_routed_up(g),
                metal_graph_batch_routed_mid(g),
                &local_down,
                model->map, model->size,
                layer->ffn_gate_exps->abs_offset,
                layer->ffn_up_exps->abs_offset,
                layer->ffn_down_exps->abs_offset,
                layer->ffn_gate_exps->type,
                layer->ffn_down_exps->type,
                gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes,
                (uint32_t)expert_in_dim,
                (uint32_t)down_in_dim,
                (uint32_t)routed_out_dim,
                &local_selected, &local_weights,
                DS4_N_EXPERT, DS4_N_EXPERT_USED,
                0, DS4_N_EXPERT / 2u,
                DS4_SWIGLU_CLAMP_EXP,
                &local_norm, il, total_rows,
                &g->batch_routed_mid_is_f16) != 0;
    }

    if (ok) {
        ok = ds4_gpu_tensor_copy_xdev_default(
                &prefill_peer_out, &peer_prefill_out,
                prefill_output_bytes) != 0;
    }
    if (ok) {
        ds4_gpu_tensor prefill_home_out;
        ok = metal_graph_borrow_tensor_view(
                &prefill_home_out, &local_out, 0, prefill_output_bytes) &&
             ds4_gpu_add_tensor(
                &prefill_home_out, &prefill_home_out, &prefill_peer_out,
                (uint32_t)((uint64_t)prefill_rows * routed_out_dim)) != 0;
    }
    if (ok && decode_rows > 0) {
        ok = ds4_gpu_tensor_wait_xdev_default(&peer_down, home_tier) != 0;
    }

    const uint64_t slot_stride =
        (uint64_t)DS4_N_EXPERT_USED * routed_out_dim * sizeof(float);
    const uint64_t out_stride = routed_out_dim * sizeof(float);
    const bool combine_rows = metal_graph_tp_env_flag(
            "DS4_CUDA_SESSION_BATCH_MOE_COMBINE_ROWS", true);
    const bool scatter_rows = metal_graph_tp_env_flag(
            "DS4_CUDA_MIXED_ROUTED_SCATTER", false);
    if (ok && combine_rows && decode_rows > 0) {
        ds4_gpu_tensor decode_out, home_slots, peer_slots, selected;
        ok = metal_graph_borrow_tensor_view(
                    &decode_out, &local_out, prefill_output_bytes,
                    (uint64_t)decode_rows * out_stride) &&
             metal_graph_borrow_tensor_view(
                    &home_slots, &local_down,
                    (uint64_t)prefill_rows * slot_stride,
                    (uint64_t)decode_rows * slot_stride) &&
             metal_graph_borrow_tensor_view(
                    &peer_slots, &peer_down,
                    (uint64_t)prefill_rows * slot_stride,
                    (uint64_t)decode_rows * slot_stride) &&
             metal_graph_borrow_tensor_view(
                    &selected, &original_selected,
                    (uint64_t)prefill_rows * DS4_N_EXPERT_USED *
                        sizeof(int32_t),
                    (uint64_t)decode_rows * DS4_N_EXPERT_USED *
                        sizeof(int32_t));
        if (ok) {
            ok = ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(
                    &decode_out, &home_slots, &peer_slots, &selected,
                    (uint32_t)routed_out_dim, DS4_N_EXPERT / 2u,
                    decode_rows) != 0;
        }
    }
    for (int i = 0;
         ok && (!combine_rows || scatter_rows) && i < decode_count;
         i++) {
        const uint32_t row = prefill_rows + (uint32_t)i;
        ds4_gpu_tensor out_row, home_slots, peer_slots, selected_row;
        ok = metal_graph_borrow_tensor_view(
                     &out_row, &local_out, (uint64_t)row * out_stride,
                     out_stride) &&
             metal_graph_borrow_tensor_view(
                     &home_slots, &local_down, (uint64_t)row * slot_stride,
                     slot_stride) &&
             metal_graph_borrow_tensor_view(
                     &peer_slots, &peer_down, (uint64_t)row * slot_stride,
                     slot_stride) &&
             metal_graph_borrow_tensor_view(
                     &selected_row, &original_selected,
                     (uint64_t)row * DS4_N_EXPERT_USED * sizeof(int32_t),
                     (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
        if (ok && !combine_rows) {
            ok = ds4_gpu_routed_moe_owned_slots_combine_tensor(
                    &out_row, &home_slots, &peer_slots, &selected_row,
                    (uint32_t)routed_out_dim, DS4_N_EXPERT / 2u) != 0;
        }
        if (ok && scatter_rows) {
            ok = ds4_gpu_tensor_copy_xdev_default(
                    metal_graph_routed_out(&decode_items[i].session->graph),
                    &out_row, out_stride) != 0;
        }
    }
    return ok;
}

static bool metal_graph_encode_shared_rows_exact(
        ds4_gpu_tensor *shared_gate,
        ds4_gpu_tensor *shared_up,
        ds4_gpu_tensor *shared_mid,
        ds4_gpu_tensor *shared_out,
        ds4_gpu_tensor *norm,
        const ds4_model *model,
        const ds4_layer_weights *layer,
        uint32_t rows) {
    if (!shared_gate || !shared_up || !shared_mid || !shared_out || !norm ||
        !model || !layer || rows == 0) {
        return false;
    }
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    bool ok = ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
            shared_gate,
            shared_up,
            model->map,
            model->size,
            layer->ffn_gate_shexp->abs_offset,
            layer->ffn_up_shexp->abs_offset,
            DS4_N_EMBD,
            shared_dim,
            shared_dim,
            norm,
            rows) != 0;
    if (ok) {
        ok = ds4_gpu_swiglu_tensor(
                shared_mid,
                shared_gate,
                shared_up,
                (uint32_t)((uint64_t)rows * shared_dim),
                DS4_SWIGLU_CLAMP_EXP,
                1.0f) != 0;
    }
    if (ok) {
        ok = ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                shared_out,
                model->map,
                model->size,
                layer->ffn_down_shexp->abs_offset,
                shared_dim,
                DS4_N_EMBD,
                shared_mid,
                rows) != 0;
    }
    return ok;
}

/* Finish the cache-independent half of one FFN for all session rows together.
 * The routed outputs and normalized inputs already occupy the first graph's
 * shared prefill workspace. Gather only the per-session HC residual/mix,
 * execute the shared expert as exact row batches, and scatter final HC rows. */
static bool metal_graph_encode_shared_session_batch(
        ds4_decode_item *items,
        int count,
        const ds4_model *model,
        const ds4_layer_weights *layer,
        uint32_t row_base,
        bool gather_hc_rows) {
    if (!items || count < 2 || !model || !layer) return false;

    ds4_gpu_graph *g = &items[0].session->graph;
    const int home_tier = g->active_tier;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC +
                            (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t norm_bytes =
        (uint64_t)count * DS4_N_EMBD * sizeof(float);
    const uint64_t shared_bytes =
        (uint64_t)count * shared_dim * sizeof(float);
    const uint64_t hc_bytes = (uint64_t)count * hc_dim * sizeof(float);
    const uint64_t split_bytes = (uint64_t)count * mix_hc * sizeof(float);

    ds4_gpu_tensor norm, routed, shared_gate, shared_up, shared_mid, shared_out;
    ds4_gpu_tensor residual_hc, split, next_hc;
    bool ok =
        home_tier >= 0 &&
        (uint64_t)row_base + (uint32_t)count <= g->prefill_cap &&
        metal_graph_borrow_tensor_view(&norm,
                metal_graph_batch_ffn_norm(g),
                (uint64_t)row_base * DS4_N_EMBD * sizeof(float), norm_bytes) &&
        metal_graph_borrow_tensor_view(&routed,
                metal_graph_batch_routed_out(g),
                (uint64_t)row_base * DS4_N_EMBD * sizeof(float), norm_bytes) &&
        metal_graph_borrow_tensor_view(&shared_gate,
                metal_graph_batch_shared_gate(g),
                (uint64_t)row_base * shared_dim * sizeof(float), shared_bytes) &&
        metal_graph_borrow_tensor_view(&shared_up,
                metal_graph_batch_shared_up(g),
                (uint64_t)row_base * shared_dim * sizeof(float), shared_bytes) &&
        metal_graph_borrow_tensor_view(&shared_mid,
                metal_graph_batch_shared_mid(g),
                (uint64_t)row_base * shared_dim * sizeof(float), shared_bytes) &&
        metal_graph_borrow_tensor_view(&shared_out,
                metal_graph_batch_shared_out(g),
                (uint64_t)row_base * DS4_N_EMBD * sizeof(float), norm_bytes) &&
        metal_graph_borrow_tensor_view(&residual_hc,
                metal_graph_batch_after_attn_hc(g),
                (uint64_t)row_base * hc_dim * sizeof(float), hc_bytes) &&
        metal_graph_borrow_tensor_view(&split,
                metal_graph_batch_hc_split(g),
                (uint64_t)row_base * mix_hc * sizeof(float), split_bytes) &&
        metal_graph_borrow_tensor_view(&next_hc,
                metal_graph_batch_next_hc(g),
                (uint64_t)row_base * hc_dim * sizeof(float), hc_bytes);

    for (int i = 0; gather_hc_rows && ok && i < count; i++) {
        ds4_gpu_graph *srcg = &items[i].session->graph;
        ds4_gpu_tensor residual_row, split_row;
        ok = srcg->active_tier == home_tier &&
             metal_graph_borrow_tensor_view(
                     &residual_row, &residual_hc,
                     (uint64_t)i * hc_dim * sizeof(float),
                     hc_dim * sizeof(float)) &&
             metal_graph_borrow_tensor_view(
                     &split_row, &split,
                     (uint64_t)i * mix_hc * sizeof(float),
                     mix_hc * sizeof(float));
        if (ok) {
            ok = ds4_gpu_tensor_copy_xdev3_default_dst(
                    &residual_row, metal_graph_after_attn_hc(srcg),
                    residual_row.bytes,
                    &split_row, metal_graph_hc_split(srcg), split_row.bytes,
                    NULL, NULL, 0) != 0;
        }
    }

    if (ok) {
        ok = metal_graph_encode_shared_rows_exact(
                &shared_gate, &shared_up, &shared_mid, &shared_out, &norm,
                model, layer, (uint32_t)count);
    }
    if (ok) {
        ok = ds4_gpu_hc_expand_add_split_tensor(
                &next_hc,
                &routed,
                &shared_out,
                &residual_hc,
                &split,
                DS4_N_EMBD,
                DS4_N_HC) != 0;
    }

    for (int i = 0; ok && i < count; i++) {
        ds4_gpu_tensor next_row;
        ok = metal_graph_borrow_tensor_view(
                &next_row, &next_hc,
                (uint64_t)i * hc_dim * sizeof(float),
                hc_dim * sizeof(float));
        if (ok) {
            ok = ds4_gpu_tensor_copy_xdev_default(
                    metal_graph_after_ffn_hc(&items[i].session->graph),
                    &next_row,
                    next_row.bytes) != 0;
        }
    }
    return ok;
}

/* Submit independent decode graphs stage by stage.  A whole-graph ordering
 * puts session A's all-device output head ahead of session B's first stage,
 * which serializes the nominal pipeline.  Stage-major submission leaves each
 * session's kernel arithmetic untouched while allowing B's previous stage to
 * overlap A's next stage. */
static bool metal_graph_encode_session_pipeline_batch(
        ds4_decode_item *items,
        int count,
        const ds4_model *model,
        const ds4_weights *weights) {
    if (!items || count <= 0 || !model || !weights) return false;
    ds4_gpu_graph *first = &items[0].session->graph;
    if (!first->placement) return false;

    const bool group_routed_moe =
        metal_graph_tp_env_flag("DS4_CUDA_SESSION_BATCH_MOE", true) &&
        metal_graph_session_batch_moe_supported(items, count, weights);
    const bool group_shared_ffn =
        group_routed_moe &&
        metal_graph_tp_env_flag("DS4_CUDA_SESSION_BATCH_SHARED", true) &&
        metal_graph_session_batch_shared_supported(items, count, weights);
    const bool group_ffn_pre =
        group_shared_ffn &&
        metal_graph_tp_env_flag("DS4_CUDA_SESSION_BATCH_FFN_PRE", true) &&
        metal_graph_session_batch_ffn_pre_supported(items, count, weights);
    const bool group_attn_pre =
        group_ffn_pre &&
        metal_graph_tp_env_flag("DS4_CUDA_SESSION_BATCH_ATTN_PRE", true) &&
        metal_graph_session_batch_attn_pre_supported(items, count, weights);
    const bool alias_attn_pre =
        group_attn_pre &&
        metal_graph_tp_env_flag("DS4_CUDA_SESSION_BATCH_ATTN_ALIAS", true);
    const bool group_attn_core =
        alias_attn_pre &&
        metal_graph_tp_env_flag("DS4_CUDA_SESSION_BATCH_ATTN_CORE", true) &&
        metal_graph_session_batch_attn_core_supported(items, count, weights);
    const bool group_qkv =
        group_attn_core &&
        metal_graph_tp_env_flag("DS4_CUDA_SESSION_BATCH_QKV", true) &&
        metal_graph_session_batch_qkv_supported(items, count, weights);
    const bool group_kv_store =
        group_qkv &&
        metal_graph_tp_env_flag("DS4_CUDA_SESSION_BATCH_KV_STORE", true) &&
        metal_graph_session_batch_kv_store_supported(items, count);
    const bool group_attn_post =
        group_attn_core &&
        metal_graph_tp_env_flag("DS4_CUDA_SESSION_BATCH_ATTN_POST", true) &&
        metal_graph_session_batch_attn_post_supported(items, count, weights);
    bool ok = true;
    if (group_ffn_pre) {
        int32_t *router_tokens = malloc((size_t)count * sizeof(*router_tokens));
        if (!router_tokens) return false;
        for (int i = 0; i < count; i++) router_tokens[i] = items[i].token;
        ok = ds4_gpu_tensor_write(
                metal_graph_prefill_tokens(first), 0, router_tokens,
                (uint64_t)count * sizeof(*router_tokens)) != 0;
        free(router_tokens);
    }
    for (int i = 0; ok && i < count; i++) {
        ds4_session *s = items[i].session;
        ds4_gpu_graph *g = &s->graph;
        if (!g->placement || g->raw_cap == 0) return false;
        metal_graph_dspark_capture_begin(g);
        if (!metal_graph_set_active_tier_decode(g, g->emb_tier)) return false;
        ok = ds4_gpu_embed_token_hc_tensor(
                 metal_graph_cur_hc(g),
                 model->map,
                 model->size,
                 weights->token_embd->abs_offset,
                 (uint32_t)weights->token_embd->dim[1],
                 (uint32_t)items[i].token,
                 DS4_N_EMBD,
                 DS4_N_HC) != 0;
    }

    uint32_t stage_begin = 0;
    while (ok && stage_begin < DS4_N_LAYER) {
        const int stage_tier = first->placement[stage_begin + 1u];
        uint32_t stage_end = stage_begin + 1u;
        while (stage_end < DS4_N_LAYER &&
               first->placement[stage_end + 1u] == stage_tier) {
            stage_end++;
        }

        if (group_routed_moe) {
            for (uint32_t il = stage_begin; ok && il < stage_end; il++) {
                if (group_attn_pre) {
                    ok = metal_graph_encode_attn_pre_session_batch(
                            items, count, model, &weights->layer[il], il,
                            0,
                            !alias_attn_pre);
                }
                if (ok && group_qkv) {
                    ok = metal_graph_encode_qkv_session_batch(
                            items, count, model, &weights->layer[il], il, 0,
                            group_kv_store);
                }
                for (int i = 0; ok && i < count; i++) {
                    ds4_session *s = items[i].session;
                    ds4_gpu_graph *g = &s->graph;
                    const uint32_t pos = (uint32_t)s->checkpoint.len;
                    const uint32_t raw_row = pos % g->raw_cap;
                    const uint32_t n_raw =
                        metal_graph_raw_span_for_batch(g, pos, 1);
                    if (g->placement[il + 1u] != stage_tier) {
                        ok = false;
                        break;
                    }
                    metal_graph_attn_pre_alias attn_alias = {0};
                    if (alias_attn_pre) {
                        ok = metal_graph_bind_attn_pre_batch_row(
                                g, first, &weights->layer[il], (uint32_t)i,
                                group_attn_core, &attn_alias);
                    }
                    if (ok) {
                        ok = metal_graph_encode_decode_layer_phase(
                                g,
                                model,
                                &weights->layer[il],
                                il,
                                pos,
                                g->layer_raw_cache[il],
                                g->raw_cap,
                                raw_row,
                                n_raw,
                                items[i].token,
                                group_kv_store ?
                                    METAL_DECODE_LAYER_FROM_KV_STORE_TO_ATTN :
                                group_qkv ?
                                    METAL_DECODE_LAYER_FROM_QKV_TO_ATTN :
                                group_attn_core ?
                                    METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_ATTN :
                                group_attn_pre ?
                                    METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_FFN :
                                group_ffn_pre ? METAL_DECODE_LAYER_TO_FFN :
                                                METAL_DECODE_LAYER_TO_ROUTER);
                    }
                    metal_graph_unbind_attn_pre_batch_row(g, &attn_alias);
                }
                if (ok && group_attn_core) {
                    ok = metal_graph_encode_attention_session_batch(
                            items, count, model, &weights->layer[il], il, 0);
                }
                if (ok && group_attn_post) {
                    ok = metal_graph_encode_attn_post_session_batch(
                            items, count, model, &weights->layer[il], il, 0);
                }
                for (int i = 0;
                     ok && group_attn_core && !group_attn_post && i < count;
                     i++) {
                    ds4_gpu_graph *g = &items[i].session->graph;
                    const uint32_t pos =
                        (uint32_t)items[i].session->checkpoint.len;
                    const uint32_t raw_row = pos % g->raw_cap;
                    const uint32_t n_raw =
                        metal_graph_raw_span_for_batch(g, pos, 1);
                    metal_graph_attn_pre_alias attn_alias = {0};
                    ok = metal_graph_bind_attn_pre_batch_row(
                            g, first, &weights->layer[il], (uint32_t)i,
                            true, &attn_alias);
                    if (ok) {
                        ok = metal_graph_encode_decode_layer_phase(
                                g,
                                model,
                                &weights->layer[il],
                                il,
                                pos,
                                g->layer_raw_cache[il],
                                g->raw_cap,
                                raw_row,
                                n_raw,
                                items[i].token,
                                METAL_DECODE_LAYER_FROM_ATTN_TO_FFN);
                    }
                    metal_graph_unbind_attn_pre_batch_row(g, &attn_alias);
                }
                if (ok && group_ffn_pre) {
                    ok = metal_graph_encode_ffn_pre_session_batch(
                            items, count, model, &weights->layer[il],
                            0,
                            !group_attn_core);
                }
                if (ok) {
                    ok = metal_graph_encode_routed_session_batch(
                            items, count, model, &weights->layer[il], il,
                            0,
                            !group_ffn_pre,
                            !group_shared_ffn);
                }
                if (ok && group_shared_ffn) {
                    ok = metal_graph_encode_shared_session_batch(
                            items, count, model, &weights->layer[il],
                            0,
                            !group_ffn_pre);
                }
                for (int i = 0; ok && i < count; i++) {
                    ds4_session *s = items[i].session;
                    ds4_gpu_graph *g = &s->graph;
                    if (!group_shared_ffn) {
                        const uint32_t pos = (uint32_t)s->checkpoint.len;
                        const uint32_t raw_row = pos % g->raw_cap;
                        const uint32_t n_raw =
                            metal_graph_raw_span_for_batch(g, pos, 1);
                        ok = metal_graph_encode_decode_layer_phase(
                                g,
                                model,
                                &weights->layer[il],
                                il,
                                pos,
                                g->layer_raw_cache[il],
                                g->raw_cap,
                                raw_row,
                                n_raw,
                                items[i].token,
                                METAL_DECODE_LAYER_FROM_ROUTER);
                    }
                    if (ok) {
                        ds4_gpu_tensor *tmp = metal_graph_cur_hc(g);
                        g->cur_hc_by_tier[g->active_tier] =
                            metal_graph_after_ffn_hc(g);
                        g->after_ffn_hc_by_tier[g->active_tier] = tmp;
                        ok = metal_graph_dspark_capture_decode_layer(g, il);
                    }
                }
            }
        } else {
            for (int i = 0; ok && i < count; i++) {
                ds4_session *s = items[i].session;
                ds4_gpu_graph *g = &s->graph;
                const uint32_t pos = (uint32_t)s->checkpoint.len;
                const uint32_t raw_row = pos % g->raw_cap;
                const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
                if (g->placement[stage_begin + 1u] != stage_tier) {
                    ok = false;
                    break;
                }
                for (uint32_t il = stage_begin; ok && il < stage_end; il++) {
                    if (g->placement[il + 1u] != stage_tier) {
                        ok = false;
                        break;
                    }
                    ok = metal_graph_encode_decode_layer(
                            g,
                            model,
                            &weights->layer[il],
                            il,
                            pos,
                            g->layer_raw_cache[il],
                            g->raw_cap,
                            raw_row,
                            n_raw,
                            items[i].token);
                    if (ok) {
                        ds4_gpu_tensor *tmp = metal_graph_cur_hc(g);
                        g->cur_hc_by_tier[g->active_tier] =
                            metal_graph_after_ffn_hc(g);
                        g->after_ffn_hc_by_tier[g->active_tier] = tmp;
                        ok = metal_graph_dspark_capture_decode_layer(g, il);
                    }
                }
            }
        }
        stage_begin = stage_end;
    }

    for (int i = 0; ok && i < count; i++) {
        ds4_session *s = items[i].session;
        ok = metal_graph_encode_output_head(&s->graph, model, weights,
                                            weights->output->dim[1]);
    }
    return ok;
}

static DS4_MAYBE_UNUSED bool metal_graph_mixed_workspace_compatible(
        const ds4_gpu_graph *owner,
        const ds4_gpu_graph *member) {
    if (!owner || !member ||
        metal_graph_prefill_tokens(owner) != metal_graph_prefill_tokens(member)) {
        return false;
    }
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        ds4_gpu_tensor *owner_cur = owner->batch_cur_hc_by_tier[t];
        ds4_gpu_tensor *owner_next = owner->batch_next_hc_by_tier[t];
        ds4_gpu_tensor *member_cur = member->batch_cur_hc_by_tier[t];
        ds4_gpu_tensor *member_next = member->batch_next_hc_by_tier[t];
        const bool same_orientation =
            owner_cur == member_cur && owner_next == member_next;
        const bool opposite_orientation =
            owner_cur == member_next && owner_next == member_cur;
        if ((!same_orientation && !opposite_orientation) ||
            owner->batch_after_attn_hc_by_tier[t] !=
                member->batch_after_attn_hc_by_tier[t] ||
            owner->batch_ffn_norm_by_tier[t] !=
                member->batch_ffn_norm_by_tier[t] ||
            owner->batch_router_selected_by_tier[t] !=
                member->batch_router_selected_by_tier[t] ||
            owner->batch_routed_down_by_tier[t] !=
                member->batch_routed_down_by_tier[t] ||
            owner->batch_shared_out_by_tier[t] !=
                member->batch_shared_out_by_tier[t]) {
            return false;
        }
    }
    return true;
}

static void metal_graph_align_mixed_workspace(
        const ds4_gpu_graph *owner,
        ds4_gpu_graph *member) {
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        member->batch_cur_hc_by_tier[t] = owner->batch_cur_hc_by_tier[t];
        member->batch_next_hc_by_tier[t] = owner->batch_next_hc_by_tier[t];
    }
}

/* The Q4 model-backed oracle is exact through 512 prefill rows on the
 * layer-interleaved path. Larger waves currently change prefill logits and
 * must use the serialized public-API fallback. */
static DS4_MAYBE_UNUSED uint32_t metal_graph_mixed_routed_max_prefill_rows(void) {
    const char *value = getenv("DS4_CUDA_MIXED_ROUTED_MAX_PREFILL");
    if (!value || !value[0]) return 512u;
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed > UINT32_MAX) return 512u;
    return (uint32_t)parsed;
}

static bool metal_graph_mixed_prefill_decode_supported(
        ds4_session *prefill_session,
        const token_vec *prompt,
        uint32_t start,
        uint32_t prefill_rows,
        ds4_decode_item *decode_items,
        int decode_count,
        const ds4_weights *weights) {
#if defined(__APPLE__)
    (void)prefill_session;
    (void)prompt;
    (void)start;
    (void)prefill_rows;
    (void)decode_items;
    (void)decode_count;
    (void)weights;
    return false;
#else
    if (!prefill_session || !prompt || !decode_items || decode_count < 3 ||
        !weights || prefill_rows < metal_graph_resume_prefill_min_tokens() ||
        prefill_rows > metal_graph_mixed_routed_max_prefill_rows() ||
        start > (uint32_t)prompt->len ||
        prefill_rows > (uint32_t)prompt->len - start) {
        return false;
    }
    ds4_gpu_graph *g = &prefill_session->graph;
    ds4_gpu_graph *batch_graph = &decode_items[0].session->graph;
    if (!g->placement || !batch_graph->placement || !g->cuda_tp_ep ||
        !g->cuda_tp_prefill_ffn ||
        (uint64_t)prefill_rows + (uint32_t)decode_count > g->prefill_cap ||
        prefill_rows > g->raw_cap ||
        (start % g->prefill_cap) + prefill_rows > g->prefill_cap ||
        metal_graph_directional_steering_ffn_enabled(g) ||
        getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != NULL ||
        getenv("DS4_METAL_LAYER_STAGE_PROFILE") != NULL) {
        return false;
    }
    if (!metal_graph_mixed_workspace_compatible(g, batch_graph)) return false;
    if (!metal_graph_session_batch_moe_supported(
                decode_items, decode_count, weights) ||
        !metal_graph_session_batch_shared_supported(
                decode_items, decode_count, weights) ||
        !metal_graph_session_batch_ffn_pre_supported(
                decode_items, decode_count, weights) ||
        !metal_graph_session_batch_attn_pre_supported(
                decode_items, decode_count, weights)) {
        return false;
    }
    for (int i = 0; i < decode_count; i++) {
        ds4_session *s = decode_items[i].session;
        if (!s || s == prefill_session || s->engine != prefill_session->engine ||
            !s->checkpoint_valid || s->graph.prefill_cap != g->prefill_cap ||
            !metal_graph_mixed_workspace_compatible(g, &s->graph)) {
            return false;
        }
    }
    return true;
#endif
}

#endif /* !DS4_NO_GPU */
