#if defined(__APPLE__)
        const bool fuse_producer_pre_norm =
            fuse_norm_mix && fuse_hc_norm &&
            metal_graph_ported_m5_decode_feature_enabled(
                "DS4_METAL_DISABLE_PRE_M5_HC_PRODUCER_PRE_NORM_FUSE",
                "DS4_METAL_DISABLE_M5_HC_PRODUCER_PRE_NORM_FUSE");
        if (fuse_producer_pre_norm) {
            const int fused =
                ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor(
                    metal_graph_hc_mix(g),
                    metal_graph_ffn_cur(g),
                    metal_graph_ffn_norm(g),
                    metal_graph_hc_split(g),
                    metal_graph_after_attn_hc(g),
                    model->map,
                    model->size,
                    layer->hc_ffn_fn->abs_offset,
                    layer->hc_ffn_scale->abs_offset,
                    layer->hc_ffn_base->abs_offset,
                    layer->ffn_norm->abs_offset,
                    (uint32_t)hc_dim,
                    (uint32_t)mix_hc,
                    DS4_N_EMBD,
                    DS4_N_HC,
                    DS4_N_HC_SINKHORN_ITER,
                    DS4_RMS_EPS,
                    DS4_HC_EPS,
                    DS4_RMS_EPS);
            if (fused < 0) {
                ok = false;
            } else {
                ffn_hc_producer_pre_norm_fused = fused > 0;
            }
        }
#endif
        if (ok && !ffn_hc_producer_pre_norm_fused) {
            if (fuse_norm_mix) {
                ok = ds4_gpu_hc_rms_norm_mix_f16_tensor(
                        metal_graph_hc_mix(g), metal_graph_after_attn_hc(g),
                        model->map, model->size,
                        layer->hc_ffn_fn->abs_offset,
                        (uint32_t)hc_dim, (uint32_t)mix_hc,
                        DS4_RMS_EPS) != 0;
            } else {
                ok = ds4_gpu_rms_norm_plain_tensor(
                        metal_graph_flat_hc(g), metal_graph_after_attn_hc(g),
                        (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
                if (ok) {
                    ok = metal_graph_matmul_plain_tensor(
                            metal_graph_hc_mix(g), model, layer->hc_ffn_fn,
                            hc_dim, mix_hc, metal_graph_flat_hc(g), 1);
                }
            }
        }
    }
    if (ok && fuse_hc_norm) {
        if (!ffn_hc_producer_pre_norm_fused) {
            ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(metal_graph_ffn_cur(g),
                                                             metal_graph_ffn_norm(g),
                                                             metal_graph_hc_split(g),
                                                             metal_graph_hc_mix(g),
                                                             metal_graph_after_attn_hc(g),
                                                             model->map,
                                                             model->size,
                                                             layer->hc_ffn_scale->abs_offset,
                                                             layer->hc_ffn_base->abs_offset,
                                                             layer->ffn_norm->abs_offset,
                                                             DS4_N_EMBD,
                                                             DS4_N_HC,
                                                             DS4_N_HC_SINKHORN_ITER,
                                                             DS4_HC_EPS,
                                                             DS4_RMS_EPS) != 0;
        }
        if (ok) {
            ok = metal_graph_check_hc_norm_fusion("ffn",
                                                  metal_graph_ffn_cur(g),
                                                  metal_graph_ffn_norm(g),
                                                  metal_graph_hc_mix(g),
                                                  metal_graph_after_attn_hc(g),
                                                  model,
                                                  layer->hc_ffn_scale->abs_offset,
                                                  layer->hc_ffn_base->abs_offset,
                                                  layer->ffn_norm->abs_offset,
                                                  il,
                                                  pos);
        }
    } else if (ok) {
        ok = metal_graph_decode_hc_pre(metal_graph_ffn_cur(g),
                                       metal_graph_hc_split(g),
                                       metal_graph_hc_mix(g),
                                       metal_graph_after_attn_hc(g),
                                       model,
                                       layer->hc_ffn_scale->abs_offset,
                                       layer->hc_ffn_base->abs_offset);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_pre");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre_mixes", metal_graph_hc_mix(g), mix_hc, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_weights", metal_graph_hc_pre(g), DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_post_weights", metal_graph_hc_post(g), DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_ffn_pre_comb", metal_graph_hc_comb(g), (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre", metal_graph_ffn_cur(g), DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_ffn_norm(g), metal_graph_ffn_cur(g),
                                                                   model->map, model->size,
                                                                   layer->ffn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_norm");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_norm", metal_graph_ffn_norm(g), DS4_N_EMBD, il, pos);
    }
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes DS4_MAYBE_UNUSED = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes DS4_MAYBE_UNUSED = routed_out_dim * down_row_bytes;
    if (ok) {
        ok = metal_graph_matmul_plain_tensor(
                metal_graph_router_logits(g), model, layer->ffn_gate_inp,
                DS4_N_EMBD, DS4_N_EXPERT, metal_graph_ffn_norm(g), 1);
    }
    if (ok) {
        ok = ds4_gpu_router_select_tensor(
                metal_graph_router_selected(g),
                metal_graph_router_weights(g),
                metal_graph_router_probs(g),
                model->map,
                model->size,
                layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                layer->ffn_gate_tid2eid
                    ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                (uint32_t)token,
                DS4_N_EXPERT,
                DS4_N_EXPERT_USED,
                DS4_EXPERT_WEIGHT_SCALE,
                0,
                0,
                layer->ffn_exp_probs_b != NULL,
                layer->ffn_gate_tid2eid != NULL,
                metal_graph_router_logits(g)) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("router");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_logits", metal_graph_router_logits(g), DS4_N_EXPERT, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_probs", metal_graph_router_probs(g), DS4_N_EXPERT, il, pos);
        metal_graph_debug_dump_i32_tensor("ffn_moe_topk", metal_graph_router_selected(g), DS4_N_EXPERT_USED, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", metal_graph_router_weights(g), DS4_N_EXPERT_USED, il, pos);
    }
    const bool fuse_shared_down_hc =
        layer->ffn_down_shexp->type == DS4_TENSOR_Q8_0 &&
        !keep_ffn_out &&
        !metal_graph_use_reference_shared_down_hc();

    if (ok) {
        ok = ds4x_graph_routed_moe(
                g, model, layer, il,
                gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes,
                (uint32_t)expert_in_dim, (uint32_t)down_in_dim,
                (uint32_t)routed_out_dim);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("routed_moe");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_gate_clamped",
                                      metal_graph_routed_gate(g),
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim,
                                      il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_up_clamped",
                                      metal_graph_routed_up(g),
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim,
                                      il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu",
                                      metal_graph_routed_mid(g),
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim,
                                      il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_down",
                                      metal_graph_routed_down(g),
                                      (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD,
                                      il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_out",
                                      metal_graph_routed_out(g),
                                      DS4_N_EMBD, il, pos);
    }

    if (ok && fuse_shared_gate_up) {
        ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                metal_graph_shared_gate(g),
                metal_graph_shared_up(g),
                metal_graph_shared_mid(g),
                model->map,
                model->size,
                layer->ffn_gate_shexp->abs_offset,
                layer->ffn_up_shexp->abs_offset,
                DS4_N_EMBD,
                shared_dim,
                metal_graph_ffn_norm(g),
                DS4_SWIGLU_CLAMP_EXP) != 0;
    } else if (ok) {
        ok = metal_graph_matmul_dense_quant_tensor(
                metal_graph_shared_gate(g), model, layer->ffn_gate_shexp,
                DS4_N_EMBD, shared_dim, metal_graph_ffn_norm(g), 1);
        if (ok) {
            ok = metal_graph_matmul_dense_quant_tensor(
                    metal_graph_shared_up(g), model, layer->ffn_up_shexp,
                    DS4_N_EMBD, shared_dim, metal_graph_ffn_norm(g), 1);
        }
        if (ok) {
            ok = ds4_gpu_swiglu_tensor(
                    metal_graph_shared_mid(g), metal_graph_shared_gate(g),
                    metal_graph_shared_up(g), shared_dim,
                    DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
        }
    }
    DS4_METAL_PROFILE_DECODE_STAGE("shared_gate_up");

    if (ok && fuse_shared_down_hc) {
        ok = ds4_gpu_shared_down_hc_expand_q8_0_tensor(
                metal_graph_after_ffn_hc(g),
                metal_graph_shared_out(g),
                model->map,
                model->size,
                layer->ffn_down_shexp->abs_offset,
                shared_dim,
                DS4_N_EMBD,
                metal_graph_shared_mid(g),
                metal_graph_routed_out(g),
                metal_graph_after_attn_hc(g),
                metal_graph_hc_split(g),
                DS4_N_EMBD,
                DS4_N_HC) != 0;
    } else if (ok) {
        ok = metal_graph_matmul_dense_quant_tensor(
                metal_graph_shared_out(g), model, layer->ffn_down_shexp,
                shared_dim, DS4_N_EMBD, metal_graph_shared_mid(g), 1);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("shared_down");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_shexp", metal_graph_shared_out(g),
                                      DS4_N_EMBD, il, pos);
    }

    if (ok && keep_ffn_out) {
        ok = metal_graph_ensure_ffn_out(g) &&
             ds4_gpu_add_tensor(metal_graph_ffn_out(g),
                                metal_graph_shared_out(g),
                                metal_graph_routed_out(g), DS4_N_EMBD) != 0;
    }
    if (ok && keep_ffn_out) {
        metal_graph_debug_dump_tensor("ffn_out", metal_graph_ffn_out(g),
                                      DS4_N_EMBD, il, pos);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_ffn(
                g, metal_graph_ffn_out(g), il, 1);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_tensor(
                metal_graph_after_ffn_hc(g), metal_graph_ffn_out(g),
                metal_graph_after_attn_hc(g), metal_graph_hc_post(g),
                metal_graph_hc_comb(g), DS4_N_EMBD, DS4_N_HC) != 0;
    } else if (ok && !fuse_shared_down_hc) {
        ok = ds4_gpu_hc_expand_add_split_tensor(
                metal_graph_after_ffn_hc(g), metal_graph_routed_out(g),
                metal_graph_shared_out(g), metal_graph_after_attn_hc(g),
                metal_graph_hc_split(g), DS4_N_EMBD, DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_post");
#undef DS4_METAL_PROFILE_DECODE_STAGE
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_post",
                                      metal_graph_after_ffn_hc(g),
                                      hc_dim, il, pos);
    }
    return ok;
}

static bool metal_graph_encode_decode_layer(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token) {
    return metal_graph_encode_decode_layer_phase(
            g, model, layer, il, pos, raw_cache, raw_cap, raw_row, n_raw,
            token, METAL_DECODE_LAYER_FULL);
}

static bool metal_graph_output_logits_head_matmul(
        ds4_gpu_graph        *g,
        const ds4_model      *model,
        const ds4_weights    *weights,
        ds4_gpu_tensor       *norm_full,
        ds4_gpu_tensor       *dst_logits,
        uint32_t              n_tokens,
        uint64_t              vocab_dim);
