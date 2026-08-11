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
    if (ok && metal_graph_decode_cpu_router_applicable(g, layer)) {
        ok = metal_graph_decode_cpu_router(g, model, layer, il, (uint32_t)token);
    } else {
        if (ok && !metal_graph_tp_ablate("router")) {
        /* Fused router + shared-expert gate/up: one dispatch instead of two
         * on the same normalized FFN input; per-row math is unchanged (see
         * metal/dense.metal).  Restricted to the full single-pass decode
         * phase so split-phase/graph flows keep the original ordering. */
        if (phase == METAL_DECODE_LAYER_FULL &&
            !g->ssd_streaming &&
            !g->quality &&
            g->tp_world < 2 &&
            layer->ffn_gate_shexp->type == DS4_TENSOR_Q8_0 &&
            layer->ffn_up_shexp->type == DS4_TENSOR_Q8_0 &&
            g->shared_gate_up_swiglu_fuse &&
            layer->ffn_gate_inp->type == DS4_TENSOR_F16 &&
            layer->ffn_gate_inp->dim[0] == DS4_N_EMBD &&
            layer->ffn_gate_inp->dim[1] == DS4_N_EXPERT &&
            (!ds4_gpu_device_is_pre_m5_apple_silicon() ||
             getenv("DS4_METAL_DISABLE_PRE_M5_ROUTER_SHARED_FUSE") == NULL) &&
            (ds4_gpu_device_is_pre_m5_apple_silicon() ||
             ds4_gpu_device_is_m5_apple_silicon())) {
#if defined(__APPLE__)
            const bool fuse_router_project_select =
                parallel_full_ffn_eligible &&
                layer->ffn_gate_tid2eid == NULL &&
                getenv("DS4_METAL_DISABLE_M5_ROUTER_PROJECT_SELECT_FUSE") == NULL &&
                ds4_gpu_device_is_m5_apple_silicon();
            if (fuse_router_project_select) {
                const int fused =
                    ds4_gpu_router_project_select_fused_tensor(
                            metal_graph_router_logits(g),
                            metal_graph_router_probs(g),
                            metal_graph_router_selected(g),
                            metal_graph_router_weights(g),
                            model->map,
                            model->size,
                            layer->ffn_gate_inp->abs_offset,
                            layer->ffn_exp_probs_b
                                ? layer->ffn_exp_probs_b->abs_offset : 0,
                            layer->ffn_exp_probs_b != NULL,
                            metal_graph_ffn_norm(g));
                if (fused < 0) {
                    ok = false;
                } else {
                    router_project_select_fused = fused > 0;
                }
            }
#endif
            if (ok && !router_project_select_fused) {
                router_shared_done =
                    ds4_gpu_router_shared_gate_up_q8_0_tensor(
                            metal_graph_router_logits(g),
                            metal_graph_shared_gate(g),
                            metal_graph_shared_up(g),
                            metal_graph_shared_mid(g),
                            model->map,
                            model->size,
                            layer->ffn_gate_inp->abs_offset,
                            layer->ffn_gate_shexp->abs_offset,
                            layer->ffn_up_shexp->abs_offset,
                            DS4_N_EMBD,
                            DS4_N_EXPERT,
                            shared_dim,
                            metal_graph_ffn_norm(g),
                            DS4_SWIGLU_CLAMP_EXP,
                            parallel_full_ffn_eligible);
            } else {
                router_only_done = 1;
            }
            if (router_shared_done < 0) {
                ok = false;
                router_shared_done = 0;
            }
            if (router_shared_done != 0 && parallel_full_ffn_eligible) {
                router_only_done = 1;
                router_shared_done = 0;
            }
        }
        if (ok && router_shared_done == 0 && router_only_done == 0)
        ok = metal_graph_matmul_plain_tensor(metal_graph_router_logits(g), model, layer->ffn_gate_inp,
                                                     DS4_N_EMBD, DS4_N_EXPERT, metal_graph_ffn_norm(g), 1);
        if (ok && !router_project_select_fused)
            ok = ds4_gpu_router_select_tensor(metal_graph_router_selected(g), metal_graph_router_weights(g), metal_graph_router_probs(g),
                                                    model->map, model->size,
                                                    layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                    layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                    layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
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
        if (ok) ok = metal_graph_decode_set_hash_selected_override(model,
                                                                   layer,
                                                                   il,
                                                                   (uint32_t)token,
                                                                   layer->ffn_gate_exps->bytes,
                                                                   layer->ffn_down_exps->bytes,
                                                                   g);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("router");
    if (ok) ok = metal_graph_profile_router_selection(g, layer, il, pos);
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_logits", metal_graph_router_logits(g), DS4_N_EXPERT, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_probs", metal_graph_router_probs(g), DS4_N_EXPERT, il, pos);
        metal_graph_debug_dump_i32_tensor("ffn_moe_topk", metal_graph_router_selected(g), DS4_N_EXPERT_USED, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", metal_graph_router_weights(g), DS4_N_EXPERT_USED, il, pos);
    }
    if (phase == METAL_DECODE_LAYER_TO_ROUTER) return ok;
    }
    const bool external_routed = phase == METAL_DECODE_LAYER_FROM_ROUTER;
    const bool cuda_tp_shared_requested = g->cuda_tp_shared;
    const bool cuda_tp_moe_requested = !external_routed && g->cuda_tp_moe;
    const uint64_t shared_tp_local = shared_dim / 2u;
    const uint64_t shared_tp_peer = shared_dim - shared_tp_local;
    const uint64_t shared_q8_blocks = ((uint64_t)DS4_N_EMBD + 31u) / 32u;
    const uint64_t shared_q8_x_bytes = shared_q8_blocks * 32u;
    const uint64_t shared_q8_scale_offset =
        (shared_q8_x_bytes + 15u) & ~15ull;
    const uint64_t shared_q8_prequant_bytes DS4_MAYBE_UNUSED =
        shared_q8_scale_offset + shared_q8_blocks * sizeof(float);
    const bool cuda_tp_shared =
        cuda_tp_shared_requested &&
        fuse_shared_gate_up &&
        !metal_graph_use_reference_shared_down_hc() &&
        cuda_tp_partner_tier >= 0 &&
        shared_tp_local != 0 &&
        shared_tp_peer != 0 &&
        (shared_tp_local % 32u) == 0 &&
        (shared_tp_peer % 32u) == 0 &&
        g->ffn_norm_by_tier[cuda_tp_partner_tier] &&
        g->shared_gate_by_tier[cuda_tp_partner_tier] &&
        g->shared_up_by_tier[cuda_tp_partner_tier] &&
        g->shared_mid_by_tier[cuda_tp_partner_tier] &&
        g->shared_out_by_tier[cuda_tp_partner_tier] &&
        g->tp_peer_tmp_by_tier[cuda_tp_home_tier];
    const bool cuda_tp_shared_fold =
        cuda_tp_shared &&
        g->cuda_tp_shared_fold &&
        !g->cuda_tp_ep &&
        cuda_tp_moe_requested &&
        !keep_ffn_out &&
        !metal_graph_directional_steering_ffn_enabled(g) &&
        !metal_graph_debug_wants("ffn_moe_out", il, pos) &&
        !metal_graph_debug_wants("ffn_shexp", il, pos);
    const bool fuse_shared_down_hc =
        g->tp_world < 2 &&
        layer->ffn_down_shexp->type == DS4_TENSOR_Q8_0 &&
        !cuda_tp_shared &&
        !keep_ffn_out &&
        !metal_graph_use_reference_shared_down_hc();
    const bool cuda_tp_moe =
        cuda_tp_moe_requested &&
        cuda_tp_partner_tier >= 0 &&
        (DS4_N_EXPERT_USED % 2u) == 0u &&
        g->ffn_norm_by_tier[cuda_tp_partner_tier] &&
        g->router_selected_by_tier[cuda_tp_partner_tier] &&
        g->router_weights_by_tier[cuda_tp_partner_tier] &&
        g->routed_gate_by_tier[cuda_tp_partner_tier] &&
        g->routed_up_by_tier[cuda_tp_partner_tier] &&
        g->routed_mid_by_tier[cuda_tp_partner_tier] &&
        g->routed_down_by_tier[cuda_tp_partner_tier] &&
        g->routed_out_by_tier[cuda_tp_partner_tier] &&
        g->tp_peer_tmp_by_tier[cuda_tp_home_tier] &&
        g->tp_peer_tmp_by_tier[cuda_tp_partner_tier];
    const bool cuda_tp_ep = cuda_tp_moe && g->cuda_tp_ep;
    const bool cuda_tp_moe_delay_reduce =
        cuda_tp_moe &&
        g->cuda_tp_moe_delay_reduce &&
        fuse_shared_down_hc &&
        !metal_graph_debug_wants("ffn_moe_out", il, pos);
    bool cuda_tp_moe_peer_tmp = false;
    bool cuda_tp_moe_peer_copy_deferred = false;
    bool cuda_tp_ep_reduce_deferred = false;
    bool cuda_tp_ep_fused_hc_reduce = false;
    bool cuda_tp_ep_direct_return = false;
    bool cuda_tp_ep_balanced_shared_mid DS4_MAYBE_UNUSED = false;
    bool cuda_tp_ep_dual_prequant = false;
    uint64_t cuda_tp_ep_return_bytes = 0;
    bool cuda_tp_shared_fold_peer_tmp = false;
    if (ok && cuda_tp_moe_requested && !cuda_tp_moe) {
        fprintf(stderr,
                "ds4: CUDA decode TP cannot split routed MoE for tier %d "
                "(partner=%d experts=%u)\n",
                cuda_tp_home_tier, cuda_tp_partner_tier, DS4_N_EXPERT_USED);
        ok = false;
    }
    if (ok && cuda_tp_moe) {
        const uint32_t tp_experts = cuda_tp_ep
            ? DS4_N_EXPERT_USED : DS4_N_EXPERT_USED / 2u;
        const uint64_t tp_selected_bytes = (uint64_t)tp_experts * sizeof(int32_t);
        const uint64_t tp_weights_bytes = (uint64_t)tp_experts * sizeof(float);
        const uint64_t peer_selected_offset = cuda_tp_ep ? 0 : tp_selected_bytes;
        const uint64_t peer_weights_offset = cuda_tp_ep ? 0 : tp_weights_bytes;
        ds4_gpu_tensor local_selected;
        ds4_gpu_tensor local_weights;
        ds4_gpu_tensor peer_selected_src;
        ds4_gpu_tensor peer_weights_src;
        ds4_gpu_tensor packed_peer_ffn_norm;
        ds4_gpu_tensor packed_peer_selected;
        ds4_gpu_tensor packed_peer_weights;
        ds4_gpu_tensor direct_peer_down;
        ds4_gpu_tensor *peer_down_output = NULL;
        cuda_tp_ep_return_bytes =
            (uint64_t)(g->cuda_tp_ep_pack_exact ? 4u : DS4_N_EXPERT_USED) *
            DS4_N_EMBD * sizeof(float);
        cuda_tp_ep_direct_return =
            cuda_tp_ep &&
            metal_graph_cuda_tp_ep_direct_return_requested() &&
            g_gpu_peer_ok[cuda_tp_partner_tier][cuda_tp_home_tier] &&
            metal_graph_borrow_tensor_view(
                    &direct_peer_down,
                    g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                    0,
                    cuda_tp_ep_return_bytes);
        if (cuda_tp_ep_direct_return) peer_down_output = &direct_peer_down;
#if !defined(__APPLE__)
        /* Run the shared gate/up projection on the less-loaded EP rank.  The
         * two kernels use complementary predicates over the same top-k IDs;
         * a partner result is ordered by the existing direct-return event. */
        cuda_tp_ep_balanced_shared_mid =
            cuda_tp_ep_direct_return &&
            cuda_tp_moe_delay_reduce &&
            metal_graph_cuda_tp_ep_delay_reduce_requested() &&
            metal_graph_cuda_tp_ep_fused_shared_mid_requested() &&
            metal_graph_cuda_tp_ep_balanced_shared_mid_requested() &&
            fuse_shared_gate_up &&
            !cuda_tp_shared_requested &&
            !g->cuda_tp_moe_peer_read &&
            !g->cuda_tp_moe_peer_router &&
            !g->decode_stage_profile;
        cuda_tp_ep_dual_prequant =
            cuda_tp_ep_balanced_shared_mid &&
            metal_graph_cuda_tp_ep_dual_prequant_requested() &&
            metal_graph_shared_gate(g) &&
            metal_graph_shared_gate(g)->bytes >= shared_q8_prequant_bytes &&
            g->shared_gate_by_tier[cuda_tp_partner_tier] &&
            g->shared_gate_by_tier[cuda_tp_partner_tier]->bytes >=
                shared_q8_prequant_bytes;
#endif
        const bool cuda_tp_moe_peer_read =
            g->cuda_tp_moe_peer_read &&
            g_gpu_peer_ok[cuda_tp_partner_tier][cuda_tp_home_tier];
        const bool cuda_tp_moe_peer_router =
            !cuda_tp_moe_peer_read &&
            g->cuda_tp_moe_peer_router &&
            g_gpu_peer_ok[cuda_tp_partner_tier][cuda_tp_home_tier];
        const ds4_gpu_tensor *peer_selected = g->router_selected_by_tier[cuda_tp_partner_tier];
        const ds4_gpu_tensor *peer_weights = g->router_weights_by_tier[cuda_tp_partner_tier];
        const ds4_gpu_tensor *peer_ffn_norm = g->ffn_norm_by_tier[cuda_tp_partner_tier];
        ok = metal_graph_borrow_tensor_view(&local_selected,
                                            metal_graph_router_selected(g),
                                            0,
                                            tp_selected_bytes) &&
             metal_graph_borrow_tensor_view(&local_weights,
                                            metal_graph_router_weights(g),
                                            0,
                                            tp_weights_bytes) &&
             metal_graph_borrow_tensor_view(&peer_selected_src,
                                            metal_graph_router_selected(g),
                                            peer_selected_offset,
                                            tp_selected_bytes) &&
             metal_graph_borrow_tensor_view(&peer_weights_src,
                                            metal_graph_router_weights(g),
                                            peer_weights_offset,
                                            tp_weights_bytes);
        if (ok && cuda_tp_moe_peer_read) {
            ok = ds4_gpu_tensor_wait_xdev(metal_graph_router_weights(g),
                                          cuda_tp_partner_tier) != 0;
            peer_selected = &peer_selected_src;
            peer_weights = &peer_weights_src;
            peer_ffn_norm = metal_graph_ffn_norm(g);
        } else if (ok && cuda_tp_moe_peer_router) {
            ok = ds4_gpu_tensor_copy_xdev(g->ffn_norm_by_tier[cuda_tp_partner_tier],
                                          metal_graph_ffn_norm(g),
                                          (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
            peer_selected = &peer_selected_src;
            peer_weights = &peer_weights_src;
            peer_ffn_norm = g->ffn_norm_by_tier[cuda_tp_partner_tier];
        } else if (ok && g->cuda_tp_moe_pack_handoff) {
            const uint64_t norm_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
            const uint64_t packed_selected_off = norm_bytes;
            const uint64_t packed_weights_off = packed_selected_off + tp_selected_bytes;
            const uint64_t packed_bytes = packed_weights_off + tp_weights_bytes;
            ok = metal_graph_borrow_tensor_view(&packed_peer_ffn_norm,
                                                g->tp_peer_tmp_by_tier[cuda_tp_partner_tier],
                                                0,
                                                norm_bytes) &&
                 metal_graph_borrow_tensor_view(&packed_peer_selected,
                                                g->tp_peer_tmp_by_tier[cuda_tp_partner_tier],
                                                packed_selected_off,
                                                tp_selected_bytes) &&
                 metal_graph_borrow_tensor_view(&packed_peer_weights,
                                                g->tp_peer_tmp_by_tier[cuda_tp_partner_tier],
                                                packed_weights_off,
                                                tp_weights_bytes);
            if (ok) {
                ok = ds4_gpu_moe_handoff_pack_tensor(
                        g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                        metal_graph_ffn_norm(g),
                        &peer_selected_src,
                        &peer_weights_src,
                        DS4_N_EMBD,
                        tp_experts) != 0;
            }
            if (ok) {
                ok = ds4_gpu_tensor_copy_xdev(
                        g->tp_peer_tmp_by_tier[cuda_tp_partner_tier],
                        g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                        packed_bytes) != 0;
            }
            peer_selected = &packed_peer_selected;
            peer_weights = &packed_peer_weights;
            peer_ffn_norm = &packed_peer_ffn_norm;
        } else if (ok && g->cuda_tp_moe_copy3_handoff) {
            ok = ds4_gpu_tensor_copy_xdev3(
                    g->ffn_norm_by_tier[cuda_tp_partner_tier],
                    metal_graph_ffn_norm(g),
                    (uint64_t)DS4_N_EMBD * sizeof(float),
                    g->router_selected_by_tier[cuda_tp_partner_tier],
                    &peer_selected_src,
                    tp_selected_bytes,
                    g->router_weights_by_tier[cuda_tp_partner_tier],
                    &peer_weights_src,
                    tp_weights_bytes) != 0;
        } else if (ok) {
            ok = ds4_gpu_tensor_copy_xdev(g->ffn_norm_by_tier[cuda_tp_partner_tier],
                                          metal_graph_ffn_norm(g),
                                          (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
                 ds4_gpu_tensor_copy_xdev(g->router_selected_by_tier[cuda_tp_partner_tier],
                                          &peer_selected_src,
                                          tp_selected_bytes) != 0 &&
                 ds4_gpu_tensor_copy_xdev(g->router_weights_by_tier[cuda_tp_partner_tier],
                                          &peer_weights_src,
                                          tp_weights_bytes) != 0;
        }

        bool switched_to_partner = false;
        if (ok) {
            ok = ds4_gpu_set_current_device(cuda_tp_partner_tier) == 0;
            switched_to_partner = ok;
        }
        if (ok) {
            if (cuda_tp_ep) {
                ok = ds4_gpu_routed_moe_one_owned_tensor(
                        g->routed_out_by_tier[cuda_tp_partner_tier],
                        g->routed_gate_by_tier[cuda_tp_partner_tier],
                        g->routed_up_by_tier[cuda_tp_partner_tier],
                        g->routed_mid_by_tier[cuda_tp_partner_tier],
                        g->routed_down_by_tier[cuda_tp_partner_tier],
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
                        peer_selected,
                        peer_weights,
                        DS4_N_EXPERT,
                        tp_experts,
                        DS4_N_EXPERT / 2u,
                        DS4_N_EXPERT - DS4_N_EXPERT / 2u,
                        DS4_SWIGLU_CLAMP_EXP,
                        peer_ffn_norm,
                        peer_down_output,
                        g->cuda_tp_ep_pack_exact,
                        cuda_tp_ep_dual_prequant
                            ? g->shared_gate_by_tier[cuda_tp_partner_tier]
                            : NULL) != 0;
            } else {
                ok = ds4_gpu_routed_moe_one_tensor(
                        g->routed_out_by_tier[cuda_tp_partner_tier],
                        g->routed_gate_by_tier[cuda_tp_partner_tier],
                        g->routed_up_by_tier[cuda_tp_partner_tier],
                        g->routed_mid_by_tier[cuda_tp_partner_tier],
                        g->routed_down_by_tier[cuda_tp_partner_tier],
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
                        peer_selected,
                        peer_weights,
                        DS4_N_EXPERT,
                        tp_experts,
                        DS4_SWIGLU_CLAMP_EXP,
                        peer_ffn_norm, NULL, 0, false) != 0;
            }
        }
#if !defined(__APPLE__)
        if (ok && cuda_tp_ep_balanced_shared_mid) {
            ok = ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
                    metal_graph_shared_mid(g),
                    model->map,
                    model->size,
                    layer->ffn_gate_shexp->abs_offset,
                    layer->ffn_up_shexp->abs_offset,
                    DS4_N_EMBD,
                    shared_dim,
                    peer_ffn_norm,
                    DS4_SWIGLU_CLAMP_EXP,
                    peer_selected,
                    cuda_tp_ep_dual_prequant
                        ? g->shared_gate_by_tier[cuda_tp_partner_tier]
                        : NULL,
                    DS4_N_EXPERT / 2u,
                    false) != 0;
        }
#endif
        if (switched_to_partner && ds4_gpu_set_current_device(cuda_tp_home_tier) != 0) {
            ok = false;
        }
        if (ok) {
            if (cuda_tp_ep) {
                ok = ds4_gpu_routed_moe_one_owned_tensor(
                        metal_graph_routed_out(g),
                        metal_graph_routed_gate(g),
                        metal_graph_routed_up(g),
                        metal_graph_routed_mid(g),
                        metal_graph_routed_down(g),
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
                        &local_selected,
                        &local_weights,
                        DS4_N_EXPERT,
                        tp_experts,
                        0,
                        DS4_N_EXPERT / 2u,
                        DS4_SWIGLU_CLAMP_EXP,
                        metal_graph_ffn_norm(g),
                        NULL,
                        false,
                        cuda_tp_ep_dual_prequant
                            ? metal_graph_shared_gate(g)
                            : NULL) != 0;
            } else {
                ok = ds4_gpu_routed_moe_one_tensor(
                        metal_graph_routed_out(g),
                        metal_graph_routed_gate(g),
                        metal_graph_routed_up(g),
                        metal_graph_routed_mid(g),
                        metal_graph_routed_down(g),
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
                        &local_selected,
                        &local_weights,
                        DS4_N_EXPERT,
                        tp_experts,
                        DS4_SWIGLU_CLAMP_EXP,
                        metal_graph_ffn_norm(g), NULL, 0, false) != 0;
            }
        }
#if !defined(__APPLE__)
        if (ok && cuda_tp_ep_balanced_shared_mid) {
            ok = ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
                    metal_graph_shared_mid(g),
                    model->map,
                    model->size,
                    layer->ffn_gate_shexp->abs_offset,
                    layer->ffn_up_shexp->abs_offset,
                    DS4_N_EMBD,
                    shared_dim,
                    metal_graph_ffn_norm(g),
                    DS4_SWIGLU_CLAMP_EXP,
                    &local_selected,
                    cuda_tp_ep_dual_prequant
                        ? metal_graph_shared_gate(g)
                        : NULL,
                    DS4_N_EXPERT / 2u,
                    true) != 0;
        }
#endif
        if (ok) {
            if (cuda_tp_ep) {
                if (cuda_tp_moe_delay_reduce &&
                    metal_graph_cuda_tp_ep_delay_reduce_requested() &&
                    !g->decode_stage_profile) {
                    cuda_tp_ep_reduce_deferred = true;
                    cuda_tp_ep_fused_hc_reduce =
                        g->cuda_tp_ep_pack_exact &&
                        metal_graph_cuda_tp_ep_fused_hc_reduce_requested();
                } else {
                    ok = metal_graph_cuda_tp_ep_finish_reduce(
                            g,
                            cuda_tp_home_tier,
                            cuda_tp_partner_tier,
                            cuda_tp_ep_direct_return,
                            cuda_tp_ep_return_bytes,
                            true);
                }
            } else if (cuda_tp_moe_delay_reduce &&
                !cuda_tp_shared && !cuda_tp_shared_fold) {
                /* Defer the peer routed-half copy until after the shared
                 * expert gate/up launch below: that work depends only on
                 * ffn_norm, so the home stream computes it while waiting
                 * for the partner instead of idling at the copy. */
                cuda_tp_moe_peer_copy_deferred = true;
            } else if (cuda_tp_moe_delay_reduce) {
                ok = ds4_gpu_tensor_copy_xdev(
                        g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                        g->routed_out_by_tier[cuda_tp_partner_tier],
                        (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
                cuda_tp_moe_peer_tmp = ok;
            } else if (!cuda_tp_shared_fold) {
                ok = ds4_gpu_add_xdev_tensor(metal_graph_routed_out(g),
                                             metal_graph_routed_out(g),
                                             g->routed_out_by_tier[cuda_tp_partner_tier],
                                             g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                                             DS4_N_EMBD) != 0;
            }
        }
    }
    /* Real TP split slices the shared expert by intermediate lanes, which
     * needs the unfused gate/up/swiglu/down sequence. */
    const bool tp_split_shared = g->tp_world == 2;
    const bool q4_selected_shared_overlap =
        metal_graph_use_q4_selected_shared_overlap(g) &&
        metal_graph_decode_q4_selected_slots_expected(g,
                                                      layer,
                                                      layer->ffn_gate_exps->bytes,
                                                      layer->ffn_down_exps->bytes);
    const bool iq2_selected_shared_overlap =
        metal_graph_use_iq2_selected_shared_overlap(g) &&
        metal_graph_decode_iq2_selected_slots_expected(g, layer);
    const bool mxfp4_selected_shared_overlap =
        metal_graph_use_iq2_selected_shared_overlap(g) &&
        metal_graph_decode_mxfp4_selected_slots_expected(g, layer);
    const bool cuda_selected_shared_overlap =
        metal_graph_use_cuda_selected_shared_overlap(g) &&
        metal_graph_decode_cuda_selected_slots_expected(g, layer);
    const bool overlap_selected_shared =
        ok &&
        g->tp_world < 2 &&
        !decode_stage_profile &&
        !metal_graph_decode_cpu_router_applicable(g, layer) &&
        layer->ffn_gate_tid2eid == NULL &&
        getenv("DS4_MOE_REPLAY_SELECTED_IDS") == NULL &&
        (q4_selected_shared_overlap ||
         iq2_selected_shared_overlap ||
         mxfp4_selected_shared_overlap ||
         cuda_selected_shared_overlap);
    const bool async_selected_load =
        overlap_selected_shared &&
        ((iq2_selected_shared_overlap &&
          metal_graph_use_iq2_selected_async_load(g)) ||
         (mxfp4_selected_shared_overlap &&
          metal_graph_use_iq2_selected_async_load(g)) ||
         cuda_selected_shared_overlap);
    const bool selected_readahead_shared_delay =
        ok &&
        g->tp_world < 2 &&
        !overlap_selected_shared &&
        !decode_stage_profile &&
        metal_graph_use_iq2_selected_readahead_shared_delay(g) &&
        metal_graph_decode_iq2_selected_slots_expected(g, layer) &&
        !metal_graph_decode_cpu_router_applicable(g, layer) &&
        layer->ffn_gate_tid2eid == NULL &&
        getenv("DS4_MOE_REPLAY_SELECTED_IDS") == NULL;
    const bool cuda_stream_selected_load =
        ok &&
        !overlap_selected_shared &&
        !selected_readahead_shared_delay &&
        g->ssd_streaming &&
        metal_graph_decode_cuda_selected_slots_expected(g, layer) &&
        layer->ffn_gate_tid2eid == NULL &&
        getenv("DS4_MOE_REPLAY_SELECTED_IDS") == NULL;
    if (cuda_stream_selected_load) {
        ok = metal_graph_decode_cuda_selected_load(g,
                                                   model,
                                                   layer,
                                                   il,
                                                   gate_expert_bytes,
                                                   down_expert_bytes);
    }
    if (selected_readahead_shared_delay) {
        if (ok) {
            ok = metal_graph_decode_selected_readahead_override(g,
                                                                model,
                                                                layer,
                                                                il,
                                                                gate_expert_bytes,
                                                                down_expert_bytes);
        }
        if (ok && fuse_shared_gate_up) {
            ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(metal_graph_shared_gate(g),
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
            if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_gate(g),
                                                               model,
                                                               layer->ffn_gate_shexp,
                                                               DS4_N_EMBD,
                                                               shared_dim,
                                                               metal_graph_ffn_norm(g),
                                                               1);
            if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_up(g),
                                                               model,
                                                               layer->ffn_up_shexp,
                                                               DS4_N_EMBD,
                                                               shared_dim,
                                                               metal_graph_ffn_norm(g),
                                                               1);
            if (ok) ok = ds4_gpu_swiglu_tensor(metal_graph_shared_mid(g), metal_graph_shared_gate(g), metal_graph_shared_up(g),
                                               shared_dim, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
        }
        DS4_METAL_PROFILE_DECODE_STAGE("shared_gate_up");
        if (ok) ok = ds4_gpu_routed_moe_one_tensor(metal_graph_routed_out(g),
                                                     metal_graph_routed_gate(g),
                                                     metal_graph_routed_up(g),
                                                     metal_graph_routed_mid(g),
                                                     metal_graph_routed_down(g),
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
                                                     metal_graph_router_selected(g), metal_graph_router_weights(g),
                                                     DS4_N_EXPERT,
                                                     DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, metal_graph_ffn_norm(g),
                                                     NULL,
                                                     il,
                                                     false) != 0;
        DS4_METAL_PROFILE_DECODE_STAGE("routed_moe");
        if (ok) {
            metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", metal_graph_routed_gate(g),
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_up_clamped", metal_graph_routed_up(g),
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", metal_graph_routed_mid(g),
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_down", metal_graph_routed_down(g),
                                          (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_out", metal_graph_routed_out(g), DS4_N_EMBD, il, pos);
        }
        if (ok && fuse_shared_down_hc) {
            ok = ds4_gpu_shared_down_hc_expand_q8_0_tensor(metal_graph_after_ffn_hc(g),
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
            ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_out(g),
                                                       model,
                                                       layer->ffn_down_shexp,
                                                       shared_dim,
                                                       DS4_N_EMBD,
                                                       metal_graph_shared_mid(g),
                                                       1);
        }
        DS4_METAL_PROFILE_DECODE_STAGE("shared_down");
        if (ok) {
            metal_graph_debug_dump_tensor("ffn_shexp", metal_graph_shared_out(g), DS4_N_EMBD, il, pos);
        }
        if (ok && keep_ffn_out) {
            ok = metal_graph_ensure_ffn_out(g) &&
                 ds4_gpu_add_tensor(metal_graph_ffn_out(g), metal_graph_shared_out(g), metal_graph_routed_out(g), DS4_N_EMBD) != 0;
        }
        if (ok && keep_ffn_out) {
            metal_graph_debug_dump_tensor("ffn_out", metal_graph_ffn_out(g), DS4_N_EMBD, il, pos);
        }
        if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
            ok = metal_graph_apply_directional_steering_ffn(g, metal_graph_ffn_out(g), il, 1);
        }
        if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
            ok = ds4_gpu_hc_expand_tensor(metal_graph_after_ffn_hc(g),
                                            metal_graph_ffn_out(g),
                                            metal_graph_after_attn_hc(g),
                                            metal_graph_hc_post(g),
                                            metal_graph_hc_comb(g),
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
        } else if (ok && !fuse_shared_down_hc) {
            ok = ds4_gpu_hc_expand_add_split_tensor(metal_graph_after_ffn_hc(g),
                                                      metal_graph_routed_out(g),
                                                      metal_graph_shared_out(g),
                                                      metal_graph_after_attn_hc(g),
                                                      metal_graph_hc_split(g),
                                                      DS4_N_EMBD,
                                                      DS4_N_HC) != 0;
        }
        DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_post");
        if (ok) {
            metal_graph_debug_dump_tensor("hc_ffn_post", metal_graph_after_ffn_hc(g), hc_dim, il, pos);
        }
        return ok;
    }
    if (overlap_selected_shared) {
        uint64_t selected_event = 0;
        if (ok) ok = ds4_gpu_signal_selected_readback_ready(&selected_event) != 0;
        metal_graph_selected_async_load async_load = {0};
        bool async_load_started = false;
        const bool async_early_commit =
            async_selected_load &&
            metal_graph_use_iq2_selected_async_early_commit(g);
        if (ok && async_selected_load) {
            ok = metal_graph_selected_async_load_start(&async_load,
                                                       g,
                                                       model,
                                                       layer,
                                                       il,
                                                       selected_event,
                                                       gate_expert_bytes,
                                                       down_expert_bytes);
            async_load_started = ok;
        }
        if (ok && async_early_commit) {
            ok = ds4_gpu_flush_commands() != 0;
        }
        if (ok && fuse_shared_gate_up) {
            ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(metal_graph_shared_gate(g),
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
            if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_gate(g),
                                                               model,
                                                               layer->ffn_gate_shexp,
                                                               DS4_N_EMBD,
                                                               shared_dim,
                                                               metal_graph_ffn_norm(g),
                                                               1);
            if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_up(g),
                                                               model,
                                                               layer->ffn_up_shexp,
                                                               DS4_N_EMBD,
                                                               shared_dim,
                                                               metal_graph_ffn_norm(g),
                                                               1);
            if (ok) ok = ds4_gpu_swiglu_tensor(metal_graph_shared_mid(g), metal_graph_shared_gate(g), metal_graph_shared_up(g),
                                               shared_dim, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
        }
        DS4_METAL_PROFILE_DECODE_STAGE("shared_gate_up");
        if (ok && !fuse_shared_down_hc) {
            ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_out(g),
                                                       model,
                                                       layer->ffn_down_shexp,
                                                       shared_dim,
                                                       DS4_N_EMBD,
                                                       metal_graph_shared_mid(g),
                                                       1);
        }
        DS4_METAL_PROFILE_DECODE_STAGE("shared_down");
        if (async_load_started) {
            const bool flush_ok = ds4_gpu_flush_commands() != 0;
            bool finish_ok =
                metal_graph_selected_async_load_finish(&async_load);
            if (!finish_ok && async_load.ids_ok) {
                /* The worker read valid ids but could not stage the load
                 * (it is not allowed to wait on in-flight cache entries).
                 * This thread is, so retry the same load synchronously. */
                const ds4_gpu_stream_expert_table retry_table =
                    graph_stream_expert_table_make(model,
                                                   layer,
                                                   il,
                                                   gate_expert_bytes,
                                                   down_expert_bytes);
                finish_ok =
                    ds4_gpu_stream_expert_cache_begin_selected_load(
                            &retry_table,
                            async_load.selected_ids,
                            DS4_N_EXPERT_USED) != 0 &&
                    ds4_gpu_routed_moe_set_selected_override(
                            async_load.selected_ids,
                            DS4_N_EXPERT_USED) != 0;
            }
            ok = ok && flush_ok && finish_ok;
        } else if (ok) {
            ok = ds4_gpu_commit_and_wait_selected_readback(selected_event,
                                                           "selected-id shared-overlap") != 0;
        }
        if (ok && !async_load_started) {
            int32_t selected_ids[DS4_MAX_EXPERT_USED];
            ok = ds4_gpu_tensor_read(metal_graph_router_selected(g),
                                     0,
                                     selected_ids,
                                     (uint64_t)DS4_N_EXPERT_USED * sizeof(selected_ids[0])) != 0 &&
                 ds4_gpu_routed_moe_set_selected_override(selected_ids,
                                                          DS4_N_EXPERT_USED) != 0;
            if (ok) {
                const ds4_gpu_stream_expert_table table =
                    graph_stream_expert_table_make(model,
                                                   layer,
                                                   il,
                                                   gate_expert_bytes,
                                                   down_expert_bytes);
                ok = ds4_gpu_stream_expert_cache_begin_selected_load(
                            &table,
                            selected_ids,
                            DS4_N_EXPERT_USED) != 0;
            }
        }
        if (ok) ok = ds4_gpu_routed_moe_one_tensor(metal_graph_routed_out(g),
                                                     metal_graph_routed_gate(g),
                                                     metal_graph_routed_up(g),
                                                     metal_graph_routed_mid(g),
                                                     metal_graph_routed_down(g),
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
                                                     metal_graph_router_selected(g), metal_graph_router_weights(g),
                                                     DS4_N_EXPERT,
                                                     DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, metal_graph_ffn_norm(g),
                                                     NULL,
                                                     il,
                                                     false) != 0;
        DS4_METAL_PROFILE_DECODE_STAGE("routed_moe");
        if (ok) {
            metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", metal_graph_routed_gate(g),
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_up_clamped", metal_graph_routed_up(g),
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", metal_graph_routed_mid(g),
                                          (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_down", metal_graph_routed_down(g),
                                          (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
            metal_graph_debug_dump_tensor("ffn_moe_out", metal_graph_routed_out(g), DS4_N_EMBD, il, pos);
        }
        if (ok && fuse_shared_down_hc) {
            ok = ds4_gpu_shared_down_hc_expand_q8_0_tensor(metal_graph_after_ffn_hc(g),
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
        }
        DS4_METAL_PROFILE_DECODE_STAGE("shared_down");
        if (ok) {
            metal_graph_debug_dump_tensor("ffn_shexp", metal_graph_shared_out(g), DS4_N_EMBD, il, pos);
        }
        if (ok && keep_ffn_out) {
            ok = metal_graph_ensure_ffn_out(g) &&
                 ds4_gpu_add_tensor(metal_graph_ffn_out(g), metal_graph_shared_out(g), metal_graph_routed_out(g), DS4_N_EMBD) != 0;
        }
        if (ok && keep_ffn_out) {
            metal_graph_debug_dump_tensor("ffn_out", metal_graph_ffn_out(g), DS4_N_EMBD, il, pos);
        }
        if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
            ok = metal_graph_apply_directional_steering_ffn(g, metal_graph_ffn_out(g), il, 1);
        }
        if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
            ok = ds4_gpu_hc_expand_tensor(metal_graph_after_ffn_hc(g),
                                            metal_graph_ffn_out(g),
                                            metal_graph_after_attn_hc(g),
                                            metal_graph_hc_post(g),
                                            metal_graph_hc_comb(g),
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
        } else if (ok && !fuse_shared_down_hc) {
            ok = ds4_gpu_hc_expand_add_split_tensor(metal_graph_after_ffn_hc(g),
                                                      metal_graph_routed_out(g),
                                                      metal_graph_shared_out(g),
                                                      metal_graph_after_attn_hc(g),
                                                      metal_graph_hc_split(g),
                                                      DS4_N_EMBD,
                                                      DS4_N_HC) != 0;
        }
        DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_post");
        if (ok) {
            metal_graph_debug_dump_tensor("hc_ffn_post", metal_graph_after_ffn_hc(g), hc_dim, il, pos);
        }
        return ok;
    }
    /* Under the TP split the routed experts run after the shared expert so
     * the sum6 kernel can fold the shared partial and write the slab slot
     * directly (no separate local add). */
    const bool tp_fold_ffn = tp_split_shared &&
                             !keep_ffn_out &&
                             !metal_graph_directional_steering_ffn_enabled(g);
    bool parallel_full_ffn =
        ok && parallel_full_ffn_eligible &&
        router_shared_done == 0 && router_only_done != 0 &&
        fuse_shared_down_hc;
    if (parallel_full_ffn) {
#if defined(__APPLE__)
        parallel_full_ffn =
            ds4_gpu_parallel_ffn_start(
                    metal_graph_shared_gate(g),
                    metal_graph_shared_up(g),
                    metal_graph_shared_mid(g),
                    metal_graph_shared_out(g),
                    model->map,
                    model->size,
                    layer->ffn_gate_shexp->abs_offset,
                    layer->ffn_up_shexp->abs_offset,
                    layer->ffn_down_shexp->abs_offset,
                    DS4_N_EMBD,
                    shared_dim,
                    metal_graph_ffn_norm(g),
                    DS4_SWIGLU_CLAMP_EXP) != 0;
#else
        parallel_full_ffn = false;
#endif
    }
    if (ok && !tp_fold_ffn && !cuda_tp_moe) ok = ds4_gpu_routed_moe_one_tensor(metal_graph_routed_out(g),
                                                 metal_graph_routed_gate(g),
                                                 metal_graph_routed_up(g),
                                                 metal_graph_routed_mid(g),
                                                 metal_graph_routed_down(g),
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
                                                 metal_graph_router_selected(g), metal_graph_router_weights(g),
                                                 DS4_N_EXPERT,
                                                 DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, metal_graph_ffn_norm(g),
                                                 NULL,
                                                 il,
                                                 false) != 0;
    if (!ok && parallel_full_ffn) {
#if defined(__APPLE__)
        ds4_gpu_parallel_ffn_abort();
#endif
    }
    DS4_METAL_PROFILE_DECODE_STAGE("routed_moe");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", metal_graph_routed_gate(g),
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
        metal_graph_debug_dump_tensor("ffn_moe_up_clamped", metal_graph_routed_up(g),
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", metal_graph_routed_mid(g),
                                      (uint64_t)DS4_N_EXPERT_USED * down_in_dim, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_down", metal_graph_routed_down(g),
                                      (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_out", metal_graph_routed_out(g), DS4_N_EMBD, il, pos);
    }
    if (phase == METAL_DECODE_LAYER_TO_SHARED_MID ||
        phase == METAL_DECODE_LAYER_FROM_QA_KV_RAW_TO_SHARED_MID) {
        if (parallel_full_ffn) {
    #if defined(__APPLE__)
        ds4_gpu_parallel_ffn_abort();
#endif
        }
        return ok;
    }
    if (parallel_full_ffn) {
        /* Gate/up/SwiGLU was encoded as the first independent dispatch in
         * the concurrent FFN encoder. */
    } else if (ok && tp_split_shared) {
        /* Shared expert lane slice: the fused gate/up/swiglu kernel covers
         * this rank's half of the intermediate (row slicing is pure offset
         * math), compact at the buffer base; the down k-slice below turns
         * it into a partial output. */
        const uint32_t tp_half = shared_dim / 2;
        uint64_t shexp_row_bytes = 0;
        ok = metal_graph_dense_quant_row_bytes(layer->ffn_gate_shexp,
                                               DS4_N_EMBD,
                                               &shexp_row_bytes) &&
             layer->ffn_gate_shexp->type == layer->ffn_up_shexp->type;
        const uint64_t tp_lane_off = (uint64_t)g->tp_rank * tp_half * shexp_row_bytes;
        ok = ok && (tp_half % 32u) == 0;
        if (!ok) {
            fprintf(stderr, "ds4: TP shared expert width %u is not sliceable\n", shared_dim);
        }
        if (ok && layer->ffn_gate_shexp->type == DS4_TENSOR_Q8_0) {
            ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(metal_graph_shared_gate(g),
                                                           metal_graph_shared_up(g),
                                                           metal_graph_shared_mid(g),
                                                           model->map,
                                                           model->size,
                                                           layer->ffn_gate_shexp->abs_offset + tp_lane_off,
                                                           layer->ffn_up_shexp->abs_offset + tp_lane_off,
                                                           DS4_N_EMBD,
                                                           tp_half,
                                                           metal_graph_ffn_norm(g),
                                                           DS4_SWIGLU_CLAMP_EXP) != 0;
        } else if (ok) {
            ok = metal_graph_matmul_dense_quant_abs(metal_graph_shared_gate(g),
                                                    model,
                                                    layer->ffn_gate_shexp,
                                                    layer->ffn_gate_shexp->abs_offset + tp_lane_off,
                                                    DS4_N_EMBD,
                                                    tp_half,
                                                    metal_graph_ffn_norm(g),
                                                    1);
            if (ok) ok = metal_graph_matmul_dense_quant_abs(metal_graph_shared_up(g),
                                                            model,
                                                            layer->ffn_up_shexp,
                                                            layer->ffn_up_shexp->abs_offset + tp_lane_off,
                                                            DS4_N_EMBD,
                                                            tp_half,
                                                            metal_graph_ffn_norm(g),
                                                            1);
            if (ok) ok = ds4_gpu_swiglu_tensor(metal_graph_shared_mid(g), metal_graph_shared_gate(g), metal_graph_shared_up(g),
                                               tp_half, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
        }
    } else if (ok && fuse_shared_gate_up && router_shared_done == 0) {
        ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(metal_graph_shared_gate(g),
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
    } else if (router_shared_done == 0) {
        if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_gate(g),
                                                           model,
                                                           layer->ffn_gate_shexp,
                                                           DS4_N_EMBD,
                                                           shared_dim,
                                                           metal_graph_ffn_norm(g),
                                                           1);
        if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_up(g),
                                                           model,
                                                           layer->ffn_up_shexp,
                                                           DS4_N_EMBD,
                                                           shared_dim,
                                                           metal_graph_ffn_norm(g),
                                                           1);
        if (ok) ok = ds4_gpu_swiglu_tensor(metal_graph_shared_mid(g), metal_graph_shared_gate(g), metal_graph_shared_up(g),
                                           shared_dim, DS4_SWIGLU_CLAMP_EXP, 1.0f) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("shared_gate_up");
    if (ok && cuda_tp_ep_reduce_deferred) {
        ok = metal_graph_cuda_tp_ep_finish_reduce(
                g,
                cuda_tp_home_tier,
                cuda_tp_partner_tier,
                cuda_tp_ep_direct_return,
                cuda_tp_ep_return_bytes,
                !cuda_tp_ep_fused_hc_reduce);
    }
    if (ok && cuda_tp_moe_peer_copy_deferred) {
        ok = ds4_gpu_tensor_copy_xdev(
                g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                g->routed_out_by_tier[cuda_tp_partner_tier],
                (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
        cuda_tp_moe_peer_tmp = ok;
    }
    if (ok && cuda_tp_shared_fold) {
        bool switched_to_partner = false;
        ok = ds4_gpu_set_current_device(cuda_tp_partner_tier) == 0;
        switched_to_partner = ok;
        if (ok) {
            ok = ds4_gpu_add_tensor(
                    g->routed_out_by_tier[cuda_tp_partner_tier],
                    g->routed_out_by_tier[cuda_tp_partner_tier],
                    g->shared_out_by_tier[cuda_tp_partner_tier],
                    DS4_N_EMBD) != 0;
        }
        if (switched_to_partner &&
            ds4_gpu_set_current_device(cuda_tp_home_tier) != 0) {
            ok = false;
        }
        if (ok) {
            ok = ds4_gpu_add_tensor(metal_graph_routed_out(g),
                                    metal_graph_routed_out(g),
                                    metal_graph_shared_out(g),
                                    DS4_N_EMBD) != 0;
        }
        if (ok) {
            ok = ds4_gpu_tensor_copy_xdev(
                    g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                    g->routed_out_by_tier[cuda_tp_partner_tier],
                    (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
            cuda_tp_shared_fold_peer_tmp = ok;
        }
        if (ok) {
            ok = cuda_tp_shared_fold_peer_tmp &&
                 ds4_gpu_hc_expand_add_split_tensor(
                         metal_graph_after_ffn_hc(g),
                         metal_graph_routed_out(g),
                         g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                         metal_graph_after_attn_hc(g),
                         metal_graph_hc_split(g),
                         DS4_N_EMBD,
                         DS4_N_HC) != 0;
        }
    } else if (ok && cuda_tp_shared) {
        /* shared_out already contains the reduced local and partner partials. */
    } else if (ok && cuda_tp_ep_fused_hc_reduce) {
        ok = ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor(
                metal_graph_after_ffn_hc(g),
                metal_graph_shared_out(g),
                model->map,
                model->size,
                layer->ffn_down_shexp->abs_offset,
                shared_dim,
                DS4_N_EMBD,
                metal_graph_shared_mid(g),
                metal_graph_routed_down(g),
                g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                metal_graph_router_selected(g),
                DS4_N_EXPERT / 2u,
                metal_graph_after_attn_hc(g),
                metal_graph_hc_split(g),
                DS4_N_EMBD,
                DS4_N_HC) != 0;
    } else if (parallel_full_ffn) {
#if defined(__APPLE__)
        const bool parallel_joined =
            ds4_gpu_parallel_ffn_finish() != 0;
        ok = ok && parallel_joined;
#else
        ok = false;
#endif
        if (ok) {
            ok = ds4_gpu_hc_expand_add_split_tensor(
                    metal_graph_after_ffn_hc(g),
                    metal_graph_routed_out(g),
                    metal_graph_shared_out(g),
                    metal_graph_after_attn_hc(g),
                    metal_graph_hc_split(g),
                    DS4_N_EMBD,
                    DS4_N_HC) != 0;
        }
    } else if (ok && fuse_shared_down_hc) {
        if (cuda_tp_moe_peer_tmp) {
            ok = ds4_gpu_shared_down_hc_expand_add_q8_0_tensor(
                    metal_graph_after_ffn_hc(g),
                    metal_graph_shared_out(g),
                    model->map,
                    model->size,
                    layer->ffn_down_shexp->abs_offset,
                    shared_dim,
                    DS4_N_EMBD,
                    metal_graph_shared_mid(g),
                    metal_graph_routed_out(g),
                    g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                    metal_graph_after_attn_hc(g),
                    metal_graph_hc_split(g),
                    DS4_N_EMBD,
                    DS4_N_HC) != 0;
        } else {
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
        }
    } else if (ok && tp_split_shared) {
        ok = metal_graph_matmul_dense_quant_kslice(metal_graph_shared_out(g),
                                                   model,
                                                   layer->ffn_down_shexp,
                                                   shared_dim,
                                                   (uint64_t)g->tp_rank * (shared_dim / 2),
                                                   shared_dim / 2,
                                                   DS4_N_EMBD,
                                                   metal_graph_shared_mid(g),
                                                   0);
    } else if (ok) {
        ok = metal_graph_matmul_dense_quant_tensor(metal_graph_shared_out(g),
                                                   model,
                                                   layer->ffn_down_shexp,
                                                   shared_dim,
                                                   DS4_N_EMBD,
                                                   metal_graph_shared_mid(g),
                                                   1);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("shared_down");
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_shexp", metal_graph_shared_out(g), DS4_N_EMBD, il, pos);
    }
    if (ok && tp_fold_ffn) {
        ok = ds4_gpu_routed_moe_one_tensor(
                g->tp_out[il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_FFN],
                metal_graph_routed_gate(g),
                metal_graph_routed_up(g),
                metal_graph_routed_mid(g),
                metal_graph_routed_down(g),
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
                metal_graph_router_selected(g), metal_graph_router_weights(g),
                DS4_N_EXPERT,
                DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, metal_graph_ffn_norm(g),
                metal_graph_shared_out(g),
                il,
                false) != 0;
        DS4_METAL_PROFILE_DECODE_STAGE("routed_moe_folded");
    }
    ds4_gpu_tensor *tp_ffn_a = NULL;    /* rank0/rank1 partials consumed */
    ds4_gpu_tensor *tp_ffn_b = NULL;    /* directly by the HC expand */
    if (ok && g->tp_world == 2) {
        /* Gate FFN: local partial = shared expert + owned routed experts.
         * The HC expand below already sums two block vectors, so after the
         * exchange the two rank partials feed it directly (canonical rank
         * order) with no separate combine dispatch.  The paths that need
         * the materialized sum (ffn_out consumers) still builds it in
         * routed_out. */
        const uint32_t tp_slot = il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_FFN;
        if (!tp_fold_ffn) {
            ok = ds4_gpu_add_tensor(g->tp_out[tp_slot], metal_graph_shared_out(g), metal_graph_routed_out(g),
                                    DS4_N_EMBD) != 0;
        }
        if (ok) ok = ds4_gpu_tp_gate_encode(il, DS4_TP_GATE_FFN) != 0;
        if (ok) {
            ds4_gpu_tensor *first = g->tp_rank == 0 ? g->tp_out[tp_slot] : g->tp_in[tp_slot];
            ds4_gpu_tensor *second = g->tp_rank == 0 ? g->tp_in[tp_slot] : g->tp_out[tp_slot];
            if (keep_ffn_out || metal_graph_directional_steering_ffn_enabled(g)) {
                ok = ds4_gpu_add_tensor(metal_graph_routed_out(g), first, second, DS4_N_EMBD) != 0;
            } else {
                tp_ffn_a = first;
                tp_ffn_b = second;
            }
        }
    }
    if (ok && keep_ffn_out) {
        ok = metal_graph_ensure_ffn_out(g) &&
             ds4_gpu_add_tensor(metal_graph_ffn_out(g),
                                g->tp_world == 2 ? g->tp_zero : metal_graph_shared_out(g),
                                metal_graph_routed_out(g), DS4_N_EMBD) != 0;
    }
    if (ok && keep_ffn_out) {
        metal_graph_debug_dump_tensor("ffn_out", metal_graph_ffn_out(g), DS4_N_EMBD, il, pos);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_ffn(g, metal_graph_ffn_out(g), il, 1);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_tensor(metal_graph_after_ffn_hc(g),
                                        metal_graph_ffn_out(g),
                                        metal_graph_after_attn_hc(g),
                                        metal_graph_hc_post(g),
                                        metal_graph_hc_comb(g),
                                        DS4_N_EMBD,
                                        DS4_N_HC) != 0;
    } else if (ok && !cuda_tp_shared_fold && !fuse_shared_down_hc) {
        ok = ds4_gpu_hc_expand_add_split_tensor(metal_graph_after_ffn_hc(g),
                                                  tp_ffn_a ? tp_ffn_a : metal_graph_routed_out(g),
                                                  tp_ffn_a ? tp_ffn_b :
                                                  (g->tp_world == 2 ? g->tp_zero : metal_graph_shared_out(g)),
                                                  metal_graph_after_attn_hc(g),
                                                  metal_graph_hc_split(g),
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_post");
#undef DS4_METAL_PROFILE_DECODE_STAGE
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_post", metal_graph_after_ffn_hc(g), hc_dim, il, pos);
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

