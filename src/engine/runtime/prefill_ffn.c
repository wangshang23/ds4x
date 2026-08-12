#include "engine_internal.h"

/* Prefill Ffn module. */
#ifndef DS4_NO_GPU
/* Encode the multi-token prefill/verification FFN half. */
bool metal_graph_encode_layer_ffn_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    const bool layer_stage_profile = metal_graph_layer_stage_profile_enabled(il);
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_FFN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("ffn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            metal_graph_batch_hc_mix(g), 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            metal_graph_batch_hc_split(g), 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *ffn_cur_view = ds4_gpu_tensor_view(
            metal_graph_batch_ffn_cur(g), 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *next_hc_view = ds4_gpu_tensor_view(
            metal_graph_batch_next_hc(g), 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && ffn_cur_view && next_hc_view;
    const bool fuse_hc_norm = n_tokens > 1 &&
                              DS4_N_HC == 4 &&
                              !metal_graph_use_reference_hc_decode() &&
                              metal_graph_enable_batch_hc_norm_fusion();
    if (ok) ok = metal_graph_hc_rms_scale_project(hc_mix_view,
                                                    metal_graph_batch_flat_hc(g),
                                                    model,
                                                    layer->hc_ffn_fn,
                                                    metal_graph_batch_after_attn_hc(g),
                                                    hc_dim,
                                                    n_tokens);
    if (metal_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_ffn_scale->abs_offset,
                                                        layer->hc_ffn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(ffn_cur_view,
                                                            metal_graph_batch_after_attn_hc(g),
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else if (fuse_hc_norm) {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(ffn_cur_view,
                                                                 metal_graph_batch_ffn_norm(g),
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 metal_graph_batch_after_attn_hc(g),
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
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            metal_graph_batch_after_attn_hc(g),
                                                            model->map,
                                                            model->size,
                                                            layer->hc_ffn_scale->abs_offset,
                                                            layer->hc_ffn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre", metal_graph_batch_ffn_cur(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_pre");
    if (ok && !fuse_hc_norm) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_ffn_norm(g),
                                                  metal_graph_batch_ffn_cur(g),
                                                  model->map,
                                                  model->size,
                                                  layer->ffn_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  n_tokens,
                                                  DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_norm", metal_graph_batch_ffn_norm(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("norm");
    if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_batch_router_logits(g),
                                                 model,
                                                 layer->ffn_gate_inp,
                                                 DS4_N_EMBD,
                                                 DS4_N_EXPERT,
                                                 metal_graph_batch_ffn_norm(g),
                                                 n_tokens);

    ds4_gpu_tensor *router_tokens = NULL;
    if (ok) {
        router_tokens = ds4_gpu_tensor_view(metal_graph_prefill_tokens(g),
                                              (uint64_t)g->batch_token_offset * sizeof(int32_t),
                                              (uint64_t)n_tokens * sizeof(int32_t));
        ok = router_tokens != NULL;
    }
    if (ok) ok = ds4_gpu_router_select_batch_tensor(metal_graph_batch_router_selected(g),
                                                      metal_graph_batch_router_weights(g),
                                                      metal_graph_batch_router_probs(g),
                                                      model->map,
                                                      model->size,
                                                      layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                      0,
                                                      0,
                                                      layer->ffn_exp_probs_b != NULL,
                                                      layer->ffn_gate_tid2eid != NULL,
                                                      metal_graph_batch_router_logits(g),
                                                      metal_graph_prefill_tokens(g),
                                                      DS4_N_EXPERT,
                                                      DS4_N_EXPERT_USED,
                                                      DS4_EXPERT_WEIGHT_SCALE,
                                                      n_tokens) != 0;
    ds4_gpu_tensor_free(router_tokens);
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_logits", metal_graph_batch_router_logits(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_probs", metal_graph_batch_router_probs(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        metal_graph_debug_dump_i32_tensor("ffn_moe_topk", metal_graph_batch_router_selected(g),
                                          (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", metal_graph_batch_router_weights(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("router");

    const bool keep_ffn_out = metal_graph_needs_ffn_out(g, il, pos0);
    bool shared_down_f16 = false;

#define DS4_METAL_TRY_SHARED_DOWN_F16() do { \
        if (ok && !keep_ffn_out && \
            !metal_graph_debug_wants("ffn_shexp", il, pos0)) { \
            shared_down_f16 = ds4_gpu_matmul_q8_0_f16_out_tensor(g->batch_q_half, \
                                                                 model->map, \
                                                                 model->size, \
                                                                 layer->ffn_down_shexp->abs_offset, \
                                                                 shared_dim, \
                                                                 DS4_N_EMBD, \
                                                                 metal_graph_batch_shared_mid(g), \
                                                                 n_tokens) != 0; \
        } \
    } while (0)

#define DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT() do { \
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("shared_gate", \
                                                          il, \
                                                          pos0, \
                                                          metal_graph_batch_shared_gate(g), \
                                                          model, \
                                                          layer->ffn_gate_shexp, \
                                                          DS4_N_EMBD, \
                                                          shared_dim, \
                                                          metal_graph_batch_ffn_norm(g), \
                                                          n_tokens); \
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("shared_up", \
                                                          il, \
                                                          pos0, \
                                                          metal_graph_batch_shared_up(g), \
                                                          model, \
                                                          layer->ffn_up_shexp, \
                                                          DS4_N_EMBD, \
                                                          shared_dim, \
                                                          metal_graph_batch_ffn_norm(g), \
                                                          n_tokens); \
        DS4_METAL_PROFILE_FFN_STAGE("shared_gate_up"); \
        if (ok) ok = ds4_gpu_swiglu_tensor(metal_graph_batch_shared_mid(g), \
                                             metal_graph_batch_shared_gate(g), \
                                             metal_graph_batch_shared_up(g), \
                                             (uint32_t)((uint64_t)n_tokens * shared_dim), \
                                             DS4_SWIGLU_CLAMP_EXP, \
                                             1.0f) != 0; \
        DS4_METAL_TRY_SHARED_DOWN_F16(); \
        if (ok && !shared_down_f16) ok = metal_graph_matmul_q8_0_named_tensor("shared_down", \
                                                                              il, \
                                                                              pos0, \
                                                                              metal_graph_batch_shared_out(g), \
                                                                              model, \
                                                                              layer->ffn_down_shexp, \
                                                                              shared_dim, \
                                                                              DS4_N_EMBD, \
                                                                              metal_graph_batch_shared_mid(g), \
                                                                              n_tokens); \
        DS4_METAL_PROFILE_FFN_STAGE("shared_down"); \
        if (ok && !shared_down_f16) { \
            metal_graph_debug_dump_tensor("ffn_shexp", metal_graph_batch_shared_out(g), \
                                          (uint64_t)n_tokens * DS4_N_EMBD, il, pos0); \
        } \
    } while (0)

    bool shared_done = false;
    if (ok) {
        ok = ds4_gpu_routed_moe_batch_tensor(metal_graph_batch_routed_out(g),
                                               metal_graph_batch_routed_gate(g),
                                               metal_graph_batch_routed_up(g),
                                               metal_graph_batch_routed_mid(g),
                                               metal_graph_batch_routed_down(g),
                                               model->map,
                                               model->size,
                                               layer->ffn_gate_exps->abs_offset,
                                               layer->ffn_up_exps->abs_offset,
                                               layer->ffn_down_exps->abs_offset,
                                               layer->ffn_gate_exps->type,
                                               layer->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               (uint32_t)expert_in_dim,
                                               (uint32_t)down_in_dim,
                                               (uint32_t)routed_out_dim,
                                               metal_graph_batch_router_selected(g),
                                               metal_graph_batch_router_weights(g),
                                               DS4_N_EXPERT,
                                               DS4_N_EXPERT_USED,
                                               DS4_SWIGLU_CLAMP_EXP,
                                               metal_graph_batch_ffn_norm(g),
                                               il,
                                               n_tokens,
                                               &g->batch_routed_mid_is_f16,
                                               false) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", metal_graph_batch_routed_gate(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_up_clamped", metal_graph_batch_routed_up(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
    }
    if (ok) {
        const uint64_t routed_mid_elems = (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim;
        if (g->batch_routed_mid_is_f16) {
            metal_graph_debug_dump_f16_tensor("ffn_moe_weighted_swiglu", metal_graph_batch_routed_mid(g),
                                              routed_mid_elems, il, pos0);
        } else {
            metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", metal_graph_batch_routed_mid(g),
                                          routed_mid_elems, il, pos0);
        }
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_down", metal_graph_batch_routed_down(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos0);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_out", metal_graph_batch_routed_out(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("routed_moe");
    if (!shared_done) {
        DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT();
    }
#undef DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT
#undef DS4_METAL_TRY_SHARED_DOWN_F16

    if (ok && keep_ffn_out) {
        ok = metal_graph_ensure_batch_ffn_out(g) &&
             ds4_gpu_add_tensor(metal_graph_batch_ffn_out(g),
                                  metal_graph_batch_shared_out(g),
                                  metal_graph_batch_routed_out(g),
                                  (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
    }
    if (ok && keep_ffn_out) {
        metal_graph_debug_dump_tensor("ffn_out", metal_graph_batch_ffn_out(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_ffn(g, metal_graph_batch_ffn_out(g), il, n_tokens);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_split_tensor(next_hc_view,
                                              metal_graph_batch_ffn_out(g),
                                              metal_graph_batch_after_attn_hc(g),
                                              hc_split_view,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    }
    else if (ok && shared_down_f16) {
        ok = ds4_gpu_hc_expand_add_split_half_add_tensor(next_hc_view,
                                                         metal_graph_batch_routed_out(g),
                                                         g->batch_q_half,
                                                         metal_graph_batch_after_attn_hc(g),
                                                         hc_split_view,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC) != 0;
    }
    else if (ok) {
        ok = ds4_gpu_hc_expand_add_split_tensor(next_hc_view,
                                                  metal_graph_batch_routed_out(g),
                                                  metal_graph_batch_shared_out(g),
                                                  metal_graph_batch_after_attn_hc(g),
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_post");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_post", metal_graph_batch_next_hc(g),
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_post");
    ds4_gpu_tensor_free(next_hc_view);
    ds4_gpu_tensor_free(ffn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
#undef DS4_METAL_PROFILE_FFN_STAGE
    return ok;
}

/* Encode one complete layer for prefill by chaining attention and FFN batches. */
bool metal_graph_encode_layer_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    bool ok = metal_graph_layer_stage_profile_start(il);
    if (ok) {
        ok = metal_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
    }
    if (!ok) {
        fprintf(stderr, "ds4: gpu layer %u attention batch encode failed\n", il);
    }
    if (ok) {
        ok = metal_graph_encode_layer_ffn_batch(g, model, layer, il, pos0,
                                                 n_tokens);
        if (!ok) {
            fprintf(stderr, "ds4: gpu layer %u ffn batch encode failed\n", il);
        }
    }
    if (ok) {
        ds4_gpu_tensor *tmp = metal_graph_batch_cur_hc(g);
        g->batch_cur_hc_by_tier[g->active_tier] = metal_graph_batch_next_hc(g);
        g->batch_next_hc_by_tier[g->active_tier] = tmp;
    }
    return ok;
}

 /* Execute one Metal decode token and read back logits. */
bool metal_graph_eval_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    const bool profile =
        glm_graph_env_present("DS4_ROCM_GRAPH_TOKEN_PROFILE",
                              "DS4_METAL_GRAPH_TOKEN_PROFILE");
    const bool throttle = graph_power_throttle_enabled(g);
    const double t0 = (profile || throttle) ? now_sec() : 0.0;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_token_raw_swa(g, model, weights, token, pos, logits != NULL, true);
    const double t_encoded = (profile || throttle) ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_done = (profile || throttle) ? now_sec() : 0.0;

    if (ok && logits) {
        ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    const double t_read = (profile || throttle) ? now_sec() : 0.0;
    if (profile) {
        fprintf(stderr,
                "ds4: metal graph token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                pos,
                (t_encoded - t0) * 1000.0,
                (t_done - t_encoded) * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
    if (ok) graph_power_note_decode_token(g, t_read - t0);
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after graph eval failure also failed\n");
        }
    }
    return ok;
}

#endif /* !DS4_NO_GPU */
