#include "engine_internal.h"

/* Verifier module. */
#ifndef DS4_NO_GPU
/* Greedy verifier helper.  Speculative decoding only needs the target model's
 * top token after most accepted draft rows; the full vocabulary row is needed
 * once, for the final committed state that normal sampling will continue from.
 * Keeping intermediate rows device-resident avoids turning verification into a
 * sequence of large CPU readbacks. */
bool dspark_stage0_weights_ready(
        const ds4_gpu_graph     *g,
        const ds4_dspark_weights *dw) {
    if (!g || !dw || dw->n_stages == 0 || dw->target_layer_count == 0 ||
        dw->target_layer_count != g->dspark_target_layer_count ||
        !g->dspark_target_hidden || !g->dspark_stage0_proj ||
        !g->dspark_main_x) {
        return false;
    }

    const ds4_dspark_stage_weights *stage0 = &dw->stage[0];
    const ds4_tensor *main_proj = stage0->main_proj;
    const ds4_tensor *main_norm = stage0->main_norm;
    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    return main_proj &&
           main_norm &&
           dspark_tensor_type_matches(main_proj->type, DS4_DSPARK_LAYOUT_DENSE) &&
           main_norm->type == DS4_TENSOR_F32 &&
           main_proj->ndim == 2 &&
           main_proj->dim[0] == in_dim &&
           main_proj->dim[1] == DS4_N_EMBD &&
           main_norm->ndim == 1 &&
           main_norm->dim[0] == DS4_N_EMBD;
}

bool metal_graph_eval_dspark_stage0(
        ds4_gpu_graph          *g,
        const ds4_model        *dspark_model,
        const ds4_dspark_weights *dw) {
    if (!g || !dspark_model || !dw || !dspark_stage0_weights_ready(g, dw)) {
        return false;
    }

    const ds4_dspark_stage_weights *stage0 = &dw->stage[0];
    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = metal_graph_matmul_plain_tensor(g->dspark_stage0_proj,
                                             dspark_model,
                                             stage0->main_proj,
                                             in_dim,
                                             DS4_N_EMBD,
                                             g->dspark_target_hidden,
                                             1);
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_tensor(g->dspark_main_x,
                                            g->dspark_stage0_proj,
                                            dspark_model->map,
                                            dspark_model->size,
                                            stage0->main_norm->abs_offset,
                                            DS4_N_EMBD,
                                            DS4_RMS_EPS) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) (void)ds4_gpu_synchronize();
    return ok;
}

static bool dspark_stage0_batch_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw,
        uint32_t                  n_tokens) {
    if (!dspark_stage0_weights_ready(g, dw) ||
        n_tokens == 0 ||
        n_tokens > g->prefill_cap ||
        !g->dspark_target_hidden_batch ||
        !metal_graph_batch_ffn_cur(g) ||
        !metal_graph_batch_ffn_norm(g) ||
        !metal_graph_batch_cur_hc(g)) {
        return false;
    }
    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    return ds4_gpu_tensor_bytes(metal_graph_batch_ffn_cur(g)) >=
               (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_ffn_norm(g)) >=
               (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_cur_hc(g)) >=
               (uint64_t)n_tokens * DS4_N_HC * DS4_N_EMBD * sizeof(float) &&
           in_dim <= SIZE_MAX / sizeof(float);
}

static bool metal_graph_pack_dspark_target_hidden_batch(
        ds4_gpu_graph     *g,
        const ds4_dspark_weights *dw,
        ds4_gpu_tensor    *packed,
        uint32_t           n_tokens) {
    if (!g || !dw || !packed || n_tokens == 0 ||
        n_tokens > g->prefill_cap ||
        dw->target_layer_count != g->dspark_target_layer_count) {
        return false;
    }

    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    const uint64_t packed_count = (uint64_t)n_tokens * in_dim;
    if (packed_count == 0 ||
        packed_count > (uint64_t)SIZE_MAX / sizeof(float)) {
        return false;
    }
    return ds4_gpu_pack_slot_rows_f32_tensor(packed,
                                             g->dspark_target_hidden_batch,
                                             n_tokens,
                                             DS4_N_EMBD,
                                             dw->target_layer_count,
                                             g->prefill_cap) != 0;
}

static bool metal_graph_eval_dspark_stage0_batch(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  n_tokens,
        bool                      commands_open) {
    if (!g || !dspark_model || !dw ||
        !dspark_stage0_batch_ready(g, dw, n_tokens)) {
        return false;
    }

    const ds4_dspark_stage_weights *stage0 = &dw->stage[0];
    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    const uint64_t packed_bytes = (uint64_t)n_tokens * in_dim * sizeof(float);
    bool packed_owned = false;
    ds4_gpu_tensor *packed = NULL;
    if (g->dspark_stage0_packed &&
        ds4_gpu_tensor_bytes(g->dspark_stage0_packed) >= packed_bytes) {
        packed = g->dspark_stage0_packed;
    } else {
        packed = ds4_gpu_tensor_alloc(packed_bytes);
        packed_owned = true;
    }
    if (!packed) return false;

    bool ok = metal_graph_pack_dspark_target_hidden_batch(g, dw, packed, n_tokens);
    if (ok && !commands_open) ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = metal_graph_matmul_plain_tensor(metal_graph_batch_ffn_cur(g),
                                             dspark_model,
                                             stage0->main_proj,
                                             in_dim,
                                             DS4_N_EMBD,
                                             packed,
                                             n_tokens);
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_ffn_norm(g),
                                                 metal_graph_batch_ffn_cur(g),
                                                 dspark_model->map,
                                                 dspark_model->size,
                                                 stage0->main_norm->abs_offset,
                                                 DS4_N_EMBD,
                                                 n_tokens,
                                                 DS4_RMS_EPS) != 0;
    }
    if (ok && !commands_open) ok = ds4_gpu_end_commands() != 0;
    if (!ok && !commands_open) (void)ds4_gpu_synchronize();
    if (packed_owned) ds4_gpu_tensor_free(packed);
    return ok;
}

bool dspark_draft_block_ready(
        const ds4_gpu_graph      *g,
        const ds4_weights        *base_weights,
        const ds4_dspark_weights *dw,
        int                       token) {
    if (!g || !base_weights || !dw || !base_weights->token_embd ||
        !g->dspark_draft_tokens || !g->dspark_draft_hc ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        g->dspark_block_size != dw->block_size ||
        !dw->has_noise_token_id) {
        return false;
    }
    const uint32_t n_vocab = (uint32_t)base_weights->token_embd->dim[1];
    return token >= 0 &&
           (uint32_t)token < n_vocab &&
           dw->noise_token_id < n_vocab;
}

bool dspark_stage_input_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw) {
    if (!g || !dw ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        g->dspark_block_size != dw->block_size ||
        !g->dspark_main_x || !g->dspark_draft_hc ||
        !g->dspark_target_hc || !g->dspark_stage_input_hc ||
        !g->dspark_position_ids) {
        return false;
    }
    if (dw->block_size == UINT32_MAX) return false;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t rows = (uint64_t)dw->block_size + 1u;
    return ds4_gpu_tensor_bytes(g->dspark_target_hc) >=
               hc_dim * sizeof(float) &&
           ds4_gpu_tensor_bytes(g->dspark_stage_input_hc) >=
               rows * hc_dim * sizeof(float) &&
           ds4_gpu_tensor_bytes(g->dspark_position_ids) >=
               rows * sizeof(int32_t);
}

bool dspark_stage_cache_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw) {
    if (!g || !dw ||
        dw->n_stages == 0 ||
        dw->n_stages > DS4_DSPARK_MAX_STAGES ||
        g->dspark_cache_cap == 0 ||
        !metal_graph_dspark_cache_current_window_valid(g)) {
        return false;
    }
    const uint64_t bytes =
        (uint64_t)g->dspark_cache_cap * DS4_N_HEAD_DIM * sizeof(float);
    for (uint32_t stage = 0; stage < dw->n_stages; stage++) {
        if (!g->dspark_raw_cache[stage] ||
            ds4_gpu_tensor_bytes(g->dspark_raw_cache[stage]) < bytes) {
            return false;
        }
    }
    return true;
}

bool dspark_noncausal_attention_probe_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw) {
    if (!g || !dw ||
        dw->n_stages == 0 ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        g->prefill_cap < dw->block_size + 1u ||
        !metal_graph_batch_q(g) || !metal_graph_batch_heads(g) ||
        !g->dspark_raw_cache[0]) {
        return false;
    }
    const ds4_layer_weights *block = &dw->stage[0].block;
    if (!block->attn_sinks) return false;

    const uint64_t rows = (uint64_t)dw->block_size + 1u;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    return ds4_gpu_tensor_bytes(metal_graph_batch_q(g)) >=
               rows * q_dim * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_heads(g)) >=
               rows * q_dim * sizeof(float) &&
           ds4_gpu_tensor_bytes(g->dspark_raw_cache[0]) >=
               rows * DS4_N_HEAD_DIM * sizeof(float);
}

bool metal_graph_probe_dspark_noncausal_attention(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw) {
    if (!g || !dspark_model || !dspark_noncausal_attention_probe_ready(g, dw)) {
        return false;
    }

    const uint32_t rows = dw->block_size + 1u;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const ds4_layer_weights *block = &dw->stage[0].block;
    bool ok = ds4_gpu_tensor_fill_f32(metal_graph_batch_q(g),
                                      0.0f,
                                      (uint64_t)rows * q_dim) != 0 &&
              ds4_gpu_tensor_fill_f32(g->dspark_raw_cache[0],
                                      0.0f,
                                      (uint64_t)rows * DS4_N_HEAD_DIM) != 0;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
                metal_graph_batch_heads(g),
                dspark_model->map,
                dspark_model->size,
                block->attn_sinks->abs_offset,
                metal_graph_batch_q(g),
                g->dspark_raw_cache[0],
                rows,
                rows,
                rows,
                0,
                DS4_N_HEAD,
                DS4_N_HEAD_DIM) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) (void)ds4_gpu_synchronize();
    return ok;
}

bool metal_graph_prepare_dspark_setup_block(
        ds4_gpu_graph            *g,
        const ds4_model          *base_model,
        const ds4_weights        *base_weights,
        const ds4_dspark_weights *dw,
        int                       token,
        uint32_t                  pos) {
    if (!g || !base_model ||
        !dspark_draft_block_ready(g, base_weights, dw, token) ||
        !dspark_stage_input_ready(g, dw)) {
        return false;
    }
    if (pos > (uint32_t)INT32_MAX ||
        dw->block_size > (uint32_t)INT32_MAX ||
        pos > (uint32_t)INT32_MAX - dw->block_size) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = hc_dim * sizeof(float);
    int32_t positions[DS4_DSPARK_MAX_BLOCK_SIZE + 1u];
    positions[0] = (int32_t)pos;
    for (uint32_t i = 0; i < dw->block_size; i++) {
        positions[i + 1u] = (int32_t)(pos + i);
    }
    int32_t ids[DS4_DSPARK_MAX_BLOCK_SIZE];
    ids[0] = (int32_t)token;
    for (uint32_t i = 1; i < dw->block_size; i++) {
        ids[i] = (int32_t)dw->noise_token_id;
    }

    bool ok = ds4_gpu_tensor_write(g->dspark_draft_tokens,
                                   0,
                                   ids,
                                   (uint64_t)dw->block_size * sizeof(ids[0])) != 0 &&
              ds4_gpu_tensor_write(g->dspark_position_ids,
                                   0,
                                   positions,
                                   ((uint64_t)dw->block_size + 1u) *
                                       sizeof(positions[0])) != 0;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_embed_tokens_hc_tensor(g->dspark_draft_hc,
                                            g->dspark_draft_tokens,
                                            base_model->map,
                                            base_model->size,
                                            base_weights->token_embd->abs_offset,
                                            (uint32_t)base_weights->token_embd->dim[1],
                                            dw->block_size,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
    }
    if (ok) {
        ok = ds4_gpu_repeat_hc_tensor(g->dspark_target_hc,
                                      g->dspark_main_x,
                                      DS4_N_EMBD,
                                      DS4_N_HC) != 0;
    }
    if (ok) {
        ok = ds4_gpu_tensor_copy(g->dspark_stage_input_hc,
                                 0,
                                 g->dspark_target_hc,
                                 0,
                                 hc_bytes) != 0;
    }
    if (ok) {
        ok = ds4_gpu_tensor_copy(g->dspark_stage_input_hc,
                                 hc_bytes,
                                 g->dspark_draft_hc,
                                 0,
                                 (uint64_t)dw->block_size * hc_bytes) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) (void)ds4_gpu_synchronize();
    return ok;
}

bool metal_graph_prepare_dspark_stage0_setup_block(
        ds4_gpu_graph            *g,
        const ds4_model          *base_model,
        const ds4_weights        *base_weights,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        int                       token,
        uint32_t                  pos) {
    if (!g || !base_model || !dspark_model ||
        !dspark_stage0_weights_ready(g, dw) ||
        !dspark_draft_block_ready(g, base_weights, dw, token) ||
        !dspark_stage_input_ready(g, dw)) {
        return false;
    }
    if (pos > (uint32_t)INT32_MAX ||
        dw->block_size > (uint32_t)INT32_MAX ||
        pos > (uint32_t)INT32_MAX - dw->block_size) {
        return false;
    }

    const ds4_dspark_stage_weights *stage0 = &dw->stage[0];
    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = hc_dim * sizeof(float);
    int32_t positions[DS4_DSPARK_MAX_BLOCK_SIZE + 1u];
    positions[0] = (int32_t)pos;
    for (uint32_t i = 0; i < dw->block_size; i++) {
        positions[i + 1u] = (int32_t)(pos + i);
    }
    int32_t ids[DS4_DSPARK_MAX_BLOCK_SIZE];
    ids[0] = (int32_t)token;
    for (uint32_t i = 1; i < dw->block_size; i++) {
        ids[i] = (int32_t)dw->noise_token_id;
    }

    /* DS4_DSPARK_PROP_PROFILE=1: break the setup block into phases to
     * localize the TP-only prop_setup inflation (26ms vs 1.3ms single). */
    const bool prop_profile = getenv("DS4_DSPARK_PROP_PROFILE") != NULL;
    const double pp_t0 = prop_profile ? now_sec() : 0.0;
    bool ok = ds4_gpu_tensor_write(g->dspark_draft_tokens,
                                   0,
                                   ids,
                                   (uint64_t)dw->block_size * sizeof(ids[0])) != 0 &&
              ds4_gpu_tensor_write(g->dspark_position_ids,
                                   0,
                                   positions,
                                   ((uint64_t)dw->block_size + 1u) *
                                       sizeof(positions[0])) != 0;
    const double pp_t1 = prop_profile ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    const double pp_t2 = prop_profile ? now_sec() : 0.0;
    if (ok) {
        ok = metal_graph_matmul_plain_tensor(g->dspark_stage0_proj,
                                             dspark_model,
                                             stage0->main_proj,
                                             in_dim,
                                             DS4_N_EMBD,
                                             g->dspark_target_hidden,
                                             1);
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_tensor(g->dspark_main_x,
                                            g->dspark_stage0_proj,
                                            dspark_model->map,
                                            dspark_model->size,
                                            stage0->main_norm->abs_offset,
                                            DS4_N_EMBD,
                                            DS4_RMS_EPS) != 0;
    }
    if (ok) {
        ok = ds4_gpu_embed_tokens_hc_tensor(g->dspark_draft_hc,
                                            g->dspark_draft_tokens,
                                            base_model->map,
                                            base_model->size,
                                            base_weights->token_embd->abs_offset,
                                            (uint32_t)base_weights->token_embd->dim[1],
                                            dw->block_size,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
    }
    if (ok) {
        ok = ds4_gpu_repeat_hc_tensor(g->dspark_target_hc,
                                      g->dspark_main_x,
                                      DS4_N_EMBD,
                                      DS4_N_HC) != 0;
    }
    if (ok) {
        ok = ds4_gpu_tensor_copy(g->dspark_stage_input_hc,
                                 0,
                                 g->dspark_target_hc,
                                 0,
                                 hc_bytes) != 0;
    }
    if (ok) {
        ok = ds4_gpu_tensor_copy(g->dspark_stage_input_hc,
                                 hc_bytes,
                                 g->dspark_draft_hc,
                                 0,
                                 (uint64_t)dw->block_size * hc_bytes) != 0;
    }
    const double pp_t3 = prop_profile ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (prop_profile) {
        const double pp_t4 = now_sec();
        fprintf(stderr,
                "ds4: DSpark prop-setup phases: writes=%.3fms begin=%.3fms "
                "encode=%.3fms end/wait=%.3fms\n",
                (pp_t1 - pp_t0) * 1000.0,
                (pp_t2 - pp_t1) * 1000.0,
                (pp_t3 - pp_t2) * 1000.0,
                (pp_t4 - pp_t3) * 1000.0);
    }
    if (!ok) (void)ds4_gpu_synchronize();
    return ok;
}

bool dspark_stage_block_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw,
        uint32_t                  stage) {
    if (!g || !dw ||
        stage >= dw->n_stages ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        g->prefill_cap < dw->block_size + 1u ||
        !g->dspark_stage_output_hc ||
        !dspark_stage_input_ready(g, dw) ||
        !dspark_stage_cache_ready(g, dw)) {
        return false;
    }

    const ds4_layer_weights *l = &dw->stage[stage].block;
    if (!l->hc_attn_fn || !l->hc_attn_scale || !l->hc_attn_base ||
        !l->attn_norm || !l->attn_q_a || !l->attn_q_a_norm ||
        !l->attn_q_b || !l->attn_kv || !l->attn_kv_a_norm ||
        !l->attn_sinks || !l->attn_output_a || !l->attn_output_b ||
        !l->hc_ffn_fn || !l->hc_ffn_scale || !l->hc_ffn_base ||
        !l->ffn_norm || !l->ffn_gate_inp || !l->ffn_exp_probs_b ||
        !l->ffn_gate_exps || !l->ffn_up_exps || !l->ffn_down_exps ||
        !l->ffn_gate_shexp || !l->ffn_up_shexp || !l->ffn_down_shexp) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t rows = (uint64_t)dw->block_size + 1u;
    const uint64_t draft = dw->block_size;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim =
        (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);

    return
        dspark_tensor_type_matches(l->hc_attn_fn->type, DS4_DSPARK_LAYOUT_PLAIN) &&
        l->hc_attn_scale->type == DS4_TENSOR_F32 &&
        l->hc_attn_base->type == DS4_TENSOR_F32 &&
        l->attn_norm->type == DS4_TENSOR_F32 &&
        dspark_tensor_type_matches(l->attn_q_a->type, DS4_DSPARK_LAYOUT_DENSE) &&
        l->attn_q_a_norm->type == DS4_TENSOR_F32 &&
        dspark_tensor_type_matches(l->attn_q_b->type, DS4_DSPARK_LAYOUT_DENSE) &&
        dspark_tensor_type_matches(l->attn_kv->type, DS4_DSPARK_LAYOUT_DENSE) &&
        l->attn_kv_a_norm->type == DS4_TENSOR_F32 &&
        l->attn_sinks->type == DS4_TENSOR_F32 &&
        l->attn_output_a->type == DS4_TENSOR_Q8_0 &&
        l->attn_output_b->type == DS4_TENSOR_Q8_0 &&
        dspark_tensor_type_matches(l->hc_ffn_fn->type, DS4_DSPARK_LAYOUT_PLAIN) &&
        l->hc_ffn_scale->type == DS4_TENSOR_F32 &&
        l->hc_ffn_base->type == DS4_TENSOR_F32 &&
        l->ffn_norm->type == DS4_TENSOR_F32 &&
        dspark_tensor_type_matches(l->ffn_gate_inp->type, DS4_DSPARK_LAYOUT_DENSE) &&
        l->ffn_exp_probs_b->type == DS4_TENSOR_F32 &&
        tensor_is_routed_expert_type(l->ffn_gate_exps->type) &&
        l->ffn_gate_exps->type == l->ffn_up_exps->type &&
        tensor_is_routed_expert_type(l->ffn_down_exps->type) &&
        l->ffn_gate_shexp->type == DS4_TENSOR_Q8_0 &&
        l->ffn_up_shexp->type == DS4_TENSOR_Q8_0 &&
        l->ffn_down_shexp->type == DS4_TENSOR_Q8_0 &&
        l->hc_attn_fn->ndim == 2 &&
        l->hc_attn_fn->dim[0] == hc_dim &&
        l->hc_attn_fn->dim[1] == mix_hc &&
        l->attn_q_a->ndim == 2 &&
        l->attn_q_a->dim[0] == DS4_N_EMBD &&
        l->attn_q_a->dim[1] == DS4_N_LORA_Q &&
        l->attn_q_b->ndim == 2 &&
        l->attn_q_b->dim[0] == DS4_N_LORA_Q &&
        l->attn_q_b->dim[1] == q_dim &&
        l->attn_kv->ndim == 2 &&
        l->attn_kv->dim[0] == DS4_N_EMBD &&
        l->attn_kv->dim[1] == DS4_N_HEAD_DIM &&
        l->attn_output_a->ndim == 2 &&
        l->attn_output_a->dim[0] == group_dim &&
        l->attn_output_a->dim[1] == out_low_dim &&
        l->attn_output_b->ndim == 2 &&
        l->attn_output_b->dim[0] == out_low_dim &&
        l->attn_output_b->dim[1] == DS4_N_EMBD &&
        ds4_gpu_tensor_bytes(metal_graph_batch_hc_mix(g)) >= rows * mix_hc * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_hc_split(g)) >= rows * mix_hc * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_flat_hc(g)) >= rows * hc_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_attn_cur(g)) >= rows * DS4_N_EMBD * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_attn_norm(g)) >= rows * DS4_N_EMBD * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_qr(g)) >= draft * DS4_N_LORA_Q * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_qr_norm(g)) >= draft * DS4_N_LORA_Q * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_q(g)) >= draft * q_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_kv_raw(g)) >= rows * DS4_N_HEAD_DIM * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_kv(g)) >= rows * DS4_N_HEAD_DIM * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_heads(g)) >= draft * q_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_attn_out(g)) >= draft * DS4_N_EMBD * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_after_attn_hc(g)) >= draft * hc_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_next_hc(g)) >= draft * hc_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_prefill_tokens(g)) >= draft * sizeof(int32_t) &&
        ds4_gpu_tensor_bytes(g->dspark_stage_output_hc) >= draft * hc_dim * sizeof(float);
}

static bool dspark_stage_target_cache_seed_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw,
        uint32_t                  stage,
        uint32_t                  n_tokens) {
    if (!g || !dw ||
        stage >= dw->n_stages ||
        n_tokens == 0 ||
        n_tokens > g->prefill_cap ||
        !dspark_stage_cache_ready(g, dw) ||
        !metal_graph_batch_cur_hc(g) ||
        !metal_graph_batch_hc_mix(g) ||
        !metal_graph_batch_hc_split(g) ||
        !metal_graph_batch_flat_hc(g) ||
        !metal_graph_batch_attn_cur(g) ||
        !metal_graph_batch_attn_norm(g) ||
        !metal_graph_batch_kv_raw(g) ||
        !metal_graph_batch_kv(g)) {
        return false;
    }

    const ds4_layer_weights *l = &dw->stage[stage].block;
    if (!l->hc_attn_fn || !l->hc_attn_scale || !l->hc_attn_base ||
        !l->attn_norm || !l->attn_kv || !l->attn_kv_a_norm) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    return
        dspark_tensor_type_matches(l->hc_attn_fn->type, DS4_DSPARK_LAYOUT_PLAIN) &&
        l->hc_attn_scale->type == DS4_TENSOR_F32 &&
        l->hc_attn_base->type == DS4_TENSOR_F32 &&
        l->attn_norm->type == DS4_TENSOR_F32 &&
        dspark_tensor_type_matches(l->attn_kv->type, DS4_DSPARK_LAYOUT_DENSE) &&
        l->attn_kv_a_norm->type == DS4_TENSOR_F32 &&
        l->hc_attn_fn->ndim == 2 &&
        l->hc_attn_fn->dim[0] == hc_dim &&
        l->hc_attn_fn->dim[1] == mix_hc &&
        l->attn_kv->ndim == 2 &&
        l->attn_kv->dim[0] == DS4_N_EMBD &&
        l->attn_kv->dim[1] == DS4_N_HEAD_DIM &&
        ds4_gpu_tensor_bytes(metal_graph_batch_cur_hc(g)) >=
            (uint64_t)n_tokens * hc_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_hc_mix(g)) >=
            (uint64_t)n_tokens * mix_hc * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_hc_split(g)) >=
            (uint64_t)n_tokens * mix_hc * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_flat_hc(g)) >=
            (uint64_t)n_tokens * hc_dim * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_attn_cur(g)) >=
            (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_attn_norm(g)) >=
            (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_kv_raw(g)) >=
            (uint64_t)n_tokens * DS4_N_HEAD_DIM * sizeof(float) &&
        ds4_gpu_tensor_bytes(metal_graph_batch_kv(g)) >=
            (uint64_t)n_tokens * DS4_N_HEAD_DIM * sizeof(float);
}

static bool metal_graph_seed_dspark_stage_target_cache(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  stage,
        uint32_t                  pos0,
        uint32_t                  n_tokens,
        bool                      commands_open) {
    if (!g || !dspark_model || !dw ||
        !dspark_stage_target_cache_seed_ready(g, dw, stage, n_tokens) ||
        n_tokens > g->dspark_cache_cap) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const ds4_layer_weights *block = &dw->stage[stage].block;
    const bool fuse_hc_norm = DS4_N_HC == 4 &&
                              !metal_graph_use_reference_hc_decode() &&
                              metal_graph_enable_batch_hc_norm_fusion();

    ds4_gpu_tensor *hc_mix_view =
        ds4_gpu_tensor_view(metal_graph_batch_hc_mix(g),
                            0,
                            (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view =
        ds4_gpu_tensor_view(metal_graph_batch_hc_split(g),
                            0,
                            (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view =
        ds4_gpu_tensor_view(metal_graph_batch_attn_cur(g),
                            0,
                            (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view;

    const float freq_base = DS4_ROPE_FREQ_BASE;
    const float freq_scale = 1.0f;
    const float ext_factor = 0.0f;
    const float attn_factor = 1.0f;

    if (ok && !commands_open) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(metal_graph_batch_flat_hc(g),
                                                      metal_graph_batch_cur_hc(g),
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(hc_mix_view,
                                                 dspark_model,
                                                 block->hc_attn_fn,
                                                 hc_dim,
                                                 mix_hc,
                                                 metal_graph_batch_flat_hc(g),
                                                 n_tokens);
    if (fuse_hc_norm) {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(attn_cur_view,
                                                                 metal_graph_batch_attn_norm(g),
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 metal_graph_batch_cur_hc(g),
                                                                 dspark_model->map,
                                                                 dspark_model->size,
                                                                 block->hc_attn_scale->abs_offset,
                                                                 block->hc_attn_base->abs_offset,
                                                                 block->attn_norm->abs_offset,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_HC,
                                                                 DS4_N_HC_SINKHORN_ITER,
                                                                 DS4_HC_EPS,
                                                                 DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            metal_graph_batch_cur_hc(g),
                                                            dspark_model->map,
                                                            dspark_model->size,
                                                            block->hc_attn_scale->abs_offset,
                                                            block->hc_attn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_attn_norm(g),
                                                          metal_graph_batch_attn_cur(g),
                                                          dspark_model->map,
                                                          dspark_model->size,
                                                          block->attn_norm->abs_offset,
                                                          DS4_N_EMBD,
                                                          n_tokens,
                                                          DS4_RMS_EPS) != 0;
    }
    if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_batch_kv_raw(g),
                                                 dspark_model,
                                                 block->attn_kv,
                                                 DS4_N_EMBD,
                                                 DS4_N_HEAD_DIM,
                                                 metal_graph_batch_attn_norm(g),
                                                 n_tokens);
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_kv(g),
                                                      metal_graph_batch_kv_raw(g),
                                                      dspark_model->map,
                                                      dspark_model->size,
                                                      block->attn_kv_a_norm->abs_offset,
                                                      DS4_N_HEAD_DIM,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_batch_kv(g),
                                           n_tokens,
                                           1,
                                           DS4_N_HEAD_DIM,
                                           DS4_N_ROT,
                                           pos0,
                                           0,
                                           false,
                                           freq_base,
                                           freq_scale,
                                           ext_factor,
                                           attn_factor,
                                           DS4_ROPE_YARN_BETA_FAST,
                                           DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(metal_graph_batch_kv(g),
                                                       n_tokens,
                                                       DS4_N_HEAD_DIM,
                                                       DS4_N_ROT) != 0;
    if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(g->dspark_raw_cache[stage],
                                                    metal_graph_batch_kv(g),
                                                    g->dspark_cache_cap,
                                                    pos0,
                                                    n_tokens,
                                                    DS4_N_HEAD_DIM) != 0;
    if (ok && !commands_open) ok = ds4_gpu_end_commands() != 0;

    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    if (!ok && !commands_open) (void)ds4_gpu_synchronize();
    return ok;
}

bool metal_graph_seed_dspark_initial_cache_from_prefill(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  batch_start,
        uint32_t                  n_tokens,
        uint32_t                 *seeded_rows) {
    if (seeded_rows) *seeded_rows = 0;
    if (!g || !dspark_model || !dw ||
        n_tokens == 0 ||
        n_tokens > g->prefill_cap ||
        n_tokens > g->dspark_cache_cap ||
        dw->n_stages == 0 ||
        dw->n_stages > DS4_DSPARK_MAX_STAGES ||
        !dspark_stage0_batch_ready(g, dw, n_tokens) ||
        !dspark_stage_cache_ready(g, dw)) {
        return false;
    }

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = metal_graph_eval_dspark_stage0_batch(g,
                                                  dspark_model,
                                                  dw,
                                                  n_tokens,
                                                  true);
    }
    if (ok) {
        ok = ds4_gpu_repeat_hc_rows_tensor(metal_graph_batch_cur_hc(g),
                                           metal_graph_batch_ffn_norm(g),
                                           n_tokens,
                                           DS4_N_EMBD,
                                           DS4_N_HC) != 0;
    }
    for (uint32_t stage = 0; ok && stage < dw->n_stages; stage++) {
        ok = metal_graph_seed_dspark_stage_target_cache(g,
                                                        dspark_model,
                                                        dw,
                                                        stage,
                                                        batch_start,
                                                        n_tokens,
                                                        true);
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) {
        (void)ds4_gpu_synchronize();
        return false;
    }
    if (!metal_graph_dspark_cache_set_window(g, batch_start, n_tokens)) {
        return false;
    }
    if (seeded_rows) *seeded_rows = n_tokens;
    return true;
}

static bool metal_graph_encode_dspark_next_stage_draft_input_from(
        ds4_gpu_graph            *g,
        const ds4_dspark_weights *dw,
        const ds4_gpu_tensor     *draft_hc) {
    if (!g || !dw || !dspark_stage_input_ready(g, dw) ||
        !draft_hc) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = hc_dim * sizeof(float);
    if (ds4_gpu_tensor_bytes(draft_hc) <
        (uint64_t)dw->block_size * hc_bytes) {
        return false;
    }

    return ds4_gpu_tensor_copy(g->dspark_stage_input_hc,
                               hc_bytes,
                               draft_hc,
                               0,
                               (uint64_t)dw->block_size * hc_bytes) != 0;
}

static bool metal_graph_profile_layer_env_match(const char *env_name, uint32_t il) {
    const char *layer_env = getenv(env_name);
    if (!layer_env || !layer_env[0]) return true;

    char *end = NULL;
    const unsigned long layer = strtoul(layer_env, &end, 10);
    return end != layer_env &&
           *end == '\0' &&
           layer <= UINT32_MAX &&
           (uint32_t)layer == il;
}

static bool metal_graph_dspark_stage_profile_enabled(uint32_t stage) {
    return getenv("DS4_DSPARK_STAGE_PROFILE") != NULL &&
           metal_graph_profile_layer_env_match("DS4_DSPARK_STAGE_PROFILE_STAGE",
                                               stage);
}

static bool metal_graph_dspark_stage_profile_boundary(
        const char *part,
        uint32_t    stage,
        uint32_t    pos,
        uint32_t    rows,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: DSpark stage profile stage=%u pos=%u rows=%u %s=%.3f ms\n",
            stage,
            pos,
            rows,
            part,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool metal_graph_eval_dspark_stage_block(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  stage,
        uint32_t                  pos,
        uint32_t                  support_len,
        uint32_t                  raw_start,
        bool                      prepare_next_stage_input,
        bool                      commands_open) {
    if (!g || !dspark_model || !dw ||
        !dspark_stage_block_ready(g, dw, stage)) {
        return false;
    }

    const uint32_t draft = dw->block_size;
    const uint32_t rows = draft + 1u;
    if (support_len > g->dspark_cache_cap ||
        rows > g->dspark_cache_cap - support_len ||
        (support_len != 0 && raw_start >= g->dspark_cache_cap)) {
        return false;
    }
    const uint32_t visible_rows = support_len + rows;
    const uint32_t attention_raw_start =
        support_len ? raw_start : (pos % g->dspark_cache_cap);
    const uint32_t append_pos = support_len ?
        (uint32_t)(((uint64_t)raw_start + support_len) %
                   g->dspark_cache_cap) :
        attention_raw_start;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t group_dim =
        (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const ds4_layer_weights *block = &dw->stage[stage].block;
    const bool fuse_hc_norm = DS4_N_HC == 4 &&
                              !metal_graph_use_reference_hc_decode() &&
                              metal_graph_enable_batch_hc_norm_fusion();

    ds4_gpu_tensor *hc_mix_view =
        ds4_gpu_tensor_view(metal_graph_batch_hc_mix(g),
                            0,
                            (uint64_t)rows * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view =
        ds4_gpu_tensor_view(metal_graph_batch_hc_split(g),
                            0,
                            (uint64_t)rows * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view =
        ds4_gpu_tensor_view(metal_graph_batch_attn_cur(g),
                            0,
                            (uint64_t)rows * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *draft_attn_norm_view =
        ds4_gpu_tensor_view(metal_graph_batch_attn_norm(g),
                            (uint64_t)DS4_N_EMBD * sizeof(float),
                            (uint64_t)draft * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *draft_hc_view =
        ds4_gpu_tensor_view(g->dspark_stage_input_hc,
                            hc_dim * sizeof(float),
                            (uint64_t)draft * hc_dim * sizeof(float));
    ds4_gpu_tensor *draft_hc_split_view =
        ds4_gpu_tensor_view(metal_graph_batch_hc_split(g),
                            mix_hc * sizeof(float),
                            (uint64_t)draft * mix_hc * sizeof(float));
    ds4_gpu_tensor *kv_target_view =
        ds4_gpu_tensor_view(metal_graph_batch_kv(g),
                            0,
                            (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *kv_draft_view =
        ds4_gpu_tensor_view(metal_graph_batch_kv(g),
                            (uint64_t)DS4_N_HEAD_DIM * sizeof(float),
                            (uint64_t)draft * DS4_N_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view =
        ds4_gpu_tensor_view(metal_graph_batch_after_attn_hc(g),
                            0,
                            (uint64_t)draft * hc_dim * sizeof(float));

    bool ok = hc_mix_view && hc_split_view && attn_cur_view &&
              draft_attn_norm_view && draft_hc_view &&
              draft_hc_split_view && kv_target_view && kv_draft_view &&
              after_attn_hc_view;
    const float freq_base = DS4_ROPE_FREQ_BASE;
    const float freq_scale = 1.0f;
    const float ext_factor = 0.0f;
    const float attn_factor = 1.0f;
    const bool stage_profile =
        metal_graph_dspark_stage_profile_enabled(stage);
    double stage_t0 = stage_profile ? now_sec() : 0.0;
#define DS4_DSPARK_PROFILE_STAGE(part_) do {                               \
        if (ok && stage_profile) {                                          \
            ok = metal_graph_dspark_stage_profile_boundary((part_),         \
                                                           stage,           \
                                                           pos,             \
                                                           rows,            \
                                                           &stage_t0);      \
        }                                                                   \
    } while (0)

    if (ok && !commands_open) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(metal_graph_batch_flat_hc(g),
                                                      g->dspark_stage_input_hc,
                                                      (uint32_t)hc_dim,
                                                      rows,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(hc_mix_view,
                                                 dspark_model,
                                                 block->hc_attn_fn,
                                                 hc_dim,
                                                 mix_hc,
                                                 metal_graph_batch_flat_hc(g),
                                                 rows);
    DS4_DSPARK_PROFILE_STAGE("attn_hc_pre");
    if (fuse_hc_norm) {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(attn_cur_view,
                                                                 metal_graph_batch_attn_norm(g),
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 g->dspark_stage_input_hc,
                                                                 dspark_model->map,
                                                                 dspark_model->size,
                                                                 block->hc_attn_scale->abs_offset,
                                                                 block->hc_attn_base->abs_offset,
                                                                 block->attn_norm->abs_offset,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_HC,
                                                                 DS4_N_HC_SINKHORN_ITER,
                                                                 DS4_HC_EPS,
                                                                 DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            g->dspark_stage_input_hc,
                                                            dspark_model->map,
                                                            dspark_model->size,
                                                            block->hc_attn_scale->abs_offset,
                                                            block->hc_attn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_attn_norm(g),
                                                          metal_graph_batch_attn_cur(g),
                                                          dspark_model->map,
                                                          dspark_model->size,
                                                          block->attn_norm->abs_offset,
                                                          DS4_N_EMBD,
                                                          rows,
                                                          DS4_RMS_EPS) != 0;
    }
    DS4_DSPARK_PROFILE_STAGE("attn_norm");

    if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_batch_qr(g),
                                                 dspark_model,
                                                 block->attn_q_a,
                                                 DS4_N_EMBD,
                                                 DS4_N_LORA_Q,
                                                 draft_attn_norm_view,
                                                 draft);
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_qr_norm(g),
                                                      metal_graph_batch_qr(g),
                                                      dspark_model->map,
                                                      dspark_model->size,
                                                      block->attn_q_a_norm->abs_offset,
                                                      DS4_N_LORA_Q,
                                                      draft,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_batch_q(g),
                                                 dspark_model,
                                                 block->attn_q_b,
                                                 DS4_N_LORA_Q,
                                                 q_dim,
                                                 metal_graph_batch_qr_norm(g),
                                                 draft);
    if (ok) ok = ds4_gpu_head_rms_norm_tensor(metal_graph_batch_q(g),
                                               draft,
                                               DS4_N_HEAD,
                                               DS4_N_HEAD_DIM,
                                               DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_batch_q(g),
                                           draft,
                                           DS4_N_HEAD,
                                           DS4_N_HEAD_DIM,
                                           DS4_N_ROT,
                                           pos,
                                           0,
                                           false,
                                           freq_base,
                                           freq_scale,
                                           ext_factor,
                                           attn_factor,
                                           DS4_ROPE_YARN_BETA_FAST,
                                           DS4_ROPE_YARN_BETA_SLOW) != 0;
    DS4_DSPARK_PROFILE_STAGE("q_path");

    if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_batch_kv_raw(g),
                                                 dspark_model,
                                                 block->attn_kv,
                                                 DS4_N_EMBD,
                                                 DS4_N_HEAD_DIM,
                                                 metal_graph_batch_attn_norm(g),
                                                 rows);
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_kv(g),
                                                      metal_graph_batch_kv_raw(g),
                                                      dspark_model->map,
                                                      dspark_model->size,
                                                      block->attn_kv_a_norm->abs_offset,
                                                      DS4_N_HEAD_DIM,
                                                      rows,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_rope_tail_tensor(kv_target_view,
                                           1,
                                           1,
                                           DS4_N_HEAD_DIM,
                                           DS4_N_ROT,
                                           pos,
                                           0,
                                           false,
                                           freq_base,
                                           freq_scale,
                                           ext_factor,
                                           attn_factor,
                                           DS4_ROPE_YARN_BETA_FAST,
                                           DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) ok = ds4_gpu_rope_tail_tensor(kv_draft_view,
                                           draft,
                                           1,
                                           DS4_N_HEAD_DIM,
                                           DS4_N_ROT,
                                           pos,
                                           0,
                                           false,
                                           freq_base,
                                           freq_scale,
                                           ext_factor,
                                           attn_factor,
                                           DS4_ROPE_YARN_BETA_FAST,
                                           DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(metal_graph_batch_kv(g),
                                                       rows,
                                                       DS4_N_HEAD_DIM,
                                                       DS4_N_ROT) != 0;
    if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(g->dspark_raw_cache[stage],
                                                    metal_graph_batch_kv(g),
                                                    g->dspark_cache_cap,
                                                    append_pos,
                                                    rows,
                                                    DS4_N_HEAD_DIM) != 0;
    DS4_DSPARK_PROFILE_STAGE("kv_path");

    if (ok) ok = ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
                    metal_graph_batch_heads(g),
                    dspark_model->map,
                    dspark_model->size,
                    block->attn_sinks->abs_offset,
                    metal_graph_batch_q(g),
                    g->dspark_raw_cache[stage],
                    draft,
                    visible_rows,
                    g->dspark_cache_cap,
                    attention_raw_start,
                    DS4_N_HEAD,
                    DS4_N_HEAD_DIM) != 0;
    if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_batch_heads(g),
                                           draft,
                                           DS4_N_HEAD,
                                           DS4_N_HEAD_DIM,
                                           DS4_N_ROT,
                                           pos,
                                           0,
                                           true,
                                           freq_base,
                                           freq_scale,
                                           ext_factor,
                                           attn_factor,
                                           DS4_ROPE_YARN_BETA_FAST,
                                           DS4_ROPE_YARN_BETA_SLOW) != 0;
    DS4_DSPARK_PROFILE_STAGE("attention");
    if (ok) ok = ds4_gpu_attention_output_q8_batch_tensor(
                    metal_graph_batch_attn_out(g),
                    metal_graph_batch_attn_low(g),
                    metal_graph_batch_group_tmp(g),
                    metal_graph_batch_low_tmp(g),
                    dspark_model->map,
                    dspark_model->size,
                    block->attn_output_a->abs_offset,
                    block->attn_output_b->abs_offset,
                    group_dim,
                    DS4_N_LORA_O,
                    DS4_N_OUT_GROUP,
                    DS4_N_EMBD,
                    metal_graph_batch_heads(g),
                    draft) != 0;
    if (ok) ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                                 metal_graph_batch_attn_out(g),
                                                 draft_hc_view,
                                                 draft_hc_split_view,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    DS4_DSPARK_PROFILE_STAGE("attn_output_hc");

    if (ok) ok = metal_graph_encode_layer_ffn_batch(g,
                                                     dspark_model,
                                                     block,
                                                     stage,
                                                     pos,
                                                     draft);
    DS4_DSPARK_PROFILE_STAGE("ffn");
    if (ok &&
        !prepare_next_stage_input &&
        getenv("DS4_DSPARK_DISABLE_FINAL_OUTPUT_ALIAS") != NULL) {
        ok = ds4_gpu_tensor_copy(g->dspark_stage_output_hc,
                                  0,
                                  metal_graph_batch_next_hc(g),
                                  0,
                                  (uint64_t)draft * hc_dim * sizeof(float)) != 0;
    }
    DS4_DSPARK_PROFILE_STAGE("copy_output");
    if (ok && prepare_next_stage_input) {
        ok = metal_graph_encode_dspark_next_stage_draft_input_from(
                g, dw, metal_graph_batch_next_hc(g));
    }
    DS4_DSPARK_PROFILE_STAGE("next_input");
    if (ok && !commands_open) ok = ds4_gpu_end_commands() != 0;
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(kv_draft_view);
    ds4_gpu_tensor_free(kv_target_view);
    ds4_gpu_tensor_free(draft_hc_split_view);
    ds4_gpu_tensor_free(draft_hc_view);
    ds4_gpu_tensor_free(draft_attn_norm_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    if (!ok && !commands_open) (void)ds4_gpu_synchronize();
#undef DS4_DSPARK_PROFILE_STAGE
    return ok;
}

bool metal_graph_eval_dspark_stage_chain(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  pos,
        bool                      encode_final_hidden,
        uint32_t                 *completed_stages,
        uint32_t                 *cache_start_out,
        uint32_t                 *cache_rows_out) {
    if (completed_stages) *completed_stages = 0;
    if (cache_start_out) *cache_start_out = 0;
    if (cache_rows_out) *cache_rows_out = 0;
    if (!g || !dspark_model || !dw ||
        dw->n_stages == 0 ||
        dw->n_stages > DS4_DSPARK_MAX_STAGES ||
        !dspark_stage_input_ready(g, dw) ||
        !dspark_stage_cache_ready(g, dw) ||
        !metal_graph_prefill_tokens(g) ||
        !g->dspark_draft_tokens) {
        return false;
    }

    const uint32_t rows = dw->block_size + 1u;
    const uint32_t support_len = g->dspark_cache_len;
    const uint32_t raw_start = support_len ? g->dspark_cache_start : 0;
    if (support_len > g->dspark_cache_cap ||
        rows > g->dspark_cache_cap - support_len ||
        (support_len != 0 && raw_start >= g->dspark_cache_cap) ||
        !metal_graph_dspark_cache_ends_at(g, pos)) {
        return false;
    }
    if (cache_start_out) {
        *cache_start_out = support_len ? raw_start :
            (pos % g->dspark_cache_cap);
    }
    if (cache_rows_out) *cache_rows_out = support_len + rows;

    for (uint32_t stage = 0; stage < dw->n_stages; stage++) {
        if (!dspark_stage_block_ready(g, dw, stage)) return false;
    }

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_tensor_copy(metal_graph_prefill_tokens(g),
                                  0,
                                  g->dspark_draft_tokens,
                                  0,
                                  (uint64_t)dw->block_size * sizeof(int32_t)) != 0;
    }
    for (uint32_t stage = 0; ok && stage < dw->n_stages; stage++) {
        const bool stage_ok =
            metal_graph_eval_dspark_stage_block(g,
                                                dspark_model,
                                                dw,
                                                stage,
                                                pos,
                                                support_len,
                                                raw_start,
                                                stage + 1u < dw->n_stages,
                                                true);
        if (!stage_ok) {
            ok = false;
            break;
        }
        if (completed_stages) *completed_stages = stage + 1u;
    }
    /* Final hidden rows consume only the last stage's GPU output. Keep them in
     * this command buffer when the CPU confidence precheck will need them. */
    if (ok && encode_final_hidden) {
        ok = metal_graph_eval_dspark_final_hidden(g, dspark_model, dw, true);
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) {
        (void)ds4_gpu_synchronize();
        return false;
    }
    return true;
}

#endif /* !DS4_NO_GPU */
