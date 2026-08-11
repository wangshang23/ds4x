#ifndef DS4_NO_GPU
/* ------------------------------------------------------------------------
 * GLM MTP (nextn block) drafting.
 *
 * blk.(N_LAYER-1) is GLM 5.2's multi-token-prediction block: a full
 * attention+MoE layer fed with eh_proj(concat(enorm(embed(token[p+1])),
 * hnorm(h[p]))), predicting token[p+2] through the shared output head.
 * It keeps a private compact KV cache (slot = absolute position; only
 * positions >= mtp_min_pos are ever selected, so the unwritten prompt
 * range is never read). Under TP the routed experts are combined over
 * the BIG-gate exchange, never the decode row gate, so the RDMA row-gate
 * schedule stays intact.
 * --------------------------------------------------------------------- */
static bool glm_graph_mtp_ensure(ds4_glm_gpu_graph *g) {
    if (g->mtp_ready) return true;
    if (g->compact_cache_cap == 0) return false;
    const uint64_t elem = glm_graph_compact_cache_elem_bytes();
    const uint64_t kv_bytes = (uint64_t)g->compact_cache_cap * DS4_N_KV_LORA * elem;
    const uint64_t rope_bytes = (uint64_t)g->compact_cache_cap * DS4_N_ROT * elem;
    g->mtp_kv_lora_cache = ds4_gpu_tensor_alloc(kv_bytes);
    g->mtp_k_rope_cache = ds4_gpu_tensor_alloc(rope_bytes);
    g->mtp_concat = ds4_gpu_tensor_alloc(2ull * DS4_N_EMBD * sizeof(float));
    g->mtp_selected = ds4_gpu_tensor_alloc((uint64_t)g->compact_cache_cap * sizeof(int32_t));
    g->mtp_logits_host = malloc((size_t)DS4_N_VOCAB * sizeof(float));
    if (!g->mtp_kv_lora_cache || !g->mtp_k_rope_cache || !g->mtp_concat ||
        !g->mtp_selected || !g->mtp_logits_host) {
        return false;
    }
    g->mtp_ready = 1;
    return true;
}

/* One MTP step at (absolute) position pos: consumes the main model's last
 * hidden h[pos] (expected in g->cur, pre output-norm) and next_token
 * (= token[pos+1]), writes the nextn KV at slot pos, and returns the
 * drafted token[pos+2] by greedy argmax. Clobbers the decode scratch. */
static bool glm_graph_mtp_step(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                next_token,
        uint32_t           pos,
        uint32_t           min_pos,
        int               *draft_out) {
    if (!g || !model || !weights || !draft_out) return false;
    if (DS4_N_NEXTN_PREDICT == 0) return false;
    if (pos >= g->compact_cache_cap || min_pos > pos) {
        fprintf(stderr, "ds4: glm mtp: pos %u/min %u out of range (cap %u)\n",
                pos, min_pos, g->compact_cache_cap);
        return false;
    }
    const uint32_t il = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    const ds4_layer_weights *l = &weights->layer[il];
    if (!l->nextn_eh_proj || !l->nextn_enorm || !l->nextn_hnorm ||
        !l->nextn_shared_head_norm || !l->ffn_gate_exps) {
        fprintf(stderr, "ds4: glm mtp: nextn weights missing at layer %u\n", il);
        return false;
    }
    const uint32_t kv_raw_dim = (uint32_t)l->attn_kv_a_mqa->dim[1];
    const float rope_base = layer_rope_freq_base(il);
    const float rope_scale = layer_rope_freq_scale(il);
    const uint32_t n_selected = pos - min_pos + 1u;
    bool input_ready = false;

    if (g->placement) {
        const int embedding_tier = g->placement[0];
        const int mtp_tier = g->placement[il + 1u];
        bool handoff_ok = glm_graph_ws_switch(g, embedding_tier, true);
        if (handoff_ok) handoff_ok = glm_graph_begin_commands_if_needed();
        if (handoff_ok) {
            handoff_ok = ds4_gpu_embed_token_quant_tensor(
                    g->next,
                    model->map,
                    model->size,
                    weights->token_embd->abs_offset,
                    weights->token_embd->type,
                    DS4_N_VOCAB,
                    (uint32_t)next_token,
                    DS4_N_EMBD) != 0;
        }
        if (handoff_ok) handoff_ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        ds4_gpu_tensor *embedded_next = handoff_ok ? g->next : NULL;
        if (handoff_ok) handoff_ok = glm_graph_ws_switch(g, mtp_tier, true);
        if (handoff_ok) {
            handoff_ok = ds4_gpu_tensor_copy_async(
                    g->next, embedded_next,
                    (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
        }
        if (!handoff_ok) {
            fprintf(stderr, "ds4: glm mtp: multi-tier input handoff failed\n");
            return false;
        }
        input_ready = true;
    }
    if (!glm_graph_mtp_ensure(g)) {
        fprintf(stderr, "ds4: glm mtp: ensure failed (cap %u)\n", g->compact_cache_cap);
        return false;
    }

    /* Draft attention window: absolute cache slots [min_pos..pos]. */
    {
        int32_t *sel = malloc((size_t)n_selected * sizeof(int32_t));
        if (!sel) return false;
        for (uint32_t i = 0; i < n_selected; i++) sel[i] = (int32_t)(min_pos + i);
        const int wr = ds4_gpu_tensor_write(g->mtp_selected, 0, sel,
                                            (uint64_t)n_selected * sizeof(int32_t));
        free(sel);
        if (!wr) {
            fprintf(stderr, "ds4: glm mtp: selected write failed (%u)\n", n_selected);
            return false;
        }
    }

    ds4_gpu_tensor *enorm_view =
        ds4_gpu_tensor_view(g->mtp_concat, 0, (uint64_t)DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *hnorm_view =
        ds4_gpu_tensor_view(g->mtp_concat,
                            (uint64_t)DS4_N_EMBD * sizeof(float),
                            (uint64_t)DS4_N_EMBD * sizeof(float));
    if (!enorm_view || !hnorm_view) {
        ds4_gpu_tensor_free(enorm_view);
        ds4_gpu_tensor_free(hnorm_view);
        fprintf(stderr, "ds4: glm mtp: concat views failed\n");
        return false;
    }

    const char *mtp_stage = "begin";
#define DS4_GLM_MTP_STAGE(name_) do { if (ok) mtp_stage = (name_); } while (0)
    bool ok = glm_graph_begin_commands_if_needed();
    /* MTP input: concat(enorm(embed(next_token)), hnorm(h)) -> eh_proj. */
    if (ok && !input_ready) {
        ok = ds4_gpu_embed_token_quant_tensor(g->next,
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              weights->token_embd->type,
                                              DS4_N_VOCAB,
                                              (uint32_t)next_token,
                                              DS4_N_EMBD) != 0;
    }
    DS4_GLM_MTP_STAGE("enorm");
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(enorm_view,
                                                g->next,
                                                model->map,
                                                model->size,
                                                l->nextn_enorm->abs_offset,
                                                DS4_N_EMBD,
                                                DS4_RMS_EPS) != 0;
    DS4_GLM_MTP_STAGE("hnorm");
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(hnorm_view,
                                                g->cur,
                                                model->map,
                                                model->size,
                                                l->nextn_hnorm->abs_offset,
                                                DS4_N_EMBD,
                                                DS4_RMS_EPS) != 0;
    DS4_GLM_MTP_STAGE("eh_proj");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->cur,
                                                     model,
                                                     l->nextn_eh_proj->abs_offset,
                                                     2ull * DS4_N_EMBD,
                                                     DS4_N_EMBD,
                                                     g->mtp_concat,
                                                     false);
    /* nextn attention (full causal over the MTP window, no indexer). */
    DS4_GLM_MTP_STAGE("attn_norm");
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->attn_norm,
                                                g->cur,
                                                model->map,
                                                model->size,
                                                l->attn_norm->abs_offset,
                                                DS4_N_EMBD,
                                                DS4_RMS_EPS) != 0;
    DS4_GLM_MTP_STAGE("q_a");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->q_rank,
                                                     model,
                                                     l->attn_q_a->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_LORA_Q,
                                                     g->attn_norm,
                                                     false);
    DS4_GLM_MTP_STAGE("q_a_norm");
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->q_rank_norm,
                                                g->q_rank,
                                                model->map,
                                                model->size,
                                                l->attn_q_a_norm->abs_offset,
                                                DS4_N_LORA_Q,
                                                DS4_RMS_EPS) != 0;
    DS4_GLM_MTP_STAGE("q_b");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->q,
                                                     model,
                                                     l->attn_q_b->abs_offset,
                                                     DS4_N_LORA_Q,
                                                     g->q_dim,
                                                     g->q_rank_norm,
                                                     false);
    DS4_GLM_MTP_STAGE("rope");
    if (ok) ok = ds4_gpu_glm_rope_tail_tensor(g->q,
                                              1,
                                              DS4_N_HEAD,
                                              DS4_N_KEY_MLA,
                                              DS4_N_ROT,
                                              pos,
                                              0,
                                              rope_base,
                                              rope_scale,
                                              0.0f,
                                              1.0f,
                                              DS4_ROPE_YARN_BETA_FAST,
                                              DS4_ROPE_YARN_BETA_SLOW) != 0;
    DS4_GLM_MTP_STAGE("kv_a");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->kv_raw,
                                                     model,
                                                     l->attn_kv_a_mqa->abs_offset,
                                                     DS4_N_EMBD,
                                                     kv_raw_dim,
                                                     g->attn_norm,
                                                     false);
    DS4_GLM_MTP_STAGE("kv_norm");
    if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(g->kv_norm,
                                                     g->kv_raw,
                                                     model->map,
                                                     model->size,
                                                     l->attn_kv_a_norm->abs_offset,
                                                     1,
                                                     kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_RMS_EPS) != 0;
    DS4_GLM_MTP_STAGE("kv_store");
    if (ok) ok = ds4_gpu_glm_store_compact_kv_tensor(g->mtp_kv_lora_cache,
                                                     g->mtp_k_rope_cache,
                                                     g->kv_norm,
                                                     g->kv_raw,
                                                     pos,
                                                     1,
                                                     g->compact_cache_cap,
                                                     kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_N_ROT,
                                                     glm_graph_compact_cache_is_f16()) != 0;
    DS4_GLM_MTP_STAGE("qk_low");
    if (ok) ok = ds4_gpu_glm_qk_lowrank_typed_tensor(g->qk_low,
                                                     g->q,
                                                     model->map,
                                                     model->size,
                                                     l->attn_k_b->abs_offset,
                                                     l->attn_k_b->type,
                                                     DS4_N_HEAD,
                                                     DS4_N_KV_LORA,
                                                     (uint32_t)g->q_nope,
                                                     DS4_N_KEY_MLA) != 0;
    DS4_GLM_MTP_STAGE("attention");
    if (ok) ok = ds4_gpu_glm_attention_indexed_decode_typed_tensor(g->heads,
                                                                   g->q,
                                                                   g->qk_low,
                                                                   g->mtp_kv_lora_cache,
                                                                   g->mtp_k_rope_cache,
                                                                   model->map,
                                                                   model->size,
                                                                   l->attn_v_b->abs_offset,
                                                                   l->attn_v_b->type,
                                                                   g->mtp_selected,
                                                                   n_selected,
                                                                   g->compact_cache_cap,
                                                                   glm_graph_compact_cache_is_f16(),
                                                                   DS4_N_HEAD,
                                                                   DS4_N_KV_LORA,
                                                                   (uint32_t)g->q_nope,
                                                                   DS4_N_ROT,
                                                                   DS4_N_VALUE_MLA,
                                                                   0,
                                                                   rope_base,
                                                                   rope_scale,
                                                                   0.0f,
                                                                   1.0f,
                                                                   DS4_ROPE_YARN_BETA_FAST,
                                                                   DS4_ROPE_YARN_BETA_SLOW) != 0;
    DS4_GLM_MTP_STAGE("attn_out");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->attn_out,
                                                     model,
                                                     l->attn_output->abs_offset,
                                                     g->heads_dim,
                                                     DS4_N_EMBD,
                                                     g->heads,
                                                     false);
    DS4_GLM_MTP_STAGE("ffn_norm");
    if (ok) ok = ds4_gpu_add_rms_norm_weight_tensor(g->ffn_norm,
                                                    g->after_attn,
                                                    g->cur,
                                                    g->attn_out,
                                                    model->map,
                                                    model->size,
                                                    l->ffn_norm->abs_offset,
                                                    DS4_N_EMBD,
                                                    DS4_RMS_EPS) != 0;
    /* nextn sparse FFN: router + split routed experts (BIG-gate combine
     * under TP) + shared expert. */
    DS4_GLM_MTP_STAGE("router");
    if (ok) ok = ds4_gpu_matmul_f32_tensor(g->router_logits,
                                           model->map,
                                           model->size,
                                           l->ffn_gate_inp->abs_offset,
                                           DS4_N_EMBD,
                                           DS4_N_EXPERT,
                                           g->ffn_norm,
                                           1) != 0;
    DS4_GLM_MTP_STAGE("router_select");
    if (ok) ok = ds4_gpu_glm_router_select_tensor(g->router_selected,
                                                  g->router_weights,
                                                  g->router_probs,
                                                  model->map,
                                                  model->size,
                                                  l->ffn_exp_probs_b->abs_offset,
                                                  g->router_logits,
                                                  DS4_N_EXPERT,
                                                  DS4_N_EXPERT_USED,
                                                  DS4_EXPERT_WEIGHT_SCALE) != 0;
    DS4_GLM_MTP_STAGE("routed");
    if (ok) {
        uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
        uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
        uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;
        (void)tensor_expert_bytes(model, l->ffn_gate_exps, 0, &gate_in, &gate_out, &gate_row_bytes);
        (void)tensor_expert_bytes(model, l->ffn_up_exps, 0, &up_in, &up_out, &up_row_bytes);
        (void)tensor_expert_bytes(model, l->ffn_down_exps, 0, &down_in, &down_out, &down_row_bytes);
#ifdef DS4_ROCM_BUILD
        if (g->ssd_streaming) {
            const ds4_gpu_stream_expert_table table = {
                .model_map = model->map,
                .model_size = model->size,
                .layer = il,
                .n_total_expert = DS4_N_EXPERT,
                .gate_offset = l->ffn_gate_exps->abs_offset,
                .up_offset = l->ffn_up_exps->abs_offset,
                .down_offset = l->ffn_down_exps->abs_offset,
                .gate_expert_bytes = gate_out * gate_row_bytes,
                .down_expert_bytes = down_out * down_row_bytes,
            };
            ok = ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
                    &table, g->router_selected, DS4_N_EXPERT_USED) != 0;
        }
#endif
        const bool tp_split = g->tp_world == 2 && g->tp_out && g->tp_in;
        ds4_gpu_tensor *routed_dst = g->ffn_out;
        if (tp_split) {
            ok = glm_graph_tp_batch_bounce_ready(g, 1);
            routed_dst = g->tp_bounce_out;
        }
        if (ok) ok = glm_graph_routed_moe_one_dispatch(g,
                                                       model,
                                                       l,
                                                       il,
                                                       routed_dst,
                                                       g->ffn_mid,
                                                       gate_out * gate_row_bytes,
                                                       gate_row_bytes,
                                                       up_out * up_row_bytes,
                                                       up_row_bytes,
                                                       down_out * down_row_bytes,
                                                       down_row_bytes,
                                                       g->router_selected,
                                                       g->router_weights,
                                                       g->ffn_norm,
                                                       false) != 0;
        if (ok && tp_split) {
            ok = glm_graph_tp_batch_ffn_combine(g, il, g->ffn_out, 1);
        }
    }
    DS4_GLM_MTP_STAGE("shared");
    if (ok) ok = glm_graph_encode_shared_swiglu_one(g->ffn_mid,
                                                    g->ffn_gate,
                                                    g->ffn_up,
                                                    model,
                                                    l,
                                                    il,
                                                    pos,
                                                    g->ffn_norm,
                                                    false,
                                                    false,
                                                    NULL);
    DS4_GLM_MTP_STAGE("shared_down");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->ffn_sum,
                                                     model,
                                                     l->ffn_down_shexp->abs_offset,
                                                     DS4_N_FF_EXP,
                                                     DS4_N_EMBD,
                                                     g->ffn_mid,
                                                     false);
    DS4_GLM_MTP_STAGE("residual");
    if (ok) ok = ds4_gpu_add3_tensor(g->next,
                                     g->after_attn,
                                     g->ffn_out,
                                     g->ffn_sum,
                                     DS4_N_EMBD) != 0;
    /* Shared output head behind the nextn head norm. */
    DS4_GLM_MTP_STAGE("head_norm");
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                                g->next,
                                                model->map,
                                                model->size,
                                                l->nextn_shared_head_norm->abs_offset,
                                                DS4_N_EMBD,
                                                DS4_RMS_EPS) != 0;
    DS4_GLM_MTP_STAGE("head");
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->logits,
                                                     model,
                                                     weights->output->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_VOCAB,
                                                     g->output_norm,
                                                     false);
    DS4_GLM_MTP_STAGE("end");
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok) {
        ok = ds4_gpu_tensor_read(g->logits,
                                 0,
                                 g->mtp_logits_host,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    ds4_gpu_tensor_free(enorm_view);
    ds4_gpu_tensor_free(hnorm_view);
    if (!ok) {
        fprintf(stderr, "ds4: glm mtp step failed at stage '%s' (pos %u)\n",
                mtp_stage, pos);
        return false;
    }
#undef DS4_GLM_MTP_STAGE
    int best = 0;
    float best_v = g->mtp_logits_host[0];
    for (uint32_t i = 1; i < DS4_N_VOCAB; i++) {
        if (g->mtp_logits_host[i] > best_v) {
            best_v = g->mtp_logits_host[i];
            best = (int)i;
        }
    }
    *draft_out = best;
    return true;
}

static bool glm_graph_forward_token(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                token,
        const float       *input_hc,
        uint32_t           pos,
        float             *output_hc,
        float             *logits_out,
        bool               defer_completion);


/* Decode-style verify pass for tiny row counts (MTP): the indexed batch
 * fn measures ~1.4ms/layer at n=2 (gate-profile: gpu-wait 1.22ms/layer)
 * while decode does the same math in 0.79ms. This pass mirrors the
 * decode encoders at n rows over the batch scratch, attends causally
 * over the compact caches (valid while pos+n fits the indexer window),
 * and reuses the batch FFN encoder (routed split + big-gate combine).
 * KV/indexer caches are updated exactly like the indexed batch path. */
static bool glm_graph_verify_rows(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        const int         *tokens,
        uint32_t           pos,
        uint32_t           n,
        float             *output_hc,
        float             *logits_out) {
    if (!g || !model || !weights || !tokens || n == 0 ||
        g->compact_cache_cap == 0 ||
        pos + n > g->compact_cache_cap ||
        !g->batch_cur || !g->batch_next || !g->prefill_tokens) {
        return false;
    }
    const uint32_t executable = glm_graph_normal_layer_count();
    ds4_gpu_tensor *cur = g->batch_cur;
    ds4_gpu_tensor *nxt = g->batch_next;
    if (!ds4_gpu_tensor_write(g->prefill_tokens, 0, tokens,
                              (uint64_t)n * sizeof(int32_t))) {
        return false;
    }
    if (g->placement &&
        !glm_graph_verify_ws_switch(g, g->placement[0], false, n)) {
        glm_graph_verify_ws_restore(g);
        return false;
    }
    cur = g->batch_cur;
    nxt = g->batch_next;
    bool ok = glm_graph_begin_commands_if_needed();
    if (ok) ok = ds4_gpu_embed_tokens_quant_tensor(cur,
                                                   g->prefill_tokens,
                                                   model->map,
                                                   model->size,
                                                   weights->token_embd->abs_offset,
                                                   weights->token_embd->type,
                                                   DS4_N_VOCAB,
                                                   n,
                                                   DS4_N_EMBD) != 0;
    for (uint32_t il = 0; ok && il < executable; il++) {
        if (g->placement) {
            g->batch_cur = cur;
            g->batch_next = nxt;
            ok = glm_graph_verify_ws_switch(g,
                                             g->placement[il + 1u],
                                             il != 0u,
                                             n);
            if (ok) {
                ok = glm_graph_ws_switch(g,
                                         g->placement[il + 1u],
                                         false);
            }
            cur = g->batch_cur;
            nxt = g->batch_next;
        }
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t kv_raw_dim = (uint32_t)l->attn_kv_a_mqa->dim[1];
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                 cur,
                                                 model->map,
                                                 model->size,
                                                 l->attn_norm->abs_offset,
                                                 DS4_N_EMBD,
                                                 n,
                                                 DS4_RMS_EPS) != 0;
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_q_rank,
                                                  model,
                                                  l->attn_q_a->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_LORA_Q,
                                                  g->batch_attn_norm,
                                                  n);
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_q_rank_norm,
                                                         g->batch_q_rank,
                                                         model->map,
                                                         model->size,
                                                         l->attn_q_a_norm->abs_offset,
                                                         DS4_N_LORA_Q,
                                                         n,
                                                         DS4_RMS_EPS) != 0;
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_q,
                                                  model,
                                                  l->attn_q_b->abs_offset,
                                                  DS4_N_LORA_Q,
                                                  g->q_dim,
                                                  g->batch_q_rank_norm,
                                                  n);
        if (ok) ok = ds4_gpu_glm_rope_tail_tensor(g->batch_q,
                                                  n,
                                                  DS4_N_HEAD,
                                                  DS4_N_KEY_MLA,
                                                  DS4_N_ROT,
                                                  pos,
                                                  0,
                                                  rope_base,
                                                  rope_scale,
                                                  0.0f,
                                                  1.0f,
                                                  DS4_ROPE_YARN_BETA_FAST,
                                                  DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok && glm_graph_layer_uses_full_indexer(il)) {
            ok = glm_graph_matmul_q8_0_tensor(g->batch_indexer_k,
                                              model,
                                              l->indexer_attn_k->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_INDEXER_HEAD_DIM,
                                              cur,
                                              n);
            if (ok) ok = ds4_gpu_glm_store_indexer_k_tensor(
                    g->layer_indexer_key_cache[il],
                    g->batch_indexer_k,
                    model->map,
                    model->size,
                    l->indexer_k_norm->abs_offset,
                    l->indexer_k_norm_b->abs_offset,
                    pos,
                    n,
                    g->compact_cache_cap,
                    DS4_N_INDEXER_HEAD_DIM,
                    DS4_N_ROT,
                    0,
                    1.0e-6f,
                    rope_base,
                    rope_scale,
                    0.0f,
                    1.0f,
                    DS4_ROPE_YARN_BETA_FAST,
                    DS4_ROPE_YARN_BETA_SLOW,
                    glm_graph_compact_cache_is_f16()) != 0;
        }
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_kv_raw,
                                                  model,
                                                  l->attn_kv_a_mqa->abs_offset,
                                                  DS4_N_EMBD,
                                                  kv_raw_dim,
                                                  g->batch_attn_norm,
                                                  n);
        if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(g->batch_kv_norm,
                                                         g->batch_kv_raw,
                                                         model->map,
                                                         model->size,
                                                         l->attn_kv_a_norm->abs_offset,
                                                         n,
                                                         kv_raw_dim,
                                                         DS4_N_KV_LORA,
                                                         DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_glm_store_compact_kv_tensor(g->layer_kv_lora_cache[il],
                                                         g->layer_k_rope_cache[il],
                                                         g->batch_kv_norm,
                                                         g->batch_kv_raw,
                                                         pos,
                                                         n,
                                                         g->compact_cache_cap,
                                                         kv_raw_dim,
                                                         DS4_N_KV_LORA,
                                                         DS4_N_ROT,
                                                         glm_graph_compact_cache_is_f16()) != 0;
        if (ok) ok = ds4_gpu_glm_qk_lowrank_typed_batch_tensor(g->batch_qk_low,
                                                               g->batch_q,
                                                               model->map,
                                                               model->size,
                                                               l->attn_k_b->abs_offset,
                                                               l->attn_k_b->type,
                                                               n,
                                                               DS4_N_HEAD,
                                                               DS4_N_KV_LORA,
                                                               (uint32_t)g->q_nope,
                                                               DS4_N_KEY_MLA) != 0;
        if (ok) ok = ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
                g->batch_attn_lora,
                g->batch_q,
                g->batch_qk_low,
                g->layer_kv_lora_cache[il],
                g->layer_k_rope_cache[il],
                n,
                pos,
                pos + n,
                g->compact_cache_cap,
                glm_graph_compact_cache_is_f16(),
                DS4_N_HEAD,
                DS4_N_KV_LORA,
                (uint32_t)g->q_nope,
                DS4_N_ROT,
                0,
                rope_base,
                rope_scale,
                0.0f,
                1.0f,
                DS4_ROPE_YARN_BETA_FAST,
                DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok) ok = ds4_gpu_glm_value_project_typed_batch_heads_tensor(
                g->batch_heads,
                g->batch_attn_lora,
                model->map,
                model->size,
                l->attn_v_b->abs_offset,
                l->attn_v_b->type,
                n,
                DS4_N_HEAD,
                DS4_N_KV_LORA,
                DS4_N_VALUE_MLA) != 0;
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_attn_out,
                                                  model,
                                                  l->attn_output->abs_offset,
                                                  g->heads_dim,
                                                  DS4_N_EMBD,
                                                  g->batch_heads,
                                                  n);
        if (ok) ok = ds4_gpu_add_tensor(g->batch_after_attn,
                                        cur,
                                        g->batch_attn_out,
                                        (uint32_t)((uint64_t)n * DS4_N_EMBD)) != 0;
        if (ok) ok = glm_graph_encode_ffn_batch(g,
                                                model,
                                                weights,
                                                l,
                                                il,
                                                pos,
                                                g->batch_after_attn,
                                                nxt,
                                                n,
                                                false,
                                                false,
                                                false,
                                                NULL);
        if (ok) {
            ds4_gpu_tensor *tmp = cur;
            cur = nxt;
            nxt = tmp;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok && output_hc) {
        ok = ds4_gpu_tensor_read(cur,
                                 0,
                                 output_hc,
                                 (uint64_t)n * DS4_N_EMBD * sizeof(float)) != 0;
    }
    if (ok && logits_out) {
        ds4_gpu_tensor *last = glm_graph_tensor_row_view_strided(cur,
                                                                 n - 1u,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_EMBD);
        ok = last != NULL &&
             glm_graph_forward_output_head(g, model, weights, last, logits_out);
        ds4_gpu_tensor_free(last);
    }
    glm_graph_verify_ws_restore(g);
    return ok;
}

static bool glm_graph_forward_tokens(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        const int         *tokens,
        const float       *input_hc,
        uint32_t           pos0,
        uint32_t           n_tokens,
        float             *output_hc,
        float             *logits_out,
        ds4_session_progress_fn display_progress,
        void              *display_progress_ud,
        uint32_t           display_absolute_base,
        uint32_t           work_done_base,
        uint32_t           work_total) {
    if (!g || !model || !weights || !tokens ||
        n_tokens == 0 ||
        g->layer_count == 0 ||
        !glm_graph_span_fits_context(g, pos0, n_tokens)) {
        return false;
    }
    if (!glm_graph_span_fits_full_attention(g, pos0, n_tokens)) {
        glm_graph_log_full_attention_limit(g, pos0, n_tokens);
        return false;
    }
    if (!g->full_kv_cache) {
        fprintf(stderr,
                "ds4: GLM full-attention prefill was requested without an expanded KV cache\n");
        return false;
    }
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= (int)DS4_N_VOCAB) return false;
    }
    if (!input_hc && !g->has_token_embd) return false;
    if (logits_out && !g->has_output_head) return false;
    glm_graph_reset_prefill_seed_capture(g);
    const uint32_t n_rows = pos0 + n_tokens;
    const bool trace = glm_graph_full_prefill_trace_enabled();
    const bool trace_all = trace && glm_graph_full_prefill_trace_all();
    const double trace_slow_ms = trace ?
        (double)glm_graph_full_prefill_trace_slow_ms() : 0.0;
    const double trace_chunk_t0 = trace ? now_sec() : 0.0;
    if (trace) {
        glm_graph_full_prefill_tracef(
                "chunk begin pos=%u tokens=%u rows=%u compact_cap=%u work_base=%u work_total=%u",
                pos0,
                n_tokens,
                n_rows,
                g->compact_cache_cap,
                work_done_base,
                work_total);
    }
    if (g->compact_cache_cap != 0) {
        const double trace_cache_t0 = trace ? now_sec() : 0.0;
        if (!glm_graph_ensure_compact_cache(g, n_rows)) {
            if (trace) {
                glm_graph_full_prefill_tracef(
                        "ensure_cache failed pos=%u tokens=%u rows=%u compact_cap=%u",
                        pos0,
                        n_tokens,
                        n_rows,
                        g->compact_cache_cap);
            }
            return false;
        }
        if (trace) {
            const double ms = (now_sec() - trace_cache_t0) * 1000.0;
            if (trace_all || ms >= trace_slow_ms) {
                glm_graph_full_prefill_tracef(
                        "ensure_cache done pos=%u tokens=%u rows=%u compact_cap=%u %.3f ms",
                        pos0,
                        n_tokens,
                        n_rows,
                        g->compact_cache_cap,
                        ms);
            }
        }
    }

    const double trace_upload_t0 = trace ? now_sec() : 0.0;
    bool ok = glm_graph_upload_tokens(g->prefill_tokens, tokens, n_tokens);
    if (trace) {
        const double ms = (now_sec() - trace_upload_t0) * 1000.0;
        if (trace_all || ms >= trace_slow_ms || !ok) {
            glm_graph_full_prefill_tracef(
                    "upload_tokens %s pos=%u tokens=%u %.3f ms",
                    ok ? "done" : "failed",
                    pos0,
                    n_tokens,
                    ms);
        }
    }
    ds4_gpu_tensor *cur = g->batch_cur;
    ds4_gpu_tensor *next = g->batch_next;
    ds4_gpu_tensor *last_hidden = NULL;

    glm_graph_report_prefill_display_progress(display_progress,
                                              display_progress_ud,
                                              display_absolute_base,
                                              work_done_base,
                                              n_tokens,
                                              0,
                                              g->layer_count,
                                              work_total,
                                              true);

    const bool stage_sync =
        glm_graph_small_prefill_stage_sync(n_tokens, logits_out != NULL);
    const uint32_t layer_flush_interval = stage_sync ? 0u :
        glm_graph_full_prefill_layer_flush_interval(n_tokens,
                                                    n_rows,
                                                    logits_out != NULL);
    const uint32_t progress_flush_interval =
        glm_graph_prefill_progress_flush_interval(layer_flush_interval,
                                                  n_tokens,
                                                  display_progress,
                                                  work_total);
    const bool progress_requested = display_progress && work_total > 0;
    const uint32_t drain_interval =
        (progress_requested && progress_flush_interval != 0) ?
        glm_graph_full_prefill_drain_interval() : 0u;
    metal_graph_stream_prepare_slot layer_prepare_slots[DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD];
    memset(layer_prepare_slots, 0, sizeof(layer_prepare_slots));
    const bool full_layer_prefill =
        glm_graph_stream_prefill_full_layer_enabled(g, n_tokens);
    ds4_gpu_set_glm_streaming_prefill_full_layer(full_layer_prefill);
    const bool streaming_prefill_sync_each_layer =
        !g->ssd_streaming ||
        glm_graph_streaming_prefill_sync_each_layer(full_layer_prefill);
#ifdef DS4_ROCM_BUILD
    rocm_graph_stream_layer_expert_load rocm_full_layer_load;
    memset(&rocm_full_layer_load, 0, sizeof(rocm_full_layer_load));
#endif
    const bool full_layer_prepare_base =
        glm_graph_stream_prefill_full_layer_prepare_enabled(g,
                                                            full_layer_prefill);
    const bool layer_pagein =
        full_layer_prepare_base &&
        glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN",
                              "DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN");
    const bool layer_readahead =
        full_layer_prepare_base &&
        !layer_pagein &&
        glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD",
                              "DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE");
    const bool layer_pread =
        full_layer_prepare_base &&
        !layer_pagein &&
        !layer_readahead &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREAD",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREAD") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE");
    const bool layer_madvise =
        full_layer_prepare_base &&
        !layer_pagein &&
        !layer_pread &&
        !layer_readahead &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_MADVISE",
                               "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_MADVISE");
    const bool layer_prepare =
        layer_pagein || layer_pread || layer_readahead || layer_madvise;
    const bool layer_prepare_overlap =
        layer_prepare &&
        metal_graph_stream_prefill_layer_pagein_overlap_enabled();
    const bool full_layer_flush_intermediate = full_layer_prefill;
    const uint32_t layer_prepare_ahead =
        layer_prepare && layer_prepare_overlap ?
        metal_graph_stream_prefill_layer_prepare_ahead() : 1u;
    if (trace) {
        glm_graph_full_prefill_tracef(
                "mode pos=%u tokens=%u stage_sync=%u layer_flush_interval=%u progress_flush_interval=%u drain_interval=%u",
                pos0,
                n_tokens,
                stage_sync ? 1u : 0u,
                layer_flush_interval,
                progress_flush_interval,
                drain_interval);
    }
    if (ok && layer_prepare && g->layer_count > 0 &&
        !metal_graph_stream_prepare_start_if_needed(NULL,
                                                    model,
                                                    weights,
                                                    g->layer_start,
                                                    n_tokens,
                                                    layer_madvise,
                                                    layer_pread,
                                                    layer_readahead,
                                                    full_layer_prefill &&
                                                    rocm_graph_glm_stream_prefill_full_layer_enabled(
                                                            g,
                                                            &weights->layer[g->layer_start],
                                                            g->layer_start,
                                                            n_tokens),
                                                    layer_prepare_slots,
                                                    layer_prepare_ahead)) {
        ok = false;
    }
#ifdef DS4_ROCM_BUILD
    if (ok &&
        full_layer_prefill &&
        !rocm_graph_glm_stream_layer_expert_load_start_next(
                &rocm_full_layer_load,
                g,
                model,
                weights,
                g->layer_start,
                g->layer_end,
                n_tokens)) {
        ok = false;
    }
#endif
    if (ok) {
        const double t0 = trace ? now_sec() : 0.0;
        if (input_hc) {
            ok = ds4_gpu_tensor_write(cur,
                                      0,
                                      input_hc,
                                      (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float)) != 0;
        } else {
            ok = glm_graph_stream_map_token(g, model, weights);
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (trace) {
            const double ms = (now_sec() - t0) * 1000.0;
            if (trace_all || ms >= trace_slow_ms || !ok) {
                glm_graph_full_prefill_tracef(
                        "begin_commands%s %s pos=%u tokens=%u %.3f ms",
                        input_hc ? "_from_hidden" : "",
                        ok ? "done" : "failed",
                        pos0,
                        n_tokens,
                        ms);
            }
        }
    }
    if (ok && !input_hc) {
        const double t0 = trace ? now_sec() : 0.0;
        ok = ds4_gpu_embed_tokens_quant_tensor(cur,
                                               g->prefill_tokens,
                                               model->map,
                                               model->size,
                                               weights->token_embd->abs_offset,
                                               weights->token_embd->type,
                                               DS4_N_VOCAB,
                                               n_tokens,
                                               DS4_N_EMBD) != 0;
        if (trace) {
            const double ms = (now_sec() - t0) * 1000.0;
            if (trace_all || ms >= trace_slow_ms || !ok) {
                glm_graph_full_prefill_tracef(
                        "embed %s pos=%u tokens=%u %.3f ms",
                        ok ? "done" : "failed",
                        pos0,
                        n_tokens,
                        ms);
            }
        }
    }
    if (ok && g->ssd_streaming && streaming_prefill_sync_each_layer) {
        ok = ds4_gpu_end_commands() != 0;
    }
#define DS4_GLM_PROFILE_PREFILL_STAGE(part_, name_) do { \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary((part_), (name_), il, pos0, n_tokens, &layer_stage_t0); \
        } else if (ok && stage_sync) { \
            ok = glm_graph_prefill_stage_sync_boundary(); \
        } \
    } while (0)
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        const uint32_t slice_layer_done = il - g->layer_start + 1u;
        if (layer_prepare &&
            !metal_graph_stream_prepare_join_layer(NULL,
                                                   model,
                                                   weights,
                                                   il,
                                                   n_tokens,
                                                   layer_madvise,
                                                   layer_pread,
                                                   layer_readahead,
                                                   full_layer_prefill &&
                                                   rocm_graph_glm_stream_prefill_full_layer_enabled(
                                                           g,
                                                           &weights->layer[il],
                                                           il,
                                                           n_tokens),
                                                   layer_prepare_slots,
                                                   layer_prepare_ahead)) {
            ok = false;
            break;
        }
#ifdef DS4_ROCM_BUILD
        if (full_layer_prefill &&
            !rocm_graph_glm_stream_layer_expert_load_ready(
                    &rocm_full_layer_load,
                    g,
                    model,
                    weights,
                    il,
                    n_tokens)) {
            ok = false;
            break;
        }
        if (full_layer_prefill &&
            !rocm_graph_glm_stream_layer_expert_load_start_next(
                    &rocm_full_layer_load,
                    g,
                    model,
                    weights,
                    il + 1u,
                    g->layer_end,
                    n_tokens)) {
            ok = false;
            break;
        }
#endif
        if (g->ssd_streaming) {
            ok = glm_graph_stream_map_prefill_layer(g,
                                                    model,
                                                    weights,
                                                    il,
                                                    n_tokens,
                                                    full_layer_prefill);
            if (ok && layer_prepare && layer_prepare_overlap) {
                bool started_future = false;
                for (uint32_t ahead = 1; ahead <= layer_prepare_ahead; ahead++) {
                    if (il + ahead > g->layer_end) break;
                    started_future = true;
                    if (!metal_graph_stream_prepare_start_if_needed(NULL,
                                                                    model,
                                                                    weights,
                                                                    il + ahead,
                                                                    n_tokens,
                                                                    layer_madvise,
                                                                    layer_pread,
                                                                    layer_readahead,
                                                                    full_layer_prefill &&
                                                                    rocm_graph_glm_stream_prefill_full_layer_enabled(
                                                                            g,
                                                                            &weights->layer[il + ahead],
                                                                            il + ahead,
                                                                            n_tokens),
                                                                    layer_prepare_slots,
                                                                    layer_prepare_ahead)) {
                        ok = false;
                        break;
                    }
                }
                if (ok && !started_future && logits_out) {
                    metal_graph_stream_readahead_output(model, weights);
                }
            }
            if (ok && !ds4_gpu_commands_active()) {
                ok = ds4_gpu_begin_commands() != 0;
            }
        }
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t kv_raw_dim = (uint32_t)l->attn_kv_a_mqa->dim[1];
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        const uint32_t cache_len = pos0 + n_tokens;
        const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
        const bool layer_stage_profile = metal_graph_layer_stage_profile_enabled(il);
        double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
        const double trace_layer_t0 = trace ? now_sec() : 0.0;
        bool trace_layer_flushed = false;
        if (trace && trace_all) {
            glm_graph_full_prefill_tracef(
                    "layer begin layer=%u pos=%u tokens=%u rows=%u",
                    il,
                    pos0,
                    n_tokens,
                    n_rows);
        }
        if (residual_elems > UINT32_MAX) {
            ok = false;
            break;
        }
        if (layer_stage_profile) {
            ok = metal_graph_layer_stage_profile_boundary("glm_attn",
                                                          NULL,
                                                          il,
                                                          pos0,
                                                          n_tokens,
                                                          &layer_stage_t0);
        }

        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                         cur,
                                                         model->map,
                                                         model->size,
                                                         l->attn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         n_tokens,
                                                         DS4_RMS_EPS) != 0;
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "attn_norm");
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_q_rank,
                                                  model,
                                                  l->attn_q_a->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_LORA_Q,
                                                  g->batch_attn_norm,
                                                  n_tokens);
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_q_rank_norm,
                                                         g->batch_q_rank,
                                                         model->map,
                                                         model->size,
                                                         l->attn_q_a_norm->abs_offset,
                                                         DS4_N_LORA_Q,
                                                         n_tokens,
                                                         DS4_RMS_EPS) != 0;
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_q,
                                                  model,
                                                  l->attn_q_b->abs_offset,
                                                  DS4_N_LORA_Q,
                                                  g->q_dim,
                                                  g->batch_q_rank_norm,
                                                  n_tokens);
        if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_q,
                                              n_tokens,
                                              DS4_N_HEAD,
                                              DS4_N_KEY_MLA,
                                              DS4_N_ROT,
                                              pos0,
                                              0,
                                              false,
                                              rope_base,
                                              rope_scale,
                                              0.0f,
                                              1.0f,
                                              DS4_ROPE_YARN_BETA_FAST,
                                              DS4_ROPE_YARN_BETA_SLOW) != 0;
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "q_path");
        if (ok && g->compact_cache_cap != 0 && glm_graph_layer_uses_full_indexer(il)) {
            ok = glm_graph_matmul_q8_0_tensor(g->batch_indexer_k,
                                              model,
                                              l->indexer_attn_k->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_INDEXER_HEAD_DIM,
                                              cur,
                                              n_tokens);
            if (ok) {
                ok = ds4_gpu_glm_store_indexer_k_tensor(
                        g->layer_indexer_key_cache[il],
                        g->batch_indexer_k,
                        model->map,
                        model->size,
                        l->indexer_k_norm->abs_offset,
                        l->indexer_k_norm_b->abs_offset,
                        pos0,
                        n_tokens,
                        g->compact_cache_cap,
                        DS4_N_INDEXER_HEAD_DIM,
                        DS4_N_ROT,
                        0,
                        1.0e-6f,
                        rope_base,
                        rope_scale,
                        0.0f,
                        1.0f,
                        DS4_ROPE_YARN_BETA_FAST,
                        DS4_ROPE_YARN_BETA_SLOW,
                        glm_graph_compact_cache_is_f16()) != 0;
            }
        }
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "indexer_k");
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_kv_raw,
                                                  model,
                                                  l->attn_kv_a_mqa->abs_offset,
                                                  DS4_N_EMBD,
                                                  kv_raw_dim,
                                                  g->batch_attn_norm,
                                                  n_tokens);
        if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(g->batch_kv_norm,
                                                         g->batch_kv_raw,
                                                         model->map,
                                                         model->size,
                                                         l->attn_kv_a_norm->abs_offset,
                                                         n_tokens,
                                                         kv_raw_dim,
                                                         DS4_N_KV_LORA,
                                                         DS4_RMS_EPS) != 0;
        if (ok && g->compact_cache_cap != 0) {
            ok = ds4_gpu_glm_store_compact_kv_tensor(g->layer_kv_lora_cache[il],
                                                     g->layer_k_rope_cache[il],
                                                     g->batch_kv_norm,
                                                     g->batch_kv_raw,
                                                     pos0,
                                                     n_tokens,
                                                     g->compact_cache_cap,
                                                     kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_N_ROT,
                                                     glm_graph_compact_cache_is_f16()) != 0;
        }
        if (ok) ok = ds4_gpu_glm_k_b_project_typed_tensor(g->batch_k_nope,
                                                          g->batch_kv_norm,
                                                          model->map,
                                                          model->size,
                                                          l->attn_k_b->abs_offset,
                                                          l->attn_k_b->type,
                                                          n_tokens,
                                                          DS4_N_KV_LORA,
                                                          (uint32_t)g->q_nope,
                                                          DS4_N_HEAD) != 0;
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_value,
                                                  model,
                                                  l->attn_v_b->abs_offset,
                                                  DS4_N_KV_LORA,
                                                  g->heads_dim,
                                                  g->batch_kv_norm,
                                                  n_tokens);
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "kv_path");
        const bool flash_requested = glm_graph_use_flash_attention_prefill(n_tokens);
        const bool use_staged_flash_kv =
            flash_requested && glm_graph_use_flash_attention_staged_kv(pos0, n_tokens, cache_len);
        const bool use_flash_attn = flash_requested;
        if (ok) {
            if (use_staged_flash_kv) {
                ok = ds4_gpu_glm_build_kv_cache_flash_tensor(g->layer_key_cache[il],
                                                             g->layer_value_cache[il],
                                                             g->batch_kv_raw,
                                                             g->batch_k_nope,
                                                             g->batch_value,
                                                             pos0,
                                                             n_tokens,
                                                             g->ctx_cap,
                                                             DS4_N_HEAD,
                                                             kv_raw_dim,
                                                             DS4_N_KV_LORA,
                                                             (uint32_t)g->q_nope,
                                                             DS4_N_ROT,
                                                             DS4_N_VALUE_MLA,
                                                             0,
                                                             rope_base,
                                                             rope_scale,
                                                             0.0f,
                                                             1.0f,
                                                             DS4_ROPE_YARN_BETA_FAST,
                                                             DS4_ROPE_YARN_BETA_SLOW,
                                                             true) != 0;
            } else {
                ok = ds4_gpu_glm_build_kv_cache_tensor(g->layer_key_cache[il],
                                                       g->layer_value_cache[il],
                                                       g->batch_kv_raw,
                                                       g->batch_k_nope,
                                                       g->batch_value,
                                                       pos0,
                                                       n_tokens,
                                                       g->ctx_cap,
                                                       DS4_N_HEAD,
                                                       kv_raw_dim,
                                                       DS4_N_KV_LORA,
                                                       (uint32_t)g->q_nope,
                                                       DS4_N_ROT,
                                                       DS4_N_VALUE_MLA,
                                                       0,
                                                       rope_base,
                                                       rope_scale,
                                                       0.0f,
                                                       1.0f,
                                                       DS4_ROPE_YARN_BETA_FAST,
                                                       DS4_ROPE_YARN_BETA_SLOW,
                                                       true) != 0;
            }
        }
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "kv_cache");
        if (ok) {
            if (use_flash_attn) {
                if (use_staged_flash_kv) {
                    ok = ds4_gpu_glm_attention_flash_staged_tensor(g->batch_heads,
                                                                   g->batch_q,
                                                                   g->layer_key_cache[il],
                                                                   g->layer_value_cache[il],
                                                                   pos0,
                                                                   n_tokens,
                                                                   cache_len,
                                                                   g->ctx_cap,
                                                                   DS4_N_HEAD,
                                                                   DS4_N_KEY_MLA,
                                                                   DS4_N_VALUE_MLA,
                                                                   true) != 0;
                } else {
                    ok = ds4_gpu_glm_attention_flash_tensor(g->batch_heads,
                                                            g->batch_q,
                                                            g->layer_key_cache[il],
                                                            g->layer_value_cache[il],
                                                            pos0,
                                                            n_tokens,
                                                            cache_len,
                                                            g->ctx_cap,
                                                            DS4_N_HEAD,
                                                            DS4_N_KEY_MLA,
                                                            DS4_N_VALUE_MLA,
                                                            true) != 0;
                }
            } else {
                ok = ds4_gpu_glm_attention_full_tensor(g->batch_heads,
                                                       g->batch_q,
                                                       g->layer_key_cache[il],
                                                       g->layer_value_cache[il],
                                                       pos0,
                                                       n_tokens,
                                                       cache_len,
                                                       g->ctx_cap,
                                                       DS4_N_HEAD,
                                                       DS4_N_KEY_MLA,
                                                       DS4_N_VALUE_MLA,
                                                       true) != 0;
            }
        }
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "attention");
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_attn_out,
                                                  model,
                                                  l->attn_output->abs_offset,
                                                  g->heads_dim,
                                                  DS4_N_EMBD,
                                                  g->batch_heads,
                                                  n_tokens);
        if (ok) ok = ds4_gpu_add_tensor(g->batch_after_attn,
                                        cur,
                                        g->batch_attn_out,
                                        (uint32_t)residual_elems) != 0;
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "attn_output");
        if (ok) ok = glm_graph_encode_ffn_batch(g,
                                                model,
                                                weights,
                                                l,
                                                il,
                                                pos0,
                                                g->batch_after_attn,
                                                next,
                                                n_tokens,
                                                full_layer_prefill,
                                                layer_stage_profile,
                                                stage_sync,
                                                layer_stage_profile ? &layer_stage_t0 : NULL);
        if (ok) {
            ds4_gpu_tensor *tmp = cur;
            cur = next;
            next = tmp;
        }
        if (ok && glm_debug_hidden_dump_layer_match(il)) {
            ok = ds4_gpu_end_commands() != 0;
            if (ok) {
                for (uint32_t r = 0; r < n_tokens; r++)
                    glm_debug_dump_hidden_layer(cur, r, il, pos0 + r);
                glm_debug_dump_raw_layer(g->batch_router_selected, "sel",
                                         (uint64_t)n_tokens * DS4_N_EXPERT_USED * sizeof(int32_t),
                                         il, -1);
                glm_debug_dump_raw_layer(g->batch_router_weights, "selw",
                                         (uint64_t)n_tokens * DS4_N_EXPERT_USED * sizeof(float),
                                         il, -1);
                ok = ds4_gpu_begin_commands() != 0;
            }
        }
        if (ok &&
            !g->ssd_streaming &&
            progress_flush_interval != 0 &&
            (il < g->layer_end || progress_requested) &&
            (slice_layer_done % progress_flush_interval) == 0) {
            const uint32_t work_done =
                work_done_base + (uint32_t)(((uint64_t)n_tokens * slice_layer_done) / g->layer_count);
            const bool drain_now =
                drain_interval != 0 &&
                il < g->layer_end &&
                (slice_layer_done % drain_interval) == 0;
            const char *command_action = drain_now ? "drain" : "flush";
            const double trace_command_t0 = trace ? now_sec() : 0.0;
            if (trace && (trace_all || drain_now)) {
                glm_graph_full_prefill_tracef(
                        "layer %s begin layer=%u pos=%u tokens=%u work=%u/%u",
                        command_action,
                        il,
                        pos0,
                        n_tokens,
                        work_done,
                        work_total);
            }
            if (drain_now) {
                ok = ds4_gpu_end_commands() != 0;
                if (ok) ok = ds4_gpu_begin_commands() != 0;
            } else {
                ok = ds4_gpu_flush_commands() != 0;
            }
            if (trace) {
                const double trace_command_done = now_sec();
                const double command_ms = (trace_command_done - trace_command_t0) * 1000.0;
                const double layer_ms = (trace_command_done - trace_layer_t0) * 1000.0;
                trace_layer_flushed = true;
                if (trace_all || drain_now ||
                    command_ms >= trace_slow_ms || layer_ms >= trace_slow_ms || !ok) {
                    glm_graph_full_prefill_tracef(
                            "layer %s %s layer=%u pos=%u tokens=%u command=%.3f ms layer_total=%.3f ms work=%u/%u",
                            command_action,
                            ok ? "done" : "failed",
                            il,
                            pos0,
                            n_tokens,
                            command_ms,
                            layer_ms,
                            work_done,
                            work_total);
                }
            }
            if (ok) {
                const bool progress_completed =
                    drain_interval == 0 || drain_now;
                if (progress_completed) {
                    glm_graph_report_prefill_display_progress(display_progress,
                                                              display_progress_ud,
                                                              display_absolute_base,
                                                              work_done_base,
                                                              n_tokens,
                                                              slice_layer_done,
                                                              g->layer_count,
                                                              work_total,
                                                              logits_out == NULL && output_hc == NULL);
                }
            }
        }
        if (g->ssd_streaming) {
            if (streaming_prefill_sync_each_layer) {
                if (ok && full_layer_flush_intermediate &&
                    il < g->layer_end) {
                    ok = ds4_gpu_flush_commands() != 0;
                } else if (ok) {
                    ok = ds4_gpu_end_commands() != 0;
                }
                else (void)ds4_gpu_synchronize();
            }
            else if (!ok) (void)ds4_gpu_synchronize();
            if (ok) {
                glm_graph_report_prefill_display_progress(display_progress,
                                                          display_progress_ud,
                                                          display_absolute_base,
                                                          work_done_base,
                                                          n_tokens,
                                                          slice_layer_done,
                                                          g->layer_count,
                                                          work_total,
                                                          logits_out == NULL && output_hc == NULL);
            }
        }
        if (ok &&
            g->ssd_streaming &&
            layer_prepare &&
            !layer_prepare_overlap) {
            if (il < g->layer_end) {
                if (!metal_graph_stream_prepare_start_if_needed(NULL,
                                                                model,
                                                                weights,
                                                                il + 1u,
                                                                n_tokens,
                                                                layer_madvise,
                                                                layer_pread,
                                                                layer_readahead,
                                                                full_layer_prefill &&
                                                                rocm_graph_glm_stream_prefill_full_layer_enabled(
                                                                        g,
                                                                        &weights->layer[il + 1u],
                                                                        il + 1u,
                                                                        n_tokens),
                                                                layer_prepare_slots,
                                                                layer_prepare_ahead)) {
                    ok = false;
                }
            } else if (logits_out) {
                metal_graph_stream_readahead_output(model, weights);
            }
        }
        if (trace && ok) {
            const double layer_ms = (now_sec() - trace_layer_t0) * 1000.0;
            if (trace_all || (!trace_layer_flushed && layer_ms >= trace_slow_ms)) {
                glm_graph_full_prefill_tracef(
                        "layer end layer=%u pos=%u tokens=%u flushed=%u layer_total=%.3f ms",
                        il,
                        pos0,
                        n_tokens,
                        trace_layer_flushed ? 1u : 0u,
                        layer_ms);
            }
        }
    }
#undef DS4_GLM_PROFILE_PREFILL_STAGE
    if (ok && !g->ssd_streaming) {
        const double trace_end_t0 = trace ? now_sec() : 0.0;
        if (trace) {
            glm_graph_full_prefill_tracef(
                    "chunk end_commands begin pos=%u tokens=%u",
                    pos0,
                    n_tokens);
        }
        ok = ds4_gpu_end_commands() != 0;
        if (trace) {
            const double end_ms = (now_sec() - trace_end_t0) * 1000.0;
            const double chunk_ms = (now_sec() - trace_chunk_t0) * 1000.0;
            glm_graph_full_prefill_tracef(
                    "chunk end_commands %s pos=%u tokens=%u end=%.3f ms chunk_total=%.3f ms",
                    ok ? "done" : "failed",
                    pos0,
                    n_tokens,
                    end_ms,
                    chunk_ms);
        }
    } else if (!ok) {
        if (trace) {
            glm_graph_full_prefill_tracef(
                    "chunk failed before end pos=%u tokens=%u elapsed=%.3f ms",
                    pos0,
                    n_tokens,
                    (now_sec() - trace_chunk_t0) * 1000.0);
        }
#ifdef DS4_ROCM_BUILD
        (void)rocm_graph_stream_layer_expert_load_join(&rocm_full_layer_load);
        if (full_layer_prefill) {
            (void)ds4_gpu_stream_expert_cache_release_layer_cache();
        }
#endif
        if (layer_prepare) {
            (void)metal_graph_stream_prepare_join_all(layer_prepare_slots,
                                                      layer_prepare_ahead);
        }
        (void)ds4_gpu_synchronize();
    }
    if (ok && layer_prepare &&
        !metal_graph_stream_prepare_join_all(layer_prepare_slots,
                                             layer_prepare_ahead)) {
        ok = false;
    }
#ifdef DS4_ROCM_BUILD
    if (!rocm_graph_stream_layer_expert_load_join(&rocm_full_layer_load)) {
        ok = false;
    }
    if (full_layer_prefill) {
        (void)ds4_gpu_stream_expert_cache_release_layer_cache();
    }
#endif
    if (ok &&
        g->ssd_streaming &&
        !streaming_prefill_sync_each_layer &&
        !output_hc &&
        !logits_out) {
        ok = ds4_gpu_end_commands() != 0;
    }
    if (ok) {
        glm_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  display_absolute_base,
                                                  work_done_base,
                                                  n_tokens,
                                                  g->layer_count,
                                                  g->layer_count,
                                                  work_total,
                                                  logits_out == NULL && output_hc == NULL);
    }
    if (ok && output_hc) {
        ok = ds4_gpu_tensor_read(cur,
                                 0,
                                 output_hc,
                                 (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float)) != 0;
    }
    if (ok && logits_out) {
        ok = glm_graph_seed_streaming_expert_cache_from_prefill(g,
                                                                model,
                                                                weights);
    }
    if (ok && logits_out) {
        last_hidden = glm_graph_tensor_row_view_strided(cur,
                                                        n_tokens - 1u,
                                                        DS4_N_EMBD,
                                                        DS4_N_EMBD);
        ok = last_hidden != NULL;
        if (ok && g->ssd_streaming) ok = glm_graph_stream_map_output(g, model, weights);
        if (ok) ok = glm_graph_forward_output_head(g, model, weights, last_hidden, logits_out);
        if (ok) {
            glm_graph_report_prefill_display_progress(display_progress,
                                                      display_progress_ud,
                                                      display_absolute_base,
                                                      work_done_base,
                                                      n_tokens,
                                                      g->layer_count,
                                                      g->layer_count,
                                                      work_total,
                                                      true);
        }
    }
    ds4_gpu_tensor_free(last_hidden);
    ds4_gpu_set_glm_streaming_prefill_full_layer(false);
    return ok;
}

#endif /* !DS4_NO_GPU */
