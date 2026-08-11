/* Per-tier decode scratch keeps layer kernels off peer-mapped work buffers. */
#define DS4_GLM_WS_FIELDS(X) \
    X(cur) X(next) X(attn_norm) X(q_rank) X(q_rank_norm) X(q) X(kv_raw) \
    X(kv_norm) X(k_nope) X(value) X(heads) X(attn_out) X(after_attn) \
    X(ffn_norm) X(ffn_gate) X(ffn_up) X(ffn_mid) X(ffn_out) X(ffn_sum) \
    X(router_logits) X(router_probs) X(router_selected) X(router_weights) \
    X(indexer_k) X(indexer_q) X(indexer_weights) X(indexer_scores) \
    X(indexer_selected) X(qk_low)

static ds4_gpu_tensor **glm_graph_ws_slot(ds4_glm_gpu_graph *g, int i) {
    int n = 0;
#define DS4_GLM_WS_SLOT_CASE(field) if (n++ == i) return &g->field;
    DS4_GLM_WS_FIELDS(DS4_GLM_WS_SLOT_CASE)
#undef DS4_GLM_WS_SLOT_CASE
    return NULL;
}

static void glm_graph_ws_free(ds4_glm_gpu_graph *g) {
    if (!g || !g->ws_ready) return;
    for (int i = 0; i < DS4_GLM_WS_SLOTS; i++) {
        ds4_gpu_tensor **slot = glm_graph_ws_slot(g, i);
        if (slot) *slot = g->ws_orig[i];
    }
    for (int tier = 0; tier < DS4_MAX_GPUS; tier++) {
        for (int i = 0; i < DS4_GLM_WS_SLOTS; i++) {
            ds4_gpu_tensor_free(g->ws_mirror[tier][i]);
            g->ws_mirror[tier][i] = NULL;
        }
    }
    g->ws_ready = 0;
    g->ws_tier = -1;
}

static void glm_graph_ws_init(ds4_glm_gpu_graph *g) {
    g->ws_ready = 0;
    g->ws_tier = -1;
    if (!g->placement) return;
    bool used[DS4_MAX_GPUS] = { false };
    if (g->placement[0] >= 0 && g->placement[0] < DS4_MAX_GPUS) {
        used[g->placement[0]] = true;
    }
    for (uint32_t il = g->layer_start; il <= g->layer_end; il++) {
        const int tier = g->placement[il + 1u];
        if (tier >= 0 && tier < DS4_MAX_GPUS) used[tier] = true;
    }
    for (int i = 0; i < DS4_GLM_WS_SLOTS; i++) {
        ds4_gpu_tensor **slot = glm_graph_ws_slot(g, i);
        if (!slot) return;
        g->ws_orig[i] = *slot;
    }
    for (int tier = 0; tier < DS4_MAX_GPUS; tier++) {
        if (!used[tier]) continue;
        for (int i = 0; i < DS4_GLM_WS_SLOTS; i++) {
            const ds4_gpu_tensor *orig = g->ws_orig[i];
            if (!orig) continue;
            g->ws_mirror[tier][i] =
                ds4_gpu_tensor_alloc_ptr_on(tier, ds4_gpu_tensor_bytes(orig));
            if (!g->ws_mirror[tier][i]) {
                fprintf(stderr,
                        "ds4: GLM per-tier working-set alloc failed (tier %d); "
                        "falling back to base buffers\n",
                        tier);
                g->ws_ready = 1;
                glm_graph_ws_free(g);
                return;
            }
        }
    }
    g->ws_ready = 1;
}

static bool glm_graph_ws_switch(ds4_glm_gpu_graph *g,
                                int tier,
                                bool carry_hidden) {
    if (!g->placement) return true;
    if (tier < 0 || tier >= DS4_MAX_GPUS ||
        ds4_gpu_set_current_device_fenced(tier) != 0) {
        return false;
    }
    if (!g->ws_ready) return true;
    if (g->ws_tier == tier) return true;
    ds4_gpu_tensor *old_cur = g->cur;
    for (int i = 0; i < DS4_GLM_WS_SLOTS; i++) {
        ds4_gpu_tensor **slot = glm_graph_ws_slot(g, i);
        if (slot && g->ws_mirror[tier][i]) {
            *slot = g->ws_mirror[tier][i];
        }
    }
    if (carry_hidden && old_cur && g->cur && g->cur != old_cur) {
        const uint64_t bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
        if (ds4_gpu_tensor_copy_async(g->cur, old_cur, bytes) == 0) {
            return false;
        }
    }
    g->ws_tier = tier;
    return true;
}

#define DS4_GLM_VERIFY_WS_FIELDS(X) \
    X(batch_cur) X(batch_next) X(batch_attn_norm) X(batch_q_rank) \
    X(batch_q_rank_norm) X(batch_q) X(batch_indexer_k) X(batch_kv_raw) \
    X(batch_kv_norm) X(batch_qk_low) X(batch_attn_lora) \
    X(batch_indexer_selected) X(batch_heads) X(batch_attn_out) \
    X(batch_after_attn) X(batch_ffn_norm) X(batch_ffn_gate) X(batch_ffn_up) \
    X(batch_shared_mid) X(batch_ffn_mid) X(batch_routed_gate) \
    X(batch_routed_up) X(batch_routed_down) X(batch_ffn_out) \
    X(batch_router_logits) X(batch_router_probs) X(batch_router_selected) \
    X(batch_router_weights)

static ds4_gpu_tensor **glm_graph_verify_ws_slot(ds4_glm_gpu_graph *g,
                                                  int i) {
    int n = 0;
#define DS4_GLM_VERIFY_WS_SLOT_CASE(field) if (n++ == i) return &g->field;
    DS4_GLM_VERIFY_WS_FIELDS(DS4_GLM_VERIFY_WS_SLOT_CASE)
#undef DS4_GLM_VERIFY_WS_SLOT_CASE
    return NULL;
}

static void glm_graph_verify_ws_restore(ds4_glm_gpu_graph *g) {
    if (!g || !g->verify_ws_ready) return;
    for (int i = 0; i < DS4_GLM_VERIFY_WS_SLOTS; i++) {
        ds4_gpu_tensor **slot = glm_graph_verify_ws_slot(g, i);
        if (slot) *slot = g->verify_ws_orig[i];
    }
    g->verify_ws_tier = -1;
}

static void glm_graph_verify_ws_free(ds4_glm_gpu_graph *g) {
    if (!g || !g->verify_ws_ready) return;
    glm_graph_verify_ws_restore(g);
    for (int tier = 0; tier < DS4_MAX_GPUS; tier++) {
        for (int i = 0; i < DS4_GLM_VERIFY_WS_SLOTS; i++) {
            ds4_gpu_tensor_free(g->verify_ws_mirror[tier][i]);
            g->verify_ws_mirror[tier][i] = NULL;
        }
    }
    memset(g->verify_ws_orig, 0, sizeof(g->verify_ws_orig));
    g->verify_ws_ready = 0;
}

static bool glm_graph_verify_ws_init(ds4_glm_gpu_graph *g) {
    if (!g) return false;
    if (g->verify_ws_ready) return true;
    if (!g->placement) {
        g->verify_ws_ready = 1;
        g->verify_ws_tier = -1;
        return true;
    }
    bool used[DS4_MAX_GPUS] = { false };
    for (uint32_t il = g->layer_start; il <= g->layer_end; il++) {
        const int tier = g->placement[il + 1u];
        if (tier < 0 || tier >= DS4_MAX_GPUS) return false;
        used[tier] = true;
    }
    for (int i = 0; i < DS4_GLM_VERIFY_WS_SLOTS; i++) {
        ds4_gpu_tensor **slot = glm_graph_verify_ws_slot(g, i);
        if (!slot) return false;
        g->verify_ws_orig[i] = *slot;
    }
    for (int tier = 0; tier < DS4_MAX_GPUS; tier++) {
        if (!used[tier]) continue;
        for (int i = 0; i < DS4_GLM_VERIFY_WS_SLOTS; i++) {
            const ds4_gpu_tensor *orig = g->verify_ws_orig[i];
            if (!orig) continue;
            const uint32_t cap = (i >= 9 && i <= 11) ?
                g->indexed_prefill_cap : g->ctx_cap;
            const uint64_t orig_bytes = ds4_gpu_tensor_bytes(orig);
            if (cap == 0 || orig_bytes % cap != 0 ||
                orig_bytes / cap > UINT64_MAX / 2u) {
                g->verify_ws_ready = 1;
                glm_graph_verify_ws_free(g);
                return false;
            }
            const uint64_t bytes = (orig_bytes / cap) * 2u;
            g->verify_ws_mirror[tier][i] =
                ds4_gpu_tensor_alloc_ptr_on(tier, bytes);
            if (!g->verify_ws_mirror[tier][i]) {
                g->verify_ws_ready = 1;
                glm_graph_verify_ws_free(g);
                return false;
            }
        }
    }
    g->verify_ws_ready = 1;
    g->verify_ws_tier = -1;
    return true;
}

static bool glm_graph_verify_ws_switch(ds4_glm_gpu_graph *g,
                                       int tier,
                                       bool carry_hidden,
                                       uint32_t n_rows) {
    if (!g || n_rows == 0 || n_rows > 2) return false;
    if (!g->placement) return true;
    if (!glm_graph_verify_ws_init(g) ||
        tier < 0 || tier >= DS4_MAX_GPUS ||
        ds4_gpu_set_current_device_fenced(tier) != 0) {
        return false;
    }
    if (g->verify_ws_tier == tier) return true;
    ds4_gpu_tensor *old_cur = g->batch_cur;
    for (int i = 0; i < DS4_GLM_VERIFY_WS_SLOTS; i++) {
        ds4_gpu_tensor **slot = glm_graph_verify_ws_slot(g, i);
        if (slot && g->verify_ws_mirror[tier][i]) {
            *slot = g->verify_ws_mirror[tier][i];
        }
    }
    if (carry_hidden && old_cur && g->batch_cur && old_cur != g->batch_cur) {
        const uint64_t bytes =
            (uint64_t)n_rows * DS4_N_EMBD * sizeof(float);
        if (ds4_gpu_tensor_copy_async(g->batch_cur, old_cur, bytes) == 0) {
            return false;
        }
    }
    g->verify_ws_tier = tier;
    return true;
}
#undef DS4_GLM_VERIFY_WS_FIELDS

static void glm_graph_free(ds4_glm_gpu_graph *g) {
    glm_graph_verify_ws_free(g);
    glm_graph_ws_free(g);
    if (!g) return;
    ds4_gpu_tensor_free(g->mtp_kv_lora_cache);
    ds4_gpu_tensor_free(g->mtp_k_rope_cache);
    ds4_gpu_tensor_free(g->mtp_concat);
    ds4_gpu_tensor_free(g->mtp_selected);
    free(g->mtp_logits_host);
    g->mtp_kv_lora_cache = NULL;
    g->mtp_k_rope_cache = NULL;
    g->mtp_concat = NULL;
    g->mtp_selected = NULL;
    g->mtp_logits_host = NULL;
    g->mtp_ready = 0;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_indexer_key_cache[il]);
        ds4_gpu_tensor_free(g->layer_k_rope_cache[il]);
        ds4_gpu_tensor_free(g->layer_kv_lora_cache[il]);
        ds4_gpu_tensor_free(g->layer_value_cache[il]);
        ds4_gpu_tensor_free(g->layer_key_cache[il]);
    }
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->batch_router_weights);
    ds4_gpu_tensor_free(g->prefill_seed_router_selected);
    ds4_gpu_tensor_free(g->batch_router_selected);
    ds4_gpu_tensor_free(g->batch_router_probs);
    ds4_gpu_tensor_free(g->batch_router_logits);
    ds4_gpu_tensor_free(g->batch_routed_down);
    ds4_gpu_tensor_free(g->batch_routed_up);
    ds4_gpu_tensor_free(g->batch_routed_gate);
    ds4_gpu_tensor_free(g->batch_ffn_out);
    ds4_gpu_tensor_free(g->batch_ffn_mid);
    ds4_gpu_tensor_free(g->batch_shared_mid);
    ds4_gpu_tensor_free(g->batch_ffn_up);
    ds4_gpu_tensor_free(g->batch_ffn_gate);
    ds4_gpu_tensor_free(g->batch_ffn_norm);
    ds4_gpu_tensor_free(g->batch_after_attn);
    ds4_gpu_tensor_free(g->batch_attn_out);
    ds4_gpu_tensor_free(g->batch_heads);
    ds4_gpu_tensor_free(g->batch_value);
    ds4_gpu_tensor_free(g->batch_k_nope);
    ds4_gpu_tensor_free(g->batch_kv_norm);
    ds4_gpu_tensor_free(g->batch_kv_raw);
    ds4_gpu_tensor_free(g->batch_attn_lora);
    ds4_gpu_tensor_free(g->batch_qk_low);
    ds4_gpu_tensor_free(g->batch_indexer_selected);
    ds4_gpu_tensor_free(g->batch_indexer_scores);
    ds4_gpu_tensor_free(g->batch_indexer_weights);
    ds4_gpu_tensor_free(g->batch_indexer_q);
    ds4_gpu_tensor_free(g->batch_indexer_k);
    ds4_gpu_tensor_free(g->batch_q);
    ds4_gpu_tensor_free(g->batch_q_rank_norm);
    ds4_gpu_tensor_free(g->batch_q_rank);
    ds4_gpu_tensor_free(g->batch_attn_norm);
    ds4_gpu_tensor_free(g->batch_next);
    ds4_gpu_tensor_free(g->batch_cur);
    ds4_gpu_tensor_free(g->prefill_tokens);
    ds4_gpu_tensor_free(g->output_norm);
    ds4_gpu_tensor_free(g->router_weights);
    ds4_gpu_tensor_free(g->router_selected);
    ds4_gpu_tensor_free(g->router_probs);
    ds4_gpu_tensor_free(g->router_logits);
    ds4_gpu_tensor_free(g->ffn_sum);
    ds4_gpu_tensor_free(g->ffn_out);
    ds4_gpu_tensor_free(g->routed_down);
    ds4_gpu_tensor_free(g->routed_up);
    ds4_gpu_tensor_free(g->tp_bounce_out);
    ds4_gpu_tensor_free(g->tp_bounce_in);
    ds4_gpu_tensor_free(g->routed_gate);
    ds4_gpu_tensor_free(g->ffn_mid);
    ds4_gpu_tensor_free(g->ffn_up);
    ds4_gpu_tensor_free(g->ffn_gate);
    ds4_gpu_tensor_free(g->ffn_norm);
    ds4_gpu_tensor_free(g->after_attn);
    ds4_gpu_tensor_free(g->attn_out);
    ds4_gpu_tensor_free(g->heads);
    ds4_gpu_tensor_free(g->value);
    ds4_gpu_tensor_free(g->k_nope);
    ds4_gpu_tensor_free(g->kv_norm);
    ds4_gpu_tensor_free(g->kv_raw);
    ds4_gpu_tensor_free(g->attn_partial_ms);
    ds4_gpu_tensor_free(g->attn_partial_lora);
    ds4_gpu_tensor_free(g->qk_low);
    ds4_gpu_tensor_free(g->indexer_selected);
    ds4_gpu_tensor_free(g->indexer_scores);
    ds4_gpu_tensor_free(g->indexer_weights);
    ds4_gpu_tensor_free(g->indexer_q);
    ds4_gpu_tensor_free(g->indexer_k);
    ds4_gpu_tensor_free(g->q);
    ds4_gpu_tensor_free(g->q_rank_norm);
    ds4_gpu_tensor_free(g->q_rank);
    ds4_gpu_tensor_free(g->attn_norm);
    ds4_gpu_tensor_free(g->next);
    ds4_gpu_tensor_free(g->cur);
    memset(g, 0, sizeof(*g));
}

static bool glm_graph_ensure_compact_cache(
        const ds4_glm_gpu_graph *g,
        uint32_t                 needed_rows) {
    if (!g || needed_rows == 0) return false;
    if (needed_rows <= g->ctx_cap && g->compact_cache_cap == 0) return true;
    if (needed_rows <= g->compact_cache_cap) return true;
    fprintf(stderr,
            "ds4: GLM compact DSA cache capacity %u is smaller than required row %u "
            "(ctx=%u, full_cap=%u)\n",
            g->compact_cache_cap,
            needed_rows,
            g->ctx_size,
            g->ctx_cap);
    return false;
}

static bool glm_graph_warm_compact_indexer_store(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           warm_pos) {
    if (!g || !model || !weights) return false;
    if (g->compact_cache_cap == 0 || g->indexer_full_layers == 0) return true;
    if (!g->indexer_k) return false;
    if (warm_pos >= g->compact_cache_cap) warm_pos = g->compact_cache_cap - 1u;

    if (ds4_gpu_tensor_fill_f32(g->indexer_k,
                                0.0f,
                                DS4_N_INDEXER_HEAD_DIM) == 0) {
        return false;
    }

    const bool profile = false;
    const double t0 = 0.0;
    bool ok = ds4_gpu_begin_commands() != 0;
    uint32_t warmed = 0;
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        if (!glm_graph_layer_uses_full_indexer(il)) continue;
        const ds4_layer_weights *l = &weights->layer[il];
        if (!g->layer_indexer_key_cache[il] ||
            !l->indexer_k_norm ||
            !l->indexer_k_norm_b) {
            ok = false;
            break;
        }
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        ok = ds4_gpu_glm_store_indexer_k_tensor(
                g->layer_indexer_key_cache[il],
                g->indexer_k,
                model->map,
                model->size,
                l->indexer_k_norm->abs_offset,
                l->indexer_k_norm_b->abs_offset,
                warm_pos,
                1,
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
        if (ok) warmed++;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();

    if (profile) {
        fprintf(stderr,
                "ds4: GLM compact indexer warmup pos=%u layers=%u %.3f ms\n",
                warm_pos,
                warmed,
                (now_sec() - t0) * 1000.0);
    }
    return ok;
}

static bool glm_graph_alloc_slice(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                ctx_size,
        bool               ssd_streaming,
        bool               ssd_streaming_cold,
        uint64_t           streaming_transient_guard_bytes,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               require_token_embd,
        bool               require_output_head) {
    if (!g || !model || !weights || ctx_size <= 0) return false;
    const int *placement = g->placement;
    memset(g, 0, sizeof(*g));
    g->placement = placement;
    g->ssd_streaming = ssd_streaming;
    g->ssd_streaming_cold = ssd_streaming_cold;

    if (!glm_graph_context_request(ctx_size, &g->ctx_size)) return false;
    if (!glm_graph_memory_guard_slice_with_transient(
                model,
                weights,
                g->ssd_streaming,
                layer_start,
                layer_end,
                require_token_embd,
                require_output_head,
                g->ctx_size,
                streaming_transient_guard_bytes,
                "before GLM graph allocation")) {
        return false;
    }
    if (!glm_graph_validate_layout(model,
                                   weights,
                                   g,
                                   layer_start,
                                   layer_end,
                                   require_token_embd,
                                   require_output_head)) {
        return false;
    }
    g->ctx_cap = glm_graph_full_attention_cap(g->ctx_size,
                                              g->ssd_streaming);
    g->full_kv_cache = glm_graph_expanded_kv_cache_enabled(g->ssd_streaming);
    g->compact_cache_cap =
        glm_graph_compact_cache_initial_cap(g->ctx_size, g->ctx_cap);
    g->indexed_prefill_cap =
        g->compact_cache_cap != 0 ?
            glm_graph_indexed_prefill_chunk_tokens(g->ctx_cap, g->compact_cache_cap) :
            0;
    g->indexed_prefill_score_cap =
        glm_graph_indexed_prefill_score_tokens(g->indexed_prefill_cap,
                                               g->compact_cache_cap);
    g->indexer_full_layers =
        glm_graph_full_indexer_layer_count_range(g->layer_start,
                                                 g->layer_end);
    if (g->ctx_size > g->ctx_cap) {
        fprintf(stderr,
                "ds4: GLM session ctx=%u (model max=%u); "
                "full-attention prefill/work cap=%u; compact indexed decode is used beyond the cap\n",
                g->ctx_size,
                glm_graph_model_context_limit(),
                g->ctx_cap);
    }

    const uint64_t emb_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t q_rank_bytes = (uint64_t)DS4_N_LORA_Q * sizeof(float);
    const uint64_t q_bytes = g->q_dim * sizeof(float);
    const uint64_t kv_raw_bytes = g->kv_raw_dim * sizeof(float);
    const uint64_t kv_norm_bytes = (uint64_t)DS4_N_KV_LORA * sizeof(float);
    const uint64_t k_nope_bytes = (uint64_t)DS4_N_HEAD * g->q_nope * sizeof(float);
    const uint64_t heads_bytes = g->heads_dim * sizeof(float);
    const uint64_t indexer_k_bytes = (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    const uint64_t indexer_q_bytes =
        (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    const uint64_t indexer_weights_bytes =
        (uint64_t)DS4_N_INDEXER_HEAD * sizeof(float);
    const uint64_t indexer_work_cap =
        g->compact_cache_cap != 0 ? g->compact_cache_cap : g->ctx_cap;
    const uint64_t indexer_scores_bytes = indexer_work_cap * sizeof(float);
    const uint32_t indexer_top_k = glm_graph_indexer_top_k_limit();
    const uint64_t indexer_selected_bytes =
        (uint64_t)indexer_top_k * sizeof(uint32_t);
    const uint64_t qk_low_bytes =
        (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float);
    const uint32_t split_attn_blocks = glm_graph_indexed_decode_split_blocks();
    const uint64_t attn_partial_lora_bytes =
        (uint64_t)split_attn_blocks * qk_low_bytes;
    const uint64_t attn_partial_ms_bytes =
        (uint64_t)split_attn_blocks * DS4_N_HEAD * 2u * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    const uint64_t full_kv_elem_bytes = glm_graph_full_kv_cache_elem_bytes();
    const uint64_t key_cache_bytes = g->full_kv_cache ?
        (uint64_t)g->ctx_cap * g->q_dim * full_kv_elem_bytes : 0;
    const uint64_t value_cache_bytes = g->full_kv_cache ?
        (uint64_t)g->ctx_cap * g->heads_dim * full_kv_elem_bytes : 0;
    const uint64_t compact_kv_lora_bytes =
        (uint64_t)g->compact_cache_cap * DS4_N_KV_LORA *
        glm_graph_compact_cache_elem_bytes();
    const uint64_t compact_k_rope_bytes =
        (uint64_t)g->compact_cache_cap * DS4_N_ROT *
        glm_graph_compact_cache_elem_bytes();
    const uint64_t compact_indexer_key_bytes =
        (uint64_t)g->compact_cache_cap * DS4_N_INDEXER_HEAD_DIM *
        glm_graph_compact_cache_elem_bytes();
    const uint64_t batch_rows =
        g->full_kv_cache || g->indexed_prefill_cap == 0 ?
            g->ctx_cap :
            g->indexed_prefill_cap;
    const uint64_t indexed_batch_rows = g->indexed_prefill_cap;
    const uint64_t indexed_score_rows = g->indexed_prefill_score_cap;
    const uint64_t batch_indexer_q_bytes = indexed_batch_rows * indexer_q_bytes;
    const uint64_t batch_indexer_weights_bytes = indexed_batch_rows * indexer_weights_bytes;
    const uint64_t batch_indexer_scores_bytes =
        indexed_score_rows * indexer_work_cap * sizeof(float);
    const uint64_t batch_indexer_selected_bytes =
        indexed_batch_rows * indexer_top_k * sizeof(uint32_t);
    const uint64_t batch_qk_low_bytes = indexed_batch_rows * qk_low_bytes;
    const uint64_t batch_attn_lora_bytes =
        indexed_batch_rows * (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float);
    const uint64_t routed_mid_bytes =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(float);
    const uint64_t routed_down_bytes =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float);
    const double cache_gib =
        (double)(g->layer_count * (key_cache_bytes + value_cache_bytes)) /
        (1024.0 * 1024.0 * 1024.0);
    if (g->full_kv_cache) {
        fprintf(stderr,
                "ds4: GLM graph allocating full-attention KV cache: work_ctx=%u layers=%u:%u (%u) %s %.2f GiB\n",
                g->ctx_cap,
                g->layer_start,
                g->layer_end,
                g->layer_count,
                "f16",
                cache_gib);
    } else {
        fprintf(stderr,
                "ds4: GLM graph using compact DSA KV only; expanded full-attention KV cache is skipped\n");
    }
#ifdef DS4_ROCM_BUILD
    if (glm_graph_env_truthy(
            getenv("DS4_ROCM_GLM_LAYER_SLICE_TOKEN_DECODE"))) {
        fprintf(stderr,
                "ds4: ROCm GLM one-token layer slices use the optimized token graph\n");
    }
#endif
    if (g->compact_cache_cap != 0) {
        const uint64_t compact_kv_total =
            (uint64_t)g->layer_count * (compact_kv_lora_bytes + compact_k_rope_bytes);
        const uint64_t compact_indexer_total =
            (uint64_t)g->indexer_full_layers * compact_indexer_key_bytes;
        const double compact_gib =
            (double)(compact_kv_total + compact_indexer_total) /
            (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr,
                "ds4: GLM graph allocating compact DSA cache: rows=%u logical_ctx=%u kv_layers=%u indexer_layers=%u %s %.2f GiB\n",
                g->compact_cache_cap,
                g->ctx_size,
                g->layer_count,
                g->indexer_full_layers,
                glm_graph_compact_cache_is_f16() ? "f16" : "f32",
                compact_gib);
        fprintf(stderr,
                "ds4: GLM compact indexed prefill chunk=%u score_rows=%u score_scratch=%.2f MiB\n",
                g->indexed_prefill_cap,
                g->indexed_prefill_score_cap,
                (double)batch_indexer_scores_bytes / (1024.0 * 1024.0));
    }

    bool ok = true;
#define DS4_GLM_GRAPH_ALLOC_TENSOR(var, bytes_) \
    do { \
        (var) = ds4_gpu_tensor_alloc((bytes_)); \
        if (!(var)) { \
            fprintf(stderr, "ds4: GLM graph could not allocate %s\n", #var); \
            ok = false; \
        } \
    } while (0)

    DS4_GLM_GRAPH_ALLOC_TENSOR(g->cur, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->next, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->attn_norm, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->q_rank, q_rank_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->q_rank_norm, q_rank_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->q, q_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_k, indexer_k_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_q, indexer_q_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_weights, indexer_weights_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_scores, indexer_scores_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_selected, indexer_selected_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->qk_low, qk_low_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->attn_partial_lora, attn_partial_lora_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->attn_partial_ms, attn_partial_ms_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->kv_raw, kv_raw_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->kv_norm, kv_norm_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->k_nope, k_nope_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->value, heads_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->heads, heads_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->attn_out, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->after_attn, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_norm, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_gate, g->dense_hidden_max * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_up, g->dense_hidden_max * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_mid, g->ffn_mid_elems * sizeof(float));
    if (g->generic_routed_moe) {
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->routed_gate, routed_mid_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->routed_up, routed_mid_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->routed_down, routed_down_bytes);
    }
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_out, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_sum, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->router_logits, (uint64_t)DS4_N_EXPERT * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->router_probs, (uint64_t)DS4_N_EXPERT * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->router_selected, (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->router_weights, (uint64_t)DS4_N_EXPERT_USED * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->output_norm, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->logits, logits_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_router_logits, batch_rows * DS4_N_EXPERT * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_router_probs, batch_rows * DS4_N_EXPERT * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_router_selected, batch_rows * DS4_N_EXPERT_USED * sizeof(int32_t));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_router_weights, batch_rows * DS4_N_EXPERT_USED * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->prefill_seed_router_selected,
                               (uint64_t)DS4_N_LAYER *
                               DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                               DS4_N_EXPERT_USED *
                               sizeof(int32_t));

    DS4_GLM_GRAPH_ALLOC_TENSOR(g->prefill_tokens, batch_rows * sizeof(int32_t));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_cur, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_next, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_attn_norm, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_q_rank, batch_rows * q_rank_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_q_rank_norm, batch_rows * q_rank_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_q, batch_rows * q_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_k, batch_rows * indexer_k_bytes);
    if (g->compact_cache_cap != 0 && g->indexed_prefill_cap != 0) {
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_q, batch_indexer_q_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_weights, batch_indexer_weights_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_scores, batch_indexer_scores_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_selected, batch_indexer_selected_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_qk_low, batch_qk_low_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_attn_lora, batch_attn_lora_bytes);
    }
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_kv_raw, batch_rows * kv_raw_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_kv_norm, batch_rows * kv_norm_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_k_nope, batch_rows * k_nope_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_value, batch_rows * heads_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_heads, batch_rows * heads_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_attn_out, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_after_attn, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_norm, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_gate, batch_rows * g->dense_hidden_max * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_up, batch_rows * g->dense_hidden_max * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_shared_mid, batch_rows * DS4_N_FF_EXP * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_mid, batch_rows * g->ffn_mid_elems * sizeof(float));
    if (g->generic_routed_moe) {
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_routed_gate, batch_rows * routed_mid_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_routed_up, batch_rows * routed_mid_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_routed_down, batch_rows * routed_down_bytes);
    }
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_out, batch_rows * emb_bytes);

    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        int cache_tier = 0;
        if (g->placement && g->placement[il + 1u] >= 0 &&
            g->placement[il + 1u] < DS4_MAX_GPUS) {
            cache_tier = g->placement[il + 1u];
        }
#define DS4_GLM_GRAPH_ALLOC_TENSOR_TIER(var, bytes_) \
    do { \
        (var) = ds4_gpu_tensor_alloc_ptr_on(cache_tier, (bytes_)); \
        if (!(var)) { \
            fprintf(stderr, "ds4: GLM graph could not allocate %s on tier %d\n", \
                    #var, cache_tier); \
            ok = false; \
        } \
    } while (0)
        if (g->full_kv_cache) {
            DS4_GLM_GRAPH_ALLOC_TENSOR_TIER(g->layer_key_cache[il], key_cache_bytes);
            DS4_GLM_GRAPH_ALLOC_TENSOR_TIER(g->layer_value_cache[il], value_cache_bytes);
        }
        if (g->compact_cache_cap != 0) {
            DS4_GLM_GRAPH_ALLOC_TENSOR_TIER(g->layer_kv_lora_cache[il], compact_kv_lora_bytes);
            DS4_GLM_GRAPH_ALLOC_TENSOR_TIER(g->layer_k_rope_cache[il], compact_k_rope_bytes);
            if (glm_graph_layer_uses_full_indexer(il)) {
                DS4_GLM_GRAPH_ALLOC_TENSOR_TIER(g->layer_indexer_key_cache[il],
                                                compact_indexer_key_bytes);
            }
        }
#undef DS4_GLM_GRAPH_ALLOC_TENSOR_TIER
    }
#undef DS4_GLM_GRAPH_ALLOC_TENSOR

    if (!ok) {
        glm_graph_free(g);
        return false;
    }
    glm_graph_ws_init(g);
    return true;
}

static bool glm_graph_alloc(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                ctx_size,
        bool               ssd_streaming,
        bool               ssd_streaming_cold) {
    const uint32_t normal_layers = glm_graph_normal_layer_count();
    if (normal_layers == 0) {
        fprintf(stderr, "ds4: GLM Metal graph found no normal transformer layers\n");
        return false;
    }
    return glm_graph_alloc_slice(g,
                                 model,
                                 weights,
                                 ctx_size,
                                 ssd_streaming,
                                 ssd_streaming_cold,
                                 0,
                                 0,
                                 normal_layers - 1u,
                                 true,
                                 true);
}

static uint32_t glm_graph_weight_type_for_offset(
        const ds4_model *model,
        uint64_t         weight_offset);

static int glm_graph_matmul_q8_0_decode_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        bool                  ssd_streaming) {
    if (!model) return 0;
    const uint32_t weight_type = glm_graph_weight_type_for_offset(model, weight_offset);
    if (ssd_streaming) {
        return ds4_gpu_matmul_quant_tensor(out,
                                           model->map,
                                           model->size,
                                           weight_offset,
                                           weight_type,
                                           in_dim,
                                           out_dim,
                                           x,
                                           1);
    }
    return ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(out,
                                                             model->map,
                                                             model->size,
                                                             weight_offset,
                                                             weight_type,
                                                             in_dim,
                                                             out_dim,
                                                             x,
                                                             1);
}

static bool glm_graph_q8_decode_profile_enabled(uint32_t il, const char *label) {
    (void)il;
    (void)label;
    return false;
}

static uint32_t glm_graph_weight_type_for_offset(
        const ds4_model *model,
        uint64_t         weight_offset) {
    if (!model || !model->tensors) return DS4_TENSOR_Q8_0;
    for (uint64_t i = 0; i < model->n_tensors; i++) {
        const ds4_tensor *t = &model->tensors[i];
        if (t->abs_offset == weight_offset) return t->type;
    }
    return DS4_TENSOR_Q8_0;
}

static bool glm_graph_weights_are_q8_0(
        const ds4_model *model,
        uint64_t         offset_a,
        uint64_t         offset_b) {
    return glm_graph_weight_type_for_offset(model, offset_a) == DS4_TENSOR_Q8_0 &&
           glm_graph_weight_type_for_offset(model, offset_b) == DS4_TENSOR_Q8_0;
}

static int glm_graph_matmul_q8_0_decode_profiled_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              il,
        uint32_t              pos,
        const char           *label,
        bool                  ssd_streaming) {
    const bool profile = glm_graph_q8_decode_profile_enabled(il, label);
    if (profile) {
        if (ds4_gpu_end_commands() == 0) return 0;
        if (ds4_gpu_begin_commands() == 0) return 0;
    }
    const double t0 = profile ? now_sec() : 0.0;
    int ok = glm_graph_matmul_q8_0_decode_tensor(out,
                                                 model,
                                                 weight_offset,
                                                 in_dim,
                                                 out_dim,
                                                 x,
                                                 ssd_streaming);
    if (profile) {
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        const double now = now_sec();
        fprintf(stderr,
                "ds4: GLM Q8 decode profile layer=%u pos=%u label=%s in=%llu out=%llu %.3f ms\n",
                il,
                pos,
                label ? label : "?",
                (unsigned long long)in_dim,
                (unsigned long long)out_dim,
                (now - t0) * 1000.0);
        if (ok) ok = ds4_gpu_begin_commands() != 0;
    }
    return ok;
}

static bool glm_graph_encode_output_head_from(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        const ds4_gpu_tensor *hidden) {
    bool ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                             hidden,
                                             model->map,
                                             model->size,
                                             weights->output_norm->abs_offset,
                                             DS4_N_EMBD,
                                             DS4_RMS_EPS) != 0;
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->logits,
                                                     model,
                                                     weights->output->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_VOCAB,
                                                     g->output_norm,
                                                     g->ssd_streaming) != 0;
    return ok;
}

static bool glm_graph_encode_output_head(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    return glm_graph_encode_output_head_from(g, model, weights, g->cur);
}

static bool glm_graph_forward_output_head(
        ds4_glm_gpu_graph    *g,
        const ds4_model      *model,
        const ds4_weights    *weights,
        const ds4_gpu_tensor *hidden,
        float                *logits_out) {
    if (!g || !model || !weights || !hidden || !logits_out) return false;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = glm_graph_encode_output_head_from(g, model, weights, hidden);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok && glm_debug_hidden_dump_layer() < 0)
        glm_debug_dump_hidden_row(hidden, 0);
    if (ok) {
        ok = ds4_gpu_tensor_read(g->logits,
                                 0,
                                 logits_out,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    return ok;
}

static bool glm_graph_profile_stage(
        bool        enabled,
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (!enabled) return true;
    if (!stage_t0) return false;
    return metal_graph_layer_stage_profile_boundary(part, stage, il, pos0, n_tokens, stage_t0);
}

static bool glm_graph_profile_router_selection(
        ds4_glm_gpu_graph       *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 pos) {
    if (!g_expert_profile.active) return true;
    if (!g || !layer || !g->router_selected || !g->router_weights) return false;

    if (ds4_gpu_end_commands() == 0) {
        fprintf(stderr,
                "ds4: failed to end GLM Metal command batch for expert profile readback\n");
        return false;
    }

    int32_t selected[DS4_MAX_EXPERT_USED] = {0};
    float weights[DS4_MAX_EXPERT_USED] = {0};
    const bool read_ok =
        ds4_gpu_tensor_read(g->router_selected,
                            0,
                            selected,
                            (uint64_t)DS4_N_EXPERT_USED * sizeof(selected[0])) != 0 &&
        ds4_gpu_tensor_read(g->router_weights,
                            0,
                            weights,
                            (uint64_t)DS4_N_EXPERT_USED * sizeof(weights[0])) != 0;

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr,
                "ds4: failed to resume GLM Metal command batch after expert profile readback\n");
        return false;
    }
    if (!read_ok) {
        fprintf(stderr, "ds4: failed to read GLM Metal router tensors for expert profile\n");
        return false;
    }

    ds4_expert_profile_record(il,
                              pos,
                              selected,
                              weights,
                              layer->ffn_gate_tid2eid != NULL);
    return true;
}

static bool glm_graph_profile_router_selection_batch(
        ds4_glm_gpu_graph       *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    if (!g_expert_profile.active) return true;
    if (!g || !layer || !g->batch_router_selected ||
        !g->batch_router_weights || n_tokens == 0) {
        return false;
    }

    const size_t selected_count = (size_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_tokens != 0 && selected_count / n_tokens != DS4_N_EXPERT_USED) {
        return false;
    }
    if (selected_count > SIZE_MAX / sizeof(int32_t) ||
        selected_count > SIZE_MAX / sizeof(float)) {
        return false;
    }

    if (ds4_gpu_end_commands() == 0) {
        fprintf(stderr,
                "ds4: failed to end GLM Metal command batch for batch expert profile readback\n");
        return false;
    }

    int32_t *selected = xmalloc(selected_count * sizeof(selected[0]));
    float *weights = xmalloc(selected_count * sizeof(weights[0]));
    const bool read_ok =
        ds4_gpu_tensor_read(g->batch_router_selected,
                            0,
                            selected,
                            (uint64_t)selected_count * sizeof(selected[0])) != 0 &&
        ds4_gpu_tensor_read(g->batch_router_weights,
                            0,
                            weights,
                            (uint64_t)selected_count * sizeof(weights[0])) != 0;

    if (ds4_gpu_begin_commands() == 0) {
        free(weights);
        free(selected);
        fprintf(stderr,
                "ds4: failed to resume GLM Metal command batch after batch expert profile readback\n");
        return false;
    }
    if (!read_ok) {
        free(weights);
        free(selected);
        fprintf(stderr, "ds4: failed to read GLM Metal batch router tensors for expert profile\n");
        return false;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const size_t off = (size_t)t * DS4_N_EXPERT_USED;
        ds4_expert_profile_record(il,
                                  pos0 + t,
                                  selected + off,
                                  weights + off,
                                  layer->ffn_gate_tid2eid != NULL);
    }

    free(weights);
    free(selected);
    return true;
}

static bool glm_graph_prefill_stage_boundary(
        bool        stage_profile,
        bool        stage_sync,
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (stage_profile) {
        return glm_graph_profile_stage(true, part, stage, il, pos0, n_tokens, stage_t0);
    }
    if (stage_sync) return glm_graph_prefill_stage_sync_boundary();
    return true;
}

static int glm_graph_routed_moe_one_dispatch(
        const ds4_glm_gpu_graph *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        ds4_gpu_tensor          *out,
        ds4_gpu_tensor          *mid,
        uint64_t                 gate_expert_bytes,
        uint64_t                 gate_row_bytes,
        uint64_t                 up_expert_bytes,
        uint64_t                 up_row_bytes,
        uint64_t                 down_expert_bytes,
        uint64_t                 down_row_bytes,
        const ds4_gpu_tensor    *selected,
        const ds4_gpu_tensor    *weights,
        const ds4_gpu_tensor    *x,
        bool                     force_resident) {
    if (!g || !model || !l) return 0;
    /* Under the TP expert split only the ownership-aware kernels may run:
     * the generic mul_mv_id family and the GLM q2_K resident pair/down.
     * Anything else would silently compute the full expert set. */
    if (g->tp_world == 2 &&
        !glm_graph_layer_uses_generic_routed_moe(l) &&
        l->ffn_gate_exps->type != DS4_TENSOR_Q2_K) {
        fprintf(stderr,
                "ds4: GLM TP split lacks ownership-aware kernels for expert type %u (layer %u)\n",
                l->ffn_gate_exps->type, il);
        return 0;
    }
    if (glm_graph_layer_uses_generic_routed_moe(l)) {
        if (!g->routed_gate || !g->routed_up || !g->routed_down ||
            l->ffn_gate_exps->type != l->ffn_up_exps->type) {
            if (getenv("DS4_GLM_TP_DEBUG")) {
                fprintf(stderr,
                        "ds4: glm dispatch guard: gate=%p up=%p down=%p types=%u/%u\n",
                        (void *)g->routed_gate, (void *)g->routed_up,
                        (void *)g->routed_down,
                        l->ffn_gate_exps->type, l->ffn_up_exps->type);
            }
            return 0;
        }
        return ds4_gpu_routed_moe_one_tensor(out,
                                             g->routed_gate,
                                             g->routed_up,
                                             mid,
                                             g->routed_down,
                                             model->map,
                                             model->size,
                                             l->ffn_gate_exps->abs_offset,
                                             l->ffn_up_exps->abs_offset,
                                             l->ffn_down_exps->abs_offset,
                                             l->ffn_gate_exps->type,
                                             l->ffn_down_exps->type,
                                             gate_expert_bytes,
                                             gate_row_bytes,
                                             down_expert_bytes,
                                             down_row_bytes,
                                             DS4_N_EMBD,
                                             DS4_N_FF_EXP,
                                             DS4_N_EMBD,
                                             selected,
                                             weights,
                                             DS4_N_EXPERT,
                                             DS4_N_EXPERT_USED,
                                             0.0f,
                                             x,
                                             NULL,
                                             il,
                                             force_resident);
    }

    return ds4_gpu_glm_routed_moe_one_tensor(out,
                                             mid,
                                             model->map,
                                             model->size,
                                             l->ffn_gate_exps->abs_offset,
                                             l->ffn_up_exps->abs_offset,
                                             l->ffn_down_exps->abs_offset,
                                             l->ffn_gate_exps->type,
                                             l->ffn_up_exps->type,
                                             l->ffn_down_exps->type,
                                             gate_expert_bytes,
                                             gate_row_bytes,
                                             up_expert_bytes,
                                             up_row_bytes,
                                             down_expert_bytes,
                                             down_row_bytes,
                                             DS4_N_EMBD,
                                             DS4_N_FF_EXP,
                                             DS4_N_EMBD,
                                             selected,
                                             weights,
                                             DS4_N_EXPERT,
                                             DS4_N_EXPERT_USED,
                                             il,
                                             x,
                                             force_resident);
}

/* Post-compute visibility for GLM TP debugging: the combine stashes the
 * selected-ids contents pointer; by exchange time the router kernels have
 * completed, so the service thread sees final ids. */
static const int32_t *g_glm_tp_debug_ids DS4_MAYBE_UNUSED;

/* After the TP ownership-split batch routed MoE, exchange the
 * per-token routed partial rows with the peer through shared bounce
 * buffers (one gate per sparse layer per chunk) and rebuild the full
 * routed output with a commutative add. */
/* The routed batch dispatch writes its local partial DIRECTLY into the
 * shared bounce buffer (graph scratch may be private/untracked on M5, so
 * a blit from it is not reliable); ensure capacity before dispatching. */
static bool glm_graph_tp_batch_bounce_ready(ds4_glm_gpu_graph *g,
                                            uint32_t n_tokens) {
    const uint64_t bytes = (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float);
    if (!g->tp_bounce_out || ds4_gpu_tensor_bytes(g->tp_bounce_out) < bytes) {
        ds4_gpu_tensor_free(g->tp_bounce_out);
        ds4_gpu_tensor_free(g->tp_bounce_in);
        g->tp_bounce_out = ds4_gpu_tensor_alloc(bytes);
        g->tp_bounce_in = ds4_gpu_tensor_alloc(bytes);
    }
    return g->tp_bounce_out && g->tp_bounce_in;
}

static bool glm_graph_tp_batch_ffn_combine(
        ds4_glm_gpu_graph *g,
        uint32_t           il,
        ds4_gpu_tensor    *ffn_out,
        uint32_t           n_tokens) {
    const uint64_t bytes = (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float);
    if (!g->tp_bounce_out || !g->tp_bounce_in) return false;
    if (getenv("DS4_GLM_ABLATE_COMBINE")) {
        /* Timing probe: local half only, no exchange (garbage output;
         * both ranks must set the env or the gates desync). */
        return ds4_gpu_add_tensor(ffn_out,
                                  g->tp_bounce_out,
                                  g->tp_bounce_out,
                                  (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
    }
    if (!ds4_gpu_tp_big_gate_encode(il, n_tokens,
                                    g->tp_bounce_out, g->tp_bounce_in,
                                    bytes)) {
        return false;
    }
    return ds4_gpu_add_tensor(ffn_out,
                              g->tp_bounce_out,
                              g->tp_bounce_in,
                              (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
}

static int glm_graph_routed_moe_batch_dispatch(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        ds4_gpu_tensor          *out,
        ds4_gpu_tensor          *mid,
        uint64_t                 gate_expert_bytes,
        uint64_t                 gate_row_bytes,
        uint64_t                 up_expert_bytes,
        uint64_t                 up_row_bytes,
        uint64_t                 down_expert_bytes,
        uint64_t                 down_row_bytes,
        const ds4_gpu_tensor    *selected,
        const ds4_gpu_tensor    *weights,
        const ds4_gpu_tensor    *x,
        uint32_t                 n_tokens,
        uint32_t                 mid_token_stride,
        bool                     force_resident,
        bool                     direct_scalar_q4) {
    if (!g || !model || !l) return 0;
    g->batch_routed_mid_is_f16 = false;

    if (glm_graph_layer_uses_generic_routed_moe(l)) {
        if (!g->batch_routed_gate || !g->batch_routed_up || !g->batch_routed_down ||
            l->ffn_gate_exps->type != l->ffn_up_exps->type) {
            return 0;
        }
        return ds4_gpu_routed_moe_batch_tensor(out,
                                               g->batch_routed_gate,
                                               g->batch_routed_up,
                                               mid,
                                               g->batch_routed_down,
                                               model->map,
                                               model->size,
                                               l->ffn_gate_exps->abs_offset,
                                               l->ffn_up_exps->abs_offset,
                                               l->ffn_down_exps->abs_offset,
                                               l->ffn_gate_exps->type,
                                               l->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               DS4_N_EMBD,
                                               DS4_N_FF_EXP,
                                               DS4_N_EMBD,
                                               selected,
                                               weights,
                                               DS4_N_EXPERT,
                                               DS4_N_EXPERT_USED,
                                               0.0f,
                                               x,
                                               il,
                                               n_tokens,
                                               &g->batch_routed_mid_is_f16,
                                               force_resident);
    }

    if (direct_scalar_q4) {
        return ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
                out,
                mid,
                model->map,
                model->size,
                l->ffn_gate_exps->abs_offset,
                l->ffn_up_exps->abs_offset,
                l->ffn_down_exps->abs_offset,
                l->ffn_gate_exps->type,
                l->ffn_up_exps->type,
                l->ffn_down_exps->type,
                gate_expert_bytes,
                gate_row_bytes,
                up_expert_bytes,
                up_row_bytes,
                down_expert_bytes,
                down_row_bytes,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                DS4_N_EMBD,
                selected,
                weights,
                DS4_N_EXPERT,
                DS4_N_EXPERT_USED,
                il,
                x,
                n_tokens,
                mid_token_stride);
    }

    return ds4_gpu_glm_routed_moe_batch_tensor(
            out,
            mid,
            model->map,
            model->size,
            l->ffn_gate_exps->abs_offset,
            l->ffn_up_exps->abs_offset,
            l->ffn_down_exps->abs_offset,
            l->ffn_gate_exps->type,
            l->ffn_up_exps->type,
            l->ffn_down_exps->type,
            gate_expert_bytes,
            gate_row_bytes,
            up_expert_bytes,
            up_row_bytes,
            down_expert_bytes,
            down_row_bytes,
            DS4_N_EMBD,
            DS4_N_FF_EXP,
            DS4_N_EMBD,
            selected,
            weights,
            DS4_N_EXPERT,
            DS4_N_EXPERT_USED,
            il,
            x,
            n_tokens,
            mid_token_stride,
            force_resident);
}

static bool glm_graph_disable_add3_residual(void);

static bool glm_graph_use_streaming_selected_async_load(
        const ds4_glm_gpu_graph *g) {
    if (!g || !g->ssd_streaming) return false;
    if (glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_SELECTED_ASYNC_LOAD",
                              "DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_LOAD") ||
        glm_graph_env_present("DS4_ROCM_DISABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD",
                              "DS4_METAL_DISABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD")) {
        return false;
    }
#ifdef DS4_ROCM_BUILD
    return true;
#else
    return getenv("DS4_METAL_ENABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD") != NULL;
#endif
}

typedef struct glm_graph_streaming_async_profile {
    uint64_t async_calls;
    uint64_t sync_calls;
    double async_total_ms;
    double sync_total_ms;
    double async_signal_start_ms;
    double async_flush_router_ms;
    double async_shared_ms;
    double async_flush_shared_ms;
    double async_finish_ms;
    double async_routed_ms;
    double async_post_ms;
    double sync_early_load_ms;
    double sync_shared_ms;
    double sync_routed_ms;
    double sync_post_ms;
} glm_graph_streaming_async_profile;

static glm_graph_streaming_async_profile g_glm_streaming_async_profile;
static bool g_glm_streaming_async_profile_registered;

static bool glm_graph_streaming_async_profile_enabled(void) {
    return glm_graph_env_present("DS4_ROCM_GLM_STREAMING_ASYNC_PROFILE",
                                 "DS4_METAL_GLM_STREAMING_ASYNC_PROFILE");
}

static void glm_graph_streaming_async_profile_print(void) {
    const glm_graph_streaming_async_profile *p =
        &g_glm_streaming_async_profile;
    if (p->async_calls == 0 && p->sync_calls == 0) return;
    const double async_calls = p->async_calls ? (double)p->async_calls : 1.0;
    const double sync_calls = p->sync_calls ? (double)p->sync_calls : 1.0;
    fprintf(stderr,
            "ds4: GLM streaming async profile async_calls=%llu "
            "total=%.3f ms avg=%.3f ms signal_start=%.3f ms "
            "flush_router=%.3f ms shared=%.3f ms flush_shared=%.3f ms "
            "finish=%.3f ms routed=%.3f ms post=%.3f ms\n",
            (unsigned long long)p->async_calls,
            p->async_total_ms,
            p->async_total_ms / async_calls,
            p->async_signal_start_ms,
            p->async_flush_router_ms,
            p->async_shared_ms,
            p->async_flush_shared_ms,
            p->async_finish_ms,
            p->async_routed_ms,
            p->async_post_ms);
    fprintf(stderr,
            "ds4: GLM streaming sync profile calls=%llu "
            "total=%.3f ms avg=%.3f ms early_load=%.3f ms "
            "shared=%.3f ms routed=%.3f ms post=%.3f ms\n",
            (unsigned long long)p->sync_calls,
            p->sync_total_ms,
            p->sync_total_ms / sync_calls,
            p->sync_early_load_ms,
            p->sync_shared_ms,
            p->sync_routed_ms,
            p->sync_post_ms);
}

static void glm_graph_streaming_async_profile_register(void) {
    if (g_glm_streaming_async_profile_registered) return;
    if (!glm_graph_streaming_async_profile_enabled()) return;
    atexit(glm_graph_streaming_async_profile_print);
    g_glm_streaming_async_profile_registered = true;
}

static double glm_graph_streaming_async_profile_ms(void) {
    return now_sec() * 1000.0;
}

