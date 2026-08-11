#ifndef DS4_NO_GPU
static uint32_t glm_graph_prefill_chunk_tokens(uint32_t full_attention_cap) {
    return full_attention_cap ? full_attention_cap : DS4_GLM_METAL_FULL_ATTN_DEFAULT_CONTEXT;
}

static bool glm_graph_forward_indexed_tokens(
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
        g->compact_cache_cap == 0 ||
        g->indexed_prefill_cap == 0 ||
        g->indexed_prefill_score_cap == 0 ||
        !g->batch_indexer_q ||
        !g->batch_indexer_weights ||
        !g->batch_indexer_scores ||
        !g->batch_indexer_selected ||
        !g->batch_qk_low ||
        n_tokens == 0 ||
        g->layer_count == 0 ||
        n_tokens > g->indexed_prefill_cap ||
        !glm_graph_span_fits_context(g, pos0, n_tokens)) {
        return false;
    }
    const uint32_t n_rows = pos0 + n_tokens;
    const uint32_t indexer_top_k = glm_graph_indexer_top_k_limit();
    if (pos0 < indexer_top_k && n_rows > indexer_top_k) {
        return false;
    }
    const uint32_t indexed_selected_count =
        n_rows <= indexer_top_k ? n_rows : indexer_top_k;
    const bool use_causal_range_select = n_rows <= indexer_top_k;
    const bool trace = glm_graph_indexed_prefill_trace_enabled();
    const bool trace_all = trace && glm_graph_indexed_prefill_trace_all();
    const double trace_slow_ms = trace ?
        (double)glm_graph_indexed_prefill_trace_slow_ms() : 0.0;
    const double trace_chunk_t0 = trace ? now_sec() : 0.0;
    if (trace) {
        glm_graph_indexed_prefill_tracef(
                "chunk begin pos=%u tokens=%u rows=%u selected=%u compact_cap=%u score_cap=%u work_base=%u work_total=%u",
                pos0,
                n_tokens,
                n_rows,
                indexed_selected_count,
                g->compact_cache_cap,
                g->indexed_prefill_score_cap,
                work_done_base,
                work_total);
    }
    const double trace_cache_t0 = trace ? now_sec() : 0.0;
    if (!glm_graph_ensure_compact_cache(g, n_rows)) {
        if (trace) {
            glm_graph_indexed_prefill_tracef(
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
            glm_graph_indexed_prefill_tracef(
                    "ensure_cache done pos=%u tokens=%u rows=%u compact_cap=%u %.3f ms",
                    pos0,
                    n_tokens,
                    n_rows,
                    g->compact_cache_cap,
                    ms);
        }
    }
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= (int)DS4_N_VOCAB) return false;
    }
    if (!input_hc && !g->has_token_embd) return false;
    if (logits_out && !g->has_output_head) return false;
    glm_graph_reset_prefill_seed_capture(g);

    const double trace_upload_t0 = trace ? now_sec() : 0.0;
    bool ok = glm_graph_upload_tokens(g->prefill_tokens, tokens, n_tokens);
    if (trace) {
        const double ms = (now_sec() - trace_upload_t0) * 1000.0;
        if (trace_all || ms >= trace_slow_ms || !ok) {
            glm_graph_indexed_prefill_tracef(
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

    const bool use_all_scalar_kernels =
        n_tokens == 1u && glm_graph_indexed_prefill_scalar_kernels();
    const bool use_scalar_indexer =
        use_all_scalar_kernels ||
        glm_graph_indexed_prefill_scalar_indexer() ||
        !glm_graph_indexed_prefill_batch_indexer();
    const bool force_scalar_attn =
        use_all_scalar_kernels || glm_graph_indexed_prefill_scalar_attn();
    const bool use_batch_qk_low =
        !force_scalar_attn && glm_graph_indexed_prefill_batch_qk_low();
    const bool use_batch_attn_kernel =
        !force_scalar_attn && glm_graph_indexed_prefill_batch_attn_kernel();
    const bool use_split_value_proj =
        use_batch_attn_kernel &&
        g->batch_attn_lora;
    /* Tensor-parallel attention head split: each rank computes half the
     * heads in the qk-low / attention-lora / value-project kernels, the
     * unowned half of batch_heads stays zero, and the full-width attn
     * output projection yields partials combined over the big-gate
     * exchange (same commutative add as the routed-FFN combine). Only the
     * split-value-proj batch chain has head ownership. */
    const bool tp_attn_head_split =
        g->tp_world == 2 &&
        use_batch_attn_kernel &&
        use_split_value_proj &&
        (DS4_N_HEAD % 16u) == 0u &&
        n_tokens >= glm_tp_head_split_min(); /* small batches replicate;
                          * the floor is env-tunable for correctness
                          * isolation (DS4_GLM_TP_HEAD_SPLIT_MIN). */
    const bool use_batch_q_rank_proj = true;
    const bool use_batch_q_proj = true;
    const bool use_batch_indexer_k_proj = true;
    const bool use_batch_kv_proj = true;
    const bool use_batch_indexer_q_proj = true;
    const bool use_batch_indexer_weights_proj = true;
    const bool use_batch_attn_out_proj = true;
    const bool use_batch_ffn = glm_graph_indexed_prefill_batch_ffn();
    const bool stage_sync =
        glm_graph_small_prefill_stage_sync(n_tokens, logits_out != NULL);
    const uint32_t layer_flush_interval = stage_sync ? 0u :
        glm_graph_full_prefill_layer_flush_interval(n_tokens,
                                                    n_tokens,
                                                    logits_out != NULL);
    const uint32_t progress_flush_interval =
        glm_graph_prefill_progress_flush_interval(layer_flush_interval,
                                                  n_tokens,
                                                  display_progress,
                                                  work_total);
    const uint32_t drain_interval =
        progress_flush_interval != 0 ? glm_graph_indexed_prefill_drain_interval() : 0u;
    const bool progress_requested = display_progress && work_total > 0;
    ds4_gpu_set_glm_streaming_prefill_full_layer(false);
    const bool streaming_prefill_sync_each_layer =
        !g->ssd_streaming ||
        glm_graph_streaming_prefill_sync_each_layer(false);

    if (trace) {
        glm_graph_indexed_prefill_tracef(
                "mode pos=%u tokens=%u scalar_indexer=%u batch_qk_low=%u batch_attn=%u split_value=%u batch_ffn=%u progress_flush_interval=%u drain_interval=%u",
                pos0,
                n_tokens,
                use_scalar_indexer ? 1u : 0u,
                use_batch_qk_low ? 1u : 0u,
                use_batch_attn_kernel ? 1u : 0u,
                use_split_value_proj ? 1u : 0u,
                use_batch_ffn ? 1u : 0u,
                progress_flush_interval,
                drain_interval);
    }

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
                glm_graph_indexed_prefill_tracef(
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
                glm_graph_indexed_prefill_tracef(
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

#define DS4_GLM_PROFILE_INDEXED_STAGE(part_, name_) do { \
        if (ok && trace) { \
            const double _trace_stage_now = now_sec(); \
            const double _trace_stage_ms = (_trace_stage_now - trace_stage_t0) * 1000.0; \
            if (trace_all || _trace_stage_ms >= trace_slow_ms) { \
                glm_graph_indexed_prefill_tracef( \
                        "stage layer=%u pos=%u tokens=%u %s.%s encode %.3f ms", \
                        il, \
                        pos0, \
                        n_tokens, \
                        (part_), \
                        (name_), \
                        _trace_stage_ms); \
            } \
            trace_stage_t0 = _trace_stage_now; \
        } \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary((part_), (name_), il, pos0, n_tokens, &layer_stage_t0); \
        } else if (ok && stage_sync) { \
            ok = glm_graph_prefill_stage_sync_boundary(); \
        } \
    } while (0)
    ds4_gpu_tensor *last_indexer_selected = NULL;
    uint32_t last_indexer_selected_count = 0;
    if (ok && tp_attn_head_split) {
        /* The unowned head range of batch_heads must be exactly zero so the
         * full-width attn-output matmul produces partial sums. Owned heads
         * are rewritten every layer, so one fill per chunk suffices. */
        ok = ds4_gpu_tensor_fill_f32(g->batch_heads, 0.0f,
                                     (uint64_t)n_tokens * g->heads_dim) != 0;
    }
    ds4_gpu_tp_set_attn_head_split(tp_attn_head_split ? 1 : 0);
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        const uint32_t slice_layer_done = il - g->layer_start + 1u;
        if (g->ssd_streaming) {
            ok = glm_graph_stream_map_prefill_layer(g,
                                                    model,
                                                    weights,
                                                    il,
                                                    n_tokens,
                                                    false);
            if (ok) ok = ds4_gpu_begin_commands() != 0;
        }
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t kv_raw_dim = (uint32_t)l->attn_kv_a_mqa->dim[1];
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
        const bool layer_stage_profile = metal_graph_layer_stage_profile_enabled(il);
        double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
        double trace_stage_t0 = trace ? now_sec() : 0.0;
        const double trace_layer_t0 = trace_stage_t0;
        const bool trace_full_indexer = glm_graph_layer_uses_full_indexer(il);
        bool trace_layer_flushed = false;
        if (trace && (trace_all || trace_full_indexer)) {
            glm_graph_indexed_prefill_tracef(
                    "layer begin layer=%u pos=%u tokens=%u rows=%u selected=%u full_indexer=%u",
                    il,
                    pos0,
                    n_tokens,
                    n_rows,
                    indexed_selected_count,
                    trace_full_indexer ? 1u : 0u);
        }
        if (residual_elems > UINT32_MAX) {
            ok = false;
            break;
        }
        if (layer_stage_profile) {
            ok = metal_graph_layer_stage_profile_boundary("glm_indexed_attn",
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
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "attn_norm");
        if (ok) {
            if (n_tokens <= 8u && (glm_decode_ablate_mask() & DS4_GLM_ABLATE_QPATH)) { /* ablate */ } else
            ok = (use_batch_q_rank_proj ?
                  glm_graph_matmul_q8_0_tensor(g->batch_q_rank,
                                               model,
                                               l->attn_q_a->abs_offset,
                                               DS4_N_EMBD,
                                               DS4_N_LORA_Q,
                                               g->batch_attn_norm,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(g->batch_q_rank,
                                                    model,
                                                    l->attn_q_a->abs_offset,
                                                    DS4_N_EMBD,
                                                    DS4_N_LORA_Q,
                                                    g->batch_attn_norm,
                                                    n_tokens));
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_q_rank_norm,
                                                         g->batch_q_rank,
                                                         model->map,
                                                         model->size,
                                                         l->attn_q_a_norm->abs_offset,
                                                         DS4_N_LORA_Q,
                                                         n_tokens,
                                                         DS4_RMS_EPS) != 0;
        if (ok) {
            ok = (use_batch_q_proj ?
                  glm_graph_matmul_q8_0_tensor(g->batch_q,
                                               model,
                                               l->attn_q_b->abs_offset,
                                               DS4_N_LORA_Q,
                                               g->q_dim,
                                               g->batch_q_rank_norm,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(g->batch_q,
                                                    model,
                                                    l->attn_q_b->abs_offset,
                                                    DS4_N_LORA_Q,
                                                    g->q_dim,
                                                    g->batch_q_rank_norm,
                                                    n_tokens));
        }
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
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "q_path");

        if (ok && glm_graph_layer_uses_full_indexer(il)) {
            ok = (use_batch_indexer_k_proj ?
                  glm_graph_matmul_q8_0_tensor(g->batch_indexer_k,
                                               model,
                                               l->indexer_attn_k->abs_offset,
                                               DS4_N_EMBD,
                                               DS4_N_INDEXER_HEAD_DIM,
                                               cur,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(g->batch_indexer_k,
                                                    model,
                                                    l->indexer_attn_k->abs_offset,
                                                    DS4_N_EMBD,
                                                    DS4_N_INDEXER_HEAD_DIM,
                                                    cur,
                                                    n_tokens));
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
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_k");

        if (ok) {
            ok = (use_batch_kv_proj ?
                  glm_graph_matmul_q8_0_tensor(g->batch_kv_raw,
                                               model,
                                               l->attn_kv_a_mqa->abs_offset,
                                               DS4_N_EMBD,
                                               kv_raw_dim,
                                               g->batch_attn_norm,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(g->batch_kv_raw,
                                                    model,
                                                    l->attn_kv_a_mqa->abs_offset,
                                                    DS4_N_EMBD,
                                                    kv_raw_dim,
                                                    g->batch_attn_norm,
                                                    n_tokens));
        }
        if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(g->batch_kv_norm,
                                                         g->batch_kv_raw,
                                                         model->map,
                                                         model->size,
                                                         l->attn_kv_a_norm->abs_offset,
                                                         n_tokens,
                                                         kv_raw_dim,
                                                         DS4_N_KV_LORA,
                                                         DS4_RMS_EPS) != 0;
        if (ok) {
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
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "kv_path");

        if (ok && glm_graph_layer_uses_full_indexer(il)) {
            if (ok && !use_causal_range_select) {
                ok = (use_batch_indexer_q_proj ?
                      glm_graph_matmul_q8_0_tensor(g->batch_indexer_q,
                                                   model,
                                                   l->indexer_attn_q_b->abs_offset,
                                                   DS4_N_LORA_Q,
                                                   (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                   g->batch_q_rank_norm,
                                                   n_tokens) :
                      glm_graph_matmul_q8_0_rows_scalar(g->batch_indexer_q,
                                                        model,
                                                        l->indexer_attn_q_b->abs_offset,
                                                        DS4_N_LORA_Q,
                                                        (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                        g->batch_q_rank_norm,
                                                        n_tokens));
                if (ok) ok = ds4_gpu_glm_indexer_rope_tail_tensor(g->batch_indexer_q,
                                                                  n_tokens,
                                                                  DS4_N_INDEXER_HEAD,
                                                                  DS4_N_INDEXER_HEAD_DIM,
                                                                  DS4_N_ROT,
                                                                  pos0,
                                                                  0,
                                                                  rope_base,
                                                                  rope_scale,
                                                                  0.0f,
                                                                  1.0f,
                                                                  DS4_ROPE_YARN_BETA_FAST,
                                                                  DS4_ROPE_YARN_BETA_SLOW) != 0;
                if (ok && glm_graph_indexer_qat()) {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(g->batch_indexer_q,
                                                         n_tokens * DS4_N_INDEXER_HEAD,
                                                         DS4_N_INDEXER_HEAD_DIM) != 0;
                }
                if (ok) {
                    ok = (use_batch_indexer_weights_proj ?
                          ds4_gpu_matmul_f32_tensor(g->batch_indexer_weights,
                                                    model->map,
                                                    model->size,
                                                    l->indexer_proj->abs_offset,
                                                    DS4_N_EMBD,
                                                    DS4_N_INDEXER_HEAD,
                                                    cur,
                                                    n_tokens) != 0 :
                          glm_graph_matmul_f32_rows_scalar(g->batch_indexer_weights,
                                                           model,
                                                           l->indexer_proj->abs_offset,
                                                           DS4_N_EMBD,
                                                           DS4_N_INDEXER_HEAD,
                                                           cur,
                                                           n_tokens));
                }
                DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_q_weights");
            }
            if (ok) {
                if (use_causal_range_select) {
                    ok = ds4_gpu_glm_fill_selected_range_batch_tensor(
                                 g->batch_indexer_selected,
                                 n_tokens,
                                 pos0,
                                 indexed_selected_count,
                                 g->compact_cache_cap) != 0;
                    DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_range");
                } else if (use_scalar_indexer) {
                    const float indexer_scale =
                        1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    for (uint32_t t = 0; ok && t < n_tokens; t++) {
                        const uint32_t visible = pos0 + t + 1u;
                        ds4_gpu_tensor *indexer_q_view =
                            glm_graph_tensor_row_view_strided(
                                    g->batch_indexer_q,
                                    t,
                                    (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                    (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM);
                        ds4_gpu_tensor *indexer_weights_view =
                            glm_graph_tensor_row_view_strided(g->batch_indexer_weights,
                                                              t,
                                                              DS4_N_INDEXER_HEAD,
                                                              DS4_N_INDEXER_HEAD);
                        ds4_gpu_tensor *scores_view =
                            ds4_gpu_tensor_view(g->batch_indexer_scores,
                                                0,
                                                (uint64_t)visible * sizeof(float));
                        ds4_gpu_tensor *selected_view =
                            ds4_gpu_tensor_view(g->batch_indexer_selected,
                                                (uint64_t)t * indexed_selected_count * sizeof(uint32_t),
                                                (uint64_t)indexed_selected_count * sizeof(uint32_t));
                        ok = indexer_q_view && indexer_weights_view && scores_view && selected_view;
                        if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill failed to create indexer row views at layer %u token %u\n", il, t);
                        if (ok) {
                            int rc = ds4_gpu_glm_indexer_score_one_tensor(
                                             scores_view,
                                             indexer_q_view,
                                             indexer_weights_view,
                                             g->layer_indexer_key_cache[il],
                                             visible,
                                             DS4_N_INDEXER_HEAD,
                                             DS4_N_INDEXER_HEAD_DIM,
                                             indexer_scale,
                                             glm_graph_compact_cache_is_f16());
                            ok = rc != 0;
                            if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill indexer scores failed at layer %u token %u\n", il, t);
                        }
                        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_score_scalar");
                        if (ok) {
                            int rc = ds4_gpu_indexer_topk_tensor(selected_view,
                                                                 scores_view,
                                                                 visible,
                                                                 1,
                                                                 indexed_selected_count);
                            ok = rc != 0;
                            if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill topk failed at layer %u token %u\n", il, t);
                        }
                        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_topk_scalar");
                        ds4_gpu_tensor_free(selected_view);
                        ds4_gpu_tensor_free(scores_view);
                        ds4_gpu_tensor_free(indexer_weights_view);
                        ds4_gpu_tensor_free(indexer_q_view);
                    }
                } else {
                    const float indexer_scale =
                        1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    const uint32_t score_cap =
                        g->indexed_prefill_score_cap != 0 ?
                            g->indexed_prefill_score_cap :
                            g->indexed_prefill_cap;
                    const uint64_t indexer_q_row_bytes =
                        (uint64_t)DS4_N_INDEXER_HEAD *
                        DS4_N_INDEXER_HEAD_DIM *
                        sizeof(float);
                    const uint64_t indexer_weights_row_bytes =
                        (uint64_t)DS4_N_INDEXER_HEAD * sizeof(float);
                    const uint64_t selected_row_bytes =
                        (uint64_t)indexed_selected_count * sizeof(uint32_t);
                    for (uint32_t t0 = 0; ok && t0 < n_tokens; ) {
                        uint32_t slice = n_tokens - t0;
                        if (slice > score_cap) slice = score_cap;
                        if (slice == 0) {
                            ok = false;
                            break;
                        }

                        ds4_gpu_tensor *indexer_q_view =
                            ds4_gpu_tensor_view(g->batch_indexer_q,
                                                (uint64_t)t0 * indexer_q_row_bytes,
                                                (uint64_t)slice * indexer_q_row_bytes);
                        ds4_gpu_tensor *indexer_weights_view =
                            ds4_gpu_tensor_view(g->batch_indexer_weights,
                                                (uint64_t)t0 * indexer_weights_row_bytes,
                                                (uint64_t)slice * indexer_weights_row_bytes);
                        ds4_gpu_tensor *selected_view =
                            ds4_gpu_tensor_view(g->batch_indexer_selected,
                                                (uint64_t)t0 * selected_row_bytes,
                                                (uint64_t)slice * selected_row_bytes);
                        ok = indexer_q_view && indexer_weights_view && selected_view;
                        if (!ok) {
                            fprintf(stderr,
                                    "ds4: GLM indexed prefill failed to create indexer score slice views at layer %u token %u\n",
                                    il,
                                    t0);
                        }
                        if (ok) {
                            ok = ds4_gpu_glm_indexer_scores_batch_tensor(
                                         g->batch_indexer_scores,
                                         indexer_q_view,
                                         indexer_weights_view,
                                         g->layer_indexer_key_cache[il],
                                         n_rows,
                                         slice,
                                         pos0 + t0,
                                         DS4_N_INDEXER_HEAD,
                                         DS4_N_INDEXER_HEAD_DIM,
                                         indexer_scale,
                                         glm_graph_compact_cache_is_f16()) != 0;
                        }
                        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_score");
                        if (ok) {
                            ok = ds4_gpu_indexer_topk_tensor(selected_view,
                                                             g->batch_indexer_scores,
                                                             n_rows,
                                                             slice,
                                                             indexed_selected_count) != 0;
                        }
                        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_topk");
                        ds4_gpu_tensor_free(selected_view);
                        ds4_gpu_tensor_free(indexer_weights_view);
                        ds4_gpu_tensor_free(indexer_q_view);
                        t0 += slice;
                    }
                }
            }
            if (ok) {
                last_indexer_selected = g->batch_indexer_selected;
                last_indexer_selected_count = indexed_selected_count;
            }
        } else if (ok && (!last_indexer_selected || last_indexer_selected_count == 0)) {
            ok = false;
        }
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_select");
        metal_graph_debug_dump_tensor("glm_indexed_q",
                                      g->batch_q,
                                      (uint64_t)n_tokens * DS4_N_HEAD * DS4_N_KEY_MLA,
                                      il,
                                      pos0);
        if (use_batch_qk_low) {
            if (n_tokens <= 8u && (glm_decode_ablate_mask() & DS4_GLM_ABLATE_ATTN_CORE)) { /* ablate */ } else
            if (ok) ok = ds4_gpu_glm_qk_lowrank_typed_batch_tensor(g->batch_qk_low,
                                                                   g->batch_q,
                                                                   model->map,
                                                                   model->size,
                                                                   l->attn_k_b->abs_offset,
                                                                   l->attn_k_b->type,
                                                                   n_tokens,
                                                                   DS4_N_HEAD,
                                                                   DS4_N_KV_LORA,
                                                                   (uint32_t)g->q_nope,
                                                                   DS4_N_KEY_MLA) != 0;
        } else {
            for (uint32_t t = 0; ok && t < n_tokens; t++) {
                ds4_gpu_tensor *q_view =
                    glm_graph_tensor_row_view_strided(g->batch_q,
                                                      t,
                                                      g->q_dim,
                                                      g->q_dim);
                ds4_gpu_tensor *qk_low_view =
                    glm_graph_tensor_row_view_strided(
                            g->batch_qk_low,
                            t,
                            (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA,
                            (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA);
                ok = q_view && qk_low_view;
                if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill failed to create qk-low row views at layer %u token %u\n", il, t);
                if (ok) {
                    int rc = ds4_gpu_glm_qk_lowrank_typed_tensor(qk_low_view,
                                                                 q_view,
                                                                 model->map,
                                                                 model->size,
                                                                 l->attn_k_b->abs_offset,
                                                                 l->attn_k_b->type,
                                                                 DS4_N_HEAD,
                                                                 DS4_N_KV_LORA,
                                                                 (uint32_t)g->q_nope,
                                                                 DS4_N_KEY_MLA);
                    ok = rc != 0;
                    if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill qk-low failed at layer %u token %u\n", il, t);
                }
                ds4_gpu_tensor_free(qk_low_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "qk_low");
        metal_graph_debug_dump_tensor("glm_indexed_qk_low",
                                      g->batch_qk_low,
                                      (uint64_t)n_tokens * DS4_N_HEAD * DS4_N_KV_LORA,
                                      il,
                                      pos0);
        if (ok && use_batch_attn_kernel) ok = glm_graph_indexed_prefill_attention_boundary();

        if (use_batch_attn_kernel) {
            const uint32_t attn_slice_cap =
                glm_graph_indexed_prefill_batch_attn_slice_tokens();
            for (uint32_t t0 = 0; ok && t0 < n_tokens; ) {
                uint32_t slice = n_tokens - t0;
                if (slice > attn_slice_cap) slice = attn_slice_cap;

                ds4_gpu_tensor *q_view =
                    ds4_gpu_tensor_view(g->batch_q,
                                        (uint64_t)t0 * g->q_dim * sizeof(float),
                                        (uint64_t)slice * g->q_dim * sizeof(float));
                ds4_gpu_tensor *qk_low_view =
                    ds4_gpu_tensor_view(g->batch_qk_low,
                                        (uint64_t)t0 * DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float),
                                        (uint64_t)slice * DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float));
                ds4_gpu_tensor *heads_view =
                    ds4_gpu_tensor_view(g->batch_heads,
                                        (uint64_t)t0 * g->heads_dim * sizeof(float),
                                        (uint64_t)slice * g->heads_dim * sizeof(float));
                ds4_gpu_tensor *attn_lora_view = use_split_value_proj ?
                    ds4_gpu_tensor_view(g->batch_attn_lora,
                                        (uint64_t)t0 * DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float),
                                        (uint64_t)slice * DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float)) :
                    NULL;
                ds4_gpu_tensor *selected_view =
                    ds4_gpu_tensor_view(last_indexer_selected,
                                        (uint64_t)t0 * last_indexer_selected_count * sizeof(uint32_t),
                                        (uint64_t)slice * last_indexer_selected_count * sizeof(uint32_t));
                ok = q_view && qk_low_view && heads_view && selected_view &&
                     (!use_split_value_proj || attn_lora_view);
                if (!ok) {
                    fprintf(stderr, "ds4: GLM sliced indexed prefill failed to create attention views at layer %u token %u\n", il, t0);
                }
                if (n_tokens <= 8u && (glm_decode_ablate_mask() & DS4_GLM_ABLATE_ATTN_CORE)) { /* ablate */ } else if (ok && use_split_value_proj) {
                    int rc = 0;
                    if (use_causal_range_select) {
                        rc = ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
                                attn_lora_view,
                                q_view,
                                qk_low_view,
                                g->layer_kv_lora_cache[il],
                                g->layer_k_rope_cache[il],
                                slice,
                                pos0 + t0,
                                last_indexer_selected_count,
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
                                DS4_ROPE_YARN_BETA_SLOW);
                    } else {
                        rc = ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
                                attn_lora_view,
                                q_view,
                                qk_low_view,
                                g->layer_kv_lora_cache[il],
                                g->layer_k_rope_cache[il],
                                selected_view,
                                slice,
                                last_indexer_selected_count,
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
                                DS4_ROPE_YARN_BETA_SLOW);
                    }
                    ok = rc != 0;
                    if (!ok) fprintf(stderr, "ds4: GLM sliced indexed prefill attention-lora failed at layer %u token %u\n", il, t0);
                    if (ok && layer_stage_profile) {
                        ok = metal_graph_layer_stage_profile_boundary("glm_indexed_attn",
                                                                      "attention_lora",
                                                                      il,
                                                                      pos0 + t0,
                                                                      slice,
                                                                      &layer_stage_t0);
                    }
                    if (ok) {
                        rc = ds4_gpu_glm_value_project_typed_batch_heads_tensor(
                                    heads_view,
                                    attn_lora_view,
                                    model->map,
                                    model->size,
                                    l->attn_v_b->abs_offset,
                                    l->attn_v_b->type,
                                    slice,
                                    DS4_N_HEAD,
                                    DS4_N_KV_LORA,
                                    DS4_N_VALUE_MLA);
                        ok = rc != 0;
                        if (!ok) fprintf(stderr, "ds4: GLM sliced indexed prefill value project failed at layer %u token %u\n", il, t0);
                    }
                    if (ok && layer_stage_profile) {
                        ok = metal_graph_layer_stage_profile_boundary("glm_indexed_attn",
                                                                      "value_project",
                                                                      il,
                                                                      pos0 + t0,
                                                                      slice,
                                                                      &layer_stage_t0);
                    }
                } else if (ok) {
                    int rc = ds4_gpu_glm_attention_indexed_batch_typed_tensor(heads_view,
                                                                              q_view,
                                                                              qk_low_view,
                                                                              g->layer_kv_lora_cache[il],
                                                                              g->layer_k_rope_cache[il],
                                                                              model->map,
                                                                              model->size,
                                                                              l->attn_v_b->abs_offset,
                                                                              l->attn_v_b->type,
                                                                              selected_view,
                                                                              slice,
                                                                              last_indexer_selected_count,
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
                                                                              DS4_ROPE_YARN_BETA_SLOW);
                    ok = rc != 0;
                    if (!ok) fprintf(stderr, "ds4: GLM sliced indexed prefill indexed attention failed at layer %u token %u\n", il, t0);
                    if (ok && layer_stage_profile) {
                        ok = metal_graph_layer_stage_profile_boundary("glm_indexed_attn",
                                                                      "attention_fused",
                                                                      il,
                                                                      pos0 + t0,
                                                                      slice,
                                                                      &layer_stage_t0);
                    }
                }
                ds4_gpu_tensor_free(selected_view);
                ds4_gpu_tensor_free(attn_lora_view);
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(qk_low_view);
                ds4_gpu_tensor_free(q_view);
                t0 += slice;
            }
            if (ok) ok = glm_graph_indexed_prefill_attention_boundary();
        } else {
            for (uint32_t t = 0; ok && t < n_tokens; t++) {
                ds4_gpu_tensor *q_view =
                    glm_graph_tensor_row_view_strided(g->batch_q,
                                                      t,
                                                      g->q_dim,
                                                      g->q_dim);
                ds4_gpu_tensor *qk_low_view =
                    glm_graph_tensor_row_view_strided(
                            g->batch_qk_low,
                            t,
                            (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA,
                            (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA);
                ds4_gpu_tensor *heads_view =
                    glm_graph_tensor_row_view_strided(g->batch_heads,
                                                      t,
                                                      g->heads_dim,
                                                      g->heads_dim);
                ds4_gpu_tensor *selected_view =
                    ds4_gpu_tensor_view(last_indexer_selected,
                                        (uint64_t)t * last_indexer_selected_count * sizeof(uint32_t),
                                        (uint64_t)last_indexer_selected_count * sizeof(uint32_t));
                ok = q_view && qk_low_view && heads_view && selected_view;
                if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill failed to create attention row views at layer %u token %u\n", il, t);
                if (ok) {
                    int rc = ds4_gpu_glm_attention_indexed_decode_typed_tensor(heads_view,
                                                                               q_view,
                                                                               qk_low_view,
                                                                               g->layer_kv_lora_cache[il],
                                                                               g->layer_k_rope_cache[il],
                                                                               model->map,
                                                                               model->size,
                                                                               l->attn_v_b->abs_offset,
                                                                               l->attn_v_b->type,
                                                                               selected_view,
                                                                               last_indexer_selected_count,
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
                                                                               DS4_ROPE_YARN_BETA_SLOW);
                    ok = rc != 0;
                    if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill indexed attention failed at layer %u token %u\n", il, t);
                }
                ds4_gpu_tensor_free(selected_view);
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(qk_low_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "attention");
        metal_graph_debug_dump_tensor("glm_indexed_heads",
                                      g->batch_heads,
                                      (uint64_t)n_tokens * g->heads_dim,
                                      il,
                                      pos0);
        if (ok && tp_attn_head_split) {
            ok = glm_graph_tp_batch_bounce_ready(g, n_tokens);
        }
        if (ok) {
            /* Under the head split the projection input has zeros in the
             * unowned head columns, so the result is this rank's partial;
             * it must land in the shared bounce tensor for the exchange. */
            ds4_gpu_tensor *attn_out_dst =
                tp_attn_head_split ? g->tp_bounce_out : g->batch_attn_out;
            if (n_tokens <= 8u && (glm_decode_ablate_mask() & DS4_GLM_ABLATE_ATTN_OUT)) { /* ablate */ } else
            ok = (use_batch_attn_out_proj ?
                  glm_graph_matmul_q8_0_tensor(attn_out_dst,
                                               model,
                                               l->attn_output->abs_offset,
                                               g->heads_dim,
                                               DS4_N_EMBD,
                                               g->batch_heads,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(attn_out_dst,
                                                    model,
                                                    l->attn_output->abs_offset,
                                                    g->heads_dim,
                                                    DS4_N_EMBD,
                                                    g->batch_heads,
                                                    n_tokens));
        }
        if (ok && tp_attn_head_split) {
            ok = glm_graph_tp_batch_ffn_combine(g, il, g->batch_attn_out, n_tokens);
        }
        if (ok) ok = ds4_gpu_add_tensor(g->batch_after_attn,
                                        cur,
                                        g->batch_attn_out,
                                        (uint32_t)residual_elems) != 0;
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "attn_output");
        metal_graph_debug_dump_tensor("glm_indexed_after_attn",
                                      g->batch_after_attn,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
        metal_graph_debug_dump_tensor("glm_indexed_attn_out",
                                      g->batch_attn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
        if (ok && use_batch_ffn) {
            ok = glm_graph_encode_ffn_batch(g,
                                            model,
                                            weights,
                                            l,
                                            il,
                                            pos0,
                                            g->batch_after_attn,
                                            next,
                                            n_tokens,
                                            false,
                                            layer_stage_profile,
                                            stage_sync,
                                            layer_stage_profile ? &layer_stage_t0 : NULL);
        } else if (ok) {
            const bool use_batch_ffn_norm =
                n_tokens > 1 && glm_graph_indexed_prefill_batch_ffn_norm();
            if (use_batch_ffn_norm) {
                ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_ffn_norm,
                                                          g->batch_after_attn,
                                                          model->map,
                                                          model->size,
                                                          l->ffn_norm->abs_offset,
                                                          DS4_N_EMBD,
                                                          n_tokens,
                                                          DS4_RMS_EPS) != 0;
            }
            if (use_batch_ffn_norm) {
                DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_ffn", "ffn_norm");
            }
            if (ok &&
                use_batch_ffn_norm &&
                il >= DS4_N_LEADING_DENSE &&
                glm_graph_indexed_prefill_batch_routed_moe()) {
                ok = glm_graph_encode_sparse_ffn_indexed_batch_routed_moe(
                        g,
                        model,
                        l,
                        il,
                        pos0,
                        g->batch_after_attn,
                        next,
                        n_tokens,
                        layer_stage_profile,
                        stage_sync,
                        layer_stage_profile ? &layer_stage_t0 : NULL);
            } else for (uint32_t t = 0; ok && t < n_tokens; t++) {
                ds4_gpu_tensor *after_attn_view =
                    glm_graph_tensor_row_view_strided(g->batch_after_attn,
                                                      t,
                                                      DS4_N_EMBD,
                                                      DS4_N_EMBD);
                ds4_gpu_tensor *ffn_norm_view = use_batch_ffn_norm ?
                    glm_graph_tensor_row_view_strided(g->batch_ffn_norm,
                                                      t,
                                                      DS4_N_EMBD,
                                                      DS4_N_EMBD) :
                    NULL;
                ds4_gpu_tensor *next_view =
                    glm_graph_tensor_row_view_strided(next,
                                                      t,
                                                      DS4_N_EMBD,
                                                      DS4_N_EMBD);
                ok = after_attn_view && next_view &&
                     (!use_batch_ffn_norm || ffn_norm_view);
                if (ok && use_batch_ffn_norm) {
                    ok = glm_graph_encode_ffn_one_normed_from(g,
                                                              model,
                                                              l,
                                                              il,
                                                              pos0 + t,
                                                              ffn_norm_view,
                                                              after_attn_view,
                                                              next_view,
                                                              g->ffn_gate,
                                                              g->ffn_up,
                                                              g->ffn_mid,
                                                              g->ffn_out,
                                                              g->ffn_sum,
                                                              g->attn_out,
                                                              false,
                                                              NULL);
                } else if (ok) {
                    ok = glm_graph_encode_ffn_one_from(g,
                                                       model,
                                                       l,
                                                       il,
                                                       pos0 + t,
                                                       after_attn_view,
                                                       next_view,
                                                       g->ffn_norm,
                                                       g->ffn_gate,
                                                       g->ffn_up,
                                                       g->ffn_mid,
                                                       g->ffn_out,
                                                       g->ffn_sum,
                                                       g->attn_out,
                                                       false,
                                                       NULL);
                }
                ds4_gpu_tensor_free(next_view);
                ds4_gpu_tensor_free(ffn_norm_view);
                ds4_gpu_tensor_free(after_attn_view);
            }
        }
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
            if (trace && (trace_all || trace_full_indexer || drain_now)) {
                glm_graph_indexed_prefill_tracef(
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
                if (trace_all || trace_full_indexer || drain_now ||
                    command_ms >= trace_slow_ms || layer_ms >= trace_slow_ms || !ok) {
                    glm_graph_indexed_prefill_tracef(
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
                if (ok) ok = ds4_gpu_end_commands() != 0;
                else (void)ds4_gpu_synchronize();
            } else if (!ok) {
                (void)ds4_gpu_synchronize();
            }
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
        if (trace && ok) {
            const double layer_ms = (now_sec() - trace_layer_t0) * 1000.0;
            if (trace_all || (!trace_layer_flushed && layer_ms >= trace_slow_ms)) {
                glm_graph_indexed_prefill_tracef(
                        "layer end layer=%u pos=%u tokens=%u flushed=%u layer_total=%.3f ms",
                        il,
                        pos0,
                        n_tokens,
                        trace_layer_flushed ? 1u : 0u,
                        layer_ms);
            }
        }
    }
    ds4_gpu_tp_set_attn_head_split(0);
#undef DS4_GLM_PROFILE_INDEXED_STAGE
    if (ok && !g->ssd_streaming) {
        const double trace_end_t0 = trace ? now_sec() : 0.0;
        if (trace) {
            glm_graph_indexed_prefill_tracef(
                    "chunk end_commands begin pos=%u tokens=%u",
                    pos0,
                    n_tokens);
        }
        ok = ds4_gpu_end_commands() != 0;
        if (trace) {
            const double end_ms = (now_sec() - trace_end_t0) * 1000.0;
            const double chunk_ms = (now_sec() - trace_chunk_t0) * 1000.0;
            glm_graph_indexed_prefill_tracef(
                    "chunk end_commands %s pos=%u tokens=%u end=%.3f ms chunk_total=%.3f ms",
                    ok ? "done" : "failed",
                    pos0,
                    n_tokens,
                    end_ms,
                    chunk_ms);
        }
    } else if (!ok) {
        if (trace) {
            glm_graph_indexed_prefill_tracef(
                    "chunk failed before end pos=%u tokens=%u elapsed=%.3f ms",
                    pos0,
                    n_tokens,
                    (now_sec() - trace_chunk_t0) * 1000.0);
        }
        (void)ds4_gpu_synchronize();
    }
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

static bool glm_graph_use_streaming_token_prefill(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens);
static uint32_t glm_graph_streaming_token_prefill_max_tokens(void);
static bool glm_graph_prefill_token_major(
        ds4_glm_gpu_graph      *g,
        const ds4_model        *model,
        const ds4_weights      *weights,
        const int              *tokens,
        uint32_t                pos0,
        uint32_t                n_tokens,
        float                  *logits_out,
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                display_absolute_base,
        uint32_t                work_done_base,
        uint32_t                work_total);

static bool glm_graph_prefill_range(
        ds4_glm_gpu_graph      *g,
        const ds4_model        *model,
        const ds4_weights      *weights,
        const int              *tokens,
        uint32_t                pos0,
        uint32_t                n_tokens,
        float                  *logits_out,
        ds4_session_progress_fn progress,
        void                   *progress_ud,
        uint32_t                progress_total) {
    if (n_tokens == 0) return true;
    if (!glm_graph_span_fits_context(g, pos0, n_tokens)) return false;
    const uint32_t chunk_max = glm_graph_prefill_chunk_tokens(g->ctx_cap);
    uint32_t done = 0;
    while (done < n_tokens) {
        const uint32_t pos = pos0 + done;
        if (!g->full_kv_cache) {
            const uint32_t remaining = n_tokens - done;
            uint32_t chunk = 1;
            if (glm_graph_use_streaming_token_prefill(g, pos, remaining)) {
                chunk = remaining;
                const uint32_t token_prefill_max =
                    glm_graph_streaming_token_prefill_max_tokens();
                if (token_prefill_max != 0 && chunk > token_prefill_max) {
                    chunk = token_prefill_max;
                }
                float *dst_logits = (done + chunk == n_tokens) ? logits_out : NULL;
                if (!glm_graph_prefill_token_major(g,
                                                   model,
                                                   weights,
                                                   tokens + done,
                                                   pos,
                                                   chunk,
                                                   dst_logits,
                                                   progress,
                                                   progress_ud,
                                                   pos0,
                                                   done,
                                                   n_tokens)) {
                    return false;
                }
            } else if (glm_graph_indexed_prefill_batch_ready(g, pos)) {
                chunk = remaining;
                if (chunk > g->indexed_prefill_cap) chunk = g->indexed_prefill_cap;
                chunk = glm_graph_limit_indexed_prefill_chunk(pos, chunk);
                if (chunk == 0) chunk = 1;
                float *dst_logits = (done + chunk == n_tokens) ? logits_out : NULL;
                if (!glm_graph_forward_indexed_tokens(g,
                                                      model,
                                                      weights,
                                                      tokens + done,
                                                      NULL,
                                                      pos,
                                                      chunk,
                                                      NULL,
                                                      dst_logits,
                                                      progress,
                                                      progress_ud,
                                                      pos0,
                                                      done,
                                                      n_tokens)) {
                    return false;
                }
            } else {
                float *dst_logits = (done + 1u == n_tokens) ? logits_out : NULL;
                if (!glm_graph_forward_token(g,
                                             model,
                                             weights,
                                             tokens[done],
                                             NULL,
                                             pos,
                                             NULL,
                                             dst_logits,
                                             false)) {
                    return false;
                }
            }
            done += chunk;
            if (progress) {
                const uint32_t current = pos0 + done;
                progress(progress_ud,
                         "prefill_chunk",
                         current,
                         progress_total ? progress_total : pos0 + n_tokens);
            }
            continue;
        }
        if (pos >= g->ctx_cap) {
            if (g->compact_cache_cap == 0) {
                glm_graph_log_full_attention_limit(g, pos, n_tokens - done);
                return false;
            }
            while (done < n_tokens) {
                const uint32_t cur_pos = pos0 + done;
                const bool use_indexed_batch =
                    glm_graph_indexed_prefill_batch_ready(g, cur_pos);
                if (use_indexed_batch) {
                    uint32_t chunk = n_tokens - done;
                    if (chunk > g->indexed_prefill_cap) chunk = g->indexed_prefill_cap;
                    chunk = glm_graph_limit_indexed_prefill_chunk(cur_pos, chunk);
                    float *dst_logits = (done + chunk == n_tokens) ? logits_out : NULL;
                    if (!glm_graph_forward_indexed_tokens(g,
                                                          model,
                                                          weights,
                                                          tokens + done,
                                                          NULL,
                                                          cur_pos,
                                                          chunk,
                                                          NULL,
                                                          dst_logits,
                                                          progress,
                                                          progress_ud,
                                                          pos0,
                                                          done,
                                                          n_tokens)) {
                        return false;
                    }
                    done += chunk;
                } else {
                    float *dst_logits = (done + 1u == n_tokens) ? logits_out : NULL;
                    if (!glm_graph_forward_token(g,
                                                 model,
                                                 weights,
                                                 tokens[done],
                                                 NULL,
                                                 cur_pos,
                                                 NULL,
                                                 dst_logits,
                                                 false)) {
                        return false;
                    }
                    done++;
                }
                if (progress) {
                    const uint32_t current = pos0 + done;
                    progress(progress_ud,
                             "prefill_chunk",
                             current,
                             progress_total ? progress_total : pos0 + n_tokens);
                }
            }
            return true;
        }
        uint32_t chunk = n_tokens - done;
        const uint32_t full_remaining = g->ctx_cap - pos;
        if (chunk > full_remaining) chunk = full_remaining;
        if (chunk > chunk_max) chunk = chunk_max;
        float *dst_logits = (done + chunk == n_tokens) ? logits_out : NULL;
        if (glm_graph_use_streaming_token_prefill(g, pos, chunk)) {
            if (!glm_graph_prefill_token_major(g,
                                               model,
                                               weights,
                                               tokens + done,
                                               pos,
                                               chunk,
                                               dst_logits,
                                               progress,
                                               progress_ud,
                                               pos0,
                                               done,
                                               n_tokens)) {
                return false;
            }
        } else if (!glm_graph_forward_tokens(g,
                                             model,
                                             weights,
                                             tokens + done,
                                             NULL,
                                             pos,
                                             chunk,
                                             NULL,
                                             dst_logits,
                                             progress,
                                             progress_ud,
                                             pos0,
                                             done,
                                             n_tokens)) {
            return false;
        }
        done += chunk;
        if (progress) {
            const uint32_t current = pos0 + done;
            progress(progress_ud,
                     "prefill_chunk",
                     current,
                     progress_total ? progress_total : pos0 + n_tokens);
        }
    }
    return true;
}

#endif /* !DS4_NO_GPU */
