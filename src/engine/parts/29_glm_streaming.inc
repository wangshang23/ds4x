#ifndef DS4_NO_GPU
/*
 * For very short GLM SSD-streaming prefills, Metal still benefits from the
 * token-major path because it reuses the normal decode graph and warms the
 * decode expert cache. On ROCm/Strix Halo the indexed batch prefill is faster
 * now that streamed batch routing and expert cache seeding are implemented, so
 * ROCm defaults to canonical batch prefill unless the env override below opts
 * token-major prefill back in.
 */
enum { DS4_GLM_STREAM_PREFILL_TOKEN_MAJOR_MAX_TOKENS = 64 };

static uint32_t glm_graph_streaming_token_prefill_default_max_tokens(void) {
#ifdef DS4_ROCM_BUILD
    return 0;
#else
    return DS4_GLM_STREAM_PREFILL_TOKEN_MAJOR_MAX_TOKENS;
#endif
}

static uint32_t glm_graph_streaming_token_prefill_max_tokens(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_GLM_STREAMING_TOKEN_PREFILL_MAX",
            "DS4_METAL_GLM_STREAMING_TOKEN_PREFILL_MAX");
    if (!env || !env[0]) env = getenv("DS4_GLM_STREAMING_TOKEN_PREFILL_MAX");
    const uint32_t default_max =
        glm_graph_streaming_token_prefill_default_max_tokens();
    if (!env || !env[0]) return default_max;
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || errno != 0 || v > UINT32_MAX) {
        return default_max;
    }
    return (uint32_t)v;
}

static bool glm_graph_use_streaming_token_prefill(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    if (!g || !g->ssd_streaming || g->quality || n_tokens == 0) return false;
    if (getenv("DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL") != NULL ||
        glm_graph_env_present("DS4_ROCM_GLM_DISABLE_STREAMING_TOKEN_PREFILL",
                              "DS4_METAL_GLM_DISABLE_STREAMING_TOKEN_PREFILL")) {
        return false;
    }
    if (!glm_graph_span_fits_full_attention(g, pos0, n_tokens)) return false;
    const uint32_t max_tokens = glm_graph_streaming_token_prefill_max_tokens();
    return max_tokens != 0 && n_tokens <= max_tokens;
}

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
        uint32_t                work_total) {
    if (!g || !model || !weights || !tokens || n_tokens == 0) return false;
    for (uint32_t i = 0; i < n_tokens; i++) {
        const bool last = i + 1u == n_tokens;
        float *dst_logits = (last && logits_out) ? logits_out : NULL;
        if (!glm_graph_forward_token(g,
                                     model,
                                     weights,
                                     tokens[i],
                                     NULL,
                                     pos0 + i,
                                     NULL,
                                     dst_logits,
                                     false)) {
            return false;
        }
        glm_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  display_absolute_base,
                                                  work_done_base + i + 1u,
                                                  n_tokens,
                                                  0,
                                                  1,
                                                  work_total,
                                                  false);
    }
    return true;
}

static bool glm_graph_maybe_warm_compact_indexer_after_prefill(
        ds4_glm_gpu_graph      *g,
        const ds4_model        *model,
        const ds4_weights      *weights,
        uint32_t                next_pos) {
    if (!g || !model || !weights) return false;
    if (g->compact_cache_cap == 0 ||
        g->indexer_full_layers == 0 ||
        next_pos < g->ctx_cap) {
        return true;
    }
    if (next_pos >= g->ctx_size) return true;
    return glm_graph_warm_compact_indexer_store(g, model, weights, next_pos);
}

static bool glm_graph_begin_commands_if_needed(void) {
    return ds4_gpu_commands_active() || ds4_gpu_begin_commands() != 0;
}

static bool glm_graph_end_commands_if_active(void) {
    return !ds4_gpu_commands_active() || ds4_gpu_end_commands() != 0;
}

static bool glm_graph_streaming_decode_sync_each_layer(void) {
#ifdef DS4_ROCM_BUILD
    const char *env = glm_graph_env_value(
            "DS4_ROCM_GLM_STREAMING_DECODE_SYNC_EACH_LAYER",
            "DS4_METAL_GLM_STREAMING_DECODE_SYNC_EACH_LAYER");
    if (!env) env = getenv("DS4_GLM_STREAMING_DECODE_SYNC_EACH_LAYER");
    return glm_graph_env_truthy(env);
#else
    return true;
#endif
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
        bool               defer_completion) {
#define DS4_GLM_FT_FAIL(why) do { \
        if (getenv("DS4_GLM_TP_DEBUG")) \
            fprintf(stderr, "ds4: glm forward_token fail pos=%u: %s\n", pos, why); \
    } while (0)
    if (!g || !model || !weights ||
        token < 0 || token >= (int)DS4_N_VOCAB ||
        g->layer_count == 0 ||
        pos >= g->ctx_size ||
        (defer_completion &&
         (g->ssd_streaming || !ds4_gpu_commands_active()))) {
        DS4_GLM_FT_FAIL("arg guard");
        return false;
    }
    if (!input_hc && !g->has_token_embd) { DS4_GLM_FT_FAIL("no token embd"); return false; }
    if (logits_out && !g->has_output_head) { DS4_GLM_FT_FAIL("no output head"); return false; }
    const bool use_indexed_attention =
        glm_graph_decode_uses_indexed_attention(g, pos, logits_out);
    uint32_t decode_layer_flush_interval = 0;
    if (logits_out != NULL) {
        decode_layer_flush_interval = use_indexed_attention ? 4u : 32u;
        const char *dfi = getenv("DS4_GLM_DECODE_FLUSH_INTERVAL");
        if (dfi && dfi[0]) {
            int v = atoi(dfi);
            decode_layer_flush_interval = v <= 0 ? 0u : (uint32_t)v;
        }
        if (decode_layer_flush_interval > g->layer_count) {
            decode_layer_flush_interval = g->layer_count;
        }
        if (defer_completion) decode_layer_flush_interval = 0;
    }
    if (pos >= g->ctx_cap && !use_indexed_attention) {
        glm_graph_log_full_attention_limit(g, pos, 1);
        DS4_GLM_FT_FAIL("full attention limit");
        return false;
    }
    if (g->compact_cache_cap != 0 &&
        !glm_graph_ensure_compact_cache(g, pos + 1u)) {
        DS4_GLM_FT_FAIL("compact cache ensure");
        return false;
    }

    const bool decode_output_profile = false;
    const bool merge_indexed_output =
        logits_out != NULL && use_indexed_attention && !decode_output_profile;
    double decode_output_stage_t0 = decode_output_profile ? now_sec() : 0.0;
    const bool decode_flush_profile = false;
    uint32_t decode_flush_layer0 = 0;
    double decode_flush_stage_t0 = decode_flush_profile ? now_sec() : 0.0;

    const bool static_decode_map =
        !input_hc &&
        g->has_token_embd &&
        g->ssd_streaming &&
        metal_graph_stream_decode_static_map_enabled();
    const bool static_map_state_cache =
        static_decode_map && metal_graph_stream_decode_static_map_state_cache_enabled();
    const bool streaming_decode_sync_each_layer =
        g->ssd_streaming &&
        !static_decode_map &&
        glm_graph_streaming_decode_sync_each_layer();
    bool ok = true;
    if (input_hc) {
        ok = ds4_gpu_tensor_write(g->cur,
                                  0,
                                  input_hc,
                                  (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
    } else if (static_decode_map) {
        if (!static_map_state_cache || !g->streaming_static_decode_map_current) {
            ok = metal_graph_stream_map_decode_static_all(model, weights);
            if (ok) g->streaming_static_decode_map_current = static_map_state_cache;
        }
    } else {
        ok = glm_graph_stream_map_token(g, model, weights);
    }
    if (ok) ok = glm_graph_begin_commands_if_needed();
    if (ok && !input_hc) {
        if (g->placement) {
            ok = glm_graph_ws_switch(g, g->placement[0], false);
        }
    }
    if (ok && !input_hc) {
        ok = ds4_gpu_embed_token_quant_tensor(g->cur,
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              weights->token_embd->type,
                                              DS4_N_VOCAB,
                                              (uint32_t)token,
                                              DS4_N_EMBD) != 0;
    }
    if (ok && streaming_decode_sync_each_layer) {
        ok = ds4_gpu_end_commands() != 0;
    }
    const uint32_t indexer_top_k = glm_graph_indexer_top_k_limit();
    ds4_gpu_tensor *last_indexer_selected = NULL;
    uint32_t last_indexer_selected_count = 0;
#define DS4_GLM_PROFILE_DECODE_STAGE(part_, name_) do { \
        if (ok && decode_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary((part_), (name_), il, pos, 1, &decode_stage_t0); \
        } \
    } while (0)
    uint32_t glm_ft_fail_il = UINT32_MAX;
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        if (g->placement) {
            ok = glm_graph_ws_switch(g, g->placement[il + 1u], true);
            if (!ok) break;
        }
        glm_ft_fail_il = il;
        const uint32_t slice_layer_done = il - g->layer_start + 1u;
        if (g->ssd_streaming) {
            if (!static_decode_map) {
                ok = glm_graph_stream_map_decode_layer(g, model, weights, il);
            }
            if (ok) ok = glm_graph_begin_commands_if_needed();
        }
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t kv_raw_dim = (uint32_t)l->attn_kv_a_mqa->dim[1];
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        const bool decode_stage_profile = metal_graph_decode_stage_profile_enabled(il);
        double decode_stage_t0 = decode_stage_profile ? now_sec() : 0.0;
        if (decode_stage_profile) {
            ok = metal_graph_layer_stage_profile_boundary("glm_decode_attn",
                                                          NULL,
                                                          il,
                                                          pos,
                                                          1,
                                                          &decode_stage_t0);
        }

        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->attn_norm,
                                                    g->cur,
                                                    model->map,
                                                    model->size,
                                                    l->attn_norm->abs_offset,
                                                    DS4_N_EMBD,
                                                    DS4_RMS_EPS) != 0;
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "attn_norm");
        const uint32_t decode_ablate = glm_decode_ablate_mask();
        if (ok && !(decode_ablate & DS4_GLM_ABLATE_QPATH)) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->q_rank,
                                                              model,
                                                              l->attn_q_a->abs_offset,
                                                              DS4_N_EMBD,
                                                              DS4_N_LORA_Q,
                                                              g->attn_norm,
                                                              il,
                                                              pos,
                                                              "attn_q_a",
                                                              g->ssd_streaming) != 0;
        }
        const bool fuse_qkv_norm_store = use_indexed_attention &&
                                         !decode_stage_profile &&
                                         g->compact_cache_cap != 0;
        const bool fuse_qkv_norm = !decode_stage_profile && !fuse_qkv_norm_store;
        if (ok && fuse_qkv_norm_store) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->kv_raw,
                                                              model,
                                                              l->attn_kv_a_mqa->abs_offset,
                                                              DS4_N_EMBD,
                                                              kv_raw_dim,
                                                              g->attn_norm,
                                                              il,
                                                              pos,
                                                              "attn_kv_a_store",
                                                              g->ssd_streaming) != 0;
            if (ok) {
                ok = ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
                        g->q_rank_norm,
                        g->q_rank,
                        model->map,
                        model->size,
                        l->attn_q_a_norm->abs_offset,
                        DS4_N_LORA_Q,
                        g->layer_kv_lora_cache[il],
                        g->layer_k_rope_cache[il],
                        g->kv_raw,
                        l->attn_kv_a_norm->abs_offset,
                        pos,
                        1,
                        g->compact_cache_cap,
                        kv_raw_dim,
                        DS4_N_KV_LORA,
                        DS4_N_ROT,
                        glm_graph_compact_cache_is_f16(),
                        DS4_RMS_EPS) != 0;
            }
        } else if (ok && fuse_qkv_norm) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->kv_raw,
                                                              model,
                                                              l->attn_kv_a_mqa->abs_offset,
                                                              DS4_N_EMBD,
                                                              kv_raw_dim,
                                                              g->attn_norm,
                                                              il,
                                                              pos,
                                                              "attn_kv_a_norm",
                                                              g->ssd_streaming) != 0;
            if (ok) {
                ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->q_rank_norm,
                                                           g->q_rank,
                                                           model->map,
                                                           model->size,
                                                           l->attn_q_a_norm->abs_offset,
                                                           DS4_N_LORA_Q,
                                                           g->kv_norm,
                                                           g->kv_raw,
                                                           l->attn_kv_a_norm->abs_offset,
                                                           DS4_N_KV_LORA,
                                                           1,
                                                           DS4_RMS_EPS) != 0;
            }
        } else if (ok) {
            ok = ds4_gpu_rms_norm_weight_tensor(g->q_rank_norm,
                                                g->q_rank,
                                                model->map,
                                                model->size,
                                                l->attn_q_a_norm->abs_offset,
                                                DS4_N_LORA_Q,
                                                DS4_RMS_EPS) != 0;
        }
        if (ok && !(decode_ablate & DS4_GLM_ABLATE_QPATH)) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->q,
                                                              model,
                                                              l->attn_q_b->abs_offset,
                                                              DS4_N_LORA_Q,
                                                              g->q_dim,
                                                              g->q_rank_norm,
                                                              il,
                                                              pos,
                                                              "attn_q_b",
                                                              g->ssd_streaming) != 0;
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
        }
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "q_path");
        if (ok) metal_graph_debug_dump_tensor("glm_decode_q",
                                              g->q,
                                              (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA,
                                              il,
                                              pos);
        if (ok && g->compact_cache_cap != 0 && glm_graph_layer_uses_full_indexer(il) &&
            !(decode_ablate & DS4_GLM_ABLATE_INDEXER)) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->indexer_k,
                                                              model,
                                                              l->indexer_attn_k->abs_offset,
                                                              DS4_N_EMBD,
                                                              DS4_N_INDEXER_HEAD_DIM,
                                                              g->cur,
                                                              il,
                                                              pos,
                                                              "indexer_k",
                                                              g->ssd_streaming) != 0;
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_k_proj");
            if (ok) {
                ok = ds4_gpu_glm_store_indexer_k_tensor(
                        g->layer_indexer_key_cache[il],
                        g->indexer_k,
                        model->map,
                        model->size,
                        l->indexer_k_norm->abs_offset,
                        l->indexer_k_norm_b->abs_offset,
                        pos,
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
            }
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_k_store");
        }
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_k");
        if (ok && !fuse_qkv_norm && !fuse_qkv_norm_store) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->kv_raw,
                                                              model,
                                                              l->attn_kv_a_mqa->abs_offset,
                                                              DS4_N_EMBD,
                                                              kv_raw_dim,
                                                              g->attn_norm,
                                                              il,
                                                              pos,
                                                              "attn_kv_a",
                                                              g->ssd_streaming) != 0;
            if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(g->kv_norm,
                                                             g->kv_raw,
                                                             model->map,
                                                             model->size,
                                                             l->attn_kv_a_norm->abs_offset,
                                                             1,
                                                             kv_raw_dim,
                                                             DS4_N_KV_LORA,
                                                             DS4_RMS_EPS) != 0;
        }
        if (ok && g->compact_cache_cap != 0 && !fuse_qkv_norm_store) {
            ok = ds4_gpu_glm_store_compact_kv_tensor(g->layer_kv_lora_cache[il],
                                                     g->layer_k_rope_cache[il],
                                                     g->kv_norm,
                                                     g->kv_raw,
                                                     pos,
                                                     1,
                                                     g->compact_cache_cap,
                                                     kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_N_ROT,
                                                     glm_graph_compact_cache_is_f16()) != 0;
        }
        if (use_indexed_attention) {
            if (ok && glm_graph_layer_uses_full_indexer(il)) {
                const uint32_t visible = pos + 1u;
                if (ok && visible <= indexer_top_k) {
                    ok = ds4_gpu_glm_fill_selected_range_tensor(g->indexer_selected,
                                                                visible) != 0;
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_fill");
                    last_indexer_selected_count = visible;
                } else if (ok && (decode_ablate & DS4_GLM_ABLATE_INDEXER)) {
                    /* Ablation: valid selected ids without the score/topk
                     * chain, so downstream attention timing stays real. */
                    ok = ds4_gpu_glm_fill_selected_range_tensor(g->indexer_selected,
                                                                indexer_top_k) != 0;
                    last_indexer_selected_count = indexer_top_k;
                } else if (ok) {
                    ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->indexer_q,
                                                                      model,
                                                                      l->indexer_attn_q_b->abs_offset,
                                                                      DS4_N_LORA_Q,
                                                                      (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                                      g->q_rank_norm,
                                                                      il,
                                                                      pos,
                                                                      "indexer_q",
                                                                      g->ssd_streaming) != 0;
                    if (ok) ok = ds4_gpu_glm_indexer_rope_tail_tensor(g->indexer_q,
                                                                      1,
                                                                      DS4_N_INDEXER_HEAD,
                                                                      DS4_N_INDEXER_HEAD_DIM,
                                                                      DS4_N_ROT,
                                                                      pos,
                                                                      0,
                                                                      rope_base,
                                                                      rope_scale,
                                                                      0.0f,
                                                                      1.0f,
                                                                      DS4_ROPE_YARN_BETA_FAST,
                                                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
                    if (ok && glm_graph_indexer_qat()) {
                        ok = ds4_gpu_dsv4_indexer_qat_tensor(g->indexer_q,
                                                             DS4_N_INDEXER_HEAD,
                                                             DS4_N_INDEXER_HEAD_DIM) != 0;
                    }
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_q");
                    if (ok) ok = ds4_gpu_matmul_f32_tensor(g->indexer_weights,
                                                           model->map,
                                                           model->size,
                                                           l->indexer_proj->abs_offset,
                                                           DS4_N_EMBD,
                                                           DS4_N_INDEXER_HEAD,
                                                           g->cur,
                                                           1) != 0;
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_weights");
                    const float indexer_scale =
                        1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    ok = ds4_gpu_glm_indexer_score_one_tensor(g->indexer_scores,
                                                              g->indexer_q,
                                                              g->indexer_weights,
                                                              g->layer_indexer_key_cache[il],
                                                              visible,
                                                              DS4_N_INDEXER_HEAD,
                                                              DS4_N_INDEXER_HEAD_DIM,
                                                              indexer_scale,
                                                              glm_graph_compact_cache_is_f16()) != 0;
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_scores");
                    if (ok) ok = ds4_gpu_indexer_topk_tensor(g->indexer_selected,
                                                             g->indexer_scores,
                                                             visible,
                                                             1,
                                                             indexer_top_k) != 0;
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_topk");
                    last_indexer_selected_count = indexer_top_k;
                }
                if (ok) last_indexer_selected = g->indexer_selected;
            } else if (ok && (!last_indexer_selected || last_indexer_selected_count == 0)) {
                ok = false;
            }
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_select");
            if (ok && !(decode_ablate & (DS4_GLM_ABLATE_ATTN_CORE | DS4_GLM_ABLATE_QKLOW))) {
                ok = ds4_gpu_glm_qk_lowrank_typed_tensor(g->qk_low,
                                                         g->q,
                                                         model->map,
                                                         model->size,
                                                         l->attn_k_b->abs_offset,
                                                         l->attn_k_b->type,
                                                         DS4_N_HEAD,
                                                         DS4_N_KV_LORA,
                                                         (uint32_t)g->q_nope,
                                                         DS4_N_KEY_MLA) != 0;
                if (ok) metal_graph_debug_dump_tensor("glm_decode_qk_low",
                                                      g->qk_low,
                                                      (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA,
                                                      il,
                                                      pos);
            }
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "kv_path");
            if (ok && (decode_ablate & DS4_GLM_ABLATE_ATTN_CORE)) {
                /* Skip the indexed attention kernels; zero heads so the
                 * rest of the layer stays finite (timing-only). */
                ok = ds4_gpu_tensor_fill_f32(g->heads, 0.0f,
                                             (uint64_t)g->heads_dim) != 0;
            } else if (ok && glm_graph_indexed_decode_split_group8_available(last_indexer_selected_count)) {
                const uint32_t split_block_rows =
                    glm_graph_indexed_decode_split_block_rows_for(last_indexer_selected_count);
                const uint32_t split_blocks =
                    (last_indexer_selected_count + split_block_rows - 1u) / split_block_rows;
                ok = ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor(g->heads,
                                                                                    g->attn_partial_lora,
                                                                                    g->attn_partial_ms,
                                                                                    g->q,
                                                                                    g->qk_low,
                                                                                    g->layer_kv_lora_cache[il],
                                                                                    g->layer_k_rope_cache[il],
                                                                                    model->map,
                                                                                    model->size,
                                                                                    l->attn_v_b->abs_offset,
                                                                                    l->attn_v_b->type,
                                                                                    last_indexer_selected,
                                                                                    last_indexer_selected_count,
                                                                                    true,
                                                                                    g->compact_cache_cap,
                                                                                    glm_graph_compact_cache_is_f16(),
                                                                                    DS4_N_HEAD,
                                                                                    DS4_N_KV_LORA,
                                                                                    (uint32_t)g->q_nope,
                                                                                    DS4_N_ROT,
                                                                                    DS4_N_VALUE_MLA,
                                                                                    0,
                                                                                    split_block_rows,
                                                                                    split_blocks,
                                                                                    rope_base,
                                                                                    rope_scale,
                                                                                    0.0f,
                                                                                    1.0f,
                                                                                    DS4_ROPE_YARN_BETA_FAST,
                                                                                    DS4_ROPE_YARN_BETA_SLOW) != 0;
            } else if (ok) {
                ok = ds4_gpu_glm_attention_indexed_decode_typed_tensor(g->heads,
                                                                       g->q,
                                                                       g->qk_low,
                                                                       g->layer_kv_lora_cache[il],
                                                                       g->layer_k_rope_cache[il],
                                                                       model->map,
                                                                       model->size,
                                                                       l->attn_v_b->abs_offset,
                                                                       l->attn_v_b->type,
                                                                       last_indexer_selected,
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
                                                                       DS4_ROPE_YARN_BETA_SLOW) != 0;
            }
        } else {
            if (ok) ok = ds4_gpu_glm_k_b_project_typed_tensor(g->k_nope,
                                                              g->kv_norm,
                                                              model->map,
                                                              model->size,
                                                              l->attn_k_b->abs_offset,
                                                              l->attn_k_b->type,
                                                              1,
                                                              DS4_N_KV_LORA,
                                                              (uint32_t)g->q_nope,
                                                              DS4_N_HEAD) != 0;
            if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->value,
                                                                      model,
                                                                      l->attn_v_b->abs_offset,
                                                                      DS4_N_KV_LORA,
                                                                      g->heads_dim,
                                                                      g->kv_norm,
                                                                      il,
                                                                      pos,
                                                                      "attn_v_b",
                                                                      g->ssd_streaming) != 0;
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "kv_path");
            if (ok) ok = ds4_gpu_glm_build_kv_cache_tensor(g->layer_key_cache[il],
                                                           g->layer_value_cache[il],
                                                           g->kv_raw,
                                                           g->k_nope,
                                                           g->value,
                                                           pos,
                                                           1,
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
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "kv_cache");
            if (ok) ok = ds4_gpu_glm_attention_full_tensor(g->heads,
                                                           g->q,
                                                           g->layer_key_cache[il],
                                                           g->layer_value_cache[il],
                                                           pos,
                                                           1,
                                                           pos + 1u,
                                                           g->ctx_cap,
                                                           DS4_N_HEAD,
                                                           DS4_N_KEY_MLA,
                                                           DS4_N_VALUE_MLA,
                                                           true) != 0;
        }
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "attention");
        if (ok) metal_graph_debug_dump_tensor("glm_decode_heads",
                                              g->heads,
                                              g->heads_dim,
                                              il,
                                              pos);
        if (ok && !(decode_ablate & DS4_GLM_ABLATE_ATTN_OUT)) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->attn_out,
                                                              model,
                                                              l->attn_output->abs_offset,
                                                              g->heads_dim,
                                                              DS4_N_EMBD,
                                                              g->heads,
                                                              il,
                                                              pos,
                                                              "attn_o",
                                                              g->ssd_streaming) != 0;
        }
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "attn_output");
        if (ok) ok = ds4_gpu_add_rms_norm_weight_tensor(g->ffn_norm,
                                                        g->after_attn,
                                                        g->cur,
                                                        g->attn_out,
                                                        model->map,
                                                        model->size,
                                                        l->ffn_norm->abs_offset,
                                                        DS4_N_EMBD,
                                                        DS4_RMS_EPS) != 0;
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_ffn", "ffn_norm");
        if (ok) ok = glm_graph_encode_ffn_one_normed_from(g,
                                                          model,
                                                          l,
                                                          il,
                                                          pos,
                                                          g->ffn_norm,
                                                          g->after_attn,
                                                          g->next,
                                                          g->ffn_gate,
                                                          g->ffn_up,
                                                          g->ffn_mid,
                                                          g->ffn_out,
                                                          g->ffn_sum,
                                                          g->attn_out,
                                                          decode_stage_profile,
                                                          decode_stage_profile ? &decode_stage_t0 : NULL);
        if (ok) {
            ds4_gpu_tensor *tmp = g->cur;
            g->cur = g->next;
            g->next = tmp;
        }
        if (ok && glm_debug_hidden_dump_layer_match(il)) {
            ok = ds4_gpu_end_commands() != 0;
            if (ok) {
                glm_debug_dump_hidden_layer(g->cur, 0, il, pos);
                glm_debug_dump_raw_layer(g->router_selected, "sel",
                                         (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t),
                                         il, (int)pos);
                glm_debug_dump_raw_layer(g->router_weights, "selw",
                                         (uint64_t)DS4_N_EXPERT_USED * sizeof(float),
                                         il, (int)pos);
                ok = ds4_gpu_begin_commands() != 0;
            }
        }
        if (ok &&
            !g->ssd_streaming &&
            decode_layer_flush_interval != 0 &&
            il < g->layer_end &&
            (slice_layer_done % decode_layer_flush_interval) == 0) {
            if (decode_flush_profile) {
                ok = ds4_gpu_flush_commands() != 0;
                if (ok) ok = ds4_gpu_synchronize() != 0;
                if (ok) {
                    const double now = now_sec();
                    fprintf(stderr,
                            "ds4: GLM decode layer flush pos=%u layers=%u..%u %.3f ms\n",
                            pos,
                            decode_flush_layer0,
                            il,
                            (now - decode_flush_stage_t0) * 1000.0);
                    decode_flush_layer0 = il + 1u;
                    decode_flush_stage_t0 = now;
                    ok = ds4_gpu_begin_commands() != 0;
                }
            } else {
                ok = ds4_gpu_flush_commands() != 0;
            }
        }
        if (streaming_decode_sync_each_layer) {
            if (ok) ok = ds4_gpu_end_commands() != 0;
            else (void)ds4_gpu_synchronize();
        }
    }
#undef DS4_GLM_PROFILE_DECODE_STAGE
    if (ok && (merge_indexed_output ||
               (defer_completion && logits_out != NULL))) {
        if (g->ssd_streaming) {
            if (!static_decode_map) {
                ok = glm_graph_stream_map_output(g, model, weights);
            }
            if (ok) ok = glm_graph_begin_commands_if_needed();
        }
        ok = glm_graph_encode_output_head(g, model, weights);
        if (g->ssd_streaming) {
            if (ok) ok = glm_graph_end_commands_if_active();
            else (void)ds4_gpu_synchronize();
        }
    }
    if (!g->ssd_streaming && !defer_completion) {
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
    } else if (!ok) {
        (void)ds4_gpu_synchronize();
    }
    if (decode_output_profile) {
        const double now = now_sec();
        fprintf(stderr,
                "ds4: GLM decode output profile pos=%u layers=%.3f ms\n",
                pos,
                (now - decode_output_stage_t0) * 1000.0);
        decode_output_stage_t0 = now;
    }
    if (ok && output_hc && !defer_completion) {
        ok = ds4_gpu_tensor_read(g->cur,
                                 0,
                                 output_hc,
                                 (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
    }
    if (ok && logits_out && !defer_completion) {
        if (use_indexed_attention) {
            if (!merge_indexed_output) {
                if (g->ssd_streaming && !static_decode_map) {
                    ok = glm_graph_stream_map_output(g, model, weights);
                }
                if (ok) ok = glm_graph_begin_commands_if_needed();
                if (ok) ok = glm_graph_encode_output_head(g, model, weights);
                if (ok) ok = glm_graph_end_commands_if_active();
                else (void)ds4_gpu_synchronize();
                if (decode_output_profile) {
                    const double now = now_sec();
                    fprintf(stderr,
                            "ds4: GLM decode output profile pos=%u output_head=%.3f ms\n",
                            pos,
                            (now - decode_output_stage_t0) * 1000.0);
                    decode_output_stage_t0 = now;
                }
            }
            if (ok) {
                if (glm_debug_hidden_dump_layer() < 0)
                    glm_debug_dump_hidden_row(g->cur, 0);
                ok = ds4_gpu_tensor_read(g->logits,
                                         0,
                                         logits_out,
                                         (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
                if (decode_output_profile) {
                    const double now = now_sec();
                    fprintf(stderr,
                            "ds4: GLM decode output profile pos=%u logits_read=%.3f ms\n",
                            pos,
                            (now - decode_output_stage_t0) * 1000.0);
                }
            }
        } else {
            if (g->ssd_streaming && !static_decode_map) {
                ok = glm_graph_stream_map_output(g, model, weights);
            }
            if (ok && ds4_gpu_commands_active()) {
                ok = ds4_gpu_end_commands() != 0;
            }
            if (ok) ok = glm_graph_forward_output_head(g, model, weights, g->cur, logits_out);
            if (decode_output_profile) {
                const double now = now_sec();
                fprintf(stderr,
                        "ds4: GLM decode output profile pos=%u fallback_output=%.3f ms\n",
                        pos,
                        (now - decode_output_stage_t0) * 1000.0);
            }
        }
    }
    if (ok &&
        !logits_out &&
        !output_hc &&
        g->ssd_streaming &&
        !streaming_decode_sync_each_layer) {
        ok = ds4_gpu_end_commands() != 0;
    } else if (ok && !logits_out && g->ssd_streaming) {
        ok = glm_graph_end_commands_if_active();
    } else if (!ok) {
        (void)ds4_gpu_synchronize();
    }
    if (!ok && getenv("DS4_GLM_TP_DEBUG")) {
        fprintf(stderr,
                "ds4: glm forward_token fail pos=%u around layer %u\n",
                pos, glm_ft_fail_il);
    }
    (void)glm_ft_fail_il;
    return ok;
#undef DS4_GLM_FT_FAIL
}

static int glm_metal_first_token_logits(
        const ds4_model   *model,
        const ds4_weights *weights,
        int                token,
        float             *logits_out) {
    if (!model || !weights || !logits_out) return 1;
    if (token < 0 || token >= (int)DS4_N_VOCAB) {
        fprintf(stderr, "ds4: GLM token %d is outside vocab\n", token);
        return 1;
    }
    if (!weights->token_embd || weights->token_embd->type != DS4_TENSOR_Q8_0 ||
        !weights->output_norm || weights->output_norm->type != DS4_TENSOR_F32 ||
        !weights->output || weights->output->type != DS4_TENSOR_Q8_0 ||
        weights->output_norm->dim[0] != DS4_N_EMBD ||
        weights->output->dim[0] != DS4_N_EMBD ||
        weights->output->dim[1] != DS4_N_VOCAB) {
        fprintf(stderr, "ds4: GLM Metal first-token path found unexpected embedding/output layout\n");
        return 1;
    }
    if (DS4_N_LAYER <= DS4_N_NEXTN_PREDICT) {
        fprintf(stderr, "ds4: GLM Metal first-token path has no normal transformer layers\n");
        return 1;
    }

    const uint32_t normal_layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    const uint64_t heads_dim = (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA;
    uint64_t kv_raw_dim = 0;
    uint64_t dense_hidden_max = DS4_N_FF_EXP;
    bool generic_routed_moe = false;
    for (uint32_t il = 0; il < normal_layers; il++) {
        const ds4_layer_weights *l = &weights->layer[il];
        if (l->attn_kv_a_mqa && l->attn_kv_a_mqa->dim[1] > kv_raw_dim) {
            kv_raw_dim = l->attn_kv_a_mqa->dim[1];
        }
        if (il < DS4_N_LEADING_DENSE && l->ffn_gate &&
            l->ffn_gate->dim[1] > dense_hidden_max) {
            dense_hidden_max = l->ffn_gate->dim[1];
        }
        if (glm_graph_layer_uses_generic_routed_moe(l)) generic_routed_moe = true;
    }
    if (kv_raw_dim < DS4_N_KV_LORA) {
        fprintf(stderr, "ds4: GLM Metal first-token path found no valid KV projection\n");
        return 1;
    }

    const uint64_t emb_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t sparse_mid_elems = (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    const uint64_t ffn_mid_elems =
        dense_hidden_max > sparse_mid_elems ? dense_hidden_max : sparse_mid_elems;
    const uint64_t routed_mid_bytes =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(float);
    const uint64_t routed_down_bytes =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);

    ds4_gpu_tensor *cur = NULL;
    ds4_gpu_tensor *attn_norm = NULL;
    ds4_gpu_tensor *kv_raw = NULL;
    ds4_gpu_tensor *kv_norm = NULL;
    ds4_gpu_tensor *heads = NULL;
    ds4_gpu_tensor *attn_out = NULL;
    ds4_gpu_tensor *after_attn = NULL;
    ds4_gpu_tensor *ffn_norm = NULL;
    ds4_gpu_tensor *ffn_gate = NULL;
    ds4_gpu_tensor *ffn_up = NULL;
    ds4_gpu_tensor *ffn_mid = NULL;
    ds4_gpu_tensor *routed_gate = NULL;
    ds4_gpu_tensor *routed_up = NULL;
    ds4_gpu_tensor *routed_down = NULL;
    ds4_gpu_tensor *ffn_out = NULL;
    ds4_gpu_tensor *ffn_sum = NULL;
    ds4_gpu_tensor *next = NULL;
    ds4_gpu_tensor *router_logits = NULL;
    ds4_gpu_tensor *router_probs = NULL;
    ds4_gpu_tensor *router_selected = NULL;
    ds4_gpu_tensor *router_weights = NULL;
    ds4_gpu_tensor *logits = NULL;

    int ok = 1;
#define DS4_GLM_FIRST_ALLOC_TENSOR(var, bytes_) \
    do { \
        (var) = ds4_gpu_tensor_alloc((bytes_)); \
        if (!(var)) { \
            fprintf(stderr, "ds4: GLM Metal first-token path could not allocate %s\n", #var); \
            ok = 0; \
        } \
    } while (0)

    DS4_GLM_FIRST_ALLOC_TENSOR(cur, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(attn_norm, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(kv_raw, kv_raw_dim * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(kv_norm, (uint64_t)DS4_N_KV_LORA * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(heads, heads_dim * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(attn_out, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(after_attn, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(ffn_norm, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(ffn_gate, dense_hidden_max * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(ffn_up, dense_hidden_max * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(ffn_mid, ffn_mid_elems * sizeof(float));
    if (generic_routed_moe) {
        DS4_GLM_FIRST_ALLOC_TENSOR(routed_gate, routed_mid_bytes);
        DS4_GLM_FIRST_ALLOC_TENSOR(routed_up, routed_mid_bytes);
        DS4_GLM_FIRST_ALLOC_TENSOR(routed_down, routed_down_bytes);
    }
    DS4_GLM_FIRST_ALLOC_TENSOR(ffn_out, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(ffn_sum, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(next, emb_bytes);
    DS4_GLM_FIRST_ALLOC_TENSOR(router_logits, (uint64_t)DS4_N_EXPERT * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(router_probs, (uint64_t)DS4_N_EXPERT * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(router_selected, (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
    DS4_GLM_FIRST_ALLOC_TENSOR(router_weights, (uint64_t)DS4_N_EXPERT_USED * sizeof(float));
    DS4_GLM_FIRST_ALLOC_TENSOR(logits, logits_bytes);
#undef DS4_GLM_FIRST_ALLOC_TENSOR

    if (ok) {
        ok = ds4_gpu_embed_token_q8_0_tensor(cur,
                                             model->map,
                                             model->size,
                                             weights->token_embd->abs_offset,
                                             DS4_N_VOCAB,
                                             (uint32_t)token,
                                             DS4_N_EMBD);
    }
    for (uint32_t il = 0; ok && il < normal_layers; il++) {
        const ds4_layer_weights *gl = &weights->layer[il];
        const uint64_t gl_kv_raw_dim = gl->attn_kv_a_mqa ? gl->attn_kv_a_mqa->dim[1] : 0;
        if (!gl->attn_norm ||
            !gl->attn_kv_a_mqa ||
            !gl->attn_kv_a_norm ||
            !gl->attn_v_b ||
            !gl->attn_output ||
            !gl->ffn_norm ||
            gl_kv_raw_dim < DS4_N_KV_LORA ||
            gl_kv_raw_dim > kv_raw_dim ||
            gl->attn_kv_a_mqa->type != DS4_TENSOR_Q8_0 ||
            gl->attn_kv_a_mqa->dim[0] != DS4_N_EMBD ||
            gl->attn_v_b->type != DS4_TENSOR_Q8_0 ||
            gl->attn_v_b->dim[0] != DS4_N_KV_LORA ||
            gl->attn_v_b->dim[1] != DS4_N_VALUE_MLA ||
            gl->attn_v_b->dim[2] != DS4_N_HEAD ||
            gl->attn_output->type != DS4_TENSOR_Q8_0 ||
            gl->attn_output->dim[0] != heads_dim ||
            gl->attn_output->dim[1] != DS4_N_EMBD) {
            fprintf(stderr,
                    "ds4: GLM Metal first-token path found unexpected attention layout in layer %u\n",
                    il);
            ok = 0;
            break;
        }

        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(attn_norm, cur,
                                                    model->map, model->size,
                                                    gl->attn_norm->abs_offset,
                                                    DS4_N_EMBD, DS4_RMS_EPS);
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(kv_raw,
                                                model->map,
                                                model->size,
                                                gl->attn_kv_a_mqa->abs_offset,
                                                DS4_N_EMBD,
                                                gl_kv_raw_dim,
                                                attn_norm,
                                                1);
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(kv_norm, kv_raw,
                                                    model->map, model->size,
                                                    gl->attn_kv_a_norm->abs_offset,
                                                    DS4_N_KV_LORA, DS4_RMS_EPS);
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(heads,
                                                model->map,
                                                model->size,
                                                gl->attn_v_b->abs_offset,
                                                DS4_N_KV_LORA,
                                                heads_dim,
                                                kv_norm,
                                                1);
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(attn_out,
                                                model->map,
                                                model->size,
                                                gl->attn_output->abs_offset,
                                                heads_dim,
                                                DS4_N_EMBD,
                                                heads,
                                                1);
        if (ok) ok = ds4_gpu_add_tensor(after_attn, cur, attn_out, DS4_N_EMBD);
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(ffn_norm, after_attn,
                                                    model->map, model->size,
                                                    gl->ffn_norm->abs_offset,
                                                    DS4_N_EMBD, DS4_RMS_EPS);
        if (il < DS4_N_LEADING_DENSE) {
            const uint64_t gl_ffn_hidden = gl->ffn_gate ? gl->ffn_gate->dim[1] : 0;
            if (!gl->ffn_gate ||
                !gl->ffn_up ||
                !gl->ffn_down ||
                gl->ffn_gate->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_up->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_down->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_gate->dim[0] != DS4_N_EMBD ||
                gl->ffn_up->dim[0] != DS4_N_EMBD ||
                gl->ffn_up->dim[1] != gl_ffn_hidden ||
                gl->ffn_down->dim[0] != gl_ffn_hidden ||
                gl->ffn_down->dim[1] != DS4_N_EMBD ||
                gl_ffn_hidden > dense_hidden_max) {
                fprintf(stderr,
                        "ds4: GLM Metal first-token path found unexpected dense FFN layout in layer %u\n",
                        il);
                ok = 0;
                break;
            }
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_gate,
                                                    model->map,
                                                    model->size,
                                                    gl->ffn_gate->abs_offset,
                                                    DS4_N_EMBD,
                                                    gl_ffn_hidden,
                                                    ffn_norm,
                                                    1);
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_up,
                                                    model->map,
                                                    model->size,
                                                    gl->ffn_up->abs_offset,
                                                    DS4_N_EMBD,
                                                    gl_ffn_hidden,
                                                    ffn_norm,
                                                    1);
            if (ok) ok = ds4_gpu_swiglu_tensor(ffn_mid, ffn_gate, ffn_up,
                                               (uint32_t)gl_ffn_hidden, 0.0f, 1.0f);
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_out,
                                                    model->map,
                                                    model->size,
                                                    gl->ffn_down->abs_offset,
                                                    gl_ffn_hidden,
                                                    DS4_N_EMBD,
                                                    ffn_mid,
                                                    1);
            if (ok) ok = ds4_gpu_add_tensor(next, after_attn, ffn_out, DS4_N_EMBD);
        } else {
            const uint32_t gl_gate_type = gl->ffn_gate_exps ? gl->ffn_gate_exps->type : 0;
            const uint32_t gl_up_type = gl->ffn_up_exps ? gl->ffn_up_exps->type : 0;
            const bool gl_gate_pair_supported =
                glm_graph_gate_pair_type_supported(gl_gate_type, gl_up_type);
            uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
            uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
            uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;

            if (!gl->ffn_gate_inp ||
                !gl->ffn_exp_probs_b ||
                !gl->ffn_gate_exps ||
                !gl->ffn_up_exps ||
                !gl->ffn_down_exps ||
                !gl->ffn_gate_shexp ||
                !gl->ffn_up_shexp ||
                !gl->ffn_down_shexp ||
                gl->ffn_gate_inp->type != DS4_TENSOR_F32 ||
                gl->ffn_gate_inp->dim[0] != DS4_N_EMBD ||
                gl->ffn_gate_inp->dim[1] != DS4_N_EXPERT ||
                gl->ffn_exp_probs_b->type != DS4_TENSOR_F32 ||
                gl->ffn_exp_probs_b->dim[0] != DS4_N_EXPERT ||
                !gl_gate_pair_supported ||
                !glm_graph_down_type_supported(gl->ffn_down_exps->type) ||
                gl->ffn_gate_shexp->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_up_shexp->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_down_shexp->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_gate_shexp->dim[0] != DS4_N_EMBD ||
                gl->ffn_gate_shexp->dim[1] != DS4_N_FF_EXP ||
                gl->ffn_up_shexp->dim[0] != DS4_N_EMBD ||
                gl->ffn_up_shexp->dim[1] != DS4_N_FF_EXP ||
                gl->ffn_down_shexp->dim[0] != DS4_N_FF_EXP ||
                gl->ffn_down_shexp->dim[1] != DS4_N_EMBD ||
                sparse_mid_elems > ffn_mid_elems) {
                fprintf(stderr,
                        "ds4: GLM Metal first-token path found unexpected sparse FFN layout in layer %u\n",
                        il);
                ok = 0;
                break;
            }

            (void)tensor_expert_bytes(model, gl->ffn_gate_exps, 0,
                                      &gate_in, &gate_out, &gate_row_bytes);
            (void)tensor_expert_bytes(model, gl->ffn_up_exps, 0,
                                      &up_in, &up_out, &up_row_bytes);
            (void)tensor_expert_bytes(model, gl->ffn_down_exps, 0,
                                      &down_in, &down_out, &down_row_bytes);
            if (gate_in != DS4_N_EMBD ||
                up_in != DS4_N_EMBD ||
                down_in != DS4_N_FF_EXP ||
                gate_out != DS4_N_FF_EXP ||
                up_out != DS4_N_FF_EXP ||
                down_out != DS4_N_EMBD) {
                fprintf(stderr,
                        "ds4: GLM Metal first-token path found unexpected expert strides in layer %u\n",
                        il);
                ok = 0;
                break;
            }

            if (ok) ok = ds4_gpu_matmul_f32_tensor(router_logits,
                                                   model->map,
                                                   model->size,
                                                   gl->ffn_gate_inp->abs_offset,
                                                   DS4_N_EMBD,
                                                   DS4_N_EXPERT,
                                                   ffn_norm,
                                                   1);
            if (ok) ok = ds4_gpu_glm_router_select_tensor(router_selected,
                                                          router_weights,
                                                          router_probs,
                                                          model->map,
                                                          model->size,
                                                          gl->ffn_exp_probs_b->abs_offset,
                                                          router_logits,
                                                          DS4_N_EXPERT,
                                                          DS4_N_EXPERT_USED,
                                                          DS4_EXPERT_WEIGHT_SCALE);
            if (ok) {
                const ds4_gpu_stream_expert_table table = {
                    .model_map = model->map,
                    .model_size = model->size,
                    .layer = il,
                    .n_total_expert = DS4_N_EXPERT,
                    .gate_offset = gl->ffn_gate_exps->abs_offset,
                    .up_offset = gl->ffn_up_exps->abs_offset,
                    .down_offset = gl->ffn_down_exps->abs_offset,
                    .gate_expert_bytes = gate_out * gate_row_bytes,
                    .down_expert_bytes = down_out * down_row_bytes,
                };
                ok = ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
                        &table,
                        router_selected,
                        DS4_N_EXPERT_USED) != 0;
            }
            ds4_glm_gpu_graph route_g = {
                .routed_gate = routed_gate,
                .routed_up = routed_up,
                .routed_down = routed_down,
                .ssd_streaming = false,
            };
            if (ok) ok = glm_graph_routed_moe_one_dispatch(
                    &route_g,
                    model,
                    gl,
                    il,
                    ffn_out,
                    ffn_mid,
                    gate_out * gate_row_bytes,
                    gate_row_bytes,
                    up_out * up_row_bytes,
                    up_row_bytes,
                    down_out * down_row_bytes,
                    down_row_bytes,
                    router_selected,
                    router_weights,
                    ffn_norm,
                    false);
            if (ok) ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                    ffn_gate,
                    ffn_up,
                    ffn_mid,
                    model->map,
                    model->size,
                    gl->ffn_gate_shexp->abs_offset,
                    gl->ffn_up_shexp->abs_offset,
                    DS4_N_EMBD,
                    DS4_N_FF_EXP,
                    ffn_norm,
                    0.0f);
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_sum,
                                                    model->map,
                                                    model->size,
                                                    gl->ffn_down_shexp->abs_offset,
                                                    DS4_N_FF_EXP,
                                                    DS4_N_EMBD,
                                                    ffn_mid,
                                                    1);
            if (ok) ok = ds4_gpu_add_tensor(attn_out, ffn_out, ffn_sum, DS4_N_EMBD);
            if (ok) ok = ds4_gpu_add_tensor(next, after_attn, attn_out, DS4_N_EMBD);
        }

        if (ok) {
            ds4_gpu_tensor *tmp = cur;
            cur = next;
            next = tmp;
        }
    }
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(ffn_norm, cur,
                                                model->map, model->size,
                                                weights->output_norm->abs_offset,
                                                DS4_N_EMBD, DS4_RMS_EPS);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(logits,
                                            model->map,
                                            model->size,
                                            weights->output->abs_offset,
                                            DS4_N_EMBD,
                                            DS4_N_VOCAB,
                                            ffn_norm,
                                            1);
    if (ok) ok = ds4_gpu_tensor_read(logits, 0, logits_out, logits_bytes) != 0;

    ds4_gpu_tensor_free(router_weights);
    ds4_gpu_tensor_free(router_selected);
    ds4_gpu_tensor_free(router_probs);
    ds4_gpu_tensor_free(router_logits);
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(next);
    ds4_gpu_tensor_free(ffn_sum);
    ds4_gpu_tensor_free(ffn_out);
    ds4_gpu_tensor_free(routed_down);
    ds4_gpu_tensor_free(routed_up);
    ds4_gpu_tensor_free(routed_gate);
    ds4_gpu_tensor_free(ffn_mid);
    ds4_gpu_tensor_free(ffn_up);
    ds4_gpu_tensor_free(ffn_gate);
    ds4_gpu_tensor_free(ffn_norm);
    ds4_gpu_tensor_free(after_attn);
    ds4_gpu_tensor_free(attn_out);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(kv_norm);
    ds4_gpu_tensor_free(kv_raw);
    ds4_gpu_tensor_free(attn_norm);
    ds4_gpu_tensor_free(cur);
    return ok ? 0 : 1;
}

static DS4_MAYBE_UNUSED int generate_glm_metal_first_token(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud) {
    fprintf(stderr, "ds4: using GLM Metal first-token generation path\n");

    if (prompt->len != 1 || prompt->len > ctx_size) {
        fprintf(stderr,
                "ds4: GLM Metal generation currently supports exactly one prompt token; "
                "multi-token prefill needs the GLM KV/DSA graph\n");
        return 1;
    }
    if (n_predict <= 0) {
        if (done) done(emit_ud);
        return 0;
    }
    if (n_predict > 1) {
        fprintf(stderr,
                "ds4: GLM Metal generation currently emits only the first generated token; "
                "stopping after one token\n");
    }

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    const double t0 = now_sec();
    const int rc = glm_metal_first_token_logits(model, weights, prompt->v[0], logits);
    const double t1 = now_sec();
    if (rc != 0) {
        free(logits);
        return 1;
    }

    if (getenv("DS4_TRACE_TOP") != NULL) {
        print_top_logits(stderr, "GLM first-token", vocab, logits, DS4_N_VOCAB, 10);
    }
    const int token = sample_argmax(logits, DS4_N_VOCAB);
    if (!vocab_token_is_generation_stop(vocab, token) && emit) emit(emit_ud, token);
    if (done) done(emit_ud);

    const double eval_s = t1 - t0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: GLM first-token eval: %.2f t/s\n",
            eval_s > 0.0 ? 1.0 / eval_s : 0.0);

    free(logits);
    return 0;
}

static int generate_glm_metal_argmax(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        bool                quality,
        bool                ssd_streaming,
        bool                ssd_streaming_cold,
        uint32_t            ssd_streaming_preload_experts,
        uint64_t            ssd_streaming_cache_bytes,
        uint64_t            ssd_streaming_prefill_headroom_bytes,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud,
        ds4_session_progress_fn progress,
        void              * progress_ud) {
    fprintf(stderr, "ds4: using GLM full-attention argmax generation path\n");

    if (!prompt || prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        return 1;
    }
    if (n_predict <= 0) {
        if (done) done(emit_ud);
        return 0;
    }

    ds4_glm_gpu_graph g = {0};
    if (!glm_graph_alloc(&g,
                         model,
                         weights,
                         ctx_size,
                         ssd_streaming,
                         ssd_streaming_cold)) {
        fprintf(stderr, "ds4: failed to allocate GLM graph runtime\n");
        return 1;
    }
    g.quality = quality;
    if ((uint32_t)prompt->len >= g.ctx_size) {
        fprintf(stderr,
                "ds4: prompt length %d leaves no GLM context room (ctx %u)\n",
                prompt->len,
                g.ctx_size);
        glm_graph_free(&g);
        return 1;
    }
    const bool memory_report = getenv("DS4_METAL_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_gpu_print_memory_report("after GLM graph alloc");

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    bool ok = true;
    const bool seed_before_prefill =
        ssd_streaming &&
        !glm_graph_env_present("DS4_ROCM_GLM_DISABLE_STREAMING_SEED_BEFORE_PREFILL",
                               "DS4_METAL_GLM_DISABLE_STREAMING_SEED_BEFORE_PREFILL");
    const double t_prefill0 = now_sec();
    if (seed_before_prefill) {
        ds4_gpu_graph seed_graph;
        memset(&seed_graph, 0, sizeof(seed_graph));
        seed_graph.quality = quality;
        seed_graph.ssd_streaming = ssd_streaming;
        seed_graph.ssd_streaming_cold = ssd_streaming_cold;
        seed_graph.streaming_preload_experts = ssd_streaming_preload_experts;
        ok = metal_graph_seed_streaming_expert_cache_from_hotlist(&seed_graph,
                                                                  model,
                                                                  weights);
    }
    if (ok) {
        ok = glm_graph_prefill_range(&g,
                                     model,
                                     weights,
                                     prompt->v,
                                     0,
                                     (uint32_t)prompt->len,
                                     logits,
                                     progress,
                                     progress_ud,
                                     (uint32_t)prompt->len);
    }
    const double t_prefill1 = now_sec();
    if (memory_report) ds4_gpu_print_memory_report("after GLM prefill");
    if (!ok) {
        fprintf(stderr, "ds4: GLM prefill failed\n");
        free(logits);
        glm_graph_free(&g);
        return 1;
    }
#ifdef DS4_ROCM_BUILD
    /*
     * Decode is SSD-read bound, so the prefill expert headroom is worth more
     * as extra dynamic cache once prefill is done.  Opt out with
     * DS4_ROCM_GLM_STREAMING_GROW_CACHE_AFTER_PREFILL=0.
     */
    const char *grow_cache_env =
        getenv("DS4_ROCM_GLM_STREAMING_GROW_CACHE_AFTER_PREFILL");
    if (ssd_streaming &&
        ssd_streaming_cache_bytes != 0 &&
        ssd_streaming_prefill_headroom_bytes != 0 &&
        (grow_cache_env == NULL || glm_graph_env_truthy(grow_cache_env))) {
        uint64_t budget_bytes = 0;
        uint64_t per_expert_bytes = 0;
        if (ssd_streaming_cache_bytes <=
            UINT64_MAX - ssd_streaming_prefill_headroom_bytes) {
            budget_bytes =
                ssd_streaming_cache_bytes + ssd_streaming_prefill_headroom_bytes;
        }
        const uint32_t grown_budget =
            ds4_streaming_cache_experts_for_byte_budget(weights,
                                                        budget_bytes,
                                                        &per_expert_bytes);
        const uint32_t current_budget =
            ds4_gpu_stream_expert_cache_configured_count();
        if (grown_budget > current_budget) {
            ds4_gpu_set_streaming_expert_cache_budget(grown_budget);
            fprintf(stderr,
                    "ds4: ROCm GLM streaming expert cache grew after prefill: "
                    "%u -> %u experts (%.2f GiB)\n",
                    current_budget,
                    grown_budget,
                    (double)((uint64_t)grown_budget * per_expert_bytes) /
                        1073741824.0);
        }
    }
#else
    (void)ssd_streaming_cache_bytes;
    (void)ssd_streaming_prefill_headroom_bytes;
#endif

    int n_generated = 0;
    int n_decode_eval = 0;
    uint32_t pos = (uint32_t)prompt->len;
    const bool token_timing = getenv("DS4_TOKEN_TIMING") != NULL;
    const double t_decode0 = now_sec();
    for (int i = 0; i < n_predict && pos < g.ctx_size; i++) {
        if (getenv("DS4_TRACE_TOP") != NULL) {
            char label[64];
            snprintf(label, sizeof(label), "GLM step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }
        const int token = sample_argmax(logits, DS4_N_VOCAB);
        if (vocab_token_is_generation_stop(vocab, token)) break;
        if (emit) emit(emit_ud, token);
        n_generated++;

        if (i == n_predict - 1 || pos + 1u >= g.ctx_size) {
            pos++;
            break;
        }

        const double t_eval0 = token_timing ? now_sec() : 0.0;
        ok = glm_graph_forward_token(&g, model, weights, token, NULL, pos,
                                     NULL, logits, false);
        if (!ok) {
            fprintf(stderr, "ds4: GLM decode failed at position %u\n", pos);
            free(logits);
            glm_graph_free(&g);
            return 1;
        }
        if (token_timing) {
            const double t_eval1 = now_sec();
            fprintf(stderr,
                    "ds4: GLM decode eval %d took %.3f ms\n",
                    n_decode_eval + 1,
                    (t_eval1 - t_eval0) * 1000.0);
        }
        n_decode_eval++;
        pos++;
    }
    const double t_decode1 = now_sec();
    if (done) done(emit_ud);

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: GLM prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)n_generated / decode_s : 0.0);

    if (memory_report) ds4_gpu_print_memory_report("before GLM graph free");
    free(logits);
    glm_graph_free(&g);
    return 0;
}

#endif /* !DS4_NO_GPU */
