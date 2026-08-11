#ifndef DS4_NO_GPU
/* Execute graph-backend prefill in layer-major order so intermediate
 * activations stay on the GPU and cache state is built exactly once. */
static void gpu_graph_report_prefill_display_progress(
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                start,
        uint32_t                n_tokens,
        uint32_t                layer_done,
        int                     total) {
    if (!display_progress) return;
    if (layer_done > (uint32_t)DS4_N_LAYER) layer_done = (uint32_t)DS4_N_LAYER;
    uint64_t done = (uint64_t)n_tokens * layer_done / (uint32_t)DS4_N_LAYER;
    if (layer_done == (uint32_t)DS4_N_LAYER) done = n_tokens;
    display_progress(display_progress_ud, "prefill_display",
                     (int)(start + (uint32_t)done), total);
}

typedef struct {
    int tier;
    uint32_t first_layer;
    uint32_t end_layer;
} metal_graph_prefill_stage;

static bool metal_graph_build_prefill_stages(
        const ds4_gpu_graph          *g,
        metal_graph_prefill_stage    *stages,
        uint32_t                     *n_stages) {
    if (!g || !g->placement || !stages || !n_stages) return false;
    uint32_t ns = 0;
    int prev_tier = -1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const int tier = g->placement[il + 1];
        if (tier < 0 || tier >= DS4_MAX_GPUS) return false;
        if (il == 0 || tier != prev_tier) {
            if (ns >= DS4_MAX_GPUS) return false;
            stages[ns].tier = tier;
            stages[ns].first_layer = il;
            stages[ns].end_layer = il + 1u;
            ns++;
            prev_tier = tier;
        } else {
            stages[ns - 1u].end_layer = il + 1u;
        }
    }
    *n_stages = ns;
    return ns != 0;
}

static bool metal_graph_encode_prefill_stage_batch(
        ds4_gpu_graph              *g,
        const ds4_model            *model,
        const ds4_weights          *weights,
        const metal_graph_prefill_stage *stage,
        uint32_t                    pos0,
        uint32_t                    n_tokens) {
    if (!g || !model || !weights || !stage || n_tokens == 0) return false;
    if (!metal_graph_set_active_tier_no_copy(g, stage->tier)) return false;
    for (uint32_t il = stage->first_layer; il < stage->end_layer; il++) {
        if (g->placement && g->placement[il + 1] != stage->tier) return false;
        if (!metal_graph_encode_layer_batch(g,
                                            model,
                                            &weights->layer[il],
                                            il,
                                            pos0,
                                            n_tokens)) {
            return false;
        }
        if (g->pipeline_capture_chunk_len != 0 &&
            !metal_graph_dspark_capture_prefill_rows(
                    g, il,
                    g->pipeline_capture_chunk_start,
                    g->pipeline_capture_chunk_len,
                    pos0,
                    n_tokens)) {
            return false;
        }
    }
    return true;
}

static bool metal_graph_prefill_pipeline_stage_major(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud) {
    if (!g || !model || !weights || !prompt || !g->placement ||
        n_tokens == 0 || n_tokens > g->prefill_cap ||
        start > (uint32_t)prompt->len ||
        n_tokens > (uint32_t)prompt->len - start) {
        return false;
    }

    metal_graph_prefill_stage stages[DS4_MAX_GPUS];
    uint32_t n_stages = 0;
    g->pipeline_capture_chunk_start = start;
    g->pipeline_capture_chunk_len =
        g->dspark_capture_enabled ? n_tokens : 0;
    if (!metal_graph_build_prefill_stages(g, stages, &n_stages) || n_stages < 2) {
        return false;
    }
    if (stages[0].tier != g->emb_tier) {
        return false;
    }

    uint32_t mb_cap = metal_graph_cuda_prefill_pipeline_microbatch();
    if (mb_cap == 0 || mb_cap >= n_tokens) return false;
    if (mb_cap > g->prefill_cap) mb_cap = g->prefill_cap;
    const uint32_t n_mb = (n_tokens + mb_cap - 1u) / mb_cap;
    if (n_mb < 2) return false;

    if (display_progress)
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = true;
    const double t0 = getenv("DS4_METAL_GRAPH_PREFILL_PROFILE") ? now_sec() : 0.0;

    const bool sequential = getenv("DS4_CUDA_PREFILL_PIPELINE_SEQUENTIAL") != NULL;
    const bool suppress_q8_cache =
        !metal_graph_cuda_prefill_pipeline_q8_cache_requested();
    const int saved_q8_cache_suppressed = ds4_gpu_q8_cache_suppressed();
    if (suppress_q8_cache) ds4_gpu_set_q8_cache_suppressed(1);
    ok = ds4_gpu_begin_commands() != 0;
    if (sequential) {
        for (uint32_t mb_i = 0; ok && mb_i < n_mb; mb_i++) {
            const uint32_t mb_off = mb_i * mb_cap;
            uint32_t mb_len = n_tokens - mb_off;
            if (mb_len > mb_cap) mb_len = mb_cap;
            const uint32_t pos0 = start + mb_off;

            for (uint32_t stage_i = 0; ok && stage_i < n_stages; stage_i++) {
                g->batch_token_offset = mb_off;
                if (stage_i == 0) {
                    ok = metal_graph_set_active_tier_no_copy(g, stages[0].tier);
                    ds4_gpu_tensor *tokens_view = NULL;
                    if (ok) {
                        tokens_view = ds4_gpu_tensor_view(metal_graph_prefill_tokens(g),
                                                           (uint64_t)mb_off * sizeof(int32_t),
                                                           (uint64_t)mb_len * sizeof(int32_t));
                        ok = tokens_view != NULL;
                    }
                    if (ok) {
                        ok = metal_graph_upload_prompt_embeddings_hc(
                                g->batch_cur_hc_by_tier[stages[0].tier],
                                tokens_view,
                                model,
                                weights,
                                prompt,
                                pos0,
                                mb_len);
                    }
                    ds4_gpu_tensor_free(tokens_view);
                }
                if (ok) {
                    ok = metal_graph_encode_prefill_stage_batch(g,
                                                                model,
                                                                weights,
                                                                &stages[stage_i],
                                                                pos0,
                                                                mb_len);
                }
                if (ok && stage_i + 1u < n_stages) {
                    ds4_gpu_tensor *src = g->batch_cur_hc_by_tier[stages[stage_i].tier];
                    ds4_gpu_tensor *dst = g->batch_cur_hc_by_tier[stages[stage_i + 1u].tier];
                    if (ok && getenv("DS4_CUDA_PREFILL_PIPELINE_SYNC_BOUNDARY") != NULL) {
                        ok = metal_graph_set_active_tier_no_copy(g, stages[stage_i].tier) &&
                             ds4_gpu_synchronize() != 0;
                    }
                    ok = src && dst &&
                         ds4_gpu_tensor_copy_xdev_ordered(dst,
                                                          src,
                                                          (uint64_t)mb_len * hc_dim * sizeof(float)) != 0;
                }
            }
            if (ok) ok = ds4_gpu_end_commands() != 0;
            else (void)ds4_gpu_synchronize();
            if (ok && display_progress) {
                uint32_t done = mb_off + mb_len;
                if (done > n_tokens) done = n_tokens;
                display_progress(display_progress_ud,
                                 "prefill_display",
                                 (int)(start + done),
                                 prompt->len);
            }
            if (show_progress) {
                fprintf(stderr, "ds4: gpu sequential pipeline prefill microbatch %u/%u\r",
                        mb_i + 1u,
                        n_mb);
                fflush(stderr);
            }
            if (ok && mb_i + 1u < n_mb) ok = ds4_gpu_begin_commands() != 0;
        }
    } else {
        for (uint32_t wave = 0; ok && wave < n_mb + n_stages - 1u; wave++) {
            uint32_t smax = wave < n_stages ? wave : n_stages - 1u;
            for (int si = (int)smax; ok && si >= 0; si--) {
                const uint32_t stage_i = (uint32_t)si;
                const uint32_t mb_i = wave - stage_i;
                if (mb_i >= n_mb) continue;
                const uint32_t mb_off = mb_i * mb_cap;
                uint32_t mb_len = n_tokens - mb_off;
                if (mb_len > mb_cap) mb_len = mb_cap;
                const uint32_t pos0 = start + mb_off;

                g->batch_token_offset = mb_off;
                if (stage_i == 0) {
                    ok = metal_graph_set_active_tier_no_copy(g, stages[0].tier);
                    ds4_gpu_tensor *tokens_view = NULL;
                    if (ok) {
                        tokens_view = ds4_gpu_tensor_view(metal_graph_prefill_tokens(g),
                                                           (uint64_t)mb_off * sizeof(int32_t),
                                                           (uint64_t)mb_len * sizeof(int32_t));
                        ok = tokens_view != NULL;
                    }
                    if (ok) {
                        ok = metal_graph_upload_prompt_embeddings_hc(
                                g->batch_cur_hc_by_tier[stages[0].tier],
                                tokens_view,
                                model,
                                weights,
                                prompt,
                                pos0,
                                mb_len);
                    }
                    ds4_gpu_tensor_free(tokens_view);
                }

                if (ok) {
                    ok = metal_graph_encode_prefill_stage_batch(g,
                                                                model,
                                                                weights,
                                                                &stages[stage_i],
                                                                pos0,
                                                                mb_len);
                }
                if (ok && stage_i + 1u < n_stages) {
                    ds4_gpu_tensor *src = g->batch_cur_hc_by_tier[stages[stage_i].tier];
                    ds4_gpu_tensor *dst = g->batch_cur_hc_by_tier[stages[stage_i + 1u].tier];
                    if (ok && getenv("DS4_CUDA_PREFILL_PIPELINE_SYNC_BOUNDARY") != NULL) {
                        ok = metal_graph_set_active_tier_no_copy(g, stages[stage_i].tier) &&
                             ds4_gpu_synchronize() != 0;
                    }
                    ok = src && dst &&
                         ds4_gpu_tensor_copy_xdev_ordered(dst,
                                                          src,
                                                          (uint64_t)mb_len * hc_dim * sizeof(float)) != 0;
                }
                if (ok && display_progress && stage_i + 1u == n_stages) {
                    uint32_t done = mb_off + mb_len;
                    if (done > n_tokens) done = n_tokens;
                    display_progress(display_progress_ud,
                                     "prefill_display",
                                     (int)(start + done),
                                     prompt->len);
                }
            }
            if (show_progress) {
                fprintf(stderr, "ds4: gpu pipeline prefill wave %u/%u\r",
                        wave + 1u,
                        n_mb + n_stages - 1u);
                fflush(stderr);
            }
        }
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
    }
    if (show_progress) fputc('\n', stderr);
    g->batch_token_offset = 0;
    if (!ok) {
        if (suppress_q8_cache) ds4_gpu_set_q8_cache_suppressed(saved_q8_cache_suppressed);
        return false;
    }

    const uint32_t final_len = n_tokens - (n_mb - 1u) * mb_cap;
    const int src_tier = stages[n_stages - 1u].tier;
    if (!metal_graph_set_active_tier_no_copy(g, src_tier)) {
        if (suppress_q8_cache) ds4_gpu_set_q8_cache_suppressed(saved_q8_cache_suppressed);
        return false;
    }

    ds4_gpu_tensor *saved_cur = g->cur_hc_by_tier[src_tier];
    ds4_gpu_tensor *last_hc = NULL;
    if (logits) {
        last_hc = metal_graph_tensor_row_view(g->batch_cur_hc_by_tier[src_tier],
                                              final_len - 1u,
                                              hc_dim);
        ok = last_hc != NULL;
    }
    if (ok && logits) {
        g->cur_hc_by_tier[src_tier] = last_hc;
        ok = ds4_gpu_begin_commands() != 0;
    }
    if (ok && logits) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    if (ok && logits) ok = ds4_gpu_end_commands() != 0;
    else if (!ok) (void)ds4_gpu_synchronize();
    g->cur_hc_by_tier[src_tier] = saved_cur;
    ds4_gpu_tensor_free(last_hc);
    if (g->placement && g->active_tier != src_tier) {
        ok = metal_graph_set_active_tier_no_copy(g, src_tier);
    }
    if (ok && logits) {
        ok = ds4_gpu_tensor_read(metal_graph_logits(g),
                                 0,
                                 logits,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok && display_progress)
        display_progress(display_progress_ud, "prefill_display",
                         (int)(start + n_tokens), prompt->len);
    if (ok && t0 != 0.0) {
        const double t1 = now_sec();
        fprintf(stderr,
                "ds4: gpu pipeline prefill total tokens=%u stages=%u mb=%u total=%.3f ms\n",
                n_tokens,
                n_stages,
                mb_cap,
                (t1 - t0) * 1000.0);
    }
    if (suppress_q8_cache) ds4_gpu_set_q8_cache_suppressed(saved_q8_cache_suppressed);
    return ok;
}

static bool metal_graph_prefill_layer_major(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_imatrix_collector *imatrix,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;

    if (display_progress)
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);

    bool ok = metal_graph_upload_prompt_tokens(metal_graph_prefill_tokens(g), prompt, start, n_tokens);
    if (!ok) return false;

#ifdef DS4_ROCM_BUILD
    if (g->ssd_streaming &&
        DS4_MODEL_VARIANT == DS4_VARIANT_PRO &&
        n_tokens >= 1024u) {
        ds4_gpu_stream_expert_cache_release_resident();
    }
#endif

    if (!metal_graph_warmup_prefill_kernels(g, model, weights, n_tokens)) return false;
    if (g->placement &&
        !metal_graph_set_active_tier_no_copy(g, g->emb_tier)) {
        return false;
    }
    metal_graph_dspark_capture_begin_prefill(g);

    const bool split_profile =
        glm_graph_env_present("DS4_ROCM_GRAPH_PREFILL_SPLIT_PROFILE",
                              "DS4_METAL_GRAPH_PREFILL_SPLIT_PROFILE");
    /*
     * A full long-prompt prefill can keep the GPU busy for a long time. Split
     * non-tiny prefills when a frontend asked for display progress: completed
     * layer command buffers are real scheduling/keepalive points, while
     * callbacks emitted while encoding one huge command buffer would only be
     * cosmetic.
     */
    const bool throttle = graph_power_throttle_enabled(g);
    const bool callback_split = display_progress != NULL && n_tokens >= 32;
    const bool split_commands = g->ssd_streaming ||
                                split_profile || throttle || callback_split ||
                                n_tokens > 2048 || imatrix != NULL;
    const bool profile =
        glm_graph_env_present("DS4_ROCM_GRAPH_PREFILL_PROFILE",
                              "DS4_METAL_GRAPH_PREFILL_PROFILE") ||
        split_profile;
    const double t0 = profile ? now_sec() : 0.0;
    double encode_s = 0.0;
    double execute_s = 0.0;

    const uint32_t pipeline_mb = metal_graph_cuda_prefill_pipeline_microbatch();
    if (!split_commands &&
        !profile &&
        imatrix == NULL &&
        metal_graph_cuda_prefill_pipeline_requested(g) &&
        pipeline_mb != 0 &&
        pipeline_mb < n_tokens) {
        return metal_graph_prefill_pipeline_stage_major(g,
                                                        model,
                                                        weights,
                                                        prompt,
                                                        start,
                                                        n_tokens,
                                                        logits,
                                                        show_progress,
                                                        display_progress,
                                                        display_progress_ud);
    }

    if (!split_commands) {
        ok = metal_graph_upload_prompt_embeddings_hc(metal_graph_batch_cur_hc(g),
                                                     metal_graph_prefill_tokens(g),
                                                     model,
                                                     weights,
                                                     prompt,
                                                     start,
                                                     n_tokens);
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = metal_graph_encode_layer_batch(g,
                                                model,
                                                &weights->layer[il],
                                                il,
                                                start,
                                                n_tokens);
            if (!ok) {
                fprintf(stderr, "ds4: gpu whole-prefill layer %u encode failed\n", il);
            }
            if (ok) ok = metal_graph_dspark_capture_prefill_layer(g,
                                                                  il,
                                                                  start,
                                                                  n_tokens);
            if (show_progress) {
                fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
                fflush(stderr);
            }
        }
        if (show_progress) fputc('\n', stderr);
        if (display_progress)
            display_progress(display_progress_ud, "prefill_display",
                             (int)(start + n_tokens), prompt->len);

        const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
        uint32_t output_row = (uint32_t)n_tokens - 1u;
        const char *output_row_env = glm_graph_env_value(
                "DS4_ROCM_GRAPH_OUTPUT_ROW",
                "DS4_METAL_GRAPH_OUTPUT_ROW");
        if (output_row_env && output_row_env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(output_row_env, &end, 10);
            if (end != output_row_env && v < (unsigned long)n_tokens) {
                output_row = (uint32_t)v;
            }
        }
        const int src_tier = g->active_tier;
        ds4_gpu_tensor *saved_cur = g->cur_hc_by_tier[src_tier];
        ds4_gpu_tensor *last_hc = NULL;
        if (ok && logits) {
            last_hc = metal_graph_tensor_row_view(metal_graph_batch_cur_hc(g), output_row, hc_dim);
            ok = last_hc != NULL;
        }
        if (ok && logits) {
            g->cur_hc_by_tier[src_tier] = last_hc;
            ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
            g->cur_hc_by_tier[src_tier] = saved_cur;
        }

        const double t_encoded = profile ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double t_done = profile ? now_sec() : 0.0;
        g->cur_hc_by_tier[src_tier] = saved_cur;
        if (last_hc) ds4_gpu_tensor_free(last_hc);
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after whole-prefill graph failure also failed\n");
            }
            return false;
        }
#ifdef __APPLE__
        ds4_gpu_release_zero_prefix_prefill_mask_cache();
#endif

        const double t_before_read = profile ? now_sec() : 0.0;
        if (logits) {
            ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (profile) {
            const double t_read = now_sec();
            fprintf(stderr,
                    "ds4: gpu graph prefill total tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                    n_tokens,
                    (t_encoded - t0) * 1000.0,
                    (t_done - t_encoded) * 1000.0,
                    (t_read - t_before_read) * 1000.0,
                    (t_read - t0) * 1000.0);
        }
        return ok;
    }

    if (g->ssd_streaming) {
        g->streaming_static_decode_map_current = false;
        if (!metal_graph_stream_map_token(model, weights)) return false;
    }
    metal_graph_stream_prefill_selected_profile_reset(g);
    metal_graph_stream_prepare_slot layer_prepare_slots[DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD];
    memset(layer_prepare_slots, 0, sizeof(layer_prepare_slots));
    const bool layer_pagein =
        metal_graph_stream_prefill_layer_pagein_enabled(g);
    const bool layer_readahead =
        !layer_pagein &&
        metal_graph_stream_prefill_layer_readahead_enabled(g);
    const bool layer_pread =
        !layer_pagein && !layer_readahead &&
        metal_graph_stream_prefill_layer_pread_enabled(g);
    const bool layer_madvise =
        !layer_pagein && !layer_pread && !layer_readahead &&
        metal_graph_stream_prefill_layer_madvise_enabled(g);
    const bool layer_prepare =
        layer_pagein || layer_pread || layer_readahead || layer_madvise;
    const bool layer_prepare_overlap =
        layer_prepare && metal_graph_stream_prefill_layer_pagein_overlap_enabled();
    const uint32_t layer_prepare_ahead =
        layer_prepare && layer_prepare_overlap ?
        metal_graph_stream_prefill_layer_prepare_ahead() : 1u;
    const bool batch_selected_addr =
        metal_graph_stream_prefill_batch_selected_addr_enabled(g, weights, n_tokens) ||
        metal_graph_cuda_stream_prefill_batch_selected_addr_enabled(g, weights, n_tokens);
#ifdef DS4_ROCM_BUILD
    rocm_graph_stream_layer_expert_load rocm_full_layer_load;
    memset(&rocm_full_layer_load, 0, sizeof(rocm_full_layer_load));
#endif
    if (g->ssd_streaming && DS4_N_LAYER > 0) {
        const bool layer_selected_addr =
            batch_selected_addr &&
            metal_graph_stream_prefill_batch_selected_addr_layer_supported(weights, 0);
        if (layer_prepare) {
            if (!metal_graph_stream_prepare_start_if_needed(g,
                                                            model,
                                                            weights,
                                                            0,
                                                            n_tokens,
                                                            layer_madvise,
                                                            layer_pread,
                                                            layer_readahead,
                                                            layer_selected_addr,
                                                            layer_prepare_slots,
                                                            layer_prepare_ahead)) {
                return false;
            }
        } else {
            if (layer_selected_addr) {
                metal_graph_stream_readahead_layer_decode(model, weights, 0);
            } else {
                metal_graph_stream_readahead_layer(model, weights, 0);
            }
        }
    }
#ifdef DS4_ROCM_BUILD
    if (g->ssd_streaming && DS4_N_LAYER > 0 &&
        !rocm_graph_stream_layer_expert_load_start_next(&rocm_full_layer_load,
                                                        g,
                                                        model,
                                                        weights,
                                                        0,
                                                        n_tokens)) {
        return false;
    }
#endif

    double t_layer0 = (profile || throttle) ? now_sec() : 0.0;
    ok = metal_graph_upload_prompt_embeddings_hc(metal_graph_batch_cur_hc(g),
                                                 metal_graph_prefill_tokens(g),
                                                 model,
                                                 weights,
                                                 prompt,
                                                 start,
                                                 n_tokens);
    const double t_embed_encoded = (profile || throttle) ? now_sec() : 0.0;
    const double t_embed_done = (profile || throttle) ? now_sec() : 0.0;
    if (profile) {
        encode_s += t_embed_encoded - t_layer0;
        execute_s += t_embed_done - t_embed_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: metal layer-major prefill embed encode=%.3f ms execute=%.3f ms\n",
                    (t_embed_encoded - t_layer0) * 1000.0,
                    (t_embed_done - t_embed_encoded) * 1000.0);
        }
    }
    if (!ok) {
#ifdef DS4_ROCM_BUILD
        (void)rocm_graph_stream_layer_expert_load_join(&rocm_full_layer_load);
        (void)ds4_gpu_stream_expert_cache_release_layer_cache();
#endif
        if (layer_prepare) {
            (void)metal_graph_stream_prepare_join_all(layer_prepare_slots,
                                                      layer_prepare_ahead);
        }
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after layer-major prefill embed failure also failed\n");
        }
        return false;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        double layer_elapsed = 0.0;
        const bool layer_selected_addr =
            batch_selected_addr &&
            metal_graph_stream_prefill_batch_selected_addr_layer_supported(weights, il);
        if (layer_prepare &&
            !metal_graph_stream_prepare_join_layer(g,
                                                   model,
                                                   weights,
                                                   il,
                                                   n_tokens,
                                                   layer_madvise,
                                                   layer_pread,
                                                   layer_readahead,
                                                   layer_selected_addr,
                                                   layer_prepare_slots,
                                                   layer_prepare_ahead)) {
            ok = false;
            break;
        }
#ifdef DS4_ROCM_BUILD
        const bool rocm_full_layer_stream_prefill =
            rocm_graph_stream_prefill_full_layer_enabled(g,
                                                         &weights->layer[il],
                                                         il,
                                                         n_tokens);
        if (rocm_full_layer_stream_prefill &&
            !rocm_graph_stream_layer_expert_load_ready(&rocm_full_layer_load,
                                                       g,
                                                       model,
                                                       weights,
                                                       il,
                                                       n_tokens)) {
            ok = false;
            break;
        }
        if (rocm_full_layer_stream_prefill &&
            !rocm_graph_stream_layer_expert_load_start_next(&rocm_full_layer_load,
                                                            g,
                                                            model,
                                                            weights,
                                                            il + 1u,
                                                            n_tokens)) {
            ok = false;
            break;
        }
#endif
        if (g->ssd_streaming) {
            g->streaming_static_decode_map_current = false;
            bool decode_only_map = layer_selected_addr;
#ifdef DS4_ROCM_BUILD
            decode_only_map = decode_only_map || rocm_full_layer_stream_prefill;
#endif
            const bool map_ok = decode_only_map ?
                metal_graph_stream_map_layer_decode(model, weights, il) :
                metal_graph_stream_map_layer(model, weights, il);
            if (!map_ok) {
                ok = false;
                break;
            }
        }
        if (g->ssd_streaming) {
            if (layer_prepare && layer_prepare_overlap) {
                bool started_future = false;
                for (uint32_t ahead = 1; ahead <= layer_prepare_ahead; ahead++) {
                    if (il + ahead >= DS4_N_LAYER) break;
                    started_future = true;
                    const uint32_t future_il = il + ahead;
                    const bool future_selected_addr =
                        batch_selected_addr &&
                        metal_graph_stream_prefill_batch_selected_addr_layer_supported(
                            weights,
                            future_il);
                    if (!metal_graph_stream_prepare_start_if_needed(g,
                                                                    model,
                                                                    weights,
                                                                    future_il,
                                                                    n_tokens,
                                                                    layer_madvise,
                                                                    layer_pread,
                                                                    layer_readahead,
                                                                    future_selected_addr,
                                                                    layer_prepare_slots,
                                                                    layer_prepare_ahead)) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) break;
                if (!started_future && logits) {
                    metal_graph_stream_readahead_output(model, weights);
                }
            } else if (!layer_prepare && il + 1 < DS4_N_LAYER) {
                const bool next_selected_addr =
                    batch_selected_addr &&
                    metal_graph_stream_prefill_batch_selected_addr_layer_supported(
                        weights,
                        il + 1);
                if (next_selected_addr) {
                    metal_graph_stream_readahead_layer_decode(model, weights, il + 1);
                } else {
                    metal_graph_stream_readahead_layer(model, weights, il + 1);
                }
            } else if (logits) {
                metal_graph_stream_readahead_output(model, weights);
            }
        }
        if (split_profile) {
            /* (B6 fix): split-profile diagnostic bypasses the
             * metal_graph_encode_layer_batch wrapper that normally does
             * the per-layer tier switch. Replicate the switch here so the
             * diagnostic / profile mode stays multi-tier-correct.
             * Single-tier (g->placement == NULL): no-op. */
            if (g->placement) {
                const int this_tier = g->placement[il + 1];
                if (!metal_graph_set_active_tier_batch(g, this_tier, (uint32_t)n_tokens)) {
                    ok = false;
                    break;
                }
            }
            const double t_attn0 = now_sec();
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_layer_attention_batch(g,
                                                                  model,
                                                                  &weights->layer[il],
                                                                  il,
                                                                  start,
                                                                  n_tokens);
            if (!ok) {
                fprintf(stderr, "ds4: gpu layer-major prefill layer %u attention encode failed\n", il);
            }
            const double t_attn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_attn_done = now_sec();

            const double t_ffn0 = now_sec();
            if (ok) ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_layer_ffn_batch(g,
                                                            model,
                                                            &weights->layer[il],
                                                            il,
                                                            start,
                                                            n_tokens,
                                                            NULL,
                                                            0);
            if (!ok) {
                fprintf(stderr, "ds4: gpu layer-major prefill layer %u ffn encode failed\n", il);
            }
            if (ok) {
                ds4_gpu_tensor *tmp = metal_graph_batch_cur_hc(g);
                g->batch_cur_hc_by_tier[g->active_tier] = metal_graph_batch_next_hc(g);
                g->batch_next_hc_by_tier[g->active_tier] = tmp;
            }
            if (ok) ok = metal_graph_dspark_capture_prefill_layer(g,
                                                                  il,
                                                                  start,
                                                                  n_tokens);
            if (ok) ok = metal_graph_capture_prefill_seed_router_selected(g,
                                                                          il,
                                                                          n_tokens);
#ifdef __APPLE__
            if (ok && !layer_selected_addr) {
                ok = metal_graph_seed_streaming_expert_cache_layer_from_mapped_hotlist(
                        g,
                        model,
                        weights,
                        il);
            }
#endif
            const double t_ffn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_ffn_done = now_sec();
#ifdef DS4_ROCM_BUILD
            if (ok) {
                ok = rocm_graph_stream_seed_full_layer_selected(g,
                                                                model,
                                                                &weights->layer[il],
                                                                il,
                                                                n_tokens);
            }
#endif
            if (ok) {
                ok = metal_graph_stream_prefill_selected_profile_layer(
                        g,
                        &weights->layer[il],
                        il,
                        n_tokens);
            }
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);
            layer_elapsed = (t_attn_done - t_attn0) + (t_ffn_done - t_ffn0);

            encode_s += (t_attn_encoded - t_attn0) + (t_ffn_encoded - t_ffn0);
            execute_s += (t_attn_done - t_attn_encoded) + (t_ffn_done - t_ffn_encoded);
            fprintf(stderr,
                    "ds4: metal layer-major prefill layer %u attn encode=%.3f execute=%.3f ms ffn encode=%.3f execute=%.3f ms\n",
                    il,
                    (t_attn_encoded - t_attn0) * 1000.0,
                    (t_attn_done - t_attn_encoded) * 1000.0,
                    (t_ffn_encoded - t_ffn0) * 1000.0,
                    (t_ffn_done - t_ffn_encoded) * 1000.0);
        } else {
            const double t_chunk0 = (profile || throttle) ? now_sec() : 0.0;
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_layer_batch(g,
                                                        model,
                                                        &weights->layer[il],
                                                        il,
                                                        start,
                                                        n_tokens);
            if (!ok) {
                fprintf(stderr, "ds4: gpu layer-major prefill layer %u encode failed\n", il);
            }
            if (ok) ok = metal_graph_dspark_capture_prefill_layer(g,
                                                                  il,
                                                                  start,
                                                                  n_tokens);
            if (ok) ok = metal_graph_capture_prefill_seed_router_selected(g,
                                                                          il,
                                                                          n_tokens);
#ifdef __APPLE__
            if (ok && !layer_selected_addr) {
                ok = metal_graph_seed_streaming_expert_cache_layer_from_mapped_hotlist(
                        g,
                        model,
                        weights,
                        il);
            }
#endif
            const double t_encoded = (profile || throttle) ? now_sec() : 0.0;
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_done = (profile || throttle) ? now_sec() : 0.0;
#ifdef DS4_ROCM_BUILD
            if (ok) {
                ok = rocm_graph_stream_seed_full_layer_selected(g,
                                                                model,
                                                                &weights->layer[il],
                                                                il,
                                                                n_tokens);
            }
#endif
            if (ok) {
                ok = metal_graph_stream_prefill_selected_profile_layer(
                        g,
                        &weights->layer[il],
                        il,
                        n_tokens);
            }
            if (ok && imatrix) ok = imatrix_collect_layer_batch(imatrix, g, il, (uint32_t)n_tokens);
            layer_elapsed = t_done - t_chunk0;
            if (profile) {
                encode_s += t_encoded - t_chunk0;
                execute_s += t_done - t_encoded;
                fprintf(stderr,
                        "ds4: gpu layer-major prefill layer %u encode=%.3f ms execute=%.3f ms\n",
                        il,
                        (t_encoded - t_chunk0) * 1000.0,
                        (t_done - t_encoded) * 1000.0);
            }
        }
        if (ok &&
            g->ssd_streaming &&
            layer_prepare &&
            !layer_prepare_overlap) {
            if (il + 1 < DS4_N_LAYER) {
                const bool next_selected_addr =
                    batch_selected_addr &&
                    metal_graph_stream_prefill_batch_selected_addr_layer_supported(
                        weights,
                        il + 1);
                if (!metal_graph_stream_prepare_start_if_needed(g,
                                                                model,
                                                                weights,
                                                                il + 1,
                                                                n_tokens,
                                                                layer_madvise,
                                                                layer_pread,
                                                                layer_readahead,
                                                                next_selected_addr,
                                                                layer_prepare_slots,
                                                                layer_prepare_ahead)) {
                    ok = false;
                }
            } else if (logits) {
                metal_graph_stream_readahead_output(model, weights);
            }
        }
        if (!ok) {
#ifdef DS4_ROCM_BUILD
            (void)rocm_graph_stream_layer_expert_load_join(&rocm_full_layer_load);
            (void)ds4_gpu_stream_expert_cache_release_layer_cache();
#endif
            if (layer_prepare) {
                (void)metal_graph_stream_prepare_join_all(layer_prepare_slots,
                                                          layer_prepare_ahead);
            }
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after layer-major prefill failure also failed\n");
            }
            return false;
        }
        graph_power_note_prefill_layer(g, il, layer_elapsed);
        gpu_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  start,
                                                  n_tokens,
                                                  il + 1,
                                                  prompt->len);
        if (show_progress) {
            fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
            fflush(stderr);
        }
    }
    if (!ok) {
#ifdef DS4_ROCM_BUILD
        (void)rocm_graph_stream_layer_expert_load_join(&rocm_full_layer_load);
        (void)ds4_gpu_stream_expert_cache_release_layer_cache();
#endif
        if (layer_prepare) {
            (void)metal_graph_stream_prepare_join_all(layer_prepare_slots,
                                                      layer_prepare_ahead);
        }
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after layer-major prefill failure also failed\n");
        }
        return false;
    }
#ifdef __APPLE__
    /* Zero-prefix masks are shared across the 43 per-layer command batches,
     * then become dead weight. Release them before the output head and later
     * replay/decode chunks so the prefill win does not add residency pressure. */
    ds4_gpu_release_zero_prefix_prefill_mask_cache();
#endif
    if (show_progress) fputc('\n', stderr);
    metal_graph_stream_prefill_selected_profile_summary(g);
#ifdef DS4_ROCM_BUILD
    /*
     * The final layer can use its fully mapped prefill table after the
     * asynchronous selected-expert loader has already queued a resident-cache
     * batch.  Drain that completed batch before hotlist seeding starts another
     * read-pool job set, otherwise both callers wait forever for ownership of
     * the single ROCm streaming read pool.
     */
    if (g->ssd_streaming &&
        !ds4_gpu_stream_expert_cache_finish_pending_batch()) {
        return false;
    }
    (void)ds4_gpu_stream_expert_cache_release_layer_cache();
    if (g->ssd_streaming) ds4_gpu_release_q8_f16_cache();
#endif
    if (!metal_graph_seed_streaming_expert_cache_from_hotlist(g, model, weights)) {
        return false;
    }
    if (!metal_graph_seed_streaming_expert_cache_from_prefill(g, model, weights)) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    uint32_t output_row = (uint32_t)n_tokens - 1u;
    const char *output_row_env = glm_graph_env_value(
            "DS4_ROCM_GRAPH_OUTPUT_ROW",
            "DS4_METAL_GRAPH_OUTPUT_ROW");
    if (output_row_env && output_row_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(output_row_env, &end, 10);
        if (end != output_row_env && v < (unsigned long)n_tokens) {
            output_row = (uint32_t)v;
        }
    }
    ds4_gpu_tensor *saved_cur = metal_graph_cur_hc(g);
    ds4_gpu_tensor *last_hc = NULL;

    const double t_head0 = profile ? now_sec() : 0.0;
    if (logits) {
        last_hc = metal_graph_tensor_row_view(metal_graph_batch_cur_hc(g),
                                              output_row,
                                              hc_dim);
        ok = last_hc != NULL;
    }
    if (ok && logits && g->ssd_streaming) {
        const bool static_decode_map =
            metal_graph_stream_decode_static_map_enabled();
        const bool static_map_state_cache =
            static_decode_map &&
            metal_graph_stream_decode_static_map_state_cache_enabled();
        g->streaming_static_decode_map_current = false;
        if (static_map_state_cache) {
            ok = metal_graph_stream_map_decode_static_all(model, weights);
            if (ok) g->streaming_static_decode_map_current = true;
        } else {
            ok = metal_graph_stream_map_output(model, weights);
        }
    }
    if (ok && logits) {
        g->cur_hc_by_tier[g->active_tier] = last_hc;
        ok = ds4_gpu_begin_commands() != 0;
    }
    if (ok && logits) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    const double t_head_encoded = profile ? now_sec() : 0.0;
    if (ok && logits) ok = ds4_gpu_end_commands() != 0;
    const double t_head_done = profile ? now_sec() : 0.0;
    g->cur_hc_by_tier[g->active_tier] = saved_cur;
    if (last_hc) ds4_gpu_tensor_free(last_hc);
    if (!ok) return false;

    const double t_before_read = profile ? now_sec() : 0.0;
    if (logits) {
        ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = now_sec();
        encode_s += t_head_encoded - t_head0;
        execute_s += t_head_done - t_head_encoded;
        if (split_profile) {
            fprintf(stderr,
                    "ds4: gpu layer-major prefill head encode=%.3f ms execute=%.3f ms\n",
                    (t_head_encoded - t_head0) * 1000.0,
                    (t_head_done - t_head_encoded) * 1000.0);
        }
        fprintf(stderr,
                "ds4: gpu layer-major prefill total tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                n_tokens,
                encode_s * 1000.0,
                execute_s * 1000.0,
                (t_read - t_before_read) * 1000.0,
                (t_read - t0) * 1000.0);
    }
    return ok;
}

static bool metal_graph_prefill_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (n_tokens <= 0 || n_tokens > prompt->len) return false;
    if ((uint32_t)n_tokens > g->prefill_cap) return false;
    if (metal_graph_use_streaming_decode_prefill_range(g, weights, 0,
                                                       (uint32_t)n_tokens)) {
        return metal_graph_prefill_decode_streaming_range(g,
                                                          model,
                                                          weights,
                                                          prompt,
                                                          0,
                                                          (uint32_t)n_tokens,
                                                          logits,
                                                          show_progress,
                                                          NULL,
                                                          NULL,
                                                          display_progress,
                                                          display_progress_ud,
                                                          cancel,
                                                          cancel_ud,
                                                          cancelled);
    }
    /* The layer-major fallback below may submit the whole short prefill as one
     * Metal command buffer.  Once that command is in flight there is no useful
     * safe prefix to expose: by the time cancellation can be observed again,
     * the prompt has already been fully read and the KV is valid.  Let the
     * caller observe the pending interrupt at generation time instead. */
    (void)cancel;
    (void)cancel_ud;
    (void)cancelled;
    return metal_graph_prefill_layer_major(g,
                                           model,
                                           weights,
                                           prompt,
                                           0,
                                           (uint32_t)n_tokens,
                                           logits,
                                           show_progress,
                                           NULL,
                                           display_progress,
                                           display_progress_ud);
}

/* Prefill a contiguous token range in fixed-size chunks.
 *
 * The common case starts at token zero, but server sessions also use this to
 * extend an existing KV cache with a long suffix.  Resumed chunks are aligned
 * to the same absolute prefill-cap boundaries used by a cold full prompt, so
 * compression windows and row finalization follow the same schedule after the
 * cached prefix.
 */
static bool metal_graph_prefill_chunked_range(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_imatrix_collector *imatrix,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (n_tokens == 0 || g->prefill_cap == 0) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;
    if (g->ssd_streaming && start == 0) {
        ds4_gpu_stream_expert_cache_reset_route_hotness();
    }
    if (!imatrix &&
        metal_graph_use_streaming_decode_prefill_range(g, weights,
                                                       start, n_tokens)) {
        return metal_graph_prefill_decode_streaming_range(g,
                                                          model,
                                                          weights,
                                                          prompt,
                                                          start,
                                                          n_tokens,
                                                          logits,
                                                          show_progress,
                                                          progress,
                                                          progress_ud,
                                                          display_progress,
                                                          display_progress_ud,
                                                          cancel,
                                                          cancel_ud,
                                                          cancelled);
    }

    uint32_t chunk_cap = g->prefill_cap;
    if (start != 0 && chunk_cap > g->raw_cap) chunk_cap = g->raw_cap;
    if (chunk_cap == 0) return false;

    const bool profile =
        glm_graph_env_present("DS4_ROCM_GRAPH_PREFILL_PROFILE",
                              "DS4_METAL_GRAPH_PREFILL_PROFILE");
    const double t0 = profile ? now_sec() : 0.0;
    const uint32_t end = start + n_tokens;

    if (progress) {
        progress(progress_ud, "prefill_chunk", (int)start, prompt->len);
    }
    if (display_progress) {
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);
    }

    for (uint32_t pos0 = start; pos0 < end; ) {
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        const uint32_t remaining = end - pos0;
        uint32_t local_cap = chunk_cap;
        if (start != 0 && g->prefill_cap != 0) {
            const uint32_t mod = pos0 % g->prefill_cap;
            if (mod != 0) {
                const uint32_t to_boundary = g->prefill_cap - mod;
                if (to_boundary < local_cap) local_cap = to_boundary;
            }
        }
        const uint32_t chunk = remaining < local_cap ? remaining : local_cap;
        const uint32_t chunk_end = pos0 + chunk;
        float *chunk_logits = (progress || chunk_end == end) ? logits : NULL;
        bool ok = metal_graph_prefill_layer_major(g,
                                                  model,
                                                  weights,
                                                  prompt,
                                                  pos0,
                                                  chunk,
                                                  chunk_logits,
                                                  show_progress,
                                                  imatrix,
                                                  display_progress,
                                                  display_progress_ud);
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after chunked prefill failure also failed\n");
            }
            return false;
        }
        if (progress) {
            progress(progress_ud, "prefill_chunk", (int)chunk_end, prompt->len);
        }
        if (display_progress) {
            display_progress(display_progress_ud, "prefill_display", (int)chunk_end, prompt->len);
        }
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        pos0 = chunk_end;
    }
    if (show_progress) fputc('\n', stderr);
    if (profile) {
        const double t_read = now_sec();
        fprintf(stderr,
                "ds4: gpu chunked prefill start=%u tokens=%u chunk=%u total=%.3f ms\n",
                start,
                n_tokens,
                chunk_cap,
                (t_read - t0) * 1000.0);
    }
    return true;
}

#endif /* !DS4_NO_GPU */
