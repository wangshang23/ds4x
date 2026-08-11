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

static bool metal_graph_prefill_layer_major(
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
    if (!g || !model || !weights || !prompt ||
        n_tokens == 0 || n_tokens > g->prefill_cap ||
        start > (uint32_t)prompt->len ||
        n_tokens > (uint32_t)prompt->len - start) {
        return false;
    }

    if (display_progress) {
        display_progress(display_progress_ud, "prefill_display", (int)start,
                         prompt->len);
    }
    if (!metal_graph_upload_prompt_tokens(metal_graph_prefill_tokens(g),
                                          prompt, start, n_tokens) ||
        !metal_graph_warmup_prefill_kernels(g, model, weights, n_tokens)) {
        return false;
    }
    metal_graph_dspark_capture_begin_prefill(g);

    const bool split_profile =
        getenv("DS4_CUDA_GRAPH_PREFILL_SPLIT_PROFILE") != NULL;
    const bool profile = split_profile ||
        getenv("DS4_CUDA_GRAPH_PREFILL_PROFILE") != NULL;
    const bool throttle = graph_power_throttle_enabled(g);
    const bool callback_split = display_progress != NULL && n_tokens >= 32u;
    const bool split_commands = split_profile || throttle || callback_split ||
                                n_tokens > 2048u;
    const double t0 = profile ? now_sec() : 0.0;
    double encode_s = 0.0;
    double execute_s = 0.0;
    bool ok = metal_graph_upload_prompt_embeddings_hc(
            metal_graph_batch_cur_hc(g), metal_graph_prefill_tokens(g),
            model, weights, prompt, start, n_tokens);

    if (!split_commands) {
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = metal_graph_encode_layer_batch(g, model, &weights->layer[il],
                                                il, start, n_tokens);
            if (ok) {
                ok = metal_graph_dspark_capture_prefill_layer(g, il, start,
                                                              n_tokens);
            }
            if (show_progress) {
                fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1u,
                        (uint32_t)DS4_N_LAYER);
                fflush(stderr);
            }
        }

        const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
        ds4_gpu_tensor *saved_cur = metal_graph_cur_hc(g);
        ds4_gpu_tensor *last_hc = NULL;
        if (ok && logits) {
            last_hc = metal_graph_tensor_row_view(metal_graph_batch_cur_hc(g),
                                                  n_tokens - 1u, hc_dim);
            ok = last_hc != NULL;
        }
        if (ok && logits) {
            g->cur_hc_by_tier[g->active_tier] = last_hc;
            ok = metal_graph_encode_output_head(g, model, weights,
                                                weights->output->dim[1]);
        }
        const double t_encoded = profile ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double t_done = profile ? now_sec() : 0.0;
        g->cur_hc_by_tier[g->active_tier] = saved_cur;
        ds4_gpu_tensor_free(last_hc);
        if (!ok) return false;

        const double t_before_read = profile ? now_sec() : 0.0;
        if (logits) {
            ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits,
                                     (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (profile) {
            const double t_read = now_sec();
            fprintf(stderr,
                    "ds4: gpu graph prefill tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
                    n_tokens,
                    (t_encoded - t0) * 1000.0,
                    (t_done - t_encoded) * 1000.0,
                    (t_read - t_before_read) * 1000.0,
                    (t_read - t0) * 1000.0);
        }
        if (show_progress) fputc('\n', stderr);
        if (display_progress) {
            display_progress(display_progress_ud, "prefill_display",
                             (int)(start + n_tokens), prompt->len);
        }
        return ok;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        double layer_elapsed = 0.0;
        if (split_profile) {
            const double t_attn0 = now_sec();
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) {
                ok = metal_graph_encode_layer_attention_batch(
                        g, model, &weights->layer[il], il, start, n_tokens);
            }
            const double t_attn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_attn_done = now_sec();

            const double t_ffn0 = now_sec();
            if (ok) ok = ds4_gpu_begin_commands() != 0;
            if (ok) {
                ok = metal_graph_encode_layer_ffn_batch(
                        g, model, &weights->layer[il], il, start, n_tokens);
            }
            if (ok) {
                ds4_gpu_tensor *tmp = metal_graph_batch_cur_hc(g);
                g->batch_cur_hc_by_tier[g->active_tier] =
                    metal_graph_batch_next_hc(g);
                g->batch_next_hc_by_tier[g->active_tier] = tmp;
                ok = metal_graph_dspark_capture_prefill_layer(g, il, start,
                                                              n_tokens);
            }
            const double t_ffn_encoded = now_sec();
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_ffn_done = now_sec();

            encode_s += (t_attn_encoded - t_attn0) +
                        (t_ffn_encoded - t_ffn0);
            execute_s += (t_attn_done - t_attn_encoded) +
                         (t_ffn_done - t_ffn_encoded);
            layer_elapsed = (t_attn_done - t_attn0) +
                            (t_ffn_done - t_ffn0);
            fprintf(stderr,
                    "ds4: gpu prefill layer %u attn encode=%.3f execute=%.3f ms ffn encode=%.3f execute=%.3f ms\n",
                    il,
                    (t_attn_encoded - t_attn0) * 1000.0,
                    (t_attn_done - t_attn_encoded) * 1000.0,
                    (t_ffn_encoded - t_ffn0) * 1000.0,
                    (t_ffn_done - t_ffn_encoded) * 1000.0);
        } else {
            const double t_layer0 = (profile || throttle) ? now_sec() : 0.0;
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) {
                ok = metal_graph_encode_layer_batch(g, model,
                                                    &weights->layer[il], il,
                                                    start, n_tokens);
            }
            if (ok) {
                ok = metal_graph_dspark_capture_prefill_layer(g, il, start,
                                                              n_tokens);
            }
            const double t_encoded = (profile || throttle) ? now_sec() : 0.0;
            if (ok) ok = ds4_gpu_end_commands() != 0;
            const double t_done = (profile || throttle) ? now_sec() : 0.0;
            layer_elapsed = t_done - t_layer0;
            if (profile) {
                encode_s += t_encoded - t_layer0;
                execute_s += t_done - t_encoded;
            }
        }
        if (!ok) {
            fprintf(stderr, "ds4: gpu prefill layer %u failed\n", il);
            return false;
        }
        graph_power_note_prefill_layer(g, il, layer_elapsed);
        gpu_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  start, n_tokens, il + 1u,
                                                  prompt->len);
        if (show_progress) {
            fprintf(stderr, "ds4: gpu prefill layer %u/%u\r", il + 1u,
                    (uint32_t)DS4_N_LAYER);
            fflush(stderr);
        }
    }
    if (show_progress) fputc('\n', stderr);

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *saved_cur = metal_graph_cur_hc(g);
    ds4_gpu_tensor *last_hc = NULL;
    const double t_head0 = profile ? now_sec() : 0.0;
    if (logits) {
        last_hc = metal_graph_tensor_row_view(metal_graph_batch_cur_hc(g),
                                              n_tokens - 1u, hc_dim);
        ok = last_hc != NULL;
    }
    if (ok && logits) {
        g->cur_hc_by_tier[g->active_tier] = last_hc;
        ok = ds4_gpu_begin_commands() != 0;
    }
    if (ok && logits) {
        ok = metal_graph_encode_output_head(g, model, weights,
                                            weights->output->dim[1]);
    }
    const double t_head_encoded = profile ? now_sec() : 0.0;
    if (ok && logits) ok = ds4_gpu_end_commands() != 0;
    const double t_head_done = profile ? now_sec() : 0.0;
    g->cur_hc_by_tier[g->active_tier] = saved_cur;
    ds4_gpu_tensor_free(last_hc);
    if (!ok) return false;

    const double t_before_read = profile ? now_sec() : 0.0;
    if (logits) {
        ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (profile) {
        const double t_read = now_sec();
        encode_s += t_head_encoded - t_head0;
        execute_s += t_head_done - t_head_encoded;
        fprintf(stderr,
                "ds4: gpu layer-major prefill tokens=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms\n",
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
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (n_tokens == 0 || g->prefill_cap == 0) return false;
    if (start > (uint32_t)prompt->len) return false;
    if (n_tokens > (uint32_t)prompt->len - start) return false;
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
