#ifndef DS4_NO_GPU
/* Metal generation entry point.  The model runs as one local whole-graph
 * pipeline: graph prefill followed by graph decode steps.  Streaming PRO may
 * use decode-style prefill for short prompts. */
static int generate_metal_graph_raw_swa(
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
        int                 power_percent,
        uint32_t            prefill_chunk,
        const char        * directional_steering_file,
        float               directional_steering_attn,
        float               directional_steering_ffn,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud,
        ds4_session_progress_fn progress,
        void              * progress_ud) {
    fprintf(stderr, "ds4: using GPU graph generation with graph prefill\n");

    if (prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        return 1;
    }
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        if (power_percent > 0 && power_percent < 100) {
            fprintf(stderr, "ds4: --power is not supported by the GLM Metal path yet\n");
            return 1;
        }
        if (prefill_chunk != 0) {
            fprintf(stderr,
                    "ds4: --prefill-chunk is not supported by the GLM Metal path; "
                    "GLM uses graph-selected prefill chunks\n");
            return 1;
        }
        if ((directional_steering_file && directional_steering_file[0]) ||
            directional_steering_attn != 0.0f ||
            directional_steering_ffn != 0.0f) {
            fprintf(stderr,
                    "ds4: directional steering is not supported by the GLM Metal path yet\n");
            return 1;
        }
        return generate_glm_metal_argmax(model,
                                         vocab,
                                         weights,
                                         prompt,
                                         n_predict,
                                         ctx_size,
                                         quality,
                                         ssd_streaming,
                                         ssd_streaming_cold,
                                         ssd_streaming_preload_experts,
                                         ssd_streaming_cache_bytes,
                                         ssd_streaming_prefill_headroom_bytes,
                                         emit,
                                         done,
                                         emit_ud,
                                         progress,
                                         progress_ud);
    }

    const uint32_t prefill_cap =
        metal_graph_prefill_cap_for_prompt(prompt->len, prefill_chunk);
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, prefill_cap);
    if (prefill_cap < (uint32_t)prompt->len) {
        fprintf(stderr,
                "ds4: using chunked GPU prefill (%u-token chunks for %d prompt tokens)\n",
                prefill_cap,
                prompt->len);
    }
    ds4_gpu_graph g;
    /* diagnostic single-tier callsite; placement=NULL. */
    bool ok = metal_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size,
                                        prefill_cap, false, NULL, false, NULL);
    if (!ok) {
        fprintf(stderr, "ds4: failed to allocate GPU graph runtime\n");
        return 1;
    }
    g.quality = quality;
    g.ssd_streaming = ssd_streaming;
    g.ssd_streaming_cold = ssd_streaming_cold;
    g.streaming_preload_experts = ssd_streaming_preload_experts;
    g.power_percent = power_percent > 0 ? (uint32_t)power_percent : 100u;
    if (!metal_graph_load_directional_steering(&g,
                                               directional_steering_file,
                                               directional_steering_attn,
                                               directional_steering_ffn)) {
        metal_graph_free(&g);
        return 1;
    }
    const bool memory_report = getenv("DS4_METAL_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_gpu_print_memory_report("after graph alloc");

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    const bool trace_top = getenv("DS4_TRACE_TOP") != NULL;
    const bool token_timing = getenv("DS4_TOKEN_TIMING") != NULL;

    const double t_prefill0 = now_sec();
    if (prefill_cap < (uint32_t)prompt->len) {
        ok = metal_graph_prefill_chunked(&g, model, weights, prompt,
                                         prompt->len, logits, false,
                                         progress, progress_ud,
                                         progress, progress_ud,
                                         NULL, NULL, NULL);
    } else {
        ok = metal_graph_prefill_raw_swa(&g, model, weights, prompt,
                                         prompt->len, logits, true,
                                         progress, progress_ud,
                                         NULL, NULL, NULL);
    }
    const double t_prefill1 = now_sec();
    if (memory_report) ds4_gpu_print_memory_report("after prefill");

    if (!ok) {
        free(logits);
        metal_graph_free(&g);
        return 1;
    }
    const char *dump_prefill_logits = getenv("DS4_METAL_DUMP_PREFILL_LOGITS");
    if (dump_prefill_logits && dump_prefill_logits[0]) {
        if (!write_f32_binary_file(dump_prefill_logits, logits, DS4_N_VOCAB)) {
            free(logits);
            metal_graph_free(&g);
            return 1;
        }
        fprintf(stderr, "ds4: wrote GPU prefill logits to %s\n", dump_prefill_logits);
    }

    int pos = prompt->len;
    int n_generated = 0;
    int n_decode_eval = 0;
    const double t_decode0 = now_sec();
    for (int i = 0; i < n_predict && pos < ctx_size; i++) {
        if (trace_top) {
            char label[64];
            snprintf(label, sizeof(label), "step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }

        int token = sample_argmax(logits, DS4_N_VOCAB);
        if (vocab_token_is_generation_stop(vocab, token)) break;

        if (emit) emit(emit_ud, token);
        n_generated++;

        if (i == n_predict - 1 || pos + 1 >= ctx_size) {
            pos++;
            break;
        }

        const double t_eval0 = token_timing ? now_sec() : 0.0;
        ok = metal_graph_eval_token_raw_swa(&g,
                                            model,
                                            weights,
                                            (uint32_t)token,
                                            (uint32_t)pos,
                                            logits);
        if (!ok) break;
        if (token_timing) {
            const double t_eval1 = now_sec();
            fprintf(stderr, "ds4: gpu decode eval %d took %.3f ms\n", n_decode_eval + 1, (t_eval1 - t_eval0) * 1000.0);
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
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)n_generated / decode_s : 0.0);

    if (memory_report) ds4_gpu_print_memory_report("before graph free");
    free(logits);
    metal_graph_free(&g);
    return ok ? 0 : 1;
}
#endif /* !DS4_NO_GPU */
