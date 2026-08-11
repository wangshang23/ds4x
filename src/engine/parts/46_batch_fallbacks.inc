
static int ds4_sessions_eval_batch_cuda(ds4_decode_item *items, int count,
                            char *err, size_t errlen) {
    if (!items || count <= 0) {
        if (err && errlen) snprintf(err, errlen, "empty decode batch");
        return 1;
    }
    if (count == 1) {
        return ds4_session_eval(items[0].session, items[0].token, err, errlen);
    }

    ds4_session *first = items[0].session;
    if (!first || !first->engine) {
        if (err && errlen) snprintf(err, errlen, "decode batch has no session");
        return 1;
    }
    ds4_engine *e = first->engine;
    for (int i = 0; i < count; i++) {
        ds4_session *s = items[i].session;
        if (!s || s->engine != e) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d belongs to a different engine", i);
            }
            return 1;
        }
        for (int j = 0; j < i; j++) {
            if (items[j].session == s) {
                if (err && errlen) {
                    snprintf(err, errlen,
                             "decode batch repeats session at items %d and %d",
                             j, i);
                }
                return 1;
            }
        }
        if (ds4_session_pos(s) >= ds4_session_ctx(s)) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d reached its context limit", i);
            }
            return 1;
        }
    }

#ifndef DS4_NO_GPU
    /* The DeepSeek CUDA graph uses one default stream per device plus ordered
     * cross-device events.  Encoding several independent graphs before the
     * final synchronization lets their packets fill different pipeline stages
     * while preserving the exact one-token kernels and per-session KV order. */
    if (e->backend == DS4_BACKEND_CUDA &&
        !ds4_session_is_glm(first) &&
        e->support_kind == DS4_SUPPORT_NONE) {
        bool ok = ds4_gpu_begin_commands() != 0;
        for (int i = 0; ok && i < count; i++) {
            ds4_session *s = items[i].session;
            if (ds4_session_is_cpu(s) || ds4_session_is_glm(s) ||
                !s->checkpoint_valid || s->engine->support_kind != DS4_SUPPORT_NONE) {
                ok = false;
            }
        }
        const char *interleave = getenv("DS4_CUDA_SESSION_BATCH_INTERLEAVE");
        const bool use_pipeline = !interleave || !interleave[0] ||
                                  strcmp(interleave, "0") != 0;
        if (ok && use_pipeline && first->graph.placement) {
            ok = metal_graph_encode_session_pipeline_batch(
                    items, count, &e->model, &e->weights);
        } else {
            for (int i = 0; ok && i < count; i++) {
                ds4_session *s = items[i].session;
                ok = metal_graph_encode_token_raw_swa(
                        &s->graph,
                        &e->model,
                        &e->weights,
                        items[i].token,
                        (uint32_t)s->checkpoint.len,
                        true,
                        false);
            }
        }
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();

        for (int i = 0; ok && i < count; i++) {
            ds4_session *s = items[i].session;
            ok = ds4_gpu_tensor_read(
                    metal_graph_logits(&s->graph),
                    0,
                    s->logits,
                    (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (!ok) {
            for (int i = 0; i < count; i++) {
                items[i].session->checkpoint_valid = false;
                items[i].session->mtp_draft_valid = false;
            }
            if (err && errlen) {
                snprintf(err, errlen, "%s batched decode failed",
                         ds4_backend_name(e->backend));
            }
            return 1;
        }

        for (int i = 0; i < count; i++) {
            ds4_session *s = items[i].session;
            token_vec_push(&s->checkpoint, items[i].token);
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            ds4_session_dspark_capture_note_checkpoint(s);
        }
        return 0;
    }
#endif

    /* Correctness fallback.  If a later item fails, invalidate every member:
     * earlier sessions may already have advanced and must be rebuilt rather
     * than exposing a partially committed logical batch. */
    for (int i = 0; i < count; i++) {
        if (ds4_session_eval(items[i].session, items[i].token,
                             err, errlen) != 0) {
            for (int j = 0; j < count; j++) ds4_session_invalidate(items[j].session);
            return 1;
        }
    }
    return 0;
}

#ifdef DS4_TEST_HOOKS
static uint64_t ds4_test_mixed_native_evals = 0;

uint64_t ds4_test_mixed_native_count(void) {
    return ds4_test_mixed_native_evals;
}
#endif

static int ds4_sessions_eval_batch_with_prefill_cuda(
        ds4_decode_item *items,
        int count,
        ds4_session *prefill_session,
        const ds4_tokens *prefill_prompt,
        char *err,
        size_t errlen) {
    if (!items || count <= 0 || !prefill_session || !prefill_prompt ||
        !prefill_session->engine) {
        if (err && errlen) snprintf(err, errlen, "invalid mixed model batch");
        return 1;
    }
    ds4_engine *e = prefill_session->engine;
    if (!prefill_session->checkpoint_valid ||
        prefill_prompt->len <= prefill_session->checkpoint.len ||
        prefill_prompt->len >= prefill_session->ctx_size ||
        !ds4_tokens_starts_with(prefill_prompt, &prefill_session->checkpoint)) {
        if (err && errlen) {
            snprintf(err, errlen,
                     "mixed prefill must extend a valid session checkpoint");
        }
        return 1;
    }
    for (int i = 0; i < count; i++) {
        ds4_session *s = items[i].session;
        if (!s || s == prefill_session || s->engine != e ||
            !s->checkpoint_valid || s->checkpoint.len >= s->ctx_size) {
            if (err && errlen) {
                snprintf(err, errlen, "invalid mixed decode item %d", i);
            }
            return 1;
        }
        for (int j = 0; j < i; j++) {
            if (items[j].session == s) {
                if (err && errlen) {
                    snprintf(err, errlen,
                             "mixed decode repeats session at items %d and %d",
                             j, i);
                }
                return 1;
            }
        }
    }

#ifndef DS4_NO_GPU
    const uint32_t start = (uint32_t)prefill_session->checkpoint.len;
    const uint32_t prefill_rows =
        (uint32_t)prefill_prompt->len - start;
    const bool native_requested =
        metal_graph_tp_env_flag("DS4_CUDA_MIXED_PREFILL_DECODE", true);
    const bool native_supported =
        native_requested &&
        e->backend == DS4_BACKEND_CUDA &&
        !ds4_session_is_glm(prefill_session) &&
        e->support_kind == DS4_SUPPORT_NONE &&
        metal_graph_mixed_prefill_decode_supported(
                prefill_session, prefill_prompt, start, prefill_rows,
                items, count, &e->weights);
    if (native_supported) {
        bool ok = metal_graph_eval_mixed_prefill_decode(
                prefill_session, prefill_prompt, start, prefill_rows,
                items, count, &e->model, &e->weights);
        if (!ok) {
            prefill_session->checkpoint_valid = false;
            prefill_session->mtp_draft_valid = false;
            for (int i = 0; i < count; i++) {
                items[i].session->checkpoint_valid = false;
                items[i].session->mtp_draft_valid = false;
            }
            if (err && errlen) snprintf(err, errlen, "CUDA mixed model batch failed");
            return 1;
        }

        ds4_tokens_copy(&prefill_session->checkpoint, prefill_prompt);
        prefill_session->checkpoint_valid = true;
        prefill_session->mtp_draft_valid = false;
        ds4_session_dspark_capture_note_checkpoint(prefill_session);
        session_greedy_splitkv_reset(prefill_session);
        if (prefill_session->progress) {
            prefill_session->progress(
                    prefill_session->progress_ud, "prefill_chunk",
                    prefill_prompt->len, prefill_prompt->len);
        }
        for (int i = 0; i < count; i++) {
            ds4_session *s = items[i].session;
            token_vec_push(&s->checkpoint, items[i].token);
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            ds4_session_dspark_capture_note_checkpoint(s);
        }
#ifdef DS4_TEST_HOOKS
        ds4_test_mixed_native_evals++;
#endif
        return 0;
    }
#endif

    int rc = ds4_session_sync(
            prefill_session, prefill_prompt, err, errlen);
    if (rc != 0) return rc;
    rc = ds4_sessions_eval_batch(items, count, err, errlen);
    if (rc != 0) ds4_session_invalidate(prefill_session);
    return rc;
}

int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    if (!s || max_tokens <= 0 || accepted_cap <= 0) return 0;
    if (s->distributed) {
        if (!accepted) return 0;
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        (void)max_tokens;
        (void)eos_token;
        if (!accepted || accepted_cap <= 0) return 0;
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
    if (ds4_session_is_glm(s)) {
        (void)max_tokens;
        (void)eos_token;
        if (!accepted || accepted_cap <= 0) return 0;
#ifndef DS4_NO_GPU
        if (s->engine->glm_mtp && DS4_N_NEXTN_PREDICT != 0 &&
            s->glm_graph_ready) {
            if (ds4_session_tp_leader(s)) {
                ds4_engine *ge = s->engine;
                if (!ds4_tp_send_eval(ge->tp.ctx, s->tp_session_id,
                                      ++ge->tp.eval_seq, first_token)) {
                    snprintf(err, errlen, "tp: worker eval send failed");
                    return -1;
                }
            }
            int rc = ds4_session_glm_spec_cycle(s, first_token, accepted,
                                                accepted_cap, err, errlen);
#if defined(__APPLE__)
            if (rc >= 0 && s->engine && s->engine->tp.active && ds4_gpu_tp_failed()) {
                snprintf(err, errlen, "tp: gate transport failed");
                return -1;
            }
#endif
            return rc;
        }
#endif
        if (ds4_session_eval(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
#ifdef DS4_NO_GPU
    (void)s; (void)first_token; (void)max_tokens; (void)eos_token;
    (void)accepted; (void)accepted_cap;
    snprintf(err, errlen, "GPU support is not compiled in");
    return -1;
#else
    ds4_engine *e = s->engine;
    if (ds4_session_is_glm(s) && ds4_engine_glm_mtp_spec_enabled(e)) {
        int cycle_cap = accepted_cap;
        if (cycle_cap > max_tokens) cycle_cap = max_tokens;
        return ds4_session_glm_spec_cycle(s, first_token,
                                          accepted, cycle_cap,
                                          err, errlen);
    }

    /*
     * MTP in DeepSeek V4 is a speculative drafter, not a replacement sampler.
     * The target model still defines the exact output stream.  A cycle starts
     * by accepting one normal target token, then asks the MTP block to propose
     * a short suffix.  The suffix is useful only if the target model can verify
     * several proposed positions together; running ordinary decode once per
     * draft token is correctness-safe but cannot be faster than baseline.
     */
    const bool strict_dspark =
        e->support_kind == DS4_SUPPORT_DSPARK &&
        (e->quality || e->dspark_strict);
    bool can_prepare_support_draft =
        !strict_dspark &&
        first_token != eos_token && max_tokens > 1 && accepted_cap > 1;
    if (can_prepare_support_draft &&
        e->tp.active &&
        e->support_kind == DS4_SUPPORT_MTP_LEGACY) {
        can_prepare_support_draft = false;
    }
    bool dspark_tail_skip = false;
    if (can_prepare_support_draft && e->support_kind == DS4_SUPPORT_DSPARK &&
        ds4_dspark_scheduler_enabled()) {
        const uint32_t tail_min = ds4_dspark_scheduler_tail_min_tokens();
        if (tail_min != 0 && (uint32_t)max_tokens < tail_min) {
            can_prepare_support_draft = false;
            dspark_tail_skip = true;
            if (ds4_dspark_stats_enabled()) s->dspark_stats.tail_skips++;
            if (getenv("DS4_DSPARK_SPEC_LOG") != NULL) {
                fprintf(stderr,
                        "ds4: DSpark scheduler tail skip max=%d min=%u\n",
                        max_tokens,
                        tail_min);
            }
        }
    }
    if (ds4_session_eval_probe_tp(s,
                                  first_token,
                                  can_prepare_support_draft,
                                  err,
                                  errlen) != 0) return -1;
    int n_accept = 0;
    accepted[n_accept++] = first_token;
    if (first_token == eos_token || max_tokens == 1 || n_accept >= accepted_cap) return n_accept;
    if (strict_dspark) return n_accept;
    if (dspark_tail_skip) return n_accept;

    if (e->support_kind == DS4_SUPPORT_DSPARK) {
        return ds4_session_eval_dspark_speculative_argmax(s,
                                                          n_accept,
                                                          max_tokens,
                                                          eos_token,
                                                          accepted,
                                                          accepted_cap,
                                                          err,
                                                          errlen);
    }

    if (metal_graph_cuda_splitkv_spec_requested() &&
        (!e->mtp_ready || e->mtp_draft_tokens <= 1)) {
        int extra = ds4_session_eval_splitkv_spec_after_first(
                s,
                max_tokens - n_accept,
                eos_token,
                accepted + n_accept,
                accepted_cap - n_accept,
                err,
                errlen);
        if (extra < 0) return -1;
        return n_accept + extra;
    }

    /* Legacy MTP verify is not TP-mirrored; fall back to per-token decode. */
    if (e->tp.active) return n_accept;

    if (!e->mtp_ready || !s->mtp_draft_valid || e->mtp_draft_tokens <= 1) return n_accept;

    int draft_cap = e->mtp_draft_tokens;
    if (draft_cap > max_tokens - n_accept) draft_cap = max_tokens - n_accept;
    if (draft_cap > accepted_cap - n_accept) draft_cap = accepted_cap - n_accept;
    int room = s->ctx_size - s->checkpoint.len;
    if (draft_cap > room - 1) draft_cap = room - 1;
    if (draft_cap <= 0) return n_accept;

    int drafts[16];
    int draft_n = 1;
    drafts[0] = s->mtp_draft_token;
    s->mtp_draft_valid = false;
    const bool strict_mtp = e->quality || getenv("DS4_MTP_STRICT") != NULL;
    float mtp_margin_threshold = e->mtp_margin;
    const char *mtp_margin_env = getenv("DS4_MTP_MIN_MARGIN");
    if (mtp_margin_env && mtp_margin_env[0]) {
        char *end = NULL;
        float v = strtof(mtp_margin_env, &end);
        if (end != mtp_margin_env && v >= 0.0f) mtp_margin_threshold = v;
    }
    const bool mtp_timing = getenv("DS4_MTP_TIMING") != NULL;
    const bool mtp_conf_log = getenv("DS4_MTP_CONF_LOG") != NULL;
    const bool mtp_need_logits = mtp_conf_log ||
        getenv("DS4_MTP_FULL_LOGITS") != NULL ||
        (!strict_mtp && mtp_margin_threshold > 0.0f);
    const double mtp_t0 = mtp_timing ? now_sec() : 0.0;
    double mtp_t_after_draft = mtp_t0;
    float mtp_last_margin = 0.0f;
    int mtp_last_top0 = -1, mtp_last_top1 = -1;

    /*
     * The first proposed token is verified for free: ds4_session_eval() just
     * produced the base logits for the committed prefix.  If MTP disagrees at
     * this point there is no suffix to verify, so the exact behavior is to emit
     * only first_token and skip all speculative work.
     */
    if (sample_argmax(s->logits, DS4_N_VOCAB) != drafts[0]) {
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: mtp spec miss first draft=%d\n", drafts[0]);
        }
        return n_accept;
    }
    if (drafts[0] == eos_token) draft_cap = 1;
    const uint32_t mtp_base_raw = s->graph.mtp_n_raw;
    /*
     * MTP has its own raw SWA cache. Recursive drafting writes speculative
     * future rows into it; after verification, rows beyond the accepted prefix
     * must become invisible.  We do not copy/rollback the cache body because the
     * next draft attempt will overwrite future slots.  A counter is enough.
     */
#define DS4_MTP_KEEP_ACCEPTED(n_) do { \
        uint32_t keep_ = mtp_base_raw + (uint32_t)(n_); \
        if (keep_ > s->graph.raw_window) keep_ = s->graph.raw_window; \
        s->graph.mtp_n_raw = keep_; \
    } while (0)

    for (; draft_n < draft_cap; draft_n++) {
        ds4_gpu_tensor *prev_hc = (draft_n & 1) ? s->graph.mtp_state_hc : s->graph.mtp_next_hc;
        ds4_gpu_tensor *out_hc = (draft_n & 1) ? s->graph.mtp_next_hc : s->graph.mtp_state_hc;
        int mtp_top = -1;
        if (!metal_graph_eval_mtp_draft_from_hc(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                &e->mtp_model,
                                                &e->mtp_weights,
                                                prev_hc,
                                                out_hc,
                                                drafts[draft_n - 1],
                                                (uint32_t)(s->checkpoint.len + draft_n - 1),
                                                mtp_need_logits ? s->mtp_logits : NULL,
                                                &mtp_top))
        {
            return n_accept;
        }
        drafts[draft_n] = mtp_top >= 0 ? mtp_top : sample_argmax(s->mtp_logits, DS4_N_VOCAB);
        if (drafts[draft_n] == eos_token) {
            draft_n++;
            break;
        }
    }
    if (mtp_conf_log && draft_n > 1) {
        float v0 = 0.0f, v1 = 0.0f;
        logits_top2(s->mtp_logits, DS4_N_VOCAB, &mtp_last_top0, &v0, &mtp_last_top1, &v1);
        mtp_last_margin = v0 - v1;
    }
    if (mtp_timing) mtp_t_after_draft = now_sec();

    if (!strict_mtp && draft_n == 2 && mtp_margin_threshold > 0.0f) {
        if (!mtp_conf_log) {
            float v0 = 0.0f, v1 = 0.0f;
            logits_top2(s->mtp_logits, DS4_N_VOCAB, &mtp_last_top0, &v0, &mtp_last_top1, &v1);
            mtp_last_margin = v0 - v1;
        }
        if (mtp_last_margin < mtp_margin_threshold) {
            float *row_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row_logits[0]));
            const int start = s->checkpoint.len;
            const double verify_t0 = mtp_timing ? now_sec() : 0.0;
            bool ok = metal_graph_eval_token_raw_swa(&s->graph,
                                                     &e->model,
                                                     &e->weights,
                                                     drafts[0],
                                                     (uint32_t)start,
                                                     row_logits);
            if (!ok) {
                free(row_logits);
                snprintf(err, errlen, "%s decode failed", ds4_backend_name(e->backend));
                s->checkpoint_valid = false;
                return -1;
            }
            memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
            free(row_logits);
            token_vec_push(&s->checkpoint, drafts[0]);
            accepted[n_accept++] = drafts[0];
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            DS4_MTP_KEEP_ACCEPTED(1);
            ds4_session_dspark_capture_note_checkpoint(s);
            if (mtp_timing) {
                const double done = now_sec();
                fprintf(stderr,
                        "ds4: mtp timing margin-skip drafted=2 committed=1 margin=%.3f threshold=%.3f draft=%.3f ms verify=%.3f ms total=%.3f ms\n",
                        mtp_last_margin,
                        mtp_margin_threshold,
                        (mtp_t_after_draft - mtp_t0) * 1000.0,
                        (done - verify_t0) * 1000.0,
                        (done - mtp_t0) * 1000.0);
            }
            return n_accept;
        }
    }

    /*
     * Metal normally benefits from the tiny-batch verifier: it checks two
     * target positions in one layer-major pass and commits prefix-1 directly
     * on a partial accept. Like the rest of the non-quality Metal path, it may
     * pick a different greedy token when batched reductions perturb nearly
     * tied logits.
     *
     * ROCm is different. Its DeepSeek one-token graph uses the prequantized Q8
     * decode kernels, while the generic N=2 batch graph uses full-F32
     * activations and loses the paired and HC-expand decode fusions. Running
     * the exact two-row verifier reuses those Q8 kernels and is faster.
     * DS4_MTP_BATCH_VERIFY remains an explicit diagnostic rollback to the
     * generic batch verifier.
     */
#ifdef DS4_ROCM_BUILD
    const bool prefer_decode2_exact = true;
#else
    const bool prefer_decode2_exact = false;
#endif
    const bool use_decode2_exact =
        draft_n == 2 &&
        (strict_mtp || prefer_decode2_exact) &&
        getenv("DS4_MTP_BATCH_VERIFY") == NULL;
    if (use_decode2_exact) {
        ds4_spec_frontier frontier;
        memset(&frontier, 0, sizeof(frontier));
        float *row_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row_logits[0]));
        float *row0_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row0_logits[0]));
        const int start = s->checkpoint.len;
        int row0_top = -1;
        const double snapshot_t0 = mtp_timing ? now_sec() : 0.0;
        bool have_frontier = spec_frontier_snapshot(&frontier, s);
        const double snapshot_done = mtp_timing ? now_sec() : 0.0;
        bool ok = have_frontier;
        if (ok) {
            ok = metal_graph_verify_decode2_exact(&s->graph,
                                                  &e->model,
                                                  &e->weights,
                                                  drafts[0],
                                                  drafts[1],
                                                  (uint32_t)start,
                                                  &row0_top,
                                                  NULL,
                                                  row0_logits,
                                                  row_logits);
        }
        const double verify_done = mtp_timing ? now_sec() : 0.0;
        if (ok && row0_top == drafts[1]) {
            memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
            token_vec_push(&s->checkpoint, drafts[0]);
            token_vec_push(&s->checkpoint, drafts[1]);
            accepted[n_accept++] = drafts[0];
            if (n_accept < accepted_cap) accepted[n_accept++] = drafts[1];
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            DS4_MTP_KEEP_ACCEPTED(2);
            ds4_session_dspark_capture_note_checkpoint(s);
            if (mtp_timing) {
                fprintf(stderr,
                        "ds4: mtp timing decode2 drafted=2 committed=2 draft=%.3f ms snapshot=%.3f ms verify=%.3f ms total=%.3f ms\n",
                        (mtp_t_after_draft - mtp_t0) * 1000.0,
                        (snapshot_done - snapshot_t0) * 1000.0,
                        (verify_done - snapshot_done) * 1000.0,
                        (now_sec() - mtp_t0) * 1000.0);
            }
            spec_frontier_free(&frontier);
            free(row0_logits);
            free(row_logits);
            return n_accept;
        }

        if (ok) {
            s->checkpoint.len = start;
            ds4_session_dspark_capture_invalidate(s);
            ok = spec_frontier_commit_prefix1(s);
        }
        if (ok) memcpy(s->logits, row0_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        if (ok) {
            token_vec_push(&s->checkpoint, drafts[0]);
            accepted[n_accept++] = drafts[0];
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            DS4_MTP_KEEP_ACCEPTED(1);
            ds4_session_dspark_capture_note_checkpoint(s);
            if (mtp_timing) {
                const double replay_done = now_sec();
                fprintf(stderr,
                        "ds4: mtp timing decode2 drafted=2 committed=1 draft=%.3f ms snapshot=%.3f ms verify=%.3f ms prefix=%.3f ms total=%.3f ms\n",
                        (mtp_t_after_draft - mtp_t0) * 1000.0,
                        (snapshot_done - snapshot_t0) * 1000.0,
                        (verify_done - snapshot_done) * 1000.0,
                        (replay_done - verify_done) * 1000.0,
                        (replay_done - mtp_t0) * 1000.0);
            }
            spec_frontier_free(&frontier);
            free(row0_logits);
            free(row_logits);
            return n_accept;
        }
        if (have_frontier) {
            s->checkpoint.len = start;
            ds4_session_dspark_capture_invalidate(s);
            (void)spec_frontier_restore(&frontier, s);
        }
        spec_frontier_free(&frontier);
        free(row0_logits);
        free(row_logits);
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: mtp decode2 verifier failed, falling back to sequential\n");
        }
    }

    if (!use_decode2_exact)
    {
        ds4_spec_frontier frontier;
        memset(&frontier, 0, sizeof(frontier));
        int *row_tops = xmalloc((size_t)draft_n * sizeof(row_tops[0]));
        float *row_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(row_logits[0]));
        const int start = s->checkpoint.len;
        /*
         * The production MTP depth is two.  Prefix-1 capture makes partial
         * accepts cheap, but it copies per-layer compressor frontiers even when
         * both draft tokens are accepted.  Full accepts are the path that makes
         * MTP worthwhile, so by default we snapshot before the verifier and
         * replay one token on partial accept.  DS4_MTP_CAPTURE_PREFIX1 restores
         * the older no-replay partial path for measurement.
         */
        const bool capture_prefix1 =
            draft_n == 2 && (!strict_mtp || getenv("DS4_MTP_CAPTURE_PREFIX1") != NULL);
        const bool exact_replay_debug = getenv("DS4_MTP_EXACT_REPLAY") != NULL;
        const bool snapshot_required =
            draft_n > 2 ||
            (draft_n == 2 && (!capture_prefix1 || exact_replay_debug)) ||
            getenv("DS4_MTP_FORCE_SNAPSHOT") != NULL;
        bool have_frontier = false;
        bool ok = true;
        bool verifier_may_have_mutated = false;
        const double snapshot_t0 = mtp_timing ? now_sec() : 0.0;
        if (snapshot_required) {
            have_frontier = spec_frontier_snapshot(&frontier, s);
            ok = have_frontier;
        }
        const double snapshot_done = mtp_timing ? now_sec() : 0.0;
        if (ok) {
            for (int i = 0; i < draft_n; i++) token_vec_push(&s->checkpoint, drafts[i]);
            verifier_may_have_mutated = true;
            ok = metal_graph_verify_suffix_tops(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                &s->checkpoint,
                                                (uint32_t)start,
                                                (uint32_t)draft_n,
                                                capture_prefix1,
                                                false,
                                                row_tops,
                                                NULL,
                                                NULL);
        }
        const double micro_verify_done = mtp_timing ? now_sec() : 0.0;
        if (ok) {
            int commit_drafts = 1;
            for (int i = 1; i < draft_n; i++) {
                if (row_tops[i - 1] != drafts[i]) break;
                commit_drafts++;
            }
            if (mtp_conf_log) {
                fprintf(stderr,
                        "ds4: mtp conf drafted=%d committed=%d mtp_top=%d runner=%d margin=%.6f target_next=%d draft_next=%d\n",
                        draft_n,
                        commit_drafts,
                        mtp_last_top0,
                        mtp_last_top1,
                        mtp_last_margin,
                        draft_n > 1 ? row_tops[0] : -1,
                        draft_n > 1 ? drafts[1] : -1);
            }
            if (exact_replay_debug && have_frontier) {
                s->checkpoint.len = start;
                ds4_session_dspark_capture_invalidate(s);
                ok = spec_frontier_restore(&frontier, s);
                if (ok) {
                    int replayed = 0;
                    for (; replayed < commit_drafts && ok; replayed++) {
                        ok = metal_graph_eval_token_raw_swa(&s->graph,
                                                            &e->model,
                                                            &e->weights,
                                                            drafts[replayed],
                                                            (uint32_t)(start + replayed),
                                                            row_logits);
                        if (ok) token_vec_push(&s->checkpoint, drafts[replayed]);
                    }
                    if (ok) {
                        memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                        for (int i = 0; i < replayed && n_accept < accepted_cap; i++) {
                            accepted[n_accept++] = drafts[i];
                            if (drafts[i] == eos_token) break;
                        }
                        s->checkpoint_valid = true;
                        s->mtp_draft_valid = false;
                        DS4_MTP_KEEP_ACCEPTED(replayed);
                        ds4_session_dspark_capture_note_checkpoint(s);
                        spec_frontier_free(&frontier);
                        free(row_logits);
                        free(row_tops);
                        return n_accept;
                    }
                }
            }

            if (commit_drafts == draft_n) {
                ok = metal_graph_read_spec_logits_row(&s->graph,
                                                      (uint32_t)(draft_n - 1),
                                                      row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    for (int i = 0; i < draft_n && n_accept < accepted_cap; i++) {
                        accepted[n_accept++] = drafts[i];
                        if (drafts[i] == eos_token) break;
                    }
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(draft_n);
                    ds4_session_dspark_capture_note_checkpoint(s);
                    if (mtp_timing) {
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms total=%.3f ms\n",
                                draft_n,
                                draft_n,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (now_sec() - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            }

            if (draft_n == 2 && commit_drafts == 1 && capture_prefix1) {
                s->checkpoint.len = start;
                ds4_session_dspark_capture_invalidate(s);
                const double prefix_t0 = mtp_timing ? now_sec() : 0.0;
                ok = spec_frontier_commit_prefix1(s);
                const double prefix_done = mtp_timing ? now_sec() : 0.0;
                if (ok) ok = metal_graph_read_spec_logits_row(&s->graph, 0, row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    accepted[n_accept++] = drafts[0];
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(1);
                    token_vec_push(&s->checkpoint, drafts[0]);
                    ds4_session_dspark_capture_note_checkpoint(s);
                    if (mtp_timing) {
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms prefix=%.3f ms total=%.3f ms noreplay=1\n",
                                draft_n,
                                commit_drafts,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (prefix_done - prefix_t0) * 1000.0,
                                (now_sec() - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            } else {
                s->checkpoint.len = start;
                ds4_session_dspark_capture_invalidate(s);
                ok = have_frontier && spec_frontier_restore(&frontier, s);
            }
            if (ok && draft_n == 2 && commit_drafts == 1) {
                ok = metal_graph_eval_token_raw_swa(&s->graph,
                                                    &e->model,
                                                    &e->weights,
                                                    drafts[0],
                                                    (uint32_t)start,
                                                    row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    accepted[n_accept++] = drafts[0];
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(1);
                    token_vec_push(&s->checkpoint, drafts[0]);
                    ds4_session_dspark_capture_note_checkpoint(s);
                    if (mtp_timing) {
                        const double replay_done = now_sec();
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms exact_replay=%.3f ms total=%.3f ms\n",
                                draft_n,
                                commit_drafts,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (replay_done - micro_verify_done) * 1000.0,
                                (replay_done - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            }
            if (ok) {
                for (int i = 0; i < commit_drafts; i++) token_vec_push(&s->checkpoint, drafts[i]);
                ok = metal_graph_verify_suffix_tops(&s->graph,
                                                    &e->model,
                                                    &e->weights,
                                                    &s->checkpoint,
                                                    (uint32_t)start,
                                                    (uint32_t)commit_drafts,
                                                    false,
                                                    false,
                                                    row_tops,
                                                    NULL,
                                                    NULL);
                if (ok) ok = metal_graph_read_spec_logits_row(&s->graph,
                                                              (uint32_t)(commit_drafts - 1),
                                                              row_logits);
                if (ok) {
                    memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
                    for (int i = 0; i < commit_drafts && n_accept < accepted_cap; i++) {
                        accepted[n_accept++] = drafts[i];
                        if (drafts[i] == eos_token) break;
                    }
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    DS4_MTP_KEEP_ACCEPTED(commit_drafts);
                    ds4_session_dspark_capture_note_checkpoint(s);
                    if (mtp_timing) {
                        const double replay_done = now_sec();
                        fprintf(stderr,
                                "ds4: mtp timing micro drafted=%d committed=%d draft=%.3f ms snapshot=%.3f ms verify=%.3f ms replay=%.3f ms total=%.3f ms\n",
                                draft_n,
                                commit_drafts,
                                (mtp_t_after_draft - mtp_t0) * 1000.0,
                                (snapshot_done - snapshot_t0) * 1000.0,
                                (micro_verify_done - snapshot_done) * 1000.0,
                                (replay_done - micro_verify_done) * 1000.0,
                                (replay_done - mtp_t0) * 1000.0);
                    }
                    spec_frontier_free(&frontier);
                    free(row_logits);
                    free(row_tops);
                    return n_accept;
                }
            }
        }
        s->checkpoint.len = start;
        ds4_session_dspark_capture_invalidate(s);
        if (have_frontier) {
            (void)spec_frontier_restore(&frontier, s);
        } else if (!verifier_may_have_mutated) {
            /* Snapshot setup failed before the verifier touched Metal state.
             * Fall through to the exact sequential verifier below. */
        } else {
            snprintf(err, errlen, "MTP verifier failed");
            s->checkpoint_valid = false;
            DS4_MTP_KEEP_ACCEPTED(0);
            spec_frontier_free(&frontier);
            free(row_logits);
            free(row_tops);
            return -1;
        }
        spec_frontier_free(&frontier);
        free(row_logits);
        free(row_tops);
        if (getenv("DS4_MTP_SPEC_LOG")) {
            fprintf(stderr, "ds4: mtp spec micro verifier failed, falling back to sequential\n");
        }
    }

    /*
     * Safety fallback: if the production microbatch verifier fails, verify
     * drafts with the exact normal one-token decode path instead of returning
     * wrong state.  This path is deliberately slow and should not be selected
     * during normal --mtp operation.
     */
    int verified = 0;
    int target_top = sample_argmax(s->logits, DS4_N_VOCAB);
    bool logits_on_host = true;
    const double seq_t0 = mtp_timing ? now_sec() : 0.0;
    for (int i = 0; i < draft_n && n_accept < accepted_cap; i++) {
        if (target_top != drafts[i]) {
            if (getenv("DS4_MTP_SPEC_LOG")) {
                fprintf(stderr,
                        "ds4: mtp spec seq miss at=%d draft=%d base=%d drafted=%d accepted=%d\n",
                        i,
                        drafts[i],
                        target_top,
                        draft_n,
                        n_accept);
            }
            break;
        }
        if (!metal_graph_eval_token_raw_swa_top(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                drafts[i],
                                                (uint32_t)s->checkpoint.len,
                                                &target_top,
                                                NULL,
                                                false,
                                                NULL,
                                                false))
        {
            snprintf(err, errlen, "%s decode failed", ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return -1;
        }
        token_vec_push(&s->checkpoint, drafts[i]);
        logits_on_host = false;
        accepted[n_accept++] = drafts[i];
        verified++;
        if (drafts[i] == eos_token) break;
    }
    if (verified > 0 && !logits_on_host) {
        if (ds4_gpu_tensor_read(metal_graph_logits(&s->graph),
                                  0,
                                  s->logits,
                                  (uint64_t)DS4_N_VOCAB * sizeof(s->logits[0])) == 0)
        {
            snprintf(err, errlen, "%s logits readback failed", ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return -1;
        }
        logits_on_host = true;
    }
    (void)logits_on_host;
    DS4_MTP_KEEP_ACCEPTED(verified);
    if (verified > 0) ds4_session_dspark_capture_note_checkpoint(s);
#undef DS4_MTP_KEEP_ACCEPTED
    if (mtp_timing) {
        fprintf(stderr,
                "ds4: mtp timing seq drafted=%d verified=%d draft=%.3f ms verify=%.3f ms total=%.3f ms\n",
                draft_n,
                verified,
                (mtp_t_after_draft - mtp_t0) * 1000.0,
                (now_sec() - seq_t0) * 1000.0,
                (now_sec() - mtp_t0) * 1000.0);
    }
    if (getenv("DS4_MTP_SPEC_LOG")) {
        if (verified == draft_n) {
            fprintf(stderr,
                    "ds4: mtp spec seq accept drafted=%d accepted=%d\n",
                    draft_n,
                    n_accept);
        } else {
            fprintf(stderr,
                    "ds4: mtp spec seq partial drafted=%d verified=%d accepted=%d\n",
                    draft_n,
                    verified,
                    n_accept);
        }
    }
    return n_accept;
#endif
}

void ds4_session_invalidate(ds4_session *s) {
    if (!s) return;
    if (ds4_session_tp_leader(s) &&
        !ds4_tp_failed(s->engine->tp.ctx)) {
        (void)ds4_tp_send_invalidate(s->engine->tp.ctx, s->tp_session_id);
    }
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    ds4_session_dspark_capture_invalidate(s);
#ifndef DS4_NO_GPU
    ds4_session_glm_reset_dense_cache(s);
#endif
}

void ds4_session_rewind(ds4_session *s, int pos) {
    if (ds4_session_tp_leader(s) &&
        !ds4_tp_failed(s->engine->tp.ctx)) {
        (void)ds4_tp_send_rewind(s->engine->tp.ctx, s->tp_session_id, pos);
    }
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->checkpoint.len = pos;
    s->mtp_draft_valid = false;
    ds4_session_dspark_capture_invalidate(s);
#ifndef DS4_NO_GPU
    ds4_session_glm_cap_dense_cache(s);
#endif
}

int ds4_session_pos(ds4_session *s) {
    return s->checkpoint.len;
}

int ds4_session_ctx(ds4_session *s) {
    return s->ctx_size;
}

int ds4_session_prefill_cap(ds4_session *s) {
    return s ? (int)s->prefill_cap : 0;
}
