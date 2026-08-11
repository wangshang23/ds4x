
static int ds4_sessions_eval_batch_cuda(ds4_decode_item *items, int count,
                                        char *err, size_t errlen);
static int ds4_sessions_eval_batch_with_prefill_cuda(
        ds4_decode_item *items,
        int count,
        ds4_session *prefill_session,
        const ds4_tokens *prefill_prompt,
        char *err,
        size_t errlen);

int ds4_sessions_eval_batch(ds4_decode_item *items, int count,
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
        if (items[i].token < 0 || items[i].token >= (int)DS4_N_VOCAB) {
            if (err && errlen) {
                snprintf(err, errlen, "decode batch item %d has an invalid token", i);
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
        if (s->checkpoint.len >= s->ctx_size) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d reached its context limit", i);
            }
            return 1;
        }
    }

#ifndef DS4_NO_GPU
    if (e->backend == DS4_BACKEND_CUDA) {
        return ds4_sessions_eval_batch_cuda(items, count, err, errlen);
    }
    if (ds4_sessions_eval_batch_metal_supported(items, count, e)) {
        return ds4_sessions_eval_batch_metal(items, count, e, err, errlen);
    }
#endif

    /* Preserve logical all-or-nothing behavior even on the serialized path.
     * A failure can leave earlier members advanced, so force every member to
     * rebuild before it is used again. */
    for (int i = 0; i < count; i++) {
        if (ds4_session_eval(items[i].session, items[i].token,
                             err, errlen) != 0) {
            for (int j = 0; j < count; j++) {
                ds4_session_invalidate(items[j].session);
            }
            return 1;
        }
    }
    return 0;
}

int ds4_sessions_eval_batch_with_prefill(
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
    if (!prefill_session->checkpoint_valid ||
        prefill_prompt->len <= prefill_session->checkpoint.len ||
        prefill_prompt->len > prefill_session->ctx_size ||
        !ds4_tokens_starts_with(prefill_prompt, &prefill_session->checkpoint)) {
        if (err && errlen) {
            snprintf(err, errlen,
                     "mixed prefill must extend a valid session checkpoint");
        }
        return 1;
    }
    for (int i = 0; i < count; i++) {
        ds4_session *s = items[i].session;
        if (!s || s == prefill_session ||
            s->engine != prefill_session->engine ||
            !s->checkpoint_valid || s->checkpoint.len >= s->ctx_size ||
            items[i].token < 0 || items[i].token >= (int)DS4_N_VOCAB) {
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
    if (prefill_session->engine->backend == DS4_BACKEND_CUDA) {
        return ds4_sessions_eval_batch_with_prefill_cuda(
                items, count, prefill_session, prefill_prompt, err, errlen);
    }
    if (ds4_sessions_eval_batch_with_prefill_metal_supported(
                items, count, prefill_session, prefill_prompt)) {
        return ds4_sessions_eval_batch_with_prefill_metal(
                items, count, prefill_session, prefill_prompt, err, errlen);
    }
#endif

    int rc = ds4_session_sync(prefill_session, prefill_prompt, err, errlen);
    if (rc != 0) return rc;
    rc = ds4_sessions_eval_batch(items, count, err, errlen);
    if (rc != 0) ds4_session_invalidate(prefill_session);
    return rc;
}

#ifndef DS4_NO_GPU
static int ds4_session_eval_dspark_speculative_argmax(
        ds4_session *s,
        int          n_accept,
        int          max_tokens,
        int          eos_token,
        int         *accepted,
        int          accepted_cap,
        char        *err,
        size_t       errlen) {
    const bool spec_log = getenv("DS4_DSPARK_SPEC_LOG") != NULL;
    const bool stats_enabled = s && ds4_dspark_stats_enabled();
    const bool scheduler_enabled = s && ds4_dspark_scheduler_enabled();
    const double stats_t0 =
        (stats_enabled ||
         (scheduler_enabled && ds4_dspark_scheduler_timing_enabled()))
            ? now_sec() : 0.0;
#define DS4_DSPARK_STATS_FINISH() do {                                      \
        if (stats_enabled) {                                                \
            s->dspark_stats.total_ms += (now_sec() - stats_t0) * 1000.0;    \
        }                                                                   \
    } while (0)
#define DS4_DSPARK_SCHED_EXTRA_MS()                                         \
    ((scheduler_enabled && stats_t0 != 0.0) ?                                \
     s->dspark_last_propose_ms + (now_sec() - stats_t0) * 1000.0 : 0.0)
    if (stats_enabled) {
        s->dspark_stats.cycles++;
        if (n_accept > 0) s->dspark_stats.first_tokens++;
    }
    if (spec_log) {
        fprintf(stderr,
                "ds4: DSpark spec enter accepted=%d max=%d valid=%d len=%u pos=%d\n",
                n_accept,
                max_tokens,
                s ? (s->dspark_draft_valid ? 1 : 0) : 0,
                s ? s->dspark_draft_len : 0,
                s ? s->checkpoint.len : -1);
    }
    if (!s || !s->dspark_draft_valid || s->dspark_draft_len == 0) {
        if (stats_enabled) {
            s->dspark_stats.no_draft++;
            ds4_dspark_stats_note_len(s->dspark_stats.accepted_len_hist, 0);
        }
        if (s) {
            ds4_session_dspark_scheduler_note(
                    s, 0, true, DS4_DSPARK_SCHED_EXTRA_MS());
        }
        if (spec_log) {
            fprintf(stderr, "ds4: DSpark spec skip no-draft\n");
        }
        DS4_DSPARK_STATS_FINISH();
        return n_accept;
    }

    int draft_n = (int)s->dspark_draft_len;
    if (draft_n > max_tokens - n_accept) draft_n = max_tokens - n_accept;
    if (draft_n > accepted_cap - n_accept) draft_n = accepted_cap - n_accept;
    int room = s->ctx_size - s->checkpoint.len;
    if (draft_n > room - 1) draft_n = room - 1;
    if (draft_n <= 0) {
        s->dspark_draft_valid = false;
        s->dspark_draft_len = 0;
        if (stats_enabled) {
            s->dspark_stats.no_room++;
            ds4_dspark_stats_note_len(s->dspark_stats.accepted_len_hist, 0);
        }
        if (spec_log) {
            fprintf(stderr, "ds4: DSpark spec skip no-room\n");
        }
        DS4_DSPARK_STATS_FINISH();
        return n_accept;
    }

    int drafts[DS4_DSPARK_MAX_BLOCK_SIZE];
    for (int i = 0; i < draft_n; i++) {
        drafts[i] = s->dspark_draft_tokens[i];
        if (drafts[i] < 0 || drafts[i] >= (int)DS4_N_VOCAB) {
            s->dspark_draft_valid = false;
            s->dspark_draft_len = 0;
            if (stats_enabled) {
                s->dspark_stats.invalid_draft++;
                ds4_dspark_stats_note_len(s->dspark_stats.accepted_len_hist, 0);
            }
            if (spec_log) {
                fprintf(stderr,
                        "ds4: DSpark spec skip invalid-draft index=%d token=%d\n",
                        i,
                        drafts[i]);
            }
            DS4_DSPARK_STATS_FINISH();
            return n_accept;
        }
    }
    if (stats_enabled) {
        s->dspark_stats.proposed_tokens += (uint64_t)draft_n;
        ds4_dspark_stats_note_len(s->dspark_stats.draft_len_hist,
                                  (uint32_t)draft_n);
    }
    s->dspark_draft_valid = false;
    s->dspark_draft_len = 0;

    const int target_top = sample_argmax(s->logits, DS4_N_VOCAB);
    if (target_top != drafts[0]) {
        if (stats_enabled) {
            s->dspark_stats.first_misses++;
            ds4_dspark_stats_note_len(s->dspark_stats.accepted_len_hist, 0);
        }
        ds4_session_dspark_scheduler_note(
                s, 0, false, DS4_DSPARK_SCHED_EXTRA_MS());
        if (spec_log) {
            fprintf(stderr,
                    "ds4: DSpark spec miss first draft=%d base=%d\n",
                    drafts[0],
                    target_top);
        }
        DS4_DSPARK_STATS_FINISH();
        return n_accept;
    }
    if (drafts[0] == eos_token) draft_n = 1;

    ds4_engine *e = s->engine;
    ds4_spec_frontier frontier;
    memset(&frontier, 0, sizeof(frontier));
    int row_tops_buf[DS4_DSPARK_MAX_BLOCK_SIZE];
    int *row_tops = draft_n > 1 ? row_tops_buf : NULL;
    float *row_logits = s->spec_row_logits;
    const int start = s->checkpoint.len;
    const double snapshot_t0 = stats_enabled ? now_sec() : 0.0;
    bool have_frontier = spec_frontier_snapshot(&frontier, s);
    if (stats_enabled) {
        s->dspark_stats.snapshot_ms += (now_sec() - snapshot_t0) * 1000.0;
    }
    bool ok = have_frontier && row_logits && (draft_n <= 1 || row_tops);
    bool verifier_may_have_mutated = false;
    bool tp_verify_sent = false;
    if (ok && ds4_session_tp_leader(s)) {
        /* Announce the block before mutating anything: the worker runs its
         * half of the verify and then waits for our commit decision. */
        if (!ds4_tp_send_verify(e->tp.ctx, s->tp_session_id,
                                drafts, (uint32_t)draft_n)) {
            snprintf(err, errlen, "tp: verify send failed");
            spec_frontier_free(&frontier);
            DS4_DSPARK_STATS_FINISH();
            return -1;
        }
        tp_verify_sent = true;
    }
    if (ok) {
        for (int i = 0; i < draft_n; i++) token_vec_push(&s->checkpoint, drafts[i]);
        verifier_may_have_mutated = true;
        ds4_verify_suffix_timing verify_timing;
        const double verify_t0 = stats_enabled ? now_sec() : 0.0;
        ok = metal_graph_verify_suffix_tops(&s->graph,
                                            &e->model,
                                            &e->weights,
                                            &s->checkpoint,
                                            (uint32_t)start,
                                            (uint32_t)draft_n,
                                            draft_n > 1 &&
                                                draft_n <=
                                                    (int)DS4_SPEC_PREFIX_SLOTS + 1,
                                            true,
                                            row_tops,
                                            NULL,
                                            stats_enabled ? &verify_timing : NULL);
        if (stats_enabled) {
            s->dspark_stats.verify_ms += (now_sec() - verify_t0) * 1000.0;
            s->dspark_stats.verify_upload_ms += verify_timing.upload_ms;
            s->dspark_stats.verify_layer_ms += verify_timing.layer_ms;
            s->dspark_stats.verify_head_ms += verify_timing.head_ms;
            s->dspark_stats.verify_read_ms += verify_timing.read_ms;
            if (verify_timing.fused_head) {
                s->dspark_stats.verifier_fused_head++;
            }
        }
    }

    int commit_drafts = 0;
    if (ok) {
        commit_drafts = 1;
        for (int i = 1; i < draft_n; i++) {
            if (row_tops[i - 1] != drafts[i]) break;
            commit_drafts++;
        }
    }

    /* The batched verifier and ordinary one-token decode execute the same
     * model graph with different floating-point reduction orders. Keep the
     * verifier state when it can be committed safely: DSpark is an opt-in
     * execution mode and does not promise byte-identical output to ordinary
     * decode. A failed direct commit still falls back to rollback and replay. */

    bool final_logits_ok = false;
    if (ok && commit_drafts == draft_n) {
        const double read_t0 = stats_enabled ? now_sec() : 0.0;
        final_logits_ok = metal_graph_read_spec_logits_row(
                &s->graph, (uint32_t)(draft_n - 1), row_logits);
        if (stats_enabled) {
            s->dspark_stats.verify_read_ms +=
                (now_sec() - read_t0) * 1000.0;
        }
    }

    if (ok && commit_drafts == draft_n && final_logits_ok) {
        if (tp_verify_sent &&
            !ds4_tp_send_verify_commit(e->tp.ctx, 1, 0)) {
            snprintf(err, errlen, "tp: verify commit send failed");
            s->checkpoint_valid = false;
            spec_frontier_free(&frontier);
            DS4_DSPARK_STATS_FINISH();
            return -1;
        }
        memcpy(s->logits, row_logits,
               (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        int emitted_drafts = 0;
        for (int i = 0; i < draft_n && n_accept < accepted_cap; i++) {
            accepted[n_accept++] = drafts[i];
            emitted_drafts++;
            if (drafts[i] == eos_token) break;
        }
        s->checkpoint_valid = true;
        ds4_session_dspark_capture_note_checkpoint(s);
        if (stats_enabled) {
            s->dspark_stats.full_accepts++;
            s->dspark_stats.direct_full_commits++;
            s->dspark_stats.accepted_draft_tokens +=
                (uint64_t)emitted_drafts;
            ds4_dspark_stats_note_len(s->dspark_stats.accepted_len_hist,
                                      (uint32_t)emitted_drafts);
        }
        ds4_session_dspark_scheduler_note(
                s, (uint32_t)emitted_drafts, false,
                DS4_DSPARK_SCHED_EXTRA_MS());
        if (spec_log) {
            fprintf(stderr,
                    "ds4: DSpark spec direct-full drafted=%d accepted=%d\n",
                    draft_n,
                    n_accept);
        }
        spec_frontier_free(&frontier);
        DS4_DSPARK_STATS_FINISH();
        return n_accept;
    }

    /* Prefix snapshots make partial accepts cheap on one host. TP workers do
     * not yet receive these intermediate snapshots, so TP partial accepts use
     * the existing mirrored replay fallback. */
    if (ok && !tp_verify_sent &&
        commit_drafts > 0 && commit_drafts < draft_n &&
        commit_drafts <= (int)DS4_SPEC_PREFIX_SLOTS) {
        const double read_t0 = stats_enabled ? now_sec() : 0.0;
        bool prefix_ok = metal_graph_read_spec_logits_row(
                &s->graph, (uint32_t)(commit_drafts - 1), row_logits);
        if (stats_enabled) {
            s->dspark_stats.verify_read_ms +=
                (now_sec() - read_t0) * 1000.0;
        }
        s->checkpoint.len = start;
        ds4_session_dspark_capture_invalidate(s);
        if (prefix_ok) {
            prefix_ok = spec_frontier_commit_prefix(
                    s, (uint32_t)commit_drafts);
        }
        if (prefix_ok) {
            memcpy(s->logits, row_logits,
                   (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
            int emitted_drafts = 0;
            for (int i = 0; i < commit_drafts && n_accept < accepted_cap; i++) {
                token_vec_push(&s->checkpoint, drafts[i]);
                accepted[n_accept++] = drafts[i];
                emitted_drafts++;
                if (drafts[i] == eos_token) break;
            }
            s->checkpoint_valid = true;
            ds4_session_dspark_capture_note_checkpoint(s);
            if (stats_enabled) {
                s->dspark_stats.partial_accepts++;
                s->dspark_stats.direct_partial_commits++;
                s->dspark_stats.accepted_draft_tokens +=
                    (uint64_t)emitted_drafts;
                ds4_dspark_stats_note_len(
                        s->dspark_stats.accepted_len_hist,
                        (uint32_t)emitted_drafts);
            }
            ds4_session_dspark_scheduler_note(
                    s, (uint32_t)emitted_drafts, false,
                    DS4_DSPARK_SCHED_EXTRA_MS());
            if (spec_log) {
                fprintf(stderr,
                        "ds4: DSpark spec direct-partial drafted=%d committed=%d accepted=%d\n",
                        draft_n,
                        emitted_drafts,
                        n_accept);
            }
            spec_frontier_free(&frontier);
            DS4_DSPARK_STATS_FINISH();
            return n_accept;
        }
    }

    if (verifier_may_have_mutated) {
        s->checkpoint.len = start;
        ds4_session_dspark_capture_invalidate(s);
        if (!have_frontier || !spec_frontier_restore(&frontier, s)) {
            if (tp_verify_sent)
                (void)ds4_tp_send_verify_commit(e->tp.ctx, 0, 0);
            snprintf(err, errlen, "DSpark verifier rollback failed");
            s->checkpoint_valid = false;
            if (stats_enabled) {
                s->dspark_stats.verifier_errors++;
                ds4_dspark_stats_note_len(s->dspark_stats.accepted_len_hist, 0);
            }
            spec_frontier_free(&frontier);
            DS4_DSPARK_STATS_FINISH();
            return -1;
        }
    }

    if (!ok) {
        if (tp_verify_sent &&
            !ds4_tp_send_verify_commit(e->tp.ctx, 0, 0)) {
            snprintf(err, errlen, "tp: verify commit send failed");
            spec_frontier_free(&frontier);
            DS4_DSPARK_STATS_FINISH();
            return -1;
        }
        if (stats_enabled) {
            s->dspark_stats.verifier_unavailable++;
            ds4_dspark_stats_note_len(s->dspark_stats.accepted_len_hist, 0);
        }
        if (spec_log) {
            fprintf(stderr,
                    "ds4: DSpark spec verifier unavailable frontier=%d row_logits=%d row_tops=%d mutated=%d\n",
                    have_frontier ? 1 : 0,
                    row_logits ? 1 : 0,
                    row_tops || draft_n <= 1 ? 1 : 0,
                    verifier_may_have_mutated ? 1 : 0);
        }
        spec_frontier_free(&frontier);
        DS4_DSPARK_STATS_FINISH();
        return n_accept;
    }

    /* Precompute the exact replay count (cap + eos cuts) so the worker can
     * run the same gated replay evals in lockstep. */
    int replay_budget = commit_drafts;
    if (replay_budget > accepted_cap - n_accept)
        replay_budget = accepted_cap - n_accept;
    if (replay_budget < 0) replay_budget = 0;
    for (int i = 0; i < replay_budget; i++) {
        if (drafts[i] == eos_token) { replay_budget = i + 1; break; }
    }
    if (tp_verify_sent &&
        !ds4_tp_send_verify_commit(e->tp.ctx, 0, replay_budget)) {
        snprintf(err, errlen, "tp: verify commit send failed");
        spec_frontier_free(&frontier);
        DS4_DSPARK_STATS_FINISH();
        return -1;
    }
    const double replay_t0 = stats_enabled ? now_sec() : 0.0;
    int replayed_drafts = 0;
    if (stats_enabled && replay_budget > 0) {
        s->dspark_stats.replay_fallbacks++;
    }
    for (int i = 0; i < replay_budget; i++) {
        ok = metal_graph_eval_token_raw_swa(&s->graph,
                                            &e->model,
                                            &e->weights,
                                            drafts[i],
                                            (uint32_t)s->checkpoint.len,
                                            row_logits);
        if (!ok) {
            snprintf(err, errlen, "%s decode failed", ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            if (stats_enabled) {
                s->dspark_stats.verifier_errors++;
                s->dspark_stats.replay_ms += (now_sec() - replay_t0) * 1000.0;
                ds4_dspark_stats_note_len(s->dspark_stats.accepted_len_hist, 0);
            }
            spec_frontier_free(&frontier);
            DS4_DSPARK_STATS_FINISH();
            return -1;
        }
        token_vec_push(&s->checkpoint, drafts[i]);
        accepted[n_accept++] = drafts[i];
        replayed_drafts++;
        if (drafts[i] == eos_token) break;
    }
    if (stats_enabled) {
        s->dspark_stats.replay_ms += (now_sec() - replay_t0) * 1000.0;
    }
    /* Vocab-split head: the last replay eval produced only our logits half;
     * merge the worker's before installing them as the session logits. */
    if (replayed_drafts > 0 && tp_verify_sent && e->tp.vocab_split) {
        const uint32_t vhalf = (uint32_t)DS4_N_VOCAB / 2u;
        if (!ds4_tp_recv_logits_half(e->tp.ctx, row_logits + vhalf, vhalf)) {
            snprintf(err, errlen, "tp: replay logits half missing");
            s->checkpoint_valid = false;
            spec_frontier_free(&frontier);
            DS4_DSPARK_STATS_FINISH();
            return -1;
        }
    }
    if (replayed_drafts > 0) {
        memcpy(s->logits, row_logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        s->checkpoint_valid = true;
        ds4_session_dspark_capture_note_checkpoint(s);
        if (stats_enabled) {
            if (replayed_drafts == draft_n) s->dspark_stats.full_accepts++;
            else s->dspark_stats.partial_accepts++;
            s->dspark_stats.accepted_draft_tokens += (uint64_t)replayed_drafts;
            ds4_dspark_stats_note_len(s->dspark_stats.accepted_len_hist,
                                      (uint32_t)replayed_drafts);
        }
    } else if (stats_enabled) {
        ds4_dspark_stats_note_len(s->dspark_stats.accepted_len_hist, 0);
    }
    ds4_session_dspark_scheduler_note(
            s,
            (uint32_t)replayed_drafts,
            false,
            DS4_DSPARK_SCHED_EXTRA_MS());
    if (spec_log) {
        fprintf(stderr,
                "ds4: DSpark spec partial drafted=%d verified=%d accepted=%d\n",
                draft_n,
                commit_drafts,
                n_accept);
    }
    spec_frontier_free(&frontier);
    DS4_DSPARK_STATS_FINISH();
#undef DS4_DSPARK_SCHED_EXTRA_MS
#undef DS4_DSPARK_STATS_FINISH
    return n_accept;
}
#endif

/* TP worker side of a mirrored speculative-verify block. Runs its half of the
 * same batch verify as the leader, including per-layer combine gates, purely
 * for KV, compressor, and indexer side effects; then it obeys the commit
 * frame: keep the pushed rows, or roll back and replay the accepted prefix
 * through the gated single-token decode in lockstep with the leader. */
int ds4_session_tp_spec_cycle(ds4_session *s, const int *drafts, int draft_n,
                              char *err, size_t errlen) {
#ifdef DS4_NO_GPU
    (void)s; (void)drafts; (void)draft_n;
    snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_engine *e = s ? s->engine : NULL;
    if (!e || !e->tp.active || e->tp.rank != 1) {
        snprintf(err, errlen, "tp: spec cycle outside worker mode");
        return 1;
    }
    if (draft_n <= 0 || draft_n > DS4_DSPARK_MAX_BLOCK_SIZE) {
        snprintf(err, errlen, "tp: bad verify block size %d", draft_n);
        return 1;
    }
    for (int i = 0; i < draft_n; i++) {
        if (drafts[i] < 0 || drafts[i] >= (int)DS4_N_VOCAB) {
            snprintf(err, errlen, "tp: bad draft token %d", drafts[i]);
            return 1;
        }
    }
    if (draft_n > s->ctx_size - s->checkpoint.len) {
        snprintf(err, errlen, "tp: verify block beyond context");
        return 1;
    }
    ds4_spec_frontier frontier;
    memset(&frontier, 0, sizeof(frontier));
    const int start = s->checkpoint.len;
    if (!spec_frontier_snapshot(&frontier, s)) {
        snprintf(err, errlen, "tp: frontier snapshot failed");
        return 1;
    }
    for (int i = 0; i < draft_n; i++) token_vec_push(&s->checkpoint, drafts[i]);
    int row_tops[DS4_DSPARK_MAX_BLOCK_SIZE];
    bool ok = metal_graph_verify_suffix_tops(&s->graph,
                                             &e->model,
                                             &e->weights,
                                             &s->checkpoint,
                                             (uint32_t)start,
                                             (uint32_t)draft_n,
                                             false,
                                             false,
                                             draft_n > 1 ? row_tops : NULL,
                                             NULL,
                                             NULL);
    int32_t full_accept = 0, replay_n = 0;
    if (!ds4_tp_recv_verify_commit(e->tp.ctx, &full_accept, &replay_n)) {
        spec_frontier_free(&frontier);
        snprintf(err, errlen, "tp: verify commit frame missing");
        return 1;
    }
    if (full_accept) {
        spec_frontier_free(&frontier);
        if (!ok) {
            /* The leader committed a block our verify failed to apply:
             * the states have diverged and lockstep cannot continue. */
            snprintf(err, errlen, "tp: worker verify failed on accepted block");
            s->checkpoint_valid = false;
            return 1;
        }
        s->checkpoint_valid = true;
        return 0;
    }
    s->checkpoint.len = start;
    if (!spec_frontier_restore(&frontier, s)) {
        spec_frontier_free(&frontier);
        snprintf(err, errlen, "tp: verify rollback failed");
        s->checkpoint_valid = false;
        return 1;
    }
    spec_frontier_free(&frontier);
    if (replay_n > draft_n) replay_n = draft_n;
    float *logits = s->spec_row_logits;
    float *scratch = NULL;
    if (!logits) {
        scratch = malloc((size_t)DS4_N_VOCAB * sizeof(float));
        if (!scratch) {
            snprintf(err, errlen, "tp: replay logits alloc failed");
            return 1;
        }
        logits = scratch;
    }
    for (int i = 0; i < replay_n; i++) {
        if (!metal_graph_eval_token_raw_swa(&s->graph,
                                            &e->model,
                                            &e->weights,
                                            drafts[i],
                                            (uint32_t)s->checkpoint.len,
                                            logits)) {
            free(scratch);
            snprintf(err, errlen, "tp: replay decode failed");
            s->checkpoint_valid = false;
            return 1;
        }
        token_vec_push(&s->checkpoint, drafts[i]);
    }
    if (replay_n > 0) {
        s->checkpoint_valid = true;
        if (e->tp.vocab_split) {
            const uint32_t vhalf = (uint32_t)DS4_N_VOCAB / 2u;
            if (!ds4_tp_send_logits_half(e->tp.ctx, logits + vhalf, vhalf)) {
                free(scratch);
                snprintf(err, errlen, "tp: replay logits half send failed");
                return 1;
            }
        }
    }
    free(scratch);
    return 0;
#endif
}

