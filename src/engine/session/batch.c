#include "engine_internal.h"

/* Batch module. */

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

    if (ok && commit_drafts > 0 && commit_drafts < draft_n &&
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

    int replay_budget = commit_drafts;
    if (replay_budget > accepted_cap - n_accept)
        replay_budget = accepted_cap - n_accept;
    if (replay_budget < 0) replay_budget = 0;
    for (int i = 0; i < replay_budget; i++) {
        if (drafts[i] == eos_token) { replay_budget = i + 1; break; }
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

int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    if (!s || !accepted || max_tokens <= 0 || accepted_cap <= 0) return 0;

    ds4_engine *e = s->engine;
    const bool strict_dspark =
        e->support_kind == DS4_SUPPORT_DSPARK &&
        (e->quality || e->dspark_strict);
    bool prepare_draft =
        !strict_dspark && first_token != eos_token &&
        max_tokens > 1 && accepted_cap > 1;
    bool tail_skip = false;

    if (prepare_draft && e->support_kind == DS4_SUPPORT_DSPARK &&
        ds4_dspark_scheduler_enabled()) {
        const uint32_t tail_min = ds4_dspark_scheduler_tail_min_tokens();
        if (tail_min != 0 && (uint32_t)max_tokens < tail_min) {
            prepare_draft = false;
            tail_skip = true;
            if (ds4_dspark_stats_enabled()) s->dspark_stats.tail_skips++;
            if (getenv("DS4_DSPARK_SPEC_LOG") != NULL) {
                fprintf(stderr,
                        "ds4: DSpark scheduler tail skip max=%d min=%u\n",
                        max_tokens,
                        tail_min);
            }
        }
    }

    if (ds4_session_eval_probe_tp(s, first_token, prepare_draft,
                                  err, errlen) != 0) {
        return -1;
    }

    int n_accept = 0;
    accepted[n_accept++] = first_token;
    if (first_token == eos_token || max_tokens == 1 ||
        n_accept >= accepted_cap || strict_dspark || tail_skip) {
        return n_accept;
    }

    if (e->support_kind == DS4_SUPPORT_DSPARK) {
        return ds4_session_eval_dspark_speculative_argmax(
                s, n_accept, max_tokens, eos_token,
                accepted, accepted_cap, err, errlen);
    }
    return n_accept;
}

void ds4_session_invalidate(ds4_session *s) {
    if (!s) return;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    ds4_session_dspark_capture_invalidate(s);
}

void ds4_session_rewind(ds4_session *s, int pos) {
    if (!s) return;
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->checkpoint.len = pos;
    ds4_session_dspark_capture_invalidate(s);
}

int ds4_session_pos(ds4_session *s) {
    return s ? s->checkpoint.len : 0;
}

int ds4_session_ctx(ds4_session *s) {
    return s ? s->ctx_size : 0;
}

int ds4_session_prefill_cap(ds4_session *s) {
    return s ? (int)s->prefill_cap : 0;
}
