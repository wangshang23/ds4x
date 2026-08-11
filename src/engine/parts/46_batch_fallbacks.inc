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
