void ds4_engine_close(ds4_engine *e) {
    if (!e) return;
    weights_free(&e->weights);
    vocab_free(&e->vocab);
    ds4_threads_shutdown();
    if (e->support_model.map) model_close(&e->support_model);
    if (e->model.map) model_close(&e->model);
    ds4_gpu_cleanup();
    ds4_memory_lock_release(&e->simulated_memory);
    ds4_release_instance_lock();
    free(e->directional_steering_file);
    free(e);
}

static bool ds4_dspark_stats_enabled(void) {
    const char *env = getenv("DS4_DSPARK_STATS");
    return env && env[0] && strcmp(env, "0") != 0;
}

static void ds4_format_len_hist(char *buf, size_t buflen,
                                const uint64_t *hist) {
    if (!buf || buflen == 0 || !hist) return;
    size_t off = 0;
    buf[0] = '\0';
    for (uint32_t i = 0; i <= DS4_DSPARK_MAX_BLOCK_SIZE; i++) {
        if (hist[i] == 0) continue;
        const int n = snprintf(buf + off, buflen - off, "%s%u:%llu",
                               off ? "," : "", i,
                               (unsigned long long)hist[i]);
        if (n < 0 || (size_t)n >= buflen - off) break;
        off += (size_t)n;
    }
    if (off == 0) snprintf(buf, buflen, "none");
}

static void ds4_session_print_dspark_stats(const ds4_session *s) {
    if (!s || !ds4_dspark_stats_enabled()) return;
    const ds4_dspark_spec_stats *st = &s->dspark_stats;
    if (st->cycles == 0 && st->propose_ms == 0.0) return;
    char draft_hist[192];
    char accept_hist[192];
    ds4_format_len_hist(draft_hist, sizeof(draft_hist), st->draft_len_hist);
    ds4_format_len_hist(accept_hist, sizeof(accept_hist),
                        st->accepted_len_hist);
    const double accept_rate = st->proposed_tokens
        ? 100.0 * (double)st->accepted_draft_tokens /
          (double)st->proposed_tokens
        : 0.0;
    fprintf(stderr,
            "ds4: DSpark cycles=%llu proposed=%llu accepted=%llu "
            "accept_rate=%.2f%% full=%llu partial=%llu "
            "propose=%.3fms verify=%.3fms replay=%.3fms "
            "draft_hist=%s accept_hist=%s\n",
            (unsigned long long)st->cycles,
            (unsigned long long)st->proposed_tokens,
            (unsigned long long)st->accepted_draft_tokens,
            accept_rate,
            (unsigned long long)st->full_accepts,
            (unsigned long long)st->partial_accepts,
            st->propose_ms,
            st->verify_ms,
            st->replay_ms,
            draft_hist,
            accept_hist);
}

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0 ||
        e->backend != DS4_BACKEND_CUDA || !e->metal_ready) {
        return 1;
    }

    ds4_session *s = xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->prefill_cap = metal_graph_prefill_cap_for_prompt(
            ctx_size, e->prefill_chunk);
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(
            ctx_size, s->prefill_cap);
    const ds4_layer_weights *shape_layer =
        weights_first_bound_layer(&e->weights);
    if (!shape_layer) {
        fprintf(stderr, "ds4: no transformer layers are loaded\n");
        free(s);
        return 1;
    }

    const bool need_spec_verifier =
        e->support_kind == DS4_SUPPORT_DSPARK && e->dspark;
    s->graph.dspark_exec_tier = 0;
    if (!metal_graph_alloc_raw_cap(&s->graph,
                                   &e->weights,
                                   shape_layer,
                                   raw_cap,
                                   (uint32_t)ctx_size,
                                   s->prefill_cap,
                                   need_spec_verifier)) {
        free(s);
        return 1;
    }
    s->graph.quality = e->quality;
    s->graph.power_percent = (uint32_t)e->power_percent;

    if (!metal_graph_load_directional_steering(
                &s->graph,
                e->directional_steering_file,
                e->directional_steering_attn_scale,
                e->directional_steering_ffn_scale)) {
        metal_graph_free(&s->graph);
        free(s);
        return 1;
    }
    if (e->support_kind == DS4_SUPPORT_DSPARK &&
        !metal_graph_configure_dspark_capture(&s->graph,
                                              &e->dspark_weights)) {
        fprintf(stderr, "ds4: failed to configure DSpark target capture\n");
        metal_graph_free(&s->graph);
        free(s);
        return 1;
    }

    s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
    s->sample_probs =
        xmalloc((size_t)DS4_N_VOCAB * sizeof(s->sample_probs[0]));
    if (need_spec_verifier) {
        s->spec_row_logits =
            xmalloc((size_t)DS4_N_VOCAB * sizeof(s->spec_row_logits[0]));
    }
    if (e->support_kind == DS4_SUPPORT_DSPARK && e->dspark) {
        const uint64_t feature_count =
            (uint64_t)DS4_N_EMBD + e->dspark_weights.markov_rank;
        s->dspark_markov_bias =
            xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
        s->dspark_conf_features =
            xmalloc((size_t)feature_count * sizeof(float));
        s->dspark_conf_features_cap = (size_t)feature_count;
    }
    *out = s;
    return 0;
}

void ds4_session_free(ds4_session *s) {
    if (!s) return;
    ds4_session_print_dspark_stats(s);
    metal_graph_free(&s->graph);
    token_vec_free(&s->checkpoint);
    token_vec_free(&s->greedy_splitkv_segment);
    free(s->logits);
    free(s->sample_probs);
    free(s->spec_row_logits);
    free(s->dspark_markov_bias);
    free(s->dspark_conf_features);
    free(s);
}

int ds4_session_power(ds4_session *s) {
    return s && s->engine ? s->engine->power_percent : 100;
}

int ds4_session_set_power(ds4_session *s, int power_percent) {
    if (!s || !s->engine || power_percent < 1 || power_percent > 100) return 1;
    s->engine->power_percent = power_percent;
    s->graph.power_percent = (uint32_t)power_percent;
    return 0;
}

void ds4_session_set_progress(ds4_session *s,
                              ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}

void ds4_session_set_display_progress(ds4_session *s,
                                      ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->display_progress = fn;
    s->display_progress_ud = ud;
}

void ds4_session_set_cancel(ds4_session *s,
                            ds4_session_cancel_fn fn, void *ud) {
    if (!s) return;
    s->cancel = fn;
    s->cancel_ud = ud;
}

static bool ds4_session_cancelled(ds4_session *s) {
    return s && s->cancel && s->cancel(s->cancel_ud);
}

static bool ds4_session_cancelled_cb(void *ud) {
    return ds4_session_cancelled((ds4_session *)ud);
}

void ds4_session_report_progress(ds4_session *s, const char *event,
                                 int current, int total) {
    if (s && s->progress) s->progress(s->progress_ud, event, current, total);
}

typedef struct {
    ds4_session *session;
    const ds4_tokens *prompt;
    ds4_session_progress_fn user;
    void *user_ud;
} ds4_sync_progress;

static void ds4_session_note_prefill_progress(void *ud, const char *event,
                                              int current, int total) {
    ds4_sync_progress *p = ud;
    if (!p || !p->session || !p->prompt) return;
    if (!strcmp(event, "prefill_chunk") && current > 0 &&
        current <= p->prompt->len) {
        p->session->checkpoint.len = 0;
        for (int i = 0; i < current; i++) {
            token_vec_push(&p->session->checkpoint, p->prompt->v[i]);
        }
        p->session->checkpoint_valid = true;
        ds4_session_dspark_capture_note_checkpoint(p->session);
    }
    if (p->user) p->user(p->user_ud, event, current, total);
}

int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt,
                     char *err, size_t errlen) {
    if (!s || !prompt || prompt->len <= 0) {
        snprintf(err, errlen, "missing or empty prompt");
        return 1;
    }
    if (prompt->len >= s->ctx_size) {
        snprintf(err, errlen,
                 "prompt length %d exceeds context %d",
                 prompt->len, s->ctx_size);
        return 1;
    }
    if (ds4_session_cancelled(s)) {
        snprintf(err, errlen, "interrupted");
        return DS4_SESSION_SYNC_INTERRUPTED;
    }

    ds4_engine *e = s->engine;
    if (s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        ds4_tokens_starts_with(prompt, &s->checkpoint)) {
        const int suffix = prompt->len - s->checkpoint.len;
        const uint32_t resume_min = metal_graph_resume_prefill_min_tokens();
        if (suffix > 0 && (uint32_t)suffix >= resume_min) {
            bool cancelled = false;
            ds4_sync_progress progress = {
                .session = s,
                .prompt = prompt,
                .user = s->progress,
                .user_ud = s->progress_ud,
            };
            const bool ok = metal_graph_prefill_chunked_range(
                    &s->graph, &e->model, &e->weights, prompt,
                    (uint32_t)s->checkpoint.len, (uint32_t)suffix,
                    s->logits, false,
                    ds4_session_note_prefill_progress, &progress,
                    s->display_progress, s->display_progress_ud,
                    ds4_session_cancelled_cb, s, &cancelled);
            if (cancelled) {
                snprintf(err, errlen, "interrupted");
                return DS4_SESSION_SYNC_INTERRUPTED;
            }
            if (!ok) {
                snprintf(err, errlen, "CUDA resumed prefill failed");
                s->checkpoint_valid = false;
                return 1;
            }
            ds4_tokens_copy(&s->checkpoint, prompt);
            s->checkpoint_valid = true;
            ds4_session_dspark_capture_note_checkpoint(s);
            return 0;
        }

        for (int i = s->checkpoint.len; i < prompt->len; i++) {
            if (ds4_session_cancelled(s)) {
                snprintf(err, errlen, "interrupted");
                return DS4_SESSION_SYNC_INTERRUPTED;
            }
            if (!metal_graph_eval_token_raw_swa(
                        &s->graph, &e->model, &e->weights,
                        (uint32_t)prompt->v[i],
                        (uint32_t)s->checkpoint.len, s->logits)) {
                snprintf(err, errlen, "CUDA decode extension failed");
                s->checkpoint_valid = false;
                return 1;
            }
            token_vec_push(&s->checkpoint, prompt->v[i]);
            s->checkpoint_valid = true;
            ds4_session_dspark_capture_note_checkpoint(s);
        }
        session_greedy_splitkv_reset(s);
        return 0;
    }

    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    ds4_session_dspark_capture_invalidate(s);
    if (!metal_graph_reset_prefill_state(&s->graph)) {
        snprintf(err, errlen, "CUDA prefill reset failed");
        return 1;
    }

    bool cancelled = false;
    bool ok;
    if (s->prefill_cap < (uint32_t)prompt->len) {
        ds4_sync_progress progress = {
            .session = s,
            .prompt = prompt,
            .user = s->progress,
            .user_ud = s->progress_ud,
        };
        ok = metal_graph_prefill_chunked(
                &s->graph, &e->model, &e->weights,
                prompt, prompt->len, s->logits, false,
                ds4_session_note_prefill_progress, &progress,
                s->display_progress, s->display_progress_ud,
                ds4_session_cancelled_cb, s, &cancelled);
    } else {
        ok = metal_graph_prefill_raw_swa(
                &s->graph, &e->model, &e->weights,
                prompt, prompt->len, s->logits, false,
                s->display_progress, s->display_progress_ud,
                ds4_session_cancelled_cb, s, &cancelled);
    }
    if (cancelled) {
        snprintf(err, errlen, "interrupted");
        s->checkpoint_valid = s->checkpoint.len > 0;
        return DS4_SESSION_SYNC_INTERRUPTED;
    }
    if (!ok) {
        snprintf(err, errlen, "CUDA prefill failed");
        return 1;
    }

    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    ds4_session_dspark_capture_note_checkpoint(s);
    session_greedy_splitkv_reset(s);
    return 0;
}
