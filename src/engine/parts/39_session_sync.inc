/* Return true when canonicalization would replace already-sampled tokens.
 *
 * A DS4 session checkpoint is more than a token vector: the backend state also
 * contains raw SWA rows, compressed KV rows, indexer rows, and compressor
 * frontiers.  Replacing any part of the live tail requires restoring that whole
 * frontier first.  Extending exactly at the live end is safe; rewriting behind
 * it is not an in-place operation. */
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}

/* Replace the live suffix after a shared prefix.
 *
 * This is used after parsing a generated tool call.  The model may have emitted
 * DSML in an order that is semantically valid but not byte-for-byte equal to the
 * canonical prompt we will see on the next request.  Rewriting only the token
 * checkpoint is not enough: the backend still contains raw and compressed rows
 * for the old suffix.  Until we have a real frontier snapshot at the
 * rewrite point, any replacement behind the live end reports that a rebuild is
 * needed without mutating the session.  The server may still find an older disk KV
 * checkpoint before falling back to a full replay. */
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen) {
    if (!s || !prompt) {
        snprintf(err, errlen, "missing session or prompt");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (prompt->len <= 0) {
        snprintf(err, errlen, "empty prompt");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (prompt->len >= s->ctx_size) {
        snprintf(err, errlen,
                 "prompt length %d exceeds context %d (one token of generation room is required)",
                 prompt->len, s->ctx_size);
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (!s->checkpoint_valid) {
        snprintf(err, errlen, "session has no valid checkpoint");
        return DS4_SESSION_REWRITE_ERROR;
    }
    if (common < 0 || common > s->checkpoint.len || common > prompt->len) {
        snprintf(err, errlen, "invalid rewrite prefix");
        return DS4_SESSION_REWRITE_ERROR;
    }
    for (int i = 0; i < common; i++) {
        if (s->checkpoint.v[i] != prompt->v[i]) {
            snprintf(err, errlen, "rewrite prefix does not match live checkpoint");
            return DS4_SESSION_REWRITE_ERROR;
        }
    }

    if (common == s->checkpoint.len) {
        return ds4_session_sync(s, prompt, err, errlen) == 0 ?
            DS4_SESSION_REWRITE_OK : DS4_SESSION_REWRITE_ERROR;
    }

    if (ds4_session_rewrite_requires_rebuild(s->checkpoint.len, prompt->len, common)) {
        snprintf(err, errlen, "rewrite needs rebuild: common=%d live=%d canonical=%d",
                 common, s->checkpoint.len, prompt->len);
        return DS4_SESSION_REWRITE_REBUILD_NEEDED;
    }

    snprintf(err, errlen, "unexpected canonical rewrite state");
    return DS4_SESSION_REWRITE_ERROR;
}

int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt) {
    if (!s->checkpoint_valid) return 0;
    int n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    int i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}

int ds4_session_argmax(ds4_session *s) {
    return sample_argmax(s->logits, DS4_N_VOCAB);
}

int ds4_session_argmax_excluding(ds4_session *s, int excluded_id) {
    if (!s || !s->logits) return -1;
    if (getenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == NULL) {
        return argmax_f32_excluding_unrolled8(
                s->logits, DS4_N_VOCAB, excluded_id);
    }
    int best = -1;
    float best_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        if ((int)i == excluded_id) continue;
        const float v = s->logits[i];
        if (best < 0 || v > best_logit) {
            best = (int)i;
            best_logit = v;
        }
    }
    return best;
}

int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!logits || n_vocab <= 0) return 0;
    float *scratch = xmalloc((size_t)n_vocab * sizeof(scratch[0]));
    const int token = sample_top_p_min_p(logits, (uint32_t)n_vocab,
                                         temperature, top_k, top_p, min_p,
                                         rng, scratch);
    free(scratch);
    return token;
}

int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    return sample_top_p_min_p(s->logits, DS4_N_VOCAB, temperature, top_k,
                              top_p, min_p, rng, s->sample_probs);
}

int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k) {
    if (!s || !out || k <= 0) return 0;
    if (k > (int)DS4_N_VOCAB) k = (int)DS4_N_VOCAB;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = DS4_NEG_INF;
        out[i].logprob = DS4_NEG_INF;
    }

    float max_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > max_logit) max_logit = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = (int)i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    for (int i = 0; i < k && out[i].id >= 0; i++) {
        out[i].logprob = isfinite(out[i].logit) ? (float)((double)out[i].logit - logsum) : DS4_NEG_INF;
    }
    return k;
}

int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out) {
    if (!s || !out || token < 0 || token >= (int)DS4_N_VOCAB) return 0;

    float max_logit = DS4_NEG_INF;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v) && v > max_logit) max_logit = v;
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    out->id = token;
    out->logit = s->logits[token];
    out->logprob = isfinite(out->logit) ? (float)((double)out->logit - logsum) : DS4_NEG_INF;
    return 1;
}

int ds4_session_copy_logits(ds4_session *s, float *out, int cap) {
    if (!s || !out || cap < (int)DS4_N_VOCAB) return 0;
    memcpy(out, s->logits, (size_t)DS4_N_VOCAB * sizeof(out[0]));
    return (int)DS4_N_VOCAB;
}

int ds4_session_set_logits(ds4_session *s, const float *logits, int n) {
    if (!s || !logits || n != (int)DS4_N_VOCAB) return 1;
    memcpy(s->logits, logits, (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
    return 0;
}

/* Pay the one-time first-submission GPU cost (pipeline ramp plus model-heap
 * residency for the batched prefill kernels) outside any measured window.
 * The TP worker calls this right after creating its session: it otherwise
 * encodes no main-queue GPU work until the first mirrored sync arrives, so
 * the cost lands inside the leader-timed prefill (measured ~1.1 s per run
 * on the M5 Max pair -- the whole first-run TP deficit vs single
 * node, which pays the same cost before its timing window starts). */
void ds4_session_gpu_warmup(ds4_session *s) {
#ifndef DS4_NO_GPU
    if (!s || ds4_session_is_cpu(s)) return;
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DEEPSEEK4) return;
    if (!metal_graph_batch_hc_mix(&s->graph) ||
        !metal_graph_batch_flat_hc(&s->graph)) return;
    if (!s->engine->weights.layer[0].hc_attn_fn) return;
    (void)metal_graph_warmup_prefill_kernels(&s->graph,
                                             &s->engine->model,
                                             &s->engine->weights,
                                             32);
#else
    (void)s;
#endif
}

#ifndef DS4_NO_GPU
static bool ds4_session_dspark_capture_current(const ds4_session *s) {
    if (!s || ds4_session_is_cpu(s) || !s->checkpoint_valid) return false;
    const ds4_gpu_graph *g = &s->graph;
    return g->dspark_capture_enabled &&
           g->dspark_capture_valid &&
           g->dspark_capture_checkpoint_len == (uint32_t)s->checkpoint.len;
}

static bool ds4_session_dspark_capture_batch_current(const ds4_session *s) {
    if (!s || ds4_session_is_cpu(s) || !s->checkpoint_valid) return false;
    const ds4_gpu_graph *g = &s->graph;
    if (!g->dspark_capture_enabled || !g->dspark_capture_batch_valid) {
        return false;
    }
    if (g->dspark_capture_batch_tokens == 0 ||
        g->dspark_capture_batch_start > (uint32_t)s->checkpoint.len) {
        return false;
    }
    const uint32_t batch_end =
        g->dspark_capture_batch_start + g->dspark_capture_batch_tokens;
    return batch_end >= g->dspark_capture_batch_start &&
           batch_end <= (uint32_t)s->checkpoint.len;
}

static void ds4_session_note_legacy_mtp_probe(ds4_session *s,
                                              int token,
                                              bool mtp_probe_log) {
    if (!s || !s->mtp_draft_valid) return;
    if (mtp_probe_log) {
        s->mtp_probe_total++;
        if (s->mtp_draft_token == token) s->mtp_probe_hit++;
        fprintf(stderr,
                "ds4: mtp probe token=%d draft=%d hit=%llu/%llu\n",
                token,
                s->mtp_draft_token,
                (unsigned long long)s->mtp_probe_hit,
                (unsigned long long)s->mtp_probe_total);
    }
    s->mtp_draft_valid = false;
}

static bool ds4_session_prepare_legacy_mtp_draft(ds4_session *s,
                                                 int token,
                                                 uint32_t pos,
                                                 bool mtp_probe_log) {
    ds4_engine *e = s->engine;
    if (!e->mtp_ready || !s->mtp_logits ||
        (e->mtp_draft_tokens <= 1 && !mtp_probe_log)) {
        return false;
    }

    int mtp_top = -1;
    if (metal_graph_eval_mtp_draft(&s->graph,
                                   &e->model,
                                   &e->weights,
                                   &e->mtp_model,
                                   &e->mtp_weights,
                                   token,
                                   pos,
                                   getenv("DS4_MTP_FULL_LOGITS") ? s->mtp_logits : NULL,
                                   &mtp_top)) {
        s->mtp_draft_token = mtp_top >= 0 ? mtp_top : sample_argmax(s->mtp_logits, DS4_N_VOCAB);
        s->mtp_draft_valid = true;
        return true;
    }
    if (mtp_probe_log) {
        fprintf(stderr, "ds4: mtp probe draft failed\n");
    }
    return false;
}

static bool ds4_session_prepare_dspark_draft_impl(ds4_session *s,
                                                  int token,
                                                  uint32_t pos) {
    const char *probe = getenv("DS4_DSPARK_PROBE");
    const bool probe_log = probe && probe[0];
    const bool enabled = s->engine->dspark;
    const char *fake_argmax = getenv("DS4_DSPARK_FAKE_ARGMAX_PROPOSAL");
    const bool fake_argmax_enabled =
        enabled &&
        fake_argmax && fake_argmax[0] && strcmp(fake_argmax, "0") != 0;
    const float confidence_threshold = s->engine->dspark_confidence_threshold;
    const bool stats_enabled = ds4_dspark_stats_enabled();
    const bool scheduler_enabled = ds4_dspark_scheduler_enabled();
    const bool time_enabled =
        stats_enabled ||
        (scheduler_enabled && ds4_dspark_scheduler_timing_enabled());
    const double stats_t0 = time_enabled ? now_sec() : 0.0;
#define DS4_DSPARK_PROP_T0() (stats_enabled ? now_sec() : 0.0)
#define DS4_DSPARK_PROP_ADD(field_, t0_) do {                              \
        if (stats_enabled) {                                                \
            s->dspark_stats.field_ += (now_sec() - (t0_)) * 1000.0;         \
        }                                                                   \
    } while (0)
    s->dspark_draft_valid = false;
    s->dspark_draft_len = 0;
    s->dspark_last_confidence0 = 0.0f;
    s->dspark_last_confidence0_valid = false;
    if (scheduler_enabled) s->dspark_last_propose_ms = 0.0;
    if (enabled && !fake_argmax_enabled &&
        ds4_session_dspark_scheduler_should_skip(s)) {
        (void)metal_graph_dspark_ring_maintain(&s->graph,
                                               &s->engine->mtp_model,
                                               &s->engine->dspark_weights,
                                               pos);
        const double propose_ms =
            time_enabled ? (now_sec() - stats_t0) * 1000.0 : 0.0;
        if (scheduler_enabled) s->dspark_last_propose_ms = propose_ms;
        if (stats_enabled) {
            s->dspark_stats.propose_ms += propose_ms;
        }
        return false;
    }
    if (probe_log || enabled) {
        const bool capture_ok = ds4_session_dspark_capture_current(s);
        const bool batch_capture_ok =
            ds4_session_dspark_capture_batch_current(s);
        const ds4_dspark_weights *dw = &s->engine->dspark_weights;
        const bool stage0_ready = dspark_stage0_weights_ready(&s->graph, dw);
        const bool runtime_fused_stage0_setup =
            enabled && !fake_argmax_enabled && !probe_log;
        const double stage0_t0 = DS4_DSPARK_PROP_T0();
        bool stage0_ok = false;
        if (capture_ok && stage0_ready && !runtime_fused_stage0_setup) {
            stage0_ok =
                metal_graph_eval_dspark_stage0(&s->graph,
                                               &s->engine->mtp_model,
                                               dw);
        } else if (capture_ok && stage0_ready && runtime_fused_stage0_setup) {
            stage0_ok = true;
        }
        DS4_DSPARK_PROP_ADD(propose_stage0_ms, stage0_t0);
        const double setup_t0 = DS4_DSPARK_PROP_T0();
        const bool draft_block_ready =
            dspark_draft_block_ready(&s->graph, &s->engine->weights, dw, token);
        const bool stage_input_ready =
            dspark_stage_input_ready(&s->graph, dw);
        bool stage_input_ok = false;
        if (stage0_ok && draft_block_ready && stage_input_ready) {
            if (runtime_fused_stage0_setup) {
                stage_input_ok =
                    metal_graph_prepare_dspark_stage0_setup_block(
                            &s->graph,
                            &s->engine->model,
                            &s->engine->weights,
                            &s->engine->mtp_model,
                            dw,
                            token,
                            pos);
                stage0_ok = stage_input_ok;
            } else {
                stage_input_ok =
                    metal_graph_prepare_dspark_setup_block(&s->graph,
                                                          &s->engine->model,
                                                          &s->engine->weights,
                                                          dw,
                                                          token,
                                                          pos);
            }
        }
        DS4_DSPARK_PROP_ADD(propose_setup_ms, setup_t0);
        const bool draft_cache_ready =
            dspark_stage_cache_ready(&s->graph, dw);
        uint32_t initial_cache_rows = 0;
        const uint64_t captured_batch_end =
            (uint64_t)s->graph.dspark_capture_batch_start +
            s->graph.dspark_capture_batch_tokens;
        const bool captured_batch_end_ok =
            s->graph.dspark_capture_batch_tokens != 0 &&
            captured_batch_end <= UINT32_MAX;
        const bool initial_cache_ready =
            batch_capture_ok &&
            draft_cache_ready &&
            captured_batch_end_ok &&
            (uint32_t)captured_batch_end == pos;
        const double cache_t0 = DS4_DSPARK_PROP_T0();
        bool initial_cache_ok = false;
        if (initial_cache_ready) {
            initial_cache_ok =
                metal_graph_seed_dspark_initial_cache_from_prefill(
                    &s->graph,
                    &s->engine->mtp_model,
                    dw,
                    s->graph.dspark_capture_batch_start,
                    s->graph.dspark_capture_batch_tokens,
                    &initial_cache_rows);
        }
        bool cache_window_ok = draft_cache_ready;
        if (draft_cache_ready) {
            if (initial_cache_ready) {
                if (!initial_cache_ok) metal_graph_dspark_cache_reset(&s->graph);
                cache_window_ok = initial_cache_ok;
            } else {
                cache_window_ok =
                    metal_graph_dspark_cache_crop_to_prefix(&s->graph, pos) &&
                    metal_graph_dspark_cache_ends_at(&s->graph, pos);
            }
        }
        DS4_DSPARK_PROP_ADD(propose_cache_ms, cache_t0);
        const bool noncausal_attn_ready =
            probe_log &&
            stage_input_ok &&
            draft_cache_ready &&
            dspark_noncausal_attention_probe_ready(&s->graph, dw);
        const bool noncausal_attn_ok =
            noncausal_attn_ready &&
            metal_graph_probe_dspark_noncausal_attention(&s->graph,
                                                         &s->engine->mtp_model,
                                                         dw);
        const bool stage_block_ready =
            stage_input_ok &&
            draft_cache_ready &&
            dspark_stage_block_ready(&s->graph, dw, 0);
        uint32_t stage_chain_done = 0;
        const bool stage_chain_ready =
            stage_input_ok &&
            draft_cache_ready &&
            cache_window_ok &&
            dw->n_stages != 0 &&
            dw->n_stages <= DS4_DSPARK_MAX_STAGES;
        uint32_t stage_cache_start = 0;
        uint32_t stage_cache_rows = 0;
        const double chain_t0 = DS4_DSPARK_PROP_T0();
        bool stage_chain_ok = false;
        const bool fuse_final_hidden =
            stage_chain_ready &&
            !probe_log &&
            confidence_threshold > 0.0f &&
            dspark_confidence_probe_ready(dw);
        if (stage_chain_ready) {
            stage_chain_ok =
                metal_graph_eval_dspark_stage_chain(&s->graph,
                                                &s->engine->mtp_model,
                                                dw,
                                                pos,
                                                fuse_final_hidden,
                                                &stage_chain_done,
                                                &stage_cache_start,
                                                &stage_cache_rows);
        }
        DS4_DSPARK_PROP_ADD(propose_chain_ms, chain_t0);
        const bool stage_block_ok = stage_chain_done >= 1u;
        const bool base_logits_ready =
            stage_chain_ok &&
            dspark_final_head_ready(&s->graph, &s->engine->weights, dw);
        int32_t markov_proposal[DS4_DSPARK_MAX_BLOCK_SIZE];
        for (uint32_t i = 0; i < DS4_DSPARK_MAX_BLOCK_SIZE; i++) {
            markov_proposal[i] = -1;
        }
        uint32_t markov_proposal_len = 0;
        bool markov_ok = false;
        uint32_t confidence_len = 0;
        uint32_t confidence_prefix_len = 0;
        float confidence0 = 0.0f;
        bool confidence_ok = false;
        bool reuse_confidence0_markov = false;
        const bool runtime_confidence_precheck =
            base_logits_ready &&
            !probe_log &&
            confidence_threshold > 0.0f &&
            dspark_confidence_probe_ready(dw);
        bool base_logits_ok = false;
        if (runtime_confidence_precheck) {
            const double hidden_t0 = DS4_DSPARK_PROP_T0();
            const bool hidden_ok =
                fuse_final_hidden ||
                metal_graph_eval_dspark_final_hidden(&s->graph,
                                                     &s->engine->mtp_model,
                                                     dw,
                                                     false);
            DS4_DSPARK_PROP_ADD(propose_hidden_ms, hidden_t0);
            bool conf0_ok = false;
            if (hidden_ok) {
                const double conf0_t0 = DS4_DSPARK_PROP_T0();
                conf0_ok =
                    dspark_eval_confidence0_runtime(&s->graph,
                                                    &s->engine->mtp_model,
                                                    dw,
                                                    token,
                                                    s->dspark_conf_features,
                                                    s->dspark_conf_features_cap,
                                                    &confidence0);
                DS4_DSPARK_PROP_ADD(propose_conf0_ms, conf0_t0);
            }
            if (conf0_ok) {
                confidence_ok = true;
                confidence_len = 1;
                if (sigmoid_stable(confidence0) >= confidence_threshold) {
                    confidence_prefix_len = 1;
                    const double logits_t0 = DS4_DSPARK_PROP_T0();
                    base_logits_ok =
                        metal_graph_eval_dspark_base_logits_from_hidden(
                                &s->graph,
                                &s->engine->model,
                                &s->engine->weights,
                                dw);
                    DS4_DSPARK_PROP_ADD(propose_logits_ms, logits_t0);
                    reuse_confidence0_markov =
                        base_logits_ok &&
                        !dspark_disable_reuse_confidence0_markov();
                }
            }
        } else if (base_logits_ready) {
            const double logits_t0 = DS4_DSPARK_PROP_T0();
            base_logits_ok =
                metal_graph_eval_dspark_base_logits(&s->graph,
                                                    &s->engine->model,
                                                    &s->engine->weights,
                                                    &s->engine->mtp_model,
                                                    dw);
            DS4_DSPARK_PROP_ADD(propose_logits_ms, logits_t0);
        }
        const bool markov_ready =
            base_logits_ok && dspark_markov_probe_ready(dw);
        const bool lazy_runtime_confidence =
            markov_ready && !probe_log && confidence_threshold > 0.0f;
        if (lazy_runtime_confidence) {
            const double markov_t0 = DS4_DSPARK_PROP_T0();
            markov_ok =
                dspark_apply_markov_confidence_lazy_runtime(
                        &s->graph,
                        &s->engine->mtp_model,
                        dw,
                        token,
                        confidence_threshold,
                        s->spec_row_logits,
                        s->dspark_markov_bias,
                        s->dspark_conf_features,
                        s->dspark_conf_features_cap,
                        markov_proposal,
                        &markov_proposal_len,
                        &confidence_len,
                        &confidence_prefix_len,
                        reuse_confidence0_markov,
                        &confidence0);
            DS4_DSPARK_PROP_ADD(propose_markov_ms, markov_t0);
            confidence_ok = markov_ok;
        } else if (markov_ready) {
            const uint64_t logits_count =
                (uint64_t)dw->block_size * (uint64_t)DS4_N_VOCAB;
            if (logits_count != 0 &&
                logits_count <= (uint64_t)SIZE_MAX / sizeof(float)) {
                const double markov_t0 = DS4_DSPARK_PROP_T0();
                const uint64_t logits_bytes = logits_count * sizeof(float);
                float *logits = xmalloc((size_t)logits_bytes);
                float *markov_state =
                    xmalloc((size_t)dw->markov_rank * sizeof(markov_state[0]));
                float *markov_bias =
                    xmalloc((size_t)DS4_N_VOCAB * sizeof(markov_bias[0]));
                markov_ok =
                    ds4_gpu_tensor_read(s->graph.spec_logits,
                                        0,
                                        logits,
                                        logits_bytes) != 0 &&
                    dspark_apply_markov_greedy_probe(logits,
                                                     &s->engine->mtp_model,
                                                     dw,
                                                     token,
                                                     markov_state,
                                                     markov_bias,
                                                     markov_proposal,
                                                     &markov_proposal_len);
                if (markov_ok && probe_log) {
                    /* Runtime only needs the greedy proposal. Keep the GPU
                     * writeback for probe mode, where spec_logits may be
                     * inspected after Markov correction. */
                    markov_ok =
                        ds4_gpu_tensor_write(s->graph.spec_logits,
                                             0,
                                             logits,
                                             logits_bytes) != 0;
                }
                free(markov_bias);
                free(markov_state);
                free(logits);
                DS4_DSPARK_PROP_ADD(propose_markov_ms, markov_t0);
            }
        }
        const bool confidence_ready =
            !lazy_runtime_confidence &&
            markov_ok &&
            dspark_confidence_probe_ready(dw);
        if (confidence_ready) {
            const uint64_t hidden_count =
                (uint64_t)dw->block_size * (uint64_t)DS4_N_EMBD;
            const uint64_t feature_count =
                (uint64_t)DS4_N_EMBD + (uint64_t)dw->markov_rank;
            if (hidden_count != 0 &&
                hidden_count <= (uint64_t)SIZE_MAX / sizeof(float) &&
                feature_count <= (uint64_t)SIZE_MAX / sizeof(float)) {
                const double confidence_t0 = DS4_DSPARK_PROP_T0();
                const uint64_t hidden_bytes = hidden_count * sizeof(float);
                float *hidden_rows = xmalloc((size_t)hidden_bytes);
                float *markov_state =
                    xmalloc((size_t)dw->markov_rank * sizeof(markov_state[0]));
                float *features =
                    xmalloc((size_t)feature_count * sizeof(features[0]));
                float *confidence_logits =
                    xmalloc((size_t)dw->block_size * sizeof(confidence_logits[0]));
                confidence_ok =
                    ds4_gpu_tensor_read(metal_graph_batch_ffn_norm(&s->graph),
                                        0,
                                        hidden_rows,
                                        hidden_bytes) != 0 &&
                    dspark_eval_confidence_probe(confidence_logits,
                                                 hidden_rows,
                                                 &s->engine->mtp_model,
                                                 dw,
                                                 token,
                                                 markov_proposal,
                                                 markov_state,
                                                 features,
                                                 &confidence_len);
                if (confidence_ok && confidence_len != 0) {
                    confidence0 = confidence_logits[0];
                    confidence_prefix_len =
                        dspark_confident_prefix_len(confidence_logits,
                                                    confidence_len,
                                                    confidence_threshold);
                }
                free(confidence_logits);
                free(features);
                free(markov_state);
                free(hidden_rows);
                DS4_DSPARK_PROP_ADD(propose_confidence_ms, confidence_t0);
            }
        }
        if (markov_ok && markov_proposal_len != 0) {
            uint32_t proposal_len = markov_proposal_len;
            if (confidence_threshold > 0.0f) {
                proposal_len = confidence_ok ? confidence_prefix_len : 0;
            }
            s->dspark_draft_len = proposal_len;
            if (s->dspark_draft_len > DS4_DSPARK_MAX_BLOCK_SIZE) {
                s->dspark_draft_len = DS4_DSPARK_MAX_BLOCK_SIZE;
            }
            for (uint32_t i = 0; i < s->dspark_draft_len; i++) {
                s->dspark_draft_tokens[i] = markov_proposal[i];
            }
            s->dspark_draft_valid = s->dspark_draft_len != 0;
        }
        if (confidence_ok && confidence_len != 0) {
            s->dspark_last_confidence0 = confidence0;
            s->dspark_last_confidence0_valid = true;
        }
        bool fake_argmax_ok = false;
        if (!s->dspark_draft_valid && fake_argmax_enabled) {
            s->dspark_draft_tokens[0] = sample_argmax(s->logits, DS4_N_VOCAB);
            s->dspark_draft_len = 1;
            s->dspark_draft_valid = true;
            fake_argmax_ok = true;
        }
        if (probe_log) {
            const char *stage0_status =
                stage0_ok ? "ok" :
                !capture_ok ? "capture-invalid" :
                !stage0_ready ? "unavailable" : "failed";
            const char *draft_block_status =
                stage_input_ok ? "ok" :
                !stage0_ok ? "stage0-not-ready" :
                !draft_block_ready ? "unavailable" :
                !stage_input_ready ? "stage-input-not-ready" : "failed";
            const char *stage_input_status =
                stage_input_ok ? "ok" :
                !stage0_ok ? "stage0-not-ready" :
                !draft_block_ready ? "draft-block-not-ready" :
                !stage_input_ready ? "unavailable" : "failed";
            const char *draft_cache_status =
                draft_cache_ready ? "ok" : "unavailable";
            const char *cache_window_status =
                !draft_cache_ready ? "draft-cache-not-ready" :
                cache_window_ok ? "ok" :
                initial_cache_ready ? "seed-failed" : "invalid";
            const char *initial_cache_status =
                initial_cache_ok ? "ok" :
                !batch_capture_ok ? "batch-capture-invalid" :
                !draft_cache_ready ? "draft-cache-not-ready" :
                !initial_cache_ready ? "unsupported-window" : "failed";
            const char *noncausal_attn_status =
                noncausal_attn_ok ? "ok" :
                !stage_input_ok ? "stage-input-not-ready" :
                !draft_cache_ready ? "draft-cache-not-ready" :
                !noncausal_attn_ready ? "unavailable" : "failed";
            const char *stage_block_status =
                stage_block_ok ? "ok" :
                !stage_input_ok ? "stage-input-not-ready" :
                !draft_cache_ready ? "draft-cache-not-ready" :
                !stage_block_ready ? "unavailable" : "failed";
            const char *stage_chain_status =
                stage_chain_ok ? "ok" :
                !stage_input_ok ? "stage-input-not-ready" :
                !draft_cache_ready ? "draft-cache-not-ready" :
                !cache_window_ok ? "cache-window-not-ready" :
                !stage_chain_ready ? "unavailable" :
                stage_chain_done != 0 ? "partial" : "failed";
            const char *base_logits_status =
                base_logits_ok ? "ok" :
                !stage_chain_ok ? "stage-chain-not-ready" :
                !base_logits_ready ? "unavailable" : "failed";
            const char *markov_status =
                markov_ok ? "ok" :
                !base_logits_ok ? "base-logits-not-ready" :
                !markov_ready ? "unavailable" : "failed";
            const char *confidence_status =
                confidence_ok ? "ok" :
                !markov_ok ? "markov-not-ready" :
                !confidence_ready ? "unavailable" : "failed";
            const char *fake_argmax_status =
                fake_argmax_ok ? "ok" :
                !fake_argmax_enabled ? "disabled" :
                s->dspark_draft_valid ? "real-proposal-present" : "skipped";
            fprintf(stderr,
                    "ds4: DSpark proposer pending token=%d pos=%u capture=%s "
                    "batch_capture=%s batch_start=%u batch_tokens=%u "
                    "stage0=%s draft_block=%s stage_input=%s draft_cache=%s "
                    "initial_cache=%s initial_cache_rows=%u "
                    "cache_window=%s cache_token_start=%u "
                    "cache_raw_start=%u cache_len=%u "
                    "noncausal_attn=%s stage_block=%s stage_chain=%s "
                    "stage_chain_done=%u/%u stage_raw_start=%u cache_rows=%u "
                    "base_logits=%s markov=%s "
                    "proposal_len=%u proposal0=%d fake_argmax=%s confidence=%s "
                    "confidence_len=%u confidence_prefix=%u "
                    "confidence_threshold=%.3f confidence0=%.3f block=%u "
                    "tensors=%u missing=%u invalid=%u metadata_errors=%u\n",
                    token,
                    pos,
                    capture_ok ? "current" : "invalid",
                    batch_capture_ok ? "current" : "invalid",
                    s->graph.dspark_capture_batch_start,
                    s->graph.dspark_capture_batch_tokens,
                    stage0_status,
                    draft_block_status,
                    stage_input_status,
                    draft_cache_status,
                    initial_cache_status,
                    initial_cache_rows,
                    cache_window_status,
                    s->graph.dspark_cache_token_start,
                    s->graph.dspark_cache_start,
                    s->graph.dspark_cache_len,
                    noncausal_attn_status,
                    stage_block_status,
                    stage_chain_status,
                    stage_chain_done,
                    dw->n_stages,
                    stage_cache_start,
                    stage_cache_rows,
                    base_logits_status,
                    markov_status,
                    s->dspark_draft_len,
                    s->dspark_draft_len ? s->dspark_draft_tokens[0] : -1,
                    fake_argmax_status,
                    confidence_status,
                    confidence_len,
                    confidence_prefix_len,
                    confidence_threshold,
                    confidence0,
                    dw->block_size,
                    dw->present_tensors,
                    dw->missing_tensors,
                    dw->invalid_tensors,
                    dw->metadata_errors);
        }
    }
    if (time_enabled) {
        const double propose_ms = (now_sec() - stats_t0) * 1000.0;
        if (scheduler_enabled) s->dspark_last_propose_ms = propose_ms;
        if (stats_enabled) s->dspark_stats.propose_ms += propose_ms;
    }
#undef DS4_DSPARK_PROP_ADD
#undef DS4_DSPARK_PROP_T0
    return s->dspark_draft_valid;
}

/* Support weights and DSpark scratch live on the configured executor tier. */
static bool ds4_session_prepare_dspark_draft(ds4_session *s,
                                             int token,
                                             uint32_t pos) {
    ds4_gpu_graph *g = &s->graph;
    const int exec_tier =
        g->dspark_exec_tier >= 0 && g->dspark_exec_tier < DS4_MAX_GPUS
            ? g->dspark_exec_tier : 0;
    const int saved_tier = g->active_tier;
    if (exec_tier != saved_tier) {
        if (ds4_gpu_set_current_device(exec_tier) != 0) return false;
        g->active_tier = exec_tier;
    }
    const bool ok = ds4_session_prepare_dspark_draft_impl(s, token, pos);
    if (exec_tier != saved_tier) {
        g->active_tier = saved_tier;
        if (ds4_gpu_set_current_device(saved_tier) != 0) return false;
    }
    return ok;
}

static void ds4_session_prepare_support_draft(ds4_session *s,
                                              int token,
                                              uint32_t pos,
                                              bool probe_mtp,
                                              bool mtp_probe_log) {
    if (!s || !probe_mtp || !s->engine) return;

    switch (s->engine->support_kind) {
    case DS4_SUPPORT_MTP_LEGACY:
        (void)ds4_session_prepare_legacy_mtp_draft(s, token, pos, mtp_probe_log);
        break;
    case DS4_SUPPORT_DSPARK:
        (void)ds4_session_prepare_dspark_draft(s, token, pos);
        break;
    case DS4_SUPPORT_NONE:
    default:
        break;
    }
}
#endif

static int ds4_session_eval_internal(ds4_session *s, int token, bool probe_mtp,
                                     char *err, size_t errlen) {
    if (!s) return 1;
    if (s->distributed) {
        if (!s->checkpoint_valid) {
            if (errlen) snprintf(err, errlen, "distributed decode requires a valid checkpoint");
            return 1;
        }
        (void)probe_mtp;
        return ds4_dist_session_eval(s->distributed,
                                     s,
                                     &s->checkpoint,
                                     token,
                                     s->logits,
                                     err,
                                     errlen);
    }
    if (ds4_session_is_cpu(s)) {
        ds4_engine *e = s->engine;
        forward_token_raw_swa_cpu_decode_scratch(s->logits,
                                                 &e->model,
                                                 &e->weights,
                                                 &s->cpu_cache,
                                                 token,
                                                 (uint32_t)s->checkpoint.len,
                                                 e->directional_steering_dirs,
                                                 e->directional_steering_attn_scale,
                                                 e->directional_steering_ffn_scale,
                                                 &s->cpu_scratch);
        token_vec_push(&s->checkpoint, token);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        ds4_session_dspark_capture_note_checkpoint(s);
        (void)probe_mtp;
        return 0;
    }
#ifdef DS4_NO_GPU
    (void)s;
    (void)token;
    (void)probe_mtp;
    snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    ds4_engine *e = s->engine;
    if (ds4_session_is_glm(s)) {
        /* TP worker under GLM MTP: run the full speculative cycle off the
         * mirrored EVAL frame so drafts, verify batches, and gate traffic
         * stay in lockstep with the leader's cycle. */
        if (!s->glm_spec_inside && s->glm_graph_ready &&
            e->glm_mtp && DS4_N_NEXTN_PREDICT != 0 &&
            e->tp.active && e->tp.rank != 0) {
            int acc[2];
            const int rc = ds4_session_glm_spec_cycle(s, token, acc, 2, err, errlen);
            (void)probe_mtp;
            return rc < 0 ? 1 : 0;
        }
        if (!s->glm_graph_ready) {
            if (errlen) snprintf(err, errlen, "%s GLM graph is not initialized",
                                 ds4_backend_name(e->backend));
            return 1;
        }
        if ((uint32_t)s->checkpoint.len >= s->glm_graph.ctx_size) {
            if (errlen) snprintf(err, errlen, "GLM Metal context reached (%u)",
                                 s->glm_graph.ctx_size);
            return 1;
        }
        const uint32_t pos = (uint32_t)s->checkpoint.len;
        const bool updates_dense =
            glm_graph_decode_updates_dense_cache(&s->glm_graph, pos, s->logits);
        if (!glm_graph_forward_token(&s->glm_graph,
                                     &e->model,
                                     &e->weights,
                                     token,
                                     NULL,
                                     pos,
                                     NULL,
                                     s->logits,
                                     false))
        {
            if (errlen) snprintf(err, errlen, "%s GLM decode failed",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            s->mtp_draft_valid = false;
            ds4_session_glm_cap_dense_cache(s);
            return 1;
        }
        token_vec_push(&s->checkpoint, token);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        if (updates_dense) ds4_session_glm_note_dense_cache(s, pos, 1);
        /* MTP acceptance probe (timing/quality only, output untouched):
         * after each committed token, draft the token two positions ahead
         * from the nextn block and score it against the next greedy pick.
         * Both TP ranks must set the env (the draft's routed experts ride
         * a big-gate exchange). */
        if (getenv("DS4_GLM_MTP_PROBE") && DS4_N_NEXTN_PREDICT != 0) {
            static int probe_hits, probe_total, probe_have, probe_draft;
            static uint32_t probe_min_pos = UINT32_MAX;
            int nmax = 0;
            float nbest = s->logits[0];
            for (uint32_t i = 1; i < DS4_N_VOCAB; i++) {
                if (s->logits[i] > nbest) { nbest = s->logits[i]; nmax = (int)i; }
            }
            if (probe_have) {
                probe_total++;
                probe_hits += (probe_draft == nmax);
                if ((probe_total % 16) == 0) {
                    fprintf(stderr, "ds4: glm mtp probe: %d/%d hits (%.1f%%)\n",
                            probe_hits, probe_total,
                            100.0 * probe_hits / probe_total);
                }
            }
            if (probe_min_pos == UINT32_MAX) probe_min_pos = pos;
            int draft = -1;
            if (glm_graph_mtp_step(&s->glm_graph, &e->model, &e->weights,
                                   nmax, pos, probe_min_pos, &draft)) {
                probe_draft = draft;
                probe_have = 1;
            } else {
                probe_have = 0;
                fprintf(stderr, "ds4: glm mtp probe: draft step failed at pos %u\n", pos);
            }
        }
        (void)probe_mtp;
        return 0;
    }
    const bool mtp_probe_log = getenv("DS4_MTP_PROBE") != NULL;
    if (probe_mtp && e->support_kind == DS4_SUPPORT_MTP_LEGACY) {
        ds4_session_note_legacy_mtp_probe(s, token, mtp_probe_log);
    }
    const bool dspark_target_timing =
        e->support_kind == DS4_SUPPORT_DSPARK &&
        (ds4_dspark_stats_enabled() ||
         (ds4_dspark_scheduler_enabled() &&
          ds4_dspark_scheduler_timing_enabled()));
    const double target_t0 = dspark_target_timing ? now_sec() : 0.0;
    if (!metal_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                        (uint32_t)token,
                                        (uint32_t)s->checkpoint.len,
                                        s->logits))
    {
        snprintf(err, errlen, "%s decode failed", ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }
    if (dspark_target_timing) {
        const double target_ms = (now_sec() - target_t0) * 1000.0;
        s->dspark_last_target_eval_ms = target_ms;
        if (ds4_dspark_stats_enabled()) {
            s->dspark_stats.target_ms += target_ms;
        }
    }
    token_vec_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    ds4_session_dspark_capture_note_checkpoint(s);
    ds4_session_prepare_support_draft(s,
                                      token,
                                      (uint32_t)(s->checkpoint.len - 1),
                                      probe_mtp,
                                      mtp_probe_log);
    return 0;
#endif
}

/* TP-aware eval: mirrors the eval to the worker, runs it locally with the
 * requested draft-probe flag, then merges the vocab-split logits halves.
 * Both ds4_session_eval and the speculative driver funnel through here so
 * the leader/worker lockstep survives every eval entry point. */
static int ds4_session_eval_probe_tp(ds4_session *s, int token, bool probe_mtp,
                                     char *err, size_t errlen) {
    if (ds4_session_tp_leader(s)) {
        ds4_engine *e = s->engine;
        if (!ds4_tp_send_eval(e->tp.ctx, s->tp_session_id,
                              ++e->tp.eval_seq, token)) {
            snprintf(err, errlen, "tp: worker eval send failed");
            return 1;
        }
    }
    int rc = ds4_session_eval_internal(s, token, probe_mtp, err, errlen);
    if (rc != 0 && ds4_session_tp_leader(s)) {
        ds4_session_invalidate(s);
        return rc;
    }
#if !defined(DS4_NO_GPU) && defined(__APPLE__)
    if (rc == 0 && s->engine && s->engine->tp.active && ds4_gpu_tp_failed()) {
        snprintf(err, errlen, "tp: gate transport failed");
        if (ds4_session_tp_leader(s)) ds4_session_invalidate(s);
        return 1;
    }
#endif
    /* Vocab-split head: merge the halves after every eval (DS4 only). */
    if (rc == 0 && s->engine && s->engine->tp.active && s->engine->tp.vocab_split) {
        const uint32_t vhalf = (uint32_t)DS4_N_VOCAB / 2u;
        if (s->engine->tp.rank == 0) {
            if (!ds4_tp_recv_logits_half(s->engine->tp.ctx, s->logits + vhalf, vhalf)) {
                snprintf(err, errlen, "tp: worker logits half missing");
                ds4_session_invalidate(s);
                return 1;
            }
        } else {
            if (!ds4_tp_send_logits_half(s->engine->tp.ctx, s->logits + vhalf, vhalf)) {
                snprintf(err, errlen, "tp: logits half send failed");
                return 1;
            }
        }
    }
    return rc;
}

int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    bool probe_mtp = true;
#ifndef DS4_NO_GPU
    if (s && s->engine && s->engine->support_kind == DS4_SUPPORT_DSPARK) {
        probe_mtp = false;
    }
#endif
    return ds4_session_eval_probe_tp(s, token, probe_mtp, err, errlen);
}

