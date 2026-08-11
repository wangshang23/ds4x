/* Decode gate firing schedule for the TP transport (see ds4_tp_identity):
 * DS4 fires ATTN+FFN gates on every layer (identity mapping); GLM fires a
 * single FFN gate per sparse layer, skipping the leading dense blocks. */
void ds4_engine_tp_gate_schedule(ds4_engine *e,
                                 uint32_t *start,
                                 uint32_t *step,
                                 uint32_t *per_token) {
    (void)e;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        /* One FFN gate per sparse layer of the NORMAL pass: the leading
         * dense blocks fire nothing and the trailing nextn/MTP block is
         * not part of the decode pass at all. */
        *start = DS4_N_LEADING_DENSE * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_FFN;
        *step = DS4_TP_GATES_PER_LAYER;
        *per_token = DS4_N_LAYER - DS4_N_NEXTN_PREDICT - DS4_N_LEADING_DENSE;
    } else {
        *start = 0;
        *step = 1;
        *per_token = DS4_N_LAYER * DS4_TP_GATES_PER_LAYER;
    }
}

int ds4_engine_embd_dim(ds4_engine *e) {
    (void)e;
    return (int)DS4_N_EMBD;
}

uint64_t ds4_engine_model_bytes(ds4_engine *e) {
    return e->model.size;
}

int ds4_engine_tp_vocab_split(ds4_engine *e) {
    return e && e->tp.active && e->tp.vocab_split;
}

#if !defined(DS4_NO_GPU) && defined(__APPLE__)
static int ds4_engine_tp_exchange(void *ud, uint32_t layer, uint32_t gate, uint64_t seq) {
    ds4_tp *tp = ud;
    const int ok = ds4_tp_gate_exchange(tp, layer, gate, seq);
    if (!ok) ds4_tp_mark_failed(tp);
    return ok;
}

static int ds4_engine_tp_batch_exchange(void *ud, uint32_t layer,
                                        uint32_t rows, uint64_t seq) {
    ds4_tp *tp = ud;
    const int ok = ds4_tp_batch_gate_exchange(tp, layer, rows, seq);
    if (!ok) ds4_tp_mark_failed(tp);
    return ok;
}

static int ds4_engine_tp_big_exchange(void *ud, uint32_t layer, uint64_t seq,
                                      const void *out, void *in,
                                      uint64_t bytes) {
    if (g_glm_tp_debug_ids && getenv("DS4_GLM_TP_DEBUG")) {
        fprintf(stderr,
                "ds4-tp: big gate l=%u ids[0..7]=%d %d %d %d %d %d %d %d\n",
                layer,
                g_glm_tp_debug_ids[0], g_glm_tp_debug_ids[1],
                g_glm_tp_debug_ids[2], g_glm_tp_debug_ids[3],
                g_glm_tp_debug_ids[4], g_glm_tp_debug_ids[5],
                g_glm_tp_debug_ids[6], g_glm_tp_debug_ids[7]);
    }
    ds4_tp *tp = ud;
    const int ok = ds4_tp_big_gate_exchange(tp, layer, seq, out, in, bytes);
    if (!ok) ds4_tp_mark_failed(tp);
    return ok;
}
#endif

int ds4_engine_tp_bind(ds4_engine *e, struct ds4_tp *tp, char *err, size_t errlen) {
#if defined(DS4_NO_GPU) || !defined(__APPLE__)
    (void)e; (void)tp;
    snprintf(err, errlen, "tensor parallelism requires the Metal backend");
    return 0;
#else
    if (e->backend != DS4_BACKEND_METAL) {
        snprintf(err, errlen, "tensor parallelism requires the Metal backend");
        return 0;
    }
    if (e->tp.active) {
        snprintf(err, errlen, "tensor parallelism already bound");
        return 0;
    }
    const uint32_t slots = (uint32_t)DS4_N_LAYER * DS4_TP_GATES_PER_LAYER;
    const uint64_t vec_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t slab_bytes = ds4_tp_slab_bytes((uint32_t)DS4_N_LAYER, (uint32_t)DS4_N_EMBD);
    e->tp.slab = ds4_gpu_tensor_alloc(slab_bytes);
    e->tp.zero_vec = ds4_gpu_tensor_alloc(vec_bytes);
    e->tp.out_views = calloc(slots, sizeof(*e->tp.out_views));
    e->tp.in_views = calloc(slots, sizeof(*e->tp.in_views));
    e->tp.batch_out_views = calloc((size_t)DS4_N_LAYER, sizeof(*e->tp.batch_out_views));
    e->tp.batch_in_views = calloc((size_t)DS4_N_LAYER, sizeof(*e->tp.batch_in_views));
    if (!e->tp.batch_out_views || !e->tp.batch_in_views) {
        snprintf(err, errlen, "tp: batch view table allocation failed");
        return 0;
    }
    if (!e->tp.slab || !e->tp.zero_vec || !e->tp.out_views || !e->tp.in_views) {
        snprintf(err, errlen, "tp: slab allocation failed (%llu bytes)",
                 (unsigned long long)slab_bytes);
        return 0;
    }
    memset(ds4_gpu_tensor_contents(e->tp.slab), 0, slab_bytes);
    memset(ds4_gpu_tensor_contents(e->tp.zero_vec), 0, vec_bytes);
    if (!ds4_tp_attach_slab(tp, ds4_gpu_tensor_contents(e->tp.slab), err, errlen))
        return 0;
    for (uint32_t l = 0; l < (uint32_t)DS4_N_LAYER; l++) {
        for (uint32_t gate = 0; gate < DS4_TP_GATES_PER_LAYER; gate++) {
            const uint32_t slot = l * DS4_TP_GATES_PER_LAYER + gate;
            e->tp.out_views[slot] = ds4_gpu_tensor_view(
                    e->tp.slab, ds4_tp_slab_out_offset(tp, l, gate), vec_bytes);
            e->tp.in_views[slot] = ds4_gpu_tensor_view(
                    e->tp.slab, ds4_tp_slab_in_offset(tp, l, gate), vec_bytes);
            if (!e->tp.out_views[slot] || !e->tp.in_views[slot]) {
                snprintf(err, errlen, "tp: slab view creation failed");
                return 0;
            }
        }
        e->tp.batch_out_views[l] = ds4_gpu_tensor_view(
                e->tp.slab, ds4_tp_slab_batch_out_offset(tp, l),
                (uint64_t)DS4_TP_BATCH_MAX_ROWS * vec_bytes);
        e->tp.batch_in_views[l] = ds4_gpu_tensor_view(
                e->tp.slab, ds4_tp_slab_batch_in_offset(tp, l),
                (uint64_t)DS4_TP_BATCH_MAX_ROWS * vec_bytes);
        if (!e->tp.batch_out_views[l] || !e->tp.batch_in_views[l]) {
            snprintf(err, errlen, "tp: batch slab view creation failed");
            return 0;
        }
    }
    if (!ds4_gpu_tp_init((uint32_t)ds4_tp_rank(tp),
                         e->tp.slab, ds4_tp_slab_gpu_flags_offset(tp),
                         ds4_engine_tp_exchange, tp)) {
        snprintf(err, errlen, "tp: gate service init failed");
        return 0;
    }
    ds4_gpu_tp_set_batch_exchange(ds4_engine_tp_batch_exchange);
    ds4_gpu_tp_set_big_exchange(ds4_engine_tp_big_exchange);
    /* GLM keeps its replicated output head unsplit in v0: the
     * leader computes full logits and nothing crosses the wire. */
    e->tp.vocab_split = DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM_DSA;
    e->tp.ctx = tp;
    e->tp.rank = ds4_tp_rank(tp);
    e->tp.eval_seq = 0;
    e->tp.active = true;
    ds4_log(stderr, DS4_LOG_OK,
            "tensor parallelism bound: rank %d, 50/50 expert split, %s transport",
            e->tp.rank, ds4_tp_is_rdma(tp) ? "rdma" : "tcp");
    return 1;
#endif
}

bool ds4_engine_is_glm_dsa(ds4_engine *e) {
    (void)e;
    return DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA;
}

void ds4_engine_close(ds4_engine *e) {
    if (!e) return;
#if !defined(DS4_NO_GPU) && defined(__APPLE__)
    if (e->tp.active) {
        ds4_gpu_tp_shutdown();
        const uint32_t slots = (uint32_t)DS4_N_LAYER * DS4_TP_GATES_PER_LAYER;
        for (uint32_t i = 0; i < slots; i++) {
            if (e->tp.out_views) ds4_gpu_tensor_free(e->tp.out_views[i]);
            if (e->tp.in_views) ds4_gpu_tensor_free(e->tp.in_views[i]);
        }
        for (uint32_t i = 0; i < (uint32_t)DS4_N_LAYER; i++) {
            if (e->tp.batch_out_views) ds4_gpu_tensor_free(e->tp.batch_out_views[i]);
            if (e->tp.batch_in_views) ds4_gpu_tensor_free(e->tp.batch_in_views[i]);
        }
        free(e->tp.batch_out_views);
        free(e->tp.batch_in_views);
        free(e->tp.out_views);
        free(e->tp.in_views);
        ds4_gpu_tensor_free(e->tp.zero_vec);
        ds4_gpu_tensor_free(e->tp.slab);
        memset(&e->tp, 0, sizeof(e->tp));
    }
#endif
    ds4_expert_profile_close();
    weights_free(&e->weights);
    vocab_free(&e->vocab);
    ds4_threads_shutdown();
    if (e->mtp_model.map) model_close(&e->mtp_model);
    model_close(&e->model);
#ifndef DS4_NO_GPU
    if (e->shared_prefill_workspace_ready) {
        metal_graph_free_prefill_workspace(&e->shared_prefill_workspace);
        e->shared_prefill_workspace_ready = false;
    }
    ds4_gpu_cleanup();
#endif
    ds4_ssd_memory_lock_release(&e->simulated_memory);
    ds4_release_instance_lock();
    free(e->directional_steering_dirs);
    free(e->directional_steering_file);
    free(e);
}

#ifndef DS4_NO_GPU
static bool ds4_dspark_stats_enabled(void) {
    const char *env = getenv("DS4_DSPARK_STATS");
    return env && env[0] && strcmp(env, "0") != 0;
}

static void ds4_format_len_hist(
        char           *buf,
        size_t          buflen,
        const uint64_t *hist) {
    if (!buf || buflen == 0 || !hist) return;
    size_t off = 0;
    buf[0] = '\0';
    for (uint32_t i = 0; i <= DS4_DSPARK_MAX_BLOCK_SIZE; i++) {
        if (hist[i] == 0) continue;
        const int n = snprintf(buf + off,
                               off < buflen ? buflen - off : 0,
                               "%s%u:%llu",
                               off ? "," : "",
                               i,
                               (unsigned long long)hist[i]);
        if (n < 0) break;
        if ((size_t)n >= (off < buflen ? buflen - off : 0)) {
            off = buflen - 1u;
            break;
        }
        off += (size_t)n;
    }
    if (off == 0) {
        snprintf(buf, buflen, "none");
    }
}

static void ds4_session_print_dspark_stats(const ds4_session *s) {
    if (!s || !ds4_dspark_stats_enabled()) return;
    const ds4_dspark_spec_stats *st = &s->dspark_stats;
    if (st->cycles == 0 && st->propose_ms == 0.0) return;
    char draft_hist[192];
    char accept_hist[192];
    ds4_format_len_hist(draft_hist, sizeof(draft_hist), st->draft_len_hist);
    ds4_format_len_hist(accept_hist, sizeof(accept_hist), st->accepted_len_hist);
    const double accept_rate =
        st->proposed_tokens ?
            (100.0 * (double)st->accepted_draft_tokens /
             (double)st->proposed_tokens) : 0.0;
    const double avg_accept =
        st->cycles ?
            (double)st->accepted_draft_tokens / (double)st->cycles : 0.0;
    const double extra_ms = st->propose_ms + st->total_ms;
    const double net_saved_ms = st->saved_ms - extra_ms;
    fprintf(stderr,
            "ds4: DSpark stats cycles=%llu first_tokens=%llu proposed=%llu "
            "accepted_draft=%llu accept_rate=%.2f%% avg_accept=%.3f "
            "full=%llu partial=%llu direct_full=%llu direct_partial=%llu "
            "replay_fallbacks=%llu miss_first=%llu no_draft=%llu "
            "no_room=%llu invalid=%llu scheduler_skips=%llu "
            "tail_skips=%llu verifier_unavailable=%llu errors=%llu time_ms propose=%.3f "
            "prop_stage0=%.3f prop_setup=%.3f prop_cache=%.3f "
            "prop_chain=%.3f prop_hidden=%.3f prop_conf0=%.3f "
            "prop_logits=%.3f prop_markov=%.3f prop_confidence=%.3f "
            "snapshot=%.3f verify=%.3f verify_upload=%.3f "
            "verify_layer=%.3f verify_head=%.3f verify_read=%.3f "
            "verify_fused_head=%llu replay=%.3f spec_total=%.3f "
            "target=%.3f saved=%.3f net_saved=%.3f "
            "draft_len_hist=%s accepted_len_hist=%s\n",
            (unsigned long long)st->cycles,
            (unsigned long long)st->first_tokens,
            (unsigned long long)st->proposed_tokens,
            (unsigned long long)st->accepted_draft_tokens,
            accept_rate,
            avg_accept,
            (unsigned long long)st->full_accepts,
            (unsigned long long)st->partial_accepts,
            (unsigned long long)st->direct_full_commits,
            (unsigned long long)st->direct_partial_commits,
            (unsigned long long)st->replay_fallbacks,
            (unsigned long long)st->first_misses,
            (unsigned long long)st->no_draft,
            (unsigned long long)st->no_room,
            (unsigned long long)st->invalid_draft,
            (unsigned long long)st->scheduler_skips,
            (unsigned long long)st->tail_skips,
            (unsigned long long)st->verifier_unavailable,
            (unsigned long long)st->verifier_errors,
            st->propose_ms,
            st->propose_stage0_ms,
            st->propose_setup_ms,
            st->propose_cache_ms,
            st->propose_chain_ms,
            st->propose_hidden_ms,
            st->propose_conf0_ms,
            st->propose_logits_ms,
            st->propose_markov_ms,
            st->propose_confidence_ms,
            st->snapshot_ms,
            st->verify_ms,
            st->verify_upload_ms,
            st->verify_layer_ms,
            st->verify_head_ms,
            st->verify_read_ms,
            (unsigned long long)st->verifier_fused_head,
            st->replay_ms,
            st->total_ms,
            st->target_ms,
            st->saved_ms,
            net_saved_ms,
            draft_hist,
            accept_hist);
}
#endif

static bool ds4_session_tp_leader(const ds4_session *s) {
    return s && s->engine && s->engine->tp.active && s->engine->tp.rank == 0;
}

static int ds4_session_tp_register(ds4_session *s) {
    if (!ds4_session_tp_leader(s)) return 1;
    ds4_engine *e = s->engine;
    uint64_t id = ++e->tp.next_session_id;
    if (id == 0) id = ++e->tp.next_session_id;
    char err[256] = "";
    if (!ds4_tp_send_session_create(e->tp.ctx, id, s->ctx_size) ||
        !ds4_tp_wait_command_ack(e->tp.ctx, id, "session create",
                                 err, sizeof(err))) {
        fprintf(stderr, "ds4: %s\n",
                err[0] ? err : "tp: worker session create send failed");
        return 0;
    }
    s->tp_session_id = id;
    return 1;
}

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0) return 1;
    if (e->backend == DS4_BACKEND_CPU) {
        if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
            fprintf(stderr, "ds4: GLM sessions currently require a graph backend\n");
            return 1;
        }
        if (e->distributed.role == DS4_DISTRIBUTED_COORDINATOR) {
            fprintf(stderr, "ds4: distributed coordinator sessions require the graph backend\n");
            return 1;
        }
        ds4_session *s = xcalloc(1, sizeof(*s));
        s->engine = e;
        s->ctx_size = ctx_size;
        s->prefill_cap = ds4_prefill_cap_for_prompt(ctx_size,
                                                     e->prefill_chunk);
        kv_cache_init(&s->cpu_cache, (uint32_t)ctx_size, 0);
        cpu_decode_scratch_init(&s->cpu_scratch, (uint32_t)ctx_size);
        s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        s->sample_probs = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->sample_probs[0]));
        if (!ds4_session_tp_register(s)) {
            ds4_session_free(s);
            return 1;
        }
        *out = s;
        return 0;
    }
#ifdef DS4_NO_GPU
    return 1;
#else
    if (!ds4_backend_uses_graph(e->backend) || !e->metal_ready) return 1;

    ds4_session *s = xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        const uint32_t normal_layers = glm_graph_normal_layer_count();
        uint32_t layer_start = 0;
        uint32_t layer_end = normal_layers ? normal_layers - 1u : 0;
        bool require_token_embd = true;
        bool require_output = true;
        if (e->distributed.role != DS4_DISTRIBUTED_NONE && e->distributed.layers.set) {
            layer_start = e->distributed.layers.start;
            layer_end = e->distributed.layers.has_output ?
                (normal_layers ? normal_layers - 1u : 0u) :
                e->distributed.layers.end;
            require_token_embd = layer_start == 0;
            require_output = e->distributed.layers.has_output;
        }
        if (!normal_layers || layer_start > layer_end || layer_end >= normal_layers) {
            fprintf(stderr, "ds4: invalid GLM layer slice %u:%u\n", layer_start, layer_end);
            free(s);
            return 1;
        }
        s->glm_graph.placement = e->multi_tier ? e->placement : NULL;
        if (!glm_graph_alloc_slice(&s->glm_graph,
                                   &e->model,
                                   &e->weights,
                                   ctx_size,
                                   e->ssd_streaming,
                                   e->ssd_streaming_cold,
                                   ds4_engine_streaming_transient_guard_bytes(e),
                                   layer_start,
                                   layer_end,
                                   require_token_embd,
                                   require_output)) {
            free(s);
            return 1;
        }
        s->prefill_cap = s->glm_graph.ctx_cap;
        ds4_gpu_enable_q8_dequant_gemm();
        s->glm_graph_ready = true;
        s->glm_graph.quality = e->quality;
        s->glm_graph.ssd_streaming = e->ssd_streaming;
        s->glm_graph.ssd_streaming_cold = e->ssd_streaming_cold;
#if !defined(DS4_NO_GPU) && defined(__APPLE__)
        if (e->tp.active) {
            s->glm_graph.tp_world = 2;
            s->glm_graph.tp_rank = (uint32_t)e->tp.rank;
            s->glm_graph.tp_out = e->tp.out_views;
            s->glm_graph.tp_in = e->tp.in_views;
        }
#endif
        if (e->ssd_streaming &&
            !glm_graph_env_present("DS4_ROCM_GLM_DISABLE_STREAMING_SEED_BEFORE_PREFILL",
                                   "DS4_METAL_GLM_DISABLE_STREAMING_SEED_BEFORE_PREFILL")) {
            ds4_gpu_graph seed_graph;
            memset(&seed_graph, 0, sizeof(seed_graph));
            seed_graph.quality = e->quality;
            seed_graph.ssd_streaming = e->ssd_streaming;
            seed_graph.ssd_streaming_cold = e->ssd_streaming_cold;
            seed_graph.streaming_preload_experts =
                e->ssd_streaming_preload_experts;
            if (!metal_graph_seed_streaming_expert_cache_from_hotlist(
                        &seed_graph,
                        &e->model,
                        &e->weights)) {
                glm_graph_free(&s->glm_graph);
                free(s);
                return 1;
            }
        }
        s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        s->sample_probs =
            xmalloc((size_t)DS4_N_VOCAB * sizeof(s->sample_probs[0]));
        if (e->distributed.role == DS4_DISTRIBUTED_COORDINATOR) {
            char err[256];
            if (ds4_dist_session_create(&s->distributed,
                                        e,
                                        &e->distributed,
                                        s,
                                        ctx_size,
                                        err,
                                        sizeof(err)) != 0) {
                fprintf(stderr,
                        "ds4: failed to create distributed coordinator session: %s\n",
                        err[0] ? err : "unknown error");
                glm_graph_free(&s->glm_graph);
                free(s->glm_mtp_hc);
                free(s->glm_mtp_logits0);
                free(s->logits);
                free(s->sample_probs);
                free(s);
                return 1;
            }
        }
        if (!ds4_session_tp_register(s)) {
            ds4_session_free(s);
            return 1;
        }
        *out = s;
        return 0;
    }
    s->prefill_cap = metal_graph_prefill_cap_for_prompt(ctx_size,
                                                        e->prefill_chunk);
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, s->prefill_cap);
    const ds4_layer_weights *shape_layer = weights_first_bound_layer(&e->weights);
    if (!shape_layer) {
        fprintf(stderr, "ds4: no transformer layers are loaded\n");
        free(s);
        return 1;
    }
    const bool need_spec_verifier =
        e->mtp_ready ||
        (e->support_kind == DS4_SUPPORT_DSPARK && e->dspark) ||
        e->tp.active; /* TP worker mirrors the leader's verify blocks */
    const int *placement = e->multi_tier ? e->placement : NULL;
    const ds4_gpu_graph *shared_prefill_workspace =
        e->share_session_prefill_workspace &&
        e->shared_prefill_workspace_ready
            ? &e->shared_prefill_workspace
            : NULL;
    s->graph.dspark_exec_tier = e->multi_tier ? e->dspark_exec_tier : 0;
    if (!metal_graph_alloc_raw_cap(&s->graph, &e->weights, shape_layer,
                                   raw_cap, (uint32_t)ctx_size, s->prefill_cap,
                                   need_spec_verifier,
                                   placement,
                                   e->cuda_tensor_parallel,
                                   shared_prefill_workspace))
    {
        free(s);
        return 1;
    }
    if (e->share_session_prefill_workspace &&
        !e->shared_prefill_workspace_ready) {
        bool workspace_ok = true;
        for (int t = 0; workspace_ok && t < DS4_MAX_GPUS; t++) {
            if (!s->graph.batch_cur_hc_by_tier[t]) continue;
            workspace_ok = metal_graph_ensure_batch_ffn_out_on(&s->graph, t);
        }
        if (!workspace_ok) {
            fprintf(stderr,
                    "ds4: failed to complete shared prefill workspace allocation\n");
            metal_graph_free(&s->graph);
            free(s);
            return 1;
        }
        const uint64_t workspace_bytes =
            metal_graph_prefill_workspace_bytes(&s->graph);
        metal_graph_transfer_prefill_workspace(
                &e->shared_prefill_workspace, &s->graph);
        e->shared_prefill_workspace_ready = true;
        fprintf(stderr,
                "ds4: shared session prefill workspace enabled: "
                "cap=%u, %.2f GiB total GPU memory; each additional "
                "session aliases this allocation\n",
                e->shared_prefill_workspace.prefill_cap,
                (double)workspace_bytes / 1073741824.0);
    }
    s->graph.quality = e->quality;
    s->graph.ssd_streaming = e->ssd_streaming;
    s->graph.ssd_streaming_cold = e->ssd_streaming_cold;
    s->graph.streaming_preload_experts = e->ssd_streaming_preload_experts;
    if (e->tp.active) {
        s->graph.tp_world = 2;
        s->graph.tp_rank = (uint32_t)e->tp.rank;
        s->graph.tp_out = e->tp.out_views;
        s->graph.tp_in = e->tp.in_views;
        s->graph.tp_batch_out = e->tp.batch_out_views;
        s->graph.tp_batch_in = e->tp.batch_in_views;
        s->graph.tp_zero = e->tp.zero_vec;
        const uint64_t half = (uint64_t)DS4_N_VOCAB / 2u;
        s->graph.tp_logits_half = ds4_gpu_tensor_view(
                metal_graph_logits(&s->graph),
                (uint64_t)e->tp.rank * half * sizeof(float),
                half * sizeof(float));
        if (!s->graph.tp_logits_half) {
            metal_graph_free(&s->graph);
            free(s);
            return 1;
        }
    }
    s->graph.power_percent = (uint32_t)e->power_percent;
    if (!metal_graph_load_directional_steering(&s->graph,
                                               e->directional_steering_file,
                                               e->directional_steering_attn_scale,
                                               e->directional_steering_ffn_scale)) {
        metal_graph_free(&s->graph);
        free(s);
        return 1;
    }
    if (e->support_kind == DS4_SUPPORT_DSPARK) {
        if (!metal_graph_configure_dspark_capture(&s->graph,
                                                  &e->dspark_weights)) {
            fprintf(stderr, "ds4: failed to configure DSpark target-hidden capture\n");
            metal_graph_free(&s->graph);
            free(s);
            return 1;
        }
        if (s->graph.dspark_capture_enabled) {
            fprintf(stderr, "ds4: DSpark target-hidden capture enabled: layers=");
            for (uint32_t i = 0; i < s->graph.dspark_target_layer_count; i++) {
                fprintf(stderr,
                        "%s%u",
                        i == 0 ? "" : ",",
                        s->graph.dspark_target_layers[i]);
            }
            fprintf(stderr, "\n");
        }
    }
    s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
    s->sample_probs =
        xmalloc((size_t)DS4_N_VOCAB * sizeof(s->sample_probs[0]));
    if (need_spec_verifier) {
        s->spec_row_logits =
            xmalloc((size_t)DS4_N_VOCAB * sizeof(s->spec_row_logits[0]));
    }
    if (e->support_kind == DS4_SUPPORT_DSPARK && e->dspark) {
        const uint64_t dspark_feature_count =
            (uint64_t)DS4_N_EMBD + (uint64_t)e->dspark_weights.markov_rank;
        if (dspark_feature_count <= (uint64_t)SIZE_MAX / sizeof(float)) {
            s->dspark_markov_bias =
                xmalloc((size_t)DS4_N_VOCAB *
                        sizeof(s->dspark_markov_bias[0]));
            s->dspark_conf_features =
                xmalloc((size_t)dspark_feature_count *
                        sizeof(s->dspark_conf_features[0]));
            s->dspark_conf_features_cap = (size_t)dspark_feature_count;
        }
    }
    if (e->mtp_ready) {
        s->mtp_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->mtp_logits[0]));
        s->mtp_draft_token = -1;
    }
    if (e->distributed.role == DS4_DISTRIBUTED_COORDINATOR) {
        char err[256];
        if (ds4_dist_session_create(&s->distributed,
                                    e,
                                    &e->distributed,
                                    s,
                                    ctx_size,
                                    err,
                                    sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4: failed to create distributed coordinator session: %s\n",
                    err[0] ? err : "unknown error");
            metal_graph_free(&s->graph);
            free(s->logits);
            free(s->sample_probs);
            free(s->mtp_logits);
            free(s->spec_row_logits);
            free(s->dspark_markov_bias);
            free(s->dspark_conf_features);
            free(s);
            return 1;
        }
    }
    if (!ds4_session_tp_register(s)) {
        ds4_session_free(s);
        return 1;
    }
    *out = s;
    return 0;
#endif
}

void ds4_session_free(ds4_session *s) {
    if (!s) return;
    if (ds4_session_tp_leader(s) && s->tp_session_id != 0 &&
        !ds4_tp_failed(s->engine->tp.ctx)) {
        char err[256] = "";
        if (!ds4_tp_send_session_destroy(s->engine->tp.ctx,
                                         s->tp_session_id) ||
            !ds4_tp_wait_command_ack(s->engine->tp.ctx, s->tp_session_id,
                                     "session destroy", err, sizeof(err))) {
            fprintf(stderr, "ds4: %s\n",
                    err[0] ? err : "tp: worker session destroy send failed");
        }
        s->tp_session_id = 0;
    }
#ifndef DS4_NO_GPU
    ds4_session_print_dspark_stats(s);
#endif
    ds4_dist_session_free(s->distributed);
    if (ds4_session_is_cpu(s)) {
        kv_cache_free(&s->cpu_cache);
        cpu_decode_scratch_free(&s->cpu_scratch);
    }
#ifndef DS4_NO_GPU
    else {
        if (ds4_session_is_glm(s)) {
            glm_graph_free(&s->glm_graph);
        } else {
            metal_graph_free(&s->graph);
        }
    }
#endif
    token_vec_free(&s->checkpoint);
    token_vec_free(&s->greedy_splitkv_segment);
    free(s->logits);
    free(s->sample_probs);
#ifndef DS4_NO_GPU
    free(s->glm_mtp_hc);
    free(s->glm_mtp_logits0);
#endif
    free(s->mtp_logits);
#ifndef DS4_NO_GPU
    free(s->spec_row_logits);
    free(s->dspark_markov_bias);
    free(s->dspark_conf_features);
#endif
    free(s);
}

int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen) {
    if (!s || !s->distributed) {
        if (errlen) snprintf(err, errlen, "session is not a distributed coordinator");
        return -1;
    }
    return ds4_dist_session_route_ready(s->distributed, err, errlen);
}

int ds4_session_power(ds4_session *s) {
    if (!s || !s->engine) return 100;
    return s->engine->power_percent;
}

bool ds4_session_is_distributed(ds4_session *s) {
    return s && s->distributed != NULL;
}

int ds4_session_set_power(ds4_session *s, int power_percent) {
    if (!s || !s->engine || power_percent < 1 || power_percent > 100) return 1;
#ifndef DS4_NO_GPU
    if (ds4_session_is_glm(s) && power_percent != 100) {
        fprintf(stderr, "ds4: session power throttling is not supported for GLM 5.2 yet\n");
        return 1;
    }
#endif
    s->engine->power_percent = power_percent;
#ifndef DS4_NO_GPU
    if (!ds4_session_is_cpu(s) && !ds4_session_is_glm(s)) s->graph.power_percent = (uint32_t)power_percent;
#endif
    return 0;
}

void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}

void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->display_progress = fn;
    s->display_progress_ud = ud;
}

void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud) {
    if (!s) return;
    s->cancel = fn;
    s->cancel_ud = ud;
}

static bool ds4_session_cancelled(ds4_session *s) {
    return s && s->cancel && s->cancel(s->cancel_ud);
}

static bool ds4_session_cancelled_cb(void *ud) {
    return ds4_session_cancelled(ud);
}

void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total) {
    if (!s || !s->progress || !event) return;
    s->progress(s->progress_ud, event, current, total);
}

int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen) {
    if (!s) {
        if (errlen) snprintf(err, errlen, "missing layer-slice session");
        return 1;
    }
    ds4_session_invalidate(s);
    if (ds4_session_is_cpu(s)) {
        session_cpu_reset_cache(s);
        return 0;
    }
#ifdef DS4_NO_GPU
    if (errlen) snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    if (ds4_session_is_glm(s)) {
        s->checkpoint.len = 0;
        s->checkpoint_valid = false;
        s->mtp_draft_valid = false;
        ds4_session_glm_reset_dense_cache(s);
        return 0;
    }
    if (!metal_graph_reset_prefill_state(&s->graph)) {
        if (errlen) snprintf(err, errlen, "%s layer-slice state reset failed",
                             ds4_backend_name(s->engine->backend));
        return 1;
    }
    s->graph.mtp_n_raw = 0;
    return 0;
#endif
}

int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err,
                                         size_t errlen) {
    if (!s || !s->engine || !hidden_hc || n_tokens == 0 || !logits) {
        if (errlen) snprintf(err, errlen, "invalid output-head hidden-state input");
        return 1;
    }

    ds4_engine *e = s->engine;
    if (!weights_have_output_head(&e->weights)) {
        if (errlen) snprintf(err, errlen, "output head is not loaded");
        return 1;
    }
    const uint64_t hidden_dim = ds4_engine_hidden_f32_values(e);
    const float *last_hc = hidden_hc + (uint64_t)(n_tokens - 1u) * hidden_dim;

    if (ds4_session_is_cpu(s)) {
        output_logits_one(logits, &e->model, &e->weights, last_hc);
        return 0;
    }
#ifdef DS4_NO_GPU
    (void)e;
    if (errlen) snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    if (ds4_session_is_glm(s)) {
        ds4_glm_gpu_graph *gg = &s->glm_graph;
        bool ok = ds4_gpu_tensor_write(gg->cur,
                                       0,
                                       last_hc,
                                       hidden_dim * sizeof(float)) != 0;
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = glm_graph_encode_output_head(gg, &e->model, &e->weights);
        if (ok) ok = ds4_gpu_end_commands() != 0;
        if (ok) ok = ds4_gpu_tensor_read(gg->logits,
                                         0,
                                         logits,
                                         (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: synchronize after GLM output-head hidden-state failure also failed\n");
            }
            if (errlen) snprintf(err, errlen, "%s GLM output-head hidden-state evaluation failed",
                                 ds4_backend_name(e->backend));
            return 1;
        }
        return 0;
    }
    ds4_gpu_graph *g = &s->graph;
    bool ok = ds4_gpu_tensor_write(metal_graph_cur_hc(g),
                                   0,
                                   last_hc,
                                   hidden_dim * sizeof(float)) != 0;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_output_head(g,
                                                &e->model,
                                                &e->weights,
                                                e->weights.output->dim[1]);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (ok) ok = ds4_gpu_tensor_read(metal_graph_logits(g),
                                     0,
                                     logits,
                                     (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: synchronize after output-head hidden-state failure also failed\n");
        }
        if (errlen) snprintf(err, errlen, "%s output-head hidden-state evaluation failed",
                             ds4_backend_name(e->backend));
        return 1;
    }
    return 0;
#endif
}


static int ds4_session_eval_internal(ds4_session *s, int token, bool probe_mtp,
                                     char *err, size_t errlen);

#ifndef DS4_NO_GPU
static int glm_session_logits_argmax(const float *logits) {
    int best = 0;
    float bv = logits[0];
    for (uint32_t i = 1; i < DS4_N_VOCAB; i++) {
        if (logits[i] > bv) { bv = logits[i]; best = (int)i; }
    }
    return best;
}

/* One GLM MTP speculative cycle (greedy): evaluate first_token, and when a
 * pending draft exists verify [first_token, draft] in one 2-row batch pass
 * (big gates only under TP). Returns the number of committed tokens (1 or
 * 2) with s->logits left at the last committed position, or -1 on error.
 * Both TP ranks run this off the same mirrored EVAL frame and derive
 * identical drafts/decisions from their identical logits. */
static int ds4_session_glm_spec_cycle(ds4_session *s, int first_token,
                                      int *accepted, int accepted_cap,
                                      char *err, size_t errlen) {
    ds4_engine *e = s->engine;
    ds4_glm_gpu_graph *g = &s->glm_graph;
    const uint32_t pos = (uint32_t)s->checkpoint.len;
    const uint32_t executable = glm_graph_normal_layer_count();
    const bool timing = s->engine->glm_mtp_timing;
    if (!s->glm_mtp_hc) {
        s->glm_mtp_hc = malloc(2ull * DS4_N_EMBD * sizeof(float));
        s->glm_mtp_logits0 = malloc((size_t)DS4_N_VOCAB * sizeof(float));
        if (!s->glm_mtp_hc || !s->glm_mtp_logits0) {
            if (errlen) snprintf(err, errlen, "glm mtp: out of memory");
            return -1;
        }
    }
    if (!s->glm_mtp_have || accepted_cap < 2 ||
        pos + 2 > g->ctx_size ||
        (g->compact_cache_cap != 0 && pos + 1 >= g->compact_cache_cap)) {
        /* Plain step, then seed the first draft from g->cur. */
        s->glm_spec_inside = 1;
        const int rc = ds4_session_eval_internal(s, first_token, false, err, errlen);
        s->glm_spec_inside = 0;
        if (rc != 0) return -1;
        const int n1 = glm_session_logits_argmax(s->logits);
        if (s->glm_mtp_min_pos == 0 || s->glm_mtp_min_pos > pos) {
            s->glm_mtp_min_pos = pos;
        }
        int d = -1;
        s->glm_mtp_have = 0;
        if (glm_graph_mtp_step(g, &e->model, &e->weights, n1, pos,
                               s->glm_mtp_min_pos, &d)) {
            s->glm_mtp_draft = d;
            s->glm_mtp_have = 1;
        }
        accepted[0] = first_token;
        return 1;
    }

    const double t0 = timing ? now_sec() : 0.0;
    s->glm_mtp_have = 0;
    const int d = s->glm_mtp_draft;
    int toks[2] = { first_token, d };
    /* Verify MUST run in the same compact/indexed attention world the
     * decode path writes: the full-KV batch branch attends over rows the
     * indexed decode never populates, corrupting every prediction. */
    (void)executable;
    /* Fast decode-style verify while the causal window covers everything
     * the decode path would attend (pre-indexer-selection regime); the
     * indexed batch path remains the fallback beyond it. */
    bool verified = false;
    if (pos + 2u <= glm_graph_indexer_top_k_limit()) {
        verified = glm_graph_verify_rows(g, &e->model, &e->weights,
                                         toks, pos, 2,
                                         s->glm_mtp_hc, s->logits);
    }
    if (!verified) {
        if (!glm_graph_indexed_prefill_batch_ready(g, pos) ||
            glm_graph_limit_indexed_prefill_chunk(pos, 2u) < 2u ||
            !glm_graph_forward_indexed_tokens(g, &e->model, &e->weights,
                                              toks, NULL, pos, 2,
                                              s->glm_mtp_hc, s->logits,
                                              NULL, NULL, pos, 0, 2)) {
            if (errlen) snprintf(err, errlen, "glm mtp: verify failed");
            s->checkpoint_valid = false;
            return -1;
        }
    }
    const double t1 = timing ? now_sec() : 0.0;
    /* Row0 logits through the shared head. */
    if (!glm_graph_mtp_ensure(g)) {
        if (errlen) snprintf(err, errlen, "glm mtp: scratch alloc failed");
        return -1;
    }
    ds4_gpu_tensor *h0_view =
        ds4_gpu_tensor_view(g->mtp_concat, 0, (uint64_t)DS4_N_EMBD * sizeof(float));
    if (!h0_view) {
        if (errlen) snprintf(err, errlen, "glm mtp: hidden view failed");
        return -1;
    }
    bool head_ok =
        ds4_gpu_tensor_write(h0_view, 0, s->glm_mtp_hc,
                             (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
        glm_graph_forward_output_head(g, &e->model, &e->weights,
                                      h0_view, s->glm_mtp_logits0);
    ds4_gpu_tensor_free(h0_view);
    if (!head_ok) {
        if (errlen) snprintf(err, errlen, "glm mtp: row0 head failed");
        return -1;
    }
    const int n1 = glm_session_logits_argmax(s->glm_mtp_logits0);
    token_vec_push(&s->checkpoint, first_token);
    s->checkpoint_valid = true;
    const int accept = n1 == d;
    int n_committed = 1;
    if (accept) {
        token_vec_push(&s->checkpoint, d);
        ds4_session_glm_note_dense_cache(s, pos, 2);
        n_committed = 2;
        /* s->logits already holds row1 (position pos+1) logits. */
        const int n2 = glm_session_logits_argmax(s->logits);
        int dummy = -1, nd = -1;
        const bool cu =
            ds4_gpu_tensor_write(g->cur, 0, s->glm_mtp_hc,
                                 (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
            glm_graph_mtp_step(g, &e->model, &e->weights, d, pos,
                               s->glm_mtp_min_pos, &dummy) &&
            ds4_gpu_tensor_write(g->cur, 0, s->glm_mtp_hc + DS4_N_EMBD,
                                 (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
            glm_graph_mtp_step(g, &e->model, &e->weights, n2, pos + 1u,
                               s->glm_mtp_min_pos, &nd);
        if (cu) {
            s->glm_mtp_draft = nd;
            s->glm_mtp_have = 1;
        }
        accepted[0] = first_token;
        accepted[1] = d;
    } else {
        ds4_session_glm_note_dense_cache(s, pos, 1);
        memcpy(s->logits, s->glm_mtp_logits0,
               (size_t)DS4_N_VOCAB * sizeof(float));
        int nd = -1;
        const bool cu =
            ds4_gpu_tensor_write(g->cur, 0, s->glm_mtp_hc,
                                 (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
            glm_graph_mtp_step(g, &e->model, &e->weights, n1, pos,
                               s->glm_mtp_min_pos, &nd);
        if (cu) {
            s->glm_mtp_draft = nd;
            s->glm_mtp_have = 1;
        }
        accepted[0] = first_token;
    }
    if (timing) {
        const double t2 = now_sec();
        char *dt = ds4_token_text(e, d, NULL);
        char *nt = ds4_token_text(e, n1, NULL);
        fprintf(stderr,
                "ds4: glm mtp cycle: verify2 %.1f ms, head+draft %.1f ms, %s "
                "(draft %d '%s' vs true %d '%s')\n",
                (t1 - t0) * 1000.0, (t2 - t1) * 1000.0,
                accept ? "ACCEPT" : "reject",
                d, dt ? dt : "?", n1, nt ? nt : "?");
        free(dt);
        free(nt);
    }
    return n_committed;
}
#endif

static int ds4_session_slice_check_timeline(
        ds4_session *s,
        const int   *tokens,
        uint32_t     n_tokens,
        uint32_t     pos0,
        char        *err,
        size_t       errlen) {
    if (!s || !tokens || n_tokens == 0) {
        if (errlen) snprintf(err, errlen, "invalid layer-slice token span");
        return 1;
    }
    const uint32_t ctx_size = (uint32_t)s->ctx_size;
    if (pos0 > (uint32_t)INT_MAX || n_tokens > (uint32_t)INT_MAX ||
        pos0 > ctx_size || n_tokens > ctx_size - pos0) {
        if (errlen) snprintf(err, errlen, "layer-slice token span exceeds context");
        return 1;
    }
    if (!s->checkpoint_valid) {
        if (pos0 != 0) {
            if (errlen) snprintf(err, errlen, "layer-slice session needs reset before pos %u", pos0);
            return 1;
        }
        return 0;
    }
    if ((uint32_t)s->checkpoint.len != pos0) {
        if (errlen) snprintf(err, errlen, "layer-slice KV position mismatch: have %d want %u",
                             s->checkpoint.len, pos0);
        return 1;
    }
    return 0;
}

static DS4_MAYBE_UNUSED void ds4_session_slice_commit_timeline(ds4_session *s, const int *tokens, uint32_t n_tokens) {
    for (uint32_t i = 0; i < n_tokens; i++) token_vec_push(&s->checkpoint, tokens[i]);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    ds4_session_dspark_capture_note_checkpoint(s);
}

int ds4_session_eval_layer_slice(ds4_session *s,
                                 const int *tokens,
                                 uint32_t n_tokens,
                                 uint32_t pos0,
                                 uint32_t layer_start,
                                 uint32_t layer_end,
                                 const float *input_hc,
                                 float *output_hc,
                                 bool output_logits,
                                 float *logits,
                                 char *err,
                                 size_t errlen) {
    if (!s || !s->engine) {
        if (errlen) snprintf(err, errlen, "missing layer-slice session");
        return 1;
    }
    const uint32_t executable_layers = ds4_model_normal_layer_count();
    if (executable_layers == 0 ||
        layer_start > layer_end ||
        layer_end >= executable_layers) {
        if (errlen) snprintf(err, errlen, "invalid layer-slice layer range %u:%u",
                             layer_start, layer_end);
        return 1;
    }
    if (layer_start != 0 && !input_hc) {
        if (errlen) snprintf(err, errlen, "layer-slice layer %u requires input hidden-state",
                             layer_start);
        return 1;
    }
    if (output_logits && layer_end + 1u != executable_layers) {
        if (errlen) snprintf(err, errlen, "layer-slice logits require final transformer layer");
        return 1;
    }
    if (output_logits && !logits) {
        if (errlen) snprintf(err, errlen, "layer-slice logits output is missing");
        return 1;
    }
    if (!weights_layers_bound(&s->engine->weights, layer_start, layer_end)) {
        if (errlen) snprintf(err, errlen, "requested layer slice %u:%u is not loaded",
                             layer_start, layer_end);
        return 1;
    }
    if (!input_hc && !s->engine->weights.token_embd) {
        if (errlen) snprintf(err, errlen, "token embedding is not loaded");
        return 1;
    }
    if (output_logits && !weights_have_output_head(&s->engine->weights)) {
        if (errlen) snprintf(err, errlen, "output head is not loaded");
        return 1;
    }
    /* A distributed prefill pipeline may need only the KV side effect for
     * non-final chunks. In that case both output_hc and logits are NULL. */
    if (ds4_session_slice_check_timeline(s, tokens, n_tokens, pos0, err, errlen) != 0) {
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        if (errlen) snprintf(err, errlen, "layer slices require the graph backend");
        s->checkpoint_valid = false;
        return 1;
    }
#ifdef DS4_NO_GPU
    (void)output_hc;
    if (errlen) snprintf(err, errlen, "GPU support is not compiled in");
    s->checkpoint_valid = false;
    return 1;
#else
    if (ds4_session_is_glm(s)) {
        ds4_engine *e = s->engine;
        ds4_glm_gpu_graph *g = &s->glm_graph;
        if (!s->glm_graph_ready) {
            if (errlen) snprintf(err, errlen, "%s GLM graph is not initialized",
                                 ds4_backend_name(e->backend));
            return 1;
        }
        if (layer_start != g->layer_start || layer_end != g->layer_end) {
            if (errlen) snprintf(err, errlen,
                                 "requested GLM layer slice %u:%u does not match loaded slice %u:%u",
                                 layer_start,
                                 layer_end,
                                 g->layer_start,
                                 g->layer_end);
            s->checkpoint_valid = false;
            return 1;
        }
        if (n_tokens > s->prefill_cap) {
            if (errlen) snprintf(err, errlen, "GLM layer-slice chunk %u exceeds prefill cap %u",
                                 n_tokens, s->prefill_cap);
            return 1;
        }

        const uint64_t hidden_dim = DS4_N_EMBD;
#ifdef DS4_ROCM_BUILD
        const bool rocm_layer_slice_token_decode =
            glm_graph_env_truthy(
                    getenv("DS4_ROCM_GLM_LAYER_SLICE_TOKEN_DECODE"));
#else
        const bool rocm_layer_slice_token_decode = false;
#endif
        uint32_t done = 0;
        while (done < n_tokens) {
            const uint32_t pos = pos0 + done;
            const uint32_t remaining = n_tokens - done;
            const float *chunk_input = input_hc ? input_hc + (uint64_t)done * hidden_dim : NULL;
            float *chunk_output = output_hc ? output_hc + (uint64_t)done * hidden_dim : NULL;
            uint32_t chunk = 1;
            bool ok = false;

            /*
             * The token graph accepts both embeddings and inter-node hidden
             * states. Keep the resident ROCm continuation path opt-in until
             * remote output and timing tests validate it across GLM quants.
             */
            if (remaining == 1 && pos > 0 &&
                ((!input_hc && !output_hc) ||
                 rocm_layer_slice_token_decode)) {
                float *chunk_logits = output_logits ? logits : NULL;
                ok = glm_graph_forward_token(g,
                                             &e->model,
                                             &e->weights,
                                             tokens[done],
                                             chunk_input,
                                             pos,
                                             chunk_output,
                                             chunk_logits,
                                             false);
                if (ok && glm_graph_decode_updates_dense_cache(g, pos, chunk_logits)) {
                    ds4_session_glm_note_dense_cache(s, pos, 1);
                }
            } else if (g->full_kv_cache && pos < g->ctx_cap) {
                chunk = remaining;
                const uint32_t dense_remaining = g->ctx_cap - pos;
                if (chunk > dense_remaining) chunk = dense_remaining;
                float *chunk_logits = (output_logits && done + chunk == n_tokens) ? logits : NULL;
                ok = glm_graph_forward_tokens(g,
                                              &e->model,
                                              &e->weights,
                                              tokens + done,
                                              chunk_input,
                                              pos,
                                              chunk,
                                              chunk_output,
                                              chunk_logits,
                                              NULL,
                                              NULL,
                                              pos0,
                                              done,
                                              n_tokens);
                if (ok) ds4_session_glm_note_dense_cache(s, pos, chunk);
            } else if (glm_graph_indexed_prefill_batch_ready(g, pos)) {
                chunk = remaining;
                if (chunk > g->indexed_prefill_cap) chunk = g->indexed_prefill_cap;
                chunk = glm_graph_limit_indexed_prefill_chunk(pos, chunk);
                if (chunk == 0) chunk = 1;
                float *chunk_logits = (output_logits && done + chunk == n_tokens) ? logits : NULL;
                ok = glm_graph_forward_indexed_tokens(g,
                                                      &e->model,
                                                      &e->weights,
                                                      tokens + done,
                                                      chunk_input,
                                                      pos,
                                                      chunk,
                                                      chunk_output,
                                                      chunk_logits,
                                                      NULL,
                                                      NULL,
                                                      pos0,
                                                      done,
                                                      n_tokens);
            } else {
                float *chunk_logits = (output_logits && done + 1u == n_tokens) ? logits : NULL;
                ok = glm_graph_forward_token(g,
                                             &e->model,
                                             &e->weights,
                                             tokens[done],
                                             chunk_input,
                                             pos,
                                             chunk_output,
                                             chunk_logits,
                                             false);
                chunk = 1;
            }

            if (!ok) {
                if (errlen) snprintf(err, errlen,
                                     "%s GLM layer-slice evaluation failed at pos %u",
                                     ds4_backend_name(e->backend),
                                     pos);
                s->checkpoint_valid = false;
                s->mtp_draft_valid = false;
                ds4_session_glm_cap_dense_cache(s);
                return 1;
            }
            done += chunk;
        }
        if (!output_hc && !output_logits && ds4_gpu_synchronize() == 0) {
            if (errlen) snprintf(err, errlen,
                                 "%s GLM layer-slice synchronization failed",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            s->mtp_draft_valid = false;
            ds4_session_glm_cap_dense_cache(s);
            return 1;
        }
        ds4_session_slice_commit_timeline(s, tokens, n_tokens);
        return 0;
    }
    if (n_tokens > s->prefill_cap) {
        if (errlen) snprintf(err, errlen, "layer-slice chunk %u exceeds prefill cap %u",
                             n_tokens, s->prefill_cap);
        return 1;
    }

    ds4_engine *e = s->engine;
    ds4_gpu_graph *g = &s->graph;
    if (!input_hc && !output_hc && output_logits &&
        layer_start == 0 && layer_end + 1u == (uint32_t)DS4_N_LAYER) {
        bool ok = false;
        ds4_tokens span = {0};
        if (pos0 == 0) {
            span.v = (int *)tokens;
            span.len = (int)n_tokens;
            span.cap = (int)n_tokens;
            ok = metal_graph_prefill_layer_major(g,
                                                 &e->model,
                                                 &e->weights,
                                                 &span,
                                                 0,
                                                 n_tokens,
                                                 logits,
                                                 false,
                                                 NULL,
                                                 NULL,
                                                 NULL);
        } else if (n_tokens == 1) {
            ok = metal_graph_eval_token_raw_swa(g,
                                                &e->model,
                                                &e->weights,
                                                tokens[0],
                                                pos0,
                                                logits);
        } else {
            if (pos0 > (uint32_t)INT_MAX - n_tokens) {
                if (errlen) snprintf(err, errlen, "layer-slice full span is too large");
                s->checkpoint_valid = false;
                return 1;
            }
            span.len = (int)(pos0 + n_tokens);
            span.cap = span.len;
            span.v = calloc((size_t)span.len, sizeof(span.v[0]));
            if (span.v) {
                for (uint32_t i = 0; i < n_tokens; i++) span.v[pos0 + i] = tokens[i];
                ok = metal_graph_prefill_layer_major(g,
                                                     &e->model,
                                                     &e->weights,
                                                     &span,
                                                     pos0,
                                                     n_tokens,
                                                     logits,
                                                     false,
                                                     NULL,
                                                     NULL,
                                                     NULL);
            }
            free(span.v);
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: synchronize after layer-slice full failure also failed\n");
            }
            if (errlen) snprintf(err, errlen, "%s layer-slice full evaluation failed",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return 1;
        }
        ds4_session_slice_commit_timeline(s, tokens, n_tokens);
        return 0;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = (uint64_t)n_tokens * hc_dim * sizeof(float);
    if (n_tokens == 1 && pos0 > 0) {
        if (g->raw_cap == 0) {
            if (errlen) snprintf(err, errlen, "%s layer-slice decode has no raw KV cache",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return 1;
        }

        bool ok = true;
        if (g->ssd_streaming && !input_hc) {
            g->streaming_static_decode_map_current = false;
            ok = metal_graph_stream_map_token(&e->model, &e->weights);
        }
        if (input_hc) {
            ok = ds4_gpu_tensor_write(metal_graph_cur_hc(g), 0, input_hc, hc_dim * sizeof(float)) != 0;
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok && !input_hc) {
            ok = ds4_gpu_embed_token_hc_tensor(metal_graph_cur_hc(g),
                                               e->model.map,
                                               e->model.size,
                                               e->weights.token_embd->abs_offset,
                                               (uint32_t)e->weights.token_embd->dim[1],
                                               (uint32_t)tokens[0],
                                               DS4_N_EMBD,
                                               DS4_N_HC) != 0;
        }
        const uint32_t raw_row = pos0 % g->raw_cap;
        const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos0, 1);
        const uint32_t split_after_layers = metal_graph_token_split_after_layers();
        uint32_t encoded_layers = 0;
        if (g->ssd_streaming) {
            if (ok) ok = ds4_gpu_end_commands() != 0;
            for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
                g->streaming_static_decode_map_current = false;
                ok = metal_graph_stream_map_layer_decode(&e->model, &e->weights, il);
                if (ok) ok = ds4_gpu_begin_commands() != 0;
                if (ok) {
                    ok = metal_graph_encode_decode_layer(g,
                                                         &e->model,
                                                         &e->weights.layer[il],
                                                         il,
                                                         pos0,
                                                         g->layer_raw_cache[il],
                                                         g->raw_cap,
                                                         raw_row,
                                                         n_raw,
                                                         tokens[0]);
                    ds4_gpu_tensor *tmp = metal_graph_cur_hc(g);
                    g->cur_hc_by_tier[g->active_tier] = metal_graph_after_ffn_hc(g);
                    g->after_ffn_hc_by_tier[g->active_tier] = tmp;
                }
                if (ok) ok = ds4_gpu_end_commands() != 0;
            }
            if (ok && output_logits) {
                g->streaming_static_decode_map_current = false;
                ok = metal_graph_stream_map_output(&e->model, &e->weights);
                if (ok) ok = ds4_gpu_begin_commands() != 0;
                if (ok) ok = metal_graph_encode_output_head(g, &e->model, &e->weights, e->weights.output->dim[1]);
                if (ok) ok = ds4_gpu_end_commands() != 0;
            }
        } else {
            for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
                ok = metal_graph_encode_decode_layer(g,
                                                     &e->model,
                                                     &e->weights.layer[il],
                                                     il,
                                                     pos0,
                                                     g->layer_raw_cache[il],
                                                     g->raw_cap,
                                                     raw_row,
                                                     n_raw,
                                                     tokens[0]);
                ds4_gpu_tensor *tmp = metal_graph_cur_hc(g);
                g->cur_hc_by_tier[g->active_tier] = metal_graph_after_ffn_hc(g);
                g->after_ffn_hc_by_tier[g->active_tier] = tmp;
                encoded_layers++;
                if (ok &&
                    split_after_layers != 0 &&
                    encoded_layers == split_after_layers &&
                    il < layer_end)
                {
                    ok = ds4_gpu_flush_commands() != 0;
                }
            }
            if (ok && output_logits) {
                ok = metal_graph_encode_output_head(g, &e->model, &e->weights, e->weights.output->dim[1]);
            }
            if (ok) ok = ds4_gpu_end_commands() != 0;
        }
        if (ok && !output_hc && !output_logits) ok = ds4_gpu_synchronize() != 0;
        if (ok && output_hc) {
            ok = ds4_gpu_tensor_read(metal_graph_cur_hc(g), 0, output_hc, hc_dim * sizeof(float)) != 0;
        }
        if (ok && output_logits) {
            ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: synchronize after layer-slice decode failure also failed\n");
            }
            if (errlen) snprintf(err, errlen, "%s layer-slice decode failed",
                                 ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return 1;
        }

        ds4_session_slice_commit_timeline(s, tokens, n_tokens);
        return 0;
    }

    ds4_tokens span = {
        .v = (int *)tokens,
        .len = (int)n_tokens,
        .cap = (int)n_tokens,
    };

    bool ok = true;
    if (g->ssd_streaming && !input_hc) {
        g->streaming_static_decode_map_current = false;
        ok = metal_graph_stream_map_token(&e->model, &e->weights);
    }
    if (ok) ok = metal_graph_upload_prompt_tokens(metal_graph_prefill_tokens(g), &span, 0, n_tokens);
    if (ok && input_hc) {
        ok = ds4_gpu_tensor_write(metal_graph_batch_cur_hc(g), 0, input_hc, hc_bytes) != 0;
    } else if (ok) {
        ok = metal_graph_upload_prompt_embeddings_hc(metal_graph_batch_cur_hc(g),
                                                     metal_graph_prefill_tokens(g),
                                                     &e->model,
                                                     &e->weights,
                                                     &span,
                                                     0,
                                                     n_tokens);
    }

    ds4_gpu_tensor *last_hc = NULL;
    ds4_gpu_tensor *saved_cur = NULL;
    const int src_tier = g->active_tier;
    const bool batch_selected_addr =
        g->ssd_streaming &&
        layer_start == 0 &&
        (metal_graph_stream_prefill_batch_selected_addr_enabled(g, &e->weights, n_tokens) ||
         metal_graph_cuda_stream_prefill_batch_selected_addr_enabled(g, &e->weights, n_tokens));
    if (g->ssd_streaming) {
        for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
            g->streaming_static_decode_map_current = false;
            const bool layer_selected_addr =
                batch_selected_addr &&
                metal_graph_stream_prefill_batch_selected_addr_layer_supported(
                    &e->weights,
                    il);
            ok = layer_selected_addr ?
                 metal_graph_stream_map_layer_decode(&e->model, &e->weights, il) :
                 metal_graph_stream_map_layer(&e->model, &e->weights, il);
            if (ok) ok = ds4_gpu_begin_commands() != 0;
            if (ok) {
                ok = metal_graph_encode_layer_batch(g,
                                                    &e->model,
                                                    &e->weights.layer[il],
                                                    il,
                                                    pos0,
                                                    n_tokens);
            }
            if (ok) ok = ds4_gpu_end_commands() != 0;
        }
    } else {
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        for (uint32_t il = layer_start; ok && il <= layer_end; il++) {
            ok = metal_graph_encode_layer_batch(g,
                                                &e->model,
                                                &e->weights.layer[il],
                                                il,
                                                pos0,
                                                n_tokens);
        }
    }
    if (ok && output_logits) {
        saved_cur = g->cur_hc_by_tier[src_tier];
        last_hc = metal_graph_tensor_row_view(metal_graph_batch_cur_hc(g), n_tokens - 1u, hc_dim);
        ok = last_hc != NULL;
        if (ok && g->ssd_streaming) {
            g->streaming_static_decode_map_current = false;
            ok = metal_graph_stream_map_output(&e->model, &e->weights);
        }
        if (ok) {
            g->cur_hc_by_tier[src_tier] = last_hc;
            if (g->ssd_streaming) ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_output_head(g, &e->model, &e->weights, e->weights.output->dim[1]);
            if (ok && g->ssd_streaming) ok = ds4_gpu_end_commands() != 0;
            g->cur_hc_by_tier[src_tier] = saved_cur;
        }
    }
    if (ok && !g->ssd_streaming) ok = ds4_gpu_end_commands() != 0;
    if (saved_cur) g->cur_hc_by_tier[src_tier] = saved_cur;
    if (last_hc) ds4_gpu_tensor_free(last_hc);

    if (ok && !output_hc && !output_logits) ok = ds4_gpu_synchronize() != 0;
    if (ok && output_hc) {
        ok = ds4_gpu_tensor_read(metal_graph_batch_cur_hc(g), 0, output_hc, hc_bytes) != 0;
    }
    if (ok && output_logits) {
        ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: synchronize after layer-slice failure also failed\n");
        }
        if (errlen) snprintf(err, errlen, "%s layer-slice failed",
                             ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return 1;
    }

    ds4_session_slice_commit_timeline(s, tokens, n_tokens);
    return 0;
#endif
}

#ifndef DS4_NO_GPU
typedef struct {
    ds4_session *session;
    const ds4_tokens *prompt;
    ds4_session_progress_fn user;
    void *user_ud;
} ds4_sync_progress;

static void ds4_session_note_prefill_progress(void *ud, const char *event, int current, int total) {
    ds4_sync_progress *p = ud;
    if (!p || !p->session || !p->prompt) return;
    if (!strcmp(event, "prefill_chunk") && current > 0 && current <= p->prompt->len) {
        p->session->checkpoint.len = 0;
        for (int i = 0; i < current; i++) token_vec_push(&p->session->checkpoint, p->prompt->v[i]);
        p->session->checkpoint_valid = true;
        p->session->mtp_draft_valid = false;
        ds4_session_dspark_capture_note_checkpoint(p->session);
    }
    if (p->user) p->user(p->user_ud, event, current, total);
}
#endif

/* Bring the live backend state to exactly the supplied token prefix.
 *
 * ds4-server and the REPL are stateless at the text/API layer but stateful here:
 * they resend or rebuild the full transcript, and this function decides whether
 * the live checkpoint is a prefix.  A matching prefix is extended in one of two
 * ways:
 *
 *   - long suffix: batched layer-major prefill, aligned to absolute chunk
 *     boundaries so compressor/indexer rows finalize in the same order as a
 *     cold prompt;
 *   - short suffix: ordinary one-token decode, which is faster below the
 *     measured crossover and preserves exact autoregressive semantics.
 *
 * A non-matching prompt discards the checkpoint and prefills from token zero.
 */
static int ds4_session_sync_internal(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);

/* Under tensor parallelism the leader mirrors every public sync/eval to the
 * worker before doing the work itself, so both engines execute the same
 * graph sequence and the per-layer gates pair up.  The worker acks a sync
 * once its matching prefill completes, surfacing worker-side failures
 * here instead of as a gate timeout mid-decode. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    const bool mirror = ds4_session_tp_leader(s);
    if (mirror && prompt && prompt->len > 0) {
        if (!ds4_tp_send_sync(s->engine->tp.ctx, s->tp_session_id,
                              prompt->v, (uint32_t)prompt->len)) {
            snprintf(err, errlen, "tp: worker sync send failed");
            return 1;
        }
    }
    int rc = ds4_session_sync_internal(s, prompt, err, errlen);
#ifndef DS4_NO_GPU
    if (rc == 0) glm_debug_dump_prefill_logits(s->logits);
    if (rc == 0) {
        const char *kvp = getenv("DS4_GLM_KV_DUMP");
        if (kvp && kvp[0] && s->glm_graph.layer_kv_lora_cache[0]) {
            const ds4_glm_gpu_graph *g = &s->glm_graph;
            const uint32_t rows = prompt ? (uint32_t)prompt->len : 0;
            const uint64_t eb = glm_graph_compact_cache_elem_bytes();
            const struct { const char *sfx; ds4_gpu_tensor *t; uint64_t rb; } kd[2] = {
                { "lora0", g->layer_kv_lora_cache[0], DS4_N_KV_LORA * eb },
                { "rope0", g->layer_k_rope_cache[0],  DS4_N_ROT * eb },
            };
            for (int i = 0; i < 2 && rows; i++) {
                char fp[1024];
                snprintf(fp, sizeof(fp), "%s.%s", kvp, kd[i].sfx);
                void *buf = malloc(rows * kd[i].rb);
                if (buf &&
                    ds4_gpu_tensor_read(kd[i].t, 0, buf, rows * kd[i].rb)) {
                    FILE *f = fopen(fp, "wb");
                    if (f) { fwrite(buf, 1, rows * kd[i].rb, f); fclose(f); }
                }
                free(buf);
            }
        }
    }
#endif
    if (mirror) {
        const bool worker_ok = ds4_tp_wait_command_ack(
            s->engine->tp.ctx, s->tp_session_id, "prefill sync", err, errlen);
        bool logits_ok = true;
        /* A successful worker sends its split logits even if the leader's
         * local prefill failed. Drain them to keep the control stream framed
         * before invalidating the mirrored session. */
        if (worker_ok && s->engine->tp.vocab_split) {
            const uint32_t vhalf = (uint32_t)DS4_N_VOCAB / 2u;
            if (!ds4_tp_recv_logits_half(s->engine->tp.ctx, s->logits + vhalf, vhalf)) {
                snprintf(err, errlen, "tp: worker sync logits half missing");
                logits_ok = false;
            }
        }
        if (rc != 0 || !worker_ok || !logits_ok) {
            ds4_session_invalidate(s);
            return rc != 0 ? rc : 1;
        }
    }
    return rc;
}

static int ds4_session_sync_internal(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    if (!s || !prompt) {
        snprintf(err, errlen, "missing session or prompt");
        return 1;
    }
    if (prompt->len <= 0) {
        snprintf(err, errlen, "empty prompt");
        return 1;
    }
    if (prompt->len >= s->ctx_size) {
        snprintf(err, errlen,
                 "prompt length %d exceeds context %d (one token of generation room is required)",
                 prompt->len, s->ctx_size);
        return 1;
    }
    if (ds4_session_cancelled(s)) {
        snprintf(err, errlen, "interrupted");
        return DS4_SESSION_SYNC_INTERRUPTED;
    }
    if (s->distributed) {
        const ds4_tokens *checkpoint = s->checkpoint_valid ? &s->checkpoint : NULL;
        return ds4_dist_session_sync(s->distributed,
                                     s,
                                     checkpoint,
                                     prompt,
                                     s->logits,
                                     err,
                                     errlen);
    }
    if (ds4_session_is_cpu(s)) {
        ds4_engine *e = s->engine;
        if (s->checkpoint_valid &&
            prompt->len >= s->checkpoint.len &&
            ds4_tokens_starts_with(prompt, &s->checkpoint))
        {
            s->mtp_draft_valid = false;
            for (int i = s->checkpoint.len; i < prompt->len; i++) {
                if (ds4_session_cancelled(s)) {
                    snprintf(err, errlen, "interrupted");
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    return DS4_SESSION_SYNC_INTERRUPTED;
                }
                forward_token_raw_swa_cpu_decode_scratch(s->logits,
                                                         &e->model,
                                                         &e->weights,
                                                         &s->cpu_cache,
                                                         prompt->v[i],
                                                         (uint32_t)s->checkpoint.len,
                                                         e->directional_steering_dirs,
                                                         e->directional_steering_attn_scale,
                                                         e->directional_steering_ffn_scale,
                                                         &s->cpu_scratch);
                token_vec_push(&s->checkpoint, prompt->v[i]);
                if (s->progress) s->progress(s->progress_ud, "prefill_chunk", i + 1, prompt->len);
            }
            s->checkpoint_valid = true;
            s->greedy_splitkv_segment.len = 0;
            s->greedy_splitkv_anchor_valid = false;
            return 0;
        }

        session_cpu_reset_cache(s);
        prefill_layer_major_cpu(s->logits,
                                &e->model,
                                &e->weights,
                                &s->cpu_cache,
                                prompt,
                                e->directional_steering_dirs,
                                e->directional_steering_attn_scale,
                                e->directional_steering_ffn_scale);
        ds4_tokens_copy(&s->checkpoint, prompt);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        ds4_session_dspark_capture_note_checkpoint(s);
        if (s->progress) s->progress(s->progress_ud, "prefill_chunk", prompt->len, prompt->len);
        return 0;
    }

#ifdef DS4_NO_GPU
    (void)s;
    (void)prompt;
    snprintf(err, errlen, "GPU support is not compiled in");
    return 1;
#else
    {
    ds4_engine *e = s->engine;
    const char *backend_name = ds4_backend_name(e->backend);
    (void)backend_name; (void)e;
    if (ds4_session_is_glm(s)) {
        /* Debug: truncate the prompt so the dumped prefill logits line up
         * with the CPU first-token reference (DS4_GLM_LOGIT_DUMP). */
        ds4_tokens glm_trunc_prompt;
        {
            const char *tr = getenv("DS4_GLM_PREFILL_TRUNC");
            if (tr && tr[0]) {
                const int tn = atoi(tr);
                if (tn > 0 && tn < prompt->len) {
                    glm_trunc_prompt = *prompt;
                    glm_trunc_prompt.len = tn;
                    prompt = &glm_trunc_prompt;
                }
            }
        }
        if (!s->glm_graph_ready) {
            snprintf(err, errlen, "%s GLM graph is not initialized", backend_name);
            return 1;
        }
        if ((uint32_t)prompt->len >= s->glm_graph.ctx_size) {
            snprintf(err, errlen, "prompt length %d leaves no GLM Metal context room (ctx %u)",
                     prompt->len, s->glm_graph.ctx_size);
            s->checkpoint_valid = false;
            ds4_session_glm_reset_dense_cache(s);
            return 1;
        }

        int start = 0;
        bool resumed_checkpoint = false;
        if (s->checkpoint_valid &&
            prompt->len >= s->checkpoint.len &&
            ds4_tokens_starts_with(prompt, &s->checkpoint))
        {
            start = s->checkpoint.len;
            resumed_checkpoint = true;
            s->mtp_draft_valid = false;
        } else {
            s->checkpoint.len = 0;
            s->checkpoint_valid = false;
            s->mtp_draft_valid = false;
            ds4_session_glm_reset_dense_cache(s);
        }

        uint32_t glm_exact_prefill_max = 64;
        {
            const char *em = getenv("DS4_GLM_TP_EXACT_PREFILL_MAX");
            if (em && em[0]) glm_exact_prefill_max = (uint32_t)atoi(em);
        }
        if (s->engine->glm_tp_token_prefill ||
            s->glm_graph.placement != NULL ||
            (s->glm_graph.tp_world == 2 &&
             (uint32_t)(prompt->len - start) <= glm_exact_prefill_max)) {
            /* Multi-tier GLM has per-tier decode workspaces and KV caches;
             * its large batch-prefill workspace is not mirrored across
             * devices. Small TP prompts also use this path to retain exact
             * single-node arithmetic. */
            for (int i = start; i < prompt->len; i++) {
                const uint32_t pos = (uint32_t)i;
                const bool updates_dense =
                    glm_graph_decode_updates_dense_cache(&s->glm_graph, pos,
                                                         s->logits);
                const bool last = i + 1 == prompt->len;
                /* Always request logits: it forces the per-token commit
                 * cadence of normal decode, so the per-layer gates execute
                 * token by token on both ranks instead of piling up in one
                 * giant open batch that times out at the late flush. */
                if (!glm_graph_forward_token(&s->glm_graph,
                                             &s->engine->model,
                                             &s->engine->weights,
                                             prompt->v[i],
                                             NULL,
                                             pos,
                                             NULL,
                                             s->logits,
                                             false))
                {
                    snprintf(err, errlen, "%s GLM TP prefill failed at token %d",
                             backend_name, i);
                    s->checkpoint_valid = false;
                    ds4_session_glm_cap_dense_cache(s);
                    return 1;
                }
                token_vec_push(&s->checkpoint, prompt->v[i]);
                if (updates_dense) ds4_session_glm_note_dense_cache(s, pos, 1);
                if (s->progress && ((i - start) % 8 == 0 || last)) {
                    s->progress(s->progress_ud, "prefill_chunk", i + 1 - start,
                                prompt->len - start);
                }
            }
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            (void)resumed_checkpoint;
            glm_debug_dump_prefill_logits(s->logits);
            return 0;
        }

        int checkpoint_len_with_logits = s->checkpoint_valid ? s->checkpoint.len : 0;
        int suffix = prompt->len - start;
        const uint32_t resume_min = glm_graph_resume_prefill_min_tokens();
        bool dense_cache_gap = (uint32_t)start > s->glm_dense_cache_len;
        const bool prompt_fits_dense_attention =
            s->glm_graph.full_kv_cache &&
            (uint32_t)prompt->len <= s->glm_graph.ctx_cap;
        const bool sync_trace = getenv("DS4_GLM_SYNC_TRACE") != NULL;
        const bool indexed_batch_available =
            glm_graph_indexed_prefill_batch_available(&s->glm_graph);
        const bool indexed_resume_keeps_sparse_state =
            resumed_checkpoint && suffix > 0 && dense_cache_gap &&
            glm_graph_decode_uses_indexed_attention(&s->glm_graph,
                                                    (uint32_t)start,
                                                    s->logits);
        if (sync_trace) {
            fprintf(stderr,
                    "ds4: GLM sync start=%d prompt=%d suffix=%d checkpoint=%d dense_len=%u ctx_cap=%u dense_fit=%d resume_min=%u dense_gap=%d indexed_keep=%d indexed_batch=%d batch_ffn=%d\n",
                    start,
                    prompt->len,
                    suffix,
                    s->checkpoint_valid ? s->checkpoint.len : 0,
                    s->glm_dense_cache_len,
                    s->glm_graph.ctx_cap,
                    prompt_fits_dense_attention ? 1 : 0,
                    resume_min,
                    dense_cache_gap ? 1 : 0,
                    indexed_resume_keeps_sparse_state ? 1 : 0,
                    indexed_batch_available ? 1 : 0,
                    glm_graph_indexed_prefill_batch_ffn() ? 1 : 0);
        }
        /* If the live checkpoint advanced with indexed attention, the compact
         * cache already represents the live prefix but dense KV may lag behind.
         * Continue from compact/indexed state instead of replaying the old
         * assistant text just to repair dense KV before the new suffix. */
        if (resumed_checkpoint && suffix > 0 && dense_cache_gap &&
            !indexed_resume_keeps_sparse_state &&
            s->glm_graph.full_kv_cache &&
            s->glm_dense_cache_len < s->glm_graph.ctx_cap)
        {
            const uint32_t dense_end =
                (uint32_t)prompt->len < s->glm_graph.ctx_cap ?
                    (uint32_t)prompt->len :
                    s->glm_graph.ctx_cap;
            const uint32_t dense_progress_start = (uint32_t)start;
            const uint32_t dense_progress_total = (uint32_t)suffix;
            const uint32_t dense_chunk_max =
                glm_graph_prefill_chunk_tokens(s->glm_graph.ctx_cap);
            for (uint32_t dense_pos = s->glm_dense_cache_len;
                 dense_pos < dense_end; )
            {
                if (ds4_session_cancelled(s)) {
                    snprintf(err, errlen, "interrupted");
                    if (!s->checkpoint_valid) {
                        s->checkpoint.len = checkpoint_len_with_logits;
                        s->checkpoint_valid = checkpoint_len_with_logits > 0;
                    }
                    s->mtp_draft_valid = false;
                    ds4_session_glm_cap_dense_cache(s);
                    return DS4_SESSION_SYNC_INTERRUPTED;
                }

                uint32_t chunk = dense_end - dense_pos;
                if (chunk > dense_chunk_max) chunk = dense_chunk_max;
                float *chunk_logits =
                    (dense_pos + chunk >= (uint32_t)prompt->len) ?
                        s->logits :
                        NULL;
                if (sync_trace) {
                    fprintf(stderr,
                            "ds4: GLM sync branch=dense_bridge pos=%u chunk=%u logits=%d\n",
                            dense_pos,
                            chunk,
                            chunk_logits ? 1 : 0);
                }
                if (!glm_graph_forward_tokens(&s->glm_graph,
                                              &e->model,
                                              &e->weights,
                                              prompt->v + dense_pos,
                                              NULL,
                                              dense_pos,
                                              chunk,
                                              NULL,
                                              chunk_logits,
                                              s->display_progress,
                                              s->display_progress_ud,
                                              dense_progress_start,
                                              dense_pos > dense_progress_start ?
                                                  dense_pos - dense_progress_start :
                                                  0,
                                              dense_progress_total))
                {
                    snprintf(err, errlen,
                             "%s GLM dense prefill failed while rebuilding checkpoint cache",
                             backend_name);
                    if (!s->checkpoint_valid) {
                        s->checkpoint.len = checkpoint_len_with_logits;
                        s->checkpoint_valid = checkpoint_len_with_logits > 0;
                    }
                    s->mtp_draft_valid = false;
                    ds4_session_glm_cap_dense_cache(s);
                    return 1;
                }
                const uint32_t dense_next = dense_pos + chunk;
                ds4_session_glm_note_dense_cache(s, dense_pos, chunk);
                if (dense_next > (uint32_t)start) {
                    const uint32_t push_from =
                        dense_pos > (uint32_t)start ? dense_pos : (uint32_t)start;
                    if ((uint32_t)s->checkpoint.len != push_from) {
                        snprintf(err, errlen,
                                 "%s GLM checkpoint/cache bridge is inconsistent",
                                 backend_name);
                        s->checkpoint_valid = false;
                        s->mtp_draft_valid = false;
                        ds4_session_glm_cap_dense_cache(s);
                        return 1;
                    }
                    for (uint32_t j = push_from; j < dense_next; j++) {
                        token_vec_push(&s->checkpoint, prompt->v[j]);
                    }
                    start = s->checkpoint.len;
                    if (chunk_logits) {
                        checkpoint_len_with_logits = s->checkpoint.len;
                        s->checkpoint_valid = true;
                    } else {
                        s->checkpoint_valid = false;
                    }
                    s->mtp_draft_valid = false;
                }
                dense_pos = dense_next;
                if (s->progress) {
                    s->progress(s->progress_ud, "prefill_chunk",
                                (int)dense_pos, prompt->len);
                }
            }
            suffix = prompt->len - start;
            dense_cache_gap = (uint32_t)start > s->glm_dense_cache_len;
        }
        if (resumed_checkpoint && suffix > 0 && dense_cache_gap &&
            (uint32_t)suffix >= resume_min &&
            indexed_batch_available)
        {
            for (int i = start; i < prompt->len; ) {
                if (ds4_session_cancelled(s)) {
                    snprintf(err, errlen, "interrupted");
                    s->checkpoint.len = checkpoint_len_with_logits;
                    s->checkpoint_valid = checkpoint_len_with_logits > 0;
                    s->mtp_draft_valid = false;
                    ds4_session_glm_cap_dense_cache(s);
                    return DS4_SESSION_SYNC_INTERRUPTED;
                }

                const uint32_t pos = (uint32_t)s->checkpoint.len;
                uint32_t chunk = (uint32_t)(prompt->len - i);
                if (chunk > s->glm_graph.indexed_prefill_cap) {
                    chunk = s->glm_graph.indexed_prefill_cap;
                }
                chunk = glm_graph_limit_indexed_prefill_chunk(pos, chunk);
                if (chunk == 0) chunk = 1;
                float *chunk_logits =
                    (i + (int)chunk >= prompt->len) ? s->logits : NULL;
                if (sync_trace) {
                    fprintf(stderr,
                            "ds4: GLM sync branch=indexed_resume pos=%u chunk=%u logits=%d\n",
                            pos,
                            chunk,
                            chunk_logits ? 1 : 0);
                }
                if (!glm_graph_forward_indexed_tokens(&s->glm_graph,
                                                      &e->model,
                                                      &e->weights,
                                                      prompt->v + i,
                                                      NULL,
                                                      pos,
                                                      chunk,
                                                      NULL,
                                                      chunk_logits,
                                                      s->display_progress,
                                                      s->display_progress_ud,
                                                      (uint32_t)start,
                                                      (uint32_t)(i - start),
                                                      (uint32_t)suffix))
                {
                    snprintf(err, errlen, "%s GLM indexed prefill failed while extending checkpoint",
                             backend_name);
                    s->checkpoint_valid = false;
                    s->mtp_draft_valid = false;
                    ds4_session_glm_cap_dense_cache(s);
                    return 1;
                }
                for (uint32_t j = 0; j < chunk; j++) {
                    token_vec_push(&s->checkpoint, prompt->v[i + (int)j]);
                }
                i += (int)chunk;
                if (chunk_logits) {
                    checkpoint_len_with_logits = s->checkpoint.len;
                    s->checkpoint_valid = true;
                } else {
                    s->checkpoint_valid = false;
                }
                s->mtp_draft_valid = false;
                if (s->progress) {
                    s->progress(s->progress_ud, "prefill_chunk", i, prompt->len);
                }
            }
            if (!glm_graph_maybe_warm_compact_indexer_after_prefill(
                        &s->glm_graph,
                        &e->model,
                        &e->weights,
                        (uint32_t)s->checkpoint.len)) {
                snprintf(err, errlen, "%s GLM compact indexer warmup failed", backend_name);
                s->checkpoint_valid = false;
                s->mtp_draft_valid = false;
                ds4_session_glm_cap_dense_cache(s);
                return 1;
            }
            glm_debug_dump_prefill_logits(s->logits);
            return 0;
        }
        if (resumed_checkpoint && suffix > 0 &&
            (dense_cache_gap || (uint32_t)suffix < resume_min))
        {
            for (int i = start; i < prompt->len; i++) {
                if (ds4_session_cancelled(s)) {
                    snprintf(err, errlen, "interrupted");
                    s->checkpoint_valid = true;
                    s->mtp_draft_valid = false;
                    ds4_session_glm_cap_dense_cache(s);
                    return DS4_SESSION_SYNC_INTERRUPTED;
                }
                const uint32_t pos = (uint32_t)s->checkpoint.len;
                const bool updates_dense =
                    glm_graph_decode_updates_dense_cache(&s->glm_graph, pos, s->logits);
                if (sync_trace) {
                    fprintf(stderr,
                            "ds4: GLM sync branch=decode_resume pos=%u token_index=%d updates_dense=%d\n",
                            pos,
                            i,
                            updates_dense ? 1 : 0);
                }
                if (!glm_graph_forward_token(&s->glm_graph,
                                             &e->model,
                                             &e->weights,
                                             prompt->v[i],
                                             NULL,
                                             pos,
                                             NULL,
                                             s->logits,
                                             false))
                {
                    snprintf(err, errlen, "%s GLM decode failed while extending checkpoint",
                             backend_name);
                    s->checkpoint_valid = false;
                    s->mtp_draft_valid = false;
                    ds4_session_glm_cap_dense_cache(s);
                    return 1;
                }
                token_vec_push(&s->checkpoint, prompt->v[i]);
                s->checkpoint_valid = true;
                s->mtp_draft_valid = false;
                if (updates_dense) ds4_session_glm_note_dense_cache(s, pos, 1);
                if (s->progress) {
                    s->progress(s->progress_ud, "prefill_chunk", i + 1, prompt->len);
                }
            }
            if (!glm_graph_maybe_warm_compact_indexer_after_prefill(
                        &s->glm_graph,
                        &e->model,
                        &e->weights,
                        (uint32_t)s->checkpoint.len)) {
                snprintf(err, errlen, "%s GLM compact indexer warmup failed", backend_name);
                s->checkpoint_valid = false;
                s->mtp_draft_valid = false;
                ds4_session_glm_cap_dense_cache(s);
                return 1;
            }
            glm_debug_dump_prefill_logits(s->logits);
            return 0;
        }
        const uint32_t chunk_max = glm_graph_prefill_chunk_tokens(s->glm_graph.ctx_cap);
        const uint32_t full_work_start = (uint32_t)start;
        const uint32_t full_work_total =
            (uint32_t)prompt->len > full_work_start ?
                (uint32_t)prompt->len - full_work_start :
                0;
        for (int i = start; i < prompt->len; ) {
            if (ds4_session_cancelled(s)) {
                snprintf(err, errlen, "interrupted");
                if (!s->checkpoint_valid) {
                    s->checkpoint.len = checkpoint_len_with_logits;
                    s->checkpoint_valid = checkpoint_len_with_logits > 0;
                }
                s->mtp_draft_valid = false;
                ds4_session_glm_cap_dense_cache(s);
                return DS4_SESSION_SYNC_INTERRUPTED;
            }
            if (!s->glm_graph.full_kv_cache || (uint32_t)i >= s->glm_graph.ctx_cap) {
                const uint32_t remaining = (uint32_t)(prompt->len - i);
                const bool use_token_major_prefill =
                    glm_graph_use_streaming_token_prefill(&s->glm_graph,
                                                          (uint32_t)i,
                                                          remaining);
                const bool use_indexed_batch =
                    !use_token_major_prefill &&
                    glm_graph_indexed_prefill_batch_ready(&s->glm_graph,
                                                          (uint32_t)i);
                uint32_t chunk = 1;
                bool prefill_ok = false;
                if (use_token_major_prefill) {
                    chunk = remaining;
                    const uint32_t token_prefill_max =
                        glm_graph_streaming_token_prefill_max_tokens();
                    if (token_prefill_max != 0 && chunk > token_prefill_max) {
                        chunk = token_prefill_max;
                    }
                    float *chunk_logits =
                        (i + (int)chunk >= prompt->len) ? s->logits : NULL;
                    if (sync_trace) {
                        fprintf(stderr,
                                "ds4: GLM sync branch=full_compact_token_major pos=%d chunk=%u logits=%d\n",
                                i,
                                chunk,
                                chunk_logits ? 1 : 0);
                    }
                    prefill_ok = glm_graph_prefill_token_major(&s->glm_graph,
                                                               &e->model,
                                                               &e->weights,
                                                               prompt->v + i,
                                                               (uint32_t)i,
                                                               chunk,
                                                               chunk_logits,
                                                               s->display_progress,
                                                               s->display_progress_ud,
                                                               full_work_start,
                                                               (uint32_t)i - full_work_start,
                                                               full_work_total);
                } else if (use_indexed_batch) {
                    chunk = (uint32_t)(prompt->len - i);
                    if (chunk > s->glm_graph.indexed_prefill_cap) {
                        chunk = s->glm_graph.indexed_prefill_cap;
                    }
                    chunk = glm_graph_limit_indexed_prefill_chunk((uint32_t)i, chunk);
                    float *chunk_logits =
                        (i + (int)chunk >= prompt->len) ? s->logits : NULL;
                    if (sync_trace) {
                        fprintf(stderr,
                                "ds4: GLM sync branch=full_indexed pos=%d chunk=%u logits=%d\n",
                                i,
                                chunk,
                                chunk_logits ? 1 : 0);
                    }
                    prefill_ok = glm_graph_forward_indexed_tokens(&s->glm_graph,
                                                                  &e->model,
                                                                  &e->weights,
                                                                  prompt->v + i,
                                                                  NULL,
                                                                  (uint32_t)i,
                                                                  chunk,
                                                                  NULL,
                                                                  chunk_logits,
                                                                  s->display_progress,
                                                                  s->display_progress_ud,
                                                                  full_work_start,
                                                                  (uint32_t)i - full_work_start,
                                                                  full_work_total);
                } else {
                    float *chunk_logits =
                        (i + 1 >= prompt->len) ? s->logits : NULL;
                    if (sync_trace) {
                        fprintf(stderr,
                                "ds4: GLM sync branch=full_decode pos=%d logits=%d\n",
                                i,
                                chunk_logits ? 1 : 0);
                    }
                    prefill_ok = glm_graph_forward_token(&s->glm_graph,
                                                        &e->model,
                                                        &e->weights,
                                                        prompt->v[i],
                                                        NULL,
                                                        (uint32_t)i,
                                                        NULL,
                                                        chunk_logits,
                                                        false);
                }
                if (!prefill_ok) {
                    snprintf(err, errlen, "%s GLM prefill failed", backend_name);
                    if (!s->checkpoint_valid) {
                        s->checkpoint.len = checkpoint_len_with_logits;
                        s->checkpoint_valid = checkpoint_len_with_logits > 0;
                    }
                    s->mtp_draft_valid = false;
                    ds4_session_glm_cap_dense_cache(s);
                    return 1;
                }
                for (uint32_t j = 0; j < chunk; j++) {
                    token_vec_push(&s->checkpoint, prompt->v[i + (int)j]);
                }
                i += (int)chunk;
                if (i >= prompt->len) {
                    checkpoint_len_with_logits = s->checkpoint.len;
                    s->checkpoint_valid = true;
                } else {
                    s->checkpoint_valid = false;
                }
                s->mtp_draft_valid = false;
                if (s->progress) s->progress(s->progress_ud, "prefill_chunk", i, prompt->len);
                continue;
            }
            uint32_t chunk = (uint32_t)(prompt->len - i);
            const uint32_t full_remaining = s->glm_graph.ctx_cap - (uint32_t)i;
            if (chunk > full_remaining) chunk = full_remaining;
            if (chunk > chunk_max) chunk = chunk_max;
            float *chunk_logits =
                (i + (int)chunk >= prompt->len) ? s->logits : NULL;
            const bool use_token_major_prefill =
                glm_graph_use_streaming_token_prefill(&s->glm_graph,
                                                      (uint32_t)i,
                                                      chunk);
            if (sync_trace) {
                fprintf(stderr,
                        "ds4: GLM sync branch=%s pos=%d chunk=%u logits=%d\n",
                        use_token_major_prefill ? "full_token_major" : "full_dense",
                        i,
                        chunk,
                        chunk_logits ? 1 : 0);
            }
            bool prefill_ok = false;
            if (use_token_major_prefill) {
                prefill_ok = glm_graph_prefill_token_major(&s->glm_graph,
                                                           &e->model,
                                                           &e->weights,
                                                           prompt->v + i,
                                                           (uint32_t)i,
                                                           chunk,
                                                           chunk_logits,
                                                           s->display_progress,
                                                           s->display_progress_ud,
                                                           full_work_start,
                                                           (uint32_t)i - full_work_start,
                                                           full_work_total);
            } else {
                prefill_ok = glm_graph_forward_tokens(&s->glm_graph,
                                                      &e->model,
                                                      &e->weights,
                                                      prompt->v + i,
                                                      NULL,
                                                      (uint32_t)i,
                                                      chunk,
                                                      NULL,
                                                      chunk_logits,
                                                      s->display_progress,
                                                      s->display_progress_ud,
                                                      full_work_start,
                                                      (uint32_t)i - full_work_start,
                                                      full_work_total);
            }
            if (!prefill_ok) {
                snprintf(err, errlen, "%s GLM prefill failed", backend_name);
                if (!s->checkpoint_valid) {
                    s->checkpoint.len = checkpoint_len_with_logits;
                    s->checkpoint_valid = checkpoint_len_with_logits > 0;
                }
                s->mtp_draft_valid = false;
                ds4_session_glm_cap_dense_cache(s);
                return 1;
            }
            for (uint32_t j = 0; j < chunk; j++) {
                token_vec_push(&s->checkpoint, prompt->v[i + (int)j]);
            }
            ds4_session_glm_note_dense_cache(s, (uint32_t)i, chunk);
            i += (int)chunk;
            if (i >= prompt->len) {
                checkpoint_len_with_logits = s->checkpoint.len;
                s->checkpoint_valid = true;
            } else {
                s->checkpoint_valid = false;
            }
            s->mtp_draft_valid = false;
            if (s->progress) s->progress(s->progress_ud, "prefill_chunk", i, prompt->len);
        }
        if (!glm_graph_maybe_warm_compact_indexer_after_prefill(
                    &s->glm_graph,
                    &e->model,
                    &e->weights,
                    (uint32_t)s->checkpoint.len)) {
            snprintf(err, errlen, "%s GLM compact indexer warmup failed", backend_name);
            s->checkpoint_valid = false;
            s->mtp_draft_valid = false;
            ds4_session_glm_cap_dense_cache(s);
            return 1;
        }
        glm_debug_dump_prefill_logits(s->logits);
        return 0;
    }

    if (s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        ds4_tokens_starts_with(prompt, &s->checkpoint))
    {
        s->mtp_draft_valid = false;
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
            bool ok = metal_graph_prefill_chunked_range(&s->graph,
                                                        &e->model,
                                                        &e->weights,
                                                        prompt,
                                                        (uint32_t)s->checkpoint.len,
                                                        (uint32_t)suffix,
                                                        s->logits,
                                                        false,
                                                        ds4_session_note_prefill_progress,
                                                        &progress,
                                                        s->display_progress,
                                                        s->display_progress_ud,
                                                        NULL,
                                                        ds4_session_cancelled_cb,
                                                        s,
                                                        &cancelled);
            if (cancelled) {
                snprintf(err, errlen, "interrupted");
                s->checkpoint_valid = true;
                s->mtp_draft_valid = false;
                return DS4_SESSION_SYNC_INTERRUPTED;
            }
            if (!ok) {
                snprintf(err, errlen, "%s resumed prefill failed while extending checkpoint", backend_name);
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
                s->checkpoint_valid = true;
                s->mtp_draft_valid = false;
                return DS4_SESSION_SYNC_INTERRUPTED;
            }
            if (!metal_graph_eval_token_raw_swa(&s->graph, &e->model, &e->weights,
                                                (uint32_t)prompt->v[i],
                                                (uint32_t)s->checkpoint.len,
                                                s->logits))
            {
                snprintf(err, errlen, "%s decode failed while extending checkpoint", backend_name);
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

    bool ok;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
    s->mtp_draft_valid = false;
    ds4_session_dspark_capture_invalidate(s);
    if (!metal_graph_reset_prefill_state(&s->graph)) {
        snprintf(err, errlen, "%s prefill state reset failed", backend_name);
        return 1;
    }
    if (s->prefill_cap < (uint32_t)prompt->len) {
        bool cancelled = false;
        ds4_sync_progress progress = {
            .session = s,
            .prompt = prompt,
            .user = s->progress,
            .user_ud = s->progress_ud,
        };
        ok = metal_graph_prefill_chunked(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         ds4_session_note_prefill_progress, &progress,
                                         s->display_progress,
                                         s->display_progress_ud,
                                         ds4_session_cancelled_cb,
                                         s,
                                         &cancelled);
        if (cancelled) {
            snprintf(err, errlen, "interrupted");
            s->checkpoint_valid = s->checkpoint.len > 0;
            s->mtp_draft_valid = false;
            return DS4_SESSION_SYNC_INTERRUPTED;
        }
    } else {
        bool cancelled = false;
        ok = metal_graph_prefill_raw_swa(&s->graph, &e->model, &e->weights,
                                         prompt, prompt->len, s->logits, false,
                                         s->display_progress,
                                         s->display_progress_ud,
                                         ds4_session_cancelled_cb,
                                         s,
                                         &cancelled);
        if (cancelled) {
            snprintf(err, errlen, "interrupted");
            return DS4_SESSION_SYNC_INTERRUPTED;
        }
    }
    if (!ok) {
        snprintf(err, errlen, "%s prefill failed", backend_name);
        s->checkpoint_valid = false;
        return 1;
    }
    ds4_tokens_copy(&s->checkpoint, prompt);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    ds4_session_dspark_capture_note_checkpoint(s);
    s->graph.mtp_n_raw = 0;
    session_greedy_splitkv_reset(s);
    return 0;
    }
#endif
}

