/* =========================================================================
 * Coordinator Session API
 * =========================================================================
 *
 * These functions are the distributed backend for the normal ds4_session API.
 * Program frontends should keep using ds4_session_sync/eval/save/load; ds4.c
 * selects these calls when the owning session has a coordinator attached.
 */

int ds4_dist_session_create(
        ds4_dist_session **out,
        ds4_engine *engine,
        const ds4_dist_options *opt,
        ds4_session *owner,
        int ctx_size,
        char *err,
        size_t errlen) {
    (void)owner;
    if (!out || !engine || !opt) {
        if (errlen) snprintf(err, errlen, "missing distributed session parameters");
        return 1;
    }
    *out = NULL;
    if (opt->role != DS4_DISTRIBUTED_COORDINATOR) {
        if (errlen) snprintf(err, errlen, "distributed session requires coordinator role");
        return 1;
    }
    if (dist_validate_options(opt, err, errlen) != 0) return 1;

    int listen_fd = dist_open_listener(opt->listen_host, opt->listen_port, err, errlen);
    if (listen_fd < 0) return 1;

    ds4_dist_session *d = calloc(1, sizeof(*d));
    if (!d) {
        close(listen_fd);
        if (errlen) snprintf(err, errlen, "out of memory creating distributed session");
        return 1;
    }

    d->listen_fd = listen_fd;
    d->state.engine = engine;
    d->state.model_id = (uint32_t)ds4_engine_model_id(engine);
    d->state.n_layers = (uint32_t)ds4_engine_layer_count(engine);
    d->state.local_start = opt->layers.start;
    d->state.local_end = dist_resolved_layer_end(opt, d->state.n_layers);
    d->state.ctx_size = ctx_size > 0 ? (uint32_t)ctx_size : 0u;
    d->state.local_has_output = opt->layers.has_output;
    d->state.local_can_output_head = ds4_engine_has_output_head(engine);
    d->state.replay_check = opt->replay_check;
    d->state.debug = opt->debug;
    d->state.use_control_for_work = true;
    d->state.prefill_chunk = opt->prefill_chunk;
    d->state.prefill_window = opt->prefill_window;
    d->state.activation_bits = dist_activation_bits_or_default(opt->activation_bits);
    pthread_mutex_init(&d->state.mu, NULL);
    d->session_id = dist_make_session_id(d);
    d->request_id = 1;
    /* KV snapshots use separate data connections and can run while pipelined
     * WORK results are outstanding.  Keep them out of the WORK request-id stream
     * so progress callbacks cannot perturb the reader's contiguous expectations. */
    d->snapshot_request_id = UINT64_C(1) << 63;

    char local_end[32];
    if (opt->layers.has_output) snprintf(local_end, sizeof(local_end), "output");
    else snprintf(local_end, sizeof(local_end), "%u", opt->layers.end);
    DIST_COORD_DEBUG(&d->state,
                     "ds4: distributed coordinator API: listening on %s:%d model_id=%u layers=%u local=%u:%s activation_bits=%u\n",
                     opt->listen_host,
                     opt->listen_port,
                     d->state.model_id,
                     d->state.n_layers,
                     opt->layers.start,
                     local_end,
                     d->state.activation_bits);

    d->accept_ctx.state = &d->state;
    d->accept_ctx.listen_fd = listen_fd;
    if (pthread_create(&d->accept_tid, NULL, dist_coordinator_accept_main, &d->accept_ctx) != 0) {
        close(listen_fd);
        pthread_mutex_destroy(&d->state.mu);
        free(d);
        if (errlen) snprintf(err, errlen, "failed to start distributed coordinator accept loop");
        return 1;
    }
    pthread_detach(d->accept_tid);
    d->accept_started = true;
    *out = d;
    return 0;
}

void ds4_dist_session_free(ds4_dist_session *d) {
    if (!d) return;
    if (d->listen_fd >= 0) {
        shutdown(d->listen_fd, SHUT_RDWR);
        close(d->listen_fd);
        d->listen_fd = -1;
    }
    dist_route_plan_free(&d->plan);
    pthread_mutex_lock(&d->state.mu);
    d->state.shutting_down = true;
    for (ds4_dist_worker_entry *it = d->state.workers; it; it = it->next) {
        if (it->fd >= 0) shutdown(it->fd, SHUT_RDWR);
    }
    pthread_mutex_unlock(&d->state.mu);
    /* Client threads are detached and remove their registry entries after the
     * socket closes. Keep this small coordinator object process-lifetime to
     * avoid racing those threads during application shutdown. */
}

int ds4_dist_session_route_ready(ds4_dist_session *d, char *err, size_t errlen) {
    if (!d) {
        if (errlen) snprintf(err, errlen, "missing distributed session");
        return -1;
    }

    ds4_dist_route_plan probe = {0};
    if (!dist_coordinator_build_route_plan(&d->state, &probe, NULL, err, errlen)) {
        return 0;
    }
    dist_route_plan_free(&probe);
    if (errlen) err[0] = '\0';
    return 1;
}

int ds4_dist_session_sync(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        const ds4_tokens *prompt,
        float *logits,
        char *err,
        size_t errlen) {
    if (!d || !owner || !prompt || prompt->len <= 0 || !logits) {
        if (errlen) snprintf(err, errlen, "invalid distributed sync request");
        return 1;
    }
    if (dist_session_ensure_route(d, err, errlen) != 0) return 1;

    if (checkpoint &&
        checkpoint->len >= 0 &&
        checkpoint->len <= prompt->len &&
        ds4_tokens_starts_with(prompt, checkpoint))
    {
        if (checkpoint->len == prompt->len) return 0;

        uint32_t chunk_cap = 0;
        if (dist_coordinator_prefill_chunk_cap(&d->state, owner, &chunk_cap, err, errlen) != 0) {
            return 1;
        }
        const uint32_t pos0 = (uint32_t)checkpoint->len;
        const uint32_t suffix = (uint32_t)prompt->len - pos0;
        if (dist_coordinator_can_pipeline_prefill(&d->state, &d->plan, owner, suffix, chunk_cap)) {
            int prefill_rc = dist_coordinator_prefill_prompt_pipelined(&d->state,
                                                                       owner,
                                                                       &d->plan,
                                                                       prompt,
                                                                       pos0,
                                                                       suffix,
                                                                       false,
                                                                       chunk_cap,
                                                                       d->session_id,
                                                                       &d->request_id,
                                                                       logits,
                                                                       err,
                                                                       errlen);
            if (prefill_rc != 0) {
                if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                             owner,
                                                             &d->plan,
                                                             prompt,
                                                             d->session_id,
                                                             &d->request_id,
                                                             logits,
                                                             &d->plan_generation,
                                                             prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                             err,
                                                             errlen) != 0) {
                    d->plan_ready = false;
                    d->plan_generation = 0;
                    return 1;
                }
                d->plan_ready = true;
            }
            return 0;
        }

        uint32_t pos = pos0;
        while (pos < (uint32_t)prompt->len) {
            const uint32_t remaining = (uint32_t)prompt->len - pos;
            const uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
            int eval_rc = dist_coordinator_eval_span(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     prompt->v + pos,
                                                     chunk,
                                                     pos,
                                                     d->session_id,
                                                     d->request_id++,
                                                     false,
                                                     logits,
                                                     err,
                                                     errlen);
            if (eval_rc != 0) {
                if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                             owner,
                                                             &d->plan,
                                                             prompt,
                                                             d->session_id,
                                                             &d->request_id,
                                                             logits,
                                                             &d->plan_generation,
                                                             eval_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                             err,
                                                             errlen) != 0) {
                    d->plan_ready = false;
                    d->plan_generation = 0;
                    return 1;
                }
                d->plan_ready = true;
                return 0;
            }
            pos += chunk;
            dist_report_prefill_progress(owner, pos, (uint32_t)prompt->len);
        }
        return 0;
    }

    int prefill_rc = dist_coordinator_prefill_prompt(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     prompt,
                                                     d->session_id,
                                                     &d->request_id,
                                                     logits,
                                                     err,
                                                     errlen);
    if (prefill_rc != 0) {
        if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     prompt,
                                                     d->session_id,
                                                     &d->request_id,
                                                     logits,
                                                     &d->plan_generation,
                                                     prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                     err,
                                                     errlen) != 0) {
            d->plan_ready = false;
            d->plan_generation = 0;
            return 1;
        }
        d->plan_ready = true;
    }
    return 0;
}

int ds4_dist_session_eval(
        ds4_dist_session *d,
        ds4_session *owner,
        const ds4_tokens *checkpoint,
        int token,
        float *logits,
        char *err,
        size_t errlen) {
    if (!d || !owner || !checkpoint || checkpoint->len < 0 || !logits) {
        if (errlen) snprintf(err, errlen, "invalid distributed decode request");
        return 1;
    }
    if (dist_session_ensure_route(d, err, errlen) != 0) return 1;

    int rc = dist_coordinator_eval_span(&d->state,
                                        owner,
                                        &d->plan,
                                        &token,
                                        1,
                                        (uint32_t)checkpoint->len,
                                        d->session_id,
                                        d->request_id++,
                                        false,
                                        logits,
                                        err,
                                        errlen);
    if (rc != 0) {
        ds4_tokens transcript = {0};
        ds4_tokens_copy(&transcript, checkpoint);
        ds4_tokens_push(&transcript, token);
        if (dist_coordinator_rebuild_from_transcript(&d->state,
                                                     owner,
                                                     &d->plan,
                                                     &transcript,
                                                     d->session_id,
                                                     &d->request_id,
                                                     logits,
                                                     &d->plan_generation,
                                                     rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                     err,
                                                     errlen) != 0) {
            d->plan_ready = false;
            d->plan_generation = 0;
            ds4_tokens_free(&transcript);
            return 1;
        }
        d->plan_ready = true;
        ds4_tokens_free(&transcript);
        rc = 0;
    }
    return rc;
}

/* =========================================================================
 * Standalone Coordinator Entrypoint
 * ========================================================================= */

static int dist_run_coordinator(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen) {
    char err[256];
    int listen_fd = dist_open_listener(opt->listen_host, opt->listen_port, err, sizeof(err));
    if (listen_fd < 0) {
        fprintf(stderr, "ds4: distributed coordinator: %s\n", err);
        return 1;
    }

    ds4_dist_coordinator_state state;
    memset(&state, 0, sizeof(state));
    state.engine = engine;
    state.model_id = (uint32_t)ds4_engine_model_id(engine);
    state.n_layers = (uint32_t)ds4_engine_layer_count(engine);
    state.local_start = opt->layers.start;
    state.local_end = dist_resolved_layer_end(opt, state.n_layers);
    state.ctx_size = gen && gen->ctx_size > 0 ? (uint32_t)gen->ctx_size : 0u;
    state.local_has_output = opt->layers.has_output;
    state.local_can_output_head = ds4_engine_has_output_head(engine);
    state.replay_check = opt->replay_check;
    state.debug = opt->debug;
    state.use_control_for_work = gen && gen->prompt;
    state.prefill_chunk = opt->prefill_chunk;
    state.prefill_window = opt->prefill_window;
    state.activation_bits = dist_activation_bits_or_default(opt->activation_bits);
    pthread_mutex_init(&state.mu, NULL);

    char local_end[32];
    if (opt->layers.has_output) snprintf(local_end, sizeof(local_end), "output");
    else snprintf(local_end, sizeof(local_end), "%u", opt->layers.end);
    DIST_COORD_DEBUG(&state,
                     "ds4: distributed coordinator: listening on %s:%d model_id=%u layers=%u local=%u:%s activation_bits=%u\n",
                     opt->listen_host,
                     opt->listen_port,
                     state.model_id,
                     state.n_layers,
                     opt->layers.start,
                     local_end,
                     state.activation_bits);

    ds4_dist_accept_ctx accept_ctx = {
        .state = &state,
        .listen_fd = listen_fd,
    };
    if (!gen || !gen->prompt) {
        dist_coordinator_accept_main(&accept_ctx);
        return 0;
    }

    pthread_t accept_tid;
    if (pthread_create(&accept_tid, NULL, dist_coordinator_accept_main, &accept_ctx) != 0) {
        fprintf(stderr, "ds4: distributed coordinator: pthread_create failed for accept loop\n");
        close(listen_fd);
        return 1;
    }
    pthread_detach(accept_tid);

    return dist_run_coordinator_generation(&state, gen);
}

/* =========================================================================
 * Worker Control Loop And Result Frames
 * ========================================================================= */

static int dist_worker_read_loop(ds4_dist_worker_state *state, int fd) {
    ds4_dist_worker_upstream upstream;
    dist_worker_upstream_init(&upstream, state, fd);
    int loop_rc = 0;

    for (;;) {
        uint32_t type = 0, bytes = 0;
        char err[256];
        int rc = dist_read_frame_header(fd, &type, &bytes, err, sizeof(err));
        if (rc == 0) break;
        if (rc < 0) {
            fprintf(stderr, "ds4: distributed worker: protocol error: %s\n", err);
            loop_rc = 1;
            break;
        }
        if (type == DS4_DIST_MSG_ERROR) {
            char msg[512];
            uint32_t n = bytes < sizeof(msg) - 1u ? bytes : (uint32_t)sizeof(msg) - 1u;
            rc = dist_read_full(fd, msg, n);
            if (rc <= 0) {
                loop_rc = 1;
                break;
            }
            msg[n] = '\0';
            if (bytes > n) dist_discard_bytes(fd, bytes - n);
            fprintf(stderr, "ds4: distributed worker: coordinator error: %s\n", msg);
            loop_rc = 1;
            break;
        }
        if (type == DS4_DIST_MSG_WORK) {
            rc = dist_worker_handle_work(state, &upstream, bytes);
            if (rc <= 0) {
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            continue;
        }
        if (type == DS4_DIST_MSG_SNAPSHOT_SAVE_REQ) {
            rc = dist_worker_handle_snapshot_save(state, &upstream, bytes);
            if (rc <= 0) {
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            continue;
        }
        if (type == DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN) {
            rc = dist_worker_handle_snapshot_load(state, &upstream, bytes);
            if (rc <= 0) {
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            continue;
        }
        rc = dist_discard_bytes(fd, bytes);
        if (rc <= 0) {
            loop_rc = rc == 0 ? 0 : 1;
            break;
        }
        pthread_mutex_lock(&upstream.write_mu);
        dist_send_error(fd, "unsupported distributed worker frame");
        pthread_mutex_unlock(&upstream.write_mu);
        fprintf(stderr, "ds4: distributed worker: rejected unsupported frame type %u\n", type);
        loop_rc = 1;
        break;
    }

    dist_worker_upstream_destroy(&upstream);
    return loop_rc;
}

static int dist_send_work_result(
        int fd,
        uint64_t request_id,
        uint64_t result_hash,
        uint32_t status,
        uint32_t result_kind,
        uint32_t payload_bits,
        const ds4_dist_telemetry_fixed *telemetry,
        uint32_t telemetry_count,
        const void *payload,
        uint32_t payload_bytes) {
    if (payload_bytes != 0 && !payload) return -1;
    if (telemetry_count != 0 && !telemetry) return -1;
    uint32_t wire_payload_bytes = payload_bytes;
    uint64_t hidden_values = 0;
    if (status == 0 && result_kind == DS4_DIST_RESULT_HIDDEN_STATE) {
        payload_bits = dist_activation_bits_or_default(payload_bits);
        if (!dist_activation_bits_valid(payload_bits) ||
            (payload_bytes % (uint32_t)sizeof(float)) != 0)
            return -1;
        hidden_values = payload_bytes / (uint32_t)sizeof(float);
        if (!dist_activation_wire_bytes(payload_bits, hidden_values, &wire_payload_bytes))
            return -1;
    } else if (status == 0 && result_kind == DS4_DIST_RESULT_LOGITS) {
        payload_bits = 32u;
    } else {
        payload_bits = 0;
    }
    const uint64_t telemetry_bytes64 =
        (uint64_t)telemetry_count * sizeof(ds4_dist_telemetry_fixed);
    if (telemetry_bytes64 > UINT32_MAX) return -1;
    const uint32_t telemetry_bytes = (uint32_t)telemetry_bytes64;
    const uint64_t frame_bytes = sizeof(ds4_dist_result_fixed) +
                                 telemetry_bytes64 +
                                 (uint64_t)wire_payload_bytes;
    if (frame_bytes > UINT32_MAX) return -1;

    ds4_dist_result_fixed r;
    dist_u64_to_halves(request_id, &r.request_hi, &r.request_lo);
    dist_u64_to_halves(status == 0 ? result_hash : 0,
                       &r.result_hash_hi,
                       &r.result_hash_lo);
    r.status = status;
    r.result_kind = result_kind;
    r.telemetry_count = telemetry_count;
    r.telemetry_bytes = telemetry_bytes;
    r.payload_bytes = wire_payload_bytes;
    r.payload_bits = payload_bits;

    ds4_dist_result_fixed wire = r;
    dist_result_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_RESULT, (uint32_t)frame_bytes) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (uint32_t i = 0; i < telemetry_count; i++) {
        ds4_dist_telemetry_fixed tw = telemetry[i];
        dist_telemetry_to_wire(&tw);
        if (dist_write_full(fd, &tw, sizeof(tw)) != 0) return -1;
    }
    if (status == 0 && result_kind == DS4_DIST_RESULT_HIDDEN_STATE && wire_payload_bytes != 0) {
        if (dist_write_activation_payload(fd, payload, hidden_values, payload_bits) != 0) return -1;
    } else if (payload_bytes && payload && dist_write_full(fd, payload, payload_bytes) != 0) {
        return -1;
    }
    return 1;
}

static int dist_send_work_error(int fd, uint64_t request_id, const char *msg) {
    if (!msg) msg = "distributed work failed";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    return dist_send_work_result(fd, request_id, 0, 1, 0, 0, NULL, 0, msg, (uint32_t)len);
}

static int dist_send_snapshot_begin(
        int fd,
        const ds4_dist_snapshot_begin_fixed *begin,
        const int *tokens,
        const char *msg) {
    uint64_t token_bytes64 = (uint64_t)begin->token_count * sizeof(uint32_t);
    if (token_bytes64 > UINT32_MAX || begin->token_bytes != (uint32_t)token_bytes64) return -1;
    uint32_t msg_len = msg ? (uint32_t)strlen(msg) : 0u;
    if (msg_len != begin->message_bytes) return -1;
    uint64_t frame_bytes64 = sizeof(*begin) + token_bytes64 + msg_len;
    if (frame_bytes64 > UINT32_MAX) return -1;
    ds4_dist_snapshot_begin_fixed wire = *begin;
    dist_snapshot_begin_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_BEGIN, (uint32_t)frame_bytes64) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (uint32_t i = 0; i < begin->token_count; i++) {
        uint32_t t = htonl((uint32_t)tokens[i]);
        if (dist_write_full(fd, &t, sizeof(t)) != 0) return -1;
    }
    if (msg_len && dist_write_full(fd, msg, msg_len) != 0) return -1;
    return 1;
}

static int dist_send_snapshot_error(
        int fd,
        uint64_t request_id,
        uint64_t session_id,
        uint32_t model_id,
        uint32_t layer_start,
        uint32_t layer_end,
        const char *msg) {
    if (!msg) msg = "distributed snapshot failed";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    ds4_dist_snapshot_begin_fixed begin;
    memset(&begin, 0, sizeof(begin));
    begin.model_id = model_id;
    dist_u64_to_halves(session_id, &begin.session_hi, &begin.session_lo);
    dist_u64_to_halves(request_id, &begin.request_hi, &begin.request_lo);
    begin.layer_start = layer_start;
    begin.layer_end = layer_end;
    begin.status = 1;
    begin.message_bytes = (uint32_t)len;
    return dist_send_snapshot_begin(fd, &begin, NULL, msg);
}

static int dist_send_snapshot_done(int fd, uint64_t request_id, uint32_t status, const char *msg) {
    if (!msg) msg = "";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    ds4_dist_snapshot_done_fixed done;
    memset(&done, 0, sizeof(done));
    dist_u64_to_halves(request_id, &done.request_hi, &done.request_lo);
    done.status = status;
    done.message_bytes = (uint32_t)len;
    uint64_t frame_bytes64 = sizeof(done) + len;
    if (frame_bytes64 > UINT32_MAX) return -1;
    ds4_dist_snapshot_done_fixed wire = done;
    dist_snapshot_done_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_DONE, (uint32_t)frame_bytes64) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    if (len && dist_write_full(fd, msg, len) != 0) return -1;
    return 1;
}

static int dist_send_snapshot_file_chunks(int fd, uint64_t request_id, FILE *fp, uint64_t bytes) {
    uint8_t *buf = malloc(DS4_DIST_SNAPSHOT_CHUNK_BYTES);
    if (!buf) return -1;
    int rc = 1;
    while (bytes != 0) {
        const uint32_t n = bytes > DS4_DIST_SNAPSHOT_CHUNK_BYTES ?
            DS4_DIST_SNAPSHOT_CHUNK_BYTES : (uint32_t)bytes;
        if (fread(buf, 1, n, fp) != n) {
            rc = -1;
            break;
        }
        ds4_dist_snapshot_chunk_fixed chunk;
        dist_u64_to_halves(request_id, &chunk.request_hi, &chunk.request_lo);
        chunk.chunk_bytes = n;
        ds4_dist_snapshot_chunk_fixed wire = chunk;
        dist_snapshot_chunk_to_wire(&wire);
        const uint32_t frame_bytes = (uint32_t)sizeof(wire) + n;
        if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_CHUNK, frame_bytes) != 0 ||
            dist_write_full(fd, &wire, sizeof(wire)) != 0 ||
            dist_write_full(fd, buf, n) != 0) {
            rc = -1;
            break;
        }
        bytes -= n;
    }
    free(buf);
    return rc;
}

