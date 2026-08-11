/* =========================================================================
 * Standalone Coordinator Recovery And Generation
 * ========================================================================= */

static int dist_replay_check_logits(
        ds4_dist_coordinator_state *state,
        const float *before,
        const float *after) {
    const int vocab = ds4_engine_vocab_size(state->engine);
    float max_abs = 0.0f;
    int max_i = 0;
    uint32_t mismatches = 0;
    for (int i = 0; i < vocab; i++) {
        if (before[i] != after[i]) {
            const float d = fabsf(before[i] - after[i]);
            if (d > max_abs) {
                max_abs = d;
                max_i = i;
            }
            mismatches++;
        }
    }
    if (mismatches != 0) {
        fprintf(stderr,
                "ds4: distributed replay check failed: mismatches=%u max_abs=%g token=%d before=%g after=%g\n",
                mismatches,
                max_abs,
                max_i,
                before[max_i],
                after[max_i]);
        return 1;
    }
    fprintf(stderr, "ds4: distributed replay check passed: logits exact match across reset/replay\n");
    return 0;
}

static int dist_run_coordinator_generation(
        ds4_dist_coordinator_state *state,
        const ds4_dist_generation_options *gen) {
    char err[256];
    ds4_dist_route_plan plan;
    uint64_t plan_generation = 0;
    if (!dist_coordinator_ensure_route(state, &plan, &plan_generation, err, sizeof(err))) {
        fprintf(stderr, "ds4: distributed coordinator: %s\n", err);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, state->engine, gen->ctx_size) != 0) {
        fprintf(stderr, "ds4: distributed coordinator: failed to create local session\n");
        dist_route_plan_free(&plan);
        return 1;
    }

    ds4_tokens prompt = {0};
    if (dist_prompt_is_rendered_chat(gen->prompt)) {
        ds4_tokenize_rendered_chat(state->engine, gen->prompt, &prompt);
    } else {
        ds4_encode_chat_prompt(state->engine, gen->system, gen->prompt, gen->think_mode, &prompt);
    }
    if (prompt.len <= 0) {
        fprintf(stderr, "ds4: distributed coordinator: empty prompt\n");
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return 1;
    }

    const uint64_t session_id = ((uint64_t)(uint32_t)time(NULL) << 32) ^ (uint64_t)getpid();
    uint64_t request_id = 1;
    float *logits = malloc((size_t)ds4_engine_vocab_size(state->engine) * sizeof(float));
    if (!logits) {
        fprintf(stderr, "ds4: distributed coordinator: out of memory allocating logits\n");
        ds4_tokens_free(&prompt);
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return 1;
    }

    int prefill_rc = dist_coordinator_prefill_prompt(state,
                                                     session,
                                                     &plan,
                                                     &prompt,
                                                     session_id,
                                                     &request_id,
                                                     logits,
                                                     err,
                                                     sizeof(err));
    if (prefill_rc != 0) {
        fprintf(stderr,
                "ds4: distributed prompt processing failed: %s\n",
                err);
        if (dist_coordinator_rebuild_from_transcript(state,
                                                     session,
                                                     &plan,
                                                     &prompt,
                                                     session_id,
                                                     &request_id,
                                                     logits,
                                                     NULL,
                                                     prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                     err,
                                                     sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4: distributed prompt recovery failed: %s\n",
                    err);
            free(logits);
            ds4_tokens_free(&prompt);
            ds4_session_free(session);
            dist_route_plan_free(&plan);
            return 1;
        }
    }

    if (state->replay_check) {
        const size_t logits_bytes = (size_t)ds4_engine_vocab_size(state->engine) * sizeof(logits[0]);
        float *before = malloc(logits_bytes);
        if (!before) {
            fprintf(stderr, "ds4: distributed replay check: out of memory allocating logits copy\n");
            free(logits);
            ds4_tokens_free(&prompt);
            ds4_session_free(session);
            dist_route_plan_free(&plan);
            return 1;
        }
        memcpy(before, logits, logits_bytes);
        int replay_prefill_rc = dist_coordinator_prefill_prompt(state,
                                                                session,
                                                                &plan,
                                                                &prompt,
                                                                session_id,
                                                                &request_id,
                                                                logits,
                                                                err,
                                                                sizeof(err));
        if (replay_prefill_rc != 0) {
            fprintf(stderr,
                    "ds4: distributed replay prompt processing failed: %s\n",
                    err);
            if (dist_coordinator_rebuild_from_transcript(state,
                                                         session,
                                                         &plan,
                                                         &prompt,
                                                         session_id,
                                                         &request_id,
                                                         logits,
                                                         NULL,
                                                         replay_prefill_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                         err,
                                                         sizeof(err)) != 0) {
                fprintf(stderr,
                        "ds4: distributed replay recovery failed: %s\n",
                        err);
                free(before);
                free(logits);
                ds4_tokens_free(&prompt);
                ds4_session_free(session);
                dist_route_plan_free(&plan);
                return 1;
            }
        }
        int replay_rc = dist_replay_check_logits(state, before, logits);
        free(before);
        if (replay_rc != 0) {
            free(logits);
            ds4_tokens_free(&prompt);
            ds4_session_free(session);
            dist_route_plan_free(&plan);
            return 1;
        }
    }

    if (gen->dump_logits_path) {
        int rc = dist_write_logits_dump(state, gen, &prompt, &plan, logits);
        free(logits);
        ds4_tokens_free(&prompt);
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return rc;
    }
    if (gen->dump_logprobs_path) {
        int rc = dist_write_logprobs_dump(state,
                                          gen,
                                          &prompt,
                                          &plan,
                                          session,
                                          session_id,
                                          &request_id,
                                          logits);
        free(logits);
        ds4_tokens_free(&prompt);
        ds4_session_free(session);
        dist_route_plan_free(&plan);
        return rc;
    }

    int generated = 0;
    int max_tokens = gen->n_predict > 0 ? gen->n_predict : 1;
    int room = gen->ctx_size - prompt.len;
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;
    uint64_t rng = gen->seed ? gen->seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    const int eos = ds4_token_eos(state->engine);
    ds4_tokens transcript = {0};
    ds4_tokens_copy(&transcript, &prompt);
    while (generated < max_tokens) {
        int token = ds4_sample_logits(logits,
                                      ds4_engine_vocab_size(state->engine),
                                      gen->temperature,
                                      0,
                                      gen->top_p,
                                      gen->min_p,
                                      &rng);
        if (token == eos) break;

        size_t len = 0;
        char *text = ds4_token_text(state->engine, token, &len);
        if (len) fwrite(text, 1, len, stdout);
        fflush(stdout);
        free(text);

        uint32_t token_pos = (uint32_t)prompt.len + (uint32_t)generated;
        generated++;
        ds4_tokens_push(&transcript, token);
        if (generated >= max_tokens) break;

        int decode_rc = dist_coordinator_eval_span(state, session, &plan,
                                                   &token, 1, token_pos,
                                                   session_id, request_id++,
                                                   false, logits, err, sizeof(err));
        if (decode_rc != 0) {
            fprintf(stderr, "\nds4: distributed decode failed: %s\n", err);
            if (dist_coordinator_rebuild_from_transcript(state,
                                                         session,
                                                         &plan,
                                                         &transcript,
                                                         session_id,
                                                         &request_id,
                                                         logits,
                                                         NULL,
                                                         decode_rc != DS4_DIST_RECV_REMOTE_ERROR,
                                                         err,
                                                         sizeof(err)) != 0) {
                fprintf(stderr, "ds4: distributed decode recovery failed: %s\n", err);
                ds4_tokens_free(&transcript);
                free(logits);
                ds4_tokens_free(&prompt);
                ds4_session_free(session);
                dist_route_plan_free(&plan);
                return 1;
            }
        }
    }
    fputc('\n', stdout);

    ds4_tokens_free(&transcript);
    free(logits);
    ds4_tokens_free(&prompt);
    ds4_session_free(session);
    dist_route_plan_free(&plan);
    return 0;
}

/* =========================================================================
 * Coordinator Control Plane
 * ========================================================================= */

static void dist_coordinator_remove_worker(ds4_dist_coordinator_state *state, int fd) {
    pthread_mutex_lock(&state->mu);
    ds4_dist_worker_entry **link = &state->workers;
    while (*link) {
        ds4_dist_worker_entry *entry = *link;
        if (entry->fd == fd) {
            *link = entry->next;
            state->generation++;
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: removed worker %s:%s layers=%u:%u%s\n",
                             entry->peer_host,
                             entry->peer_port,
                             entry->layer_start,
                             entry->layer_end,
                             entry->has_output ? "+output" : "");
            pthread_mutex_unlock(&state->mu);
            free(entry);
            if (dist_coordinator_debug_enabled(state)) dist_coordinator_report_plan(state);
            return;
        }
        link = &entry->next;
    }
    pthread_mutex_unlock(&state->mu);
}

static void dist_coordinator_monitor_worker_fd(
        ds4_dist_coordinator_state *state,
        int fd,
        const char *peer_host,
        const char *peer_port) {
    for (;;) {
        struct pollfd pfd = {
            .fd = fd,
            .events = 0,
            .revents = 0,
        };
        int rc = poll(&pfd, 1, 1000);
        if (rc < 0) {
            if (errno == EINTR) continue;
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: worker %s:%s poll failed: %s\n",
                             peer_host,
                             peer_port,
                             strerror(errno));
            break;
        }
        if (rc == 0) continue;
        if ((pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) break;
    }
}

static void *dist_coordinator_client_main(void *arg) {
    ds4_dist_client_ctx *ctx = arg;
    int fd = ctx->fd;
    ds4_dist_coordinator_state *state = ctx->state;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
    snprintf(peer_host, sizeof(peer_host), "%s", ctx->peer_host);
    snprintf(peer_port, sizeof(peer_port), "%s", ctx->peer_port);
    free(ctx);

    ds4_dist_hello_fixed hello;
    char model_name[DS4_DIST_MAX_MODEL_NAME + 1u];
    char err[256];
    int rc = dist_recv_hello(fd, &hello, model_name, sizeof(model_name), err, sizeof(err));
    if (rc <= 0) {
        if (rc < 0) DIST_COORD_DEBUG(state, "ds4: distributed coordinator: bad HELLO from %s:%s: %s\n", peer_host, peer_port, err);
        close(fd);
        return NULL;
    }

    if (hello.model_id != state->model_id) {
        snprintf(err, sizeof(err), "model id mismatch: worker=%u coordinator=%u", hello.model_id, state->model_id);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    const char *expected_model_name = ds4_engine_model_name(state->engine);
    if (!expected_model_name) expected_model_name = "unknown";
    if (strcmp(model_name, expected_model_name) != 0) {
        snprintf(err,
                 sizeof(err),
                 "model family mismatch: worker=%s coordinator=%s",
                 model_name,
                 expected_model_name);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.n_layers != state->n_layers) {
        snprintf(err, sizeof(err), "layer count mismatch: worker=%u coordinator=%u", hello.n_layers, state->n_layers);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.quant_bits != 2u && hello.quant_bits != 4u) {
        snprintf(err, sizeof(err), "unsupported worker quant profile Q%u", hello.quant_bits);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.has_output > 1u) {
        snprintf(err, sizeof(err), "invalid worker output-head flag %u", hello.has_output);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.has_hidden > 1u) {
        snprintf(err, sizeof(err), "invalid worker hidden-state flag %u", hello.has_hidden);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.layer_start >= hello.n_layers || hello.layer_end >= hello.n_layers || hello.layer_end < hello.layer_start) {
        snprintf(err, sizeof(err), "invalid worker layer range %u:%u for %u layers", hello.layer_start, hello.layer_end, hello.n_layers);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.has_output && hello.layer_end + 1u != hello.n_layers) {
        snprintf(err,
                 sizeof(err),
                 "worker output head requires final layer: range=%u:%u layers=%u",
                 hello.layer_start,
                 hello.layer_end,
                 hello.n_layers);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (state->ctx_size != 0 && hello.ctx_size < state->ctx_size) {
        snprintf(err,
                 sizeof(err),
                 "worker context too small: worker=%u coordinator=%u",
                 hello.ctx_size,
                 state->ctx_size);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }
    if (hello.listen_port == 0 || hello.listen_port > 65535u) {
        snprintf(err, sizeof(err), "invalid worker data listen port %u", hello.listen_port);
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: rejecting %s:%s: %s\n", peer_host, peer_port, err);
        dist_send_error(fd, err);
        close(fd);
        return NULL;
    }

    dist_coordinator_add_worker(state, fd, peer_host, peer_port, &hello, model_name);

    if (state->use_control_for_work) {
        dist_coordinator_monitor_worker_fd(state, fd, peer_host, peer_port);
    } else {
        for (;;) {
            uint32_t type = 0, bytes = 0;
            rc = dist_read_frame_header(fd, &type, &bytes, err, sizeof(err));
            if (rc == 0) break;
            if (rc < 0) {
                DIST_COORD_DEBUG(state, "ds4: distributed coordinator: worker %s:%s protocol error: %s\n", peer_host, peer_port, err);
                break;
            }
            if (type == DS4_DIST_MSG_HELLO) {
                DIST_COORD_DEBUG(state, "ds4: distributed coordinator: worker %s:%s sent duplicate HELLO\n", peer_host, peer_port);
                dist_discard_bytes(fd, bytes);
                break;
            }
            rc = dist_discard_bytes(fd, bytes);
            if (rc <= 0) break;
        }
    }

    dist_coordinator_remove_worker(state, fd);
    close(fd);
    return NULL;
}

static void *dist_coordinator_accept_main(void *arg) {
    ds4_dist_accept_ctx *accept_ctx = arg;
    int listen_fd = accept_ctx->listen_fd;
    ds4_dist_coordinator_state *state = accept_ctx->state;

    for (;;) {
        struct sockaddr_storage ss;
        socklen_t slen = sizeof(ss);
        int fd = accept(listen_fd, (struct sockaddr *)&ss, &slen);
        if (fd < 0) {
            if (errno == EINTR) continue;
            if (errno == EBADF || errno == EINVAL) break;
            DIST_COORD_DEBUG(state, "ds4: distributed coordinator: accept failed: %s\n", strerror(errno));
            continue;
        }
        dist_set_socket_low_latency(fd);

        ds4_dist_client_ctx *ctx = calloc(1, sizeof(*ctx));
        if (!ctx) {
            DIST_COORD_DEBUG(state, "ds4: distributed coordinator: out of memory accepting worker\n");
            close(fd);
            continue;
        }
        ctx->state = state;
        ctx->fd = fd;
        if (getnameinfo((struct sockaddr *)&ss, slen,
                        ctx->peer_host, sizeof(ctx->peer_host),
                        ctx->peer_port, sizeof(ctx->peer_port),
                        NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
            snprintf(ctx->peer_host, sizeof(ctx->peer_host), "unknown");
            snprintf(ctx->peer_port, sizeof(ctx->peer_port), "0");
        }

        pthread_t tid;
        if (pthread_create(&tid, NULL, dist_coordinator_client_main, ctx) != 0) {
            DIST_COORD_DEBUG(state, "ds4: distributed coordinator: pthread_create failed\n");
            close(fd);
            free(ctx);
            continue;
        }
        pthread_detach(tid);
    }
    return NULL;
}

static uint64_t dist_make_session_id(const void *ptr) {
    uint64_t id = ((uint64_t)(uint32_t)time(NULL) << 32) ^ (uint64_t)getpid();
    id ^= ((uint64_t)(uintptr_t)ptr << 17) ^ (uint64_t)(uintptr_t)ptr;
    id ^= (uint64_t)clock();
    return id ? id : 1u;
}

static int dist_session_ensure_route(ds4_dist_session *d, char *err, size_t errlen) {
    if (!d) {
        if (errlen) snprintf(err, errlen, "missing distributed session");
        return 1;
    }
    uint64_t generation = dist_coordinator_generation(&d->state);
    if (d->plan_ready && d->plan_generation == generation) return 0;
    dist_route_plan_free(&d->plan);
    if (!dist_coordinator_ensure_route(&d->state, &d->plan, &generation, err, errlen)) {
        d->plan_ready = false;
        d->plan_generation = 0;
        return 1;
    }
    d->plan_ready = true;
    d->plan_generation = generation;
    return 0;
}

/* =========================================================================
 * Distributed KV Snapshot Transport
 * ========================================================================= */

static int dist_write_snapshot_load_begin(
        int fd,
        const ds4_dist_snapshot_begin_fixed *begin,
        const int *tokens) {
    uint64_t token_bytes64 = (uint64_t)begin->token_count * sizeof(uint32_t);
    if (token_bytes64 > UINT32_MAX ||
        begin->token_bytes != (uint32_t)token_bytes64 ||
        begin->message_bytes != 0)
        return -1;
    uint64_t frame_bytes64 = sizeof(*begin) + token_bytes64;
    if (frame_bytes64 > UINT32_MAX) return -1;
    ds4_dist_snapshot_begin_fixed wire = *begin;
    dist_snapshot_begin_to_wire(&wire);
    if (dist_write_frame_header(fd, DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN,
                                (uint32_t)frame_bytes64) != 0)
        return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    for (uint32_t i = 0; i < begin->token_count; i++) {
        uint32_t t = htonl((uint32_t)tokens[i]);
        if (dist_write_full(fd, &t, sizeof(t)) != 0) return -1;
    }
    return 1;
}

static int dist_read_snapshot_begin_frame(
        int fd,
        ds4_dist_snapshot_begin_fixed *begin,
        char *msg,
        size_t msg_cap,
        char *err,
        size_t errlen) {
    if (msg_cap) msg[0] = '\0';
    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) {
        if (rc == 0 && errlen) snprintf(err, errlen, "distributed worker closed snapshot connection");
        return 1;
    }
    if (type != DS4_DIST_MSG_SNAPSHOT_BEGIN ||
        bytes < sizeof(ds4_dist_snapshot_begin_fixed)) {
        dist_discard_bytes(fd, bytes);
        if (errlen) snprintf(err, errlen, "distributed worker returned invalid snapshot frame");
        return 1;
    }
    rc = dist_read_full(fd, begin, sizeof(*begin));
    if (rc <= 0) {
        if (errlen) snprintf(err, errlen, "failed to read distributed snapshot header");
        return 1;
    }
    dist_snapshot_begin_from_wire(begin);
    uint32_t body = bytes - (uint32_t)sizeof(*begin);
    uint64_t expected_token_bytes = (uint64_t)begin->token_count * sizeof(uint32_t);
    if (expected_token_bytes > UINT32_MAX ||
        begin->token_bytes != (uint32_t)expected_token_bytes ||
        begin->token_bytes > body ||
        begin->message_bytes > body - begin->token_bytes) {
        dist_discard_bytes(fd, body);
        if (errlen) snprintf(err, errlen, "invalid distributed snapshot response header");
        return 1;
    }
    if (begin->token_bytes != 0) {
        rc = dist_discard_bytes(fd, begin->token_bytes);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to discard distributed snapshot response tokens");
            return 1;
        }
        body -= begin->token_bytes;
    }
    if (begin->message_bytes != 0) {
        uint32_t n = begin->message_bytes;
        uint32_t copy = msg_cap && n < msg_cap ? n : (msg_cap ? (uint32_t)msg_cap - 1u : 0u);
        if (copy != 0) {
            rc = dist_read_full(fd, msg, copy);
            if (rc <= 0) {
                if (errlen) snprintf(err, errlen, "failed to read distributed snapshot response message");
                return 1;
            }
            msg[copy] = '\0';
        }
        if (n > copy) {
            rc = dist_discard_bytes(fd, n - copy);
            if (rc <= 0) {
                if (errlen) snprintf(err, errlen, "failed to discard distributed snapshot response message");
                return 1;
            }
        }
        body -= n;
    }
    if (body != 0) {
        rc = dist_discard_bytes(fd, body);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to discard trailing distributed snapshot response bytes");
            return 1;
        }
    }
    return 0;
}

static int dist_read_snapshot_done_frame(
        int fd,
        uint64_t request_id,
        char *err,
        size_t errlen) {
    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) {
        if (rc == 0 && errlen) snprintf(err, errlen, "distributed worker closed before snapshot completion");
        return 1;
    }
    if (type != DS4_DIST_MSG_SNAPSHOT_DONE ||
        bytes < sizeof(ds4_dist_snapshot_done_fixed)) {
        dist_discard_bytes(fd, bytes);
        if (errlen) snprintf(err, errlen, "distributed worker returned invalid snapshot completion frame");
        return 1;
    }
    ds4_dist_snapshot_done_fixed done;
    rc = dist_read_full(fd, &done, sizeof(done));
    if (rc <= 0) {
        if (errlen) snprintf(err, errlen, "failed to read distributed snapshot completion");
        return 1;
    }
    dist_snapshot_done_from_wire(&done);
    uint32_t body = bytes - (uint32_t)sizeof(done);
    char msg[256];
    msg[0] = '\0';
    if (done.message_bytes > body) {
        dist_discard_bytes(fd, body);
        if (errlen) snprintf(err, errlen, "invalid distributed snapshot completion message");
        return 1;
    }
    if (done.message_bytes != 0) {
        uint32_t copy = done.message_bytes < sizeof(msg) ?
            done.message_bytes : (uint32_t)sizeof(msg) - 1u;
        rc = dist_read_full(fd, msg, copy);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to read distributed snapshot completion message");
            return 1;
        }
        msg[copy] = '\0';
        if (done.message_bytes > copy) {
            rc = dist_discard_bytes(fd, done.message_bytes - copy);
            if (rc <= 0) {
                if (errlen) snprintf(err, errlen, "failed to discard distributed snapshot completion message");
                return 1;
            }
        }
        body -= done.message_bytes;
    }
    if (body != 0) {
        rc = dist_discard_bytes(fd, body);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to discard trailing distributed snapshot completion bytes");
            return 1;
        }
    }
    uint64_t got_request = dist_u64_from_halves(done.request_hi, done.request_lo);
    if (got_request != request_id) {
        if (errlen) snprintf(err, errlen, "distributed snapshot completion request mismatch");
        return 1;
    }
    if (done.status != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             msg[0] ? msg : "distributed worker failed snapshot request");
        return 1;
    }
    return 0;
}

static int dist_receive_snapshot_chunks_to_file(
        int fd,
        uint64_t request_id,
        FILE *fp,
        uint64_t payload_bytes,
        char *err,
        size_t errlen) {
    uint8_t *buf = malloc(DS4_DIST_SNAPSHOT_CHUNK_BYTES);
    if (!buf) {
        if (errlen) snprintf(err, errlen, "out of memory receiving distributed KV shard");
        return 1;
    }
    uint64_t received = 0;
    int fail = 0;
    while (!fail && received < payload_bytes) {
        uint32_t type = 0, bytes = 0;
        int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
        if (rc <= 0) {
            if (rc == 0 && errlen) snprintf(err, errlen, "distributed worker closed while sending KV shard");
            fail = 1;
            break;
        }
        if (type != DS4_DIST_MSG_SNAPSHOT_CHUNK ||
            bytes < sizeof(ds4_dist_snapshot_chunk_fixed)) {
            dist_discard_bytes(fd, bytes);
            if (errlen) snprintf(err, errlen, "expected distributed KV shard chunk");
            fail = 1;
            break;
        }
        ds4_dist_snapshot_chunk_fixed chunk;
        rc = dist_read_full(fd, &chunk, sizeof(chunk));
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to read distributed KV shard chunk header");
            fail = 1;
            break;
        }
        dist_snapshot_chunk_from_wire(&chunk);
        uint64_t got_request = dist_u64_from_halves(chunk.request_hi,
                                                    chunk.request_lo);
        uint32_t chunk_bytes = bytes - (uint32_t)sizeof(chunk);
        if (got_request != request_id ||
            chunk.chunk_bytes != chunk_bytes ||
            chunk_bytes > DS4_DIST_SNAPSHOT_CHUNK_BYTES ||
            chunk_bytes > payload_bytes - received) {
            dist_discard_bytes(fd, chunk_bytes);
            if (errlen) snprintf(err, errlen, "invalid distributed KV shard chunk");
            fail = 1;
            break;
        }
        rc = dist_read_full(fd, buf, chunk_bytes);
        if (rc <= 0) {
            if (errlen) snprintf(err, errlen, "failed to read distributed KV shard chunk");
            fail = 1;
            break;
        }
        if (fwrite(buf, 1, chunk_bytes, fp) != chunk_bytes) {
            if (errlen) snprintf(err, errlen, "failed to write distributed KV shard");
            fail = 1;
            break;
        }
        received += chunk_bytes;
    }
    free(buf);
    return fail;
}

