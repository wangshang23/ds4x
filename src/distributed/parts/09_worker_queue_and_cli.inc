/* =========================================================================
 * Worker Prefetch Queue
 * ========================================================================= */

static void dist_worker_job_free(ds4_dist_worker_job *job) {
    if (!job) return;
    free(job->payload);
    free(job);
}

static void dist_worker_job_queue_init(
        ds4_dist_worker_job_queue *q,
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream) {
    memset(q, 0, sizeof(*q));
    q->state = state;
    q->upstream = upstream;
    q->depth = dist_worker_prefetch_depth();
    pthread_mutex_init(&q->mu, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

static void dist_worker_job_queue_clear_locked(ds4_dist_worker_job_queue *q) {
    ds4_dist_worker_job *it = q->head;
    q->head = NULL;
    q->tail = NULL;
    q->queued = 0;
    while (it) {
        ds4_dist_worker_job *next = it->next;
        dist_worker_job_free(it);
        it = next;
    }
}

static void dist_worker_job_queue_destroy(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    dist_worker_job_queue_clear_locked(q);
    pthread_mutex_unlock(&q->mu);
    pthread_cond_destroy(&q->not_full);
    pthread_cond_destroy(&q->not_empty);
    pthread_mutex_destroy(&q->mu);
}

static void dist_worker_job_queue_finish(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    q->closed = true;
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->mu);
}

static void dist_worker_job_queue_cancel(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    q->closed = true;
    q->canceled = true;
    dist_worker_job_queue_clear_locked(q);
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->mu);
}

static bool dist_worker_job_queue_enqueue(
        ds4_dist_worker_job_queue *q,
        ds4_dist_worker_job *job) {
    pthread_mutex_lock(&q->mu);
    while (!q->closed && !q->canceled && q->queued >= q->depth) {
        pthread_cond_wait(&q->not_full, &q->mu);
    }
    if (q->closed || q->canceled) {
        pthread_mutex_unlock(&q->mu);
        return false;
    }
    if (q->tail) q->tail->next = job;
    else q->head = job;
    q->tail = job;
    q->queued++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mu);
    return true;
}

static ds4_dist_worker_job *dist_worker_job_queue_pop(ds4_dist_worker_job_queue *q) {
    pthread_mutex_lock(&q->mu);
    while (!q->head && !q->closed && !q->canceled) {
        pthread_cond_wait(&q->not_empty, &q->mu);
    }
    if (q->canceled || !q->head) {
        pthread_mutex_unlock(&q->mu);
        return NULL;
    }
    ds4_dist_worker_job *job = q->head;
    q->head = job->next;
    if (!q->head) q->tail = NULL;
    q->queued--;
    job->next = NULL;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mu);
    return job;
}

static void *dist_worker_prefetch_eval_main(void *arg) {
    ds4_dist_worker_job_queue *q = arg;
    for (;;) {
        ds4_dist_worker_job *job = dist_worker_job_queue_pop(q);
        if (!job) break;
        int rc = dist_worker_process_work_payload(q->state,
                                                  q->upstream,
                                                  job->payload,
                                                  job->bytes);
        dist_worker_job_free(job);
        if (rc <= 0) {
            pthread_mutex_lock(&q->mu);
            q->rc = rc == 0 ? 0 : 1;
            pthread_mutex_unlock(&q->mu);
            dist_worker_job_queue_cancel(q);
            shutdown(q->upstream->fd, SHUT_RDWR);
            break;
        }
    }
    return NULL;
}

static int dist_worker_read_loop_prefetch(ds4_dist_worker_state *state, int fd) {
    ds4_dist_worker_upstream upstream;
    dist_worker_upstream_init(&upstream, state, fd);

    ds4_dist_worker_job_queue queue;
    dist_worker_job_queue_init(&queue, state, &upstream);

    pthread_t eval_tid;
    if (pthread_create(&eval_tid, NULL, dist_worker_prefetch_eval_main, &queue) != 0) {
        dist_worker_job_queue_destroy(&queue);
        dist_worker_upstream_destroy(&upstream);
        return 1;
    }

    int loop_rc = 0;
    fprintf(stderr,
            "ds4: distributed worker: receive prefetch depth %u enabled\n",
            queue.depth);

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
            ds4_dist_worker_job *job = calloc(1, sizeof(*job));
            if (!job) {
                dist_discard_bytes(fd, bytes);
                dist_worker_upstream_send_work_error(&upstream, 0, "out of memory queueing distributed WORK");
                loop_rc = 1;
                break;
            }
            job->payload = malloc(bytes);
            job->bytes = bytes;
            if (!job->payload) {
                dist_worker_job_free(job);
                dist_discard_bytes(fd, bytes);
                dist_worker_upstream_send_work_error(&upstream, 0, "out of memory reading distributed WORK frame");
                loop_rc = 1;
                break;
            }
            rc = dist_read_full(fd, job->payload, bytes);
            if (rc <= 0) {
                dist_worker_job_free(job);
                loop_rc = rc == 0 ? 0 : 1;
                break;
            }
            if (!dist_worker_job_queue_enqueue(&queue, job)) {
                dist_worker_job_free(job);
                loop_rc = 1;
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

    if (loop_rc == 0) dist_worker_job_queue_finish(&queue);
    else dist_worker_job_queue_cancel(&queue);
    pthread_join(eval_tid, NULL);
    if (loop_rc == 0 && queue.rc != 0) loop_rc = 1;
    dist_worker_job_queue_destroy(&queue);
    dist_worker_upstream_destroy(&upstream);
    return loop_rc;
}

static void *dist_worker_data_client_main(void *arg) {
    ds4_dist_data_client_ctx *ctx = arg;
    ds4_dist_worker_state *state = ctx->state;
    int fd = ctx->fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
    snprintf(peer_host, sizeof(peer_host), "%s", ctx->peer_host);
    snprintf(peer_port, sizeof(peer_port), "%s", ctx->peer_port);
    free(ctx);

    int rc = getenv("DS4_DIST_DISABLE_WORKER_PREFETCH")
        ? dist_worker_read_loop(state, fd)
        : dist_worker_read_loop_prefetch(state, fd);
    if (rc != 0) {
        fprintf(stderr,
                "ds4: distributed worker: data connection %s:%s closed after error\n",
                peer_host,
                peer_port);
    }

    close(fd);
    return NULL;
}

static void *dist_worker_data_listener_main(void *arg) {
    ds4_dist_worker_state *state = arg;
    int listen_fd = state->listen_fd;
    for (;;) {
        struct sockaddr_storage ss;
        socklen_t slen = sizeof(ss);
        int fd = accept(listen_fd, (struct sockaddr *)&ss, &slen);
        if (fd < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "ds4: distributed worker: data accept failed: %s\n", strerror(errno));
            continue;
        }
        dist_set_socket_low_latency(fd);

        ds4_dist_data_client_ctx *ctx = calloc(1, sizeof(*ctx));
        if (!ctx) {
            fprintf(stderr, "ds4: distributed worker: out of memory accepting data connection\n");
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
        if (pthread_create(&tid, NULL, dist_worker_data_client_main, ctx) != 0) {
            fprintf(stderr, "ds4: distributed worker: pthread_create failed for data connection\n");
            close(fd);
            free(ctx);
            continue;
        }
        pthread_detach(tid);
    }
    return NULL;
}

/* =========================================================================
 * Worker Entrypoint
 * ========================================================================= */

static int dist_run_worker(ds4_engine *engine, const ds4_dist_options *opt, int ctx_size) {
    char layer_end[32];
    if (opt->layers.has_output) snprintf(layer_end, sizeof(layer_end), "output");
    else snprintf(layer_end, sizeof(layer_end), "%u", opt->layers.end);

    char err[256];
    const char *listen_host = opt->listen_host;
    int requested_port = opt->listen_port > 0 ? opt->listen_port : 0;
    int listen_fd = dist_open_listener(listen_host, requested_port, err, sizeof(err));
    if (listen_fd < 0) {
        fprintf(stderr, "ds4: distributed worker: %s\n", err);
        return 1;
    }
    int listen_port_i = dist_listener_port(listen_fd);
    if (listen_port_i <= 0) {
        fprintf(stderr, "ds4: distributed worker: could not determine data listener port\n");
        close(listen_fd);
        return 1;
    }
    const uint32_t listen_port = (uint32_t)listen_port_i;

    ds4_dist_worker_state state;
    memset(&state, 0, sizeof(state));
    state.engine = engine;
    state.model_id = (uint32_t)ds4_engine_model_id(engine);
    state.layer_start = opt->layers.start;
    state.layer_end = dist_resolved_layer_end(opt, (uint32_t)ds4_engine_layer_count(engine));
    state.has_output = opt->layers.has_output;
    state.ctx_size = ctx_size;
    state.listen_fd = listen_fd;
    pthread_mutex_init(&state.mu, NULL);

    pthread_t data_tid;
    if (pthread_create(&data_tid, NULL, dist_worker_data_listener_main, &state) != 0) {
        fprintf(stderr, "ds4: distributed worker: pthread_create failed for data listener\n");
        close(listen_fd);
        return 1;
    }
    pthread_detach(data_tid);

    fprintf(stderr,
            "ds4: distributed worker: layers %u:%s model_id=%d data_listen=%s:%u connecting to coordinator %s:%d\n",
            opt->layers.start,
            layer_end,
            ds4_engine_model_id(engine),
            listen_host ? listen_host : "*",
            listen_port,
            opt->coordinator_host,
            opt->coordinator_port);

    for (;;) {
        int fd = dist_connect_endpoint(opt->coordinator_host, opt->coordinator_port, err, sizeof(err));
        if (fd < 0) {
            fprintf(stderr, "ds4: distributed worker: %s; retrying\n", err);
            dist_sleep_reconnect();
            continue;
        }

        char peer_host[NI_MAXHOST], peer_port[NI_MAXSERV];
        dist_peer_name(fd, peer_host, sizeof(peer_host), peer_port, sizeof(peer_port));
        fprintf(stderr, "ds4: distributed worker: connected to coordinator %s:%s\n", peer_host, peer_port);

        if (dist_send_hello(engine, opt, ctx_size, listen_port, fd) != 0) {
            fprintf(stderr, "ds4: distributed worker: failed to send HELLO: %s\n", strerror(errno));
            close(fd);
            dist_sleep_reconnect();
            continue;
        }

        int rc = getenv("DS4_DIST_DISABLE_WORKER_PREFETCH")
            ? dist_worker_read_loop(&state, fd)
            : dist_worker_read_loop_prefetch(&state, fd);
        close(fd);
        uint32_t dropped_sessions = dist_worker_clear_sessions(&state);
        if (dropped_sessions) {
            fprintf(stderr,
                    "ds4: distributed worker: cleared %u sessions after coordinator disconnect\n",
                    dropped_sessions);
        }
        fprintf(stderr, "ds4: distributed worker: coordinator disconnected%s; reconnecting\n",
                rc ? " after error" : "");
        dist_sleep_reconnect();
    }
}

/* =========================================================================
 * CLI Option Parsing And Public Entrypoint
 * ========================================================================= */

static bool dist_parse_role(const char *s, ds4_distributed_role *out) {
    if (!s || !out) return false;
    if (!strcmp(s, "none")) {
        *out = DS4_DISTRIBUTED_NONE;
        return true;
    }
    if (!strcmp(s, "coordinator")) {
        *out = DS4_DISTRIBUTED_COORDINATOR;
        return true;
    }
    if (!strcmp(s, "worker")) {
        *out = DS4_DISTRIBUTED_WORKER;
        return true;
    }
    return false;
}

static bool dist_parse_u32_component(const char *p, size_t len, uint32_t *out) {
    if (!p || len == 0 || !out) return false;
    char buf[32];
    if (len >= sizeof(buf)) return false;
    memcpy(buf, p, len);
    buf[len] = '\0';

    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(buf, &end, 10);
    if (errno != 0 || end == buf || *end != '\0' || v > UINT32_MAX) return false;
    *out = (uint32_t)v;
    return true;
}

static bool dist_parse_layers(const char *s, ds4_distributed_layers *out, char *err, size_t errlen) {
    if (!s || !out) {
        if (errlen) snprintf(err, errlen, "missing layer range");
        return false;
    }

    const char *colon = strchr(s, ':');
    if (!colon || colon == s || colon[1] == '\0') {
        if (errlen) snprintf(err, errlen, "expected A:B or A:output");
        return false;
    }
    if (strchr(colon + 1, ':')) {
        if (errlen) snprintf(err, errlen, "layer range has too many ':' separators");
        return false;
    }

    ds4_distributed_layers parsed = {0};
    if (!dist_parse_u32_component(s, (size_t)(colon - s), &parsed.start)) {
        if (errlen) snprintf(err, errlen, "invalid start layer in %s", s);
        return false;
    }

    const char *end = colon + 1;
    if (!strcmp(end, "output")) {
        parsed.end = UINT32_MAX;
        parsed.has_output = true;
    } else {
        if (!dist_parse_u32_component(end, strlen(end), &parsed.end)) {
            if (errlen) snprintf(err, errlen, "invalid end layer in %s", s);
            return false;
        }
        if (parsed.end < parsed.start) {
            if (errlen) snprintf(err, errlen, "layer range end precedes start in %s", s);
            return false;
        }
    }

    parsed.set = true;
    *out = parsed;
    return true;
}

static const char *dist_cli_need_arg(
        int *index,
        int argc,
        char **argv,
        const char *arg,
        char *err,
        size_t errlen) {
    if (!index || !argv || *index + 1 >= argc) {
        if (errlen) snprintf(err, errlen, "%s requires an argument", arg);
        return NULL;
    }
    return argv[++*index];
}

static bool dist_cli_parse_port(const char *s, const char *arg, int *out, char *err, size_t errlen) {
    if (!s || !out) {
        if (errlen) snprintf(err, errlen, "%s requires a TCP port", arg);
        return false;
    }
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno != 0 || s[0] == '\0' || *end != '\0' || v <= 0 || v > 65535) {
        if (errlen) snprintf(err, errlen, "invalid value for %s: %s", arg, s);
        return false;
    }
    *out = (int)v;
    return true;
}

bool ds4_dist_enabled(const ds4_dist_options *opt) {
    return opt && opt->role != DS4_DISTRIBUTED_NONE;
}

ds4_dist_options *ds4_dist_options_create(void) {
    return calloc(1, sizeof(ds4_dist_options));
}

void ds4_dist_options_free(ds4_dist_options *opt) {
    free(opt);
}

void ds4_dist_usage(FILE *fp) {
    fprintf(fp,
        "  --role ROLE\n"
        "      Distributed role: coordinator or worker.\n"
        "  --layers A:B\n"
        "      Inclusive distributed layer slice, e.g. 10:20 or 21:output.\n"
        "  --listen HOST PORT\n"
        "      Coordinator TCP listen address. Workers may later use it to force their data listener.\n"
        "  --coordinator HOST PORT\n"
        "      Coordinator TCP address for --role worker.\n"
        "  --dist-prefill-chunk N\n"
        "      Coordinator prefill pipeline chunk size. Default: session cap, normally 4096.\n"
        "      Non-default values are experimental and can change logits unless validated.\n"
        "  --dist-prefill-window N\n"
        "      Coordinator max end-to-end prefill chunks in flight. Default: workers+2, capped at 8.\n"
        "  --dist-activation-bits N\n"
        "      Coordinator hidden-state transport width: 32, 16, or 8. Default: 32.\n"
        "  --dist-replay-check\n"
        "      Coordinator diagnostic: reset and replay the prompt, then compare logits.\n"
        "  --debug\n"
        "      Print coordinator route/debug logs. Workers keep their normal logs without this.\n"
    );
}

ds4_dist_cli_parse_result ds4_dist_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_dist_options *opt,
        char *err,
        size_t errlen) {
    if (!arg) return DS4_DIST_CLI_NOT_MATCHED;
    if (!strcmp(arg, "--role")) {
        const char *role = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!role) return DS4_DIST_CLI_ERROR;
        if (!opt || !dist_parse_role(role, &opt->role)) {
            if (errlen) snprintf(err, errlen,
                                 "invalid distributed role: %s (valid roles: none, coordinator, worker)",
                                 role);
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--layers")) {
        const char *layers = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!layers) return DS4_DIST_CLI_ERROR;
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (!dist_parse_layers(layers, &opt->layers, err, errlen)) {
            char detail[160];
            if (errlen && err[0] != '\0') {
                snprintf(detail, sizeof(detail), "%s", err);
                snprintf(err, errlen, "invalid --layers %s: %s", layers, detail);
            }
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--listen")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->listen_host || opt->listen_port) {
            if (errlen) snprintf(err, errlen, "specify --listen only once");
            return DS4_DIST_CLI_ERROR;
        }
        const char *host = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!host) return DS4_DIST_CLI_ERROR;
        const char *port = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!port) return DS4_DIST_CLI_ERROR;
        if (!dist_cli_parse_port(port, arg, &opt->listen_port, err, errlen)) return DS4_DIST_CLI_ERROR;
        opt->listen_host = host;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--coordinator")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->coordinator_host || opt->coordinator_port) {
            if (errlen) snprintf(err, errlen, "specify --coordinator only once");
            return DS4_DIST_CLI_ERROR;
        }
        const char *host = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!host) return DS4_DIST_CLI_ERROR;
        const char *port = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!port) return DS4_DIST_CLI_ERROR;
        if (!dist_cli_parse_port(port, arg, &opt->coordinator_port, err, errlen)) return DS4_DIST_CLI_ERROR;
        opt->coordinator_host = host;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-prefill-chunk")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        if (!dist_parse_positive_u32(value, arg, &opt->prefill_chunk, err, errlen)) {
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-prefill-window")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        if (!dist_parse_positive_u32(value, arg, &opt->prefill_window, err, errlen)) {
            return DS4_DIST_CLI_ERROR;
        }
        if (opt->prefill_window > 64u) {
            if (errlen) snprintf(err, errlen, "%s must be <= 64", arg);
            return DS4_DIST_CLI_ERROR;
        }
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-activation-bits")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        const char *value = dist_cli_need_arg(index, argc, argv, arg, err, errlen);
        if (!value) return DS4_DIST_CLI_ERROR;
        uint32_t bits = 0;
        if (!dist_parse_positive_u32(value, arg, &bits, err, errlen)) {
            return DS4_DIST_CLI_ERROR;
        }
        if (!dist_activation_bits_valid(bits)) {
            if (errlen) snprintf(err, errlen, "%s must be 32, 16, or 8", arg);
            return DS4_DIST_CLI_ERROR;
        }
        opt->activation_bits = bits;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--dist-replay-check")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        opt->replay_check = true;
        return DS4_DIST_CLI_MATCHED;
    }
    if (!strcmp(arg, "--debug")) {
        if (!opt) {
            if (errlen) snprintf(err, errlen, "missing distributed options");
            return DS4_DIST_CLI_ERROR;
        }
        opt->debug = true;
        return DS4_DIST_CLI_MATCHED;
    }
    return DS4_DIST_CLI_NOT_MATCHED;
}

static int dist_validate_options(const ds4_dist_options *opt, char *err, size_t errlen) {
    if (!opt) {
        if (errlen) snprintf(err, errlen, "missing distributed options");
        return 1;
    }

    if (opt->role == DS4_DISTRIBUTED_NONE) {
        if (opt->layers.set || opt->listen_host || opt->listen_port ||
            opt->coordinator_host || opt->coordinator_port ||
            opt->prefill_chunk != 0 || opt->prefill_window != 0 ||
            opt->activation_bits != 0) {
            if (errlen) snprintf(err, errlen, "distributed options require --role coordinator or --role worker");
            return 1;
        }
        return 0;
    }

    if (!opt->layers.set) {
        if (errlen) snprintf(err, errlen, "--role %s requires --layers", dist_role_name(opt->role));
        return 1;
    }
    if (opt->prefill_window > 64u) {
        if (errlen) snprintf(err, errlen, "--dist-prefill-window must be <= 64");
        return 1;
    }
    if (opt->activation_bits != 0 && !dist_activation_bits_valid(opt->activation_bits)) {
        if (errlen) snprintf(err, errlen, "--dist-activation-bits must be 32, 16, or 8");
        return 1;
    }

    if (opt->role == DS4_DISTRIBUTED_COORDINATOR) {
        if (!opt->listen_host || opt->listen_port <= 0) {
            if (errlen) snprintf(err, errlen, "--role coordinator requires --listen HOST PORT");
            return 1;
        }
        if (opt->coordinator_host || opt->coordinator_port) {
            if (errlen) snprintf(err, errlen, "--role coordinator must not use --coordinator");
            return 1;
        }
        return 0;
    }

    if (opt->role == DS4_DISTRIBUTED_WORKER) {
        if (!opt->coordinator_host || opt->coordinator_port <= 0) {
            if (errlen) snprintf(err, errlen, "--role worker requires --coordinator HOST PORT");
            return 1;
        }
        if (opt->prefill_chunk != 0) {
            if (errlen) snprintf(err, errlen, "--dist-prefill-chunk requires --role coordinator");
            return 1;
        }
        if (opt->prefill_window != 0) {
            if (errlen) snprintf(err, errlen, "--dist-prefill-window requires --role coordinator");
            return 1;
        }
        if (opt->activation_bits != 0) {
            if (errlen) snprintf(err, errlen, "--dist-activation-bits requires --role coordinator");
            return 1;
        }
        return 0;
    }

    if (errlen) snprintf(err, errlen, "invalid distributed role");
    return 1;
}

int ds4_dist_prepare_engine_options(
        const ds4_dist_options *opt,
        ds4_engine_options *engine,
        char *err,
        size_t errlen) {
    if (dist_validate_options(opt, err, errlen) != 0) return 1;
    if (opt && opt->replay_check && opt->role != DS4_DISTRIBUTED_COORDINATOR) {
        if (errlen) snprintf(err, errlen, "--dist-replay-check requires --role coordinator");
        return 1;
    }
    if (engine && opt) {
        engine->distributed = *opt;
        if (ds4_dist_enabled(opt)) {
            engine->load_slice = true;
            engine->load_layer_start = opt->layers.start;
            engine->load_layer_end = opt->layers.has_output ? UINT32_MAX : opt->layers.end;
            engine->load_output = opt->layers.has_output;
        }
    }
    return 0;
}

static int dist_validate_layers_for_model(const ds4_dist_options *opt, uint32_t n_layers, char *err, size_t errlen) {
    if (!opt || opt->role == DS4_DISTRIBUTED_NONE || !opt->layers.set) return 0;
    if (n_layers == 0) {
        if (errlen) snprintf(err, errlen, "model reports no layers");
        return 1;
    }

    const uint32_t last = n_layers - 1u;
    if (opt->layers.start > last) {
        if (errlen) snprintf(err, errlen, "layer range starts past final model layer %u", last);
        return 1;
    }
    if (!opt->layers.has_output && opt->layers.end > last) {
        if (errlen) snprintf(err, errlen, "layer range ends past final model layer %u", last);
        return 1;
    }
    if (opt->role == DS4_DISTRIBUTED_COORDINATOR && opt->layers.start != 0) {
        if (errlen) snprintf(err, errlen, "coordinator layer range must start at layer 0");
        return 1;
    }
    return 0;
}

int ds4_dist_run(ds4_engine *engine, const ds4_dist_options *opt, const ds4_dist_generation_options *gen) {
    if (!engine || !opt) {
        fprintf(stderr, "ds4: distributed runtime requires an open engine and options\n");
        return 1;
    }
    char err[256];
    if (dist_validate_options(opt, err, sizeof(err)) != 0 ||
        dist_validate_layers_for_model(opt, (uint32_t)ds4_engine_layer_count(engine), err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4: %s\n", err);
        return 2;
    }

    signal(SIGPIPE, SIG_IGN);

    if (opt->role == DS4_DISTRIBUTED_COORDINATOR) {
        return dist_run_coordinator(engine, opt, gen);
    }
    if (opt->role == DS4_DISTRIBUTED_WORKER) {
        return dist_run_worker(engine, opt, gen ? gen->ctx_size : 0);
    }

    fprintf(stderr, "ds4: distributed runtime requested without a distributed role\n");
    return 1;
}
