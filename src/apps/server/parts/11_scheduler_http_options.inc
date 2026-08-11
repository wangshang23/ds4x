/* Keep cancellation installed for the entire request, including every early
 * return in the large protocol/generation path. The callback is cleared before
 * the client thread can destroy its stack-owned job. */
static void generate_job(server *s, server_slot *slot, job *j) {
    pthread_mutex_lock(&s->model_mu);
    slot->running = j;
    pthread_mutex_unlock(&s->model_mu);

    ds4_session_set_cancel(slot->session, job_cancelled, j);
    if (!job_cancelled(j)) generate_job_inner(s, slot, j);
    ds4_session_set_cancel(slot->session, NULL, NULL);

    pthread_mutex_lock(&s->model_mu);
    if (slot->running == j) slot->running = NULL;
    pthread_cond_broadcast(&s->model_cv);
    pthread_mutex_unlock(&s->model_mu);
}

static bool live_state_contains_all(const live_tool_state *state,
                                    const stop_list *ids) {
    if (!state || !state->valid || !ids || ids->len == 0) return false;
    for (int i = 0; i < ids->len; i++) {
        if (!id_list_contains(&state->call_ids, ids->v[i])) return false;
    }
    return true;
}

/* Return the only slot eligible for an explicit live continuation, or -1 when
 * the request has no resident binding. A missing binding is intentionally not
 * treated as ineligible: generate_job() then emits the existing 409 response. */
static int job_required_slot_locked(server *s, const job *j) {
    if (!s || !j) return -1;
    const request *r = &j->req;
    for (int i = 0; i < s->slot_count; i++) {
        server_slot *slot = &s->slots[i];
        if (r->responses_requires_live_tool_state &&
            live_state_contains_all(&slot->responses_live,
                                    &r->responses_live_call_ids)) {
            return i;
        }
        if (r->anthropic_requires_live_tool_state &&
            live_state_contains_all(&slot->anthropic_live,
                                    &r->anthropic_live_call_ids)) {
            return i;
        }
    }
    return -1;
}

static int job_slot_score(server *s, server_slot *slot, const job *j,
                          int required_slot) {
    if (!s || !slot || !j || slot->busy || slot->assigned) return INT_MIN;
    if (required_slot >= 0 && slot->id != required_slot) return INT_MIN;
    if (required_slot == slot->id) return INT_MAX;
    int common = ds4_session_common_prefix(slot->session, &j->req.prompt);
    return common;
}

static void dispatch_jobs_locked(server *s) {
    if (!s || !s->batched_mode) return;
    for (;;) {
        job *chosen = NULL;
        job *chosen_prev = NULL;
        server_slot *chosen_slot = NULL;
        int chosen_score = INT_MIN;

        pthread_mutex_lock(&s->tool_mu);
        job *prev = NULL;
        for (job *j = s->head; j; prev = j, j = j->next) {
            int required = job_required_slot_locked(s, j);
            server_slot *best = NULL;
            int best_score = INT_MIN;
            for (int i = 0; i < s->slot_count; i++) {
                int score = job_slot_score(s, &s->slots[i], j, required);
                if (score > best_score) {
                    best_score = score;
                    best = &s->slots[i];
                }
            }
            if (best) {
                chosen = j;
                chosen_prev = prev;
                chosen_slot = best;
                chosen_score = best_score;
                break; /* FIFO among jobs that can run now. */
            }
        }
        pthread_mutex_unlock(&s->tool_mu);
        (void)chosen_score;
        if (!chosen || !chosen_slot) break;

        if (chosen_prev) chosen_prev->next = chosen->next;
        else s->head = chosen->next;
        if (s->tail == chosen) s->tail = chosen_prev;
        chosen->next = NULL;
        chosen_slot->assigned = chosen;
        chosen_slot->busy = true;
        pthread_cond_broadcast(&s->cv);
    }
}

static bool enqueue(server *s, job *j) {
    pthread_mutex_lock(&s->mu);
    if (s->stopping) {
        pthread_mutex_unlock(&s->mu);
        return false;
    }
    if (s->tail) s->tail->next = j; else s->head = j;
    s->tail = j;
    if (s->batched_mode) {
        dispatch_jobs_locked(s);
        pthread_cond_broadcast(&s->cv);
    } else {
        pthread_cond_signal(&s->cv);
    }
    pthread_mutex_unlock(&s->mu);
    return true;
}

static job *dequeue(server *s) {
    pthread_mutex_lock(&s->mu);
    while (!s->head && !s->stopping) pthread_cond_wait(&s->cv, &s->mu);
    if (!s->head) {
        pthread_mutex_unlock(&s->mu);
        return NULL;
    }
    job *j = s->head;
    s->head = j->next;
    if (!s->head) s->tail = NULL;
    pthread_mutex_unlock(&s->mu);
    j->next = NULL;
    return j;
}

static void *worker_main(void *arg) {
    server *s = arg;
    for (;;) {
        job *j = dequeue(s);
        if (!j) break;
        generate_job(s, &s->slots[0], j);
        job_complete(j);
    }
    return NULL;
}

static void *slot_worker_main(void *arg) {
    server_slot *slot = arg;
    server *s = slot->srv;
    for (;;) {
        pthread_mutex_lock(&s->mu);
        while (!slot->assigned && (!s->stopping || s->head)) {
            pthread_cond_wait(&s->cv, &s->mu);
        }
        if (!slot->assigned && s->stopping && !s->head) {
            pthread_mutex_unlock(&s->mu);
            break;
        }
        job *j = slot->assigned;
        slot->assigned = NULL;
        pthread_mutex_unlock(&s->mu);

        generate_job(s, slot, j);
        job_complete(j);

        pthread_mutex_lock(&s->mu);
        slot->busy = false;
        dispatch_jobs_locked(s);
        pthread_mutex_unlock(&s->mu);
    }
    return NULL;
}

typedef struct {
    char method[8];
    char path[256];
    char *body;
    size_t body_len;
} http_request;

static void http_request_free(http_request *r) {
    free(r->body);
    memset(r, 0, sizeof(*r));
}

static ssize_t header_end(const char *p, size_t n) {
    for (size_t i = 3; i < n; i++) {
        if (p[i - 3] == '\r' && p[i - 2] == '\n' && p[i - 1] == '\r' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    for (size_t i = 1; i < n; i++) {
        if (p[i - 1] == '\n' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    return -1;
}

static long content_length(const char *h, size_t n) {
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len >= 15 && strncasecmp(line, "Content-Length:", 15) == 0) {
            const char *v = line + 15;
            while (v < line + len && isspace((unsigned char)*v)) v++;
            return strtol(v, NULL, 10);
        }
        if (p < end) p++;
    }
    return 0;
}

static bool read_http_request(int fd, http_request *r) {
    buf b = {0};
    ssize_t hend = -1;
    const size_t max_header = 64 * 1024;
    const size_t max_body = 64 * 1024 * 1024;

    while (hend < 0 && b.len < max_header) {
        char tmp[4096];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        buf_append(&b, tmp, (size_t)n);
        hend = header_end(b.ptr, b.len);
    }
    if (hend < 0) goto fail;

    char line[512];
    size_t i = 0;
    while (i < b.len && b.ptr[i] != '\n' && i + 1 < sizeof(line)) {
        line[i] = b.ptr[i];
        i++;
    }
    line[i] = '\0';
    if (sscanf(line, "%7s %255s", r->method, r->path) != 2) goto fail;
    char *q = strchr(r->path, '?');
    if (q) *q = '\0';

    long clen = content_length(b.ptr, (size_t)hend);
    if (clen < 0 || (size_t)clen > max_body) goto fail;
    while (b.len < (size_t)hend + (size_t)clen) {
        char tmp[8192];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        buf_append(&b, tmp, (size_t)n);
    }

    r->body_len = (size_t)clen;
    r->body = xmalloc(r->body_len + 1);
    memcpy(r->body, b.ptr + hend, r->body_len);
    r->body[r->body_len] = '\0';
    buf_free(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

typedef struct {
    server *srv;
    int fd;
} client_arg;

static void append_model_json_values(buf *b, const char *id, const char *name,
                                     int ctx, int default_tokens) {
    const int max_completion = default_tokens < ctx ? default_tokens : ctx;
    buf_printf(b,
        "{\"id\":");
    json_escape(b, id);
    buf_puts(b,
        ",\"object\":\"model\","
        "\"created\":1767225600,"
        "\"owned_by\":\"ds4.c\","
        "\"name\":");
    json_escape(b, name);
    buf_printf(b,
        ","
        "\"context_length\":%d,"
        "\"top_provider\":{"
            "\"context_length\":%d,"
            "\"max_completion_tokens\":%d,"
            "\"is_moderated\":false},"
        "\"supported_parameters\":["
            "\"tools\","
            "\"tool_choice\","
            "\"max_tokens\","
            "\"temperature\","
            "\"top_p\","
            "\"top_k\","
            "\"min_p\","
            "\"stop\","
            "\"seed\","
            "\"stream\","
            "\"reasoning_effort\"]}",
        ctx,
        ctx,
        max_completion);
}

static void append_model_json(buf *b, const server *s, const char *id) {
    append_model_json_values(b,
                             id,
                             ds4_engine_model_name(s->engine),
                             s->ctx_size,
                             s->default_tokens);
}

static bool send_model(server *s, int fd, const char *id) {
    buf b = {0};
    append_model_json(&b, s, id);
    buf_putc(&b, '\n');
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static bool send_models(server *s, int fd) {
    buf b = {0};
    buf_puts(&b, "{\"object\":\"list\",\"data\":[");
    if (ds4_engine_is_glm_dsa(s->engine)) {
        append_model_json(&b, s, "glm-5.2");
        buf_putc(&b, ',');
        append_model_json(&b, s, "glm-5.2-chat");
        buf_putc(&b, ',');
        append_model_json(&b, s, "glm-5.2-reasoner");
    } else {
        append_model_json(&b, s, "deepseek-v4-flash");
        buf_putc(&b, ',');
        append_model_json(&b, s, "deepseek-v4-pro");
    }
    buf_puts(&b, "]}\n");
    bool ok = http_response(fd, s->enable_cors, 200, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static void client_done(server *s) {
    pthread_mutex_lock(&s->mu);
    if (s->clients > 0) s->clients--;
    pthread_cond_broadcast(&s->clients_cv);
    pthread_mutex_unlock(&s->mu);
}

static void set_client_socket_nonblocking(int fd);

static bool client_poll_revents_disconnected(short revents) {
    return (revents & (POLLERR | POLLHUP | POLLNVAL)) != 0;
}

static bool client_recv_errno_disconnected(int err) {
    return err != EINTR && err != EAGAIN && err != EWOULDBLOCK;
}

/* The request body has already been consumed and this one-request server sends
 * Connection: close, so EOF is cancellation. Discard unsupported pipelined
 * bytes nonblockingly: otherwise they can hide the FIN behind readable data. */
static bool client_socket_disconnected(int fd) {
    struct pollfd pfd = {.fd = fd, .events = POLLIN};
    int rc;
    do {
        rc = poll(&pfd, 1, 0);
    } while (rc < 0 && errno == EINTR);
    if (rc < 0) return client_recv_errno_disconnected(errno);
    if (rc == 0) return false;
    if (client_poll_revents_disconnected(pfd.revents)) return true;
    if (!(pfd.revents & POLLIN)) return false;

    char discard[256];
    for (;;) {
        ssize_t n = recv(fd, discard, sizeof(discard), 0);
        if (n > 0) continue;
        if (n == 0) return true;
        if (errno == EINTR) continue;
        return client_recv_errno_disconnected(errno);
    }
}

/* Mark first, then detach only work that no worker owns yet. No job mutex is
 * held while entering either server scheduler mutex. */
static void server_cancel_job(server *s, job *j) {
    job_mark_cancelled(j);

    bool detached = false;
    pthread_mutex_lock(&s->mu);
    job *prev = NULL;
    for (job *it = s->head; it; prev = it, it = it->next) {
        if (it != j) continue;
        if (prev) prev->next = it->next;
        else s->head = it->next;
        if (s->tail == it) s->tail = prev;
        it->next = NULL;
        detached = true;
        break;
    }
    if (!detached && s->batched_mode) {
        for (int i = 0; i < s->slot_count; i++) {
            server_slot *slot = &s->slots[i];
            if (slot->assigned != j) continue;
            slot->assigned = NULL;
            slot->busy = false;
            detached = true;
            dispatch_jobs_locked(s);
            break;
        }
    }
    pthread_cond_broadcast(&s->cv);
    pthread_mutex_unlock(&s->mu);

    pthread_mutex_lock(&s->model_mu);
    for (int i = 0; i < s->slot_count; i++) {
        server_slot *slot = &s->slots[i];
        if (slot->running == j) {
            (void)server_cancel_pending_decode_locked(s, slot);
            break;
        }
    }
    pthread_cond_broadcast(&s->model_cv);
    pthread_mutex_unlock(&s->model_mu);

    if (detached) job_complete(j);
}

static void wait_for_job_or_disconnect(server *s, job *j) {
    pthread_mutex_lock(&j->mu);
    while (!j->done) {
        struct timespec deadline;
        clock_gettime(CLOCK_REALTIME, &deadline);
        timespec_add_us(&deadline, 100000);
        (void)pthread_cond_timedwait(&j->cv, &j->mu, &deadline);
        bool done = j->done;
        pthread_mutex_unlock(&j->mu);
        if (!done && client_socket_disconnected(j->fd)) {
            server_cancel_job(s, j);
        }
        pthread_mutex_lock(&j->mu);
    }
    pthread_mutex_unlock(&j->mu);
}

static void *client_main(void *arg) {
    client_arg *ca = arg;
    server *s = ca->srv;
    int fd = ca->fd;
    free(ca);

    http_request hr = {0};
    if (!read_http_request(fd, &hr)) {
        http_error(fd, s->enable_cors, 400, "bad HTTP request");
        goto done;
    }

    if (!strcmp(hr.method, "OPTIONS")) {
        http_response(fd, s->enable_cors, 204, NULL, "");
        http_request_free(&hr);
        goto done;
    }

    if (!strcmp(hr.method, "GET") && !strcmp(hr.path, "/v1/models")) {
        send_models(s, fd);
        http_request_free(&hr);
        goto done;
    }
    const char *model_path_prefix = "/v1/models/";
    const size_t model_path_prefix_len = strlen(model_path_prefix);
    if (!strcmp(hr.method, "GET") &&
        !strncmp(hr.path, model_path_prefix, model_path_prefix_len) &&
        server_model_alias_known(hr.path + model_path_prefix_len))
    {
        send_model(s, fd, hr.path + model_path_prefix_len);
        http_request_free(&hr);
        goto done;
    }

    request req;
    char err[160];
    bool ok = false;
    const int ctx_size = s->ctx_size;
    if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/messages")) {
        ok = parse_anthropic_request(s->engine, s, hr.body, s->default_tokens,
                                     ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/chat/completions")) {
        ok = parse_chat_request(s->engine, s, hr.body, s->default_tokens,
                                ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/responses")) {
        ok = parse_responses_request(s->engine, s, hr.body, s->default_tokens,
                                     ctx_size, &req, err, sizeof(err));
    } else if (!strcmp(hr.method, "POST") && !strcmp(hr.path, "/v1/completions")) {
        ok = parse_completion_request(s->engine, hr.body, s->default_tokens,
                                      ctx_size, &req, err, sizeof(err));
    } else {
        http_error(fd, s->enable_cors, 404, "unknown endpoint");
        http_request_free(&hr);
        goto done;
    }
    if (ok) req.raw_body = xstrndup(hr.body, hr.body_len);
    http_request_free(&hr);
    if (!ok) {
        http_error(fd, s->enable_cors, 400, err);
        goto done;
    }
    if (!req.model_from_request) {
        free(req.model);
        req.model = xstrdup(server_model_id_from_engine(s->engine));
    }
    if (request_exceeds_context(&req, ctx_size)) {
        http_error_context_length_exceeded(fd, s->enable_cors, &req, req.prompt.len, ctx_size);
        request_free(&req);
        goto done;
    }

    set_client_socket_nonblocking(fd);
    job j;
    memset(&j, 0, sizeof(j));
    j.fd = fd;
    j.req = req;
    pthread_mutex_init(&j.mu, NULL);
    pthread_cond_init(&j.cv, NULL);

    if (!enqueue(s, &j)) {
        http_error(fd, s->enable_cors, 503, "server shutting down");
        pthread_cond_destroy(&j.cv);
        pthread_mutex_destroy(&j.mu);
        request_free(&j.req);
        goto done;
    }
    wait_for_job_or_disconnect(s, &j);

    pthread_cond_destroy(&j.cv);
    pthread_mutex_destroy(&j.mu);
    request_free(&j.req);
done:
    close(fd);
    client_done(s);
    return NULL;
}

static int listen_on(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    if (!strcmp(host, "localhost")) host = "127.0.0.1";
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) {
        close(fd);
        errno = EINVAL;
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 128) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void configure_client_socket(int fd) {
    struct timeval tv;
    tv.tv_sec = DS4_SERVER_IO_TIMEOUT_SEC;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

static void set_client_socket_nonblocking(int fd) {
    /* The inference worker writes streaming responses itself.  Once a request is
     * queued, a blocked socket would block every other request too, so slow
     * clients are failed instead of back-pressuring the model session. */
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

typedef struct {
    ds4_engine_options engine;
    const char *gpu_vram_arg;
    const char *gpu_devices_arg;
    const char *host;
    int port;
    int ctx_size;
    int default_tokens;
    const char *chdir_path;
    const char *trace_path;
    const char *kv_disk_dir;
    uint64_t kv_disk_space_mb;
    kv_cache_options kv_cache;
    bool kv_cache_reject_different_quant;
    bool disable_exact_dsml_tool_replay;
    int tool_memory_max_ids;
    bool enable_cors;
    int batched_sessions;
    int mixed_prefill_quantum;
} server_config;

static int parse_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v <= 0 || v > INT_MAX) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonneg_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v < 0 || v > INT_MAX) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return (int)v;
}

static float parse_float_arg(const char *s, const char *opt, float minv, float maxv) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (!s[0] || *end || v < minv || v > maxv) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: invalid value for %s: %s", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: missing value for %s", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static void log_context_memory(ds4_backend backend, int ctx_size,
                               uint32_t prefill_chunk,
                               bool ssd_streaming,
                               int session_count) {
    ds4_context_memory m =
        ds4_context_memory_estimate_with_prefill_mode(backend,
                                                      ctx_size,
                                                      prefill_chunk,
                                                      ssd_streaming);
    server_log(DS4_LOG_DEFAULT,
               "ds4-server: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)",
               (double)m.total_bytes / (1024.0 * 1024.0),
               ctx_size,
               ds4_backend_name(backend),
               m.prefill_cap,
               m.raw_cap,
               m.comp_cap);
    if (session_count > 1) {
        server_log(DS4_LOG_DEFAULT,
                   "ds4-server: %d resident sessions request at least %.2f GiB of context buffers",
                   session_count,
                   (double)m.total_bytes * (double)session_count /
                       (1024.0 * 1024.0 * 1024.0));
    }
}
static void server_close_resources(server *s) {
    if (s->trace) {
        fclose(s->trace);
        s->trace = NULL;
    }
    kv_cache_close(&s->kv);
    tool_memory_free(&s->tool_mem);
    for (int i = 0; i < s->slot_count; i++) {
        server_slot *slot = &s->slots[i];
        live_tool_state_free(&slot->responses_live);
        live_tool_state_free(&slot->anthropic_live);
        visible_live_free(&slot->thinking_live);
        if (slot->session) ds4_session_free(slot->session);
    }
    free(s->slot_threads);
    free(s->slots);
    pthread_mutex_destroy(&s->tool_mu);
    pthread_mutex_destroy(&s->kv_mu);
    pthread_mutex_destroy(&s->inference_mu);
    pthread_mutex_destroy(&s->model_mu);
    pthread_mutex_destroy(&s->trace_mu);
    pthread_cond_destroy(&s->model_cv);
    pthread_cond_destroy(&s->clients_cv);
    pthread_cond_destroy(&s->cv);
    pthread_mutex_destroy(&s->mu);
    ds4_engine_close(s->engine);
    memset(s, 0, sizeof(*s));
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_SERVER, topic);
}

static ds4_backend parse_backend_arg(const char *s, const char *arg) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    server_log(DS4_LOG_DEFAULT, "ds4-server: invalid %s value: %s", arg, s);
#ifdef DS4_ROCM_BUILD
    server_log(DS4_LOG_DEFAULT, "ds4-server: valid server backends are: metal, rocm, cpu");
#else
    server_log(DS4_LOG_DEFAULT, "ds4-server: valid server backends are: metal, cuda, cpu");
#endif
    exit(2);
}

static ds4_backend default_server_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static server_config parse_options(int argc, char **argv) {
    server_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_server_backend(),
            .mtp_draft_tokens = 1,
            .mtp_margin = 3.0f,
        },
        .host = "127.0.0.1",
        .port = 8000,
        .ctx_size = 32768,
        .default_tokens = 393216,
        .tool_memory_max_ids = DS4_TOOL_MEMORY_DEFAULT_MAX_IDS,
        .mixed_prefill_quantum = 128,
    };
    c.kv_cache = kv_cache_default_options();

    bool directional_steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg,
                                   &i,
                                   argc,
                                   argv,
                                   &c.engine.distributed,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            server_log(DS4_LOG_DEFAULT,
                       "ds4-server: %s",
                       dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.engine.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.engine.mtp_draft_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-margin")) {
            c.engine.mtp_margin = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1000.0f);
        } else if (!strcmp(arg, "--glm-mtp")) {
            c.engine.glm_mtp = true;
        } else if (!strcmp(arg, "--glm-mtp-timing")) {
            c.engine.glm_mtp = true;
            c.engine.glm_mtp_timing = true;
        } else if (!strcmp(arg, "--dspark")) {
            c.engine.dspark = true;
        } else if (!strcmp(arg, "--dspark-confidence")) {
            c.engine.dspark = true;
            c.engine.dspark_confidence_threshold =
                parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.engine.dspark_confidence_threshold_set = true;
        } else if (!strcmp(arg, "--dspark-strict")) {
            c.engine.dspark = true;
            c.engine.dspark_strict = true;
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.ctx_size = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.default_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--chdir")) {
            c.chdir_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--host")) {
            c.host = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--port")) {
            c.port = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--cors")) {
            c.enable_cors = true;
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--batched-session")) {
            c.batched_sessions = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mixed-prefill-quantum")) {
            c.mixed_prefill_quantum =
                parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-disk-dir")) {
            c.kv_disk_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--kv-disk-space-mb")) {
            c.kv_disk_space_mb = (uint64_t)parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-min-tokens")) {
            c.kv_cache.min_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-cold-max-tokens")) {
            c.kv_cache.cold_max_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-continued-interval-tokens")) {
            c.kv_cache.continued_interval_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-boundary-trim-tokens")) {
            c.kv_cache.boundary_trim_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-boundary-align-tokens")) {
            c.kv_cache.boundary_align_tokens = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--kv-cache-reject-different-quant")) {
            c.kv_cache_reject_different_quant = true;
        } else if (!strcmp(arg, "--disable-exact-dsml-tool-replay")) {
            c.disable_exact_dsml_tool_replay = true;
        } else if (!strcmp(arg, "--tool-memory-max-ids")) {
            c.tool_memory_max_ids = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.engine.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.engine.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                server_log(DS4_LOG_DEFAULT,
                           "ds4-server: --ssd-streaming-cache-experts must be a positive count or <number>GB");
                exit(2);
            }
            c.engine.ssd_streaming_cache_experts = experts;
            c.engine.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            int v = parse_nonneg_int_arg(need_arg(&i, argc, argv, arg), arg);
            c.engine.ssd_streaming_full_layers = (uint32_t)v;
            c.engine.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                server_log(DS4_LOG_DEFAULT,
                           "ds4-server: --ssd-streaming-preload-experts must be positive");
                exit(2);
            }
            c.engine.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.engine.simulate_used_memory_bytes)) {
                server_log(DS4_LOG_DEFAULT,
                           "ds4-server: --simulate-used-memory must be a positive GiB value, e.g. 64GB");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            int v = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                server_log(DS4_LOG_DEFAULT,
                           "ds4-server: --prefill-chunk must be positive");
                exit(2);
            }
            c.engine.prefill_chunk = (uint32_t)v;
        } else if (!strcmp(arg, "--power")) {
            c.engine.power_percent = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (c.engine.power_percent < 1 || c.engine.power_percent > 100) {
                server_log(DS4_LOG_DEFAULT, "ds4-server: --power must be between 1 and 100");
                exit(2);
            }
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_arg(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            directional_steering_scale_set = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.engine.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--gpu-vram")) {
            c.gpu_vram_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gpu-devices")) {
            c.gpu_devices_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--cuda-tensor-parallel")) {
            c.engine.cuda_tensor_parallel = true;
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else {
            server_log(DS4_LOG_DEFAULT, "ds4-server: unknown option: %s", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }
    if (c.kv_cache.cold_max_tokens > 0 &&
        c.kv_cache.cold_max_tokens < c.kv_cache.min_tokens)
    {
        server_log(DS4_LOG_DEFAULT,
                   "ds4-server: --kv-cache-cold-max-tokens must be 0 or >= --kv-cache-min-tokens");
        exit(2);
    }
    if (c.engine.directional_steering_file && !directional_steering_scale_set) {
        c.engine.directional_steering_ffn = 1.0f;
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.engine.distributed,
                                        &c.engine,
                                        dist_err,
                                        sizeof(dist_err)) != 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: %s", dist_err);
        exit(2);
    }
    return c;
}

