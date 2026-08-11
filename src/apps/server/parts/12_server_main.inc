static void server_request_worker_stop(server *s) {
    pthread_mutex_lock(&s->mu);
    s->stopping = true;
    dispatch_jobs_locked(s);
    pthread_cond_broadcast(&s->cv);
    pthread_mutex_unlock(&s->mu);
}

static void server_request_decode_stop(server *s) {
    if (!s->batched_mode) return;
    pthread_mutex_lock(&s->model_mu);
    s->model_stopping = true;
    pthread_cond_broadcast(&s->model_cv);
    pthread_mutex_unlock(&s->model_mu);
}

int main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = stop_signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    server_config cfg = parse_options(argc, argv);
    if (cfg.chdir_path && chdir(cfg.chdir_path) != 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to chdir to %s: %s",
                   cfg.chdir_path, strerror(errno));
        return 1;
    }

    cfg.engine.context_size = cfg.ctx_size;
    cfg.engine.placement_ctx_hint = cfg.ctx_size;
    cfg.engine.placement_session_count_hint =
        cfg.batched_sessions > 0 ? cfg.batched_sessions : 1;
    cfg.engine.share_session_prefill_workspace = cfg.batched_sessions > 0;
    ds4_engine *engine = NULL;
    if (cfg.gpu_vram_arg || cfg.gpu_devices_arg) {
        ds4_gpu_config gpu_cfg = {0};
        bool skip_cuda = false;
        char gpu_err[256];
        if (parse_gpu_vram_arg(cfg.gpu_vram_arg, cfg.gpu_devices_arg,
                               &gpu_cfg, &skip_cuda,
                               gpu_err, sizeof(gpu_err)) != 0) {
            fprintf(stderr, "ds4-server: %s\n", gpu_err);
            return 2;
        }
        cfg.engine.backend = skip_cuda ? DS4_BACKEND_CPU : DS4_BACKEND_CUDA;
        if (skip_cuda) {
            if (ds4_engine_open(&engine, &cfg.engine) != 0) return 1;
        } else {
            const bool was_auto =
                (cfg.gpu_vram_arg && !strcmp(cfg.gpu_vram_arg, "auto")) ||
                (!cfg.gpu_vram_arg && cfg.gpu_devices_arg);
            char layout[256];
            if (format_gpu_layout_line(&gpu_cfg, was_auto,
                                       layout, sizeof(layout)) > 0) {
                fprintf(stdout, "%s\n", layout);
                fflush(stdout);
            }
            if (ds4_engine_create_with_gpu_config(
                    &engine, &cfg.engine, &gpu_cfg) != 0) return 1;
        }
    } else if (ds4_engine_open(&engine, &cfg.engine) != 0) {
        return 1;
    }

    if (cfg.engine.distributed.role == DS4_DISTRIBUTED_WORKER) {
        ds4_dist_generation_options gen = {
            .ctx_size = cfg.ctx_size,
        };
        int rc = ds4_dist_run(engine, &cfg.engine.distributed, &gen);
        ds4_engine_close(engine);
        return rc;
    }

    const int slot_count = cfg.batched_sessions > 0 ? cfg.batched_sessions : 1;
    log_context_memory(cfg.engine.backend,
                       cfg.ctx_size,
                       ds4_engine_prefill_chunk(engine),
                       cfg.engine.ssd_streaming,
                       slot_count);

    server s = {0};
    s.engine = engine;
    s.ctx_size = cfg.ctx_size;
    s.slot_count = slot_count;
    s.batched_mode = cfg.batched_sessions > 0;
    s.mixed_prefill_quantum = cfg.mixed_prefill_quantum;
    s.last_prefill_slot = slot_count - 1;
    s.default_tokens = cfg.default_tokens;
    s.disable_exact_dsml_tool_replay = cfg.disable_exact_dsml_tool_replay;
    s.tool_mem.max_entries = cfg.tool_memory_max_ids;
    s.enable_cors = cfg.enable_cors;
    s.slots = xmalloc((size_t)slot_count * sizeof(*s.slots));
    memset(s.slots, 0, (size_t)slot_count * sizeof(*s.slots));
    if (s.batched_mode) {
        s.slot_threads = xmalloc((size_t)slot_count * sizeof(*s.slot_threads));
        memset(s.slot_threads, 0, (size_t)slot_count * sizeof(*s.slot_threads));
    }

    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cv, NULL);
    pthread_cond_init(&s.clients_cv, NULL);
    pthread_mutex_init(&s.tool_mu, NULL);
    pthread_mutex_init(&s.kv_mu, NULL);
    pthread_mutexattr_t inference_attr;
    pthread_mutexattr_init(&inference_attr);
    pthread_mutexattr_settype(&inference_attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&s.inference_mu, &inference_attr);
    pthread_mutexattr_destroy(&inference_attr);
    pthread_mutex_init(&s.model_mu, NULL);
    pthread_cond_init(&s.model_cv, NULL);
    pthread_mutex_init(&s.trace_mu, NULL);

    for (int i = 0; i < slot_count; i++) {
        server_slot *slot = &s.slots[i];
        slot->srv = &s;
        slot->id = i;
        if (ds4_session_create(&slot->session, engine, cfg.ctx_size) != 0) {
            server_log(DS4_LOG_DEFAULT,
                       "ds4-server: failed to create %s session %d/%d",
                       ds4_backend_name(cfg.engine.backend), i + 1, slot_count);
            server_close_resources(&s);
            return 1;
        }
    }

    if (cfg.kv_disk_dir) {
        kv_cache_open(&s.kv, cfg.kv_disk_dir, cfg.kv_disk_space_mb,
                      cfg.kv_cache_reject_different_quant, cfg.kv_cache);
    }
    if (s.disable_exact_dsml_tool_replay) {
        server_log(DS4_LOG_DEFAULT,
                   "ds4-server: exact DSML tool replay disabled; tool history uses canonical JSON rendering");
    }
    if (s.batched_mode) {
        server_log(DS4_LOG_DEFAULT,
                   "ds4-server: batched mode enabled resident_sessions=%d prefill_quantum=%d mixed_prefill_quantum=%d decode_coalesce_us=%ld",
                   s.slot_count,
                   server_prefill_quantum_for(&s, false),
                   server_prefill_quantum_for(&s, true),
                   server_decode_coalesce_us());
        if (ds4_engine_has_mtp(engine)) {
            server_log(DS4_LOG_DEFAULT,
                       "ds4-server: MTP speculative decoding is disabled while native session batching is active");
        }
    }
    if (cfg.trace_path) {
        s.trace = fopen(cfg.trace_path, "w");
        if (!s.trace) {
            server_log(DS4_LOG_DEFAULT, "ds4-server: failed to open trace file %s: %s",
                       cfg.trace_path, strerror(errno));
            server_close_resources(&s);
            return 1;
        }
        setvbuf(s.trace, NULL, _IONBF, 0);
        server_log(DS4_LOG_DEFAULT, "ds4-server: tracing session to %s", cfg.trace_path);
    }

    pthread_t worker = (pthread_t){0};
    int slot_threads_started = 0;
    bool decode_thread_started = false;
    if (s.batched_mode) {
        if (pthread_create(&s.decode_thread, NULL, decode_worker_main, &s) != 0) {
            server_log(DS4_LOG_DEFAULT, "ds4-server: failed to start decode coordinator");
            server_close_resources(&s);
            return 1;
        }
        decode_thread_started = true;
        for (int i = 0; i < s.slot_count; i++) {
            if (pthread_create(&s.slot_threads[i], NULL, slot_worker_main,
                               &s.slots[i]) != 0) {
                server_log(DS4_LOG_DEFAULT,
                           "ds4-server: failed to start session worker %d/%d",
                           i + 1, s.slot_count);
                server_request_worker_stop(&s);
                for (int j = 0; j < slot_threads_started; j++) {
                    pthread_join(s.slot_threads[j], NULL);
                }
                server_request_decode_stop(&s);
                pthread_join(s.decode_thread, NULL);
                server_close_resources(&s);
                return 1;
            }
            slot_threads_started++;
        }
    } else if (pthread_create(&worker, NULL, worker_main, &s) != 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to start worker");
        server_close_resources(&s);
        return 1;
    }

    int lfd = listen_on(cfg.host, cfg.port);
    if (lfd < 0) {
        server_log(DS4_LOG_DEFAULT, "ds4-server: failed to listen on %s:%d: %s", cfg.host, cfg.port, strerror(errno));
        server_request_worker_stop(&s);
        if (s.batched_mode) {
            for (int i = 0; i < slot_threads_started; i++) {
                pthread_join(s.slot_threads[i], NULL);
            }
            server_request_decode_stop(&s);
            if (decode_thread_started) pthread_join(s.decode_thread, NULL);
        } else {
            pthread_join(worker, NULL);
        }
        server_close_resources(&s);
        return 1;
    }
    g_listen_fd = lfd;
    server_log(DS4_LOG_DEFAULT, "ds4-server: listening on http://%s:%d", cfg.host, cfg.port);

    while (!g_stop_requested) {
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) {
            if (g_stop_requested) break;
            if (errno == EINTR) continue;
            server_log(DS4_LOG_DEFAULT, "ds4-server: accept failed: %s", strerror(errno));
            continue;
        }
        if (g_stop_requested) {
            close(fd);
            break;
        }

        configure_client_socket(fd);
        client_arg *ca = xmalloc(sizeof(*ca));
        ca->srv = &s;
        ca->fd = fd;
        pthread_mutex_lock(&s.mu);
        s.clients++;
        pthread_mutex_unlock(&s.mu);
        pthread_t th;
        if (pthread_create(&th, NULL, client_main, ca) != 0) {
            pthread_mutex_lock(&s.mu);
            s.clients--;
            pthread_cond_broadcast(&s.clients_cv);
            pthread_mutex_unlock(&s.mu);
            free(ca);
            close(fd);
            continue;
        }
        pthread_detach(th);
    }
    if (g_listen_fd >= 0) {
        close(lfd);
        g_listen_fd = -1;
    }

    server_log(DS4_LOG_DEFAULT, "ds4-server: shutdown requested, draining requests");
    server_request_worker_stop(&s);
    if (s.batched_mode) {
        for (int i = 0; i < slot_threads_started; i++) {
            pthread_join(s.slot_threads[i], NULL);
        }
        server_request_decode_stop(&s);
        if (decode_thread_started) pthread_join(s.decode_thread, NULL);
    } else {
        pthread_join(worker, NULL);
    }
    pthread_mutex_lock(&s.mu);
    while (s.clients > 0) pthread_cond_wait(&s.clients_cv, &s.mu);
    pthread_mutex_unlock(&s.mu);

    for (int i = 0; s.kv.enabled && i < s.slot_count; i++) {
        server_slot *slot = &s.slots[i];
        const ds4_tokens *tokens = ds4_session_tokens(slot->session);
        if (!tokens || tokens->len < s.kv.opt.min_tokens) continue;
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: persisting resident KV cache before shutdown slot=%d tokens=%d",
                   i, tokens->len);
        kv_cache_store_current(&s, slot, "shutdown");
    }
    server_close_resources(&s);
    return 0;
}
