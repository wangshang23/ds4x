#ifndef DS4_NO_GPU
static void metal_graph_stream_prefill_selected_profile_summary(
        const ds4_gpu_graph *g) {
    if (!metal_graph_stream_prefill_selected_profile_enabled(g)) return;
    metal_graph_selected_profile_summary_impl(
            g,
            "Metal streaming prefill selected profile");
}

static bool metal_graph_stream_pagein_touch_range(
        const ds4_model *model,
        uint64_t         offset,
        uint64_t         size,
        uint64_t        *touched,
        uint8_t         *sink) {
    if (!model ||
        !model->map ||
        model->size == 0 ||
        offset > model->size ||
        size == 0 ||
        size > model->size - offset) {
        return false;
    }

    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t page_offset = offset & ~(page - 1u);
    const uint64_t leading = offset - page_offset;
    if (size > UINT64_MAX - leading ||
        leading + size > UINT64_MAX - (page - 1u)) {
        return false;
    }
    uint64_t touch_bytes = align_up(leading + size, page);
    if (touch_bytes > model->size - page_offset) {
        touch_bytes = model->size - page_offset;
    }
    if (touch_bytes == 0 || touch_bytes > (uint64_t)SIZE_MAX) {
        return false;
    }

    const uint8_t *base = (const uint8_t *)model->map;
    const volatile uint8_t *p =
        (const volatile uint8_t *)(base + page_offset);

#if defined(POSIX_MADV_WILLNEED)
    (void)posix_madvise((void *)(base + page_offset),
                        (size_t)touch_bytes,
                        POSIX_MADV_WILLNEED);
#endif

    uint8_t s = sink ? *sink : 0;
    for (uint64_t off = 0; off < touch_bytes; off += page) {
        s ^= p[off];
    }
    s ^= p[touch_bytes - 1u];
    if (sink) *sink = s;
    if (touched) *touched += touch_bytes;
    return true;
}

static bool metal_graph_stream_pread_range(
        const ds4_model *model,
        uint64_t         offset,
        uint64_t         size,
        uint64_t        *read_bytes,
        uint8_t         *sink) {
    if (!model ||
        model->fd < 0 ||
        offset > model->size ||
        size == 0 ||
        size > model->size - offset) {
        return false;
    }
    if (offset > (uint64_t)LLONG_MAX) return false;

    const size_t chunk = 1024u * 1024u;
    uint8_t *buf = xmalloc(chunk);
    uint64_t pos = offset;
    uint64_t rem = size;
    uint8_t s = sink ? *sink : 0;
    bool ok = true;
    while (rem != 0) {
        const size_t want = rem > (uint64_t)chunk ? chunk : (size_t)rem;
        ssize_t nread;
        do {
            nread = pread(model->fd, buf, want, (off_t)pos);
        } while (nread < 0 && errno == EINTR);
        if (nread <= 0) {
            ok = false;
            break;
        }
        s ^= buf[0];
        s ^= buf[(size_t)nread - 1u];
        pos += (uint64_t)nread;
        rem -= (uint64_t)nread;
        if (read_bytes) {
            *read_bytes = *read_bytes > UINT64_MAX - (uint64_t)nread ?
                UINT64_MAX : *read_bytes + (uint64_t)nread;
        }
    }
    if (sink) *sink = s;
    free(buf);
    return ok;
}

static bool metal_graph_stream_prepare_range(
        const metal_graph_stream_pagein_job *job,
        uint64_t                             offset,
        uint64_t                             size,
        uint64_t                            *touched,
        uint8_t                             *sink) {
    if (!job) return false;
    if (job->pread_only) {
        return metal_graph_stream_pread_range(job->model,
                                              offset,
                                              size,
                                              touched,
                                              sink);
    }
    if (job->readahead_only) {
        metal_graph_stream_readahead_range_impl(job->model,
                                                offset,
                                                size,
                                                true);
        if (touched) {
            *touched = *touched > UINT64_MAX - size ?
                UINT64_MAX : *touched + size;
        }
        return true;
    }
    if (job->madvise_only) {
        return metal_graph_stream_madvise_willneed_range_impl(job->model,
                                                              offset,
                                                              size,
                                                              true,
                                                              touched);
    }
    return metal_graph_stream_pagein_touch_range(job->model,
                                                offset,
                                                size,
                                                touched,
                                                sink);
}

static void *metal_graph_stream_pagein_thread_main(void *arg) {
    metal_graph_stream_pagein_job *job = arg;
    const double t0 = job->profile ? now_sec() : 0.0;
    job->ok = true;
    for (uint32_t i = 0; i < job->n_ranges; i++) {
        const bool ok = metal_graph_stream_prepare_range(job,
                                                         job->ranges[i].off,
                                                         job->ranges[i].size,
                                                         &job->touched,
                                                         &job->sink);
        if (!ok) {
            job->ok = false;
            break;
        }
    }
    if (job->profile) {
        job->thread_ms = (now_sec() - t0) * 1000.0;
    }
    return NULL;
}

static void *metal_graph_stream_pagein_worker_main(void *arg) {
    metal_graph_stream_pagein_worker *worker = arg;
    metal_graph_stream_pagein_job *job = worker ? worker->job : NULL;
    const double t0 = job && job->profile ? now_sec() : 0.0;
    worker->ok = true;
    if (!job || worker->stride == 0) {
        worker->ok = false;
        return NULL;
    }
    for (uint32_t i = worker->first; i < job->n_ranges; i += worker->stride) {
        const bool ok = metal_graph_stream_prepare_range(job,
                                                         job->ranges[i].off,
                                                         job->ranges[i].size,
                                                         &worker->touched,
                                                         &worker->sink);
        if (!ok) {
            worker->ok = false;
            break;
        }
    }
    if (job->profile) {
        worker->thread_ms = (now_sec() - t0) * 1000.0;
    }
    return NULL;
}

static uint32_t metal_graph_stream_prefill_layer_pagein_threads(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_LAYER_PREPARE_THREADS",
            "DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_THREADS");
    if (!env || !env[0]) {
        env = glm_graph_env_value(
                "DS4_ROCM_STREAMING_PREFILL_LAYER_PAGEIN_THREADS",
                "DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_THREADS");
    }
    if (!env || !env[0]) return 8;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || v == 0) return 1;
    return v > 16 ? 16u : (uint32_t)v;
}

static uint32_t metal_graph_stream_prefill_selected_prepare_threads(
        bool madvise_only) {
    if (!madvise_only) return 1;
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_SELECTED_PREPARE_THREADS",
            "DS4_METAL_STREAMING_PREFILL_SELECTED_PREPARE_THREADS");
    if (!env || !env[0]) {
        env = glm_graph_env_value(
                "DS4_ROCM_STREAMING_PREFILL_SELECTED_MADVISE_THREADS",
                "DS4_METAL_STREAMING_PREFILL_SELECTED_MADVISE_THREADS");
    }
    if (!env || !env[0]) return metal_graph_stream_prefill_layer_pagein_threads();
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || v == 0) return 1;
    return v > 16 ? 16u : (uint32_t)v;
}

static uint32_t metal_graph_stream_prefill_selected_prepare_gap(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_SELECTED_PREPARE_GAP",
            "DS4_METAL_STREAMING_PREFILL_SELECTED_PREPARE_GAP");
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    return v > 8 ? 8u : (uint32_t)v;
}

static bool metal_graph_stream_prefill_layer_pagein_overlap_enabled(void) {
    return !glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_LAYER_PREPARE_NO_OVERLAP",
                                  "DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_NO_OVERLAP") &&
           !glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_LAYER_PAGEIN_NO_OVERLAP",
                                  "DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_NO_OVERLAP") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE_OVERLAP",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE_OVERLAP") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN_OVERLAP",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN_OVERLAP");
}

enum { DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD = 4 };

static uint32_t metal_graph_stream_prefill_layer_prepare_ahead(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_LAYER_PREPARE_AHEAD",
            "DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_AHEAD");
    if (!env || !env[0]) return 1;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || v == 0) return 1;
    if (v > DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD) {
        return DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD;
    }
    return (uint32_t)v;
}

static bool metal_graph_stream_prefill_selected_pagein_start(
        ds4_gpu_graph                 *g,
        const ds4_model               *model,
        const ds4_layer_weights       *layer,
        uint32_t                       il,
        uint32_t                       n_tokens,
        uint64_t                       gate_expert_bytes,
        uint64_t                       down_expert_bytes,
        metal_graph_stream_pagein_job *job) {
    if (!job) return false;
    memset(job, 0, sizeof(*job));
    job->ok = true;
    job->profile =
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_SELECTED_PAGEIN_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_SELECTED_PAGEIN_PROFILE") ||
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_SELECTED_MADVISE_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_SELECTED_MADVISE_PROFILE");
    job->layer = il;
    job->n_tokens = n_tokens;

    const bool madvise_only =
        metal_graph_stream_prefill_selected_madvise_enabled(g);
    job->madvise_only = madvise_only;
    if (!metal_graph_stream_prefill_selected_pagein_enabled(g) &&
        !madvise_only) return true;
    if (!model || !layer || !metal_graph_batch_router_selected(g) || n_tokens == 0) {
        return false;
    }

    const uint64_t n_ids = (uint64_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_ids > SIZE_MAX / sizeof(int32_t)) return false;
    int32_t *selected = xmalloc((size_t)n_ids * sizeof(selected[0]));

    const double t_read0 = job->profile ? now_sec() : 0.0;
    bool ok = ds4_gpu_tensor_read(metal_graph_batch_router_selected(g),
                                  0,
                                  selected,
                                  n_ids * sizeof(selected[0])) != 0;
    if (job->profile) {
        job->read_ms = (now_sec() - t_read0) * 1000.0;
    }

    bool seen[DS4_MAX_EXPERT] = { false };
    if (ok) {
        for (uint64_t i = 0; i < n_ids; i++) {
            const int32_t expert = selected[i];
            if (expert < 0 || (uint32_t)expert >= DS4_N_EXPERT) {
                fprintf(stderr,
                        "ds4: Metal streaming prefill selected page-in expert id %d is outside 0..%u at layer %u\n",
                        expert,
                        (uint32_t)DS4_N_EXPERT,
                        il);
                ok = false;
                break;
            }
            if (seen[expert]) continue;
            seen[expert] = true;
            job->unique++;
        }
    }
    free(selected);

    metal_graph_stream_pagein_range *ranges = NULL;
    uint32_t n_ranges = 0;
    if (ok && job->unique != 0) {
        ranges = xmalloc((size_t)DS4_N_EXPERT * 3u * sizeof(ranges[0]));
        const uint32_t gap = madvise_only ?
            metal_graph_stream_prefill_selected_prepare_gap() : 0;
        uint32_t e = 0;
        while (e < DS4_N_EXPERT) {
            while (e < DS4_N_EXPERT && !seen[e]) e++;
            if (e >= DS4_N_EXPERT) break;
            const uint32_t first = e;
            uint32_t last = e;
            uint32_t skipped = 0;
            e++;
            while (e < DS4_N_EXPERT) {
                if (seen[e]) {
                    last = e;
                    skipped = 0;
                } else if (skipped < gap) {
                    skipped++;
                } else {
                    break;
                }
                e++;
            }

            const uint64_t first_id = first;
            const uint64_t n_experts = (uint64_t)last - (uint64_t)first + 1u;
            if (first_id > UINT64_MAX / gate_expert_bytes ||
                first_id > UINT64_MAX / down_expert_bytes ||
                n_experts > UINT64_MAX / gate_expert_bytes ||
                n_experts > UINT64_MAX / down_expert_bytes) {
                fprintf(stderr, "ds4: Metal streaming prefill selected page-in offset overflow\n");
                ok = false;
                break;
            }
            const uint64_t gate_rel = first_id * gate_expert_bytes;
            const uint64_t down_rel = first_id * down_expert_bytes;
            const uint64_t gate_bytes = n_experts * gate_expert_bytes;
            const uint64_t down_bytes = n_experts * down_expert_bytes;
            if (gate_rel > UINT64_MAX - layer->ffn_gate_exps->abs_offset ||
                gate_rel > UINT64_MAX - layer->ffn_up_exps->abs_offset ||
                down_rel > UINT64_MAX - layer->ffn_down_exps->abs_offset) {
                fprintf(stderr, "ds4: Metal streaming prefill selected page-in offset overflow\n");
                ok = false;
                break;
            }
            ranges[n_ranges++] = (metal_graph_stream_pagein_range){
                layer->ffn_gate_exps->abs_offset + gate_rel,
                gate_bytes,
            };
            ranges[n_ranges++] = (metal_graph_stream_pagein_range){
                layer->ffn_up_exps->abs_offset + gate_rel,
                gate_bytes,
            };
            ranges[n_ranges++] = (metal_graph_stream_pagein_range){
                layer->ffn_down_exps->abs_offset + down_rel,
                down_bytes,
            };
            uint64_t run_bytes = UINT64_MAX;
            if (gate_bytes <= (UINT64_MAX - down_bytes) / 2ull) {
                run_bytes = gate_bytes * 2ull + down_bytes;
            }
            if (run_bytes == UINT64_MAX ||
                job->bytes > UINT64_MAX - run_bytes) {
                job->bytes = UINT64_MAX;
            } else {
                job->bytes += run_bytes;
            }
        }
    }

    if (!ok || n_ranges == 0) {
        free(ranges);
        return ok;
    }

    job->model = model;
    job->ranges = ranges;
    job->n_ranges = n_ranges;
    job->n_threads = metal_graph_stream_prefill_selected_prepare_threads(madvise_only);
    if (job->n_threads <= 1) {
        const int rc = pthread_create(&job->thread,
                                      NULL,
                                      metal_graph_stream_pagein_thread_main,
                                      job);
        if (rc != 0) {
            fprintf(stderr,
                    "ds4: Metal streaming prefill selected page-in thread failed: %s\n",
                    strerror(rc));
            free(ranges);
            memset(job, 0, sizeof(*job));
            return false;
        }
    } else {
        job->threads = xcalloc(job->n_threads, sizeof(job->threads[0]));
        job->workers = xcalloc(job->n_threads, sizeof(job->workers[0]));
        for (uint32_t t = 0; t < job->n_threads; t++) {
            job->workers[t].job = job;
            job->workers[t].first = t;
            job->workers[t].stride = job->n_threads;
            const int rc = pthread_create(&job->threads[t],
                                          NULL,
                                          metal_graph_stream_pagein_worker_main,
                                          &job->workers[t]);
            if (rc != 0) {
                fprintf(stderr,
                        "ds4: Metal streaming prefill selected page-in worker failed: %s\n",
                        strerror(rc));
                for (uint32_t j = 0; j < t; j++) {
                    (void)pthread_join(job->threads[j], NULL);
                }
                free(job->workers);
                free(job->threads);
                free(ranges);
                memset(job, 0, sizeof(*job));
                return false;
            }
        }
    }
    job->started = true;
    return true;
}

static bool metal_graph_stream_prefill_selected_pagein_join(
        metal_graph_stream_pagein_job *job) {
    if (!job || !job->started) return true;
    const double t0 = job->profile ? now_sec() : 0.0;
    int rc = 0;
    bool ok = true;
    if (job->n_threads <= 1) {
        rc = pthread_join(job->thread, NULL);
        ok = rc == 0 && job->ok;
    } else {
        job->touched = 0;
        job->thread_ms = 0.0;
        job->sink = 0;
        for (uint32_t t = 0; t < job->n_threads; t++) {
            const int trc = pthread_join(job->threads[t], NULL);
            if (trc != 0 && rc == 0) rc = trc;
            if (trc != 0 || !job->workers[t].ok) ok = false;
            if (job->touched > UINT64_MAX - job->workers[t].touched) {
                job->touched = UINT64_MAX;
            } else {
                job->touched += job->workers[t].touched;
            }
            if (job->workers[t].thread_ms > job->thread_ms) {
                job->thread_ms = job->workers[t].thread_ms;
            }
            job->sink ^= job->workers[t].sink;
        }
    }
    const double wait_ms = job->profile ? (now_sec() - t0) * 1000.0 : 0.0;
    if (job->profile) {
        const char *kind = job->madvise_only ? "madvise" : "page-in";
        const char *bytes_label = job->madvise_only ? "advised" : "touched";
        fprintf(stderr,
                "ds4: Metal streaming prefill selected %s layer=%u "
                "tokens=%u unique=%u ranges=%u bytes=%.2f GiB "
                "read=%.3f ms wait=%.3f ms thread=%.3f ms %s=%.2f GiB ok=%d\n",
                kind,
                job->layer,
                job->n_tokens,
                job->unique,
                job->n_ranges,
                (double)job->bytes / (1024.0 * 1024.0 * 1024.0),
                job->read_ms,
                wait_ms,
                job->thread_ms,
                bytes_label,
                (double)job->touched / (1024.0 * 1024.0 * 1024.0),
                ok ? 1 : 0);
    }
    if (rc != 0) {
        fprintf(stderr,
                "ds4: Metal streaming prefill selected page-in join failed: %s\n",
                strerror(rc));
    }
    free(job->workers);
    free(job->threads);
    free(job->ranges);
    memset(job, 0, sizeof(*job));
    return ok;
}

static bool metal_graph_stream_prefill_layer_pagein_start(
        const ds4_gpu_graph           *g,
        const ds4_model               *model,
        const ds4_weights             *weights,
        uint32_t                       il,
        uint32_t                       n_tokens,
        bool                           madvise_only,
        bool                           pread_only,
        bool                           readahead_only,
        bool                           decode_only,
        metal_graph_stream_pagein_job *job) {
    if (!job) return false;
    memset(job, 0, sizeof(*job));
    job->ok = true;
    job->madvise_only = madvise_only;
    job->pread_only = pread_only;
    job->readahead_only = readahead_only;
    job->profile =
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_LAYER_PAGEIN_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_PROFILE") ||
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_LAYER_PREAD_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE") ||
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_LAYER_MADVISE_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_LAYER_MADVISE_PROFILE") ||
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_LAYER_READAHEAD_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_LAYER_READAHEAD_PROFILE");
    job->layer = il;
    job->n_tokens = n_tokens;

    if (pread_only) {
        if (g) {
            if (!metal_graph_stream_prefill_layer_pread_enabled(g)) return true;
        } else if (glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREAD",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREAD") ||
                   glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE")) {
            return true;
        }
    } else if (readahead_only) {
        if (g) {
            if (!metal_graph_stream_prefill_layer_readahead_enabled(g)) return true;
        } else if (!glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD",
                                          "DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD") ||
                   glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD") ||
                   glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE")) {
            return true;
        }
    } else if (madvise_only) {
        if (g) {
            if (!metal_graph_stream_prefill_layer_madvise_enabled(g)) return true;
        } else if (glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE") ||
                   glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_MADVISE",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_MADVISE")) {
            return true;
        }
    } else {
        if (g) {
            if (!metal_graph_stream_prefill_layer_pagein_enabled(g)) return true;
        } else if (!glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN",
                                          "DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN") ||
                   glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN",
                                         "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN")) {
            return true;
        }
    }
    if (!model || !weights || il >= DS4_N_LAYER) return false;

    const uint32_t n_threads =
        metal_graph_stream_prefill_layer_pagein_threads();
    ds4_model_map_span_vec spans;
    const bool spans_ok = decode_only ?
        weights_model_map_decode_layer_spans(weights, il, &spans) :
        weights_model_map_spans(weights, il, il, false, &spans);
    if (!spans_ok) return false;
    metal_graph_stream_pagein_range *ranges =
        xmalloc((size_t)spans.len * n_threads * sizeof(ranges[0]));
    uint32_t n_ranges = 0;
    const uint64_t page = (uint64_t)getpagesize();
    for (uint32_t i = 0; i < spans.len; i++) {
        const uint64_t size = spans.v[i].end - spans.v[i].off;
        uint64_t consumed = 0;
        uint64_t chunk = size / n_threads;
        if (chunk > page) chunk = (chunk / page) * page;
        if (chunk == 0) chunk = size;
        for (uint32_t t = 0; t < n_threads && consumed < size; t++) {
            uint64_t this_size =
                (t + 1u == n_threads || size - consumed <= chunk) ?
                size - consumed : chunk;
            ranges[n_ranges++] = (metal_graph_stream_pagein_range){
                spans.v[i].off + consumed,
                this_size,
            };
            consumed += this_size;
        }
        if (job->bytes > UINT64_MAX - size) {
            job->bytes = UINT64_MAX;
        } else {
            job->bytes += size;
        }
    }
    job->unique = spans.len;
    free(spans.v);

    job->model = model;
    job->ranges = ranges;
    job->n_ranges = n_ranges;
    job->n_threads = n_threads;
    if (n_threads == 1) {
        const int rc = pthread_create(&job->thread,
                                      NULL,
                                      metal_graph_stream_pagein_thread_main,
                                      job);
        if (rc != 0) {
            fprintf(stderr,
                    "ds4: Metal streaming prefill layer page-in thread failed: %s\n",
                    strerror(rc));
            free(ranges);
            memset(job, 0, sizeof(*job));
            return false;
        }
    } else {
        job->threads = xcalloc(n_threads, sizeof(job->threads[0]));
        job->workers = xcalloc(n_threads, sizeof(job->workers[0]));
        for (uint32_t t = 0; t < n_threads; t++) {
            job->workers[t].job = job;
            job->workers[t].first = t;
            job->workers[t].stride = n_threads;
            const int rc = pthread_create(&job->threads[t],
                                          NULL,
                                          metal_graph_stream_pagein_worker_main,
                                          &job->workers[t]);
            if (rc != 0) {
                fprintf(stderr,
                        "ds4: Metal streaming prefill layer page-in worker failed: %s\n",
                        strerror(rc));
                for (uint32_t j = 0; j < t; j++) {
                    (void)pthread_join(job->threads[j], NULL);
                }
                free(job->workers);
                free(job->threads);
                free(ranges);
                memset(job, 0, sizeof(*job));
                return false;
            }
        }
    }
    job->started = true;
    return true;
}

static bool metal_graph_stream_prefill_layer_pagein_join(
        metal_graph_stream_pagein_job *job) {
    if (!job || !job->started) return true;
    const double t0 = job->profile ? now_sec() : 0.0;
    int rc = 0;
    bool ok = true;
    if (job->n_threads <= 1) {
        rc = pthread_join(job->thread, NULL);
        ok = rc == 0 && job->ok;
    } else {
        job->touched = 0;
        job->thread_ms = 0.0;
        job->sink = 0;
        for (uint32_t t = 0; t < job->n_threads; t++) {
            const int trc = pthread_join(job->threads[t], NULL);
            if (trc != 0 && rc == 0) rc = trc;
            if (trc != 0 || !job->workers[t].ok) ok = false;
            if (job->touched > UINT64_MAX - job->workers[t].touched) {
                job->touched = UINT64_MAX;
            } else {
                job->touched += job->workers[t].touched;
            }
            if (job->workers[t].thread_ms > job->thread_ms) {
                job->thread_ms = job->workers[t].thread_ms;
            }
            job->sink ^= job->workers[t].sink;
        }
    }
    const double wait_ms = job->profile ? (now_sec() - t0) * 1000.0 : 0.0;
    if (job->profile) {
        const char *kind = job->pread_only ? "pread" :
                           job->readahead_only ? "readahead" :
                           job->madvise_only ? "madvise" : "page-in";
        const char *bytes_label = job->pread_only ? "read" :
                                  job->readahead_only ? "requested" :
                                  job->madvise_only ? "advised" : "touched";
        fprintf(stderr,
                "ds4: Metal streaming prefill layer %s layer=%u "
                "tokens=%u threads=%u ranges=%u bytes=%.2f GiB wait=%.3f ms "
                "thread=%.3f ms %s=%.2f GiB ok=%d\n",
                kind,
                job->layer,
                job->n_tokens,
                job->n_threads ? job->n_threads : 1u,
                job->n_ranges,
                (double)job->bytes / (1024.0 * 1024.0 * 1024.0),
                wait_ms,
                job->thread_ms,
                bytes_label,
                (double)job->touched / (1024.0 * 1024.0 * 1024.0),
                ok ? 1 : 0);
    }
    if (rc != 0) {
        fprintf(stderr,
                "ds4: Metal streaming prefill layer page-in join failed: %s\n",
                strerror(rc));
    }
    free(job->workers);
    free(job->threads);
    free(job->ranges);
    memset(job, 0, sizeof(*job));
    return ok;
}

typedef struct {
    metal_graph_stream_pagein_job job;
    uint32_t layer;
    bool active;
} metal_graph_stream_prepare_slot;

static metal_graph_stream_prepare_slot *metal_graph_stream_prepare_slot_find(
        metal_graph_stream_prepare_slot *slots,
        uint32_t                         n_slots,
        uint32_t                         layer) {
    for (uint32_t i = 0; i < n_slots; i++) {
        if (slots[i].active && slots[i].layer == layer) return &slots[i];
    }
    return NULL;
}

static metal_graph_stream_prepare_slot *metal_graph_stream_prepare_slot_free(
        metal_graph_stream_prepare_slot *slots,
        uint32_t                         n_slots) {
    for (uint32_t i = 0; i < n_slots; i++) {
        if (!slots[i].active) return &slots[i];
    }
    return NULL;
}

static bool metal_graph_stream_prepare_start_if_needed(
        const ds4_gpu_graph          *g,
        const ds4_model              *model,
        const ds4_weights            *weights,
        uint32_t                      layer,
        uint32_t                      n_tokens,
        bool                          madvise_only,
        bool                          pread_only,
        bool                          readahead_only,
        bool                          decode_only,
        metal_graph_stream_prepare_slot *slots,
        uint32_t                      n_slots) {
    if (layer >= DS4_N_LAYER) return true;
    if (metal_graph_stream_prepare_slot_find(slots, n_slots, layer)) {
        return true;
    }
    metal_graph_stream_prepare_slot *slot =
        metal_graph_stream_prepare_slot_free(slots, n_slots);
    if (!slot) {
        fprintf(stderr,
                "ds4: Metal streaming prefill prepare queue is full before layer %u\n",
                layer);
        return false;
    }
    memset(slot, 0, sizeof(*slot));
    slot->layer = layer;
    if (!metal_graph_stream_prefill_layer_pagein_start(g,
                                                       model,
                                                       weights,
                                                       layer,
                                                       n_tokens,
                                                       madvise_only,
                                                       pread_only,
                                                       readahead_only,
                                                       decode_only,
                                                       &slot->job)) {
        memset(slot, 0, sizeof(*slot));
        return false;
    }
    slot->active = slot->job.started;
    return true;
}

static bool metal_graph_stream_prepare_join_layer(
        const ds4_gpu_graph          *g,
        const ds4_model              *model,
        const ds4_weights            *weights,
        uint32_t                      layer,
        uint32_t                      n_tokens,
        bool                          madvise_only,
        bool                          pread_only,
        bool                          readahead_only,
        bool                          decode_only,
        metal_graph_stream_prepare_slot *slots,
        uint32_t                      n_slots) {
    metal_graph_stream_prepare_slot *slot =
        metal_graph_stream_prepare_slot_find(slots, n_slots, layer);
    if (!slot) {
        metal_graph_stream_pagein_job job;
        memset(&job, 0, sizeof(job));
        if (!metal_graph_stream_prefill_layer_pagein_start(g,
                                                           model,
                                                           weights,
                                                           layer,
                                                           n_tokens,
                                                           madvise_only,
                                                           pread_only,
                                                           readahead_only,
                                                           decode_only,
                                                           &job)) {
            return false;
        }
        return metal_graph_stream_prefill_layer_pagein_join(&job);
    }
    const bool ok = metal_graph_stream_prefill_layer_pagein_join(&slot->job);
    memset(slot, 0, sizeof(*slot));
    return ok;
}

static bool metal_graph_stream_prepare_join_all(
        metal_graph_stream_prepare_slot *slots,
        uint32_t                         n_slots) {
    bool ok = true;
    for (uint32_t i = 0; i < n_slots; i++) {
        if (!slots[i].active) continue;
        if (!metal_graph_stream_prefill_layer_pagein_join(&slots[i].job)) {
            ok = false;
        }
        memset(&slots[i], 0, sizeof(slots[i]));
    }
    return ok;
}

static void metal_graph_stream_readahead_layer(
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_spans(weights, il, il, false, &spans)) return;
    metal_graph_stream_readahead_spans(model, &spans);
    free(spans.v);
}

static void metal_graph_stream_readahead_layer_decode(
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_decode_layer_spans(weights, il, &spans)) return;
    metal_graph_stream_readahead_spans(model, &spans);
    free(spans.v);
}

static void metal_graph_stream_readahead_output(
        const ds4_model   *model,
        const ds4_weights *weights) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_output_spans(weights, &spans)) return;
    metal_graph_stream_readahead_spans(model, &spans);
    free(spans.v);
}

static bool metal_graph_stream_prefill_selected_readahead_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           (glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD",
                                  "DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD") ||
            glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED",
                                  "DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED")) &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD");
}

static bool metal_graph_stream_prefill_selected_readahead_shared_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED",
                                 "DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD_SHARED") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_READAHEAD");
}

static uint32_t metal_graph_stream_prefill_selected_readahead_gap(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_SELECTED_READAHEAD_GAP",
            "DS4_METAL_STREAMING_PREFILL_SELECTED_READAHEAD_GAP");
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    return v > 8 ? 8u : (uint32_t)v;
}

static bool metal_graph_stream_readahead_selected_run(
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        uint32_t                 first,
        uint32_t                 last,
        uint64_t                 gate_expert_bytes,
        uint64_t                 down_expert_bytes,
        uint64_t                *hint_bytes) {
    if (!model || !layer || first > last || last >= DS4_N_EXPERT) return false;

    const uint64_t first_id = first;
    const uint64_t n_experts = (uint64_t)last - (uint64_t)first + 1u;
    if (first_id > UINT64_MAX / gate_expert_bytes ||
        first_id > UINT64_MAX / down_expert_bytes ||
        n_experts > UINT64_MAX / gate_expert_bytes ||
        n_experts > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr, "ds4: Metal streaming prefill selected expert readahead overflow\n");
        return false;
    }

    const uint64_t gate_rel = first_id * gate_expert_bytes;
    const uint64_t down_rel = first_id * down_expert_bytes;
    const uint64_t gate_bytes = n_experts * gate_expert_bytes;
    const uint64_t down_bytes = n_experts * down_expert_bytes;
    if (gate_rel > UINT64_MAX - layer->ffn_gate_exps->abs_offset ||
        gate_rel > UINT64_MAX - layer->ffn_up_exps->abs_offset ||
        down_rel > UINT64_MAX - layer->ffn_down_exps->abs_offset) {
        fprintf(stderr, "ds4: Metal streaming prefill selected expert readahead overflow\n");
        return false;
    }

    metal_graph_stream_readahead_range_impl(model,
                                            layer->ffn_gate_exps->abs_offset + gate_rel,
                                            gate_bytes,
                                            true);
    metal_graph_stream_readahead_range_impl(model,
                                            layer->ffn_up_exps->abs_offset + gate_rel,
                                            gate_bytes,
                                            true);
    metal_graph_stream_readahead_range_impl(model,
                                            layer->ffn_down_exps->abs_offset + down_rel,
                                            down_bytes,
                                            true);
    if (hint_bytes) {
        if (*hint_bytes > UINT64_MAX - gate_bytes ||
            *hint_bytes + gate_bytes > UINT64_MAX - gate_bytes ||
            *hint_bytes + gate_bytes * 2u > UINT64_MAX - down_bytes) {
            *hint_bytes = UINT64_MAX;
        } else {
            *hint_bytes += gate_bytes * 2u + down_bytes;
        }
    }
    return true;
}

static bool metal_graph_stream_readahead_selected_experts_from_gpu(
        ds4_gpu_graph           *g,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 n_tokens,
        uint64_t                 gate_expert_bytes,
        uint64_t                 down_expert_bytes) {
    if (!metal_graph_stream_prefill_selected_readahead_enabled(g)) return true;
    if (!model || !layer || !g || !metal_graph_batch_router_selected(g) || n_tokens == 0) {
        return false;
    }
    if (sizeof(int) != sizeof(int32_t) || DS4_N_EXPERT > DS4_MAX_EXPERT) {
        return false;
    }

    const uint64_t n_ids = (uint64_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_ids > SIZE_MAX / sizeof(int32_t)) return false;
    int32_t *selected = xmalloc((size_t)n_ids * sizeof(selected[0]));

    const bool profile =
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_SELECTED_READAHEAD_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_SELECTED_READAHEAD_PROFILE");
    const double t0 = profile ? now_sec() : 0.0;
    bool ok = ds4_gpu_tensor_read(metal_graph_batch_router_selected(g),
                                  0,
                                  selected,
                                  n_ids * sizeof(selected[0])) != 0;
    bool seen[DS4_MAX_EXPERT] = { false };
    uint32_t unique = 0;
    uint32_t ranges = 0;
    uint64_t hint_bytes = 0;
    const uint32_t gap = metal_graph_stream_prefill_selected_readahead_gap();
    if (ok) {
        for (uint64_t i = 0; i < n_ids; i++) {
            const int32_t expert = selected[i];
            if (expert < 0 || (uint32_t)expert >= DS4_N_EXPERT) {
                fprintf(stderr,
                        "ds4: Metal streaming prefill selected expert id %d is outside 0..%u at layer %u\n",
                        expert,
                        (uint32_t)DS4_N_EXPERT,
                        il);
                ok = false;
                break;
            }
            if (seen[expert]) continue;
            seen[expert] = true;
            unique++;
        }
    }

    if (ok) {
        uint32_t e = 0;
        while (e < DS4_N_EXPERT) {
            while (e < DS4_N_EXPERT && !seen[e]) e++;
            if (e >= DS4_N_EXPERT) break;
            const uint32_t first = e;
            uint32_t last = e;
            uint32_t skipped = 0;
            e++;
            while (e < DS4_N_EXPERT) {
                if (seen[e]) {
                    last = e;
                    skipped = 0;
                } else if (skipped < gap) {
                    skipped++;
                } else {
                    break;
                }
                e++;
            }

            if (!metal_graph_stream_readahead_selected_run(model,
                                                           layer,
                                                           first,
                                                           last,
                                                           gate_expert_bytes,
                                                           down_expert_bytes,
                                                           &hint_bytes)) {
                ok = false;
                break;
            }
            ranges++;
        }
    }
    if (profile) {
        fprintf(stderr,
                "ds4: Metal streaming prefill selected readahead layer=%u "
                "tokens=%u unique=%u ranges=%u gap=%u hint=%.2f GiB time=%.3f ms\n",
                il,
                n_tokens,
                unique,
                ranges,
                gap,
                (double)hint_bytes / (1024.0 * 1024.0 * 1024.0),
                (now_sec() - t0) * 1000.0);
    }
    free(selected);
    return ok;
}

static bool metal_graph_stream_map_token(
        const ds4_model   *model,
        const ds4_weights *weights) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_token_spans(weights, &spans)) {
        fprintf(stderr, "ds4: Metal SSD streaming could not build token embedding span\n");
        return false;
    }
    const bool ok = metal_graph_install_model_spans(model, &spans, "token embedding");
    free(spans.v);
    return ok;
}

static bool metal_graph_stream_map_decode_static_all(
        const ds4_model   *model,
        const ds4_weights *weights) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_decode_static_spans(weights, true, true, &spans)) {
        fprintf(stderr, "ds4: Metal SSD streaming could not build static decode spans\n");
        return false;
    }
    const bool ok = metal_graph_install_model_spans(model, &spans, "static decode");
    free(spans.v);
    return ok;
}

static bool metal_graph_stream_map_layer(
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_spans(weights, il, il, false, &spans)) {
        fprintf(stderr, "ds4: Metal SSD streaming could not build layer %u spans\n", il);
        return false;
    }
    const bool ok = metal_graph_install_model_spans(model, &spans, "layer");
    free(spans.v);
    return ok;
}

static bool metal_graph_stream_map_layer_decode(
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_decode_layer_spans(weights, il, &spans)) {
        fprintf(stderr, "ds4: Metal SSD streaming could not build decode layer %u spans\n", il);
        return false;
    }
    const bool ok = metal_graph_install_model_spans(model, &spans, "decode layer");
    free(spans.v);
    return ok;
}

static bool metal_graph_stream_map_output(
        const ds4_model   *model,
        const ds4_weights *weights) {
    ds4_model_map_span_vec spans;
    if (!weights_model_map_output_spans(weights, &spans)) {
        fprintf(stderr, "ds4: Metal SSD streaming could not build output head spans\n");
        return false;
    }
    const bool ok = metal_graph_install_model_spans(model, &spans, "output head");
    free(spans.v);
    return ok;
}

static uint32_t metal_graph_raw_span_for_batch(
        const ds4_gpu_graph *g,
        uint32_t               pos0,
        uint32_t               n_tokens) {
    if (!g || g->raw_cap == 0 || n_tokens == 0) return 0;

    const uint32_t window = g->raw_window ? g->raw_window : DS4_N_SWA;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    uint64_t needed = (uint64_t)n_tokens;
    if (window != 0) {
        needed += n_tokens == 1 ? (uint64_t)window - 1u : (uint64_t)window;
    }
    uint64_t available = (uint64_t)last_pos + 1u;
    if (needed > available) needed = available;
    if (needed > g->raw_cap) needed = g->raw_cap;
    return (uint32_t)needed;
}

static uint32_t metal_graph_raw_start_for_span(
        const ds4_gpu_graph *g,
        uint32_t               last_pos,
        uint32_t               n_raw) {
    if (!g || g->raw_cap == 0 || n_raw == 0) return 0;
    const uint32_t first_raw_pos = last_pos + 1u - n_raw;
    return first_raw_pos % g->raw_cap;
}

static uint32_t metal_graph_decode_raw_score_count(
        const ds4_gpu_graph *g,
        uint32_t               pos,
        uint32_t               n_raw,
        uint32_t               ratio) {
    if (!g || n_raw == 0) return 0;
    if (ratio == 0) return n_raw > 256u ? 256u : n_raw;

    const uint32_t first_raw_pos = pos + 1u - n_raw;
    const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
    uint32_t lo = first_raw_pos;
    const uint32_t window = g->raw_window ? g->raw_window : DS4_N_SWA;
    if (window != 0 && pos + 1u > window) {
        const uint32_t wlo = pos + 1u - window;
        if (wlo > lo) lo = wlo;
    }
    const uint32_t hi = pos < raw_last_pos ? pos : raw_last_pos;
    if (hi < lo) return 0;
    uint32_t raw_count = hi - lo + 1u;
    if (raw_count > 256u) raw_count = 256u;
    return raw_count;
}

static bool metal_graph_cuda_splitkv_score_may_engage(
        const ds4_gpu_graph *g,
        uint32_t               pos) {
    if (!g) return false;

    const uint32_t min_score = metal_graph_cuda_greedy_splitkv_min_score();
    const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        uint32_t visible_comp = 0;
        const uint32_t n_comp = g->layer_n_comp[il];
        if (n_comp != 0) {
            visible_comp = ratio == 0 ? n_comp : (pos + 1u) / ratio;
            if (visible_comp > n_comp) visible_comp = n_comp;
        }
        const uint32_t raw_count =
            metal_graph_decode_raw_score_count(g, pos, n_raw, ratio);
        const uint32_t n_score = raw_count + visible_comp;
        if (n_score > 1u && n_score >= min_score) return true;
    }
    return false;
}

static bool metal_graph_cuda_greedy_splitkv_may_engage(
        const ds4_gpu_graph *g,
        uint32_t               pos) {
    if (!metal_graph_cuda_greedy_splitkv_requested()) return false;
    return metal_graph_cuda_splitkv_score_may_engage(g, pos);
}

/* Capture an intermediate verifier prefix.
 *
 * Speculation is only profitable if partial accepts are cheap. For compressed
 * attention the mutable frontier is just the small compressor state plus append
 * counters, so retain each intermediate state while the tiny verifier already
 * steps the compressor token by token.
 *
 * Raw SWA rows are not captured here.  This graph uses a raw ring larger than
 * the 128-token logical SWA window, so writing speculative future rows does
 * not evict visible raw rows.  If the raw cache is ever reduced to a strict
 * 128-row ring, speculative raw rows must become shadow rows and be copied
 * into the ring only on commit. */
static bool metal_graph_capture_prefix_attn_state(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       slot) {
    if (!g->spec_capture_prefixes) return true;
    /* Leading layers without an attention compressor (ratio 0) carry no
     * frontier; mirror spec_frontier_snapshot/commit which skip them. */
    if (ds4_layer_compress_ratio(il) == 0) return true;
    if (slot >= DS4_SPEC_PREFIX_SLOTS ||
        !g->spec_prefix1_attn_state_kv[il] ||
        !g->spec_prefix1_attn_state_score[il]) {
        return false;
    }
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
    const uint64_t offset = (uint64_t)slot * bytes;
    g->spec_prefix_n_comp[slot][il] = g->layer_n_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_kv[il], offset,
                                 g->layer_attn_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_score[il], offset,
                                 g->layer_attn_state_score[il], 0, bytes) != 0;
}

static bool metal_graph_capture_prefix_index_state(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       slot) {
    if (!g->spec_capture_prefixes) return true;
    /* Index-compressor frontiers exist only on ratio-4 layers;
     * mirror spec_frontier_snapshot/commit which skip the rest. */
    if (ds4_layer_compress_ratio(il) != 4) return true;
    if (slot >= DS4_SPEC_PREFIX_SLOTS ||
        !g->spec_prefix1_index_state_kv[il] ||
        !g->spec_prefix1_index_state_score[il]) {
        return false;
    }
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
    const uint64_t offset = (uint64_t)slot * bytes;
    g->spec_prefix_n_index_comp[slot][il] = g->layer_n_index_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_index_state_kv[il], offset,
                                 g->layer_index_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_index_state_score[il], offset,
                                 g->layer_index_state_score[il], 0, bytes) != 0;
}

static uint32_t metal_graph_decode_indexer_sparse_threshold(const ds4_gpu_graph *g) {
    (void)g;
    static int parsed = -1;
    static uint32_t cached = 0;
    if (parsed < 0) {
        parsed = 0;
#ifndef DS4_ROCM_BUILD
        const char *env = getenv("DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD");
        if (env && env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(env, &end, 10);
            while (end && isspace((unsigned char)*end)) end++;
            if (end != env && end && *end == '\0' &&
                (v == 64ul || v == 128ul || v == 256ul || v == 512ul ||
                 v == 1024ul || v == 2048ul || v == 4096ul)) {
                cached = (uint32_t)v;
                parsed = 1;
            } else {
                fprintf(stderr,
                        "ds4: invalid DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD=%s; "
                        "expected 64, 128, 256, 512, 1024, 2048, or 4096\n",
                        env);
            }
        }
#endif
    }
    if (parsed > 0) return cached;

    /* Keep dense attention longer than the legacy 512-row window by default.
     * Around the 2K frontier the sparse path's score/top-k setup dominates
     * the smaller attention scan, while larger contexts benefit from sparse
     * indexed attention.  This threshold changes only the implementation used
     * to consume the compressed rows; it must not lower the 512-row indexer
     * selection defined by DS4_N_INDEXER_TOP_K. */
    return 1024u;
}

#endif /* !DS4_NO_GPU */
