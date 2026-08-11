/* =========================================================================
 * Coordinator Worker Registry And Route Planning
 * =========================================================================
 *
 * A route is a contiguous chain that starts after the coordinator's local
 * slice. The last hop can either return logits directly or return the final
 * hidden state so the coordinator can run its local output head.
 */

static void dist_coordinator_add_worker(
        ds4_dist_coordinator_state *state,
        int fd,
        const char *peer_host,
        const char *peer_port,
        const ds4_dist_hello_fixed *hello,
        const char *model_name) {
    ds4_dist_worker_entry *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        DIST_COORD_DEBUG(state, "ds4: distributed coordinator: out of memory while registering worker\n");
        return;
    }

    entry->fd = fd;
    snprintf(entry->peer_host, sizeof(entry->peer_host), "%s", peer_host);
    snprintf(entry->peer_port, sizeof(entry->peer_port), "%s", peer_port);
    snprintf(entry->model_name, sizeof(entry->model_name), "%s", model_name ? model_name : "unknown");
    entry->model_id = hello->model_id;
    entry->quant_bits = hello->quant_bits;
    entry->layer_start = hello->layer_start;
    entry->layer_end = hello->layer_end;
    entry->has_output = hello->has_output;
    entry->has_hidden = hello->has_hidden;
    entry->ctx_size = hello->ctx_size;
    entry->n_layers = hello->n_layers;
    entry->listen_port = hello->listen_port;

    pthread_mutex_lock(&state->mu);
    if (state->shutting_down) {
        pthread_mutex_unlock(&state->mu);
        free(entry);
        return;
    }
    ds4_dist_worker_entry **link = &state->workers;
    while (*link) {
        ds4_dist_worker_entry *old = *link;
        if (strcmp(old->peer_host, peer_host) == 0 &&
            old->model_id == hello->model_id &&
            old->layer_start == hello->layer_start &&
            old->layer_end == hello->layer_end &&
            old->has_output == hello->has_output)
        {
            *link = old->next;
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: dropped stale worker %s:%s layers=%u:%u%s\n",
                             old->peer_host,
                             old->peer_port,
                             old->layer_start,
                             old->layer_end,
                             old->has_output ? "+output" : "");
            free(old);
            continue;
        }
        link = &old->next;
    }
    entry->next = state->workers;
    state->workers = entry;
    state->generation++;
    pthread_mutex_unlock(&state->mu);

    char layer_end[32];
    if (entry->has_output) snprintf(layer_end, sizeof(layer_end), "output");
    else snprintf(layer_end, sizeof(layer_end), "%u", entry->layer_end);
    DIST_COORD_DEBUG(state,
                     "ds4: distributed coordinator: registered worker %s:%s data_port=%u model_id=%u quant=Q%u layers=%u:%s hidden=%u ctx=%u\n",
                     entry->peer_host,
                     entry->peer_port,
                     entry->listen_port,
                     entry->model_id,
                     entry->quant_bits,
                     entry->layer_start,
                     layer_end,
                     entry->has_hidden,
                     entry->ctx_size);
    if (dist_coordinator_debug_enabled(state)) dist_coordinator_report_plan(state);
}

static int dist_worker_route_cmp(const void *a, const void *b) {
    const ds4_dist_worker_entry *ea = *(const ds4_dist_worker_entry * const *)a;
    const ds4_dist_worker_entry *eb = *(const ds4_dist_worker_entry * const *)b;
    if (ea->layer_start < eb->layer_start) return -1;
    if (ea->layer_start > eb->layer_start) return 1;
    if (ea->has_output != eb->has_output) return ea->has_output ? -1 : 1;
    if (ea->layer_end > eb->layer_end) return -1;
    if (ea->layer_end < eb->layer_end) return 1;
    return 0;
}

static bool dist_worker_route_candidate_ok(
        const ds4_dist_coordinator_state *state,
        const ds4_dist_worker_entry *w,
        uint32_t last) {
    const bool needs_hidden = w->layer_end < last || !w->has_output;
    if (needs_hidden && !w->has_hidden) return false;
    if (w->layer_end >= last && !w->has_output && !state->local_can_output_head) return false;
    return true;
}

static bool dist_route_search_workers(
        const ds4_dist_coordinator_state *state,
        ds4_dist_worker_entry **workers,
        uint32_t n,
        uint32_t next,
        uint32_t last,
        ds4_dist_worker_entry **path,
        uint32_t *path_len,
        uint32_t *missing_layer) {
    bool saw_start = false;
    for (uint32_t i = 0; i < n; i++) {
        ds4_dist_worker_entry *w = workers[i];
        if (w->layer_start < next) continue;
        if (w->layer_start > next) break;
        saw_start = true;
        if (!dist_worker_route_candidate_ok(state, w, last)) continue;

        path[(*path_len)++] = w;
        if (w->layer_end >= last) return true;
        uint32_t child_missing = w->layer_end + 1u;
        if (dist_route_search_workers(state,
                                      workers,
                                      n,
                                      child_missing,
                                      last,
                                      path,
                                      path_len,
                                      &child_missing)) {
            return true;
        }
        if (child_missing > *missing_layer) *missing_layer = child_missing;
        (*path_len)--;
    }
    if (!saw_start && next > *missing_layer) *missing_layer = next;
    return false;
}

static void dist_coordinator_report_plan(ds4_dist_coordinator_state *state) {
    if (!dist_coordinator_debug_enabled(state)) return;
    pthread_mutex_lock(&state->mu);
    uint32_t n = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) n++;
    ds4_dist_worker_entry **workers = n ? calloc(n, sizeof(workers[0])) : NULL;
    ds4_dist_worker_entry **path = n ? calloc(n, sizeof(path[0])) : NULL;
    if ((n && !workers) || (n && !path)) {
        free(workers);
        free(path);
        pthread_mutex_unlock(&state->mu);
        fprintf(stderr, "ds4: distributed coordinator: out of memory building route plan\n");
        return;
    }
    uint32_t i = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) workers[i++] = it;
    qsort(workers, n, sizeof(workers[0]), dist_worker_route_cmp);

    const uint32_t last = state->n_layers - 1u;
    bool complete = state->local_start == 0;
    bool has_output = state->local_end == last &&
                      (state->local_has_output || state->local_can_output_head);
    uint32_t next = state->local_end + 1u;
    if (state->local_end >= last) next = state->n_layers;
    uint32_t path_len = 0;
    uint32_t missing = next;
    if (complete && !has_output) {
        complete = dist_route_search_workers(state,
                                             workers,
                                             n,
                                             next,
                                             last,
                                             path,
                                             &path_len,
                                             &missing);
        if (complete && path_len != 0) {
            ds4_dist_worker_entry *final = path[path_len - 1u];
            has_output = final->has_output || state->local_can_output_head;
            next = state->n_layers;
        }
    }

    char plan[1024];
    size_t used = 0;
    char local_end[32];
    if (state->local_has_output) snprintf(local_end, sizeof(local_end), "output");
    else snprintf(local_end, sizeof(local_end), "%u", state->local_end);
    used += (size_t)snprintf(plan + used, used < sizeof(plan) ? sizeof(plan) - used : 0,
                             "local %u:%s",
                             state->local_start,
                             local_end);
    for (i = 0; i < path_len; i++) {
        ds4_dist_worker_entry *w = path[i];
        if (used < sizeof(plan)) {
            char end[32];
            if (w->has_output) snprintf(end, sizeof(end), "output");
            else snprintf(end, sizeof(end), "%u", w->layer_end);
            used += (size_t)snprintf(plan + used, sizeof(plan) - used,
                                     " -> %s:%u Q%u %u:%s",
                                     w->peer_host,
                                     w->listen_port,
                                     w->quant_bits,
                                     w->layer_start,
                                     end);
        }
    }
    if (complete && path_len != 0 &&
        !path[path_len - 1u]->has_output && state->local_can_output_head &&
        used < sizeof(plan)) {
        used += (size_t)snprintf(plan + used, sizeof(plan) - used,
                                 " -> local output");
    }
    if (complete && path_len == 0 &&
        state->local_end == last && !state->local_has_output &&
        state->local_can_output_head && used < sizeof(plan)) {
        used += (size_t)snprintf(plan + used, sizeof(plan) - used,
                                 " -> local output");
    }
    complete = complete && has_output && next == state->n_layers;
    pthread_mutex_unlock(&state->mu);

    if (complete) {
        fprintf(stderr, "ds4: distributed coordinator: complete route ready: %s\n", plan);
    } else {
        fprintf(stderr, "ds4: distributed coordinator: route incomplete; next needed layer %u\n", missing);
    }
    free(path);
    free(workers);
}

static void dist_route_plan_free(ds4_dist_route_plan *plan) {
    if (!plan) return;
    for (uint32_t i = 0; i < plan->count; i++) {
        if (plan->entry[i].fd >= 0) close(plan->entry[i].fd);
    }
    free(plan->entry);
    free(plan->blob);
    memset(plan, 0, sizeof(*plan));
}

static bool dist_route_entry_matches_worker(
        const ds4_dist_route_entry *route,
        const ds4_dist_worker_entry *worker) {
    const bool route_has_output = (route->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0;
    return route->port == worker->listen_port &&
           strcmp(route->host, worker->peer_host) == 0 &&
           route->layer_start == worker->layer_start &&
           route->layer_end == worker->layer_end &&
           route_has_output == (worker->has_output != 0);
}

static void dist_coordinator_forget_route_workers(
        ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan) {
    bool removed_any = false;
    pthread_mutex_lock(&state->mu);
    for (uint32_t i = 0; i < plan->count; i++) {
        ds4_dist_worker_entry **link = &state->workers;
        while (*link) {
            ds4_dist_worker_entry *entry = *link;
            if (!dist_route_entry_matches_worker(&plan->entry[i], entry)) {
                link = &entry->next;
                continue;
            }
            *link = entry->next;
            close(entry->fd);
            DIST_COORD_DEBUG(state,
                             "ds4: distributed coordinator: forgot failed route worker %s:%u layers=%u:%u%s\n",
                             plan->entry[i].host,
                             plan->entry[i].port,
                             entry->layer_start,
                             entry->layer_end,
                             entry->has_output ? "+output" : "");
            free(entry);
            removed_any = true;
            break;
        }
    }
    if (removed_any) state->generation++;
    pthread_mutex_unlock(&state->mu);

    if (removed_any && dist_coordinator_debug_enabled(state)) dist_coordinator_report_plan(state);
}

static bool dist_route_plan_append_blob(
        ds4_dist_route_plan *plan,
        const ds4_dist_route_entry *entry,
        char *err,
        size_t errlen) {
    const size_t host_len = strlen(entry->host);
    if (host_len == 0 || host_len >= NI_MAXHOST) {
        if (errlen) snprintf(err, errlen, "invalid route host");
        return false;
    }
    const uint64_t add = sizeof(ds4_dist_route_fixed) + host_len;
    if (add > UINT32_MAX || plan->blob_bytes > UINT32_MAX - (uint32_t)add) {
        if (errlen) snprintf(err, errlen, "route payload is too large");
        return false;
    }
    uint32_t old_bytes = plan->blob_bytes;
    uint32_t new_bytes = old_bytes + (uint32_t)add;
    void *new_blob = realloc(plan->blob, new_bytes);
    if (!new_blob) {
        if (errlen) snprintf(err, errlen, "out of memory building route payload");
        return false;
    }
    plan->blob = new_blob;
    uint8_t *p = (uint8_t *)plan->blob + old_bytes;
    ds4_dist_route_fixed fixed = {
        (uint32_t)host_len,
        entry->port,
        entry->layer_start,
        entry->layer_end,
        entry->flags,
    };
    dist_route_to_wire(&fixed);
    memcpy(p, &fixed, sizeof(fixed));
    memcpy(p + sizeof(fixed), entry->host, host_len);
    plan->blob_bytes = new_bytes;
    return true;
}

static bool dist_route_plan_append_return_upstream(
        ds4_dist_route_plan *plan,
        char *err,
        size_t errlen) {
    const uint64_t add = sizeof(ds4_dist_route_return_fixed);
    if (plan->blob_bytes > UINT32_MAX - (uint32_t)add) {
        if (errlen) snprintf(err, errlen, "route payload is too large");
        return false;
    }
    const uint32_t old_bytes = plan->blob_bytes;
    const uint32_t new_bytes = old_bytes + (uint32_t)add;
    void *new_blob = realloc(plan->blob, new_bytes);
    if (!new_blob) {
        if (errlen) snprintf(err, errlen, "out of memory building route payload");
        return false;
    }
    plan->blob = new_blob;
    ds4_dist_route_return_fixed fixed = {
        DS4_DIST_ROUTE_RETURN_UPSTREAM,
        0,
        0,
    };
    dist_route_return_to_wire(&fixed);
    memcpy((uint8_t *)plan->blob + old_bytes, &fixed, sizeof(fixed));
    plan->blob_bytes = new_bytes;
    return true;
}

static bool dist_coordinator_build_route_plan(
        ds4_dist_coordinator_state *state,
        ds4_dist_route_plan *plan,
        uint64_t *generation,
        char *err,
        size_t errlen) {
    memset(plan, 0, sizeof(*plan));
    if (generation) *generation = 0;

    pthread_mutex_lock(&state->mu);
    uint32_t n = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) n++;
    ds4_dist_worker_entry **workers = n ? calloc(n, sizeof(workers[0])) : NULL;
    ds4_dist_worker_entry **path = n ? calloc(n, sizeof(path[0])) : NULL;
    if ((n && !workers) || (n && !path)) {
        free(workers);
        free(path);
        pthread_mutex_unlock(&state->mu);
        if (errlen) snprintf(err, errlen, "out of memory building route");
        return false;
    }
    uint32_t i = 0;
    for (ds4_dist_worker_entry *it = state->workers; it; it = it->next) workers[i++] = it;
    qsort(workers, n, sizeof(workers[0]), dist_worker_route_cmp);

    const uint32_t last = state->n_layers - 1u;
    if (state->local_start != 0) {
        pthread_mutex_unlock(&state->mu);
        free(workers);
        free(path);
        if (errlen) snprintf(err, errlen, "coordinator route does not start at layer 0");
        return false;
    }
    if (state->local_end == last &&
        (state->local_has_output || state->local_can_output_head)) {
        if (generation) *generation = state->generation;
        pthread_mutex_unlock(&state->mu);
        free(workers);
        free(path);
        return true;
    }

    uint32_t next = state->local_end + 1u;
    uint32_t path_len = 0;
    uint32_t missing = next;
    if (!dist_route_search_workers(state,
                                   workers,
                                   n,
                                   next,
                                   last,
                                   path,
                                   &path_len,
                                   &missing)) {
        pthread_mutex_unlock(&state->mu);
        free(workers);
        free(path);
        if (errlen) snprintf(err, errlen, "distributed route incomplete: missing layer %u", missing);
        return false;
    }

    for (i = 0; i < path_len; i++) {
        ds4_dist_worker_entry *w = path[i];
        ds4_dist_route_entry entry;
        memset(&entry, 0, sizeof(entry));
        entry.fd = -1;
        snprintf(entry.host, sizeof(entry.host), "%s", w->peer_host);
        entry.port = w->listen_port;
        entry.layer_start = w->layer_start;
        entry.layer_end = w->layer_end;
        entry.flags = w->has_output ? DS4_DIST_ROUTE_F_OUTPUT_LOGITS : 0u;
        if (state->use_control_for_work && plan->count == 0) {
            entry.fd = dup(w->fd);
            if (entry.fd < 0) {
                pthread_mutex_unlock(&state->mu);
                free(workers);
                free(path);
                dist_route_plan_free(plan);
                if (errlen) snprintf(err, errlen, "failed to duplicate first-hop worker connection: %s", strerror(errno));
                return false;
            }
            dist_set_socket_low_latency(entry.fd);
        }

        ds4_dist_route_entry *new_entries = realloc(plan->entry, (size_t)(plan->count + 1u) * sizeof(plan->entry[0]));
        if (!new_entries) {
            pthread_mutex_unlock(&state->mu);
            free(workers);
            free(path);
            if (entry.fd >= 0) close(entry.fd);
            dist_route_plan_free(plan);
            if (errlen) snprintf(err, errlen, "out of memory building route entries");
            return false;
        }
        plan->entry = new_entries;
        plan->entry[plan->count++] = entry;
        if (!dist_route_plan_append_blob(plan, &entry, err, errlen)) {
            pthread_mutex_unlock(&state->mu);
            free(workers);
            free(path);
            dist_route_plan_free(plan);
            return false;
        }
    }
    if (generation) *generation = state->generation;
    pthread_mutex_unlock(&state->mu);
    free(workers);
    free(path);
    if (plan->count != 0 && !dist_route_plan_append_return_upstream(plan, err, errlen)) {
        dist_route_plan_free(plan);
        return false;
    }
    return true;
}

static int dist_logits_argmax(const float *logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

static bool dist_coordinator_ensure_route(
        ds4_dist_coordinator_state *state,
        ds4_dist_route_plan *plan,
        uint64_t *generation,
        char *err,
        size_t errlen) {
    return dist_coordinator_build_route_plan(state, plan, generation, err, errlen);
}

static uint64_t dist_coordinator_generation(ds4_dist_coordinator_state *state) {
    if (!state) return 0;
    pthread_mutex_lock(&state->mu);
    uint64_t generation = state->generation;
    pthread_mutex_unlock(&state->mu);
    return generation;
}

/* =========================================================================
 * Coordinator Work Dispatch
 * ========================================================================= */

static int dist_recv_result_alloc(
        int fd,
        const ds4_dist_coordinator_state *state,
        uint64_t request_id,
        uint32_t *kind,
        uint64_t *result_hash,
        void **payload,
        uint32_t *payload_bytes,
        char *err,
        size_t errlen) {
    *payload = NULL;
    *payload_bytes = 0;
    *kind = 0;
    if (result_hash) *result_hash = 0;

    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) {
        if (rc == 0 && errlen) snprintf(err, errlen, "distributed worker closed connection");
        return 1;
    }
    if (type != DS4_DIST_MSG_RESULT || bytes < sizeof(ds4_dist_result_fixed)) {
        dist_discard_bytes(fd, bytes);
        if (errlen) snprintf(err, errlen, "distributed worker returned invalid frame");
        return 1;
    }

    ds4_dist_result_fixed result;
    rc = dist_read_full(fd, &result, sizeof(result));
    if (rc <= 0) {
        if (errlen) snprintf(err, errlen, "failed to read distributed result");
        return 1;
    }
    dist_result_from_wire(&result);
    const uint64_t got_request = dist_u64_from_halves(result.request_hi, result.request_lo);
    const uint64_t got_hash = dist_u64_from_halves(result.result_hash_hi,
                                                  result.result_hash_lo);
    const uint32_t body_bytes = bytes - (uint32_t)sizeof(result);
    if (result.telemetry_bytes % (uint32_t)sizeof(ds4_dist_telemetry_fixed) != 0 ||
        result.telemetry_count != result.telemetry_bytes / (uint32_t)sizeof(ds4_dist_telemetry_fixed) ||
        result.telemetry_bytes > body_bytes ||
        result.payload_bytes != body_bytes - result.telemetry_bytes) {
        dist_discard_bytes(fd, body_bytes);
        if (errlen) snprintf(err, errlen, "distributed result telemetry metadata mismatch");
        return 1;
    }
    if (got_request != request_id) {
        dist_discard_bytes(fd, bytes - (uint32_t)sizeof(result));
        if (errlen) snprintf(err, errlen, "distributed result metadata mismatch");
        return 1;
    }

    if (result.telemetry_bytes != 0) {
        if (dist_coordinator_debug_enabled(state)) {
            ds4_dist_telemetry_fixed *telemetry = malloc(result.telemetry_bytes);
            if (!telemetry) {
                dist_discard_bytes(fd, result.telemetry_bytes);
                if (errlen) snprintf(err, errlen, "out of memory reading distributed telemetry");
                return 1;
            }
            rc = dist_read_full(fd, telemetry, result.telemetry_bytes);
            if (rc <= 0) {
                free(telemetry);
                if (errlen) snprintf(err, errlen, "failed to read distributed result telemetry");
                return 1;
            }
            for (uint32_t i = 0; i < result.telemetry_count; i++) {
                dist_telemetry_from_wire(&telemetry[i]);
                DIST_COORD_DEBUG(state,
                                 "ds4: distributed telemetry: request=%llu hop=%u layers=%u:%u route=%u pos=%u tokens=%u eval=%.3fms downstream_wait=%.3fms forward_send=%.3fms input=%.2fMiB output=%.2fMiB\n",
                                 (unsigned long long)got_request,
                                 i,
                                 telemetry[i].layer_start,
                                 telemetry[i].layer_end,
                                 telemetry[i].route_index,
                                 telemetry[i].pos0,
                                 telemetry[i].n_tokens,
                                 (double)telemetry[i].eval_usec / 1000.0,
                                 (double)telemetry[i].downstream_wait_usec / 1000.0,
                                 (double)telemetry[i].forward_send_usec / 1000.0,
                                 (double)telemetry[i].input_bytes / (1024.0 * 1024.0),
                                 (double)telemetry[i].output_bytes / (1024.0 * 1024.0));
            }
            free(telemetry);
        } else if (dist_discard_bytes(fd, result.telemetry_bytes) <= 0) {
            if (errlen) snprintf(err, errlen, "failed to read distributed result telemetry");
            return 1;
        }
    }

    void *buf = NULL;
    if (result.payload_bytes != 0) {
        buf = malloc(result.payload_bytes);
        if (!buf) {
            dist_discard_bytes(fd, result.payload_bytes);
            if (errlen) snprintf(err, errlen, "out of memory reading distributed result");
            return 1;
        }
        rc = dist_read_full(fd, buf, result.payload_bytes);
        if (rc <= 0) {
            free(buf);
            if (errlen) snprintf(err, errlen, "failed to read distributed result payload");
            return 1;
        }
    }

    if (result.status != 0) {
        if (errlen) {
            if (buf && result.payload_bytes) {
                size_t n = result.payload_bytes < errlen - 1 ? result.payload_bytes : errlen - 1;
                memcpy(err, buf, n);
                err[n] = '\0';
            } else {
                snprintf(err, errlen, "distributed worker returned an error");
            }
        }
        free(buf);
        return DS4_DIST_RECV_REMOTE_ERROR;
    }

    if (result.result_kind == DS4_DIST_RESULT_HIDDEN_STATE && result.payload_bytes != 0) {
        float *decoded = NULL;
        uint32_t decoded_bytes = 0;
        bool uses_wire = false;
        if (dist_decode_activation_payload(buf,
                                           result.payload_bits,
                                           result.payload_bytes,
                                           &decoded,
                                           &decoded_bytes,
                                           &uses_wire,
                                           err,
                                           errlen) != 0) {
            free(buf);
            return 1;
        }
        if (!uses_wire) {
            free(buf);
            buf = decoded;
        }
        result.payload_bytes = decoded_bytes;
    }

    *kind = result.result_kind;
    if (result_hash) *result_hash = got_hash;
    *payload = buf;
    *payload_bytes = result.payload_bytes;
    return 0;
}

static int dist_coordinator_send_remote_work_on_fd(
        ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        int fd,
        const int *tokens,
        uint32_t n_tokens,
        uint32_t pos0,
        uint64_t session_id,
        uint64_t request_id,
        uint64_t prefix_hash,
        uint64_t result_hash,
        bool reset_session,
        bool ack_only,
        const float *hidden_hc,
        uint32_t hidden_hc_bytes,
        char *err,
        size_t errlen) {
    if (plan->count == 0) {
        if (errlen) snprintf(err, errlen, "distributed route has no remote worker");
        return 1;
    }
    const ds4_dist_route_entry *first = &plan->entry[0];

    ds4_dist_work_fixed work;
    memset(&work, 0, sizeof(work));
    work.model_id = state->model_id;
    dist_u64_to_halves(session_id, &work.session_hi, &work.session_lo);
    dist_u64_to_halves(request_id, &work.request_hi, &work.request_lo);
    dist_u64_to_halves(prefix_hash, &work.prefix_hash_hi, &work.prefix_hash_lo);
    dist_u64_to_halves(result_hash, &work.result_hash_hi, &work.result_hash_lo);
    work.pos0 = pos0;
    work.n_tokens = n_tokens;
    work.layer_start = first->layer_start;
    work.layer_end = first->layer_end;
    work.flags = DS4_DIST_WORK_F_INPUT_HC;
    if (reset_session) work.flags |= DS4_DIST_WORK_F_RESET_SESSION;
    if (ack_only) work.flags |= DS4_DIST_WORK_F_ACK_ONLY;
    if ((first->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0) {
        work.flags |= DS4_DIST_WORK_F_OUTPUT_LOGITS;
    }
    uint32_t wire_hidden_hc_bytes = 0;
    if (!dist_activation_wire_bytes_from_f32_bytes(state->activation_bits,
                                                   hidden_hc_bytes,
                                                   &wire_hidden_hc_bytes)) {
        if (errlen) snprintf(err, errlen, "invalid distributed hidden-state size");
        return 1;
    }
    work.token_bytes = n_tokens * sizeof(uint32_t);
    work.input_hc_bytes = wire_hidden_hc_bytes;
    work.input_hc_bits = state->activation_bits;
    work.route_count = plan->count;
    work.route_index = 0;
    work.route_bytes = plan->blob_bytes;

    if (dist_send_work_frame(fd, &work, tokens, hidden_hc, plan->blob) != 0) {
        if (errlen) snprintf(err, errlen, "failed to send distributed work");
        return 1;
    }
    return 0;
}

static int dist_coordinator_eval_remote_on_fd(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        int fd,
        const int *tokens,
        uint32_t n_tokens,
        uint32_t pos0,
        uint64_t session_id,
        uint64_t request_id,
        uint64_t prefix_hash,
        uint64_t expected_result_hash,
        bool reset_session,
        const float *hidden_hc,
        uint32_t hidden_hc_bytes,
        float *logits,
        char *err,
        size_t errlen) {
    const bool profile = dist_decode_profile_enabled() && n_tokens == 1;
    const double total_t0 = profile ? dist_now_sec() : 0.0;
    const double send_t0 = profile ? dist_now_sec() : 0.0;
    int rc = dist_coordinator_send_remote_work_on_fd(state,
                                                     plan,
                                                     fd,
                                                     tokens,
                                                     n_tokens,
                                                     pos0,
                                                     session_id,
                                                     request_id,
                                                     prefix_hash,
                                                     expected_result_hash,
                                                     reset_session,
                                                     false,
                                                     hidden_hc,
                                                     hidden_hc_bytes,
                                                     err,
                                                     errlen);
    const double send_t1 = profile ? dist_now_sec() : 0.0;
    uint32_t kind = 0, payload_bytes = 0;
    uint64_t result_hash = 0;
    void *payload = NULL;
    if (rc == 0) {
        const double recv_t0 = profile ? dist_now_sec() : 0.0;
        rc = dist_recv_result_alloc(fd,
                                    state,
                                    request_id,
                                    &kind,
                                    &result_hash,
                                    &payload,
                                    &payload_bytes,
                                    err,
                                    errlen);
        const double recv_t1 = profile ? dist_now_sec() : 0.0;
        if (profile) {
            fprintf(stderr,
                    "ds4: dist decode profile: remote request=%llu pos=%u send=%.3fms wait_result=%.3fms kind=%u payload=%.2fMiB\n",
                    (unsigned long long)request_id,
                    pos0,
                    (send_t1 - send_t0) * 1000.0,
                    (recv_t1 - recv_t0) * 1000.0,
                    kind,
                    (double)payload_bytes / (1024.0 * 1024.0));
        }
    }
    if (rc != 0) return rc;
    if (result_hash != expected_result_hash) {
        free(payload);
        if (errlen) snprintf(err, errlen, "distributed result prefix hash mismatch");
        return 1;
    }

    const uint32_t logits_bytes = (uint32_t)((uint64_t)ds4_engine_vocab_size(state->engine) * sizeof(float));
    if (kind == DS4_DIST_RESULT_LOGITS && payload_bytes == logits_bytes) {
        const double copy_t0 = profile ? dist_now_sec() : 0.0;
        memcpy(logits, payload, logits_bytes);
        free(payload);
        if (profile) {
            const double copy_t1 = dist_now_sec();
            fprintf(stderr,
                    "ds4: dist decode profile: remote request=%llu copy_logits=%.3fms total=%.3fms\n",
                    (unsigned long long)request_id,
                    (copy_t1 - copy_t0) * 1000.0,
                    (copy_t1 - total_t0) * 1000.0);
        }
        return 0;
    }
    if (kind == DS4_DIST_RESULT_HIDDEN_STATE && payload_bytes == hidden_hc_bytes) {
        const double head_t0 = profile ? dist_now_sec() : 0.0;
        int head_rc = ds4_session_eval_output_head_from_hc(session,
                                                           payload,
                                                           n_tokens,
                                                           logits,
                                                           err,
                                                           errlen);
        free(payload);
        if (profile) {
            const double head_t1 = dist_now_sec();
            fprintf(stderr,
                    "ds4: dist decode profile: remote request=%llu output_head=%.3fms total=%.3fms rc=%d\n",
                    (unsigned long long)request_id,
                    (head_t1 - head_t0) * 1000.0,
                    (head_t1 - total_t0) * 1000.0,
                    head_rc);
        }
        return head_rc;
    }
    if (kind == DS4_DIST_RESULT_HIDDEN_STATE) {
        free(payload);
        if (errlen) snprintf(err, errlen, "distributed route returned invalid hidden-state size");
        return 1;
    }
    free(payload);
    if (errlen) snprintf(err, errlen, "distributed route did not return logits or hidden-state");
    return 1;
}

static int dist_coordinator_eval_span(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const int *tokens,
        uint32_t n_tokens,
        uint32_t pos0,
        uint64_t session_id,
        uint64_t request_id,
        bool reset_session,
        float *logits,
        char *err,
        size_t errlen) {
    const bool profile = dist_decode_profile_enabled() && n_tokens == 1;
    const double span_t0 = profile ? dist_now_sec() : 0.0;
    const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
    const uint64_t hidden_bytes64 = (uint64_t)n_tokens * hc_values * sizeof(float);
    if (hidden_bytes64 > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed coordinator hidden-state chunk is too large");
        return 1;
    }
    uint64_t prefix_hash = DS4_DIST_TOKEN_HASH_INIT;
    if (reset_session) {
        if (pos0 != 0) {
            if (errlen) snprintf(err, errlen, "distributed reset span must start at position 0");
            return 1;
        }
    } else if (dist_session_token_hash_prefix(session,
                                              pos0,
                                              &prefix_hash,
                                              err,
                                              errlen) != 0) {
        return 1;
    }
    const uint64_t result_hash = dist_token_hash_update_span(prefix_hash, tokens, n_tokens);
    const uint32_t hidden_bytes = (uint32_t)hidden_bytes64;
    float *hidden = NULL;
    if (plan->count != 0) {
        hidden = malloc(hidden_bytes);
        if (!hidden) {
            if (errlen) snprintf(err, errlen, "out of memory allocating coordinator hidden-state");
            return 1;
        }
    }
    if (reset_session &&
        ds4_session_layer_slice_reset(session, err, errlen) != 0) {
        free(hidden);
        return 1;
    }

    const bool local_logits = plan->count == 0;
    int remote_fd = -1;
    if (plan->count != 0) {
        const ds4_dist_route_entry *first = &plan->entry[0];
        remote_fd = first->fd;
        if (remote_fd < 0) {
            if (errlen) snprintf(err, errlen, "distributed route has no live first-hop connection");
            free(hidden);
            return 1;
        }
    }

    const double local_t0 = profile ? dist_now_sec() : 0.0;
    int rc = ds4_session_eval_layer_slice(session,
                                          tokens,
                                          n_tokens,
                                          pos0,
                                          state->local_start,
                                          state->local_end,
                                          NULL,
                                          local_logits ? NULL : hidden,
                                          local_logits,
                                          local_logits ? logits : NULL,
                                          err,
                                          errlen);
    const double local_t1 = profile ? dist_now_sec() : 0.0;
    double remote_t0 = 0.0, remote_t1 = 0.0;
    if (rc == 0 && plan->count != 0) {
        remote_t0 = profile ? dist_now_sec() : 0.0;
        rc = dist_coordinator_eval_remote_on_fd(state,
                                                session,
                                                plan,
                                                remote_fd,
                                                tokens,
                                                n_tokens,
                                                pos0,
                                                session_id,
                                                request_id,
                                                prefix_hash,
                                                result_hash,
                                                reset_session,
                                                hidden,
                                                hidden_bytes,
                                                logits,
                                                err,
                                                errlen);
        remote_t1 = profile ? dist_now_sec() : 0.0;
    }
    if (profile) {
        const double span_t1 = dist_now_sec();
        fprintf(stderr,
                "ds4: dist decode profile: span request=%llu pos=%u layers=%u:%u local=%.3fms remote=%.3fms total=%.3fms hidden=%.2fMiB rc=%d\n",
                (unsigned long long)request_id,
                pos0,
                state->local_start,
                state->local_end,
                (local_t1 - local_t0) * 1000.0,
                (remote_t1 - remote_t0) * 1000.0,
                (span_t1 - span_t0) * 1000.0,
                (double)hidden_bytes / (1024.0 * 1024.0),
                rc);
    }
    free(hidden);
    return rc;
}

