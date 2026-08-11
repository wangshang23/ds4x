/* =========================================================================
 * Worker KV Sessions And Snapshot Handlers
 * ========================================================================= */

static ds4_dist_worker_session *dist_worker_get_session_locked(
        ds4_dist_worker_state *state,
        uint64_t session_id,
        char *err,
        size_t errlen) {
    for (ds4_dist_worker_session *it = state->sessions; it; it = it->next) {
        if (it->session_id == session_id) return it;
    }

    ds4_dist_worker_session *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        if (errlen) snprintf(err, errlen, "out of memory creating distributed session");
        return NULL;
    }
    if (ds4_session_create(&entry->session, state->engine, state->ctx_size) != 0) {
        free(entry);
        if (errlen) snprintf(err, errlen, "failed to create distributed worker session");
        return NULL;
    }
    entry->session_id = session_id;
    entry->next = state->sessions;
    state->sessions = entry;
    return entry;
}

static ds4_dist_worker_session *dist_worker_find_session_locked(
        ds4_dist_worker_state *state,
        uint64_t session_id) {
    for (ds4_dist_worker_session *it = state->sessions; it; it = it->next) {
        if (it->session_id == session_id) return it;
    }
    return NULL;
}

static uint32_t dist_worker_clear_sessions(ds4_dist_worker_state *state) {
    uint32_t n = 0;
    pthread_mutex_lock(&state->mu);
    ds4_dist_worker_session *it = state->sessions;
    state->sessions = NULL;
    pthread_mutex_unlock(&state->mu);

    while (it) {
        ds4_dist_worker_session *next = it->next;
        ds4_session_free(it->session);
        free(it);
        it = next;
        n++;
    }
    return n;
}

typedef struct {
    const uint8_t *p;
    uint32_t remaining;
} ds4_dist_mem_reader;

static int dist_mem_read(ds4_dist_mem_reader *r, void *dst, uint32_t len) {
    if (!r || len > r->remaining) return -1;
    if (len != 0) memcpy(dst, r->p, len);
    r->p += len;
    r->remaining -= len;
    return 1;
}

static int dist_temp_file(const char *prefix, char *path, size_t path_len, FILE **fp_out) {
    char tmpl[PATH_MAX];
    snprintf(tmpl, sizeof(tmpl), "/tmp/%s.XXXXXX", prefix);
    int fd = mkstemp(tmpl);
    if (fd < 0) return -1;
    FILE *fp = fdopen(fd, "w+b");
    if (!fp) {
        close(fd);
        unlink(tmpl);
        return -1;
    }
    snprintf(path, path_len, "%s", tmpl);
    *fp_out = fp;
    return 0;
}

static int dist_worker_handle_snapshot_save(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes) {
    ds4_dist_snapshot_req_fixed req;
    uint64_t request_id = 0;
    uint64_t session_id = 0;
    if (bytes != sizeof(req)) {
        dist_discard_bytes(upstream->fd, bytes);
        pthread_mutex_lock(&upstream->write_mu);
        int rc = dist_send_snapshot_error(upstream->fd, 0, 0, state->model_id,
                                          state->layer_start, state->layer_end,
                                          "invalid distributed snapshot save request");
        pthread_mutex_unlock(&upstream->write_mu);
        return rc;
    }
    int rc = dist_read_full(upstream->fd, &req, sizeof(req));
    if (rc <= 0) return rc == 0 ? 0 : -1;
    dist_snapshot_req_from_wire(&req);
    request_id = dist_u64_from_halves(req.request_hi, req.request_lo);
    session_id = dist_u64_from_halves(req.session_hi, req.session_lo);
    const uint64_t token_hash = dist_u64_from_halves(req.token_hash_hi, req.token_hash_lo);

    char err[256] = {0};
    FILE *tmp = NULL;
    char tmp_path[PATH_MAX];
    uint64_t payload_bytes = 0;

    if (req.model_id != state->model_id ||
        req.layer_start != state->layer_start ||
        req.layer_end != state->layer_end ||
        req.token_count > (uint32_t)state->ctx_size) {
        snprintf(err, sizeof(err), "snapshot save request does not match worker state");
    } else {
        pthread_mutex_lock(&state->mu);
        ds4_dist_worker_session *session = dist_worker_find_session_locked(state, session_id);
        if (!session) {
            snprintf(err, sizeof(err), "worker has no distributed session to snapshot");
        } else {
            const ds4_tokens *timeline = ds4_session_tokens(session->session);
            uint64_t live_hash = 0;
            if (!timeline || timeline->len < 0 || (uint32_t)timeline->len != req.token_count) {
                snprintf(err, sizeof(err), "worker snapshot token count mismatch");
            } else {
                live_hash = dist_token_hash_prefix(timeline->v, (uint32_t)timeline->len);
                if (live_hash != token_hash) {
                    snprintf(err, sizeof(err), "worker snapshot token hash mismatch");
                }
            }
            if (!err[0] && dist_temp_file("ds4-dist-save", tmp_path, sizeof(tmp_path), &tmp) != 0) {
                snprintf(err, sizeof(err), "failed to create worker snapshot temp file");
            }
            if (!err[0] &&
                ds4_session_save_layer_payload(session->session,
                                               tmp,
                                               state->layer_start,
                                               state->layer_end,
                                               err,
                                               sizeof(err)) != 0) {
                if (!err[0]) snprintf(err, sizeof(err), "failed to save worker KV shard");
            }
            if (!err[0] && fflush(tmp) != 0) {
                snprintf(err, sizeof(err), "failed to flush worker KV shard");
            }
            if (!err[0]) {
                off_t pos = ftello(tmp);
                if (pos < 0) snprintf(err, sizeof(err), "failed to measure worker KV shard");
                else payload_bytes = (uint64_t)pos;
            }
            if (!err[0] && fseeko(tmp, 0, SEEK_SET) != 0) {
                snprintf(err, sizeof(err), "failed to rewind worker KV shard");
            }
        }
        pthread_mutex_unlock(&state->mu);
    }

    pthread_mutex_lock(&upstream->write_mu);
    if (err[0]) {
        rc = dist_send_snapshot_error(upstream->fd,
                                      request_id,
                                      session_id,
                                      state->model_id,
                                      state->layer_start,
                                      state->layer_end,
                                      err);
    } else {
        ds4_dist_snapshot_begin_fixed begin;
        memset(&begin, 0, sizeof(begin));
        begin.model_id = state->model_id;
        dist_u64_to_halves(session_id, &begin.session_hi, &begin.session_lo);
        dist_u64_to_halves(request_id, &begin.request_hi, &begin.request_lo);
        dist_u64_to_halves(token_hash, &begin.token_hash_hi, &begin.token_hash_lo);
        begin.layer_start = state->layer_start;
        begin.layer_end = state->layer_end;
        dist_u64_to_halves(payload_bytes, &begin.payload_hi, &begin.payload_lo);
        rc = dist_send_snapshot_begin(upstream->fd, &begin, NULL, NULL);
        if (rc > 0) rc = dist_send_snapshot_file_chunks(upstream->fd, request_id, tmp, payload_bytes);
        if (rc > 0) rc = dist_send_snapshot_done(upstream->fd, request_id, 0, NULL);
    }
    pthread_mutex_unlock(&upstream->write_mu);

    if (tmp) fclose(tmp);
    if (tmp) unlink(tmp_path);
    return rc;
}

static int dist_worker_handle_snapshot_load(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes) {
    ds4_dist_snapshot_begin_fixed begin;
    uint64_t request_id = 0;
    uint64_t session_id = 0;
    char err[256] = {0};
    if (bytes < sizeof(begin)) {
        dist_discard_bytes(upstream->fd, bytes);
        return -1;
    }
    int rc = dist_read_full(upstream->fd, &begin, sizeof(begin));
    if (rc <= 0) return rc == 0 ? 0 : -1;
    dist_snapshot_begin_from_wire(&begin);
    request_id = dist_u64_from_halves(begin.request_hi, begin.request_lo);
    session_id = dist_u64_from_halves(begin.session_hi, begin.session_lo);
    const uint64_t token_hash = dist_u64_from_halves(begin.token_hash_hi,
                                                     begin.token_hash_lo);
    const uint64_t payload_bytes = dist_u64_from_halves(begin.payload_hi,
                                                        begin.payload_lo);
    const uint32_t body_bytes = bytes - (uint32_t)sizeof(begin);
    uint64_t expected_token_bytes = (uint64_t)begin.token_count * sizeof(uint32_t);
    if (expected_token_bytes > UINT32_MAX ||
        begin.token_bytes != (uint32_t)expected_token_bytes ||
        begin.message_bytes != 0 ||
        body_bytes != begin.token_bytes) {
        dist_discard_bytes(upstream->fd, body_bytes);
        snprintf(err, sizeof(err), "invalid distributed snapshot load header");
    }

    /* Reject a token_count larger than the worker context up front, before the
     * allocation and the read loop below. token_count is only bounded to
     * ~UINT32_MAX/4 by the header check above, so without this a peer could
     * drive a multi-GB malloc and a full-length socket read before the
     * matching-state check further down (which also tests token_count > ctx_size)
     * finally rejects the request. */
    if (!err[0] && begin.token_count > (uint32_t)state->ctx_size) {
        dist_discard_bytes(upstream->fd, body_bytes);
        snprintf(err, sizeof(err), "snapshot token count exceeds worker context size");
    }

    int *tokens = NULL;
    if (!err[0]) {
        tokens = malloc((size_t)begin.token_count * sizeof(tokens[0]));
        if (!tokens && begin.token_count != 0) {
            dist_discard_bytes(upstream->fd, begin.token_bytes);
            snprintf(err, sizeof(err), "out of memory reading snapshot tokens");
        }
    }
    for (uint32_t i = 0; !err[0] && i < begin.token_count; i++) {
        uint32_t wire_token = 0;
        rc = dist_read_full(upstream->fd, &wire_token, sizeof(wire_token));
        if (rc <= 0) {
            free(tokens);
            return rc == 0 ? 0 : -1;
        }
        uint32_t token = ntohl(wire_token);
        if (token > (uint32_t)INT_MAX ||
            token >= (uint32_t)ds4_engine_vocab_size(state->engine)) {
            snprintf(err, sizeof(err), "snapshot token id is outside the model vocabulary");
            tokens[i] = 0;
        } else {
            tokens[i] = (int)token;
        }
    }
    if (!err[0] &&
        dist_token_hash_prefix(tokens, begin.token_count) != token_hash) {
        snprintf(err, sizeof(err), "snapshot load token hash mismatch");
    }
    if (!err[0] &&
        (begin.model_id != state->model_id ||
         begin.layer_start != state->layer_start ||
         begin.layer_end != state->layer_end ||
         begin.token_count > (uint32_t)state->ctx_size)) {
        snprintf(err, sizeof(err), "snapshot load request does not match worker state");
    }

    FILE *tmp = NULL;
    char tmp_path[PATH_MAX];
    if (!err[0] && dist_temp_file("ds4-dist-load", tmp_path, sizeof(tmp_path), &tmp) != 0) {
        snprintf(err, sizeof(err), "failed to create worker snapshot restore temp file");
    }

    uint8_t *buf = NULL;
    if (!err[0]) {
        buf = malloc(DS4_DIST_SNAPSHOT_CHUNK_BYTES);
        if (!buf) snprintf(err, sizeof(err), "out of memory restoring worker KV shard");
    }
    uint64_t received = 0;
    while (!err[0] && received < payload_bytes) {
        uint32_t type = 0, chunk_frame_bytes = 0;
        rc = dist_read_frame_header(upstream->fd, &type, &chunk_frame_bytes, err, sizeof(err));
        if (rc <= 0) {
            free(buf);
            free(tokens);
            if (tmp) fclose(tmp);
            if (tmp) unlink(tmp_path);
            return rc == 0 ? 0 : -1;
        }
        if (type != DS4_DIST_MSG_SNAPSHOT_CHUNK ||
            chunk_frame_bytes < sizeof(ds4_dist_snapshot_chunk_fixed)) {
            dist_discard_bytes(upstream->fd, chunk_frame_bytes);
            snprintf(err, sizeof(err), "expected distributed snapshot chunk");
            break;
        }
        ds4_dist_snapshot_chunk_fixed chunk;
        rc = dist_read_full(upstream->fd, &chunk, sizeof(chunk));
        if (rc <= 0) {
            free(buf);
            free(tokens);
            if (tmp) fclose(tmp);
            if (tmp) unlink(tmp_path);
            return rc == 0 ? 0 : -1;
        }
        dist_snapshot_chunk_from_wire(&chunk);
        uint64_t got_request = dist_u64_from_halves(chunk.request_hi, chunk.request_lo);
        uint32_t chunk_bytes = chunk_frame_bytes - (uint32_t)sizeof(chunk);
        if (got_request != request_id ||
            chunk.chunk_bytes != chunk_bytes ||
            chunk_bytes > DS4_DIST_SNAPSHOT_CHUNK_BYTES ||
            chunk_bytes > payload_bytes - received) {
            dist_discard_bytes(upstream->fd, chunk_bytes);
            snprintf(err, sizeof(err), "invalid distributed snapshot chunk");
            break;
        }
        rc = dist_read_full(upstream->fd, buf, chunk_bytes);
        if (rc <= 0) {
            free(buf);
            free(tokens);
            if (tmp) fclose(tmp);
            if (tmp) unlink(tmp_path);
            return rc == 0 ? 0 : -1;
        }
        if (fwrite(buf, 1, chunk_bytes, tmp) != chunk_bytes) {
            snprintf(err, sizeof(err), "failed to write worker KV shard temp file");
            break;
        }
        received += chunk_bytes;
    }
    free(buf);

    if (!err[0] && fflush(tmp) != 0) {
        snprintf(err, sizeof(err), "failed to flush worker KV shard restore file");
    }
    if (!err[0] && fseeko(tmp, 0, SEEK_SET) != 0) {
        snprintf(err, sizeof(err), "failed to rewind worker KV shard restore file");
    }
    if (!err[0]) {
        pthread_mutex_lock(&state->mu);
        ds4_dist_worker_session *session =
            dist_worker_get_session_locked(state, session_id, err, sizeof(err));
        if (session &&
            ds4_session_load_layer_payload(session->session,
                                           tmp,
                                           payload_bytes,
                                           tokens,
                                           begin.token_count,
                                           state->layer_start,
                                           state->layer_end,
                                           err,
                                           sizeof(err)) == 0) {
            session->token_hash = token_hash;
            session->token_hash_valid = true;
        } else {
            if (!err[0]) snprintf(err, sizeof(err), "failed to restore worker KV shard");
            if (session) session->token_hash_valid = false;
        }
        pthread_mutex_unlock(&state->mu);
    }

    if (tmp) fclose(tmp);
    if (tmp) unlink(tmp_path);
    free(tokens);

    pthread_mutex_lock(&upstream->write_mu);
    rc = dist_send_snapshot_done(upstream->fd, request_id, err[0] ? 1u : 0u,
                                 err[0] ? err : NULL);
    pthread_mutex_unlock(&upstream->write_mu);
    if (err[0] && received < payload_bytes) return -1;
    return rc;
}

/* =========================================================================
 * Worker Layer Execution
 * ========================================================================= */

static int dist_worker_process_work_payload(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        const void *payload,
        uint32_t bytes) {
    uint64_t request_id = 0;
    char err[256];
    if (bytes < sizeof(ds4_dist_work_fixed)) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "truncated distributed WORK frame");
    }

    ds4_dist_mem_reader reader = {
        .p = payload,
        .remaining = bytes,
    };
    ds4_dist_work_fixed work;
    int rc = dist_mem_read(&reader, &work, (uint32_t)sizeof(work));
    if (rc <= 0) return -1;
    dist_work_from_wire(&work);
    const uint64_t session_id = dist_u64_from_halves(work.session_hi, work.session_lo);
    request_id = dist_u64_from_halves(work.request_hi, work.request_lo);
    const uint64_t work_prefix_hash = dist_u64_from_halves(work.prefix_hash_hi,
                                                           work.prefix_hash_lo);
    const uint64_t work_result_hash = dist_u64_from_halves(work.result_hash_hi,
                                                           work.result_hash_lo);
    const bool profile = dist_decode_profile_enabled() && work.n_tokens == 1;
    const double total_t0 = profile ? dist_now_sec() : 0.0;
    DIST_DEBUG("worker work request=%llu layers=%u:%u tokens=%u pos=%u flags=0x%x token_bytes=%u input_hc=%u/%ub route_count=%u route_index=%u route_bytes=%u",
               (unsigned long long)request_id,
               work.layer_start,
               work.layer_end,
               work.n_tokens,
               work.pos0,
               work.flags,
               work.token_bytes,
               work.input_hc_bytes,
               work.input_hc_bits,
               work.route_count,
               work.route_index,
               work.route_bytes);

    const uint32_t remaining = bytes - (uint32_t)sizeof(work);
    const uint64_t token_bytes_expected = (uint64_t)work.n_tokens * sizeof(uint32_t);
    const uint64_t payload_bytes_expected =
        (uint64_t)work.token_bytes + work.input_hc_bytes + work.route_bytes;
    if ((uint64_t)work.token_bytes != token_bytes_expected ||
        payload_bytes_expected != remaining) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "invalid distributed WORK payload sizes");
    }
    if (work.route_count == 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK frame is missing distributed route");
    }
    if (work.route_index >= work.route_count) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "invalid distributed WORK route metadata");
    }
    if (work.model_id != state->model_id) {
        snprintf(err, sizeof(err), "model id mismatch: work=%u worker=%u", work.model_id, state->model_id);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if (work.layer_start != state->layer_start || work.layer_end != state->layer_end) {
        snprintf(err, sizeof(err), "worker is assigned layers %u:%u but request asked for %u:%u",
                 state->layer_start, state->layer_end, work.layer_start, work.layer_end);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if ((work.flags & ~DS4_DIST_WORK_F_VALID_MASK) != 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "invalid distributed WORK flags");
    }
    if (work.n_tokens == 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK frame has no tokens");
    }
    if (work.pos0 > (uint32_t)state->ctx_size ||
        work.n_tokens > (uint32_t)state->ctx_size - work.pos0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK token span exceeds worker context");
    }

    const bool output_logits = (work.flags & DS4_DIST_WORK_F_OUTPUT_LOGITS) != 0;
    const bool input_hc_present = (work.flags & DS4_DIST_WORK_F_INPUT_HC) != 0;
    const bool ack_only = (work.flags & DS4_DIST_WORK_F_ACK_ONLY) != 0;
    if (input_hc_present && work.layer_start == 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "layer 0 WORK must not provide input hidden-state");
    }
    if (!input_hc_present && work.layer_start != 0) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "nonzero layer WORK requires input hidden-state");
    }
    if (output_logits && !state->has_output) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "worker was not assigned the output head");
    }
    const uint32_t n_layers = (uint32_t)ds4_engine_layer_count(state->engine);
    if (output_logits && work.layer_end + 1u != n_layers) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK logits require final transformer layer");
    }

    int *tokens = malloc((size_t)work.n_tokens * sizeof(tokens[0]));
    if (!tokens) {
        return dist_worker_upstream_send_work_error(upstream, request_id, "out of memory reading WORK tokens");
    }
    for (uint32_t i = 0; i < work.n_tokens; i++) {
        uint32_t wire_token = 0;
        rc = dist_mem_read(&reader, &wire_token, (uint32_t)sizeof(wire_token));
        if (rc <= 0) {
            free(tokens);
            return -1;
        }
        uint32_t token = ntohl(wire_token);
        if (token > (uint32_t)INT_MAX || token >= (uint32_t)ds4_engine_vocab_size(state->engine)) {
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "WORK token id is outside the model vocabulary");
        }
        tokens[i] = (int)token;
    }
    if (dist_token_hash_update_span(work_prefix_hash, tokens, work.n_tokens) !=
        work_result_hash) {
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK token prefix hash metadata mismatch");
    }

    const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
    const uint64_t expected_hc_values = (uint64_t)work.n_tokens * hc_values;
    const uint64_t expected_hc_bytes64 = expected_hc_values * sizeof(float);
    if (expected_hc_bytes64 > UINT32_MAX) {
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "distributed hidden-state payload is too large");
    }
    const uint32_t expected_hc_bytes = (uint32_t)expected_hc_bytes64;
    const uint32_t input_hc_bits = dist_activation_bits_or_default(work.input_hc_bits);

    float *input_hc = NULL;
    const void *input_hc_wire = NULL;
    if (input_hc_present) {
        uint32_t expected_hc_wire_bytes = 0;
        if (!dist_activation_bits_valid(input_hc_bits) ||
            !dist_activation_wire_bytes(input_hc_bits,
                                        expected_hc_values,
                                        &expected_hc_wire_bytes)) {
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "invalid distributed activation width");
        }
        if (work.input_hc_bytes != expected_hc_wire_bytes) {
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "input hidden-state size does not match token span");
        }
        if (work.input_hc_bytes > reader.remaining) {
            DIST_DEBUG("worker input hidden read failed request=%llu rc=%d bytes=%u",
                       (unsigned long long)request_id,
                       -1,
                       work.input_hc_bytes);
            free(tokens);
            return -1;
        }
        input_hc_wire = reader.p;
        reader.p += work.input_hc_bytes;
        reader.remaining -= work.input_hc_bytes;
        DIST_DEBUG("worker input hidden read ok request=%llu bytes=%u bits=%u",
                   (unsigned long long)request_id,
                   work.input_hc_bytes,
                   input_hc_bits);
    } else if (work.input_hc_bytes != 0) {
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "WORK frame has hidden bytes without input flag");
    }
    void *route_blob = NULL;
    if (work.route_bytes != 0) {
        route_blob = malloc(work.route_bytes);
        if (!route_blob) {
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "out of memory reading distributed route");
        }
        rc = dist_mem_read(&reader, route_blob, work.route_bytes);
        if (rc <= 0) {
            DIST_DEBUG("worker route read failed request=%llu rc=%d bytes=%u",
                       (unsigned long long)request_id,
                       rc,
                       work.route_bytes);
            free(route_blob);
            free(tokens);
            return -1;
        }
        DIST_DEBUG("worker route read ok request=%llu bytes=%u",
                   (unsigned long long)request_id,
                   work.route_bytes);
    }

    ds4_dist_route_entry current_route;
    ds4_dist_route_entry next_route;
    const bool has_route = work.route_count != 0;
    const bool has_next = has_route && work.route_index + 1u < work.route_count;
    if (has_route) {
        if (!dist_route_validate_blob(route_blob, work.route_bytes, work.route_count,
                                      n_layers,
                                      err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, err);
        }
        if (!dist_route_get_entry(route_blob, work.route_bytes, work.route_count,
                                  work.route_index, &current_route, err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, err);
        }
        if (current_route.layer_start != work.layer_start ||
            current_route.layer_end != work.layer_end) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "WORK layer range does not match route entry");
        }
        const bool route_output_logits = (current_route.flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) != 0;
        if (route_output_logits != output_logits) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "WORK logits flag does not match route entry");
        }
        if (has_next &&
            !dist_route_get_entry(route_blob, work.route_bytes, work.route_count,
                                  work.route_index + 1u, &next_route, err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, err);
        }
    }
    if (has_next && output_logits) {
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "non-final route entry requested logits");
    }
    if (has_route && !has_next) {
        ds4_dist_route_return ret;
        if (!dist_route_get_return_target(route_blob, work.route_bytes, work.route_count,
                                          &ret, err, sizeof(err))) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, err);
        }
        if (ret.kind != DS4_DIST_ROUTE_RETURN_UPSTREAM) {
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "unsupported final result destination");
        }
    }

    const bool final_ack_only = ack_only && !has_next;
    const bool local_output_logits = output_logits && !has_next && !final_ack_only;
    const bool produce_hidden = !local_output_logits && !final_ack_only;
    const uint32_t result_kind = final_ack_only
        ? DS4_DIST_RESULT_ACK
        : (local_output_logits ? DS4_DIST_RESULT_LOGITS : DS4_DIST_RESULT_HIDDEN_STATE);
    const uint32_t result_bytes = final_ack_only
        ? 0u
        : (local_output_logits
            ? (uint32_t)((uint64_t)ds4_engine_vocab_size(state->engine) * sizeof(float))
            : expected_hc_bytes);
    float *result = result_bytes ? malloc(result_bytes) : NULL;
    if (result_bytes && !result) {
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "out of memory allocating distributed result");
    }

    const double decode_t0 = profile ? dist_now_sec() : 0.0;
    bool input_hc_uses_wire = false;
    uint32_t input_hc_decoded_bytes = 0;
    if (input_hc_present &&
        dist_decode_activation_payload(input_hc_wire,
                                       input_hc_bits,
                                       work.input_hc_bytes,
                                       &input_hc,
                                       &input_hc_decoded_bytes,
                                       &input_hc_uses_wire,
                                       err,
                                       sizeof(err)) != 0) {
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if (input_hc_present && input_hc_decoded_bytes != expected_hc_bytes) {
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "decoded input hidden-state size does not match token span");
    }
    const double decode_t1 = profile ? dist_now_sec() : 0.0;

    const double lock_t0 = profile ? dist_now_sec() : 0.0;
    pthread_mutex_lock(&state->mu);
    const double lock_t1 = profile ? dist_now_sec() : 0.0;
    ds4_dist_worker_session *session = dist_worker_get_session_locked(state, session_id, err, sizeof(err));
    if (!session) {
        pthread_mutex_unlock(&state->mu);
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if ((work.flags & DS4_DIST_WORK_F_RESET_SESSION) != 0 &&
        ds4_session_layer_slice_reset(session->session, err, sizeof(err)) != 0) {
        pthread_mutex_unlock(&state->mu);
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }
    if ((work.flags & DS4_DIST_WORK_F_RESET_SESSION) != 0) {
        session->token_hash = DS4_DIST_TOKEN_HASH_INIT;
        session->token_hash_valid = true;
    } else if (!session->token_hash_valid) {
        const ds4_tokens *timeline = ds4_session_tokens(session->session);
        if (!timeline || timeline->len < 0) {
            pthread_mutex_unlock(&state->mu);
            if (!input_hc_uses_wire) free(input_hc);
            free(result);
            free(route_blob);
            free(tokens);
            return dist_worker_upstream_send_work_error(upstream, request_id, "worker session has no token timeline");
        }
        session->token_hash = dist_token_hash_prefix(timeline->v, (uint32_t)timeline->len);
        session->token_hash_valid = true;
    }
    if (session->token_hash != work_prefix_hash) {
        pthread_mutex_unlock(&state->mu);
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "worker KV prefix hash mismatch");
    }
    const double eval_t0 = dist_now_sec();
    int eval_rc = ds4_session_eval_layer_slice(session->session,
                                               tokens,
                                               work.n_tokens,
                                               work.pos0,
                                               work.layer_start,
                                               work.layer_end,
                                               input_hc,
                                               produce_hidden ? result : NULL,
                                               local_output_logits,
                                               local_output_logits ? result : NULL,
                                               err,
                                               sizeof(err));
    const double eval_t1 = dist_now_sec();
    if (eval_rc == 0) {
        session->token_hash = work_result_hash;
        session->token_hash_valid = true;
    } else {
        session->token_hash_valid = false;
    }
    pthread_mutex_unlock(&state->mu);
    DIST_DEBUG("worker eval request=%llu layers=%u:%u tokens=%u pos=%u has_next=%d output=%d rc=%d",
               (unsigned long long)request_id,
               work.layer_start,
               work.layer_end,
               work.n_tokens,
               work.pos0,
               has_next ? 1 : 0,
               local_output_logits ? 1 : 0,
               eval_rc);

    if (eval_rc != 0) {
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, err);
    }

    uint32_t result_wire_bytes = result_bytes;
    if (result_kind == DS4_DIST_RESULT_HIDDEN_STATE &&
        !dist_activation_wire_bytes_from_f32_bytes(input_hc_bits,
                                                   result_bytes,
                                                   &result_wire_bytes)) {
        if (!input_hc_uses_wire) free(input_hc);
        free(result);
        free(route_blob);
        free(tokens);
        return dist_worker_upstream_send_work_error(upstream, request_id, "invalid output hidden-state size");
    }

    ds4_dist_telemetry_fixed telemetry = {
        .layer_start = work.layer_start,
        .layer_end = work.layer_end,
        .route_index = work.route_index,
        .pos0 = work.pos0,
        .n_tokens = work.n_tokens,
        .eval_usec = dist_usec_since(eval_t0, eval_t1),
        .downstream_wait_usec = 0,
        .forward_send_usec = 0,
        .input_bytes = work.token_bytes + work.input_hc_bytes,
        .output_bytes = result_wire_bytes,
    };

    const double send_t0 = profile ? dist_now_sec() : 0.0;
    int send_rc;
    if (has_next) {
        send_rc = dist_forward_work_to_next(upstream,
                                            &next_route,
                                            &work,
                                            tokens,
                                            result,
                                            result_bytes,
                                            &telemetry,
                                            route_blob);
    } else {
        send_rc = dist_worker_upstream_send_work_result(upstream,
                                                        request_id,
                                                        work_result_hash,
                                                        0,
                                                        result_kind,
                                                        result_kind == DS4_DIST_RESULT_HIDDEN_STATE ? input_hc_bits : 32u,
                                                        &telemetry,
                                                        1,
                                                        result,
                                                        result_bytes);
    }
    const double send_t1 = profile ? dist_now_sec() : 0.0;
    if (profile) {
        fprintf(stderr,
                "ds4: dist decode profile: worker request=%llu pos=%u layers=%u:%u input_decode=%.3fms lock_wait=%.3fms eval=%.3fms send=%.3fms total=%.3fms input=%.2fMiB output=%.2fMiB rc=%d\n",
                (unsigned long long)request_id,
                work.pos0,
                work.layer_start,
                work.layer_end,
                (decode_t1 - decode_t0) * 1000.0,
                (lock_t1 - lock_t0) * 1000.0,
                (eval_t1 - eval_t0) * 1000.0,
                (send_t1 - send_t0) * 1000.0,
                (send_t1 - total_t0) * 1000.0,
                (double)(work.token_bytes + work.input_hc_bytes) / (1024.0 * 1024.0),
                (double)result_wire_bytes / (1024.0 * 1024.0),
                send_rc);
    }
    DIST_DEBUG("worker send complete request=%llu has_next=%d send_rc=%d",
               (unsigned long long)request_id,
               has_next ? 1 : 0,
               send_rc);
    if (!input_hc_uses_wire) free(input_hc);
    free(result);
    free(route_blob);
    free(tokens);
    return send_rc;
}

static int dist_worker_handle_work(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes) {
    void *payload = malloc(bytes);
    if (!payload) {
        dist_discard_bytes(upstream->fd, bytes);
        return dist_worker_upstream_send_work_error(upstream, 0, "out of memory reading distributed WORK frame");
    }
    int rc = dist_read_full(upstream->fd, payload, bytes);
    if (rc <= 0) {
        free(payload);
        return rc == 0 ? 0 : -1;
    }
    rc = dist_worker_process_work_payload(state, upstream, payload, bytes);
    free(payload);
    return rc;
}

