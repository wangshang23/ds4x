#include "engine_internal.h"

/* Session State module. */
uint64_t ds4_session_payload_bytes(ds4_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    uint64_t bytes =
        (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += 2u * (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(
            &s->graph, (uint32_t)s->checkpoint.len);
    return bytes;
}

int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    if (!payload || !payload->path || !fp) {
        payload_set_err(err, errlen, "invalid staged payload");
        return 1;
    }
    FILE *src = fopen(payload->path, "rb");
    if (!src) {
        payload_set_err(err, errlen, "failed to open staged payload");
        return 1;
    }
    int rc = payload_copy_file_bytes(src, fp, payload->bytes, err, errlen);
    if (fclose(src) != 0 && rc == 0) rc = 1;
    return rc;
}

void ds4_session_payload_file_free(ds4_session_payload_file *payload) {
    if (!payload) return;
    if (payload->path) {
        unlink(payload->path);
        free(payload->path);
    }
    memset(payload, 0, sizeof(*payload));
}

int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen) {
    if (!s || !out || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "invalid payload staging request");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    char path[] = "/tmp/ds4x-session.XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) {
        payload_set_err(err, errlen, "failed to create staged payload");
        return 1;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        close(fd);
        unlink(path);
        payload_set_err(err, errlen, "failed to open staged payload");
        return 1;
    }
    int rc = ds4_session_save_payload(s, fp, err, errlen);
    off_t bytes = rc == 0 ? ftello(fp) : -1;
    if (fclose(fp) != 0 && rc == 0) rc = 1;
    if (rc != 0 || bytes < 0) {
        unlink(path);
        return 1;
    }
    out->path = ds4_strdup(path);
    out->bytes = (uint64_t)bytes;
    return 0;
}

int ds4_session_save_payload(ds4_session *s, FILE *fp,
                             char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint");
        return 1;
    }
    if (!ds4_gpu_synchronize()) {
        payload_set_err(err, errlen, "failed to synchronize before snapshot");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(
            g, (uint32_t)s->checkpoint.len);
    const uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC,
        DS4_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        DS4_N_LAYER,
        DS4_N_HEAD_DIM,
        DS4_N_INDEXER_HEAD_DIM,
        DS4_N_VOCAB,
        raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i],
                              err, errlen) != 0) return 1;
    }
    if (payload_write_bytes(fp, s->logits,
                            (uint64_t)DS4_N_VOCAB * sizeof(float),
                            err, errlen) != 0) return 1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) {
            return 1;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il],
                              err, errlen) != 0) return 1;
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        const uint32_t raw_first = (uint32_t)s->checkpoint.len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t phys = (raw_first + r) % g->raw_cap;
            rc = payload_write_tensor_span(
                    fp, g->layer_raw_cache[il],
                    (uint64_t)phys * DS4_SPARK_KV_ROW_BYTES,
                    DS4_SPARK_KV_ROW_BYTES, buf, DS4_SESSION_IO_CHUNK,
                    err, errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        rc = payload_write_tensor_span(
                fp, g->layer_attn_comp_cache[il], 0,
                (uint64_t)g->layer_n_comp[il] * DS4_SPARK_KV_ROW_BYTES,
                buf, DS4_SESSION_IO_CHUNK, err, errlen);
        if (rc == 0) rc = payload_write_tensor_span(
                fp, g->layer_attn_state_kv[il], 0,
                layer_attn_state_bytes(ratio),
                buf, DS4_SESSION_IO_CHUNK, err, errlen);
        if (rc == 0) rc = payload_write_tensor_span(
                fp, g->layer_attn_state_score[il], 0,
                layer_attn_state_bytes(ratio),
                buf, DS4_SESSION_IO_CHUNK, err, errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_tensor_span(
                    fp, g->layer_index_comp_cache[il], 0,
                    (uint64_t)g->layer_n_index_comp[il] *
                        DS4_SPARK_INDEX_ROW_BYTES,
                    buf, DS4_SESSION_IO_CHUNK, err, errlen);
            if (rc == 0) rc = payload_write_tensor_span(
                    fp, g->layer_index_state_kv[il], 0,
                    layer_index_state_bytes(ratio),
                    buf, DS4_SESSION_IO_CHUNK, err, errlen);
            if (rc == 0) rc = payload_write_tensor_span(
                    fp, g->layer_index_state_score[il], 0,
                    layer_index_state_bytes(ratio),
                    buf, DS4_SESSION_IO_CHUNK, err, errlen);
        }
    }
    free(buf);
    return rc;
}

int ds4_session_load_payload(ds4_session *s, FILE *fp,
                             uint64_t payload_bytes,
                             char *err, size_t errlen) {
    if (!s || !fp) {
        payload_set_err(err, errlen, "invalid payload load");
        return 1;
    }
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_PAYLOAD_VERSION ||
        h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
        h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB) {
        payload_set_err(err, errlen, "incompatible session payload");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_raw_live = h[12];
    const uint32_t expected_raw_live =
        saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_ctx > (uint32_t)s->ctx_size ||
        saved_tokens >= (uint32_t)s->ctx_size ||
        saved_raw_window != g->raw_window || saved_raw_cap == 0 ||
        saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap ||
        saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "session payload does not fit this graph");
        return 1;
    }

    token_vec checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t token = 0;
        if (payload_read_u32(fp, &token, &remaining, err, errlen) != 0 ||
            token >= DS4_N_VOCAB) {
            token_vec_free(&checkpoint);
            payload_set_err(err, errlen, "invalid token in session payload");
            return 1;
        }
        token_vec_push(&checkpoint, (int)token);
    }
    if (payload_read_bytes(fp, s->logits,
                           (uint64_t)DS4_N_VOCAB * sizeof(float),
                           &remaining, err, errlen) != 0) {
        token_vec_free(&checkpoint);
        return 1;
    }

    uint32_t n_comp[DS4_MAX_LAYER];
    uint32_t n_index_comp[DS4_MAX_LAYER];
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0 ||
            n_comp[il] > saved_comp_cap ||
            n_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&checkpoint);
            payload_set_err(err, errlen, "invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_index_comp[il],
                             &remaining, err, errlen) != 0 ||
            n_index_comp[il] > saved_comp_cap ||
            n_index_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&checkpoint);
            payload_set_err(err, errlen, "invalid indexer row count");
            return 1;
        }
    }
    if (!ds4_gpu_synchronize()) {
        token_vec_free(&checkpoint);
        payload_set_err(err, errlen, "failed to synchronize before restore");
        return 1;
    }

    s->checkpoint_valid = false;
    ds4_session_dspark_capture_invalidate(s);
    metal_graph_dspark_cache_reset(g);
    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t phys = (raw_first + r) % g->raw_cap;
            rc = payload_read_tensor_span(
                    fp, g->layer_raw_cache[il],
                    (uint64_t)phys * DS4_SPARK_KV_ROW_BYTES,
                    DS4_SPARK_KV_ROW_BYTES, buf, DS4_SESSION_IO_CHUNK,
                    &remaining, err, errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        rc = payload_read_tensor_span(
                fp, g->layer_attn_comp_cache[il], 0,
                (uint64_t)n_comp[il] * DS4_SPARK_KV_ROW_BYTES,
                buf, DS4_SESSION_IO_CHUNK, &remaining, err, errlen);
        if (rc == 0) rc = payload_read_tensor_span(
                fp, g->layer_attn_state_kv[il], 0,
                layer_attn_state_bytes(ratio), buf, DS4_SESSION_IO_CHUNK,
                &remaining, err, errlen);
        if (rc == 0) rc = payload_read_tensor_span(
                fp, g->layer_attn_state_score[il], 0,
                layer_attn_state_bytes(ratio), buf, DS4_SESSION_IO_CHUNK,
                &remaining, err, errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_tensor_span(
                    fp, g->layer_index_comp_cache[il], 0,
                    (uint64_t)n_index_comp[il] * DS4_SPARK_INDEX_ROW_BYTES,
                    buf, DS4_SESSION_IO_CHUNK, &remaining, err, errlen);
            if (rc == 0) rc = payload_read_tensor_span(
                    fp, g->layer_index_state_kv[il], 0,
                    layer_index_state_bytes(ratio), buf, DS4_SESSION_IO_CHUNK,
                    &remaining, err, errlen);
            if (rc == 0) rc = payload_read_tensor_span(
                    fp, g->layer_index_state_score[il], 0,
                    layer_index_state_bytes(ratio), buf, DS4_SESSION_IO_CHUNK,
                    &remaining, err, errlen);
        }
    }
    free(buf);
    if (rc != 0 || remaining != 0 || !ds4_gpu_synchronize()) {
        token_vec_free(&checkpoint);
        if (rc == 0) payload_set_err(err, errlen, "invalid restored payload");
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = checkpoint;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_n_comp[il] = n_comp[il];
        g->layer_n_index_comp[il] = n_index_comp[il];
    }
    s->checkpoint_valid = true;
    ds4_session_dspark_capture_invalidate(s);
    return 0;
}

int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap,
                              char *err, size_t errlen) {
    if (!s || !snap) return 1;
    const uint64_t bytes = ds4_session_payload_bytes(s);
    if (bytes == 0 || bytes > SIZE_MAX) return 1;
    if (snap->cap < bytes) {
        uint8_t *p = realloc(snap->ptr, (size_t)bytes);
        if (!p) return 1;
        snap->ptr = p;
        snap->cap = bytes;
    }
    FILE *fp = fmemopen(snap->ptr, (size_t)bytes, "wb");
    if (!fp) return 1;
    const int rc = ds4_session_save_payload(s, fp, err, errlen);
    const int close_rc = fclose(fp);
    if (rc != 0 || close_rc != 0) return 1;
    snap->len = bytes;
    return 0;
}

int ds4_session_load_snapshot(ds4_session *s,
                              const ds4_session_snapshot *snap,
                              char *err, size_t errlen) {
    if (!s || !snap || !snap->ptr || snap->len == 0 || snap->len > SIZE_MAX) {
        return 1;
    }
    FILE *fp = fmemopen((void *)snap->ptr, (size_t)snap->len, "rb");
    if (!fp) return 1;
    const int rc = ds4_session_load_payload(s, fp, snap->len, err, errlen);
    const int close_rc = fclose(fp);
    return rc != 0 || close_rc != 0;
}

void ds4_session_snapshot_free(ds4_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}

int ds4_engine_generate_argmax(
        ds4_engine        *e,
        const ds4_tokens  *prompt,
        int                n_predict,
        int                ctx_size,
        ds4_token_emit_fn  emit,
        ds4_generation_done_fn done,
        void              *emit_ud,
        ds4_session_progress_fn progress,
        void              *progress_ud) {
    if (!e || !prompt || n_predict < 0 || ctx_size <= 0) return 1;

    ds4_session *session = NULL;
    if (ds4_session_create(&session, e, ctx_size) != 0) {
        fprintf(stderr, "ds4: failed to create CUDA session\n");
        return 1;
    }
    ds4_session_set_progress(session, progress, progress_ud);

    char err[256] = {0};
    const double prefill_t0 = now_sec();
    int rc = ds4_session_sync(session, prompt, err, sizeof(err));
    const double prefill_t1 = now_sec();
    ds4_session_set_progress(session, NULL, NULL);
    if (rc != 0) {
        fprintf(stderr, "ds4: prefill failed: %s\n", err);
        ds4_session_free(session);
        return rc;
    }

    int generated = 0;
    const double decode_t0 = now_sec();
    while (generated < n_predict &&
           ds4_session_pos(session) + 1 < ds4_session_ctx(session)) {
        const int token = ds4_session_argmax(session);
        if (token < 0) {
            fprintf(stderr, "ds4: argmax failed\n");
            rc = 1;
            break;
        }
        if (ds4_token_is_stop(e, token)) break;
        if (emit) emit(emit_ud, token);
        generated++;
        if (generated >= n_predict ||
            ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
            break;
        }
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: decode failed: %s\n", err);
            rc = 1;
            break;
        }
    }
    const double decode_t1 = now_sec();
    if (done) done(emit_ud);
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill %.2f tok/s, generation %.2f tok/s\n",
            prefill_t1 > prefill_t0
                ? (double)prompt->len / (prefill_t1 - prefill_t0)
                : 0.0,
            decode_t1 > decode_t0
                ? (double)generated / (decode_t1 - decode_t0)
                : 0.0);
    ds4_session_free(session);
    return rc;
}
