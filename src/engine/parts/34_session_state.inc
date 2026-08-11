uint64_t ds4_session_payload_bytes(ds4_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    if (s->distributed) return 0;
    if (ds4_session_is_cpu(s)) {
        uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
        bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
        bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
        bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
        bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
        bytes += session_cpu_payload_live_tensor_bytes(s);
        return bytes;
    }
    if (ds4_session_is_glm(s)) {
#ifdef DS4_NO_GPU
        return 0;
#else
        if (!s->glm_graph_ready) return 0;
        const ds4_glm_gpu_graph *g = &s->glm_graph;
        const uint32_t checkpoint_len = (uint32_t)s->checkpoint.len;
        const uint32_t full_live =
            session_glm_full_live_rows(g, checkpoint_len, s->glm_dense_cache_len);
        const uint32_t compact_live = session_glm_compact_live_rows(g, checkpoint_len);
        uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
        bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
        bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
        bytes += (uint64_t)g->normal_layers * sizeof(uint32_t);
        bytes += (uint64_t)g->normal_layers * sizeof(uint32_t);
        bytes += session_glm_payload_live_tensor_bytes(g, full_live, compact_live);
        return bytes;
#endif
    }
#ifdef DS4_NO_GPU
    return 0;
#else
    const ds4_gpu_graph *g = &s->graph;
    uint64_t bytes = (uint64_t)DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_VOCAB * sizeof(float);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += (uint64_t)DS4_N_LAYER * sizeof(uint32_t);
    bytes += session_payload_live_tensor_bytes(g, (uint32_t)s->checkpoint.len);
    return bytes;
#endif
}

int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    if (!payload || !payload->path || !fp) {
        payload_set_err(err, errlen, "invalid staged session payload");
        return 1;
    }
    FILE *src = fopen(payload->path, "rb");
    if (!src) {
        payload_set_err(err, errlen, "failed to open staged session payload");
        return 1;
    }
    int rc = payload_copy_file_bytes(src, fp, payload->bytes, err, errlen);
    if (fclose(src) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        return 1;
    }
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
    if (!out) {
        payload_set_err(err, errlen, "invalid session payload staging request");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    if (!s || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to stage");
        return 1;
    }

    char tmpl[] = "/tmp/ds4-session-payload.XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        payload_set_err(err, errlen, "failed to create staged session payload");
        return 1;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        int saved = errno;
        close(fd);
        unlink(tmpl);
        if (errlen) snprintf(err, errlen, "failed to open staged session payload: %s",
                             strerror(saved));
        return 1;
    }

    int rc = ds4_session_save_payload(s, fp, err, errlen);
    if (rc == 0 && fflush(fp) != 0) {
        payload_set_err(err, errlen, "failed to flush staged session payload");
        rc = 1;
    }
    off_t pos = -1;
    if (rc == 0) {
        pos = ftello(fp);
        if (pos < 0) {
            payload_set_err(err, errlen, "failed to measure staged session payload");
            rc = 1;
        }
    }
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close staged session payload");
        rc = 1;
    }
    if (rc != 0) {
        unlink(tmpl);
        return 1;
    }
    out->path = ds4_strdup(tmpl);
    out->bytes = (uint64_t)pos;
    return 0;
}

int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid) {
        payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    if (s->distributed) {
        return ds4_dist_session_save_payload(s->distributed, s, fp, err, errlen);
    }
    if (ds4_session_is_glm(s)) {
#ifdef DS4_NO_GPU
        payload_set_err(err, errlen, "graph backend support is not compiled in");
        return 1;
#else
        if (!s->glm_graph_ready) {
            payload_set_err(err, errlen, "GLM graph is not ready for snapshot");
            return 1;
        }
        if (ds4_gpu_synchronize() == 0) {
            payload_set_err(err, errlen, "failed to synchronize accelerator before GLM snapshot");
            return 1;
        }

        ds4_glm_gpu_graph *g = &s->glm_graph;
        const uint32_t checkpoint_len = (uint32_t)s->checkpoint.len;
        const uint32_t full_live =
            session_glm_full_live_rows(g, checkpoint_len, s->glm_dense_cache_len);
        const uint32_t compact_live = session_glm_compact_live_rows(g, checkpoint_len);
        uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
            DS4_SESSION_PAYLOAD_MAGIC,
            DS4_SESSION_PAYLOAD_VERSION,
            (uint32_t)s->ctx_size,
            s->prefill_cap,
            g->ctx_cap,
            g->ctx_cap,
            g->compact_cache_cap,
            checkpoint_len,
            g->normal_layers,
            DS4_N_KEY_MLA,
            DS4_N_VALUE_MLA,
            DS4_N_VOCAB,
            full_live,
        };
        for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
            if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
        }
        for (int i = 0; i < s->checkpoint.len; i++) {
            if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
        }
        if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
        for (uint32_t il = 0; il < g->normal_layers; il++) {
            if (payload_write_u32(fp, compact_live, err, errlen) != 0) return 1;
        }
        for (uint32_t il = 0; il < g->normal_layers; il++) {
            const uint32_t index_rows =
                compact_live != 0 && glm_graph_layer_uses_full_indexer(il) ? compact_live : 0;
            if (payload_write_u32(fp, index_rows, err, errlen) != 0) return 1;
        }

        uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
        int rc = 0;
        for (uint32_t il = 0; rc == 0 && il < g->normal_layers; il++) {
            rc = payload_write_glm_full_kv_span(fp,
                                                g->layer_key_cache[il],
                                                (uint64_t)full_live * g->q_dim,
                                                buf,
                                                DS4_SESSION_IO_CHUNK,
                                                err,
                                                errlen);
            if (rc == 0) {
                rc = payload_write_glm_full_kv_span(fp,
                                                    g->layer_value_cache[il],
                                                    (uint64_t)full_live * g->heads_dim,
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
            }
            if (rc != 0 || compact_live == 0) continue;
            rc = payload_write_glm_compact_span(fp,
                                                g->layer_kv_lora_cache[il],
                                                (uint64_t)compact_live * DS4_N_KV_LORA,
                                                buf,
                                                DS4_SESSION_IO_CHUNK,
                                                err,
                                                errlen);
            if (rc == 0) {
                rc = payload_write_glm_compact_span(fp,
                                                    g->layer_k_rope_cache[il],
                                                    (uint64_t)compact_live * DS4_N_ROT,
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
            }
            if (rc == 0 && glm_graph_layer_uses_full_indexer(il)) {
                rc = payload_write_glm_compact_span(fp,
                                                    g->layer_indexer_key_cache[il],
                                                    (uint64_t)compact_live * DS4_N_INDEXER_HEAD_DIM,
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
            }
        }
        free(buf);
        return rc;
#endif
    }
    if (ds4_session_is_cpu(s)) {
        const uint32_t raw_live = session_cpu_raw_live_rows(s);
        const uint32_t raw_cap = ds4_default_raw_cap((uint32_t)s->ctx_size);
        const uint32_t comp_cap = session_cpu_comp_cap(s);
        uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
            DS4_SESSION_PAYLOAD_MAGIC,
            DS4_SESSION_PAYLOAD_VERSION,
            (uint32_t)s->ctx_size,
            s->prefill_cap,
            raw_cap,
            raw_cap,
            comp_cap,
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
            if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
        }
        if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_write_u32(fp, s->cpu_cache.layer[il].n_comp, err, errlen) != 0) return 1;
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_write_u32(fp, s->cpu_cache.layer[il].n_index_comp, err, errlen) != 0) return 1;
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
            if (raw_live > layer->n_raw) {
                payload_set_err(err, errlen, "CPU session raw cache has fewer live rows than checkpoint");
                return 1;
            }
            const uint32_t raw_start = layer->n_raw - raw_live;
            if (payload_write_bytes(fp,
                                    layer->raw_kv + (uint64_t)raw_start * DS4_N_HEAD_DIM,
                                    (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float),
                                    err,
                                    errlen) != 0) return 1;
            const uint32_t ratio = layer->compress_ratio;
            if (ratio == 0) continue;
            if (payload_write_bytes(fp,
                                    layer->attn_comp_kv,
                                    (uint64_t)layer->n_comp * DS4_N_HEAD_DIM * sizeof(float),
                                    err,
                                    errlen) != 0) return 1;
            if (payload_write_bytes(fp, layer->attn_state_kv, layer_attn_state_bytes(ratio), err, errlen) != 0) return 1;
            if (payload_write_bytes(fp, layer->attn_state_score, layer_attn_state_bytes(ratio), err, errlen) != 0) return 1;
            if (ratio == 4) {
                if (payload_write_bytes(fp,
                                        layer->index_comp_kv,
                                        (uint64_t)layer->n_index_comp * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                        err,
                                        errlen) != 0) return 1;
                if (payload_write_bytes(fp, layer->index_state_kv, layer_index_state_bytes(ratio), err, errlen) != 0) return 1;
                if (payload_write_bytes(fp, layer->index_state_score, layer_index_state_bytes(ratio), err, errlen) != 0) return 1;
            }
        }
        return 0;
    }
#ifdef DS4_NO_GPU
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before snapshot");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    /* Header fields:
     *   0 magic, 1 version, 2 ctx, 3 prefill chunk, 4 raw cap,
     *   5 raw window, 6 compressed cap, 7 token count,
     *   8 layers, 9 raw head dim, 10 indexer head dim, 11 vocab,
     *   12 live raw rows serialized below.
     */
    uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
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
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen) != 0) return 1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il], err, errlen) != 0) return 1;
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Write the raw ring in logical position order.  The file does not care
         * where the rows happened to live physically in the source graph. */
        const uint32_t raw_first = (uint32_t)s->checkpoint.len - raw_live;
        for (uint32_t r = 0; rc == 0 && r < raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_write_tensor_span(fp,
                                           g->layer_raw_cache[il],
                                           (uint64_t)phys * DS4_SPARK_KV_ROW_BYTES,
                                           DS4_SPARK_KV_ROW_BYTES,
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        /* Compressed rows are append-only from row zero, so the live prefix is
         * contiguous.  The two compressor state tensors hold the partial window
         * that will become the next compressed row. */
        rc = payload_write_tensor_span(fp,
                                       g->layer_attn_comp_cache[il],
                                       0,
                                       (uint64_t)g->layer_n_comp[il] * DS4_SPARK_KV_ROW_BYTES,
                                       buf,
                                       DS4_SESSION_IO_CHUNK,
                                       err,
                                       errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_kv[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0) rc = payload_write_tensor_span(fp,
                                                    g->layer_attn_state_score[il],
                                                    0,
                                                    layer_attn_state_bytes(ratio),
                                                    buf,
                                                    DS4_SESSION_IO_CHUNK,
                                                    err,
                                                    errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_write_tensor_span(fp,
                                           g->layer_index_comp_cache[il],
                                           0,
                                           (uint64_t)g->layer_n_index_comp[il] * DS4_SPARK_INDEX_ROW_BYTES,
                                           buf,
                                           DS4_SESSION_IO_CHUNK,
                                           err,
                                           errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_kv[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
            if (rc == 0) rc = payload_write_tensor_span(fp,
                                                        g->layer_index_state_score[il],
                                                        0,
                                                        layer_index_state_bytes(ratio),
                                                        buf,
                                                        DS4_SESSION_IO_CHUNK,
                                                        err,
                                                        errlen);
        }
    }
    free(buf);
    return rc;
#endif
}

int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    if (!s || !fp) {
        payload_set_err(err, errlen, "invalid session payload load");
        return 1;
    }
    if (s->distributed) {
        return ds4_dist_session_load_payload(s->distributed, s, fp, payload_bytes, err, errlen);
    }
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    if (ds4_session_is_glm(s)) {
#ifdef DS4_NO_GPU
        payload_set_err(err, errlen, "graph backend support is not compiled in");
        return 1;
#else
        if (!s->glm_graph_ready) {
            payload_set_err(err, errlen, "GLM graph is not ready for KV restore");
            return 1;
        }
        ds4_glm_gpu_graph *g = &s->glm_graph;
        const uint32_t saved_ctx = h[2];
        const uint32_t saved_prefill_cap = h[3];
        const uint32_t saved_full_cap = h[4];
        const uint32_t saved_full_window = h[5];
        const uint32_t saved_compact_cap = h[6];
        const uint32_t saved_tokens = h[7];
        const uint32_t saved_full_live = h[12];
        if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
            payload_set_err(err, errlen, "KV checkpoint does not fit current context");
            return 1;
        }
        if (h[8] != g->normal_layers || h[9] != DS4_N_KEY_MLA ||
            h[10] != DS4_N_VALUE_MLA || h[11] != DS4_N_VOCAB)
        {
            payload_set_err(err, errlen, "KV checkpoint was written for a different GLM layout");
            return 1;
        }
        (void)saved_prefill_cap;
        if (saved_full_cap == 0 || saved_full_cap > g->ctx_cap ||
            saved_full_window != saved_full_cap)
        {
            payload_set_err(err, errlen, "KV checkpoint GLM full-cache layout does not match current runtime");
            return 1;
        }
        const uint32_t max_full_live =
            saved_tokens < saved_full_cap ? saved_tokens : saved_full_cap;
        if (saved_full_live > max_full_live || saved_full_live > g->ctx_cap) {
            payload_set_err(err, errlen, "KV checkpoint GLM full-cache row count is invalid");
            return 1;
        }
        if (saved_compact_cap > g->ctx_size) {
            payload_set_err(err, errlen, "KV checkpoint GLM compact cache is larger than current context");
            return 1;
        }
        if (g->compact_cache_cap != 0 && saved_tokens != 0 && saved_compact_cap == 0) {
            payload_set_err(err, errlen, "KV checkpoint lacks the GLM compact cache required by this context");
            return 1;
        }
        const uint32_t expected_compact_live =
            saved_compact_cap != 0
                ? (saved_tokens < saved_compact_cap ? saved_tokens : saved_compact_cap)
                : 0;
        if (expected_compact_live != 0) {
            if (!s->engine ||
                !glm_graph_ensure_compact_cache(g, expected_compact_live)) {
                payload_set_err(err, errlen, "KV checkpoint GLM compact cache could not be allocated");
                return 1;
            }
        }
        if (expected_compact_live > g->compact_cache_cap) {
            payload_set_err(err, errlen, "KV checkpoint GLM compact row count is invalid");
            return 1;
        }

        token_vec new_checkpoint = {0};
        for (uint32_t i = 0; i < saved_tokens; i++) {
            uint32_t tok = 0;
            if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            token_vec_push(&new_checkpoint, (int)tok);
        }
        if (payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                               &remaining, err, errlen) != 0)
        {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        uint32_t n_comp[DS4_MAX_LAYER];
        uint32_t n_index_comp[DS4_MAX_LAYER];
        for (uint32_t il = 0; il < g->normal_layers; il++) {
            if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (n_comp[il] != expected_compact_live) {
                token_vec_free(&new_checkpoint);
                payload_set_err(err, errlen, "KV checkpoint GLM compact row count does not match token count");
                return 1;
            }
        }
        for (uint32_t il = 0; il < g->normal_layers; il++) {
            if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            const uint32_t expected_index_rows =
                expected_compact_live != 0 && glm_graph_layer_uses_full_indexer(il)
                    ? expected_compact_live
                    : 0;
            if (n_index_comp[il] != expected_index_rows) {
                token_vec_free(&new_checkpoint);
                payload_set_err(err, errlen, "KV checkpoint GLM indexer row count does not match token count");
                return 1;
            }
        }

        if (ds4_gpu_synchronize() == 0) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "failed to synchronize accelerator before GLM KV restore");
            return 1;
        }
        s->checkpoint_valid = false;
        s->mtp_draft_valid = false;
        ds4_session_glm_reset_dense_cache(s);

        uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
        int rc = 0;
        for (uint32_t il = 0; rc == 0 && il < g->normal_layers; il++) {
            rc = payload_read_or_skip_glm_full_kv_span(fp,
                                                       g->layer_key_cache[il],
                                                       (uint64_t)saved_full_live * g->q_dim,
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) {
                rc = payload_read_or_skip_glm_full_kv_span(fp,
                                                           g->layer_value_cache[il],
                                                           (uint64_t)saved_full_live * g->heads_dim,
                                                           buf,
                                                           DS4_SESSION_IO_CHUNK,
                                                           &remaining,
                                                           err,
                                                           errlen);
            }
            if (rc != 0 || expected_compact_live == 0) continue;
            rc = payload_read_glm_compact_span(fp,
                                               g->layer_kv_lora_cache[il],
                                               (uint64_t)expected_compact_live * DS4_N_KV_LORA,
                                               buf,
                                               DS4_SESSION_IO_CHUNK,
                                               &remaining,
                                               err,
                                               errlen);
            if (rc == 0) {
                rc = payload_read_glm_compact_span(fp,
                                                   g->layer_k_rope_cache[il],
                                                   (uint64_t)expected_compact_live * DS4_N_ROT,
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
            }
            if (rc == 0 && glm_graph_layer_uses_full_indexer(il)) {
                rc = payload_read_glm_compact_span(fp,
                                                   g->layer_indexer_key_cache[il],
                                                   (uint64_t)expected_compact_live * DS4_N_INDEXER_HEAD_DIM,
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
            }
        }
        free(buf);
        if (rc != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (remaining != 0) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
            return 1;
        }
        if (ds4_gpu_synchronize() == 0) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "failed to synchronize accelerator after GLM KV restore");
            return 1;
        }

        token_vec_free(&s->checkpoint);
        s->checkpoint = new_checkpoint;
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        s->glm_dense_cache_len = g->full_kv_cache ? saved_full_live : 0;
        return 0;
#endif
    }
    if (ds4_session_is_cpu(s)) {
        const uint32_t saved_ctx = h[2];
        const uint32_t saved_prefill_cap = h[3];
        const uint32_t saved_raw_cap = h[4];
        const uint32_t saved_raw_window = h[5];
        const uint32_t saved_comp_cap = h[6];
        const uint32_t saved_tokens = h[7];
        const uint32_t saved_raw_live = h[12];
        const uint32_t cpu_raw_cap = ds4_default_raw_cap((uint32_t)s->ctx_size);
        const uint32_t cpu_comp_cap = session_cpu_comp_cap(s);
        if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
            payload_set_err(err, errlen, "KV checkpoint does not fit current context");
            return 1;
        }
        if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
            h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB)
        {
            payload_set_err(err, errlen, "KV checkpoint was written for a different DS4 layout");
            return 1;
        }
        /* prefill_cap is scratch scheduling capacity, not durable KV layout.
         * Old checkpoints remain valid as long as the raw KV window matches. */
        (void)saved_prefill_cap;
        if (saved_raw_window != cpu_raw_cap) {
            payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
            return 1;
        }
        const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
        if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
            saved_raw_live > saved_raw_cap || saved_raw_live > cpu_raw_cap)
        {
            payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
            return 1;
        }
        if (saved_comp_cap > cpu_comp_cap) {
            payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
            return 1;
        }

        token_vec new_checkpoint = {0};
        for (uint32_t i = 0; i < saved_tokens; i++) {
            uint32_t tok = 0;
            if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            token_vec_push(&new_checkpoint, (int)tok);
        }
        if (payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                               &remaining, err, errlen) != 0)
        {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        uint32_t n_comp[DS4_MAX_LAYER];
        uint32_t n_index_comp[DS4_MAX_LAYER];
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (n_comp[il] > saved_comp_cap || n_comp[il] > cpu_comp_cap) {
                token_vec_free(&new_checkpoint);
                payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
                return 1;
            }
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > cpu_comp_cap) {
                token_vec_free(&new_checkpoint);
                payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
                return 1;
            }
        }

        s->checkpoint_valid = false;
        s->mtp_draft_valid = false;
        ds4_session_dspark_capture_invalidate(s);
        session_cpu_reset_cache(s);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            ds4_layer_cache *layer = &s->cpu_cache.layer[il];
            if (payload_read_bytes(fp,
                                   layer->raw_kv,
                                   (uint64_t)saved_raw_live * DS4_N_HEAD_DIM * sizeof(float),
                                   &remaining,
                                   err,
                                   errlen) != 0)
            {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            layer->n_raw = saved_raw_live;
            const uint32_t ratio = layer->compress_ratio;
            if (ratio == 0) continue;
            layer->n_comp = n_comp[il];
            layer->n_index_comp = n_index_comp[il];
            if (payload_read_bytes(fp,
                                   layer->attn_comp_kv,
                                   (uint64_t)n_comp[il] * DS4_N_HEAD_DIM * sizeof(float),
                                   &remaining,
                                   err,
                                   errlen) != 0 ||
                payload_read_bytes(fp, layer->attn_state_kv, layer_attn_state_bytes(ratio), &remaining, err, errlen) != 0 ||
                payload_read_bytes(fp, layer->attn_state_score, layer_attn_state_bytes(ratio), &remaining, err, errlen) != 0)
            {
                token_vec_free(&new_checkpoint);
                return 1;
            }
            if (ratio == 4) {
                if (payload_read_bytes(fp,
                                       layer->index_comp_kv,
                                       (uint64_t)n_index_comp[il] * DS4_N_INDEXER_HEAD_DIM * sizeof(float),
                                       &remaining,
                                       err,
                                       errlen) != 0 ||
                    payload_read_bytes(fp, layer->index_state_kv, layer_index_state_bytes(ratio), &remaining, err, errlen) != 0 ||
                    payload_read_bytes(fp, layer->index_state_score, layer_index_state_bytes(ratio), &remaining, err, errlen) != 0)
                {
                    token_vec_free(&new_checkpoint);
                    return 1;
                }
            }
        }
        if (remaining != 0) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
            return 1;
        }
        token_vec_free(&s->checkpoint);
        s->checkpoint = new_checkpoint;
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        ds4_session_dspark_capture_invalidate(s);
        return 0;
    }
#ifdef DS4_NO_GPU
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    ds4_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_raw_live = h[12];
    if (saved_ctx > (uint32_t)s->ctx_size || saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
        h[10] != DS4_N_INDEXER_HEAD_DIM || h[11] != DS4_N_VOCAB)
    {
        payload_set_err(err, errlen, "KV checkpoint was written for a different DS4 layout");
        return 1;
    }
    /* prefill_cap is scratch scheduling capacity, not durable KV layout.
     * Old checkpoints remain valid as long as the raw KV window matches. */
    (void)saved_prefill_cap;
    if (saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV checkpoint graph chunk layout does not match current runtime");
        return 1;
    }
    /* The raw rows in the file are logical rows.  We can restore them into any
     * current ring with enough capacity, but the saved live count must be exactly
     * the last window implied by the saved token count. */
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap)
    {
        payload_set_err(err, errlen, "KV checkpoint raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV checkpoint compressed cache is larger than current context");
        return 1;
    }

    token_vec new_checkpoint = {0};
    for (uint32_t i = 0; i < saved_tokens; i++) {
        uint32_t tok = 0;
        if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        token_vec_push(&new_checkpoint, (int)tok);
    }
    if (payload_read_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * sizeof(float),
                           &remaining, err, errlen) != 0)
    {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    uint32_t n_comp[DS4_MAX_LAYER];
    uint32_t n_index_comp[DS4_MAX_LAYER];
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_comp[il] > saved_comp_cap || n_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (payload_read_u32(fp, &n_index_comp[il], &remaining, err, errlen) != 0) {
            token_vec_free(&new_checkpoint);
            return 1;
        }
        if (n_index_comp[il] > saved_comp_cap || n_index_comp[il] > g->layer_comp_cap[il]) {
            token_vec_free(&new_checkpoint);
            payload_set_err(err, errlen, "KV checkpoint has invalid indexer row count");
            return 1;
        }
    }

    if (ds4_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator before KV restore");
        return 1;
    }
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    ds4_session_dspark_capture_invalidate(s);
    g->mtp_n_raw = 0;
    metal_graph_dspark_cache_reset(g);

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = 0; rc == 0 && il < DS4_N_LAYER; il++) {
        /* Rebuild the physical raw ring expected by the current graph.  This is
         * why the file stores rows in logical order instead of dumping bytes from
         * the old ring layout. */
        const uint32_t raw_first = saved_tokens - saved_raw_live;
        for (uint32_t r = 0; rc == 0 && r < saved_raw_live; r++) {
            const uint32_t pos = raw_first + r;
            const uint32_t phys = pos % g->raw_cap;
            rc = payload_read_tensor_span(fp,
                                          g->layer_raw_cache[il],
                                          (uint64_t)phys * DS4_SPARK_KV_ROW_BYTES,
                                          DS4_SPARK_KV_ROW_BYTES,
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (rc != 0 || ratio == 0) continue;
        rc = payload_read_tensor_span(fp,
                                      g->layer_attn_comp_cache[il],
                                      0,
                                      (uint64_t)n_comp[il] * DS4_SPARK_KV_ROW_BYTES,
                                      buf,
                                      DS4_SESSION_IO_CHUNK,
                                      &remaining,
                                      err,
                                      errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_kv[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0) rc = payload_read_tensor_span(fp,
                                                   g->layer_attn_state_score[il],
                                                   0,
                                                   layer_attn_state_bytes(ratio),
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
        if (rc == 0 && ratio == 4) {
            rc = payload_read_tensor_span(fp,
                                          g->layer_index_comp_cache[il],
                                          0,
                                          (uint64_t)n_index_comp[il] * DS4_SPARK_INDEX_ROW_BYTES,
                                          buf,
                                          DS4_SESSION_IO_CHUNK,
                                          &remaining,
                                          err,
                                          errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_kv[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
            if (rc == 0) rc = payload_read_tensor_span(fp,
                                                       g->layer_index_state_score[il],
                                                       0,
                                                       layer_index_state_bytes(ratio),
                                                       buf,
                                                       DS4_SESSION_IO_CHUNK,
                                                       &remaining,
                                                       err,
                                                       errlen);
        }
    }
    free(buf);
    if (rc != 0) {
        token_vec_free(&new_checkpoint);
        return 1;
    }
    if (remaining != 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "KV checkpoint has trailing payload bytes");
        return 1;
    }
    if (ds4_gpu_synchronize() == 0) {
        token_vec_free(&new_checkpoint);
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV restore");
        return 1;
    }

    token_vec_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g->layer_n_comp[il] = n_comp[il];
        g->layer_n_index_comp[il] = n_index_comp[il];
    }
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    ds4_session_dspark_capture_invalidate(s);
    g->mtp_n_raw = 0;
    metal_graph_dspark_cache_reset(g);
    return 0;
#endif
}

int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap) {
        payload_set_err(err, errlen, "invalid session snapshot save");
        return 1;
    }
    if (s->distributed) {
        payload_set_err(err, errlen, "distributed session snapshots are not supported yet");
        return 1;
    }
    const uint64_t bytes = ds4_session_payload_bytes(s);
    if (bytes == 0) {
        payload_set_err(err, errlen, "session has no valid checkpoint to snapshot");
        return 1;
    }
    if (bytes > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }
    if (snap->cap < bytes) {
        uint8_t *p = realloc(snap->ptr, (size_t)bytes);
        if (!p) {
            payload_set_err(err, errlen, "out of memory while allocating session snapshot");
            return 1;
        }
        snap->ptr = p;
        snap->cap = bytes;
    }

    FILE *fp = fmemopen(snap->ptr, (size_t)bytes, "wb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot");
        return 1;
    }
    const int rc = ds4_session_save_payload(s, fp, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to finalize memory session snapshot");
        return 1;
    }
    if (rc != 0) return 1;
    snap->len = bytes;
    return 0;
}

int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen) {
    if (!s || !snap || !snap->ptr || snap->len == 0) {
        payload_set_err(err, errlen, "invalid session snapshot load");
        return 1;
    }
    if (s->distributed) {
        payload_set_err(err, errlen, "distributed session snapshots are not supported yet");
        return 1;
    }
    if (snap->len > (uint64_t)SIZE_MAX) {
        payload_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }

    FILE *fp = fmemopen((void *)snap->ptr, (size_t)snap->len, "rb");
    if (!fp) {
        payload_set_err(err, errlen, "failed to open memory stream for session snapshot restore");
        return 1;
    }
    const int rc = ds4_session_load_payload(s, fp, snap->len, err, errlen);
    if (fclose(fp) != 0 && rc == 0) {
        payload_set_err(err, errlen, "failed to close memory session snapshot");
        return 1;
    }
    return rc;
}

void ds4_session_snapshot_free(ds4_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}

void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens) {
    dump_tokens(&e->vocab, tokens);
}

int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    ds4_model model;
    ds4_vocab vocab;
    token_vec tokens = {0};

    if (!fp) fp = stdout;
    model_open(&model, model_path, false, false);
    config_validate_model(&model);
    vocab_load(&vocab, &model);
    tokenize_rendered_chat_vocab(&vocab, text ? text : "", &tokens);

    dump_tokens_fp(fp, &vocab, &tokens);
    token_vec_free(&tokens);
    vocab_free(&vocab);
    model_close(&model);
    return 0;
}

#ifndef DS4_NO_GPU
static bool imatrix_read_text_file(const char *path, char **out, size_t *len_out) {
    *out = NULL;
    *len_out = 0;
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "ds4: failed to stat imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size > SIZE_MAX - 1) {
        fprintf(stderr, "ds4: imatrix dataset is too large: %s\n", path);
        return false;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open imatrix dataset %s: %s\n", path, strerror(errno));
        return false;
    }
    size_t n = (size_t)st.st_size;
    char *buf = xmalloc(n + 1);
    if (n != 0 && fread(buf, 1, n, fp) != n) {
        fprintf(stderr, "ds4: failed to read imatrix dataset %s\n", path);
        fclose(fp);
        free(buf);
        return false;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close imatrix dataset %s: %s\n", path, strerror(errno));
        free(buf);
        return false;
    }
    buf[n] = '\0';
    *out = buf;
    *len_out = n;
    return true;
}

static char *imatrix_trim_block(char *p, char *end) {
    while (p < end && isspace((unsigned char)*p)) p++;
    while (end > p && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return p;
}

static char *imatrix_find_marker(char *dataset, char *cursor, const char *marker) {
    const size_t marker_len = strlen(marker);
    char *p = cursor;
    while ((p = strstr(p, marker)) != NULL) {
        if (p == dataset || p[-1] == '\n') return p;
        p += marker_len;
    }
    return NULL;
}
#endif

int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)dataset_path;
    (void)output_path;
    (void)ctx_size;
    (void)max_prompts;
    (void)max_tokens;
    fprintf(stderr, "ds4: imatrix collection requires a graph backend build\n");
    return 1;
#else
    if (!e || !dataset_path || !output_path) return 1;
    if (e->backend != DS4_BACKEND_METAL || !e->metal_ready) {
        fprintf(stderr, "ds4: imatrix collection currently requires --metal\n");
        return 1;
    }
    if (ctx_size <= 0) ctx_size = 32768;

    char *dataset = NULL;
    size_t dataset_len = 0;
    if (!imatrix_read_text_file(dataset_path, &dataset, &dataset_len)) return 1;

    const ds4_model *model = &e->model;
    const ds4_weights *weights = &e->weights;
    const uint32_t prefill_cap =
        metal_graph_prefill_cap_for_prompt(ctx_size, e->prefill_chunk);
    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, prefill_cap);

    ds4_gpu_graph g;
    /* diagnostic single-tier callsite; placement=NULL. */
    bool ok = metal_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size,
                                        prefill_cap, false, NULL, false, NULL);
    if (!ok) {
        fprintf(stderr, "ds4: failed to allocate imatrix Metal graph runtime\n");
        free(dataset);
        return 1;
    }
    g.quality = e->quality;
    g.ssd_streaming = e->ssd_streaming;
    g.ssd_streaming_cold = e->ssd_streaming_cold;
    g.streaming_preload_experts = e->ssd_streaming_preload_experts;
    g.power_percent = (uint32_t)e->power_percent;

    ds4_imatrix_collector collector;
    if (!imatrix_collector_init(&collector, prefill_cap, dataset_path)) {
        fprintf(stderr, "ds4: failed to allocate imatrix collector\n");
        metal_graph_free(&g);
        free(dataset);
        return 1;
    }

    fprintf(stderr,
            "ds4: collecting routed-MoE imatrix from %s (model=%s, layers=%u, experts=%u, ctx=%d, chunk=%u)\n",
            dataset_path, DS4_MODEL_SHAPE_NAME, DS4_N_LAYER, DS4_N_EXPERT, ctx_size, prefill_cap);

    int prompts_done = 0;
    int tokens_done = 0;
    char *cursor = dataset;
    const char *marker_lit = "===== DS4_IMATRIX_PROMPT";
    while (*cursor) {
        char *start = cursor;
        char *marker = imatrix_find_marker(dataset, cursor, marker_lit);
        if (marker) {
            char *nl = strchr(marker, '\n');
            if (!nl) break;
            start = nl + 1;
        } else if (prompts_done != 0) {
            break;
        }

        char *next = imatrix_find_marker(dataset, start, marker_lit);
        char *end = next ? next : dataset + dataset_len;
        char saved = *end;
        char *prompt_text = imatrix_trim_block(start, end);
        if (prompt_text[0] != '\0') {
            token_vec prompt = {0};
            ds4_tokenize_rendered_chat(e, prompt_text, &prompt);
            if (prompt.len > ctx_size) prompt.len = ctx_size;
            if (max_tokens > 0 && prompt.len > max_tokens - tokens_done) {
                prompt.len = max_tokens - tokens_done;
            }
            if (prompt.len > 0) {
                if (!metal_graph_reset_prefill_state(&g)) {
                    fprintf(stderr, "ds4: failed to reset imatrix graph state\n");
                    ok = false;
                } else if ((uint32_t)prompt.len > prefill_cap) {
                    ok = metal_graph_prefill_chunked_range(&g, model, weights,
                                                           &prompt, 0,
                                                           (uint32_t)prompt.len,
                                                           NULL, false,
                                                           NULL, NULL,
                                                           NULL, NULL,
                                                           &collector,
                                                           NULL, NULL, NULL);
                } else {
                    ok = metal_graph_prefill_layer_major(&g, model, weights,
                                                         &prompt, 0,
                                                         (uint32_t)prompt.len,
                                                         NULL, false,
                                                         &collector,
                                                         NULL, NULL);
                }
                if (!ok) {
                    fprintf(stderr, "ds4: imatrix prefill failed at prompt %d\n", prompts_done + 1);
                    token_vec_free(&prompt);
                    *end = saved;
                    break;
                }
                prompts_done++;
                tokens_done += prompt.len;
                if (prompts_done % 10 == 0) {
                    fprintf(stderr,
                            "ds4: imatrix prompts=%d tokens=%d routes=%llu\r",
                            prompts_done,
                            tokens_done,
                            (unsigned long long)collector.observed_routes);
                    fflush(stderr);
                }
            }
            token_vec_free(&prompt);
        }
        *end = saved;
        if (!next) break;
        cursor = next;
        if (max_prompts > 0 && prompts_done >= max_prompts) break;
        if (max_tokens > 0 && tokens_done >= max_tokens) break;
    }
    fputc('\n', stderr);

    if (ok) {
        ok = imatrix_collector_save(&collector, weights, output_path);
        if (ok) {
            fprintf(stderr,
                    "ds4: wrote imatrix %s from %d prompts, %d tokens, %llu routed expert observations\n",
                    output_path,
                    prompts_done,
                    tokens_done,
                    (unsigned long long)collector.observed_routes);
        }
    }

    imatrix_collector_free(&collector);
    metal_graph_free(&g);
    free(dataset);
    return ok ? 0 : 1;
#endif
}

#ifndef DS4_NO_GPU
static bool ds4_session_greedy_splitkv_replay_exact(
        ds4_session *s,
        int          current_token,
        int         *top,
        char        *err,
        size_t       errlen) {
    if (!s || !s->greedy_splitkv_anchor_valid) return false;
    ds4_engine *e = s->engine;
    const int start = s->greedy_splitkv_anchor_len;
    if (!spec_frontier_restore(&s->greedy_splitkv_anchor, s)) {
        snprintf(err, errlen, "%s split-kv frontier restore failed",
                 ds4_backend_name(e->backend));
        return false;
    }
    s->checkpoint.len = start;
    s->checkpoint_valid = true;

    int replay_top = -1;
    const bool trust_replay =
        metal_graph_cuda_greedy_splitkv_trust_replay_requested();
    if (metal_graph_cuda_greedy_splitkv_pair_replay_requested()) {
        const int n_seq = s->greedy_splitkv_segment.len + 1;
        int *seq = xmalloc((size_t)n_seq * sizeof(seq[0]));
        for (int i = 0; i < s->greedy_splitkv_segment.len; i++) {
            seq[i] = s->greedy_splitkv_segment.v[i];
        }
        seq[n_seq - 1] = current_token;

        for (int i = 0; i < n_seq; ) {
            if (i + 1 < n_seq) {
                int top0 = -1;
                int top1 = -1;
                const bool second_is_current = (i + 1 == n_seq - 1);
                const uint32_t pos = (uint32_t)s->checkpoint.len;
                bool ok = metal_graph_verify_decode2_exact(&s->graph,
                                                           &e->model,
                                                           &e->weights,
                                                           seq[i],
                                                           seq[i + 1],
                                                           pos,
                                                           &top0,
                                                           &top1,
                                                           NULL,
                                                           NULL);
                if (!ok) {
                    snprintf(err, errlen, "%s split-kv paired exact replay failed",
                             ds4_backend_name(e->backend));
                    s->checkpoint_valid = false;
                    free(seq);
                    return false;
                }
                if (!trust_replay && top0 != seq[i + 1]) {
                    snprintf(err, errlen,
                             "%s split-kv paired exact replay mismatch at pos=%d",
                             ds4_backend_name(e->backend),
                             start + i);
                    s->checkpoint_valid = false;
                    free(seq);
                    return false;
                }
                if (!trust_replay && !second_is_current &&
                    i + 2 < n_seq && top1 != seq[i + 2])
                {
                    snprintf(err, errlen,
                             "%s split-kv paired exact replay mismatch at pos=%d",
                             ds4_backend_name(e->backend),
                             start + i + 1);
                    s->checkpoint_valid = false;
                    free(seq);
                    return false;
                }
                token_vec_push(&s->checkpoint, seq[i]);
                if (!second_is_current) {
                    token_vec_push(&s->checkpoint, seq[i + 1]);
                } else {
                    *top = top1;
                }
                i += 2;
                continue;
            }

            if (!metal_graph_eval_token_raw_swa_top(&s->graph,
                                                    &e->model,
                                                    &e->weights,
                                                    seq[i],
                                                    (uint32_t)s->checkpoint.len,
                                                    top,
                                                    NULL,
                                                    false,
                                                    NULL,
                                                    false)) {
                snprintf(err, errlen, "%s split-kv paired exact replay current token failed",
                         ds4_backend_name(e->backend));
                s->checkpoint_valid = false;
                free(seq);
                return false;
            }
            i++;
        }

        free(seq);
        s->greedy_splitkv_segment.len = 0;
        s->greedy_splitkv_anchor_valid = false;
        s->greedy_splitkv_anchor_len = 0;
        return true;
    }
    for (int i = 0; i < s->greedy_splitkv_segment.len; i++) {
        const int tok = s->greedy_splitkv_segment.v[i];
        bool ok = false;
        if (trust_replay) {
            /* Opt-in experiment: rebuild exact state for already-emitted
             * tokens without validating their intermediate greedy tops. */
            ok = metal_graph_eval_token_raw_swa(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                tok,
                                                (uint32_t)s->checkpoint.len,
                                                NULL);
        } else {
            ok = metal_graph_eval_token_raw_swa_top(&s->graph,
                                                    &e->model,
                                                    &e->weights,
                                                    tok,
                                                    (uint32_t)s->checkpoint.len,
                                                    &replay_top,
                                                    NULL,
                                                    false,
                                                    NULL,
                                                    false);
        }
        if (!ok) {
            snprintf(err, errlen, "%s split-kv exact replay failed",
                     ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            return false;
        }
        token_vec_push(&s->checkpoint, tok);
        if (!trust_replay &&
            i + 1 < s->greedy_splitkv_segment.len &&
            replay_top != s->greedy_splitkv_segment.v[i + 1])
        {
            snprintf(err, errlen,
                     "%s split-kv exact replay mismatch at pos=%d",
                     ds4_backend_name(e->backend),
                     start + i);
            s->checkpoint_valid = false;
            return false;
        }
    }
    if (!trust_replay &&
        s->greedy_splitkv_segment.len > 0 &&
        replay_top != current_token) {
        snprintf(err, errlen,
                 "%s split-kv exact replay mismatch before current token",
                 ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return false;
    }
    if (!metal_graph_eval_token_raw_swa_top(&s->graph,
                                            &e->model,
                                            &e->weights,
                                            current_token,
                                            (uint32_t)s->checkpoint.len,
                                            top,
                                            NULL,
                                            false,
                                            NULL,
                                            false)) {
        snprintf(err, errlen, "%s split-kv exact replay current token failed",
                 ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        return false;
    }

    s->greedy_splitkv_segment.len = 0;
    s->greedy_splitkv_anchor_valid = false;
    s->greedy_splitkv_anchor_len = 0;
    return true;
}
#endif

int ds4_session_eval_argmax(ds4_session *s, int token, char *err, size_t errlen) {
    if (!s) return -1;
    if (ds4_session_is_cpu(s) || ds4_session_is_glm(s)) {
        if (ds4_session_eval(s, token, err, errlen) != 0) return -1;
        return ds4_session_argmax(s);
    }
#ifdef DS4_NO_GPU
    (void)token;
    snprintf(err, errlen, "GPU support is not compiled in");
    return -1;
#else
    ds4_engine *e = s->engine;
    int top = -1;
    const uint32_t pos = (uint32_t)s->checkpoint.len;
    const bool splitkv_may_engage =
        metal_graph_cuda_greedy_splitkv_may_engage(&s->graph, pos);
    const bool splitkv_fallback =
        splitkv_may_engage && metal_graph_cuda_greedy_splitkv_fallback_requested();
    const bool splitkv_top2 =
        splitkv_may_engage && metal_graph_cuda_greedy_splitkv_top2_requested();
    const bool greedy_vec4 =
        !splitkv_may_engage && metal_graph_cuda_greedy_vec4_requested();
    const bool vec4_fallback =
        greedy_vec4 && metal_graph_cuda_greedy_vec4_fallback_requested();
    const bool approx_fallback = splitkv_fallback || vec4_fallback;

    bool anchor_ready = false;
    if (approx_fallback) {
        if (!s->greedy_splitkv_anchor_valid) {
            s->greedy_splitkv_anchor_len = (int)pos;
            s->greedy_splitkv_segment.len = 0;
            s->greedy_splitkv_anchor_valid =
                spec_frontier_snapshot(&s->greedy_splitkv_anchor, s);
            if (!s->greedy_splitkv_anchor_valid) {
                s->greedy_splitkv_anchor_len = 0;
            }
        }
        anchor_ready = s->greedy_splitkv_anchor_valid;
        if (!anchor_ready &&
            ((splitkv_fallback && getenv("DS4_CUDA_GREEDY_SPLITKV_FALLBACK_LOG")) ||
             (vec4_fallback && getenv("DS4_CUDA_GREEDY_VEC4_FALLBACK_LOG"))))
        {
            fprintf(stderr,
                    "ds4: greedy %s frontier snapshot failed at pos=%u; using exact attention\n",
                    vec4_fallback ? "vec4" : "split-kv",
                    pos);
        }
    }

    metal_graph_top2_result top2;
    memset(&top2, 0, sizeof(top2));
    const bool allow_fast_attention = !splitkv_fallback || anchor_ready;
    const bool use_vec4_attention = greedy_vec4 && (!vec4_fallback || anchor_ready);
    const int old_score_vec4 =
        ds4_gpu_set_decode_score_vec4(use_vec4_attention ? 1 : 0);
    bool ok = metal_graph_eval_token_raw_swa_top(&s->graph,
                                                 &e->model,
                                                 &e->weights,
                                                 (uint32_t)token,
                                                 pos,
                                                 &top,
                                                 NULL,
                                                 allow_fast_attention,
                                                 (anchor_ready || splitkv_top2) ? &top2 : NULL,
                                                 false);
    (void)ds4_gpu_set_decode_score_vec4(old_score_vec4);
    bool replayed_exact = false;
    if (!ok && anchor_ready) {
        ok = ds4_session_greedy_splitkv_replay_exact(s, token, &top, err, errlen);
        replayed_exact = ok;
    }
    if (ok && anchor_ready && top2.fast_attention) {
        const float margin = top2.value0 - top2.value1;
        const float threshold = metal_graph_cuda_greedy_splitkv_margin_threshold();
        if (getenv("DS4_CUDA_GREEDY_SPLITKV_TRACE")) {
            fprintf(stderr,
                    "ds4: greedy split-kv margin pos=%u top=%d second=%d margin=%.6f threshold=%.6f\n",
                    pos,
                    top2.id0,
                    top2.id1,
                    (double)margin,
                    (double)threshold);
        }
        if (!top2.valid || !isfinite(margin) || margin < threshold) {
            const int replay_segment_len = s->greedy_splitkv_segment.len;
            ok = ds4_session_greedy_splitkv_replay_exact(s, token, &top, err, errlen);
            replayed_exact = ok;
            if (ok && getenv("DS4_CUDA_GREEDY_SPLITKV_FALLBACK_LOG")) {
                fprintf(stderr,
                        "ds4: greedy split-kv exact replay pos=%u segment=%d margin=%.6f threshold=%.6f approx_top=%d second=%d exact_top=%d\n",
                        pos,
                        replay_segment_len,
                        (double)margin,
                        (double)threshold,
                        top2.id0,
                        top2.id1,
                        top);
            }
        }
    }
    if (ok && vec4_fallback && anchor_ready && use_vec4_attention) {
        const float margin = top2.value0 - top2.value1;
        const float threshold = metal_graph_cuda_greedy_vec4_margin_threshold();
        if (getenv("DS4_CUDA_GREEDY_VEC4_TRACE")) {
            fprintf(stderr,
                    "ds4: greedy vec4 margin pos=%u top=%d second=%d margin=%.6f threshold=%.6f\n",
                    pos,
                    top2.id0,
                    top2.id1,
                    (double)margin,
                    (double)threshold);
        }
        if (!top2.valid || !isfinite(margin) || margin < threshold) {
            const int replay_segment_len = s->greedy_splitkv_segment.len;
            ok = ds4_session_greedy_splitkv_replay_exact(s, token, &top, err, errlen);
            replayed_exact = ok;
            if (ok && getenv("DS4_CUDA_GREEDY_VEC4_FALLBACK_LOG")) {
                fprintf(stderr,
                        "ds4: greedy vec4 exact replay pos=%u segment=%d margin=%.6f threshold=%.6f approx_top=%d second=%d exact_top=%d\n",
                        pos,
                        replay_segment_len,
                        (double)margin,
                        (double)threshold,
                        top2.id0,
                        top2.id1,
                        top);
            }
        }
    }
    if (ok && approx_fallback && anchor_ready && !replayed_exact) {
        const uint32_t max_segment = vec4_fallback
            ? metal_graph_cuda_greedy_vec4_max_segment()
            : metal_graph_cuda_greedy_splitkv_max_segment();
        if (max_segment > 0 &&
            (uint32_t)s->greedy_splitkv_segment.len >= max_segment) {
            const int replay_segment_len = s->greedy_splitkv_segment.len;
            ok = ds4_session_greedy_splitkv_replay_exact(s, token, &top, err, errlen);
            replayed_exact = ok;
            if (ok &&
                ((vec4_fallback && getenv("DS4_CUDA_GREEDY_VEC4_FALLBACK_LOG")) ||
                 (splitkv_fallback && getenv("DS4_CUDA_GREEDY_SPLITKV_FALLBACK_LOG"))))
            {
                fprintf(stderr,
                        "ds4: greedy %s exact replay pos=%u segment=%d reason=max_segment max=%u approx_top=%d second=%d exact_top=%d\n",
                        vec4_fallback ? "vec4" : "split-kv",
                        pos,
                        replay_segment_len,
                        max_segment,
                        top2.id0,
                        top2.id1,
                        top);
            }
        }
    }
    if (!ok) {
        if (errlen != 0 && err[0] == '\0') {
            snprintf(err, errlen, "%s decode failed", ds4_backend_name(e->backend));
        }
        s->checkpoint_valid = false;
        return -1;
    }
    token_vec_push(&s->checkpoint, token);
    if (approx_fallback && anchor_ready && !replayed_exact) {
        token_vec_push(&s->greedy_splitkv_segment, token);
    } else if (!approx_fallback || !anchor_ready) {
        session_greedy_splitkv_reset(s);
    }
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    return top;
#endif
}

#ifndef DS4_NO_GPU
static int ds4_session_eval_splitkv_spec_after_first(
        ds4_session *s,
        int          max_extra_tokens,
        int          eos_token,
        int         *accepted,
        int          accepted_cap,
        char        *err,
        size_t       errlen) {
    if (!s || max_extra_tokens < 2 || accepted_cap < 2) return 0;
    if (!metal_graph_cuda_splitkv_spec_requested()) return 0;

    ds4_engine *e = s->engine;
    const uint32_t start = (uint32_t)s->checkpoint.len;
    if ((uint64_t)start + 2u > (uint64_t)s->ctx_size) {
        if (getenv("DS4_CUDA_SPLITKV_SPEC_LOG")) {
            fprintf(stderr,
                    "ds4: split-kv spec skip pos=%u reason=context-room\n",
                    start);
        }
        return 0;
    }
    if (!metal_graph_cuda_splitkv_score_may_engage(&s->graph, start)) {
        if (getenv("DS4_CUDA_SPLITKV_SPEC_LOG")) {
            fprintf(stderr,
                    "ds4: split-kv spec skip pos=%u reason=score-gate\n",
                    start);
        }
        return 0;
    }

    const int draft0 = sample_argmax(s->logits, DS4_N_VOCAB);
    if (draft0 < 0 || draft0 == eos_token) {
        if (getenv("DS4_CUDA_SPLITKV_SPEC_LOG")) {
            fprintf(stderr,
                    "ds4: split-kv spec skip pos=%u reason=%s draft0=%d\n",
                    start,
                    draft0 < 0 ? "no-draft0" : "eos",
                    draft0);
        }
        return 0;
    }

    ds4_spec_frontier frontier;
    memset(&frontier, 0, sizeof(frontier));
    if (!spec_frontier_snapshot(&frontier, s)) {
        if (getenv("DS4_CUDA_SPLITKV_SPEC_LOG")) {
            fprintf(stderr,
                    "ds4: split-kv spec frontier snapshot failed at pos=%u; using exact decode\n",
                    start);
        }
        return 0;
    }

    const bool timing = getenv("DS4_CUDA_SPLITKV_SPEC_TIMING") != NULL;
    const double t0 = timing ? now_sec() : 0.0;
    int draft1 = -1;
    bool draft_ok = metal_graph_eval_token_raw_swa_top(&s->graph,
                                                       &e->model,
                                                       &e->weights,
                                                       draft0,
                                                       start,
                                                       &draft1,
                                                       NULL,
                                                       true,
                                                       NULL,
                                                       true);
    const double t_draft = timing ? now_sec() : 0.0;
    const bool restored = spec_frontier_restore(&frontier, s);
    const double t_restore = timing ? now_sec() : 0.0;
    if (!restored) {
        snprintf(err, errlen, "%s split-kv spec frontier restore failed",
                 ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        spec_frontier_free(&frontier);
        return -1;
    }
    if (!draft_ok || draft1 < 0) {
        if (getenv("DS4_CUDA_SPLITKV_SPEC_LOG")) {
            fprintf(stderr,
                    "ds4: split-kv spec draft failed at pos=%u; using exact decode\n",
                    start);
        }
        spec_frontier_free(&frontier);
        return 0;
    }

    if (metal_graph_cuda_splitkv_spec_batch_verify_requested()) {
        float *row_logits = xmalloc(
                (size_t)2 * DS4_N_VOCAB * sizeof(row_logits[0]));
        int row0_top = -1;
        s->checkpoint.len = (int)start;
        token_vec_push(&s->checkpoint, draft0);
        token_vec_push(&s->checkpoint, draft1);
        bool ok = metal_graph_verify_suffix_tops(&s->graph,
                                                 &e->model,
                                                 &e->weights,
                                                 &s->checkpoint,
                                                 start,
                                                 2,
                                                 true,
                                                 false,
                                                 &row0_top,
                                                 row_logits,
                                                 NULL);
        const double t_verify = timing ? now_sec() : 0.0;
        if (!ok) {
            s->checkpoint.len = (int)start;
            (void)spec_frontier_restore(&frontier, s);
            snprintf(err, errlen, "%s split-kv spec batch verifier failed",
                     ds4_backend_name(e->backend));
            s->checkpoint_valid = false;
            free(row_logits);
            spec_frontier_free(&frontier);
            return -1;
        }

        int n_accept = 0;
        if (row0_top == draft1) {
            memcpy(s->logits,
                   row_logits + DS4_N_VOCAB,
                   (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
            accepted[n_accept++] = draft0;
            accepted[n_accept++] = draft1;
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
        } else {
            s->checkpoint.len = (int)start;
            if (!spec_frontier_commit_prefix1(s)) {
                (void)spec_frontier_restore(&frontier, s);
                snprintf(err, errlen,
                         "%s split-kv spec batch prefix commit failed",
                         ds4_backend_name(e->backend));
                s->checkpoint_valid = false;
                free(row_logits);
                spec_frontier_free(&frontier);
                return -1;
            }
            memcpy(s->logits,
                   row_logits,
                   (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
            token_vec_push(&s->checkpoint, draft0);
            accepted[n_accept++] = draft0;
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
        }

        if (getenv("DS4_CUDA_SPLITKV_SPEC_LOG")) {
            fprintf(stderr,
                    "ds4: split-kv spec batch pos=%u draft0=%d draft1=%d verify_next=%d accepted=%d\n",
                    start,
                    draft0,
                    draft1,
                    row0_top,
                    n_accept);
        }
        if (timing) {
            const double done = now_sec();
            fprintf(stderr,
                    "ds4: split-kv spec batch timing accepted=%d draft=%.3f ms restore=%.3f ms verify=%.3f ms total=%.3f ms\n",
                    n_accept,
                    (t_draft - t0) * 1000.0,
                    (t_restore - t_draft) * 1000.0,
                    (t_verify - t_restore) * 1000.0,
                    (done - t0) * 1000.0);
        }

        free(row_logits);
        spec_frontier_free(&frontier);
        return n_accept;
    }

    const bool toponly_row0 =
        metal_graph_cuda_splitkv_spec_toponly_row0_requested();
    float *row0_logits = toponly_row0
        ? NULL
        : xmalloc((size_t)DS4_N_VOCAB * sizeof(row0_logits[0]));
    float *row1_logits =
        xmalloc((size_t)DS4_N_VOCAB * sizeof(row1_logits[0]));
    int row0_top = -1;
    bool ok = metal_graph_verify_decode2_exact(&s->graph,
                                               &e->model,
                                               &e->weights,
                                               draft0,
                                               draft1,
                                               start,
                                               &row0_top,
                                               NULL,
                                               row0_logits,
                                               row1_logits);
    const double t_verify = timing ? now_sec() : 0.0;
    if (!ok) {
        (void)spec_frontier_restore(&frontier, s);
        snprintf(err, errlen, "%s split-kv spec exact verifier failed",
                 ds4_backend_name(e->backend));
        s->checkpoint_valid = false;
        free(row1_logits);
        free(row0_logits);
        spec_frontier_free(&frontier);
        return -1;
    }

    int n_accept = 0;
    if (row0_top == draft1) {
        memcpy(s->logits,
               row1_logits,
               (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        token_vec_push(&s->checkpoint, draft0);
        token_vec_push(&s->checkpoint, draft1);
        accepted[n_accept++] = draft0;
        accepted[n_accept++] = draft1;
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
    } else {
        s->checkpoint.len = (int)start;
        if (toponly_row0) {
            if (!spec_frontier_restore(&frontier, s) ||
                !metal_graph_eval_token_raw_swa(&s->graph,
                                                &e->model,
                                                &e->weights,
                                                draft0,
                                                start,
                                                s->logits)) {
                snprintf(err, errlen,
                         "%s split-kv spec row0 exact replay failed",
                         ds4_backend_name(e->backend));
                s->checkpoint_valid = false;
                free(row1_logits);
                spec_frontier_free(&frontier);
                return -1;
            }
        } else {
            if (!spec_frontier_commit_prefix1(s)) {
                (void)spec_frontier_restore(&frontier, s);
                snprintf(err, errlen, "%s split-kv spec prefix commit failed",
                         ds4_backend_name(e->backend));
                s->checkpoint_valid = false;
                free(row1_logits);
                free(row0_logits);
                spec_frontier_free(&frontier);
                return -1;
            }
            memcpy(s->logits,
                   row0_logits,
                   (size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
        }
        token_vec_push(&s->checkpoint, draft0);
        accepted[n_accept++] = draft0;
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
    }

    if (getenv("DS4_CUDA_SPLITKV_SPEC_LOG")) {
        fprintf(stderr,
                "ds4: split-kv spec pos=%u draft0=%d draft1=%d exact_next=%d accepted=%d\n",
                start,
                draft0,
                draft1,
                row0_top,
                n_accept);
    }
    if (timing) {
        const double done = now_sec();
        fprintf(stderr,
                "ds4: split-kv spec timing accepted=%d draft=%.3f ms restore=%.3f ms verify=%.3f ms total=%.3f ms\n",
                n_accept,
                (t_draft - t0) * 1000.0,
                (t_restore - t_draft) * 1000.0,
                (t_verify - t_restore) * 1000.0,
                (done - t0) * 1000.0);
    }

    free(row1_logits);
    free(row0_logits);
    spec_frontier_free(&frontier);
    return n_accept;
}
#endif
