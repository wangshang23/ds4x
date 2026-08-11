/* =========================================================================
 * Session Snapshot Payloads.
 * =========================================================================
 *
 * The server disk cache stores a high-level file header, then delegates the
 * graph-specific payload below to the engine.  This payload is intentionally
 * not mmaped: restoring a checkpoint copies bytes back into the already
 * allocated Metal tensors, preserving the same live graph buffers used by
 * normal prefill/decode.  The raw SWA cache is serialized as the last logical
 * window only; suffix prefill writes its own raw rows before attention.  The
 * compressed caches are serialized up to their live row counts because sparse
 * attention may select rows from the whole prefix.
 *
 * The payload is model-specific rather than self-describing.  The fixed header
 * records enough shape information to reject a file written for a different
 * DS4 runtime, then the body writes: checkpoint tokens, last logits, per-layer
 * compressed row counts, raw SWA rows in logical order, compressed attention
 * rows, and the compressor/indexer frontiers.  That is the minimum state needed
 * for the next token to match a session that had just prefetched the prefix.
 */

#define DS4_SESSION_IO_CHUNK (8u * 1024u * 1024u)

static void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (errlen != 0) snprintf(err, errlen, "%s", msg);
}

static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
}

static uint32_t payload_get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] |
           ((uint32_t)in[1] << 8) |
           ((uint32_t)in[2] << 16) |
           ((uint32_t)in[3] << 24);
}

static int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}

static DS4_MAYBE_UNUSED int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    const uint64_t original = bytes;
    uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to read session payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

static DS4_MAYBE_UNUSED int payload_skip_bytes(FILE *fp, uint64_t bytes,
                                               uint8_t *buf, size_t cap,
                                               uint64_t *remaining,
                                               char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (!buf || cap == 0) {
        payload_set_err(err, errlen, "session payload skip buffer is missing");
        return 1;
    }
    const uint64_t original = bytes;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)cap ? cap : (size_t)bytes;
        if (fread(buf, 1, n, fp) != n) {
            payload_set_err(err, errlen, "failed to skip session payload");
            return 1;
        }
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

static DS4_MAYBE_UNUSED int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

static DS4_MAYBE_UNUSED int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (remaining && *remaining < sizeof(b)) {
        payload_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    if (remaining) *remaining -= sizeof(b);
    *v = payload_get_u32(b);
    return 0;
}

static int payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes, char *err, size_t errlen) {
    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    while (bytes != 0) {
        const size_t n = bytes > DS4_SESSION_IO_CHUNK ? DS4_SESSION_IO_CHUNK : (size_t)bytes;
        if (fread(buf, 1, n, src) != n) {
            payload_set_err(err, errlen, "failed to read staged session payload");
            rc = 1;
            break;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            payload_set_err(err, errlen, "failed to write staged session payload");
            rc = 1;
            break;
        }
        bytes -= n;
    }
    free(buf);
    return rc;
}

static DS4_MAYBE_UNUSED uint64_t layer_attn_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
}

static DS4_MAYBE_UNUSED uint64_t layer_index_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio * sizeof(float);
}

#ifndef DS4_NO_GPU
/* Only the last logical sliding-window rows are needed from the raw cache.
 * The physical Metal tensor is a ring sized for ubatches, but after restore
 * the next suffix chunk will write its own raw rows before any attention read.
 * Compressed rows are different: sparse attention can select any row from the
 * prefix, so those are persisted up to their live row counts. */
static uint32_t session_raw_live_rows(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint32_t rows = g->raw_window ? g->raw_window : DS4_N_SWA;
    if (rows > g->raw_cap) rows = g->raw_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}

/* Return the exact engine-owned payload size, excluding the server's KVC file
 * header and observability text.  This is deliberately based on live row counts
 * rather than capacities so the disk cache scales with saved tokens, not with
 * the maximum context size used to allocate the graph. */
static uint64_t session_payload_live_tensor_bytes(const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_raw_live_rows(g, checkpoint_len);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        bytes += (uint64_t)raw_live * DS4_SPARK_KV_ROW_BYTES;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * DS4_SPARK_KV_ROW_BYTES;
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * DS4_SPARK_INDEX_ROW_BYTES;
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}

static uint32_t session_glm_full_live_rows(const ds4_glm_gpu_graph *g,
                                           uint32_t checkpoint_len,
                                           uint32_t dense_cache_len) {
    if (!g) return 0;
    if (!g->full_kv_cache) return 0;
    uint32_t rows = g->ctx_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    if (rows > dense_cache_len) rows = dense_cache_len;
    return rows;
}

static uint32_t session_glm_compact_live_rows(const ds4_glm_gpu_graph *g, uint32_t checkpoint_len) {
    if (!g || g->compact_cache_cap == 0) return 0;
    uint32_t rows = g->compact_cache_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}

static uint64_t session_glm_payload_live_tensor_bytes(const ds4_glm_gpu_graph *g,
                                                      uint32_t full_live,
                                                      uint32_t compact_live) {
    if (!g) return 0;
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < g->normal_layers; il++) {
        bytes += (uint64_t)full_live * g->q_dim * sizeof(float);
        bytes += (uint64_t)full_live * g->heads_dim * sizeof(float);
        if (compact_live == 0) continue;
        bytes += (uint64_t)compact_live * DS4_N_KV_LORA * sizeof(float);
        bytes += (uint64_t)compact_live * DS4_N_ROT * sizeof(float);
        if (glm_graph_layer_uses_full_indexer(il)) {
            bytes += (uint64_t)compact_live * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
        }
    }
    return bytes;
}

static bool payload_u64_add(uint64_t *acc, uint64_t value) {
    if (!acc || *acc > UINT64_MAX - value) return false;
    *acc += value;
    return true;
}

static bool payload_u64_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out) return false;
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool payload_u64_add_tensor_bytes(uint64_t *acc, uint64_t rows, uint64_t cols) {
    uint64_t elems = 0;
    uint64_t bytes = 0;
    return payload_u64_mul(rows, cols, &elems) &&
           payload_u64_mul(elems, sizeof(float), &bytes) &&
           payload_u64_add(acc, bytes);
}

static bool glm_layer_payload_tensor_bytes(uint32_t layer,
                                           uint32_t full_live,
                                           uint32_t key_dim,
                                           uint32_t value_dim,
                                           uint32_t compact_live,
                                           uint32_t index_live,
                                           uint64_t *out) {
    if (!out || layer >= glm_graph_normal_layer_count() ||
        key_dim != DS4_N_KEY_MLA ||
        value_dim != DS4_N_VALUE_MLA)
        return false;

    const bool has_indexer = glm_graph_layer_uses_full_indexer(layer);
    const uint32_t expected_index_live =
        compact_live != 0 && has_indexer ? compact_live : 0;
    if (index_live != expected_index_live) return false;

    uint64_t bytes = 0;
    if (!payload_u64_add_tensor_bytes(&bytes, full_live,
                                      (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA) ||
        !payload_u64_add_tensor_bytes(&bytes, full_live,
                                      (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA))
        return false;
    if (compact_live != 0) {
        if (!payload_u64_add_tensor_bytes(&bytes, compact_live, DS4_N_KV_LORA) ||
            !payload_u64_add_tensor_bytes(&bytes, compact_live, DS4_N_ROT))
            return false;
        if (has_indexer &&
            !payload_u64_add_tensor_bytes(&bytes, index_live, DS4_N_INDEXER_HEAD_DIM))
            return false;
    }

    *out = bytes;
    return true;
}

/* Accelerator tensors are copied through a fixed-size CPU buffer.  We do not mmap the
 * cache file and we do not allocate a second graph-sized blob just to serialize
 * it; both would be poor fits for this very large model. */
static int payload_write_tensor_span(FILE *fp, const ds4_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (ds4_gpu_tensor_read(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to read accelerator session tensor");
            return 1;
        }
        if (payload_write_bytes(fp, buf, n, err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

static int payload_read_tensor_span(FILE *fp, ds4_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap, uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the payload");
        return 1;
    }
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n = bytes - done > (uint64_t)cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0) return 1;
        if (ds4_gpu_tensor_write(tensor, offset + done, buf, n) == 0) {
            payload_set_err(err, errlen, "failed to restore accelerator session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}

static DS4_MAYBE_UNUSED int payload_write_tensor_span_f16_as_f32(FILE *fp, const ds4_gpu_tensor *tensor,
                                                                 uint64_t offset_f16, uint64_t count,
                                                                 uint8_t *buf, size_t cap, char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > ds4_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > ds4_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (ds4_gpu_tensor_read(tensor, offset_f16 + done * sizeof(uint16_t),
                                h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to read Metal F16 session tensor");
            return 1;
        }
        for (size_t i = 0; i < n; i++) f[i] = f16_to_f32(h[i]);
        if (payload_write_bytes(fp, f, (uint64_t)n * sizeof(float), err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

static DS4_MAYBE_UNUSED int payload_read_tensor_span_f32_as_f16(FILE *fp, ds4_gpu_tensor *tensor,
                                                                uint64_t offset_f16, uint64_t count,
                                                                uint8_t *buf, size_t cap, uint64_t *remaining,
                                                                char *err, size_t errlen) {
    if (!tensor ||
        count > (UINT64_MAX / sizeof(uint16_t)) ||
        count > (UINT64_MAX / sizeof(float)) ||
        offset_f16 > ds4_gpu_tensor_bytes(tensor) ||
        count * sizeof(uint16_t) > ds4_gpu_tensor_bytes(tensor) - offset_f16)
    {
        payload_set_err(err, errlen, "session tensor is smaller than the F16 payload");
        return 1;
    }

    size_t cap_elems = cap / (sizeof(uint16_t) + sizeof(float));
    cap_elems &= ~(size_t)1u;
    if (cap_elems == 0) {
        payload_set_err(err, errlen, "session tensor conversion buffer is too small");
        return 1;
    }
    uint16_t *h = (uint16_t *)buf;
    float *f = (float *)(void *)(buf + cap_elems * sizeof(uint16_t));

    uint64_t done = 0;
    while (done < count) {
        const size_t n = count - done > (uint64_t)cap_elems
            ? cap_elems
            : (size_t)(count - done);
        if (payload_read_bytes(fp, f, (uint64_t)n * sizeof(float), remaining, err, errlen) != 0) return 1;
        for (size_t i = 0; i < n; i++) h[i] = f32_to_f16(f[i]);
        if (ds4_gpu_tensor_write(tensor, offset_f16 + done * sizeof(uint16_t),
                                 h, n * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to restore Metal F16 session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}

static int payload_write_glm_compact_span(FILE *fp, const ds4_gpu_tensor *tensor,
                                          uint64_t count, uint8_t *buf, size_t cap,
                                          char *err, size_t errlen) {
    if (glm_graph_compact_cache_is_f16()) {
        return payload_write_tensor_span_f16_as_f32(fp, tensor, 0, count, buf, cap, err, errlen);
    }
    return payload_write_tensor_span(fp, tensor, 0, count * sizeof(float), buf, cap, err, errlen);
}

static int payload_read_glm_compact_span(FILE *fp, ds4_gpu_tensor *tensor,
                                         uint64_t count, uint8_t *buf, size_t cap,
                                         uint64_t *remaining, char *err, size_t errlen) {
    if (glm_graph_compact_cache_is_f16()) {
        return payload_read_tensor_span_f32_as_f16(fp, tensor, 0, count, buf, cap, remaining, err, errlen);
    }
    return payload_read_tensor_span(fp, tensor, 0, count * sizeof(float), buf, cap, remaining, err, errlen);
}

static int payload_write_glm_full_kv_span(FILE *fp,
                                          const ds4_gpu_tensor *tensor,
                                          uint64_t count,
                                          uint8_t *buf, size_t cap,
                                          char *err, size_t errlen) {
    if (count == 0) return 0;
    return payload_write_tensor_span_f16_as_f32(fp, tensor, 0, count, buf, cap, err, errlen);
}

static int payload_read_glm_full_kv_span(FILE *fp,
                                         ds4_gpu_tensor *tensor,
                                         uint64_t count,
                                         uint8_t *buf, size_t cap,
                                         uint64_t *remaining,
                                         char *err, size_t errlen) {
    if (count == 0) return 0;
    return payload_read_tensor_span_f32_as_f16(fp, tensor, 0, count, buf, cap, remaining, err, errlen);
}

static int payload_read_or_skip_glm_full_kv_span(FILE *fp,
                                                 ds4_gpu_tensor *tensor,
                                                 uint64_t count,
                                                 uint8_t *buf, size_t cap,
                                                 uint64_t *remaining,
                                                 char *err, size_t errlen) {
    if (count == 0) return 0;
    if (tensor) {
        return payload_read_glm_full_kv_span(fp,
                                             tensor,
                                             count,
                                             buf,
                                             cap,
                                             remaining,
                                             err,
                                             errlen);
    }
    if (count > UINT64_MAX / sizeof(float)) {
        payload_set_err(err, errlen, "GLM full KV payload is too large");
        return 1;
    }
    return payload_skip_bytes(fp, count * sizeof(float), buf, cap, remaining, err, errlen);
}
#endif

static bool ds4_session_is_cpu(const ds4_session *s) {
    return s && s->engine && s->engine->backend == DS4_BACKEND_CPU;
}

static void ds4_session_dspark_capture_invalidate(ds4_session *s) {
#ifndef DS4_NO_GPU
    if (!s) return;
    s->dspark_draft_valid = false;
    s->dspark_draft_len = 0;
    if (ds4_session_is_cpu(s)) return;
    metal_graph_dspark_capture_invalidate(&s->graph);
#else
    (void)s;
#endif
}

static void ds4_session_dspark_capture_note_checkpoint(ds4_session *s) {
#ifndef DS4_NO_GPU
    if (!s || ds4_session_is_cpu(s)) return;
    ds4_gpu_graph *g = &s->graph;
    if (!g->dspark_capture_enabled) return;
    if (!s->checkpoint_valid || s->checkpoint.len <= 0 || !g->dspark_capture_valid) {
        metal_graph_dspark_capture_invalidate(g);
        return;
    }
    g->dspark_capture_checkpoint_len = (uint32_t)s->checkpoint.len;
#else
    (void)s;
#endif
}

static bool ds4_session_is_glm(const ds4_session *s) {
    return s && s->engine && DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA;
}

#ifndef DS4_NO_GPU
static void ds4_session_glm_reset_dense_cache(ds4_session *s) {
    if (s) s->glm_dense_cache_len = 0;
}

static void ds4_session_glm_cap_dense_cache(ds4_session *s) {
    if (!s) return;
    if (s->glm_graph_ready && !s->glm_graph.full_kv_cache) {
        s->glm_dense_cache_len = 0;
        return;
    }
    uint32_t cap = s->checkpoint.len > 0 ? (uint32_t)s->checkpoint.len : 0;
    if (s->glm_graph_ready && cap > s->glm_graph.ctx_cap) cap = s->glm_graph.ctx_cap;
    if (s->glm_dense_cache_len > cap) s->glm_dense_cache_len = cap;
}

static void ds4_session_glm_note_dense_cache(ds4_session *s,
                                             uint32_t     pos0,
                                             uint32_t     n_tokens) {
    if (!s || n_tokens == 0) return;
    if (s->glm_graph_ready && !s->glm_graph.full_kv_cache) return;
    if (pos0 > s->glm_dense_cache_len) return;
    uint32_t end = pos0 + n_tokens;
    if (end < pos0) end = UINT32_MAX;
    if (s->glm_graph_ready && end > s->glm_graph.ctx_cap) end = s->glm_graph.ctx_cap;
    s->glm_dense_cache_len = end;
    ds4_session_glm_cap_dense_cache(s);
}
#endif

static uint32_t ds4_model_normal_layer_count(void) {
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM_DSA) {
        return (uint32_t)DS4_N_LAYER;
    }
    if (DS4_N_LAYER <= DS4_N_NEXTN_PREDICT || DS4_N_LAYER > DS4_MAX_LAYER) {
        return 0;
    }
    return DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
}

static uint32_t session_cpu_raw_live_rows(const ds4_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    uint32_t rows = ds4_default_raw_cap((uint32_t)s->ctx_size);
    if (rows > (uint32_t)s->checkpoint.len) rows = (uint32_t)s->checkpoint.len;
    return rows;
}

static uint32_t session_cpu_comp_cap(const ds4_session *s) {
    if (!s) return 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        if (layer->compress_ratio == 4) return layer->comp_cap;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        if (layer->compress_ratio != 0) return layer->comp_cap;
    }
    return (uint32_t)s->ctx_size;
}

static uint64_t session_cpu_payload_live_tensor_bytes(const ds4_session *s) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_cpu_raw_live_rows(s);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_cache *layer = &s->cpu_cache.layer[il];
        bytes += (uint64_t)raw_live * DS4_N_HEAD_DIM * sizeof(float);
        const uint32_t ratio = layer->compress_ratio;
        if (ratio == 0) continue;
        bytes += (uint64_t)layer->n_comp * DS4_N_HEAD_DIM * sizeof(float);
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)layer->n_index_comp * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}

static void session_cpu_reset_cache(ds4_session *s) {
    kv_cache_free(&s->cpu_cache);
    kv_cache_init(&s->cpu_cache, (uint32_t)s->ctx_size, 0);
}

static bool ds4_layer_payload_range_valid(uint32_t layer_start, uint32_t layer_end) {
    const uint32_t n_layers = ds4_model_normal_layer_count();
    return n_layers != 0 && layer_start <= layer_end && layer_end < n_layers;
}

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start,
                                         uint32_t layer_end) {
    if (!s || !s->checkpoint_valid ||
        !ds4_layer_payload_range_valid(layer_start, layer_end))
        return 0;
    if (ds4_session_is_cpu(s)) return 0;
    if (ds4_session_is_glm(s)) {
#ifdef DS4_NO_GPU
        (void)layer_start;
        (void)layer_end;
        return 0;
#else
        if (!s->glm_graph_ready) return 0;
        const ds4_glm_gpu_graph *g = &s->glm_graph;
        if (layer_start != g->layer_start || layer_end != g->layer_end) return 0;
        const uint32_t checkpoint_len = (uint32_t)s->checkpoint.len;
        const uint32_t full_live =
            session_glm_full_live_rows(g, checkpoint_len, s->glm_dense_cache_len);
        const uint32_t compact_live = session_glm_compact_live_rows(g, checkpoint_len);
        uint64_t bytes = (uint64_t)DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
        const uint32_t n_layers = layer_end - layer_start + 1u;
        bytes += (uint64_t)n_layers * sizeof(uint32_t);
        bytes += (uint64_t)n_layers * sizeof(uint32_t);
        for (uint32_t il = layer_start; il <= layer_end; il++) {
            const uint32_t index_live =
                compact_live != 0 && glm_graph_layer_uses_full_indexer(il)
                    ? compact_live
                    : 0;
            uint64_t layer_bytes = 0;
            if (!glm_layer_payload_tensor_bytes(il,
                                                full_live,
                                                DS4_N_KEY_MLA,
                                                DS4_N_VALUE_MLA,
                                                compact_live,
                                                index_live,
                                                &layer_bytes) ||
                !payload_u64_add(&bytes, layer_bytes))
                return 0;
        }
        return bytes;
#endif
    }
#ifdef DS4_NO_GPU
    (void)layer_start;
    (void)layer_end;
    return 0;
#else
    const ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    uint64_t bytes = (uint64_t)DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    const uint32_t n_layers = layer_end - layer_start + 1u;
    bytes += (uint64_t)n_layers * sizeof(uint32_t);
    bytes += (uint64_t)n_layers * sizeof(uint32_t);
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        bytes += (uint64_t)raw_live * DS4_SPARK_KV_ROW_BYTES;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * DS4_SPARK_KV_ROW_BYTES;
        bytes += layer_attn_state_bytes(ratio);
        bytes += layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] * DS4_SPARK_INDEX_ROW_BYTES;
            bytes += layer_index_state_bytes(ratio);
            bytes += layer_index_state_bytes(ratio);
        }
    }
    return bytes;
#endif
}

int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid ||
        !ds4_layer_payload_range_valid(layer_start, layer_end)) {
        payload_set_err(err, errlen, "invalid session layer payload save");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        payload_set_err(err, errlen, "distributed layer payloads require the graph backend");
        return 1;
    }
    if (ds4_session_is_glm(s)) {
#ifdef DS4_NO_GPU
        payload_set_err(err, errlen, "graph backend support is not compiled in");
        return 1;
#else
        if (!s->glm_graph_ready) {
            payload_set_err(err, errlen, "GLM graph is not ready for layer snapshot");
            return 1;
        }
        ds4_glm_gpu_graph *g = &s->glm_graph;
        if (layer_start != g->layer_start || layer_end != g->layer_end) {
            payload_set_err(err, errlen, "requested GLM layer snapshot does not match loaded slice");
            return 1;
        }
        if (ds4_gpu_synchronize() == 0) {
            payload_set_err(err, errlen, "failed to synchronize accelerator before GLM layer snapshot");
            return 1;
        }

        const uint32_t checkpoint_len = (uint32_t)s->checkpoint.len;
        const uint32_t full_live =
            session_glm_full_live_rows(g, checkpoint_len, s->glm_dense_cache_len);
        const uint32_t compact_live = session_glm_compact_live_rows(g, checkpoint_len);
        uint32_t header[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS] = {
            DS4_SESSION_LAYER_PAYLOAD_MAGIC,
            DS4_SESSION_LAYER_PAYLOAD_VERSION,
            (uint32_t)s->ctx_size,
            s->prefill_cap,
            g->ctx_cap,
            g->ctx_cap,
            g->compact_cache_cap,
            checkpoint_len,
            g->normal_layers,
            DS4_N_KEY_MLA,
            DS4_N_VALUE_MLA,
            layer_start,
            layer_end,
            full_live,
        };
        for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
            if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
        }
        for (uint32_t il = layer_start; il <= layer_end; il++) {
            if (payload_write_u32(fp, compact_live, err, errlen) != 0) return 1;
        }
        for (uint32_t il = layer_start; il <= layer_end; il++) {
            const uint32_t index_rows =
                compact_live != 0 && glm_graph_layer_uses_full_indexer(il) ? compact_live : 0;
            if (payload_write_u32(fp, index_rows, err, errlen) != 0) return 1;
        }

        uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
        int rc = 0;
        for (uint32_t il = layer_start; rc == 0 && il <= layer_end; il++) {
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
#ifdef DS4_NO_GPU
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    if (ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator before layer snapshot");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t raw_live = session_raw_live_rows(g, (uint32_t)s->checkpoint.len);
    uint32_t header[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_LAYER_PAYLOAD_MAGIC,
        DS4_SESSION_LAYER_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->prefill_cap,
        g->raw_cap,
        g->raw_window,
        g->comp_cap,
        (uint32_t)s->checkpoint.len,
        DS4_N_LAYER,
        DS4_N_HEAD_DIM,
        DS4_N_INDEXER_HEAD_DIM,
        layer_start,
        layer_end,
        raw_live,
    };
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (payload_write_u32(fp, g->layer_n_comp[il], err, errlen) != 0) return 1;
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (payload_write_u32(fp, g->layer_n_index_comp[il], err, errlen) != 0) return 1;
    }

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t il = layer_start; rc == 0 && il <= layer_end; il++) {
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

int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    if (!s || !fp || !tokens ||
        !ds4_layer_payload_range_valid(layer_start, layer_end)) {
        payload_set_err(err, errlen, "invalid session layer payload load");
        return 1;
    }
    if (ds4_session_is_cpu(s)) {
        payload_set_err(err, errlen, "distributed layer payloads require the graph backend");
        return 1;
    }
    if (ds4_session_is_glm(s)) {
#ifdef DS4_NO_GPU
        (void)payload_bytes;
        (void)n_tokens;
        payload_set_err(err, errlen, "graph backend support is not compiled in");
        return 1;
#else
        if (!s->glm_graph_ready) {
            payload_set_err(err, errlen, "GLM graph is not ready for KV shard restore");
            return 1;
        }
        ds4_glm_gpu_graph *g = &s->glm_graph;
        if (layer_start != g->layer_start || layer_end != g->layer_end) {
            payload_set_err(err, errlen, "requested GLM KV shard does not match loaded slice");
            return 1;
        }

        uint64_t remaining = payload_bytes;
        uint32_t h[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS];
        for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
            if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
        }
        if (h[0] != DS4_SESSION_LAYER_PAYLOAD_MAGIC ||
            h[1] != DS4_SESSION_LAYER_PAYLOAD_VERSION) {
            payload_set_err(err, errlen, "unsupported session layer payload version");
            return 1;
        }

        const uint32_t saved_ctx = h[2];
        const uint32_t saved_prefill_cap = h[3];
        const uint32_t saved_full_cap = h[4];
        const uint32_t saved_full_window = h[5];
        const uint32_t saved_compact_cap = h[6];
        const uint32_t saved_tokens = h[7];
        const uint32_t saved_layer_start = h[11];
        const uint32_t saved_layer_end = h[12];
        const uint32_t saved_full_live = h[13];
        (void)saved_prefill_cap;
        if (saved_layer_start != layer_start || saved_layer_end != layer_end) {
            payload_set_err(err, errlen, "KV shard layer range does not match requested worker");
            return 1;
        }
        if (saved_ctx > (uint32_t)s->ctx_size ||
            saved_tokens != n_tokens ||
            saved_tokens >= (uint32_t)s->ctx_size) {
            payload_set_err(err, errlen, "KV shard does not fit current context");
            return 1;
        }
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) {
                payload_set_err(err, errlen, "KV shard token is outside vocabulary");
                return 1;
            }
        }
        if (h[8] != g->normal_layers || h[9] != DS4_N_KEY_MLA ||
            h[10] != DS4_N_VALUE_MLA) {
            payload_set_err(err, errlen, "KV shard was written for a different GLM layout");
            return 1;
        }
        if (saved_full_cap == 0 || saved_full_cap > g->ctx_cap ||
            saved_full_window != saved_full_cap) {
            payload_set_err(err, errlen, "KV shard GLM full-cache layout does not match current runtime");
            return 1;
        }
        const uint32_t max_full_live =
            saved_tokens < saved_full_cap ? saved_tokens : saved_full_cap;
        if (saved_full_live > max_full_live || saved_full_live > g->ctx_cap) {
            payload_set_err(err, errlen, "KV shard GLM full-cache row count is invalid");
            return 1;
        }
        if (saved_compact_cap > g->ctx_size) {
            payload_set_err(err, errlen, "KV shard GLM compact cache is larger than current context");
            return 1;
        }
        if (g->compact_cache_cap != 0 && saved_tokens != 0 && saved_compact_cap == 0) {
            payload_set_err(err, errlen, "KV shard lacks the GLM compact cache required by this context");
            return 1;
        }
        const uint32_t expected_compact_live =
            saved_compact_cap != 0
                ? (saved_tokens < saved_compact_cap ? saved_tokens : saved_compact_cap)
                : 0;
        if (expected_compact_live != 0 &&
            (!s->engine || !glm_graph_ensure_compact_cache(g, expected_compact_live))) {
            payload_set_err(err, errlen, "KV shard GLM compact cache could not be allocated");
            return 1;
        }
        if (expected_compact_live > g->compact_cache_cap) {
            payload_set_err(err, errlen, "KV shard GLM compact row count is invalid");
            return 1;
        }

        const uint32_t n_layers = layer_end - layer_start + 1u;
        uint32_t *n_comp = xcalloc(n_layers, sizeof(n_comp[0]));
        uint32_t *n_index_comp = xcalloc(n_layers, sizeof(n_index_comp[0]));
        for (uint32_t i = 0; i < n_layers; i++) {
            if (payload_read_u32(fp, &n_comp[i], &remaining, err, errlen) != 0) {
                free(n_comp);
                free(n_index_comp);
                return 1;
            }
            if (n_comp[i] != expected_compact_live) {
                free(n_comp);
                free(n_index_comp);
                payload_set_err(err, errlen, "KV shard GLM compact row count does not match token count");
                return 1;
            }
        }
        for (uint32_t i = 0; i < n_layers; i++) {
            const uint32_t il = layer_start + i;
            if (payload_read_u32(fp, &n_index_comp[i], &remaining, err, errlen) != 0) {
                free(n_comp);
                free(n_index_comp);
                return 1;
            }
            const uint32_t expected_index_rows =
                expected_compact_live != 0 && glm_graph_layer_uses_full_indexer(il)
                    ? expected_compact_live
                    : 0;
            if (n_index_comp[i] != expected_index_rows) {
                free(n_comp);
                free(n_index_comp);
                payload_set_err(err, errlen, "KV shard GLM indexer row count does not match token count");
                return 1;
            }
        }

        if (ds4_gpu_synchronize() == 0) {
            free(n_comp);
            free(n_index_comp);
            payload_set_err(err, errlen, "failed to synchronize accelerator before GLM KV shard restore");
            return 1;
        }
        s->checkpoint_valid = false;
        s->mtp_draft_valid = false;
        ds4_session_glm_reset_dense_cache(s);

        uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
        int rc = 0;
        for (uint32_t i = 0; rc == 0 && i < n_layers; i++) {
            const uint32_t il = layer_start + i;
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
                                               (uint64_t)n_comp[i] * DS4_N_KV_LORA,
                                               buf,
                                               DS4_SESSION_IO_CHUNK,
                                               &remaining,
                                               err,
                                               errlen);
            if (rc == 0) {
                rc = payload_read_glm_compact_span(fp,
                                                   g->layer_k_rope_cache[il],
                                                   (uint64_t)n_comp[i] * DS4_N_ROT,
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
            }
            if (rc == 0 && glm_graph_layer_uses_full_indexer(il)) {
                rc = payload_read_glm_compact_span(fp,
                                                   g->layer_indexer_key_cache[il],
                                                   (uint64_t)n_index_comp[i] * DS4_N_INDEXER_HEAD_DIM,
                                                   buf,
                                                   DS4_SESSION_IO_CHUNK,
                                                   &remaining,
                                                   err,
                                                   errlen);
            }
        }
        free(buf);
        if (rc == 0 && remaining != 0) {
            payload_set_err(err, errlen, "KV shard has trailing payload bytes");
            rc = 1;
        }
        if (rc == 0 && ds4_gpu_synchronize() == 0) {
            payload_set_err(err, errlen, "failed to synchronize accelerator after GLM KV shard restore");
            rc = 1;
        }
        if (rc == 0) {
            token_vec_free(&s->checkpoint);
            memset(&s->checkpoint, 0, sizeof(s->checkpoint));
            for (uint32_t i = 0; i < n_tokens; i++) token_vec_push(&s->checkpoint, tokens[i]);
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
            s->glm_dense_cache_len = g->full_kv_cache ? saved_full_live : 0;
        }
        free(n_comp);
        free(n_index_comp);
        return rc;
#endif
    }
#ifdef DS4_NO_GPU
    (void)payload_bytes;
    (void)n_tokens;
    payload_set_err(err, errlen, "graph backend support is not compiled in");
    return 1;
#else
    uint64_t remaining = payload_bytes;
    uint32_t h[DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < DS4_SESSION_LAYER_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != DS4_SESSION_LAYER_PAYLOAD_MAGIC ||
        h[1] != DS4_SESSION_LAYER_PAYLOAD_VERSION) {
        payload_set_err(err, errlen, "unsupported session layer payload version");
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    const uint32_t saved_ctx = h[2];
    const uint32_t saved_prefill_cap = h[3];
    const uint32_t saved_raw_cap = h[4];
    const uint32_t saved_raw_window = h[5];
    const uint32_t saved_comp_cap = h[6];
    const uint32_t saved_tokens = h[7];
    const uint32_t saved_layer_start = h[11];
    const uint32_t saved_layer_end = h[12];
    const uint32_t saved_raw_live = h[13];
    (void)saved_prefill_cap;
    if (saved_layer_start != layer_start || saved_layer_end != layer_end) {
        payload_set_err(err, errlen, "KV shard layer range does not match requested worker");
        return 1;
    }
    if (saved_ctx > (uint32_t)s->ctx_size ||
        saved_tokens != n_tokens ||
        saved_tokens >= (uint32_t)s->ctx_size) {
        payload_set_err(err, errlen, "KV shard does not fit current context");
        return 1;
    }
    if (h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_DIM ||
        h[10] != DS4_N_INDEXER_HEAD_DIM) {
        payload_set_err(err, errlen, "KV shard was written for a different DS4 layout");
        return 1;
    }
    if (saved_raw_window != g->raw_window) {
        payload_set_err(err, errlen, "KV shard graph chunk layout does not match current runtime");
        return 1;
    }
    const uint32_t expected_raw_live = saved_tokens < saved_raw_window ? saved_tokens : saved_raw_window;
    if (saved_raw_cap == 0 || saved_raw_live != expected_raw_live ||
        saved_raw_live > saved_raw_cap || saved_raw_live > g->raw_cap) {
        payload_set_err(err, errlen, "KV shard raw ring layout does not match current context");
        return 1;
    }
    if (saved_comp_cap > g->comp_cap) {
        payload_set_err(err, errlen, "KV shard compressed cache is larger than current context");
        return 1;
    }

    const uint32_t n_layers = layer_end - layer_start + 1u;
    uint32_t *n_comp = xcalloc(n_layers, sizeof(n_comp[0]));
    uint32_t *n_index_comp = xcalloc(n_layers, sizeof(n_index_comp[0]));
    for (uint32_t i = 0; i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        if (payload_read_u32(fp, &n_comp[i], &remaining, err, errlen) != 0) {
            free(n_comp);
            free(n_index_comp);
            return 1;
        }
        if (n_comp[i] > saved_comp_cap || n_comp[i] > g->layer_comp_cap[il]) {
            free(n_comp);
            free(n_index_comp);
            payload_set_err(err, errlen, "KV shard has invalid compressed row count");
            return 1;
        }
    }
    for (uint32_t i = 0; i < n_layers; i++) {
        const uint32_t il = layer_start + i;
        if (payload_read_u32(fp, &n_index_comp[i], &remaining, err, errlen) != 0) {
            free(n_comp);
            free(n_index_comp);
            return 1;
        }
        if (n_index_comp[i] > saved_comp_cap || n_index_comp[i] > g->layer_comp_cap[il]) {
            free(n_comp);
            free(n_index_comp);
            payload_set_err(err, errlen, "KV shard has invalid indexer row count");
            return 1;
        }
    }

    if (ds4_gpu_synchronize() == 0) {
        free(n_comp);
        free(n_index_comp);
        payload_set_err(err, errlen, "failed to synchronize accelerator before KV shard restore");
        return 1;
    }
    s->checkpoint_valid = false;
    s->mtp_draft_valid = false;
    ds4_session_dspark_capture_invalidate(s);
    g->mtp_n_raw = 0;
    metal_graph_dspark_cache_reset(g);

    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (uint32_t i = 0; rc == 0 && i < n_layers; i++) {
        const uint32_t il = layer_start + i;
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
                                      (uint64_t)n_comp[i] * DS4_SPARK_KV_ROW_BYTES,
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
                                          (uint64_t)n_index_comp[i] * DS4_SPARK_INDEX_ROW_BYTES,
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
    if (rc == 0 && remaining != 0) {
        payload_set_err(err, errlen, "KV shard has trailing payload bytes");
        rc = 1;
    }
    if (rc == 0 && ds4_gpu_synchronize() == 0) {
        payload_set_err(err, errlen, "failed to synchronize accelerator after KV shard restore");
        rc = 1;
    }
    if (rc == 0) {
        token_vec_free(&s->checkpoint);
        memset(&s->checkpoint, 0, sizeof(s->checkpoint));
        for (uint32_t i = 0; i < n_tokens; i++) token_vec_push(&s->checkpoint, tokens[i]);
        for (uint32_t i = 0; i < n_layers; i++) {
            const uint32_t il = layer_start + i;
            g->layer_n_comp[il] = n_comp[i];
            g->layer_n_index_comp[il] = n_index_comp[i];
        }
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        ds4_session_dspark_capture_invalidate(s);
        g->mtp_n_raw = 0;
        metal_graph_dspark_cache_reset(g);
    }
    free(n_comp);
    free(n_index_comp);
    return rc;
#endif
}

int ds4_engine_routed_quant_bits(ds4_engine *e) {
    if (!e) return 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_tensor *gate = e->weights.layer[il].ffn_gate_exps;
        if (!gate) continue;
        return gate->type == DS4_TENSOR_Q4_K ? 4 : 2;
    }
    return 0;
}

bool ds4_engine_has_output_head(ds4_engine *e) {
    return e && weights_have_output_head(&e->weights);
}

#ifndef DS4_NO_GPU
static bool ds4_engine_glm_mtp_spec_enabled(const ds4_engine *e) {
    return e && e->backend != DS4_BACKEND_CPU &&
           DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA &&
           DS4_N_NEXTN_PREDICT != 0 && e->glm_mtp;
}
#endif

bool ds4_engine_has_mtp(ds4_engine *e) {
    return e && e->backend != DS4_BACKEND_CPU &&
           e->distributed.role == DS4_DISTRIBUTED_NONE &&
           e->mtp_ready;
}

int ds4_engine_mtp_draft_tokens(ds4_engine *e) {
    if (e && DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        return e->glm_mtp && DS4_N_NEXTN_PREDICT != 0 ? 2 : 0;
    }
    if (ds4_engine_has_mtp(e)) return e->mtp_draft_tokens;
#ifndef DS4_NO_GPU
    if (e &&
        e->backend != DS4_BACKEND_CPU &&
        e->distributed.role == DS4_DISTRIBUTED_NONE &&
        e->support_kind == DS4_SUPPORT_DSPARK &&
        e->dspark &&
        e->dspark_weights.block_size > 1) {
        return (int)e->dspark_weights.block_size;
    }
#endif
    return 0;
}

const ds4_tokens *ds4_session_tokens(ds4_session *s) {
    return s ? &s->checkpoint : NULL;
}

