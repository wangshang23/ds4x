#define DS4_SESSION_IO_CHUNK (8u * 1024u * 1024u)

static void payload_set_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg);
}

static void payload_put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)v;
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

static int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes,
                               char *err, size_t errlen) {
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

static int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes,
                              uint64_t *remaining,
                              char *err, size_t errlen) {
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

static int payload_write_u32(FILE *fp, uint32_t v,
                             char *err, size_t errlen) {
    uint8_t b[4];
    payload_put_u32(b, v);
    return payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

static int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining,
                            char *err, size_t errlen) {
    uint8_t b[4];
    if (payload_read_bytes(fp, b, sizeof(b), remaining, err, errlen) != 0) {
        return 1;
    }
    *v = payload_get_u32(b);
    return 0;
}

static int payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes,
                                   char *err, size_t errlen) {
    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    while (bytes != 0) {
        const size_t n = bytes > DS4_SESSION_IO_CHUNK
            ? DS4_SESSION_IO_CHUNK : (size_t)bytes;
        if (fread(buf, 1, n, src) != n || fwrite(buf, 1, n, dst) != n) {
            payload_set_err(err, errlen, "failed to copy staged payload");
            rc = 1;
            break;
        }
        bytes -= n;
    }
    free(buf);
    return rc;
}

static uint64_t layer_attn_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio * sizeof(float);
}

static uint64_t layer_index_state_bytes(uint32_t ratio) {
    const uint32_t coff = ratio == 4 ? 2u : 1u;
    return (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio *
           sizeof(float);
}

static uint32_t session_raw_live_rows(const ds4_gpu_graph *g,
                                      uint32_t checkpoint_len) {
    uint32_t rows = g->raw_window ? g->raw_window : DS4_N_SWA;
    if (rows > g->raw_cap) rows = g->raw_cap;
    if (rows > checkpoint_len) rows = checkpoint_len;
    return rows;
}

static uint64_t session_payload_live_tensor_bytes(
        const ds4_gpu_graph *g, uint32_t checkpoint_len) {
    uint64_t bytes = 0;
    const uint32_t raw_live = session_raw_live_rows(g, checkpoint_len);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        bytes += (uint64_t)raw_live * DS4_SPARK_KV_ROW_BYTES;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        bytes += (uint64_t)g->layer_n_comp[il] * DS4_SPARK_KV_ROW_BYTES;
        bytes += 2u * layer_attn_state_bytes(ratio);
        if (ratio == 4) {
            bytes += (uint64_t)g->layer_n_index_comp[il] *
                     DS4_SPARK_INDEX_ROW_BYTES;
            bytes += 2u * layer_index_state_bytes(ratio);
        }
    }
    return bytes;
}

static int payload_write_tensor_span(FILE *fp,
                                     const ds4_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap,
                                     char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset) {
        payload_set_err(err, errlen, "session tensor is smaller than payload");
        return 1;
    }
    for (uint64_t done = 0; done < bytes;) {
        const size_t n = bytes - done > cap ? cap : (size_t)(bytes - done);
        if (!ds4_gpu_tensor_read(tensor, offset + done, buf, n) ||
            payload_write_bytes(fp, buf, n, err, errlen) != 0) {
            payload_set_err(err, errlen, "failed to save session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}

static int payload_read_tensor_span(FILE *fp, ds4_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap,
                                    uint64_t *remaining,
                                    char *err, size_t errlen) {
    if (!tensor || offset > ds4_gpu_tensor_bytes(tensor) ||
        bytes > ds4_gpu_tensor_bytes(tensor) - offset) {
        payload_set_err(err, errlen, "session tensor is smaller than payload");
        return 1;
    }
    for (uint64_t done = 0; done < bytes;) {
        const size_t n = bytes - done > cap ? cap : (size_t)(bytes - done);
        if (payload_read_bytes(fp, buf, n, remaining, err, errlen) != 0 ||
            !ds4_gpu_tensor_write(tensor, offset + done, buf, n)) {
            payload_set_err(err, errlen, "failed to restore session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}

static bool ds4_session_is_cpu(const ds4_session *s) {
    (void)s;
    return false;
}

static void ds4_session_dspark_capture_invalidate(ds4_session *s) {
    if (!s) return;
    s->dspark_draft_valid = false;
    s->dspark_draft_len = 0;
    metal_graph_dspark_capture_invalidate(&s->graph);
}

static void ds4_session_dspark_capture_note_checkpoint(ds4_session *s) {
    if (!s) return;
    ds4_gpu_graph *g = &s->graph;
    if (!g->dspark_capture_enabled) return;
    if (!s->checkpoint_valid || s->checkpoint.len <= 0 ||
        !g->dspark_capture_valid) {
        metal_graph_dspark_capture_invalidate(g);
        return;
    }
    g->dspark_capture_checkpoint_len = (uint32_t)s->checkpoint.len;
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

bool ds4_engine_has_mtp(ds4_engine *e) {
    (void)e;
    return false;
}

int ds4_engine_dspark_block_size(ds4_engine *e) {
    if (e && e->support_kind == DS4_SUPPORT_DSPARK && e->dspark &&
        e->dspark_weights.block_size > 1) {
        return (int)e->dspark_weights.block_size;
    }
    return 0;
}

const ds4_tokens *ds4_session_tokens(ds4_session *s) {
    return s ? &s->checkpoint : NULL;
}
