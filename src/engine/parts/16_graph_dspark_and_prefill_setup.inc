#ifndef DS4_NO_GPU
/* =========================================================================
 * Metal Release Decode and Prefill.
 * =========================================================================
 *
 * Everything below is the user-facing Metal backend.  It uses the same layer
 * encoder as diagnostics, but diagnostics are not required for normal command
 * flow and their CPU reads stay outside these generation entry points.
 */

static uint32_t metal_graph_cuda_decode_split_after_layers(void) {
    const char *env = getenv("DS4_CUDA_DECODE_SPLIT_LAYERS");
    if (!env || !env[0]) return 4u;
    char *end = NULL;
    const unsigned long value = strtoul(env, &end, 10);
    if (end == env || *end != '\0' || value > DS4_N_LAYER) return 4u;
    return (uint32_t)value;
}

static int metal_graph_dspark_target_slot(
        const ds4_gpu_graph *g,
        uint32_t             il) {
    if (!g || !g->dspark_capture_enabled) return -1;
    for (uint32_t i = 0; i < g->dspark_target_layer_count; i++) {
        if (g->dspark_target_layers[i] == il) return (int)i;
    }
    return -1;
}

static uint32_t metal_graph_dspark_capture_complete_mask(
        const ds4_gpu_graph *g) {
    if (!g || g->dspark_target_layer_count == 0) return 0;
    return g->dspark_target_layer_count >= 32u ?
        UINT32_MAX : ((1u << g->dspark_target_layer_count) - 1u);
}

static void metal_graph_dspark_capture_note_slot(ds4_gpu_graph *g,
                                                 uint32_t       slot) {
    if (!g || slot >= g->dspark_target_layer_count) return;
    g->dspark_capture_mask |= 1u << slot;
    g->dspark_capture_valid =
        g->dspark_capture_mask == metal_graph_dspark_capture_complete_mask(g);
}

static void metal_graph_dspark_capture_row_invalidate(ds4_gpu_graph *g) {
    if (!g || !g->dspark_capture_enabled) return;
    g->dspark_capture_mask = 0;
    g->dspark_capture_checkpoint_len = 0;
    g->dspark_capture_valid = false;
}

static void metal_graph_dspark_capture_batch_invalidate(ds4_gpu_graph *g) {
    if (!g || !g->dspark_capture_enabled) return;
    g->dspark_capture_batch_mask = 0;
    g->dspark_capture_batch_start = 0;
    g->dspark_capture_batch_tokens = 0;
    g->dspark_capture_batch_valid = false;
}

static void metal_graph_dspark_capture_invalidate(ds4_gpu_graph *g) {
    metal_graph_dspark_capture_row_invalidate(g);
    metal_graph_dspark_capture_batch_invalidate(g);
}

static void metal_graph_dspark_cache_reset(ds4_gpu_graph *g) {
    if (!g) return;
    g->dspark_cache_start = 0;
    g->dspark_cache_token_start = 0;
    g->dspark_cache_len = 0;
}

static bool metal_graph_dspark_cache_window_valid(
        const ds4_gpu_graph *g,
        uint32_t             token_start,
        uint32_t             raw_start,
        uint32_t             len) {
    if (!g || len > g->dspark_cache_cap) return false;
    if (len == 0) return true;
    if (g->dspark_cache_cap == 0 ||
        raw_start >= g->dspark_cache_cap ||
        token_start > UINT32_MAX - len ||
        raw_start != token_start % g->dspark_cache_cap) {
        return false;
    }
    return true;
}

static bool metal_graph_dspark_cache_current_window_valid(
        const ds4_gpu_graph *g) {
    if (!g) return false;
    return metal_graph_dspark_cache_window_valid(g,
                                                g->dspark_cache_token_start,
                                                g->dspark_cache_start,
                                                g->dspark_cache_len);
}

static bool metal_graph_dspark_cache_set_window(ds4_gpu_graph *g,
                                                uint32_t       token_start,
                                                uint32_t       len) {
    if (!g || len > g->dspark_cache_cap) return false;
    const uint32_t raw_start =
        len && g->dspark_cache_cap ? token_start % g->dspark_cache_cap : 0;
    if (!metal_graph_dspark_cache_window_valid(g,
                                               len ? token_start : 0,
                                               raw_start,
                                               len)) {
        return false;
    }
    g->dspark_cache_start = raw_start;
    g->dspark_cache_token_start = len ? token_start : 0;
    g->dspark_cache_len = len;
    return true;
}

static bool metal_graph_dspark_cache_crop_to_prefix(ds4_gpu_graph *g,
                                                    uint32_t       prefix_len) {
    if (!g) return false;
    if (g->dspark_cache_len == 0) return true;
    if (!metal_graph_dspark_cache_current_window_valid(g)) return false;

    const uint32_t start = g->dspark_cache_token_start;
    const uint32_t end = start + g->dspark_cache_len;
    if (prefix_len <= start || prefix_len > end) {
        metal_graph_dspark_cache_reset(g);
        return true;
    }
    g->dspark_cache_len = prefix_len - start;
    return true;
}

static bool metal_graph_dspark_cache_ends_at(const ds4_gpu_graph *g,
                                             uint32_t             pos) {
    if (!metal_graph_dspark_cache_current_window_valid(g)) return false;
    if (g->dspark_cache_len == 0) return true;
    return g->dspark_cache_token_start <= UINT32_MAX - g->dspark_cache_len &&
           g->dspark_cache_token_start + g->dspark_cache_len == pos;
}

static bool metal_graph_dspark_cache_claim_appended_row(ds4_gpu_graph *g,
                                                        uint32_t pos) {
    if (!g || g->dspark_cache_len == 0 ||
        !metal_graph_dspark_cache_ends_at(g, pos)) return false;
    g->dspark_cache_len += 1u;
    if (g->dspark_cache_len > g->dspark_cache_cap) {
        const uint32_t excess = g->dspark_cache_len - g->dspark_cache_cap;
        g->dspark_cache_token_start += excess;
        g->dspark_cache_len = g->dspark_cache_cap;
        g->dspark_cache_start =
            g->dspark_cache_token_start % g->dspark_cache_cap;
    }
    return true;
}

bool ds4_test_dspark_cache_window_crop(void) {
    ds4_gpu_graph g;
    memset(&g, 0, sizeof(g));
    g.dspark_cache_cap = 8;

    if (!metal_graph_dspark_cache_set_window(&g, 10, 5)) return false;
    if (g.dspark_cache_token_start != 10 ||
        g.dspark_cache_start != 2 ||
        g.dspark_cache_len != 5) return false;
    if (!metal_graph_dspark_cache_ends_at(&g, 15)) return false;
    if (metal_graph_dspark_cache_ends_at(&g, 14)) return false;
    if (metal_graph_dspark_cache_window_valid(&g, 10, 3, 5)) return false;

    if (!metal_graph_dspark_cache_crop_to_prefix(&g, 13)) return false;
    if (g.dspark_cache_token_start != 10 ||
        g.dspark_cache_start != 2 ||
        g.dspark_cache_len != 3) return false;
    if (!metal_graph_dspark_cache_ends_at(&g, 13)) return false;
    if (!metal_graph_dspark_cache_claim_appended_row(&g, 13)) return false;
    if (g.dspark_cache_token_start != 10 ||
        g.dspark_cache_start != 2 ||
        g.dspark_cache_len != 4) return false;
    if (!metal_graph_dspark_cache_ends_at(&g, 14)) return false;
    if (metal_graph_dspark_cache_claim_appended_row(&g, 13)) return false;

    if (!metal_graph_dspark_cache_crop_to_prefix(&g, 20)) return false;
    if (g.dspark_cache_token_start != 0 ||
        g.dspark_cache_start != 0 ||
        g.dspark_cache_len != 0) return false;
    if (!metal_graph_dspark_cache_ends_at(&g, 20)) return false;

    if (metal_graph_dspark_cache_set_window(&g, UINT32_MAX - 1u, 2)) {
        return false;
    }
    return true;
}

static void metal_graph_dspark_capture_begin(ds4_gpu_graph *g) {
    metal_graph_dspark_capture_row_invalidate(g);
}

static void metal_graph_dspark_capture_begin_prefill(ds4_gpu_graph *g) {
    metal_graph_dspark_capture_invalidate(g);
}

static bool metal_graph_dspark_capture_hc(
        ds4_gpu_graph       *g,
        const ds4_gpu_tensor *hc,
        uint32_t             slot) {
    if (!g || !hc || !g->dspark_target_hidden ||
        !g->dspark_hc_mean_weights ||
        slot >= g->dspark_target_layer_count) {
        return false;
    }

    ds4_gpu_tensor *dst =
        ds4_gpu_tensor_view(g->dspark_target_hidden,
                            (uint64_t)slot * DS4_N_EMBD * sizeof(float),
                            (uint64_t)DS4_N_EMBD * sizeof(float));
    if (!dst) return false;
    const bool ok = ds4_gpu_hc_weighted_sum_tensor(dst,
                                                   hc,
                                                   g->dspark_hc_mean_weights,
                                                   DS4_N_EMBD,
                                                   DS4_N_HC) != 0;
    ds4_gpu_tensor_free(dst);
    if (ok) metal_graph_dspark_capture_note_slot(g, slot);
    return ok;
}

static bool metal_graph_dspark_capture_batch_note_slot(
        ds4_gpu_graph *g,
        uint32_t       slot,
        uint32_t       start,
        uint32_t       n_tokens) {
    if (!g || slot >= g->dspark_target_layer_count || n_tokens == 0) {
        return false;
    }
    if (g->dspark_capture_batch_mask == 0) {
        g->dspark_capture_batch_start = start;
        g->dspark_capture_batch_tokens = n_tokens;
    } else if (g->dspark_capture_batch_start != start ||
               g->dspark_capture_batch_tokens != n_tokens) {
        metal_graph_dspark_capture_batch_invalidate(g);
        return false;
    }
    g->dspark_capture_batch_mask |= 1u << slot;
    g->dspark_capture_batch_valid =
        g->dspark_capture_batch_mask ==
        metal_graph_dspark_capture_complete_mask(g);
    return true;
}

static bool metal_graph_dspark_capture_decode_layer(
        ds4_gpu_graph *g,
        uint32_t       il) {
    const int slot = metal_graph_dspark_target_slot(g, il);
    if (slot < 0) return true;
    return metal_graph_dspark_capture_hc(g, metal_graph_cur_hc(g), (uint32_t)slot);
}

static bool metal_graph_dspark_capture_prefill_layer(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       start,
        uint32_t       n_tokens) {
    const int slot = metal_graph_dspark_target_slot(g, il);
    if (slot < 0) return true;
    if (n_tokens == 0) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t embd_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    if (g->dspark_target_hidden_batch &&
        g->dspark_hc_mean_rows &&
        n_tokens <= g->prefill_cap) {
        ds4_gpu_tensor *batch_dst =
            ds4_gpu_tensor_view(g->dspark_target_hidden_batch,
                                ((uint64_t)slot * g->prefill_cap *
                                 DS4_N_EMBD) * sizeof(float),
                                (uint64_t)n_tokens * embd_bytes);
        ds4_gpu_tensor *last_src =
            batch_dst ?
                ds4_gpu_tensor_view(batch_dst,
                                    (uint64_t)(n_tokens - 1u) * embd_bytes,
                                    embd_bytes) : NULL;
        ds4_gpu_tensor *last_dst =
            ds4_gpu_tensor_view(g->dspark_target_hidden,
                                (uint64_t)slot * embd_bytes,
                                embd_bytes);
        bool ok = batch_dst && last_src && last_dst &&
                  ds4_gpu_hc_weighted_sum_tensor(batch_dst,
                                                 metal_graph_batch_cur_hc(g),
                                                 g->dspark_hc_mean_rows,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0 &&
                  ds4_gpu_tensor_copy(last_dst,
                                      0,
                                      last_src,
                                      0,
                                      embd_bytes) != 0;
        ds4_gpu_tensor_free(last_dst);
        ds4_gpu_tensor_free(last_src);
        ds4_gpu_tensor_free(batch_dst);
        if (ok) {
            metal_graph_dspark_capture_note_slot(g, (uint32_t)slot);
            ok = metal_graph_dspark_capture_batch_note_slot(g,
                                                            (uint32_t)slot,
                                                            start,
                                                            n_tokens);
        }
        return ok;
    }

    ds4_gpu_tensor *last_hc =
        ds4_gpu_tensor_view(metal_graph_batch_cur_hc(g),
                            (uint64_t)(n_tokens - 1u) * hc_dim * sizeof(float),
                            hc_dim * sizeof(float));
    if (!last_hc) return false;
    const bool ok = metal_graph_dspark_capture_hc(g, last_hc, (uint32_t)slot);
    ds4_gpu_tensor_free(last_hc);
    return ok;
}

static bool metal_graph_dspark_capture_prefill_rows(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       chunk_start,
        uint32_t       chunk_len,
        uint32_t       pos0,
        uint32_t       n_tokens) {
    const int slot = metal_graph_dspark_target_slot(g, il);
    if (slot < 0) return true;
    if (!g->dspark_target_hidden_batch ||
        !g->dspark_target_hidden ||
        !g->dspark_hc_mean_rows ||
        n_tokens == 0 ||
        chunk_len == 0 ||
        chunk_len > g->prefill_cap ||
        pos0 < chunk_start) {
        return true;
    }
    const uint32_t row0 = pos0 - chunk_start;
    if (row0 > chunk_len || n_tokens > chunk_len - row0) return true;

    const uint64_t embd_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    ds4_gpu_tensor *batch_dst =
        ds4_gpu_tensor_view(g->dspark_target_hidden_batch,
                            (((uint64_t)(uint32_t)slot * g->prefill_cap +
                              row0) * DS4_N_EMBD) * sizeof(float),
                            (uint64_t)n_tokens * embd_bytes);
    bool ok = batch_dst &&
              ds4_gpu_hc_weighted_sum_tensor(batch_dst,
                                             metal_graph_batch_cur_hc(g),
                                             g->dspark_hc_mean_rows,
                                             DS4_N_EMBD,
                                             DS4_N_HC) != 0;
    if (!ok) fprintf(stderr, "ds4: pipeline capture rows FAIL il=%u row0=%u n=%u dst=%d\n",
                     il, row0, n_tokens, batch_dst != NULL);
    if (ok && row0 + n_tokens == chunk_len) {
        ds4_gpu_tensor *last_src =
            ds4_gpu_tensor_view(batch_dst,
                                (uint64_t)(n_tokens - 1u) * embd_bytes,
                                embd_bytes);
        ds4_gpu_tensor *last_dst =
            ds4_gpu_tensor_view(g->dspark_target_hidden,
                                (uint64_t)(uint32_t)slot * embd_bytes,
                                embd_bytes);
        ok = last_src && last_dst &&
             ds4_gpu_tensor_copy(last_dst, 0, last_src, 0, embd_bytes) != 0;
        ds4_gpu_tensor_free(last_dst);
        ds4_gpu_tensor_free(last_src);
        if (ok) {
            metal_graph_dspark_capture_note_slot(g, (uint32_t)slot);
            ok = metal_graph_dspark_capture_batch_note_slot(g,
                                                            (uint32_t)slot,
                                                            chunk_start,
                                                            chunk_len);
        }
    }
    ds4_gpu_tensor_free(batch_dst);
    return ok;
}

static bool metal_graph_dspark_capture_verified_suffix_begin(
        ds4_gpu_graph *g,
        uint32_t       start,
        uint32_t       n_tokens,
        bool           commands_open) {
    if (!g || !g->dspark_capture_enabled ||
        !g->dspark_target_hidden ||
        !g->dspark_target_hidden_batch ||
        start == 0 ||
        n_tokens == 0 ||
        n_tokens + 1u < n_tokens ||
        n_tokens + 1u > g->prefill_cap ||
        !g->dspark_capture_valid ||
        g->dspark_capture_checkpoint_len != start) {
        metal_graph_dspark_capture_invalidate(g);
        return false;
    }

    const uint64_t embd_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    metal_graph_dspark_capture_batch_invalidate(g);
    bool ok = commands_open || ds4_gpu_begin_commands() != 0;
    for (uint32_t slot = 0; ok && slot < g->dspark_target_layer_count; slot++) {
        ds4_gpu_tensor *dst =
            ds4_gpu_tensor_view(g->dspark_target_hidden_batch,
                                ((uint64_t)slot * g->prefill_cap *
                                 DS4_N_EMBD) * sizeof(float),
                                embd_bytes);
        ds4_gpu_tensor *src =
            ds4_gpu_tensor_view(g->dspark_target_hidden,
                                (uint64_t)slot * embd_bytes,
                                embd_bytes);
        ok = dst && src &&
             ds4_gpu_tensor_copy(dst, 0, src, 0, embd_bytes) != 0;
        ds4_gpu_tensor_free(src);
        ds4_gpu_tensor_free(dst);
    }
    if (ok && !commands_open) ok = ds4_gpu_end_commands() != 0;
    else if (!ok && !commands_open) (void)ds4_gpu_synchronize();
    if (!ok) {
        metal_graph_dspark_capture_invalidate(g);
        return false;
    }

    metal_graph_dspark_capture_row_invalidate(g);
    return true;
}

static bool metal_graph_dspark_capture_verified_suffix_layer(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       start,
        uint32_t       n_tokens) {
    const int slot = metal_graph_dspark_target_slot(g, il);
    if (slot < 0) return true;
    if (!g || !g->dspark_target_hidden_batch ||
        !g->dspark_target_hidden ||
        !g->dspark_hc_mean_rows ||
        start == 0 ||
        n_tokens == 0 ||
        n_tokens + 1u < n_tokens ||
        n_tokens + 1u > g->prefill_cap) {
        return false;
    }

    const uint64_t embd_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    ds4_gpu_tensor *batch_dst =
        ds4_gpu_tensor_view(g->dspark_target_hidden_batch,
                            (((uint64_t)(uint32_t)slot * g->prefill_cap +
                              1u) * DS4_N_EMBD) * sizeof(float),
                            (uint64_t)n_tokens * embd_bytes);
    ds4_gpu_tensor *last_src =
        batch_dst ?
            ds4_gpu_tensor_view(batch_dst,
                                (uint64_t)(n_tokens - 1u) * embd_bytes,
                                embd_bytes) : NULL;
    ds4_gpu_tensor *last_dst =
        ds4_gpu_tensor_view(g->dspark_target_hidden,
                            (uint64_t)(uint32_t)slot * embd_bytes,
                            embd_bytes);
    bool ok = batch_dst && last_src && last_dst &&
              ds4_gpu_hc_weighted_sum_tensor(batch_dst,
                                             metal_graph_batch_cur_hc(g),
                                             g->dspark_hc_mean_rows,
                                             DS4_N_EMBD,
                                             DS4_N_HC) != 0 &&
              ds4_gpu_tensor_copy(last_dst,
                                  0,
                                  last_src,
                                  0,
                                  embd_bytes) != 0;
    ds4_gpu_tensor_free(last_dst);
    ds4_gpu_tensor_free(last_src);
    ds4_gpu_tensor_free(batch_dst);
    if (ok) {
        metal_graph_dspark_capture_note_slot(g, (uint32_t)slot);
        ok = metal_graph_dspark_capture_batch_note_slot(g,
                                                        (uint32_t)slot,
                                                        start - 1u,
                                                        n_tokens + 1u);
    }
    return ok;
}

/* Encode a full single-token decode step on CUDA.  This is the generation
 * hot path: update caches, run all layers, then produce logits. */
static bool metal_graph_encode_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        bool                   need_logits,
        bool                   allow_split_flush) {
    if (g->raw_cap == 0) {
        fprintf(stderr, "ds4: CUDA graph raw KV cache is not allocated\n");
        return false;
    }
    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
    metal_graph_dspark_capture_begin(g);

    bool ok = ds4_gpu_embed_token_hc_tensor(metal_graph_cur_hc(g),
                                              model->map,
                                              model->size,
                                              weights->token_embd->abs_offset,
                                              (uint32_t)weights->token_embd->dim[1],
                                              (uint32_t)token,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;

    /*
     * Start executing the prefix of the decode graph while the CPU is still
     * encoding the rest. The split point is layer-based because this executor is
     * a fixed DS4 tape, not a dynamic node graph; four layers is the measured
     * point where the prefix is large enough to hide useful work without
     * starving the second command buffer.
     */
    const uint32_t split_after_layers =
        metal_graph_cuda_decode_split_after_layers();

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ok = metal_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             raw_row,
                                             n_raw,
                                             token);
        ds4_gpu_tensor *tmp = metal_graph_cur_hc(g);
        g->cur_hc_by_tier[g->active_tier] = metal_graph_after_ffn_hc(g);
        g->after_ffn_hc_by_tier[g->active_tier] = tmp;
        if (ok) ok = metal_graph_dspark_capture_decode_layer(g, il);
        if (ok && allow_split_flush && split_after_layers != 0 &&
            il + 1u == split_after_layers) {
            ok = ds4_gpu_flush_commands() != 0;
        }
    }

    if (ok && need_logits) {
        ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    }
    return ok;
}

static ds4_gpu_tensor *metal_graph_tensor_row_view(
        ds4_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values) {
    return ds4_gpu_tensor_view(base,
                                 (uint64_t)row * row_values * sizeof(float),
                                 row_values * sizeof(float));
}

/* Upload prompt token ids for kernels that need token-aware hash routing. */
static bool metal_graph_upload_prompt_tokens(
        ds4_gpu_tensor *out_tokens,
        const token_vec  *prompt,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (!out_tokens || pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) {
        return false;
    }

    int32_t *tokens = xmalloc((size_t)n_tokens * sizeof(tokens[0]));
    for (uint32_t i = 0; i < n_tokens; i++) tokens[i] = prompt->v[pos0 + i];

    const bool ok = ds4_gpu_tensor_write(out_tokens,
                                           0,
                                           tokens,
                                           (uint64_t)n_tokens * sizeof(tokens[0])) != 0;
    free(tokens);
    return ok;
}

/* Rebuild ratio-4 compressor state after chunked prefill so a following decode
 * token sees the same rolling compression window. */
static bool metal_graph_refresh_ratio4_compressor_state(
        ds4_gpu_graph  *g,
        const ds4_model  *model,
        ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const ds4_tensor *kv_weight,
        const ds4_tensor *score_weight,
        const ds4_tensor *ape,
        uint32_t          head_dim,
        uint32_t          width,
        uint32_t          pos0,
        uint32_t          n_tokens) {
    if (n_tokens < 4) {
        return true;
    }
    if (!g || !model || !state_kv || !state_score || !kv_weight || !score_weight || !ape ||
        head_dim == 0 || width == 0) {
        return false;
    }

    /*
     * The recurrent ratio-4 state is intentionally rebuilt from the last
     * four tokens using the small-batch projection kernel. The full-chunk
     * projection is already available, but it uses the matrix-matrix path;
     * mixing those two accumulation orders changes a few FP8 rounding
     * decisions in later chunks.
     */
    ds4_gpu_tensor *tail_hc = ds4_gpu_tensor_view(
            metal_graph_batch_attn_norm(g),
            (uint64_t)(n_tokens - 4u) * DS4_N_EMBD * sizeof(float),
            4ull * DS4_N_EMBD * sizeof(float));
    bool ok = tail_hc != NULL;
    if (!ok) {
        fprintf(stderr, "ds4: ratio-4 compressor tail view creation failed\n");
    }
    if (ok) {
#if defined(__APPLE__)
        ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_comp_kv(g),
                                        model->map,
                                        model->size,
                                        kv_weight->abs_offset,
                                        DS4_N_EMBD,
                                        width,
                                        tail_hc,
                                        4) != 0;
        if (ok) {
            ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_comp_sc(g),
                                            model->map,
                                            model->size,
                                            score_weight->abs_offset,
                                            DS4_N_EMBD,
                                            width,
                                            tail_hc,
                                            4) != 0;
        }
#else
        ok = ds4_gpu_matmul_f16_pair_tensor(metal_graph_batch_comp_kv(g),
                                             metal_graph_batch_comp_sc(g),
                                             model->map,
                                             model->size,
                                             kv_weight->abs_offset,
                                             score_weight->abs_offset,
                                             DS4_N_EMBD,
                                             width,
                                             tail_hc,
                                             4) != 0;
#endif
        if (!ok) {
            fprintf(stderr, "ds4: ratio-4 compressor tail projection failed\n");
        }
    }
    if (ok) {
        ok = ds4_gpu_compressor_prefill_state_ratio4_tensor(state_kv,
                                                              state_score,
                                                              metal_graph_batch_comp_kv(g),
                                                              metal_graph_batch_comp_sc(g),
                                                              model->map,
                                                              model->size,
                                                              ape->abs_offset,
                                                              ape->type,
                                                              head_dim,
                                                              pos0 + n_tokens - 4u) != 0;
        if (!ok) {
            fprintf(stderr, "ds4: ratio-4 compressor state refresh failed\n");
        }
    }
    ds4_gpu_tensor_free(tail_hc);
    return ok;
}

/* Seed the batched HC state from token ids: every HC stream starts as the same
 * 4096-wide embedding.  Long prefill chunks use the Metal get-rows/repeat
 * kernel so the CPU does not build and upload a large [token, HC, dim] tensor. */
static bool metal_graph_upload_prompt_embeddings_hc(
        ds4_gpu_tensor   *out_hc,
        ds4_gpu_tensor   *tokens,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;

    if (!tokens) return false;
    return ds4_gpu_embed_tokens_hc_tensor(out_hc,
                                          tokens,
                                          model->map,
                                          model->size,
                                          weights->token_embd->abs_offset,
                                          (uint32_t)weights->token_embd->dim[1],
                                          n_tokens,
                                          DS4_N_EMBD,
                                          DS4_N_HC) != 0;
}

static bool metal_graph_hc_rms_scale_project(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_scratch,
        const ds4_model        *model,
        const ds4_tensor       *weight,
        const ds4_gpu_tensor *x,
        uint64_t                in_dim,
        uint32_t                n_tokens) {
    if (!out || !norm_scratch || !model || !weight || !x ||
        in_dim > UINT32_MAX) {
        return false;
    }
#if defined(__APPLE__)
    return ds4_gpu_hc_rms_scale_project_f16_tensor(
               out,
               norm_scratch,
               model->map,
               model->size,
               weight->abs_offset,
               (uint32_t)in_dim,
               2u * DS4_N_HC + DS4_N_HC * DS4_N_HC,
               x,
               n_tokens,
               DS4_RMS_EPS) != 0;
#else
#if !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
    if (weight->type == DS4_TENSOR_F16 &&
        ds4_gpu_matmul_f16_rms_fold_tensor(
            out,
            model->map,
            model->size,
            weight->abs_offset,
            in_dim,
            2u * DS4_N_HC + DS4_N_HC * DS4_N_HC,
            x,
            n_tokens,
            DS4_RMS_EPS)) {
        return true;
    }
#endif
    bool ok = ds4_gpu_rms_norm_plain_rows_tensor(
                  norm_scratch,
                  x,
                  (uint32_t)in_dim,
                  n_tokens,
                  DS4_RMS_EPS) != 0;
    if (ok) {
        ok = ds4_gpu_matmul_f16_tensor(
                 out,
                 model->map,
                 model->size,
                 weight->abs_offset,
                 in_dim,
                 2u * DS4_N_HC + DS4_N_HC * DS4_N_HC,
                 norm_scratch,
                 n_tokens) != 0;
    }
    return ok;
#endif
}

static bool metal_graph_warmup_prefill_kernels(
        ds4_gpu_graph   *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           n_tokens) {
    static bool warmed = false;
    if (warmed) return true;
    if (getenv("DS4_CUDA_NO_PREFILL_KERNEL_WARMUP") != NULL) return true;

    /*
     * The first batched F16 matmul can pay CUDA's one-time setup cost.
     * cost. Run the same HC attention projection on scratch storage before the
     * measured prefill. The output is overwritten by the real graph.
     */
    if (n_tokens <= 8) return true;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = metal_graph_hc_rms_scale_project(
                 metal_graph_batch_hc_mix(g),
                 metal_graph_batch_flat_hc(g),
                 model,
                 weights->layer[0].hc_attn_fn,
                 metal_graph_batch_cur_hc(g),
                 hc_dim,
                 n_tokens);
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) {
        fprintf(stderr, "ds4: CUDA prefill kernel warmup failed\n");
        return false;
    }

    warmed = true;
    return true;
}

/* Encode the batched prefill attention half for one layer.  It mirrors the CPU
 * layer-major path: HC pre/norm, Q/KV, cache/compression, prefix attention. */
static bool metal_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    if (stage != NULL) {
        fprintf(stderr,
                "ds4: CUDA indexer stage layer=%u pos=%u tokens=%u comp=%u %s=%.3f ms\n",
                il,
                pos0,
                n_tokens,
                n_comp,
                stage,
                (now - *stage_t0) * 1000.0);
    }
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool metal_graph_env_value_eq(const char *v,
                                     size_t      n,
                                     const char *literal) {
    const size_t m = strlen(literal);
    if (n != m) return false;
    for (size_t i = 0; i < n; i++) {
        if (tolower((unsigned char)v[i]) !=
            tolower((unsigned char)literal[i])) {
            return false;
        }
    }
    return true;
}

static const char *metal_graph_env_trim(const char *v, size_t *len_out) {
    if (!v) {
        if (len_out) *len_out = 0;
        return NULL;
    }
    while (isspace((unsigned char)*v)) v++;
    size_t n = strlen(v);
    while (n > 0 && isspace((unsigned char)v[n - 1])) n--;
    if (len_out) *len_out = n;
    return v;
}

static bool metal_graph_profile_layer_value_match(const char *layer_env,
                                                  uint32_t    il) {
    size_t n = 0;
    layer_env = metal_graph_env_trim(layer_env, &n);
    if (!layer_env || n == 0) return true;

    char *end = NULL;
    const unsigned long layer = strtoul(layer_env, &end, 10);
    return end != layer_env &&
           (size_t)(end - layer_env) == n &&
           layer <= UINT32_MAX &&
           (uint32_t)layer == il;
}

static bool metal_graph_stage_profile_enabled_for_layer(
        const char *flag_env_name,
        const char *layer_env_name,
        uint32_t    il) {
    size_t flag_len = 0;
    const char *flag = metal_graph_env_trim(getenv(flag_env_name), &flag_len);
    if (!flag) return false;

    const char *layer_env = getenv(layer_env_name);
    const bool has_layer_filter = layer_env && layer_env[0];

    if (flag_len != 0) {
        if (metal_graph_env_value_eq(flag, flag_len, "0") ||
            metal_graph_env_value_eq(flag, flag_len, "false") ||
            metal_graph_env_value_eq(flag, flag_len, "no") ||
            metal_graph_env_value_eq(flag, flag_len, "off")) {
            return false;
        }
        if (!has_layer_filter &&
            !metal_graph_env_value_eq(flag, flag_len, "1") &&
            !metal_graph_env_value_eq(flag, flag_len, "true") &&
            !metal_graph_env_value_eq(flag, flag_len, "yes") &&
            !metal_graph_env_value_eq(flag, flag_len, "on") &&
            !metal_graph_env_value_eq(flag, flag_len, "all")) {
            return metal_graph_profile_layer_value_match(flag, il);
        }
    }

    return metal_graph_profile_layer_value_match(layer_env, il);
}

static bool metal_graph_layer_stage_profile_enabled(uint32_t il) {
    return metal_graph_stage_profile_enabled_for_layer(
                "DS4_CUDA_LAYER_STAGE_PROFILE",
                "DS4_CUDA_LAYER_STAGE_PROFILE_LAYER",
                il);
}

static bool metal_graph_decode_stage_profile_enabled(uint32_t il) {
    return metal_graph_stage_profile_enabled_for_layer(
                "DS4_CUDA_DECODE_STAGE_PROFILE",
                "DS4_CUDA_DECODE_STAGE_PROFILE_LAYER",
                il);
}

static bool metal_graph_layer_stage_profile_start(uint32_t il) {
    if (!metal_graph_layer_stage_profile_enabled(il)) return true;
    if (ds4_gpu_end_commands() == 0) return false;
    return ds4_gpu_begin_commands() != 0;
}

/* Optional prefill stage profiler. It intentionally ends the current CUDA
 * command buffer and waits, so the printed number includes encoding plus GPU
 * execution for the stage just emitted. This is disabled by default because it
 * adds synchronization points and changes scheduling. */
static bool metal_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    if (stage != NULL) {
        fprintf(stderr,
                "ds4: CUDA layer stage part=%s layer=%u pos=%u tokens=%u %s=%.3f ms\n",
                part,
                il,
                pos0,
                n_tokens,
                stage,
                (now - *stage_t0) * 1000.0);
    }
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static bool metal_graph_q_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (ds4_gpu_end_commands() == 0) return false;
    const double now = now_sec();
    fprintf(stderr,
            "ds4: CUDA Q path stage layer=%u pos=%u tokens=%u %s=%.3f ms\n",
            il,
            pos0,
            n_tokens,
            stage,
            (now - *stage_t0) * 1000.0);
    *stage_t0 = now;
    return ds4_gpu_begin_commands() != 0;
}

static ds4_gpu_tensor *metal_graph_tensor_row_range_view(
        ds4_gpu_tensor *base,
        uint32_t          row0,
        uint32_t          rows,
        uint64_t          row_values) {
    return ds4_gpu_tensor_view(base,
                                 (uint64_t)row0 * row_values * sizeof(float),
                                 (uint64_t)rows * row_values * sizeof(float));
}

#endif /* !DS4_NO_GPU */
