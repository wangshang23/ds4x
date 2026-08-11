#ifndef DS4_NO_GPU
/* =========================================================================
 * Metal Release Decode and Prefill.
 * =========================================================================
 *
 * Everything below is the user-facing Metal backend.  It uses the same layer
 * encoder as diagnostics, but diagnostics are not required for normal command
 * flow and their CPU reads stay outside these generation entry points.
 */

static DS4_MAYBE_UNUSED bool metal_graph_pre_m5_q2_decode_schedule_eligible(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             pos,
        bool                 allow_split_flush) {
#if defined(__APPLE__)
    if (!g || !weights || !allow_split_flush || DS4_N_LAYER <= 32u ||
        !ds4_gpu_device_is_pre_m5_apple_silicon() ||
        pos < 128u || pos >= 2816u ||
        g->quality || g->ssd_streaming || g->ssd_streaming_cold ||
        g->placement != NULL || g->tp_world > 1u ||
        getenv("DS4_METAL_DISABLE_PRE_M5_Q2_DECODE_SPLIT2_32") != NULL ||
        getenv("DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS") != NULL) {
        return false;
    }
    const ds4_layer_weights *layer = &weights->layer[4];
    return layer->ffn_gate_exps && layer->ffn_up_exps &&
           layer->ffn_down_exps &&
           layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
           layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
           layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
           DS4_N_EXPERT == 256u && DS4_N_EXPERT_USED == 6u &&
           layer->ffn_gate_exps->dim[0] == 4096u &&
           layer->ffn_gate_exps->dim[1] == 2048u &&
           layer->ffn_up_exps->dim[0] == 4096u &&
           layer->ffn_up_exps->dim[1] == 2048u &&
           layer->ffn_down_exps->dim[0] == 2048u &&
           layer->ffn_down_exps->dim[1] == 4096u &&
           routed_expert_row_bytes(layer->ffn_gate_exps) == 1056u &&
           routed_expert_row_bytes(layer->ffn_up_exps) == 1056u &&
           routed_expert_row_bytes(layer->ffn_down_exps) == 672u;
#else
    (void)g;
    (void)weights;
    (void)pos;
    (void)allow_split_flush;
    return false;
#endif
}

static uint32_t metal_graph_token_split_after_layers(void) {
    uint32_t split_after_layers = 4;
#ifndef DS4_ROCM_BUILD
    const char *split_env = getenv("DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS");
    if (split_env && split_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(split_env, &end, 10);
        if (end != split_env && v <= DS4_N_LAYER) split_after_layers = (uint32_t)v;
    }
#endif
    return split_after_layers;
}

static uint32_t metal_graph_token_adaptive_split_after_layers(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             pos,
        uint32_t             second_split_after_layers,
        bool                 allow_split_flush) {
    const uint32_t split_after_layers =
        metal_graph_token_split_after_layers();
#if defined(__APPLE__)
    /* In the validated 128..2815 Q2 window, routed decode exposes enough GPU
     * work after layer two to overlap encoding before the layer-32 flush. */
    const char *split_env = getenv("DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS");
    if ((!split_env || !split_env[0]) &&
        metal_graph_pre_m5_q2_decode_schedule_eligible(
            g, weights, pos, allow_split_flush)) {
        return 2u;
    }
    /* Layer 4 is a routed layer in the fixed DS4 tape and remains available in
     * the resident full-model path this schedule targets. */
    const ds4_layer_weights *layer = weights ? &weights->layer[4] : NULL;
    const bool mxfp4_routed =
        layer && layer->ffn_gate_exps && layer->ffn_up_exps &&
        layer->ffn_down_exps &&
        layer->ffn_gate_exps->type == DS4_TENSOR_MXFP4 &&
        layer->ffn_up_exps->type == DS4_TENSOR_MXFP4 &&
        layer->ffn_down_exps->type == DS4_TENSOR_MXFP4;
    /* Once the raw SWA window is full, the short-context 4/12 schedule can
     * submit its first useful work one routed layer earlier.  Five balanced
     * exact-output blocks (three eval-length and two 1,024-token blocks)
     * favored 3/12; keep the established 4/16 and adaptive 5-layer schedules
     * unchanged outside this early second-split window. */
    if (split_after_layers == 4u &&
        second_split_after_layers == 12u &&
        allow_split_flush && pos >= 128u && pos < 2048u &&
        g && !g->quality && !g->ssd_streaming && !g->ssd_streaming_cold &&
        g->tp_world != 2u && mxfp4_routed &&
        ds4_gpu_device_is_pre_m5_apple_silicon() &&
        getenv("DS4_METAL_DISABLE_PRE_M5_DECODE_EARLY_SPLIT3") == NULL) {
        return 3u;
    }
    /* At the 2K frontier the fifth layer adds enough work to overlap command
     * encoding.  Past 2816 the larger first command buffer starts losing to
     * the established four-layer split, so retain that schedule there. */
    if (split_after_layers == 4u &&
        second_split_after_layers == 0u &&
        allow_split_flush && pos >= 2048u && pos < 2816u &&
        g && !g->quality && !g->ssd_streaming && !g->ssd_streaming_cold &&
        g->tp_world != 2u && mxfp4_routed &&
        ds4_gpu_device_is_pre_m5_apple_silicon() &&
        getenv("DS4_METAL_DISABLE_PRE_M5_DECODE_EARLY_SPLIT5") == NULL) {
        return 5u;
    }
#else
    (void)g;
    (void)weights;
    (void)pos;
    (void)second_split_after_layers;
    (void)allow_split_flush;
#endif
    return split_after_layers;
}

static DS4_MAYBE_UNUSED bool metal_graph_decode_pipeline_fast_lookup_eligible(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             pos,
        bool                 allow_split_flush) {
#if defined(__APPLE__)
    if (!g || !weights || !allow_split_flush ||
        g->quality || g->ssd_streaming || g->ssd_streaming_cold ||
        g->placement != NULL || g->tp_world > 1u || g->mtp_enabled ||
        DS4_N_LAYER <= 4u) {
        return false;
    }
    /* Eval prompts commonly decode below the 2K benchmark frontier.  The
     * mirror key is the complete pipeline specialization and is independent
     * of position, so the same lookup is valid there; keep an early-only
     * rollback for attribution without disabling the established 2K path. */
    if (pos < 2048u &&
        getenv("DS4_METAL_DISABLE_PRE_M5_DECODE_EARLY_PIPELINE_FAST_LOOKUP") != NULL) {
        return false;
    }

    const ds4_layer_weights *layer = &weights->layer[4];
    const bool mxfp4_routed =
        layer->ffn_gate_exps && layer->ffn_up_exps &&
        layer->ffn_down_exps &&
        layer->ffn_gate_exps->type == DS4_TENSOR_MXFP4 &&
        layer->ffn_up_exps->type == DS4_TENSOR_MXFP4 &&
        layer->ffn_down_exps->type == DS4_TENSOR_MXFP4;
    return mxfp4_routed && ds4_gpu_device_is_pre_m5_apple_silicon() &&
           getenv("DS4_METAL_DISABLE_PRE_M5_DECODE_PIPELINE_FAST_LOOKUP") == NULL;
#else
    (void)g;
    (void)weights;
    (void)pos;
    (void)allow_split_flush;
    return false;
#endif
}

static uint32_t metal_graph_token_second_split_after_layers(void) {
    uint32_t split_after_layers = 0;
#if defined(__APPLE__)
    const char *split_env =
        getenv("DS4_METAL_GRAPH_TOKEN_SECOND_SPLIT_LAYERS");
    if (split_env && split_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(split_env, &end, 10);
        if (end != split_env && v <= DS4_N_LAYER) {
            split_after_layers = (uint32_t)v;
        }
    }
#endif
    return split_after_layers;
}

/* Automatic second command-buffer splits for eligible resident pre-M5 decode.
 * They close the idle bubble left while the host encodes a long tail after the
 * first submission. Only command-buffer boundaries move, so kernel math,
 * ordering, and outputs are unchanged. An explicit
 * DS4_METAL_GRAPH_TOKEN_SECOND_SPLIT_LAYERS always wins. */
static uint32_t metal_graph_token_adaptive_second_split_after_layers(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             pos,
        uint32_t             env_second_split_after_layers,
        bool                 allow_split_flush) {
#if defined(__APPLE__)
    const char *second_split_env =
        getenv("DS4_METAL_GRAPH_TOKEN_SECOND_SPLIT_LAYERS");
    if (second_split_env && second_split_env[0]) {
        return env_second_split_after_layers;
    }
    if (metal_graph_token_split_after_layers() != 0u &&
        metal_graph_pre_m5_q2_decode_schedule_eligible(
            g, weights, pos, allow_split_flush)) {
        return 32u;
    }
    /* Mirror the R4 adaptive-first-split guards: layer 4 is a routed layer in
     * the fixed DS4 tape and stands in for the resident full-model path this
     * schedule targets. */
    const ds4_layer_weights *layer = weights ? &weights->layer[4] : NULL;
    const bool mxfp4_routed =
        layer && layer->ffn_gate_exps && layer->ffn_up_exps &&
        layer->ffn_down_exps &&
        layer->ffn_gate_exps->type == DS4_TENSOR_MXFP4 &&
        layer->ffn_up_exps->type == DS4_TENSOR_MXFP4 &&
        layer->ffn_down_exps->type == DS4_TENSOR_MXFP4;
    const bool eligible =
        allow_split_flush && pos < 3328u &&
        g && !g->quality && !g->ssd_streaming && !g->ssd_streaming_cold &&
        g->tp_world != 2u && mxfp4_routed &&
        ds4_gpu_device_is_pre_m5_apple_silicon();
    /* Once the raw SWA window is full, short eval prompts leave less GPU work
     * in the four-layer prefix while the host still has the same remaining
     * tape to encode.  Flushing again after layer 12 starts that middle work
     * soon enough to cover the early decode bubble; 12 measured better than
     * 8, 16, and 20 below position 2048.  Positions below 128 retain the
     * single-tail schedule, which is slightly faster before the window fills.
     * Keep a separate rollback so the established 2K schedule remains
     * independently attributable. */
    if (eligible && pos >= 128u && pos < 2048u && DS4_N_LAYER > 12u &&
        getenv("DS4_METAL_DISABLE_PRE_M5_DECODE_EARLY_SECOND_SPLIT12") == NULL) {
        return 12u;
    }
    /* The encode/GPU overlap bubble the second split closes persists past
     * 2815: attention cost grows only slowly through the compressed-KV
     * window (raw SWA saturates at 128 rows) while encode cost per layer is
     * constant, so the same 4/16 schedule stays profitable through 3327. */
    if (eligible && pos >= 2048u && DS4_N_LAYER > 16u &&
        getenv("DS4_METAL_DISABLE_PRE_M5_DECODE_SECOND_SPLIT16") == NULL) {
        return 16u;
    }
#else
    (void)g;
    (void)weights;
    (void)pos;
    (void)allow_split_flush;
#endif
    return env_second_split_after_layers;
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

/* Encode a full single-token decode step on Metal.  This is the generation
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
        fprintf(stderr, "ds4: Metal graph raw KV cache is not allocated\n");
        return false;
    }
    /* Under the vocab split both ranks materialize their logits half. */
    if (g->tp_world == 2 && g->tp_rank == 1 &&
        !g->tp_logits_half) need_logits = false;
    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
    metal_graph_dspark_capture_begin(g);

    /* write the embedded token on the embedding tier. Single-
     * tier: emb_tier == 0 == active_tier; no-op. Multi-tier: switch to
     * emb_tier (no cross-device copy needed — embed writes from scratch). */
    if (g->placement) {
        if (!metal_graph_set_active_tier_decode(g, g->emb_tier)) return false;
    }
#if defined(__APPLE__)
    /* Metal command encoding is serialized by the backend. Preserve a caller
     * hint across this nested scope, but keep it inactive unless this exact
     * token satisfies the decode-only guards. */
    const int previous_pipeline_fast_lookup =
        ds4_gpu_set_decode_pipeline_fast_lookup(0);
    const bool pipeline_fast_lookup_active =
        metal_graph_decode_pipeline_fast_lookup_eligible(
            g, weights, pos, allow_split_flush);
    if (pipeline_fast_lookup_active) {
        (void)ds4_gpu_set_decode_pipeline_fast_lookup(1);
    }
#endif
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
    const uint32_t second_split_after_layers =
        metal_graph_token_adaptive_second_split_after_layers(
            g,
            weights,
            pos,
            metal_graph_token_second_split_after_layers(),
            allow_split_flush);
    const uint32_t split_after_layers =
        metal_graph_token_adaptive_split_after_layers(
            g,
            weights,
            pos,
            second_split_after_layers,
            allow_split_flush);

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
        /* A TP gate uses one monotonic shared event for the whole token. A
         * later command buffer may signal a higher value while the prefix is
         * blocked at an earlier gate, making the transport consume a slab
         * slot before its payload is ready. Keep each TP token in one command
         * buffer; non-TP decode retains the encode/execute overlap. */
        if (ok && allow_split_flush && g->tp_world != 2 &&
            ((split_after_layers != 0 && il + 1u == split_after_layers) ||
             (second_split_after_layers != 0 &&
              il + 1u == second_split_after_layers))) {
            ok = ds4_gpu_flush_commands() != 0;
        }
    }

    if (ok && need_logits) {
        ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    }
#if defined(__APPLE__)
    (void)ds4_gpu_set_decode_pipeline_fast_lookup(
        previous_pipeline_fast_lookup);
#endif
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

/* CPU fallback for seeding batched HC state from token embeddings.  It is still
 * useful for tiny speculative verifier batches where a separate GPU embedding
 * command buffer costs more than the small host write. */
static bool metal_graph_upload_prompt_embeddings_hc_cpu(
        ds4_gpu_tensor   *out_hc,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens) {
    if (pos0 > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - pos0) return false;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t total = (uint64_t)n_tokens * hc_dim;
    float *hc = xmalloc((size_t)total * sizeof(hc[0]));
    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(plain[0]));

    for (uint32_t t = 0; t < n_tokens; t++) {
        embed_token_f16(model, weights, prompt->v[pos0 + t], plain);
        float *dst = hc + (uint64_t)t * hc_dim;
        for (uint32_t h = 0; h < DS4_N_HC; h++) {
            memcpy(dst + (uint64_t)h * DS4_N_EMBD,
                   plain,
                   (size_t)DS4_N_EMBD * sizeof(plain[0]));
        }
    }

    const bool ok = ds4_gpu_tensor_write(out_hc, 0, hc, total * sizeof(hc[0])) != 0;
    free(plain);
    free(hc);
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

    uint32_t gpu_min = 512;
#ifndef DS4_ROCM_BUILD
    const char *gpu_min_env = getenv("DS4_METAL_GPU_BATCH_EMBED_MIN");
    if (gpu_min_env && gpu_min_env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(gpu_min_env, &end, 10);
        if (end != gpu_min_env && v <= UINT32_MAX) gpu_min = (uint32_t)v;
    }
#endif

    if (tokens && n_tokens >= gpu_min) {
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

    return metal_graph_upload_prompt_embeddings_hc_cpu(out_hc,
                                                       model,
                                                       weights,
                                                       prompt,
                                                       pos0,
                                                       n_tokens);
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
    if (g && g->ssd_streaming) return true;
    if (warmed) return true;
#ifndef DS4_ROCM_BUILD
    if (getenv("DS4_METAL_NO_PREFILL_KERNEL_WARMUP") != NULL) return true;
#endif

    /*
     * The first batched F16 matmul can pay Metal's one-time pipeline execution
     * cost. Run the same HC attention projection on scratch storage before the
     * measured prefill. The output is overwritten by the real graph.
     */
    if (n_tokens <= 8) return true;

    /* (B6 fix, ): warm-up uses layer-0's hc_attn_fn
     * weight, which in multi-tier is resolved on placement[1]'s tier.
     * Switch active_tier so the F16 matmul reads/writes the correct
     * Class P scratch and resolves the weight on the right device.
     * Single-tier (g->placement == NULL): no-op. */
    if (g->placement) {
        if (!metal_graph_set_active_tier_batch(g, g->placement[1], n_tokens)) return false;
    }
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
        fprintf(stderr, "ds4: Metal prefill kernel warmup failed\n");
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
                "ds4: metal indexer stage layer=%u pos=%u tokens=%u comp=%u %s=%.3f ms\n",
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
                "DS4_ROCM_LAYER_STAGE_PROFILE",
                "DS4_ROCM_LAYER_STAGE_PROFILE_LAYER",
                il) ||
           metal_graph_stage_profile_enabled_for_layer(
                "DS4_METAL_LAYER_STAGE_PROFILE",
                "DS4_METAL_LAYER_STAGE_PROFILE_LAYER",
                il);
}

static bool metal_graph_decode_stage_profile_enabled(uint32_t il) {
    return metal_graph_stage_profile_enabled_for_layer(
                "DS4_ROCM_DECODE_STAGE_PROFILE",
                "DS4_ROCM_DECODE_STAGE_PROFILE_LAYER",
                il) ||
           metal_graph_stage_profile_enabled_for_layer(
                "DS4_METAL_DECODE_STAGE_PROFILE",
                "DS4_METAL_DECODE_STAGE_PROFILE_LAYER",
                il);
}

static bool metal_graph_layer_stage_profile_start(uint32_t il) {
    if (!metal_graph_layer_stage_profile_enabled(il)) return true;
    if (ds4_gpu_end_commands() == 0) return false;
    return ds4_gpu_begin_commands() != 0;
}

/* Optional prefill stage profiler. It intentionally ends the current Metal
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
                "ds4: metal layer stage part=%s layer=%u pos=%u tokens=%u %s=%.3f ms\n",
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
            "ds4: metal Q path stage layer=%u pos=%u tokens=%u %s=%.3f ms\n",
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

/* TP prefill threshold for row-splitting the replicated shared expert.
 * Routed experts remain ownership-split at every batch size. */
static uint32_t metal_graph_tp_prefill_split_min(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = 32;
        const char *env = getenv("DS4_TP_PREFILL_SPLIT_MIN");
        if (env && env[0]) cached = atoi(env);
        if (cached < 2) cached = 2;
    }
    return (uint32_t)cached;
}

/* Opt-in sub-chunk gate pipelining for the TP prefill row swaps.  Must be
 * set on BOTH ranks (it changes the per-layer gate count; asymmetric
 * settings deadlock the big gates).  Default off: measured net-negative
 * on the M5 Max pair, see the pipelined blocks for the numbers. */
static bool metal_graph_tp_subgate_pipeline(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_TP_SUBGATE_PIPELINE");
        cached = env && env[0] && atoi(env) != 0;
    }
    return cached != 0;
}

#endif /* !DS4_NO_GPU */
