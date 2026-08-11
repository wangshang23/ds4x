
#ifdef DS4_NO_GPU
ds4_context_memory ds4_context_memory_estimate_with_prefill_mode(
        ds4_backend backend,
        int         ctx_size,
        uint32_t    prefill_chunk,
        bool        ssd_streaming) {
    (void)backend;
    (void)prefill_chunk;
    (void)ssd_streaming;
    ds4_context_memory m = {0};
    uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;

    m.raw_cap = ds4_default_raw_cap(ctx);
    m.raw_bytes = (uint64_t)DS4_N_LAYER *
                  m.raw_cap *
                  DS4_N_HEAD_DIM *
                  sizeof(float);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t comp_cap = ctx / ratio + 2u;
        if (ratio == 4) m.comp_cap = comp_cap;
        m.compressed_bytes += (uint64_t)comp_cap *
                              DS4_N_HEAD_DIM *
                              sizeof(float);
        if (ratio == 4) {
            m.compressed_bytes += (uint64_t)comp_cap *
                                  DS4_N_INDEXER_HEAD_DIM *
                                  sizeof(float);
        }
    }
    if (m.comp_cap == 0) m.comp_cap = ctx / 4u + 2u;
    m.scratch_bytes = ((uint64_t)(m.raw_cap + m.comp_cap) * sizeof(float)) +
                      ((uint64_t)m.comp_cap * sizeof(float)) +
                      ((uint64_t)m.comp_cap * sizeof(bool));
    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int         ctx_size,
        uint32_t    prefill_chunk) {
    return ds4_context_memory_estimate_with_prefill_mode(backend,
                                                         ctx_size,
                                                         prefill_chunk,
                                                         false);
}

ds4_context_memory ds4_context_memory_estimate(ds4_backend backend,
                                               int         ctx_size) {
    return ds4_context_memory_estimate_with_prefill(backend, ctx_size, 0);
}
#endif

/* Per-layer KV byte estimate at a given context size for the Spark CUDA graph.
 * Mirrors the per-layer
 * accounting the graph backend uses at allocation time, so the multi-GPU
 * layer packer (engine_compute_entry_bytes) can price each layer
 * individually instead of dividing a global total uniformly across all
 * layers. Heterogeneous attention compression (raw / ratio-4 /
 * ratio-128) is then represented faithfully in the placement plan.
 *
 * Visible in both GPU and DS4_NO_GPU builds so the placement-test
 * harness (which builds ds4.c with -DDS4_NO_GPU) can exercise the same
 * packer code paths.
 *
 * Accounting (matches the graph branch of ds4_context_memory_estimate):
 *   raw_bytes_per_layer       = raw_cap * DS4_SPARK_KV_ROW_BYTES
 *                               applied to every layer.
 *   compressed_bytes_per_layer = ratio==0 -> 0
 *                                ratio!=0 -> (ctx/ratio + 2) *
 *                                            DS4_SPARK_KV_ROW_BYTES
 *                                            + (ratio==4 ? (ctx/ratio + 2) *
 *                                                          DS4_SPARK_INDEX_ROW_BYTES
 *                                                        : 0)
 *
 * Per-tier scratch buffers (indexer_scores_by_tier, comp_mask_by_tier,
 * attn_comp_stage_by_tier, chunked-prefill batch_*_by_tier, head extras)
 * are NOT included here. They are reserved separately by
 * engine_per_tier_graph_overhead_bytes() and pre-subtracted from each
 * device's vram_bytes in engine_classify_multi_tier. Counting them per
 * layer here would double-count them once per layer.
 *
 * Invariant (by construction): sum over il of
 * engine_per_layer_kv_bytes_planner(il, ctx, prefill_chunk) ==
 *   DS4_N_LAYER * raw_cap * DS4_SPARK_KV_ROW_BYTES
 *   + sum_over_ratio_layers(compressed_per_layer)
 */
/* Planner equivalents of metal_graph_prefill_cap_for_prompt and
 * metal_graph_raw_cap_for_context. The graph variants live inside
 * `#ifndef DS4_NO_GPU` blocks; these inline copies replicate the same numeric
 * math so the planner can be called from both build flavors.
 *
 * If the env-knob behavior in the graph helpers ever changes, update
 * these two helpers in lockstep. Tested via:
 *   sum(engine_per_layer_kv_bytes_planner(il, ctx, prefill_chunk)
 *       for il in 0..N) ==
 *   packed ds4_context_memory_estimate(CUDA, ctx).total_bytes.
 */
static uint32_t engine_planner_prefill_cap(int prompt_len,
                                           uint32_t requested_chunk) {
    return ds4_prefill_cap_for_prompt(prompt_len, requested_chunk);
}

static uint32_t engine_planner_raw_cap(int ctx_size, uint32_t prefill_cap) {
    if (ctx_size <= 0) return 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > (uint32_t)ctx_size) raw_window = (uint32_t)ctx_size;
    if (raw_window == 0) raw_window = 1;

    /* Pad to 256-row multiple so the planner matches the graph layout. */
    uint64_t wanted = (uint64_t)raw_window + prefill_cap;
    if (wanted > (uint32_t)ctx_size) wanted = (uint32_t)ctx_size;
    if (wanted == 0) wanted = 1;
    /* align_up to 256 — inline since align_up is defined upstream. */
    const uint64_t align = 256u;
    wanted = (wanted + align - 1u) & ~(align - 1u);
    if (wanted > 8192u) wanted = 8192u;
    uint32_t raw_cap = (uint32_t)wanted;
    if (raw_cap < raw_window) raw_cap = raw_window;

    /* Env override (matches metal_graph_raw_cap_for_context behavior). */
    const char *env = getenv("DS4_METAL_GRAPH_RAW_CAP");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env && v > 0) {
            raw_cap = (uint32_t)v;
            if (raw_cap > (uint32_t)ctx_size) raw_cap = (uint32_t)ctx_size;
            if (raw_cap > 8192u) raw_cap = 8192u;
            if (raw_cap < raw_window) raw_cap = raw_window;
        }
    }
    return raw_cap;
}

/* Mirrors the GPU cache-storage choice while remaining visible to the
 * DS4_NO_GPU placement tests. */
#define DS4_PLANNER_ATTN_COMP_CACHE_F16 0

static bool engine_glm_layer_uses_full_indexer(uint32_t il) {
    if (il < DS4_N_LEADING_DENSE) return true;
    return il >= 6u && ((il - 6u) % 4u) == 0u;
}

static uint32_t engine_placement_session_count(const ds4_engine *e) {
    return e && e->placement_session_count_hint > 1
        ? (uint32_t)e->placement_session_count_hint : 1u;
}

static size_t engine_size_mul_sat(size_t bytes, uint32_t count) {
    if (count != 0u && bytes > SIZE_MAX / count) return SIZE_MAX;
    return bytes * count;
}

/* GLM keeps one compact DSA row for every logical context position. These
 * caches are allocated on the same tier as their transformer layer, so they
 * must be priced per layer by the multi-GPU packer. */
static size_t engine_glm_per_layer_kv_bytes_planner(uint32_t il,
                                                    int      ctx_size) {
    if (ctx_size <= 0 || DS4_N_LAYER <= DS4_N_NEXTN_PREDICT) return 0;
    const uint32_t normal_layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    if (il >= normal_layers) return 0;

    const uint64_t elem_bytes =
        DS4_PLANNER_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float);
    uint64_t row_width = (uint64_t)DS4_N_KV_LORA + DS4_N_ROT;
    if (engine_glm_layer_uses_full_indexer(il)) {
        row_width += DS4_N_INDEXER_HEAD_DIM;
    }
    const uint64_t bytes = (uint64_t)(uint32_t)ctx_size * row_width * elem_bytes;
    return bytes > SIZE_MAX ? SIZE_MAX : (size_t)bytes;
}

static size_t engine_per_layer_kv_bytes_planner(uint32_t il,
                                                int ctx_size,
                                                uint32_t prefill_chunk) {
    if (ctx_size <= 0) return 0;
    if (il >= DS4_N_LAYER) return 0;
    const uint32_t ctx = (uint32_t)ctx_size;

    /* Every layer gets a packed raw ring sized by raw_cap. raw_cap factors in
     * raw-window padding plus prefill_cap and clamps to [raw_window, 8192]. */
    const uint32_t prefill_cap =
        engine_planner_prefill_cap((int)ctx, prefill_chunk);
    const uint32_t raw_cap = engine_planner_raw_cap((int)ctx, prefill_cap);
    size_t bytes = (size_t)raw_cap * DS4_SPARK_KV_ROW_BYTES;

    /* Ratio-4 layers own packed KV and packed indexer rows; ratio-128 layers
     * own packed KV rows only. */
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ratio != 0) {
        const uint32_t layer_comp_cap = ctx / ratio + 2u;
        bytes += (size_t)layer_comp_cap * DS4_SPARK_KV_ROW_BYTES;
        if (ratio == 4) {
            bytes += (size_t)layer_comp_cap * DS4_SPARK_INDEX_ROW_BYTES;
        }
    }

    /* Per-tier scratch (indexer_scores_by_tier, comp_mask_by_tier,
     * attn_comp_stage_by_tier, chunked-prefill batch_*_by_tier, head
     * extras) is accounted for separately by
     * engine_per_tier_graph_overhead_bytes() and pre-subtracted from
     * each device's vram_bytes in engine_classify_multi_tier. This
     * helper returns per-layer KV/index cache ONLY. Charging scratch
     * both here (per layer) and there (per tier) would double-count —
     * at large ctx the duplicate can falsely refuse layouts that
     * actually fit. */
    return bytes;
}

/* Per-used-tier Class-P graph overhead estimate. Mirrors the
 * `*_by_tier[t]` allocations in
 * metal_graph_alloc_raw_cap (ds4.c:10664-10686 + 10760-10800 + 10806-10816
 * for head extras + 10844 for prefill_tokens + 10852-10892 for batch
 * chunked-prefill scratch).
 *
 * The runtime loop replicates an entire set of Class-P kernel-scratch
 * buffers on EVERY used tier. The packer's budget math (entry_bytes) only
 * accounts for tensor weights + per-layer KV — it never reserved this
 * per-tier scratch, so a layout that fit by entry_bytes could still
 * late-OOM at session_create. This helper returns the exact byte total,
 * which is then pre-subtracted from EVERY device's vram_bytes inside
 * engine_classify_multi_tier (conservative: even tiers that end up unused
 * still reserve the overhead, so the packer cannot accept a layout that
 * would later OOM).
 *
 * Head-tier extras (output_pre/weights/embd/norm/logits at ds4.c:10806-10816)
 * and prefill_tokens (ds4.c:10844 — emb_tier only) are charged to ALL tiers
 * conservatively: only the head_tier / emb_tier actually pays at runtime, but
 * since the pre-subtract is a single scalar applied to every device, charging
 * to all is the simplest correct posture. The over-charge is a few MB total.
 *
 * Uses g_ds4_shape compile-time constants (DS4_N_*) and ctx-derived caps
 * from engine_planner_prefill_cap. The runtime computes some dims from
 * model weights (e.g. q_rank = layer->attn_q_a->dim[1]), but
 * tensor_expect_layout enforces those dims == DS4_N_LORA_Q etc., so the
 * compile-time aliases used here are byte-equivalent.
 *
 * Visible in both GPU and DS4_NO_GPU builds. */
static size_t engine_per_tier_graph_overhead_bytes(const ds4_engine *e) {
#ifndef DS4_NO_GPU
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        const int est_ctx = e && e->placement_ctx_hint > 0 ?
            e->placement_ctx_hint : 4096;
        const uint32_t ctx = (uint32_t)est_ctx;
        const bool ssd_streaming = e && e->ssd_streaming;
        const uint32_t work_ctx =
            glm_graph_full_attention_cap(ctx, ssd_streaming);
        const uint32_t compact_cap =
            glm_graph_compact_cache_initial_cap(ctx, work_ctx);
        const ds4_context_memory mem =
            glm_graph_context_memory_estimate_for_compact_cap(
                    ctx, work_ctx, compact_cap, ssd_streaming);
        const size_t scratch = mem.scratch_bytes > SIZE_MAX ?
            SIZE_MAX : (size_t)mem.scratch_bytes;
        return engine_size_mul_sat(scratch,
                                   engine_placement_session_count(e));
    }
#endif

    /* Local dim aliases — same values the runtime per-tier scratch loop
     * reads. The runtime reads q_rank etc. from layer weights; the
     * config_validate_model path (called from ds4_engine_open_internal
     * before classify) enforces dim equivalence with DS4_N_*, so these
     * aliases are byte-equivalent. */
    const uint64_t hc_dim         = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc         = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank         = DS4_N_LORA_Q;
    const uint64_t q_dim          = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t low_dim        = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim      = DS4_N_OUT_GROUP ?
                                    (uint64_t)DS4_N_HEAD_DIM *
                                    (DS4_N_HEAD / DS4_N_OUT_GROUP) : 0;
    const uint64_t shared_dim     = DS4_N_FF_EXP;
    const uint64_t routed_mid_dim = DS4_N_FF_EXP;
    const uint64_t vocab_dim      = DS4_N_VOCAB;
    uint64_t output_logits_elems  = vocab_dim;
    const int planner_n_gpus = e ? e->gpu_cfg.n_gpus : 0;
#if !defined(__APPLE__)
    const char *tp_output_env = getenv("DS4_CUDA_TP_OUTPUT");
    const bool tp_decode_requested = e && e->cuda_tensor_parallel;
    const bool tp_output_requested =
        !tp_output_env || !tp_output_env[0] || strcmp(tp_output_env, "0") != 0;
    if (planner_n_gpus >= 2 && (planner_n_gpus & 1) == 0 &&
        tp_decode_requested && tp_output_requested) {
        uint32_t output_ways = 8u;
        const char *ways_env = getenv("DS4_CUDA_TP_OUTPUT_WAYS");
        if (ways_env && ways_env[0]) {
            char *end = NULL;
            const unsigned long parsed = strtoul(ways_env, &end, 10);
            output_ways = end != ways_env && *end == '\0' &&
                          parsed >= 2u && parsed <= DS4_MAX_GPUS
                ? (uint32_t)parsed : 2u;
        }
        if (output_ways > (uint32_t)planner_n_gpus) {
            output_ways = (uint32_t)planner_n_gpus;
        }
        const uint64_t max_shard_vocab =
            (vocab_dim + output_ways - 1u) / output_ways;
        const uint64_t spec_shard_elems =
            (uint64_t)DS4_DSPARK_MAX_BLOCK_SIZE * max_shard_vocab;
        if (spec_shard_elems > output_logits_elems) {
            output_logits_elems = spec_shard_elems;
        }
    }
#else
    (void)planner_n_gpus;
#endif
    const uint64_t comp_width_max =
        2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
                ? DS4_N_HEAD_DIM
                : DS4_N_INDEXER_HEAD_DIM);
    const uint64_t indexer_q_dim  =
        (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;

    /* ctx-derived caps. The runtime uses prefill_cap derived from the
     * chunked-prefill batch size. The planner has no prompt yet, so it uses the
     * placement context and the engine's effective chunk setting. */
    const int est_ctx = (e->placement_ctx_hint > 0) ? e->placement_ctx_hint
                                                    : 4096;
    const uint32_t prefill_cap =
        engine_planner_prefill_cap(est_ctx, e ? e->prefill_chunk : 0);

    /* comp_cap and attn_comp_stage_cap: same formula as runtime line 10597.
     * If no layer has a compression ratio set (test path, where
     * g_ds4_compress_ratios is zero-init), min_ratio falls back to ctx,
     * matching runtime line 10596. */
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) {
        min_ratio = est_ctx > 0 ? (uint32_t)est_ctx : 1u;
    }
    uint32_t comp_cap = (uint32_t)est_ctx / min_ratio + 2u;
    if (comp_cap < 2u) comp_cap = 2u;
    uint32_t attn_comp_stage_cap = prefill_cap / min_ratio + 2u;
    if (attn_comp_stage_cap < 2u) attn_comp_stage_cap = 2u;
    const uint64_t pc = (uint64_t)prefill_cap;

    size_t total = 0;

    /* === Class P decode HC scratch (mirrors ds4.c:10664-10686). The
     * hc_pre/hc_post/hc_comb buffers are VIEWS of hc_split and are NOT
     * counted (they would double-count). === */
    total += hc_dim * sizeof(float);                       /* cur_hc_by_tier */
    total += hc_dim * sizeof(float);                       /* flat_hc_by_tier */
    total += mix_hc * sizeof(float);                       /* hc_mix_by_tier */
    total += mix_hc * sizeof(float);                       /* hc_split_by_tier */
    /* hc_pre_by_tier, hc_post_by_tier, hc_comb_by_tier — VIEWS of hc_split. */
    total += (uint64_t)DS4_N_EMBD * sizeof(float);         /* attn_cur_by_tier */
    total += (uint64_t)DS4_N_EMBD * sizeof(float);         /* attn_norm_by_tier */
    total += q_rank * sizeof(float);                       /* qr_by_tier */
    total += q_rank * sizeof(float);                       /* qr_norm_by_tier */
    total += q_dim * sizeof(float);                        /* q_by_tier */
    total += (uint64_t)DS4_N_HEAD_DIM * sizeof(float);     /* kv_raw_by_tier */
    total += (uint64_t)DS4_N_HEAD_DIM * sizeof(float);     /* kv_by_tier */

    /* === Class P FFN / routed-expert state (mirrors ds4.c:10760-10800). === */
    total += comp_width_max * sizeof(float);               /* comp_kv_cur_by_tier */
    total += comp_width_max * sizeof(float);               /* comp_sc_cur_by_tier */
    total += 2ull * DS4_N_INDEXER_HEAD_DIM * sizeof(float);  /* index_comp_kv_cur_by_tier */
    total += 2ull * DS4_N_INDEXER_HEAD_DIM * sizeof(float);  /* index_comp_sc_cur_by_tier */
    total += (uint64_t)attn_comp_stage_cap *
             DS4_N_HEAD_DIM * sizeof(float);               /* attn_comp_stage_by_tier */
    total += (uint64_t)attn_comp_stage_cap *
             DS4_N_INDEXER_HEAD_DIM * sizeof(float);       /* index_comp_stage_by_tier */
    total += indexer_q_dim * sizeof(float);                /* indexer_q_by_tier */
    total += (uint64_t)DS4_N_INDEXER_HEAD * sizeof(float); /* indexer_weights_by_tier */
    total += (uint64_t)comp_cap * pc * sizeof(float);      /* indexer_scores_by_tier */
    total += (uint64_t)comp_cap * pc * sizeof(float);      /* comp_mask_by_tier */
    const uint64_t top_k =
        (uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u);
    total += top_k * pc * sizeof(uint32_t);                /* comp_selected_by_tier */
    total += q_dim * sizeof(float);                        /* heads_by_tier */
    total += low_dim * sizeof(float);                      /* attn_low_by_tier */
    total += (uint64_t)DS4_N_EMBD * sizeof(float);         /* attn_out_by_tier */
    total += hc_dim * sizeof(float);                       /* after_attn_hc_by_tier */
    total += (uint64_t)DS4_N_EMBD * sizeof(float);         /* ffn_cur_by_tier */
    total += (uint64_t)DS4_N_EMBD * sizeof(float);         /* ffn_norm_by_tier */
    total += shared_dim * sizeof(float);                   /* shared_gate_by_tier */
    total += shared_dim * sizeof(float);                   /* shared_up_by_tier */
    total += shared_dim * sizeof(float);                   /* shared_mid_by_tier */
    total += (uint64_t)DS4_N_EMBD * sizeof(float);         /* shared_out_by_tier */
    total += (uint64_t)DS4_N_EXPERT * sizeof(float);       /* router_logits_by_tier */
    total += (uint64_t)DS4_N_EXPERT * sizeof(float);       /* router_probs_by_tier */
    total += (uint64_t)DS4_N_EXPERT_USED * sizeof(int);    /* router_selected_by_tier */
    total += (uint64_t)DS4_N_EXPERT_USED * sizeof(float);  /* router_weights_by_tier */
    total += (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float); /* routed_gate */
    total += (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float); /* routed_up */
    total += (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float); /* routed_mid */
    total += (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float);     /* routed_down */
    total += (uint64_t)DS4_N_EMBD * sizeof(float);                         /* routed_out */
    total += DS4_CUDA_TP_PEER_TMP_BYTES;                                   /* tp_peer_tmp_by_tier */
    total += hc_dim * sizeof(float);                                       /* after_ffn_hc */

    /* === Class P chunked-prefill batch scratch (mirrors allocation above).
     * These are the LARGEST per-tier allocations (e.g. batch_cur_hc =
     * pc * hc_dim * float = ~256 MiB at default prefill_cap=4096). Without
     * them the pre-subtract is meaningless for any non-trivial ctx. === */
    total += pc * hc_dim * sizeof(float);                  /* batch_cur_hc_by_tier */
    total += pc * hc_dim * sizeof(float);                  /* batch_next_hc_by_tier */
    total += pc * hc_dim * sizeof(float);                  /* batch_flat_hc_by_tier */
    total += pc * mix_hc * sizeof(float);                  /* batch_hc_mix_by_tier */
    total += pc * mix_hc * sizeof(float);                  /* batch_hc_split_by_tier */
    total += pc * (uint64_t)DS4_N_EMBD * sizeof(float);    /* batch_attn_cur_by_tier */
    total += pc * (uint64_t)DS4_N_EMBD * sizeof(float);    /* batch_attn_norm_by_tier */
    total += pc * q_rank * sizeof(float);                  /* batch_qr_by_tier */
    total += pc * q_rank * sizeof(float);                  /* batch_qr_norm_by_tier */
    total += pc * q_dim * sizeof(float);                   /* batch_q_by_tier */
    total += pc * (uint64_t)DS4_N_HEAD_DIM * sizeof(float);/* batch_kv_raw_by_tier */
    total += pc * (uint64_t)DS4_N_HEAD_DIM * sizeof(float);/* batch_kv_by_tier */
    total += pc * comp_width_max * sizeof(float);          /* batch_comp_kv_by_tier */
    total += pc * comp_width_max * sizeof(float);          /* batch_comp_sc_by_tier */
    total += pc * indexer_q_dim * sizeof(float);           /* batch_indexer_q_by_tier */
    total += pc * (uint64_t)DS4_N_INDEXER_HEAD * sizeof(float); /* batch_indexer_weights */
    total += pc * q_dim * sizeof(float);                   /* batch_heads_by_tier */
    total += pc * low_dim * sizeof(float);                 /* batch_attn_low_by_tier */
    total += pc * (uint64_t)DS4_N_EMBD * sizeof(float);    /* batch_attn_out_by_tier */
    total += pc * group_dim * sizeof(float);               /* batch_group_tmp_by_tier */
    total += pc * (uint64_t)DS4_N_LORA_O * sizeof(float);  /* batch_low_tmp_by_tier */
    total += pc * hc_dim * sizeof(float);                  /* batch_after_attn_hc_by_tier */
    total += pc * (uint64_t)DS4_N_EMBD * sizeof(float);    /* batch_ffn_cur_by_tier */
    total += pc * (uint64_t)DS4_N_EMBD * sizeof(float);    /* batch_ffn_norm_by_tier */
    total += pc * shared_dim * sizeof(float);              /* batch_shared_gate_by_tier */
    total += pc * shared_dim * sizeof(float);              /* batch_shared_up_by_tier */
    total += pc * shared_dim * sizeof(float);              /* batch_shared_mid_by_tier */
    total += pc * (uint64_t)DS4_N_EMBD * sizeof(float);    /* batch_shared_out_by_tier */
    total += pc * (uint64_t)DS4_N_EXPERT * sizeof(float);  /* batch_router_logits_by_tier */
    total += pc * (uint64_t)DS4_N_EXPERT * sizeof(float);  /* batch_router_probs_by_tier */
    total += pc * (uint64_t)DS4_N_EXPERT_USED * sizeof(int);   /* batch_router_selected */
    total += pc * (uint64_t)DS4_N_EXPERT_USED * sizeof(float); /* batch_router_weights */
    total += pc * (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float); /* batch_routed_gate */
    total += pc * (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float); /* batch_routed_up */
    total += pc * (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float); /* batch_routed_mid */
    total += pc * (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float);     /* batch_routed_down */
    total += pc * (uint64_t)DS4_N_EMBD * sizeof(float);    /* batch_routed_out_by_tier */
    total += pc * (uint64_t)DS4_N_EMBD * sizeof(float);    /* batch_ffn_out_by_tier */

    /* === Class E embedding-tier prefill_tokens (mirrors ds4.c:10844).
     * Charged to ALL tiers conservatively. Negligible (pc * int32). === */
    total += pc * sizeof(int32_t);                         /* prefill_tokens_by_tier */

    /* === Head-tier-only extras (mirrors ds4.c:10806-10816). Charged
     * conservatively to EVERY tier. === */
    total += (uint64_t)DS4_N_HC   * sizeof(float);         /* output_pre_by_tier */
    total += (uint64_t)DS4_N_HC   * sizeof(float);         /* output_weights_by_tier */
    total += (uint64_t)DS4_N_EMBD * sizeof(float);         /* output_embd_by_tier */
    total += (uint64_t)DS4_N_EMBD * sizeof(float);         /* output_norm_by_tier */
    /* Output-TP sessions may load DSpark after placement, so reserve its
     * block-verifier shard scratch conservatively even before support-model
     * detection. Plain decode allocates only the one-row subset. */
    total += output_logits_elems * sizeof(float);          /* logits_by_tier */

    return total;
}

