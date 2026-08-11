typedef struct {
    uint32_t ctx_size;
    uint32_t ctx_cap;
    uint32_t normal_layers;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t layer_count;
    uint64_t q_dim;
    uint64_t q_nope;
    uint64_t heads_dim;
    uint64_t kv_raw_dim;
    uint64_t dense_hidden_max;
    uint64_t ffn_mid_elems;

    ds4_gpu_tensor *cur;
    ds4_gpu_tensor *next;
    ds4_gpu_tensor *attn_norm;
    ds4_gpu_tensor *q_rank;
    ds4_gpu_tensor *q_rank_norm;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *kv_raw;
    ds4_gpu_tensor *kv_norm;
    ds4_gpu_tensor *k_nope;
    ds4_gpu_tensor *value;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *after_attn;
    ds4_gpu_tensor *ffn_norm;
    ds4_gpu_tensor *ffn_gate;
    ds4_gpu_tensor *ffn_up;
    ds4_gpu_tensor *ffn_mid;
    ds4_gpu_tensor *routed_gate;
    ds4_gpu_tensor *routed_up;
    ds4_gpu_tensor *routed_down;
    ds4_gpu_tensor *ffn_out;
    ds4_gpu_tensor *ffn_sum;
    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *router_probs;
    ds4_gpu_tensor *router_selected;
    ds4_gpu_tensor *router_weights;
    ds4_gpu_tensor *output_norm;
    ds4_gpu_tensor *logits;
    ds4_gpu_tensor *batch_router_logits;
    ds4_gpu_tensor *batch_router_probs;
    ds4_gpu_tensor *batch_router_selected;
    ds4_gpu_tensor *batch_router_weights;
    ds4_gpu_tensor *prefill_seed_router_selected;
    uint32_t prefill_seed_tokens;
    bool prefill_seed_layer_captured[DS4_MAX_LAYER];

    ds4_gpu_tensor *prefill_tokens;
    ds4_gpu_tensor *batch_cur;
    ds4_gpu_tensor *batch_next;
    ds4_gpu_tensor *batch_attn_norm;
    ds4_gpu_tensor *batch_q_rank;
    ds4_gpu_tensor *batch_q_rank_norm;
    ds4_gpu_tensor *batch_q;
    ds4_gpu_tensor *batch_kv_raw;
    ds4_gpu_tensor *batch_kv_norm;
    ds4_gpu_tensor *batch_k_nope;
    ds4_gpu_tensor *batch_value;
    ds4_gpu_tensor *batch_heads;
    ds4_gpu_tensor *batch_attn_out;
    ds4_gpu_tensor *batch_after_attn;
    ds4_gpu_tensor *batch_ffn_norm;
    ds4_gpu_tensor *batch_ffn_gate;
    ds4_gpu_tensor *batch_ffn_up;
    ds4_gpu_tensor *batch_shared_mid;
    ds4_gpu_tensor *batch_ffn_mid;
    ds4_gpu_tensor *batch_routed_gate;
    ds4_gpu_tensor *batch_routed_up;
    ds4_gpu_tensor *batch_routed_down;
    ds4_gpu_tensor *batch_ffn_out;
    bool batch_routed_mid_is_f16;

    uint32_t compact_cache_cap;
    uint32_t indexed_prefill_cap;
    uint32_t indexed_prefill_score_cap;
    uint32_t indexer_full_layers;
    ds4_gpu_tensor *indexer_k;
    ds4_gpu_tensor *indexer_q;
    ds4_gpu_tensor *indexer_weights;
    ds4_gpu_tensor *indexer_scores;
    ds4_gpu_tensor *indexer_selected;
    ds4_gpu_tensor *qk_low;
    ds4_gpu_tensor *attn_partial_lora;
    ds4_gpu_tensor *attn_partial_ms;
    ds4_gpu_tensor *batch_indexer_k;
    ds4_gpu_tensor *batch_indexer_q;
    ds4_gpu_tensor *batch_indexer_weights;
    ds4_gpu_tensor *batch_indexer_scores;
    ds4_gpu_tensor *batch_indexer_selected;
    ds4_gpu_tensor *batch_qk_low;
    ds4_gpu_tensor *batch_attn_lora;
    ds4_gpu_tensor *layer_kv_lora_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_k_rope_cache[DS4_MAX_LAYER];
    /* GLM MTP (nextn block) drafting: private compact caches for the nextn
     * layer (slot = absolute position; only [mtp_min_pos..pos] is ever
     * selected) plus small scratch. Allocated lazily on first draft. */
    ds4_gpu_tensor *mtp_kv_lora_cache;
    ds4_gpu_tensor *mtp_k_rope_cache;
    ds4_gpu_tensor *mtp_concat;
    ds4_gpu_tensor *mtp_selected;
    float          *mtp_logits_host;
    int             mtp_ready;
    ds4_gpu_tensor *layer_indexer_key_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_key_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_value_cache[DS4_MAX_LAYER];
    bool full_kv_cache;
    bool has_token_embd;
    bool has_output_head;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool generic_routed_moe;
    bool streaming_static_decode_map_current;
    /* Tensor parallelism (50/50 expert sharding): tp_world 2 means
     * this rank computes only its contiguous half of the routed experts
     * and exchanges the 24KB routed-FFN partial at one gate per sparse
     * layer.  Views alias the engine's TP slab slots [layer*2 + FFN]. */
    uint32_t tp_world;
    uint32_t tp_rank;
    ds4_gpu_tensor **tp_out;
    ds4_gpu_tensor **tp_in;
    /* Prefill batch gate bounce buffers (shared storage; grow on demand). */
    ds4_gpu_tensor *tp_bounce_out;
    ds4_gpu_tensor *tp_bounce_in;
    /* CUDA multi-tier placement and device-local decode scratch mirrors. */
    const int *placement;
#define DS4_GLM_WS_SLOTS 29
    ds4_gpu_tensor *ws_mirror[DS4_MAX_GPUS][DS4_GLM_WS_SLOTS];
    ds4_gpu_tensor *ws_orig[DS4_GLM_WS_SLOTS];
    int ws_ready;
    int ws_tier;
#define DS4_GLM_VERIFY_WS_SLOTS 28
    ds4_gpu_tensor *verify_ws_mirror[DS4_MAX_GPUS][DS4_GLM_VERIFY_WS_SLOTS];
    ds4_gpu_tensor *verify_ws_orig[DS4_GLM_VERIFY_WS_SLOTS];
    int verify_ws_ready;
    int verify_ws_tier;
} ds4_glm_gpu_graph;

static uint32_t glm_graph_model_context_limit(void) {
    if (DS4_ROPE_ORIG_CTX > UINT32_MAX) return UINT32_MAX;
    return (uint32_t)DS4_ROPE_ORIG_CTX;
}

static double glm_graph_bytes_to_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static uint64_t glm_graph_saturating_add_u64(uint64_t a, uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

static bool glm_graph_env_disabled(const char *name) {
    const char *env = getenv(name);
    if (!env || !env[0]) return false;
    return strcmp(env, "0") == 0 ||
           strcasecmp(env, "false") == 0 ||
           strcasecmp(env, "off") == 0 ||
           strcasecmp(env, "no") == 0;
}

static double glm_graph_env_double(
        const char *name,
        double      fallback,
        double      min_value,
        double      max_value) {
    const char *env = getenv(name);
    if (!env || !env[0]) return fallback;
    char *end = NULL;
    errno = 0;
    const double v = strtod(env, &end);
    if (end == env || errno != 0 || !isfinite(v)) return fallback;
    if (v < min_value) return min_value;
    if (v > max_value) return max_value;
    return v;
}

static uint64_t glm_graph_host_memory_bytes(void) {
#if defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) != 0) return 0;
    return mem;
#else
    return 0;
#endif
}

static uint64_t glm_graph_streaming_active_model_bytes(
        const ds4_weights *weights) {
    if (!weights) return 0;

    uint64_t max_bytes = 0;
    ds4_model_map_span_vec spans;

    if (weights_layer_has_required(&weights->layer[0], 0) &&
        weights_model_map_token_spans(weights, &spans)) {
        max_bytes = model_map_span_vec_total_bytes(&spans);
        free(spans.v);
    }
    if (weights_have_output_head(weights) &&
        weights_model_map_output_spans(weights, &spans)) {
        const uint64_t bytes = model_map_span_vec_total_bytes(&spans);
        if (bytes > max_bytes) max_bytes = bytes;
        free(spans.v);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!weights_model_map_spans(weights, il, il, false, &spans)) {
            continue;
        }
        const uint64_t bytes = model_map_span_vec_total_bytes(&spans);
        if (bytes > max_bytes) max_bytes = bytes;
        free(spans.v);
    }

    return max_bytes;
}

/* TP shard bytes: dense weights plus this rank's routed-expert range.
 * Zero when not sharding.  Set during engine open, before the GLM memory
 * guard runs. */
static uint64_t g_tp_shard_model_bytes;

/* A user-raised iogpu.wired_limit_mb is an explicit GPU budget grant;
 * prefer it over the fraction/reserve heuristics. */
static uint64_t glm_graph_wired_limit_bytes(void) {
#ifdef __APPLE__
    int64_t mb = 0;
    size_t len = sizeof(mb);
    if (sysctlbyname("iogpu.wired_limit_mb", &mb, &len, NULL, 0) != 0) return 0;
    if (mb <= 0) return 0;
    return (uint64_t)mb * 1024ull * 1024ull;
#else
    return 0;
#endif
}

static uint64_t glm_graph_model_bytes_for_guard(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        bool               load_slice,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               include_token,
        bool               include_output) {
    if (!model) return 0;
    /* Under TP, the sharded map bytes are authoritative regardless of
     * how the caller frames the request (TP excludes real layer slicing,
     * so any slice request here is the session's full-range accounting). */
    if (!ssd_streaming && g_tp_shard_model_bytes != 0) {
        return g_tp_shard_model_bytes;
    }
    if (load_slice && weights) {
        ds4_model_map_span_vec spans;
        bool ok = false;
        if (ssd_streaming) {
            ok = weights_model_map_decode_static_slice_spans(weights,
                                                             layer_start,
                                                             layer_end,
                                                             include_token,
                                                             include_output,
                                                             &spans);
        } else {
            ok = weights_model_map_spans(weights,
                                         layer_start,
                                         layer_end,
                                         include_output,
                                         &spans);
        }
        if (ok) {
            const uint64_t bytes = model_map_span_vec_total_bytes(&spans);
            free(spans.v);
            if (bytes != 0) return bytes;
        }
    }
    if (!ssd_streaming) {
        if (g_tp_shard_model_bytes != 0) return g_tp_shard_model_bytes;
        return model->size;
    }
    const uint64_t active_bytes = glm_graph_streaming_active_model_bytes(weights);
    return active_bytes != 0 ? active_bytes : model->size;
}

static double glm_graph_memory_guard_default_reserve_gib(
        uint64_t budget_base,
        uint64_t model_bytes) {
    const double base_gib = glm_graph_bytes_to_gib(budget_base);
    const double model_gib = glm_graph_bytes_to_gib(model_bytes);
    if (base_gib >= 480.0 &&
        base_gib <= 640.0 &&
        model_gib >= base_gib * 0.80) {
        return 24.0;
    }
    return 32.0;
}

static bool glm_graph_memory_guard_for_compact_cap(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        bool               load_slice,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               include_token,
        bool               include_output,
        uint32_t           ctx_size,
        uint32_t           compact_cap,
        uint64_t           transient_extra_bytes,
        const char        *phase) {
    if (!model || glm_graph_env_disabled("DS4_GLM_MEMORY_GUARD")) return true;

    const uint64_t host_bytes = glm_graph_host_memory_bytes();
    uint64_t budget_base = host_bytes;
    if (budget_base == 0) {
        budget_base = ds4_gpu_recommended_working_set_size();
    }
    if (budget_base == 0) return true;
    const uint64_t wired_limit = glm_graph_wired_limit_bytes();

    const uint32_t work_ctx =
        glm_graph_full_attention_cap(ctx_size, ssd_streaming);
    const ds4_context_memory mem = load_slice ?
        glm_graph_context_memory_estimate_for_compact_cap_slice(
                ctx_size,
                work_ctx,
                compact_cap,
                ssd_streaming,
                layer_start,
                layer_end) :
        glm_graph_context_memory_estimate_for_compact_cap(
                ctx_size,
                work_ctx,
                compact_cap,
                ssd_streaming);
    const uint64_t graph_bytes = mem.total_bytes;
    const uint64_t model_bytes =
        glm_graph_model_bytes_for_guard(model,
                                        weights,
                                        ssd_streaming,
                                        load_slice,
                                        layer_start,
                                        layer_end,
                                        include_token,
                                        include_output);
    uint64_t required = glm_graph_saturating_add_u64(model_bytes, graph_bytes);
    required = glm_graph_saturating_add_u64(required, transient_extra_bytes);

    const double fraction =
        glm_graph_env_double("DS4_GLM_MEMORY_GUARD_FRACTION", 0.99, 0.50, 1.00);
    double default_reserve_gib =
        glm_graph_memory_guard_default_reserve_gib(budget_base, model_bytes);
#ifdef DS4_ROCM_BUILD
    if (load_slice && !ssd_streaming) {
        /* The original fixed reserve protects Metal's shared host/GPU heap.
         * A resident ROCm layer slice already accounts its exact model spans
         * and owned graph state above. Keep proportional backend headroom for
         * driver and temporary allocations without rejecting viable UMA
         * slices merely because the heap is smaller than a high-memory Mac. */
        double rocm_reserve_gib = glm_graph_bytes_to_gib(budget_base) / 16.0;
        if (rocm_reserve_gib < 8.0) rocm_reserve_gib = 8.0;
        if (rocm_reserve_gib < default_reserve_gib) {
            default_reserve_gib = rocm_reserve_gib;
        }
    }
#endif
    const double reserve_gib =
        glm_graph_env_double("DS4_GLM_MEMORY_GUARD_RESERVE_GB",
                             default_reserve_gib,
                             0.0,
                             1024.0);
    const uint64_t fraction_budget = (uint64_t)((double)budget_base * fraction);
    const uint64_t reserve_bytes =
        (uint64_t)(reserve_gib * 1024.0 * 1024.0 * 1024.0);
    const uint64_t reserve_budget =
        reserve_bytes >= budget_base ? 0 : budget_base - reserve_bytes;
    uint64_t budget = fraction_budget;
    if (reserve_bytes != 0 && reserve_budget < budget) budget = reserve_budget;
    if (wired_limit != 0) {
        /* An explicitly raised iogpu.wired_limit_mb is the user granting
         * the GPU that much wired memory; it overrides the heuristics
         * (keep a small margin for non-model GPU allocations). */
        const uint64_t margin = 2ull * 1024ull * 1024ull * 1024ull;
        const uint64_t wired_budget =
            wired_limit > margin ? wired_limit - margin : wired_limit;
        if (wired_budget > budget) budget = wired_budget;
    }

    if (required <= budget) {
        const char *report = getenv("DS4_GLM_MEMORY_GUARD_REPORT");
        if (report && report[0]) {
            fprintf(stderr,
                    "ds4: GLM memory guard ctx=%u compact_cap=%u required=%.2f GiB "
                    "budget=%.2f GiB (model %.2f GiB, graph %.2f GiB, transient %.2f GiB)\n",
                    ctx_size,
                    mem.comp_cap,
                    glm_graph_bytes_to_gib(required),
                    glm_graph_bytes_to_gib(budget),
                    glm_graph_bytes_to_gib(model_bytes),
                    glm_graph_bytes_to_gib(graph_bytes),
                    glm_graph_bytes_to_gib(transient_extra_bytes));
            if (ssd_streaming && model_bytes != model->size) {
                fprintf(stderr,
                        "ds4:   GLM streaming guard uses active model span %.2f GiB "
                        "(full GGUF %.2f GiB)\n",
                        glm_graph_bytes_to_gib(model_bytes),
                        glm_graph_bytes_to_gib(model->size));
            } else if (load_slice && model_bytes != model->size) {
                fprintf(stderr,
                        "ds4:   GLM memory guard uses sliced model span %.2f GiB "
                        "(full GGUF %.2f GiB)\n",
                        glm_graph_bytes_to_gib(model_bytes),
                        glm_graph_bytes_to_gib(model->size));
            }
        }
        return true;
    }

    fprintf(stderr,
            "ds4: GLM memory guard refused ctx=%u compact_cap=%u %s\n",
            ctx_size,
            mem.comp_cap,
            phase ? phase : "before Metal graph allocation");
    if (ssd_streaming && model_bytes != model->size) {
        fprintf(stderr,
                "ds4:   streamed active model map: %.2f GiB "
                "(full GGUF %.2f GiB)\n",
                glm_graph_bytes_to_gib(model_bytes),
                glm_graph_bytes_to_gib(model->size));
    } else if (load_slice && model_bytes != model->size) {
        fprintf(stderr,
                "ds4:   sliced model map: %.2f GiB "
                "(full GGUF %.2f GiB)\n",
                glm_graph_bytes_to_gib(model_bytes),
                glm_graph_bytes_to_gib(model->size));
    } else {
        fprintf(stderr,
                "ds4:   model map: %.2f GiB\n",
                glm_graph_bytes_to_gib(model_bytes));
    }
    fprintf(stderr,
            "ds4:   graph cache/scratch: %.2f GiB "
            "(full KV %.2f GiB, compact DSA %.2f GiB, scratch %.2f GiB)\n",
            glm_graph_bytes_to_gib(graph_bytes),
            glm_graph_bytes_to_gib(mem.raw_bytes),
            glm_graph_bytes_to_gib(mem.compressed_bytes),
            glm_graph_bytes_to_gib(mem.scratch_bytes));
    fprintf(stderr,
            "ds4:   required model+graph: %.2f GiB; guard budget: %.2f GiB "
            "(base %.2f GiB, fraction %.2f, reserve %.2f GiB, transient %.2f GiB)\n",
            glm_graph_bytes_to_gib(required),
            glm_graph_bytes_to_gib(budget),
            glm_graph_bytes_to_gib(budget_base),
            fraction,
            reserve_gib,
            glm_graph_bytes_to_gib(transient_extra_bytes));
    fprintf(stderr,
            "ds4:   set DS4_GLM_MEMORY_GUARD=0 to bypass, use a smaller --ctx, "
            "or use SSD streaming\n");
    return false;
}

static bool glm_graph_memory_guard(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        uint32_t           ctx_size) {
    const uint32_t work_ctx =
        glm_graph_full_attention_cap(ctx_size, ssd_streaming);
    const uint32_t compact_cap =
        glm_graph_compact_cache_initial_cap(ctx_size, work_ctx);
    return glm_graph_memory_guard_for_compact_cap(
            model,
            weights,
            ssd_streaming,
            false,
            0,
            0,
            true,
            true,
            ctx_size,
            compact_cap,
            0,
            "before GLM graph allocation");
}

static bool glm_graph_memory_guard_with_transient(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        uint32_t           ctx_size,
        uint64_t           transient_extra_bytes,
        const char        *phase) {
    const uint32_t work_ctx =
        glm_graph_full_attention_cap(ctx_size, ssd_streaming);
    const uint32_t compact_cap =
        glm_graph_compact_cache_initial_cap(ctx_size, work_ctx);
    return glm_graph_memory_guard_for_compact_cap(
            model,
            weights,
            ssd_streaming,
            false,
            0,
            0,
            true,
            true,
            ctx_size,
            compact_cap,
            transient_extra_bytes,
            phase);
}

static bool glm_graph_memory_guard_slice(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               include_token,
        bool               include_output,
        uint32_t           ctx_size) {
    const uint32_t work_ctx =
        glm_graph_full_attention_cap(ctx_size, ssd_streaming);
    const uint32_t compact_cap =
        glm_graph_compact_cache_initial_cap(ctx_size, work_ctx);
    return glm_graph_memory_guard_for_compact_cap(
            model,
            weights,
            ssd_streaming,
            true,
            layer_start,
            layer_end,
            include_token,
            include_output,
            ctx_size,
            compact_cap,
            0,
            "before GLM graph allocation");
}

static bool glm_graph_memory_guard_slice_with_transient(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               include_token,
        bool               include_output,
        uint32_t           ctx_size,
        uint64_t           transient_extra_bytes,
        const char        *phase) {
    const uint32_t work_ctx =
        glm_graph_full_attention_cap(ctx_size, ssd_streaming);
    const uint32_t compact_cap =
        glm_graph_compact_cache_initial_cap(ctx_size, work_ctx);
    return glm_graph_memory_guard_for_compact_cap(
            model,
            weights,
            ssd_streaming,
            true,
            layer_start,
            layer_end,
            include_token,
            include_output,
            ctx_size,
            compact_cap,
            transient_extra_bytes,
            phase);
}

static uint32_t glm_graph_full_attention_cap(uint32_t ctx_size,
                                             bool     ssd_streaming) {
    uint32_t cap = ssd_streaming ?
        DS4_GLM_METAL_STREAMING_FULL_ATTN_CONTEXT :
        DS4_GLM_METAL_FULL_ATTN_DEFAULT_CONTEXT;
    if (ctx_size >= DS4_GLM_METAL_LONG_CONTEXT_THRESHOLD &&
        cap > DS4_GLM_METAL_LONG_CONTEXT_FULL_ATTN_CONTEXT) {
        cap = DS4_GLM_METAL_LONG_CONTEXT_FULL_ATTN_CONTEXT;
    }
    if (ctx_size > 0 && cap > ctx_size) cap = ctx_size;
    if (cap == 0) cap = 1;
    return cap;
}

static uint32_t glm_graph_full_prefill_layer_flush_interval(
        uint32_t n_tokens,
        uint32_t command_rows,
        bool     logits_requested) {
    /* Tiny logits-bearing passes (MTP verify, short prefills) must NOT
     * flush per layer: 76 command-buffer round-trips cost ~35ms while the
     * whole pass is ~70ms of GPU work. Real prefill chunks keep the
     * interactive per-layer flush. */
    return (n_tokens > DS4_GLM_METAL_FULL_ATTN_LAYER_FLUSH_CONTEXT ||
            command_rows > DS4_GLM_METAL_FULL_ATTN_LAYER_FLUSH_CONTEXT ||
            (logits_requested && n_tokens > 8u)) ? 1u : 0u;
}

static uint32_t glm_graph_prefill_progress_flush_interval(
        uint32_t                layer_flush_interval,
        uint32_t                n_tokens,
        ds4_session_progress_fn display_progress,
        uint32_t                work_total) {
    if (layer_flush_interval != 0) return layer_flush_interval;
    (void)n_tokens;
    (void)display_progress;
    (void)work_total;
    return 0;
}

static void glm_graph_report_prefill_display_progress(
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                absolute_base,
        uint32_t                work_done_base,
        uint32_t                n_tokens,
        uint32_t                layer_done,
        uint32_t                normal_layers,
        uint32_t                work_total,
        bool                    allow_complete) {
    if (!display_progress || work_total == 0) return;

    uint64_t chunk_done = 0;
    if (normal_layers == 0 || layer_done >= normal_layers) {
        chunk_done = n_tokens;
    } else {
        chunk_done = (uint64_t)n_tokens * (uint64_t)layer_done /
                     (uint64_t)normal_layers;
    }

    uint64_t done = (uint64_t)work_done_base + chunk_done;
    if (done > (uint64_t)work_total) done = work_total;
    if (!allow_complete && done >= (uint64_t)work_total) {
        done = work_total > 0 ? (uint64_t)work_total - 1u : 0u;
    }
    display_progress(display_progress_ud,
                     "prefill_display",
                     (int)((uint64_t)absolute_base + done),
                     (int)((uint64_t)absolute_base + (uint64_t)work_total));
}

static bool glm_graph_small_prefill_stage_sync(
        uint32_t n_tokens,
        bool     logits_requested) {
    return logits_requested &&
           n_tokens > 0 &&
           n_tokens <= DS4_GLM_METAL_SMALL_PREFILL_STAGE_SYNC_TOKENS;
}

static uint32_t glm_graph_indexed_decode_split_min_block_rows(void) {
    return 32u;
}

static uint32_t glm_graph_indexed_decode_split_blocks(void) {
    const uint32_t block_rows = glm_graph_indexed_decode_split_min_block_rows();
    const uint32_t top_k = glm_graph_indexer_top_k_limit();
    return (top_k + block_rows - 1u) / block_rows;
}

static uint32_t glm_graph_indexed_decode_split_block_rows_for(uint32_t n_selected) {
    return n_selected <= 1024u ? 32u : 128u;
}

static bool glm_graph_indexed_decode_split_group8_available(uint32_t n_selected) {
    const uint32_t block_rows = glm_graph_indexed_decode_split_block_rows_for(n_selected);
    const uint32_t needed_blocks =
        block_rows != 0u ? (n_selected + block_rows - 1u) / block_rows : 0u;
    return n_selected > 512u &&
           block_rows > 0 &&
           needed_blocks > 0 &&
           needed_blocks <= glm_graph_indexed_decode_split_blocks() &&
           glm_graph_indexed_decode_split_blocks() <= 64u &&
           (DS4_N_HEAD % 8u) == 0 &&
           DS4_N_KV_LORA == 512u &&
           DS4_N_ROT == 64u &&
           glm_graph_compact_cache_is_f16();
}

static bool glm_graph_prefill_stage_sync_boundary(void) {
    if (ds4_gpu_end_commands() == 0) return false;
    return ds4_gpu_begin_commands() != 0;
}

static bool glm_graph_indexed_prefill_attention_boundary(void) {
#ifdef DS4_ROCM_BUILD
    /*
     * ROCm launches in this path are ordered on the default stream. The Metal
     * backend still needs the encoder flush, but on ROCm it is a full-device
     * synchronize and stalls every indexed-prefill layer.
     */
    return true;
#else
    return ds4_gpu_flush_encoder() != 0;
#endif
}

static DS4_MAYBE_UNUSED bool glm_graph_env_truthy(const char *env) {
    return env &&
           env[0] &&
           strcmp(env, "0") != 0 &&
           strcasecmp(env, "false") != 0 &&
           strcasecmp(env, "off") != 0 &&
           strcasecmp(env, "no") != 0;
}

static bool glm_graph_streaming_prefill_sync_each_layer(
        bool full_layer_prefill) {
#ifdef DS4_ROCM_BUILD
    /*
     * ROCm command boundaries are full device synchronizes. Compact streaming
     * prefill can keep queued default-stream work alive across layer mappings:
     * streamed model-range eviction synchronizes before freeing ranges, while
     * selected-expert cache reuse/eviction is protected by reuse events. The
     * full-layer expert cache is only double-buffered, so keep its old boundary.
     */
    if (full_layer_prefill) return true;
    const char *env = glm_graph_env_value(
            "DS4_ROCM_GLM_STREAMING_PREFILL_SYNC_EACH_LAYER",
            "DS4_METAL_GLM_STREAMING_PREFILL_SYNC_EACH_LAYER");
    if (!env) env = getenv("DS4_GLM_STREAMING_PREFILL_SYNC_EACH_LAYER");
    return glm_graph_env_truthy(env);
#else
    (void)full_layer_prefill;
    return true;
#endif
}

static bool glm_graph_indexed_prefill_batch_available(
        const ds4_glm_gpu_graph *g) {
    return g &&
           g->compact_cache_cap != 0 &&
           g->indexed_prefill_cap != 0 &&
           g->indexed_prefill_score_cap != 0 &&
           g->batch_indexer_q &&
           g->batch_indexer_weights &&
           g->batch_indexer_scores &&
           g->batch_indexer_selected &&
           g->batch_qk_low &&
           g->batch_attn_lora;
}

static bool glm_graph_indexed_prefill_batch_ready(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos) {
    return glm_graph_indexed_prefill_batch_available(g) &&
           (!g->full_kv_cache || pos >= g->ctx_cap);
}

static uint32_t glm_graph_limit_indexed_prefill_chunk(
        uint32_t pos,
        uint32_t chunk) {
    const uint32_t top_k = glm_graph_indexer_top_k_limit();
    if (pos < top_k) {
        const uint32_t bridge = top_k - pos;
        if (bridge != 0 && chunk > bridge) chunk = bridge;
    }
    return chunk;
}

static uint32_t glm_graph_indexed_prefill_chunk_tokens(
        uint32_t full_attention_cap,
        uint32_t compact_cap) {
    (void)full_attention_cap;
    uint32_t chunk = DS4_GLM_METAL_INDEXED_PREFILL_CHUNK_TOKENS;
    if (compact_cap > 0 && chunk > compact_cap) chunk = compact_cap;
    if (chunk == 0) chunk = 1;
    return chunk;
}

static uint32_t glm_graph_indexed_prefill_score_tokens(
        uint32_t indexed_prefill_cap,
        uint32_t compact_cap) {
    if (indexed_prefill_cap == 0 || compact_cap == 0) return 0;
    const uint32_t scratch_mb = DS4_GLM_METAL_INDEXED_PREFILL_SCORE_SCRATCH_MB;
    const uint64_t budget_bytes = (uint64_t)scratch_mb * 1024ull * 1024ull;
    uint64_t budget_rows = budget_bytes / ((uint64_t)compact_cap * sizeof(float));
    if (budget_rows == 0) budget_rows = 1;
    if (budget_rows > indexed_prefill_cap) budget_rows = indexed_prefill_cap;
    if (budget_rows > UINT32_MAX) budget_rows = UINT32_MAX;
    return (uint32_t)budget_rows;
}

static bool glm_graph_context_request(int ctx_size, uint32_t *ctx_out) {
    if (!ctx_out || ctx_size <= 0) return false;
    const uint32_t model_ctx = glm_graph_model_context_limit();
    if ((uint64_t)(uint32_t)ctx_size > (uint64_t)model_ctx) {
        fprintf(stderr,
                "ds4: GLM context %d exceeds model context %u\n",
                ctx_size,
                model_ctx);
        return false;
    }
    *ctx_out = (uint32_t)ctx_size;
    return true;
}

static bool glm_graph_span_fits_context(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    return g && n_tokens > 0 && pos0 < g->ctx_size && n_tokens <= g->ctx_size - pos0;
}

static bool glm_graph_span_fits_full_attention(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    return g && n_tokens > 0 && pos0 < g->ctx_cap && n_tokens <= g->ctx_cap - pos0;
}

static void glm_graph_log_full_attention_limit(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    const uint32_t end = pos0 + n_tokens;
    fprintf(stderr,
            "ds4: GLM Metal full-attention work cap is %u tokens; "
            "requested span [%u,%u) in ctx %u needs compact indexed attention\n",
            g ? g->ctx_cap : 0,
            pos0,
            end,
            g ? g->ctx_size : 0);
}

static bool glm_graph_tensor_layout(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          dim0,
        uint64_t          dim1,
        uint64_t          dim2) {
    if (!t || t->type != type || t->ndim != ndim) return false;
    if (ndim > 0 && t->dim[0] != dim0) return false;
    if (ndim > 1 && t->dim[1] != dim1) return false;
    if (ndim > 2 && t->dim[2] != dim2) return false;
    return true;
}

static bool glm_graph_dense_tensor_layout(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          dim0,
        uint64_t          dim1,
        uint64_t          dim2) {
    if (!t || !tensor_type_is_glm_dense_quant(t->type) || t->ndim != ndim) return false;
    if (ndim > 0 && t->dim[0] != dim0) return false;
    if (ndim > 1 && t->dim[1] != dim1) return false;
    if (ndim > 2 && t->dim[2] != dim2) return false;
    return true;
}

static bool glm_graph_layer_uses_generic_routed_moe(
        const ds4_layer_weights *l) {
    return l &&
           l->ffn_gate_exps &&
           l->ffn_up_exps &&
           l->ffn_down_exps &&
           l->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS;
}

static bool glm_tp_validate_ownership_kernels(
        const ds4_weights *weights,
        uint32_t          *bad_layer,
        uint32_t          *bad_type) {
    if (!weights) return false;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &weights->layer[il];
        if (!l->ffn_gate_exps) continue;
        if (glm_graph_layer_uses_generic_routed_moe(l) ||
            l->ffn_gate_exps->type == DS4_TENSOR_Q2_K) {
            continue;
        }
        if (bad_layer) *bad_layer = il;
        if (bad_type) *bad_type = l->ffn_gate_exps->type;
        return false;
    }
    return true;
}

static bool glm_graph_stream_map_token(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    if (!g || !g->ssd_streaming) return true;
    g->streaming_static_decode_map_current = false;
    return metal_graph_stream_map_token(model, weights);
}

static bool glm_graph_stream_layer_expert_cache_supported(
        const ds4_weights       *weights,
        const ds4_layer_weights *l,
        uint32_t                 il) {
    if (!weights || !l) return false;
    if (il < DS4_N_LEADING_DENSE) return true;
    return glm_stream_decode_experts_are_streamed(weights, l, il);
}

static bool glm_graph_stream_prefill_expert_addr_supported(
        const ds4_weights       *weights,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 n_tokens) {
    if (il < DS4_N_LEADING_DENSE) return true;
    if (n_tokens <= 1) return false;
#ifdef DS4_ROCM_BUILD
    /*
     * ROCm selected-address batch prefill has pointer kernels for the
     * IQ2-gate/Q2-down generic path and the uniform Q2_K GLM path. Q4_K still
     * maps the full layer until matching pointer kernels exist.
     */
    if (glm_stream_selected_expert_cache_supported(l, il)) return true;
    return l &&
           l->ffn_gate_exps &&
           l->ffn_up_exps &&
           l->ffn_down_exps &&
           l->ffn_gate_exps->type == DS4_TENSOR_Q2_K &&
           l->ffn_up_exps->type == DS4_TENSOR_Q2_K &&
           l->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
           glm_stream_expert_cache_addr_layout_supported(weights, l, il);
#else
    return glm_stream_expert_cache_addr_supported(weights, l, il);
#endif
}

static bool rocm_graph_glm_stream_prefill_full_layer_enabled(
        const ds4_glm_gpu_graph *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 n_tokens);

static bool glm_graph_stream_map_decode_layer(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il) {
    if (!g || !g->ssd_streaming) return true;
    g->streaming_static_decode_map_current = false;
    if (glm_graph_env_present("DS4_ROCM_GLM_STREAMING_DECODE_FULL_LAYER_MAP",
                              "DS4_METAL_GLM_STREAMING_DECODE_FULL_LAYER_MAP") ||
        getenv("DS4_GLM_STREAMING_DECODE_FULL_LAYER_MAP") != NULL) {
        return metal_graph_stream_map_layer(model, weights, il);
    }
    if (weights && il < DS4_N_LAYER &&
        glm_stream_resident_decode_layer_enabled(&weights->layer[il], il)) {
        return metal_graph_stream_map_layer(model, weights, il);
    }
    if (weights && il < DS4_N_LAYER &&
        glm_graph_stream_layer_expert_cache_supported(weights,
                                                      &weights->layer[il],
                                                      il)) {
        return metal_graph_stream_map_layer_decode(model, weights, il);
    }
    return metal_graph_stream_map_layer(model, weights, il);
}

static bool glm_graph_stream_map_prefill_layer(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il,
        uint32_t           n_tokens,
        bool               full_layer_prefill) {
    if (!g || !g->ssd_streaming) return true;
    g->streaming_static_decode_map_current = false;
    if (full_layer_prefill) {
        const char *trace = glm_graph_env_value("DS4_ROCM_STREAMING_MAP_TRACE",
                                                "DS4_METAL_STREAMING_MAP_TRACE");
        if (trace && trace[0] && strcmp(trace, "0") != 0) {
            fprintf(stderr,
                    "ds4: GLM SSD prefill map layer=%u tokens=%u mode=full-prefill\n",
                    il,
                    n_tokens);
        }
#ifdef DS4_ROCM_BUILD
        if (weights &&
            il < DS4_N_LAYER &&
            rocm_graph_glm_stream_prefill_full_layer_enabled(g,
                                                             &weights->layer[il],
                                                             il,
                                                             n_tokens)) {
            return metal_graph_stream_map_layer_decode(model, weights, il);
        }
#endif
        return metal_graph_stream_map_layer(model, weights, il);
    }
    const bool addr_supported =
        weights && il < DS4_N_LAYER &&
        glm_graph_stream_prefill_expert_addr_supported(weights,
                                                       &weights->layer[il],
                                                       il,
                                                       n_tokens);
    const char *trace = glm_graph_env_value("DS4_ROCM_STREAMING_MAP_TRACE",
                                            "DS4_METAL_STREAMING_MAP_TRACE");
    if (trace && trace[0] && strcmp(trace, "0") != 0) {
        fprintf(stderr,
                "ds4: GLM SSD prefill map layer=%u tokens=%u mode=%s\n",
                il,
                n_tokens,
                addr_supported ? "decode-expert-cache" : "full-layer");
    }
    if (addr_supported) {
        return metal_graph_stream_map_layer_decode(model, weights, il);
    }
    return metal_graph_stream_map_layer(model, weights, il);
}

#ifdef DS4_ROCM_BUILD
enum { DS4_GLM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS = 1024 };
#else
enum { DS4_GLM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS = 64 };
#endif

static uint32_t glm_graph_stream_prefill_full_layer_min_tokens(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_GLM_STREAMING_PREFILL_FULL_LAYER_MIN_TOKENS",
            "DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER_MIN_TOKENS");
    if (!env) return DS4_GLM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS;
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || errno != 0 || v == 0 || v > UINT32_MAX) {
        return DS4_GLM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS;
    }
    return (uint32_t)v;
}

static bool glm_graph_stream_prefill_full_layer_enabled(
        const ds4_glm_gpu_graph *g,
        uint32_t                 n_tokens) {
    if (!g || !g->ssd_streaming) return false;
    if (glm_graph_env_present(
                "DS4_ROCM_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER",
                "DS4_METAL_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER")) {
        return false;
    }
    if (glm_graph_env_present("DS4_ROCM_GLM_STREAMING_PREFILL_FULL_LAYER",
                              "DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER")) {
        return true;
    }
    return n_tokens >= glm_graph_stream_prefill_full_layer_min_tokens();
}

static bool glm_graph_stream_prefill_full_layer_prepare_enabled(
        const ds4_glm_gpu_graph *g,
        bool                     full_layer_prefill) {
    return g &&
           g->ssd_streaming &&
           full_layer_prefill &&
           !glm_graph_env_present(
                "DS4_ROCM_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER_PREPARE",
                "DS4_METAL_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER_PREPARE");
}

#ifdef DS4_ROCM_BUILD
static bool rocm_graph_glm_stream_prefill_full_layer_enabled(
        const ds4_glm_gpu_graph *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 n_tokens) {
    return glm_graph_stream_prefill_full_layer_enabled(g, n_tokens) &&
           layer &&
           glm_stream_resident_decode_layer_supported(layer, il);
}

static bool rocm_graph_glm_stream_layer_expert_load_start_next(
        rocm_graph_stream_layer_expert_load *job,
        const ds4_glm_gpu_graph             *g,
        const ds4_model                     *model,
        const ds4_weights                   *weights,
        uint32_t                             first_il,
        uint32_t                             last_il,
        uint32_t                             n_tokens) {
    if (!job || !model || !weights || first_il > last_il) return true;
    if (job->active) return true;
    for (uint32_t il = first_il; il <= last_il && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!rocm_graph_glm_stream_prefill_full_layer_enabled(g,
                                                              layer,
                                                              il,
                                                              n_tokens)) {
            continue;
        }
        uint64_t gate_expert_bytes = 0;
        uint64_t down_expert_bytes = 0;
        if (!rocm_graph_stream_layer_expert_bytes(layer,
                                                  &gate_expert_bytes,
                                                  &down_expert_bytes)) {
            return false;
        }
        return rocm_graph_stream_layer_expert_load_start(job,
                                                         model,
                                                         layer,
                                                         il,
                                                         gate_expert_bytes,
                                                         down_expert_bytes);
    }
    return true;
}

static bool rocm_graph_glm_stream_layer_expert_load_ready(
        rocm_graph_stream_layer_expert_load *job,
        const ds4_glm_gpu_graph             *g,
        const ds4_model                     *model,
        const ds4_weights                   *weights,
        uint32_t                             il,
        uint32_t                             n_tokens) {
    if (!model || !weights || il >= DS4_N_LAYER) return false;
    const ds4_layer_weights *layer = &weights->layer[il];
    if (!rocm_graph_glm_stream_prefill_full_layer_enabled(g,
                                                          layer,
                                                          il,
                                                          n_tokens)) {
        return true;
    }
    uint64_t gate_expert_bytes = 0;
    uint64_t down_expert_bytes = 0;
    if (!rocm_graph_stream_layer_expert_bytes(layer,
                                              &gate_expert_bytes,
                                              &down_expert_bytes)) {
        return false;
    }
    if (job && job->active) {
        if (job->il != il) {
            fprintf(stderr,
                    "ds4: GLM ROCm streaming full-layer expert load expected "
                    "layer %u but pending job is layer %u\n",
                    il,
                    job->il);
            return false;
        }
        return rocm_graph_stream_layer_expert_load_join(job);
    }
    return rocm_graph_stream_layer_expert_load_sync(model,
                                                    layer,
                                                    il,
                                                    gate_expert_bytes,
                                                    down_expert_bytes);
}
#else
static bool rocm_graph_glm_stream_prefill_full_layer_enabled(
        const ds4_glm_gpu_graph *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 n_tokens) {
    (void)g;
    (void)layer;
    (void)il;
    (void)n_tokens;
    return false;
}
#endif

static bool glm_graph_stream_map_output(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    if (!g || !g->ssd_streaming) return true;
    g->streaming_static_decode_map_current = false;
    return metal_graph_stream_map_output(model, weights);
}

static bool glm_graph_validate_expert_layout(
        const ds4_model  *model,
        const ds4_tensor *gate,
        const ds4_tensor *up,
        const ds4_tensor *down,
        uint64_t         *gate_row_bytes,
        uint64_t         *up_row_bytes,
        uint64_t         *down_row_bytes) {
    if (!gate || !up || !down) return false;
    if (!glm_graph_gate_pair_type_supported(gate->type, up->type) ||
        !glm_graph_down_type_supported(down->type) ||
        gate->ndim != 3 || up->ndim != 3 || down->ndim != 3 ||
        gate->dim[0] != DS4_N_EMBD ||
        gate->dim[1] != DS4_N_FF_EXP ||
        gate->dim[2] != DS4_N_EXPERT ||
        up->dim[0] != DS4_N_EMBD ||
        up->dim[1] != DS4_N_FF_EXP ||
        up->dim[2] != DS4_N_EXPERT ||
        down->dim[0] != DS4_N_FF_EXP ||
        down->dim[1] != DS4_N_EMBD ||
        down->dim[2] != DS4_N_EXPERT) {
        return false;
    }

    uint64_t gate_in = 0, gate_out = 0;
    uint64_t up_in = 0, up_out = 0;
    uint64_t down_in = 0, down_out = 0;
    (void)tensor_expert_bytes(model, gate, 0, &gate_in, &gate_out, gate_row_bytes);
    (void)tensor_expert_bytes(model, up, 0, &up_in, &up_out, up_row_bytes);
    (void)tensor_expert_bytes(model, down, 0, &down_in, &down_out, down_row_bytes);
    return gate_in == DS4_N_EMBD &&
           up_in == DS4_N_EMBD &&
           down_in == DS4_N_FF_EXP &&
           gate_out == DS4_N_FF_EXP &&
           up_out == DS4_N_FF_EXP &&
           down_out == DS4_N_EMBD;
}

static bool glm_graph_validate_layer_layout(
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint64_t                 q_dim,
        uint64_t                 q_nope,
        uint64_t                 heads_dim,
        uint64_t                *kv_raw_dim_out,
        uint64_t                *dense_hidden_max) {
    if (!l) return false;
    const uint64_t kv_raw_dim = l->attn_kv_a_mqa ? l->attn_kv_a_mqa->dim[1] : 0;
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    if (!glm_graph_tensor_layout(l->attn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0) ||
        !glm_graph_dense_tensor_layout(l->attn_q_a, 2, DS4_N_EMBD, DS4_N_LORA_Q, 0) ||
        !glm_graph_tensor_layout(l->attn_q_a_norm, DS4_TENSOR_F32, 1, DS4_N_LORA_Q, 0, 0) ||
        !glm_graph_dense_tensor_layout(l->attn_q_b, 2, DS4_N_LORA_Q, q_dim, 0) ||
        !l->attn_kv_a_mqa ||
        !tensor_type_is_glm_dense_quant(l->attn_kv_a_mqa->type) ||
        l->attn_kv_a_mqa->ndim != 2 ||
        l->attn_kv_a_mqa->dim[0] != DS4_N_EMBD ||
        kv_raw_dim < (uint64_t)DS4_N_KV_LORA + DS4_N_ROT ||
        !glm_graph_tensor_layout(l->attn_kv_a_norm, DS4_TENSOR_F32, 1, DS4_N_KV_LORA, 0, 0) ||
        !glm_graph_dense_tensor_layout(l->attn_k_b, 3, q_nope, DS4_N_KV_LORA, DS4_N_HEAD) ||
        !glm_graph_dense_tensor_layout(l->attn_v_b, 3, DS4_N_KV_LORA, DS4_N_VALUE_MLA, DS4_N_HEAD) ||
        !glm_graph_dense_tensor_layout(l->attn_output, 2, heads_dim, DS4_N_EMBD, 0) ||
        !glm_graph_dense_tensor_layout(l->indexer_attn_q_b, 2, DS4_N_LORA_Q, indexer_q_dim, 0) ||
        !glm_graph_dense_tensor_layout(l->indexer_attn_k, 2, DS4_N_EMBD, DS4_N_INDEXER_HEAD_DIM, 0) ||
        !glm_graph_tensor_layout(l->indexer_k_norm, DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0) ||
        !glm_graph_tensor_layout(l->indexer_k_norm_b, DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0) ||
        !glm_graph_tensor_layout(l->indexer_proj, DS4_TENSOR_F32, 2, DS4_N_EMBD, DS4_N_INDEXER_HEAD, 0) ||
        !glm_graph_tensor_layout(l->ffn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0)) {
        fprintf(stderr, "ds4: GLM Metal graph found unexpected attention layout in layer %u\n", il);
        return false;
    }
    if (kv_raw_dim > *kv_raw_dim_out) *kv_raw_dim_out = kv_raw_dim;

    if (il < DS4_N_LEADING_DENSE) {
        const uint64_t hidden = l->ffn_gate ? l->ffn_gate->dim[1] : 0;
        if (!l->ffn_gate ||
            !glm_graph_dense_tensor_layout(l->ffn_gate, 2, DS4_N_EMBD, hidden, 0) ||
            !glm_graph_dense_tensor_layout(l->ffn_up, 2, DS4_N_EMBD, hidden, 0) ||
            !glm_graph_dense_tensor_layout(l->ffn_down, 2, hidden, DS4_N_EMBD, 0)) {
            fprintf(stderr, "ds4: GLM Metal graph found unexpected dense FFN layout in layer %u\n", il);
            return false;
        }
        if (hidden > *dense_hidden_max) *dense_hidden_max = hidden;
    } else {
        uint64_t gate_row_bytes = 0, up_row_bytes = 0, down_row_bytes = 0;
        if (!glm_graph_tensor_layout(l->ffn_gate_inp, DS4_TENSOR_F32, 2, DS4_N_EMBD, DS4_N_EXPERT, 0) ||
            !glm_graph_tensor_layout(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0) ||
            !glm_graph_validate_expert_layout(model,
                                              l->ffn_gate_exps,
                                              l->ffn_up_exps,
                                              l->ffn_down_exps,
                                              &gate_row_bytes,
                                              &up_row_bytes,
                                              &down_row_bytes) ||
            !glm_graph_dense_tensor_layout(l->ffn_gate_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0) ||
            !glm_graph_dense_tensor_layout(l->ffn_up_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0) ||
            !glm_graph_dense_tensor_layout(l->ffn_down_shexp, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0)) {
            fprintf(stderr, "ds4: GLM Metal graph found unexpected sparse FFN layout in layer %u\n", il);
            return false;
        }
        (void)gate_row_bytes;
        (void)up_row_bytes;
        (void)down_row_bytes;
    }
    return true;
}

static bool glm_graph_validate_layout(
        const ds4_model   *model,
        const ds4_weights *weights,
        ds4_glm_gpu_graph *g,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               require_token_embd,
        bool               require_output_head) {
    if (!model || !weights || !g) return false;
    const uint32_t normal_layers = glm_graph_normal_layer_count();
    if (normal_layers == 0 || DS4_N_ROT >= DS4_N_KEY_MLA) {
        fprintf(stderr, "ds4: GLM Metal graph found unsupported layer/key dimensions\n");
        return false;
    }
    if (layer_end == UINT32_MAX) layer_end = normal_layers - 1u;
    if (layer_start > layer_end || layer_end >= normal_layers) {
        fprintf(stderr,
                "ds4: GLM Metal graph found invalid layer slice %u:%u for %u normal layers\n",
                layer_start,
                layer_end,
                normal_layers);
        return false;
    }

    g->has_token_embd = weights->token_embd != NULL;
    g->has_output_head = weights_have_output_head(weights);
    if (require_token_embd && !g->has_token_embd) {
        fprintf(stderr, "ds4: GLM Metal graph layer slice requires token embeddings\n");
        return false;
    }
    if (g->has_token_embd &&
        !glm_graph_dense_tensor_layout(weights->token_embd, 2,
                                       DS4_N_EMBD, DS4_N_VOCAB, 0)) {
        fprintf(stderr, "ds4: GLM Metal graph found unexpected token embedding layout\n");
        return false;
    }
    if (require_output_head && !g->has_output_head) {
        fprintf(stderr, "ds4: GLM Metal graph layer slice requires the output head\n");
        return false;
    }
    if (weights_have_partial_output_head(weights) && !g->has_output_head) {
        fprintf(stderr, "ds4: GLM Metal graph found partial output head\n");
        return false;
    }
    if (g->has_output_head &&
        (!glm_graph_tensor_layout(weights->output_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0) ||
         !glm_graph_dense_tensor_layout(weights->output, 2,
                                        DS4_N_EMBD, DS4_N_VOCAB, 0))) {
        fprintf(stderr, "ds4: GLM Metal graph found unexpected output head layout\n");
        return false;
    }

    g->normal_layers = normal_layers;
    g->layer_start = layer_start;
    g->layer_end = layer_end;
    g->layer_count = layer_end - layer_start + 1u;
    g->q_dim = (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA;
    g->q_nope = (uint64_t)DS4_N_KEY_MLA - DS4_N_ROT;
    g->heads_dim = (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA;
    g->dense_hidden_max = DS4_N_FF_EXP;
    g->kv_raw_dim = 0;
    for (uint32_t il = g->layer_start; il <= g->layer_end; il++) {
        if (!glm_graph_validate_layer_layout(model,
                                             &weights->layer[il],
                                             il,
                                             g->q_dim,
                                             g->q_nope,
                                             g->heads_dim,
                                             &g->kv_raw_dim,
                                             &g->dense_hidden_max)) {
            return false;
        }
        if (glm_graph_layer_uses_generic_routed_moe(&weights->layer[il])) {
            g->generic_routed_moe = true;
        }
    }
    const uint64_t sparse_mid_elems = (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    g->ffn_mid_elems =
        g->dense_hidden_max > sparse_mid_elems ? g->dense_hidden_max : sparse_mid_elems;
    return g->layer_count > 0 &&
           g->kv_raw_dim >= (uint64_t)DS4_N_KV_LORA + DS4_N_ROT;
}

