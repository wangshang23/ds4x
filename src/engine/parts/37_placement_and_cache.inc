/* =========================================================================
 * Wave-2 multi-GPU placement scaffolding: engine placement helpers.
 * =========================================================================
 *
 * These helpers compute and install the multi-GPU layer placement table.
 * They are reached only when ds4_engine_create_with_gpu_config is called
 * with a non-NULL ds4_gpu_config. When the caller passes NULL (every
 * existing caller — ds4_engine_open shim, ds4_test, ds4_cli, ds4_server,
 * ds4_bench, ds4_eval, ds4_agent), these helpers are not invoked and the
 * engine state is byte-equivalent to the pre-multi-GPU CLI main branch.
 */

/* Classify each model tensor by its placement entry.
 *
 * Tensor names live in ds4_str slices (ptr+len), NOT NUL-terminated.
 * We bound every comparison by name.len to avoid out-of-buffer reads. */
static int tensor_to_entry(const ds4_tensor *t, int n_layer) {
    const char *p = t->name.ptr;
    int n = (int)t->name.len;
    if (n <= 0 || !p) return 0;

    /* "blk.<digits>." prefix -> entry il + 1. */
    if (n >= 5 && memcmp(p, "blk.", 4) == 0) {
        int i = 4;
        int il = 0;
        int digits = 0;
        while (i < n && p[i] >= '0' && p[i] <= '9') {
            if (digits >= 4) return 0;
            il = il * 10 + (p[i] - '0');
            digits++;
            i++;
        }
        if (digits > 0 && i < n && p[i] == '.' && il >= 0 && il < n_layer) {
            return il + 1;
        }
        return 0;
    }
    /* Output-head tensors -> head bucket. weights_bind() requires six
     * top-level tensors: token_embd (embedding), and five head-tier
     * tensors output, output_norm, plus three output_hc_* (the
     * heavily-compressed head's base / fn / scale). All five live on the
     * output-head tier; missing any of them here would mis-size the
     * embedding bucket and the head bucket for the packer. */
    static const char k_output[] = "output.weight";
    static const char k_output_norm[] = "output_norm.weight";
    static const char k_output_hc[] = "output_hc_";
    if (n == (int)sizeof(k_output) - 1 && memcmp(p, k_output, n) == 0) {
        return n_layer + 1;
    }
    if (n == (int)sizeof(k_output_norm) - 1 && memcmp(p, k_output_norm, n) == 0) {
        return n_layer + 1;
    }
    if (n >= (int)sizeof(k_output_hc) - 1 &&
        memcmp(p, k_output_hc, sizeof(k_output_hc) - 1) == 0) {
        return n_layer + 1;
    }
    /* "mtp." prefix -> head bucket (no current code path loads MTP into
     * e->model, but harmless). */
    if (n >= 4 && memcmp(p, "mtp.", 4) == 0) return n_layer + 1;

    /* token_embd.weight and everything else falls into entry 0. */
    return 0;
}

static bool engine_deepseek_routed_expert_tensor(
        const ds4_engine *e,
        const ds4_tensor *t,
        int               entry,
        uint64_t         *expert_bytes) {
    if (!e || !t || entry < 1 || entry > (int)DS4_N_LAYER ||
        g_ds4_shape.family != DS4_MODEL_FAMILY_DEEPSEEK4 ||
        DS4_N_EXPERT == 0u || (DS4_N_EXPERT & 1u) != 0u) {
        return false;
    }
    const ds4_layer_weights *layer = &e->weights.layer[entry - 1];
    if (t != layer->ffn_gate_exps &&
        t != layer->ffn_up_exps &&
        t != layer->ffn_down_exps) {
        return false;
    }
    if (t->bytes == 0u || t->bytes % DS4_N_EXPERT != 0u) {
        fprintf(stderr,
                "ds4: routed expert tensor %.*s is not evenly divisible "
                "across %u experts\n",
                (int)t->name.len, t->name.ptr, DS4_N_EXPERT);
        return false;
    }
    if (expert_bytes) *expert_bytes = t->bytes / DS4_N_EXPERT;
    return true;
}

static bool engine_cuda_tp_decode_requested(const ds4_engine *e);
static bool engine_cuda_tp_ep_requested(const ds4_engine *e);
static bool engine_cuda_tp_output_env_requested(void);

/* Compute per-entry byte footprint estimates. Walks the tensor table once
 * and adds each tensor's bytes to its entry bucket. Also adds a per-layer
 * KV/scratch estimate so the packer's budget math reflects runtime
 * requirements (not just weight bytes). Returns 0 on success. */
static int engine_compute_entry_bytes(const ds4_engine *e, size_t *out) {
    const int n_entries = DS4_N_LAYER + 2;
    const bool cuda_tp_ep = engine_cuda_tp_ep_requested(e);
    for (int i = 0; i < n_entries; i++) out[i] = 0;

    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        const ds4_tensor *t = &e->model.tensors[i];
        if (t->bytes == 0) continue;
        int entry = tensor_to_entry(t, DS4_N_LAYER);
        if (entry < 0 || entry >= n_entries) entry = 0;
        if (cuda_tp_ep && engine_cuda_tp_output_env_requested() &&
            t == e->weights.output) {
            /* Output TP stores one vocabulary-row slice per participating
             * tier. Those bytes are reserved per tier after the head tier is
             * known, rather than charging a full output matrix here. */
            continue;
        }
        uint64_t expert_bytes = 0;
        if (cuda_tp_ep &&
            engine_deepseek_routed_expert_tensor(e, t, entry, &expert_bytes)) {
            out[entry] += expert_bytes * (DS4_N_EXPERT / 2u);
        } else {
            out[entry] += t->bytes;
        }
    }

    /* Per-layer KV estimate. The CTX hint is plumbed from the CLI
     * (--ctx, --ctx-max, etc.) so the packer accounts for the actual
     * session size the user intends to allocate. Zero/unset falls back to
     * the legacy 4096 value for back-compat with callers that don't
     * populate the option (single-tier paths, tests). Per-tier scratch
     * is accounted separately (see engine_per_tier_graph_overhead_bytes
     * and its pre-subtract in engine_classify_multi_tier). */
    const int est_ctx = (e->placement_ctx_hint > 0) ? e->placement_ctx_hint : 4096;
    const uint32_t session_count =
        DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA ?
        engine_placement_session_count(e) : 1u;
    ds4_context_memory mem =
        ds4_context_memory_estimate_with_prefill(DS4_BACKEND_CUDA,
                                                 est_ctx,
                                                 e->prefill_chunk);
    if (mem.total_bytes > 0) {
        /* mem.total_bytes is used only as a sentinel; cache values are
         * re-derived per layer so placement follows the active model's
         * allocation geometry. */
        for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) {
            const size_t per_session =
                DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA ?
                engine_glm_per_layer_kv_bytes_planner(il, est_ctx) :
                engine_per_layer_kv_bytes_planner(il, est_ctx,
                                                  e->prefill_chunk);
            const size_t cache_bytes =
                engine_size_mul_sat(per_session, session_count);
            out[il + 1] = out[il + 1] > SIZE_MAX - cache_bytes ?
                SIZE_MAX : out[il + 1] + cache_bytes;
        }
    } else {
        /* Fallback: 128 MiB per layer as a static estimate. */
        const size_t fallback_per_layer = engine_size_mul_sat(
                (size_t)128ull * 1024ull * 1024ull, session_count);
        for (uint32_t i = 1; i <= DS4_N_LAYER; i++) out[i] += fallback_per_layer;
    }
    return 0;
}

static bool engine_cuda_tp_decode_requested(const ds4_engine *e) {
#if defined(__APPLE__) && !defined(DS4_TEST_HOOKS)
    (void)e;
    return false;
#else
    return e && e->cuda_tensor_parallel &&
           e->gpu_cfg.n_gpus >= 2 && (e->gpu_cfg.n_gpus & 1) == 0;
#endif
}

static bool engine_cuda_tp_ep_requested(const ds4_engine *e) {
#if defined(__APPLE__) && !defined(DS4_TEST_HOOKS)
    (void)e;
    return false;
#else
    return engine_cuda_tp_decode_requested(e) &&
           g_ds4_shape.family == DS4_MODEL_FAMILY_DEEPSEEK4 &&
           DS4_N_EXPERT != 0u && (DS4_N_EXPERT & 1u) == 0u;
#endif
}

static bool engine_cuda_tp_output_env_requested(void) {
#if defined(__APPLE__) && !defined(DS4_TEST_HOOKS)
    return false;
#else
    const char *env = getenv("DS4_CUDA_TP_OUTPUT");
    return !env || !env[0] || strcmp(env, "0") != 0;
#endif
}

/* Return one output-weight shard for logical_tier. Zero means the tier does
 * not participate, one means a valid span was returned, and -1 is an invalid
 * output tensor/configuration. This matches the row partition used by every
 * output-TP execution path. */
static int engine_cuda_tp_output_shard_span(
        const ds4_engine *e,
        int               head_tier,
        int               logical_tier,
        uint64_t         *source_offset,
        uint64_t         *bytes) {
    if (!e || !e->weights.output || !source_offset || !bytes) return -1;
    const ds4_tensor *output = e->weights.output;
    if (output->type != DS4_TENSOR_Q8_0 || output->ndim != 2 ||
        output->dim[0] != DS4_N_EMBD || output->dim[1] < 2u ||
        output->bytes == 0u || output->bytes % output->dim[1] != 0u) {
        return -1;
    }

    int tiers[DS4_MAX_GPUS] = {0};
    const uint32_t ways = metal_graph_cuda_tp_output_tiers_for_head(
            head_tier, true, e->gpu_cfg.n_gpus, tiers);
    if (ways < 2u) return -1;
    for (uint32_t i = 0; i < ways; i++) {
        if (tiers[i] != logical_tier) continue;
        const uint64_t vocab = output->dim[1];
        const uint64_t row_bytes = output->bytes / vocab;
        const uint64_t row0 = (vocab * i) / ways;
        const uint64_t row1 = (vocab * (i + 1u)) / ways;
        *source_offset = output->abs_offset + row0 * row_bytes;
        *bytes = (row1 - row0) * row_bytes;
        return *bytes != 0u ? 1 : -1;
    }
    *source_offset = 0u;
    *bytes = 0u;
    return 0;
}

static int engine_reserve_cuda_ep_output_shards(
        const ds4_engine       *e,
        ds4_layer_pack_config  *pcfg) {
    if (!e || !pcfg || !engine_cuda_tp_output_env_requested()) return 0;
    const int n_stages = pcfg->n_gpus / 2;
    const int head_tier = n_stages - 1;
    for (int tier = 0; tier < pcfg->n_gpus; tier++) {
        uint64_t source_offset = 0;
        uint64_t shard_bytes = 0;
        const int shard = engine_cuda_tp_output_shard_span(
                e, head_tier, tier, &source_offset, &shard_bytes);
        (void)source_offset;
        if (shard < 0 || (shard > 0 &&
            shard_bytes > pcfg->gpu_budget_bytes[tier])) {
            fprintf(stderr,
                    "ds4: CUDA EP output shard cannot fit tier %d "
                    "(need %.2f GiB, budget %.2f GiB)\n",
                    tier,
                    (double)shard_bytes / 1073741824.0,
                    (double)pcfg->gpu_budget_bytes[tier] / 1073741824.0);
            return -1;
        }
        if (shard > 0) pcfg->gpu_budget_bytes[tier] -= (size_t)shard_bytes;
    }
    return 0;
}

static void engine_adjust_output_head_for_cuda_tp(ds4_engine *e,
                                                  const size_t *entry_bytes) {
    if (!e || !entry_bytes || e->n_placement_entries < (int)DS4_N_LAYER + 2) return;
    if (!engine_cuda_tp_decode_requested(e) ||
        !engine_cuda_tp_output_env_requested()) {
        return;
    }

    const int half = e->gpu_cfg.n_gpus / 2;
    const int head_entry = (int)DS4_N_LAYER + 1;
    const int head_tier = e->placement[head_entry];
    if (head_tier >= 0 && head_tier < half) return;

    size_t used[DS4_LAYER_PACK_MAX_GPUS] = {0};
    for (int i = 0; i < e->n_placement_entries; i++) {
        if (i == head_entry) continue;
        const int dev = e->placement[i];
        if (dev >= 0 && dev < e->gpu_cfg.n_gpus) used[dev] += entry_bytes[i];
    }

    const int layer_tier = e->placement[DS4_N_LAYER];
    int preferred = layer_tier;
    if (preferred >= half) preferred -= half;
    if (preferred < 0 || preferred >= half) preferred = -1;

    const size_t head_bytes = entry_bytes[head_entry];
    int target = -1;
    for (int pass = 0; pass < 2 && target < 0; pass++) {
        const int begin = pass == 0 && preferred >= 0 ? preferred : 0;
        const int end = pass == 0 && preferred >= 0 ? preferred + 1 : half;
        for (int tier = begin; tier < end; tier++) {
            if (pass != 0 && tier == preferred) continue;
            const size_t budget = e->gpu_cfg.vram_bytes[tier];
            if (head_bytes <= budget && used[tier] <= budget - head_bytes) {
                target = tier;
                break;
            }
        }
    }
    if (target >= 0) {
        fprintf(stderr,
                "ds4: CUDA output TP moved output head from tier %d to "
                "lower-half tier %d so the head can use partner tier %d\n",
                head_tier,
                target,
                target + half);
        e->placement[head_entry] = target;
    }
}

static int engine_compute_cuda_ep_placement(
        const size_t                *entry_bytes,
        int                          n_entries,
        const ds4_layer_pack_config *pcfg,
        int                         *placement) {
    if (!entry_bytes || !pcfg || !placement ||
        n_entries != (int)DS4_N_LAYER + 2 ||
        pcfg->n_gpus < 2 || (pcfg->n_gpus & 1) != 0) {
        return -1;
    }
    const int n_stages = pcfg->n_gpus / 2;
    if (n_stages > (int)DS4_N_LAYER) return -1;

    uint64_t remaining_layers = 0;
    for (int entry = 1; entry <= (int)DS4_N_LAYER; entry++) {
        if (remaining_layers > UINT64_MAX - entry_bytes[entry]) return -1;
        remaining_layers += entry_bytes[entry];
    }

    placement[0] = 0;
    int next_entry = 1;
    for (int stage = 0; stage < n_stages; stage++) {
        const int stages_left = n_stages - stage;
        const int layers_left = (int)DS4_N_LAYER - next_entry + 1;
        const uint64_t fixed_here =
            (stage == 0 ? entry_bytes[0] : 0u) +
            (stage == n_stages - 1 ? entry_bytes[n_entries - 1] : 0u);
        const uint64_t fixed_left =
            (stage == 0 ? entry_bytes[0] : 0u) + entry_bytes[n_entries - 1];
        const uint64_t target_total =
            (remaining_layers + fixed_left + (uint64_t)stages_left - 1u) /
            (uint64_t)stages_left;
        const uint64_t target_layers =
            target_total > fixed_here ? target_total - fixed_here : 0u;

        int take = 0;
        uint64_t layer_bytes = 0;
        const int max_take = layers_left - (stages_left - 1);
        const uint64_t home_budget = pcfg->gpu_budget_bytes[stage];
        const uint64_t partner_budget =
            pcfg->gpu_budget_bytes[stage + n_stages];
        if (fixed_here > home_budget) {
            fprintf(stderr,
                    "ds4: CUDA EP fixed weights do not fit stage %d home "
                    "(need %.2f GiB, budget %.2f GiB)\n",
                    stage,
                    (double)fixed_here / 1073741824.0,
                    (double)home_budget / 1073741824.0);
            return -1;
        }
        const uint64_t home_layer_budget = home_budget - fixed_here;
        const uint64_t layer_budget = home_layer_budget < partner_budget
            ? home_layer_budget : partner_budget;
        while (take < max_take) {
            const uint64_t next = entry_bytes[next_entry + take];
            if (layer_bytes > UINT64_MAX - next) return -1;
            const uint64_t candidate = layer_bytes + next;
            if (candidate > layer_budget) break;
            if (take > 0) {
                const uint64_t old_diff = layer_bytes > target_layers
                    ? layer_bytes - target_layers : target_layers - layer_bytes;
                const uint64_t new_diff = candidate > target_layers
                    ? candidate - target_layers : target_layers - candidate;
                if (new_diff > old_diff) break;
            }
            layer_bytes = candidate;
            take++;
        }
        if (take == 0 || (stage == n_stages - 1 && take != max_take)) {
            fprintf(stderr,
                    "ds4: CUDA EP cannot fit balanced stage %d in pair "
                    "budgets %.2f/%.2f GiB (%d layers remain)\n",
                    stage,
                    (double)home_budget / 1073741824.0,
                    (double)partner_budget / 1073741824.0,
                    layers_left);
            return -1;
        }
        for (int i = 0; i < take; i++) placement[next_entry + i] = stage;
        next_entry += take;
        remaining_layers -= layer_bytes;
    }
    if (next_entry != (int)DS4_N_LAYER + 1) return -1;
    placement[n_entries - 1] = n_stages - 1;
    return 0;
}

/* Phase A: classify multi-tier on a freshly-opened engine (model loaded,
 * weights bound). Pure CPU — no GPU init required. Sets e->multi_tier,
 * e->n_placement_entries, e->placement[], and e->gpu_cfg. Returns 0 on
 * success. NULL config is a no-op. */
static int engine_classify_multi_tier(ds4_engine *e, const ds4_gpu_config *cfg) {
    if (!cfg) {
        e->multi_tier = 0;
        e->n_placement_entries = 0;
        return 0;
    }
    if (cfg->n_gpus <= 0 || cfg->n_gpus > DS4_LAYER_PACK_MAX_GPUS) return -1;

    /* Caller-bug guard: ds4_gpu_config has no auto-detect; a zero-init
     * struct with only n_gpus and device_indices populated would
     * classify every entry as CPU spill and the engine would refuse.
     * That outcome is never useful, so reject it loudly. Auto-detection
     * belongs in the CLI layer (CLI flag wiring maps --gpu-vram auto to
     * cudaMemGetInfo before constructing the config). */
    size_t total_budget = 0;
    for (int d = 0; d < cfg->n_gpus; d++) total_budget += cfg->vram_bytes[d];
    if (total_budget == 0) {
        fprintf(stderr,
                "ds4: ds4_gpu_config has n_gpus=%d but every vram_bytes[d]==0; "
                "caller must populate explicit budgets (auto-detect is the CLI's "
                "job, not the engine's)\n",
                cfg->n_gpus);
        return -1;
    }

    e->gpu_cfg = *cfg;

    /* Pre-subtract per-tier Class-P graph scratch from EVERY device
     * budget BEFORE the packer reads vram_bytes.
     * Conservative: tiers that end up unused still reserve the overhead, so
     * the packer cannot accept a layout that would later OOM at
     * session_create when metal_graph_alloc_raw_cap's per-tier scratch loop
     * runs. The reservation flows into the packer via e->gpu_cfg (NOT
     * the caller's cfg) — see the budget loop below. */
    const size_t per_tier_overhead = engine_per_tier_graph_overhead_bytes(e);
    for (int d = 0; d < e->gpu_cfg.n_gpus; d++) {
        if (e->gpu_cfg.vram_bytes[d] <= per_tier_overhead) {
            fprintf(stderr,
                    "ds4: GPU%d budget %.2f GiB <= per-tier graph overhead "
                    "%.2f GiB; tier cannot hold its own runtime scratch — "
                    "refusing upfront.\n",
                    e->gpu_cfg.device_indices[d],
                    (double)e->gpu_cfg.vram_bytes[d] / (1024.0 * 1024.0 * 1024.0),
                    (double)per_tier_overhead       / (1024.0 * 1024.0 * 1024.0));
            return -1;
        }
        e->gpu_cfg.vram_bytes[d] -= per_tier_overhead;
    }

    size_t entry_bytes[DS4_MAX_LAYER + 2];
    if (engine_compute_entry_bytes(e, entry_bytes) != 0) return -1;

    ds4_layer_pack_config pcfg;
    memset(&pcfg, 0, sizeof(pcfg));
    pcfg.n_gpus = e->gpu_cfg.n_gpus;
    const size_t cublas_workspace_overhead = (size_t)64ull * 1024ull * 1024ull;
    for (int d = 0; d < e->gpu_cfg.n_gpus; d++) {
        /* Read post-subtract budgets from e->gpu_cfg (NOT the caller's
         * cfg) so the per-tier overhead pre-subtract actually flows into
         * the packer. */
        size_t budget = e->gpu_cfg.vram_bytes[d];
        size_t reserve = e->gpu_cfg.safety_margin_bytes + cublas_workspace_overhead;
        pcfg.gpu_budget_bytes[d] = budget > reserve ? budget - reserve : 0;
    }

    const bool cuda_tp_ep = engine_cuda_tp_ep_requested(e);
    if (cuda_tp_ep && engine_reserve_cuda_ep_output_shards(e, &pcfg) != 0) {
        return -1;
    }
    const int placement_rc = cuda_tp_ep
        ? engine_compute_cuda_ep_placement(entry_bytes, DS4_N_LAYER + 2,
                                           &pcfg, e->placement)
        : ds4_compute_layer_placement(entry_bytes, DS4_N_LAYER + 2, &pcfg,
                                      e->placement);
    if (placement_rc != 0) {
        return -1;
    }
    e->n_placement_entries = DS4_N_LAYER + 2;
    engine_adjust_output_head_for_cuda_tp(e, entry_bytes);

    int first_tier = e->placement[0];
    int multi_tier = 0;
    for (int i = 1; i < e->n_placement_entries; i++) {
        if (e->placement[i] != first_tier) { multi_tier = 1; break; }
    }
    if (!multi_tier) {
        for (int i = 0; i < e->n_placement_entries; i++) {
            if (e->placement[i] == DS4_LAYER_PACK_CPU) { multi_tier = 1; break; }
        }
    }
    e->multi_tier = multi_tier;
    return 0;
}

#ifndef DS4_NO_GPU
static int engine_append_device_cache_span(
        ds4_tensor_range **per_dev_ranges,
        int               *per_dev_n,
        int               *per_dev_cap,
        int                logical_tier,
        int                physical_device,
        uint64_t           source_offset,
        uint64_t           bytes) {
    if (logical_tier < 0 || logical_tier >= DS4_MAX_GPUS || bytes == 0u) return -1;
    if (per_dev_n[logical_tier] >= per_dev_cap[logical_tier]) {
        int new_cap = per_dev_cap[logical_tier] == 0 ? 64 : per_dev_cap[logical_tier] * 2;
        ds4_tensor_range *grow = realloc(per_dev_ranges[logical_tier],
                                          (size_t)new_cap * sizeof(*grow));
        if (!grow) {
            fprintf(stderr, "ds4: out of memory growing per-device range list\n");
            return -1;
        }
        per_dev_ranges[logical_tier] = grow;
        per_dev_cap[logical_tier] = new_cap;
    }
    per_dev_ranges[logical_tier][per_dev_n[logical_tier]].source_offset = source_offset;
    per_dev_ranges[logical_tier][per_dev_n[logical_tier]].bytes = bytes;
    per_dev_ranges[logical_tier][per_dev_n[logical_tier]].target_device = physical_device;
    per_dev_n[logical_tier]++;
    return 0;
}

static int engine_append_device_cache_range(
        ds4_tensor_range **per_dev_ranges,
        int               *per_dev_n,
        int               *per_dev_cap,
        int                logical_tier,
        int                physical_device,
        const ds4_tensor  *t) {
    if (!t) return -1;
    return engine_append_device_cache_span(per_dev_ranges, per_dev_n, per_dev_cap,
                                           logical_tier, physical_device,
                                           t->abs_offset, t->bytes);
}

/* Install per-device selective caches for GPU-placed tensors. Skips any
 * tensor whose entry is placed on CPU — that case must be rejected at a
 * higher level for this PR (execution wiring not yet shipped). */
static int engine_install_per_device_caches(ds4_engine *e) {
    /* Prereq: register the host model map so ds4_gpu_device_cache_tensors
     * can resolve g_model_host_base. We use the no-copy variant so
     * DS4_CUDA_COPY_MODEL cannot reintroduce a full-model copy that would
     * defeat the per-device selective cache. */
    if (!ds4_gpu_register_model_map_no_copy(e->model.map, e->model.size)) return -1;

    /* Per-logical-tier dynamic range lists. */
    ds4_tensor_range *per_dev_ranges[DS4_MAX_GPUS] = {0};
    int per_dev_n[DS4_MAX_GPUS] = {0};
    int per_dev_cap[DS4_MAX_GPUS] = {0};

    int rc = -1;
    const bool cuda_tp_decode = engine_cuda_tp_decode_requested(e);
    const bool cuda_tp_ep = engine_cuda_tp_ep_requested(e);
    const int tp_half = cuda_tp_decode ? e->gpu_cfg.n_gpus / 2 : 0;
    const bool cuda_tp_output =
        cuda_tp_decode && metal_graph_cuda_tp_output_requested();
    int output_tp_tiers[DS4_MAX_GPUS] = {0};
    const int output_head_entry = (int)DS4_N_LAYER + 1;
    const uint32_t output_tp_ways = cuda_tp_output
        ? metal_graph_cuda_tp_output_tiers_for_head(e->placement[output_head_entry],
                                                    true,
                                                    e->gpu_cfg.n_gpus,
                                                    output_tp_tiers)
        : 0;
    if (cuda_tp_ep && !metal_graph_cuda_tp_moe_requested()) {
        fprintf(stderr,
                "ds4: CUDA tensor parallelism requires routed MoE TP, but it is disabled\n");
        goto cleanup;
    }
    if (cuda_tp_ep && !metal_graph_cuda_tp_prefill_ffn_requested()) {
        fprintf(stderr,
                "ds4: CUDA tensor parallelism requires owned routed-MoE prefill, but it is disabled\n");
        goto cleanup;
    }
    if (cuda_tp_ep) {
        fprintf(stderr,
                "ds4: CUDA decode TP half-resident routed experts enabled "
                "([0,%u) home, [%u,%u) partner); other layer tensors remain replicated\n",
                DS4_N_EXPERT / 2u, DS4_N_EXPERT / 2u, DS4_N_EXPERT);
    } else if (cuda_tp_decode) {
        fprintf(stderr,
                "ds4: CUDA decode TP cache duplication enabled for layer tensors "
                "(tier N -> tier N+%d)\n",
                tp_half);
    }
    if (cuda_tp_output) {
        if (output_tp_ways < 2u) {
            fprintf(stderr,
                    "ds4: DS4_CUDA_TP_OUTPUT requires output-head home in "
                    "lower-half tiers; output head is on tier %d\n",
                    e->placement[output_head_entry]);
            goto cleanup;
        }
        fprintf(stderr,
                cuda_tp_ep
                    ? "ds4: CUDA output TP stores one vocabulary shard per tier (%u ways)\n"
                    : "ds4: CUDA output TP cache duplication enabled for output head (%u ways)\n",
                output_tp_ways);
    }
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        const ds4_tensor *t = &e->model.tensors[i];
        if (t->bytes == 0) continue;
        int entry = tensor_to_entry(t, DS4_N_LAYER);
        if (entry < 0 || entry >= e->n_placement_entries) entry = 0;
        int logical_tier = e->placement[entry];
        if (logical_tier == DS4_LAYER_PACK_CPU) continue;        /* CPU spill: skip here. */
        if (logical_tier < 0 || logical_tier >= e->gpu_cfg.n_gpus) {
            fprintf(stderr,
                    "ds4: placement tier %d out of range for tensor %.*s\n",
                    logical_tier,
                    (int)t->name.len, t->name.ptr ? t->name.ptr : "");
            goto cleanup;
        }
        const int physical_device = g_gpu[logical_tier].device_id;
        if (cuda_tp_ep && cuda_tp_output &&
            entry == output_head_entry && t == e->weights.output) {
            for (int tier = 0; tier < e->gpu_cfg.n_gpus; tier++) {
                uint64_t shard_offset = 0;
                uint64_t shard_bytes = 0;
                const int shard = engine_cuda_tp_output_shard_span(
                        e, logical_tier, tier, &shard_offset, &shard_bytes);
                if (shard < 0) {
                    fprintf(stderr,
                            "ds4: invalid CUDA output TP shard for tier %d\n",
                            tier);
                    goto cleanup;
                }
                if (shard == 0) continue;
                if (engine_append_device_cache_span(
                        per_dev_ranges, per_dev_n, per_dev_cap,
                        tier, g_gpu[tier].device_id,
                        shard_offset, shard_bytes) != 0) {
                    goto cleanup;
                }
            }
            continue;
        }
        uint64_t expert_bytes = 0;
        const bool shard_experts =
            cuda_tp_ep && engine_deepseek_routed_expert_tensor(
                                  e, t, entry, &expert_bytes);
        const uint64_t home_offset = t->abs_offset;
        const uint64_t home_bytes = shard_experts
            ? expert_bytes * (DS4_N_EXPERT / 2u) : t->bytes;
        if (engine_append_device_cache_span(per_dev_ranges, per_dev_n, per_dev_cap,
                                            logical_tier, physical_device,
                                            home_offset, home_bytes) != 0) {
            goto cleanup;
        }
        if (cuda_tp_decode && entry >= 1 && entry <= (int)DS4_N_LAYER) {
            if (logical_tier >= tp_half) {
                fprintf(stderr,
                        "ds4: CUDA tensor parallelism only supports layer homes in "
                        "lower-half tiers; tensor %.*s is on tier %d\n",
                        (int)t->name.len, t->name.ptr ? t->name.ptr : "",
                        logical_tier);
                goto cleanup;
            }
            const int partner_tier = logical_tier + tp_half;
            const int partner_device = g_gpu[partner_tier].device_id;
            const uint64_t partner_offset = shard_experts
                ? t->abs_offset + home_bytes : t->abs_offset;
            const uint64_t partner_bytes = shard_experts
                ? t->bytes - home_bytes : t->bytes;
            if (engine_append_device_cache_span(per_dev_ranges, per_dev_n, per_dev_cap,
                                                partner_tier, partner_device,
                                                partner_offset, partner_bytes) != 0) {
                goto cleanup;
            }
        }
        if (cuda_tp_output && !cuda_tp_ep &&
            entry == (int)DS4_N_LAYER + 1) {
            for (uint32_t j = 1; j < output_tp_ways; j++) {
                const int peer_tier = output_tp_tiers[j];
                if (peer_tier == logical_tier) continue;
                if (peer_tier < 0 || peer_tier >= e->gpu_cfg.n_gpus) {
                    fprintf(stderr,
                            "ds4: DS4_CUDA_TP_OUTPUT selected invalid tier %d "
                            "for tensor %.*s\n",
                            peer_tier,
                            (int)t->name.len, t->name.ptr ? t->name.ptr : "");
                    goto cleanup;
                }
                const int peer_device = g_gpu[peer_tier].device_id;
                if (engine_append_device_cache_range(per_dev_ranges, per_dev_n, per_dev_cap,
                                                     peer_tier, peer_device, t) != 0) {
                    goto cleanup;
                }
            }
        }
    }

    for (int d = 0; d < e->gpu_cfg.n_gpus; d++) {
        if (per_dev_n[d] == 0) continue;
        const int physical_device = g_gpu[d].device_id;
        uint64_t cache_bytes = 0;
        for (int i = 0; i < per_dev_n[d]; i++) {
            if (cache_bytes > UINT64_MAX - per_dev_ranges[d][i].bytes) {
                fprintf(stderr, "ds4: CUDA cache byte accounting overflow\n");
                goto cleanup;
            }
            cache_bytes += per_dev_ranges[d][i].bytes;
        }
        fprintf(stderr,
                "ds4: CUDA tier %d (device %d) selective weights: %.2f GiB in %d ranges\n",
                d, physical_device,
                (double)cache_bytes / 1073741824.0,
                per_dev_n[d]);
        int cache_rc = ds4_gpu_device_cache_tensors(physical_device,
                                                     per_dev_ranges[d],
                                                     per_dev_n[d]);
        if (cache_rc != 0) {
            fprintf(stderr,
                    "ds4: ds4_gpu_device_cache_tensors failed for tier %d "
                    "(physical device %d, %d ranges) rc=%d\n",
                    d, physical_device, per_dev_n[d], cache_rc);
            goto cleanup;
        }
    }
    rc = 0;

cleanup:
    for (int d = 0; d < DS4_MAX_GPUS; d++) free(per_dev_ranges[d]);
    return rc;
}

/* Pretty-print the layout via the multi-GPU plumbing helper, then emit a peer-access
 * summary by walking g_gpu_peer_ok[][]. Always called in multi-tier
 * mode so the operator sees what the packer decided. */
static void engine_print_layout(const ds4_engine *e) {
    size_t entry_bytes[DS4_MAX_LAYER + 2];
    (void)engine_compute_entry_bytes(e, entry_bytes);

    size_t used[DS4_LAYER_PACK_MAX_GPUS] = {0};
    size_t budget[DS4_LAYER_PACK_MAX_GPUS] = {0};
    for (int d = 0; d < e->gpu_cfg.n_gpus; d++) {
        budget[d] = e->gpu_cfg.vram_bytes[d];
    }
    for (int i = 0; i < e->n_placement_entries; i++) {
        int dev = e->placement[i];
        if (dev >= 0 && dev < e->gpu_cfg.n_gpus) used[dev] += entry_bytes[i];
    }

    ds4_layer_pack_print(stderr,
                          e->placement,
                          e->n_placement_entries,
                          DS4_N_LAYER,
                          entry_bytes,
                          used,
                          budget,
                          e->gpu_cfg.n_gpus);

    /* Show the per-tier graph scratch reservation so operators can
     * correlate "47 GiB free" with the smaller post-subtract budget the
     * packer actually had to spend. */
    const size_t per_tier_overhead = engine_per_tier_graph_overhead_bytes(e);
    fprintf(stderr,
            "ds4: per-tier graph scratch reserved: %.2f GiB "
            "(pre-subtracted from each GPU budget)\n",
            (double)per_tier_overhead / (1024.0 * 1024.0 * 1024.0));

    fprintf(stderr, "ds4: peer access matrix (validated):");
    int peer_any = 0;
    for (int i = 0; i < g_n_gpus; i++) {
        for (int j = 0; j < g_n_gpus; j++) {
            if (i == j) continue;
            peer_any = 1;
            const char *mode = g_gpu_peer_ok[i][j] ? "DIRECT" : "BOUNCE";
            fprintf(stderr, " %d->%d %s", i, j, mode);
        }
    }
    if (!peer_any) fprintf(stderr, " (single-device)");
    fputc('\n', stderr);
}

/* Install GPU-side placement state.
 *
 * Ordering:
 *   1. Print layout FIRST so it is always visible to the operator,
 *      even if subsequent steps fail.
 *   2. Detect any CPU-spill placement; this PR refuses to open
 *      multi-tier engines with CPU spill because the execution-side
 *      wiring (CPU↔GPU boundary materialization) lands in the
 *      follow-up multi-GPU execution task.
 *   3. Register host model map and install per-device selective
 *      caches.
 *
 * Returns 0 on success. */
/* Install the DSpark support model's tensors on one executor tier: the
 * TP partner of the output-head tier when decode TP is active (it has the
 * output-head copy from output TP and plenty of free VRAM), otherwise the
 * head tier itself. The strict multi-tier weight cache is offset-keyed, so
 * support tensors are registered with a disjoint offset bias. */
static int engine_install_dspark_support_cache(ds4_engine *e) {
    if (!e->multi_tier || e->support_kind != DS4_SUPPORT_DSPARK) return 0;
    if (!e->dspark) return 0;
    if (!e->mtp_model.map || e->mtp_model.n_tensors == 0) return 0;

    int exec_tier = e->placement[DS4_N_LAYER + 1];
    if (exec_tier < 0 || exec_tier >= e->gpu_cfg.n_gpus) exec_tier = 0;
    const bool tp_decode = e->cuda_tensor_parallel;
    const int tp_half = e->gpu_cfg.n_gpus / 2;
    if (tp_decode && e->gpu_cfg.n_gpus >= 2 && exec_tier < tp_half) {
        exec_tier += tp_half;
    }
    if (tp_decode && e->gpu_cfg.n_gpus >= 2) {
        /* Prefer the partner tier with the most free VRAM so the support
         * weights stay local to the executor. */
        uint64_t best_free = ds4_gpu_tier_free_vram(exec_tier);
        for (int t = tp_half; t < e->gpu_cfg.n_gpus; t++) {
            const uint64_t f = ds4_gpu_tier_free_vram(t);
            if (f > best_free) { best_free = f; exec_tier = t; }
        }
    }
    const char *tier_env = getenv("DS4_DSPARK_EXEC_TIER");
    if (tier_env && tier_env[0]) {
        const int v = atoi(tier_env);
        if (v >= 0 && v < e->gpu_cfg.n_gpus) exec_tier = v;
    }
    e->dspark_exec_tier = exec_tier;

    const uint64_t bias = (e->model.size + 4095ull) & ~4095ull;
    if (!ds4_gpu_register_support_map(e->mtp_model.map, e->mtp_model.size, bias)) {
        fprintf(stderr, "ds4: failed to register DSpark support model map\n");
        return -1;
    }

    /* Greedy pack: exec tier first, then the other TP-partner tiers.
     * Entries always claim the executor device; spilled tensors are read
     * through peer access. Per-tier budget leaves room for the graph
     * scratch and allocator slack that session_create allocates later. */
    int order[DS4_MAX_GPUS];
    int n_order = 0;
    order[n_order++] = exec_tier;
    for (int t = e->gpu_cfg.n_gpus - 1; t >= 0; t--) {
        if (t == exec_tier) continue;
        if (tp_decode && t < tp_half) continue; /* home tiers are packed full */
        order[n_order++] = t;
    }
    uint64_t reserve = 4ull * 1024ull * 1024ull * 1024ull + (1ull << 29);
    {
        const char *renv = getenv("DS4_DSPARK_CACHE_RESERVE_GB");
        if (renv && renv[0]) {
            const int gv = atoi(renv);
            if (gv >= 1 && gv <= 32) reserve = (uint64_t)gv << 30;
        }
    }
    const uint64_t range_cap =
        e->mtp_model.n_tensors > e->model.n_tensors
            ? e->mtp_model.n_tensors : e->model.n_tensors;
    ds4_tensor_range *ranges =
        xmalloc((size_t)range_cap * sizeof(ranges[0]));
    bool *placed = xmalloc((size_t)e->mtp_model.n_tensors * sizeof(placed[0]));
    memset(placed, 0, (size_t)e->mtp_model.n_tensors * sizeof(placed[0]));
    uint64_t remaining = 0;
    for (uint64_t ti = 0; ti < e->mtp_model.n_tensors; ti++) {
        if (e->mtp_model.tensors[ti].bytes != 0) remaining++;
    }
    for (int oi = 0; oi < n_order && remaining != 0; oi++) {
        const int tier = order[oi];
        const uint64_t free_b = ds4_gpu_tier_free_vram(tier);
        /* The executor tier later hosts the per-tier graph scratch, the
         * dspark capture buffers, and cuBLAS workspace; spill tiers only
         * host the graph scratch. */
        const uint64_t tier_reserve =
            tier == exec_tier ? reserve : (reserve > (1ull << 30)
                                               ? reserve - (1ull << 30) : reserve);
        uint64_t budget = free_b > tier_reserve ? free_b - tier_reserve : 0;
        fprintf(stderr,
                "ds4: DSpark support pack tier=%d free=%.2f GiB budget=%.2f GiB\n",
                tier,
                (double)free_b / 1073741824.0,
                (double)budget / 1073741824.0);
        if (budget == 0) continue;
        int n = 0;
        for (uint64_t ti = 0; ti < e->mtp_model.n_tensors; ti++) {
            const ds4_tensor *t = &e->mtp_model.tensors[ti];
            if (placed[ti] || t->bytes == 0) continue;
            if (t->bytes > budget) continue;
            ranges[n].source_offset = t->abs_offset;
            ranges[n].bytes = t->bytes;
            ranges[n].target_device = g_gpu[tier].device_id;
            n++;
            placed[ti] = true;
            budget -= t->bytes;
            remaining--;
        }
        if (n == 0) continue;
        const int rc = ds4_gpu_device_cache_support_tensors(
                g_gpu[tier].device_id,
                g_gpu[exec_tier].device_id,
                ranges, n, 0);
        if (rc != 0) {
            fprintf(stderr,
                    "ds4: DSpark support cache install failed on tier %d (rc=%d)\n",
                    tier, rc);
            free(placed);
            free(ranges);
            return -1;
        }
        fprintf(stderr,
                "ds4: DSpark support tensors cached on tier %d (%d tensors%s)\n",
                tier, n, tier == exec_tier ? ", executor tier" : ", peer spill");
    }
    /* The DSpark draft block embeds tokens through the BASE model's
     * embedding tensors, which normally live only on the embedding tier.
     * Install the embedding bucket (placement entry 0) on the executor
     * tier as well, through the normal main-model cache API. */
    {
        int n = 0;
        for (uint64_t ti = 0; ti < e->model.n_tensors; ti++) {
            const ds4_tensor *t = &e->model.tensors[ti];
            if (t->bytes == 0) continue;
            if (tensor_to_entry(t, DS4_N_LAYER) != 0) continue;
            ranges[n].source_offset = t->abs_offset;
            ranges[n].bytes = t->bytes;
            ranges[n].target_device = g_gpu[exec_tier].device_id;
            n++;
        }
        if (n != 0) {
            const int rc = ds4_gpu_device_cache_support_tensors(
                    g_gpu[exec_tier].device_id,
                    g_gpu[exec_tier].device_id,
                    ranges, n, 1);
            if (rc != 0) {
                fprintf(stderr,
                        "ds4: DSpark base embedding cache install failed on "
                        "tier %d (rc=%d)\n",
                        exec_tier, rc);
                free(placed);
                free(ranges);
                return -1;
            }
            fprintf(stderr,
                    "ds4: DSpark base embedding bucket cached on tier %d "
                    "(%d tensors)\n",
                    exec_tier, n);
        }
    }
    free(placed);
    free(ranges);
    if (remaining != 0) {
        fprintf(stderr,
                "ds4: DSpark support cache could not place %llu tensors "
                "(insufficient free VRAM); disabling DSpark runtime\n",
                (unsigned long long)remaining);
        return -1;
    }
    return 0;
}

static int engine_install_gpu_placement(ds4_engine *e) {
    if (!e->multi_tier) return 0;

    engine_print_layout(e);

    int has_cpu_spill = 0;
    for (int i = 0; i < e->n_placement_entries; i++) {
        if (e->placement[i] == DS4_LAYER_PACK_CPU) { has_cpu_spill = 1; break; }
    }
    if (has_cpu_spill) {
        fprintf(stderr,
            "ds4: CPU-spill placement detected; CPU-tier execution wiring lands in\n"
            "ds4: cpu-spill execution (follow-up) (CPU-spill execution). Aborting engine creation.\n");
        return -1;
    }

    if (engine_install_per_device_caches(e) != 0) return -1;
    return 0;
}
#endif /* !DS4_NO_GPU */

#ifdef DS4_TEST_HOOKS
typedef struct {
    const char *name;
    uint64_t bytes;
} ds4_test_fake_tensor;

int ds4_test_tensor_to_entry(const char *name, int name_len) {
    ds4_tensor fake;
    memset(&fake, 0, sizeof(fake));
    fake.name.ptr = name;
    fake.name.len = (uint64_t)(name_len > 0 ? name_len : 0);
    return tensor_to_entry(&fake, DS4_N_LAYER);
}

static int ds4_test_make_engine(
        ds4_engine *eng,
        const ds4_test_fake_tensor *tensors,
        int n_tensors,
        int placement_ctx_hint) {
    memset(eng, 0, sizeof(*eng));
    eng->model.fd = -1;
    eng->placement_ctx_hint = placement_ctx_hint;
    eng->placement_session_count_hint = 1;
    eng->model.n_tensors = (uint64_t)(n_tensors > 0 ? n_tensors : 0);
    if (n_tensors <= 0) return 0;
    eng->model.tensors = calloc((size_t)n_tensors,
                                sizeof(*eng->model.tensors));
    if (!eng->model.tensors) return -1;
    for (int i = 0; i < n_tensors; i++) {
        eng->model.tensors[i].name.ptr = tensors[i].name;
        eng->model.tensors[i].name.len = tensors[i].name
            ? (uint64_t)strlen(tensors[i].name) : 0;
        eng->model.tensors[i].bytes = tensors[i].bytes;
        if (tensors[i].name && !strcmp(tensors[i].name, "output.weight")) {
            eng->model.tensors[i].type = DS4_TENSOR_Q8_0;
            eng->model.tensors[i].ndim = 2;
            eng->model.tensors[i].dim[0] = DS4_N_EMBD;
            eng->model.tensors[i].dim[1] = DS4_N_VOCAB;
            eng->weights.output = &eng->model.tensors[i];
        } else if (tensors[i].name) {
            int il = -1;
            char suffix = '\0';
            if (sscanf(tensors[i].name,
                       "blk.%d.ffn_gate_exps.weight%c", &il, &suffix) == 1 &&
                il >= 0 && il < (int)DS4_N_LAYER) {
                eng->weights.layer[il].ffn_gate_exps = &eng->model.tensors[i];
            }
        }
    }
    return 0;
}

static int ds4_test_classify_multi_tier_impl(
        const ds4_test_fake_tensor *tensors,
        int n_tensors,
        const ds4_gpu_config *cfg,
        int placement_ctx_hint,
        bool cuda_tensor_parallel,
        int placement_out[DS4_MAX_LAYER + 2],
        int *out_multi_tier,
        int *out_n_entries) {
    ds4_engine eng;
    if (ds4_test_make_engine(&eng, tensors, n_tensors,
                             placement_ctx_hint) != 0) {
        return -1;
    }
    eng.cuda_tensor_parallel = cuda_tensor_parallel;
    eng.prefill_chunk =
        ds4_effective_prefill_chunk(cuda_tensor_parallel, 0);
    const int rc = engine_classify_multi_tier(&eng, cfg);
    if (rc == 0) {
        if (out_multi_tier) *out_multi_tier = eng.multi_tier;
        if (out_n_entries) *out_n_entries = eng.n_placement_entries;
        if (placement_out) {
            for (uint32_t i = 0; i < (uint32_t)DS4_N_LAYER + 2u; i++) {
                placement_out[i] = eng.placement[i];
            }
        }
    }
    free(eng.model.tensors);
    return rc;
}

int ds4_test_classify_multi_tier(
        const ds4_test_fake_tensor *tensors,
        int n_tensors,
        const ds4_gpu_config *cfg,
        int placement_out[DS4_MAX_LAYER + 2],
        int *out_multi_tier,
        int *out_n_entries) {
    return ds4_test_classify_multi_tier_impl(
            tensors, n_tensors, cfg, 0, false, placement_out,
            out_multi_tier, out_n_entries);
}

int ds4_test_classify_multi_tier_with_ctx(
        const ds4_test_fake_tensor *tensors,
        int n_tensors,
        const ds4_gpu_config *cfg,
        int placement_ctx_hint,
        int placement_out[DS4_MAX_LAYER + 2],
        int *out_multi_tier,
        int *out_n_entries) {
    return ds4_test_classify_multi_tier_impl(
            tensors, n_tensors, cfg, placement_ctx_hint, false, placement_out,
            out_multi_tier, out_n_entries);
}

int ds4_test_classify_multi_tier_with_ctx_cuda_tp(
        const ds4_test_fake_tensor *tensors,
        int n_tensors,
        const ds4_gpu_config *cfg,
        int placement_ctx_hint,
        int placement_out[DS4_MAX_LAYER + 2],
        int *out_multi_tier,
        int *out_n_entries) {
    return ds4_test_classify_multi_tier_impl(
            tensors, n_tensors, cfg, placement_ctx_hint, true, placement_out,
            out_multi_tier, out_n_entries);
}

void ds4_test_seed_compress_ratios(void) {
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        g_ds4_compress_ratios[il] = ds4_expected_layer_compress_ratio(il);
    }
}

void ds4_test_clear_compress_ratios(void) {
    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));
}

uint32_t ds4_test_effective_prefill_chunk(bool cuda_tensor_parallel,
                                          uint32_t requested_chunk) {
    return ds4_effective_prefill_chunk(cuda_tensor_parallel, requested_chunk);
}

uint32_t ds4_test_planner_prefill_cap(int prompt_len,
                                      uint32_t prefill_chunk) {
    return engine_planner_prefill_cap(prompt_len, prefill_chunk);
}

uint32_t ds4_test_planner_raw_cap(int ctx_size, uint32_t prefill_cap) {
    return engine_planner_raw_cap(ctx_size, prefill_cap);
}

size_t ds4_test_per_tier_graph_overhead_bytes_with_prefill(
        int placement_ctx_hint,
        uint32_t prefill_chunk) {
    ds4_engine eng;
    memset(&eng, 0, sizeof(eng));
    eng.model.fd = -1;
    eng.placement_ctx_hint = placement_ctx_hint;
    eng.prefill_chunk = prefill_chunk;
    return engine_per_tier_graph_overhead_bytes(&eng);
}

size_t ds4_test_per_tier_graph_overhead_bytes(int placement_ctx_hint) {
    return ds4_test_per_tier_graph_overhead_bytes_with_prefill(
            placement_ctx_hint, 0);
}

size_t ds4_test_compute_entry_bytes_sum_with_prefill(
        const ds4_test_fake_tensor *tensors,
        int n_tensors,
        int placement_ctx_hint,
        uint32_t prefill_chunk) {
    ds4_engine eng;
    if (ds4_test_make_engine(&eng, tensors, n_tensors,
                             placement_ctx_hint) != 0) {
        return 0;
    }
    eng.prefill_chunk = prefill_chunk;
    size_t entry_bytes[DS4_MAX_LAYER + 2];
    size_t sum = 0;
    if (engine_compute_entry_bytes(&eng, entry_bytes) == 0) {
        for (uint32_t i = 0; i < (uint32_t)DS4_N_LAYER + 2u; i++) {
            sum += entry_bytes[i];
        }
    }
    free(eng.model.tensors);
    return sum;
}

size_t ds4_test_compute_entry_bytes_sum(
        const ds4_test_fake_tensor *tensors,
        int n_tensors,
        int placement_ctx_hint) {
    return ds4_test_compute_entry_bytes_sum_with_prefill(
            tensors, n_tensors, placement_ctx_hint, 0);
}

size_t ds4_test_compute_glm_entry_bytes_sum_with_sessions(
        const ds4_test_fake_tensor *tensors,
        int n_tensors,
        int placement_ctx_hint,
        int placement_session_count_hint) {
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_GLM52;
    ds4_engine eng;
    if (ds4_test_make_engine(&eng, tensors, n_tensors,
                             placement_ctx_hint) != 0) {
        g_ds4_shape = saved_shape;
        return 0;
    }
    eng.placement_session_count_hint = placement_session_count_hint;
    size_t entry_bytes[DS4_MAX_LAYER + 2];
    size_t sum = 0;
    if (engine_compute_entry_bytes(&eng, entry_bytes) == 0) {
        for (uint32_t i = 0; i < (uint32_t)DS4_N_LAYER + 2u; i++) {
            sum += entry_bytes[i];
        }
    }
    free(eng.model.tensors);
    g_ds4_shape = saved_shape;
    return sum;
}

size_t ds4_test_glm_per_layer_kv_bytes(uint32_t layer, int ctx_size) {
    const ds4_shape saved_shape = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_GLM52;
    const size_t bytes =
        engine_glm_per_layer_kv_bytes_planner(layer, ctx_size);
    g_ds4_shape = saved_shape;
    return bytes;
}

int ds4_test_session_read_logits(ds4_session *s, float *out,
                                 uint64_t out_bytes) {
    if (!s || !out ||
        out_bytes < (uint64_t)DS4_N_VOCAB * sizeof(float)) {
        return 1;
    }
    return ds4_session_copy_logits(s, out, (int)DS4_N_VOCAB) ==
                   (int)DS4_N_VOCAB ? 0 : 1;
}

int ds4_test_session_seed_frontier(ds4_session *s, uint32_t pos,
                                   bool initialize_cache) {
#ifdef DS4_NO_GPU
    (void)s;
    (void)pos;
    (void)initialize_cache;
    return 1;
#else
    if (!s || !s->engine || s->distributed || ds4_session_is_cpu(s) ||
        ds4_session_is_glm(s) || s->engine->backend != DS4_BACKEND_CUDA ||
        pos >= (uint32_t)s->ctx_size || s->graph.raw_cap == 0u) {
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    if (pos > (uint32_t)INT_MAX) return 1;
    if (s->checkpoint.cap < (int)pos + 1) {
        s->checkpoint.v = xrealloc(
                s->checkpoint.v, ((size_t)pos + 1u) * sizeof(s->checkpoint.v[0]));
        s->checkpoint.cap = (int)pos + 1;
    }
    s->checkpoint.len = (int)pos;
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t n_comp = ratio ? pos / ratio : 0u;
        if (n_comp > g->layer_comp_cap[il]) return 1;
        g->layer_n_comp[il] = n_comp;
        g->layer_n_index_comp[il] = ratio == 4u ? n_comp : 0u;

        if (!initialize_cache) continue;
        if (!ds4_gpu_spark_zero_kv_rows_tensor(g->layer_raw_cache[il],
                                                g->raw_cap)) {
            return 1;
        }
        if (ratio == 0u) continue;
        if (n_comp != 0u &&
            !ds4_gpu_spark_zero_kv_rows_tensor(g->layer_attn_comp_cache[il],
                                                n_comp)) {
            return 1;
        }
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        const uint64_t attn_state_count =
            (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio;
        if (!metal_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f,
                                   attn_state_count) ||
            !metal_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF,
                                   attn_state_count)) {
            return 1;
        }
        if (ratio == 4u) {
            if (n_comp != 0u &&
                !ds4_gpu_spark_zero_index_rows_tensor(
                        g->layer_index_comp_cache[il], n_comp)) {
                return 1;
            }
            const uint64_t index_state_count =
                (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio;
            if (!metal_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f,
                                       index_state_count) ||
                !metal_tensor_fill_f32(g->layer_index_state_score[il],
                                       DS4_NEG_INF, index_state_count)) {
                return 1;
            }
        }
    }
    return ds4_gpu_synchronize() != 0 ? 0 : 1;
#endif
}

const int *ds4_test_engine_placement(const ds4_engine *e) {
    return e ? e->placement : NULL;
}
#endif /* DS4_TEST_HOOKS */

static int engine_install_dspark_support_cache(ds4_engine *e);
static int engine_install_gpu_placement(ds4_engine *e);
static int ds4_engine_open_internal(ds4_engine **out,
                                    const ds4_engine_options *opt,
                                    const ds4_gpu_config *gpu_cfg);

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    return ds4_engine_open_internal(out, opt, NULL);
}

int ds4_engine_create_with_gpu_config(ds4_engine **out,
                                       const ds4_engine_options *opt,
                                       const struct ds4_gpu_config *gpu_cfg) {
    return ds4_engine_open_internal(out, opt, gpu_cfg);
}

static int ds4_engine_open_internal(ds4_engine **out,
                                     const ds4_engine_options *opt,
                                     const ds4_gpu_config *gpu_cfg) {
    ds4_engine *e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->mtp_model.fd = -1;
    e->backend = opt->backend;
    e->quality = opt->quality;
    e->glm_mtp = opt->glm_mtp;
    e->glm_mtp_timing = opt->glm_mtp_timing;
    e->dspark = opt->dspark;
    e->dspark_strict = opt->dspark_strict;
    e->cuda_tensor_parallel = opt->cuda_tensor_parallel;
    e->glm_tp_token_prefill = opt->tp.glm_token_prefill;
    e->ssd_streaming = opt->ssd_streaming;
    e->ssd_streaming_cold = opt->ssd_streaming_cold;
    e->ssd_streaming_full_layers_set = opt->ssd_streaming_full_layers_set;
    e->distributed = opt->distributed;
    e->power_percent = opt->power_percent > 0 ? opt->power_percent : 100;
    e->prefill_chunk =
        ds4_effective_prefill_chunk(opt->cuda_tensor_parallel,
                                    opt->prefill_chunk);
    e->ssd_streaming_cache_experts = opt->ssd_streaming_cache_experts;
    e->ssd_streaming_cache_bytes = opt->ssd_streaming_cache_bytes;
    e->ssd_streaming_full_layers = opt->ssd_streaming_full_layers;
    e->ssd_streaming_preload_experts = opt->ssd_streaming_preload_experts;
    if (e->power_percent > 100) e->power_percent = 100;
    e->mtp_draft_tokens = opt->mtp_draft_tokens > 0 ? opt->mtp_draft_tokens : 1;
    if (e->mtp_draft_tokens > 16) e->mtp_draft_tokens = 16;
    e->mtp_margin = opt->mtp_margin >= 0.0f ? opt->mtp_margin : 3.0f;
    if (opt->dspark_confidence_threshold_set) {
        e->dspark_confidence_threshold = opt->dspark_confidence_threshold;
    } else {
        e->dspark_confidence_threshold =
            e->backend == DS4_BACKEND_METAL ? 0.6f : 0.7f;
    }
    if (opt->cuda_tensor_parallel &&
        (opt->backend != DS4_BACKEND_CUDA || !gpu_cfg ||
         gpu_cfg->n_gpus < 2 || (gpu_cfg->n_gpus & 1) != 0)) {
        fprintf(stderr,
                "ds4: --cuda-tensor-parallel requires an even multi-GPU CUDA placement\n");
        free(e);
        *out = NULL;
        return 1;
    }
    if (opt->dspark && (!opt->mtp_path || !opt->mtp_path[0])) {
        fprintf(stderr, "ds4: --dspark requires --mtp FILE\n");
        free(e);
        *out = NULL;
        return 1;
    }
    if ((opt->directional_steering_attn != 0.0f || opt->directional_steering_ffn != 0.0f) &&
        (!opt->directional_steering_file || !opt->directional_steering_file[0]))
    {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        free(e);
        *out = NULL;
        return 1;
    }
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        e->directional_steering_file = ds4_strdup(opt->directional_steering_file);
        e->directional_steering_attn_scale = opt->directional_steering_attn;
        e->directional_steering_ffn_scale = opt->directional_steering_ffn;
    }
    if (opt->n_threads > 0) g_requested_threads = (uint32_t)opt->n_threads;
    e->placement_ctx_hint = opt->placement_ctx_hint;
    e->placement_session_count_hint = opt->placement_session_count_hint;
    e->share_session_prefill_workspace = opt->share_session_prefill_workspace;
    ds4_acquire_instance_lock();

    if (opt->simulate_used_memory_bytes != 0 &&
        !ds4_ssd_memory_lock_acquire(&e->simulated_memory,
                                     opt->simulate_used_memory_bytes)) {
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }

    bool load_slice = opt->load_slice;
    uint32_t load_layer_start = opt->load_layer_start;
    uint32_t load_layer_end = opt->load_layer_end;
    bool load_output = opt->load_output;
    bool load_output_optional = false;
    if (opt->distributed.role != DS4_DISTRIBUTED_NONE &&
        opt->distributed.layers.set)
    {
        load_slice = true;
        load_layer_start = opt->distributed.layers.start;
        load_layer_end = opt->distributed.layers.end;
        load_output = opt->distributed.layers.has_output;
        /* A coordinator may need to apply the output head locally when the
         * last worker advertises N:M rather than N:output. Bind it when the
         * local GGUF contains it, without requiring split GGUFs to do so. */
        load_output_optional =
            opt->distributed.role == DS4_DISTRIBUTED_COORDINATOR;
    }

    const bool graph_backend = ds4_backend_uses_graph(opt->backend);
    if (graph_backend) ds4_linux_graph_backend_set_oom_score(opt->backend);
    model_open(&e->model, opt->model_path, graph_backend, !opt->inspect_only);
    if (opt->warm_weights) model_warm_weights(&e->model);
    config_validate_model(&e->model);
    if (load_slice && load_layer_end == UINT32_MAX) {
        const uint32_t normal_layers = ds4_model_normal_layer_count();
        if (normal_layers == 0) {
            fprintf(stderr, "ds4: model reports no executable transformer layers\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        load_layer_end = normal_layers - 1u;
    }
    if (e->distributed.role != DS4_DISTRIBUTED_NONE &&
        e->distributed.layers.set) {
        e->distributed.layers.end = load_layer_end;
    }
    if (e->cuda_tensor_parallel &&
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DEEPSEEK4) {
        fprintf(stderr,
                "ds4: --cuda-tensor-parallel is currently supported only for DeepSeek models\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    if (e->ssd_streaming && !ds4_backend_supports_ssd_streaming(e->backend)) {
        fprintf(stderr, "ds4: --ssd-streaming is currently supported only with --metal/--cuda/--rocm\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    const char *expert_profile_path = opt->expert_profile_path;
    if (!expert_profile_path || !expert_profile_path[0]) {
        expert_profile_path = getenv("DS4_EXPERT_PROFILE");
    }
    const char *expert_hotlist_path = getenv("DS4_EXPERT_HOTLIST");
    if ((expert_profile_path && expert_profile_path[0]) ||
        (expert_hotlist_path && expert_hotlist_path[0])) {
        if (e->backend == DS4_BACKEND_METAL) {
            ds4_expert_profile_init(expert_profile_path, expert_hotlist_path);
        } else {
            fprintf(stderr,
                    "ds4: expert profile/hotlist is Metal-only for now; ignoring for %s backend\n",
                    ds4_backend_name(e->backend));
        }
    }
    weights_bind(&e->weights,
                 &e->model,
                 load_slice,
                 load_layer_start,
                 load_layer_end,
                 load_output,
                 load_output_optional);

    /* TP always maps one contiguous routed-expert half per rank. Decide
     * immediately after binding so memory guards account only the bytes this
     * rank owns (replicated dense weights plus its expert shard). */
#ifndef DS4_NO_GPU
    const bool tp_shard =
        opt->tp.role != DS4_TP_NONE &&
        !e->ssd_streaming;
    const int tp_shard_rank = opt->tp.role == DS4_TP_WORKER ? 1 : 0;
    if (tp_shard && DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        uint32_t bad_layer = 0;
        uint32_t bad_type = 0;
        if (!glm_tp_validate_ownership_kernels(&e->weights,
                                               &bad_layer,
                                               &bad_type)) {
            fprintf(stderr,
                    "ds4: GLM tensor parallelism lacks ownership-aware "
                    "kernels for routed expert type %u in layer %u\n",
                    bad_type,
                    bad_layer);
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
    }
    g_tp_shard_model_bytes = 0;
    if (tp_shard) {
        ds4_model_map_span_vec shard_spans;
        if (weights_model_map_sharded_spans(&e->weights, &e->model,
                                            tp_shard_rank, &shard_spans)) {
            g_tp_shard_model_bytes =
                model_map_span_vec_total_bytes(&shard_spans);
            free(shard_spans.v);
        }
    }
#endif
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        if (opt->inspect_only) {
            *out = e;
            return 0;
        }
        if ((opt->directional_steering_file && opt->directional_steering_file[0]) ||
            opt->directional_steering_attn != 0.0f ||
            opt->directional_steering_ffn != 0.0f) {
            fprintf(stderr, "ds4: directional steering is not supported for GLM 5.2 yet\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (e->power_percent < 100) {
            fprintf(stderr, "ds4: --power is not supported for GLM 5.2 yet\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (opt->prefill_chunk != 0) {
            fprintf(stderr,
                    "ds4: --prefill-chunk is not supported for GLM 5.2; "
                    "GLM uses graph-selected prefill chunks\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (opt->mtp_path && opt->mtp_path[0]) {
            fprintf(stderr, "ds4: --mtp is not supported for GLM 5.2 yet\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (opt->first_token_test) {
            if (e->backend != DS4_BACKEND_CPU) {
                fprintf(stderr, "ds4: GLM first-token test is CPU-only; pass --cpu\n");
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            vocab_load(&e->vocab, &e->model);
            *out = e;
            return 0;
        }
        bool glm_backend_supported = ds4_backend_uses_graph(e->backend);
#ifdef DS4_ROCM_BUILD
        const bool rocm_full_model_requires_streaming =
            e->backend == DS4_BACKEND_CUDA &&
            !e->ssd_streaming &&
            !load_slice;
        if (rocm_full_model_requires_streaming) {
            glm_backend_supported = false;
        }
#endif
        if (!glm_backend_supported) {
#ifdef DS4_ROCM_BUILD
            if (rocm_full_model_requires_streaming) {
                fprintf(stderr,
                        "ds4: full-model GLM 5.2 ROCm inference requires "
                        "--ssd-streaming; distributed layer slices can run "
                        "fully resident\n");
            } else {
                fprintf(stderr,
                        "ds4: GLM 5.2 inference requires the ROCm graph "
                        "backend; use --inspect or --cpu --first-token-test "
                        "for CPU diagnostics\n");
            }
#else
            fprintf(stderr,
                    "ds4: GLM 5.2 inference requires the Metal or CUDA graph "
                    "backend; use --inspect or --cpu --first-token-test for CPU diagnostics\n");
#endif
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
#ifndef DS4_NO_GPU
        if (opt->context_size > 0) {
            uint32_t guard_ctx = 0;
            if (!glm_graph_context_request(opt->context_size, &guard_ctx) ||
                !(load_slice ?
                    glm_graph_memory_guard_slice(&e->model,
                                                 &e->weights,
                                                 e->ssd_streaming,
                                                 load_layer_start,
                                                 load_layer_end,
                                                 load_layer_start == 0,
                                                 load_output ||
                                                     (load_output_optional &&
                                                      weights_have_output_head(
                                                              &e->weights)),
                                                 guard_ctx) :
                    glm_graph_memory_guard(&e->model,
                                           &e->weights,
                                           e->ssd_streaming,
                                           guard_ctx))) {
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
        }
#endif
        vocab_load(&e->vocab, &e->model);
    } else if (!opt->inspect_only) {
        vocab_load(&e->vocab, &e->model);
    }
    if (e->ssd_streaming && e->ssd_streaming_cache_bytes != 0) {
        const uint64_t requested_cache_bytes = e->ssd_streaming_cache_bytes;
        const uint64_t safe_cache_bytes =
            ds4_streaming_manual_cache_safe_bytes(e->backend,
                                                  opt->context_size,
                                                  e->prefill_chunk,
                                                  e->ssd_streaming);
        if (safe_cache_bytes != 0 &&
            e->ssd_streaming_cache_bytes > safe_cache_bytes) {
            e->ssd_streaming_cache_bytes = safe_cache_bytes;
            fprintf(stderr,
                    "ds4: %s SSD streaming cache budget %.2f GiB capped to %.2f GiB "
                    "to stay below the graph working-set pressure budget\n",
                    ds4_backend_name(e->backend),
                    (double)requested_cache_bytes / 1073741824.0,
                    (double)e->ssd_streaming_cache_bytes / 1073741824.0);
        }
    }
    if (opt->inspect_only) {
        if (opt->mtp_path && opt->mtp_path[0] &&
            opt->distributed.role == DS4_DISTRIBUTED_NONE) {
            model_open(&e->mtp_model, opt->mtp_path, false, false);
            ds4_dspark_summary dspark = {0};
            e->support_kind =
                support_model_detect(&e->mtp_model, &e->support_stages, &dspark);
            if (e->support_kind == DS4_SUPPORT_DSPARK) {
                dspark_weights_bind_optional(&e->dspark_weights,
                                             &e->mtp_model,
                                             &dspark);
            }
        }
        *out = e;
        return 0;
    }
    if (e->backend == DS4_BACKEND_CPU && !cpu_load_directional_steering(e)) {
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    if (engine_classify_multi_tier(e, gpu_cfg) != 0) {
        fprintf(stderr, "ds4: failed to classify multi-tier placement\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    if (e->ssd_streaming && e->multi_tier) {
        fprintf(stderr,
                "ds4: --ssd-streaming is not compatible with multi-GPU placement\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
    if (gpu_cfg && e->n_placement_entries > 0) {
        int spilled = 0;
        size_t spilled_bytes = 0;
        size_t entry_bytes_buf[DS4_MAX_LAYER + 2];
        size_t used_bytes[DS4_LAYER_PACK_MAX_GPUS] = {0};
        size_t budget_bytes[DS4_LAYER_PACK_MAX_GPUS] = {0};
        const int have_entry_bytes =
            engine_compute_entry_bytes(e, entry_bytes_buf) == 0;
        if (have_entry_bytes) {
            for (int d = 0; d < e->gpu_cfg.n_gpus; d++) {
                budget_bytes[d] = e->gpu_cfg.vram_bytes[d];
            }
            for (int i = 0; i < e->n_placement_entries; i++) {
                const int dev = e->placement[i];
                if (dev == DS4_LAYER_PACK_CPU) {
                    spilled++;
                    spilled_bytes += entry_bytes_buf[i];
                } else if (dev >= 0 && dev < e->gpu_cfg.n_gpus) {
                    used_bytes[dev] += entry_bytes_buf[i];
                }
            }
        }
        if (spilled > 0) {
            size_t total_budget = 0;
            for (int d = 0; d < e->gpu_cfg.n_gpus; d++) {
                total_budget += e->gpu_cfg.vram_bytes[d];
            }
            if (have_entry_bytes) {
                ds4_layer_pack_print(stderr,
                                     e->placement,
                                     e->n_placement_entries,
                                     DS4_N_LAYER,
                                     entry_bytes_buf,
                                     used_bytes,
                                     budget_bytes,
                                     e->gpu_cfg.n_gpus);
            }
            fprintf(stderr,
                    "ds4: per-tier graph scratch reserved on each GPU: "
                    "%.2f GiB (pre-subtracted from each --gpu-vram budget "
                    "before packing).\n",
                    (double)engine_per_tier_graph_overhead_bytes(e) /
                        (1024.0 * 1024.0 * 1024.0));
            fprintf(stderr,
                    "ds4: CPU-spill placement detected; CPU-tier execution wiring "
                    "is the wave-3b mgpu-graph-session-cpu-spill follow-up.\n");
            fprintf(stderr,
                    "ds4: --gpu-vram placement does not fit at the requested "
                    "context (ctx hint = %d):\n"
                    "ds4:   %d placement entries spilled to CPU "
                    "(%.2f GiB unaccommodated of %.2f GiB total per-device budget).\n"
                    "ds4: Lower --ctx / --ctx-max, raise --gpu-vram budgets, or use "
                    "--gpu-vram auto on a host with more free VRAM.\n"
                    "ds4: Refusing upfront to avoid silent OOM at session_create.\n",
                    e->placement_ctx_hint,
                    spilled,
                    (double)spilled_bytes / (1024.0 * 1024.0 * 1024.0),
                    (double)total_budget / (1024.0 * 1024.0 * 1024.0));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
    }
    if (opt->mtp_path && opt->mtp_path[0] &&
        opt->distributed.role == DS4_DISTRIBUTED_NONE) {
        if (e->ssd_streaming) {
            fprintf(stderr, "ds4: --ssd-streaming is not compatible with --mtp yet\n");
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        model_open(&e->mtp_model, opt->mtp_path, graph_backend, true);
        ds4_dspark_summary dspark = {0};
        e->support_kind =
            support_model_detect(&e->mtp_model, &e->support_stages, &dspark);
        if (e->support_kind == DS4_SUPPORT_MTP_LEGACY) {
            if (opt->tp.role != DS4_TP_NONE) {
                fprintf(stderr,
                        "ds4: legacy MTP support is ignored under tensor parallelism; "
                        "using the normal TP decode path\n");
                model_close(&e->mtp_model);
                e->support_kind = DS4_SUPPORT_NONE;
                e->support_stages = 0;
            } else {
                mtp_weights_bind(&e->mtp_weights, &e->mtp_model);
                e->mtp_ready = true;
                fprintf(stderr, "ds4: MTP support model loaded: %s (draft=%d)\n",
                        opt->mtp_path,
                        e->mtp_draft_tokens);
            }
        } else if (e->support_kind == DS4_SUPPORT_DSPARK) {
            dspark_weights_bind_optional(&e->dspark_weights,
                                         &e->mtp_model,
                                         &dspark);
            fprintf(stderr,
                    "ds4: DSpark support model detected: %s "
                    "(stages=%u block=%u markov_rank=%u tensors=%u missing=%u "
                    "invalid=%u metadata_errors=%u); "
                    "use --dspark to enable experimental runtime decode\n",
                    opt->mtp_path,
                    e->support_stages,
                    dspark.block_size,
                    dspark.markov_rank,
                    e->dspark_weights.present_tensors,
                    e->dspark_weights.missing_tensors,
                    e->dspark_weights.invalid_tensors,
                    e->dspark_weights.metadata_errors);
            if (e->dspark && !e->quality && !e->dspark_strict) {
                fprintf(stderr,
                        "ds4: DSpark direct verifier-state commits enabled; "
                        "greedy output may differ from one-token decode due "
                        "to batched floating-point operation order\n");
            }
        } else {
            fprintf(stderr,
                    "ds4: unsupported --mtp support model %s (detected=%s); "
                    "expected legacy MTP or DSpark tensors\n",
                    opt->mtp_path,
                    support_kind_name(e->support_kind));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
    }

#ifndef DS4_NO_GPU
    if (e->backend == DS4_BACKEND_CUDA) {
#ifdef __APPLE__
        fprintf(stderr, "ds4: CUDA backend requested but this build is linked with Metal, not CUDA\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
#endif
    }
    if (e->backend == DS4_BACKEND_METAL) {
#ifndef __APPLE__
        fprintf(stderr, "ds4: Metal backend requested but this build is linked with CUDA, not Metal\n");
        ds4_engine_close(e);
        *out = NULL;
        return 1;
#endif
    }
    /* With a raised wired limit the sharded span views (~97 GiB) fit the
     * GPU budget, so let the residency set pin them — that is what makes
     * the shard actually resident.  Without the sysctl, fall back to lazy
     * faulting (slow but functional). */
    if (graph_backend && tp_shard && glm_graph_wired_limit_bytes() == 0) {
        fprintf(stderr,
                "ds4: iogpu.wired_limit_mb is 0 -- TP expert shard will page "
                "lazily; raise it (e.g. sudo sysctl iogpu.wired_limit_mb=120000) "
                "for full residency\n");
        ds4_gpu_model_residency_skip(1);
    }
    if (graph_backend) {
        if (e->multi_tier) {
            /* Wave-2 multi-tier branch.
             *
             * 1. Initialize all configured CUDA devices via ds4_gpu_init_multi
             *    (the multi-GPU plumbing multi-device init that populates g_gpu[]).
             * 2. Skip the legacy ds4_gpu_set_model_map_range / accelerator
             *    cache calls — those are single-tier behavior.
             * 3. Install per-device selective caches per the placement.
             * 4. Refuse engine creation with the documented notice: this PR
             *    ships scaffolding only; execution wiring lands in
             *    multi-GPU execution. */
            /* ds4_gpu_init_multi returns 1 on success and 0 on failure
             * (matches ds4_gpu_init). */
            if (ds4_gpu_init_multi(gpu_cfg) == 0) {
                fprintf(stderr, "ds4: ds4_gpu_init_multi failed; aborting multi-tier startup\n");
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            e->metal_ready = true;
            ds4_gpu_set_quality(e->quality);
#ifdef DS4_ROCM_BUILD
            /*
             * The ROCm Q8 decode selector must know that a multi-tier model
             * is GLM before any layer dispatch.  The single-tier path sets
             * the same model-family state below.
             */
            ds4_gpu_set_glm_model(
                    DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA);
#endif
            (void)ds4_gpu_set_model_fd(e->model.fd);

            if (engine_install_gpu_placement(e) != 0) {
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            if (engine_install_dspark_support_cache(e) != 0) {
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            /* GPU-only multi-tier execution is now wired up
             * (B2-B6: per-tier graph allocation, dispatch loops, boundary
             * copies). CPU-spill placements were rejected by
             * engine_install_gpu_placement above with stderr naming the
             * CPU-spill execution follow-up. Skip the single-tier ds4_gpu_init /
             * set_model_map_range path below — those calls are explicitly
             * single-tier behavior (per the pre-B7 comment block). */
            *out = e;
            return 0;
        }

        /* Single-tier path (every existing caller). Body is byte-equivalent
         * to pre-multi-GPU CLI main. */
        e->metal_ready = ds4_gpu_init() != 0;
        if (!e->metal_ready) {
            fprintf(stderr, "ds4: %s backend unavailable; aborting startup\n",
                    ds4_backend_name(e->backend));
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        ds4_gpu_set_quality(e->quality);
        ds4_gpu_set_glm_model(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA);
        ds4_gpu_set_ssd_streaming(e->ssd_streaming);
        if (!ds4_engine_configure_streaming_auto_cache(e)) {
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (!ds4_engine_configure_streaming_cache_budget(e)) {
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (!ds4_engine_glm_streaming_memory_guard(
                    e,
                    load_slice,
                    load_layer_start,
                    load_layer_end,
                    load_output ||
                        (load_output_optional &&
                         weights_have_output_head(&e->weights)),
                    opt->context_size,
                    "after GLM streaming cache budget")) {
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        ds4_gpu_set_streaming_expert_cache_budget(e->ssd_streaming_cache_experts);
        if (e->ssd_streaming) {
            /*
             * Pin the expert cache's slab size class to the model's uniform
             * per-expert bytes, and count mixed-precision (boosted) layers:
             * those are served through mapped model views instead of the
             * cache (see weights_streaming_layer_experts_uniform).
             */
            uint64_t slab_expert_bytes = 0;
            if (ds4_streaming_routed_expert_bytes(&e->weights, &slab_expert_bytes)) {
                ds4_gpu_set_streaming_expert_cache_expert_bytes(slab_expert_bytes);
                uint32_t routed = 0, boosted = 0;
                for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                    const ds4_layer_weights *l = &e->weights.layer[il];
                    if (!l->ffn_gate_exps || !l->ffn_up_exps || !l->ffn_down_exps) continue;
                    routed++;
                    if (!weights_streaming_layer_experts_uniform(&e->weights, il)) boosted++;
                }
                if (boosted > 0) {
                    fprintf(stderr,
                            "ds4: SSD streaming mixed-precision model: %u/%u routed layers "
                            "off the slab size class will bypass the expert cache and read "
                            "experts via mapped model views\n",
                            boosted, routed);
                }
                if (boosted * 2 > routed) {
                    fprintf(stderr,
                            "ds4: WARNING: the majority of routed layers (%u/%u) are off the "
                            "slab size class (is the FIRST routed layer itself boosted?); "
                            "expert-cache hit rate will be catastrophic\n",
                            boosted, routed);
                }
                /*
                 * Below one token's routed working set (uniform routed layers
                 * x experts used) every token evicts entries it is about to
                 * reuse, and prefill serves layer overflow through mapped
                 * model views.  Output stays byte-identical at any budget
                 * (the addr-table kernels read the same bytes either way);
                 * only throughput collapses, so warn instead of refusing.
                 */
                const uint64_t min_experts =
                    (uint64_t)(routed - boosted) * DS4_N_EXPERT_USED;
                if (min_experts != 0 &&
                    e->ssd_streaming_cache_experts != 0 &&
                    e->ssd_streaming_cache_experts < 2u * min_experts) {
                    fprintf(stderr,
                            "ds4: WARNING: SSD streaming expert cache (%u experts) is "
                            "under twice the per-token routed working set (%u layers "
                            "x %u experts = %llu); expect heavy thrashing below "
                            "%.2f GiB\n",
                            e->ssd_streaming_cache_experts,
                            routed - boosted,
                            DS4_N_EXPERT_USED,
                            (unsigned long long)min_experts,
                            (double)(2u * min_experts * slab_expert_bytes) /
                                1073741824.0);
                }
            }
        }
        (void)ds4_gpu_set_model_fd(e->model.fd);
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
        if (e->backend == DS4_BACKEND_CUDA &&
            !load_slice && !tp_shard && !e->ssd_streaming) {
            (void)ds4_gpu_build_derived_artifacts(e->model.map,
                                                  e->model.size,
                                                  opt->model_path);
        }
#endif
        int model_map_ok = 0;
        uint64_t *load_offsets = NULL;
        uint64_t *load_sizes = NULL;
        uint32_t load_span_count = 0;
        if (e->ssd_streaming) {
            const bool map_output = load_slice &&
                                    (load_output ||
                                     (load_output_optional &&
                                      weights_have_output_head(&e->weights)));
            ds4_model_map_span_vec spans;
            bool spans_ok = false;
            if (load_slice) {
                spans_ok = weights_model_map_decode_static_slice_spans(
                        &e->weights,
                        load_layer_start,
                        load_layer_end,
                        load_layer_start == 0,
                        map_output,
                        &spans);
            } else {
                spans_ok = weights_model_map_token_spans(&e->weights, &spans);
            }
            if (!spans_ok) {
                fprintf(stderr, "ds4: invalid SSD streaming initial token embedding map\n");
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            uint64_t *offsets = xmalloc((size_t)spans.len * sizeof(offsets[0]));
            uint64_t *sizes = xmalloc((size_t)spans.len * sizeof(sizes[0]));
            uint64_t span_bytes = 0;
            for (uint32_t i = 0; i < spans.len; i++) {
                offsets[i] = spans.v[i].off;
                sizes[i] = spans.v[i].end - spans.v[i].off;
                span_bytes += sizes[i];
            }
            load_offsets = offsets;
            load_sizes = sizes;
            load_span_count = spans.len;
            e->startup_model_span_bytes = span_bytes;
            if (load_slice) {
                char load_end[32];
                if (map_output && load_layer_end == UINT32_MAX) {
                    snprintf(load_end, sizeof(load_end), "output");
                } else if (map_output) {
                    snprintf(load_end, sizeof(load_end), "%u+output", load_layer_end);
                } else {
                    snprintf(load_end, sizeof(load_end), "%u", load_layer_end);
                }
                fprintf(stderr,
                        "ds4: SSD streaming initial %s model map restricted to "
                        "%snon-routed layers %u:%s (%u spans, %.2f GiB tensor span)\n",
                        ds4_backend_name(e->backend),
                        load_layer_start == 0 ? "token + " : "",
                        load_layer_start,
                        load_end,
                        spans.len,
                        (double)span_bytes / 1073741824.0);
            } else {
                fprintf(stderr,
                        "ds4: SSD streaming initial %s model map restricted to token embedding (%u spans, %.2f GiB tensor span)\n",
                        ds4_backend_name(e->backend),
                        spans.len,
                        (double)span_bytes / 1073741824.0);
            }
            model_map_ok = ds4_gpu_set_model_map_spans(e->model.map,
                                                        e->model.size,
                                                        load_offsets,
                                                        load_sizes,
                                                        load_span_count,
                                                        spans.max_tensor_bytes);
            free(spans.v);
        } else if (load_slice) {
            const bool map_output =
                load_output ||
                (load_output_optional &&
                 weights_have_output_head(&e->weights));
            char load_end[32];
            if (map_output && load_layer_end == UINT32_MAX) {
                snprintf(load_end, sizeof(load_end), "output");
            } else if (map_output) {
                snprintf(load_end, sizeof(load_end), "%u+output", load_layer_end);
            } else {
                snprintf(load_end, sizeof(load_end), "%u", load_layer_end);
            }

            ds4_model_map_span_vec spans;
            if (!weights_model_map_spans(&e->weights,
                                         load_layer_start,
                                         load_layer_end,
                                         map_output,
                                         &spans))
            {
                fprintf(stderr, "ds4: invalid model load layer slice %u:%s\n",
                        load_layer_start,
                        load_end);
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            uint64_t *offsets = xmalloc((size_t)spans.len * sizeof(offsets[0]));
            uint64_t *sizes = xmalloc((size_t)spans.len * sizeof(sizes[0]));
            uint64_t span_bytes = 0;
            for (uint32_t i = 0; i < spans.len; i++) {
                offsets[i] = spans.v[i].off;
                sizes[i] = spans.v[i].end - spans.v[i].off;
                span_bytes += sizes[i];
            }
            load_offsets = offsets;
            load_sizes = sizes;
            load_span_count = spans.len;
            e->startup_model_span_bytes = span_bytes;
            fprintf(stderr,
                    "ds4: restricting %s model map to layers %u:%s (%u spans, %.2f GiB tensor span)\n",
                    ds4_backend_name(e->backend),
                    load_layer_start,
                    load_end,
                    spans.len,
                    (double)span_bytes / 1073741824.0);
            model_map_ok = ds4_gpu_set_model_map_spans(e->model.map,
                                                        e->model.size,
                                                        load_offsets,
                                                        load_sizes,
                                                        load_span_count,
                                                        spans.max_tensor_bytes);
            free(spans.v);
        } else if (tp_shard) {
            ds4_model_map_span_vec spans;
            if (!weights_model_map_sharded_spans(&e->weights, &e->model,
                                                 tp_shard_rank, &spans)) {
                fprintf(stderr, "ds4: sharded model span build failed\n");
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            uint64_t *offsets = xmalloc((size_t)spans.len * sizeof(offsets[0]));
            uint64_t *sizes = xmalloc((size_t)spans.len * sizeof(sizes[0]));
            uint64_t span_bytes = 0;
            for (uint32_t i = 0; i < spans.len; i++) {
                offsets[i] = spans.v[i].off;
                sizes[i] = spans.v[i].end - spans.v[i].off;
                span_bytes += sizes[i];
            }
            load_offsets = offsets;
            load_sizes = sizes;
            load_span_count = spans.len;
            e->startup_model_span_bytes = span_bytes;
            fprintf(stderr,
                    "ds4: TP expert shard (rank %d): mapping %u spans, %.2f GiB of %.2f GiB\n",
                    tp_shard_rank,
                    spans.len,
                    (double)span_bytes / 1073741824.0,
                    (double)(e->model.size - e->model.tensor_data_pos) / 1073741824.0);
            model_map_ok = ds4_gpu_set_model_map_spans(e->model.map,
                                                        e->model.size,
                                                        load_offsets,
                                                        load_sizes,
                                                        load_span_count,
                                                        spans.max_tensor_bytes);
            free(spans.v);
        } else {
            e->startup_model_span_bytes = e->model.size > e->model.tensor_data_pos ?
                e->model.size - e->model.tensor_data_pos : 0;
            model_map_ok = ds4_gpu_set_model_map_range(e->model.map,
                                                       e->model.size,
                                                       e->model.tensor_data_pos,
                                                       e->model.size - e->model.tensor_data_pos,
                                                       e->model.max_tensor_bytes);
        }
        if (!model_map_ok) {
            fprintf(stderr,
                    "ds4: %s failed to map model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    ds4_backend_name(e->backend));
            free(load_offsets);
            free(load_sizes);
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (tp_shard) {
            model_warm_weights_sharded(&e->model, &e->weights,
                                       tp_shard_rank);
        }
        const bool support_model_runtime_ready =
            e->mtp_ready ||
            (e->support_kind == DS4_SUPPORT_DSPARK && e->dspark);
        if (support_model_runtime_ready &&
            !ds4_gpu_set_model_map_range(e->mtp_model.map,
                                           e->mtp_model.size,
                                           e->mtp_model.tensor_data_pos,
                                           e->mtp_model.size - e->mtp_model.tensor_data_pos,
                                           e->mtp_model.max_tensor_bytes))
        {
            fprintf(stderr,
                    "ds4: %s failed to map support model views; aborting startup. "
                    "This is commonly caused by insufficient memory or accelerator VM budget.\n",
                    ds4_backend_name(e->backend));
            free(load_offsets);
            free(load_sizes);
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        if (!ds4_engine_preload_pro_q4_expert_tables(e,
                                                     load_slice,
                                                     load_layer_start,
                                                     load_layer_end)) {
            free(load_offsets);
            free(load_sizes);
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        (void)ds4_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
        if (!accelerator_cache_model_tensors(e->backend, &e->model,
                                             load_offsets, load_sizes,
                                             load_span_count)) {
            fprintf(stderr, "ds4: %s failed to prepare optional model cache\n",
                    ds4_backend_name(e->backend));
            free(load_offsets);
            free(load_sizes);
            ds4_engine_close(e);
            *out = NULL;
            return 1;
        }
        free(load_offsets);
        free(load_sizes);
        /* Also apply explicit optional Q8 preload settings to the runtime
         * support model when loaded. */
        if (support_model_runtime_ready) {
            (void)ds4_gpu_set_model_fd_for_map(e->mtp_model.fd, e->mtp_model.map);
            if (!accelerator_cache_model_tensors(e->backend, &e->mtp_model,
                                                 NULL, NULL, 0)) {
                fprintf(stderr, "ds4: %s failed to prepare optional support model cache\n",
                        ds4_backend_name(e->backend));
                ds4_engine_close(e);
                *out = NULL;
                return 1;
            }
            (void)ds4_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
        }
        fprintf(stderr, "ds4: %s backend initialized for graph diagnostics\n",
                ds4_backend_name(e->backend));
    }
#else
    if (graph_backend) {
        fprintf(stderr, "ds4: %s backend requested but this build has no graph backend support; aborting startup\n",
                ds4_backend_name(e->backend));
        ds4_engine_close(e);
        *out = NULL;
        return 1;
    }
#endif

    if (!opt->inspect_only) {
        ds4_engine_print_startup_memory(e, opt->context_size);
    }
    *out = e;
    return 0;
}

void ds4_engine_summary(ds4_engine *e) {
    model_summary(&e->model);
    if (e->mtp_model.map) {
        printf("\nsupport model");
        if (e->support_kind != DS4_SUPPORT_NONE) {
            printf(" (%s", support_kind_name(e->support_kind));
            if (e->support_stages) printf(", stages=%u", e->support_stages);
            printf(")");
        }
        printf(":\n");
        model_summary(&e->mtp_model);
        if (e->support_kind == DS4_SUPPORT_DSPARK &&
            e->dspark_weights.n_stages != 0) {
            printf("support binding: tensors=%u missing=%u invalid=%u metadata_errors=%u\n",
                   e->dspark_weights.present_tensors,
                   e->dspark_weights.missing_tensors,
                   e->dspark_weights.invalid_tensors,
                   e->dspark_weights.metadata_errors);
        }
    }
}

int ds4_engine_vocab_size(ds4_engine *e) {
    return e ? e->vocab.n_vocab : 0;
}

uint32_t ds4_engine_prefill_chunk(ds4_engine *e) {
    return e ? e->prefill_chunk : 0;
}


int ds4_engine_power(ds4_engine *e) {
    return e ? e->power_percent : 100;
}

int ds4_engine_set_power(ds4_engine *e, int power_percent) {
    if (!e || power_percent < 1 || power_percent > 100) return 1;
    e->power_percent = power_percent;
    return 0;
}

const char *ds4_engine_model_name(ds4_engine *e) {
    (void)e;
    return DS4_MODEL_SHAPE_NAME;
}

int ds4_engine_layer_count(ds4_engine *e) {
    (void)e;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        if (DS4_N_LAYER <= DS4_N_NEXTN_PREDICT) return 0;
        return (int)(DS4_N_LAYER - DS4_N_NEXTN_PREDICT);
    }
    return (int)DS4_N_LAYER;
}

uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer) {
    (void)e;
    if (layer >= DS4_N_LAYER) return 0;
    return ds4_layer_compress_ratio(layer);
}

uint64_t ds4_engine_hidden_f32_values(ds4_engine *e) {
    (void)e;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) return DS4_N_EMBD;
    return (uint64_t)DS4_N_HC * DS4_N_EMBD;
}

bool ds4_engine_glm_layer_payload_bytes(ds4_engine *e,
                                        uint32_t layer,
                                        uint32_t full_live,
                                        uint32_t key_dim,
                                        uint32_t value_dim,
                                        uint32_t compact_live,
                                        uint32_t index_live,
                                        uint64_t *out) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)layer;
    (void)full_live;
    (void)key_dim;
    (void)value_dim;
    (void)compact_live;
    (void)index_live;
    (void)out;
    return false;
#else
    (void)e;
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM_DSA) return false;
    return glm_layer_payload_tensor_bytes(layer,
                                          full_live,
                                          key_dim,
                                          value_dim,
                                          compact_live,
                                          index_live,
                                          out);
#endif
}

int ds4_engine_model_id(ds4_engine *e) {
    (void)e;
    return (int)DS4_MODEL_VARIANT;
}

