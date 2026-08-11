#ifndef DS4_NO_GPU
/* =========================================================================
 * Metal Release Graph Allocation.
 * ========================================================================= */

/* Allocate the Metal graph state for a chosen raw-cache capacity.  The model
 * weights are not copied here; tensors reference the mapped GGUF.
 *
 * tier-aware per-layer allocation.
 *   placement: when non-NULL, an array of DS4_N_LAYER + 2 logical tiers
 *     (embedding, per-layer..., head). The per-layer KV / state allocations
 *     in this function use placement[il + 1] as the home tier for each
 *     layer il. When NULL (single-tier callers, diagnostic paths), all
 *     per-layer allocations land on tier 0 — byte-equivalent to legacy.
 *
 * Single-tier (g_n_gpus <= 1) is byte-equivalent regardless of placement,
 * because metal_graph_alloc_kv_cache_tensor_on short-circuits to the
 * legacy 1-arg helpers when g_n_gpus <= 1. */
static bool metal_graph_alloc_raw_cap(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_mtp,
        const int              *placement,
        bool                    cuda_tensor_parallel,
        const ds4_gpu_graph    *shared_prefill_workspace) {
    const int saved_dspark_exec_tier = g->dspark_exec_tier;
    memset(g, 0, sizeof(*g));
    g->dspark_exec_tier = saved_dspark_exec_tier;
    g->owns_prefill_workspace = shared_prefill_workspace == NULL;
    g->cpu_router_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(g->cpu_router_norm[0]));
    g->active_tier = placement ? -1 : 0;
    /* cache placement on the graph so the dispatch loops can
     * walk it without threading the engine pointer through every
     * kernel-dispatch wrapper. NULL in single-tier callers (placement
     * was already NULL on entry). */
    g->placement = placement;
    g->cuda_tp_decode = placement && cuda_tensor_parallel;
    g->cuda_tp_attn = g->cuda_tp_decode && metal_graph_cuda_tp_attn_requested();
    g->cuda_tp_attn_peer_read = metal_graph_cuda_tp_attn_peer_read_requested();
    g->cuda_tp_attn_heads = g->cuda_tp_decode && metal_graph_cuda_tp_attn_heads_requested();
    g->cuda_tp_attn_cache_dup = g->cuda_tp_attn_heads &&
                                metal_graph_cuda_tp_attn_cache_dup_requested();
    g->cuda_tp_moe = g->cuda_tp_decode && metal_graph_cuda_tp_moe_requested();
    g->cuda_tp_ep = g->cuda_tp_moe && cuda_tensor_parallel;
    g->cuda_tp_ep_pack_exact =
        g->cuda_tp_ep && metal_graph_cuda_tp_ep_pack_exact_requested();
    g->cuda_tp_moe_delay_reduce = metal_graph_cuda_tp_moe_delay_reduce_requested();
    g->cuda_tp_moe_copy3_handoff = metal_graph_cuda_tp_moe_copy3_handoff_requested();
    g->cuda_tp_moe_pack_handoff = metal_graph_cuda_tp_moe_pack_handoff_requested();
    g->cuda_tp_moe_peer_read = metal_graph_cuda_tp_moe_peer_read_requested();
    g->cuda_tp_moe_peer_router = metal_graph_cuda_tp_moe_peer_router_requested();
    g->cuda_tp_shared = g->cuda_tp_decode && metal_graph_cuda_tp_shared_requested();
    g->cuda_tp_shared_fold = metal_graph_cuda_tp_shared_fold_requested();
    g->cuda_tp_q = g->cuda_tp_decode && metal_graph_cuda_tp_q_requested();
    g->cuda_tp_output = g->cuda_tp_decode && metal_graph_cuda_tp_output_requested();
    g->cuda_tp_prefill_ffn = g->cuda_tp_decode && metal_graph_cuda_tp_prefill_ffn_requested();
    g->cuda_tp_prefill_attn_output =
        g->cuda_tp_decode && metal_graph_cuda_tp_prefill_attn_output_requested();
    g->cuda_q_norm_rope_fuse = metal_graph_cuda_q_norm_rope_fuse_requested();
    g->cuda_qkv_kv_rope_fuse = metal_graph_cuda_qkv_kv_rope_fuse_requested();
    g->cuda_qkv_pair = getenv("DS4_CUDA_NO_QKV_PAIR") == NULL;
    g->cuda_tp_attn_out_hc_fuse =
        getenv("DS4_CUDA_TP_ATTN_OUT_HC_FUSE") != NULL &&
        getenv("DS4_CUDA_NO_TP_ATTN_OUT_HC_FUSE") == NULL;
    g->shared_gate_up_swiglu_fuse =
        getenv("DS4_METAL_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION") == NULL;
    g->decode_stage_profile = getenv("DS4_METAL_DECODE_STAGE_PROFILE") != NULL;
    g->decode_index_stage_profile = getenv("DS4_METAL_INDEXER_STAGE_PROFILE") != NULL;
    g->output_stage_profile = getenv("DS4_METAL_OUTPUT_STAGE_PROFILE") != NULL;
    const bool enable_splitkv_spec = metal_graph_cuda_splitkv_spec_requested();
    const bool enable_splitkv_batch_verify =
        enable_splitkv_spec && metal_graph_cuda_splitkv_spec_batch_verify_requested();
    const bool enable_spec_logits = enable_mtp || enable_splitkv_batch_verify;
    const bool enable_prefix1_snapshot = enable_mtp || enable_splitkv_spec;
    const bool enable_frontier_snapshot =
        enable_mtp ||
        enable_splitkv_spec ||
        (metal_graph_cuda_greedy_splitkv_requested() &&
         metal_graph_cuda_greedy_splitkv_fallback_requested()) ||
        (metal_graph_cuda_greedy_vec4_requested() &&
         metal_graph_cuda_greedy_vec4_fallback_requested());
    if (g->cuda_tp_decode && metal_graph_cuda_tp_partner_tier(0) < 0) {
        fprintf(stderr,
                "ds4: CUDA tensor parallelism requires an even multi-GPU placement; "
                "have %d GPU tiers\n",
                g_n_gpus);
        metal_graph_free(g);
        return false;
    }
    if (g->cuda_tp_ep &&
        (g_ds4_shape.family != DS4_MODEL_FAMILY_DEEPSEEK4 ||
         (DS4_N_EXPERT & 1u) != 0u)) {
        fprintf(stderr,
                "ds4: CUDA tensor parallelism requires an even-expert DeepSeek model\n");
        metal_graph_free(g);
        return false;
    }
    if (g->cuda_tp_ep) {
        fprintf(stderr,
                "ds4: CUDA routed MoE expert ownership enabled "
                "(half-resident decode and prefill)\n");
    }
    g->mtp_enabled = enable_mtp;
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (raw_cap < raw_window) raw_cap = raw_window;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    g->raw_cap = raw_cap;
    g->raw_window = raw_window;
    g->prefill_cap = prefill_cap;
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!weights_layer_has_required(&weights->layer[il], il)) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    g->comp_cap = ctx_size / min_ratio + 2u;
    if (g->comp_cap < 2u) g->comp_cap = 2u;
    g->attn_comp_stage_cap = prefill_cap / min_ratio + 2u;
    if (g->attn_comp_stage_cap < 2u) g->attn_comp_stage_cap = 2u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!weights_layer_has_required(&weights->layer[il], il)) {
            g->layer_comp_cap[il] = 0;
            continue;
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) {
            g->layer_comp_cap[il] = 0;
        } else {
            g->layer_comp_cap[il] = ctx_size / ratio + 2u;
            if (g->layer_comp_cap[il] < 2u) g->layer_comp_cap[il] = 2u;
        }
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim = (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t routed_mid_dim = layer->ffn_gate_exps->dim[1];
    /* Distributed coordinators do not normally own the output head. The
     * logits workspace still has a fixed model-vocabulary shape, while the
     * actual head is encoded only on a node that bound its tensors. */
    const uint64_t vocab_dim =
        weights->output ? weights->output->dim[1] : DS4_N_VOCAB;
    const uint64_t comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t pc = prefill_cap;
    uint64_t kv_cache_bytes = 0;
    const uint64_t context_bytes =
        metal_graph_context_bytes_for_kv_policy(ctx_size, raw_cap, prefill_cap, &kv_cache_bytes);
    const bool managed_kv_cache =
        ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes, context_bytes) != 0;
    if (managed_kv_cache) {
        /*
         * CUDA device allocations are fastest, but a million-token KV cache is
         * large enough to starve DGX Spark's unified CPU/GPU memory once the
         * model cache and driver allocations are present.  For this one
         * long-lived cache class, managed memory restores the old demand-paged
         * behavior.  It can be slower, but it keeps oversized contexts from
         * turning memory pressure into a machine-wide lockup.
         */
        fprintf(stderr,
                "ds4: CUDA using managed KV cache for ctx=%u "
                "(kv cache %.2f GiB, context buffers %.2f GiB); "
                "this may degrade performance but is needed for very large contexts\n",
                ctx_size,
                (double)kv_cache_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0);
    }

    /* Class P decode HC scratch — replicated across every tier
     * the placement uses (per-tier kernel-scratch). Single-tier path
     * (placement == NULL) collapses to tier 0 only; _ptr_on(0, ...) short-
     * circuits to legacy ds4_gpu_tensor_alloc when g_n_gpus <= 1 — byte-
     * equivalent. The hc_pre/hc_post/hc_comb buffers are VIEWS of hc_split
     * and therefore allocated per tier alongside their parent. */
    bool used_tier[DS4_MAX_GPUS] = {0};
    used_tier[0] = true; /* single-tier baseline always uses tier 0 */
    if (placement) {
        for (uint32_t i = 0; i < (uint32_t)DS4_N_LAYER + 2u; i++) {
            const int p = placement[i];
            if (p >= 0 && p < DS4_MAX_GPUS) used_tier[p] = true;
        }
    }
    if (g->cuda_tp_decode) {
        const int half = g_n_gpus / 2;
        for (int t = half; t < g_n_gpus; t++) {
            if (used_tier[t]) {
                fprintf(stderr,
                        "ds4: CUDA tensor parallelism expects layer homes in lower-half "
                        "tiers; placement already uses tier %d\n",
                        t);
                metal_graph_free(g);
                return false;
            }
        }
        for (int t = 0; t < half; t++) {
            if (used_tier[t]) used_tier[t + half] = true;
        }
        fprintf(stderr,
                "ds4: CUDA decode TP enabled: pairing lower-half tiers with "
                "upper-half tiers\n");
    }
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        if (!used_tier[t]) continue;
        g->cur_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, hc_dim * sizeof(float));
        g->flat_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, hc_dim * sizeof(float));
        g->hc_mix_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, mix_hc * sizeof(float));
        g->hc_split_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, mix_hc * sizeof(float));
        g->hc_pre_by_tier[t] = ds4_gpu_tensor_view(g->hc_split_by_tier[t],
                                                    0,
                                                    (uint64_t)DS4_N_HC * sizeof(float));
        g->hc_post_by_tier[t] = ds4_gpu_tensor_view(g->hc_split_by_tier[t],
                                                     (uint64_t)DS4_N_HC * sizeof(float),
                                                     (uint64_t)DS4_N_HC * sizeof(float));
        g->hc_comb_by_tier[t] = ds4_gpu_tensor_view(g->hc_split_by_tier[t],
                                                     2ull * DS4_N_HC * sizeof(float),
                                                     (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float));
        g->attn_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->attn_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->qr_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, q_rank * sizeof(float));
        g->qr_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, q_rank * sizeof(float));
        g->q_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, q_dim * sizeof(float));
        g->kv_raw_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        g->kv_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    }
    bool state_init_ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        /* A distributed process owns only its bound layer slice. Persistent
         * KV state must follow that ownership just like the model tensors;
         * allocating every model layer here defeats split-model residency. */
        if (!weights_layer_has_required(&weights->layer[il], il)) continue;
        /* per-layer Class L allocations land on the layer's
         * home tier. placement is NULL on single-tier / diagnostic paths
         * (all-tier-0); non-NULL on the engine path that opted into
         * multi-tier. layer_tier == 0 in single-tier mode is the
         * byte-equivalent path through metal_graph_alloc_kv_cache_tensor_on
         * and ds4_gpu_tensor_alloc_ptr_on. */
        const int layer_tier = placement ? placement[il + 1] : 0;
        g->layer_raw_cache[il] = metal_graph_alloc_kv_cache_tensor_on(
                managed_kv_cache,
                layer_tier,
                (uint64_t)raw_cap * DS4_SPARK_KV_ROW_BYTES);
        const int layer_tp_partner = g->cuda_tp_attn_cache_dup
            ? metal_graph_cuda_tp_partner_tier(layer_tier) : -1;
        if (layer_tp_partner >= 0) {
            g->layer_raw_cache_tp[il] = metal_graph_alloc_kv_cache_tensor_on(
                    managed_kv_cache,
                    layer_tp_partner,
                    (uint64_t)raw_cap * DS4_SPARK_KV_ROW_BYTES);
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            const uint64_t attn_rows = (uint64_t)coff * ratio;
            g->layer_attn_comp_cache[il] = metal_graph_alloc_kv_cache_tensor_on(
                    managed_kv_cache,
                    layer_tier,
                    (uint64_t)g->layer_comp_cap[il] * DS4_SPARK_KV_ROW_BYTES);
            if (layer_tp_partner >= 0) {
                g->layer_attn_comp_cache_tp[il] = metal_graph_alloc_kv_cache_tensor_on(
                        managed_kv_cache,
                        layer_tp_partner,
                        (uint64_t)g->layer_comp_cap[il] * DS4_SPARK_KV_ROW_BYTES);
            }
            g->layer_attn_state_kv[il] = ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
            g->layer_attn_state_score[il] = ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
            if (enable_frontier_snapshot) {
                g->spec_attn_state_kv[il] =
                    ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
                g->spec_attn_state_score[il] =
                    ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
                if (enable_prefix1_snapshot) {
                    g->spec_prefix1_attn_state_kv[il] =
                        ds4_gpu_tensor_alloc_ptr_on(
                                layer_tier,
                                DS4_SPEC_PREFIX_SLOTS * attn_width * attn_rows *
                                    sizeof(float));
                    g->spec_prefix1_attn_state_score[il] =
                        ds4_gpu_tensor_alloc_ptr_on(
                                layer_tier,
                                DS4_SPEC_PREFIX_SLOTS * attn_width * attn_rows *
                                    sizeof(float));
                }
            }
            if (g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4) {
                const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                g->layer_index_comp_cache[il] = metal_graph_alloc_kv_cache_tensor_on(
                        managed_kv_cache,
                        layer_tier,
                        (uint64_t)g->layer_comp_cap[il] * DS4_SPARK_INDEX_ROW_BYTES);
                g->layer_index_state_kv[il] = ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                g->layer_index_state_score[il] = ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                if (enable_frontier_snapshot) {
                    g->spec_index_state_kv[il] =
                        ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                    g->spec_index_state_score[il] =
                        ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                    if (enable_prefix1_snapshot) {
                        g->spec_prefix1_index_state_kv[il] =
                            ds4_gpu_tensor_alloc_ptr_on(
                                    layer_tier,
                                    DS4_SPEC_PREFIX_SLOTS * index_width * index_rows *
                                        sizeof(float));
                        g->spec_prefix1_index_state_score[il] =
                            ds4_gpu_tensor_alloc_ptr_on(
                                    layer_tier,
                                    DS4_SPEC_PREFIX_SLOTS * index_width * index_rows *
                                        sizeof(float));
                    }
                }
                if (g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows);
                }
            }
        }
    }
    /* Class P per-layer decode scratch + routed-expert state —
     * replicated across every used tier. ffn_out is lazily allocated by
     * metal_graph_ensure_ffn_out (per-tier on first touch). */
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        if (!used_tier[t]) continue;
        g->comp_kv_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, comp_width_max * sizeof(float));
        g->comp_sc_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, comp_width_max * sizeof(float));
        /* Decode-only scratch for the quad compressor projection: the indexer
         * pair outputs must not alias the attention compressor outputs inside
         * the single fused dispatch. */
        g->index_comp_kv_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, 2ull * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
        g->index_comp_sc_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, 2ull * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
        g->attn_comp_stage_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)g->attn_comp_stage_cap * DS4_N_HEAD_DIM * sizeof(float));
        g->index_comp_stage_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)g->attn_comp_stage_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
        g->indexer_q_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, indexer_q_dim * sizeof(float));
        g->indexer_weights_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_INDEXER_HEAD * sizeof(float));
        g->indexer_scores_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)g->comp_cap * pc * sizeof(float));
        g->comp_mask_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)g->comp_cap * pc * sizeof(float));
        g->comp_selected_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) * pc * sizeof(uint32_t));
        g->heads_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, q_dim * sizeof(float));
        g->attn_low_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, low_dim * sizeof(float));
        g->attn_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->after_attn_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, hc_dim * sizeof(float));
        g->ffn_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->ffn_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->shared_gate_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, shared_dim * sizeof(float));
        g->shared_up_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, shared_dim * sizeof(float));
        g->shared_mid_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, shared_dim * sizeof(float));
        g->shared_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->router_logits_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, DS4_N_EXPERT * sizeof(float));
        g->router_probs_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, DS4_N_EXPERT * sizeof(float));
        g->router_selected_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, DS4_N_EXPERT_USED * sizeof(int));
        g->router_weights_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, DS4_N_EXPERT_USED * sizeof(float));
        g->routed_gate_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
        g->routed_up_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
        g->routed_mid_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
        g->routed_down_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
        g->routed_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        if (g->cuda_tp_decode) {
            g->tp_peer_tmp_by_tier[t] =
                ds4_gpu_tensor_alloc_ptr_on(t, DS4_CUDA_TP_PEER_TMP_BYTES);
        }
        g->after_ffn_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, hc_dim * sizeof(float));
    }
    /* Class H — head_tier captured from placement[DS4_N_LAYER + 1]
     * (or 0 in single-tier / diagnostic paths). Output-head tensors and the
     * final logits buffer allocate on head_tier only; other tier slots stay
     * NULL. The _ptr_on(0, ...) path short-circuits to the legacy
     * ds4_gpu_tensor_alloc when g_n_gpus <= 1 — byte-equivalent. */
    g->head_tier = placement ? placement[DS4_N_LAYER + 1] : 0;
    int output_tp_tiers[DS4_MAX_GPUS] = {0};
    const uint32_t output_tp_ways = g->cuda_tp_output
        ? metal_graph_cuda_tp_output_tiers(g, output_tp_tiers) : 0;
    if (g->cuda_tp_output && output_tp_ways < 2u) {
        fprintf(stderr,
                "ds4: CUDA output TP requires output head tier %d to be in "
                "the lower half of the CUDA placement\n",
                g->head_tier);
        metal_graph_free(g);
        return false;
    }
    uint64_t output_logits_elems = vocab_dim;
    if (enable_spec_logits && output_tp_ways >= 2u) {
        const uint64_t max_shard_vocab =
            (vocab_dim + output_tp_ways - 1u) / output_tp_ways;
        const uint64_t spec_shard_elems =
            (uint64_t)DS4_DSPARK_MAX_BLOCK_SIZE * max_shard_vocab;
        if (spec_shard_elems > output_logits_elems) {
            output_logits_elems = spec_shard_elems;
        }
    }
    g->output_pre_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier, (uint64_t)DS4_N_HC * sizeof(float));
    g->output_weights_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier, (uint64_t)DS4_N_HC * sizeof(float));
    g->output_embd_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier, (uint64_t)DS4_N_EMBD * sizeof(float));
    g->output_norm_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier, (uint64_t)DS4_N_EMBD * sizeof(float));
    g->logits_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier,
                                    output_logits_elems * sizeof(float));
    for (uint32_t i = 1; i < output_tp_ways; i++) {
        const int t = output_tp_tiers[i];
        if (t < 0 || t >= DS4_MAX_GPUS || t == g->head_tier) continue;
        g->output_norm_by_tier[t] =
            ds4_gpu_tensor_alloc_ptr_on(t,
                                        (uint64_t)DS4_N_EMBD * sizeof(float));
        g->logits_by_tier[t] =
            ds4_gpu_tensor_alloc_ptr_on(t,
                                        output_logits_elems * sizeof(float));
    }
    /*
     * MTP is deliberately outside the normal graph footprint.  A session that
     * does not opt in with --mtp must allocate and execute exactly the same
     * buffers as the plain decoder: no support-model mapping, no draft logits,
     * and no MTP scratch hidden behind otherwise unused tensors.
     */
    if (enable_mtp) {
        g->mtp_embed = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_enorm = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj = ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
        g->mtp_eproj_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hnorm_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_hproj_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_input_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_state_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_next_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->mtp_raw_cache = metal_graph_alloc_kv_cache_tensor(
                managed_kv_cache,
                (uint64_t)raw_cap * DS4_N_HEAD_DIM * sizeof(float));
        g->mtp_n_raw = 0;
    }
    if (enable_spec_logits) {
        const int spec_tier =
            g->dspark_exec_tier > 0 && g->dspark_exec_tier < DS4_MAX_GPUS
                ? g->dspark_exec_tier : 0;
        g->spec_logits = spec_tier
            ? ds4_gpu_tensor_alloc_ptr_on(spec_tier, (uint64_t)16 * DS4_N_VOCAB * sizeof(float))
            : ds4_gpu_tensor_alloc((uint64_t)16 * DS4_N_VOCAB * sizeof(float));
    }

    /* Class E — emb_tier captured from placement[0] (or 0 in
     * single-tier / diagnostic paths). _ptr_on(0, ...) short-circuits to the
     * legacy ds4_gpu_tensor_alloc when g_n_gpus <= 1 — byte-equivalent. */
    g->emb_tier = placement ? placement[0] : 0;
    /* Class P chunked-prefill batch scratch — replicated across
     * every used tier. The cur/next pair (batch_cur_hc / batch_next_hc) is
     * ping-ponged per layer step on each tier; tier transitions copy via
     * ds4_gpu_tensor_copy_xdev (handled in B6). batch_ffn_out is lazily
     * allocated by metal_graph_ensure_batch_ffn_out (per-tier on first touch)
     * and included in the CUDA scratch estimate because TP prefill can use it
     * as the combined routed+shared FFN buffer. */
    if (shared_prefill_workspace) {
        if (shared_prefill_workspace->prefill_cap < prefill_cap ||
            shared_prefill_workspace->emb_tier != g->emb_tier) {
            fprintf(stderr,
                    "ds4: shared prefill workspace is incompatible "
                    "(capacity %u/%u, embedding tier %d/%d)\n",
                    shared_prefill_workspace->prefill_cap,
                    prefill_cap,
                    shared_prefill_workspace->emb_tier,
                    g->emb_tier);
        } else {
            metal_graph_copy_prefill_workspace_pointers(
                    g, shared_prefill_workspace);
        }
    } else {
        g->prefill_tokens_by_tier[g->emb_tier] =
            ds4_gpu_tensor_alloc_ptr_on(g->emb_tier, pc * sizeof(int32_t));
        for (int t = 0; t < DS4_MAX_GPUS; t++) {
            if (!used_tier[t]) continue;
            g->batch_cur_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * hc_dim * sizeof(float));
            g->batch_next_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * hc_dim * sizeof(float));
            g->batch_flat_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * hc_dim * sizeof(float));
            g->batch_hc_mix_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * mix_hc * sizeof(float));
            g->batch_hc_split_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * mix_hc * sizeof(float));
            g->batch_attn_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_attn_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_qr_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * q_rank * sizeof(float));
            g->batch_qr_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * q_rank * sizeof(float));
            g->batch_q_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * q_dim * sizeof(float));
            g->batch_kv_raw_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_HEAD_DIM * sizeof(float));
            g->batch_kv_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_HEAD_DIM * sizeof(float));
            g->batch_comp_kv_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * comp_width_max * sizeof(float));
            g->batch_comp_sc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * comp_width_max * sizeof(float));
            g->batch_indexer_q_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * indexer_q_dim * sizeof(float));
            g->batch_indexer_weights_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_INDEXER_HEAD * sizeof(float));
            g->batch_heads_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * q_dim * sizeof(float));
            g->batch_attn_low_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * low_dim * sizeof(float));
            g->batch_attn_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_group_tmp_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * group_dim * sizeof(float));
            g->batch_low_tmp_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_LORA_O * sizeof(float));
            g->batch_after_attn_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * hc_dim * sizeof(float));
            g->batch_ffn_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_ffn_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_shared_gate_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * shared_dim * sizeof(float));
            g->batch_shared_up_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * shared_dim * sizeof(float));
            g->batch_shared_mid_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * shared_dim * sizeof(float));
            g->batch_shared_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_router_logits_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT * sizeof(float));
            g->batch_router_probs_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT * sizeof(float));
            g->batch_router_selected_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * sizeof(int));
            g->batch_router_weights_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * sizeof(float));
            g->batch_routed_gate_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
            g->batch_routed_up_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
            g->batch_routed_mid_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
            g->batch_routed_down_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
            g->batch_routed_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
        }
        if (DS4_GPU_ATTN_COMP_CACHE_F16) {
            g->batch_q_half = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(uint16_t));
        }
        g->prefill_seed_router_selected = ds4_gpu_tensor_alloc(
                (uint64_t)DS4_N_LAYER * DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                DS4_N_EXPERT_USED * sizeof(int32_t));
    }

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < DS4_N_LAYER; il++) {
        if (!weights_layer_has_required(&weights->layer[il], il)) continue;
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        if (layer_cache_ok && g->cuda_tp_attn_cache_dup) {
            layer_cache_ok = g->layer_raw_cache_tp[il] != NULL;
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (layer_cache_ok && ratio != 0) {
            layer_cache_ok = g->layer_attn_comp_cache[il] != NULL &&
                             (!g->cuda_tp_attn_cache_dup ||
                              g->layer_attn_comp_cache_tp[il] != NULL) &&
                             g->layer_attn_state_kv[il] != NULL &&
                             g->layer_attn_state_score[il] != NULL &&
                             (!enable_frontier_snapshot ||
                              (g->spec_attn_state_kv[il] != NULL &&
                               g->spec_attn_state_score[il] != NULL)) &&
                             (!enable_prefix1_snapshot ||
                              (g->spec_prefix1_attn_state_kv[il] != NULL &&
                               g->spec_prefix1_attn_state_score[il] != NULL));
        }
        if (layer_cache_ok && ratio == 4) {
            layer_cache_ok = g->layer_index_comp_cache[il] != NULL &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL &&
                             (!enable_frontier_snapshot ||
                              (g->spec_index_state_kv[il] != NULL &&
                               g->spec_index_state_score[il] != NULL)) &&
                             (!enable_prefix1_snapshot ||
                              (g->spec_prefix1_index_state_kv[il] != NULL &&
                               g->spec_prefix1_index_state_score[il] != NULL));
        }
    }

    /* Class P validation — check every used tier's slot. */
    bool class_p_ok = true;
    for (int t = 0; class_p_ok && t < DS4_MAX_GPUS; t++) {
        if (!used_tier[t]) continue;
        class_p_ok =
            g->cur_hc_by_tier[t] && g->flat_hc_by_tier[t] && g->hc_mix_by_tier[t] && g->hc_split_by_tier[t] &&
            g->hc_pre_by_tier[t] && g->hc_post_by_tier[t] && g->hc_comb_by_tier[t] &&
            g->attn_cur_by_tier[t] && g->attn_norm_by_tier[t] && g->qr_by_tier[t] && g->qr_norm_by_tier[t] &&
            g->q_by_tier[t] && g->kv_raw_by_tier[t] && g->kv_by_tier[t] &&
            g->comp_kv_cur_by_tier[t] && g->comp_sc_cur_by_tier[t] &&
            g->index_comp_kv_cur_by_tier[t] && g->index_comp_sc_cur_by_tier[t] &&
            g->attn_comp_stage_by_tier[t] && g->index_comp_stage_by_tier[t] &&
            g->indexer_q_by_tier[t] && g->indexer_weights_by_tier[t] && g->indexer_scores_by_tier[t] &&
            g->comp_mask_by_tier[t] && g->comp_selected_by_tier[t] &&
            g->heads_by_tier[t] && g->attn_low_by_tier[t] && g->attn_out_by_tier[t] &&
            g->after_attn_hc_by_tier[t] && g->ffn_cur_by_tier[t] && g->ffn_norm_by_tier[t] &&
            g->shared_gate_by_tier[t] && g->shared_up_by_tier[t] && g->shared_mid_by_tier[t] &&
            g->shared_out_by_tier[t] &&
            g->router_logits_by_tier[t] && g->router_probs_by_tier[t] &&
            g->router_selected_by_tier[t] && g->router_weights_by_tier[t] &&
            g->routed_gate_by_tier[t] && g->routed_up_by_tier[t] && g->routed_mid_by_tier[t] &&
            g->routed_down_by_tier[t] && g->routed_out_by_tier[t] &&
            (!g->cuda_tp_decode || g->tp_peer_tmp_by_tier[t]) &&
            g->after_ffn_hc_by_tier[t] &&
            g->batch_cur_hc_by_tier[t] && g->batch_next_hc_by_tier[t] && g->batch_flat_hc_by_tier[t] &&
            g->batch_hc_mix_by_tier[t] && g->batch_hc_split_by_tier[t] &&
            g->batch_attn_cur_by_tier[t] && g->batch_attn_norm_by_tier[t] &&
            g->batch_qr_by_tier[t] && g->batch_qr_norm_by_tier[t] && g->batch_q_by_tier[t] &&
            g->batch_kv_raw_by_tier[t] && g->batch_kv_by_tier[t] &&
            g->batch_comp_kv_by_tier[t] && g->batch_comp_sc_by_tier[t] &&
            g->batch_indexer_q_by_tier[t] && g->batch_indexer_weights_by_tier[t] &&
            g->batch_heads_by_tier[t] && g->batch_attn_low_by_tier[t] && g->batch_attn_out_by_tier[t] &&
            g->batch_group_tmp_by_tier[t] && g->batch_low_tmp_by_tier[t] && g->batch_after_attn_hc_by_tier[t] &&
            g->batch_ffn_cur_by_tier[t] && g->batch_ffn_norm_by_tier[t] &&
            g->batch_shared_gate_by_tier[t] && g->batch_shared_up_by_tier[t] &&
            g->batch_shared_mid_by_tier[t] && g->batch_shared_out_by_tier[t] &&
            g->batch_router_logits_by_tier[t] && g->batch_router_probs_by_tier[t] &&
            g->batch_router_selected_by_tier[t] && g->batch_router_weights_by_tier[t] &&
            g->batch_routed_gate_by_tier[t] && g->batch_routed_up_by_tier[t] &&
            g->batch_routed_mid_by_tier[t] && g->batch_routed_down_by_tier[t] &&
            g->batch_routed_out_by_tier[t];
    }
    bool output_tp_ok = true;
    for (uint32_t i = 1; i < output_tp_ways; i++) {
        const int t = output_tp_tiers[i];
        if (t < 0 || t >= DS4_MAX_GPUS || t == g->head_tier) continue;
        output_tp_ok = output_tp_ok &&
                       g->output_norm_by_tier[t] != NULL &&
                       g->logits_by_tier[t] != NULL;
    }
    const bool ok = state_init_ok && layer_cache_ok && class_p_ok &&
                    /* Class H — validate the head_tier slot
                     * (single-tier: head_tier == 0, byte-equivalent). */
                    metal_graph_output_pre(g) && metal_graph_output_weights(g) &&
                    metal_graph_output_embd(g) && metal_graph_output_norm(g) &&
                    metal_graph_logits(g) && output_tp_ok &&
                    (!enable_mtp ||
                     (g->mtp_embed && g->mtp_enorm && g->mtp_eproj &&
                      g->mtp_eproj_hc && g->mtp_hnorm_hc && g->mtp_hproj_hc &&
                      g->mtp_input_hc && g->mtp_state_hc && g->mtp_next_hc &&
                      g->mtp_raw_cache)) &&
                    (!enable_spec_logits || g->spec_logits) &&
                    /* Class E — validate the emb_tier slot. */
                    metal_graph_prefill_tokens(g) &&
                    g->cpu_router_norm &&
                    (!DS4_GPU_ATTN_COMP_CACHE_F16 || g->batch_q_half) &&
                    g->prefill_seed_router_selected;
    if (!ok) metal_graph_free(g);
    return ok;
}

static bool metal_graph_alloc(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer) {
    /* single-tier convenience wrapper; placement=NULL routes
     * all per-layer allocations to tier 0. */
    return metal_graph_alloc_raw_cap(g, weights, layer, DS4_N_SWA, DS4_N_SWA,
                                     1, false, NULL, false, NULL);
}

static bool metal_graph_install_model_spans(
        const ds4_model              *model,
        const ds4_model_map_span_vec *spans,
        const char                   *label) {
    if (!model || !spans || spans->len == 0) return false;

    uint64_t *offsets = xmalloc((size_t)spans->len * sizeof(offsets[0]));
    uint64_t *sizes = xmalloc((size_t)spans->len * sizeof(sizes[0]));
    for (uint32_t i = 0; i < spans->len; i++) {
        offsets[i] = spans->v[i].off;
        sizes[i] = spans->v[i].end - spans->v[i].off;
    }

    const bool ok = ds4_gpu_set_model_map_spans(model->map,
                                                model->size,
                                                offsets,
                                                sizes,
                                                spans->len,
                                                spans->max_tensor_bytes) != 0;
    if (!ok) {
        fprintf(stderr,
                "ds4: Metal SSD streaming failed to map %s model spans\n",
                label ? label : "requested");
    }
    free(offsets);
    free(sizes);
    return ok;
}

static bool metal_graph_stream_readahead_enabled(void) {
    return glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_READAHEAD",
                                 "DS4_METAL_ENABLE_STREAMING_READAHEAD") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_READAHEAD",
                                  "DS4_METAL_DISABLE_STREAMING_READAHEAD");
}

static bool metal_graph_stream_madvise_willneed_enabled(void) {
    return glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_MADVISE_WILLNEED",
                                 "DS4_METAL_ENABLE_STREAMING_MADVISE_WILLNEED") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_MADVISE_WILLNEED",
                                  "DS4_METAL_DISABLE_STREAMING_MADVISE_WILLNEED");
}

static bool metal_graph_stream_decode_static_map_enabled(void) {
    if (glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_STATIC_DECODE_MAP",
                              "DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP")) {
        return false;
    }
#ifdef DS4_ROCM_BUILD
    return glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_STATIC_DECODE_MAP",
                                 "DS4_METAL_ENABLE_STREAMING_STATIC_DECODE_MAP");
#else
    return true;
#endif
}

static bool metal_graph_stream_decode_static_map_state_cache_enabled(void) {
    return !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_STATIC_MAP_STATE_CACHE",
                                  "DS4_METAL_DISABLE_STREAMING_STATIC_MAP_STATE_CACHE");
}

static bool metal_graph_stream_decode_layer_batch_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           !g_expert_profile.active &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_LAYER_BATCH",
                                  "DS4_METAL_DISABLE_STREAMING_LAYER_BATCH") &&
           (!glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE",
                                   "DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE") ||
            glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_FULL_EXPERT_ADDR_TABLE",
                                  "DS4_METAL_DISABLE_STREAMING_FULL_EXPERT_ADDR_TABLE")) &&
           !glm_graph_env_present("DS4_ROCM_DECODE_STAGE_PROFILE",
                                  "DS4_METAL_DECODE_STAGE_PROFILE") &&
           !glm_graph_env_present("DS4_ROCM_GRAPH_DUMP_PREFIX",
                                  "DS4_METAL_GRAPH_DUMP_PREFIX");
}

static void metal_graph_stream_readahead_range_impl(
        const ds4_model *model,
        uint64_t         offset,
        uint64_t         size,
        bool             enabled) {
    if (!enabled ||
        !model ||
        model->fd < 0 ||
        !model->map ||
        offset > model->size ||
        size == 0 ||
        size > model->size - offset) {
        return;
    }

#if defined(F_RDADVISE)
    uint64_t pos = offset;
    uint64_t rem = size;
    while (rem > 0) {
        const uint64_t chunk64 =
            rem > (uint64_t)INT_MAX ? (uint64_t)INT_MAX : rem;
        if (pos > (uint64_t)LLONG_MAX) break;

        struct radvisory ra;
        ra.ra_offset = (off_t)pos;
        ra.ra_count = (int)chunk64;
        (void)fcntl(model->fd, F_RDADVISE, &ra);

        pos += chunk64;
        rem -= chunk64;
    }
#else
    (void)model;
    (void)offset;
    (void)size;
#endif
}

static bool metal_graph_stream_madvise_willneed_range_impl(
        const ds4_model *model,
        uint64_t         offset,
        uint64_t         size,
        bool             enabled,
        uint64_t        *advised) {
    if (!enabled ||
        !model ||
        !model->map ||
        offset > model->size ||
        size == 0 ||
        size > model->size - offset) {
        return !enabled;
    }

#if defined(POSIX_MADV_WILLNEED)
    const uint64_t page = (uint64_t)getpagesize();
    if (page == 0) return false;
    const uint64_t page_offset = offset & ~(page - 1u);
    const uint64_t leading = offset - page_offset;
    if (size > UINT64_MAX - leading ||
        leading + size > UINT64_MAX - (page - 1u)) {
        return false;
    }
    uint64_t advise_bytes = align_up(leading + size, page);
    if (advise_bytes > model->size - page_offset) {
        advise_bytes = model->size - page_offset;
    }
    if (advise_bytes == 0 || advise_bytes > (uint64_t)SIZE_MAX) {
        return false;
    }
    uint8_t *base = (uint8_t *)model->map;
    const int rc = posix_madvise((void *)(base + page_offset),
                                 (size_t)advise_bytes,
                                 POSIX_MADV_WILLNEED);
    if (rc != 0) return false;
    if (advised) {
        if (*advised > UINT64_MAX - advise_bytes) {
            *advised = UINT64_MAX;
        } else {
            *advised += advise_bytes;
        }
    }
    return true;
#else
    (void)model;
    (void)offset;
    (void)size;
    (void)advised;
    return true;
#endif
}

static void metal_graph_stream_readahead_range(
        const ds4_model *model,
        uint64_t         offset,
        uint64_t         size) {
    metal_graph_stream_readahead_range_impl(model,
                                            offset,
                                            size,
                                            metal_graph_stream_readahead_enabled());
    metal_graph_stream_madvise_willneed_range_impl(
            model,
            offset,
            size,
            metal_graph_stream_madvise_willneed_enabled(),
            NULL);
}

static void metal_graph_stream_readahead_spans(
        const ds4_model              *model,
        const ds4_model_map_span_vec *spans) {
    if (!spans) return;
    for (uint32_t i = 0; i < spans->len; i++) {
        metal_graph_stream_readahead_range(model,
                                           spans->v[i].off,
                                           spans->v[i].end - spans->v[i].off);
    }
}

typedef struct {
    uint64_t off;
    uint64_t size;
} metal_graph_stream_pagein_range;

typedef struct {
    pthread_t thread;
    const ds4_model *model;
    metal_graph_stream_pagein_range *ranges;
    pthread_t *threads;
    struct metal_graph_stream_pagein_worker *workers;
    uint32_t n_ranges;
    uint32_t n_threads;
    uint32_t layer;
    uint32_t n_tokens;
    uint32_t unique;
    uint64_t bytes;
    uint64_t touched;
    double read_ms;
    double thread_ms;
    bool profile;
    bool madvise_only;
    bool pread_only;
    bool readahead_only;
    bool started;
    bool ok;
    uint8_t sink;
} metal_graph_stream_pagein_job;

typedef struct metal_graph_stream_pagein_worker {
    metal_graph_stream_pagein_job *job;
    uint32_t first;
    uint32_t stride;
    uint64_t touched;
    double thread_ms;
    bool ok;
    uint8_t sink;
} metal_graph_stream_pagein_worker;

static bool metal_graph_stream_prefill_selected_pagein_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_SELECTED_PAGEIN",
                                 "DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_PAGEIN") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_PAGEIN",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_PAGEIN");
}

static bool metal_graph_stream_prefill_selected_madvise_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_SELECTED_MADVISE",
                                 "DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_MADVISE") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_MADVISE",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_MADVISE");
}

static bool metal_graph_stream_prefill_layer_pagein_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN",
                                 "DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN");
}

static bool metal_graph_stream_prefill_layer_readahead_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD",
                                 "DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE");
}

static bool metal_graph_stream_prefill_layer_pread_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREAD",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREAD") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE");
}

static bool metal_graph_stream_prefill_layer_madvise_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_LAYER_MADVISE",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_MADVISE");
}

static uint32_t metal_graph_stream_prefill_batch_selected_addr_auto_max(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX",
            "DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX");
    if (env && env[0]) {
        char *end = NULL;
        const long v = strtol(env, &end, 10);
        if (end != env) {
            if (v <= 0) return 0;
            if ((unsigned long)v > (unsigned long)UINT32_MAX) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
#ifdef DS4_ROCM_BUILD
    if (DS4_MODEL_VARIANT == DS4_VARIANT_PRO ||
        DS4_MODEL_VARIANT == DS4_VARIANT_FLASH ||
        DS4_MODEL_VARIANT == DS4_VARIANT_GLM52) return UINT32_MAX;
#endif
    if (DS4_MODEL_VARIANT == DS4_VARIANT_PRO) return 800u;
    if (DS4_MODEL_VARIANT == DS4_VARIANT_FLASH) return 760u;
    return 0;
}

static uint32_t metal_graph_stream_prefill_batch_selected_addr_auto_min(void) {
    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MIN",
            "DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MIN");
    if (env && env[0]) {
        char *end = NULL;
        const long v = strtol(env, &end, 10);
        if (end != env) {
            if (v <= 0) return 0;
            if ((unsigned long)v > (unsigned long)UINT32_MAX) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
#ifdef DS4_ROCM_BUILD
    if (DS4_MODEL_VARIANT == DS4_VARIANT_GLM52) return 2u;
#endif
    if (DS4_MODEL_VARIANT == DS4_VARIANT_PRO ||
        DS4_MODEL_VARIANT == DS4_VARIANT_FLASH) return 2u;
    return 0;
}

static bool metal_graph_stream_prefill_batch_selected_addr_enabled(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             n_tokens) {
    if (!g ||
        !g->ssd_streaming ||
        g->quality ||
        !weights ||
        n_tokens <= 1 ||
        glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR",
                              "DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") ||
        glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_EXPERT_ADDR_TABLE",
                              "DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE") ||
        glm_graph_env_present("DS4_ROCM_MOE_WRITE_CLAMPED_ACT",
                              "DS4_METAL_MOE_WRITE_CLAMPED_ACT") ||
        glm_graph_env_present("DS4_ROCM_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                              "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") ||
        DS4_N_LAYER == 0) {
        return false;
    }
    const uint32_t routed_il =
        DS4_N_LEADING_DENSE < DS4_N_LAYER ? DS4_N_LEADING_DENSE : 0u;
    const ds4_layer_weights *layer = &weights->layer[routed_il];
    if (!layer->ffn_gate_exps || !layer->ffn_up_exps || !layer->ffn_down_exps) {
        return false;
    }
#ifdef DS4_ROCM_BUILD
    const bool selected_iq2 =
        glm_stream_selected_expert_cache_supported(layer, routed_il);
    const bool selected_q2 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q2_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q2_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
        glm_stream_expert_cache_addr_layout_supported(weights, layer, routed_il);
    if (!selected_iq2 && !selected_q2) return false;
#else
    if (DS4_N_EXPERT_USED != 6 ||
        layer->ffn_gate_exps->type != DS4_TENSOR_IQ2_XXS ||
        layer->ffn_up_exps->type != DS4_TENSOR_IQ2_XXS ||
        layer->ffn_down_exps->type != DS4_TENSOR_Q2_K) {
        return false;
    }
#endif

    const uint32_t cache_configured =
        ds4_gpu_stream_expert_cache_configured_count();
#ifdef DS4_ROCM_BUILD
    if (cache_configured == 0) {
        return false;
    }
#else
    if (cache_configured < DS4_N_EXPERT) {
        return false;
    }
#endif

    if (glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR",
                              "DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR")) {
        return true;
    }

    const uint32_t max_tokens =
        metal_graph_stream_prefill_batch_selected_addr_auto_max();
    const uint32_t min_tokens =
        metal_graph_stream_prefill_batch_selected_addr_auto_min();
    return max_tokens != 0 && n_tokens >= min_tokens && n_tokens <= max_tokens;
}

static bool metal_graph_cuda_stream_prefill_batch_selected_addr_enabled(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             n_tokens) {
#if !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU) && !defined(__APPLE__)
    if (!g ||
        !g->ssd_streaming ||
        g->quality ||
        !weights ||
        n_tokens <= 1 ||
        getenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") != NULL ||
        getenv("DS4_CUDA_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR") != NULL ||
        getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE") != NULL ||
        getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") != NULL ||
        getenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") != NULL ||
        DS4_N_LAYER == 0 ||
        DS4_N_EXPERT < 128 ||
        DS4_N_EXPERT_USED != 6) {
        return false;
    }

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!layer->ffn_gate_exps || !layer->ffn_up_exps ||
            !layer->ffn_down_exps) {
            continue;
        }
        const bool q4 =
            layer->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
            layer->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
            layer->ffn_down_exps->type == DS4_TENSOR_Q4_K;
        const bool iq2 =
            layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
            layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
            layer->ffn_down_exps->type == DS4_TENSOR_Q2_K;
        if (q4 || iq2) return true;
    }
    return false;
#else
    (void)g;
    (void)weights;
    (void)n_tokens;
    return false;
#endif
}

static bool metal_graph_stream_prefill_batch_selected_addr_layer_supported(
        const ds4_weights *weights,
        uint32_t           il) {
    if (!weights || il >= DS4_N_LAYER) return false;

    const ds4_layer_weights *layer = &weights->layer[il];
    if (!layer->ffn_gate_exps || !layer->ffn_up_exps ||
        !layer->ffn_down_exps) {
        return false;
    }

#ifdef DS4_ROCM_BUILD
    const bool selected_iq2 =
        glm_stream_selected_expert_cache_supported(layer, il);
    const bool selected_q2 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q2_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q2_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
        glm_stream_expert_cache_addr_layout_supported(weights, layer, il);
    return selected_iq2 || selected_q2;
#elif defined(__APPLE__)
    return DS4_N_EXPERT_USED == 6 &&
           layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
           layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
           layer->ffn_down_exps->type == DS4_TENSOR_Q2_K;
#elif !defined(DS4_NO_GPU)
    const bool q4 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q4_K;
    const bool iq2 =
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K;
    return q4 || iq2;
#else
    return false;
#endif
}

#ifdef DS4_ROCM_BUILD
enum { DS4_ROCM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS = 1024 };
enum { DS4_ROCM_STREAM_PREFILL_FULL_LAYER_MAX_SEED_TOKENS = 8 };

typedef struct rocm_graph_stream_layer_expert_load {
    pthread_t                 thread;
    bool                      active;
    bool                      ok;
    const ds4_model          *model;
    const ds4_layer_weights  *layer;
    uint32_t                  il;
    uint64_t                  gate_expert_bytes;
    uint64_t                  down_expert_bytes;
} rocm_graph_stream_layer_expert_load;

static bool rocm_graph_stream_prefill_full_layer_enabled(
        const ds4_gpu_graph      *g,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint32_t                  n_tokens) {
    return g &&
           g->ssd_streaming &&
           !g->quality &&
           layer &&
           n_tokens >= DS4_ROCM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS &&
           glm_stream_resident_decode_layer_supported(layer, il);
}

static uint32_t rocm_graph_stream_prefill_full_layer_seed_tokens(void) {
    const uint32_t budget = ds4_gpu_stream_expert_cache_configured_count();
    const uint64_t entries_per_token =
        (uint64_t)DS4_N_LAYER * (uint64_t)DS4_N_EXPERT_USED;
    if (entries_per_token == 0) return 1;
    uint32_t seed_tokens = budget == 0 ? 1 : (uint32_t)(budget / entries_per_token);
    if (seed_tokens < 1) seed_tokens = 1;
    if (seed_tokens > DS4_ROCM_STREAM_PREFILL_FULL_LAYER_MAX_SEED_TOKENS) {
        seed_tokens = DS4_ROCM_STREAM_PREFILL_FULL_LAYER_MAX_SEED_TOKENS;
    }
    return seed_tokens;
}

static bool rocm_graph_stream_layer_expert_bytes(
        const ds4_layer_weights  *layer,
        uint64_t                 *gate_expert_bytes,
        uint64_t                 *down_expert_bytes) {
    return streaming_layer_gate_down_expert_bytes(layer,
                                                  gate_expert_bytes,
                                                  down_expert_bytes);
}

static bool rocm_graph_stream_layer_expert_load_sync(
        const ds4_model          *model,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint64_t                  gate_expert_bytes,
        uint64_t                  down_expert_bytes) {
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(model,
                                       layer,
                                       il,
                                       gate_expert_bytes,
                                       down_expert_bytes);
    return model &&
           layer &&
           ds4_gpu_stream_expert_cache_load_layer(&table) != 0;
}

static void *rocm_graph_stream_layer_expert_load_thread_main(void *arg) {
    rocm_graph_stream_layer_expert_load *job = arg;
    if (!job) return NULL;
    job->ok = rocm_graph_stream_layer_expert_load_sync(job->model,
                                                       job->layer,
                                                       job->il,
                                                       job->gate_expert_bytes,
                                                       job->down_expert_bytes);
    return NULL;
}

static bool rocm_graph_stream_layer_expert_load_join(
        rocm_graph_stream_layer_expert_load *job) {
    if (!job || !job->active) return true;
    const int rc = pthread_join(job->thread, NULL);
    const bool ok = rc == 0 && job->ok;
    if (rc != 0) {
        fprintf(stderr,
                "ds4: ROCm streaming full-layer expert load join failed: %s\n",
                strerror(rc));
    }
    memset(job, 0, sizeof(*job));
    return ok;
}

static bool rocm_graph_stream_layer_expert_load_start(
        rocm_graph_stream_layer_expert_load *job,
        const ds4_model                     *model,
        const ds4_layer_weights             *layer,
        uint32_t                             il,
        uint64_t                             gate_expert_bytes,
        uint64_t                             down_expert_bytes) {
    if (!job || job->active || !model || !layer) return false;
    memset(job, 0, sizeof(*job));
    job->model = model;
    job->layer = layer;
    job->il = il;
    job->gate_expert_bytes = gate_expert_bytes;
    job->down_expert_bytes = down_expert_bytes;
    const int rc = pthread_create(&job->thread,
                                  NULL,
                                  rocm_graph_stream_layer_expert_load_thread_main,
                                  job);
    if (rc != 0) {
        fprintf(stderr,
                "ds4: failed to start ROCm streaming full-layer expert load "
                "thread for layer %u: %s\n",
                il,
                strerror(rc));
        memset(job, 0, sizeof(*job));
        return false;
    }
    job->active = true;
    return true;
}

static bool rocm_graph_stream_layer_expert_load_start_next(
        rocm_graph_stream_layer_expert_load *job,
        const ds4_gpu_graph                 *g,
        const ds4_model                     *model,
        const ds4_weights                   *weights,
        uint32_t                             il,
        uint32_t                             n_tokens) {
    if (!job ||
        !model ||
        !weights ||
        il >= DS4_N_LAYER ||
        !rocm_graph_stream_prefill_full_layer_enabled(g,
                                                      &weights->layer[il],
                                                      il,
                                                      n_tokens)) {
        return true;
    }
    uint64_t gate_expert_bytes = 0;
    uint64_t down_expert_bytes = 0;
    if (!rocm_graph_stream_layer_expert_bytes(&weights->layer[il],
                                              &gate_expert_bytes,
                                              &down_expert_bytes)) {
        return false;
    }
    return rocm_graph_stream_layer_expert_load_start(job,
                                                     model,
                                                     &weights->layer[il],
                                                     il,
                                                     gate_expert_bytes,
                                                     down_expert_bytes);
}

static bool rocm_graph_stream_layer_expert_load_ready(
        rocm_graph_stream_layer_expert_load *job,
        const ds4_gpu_graph                 *g,
        const ds4_model                     *model,
        const ds4_weights                   *weights,
        uint32_t                             il,
        uint32_t                             n_tokens) {
    if (!model || !weights || il >= DS4_N_LAYER) return false;
    if (!rocm_graph_stream_prefill_full_layer_enabled(g,
                                                      &weights->layer[il],
                                                      il,
                                                      n_tokens)) {
        return true;
    }
    uint64_t gate_expert_bytes = 0;
    uint64_t down_expert_bytes = 0;
    if (!rocm_graph_stream_layer_expert_bytes(&weights->layer[il],
                                              &gate_expert_bytes,
                                              &down_expert_bytes)) {
        return false;
    }
    if (job && job->active) {
        if (job->il != il) {
            fprintf(stderr,
                    "ds4: ROCm streaming full-layer expert load expected layer "
                    "%u but pending job is layer %u\n",
                    il,
                    job->il);
            return false;
        }
        return rocm_graph_stream_layer_expert_load_join(job);
    }
    return rocm_graph_stream_layer_expert_load_sync(model,
                                                    &weights->layer[il],
                                                    il,
                                                    gate_expert_bytes,
                                                    down_expert_bytes);
}

static bool rocm_graph_stream_seed_full_layer_selected(
        ds4_gpu_graph           *g,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 n_tokens) {
    if (!rocm_graph_stream_prefill_full_layer_enabled(g, layer, il, n_tokens)) {
        return true;
    }
    uint64_t gate_expert_bytes = 0;
    uint64_t down_expert_bytes = 0;
    if (!rocm_graph_stream_layer_expert_bytes(layer,
                                              &gate_expert_bytes,
                                              &down_expert_bytes)) {
        return false;
    }
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(model,
                                       layer,
                                       il,
                                       gate_expert_bytes,
                                       down_expert_bytes);
    if (ds4_gpu_stream_expert_cache_seed_from_layer_selected(
                &table,
                metal_graph_batch_router_selected(g),
                n_tokens,
                rocm_graph_stream_prefill_full_layer_seed_tokens(),
                DS4_N_EXPERT_USED) == 0) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr,
                    "ds4: ROCm streaming full-layer prefill seed skipped; "
                    "decode may start with a colder expert cache\n");
            warned = true;
        }
    }
    return true;
}
#endif

static bool metal_graph_stream_prefill_selected_profile_enabled(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_SELECTED_PROFILE",
                                 "DS4_METAL_STREAMING_PREFILL_SELECTED_PROFILE") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_PROFILE",
                                  "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_PROFILE");
}

static void metal_graph_stream_prefill_selected_profile_reset(
        ds4_gpu_graph *g) {
    if (!g) return;
    g->prefill_selected_profile_rows = 0;
    g->prefill_selected_profile_unique = 0;
    g->prefill_selected_profile_selected_bytes = 0;
    g->prefill_selected_profile_full_bytes = 0;
    g->prefill_selected_profile_layers = 0;
    g->prefill_selected_profile_min_unique = UINT32_MAX;
    g->prefill_selected_profile_max_unique = 0;
}

static uint64_t metal_graph_stream_prefill_selected_profile_add_bytes(
        uint64_t a,
        uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

static bool metal_graph_selected_profile_layer_impl(
        ds4_gpu_graph          *g,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                n_tokens,
        const char             *label) {
    if (!layer || !metal_graph_batch_router_selected(g) || n_tokens == 0 ||
        DS4_N_EXPERT == 0 || DS4_N_EXPERT > DS4_MAX_EXPERT ||
        DS4_N_EXPERT_USED == 0 || DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return false;
    }

    const uint64_t n_ids = (uint64_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_ids > SIZE_MAX / sizeof(int32_t)) return false;
    int32_t *selected = xmalloc((size_t)n_ids * sizeof(selected[0]));
    const bool read_ok = ds4_gpu_tensor_read(metal_graph_batch_router_selected(g),
                                             0,
                                             selected,
                                             n_ids * sizeof(selected[0])) != 0;
    if (!read_ok) {
        free(selected);
        return false;
    }

    bool seen[DS4_MAX_EXPERT] = { false };
    uint32_t unique = 0;
    for (uint64_t i = 0; i < n_ids; i++) {
        const int32_t expert = selected[i];
        if (expert < 0 || (uint32_t)expert >= DS4_N_EXPERT) {
            fprintf(stderr,
                    "ds4: Metal streaming prefill selected profile expert id %d is outside 0..%u at layer %u\n",
                    expert,
                    (uint32_t)DS4_N_EXPERT,
                    il);
            free(selected);
            return false;
        }
        if (!seen[expert]) {
            seen[expert] = true;
            unique++;
        }
    }
    free(selected);

    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    if (layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
        layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
        fprintf(stderr, "ds4: Metal streaming prefill selected profile byte size overflow at layer %u\n", il);
        return false;
    }
    const uint64_t gate_expert_bytes = layer->ffn_gate_exps->dim[1] * gate_row_bytes;
    const uint64_t down_expert_bytes = layer->ffn_down_exps->dim[1] * down_row_bytes;
    if (gate_expert_bytes > UINT64_MAX - gate_expert_bytes ||
        gate_expert_bytes + gate_expert_bytes > UINT64_MAX - down_expert_bytes) {
        fprintf(stderr, "ds4: Metal streaming prefill selected profile byte size overflow at layer %u\n", il);
        return false;
    }
    const uint64_t per_expert_bytes = gate_expert_bytes + gate_expert_bytes +
                                      down_expert_bytes;
    const uint64_t selected_bytes =
        unique > UINT64_MAX / per_expert_bytes ?
        UINT64_MAX : (uint64_t)unique * per_expert_bytes;
    const uint64_t full_bytes =
        (uint64_t)DS4_N_EXPERT > UINT64_MAX / per_expert_bytes ?
        UINT64_MAX : (uint64_t)DS4_N_EXPERT * per_expert_bytes;
    const double ratio = full_bytes == 0 ? 0.0 :
        (double)selected_bytes / (double)full_bytes;

    g->prefill_selected_profile_layers++;
    g->prefill_selected_profile_rows =
        metal_graph_stream_prefill_selected_profile_add_bytes(
                g->prefill_selected_profile_rows,
                n_ids);
    g->prefill_selected_profile_unique =
        metal_graph_stream_prefill_selected_profile_add_bytes(
                g->prefill_selected_profile_unique,
                unique);
    g->prefill_selected_profile_selected_bytes =
        metal_graph_stream_prefill_selected_profile_add_bytes(
                g->prefill_selected_profile_selected_bytes,
                selected_bytes);
    g->prefill_selected_profile_full_bytes =
        metal_graph_stream_prefill_selected_profile_add_bytes(
                g->prefill_selected_profile_full_bytes,
                full_bytes);
    if (unique < g->prefill_selected_profile_min_unique) {
        g->prefill_selected_profile_min_unique = unique;
    }
    if (unique > g->prefill_selected_profile_max_unique) {
        g->prefill_selected_profile_max_unique = unique;
    }

    fprintf(stderr,
            "ds4: %s layer=%u "
            "tokens=%u unique=%u/%u selected=%.2f GiB full=%.2f GiB ratio=%.3f\n",
            label ? label : "selected expert profile",
            il,
            n_tokens,
            unique,
            (uint32_t)DS4_N_EXPERT,
            (double)selected_bytes / (1024.0 * 1024.0 * 1024.0),
            (double)full_bytes / (1024.0 * 1024.0 * 1024.0),
            ratio);
    return true;
}

static bool metal_graph_stream_prefill_selected_profile_layer(
        ds4_gpu_graph          *g,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                n_tokens) {
    if (!metal_graph_stream_prefill_selected_profile_enabled(g)) return true;
    return metal_graph_selected_profile_layer_impl(
            g,
            layer,
            il,
            n_tokens,
            "Metal streaming prefill selected profile");
}

static void metal_graph_selected_profile_summary_impl(
        const ds4_gpu_graph *g,
        const char          *label) {
    if (!g || g->prefill_selected_profile_layers == 0) {
        return;
    }
    const double layers = (double)g->prefill_selected_profile_layers;
    const double avg_unique = (double)g->prefill_selected_profile_unique / layers;
    const double ratio = g->prefill_selected_profile_full_bytes == 0 ? 0.0 :
        (double)g->prefill_selected_profile_selected_bytes /
        (double)g->prefill_selected_profile_full_bytes;
    fprintf(stderr,
            "ds4: %s summary "
            "layers=%u avg_unique=%.1f min_unique=%u max_unique=%u "
            "selected=%.2f GiB full=%.2f GiB ratio=%.3f rows=%" PRIu64 "\n",
            label ? label : "selected expert profile",
            g->prefill_selected_profile_layers,
            avg_unique,
            g->prefill_selected_profile_min_unique == UINT32_MAX ?
                0 : g->prefill_selected_profile_min_unique,
            g->prefill_selected_profile_max_unique,
            (double)g->prefill_selected_profile_selected_bytes /
                (1024.0 * 1024.0 * 1024.0),
            (double)g->prefill_selected_profile_full_bytes /
                (1024.0 * 1024.0 * 1024.0),
            ratio,
            g->prefill_selected_profile_rows);
}

#endif /* !DS4_NO_GPU */
