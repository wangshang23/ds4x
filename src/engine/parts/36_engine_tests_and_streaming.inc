int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt) {
#ifndef DS4_NO_GPU
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: %s graph test requested but backend is unavailable\n",
                ds4_backend_name(e->backend));
        return 1;
    }
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        return glm_metal_graph_test(e, prompt);
    }
    return metal_graph_decode_test(&e->model, &e->weights, prompt, e->quality);
#else
    (void)e;
    (void)prompt;
    fprintf(stderr, "ds4: graph test requested but this build has no graph backend support\n");
    return 1;
#endif
}

int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt) {
#ifndef DS4_NO_GPU
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: %s full graph test requested but backend is unavailable\n",
                ds4_backend_name(e->backend));
        return 1;
    }
    return metal_graph_first_token_full_test(&e->model, &e->weights, prompt, e->quality);
#else
    (void)e;
    (void)prompt;
    fprintf(stderr, "ds4: full graph test requested but this build has no graph backend support\n");
    return 1;
#endif
}

int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size) {
#ifndef DS4_NO_GPU
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: %s prompt graph test requested but backend is unavailable\n",
                ds4_backend_name(e->backend));
        return 1;
    }
    return metal_graph_prompt_logits_test(&e->model, &e->weights, prompt, ctx_size);
#else
    (void)e;
    (void)prompt;
    (void)ctx_size;
    fprintf(stderr, "ds4: prompt graph test requested but this build has no graph backend support\n");
    return 1;
#endif
}

int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "ds4: head test requires a non-empty prompt\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;
    const ds4_layer_weights *layer0 = &weights->layer[0];

    float *prompt_embd = xmalloc((size_t)prompt->len * DS4_N_EMBD * sizeof(prompt_embd[0]));
    embed_prompt(model, weights, prompt, DS4_N_EMBD, prompt_embd);

    const uint32_t n_hc = DS4_N_HC;
    float *hc0 = xmalloc((size_t)DS4_N_EMBD * sizeof(hc0[0]));
    float *residual_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(residual_hc[0]));
    float hc_post[4];
    float hc_comb[16];
    layer_attn_pre_one(model, layer0,
        prompt_embd + (uint64_t)(prompt->len - 1) * DS4_N_EMBD,
        hc0, residual_hc, hc_post, hc_comb);
    print_vec_stats("blk.0 attn_pre", hc0, DS4_N_EMBD);

    float *attn_norm0 = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_norm0[0]));
    layer_attn_norm_one(attn_norm0, model, layer0, hc0);

    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    float *q0 = xmalloc((size_t)q_dim * sizeof(q0[0]));
    layer_q_projection_normed_one(model, layer0, attn_norm0, q0);
    print_vec_stats("blk.0 q", q0, q_dim);

    float *kv0 = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(kv0[0]));
    layer_kv_projection_normed_one(model, layer0, attn_norm0, kv0);
    print_vec_stats("blk.0 kv", kv0, DS4_N_HEAD_DIM);
    rope_tail_layer_inplace(q0, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    rope_tail_layer_inplace(kv0, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv0, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(kv0, DS4_N_HEAD_DIM);

    float *attn_heads = xmalloc((size_t)q_dim * sizeof(attn_heads[0]));
    layer_attention_one(attn_heads, model, layer0, q0, kv0);
    print_vec_stats("blk.0 attn_heads", attn_heads, q_dim);
    rope_tail_layer_inplace(attn_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, (uint32_t)(prompt->len - 1), 0, true);

    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    layer_grouped_out_one(attn_out, model, layer0, attn_heads);
    print_vec_stats("blk.0 attn_out", attn_out, DS4_N_EMBD);

    float *after_attn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_attn_hc[0]));
    hc_post_one(after_attn_hc, attn_out, residual_hc, hc_post, hc_comb, DS4_N_EMBD, n_hc);
    print_vec_stats("blk.0 after_attn_hc", after_attn_hc, (uint64_t)n_hc * DS4_N_EMBD);

    float *after_ffn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_ffn_hc[0]));
    layer_ffn_one(after_ffn_hc, model, layer0, after_attn_hc, 0, prompt->v[prompt->len - 1],
                  NULL, 0.0f, true);
    print_vec_stats("blk.0 after_ffn_hc", after_ffn_hc, (uint64_t)n_hc * DS4_N_EMBD);

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    output_logits_one(logits, model, weights, after_ffn_hc);
    print_vec_stats("logits", logits, DS4_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after native blk.0 slice:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(after_ffn_hc);
    free(after_attn_hc);
    free(attn_out);
    free(attn_heads);
    free(kv0);
    free(q0);
    free(attn_norm0);
    free(residual_hc);
    free(hc0);
    free(prompt_embd);
    return 0;
}

int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "ds4: first-token test requires a non-empty prompt\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;

    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        float *hidden = xmalloc((size_t)DS4_N_EMBD * sizeof(hidden[0]));
        float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));

        forward_glm_first_token_cpu(hidden, model, weights, prompt->v[0]);
        print_vec_stats("first-token final_hidden", hidden, DS4_N_EMBD);
        output_logits_glm_one(logits, model, weights, hidden);
        print_vec_stats("first-token logits", logits, DS4_N_VOCAB);

        int best[8];
        for (int i = 0; i < 8; i++) best[i] = -1;
        for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
            for (int j = 0; j < 8; j++) {
                if (best[j] < 0 || logits[i] > logits[best[j]]) {
                    for (int k = 7; k > j; k--) best[k] = best[k - 1];
                    best[j] = (int)i;
                    break;
                }
            }
        }

        printf("top logits after GLM first-token CPU pass:\n");
        for (int i = 0; i < 8; i++) {
            printf("  %6d  %9.4f  %.*s\n",
                best[i],
                logits[best[i]],
                (int)vocab->token[best[i]].len,
                vocab->token[best[i]].ptr);
        }

        free(logits);
        free(hidden);
        return 0;
    }

    float *hc = xmalloc((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(hc[0]));
    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    forward_first_token_cpu(hc, model, weights, prompt->v[0]);
    print_vec_stats("first-token final_hc", hc, (uint64_t)DS4_N_HC * DS4_N_EMBD);
    output_logits_one(logits, model, weights, hc);
    print_vec_stats("first-token logits", logits, DS4_N_VOCAB);

    int best[8];
    for (int i = 0; i < 8; i++) best[i] = -1;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
        for (int j = 0; j < 8; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int k = 7; k > j; k--) best[k] = best[k - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    printf("top logits after first-token whole-model CPU pass:\n");
    for (int i = 0; i < 8; i++) {
        printf("  %6d  %9.4f  %.*s\n",
            best[i],
            logits[best[i]],
            (int)vocab->token[best[i]].len,
            vocab->token[best[i]].ptr);
    }

    free(logits);
    free(hc);
    return 0;
}

static bool ds4_engine_configure_streaming_auto_cache(ds4_engine *e) {
#ifdef DS4_NO_GPU
    (void)e;
    return true;
#else
    if (!e ||
        !e->ssd_streaming ||
        !ds4_backend_supports_ssd_streaming(e->backend) ||
        e->ssd_streaming_cache_experts != 0 ||
        e->ssd_streaming_cache_bytes != 0) {
        return true;
    }
    if (!ds4_backend_supports_streaming_auto_cache(e->backend)) {
        return true;
    }

    const uint64_t recommended = ds4_gpu_recommended_working_set_size();
    if (recommended == 0) {
        fprintf(stderr,
                "ds4: SSD streaming auto cache: recommended working set unavailable; "
                "set --ssd-streaming-cache-experts N or NGB explicitly\n");
        return false;
    }

    uint64_t non_routed_bytes = 0;
    if (!weights_streaming_non_routed_bytes(&e->weights, &non_routed_bytes)) {
        fprintf(stderr,
                "ds4: SSD streaming auto cache could not measure non-routed model weights\n");
        return false;
    }

    uint64_t per_expert_bytes = 0;
    if (!ds4_streaming_routed_expert_bytes(&e->weights, &per_expert_bytes)) {
        /* A valid distributed GLM slice can contain only the leading dense
         * layers. It has no routed weights to stream or cache. */
        if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA &&
            e->distributed.role != DS4_DISTRIBUTED_NONE &&
            e->distributed.layers.set) {
            fprintf(stderr,
                    "ds4: SSD streaming layer slice has no routed experts; "
                    "expert cache disabled\n");
            e->ssd_streaming_cache_experts = 0;
            e->ssd_streaming_cache_bytes = 0;
            return true;
        }
        fprintf(stderr,
                "ds4: SSD streaming auto cache could not measure routed expert size\n");
        return false;
    }

    uint64_t max_model_experts = 0;
    if (!ds4_streaming_cacheable_expert_count(&e->weights,
                                              &max_model_experts,
                                              NULL)) {
        fprintf(stderr,
                "ds4: SSD streaming auto cache could not count local cacheable experts\n");
        return false;
    }
    ds4_ssd_cache_plan plan;
    if (!ds4_ssd_auto_cache_plan(recommended,
                                 non_routed_bytes,
                                 per_expert_bytes,
                                 max_model_experts,
                                 &plan)) {
        fprintf(stderr,
                "ds4: SSD streaming auto cache could not compute a valid cache budget\n");
        return false;
    }

    uint32_t cache_experts = plan.cache_experts;
    uint64_t effective_cache_bytes = plan.effective_cache_bytes;
    const bool glm_full_layer_reserve =
        DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA &&
        ds4_backend_supports_glm_streaming_full_layers(e->backend);
    /*
     * The 12 GiB GLM graph cap is a Metal-era VM pressure guard. ROCm/Strix
     * reports its own working-set recommendation and is hurt badly by this cap:
     * it leaves less than one routed token's experts in the dynamic cache.
     */
    const bool glm_auto_cap = glm_full_layer_reserve &&
                              e->backend == DS4_BACKEND_METAL;
    const uint64_t glm_auto_cap_bytes = 12ull * 1024ull * 1024ull * 1024ull;
    if (glm_auto_cap && effective_cache_bytes > glm_auto_cap_bytes) {
        uint64_t capped_experts = glm_auto_cap_bytes / per_expert_bytes;
        if (capped_experts == 0) capped_experts = 1;
        if (capped_experts > max_model_experts) capped_experts = max_model_experts;
        cache_experts = capped_experts > UINT32_MAX ?
            UINT32_MAX : (uint32_t)capped_experts;
        effective_cache_bytes = (uint64_t)cache_experts * per_expert_bytes;
    }

#ifdef DS4_ROCM_BUILD
    uint64_t glm_rocm_guard_cap_bytes = 0;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA &&
        e->backend == DS4_BACKEND_CUDA) {
        const int requested_ctx = e->placement_ctx_hint > 0 ?
            e->placement_ctx_hint : 4096;
        uint32_t guard_ctx = 0;
        if (!glm_graph_context_request(requested_ctx, &guard_ctx)) return false;
        const uint32_t work_ctx =
            glm_graph_full_attention_cap(guard_ctx, true);
        const uint32_t compact_cap =
            glm_graph_compact_cache_initial_cap(guard_ctx, work_ctx);
        ds4_context_memory graph_mem;
        if (e->distributed.role != DS4_DISTRIBUTED_NONE &&
            e->distributed.layers.set) {
            const uint32_t normal_layers = glm_graph_normal_layer_count();
            const uint32_t layer_end = e->distributed.layers.has_output ?
                (normal_layers ? normal_layers - 1u : 0u) :
                e->distributed.layers.end;
            graph_mem = glm_graph_context_memory_estimate_for_compact_cap_slice(
                    guard_ctx,
                    work_ctx,
                    compact_cap,
                    true,
                    e->distributed.layers.start,
                    layer_end);
        } else {
            graph_mem = glm_graph_context_memory_estimate_for_compact_cap(
                    guard_ctx, work_ctx, compact_cap, true);
        }
        uint64_t active_model_bytes =
            glm_graph_streaming_active_model_bytes(&e->weights);
        if (non_routed_bytes > active_model_bytes) {
            active_model_bytes = non_routed_bytes;
        }
        const double fraction = glm_graph_env_double(
                "DS4_GLM_MEMORY_GUARD_FRACTION", 0.99, 0.50, 1.00);
        const double reserve_gib = glm_graph_env_double(
                "DS4_GLM_MEMORY_GUARD_RESERVE_GB",
                glm_graph_memory_guard_default_reserve_gib(
                        recommended, active_model_bytes),
                0.0,
                1024.0);
        const uint64_t fraction_budget =
            (uint64_t)((double)recommended * fraction);
        const uint64_t reserve_bytes =
            (uint64_t)(reserve_gib * 1024.0 * 1024.0 * 1024.0);
        const uint64_t reserve_budget = reserve_bytes >= recommended ?
            0 : recommended - reserve_bytes;
        uint64_t guard_budget = fraction_budget;
        if (reserve_bytes != 0 && reserve_budget < guard_budget) {
            guard_budget = reserve_budget;
        }
        const uint64_t fixed_bytes = glm_graph_saturating_add_u64(
                active_model_bytes, graph_mem.total_bytes);
        if (guard_budget > fixed_bytes) {
            glm_rocm_guard_cap_bytes = guard_budget - fixed_bytes;
        }
        if (glm_rocm_guard_cap_bytes < per_expert_bytes) {
            fprintf(stderr,
                    "ds4: GLM ROCm auto cache has no room after model, graph, "
                    "and memory-guard reserves\n");
            return false;
        }
        if (effective_cache_bytes > glm_rocm_guard_cap_bytes) {
            uint64_t capped_experts =
                glm_rocm_guard_cap_bytes / per_expert_bytes;
            if (capped_experts > max_model_experts) {
                capped_experts = max_model_experts;
            }
            cache_experts = capped_experts > UINT32_MAX ?
                UINT32_MAX : (uint32_t)capped_experts;
            effective_cache_bytes =
                (uint64_t)cache_experts * per_expert_bytes;
        }
    }
#endif

    e->ssd_streaming_cache_experts = cache_experts;
    e->ssd_streaming_cache_bytes = effective_cache_bytes;
    fprintf(stderr,
            "ds4: SSD streaming auto cache budget\n");
    fprintf(stderr,
            "ds4:   %s recommends %.2f GiB working set\n",
            ds4_backend_name(e->backend),
            (double)recommended / 1073741824.0);
    fprintf(stderr,
            "ds4:   using %.0f%% total for model + cached experts: %.2f GiB\n",
            recommended != 0 ?
                100.0 * (double)plan.model_target_bytes / (double)recommended :
                0.0,
            (double)plan.model_target_bytes / 1073741824.0);
    fprintf(stderr,
            "ds4:   non-routed weights: %.2f GiB\n",
            (double)non_routed_bytes / 1073741824.0);
    fprintf(stderr,
            "ds4:   routed expert size: %.2f MiB\n",
            (double)per_expert_bytes / 1048576.0);
    if (glm_full_layer_reserve) {
        fprintf(stderr,
                "ds4:   expert budget before prefill/full-layer reserve: %u "
                "(%.2f GiB)\n",
                e->ssd_streaming_cache_experts,
                (double)effective_cache_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                "ds4:   expert budget before prefill reserve: %u (%.2f GiB)\n",
                e->ssd_streaming_cache_experts,
                (double)effective_cache_bytes / 1073741824.0);
    }
    if (glm_auto_cap && plan.effective_cache_bytes != effective_cache_bytes) {
        fprintf(stderr,
                "ds4:   GLM graph auto cache capped at %.2f GiB; pass "
                "--ssd-streaming-cache-experts NGB to override\n",
                (double)glm_auto_cap_bytes / 1073741824.0);
    }
#ifdef DS4_ROCM_BUILD
    if (glm_rocm_guard_cap_bytes != 0 &&
        plan.effective_cache_bytes != effective_cache_bytes) {
        fprintf(stderr,
                "ds4:   GLM ROCm cache capped to %.2f GiB by the memory "
                "guard for ctx=%d\n",
                (double)effective_cache_bytes / 1073741824.0,
                e->placement_ctx_hint > 0 ? e->placement_ctx_hint : 4096);
    }
#endif
    if (plan.model_target_bytes <= non_routed_bytes) {
        fprintf(stderr,
                "ds4:   note: non-routed weights already fill the 80%% target; keeping a one-expert cache\n");
    }
    return true;
#endif
}

static uint32_t ds4_glm_streaming_normal_layer_count(void) {
    if (DS4_N_NEXTN_PREDICT != 0 && DS4_N_LAYER > DS4_N_NEXTN_PREDICT) {
        return DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    }
    return DS4_N_LAYER;
}

static uint32_t ds4_glm_streaming_supported_resident_prefix_layers(
        const ds4_weights *weights,
        uint32_t          *layer_start_out) {
    if (layer_start_out) *layer_start_out = 0;
    if (!weights) return 0;
    const uint32_t normal_layers = ds4_glm_streaming_normal_layer_count();
    if (normal_layers <= DS4_N_LEADING_DENSE) return 0;

    uint32_t layer_start = DS4_N_LEADING_DENSE;
    while (layer_start < normal_layers &&
           !glm_stream_resident_decode_layer_supported(
                   &weights->layer[layer_start], layer_start)) {
        layer_start++;
    }
    if (layer_start == normal_layers) return 0;

    uint32_t n = 0;
    for (uint32_t il = layer_start; il < normal_layers; il++) {
        if (!glm_stream_resident_decode_layer_supported(&weights->layer[il],
                                                        il)) {
            break;
        }
        n++;
    }
    if (layer_start_out) *layer_start_out = layer_start;
    return n;
}

static bool ds4_glm_streaming_resident_prefix_bytes(
        const ds4_weights *weights,
        uint32_t           layer_start,
        uint32_t           layers,
        uint64_t          *bytes_out) {
    if (bytes_out) *bytes_out = 0;
    if (!weights || !bytes_out || DS4_N_EXPERT == 0) return false;

    uint64_t total = 0;
    for (uint32_t i = 0; i < layers; i++) {
        const uint32_t il = layer_start + i;
        if (!glm_stream_resident_decode_layer_supported(&weights->layer[il],
                                                        il)) {
            return false;
        }
        uint64_t per_expert_bytes = 0;
        if (!streaming_layer_routed_expert_bytes(&weights->layer[il],
                                                 &per_expert_bytes) ||
            per_expert_bytes > UINT64_MAX / (uint64_t)DS4_N_EXPERT) {
            return false;
        }
        const uint64_t layer_bytes =
            per_expert_bytes * (uint64_t)DS4_N_EXPERT;
        if (total > UINT64_MAX - layer_bytes) return false;
        total += layer_bytes;
    }

    *bytes_out = total;
    return true;
}

static uint32_t ds4_glm_streaming_auto_full_layers(
        const ds4_weights *weights,
        uint32_t           layer_start,
        uint32_t           supported_layers,
        uint64_t           total_budget_bytes) {
    if (!weights || supported_layers == 0 || total_budget_bytes == 0) {
        return 0;
    }

    const uint64_t max_auto_bytes = 10ull * 1024ull * 1024ull * 1024ull;
    uint64_t target_bytes = total_budget_bytes / 7ull;
    if (target_bytes > max_auto_bytes) target_bytes = max_auto_bytes;
    if (target_bytes == 0) return 0;

    uint32_t best = 0;
    for (uint32_t n = 1; n <= supported_layers; n++) {
        uint64_t bytes = 0;
        if (!ds4_glm_streaming_resident_prefix_bytes(weights,
                                                     layer_start,
                                                     n,
                                                     &bytes)) {
            break;
        }
        if (bytes > target_bytes) break;
        best = n;
    }
    return best;
}

static bool ds4_engine_configure_streaming_cache_budget(ds4_engine *e) {
    g_glm_streaming_full_resident_start = 0;
    g_glm_streaming_full_resident_layers = 0;
#ifdef DS4_NO_GPU
    (void)e;
    return true;
#else
    if (!e || !e->ssd_streaming) return true;

    const bool glm_full_layer_streaming =
        DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA &&
        ds4_backend_supports_glm_streaming_full_layers(e->backend);
    if (e->ssd_streaming_full_layers_set &&
        e->ssd_streaming_full_layers != 0 &&
        !glm_full_layer_streaming) {
        fprintf(stderr,
                "ds4: --ssd-streaming-full-layers is currently supported "
                "only for GLM graph SSD streaming; ignoring\n");
        e->ssd_streaming_full_layers = 0;
    }

    uint64_t per_expert_bytes = 0;
    const bool need_expert_bytes =
        e->ssd_streaming_cache_experts != 0 ||
        e->ssd_streaming_cache_bytes != 0 ||
        (glm_full_layer_streaming && !e->ssd_streaming_full_layers_set) ||
        e->ssd_streaming_full_layers != 0;
    if (need_expert_bytes &&
        !ds4_streaming_routed_expert_bytes(&e->weights,
                                           &per_expert_bytes)) {
        const bool no_routed_slice =
            DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA &&
            e->distributed.role != DS4_DISTRIBUTED_NONE &&
            e->distributed.layers.set;
        if (no_routed_slice &&
            e->ssd_streaming_cache_experts == 0 &&
            e->ssd_streaming_cache_bytes == 0 &&
            e->ssd_streaming_full_layers == 0) {
            return true;
        }
        fprintf(stderr,
                "ds4: SSD streaming could not measure routed expert size\n");
        return false;
    }
    uint64_t max_cache_experts_u64 = 0;
    if (need_expert_bytes &&
        !ds4_streaming_cacheable_expert_count(&e->weights,
                                              &max_cache_experts_u64,
                                              NULL)) {
        fprintf(stderr,
                "ds4: SSD streaming could not count local cacheable experts\n");
        return false;
    }
    const uint32_t max_cache_experts = max_cache_experts_u64 > UINT32_MAX ?
        UINT32_MAX : (uint32_t)max_cache_experts_u64;
    if (e->ssd_streaming_cache_bytes == 0 &&
        max_cache_experts != 0 &&
        e->ssd_streaming_cache_experts > max_cache_experts) {
        fprintf(stderr,
                "ds4: SSD streaming expert cache capped from %u to %u "
                "experts cacheable by this layer slice\n",
                e->ssd_streaming_cache_experts,
                max_cache_experts);
        e->ssd_streaming_cache_experts = max_cache_experts;
    }

    uint32_t full_layers = 0;
    uint64_t full_layers_bytes = 0;
    bool full_layers_auto = false;
    uint32_t supported = 0;
    uint32_t supported_start = 0;
    uint64_t total_cache_bytes = e->ssd_streaming_cache_bytes;
    uint64_t prefill_headroom_bytes = 0;
    uint64_t budget_after_prefill_headroom = total_cache_bytes;
    if (total_cache_bytes != 0) {
        if (!ds4_streaming_prefill_headroom_bytes(&e->weights,
                                                  &prefill_headroom_bytes)) {
            fprintf(stderr,
                    "ds4: SSD streaming prefill headroom byte accounting failed\n");
            return false;
        }
        if (prefill_headroom_bytes >= total_cache_bytes) {
            fprintf(stderr,
                    "ds4: --ssd-streaming-cache-experts byte budget %.2f GiB "
                    "is too small: routed prefill headroom needs %.2f GiB\n",
                    (double)total_cache_bytes / 1073741824.0,
                    (double)prefill_headroom_bytes / 1073741824.0);
            return false;
        }
        budget_after_prefill_headroom =
            total_cache_bytes - prefill_headroom_bytes;
    }
    e->ssd_streaming_prefill_headroom_bytes = prefill_headroom_bytes;
    if (glm_full_layer_streaming) {
        supported =
            ds4_glm_streaming_supported_resident_prefix_layers(
                    &e->weights, &supported_start);
        if (!e->ssd_streaming_full_layers_set &&
            e->ssd_streaming_cache_bytes != 0) {
            /*
             * On ROCm/Strix, the dynamic selected-expert cache is a better use
             * of memory than pinning full routed layers by default. Manual
             * --ssd-streaming-full-layers remains available for experiments.
             */
#ifdef DS4_ROCM_BUILD
            if (e->backend == DS4_BACKEND_CUDA) {
                e->ssd_streaming_full_layers = 0;
            } else
#endif
            {
                e->ssd_streaming_full_layers =
                    ds4_glm_streaming_auto_full_layers(
                            &e->weights,
                            supported_start,
                            supported,
                            budget_after_prefill_headroom);
            }
            full_layers_auto = true;
        }
    }

    if (e->ssd_streaming_full_layers != 0) {
        const uint32_t requested = e->ssd_streaming_full_layers;
        supported =
            ds4_glm_streaming_supported_resident_prefix_layers(
                    &e->weights, &supported_start);
        full_layers = requested < supported ? requested : supported;
        if (full_layers != requested) {
            fprintf(stderr,
                    "ds4: GLM SSD streaming full resident layers capped "
                    "from %u to %u supported routed prefix layers\n",
                    requested,
                    full_layers);
        }

        if (total_cache_bytes != 0 && per_expert_bytes != 0) {
            /*
             * The dynamic cache must be large enough for batch prefill too:
             * selected-address prefill can see many unique experts for a
             * layer, not just the top-k used by one decode token.
             */
            uint64_t min_dynamic_experts =
                DS4_N_EXPERT != 0 ? (uint64_t)DS4_N_EXPERT : 1u;
            if (min_dynamic_experts > UINT64_MAX / per_expert_bytes) {
                fprintf(stderr,
                        "ds4: SSD streaming full-layer budget overflow\n");
                return false;
            }
            const uint64_t min_dynamic_bytes =
                min_dynamic_experts * per_expert_bytes;
            const uint32_t before_budget_cap = full_layers;
            while (full_layers != 0) {
                uint64_t bytes = 0;
                if (!ds4_glm_streaming_resident_prefix_bytes(&e->weights,
                                                             supported_start,
                                                             full_layers,
                                                             &bytes)) {
                    fprintf(stderr,
                            "ds4: SSD streaming full-layer byte accounting failed\n");
                    return false;
                }
                if (budget_after_prefill_headroom > min_dynamic_bytes &&
                    bytes <= budget_after_prefill_headroom - min_dynamic_bytes) {
                    full_layers_bytes = bytes;
                    break;
                }
                full_layers--;
            }
            if (full_layers == 0) full_layers_bytes = 0;
            if (full_layers != before_budget_cap) {
                fprintf(stderr,
                        "ds4: GLM SSD streaming full resident layers capped "
                        "from %u to %u by %.2f GiB total expert budget\n",
                        before_budget_cap,
                        full_layers,
                        (double)total_cache_bytes / 1073741824.0);
            }
        } else if (full_layers != 0 &&
                   !ds4_glm_streaming_resident_prefix_bytes(&e->weights,
                                                            supported_start,
                                                            full_layers,
                                                            &full_layers_bytes)) {
            fprintf(stderr,
                    "ds4: SSD streaming full-layer byte accounting failed\n");
            return false;
        }
    }

    if (total_cache_bytes != 0) {
        uint64_t dynamic_cache_bytes = budget_after_prefill_headroom;
        if (full_layers_bytes != 0) {
            if (full_layers_bytes >= dynamic_cache_bytes) {
                fprintf(stderr,
                        "ds4: SSD streaming full-layer budget leaves no dynamic expert cache\n");
                return false;
            }
            dynamic_cache_bytes -= full_layers_bytes;
        }

        uint64_t budget_expert_bytes = 0;
        uint32_t budget =
            ds4_streaming_cache_experts_for_byte_budget(
                    &e->weights,
                    dynamic_cache_bytes,
                    &budget_expert_bytes);
        if (max_cache_experts != 0 && budget > max_cache_experts) {
            budget = max_cache_experts;
        }
        if (budget == 0 || budget_expert_bytes == 0) {
            fprintf(stderr,
                    "ds4: --ssd-streaming-cache-experts byte budget is too small or invalid for this model\n");
            return false;
        }
        e->ssd_streaming_cache_experts = budget;
        e->ssd_streaming_cache_bytes =
            (uint64_t)budget * budget_expert_bytes;

        if (full_layers != 0) {
            fprintf(stderr,
                    "ds4: GLM SSD streaming full resident layers: %u "
                    "(%.2f GiB, %s)\n",
                    full_layers,
                    (double)full_layers_bytes / 1073741824.0,
                    full_layers_auto ? "auto" : "explicit");
            fprintf(stderr,
                    "ds4: %s SSD streaming total expert budget %.2f GiB = "
                    "%.2f GiB prefill headroom + %.2f GiB full layers + "
                    "%.2f GiB dynamic cache (%u experts, %.2f MiB each)\n",
                    ds4_backend_name(e->backend),
                    (double)total_cache_bytes / 1073741824.0,
                    (double)prefill_headroom_bytes / 1073741824.0,
                    (double)full_layers_bytes / 1073741824.0,
                    (double)e->ssd_streaming_cache_bytes / 1073741824.0,
                    budget,
                    (double)budget_expert_bytes / 1048576.0);
        } else {
            if (glm_full_layer_streaming) {
                const char *reason =
                    e->ssd_streaming_full_layers_set ?
                    "disabled by --ssd-streaming-full-layers 0" :
                    "auto selected 0 layers";
                fprintf(stderr,
                        "ds4: GLM SSD streaming full resident layers: 0 (%s)\n",
                        reason);
            }
            fprintf(stderr,
                    "ds4: %s SSD streaming total expert budget %.2f GiB = "
                    "%.2f GiB prefill headroom + %.2f GiB dynamic cache "
                    "(%u experts, %.2f MiB each)\n",
                    ds4_backend_name(e->backend),
                    (double)total_cache_bytes / 1073741824.0,
                    (double)prefill_headroom_bytes / 1073741824.0,
                    (double)e->ssd_streaming_cache_bytes / 1073741824.0,
                    budget,
                    (double)budget_expert_bytes / 1048576.0);
        }
    } else if (full_layers != 0) {
        fprintf(stderr,
                "ds4: GLM SSD streaming full resident layers: %u "
                "(%.2f GiB, %s), dynamic cache budget: %u experts\n",
                full_layers,
                (double)full_layers_bytes / 1073741824.0,
                full_layers_auto ? "auto" : "explicit",
                e->ssd_streaming_cache_experts);
    } else if (glm_full_layer_streaming) {
        const char *reason =
            e->ssd_streaming_full_layers_set ?
            "disabled by --ssd-streaming-full-layers 0" :
            "not auto-tuned for expert-count cache budgets";
        fprintf(stderr,
                "ds4: GLM SSD streaming full resident layers: 0 (%s)\n",
                reason);
    }

    e->ssd_streaming_full_layers = full_layers;
    e->ssd_streaming_full_layer_bytes = full_layers_bytes;
    g_glm_streaming_full_resident_start = supported_start;
    g_glm_streaming_full_resident_layers = full_layers;
    return true;
#endif
}

static bool ds4_engine_glm_streaming_memory_guard(
        const ds4_engine *e,
        bool              load_slice,
        uint32_t          load_layer_start,
        uint32_t          load_layer_end,
        bool              load_output,
        int               ctx_size,
        const char       *phase) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)load_slice;
    (void)load_layer_start;
    (void)load_layer_end;
    (void)load_output;
    (void)ctx_size;
    (void)phase;
    return true;
#else
    if (!e || DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM_DSA || ctx_size <= 0) {
        return true;
    }

    const uint64_t transient_extra_bytes =
        ds4_engine_streaming_transient_guard_bytes(e);
    if (transient_extra_bytes == 0) return true;

    uint32_t guard_ctx = 0;
    if (!glm_graph_context_request(ctx_size, &guard_ctx)) return false;
    if (load_slice) {
        return glm_graph_memory_guard_slice_with_transient(
                &e->model,
                &e->weights,
                e->ssd_streaming,
                load_layer_start,
                load_layer_end,
                load_layer_start == 0,
                load_output,
                guard_ctx,
                transient_extra_bytes,
                phase);
    }
    return glm_graph_memory_guard_with_transient(
            &e->model,
            &e->weights,
            e->ssd_streaming,
            guard_ctx,
            transient_extra_bytes,
            phase);
#endif
}

static bool ds4_engine_preload_pro_q4_expert_tables(
        ds4_engine *e,
        bool        load_slice,
        uint32_t    load_layer_start,
        uint32_t    load_layer_end) {
#ifdef DS4_NO_GPU
    (void)e;
    (void)load_slice;
    (void)load_layer_start;
    (void)load_layer_end;
    return true;
#else
    if (!e ||
        e->backend != DS4_BACKEND_METAL ||
        e->ssd_streaming ||
        DS4_MODEL_VARIANT != DS4_VARIANT_PRO ||
        getenv("DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_PRELOAD") != NULL) {
        return true;
    }
    if (!metal_graph_q4_non_streaming_opt_in_enabled()) {
        return true;
    }
    if (ds4_gpu_pro_q4_expert_table_auto_available() == 0 &&
        getenv("DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO") == NULL &&
        getenv("DS4_METAL_ENABLE_PRO_Q4_EXPERT_ADDRESS_AUTO") == NULL &&
        getenv("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") == NULL &&
        getenv("DS4_METAL_ENABLE_Q4_EXPERT_ADDRESS_TABLE") == NULL) {
        return true;
    }

    uint32_t start = load_slice ? load_layer_start : 0;
    uint32_t end = load_slice ? load_layer_end : DS4_N_LAYER - 1u;
    if (start >= DS4_N_LAYER) return true;
    if (end == UINT32_MAX || end >= DS4_N_LAYER) end = DS4_N_LAYER - 1u;
    if (end < start) return true;

    bool any = false;
    const double t0 = now_sec();
    for (uint32_t il = start; il <= end; il++) {
        const ds4_layer_weights *layer = &e->weights.layer[il];
        if (!layer->ffn_gate_exps || !layer->ffn_up_exps || !layer->ffn_down_exps) {
            continue;
        }
        if (layer->ffn_gate_exps->type != DS4_TENSOR_Q4_K ||
            layer->ffn_up_exps->type != DS4_TENSOR_Q4_K ||
            layer->ffn_down_exps->type != DS4_TENSOR_Q4_K ||
            DS4_N_EXPERT != 384 ||
            DS4_N_EXPERT_USED != 6) {
            continue;
        }

        const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
        const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
        if (layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
            layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
            fprintf(stderr, "ds4: PRO Q4 expert table preload byte size overflow at layer %u\n", il);
            return false;
        }
        const uint64_t gate_expert_bytes = layer->ffn_gate_exps->dim[1] * gate_row_bytes;
        const uint64_t down_expert_bytes = layer->ffn_down_exps->dim[1] * down_row_bytes;

        if (!ds4_gpu_preload_q4_expert_tables(e->model.map,
                                              e->model.size,
                                              layer->ffn_gate_exps->abs_offset,
                                              layer->ffn_up_exps->abs_offset,
                                              layer->ffn_down_exps->abs_offset,
                                              gate_expert_bytes,
                                              down_expert_bytes,
                                              DS4_N_EXPERT)) {
            fprintf(stderr,
                    "ds4: Metal failed to preload PRO Q4 expert tables for layer %u\n",
                    il);
            return false;
        }
        any = true;
    }

    if (any) {
        fprintf(stderr,
                "ds4: Metal preloaded PRO Q4 expert tables for layers %u:%u in %.2fs\n",
                start,
                end,
                now_sec() - t0);
    }
    return true;
#endif
}

/* TP sharding: touch the dense weights and only this rank's contiguous range
 * of every routed-expert blob, so the other range is never faulted in.
 * Replaces the whole-file residency request; the pages
 * fault through the same view buffers, streaming-style, but everything a
 * rank will ever read is pre-faulted here. */
static void model_warm_weights_sharded(const ds4_model *m,
                                       const ds4_weights *w,
                                       int rank) {
    typedef struct { uint64_t off, len; } skip_range;
    if (rank != 0 && rank != 1) return;
    skip_range *skips = xmalloc((size_t)DS4_N_LAYER * 3 * sizeof(*skips));
    uint32_t n_skips = 0;
    uint64_t skip_bytes = 0;
    for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        const ds4_tensor *exps[3] = { l->ffn_gate_exps, l->ffn_up_exps,
                                      l->ffn_down_exps };
        for (int t = 0; t < 3; t++) {
            const ds4_tensor *x = exps[t];
            if (!x || x->ndim != 3 || x->dim[2] < 2) continue;
            uint64_t in_dim = 0, out_dim = 0, row_bytes = 0;
            (void)tensor_expert_bytes(m, x, 0, &in_dim, &out_dim, &row_bytes);
            const uint64_t expert_bytes = out_dim * row_bytes;
            const uint64_t total_bytes = x->dim[2] * expert_bytes;
            const uint64_t low_bytes = (x->dim[2] / 2) * expert_bytes;
            /* Unowned range: rank 0 owns the low ids; rank 1 owns the high
             * ids and takes any odd-count remainder. */
            skips[n_skips].off = x->abs_offset + (rank == 0 ? low_bytes : 0);
            skips[n_skips].len = rank == 0 ? total_bytes - low_bytes : low_bytes;
            skip_bytes += skips[n_skips].len;
            n_skips++;
        }
    }
    /* File order should already ascend, but do not rely on it. */
    for (uint32_t i = 1; i < n_skips; i++) {
        skip_range key = skips[i];
        uint32_t j = i;
        while (j > 0 && skips[j - 1].off > key.off) {
            skips[j] = skips[j - 1];
            j--;
        }
        skips[j] = key;
    }
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint8_t *p = m->map;
    const uint64_t start = m->tensor_data_pos, end = m->size;
    fprintf(stderr,
            "ds4: warming sharded tensor pages (rank %d): %.2f of %.2f GiB\n",
            rank,
            (double)(end - start - skip_bytes) / 1073741824.0,
            (double)(end - start) / 1073741824.0);
    const double t0 = now_sec();
    volatile uint64_t checksum = 0;
    uint64_t off = start;
    uint32_t si = 0;
    /* skips are naturally sorted: layers ascend and tensors within a layer
     * are laid out in file order. */
    while (off < end) {
        uint64_t stop = end;
        while (si < n_skips && skips[si].off + skips[si].len <= off) si++;
        if (si < n_skips && skips[si].off <= off) {
            off = skips[si].off + skips[si].len;
            si++;
            continue;
        }
        if (si < n_skips && skips[si].off < stop) stop = skips[si].off;
#if defined(POSIX_MADV_WILLNEED)
        (void)posix_madvise((void *)(p + off), (size_t)(stop - off),
                            POSIX_MADV_WILLNEED);
#endif
        for (; off < stop; off += page) checksum += p[off];
        off = stop;
    }
    free(skips);
    fprintf(stderr, "ds4: sharded warm done in %.1fs (checksum=%llu)\n",
            now_sec() - t0, (unsigned long long)checksum);
}

