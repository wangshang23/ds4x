/* =========================================================================
 * Fixed Weight Binding and Model Validation.
 * =========================================================================
 *
 * The GGUF tensor directory is converted into a DS4-specific pointer table.
 * After this section, the rest of the program addresses tensors by semantic
 * fields such as layer->attn_q_a or layer->ffn_gate_exps rather than by string
 * lookup.  Shape validation is intentionally strict.
 */

static uint32_t required_u32(const ds4_model *m, const char *key) {
    uint32_t v = 0;
    if (!model_get_u32(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static uint64_t required_u64_compat(const ds4_model *m, const char *key) {
    uint64_t v = 0;
    if (!model_get_u64_compat(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static float required_f32(const ds4_model *m, const char *key) {
    float v = 0.0f;
    if (!model_get_f32_compat(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static bool required_bool(const ds4_model *m, const char *key) {
    bool v = false;
    if (!model_get_bool(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static ds4_tensor *required_tensor(const ds4_model *m, const char *name) {
    ds4_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "ds4: required tensor is missing: %s\n", name);
        exit(1);
    }
    return t;
}

static ds4_tensor *tensor_by_namef(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return model_find_tensor(m, name);
}

static ds4_tensor *required_tensorf(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return required_tensor(m, name);
}

static ds4_tensor *tensor_by_mtp_stage_suffix(
        const ds4_model *m,
        uint32_t         stage,
        const char      *suffix) {
    char name[160];
    int n = snprintf(name, sizeof(name), "mtp.%u.%s", stage, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return model_find_tensor(m, name);
}

static void tensor_expect_layout(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating layout");
    if (t->type != type) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected %s\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type),
                tensor_type_name(type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}

static bool tensor_type_is_glm_dense_quant(uint32_t type) {
    return type == DS4_TENSOR_Q8_0 ||
           type == DS4_TENSOR_Q4_K ||
           type == DS4_TENSOR_Q4_0;
}

static bool tensor_type_is_dense_quant(uint32_t type) {
    return type == DS4_TENSOR_Q8_0 ||
           type == DS4_TENSOR_Q4_K ||
           type == DS4_TENSOR_Q4_0;
}

static void tensor_expect_glm_dense_quant_layout(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating GLM dense layout");
    if (!tensor_type_is_glm_dense_quant(t->type)) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected q8_0, q4_K, or q4_0\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_dense_quant_layout(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating dense quant layout");
    if (!tensor_type_is_dense_quant(t->type)) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected q8_0, q4_K, or q4_0\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_optional(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (t) tensor_expect_layout(t, type, ndim, d0, d1, d2);
}

static void tensor_expect_plain_layout(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating layout");
    if (t->type != DS4_TENSOR_F16 && t->type != DS4_TENSOR_F32) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected F16 or F32\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static bool tensor_type_is_f16_or_q8_0(uint32_t type) {
    return type == DS4_TENSOR_F16 || type == DS4_TENSOR_Q8_0;
}

static void tensor_expect_f16_or_q8_0_layout(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating layout");
    if (!tensor_type_is_f16_or_q8_0(t->type)) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected f16 or q8_0\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static bool tensor_is_routed_expert_type(uint32_t type) {
    return type == DS4_TENSOR_Q8_0 ||
           type == DS4_TENSOR_IQ2_XXS ||
           type == DS4_TENSOR_Q2_K ||
           type == DS4_TENSOR_Q4_K ||
           type == DS4_TENSOR_Q5_K ||
           type == DS4_TENSOR_Q6_K ||
           type == DS4_TENSOR_MXFP4;
}

static DS4_MAYBE_UNUSED uint64_t routed_expert_block_bytes(uint32_t type) {
    switch (type) {
    case DS4_TENSOR_Q8_0:    return 34;
    case DS4_TENSOR_IQ2_XXS: return sizeof(block_iq2_xxs);
    case DS4_TENSOR_Q2_K:    return sizeof(block_q2_K);
    case DS4_TENSOR_Q4_K:    return sizeof(block_q4_K);
    case DS4_TENSOR_Q5_K:    return sizeof(block_q5_K);
    case DS4_TENSOR_Q6_K:    return sizeof(block_q6_K);
    case DS4_TENSOR_MXFP4:   return sizeof(block_mxfp4);
    default:                 ds4_die("unsupported routed expert tensor type");
    }
    return 0;
}

static DS4_MAYBE_UNUSED uint64_t routed_expert_row_bytes(const ds4_tensor *t) {
    const gguf_type_info *info = tensor_type(t->type);
    if (!info || info->block_elems == 0) ds4_die("unsupported routed expert tensor type");
    if ((t->dim[0] % info->block_elems) != 0) ds4_die("routed expert row is not quant block aligned");
    return (t->dim[0] / info->block_elems) * routed_expert_block_bytes(t->type);
}

static bool streaming_layer_routed_expert_bytes(
        const ds4_layer_weights *layer,
        uint64_t               *per_expert_bytes_out) {
    if (per_expert_bytes_out) *per_expert_bytes_out = 0;
    if (!layer ||
        !per_expert_bytes_out ||
        !layer->ffn_gate_exps ||
        !layer->ffn_up_exps ||
        !layer->ffn_down_exps) {
        return false;
    }

    const uint64_t gate_row_bytes =
        routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t up_row_bytes =
        routed_expert_row_bytes(layer->ffn_up_exps);
    const uint64_t down_row_bytes =
        routed_expert_row_bytes(layer->ffn_down_exps);
    if (layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
        layer->ffn_up_exps->dim[1] > UINT64_MAX / up_row_bytes ||
        layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
        return false;
    }

    const uint64_t gate_expert_bytes =
        layer->ffn_gate_exps->dim[1] * gate_row_bytes;
    const uint64_t up_expert_bytes =
        layer->ffn_up_exps->dim[1] * up_row_bytes;
    const uint64_t down_expert_bytes =
        layer->ffn_down_exps->dim[1] * down_row_bytes;
    if (gate_expert_bytes > UINT64_MAX - up_expert_bytes ||
        gate_expert_bytes + up_expert_bytes >
            UINT64_MAX - down_expert_bytes) {
        return false;
    }

    const uint64_t per_expert_bytes =
        gate_expert_bytes + up_expert_bytes + down_expert_bytes;
    if (per_expert_bytes == 0) return false;
    *per_expert_bytes_out = per_expert_bytes;
    return true;
}

static DS4_MAYBE_UNUSED bool streaming_layer_gate_down_expert_bytes(
        const ds4_layer_weights *layer,
        uint64_t               *gate_expert_bytes,
        uint64_t               *down_expert_bytes) {
    if (gate_expert_bytes) *gate_expert_bytes = 0;
    if (down_expert_bytes) *down_expert_bytes = 0;
    if (!layer ||
        !gate_expert_bytes ||
        !down_expert_bytes ||
        !layer->ffn_gate_exps ||
        !layer->ffn_down_exps) {
        return false;
    }

    const uint64_t gate_row_bytes =
        routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t down_row_bytes =
        routed_expert_row_bytes(layer->ffn_down_exps);
    if (gate_row_bytes == 0 ||
        down_row_bytes == 0 ||
        layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
        layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
        return false;
    }

    *gate_expert_bytes = layer->ffn_gate_exps->dim[1] * gate_row_bytes;
    *down_expert_bytes = layer->ffn_down_exps->dim[1] * down_row_bytes;
    return *gate_expert_bytes != 0 && *down_expert_bytes != 0;
}

static bool ds4_streaming_routed_expert_bytes(
        const ds4_weights *weights,
        uint64_t          *per_expert_bytes_out) {
    if (per_expert_bytes_out) *per_expert_bytes_out = 0;
    if (!weights || !per_expert_bytes_out) return false;

    /* Mixed-precision models can put an outlier quant at the first routed
     * layer owned by a distributed slice.  Choosing that first layer as the
     * slab class makes every ordinary layer bypass the cache.  Use the most
     * common local size class instead (ties retain the earliest class). */
    uint64_t best_bytes = 0;
    uint32_t best_count = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        uint64_t candidate = 0;
        if (!streaming_layer_routed_expert_bytes(&weights->layer[il],
                                                 &candidate)) {
            continue;
        }
        uint32_t count = 0;
        for (uint32_t jl = 0; jl < DS4_N_LAYER; jl++) {
            uint64_t bytes = 0;
            if (streaming_layer_routed_expert_bytes(&weights->layer[jl],
                                                    &bytes) &&
                bytes == candidate) {
                count++;
            }
        }
        if (count > best_count) {
            best_bytes = candidate;
            best_count = count;
        }
    }
    if (best_count == 0) return false;
    *per_expert_bytes_out = best_bytes;
    return true;
}

enum { DS4_STREAMING_PREFILL_HEADROOM_LAYERS = 2 };

static bool ds4_streaming_cacheable_expert_count(
        const ds4_weights *weights,
        uint64_t          *experts_out,
        uint32_t          *layers_out) {
    if (experts_out) *experts_out = 0;
    if (layers_out) *layers_out = 0;
    if (!weights || !experts_out || DS4_N_EXPERT == 0) return false;

    uint64_t slab_bytes = 0;
    if (!ds4_streaming_routed_expert_bytes(weights, &slab_bytes)) return false;

    uint32_t layers = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        uint64_t per_expert_bytes = 0;
        if (!streaming_layer_routed_expert_bytes(&weights->layer[il],
                                                 &per_expert_bytes)) {
            continue;
        }
        if (per_expert_bytes == slab_bytes) layers++;
    }

    if (layers == 0 ||
        (uint64_t)layers > UINT64_MAX / (uint64_t)DS4_N_EXPERT) {
        return false;
    }
    *experts_out = (uint64_t)layers * (uint64_t)DS4_N_EXPERT;
    if (layers_out) *layers_out = layers;
    return true;
}

static bool ds4_streaming_prefill_headroom_bytes(
        const ds4_weights *weights,
        uint64_t          *bytes_out) {
    if (bytes_out) *bytes_out = 0;
    if (!weights || !bytes_out) return false;

    uint64_t per_expert_bytes = 0;
    uint64_t cacheable_experts = 0;
    uint32_t cacheable_layers = 0;
    if (!ds4_streaming_routed_expert_bytes(weights, &per_expert_bytes) ||
        !ds4_streaming_cacheable_expert_count(weights,
                                              &cacheable_experts,
                                              &cacheable_layers)) {
        return false;
    }
    (void)cacheable_experts;
    const uint32_t reserve_layers =
        cacheable_layers < DS4_STREAMING_PREFILL_HEADROOM_LAYERS ?
        cacheable_layers : DS4_STREAMING_PREFILL_HEADROOM_LAYERS;
    if (per_expert_bytes > UINT64_MAX / (uint64_t)DS4_N_EXPERT) {
        return false;
    }
    const uint64_t layer_bytes =
        per_expert_bytes * (uint64_t)DS4_N_EXPERT;
    if (reserve_layers != 0 &&
        layer_bytes > UINT64_MAX / (uint64_t)reserve_layers) return false;

    *bytes_out = layer_bytes * (uint64_t)reserve_layers;
    return true;
}

/*
 * Mixed-precision ("boosted") GGUFs upcast a few layers' routed experts to a
 * bigger quant (e.g. Q4_K among IQ2 layers). The streaming expert cache is a
 * single-size-class slab allocator sized from the dominant local routed-layer
 * size class, so other layers can never be served from it: they must read
 * expert weights through the mapped-model views instead. A layer is "uniform"
 * iff its per-expert bytes match the slab class.
 */
static DS4_MAYBE_UNUSED bool weights_streaming_layer_experts_uniform(
        const ds4_weights *w,
        uint32_t           il) {
    uint64_t base = 0;
    uint64_t bytes = 0;
    if (!w || il >= DS4_N_LAYER) return true;
    const ds4_layer_weights *l = &w->layer[il];
    if (!streaming_layer_routed_expert_bytes(l, &bytes)) return true;
    if (!ds4_streaming_routed_expert_bytes(w, &base)) return true;
    return bytes == base;
}

static uint32_t ds4_streaming_cache_experts_for_byte_budget(
        const ds4_weights *weights,
        uint64_t           bytes,
        uint64_t          *per_expert_bytes_out) {
    uint64_t per_expert_bytes = 0;
    if (per_expert_bytes_out) *per_expert_bytes_out = 0;
    if (!weights ||
        bytes == 0 ||
        !ds4_streaming_routed_expert_bytes(weights, &per_expert_bytes)) {
        return 0;
    }
    if (per_expert_bytes_out) *per_expert_bytes_out = per_expert_bytes;
    return ds4_ssd_cache_experts_for_byte_budget(bytes, per_expert_bytes);
}

#ifndef DS4_NO_GPU
static ds4_gpu_stream_expert_table graph_stream_expert_table_make(
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint64_t                 gate_expert_bytes,
        uint64_t                 down_expert_bytes) {
    ds4_gpu_stream_expert_table table;
    memset(&table, 0, sizeof(table));
    if (!model || !layer) return table;
    table.model_map = model->map;
    table.model_size = model->size;
    table.layer = il;
    table.n_total_expert = DS4_N_EXPERT;
    table.gate_offset = layer->ffn_gate_exps ? layer->ffn_gate_exps->abs_offset : 0;
    table.up_offset = layer->ffn_up_exps ? layer->ffn_up_exps->abs_offset : 0;
    table.down_offset = layer->ffn_down_exps ? layer->ffn_down_exps->abs_offset : 0;
    table.gate_expert_bytes = gate_expert_bytes;
    table.down_expert_bytes = down_expert_bytes;
    return table;
}
#endif

static uint64_t ds4_streaming_manual_cache_safe_bytes(
        ds4_backend backend,
        int         ctx_size,
        uint32_t    prefill_chunk,
        bool        ssd_streaming) {
#ifdef DS4_NO_GPU
    (void)backend;
    (void)ctx_size;
    (void)prefill_chunk;
    (void)ssd_streaming;
    return 0;
#else
    const uint64_t gib = 1024ull * 1024ull * 1024ull;
    const uint64_t recommended = ds4_gpu_recommended_working_set_size();
    if (recommended == 0) return 0;

    /*
     * Explicit NGB budgets name only the routed expert cache. Keep that cache
     * below the graph backend's working-set recommendation after accounting for
     * the graph context/KV buffers. This is intentionally not an mlock-derived
     * cap: crossing too close to the recommended working set makes short
     * token-major prefill spend most of its time in VM/driver synchronization.
     */
    uint64_t target = recommended > UINT64_MAX / 7ull ?
        UINT64_MAX : (recommended * 7ull) / 8ull;
    const ds4_context_memory ctx_mem =
        ds4_context_memory_estimate_with_prefill_mode(backend,
                                                      ctx_size,
                                                      prefill_chunk,
                                                      ssd_streaming);
    uint64_t safe = 0;
    if (target > ctx_mem.total_bytes) safe = target - ctx_mem.total_bytes;
    safe = (safe / gib) * gib;
    if (safe == 0) safe = gib;
    return safe;
#endif
}

static uint64_t ds4_add_sat_u64(uint64_t a, uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

static uint64_t ds4_mul_sat_u64(uint64_t a, uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a) return UINT64_MAX;
    return a * b;
}

static double ds4_bytes_to_gib(uint64_t bytes) {
    return (double)bytes / 1073741824.0;
}

static void tensor_expect_routed_expert(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing routed expert tensor while validating layout");
    if (!tensor_is_routed_expert_type(t->type)) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %u (%s), expected a routed expert quant type\n",
                (int)t->name.len,
                t->name.ptr,
                t->type,
                tensor_type_name(t->type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}

static bool weights_have_output_head(const ds4_weights *w) {
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        return w && w->output_norm && w->output;
    }
    return w &&
           w->output_hc_base &&
           w->output_hc_fn &&
           w->output_hc_scale &&
           w->output_norm &&
           w->output;
}

static bool weights_have_partial_output_head(const ds4_weights *w) {
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        return w && (w->output_norm || w->output);
    }
    return w &&
           (w->output_hc_base ||
            w->output_hc_fn ||
            w->output_hc_scale ||
            w->output_norm ||
            w->output);
}

static bool weights_glm_dsa_layer_has_required(const ds4_layer_weights *l, uint32_t il) {
    if (!l) return false;
    if (!l->attn_norm ||
        !l->attn_q_a ||
        !l->attn_q_a_norm ||
        !l->attn_q_b ||
        !l->attn_kv_a_mqa ||
        !l->attn_kv_a_norm ||
        !l->attn_k_b ||
        !l->attn_v_b ||
        !l->attn_output ||
        !l->indexer_attn_q_b ||
        !l->indexer_attn_k ||
        !l->indexer_k_norm ||
        !l->indexer_k_norm_b ||
        !l->indexer_proj ||
        !l->ffn_norm)
    {
        return false;
    }

    if (il < DS4_N_LEADING_DENSE) {
        if (!l->ffn_gate || !l->ffn_up || !l->ffn_down) return false;
    } else {
        if (!l->ffn_gate_inp ||
            !l->ffn_exp_probs_b ||
            !l->ffn_gate_exps ||
            !l->ffn_up_exps ||
            !l->ffn_down_exps ||
            !l->ffn_gate_shexp ||
            !l->ffn_up_shexp ||
            !l->ffn_down_shexp)
        {
            return false;
        }
    }

    if (DS4_N_NEXTN_PREDICT != 0 &&
        il + DS4_N_NEXTN_PREDICT >= DS4_N_LAYER &&
        (!l->nextn_eh_proj ||
         !l->nextn_enorm ||
         !l->nextn_hnorm ||
         !l->nextn_shared_head_norm))
    {
        return false;
    }

    return true;
}

static bool weights_layer_has_required(const ds4_layer_weights *l, uint32_t il) {
    if (!l) return false;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        return weights_glm_dsa_layer_has_required(l, il);
    }
    if (!l->hc_attn_fn ||
        !l->hc_attn_scale ||
        !l->hc_attn_base ||
        !l->attn_norm ||
        !l->attn_q_a ||
        !l->attn_q_a_norm ||
        !l->attn_q_b ||
        !l->attn_kv ||
        !l->attn_kv_a_norm ||
        !l->attn_sinks ||
        !l->attn_output_a ||
        !l->attn_output_b ||
        !l->hc_ffn_fn ||
        !l->hc_ffn_scale ||
        !l->hc_ffn_base ||
        !l->ffn_norm ||
        !l->ffn_gate_inp ||
        !l->ffn_gate_exps ||
        !l->ffn_up_exps ||
        !l->ffn_down_exps ||
        !l->ffn_gate_shexp ||
        !l->ffn_up_shexp ||
        !l->ffn_down_shexp)
    {
        return false;
    }

    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ratio != 0 &&
        (!l->attn_compressor_ape ||
         !l->attn_compressor_kv ||
         !l->attn_compressor_gate ||
         !l->attn_compressor_norm))
    {
        return false;
    }
    if (ratio == 4 &&
        (!l->indexer_attn_q_b ||
         !l->indexer_proj ||
         !l->indexer_compressor_ape ||
         !l->indexer_compressor_kv ||
         !l->indexer_compressor_gate ||
         !l->indexer_compressor_norm))
    {
        return false;
    }
    if (il < DS4_N_HASH_LAYER && !l->ffn_gate_tid2eid) return false;
    return true;
}

static bool weights_layers_bound(const ds4_weights *w, uint32_t layer_start, uint32_t layer_end) {
    if (!w || layer_start >= DS4_N_LAYER) return false;
    if (layer_end == UINT32_MAX) layer_end = DS4_N_LAYER - 1u;
    if (layer_end >= DS4_N_LAYER || layer_end < layer_start) return false;
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (!weights_layer_has_required(&w->layer[il], il)) return false;
    }
    return true;
}

static const ds4_layer_weights *weights_first_bound_layer(const ds4_weights *w) {
    if (!w) return NULL;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (weights_layer_has_required(&w->layer[il], il)) return &w->layer[il];
    }
    return NULL;
}

/* Verify every tensor type and dimension used by the specialized pipeline.
 * For distributed sliced GGUFs, only the advertised local layer range is
 * required; token embedding and output head are validated when present. */
static void weights_validate_glm_dsa_layout(
        const ds4_weights *w,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               require_token_embd,
        bool               require_output) {
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA;
    const uint64_t q_nope = DS4_N_KEY_MLA - DS4_N_ROT;
    const uint64_t index_q_dim =
        (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;

    if (!w) ds4_die("internal error: missing weights while validating GLM layout");
    if (layer_start >= DS4_N_LAYER) ds4_die("invalid first layer in GLM weight layout validation");
    if (layer_end == UINT32_MAX) layer_end = DS4_N_LAYER - 1u;
    if (layer_end >= DS4_N_LAYER || layer_end < layer_start) {
        ds4_die("invalid layer range in GLM weight layout validation");
    }

    if (require_token_embd && !w->token_embd) ds4_die("required token embedding tensor is missing");
    if (w->token_embd) {
        tensor_expect_glm_dense_quant_layout(w->token_embd, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    }

    const bool have_output = weights_have_output_head(w);
    if (require_output && !have_output) ds4_die("required output head tensors are missing");
    if (weights_have_partial_output_head(w) && !have_output) ds4_die("partial output head in GGUF");
    if (have_output) {
        tensor_expect_layout(w->output_norm, DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_glm_dense_quant_layout(w->output, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    }

    for (uint32_t il = layer_start; il <= layer_end; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        if (!weights_glm_dsa_layer_has_required(l, il)) {
            fprintf(stderr, "ds4: required GLM tensors for layer %u are missing\n", il);
            exit(1);
        }

        tensor_expect_layout(l->attn_norm,       DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_glm_dense_quant_layout(l->attn_q_a,        2, DS4_N_EMBD, DS4_N_LORA_Q, 0);
        tensor_expect_layout(l->attn_q_a_norm,   DS4_TENSOR_F32,  1, DS4_N_LORA_Q, 0, 0);
        tensor_expect_glm_dense_quant_layout(l->attn_q_b,        2, DS4_N_LORA_Q, q_dim, 0);
        tensor_expect_glm_dense_quant_layout(l->attn_kv_a_mqa,   2, DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
        tensor_expect_layout(l->attn_kv_a_norm,  DS4_TENSOR_F32,  1, DS4_N_KV_LORA, 0, 0);
        tensor_expect_glm_dense_quant_layout(l->attn_k_b,        3, q_nope, DS4_N_KV_LORA, DS4_N_HEAD);
        tensor_expect_glm_dense_quant_layout(l->attn_v_b,        3, DS4_N_KV_LORA, DS4_N_VALUE_MLA, DS4_N_HEAD);
        tensor_expect_glm_dense_quant_layout(l->attn_output,     2, DS4_N_HEAD * DS4_N_VALUE_MLA, DS4_N_EMBD, 0);
        tensor_expect_glm_dense_quant_layout(l->indexer_attn_k,  2, DS4_N_EMBD, DS4_N_INDEXER_HEAD_DIM, 0);
        tensor_expect_glm_dense_quant_layout(l->indexer_attn_q_b, 2, DS4_N_LORA_Q, index_q_dim, 0);
        tensor_expect_layout(l->indexer_k_norm,  DS4_TENSOR_F32,  1, DS4_N_INDEXER_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->indexer_k_norm_b, DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->indexer_proj,    DS4_TENSOR_F32,  2, DS4_N_EMBD, DS4_N_INDEXER_HEAD, 0);
        tensor_expect_layout(l->ffn_norm,        DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);

        if (il < DS4_N_LEADING_DENSE) {
            tensor_expect_glm_dense_quant_layout(l->ffn_gate, 2, DS4_N_EMBD, DS4_N_FF_DENSE, 0);
            tensor_expect_glm_dense_quant_layout(l->ffn_up,   2, DS4_N_EMBD, DS4_N_FF_DENSE, 0);
            tensor_expect_glm_dense_quant_layout(l->ffn_down, 2, DS4_N_FF_DENSE, DS4_N_EMBD, 0);
        } else {
            tensor_expect_layout(l->ffn_gate_inp, DS4_TENSOR_F32, 2, DS4_N_EMBD, DS4_N_EXPERT, 0);
            tensor_expect_layout(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0);
            tensor_expect_routed_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
            tensor_expect_routed_expert(l->ffn_up_exps,   3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
            tensor_expect_routed_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
            if (l->ffn_gate_exps->type != l->ffn_up_exps->type) {
                fprintf(stderr, "ds4: GLM routed gate/up experts use different quant types in layer %u\n", il);
                exit(1);
            }
            tensor_expect_glm_dense_quant_layout(l->ffn_gate_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
            tensor_expect_glm_dense_quant_layout(l->ffn_up_shexp,   2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
            tensor_expect_glm_dense_quant_layout(l->ffn_down_shexp, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
        }

        if (DS4_N_NEXTN_PREDICT != 0 &&
            il + DS4_N_NEXTN_PREDICT >= DS4_N_LAYER) {
            tensor_expect_glm_dense_quant_layout(l->nextn_eh_proj, 2, 2u * DS4_N_EMBD, DS4_N_EMBD, 0);
            tensor_expect_layout(l->nextn_enorm,   DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
            tensor_expect_layout(l->nextn_hnorm,   DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
            tensor_expect_layout(l->nextn_shared_head_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0);
        }
    }
}

static void weights_validate_layout(
        const ds4_weights *w,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               require_token_embd,
        bool               require_output) {
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        weights_validate_glm_dsa_layout(w,
                                        layer_start,
                                        layer_end,
                                        require_token_embd,
                                        require_output);
        return;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;

    if (!w) ds4_die("internal error: missing weights while validating layout");
    if (layer_start >= DS4_N_LAYER) ds4_die("invalid first layer in weight layout validation");
    if (layer_end == UINT32_MAX) layer_end = DS4_N_LAYER - 1u;
    if (layer_end >= DS4_N_LAYER || layer_end < layer_start) {
        ds4_die("invalid layer range in weight layout validation");
    }

    if (require_token_embd && !w->token_embd) ds4_die("required token embedding tensor is missing");
    if (w->token_embd) {
        tensor_expect_layout(w->token_embd, DS4_TENSOR_F16, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    }

    const bool have_output = weights_have_output_head(w);
    if (require_output && !have_output) ds4_die("required output head tensors are missing");
    if (weights_have_partial_output_head(w) && !have_output) ds4_die("partial output head in GGUF");
    if (have_output) {
        tensor_expect_layout(w->output_hc_base,  DS4_TENSOR_F32,  1, DS4_N_HC, 0, 0);
        tensor_expect_layout(w->output_hc_fn,    DS4_TENSOR_F16,  2, hc_dim, DS4_N_HC, 0);
        tensor_expect_layout(w->output_hc_scale, DS4_TENSOR_F32,  1, 1, 0, 0);
        tensor_expect_layout(w->output_norm,     DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_dense_quant_layout(w->output,          2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    }

    for (uint32_t il = layer_start; il <= layer_end; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!weights_layer_has_required(l, il)) {
            fprintf(stderr, "ds4: required tensors for layer %u are missing\n", il);
            exit(1);
        }

        tensor_expect_layout(l->hc_attn_fn,     DS4_TENSOR_F16,  2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_attn_scale,  DS4_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_attn_base,   DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->attn_norm,      DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_dense_quant_layout(l->attn_q_a,       2, DS4_N_EMBD, DS4_N_LORA_Q, 0);
        tensor_expect_layout(l->attn_q_a_norm,  DS4_TENSOR_F32,  1, DS4_N_LORA_Q, 0, 0);
        tensor_expect_dense_quant_layout(l->attn_q_b,       2, DS4_N_LORA_Q, q_dim, 0);
        tensor_expect_dense_quant_layout(l->attn_kv,        2, DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
        tensor_expect_layout(l->attn_kv_a_norm, DS4_TENSOR_F32,  1, DS4_N_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->attn_sinks,     DS4_TENSOR_F32,  1, DS4_N_HEAD, 0, 0);
        tensor_expect_dense_quant_layout(l->attn_output_a,  2, DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low_dim, 0);
        tensor_expect_dense_quant_layout(l->attn_output_b,  2, out_low_dim, DS4_N_EMBD, 0);

        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t comp_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            tensor_expect_layout(l->attn_compressor_ape,  DS4_TENSOR_F16, 2, comp_width, ratio, 0);
            tensor_expect_layout(l->attn_compressor_kv,   DS4_TENSOR_F16, 2, DS4_N_EMBD, comp_width, 0);
            tensor_expect_layout(l->attn_compressor_gate, DS4_TENSOR_F16, 2, DS4_N_EMBD, comp_width, 0);
            tensor_expect_layout(l->attn_compressor_norm, DS4_TENSOR_F32, 1, DS4_N_HEAD_DIM, 0, 0);
        }
        if (ratio == 4) {
            const uint64_t index_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_width = 2u * DS4_N_INDEXER_HEAD_DIM;
            tensor_expect_f16_or_q8_0_layout(l->indexer_attn_q_b, 2, DS4_N_LORA_Q, index_q_dim, 0);
            tensor_expect_layout(l->indexer_proj,              DS4_TENSOR_F16, 2, DS4_N_EMBD, DS4_N_INDEXER_HEAD, 0);
            tensor_expect_layout(l->indexer_compressor_ape,    DS4_TENSOR_F16, 2, index_width, ratio, 0);
            tensor_expect_layout(l->indexer_compressor_kv,     DS4_TENSOR_F16, 2, DS4_N_EMBD, index_width, 0);
            tensor_expect_layout(l->indexer_compressor_gate,   DS4_TENSOR_F16, 2, DS4_N_EMBD, index_width, 0);
            tensor_expect_layout(l->indexer_compressor_norm,   DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0);
        }

        tensor_expect_layout(l->hc_ffn_fn,      DS4_TENSOR_F16,  2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_ffn_scale,   DS4_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_ffn_base,    DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->ffn_norm,       DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_layout(l->ffn_gate_inp,   DS4_TENSOR_F16,  2, DS4_N_EMBD, DS4_N_EXPERT, 0);
        tensor_expect_optional(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0);
        tensor_expect_routed_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
        tensor_expect_routed_expert(l->ffn_up_exps,   3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
        tensor_expect_routed_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
        if (l->ffn_gate_exps->type != l->ffn_up_exps->type) {
            fprintf(stderr, "ds4: routed gate/up experts use different quant types in layer %u\n", il);
            exit(1);
        }
        tensor_expect_dense_quant_layout(l->ffn_gate_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
        tensor_expect_dense_quant_layout(l->ffn_up_shexp,   2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
        tensor_expect_dense_quant_layout(l->ffn_down_shexp, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
        if (il < DS4_N_HASH_LAYER) {
            tensor_expect_layout(l->ffn_gate_tid2eid, DS4_TENSOR_I32, 2, DS4_N_EXPERT_USED, DS4_N_VOCAB, 0);
        }
    }
}

static void mtp_weights_validate_layout(const ds4_mtp_weights *w) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const ds4_layer_weights *l = &w->block;

    tensor_expect_layout(w->hc_head_base,  DS4_TENSOR_F32,  1, DS4_N_HC, 0, 0);
    tensor_expect_plain_layout(w->hc_head_fn, 2, hc_dim, DS4_N_HC, 0);
    tensor_expect_layout(w->hc_head_scale, DS4_TENSOR_F32,  1, 1, 0, 0);
    tensor_expect_layout(w->e_proj,        DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_EMBD, 0);
    tensor_expect_layout(w->h_proj,        DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_EMBD, 0);
    tensor_expect_layout(w->enorm,         DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->hnorm,         DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->norm,          DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);

    tensor_expect_plain_layout(l->hc_attn_fn, 2, hc_dim, hc_mix_dim, 0);
    tensor_expect_layout(l->hc_attn_scale,  DS4_TENSOR_F32,  1, 3, 0, 0);
    tensor_expect_layout(l->hc_attn_base,   DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
    tensor_expect_layout(l->attn_norm,      DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(l->attn_q_a,       DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_LORA_Q, 0);
    tensor_expect_layout(l->attn_q_a_norm,  DS4_TENSOR_F32,  1, DS4_N_LORA_Q, 0, 0);
    tensor_expect_layout(l->attn_q_b,       DS4_TENSOR_Q8_0, 2, DS4_N_LORA_Q, q_dim, 0);
    tensor_expect_layout(l->attn_kv,        DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
    tensor_expect_layout(l->attn_kv_a_norm, DS4_TENSOR_F32,  1, DS4_N_HEAD_DIM, 0, 0);
    tensor_expect_layout(l->attn_sinks,     DS4_TENSOR_F32,  1, DS4_N_HEAD, 0, 0);
    tensor_expect_layout(l->attn_output_a,  DS4_TENSOR_Q8_0, 2, DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low_dim, 0);
    tensor_expect_layout(l->attn_output_b,  DS4_TENSOR_Q8_0, 2, out_low_dim, DS4_N_EMBD, 0);

    tensor_expect_plain_layout(l->hc_ffn_fn, 2, hc_dim, hc_mix_dim, 0);
    tensor_expect_layout(l->hc_ffn_scale,   DS4_TENSOR_F32,  1, 3, 0, 0);
    tensor_expect_layout(l->hc_ffn_base,    DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
    tensor_expect_layout(l->ffn_norm,       DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_plain_layout(l->ffn_gate_inp, 2, DS4_N_EMBD, DS4_N_EXPERT, 0);
    tensor_expect_layout(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0);
    tensor_expect_routed_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    tensor_expect_routed_expert(l->ffn_up_exps,   3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    tensor_expect_routed_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
    if (l->ffn_gate_exps->type != l->ffn_up_exps->type) {
        ds4_die("MTP routed gate/up experts use different quant types");
    }
    tensor_expect_layout(l->ffn_gate_shexp, DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
    tensor_expect_layout(l->ffn_up_shexp,   DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
    tensor_expect_layout(l->ffn_down_shexp, DS4_TENSOR_Q8_0, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
}

typedef enum {
    DS4_DSPARK_LAYOUT_F32,
    DS4_DSPARK_LAYOUT_PLAIN,
    DS4_DSPARK_LAYOUT_DENSE,
    DS4_DSPARK_LAYOUT_ROUTED,
} ds4_dspark_layout_kind;

static const char *dspark_layout_kind_name(ds4_dspark_layout_kind kind) {
    switch (kind) {
    case DS4_DSPARK_LAYOUT_F32:    return "F32";
    case DS4_DSPARK_LAYOUT_PLAIN:  return "F16 or F32";
    case DS4_DSPARK_LAYOUT_DENSE:  return "F16, F32, or Q8_0";
    case DS4_DSPARK_LAYOUT_ROUTED: return "routed expert quant";
    }
    return "unknown";
}

static bool dspark_tensor_type_matches(uint32_t type,
                                       ds4_dspark_layout_kind kind) {
    switch (kind) {
    case DS4_DSPARK_LAYOUT_F32:
        return type == DS4_TENSOR_F32;
    case DS4_DSPARK_LAYOUT_PLAIN:
        return type == DS4_TENSOR_F16 || type == DS4_TENSOR_F32;
    case DS4_DSPARK_LAYOUT_DENSE:
        return type == DS4_TENSOR_F16 ||
               type == DS4_TENSOR_F32 ||
               type == DS4_TENSOR_Q8_0;
    case DS4_DSPARK_LAYOUT_ROUTED:
        return tensor_is_routed_expert_type(type);
    }
    return false;
}

static void dspark_validate_tensor_layout(
        ds4_dspark_weights       *dw,
        const ds4_tensor         *t,
        const char               *role,
        ds4_dspark_layout_kind    kind,
        uint32_t                  ndim,
        uint64_t                  d0,
        uint64_t                  d1,
        uint64_t                  d2) {
    if (!dw || !t) return;

    bool ok = true;
    if (!dspark_tensor_type_matches(t->type, kind)) {
        fprintf(stderr,
                "ds4: DSpark tensor %.*s (%s) has type %s, expected %s\n",
                (int)t->name.len,
                t->name.ptr,
                role,
                tensor_type_name(t->type),
                dspark_layout_kind_name(kind));
        ok = false;
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: DSpark tensor %.*s (%s) has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                role,
                t->ndim,
                ndim);
        ok = false;
    }

    const uint64_t want[3] = { d0, d1, d2 };
    const uint32_t n = t->ndim < ndim ? t->ndim : ndim;
    for (uint32_t i = 0; i < n; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: DSpark tensor %.*s (%s) has dim[%u]=%" PRIu64
                ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                role,
                i,
                t->dim[i],
                want[i]);
        ok = false;
    }
    if (!ok) dw->invalid_tensors++;
}

static void dspark_weights_note_metadata_error(
        ds4_dspark_weights *dw,
        const char         *msg) {
    if (!dw) return;
    fprintf(stderr, "ds4: DSpark metadata error: %s\n", msg);
    dw->metadata_errors++;
}

static void dspark_weights_validate_metadata(ds4_dspark_weights *dw) {
    if (!dw) return;
    if (!dw->has_block_size || dw->block_size == 0) {
        dspark_weights_note_metadata_error(dw, "missing or zero block size");
    } else if (dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE) {
        dspark_weights_note_metadata_error(dw, "block size exceeds runtime limit");
    }
    if (!dw->has_markov_rank || dw->markov_rank == 0) {
        dspark_weights_note_metadata_error(dw, "missing or zero Markov rank");
    }
    if (!dw->has_noise_token_id || dw->noise_token_id >= DS4_N_VOCAB) {
        dspark_weights_note_metadata_error(dw, "missing or out-of-range noise token");
    }
    if (!dw->has_target_layers || dw->target_layer_count == 0) {
        dspark_weights_note_metadata_error(dw, "missing target layer list");
        return;
    }

    uint32_t prev = UINT32_MAX;
    for (uint32_t i = 0; i < dw->target_layer_count; i++) {
        const uint32_t layer = dw->target_layers[i];
        if (layer >= DS4_N_LAYER) {
            dspark_weights_note_metadata_error(dw, "target layer is outside the target model");
        }
        if (i != 0 && layer <= prev) {
            dspark_weights_note_metadata_error(dw, "target layers are not strictly increasing");
        }
        prev = layer;
    }
}

static void dspark_weights_validate_block_layout(
        ds4_dspark_weights *dw,
        const ds4_layer_weights *l) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;

    dspark_validate_tensor_layout(dw, l->hc_attn_fn, "hc_attn_fn",
                                  DS4_DSPARK_LAYOUT_PLAIN, 2,
                                  hc_dim, hc_mix_dim, 0);
    dspark_validate_tensor_layout(dw, l->hc_attn_scale, "hc_attn_scale",
                                  DS4_DSPARK_LAYOUT_F32, 1, 3, 0, 0);
    dspark_validate_tensor_layout(dw, l->hc_attn_base, "hc_attn_base",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  hc_mix_dim, 0, 0);
    dspark_validate_tensor_layout(dw, l->attn_norm, "attn_norm",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_EMBD, 0, 0);
    dspark_validate_tensor_layout(dw, l->attn_q_a, "attn_q_a",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_EMBD, DS4_N_LORA_Q, 0);
    dspark_validate_tensor_layout(dw, l->attn_q_a_norm, "attn_q_a_norm",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_LORA_Q, 0, 0);
    dspark_validate_tensor_layout(dw, l->attn_q_b, "attn_q_b",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_LORA_Q, q_dim, 0);
    dspark_validate_tensor_layout(dw, l->attn_kv, "attn_kv",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
    dspark_validate_tensor_layout(dw, l->attn_kv_a_norm, "attn_kv_a_norm",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_HEAD_DIM, 0, 0);
    dspark_validate_tensor_layout(dw, l->attn_sinks, "attn_sinks",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_HEAD, 0, 0);
    dspark_validate_tensor_layout(dw, l->attn_output_a, "attn_output_a",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP),
                                  out_low_dim, 0);
    dspark_validate_tensor_layout(dw, l->attn_output_b, "attn_output_b",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  out_low_dim, DS4_N_EMBD, 0);

    dspark_validate_tensor_layout(dw, l->hc_ffn_fn, "hc_ffn_fn",
                                  DS4_DSPARK_LAYOUT_PLAIN, 2,
                                  hc_dim, hc_mix_dim, 0);
    dspark_validate_tensor_layout(dw, l->hc_ffn_scale, "hc_ffn_scale",
                                  DS4_DSPARK_LAYOUT_F32, 1, 3, 0, 0);
    dspark_validate_tensor_layout(dw, l->hc_ffn_base, "hc_ffn_base",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  hc_mix_dim, 0, 0);
    dspark_validate_tensor_layout(dw, l->ffn_norm, "ffn_norm",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_EMBD, 0, 0);
    dspark_validate_tensor_layout(dw, l->ffn_gate_inp, "ffn_gate_inp",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_EMBD, DS4_N_EXPERT, 0);
    dspark_validate_tensor_layout(dw, l->ffn_exp_probs_b, "exp_probs_b",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_EXPERT, 0, 0);
    dspark_validate_tensor_layout(dw, l->ffn_gate_exps, "ffn_gate_exps",
                                  DS4_DSPARK_LAYOUT_ROUTED, 3,
                                  DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    dspark_validate_tensor_layout(dw, l->ffn_up_exps, "ffn_up_exps",
                                  DS4_DSPARK_LAYOUT_ROUTED, 3,
                                  DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    dspark_validate_tensor_layout(dw, l->ffn_down_exps, "ffn_down_exps",
                                  DS4_DSPARK_LAYOUT_ROUTED, 3,
                                  DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
    if (l->ffn_gate_exps &&
        l->ffn_up_exps &&
        l->ffn_gate_exps->type != l->ffn_up_exps->type) {
        fprintf(stderr,
                "ds4: DSpark routed gate/up experts use different quant types\n");
        dw->invalid_tensors++;
    }
    dspark_validate_tensor_layout(dw, l->ffn_gate_shexp, "ffn_gate_shexp",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_EMBD, DS4_N_FF_EXP, 0);
    dspark_validate_tensor_layout(dw, l->ffn_up_shexp, "ffn_up_shexp",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_EMBD, DS4_N_FF_EXP, 0);
    dspark_validate_tensor_layout(dw, l->ffn_down_shexp, "ffn_down_shexp",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_FF_EXP, DS4_N_EMBD, 0);
}

static void dspark_weights_validate_layout(ds4_dspark_weights *dw) {
    if (!dw) return;
    dspark_weights_validate_metadata(dw);

    for (uint32_t stage = 0; stage < dw->n_stages; stage++) {
        ds4_dspark_stage_weights *sw = &dw->stage[stage];
        dspark_weights_validate_block_layout(dw, &sw->block);
        if (stage == 0) {
            dspark_validate_tensor_layout(dw, sw->main_proj, "main_proj",
                                          DS4_DSPARK_LAYOUT_DENSE, 2,
                                          (uint64_t)dw->target_layer_count *
                                              DS4_N_EMBD,
                                          DS4_N_EMBD, 0);
            dspark_validate_tensor_layout(dw, sw->main_norm, "main_norm",
                                          DS4_DSPARK_LAYOUT_F32, 1,
                                          DS4_N_EMBD, 0, 0);
        }
    }

    if (dw->n_stages == 0) return;
    ds4_dspark_stage_weights *final = &dw->stage[dw->n_stages - 1u];
    dspark_validate_tensor_layout(dw, final->norm, "norm",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_EMBD, 0, 0);
    dspark_validate_tensor_layout(dw, final->hc_head_base, "hc_head_base",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_HC, 0, 0);
    dspark_validate_tensor_layout(dw, final->hc_head_fn, "hc_head_fn",
                                  DS4_DSPARK_LAYOUT_PLAIN, 2,
                                  (uint64_t)DS4_N_EMBD * DS4_N_HC,
                                  DS4_N_HC, 0);
    dspark_validate_tensor_layout(dw, final->hc_head_scale, "hc_head_scale",
                                  DS4_DSPARK_LAYOUT_F32, 1, 1, 0, 0);
    dspark_validate_tensor_layout(dw, final->markov_w1, "markov_w1",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  dw->markov_rank, DS4_N_VOCAB, 0);
    dspark_validate_tensor_layout(dw, final->markov_w2, "markov_w2",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  dw->markov_rank, DS4_N_VOCAB, 0);
    dspark_validate_tensor_layout(dw, final->confidence_proj,
                                  "confidence_proj",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  (uint64_t)DS4_N_EMBD + dw->markov_rank,
                                  1, 0);
}

static bool ds4_shape_matches_metadata(
        const ds4_shape *s,
        uint32_t n_layer,
        uint32_t n_embd,
        uint32_t n_vocab,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_head_dim,
        uint32_t n_value_dim,
        uint32_t n_rot,
        uint32_t n_lora_q,
        uint32_t n_lora_o,
        uint32_t n_out_group,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_ff_exp,
        uint32_t n_expert_shared,
        uint32_t n_hash_layer,
        uint32_t n_swa,
        uint32_t n_indexer_head,
        uint32_t n_indexer_head_dim,
        uint32_t n_indexer_top_k,
        uint32_t n_hc,
        uint32_t n_hc_sinkhorn_iter) {
    return s->n_layer == n_layer &&
           s->n_embd == n_embd &&
           s->n_vocab == n_vocab &&
           s->n_head == n_head &&
           s->n_head_kv == n_head_kv &&
           s->n_head_dim == n_head_dim &&
           s->n_value_dim == n_value_dim &&
           s->n_rot == n_rot &&
           s->n_lora_q == n_lora_q &&
           s->n_lora_o == n_lora_o &&
           s->n_out_group == n_out_group &&
           s->n_expert == n_expert &&
           s->n_expert_used == n_expert_used &&
           s->n_ff_exp == n_ff_exp &&
           s->n_expert_shared == n_expert_shared &&
           s->n_hash_layer == n_hash_layer &&
           s->n_swa == n_swa &&
           s->n_indexer_head == n_indexer_head &&
           s->n_indexer_head_dim == n_indexer_head_dim &&
           s->n_indexer_top_k == n_indexer_top_k &&
           s->n_hc == n_hc &&
           s->n_hc_sinkhorn_iter == n_hc_sinkhorn_iter;
}

static void ds4_select_shape_from_metadata(
        uint32_t n_layer,
        uint32_t n_embd,
        uint32_t n_vocab,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_head_dim,
        uint32_t n_value_dim,
        uint32_t n_rot,
        uint32_t n_lora_q,
        uint32_t n_lora_o,
        uint32_t n_out_group,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_ff_exp,
        uint32_t n_expert_shared,
        uint32_t n_hash_layer,
        uint32_t n_swa,
        uint32_t n_indexer_head,
        uint32_t n_indexer_head_dim,
        uint32_t n_indexer_top_k,
        uint32_t n_hc,
        uint32_t n_hc_sinkhorn_iter) {
    if (ds4_shape_matches_metadata(&DS4_SHAPE_FLASH,
                                   n_layer, n_embd, n_vocab, n_head, n_head_kv,
                                   n_head_dim, n_value_dim, n_rot, n_lora_q,
                                   n_lora_o, n_out_group, n_expert,
                                   n_expert_used, n_ff_exp, n_expert_shared,
                                   n_hash_layer, n_swa, n_indexer_head,
                                   n_indexer_head_dim, n_indexer_top_k, n_hc,
                                   n_hc_sinkhorn_iter)) {
        g_ds4_shape = DS4_SHAPE_FLASH;
        return;
    }
    if (ds4_shape_matches_metadata(&DS4_SHAPE_PRO,
                                   n_layer, n_embd, n_vocab, n_head, n_head_kv,
                                   n_head_dim, n_value_dim, n_rot, n_lora_q,
                                   n_lora_o, n_out_group, n_expert,
                                   n_expert_used, n_ff_exp, n_expert_shared,
                                   n_hash_layer, n_swa, n_indexer_head,
                                   n_indexer_head_dim, n_indexer_top_k, n_hc,
                                   n_hc_sinkhorn_iter)) {
        g_ds4_shape = DS4_SHAPE_PRO;
        return;
    }

    fprintf(stderr,
            "ds4: unsupported DeepSeek4 shape: layers=%u embd=%u heads=%u "
            "q_lora=%u out_groups=%u experts=%u ff_exp=%u indexer_top_k=%u\n",
            n_layer,
            n_embd,
            n_head,
            n_lora_q,
            n_out_group,
            n_expert,
            n_ff_exp,
            n_indexer_top_k);
    exit(1);
}

static void validate_compress_ratio_metadata(const ds4_model *m) {
    const char *key = "deepseek4.attention.compress_ratios";
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)) {
        fprintf(stderr, "ds4: required int32/uint32 array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < DS4_N_LAYER) {
        ds4_die("deepseek4.attention.compress_ratios is shorter than the layer count");
    }

    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        uint32_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &got)) ds4_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            if (v < 0) ds4_die("metadata array contains a negative value");
            got = (uint32_t)v;
        }

        const uint32_t expected = ds4_expected_layer_compress_ratio(il);
        if (got != expected) {
            fprintf(stderr,
                    "ds4: unexpected DeepSeek4 compression ratio at layer %u for %s: got %u, expected %u\n",
                    il, DS4_MODEL_SHAPE_NAME, got, expected);
            exit(1);
        }
        g_ds4_compress_ratios[il] = got;
    }
}

static void config_expect_f32(const char *name, float got, float expected);

static void validate_swiglu_clamp_metadata(const ds4_model *m) {
    const char *key = "deepseek4.swiglu_clamp_exp";
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_FLOAT32 && arr.type != GGUF_VALUE_FLOAT64)) {
        fprintf(stderr, "ds4: required float array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < DS4_N_LAYER) {
        ds4_die("deepseek4.swiglu_clamp_exp is shorter than the layer count");
    }

    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t i = 0; i < DS4_N_LAYER; i++) {
        float got = 0.0f;
        if (arr.type == GGUF_VALUE_FLOAT32) {
            if (!cursor_read(&c, &got, sizeof(got))) ds4_die(c.error);
        } else {
            double v = 0.0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            got = (float)v;
        }
        config_expect_f32("swiglu_clamp_exp", got, DS4_SWIGLU_CLAMP_EXP);
    }
}

static void config_expect_u32(const char *name, uint32_t got, uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%u for %s, got %u\n",
            name, expected, DS4_MODEL_SHAPE_NAME, got);
    exit(1);
}

static void config_expect_u64(const char *name, uint64_t got, uint64_t expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%" PRIu64 " for %s, got %" PRIu64 "\n",
            name, expected, DS4_MODEL_SHAPE_NAME, got);
    exit(1);
}

static void config_expect_f32(const char *name, float got, float expected) {
    const float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-6f) return;
    fprintf(stderr, "ds4: expected %s=%.9g for %s, got %.9g\n",
            name, (double)expected, DS4_MODEL_SHAPE_NAME, (double)got);
    exit(1);
}

static void config_expect_bool(const char *name, bool got, bool expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%s for %s, got %s\n",
            name, expected ? "true" : "false", DS4_MODEL_SHAPE_NAME, got ? "true" : "false");
    exit(1);
}

static void config_validate_fixed_shape(uint32_t n_layer) {
    config_expect_u32("block_count",                  n_layer,                 DS4_N_LAYER);
}

/* Validate metadata values that affect semantics: attention shape, HC count,
 * expert routing, RoPE scaling, compression ratios, and SwiGLU clamp. */
static void config_validate_deepseek4_model(const ds4_model *m) {
    const uint32_t n_layer = required_u32(m, "deepseek4.block_count");
    const uint32_t n_embd = required_u32(m, "deepseek4.embedding_length");
    const uint32_t n_vocab = required_u32(m, "deepseek4.vocab_size");
    const uint32_t n_head = required_u32(m, "deepseek4.attention.head_count");
    const uint32_t n_head_kv = required_u32(m, "deepseek4.attention.head_count_kv");
    const uint32_t n_head_dim = required_u32(m, "deepseek4.attention.key_length");
    const uint32_t n_value_dim = required_u32(m, "deepseek4.attention.value_length");
    const uint32_t n_rot = required_u32(m, "deepseek4.rope.dimension_count");
    const uint32_t n_lora_q = required_u32(m, "deepseek4.attention.q_lora_rank");
    const uint32_t n_lora_o = required_u32(m, "deepseek4.attention.output_lora_rank");
    const uint32_t n_out_group = required_u32(m, "deepseek4.attention.output_group_count");
    const uint32_t n_expert = required_u32(m, "deepseek4.expert_count");
    const uint32_t n_expert_used = required_u32(m, "deepseek4.expert_used_count");
    const uint32_t n_ff_exp = required_u32(m, "deepseek4.expert_feed_forward_length");
    const uint32_t n_expert_shared = required_u32(m, "deepseek4.expert_shared_count");
    const uint32_t n_hash_layer = required_u32(m, "deepseek4.hash_layer_count");
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups);
    model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used);
    const uint32_t n_swa = required_u32(m, "deepseek4.attention.sliding_window");
    const uint32_t n_indexer_head = required_u32(m, "deepseek4.attention.indexer.head_count");
    const uint32_t n_indexer_head_dim = required_u32(m, "deepseek4.attention.indexer.key_length");
    const uint32_t n_indexer_top_k = required_u32(m, "deepseek4.attention.indexer.top_k");
    const uint32_t n_hc = required_u32(m, "deepseek4.hyper_connection.count");
    const uint32_t n_hc_sinkhorn_iter = required_u32(m, "deepseek4.hyper_connection.sinkhorn_iterations");

    ds4_select_shape_from_metadata(n_layer,
                                   n_embd,
                                   n_vocab,
                                   n_head,
                                   n_head_kv,
                                   n_head_dim,
                                   n_value_dim,
                                   n_rot,
                                   n_lora_q,
                                   n_lora_o,
                                   n_out_group,
                                   n_expert,
                                   n_expert_used,
                                   n_ff_exp,
                                   n_expert_shared,
                                   n_hash_layer,
                                   n_swa,
                                   n_indexer_head,
                                   n_indexer_head_dim,
                                   n_indexer_top_k,
                                   n_hc,
                                   n_hc_sinkhorn_iter);

    config_expect_u32("embedding_length",            n_embd,         DS4_N_EMBD);
    config_expect_u32("vocab_size",                  n_vocab,        DS4_N_VOCAB);
    config_expect_u32("attention.head_count",        n_head,         DS4_N_HEAD);
    config_expect_u32("attention.key_length",        n_head_dim,     DS4_N_HEAD_DIM);
    config_expect_u32("attention.head_count_kv",     n_head_kv,      DS4_N_HEAD_KV);
    config_expect_u32("attention.value_length",      n_value_dim,    DS4_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count",        n_rot,          DS4_N_ROT);
    config_expect_u32("attention.output_group_count", n_out_group,    DS4_N_OUT_GROUP);
    config_expect_u32("attention.q_lora_rank",       n_lora_q,        DS4_N_LORA_Q);
    config_expect_u32("attention.output_lora_rank",  n_lora_o,        DS4_N_LORA_O);
    config_expect_u32("expert_count",               n_expert,        DS4_N_EXPERT);
    config_expect_u32("expert_used_count",          n_expert_used,   DS4_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length", n_ff_exp,        DS4_N_FF_EXP);
    config_expect_u32("expert_shared_count",         n_expert_shared, DS4_N_EXPERT_SHARED);
    config_expect_u32("hash_layer_count",            n_hash_layer,    DS4_N_HASH_LAYER);
    config_expect_u32("expert_group_count",         n_expert_groups, 0);
    config_expect_u32("expert_group_used_count",    n_group_used,    0);

    config_expect_u32("attention.sliding_window",     n_swa,                   DS4_N_SWA);
    config_expect_u32("attention.indexer.head_count", n_indexer_head,     DS4_N_INDEXER_HEAD);
    config_expect_u32("attention.indexer.key_length", n_indexer_head_dim, DS4_N_INDEXER_HEAD_DIM);
    config_expect_u32("attention.indexer.top_k",      n_indexer_top_k,    DS4_N_INDEXER_TOP_K);
    config_expect_u32("hyper_connection.count", n_hc, DS4_N_HC);
    config_expect_u32("hyper_connection.sinkhorn_iterations", n_hc_sinkhorn_iter, DS4_N_HC_SINKHORN_ITER);

    config_validate_fixed_shape(n_layer);
    validate_compress_ratio_metadata(m);

    validate_swiglu_clamp_metadata(m);

    uint64_t rope_orig_ctx = DS4_ROPE_ORIG_CTX;
    model_get_u64_compat(m, "deepseek4.rope.scaling.original_context_length", &rope_orig_ctx);
    if (rope_orig_ctx != DS4_ROPE_ORIG_CTX) {
        fprintf(stderr, "ds4: expected rope.scaling.original_context_length=%" PRIu64
                " for %s, got %" PRIu64 "\n",
                (uint64_t)DS4_ROPE_ORIG_CTX, DS4_MODEL_SHAPE_NAME, rope_orig_ctx);
        exit(1);
    }
    const float rope_freq_base = required_f32(m, "deepseek4.rope.freq_base");
    config_expect_f32("rope.freq_base", rope_freq_base, DS4_ROPE_FREQ_BASE);
    float rope_scale_factor = DS4_ROPE_SCALE_FACTOR;
    model_get_f32_compat(m, "deepseek4.rope.scaling.factor", &rope_scale_factor);
    config_expect_f32("rope.scaling.factor", rope_scale_factor, DS4_ROPE_SCALE_FACTOR);
    float rope_yarn_beta_fast = DS4_ROPE_YARN_BETA_FAST;
    model_get_f32_compat(m, "deepseek4.rope.scaling.yarn_beta_fast", &rope_yarn_beta_fast);
    config_expect_f32("rope.scaling.yarn_beta_fast", rope_yarn_beta_fast, DS4_ROPE_YARN_BETA_FAST);
    float rope_yarn_beta_slow = DS4_ROPE_YARN_BETA_SLOW;
    model_get_f32_compat(m, "deepseek4.rope.scaling.yarn_beta_slow", &rope_yarn_beta_slow);
    config_expect_f32("rope.scaling.yarn_beta_slow", rope_yarn_beta_slow, DS4_ROPE_YARN_BETA_SLOW);
    const float compress_rope_freq_base = required_f32(m, "deepseek4.attention.compress_rope_freq_base");
    config_expect_f32("attention.compress_rope_freq_base", compress_rope_freq_base, DS4_COMPRESS_ROPE_FREQ_BASE);
    const float expert_weight_scale = required_f32(m, "deepseek4.expert_weights_scale");
    config_expect_f32("expert_weights_scale", expert_weight_scale, DS4_EXPERT_WEIGHT_SCALE);
    const float rms_eps = required_f32(m, "deepseek4.attention.layer_norm_rms_epsilon");
    config_expect_f32("attention.layer_norm_rms_epsilon", rms_eps, DS4_RMS_EPS);
    const float hc_eps = required_f32(m, "deepseek4.hyper_connection.epsilon");
    config_expect_f32("hyper_connection.epsilon", hc_eps, DS4_HC_EPS);
    const bool expert_weight_norm = required_bool(m, "deepseek4.expert_weights_norm");
    config_expect_bool("expert_weights_norm", expert_weight_norm, true);
}

static void config_validate_glm_dsa_model(const ds4_model *m) {
    g_ds4_shape = DS4_SHAPE_GLM52;
    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));

    const uint32_t n_layer = required_u32(m, "glm-dsa.block_count");
    const uint64_t n_ctx = required_u64_compat(m, "glm-dsa.context_length");
    const uint32_t n_embd = required_u32(m, "glm-dsa.embedding_length");
    const uint32_t n_vocab = required_u32(m, "glm-dsa.vocab_size");
    const uint32_t n_ff_dense = required_u32(m, "glm-dsa.feed_forward_length");
    const uint32_t n_head = required_u32(m, "glm-dsa.attention.head_count");
    const uint32_t n_head_kv = required_u32(m, "glm-dsa.attention.head_count_kv");
    const uint32_t n_head_dim = required_u32(m, "glm-dsa.attention.key_length");
    const uint32_t n_value_dim = required_u32(m, "glm-dsa.attention.value_length");
    const uint32_t n_rot = required_u32(m, "glm-dsa.rope.dimension_count");
    const uint32_t n_lora_q = required_u32(m, "glm-dsa.attention.q_lora_rank");
    const uint32_t n_kv_lora = required_u32(m, "glm-dsa.attention.kv_lora_rank");
    const uint32_t n_key_mla = required_u32(m, "glm-dsa.attention.key_length_mla");
    const uint32_t n_value_mla = required_u32(m, "glm-dsa.attention.value_length_mla");
    const uint32_t n_expert = required_u32(m, "glm-dsa.expert_count");
    const uint32_t n_expert_used = required_u32(m, "glm-dsa.expert_used_count");
    const uint32_t n_ff_exp = required_u32(m, "glm-dsa.expert_feed_forward_length");
    const uint32_t n_expert_shared = required_u32(m, "glm-dsa.expert_shared_count");
    const uint32_t n_expert_group = required_u32(m, "glm-dsa.expert_group_count");
    const uint32_t n_expert_group_used = required_u32(m, "glm-dsa.expert_group_used_count");
    const uint32_t expert_gating_func = required_u32(m, "glm-dsa.expert_gating_func");
    const uint32_t n_leading_dense = required_u32(m, "glm-dsa.leading_dense_block_count");
    const uint32_t n_nextn = required_u32(m, "glm-dsa.nextn_predict_layers");
    const uint32_t n_indexer_head = required_u32(m, "glm-dsa.attention.indexer.head_count");
    const uint32_t n_indexer_head_dim = required_u32(m, "glm-dsa.attention.indexer.key_length");
    const uint32_t n_indexer_top_k = required_u32(m, "glm-dsa.attention.indexer.top_k");

    config_expect_u32("block_count", n_layer, DS4_N_LAYER);
    config_expect_u64("context_length", n_ctx, DS4_ROPE_ORIG_CTX);
    config_expect_u32("embedding_length", n_embd, DS4_N_EMBD);
    config_expect_u32("vocab_size", n_vocab, DS4_N_VOCAB);
    config_expect_u32("feed_forward_length", n_ff_dense, DS4_N_FF_DENSE);
    config_expect_u32("attention.head_count", n_head, DS4_N_HEAD);
    config_expect_u32("attention.head_count_kv", n_head_kv, DS4_N_HEAD_KV);
    config_expect_u32("attention.key_length", n_head_dim, DS4_N_HEAD_DIM);
    config_expect_u32("attention.value_length", n_value_dim, DS4_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count", n_rot, DS4_N_ROT);
    config_expect_u32("attention.q_lora_rank", n_lora_q, DS4_N_LORA_Q);
    config_expect_u32("attention.kv_lora_rank", n_kv_lora, DS4_N_KV_LORA);
    config_expect_u32("attention.key_length_mla", n_key_mla, DS4_N_KEY_MLA);
    config_expect_u32("attention.value_length_mla", n_value_mla, DS4_N_VALUE_MLA);
    config_expect_u32("expert_count", n_expert, DS4_N_EXPERT);
    config_expect_u32("expert_used_count", n_expert_used, DS4_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length", n_ff_exp, DS4_N_FF_EXP);
    config_expect_u32("expert_shared_count", n_expert_shared, DS4_N_EXPERT_SHARED);
    config_expect_u32("expert_group_count", n_expert_group, 1);
    config_expect_u32("expert_group_used_count", n_expert_group_used, 1);
    config_expect_u32("expert_gating_func", expert_gating_func, 2);
    config_expect_u32("leading_dense_block_count", n_leading_dense, DS4_N_LEADING_DENSE);
    config_expect_u32("nextn_predict_layers", n_nextn, DS4_N_NEXTN_PREDICT);
    config_expect_u32("attention.indexer.head_count", n_indexer_head, DS4_N_INDEXER_HEAD);
    config_expect_u32("attention.indexer.key_length", n_indexer_head_dim, DS4_N_INDEXER_HEAD_DIM);
    config_expect_u32("attention.indexer.top_k", n_indexer_top_k, DS4_N_INDEXER_TOP_K);

    const float rope_freq_base = required_f32(m, "glm-dsa.rope.freq_base");
    config_expect_f32("rope.freq_base", rope_freq_base, DS4_ROPE_FREQ_BASE);
    const float rms_eps = required_f32(m, "glm-dsa.attention.layer_norm_rms_epsilon");
    config_expect_f32("attention.layer_norm_rms_epsilon", rms_eps, DS4_RMS_EPS);
    const float expert_weight_scale = required_f32(m, "glm-dsa.expert_weights_scale");
    config_expect_f32("expert_weights_scale", expert_weight_scale, DS4_EXPERT_WEIGHT_SCALE);
    const bool expert_weight_norm = required_bool(m, "glm-dsa.expert_weights_norm");
    config_expect_bool("expert_weights_norm", expert_weight_norm, true);
}

static void config_validate_model(const ds4_model *m) {
    ds4_str arch = {0};
    if (model_get_string(m, "general.architecture", &arch) &&
        ds4_streq(arch, "glm-dsa")) {
        config_validate_glm_dsa_model(m);
        return;
    }
    config_validate_deepseek4_model(m);
}

static void weights_bind_output(
        ds4_weights     *w,
        const ds4_model *m,
        bool             required,
        bool             optional) {
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        if (required) {
            w->output_norm = required_tensor(m, "output_norm.weight");
            w->output      = required_tensor(m, "output.weight");
        } else if (optional) {
            w->output_norm = model_find_tensor(m, "output_norm.weight");
            w->output      = model_find_tensor(m, "output.weight");
        }
    } else if (required) {
        w->output_hc_base   = required_tensor(m, "output_hc_base.weight");
        w->output_hc_fn     = required_tensor(m, "output_hc_fn.weight");
        w->output_hc_scale  = required_tensor(m, "output_hc_scale.weight");
        w->output_norm      = required_tensor(m, "output_norm.weight");
        w->output           = required_tensor(m, "output.weight");
    } else if (optional) {
        w->output_hc_base   = model_find_tensor(m, "output_hc_base.weight");
        w->output_hc_fn     = model_find_tensor(m, "output_hc_fn.weight");
        w->output_hc_scale  = model_find_tensor(m, "output_hc_scale.weight");
        w->output_norm      = model_find_tensor(m, "output_norm.weight");
        w->output           = model_find_tensor(m, "output.weight");
    }

    if (optional &&
        weights_have_partial_output_head(w) &&
        !weights_have_output_head(w)) {
        ds4_die("partial output head in GGUF");
    }
}

static void weights_bind_glm_dsa_layer(ds4_layer_weights *l, const ds4_model *m, uint32_t il) {
    l->attn_norm       = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    l->attn_q_a        = required_tensorf(m, "blk.%u.attn_q_a.weight", il);
    l->attn_q_a_norm   = required_tensorf(m, "blk.%u.attn_q_a_norm.weight", il);
    l->attn_q_b        = required_tensorf(m, "blk.%u.attn_q_b.weight", il);
    l->attn_kv_a_mqa   = required_tensorf(m, "blk.%u.attn_kv_a_mqa.weight", il);
    l->attn_kv_a_norm  = required_tensorf(m, "blk.%u.attn_kv_a_norm.weight", il);
    l->attn_k_b        = required_tensorf(m, "blk.%u.attn_k_b.weight", il);
    l->attn_v_b        = required_tensorf(m, "blk.%u.attn_v_b.weight", il);
    l->attn_output     = required_tensorf(m, "blk.%u.attn_output.weight", il);
    l->indexer_attn_q_b = required_tensorf(m, "blk.%u.indexer.attn_q_b.weight", il);
    l->indexer_attn_k  = required_tensorf(m, "blk.%u.indexer.attn_k.weight", il);
    l->indexer_k_norm  = required_tensorf(m, "blk.%u.indexer.k_norm.weight", il);
    l->indexer_k_norm_b = required_tensorf(m, "blk.%u.indexer.k_norm.bias", il);
    l->indexer_proj    = required_tensorf(m, "blk.%u.indexer.proj.weight", il);
    l->ffn_norm        = required_tensorf(m, "blk.%u.ffn_norm.weight", il);

    if (il < DS4_N_LEADING_DENSE) {
        l->ffn_gate = required_tensorf(m, "blk.%u.ffn_gate.weight", il);
        l->ffn_up   = required_tensorf(m, "blk.%u.ffn_up.weight", il);
        l->ffn_down = required_tensorf(m, "blk.%u.ffn_down.weight", il);
    } else {
        l->ffn_gate_inp    = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
        l->ffn_exp_probs_b = required_tensorf(m, "blk.%u.exp_probs_b.bias", il);
        l->ffn_gate_exps   = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
        l->ffn_up_exps     = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
        l->ffn_down_exps   = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
        l->ffn_gate_shexp  = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
        l->ffn_up_shexp    = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
        l->ffn_down_shexp  = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);
    }

    if (DS4_N_NEXTN_PREDICT != 0 &&
        il + DS4_N_NEXTN_PREDICT >= DS4_N_LAYER) {
        l->nextn_eh_proj = required_tensorf(m, "blk.%u.nextn.eh_proj.weight", il);
        l->nextn_enorm = required_tensorf(m, "blk.%u.nextn.enorm.weight", il);
        l->nextn_hnorm = required_tensorf(m, "blk.%u.nextn.hnorm.weight", il);
        l->nextn_shared_head_norm =
            required_tensorf(m, "blk.%u.nextn.shared_head_norm.weight", il);
    }
}

static void weights_bind_layer(ds4_layer_weights *l, const ds4_model *m, uint32_t il) {
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        weights_bind_glm_dsa_layer(l, m, il);
        return;
    }

    const uint32_t compress_ratio = ds4_layer_compress_ratio(il);

    l->hc_attn_fn      = required_tensorf(m, "blk.%u.hc_attn_fn.weight", il);
    l->hc_attn_scale   = required_tensorf(m, "blk.%u.hc_attn_scale.weight", il);
    l->hc_attn_base    = required_tensorf(m, "blk.%u.hc_attn_base.weight", il);
    l->attn_norm       = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    l->attn_q_a        = required_tensorf(m, "blk.%u.attn_q_a.weight", il);
    l->attn_q_a_norm   = required_tensorf(m, "blk.%u.attn_q_a_norm.weight", il);
    l->attn_q_b        = required_tensorf(m, "blk.%u.attn_q_b.weight", il);
    l->attn_kv         = required_tensorf(m, "blk.%u.attn_kv.weight", il);
    l->attn_kv_a_norm  = required_tensorf(m, "blk.%u.attn_kv_a_norm.weight", il);
    l->attn_sinks      = required_tensorf(m, "blk.%u.attn_sinks.weight", il);
    l->attn_output_a   = required_tensorf(m, "blk.%u.attn_output_a.weight", il);
    l->attn_output_b   = required_tensorf(m, "blk.%u.attn_output_b.weight", il);
    if (compress_ratio != 0) {
        l->attn_compressor_ape  = required_tensorf(m, "blk.%u.attn_compressor_ape.weight", il);
        l->attn_compressor_kv   = required_tensorf(m, "blk.%u.attn_compressor_kv.weight", il);
        l->attn_compressor_gate = required_tensorf(m, "blk.%u.attn_compressor_gate.weight", il);
        l->attn_compressor_norm = required_tensorf(m, "blk.%u.attn_compressor_norm.weight", il);
    }
    if (compress_ratio == 4) {
        l->indexer_attn_q_b = required_tensorf(m, "blk.%u.indexer.attn_q_b.weight", il);
        l->indexer_proj     = required_tensorf(m, "blk.%u.indexer.proj.weight", il);
        l->indexer_compressor_ape  = required_tensorf(m, "blk.%u.indexer_compressor_ape.weight", il);
        l->indexer_compressor_kv   = required_tensorf(m, "blk.%u.indexer_compressor_kv.weight", il);
        l->indexer_compressor_gate = required_tensorf(m, "blk.%u.indexer_compressor_gate.weight", il);
        l->indexer_compressor_norm = required_tensorf(m, "blk.%u.indexer_compressor_norm.weight", il);
    }
    l->hc_ffn_fn       = required_tensorf(m, "blk.%u.hc_ffn_fn.weight", il);
    l->hc_ffn_scale    = required_tensorf(m, "blk.%u.hc_ffn_scale.weight", il);
    l->hc_ffn_base     = required_tensorf(m, "blk.%u.hc_ffn_base.weight", il);
    l->ffn_norm        = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
    l->ffn_gate_inp    = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
    l->ffn_exp_probs_b = tensor_by_namef(m, "blk.%u.exp_probs_b.bias", il);
    l->ffn_gate_exps   = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
    l->ffn_up_exps     = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
    l->ffn_down_exps   = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
    l->ffn_gate_shexp  = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
    l->ffn_up_shexp    = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
    l->ffn_down_shexp  = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);

    if (il < DS4_N_HASH_LAYER) {
        l->ffn_gate_tid2eid = required_tensorf(m, "blk.%u.ffn_gate_tid2eid.weight", il);
    }
}

/* Bind tensor names once into the fixed DS4 layer layout.  This is the point
 * where stringly GGUF metadata becomes direct model-specific pointers. */
static void weights_bind(
        ds4_weights     *w,
        const ds4_model *m,
        bool             load_slice,
        uint32_t         load_layer_start,
        uint32_t         load_layer_end,
        bool             require_output,
        bool             optional_output) {
    memset(w, 0, sizeof(*w));

    uint32_t executable_layers = DS4_N_LAYER;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA &&
        DS4_N_LAYER > DS4_N_NEXTN_PREDICT) {
        executable_layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    }
    uint32_t start = 0;
    uint32_t end = executable_layers - 1u;
    bool require_token_embd = true;
    if (load_slice) {
        if (load_layer_start >= executable_layers) ds4_die("invalid model load layer slice");
        start = load_layer_start;
        end = load_layer_end == UINT32_MAX ? executable_layers - 1u : load_layer_end;
        if (end >= executable_layers || end < start) ds4_die("invalid model load layer slice");
        require_token_embd = start == 0;
    } else {
        require_output = true;
        optional_output = false;
    }

    if (require_token_embd) {
        w->token_embd = required_tensor(m, "token_embd.weight");
    } else {
        w->token_embd = model_find_tensor(m, "token_embd.weight");
    }
    weights_bind_output(w, m, require_output, optional_output);

    for (uint32_t il = start; il <= end; il++) {
        weights_bind_layer(&w->layer[il], m, il);
    }
    /* GLM nextn/MTP block(s): excluded from the executable pass but bound
     * so the drafter can run them. Only when the full model is loaded. */
    if (!load_slice &&
        DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA &&
        start == 0 && end == executable_layers - 1u) {
        for (uint32_t il = executable_layers; il < DS4_N_LAYER; il++) {
            weights_bind_layer(&w->layer[il], m, il);
        }
    }

    weights_validate_layout(w, start, end, require_token_embd, require_output);
}

typedef struct {
    uint64_t off;
    uint64_t end;
    bool isolate;
} ds4_model_map_span;

typedef struct {
    ds4_model_map_span *v;
    uint32_t len;
    uint32_t cap;
    uint64_t max_tensor_bytes;
} ds4_model_map_span_vec;

static void model_map_span_include_tensor(
        const ds4_tensor *t,
        uint64_t *lo,
        uint64_t *hi,
        uint64_t *max_tensor_bytes) {
    if (!t || t->bytes == 0) return;
    const uint64_t end = t->abs_offset + t->bytes;
    if (*lo == UINT64_MAX || t->abs_offset < *lo) *lo = t->abs_offset;
    if (end > *hi) *hi = end;
    if (t->bytes > *max_tensor_bytes) *max_tensor_bytes = t->bytes;
}

static void model_map_span_vec_append(ds4_model_map_span_vec *spans, uint64_t lo, uint64_t hi, bool isolate) {
    if (!spans || lo == UINT64_MAX || hi <= lo) return;
    if (spans->len == spans->cap) {
        uint32_t new_cap = spans->cap ? spans->cap * 2u : 16u;
        spans->v = xrealloc(spans->v, (size_t)new_cap * sizeof(spans->v[0]));
        spans->cap = new_cap;
    }
    spans->v[spans->len++] = (ds4_model_map_span){lo, hi, isolate};
}

static uint32_t model_map_q4_pro_group_views(void) {
    uint32_t views = 1;
    const char *env = getenv("DS4_METAL_Q4_PRO_MAP_GROUPS");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0' && v > 0 && v <= 384 && (384u % (uint32_t)v) == 0) {
            views = (uint32_t)v;
        }
    }
    return views;
}

static void model_map_span_vec_include_one(ds4_model_map_span_vec *spans, const ds4_tensor *t) {
    if (!t || t->bytes == 0) return;
    const uint64_t q4_isolated_min_bytes = 2ull * 1024ull * 1024ull * 1024ull;
    const uint32_t q4_pro_group_views = model_map_q4_pro_group_views();
    if (t->type == DS4_TENSOR_Q4_K &&
        t->ndim == 3 &&
        t->dim[2] == 384 &&
        t->bytes >= q4_isolated_min_bytes &&
        (t->bytes % q4_pro_group_views) == 0)
    {
        /*
         * PRO Q4 routed expert tensors are too large to hide inside broad
         * layer spans. Isolate them so the default selected-expert path does
         * not stack large aliases on top of layer-sized model views. Optional
         * group splits are enabled by DS4_METAL_Q4_PRO_MAP_GROUPS for Metal
         * experiments that bind stable grouped views.
         */
        const uint64_t group_bytes = t->bytes / q4_pro_group_views;
        if (group_bytes > spans->max_tensor_bytes) spans->max_tensor_bytes = group_bytes;
        for (uint32_t i = 0; i < q4_pro_group_views; i++) {
            const uint64_t lo = t->abs_offset + (uint64_t)i * group_bytes;
            model_map_span_vec_append(spans, lo, lo + group_bytes, true);
        }
        return;
    }

    uint64_t lo = UINT64_MAX, hi = 0;
    model_map_span_include_tensor(t, &lo, &hi, &spans->max_tensor_bytes);
    const bool isolate = (t->type == DS4_TENSOR_Q4_K ||
                          t->type == DS4_TENSOR_MXFP4) &&
                         t->bytes >= q4_isolated_min_bytes;
    model_map_span_vec_append(spans, lo, hi, isolate);
}

static void model_map_span_vec_include_layer(ds4_model_map_span_vec *spans, const ds4_layer_weights *l) {
#define DS4_INCLUDE_TENSOR(t_) model_map_span_vec_include_one(spans, (t_))
    DS4_INCLUDE_TENSOR(l->hc_attn_fn);
    DS4_INCLUDE_TENSOR(l->hc_attn_scale);
    DS4_INCLUDE_TENSOR(l->hc_attn_base);
    DS4_INCLUDE_TENSOR(l->attn_norm);
    DS4_INCLUDE_TENSOR(l->attn_q_a);
    DS4_INCLUDE_TENSOR(l->attn_q_a_norm);
    DS4_INCLUDE_TENSOR(l->attn_q_b);
    DS4_INCLUDE_TENSOR(l->attn_kv);
    DS4_INCLUDE_TENSOR(l->attn_kv_a_mqa);
    DS4_INCLUDE_TENSOR(l->attn_kv_a_norm);
    DS4_INCLUDE_TENSOR(l->attn_k_b);
    DS4_INCLUDE_TENSOR(l->attn_v_b);
    DS4_INCLUDE_TENSOR(l->attn_sinks);
    DS4_INCLUDE_TENSOR(l->attn_output);
    DS4_INCLUDE_TENSOR(l->attn_output_a);
    DS4_INCLUDE_TENSOR(l->attn_output_b);
    DS4_INCLUDE_TENSOR(l->attn_compressor_ape);
    DS4_INCLUDE_TENSOR(l->attn_compressor_kv);
    DS4_INCLUDE_TENSOR(l->attn_compressor_gate);
    DS4_INCLUDE_TENSOR(l->attn_compressor_norm);
    DS4_INCLUDE_TENSOR(l->indexer_attn_q_b);
    DS4_INCLUDE_TENSOR(l->indexer_attn_k);
    DS4_INCLUDE_TENSOR(l->indexer_k_norm);
    DS4_INCLUDE_TENSOR(l->indexer_k_norm_b);
    DS4_INCLUDE_TENSOR(l->indexer_proj);
    DS4_INCLUDE_TENSOR(l->indexer_compressor_ape);
    DS4_INCLUDE_TENSOR(l->indexer_compressor_kv);
    DS4_INCLUDE_TENSOR(l->indexer_compressor_gate);
    DS4_INCLUDE_TENSOR(l->indexer_compressor_norm);
    DS4_INCLUDE_TENSOR(l->hc_ffn_fn);
    DS4_INCLUDE_TENSOR(l->hc_ffn_scale);
    DS4_INCLUDE_TENSOR(l->hc_ffn_base);
    DS4_INCLUDE_TENSOR(l->ffn_norm);
    DS4_INCLUDE_TENSOR(l->ffn_gate_tid2eid);
    DS4_INCLUDE_TENSOR(l->ffn_gate);
    DS4_INCLUDE_TENSOR(l->ffn_up);
    DS4_INCLUDE_TENSOR(l->ffn_down);
    DS4_INCLUDE_TENSOR(l->ffn_gate_inp);
    DS4_INCLUDE_TENSOR(l->ffn_exp_probs_b);
    DS4_INCLUDE_TENSOR(l->ffn_gate_exps);
    DS4_INCLUDE_TENSOR(l->ffn_up_exps);
    DS4_INCLUDE_TENSOR(l->ffn_down_exps);
    DS4_INCLUDE_TENSOR(l->ffn_gate_shexp);
    DS4_INCLUDE_TENSOR(l->ffn_up_shexp);
    DS4_INCLUDE_TENSOR(l->ffn_down_shexp);
    DS4_INCLUDE_TENSOR(l->nextn_eh_proj);
    DS4_INCLUDE_TENSOR(l->nextn_enorm);
    DS4_INCLUDE_TENSOR(l->nextn_hnorm);
    DS4_INCLUDE_TENSOR(l->nextn_shared_head_norm);
#undef DS4_INCLUDE_TENSOR
}

static void model_map_span_vec_include_layer_decode_static(ds4_model_map_span_vec *spans, const ds4_layer_weights *l) {
#define DS4_INCLUDE_TENSOR(t_) model_map_span_vec_include_one(spans, (t_))
    DS4_INCLUDE_TENSOR(l->hc_attn_fn);
    DS4_INCLUDE_TENSOR(l->hc_attn_scale);
    DS4_INCLUDE_TENSOR(l->hc_attn_base);
    DS4_INCLUDE_TENSOR(l->attn_norm);
    DS4_INCLUDE_TENSOR(l->attn_q_a);
    DS4_INCLUDE_TENSOR(l->attn_q_a_norm);
    DS4_INCLUDE_TENSOR(l->attn_q_b);
    DS4_INCLUDE_TENSOR(l->attn_kv);
    DS4_INCLUDE_TENSOR(l->attn_kv_a_mqa);
    DS4_INCLUDE_TENSOR(l->attn_kv_a_norm);
    DS4_INCLUDE_TENSOR(l->attn_k_b);
    DS4_INCLUDE_TENSOR(l->attn_v_b);
    DS4_INCLUDE_TENSOR(l->attn_sinks);
    DS4_INCLUDE_TENSOR(l->attn_output);
    DS4_INCLUDE_TENSOR(l->attn_output_a);
    DS4_INCLUDE_TENSOR(l->attn_output_b);
    DS4_INCLUDE_TENSOR(l->attn_compressor_ape);
    DS4_INCLUDE_TENSOR(l->attn_compressor_kv);
    DS4_INCLUDE_TENSOR(l->attn_compressor_gate);
    DS4_INCLUDE_TENSOR(l->attn_compressor_norm);
    DS4_INCLUDE_TENSOR(l->indexer_attn_q_b);
    DS4_INCLUDE_TENSOR(l->indexer_attn_k);
    DS4_INCLUDE_TENSOR(l->indexer_k_norm);
    DS4_INCLUDE_TENSOR(l->indexer_k_norm_b);
    DS4_INCLUDE_TENSOR(l->indexer_proj);
    DS4_INCLUDE_TENSOR(l->indexer_compressor_ape);
    DS4_INCLUDE_TENSOR(l->indexer_compressor_kv);
    DS4_INCLUDE_TENSOR(l->indexer_compressor_gate);
    DS4_INCLUDE_TENSOR(l->indexer_compressor_norm);
    DS4_INCLUDE_TENSOR(l->hc_ffn_fn);
    DS4_INCLUDE_TENSOR(l->hc_ffn_scale);
    DS4_INCLUDE_TENSOR(l->hc_ffn_base);
    DS4_INCLUDE_TENSOR(l->ffn_norm);
    DS4_INCLUDE_TENSOR(l->ffn_gate_tid2eid);
    DS4_INCLUDE_TENSOR(l->ffn_gate);
    DS4_INCLUDE_TENSOR(l->ffn_up);
    DS4_INCLUDE_TENSOR(l->ffn_down);
    DS4_INCLUDE_TENSOR(l->ffn_gate_inp);
    DS4_INCLUDE_TENSOR(l->ffn_exp_probs_b);
    DS4_INCLUDE_TENSOR(l->ffn_gate_shexp);
    DS4_INCLUDE_TENSOR(l->ffn_up_shexp);
    DS4_INCLUDE_TENSOR(l->ffn_down_shexp);
    DS4_INCLUDE_TENSOR(l->nextn_eh_proj);
    DS4_INCLUDE_TENSOR(l->nextn_enorm);
    DS4_INCLUDE_TENSOR(l->nextn_hnorm);
    DS4_INCLUDE_TENSOR(l->nextn_shared_head_norm);
#undef DS4_INCLUDE_TENSOR
}

static bool glm_stream_resident_decode_layer_supported(
        const ds4_layer_weights *l,
        uint32_t                 il) {
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM_DSA ||
        !l ||
        il < DS4_N_LEADING_DENSE ||
        !l->ffn_gate_exps ||
        !l->ffn_up_exps ||
        !l->ffn_down_exps) {
        return false;
    }
    if (l->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        l->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        (l->ffn_down_exps->type == DS4_TENSOR_IQ2_XXS ||
         l->ffn_down_exps->type == DS4_TENSOR_Q2_K)) {
        return true;
    }
    return l->ffn_gate_exps->type == l->ffn_up_exps->type &&
           l->ffn_gate_exps->type == l->ffn_down_exps->type &&
           (l->ffn_gate_exps->type == DS4_TENSOR_Q2_K ||
            l->ffn_gate_exps->type == DS4_TENSOR_Q4_K);
}

static uint32_t g_glm_streaming_full_resident_start;
static uint32_t g_glm_streaming_full_resident_layers;

static bool glm_stream_resident_decode_layer_enabled(
        const ds4_layer_weights *l,
        uint32_t                 il) {
    if (!glm_stream_resident_decode_layer_supported(l, il)) return false;
    return g_glm_streaming_full_resident_layers != 0 &&
           il >= g_glm_streaming_full_resident_start &&
           il - g_glm_streaming_full_resident_start <
               g_glm_streaming_full_resident_layers;
}

static bool glm_stream_expert_cache_addr_layout_supported(
        const ds4_weights       *w,
        const ds4_layer_weights *l,
        uint32_t                 il) {
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM_DSA ||
        !w ||
        !l ||
        il >= DS4_N_LAYER ||
        il < DS4_N_LEADING_DENSE ||
        !l->ffn_gate_exps ||
        !l->ffn_up_exps ||
        !l->ffn_down_exps ||
        DS4_N_EXPERT_USED == 0 ||
        DS4_N_EXPERT_USED > 8 ||
        DS4_N_EXPERT < 128 ||
        glm_graph_env_present("DS4_ROCM_GLM_DISABLE_STREAMING_EXPERT_CACHE",
                              "DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE")) {
        return false;
    }
    if (!weights_streaming_layer_experts_uniform(w, il)) return false;

    if (l->ffn_gate_exps->type != l->ffn_up_exps->type) return false;
    const bool q2_addr =
        l->ffn_gate_exps->type == DS4_TENSOR_Q2_K &&
        l->ffn_down_exps->type == DS4_TENSOR_Q2_K;
    const bool q4_addr =
        l->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
        l->ffn_down_exps->type == DS4_TENSOR_Q4_K;
    return q2_addr || q4_addr;
}

static DS4_MAYBE_UNUSED bool glm_stream_expert_cache_addr_supported(
        const ds4_weights       *w,
        const ds4_layer_weights *l,
        uint32_t                 il) {
    if (!glm_stream_expert_cache_addr_layout_supported(w, l, il)) {
        return false;
    }

#ifdef DS4_NO_GPU
    return false;
#else
    uint64_t gate_expert_bytes = 0;
    uint64_t down_expert_bytes = 0;
    if (!streaming_layer_gate_down_expert_bytes(l,
                                                &gate_expert_bytes,
                                                &down_expert_bytes)) {
        return false;
    }
    return ds4_gpu_stream_expert_cache_budget_for_expert_size(
                   gate_expert_bytes,
                   down_expert_bytes) >= DS4_N_EXPERT_USED;
#endif
}

static bool glm_stream_selected_expert_cache_supported(
        const ds4_layer_weights *l,
        uint32_t                 il) {
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM_DSA ||
        !l ||
        il < DS4_N_LEADING_DENSE ||
        !l->ffn_gate_exps ||
        !l->ffn_up_exps ||
        !l->ffn_down_exps ||
        DS4_N_EXPERT_USED == 0 ||
        DS4_N_EXPERT_USED > 8 ||
        DS4_N_EXPERT < 128 ||
        glm_graph_env_present("DS4_ROCM_MOE_WRITE_CLAMPED_ACT",
                              "DS4_METAL_MOE_WRITE_CLAMPED_ACT") ||
        glm_graph_env_present("DS4_ROCM_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                              "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") ||
        glm_graph_env_present("DS4_ROCM_GLM_DISABLE_STREAMING_EXPERT_CACHE",
                              "DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE")) {
        return false;
    }

    if (l->ffn_gate_exps->type != DS4_TENSOR_IQ2_XXS ||
        l->ffn_up_exps->type != DS4_TENSOR_IQ2_XXS) {
        return false;
    }

    if (l->ffn_down_exps->type == DS4_TENSOR_Q2_K) {
        return !glm_graph_env_present("DS4_ROCM_DISABLE_IQ2_SELECTED_EXPERT_VIEWS",
                                      "DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS");
    }
    if (l->ffn_down_exps->type == DS4_TENSOR_IQ2_XXS) {
        return !glm_graph_env_present("DS4_ROCM_DISABLE_IQ2_STREAM_ADDR_TABLE",
                                      "DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE");
    }
    return false;
}

static bool glm_stream_decode_experts_are_streamed(
        const ds4_weights       *w,
        const ds4_layer_weights *l,
        uint32_t                 il) {
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM_DSA) return false;
    return glm_stream_expert_cache_addr_layout_supported(w, l, il) ||
           glm_stream_selected_expert_cache_supported(l, il);
}

/*
 * Decode-time spans for one layer. The static set excludes routed expert
 * tensors only when the streaming expert-cache path can really serve them.
 * Boosted layers, mixed GLM quant layouts such as Q4 gate/up plus Q5 down, or
 * undersized expert caches fall back to direct model-range reads. Include
 * those expert tensors so cache-hit prefill extension and decode are covered.
 */
static void model_map_span_vec_include_layer_decode(
        ds4_model_map_span_vec *spans,
        const ds4_weights      *w,
        uint32_t                il) {
    const ds4_layer_weights *l = &w->layer[il];
    model_map_span_vec_include_layer_decode_static(spans, l);
    if (!weights_streaming_layer_experts_uniform(w, il) ||
        (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA &&
         !glm_stream_decode_experts_are_streamed(w, l, il)) ||
        glm_stream_resident_decode_layer_enabled(l, il)) {
        model_map_span_vec_include_one(spans, l->ffn_gate_exps);
        model_map_span_vec_include_one(spans, l->ffn_up_exps);
        model_map_span_vec_include_one(spans, l->ffn_down_exps);
    }
}

static void model_map_span_vec_include_output(ds4_model_map_span_vec *spans, const ds4_weights *w) {
    model_map_span_vec_include_one(spans, w->output_hc_base);
    model_map_span_vec_include_one(spans, w->output_hc_fn);
    model_map_span_vec_include_one(spans, w->output_hc_scale);
    model_map_span_vec_include_one(spans, w->output_norm);
    model_map_span_vec_include_one(spans, w->output);
}

static int model_map_span_cmp(const void *a, const void *b) {
    const ds4_model_map_span *sa = a;
    const ds4_model_map_span *sb = b;
    if (sa->off < sb->off) return -1;
    if (sa->off > sb->off) return 1;
    if (sa->end < sb->end) return -1;
    if (sa->end > sb->end) return 1;
    return 0;
}

static bool model_map_span_vec_finish(ds4_model_map_span_vec *spans) {
    if (!spans || spans->len == 0 || spans->max_tensor_bytes == 0) return false;

    qsort(spans->v, spans->len, sizeof(spans->v[0]), model_map_span_cmp);
    uint32_t out = 0;
    for (uint32_t i = 0; i < spans->len; i++) {
        if (out == 0 ||
            spans->v[i].off > spans->v[out - 1u].end ||
            spans->v[i].isolate ||
            spans->v[out - 1u].isolate) {
            spans->v[out++] = spans->v[i];
        } else if (spans->v[i].end > spans->v[out - 1u].end) {
            spans->v[out - 1u].end = spans->v[i].end;
        }
    }
    spans->len = out;
    return spans->len != 0;
}

static DS4_MAYBE_UNUSED bool weights_model_map_spans(
        const ds4_weights *w,
        uint32_t layer_start,
        uint32_t layer_end,
        bool include_output,
        ds4_model_map_span_vec *spans) {
    if (!w || !spans) return false;
    if (layer_start >= DS4_N_LAYER) return false;
    if (layer_end == UINT32_MAX) layer_end = DS4_N_LAYER - 1u;
    if (layer_end >= DS4_N_LAYER || layer_end < layer_start) return false;

    memset(spans, 0, sizeof(*spans));
    if (layer_start == 0) model_map_span_vec_include_one(spans, w->token_embd);
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        model_map_span_vec_include_layer(spans, &w->layer[il]);
    }
    if (include_output) model_map_span_vec_include_output(spans, w);
    return model_map_span_vec_finish(spans);
}

static const uint8_t *tensor_expert_bytes(
        const ds4_model  *m,
        const ds4_tensor *w,
        uint32_t          expert,
        uint64_t         *in_dim,
        uint64_t         *out_dim,
        uint64_t         *row_bytes);

