#include "engine_internal.h"

/* Decode Primitives module. */
/* Decode helpers and diagnostic fallbacks for the CUDA graph runtime. */

static bool metal_graph_env_flag(const char *name, int *cache) {
    if (*cache == -1) {
        const char *env = getenv(name);
        *cache = env && env[0] && strcmp(env, "0") != 0;
    }
    return *cache != 0;
}

bool ds4x_graph_cache_zero(ds4_gpu_graph *g,
                                  ds4x_cache_kind kind,
                                  ds4_gpu_tensor *dst,
                                  uint32_t rows) {
    const ds4x_cache_args args = {
        .kind = kind,
        .operation = DS4X_CACHE_ZERO,
        .dst = dst,
        .src = NULL,
        .dst_row = 0,
        .src_row = 0,
        .rows = rows,
    };
    return ds4x_cache_launch(metal_graph_runtime(g), &args) != 0;
}

bool ds4x_graph_cache_pack(ds4_gpu_graph *g,
                                  ds4x_cache_kind kind,
                                  ds4_gpu_tensor *dst,
                                  uint64_t dst_row,
                                  const ds4_gpu_tensor *src,
                                  uint32_t src_row,
                                  uint32_t rows) {
    const ds4x_cache_args args = {
        .kind = kind,
        .operation = DS4X_CACHE_PACK,
        .dst = dst,
        .src = src,
        .dst_row = dst_row,
        .src_row = src_row,
        .rows = rows,
    };
    return ds4x_cache_launch(metal_graph_runtime(g), &args) != 0;
}

bool ds4x_graph_indexer(ds4_gpu_graph *g,
                               ds4x_indexer_mode mode,
                               ds4_gpu_tensor *scores,
                               ds4_gpu_tensor *selected,
                               const ds4_gpu_tensor *query,
                               const ds4_gpu_tensor *weights,
                               const ds4_gpu_tensor *cache,
                               uint32_t n_comp,
                               uint32_t n_tokens,
                               uint32_t pos0,
                               uint32_t ratio,
                               float scale) {
    const ds4x_indexer_args args = {
        .mode = mode,
        .scores = scores,
        .selected = selected,
        .query = query,
        .weights = weights,
        .cache = cache,
        .n_comp = n_comp,
        .n_tokens = n_tokens,
        .pos0 = pos0,
        .n_head = DS4_N_INDEXER_HEAD,
        .head_dim = DS4_N_INDEXER_HEAD_DIM,
        .ratio = ratio,
        .top_k = DS4_N_INDEXER_TOP_K < n_comp
            ? DS4_N_INDEXER_TOP_K : n_comp,
        .scale = scale,
    };
    return ds4x_indexer_launch(metal_graph_runtime(g), &args) != 0;
}

bool ds4x_graph_attention_decode(
        ds4_gpu_graph *g,
        ds4_gpu_tensor *heads,
        const ds4_model *model,
        uint64_t sinks_offset,
        const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *raw_kv,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        const ds4_gpu_tensor *compressed_kv,
        uint32_t n_comp) {
    const ds4x_attention_decode_args args = {
        .heads = heads,
        .model_map = model->map,
        .model_size = model->size,
        .sinks_offset = sinks_offset,
        .query = query,
        .raw_kv = raw_kv,
        .compressed_kv = compressed_kv,
        .compressed_mask = NULL,
        .n_raw = n_raw,
        .raw_cap = raw_cap,
        .raw_start = raw_start,
        .n_comp = n_comp,
        .use_mask = 0,
        .n_head = DS4_N_HEAD,
        .head_dim = DS4_N_HEAD_DIM,
    };
    return ds4x_attention_decode_launch(metal_graph_runtime(g), &args) != 0;
}

bool ds4x_graph_routed_moe(
        ds4_gpu_graph *g,
        const ds4_model *model,
        const ds4_layer_weights *layer,
        uint32_t layer_index,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t output_dim) {
    const ds4x_routed_moe_args args = {
        .output = metal_graph_routed_out(g),
        .gate = metal_graph_routed_gate(g),
        .up = metal_graph_routed_up(g),
        .mid = metal_graph_routed_mid(g),
        .down = metal_graph_routed_down(g),
        .model_map = model->map,
        .model_size = model->size,
        .gate_offset = layer->ffn_gate_exps->abs_offset,
        .up_offset = layer->ffn_up_exps->abs_offset,
        .down_offset = layer->ffn_down_exps->abs_offset,
        .gate_type = layer->ffn_gate_exps->type,
        .down_type = layer->ffn_down_exps->type,
        .gate_expert_bytes = gate_expert_bytes,
        .gate_row_bytes = gate_row_bytes,
        .down_expert_bytes = down_expert_bytes,
        .down_row_bytes = down_row_bytes,
        .expert_in_dim = expert_in_dim,
        .expert_mid_dim = expert_mid_dim,
        .output_dim = output_dim,
        .selected = metal_graph_router_selected(g),
        .weights = metal_graph_router_weights(g),
        .total_experts = DS4_N_EXPERT,
        .selected_experts = DS4_N_EXPERT_USED,
        .clamp = DS4_SWIGLU_CLAMP_EXP,
        .input = metal_graph_ffn_norm(g),
        .add_input = NULL,
        .layer = layer_index,
    };
    return ds4x_routed_moe_launch(metal_graph_runtime(g), &args) != 0;
}

bool ds4x_graph_output_hc(ds4_gpu_graph *g,
                                 ds4_gpu_tensor *output,
                                 const ds4_gpu_tensor *pre,
                                 const ds4_model *model,
                                 uint64_t scale_offset,
                                 uint64_t base_offset) {
    const ds4x_output_hc_args args = {
        .output = output,
        .pre = pre,
        .model_map = model->map,
        .model_size = model->size,
        .scale_offset = scale_offset,
        .base_offset = base_offset,
        .n_hc = DS4_N_HC,
        .epsilon = DS4_HC_EPS,
    };
    return ds4x_output_hc_launch(metal_graph_runtime(g), &args) != 0;
}

ds4x_decode_graph_args ds4x_graph_decode_args(
        ds4_gpu_graph *g,
        uint32_t layer,
        uint32_t island,
        uint32_t variant) {
    const ds4x_decode_graph_args args = {
        .layer = layer,
        .island = island,
        .variant = variant,
        .current_hc = metal_graph_cur_hc(g),
        .after_attention_hc = metal_graph_after_attn_hc(g),
        .after_ffn_hc = metal_graph_after_ffn_hc(g),
        .attention_norm = metal_graph_attn_norm(g),
    };
    return args;
}

bool metal_graph_use_reference_hc_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_HC_FUSION", &cache);
}

bool metal_graph_use_reference_kv_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_KV_FUSION", &cache);
}

bool metal_graph_use_reference_qkv_norm(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_QKV_NORM_FUSION", &cache);
}

bool metal_graph_use_reference_qkv_pair_proj(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_QKV_PAIR_PROJ", &cache);
}

bool metal_graph_use_reference_compressor_pair_proj(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ", &cache);
}

bool metal_graph_use_reference_hc_norm_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_HC_NORM_FUSION", &cache);
}

bool metal_graph_enable_batch_hc_norm_fusion(void) {
    static int cache = -1;
    if (metal_graph_use_reference_hc_norm_decode()) return false;
    if (cache == -1) {
        const char *disable = getenv("DS4_METAL_DISABLE_BATCH_HC_NORM_FUSION");
        if (disable && disable[0] && strcmp(disable, "0") != 0) {
            cache = 0;
        } else {
            const char *legacy_enable =
                getenv("DS4_METAL_ENABLE_BATCH_HC_NORM_FUSION");
            cache = (!legacy_enable || !legacy_enable[0] ||
                     strcmp(legacy_enable, "0") != 0) ? 1 : 0;
        }
    }
    return cache != 0;
}

bool metal_graph_use_reference_shared_down_hc(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION", &cache);
}

bool metal_graph_use_reference_attn_out_hc(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_ATTN_OUT_HC_FUSION", &cache);
}

bool metal_graph_decode_hc_pre(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const ds4_model        *model,
        uint64_t                scale_offset,
        uint64_t                base_offset) {
    if (metal_graph_use_reference_hc_decode()) {
        return ds4_gpu_hc_split_sinkhorn_tensor(split,
                                                  mix,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0 &&
               ds4_gpu_hc_weighted_sum_tensor(out,
                                                 residual_hc,
                                                 split,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    }

    return ds4_gpu_hc_split_weighted_sum_tensor(out,
                                                  split,
                                                  mix,
                                                  residual_hc,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0;
}

bool metal_graph_hc_norm_fusion_check_enabled(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_HC_NORM_FUSION_CHECK", &cache);
}

static float metal_graph_hc_norm_fusion_check_tolerance(void) {
    static int initialized;
    static float tolerance;
    if (initialized) return tolerance;
    tolerance = 2.0e-4f;
    const char *env = getenv("DS4_METAL_HC_NORM_FUSION_CHECK_TOL");
    if (env && env[0]) {
        char *end = NULL;
        const float v = strtof(env, &end);
        if (end != env && isfinite(v) && v > 0.0f) tolerance = v;
    }
    initialized = 1;
    return tolerance;
}

bool metal_graph_check_hc_norm_fusion(
        const char            *label,
        ds4_gpu_tensor        *fused_out,
        ds4_gpu_tensor        *fused_norm,
        const ds4_gpu_tensor  *mix,
        const ds4_gpu_tensor  *residual_hc,
        const ds4_model       *model,
        uint64_t               scale_offset,
        uint64_t               base_offset,
        uint64_t               norm_weight_offset,
        uint32_t               il,
        uint32_t               pos) {
    if (!metal_graph_hc_norm_fusion_check_enabled()) return true;
    if (!fused_out || !fused_norm || !mix || !residual_hc || !model) return false;

    const uint64_t n_embd = DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    ds4_gpu_tensor *ref_split = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    bool ok = ref_split && ref_out && ref_norm;

    if (ok) {
        ok = ds4_gpu_hc_split_sinkhorn_tensor(ref_split,
                                              mix,
                                              model->map,
                                              model->size,
                                              scale_offset,
                                              base_offset,
                                              DS4_N_HC,
                                              DS4_N_HC_SINKHORN_ITER,
                                              DS4_HC_EPS) != 0 &&
             ds4_gpu_hc_weighted_sum_tensor(ref_out,
                                            residual_hc,
                                            ref_split,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0 &&
             ds4_gpu_rms_norm_weight_tensor(ref_norm,
                                            ref_out,
                                            model->map,
                                            model->size,
                                            norm_weight_offset,
                                            DS4_N_EMBD,
                                            DS4_RMS_EPS) != 0;
    }

    if (ok) ok = ds4_gpu_end_commands() != 0;

    float *fused_out_cpu = NULL;
    float *ref_out_cpu = NULL;
    float *fused_norm_cpu = NULL;
    float *ref_norm_cpu = NULL;
    if (ok) {
        fused_out_cpu = xmalloc((size_t)n_embd * sizeof(float));
        ref_out_cpu = xmalloc((size_t)n_embd * sizeof(float));
        fused_norm_cpu = xmalloc((size_t)n_embd * sizeof(float));
        ref_norm_cpu = xmalloc((size_t)n_embd * sizeof(float));
        ok = ds4_gpu_tensor_read(fused_out, 0, fused_out_cpu, n_embd * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(ref_out, 0, ref_out_cpu, n_embd * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(fused_norm, 0, fused_norm_cpu, n_embd * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(ref_norm, 0, ref_norm_cpu, n_embd * sizeof(float)) != 0;
    }

    if (ok) {
        const float out_max = max_abs_diff(fused_out_cpu, ref_out_cpu, n_embd);
        const float out_rms = rms_abs_diff(fused_out_cpu, ref_out_cpu, n_embd);
        const float norm_max = max_abs_diff(fused_norm_cpu, ref_norm_cpu, n_embd);
        const float norm_rms = rms_abs_diff(fused_norm_cpu, ref_norm_cpu, n_embd);
        const float tol = metal_graph_hc_norm_fusion_check_tolerance();
        fprintf(stderr,
                "ds4: CUDA HC norm fusion check %s layer=%u pos=%u "
                "out_max=%g out_rms=%g norm_max=%g norm_rms=%g tol=%g\n",
                label ? label : "hc",
                il,
                pos,
                out_max,
                out_rms,
                norm_max,
                norm_rms,
                tol);
        if (out_max > tol || norm_max > tol) {
            fprintf(stderr,
                    "ds4: CUDA HC norm fusion check failed for %s layer=%u pos=%u\n",
                    label ? label : "hc",
                    il,
                    pos);
            ok = false;
        }
    }

    free(fused_out_cpu);
    free(ref_out_cpu);
    free(fused_norm_cpu);
    free(ref_norm_cpu);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(ref_split);

    const bool restart_ok = ds4_gpu_begin_commands() != 0;
    return ok && restart_ok;
}

bool metal_graph_decode_kv_store(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row) {
    if (metal_graph_use_reference_kv_decode()) {
        return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0 &&
               ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, DS4_N_HEAD_DIM) != 0;
    }

    return ds4_gpu_kv_fp8_store_raw_tensor(kv,
                                             raw_cache,
                                             raw_cap,
                                             raw_row,
                                             DS4_N_HEAD_DIM,
                                             DS4_N_ROT) != 0;
}

uint32_t metal_graph_attn_comp_cache_format(void) {
    return DS4_GPU_CACHE_SPARK_KV;
}

static bool metal_graph_store_attn_comp_stage(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (!g || il >= DS4_N_LAYER) return false;
    if (rows == 0) return true;
    if (!g->layer_attn_comp_cache[il] || !metal_graph_attn_comp_stage(g)) return false;
    if (rows > g->attn_comp_stage_cap || first_row > g->layer_comp_cap[il] ||
        rows > g->layer_comp_cap[il] - first_row) {
        return false;
    }

    return ds4x_graph_cache_pack(g, DS4X_CACHE_KV,
                                 g->layer_attn_comp_cache[il], first_row,
                                 metal_graph_attn_comp_stage(g), 0, rows);
}

ds4_gpu_tensor *metal_graph_attn_comp_update_target(
        ds4_gpu_graph *g,
        uint32_t       il) {
    (void)il;
    return metal_graph_attn_comp_stage(g);
}

uint32_t metal_graph_attn_comp_update_row(uint32_t row) {
    (void)row;
    return 0u;
}

bool metal_graph_commit_attn_comp_stage(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    return metal_graph_store_attn_comp_stage(g, il, first_row, rows);
}

ds4_gpu_tensor *metal_graph_attn_comp_row_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row) {
    (void)il;
    (void)row;
    return ds4_gpu_tensor_view(metal_graph_attn_comp_stage(g),
                               0,
                               (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
}

ds4_gpu_tensor *metal_graph_attn_comp_prefill_target(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    (void)il;
    (void)first_row;
    (void)rows;
    return metal_graph_attn_comp_stage(g);
}

void metal_graph_attn_comp_prefill_target_free(ds4_gpu_tensor *t) {
    (void)t;
}

uint32_t metal_graph_raw_start_for_span(
        const ds4_gpu_graph *g,
        uint32_t last_pos,
        uint32_t n_raw) {
    if (!g || g->raw_cap == 0u || n_raw == 0u) return 0u;
    return (last_pos + 1u - n_raw) % g->raw_cap;
}

uint32_t metal_graph_raw_span_for_batch(
        const ds4_gpu_graph *g,
        uint32_t pos0,
        uint32_t n_tokens) {
    if (!g || g->raw_cap == 0u || n_tokens == 0u) return 0u;
    const uint32_t window = g->raw_window ? g->raw_window : DS4_N_SWA;
    const uint32_t last_pos = pos0 + n_tokens - 1u;
    uint64_t needed = n_tokens;
    if (window != 0u) {
        needed += n_tokens == 1u ? (uint64_t)window - 1u : window;
    }
    const uint64_t available = (uint64_t)last_pos + 1u;
    if (needed > available) needed = available;
    if (needed > g->raw_cap) needed = g->raw_cap;
    return (uint32_t)needed;
}

bool metal_graph_capture_prefix_attn_state(
        ds4_gpu_graph *g,
        uint32_t il,
        uint32_t slot) {
    if (!g->spec_capture_prefixes || ds4_layer_compress_ratio(il) == 0u) {
        return true;
    }
    if (slot >= DS4_SPEC_PREFIX_SLOTS ||
        !g->spec_prefix1_attn_state_kv[il] ||
        !g->spec_prefix1_attn_state_score[il]) {
        return false;
    }
    const uint64_t bytes =
        ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
    const uint64_t offset = (uint64_t)slot * bytes;
    g->spec_prefix_n_comp[slot][il] = g->layer_n_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_kv[il], offset,
                               g->layer_attn_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_score[il], offset,
                               g->layer_attn_state_score[il], 0, bytes) != 0;
}

bool metal_graph_capture_prefix_index_state(
        ds4_gpu_graph *g,
        uint32_t il,
        uint32_t slot) {
    if (!g->spec_capture_prefixes || ds4_layer_compress_ratio(il) != 4u) {
        return true;
    }
    if (slot >= DS4_SPEC_PREFIX_SLOTS ||
        !g->spec_prefix1_index_state_kv[il] ||
        !g->spec_prefix1_index_state_score[il]) {
        return false;
    }
    const uint64_t bytes =
        ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
    const uint64_t offset = (uint64_t)slot * bytes;
    g->spec_prefix_n_index_comp[slot][il] = g->layer_n_index_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_index_state_kv[il], offset,
                               g->layer_index_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_index_state_score[il], offset,
                               g->layer_index_state_score[il], 0, bytes) != 0;
}

uint32_t metal_graph_decode_indexer_sparse_threshold(
        const ds4_gpu_graph *g) {
    (void)g;
    const char *env = getenv("DS4_CUDA_DECODE_INDEXER_SPARSE_THRESHOLD");
    if (!env || !env[0]) {
        env = getenv("DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD");
    }
    if (env && env[0]) {
        char *end = NULL;
        const unsigned long value = strtoul(env, &end, 10);
        if (end != env && *end == '\0' &&
            (value == 64ul || value == 128ul || value == 256ul ||
             value == 512ul || value == 1024ul || value == 2048ul ||
             value == 4096ul)) {
            return (uint32_t)value;
        }
        fprintf(stderr,
                "ds4: invalid indexer sparse threshold: %s\n", env);
    }
    return 1024u;
}
