#ifndef DS4_NO_GPU
/* =========================================================================
 * Metal Decode Release Helpers and Reference Fallbacks.
 * =========================================================================
 *
 * The normal generation path uses the fused helpers below.  The older unfused
 * kernels remain available as diagnostic reference paths selected only by the
 * DS4_METAL_DISABLE_*_FUSION environment switches.
 */

static bool metal_graph_env_flag(const char *name, int *cache) {
    if (*cache == -1) {
#ifdef DS4_ROCM_BUILD
        (void)name;
        *cache = 0;
#else
        const char *env = getenv(name);
        *cache = env && env[0] && strcmp(env, "0") != 0;
#endif
    }
    return *cache != 0;
}

static bool metal_graph_use_reference_hc_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_HC_FUSION", &cache);
}

static bool metal_graph_use_reference_kv_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_KV_FUSION", &cache);
}

static bool metal_graph_use_reference_qkv_norm(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_QKV_NORM_FUSION", &cache);
}

static bool metal_graph_use_reference_qkv_pair_proj(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_QKV_PAIR_PROJ", &cache);
}

static bool metal_graph_use_reference_compressor_pair_proj(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ", &cache);
}

static bool metal_graph_use_reference_hc_norm_decode(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_HC_NORM_FUSION", &cache);
}

static bool metal_graph_enable_batch_hc_norm_fusion(void) {
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

static bool metal_graph_use_reference_shared_down_hc(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_SHARED_DOWN_HC_FUSION", &cache);
}

static bool metal_graph_use_reference_attn_out_hc(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_DISABLE_ATTN_OUT_HC_FUSION", &cache);
}

static bool metal_graph_decode_hc_pre(
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

static bool metal_graph_hc_norm_fusion_check_enabled(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_HC_NORM_FUSION_CHECK", &cache);
}

static float metal_graph_hc_norm_fusion_check_tolerance(void) {
    static int initialized;
    static float tolerance;
    if (initialized) return tolerance;
    tolerance = 2.0e-4f;
#ifndef DS4_ROCM_BUILD
    const char *env = getenv("DS4_METAL_HC_NORM_FUSION_CHECK_TOL");
    if (env && env[0]) {
        char *end = NULL;
        const float v = strtof(env, &end);
        if (end != env && isfinite(v) && v > 0.0f) tolerance = v;
    }
#endif
    initialized = 1;
    return tolerance;
}

static bool metal_graph_check_hc_norm_fusion(
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
                "ds4: Metal HC norm fusion check %s layer=%u pos=%u "
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
                    "ds4: Metal HC norm fusion check failed for %s layer=%u pos=%u\n",
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

extern int ds4_gpu_kv_rope_fp8_fuse_available(void);
extern int ds4_gpu_decode_attn_rope_fuse_available(void);
extern int ds4_gpu_decode_attn_rope_fuse_used(void);
extern void ds4_gpu_set_decode_attn_rope_fuse(
        uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig,
        bool inverse, float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow);
extern int ds4_gpu_kv_rope_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv, ds4_gpu_tensor *raw_cache, uint32_t raw_cap,
        uint32_t row, uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
        uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow);

static bool metal_graph_decode_kv_store(
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

static uint64_t metal_graph_attn_comp_cache_row_bytes(void) {
    return DS4_SPARK_KV_ROW_BYTES;
}

static uint32_t metal_graph_attn_comp_cache_format(void) {
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

    return ds4_gpu_spark_pack_kv_rows_tensor(g->layer_attn_comp_cache[il],
                                             first_row,
                                             metal_graph_attn_comp_stage(g),
                                             0,
                                             rows) != 0;
}

static ds4_gpu_tensor *metal_graph_attn_comp_update_target(
        ds4_gpu_graph *g,
        uint32_t       il) {
    (void)il;
    return metal_graph_attn_comp_stage(g);
}

static uint32_t metal_graph_attn_comp_update_row(uint32_t row) {
    (void)row;
    return 0u;
}

static bool metal_graph_commit_attn_comp_stage(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    return metal_graph_store_attn_comp_stage(g, il, first_row, rows);
}

static ds4_gpu_tensor *metal_graph_attn_comp_row_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row) {
    (void)il;
    (void)row;
    return ds4_gpu_tensor_view(metal_graph_attn_comp_stage(g),
                               0,
                               (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
}

static ds4_gpu_tensor *metal_graph_attn_comp_prefill_target(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    (void)il;
    (void)first_row;
    (void)rows;
    return metal_graph_attn_comp_stage(g);
}

static void metal_graph_attn_comp_prefill_target_free(ds4_gpu_tensor *t) {
    (void)t;
}

static bool metal_graph_cuda_tp_attn_cache_dup_layer_ready(
        const ds4_gpu_graph *g,
        uint32_t             il) {
    if (!g || il >= DS4_N_LAYER || !g->cuda_tp_attn_cache_dup) return false;
    if (!g->placement || !g->layer_raw_cache[il] || !g->layer_raw_cache_tp[il]) {
        return false;
    }
    const int layer_tier = g->placement[il + 1];
    if (metal_graph_cuda_tp_partner_tier(layer_tier) < 0) return false;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ratio != 0 &&
        (!g->layer_attn_comp_cache[il] ||
         !g->layer_attn_comp_cache_tp[il])) {
        return false;
    }
    return true;
}

static bool metal_graph_cuda_tp_attn_cache_copy_row(
        ds4_gpu_tensor       *dst_base,
        const ds4_gpu_tensor *src_base,
        uint64_t              offset,
        uint64_t              bytes) {
    if (bytes == 0) return true;
    ds4_gpu_tensor *dst = ds4_gpu_tensor_view(dst_base, offset, bytes);
    ds4_gpu_tensor *src = ds4_gpu_tensor_view(src_base, offset, bytes);
    bool ok = dst && src && ds4_gpu_tensor_copy_xdev(dst, src, bytes) != 0;
    ds4_gpu_tensor_free(src);
    ds4_gpu_tensor_free(dst);
    return ok;
}

static bool metal_graph_cuda_tp_attn_cache_sync_raw_row(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       raw_row) {
    if (!metal_graph_cuda_tp_attn_cache_dup_layer_ready(g, il)) return true;
    if (raw_row >= g->raw_cap) return false;
    const uint64_t row_bytes = DS4_SPARK_KV_ROW_BYTES;
    return metal_graph_cuda_tp_attn_cache_copy_row(
            g->layer_raw_cache_tp[il],
            g->layer_raw_cache[il],
            (uint64_t)raw_row * row_bytes,
            row_bytes);
}

static bool metal_graph_cuda_tp_attn_cache_sync_all(ds4_gpu_graph *g) {
    if (!g || !g->cuda_tp_attn_cache_dup) return true;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!metal_graph_cuda_tp_attn_cache_dup_layer_ready(g, il)) return false;
        const uint64_t raw_bytes =
            (uint64_t)g->raw_cap * DS4_SPARK_KV_ROW_BYTES;
        if (!ds4_gpu_tensor_copy_xdev(g->layer_raw_cache_tp[il],
                                      g->layer_raw_cache[il],
                                      raw_bytes)) {
            return false;
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t n_comp = g->layer_n_comp[il];
        if (ratio != 0 && n_comp != 0) {
            const uint64_t comp_bytes =
                (uint64_t)n_comp * metal_graph_attn_comp_cache_row_bytes();
            if (!ds4_gpu_tensor_copy_xdev(g->layer_attn_comp_cache_tp[il],
                                          g->layer_attn_comp_cache[il],
                                          comp_bytes)) {
                return false;
            }
        }
    }
    return true;
}

/* Encode one DS4 decode layer on Metal.  This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */
static bool metal_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0);
static bool metal_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);
static bool metal_graph_decode_stage_profile_enabled(uint32_t il);
static bool metal_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);
static bool metal_graph_matmul_dense_quant_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);
static bool metal_graph_dense_quant_row_bytes(
        const ds4_tensor *w,
        uint64_t          in_dim,
        uint64_t         *row_bytes);
static bool metal_graph_matmul_dense_quant_abs(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);
static bool metal_graph_matmul_dense_quant_kslice(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off);
static bool metal_graph_attention_output_dense_quant_low(
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads);
static bool metal_graph_attention_output_dense_quant_tp(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        const ds4_tensor       *out_b,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads);
static bool metal_graph_attention_output_dense_quant_batch(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        const ds4_tensor       *out_b,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

static bool metal_graph_use_pro_q4_cpu_router(void) {
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_PRO_Q4_CPU_ROUTER", &cache);
}

static bool metal_graph_use_streaming_iq2_cpu_router(void) {
    return getenv("DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER") != NULL &&
           getenv("DS4_METAL_DISABLE_STREAMING_IQ2_CPU_ROUTER") == NULL;
}

static bool metal_graph_use_q4_selected_shared_overlap(
        const ds4_gpu_graph *g) {
#ifdef DS4_ROCM_BUILD
    /*
     * ROCm SSD streaming maps only the selected routed experts, not the full
     * Q4 expert table.  Stage that compact selection while the shared expert
     * runs so decode reaches the existing pointer-backed MoE path.
     *
     * GLM owns a separate routed-MoE graph and must retain its validated
     * selected-expert policy.
     */
    if (g &&
        g->ssd_streaming &&
        DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM_DSA) {
        return true;
    }
#else
    (void)g;
#endif
    static int cache = -1;
    return metal_graph_env_flag("DS4_METAL_Q4_SELECTED_OVERLAP_SHARED", &cache);
}

static bool metal_graph_use_cuda_selected_shared_overlap(const ds4_gpu_graph *g) {
#if !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU) && !defined(__APPLE__)
    return g &&
           g->ssd_streaming &&
           getenv("DS4_CUDA_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP") == NULL;
#else
    (void)g;
    return false;
#endif
}

static bool metal_graph_q4_non_streaming_opt_in_enabled(void) {
    return getenv("DS4_METAL_ENABLE_Q4_SELECTED_EXPERT_VIEWS") != NULL ||
           getenv("DS4_METAL_ENABLE_PRO_Q4_SELECTED_EXPERT_VIEWS") != NULL ||
           getenv("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") != NULL ||
           getenv("DS4_METAL_ENABLE_Q4_EXPERT_ADDRESS_TABLE") != NULL ||
           getenv("DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO") != NULL ||
           getenv("DS4_METAL_ENABLE_PRO_Q4_EXPERT_ADDRESS_AUTO") != NULL;
}

static bool metal_graph_q4_selected_paths_allowed(const ds4_gpu_graph *g) {
    if (!g) return false;
    if (g->ssd_streaming) return true;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) return false;
    return metal_graph_q4_non_streaming_opt_in_enabled();
}

static bool metal_graph_use_iq2_selected_shared_overlap(const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           getenv("DS4_METAL_DISABLE_STREAMING_SELECTED_SHARED_OVERLAP") == NULL &&
           getenv("DS4_METAL_DISABLE_IQ2_SELECTED_SHARED_OVERLAP") == NULL;
}

static bool metal_graph_use_iq2_selected_async_load(const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
#ifndef DS4_ROCM_BUILD
           getenv("DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_LOAD") == NULL;
#else
           true;
#endif
}

static bool metal_graph_use_iq2_selected_async_early_commit(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
#ifndef DS4_ROCM_BUILD
           getenv("DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_EARLY_COMMIT") == NULL;
#else
           false;
#endif
}

static bool metal_graph_use_pro_q4_expert_table_auto(const ds4_gpu_graph *g) {
    if (getenv("DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_AUTO") != NULL ||
        getenv("DS4_METAL_DISABLE_Q4_EXPERT_TABLE") != NULL) {
        return false;
    }
    if (!g || (!g->ssd_streaming &&
               getenv("DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO") == NULL)) {
        return false;
    }
#ifndef DS4_NO_GPU
    return ds4_gpu_pro_q4_expert_table_auto_available() != 0;
#else
    return false;
#endif
}

static bool metal_graph_decode_cpu_router_applicable(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer) {
    const bool pro_q4 =
        DS4_MODEL_VARIANT == DS4_VARIANT_PRO &&
        metal_graph_use_pro_q4_cpu_router() &&
        layer->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q4_K;
    const bool streaming_iq2 =
        g &&
        g->ssd_streaming &&
        !g->quality &&
        metal_graph_use_streaming_iq2_cpu_router() &&
        layer->ffn_gate_tid2eid == NULL &&
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
        DS4_N_EXPERT_USED == 6 &&
        DS4_N_EXPERT >= 128 &&
        !glm_graph_env_present("DS4_ROCM_MOE_WRITE_CLAMPED_ACT",
                               "DS4_METAL_MOE_WRITE_CLAMPED_ACT") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                               "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") &&
        !glm_graph_env_present("DS4_ROCM_DISABLE_IQ2_SELECTED_EXPERT_VIEWS",
                               "DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS");
    return pro_q4 || streaming_iq2;
}

static bool metal_graph_decode_pro_q4_expert_table_expected(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer,
        uint64_t                 gate_tensor_bytes,
        uint64_t                 down_tensor_bytes) {
    const uint64_t q4_selected_min_tensor_bytes = 2ull * 1024ull * 1024ull * 1024ull;
    return !g->quality &&
           DS4_MODEL_VARIANT == DS4_VARIANT_PRO &&
           metal_graph_q4_selected_paths_allowed(g) &&
           layer->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
           layer->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
           layer->ffn_down_exps->type == DS4_TENSOR_Q4_K &&
           DS4_N_EXPERT == 384 &&
           DS4_N_EXPERT_USED == 6 &&
           gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
           down_tensor_bytes >= q4_selected_min_tensor_bytes &&
           !glm_graph_env_present("DS4_ROCM_MOE_WRITE_CLAMPED_ACT",
                                  "DS4_METAL_MOE_WRITE_CLAMPED_ACT") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                                  "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") &&
           (metal_graph_use_pro_q4_expert_table_auto(g) ||
            getenv("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") != NULL) &&
           getenv("DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_AUTO") == NULL &&
           getenv("DS4_METAL_DISABLE_Q4_EXPERT_TABLE") == NULL;
}

static bool metal_graph_decode_q4_selected_slots_expected(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer,
        uint64_t                 gate_tensor_bytes,
        uint64_t                 down_tensor_bytes) {
    if (metal_graph_decode_pro_q4_expert_table_expected(g, layer,
                                                        gate_tensor_bytes,
                                                        down_tensor_bytes)) {
        return false;
    }
    const uint64_t q4_selected_min_tensor_bytes = 2ull * 1024ull * 1024ull * 1024ull;
    return !g->quality &&
           metal_graph_q4_selected_paths_allowed(g) &&
           layer->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
           layer->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
           layer->ffn_down_exps->type == DS4_TENSOR_Q4_K &&
           DS4_N_EXPERT_USED == 6 &&
           DS4_N_EXPERT >= 128 &&
           (g->ssd_streaming ||
            (gate_tensor_bytes >= q4_selected_min_tensor_bytes &&
             down_tensor_bytes >= q4_selected_min_tensor_bytes)) &&
           !glm_graph_env_present("DS4_ROCM_MOE_WRITE_CLAMPED_ACT",
                                  "DS4_METAL_MOE_WRITE_CLAMPED_ACT") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                                  "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_Q4_SELECTED_EXPERT_VIEWS",
                                  "DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS");
}

static bool metal_graph_decode_iq2_selected_slots_expected(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer) {
    return g &&
           g->ssd_streaming &&
           !g->quality &&
           layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
           layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
           layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
           DS4_N_EXPERT_USED == 6 &&
           DS4_N_EXPERT >= 128 &&
           !glm_graph_env_present("DS4_ROCM_MOE_WRITE_CLAMPED_ACT",
                                  "DS4_METAL_MOE_WRITE_CLAMPED_ACT") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                                  "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_IQ2_SELECTED_EXPERT_VIEWS",
                                  "DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS");
}

static bool metal_graph_decode_mxfp4_selected_slots_expected(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer) {
    return g &&
           g->ssd_streaming &&
           !g->quality &&
           layer->ffn_gate_exps->type == DS4_TENSOR_MXFP4 &&
           layer->ffn_up_exps->type == DS4_TENSOR_MXFP4 &&
           layer->ffn_down_exps->type == DS4_TENSOR_MXFP4 &&
           DS4_N_EXPERT_USED == 6 &&
           DS4_N_EXPERT >= 128 &&
           !glm_graph_env_present("DS4_ROCM_MOE_WRITE_CLAMPED_ACT",
                                  "DS4_METAL_MOE_WRITE_CLAMPED_ACT") &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                                  "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") &&
           getenv("DS4_METAL_DISABLE_MXFP4_SELECTED_EXPERT_VIEWS") == NULL;
}

static bool metal_graph_streaming_expert_cache_seed_layer_expected(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer) {
    if (!g ||
        !g->ssd_streaming ||
        !layer ||
        !layer->ffn_gate_exps ||
        !layer->ffn_up_exps ||
        !layer->ffn_down_exps) {
        return false;
    }
    if (metal_graph_decode_iq2_selected_slots_expected(g, layer) ||
        metal_graph_decode_mxfp4_selected_slots_expected(g, layer)) return true;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA &&
        !g->quality &&
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_IQ2_XXS &&
        DS4_N_EXPERT_USED != 0 &&
        DS4_N_EXPERT_USED <= 8 &&
        DS4_N_EXPERT >= 128 &&
        !glm_graph_env_present("DS4_ROCM_GLM_DISABLE_STREAMING_EXPERT_CACHE",
                               "DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE")) {
        return true;
    }
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_GLM_DSA ||
        g->quality ||
        layer->ffn_gate_exps->type != layer->ffn_up_exps->type ||
        layer->ffn_gate_exps->type != layer->ffn_down_exps->type ||
        DS4_N_EXPERT_USED == 0 ||
        DS4_N_EXPERT_USED > 8 ||
        DS4_N_EXPERT < 128 ||
        glm_graph_env_present("DS4_ROCM_GLM_DISABLE_STREAMING_EXPERT_CACHE",
                              "DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE")) {
        return false;
    }
    const uint32_t type = layer->ffn_gate_exps->type;
    return type == DS4_TENSOR_Q2_K || type == DS4_TENSOR_Q4_K;
}

static bool metal_graph_decode_cuda_selected_slots_expected(
        const ds4_gpu_graph     *g,
        const ds4_layer_weights *layer) {
#if !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU) && !defined(__APPLE__)
    if (!g ||
        !g->ssd_streaming ||
        g->quality ||
        !layer ||
        !layer->ffn_gate_exps ||
        !layer->ffn_up_exps ||
        !layer->ffn_down_exps ||
        DS4_N_EXPERT_USED != 6 ||
        DS4_N_EXPERT < 128 ||
        getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") != NULL ||
        getenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") != NULL) {
        return false;
    }
    const bool q4 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q4_K &&
        getenv("DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS") == NULL;
    const bool iq2 =
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
        getenv("DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS") == NULL;
    const bool mxfp4 =
        layer->ffn_gate_exps->type == DS4_TENSOR_MXFP4 &&
        layer->ffn_up_exps->type == DS4_TENSOR_MXFP4 &&
        layer->ffn_down_exps->type == DS4_TENSOR_MXFP4 &&
        getenv("DS4_METAL_DISABLE_MXFP4_SELECTED_EXPERT_VIEWS") == NULL;
    return q4 || iq2 || mxfp4;
#else
    (void)g;
    (void)layer;
    return false;
#endif
}

static uint32_t metal_graph_streaming_prefill_cache_seed_k(const ds4_gpu_graph *g) {
    const bool enabled =
        glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_CACHE_SEED",
                              "DS4_METAL_ENABLE_STREAMING_PREFILL_CACHE_SEED");
    if (!g ||
        !g->ssd_streaming ||
        !enabled) {
        return 0;
    }

    uint32_t k = 1;
    const char *env = glm_graph_env_value("DS4_ROCM_STREAMING_PREFILL_CACHE_SEED_K",
                                          "DS4_METAL_STREAMING_PREFILL_CACHE_SEED_K");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0') {
            if (v == 0) return 0;
            k = v > DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS ?
                DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS : (uint32_t)v;
        }
    }
    return k;
}

static bool metal_graph_streaming_prefill_cache_seed_enabled(const ds4_gpu_graph *g) {
    return metal_graph_streaming_prefill_cache_seed_k(g) != 0;
}

static bool metal_graph_streaming_expert_hotlist_enabled(const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           !g->ssd_streaming_cold &&
           !glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_EXPERT_HOTLIST",
                                  "DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST");
}

static bool metal_graph_streaming_expert_hotlist_add(
        uint32_t    layer,
        uint32_t    expert,
        uint32_t    priority,
        int32_t     experts[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    priorities[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    counts[DS4_MAX_LAYER],
        bool        seen[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t   *loaded) {
    if (layer >= DS4_N_LAYER || expert >= DS4_N_EXPERT) return true;
    if (layer >= DS4_MAX_LAYER || expert >= DS4_MAX_EXPERT) return true;
    if (seen[layer][expert]) return true;
    if (counts[layer] >= DS4_MAX_EXPERT) return false;
    seen[layer][expert] = true;
    if (priority == 0) priority = 1;
    priorities[layer][counts[layer]] = priority;
    experts[layer][counts[layer]++] = (int32_t)expert;
    (*loaded)++;
    return true;
}

static bool metal_graph_streaming_expert_hotlist_load_file(
        const char *path,
        uint32_t    max_entries,
        int32_t     experts[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    priorities[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    counts[DS4_MAX_LAYER],
        bool        seen[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t   *loaded_out) {
    if (!path || !path[0] || max_entries == 0 ||
        !experts || !priorities || !counts || !seen || !loaded_out) {
        return false;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr,
                "ds4: failed to open streaming expert hotlist %s: %s\n",
                path,
                strerror(errno));
        return false;
    }

    char line[256];
    uint64_t lineno = 0;
    uint32_t loaded = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (loaded >= max_entries) break;
        lineno++;
        char *p = line;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;

        errno = 0;
        char *end = NULL;
        unsigned long layer = strtoul(p, &end, 10);
        if (end == p || errno != 0) goto bad_line;
        p = end;
        while (*p && isspace((unsigned char)*p)) p++;

        errno = 0;
        unsigned long expert = strtoul(p, &end, 10);
        if (end == p || errno != 0) goto bad_line;
        p = end;
        while (*p && isspace((unsigned char)*p)) p++;

        errno = 0;
        unsigned long long hits = strtoull(p, &end, 10);
        if (end == p || errno != 0) goto bad_line;
        if (hits == 0) continue;
        const uint32_t priority =
            hits > UINT32_MAX ? UINT32_MAX : (uint32_t)hits;
        if (!metal_graph_streaming_expert_hotlist_add((uint32_t)layer,
                                                      (uint32_t)expert,
                                                      priority,
                                                      experts,
                                                      priorities,
                                                      counts,
                                                      seen,
                                                      &loaded)) {
            goto bad_line;
        }
        continue;

bad_line:
        fprintf(stderr,
                "ds4: invalid streaming expert hotlist line %" PRIu64 " in %s\n",
                lineno,
                path);
        fclose(fp);
        return false;
    }
    if (ferror(fp)) {
        fprintf(stderr,
                "ds4: failed to read streaming expert hotlist %s: %s\n",
                path,
                strerror(errno));
        fclose(fp);
        return false;
    }
    fclose(fp);

    if (loaded == 0) {
        fprintf(stderr, "ds4: streaming expert hotlist %s had no usable nonzero entries\n", path);
    }
    *loaded_out = loaded;
    return true;
}

static bool metal_graph_streaming_expert_hotlist_load_default(
        uint32_t    max_entries,
        int32_t     experts[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    priorities[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t    counts[DS4_MAX_LAYER],
        bool        seen[DS4_MAX_LAYER][DS4_MAX_EXPERT],
        uint32_t   *loaded_out) {
    if (max_entries == 0 || !experts || !priorities || !counts || !seen || !loaded_out) {
        return false;
    }
    const uint16_t (*hotlist)[2] = NULL;
    uint32_t hotlist_count = 0;
    if (g_ds4_shape.variant == DS4_VARIANT_PRO) {
        hotlist = ds4_default_streaming_hotlist_pro;
        hotlist_count = ds4_default_streaming_hotlist_pro_count;
    } else if (g_ds4_shape.variant == DS4_VARIANT_FLASH) {
        hotlist = ds4_default_streaming_hotlist_flash;
        hotlist_count = ds4_default_streaming_hotlist_flash_count;
    } else if (g_ds4_shape.variant == DS4_VARIANT_GLM52) {
        hotlist = ds4_default_streaming_hotlist_glm52;
        hotlist_count = ds4_default_streaming_hotlist_glm52_count;
    } else {
        *loaded_out = 0;
        return true;
    }
    uint32_t loaded = 0;
    for (uint32_t i = 0;
         i < hotlist_count && loaded < max_entries;
         i++) {
        if (!metal_graph_streaming_expert_hotlist_add(
                hotlist[i][0],
                hotlist[i][1],
                max_entries - loaded,
                experts,
                priorities,
                counts,
                seen,
                &loaded)) {
            return false;
        }
    }
    *loaded_out = loaded;
    return true;
}

static uint32_t metal_graph_streaming_expert_preload_count(
        const ds4_gpu_graph *g,
        uint32_t             cache_budget) {
    if (!g || cache_budget == 0) return 0;
    uint32_t preload = g->streaming_preload_experts;
    if (preload == 0) {
        preload = cache_budget;
        const char *env = glm_graph_env_value(
                "DS4_ROCM_STREAMING_EXPERT_AUTO_PRELOAD_CAP",
                "DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP");
#ifdef DS4_ROCM_BUILD
        if (g_ds4_shape.variant == DS4_VARIANT_GLM52 &&
            (!env || !env[0])) {
            return 0;
        }
#endif
        /* Auto mode is a hot seed, not a request to synchronously fill the
         * whole cache. Large Flash caches can otherwise spend startup doing
         * thousands of preads into shared Metal buffers and trip the system
         * watchdog before decode begins. ROCm GLM52 uses indexed batch prefill
         * by default, which already populates the cache; explicit CLI preload
         * counts and auto-preload env caps bypass that default. */
        uint32_t cap = 4096;
        if (env && env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(env, &end, 10);
            if (end != env && *end == '\0') {
                cap = v > UINT32_MAX ? UINT32_MAX : (uint32_t)v;
            }
        }
        if (cap != 0 && preload > cap) preload = cap;
    }
    if (preload > cache_budget) preload = cache_budget;
    const uint64_t max_possible = (uint64_t)DS4_N_LAYER * DS4_N_EXPERT;
    if ((uint64_t)preload > max_possible) preload = (uint32_t)max_possible;
    return preload;
}

#ifdef __APPLE__
/* Layer-major prefill has already made this layer's weights GPU-visible.
 * Preserve the built-in hot seed with model-to-cache blits before the next
 * layer replaces those mapped views, rather than rereading it after prefill. */
static bool metal_graph_seed_streaming_expert_cache_layer_from_mapped_hotlist(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il) {
    const char *hotlist_path = glm_graph_env_value(
            "DS4_ROCM_STREAMING_EXPERT_HOTLIST",
            "DS4_METAL_STREAMING_EXPERT_HOTLIST");
    if (!metal_graph_streaming_expert_hotlist_enabled(g) ||
        !model || !weights || il >= DS4_N_LAYER ||
        (hotlist_path && hotlist_path[0]) ||
        (g_ds4_shape.variant != DS4_VARIANT_FLASH &&
         g_ds4_shape.variant != DS4_VARIANT_PRO)) {
        return true;
    }

    const ds4_layer_weights *layer = &weights->layer[il];
    if (!metal_graph_streaming_expert_cache_seed_layer_expected(g, layer)) {
        return true;
    }
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    if (gate_row_bytes == 0 || down_row_bytes == 0 ||
        layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
        layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
        fprintf(stderr,
                "ds4: streaming resident hotlist seed byte size overflow at layer %u\n",
                il);
        return false;
    }
    const uint64_t gate_expert_bytes =
        layer->ffn_gate_exps->dim[1] * gate_row_bytes;
    const uint64_t down_expert_bytes =
        layer->ffn_down_exps->dim[1] * down_row_bytes;
    const uint32_t cache_budget =
        ds4_gpu_stream_expert_cache_budget_for_expert_size(gate_expert_bytes,
                                                            down_expert_bytes);
    const uint32_t preload_count =
        metal_graph_streaming_expert_preload_count(g, cache_budget);
    if (preload_count == 0) return true;

    const uint16_t (*hotlist)[2] = NULL;
    uint32_t hotlist_count = 0;
    if (g_ds4_shape.variant == DS4_VARIANT_FLASH) {
        hotlist = ds4_default_streaming_hotlist_flash;
        hotlist_count = ds4_default_streaming_hotlist_flash_count;
    } else {
        hotlist = ds4_default_streaming_hotlist_pro;
        hotlist_count = ds4_default_streaming_hotlist_pro_count;
    }
    const uint32_t target_count = preload_count < hotlist_count ?
        preload_count : hotlist_count;
    if (ds4_gpu_stream_expert_cache_current_count() >= target_count) {
        return true;
    }

    int32_t experts[DS4_MAX_EXPERT];
    uint32_t priorities[DS4_MAX_EXPERT];
    uint32_t n = 0;
    uint32_t loaded = 0;
    for (uint32_t i = 0;
         i < hotlist_count && loaded < preload_count;
         i++) {
        const uint32_t hot_layer = hotlist[i][0];
        const uint32_t hot_expert = hotlist[i][1];
        if (hot_layer >= DS4_N_LAYER || hot_expert >= DS4_N_EXPERT) continue;
        const uint32_t priority = preload_count - loaded;
        loaded++;
        if (hot_layer != il) continue;
        if (n >= DS4_MAX_EXPERT) return false;
        experts[n] = (int32_t)hot_expert;
        priorities[n] = priority;
        n++;
    }
    if (n == 0) return true;

    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(model,
                                       layer,
                                       il,
                                       gate_expert_bytes,
                                       down_expert_bytes);
    return ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(&table,
                                                              experts,
                                                              priorities,
                                                              n) != 0;
}
#endif

static bool metal_graph_decode_set_hash_selected_override(
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 token,
        uint64_t                 gate_tensor_bytes,
        uint64_t                 down_tensor_bytes,
        const ds4_gpu_graph     *g) {
    if (!layer->ffn_gate_tid2eid) return true;

    const bool q4_selected =
        metal_graph_decode_q4_selected_slots_expected(g,
                                                      layer,
                                                      gate_tensor_bytes,
                                                      down_tensor_bytes);
    const bool iq2_selected =
        metal_graph_decode_iq2_selected_slots_expected(g, layer);
    const bool mxfp4_selected =
        metal_graph_decode_mxfp4_selected_slots_expected(g, layer);
    if (!q4_selected && !iq2_selected && !mxfp4_selected) {
        return true;
    }

    int selected[DS4_MAX_EXPERT_USED];
    int32_t selected_i32[DS4_MAX_EXPERT_USED];
    layer_hash_selected_experts(selected, model, layer, (int)token);
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        selected_i32[i] = (int32_t)selected[i];
    }
    if (g && g->ssd_streaming) {
        if (DS4_N_EXPERT == 0 ||
            gate_tensor_bytes % DS4_N_EXPERT != 0 ||
            down_tensor_bytes % DS4_N_EXPERT != 0) {
            return false;
        }
        const uint64_t gate_expert_bytes = gate_tensor_bytes / DS4_N_EXPERT;
        const uint64_t down_expert_bytes = down_tensor_bytes / DS4_N_EXPERT;
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        if (ds4_gpu_stream_expert_cache_begin_selected_load(
                    &table,
                    selected_i32,
                    DS4_N_EXPERT_USED) == 0) {
            return false;
        }
    }
    return ds4_gpu_routed_moe_set_selected_override(selected_i32, DS4_N_EXPERT_USED) != 0;
}

static bool metal_graph_decode_cpu_router(
        ds4_gpu_graph          *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                token) {
    const bool profile =
        getenv("DS4_METAL_PRO_Q4_CPU_ROUTER_PROFILE") != NULL ||
        getenv("DS4_METAL_STREAMING_IQ2_CPU_ROUTER_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;
    if (ds4_gpu_end_commands() == 0) return false;
    const double t_sync = profile ? now_sec() : 0.0;
    if (ds4_gpu_tensor_read(metal_graph_ffn_norm(g),
                            0,
                            g->cpu_router_norm,
                            (uint64_t)DS4_N_EMBD * sizeof(g->cpu_router_norm[0])) == 0) {
        return false;
    }
    const double t_read = profile ? now_sec() : 0.0;

    float logits[DS4_MAX_EXPERT];
    float probs[DS4_MAX_EXPERT];
    int selected[DS4_MAX_EXPERT_USED];
    int32_t selected_i32[DS4_MAX_EXPERT_USED];
    float weights[DS4_MAX_EXPERT_USED];

    matvec_any(logits, model, layer->ffn_gate_inp, g->cpu_router_norm);
    for (uint32_t i = 0; i < DS4_N_EXPERT; i++) {
        probs[i] = sqrtf(softplus_stable(logits[i]));
    }
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, (int)token);
        layer_hash_router_weights_from_probs(weights, probs, selected);
    } else {
        layer_topk_selected_experts_from_probs(selected, weights, model, layer, probs);
    }
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        selected_i32[i] = (int32_t)selected[i];
    }
    const double t_cpu = profile ? now_sec() : 0.0;

    if (ds4_gpu_tensor_write(metal_graph_router_logits(g),
                             0,
                             logits,
                             (uint64_t)DS4_N_EXPERT * sizeof(logits[0])) == 0 ||
        ds4_gpu_tensor_write(metal_graph_router_probs(g),
                             0,
                             probs,
                             (uint64_t)DS4_N_EXPERT * sizeof(probs[0])) == 0 ||
        ds4_gpu_tensor_write(metal_graph_router_selected(g),
                             0,
                             selected_i32,
                             (uint64_t)DS4_N_EXPERT_USED * sizeof(selected_i32[0])) == 0 ||
        ds4_gpu_tensor_write(metal_graph_router_weights(g),
                             0,
                             weights,
                             (uint64_t)DS4_N_EXPERT_USED * sizeof(weights[0])) == 0) {
        return false;
    }
    const double t_write = profile ? now_sec() : 0.0;
    if (ds4_gpu_begin_commands() == 0) return false;
    if (ds4_gpu_routed_moe_set_selected_override(selected_i32, DS4_N_EXPERT_USED) == 0) return false;

    if (profile) {
        fprintf(stderr,
                "ds4: Metal CPU router layer=%u gate=%s down=%s sync=%.3f ms read=%.3f ms cpu=%.3f ms write=%.3f ms total=%.3f ms\n",
                il,
                tensor_type_name(layer->ffn_gate_exps->type),
                tensor_type_name(layer->ffn_down_exps->type),
                (t_sync - t0) * 1000.0,
                (t_read - t_sync) * 1000.0,
                (t_cpu - t_read) * 1000.0,
                (t_write - t_cpu) * 1000.0,
                (t_write - t0) * 1000.0);
    }
    return true;
}

static bool metal_graph_use_iq2_selected_readahead_shared_delay(
        const ds4_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           getenv("DS4_METAL_ENABLE_STREAMING_SELECTED_READAHEAD_SHARED_DELAY") != NULL &&
           getenv("DS4_METAL_DISABLE_STREAMING_SELECTED_READAHEAD_SHARED_DELAY") == NULL;
}

static bool metal_graph_decode_selected_readahead_override(
        ds4_gpu_graph            *g,
        const ds4_model          *model,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint64_t                  gate_expert_bytes,
        uint64_t                  down_expert_bytes) {
    if (!g || !model || !layer || !metal_graph_router_selected(g) ||
        DS4_N_EXPERT == 0 || DS4_N_EXPERT > DS4_MAX_EXPERT ||
        DS4_N_EXPERT_USED == 0 || DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return false;
    }

    const bool profile =
        getenv("DS4_METAL_STREAMING_SELECTED_READAHEAD_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;
    if (ds4_gpu_end_commands() == 0) return false;
    const double t_sync = profile ? now_sec() : 0.0;

    int32_t selected_ids[DS4_MAX_EXPERT_USED] = {0};
    if (ds4_gpu_tensor_read(metal_graph_router_selected(g),
                            0,
                            selected_ids,
                            (uint64_t)DS4_N_EXPERT_USED * sizeof(selected_ids[0])) == 0) {
        return false;
    }
    const double t_read = profile ? now_sec() : 0.0;

    bool seen[DS4_MAX_EXPERT] = {0};
    uint32_t unique = 0;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= DS4_N_EXPERT) {
            fprintf(stderr,
                    "ds4: Metal streaming selected readahead expert id %d is outside 0..%u at layer %u\n",
                    selected_ids[i],
                    DS4_N_EXPERT,
                    il);
            return false;
        }
        const uint32_t expert = (uint32_t)selected_ids[i];
        if (seen[expert]) continue;
        seen[expert] = true;
        unique++;

        const uint64_t expert_id = (uint64_t)expert;
        if (expert_id > UINT64_MAX / gate_expert_bytes ||
            expert_id > UINT64_MAX / down_expert_bytes) {
            fprintf(stderr, "ds4: Metal streaming selected readahead offset overflow\n");
            return false;
        }
        const uint64_t gate_rel = expert_id * gate_expert_bytes;
        const uint64_t down_rel = expert_id * down_expert_bytes;
        if (gate_rel > UINT64_MAX - layer->ffn_gate_exps->abs_offset ||
            gate_rel > UINT64_MAX - layer->ffn_up_exps->abs_offset ||
            down_rel > UINT64_MAX - layer->ffn_down_exps->abs_offset) {
            fprintf(stderr, "ds4: Metal streaming selected readahead offset overflow\n");
            return false;
        }

        metal_graph_stream_readahead_range_impl(model,
                                                layer->ffn_gate_exps->abs_offset + gate_rel,
                                                gate_expert_bytes,
                                                true);
        metal_graph_stream_readahead_range_impl(model,
                                                layer->ffn_up_exps->abs_offset + gate_rel,
                                                gate_expert_bytes,
                                                true);
        metal_graph_stream_readahead_range_impl(model,
                                                layer->ffn_down_exps->abs_offset + down_rel,
                                                down_expert_bytes,
                                                true);
    }
    const double t_hint = profile ? now_sec() : 0.0;

    if (ds4_gpu_routed_moe_set_selected_override(selected_ids,
                                                 DS4_N_EXPERT_USED) == 0) {
        return false;
    }
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(model,
                                       layer,
                                       il,
                                       gate_expert_bytes,
                                       down_expert_bytes);
    if (ds4_gpu_stream_expert_cache_begin_selected_load(
                &table,
                selected_ids,
                DS4_N_EXPERT_USED) == 0) {
        return false;
    }
    if (ds4_gpu_begin_commands() == 0) return false;
    const double t_done = profile ? now_sec() : 0.0;

    if (profile) {
        fprintf(stderr,
                "ds4: Metal streaming selected readahead layer=%u unique=%u sync=%.3f ms read=%.3f ms hint=%.3f ms resume=%.3f ms total=%.3f ms\n",
                il,
                unique,
                (t_sync - t0) * 1000.0,
                (t_read - t_sync) * 1000.0,
                (t_hint - t_read) * 1000.0,
                (t_done - t_hint) * 1000.0,
                (t_done - t0) * 1000.0);
    }
    return true;
}

static bool metal_graph_decode_cuda_selected_load(
        ds4_gpu_graph            *g,
        const ds4_model          *model,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint64_t                  gate_expert_bytes,
        uint64_t                  down_expert_bytes) {
#if !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU) && !defined(__APPLE__)
    if (!metal_graph_decode_cuda_selected_slots_expected(g, layer) ||
        !model ||
        !metal_graph_router_selected(g) ||
        DS4_N_EXPERT == 0 ||
        DS4_N_EXPERT > DS4_MAX_EXPERT ||
        DS4_N_EXPERT_USED == 0 ||
        DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return false;
    }

    const bool profile =
        getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;

    if (ds4_gpu_end_commands() == 0) return false;
    const double t_sync = profile ? now_sec() : 0.0;

    int32_t selected_ids[DS4_MAX_EXPERT_USED] = {0};
    bool ok = ds4_gpu_tensor_read(metal_graph_router_selected(g),
                                  0,
                                  selected_ids,
                                  (uint64_t)DS4_N_EXPERT_USED *
                                      sizeof(selected_ids[0])) != 0;
    const double t_read = profile ? now_sec() : 0.0;

    if (ok) {
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        ok = ds4_gpu_stream_expert_cache_begin_selected_load(
                    &table,
                    selected_ids,
                    DS4_N_EXPERT_USED) != 0;
    }
    const double t_load = profile ? now_sec() : 0.0;

    if (ds4_gpu_begin_commands() == 0) ok = false;
    const double t_done = profile ? now_sec() : 0.0;

    if (profile) {
        fprintf(stderr,
                "ds4: CUDA streaming selected load layer=%u sync=%.3f ms read=%.3f ms load=%.3f ms resume=%.3f ms total=%.3f ms\n",
                il,
                (t_sync - t0) * 1000.0,
                (t_read - t_sync) * 1000.0,
                (t_load - t_read) * 1000.0,
                (t_done - t_load) * 1000.0,
                (t_done - t0) * 1000.0);
    }
    return ok;
#else
    (void)g;
    (void)model;
    (void)layer;
    (void)il;
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    return false;
#endif
}

static bool metal_graph_cuda_stream_prefill_batch_selected_load(
        ds4_gpu_graph            *g,
        const ds4_model          *model,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint32_t                  n_tokens,
        uint64_t                  gate_expert_bytes,
        uint64_t                  down_expert_bytes) {
#if !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU) && !defined(__APPLE__)
    if (!metal_graph_decode_cuda_selected_slots_expected(g, layer) ||
        !model ||
        !metal_graph_batch_router_selected(g) ||
        n_tokens <= 1 ||
        DS4_N_EXPERT == 0 ||
        DS4_N_EXPERT_USED == 0 ||
        getenv("DS4_CUDA_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_LOAD") != NULL) {
        return true;
    }

    if ((uint64_t)n_tokens > UINT64_MAX / (uint64_t)DS4_N_EXPERT_USED) {
        fprintf(stderr, "ds4: CUDA streaming prefill selected-id count overflow at layer %u\n", il);
        return false;
    }
    const uint64_t n_ids64 = (uint64_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_ids64 == 0 || n_ids64 > SIZE_MAX / sizeof(int32_t)) {
        fprintf(stderr, "ds4: CUDA streaming prefill selected-id byte size overflow at layer %u\n", il);
        return false;
    }

    const bool profile =
        getenv("DS4_CUDA_STREAMING_PREFILL_BATCH_SELECTED_PROFILE") != NULL;
    const double t0 = profile ? now_sec() : 0.0;

    if (ds4_gpu_end_commands() == 0) return false;
    const double t_sync = profile ? now_sec() : 0.0;

    int32_t *selected_ids = xmalloc((size_t)n_ids64 * sizeof(selected_ids[0]));
    bool ok = ds4_gpu_tensor_read(metal_graph_batch_router_selected(g),
                                  0,
                                  selected_ids,
                                  n_ids64 * sizeof(selected_ids[0])) != 0;
    const double t_read = profile ? now_sec() : 0.0;
    if (ok) {
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        ok = ds4_gpu_stream_expert_cache_prepare_selected_batch(
                    &table,
                    selected_ids,
                    n_tokens,
                    DS4_N_EXPERT_USED) != 0;
    }
    free(selected_ids);
    const double t_load = profile ? now_sec() : 0.0;

    if (ds4_gpu_begin_commands() == 0) ok = false;
    const double t_done = profile ? now_sec() : 0.0;

    if (profile) {
        fprintf(stderr,
                "ds4: CUDA streaming prefill batch selected load layer=%u tokens=%u sync=%.3f ms read=%.3f ms load=%.3f ms resume=%.3f ms total=%.3f ms\n",
                il,
                n_tokens,
                (t_sync - t0) * 1000.0,
                (t_read - t_sync) * 1000.0,
                (t_load - t_read) * 1000.0,
                (t_done - t_load) * 1000.0,
                (t_done - t0) * 1000.0);
    }
    return ok;
#else
    (void)g;
    (void)model;
    (void)layer;
    (void)il;
    (void)n_tokens;
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    return true;
#endif
}

typedef struct metal_graph_selected_async_load {
    bool                      active;
    bool                      ok;
    /* Selected ids remain usable for a synchronous retry if the service
     * thread cannot stage the cache load without waiting on GPU work. */
    bool                      ids_ok;
    ds4_gpu_tensor           *router_selected;
    const ds4_model          *model;
    const ds4_layer_weights  *layer;
    uint32_t                  il;
    uint64_t                  event_value;
    uint64_t                  gate_expert_bytes;
    uint64_t                  down_expert_bytes;
    int32_t                   selected_ids[DS4_MAX_EXPERT_USED];
} metal_graph_selected_async_load;

static pthread_mutex_t g_metal_graph_selected_async_load_mutex =
    PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_metal_graph_selected_async_load_cond =
    PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_metal_graph_selected_async_load_done_cond =
    PTHREAD_COND_INITIALIZER;
static pthread_t g_metal_graph_selected_async_load_thread;
static bool g_metal_graph_selected_async_load_thread_started = false;
static bool g_metal_graph_selected_async_load_has_job = false;
static bool g_metal_graph_selected_async_load_done = false;
static metal_graph_selected_async_load g_metal_graph_selected_async_load_job;

static void metal_graph_selected_async_load_run(
        metal_graph_selected_async_load *job) {
    job->ok = false;

    if (!job->router_selected || !job->model || !job->layer ||
        DS4_N_EXPERT == 0 || DS4_N_EXPERT > DS4_MAX_EXPERT ||
        DS4_N_EXPERT_USED == 0 || DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return;
    }
    if (job->event_value != 0) {
#ifdef DS4_ROCM_BUILD
        if (ds4_gpu_tensor_read_after_selected_event(
                    job->router_selected,
                    0,
                    job->selected_ids,
                    (uint64_t)DS4_N_EXPERT_USED *
                        sizeof(job->selected_ids[0]),
                    job->event_value,
                    "selected-id async expert load") == 0) {
            return;
        }
#else
        if (ds4_gpu_wait_selected_readback_ready(job->event_value,
                                                 "selected-id async expert load") == 0) {
            return;
        }
        if (ds4_gpu_tensor_read(job->router_selected,
                                0,
                                job->selected_ids,
                                (uint64_t)DS4_N_EXPERT_USED *
                                    sizeof(job->selected_ids[0])) == 0) {
            return;
        }
#endif
    }
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        if (job->selected_ids[i] < 0 ||
            (uint32_t)job->selected_ids[i] >= DS4_N_EXPERT) {
            fprintf(stderr,
                    "ds4: Metal streaming async selected expert id %d is outside 0..%u at layer %u\n",
                    job->selected_ids[i],
                    DS4_N_EXPERT,
                    job->il);
            return;
        }
    }
    job->ids_ok = true;
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(job->model,
                                       job->layer,
                                       job->il,
                                       job->gate_expert_bytes,
                                       job->down_expert_bytes);
    if (ds4_gpu_stream_expert_cache_begin_selected_load(
                &table,
                job->selected_ids,
                DS4_N_EXPERT_USED) == 0) {
        return;
    }

    job->ok = true;
}

static void *metal_graph_selected_async_load_worker_main(void *arg) {
    (void)arg;
#ifdef __APPLE__
    /* The Metal cache paths must never wait on command buffers from this
     * thread while the main thread is encoding; register it so those waits
     * turn into load failures that the caller retries synchronously. */
    ds4_gpu_stream_expert_cache_note_service_thread();
#endif
    for (;;) {
        pthread_mutex_lock(&g_metal_graph_selected_async_load_mutex);
        while (!g_metal_graph_selected_async_load_has_job) {
            pthread_cond_wait(&g_metal_graph_selected_async_load_cond,
                              &g_metal_graph_selected_async_load_mutex);
        }
        metal_graph_selected_async_load job =
            g_metal_graph_selected_async_load_job;
        pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);

        metal_graph_selected_async_load_run(&job);

        pthread_mutex_lock(&g_metal_graph_selected_async_load_mutex);
        g_metal_graph_selected_async_load_job = job;
        g_metal_graph_selected_async_load_has_job = false;
        g_metal_graph_selected_async_load_done = true;
        pthread_cond_signal(&g_metal_graph_selected_async_load_done_cond);
        pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
    }
    return NULL;
}

static bool metal_graph_selected_async_load_ensure_worker(void) {
    pthread_mutex_lock(&g_metal_graph_selected_async_load_mutex);
    if (g_metal_graph_selected_async_load_thread_started) {
        pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
        return true;
    }
    const int rc = pthread_create(&g_metal_graph_selected_async_load_thread,
                                  NULL,
                                  metal_graph_selected_async_load_worker_main,
                                  NULL);
    if (rc != 0) {
        pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
        fprintf(stderr,
                "ds4: failed to start Metal streaming async selected load worker: %s\n",
                strerror(rc));
        return false;
    }
    g_metal_graph_selected_async_load_thread_started = true;
    pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
    return true;
}

static DS4_MAYBE_UNUSED bool metal_graph_selected_async_load_start_tensor(
        metal_graph_selected_async_load *job,
        ds4_gpu_tensor                  *router_selected,
        const ds4_model                 *model,
        const ds4_layer_weights         *layer,
        uint32_t                         il,
        uint64_t                         event_value,
        uint64_t                         gate_expert_bytes,
        uint64_t                         down_expert_bytes) {
    if (!job || !router_selected || event_value == 0) return false;
    if (!metal_graph_selected_async_load_ensure_worker()) return false;
    memset(job, 0, sizeof(*job));
    job->router_selected = router_selected;
    job->model = model;
    job->layer = layer;
    job->il = il;
    job->event_value = event_value;
    job->gate_expert_bytes = gate_expert_bytes;
    job->down_expert_bytes = down_expert_bytes;

    pthread_mutex_lock(&g_metal_graph_selected_async_load_mutex);
    if (g_metal_graph_selected_async_load_has_job ||
        g_metal_graph_selected_async_load_done) {
        pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
        return false;
    }
    g_metal_graph_selected_async_load_job = *job;
    g_metal_graph_selected_async_load_job.ok = false;
    g_metal_graph_selected_async_load_has_job = true;
    pthread_cond_signal(&g_metal_graph_selected_async_load_cond);
    pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
    job->active = true;
    return true;
}

static DS4_MAYBE_UNUSED bool metal_graph_selected_async_load_start(
        metal_graph_selected_async_load *job,
        ds4_gpu_graph                   *g,
        const ds4_model                 *model,
        const ds4_layer_weights         *layer,
        uint32_t                         il,
        uint64_t                         event_value,
        uint64_t                         gate_expert_bytes,
        uint64_t                         down_expert_bytes) {
    return metal_graph_selected_async_load_start_tensor(
            job,
            g ? metal_graph_router_selected(g) : NULL,
            model,
            layer,
            il,
            event_value,
            gate_expert_bytes,
            down_expert_bytes);
}

static bool metal_graph_selected_async_load_finish(
        metal_graph_selected_async_load *job) {
    if (!job || !job->active) return false;
    pthread_mutex_lock(&g_metal_graph_selected_async_load_mutex);
    while (!g_metal_graph_selected_async_load_done) {
        pthread_cond_wait(&g_metal_graph_selected_async_load_done_cond,
                          &g_metal_graph_selected_async_load_mutex);
    }
    *job = g_metal_graph_selected_async_load_job;
    g_metal_graph_selected_async_load_done = false;
    pthread_mutex_unlock(&g_metal_graph_selected_async_load_mutex);
    job->active = false;
    if (!job->ok) return false;
    return ds4_gpu_routed_moe_set_selected_override(job->selected_ids,
                                                   DS4_N_EXPERT_USED) != 0;
}

#ifdef DS4_ROCM_BUILD
typedef struct rocm_graph_batch_selected_async_load {
    bool                      active;
    bool                      ok;
    const ds4_gpu_tensor     *selected;
    const ds4_model          *model;
    const ds4_layer_weights  *layer;
    uint32_t                  il;
    uint32_t                  n_tokens;
    uint64_t                  event_value;
    uint64_t                  gate_expert_bytes;
    uint64_t                  down_expert_bytes;
    int32_t                  *selected_ids;
} rocm_graph_batch_selected_async_load;

static pthread_mutex_t g_rocm_graph_batch_selected_async_load_mutex =
    PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_rocm_graph_batch_selected_async_load_cond =
    PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_rocm_graph_batch_selected_async_load_done_cond =
    PTHREAD_COND_INITIALIZER;
static pthread_t g_rocm_graph_batch_selected_async_load_thread;
static bool g_rocm_graph_batch_selected_async_load_thread_started = false;
static bool g_rocm_graph_batch_selected_async_load_has_job = false;
static bool g_rocm_graph_batch_selected_async_load_done = false;
static rocm_graph_batch_selected_async_load
    g_rocm_graph_batch_selected_async_load_job;

static void rocm_graph_batch_selected_async_load_run(
        rocm_graph_batch_selected_async_load *job) {
    job->ok = false;
    if (!job->selected || !job->model || !job->layer || !job->selected_ids ||
        job->n_tokens <= 1 ||
        DS4_N_EXPERT == 0 || DS4_N_EXPERT > DS4_MAX_EXPERT ||
        DS4_N_EXPERT_USED == 0 || DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return;
    }
    if (DS4_N_EXPERT_USED != 0 &&
        job->n_tokens > UINT64_MAX / DS4_N_EXPERT_USED) {
        return;
    }
    const uint64_t n_ids = (uint64_t)job->n_tokens * DS4_N_EXPERT_USED;
    if (n_ids > SIZE_MAX / sizeof(job->selected_ids[0])) return;
    if (ds4_gpu_tensor_read_after_selected_event(
                job->selected,
                0,
                job->selected_ids,
                n_ids * sizeof(job->selected_ids[0]),
                job->event_value,
                "prefill selected-id async expert load") == 0) {
        return;
    }
    for (uint64_t i = 0; i < n_ids; i++) {
        if (job->selected_ids[i] < 0 ||
            (uint32_t)job->selected_ids[i] >= DS4_N_EXPERT) {
            fprintf(stderr,
                    "ds4: ROCm streaming async batch selected expert id %d "
                    "is outside 0..%u at layer %u\n",
                    job->selected_ids[i],
                    DS4_N_EXPERT,
                    job->il);
            return;
        }
    }
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(job->model,
                                       job->layer,
                                       job->il,
                                       job->gate_expert_bytes,
                                       job->down_expert_bytes);
    if (ds4_gpu_stream_expert_cache_prepare_selected_batch(
                &table,
                job->selected_ids,
                job->n_tokens,
                DS4_N_EXPERT_USED) == 0) {
        return;
    }
    job->ok = true;
}

static void *rocm_graph_batch_selected_async_load_worker_main(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_rocm_graph_batch_selected_async_load_mutex);
        while (!g_rocm_graph_batch_selected_async_load_has_job) {
            pthread_cond_wait(&g_rocm_graph_batch_selected_async_load_cond,
                              &g_rocm_graph_batch_selected_async_load_mutex);
        }
        rocm_graph_batch_selected_async_load job =
            g_rocm_graph_batch_selected_async_load_job;
        pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);

        rocm_graph_batch_selected_async_load_run(&job);

        pthread_mutex_lock(&g_rocm_graph_batch_selected_async_load_mutex);
        g_rocm_graph_batch_selected_async_load_job = job;
        g_rocm_graph_batch_selected_async_load_has_job = false;
        g_rocm_graph_batch_selected_async_load_done = true;
        pthread_cond_signal(&g_rocm_graph_batch_selected_async_load_done_cond);
        pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
    }
    return NULL;
}

static bool rocm_graph_batch_selected_async_load_ensure_worker(void) {
    pthread_mutex_lock(&g_rocm_graph_batch_selected_async_load_mutex);
    if (g_rocm_graph_batch_selected_async_load_thread_started) {
        pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
        return true;
    }
    const int rc = pthread_create(&g_rocm_graph_batch_selected_async_load_thread,
                                  NULL,
                                  rocm_graph_batch_selected_async_load_worker_main,
                                  NULL);
    if (rc != 0) {
        pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
        fprintf(stderr,
                "ds4: failed to start ROCm streaming async batch selected "
                "load worker: %s\n",
                strerror(rc));
        return false;
    }
    g_rocm_graph_batch_selected_async_load_thread_started = true;
    pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
    return true;
}

static bool rocm_graph_batch_selected_async_load_start(
        rocm_graph_batch_selected_async_load *job,
        const ds4_gpu_tensor                 *selected,
        const ds4_model                      *model,
        const ds4_layer_weights              *layer,
        uint32_t                              il,
        uint32_t                              n_tokens,
        uint64_t                              event_value,
        uint64_t                              gate_expert_bytes,
        uint64_t                              down_expert_bytes) {
    if (!job || !selected || event_value == 0 || n_tokens <= 1) return false;
    if (!rocm_graph_batch_selected_async_load_ensure_worker()) return false;
    if (DS4_N_EXPERT_USED != 0 &&
        n_tokens > UINT64_MAX / DS4_N_EXPERT_USED) {
        return false;
    }
    const uint64_t n_ids = (uint64_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_ids > SIZE_MAX / sizeof(int32_t)) return false;
    memset(job, 0, sizeof(*job));
    job->selected_ids = xmalloc((size_t)n_ids * sizeof(job->selected_ids[0]));
    job->selected = selected;
    job->model = model;
    job->layer = layer;
    job->il = il;
    job->n_tokens = n_tokens;
    job->event_value = event_value;
    job->gate_expert_bytes = gate_expert_bytes;
    job->down_expert_bytes = down_expert_bytes;

    pthread_mutex_lock(&g_rocm_graph_batch_selected_async_load_mutex);
    if (g_rocm_graph_batch_selected_async_load_has_job ||
        g_rocm_graph_batch_selected_async_load_done) {
        pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
        free(job->selected_ids);
        memset(job, 0, sizeof(*job));
        return false;
    }
    g_rocm_graph_batch_selected_async_load_job = *job;
    g_rocm_graph_batch_selected_async_load_job.ok = false;
    g_rocm_graph_batch_selected_async_load_has_job = true;
    pthread_cond_signal(&g_rocm_graph_batch_selected_async_load_cond);
    pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
    job->active = true;
    return true;
}

static bool rocm_graph_batch_selected_async_load_finish(
        rocm_graph_batch_selected_async_load *job) {
    if (!job || !job->active) return false;
    pthread_mutex_lock(&g_rocm_graph_batch_selected_async_load_mutex);
    while (!g_rocm_graph_batch_selected_async_load_done) {
        pthread_cond_wait(&g_rocm_graph_batch_selected_async_load_done_cond,
                          &g_rocm_graph_batch_selected_async_load_mutex);
    }
    *job = g_rocm_graph_batch_selected_async_load_job;
    g_rocm_graph_batch_selected_async_load_done = false;
    pthread_mutex_unlock(&g_rocm_graph_batch_selected_async_load_mutex);
    const bool ok = job->ok;
    free(job->selected_ids);
    memset(job, 0, sizeof(*job));
    return ok;
}
#endif

static bool metal_graph_profile_router_selection(
        ds4_gpu_graph            *g,
        const ds4_layer_weights  *layer,
        uint32_t                  il,
        uint32_t                  pos) {
    if (!g_expert_profile.active) return true;
    if (!g || !layer || !metal_graph_router_selected(g) || !metal_graph_router_weights(g)) return false;

    if (ds4_gpu_end_commands() == 0) {
        fprintf(stderr,
                "ds4: failed to end Metal command batch for expert profile readback\n");
        return false;
    }

    int32_t selected[DS4_MAX_EXPERT_USED] = {0};
    float weights[DS4_MAX_EXPERT_USED] = {0};
    const bool read_ok =
        ds4_gpu_tensor_read(metal_graph_router_selected(g),
                            0,
                            selected,
                            (uint64_t)DS4_N_EXPERT_USED * sizeof(selected[0])) != 0 &&
        ds4_gpu_tensor_read(metal_graph_router_weights(g),
                            0,
                            weights,
                            (uint64_t)DS4_N_EXPERT_USED * sizeof(weights[0])) != 0;

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr,
                "ds4: failed to resume Metal command batch after expert profile readback\n");
        return false;
    }
    if (!read_ok) {
        fprintf(stderr, "ds4: failed to read Metal router tensors for expert profile\n");
        return false;
    }

    ds4_expert_profile_record(il,
                              pos,
                              selected,
                              weights,
                              layer->ffn_gate_tid2eid != NULL);
    return true;
}

/* Diagnostic skip-ablation for TP profiling (DS4_TP_ABLATE=chain[,chain]):
 * drops whole encode chains so their true in-situ cost shows up as a t/s
 * delta.  Output is semantically wrong while enabled; both ranks must set
 * the same value.  Chains: hcpre, router, kv, compidx. */
static bool metal_graph_tp_ablate(const char *chain) {
    static const char *env = NULL;
    static int init = 0;
    if (!init) {
        env = getenv("DS4_TP_ABLATE");
        init = 1;
    }
    return env && strstr(env, chain) != NULL;
}

static bool metal_graph_borrow_tensor_view(
        ds4_gpu_tensor       *view,
        const ds4_gpu_tensor *base,
        uint64_t              offset,
        uint64_t              bytes) {
    if (!view || !base || offset > base->bytes || bytes > base->bytes - offset) {
        return false;
    }
    memset(view, 0, sizeof(*view));
    view->ptr = (char *)base->ptr + offset;
    view->bytes = bytes;
    view->owner = 0;
    view->device_id = base->device_id;
    return true;
}

static bool metal_graph_cuda_tp_ep_finish_reduce(
        ds4_gpu_graph *g,
        int            home_tier,
        int            partner_tier,
        bool           direct_return,
        uint64_t       return_bytes,
        bool           combine) {
    bool ok;
    if (direct_return) {
        ok = ds4_gpu_tensor_wait_xdev_default(
                g->routed_down_by_tier[partner_tier], home_tier) != 0;
    } else {
        ok = ds4_gpu_tensor_copy_xdev_default(
                g->tp_peer_tmp_by_tier[home_tier],
                g->routed_down_by_tier[partner_tier],
                return_bytes) != 0;
    }
    if (!combine) return ok;
    if (ok && g->cuda_tp_ep_pack_exact) {
        ok = ds4_gpu_routed_moe_owned_packed_combine_tensor(
                metal_graph_routed_out(g),
                metal_graph_routed_down(g),
                g->tp_peer_tmp_by_tier[home_tier],
                metal_graph_router_selected(g),
                DS4_N_EMBD,
                DS4_N_EXPERT / 2u) != 0;
    } else if (ok) {
        ok = ds4_gpu_routed_moe_owned_slots_combine_tensor(
                metal_graph_routed_out(g),
                metal_graph_routed_down(g),
                g->tp_peer_tmp_by_tier[home_tier],
                metal_graph_router_selected(g),
                DS4_N_EMBD,
                DS4_N_EXPERT / 2u) != 0;
    }
    return ok;
}

typedef enum {
    METAL_DECODE_LAYER_FULL = 0,
    METAL_DECODE_LAYER_TO_FFN,
    METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_FFN,
    METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_ATTN,
    METAL_DECODE_LAYER_TO_QKV,
    METAL_DECODE_LAYER_FROM_QKV_TO_ATTN,
    METAL_DECODE_LAYER_FROM_QA_KV_RAW_TO_SHARED_MID,
    METAL_DECODE_LAYER_FROM_KV_STORE_TO_ATTN,
    METAL_DECODE_LAYER_FROM_ATTN_TO_FFN,
    METAL_DECODE_LAYER_TO_ROUTER,
    METAL_DECODE_LAYER_TO_SHARED_MID,
    METAL_DECODE_LAYER_FROM_ROUTER,
} metal_decode_layer_phase;

#endif /* !DS4_NO_GPU */
