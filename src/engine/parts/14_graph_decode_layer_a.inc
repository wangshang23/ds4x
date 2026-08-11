/* M5 ports default to the existing standard M1-M4 Metal path.  Rollbacks
 * dominate the benchmark force-enables on pre-M5 devices so the aggregate
 * switch is a reliable whole-bundle control. */
static bool metal_graph_ported_m5_decode_feature_enabled(
        const char *pre_m5_disable_env,
        const char *m5_disable_env) {
#if defined(__APPLE__)
    if (m5_disable_env && getenv(m5_disable_env) != NULL) return false;
    const bool pre_m5 = ds4_gpu_device_is_pre_m5_apple_silicon();
    if (pre_m5 &&
        (getenv("DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS") != NULL ||
         (pre_m5_disable_env && getenv(pre_m5_disable_env) != NULL))) {
        return false;
    }
    return pre_m5 || ds4_gpu_device_is_m5_apple_silicon();
#else
    (void)pre_m5_disable_env;
    (void)m5_disable_env;
    return false;
#endif
}

static bool metal_graph_encode_decode_layer_phase(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token,
        metal_decode_layer_phase phase) {
    /* Defer the post-attention inverse RoPE tail into the FlashAttention reduce
     * kernel, which already owns each head's whole row. Removes one
     * 64-threadgroup dispatch per layer. */
    const bool fuse_attn_inv_rope =
        getenv("DS4_METAL_DISABLE_PRE_M5_ATTN_INV_ROPE_FUSE") == NULL &&
        (ds4_gpu_device_is_pre_m5_apple_silicon() ||
         ds4_gpu_device_is_m5_apple_silicon()) &&
        ds4_gpu_decode_attn_rope_fuse_available() != 0;
    /* The backend's consumed flag is process-global and remains true after a
     * gathered-attention layer. Track whether this layer actually armed the
     * fusion so a following indexed-attention layer cannot mistake that stale
     * flag for its own and skip the required standalone inverse RoPE. */
    bool attn_inv_rope_fuse_armed = false;
    /* switch to this layer's home tier before any Class P
     * accessor reads. Single-tier (placement == NULL): no-op. */
    if (g->placement) {
        const int this_tier = g->placement[il + 1];
        if (!metal_graph_set_active_tier_decode(g, this_tier)) return false;
    }
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t shared_dim = (uint32_t)layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    const bool compressed = ds4_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const bool qkv_rms_fused = !metal_graph_use_reference_qkv_norm();
    const int cuda_tp_home_tier = g->active_tier;
    const int cuda_tp_partner_tier = g->cuda_tp_decode
        ? metal_graph_cuda_tp_partner_tier(cuda_tp_home_tier) : -1;
    const bool tp_split_attn = g->tp_world == 2;
    const uint32_t tp_heads = tp_split_attn ?
        (uint32_t)DS4_N_HEAD / 2u : (uint32_t)DS4_N_HEAD;
    const uint32_t tp_head0 = tp_split_attn ? g->tp_rank * tp_heads : 0;

    bool ok = true;
    const bool decode_stage_profile = metal_graph_decode_stage_profile_enabled(il);
    double decode_stage_t0 = decode_stage_profile ? now_sec() : 0.0;
    const bool fuse_shared_gate_up =
        !g->quality &&
        g->tp_world < 2 &&
        layer->ffn_gate_shexp->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_up_shexp->type == DS4_TENSOR_Q8_0 &&
        g->shared_gate_up_swiglu_fuse;
    const bool keep_ffn_out = metal_graph_needs_ffn_out(g, il, pos);

    /* A concurrent compute encoder removes all of the generic routed-MoE
     * function's implicit dispatch ordering. Keep admission deliberately
     * narrower than the normal decode path: fixed resident IQ2 pair-SwiGLU,
     * direct Q2 top-6 sum, no readback/profiling/TP detours. */
#if defined(__APPLE__)
    const bool parallel_ffn_route_eligible =
        phase == METAL_DECODE_LAYER_FULL &&
        metal_graph_ported_m5_decode_feature_enabled(
            "DS4_METAL_DISABLE_PRE_M5_PARALLEL_FULL_FFN",
            "DS4_METAL_DISABLE_M5_PARALLEL_FULL_FFN") &&
        !g->quality && g->tp_world < 2 &&
        !g->ssd_streaming && !g->ssd_streaming_cold &&
        !g->cuda_tp_decode && !g->cuda_tp_moe && !g->cuda_tp_shared &&
        !decode_stage_profile &&
        !metal_graph_directional_steering_ffn_enabled(g) &&
        metal_graph_debug_get_config()->prefix == NULL &&
        getenv("DS4_METAL_MOE_ONE_STAGE_PROFILE") == NULL &&
        getenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT") == NULL &&
        getenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION") == NULL &&
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K &&
        DS4_N_EXPERT == 256u && DS4_N_EXPERT_USED == 6u &&
        expert_in_dim == 4096u && expert_mid_dim == 2048u &&
        routed_out_dim == 4096u && gate_row_bytes == 1056u &&
        gate_expert_bytes == 2162688u && down_row_bytes == 672u &&
        down_expert_bytes == 2752512u;
#else
    const bool parallel_ffn_route_eligible = false;
#endif
    const bool parallel_full_ffn_eligible =
        parallel_ffn_route_eligible &&
        getenv("DS4_METAL_Q8_MV_NSG") == NULL &&
        fuse_shared_gate_up &&
        layer->ffn_down_shexp->type == DS4_TENSOR_Q8_0 &&
        !keep_ffn_out && !metal_graph_use_reference_shared_down_hc() &&
        layer->ffn_gate_inp->type == DS4_TENSOR_F16 &&
        layer->ffn_gate_inp->dim[0] == DS4_N_EMBD &&
        layer->ffn_gate_inp->dim[1] == DS4_N_EXPERT;
#define DS4_METAL_PROFILE_DECODE_STAGE(name) do { \
        if (ok && decode_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("decode", (name), il, pos, 1, &decode_stage_t0); \
        } \
    } while (0)
    const bool tp_ablate_hcpre = metal_graph_tp_ablate("hcpre");
    /* Decode-island CUDA graph capture (design ported from the Entrpi/ds4
     * batched-serving fork).  Two position-independent islands of the
     * decode layer are captured and replayed: island 0 is the hc-pre
     * stage (exactly the METAL_DECODE_LAYER_TO_QKV prefix), island 1 is
     * the attention-output-to-layer-end tail (exactly the
     * METAL_DECODE_LAYER_FROM_ATTN_TO_FFN suffix).  Everything a capture
     * would freeze incorrectly -- TP, multi-tier, SSD streaming, debug
     * dumps, stage profiling, expert profiling, reference kernels, the
     * hash router's token argument -- disqualifies the layer and it runs
     * eagerly, byte-identical to before.  A replayed graph re-runs the
     * captured kernels byte-for-byte, so replay output is bit-identical
     * to the eager encode it recorded. */
    const bool decode_graphs_common_ok =
        !g->placement &&
        g->tp_world <= 1 &&
        !g->cuda_tp_decode &&
        !g->ssd_streaming &&
        !g->materialize_ffn_out &&
        !decode_stage_profile &&
        !g_expert_profile.active &&
        metal_graph_debug_get_config()->prefix == NULL &&
        ds4_gpu_decode_graphs_supported() != 0;
    const bool island_a_ok =
        decode_graphs_common_ok &&
        phase == METAL_DECODE_LAYER_FULL &&
        !tp_ablate_hcpre &&
        DS4_N_HC == 4 &&
        !metal_graph_use_reference_hc_decode() &&
        !metal_graph_use_reference_hc_norm_decode() &&
        !metal_graph_hc_norm_fusion_check_enabled();
    if (island_a_ok && ok) {
        ds4_decode_graph_key isla_key;
        memset(&isla_key, 0, sizeof(isla_key));
        isla_key.il = il;
        isla_key.island = 0u;
        isla_key.cur_hc = (void *)metal_graph_cur_hc(g);
        isla_key.after_attn_hc = (void *)metal_graph_after_attn_hc(g);
        isla_key.after_ffn_hc = (void *)metal_graph_after_ffn_hc(g);
        isla_key.attn_norm = (void *)metal_graph_attn_norm(g);
        for (;;) {
            const int isla_state = ds4_gpu_decode_graph_begin(&isla_key);
            if (isla_state == 1) {
                return metal_graph_encode_decode_layer_phase(
                        g, model, layer, il, pos, raw_cache, raw_cap,
                        raw_row, n_raw, token,
                        METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_FFN);
            }
            if (isla_state == 0) {
                const bool isla_ok = metal_graph_encode_decode_layer_phase(
                        g, model, layer, il, pos, raw_cache, raw_cap,
                        raw_row, n_raw, token,
                        METAL_DECODE_LAYER_TO_QKV);
                if (!isla_ok) {
                    /* Capture consumed no work; the entry is retired and
                     * the retry runs the island eagerly. */
                    ds4_gpu_decode_graph_abort(&isla_key);
                    continue;
                }
                if (ds4_gpu_decode_graph_end(&isla_key) == 0) {
                    return metal_graph_encode_decode_layer_phase(
                            g, model, layer, il, pos, raw_cache, raw_cap,
                            raw_row, n_raw, token,
                            METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_FFN);
                }
                continue;
            }
            break;      /* eager: disabled, warm pass, or retired entry */
        }
    }
    int router_shared_done = 0;
    int router_only_done = 0;
    bool router_project_select_fused = false;
    if (phase != METAL_DECODE_LAYER_FROM_ROUTER) {
    const bool fuse_hc_norm =
        DS4_N_HC == 4 &&
        !metal_graph_use_reference_hc_decode() &&
        !metal_graph_use_reference_hc_norm_decode();
    const bool stop_before_attn =
        phase == METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_ATTN ||
        phase == METAL_DECODE_LAYER_FROM_QKV_TO_ATTN ||
        phase == METAL_DECODE_LAYER_FROM_KV_STORE_TO_ATTN;
    const bool resume_after_qkv =
        phase == METAL_DECODE_LAYER_FROM_QKV_TO_ATTN ||
        phase == METAL_DECODE_LAYER_FROM_KV_STORE_TO_ATTN;
    const bool resume_after_qa_kv_raw =
        phase == METAL_DECODE_LAYER_FROM_QA_KV_RAW_TO_SHARED_MID;
    const bool resume_after_kv_store =
        phase == METAL_DECODE_LAYER_FROM_KV_STORE_TO_ATTN;
    const bool resume_after_attn =
        phase == METAL_DECODE_LAYER_FROM_ATTN_TO_FFN;
    bool cuda_tp_attn_heads_active = false;
    bool attn_inv_rope_done = resume_after_attn;
    if (phase != METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_FFN &&
        phase != METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_ATTN &&
        phase != METAL_DECODE_LAYER_FROM_QKV_TO_ATTN &&
        phase != METAL_DECODE_LAYER_FROM_QA_KV_RAW_TO_SHARED_MID &&
        phase != METAL_DECODE_LAYER_FROM_KV_STORE_TO_ATTN &&
        !resume_after_attn) {
    bool attn_hc_producer_pre_norm_fused = false;
    if (ok && !tp_ablate_hcpre) {
        /* Fused norm+mix removes one decode dispatch per layer; the kernel
         * reproduces both reduction trees bit-exactly (see dsv4_hc.metal). */
        const bool fuse_norm_mix =
            hc_dim == 16384u && mix_hc == 24u &&
            layer->hc_attn_fn->type == DS4_TENSOR_F16 &&
            !metal_graph_use_reference_hc_decode() &&
            getenv("DS4_METAL_DISABLE_PRE_M5_HC_NORM_MIX_FUSE") == NULL &&
            (ds4_gpu_device_is_pre_m5_apple_silicon() ||
             ds4_gpu_device_is_m5_apple_silicon()) &&
            ds4_gpu_hc_rms_norm_mix_f16_available() != 0;
#if defined(__APPLE__)
        const bool fuse_producer_pre_norm =
            fuse_norm_mix && fuse_hc_norm &&
            metal_graph_ported_m5_decode_feature_enabled(
                "DS4_METAL_DISABLE_PRE_M5_HC_PRODUCER_PRE_NORM_FUSE",
                "DS4_METAL_DISABLE_M5_HC_PRODUCER_PRE_NORM_FUSE");
        if (fuse_producer_pre_norm) {
            const int fused =
                ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor(
                    metal_graph_hc_mix(g),
                    metal_graph_attn_cur(g),
                    metal_graph_attn_norm(g),
                    metal_graph_hc_split(g),
                    metal_graph_cur_hc(g),
                    model->map,
                    model->size,
                    layer->hc_attn_fn->abs_offset,
                    layer->hc_attn_scale->abs_offset,
                    layer->hc_attn_base->abs_offset,
                    layer->attn_norm->abs_offset,
                    (uint32_t)hc_dim,
                    (uint32_t)mix_hc,
                    DS4_N_EMBD,
                    DS4_N_HC,
                    DS4_N_HC_SINKHORN_ITER,
                    DS4_RMS_EPS,
                    DS4_HC_EPS,
                    DS4_RMS_EPS);
            if (fused < 0) {
                ok = false;
            } else {
                attn_hc_producer_pre_norm_fused = fused > 0;
            }
        }
#endif
        if (ok && !attn_hc_producer_pre_norm_fused) {
            if (fuse_norm_mix) {
                ok = ds4_gpu_hc_rms_norm_mix_f16_tensor(
                        metal_graph_hc_mix(g), metal_graph_cur_hc(g),
                        model->map, model->size,
                        layer->hc_attn_fn->abs_offset,
                        (uint32_t)hc_dim, (uint32_t)mix_hc,
                        DS4_RMS_EPS) != 0;
            } else {
                ok = ds4_gpu_rms_norm_plain_tensor(
                        metal_graph_flat_hc(g), metal_graph_cur_hc(g),
                        (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
                if (ok) {
                    ok = metal_graph_matmul_plain_tensor(
                            metal_graph_hc_mix(g), model, layer->hc_attn_fn,
                            hc_dim, mix_hc, metal_graph_flat_hc(g), 1);
                }
            }
        }
    }
    if (ok && fuse_hc_norm) {
        if (!attn_hc_producer_pre_norm_fused) {
            ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(metal_graph_attn_cur(g),
                                                             metal_graph_attn_norm(g),
                                                             metal_graph_hc_split(g),
                                                             metal_graph_hc_mix(g),
                                                             metal_graph_cur_hc(g),
                                                             model->map,
                                                             model->size,
                                                             layer->hc_attn_scale->abs_offset,
                                                             layer->hc_attn_base->abs_offset,
                                                             layer->attn_norm->abs_offset,
                                                             DS4_N_EMBD,
                                                             DS4_N_HC,
                                                             DS4_N_HC_SINKHORN_ITER,
                                                             DS4_HC_EPS,
                                                             DS4_RMS_EPS) != 0;
        }
        if (ok) {
            ok = metal_graph_check_hc_norm_fusion("attn",
                                                  metal_graph_attn_cur(g),
                                                  metal_graph_attn_norm(g),
                                                  metal_graph_hc_mix(g),
                                                  metal_graph_cur_hc(g),
                                                  model,
                                                  layer->hc_attn_scale->abs_offset,
                                                  layer->hc_attn_base->abs_offset,
                                                  layer->attn_norm->abs_offset,
                                                  il,
                                                  pos);
        }
    } else if (ok) {
        ok = metal_graph_decode_hc_pre(metal_graph_attn_cur(g),
                                       metal_graph_hc_split(g),
                                       metal_graph_hc_mix(g),
                                       metal_graph_cur_hc(g),
                                       model,
                                       layer->hc_attn_scale->abs_offset,
                                       layer->hc_attn_base->abs_offset);
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_hc_pre");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre_mixes", metal_graph_hc_mix(g), mix_hc, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_weights", metal_graph_hc_pre(g), DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_post_weights", metal_graph_hc_post(g), DS4_N_HC, il, pos);
        metal_graph_debug_dump_tensor("hc_attn_pre_comb", metal_graph_hc_comb(g), (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre", metal_graph_attn_cur(g), DS4_N_EMBD, il, pos);
    }
    if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_attn_norm(g), metal_graph_attn_cur(g),
                                                                   model->map, model->size,
                                                                   layer->attn_norm->abs_offset,
                                                                   DS4_N_EMBD, DS4_RMS_EPS) != 0;
    DS4_METAL_PROFILE_DECODE_STAGE("attn_norm");
    if (ok) {
        metal_graph_debug_dump_tensor("attn_norm", metal_graph_attn_norm(g), DS4_N_EMBD, il, pos);
    }
    if (phase == METAL_DECODE_LAYER_TO_QKV) return ok;
    }
    bool kv_norm_store_fused = false;
    if (!resume_after_attn) {
    /* Fuse the KV RoPE tail into the FP8/raw finalizer: both were single
     * 64-thread threadgroups on the same 2 KB row, so the pair paid two full
     * dispatch launches. Deferring the RoPE and running it inside the
     * finalizer removes one dispatch per layer. */
    bool fuse_kv_rope_store = false;
    bool qkv_pair_quad_fused = false;
    if (!resume_after_qkv) {
    bool qkv_pair_projected = resume_after_qa_kv_raw;
    /* M1-M5 decode fusion: the q_a/kv Q8 pair and the four F16 compressor
     * projections all read the same normalized attention input and write
     * disjoint outputs, so one dispatch covers both stages with unchanged
     * per-row reduction trees (see the kernel comment).  Restricted to the
     * resident FULL phase so split-phase / CUDA / SSD flows keep their
     * original ordering. */
    if (!resume_after_qa_kv_raw && ok && qkv_rms_fused && compressed &&
        phase == METAL_DECODE_LAYER_FULL &&
        !g->ssd_streaming && !g->ssd_streaming_cold &&
        ds4_layer_compress_ratio(il) == 4u &&
        layer->attn_q_a->type == DS4_TENSOR_Q8_0 &&
        layer->attn_kv->type == DS4_TENSOR_Q8_0 &&
        g->cuda_qkv_pair && !metal_graph_use_reference_qkv_pair_proj() &&
        !metal_graph_use_reference_compressor_pair_proj() &&
        layer->attn_compressor_kv && layer->attn_compressor_gate &&
        layer->attn_compressor_ape &&
        layer->attn_compressor_kv->type == DS4_TENSOR_F16 &&
        layer->attn_compressor_gate->type == DS4_TENSOR_F16 &&
        layer->attn_compressor_kv->dim[0] == DS4_N_EMBD &&
        layer->attn_compressor_gate->dim[0] == DS4_N_EMBD &&
        layer->attn_compressor_kv->dim[1] == 2u * DS4_N_HEAD_DIM &&
        layer->attn_compressor_gate->dim[1] == 2u * DS4_N_HEAD_DIM &&
        layer->indexer_compressor_kv && layer->indexer_compressor_gate &&
        layer->indexer_compressor_ape &&
        layer->indexer_compressor_kv->type == DS4_TENSOR_F16 &&
        layer->indexer_compressor_gate->type == DS4_TENSOR_F16 &&
        layer->indexer_compressor_kv->dim[0] == DS4_N_EMBD &&
        layer->indexer_compressor_gate->dim[0] == DS4_N_EMBD &&
        layer->indexer_compressor_kv->dim[1] == 2u * DS4_N_INDEXER_HEAD_DIM &&
        layer->indexer_compressor_gate->dim[1] == 2u * DS4_N_INDEXER_HEAD_DIM &&
        metal_graph_ported_m5_decode_feature_enabled(
            "DS4_METAL_DISABLE_PRE_M5_QKV_PAIR_QUAD_FUSE",
            "DS4_METAL_DISABLE_M5_QKV_PAIR_QUAD_FUSE")) {
        const int fused = ds4_gpu_qkv_pair_quad_compressor_store_tensor(
                metal_graph_qr(g),
                metal_graph_kv_raw(g),
                metal_graph_comp_kv_cur(g),
                metal_graph_comp_sc_cur(g),
                metal_graph_index_comp_kv_cur(g),
                metal_graph_index_comp_sc_cur(g),
                g->layer_attn_state_kv[il],
                g->layer_attn_state_score[il],
                g->layer_index_state_kv[il],
                g->layer_index_state_score[il],
                model->map,
                model->size,
                layer->attn_q_a->abs_offset,
                layer->attn_kv->abs_offset,
                layer->attn_compressor_kv->abs_offset,
                layer->attn_compressor_gate->abs_offset,
                layer->indexer_compressor_kv->abs_offset,
                layer->indexer_compressor_gate->abs_offset,
                layer->attn_compressor_ape->abs_offset,
                layer->attn_compressor_ape->type,
                layer->indexer_compressor_ape->abs_offset,
                layer->indexer_compressor_ape->type,
                DS4_N_EMBD,
                q_rank,
                DS4_N_HEAD_DIM,
                2u * DS4_N_HEAD_DIM,
                2u * DS4_N_INDEXER_HEAD_DIM,
                metal_graph_attn_norm(g),
                4u,
                pos);
        if (fused < 0) {
            ok = false;
        } else if (fused > 0) {
            qkv_pair_projected = true;
            qkv_pair_quad_fused = true;
        }
    }
    if (!resume_after_qa_kv_raw && ok && !qkv_pair_quad_fused &&
        qkv_rms_fused && compressed &&
        phase == METAL_DECODE_LAYER_FULL &&
        !g->ssd_streaming && !g->ssd_streaming_cold &&
        ds4_layer_compress_ratio(il) == 128u &&
        layer->attn_q_a->type == DS4_TENSOR_Q8_0 &&
        layer->attn_kv->type == DS4_TENSOR_Q8_0 &&
        g->cuda_qkv_pair && !metal_graph_use_reference_qkv_pair_proj() &&
        !metal_graph_use_reference_compressor_pair_proj() &&
        layer->attn_compressor_kv && layer->attn_compressor_gate &&
        layer->attn_compressor_ape &&
        layer->attn_compressor_kv->type == DS4_TENSOR_F16 &&
        layer->attn_compressor_gate->type == DS4_TENSOR_F16 &&
        layer->attn_compressor_kv->dim[0] == DS4_N_EMBD &&
        layer->attn_compressor_gate->dim[0] == DS4_N_EMBD &&
        layer->attn_compressor_kv->dim[1] == DS4_N_HEAD_DIM &&
        layer->attn_compressor_gate->dim[1] == DS4_N_HEAD_DIM &&
        metal_graph_ported_m5_decode_feature_enabled(
            "DS4_METAL_DISABLE_PRE_M5_QKV_PAIR_COMPRESSOR_FUSE",
            "DS4_METAL_DISABLE_M5_QKV_PAIR_COMPRESSOR_FUSE")) {
        const int fused = ds4_gpu_qkv_pair_quad_compressor_store_tensor(
                metal_graph_qr(g),
                metal_graph_kv_raw(g),
                metal_graph_comp_kv_cur(g),
                metal_graph_comp_sc_cur(g),
                metal_graph_comp_kv_cur(g),
                metal_graph_comp_sc_cur(g),
                g->layer_attn_state_kv[il],
                g->layer_attn_state_score[il],
                g->layer_attn_state_kv[il],
                g->layer_attn_state_score[il],
                model->map,
                model->size,
                layer->attn_q_a->abs_offset,
                layer->attn_kv->abs_offset,
                layer->attn_compressor_kv->abs_offset,
                layer->attn_compressor_gate->abs_offset,
                layer->attn_compressor_kv->abs_offset,
                layer->attn_compressor_gate->abs_offset,
                layer->attn_compressor_ape->abs_offset,
                layer->attn_compressor_ape->type,
                layer->attn_compressor_ape->abs_offset,
                layer->attn_compressor_ape->type,
                DS4_N_EMBD,
                q_rank,
                DS4_N_HEAD_DIM,
                DS4_N_HEAD_DIM,
                0u,
                metal_graph_attn_norm(g),
                128u,
                pos);
        if (fused < 0) {
            ok = false;
        } else if (fused > 0) {
            qkv_pair_projected = true;
            qkv_pair_quad_fused = true;
        }
    }
    if (!resume_after_qa_kv_raw && ok && !qkv_pair_quad_fused && qkv_rms_fused &&
        layer->attn_q_a->type == DS4_TENSOR_Q8_0 &&
        layer->attn_kv->type == DS4_TENSOR_Q8_0 &&
        g->cuda_qkv_pair && !metal_graph_use_reference_qkv_pair_proj()) {
        qkv_pair_projected = ds4_gpu_matmul_q8_0_pair_tensor(
                metal_graph_qr(g),
                metal_graph_kv_raw(g),
                model->map,
                model->size,
                layer->attn_q_a->abs_offset,
                layer->attn_kv->abs_offset,
                DS4_N_EMBD,
                q_rank,
                DS4_N_HEAD_DIM,
                metal_graph_attn_norm(g),
                1) != 0;
    }
    if (!resume_after_qa_kv_raw && ok && !qkv_pair_projected) {
        ok = metal_graph_matmul_dense_quant_tensor(metal_graph_qr(g),
                                                   model,
                                                   layer->attn_q_a,
                                                   DS4_N_EMBD,
                                                   q_rank,
                                                   metal_graph_attn_norm(g),
                                                   1);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora", metal_graph_qr(g), q_rank, il, pos);
    }
    const bool kvnorm_dump = metal_graph_debug_wants("KVnorm", il, pos);
    bool kv_rope_fused = false;
    if (qkv_rms_fused) {
        if (!resume_after_qa_kv_raw && ok && !qkv_pair_projected) {
            ok = metal_graph_matmul_dense_quant_tensor(metal_graph_kv_raw(g),
                                                       model,
                                                       layer->attn_kv,
                                                       DS4_N_EMBD,
                                                       DS4_N_HEAD_DIM,
                                                       metal_graph_attn_norm(g),
                                                       1);
        }
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", metal_graph_kv_raw(g), DS4_N_HEAD_DIM, il, pos);
        }
        if (ok && g->cuda_qkv_kv_rope_fuse && !kvnorm_dump && DS4_N_HEAD_KV == 1u) {
            ok = ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
                    metal_graph_qr_norm(g),
                    metal_graph_qr(g),
                    model->map,
                    model->size,
                    layer->attn_q_a_norm->abs_offset,
                    (uint32_t)q_rank,
                    metal_graph_kv(g),
                    metal_graph_kv_raw(g),
                    layer->attn_kv_a_norm->abs_offset,
                    DS4_N_HEAD_DIM,
                    1,
                    DS4_N_HEAD_KV,
                    DS4_N_HEAD_DIM,
                    DS4_N_ROT,
                    pos,
                    compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                    false,
                    freq_base,
                    freq_scale,
                    ext_factor,
                    attn_factor,
                    DS4_ROPE_YARN_BETA_FAST,
                    DS4_ROPE_YARN_BETA_SLOW,
                    DS4_RMS_EPS) != 0;
            kv_rope_fused = ok;
        } else if (ok) {
            /* Triple fusion: q/kv norm + KV RoPE tail + FP8/raw store in one
             * dispatch; verbatim arithmetic (see metal/norm.metal). */
            if (!metal_graph_tp_ablate("kv") &&
                !kvnorm_dump &&
                !metal_graph_use_reference_kv_decode() &&
                !resume_after_kv_store &&
                DS4_N_HEAD_KV == 1u &&
                DS4_N_HEAD_DIM == 512u &&
                DS4_N_ROT == 64u &&
                raw_cache != NULL &&
                raw_row < raw_cap &&
                phase == METAL_DECODE_LAYER_FULL &&
                getenv("DS4_METAL_DISABLE_PRE_M5_QKV_NORM_KV_STORE_FUSE") == NULL &&
                (ds4_gpu_device_is_pre_m5_apple_silicon() ||
                 ds4_gpu_device_is_m5_apple_silicon()) &&
                ds4_gpu_kv_rope_fp8_fuse_available() != 0) {
                kv_norm_store_fused =
                    ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor(
                            metal_graph_qr_norm(g),
                            metal_graph_qr(g),
                            model->map,
                            model->size,
                            layer->attn_q_a_norm->abs_offset,
                            (uint32_t)q_rank,
                            metal_graph_kv(g),
                            metal_graph_kv_raw(g),
                            layer->attn_kv_a_norm->abs_offset,
                            DS4_N_HEAD_DIM,
                            raw_cache,
                            raw_cap,
                            raw_row,
                            DS4_N_ROT,
                            pos,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                if (!kv_norm_store_fused) ok = false;
                kv_rope_fused = kv_rope_fused || kv_norm_store_fused;
            }
            if (ok && !kv_norm_store_fused) {
            ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(metal_graph_qr_norm(g),
                                                         metal_graph_qr(g),
                                                         model->map,
                                                         model->size,
                                                         layer->attn_q_a_norm->abs_offset,
                                                         (uint32_t)q_rank,
                                                         metal_graph_kv(g),
                                                         metal_graph_kv_raw(g),
                                                         layer->attn_kv_a_norm->abs_offset,
                                                         DS4_N_HEAD_DIM,
                                                         1,
                                                         DS4_RMS_EPS) != 0;
            }
        }
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_qr_norm(g), metal_graph_qr(g),
                                                      model->map, model->size,
                                                      layer->attn_q_a_norm->abs_offset,
                                                      (uint32_t)q_rank, DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora_norm", metal_graph_qr_norm(g), q_rank, il, pos);
    }
    if (qkv_rms_fused && ok && !kv_rope_fused) {
        metal_graph_debug_dump_tensor("KVnorm", metal_graph_kv(g), DS4_N_HEAD_DIM, il, pos);
    }
    /* Phase B head slice: under the real TP split this rank computes only
     * its heads [tp_head0, tp_head0 + tp_heads) end to end — q_b rows, the
     * per-head norm/rope, the attention core and its owned output groups.
     * q and heads hold the owned half compactly at the buffer base; the
     * head range lines up with the output-group split (32 heads = 4 of the
     * 8 groups). */
    uint64_t tp_q_row_bytes = 0;
    if (ok) ok = metal_graph_dense_quant_row_bytes(layer->attn_q_b,
                                                   q_rank,
                                                   &tp_q_row_bytes);
    const uint64_t tp_q_rows_off =
        (uint64_t)tp_head0 * DS4_N_HEAD_DIM * tp_q_row_bytes;
    if (ok) ok = metal_graph_matmul_dense_quant_abs(metal_graph_q(g),
                                                    model,
                                                    layer->attn_q_b,
                                                    layer->attn_q_b->abs_offset + tp_q_rows_off,
                                                    q_rank,
                                                    (uint64_t)tp_heads * DS4_N_HEAD_DIM,
                                                    metal_graph_qr_norm(g),
                                                    1);
    if (ok) {
        metal_graph_debug_dump_tensor("Qraw", metal_graph_q(g), q_dim, il, pos);
    }
    const bool decode_q_norm_debug = metal_graph_debug_wants("Qnorm", il, pos);
    bool decode_q_norm_rope_fused = false;
    if (ok && !decode_q_norm_debug) {
        decode_q_norm_rope_fused =
            ds4_gpu_head_rms_norm_rope_tail_tensor(metal_graph_q(g),
                                                   1,
                                                   tp_heads,
                                                   DS4_N_HEAD_DIM,
                                                   DS4_N_ROT,
                                                   pos,
                                                   compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                   false,
                                                   freq_base,
                                                   freq_scale,
                                                   ext_factor,
                                                   attn_factor,
                                                   DS4_ROPE_YARN_BETA_FAST,
                                                   DS4_ROPE_YARN_BETA_SLOW,
                                                   DS4_RMS_EPS) != 0;
    }
    if (!decode_q_norm_rope_fused) {
        if (ok) ok = ds4_gpu_head_rms_norm_tensor(metal_graph_q(g), 1, tp_heads, DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("Qnorm", metal_graph_q(g), q_dim, il, pos);
        }
        if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_q(g), 1, tp_heads, DS4_N_HEAD_DIM,
                                                DS4_N_ROT, pos,
                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                false, freq_base, freq_scale, ext_factor, attn_factor,
                                                DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("q_path");
    if (ok) {
        metal_graph_debug_dump_tensor("Qcur", metal_graph_q(g), q_dim, il, pos);
    }
    if (!qkv_rms_fused) {
        if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_kv_raw(g),
                                                           model,
                                                           layer->attn_kv,
                                                           DS4_N_EMBD,
                                                           DS4_N_HEAD_DIM,
                                                           metal_graph_attn_norm(g),
                                                           1);
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", metal_graph_kv_raw(g), DS4_N_HEAD_DIM, il, pos);
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_kv(g), metal_graph_kv_raw(g),
                                                      model->map, model->size,
                                                      layer->attn_kv_a_norm->abs_offset,
                                                      DS4_N_HEAD_DIM, DS4_RMS_EPS) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVnorm", metal_graph_kv(g), DS4_N_HEAD_DIM, il, pos);
        }
    }
    const bool tp_ablate_kv = metal_graph_tp_ablate("kv");
    fuse_kv_rope_store =
        !tp_ablate_kv && !kv_rope_fused &&
        !metal_graph_use_reference_kv_decode() &&
        !resume_after_kv_store &&
        getenv("DS4_METAL_DISABLE_PRE_M5_KV_ROPE_FP8_FUSE") == NULL &&
        ds4_gpu_device_is_pre_m5_apple_silicon() &&
        ds4_gpu_kv_rope_fp8_fuse_available() != 0;
    if (ok && !tp_ablate_kv && !kv_rope_fused && !fuse_kv_rope_store) {
        ok = ds4_gpu_rope_tail_tensor(metal_graph_kv(g), 1,
                                      DS4_N_HEAD_KV, DS4_N_HEAD_DIM,
                                      DS4_N_ROT, pos,
                                      compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                      false, freq_base, freq_scale,
                                      ext_factor, attn_factor,
                                      DS4_ROPE_YARN_BETA_FAST,
                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("KVrope", metal_graph_kv(g), DS4_N_HEAD_DIM, il, pos);
    }
    }
    if (!resume_after_kv_store) {
        /* The common no-debug path may fuse KV RMS with RoPE above. KV
         * storage starts here after metal_graph_kv(g) contains the RoPE row. */
        if (ok && !kv_norm_store_fused) {
            ok = fuse_kv_rope_store
                ? (ds4_gpu_kv_rope_fp8_store_raw_tensor(
                       metal_graph_kv(g), raw_cache, raw_cap, raw_row,
                       DS4_N_HEAD_DIM, DS4_N_ROT, pos,
                       compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                       false, freq_base, freq_scale, ext_factor, attn_factor,
                       DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0)
                : metal_graph_decode_kv_store(metal_graph_kv(g), raw_cache, raw_cap, raw_row);
        }
        if (ok) ok = metal_graph_cuda_tp_attn_cache_sync_raw_row(g, il, raw_row);
        DS4_METAL_PROFILE_DECODE_STAGE("kv_path");
        if (ok) {
            metal_graph_debug_dump_tensor("KVcur", metal_graph_kv(g), DS4_N_HEAD_DIM, il, pos);
        }
    }

    uint32_t n_comp = 0;
    ds4_gpu_tensor *comp_cache = NULL;
    ds4_gpu_tensor *comp_selected = NULL;
    uint32_t n_selected = 0;
    double decode_index_stage_t0 = 0.0;
    const bool decode_index_stage_profile = g->decode_index_stage_profile;
    if (ok && compressed) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool emit = ((pos + 1u) % ratio) == 0u;
        if (!layer->attn_compressor_kv || !layer->attn_compressor_gate ||
            !layer->attn_compressor_ape || !layer->attn_compressor_norm ||
            layer->attn_compressor_kv->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_gate->type != DS4_TENSOR_F16 ||
            layer->attn_compressor_kv->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_gate->dim[0] != DS4_N_EMBD ||
            layer->attn_compressor_kv->dim[1] != comp_width ||
            layer->attn_compressor_gate->dim[1] != comp_width) {
            fprintf(stderr, "ds4: Metal graph compressor expects paired F16 compressor projections\n");
            ok = false;
        }
        if (ok && emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
            fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
            ok = false;
        }
        bool comp_state_already_stored = qkv_pair_quad_fused;
        /* Quad projection: the attention and indexer compressor pairs share
         * the normalized input and the F16 matvec shape, so a single dispatch
         * covers all four matrices with unchanged per-row reduction trees.
         * Removes one dispatch per decode layer.  Either preceding fused
         * QKV path may already have performed the same work. */
        int quad_store = comp_state_already_stored ? 1 : 0;
        if (ok && quad_store == 0 && ratio == 4u &&
            !metal_graph_use_reference_compressor_pair_proj() &&
            getenv("DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_QUAD_STORE") == NULL &&
            (ds4_gpu_device_is_pre_m5_apple_silicon() ||
             ds4_gpu_device_is_m5_apple_silicon()) &&
            layer->indexer_compressor_kv && layer->indexer_compressor_gate &&
            layer->indexer_compressor_ape &&
            layer->indexer_compressor_kv->type == DS4_TENSOR_F16 &&
            layer->indexer_compressor_gate->type == DS4_TENSOR_F16 &&
            layer->indexer_compressor_kv->dim[0] == DS4_N_EMBD &&
            layer->indexer_compressor_gate->dim[0] == DS4_N_EMBD &&
            layer->indexer_compressor_kv->dim[1] == 2u * DS4_N_INDEXER_HEAD_DIM &&
            layer->indexer_compressor_gate->dim[1] == 2u * DS4_N_INDEXER_HEAD_DIM) {
            quad_store =
                ds4_gpu_matmul_f16_quad_compressor_store_tensor(
                        metal_graph_comp_kv_cur(g),
                        metal_graph_comp_sc_cur(g),
                        metal_graph_index_comp_kv_cur(g),
                        metal_graph_index_comp_sc_cur(g),
                        g->layer_attn_state_kv[il],
                        g->layer_attn_state_score[il],
                        g->layer_index_state_kv[il],
                        g->layer_index_state_score[il],
                        model->map,
                        model->size,
                        layer->attn_compressor_kv->abs_offset,
                        layer->attn_compressor_gate->abs_offset,
                        layer->indexer_compressor_kv->abs_offset,
                        layer->indexer_compressor_gate->abs_offset,
                        layer->attn_compressor_ape->abs_offset,
                        layer->attn_compressor_ape->type,
                        layer->indexer_compressor_ape->abs_offset,
                        layer->indexer_compressor_ape->type,
                        DS4_N_EMBD,
                        comp_width,
                        2u * DS4_N_INDEXER_HEAD_DIM,
                        metal_graph_attn_norm(g),
                        ratio,
                        pos);
        }
        if (quad_store < 0) {
            ok = false;
        } else if (quad_store > 0) {
            comp_state_already_stored = true;
        }
        if (ok && quad_store == 0 && !metal_graph_use_reference_compressor_pair_proj()) {
            const int fused_store =
                ds4_gpu_matmul_f16_pair_compressor_store_tensor(
                        metal_graph_comp_kv_cur(g),
                        metal_graph_comp_sc_cur(g),
                        g->layer_attn_state_kv[il],
                        g->layer_attn_state_score[il],
                        model->map,
                        model->size,
                        layer->attn_compressor_kv->abs_offset,
                        layer->attn_compressor_gate->abs_offset,
                        layer->attn_compressor_ape->abs_offset,
                        layer->attn_compressor_ape->type,
                        DS4_N_EMBD,
                        comp_width,
                        metal_graph_attn_norm(g),
                        ratio,
                        pos);
            if (fused_store < 0) {
                ok = false;
            } else if (fused_store > 0) {
                comp_state_already_stored = true;
            } else {
                ok = ds4_gpu_matmul_f16_pair_tensor(metal_graph_comp_kv_cur(g),
                                                      metal_graph_comp_sc_cur(g),
                                                      model->map,
                                                      model->size,
                                                      layer->attn_compressor_kv->abs_offset,
                                                      layer->attn_compressor_gate->abs_offset,
                                                      DS4_N_EMBD,
                                                      comp_width,
                                                      metal_graph_attn_norm(g),
                                                      1) != 0;
            }
        } else if (quad_store == 0) {
            if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_comp_kv_cur(g), model->map, model->size,
                                                     layer->attn_compressor_kv->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     metal_graph_attn_norm(g), 1) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_comp_sc_cur(g), model->map, model->size,
                                                     layer->attn_compressor_gate->abs_offset,
                                                     DS4_N_EMBD, comp_width,
                                                     metal_graph_attn_norm(g), 1) != 0;
        }
        DS4_METAL_PROFILE_DECODE_STAGE("compressor_proj");
        const uint32_t comp_row = g->layer_n_comp[il];
        /* Eligible Metal decode emit-path fusion: the pooled attention and
         * indexer compressor rows each need norm + RoPE + quantize (+ F16
         * commit for attention) — seven tiny single-row dispatches per layer
         * every ratio-th token.  The fused kernel runs them as one
         * two-threadgroup dispatch with each standalone kernel's reduction
         * tree preserved bit-exactly (see kernel_dsv4_comp_row_finalize_f32).
         * The updates below then stop after the pool (and the state shift). */
        const bool comp_finalize_fuse =
            ok && emit && ratio == 4u && DS4_GPU_ATTN_COMP_CACHE_F16 &&
            phase == METAL_DECODE_LAYER_FULL &&
            !g->ssd_streaming && !g->ssd_streaming_cold &&
            DS4_N_HEAD_DIM == 512u && DS4_N_INDEXER_HEAD_DIM == 128u &&
            layer->attn_compressor_norm &&
            layer->attn_compressor_norm->type == DS4_TENSOR_F32 &&
            layer->indexer_compressor_norm &&
            layer->indexer_compressor_norm->type == DS4_TENSOR_F32 &&
            ds4_gpu_kv_rope_fp8_fuse_available() != 0 &&
            metal_graph_ported_m5_decode_feature_enabled(
                "DS4_METAL_DISABLE_PRE_M5_COMP_FINALIZE_FUSE",
                "DS4_METAL_DISABLE_M5_COMP_FINALIZE_FUSE");
        if (ok) ok = ds4_gpu_compressor_update_tensor(metal_graph_comp_kv_cur(g),
                                                        metal_graph_comp_sc_cur(g),
                                                        g->layer_attn_state_kv[il],
                                                        g->layer_attn_state_score[il],
                                                        metal_graph_attn_comp_update_target(g, il),
                                                        model->map,
                                                        model->size,
                                                        layer->attn_compressor_ape->abs_offset,
                                                        layer->attn_compressor_ape->type,
                                                        layer->attn_compressor_norm->abs_offset,
                                                        layer->attn_compressor_norm->type,
                                                        DS4_N_HEAD_DIM,
                                                        ratio,
                                                        pos,
                                                        metal_graph_attn_comp_update_row(comp_row),
                                                        DS4_N_ROT,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW,
                                                        DS4_RMS_EPS,
                                                        comp_state_already_stored,
                                                        true,
                                                        comp_finalize_fuse) != 0;
        DS4_METAL_PROFILE_DECODE_STAGE("compressor_update");
        if (ok && emit && !comp_finalize_fuse) {
            ds4_gpu_tensor *comp_row_view = metal_graph_attn_comp_row_view(g, il, comp_row);
            if (!comp_row_view) {
                ok = false;
            } else {
                ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view, 1, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
                if (ok) {
                    metal_graph_debug_dump_tensor("KVcompress", comp_row_view, DS4_N_HEAD_DIM, il, pos);
                }
            }
            ds4_gpu_tensor_free(comp_row_view);
            DS4_METAL_PROFILE_DECODE_STAGE("compressor_quantize");
            if (ok) ok = metal_graph_commit_attn_comp_stage(g, il, comp_row, 1);
            DS4_METAL_PROFILE_DECODE_STAGE("compressor_commit");
        }
        if (ok && emit) g->layer_n_comp[il]++;

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                layer->indexer_compressor_kv->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_gate->type != DS4_TENSOR_F16 ||
                layer->indexer_compressor_kv->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_gate->dim[0] != DS4_N_EMBD ||
                layer->indexer_compressor_kv->dim[1] != index_width ||
                layer->indexer_compressor_gate->dim[1] != index_width) {
                fprintf(stderr, "ds4: Metal graph indexer compressor expects paired F16 projections\n");
                ok = false;
            }
            if (ok && emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            bool index_state_already_stored = quad_store > 0;
            if (ok && quad_store == 0 && !metal_graph_use_reference_compressor_pair_proj()) {
                const int fused_store =
                    ds4_gpu_matmul_f16_pair_compressor_store_tensor(
                            metal_graph_comp_kv_cur(g),
                            metal_graph_comp_sc_cur(g),
                            g->layer_index_state_kv[il],
                            g->layer_index_state_score[il],
                            model->map,
                            model->size,
                            layer->indexer_compressor_kv->abs_offset,
                            layer->indexer_compressor_gate->abs_offset,
                            layer->indexer_compressor_ape->abs_offset,
                            layer->indexer_compressor_ape->type,
                            DS4_N_EMBD,
                            index_width,
                            metal_graph_attn_norm(g),
                            ratio,
                            pos);
                if (fused_store < 0) {
                    ok = false;
                } else if (fused_store > 0) {
                    index_state_already_stored = true;
                } else {
                    ok = ds4_gpu_matmul_f16_pair_tensor(metal_graph_comp_kv_cur(g),
                                                          metal_graph_comp_sc_cur(g),
                                                          model->map,
                                                          model->size,
                                                          layer->indexer_compressor_kv->abs_offset,
                                                          layer->indexer_compressor_gate->abs_offset,
                                                          DS4_N_EMBD,
                                                          index_width,
                                                          metal_graph_attn_norm(g),
                                                          1) != 0;
                }
            } else if (quad_store == 0) {
                if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_comp_kv_cur(g), model->map, model->size,
                                                         layer->indexer_compressor_kv->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         metal_graph_attn_norm(g), 1) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_comp_sc_cur(g), model->map, model->size,
                                                         layer->indexer_compressor_gate->abs_offset,
                                                         DS4_N_EMBD, index_width,
                                                         metal_graph_attn_norm(g), 1) != 0;
            }
            DS4_METAL_PROFILE_DECODE_STAGE("indexer_compressor_proj");
            const uint32_t index_row = g->layer_n_index_comp[il];
            if (ok) ok = ds4_gpu_compressor_update_tensor(metal_graph_comp_kv_cur(g),
                                                            metal_graph_comp_sc_cur(g),
                                                            g->layer_index_state_kv[il],
                                                            g->layer_index_state_score[il],
                                                            metal_graph_index_comp_stage(g),
                                                            model->map,
                                                            model->size,
                                                            layer->indexer_compressor_ape->abs_offset,
                                                            layer->indexer_compressor_ape->type,
                                                            layer->indexer_compressor_norm->abs_offset,
                                                            layer->indexer_compressor_norm->type,
                                                            DS4_N_INDEXER_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            0,
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS,
                                                            index_state_already_stored,
                                                            true,
                                                            comp_finalize_fuse) != 0;
            DS4_METAL_PROFILE_DECODE_STAGE("indexer_compressor_update");
            if (ok && emit && comp_finalize_fuse) {
                /* One dispatch finalizes both pooled compressor rows; the
                 * attention row lives at offset 0 of the staging tensor and
                 * commits to the F16 cache, the indexer row finalizes in
                 * place.  A missing pipeline fails loudly rather than
                 * skipping the finalize the updates deferred. */
                ok = ds4_gpu_dsv4_comp_row_finalize_tensor(
                        metal_graph_attn_comp_stage(g),
                        g->layer_attn_comp_cache[il],
                        comp_row,
                        layer->attn_compressor_norm->abs_offset,
                        g->layer_index_comp_cache[il],
                        index_row,
                        layer->indexer_compressor_norm->abs_offset,
                        g->layer_attn_state_kv[il],
                        g->layer_attn_state_score[il],
                        g->layer_index_state_kv[il],
                        g->layer_index_state_score[il],
                        model->map,
                        model->size,
                        pos + 1u - ratio,
                        DS4_N_ROT,
                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0u,
                        freq_base,
                        freq_scale,
                        ext_factor,
                        attn_factor,
                        DS4_ROPE_YARN_BETA_FAST,
                        DS4_ROPE_YARN_BETA_SLOW,
                        DS4_RMS_EPS) == 1;
            }
            if (ok && emit && !comp_finalize_fuse) {
                ds4_gpu_tensor *index_row_view = ds4_gpu_tensor_view(
                        metal_graph_index_comp_stage(g),
                        0,
                        (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                if (!index_row_view) {
                    ok = false;
                } else {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(index_row_view,
                                                          1,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                }
                ds4_gpu_tensor_free(index_row_view);
                if (ok) {
                    ok = ds4_gpu_spark_pack_index_rows_tensor(
                            g->layer_index_comp_cache[il],
                            index_row,
                            metal_graph_index_comp_stage(g),
                            0,
                            1) != 0;
                }
                DS4_METAL_PROFILE_DECODE_STAGE("indexer_compressor_qat");
            }
            if (ok && emit) g->layer_n_index_comp[il]++;
            const uint32_t decode_sparse_threshold =
                metal_graph_decode_indexer_sparse_threshold(g);
            if (ok &&
                g->layer_n_comp[il] > decode_sparse_threshold &&
                g->layer_n_index_comp[il] > DS4_N_INDEXER_TOP_K) {
                const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
                if (!layer->indexer_attn_q_b ||
                    !tensor_type_is_f16_or_q8_0(layer->indexer_attn_q_b->type) ||
                    layer->indexer_attn_q_b->dim[0] != q_rank ||
                    layer->indexer_attn_q_b->dim[1] != indexer_q_dim) {
                    fprintf(stderr, "ds4: Metal graph indexer q projection expects F16 or Q8_0 weights\n");
                    ok = false;
                }
                if (ok && (!layer->indexer_proj ||
                           layer->indexer_proj->type != DS4_TENSOR_F16 ||
                           layer->indexer_proj->dim[0] != DS4_N_EMBD ||
                           layer->indexer_proj->dim[1] != DS4_N_INDEXER_HEAD)) {
                    fprintf(stderr, "ds4: Metal graph indexer weight projection expects F16 weights\n");
                    ok = false;
                }
                if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_indexer_q(g),
                                                              model,
                                                              layer->indexer_attn_q_b,
                                                              q_rank,
                                                              indexer_q_dim,
                                                              metal_graph_qr_norm(g),
                                                              1);
                if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_indexer_q(g), 1,
                                                        DS4_N_INDEXER_HEAD,
                                                        DS4_N_INDEXER_HEAD_DIM,
                                                        DS4_N_ROT,
                                                        pos,
                                                        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                        false,
                                                        freq_base,
                                                        freq_scale,
                                                        ext_factor,
                                                        attn_factor,
                                                        DS4_ROPE_YARN_BETA_FAST,
                                                        DS4_ROPE_YARN_BETA_SLOW) != 0;
                if (ok) ok = ds4_gpu_dsv4_indexer_qat_tensor(metal_graph_indexer_q(g),
                                                              DS4_N_INDEXER_HEAD,
                                                              DS4_N_INDEXER_HEAD_DIM) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_indexer_weights(g), model->map, model->size,
                                                         layer->indexer_proj->abs_offset,
                                                         DS4_N_EMBD, DS4_N_INDEXER_HEAD,
                                                         metal_graph_attn_norm(g), 1) != 0;
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_score_one_tensor(metal_graph_indexer_scores(g),
                                                                metal_graph_indexer_q(g),
                                                                metal_graph_indexer_weights(g),
                                                                g->layer_index_comp_cache[il],
                                                                g->layer_n_index_comp[il],
                                                                DS4_N_INDEXER_HEAD,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                index_scale) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("decode_score",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                if (ok) ok = ds4_gpu_indexer_topk_tensor(metal_graph_comp_selected(g),
                                                           metal_graph_indexer_scores(g),
                                                           g->layer_n_index_comp[il],
                                                           1,
                                                           DS4_N_INDEXER_TOP_K) != 0;
                if (ok && decode_index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("decode_topk",
                                                                    il,
                                                                    pos,
                                                                    1,
                                                                    g->layer_n_index_comp[il],
                                                                    &decode_index_stage_t0);
                }
                /* Decode used to materialize a dense compressed-row mask and
                 * call the generic gathered FlashAttention wrapper below.
                 * That wrapper scans every compressed row and rejects long
                 * contexts once raw+compressed rows exceed 8192.  Ratio-4 DS4
                 * attention is sparse after indexer top-k, so use the private
                 * indexed attention kernel instead: it scans only SWA raw rows
                 * plus the selected compressed rows, matching prefill and
                 * avoiding the long-context decode failure. */
                if (ok) {
                    comp_selected = metal_graph_comp_selected(g);
                    /*
                     * Contract: the indexer top-k is fixed by the model config
                     * and must remain the full 512 rows.  Do not reduce this for
                     * throughput benchmarks.
                     *
                     * Why: the indexer is not just an implementation detail.  It
                     * decides which compressed memory rows are visible to the
                     * attention kernel.  If we keep only 128/256 rows, the later
                     * indexed-attention math may be perfectly computed, but it is
                     * computed over the wrong candidate set: rows ranked 257-512
                     * are removed before softmax/PV can use them.  Those rows may
                     * carry weak-but-necessary evidence for retrieval, name/number
                     * recall, or long-context disambiguation.  The error is
                     * therefore semantic/algorithmic, not the acceptable kind of
                     * local numerical drift caused by a different reduction order
                     * or Tensor/NAX precision.
                     *
                     * Short prompt tests, first-token agreement, or even a small
                     * official-vector set can miss this because many prompts do
                     * not need the tail of the 512 selected compressed rows.  The
                     * failure appears only when the model needs information that
                     * fell below the reduced cutoff.  Optimizations belong inside
                     * the score/top-k/attention implementation while preserving
                     * DS4_N_INDEXER_TOP_K.
                     */
                    n_selected = DS4_N_INDEXER_TOP_K < g->layer_n_index_comp[il]
                        ? DS4_N_INDEXER_TOP_K
                        : g->layer_n_index_comp[il];
                }
            }
        }

        n_comp = g->layer_n_comp[il];
        comp_cache = g->layer_attn_comp_cache[il];
    }
    DS4_METAL_PROFILE_DECODE_STAGE("compressor_indexer");

    if (stop_before_attn) return ok;
    if (ok) {
        const uint32_t raw_start = metal_graph_raw_start_for_span(g, pos, n_raw);
        const bool indexed_attention = n_comp != 0 && comp_selected != NULL && n_selected != 0;
        const bool cuda_tp_attn_heads_requested = g->cuda_tp_attn_heads;
        const uint32_t cuda_tp_heads = DS4_N_HEAD / 2u;
        const bool cuda_tp_attn_local_cache =
            metal_graph_cuda_tp_attn_cache_dup_layer_ready(g, il);
        cuda_tp_attn_heads_active =
            cuda_tp_attn_heads_requested &&
            cuda_tp_partner_tier >= 0 &&
            g_gpu_peer_ok[cuda_tp_partner_tier][cuda_tp_home_tier] &&
            (DS4_N_HEAD % 2u) == 0u &&
            (n_groups % 2u) == 0u &&
            g->q_by_tier[cuda_tp_partner_tier] &&
            g->heads_by_tier[cuda_tp_partner_tier] &&
            !metal_graph_debug_wants("kqv_out", il, pos) &&
            !metal_graph_debug_wants("kqv_back", il, pos);
        if (cuda_tp_attn_heads_requested && !cuda_tp_attn_heads_active) {
            fprintf(stderr,
                    "ds4: CUDA decode TP cannot split attention heads for tier %d "
                    "(partner=%d heads=%u peer=%d)\n",
                    cuda_tp_home_tier,
                    cuda_tp_partner_tier,
                    DS4_N_HEAD,
                    cuda_tp_partner_tier >= 0 ?
                        g_gpu_peer_ok[cuda_tp_partner_tier][cuda_tp_home_tier] : 0);
            ok = false;
        }
        if (ok && cuda_tp_attn_heads_active) {
            const uint64_t tp_head_bytes = (uint64_t)cuda_tp_heads * DS4_N_HEAD_DIM * sizeof(float);
            ds4_gpu_tensor q_peer_src;
            ds4_gpu_tensor q_peer_dst;
            ds4_gpu_tensor peer_heads_tail;
            ok = metal_graph_borrow_tensor_view(&q_peer_src,
                                                metal_graph_q(g),
                                                tp_head_bytes,
                                                tp_head_bytes) &&
                 metal_graph_borrow_tensor_view(&q_peer_dst,
                                                g->q_by_tier[cuda_tp_partner_tier],
                                                0,
                                                tp_head_bytes) &&
                 metal_graph_borrow_tensor_view(&peer_heads_tail,
                                                g->heads_by_tier[cuda_tp_partner_tier],
                                                tp_head_bytes,
                                                tp_head_bytes);
            if (ok) {
                ok = ds4_gpu_tensor_copy_xdev(&q_peer_dst, &q_peer_src,
                                              tp_head_bytes) != 0;
            }
            ds4_gpu_tensor *peer_raw_cache = cuda_tp_attn_local_cache
                ? g->layer_raw_cache_tp[il] : raw_cache;
            ds4_gpu_tensor *peer_comp_cache = cuda_tp_attn_local_cache
                ? g->layer_attn_comp_cache_tp[il] : comp_cache;
            ds4_gpu_tensor *peer_selected = comp_selected;
            if (ok && indexed_attention && cuda_tp_attn_local_cache) {
                peer_selected = g->comp_selected_by_tier[cuda_tp_partner_tier];
                ok = peer_selected &&
                     ds4_gpu_tensor_copy_xdev(peer_selected,
                                              comp_selected,
                                              (uint64_t)n_selected * sizeof(int32_t)) != 0;
            }
            if (ok && !cuda_tp_attn_local_cache) {
                ok = ds4_gpu_tensor_wait_xdev(raw_cache, cuda_tp_partner_tier) != 0;
            }
            if (ok) ok = ds4_gpu_set_current_device(cuda_tp_partner_tier) == 0;
            if (ok && indexed_attention) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                        &peer_heads_tail,
                        model->map,
                        model->size,
                        layer->attn_sinks->abs_offset + (uint64_t)cuda_tp_heads * sizeof(float),
                        &q_peer_dst,
                        peer_raw_cache,
                        peer_comp_cache,
                        metal_graph_attn_comp_cache_format(),
                        peer_selected,
                        1,
                        pos,
                        n_raw,
                        raw_cap,
                        raw_start,
                        n_comp,
                        n_selected,
                        g->raw_window,
                        ds4_layer_compress_ratio(il),
                        cuda_tp_heads,
                        DS4_N_HEAD_DIM) != 0;
            } else if (ok) {
                ok = ds4_gpu_attention_decode_heads_tensor(
                        &peer_heads_tail,
                        model->map,
                        model->size,
                        layer->attn_sinks->abs_offset + (uint64_t)cuda_tp_heads * sizeof(float),
                        &q_peer_dst,
                        peer_raw_cache,
                        n_raw,
                        raw_cap,
                        raw_start,
                        n_comp ? peer_comp_cache : NULL,
                        metal_graph_attn_comp_cache_format(),
                        n_comp,
                        NULL,
                        0,
                        cuda_tp_heads,
                        DS4_N_HEAD_DIM) != 0;
            }
            if (ok) {
                ok = ds4_gpu_rope_tail_tensor(&peer_heads_tail,
                                              1, cuda_tp_heads, DS4_N_HEAD_DIM,
                                              DS4_N_ROT, pos,
                                              compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                              true,
                                              freq_base,
                                              freq_scale,
                                              ext_factor,
                                              attn_factor,
                                              DS4_ROPE_YARN_BETA_FAST,
                                              DS4_ROPE_YARN_BETA_SLOW) != 0;
            }
            if (ok) ok = ds4_gpu_set_current_device(cuda_tp_home_tier) == 0;
            if (ok && indexed_attention) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                        metal_graph_heads(g),
                        model->map,
                        model->size,
                        layer->attn_sinks->abs_offset,
                        metal_graph_q(g),
                        raw_cache,
                        g->layer_attn_comp_cache[il],
                        metal_graph_attn_comp_cache_format(),
                        comp_selected,
                        1,
                        pos,
                        n_raw,
                        raw_cap,
                        raw_start,
                        n_comp,
                        n_selected,
                        g->raw_window,
                        ds4_layer_compress_ratio(il),
                        cuda_tp_heads,
                        DS4_N_HEAD_DIM) != 0;
            } else if (ok) {
                ok = ds4_gpu_attention_decode_heads_tensor(metal_graph_heads(g),
                                                             model->map, model->size,
                                                             layer->attn_sinks->abs_offset,
                                                             metal_graph_q(g), raw_cache, n_raw,
                                                             raw_cap,
                                                             raw_start,
                                                             n_comp ? comp_cache : NULL,
                                                             metal_graph_attn_comp_cache_format(),
                                                             n_comp,
                                                             NULL,
                                                             0,
                                                             cuda_tp_heads, DS4_N_HEAD_DIM) != 0;
            }
            if (ok) {
                ok = ds4_gpu_rope_tail_tensor(metal_graph_heads(g),
                                                1, cuda_tp_heads, DS4_N_HEAD_DIM,
                                                DS4_N_ROT, pos,
                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                true,
                                                freq_base,
                                                freq_scale,
                                                ext_factor,
                                                attn_factor,
                                                DS4_ROPE_YARN_BETA_FAST,
                                                DS4_ROPE_YARN_BETA_SLOW) != 0;
            }
            if (ok && indexed_attention && decode_index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary("decode_attention",
                                                                il,
                                                                pos,
                                                                1,
                                                                n_comp,
                                                                &decode_index_stage_t0);
            }
        } else if (ok && indexed_attention) {
            ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                    metal_graph_heads(g),
                    model->map,
                    model->size,
                    layer->attn_sinks->abs_offset + (uint64_t)tp_head0 * (layer->attn_sinks->bytes / DS4_N_HEAD),
                    metal_graph_q(g),
                    raw_cache,
                    g->layer_attn_comp_cache[il],
                    metal_graph_attn_comp_cache_format(),
                    comp_selected,
                    1,
                    pos,
                    n_raw,
                    raw_cap,
                    raw_start,
                    n_comp,
                    n_selected,
                    g->raw_window,
                    ds4_layer_compress_ratio(il),
                    tp_heads,
                    DS4_N_HEAD_DIM) != 0;
            if (ok && decode_index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary("decode_attention",
                                                                il,
                                                                pos,
                                                                1,
                                                                n_comp,
                                                                &decode_index_stage_t0);
            }
        } else {
            /* Defer the inverse RoPE tail into the FlashAttention reduce: each
             * reduce threadgroup already owns one head's entire row, so the
             * rotation is an intra-threadgroup dependency and the separate
             * 64-threadgroup RoPE dispatch disappears. */
            if (fuse_attn_inv_rope) {
                ds4_gpu_set_decode_attn_rope_fuse(
                    DS4_N_HEAD_DIM, DS4_N_ROT, pos,
                    compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                    true, freq_base, freq_scale, ext_factor, attn_factor,
                    DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW);
                attn_inv_rope_fuse_armed = true;
            }
            ok = ds4_gpu_attention_decode_heads_tensor(metal_graph_heads(g),
                                                         model->map, model->size,
                                                         layer->attn_sinks->abs_offset + (uint64_t)tp_head0 * (layer->attn_sinks->bytes / DS4_N_HEAD),
                                                         metal_graph_q(g), raw_cache, n_raw,
                                                         raw_cap,
                                                         raw_start,
                                                         n_comp ? comp_cache : NULL,
                                                         metal_graph_attn_comp_cache_format(),
                                                         n_comp,
                                                         NULL,
                                                         0,
                                                         tp_heads, DS4_N_HEAD_DIM) != 0;
        }
    }
    }
    if (ok && !cuda_tp_attn_heads_active && !attn_inv_rope_done &&
        !(attn_inv_rope_fuse_armed && ds4_gpu_decode_attn_rope_fuse_used())) {
        ok = ds4_gpu_rope_tail_tensor(metal_graph_heads(g),
                                      1, tp_heads, DS4_N_HEAD_DIM,
                                      DS4_N_ROT, pos,
                                      compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                      true,
                                      freq_base,
                                      freq_scale,
                                      ext_factor,
                                      attn_factor,
                                      DS4_ROPE_YARN_BETA_FAST,
                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_inv_rope");
    if (ok && !cuda_tp_attn_heads_active) {
        metal_graph_debug_dump_tensor("kqv_back", metal_graph_heads(g), q_dim, il, pos);
    }
    ds4_gpu_tensor *cuda_tp_attn_peer = NULL;
    bool cuda_tp_attn_hc_fused = false;
    const bool cuda_tp_attn_requested = g->cuda_tp_attn;
    const bool cuda_tp_attn =
        cuda_tp_attn_requested &&
        !metal_graph_directional_steering_attn_enabled(g) &&
        cuda_tp_partner_tier >= 0 &&
        (n_groups % 2u) == 0u;
    ds4_gpu_tensor *tp_attn_a = NULL;   /* rank partials consumed directly */
    ds4_gpu_tensor *tp_attn_b = NULL;   /* by the HC expand */
    const bool fuse_attn_out_hc =
        !cuda_tp_attn &&
        g->tp_world < 2 &&
        layer->attn_output_a->type == DS4_TENSOR_Q8_0 &&
        layer->attn_output_b->type == DS4_TENSOR_Q8_0 &&
        !metal_graph_directional_steering_attn_enabled(g) &&
        !metal_graph_use_reference_attn_out_hc();
    const bool fuse_tp_attn_out_hc =
        cuda_tp_attn &&
        !metal_graph_use_reference_attn_out_hc() &&
        g->cuda_tp_attn_out_hc_fuse;
    /* Decode island 1: the attention-output-to-layer-end tail, which is
     * position-independent (the hash router's token argument disqualifies
     * layers with ffn_gate_tid2eid).  Capture and replay both go through
     * the METAL_DECODE_LAYER_FROM_ATTN_TO_FFN phase, which encodes
     * exactly this suffix; the phase check keeps the capture recursion
     * from re-entering this block. */
    const bool island_b_ok =
        decode_graphs_common_ok &&
        ok &&
        (phase == METAL_DECODE_LAYER_FULL ||
         phase == METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_FFN) &&
        fuse_attn_out_hc &&
        !cuda_tp_attn &&
        !cuda_tp_attn_heads_active &&
        layer->ffn_gate_tid2eid == NULL;
    if (island_b_ok) {
        ds4_decode_graph_key islb_key;
        memset(&islb_key, 0, sizeof(islb_key));
        islb_key.il = il;
        islb_key.island = 1u;
        islb_key.cur_hc = (void *)metal_graph_cur_hc(g);
        islb_key.after_attn_hc = (void *)metal_graph_after_attn_hc(g);
        islb_key.after_ffn_hc = (void *)metal_graph_after_ffn_hc(g);
        islb_key.attn_norm = (void *)metal_graph_attn_norm(g);
        for (;;) {
            const int islb_state = ds4_gpu_decode_graph_begin(&islb_key);
            if (islb_state == 1) return ok;
            if (islb_state == 0) {
                const bool islb_ok = metal_graph_encode_decode_layer_phase(
                        g, model, layer, il, pos, raw_cache, raw_cap,
                        raw_row, n_raw, token,
                        METAL_DECODE_LAYER_FROM_ATTN_TO_FFN);
                if (!islb_ok) {
                    ds4_gpu_decode_graph_abort(&islb_key);
                    continue;
                }
                if (ds4_gpu_decode_graph_end(&islb_key) == 0) return true;
                continue;
            }
            break;
        }
    }
    if (ok && cuda_tp_attn_requested && !cuda_tp_attn) {
        fprintf(stderr,
                "ds4: CUDA decode TP cannot split attention output for tier %d "
                "(partner=%d groups=%u)\n",
                cuda_tp_home_tier, cuda_tp_partner_tier, n_groups);
        ok = false;
    }
    if (ok && cuda_tp_attn) {
        const uint32_t tp_groups = n_groups / 2u;
        const uint64_t tp_heads_bytes = (uint64_t)tp_groups * group_dim * sizeof(float);
        const uint64_t tp_heads_off = (uint64_t)tp_groups * group_dim * sizeof(float);
        const bool cuda_tp_attn_peer_read =
            !cuda_tp_attn_heads_active &&
            g->cuda_tp_attn_peer_read &&
            g_gpu_peer_ok[cuda_tp_partner_tier][cuda_tp_home_tier];
        ds4_gpu_tensor peer_heads_dst_view;
        ds4_gpu_tensor peer_heads_src_view;
        ds4_gpu_tensor *peer_heads_dst = NULL;
        ds4_gpu_tensor *peer_heads_src = NULL;
        if (cuda_tp_attn_heads_active) {
            peer_heads_src = g->heads_by_tier[cuda_tp_partner_tier];
        } else if (!cuda_tp_attn_peer_read) {
            ok = metal_graph_borrow_tensor_view(
                    &peer_heads_dst_view,
                    g->heads_by_tier[cuda_tp_partner_tier],
                    tp_heads_off,
                    tp_heads_bytes);
            if (ok) peer_heads_dst = &peer_heads_dst_view;
        }
        if (ok) {
            if (cuda_tp_attn_heads_active) {
                peer_heads_src = g->heads_by_tier[cuda_tp_partner_tier];
            } else if (cuda_tp_attn_peer_read) {
                peer_heads_src = metal_graph_heads(g);
            } else {
                ok = metal_graph_borrow_tensor_view(&peer_heads_src_view,
                                                    metal_graph_heads(g),
                                                    tp_heads_off,
                                                    tp_heads_bytes);
                if (ok) peer_heads_src = &peer_heads_src_view;
            }
        }
        ds4_gpu_tensor *peer_heads = cuda_tp_attn_peer_read ?
            metal_graph_heads(g) : g->heads_by_tier[cuda_tp_partner_tier];
        ok = (cuda_tp_attn_heads_active || cuda_tp_attn_peer_read || peer_heads_dst) &&
             peer_heads_src && peer_heads &&
             g->attn_low_by_tier[cuda_tp_partner_tier] &&
             g->attn_out_by_tier[cuda_tp_partner_tier] &&
             g->tp_peer_tmp_by_tier[cuda_tp_home_tier];
        if (ok && !cuda_tp_attn_heads_active && !cuda_tp_attn_peer_read) {
            ok = ds4_gpu_tensor_copy_xdev(peer_heads_dst, peer_heads_src,
                                          tp_heads_bytes) != 0;
        } else if (ok && cuda_tp_attn_peer_read) {
            ok = ds4_gpu_tensor_wait_xdev(peer_heads_src, cuda_tp_partner_tier) != 0;
        }
        if (ok) ok = ds4_gpu_set_current_device(cuda_tp_partner_tier) == 0;
        if (ok) {
            ok = ds4_gpu_attention_output_q8_tp_tensor(
                    g->attn_out_by_tier[cuda_tp_partner_tier],
                    g->attn_low_by_tier[cuda_tp_partner_tier],
                    model->map,
                    model->size,
                    layer->attn_output_a->abs_offset,
                    layer->attn_output_b->abs_offset,
                    group_dim,
                    rank,
                    n_groups,
                    tp_groups,
                    tp_groups,
                    DS4_N_EMBD,
                    peer_heads) != 0;
        }
        if (ok) ok = ds4_gpu_set_current_device(cuda_tp_home_tier) == 0;
        if (ok && fuse_tp_attn_out_hc) {
            ok = ds4_gpu_attention_output_low_q8_tensor(
                    metal_graph_attn_low(g),
                    model->map,
                    model->size,
                    layer->attn_output_a->abs_offset,
                    group_dim,
                    rank,
                    tp_groups,
                    metal_graph_heads(g)) != 0;
        } else if (ok) {
            ok = ds4_gpu_attention_output_q8_tp_tensor(
                    metal_graph_attn_out(g),
                    metal_graph_attn_low(g),
                    model->map,
                    model->size,
                    layer->attn_output_a->abs_offset,
                    layer->attn_output_b->abs_offset,
                    group_dim,
                    rank,
                    n_groups,
                    0,
                    tp_groups,
                    DS4_N_EMBD,
                    metal_graph_heads(g)) != 0;
        }
        if (ok) {
            ok = ds4_gpu_tensor_copy_xdev(g->tp_peer_tmp_by_tier[cuda_tp_home_tier],
                                          g->attn_out_by_tier[cuda_tp_partner_tier],
                                          (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
            if (ok) cuda_tp_attn_peer = g->tp_peer_tmp_by_tier[cuda_tp_home_tier];
        }
        if (ok && fuse_tp_attn_out_hc) {
            ok = ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
                    metal_graph_after_attn_hc(g),
                    metal_graph_attn_out(g),
                    model->map,
                    model->size,
                    layer->attn_output_b->abs_offset,
                    (uint64_t)n_groups * rank,
                    DS4_N_EMBD,
                    0,
                    (uint64_t)tp_groups * rank,
                    metal_graph_attn_low(g),
                    cuda_tp_attn_peer,
                    metal_graph_cur_hc(g),
                    metal_graph_hc_split(g),
                    DS4_N_EMBD,
                    DS4_N_HC) != 0;
            if (ok) cuda_tp_attn_hc_fused = true;
        }
    } else if (ok && fuse_attn_out_hc) {
        ok = ds4_gpu_attention_output_low_q8_tensor(metal_graph_attn_low(g),
                                                      model->map,
                                                      model->size,
                                                      layer->attn_output_a->abs_offset,
                                                      group_dim,
                                                      rank,
                                                      n_groups,
                                                      metal_graph_heads(g)) != 0;
        if (ok) {
            ok = ds4_gpu_matmul_q8_0_hc_expand_tensor(metal_graph_after_attn_hc(g),
                                                        metal_graph_attn_out(g),
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_b->abs_offset,
                                                        (uint64_t)n_groups * rank,
                                                        DS4_N_EMBD,
                                                        metal_graph_attn_low(g),
                                                        metal_graph_cur_hc(g),
                                                        metal_graph_hc_split(g),
                                                        DS4_N_EMBD,
                                                        DS4_N_HC) != 0;
        }
    } else if (ok && g->tp_world == 2) {
        /* Group-sliced attention output: this rank computes its half of the
         * output groups and the matching k-window of the expand projection,
         * leaving a partial block output in the gate slot. */
        const uint32_t tp_groups = n_groups / 2;
        ok = metal_graph_attention_output_dense_quant_tp(
                g->tp_out[il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_ATTN],
                metal_graph_attn_low(g),
                g,
                model,
                layer->attn_output_a,
                layer->attn_output_b,
                group_dim, rank,
                n_groups,
                g->tp_rank * tp_groups, tp_groups,
                DS4_N_EMBD,
                metal_graph_heads(g));
    } else if (ok && layer->attn_output_a->type != DS4_TENSOR_Q8_0) {
        ds4_gpu_tensor *attn_out_dst = g->tp_world == 2 ?
                g->tp_out[il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_ATTN] : metal_graph_attn_out(g);
        ok = metal_graph_attention_output_dense_quant_low(metal_graph_attn_low(g),
                                                          g,
                                                          model,
                                                          layer->attn_output_a,
                                                          group_dim,
                                                          rank,
                                                          0,
                                                          n_groups,
                                                          metal_graph_heads(g));
        if (ok) ok = metal_graph_matmul_dense_quant_tensor(attn_out_dst,
                                                           model,
                                                           layer->attn_output_b,
                                                           (uint64_t)n_groups * rank,
                                                           DS4_N_EMBD,
                                                           metal_graph_attn_low(g),
                                                           1);
    } else if (ok) {
        ds4_gpu_tensor *attn_out_dst = g->tp_world == 2 ?
                g->tp_out[il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_ATTN] : metal_graph_attn_out(g);
        ok = ds4_gpu_attention_output_q8_batch_tensor(attn_out_dst,
                                                        metal_graph_attn_low(g),
                                                        metal_graph_batch_group_tmp(g),
                                                        metal_graph_batch_low_tmp(g),
                                                        model->map,
                                                        model->size,
                                                        layer->attn_output_a->abs_offset,
                                                        layer->attn_output_b->abs_offset,
                                                        group_dim, rank,
                                                        n_groups, DS4_N_EMBD,
                                                        metal_graph_heads(g), 1) != 0;
    }
    if (ok && g->tp_world == 2) {
        /* Gate ATTN: exchange the attention block output with the peer and
         * rebuild the canonical sum (rank0 first, then rank1) in attn_out
         * on both ranks — identical expression on both machines keeps them
         * bit-exact. */
        const uint32_t slot = il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_ATTN;
        ok = ds4_gpu_tp_gate_encode(il, DS4_TP_GATE_ATTN) != 0;
        if (ok) {
            ds4_gpu_tensor *first = g->tp_rank == 0 ? g->tp_out[slot] : g->tp_in[slot];
            if (metal_graph_directional_steering_attn_enabled(g)) {
                ds4_gpu_tensor *second = g->tp_rank == 0 ? g->tp_in[slot] : g->tp_out[slot];
                ok = ds4_gpu_add_tensor(metal_graph_attn_out(g), first, second, DS4_N_EMBD) != 0;
            } else {
                /* Combine folded into the HC expand below; attn_out is not
                 * materialized on this path. */
                tp_attn_a = first;
                tp_attn_b = g->tp_rank == 0 ? g->tp_in[slot] : g->tp_out[slot];
            }
        }
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_output");
    if (ok) {
        metal_graph_debug_dump_tensor("attn_low", metal_graph_attn_low(g), (uint64_t)n_groups * rank, il, pos);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("attn_out", metal_graph_attn_out(g), DS4_N_EMBD, il, pos);
    }
    if (ok && metal_graph_directional_steering_attn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_attn(g, metal_graph_attn_out(g), il, 1);
    }
    if (ok && !fuse_attn_out_hc && !cuda_tp_attn_hc_fused) {
        if (tp_attn_a) {
            ok = ds4_gpu_hc_expand_add_tensor(metal_graph_after_attn_hc(g), tp_attn_a, tp_attn_b,
                                                metal_graph_cur_hc(g), metal_graph_hc_post(g), metal_graph_hc_comb(g),
                                                DS4_N_EMBD, DS4_N_HC) != 0;
        } else if (cuda_tp_attn_peer) {
            ok = ds4_gpu_hc_expand_add_tensor(
                    metal_graph_after_attn_hc(g),
                    metal_graph_attn_out(g),
                    cuda_tp_attn_peer,
                    metal_graph_cur_hc(g),
                    metal_graph_hc_post(g),
                    metal_graph_hc_comb(g),
                    DS4_N_EMBD,
                    DS4_N_HC) != 0;
        } else {
            ok = ds4_gpu_hc_expand_tensor(metal_graph_after_attn_hc(g), metal_graph_attn_out(g), metal_graph_cur_hc(g),
                                            metal_graph_hc_post(g), metal_graph_hc_comb(g), DS4_N_EMBD, DS4_N_HC) != 0;
        }
    }
    DS4_METAL_PROFILE_DECODE_STAGE("attn_hc_post");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_post", metal_graph_after_attn_hc(g), hc_dim, il, pos);
    }
    bool ffn_hc_producer_pre_norm_fused = false;
    if (ok && !tp_ablate_hcpre) {
        const bool fuse_norm_mix =
            hc_dim == 16384u && mix_hc == 24u &&
            layer->hc_ffn_fn->type == DS4_TENSOR_F16 &&
            !metal_graph_use_reference_hc_decode() &&
            getenv("DS4_METAL_DISABLE_PRE_M5_HC_NORM_MIX_FUSE") == NULL &&
            (ds4_gpu_device_is_pre_m5_apple_silicon() ||
             ds4_gpu_device_is_m5_apple_silicon()) &&
            ds4_gpu_hc_rms_norm_mix_f16_available() != 0;
