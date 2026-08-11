#ifndef DS4_NO_GPU
static bool metal_graph_session_batch_moe_supported(
        ds4_decode_item *items,
        int count,
        const ds4_weights *weights) {
    if (!items || count < 3 || !weights || DS4_N_EXPERT_USED != 6u ||
        getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != NULL) {
        return false;
    }

    ds4_gpu_graph *first = &items[0].session->graph;
    if (!first->placement || !first->cuda_tp_decode || !first->cuda_tp_ep ||
        first->quality || (uint32_t)count > first->prefill_cap) {
        return false;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!layer->ffn_gate_exps || !layer->ffn_down_exps ||
            layer->ffn_gate_exps->type != 12u ||
            layer->ffn_down_exps->type != 12u) {
            return false;
        }
        const int home = first->placement[il + 1u];
        const int partner = metal_graph_cuda_tp_partner_tier(home);
        if (partner < 0 || !g_gpu_peer_ok[partner][home] ||
            !first->batch_ffn_norm_by_tier[home] ||
            !first->batch_router_logits_by_tier[home] ||
            !first->batch_router_selected_by_tier[home] ||
            !first->batch_router_weights_by_tier[home] ||
            !first->batch_routed_gate_by_tier[home] ||
            !first->batch_routed_up_by_tier[home] ||
            !first->batch_routed_mid_by_tier[home] ||
            !first->batch_routed_down_by_tier[home] ||
            !first->batch_routed_out_by_tier[home] ||
            !first->batch_ffn_norm_by_tier[partner] ||
            !first->batch_router_selected_by_tier[partner] ||
            !first->batch_router_weights_by_tier[partner] ||
            !first->batch_routed_gate_by_tier[partner] ||
            !first->batch_routed_up_by_tier[partner] ||
            !first->batch_routed_mid_by_tier[partner] ||
            !first->batch_routed_down_by_tier[partner] ||
            !first->batch_routed_out_by_tier[partner]) {
            return false;
        }
        for (int i = 1; i < count; i++) {
            ds4_gpu_graph *g = &items[i].session->graph;
            if (!g->placement || !g->cuda_tp_decode || !g->cuda_tp_ep ||
                g->quality || (uint32_t)count > g->prefill_cap ||
                g->placement[il + 1u] != home) {
                return false;
            }
        }
    }
    return true;
}

static bool metal_graph_session_batch_shared_supported(
        ds4_decode_item *items,
        int count,
        const ds4_weights *weights) {
    if (!items || count < 3 || !weights ||
        metal_graph_use_reference_shared_down_hc()) {
        return false;
    }

    ds4_gpu_graph *first = &items[0].session->graph;
    if (first->quality || first->cuda_tp_shared ||
        !first->shared_gate_up_swiglu_fuse ||
        (uint32_t)count > first->prefill_cap) {
        return false;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!layer->ffn_gate_shexp || !layer->ffn_up_shexp ||
            !layer->ffn_down_shexp ||
            layer->ffn_gate_shexp->type != DS4_TENSOR_Q8_0 ||
            layer->ffn_up_shexp->type != DS4_TENSOR_Q8_0 ||
            layer->ffn_down_shexp->type != DS4_TENSOR_Q8_0) {
            return false;
        }
        const int home = first->placement[il + 1u];
        if (!first->batch_ffn_norm_by_tier[home] ||
            !first->batch_routed_out_by_tier[home] ||
            !first->batch_shared_gate_by_tier[home] ||
            !first->batch_shared_up_by_tier[home] ||
            !first->batch_shared_mid_by_tier[home] ||
            !first->batch_shared_out_by_tier[home] ||
            !first->batch_after_attn_hc_by_tier[home] ||
            !first->batch_hc_split_by_tier[home] ||
            !first->batch_next_hc_by_tier[home]) {
            return false;
        }
        for (int i = 0; i < count; i++) {
            ds4_gpu_graph *g = &items[i].session->graph;
            const uint32_t pos = (uint32_t)items[i].session->checkpoint.len;
            if (g->quality || g->cuda_tp_shared ||
                !g->shared_gate_up_swiglu_fuse ||
                g->placement[il + 1u] != home ||
                metal_graph_needs_ffn_out(g, il, pos) ||
                metal_graph_directional_steering_ffn_enabled(g)) {
                return false;
            }
        }
    }
    return true;
}

static bool metal_graph_session_batch_attn_pre_supported(
        ds4_decode_item *items,
        int count,
        const ds4_weights *weights) {
    if (!items || count < 3 || !weights ||
        DS4_MODEL_VARIANT != DS4_VARIANT_FLASH ||
        metal_graph_use_reference_hc_decode() ||
        metal_graph_use_reference_hc_norm_decode()) {
        return false;
    }

    ds4_gpu_graph *first = &items[0].session->graph;
    if ((uint32_t)count > first->prefill_cap) return false;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!layer->hc_attn_fn || !layer->hc_attn_scale ||
            !layer->hc_attn_base || !layer->attn_norm ||
            layer->hc_attn_fn->type != DS4_TENSOR_F16) {
            return false;
        }
        const int home = first->placement[il + 1u];
        if (!first->batch_cur_hc_by_tier[home] ||
            !first->batch_flat_hc_by_tier[home] ||
            !first->batch_hc_mix_by_tier[home] ||
            !first->batch_hc_split_by_tier[home] ||
            !first->batch_attn_cur_by_tier[home] ||
            !first->batch_attn_norm_by_tier[home]) {
            return false;
        }
        for (int i = 1; i < count; i++) {
            ds4_gpu_graph *g = &items[i].session->graph;
            if (!g->placement || g->placement[il + 1u] != home) return false;
        }
    }
    return true;
}

static bool metal_graph_session_batch_attn_core_supported(
        ds4_decode_item *items,
        int count,
        const ds4_weights *weights) {
#if defined(__APPLE__)
    (void)items;
    (void)count;
    (void)weights;
    return false;
#else
    if (!items || count < 3 || !weights ||
        count > (int)DS4_GPU_ATTENTION_DECODE_BATCH_MAX ||
        DS4_N_HEAD_DIM != 512u ||
        metal_graph_attn_comp_cache_format() != DS4_GPU_CACHE_F32 ||
        getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != NULL ||
        getenv("DS4_CUDA_NO_EXACT_SCORE_SPLIT_DECODE") != NULL ||
        getenv("DS4_CUDA_SPLITKV_DECODE") != NULL ||
        getenv("DS4_CUDA_DECODE_HEADS8_ONLINE") != NULL ||
        getenv("DS4_CUDA_DECODE_SCORE4") != NULL ||
        getenv("DS4_CUDA_DECODE_SCORE8") != NULL ||
        getenv("DS4_CUDA_NO_DECODE_VALUE512") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_GRAPH") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_LDG") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_VEC4") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_VEC4_PLAIN") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_DIM2") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_FUSE_INV_ROPE") != NULL ||
        getenv("DS4_CUDA_NO_SCORE_TILE") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_MIN_SCORE") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_CHUNK") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_S_FLOOR") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_S_MAX") != NULL ||
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_S") != NULL) {
        return false;
    }

    ds4_gpu_graph *first = &items[0].session->graph;
    if (!first->placement || first->quality ||
        first->cuda_tp_attn_heads || first->cuda_tp_q ||
        first->decode_stage_profile ||
        first->decode_index_stage_profile ||
        (uint32_t)count > first->prefill_cap) {
        return false;
    }
    for (int i = 0; i < count; i++) {
        ds4_session *s = items[i].session;
        ds4_gpu_graph *g = &s->graph;
        if (!g->placement || g->quality ||
            g->cuda_tp_attn_heads || g->cuda_tp_q ||
            g->decode_stage_profile ||
            g->decode_index_stage_profile ||
            metal_graph_directional_steering_attn_enabled(g) ||
            s->checkpoint.len < 1 || (uint32_t)count > g->prefill_cap) {
            return false;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!layer->attn_q_a || !layer->attn_sinks ||
            layer->attn_q_a->dim[1] == 0u) {
            return false;
        }
        const int home = first->placement[il + 1u];
        if (home < 0 || !first->batch_qr_by_tier[home] ||
            !first->batch_qr_norm_by_tier[home] ||
            !first->batch_q_by_tier[home] ||
            !first->batch_kv_raw_by_tier[home] ||
            !first->batch_kv_by_tier[home] ||
            !first->batch_comp_kv_by_tier[home] ||
            !first->batch_comp_sc_by_tier[home] ||
            !first->batch_indexer_q_by_tier[home] ||
            !first->batch_indexer_weights_by_tier[home] ||
            !first->batch_heads_by_tier[home] ||
            !first->batch_attn_low_by_tier[home] ||
            !first->batch_attn_out_by_tier[home] ||
            !first->batch_after_attn_hc_by_tier[home]) {
            return false;
        }
        for (int i = 1; i < count; i++) {
            ds4_gpu_graph *g = &items[i].session->graph;
            if (g->placement[il + 1u] != home ||
                g->cuda_tp_attn != first->cuda_tp_attn) {
                return false;
            }
        }
    }
    return true;
#endif
}

static bool metal_graph_session_batch_attn_post_supported(
        ds4_decode_item *items,
        int count,
        const ds4_weights *weights) {
#if defined(__APPLE__)
    (void)items;
    (void)count;
    (void)weights;
    return false;
#else
    if (!items || count < 3 || !weights ||
        (DS4_N_OUT_GROUP & 1u) != 0u ||
        metal_graph_use_reference_attn_out_hc()) {
        return false;
    }
    const uint32_t tp_groups = DS4_N_OUT_GROUP / 2u;
    const uint64_t tp_width = (uint64_t)tp_groups * DS4_N_LORA_O;
    if (tp_width == 0u || (tp_width % 32u) != 0u) return false;

    ds4_gpu_graph *first = &items[0].session->graph;
    if (!first->placement || first->quality || !first->cuda_tp_attn ||
        first->cuda_tp_attn_heads || !first->cuda_tp_attn_peer_read ||
        first->decode_stage_profile ||
        (uint32_t)count > first->prefill_cap) {
        return false;
    }
    for (int i = 0; i < count; i++) {
        ds4_gpu_graph *g = &items[i].session->graph;
        if (!g->placement || g->quality || !g->cuda_tp_attn ||
            g->cuda_tp_attn_heads || !g->cuda_tp_attn_peer_read ||
            g->decode_stage_profile ||
            metal_graph_directional_steering_attn_enabled(g)) {
            return false;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        const int home = first->placement[il + 1u];
        const int partner = metal_graph_cuda_tp_partner_tier(home);
        if (home < 0 || partner < 0 ||
            !g_gpu_peer_ok[home][partner] ||
            !g_gpu_peer_ok[partner][home] ||
            !layer->attn_output_a || !layer->attn_output_b ||
            layer->attn_output_a->type != DS4_TENSOR_Q8_0 ||
            layer->attn_output_b->type != DS4_TENSOR_Q8_0 ||
            !first->batch_heads_by_tier[home] ||
            !first->batch_attn_low_by_tier[home] ||
            !first->batch_attn_out_by_tier[home] ||
            !first->batch_attn_low_by_tier[partner] ||
            !first->batch_attn_out_by_tier[partner] ||
            !first->batch_shared_out_by_tier[home] ||
            !first->batch_cur_hc_by_tier[home] ||
            !first->batch_hc_split_by_tier[home] ||
            !first->batch_after_attn_hc_by_tier[home]) {
            return false;
        }
        for (int i = 1; i < count; i++) {
            ds4_gpu_graph *g = &items[i].session->graph;
            if (g->placement[il + 1u] != home) return false;
        }
    }
    return true;
#endif
}

static bool metal_graph_session_batch_qkv_supported(
        ds4_decode_item *items,
        int count,
        const ds4_weights *weights) {
    if (!items || count < 3 || !weights ||
        count > (int)DS4_GPU_ATTENTION_DECODE_BATCH_MAX ||
        DS4_N_HEAD_KV != 1u || metal_graph_use_reference_qkv_norm() ||
        getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != NULL) {
        return false;
    }
    ds4_gpu_graph *first = &items[0].session->graph;
    if (!first->placement || first->quality || first->cuda_tp_q ||
        !first->cuda_qkv_pair || !first->cuda_qkv_kv_rope_fuse ||
        !first->cuda_q_norm_rope_fuse || first->decode_stage_profile ||
        (uint32_t)count > first->prefill_cap) {
        return false;
    }
    for (int i = 0; i < count; i++) {
        ds4_gpu_graph *g = &items[i].session->graph;
        if (!g->placement || g->quality || g->cuda_tp_q ||
            !g->cuda_qkv_pair || !g->cuda_qkv_kv_rope_fuse ||
            !g->cuda_q_norm_rope_fuse || g->decode_stage_profile) {
            return false;
        }
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        const int home = first->placement[il + 1u];
        if (home < 0 || !layer->attn_q_a || !layer->attn_q_b ||
            !layer->attn_kv || !layer->attn_q_a_norm ||
            !layer->attn_kv_a_norm ||
            layer->attn_q_a->type != DS4_TENSOR_Q8_0 ||
            layer->attn_q_b->type != DS4_TENSOR_Q8_0 ||
            layer->attn_kv->type != DS4_TENSOR_Q8_0 ||
            !first->batch_attn_norm_by_tier[home] ||
            !first->batch_qr_by_tier[home] ||
            !first->batch_qr_norm_by_tier[home] ||
            !first->batch_q_by_tier[home] ||
            !first->batch_kv_raw_by_tier[home] ||
            !first->batch_kv_by_tier[home]) {
            return false;
        }
        for (int i = 1; i < count; i++) {
            if (items[i].session->graph.placement[il + 1u] != home) {
                return false;
            }
        }
    }
    return true;
}

static bool metal_graph_session_batch_kv_store_supported(
        ds4_decode_item *items,
        int count) {
    if (!items || count < 2 ||
        count > (int)DS4_GPU_ATTENTION_DECODE_BATCH_MAX ||
        metal_graph_use_reference_kv_decode()) {
        return false;
    }
    ds4_gpu_graph *first = &items[0].session->graph;
    if (!first->placement || first->cuda_tp_attn_cache_dup) return false;
    for (int i = 0; i < count; i++) {
        ds4_gpu_graph *g = &items[i].session->graph;
        if (!g->placement || g->raw_cap == 0u ||
            g->cuda_tp_attn_cache_dup) {
            return false;
        }
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const int home = first->placement[il + 1u];
            if (g->placement[il + 1u] != home ||
                !g->layer_raw_cache[il] ||
                ds4_gpu_tensor_device(g->layer_raw_cache[il]) != home) {
                return false;
            }
        }
    }
    return true;
}

/* Batch the cache-independent HC preparation at the start of an attention
 * block. The narrow F16 HC projection stays rowwise for exact arithmetic;
 * both HC transforms around it execute over all session rows together. */
static bool metal_graph_encode_attn_pre_session_batch(
        ds4_decode_item *items,
        int count,
        const ds4_model *model,
        const ds4_layer_weights *layer,
        uint32_t il,
        uint32_t row_base,
        bool scatter_rows) {
    if (!items || count < 2 || !model || !layer || il >= DS4_N_LAYER) {
        return false;
    }

    ds4_gpu_graph *g = &items[0].session->graph;
    const int home_tier = g->placement[il + 1u];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC +
                            (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t hc_bytes = (uint64_t)count * hc_dim * sizeof(float);
    const uint64_t mix_bytes = (uint64_t)count * mix_hc * sizeof(float);
    const uint64_t norm_bytes =
        (uint64_t)count * DS4_N_EMBD * sizeof(float);

    bool ok = home_tier >= 0 &&
              (uint64_t)row_base + (uint32_t)count <= g->prefill_cap;
    for (int i = 0; ok && i < count; i++) {
        ok = metal_graph_set_active_tier_decode(
                &items[i].session->graph, home_tier);
    }

    ds4_gpu_tensor cur_hc, flat_hc, hc_mix, hc_split;
    ds4_gpu_tensor attn_cur, attn_norm;
    if (ok) {
        ok = metal_graph_borrow_tensor_view(&cur_hc,
                    metal_graph_batch_cur_hc(g),
                    (uint64_t)row_base * hc_dim * sizeof(float), hc_bytes) &&
             metal_graph_borrow_tensor_view(&flat_hc,
                    metal_graph_batch_flat_hc(g),
                    (uint64_t)row_base * hc_dim * sizeof(float), hc_bytes) &&
             metal_graph_borrow_tensor_view(&hc_mix,
                    metal_graph_batch_hc_mix(g),
                    (uint64_t)row_base * mix_hc * sizeof(float), mix_bytes) &&
             metal_graph_borrow_tensor_view(&hc_split,
                    metal_graph_batch_hc_split(g),
                    (uint64_t)row_base * mix_hc * sizeof(float), mix_bytes) &&
             metal_graph_borrow_tensor_view(&attn_cur,
                    metal_graph_batch_attn_cur(g),
                    (uint64_t)row_base * DS4_N_EMBD * sizeof(float), norm_bytes) &&
             metal_graph_borrow_tensor_view(&attn_norm,
                    metal_graph_batch_attn_norm(g),
                    (uint64_t)row_base * DS4_N_EMBD * sizeof(float), norm_bytes);
    }

    for (int i = 0; ok && i < count; i++) {
        ds4_gpu_graph *srcg = &items[i].session->graph;
        ds4_gpu_tensor dst_row;
        ok = srcg->active_tier == home_tier &&
             metal_graph_borrow_tensor_view(
                    &dst_row, &cur_hc,
                    (uint64_t)i * hc_dim * sizeof(float),
                    hc_dim * sizeof(float));
        if (ok) {
            ok = ds4_gpu_tensor_copy_xdev_default(
                    &dst_row, metal_graph_cur_hc(srcg), dst_row.bytes) != 0;
        }
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_plain_rows_tensor(
                &flat_hc, &cur_hc, (uint32_t)hc_dim,
                (uint32_t)count, DS4_RMS_EPS) != 0;
    }
    for (int i = 0; ok && i < count; i++) {
        ds4_gpu_tensor out_row, in_row;
        ok = metal_graph_borrow_tensor_view(
                    &out_row, &hc_mix,
                    (uint64_t)i * mix_hc * sizeof(float),
                    mix_hc * sizeof(float)) &&
             metal_graph_borrow_tensor_view(
                    &in_row, &flat_hc,
                    (uint64_t)i * hc_dim * sizeof(float),
                    hc_dim * sizeof(float));
        if (ok) {
            ok = metal_graph_matmul_plain_tensor(
                    &out_row, model, layer->hc_attn_fn,
                    hc_dim, mix_hc, &in_row, 1);
        }
    }
    if (ok) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(
                &attn_cur,
                &attn_norm,
                &hc_split,
                &hc_mix,
                &cur_hc,
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

    for (int i = 0; scatter_rows && ok && i < count; i++) {
        ds4_gpu_graph *dstg = &items[i].session->graph;
        ds4_gpu_tensor norm_row, split_row;
        ok = metal_graph_borrow_tensor_view(
                    &norm_row, &attn_norm,
                    (uint64_t)i * DS4_N_EMBD * sizeof(float),
                    (uint64_t)DS4_N_EMBD * sizeof(float)) &&
             metal_graph_borrow_tensor_view(
                    &split_row, &hc_split,
                    (uint64_t)i * mix_hc * sizeof(float),
                    mix_hc * sizeof(float));
        if (ok) {
            ok = ds4_gpu_tensor_copy_xdev3_default_dst(
                    metal_graph_attn_norm(dstg), &norm_row, norm_row.bytes,
                    metal_graph_hc_split(dstg), &split_row, split_row.bytes,
                    NULL, NULL, 0) != 0;
        }
    }

    return ok;
}

#endif /* !DS4_NO_GPU */
