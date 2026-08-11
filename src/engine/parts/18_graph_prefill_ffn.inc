#ifndef DS4_NO_GPU
/* Encode the batched prefill FFN half: HC pre/norm, shared expert, routed
 * experts, sum, and HC post.  A non-empty decode tail has already been
 * prepared in rows [n_tokens, n_tokens + decode_count); only the routed
 * expert dispatch is shared between the two arithmetic paths. */
static bool metal_graph_encode_layer_ffn_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens,
        ds4_decode_item         *decode_items,
        int                     decode_count) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;
    if (decode_count < 0 ||
        (decode_count > 0 &&
         (!decode_items || (uint64_t)n_tokens + (uint32_t)decode_count > g->prefill_cap))) {
        return false;
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_mid_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t routed_out_dim = layer->ffn_down_exps->dim[1];
    const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    const bool layer_stage_profile = metal_graph_layer_stage_profile_enabled(il);
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_FFN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("ffn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)

    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            metal_graph_batch_hc_mix(g), 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            metal_graph_batch_hc_split(g), 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *ffn_cur_view = ds4_gpu_tensor_view(
            metal_graph_batch_ffn_cur(g), 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *next_hc_view = ds4_gpu_tensor_view(
            metal_graph_batch_next_hc(g), 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && ffn_cur_view && next_hc_view;
    const bool fuse_hc_norm = n_tokens > 1 &&
                              DS4_N_HC == 4 &&
                              !metal_graph_use_reference_hc_decode() &&
                              metal_graph_enable_batch_hc_norm_fusion();
    if (ok) ok = metal_graph_hc_rms_scale_project(hc_mix_view,
                                                    metal_graph_batch_flat_hc(g),
                                                    model,
                                                    layer->hc_ffn_fn,
                                                    metal_graph_batch_after_attn_hc(g),
                                                    hc_dim,
                                                    n_tokens);
    if (metal_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_ffn_scale->abs_offset,
                                                        layer->hc_ffn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(ffn_cur_view,
                                                            metal_graph_batch_after_attn_hc(g),
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else if (fuse_hc_norm) {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(ffn_cur_view,
                                                                 metal_graph_batch_ffn_norm(g),
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 metal_graph_batch_after_attn_hc(g),
                                                                 model->map,
                                                                 model->size,
                                                                 layer->hc_ffn_scale->abs_offset,
                                                                 layer->hc_ffn_base->abs_offset,
                                                                 layer->ffn_norm->abs_offset,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_HC,
                                                                 DS4_N_HC_SINKHORN_ITER,
                                                                 DS4_HC_EPS,
                                                                 DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(ffn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            metal_graph_batch_after_attn_hc(g),
                                                            model->map,
                                                            model->size,
                                                            layer->hc_ffn_scale->abs_offset,
                                                            layer->hc_ffn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_pre", metal_graph_batch_ffn_cur(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_pre");
    if (ok && !fuse_hc_norm) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_ffn_norm(g),
                                                  metal_graph_batch_ffn_cur(g),
                                                  model->map,
                                                  model->size,
                                                  layer->ffn_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  n_tokens,
                                                  DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_norm", metal_graph_batch_ffn_norm(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("norm");
    if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_batch_router_logits(g),
                                                 model,
                                                 layer->ffn_gate_inp,
                                                 DS4_N_EMBD,
                                                 DS4_N_EXPERT,
                                                 metal_graph_batch_ffn_norm(g),
                                                 n_tokens);

    ds4_gpu_tensor *router_tokens = NULL;
    if (ok) {
        router_tokens = ds4_gpu_tensor_view(metal_graph_prefill_tokens(g),
                                              (uint64_t)g->batch_token_offset * sizeof(int32_t),
                                              (uint64_t)n_tokens * sizeof(int32_t));
        ok = router_tokens != NULL;
    }
    if (ok) ok = ds4_gpu_router_select_batch_tensor(metal_graph_batch_router_selected(g),
                                                      metal_graph_batch_router_weights(g),
                                                      metal_graph_batch_router_probs(g),
                                                      model->map,
                                                      model->size,
                                                      layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
                                                      layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                                                      0,
                                                      0,
                                                      layer->ffn_exp_probs_b != NULL,
                                                      layer->ffn_gate_tid2eid != NULL,
                                                      metal_graph_batch_router_logits(g),
                                                      metal_graph_prefill_tokens(g),
                                                      DS4_N_EXPERT,
                                                      DS4_N_EXPERT_USED,
                                                      DS4_EXPERT_WEIGHT_SCALE,
                                                      n_tokens) != 0;
    ds4_gpu_tensor_free(router_tokens);
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_logits", metal_graph_batch_router_logits(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_probs", metal_graph_batch_router_probs(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT, il, pos0);
        metal_graph_debug_dump_i32_tensor("ffn_moe_topk", metal_graph_batch_router_selected(g),
                                          (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", metal_graph_batch_router_weights(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("router");

    if (ok) {
        ok = metal_graph_cuda_stream_prefill_batch_selected_load(g,
                                                                 model,
                                                                 layer,
                                                                 il,
                                                                 n_tokens,
                                                                 gate_expert_bytes,
                                                                 down_expert_bytes);
    }

#ifdef DS4_ROCM_BUILD
    rocm_graph_batch_selected_async_load rocm_batch_selected_async = {0};
    bool rocm_batch_selected_async_started = false;
    const bool rocm_batch_selected_shared_overlap =
        ok &&
        g->ssd_streaming &&
        !g->quality &&
        n_tokens > 1 &&
        DS4_N_EXPERT_USED == 6 &&
        !rocm_graph_stream_prefill_full_layer_enabled(g, layer, il, n_tokens) &&
        layer->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q2_K;
    if (rocm_batch_selected_shared_overlap) {
        uint64_t selected_event = 0;
        if (ds4_gpu_signal_selected_readback_ready(&selected_event) == 0) {
            ok = false;
        } else {
            ok = rocm_graph_batch_selected_async_load_start(
                    &rocm_batch_selected_async,
                    metal_graph_batch_router_selected(g),
                    model,
                    layer,
                    il,
                    n_tokens,
                    selected_event,
                    gate_expert_bytes,
                    down_expert_bytes);
            rocm_batch_selected_async_started = ok;
        }
    }
#endif

    const bool selected_readahead_shared =
        metal_graph_stream_prefill_selected_readahead_shared_enabled(g)
#ifdef DS4_ROCM_BUILD
        && !rocm_batch_selected_async_started
#endif
        ;
    if (ok &&
        metal_graph_stream_prefill_selected_readahead_enabled(g) &&
#ifdef DS4_ROCM_BUILD
        !rocm_batch_selected_async_started &&
#endif
        !selected_readahead_shared) {
        if (ds4_gpu_end_commands() == 0) {
            ok = false;
        } else {
            ok = metal_graph_stream_readahead_selected_experts_from_gpu(g,
                                                                        model,
                                                                        layer,
                                                                        il,
                                                                        n_tokens,
                                                                        gate_expert_bytes,
                                                                        down_expert_bytes) &&
                 ds4_gpu_begin_commands() != 0;
        }
    }

    const bool keep_ffn_out = metal_graph_needs_ffn_out(g, il, pos0);
    bool shared_down_f16 = false;

#define DS4_METAL_TRY_SHARED_DOWN_F16() do { \
        if (ok && !tp_row_split_ffn && !keep_ffn_out && \
            !metal_graph_debug_wants("ffn_shexp", il, pos0)) { \
            shared_down_f16 = ds4_gpu_matmul_q8_0_f16_out_tensor(g->batch_q_half, \
                                                                 model->map, \
                                                                 model->size, \
                                                                 layer->ffn_down_shexp->abs_offset, \
                                                                 shared_dim, \
                                                                 DS4_N_EMBD, \
                                                                 metal_graph_batch_shared_mid(g), \
                                                                 n_tokens) != 0; \
        } \
    } while (0)

#define DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT() do { \
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("shared_gate", \
                                                          il, \
                                                          pos0, \
                                                          metal_graph_batch_shared_gate(g), \
                                                          model, \
                                                          layer->ffn_gate_shexp, \
                                                          DS4_N_EMBD, \
                                                          shared_dim, \
                                                          tp_ffn_x ? tp_ffn_x : metal_graph_batch_ffn_norm(g), \
                                                          tp_rows); \
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("shared_up", \
                                                          il, \
                                                          pos0, \
                                                          metal_graph_batch_shared_up(g), \
                                                          model, \
                                                          layer->ffn_up_shexp, \
                                                          DS4_N_EMBD, \
                                                          shared_dim, \
                                                          tp_ffn_x ? tp_ffn_x : metal_graph_batch_ffn_norm(g), \
                                                          tp_rows); \
        DS4_METAL_PROFILE_FFN_STAGE("shared_gate_up"); \
        if (ok) ok = ds4_gpu_swiglu_tensor(metal_graph_batch_shared_mid(g), \
                                             metal_graph_batch_shared_gate(g), \
                                             metal_graph_batch_shared_up(g), \
                                             (uint32_t)((uint64_t)tp_rows * shared_dim), \
                                             DS4_SWIGLU_CLAMP_EXP, \
                                             1.0f) != 0; \
        DS4_METAL_TRY_SHARED_DOWN_F16(); \
        if (ok && !shared_down_f16) ok = metal_graph_matmul_q8_0_named_tensor("shared_down", \
                                                                              il, \
                                                                              pos0, \
                                                                              metal_graph_batch_shared_out(g), \
                                                                              model, \
                                                                              layer->ffn_down_shexp, \
                                                                              shared_dim, \
                                                                              DS4_N_EMBD, \
                                                                              metal_graph_batch_shared_mid(g), \
                                                                              tp_rows); \
        DS4_METAL_PROFILE_FFN_STAGE("shared_down"); \
        if (ok && !shared_down_f16) { \
            metal_graph_debug_dump_tensor("ffn_shexp", metal_graph_batch_shared_out(g), \
                                          (uint64_t)n_tokens * DS4_N_EMBD, il, pos0); \
        } \
    } while (0)

    bool shared_done = false;
    /* With 50/50 expert residency, every rank evaluates every prompt row
     * against its local expert half.  For large chunks the replicated shared
     * expert remains row-split; its rows are folded into the local routed
     * partial before the one all-reduce-style bulk exchange. */
    const bool tp_split_ffn = g->tp_world == 2;
    const bool tp_row_split_ffn =
        tp_split_ffn && g->tp_batch_rows != n_tokens && !keep_ffn_out &&
        !metal_graph_directional_steering_ffn_enabled(g) &&
        n_tokens >= metal_graph_tp_prefill_split_min();
    const uint32_t tp_half_rows = (n_tokens + 1u) / 2u;
    const uint32_t tp_row0 = (tp_row_split_ffn && g->tp_rank != 0) ? tp_half_rows : 0;
    const uint32_t tp_rows = tp_row_split_ffn ?
        (g->tp_rank == 0 ? tp_half_rows : n_tokens - tp_half_rows) : n_tokens;
    ds4_gpu_tensor *tp_ffn_x = tp_row_split_ffn ?
        metal_graph_tensor_row_range_view(metal_graph_batch_ffn_norm(g), tp_row0, tp_rows,
                                          DS4_N_EMBD) : NULL;
    if (tp_row_split_ffn && !tp_ffn_x) ok = false;
    if (ok && selected_readahead_shared) {
        if (ds4_gpu_end_commands() == 0) {
            ok = false;
        } else {
            ok = metal_graph_stream_readahead_selected_experts_from_gpu(g,
                                                                        model,
                                                                        layer,
                                                                        il,
                                                                        n_tokens,
                                                                        gate_expert_bytes,
                                                                        down_expert_bytes) &&
                 ds4_gpu_begin_commands() != 0;
        }
        if (ok) {
            DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT();
            shared_done = ok;
        }
        if (ok) {
            if (ds4_gpu_end_commands() == 0) {
                ok = false;
            } else {
                ok = ds4_gpu_begin_commands() != 0;
            }
        }
    }

    if (ok &&
        !shared_done &&
        (metal_graph_stream_prefill_selected_pagein_enabled(g) ||
         metal_graph_stream_prefill_selected_madvise_enabled(g))) {
        metal_graph_stream_pagein_job pagein_job;
        memset(&pagein_job, 0, sizeof(pagein_job));
        bool pagein_commands_open = false;
        if (ds4_gpu_end_commands() == 0) {
            ok = false;
        } else {
            ok = metal_graph_stream_prefill_selected_pagein_start(g,
                                                                  model,
                                                                  layer,
                                                                  il,
                                                                  n_tokens,
                                                                  gate_expert_bytes,
                                                                  down_expert_bytes,
                                                                  &pagein_job);
        }
        if (ok) {
            if (ds4_gpu_begin_commands() == 0) {
                ok = false;
            } else {
                pagein_commands_open = true;
            }
        }
        if (ok) {
            DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT();
            shared_done = ok;
        }
        if (pagein_commands_open) {
            if (ds4_gpu_end_commands() == 0) ok = false;
        }
        if (!metal_graph_stream_prefill_selected_pagein_join(&pagein_job)) {
            ok = false;
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
    }

#ifdef DS4_ROCM_BUILD
    if (rocm_batch_selected_async_started) {
        if (ok && !shared_done) {
            DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT();
            shared_done = ok;
        }
        const bool finish_ok =
            rocm_graph_batch_selected_async_load_finish(&rocm_batch_selected_async);
        ok = ok && finish_ok;
    }
#endif

    const bool tp_split_batch_moe =
        g->tp_batch_rows == n_tokens && n_tokens > 0 &&
        g->tp_world == 2 &&
        g->tp_batch_out && g->tp_batch_in;
    const bool cuda_tp_owned_batch_moe =
        g->cuda_tp_ep && g->cuda_tp_prefill_ffn;
    if (ok && cuda_tp_owned_batch_moe) {
        ok = metal_graph_encode_mixed_routed_rows(
                g, decode_items, decode_count, model, layer, il, n_tokens);
    } else if (ok && tp_split_batch_moe) {
        /* Verify-block expert split: run the contiguous-half split
         * single-token routed kernels per row into the slab batch-out
         * rows, exchange all rows with one gate, then materialize the
         * combined routed output.  The add is commutative, so both ranks
         * compute bit-identical sums and stay in lockstep. */
        const uint64_t vec_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
        for (uint32_t r = 0; ok && r < n_tokens; r++) {
            ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
                    g->tp_batch_out[il], (uint64_t)r * vec_bytes, vec_bytes);
            ds4_gpu_tensor *x_row = ds4_gpu_tensor_view(
                    metal_graph_batch_ffn_norm(g), (uint64_t)r * vec_bytes, vec_bytes);
            ds4_gpu_tensor *sel_row = ds4_gpu_tensor_view(
                    metal_graph_batch_router_selected(g),
                    (uint64_t)r * DS4_N_EXPERT_USED * sizeof(int32_t),
                    (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
            ds4_gpu_tensor *w_row = ds4_gpu_tensor_view(
                    metal_graph_batch_router_weights(g),
                    (uint64_t)r * DS4_N_EXPERT_USED * sizeof(float),
                    (uint64_t)DS4_N_EXPERT_USED * sizeof(float));
            ok = out_row && x_row && sel_row && w_row &&
                 ds4_gpu_routed_moe_one_tensor(out_row,
                                               metal_graph_routed_gate(g),
                                               metal_graph_routed_up(g),
                                               metal_graph_routed_mid(g),
                                               metal_graph_routed_down(g),
                                               model->map, model->size,
                                               layer->ffn_gate_exps->abs_offset,
                                               layer->ffn_up_exps->abs_offset,
                                               layer->ffn_down_exps->abs_offset,
                                               layer->ffn_gate_exps->type,
                                               layer->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               (uint32_t)expert_in_dim,
                                               (uint32_t)down_in_dim,
                                               (uint32_t)routed_out_dim,
                                               sel_row, w_row,
                                               DS4_N_EXPERT,
                                               DS4_N_EXPERT_USED,
                                               DS4_SWIGLU_CLAMP_EXP,
                                               x_row,
                                               NULL,
                                               il,
                                               false) != 0;
            ds4_gpu_tensor_free(w_row);
            ds4_gpu_tensor_free(sel_row);
            ds4_gpu_tensor_free(x_row);
            ds4_gpu_tensor_free(out_row);
        }
        if (ok) ok = ds4_gpu_tp_batch_gate_encode(il, n_tokens) != 0;
        if (ok) {
            ok = ds4_gpu_add_tensor(metal_graph_batch_routed_out(g),
                                    g->tp_batch_out[il],
                                    g->tp_batch_in[il],
                                    (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
        }
    } else if (ok) {
        ok = ds4_gpu_routed_moe_batch_tensor(metal_graph_batch_routed_out(g),
                                               metal_graph_batch_routed_gate(g),
                                               metal_graph_batch_routed_up(g),
                                               metal_graph_batch_routed_mid(g),
                                               metal_graph_batch_routed_down(g),
                                               model->map,
                                               model->size,
                                               layer->ffn_gate_exps->abs_offset,
                                               layer->ffn_up_exps->abs_offset,
                                               layer->ffn_down_exps->abs_offset,
                                               layer->ffn_gate_exps->type,
                                               layer->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               (uint32_t)expert_in_dim,
                                               (uint32_t)down_in_dim,
                                               (uint32_t)routed_out_dim,
                                               metal_graph_batch_router_selected(g),
                                               metal_graph_batch_router_weights(g),
                                               DS4_N_EXPERT,
                                               DS4_N_EXPERT_USED,
                                               DS4_SWIGLU_CLAMP_EXP,
                                               metal_graph_batch_ffn_norm(g),
                                               il,
                                               n_tokens,
                                               &g->batch_routed_mid_is_f16,
                                               false) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_gate_clamped", metal_graph_batch_routed_gate(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
        metal_graph_debug_dump_tensor("ffn_moe_up_clamped", metal_graph_batch_routed_up(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim, il, pos0);
    }
    if (ok) {
        const uint64_t routed_mid_elems = (uint64_t)n_tokens * DS4_N_EXPERT_USED * down_in_dim;
        if (g->batch_routed_mid_is_f16) {
            metal_graph_debug_dump_f16_tensor("ffn_moe_weighted_swiglu", metal_graph_batch_routed_mid(g),
                                              routed_mid_elems, il, pos0);
        } else {
            metal_graph_debug_dump_tensor("ffn_moe_weighted_swiglu", metal_graph_batch_routed_mid(g),
                                          routed_mid_elems, il, pos0);
        }
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_down", metal_graph_batch_routed_down(g),
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED * DS4_N_EMBD, il, pos0);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("ffn_moe_out", metal_graph_batch_routed_out(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("routed_moe");
    if (!shared_done) {
        DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT();
    }
#undef DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT
#undef DS4_METAL_TRY_SHARED_DOWN_F16

    if (ok && tp_row_split_ffn) {
        /* Each shared-expert row must appear exactly once in the all-reduce.
         * Fold this rank's shared rows into its full-row routed partial; the
         * peer does the same for the complementary rows. */
        ds4_gpu_tensor *own_rows =
            metal_graph_tensor_row_range_view(metal_graph_batch_routed_out(g), tp_row0,
                                              tp_rows, DS4_N_EMBD);
        ok = own_rows &&
             ds4_gpu_add_tensor(own_rows, own_rows, metal_graph_batch_shared_out(g),
                                (uint32_t)((uint64_t)tp_rows * DS4_N_EMBD)) != 0;
        ds4_gpu_tensor_free(own_rows);
    }

    if (ok && tp_split_ffn && !tp_split_batch_moe) {
        /* All rows contain this rank's routed-expert partial.  Exchange that
         * matrix in one bulk gate, then add in canonical rank order.  The
         * batch verify path above already performed the equivalent slab gate. */
        const uint64_t bytes =
            (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float);
        ok = metal_graph_ensure_batch_ffn_out(g) &&
             ds4_gpu_tp_big_gate_encode(il, n_tokens,
                                        metal_graph_batch_routed_out(g),
                                        metal_graph_batch_ffn_out(g),
                                        bytes) != 0;
        if (ok) {
            ds4_gpu_tensor *first = g->tp_rank == 0 ?
                metal_graph_batch_routed_out(g) : metal_graph_batch_ffn_out(g);
            ds4_gpu_tensor *second = g->tp_rank == 0 ?
                metal_graph_batch_ffn_out(g) : metal_graph_batch_routed_out(g);
            ok = ds4_gpu_add_tensor(metal_graph_batch_routed_out(g), first, second,
                                    (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
        }
        if (!ok) {
            fprintf(stderr, "ds4: TP prefill FFN all-reduce failed (layer %u)\n", il);
        }
    }

    if (ok && keep_ffn_out) {
        ok = metal_graph_ensure_batch_ffn_out(g) &&
             ds4_gpu_add_tensor(metal_graph_batch_ffn_out(g),
                                  metal_graph_batch_shared_out(g),
                                  metal_graph_batch_routed_out(g),
                                  (uint32_t)((uint64_t)n_tokens * DS4_N_EMBD)) != 0;
    }
    if (ok && keep_ffn_out) {
        metal_graph_debug_dump_tensor("ffn_out", metal_graph_batch_ffn_out(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_ffn(g, metal_graph_batch_ffn_out(g), il, n_tokens);
    }
    if (ok && metal_graph_directional_steering_ffn_enabled(g)) {
        ok = ds4_gpu_hc_expand_split_tensor(next_hc_view,
                                              metal_graph_batch_ffn_out(g),
                                              metal_graph_batch_after_attn_hc(g),
                                              hc_split_view,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    }
    else if (ok && shared_down_f16) {
        ok = ds4_gpu_hc_expand_add_split_half_add_tensor(next_hc_view,
                                                         metal_graph_batch_routed_out(g),
                                                         g->batch_q_half,
                                                         metal_graph_batch_after_attn_hc(g),
                                                         hc_split_view,
                                                         DS4_N_EMBD,
                                                         DS4_N_HC) != 0;
    }
    else if (ok && tp_row_split_ffn) {
        /* Shared expert already folded into the exchanged routed rows. */
        ok = ds4_gpu_hc_expand_split_tensor(next_hc_view,
                                              metal_graph_batch_routed_out(g),
                                              metal_graph_batch_after_attn_hc(g),
                                              hc_split_view,
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    }
    else if (ok) {
        ok = ds4_gpu_hc_expand_add_split_tensor(next_hc_view,
                                                  metal_graph_batch_routed_out(g),
                                                  metal_graph_batch_shared_out(g),
                                                  metal_graph_batch_after_attn_hc(g),
                                                  hc_split_view,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_post");
    if (ok) {
        metal_graph_debug_dump_tensor("hc_ffn_post", metal_graph_batch_next_hc(g),
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_METAL_PROFILE_FFN_STAGE("hc_post");
    ds4_gpu_tensor_free(tp_ffn_x);
    ds4_gpu_tensor_free(next_hc_view);
    ds4_gpu_tensor_free(ffn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
#undef DS4_METAL_PROFILE_FFN_STAGE
    return ok;
}

/* Encode one complete layer for prefill by chaining attention and FFN batches. */
static bool metal_graph_encode_layer_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (g->placement) {
        const int this_tier = g->placement[il + 1u];
        if (!metal_graph_set_active_tier_batch(g, this_tier, n_tokens)) {
            return false;
        }
    }
    bool ok = metal_graph_layer_stage_profile_start(il);
    if (ok) {
        ok = metal_graph_encode_layer_attention_batch(g, model, layer, il, pos0, n_tokens);
    }
    if (!ok) {
        fprintf(stderr, "ds4: gpu layer %u attention batch encode failed\n", il);
    }
    if (ok) {
        ok = metal_graph_encode_layer_ffn_batch(g, model, layer, il, pos0,
                                                 n_tokens, NULL, 0);
        if (!ok) {
            fprintf(stderr, "ds4: gpu layer %u ffn batch encode failed\n", il);
        }
    }
    if (ok) {
        ds4_gpu_tensor *tmp = metal_graph_batch_cur_hc(g);
        g->batch_cur_hc_by_tier[g->active_tier] = metal_graph_batch_next_hc(g);
        g->batch_next_hc_by_tier[g->active_tier] = tmp;
    }
    return ok;
}

static bool metal_graph_eval_token_raw_swa_streaming(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    if (g->raw_cap == 0) {
        fprintf(stderr, "ds4: Metal graph raw KV cache is not allocated\n");
        return false;
    }

    const bool profile =
        glm_graph_env_present("DS4_ROCM_GRAPH_TOKEN_PROFILE",
                              "DS4_METAL_GRAPH_TOKEN_PROFILE");
    const bool throttle = graph_power_throttle_enabled(g);
    const double t0 = (profile || throttle) ? now_sec() : 0.0;
    const uint32_t raw_row = pos % g->raw_cap;
    const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
    metal_graph_dspark_capture_begin(g);

    const bool static_decode_map = metal_graph_stream_decode_static_map_enabled();
    const bool static_map_state_cache =
        static_decode_map && metal_graph_stream_decode_static_map_state_cache_enabled();
    const bool batch_static_decode =
        static_decode_map && metal_graph_stream_decode_layer_batch_enabled(g);
    bool ok = true;
    if (static_decode_map) {
        if (!static_map_state_cache || !g->streaming_static_decode_map_current) {
            ok = metal_graph_stream_map_decode_static_all(model, weights);
            if (ok) g->streaming_static_decode_map_current = static_map_state_cache;
        }
    } else {
        g->streaming_static_decode_map_current = false;
        ok = metal_graph_stream_map_token(model, weights);
    }
    if (ok && !static_decode_map && DS4_N_LAYER > 0) {
        metal_graph_stream_readahead_layer_decode(model, weights, 0);
    }
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = ds4_gpu_embed_token_hc_tensor(metal_graph_cur_hc(g),
                                           model->map,
                                           model->size,
                                           weights->token_embd->abs_offset,
                                           (uint32_t)weights->token_embd->dim[1],
                                           (uint32_t)token,
                                           DS4_N_EMBD,
                                           DS4_N_HC) != 0;
    }
    if (batch_static_decode) {
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = metal_graph_encode_decode_layer(g,
                                                 model,
                                                 &weights->layer[il],
                                                 il,
                                                 pos,
                                                 g->layer_raw_cache[il],
                                                 g->raw_cap,
                                                 raw_row,
                                                 n_raw,
                                                 token);
            if (ok) {
                ds4_gpu_tensor *tmp = metal_graph_cur_hc(g);
                g->cur_hc_by_tier[g->active_tier] = metal_graph_after_ffn_hc(g);
                g->after_ffn_hc_by_tier[g->active_tier] = tmp;
                ok = metal_graph_dspark_capture_decode_layer(g, il);
            }
        }
        if (ok && logits) {
            ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
        }
        const double t_encoded = (profile || throttle) ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double t_done = (profile || throttle) ? now_sec() : 0.0;
        if (ok && logits) {
            ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
        }
        const double t_read = (profile || throttle) ? now_sec() : 0.0;
        if (profile) {
            fprintf(stderr,
                    "ds4: metal SSD streaming batched token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                    pos,
                    (t_encoded - t0) * 1000.0,
                    (t_done - t_encoded) * 1000.0,
                    (t_read - t_done) * 1000.0,
                    (t_read - t0) * 1000.0,
                    logits != NULL);
        }
        if (ok && throttle) {
            graph_power_note_decode_token(g, t_read - t0);
        }
        if (!ok) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after batched SSD streaming graph eval failure also failed\n");
            }
        }
        return ok;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;

    double encode_s = 0.0;
    double execute_s = 0.0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const double tl0 = profile ? now_sec() : 0.0;
        if (!static_decode_map && !metal_graph_stream_map_layer_decode(model, weights, il)) {
            ok = false;
            break;
        }
        if (!static_decode_map && il + 1 < DS4_N_LAYER) {
            metal_graph_stream_readahead_layer_decode(model, weights, il + 1);
        } else if (!static_decode_map && logits) {
            metal_graph_stream_readahead_output(model, weights);
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        bool encoded_layer = false;
        if (ok) {
            ok = metal_graph_encode_decode_layer(g,
                                                 model,
                                                 &weights->layer[il],
                                                 il,
                                                 pos,
                                                 g->layer_raw_cache[il],
                                                 g->raw_cap,
                                                 raw_row,
                                                 n_raw,
                                                 token);
            encoded_layer = true;
        }
        if (encoded_layer) {
            ds4_gpu_tensor *tmp = metal_graph_cur_hc(g);
            g->cur_hc_by_tier[g->active_tier] = metal_graph_after_ffn_hc(g);
            g->after_ffn_hc_by_tier[g->active_tier] = tmp;
            if (ok) ok = metal_graph_dspark_capture_decode_layer(g, il);
        }
        const double tl_encoded = profile ? now_sec() : 0.0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        const double tl_done = profile ? now_sec() : 0.0;
        if (profile) {
            encode_s += tl_encoded - tl0;
            execute_s += tl_done - tl_encoded;
        }
    }

    if (ok && logits && !static_decode_map) ok = metal_graph_stream_map_output(model, weights);
    const double t_head0 = profile ? now_sec() : 0.0;
    if (ok && logits) ok = ds4_gpu_begin_commands() != 0;
    if (ok && logits) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
    const double t_head_encoded = profile ? now_sec() : 0.0;
    if (ok && logits) ok = ds4_gpu_end_commands() != 0;
    const double t_done = (profile || throttle) ? now_sec() : 0.0;
    if (ok && logits) {
        ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    const double t_read = (profile || throttle) ? now_sec() : 0.0;

    if (profile) {
        if (logits) {
            encode_s += t_head_encoded - t_head0;
            execute_s += t_done - t_head_encoded;
        }
        fprintf(stderr,
                "ds4: metal SSD streaming token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                pos,
                encode_s * 1000.0,
                execute_s * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
    if (ok) graph_power_note_decode_token(g, t_read - t0);
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after SSD streaming graph eval failure also failed\n");
        }
    }
    return ok;
}

/* Execute one Metal decode token and read back logits. */
static bool metal_graph_eval_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits) {
    if (g && g->ssd_streaming) {
        return metal_graph_eval_token_raw_swa_streaming(g, model, weights, token, pos, logits);
    }

    const bool profile =
        glm_graph_env_present("DS4_ROCM_GRAPH_TOKEN_PROFILE",
                              "DS4_METAL_GRAPH_TOKEN_PROFILE");
    const bool throttle = graph_power_throttle_enabled(g);
    const double t0 = (profile || throttle) ? now_sec() : 0.0;

    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_encode_token_raw_swa(g, model, weights, token, pos, logits != NULL, true);
    const double t_encoded = (profile || throttle) ? now_sec() : 0.0;
    if (ok) ok = ds4_gpu_end_commands() != 0;
    const double t_done = (profile || throttle) ? now_sec() : 0.0;

    if (ok && logits && g->tp_world == 2 && g->tp_logits_half) {
        const uint64_t tp_vhalf = (uint64_t)DS4_N_VOCAB / 2u;
        const uint64_t off = (uint64_t)g->tp_rank * tp_vhalf * sizeof(float);
        ok = ds4_gpu_tensor_read(metal_graph_logits(g), off, logits + g->tp_rank * tp_vhalf,
                                 tp_vhalf * sizeof(float)) != 0;
    } else if (ok && logits && !(g->tp_world == 2 && g->tp_rank == 1)) {
        ok = ds4_gpu_tensor_read(metal_graph_logits(g), 0, logits, (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    const double t_read = (profile || throttle) ? now_sec() : 0.0;
    if (profile) {
        fprintf(stderr,
                "ds4: metal graph token pos=%u encode=%.3f ms execute=%.3f ms read=%.3f ms total=%.3f ms logits=%d\n",
                pos,
                (t_encoded - t0) * 1000.0,
                (t_done - t_encoded) * 1000.0,
                (t_read - t_done) * 1000.0,
                (t_read - t0) * 1000.0,
                logits != NULL);
    }
    if (ok) graph_power_note_decode_token(g, t_read - t0);
    if (!ok) {
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after graph eval failure also failed\n");
        }
    }
    return ok;
}

static bool metal_graph_streaming_decode_prefill_wide_default(
        const ds4_weights *weights) {
    if (DS4_MODEL_VARIANT != DS4_VARIANT_FLASH || !weights || DS4_N_LAYER == 0) {
        return false;
    }
    const uint32_t type = weights->layer[0].ffn_gate_exps->type;
    return (type == DS4_TENSOR_Q4_K || type == DS4_TENSOR_MXFP4) &&
           weights->layer[0].ffn_up_exps->type == type &&
           weights->layer[0].ffn_down_exps->type == type;
}

static uint32_t metal_graph_streaming_decode_prefill_max_tokens(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights) {
    (void)g;
    if (glm_graph_env_present("DS4_ROCM_DISABLE_STREAMING_DECODE_PREFILL",
                              "DS4_METAL_DISABLE_STREAMING_DECODE_PREFILL")) {
        return 0;
    }

    const char *env = glm_graph_env_value(
            "DS4_ROCM_STREAMING_DECODE_PREFILL_MAX",
            "DS4_METAL_STREAMING_DECODE_PREFILL_MAX");
    if (env && env[0]) {
        char *end = NULL;
        const long v = strtol(env, &end, 10);
        if (end != env) {
            if (v <= 0) return 0;
            if ((unsigned long)v > (unsigned long)UINT32_MAX) return UINT32_MAX;
            return (uint32_t)v;
        }
    }

    if (DS4_MODEL_VARIANT != DS4_VARIANT_PRO &&
        DS4_MODEL_VARIANT != DS4_VARIANT_FLASH) {
        return 0u;
    }
    return metal_graph_streaming_decode_prefill_wide_default(weights) ? 64u : 18u;
}

static bool metal_graph_use_streaming_decode_prefill(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             n_tokens) {
    const uint32_t max_tokens =
        metal_graph_streaming_decode_prefill_max_tokens(g, weights);
    return g &&
           g->ssd_streaming &&
           !g->quality &&
           n_tokens != 0 &&
           max_tokens != 0 &&
           n_tokens <= max_tokens;
}

static bool metal_graph_use_streaming_decode_prefill_range(
        const ds4_gpu_graph *g,
        const ds4_weights   *weights,
        uint32_t             start,
        uint32_t             n_tokens) {
    /*
     * Short streamed prefill is latency-sensitive.  Use the decode-style path
     * by default for SSD streaming, while keeping a cold-only escape hatch for
     * strict-vector tests that need canonical layer-major prefill semantics.
     */
    if (start == 0) {
        if (glm_graph_env_present(
                    "DS4_ROCM_DISABLE_STREAMING_COLD_DECODE_PREFILL",
                    "DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL")) {
            return false;
        }
    }
    return metal_graph_use_streaming_decode_prefill(g, weights, n_tokens);
}

static bool metal_graph_prefill_decode_streaming_range(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (!metal_graph_use_streaming_decode_prefill(g, weights, n_tokens)) return false;
    if (!prompt || start > (uint32_t)prompt->len ||
        n_tokens > (uint32_t)prompt->len - start) return false;
    if (start == 0) {
        ds4_gpu_stream_expert_cache_reset_route_hotness();
    }

    const bool profile =
        glm_graph_env_present("DS4_ROCM_GRAPH_PREFILL_PROFILE",
                              "DS4_METAL_GRAPH_PREFILL_PROFILE");
    const double t0 = profile ? now_sec() : 0.0;

    /*
     * `prefill_chunk` is not just UI progress: ds4_session_sync() wraps it to
     * advance the live checkpoint, and ds4-server may save that checkpoint.
     * Decode-style prefill only reads logits for the final token, so report one
     * cacheable chunk at the end. `prefill_display` remains per-token UI only.
     */
    if (progress) progress(progress_ud, "prefill_chunk", (int)start, prompt->len);
    if (display_progress) {
        display_progress(display_progress_ud, "prefill_display", (int)start, prompt->len);
    }

    for (uint32_t i = 0; i < n_tokens; i++) {
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        const uint32_t pos = start + i;
        const bool last = i + 1u == n_tokens;
        float *token_logits = (last && logits) ? logits : NULL;
        if (!metal_graph_eval_token_raw_swa(g,
                                            model,
                                            weights,
                                            prompt->v[pos],
                                            pos,
                                            token_logits)) {
            if (ds4_gpu_synchronize() == 0) {
                fprintf(stderr, "ds4: Metal synchronize after decode-style streaming prefill failure also failed\n");
            }
            return false;
        }

        if (last && progress && logits) {
            progress(progress_ud, "prefill_chunk", (int)(pos + 1u), prompt->len);
        }
        if (display_progress) {
            display_progress(display_progress_ud, "prefill_display", (int)(pos + 1u), prompt->len);
        }
        if (cancel && cancel(cancel_ud)) {
            if (cancelled) *cancelled = true;
            return true;
        }
        if (show_progress) {
            fprintf(stderr, "ds4: gpu streaming prefill token %u/%u\r",
                    i + 1u,
                    n_tokens);
            fflush(stderr);
        }
    }
    if (show_progress) fputc('\n', stderr);

    if (profile) {
        const double t1 = now_sec();
        fprintf(stderr,
                "ds4: gpu decode-style streaming prefill start=%u tokens=%u total=%.3f ms\n",
                start,
                n_tokens,
                (t1 - t0) * 1000.0);
    }
    return true;
}

static bool metal_graph_capture_prefill_seed_router_selected(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       n_tokens) {
    uint32_t k = metal_graph_streaming_prefill_cache_seed_k(g);
    if (k == 0) return true;
    if (k > n_tokens) k = n_tokens;
    g->prefill_seed_tokens = k;
    if (!g->prefill_seed_router_selected || !metal_graph_batch_router_selected(g) ||
        il >= DS4_N_LAYER || n_tokens == 0 || sizeof(int) != sizeof(int32_t)) {
        return false;
    }

    const uint64_t bytes = (uint64_t)k * DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t src_off = (uint64_t)(n_tokens - k) *
                             DS4_N_EXPERT_USED * sizeof(int);
    const uint64_t dst_off = (uint64_t)il *
                             DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                             DS4_N_EXPERT_USED * sizeof(int32_t);
    return ds4_gpu_tensor_copy(g->prefill_seed_router_selected,
                               dst_off,
                               metal_graph_batch_router_selected(g),
                               src_off,
                               bytes) != 0;
}

static bool metal_graph_seed_streaming_expert_cache_from_prefill(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    const uint32_t seed_tokens = g ? g->prefill_seed_tokens : 0;
    if (!metal_graph_streaming_prefill_cache_seed_enabled(g)) return true;
    if (!model || !weights || !g->prefill_seed_router_selected || seed_tokens == 0) {
        return false;
    }

    int32_t selected[DS4_MAX_LAYER *
                     DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                     DS4_MAX_EXPERT_USED];
    const uint64_t bytes = (uint64_t)DS4_N_LAYER *
                           DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                           DS4_N_EXPERT_USED * sizeof(selected[0]);
    if (ds4_gpu_tensor_read(g->prefill_seed_router_selected,
                            0,
                            selected,
                            bytes) == 0) {
        return false;
    }

    const bool profile =
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_CACHE_SEED_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_CACHE_SEED_PROFILE");
    const double t0 = profile ? now_sec() : 0.0;
    uint32_t seeded_layers = 0;
    uint32_t seeded_rows = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!metal_graph_streaming_expert_cache_seed_layer_expected(g, layer)) {
            continue;
        }

        const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
        const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
        if (layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
            layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
            fprintf(stderr, "ds4: Metal prefill expert-cache seed byte size overflow at layer %u\n", il);
            return false;
        }
        const uint64_t gate_expert_bytes = layer->ffn_gate_exps->dim[1] * gate_row_bytes;
        const uint64_t down_expert_bytes = layer->ffn_down_exps->dim[1] * down_row_bytes;
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        for (uint32_t row = 0; row < seed_tokens; row++) {
            const size_t sel_off = ((size_t)il *
                                    DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS +
                                    row) * DS4_N_EXPERT_USED;
            if (ds4_gpu_stream_expert_cache_seed_selected(
                        &table,
                        selected + sel_off,
                        DS4_N_EXPERT_USED) == 0) {
                return false;
            }
            seeded_rows++;
        }
        seeded_layers++;
    }
    if (profile) {
        fprintf(stderr,
                "ds4: Metal streaming prefill expert-cache seed k=%u layers=%u rows=%u time=%.3f ms\n",
                seed_tokens,
                seeded_layers,
                seeded_rows,
                (now_sec() - t0) * 1000.0);
    }
    return true;
}

static bool metal_graph_seed_streaming_expert_cache_from_hotlist(
        ds4_gpu_graph     *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    if (!metal_graph_streaming_expert_hotlist_enabled(g)) return true;
    if (!model || !weights) return false;

    uint32_t cache_budget = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!metal_graph_streaming_expert_cache_seed_layer_expected(g, layer)) {
            continue;
        }

        const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
        const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
        if (layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
            layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
            fprintf(stderr, "ds4: streaming expert hotlist budget byte size overflow at layer %u\n", il);
            return false;
        }
        const uint64_t gate_expert_bytes = layer->ffn_gate_exps->dim[1] * gate_row_bytes;
        const uint64_t down_expert_bytes = layer->ffn_down_exps->dim[1] * down_row_bytes;
        cache_budget = ds4_gpu_stream_expert_cache_budget_for_expert_size(
                gate_expert_bytes,
                down_expert_bytes);
        break;
    }
    if (cache_budget == 0) return true;
    const uint32_t preload_count =
        metal_graph_streaming_expert_preload_count(g, cache_budget);
    if (preload_count == 0) return true;
    const uint32_t current_count =
        ds4_gpu_stream_expert_cache_current_count();
    const char *path = glm_graph_env_value("DS4_ROCM_STREAMING_EXPERT_HOTLIST",
                                           "DS4_METAL_STREAMING_EXPERT_HOTLIST");
    const bool from_file = path && path[0];
    const bool refresh_builtin_glm =
        !from_file && g_ds4_shape.variant == DS4_VARIANT_GLM52;
    const bool profile =
        glm_graph_env_present("DS4_ROCM_STREAMING_EXPERT_HOTLIST_PROFILE",
                              "DS4_METAL_STREAMING_EXPERT_HOTLIST_PROFILE");
    if (!from_file && !refresh_builtin_glm && current_count >= preload_count) {
        if (profile) {
            fprintf(stderr,
                    "ds4: streaming expert hotlist seed skipped preload=%u current=%u\n",
                    preload_count,
                    current_count);
        }
        return true;
    }

    int32_t experts[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    uint32_t priorities[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    uint32_t counts[DS4_MAX_LAYER];
    bool seen[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    memset(experts, 0, sizeof(experts));
    memset(priorities, 0, sizeof(priorities));
    memset(counts, 0, sizeof(counts));
    memset(seen, 0, sizeof(seen));

    uint32_t loaded = 0;
    if (from_file) {
        if (!metal_graph_streaming_expert_hotlist_load_file(path,
                                                           preload_count,
                                                           experts,
                                                           priorities,
                                                           counts,
                                                           seen,
                                                           &loaded)) {
            return false;
        }
    } else if (!metal_graph_streaming_expert_hotlist_load_default(preload_count,
                                                                  experts,
                                                                  priorities,
                                                                  counts,
                                                                  seen,
                                                                  &loaded)) {
        return false;
    }
    if (loaded == 0) return true;

    const double t0 = profile ? now_sec() : 0.0;
    uint32_t seeded_layers = 0;
    uint32_t seeded_experts = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t n = counts[il];
        if (n == 0) continue;
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!metal_graph_streaming_expert_cache_seed_layer_expected(g, layer)) {
            continue;
        }

        const uint64_t gate_row_bytes = routed_expert_row_bytes(layer->ffn_gate_exps);
        const uint64_t down_row_bytes = routed_expert_row_bytes(layer->ffn_down_exps);
        if (layer->ffn_gate_exps->dim[1] > UINT64_MAX / gate_row_bytes ||
            layer->ffn_down_exps->dim[1] > UINT64_MAX / down_row_bytes) {
            fprintf(stderr, "ds4: streaming expert hotlist seed byte size overflow at layer %u\n", il);
            return false;
        }
        const uint64_t gate_expert_bytes = layer->ffn_gate_exps->dim[1] * gate_row_bytes;
        const uint64_t down_expert_bytes = layer->ffn_down_exps->dim[1] * down_row_bytes;
        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        if (ds4_gpu_stream_expert_cache_seed_experts(
                    &table,
                    experts[il],
                    priorities[il],
                    n) == 0) {
            return false;
        }
        seeded_layers++;
        seeded_experts += n;
    }
    if (profile) {
        const char *source_name = NULL;
        if (from_file) {
            source_name = path;
        } else if (g_ds4_shape.variant == DS4_VARIANT_GLM52) {
            source_name = "built-in-glm52";
        } else if (g_ds4_shape.variant == DS4_VARIANT_FLASH) {
            source_name = "built-in-flash";
        } else if (g_ds4_shape.variant == DS4_VARIANT_PRO) {
            source_name = "built-in-pro";
        } else {
            source_name = "built-in";
        }
        fprintf(stderr,
                "ds4: streaming expert hotlist seed source=%s preload=%u loaded=%u layers=%u experts=%u time=%.3f ms\n",
                source_name,
                preload_count,
                loaded,
                seeded_layers,
                seeded_experts,
                (now_sec() - t0) * 1000.0);
    }
    return true;
}

typedef struct {
    int id0;
    int id1;
    float value0;
    float value1;
    bool valid;
    bool fast_attention;
} metal_graph_top2_result;

#endif /* !DS4_NO_GPU */
