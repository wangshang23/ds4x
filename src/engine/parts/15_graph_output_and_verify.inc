#ifndef DS4_NO_GPU
/* Encode the final HC collapse, output norm, and vocab projection on Metal. */
static bool metal_graph_encode_output_head(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim) {
    /* switch to head_tier before the output-head pipeline.
     * Single-tier (placement == NULL): no-op (head_tier == 0 == active_tier).
     * Note: head_tier was captured in metal_graph_alloc_raw_cap; this
     * helper consults it directly (and also covers the case where the
     * preceding decode layer ran on a different tier — copy_xdev ferries
     * the active cur_hc across the boundary). */
    if (g->placement) {
        if (!metal_graph_set_active_tier_decode(g, g->head_tier)) return false;
    }
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const bool output_stage_profile = g->output_stage_profile;
    double output_stage_t0 = output_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_OUTPUT_STAGE(name) do { \
        if (ok && output_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("output", (name), DS4_N_LAYER, 0, 1, &output_stage_t0); \
        } \
    } while (0)
    bool ok = ds4_gpu_rms_norm_plain_tensor(metal_graph_flat_hc(g), metal_graph_cur_hc(g), (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    DS4_METAL_PROFILE_OUTPUT_STAGE("hc_flat_norm");
    if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_output_pre(g),
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             metal_graph_flat_hc(g),
                                             1) != 0;
    DS4_METAL_PROFILE_OUTPUT_STAGE("hc_pre");
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc_pre", metal_graph_output_pre(g), DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(metal_graph_output_weights(g),
                                                    metal_graph_output_pre(g),
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    DS4_METAL_PROFILE_OUTPUT_STAGE("hc_weights");
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc_weights", metal_graph_output_weights(g), DS4_N_HC, DS4_N_LAYER, 0);
    }
    if (ok) {
        ok = ds4_gpu_hc_weighted_sum_tensor(metal_graph_output_embd(g),
                                              metal_graph_cur_hc(g),
                                              metal_graph_output_weights(g),
                                              DS4_N_EMBD,
                                              DS4_N_HC) != 0;
    }
    DS4_METAL_PROFILE_OUTPUT_STAGE("hc_weighted_sum");
    if (ok) {
        metal_graph_debug_dump_tensor("result_hc", metal_graph_output_embd(g), DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_output_norm(g),
                                              metal_graph_output_embd(g),
                                              model->map,
                                              model->size,
                                              weights->output_norm->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_RMS_EPS) != 0;
    }
    DS4_METAL_PROFILE_OUTPUT_STAGE("output_norm");
    if (ok) {
        metal_graph_debug_dump_tensor("result_norm", metal_graph_output_norm(g), DS4_N_EMBD, DS4_N_LAYER, 0);
    }
    if (ok && g->tp_world == 2 && g->tp_logits_half) {
        /* Vocab-split: this rank computes its half of the head rows into
         * its logits view; the halves are bit-identical to the full head
         * (same kernel, same rows) and the worker ships its half to the
         * leader after the eval. */
        const uint64_t tp_vhalf = vocab_dim / 2u;
        uint64_t head_row_bytes = 0;
        ok = metal_graph_dense_quant_row_bytes(weights->output,
                                               DS4_N_EMBD,
                                               &head_row_bytes);
        if (ok) ok = metal_graph_matmul_dense_quant_abs(g->tp_logits_half,
                                                        model,
                                                        weights->output,
                                                        weights->output->abs_offset +
                                                            (uint64_t)g->tp_rank * tp_vhalf * head_row_bytes,
                                                        DS4_N_EMBD,
                                                        tp_vhalf,
                                                        metal_graph_output_norm(g),
                                                        1);
    } else if (ok && g->cuda_tp_ep && g->cuda_tp_output) {
        ok = metal_graph_output_logits_head_matmul(
                g, model, weights, metal_graph_output_norm(g),
                metal_graph_logits(g), 1, vocab_dim);
    } else if (ok) {
        ok = metal_graph_matmul_dense_quant_tensor(metal_graph_logits(g),
                                                   model,
                                                   weights->output,
                                                   DS4_N_EMBD,
                                                   vocab_dim,
                                                   metal_graph_output_norm(g),
                                                   1);
    }
    if (ok) {
        metal_graph_debug_dump_tensor("result_output", metal_graph_logits(g), vocab_dim, DS4_N_LAYER, 0);
    }
#undef DS4_METAL_PROFILE_OUTPUT_STAGE
    return ok;
}

/* Greedy-only output head: compute one local top-1 candidate per output TP
 * split and leave the full split logits on their owning tiers.  This avoids
 * gathering the whole vocabulary row back to the head tier when the caller only
 * needs the next argmax token. */
static bool metal_graph_encode_output_head_split_top1(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim,
        int                    cuda_tp_output_tiers[DS4_MAX_GPUS],
        uint32_t              *cuda_tp_output_ways_out) {
    if (!g || !model || !weights ||
        !cuda_tp_output_tiers || !cuda_tp_output_ways_out ||
        vocab_dim > UINT32_MAX) {
        return false;
    }
    *cuda_tp_output_ways_out = 0;

    if (g->placement) {
        if (!metal_graph_set_active_tier_decode(g, g->head_tier)) return false;
    }
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_plain_tensor(metal_graph_flat_hc(g),
                                            metal_graph_cur_hc(g),
                                            (uint32_t)hc_dim,
                                            DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_output_pre(g),
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             metal_graph_flat_hc(g),
                                             1) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(metal_graph_output_weights(g),
                                                    metal_graph_output_pre(g),
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(metal_graph_output_embd(g),
                                                  metal_graph_cur_hc(g),
                                                  metal_graph_output_weights(g),
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_output_norm(g),
                                                  metal_graph_output_embd(g),
                                                  model->map,
                                                  model->size,
                                                  weights->output_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (!ok) return false;

    const uint32_t cuda_tp_output_ways =
        g->cuda_tp_output ? metal_graph_cuda_tp_output_tiers(g, cuda_tp_output_tiers) : 0;
    const bool cuda_tp_output =
        g->cuda_tp_output &&
        cuda_tp_output_ways >= 2u &&
        weights->output->type == DS4_TENSOR_Q8_0 &&
        weights->output->ndim == 2 &&
        weights->output->dim[0] == DS4_N_EMBD &&
        weights->output->dim[1] == vocab_dim &&
        vocab_dim >= 2;
    if (!cuda_tp_output) return false;

    for (uint32_t i = 0; i < cuda_tp_output_ways; i++) {
        const int t = cuda_tp_output_tiers[i];
        if (t < 0 || t >= DS4_MAX_GPUS ||
            !g->output_norm_by_tier[t] ||
            !g->logits_by_tier[t] ||
            !g->comp_selected_by_tier[t] ||
            !g->comp_mask_by_tier[t]) {
            return false;
        }
    }

    const bool fused_top1 = metal_graph_cuda_output_fused_top1_requested();
    const uint64_t row_bytes = metal_graph_q8_0_row_bytes(DS4_N_EMBD);
    uint64_t split_start[DS4_MAX_GPUS] = {0};
    uint64_t split_count[DS4_MAX_GPUS] = {0};
    ds4_gpu_tensor split_logits[DS4_MAX_GPUS];
    memset(split_logits, 0, sizeof(split_logits));

    for (uint32_t i = 0; ok && i < cuda_tp_output_ways; i++) {
        const int t = cuda_tp_output_tiers[i];
        split_start[i] = (vocab_dim * (uint64_t)i) / cuda_tp_output_ways;
        const uint64_t split_end =
            (vocab_dim * (uint64_t)(i + 1u)) / cuda_tp_output_ways;
        split_count[i] = split_end - split_start[i];
        if (split_count[i] == 0 || split_count[i] > UINT32_MAX) {
            ok = false;
            break;
        }
        if (fused_top1) {
            if (t != g->head_tier) {
                ok = ds4_gpu_tensor_copy_xdev(
                        g->output_norm_by_tier[t],
                        metal_graph_output_norm(g),
                        (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
            }
        } else if (t == g->head_tier) {
            ok = metal_graph_borrow_tensor_view(&split_logits[i],
                                                metal_graph_logits(g),
                                                split_start[i] * sizeof(float),
                                                split_count[i] * sizeof(float));
        } else {
            ok = metal_graph_borrow_tensor_view(&split_logits[i],
                                                g->logits_by_tier[t],
                                                0,
                                                split_count[i] * sizeof(float)) &&
                 ds4_gpu_tensor_copy_xdev(
                        g->output_norm_by_tier[t],
                        metal_graph_output_norm(g),
                        (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
        }
    }

    for (uint32_t i = 1; ok && i < cuda_tp_output_ways; i++) {
        const int t = cuda_tp_output_tiers[i];
        ok = ds4_gpu_set_current_device(t) == 0;
        if (ok && fused_top1) {
            ok = ds4_gpu_matmul_q8_0_top1_tensor(g->comp_selected_by_tier[t],
                                                 g->comp_mask_by_tier[t],
                                                 model->map,
                                                 model->size,
                                                 weights->output->abs_offset +
                                                     split_start[i] * row_bytes,
                                                 DS4_N_EMBD,
                                                 split_count[i],
                                                 g->output_norm_by_tier[t],
                                                 (uint32_t)split_start[i]) != 0;
        } else if (ok) {
            ok = ds4_gpu_matmul_q8_0_tensor(&split_logits[i],
                                            model->map,
                                            model->size,
                                            weights->output->abs_offset +
                                                split_start[i] * row_bytes,
                                            DS4_N_EMBD,
                                            split_count[i],
                                            g->output_norm_by_tier[t],
                                            1) != 0;
        }
        if (ok && !fused_top1) {
            ok = ds4_gpu_indexer_top1_value_tensor(g->comp_selected_by_tier[t],
                                                   g->comp_mask_by_tier[t],
                                                   &split_logits[i],
                                                   (uint32_t)split_count[i],
                                                   1,
                                                   (uint32_t)split_start[i]) != 0;
        }
    }
    if (ok) ok = ds4_gpu_set_current_device(g->head_tier) == 0;
    if (ok && fused_top1) {
        ok = ds4_gpu_matmul_q8_0_top1_tensor(g->comp_selected_by_tier[g->head_tier],
                                             g->comp_mask_by_tier[g->head_tier],
                                             model->map,
                                             model->size,
                                             weights->output->abs_offset,
                                             DS4_N_EMBD,
                                             split_count[0],
                                             metal_graph_output_norm(g),
                                             (uint32_t)split_start[0]) != 0;
    } else if (ok) {
        ok = ds4_gpu_matmul_q8_0_tensor(&split_logits[0],
                                        model->map,
                                        model->size,
                                        weights->output->abs_offset,
                                        DS4_N_EMBD,
                                        split_count[0],
                                        metal_graph_output_norm(g),
                                        1) != 0;
    }
    if (ok && !fused_top1) {
        ok = ds4_gpu_indexer_top1_value_tensor(g->comp_selected_by_tier[g->head_tier],
                                               g->comp_mask_by_tier[g->head_tier],
                                               &split_logits[0],
                                               (uint32_t)split_count[0],
                                               1,
                                               (uint32_t)split_start[0]) != 0;
    }
    for (uint32_t i = 1; ok && i < cuda_tp_output_ways; i++) {
        const int t = cuda_tp_output_tiers[i];
        ds4_gpu_tensor head_id_dst;
        ds4_gpu_tensor head_value_dst;
        ok = metal_graph_borrow_tensor_view(&head_id_dst,
                                            g->comp_selected_by_tier[g->head_tier],
                                            (uint64_t)i * sizeof(uint32_t),
                                            sizeof(uint32_t)) &&
             metal_graph_borrow_tensor_view(&head_value_dst,
                                            g->comp_mask_by_tier[g->head_tier],
                                            (uint64_t)i * sizeof(float),
                                            sizeof(float)) &&
             ds4_gpu_tensor_copy_xdev3(&head_id_dst,
                                        g->comp_selected_by_tier[t],
                                        sizeof(uint32_t),
                                        &head_value_dst,
                                        g->comp_mask_by_tier[t],
                                        sizeof(float),
                                        NULL,
                                        NULL,
                                        0) != 0;
    }
    if (ok) {
        ok = ds4_gpu_set_current_device(g->head_tier) == 0;
        *cuda_tp_output_ways_out = cuda_tp_output_ways;
    }
    return ok;
}

static bool metal_graph_read_output_split_top1(
        ds4_gpu_graph *g,
        uint32_t       output_ways,
        int           *top_id) {
    if (!g || !top_id || output_ways == 0 || output_ways > DS4_MAX_GPUS) {
        return false;
    }
    bool have_best = false;
    uint32_t best_id = 0;
    float best_value = 0.0f;
    uint32_t cand_ids[DS4_MAX_GPUS] = {0};
    float cand_values[DS4_MAX_GPUS] = {0.0f};
    bool ok = ds4_gpu_tensor_read(g->comp_selected_by_tier[g->head_tier],
                                  0,
                                  cand_ids,
                                  (uint64_t)output_ways * sizeof(cand_ids[0])) != 0 &&
              ds4_gpu_tensor_read(g->comp_mask_by_tier[g->head_tier],
                                  0,
                                  cand_values,
                                  (uint64_t)output_ways * sizeof(cand_values[0])) != 0;
    for (uint32_t i = 0; ok && i < output_ways; i++) {
        const uint32_t cand_id = cand_ids[i];
        const float cand_value = cand_values[i];
        if (!have_best ||
            cand_value > best_value ||
            (cand_value == best_value && cand_id < best_id)) {
            have_best = true;
            best_id = cand_id;
            best_value = cand_value;
        }
    }
    ok = ok && have_best && best_id <= (uint32_t)INT32_MAX;
    if (ok) *top_id = (int)best_id;
    return ok;
}

/* Batched output head for speculative verification.
 *
 * A target verifier only needs top-1 ids for intermediate draft rows and full
 * logits for the last accepted row.  Running the normal one-row output head in
 * a loop serializes the HC collapse, output norm, and Q8 vocab projection.  For
 * tiny MTP suffixes we instead process all rows together and let the GPU reduce
 * each row to a top id; the CPU reads back just those ids plus the last row's
 * logits needed to continue the exact target stream. */
/* Shared vocab-head matmul: pads small batches to 8 rows for the exact-mma Q8
 * kernel and shards the vocabulary across output-TP tiers. */
static bool metal_graph_output_logits_head_matmul(
        ds4_gpu_graph        *g,
        const ds4_model      *model,
        const ds4_weights    *weights,
        ds4_gpu_tensor       *norm_full,
        ds4_gpu_tensor       *dst_logits,
        uint32_t              n_tokens,
        uint64_t              vocab_dim) {
    if (!g || !model || !weights || !norm_full || n_tokens == 0 ||
        !dst_logits ||
        ds4_gpu_tensor_bytes(dst_logits) <
            (uint64_t)n_tokens * vocab_dim * sizeof(float)) {
        return false;
    }
    const uint32_t head_rows =
        (n_tokens > 1 && n_tokens < 8 &&
         ds4_gpu_tensor_bytes(dst_logits) >= 8u * vocab_dim * sizeof(float) &&
         ds4_gpu_tensor_bytes(norm_full) >=
             8u * DS4_N_EMBD * sizeof(float)) ? 8u : n_tokens;
    ds4_gpu_tensor *output_norm =
        ds4_gpu_tensor_view(norm_full,
                            0,
                            (uint64_t)head_rows * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *logits =
        ds4_gpu_tensor_view(dst_logits,
                            0,
                            (uint64_t)head_rows * vocab_dim * sizeof(float));
    bool ok = output_norm && logits;
    if (ok && head_rows > n_tokens) {
        ds4_gpu_tensor *pad =
            ds4_gpu_tensor_view(norm_full,
                                (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float),
                                (uint64_t)(head_rows - n_tokens) * DS4_N_EMBD *
                                    sizeof(float));
        ok = pad &&
             ds4_gpu_tensor_fill_f32(pad,
                                     0.0f,
                                     (uint64_t)(head_rows - n_tokens) *
                                         DS4_N_EMBD) != 0;
        ds4_gpu_tensor_free(pad);
    }
    /* Output TP for the speculative batch, mirroring the decode head: each
     * device matmuls its VRAM-resident vocab shard. Shard outputs land
     * compactly in logits_by_tier[t] ([head_rows x split]) and are gathered
     * into spec_logits rows. */
    int tp_tiers[DS4_MAX_GPUS] = {0};
    const uint32_t tp_ways = (ok && g->cuda_tp_output)
        ? metal_graph_cuda_tp_output_tiers(g, tp_tiers) : 0;
    bool tp_ok = ok && tp_ways >= 2u &&
                 weights->output->type == DS4_TENSOR_Q8_0 &&
                 weights->output->ndim == 2 &&
                 weights->output->dim[0] == DS4_N_EMBD &&
                 weights->output->dim[1] == vocab_dim &&
                 head_rows <= DS4_DSPARK_MAX_BLOCK_SIZE &&
                 getenv("DS4_DSPARK_VERIFY_HEAD_NO_TP") == NULL;
    for (uint32_t i = 0; tp_ok && i < tp_ways; i++) {
        const int t = tp_tiers[i];
        tp_ok = t >= 0 && t < DS4_MAX_GPUS &&
                g->logits_by_tier[t] &&
                ds4_gpu_tensor_bytes(g->logits_by_tier[t]) >=
                    (uint64_t)head_rows *
                        ((vocab_dim + tp_ways - 1u) / tp_ways) *
                        sizeof(float) &&
                (t == g->active_tier ||
                 (g->batch_ffn_norm_by_tier[t] &&
                  ds4_gpu_tensor_bytes(g->batch_ffn_norm_by_tier[t]) >=
                      (uint64_t)head_rows * DS4_N_EMBD * sizeof(float)));
    }
    if (tp_ok) {
        const uint64_t row_bytes = metal_graph_q8_0_row_bytes(DS4_N_EMBD);
        const int home_tier = g->active_tier;
        uint64_t split_start[DS4_MAX_GPUS] = {0};
        uint64_t split_count[DS4_MAX_GPUS] = {0};
        for (uint32_t i = 0; ok && i < tp_ways; i++) {
            const int t = tp_tiers[i];
            split_start[i] = (vocab_dim * (uint64_t)i) / tp_ways;
            const uint64_t split_end =
                (vocab_dim * (uint64_t)(i + 1u)) / tp_ways;
            split_count[i] = split_end - split_start[i];
            if (split_count[i] == 0) { ok = false; break; }
            if (t != home_tier) {
                ok = ds4_gpu_tensor_copy_xdev(
                        g->batch_ffn_norm_by_tier[t],
                        output_norm,
                        (uint64_t)head_rows * DS4_N_EMBD *
                            sizeof(float)) != 0;
            }
        }
        for (uint32_t i = 0; ok && i < tp_ways; i++) {
            const int t = tp_tiers[i];
            ok = ds4_gpu_set_current_device(t) == 0;
            if (!ok) break;
            ds4_gpu_tensor *shard_out =
                ds4_gpu_tensor_view(g->logits_by_tier[t],
                                    0,
                                    (uint64_t)head_rows * split_count[i] *
                                        sizeof(float));
            ds4_gpu_tensor *shard_in = t == home_tier ?
                NULL :
                ds4_gpu_tensor_view(g->batch_ffn_norm_by_tier[t],
                                    0,
                                    (uint64_t)head_rows * DS4_N_EMBD *
                                        sizeof(float));
            ok = shard_out && (t == home_tier || shard_in) &&
                 ds4_gpu_matmul_q8_0_tensor(shard_out,
                                            model->map,
                                            model->size,
                                            weights->output->abs_offset +
                                                split_start[i] * row_bytes,
                                            DS4_N_EMBD,
                                            split_count[i],
                                            t == home_tier ? output_norm :
                                                             shard_in,
                                            head_rows) != 0;
            ds4_gpu_tensor_free(shard_in);
            ds4_gpu_tensor_free(shard_out);
        }
        if (ok) ok = ds4_gpu_set_current_device(home_tier) == 0;
        for (uint32_t i = 0; ok && i < tp_ways; i++) {
            const int t = tp_tiers[i];
            for (uint32_t r = 0; ok && r < n_tokens; r++) {
                ds4_gpu_tensor *dst =
                    ds4_gpu_tensor_view(dst_logits,
                                        ((uint64_t)r * vocab_dim +
                                         split_start[i]) * sizeof(float),
                                        split_count[i] * sizeof(float));
                ds4_gpu_tensor *src =
                    ds4_gpu_tensor_view(g->logits_by_tier[t],
                                        (uint64_t)r * split_count[i] *
                                            sizeof(float),
                                        split_count[i] * sizeof(float));
                ok = dst && src &&
                     ds4_gpu_tensor_copy_xdev(dst,
                                              src,
                                              split_count[i] *
                                                  sizeof(float)) != 0;
                ds4_gpu_tensor_free(src);
                ds4_gpu_tensor_free(dst);
            }
        }
    } else if (ok && !(g->cuda_tp_ep && g->cuda_tp_output)) {
        ok = metal_graph_matmul_dense_quant_tensor(logits,
                                                   model,
                                                   weights->output,
                                                   DS4_N_EMBD,
                                                   vocab_dim,
                                                   output_norm,
                                                   head_rows);
    } else if (ok) {
        /* The expert-parallel cache stores only output vocabulary shards, so
         * a single-device full-head fallback would access uncached weights. */
        ok = false;
    }
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    return ok;
}

static bool metal_graph_encode_output_head_batch(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    ds4_gpu_tensor *output_pre = NULL;
    ds4_gpu_tensor *output_weights = NULL;
    ds4_gpu_tensor *output_embd = NULL;
    ds4_gpu_tensor *output_norm = NULL;

    bool ok = true;
    output_pre = ds4_gpu_tensor_view(metal_graph_batch_hc_mix(g),
                                       0,
                                       (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_weights = ds4_gpu_tensor_view(metal_graph_batch_hc_split(g),
                                           0,
                                           (uint64_t)n_tokens * DS4_N_HC * sizeof(float));
    output_embd = ds4_gpu_tensor_view(metal_graph_batch_ffn_cur(g),
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    output_norm = ds4_gpu_tensor_view(metal_graph_batch_ffn_norm(g),
                                        0,
                                        (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ok = output_pre && output_weights && output_embd && output_norm;

    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(metal_graph_batch_flat_hc(g),
                                                      metal_graph_batch_cur_hc(g),
                                                      (uint32_t)hc_dim,
                                                      n_tokens,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_f16_tensor(output_pre,
                                             model->map,
                                             model->size,
                                             weights->output_hc_fn->abs_offset,
                                             hc_dim,
                                             DS4_N_HC,
                                             metal_graph_batch_flat_hc(g),
                                             n_tokens) != 0;
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(output_weights,
                                                    output_pre,
                                                    model->map,
                                                    model->size,
                                                    weights->output_hc_scale->abs_offset,
                                                    weights->output_hc_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                  metal_graph_batch_cur_hc(g),
                                                  output_weights,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                       output_embd,
                                                       model->map,
                                                       model->size,
                                                       weights->output_norm->abs_offset,
                                                       DS4_N_EMBD,
                                                       n_tokens,
                                                       DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_output_logits_head_matmul(
            g, model, weights, metal_graph_batch_ffn_norm(g),
            g->spec_logits, n_tokens, vocab_dim);

    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

static bool metal_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (w->type == DS4_TENSOR_F16) {
        return ds4_gpu_matmul_f16_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_F32) {
        return ds4_gpu_matmul_f32_tensor(out, model->map, model->size,
                                           w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (w->type == DS4_TENSOR_Q8_0) {
        return ds4_gpu_matmul_q8_0_tensor(out, model->map, model->size,
                                            w->abs_offset, in_dim, out_dim, x, n_tok) != 0;
    }
    if (tensor_type_is_dense_quant(w->type)) {
        return ds4_gpu_matmul_quant_tensor(out,
                                           model->map,
                                           model->size,
                                           w->abs_offset,
                                           w->type,
                                           in_dim,
                                           out_dim,
                                           x,
                                           n_tok) != 0;
    }
    fprintf(stderr, "ds4: Metal plain matmul does not support %s\n", tensor_type_name(w->type));
    return false;
}

static bool metal_graph_dense_quant_row_bytes(
        const ds4_tensor *w,
        uint64_t          in_dim,
        uint64_t         *row_bytes) {
    if (row_bytes) *row_bytes = 0;
    if (!w || !row_bytes || !tensor_type_is_dense_quant(w->type)) return false;
    return tensor_nbytes(w->type, in_dim, row_bytes);
}

static bool metal_graph_matmul_dense_quant_abs(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!w || !tensor_type_is_dense_quant(w->type)) return false;
    return ds4_gpu_matmul_quant_tensor(out,
                                       model->map,
                                       model->size,
                                       weight_offset,
                                       w->type,
                                       in_dim,
                                       out_dim,
                                       x,
                                       n_tok) != 0;
}

static bool metal_graph_matmul_dense_quant_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    if (!w) return false;
    return metal_graph_matmul_dense_quant_abs(out,
                                             model,
                                             w,
                                             w->abs_offset,
                                             in_dim,
                                             out_dim,
                                             x,
                                             n_tok);
}

static bool metal_graph_matmul_dense_quant_kslice(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off) {
    if (!w || !tensor_type_is_dense_quant(w->type)) return false;
    return ds4_gpu_matmul_quant_kslice_tensor(out,
                                             model->map,
                                             model->size,
                                             w->abs_offset,
                                             w->type,
                                             full_in_dim,
                                             k_off,
                                             k_cnt,
                                             out_dim,
                                             x,
                                             x_elem_off) != 0;
}

static bool metal_graph_attention_output_dense_quant_low(
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads) {
    (void)g;
    if (!low || !model || !out_a || !heads ||
        group_dim == 0 || rank == 0 || group_cnt == 0) {
        return false;
    }
    if (out_a->type == DS4_TENSOR_Q8_0 && group0 == 0) {
        return ds4_gpu_attention_output_low_q8_tensor(low,
                                                      model->map,
                                                      model->size,
                                                      out_a->abs_offset,
                                                      group_dim,
                                                      rank,
                                                      group_cnt,
                                                      heads) != 0;
    }
    if (out_a->type == DS4_TENSOR_Q4_K) {
        return ds4_gpu_attention_output_low_q4_K_slice_tensor(low,
                                                             model->map,
                                                             model->size,
                                                             out_a->abs_offset,
                                                             group_dim,
                                                             rank,
                                                             group0,
                                                             group_cnt,
                                                             heads) != 0;
    }
    uint64_t row_bytes = 0;
    if (!metal_graph_dense_quant_row_bytes(out_a, group_dim, &row_bytes)) return false;
    const uint64_t group_weight_bytes = rank * row_bytes;
    bool ok = true;
    for (uint32_t i = 0; ok && i < group_cnt; i++) {
        ds4_gpu_tensor *head_view = ds4_gpu_tensor_view(
                heads,
                (uint64_t)i * group_dim * sizeof(float),
                group_dim * sizeof(float));
        ds4_gpu_tensor *low_view = ds4_gpu_tensor_view(
                low,
                (uint64_t)i * rank * sizeof(float),
                rank * sizeof(float));
        ok = head_view && low_view &&
             metal_graph_matmul_dense_quant_abs(low_view,
                                                model,
                                                out_a,
                                                out_a->abs_offset +
                                                    (uint64_t)(group0 + i) * group_weight_bytes,
                                                group_dim,
                                                rank,
                                                head_view,
                                                1);
        ds4_gpu_tensor_free(low_view);
        ds4_gpu_tensor_free(head_view);
    }
    return ok;
}

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
        const ds4_gpu_tensor *heads) {
    if (!out || !low || !g || !model || !out_a || !out_b || !heads ||
        group0 + group_cnt > n_groups_total) {
        return false;
    }
    if (out_a->type == DS4_TENSOR_Q8_0 && out_b->type == DS4_TENSOR_Q8_0) {
        return ds4_gpu_attention_output_q8_tp_tensor(out,
                                                     low,
                                                     model->map,
                                                     model->size,
                                                     out_a->abs_offset,
                                                     out_b->abs_offset,
                                                     group_dim,
                                                     rank,
                                                     n_groups_total,
                                                     group0,
                                                     group_cnt,
                                                     out_dim,
                                                     heads) != 0;
    }
    if (!metal_graph_attention_output_dense_quant_low(low,
                                                     g,
                                                     model,
                                                     out_a,
                                                     group_dim,
                                                     rank,
                                                     group0,
                                                     group_cnt,
                                                     heads)) {
        return false;
    }
    return metal_graph_matmul_dense_quant_kslice(out,
                                                model,
                                                out_b,
                                                (uint64_t)n_groups_total * rank,
                                                (uint64_t)group0 * rank,
                                                (uint64_t)group_cnt * rank,
                                                out_dim,
                                                low,
                                                0);
}

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
        uint32_t                n_tokens) {
    if (!out || !low || !g || !model || !out_a || !out_b || !heads ||
        n_groups == 0 || n_tokens == 0) {
        return false;
    }
    if (out_a->type == DS4_TENSOR_Q8_0 && out_b->type == DS4_TENSOR_Q8_0) {
        return ds4_gpu_attention_output_q8_batch_tensor(out,
                                                        low,
                                                        metal_graph_batch_group_tmp(g),
                                                        metal_graph_batch_low_tmp(g),
                                                        model->map,
                                                        model->size,
                                                        out_a->abs_offset,
                                                        out_b->abs_offset,
                                                        group_dim,
                                                        rank,
                                                        n_groups,
                                                        out_dim,
                                                        heads,
                                                        n_tokens) != 0;
    }
    if (out_a->type == DS4_TENSOR_Q4_K && n_tokens >= 32u) {
        if (ds4_gpu_attention_output_q4_K_batch_tensor(out,
                                                       low,
                                                       metal_graph_batch_group_tmp(g),
                                                       metal_graph_batch_low_tmp(g),
                                                       model->map,
                                                       model->size,
                                                       out_a->abs_offset,
                                                       out_b->abs_offset,
                                                       out_b->type,
                                                       group_dim,
                                                       rank,
                                                       n_groups,
                                                       out_dim,
                                                       heads,
                                                       n_tokens) != 0) {
            return true;
        }
    }

    const uint64_t heads_row_elems = (uint64_t)n_groups * group_dim;
    const uint64_t low_row_elems = (uint64_t)n_groups * rank;
    bool ok = true;
    for (uint32_t t = 0; ok && t < n_tokens; t++) {
        ds4_gpu_tensor *heads_row = ds4_gpu_tensor_view(
                heads,
                (uint64_t)t * heads_row_elems * sizeof(float),
                heads_row_elems * sizeof(float));
        ds4_gpu_tensor *low_row = ds4_gpu_tensor_view(
                low,
                (uint64_t)t * low_row_elems * sizeof(float),
                low_row_elems * sizeof(float));
        ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
                out,
                (uint64_t)t * out_dim * sizeof(float),
                out_dim * sizeof(float));
        ok = heads_row && low_row && out_row &&
             metal_graph_attention_output_dense_quant_low(low_row,
                                                          g,
                                                          model,
                                                          out_a,
                                                          group_dim,
                                                          rank,
                                                          0,
                                                          n_groups,
                                                          heads_row);
        if (ok) ok = metal_graph_matmul_dense_quant_tensor(out_row,
                                                           model,
                                                           out_b,
                                                           low_row_elems,
                                                           out_dim,
                                                           low_row,
                                                           1);
        ds4_gpu_tensor_free(out_row);
        ds4_gpu_tensor_free(low_row);
        ds4_gpu_tensor_free(heads_row);
    }
    return ok;
}

static bool metal_graph_matmul_q8_0_named_tensor(
        const char             *module,
        uint32_t                il,
        uint32_t                pos0,
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok) {
    (void)module;
    (void)il;
    (void)pos0;
    return metal_graph_matmul_dense_quant_tensor(out,
                                                model,
                                                w,
                                                in_dim,
                                                out_dim,
                                                x,
                                                n_tok);
}

static bool metal_graph_encode_output_head_mtp(
        ds4_gpu_graph       *g,
        const ds4_model       *base_model,
        const ds4_weights     *base_weights,
        const ds4_model       *mtp_model,
        const ds4_mtp_weights *mtp,
        uint64_t               vocab_dim) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    bool ok = ds4_gpu_rms_norm_plain_tensor(metal_graph_flat_hc(g), metal_graph_cur_hc(g), (uint32_t)hc_dim, DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_output_pre(g), mtp_model, mtp->hc_head_fn,
                                                 hc_dim, DS4_N_HC, metal_graph_flat_hc(g), 1);
    if (ok) ok = ds4_gpu_output_hc_weights_tensor(metal_graph_output_weights(g),
                                                    metal_graph_output_pre(g),
                                                    mtp_model->map,
                                                    mtp_model->size,
                                                    mtp->hc_head_scale->abs_offset,
                                                    mtp->hc_head_base->abs_offset,
                                                    DS4_N_HC,
                                                    DS4_HC_EPS) != 0;
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(metal_graph_output_embd(g),
                                                  metal_graph_cur_hc(g),
                                                  metal_graph_output_weights(g),
                                                  DS4_N_EMBD,
                                                  DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(metal_graph_output_norm(g),
                                                  metal_graph_output_embd(g),
                                                  mtp_model->map,
                                                  mtp_model->size,
                                                  mtp->norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_dense_quant_tensor(metal_graph_logits(g),
                                                       base_model,
                                                       base_weights->output,
                                                       DS4_N_EMBD,
                                                       vocab_dim,
                                                       metal_graph_output_norm(g),
                                                       1);
    return ok;
}

/* =========================================================================
 * Metal Diagnostic Comparisons.
 * =========================================================================
 *
 * These routines deliberately allocate CPU-side reference buffers and read
 * Metal tensors back.  They are not part of generation; command-line tests use
 * them to localize drift against the C reference pipeline.
 */

static void metal_graph_trace_layer_stages(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        const float            *cpu_in_hc,
        uint32_t                il,
        int                     token) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t shared_in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const bool routed_q8_0 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q8_0;
    const uint64_t routed_q8_x_blocks = expert_in_dim / 32u;
    const uint64_t routed_q8_mid_blocks = down_in_dim / 32u;

    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float post[4];
    float comb[16];
    float ffn_post[4];
    float ffn_comb[16];
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];
    const uint64_t shared_blocks = (shared_in_dim + 31) / 32;
    int8_t *shared_xq = xmalloc((size_t)shared_blocks * 32);
    float *shared_xscale = xmalloc((size_t)shared_blocks * sizeof(float));
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));
    int8_t *routed_q8_xq = routed_q8_0 ? xmalloc((size_t)routed_q8_x_blocks * 32u) : NULL;
    float *routed_q8_xscale = routed_q8_0 ? xmalloc((size_t)routed_q8_x_blocks * sizeof(routed_q8_xscale[0])) : NULL;
    int8_t *routed_q8_midq = routed_q8_0 ? xmalloc((size_t)DS4_N_EXPERT_USED * routed_q8_mid_blocks * 32u) : NULL;
    float *routed_q8_midscale = routed_q8_0 ? xmalloc((size_t)DS4_N_EXPERT_USED * routed_q8_mid_blocks * sizeof(routed_q8_midscale[0])) : NULL;

    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_in_hc, cpu_attn_cur, post, comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_one(cpu_heads, model, layer, cpu_q, cpu_kv);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, il, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_in_hc, post, comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, ffn_post, ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    quantize_q8_0_activation(cpu_ffn_norm, shared_xq, shared_xscale, shared_in_dim);
    matvec_q8_0_pair_prequant(cpu_shared_gate,
                              cpu_shared_up,
                              model,
                              layer->ffn_gate_shexp,
                              layer->ffn_up_shexp,
                              shared_xq,
                              shared_xscale);
    swiglu(cpu_shared_mid, cpu_shared_gate, cpu_shared_up, shared_dim, DS4_SWIGLU_CLAMP_EXP);
    matvec_q8_0(cpu_shared, model, layer->ffn_down_shexp, cpu_shared_mid);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  il,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq,
                                  routed_q8_xq,
                                  routed_q8_xscale,
                                  routed_q8_midq,
                                  routed_q8_midscale);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc, cpu_ffn_out, cpu_after_attn_hc, ffn_post, ffn_comb, DS4_N_EMBD, DS4_N_HC);

    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared_gate = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_up = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared_mid = xmalloc((size_t)shared_dim * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    int gpu_selected[DS4_MAX_EXPERT_USED];
    float gpu_expert_weight[DS4_MAX_EXPERT_USED];

    bool ok = ds4_gpu_tensor_read(metal_graph_attn_cur(g), 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_attn_norm(g), 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_q(g), 0, gpu_q, q_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_kv(g), 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_attn_out(g), 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_after_attn_hc(g), 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_ffn_cur(g), 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_ffn_norm(g), 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_shared_gate(g), 0, gpu_shared_gate, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_shared_up(g), 0, gpu_shared_up, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_shared_mid(g), 0, gpu_shared_mid, shared_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_shared_out(g), 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_router_selected(g), 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_router_weights(g), 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_routed_mid(g), 0, gpu_routed_mid_all, (uint64_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_routed_out(g), 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_ffn_out(g), 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
              ds4_gpu_tensor_read(metal_graph_cur_hc(g), 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0;

    if (ok) {
        fprintf(stderr,
                "ds4: Metal stage layer %u attn_cur=%g/%g attn_norm=%g/%g q=%g/%g kv=%g/%g attn_out=%g/%g after_attn_hc=%g/%g ffn_cur=%g/%g ffn_norm=%g/%g shared=%g/%g router_w=%g routed=%g/%g ffn_out=%g/%g after_ffn_hc=%g/%g\n",
                il,
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD), rms_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD), rms_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim), rms_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM), rms_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD), rms_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim), rms_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD), rms_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD), rms_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD), rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD), rms_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim), rms_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim));
        fprintf(stderr,
                "ds4: Metal shared layer %u gate=%g/%g up=%g/%g mid=%g/%g down=%g/%g\n",
                il,
                max_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim), rms_abs_diff(cpu_shared_gate, gpu_shared_gate, shared_dim),
                max_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim), rms_abs_diff(cpu_shared_up, gpu_shared_up, shared_dim),
                max_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim), rms_abs_diff(cpu_shared_mid, gpu_shared_mid, shared_dim),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD), rms_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD));
        fprintf(stderr,
                "ds4: Metal routed layer %u mid=%g/%g out=%g/%g\n",
                il,
                max_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                rms_abs_diff(routed_mid_all, gpu_routed_mid_all, DS4_N_EXPERT_USED * down_in_dim),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                rms_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: Metal stage layer %u router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    il,
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
    }

    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_routed_mid_all);
    free(gpu_shared);
    free(gpu_shared_mid);
    free(gpu_shared_up);
    free(gpu_shared_gate);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(routed_q8_midscale);
    free(routed_q8_midq);
    free(routed_q8_xscale);
    free(routed_q8_xq);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(shared_xscale);
    free(shared_xq);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_routed);
    free(cpu_shared);
    free(cpu_shared_mid);
    free(cpu_shared_up);
    free(cpu_shared_gate);
    free(cpu_ffn_norm);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_kv);
    free(cpu_qr_norm);
    free(cpu_q);
    free(cpu_attn_norm);
    free(cpu_attn_cur);
}

static int metal_graph_decode_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt,
        bool               quality) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: Metal graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const ds4_layer_weights *layer = &weights->layer[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t vocab_dim = weights->output->dim[1];
    const bool routed_q8_0 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q8_0;
    const uint64_t routed_q8_x_blocks = expert_in_dim / 32u;
    const uint64_t routed_q8_mid_blocks = down_in_dim / 32u;

    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_qr_norm = xmalloc((size_t)q_rank * sizeof(float));
    float *cpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *cpu_heads = xmalloc((size_t)q_dim * sizeof(float));
    float *cpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_post = xmalloc((size_t)DS4_N_HC * sizeof(float));
    float *cpu_ffn_comb = xmalloc((size_t)DS4_N_HC * DS4_N_HC * sizeof(float));
    float *cpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *cpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_q = xmalloc((size_t)q_dim * sizeof(float));
    float *gpu_kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_raw = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    float *gpu_attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_routed = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    float *gpu_after_ffn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    int gpu_selected[DS4_MAX_EXPERT_USED];
    float gpu_expert_weight[DS4_MAX_EXPERT_USED];
    float *routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * down_in_dim * sizeof(float));
    block_q8_K *routed_xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(block_q8_K));
    block_q8_K *routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(block_q8_K));
    int8_t *routed_q8_xq = routed_q8_0 ? xmalloc((size_t)routed_q8_x_blocks * 32u) : NULL;
    float *routed_q8_xscale = routed_q8_0 ? xmalloc((size_t)routed_q8_x_blocks * sizeof(routed_q8_xscale[0])) : NULL;
    int8_t *routed_q8_midq = routed_q8_0 ? xmalloc((size_t)DS4_N_EXPERT_USED * routed_q8_mid_blocks * 32u) : NULL;
    float *routed_q8_midscale = routed_q8_0 ? xmalloc((size_t)DS4_N_EXPERT_USED * routed_q8_mid_blocks * sizeof(routed_q8_midscale[0])) : NULL;
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];

    embed_token_f16(model, weights, token, plain);
    hc_from_plain_embedding(cpu_hc, plain, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          cpu_hc, cpu_attn_cur, cpu_post, cpu_comb);
    layer_attn_norm_one(cpu_attn_norm, model, layer, cpu_attn_cur);
    layer_q_projection_with_lora_one(model, layer, cpu_attn_norm, cpu_q, cpu_qr_norm);
    layer_kv_projection_normed_one(model, layer, cpu_attn_norm, cpu_kv);
    rope_tail_layer_inplace(cpu_q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    rope_tail_layer_inplace(cpu_kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(cpu_kv, DS4_N_HEAD_DIM);
    layer_attention_rows_one(cpu_heads, model, layer, cpu_q, cpu_kv, 1);
    rope_tail_layer_inplace(cpu_heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, 0, 0, true);
    layer_grouped_out_one(cpu_attn_out, model, layer, cpu_heads);
    hc_post_one(cpu_after_attn_hc, cpu_attn_out, cpu_hc, cpu_post, cpu_comb, DS4_N_EMBD, DS4_N_HC);
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          cpu_after_attn_hc, cpu_ffn_cur, cpu_ffn_post, cpu_ffn_comb);
    rms_norm_weight(cpu_ffn_norm, cpu_ffn_cur, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    layer_shared_ffn_one(cpu_shared, model, layer, cpu_ffn_norm);
    layer_routed_moe_one_prealloc(cpu_routed,
                                  model,
                                  layer,
                                  cpu_ffn_norm,
                                  0,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  routed_mid_all,
                                  routed_xq,
                                  routed_midq,
                                  routed_q8_xq,
                                  routed_q8_xscale,
                                  routed_q8_midq,
                                  routed_q8_midscale);
    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, cpu_ffn_norm, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, cpu_ffn_norm);
    }
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) cpu_ffn_out[i] = cpu_shared[i] + cpu_routed[i];
    hc_post_one(cpu_after_ffn_hc,
                cpu_ffn_out,
                cpu_after_attn_hc,
                cpu_ffn_post,
                cpu_ffn_comb,
                DS4_N_EMBD,
                DS4_N_HC);
    output_logits_one(cpu_logits, model, weights, cpu_after_ffn_hc);

    ds4_gpu_graph g;
    bool ok = metal_graph_alloc(&g, weights, layer);
    g.quality = quality;
    g.materialize_ffn_out = true;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(metal_graph_cur_hc(&g),
                                                 model->map,
                                                 model->size,
                                                 weights->token_embd->abs_offset,
                                                 (uint32_t)weights->token_embd->dim[1],
                                                 (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
    if (ok) ok = metal_graph_encode_decode_layer(&g,
                                               model,
                                               layer,
                                               0,
                                               0,
                                               g.layer_raw_cache[0],
                                               g.raw_cap,
                                               0,
                                               1,
                                               token);
    if (ok) {
        /* Single-tier diagnostic: swap the active-tier slots so the head
         * pipeline reads the embedded hidden state from cur_hc. */
        ds4_gpu_tensor *embedded_hc = g.cur_hc_by_tier[g.active_tier];
        g.cur_hc_by_tier[g.active_tier] = g.after_ffn_hc_by_tier[g.active_tier];
        g.after_ffn_hc_by_tier[g.active_tier] = embedded_hc;
    }
    if (ok) ok = metal_graph_encode_output_head(&g, model, weights, vocab_dim);
    if (ok) ok = ds4_gpu_end_commands() != 0;

    if (ok) {
        ok = ds4_gpu_tensor_read(metal_graph_after_ffn_hc(&g), 0, gpu_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_attn_cur(&g), 0, gpu_attn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_attn_norm(&g), 0, gpu_attn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_q(&g), 0, gpu_q, q_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_kv(&g), 0, gpu_kv, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(g.layer_raw_cache[0], 0, gpu_raw, (uint64_t)DS4_N_HEAD_DIM * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_attn_out(&g), 0, gpu_attn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_after_attn_hc(&g), 0, gpu_after_attn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_ffn_cur(&g), 0, gpu_ffn_cur, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_ffn_norm(&g), 0, gpu_ffn_norm, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_shared_out(&g), 0, gpu_shared, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_router_selected(&g), 0, gpu_selected, sizeof(gpu_selected)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_router_weights(&g), 0, gpu_expert_weight, sizeof(gpu_expert_weight)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_routed_out(&g), 0, gpu_routed, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_ffn_out(&g), 0, gpu_ffn_out, (uint64_t)DS4_N_EMBD * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_cur_hc(&g), 0, gpu_after_ffn_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_logits(&g), 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        fprintf(stderr,
                "ds4: Metal graph test layer0 diffs: embed_hc=%g hc_pre=%g attn_norm=%g q_rope=%g kv_rope=%g raw_cache=%g attn_out=%g after_attn_hc=%g ffn_cur=%g ffn_norm=%g shared=%g router_w=%g routed=%g ffn_out=%g after_ffn_hc=%g logits=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_attn_cur, gpu_attn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_attn_norm, gpu_attn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_q, gpu_q, q_dim),
                max_abs_diff(cpu_kv, gpu_kv, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_kv, gpu_raw, DS4_N_HEAD_DIM),
                max_abs_diff(cpu_attn_out, gpu_attn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_attn_hc, gpu_after_attn_hc, hc_dim),
                max_abs_diff(cpu_ffn_cur, gpu_ffn_cur, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_norm, gpu_ffn_norm, DS4_N_EMBD),
                max_abs_diff(cpu_shared, gpu_shared, DS4_N_EMBD),
                max_abs_diff(expert_weight, gpu_expert_weight, DS4_N_EXPERT_USED),
                max_abs_diff(cpu_routed, gpu_routed, DS4_N_EMBD),
                max_abs_diff(cpu_ffn_out, gpu_ffn_out, DS4_N_EMBD),
                max_abs_diff(cpu_after_ffn_hc, gpu_after_ffn_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim));
        if (memcmp(selected, gpu_selected, sizeof(selected)) != 0) {
            fprintf(stderr,
                    "ds4: Metal graph router selected mismatch: cpu=[%d,%d,%d,%d,%d,%d] gpu=[%d,%d,%d,%d,%d,%d]\n",
                    selected[0], selected[1], selected[2], selected[3], selected[4], selected[5],
                    gpu_selected[0], gpu_selected[1], gpu_selected[2], gpu_selected[3], gpu_selected[4], gpu_selected[5]);
        }
        print_vec_stats("metal graph q", gpu_q, q_dim);
        print_vec_stats("metal graph kv", gpu_kv, DS4_N_HEAD_DIM);
        print_vec_stats("metal graph routed", gpu_routed, DS4_N_EMBD);
    } else {
        fprintf(stderr, "ds4: Metal graph test failed while encoding first decode stages\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after graph test failure also failed\n");
        }
    }

    metal_graph_free(&g);
    free(routed_q8_midscale);
    free(routed_q8_midq);
    free(routed_q8_xscale);
    free(routed_q8_xq);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid_all);
    free(gpu_logits);
    free(gpu_after_ffn_hc);
    free(gpu_ffn_out);
    free(gpu_routed);
    free(gpu_shared);
    free(gpu_ffn_norm);
    free(gpu_ffn_cur);
    free(gpu_after_attn_hc);
    free(gpu_attn_out);
    free(gpu_raw);
    free(gpu_kv);
    free(gpu_q);
    free(gpu_attn_norm);
    free(gpu_attn_cur);
    free(gpu_hc);
    free(cpu_kv);
    free(cpu_q);
    free(cpu_attn_out);
    free(cpu_heads);
    free(cpu_ffn_norm);
    free(cpu_routed);
    free(cpu_logits);
    free(cpu_after_ffn_hc);
    free(cpu_ffn_out);
    free(cpu_shared);
    free(cpu_ffn_comb);
    free(cpu_ffn_post);
    free(cpu_ffn_cur);
    free(cpu_after_attn_hc);
    free(cpu_qr_norm);
    free(cpu_attn_norm);
    free(cpu_comb);
    free(cpu_post);
    free(cpu_attn_cur);
    free(cpu_hc);
    free(plain);
    return ok ? 0 : 1;
}

static int metal_graph_first_token_full_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt,
        bool               quality) {
    if (prompt->len <= 0) {
        fprintf(stderr, "ds4: full Metal graph test needs a non-empty prompt\n");
        return 1;
    }

    const int token = prompt->v[0];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t vocab_dim = weights->output->dim[1];
    float *cpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *gpu_hc = xmalloc((size_t)hc_dim * sizeof(float));
    float *cpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));
    float *gpu_logits = xmalloc((size_t)vocab_dim * sizeof(float));

    forward_first_token_cpu(cpu_hc, model, weights, token);
    output_logits_one(cpu_logits, model, weights, cpu_hc);

    ds4_gpu_graph g;
    bool ok = metal_graph_alloc(&g, weights, &weights->layer[0]);
    g.quality = quality;
    const bool trace_layers = getenv("DS4_METAL_GRAPH_TRACE_LAYERS") != NULL;
    if (trace_layers && ok) {
        g.materialize_ffn_out = true;
        const bool teacher_force = getenv("DS4_METAL_GRAPH_TEACHER_FORCE") != NULL;
        const char *stage_layer_env = getenv("DS4_METAL_GRAPH_TRACE_STAGE_LAYER");
        const long stage_layer = stage_layer_env ? strtol(stage_layer_env, NULL, 10) : -1;
        float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
        float *cpu_cur = xmalloc((size_t)hc_dim * sizeof(float));
        float *cpu_next = xmalloc((size_t)hc_dim * sizeof(float));

        embed_token_f16(model, weights, token, plain);
        hc_from_plain_embedding(cpu_cur, plain, DS4_N_EMBD, DS4_N_HC);
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = ds4_gpu_embed_token_hc_tensor(metal_graph_cur_hc(&g),
                                                     model->map,
                                                     model->size,
                                                     weights->token_embd->abs_offset,
                                                     (uint32_t)weights->token_embd->dim[1],
                                                     (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;
        if (ok) ok = ds4_gpu_end_commands() != 0;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            if (teacher_force) {
                ok = ds4_gpu_tensor_write(metal_graph_cur_hc(&g), 0, cpu_cur, hc_dim * sizeof(float)) != 0;
            }
            ok = ds4_gpu_begin_commands() != 0;
            if (ok) ok = metal_graph_encode_decode_layer(&g, model, &weights->layer[il],
                                                       il, 0, g.layer_raw_cache[il], g.raw_cap, 0, 1, token);
            ds4_gpu_tensor *tmp = metal_graph_cur_hc(&g);
            g.cur_hc_by_tier[g.active_tier] = metal_graph_after_ffn_hc(&g);
            g.after_ffn_hc_by_tier[g.active_tier] = tmp;
            if (ok) ok = ds4_gpu_end_commands() != 0;

            layer_forward_self_one(cpu_next, model, &weights->layer[il], cpu_cur, il, 0, token);
            if (ok) ok = ds4_gpu_tensor_read(metal_graph_cur_hc(&g), 0, gpu_hc, hc_dim * sizeof(float)) != 0;
            if (ok) {
                fprintf(stderr,
                        "ds4: Metal full graph layer %u%s hc_max=%g hc_rms=%g\n",
                        il,
                        teacher_force ? " teacher" : "",
                        max_abs_diff(cpu_next, gpu_hc, hc_dim),
                        rms_abs_diff(cpu_next, gpu_hc, hc_dim));
                if (stage_layer == (long)il) {
                    metal_graph_trace_layer_stages(&g, model, &weights->layer[il], cpu_cur, il, token);
                }
            }
            float *ctmp = cpu_cur;
            cpu_cur = cpu_next;
            cpu_next = ctmp;
        }

        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_output_head(&g, model, weights, vocab_dim);
        if (ok) ok = ds4_gpu_end_commands() != 0;

        free(cpu_next);
        free(cpu_cur);
        free(plain);
    } else {
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = ds4_gpu_embed_token_hc_tensor(metal_graph_cur_hc(&g),
                                                     model->map,
                                                     model->size,
                                                     weights->token_embd->abs_offset,
                                                     (uint32_t)weights->token_embd->dim[1],
                                                     (uint32_t)token,
                                                     DS4_N_EMBD,
                                                     DS4_N_HC) != 0;

        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = metal_graph_encode_decode_layer(&g, model, &weights->layer[il],
                                                 il, 0, g.layer_raw_cache[il],
                                                 g.raw_cap, 0, 1, token);
            ds4_gpu_tensor *tmp = metal_graph_cur_hc(&g);
            g.cur_hc_by_tier[g.active_tier] = metal_graph_after_ffn_hc(&g);
            g.after_ffn_hc_by_tier[g.active_tier] = tmp;
        }

        if (ok) ok = metal_graph_encode_output_head(&g, model, weights, vocab_dim);
        if (ok) ok = ds4_gpu_end_commands() != 0;
    }

    if (ok) {
        ok = ds4_gpu_tensor_read(metal_graph_cur_hc(&g), 0, gpu_hc, hc_dim * sizeof(float)) != 0 &&
             ds4_gpu_tensor_read(metal_graph_logits(&g), 0, gpu_logits, vocab_dim * sizeof(float)) != 0;
    }

    if (ok) {
        const uint64_t cpu_top = argmax_f32(cpu_logits, vocab_dim);
        const uint64_t gpu_top = argmax_f32(gpu_logits, vocab_dim);
        fprintf(stderr,
                "ds4: Metal full first-token graph diffs: final_hc_max=%g final_hc_rms=%g logits_max=%g logits_rms=%g cpu_top=%llu gpu_top=%llu cpu_top_logit=%g gpu_top_logit=%g\n",
                max_abs_diff(cpu_hc, gpu_hc, hc_dim),
                rms_abs_diff(cpu_hc, gpu_hc, hc_dim),
                max_abs_diff(cpu_logits, gpu_logits, vocab_dim),
                rms_abs_diff(cpu_logits, gpu_logits, vocab_dim),
                (unsigned long long)cpu_top,
                (unsigned long long)gpu_top,
                cpu_logits[cpu_top],
                gpu_logits[gpu_top]);
    } else {
        fprintf(stderr, "ds4: Metal full first-token graph test failed\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after full graph failure also failed\n");
        }
    }

    metal_graph_free(&g);
    free(gpu_logits);
    free(cpu_logits);
    free(gpu_hc);
    free(cpu_hc);
    return ok ? 0 : 1;
}

#endif /* !DS4_NO_GPU */
