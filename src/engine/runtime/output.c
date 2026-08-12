#include "engine_internal.h"

/* Output module. */
/* Encode the final HC collapse, output norm, and vocab projection on CUDA. */
bool metal_graph_encode_output_head(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim) {
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
    if (ok) ok = ds4x_graph_output_hc(
            g, metal_graph_output_weights(g), metal_graph_output_pre(g), model,
            weights->output_hc_scale->abs_offset,
            weights->output_hc_base->abs_offset);
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
    if (ok) {
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

/* Batched output head for speculative verification.
 *
 * A target verifier only needs top-1 ids for intermediate draft rows and full
 * logits for the last accepted row.  Running the normal one-row output head in
 * a loop serializes the HC collapse, output norm, and Q8 vocab projection.  For
 * tiny DSpark suffixes we instead process all rows together and let the GPU reduce
 * each row to a top id; the CPU reads back just those ids plus the last row's
 * logits needed to continue the exact target stream. */
/* Shared vocab-head matmul pads small batches to 8 rows for the exact-mma Q8
 * kernel. */
bool metal_graph_output_logits_head_matmul(
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
    if (ok) {
        ok = metal_graph_matmul_dense_quant_tensor(
                logits, model, weights->output, DS4_N_EMBD, vocab_dim,
                output_norm, head_rows);
    }
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    return ok;
}

bool metal_graph_encode_output_head_batch(
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
    if (ok) ok = ds4x_graph_output_hc(
            g, output_weights, output_pre, model,
            weights->output_hc_scale->abs_offset,
            weights->output_hc_base->abs_offset);
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

bool metal_graph_matmul_plain_tensor(
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
    fprintf(stderr, "ds4: CUDA plain matmul does not support %s\n", tensor_type_name(w->type));
    return false;
}

bool metal_graph_dense_quant_row_bytes(
        const ds4_tensor *w,
        uint64_t          in_dim,
        uint64_t         *row_bytes) {
    if (row_bytes) *row_bytes = 0;
    if (!w || !row_bytes || !tensor_type_is_dense_quant(w->type)) return false;
    return tensor_nbytes(w->type, in_dim, row_bytes);
}

bool metal_graph_matmul_dense_quant_abs(
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

bool metal_graph_matmul_dense_quant_tensor(
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

bool metal_graph_attention_output_dense_quant_low(
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

bool metal_graph_attention_output_dense_quant_batch(
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

bool metal_graph_matmul_q8_0_named_tensor(
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
