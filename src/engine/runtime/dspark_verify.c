#include "engine_internal.h"

/* Dspark Verify module. */
/* Keep the support KV ring aligned while the scheduler skips proposals. */
bool metal_graph_dspark_ring_maintain(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  pos) {
    if (!g || !dspark_model || !dw ||
        !g->dspark_capture_valid ||
        g->dspark_cache_len == 0 ||
        !metal_graph_dspark_cache_ends_at(g, pos) ||
        !dspark_stage0_weights_ready(g, dw) ||
        !dspark_stage_cache_ready(g, dw) ||
        !metal_graph_batch_kv_raw(g) || !metal_graph_batch_kv(g)) {
        return false;
    }
    for (uint32_t stage = 0; stage < dw->n_stages; stage++) {
        if (!dspark_stage_block_ready(g, dw, stage)) return false;
    }

    const ds4_dspark_stage_weights *stage0 = &dw->stage[0];
    const uint64_t in_dim = (uint64_t)dw->target_layer_count * DS4_N_EMBD;
    ds4_gpu_tensor *kv_raw_view =
        ds4_gpu_tensor_view(metal_graph_batch_kv_raw(g),
                            0,
                            (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    ds4_gpu_tensor *kv_view =
        ds4_gpu_tensor_view(metal_graph_batch_kv(g),
                            0,
                            (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    bool ok = kv_raw_view && kv_view && ds4_gpu_begin_commands() != 0;
    if (ok) {
        ok = metal_graph_matmul_plain_tensor(g->dspark_stage0_proj,
                                             dspark_model,
                                             stage0->main_proj,
                                             in_dim,
                                             DS4_N_EMBD,
                                             g->dspark_target_hidden,
                                             1);
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_weight_tensor(g->dspark_main_x,
                                            g->dspark_stage0_proj,
                                            dspark_model->map,
                                            dspark_model->size,
                                            stage0->main_norm->abs_offset,
                                            DS4_N_EMBD,
                                            DS4_RMS_EPS) != 0;
    }
    for (uint32_t stage = 0; ok && stage < dw->n_stages; stage++) {
        const ds4_layer_weights *block = &dw->stage[stage].block;
        ok = metal_graph_matmul_plain_tensor(kv_raw_view,
                                             dspark_model,
                                             block->attn_kv,
                                             DS4_N_EMBD,
                                             DS4_N_HEAD_DIM,
                                             g->dspark_main_x,
                                             1);
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(
                         kv_view,
                         kv_raw_view,
                         dspark_model->map,
                         dspark_model->size,
                         block->attn_kv_a_norm->abs_offset,
                         DS4_N_HEAD_DIM,
                         1,
                         DS4_RMS_EPS) != 0;
        if (ok) ok = ds4_gpu_rope_tail_tensor(kv_view,
                                               1,
                                               1,
                                               DS4_N_HEAD_DIM,
                                               DS4_N_ROT,
                                               pos,
                                               0,
                                               false,
                                               DS4_ROPE_FREQ_BASE,
                                               1.0f,
                                               0.0f,
                                               1.0f,
                                               DS4_ROPE_YARN_BETA_FAST,
                                               DS4_ROPE_YARN_BETA_SLOW) != 0;
        if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv_view,
                                                          1,
                                                          DS4_N_HEAD_DIM,
                                                          DS4_N_ROT) != 0;
        if (ok) ok = ds4_gpu_store_raw_kv_batch_tensor(
                         g->dspark_raw_cache[stage],
                         kv_view,
                         g->dspark_cache_cap,
                         pos,
                         1,
                         DS4_N_HEAD_DIM) != 0;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    ds4_gpu_tensor_free(kv_view);
    ds4_gpu_tensor_free(kv_raw_view);
    if (ok) (void)metal_graph_dspark_cache_claim_appended_row(g, pos);
    return ok;
}

static ds4_gpu_tensor *metal_graph_dspark_final_output_hc(const ds4_gpu_graph *g) {
    if (!g) return NULL;
    if (getenv("DS4_DSPARK_DISABLE_FINAL_OUTPUT_ALIAS") == NULL &&
        metal_graph_batch_next_hc(g)) {
        return metal_graph_batch_next_hc(g);
    }
    return g->dspark_stage_output_hc;
}

bool dspark_final_head_ready(
        const ds4_gpu_graph      *g,
        const ds4_weights        *base_weights,
        const ds4_dspark_weights *dw) {
    if (!g || !base_weights || !dw ||
        dw->n_stages == 0 ||
        dw->n_stages > DS4_DSPARK_MAX_STAGES ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        !base_weights->output ||
        !metal_graph_dspark_final_output_hc(g) ||
        !metal_graph_batch_hc_mix(g) ||
        !metal_graph_batch_hc_split(g) ||
        !metal_graph_batch_flat_hc(g) ||
        !metal_graph_batch_ffn_cur(g) ||
        !metal_graph_batch_ffn_norm(g) ||
        !g->spec_logits) {
        return false;
    }

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t draft = dw->block_size;
    const uint64_t vocab_dim = base_weights->output->dim[1];
    if (!final->norm ||
        !final->hc_head_base ||
        !final->hc_head_fn ||
        !final->hc_head_scale ||
        final->norm->type != DS4_TENSOR_F32 ||
        final->hc_head_base->type != DS4_TENSOR_F32 ||
        !dspark_tensor_type_matches(final->hc_head_fn->type,
                                    DS4_DSPARK_LAYOUT_PLAIN) ||
        final->hc_head_scale->type != DS4_TENSOR_F32 ||
        !tensor_type_is_dense_quant(base_weights->output->type)) {
        return false;
    }

    return final->norm->ndim == 1 &&
           final->norm->dim[0] == DS4_N_EMBD &&
           final->hc_head_base->ndim == 1 &&
           final->hc_head_base->dim[0] == DS4_N_HC &&
           final->hc_head_fn->ndim == 2 &&
           final->hc_head_fn->dim[0] == hc_dim &&
           final->hc_head_fn->dim[1] == DS4_N_HC &&
           final->hc_head_scale->ndim == 1 &&
           final->hc_head_scale->dim[0] == 1 &&
           base_weights->output->ndim == 2 &&
           base_weights->output->dim[0] == DS4_N_EMBD &&
           vocab_dim == DS4_N_VOCAB &&
           ds4_gpu_tensor_bytes(
                   metal_graph_dspark_final_output_hc(g)) >=
               draft * hc_dim * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_flat_hc(g)) >=
               draft * hc_dim * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_hc_mix(g)) >=
               draft * DS4_N_HC * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_hc_split(g)) >=
               draft * DS4_N_HC * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_ffn_cur(g)) >=
               draft * DS4_N_EMBD * sizeof(float) &&
           ds4_gpu_tensor_bytes(metal_graph_batch_ffn_norm(g)) >=
               draft * DS4_N_EMBD * sizeof(float) &&
           ds4_gpu_tensor_bytes(g->spec_logits) >=
               draft * vocab_dim * sizeof(float);
}

bool metal_graph_eval_dspark_base_logits(
        ds4_gpu_graph            *g,
        const ds4_model          *base_model,
        const ds4_weights        *base_weights,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw) {
    if (!g || !base_model || !base_weights || !dspark_model || !dw ||
        !dspark_final_head_ready(g, base_weights, dw)) {
        return false;
    }

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    const uint32_t draft = dw->block_size;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t vocab_dim = base_weights->output->dim[1];
    ds4_gpu_tensor *stage_output_hc = metal_graph_dspark_final_output_hc(g);
    ds4_gpu_tensor *output_pre =
        ds4_gpu_tensor_view(metal_graph_batch_hc_mix(g),
                            0,
                            (uint64_t)draft * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_weights =
        ds4_gpu_tensor_view(metal_graph_batch_hc_split(g),
                            0,
                            (uint64_t)draft * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_embd =
        ds4_gpu_tensor_view(metal_graph_batch_ffn_cur(g),
                            0,
                            (uint64_t)draft * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *output_norm =
        ds4_gpu_tensor_view(metal_graph_batch_ffn_norm(g),
                            0,
                            (uint64_t)draft * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *logits =
        ds4_gpu_tensor_view(g->spec_logits,
                            0,
                            (uint64_t)draft * vocab_dim * sizeof(float));

    bool ok = stage_output_hc && output_pre && output_weights && output_embd &&
              output_norm && logits;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(metal_graph_batch_flat_hc(g),
                                                     stage_output_hc,
                                                     (uint32_t)hc_dim,
                                                     draft,
                                                     DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(output_pre,
                                                 dspark_model,
                                                 final->hc_head_fn,
                                                 hc_dim,
                                                 DS4_N_HC,
                                                 metal_graph_batch_flat_hc(g),
                                                 draft);
    if (ok) ok = ds4x_graph_output_hc(
            g, output_weights, output_pre, dspark_model,
            final->hc_head_scale->abs_offset,
            final->hc_head_base->abs_offset);
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                 stage_output_hc,
                                                 output_weights,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                      output_embd,
                                                      dspark_model->map,
                                                      dspark_model->size,
                                                      final->norm->abs_offset,
                                                      DS4_N_EMBD,
                                                      draft,
                                                      DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(logits,
                                                  base_model,
                                                  base_weights->output,
                                                  DS4_N_EMBD,
                                                  vocab_dim,
                                                  output_norm,
                                                  draft);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) (void)ds4_gpu_synchronize();

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

bool metal_graph_eval_dspark_final_hidden(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        bool                      commands_open) {
    if (!g || !dspark_model || !dw ||
        dw->n_stages == 0 ||
        dw->n_stages > DS4_DSPARK_MAX_STAGES ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        !metal_graph_dspark_final_output_hc(g) ||
        !metal_graph_batch_hc_mix(g) ||
        !metal_graph_batch_hc_split(g) ||
        !metal_graph_batch_flat_hc(g) ||
        !metal_graph_batch_ffn_cur(g) ||
        !metal_graph_batch_ffn_norm(g)) {
        return false;
    }

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    const uint32_t draft = dw->block_size;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    if (!final->norm ||
        !final->hc_head_base ||
        !final->hc_head_fn ||
        !final->hc_head_scale ||
        final->norm->type != DS4_TENSOR_F32 ||
        final->hc_head_base->type != DS4_TENSOR_F32 ||
        !dspark_tensor_type_matches(final->hc_head_fn->type,
                                    DS4_DSPARK_LAYOUT_PLAIN) ||
        final->hc_head_scale->type != DS4_TENSOR_F32 ||
        final->norm->ndim != 1 ||
        final->norm->dim[0] != DS4_N_EMBD ||
        final->hc_head_base->ndim != 1 ||
        final->hc_head_base->dim[0] != DS4_N_HC ||
        final->hc_head_fn->ndim != 2 ||
        final->hc_head_fn->dim[0] != hc_dim ||
        final->hc_head_fn->dim[1] != DS4_N_HC ||
        final->hc_head_scale->ndim != 1 ||
        final->hc_head_scale->dim[0] != 1 ||
        ds4_gpu_tensor_bytes(metal_graph_dspark_final_output_hc(g)) <
            (uint64_t)draft * hc_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(metal_graph_batch_flat_hc(g)) <
            (uint64_t)draft * hc_dim * sizeof(float) ||
        ds4_gpu_tensor_bytes(metal_graph_batch_hc_mix(g)) <
            (uint64_t)draft * DS4_N_HC * sizeof(float) ||
        ds4_gpu_tensor_bytes(metal_graph_batch_hc_split(g)) <
            (uint64_t)draft * DS4_N_HC * sizeof(float) ||
        ds4_gpu_tensor_bytes(metal_graph_batch_ffn_cur(g)) <
            (uint64_t)draft * DS4_N_EMBD * sizeof(float) ||
        ds4_gpu_tensor_bytes(metal_graph_batch_ffn_norm(g)) <
            (uint64_t)draft * DS4_N_EMBD * sizeof(float)) {
        return false;
    }

    ds4_gpu_tensor *output_pre =
        ds4_gpu_tensor_view(metal_graph_batch_hc_mix(g),
                            0,
                            (uint64_t)draft * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_weights =
        ds4_gpu_tensor_view(metal_graph_batch_hc_split(g),
                            0,
                            (uint64_t)draft * DS4_N_HC * sizeof(float));
    ds4_gpu_tensor *output_embd =
        ds4_gpu_tensor_view(metal_graph_batch_ffn_cur(g),
                            0,
                            (uint64_t)draft * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *output_norm =
        ds4_gpu_tensor_view(metal_graph_batch_ffn_norm(g),
                            0,
                            (uint64_t)draft * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *stage_output_hc = metal_graph_dspark_final_output_hc(g);

    bool ok = stage_output_hc && output_pre && output_weights &&
              output_embd && output_norm;
    if (ok && !commands_open) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = ds4_gpu_rms_norm_plain_rows_tensor(metal_graph_batch_flat_hc(g),
                                                     stage_output_hc,
                                                     (uint32_t)hc_dim,
                                                     draft,
                                                     DS4_RMS_EPS) != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(output_pre,
                                                 dspark_model,
                                                 final->hc_head_fn,
                                                 hc_dim,
                                                 DS4_N_HC,
                                                 metal_graph_batch_flat_hc(g),
                                                 draft);
    if (ok) ok = ds4x_graph_output_hc(
            g, output_weights, output_pre, dspark_model,
            final->hc_head_scale->abs_offset,
            final->hc_head_base->abs_offset);
    if (ok) ok = ds4_gpu_hc_weighted_sum_tensor(output_embd,
                                                 stage_output_hc,
                                                 output_weights,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(output_norm,
                                                      output_embd,
                                                      dspark_model->map,
                                                      dspark_model->size,
                                                      final->norm->abs_offset,
                                                      DS4_N_EMBD,
                                                      draft,
                                                      DS4_RMS_EPS) != 0;
    if (ok && !commands_open) ok = ds4_gpu_end_commands() != 0;
    if (!ok && !commands_open) (void)ds4_gpu_synchronize();

    ds4_gpu_tensor_free(output_norm);
    ds4_gpu_tensor_free(output_embd);
    ds4_gpu_tensor_free(output_weights);
    ds4_gpu_tensor_free(output_pre);
    return ok;
}

bool metal_graph_eval_dspark_base_logits_from_hidden(
        ds4_gpu_graph      *g,
        const ds4_model    *base_model,
        const ds4_weights  *base_weights,
        const ds4_dspark_weights *dw) {
    if (!g || !base_model || !base_weights || !dw ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        !base_weights->output ||
        !tensor_type_is_dense_quant(base_weights->output->type) ||
        base_weights->output->ndim != 2 ||
        base_weights->output->dim[0] != DS4_N_EMBD ||
        base_weights->output->dim[1] != DS4_N_VOCAB ||
        !metal_graph_batch_ffn_norm(g) ||
        !g->spec_logits ||
        ds4_gpu_tensor_bytes(metal_graph_batch_ffn_norm(g)) <
            (uint64_t)dw->block_size * DS4_N_EMBD * sizeof(float) ||
        ds4_gpu_tensor_bytes(g->spec_logits) <
            (uint64_t)dw->block_size * DS4_N_VOCAB * sizeof(float)) {
        return false;
    }

    ds4_gpu_tensor *output_norm =
        ds4_gpu_tensor_view(metal_graph_batch_ffn_norm(g),
                            0,
                            (uint64_t)dw->block_size *
                                DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *logits =
        ds4_gpu_tensor_view(g->spec_logits,
                            0,
                            (uint64_t)dw->block_size *
                                DS4_N_VOCAB * sizeof(float));
    bool ok = output_norm && logits;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = metal_graph_matmul_plain_tensor(logits,
                                                  base_model,
                                                  base_weights->output,
                                                  DS4_N_EMBD,
                                                  DS4_N_VOCAB,
                                                  output_norm,
                                                  dw->block_size);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    if (!ok) (void)ds4_gpu_synchronize();

    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(output_norm);
    return ok;
}

bool dspark_markov_probe_ready(
        const ds4_dspark_weights *dw) {
    if (!dw ||
        dw->n_stages == 0 ||
        dw->n_stages > DS4_DSPARK_MAX_STAGES ||
        dw->block_size == 0 ||
        dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE ||
        dw->markov_rank == 0) {
        return false;
    }

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    if (!final->markov_w1 ||
        !final->markov_w2 ||
        !dspark_tensor_type_matches(final->markov_w1->type,
                                    DS4_DSPARK_LAYOUT_DENSE) ||
        !dspark_tensor_type_matches(final->markov_w2->type,
                                    DS4_DSPARK_LAYOUT_DENSE)) {
        return false;
    }

    return final->markov_w1->ndim == 2 &&
           final->markov_w1->dim[0] == dw->markov_rank &&
           final->markov_w1->dim[1] == DS4_N_VOCAB &&
           final->markov_w2->ndim == 2 &&
           final->markov_w2->dim[0] == dw->markov_rank &&
           final->markov_w2->dim[1] == DS4_N_VOCAB;
}

static bool dspark_dense_row_to_f32(
        float           *out,
        const ds4_model *model,
        const ds4_tensor *t,
        uint32_t         row) {
    if (!out || !model || !t || t->ndim != 2 || row >= t->dim[1]) {
        return false;
    }

    const uint64_t width = t->dim[0];
    if (t->type == DS4_TENSOR_F32) {
        const float *base = tensor_data(model, t);
        memcpy(out, base + (uint64_t)row * width, width * sizeof(out[0]));
        return true;
    }
    if (t->type == DS4_TENSOR_F16) {
        const uint16_t *base = tensor_data(model, t);
        const uint16_t *src = base + (uint64_t)row * width;
        for (uint64_t i = 0; i < width; i++) out[i] = f16_to_f32(src[i]);
        return true;
    }
    if (t->type == DS4_TENSOR_Q8_0) {
        const uint64_t blocks = (width + 31u) / 32u;
        const uint8_t *src =
            (const uint8_t *)tensor_data(model, t) +
            (uint64_t)row * blocks * 34u;
        for (uint64_t b = 0; b < blocks; b++) {
            uint16_t scale_bits;
            memcpy(&scale_bits, src + b * 34u, sizeof(scale_bits));
            const float scale = f16_to_f32(scale_bits);
            const int8_t *qs = (const int8_t *)(src + b * 34u + 2u);
            const uint64_t i0 = b * 32u;
            const uint64_t n = width - i0 < 32u ? width - i0 : 32u;
            for (uint64_t i = 0; i < n; i++) {
                out[i0 + i] = scale * (float)qs[i];
            }
        }
        return true;
    }
    return false;
}

static uint32_t dspark_argmax_f32(const float *x, uint32_t n) {
    uint32_t best = 0;
    float best_v = x[0];
    for (uint32_t i = 1; i < n; i++) {
        if (x[i] > best_v) {
            best_v = x[i];
            best = i;
        }
    }
    return best;
}

static void dspark_markov_q8_0_argmax_worker(
        void *vctx,
        uint64_t row0,
        uint64_t row1) {
    dspark_markov_q8_0_argmax_ctx *ctx = vctx;
    uint64_t slot = ctx->rows_per_slot ? row0 / ctx->rows_per_slot : 0;
    if (slot >= DS4_MAX_THREADS) slot = DS4_MAX_THREADS - 1u;

    float best_v = -FLT_MAX;
    uint32_t best = (uint32_t)row0;
    for (uint64_t row = row0; row < row1; row++) {
        const uint8_t *wrow = ctx->data + row * ctx->blocks * 34u;
        const float score =
            ctx->logits[row] +
            dot_q8_0_row(wrow, ctx->xq, ctx->xscale, ctx->in_dim, ctx->blocks);
        if (score > best_v) {
            best_v = score;
            best = (uint32_t)row;
        }
    }

    ctx->best_idx[slot] = best;
    ctx->best_val[slot] = best_v;
}

static bool dspark_markov_q8_0_argmax(
        uint32_t        *token_out,
        const ds4_model *model,
        const ds4_tensor *w,
        const float     *state,
        const float     *logits) {
    if (!token_out ||
        !model ||
        !w ||
        !state ||
        !logits ||
        w->type != DS4_TENSOR_Q8_0 ||
        w->ndim != 2 ||
        w->dim[1] > UINT32_MAX) {
        return false;
    }

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (out_dim == 0 ||
        blocks == 0 ||
        blocks > (uint64_t)SIZE_MAX / 32u ||
        blocks > (uint64_t)SIZE_MAX / sizeof(float)) {
        return false;
    }

    enum { DSPARK_MARKOV_ARGMAX_STACK_BLOCKS = 32 };
    int8_t xq_stack[DSPARK_MARKOV_ARGMAX_STACK_BLOCKS * 32u];
    float xscale_stack[DSPARK_MARKOV_ARGMAX_STACK_BLOCKS];
    const bool use_stack = blocks <= DSPARK_MARKOV_ARGMAX_STACK_BLOCKS;
    int8_t *xq = use_stack ? xq_stack : xmalloc((size_t)blocks * 32u);
    float *xscale = use_stack ? xscale_stack :
        xmalloc((size_t)blocks * sizeof(xscale[0]));
    quantize_q8_0_activation(state, xq, xscale, in_dim);

    ds4_threads_init();
    const uint32_t n_slots =
        g_pool.n_threads == 0 ? 1u : g_pool.n_threads;
    const uint64_t rows_per_slot = (out_dim + n_slots - 1u) / n_slots;
    dspark_markov_q8_0_argmax_ctx ctx = {
        .data = tensor_data(model, w),
        .xq = xq,
        .xscale = xscale,
        .logits = logits,
        .in_dim = in_dim,
        .blocks = blocks,
        .rows_per_slot = rows_per_slot,
    };
    for (uint32_t i = 0; i < DS4_MAX_THREADS; i++) {
        ctx.best_idx[i] = 0;
        ctx.best_val[i] = -FLT_MAX;
    }

    ds4_parallel_for(out_dim, dspark_markov_q8_0_argmax_worker, &ctx);

    uint32_t best = 0;
    float best_v = -FLT_MAX;
    for (uint32_t slot = 0; slot < n_slots && slot < DS4_MAX_THREADS; slot++) {
        const uint64_t row0 = (uint64_t)slot * rows_per_slot;
        if (row0 >= out_dim) break;
        if (ctx.best_val[slot] > best_v) {
            best_v = ctx.best_val[slot];
            best = ctx.best_idx[slot];
        }
    }

    if (!use_stack) {
        free(xscale);
        free(xq);
    }
    *token_out = best;
    return true;
}

/* Exact target verification preserves correctness when this diagnostic mode
 * proposes directly from the support model's base logits. */
static bool dspark_markov_bias_disabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_DSPARK_NO_MARKOV");
        cached = (env && env[0] && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cached == 1;
}

static bool dspark_disable_fused_cpu_markov_argmax(void) {
    static int cache = -1;
    if (cache < 0) {
        const char *env = getenv("DS4_DSPARK_DISABLE_FUSED_CPU_MARKOV_ARGMAX");
        cache = (env && env[0] && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cache != 0;
}

bool dspark_disable_reuse_confidence0_markov(void) {
    static int cache = -1;
    if (cache < 0) {
        const char *env = getenv("DS4_DSPARK_DISABLE_REUSE_CONFIDENCE0_MARKOV");
        cache = (env && env[0] && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cache != 0;
}

bool dspark_apply_markov_greedy_probe(
        float                  *logits,
        const ds4_model        *dspark_model,
        const ds4_dspark_weights *dw,
        int                     first_prev_token,
        float                  *markov_state,
        float                  *markov_bias,
        int32_t                 proposal[DS4_DSPARK_MAX_BLOCK_SIZE],
        uint32_t               *proposal_len) {
    if (proposal_len) *proposal_len = 0;
    if (!logits ||
        !dspark_model ||
        !dw ||
        !markov_state ||
        !markov_bias ||
        !proposal ||
        first_prev_token < 0 ||
        (uint32_t)first_prev_token >= DS4_N_VOCAB ||
        !dspark_markov_probe_ready(dw)) {
        return false;
    }

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    const bool no_bias = dspark_markov_bias_disabled();
    int32_t prev_token = first_prev_token;
    for (uint32_t draft = 0; draft < dw->block_size; draft++) {
        float *row = logits + (uint64_t)draft * DS4_N_VOCAB;
        if (!no_bias) {
            if (!dspark_dense_row_to_f32(markov_state,
                                         dspark_model,
                                         final->markov_w1,
                                         (uint32_t)prev_token)) {
                return false;
            }
            matvec_any(markov_bias, dspark_model, final->markov_w2, markov_state);
            for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
                row[i] += markov_bias[i];
            }
        }
        const uint32_t token = dspark_argmax_f32(row, DS4_N_VOCAB);
        proposal[draft] = (int32_t)token;
        prev_token = (int32_t)token;
    }

    if (proposal_len) *proposal_len = dw->block_size;
    return true;
}

bool dspark_confidence_probe_ready(
        const ds4_dspark_weights *dw) {
    if (!dspark_markov_probe_ready(dw)) return false;
    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    if (!final->confidence_proj ||
        !dspark_tensor_type_matches(final->confidence_proj->type,
                                    DS4_DSPARK_LAYOUT_DENSE)) {
        return false;
    }
    return final->confidence_proj->ndim == 2 &&
           final->confidence_proj->dim[0] ==
               (uint64_t)DS4_N_EMBD + dw->markov_rank &&
           final->confidence_proj->dim[1] == 1;
}

bool dspark_eval_confidence_probe(
        float                  *confidence_logits,
        const float            *hidden_rows,
        const ds4_model        *dspark_model,
        const ds4_dspark_weights *dw,
        int                     first_prev_token,
        const int32_t           proposal[DS4_DSPARK_MAX_BLOCK_SIZE],
        float                  *markov_state,
        float                  *features,
        uint32_t               *confidence_len) {
    if (confidence_len) *confidence_len = 0;
    if (!confidence_logits ||
        !hidden_rows ||
        !dspark_model ||
        !dw ||
        !proposal ||
        !markov_state ||
        !features ||
        first_prev_token < 0 ||
        (uint32_t)first_prev_token >= DS4_N_VOCAB ||
        !dspark_confidence_probe_ready(dw)) {
        return false;
    }

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    int32_t prev_token = first_prev_token;
    for (uint32_t draft = 0; draft < dw->block_size; draft++) {
        if (prev_token < 0 || (uint32_t)prev_token >= DS4_N_VOCAB) {
            return false;
        }
        if (!dspark_dense_row_to_f32(markov_state,
                                     dspark_model,
                                     final->markov_w1,
                                     (uint32_t)prev_token)) {
            return false;
        }
        memcpy(features,
               hidden_rows + (uint64_t)draft * DS4_N_EMBD,
               (uint64_t)DS4_N_EMBD * sizeof(features[0]));
        memcpy(features + DS4_N_EMBD,
               markov_state,
               (uint64_t)dw->markov_rank * sizeof(features[0]));
        matvec_any(confidence_logits + draft,
                   dspark_model,
                   final->confidence_proj,
                   features);
        prev_token = proposal[draft];
    }

    if (confidence_len) *confidence_len = dw->block_size;
    return true;
}

bool dspark_apply_markov_confidence_lazy_runtime(
        ds4_gpu_graph          *g,
        const ds4_model        *dspark_model,
        const ds4_dspark_weights *dw,
        int                     first_prev_token,
        float                   confidence_threshold,
        float                  *logits,
        float                  *markov_bias,
        float                  *features,
        size_t                  features_cap,
        int32_t                 proposal[DS4_DSPARK_MAX_BLOCK_SIZE],
        uint32_t               *proposal_len,
        uint32_t               *confidence_len,
        uint32_t               *confidence_prefix_len,
        bool                    reuse_first_confidence,
        float                  *confidence0) {
    if (proposal_len) *proposal_len = 0;
    if (confidence_len) *confidence_len = 0;
    if (confidence_prefix_len) *confidence_prefix_len = 0;
    if (confidence0 && !reuse_first_confidence) *confidence0 = 0.0f;
    if (!g ||
        !g->spec_logits ||
        !metal_graph_batch_ffn_norm(g) ||
        !dspark_model ||
        !dw ||
        !logits ||
        !markov_bias ||
        !features ||
        !proposal ||
        confidence_threshold <= 0.0f ||
        first_prev_token < 0 ||
        (uint32_t)first_prev_token >= DS4_N_VOCAB ||
        (reuse_first_confidence && !confidence0) ||
        !dspark_markov_probe_ready(dw) ||
        !dspark_confidence_probe_ready(dw)) {
        return false;
    }

    const uint64_t logits_bytes =
        (uint64_t)DS4_N_VOCAB * sizeof(float);
    const uint64_t hidden_bytes =
        (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t feature_count =
        (uint64_t)DS4_N_EMBD + (uint64_t)dw->markov_rank;
    if (feature_count > features_cap) return false;
    float *markov_state = features + DS4_N_EMBD;
    bool ok = true;

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    int32_t prev_token = first_prev_token;
    uint32_t produced = 0;
    uint32_t confident = 0;
    for (uint32_t draft = 0; ok && draft < dw->block_size; draft++) {
        if (prev_token < 0 || (uint32_t)prev_token >= DS4_N_VOCAB) {
            ok = false;
            break;
        }
        float confidence_logit = 0.0f;
        if (draft == 0 && reuse_first_confidence) {
            confidence_logit = *confidence0;
        } else {
            ok = dspark_dense_row_to_f32(markov_state,
                                         dspark_model,
                                         final->markov_w1,
                                         (uint32_t)prev_token);
            if (!ok) break;

            ok = ds4_gpu_tensor_read(metal_graph_batch_ffn_norm(g),
                                     (uint64_t)draft * hidden_bytes,
                                     features,
                                     hidden_bytes) != 0;
            if (!ok) break;
            matvec_any(&confidence_logit,
                       dspark_model,
                       final->confidence_proj,
                       features);
        }
        if (draft == 0 && confidence0) *confidence0 = confidence_logit;
        if (confidence_len) *confidence_len = draft + 1u;
        if (sigmoid_stable(confidence_logit) < confidence_threshold) {
            ok = true;
            break;
        }

        int32_t token = -1;
        /* Apply the Markov bias and argmax without reading back the full
         * logits row. */
        if (ok && !dspark_markov_bias_disabled() &&
            getenv("DS4_DSPARK_NO_GPU_MARKOV") == NULL &&
            g->dspark_draft_tokens &&
            dw->markov_rank != 0 && (dw->markov_rank & 31u) == 0 &&
            final->markov_w1->type == DS4_TENSOR_Q8_0 &&
            final->markov_w2->type == DS4_TENSOR_Q8_0) {
            ds4_gpu_tensor *row_view =
                ds4_gpu_tensor_view(g->spec_logits,
                                    (uint64_t)draft * logits_bytes,
                                    logits_bytes);
            uint64_t gpu_key = 0;
            bool gpu_ok = row_view &&
                ds4_gpu_dspark_markov_argmax_tensor(
                    g->dspark_draft_tokens,
                    row_view,
                    dspark_model->map,
                    dspark_model->size,
                    final->markov_w1->abs_offset,
                    final->markov_w2->abs_offset,
                    (uint32_t)prev_token,
                    DS4_N_VOCAB,
                    dw->markov_rank) != 0 &&
                ds4_gpu_tensor_read(g->dspark_draft_tokens,
                                    0,
                                    &gpu_key,
                                    sizeof(gpu_key)) != 0;
            ds4_gpu_tensor_free(row_view);
            const uint32_t gpu_token = ~(uint32_t)(gpu_key & 0xffffffffu);
            if (gpu_ok && gpu_key != 0 && gpu_token < DS4_N_VOCAB) {
                token = (int32_t)gpu_token;
                proposal[draft] = token;
                produced = draft + 1u;
                confident = produced;
                prev_token = token;
                continue;
            }
        }
        if (ok) {
            ok = ds4_gpu_tensor_read(g->spec_logits,
                                     (uint64_t)draft * logits_bytes,
                                     logits,
                                     logits_bytes) != 0;
            if (ok) {
                uint32_t fused_token = 0;
                if (dspark_markov_bias_disabled()) {
                    token = (int32_t)dspark_argmax_f32(logits, DS4_N_VOCAB);
                } else if (!dspark_disable_fused_cpu_markov_argmax() &&
                    dspark_markov_q8_0_argmax(&fused_token,
                                              dspark_model,
                                              final->markov_w2,
                                              markov_state,
                                              logits)) {
                    token = (int32_t)fused_token;
                } else {
                    matvec_any(markov_bias, dspark_model, final->markov_w2, markov_state);
                    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) {
                        logits[i] += markov_bias[i];
                    }
                    token = (int32_t)dspark_argmax_f32(logits, DS4_N_VOCAB);
                }
            }
        }
        if (!ok || token < 0 || (uint32_t)token >= DS4_N_VOCAB) {
            ok = false;
            break;
        }
        proposal[draft] = token;
        produced = draft + 1u;
        confident = produced;
        prev_token = token;
    }

    if (ok) {
        if (proposal_len) *proposal_len = produced;
        if (confidence_prefix_len) *confidence_prefix_len = confident;
    }
    return ok;
}

bool dspark_eval_confidence0_runtime(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        int                       first_prev_token,
        float                    *features,
        size_t                    features_cap,
        float                    *confidence0) {
    if (confidence0) *confidence0 = 0.0f;
    if (!confidence0 ||
        !g ||
        !metal_graph_batch_ffn_norm(g) ||
        !dspark_model ||
        !dw ||
        !features ||
        first_prev_token < 0 ||
        (uint32_t)first_prev_token >= DS4_N_VOCAB ||
        !dspark_confidence_probe_ready(dw)) {
        return false;
    }

    const uint64_t hidden_bytes =
        (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t feature_count =
        (uint64_t)DS4_N_EMBD + (uint64_t)dw->markov_rank;
    if (feature_count > features_cap ||
        ds4_gpu_tensor_bytes(metal_graph_batch_ffn_norm(g)) < hidden_bytes) {
        return false;
    }

    float *markov_state = features + DS4_N_EMBD;
    bool ok = true;

    const ds4_dspark_stage_weights *final =
        &dw->stage[dw->n_stages - 1u];
    if (ok) {
        ok = dspark_dense_row_to_f32(markov_state,
                                     dspark_model,
                                     final->markov_w1,
                                     (uint32_t)first_prev_token);
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(metal_graph_batch_ffn_norm(g),
                                 0,
                                 features,
                                 hidden_bytes) != 0;
    }
    if (ok) {
        matvec_any(confidence0, dspark_model, final->confidence_proj, features);
    }

    return ok;
}

uint32_t dspark_confident_prefix_len(
        const float *confidence_logits,
        uint32_t     confidence_len,
        float        threshold) {
    if (!confidence_logits || confidence_len == 0 || threshold <= 0.0f) {
        return confidence_len;
    }
    for (uint32_t i = 0; i < confidence_len; i++) {
        if (sigmoid_stable(confidence_logits[i]) < threshold) return i;
    }
    return confidence_len;
}

bool metal_graph_reset_prefill_state(ds4_gpu_graph *g) {
    memset(g->layer_n_comp, 0, sizeof(g->layer_n_comp));
    memset(g->layer_n_index_comp, 0, sizeof(g->layer_n_index_comp));
    metal_graph_dspark_cache_reset(g);
    metal_graph_dspark_capture_invalidate(g);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g->layer_raw_cache[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
        const uint64_t attn_rows = (uint64_t)coff * ratio;
        if (!metal_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows)) return false;
        if (!metal_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows)) return false;
        if (ratio == 4) {
            const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_rows = (uint64_t)coff * ratio;
            if (!metal_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows)) return false;
            if (!metal_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows)) return false;
        }
    }
    return true;
}
