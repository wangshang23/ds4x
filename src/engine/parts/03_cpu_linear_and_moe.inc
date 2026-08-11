/* TP sharding keeps full layers but restricts every routed-expert blob to
 * one contiguous rank range (rank 0 owns the lower expert ids, matching
 * ds4_tp_owns_expert in metal/moe.metal). */
static DS4_MAYBE_UNUSED bool weights_model_map_sharded_spans(
        const ds4_weights *w,
        const ds4_model   *m,
        int                rank,
        ds4_model_map_span_vec *spans) {
    if (!w || !m || !spans || (rank != 0 && rank != 1)) return false;
    memset(spans, 0, sizeof(*spans));
    model_map_span_vec_include_one(spans, w->token_embd);
    for (uint32_t il = 0; il < (uint32_t)DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        /* Dense/attention/router tensors only — the plain decode include
         * would map the full expert blobs in non-streaming mode. */
        model_map_span_vec_include_layer_decode_static(spans, l);
        const ds4_tensor *exps[3] = { l->ffn_gate_exps, l->ffn_up_exps,
                                      l->ffn_down_exps };
        for (int t = 0; t < 3; t++) {
            const ds4_tensor *x = exps[t];
            if (!x || x->ndim != 3 || x->dim[2] < 2) continue;
            uint64_t in_dim = 0, out_dim = 0, row_bytes = 0;
            (void)tensor_expert_bytes(m, x, 0, &in_dim, &out_dim, &row_bytes);
            const uint64_t expert_bytes = out_dim * row_bytes;
            const uint64_t low_experts = x->dim[2] / 2;
            const uint64_t first_expert = rank == 1 ? low_experts : 0;
            const uint64_t owned_experts = rank == 1 ?
                x->dim[2] - low_experts : low_experts;
            const uint64_t owned_bytes = owned_experts * expert_bytes;
            const uint64_t lo = x->abs_offset + first_expert * expert_bytes;
            /* Kernels index experts from the blob base, so the owned range
             * must sit in one contiguous view. Rank 1 takes any remainder. */
            model_map_span_vec_append(spans, lo, lo + owned_bytes, true);
            if (owned_bytes > spans->max_tensor_bytes) {
                spans->max_tensor_bytes = owned_bytes;
            }
        }
    }
    model_map_span_vec_include_output(spans, w);
    return model_map_span_vec_finish(spans);
}

static DS4_MAYBE_UNUSED bool weights_model_map_decode_layer_spans(
        const ds4_weights *w,
        uint32_t il,
        ds4_model_map_span_vec *spans) {
    if (!w || !spans || il >= DS4_N_LAYER) return false;
    memset(spans, 0, sizeof(*spans));
    model_map_span_vec_include_layer_decode(spans, w, il);
    return model_map_span_vec_finish(spans);
}

static DS4_MAYBE_UNUSED bool weights_model_map_decode_static_spans(
        const ds4_weights *w,
        bool include_token,
        bool include_output,
        ds4_model_map_span_vec *spans) {
    if (!w || !spans) return false;
    memset(spans, 0, sizeof(*spans));
    if (include_token) model_map_span_vec_include_one(spans, w->token_embd);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        model_map_span_vec_include_layer_decode(spans, w, il);
    }
    if (include_output) model_map_span_vec_include_output(spans, w);
    return model_map_span_vec_finish(spans);
}

static DS4_MAYBE_UNUSED bool weights_model_map_decode_static_slice_spans(
        const ds4_weights *w,
        uint32_t layer_start,
        uint32_t layer_end,
        bool include_token,
        bool include_output,
        ds4_model_map_span_vec *spans) {
    if (!w || !spans) return false;
    if (layer_start >= DS4_N_LAYER) return false;
    if (layer_end == UINT32_MAX) layer_end = DS4_N_LAYER - 1u;
    if (layer_end >= DS4_N_LAYER || layer_end < layer_start) return false;

    memset(spans, 0, sizeof(*spans));
    if (include_token) model_map_span_vec_include_one(spans, w->token_embd);
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        model_map_span_vec_include_layer_decode(spans, w, il);
    }
    if (include_output) model_map_span_vec_include_output(spans, w);
    return model_map_span_vec_finish(spans);
}

static DS4_MAYBE_UNUSED uint64_t model_map_span_vec_total_bytes(
        const ds4_model_map_span_vec *spans) {
    if (!spans) return 0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < spans->len; i++) {
        const uint64_t bytes = spans->v[i].end - spans->v[i].off;
        if (total > UINT64_MAX - bytes) return UINT64_MAX;
        total += bytes;
    }
    return total;
}

static DS4_MAYBE_UNUSED bool weights_streaming_non_routed_bytes(
        const ds4_weights *w,
        uint64_t          *bytes_out) {
    if (bytes_out) *bytes_out = 0;
    if (!w || !bytes_out) return false;

    ds4_model_map_span_vec spans;
    const bool include_token =
        weights_layer_has_required(&w->layer[0], 0);
    if (!weights_model_map_decode_static_spans(w,
                                               include_token,
                                               weights_have_output_head(w),
                                               &spans)) {
        return false;
    }
    *bytes_out = model_map_span_vec_total_bytes(&spans);
    free(spans.v);
    return true;
}

static DS4_MAYBE_UNUSED bool weights_model_map_token_spans(
        const ds4_weights *w,
        ds4_model_map_span_vec *spans) {
    if (!w || !spans) return false;
    memset(spans, 0, sizeof(*spans));
    model_map_span_vec_include_one(spans, w->token_embd);
    return model_map_span_vec_finish(spans);
}

static DS4_MAYBE_UNUSED bool weights_model_map_output_spans(
        const ds4_weights *w,
        ds4_model_map_span_vec *spans) {
    if (!w || !spans) return false;
    memset(spans, 0, sizeof(*spans));
    model_map_span_vec_include_output(spans, w);
    return model_map_span_vec_finish(spans);
}

static void mtp_weights_bind(ds4_mtp_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));

    w->hc_head_base  = required_tensor(m, "mtp.0.hc_head_base.weight");
    w->hc_head_fn    = required_tensor(m, "mtp.0.hc_head_fn.weight");
    w->hc_head_scale = required_tensor(m, "mtp.0.hc_head_scale.weight");
    w->e_proj        = required_tensor(m, "mtp.0.e_proj.weight");
    w->h_proj        = required_tensor(m, "mtp.0.h_proj.weight");
    w->enorm         = required_tensor(m, "mtp.0.enorm.weight");
    w->hnorm         = required_tensor(m, "mtp.0.hnorm.weight");
    w->norm          = required_tensor(m, "mtp.0.norm.weight");

    ds4_layer_weights *l = &w->block;
    l->hc_attn_fn      = required_tensor(m, "mtp.0.hc_attn_fn.weight");
    l->hc_attn_scale   = required_tensor(m, "mtp.0.hc_attn_scale.weight");
    l->hc_attn_base    = required_tensor(m, "mtp.0.hc_attn_base.weight");
    l->attn_norm       = required_tensor(m, "mtp.0.attn_norm.weight");
    l->attn_q_a        = required_tensor(m, "mtp.0.attn_q_a.weight");
    l->attn_q_a_norm   = required_tensor(m, "mtp.0.attn_q_a_norm.weight");
    l->attn_q_b        = required_tensor(m, "mtp.0.attn_q_b.weight");
    l->attn_kv         = required_tensor(m, "mtp.0.attn_kv.weight");
    l->attn_kv_a_norm  = required_tensor(m, "mtp.0.attn_kv_a_norm.weight");
    l->attn_sinks      = required_tensor(m, "mtp.0.attn_sinks.weight");
    l->attn_output_a   = required_tensor(m, "mtp.0.attn_output_a.weight");
    l->attn_output_b   = required_tensor(m, "mtp.0.attn_output_b.weight");
    l->hc_ffn_fn       = required_tensor(m, "mtp.0.hc_ffn_fn.weight");
    l->hc_ffn_scale    = required_tensor(m, "mtp.0.hc_ffn_scale.weight");
    l->hc_ffn_base     = required_tensor(m, "mtp.0.hc_ffn_base.weight");
    l->ffn_norm        = required_tensor(m, "mtp.0.ffn_norm.weight");
    l->ffn_gate_inp    = required_tensor(m, "mtp.0.ffn_gate_inp.weight");
    l->ffn_exp_probs_b = required_tensor(m, "mtp.0.exp_probs_b.bias");
    l->ffn_gate_exps   = required_tensor(m, "mtp.0.ffn_gate_exps.weight");
    l->ffn_up_exps     = required_tensor(m, "mtp.0.ffn_up_exps.weight");
    l->ffn_down_exps   = required_tensor(m, "mtp.0.ffn_down_exps.weight");
    l->ffn_gate_shexp  = required_tensor(m, "mtp.0.ffn_gate_shexp.weight");
    l->ffn_up_shexp    = required_tensor(m, "mtp.0.ffn_up_shexp.weight");
    l->ffn_down_shexp  = required_tensor(m, "mtp.0.ffn_down_shexp.weight");

    mtp_weights_validate_layout(w);
}

static ds4_tensor *dspark_bind_tensor(
        ds4_dspark_weights *dw,
        const ds4_model    *m,
        uint32_t            stage,
        const char         *suffix,
        bool                required) {
    ds4_tensor *t = tensor_by_mtp_stage_suffix(m, stage, suffix);
    if (t) {
        dw->present_tensors++;
    } else if (required) {
        dw->missing_tensors++;
    }
    return t;
}

static void dspark_bind_block(
        ds4_dspark_weights      *dw,
        ds4_layer_weights       *l,
        const ds4_model         *m,
        uint32_t                 stage) {
    l->hc_attn_fn      = dspark_bind_tensor(dw, m, stage, "hc_attn_fn.weight", true);
    l->hc_attn_scale   = dspark_bind_tensor(dw, m, stage, "hc_attn_scale.weight", true);
    l->hc_attn_base    = dspark_bind_tensor(dw, m, stage, "hc_attn_base.weight", true);
    l->attn_norm       = dspark_bind_tensor(dw, m, stage, "attn_norm.weight", true);
    l->attn_q_a        = dspark_bind_tensor(dw, m, stage, "attn_q_a.weight", true);
    l->attn_q_a_norm   = dspark_bind_tensor(dw, m, stage, "attn_q_a_norm.weight", true);
    l->attn_q_b        = dspark_bind_tensor(dw, m, stage, "attn_q_b.weight", true);
    l->attn_kv         = dspark_bind_tensor(dw, m, stage, "attn_kv.weight", true);
    l->attn_kv_a_norm  = dspark_bind_tensor(dw, m, stage, "attn_kv_a_norm.weight", true);
    l->attn_sinks      = dspark_bind_tensor(dw, m, stage, "attn_sinks.weight", true);
    l->attn_output_a   = dspark_bind_tensor(dw, m, stage, "attn_output_a.weight", true);
    l->attn_output_b   = dspark_bind_tensor(dw, m, stage, "attn_output_b.weight", true);
    l->hc_ffn_fn       = dspark_bind_tensor(dw, m, stage, "hc_ffn_fn.weight", true);
    l->hc_ffn_scale    = dspark_bind_tensor(dw, m, stage, "hc_ffn_scale.weight", true);
    l->hc_ffn_base     = dspark_bind_tensor(dw, m, stage, "hc_ffn_base.weight", true);
    l->ffn_norm        = dspark_bind_tensor(dw, m, stage, "ffn_norm.weight", true);
    l->ffn_gate_inp    = dspark_bind_tensor(dw, m, stage, "ffn_gate_inp.weight", true);
    l->ffn_exp_probs_b = dspark_bind_tensor(dw, m, stage, "exp_probs_b.bias", true);
    l->ffn_gate_exps   = dspark_bind_tensor(dw, m, stage, "ffn_gate_exps.weight", true);
    l->ffn_up_exps     = dspark_bind_tensor(dw, m, stage, "ffn_up_exps.weight", true);
    l->ffn_down_exps   = dspark_bind_tensor(dw, m, stage, "ffn_down_exps.weight", true);
    l->ffn_gate_shexp  = dspark_bind_tensor(dw, m, stage, "ffn_gate_shexp.weight", true);
    l->ffn_up_shexp    = dspark_bind_tensor(dw, m, stage, "ffn_up_shexp.weight", true);
    l->ffn_down_shexp  = dspark_bind_tensor(dw, m, stage, "ffn_down_shexp.weight", true);
}

static void dspark_weights_bind_optional(
        ds4_dspark_weights        *dw,
        const ds4_model           *m,
        const ds4_dspark_summary  *summary) {
    memset(dw, 0, sizeof(*dw));
    if (!m || !summary) return;

    dw->n_stages = summary->stages < DS4_DSPARK_MAX_STAGES ?
                   summary->stages : DS4_DSPARK_MAX_STAGES;
    dw->block_size = summary->block_size;
    dw->markov_rank = summary->markov_rank;
    dw->noise_token_id = summary->noise_token_id;
    dw->target_layer_count = summary->target_layer_count;
    dw->has_block_size = summary->has_block_size;
    dw->has_markov_rank = summary->has_markov_rank;
    dw->has_noise_token_id = summary->has_noise_token_id;
    dw->has_target_layers = summary->has_target_layers;
    memcpy(dw->target_layers,
           summary->target_layers,
           (size_t)dw->target_layer_count * sizeof(dw->target_layers[0]));
    if (summary->stages > DS4_DSPARK_MAX_STAGES) dw->missing_tensors++;

    for (uint32_t stage = 0; stage < dw->n_stages; stage++) {
        ds4_dspark_stage_weights *sw = &dw->stage[stage];
        dspark_bind_block(dw, &sw->block, m, stage);
        if (stage == 0) {
            sw->main_proj = dspark_bind_tensor(dw, m, stage, "main_proj.weight", true);
            sw->main_norm = dspark_bind_tensor(dw, m, stage, "main_norm.weight", true);
        }
    }

    if (dw->n_stages != 0) {
        const uint32_t final_stage = dw->n_stages - 1u;
        ds4_dspark_stage_weights *sw = &dw->stage[final_stage];
        sw->norm = dspark_bind_tensor(dw, m, final_stage, "norm.weight", true);
        sw->hc_head_base =
            dspark_bind_tensor(dw, m, final_stage, "hc_head_base.weight", true);
        sw->hc_head_fn =
            dspark_bind_tensor(dw, m, final_stage, "hc_head_fn.weight", true);
        sw->hc_head_scale =
            dspark_bind_tensor(dw, m, final_stage, "hc_head_scale.weight", true);
        sw->markov_w1 =
            dspark_bind_tensor(dw, m, final_stage, "markov_head.markov_w1.weight", true);
        sw->markov_w2 =
            dspark_bind_tensor(dw, m, final_stage, "markov_head.markov_w2.weight", true);
        sw->confidence_proj =
            dspark_bind_tensor(dw, m, final_stage, "confidence_head.proj.weight", true);
    }

    dspark_weights_validate_layout(dw);
}

static void weights_free(ds4_weights *w) {
    memset(w, 0, sizeof(*w));
}

/* Load one token embedding row and expand it to float activations. */
static void embed_token_f16(const ds4_model *m, const ds4_weights *w, int token, float *out) {
    ds4_tensor *te = w->token_embd;
    if (te->type != DS4_TENSOR_F16 || te->ndim != 2) {
        ds4_die("expected a 2D F16 token embedding tensor");
    }
    if (token < 0 || (uint64_t)token >= te->dim[1]) {
        ds4_die("token id is outside the embedding table");
    }

    const uint16_t *base = tensor_data(m, te);
    const uint64_t stride = te->dim[0];
    const uint16_t *row = base + (uint64_t)token * stride;

    for (uint64_t i = 0; i < stride; i++) {
        out[i] = f16_to_f32(row[i]);
    }
}

static void embed_token_q8_0(const ds4_model *m, const ds4_weights *w, int token, float *out) {
    ds4_tensor *te = w->token_embd;
    if (te->type != DS4_TENSOR_Q8_0 || te->ndim != 2) {
        ds4_die("expected a 2D Q8_0 token embedding tensor");
    }
    if (token < 0 || (uint64_t)token >= te->dim[1]) {
        ds4_die("token id is outside the embedding table");
    }

    const uint64_t n = te->dim[0];
    const uint64_t blocks = (n + 31) / 32;
    const uint8_t *row = (const uint8_t *)tensor_data(m, te) +
                         (uint64_t)token * blocks * 34;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
        const float scale = f16_to_f32(scale_bits);
        const int8_t *qs = (const int8_t *)(row + b * 34 + 2);
        const uint64_t i0 = b * 32;
        const uint64_t bn = n - i0 < 32 ? n - i0 : 32;
        for (uint64_t i = 0; i < bn; i++) {
            out[i0 + i] = scale * (float)qs[i];
        }
    }
}

static void embed_token_any(const ds4_model *m, const ds4_weights *w, int token, float *out) {
    if (!w->token_embd) ds4_die("token embedding tensor is missing");
    switch (w->token_embd->type) {
    case DS4_TENSOR_F16:
        embed_token_f16(m, w, token, out);
        break;
    case DS4_TENSOR_Q8_0:
        embed_token_q8_0(m, w, token, out);
        break;
    default:
        ds4_die("unsupported token embedding tensor type");
    }
}

/* RMSNorm without a learned scale, used by hyper-connection control vectors. */
static void rms_norm_no_weight(float *out, const float *x, uint64_t n, float eps) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];

    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale;
}

/* Standard DS4 RMSNorm with learned per-channel scale. */
static void rms_norm_weight(float *out, const float *x, const float *weight, uint64_t n, float eps) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];

    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

/* Normalize each attention head independently after Q projection. */
static void head_rms_norm_inplace(float *x, uint32_t n_head, uint32_t head_dim, float eps) {
    for (uint32_t h = 0; h < n_head; h++) {
        float *head = x + (uint64_t)h * head_dim;
        double ss = 0.0;
        for (uint32_t i = 0; i < head_dim; i++) ss += (double)head[i] * head[i];

        const float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + eps);
        for (uint32_t i = 0; i < head_dim; i++) head[i] *= scale;
    }
}

typedef struct {
    float *out;
    const uint16_t *data;
    const float *x;
    uint64_t in_dim;
} matvec_f16_ctx;

static inline float dot_f16_row(const uint16_t *row, const float *x, uint64_t n) {
#if defined(__ARM_NEON)
    uint64_t i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        const float16x8_t hv = vreinterpretq_f16_u16(vld1q_u16(row + i));
        const float32x4_t h0 = vcvt_f32_f16(vget_low_f16(hv));
        const float32x4_t h1 = vcvt_f32_f16(vget_high_f16(hv));
        acc0 = vfmaq_f32(acc0, h0, vld1q_f32(x + i));
        acc1 = vfmaq_f32(acc1, h1, vld1q_f32(x + i + 4));
    }

    float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) acc += f16_to_f32(row[i]) * x[i];
    return acc;
#else
    float acc = 0.0f;
    for (uint64_t i = 0; i < n; i++) acc += f16_to_f32(row[i]) * x[i];
    return acc;
#endif
}

static void matvec_f16_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_f16_ctx *ctx = vctx;

    for (uint64_t o = row0; o < row1; o++) {
        const uint16_t *row = ctx->data + o * ctx->in_dim;
        ctx->out[o] = dot_f16_row(row, ctx->x, ctx->in_dim);
    }
}

/* Dense F16 matvec for small control projections such as HC and router heads. */
static void matvec_f16(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    if (w->type != 1 || w->ndim != 2) ds4_die("expected a 2D F16 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    matvec_f16_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .x = x,
        .in_dim = in_dim,
    };

    const uint64_t ops = in_dim * out_dim;
    const uint64_t min_rows = ops >= 262144 ? 1 : 512;
    ds4_parallel_for_min_rows(out_dim, matvec_f16_worker, &ctx, min_rows);
}

static void matvec_f16_serial(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    if (w->type != 1 || w->ndim != 2) ds4_die("expected a 2D F16 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    const uint16_t *data = tensor_data(m, w);
    for (uint64_t o = 0; o < out_dim; o++) {
        out[o] = dot_f16_row(data + o * in_dim, x, in_dim);
    }
}

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t in_dim;
    uint64_t row0;
    uint64_t blocks;
} matvec_q8_0_ctx;

typedef struct {
    float *out0;
    float *out1;
    const uint8_t *data0;
    const uint8_t *data1;
    const int8_t *xq;
    const float *xscale;
    uint64_t in_dim;
    uint64_t blocks;
} matvec_q8_0_pair_ctx;

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t in_dim;
    uint64_t blocks;
    uint64_t rank;
} matvec_q8_0_grouped_ctx;

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t n_tok;
    uint64_t n_groups;
    uint64_t group_dim;
    uint64_t blocks;
    uint64_t rank;
} matmul_q8_0_grouped_batch_ctx;

typedef struct {
    float *out;
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    uint64_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
} matmul_q8_0_batch_ctx;

typedef struct {
    float *out0;
    float *out1;
    const uint8_t *data0;
    const uint8_t *data1;
    const int8_t *xq;
    const float *xscale;
    uint64_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
} matmul_q8_0_pair_batch_ctx;

typedef struct {
    const float *x;
    int8_t *xq;
    float *xscale;
    uint64_t in_dim;
    uint64_t blocks;
} quantize_q8_0_batch_ctx;

static inline int32_t dot_i8_32(const int8_t *a, const int8_t *b, uint64_t n) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (n == 32) {
        int32x4_t acc = vdupq_n_s32(0);
        acc = vdotq_s32(acc, vld1q_s8(a),      vld1q_s8(b));
        acc = vdotq_s32(acc, vld1q_s8(a + 16), vld1q_s8(b + 16));
        return vaddvq_s32(acc);
    }
#endif
    int32_t sum = 0;
    for (uint64_t i = 0; i < n; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

static inline float dot_q8_0_row(
        const uint8_t *row,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t accv0 = vdupq_n_f32(0.0f);
        float32x4_t accv1 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t scale_bits0;
            uint16_t scale_bits1;
            memcpy(&scale_bits0, row + b * 34, sizeof(scale_bits0));
            memcpy(&scale_bits1, row + (b + 1) * 34, sizeof(scale_bits1));

            const int8_t *qs0 = (const int8_t *)(row + b * 34 + 2);
            const int8_t *qs1 = (const int8_t *)(row + (b + 1) * 34 + 2);
            const int8_t *xq0 = xq + b * 32;
            const int8_t *xq1 = xq + (b + 1) * 32;

            int32x4_t dot0 = vdupq_n_s32(0);
            dot0 = vdotq_s32(dot0, vld1q_s8(qs0),      vld1q_s8(xq0));
            dot0 = vdotq_s32(dot0, vld1q_s8(qs0 + 16), vld1q_s8(xq0 + 16));

            int32x4_t dot1 = vdupq_n_s32(0);
            dot1 = vdotq_s32(dot1, vld1q_s8(qs1),      vld1q_s8(xq1));
            dot1 = vdotq_s32(dot1, vld1q_s8(qs1 + 16), vld1q_s8(xq1 + 16));

            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot0), f16_to_f32(scale_bits0) * xscale[b]);
            accv1 = vfmaq_n_f32(accv1, vcvtq_f32_s32(dot1), f16_to_f32(scale_bits1) * xscale[b + 1]);
        }

        if (b < blocks) {
            uint16_t scale_bits;
            memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
            const int8_t *qs = (const int8_t *)(row + b * 34 + 2);
            const int8_t *xqb = xq + b * 32;
            int32x4_t dot = vdupq_n_s32(0);
            dot = vdotq_s32(dot, vld1q_s8(qs),      vld1q_s8(xqb));
            dot = vdotq_s32(dot, vld1q_s8(qs + 16), vld1q_s8(xqb + 16));
            accv0 = vfmaq_n_f32(accv0, vcvtq_f32_s32(dot), f16_to_f32(scale_bits) * xscale[b]);
        }

        return vaddvq_f32(vaddq_f32(accv0, accv1));
    }
#endif

    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
        const int8_t *qs = (const int8_t *)(row + b * 34 + 2);

        const uint64_t i0 = b * 32;
        const uint64_t n = in_dim - i0 < 32 ? in_dim - i0 : 32;
        acc += f16_to_f32(scale_bits) * xscale[b] * (float)dot_i8_32(qs, xq + i0, n);
    }
    return acc;
}

static inline void dot_q8_0_row_2(
        const uint8_t *row,
        const int8_t  *xq0,
        const float   *xscale0,
        const int8_t  *xq1,
        const float   *xscale1,
        uint64_t       in_dim,
        uint64_t       blocks,
        float         *out0,
        float         *out1) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t acc00 = vdupq_n_f32(0.0f);
        float32x4_t acc01 = vdupq_n_f32(0.0f);
        float32x4_t acc10 = vdupq_n_f32(0.0f);
        float32x4_t acc11 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t scale_bits0;
            uint16_t scale_bits1;
            memcpy(&scale_bits0, row + b * 34, sizeof(scale_bits0));
            memcpy(&scale_bits1, row + (b + 1) * 34, sizeof(scale_bits1));

            const int8_t *qs0 = (const int8_t *)(row + b * 34 + 2);
            const int8_t *qs1 = (const int8_t *)(row + (b + 1) * 34 + 2);

            int32x4_t d00 = vdupq_n_s32(0);
            d00 = vdotq_s32(d00, vld1q_s8(qs0),      vld1q_s8(xq0 + b * 32));
            d00 = vdotq_s32(d00, vld1q_s8(qs0 + 16), vld1q_s8(xq0 + b * 32 + 16));
            int32x4_t d01 = vdupq_n_s32(0);
            d01 = vdotq_s32(d01, vld1q_s8(qs1),      vld1q_s8(xq0 + (b + 1) * 32));
            d01 = vdotq_s32(d01, vld1q_s8(qs1 + 16), vld1q_s8(xq0 + (b + 1) * 32 + 16));

            int32x4_t d10 = vdupq_n_s32(0);
            d10 = vdotq_s32(d10, vld1q_s8(qs0),      vld1q_s8(xq1 + b * 32));
            d10 = vdotq_s32(d10, vld1q_s8(qs0 + 16), vld1q_s8(xq1 + b * 32 + 16));
            int32x4_t d11 = vdupq_n_s32(0);
            d11 = vdotq_s32(d11, vld1q_s8(qs1),      vld1q_s8(xq1 + (b + 1) * 32));
            d11 = vdotq_s32(d11, vld1q_s8(qs1 + 16), vld1q_s8(xq1 + (b + 1) * 32 + 16));

            const float s0 = f16_to_f32(scale_bits0);
            const float s1 = f16_to_f32(scale_bits1);
            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d00), s0 * xscale0[b]);
            acc01 = vfmaq_n_f32(acc01, vcvtq_f32_s32(d01), s1 * xscale0[b + 1]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d10), s0 * xscale1[b]);
            acc11 = vfmaq_n_f32(acc11, vcvtq_f32_s32(d11), s1 * xscale1[b + 1]);
        }

        if (b < blocks) {
            uint16_t scale_bits;
            memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
            const int8_t *qs = (const int8_t *)(row + b * 34 + 2);

            int32x4_t d0 = vdupq_n_s32(0);
            d0 = vdotq_s32(d0, vld1q_s8(qs),      vld1q_s8(xq0 + b * 32));
            d0 = vdotq_s32(d0, vld1q_s8(qs + 16), vld1q_s8(xq0 + b * 32 + 16));
            int32x4_t d1 = vdupq_n_s32(0);
            d1 = vdotq_s32(d1, vld1q_s8(qs),      vld1q_s8(xq1 + b * 32));
            d1 = vdotq_s32(d1, vld1q_s8(qs + 16), vld1q_s8(xq1 + b * 32 + 16));

            const float s0 = f16_to_f32(scale_bits);
            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d0), s0 * xscale0[b]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d1), s0 * xscale1[b]);
        }

        *out0 = vaddvq_f32(vaddq_f32(acc00, acc01));
        *out1 = vaddvq_f32(vaddq_f32(acc10, acc11));
        return;
    }
#endif

    *out0 = dot_q8_0_row(row, xq0, xscale0, in_dim, blocks);
    *out1 = dot_q8_0_row(row, xq1, xscale1, in_dim, blocks);
}

static inline DS4_MAYBE_UNUSED void dot_q8_0_row_pair(
        const uint8_t *row0,
        const uint8_t *row1,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks,
        float         *out0,
        float         *out1) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if ((in_dim & 31u) == 0) {
        float32x4_t acc00 = vdupq_n_f32(0.0f);
        float32x4_t acc01 = vdupq_n_f32(0.0f);
        float32x4_t acc10 = vdupq_n_f32(0.0f);
        float32x4_t acc11 = vdupq_n_f32(0.0f);

        uint64_t b = 0;
        for (; b + 1 < blocks; b += 2) {
            uint16_t s00, s01, s10, s11;
            memcpy(&s00, row0 + b * 34, sizeof(s00));
            memcpy(&s01, row0 + (b + 1) * 34, sizeof(s01));
            memcpy(&s10, row1 + b * 34, sizeof(s10));
            memcpy(&s11, row1 + (b + 1) * 34, sizeof(s11));

            const int8_t *xq0 = xq + b * 32;
            const int8_t *xq1 = xq + (b + 1) * 32;
            const int8x16_t xv00 = vld1q_s8(xq0);
            const int8x16_t xv01 = vld1q_s8(xq0 + 16);
            const int8x16_t xv10 = vld1q_s8(xq1);
            const int8x16_t xv11 = vld1q_s8(xq1 + 16);

            const int8_t *q00 = (const int8_t *)(row0 + b * 34 + 2);
            const int8_t *q01 = (const int8_t *)(row0 + (b + 1) * 34 + 2);
            const int8_t *q10 = (const int8_t *)(row1 + b * 34 + 2);
            const int8_t *q11 = (const int8_t *)(row1 + (b + 1) * 34 + 2);

            int32x4_t d00 = vdupq_n_s32(0);
            d00 = vdotq_s32(d00, vld1q_s8(q00),      xv00);
            d00 = vdotq_s32(d00, vld1q_s8(q00 + 16), xv01);
            int32x4_t d01 = vdupq_n_s32(0);
            d01 = vdotq_s32(d01, vld1q_s8(q01),      xv10);
            d01 = vdotq_s32(d01, vld1q_s8(q01 + 16), xv11);
            int32x4_t d10 = vdupq_n_s32(0);
            d10 = vdotq_s32(d10, vld1q_s8(q10),      xv00);
            d10 = vdotq_s32(d10, vld1q_s8(q10 + 16), xv01);
            int32x4_t d11 = vdupq_n_s32(0);
            d11 = vdotq_s32(d11, vld1q_s8(q11),      xv10);
            d11 = vdotq_s32(d11, vld1q_s8(q11 + 16), xv11);

            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d00), f16_to_f32(s00) * xscale[b]);
            acc01 = vfmaq_n_f32(acc01, vcvtq_f32_s32(d01), f16_to_f32(s01) * xscale[b + 1]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d10), f16_to_f32(s10) * xscale[b]);
            acc11 = vfmaq_n_f32(acc11, vcvtq_f32_s32(d11), f16_to_f32(s11) * xscale[b + 1]);
        }

        if (b < blocks) {
            uint16_t s0, s1;
            memcpy(&s0, row0 + b * 34, sizeof(s0));
            memcpy(&s1, row1 + b * 34, sizeof(s1));
            const int8_t *xqb = xq + b * 32;
            const int8x16_t xv0 = vld1q_s8(xqb);
            const int8x16_t xv1 = vld1q_s8(xqb + 16);
            const int8_t *q0 = (const int8_t *)(row0 + b * 34 + 2);
            const int8_t *q1 = (const int8_t *)(row1 + b * 34 + 2);
            int32x4_t d0 = vdupq_n_s32(0);
            d0 = vdotq_s32(d0, vld1q_s8(q0),      xv0);
            d0 = vdotq_s32(d0, vld1q_s8(q0 + 16), xv1);
            int32x4_t d1 = vdupq_n_s32(0);
            d1 = vdotq_s32(d1, vld1q_s8(q1),      xv0);
            d1 = vdotq_s32(d1, vld1q_s8(q1 + 16), xv1);
            acc00 = vfmaq_n_f32(acc00, vcvtq_f32_s32(d0), f16_to_f32(s0) * xscale[b]);
            acc10 = vfmaq_n_f32(acc10, vcvtq_f32_s32(d1), f16_to_f32(s1) * xscale[b]);
        }

        *out0 = vaddvq_f32(vaddq_f32(acc00, acc01));
        *out1 = vaddvq_f32(vaddq_f32(acc10, acc11));
        return;
    }
#endif

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t s0_bits;
        uint16_t s1_bits;
        memcpy(&s0_bits, row0 + b * 34, sizeof(s0_bits));
        memcpy(&s1_bits, row1 + b * 34, sizeof(s1_bits));
        const int8_t *q0 = (const int8_t *)(row0 + b * 34 + 2);
        const int8_t *q1 = (const int8_t *)(row1 + b * 34 + 2);
        const uint64_t i0 = b * 32;
        const uint64_t n = in_dim - i0 < 32 ? in_dim - i0 : 32;
        acc0 += f16_to_f32(s0_bits) * xscale[b] * (float)dot_i8_32(q0, xq + i0, n);
        acc1 += f16_to_f32(s1_bits) * xscale[b] * (float)dot_i8_32(q1, xq + i0, n);
    }
    *out0 = acc0;
    *out1 = acc1;
}

static void quantize_q8_0_activation(const float *x, int8_t *xq, float *scale, uint64_t n) {
    const uint64_t blocks = (n + 31) / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = n - i0 < 32 ? n - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) {
            const float ax = fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scale[b] = d;
        for (uint64_t i = 0; i < bn; i++) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (uint64_t i = bn; i < 32 && i0 + i < blocks * 32; i++) {
            xq[i0 + i] = 0;
        }
    }
}

static void quantize_q8_0_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    quantize_q8_0_batch_ctx *ctx = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        quantize_q8_0_activation(ctx->x + t * ctx->in_dim,
                                 ctx->xq + t * ctx->blocks * 32,
                                 ctx->xscale + t * ctx->blocks,
                                 ctx->in_dim);
    }
}

static void quantize_q8_0_activation_batch(
        const float *x,
        int8_t      *xq,
        float       *xscale,
        uint64_t     n_tok,
        uint64_t     in_dim) {
    quantize_q8_0_batch_ctx ctx = {
        .x = x,
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .blocks = (in_dim + 31) / 32,
    };
    ds4_parallel_for(n_tok, quantize_q8_0_batch_worker, &ctx);
}

static void matvec_q8_0_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_ctx *ctx = vctx;

    for (uint64_t r = r0; r < r1; r++) {
        const uint64_t o = ctx->row0 + r;
        const uint8_t *row = ctx->data + o * ctx->blocks * 34;
        ctx->out[r] = dot_q8_0_row(row, ctx->xq, ctx->xscale, ctx->in_dim, ctx->blocks);
    }
}

static void matvec_q8_0_pair_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_pair_ctx *ctx = vctx;

    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row0 = ctx->data0 + r * ctx->blocks * 34;
        const uint8_t *row1 = ctx->data1 + r * ctx->blocks * 34;
        dot_q8_0_row_pair(row0, row1, ctx->xq, ctx->xscale, ctx->in_dim, ctx->blocks,
                          ctx->out0 + r, ctx->out1 + r);
    }
}

static void matvec_q8_0_grouped_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_grouped_ctx *ctx = vctx;

    for (uint64_t idx = r0; idx < r1; idx++) {
        const uint64_t group = idx / ctx->rank;
        const uint64_t row_in_group = idx - group * ctx->rank;
        const uint64_t tensor_row = group * ctx->rank + row_in_group;
        const uint8_t *row = ctx->data + tensor_row * ctx->blocks * 34;
        const int8_t *xq = ctx->xq + group * ctx->blocks * 32;
        const float *xscale = ctx->xscale + group * ctx->blocks;
        ctx->out[idx] = dot_q8_0_row(row, xq, xscale, ctx->in_dim, ctx->blocks);
    }
}

static void matmul_q8_0_grouped_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matmul_q8_0_grouped_batch_ctx *ctx = vctx;

    for (uint64_t idx = r0; idx < r1; idx++) {
        const uint64_t group = idx / ctx->rank;
        const uint64_t row_in_group = idx - group * ctx->rank;
        const uint64_t tensor_row = group * ctx->rank + row_in_group;
        const uint8_t *row = ctx->data + tensor_row * ctx->blocks * 34;

        uint64_t t = 0;
        for (; t + 1 < ctx->n_tok; t += 2) {
            const uint64_t xbase0 = (t * ctx->n_groups + group) * ctx->blocks;
            const uint64_t xbase1 = ((t + 1) * ctx->n_groups + group) * ctx->blocks;
            dot_q8_0_row_2(row,
                           ctx->xq + xbase0 * 32,
                           ctx->xscale + xbase0,
                           ctx->xq + xbase1 * 32,
                           ctx->xscale + xbase1,
                           ctx->group_dim,
                           ctx->blocks,
                           ctx->out + t * ctx->n_groups * ctx->rank + group * ctx->rank + row_in_group,
                           ctx->out + (t + 1) * ctx->n_groups * ctx->rank + group * ctx->rank + row_in_group);
        }
        for (; t < ctx->n_tok; t++) {
            const uint64_t xbase = (t * ctx->n_groups + group) * ctx->blocks;
            ctx->out[t * ctx->n_groups * ctx->rank + group * ctx->rank + row_in_group] =
                dot_q8_0_row(row,
                             ctx->xq + xbase * 32,
                             ctx->xscale + xbase,
                             ctx->group_dim,
                             ctx->blocks);
        }
    }
}

static void matmul_q8_0_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matmul_q8_0_batch_ctx *ctx = vctx;

    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row = ctx->data + r * ctx->blocks * 34;
        uint64_t t = 0;
        for (; t + 1 < ctx->n_tok; t += 2) {
            dot_q8_0_row_2(row,
                           ctx->xq + t * ctx->blocks * 32,
                           ctx->xscale + t * ctx->blocks,
                           ctx->xq + (t + 1) * ctx->blocks * 32,
                           ctx->xscale + (t + 1) * ctx->blocks,
                           ctx->in_dim,
                           ctx->blocks,
                           ctx->out + t * ctx->out_dim + r,
                           ctx->out + (t + 1) * ctx->out_dim + r);
        }
        for (; t < ctx->n_tok; t++) {
            ctx->out[t * ctx->out_dim + r] =
                dot_q8_0_row(row,
                             ctx->xq + t * ctx->blocks * 32,
                             ctx->xscale + t * ctx->blocks,
                             ctx->in_dim,
                             ctx->blocks);
        }
    }
}

static void matmul_q8_0_pair_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matmul_q8_0_pair_batch_ctx *ctx = vctx;

    for (uint64_t r = r0; r < r1; r++) {
        const uint8_t *row0 = ctx->data0 + r * ctx->blocks * 34;
        const uint8_t *row1 = ctx->data1 + r * ctx->blocks * 34;
        uint64_t t = 0;
        for (; t + 1 < ctx->n_tok; t += 2) {
            const int8_t *xq0 = ctx->xq + t * ctx->blocks * 32;
            const float *xscale0 = ctx->xscale + t * ctx->blocks;
            const int8_t *xq1 = ctx->xq + (t + 1) * ctx->blocks * 32;
            const float *xscale1 = ctx->xscale + (t + 1) * ctx->blocks;
            dot_q8_0_row_2(row0, xq0, xscale0, xq1, xscale1, ctx->in_dim, ctx->blocks,
                           ctx->out0 + t * ctx->out_dim + r,
                           ctx->out0 + (t + 1) * ctx->out_dim + r);
            dot_q8_0_row_2(row1, xq0, xscale0, xq1, xscale1, ctx->in_dim, ctx->blocks,
                           ctx->out1 + t * ctx->out_dim + r,
                           ctx->out1 + (t + 1) * ctx->out_dim + r);
        }
        for (; t < ctx->n_tok; t++) {
            const int8_t *xq = ctx->xq + t * ctx->blocks * 32;
            const float *xscale = ctx->xscale + t * ctx->blocks;
            dot_q8_0_row_pair(row0, row1, xq, xscale, ctx->in_dim, ctx->blocks,
                              ctx->out0 + t * ctx->out_dim + r,
                              ctx->out1 + t * ctx->out_dim + r);
        }
    }
}

/* Multiply selected Q8_0 rows by an activation that has already been quantized
 * once.  This avoids repeated activation quantization for paired projections. */
static void matvec_q8_0_rows_prequant(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          row0,
        uint64_t          n_rows) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    if (row0 > out_dim || n_rows > out_dim - row0) ds4_die("Q8_0 row range is outside tensor");
    const uint64_t ctx_blocks = (in_dim + 31) / 32;

    matvec_q8_0_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .row0 = row0,
        .blocks = ctx_blocks,
    };
    ds4_parallel_for(n_rows, matvec_q8_0_worker, &ctx);
}

static DS4_MAYBE_UNUSED void matvec_q8_0_prequant(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const int8_t    * xq,
        const float     * xscale) {
    matvec_q8_0_rows_prequant(out, m, w, xq, xscale, 0, w->dim[1]);
}

static void matvec_q8_0_3d_slice_prequant(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          slice) {
    if (w->type != DS4_TENSOR_Q8_0 || w->ndim != 3) ds4_die("expected a 3D Q8_0 tensor");
    if (slice >= w->dim[2]) ds4_die("Q8_0 slice is outside tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->dim[1];
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t slice_bytes = out_dim * blocks * 34;
    const uint8_t *data = (const uint8_t *)tensor_data(m, w) + slice * slice_bytes;

    matvec_q8_0_ctx ctx = {
        .out = out,
        .data = data,
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .row0 = 0,
        .blocks = blocks,
    };
    ds4_parallel_for(out_dim, matvec_q8_0_worker, &ctx);
}

static DS4_MAYBE_UNUSED void matvec_q8_0_3d_slice(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint64_t          slice) {
    if (w->type != DS4_TENSOR_Q8_0 || w->ndim != 3) ds4_die("expected a 3D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)blocks * 32);
    float *xscale = xmalloc((size_t)blocks * sizeof(xscale[0]));

    quantize_q8_0_activation(x, xq, xscale, in_dim);
    matvec_q8_0_3d_slice_prequant(out, m, w, xq, xscale, slice);

    free(xscale);
    free(xq);
}

/* Compute two Q8_0 projections from the same input, used by gate/up and
 * compressor kv/score pairs. */
static void matvec_q8_0_pair_prequant(
        float           * out0,
        float           * out1,
        const ds4_model * m,
        const ds4_tensor * w0,
        const ds4_tensor * w1,
        const int8_t    * xq,
        const float     * xscale) {
    if (w0->type != 8 || w1->type != 8 || w0->ndim != 2 || w1->ndim != 2) {
        ds4_die("expected two 2D Q8_0 tensors");
    }
    if (w0->dim[0] != w1->dim[0] || w0->dim[1] != w1->dim[1]) {
        ds4_die("paired Q8_0 tensors do not have the same shape");
    }

    const uint64_t in_dim = w0->dim[0];
    matvec_q8_0_pair_ctx ctx = {
        .out0 = out0,
        .out1 = out1,
        .data0 = tensor_data(m, w0),
        .data1 = tensor_data(m, w1),
        .xq = xq,
        .xscale = xscale,
        .in_dim = in_dim,
        .blocks = (in_dim + 31) / 32,
    };
    ds4_parallel_for(w0->dim[1], matvec_q8_0_pair_worker, &ctx);
}

static void matmul_q8_0_batch_prequant(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          n_tok) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");

    matmul_q8_0_batch_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .n_tok = n_tok,
        .in_dim = w->dim[0],
        .out_dim = w->dim[1],
        .blocks = (w->dim[0] + 31) / 32,
    };
    ds4_parallel_for(ctx.out_dim, matmul_q8_0_batch_worker, &ctx);
}

static void matmul_q8_0_pair_batch_prequant(
        float           * out0,
        float           * out1,
        const ds4_model * m,
        const ds4_tensor * w0,
        const ds4_tensor * w1,
        const int8_t    * xq,
        const float     * xscale,
        uint64_t          n_tok) {
    if (w0->type != 8 || w1->type != 8 || w0->ndim != 2 || w1->ndim != 2) {
        ds4_die("expected two 2D Q8_0 tensors");
    }
    if (w0->dim[0] != w1->dim[0] || w0->dim[1] != w1->dim[1]) {
        ds4_die("paired Q8_0 tensors do not have the same shape");
    }

    matmul_q8_0_pair_batch_ctx ctx = {
        .out0 = out0,
        .out1 = out1,
        .data0 = tensor_data(m, w0),
        .data1 = tensor_data(m, w1),
        .xq = xq,
        .xscale = xscale,
        .n_tok = n_tok,
        .in_dim = w0->dim[0],
        .out_dim = w0->dim[1],
        .blocks = (w0->dim[0] + 31) / 32,
    };
    ds4_parallel_for(ctx.out_dim, matmul_q8_0_pair_batch_worker, &ctx);
}

/* Batched Q8_0 matmul for prefill: quantize all token activations, then scan
 * weight rows once per output channel. */
static void matmul_q8_0_batch(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint64_t          n_tok) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)n_tok * blocks * 32);
    float *xscale = xmalloc((size_t)n_tok * blocks * sizeof(xscale[0]));

    quantize_q8_0_activation_batch(x, xq, xscale, n_tok, in_dim);
    matmul_q8_0_batch_prequant(out, m, w, xq, xscale, n_tok);

    free(xscale);
    free(xq);
}

static void matmul_q8_0_pair_batch(
        float           * out0,
        float           * out1,
        const ds4_model * m,
        const ds4_tensor * w0,
        const ds4_tensor * w1,
        const float     * x,
        uint64_t          n_tok) {
    if (w0->type != 8 || w1->type != 8 || w0->ndim != 2 || w1->ndim != 2) {
        ds4_die("expected two 2D Q8_0 tensors");
    }
    if (w0->dim[0] != w1->dim[0] || w0->dim[1] != w1->dim[1]) {
        ds4_die("paired Q8_0 tensors do not have the same shape");
    }

    const uint64_t in_dim = w0->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)n_tok * blocks * 32);
    float *xscale = xmalloc((size_t)n_tok * blocks * sizeof(xscale[0]));

    quantize_q8_0_activation_batch(x, xq, xscale, n_tok, in_dim);
    matmul_q8_0_pair_batch_prequant(out0, out1, m, w0, w1, xq, xscale, n_tok);

    free(xscale);
    free(xq);
}

static void matvec_q8_0_rows(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint64_t          row0,
        uint64_t          n_rows) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");

    const uint64_t in_dim = w->dim[0];
    const uint64_t ctx_blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)ctx_blocks * 32);
    float *xscale = xmalloc((size_t)ctx_blocks * sizeof(xscale[0]));

    quantize_q8_0_activation(x, xq, xscale, in_dim);
    matvec_q8_0_rows_prequant(out, m, w, xq, xscale, row0, n_rows);

    free(xscale);
    free(xq);
}

/* Single-token Q8_0 matvec, used heavily in decode. */
static void matvec_q8_0(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    matvec_q8_0_rows(out, m, w, x, 0, w->dim[1]);
}

static inline float dot_q8_0_row_f32_ref(
        const uint8_t *row,
        const float   *x,
        uint64_t       in_dim,
        uint64_t       blocks) {
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
        const int8_t *qs = (const int8_t *)(row + b * 34 + 2);
        const float d = f16_to_f32(scale_bits);
        const uint64_t i0 = b * 32;
        const uint64_t n = in_dim - i0 < 32 ? in_dim - i0 : 32;
        for (uint64_t i = 0; i < n; i++) {
            acc += d * (float)qs[i] * x[i0 + i];
        }
    }
    return acc;
}

typedef struct {
    float *out;
    const uint8_t *data;
    const float *x;
    uint64_t in_dim;
    uint64_t blocks;
} matvec_q8_0_f32_ref_ctx;

static void matvec_q8_0_f32_ref_worker(void *vctx, uint64_t r0, uint64_t r1) {
    matvec_q8_0_f32_ref_ctx *ctx = vctx;
    const uint64_t row_bytes = ctx->blocks * 34;
    for (uint64_t r = r0; r < r1; r++) {
        ctx->out[r] = dot_q8_0_row_f32_ref(ctx->data + r * row_bytes,
                                           ctx->x,
                                           ctx->in_dim,
                                           ctx->blocks);
    }
}

static void matvec_q8_0_f32_ref(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const float      *x) {
    if (w->type != DS4_TENSOR_Q8_0 || w->ndim < 2 || w->dim[0] == 0) {
        ds4_die("expected a Q8_0 tensor with matrix rows");
    }
    matvec_q8_0_f32_ref_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .x = x,
        .in_dim = w->dim[0],
        .blocks = (w->dim[0] + 31) / 32,
    };
    ds4_parallel_for(w->elements / w->dim[0], matvec_q8_0_f32_ref_worker, &ctx);
}

static void matvec_any(float *out, const ds4_model *m, const ds4_tensor *w, const float *x);

/* Decode scratch owns this temporary activation quantization so generation
 * can assert that the hot path performs no malloc. */
static void cpu_decode_quantize_q8_0(
        ds4_cpu_decode_scratch * scratch,
        const float            * x,
        uint64_t                 in_dim) {
    if (in_dim > scratch->q8_cap) ds4_die("CPU decode Q8_0 scratch buffer is too small");
    quantize_q8_0_activation(x, scratch->q8_xq, scratch->q8_xscale, in_dim);
}

static void matvec_q8_0_decode_scratch(
        float                  * out,
        const ds4_model        * m,
        const ds4_tensor       * w,
        const float            * x,
        ds4_cpu_decode_scratch * scratch) {
    cpu_decode_quantize_q8_0(scratch, x, w->dim[0]);
    matvec_q8_0_prequant(out, m, w, scratch->q8_xq, scratch->q8_xscale);
}

static void matvec_q8_0_pair_decode_scratch(
        float                  * out0,
        float                  * out1,
        const ds4_model        * m,
        const ds4_tensor       * w0,
        const ds4_tensor       * w1,
        const float            * x,
        ds4_cpu_decode_scratch * scratch) {
    cpu_decode_quantize_q8_0(scratch, x, w0->dim[0]);
    matvec_q8_0_pair_prequant(out0, out1, m, w0, w1, scratch->q8_xq, scratch->q8_xscale);
}

static void matvec_any_decode_scratch(
        float                  * out,
        const ds4_model        * m,
        const ds4_tensor       * w,
        const float            * x,
        ds4_cpu_decode_scratch * scratch) {
    if (w->type == 8) {
        matvec_q8_0_decode_scratch(out, m, w, x, scratch);
    } else {
        matvec_any(out, m, w, x);
    }
}

static void matvec_q8_0_grouped_rows(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint32_t          n_groups,
        uint64_t          group_dim,
        uint64_t          rank) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");
    if (w->dim[0] != group_dim || w->dim[1] < (uint64_t)n_groups * rank) {
        ds4_die("grouped Q8_0 tensor has an unexpected layout");
    }

    const uint64_t blocks = (group_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)n_groups * blocks * 32);
    float *xscale = xmalloc((size_t)n_groups * blocks * sizeof(xscale[0]));

    for (uint32_t g = 0; g < n_groups; g++) {
        quantize_q8_0_activation(x + (uint64_t)g * group_dim,
                                 xq + (uint64_t)g * blocks * 32,
                                 xscale + (uint64_t)g * blocks,
                                 group_dim);
    }

    matvec_q8_0_grouped_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .in_dim = group_dim,
        .blocks = blocks,
        .rank = rank,
    };
    ds4_parallel_for((uint64_t)n_groups * rank, matvec_q8_0_grouped_worker, &ctx);

    free(xscale);
    free(xq);
}

static void matvec_q8_0_grouped_rows_decode_scratch(
        float                  * out,
        const ds4_model        * m,
        const ds4_tensor       * w,
        const float            * x,
        uint32_t                 n_groups,
        uint64_t                 group_dim,
        uint64_t                 rank,
        ds4_cpu_decode_scratch * scratch) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");
    if (w->dim[0] != group_dim || w->dim[1] < (uint64_t)n_groups * rank) {
        ds4_die("grouped Q8_0 tensor has an unexpected layout");
    }
    if ((uint64_t)n_groups * group_dim > scratch->q8_cap) {
        ds4_die("CPU decode grouped Q8_0 scratch buffer is too small");
    }

    const uint64_t blocks = (group_dim + 31) / 32;
    for (uint32_t g = 0; g < n_groups; g++) {
        quantize_q8_0_activation(x + (uint64_t)g * group_dim,
                                 scratch->q8_xq + (uint64_t)g * blocks * 32,
                                 scratch->q8_xscale + (uint64_t)g * blocks,
                                 group_dim);
    }

    matvec_q8_0_grouped_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = scratch->q8_xq,
        .xscale = scratch->q8_xscale,
        .in_dim = group_dim,
        .blocks = blocks,
        .rank = rank,
    };
    ds4_parallel_for((uint64_t)n_groups * rank, matvec_q8_0_grouped_worker, &ctx);
}

static void matmul_q8_0_grouped_batch(
        float           * out,
        const ds4_model * m,
        const ds4_tensor * w,
        const float     * x,
        uint64_t          n_tok,
        uint32_t          n_groups,
        uint64_t          group_dim,
        uint64_t          rank) {
    if (w->type != 8 || w->ndim != 2) ds4_die("expected a 2D Q8_0 tensor");
    if (w->dim[0] != group_dim || w->dim[1] < (uint64_t)n_groups * rank) {
        ds4_die("grouped Q8_0 tensor has an unexpected layout");
    }

    const uint64_t blocks = (group_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)n_tok * n_groups * blocks * 32);
    float *xscale = xmalloc((size_t)n_tok * n_groups * blocks * sizeof(xscale[0]));

    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint32_t g = 0; g < n_groups; g++) {
            const uint64_t xbase = (t * n_groups + g) * blocks;
            quantize_q8_0_activation(x + t * n_groups * group_dim + (uint64_t)g * group_dim,
                                     xq + xbase * 32,
                                     xscale + xbase,
                                     group_dim);
        }
    }

    matmul_q8_0_grouped_batch_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .xq = xq,
        .xscale = xscale,
        .n_tok = n_tok,
        .n_groups = n_groups,
        .group_dim = group_dim,
        .blocks = blocks,
        .rank = rank,
    };
    ds4_parallel_for((uint64_t)n_groups * rank, matmul_q8_0_grouped_batch_worker, &ctx);

    free(xscale);
    free(xq);
}

typedef struct {
    float *out;
    const float *data;
    const float *x;
    uint64_t in_dim;
} matvec_f32_ctx;

static void matvec_f32_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_f32_ctx *ctx = vctx;

    for (uint64_t o = row0; o < row1; o++) {
        double acc = 0.0;
        const float *row = ctx->data + o * ctx->in_dim;
        for (uint64_t i = 0; i < ctx->in_dim; i++) {
            acc += (double)row[i] * ctx->x[i];
        }
        ctx->out[o] = (float)acc;
    }
}

static void matvec_f32(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    if (w->type != 0 || w->ndim != 2) ds4_die("expected a 2D F32 tensor");

    matvec_f32_ctx ctx = {
        .out = out,
        .data = tensor_data(m, w),
        .x = x,
        .in_dim = w->dim[0],
    };
    ds4_parallel_for(w->dim[1], matvec_f32_worker, &ctx);
}

/* Dispatch for dense F32/F16/Q8_0 tensors used by auxiliary projections. */
static void matvec_any(float *out, const ds4_model *m, const ds4_tensor *w, const float *x) {
    switch (w->type) {
    case 0: matvec_f32(out, m, w, x); break;
    case 1: matvec_f16(out, m, w, x); break;
    case 8: matvec_q8_0(out, m, w, x); break;
    default:
        ds4_die("unsupported tensor type for dense matvec");
    }
}

static float tensor_1d_value(const ds4_model *m, const ds4_tensor *t, uint64_t i) {
    if (i >= t->elements) ds4_die("tensor scalar index is out of bounds");
    if (t->type == 0) {
        const float *p = tensor_data(m, t);
        return p[i];
    }
    if (t->type == 1) {
        const uint16_t *p = tensor_data(m, t);
        return f16_to_f32(p[i]);
    }
    ds4_die("unsupported tensor scalar type");
    return 0.0f;
}

static float tensor_2d_value(const ds4_model *m, const ds4_tensor *t, uint64_t x, uint64_t y) {
    if (t->ndim != 2 || x >= t->dim[0] || y >= t->dim[1]) {
        ds4_die("tensor 2D index is out of bounds");
    }
    return tensor_1d_value(m, t, y * t->dim[0] + x);
}

/* Locate one expert's 2D matrix inside a 3D GGUF expert tensor. */
static const uint8_t *tensor_expert_bytes(
        const ds4_model  *m,
        const ds4_tensor *w,
        uint32_t          expert,
        uint64_t         *in_dim,
        uint64_t         *out_dim,
        uint64_t         *row_bytes) {
    if (w->ndim != 3) ds4_die("expected a 3D expert tensor");
    if (expert >= w->dim[2]) ds4_die("expert id is outside expert tensor");

    *in_dim = w->dim[0];
    *out_dim = w->dim[1];

    const gguf_type_info *info = tensor_type(w->type);
    if (!info || info->block_elems == 0) ds4_die("unsupported expert tensor type");
    const uint64_t blocks = (*in_dim + info->block_elems - 1) / info->block_elems;
    *row_bytes = blocks * info->block_bytes;

    const uint64_t expert_bytes = *out_dim * *row_bytes;
    return (const uint8_t *)tensor_data(m, w) + (uint64_t)expert * expert_bytes;
}

typedef struct {
    float *out0;
    float *out1;
    const uint8_t *base0;
    const uint8_t *base1;
    const block_q8_K *xq;
    uint64_t in_dim;
    uint64_t row_bytes0;
    uint64_t row_bytes1;
} matvec_iq2_xxs_pair_ctx;

static void matvec_iq2_xxs_pair_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_iq2_xxs_pair_ctx *ctx = vctx;
    for (uint64_t row = row0; row < row1; row++) {
        const block_iq2_xxs *br0 = (const block_iq2_xxs *)(ctx->base0 + row * ctx->row_bytes0);
        const block_iq2_xxs *br1 = (const block_iq2_xxs *)(ctx->base1 + row * ctx->row_bytes1);
        ds4_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &ctx->out0[row], &ctx->out1[row], br0, br1, ctx->xq);
    }
}

/* Project one routed expert's gate and up matrices.  Both are IQ2_XXS and
 * share the same Q8_K activation. */
static void matvec_iq2_xxs_expert_pair_prequant(
        float            *out0,
        float            *out1,
        const ds4_model  *m,
        const ds4_tensor *w0,
        const ds4_tensor *w1,
        const block_q8_K *xq,
        uint32_t          expert) {
    if (w0->type != 16 || w1->type != 16) ds4_die("expected IQ2_XXS expert tensors");

    uint64_t in_dim0, out_dim0, row_bytes0;
    uint64_t in_dim1, out_dim1, row_bytes1;
    const uint8_t *base0 = tensor_expert_bytes(m, w0, expert, &in_dim0, &out_dim0, &row_bytes0);
    const uint8_t *base1 = tensor_expert_bytes(m, w1, expert, &in_dim1, &out_dim1, &row_bytes1);
    if (in_dim0 != in_dim1 || out_dim0 != out_dim1) ds4_die("paired IQ2_XXS expert tensors do not match");
    if (in_dim0 % QK_K != 0) ds4_die("IQ2_XXS expert row is not QK_K aligned");

    matvec_iq2_xxs_pair_ctx ctx = {
        .out0 = out0,
        .out1 = out1,
        .base0 = base0,
        .base1 = base1,
        .xq = xq,
        .in_dim = in_dim0,
        .row_bytes0 = row_bytes0,
        .row_bytes1 = row_bytes1,
    };
    ds4_parallel_for(out_dim0, matvec_iq2_xxs_pair_worker, &ctx);
}

static float silu(float x);

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[DS4_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_iq2_xxs_mid_ctx;

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    const int8_t *xq;
    const float *xscale;
    float expert_weight[DS4_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q8_0_mid_ctx;

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[DS4_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q8_k_mid_ctx;

static void matvec_iq2_xxs_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_iq2_xxs_mid_ctx *ctx = vctx;

    for (uint64_t idx = row0; idx < row1; idx++) {
        const int slot = (int)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        float gate = 0.0f;
        float up = 0.0f;

        const block_iq2_xxs *gate_row = (const block_iq2_xxs *)(ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot]);
        const block_iq2_xxs *up_row = (const block_iq2_xxs *)(ctx->up_base[slot] + row * ctx->up_row_bytes[slot]);
        ds4_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &gate, &up, gate_row, up_row, ctx->xq);

        if (ctx->clamp > 1.0e-6f) {
            if (gate > ctx->clamp) gate = ctx->clamp;
            if (up > ctx->clamp) up = ctx->clamp;
            if (up < -ctx->clamp) up = -ctx->clamp;
        }
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}

static void matvec_q8_0_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q8_0_mid_ctx *ctx = vctx;

    for (uint64_t idx = row0; idx < row1; idx++) {
        const int slot = (int)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        float gate = 0.0f;
        float up = 0.0f;

        const uint8_t *gate_row = ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot];
        const uint8_t *up_row = ctx->up_base[slot] + row * ctx->up_row_bytes[slot];
        dot_q8_0_row_pair(gate_row, up_row, ctx->xq, ctx->xscale,
                          ctx->in_dim, ctx->blocks, &gate, &up);

        if (ctx->clamp > 1.0e-6f) {
            if (gate > ctx->clamp) gate = ctx->clamp;
            if (up > ctx->clamp) up = ctx->clamp;
            if (up < -ctx->clamp) up = -ctx->clamp;
        }
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}

static void matvec_q8_k_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q8_k_mid_ctx *ctx = vctx;

    for (uint64_t idx = row0; idx < row1; idx++) {
        const int slot = (int)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        float gate = 0.0f;
        float up = 0.0f;

        const block_q8_K *gate_row = (const block_q8_K *)(ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot]);
        const block_q8_K *up_row = (const block_q8_K *)(ctx->up_base[slot] + row * ctx->up_row_bytes[slot]);
        ds4_vec_dot_q8_K_pair_q8_K((int)ctx->in_dim, &gate, &up, gate_row, up_row, ctx->xq);

        if (ctx->clamp > 1.0e-6f) {
            if (gate > ctx->clamp) gate = ctx->clamp;
            if (up > ctx->clamp) up = ctx->clamp;
            if (up < -ctx->clamp) up = -ctx->clamp;
        }
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}

/* Build all selected expert hidden vectors: IQ2_XXS gate/up, clamp, SwiGLU,
 * and router weight.  The down projection runs later on the quantized mids. */
static void matvec_iq2_xxs_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type != 16 || up_w->type != 16) ds4_die("expected IQ2_XXS expert tensors");
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_iq2_xxs_mid_ctx ctx = {
        .mid = mid,
        .xq = xq,
        .clamp = clamp,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
                                               &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
        ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
                                             &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
        if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
            ds4_die("paired IQ2_XXS expert tensors do not match");
        }
        if (i == 0) {
            in_dim0 = gate_in_dim;
            out_dim0 = gate_out_dim;
        } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
            ds4_die("IQ2_XXS expert tensors do not share a layout");
        }
        ctx.expert_weight[i] = expert_weight[i];
    }
    if (in_dim0 % QK_K != 0) ds4_die("IQ2_XXS expert row is not QK_K aligned");

    ctx.in_dim = in_dim0;
    ctx.out_dim = out_dim0;
    ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_iq2_xxs_mid_worker, &ctx);
}

static DS4_MAYBE_UNUSED void matvec_q8_0_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const int8_t     *xq,
        const float      *xscale,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type != DS4_TENSOR_Q8_0 || up_w->type != DS4_TENSOR_Q8_0) {
        ds4_die("expected Q8_0 expert tensors");
    }
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_q8_0_mid_ctx ctx = {
        .mid = mid,
        .xq = xq,
        .xscale = xscale,
        .clamp = clamp,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
                                               &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
        ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
                                             &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
        if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
            ds4_die("paired Q8_0 expert tensors do not match");
        }
        if (i == 0) {
            in_dim0 = gate_in_dim;
            out_dim0 = gate_out_dim;
        } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
            ds4_die("Q8_0 expert tensors do not share a layout");
        }
        ctx.expert_weight[i] = expert_weight[i];
    }
    if ((in_dim0 % 32u) != 0) ds4_die("Q8_0 expert row is not QK8_0 aligned");

    ctx.in_dim = in_dim0;
    ctx.out_dim = out_dim0;
    ctx.blocks = in_dim0 / 32u;
    ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_q8_0_mid_worker, &ctx);
}

static DS4_MAYBE_UNUSED void matvec_q8_k_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type != DS4_TENSOR_Q8_K || up_w->type != DS4_TENSOR_Q8_K) {
        ds4_die("expected Q8_K expert tensors");
    }
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_q8_k_mid_ctx ctx = {
        .mid = mid,
        .xq = xq,
        .clamp = clamp,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
                                               &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
        ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
                                             &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
        if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
            ds4_die("paired Q8_K expert tensors do not match");
        }
        if (i == 0) {
            in_dim0 = gate_in_dim;
            out_dim0 = gate_out_dim;
        } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
            ds4_die("Q8_K expert tensors do not share a layout");
        }
        ctx.expert_weight[i] = expert_weight[i];
    }
    if (in_dim0 % QK_K != 0) ds4_die("Q8_K expert row is not QK_K aligned");

    ctx.in_dim = in_dim0;
    ctx.out_dim = out_dim0;
    ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_q8_k_mid_worker, &ctx);
}

typedef struct {
    float *out;
    const uint8_t *base;
    const block_q8_K *xq;
    uint64_t in_dim;
    uint64_t row_bytes;
} matvec_q2_k_ctx;

static void matvec_q2_k_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q2_k_ctx *ctx = vctx;
    for (uint64_t row = row0; row < row1; row++) {
        const block_q2_K *br = (const block_q2_K *)(ctx->base + row * ctx->row_bytes);
        ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &ctx->out[row], br, ctx->xq);
    }
}

/* Single expert Q2_K down projection, kept mostly for tracing and diagnostics. */
static void matvec_q2_k_expert(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const float      *x,
        uint32_t          expert) {
    if (w->type != 10) ds4_die("expected a Q2_K expert tensor");

    uint64_t in_dim, out_dim, row_bytes;
    const uint8_t *base = tensor_expert_bytes(m, w, expert, &in_dim, &out_dim, &row_bytes);
    if (in_dim % QK_K != 0) ds4_die("Q2_K expert row is not QK_K aligned");

    block_q8_K *xq = xmalloc((size_t)(in_dim / QK_K) * sizeof(xq[0]));
    ds4_quantize_row_q8_K(x, xq, (int64_t)in_dim);

    matvec_q2_k_ctx ctx = {
        .out = out,
        .base = base,
        .xq = xq,
        .in_dim = in_dim,
        .row_bytes = row_bytes,
    };
    ds4_parallel_for(out_dim, matvec_q2_k_worker, &ctx);

    free(xq);
}

typedef struct {
    float *out;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q2_k_accum_ctx;

static void matvec_q2_k_accum_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q2_k_accum_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        float acc = 0.0f;
        for (int i = 0; i < ctx->n_expert; i++) {
            float v = 0.0f;
            const block_q2_K *br = (const block_q2_K *)(ctx->base[i] + row * ctx->row_bytes[i]);
            ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &v, br, ctx->xq[i]);
            acc += v;
        }
        ctx->out[row] = acc;
    }
}

/* Accumulate all selected experts' Q2_K down projections directly into the
 * 4096-wide MoE output. */
static void matvec_q2_k_experts_accum_prequant(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert) {
    if (w->type != 10) ds4_die("expected a Q2_K expert tensor");
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];

    for (int i = 0; i < n_expert; i++) {
        uint64_t in_dim, out_dim;
        base[i] = tensor_expert_bytes(m, w, (uint32_t)selected[i], &in_dim, &out_dim, &row_bytes[i]);
        if (i == 0) {
            in_dim0 = in_dim;
            out_dim0 = out_dim;
        } else if (in_dim != in_dim0 || out_dim != out_dim0) {
            ds4_die("Q2_K expert tensors do not share a layout");
        }
    }
    if (in_dim0 % QK_K != 0) ds4_die("Q2_K expert row is not QK_K aligned");

    const uint64_t n_blocks = in_dim0 / QK_K;
    matvec_q2_k_accum_ctx ctx = {
        .out = out,
        .in_dim = in_dim0,
        .n_expert = n_expert,
    };
    for (int i = 0; i < n_expert; i++) {
        ctx.base[i] = base[i];
        ctx.row_bytes[i] = row_bytes[i];
        ctx.xq[i] = xq + (uint64_t)i * n_blocks;
    }

    ds4_parallel_for(out_dim0, matvec_q2_k_accum_worker, &ctx);
}

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[DS4_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q2_k_mid_ctx;

static void matvec_q2_k_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q2_k_mid_ctx *ctx = vctx;

    for (uint64_t idx = row0; idx < row1; idx++) {
        const int slot = (int)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        float gate = 0.0f;
        float up = 0.0f;

        const block_q2_K *gate_row = (const block_q2_K *)(ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot]);
        ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &gate, gate_row, ctx->xq);

        const block_q2_K *up_row = (const block_q2_K *)(ctx->up_base[slot] + row * ctx->up_row_bytes[slot]);
        ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &up, up_row, ctx->xq);

        if (ctx->clamp > 1.0e-6f) {
            if (gate > ctx->clamp) gate = ctx->clamp;
            if (up > ctx->clamp) up = ctx->clamp;
            if (up < -ctx->clamp) up = -ctx->clamp;
        }
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}

static void matvec_q2_k_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type != DS4_TENSOR_Q2_K || up_w->type != DS4_TENSOR_Q2_K) {
        ds4_die("expected Q2_K expert tensors");
    }
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_q2_k_mid_ctx ctx = {
        .mid = mid,
        .xq = xq,
        .clamp = clamp,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
                                               &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
        ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
                                             &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
        if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
            ds4_die("paired Q2_K expert tensors do not match");
        }
        if (i == 0) {
            in_dim0 = gate_in_dim;
            out_dim0 = gate_out_dim;
        } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
            ds4_die("Q2_K expert tensors do not share a layout");
        }
        ctx.expert_weight[i] = expert_weight[i];
    }
    if (in_dim0 % QK_K != 0) ds4_die("Q2_K expert row is not QK_K aligned");

    ctx.in_dim = in_dim0;
    ctx.out_dim = out_dim0;
    ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_q2_k_mid_worker, &ctx);
}

typedef struct {
    float *out;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    const int8_t *xq[DS4_MAX_EXPERT_USED];
    const float *xscale[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t blocks;
    int n_expert;
} matvec_q8_0_accum_ctx;

static void matvec_q8_0_accum_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q8_0_accum_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        float acc = 0.0f;
        for (int i = 0; i < ctx->n_expert; i++) {
            const uint8_t *br = ctx->base[i] + row * ctx->row_bytes[i];
            acc += dot_q8_0_row(br, ctx->xq[i], ctx->xscale[i],
                                ctx->in_dim, ctx->blocks);
        }
        ctx->out[row] = acc;
    }
}

static void matvec_q8_0_experts_accum_prequant(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const int8_t     *xq,
        const float      *xscale,
        const int        *selected,
        int               n_expert) {
    if (w->type != DS4_TENSOR_Q8_0) ds4_die("expected a Q8_0 expert tensor");
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_q8_0_accum_ctx ctx = {
        .out = out,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t in_dim, out_dim;
        ctx.base[i] = tensor_expert_bytes(m, w, (uint32_t)selected[i],
                                          &in_dim, &out_dim, &ctx.row_bytes[i]);
        if (i == 0) {
            in_dim0 = in_dim;
            out_dim0 = out_dim;
        } else if (in_dim != in_dim0 || out_dim != out_dim0) {
            ds4_die("Q8_0 expert tensors do not share a layout");
        }
    }
    if ((in_dim0 % 32u) != 0) ds4_die("Q8_0 expert row is not QK8_0 aligned");

    const uint64_t blocks0 = in_dim0 / 32u;
    ctx.in_dim = in_dim0;
    ctx.blocks = blocks0;
    for (int i = 0; i < n_expert; i++) {
        ctx.xq[i] = xq + (uint64_t)i * blocks0 * 32u;
        ctx.xscale[i] = xscale + (uint64_t)i * blocks0;
    }

    ds4_parallel_for(out_dim0, matvec_q8_0_accum_worker, &ctx);
}

typedef struct {
    float *out;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q8_k_accum_ctx;

static void matvec_q8_k_accum_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q8_k_accum_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        float acc = 0.0f;
        for (int i = 0; i < ctx->n_expert; i++) {
            float v = 0.0f;
            const block_q8_K *br = (const block_q8_K *)(ctx->base[i] + row * ctx->row_bytes[i]);
            ds4_vec_dot_q8_K_q8_K((int)ctx->in_dim, &v, br, ctx->xq[i]);
            acc += v;
        }
        ctx->out[row] = acc;
    }
}

static void matvec_q8_k_experts_accum_prequant(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert) {
    if (w->type != DS4_TENSOR_Q8_K) ds4_die("expected a Q8_K expert tensor");
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_q8_k_accum_ctx ctx = {
        .out = out,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t in_dim, out_dim;
        ctx.base[i] = tensor_expert_bytes(m, w, (uint32_t)selected[i],
                                          &in_dim, &out_dim, &ctx.row_bytes[i]);
        if (i == 0) {
            in_dim0 = in_dim;
            out_dim0 = out_dim;
        } else if (in_dim != in_dim0 || out_dim != out_dim0) {
            ds4_die("Q8_K expert tensors do not share a layout");
        }
    }
    if (in_dim0 % QK_K != 0) ds4_die("Q8_K expert row is not QK_K aligned");

    const uint64_t n_blocks = in_dim0 / QK_K;
    ctx.in_dim = in_dim0;
    for (int i = 0; i < n_expert; i++) {
        ctx.xq[i] = xq + (uint64_t)i * n_blocks;
    }

    ds4_parallel_for(out_dim0, matvec_q8_k_accum_worker, &ctx);
}

typedef struct {
    float *out;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_iq2_xxs_accum_ctx;

static void matvec_iq2_xxs_accum_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_iq2_xxs_accum_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        float acc = 0.0f;
        for (int i = 0; i < ctx->n_expert; i++) {
            float v = 0.0f;
            const block_iq2_xxs *br = (const block_iq2_xxs *)(ctx->base[i] + row * ctx->row_bytes[i]);
            ds4_vec_dot_iq2_xxs_q8_K((int)ctx->in_dim, &v, br, ctx->xq[i]);
            acc += v;
        }
        ctx->out[row] = acc;
    }
}

static void matvec_iq2_xxs_experts_accum_prequant(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert) {
    if (w->type != DS4_TENSOR_IQ2_XXS) ds4_die("expected an IQ2_XXS expert tensor");
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];

    for (int i = 0; i < n_expert; i++) {
        uint64_t in_dim, out_dim;
        base[i] = tensor_expert_bytes(m, w, (uint32_t)selected[i], &in_dim, &out_dim, &row_bytes[i]);
        if (i == 0) {
            in_dim0 = in_dim;
            out_dim0 = out_dim;
        } else if (in_dim != in_dim0 || out_dim != out_dim0) {
            ds4_die("IQ2_XXS expert tensors do not share a layout");
        }
    }
    if (in_dim0 % QK_K != 0) ds4_die("IQ2_XXS expert row is not QK_K aligned");

    const uint64_t n_blocks = in_dim0 / QK_K;
    matvec_iq2_xxs_accum_ctx ctx = {
        .out = out,
        .in_dim = in_dim0,
        .n_expert = n_expert,
    };
    for (int i = 0; i < n_expert; i++) {
        ctx.base[i] = base[i];
        ctx.row_bytes[i] = row_bytes[i];
        ctx.xq[i] = xq + (uint64_t)i * n_blocks;
    }

    ds4_parallel_for(out_dim0, matvec_iq2_xxs_accum_worker, &ctx);
}

typedef struct {
    uint32_t token;
    uint32_t slot;
} ds4_expert_pair;

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT];
    const uint8_t *up_base[DS4_MAX_EXPERT];
    const block_q8_K *xq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT];
    uint64_t up_row_bytes[DS4_MAX_EXPERT];
    uint64_t xq_blocks;
} matvec_q2_k_batch_mid_ctx;

static void matvec_q2_k_batch_mid_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_q2_k_batch_mid_ctx *ctx = vctx;

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];

        const block_q2_K *gate_row = (const block_q2_K *)(ctx->gate_base[expert] + row * ctx->gate_row_bytes[expert]);
        const block_q2_K *up_row = (const block_q2_K *)(ctx->up_base[expert] + row * ctx->up_row_bytes[expert]);

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const ds4_expert_pair pair = ctx->pairs[pair_id];
            const block_q8_K *xq = ctx->xq + (uint64_t)pair.token * ctx->xq_blocks;
            float gate = 0.0f;
            float up = 0.0f;

            ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &gate, gate_row, xq);
            ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &up, up_row, xq);

            if (ctx->clamp > 1.0e-6f) {
                if (gate > ctx->clamp) gate = ctx->clamp;
                if (up > ctx->clamp) up = ctx->clamp;
                if (up < -ctx->clamp) up = -ctx->clamp;
            }

            ctx->mid[(uint64_t)pair_id * ctx->out_dim + row] = silu(gate) * up * ctx->pair_weight[pair_id];
        }
    }
}

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT];
    const uint8_t *up_base[DS4_MAX_EXPERT];
    const block_q8_K *xq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT];
    uint64_t up_row_bytes[DS4_MAX_EXPERT];
    uint64_t xq_blocks;
} matvec_iq2_xxs_batch_mid_ctx;

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT];
    const uint8_t *up_base[DS4_MAX_EXPERT];
    const block_q8_K *xq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t xq_blocks;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT];
    uint64_t up_row_bytes[DS4_MAX_EXPERT];
} matvec_q8_k_batch_mid_ctx;

static void matvec_iq2_xxs_batch_mid_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_iq2_xxs_batch_mid_ctx *ctx = vctx;

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];

        const block_iq2_xxs *gate_row = (const block_iq2_xxs *)(ctx->gate_base[expert] + row * ctx->gate_row_bytes[expert]);
        const block_iq2_xxs *up_row = (const block_iq2_xxs *)(ctx->up_base[expert] + row * ctx->up_row_bytes[expert]);

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const ds4_expert_pair pair = ctx->pairs[pair_id];
            const block_q8_K *xq = ctx->xq + (uint64_t)pair.token * ctx->xq_blocks;
            float gate = 0.0f;
            float up = 0.0f;

            ds4_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &gate, &up, gate_row, up_row, xq);

            if (ctx->clamp > 1.0e-6f) {
                if (gate > ctx->clamp) gate = ctx->clamp;
                if (up > ctx->clamp) up = ctx->clamp;
                if (up < -ctx->clamp) up = -ctx->clamp;
            }

            ctx->mid[(uint64_t)pair_id * ctx->out_dim + row] = silu(gate) * up * ctx->pair_weight[pair_id];
        }
    }
}

static void matvec_q8_k_batch_mid_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_q8_k_batch_mid_ctx *ctx = vctx;

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];

        const block_q8_K *gate_row = (const block_q8_K *)(ctx->gate_base[expert] + row * ctx->gate_row_bytes[expert]);
        const block_q8_K *up_row = (const block_q8_K *)(ctx->up_base[expert] + row * ctx->up_row_bytes[expert]);

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const ds4_expert_pair pair = ctx->pairs[pair_id];
            const block_q8_K *xq = ctx->xq + (uint64_t)pair.token * ctx->xq_blocks;
            float gate = 0.0f;
            float up = 0.0f;

            ds4_vec_dot_q8_K_pair_q8_K((int)ctx->in_dim, &gate, &up, gate_row, up_row, xq);

            if (ctx->clamp > 1.0e-6f) {
                if (gate > ctx->clamp) gate = ctx->clamp;
                if (up > ctx->clamp) up = ctx->clamp;
                if (up < -ctx->clamp) up = -ctx->clamp;
            }

            ctx->mid[(uint64_t)pair_id * ctx->out_dim + row] = silu(gate) * up * ctx->pair_weight[pair_id];
        }
    }
}

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT];
    const uint8_t *up_base[DS4_MAX_EXPERT];
    const int8_t *xq;
    const float *xscale;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT];
    uint64_t up_row_bytes[DS4_MAX_EXPERT];
} matvec_q8_0_batch_mid_ctx;

static void matvec_q8_0_batch_mid_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_q8_0_batch_mid_ctx *ctx = vctx;

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];

        const uint8_t *gate_row = ctx->gate_base[expert] + row * ctx->gate_row_bytes[expert];
        const uint8_t *up_row = ctx->up_base[expert] + row * ctx->up_row_bytes[expert];

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const ds4_expert_pair pair = ctx->pairs[pair_id];
            float gate = 0.0f;
            float up = 0.0f;

            dot_q8_0_row_pair(gate_row, up_row,
                              ctx->xq + (uint64_t)pair.token * ctx->blocks * 32u,
                              ctx->xscale + (uint64_t)pair.token * ctx->blocks,
                              ctx->in_dim, ctx->blocks, &gate, &up);

            if (ctx->clamp > 1.0e-6f) {
                if (gate > ctx->clamp) gate = ctx->clamp;
                if (up > ctx->clamp) up = ctx->clamp;
                if (up < -ctx->clamp) up = -ctx->clamp;
            }

            ctx->mid[(uint64_t)pair_id * ctx->out_dim + row] = silu(gate) * up * ctx->pair_weight[pair_id];
        }
    }
}

typedef struct {
    const float *mid;
    block_q8_K *midq;
    uint64_t down_in_dim;
    uint64_t down_blocks;
} quantize_mid_pairs_ctx;

static void quantize_mid_pairs_worker(void *vctx, uint64_t p0, uint64_t p1) {
    quantize_mid_pairs_ctx *ctx = vctx;
    for (uint64_t p = p0; p < p1; p++) {
        ds4_quantize_row_q8_K(ctx->mid + p * ctx->down_in_dim,
                              ctx->midq + p * ctx->down_blocks,
                              (int64_t)ctx->down_in_dim);
    }
}

typedef struct {
    float *down_pair;
    const uint8_t *base[DS4_MAX_EXPERT];
    const block_q8_K *midq;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_q2_k_batch_down_ctx;

static DS4_MAYBE_UNUSED void matvec_q2_k_batch_down_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_q2_k_batch_down_ctx *ctx = vctx;

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];
        const block_q2_K *br = (const block_q2_K *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
            ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim,
                                  ctx->down_pair + (uint64_t)pair_id * ctx->out_dim + row,
                                  br, xq);
        }
    }
}

typedef struct {
    float *moe;
    const uint8_t *base[DS4_MAX_EXPERT];
    const block_q8_K *midq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_q2_k_batch_accum_rows_ctx;

static void matvec_q2_k_batch_accum_rows_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q2_k_batch_accum_rows_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            ctx->moe[(uint64_t)t * ctx->out_dim + row] = 0.0f;
        }

        for (uint32_t ai = 0; ai < ctx->n_active; ai++) {
            const uint32_t expert = ctx->active_expert[ai];
            const uint32_t begin = ctx->expert_offset[expert];
            const uint32_t end = ctx->expert_offset[expert + 1];
            const block_q2_K *br = (const block_q2_K *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

            for (uint32_t i = begin; i < end; i++) {
                const uint32_t pair_id = ctx->pair_ids[i];
                const ds4_expert_pair pair = ctx->pairs[pair_id];
                const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
                float v = 0.0f;

                ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &v, br, xq);
                ctx->moe[(uint64_t)pair.token * ctx->out_dim + row] += v;
            }
        }
    }
}

