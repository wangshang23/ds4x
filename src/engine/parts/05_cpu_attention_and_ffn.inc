/* =========================================================================
 * Attention Projections, RoPE, and Attention Output.
 * =========================================================================
 *
 * This block performs the attention half of a transformer layer: HC pre,
 * attention RMSNorm, Q and KV projections, layer-specific RoPE, sink-aware
 * attention over raw and compressed KV rows, and the grouped LoRA output
 * projection back to embedding width.
 */

/* Q projection is low-rank: Q8_0 into the model-specific LoRA-Q rank,
 * RMSNorm, then Q8_0 back to all attention heads. */
static void layer_q_projection_normed_one(
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * norm,
        float             * q) {
    const uint32_t q_rank = DS4_N_LORA_Q;
    float *qr = xmalloc((size_t)q_rank * sizeof(qr[0]));
    float *qr_norm = xmalloc((size_t)q_rank * sizeof(qr_norm[0]));

    const float *q_a_norm = tensor_data(model, layer->attn_q_a_norm);

    matvec_q8_0(qr, model, layer->attn_q_a, norm);
    rms_norm_weight(qr_norm, qr, q_a_norm, q_rank, DS4_RMS_EPS);
    matvec_q8_0(q, model, layer->attn_q_b, qr_norm);
    head_rms_norm_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS);

    free(qr_norm);
    free(qr);
}

static void layer_q_projection_with_lora_one(
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * norm,
        float             * q,
        float             * qr_norm) {
    const uint32_t q_rank = DS4_N_LORA_Q;
    float *qr = xmalloc((size_t)q_rank * sizeof(qr[0]));
    const float *q_a_norm = tensor_data(model, layer->attn_q_a_norm);

    matvec_q8_0(qr, model, layer->attn_q_a, norm);
    rms_norm_weight(qr_norm, qr, q_a_norm, q_rank, DS4_RMS_EPS);
    matvec_q8_0(q, model, layer->attn_q_b, qr_norm);
    head_rms_norm_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS);

    free(qr);
}

/* KV projection has one KV head of width 512, followed by a learned RMSNorm. */
static void layer_kv_projection_normed_one(
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * normed,
        float             * kv) {
    float *raw = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(raw[0]));

    const float *kv_norm = tensor_data(model, layer->attn_kv_a_norm);

    matvec_q8_0(raw, model, layer->attn_kv, normed);
    rms_norm_weight(kv, raw, kv_norm, DS4_N_HEAD_DIM, DS4_RMS_EPS);

    free(raw);
}

static void layer_q_projection_with_lora_one_decode_scratch(
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * norm,
        float                   * q,
        float                   * qr_norm,
        ds4_cpu_decode_scratch  * scratch) {
    const float *q_a_norm = tensor_data(model, layer->attn_q_a_norm);

    matvec_q8_0_decode_scratch(scratch->qr, model, layer->attn_q_a, norm, scratch);
    rms_norm_weight(qr_norm, scratch->qr, q_a_norm, DS4_N_LORA_Q, DS4_RMS_EPS);
    matvec_q8_0_decode_scratch(q, model, layer->attn_q_b, qr_norm, scratch);
    head_rms_norm_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_RMS_EPS);
}

static void layer_kv_projection_normed_one_decode_scratch(
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * normed,
        float                   * kv,
        ds4_cpu_decode_scratch  * scratch) {
    const float *kv_norm = tensor_data(model, layer->attn_kv_a_norm);

    matvec_q8_0_decode_scratch(scratch->kv_raw, model, layer->attn_kv, normed, scratch);
    rms_norm_weight(kv, scratch->kv_raw, kv_norm, DS4_N_HEAD_DIM, DS4_RMS_EPS);
}

static float rope_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static float rope_yarn_corr_dim(int n_dims, uint64_t n_ctx_orig, float n_rot, float base) {
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * (float)M_PI)) / (2.0f * logf(base));
}

static void rope_yarn_corr_dims(int n_dims, uint64_t n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
    const float start = floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    const float end = ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = fmaxf(0.0f, start);
    dims[1] = fminf((float)(n_dims - 1), end);
}

/* Apply DS4 RoPE only to the tail of each head.  Compressed layers use the
 * long-context frequency base and scale; inverse mode rotates attention output
 * back before the grouped output projection. */
static void rope_tail_ext_inplace(
        float    * x,
        uint32_t   n_head,
        uint32_t   head_dim,
        uint32_t   n_rot,
        uint32_t   pos,
        uint64_t   n_ctx_orig,
        float      freq_base,
        float      freq_scale,
        float      ext_factor,
        float      attn_factor,
        float      beta_fast,
        float      beta_slow,
        bool       inverse) {
    const uint32_t n_nope = head_dim - n_rot;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const float sin_sign = inverse ? -1.0f : 1.0f;
    float corr_dims[2] = { 0.0f, 0.0f };
    if (ext_factor != 0.0f) {
        rope_yarn_corr_dims((int)n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    }

    for (uint32_t h = 0; h < n_head; h++) {
        float *tail = x + (uint64_t)h * head_dim + n_nope;
        float theta_extrap = (float)pos;

        for (uint32_t i = 0; i < n_rot; i += 2) {
            const float theta_interp = freq_scale * theta_extrap;
            float theta = theta_interp;
            float mscale = attn_factor;

            if (ext_factor != 0.0f) {
                const float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i) * ext_factor;
                theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }

            const float c = cosf(theta) * mscale;
            const float s = sin_sign * sinf(theta) * mscale;
            const float x0 = tail[i + 0];
            const float x1 = tail[i + 1];

            tail[i + 0] = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;

            theta_extrap *= theta_scale;
        }
    }
}

/* Dense layers and compressed layers use different RoPE bases. */
static float layer_rope_freq_base(uint32_t il) {
    return ds4_layer_compress_ratio(il) != 0 && DS4_COMPRESS_ROPE_FREQ_BASE > 0.0f
        ? DS4_COMPRESS_ROPE_FREQ_BASE
        : DS4_ROPE_FREQ_BASE;
}

static float layer_rope_freq_scale(uint32_t il) {
    if (ds4_layer_compress_ratio(il) == 0 || DS4_ROPE_SCALE_FACTOR <= 0.0f) {
        return 1.0f;
    }
    return 1.0f / DS4_ROPE_SCALE_FACTOR;
}

static void rope_tail_layer_inplace(
        float            * x,
        uint32_t           n_head,
        uint32_t           head_dim,
        uint32_t           n_rot,
        uint32_t           pos,
        uint32_t           il,
        bool               inverse) {
    const bool compressed = ds4_layer_compress_ratio(il) != 0;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        /*
         * This YaRN helper applies magnitude scaling internally. DeepSeek V4
         * reference RoPE uses interpolation without that magnitude change, so
         * pass the inverse factor here and let the helper cancel itself out.
         */
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    rope_tail_ext_inplace(x, n_head, head_dim, n_rot, pos,
                          compressed ? DS4_ROPE_ORIG_CTX : 0,
                          freq_base,
                          freq_scale,
                          ext_factor,
                          attn_factor,
                          DS4_ROPE_YARN_BETA_FAST,
                          DS4_ROPE_YARN_BETA_SLOW,
                          inverse);
}

typedef struct {
    float            *x;
    uint64_t          stride;
    uint32_t          n_head;
    uint32_t          head_dim;
    uint32_t          n_rot;
    uint32_t          pos0;
    uint32_t          il;
    bool              inverse;
} rope_tail_batch_ctx;

static void rope_tail_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    rope_tail_batch_ctx *ctx = vctx;
    for (uint64_t tt = t0; tt < t1; tt++) {
        rope_tail_layer_inplace(ctx->x + tt * ctx->stride,
                                ctx->n_head,
                                ctx->head_dim,
                                ctx->n_rot,
                                ctx->pos0 + (uint32_t)tt,
                                ctx->il,
                                ctx->inverse);
    }
}

static void rope_tail_layer_batch_inplace(
        float            *x,
        uint64_t          stride,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          il,
        bool              inverse,
        uint32_t          n_tok) {
    rope_tail_batch_ctx ctx = {
        .x = x,
        .stride = stride,
        .n_head = n_head,
        .head_dim = head_dim,
        .n_rot = n_rot,
        .pos0 = pos0,
        .il = il,
        .inverse = inverse,
    };
    ds4_parallel_for_min_rows(n_tok, rope_tail_batch_worker, &ctx, 1);
}

static inline float dot_f32(const float *a, const float *b, uint32_t n) {
#if defined(__ARM_NEON)
    uint32_t i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) acc += a[i] * b[i];
    return acc;
#else
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
#endif
}

static inline void axpy_f32(float *y, const float *x, float a, uint32_t n) {
#if defined(__ARM_NEON)
    uint32_t i = 0;
    const float32x4_t av = vdupq_n_f32(a);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y + i,     vfmaq_f32(vld1q_f32(y + i),     av, vld1q_f32(x + i)));
        vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4), av, vld1q_f32(x + i + 4)));
    }
    for (; i < n; i++) y[i] += a * x[i];
#else
    for (uint32_t i = 0; i < n; i++) y[i] += a * x[i];
#endif
}

static inline void scale_f32(float *x, float a, uint32_t n) {
#if defined(__ARM_NEON)
    uint32_t i = 0;
    const float32x4_t av = vdupq_n_f32(a);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(x + i,     vmulq_f32(vld1q_f32(x + i),     av));
        vst1q_f32(x + i + 4, vmulq_f32(vld1q_f32(x + i + 4), av));
    }
    for (; i < n; i++) x[i] *= a;
#else
    for (uint32_t i = 0; i < n; i++) x[i] *= a;
#endif
}

static float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = expf(x);
        return e / (1.0f + e);
    }
}

/* Sink-aware attention over a set of KV rows.  The learned sink logit is part
 * of the softmax denominator but contributes no value vector. */
static void layer_attention_rows_one(
        float             * out_heads,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * q,
        const float       * kv_rows,
        uint32_t            n_kv) {
    const float *sinks = tensor_data(model, layer->attn_sinks);
    const float kq_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    float score_stack[512];
    float *score = n_kv <= 512 ? score_stack : xmalloc((size_t)n_kv * sizeof(score[0]));

    for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
        const float *qh = q + (uint64_t)h * DS4_N_HEAD_DIM;

        float max_score = sinks[h];
        for (uint32_t r = 0; r < n_kv; r++) {
            const float *kv = kv_rows + (uint64_t)r * DS4_N_HEAD_DIM;
            score[r] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[r] > max_score) max_score = score[r];
        }

        float *oh = out_heads + (uint64_t)h * DS4_N_HEAD_DIM;
        memset(oh, 0, (size_t)DS4_N_HEAD_DIM * sizeof(oh[0]));

        float denom = expf(sinks[h] - max_score);
        for (uint32_t r = 0; r < n_kv; r++) {
            const float weight = expf(score[r] - max_score);
            const float *kv = kv_rows + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }

        const float inv = 1.0f / denom;
        scale_f32(oh, inv, DS4_N_HEAD_DIM);
    }

    if (score != score_stack) free(score);
}

static void layer_attention_one(
        float             * out_heads,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * q,
        const float       * kv) {
    layer_attention_rows_one(out_heads, model, layer, q, kv, 1);
}

/* Attention output projection is grouped: each group first maps its heads to
 * a 1024-rank low vector, then all groups are projected back to 4096. */
static void layer_grouped_out_one(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * heads) {
    const uint32_t n_groups = 8;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = 1024;

    float *low = xcalloc((size_t)n_groups * rank, sizeof(low[0]));

    matvec_q8_0_grouped_rows(low, model, layer->attn_output_a, heads, n_groups, group_dim, rank);

    matvec_q8_0(out, model, layer->attn_output_b, low);
    free(low);
}

static void layer_grouped_out_one_decode_scratch(
        float                  * out,
        const ds4_model        * model,
        const ds4_layer_weights * layer,
        const float            * heads,
        ds4_cpu_decode_scratch * scratch) {
    const uint32_t n_groups = 8;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = 1024;

    memset(scratch->attn_low, 0, (size_t)n_groups * rank * sizeof(scratch->attn_low[0]));
    matvec_q8_0_grouped_rows_decode_scratch(scratch->attn_low, model, layer->attn_output_a,
                                            heads, n_groups, group_dim, rank, scratch);
    matvec_q8_0_decode_scratch(out, model, layer->attn_output_b, scratch->attn_low, scratch);
}

static void layer_grouped_out_batch(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * heads,
        uint32_t            n_tok) {
    const uint32_t n_groups = 8;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = 1024;

    float *low = xcalloc((size_t)n_tok * n_groups * rank, sizeof(low[0]));

    matmul_q8_0_grouped_batch(low, model, layer->attn_output_a, heads,
                              n_tok, n_groups, group_dim, rank);
    matmul_q8_0_batch(out, model, layer->attn_output_b, low, n_tok);

    free(low);
}

/* =========================================================================
 * Mixture-of-Experts FFN.
 * =========================================================================
 *
 * This is the FFN half of each layer.  It includes the shared expert, routed
 * expert selection, IQ2_XXS gate/up projections, SwiGLU, Q2_K down projection,
 * and the HC post step that returns the result to four-stream state.
 */

static float silu(float x) {
    return x * sigmoid_stable(x);
}

static float softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void swiglu(float *out, const float *gate, const float *up, uint64_t n, float clamp) {
    for (uint64_t i = 0; i < n; i++) {
        float g = gate[i];
        float u = up[i];
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        out[i] = silu(g) * u;
    }
}

/* The shared expert is a normal Q8_0 SwiGLU MLP that runs for every token. */
static void layer_shared_ffn_one(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x) {
    float *gate = xmalloc((size_t)DS4_N_FF_EXP * sizeof(gate[0]));
    float *up = xmalloc((size_t)DS4_N_FF_EXP * sizeof(up[0]));
    float *mid = xmalloc((size_t)DS4_N_FF_EXP * sizeof(mid[0]));
    const uint64_t in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    int8_t *xq = xmalloc((size_t)blocks * 32);
    float *xscale = xmalloc((size_t)blocks * sizeof(xscale[0]));

    if (layer->ffn_up_shexp->type != 8 ||
        layer->ffn_gate_shexp->type != 8 ||
        layer->ffn_up_shexp->dim[0] != in_dim) {
        ds4_die("shared expert gate/up tensors do not share a Q8_0 input layout");
    }

    quantize_q8_0_activation(x, xq, xscale, in_dim);
    matvec_q8_0_pair_prequant(gate, up, model,
                              layer->ffn_gate_shexp,
                              layer->ffn_up_shexp,
                              xq, xscale);
    swiglu(mid, gate, up, DS4_N_FF_EXP, DS4_SWIGLU_CLAMP_EXP);
    matvec_q8_0(out, model, layer->ffn_down_shexp, mid);

    free(xscale);
    free(xq);
    free(mid);
    free(up);
    free(gate);
}

static void layer_shared_ffn_one_decode_scratch(
        float                  * out,
        const ds4_model        * model,
        const ds4_layer_weights * layer,
        const float            * x,
        ds4_cpu_decode_scratch * scratch) {
    const uint64_t in_dim = layer->ffn_gate_shexp->dim[0];
    if (layer->ffn_up_shexp->type != 8 ||
        layer->ffn_gate_shexp->type != 8 ||
        layer->ffn_up_shexp->dim[0] != in_dim) {
        ds4_die("shared expert gate/up tensors do not share a Q8_0 input layout");
    }

    matvec_q8_0_pair_decode_scratch(scratch->shared_gate,
                                    scratch->shared_up,
                                    model,
                                    layer->ffn_gate_shexp,
                                    layer->ffn_up_shexp,
                                    x,
                                    scratch);
    swiglu(scratch->shared_mid, scratch->shared_gate, scratch->shared_up, DS4_N_FF_EXP,
           DS4_SWIGLU_CLAMP_EXP);
    matvec_q8_0_decode_scratch(out, model, layer->ffn_down_shexp, scratch->shared_mid, scratch);
}

typedef struct {
    float *mid;
    const float *gate;
    const float *up;
    uint64_t n;
    float clamp;
} swiglu_batch_ctx;

static void swiglu_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    swiglu_batch_ctx *ctx = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        swiglu(ctx->mid + t * ctx->n,
               ctx->gate + t * ctx->n,
               ctx->up + t * ctx->n,
               ctx->n,
               ctx->clamp);
    }
}

static void layer_shared_ffn_batch(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x,
        uint32_t            n_tok) {
    const uint64_t in_dim = layer->ffn_gate_shexp->dim[0];
    const uint64_t hidden = layer->ffn_gate_shexp->dim[1];

    if (layer->ffn_up_shexp->type != 8 ||
        layer->ffn_gate_shexp->type != 8 ||
        layer->ffn_down_shexp->type != 8 ||
        layer->ffn_up_shexp->dim[0] != in_dim ||
        layer->ffn_up_shexp->dim[1] != hidden ||
        layer->ffn_down_shexp->dim[0] != hidden) {
        ds4_die("shared expert tensors do not share the expected Q8_0 layout");
    }

    float *gate = xmalloc((size_t)n_tok * hidden * sizeof(gate[0]));
    float *up = xmalloc((size_t)n_tok * hidden * sizeof(up[0]));
    float *mid = xmalloc((size_t)n_tok * hidden * sizeof(mid[0]));

    matmul_q8_0_pair_batch(gate, up, model,
                           layer->ffn_gate_shexp,
                           layer->ffn_up_shexp,
                           x,
                           n_tok);

    swiglu_batch_ctx swiglu_ctx = {
        .mid = mid,
        .gate = gate,
        .up = up,
        .n = hidden,
        .clamp = DS4_SWIGLU_CLAMP_EXP,
    };
    ds4_parallel_for(n_tok, swiglu_batch_worker, &swiglu_ctx);

    matmul_q8_0_batch(out, model, layer->ffn_down_shexp, mid, n_tok);

    free(mid);
    free(up);
    free(gate);
}

/* Early DS4 layers use token-id hash routing instead of top-k routing. */
static void layer_hash_selected_experts(
        int                    selected[DS4_MAX_EXPERT_USED],
        const ds4_model       *model,
        const ds4_layer_weights *layer,
        int                    token) {
    ds4_tensor *t = layer->ffn_gate_tid2eid;
    if (!t) ds4_die("hash routing table is missing for this layer");
    if (t->type != 26 || t->ndim != 2 || t->dim[0] != DS4_N_EXPERT_USED) {
        ds4_die("ffn_gate_tid2eid.weight has an unexpected layout");
    }
    if (token < 0 || (uint64_t)token >= t->dim[1]) {
        ds4_die("token id is outside the hash routing table");
    }

    const int32_t *table = tensor_data(model, t);
    const int32_t *row = table + (uint64_t)token * DS4_N_EXPERT_USED;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) selected[i] = row[i];
}

/* Router scores use sqrt(softplus(logit)); normalization happens only after
 * the six selected experts are known. */
static void layer_router_probs_one(
        float             probs[DS4_MAX_EXPERT],
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x) {
    float logits[DS4_MAX_EXPERT];

    matvec_any(logits, model, layer->ffn_gate_inp, x);
    for (uint32_t i = 0; i < DS4_N_EXPERT; i++) {
        probs[i] = sqrtf(softplus_stable(logits[i]));
    }
}

static void layer_hash_router_weights_from_probs(
        float             weights_out[DS4_MAX_EXPERT_USED],
        const float       probs[DS4_MAX_EXPERT],
        const int          selected[DS4_MAX_EXPERT_USED]) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        if (selected[i] < 0 || (uint32_t)selected[i] >= DS4_N_EXPERT) ds4_die("hash-selected expert is outside router range");
        weights_out[i] = probs[selected[i]];
        sum += weights_out[i];
    }

    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        weights_out[i] = weights_out[i] / sum * DS4_EXPERT_WEIGHT_SCALE;
    }
}

static void layer_hash_router_weights_one(
        float             weights_out[DS4_MAX_EXPERT_USED],
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x,
        const int          selected[DS4_MAX_EXPERT_USED]) {
    float probs[DS4_MAX_EXPERT];

    layer_router_probs_one(probs, model, layer, x);
    layer_hash_router_weights_from_probs(weights_out, probs, selected);
}

static void topk_desc(const float *score, int n, int k, int *idx) {
    for (int i = 0; i < k; i++) idx[i] = -1;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (idx[j] < 0 || score[i] > score[idx[j]]) {
                for (int m = k - 1; m > j; m--) idx[m] = idx[m - 1];
                idx[j] = i;
                break;
            }
        }
    }
}

/* Later layers choose the six experts by biased top-k, but weight them using
 * the unbiased router probabilities. */
static void layer_topk_selected_experts_from_probs(
        int                    selected[DS4_MAX_EXPERT_USED],
        float                  expert_weight[DS4_MAX_EXPERT_USED],
        const ds4_model       *model,
        const ds4_layer_weights *layer,
        const float           probs[DS4_MAX_EXPERT]);

static void layer_topk_selected_experts(
        int                    selected[DS4_MAX_EXPERT_USED],
        float                  expert_weight[DS4_MAX_EXPERT_USED],
        const ds4_model       *model,
        const ds4_layer_weights *layer,
        const float           *x) {
    float probs[DS4_MAX_EXPERT] = {0};

    layer_router_probs_one(probs, model, layer, x);
    layer_topk_selected_experts_from_probs(selected, expert_weight, model, layer, probs);
}

static void layer_topk_selected_experts_from_probs(
        int                    selected[DS4_MAX_EXPERT_USED],
        float                  expert_weight[DS4_MAX_EXPERT_USED],
        const ds4_model       *model,
        const ds4_layer_weights *layer,
        const float           probs[DS4_MAX_EXPERT]) {
    float selection[DS4_MAX_EXPERT];

    memcpy(selection, probs, sizeof(selection));

    if (layer->ffn_exp_probs_b) {
        const float *bias = tensor_data(model, layer->ffn_exp_probs_b);
        for (uint32_t i = 0; i < DS4_N_EXPERT; i++) selection[i] += bias[i];
    }

    topk_desc(selection, (int)DS4_N_EXPERT, (int)DS4_N_EXPERT_USED, selected);

    float sum = 0.0f;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        expert_weight[i] = probs[selected[i]];
        sum += expert_weight[i];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        expert_weight[i] = expert_weight[i] / sum * DS4_EXPERT_WEIGHT_SCALE;
    }
}

static void print_vec_stats(const char *name, const float *x, uint64_t n);

/* Single-token routed MoE.  It selects six experts, runs IQ2_XXS gate/up,
 * applies SwiGLU and router weights, then accumulates Q2_K down projections. */
static void layer_routed_moe_one(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x,
        uint32_t            il,
        int                 token,
        float               clamp,
        bool                trace) {
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];
    float *gate = trace ? xmalloc((size_t)DS4_N_FF_EXP * sizeof(gate[0])) : NULL;
    float *up = trace ? xmalloc((size_t)DS4_N_FF_EXP * sizeof(up[0])) : NULL;
    float *mid = trace ? xmalloc((size_t)DS4_N_FF_EXP * sizeof(mid[0])) : NULL;
    float *mid_all = trace ? NULL : xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(mid_all[0]));
    float *down = trace ? xmalloc((size_t)DS4_N_EMBD * sizeof(down[0])) : NULL;
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const bool routed_q8_0 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q8_0;
    const bool routed_q8_k =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q8_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q8_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q8_K;
    if (routed_q8_0) {
        if (trace) ds4_die("Q8_0 routed trace mode is not supported");
        if ((expert_in_dim % 32u) != 0) ds4_die("Q8_0 expert input is not QK8_0 aligned");
        if (down_in_dim != DS4_N_FF_EXP || (down_in_dim % 32u) != 0) {
            ds4_die("Q8_0 expert input has an unexpected layout");
        }
    } else {
        if (routed_q8_k && trace) ds4_die("Q8_K routed trace mode is not supported");
        if (expert_in_dim % QK_K != 0) ds4_die("routed expert input is not QK_K aligned");
        if (down_in_dim != DS4_N_FF_EXP || down_in_dim % QK_K != 0) {
            ds4_die("routed expert down input has an unexpected layout");
        }
    }
    block_q8_K *xq = routed_q8_0 ? NULL : xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(xq[0]));
    block_q8_K *midq = (trace || routed_q8_0) ? NULL :
        xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(midq[0]));

    memset(out, 0, (size_t)DS4_N_EMBD * sizeof(out[0]));
    if (!routed_q8_0) {
        ds4_quantize_row_q8_K(x, xq, (int64_t)expert_in_dim);
    }

    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, x, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, x);
    }

    if (routed_q8_0) {
        const uint64_t x_blocks = expert_in_dim / 32u;
        int8_t *xq8 = xmalloc((size_t)x_blocks * 32u);
        float *xscale8 = xmalloc((size_t)x_blocks * sizeof(float));
        quantize_q8_0_activation(x, xq8, xscale8, expert_in_dim);
        matvec_q8_0_experts_mid_prequant(mid_all, model,
                                         layer->ffn_gate_exps,
                                         layer->ffn_up_exps,
                                         xq8, xscale8, selected,
                                         expert_weight,
                                         DS4_N_EXPERT_USED, clamp);

        const uint64_t mid_blocks = down_in_dim / 32u;
        int8_t *midq8 = xmalloc((size_t)DS4_N_EXPERT_USED * mid_blocks * 32u);
        float *midscale8 = xmalloc((size_t)DS4_N_EXPERT_USED * mid_blocks * sizeof(float));
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            quantize_q8_0_activation(mid_all + (uint64_t)i * down_in_dim,
                                     midq8 + (uint64_t)i * mid_blocks * 32u,
                                     midscale8 + (uint64_t)i * mid_blocks,
                                     down_in_dim);
        }
        matvec_q8_0_experts_accum_prequant(out, model, layer->ffn_down_exps,
                                           midq8, midscale8, selected,
                                           DS4_N_EXPERT_USED);
        free(midscale8);
        free(midq8);
        free(xscale8);
        free(xq8);
    } else if (routed_q8_k) {
        matvec_q8_k_experts_mid_prequant(mid_all, model,
                                         layer->ffn_gate_exps,
                                         layer->ffn_up_exps,
                                         xq, selected, expert_weight,
                                         DS4_N_EXPERT_USED, clamp);
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            ds4_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
                                  midq + (uint64_t)i * (down_in_dim / QK_K),
                                  (int64_t)down_in_dim);
        }
        matvec_q8_k_experts_accum_prequant(out, model, layer->ffn_down_exps,
                                           midq, selected, DS4_N_EXPERT_USED);
    } else if (!trace) {
        matvec_experts_mid_prequant(mid_all, model,
                                    layer->ffn_gate_exps,
                                    layer->ffn_up_exps,
                                    xq,
                                    selected,
                                    expert_weight,
                                    DS4_N_EXPERT_USED,
                                    clamp);
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            ds4_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
                                  midq + (uint64_t)i * (down_in_dim / QK_K),
                                  (int64_t)down_in_dim);
        }
        matvec_experts_down_accum_prequant(out, model, layer->ffn_down_exps, midq, selected, DS4_N_EXPERT_USED);
    } else {
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            const uint32_t expert = (uint32_t)selected[i];

            matvec_expert_pair_prequant(gate, up, model,
                                        layer->ffn_gate_exps,
                                        layer->ffn_up_exps,
                                        xq,
                                        expert);
            char name[64];
            snprintf(name, sizeof(name), "blk.%u expert %u gate", il, expert);
            print_vec_stats(name, gate, DS4_N_FF_EXP);
            snprintf(name, sizeof(name), "blk.%u expert %u up", il, expert);
            print_vec_stats(name, up, DS4_N_FF_EXP);

            /*
             * DeepSeek V4 clamps routed expert gate/up values before SwiGLU and
             * applies the router weight before the down projection.
             */
            const float limit = clamp;
            for (uint32_t j = 0; j < DS4_N_FF_EXP; j++) {
                if (limit > 1.0e-6f) {
                    if (gate[j] > limit) gate[j] = limit;
                    if (up[j] > limit) up[j] = limit;
                    if (up[j] < -limit) up[j] = -limit;
                }
                mid[j] = silu(gate[j]) * up[j] * expert_weight[i];
            }

            snprintf(name, sizeof(name), "blk.%u expert %u mid", il, expert);
            print_vec_stats(name, mid, DS4_N_FF_EXP);

            matvec_expert_down(down, model, layer->ffn_down_exps, mid, expert);
            snprintf(name, sizeof(name), "blk.%u expert %u down", il, expert);
            print_vec_stats(name, down, DS4_N_EMBD);
            for (uint32_t j = 0; j < DS4_N_EMBD; j++) out[j] += down[j];
        }
    }

    free(midq);
    free(xq);
    free(down);
    free(mid_all);
    free(mid);
    free(up);
    free(gate);
}

/* Decode version of routed MoE: same math as layer_routed_moe_one(), but all
 * large temporaries come from the persistent scratch arena. */
static void layer_routed_moe_one_prealloc(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x,
        uint32_t            il,
        int                 token,
        float               clamp,
        float              * mid_all,
        block_q8_K         * xq,
        block_q8_K         * midq,
        int8_t             * q8_xq,
        float              * q8_xscale,
        int8_t             * q8_midq,
        float              * q8_midscale) {
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const bool routed_q8_0 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q8_0;
    const bool routed_q8_k =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q8_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q8_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q8_K;

    if (routed_q8_0) {
        if ((expert_in_dim % 32u) != 0) ds4_die("Q8_0 expert input is not QK8_0 aligned");
        if (down_in_dim != DS4_N_FF_EXP || (down_in_dim % 32u) != 0) {
            ds4_die("Q8_0 expert input has an unexpected layout");
        }
    } else {
        if (expert_in_dim % QK_K != 0) ds4_die("routed expert input is not QK_K aligned");
        if (down_in_dim != DS4_N_FF_EXP || down_in_dim % QK_K != 0) {
            ds4_die("routed expert down input has an unexpected layout");
        }
    }

    memset(out, 0, (size_t)DS4_N_EMBD * sizeof(out[0]));

    if (layer->ffn_gate_tid2eid) {
        layer_hash_selected_experts(selected, model, layer, token);
        layer_hash_router_weights_one(expert_weight, model, layer, x, selected);
    } else {
        layer_topk_selected_experts(selected, expert_weight, model, layer, x);
    }

    if (routed_q8_0) {
        if (!q8_xq || !q8_xscale || !q8_midq || !q8_midscale) {
            ds4_die("missing Q8_0 routed decode scratch");
        }
        quantize_q8_0_activation(x, q8_xq, q8_xscale, expert_in_dim);
        matvec_q8_0_experts_mid_prequant(mid_all, model,
                                         layer->ffn_gate_exps,
                                         layer->ffn_up_exps,
                                         q8_xq, q8_xscale, selected,
                                         expert_weight,
                                         DS4_N_EXPERT_USED, clamp);
        const uint64_t mid_blocks = down_in_dim / 32u;
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            quantize_q8_0_activation(mid_all + (uint64_t)i * down_in_dim,
                                     q8_midq + (uint64_t)i * mid_blocks * 32u,
                                     q8_midscale + (uint64_t)i * mid_blocks,
                                     down_in_dim);
        }
        matvec_q8_0_experts_accum_prequant(out, model, layer->ffn_down_exps,
                                           q8_midq, q8_midscale, selected,
                                           DS4_N_EXPERT_USED);
        (void)il;
        return;
    }

    if (!mid_all || !xq || !midq) ds4_die("missing routed decode scratch");
    ds4_quantize_row_q8_K(x, xq, (int64_t)expert_in_dim);

    if (routed_q8_k) {
        matvec_q8_k_experts_mid_prequant(mid_all, model,
                                         layer->ffn_gate_exps,
                                         layer->ffn_up_exps,
                                         xq, selected, expert_weight,
                                         DS4_N_EXPERT_USED, clamp);
    } else {
        matvec_experts_mid_prequant(mid_all, model,
                                    layer->ffn_gate_exps,
                                    layer->ffn_up_exps,
                                    xq, selected, expert_weight,
                                    DS4_N_EXPERT_USED, clamp);
    }

    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        ds4_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
                              midq + (uint64_t)i * (down_in_dim / QK_K),
                              (int64_t)down_in_dim);
    }
    if (routed_q8_k) {
        matvec_q8_k_experts_accum_prequant(out, model, layer->ffn_down_exps,
                                           midq, selected, DS4_N_EXPERT_USED);
    } else {
        matvec_experts_down_accum_prequant(out, model, layer->ffn_down_exps,
                                           midq, selected, DS4_N_EXPERT_USED);
    }

    (void)il;
}

/* Prefill MoE groups token/expert pairs by expert so each active expert's
 * rows are scanned once for the whole token batch. */
static void layer_routed_moe_batch(
        float             * moe,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * norm,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        float               clamp) {
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t expert_out_dim = layer->ffn_gate_exps->dim[1];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const uint64_t down_out_dim = layer->ffn_down_exps->dim[1];
    const bool routed_q8_0 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q8_0;
    const bool routed_q8_k =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q8_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q8_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q8_K;
    if (routed_q8_0) {
        if ((expert_in_dim % 32u) != 0 || (down_in_dim % 32u) != 0) {
            ds4_die("Q8_0 routed expert input is not QK8_0 aligned");
        }
    } else {
        if (expert_in_dim % QK_K != 0) ds4_die("routed expert input is not QK_K aligned");
        if (down_in_dim % QK_K != 0) ds4_die("routed expert down input is not QK_K aligned");
    }
    if (expert_out_dim != down_in_dim || down_out_dim != DS4_N_EMBD) {
        ds4_die("routed expert tensor layout is unexpected");
    }

    const uint32_t total_pairs = n_tok * DS4_N_EXPERT_USED;
    uint32_t counts[DS4_MAX_EXPERT + 1] = {0};
    uint32_t cursor[DS4_MAX_EXPERT] = {0};
    uint32_t active_expert[DS4_MAX_EXPERT];
    uint32_t n_active = 0;

    int *selected = xmalloc((size_t)total_pairs * sizeof(selected[0]));
    float *pair_weight = xmalloc((size_t)total_pairs * sizeof(pair_weight[0]));
    ds4_expert_pair *pairs = xmalloc((size_t)total_pairs * sizeof(pairs[0]));

    for (uint32_t t = 0; t < n_tok; t++) {
        int sel[DS4_MAX_EXPERT_USED];
        float weights[DS4_MAX_EXPERT_USED];
        if (layer->ffn_gate_tid2eid) {
            layer_hash_selected_experts(sel, model, layer, token_ids[t]);
            layer_hash_router_weights_one(weights, model, layer, norm + (uint64_t)t * expert_in_dim, sel);
        } else {
            layer_topk_selected_experts(sel, weights, model, layer, norm + (uint64_t)t * expert_in_dim);
        }

        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            const uint32_t pair_id = t * DS4_N_EXPERT_USED + slot;
            selected[pair_id] = sel[slot];
            pair_weight[pair_id] = weights[slot];
            pairs[pair_id] = (ds4_expert_pair){ .token = t, .slot = slot };
            if (sel[slot] < 0 || (uint32_t)sel[slot] >= DS4_N_EXPERT) ds4_die("selected expert is outside range");
            counts[(uint32_t)sel[slot] + 1]++;
        }
    }

    for (uint32_t e = 0; e < DS4_N_EXPERT; e++) {
        counts[e + 1] += counts[e];
        cursor[e] = counts[e];
        if (counts[e + 1] != counts[e]) active_expert[n_active++] = e;
    }

    uint32_t *pair_ids = xmalloc((size_t)total_pairs * sizeof(pair_ids[0]));
    for (uint32_t p = 0; p < total_pairs; p++) {
        const uint32_t e = (uint32_t)selected[p];
        pair_ids[cursor[e]++] = p;
    }

    if (routed_q8_0) {
        const uint64_t x_blocks = expert_in_dim / 32u;
        int8_t *xq8 = xmalloc((size_t)n_tok * x_blocks * 32u);
        float *xscale8 = xmalloc((size_t)n_tok * x_blocks * sizeof(float));
        for (uint32_t t = 0; t < n_tok; t++) {
            quantize_q8_0_activation(norm + (uint64_t)t * expert_in_dim,
                                     xq8 + (uint64_t)t * x_blocks * 32u,
                                     xscale8 + (uint64_t)t * x_blocks,
                                     expert_in_dim);
        }

        float *mid = xmalloc((size_t)total_pairs * expert_out_dim * sizeof(mid[0]));
        matvec_q8_0_batch_mid_ctx mid_ctx = {
            .mid = mid,
            .xq = xq8,
            .xscale = xscale8,
            .pairs = pairs,
            .pair_ids = pair_ids,
            .expert_offset = counts,
            .active_expert = active_expert,
            .pair_weight = pair_weight,
            .clamp = clamp,
            .in_dim = expert_in_dim,
            .out_dim = expert_out_dim,
            .blocks = x_blocks,
        };
        for (uint32_t ai = 0; ai < n_active; ai++) {
            const uint32_t e = active_expert[ai];
            uint64_t gate_in_dim, gate_out_dim;
            uint64_t up_in_dim, up_out_dim;
            mid_ctx.gate_base[e] = tensor_expert_bytes(model, layer->ffn_gate_exps, e,
                                                       &gate_in_dim, &gate_out_dim, &mid_ctx.gate_row_bytes[e]);
            mid_ctx.up_base[e] = tensor_expert_bytes(model, layer->ffn_up_exps, e,
                                                     &up_in_dim, &up_out_dim, &mid_ctx.up_row_bytes[e]);
            if (gate_in_dim != expert_in_dim || up_in_dim != expert_in_dim ||
                gate_out_dim != expert_out_dim || up_out_dim != expert_out_dim) {
                ds4_die("Q8_0 batch expert tensor layout mismatch");
            }
        }
        ds4_parallel_for((uint64_t)n_active * expert_out_dim, matvec_q8_0_batch_mid_worker, &mid_ctx);

        const uint64_t mid_blocks = down_in_dim / 32u;
        int8_t *midq8 = xmalloc((size_t)total_pairs * mid_blocks * 32u);
        float *midscale8 = xmalloc((size_t)total_pairs * mid_blocks * sizeof(float));
        for (uint32_t p = 0; p < total_pairs; p++) {
            quantize_q8_0_activation(mid + (uint64_t)p * down_in_dim,
                                     midq8 + (uint64_t)p * mid_blocks * 32u,
                                     midscale8 + (uint64_t)p * mid_blocks,
                                     down_in_dim);
        }
        free(mid);

        matvec_q8_0_batch_accum_rows_ctx down_ctx = {
            .moe = moe,
            .midq = midq8,
            .midscale = midscale8,
            .pairs = pairs,
            .pair_ids = pair_ids,
            .expert_offset = counts,
            .active_expert = active_expert,
            .n_active = n_active,
            .n_tok = n_tok,
            .in_dim = down_in_dim,
            .out_dim = down_out_dim,
            .blocks = mid_blocks,
        };
        for (uint32_t ai = 0; ai < n_active; ai++) {
            const uint32_t e = active_expert[ai];
            uint64_t in_dim, out_dim;
            down_ctx.base[e] = tensor_expert_bytes(model, layer->ffn_down_exps, e,
                                                   &in_dim, &out_dim, &down_ctx.row_bytes[e]);
            if (in_dim != down_in_dim || out_dim != down_out_dim) {
                ds4_die("Q8_0 batch down expert tensor layout mismatch");
            }
        }
        ds4_parallel_for(down_out_dim, matvec_q8_0_batch_accum_rows_worker, &down_ctx);

        free(midscale8);
        free(midq8);
        free(xscale8);
        free(xq8);
        free(pair_ids);
        free(pairs);
        free(pair_weight);
        free(selected);
        (void)il;
        return;
    }

    if (routed_q8_k) {
        const uint64_t xq_blocks = expert_in_dim / QK_K;
        block_q8_K *xq = xmalloc((size_t)n_tok * xq_blocks * sizeof(xq[0]));
        for (uint32_t t = 0; t < n_tok; t++) {
            ds4_quantize_row_q8_K(norm + (uint64_t)t * expert_in_dim,
                                  xq + (uint64_t)t * xq_blocks,
                                  (int64_t)expert_in_dim);
        }

        float *mid = xmalloc((size_t)total_pairs * expert_out_dim * sizeof(mid[0]));

        matvec_q8_k_batch_mid_ctx mid_ctx = {
            .mid = mid,
            .xq = xq,
            .pairs = pairs,
            .pair_ids = pair_ids,
            .expert_offset = counts,
            .active_expert = active_expert,
            .pair_weight = pair_weight,
            .clamp = clamp,
            .in_dim = expert_in_dim,
            .out_dim = expert_out_dim,
            .xq_blocks = xq_blocks,
        };

        for (uint32_t ai = 0; ai < n_active; ai++) {
            const uint32_t e = active_expert[ai];
            uint64_t gate_in_dim, gate_out_dim;
            uint64_t up_in_dim, up_out_dim;
            mid_ctx.gate_base[e] = tensor_expert_bytes(model, layer->ffn_gate_exps, e,
                                                       &gate_in_dim, &gate_out_dim, &mid_ctx.gate_row_bytes[e]);
            mid_ctx.up_base[e] = tensor_expert_bytes(model, layer->ffn_up_exps, e,
                                                     &up_in_dim, &up_out_dim, &mid_ctx.up_row_bytes[e]);
            if (gate_in_dim != expert_in_dim || up_in_dim != expert_in_dim ||
                gate_out_dim != expert_out_dim || up_out_dim != expert_out_dim) {
                ds4_die("Q8_K batch expert tensor layout mismatch");
            }
        }

        ds4_parallel_for((uint64_t)n_active * expert_out_dim, matvec_q8_k_batch_mid_worker, &mid_ctx);

        const uint64_t midq_blocks = down_in_dim / QK_K;
        block_q8_K *midq = xmalloc((size_t)total_pairs * midq_blocks * sizeof(midq[0]));
        quantize_mid_pairs_ctx quant_ctx = {
            .mid = mid,
            .midq = midq,
            .down_in_dim = down_in_dim,
            .down_blocks = midq_blocks,
        };
        ds4_parallel_for(total_pairs, quantize_mid_pairs_worker, &quant_ctx);
        free(mid);

        matvec_q8_k_batch_accum_rows_ctx down_ctx = {
            .moe = moe,
            .midq = midq,
            .pairs = pairs,
            .pair_ids = pair_ids,
            .expert_offset = counts,
            .active_expert = active_expert,
            .n_active = n_active,
            .n_tok = n_tok,
            .in_dim = down_in_dim,
            .out_dim = down_out_dim,
            .midq_blocks = midq_blocks,
        };

        for (uint32_t ai = 0; ai < n_active; ai++) {
            const uint32_t e = active_expert[ai];
            uint64_t in_dim, out_dim;
            down_ctx.base[e] = tensor_expert_bytes(model, layer->ffn_down_exps, e,
                                                   &in_dim, &out_dim, &down_ctx.row_bytes[e]);
            if (in_dim != down_in_dim || out_dim != down_out_dim) {
                ds4_die("Q8_K batch down expert tensor layout mismatch");
            }
        }

        ds4_parallel_for(down_out_dim, matvec_q8_k_batch_accum_rows_worker, &down_ctx);

        free(midq);
        free(pair_ids);
        free(xq);
        free(pairs);
        free(pair_weight);
        free(selected);

        (void)il;
        return;
    }

    const uint64_t xq_blocks = expert_in_dim / QK_K;
    block_q8_K *xq = xmalloc((size_t)n_tok * xq_blocks * sizeof(xq[0]));
    for (uint32_t t = 0; t < n_tok; t++) {
        ds4_quantize_row_q8_K(norm + (uint64_t)t * expert_in_dim,
                              xq + (uint64_t)t * xq_blocks,
                              (int64_t)expert_in_dim);
    }

    float *mid = xmalloc((size_t)total_pairs * expert_out_dim * sizeof(mid[0]));

    const uint32_t gate_type = layer->ffn_gate_exps->type;

    /* Build mid vectors: dispatch based on gate/up tensor type. */
    if (gate_type == DS4_TENSOR_IQ2_XXS) {
        matvec_iq2_xxs_batch_mid_ctx mid_ctx = {
            .mid = mid,
            .xq = xq,
            .pairs = pairs,
            .pair_ids = pair_ids,
            .expert_offset = counts,
            .active_expert = active_expert,
            .pair_weight = pair_weight,
            .clamp = clamp,
            .in_dim = expert_in_dim,
            .out_dim = expert_out_dim,
            .xq_blocks = xq_blocks,
        };

        for (uint32_t ai = 0; ai < n_active; ai++) {
            const uint32_t e = active_expert[ai];
            uint64_t gate_in_dim, gate_out_dim;
            uint64_t up_in_dim, up_out_dim;
            mid_ctx.gate_base[e] = tensor_expert_bytes(model, layer->ffn_gate_exps, e,
                                                       &gate_in_dim, &gate_out_dim, &mid_ctx.gate_row_bytes[e]);
            mid_ctx.up_base[e] = tensor_expert_bytes(model, layer->ffn_up_exps, e,
                                                     &up_in_dim, &up_out_dim, &mid_ctx.up_row_bytes[e]);
            if (gate_in_dim != expert_in_dim || up_in_dim != expert_in_dim ||
                gate_out_dim != expert_out_dim || up_out_dim != expert_out_dim) {
                ds4_die("batch expert tensor layout mismatch");
            }
        }

        ds4_parallel_for((uint64_t)n_active * expert_out_dim, matvec_iq2_xxs_batch_mid_worker, &mid_ctx);
    } else if (gate_type == DS4_TENSOR_Q2_K) {
        matvec_q2_k_batch_mid_ctx mid_ctx = {
            .mid = mid,
            .xq = xq,
            .pairs = pairs,
            .pair_ids = pair_ids,
            .expert_offset = counts,
            .active_expert = active_expert,
            .pair_weight = pair_weight,
            .clamp = clamp,
            .in_dim = expert_in_dim,
            .out_dim = expert_out_dim,
            .xq_blocks = xq_blocks,
        };

        for (uint32_t ai = 0; ai < n_active; ai++) {
            const uint32_t e = active_expert[ai];
            uint64_t gate_in_dim, gate_out_dim;
            uint64_t up_in_dim, up_out_dim;
            mid_ctx.gate_base[e] = tensor_expert_bytes(model, layer->ffn_gate_exps, e,
                                                       &gate_in_dim, &gate_out_dim, &mid_ctx.gate_row_bytes[e]);
            mid_ctx.up_base[e] = tensor_expert_bytes(model, layer->ffn_up_exps, e,
                                                     &up_in_dim, &up_out_dim, &mid_ctx.up_row_bytes[e]);
            if (gate_in_dim != expert_in_dim || up_in_dim != expert_in_dim ||
                gate_out_dim != expert_out_dim || up_out_dim != expert_out_dim) {
                ds4_die("batch expert tensor layout mismatch");
            }
        }

        ds4_parallel_for((uint64_t)n_active * expert_out_dim, matvec_q2_k_batch_mid_worker, &mid_ctx);
    } else if (gate_type == DS4_TENSOR_Q4_K) {
        matvec_q4_k_batch_mid_ctx mid_ctx = {
            .mid = mid,
            .xq = xq,
            .pairs = pairs,
            .pair_ids = pair_ids,
            .expert_offset = counts,
            .active_expert = active_expert,
            .pair_weight = pair_weight,
            .clamp = clamp,
            .in_dim = expert_in_dim,
            .out_dim = expert_out_dim,
            .xq_blocks = xq_blocks,
        };

        for (uint32_t ai = 0; ai < n_active; ai++) {
            const uint32_t e = active_expert[ai];
            uint64_t gate_in_dim, gate_out_dim;
            uint64_t up_in_dim, up_out_dim;
            mid_ctx.gate_base[e] = tensor_expert_bytes(model, layer->ffn_gate_exps, e,
                                                       &gate_in_dim, &gate_out_dim, &mid_ctx.gate_row_bytes[e]);
            mid_ctx.up_base[e] = tensor_expert_bytes(model, layer->ffn_up_exps, e,
                                                     &up_in_dim, &up_out_dim, &mid_ctx.up_row_bytes[e]);
            if (gate_in_dim != expert_in_dim || up_in_dim != expert_in_dim ||
                gate_out_dim != expert_out_dim || up_out_dim != expert_out_dim) {
                ds4_die("batch expert tensor layout mismatch");
            }
        }

        ds4_parallel_for((uint64_t)n_active * expert_out_dim, matvec_q4_k_batch_mid_worker, &mid_ctx);
    } else {
        ds4_die("unsupported gate/up expert tensor type for batch");
    }

    const uint64_t midq_blocks = down_in_dim / QK_K;
    block_q8_K *midq = xmalloc((size_t)total_pairs * midq_blocks * sizeof(midq[0]));
    quantize_mid_pairs_ctx quant_ctx = {
        .mid = mid,
        .midq = midq,
        .down_in_dim = down_in_dim,
        .down_blocks = midq_blocks,
    };
    ds4_parallel_for(total_pairs, quantize_mid_pairs_worker, &quant_ctx);
    free(mid);

    /* Down projection: dispatch based on down tensor type. */
    const uint32_t down_type = layer->ffn_down_exps->type;

    if (down_type == DS4_TENSOR_IQ2_XXS) {
        matvec_iq2_xxs_batch_accum_rows_ctx down_ctx = {
            .moe = moe,
            .midq = midq,
            .pairs = pairs,
            .pair_ids = pair_ids,
            .expert_offset = counts,
            .active_expert = active_expert,
            .n_active = n_active,
            .n_tok = n_tok,
            .in_dim = down_in_dim,
            .out_dim = down_out_dim,
            .midq_blocks = midq_blocks,
        };

        for (uint32_t ai = 0; ai < n_active; ai++) {
            const uint32_t e = active_expert[ai];
            uint64_t in_dim, out_dim;
            down_ctx.base[e] = tensor_expert_bytes(model, layer->ffn_down_exps, e,
                                                   &in_dim, &out_dim, &down_ctx.row_bytes[e]);
            if (in_dim != down_in_dim || out_dim != down_out_dim) {
                ds4_die("batch expert tensor layout mismatch");
            }
        }

        ds4_parallel_for(down_out_dim, matvec_iq2_xxs_batch_accum_rows_worker, &down_ctx);
    } else if (down_type == DS4_TENSOR_Q2_K) {
        matvec_q2_k_batch_accum_rows_ctx down_ctx = {
            .moe = moe,
            .midq = midq,
            .pairs = pairs,
            .pair_ids = pair_ids,
            .expert_offset = counts,
            .active_expert = active_expert,
            .n_active = n_active,
            .n_tok = n_tok,
            .in_dim = down_in_dim,
            .out_dim = down_out_dim,
            .midq_blocks = midq_blocks,
        };

        for (uint32_t ai = 0; ai < n_active; ai++) {
            const uint32_t e = active_expert[ai];
            uint64_t in_dim, out_dim;
            down_ctx.base[e] = tensor_expert_bytes(model, layer->ffn_down_exps, e,
                                                   &in_dim, &out_dim, &down_ctx.row_bytes[e]);
            if (in_dim != down_in_dim || out_dim != down_out_dim) {
                ds4_die("batch expert tensor layout mismatch");
            }
        }

        ds4_parallel_for(down_out_dim, matvec_q2_k_batch_accum_rows_worker, &down_ctx);
    } else if (down_type == DS4_TENSOR_Q4_K) {
        matvec_q4_k_batch_accum_rows_ctx down_ctx = {
            .moe = moe,
            .midq = midq,
            .pairs = pairs,
            .pair_ids = pair_ids,
            .expert_offset = counts,
            .active_expert = active_expert,
            .n_active = n_active,
            .n_tok = n_tok,
            .in_dim = down_in_dim,
            .out_dim = down_out_dim,
            .midq_blocks = midq_blocks,
        };

        for (uint32_t ai = 0; ai < n_active; ai++) {
            const uint32_t e = active_expert[ai];
            uint64_t in_dim, out_dim;
            down_ctx.base[e] = tensor_expert_bytes(model, layer->ffn_down_exps, e,
                                                   &in_dim, &out_dim, &down_ctx.row_bytes[e]);
            if (in_dim != down_in_dim || out_dim != down_out_dim) {
                ds4_die("batch expert tensor layout mismatch");
            }
        }

        ds4_parallel_for(down_out_dim, matvec_q4_k_batch_accum_rows_worker, &down_ctx);
    } else {
        ds4_die("unsupported down expert tensor type for batch");
    }

    free(midq);
    free(pair_ids);
    free(xq);
    free(pairs);
    free(pair_weight);
    free(selected);

    (void)il;
}

static void print_vec_stats(const char *name, const float *x, uint64_t n);

/* Full FFN sublayer for one token: HC pre, RMSNorm, routed MoE, shared expert,
 * sum, and HC post. */
static void layer_ffn_one(
        float             * out_hc,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * inp_hc,
        uint32_t            il,
        int                 token,
        const float       * steering_dirs,
        float               steering_scale,
        bool                trace) {
    const uint32_t n_hc = DS4_N_HC;
    const bool profile = getenv("DS4_DECODE_PROFILE_DETAIL") != NULL;
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc = 0.0;
    double t_norm = 0.0;
    double t_routed = 0.0;
    double t_shared = 0.0;
    double t_post = 0.0;
    float *ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(ffn_cur[0]));
    float *norm = xmalloc((size_t)DS4_N_EMBD * sizeof(norm[0]));
    float *moe = xmalloc((size_t)DS4_N_EMBD * sizeof(moe[0]));
    float *shared = xmalloc((size_t)DS4_N_EMBD * sizeof(shared[0]));
    float *ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(ffn_out[0]));
    float post[4];
    float comb[16];

    double t0 = profile ? now_sec() : 0.0;
    hc_pre_from_state_one(model,
                          layer->hc_ffn_fn,
                          layer->hc_ffn_scale,
                          layer->hc_ffn_base,
                          inp_hc, ffn_cur, post, comb);
    if (profile) t_hc = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_cur", il);
        print_vec_stats(name, ffn_cur, DS4_N_EMBD);
    }

    t0 = profile ? now_sec() : 0.0;
    const float *ffn_norm = tensor_data(model, layer->ffn_norm);
    rms_norm_weight(norm, ffn_cur, ffn_norm, DS4_N_EMBD, DS4_RMS_EPS);
    if (profile) t_norm = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_norm", il);
        print_vec_stats(name, norm, DS4_N_EMBD);
    }

    t0 = profile ? now_sec() : 0.0;
    layer_routed_moe_one(moe, model, layer, norm, il, token, DS4_SWIGLU_CLAMP_EXP, trace);
    if (profile) t_routed = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u routed_moe", il);
        print_vec_stats(name, moe, DS4_N_EMBD);
    }
    t0 = profile ? now_sec() : 0.0;
    layer_shared_ffn_one(shared, model, layer, norm);
    if (profile) t_shared = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u shared_ffn", il);
        print_vec_stats(name, shared, DS4_N_EMBD);
    }

    t0 = profile ? now_sec() : 0.0;
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
        ffn_out[i] = moe[i] + shared[i];
    }
    cpu_directional_steering_project_rows(ffn_out, steering_dirs, il, 1, steering_scale);
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_out", il);
        print_vec_stats(name, ffn_out, DS4_N_EMBD);
    }

    hc_post_one(out_hc, ffn_out, inp_hc, post, comb, DS4_N_EMBD, n_hc);
    if (profile) t_post = now_sec() - t0;
    if (trace) {
        char name[64];
        snprintf(name, sizeof(name), "blk.%u ffn_post_hc", il);
        print_vec_stats(name, out_hc, (uint64_t)n_hc * DS4_N_EMBD);
    }

    if (profile) {
        fprintf(stderr,
                "ds4: decode detail layer %u ffn hc=%.3f norm=%.3f routed=%.3f shared=%.3f post=%.3f total=%.3f ms\n",
                il,
                t_hc * 1000.0,
                t_norm * 1000.0,
                t_routed * 1000.0,
                t_shared * 1000.0,
                t_post * 1000.0,
                (now_sec() - t_start) * 1000.0);
    }

    free(ffn_out);
    free(shared);
    free(moe);
    free(norm);
    free(ffn_cur);
}

/* Allocation-free decode FFN using the persistent CPU scratch buffers. */
static void layer_ffn_one_decode_scratch(
        float                  * out_hc,
        const ds4_model        * model,
        const ds4_layer_weights * layer,
        const float            * inp_hc,
        uint32_t                 il,
        int                      token,
        const float            * steering_dirs,
        float                    steering_scale,
        ds4_cpu_decode_scratch * scratch) {
    const uint32_t n_hc = DS4_N_HC;
    const bool profile = getenv("DS4_DECODE_PROFILE_DETAIL") != NULL;
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc = 0.0;
    double t_norm = 0.0;
    double t_routed = 0.0;
    double t_shared = 0.0;
    double t_post = 0.0;
    float post[4];
    float comb[16];

    double t0 = profile ? now_sec() : 0.0;
    hc_pre_from_state_one_scratch(model,
                                  layer->hc_ffn_fn,
                                  layer->hc_ffn_scale,
                                  layer->hc_ffn_base,
                                  inp_hc, scratch->ffn_cur, post, comb,
                                  scratch->hc_flat,
                                  false);
    if (profile) t_hc = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    const float *ffn_norm = tensor_data(model, layer->ffn_norm);
    rms_norm_weight(scratch->ffn_norm, scratch->ffn_cur, ffn_norm, DS4_N_EMBD, DS4_RMS_EPS);
    if (profile) t_norm = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_routed_moe_one_prealloc(scratch->ffn_moe,
                                  model,
                                  layer,
                                  scratch->ffn_norm,
                                  il,
                                  token,
                                  DS4_SWIGLU_CLAMP_EXP,
                                  scratch->routed_mid_all,
                                  scratch->routed_xq,
                                  scratch->routed_midq,
                                  scratch->routed_q8_xq,
                                  scratch->routed_q8_xscale,
                                  scratch->routed_q8_midq,
                                  scratch->routed_q8_midscale);
    if (profile) t_routed = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_shared_ffn_one_decode_scratch(scratch->ffn_shared, model, layer, scratch->ffn_norm, scratch);
    if (profile) t_shared = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
        scratch->ffn_out[i] = scratch->ffn_moe[i] + scratch->ffn_shared[i];
    }
    cpu_directional_steering_project_rows(scratch->ffn_out, steering_dirs, il, 1, steering_scale);
    hc_post_one(out_hc, scratch->ffn_out, inp_hc, post, comb, DS4_N_EMBD, n_hc);
    if (profile) t_post = now_sec() - t0;

    if (profile) {
        fprintf(stderr,
                "ds4: decode detail layer %u ffn hc=%.3f norm=%.3f routed=%.3f shared=%.3f post=%.3f total=%.3f ms\n",
                il,
                t_hc * 1000.0,
                t_norm * 1000.0,
                t_routed * 1000.0,
                t_shared * 1000.0,
                t_post * 1000.0,
                (now_sec() - t_start) * 1000.0);
    }
}

static void layer_ffn_batch(
        float             * out_hc,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * inp_hc,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        const float       * steering_dirs,
        float               steering_scale) {
    if (n_tok == 0) return;
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)n_hc * DS4_N_EMBD;
    float *ffn_cur = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(ffn_cur[0]));
    float *norm = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(norm[0]));
    float *moe = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(moe[0]));
    float *shared = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(shared[0]));
    float *post = xmalloc((size_t)n_tok * n_hc * sizeof(post[0]));
    float *comb = xmalloc((size_t)n_tok * n_hc * n_hc * sizeof(comb[0]));
    const float *ffn_norm = tensor_data(model, layer->ffn_norm);

    for (uint32_t t = 0; t < n_tok; t++) {
        hc_pre_from_state_one(model,
                              layer->hc_ffn_fn,
                              layer->hc_ffn_scale,
                              layer->hc_ffn_base,
                              inp_hc + (uint64_t)t * hc_dim,
                              ffn_cur + (uint64_t)t * DS4_N_EMBD,
                              post + (uint64_t)t * n_hc,
                              comb + (uint64_t)t * n_hc * n_hc);
        rms_norm_weight(norm + (uint64_t)t * DS4_N_EMBD,
                        ffn_cur + (uint64_t)t * DS4_N_EMBD,
                        ffn_norm,
                        DS4_N_EMBD,
                        DS4_RMS_EPS);
    }

    layer_routed_moe_batch(moe, model, layer, norm, token_ids, n_tok, il, DS4_SWIGLU_CLAMP_EXP);
    layer_shared_ffn_batch(shared, model, layer, norm, n_tok);

    if (cpu_directional_steering_enabled(steering_dirs, steering_scale)) {
        float *ffn_out = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(ffn_out[0]));
        for (uint64_t i = 0; i < (uint64_t)n_tok * DS4_N_EMBD; i++) {
            ffn_out[i] = moe[i] + shared[i];
        }
        cpu_directional_steering_project_rows(ffn_out, steering_dirs, il, n_tok, steering_scale);
        hc_post_batch(out_hc,
                      ffn_out,
                      inp_hc,
                      post,
                      comb,
                      n_tok,
                      DS4_N_EMBD,
                      n_hc);
        free(ffn_out);
    } else {
        hc_post_sum_batch(out_hc,
                          moe,
                          shared,
                          inp_hc,
                          post,
                          comb,
                          n_tok,
                          DS4_N_EMBD,
                          n_hc);
    }

    free(comb);
    free(post);
    free(shared);
    free(moe);
    free(norm);
    free(ffn_cur);
}

typedef struct {
    float *moe;
    const ds4_model *model;
    const ds4_layer_weights *layer;
    const float *norm;
    const int *token_ids;
    uint64_t expert_in_dim;
    uint64_t down_in_dim;
    uint32_t il;
    bool routed_q8_0;
} routed_moe_tokens_ctx;

static void routed_moe_tokens_worker(void *vctx, uint64_t t0, uint64_t t1) {
    routed_moe_tokens_ctx *ctx = vctx;
    const uint64_t q8_x_blocks = ctx->expert_in_dim / 32u;
    const uint64_t q8_mid_blocks = ctx->down_in_dim / 32u;
    float *routed_mid = xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(routed_mid[0]));
    block_q8_K *routed_xq = ctx->routed_q8_0 ? NULL : xmalloc((size_t)(ctx->expert_in_dim / QK_K) * sizeof(routed_xq[0]));
    block_q8_K *routed_midq = ctx->routed_q8_0 ? NULL : xmalloc((size_t)DS4_N_EXPERT_USED * (ctx->down_in_dim / QK_K) * sizeof(routed_midq[0]));
    int8_t *routed_q8_xq = ctx->routed_q8_0 ? xmalloc((size_t)q8_x_blocks * 32u) : NULL;
    float *routed_q8_xscale = ctx->routed_q8_0 ? xmalloc((size_t)q8_x_blocks * sizeof(routed_q8_xscale[0])) : NULL;
    int8_t *routed_q8_midq = ctx->routed_q8_0 ? xmalloc((size_t)DS4_N_EXPERT_USED * q8_mid_blocks * 32u) : NULL;
    float *routed_q8_midscale = ctx->routed_q8_0 ? xmalloc((size_t)DS4_N_EXPERT_USED * q8_mid_blocks * sizeof(routed_q8_midscale[0])) : NULL;

    for (uint64_t t = t0; t < t1; t++) {
        layer_routed_moe_one_prealloc(ctx->moe + t * DS4_N_EMBD,
                                      ctx->model,
                                      ctx->layer,
                                      ctx->norm + t * DS4_N_EMBD,
                                      ctx->il,
                                      ctx->token_ids[t],
                                      DS4_SWIGLU_CLAMP_EXP,
                                      routed_mid,
                                      routed_xq,
                                      routed_midq,
                                      routed_q8_xq,
                                      routed_q8_xscale,
                                      routed_q8_midq,
                                      routed_q8_midscale);
    }

    free(routed_q8_midscale);
    free(routed_q8_midq);
    free(routed_q8_xscale);
    free(routed_q8_xq);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid);
}

static void layer_routed_moe_tokens_parallel(
        float             * moe,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * norm,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il) {
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const bool routed_q8_k =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q8_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q8_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q8_K;
    if (routed_q8_k) {
        if (expert_in_dim % QK_K != 0) ds4_die("Q8_K expert input is not QK_K aligned");
        if (down_in_dim != DS4_N_FF_EXP || down_in_dim % QK_K != 0) {
            ds4_die("Q8_K expert input has an unexpected layout");
        }
    }
    routed_moe_tokens_ctx ctx = {
        .moe = moe,
        .model = model,
        .layer = layer,
        .norm = norm,
        .token_ids = token_ids,
        .expert_in_dim = expert_in_dim,
        .down_in_dim = down_in_dim,
        .il = il,
        .routed_q8_0 =
            layer->ffn_gate_exps->type == DS4_TENSOR_Q8_0 &&
            layer->ffn_up_exps->type == DS4_TENSOR_Q8_0 &&
            layer->ffn_down_exps->type == DS4_TENSOR_Q8_0,
    };
    ds4_parallel_for_min_rows(n_tok, routed_moe_tokens_worker, &ctx, 1);
}

/* Default prefill FFN path.  HC and shared expert are batched, while routed
 * experts can run either token-parallel or expert-grouped depending on size. */
static void layer_ffn_shared_batch(
        float             * out_hc,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * inp_hc,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        const float       * steering_dirs,
        float               steering_scale) {
    const bool profile = getenv("DS4_PREFILL_PROFILE_DETAIL") != NULL;
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc_norm = 0.0;
    double t_routed = 0.0;
    double t_shared = 0.0;
    double t_post = 0.0;
    const uint32_t n_hc = DS4_N_HC;
    float *ffn_cur = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(ffn_cur[0]));
    float *norm = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(norm[0]));
    float *moe = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(moe[0]));
    float *shared = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(shared[0]));
    float *post = xmalloc((size_t)n_tok * n_hc * sizeof(post[0]));
    float *comb = xmalloc((size_t)n_tok * n_hc * n_hc * sizeof(comb[0]));
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    const bool routed_q8_0 =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q8_0 &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q8_0;
    const bool routed_q8_k =
        layer->ffn_gate_exps->type == DS4_TENSOR_Q8_K &&
        layer->ffn_up_exps->type == DS4_TENSOR_Q8_K &&
        layer->ffn_down_exps->type == DS4_TENSOR_Q8_K;
    if (routed_q8_k) {
        if (expert_in_dim % QK_K != 0) ds4_die("Q8_K expert input is not QK_K aligned");
        if (down_in_dim != DS4_N_FF_EXP || down_in_dim % QK_K != 0) {
            ds4_die("Q8_K expert input has an unexpected layout");
        }
    }
    const uint64_t routed_q8_x_blocks = expert_in_dim / 32u;
    const uint64_t routed_q8_mid_blocks = down_in_dim / 32u;
    const bool routed_token_parallel =
        getenv("DS4_ROUTED_TOKEN_PARALLEL") != NULL ||
        (getenv("DS4_NO_ROUTED_TOKEN_PARALLEL") == NULL && n_tok >= 64);
    float *routed_mid = routed_token_parallel ? NULL : xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(routed_mid[0]));
    block_q8_K *routed_xq = (routed_token_parallel || routed_q8_0) ? NULL : xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(routed_xq[0]));
    block_q8_K *routed_midq = (routed_token_parallel || routed_q8_0) ? NULL : xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(routed_midq[0]));
    int8_t *routed_q8_xq = (!routed_token_parallel && routed_q8_0) ? xmalloc((size_t)routed_q8_x_blocks * 32u) : NULL;
    float *routed_q8_xscale = (!routed_token_parallel && routed_q8_0) ? xmalloc((size_t)routed_q8_x_blocks * sizeof(routed_q8_xscale[0])) : NULL;
    int8_t *routed_q8_midq = (!routed_token_parallel && routed_q8_0) ? xmalloc((size_t)DS4_N_EXPERT_USED * routed_q8_mid_blocks * 32u) : NULL;
    float *routed_q8_midscale = (!routed_token_parallel && routed_q8_0) ? xmalloc((size_t)DS4_N_EXPERT_USED * routed_q8_mid_blocks * sizeof(routed_q8_midscale[0])) : NULL;

    double t0 = profile ? now_sec() : 0.0;
    hc_pre_norm_batch(model,
                      layer->hc_ffn_fn,
                      layer->hc_ffn_scale,
                      layer->hc_ffn_base,
                      layer->ffn_norm,
                      inp_hc,
                      NULL,
                      ffn_cur,
                      norm,
                      post,
                      comb,
                      n_tok);
    if (profile) t_hc_norm = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    if (routed_token_parallel) {
        layer_routed_moe_tokens_parallel(moe, model, layer, norm, token_ids, n_tok, il);
    } else {
        for (uint32_t t = 0; t < n_tok; t++) {
            layer_routed_moe_one_prealloc(moe + (uint64_t)t * DS4_N_EMBD,
                                          model,
                                          layer,
                                          norm + (uint64_t)t * DS4_N_EMBD,
                                          il,
                                          token_ids[t],
                                          DS4_SWIGLU_CLAMP_EXP,
                                          routed_mid,
                                          routed_xq,
                                          routed_midq,
                                          routed_q8_xq,
                                          routed_q8_xscale,
                                          routed_q8_midq,
                                          routed_q8_midscale);
        }
    }
    if (profile) t_routed = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_shared_ffn_batch(shared, model, layer, norm, n_tok);
    if (profile) t_shared = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    if (cpu_directional_steering_enabled(steering_dirs, steering_scale)) {
        float *ffn_out = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(ffn_out[0]));
        for (uint64_t i = 0; i < (uint64_t)n_tok * DS4_N_EMBD; i++) {
            ffn_out[i] = moe[i] + shared[i];
        }
        cpu_directional_steering_project_rows(ffn_out, steering_dirs, il, n_tok, steering_scale);
        hc_post_batch(out_hc,
                      ffn_out,
                      inp_hc,
                      post,
                      comb,
                      n_tok,
                      DS4_N_EMBD,
                      n_hc);
        free(ffn_out);
    } else {
        hc_post_sum_batch(out_hc,
                          moe,
                          shared,
                          inp_hc,
                          post,
                          comb,
                          n_tok,
                          DS4_N_EMBD,
                          n_hc);
    }
    if (profile) t_post = now_sec() - t0;

    if (profile) {
        fprintf(stderr,
                "ds4: prefill detail layer %u ffn hc_norm=%.3f routed=%.3f shared=%.3f post=%.3f total=%.3f\n",
                il, t_hc_norm, t_routed, t_shared, t_post, now_sec() - t_start);
    }

    free(comb);
    free(post);
    free(routed_q8_midscale);
    free(routed_q8_midq);
    free(routed_q8_xscale);
    free(routed_q8_xq);
    free(routed_midq);
    free(routed_xq);
    free(routed_mid);
    free(shared);
    free(moe);
    free(norm);
    free(ffn_cur);
}

typedef struct {
    float *out_hc;
    const ds4_model *model;
    const ds4_layer_weights *layer;
    const float *inp_hc;
    const int *token_ids;
    const float *steering_dirs;
    float steering_scale;
    uint64_t hc_dim;
    uint32_t il;
} layer_ffn_tokens_ctx;

static void layer_ffn_tokens_worker(void *vctx, uint64_t t0, uint64_t t1) {
    layer_ffn_tokens_ctx *ctx = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        layer_ffn_one(ctx->out_hc + t * ctx->hc_dim,
                      ctx->model,
                      ctx->layer,
                      ctx->inp_hc + t * ctx->hc_dim,
                      ctx->il,
                      ctx->token_ids[t],
                      ctx->steering_dirs,
                      ctx->steering_scale,
                      false);
    }
}

static void layer_ffn_tokens_parallel(
        float             * out_hc,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * inp_hc,
        const int         * token_ids,
        uint32_t            n_tok,
        uint32_t            il,
        const float       * steering_dirs,
        float               steering_scale) {
    layer_ffn_tokens_ctx ctx = {
        .out_hc = out_hc,
        .model = model,
        .layer = layer,
        .inp_hc = inp_hc,
        .token_ids = token_ids,
        .steering_dirs = steering_dirs,
        .steering_scale = steering_scale,
        .hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD,
        .il = il,
    };
    ds4_parallel_for(n_tok, layer_ffn_tokens_worker, &ctx);
}

static void output_logits_one(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        const float       * inp_hc);

