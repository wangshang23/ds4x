/* Full transformer layer for one decode token: attention sublayer followed by
 * FFN sublayer, both operating on the HC state. */
static void layer_forward_raw_swa_one(
        float                   * out_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        ds4_layer_cache         * cache,
        const float             * inp_hc,
        uint32_t                  il,
        uint32_t                  pos,
        int                       token,
        const float             * steering_dirs,
        float                     steering_attn_scale,
        float                     steering_ffn_scale,
        ds4_cpu_decode_scratch  * scratch) {
    const uint32_t n_hc = DS4_N_HC;
    const bool profile = getenv("DS4_DECODE_PROFILE_DETAIL") != NULL;
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc = 0.0;
    double t_q = 0.0;
    double t_kv = 0.0;
    double t_rope_cache = 0.0;
    double t_compress = 0.0;
    double t_indexer = 0.0;
    double t_attn_rows = 0.0;
    double t_inv_rope = 0.0;
    double t_out = 0.0;
    double t_post = 0.0;
    double t_ffn = 0.0;

    bool *comp_allowed = NULL;
    float post[4];
    float comb[16];

    double t0 = profile ? now_sec() : 0.0;
    memcpy(scratch->attn_residual, inp_hc, (size_t)n_hc * DS4_N_EMBD * sizeof(inp_hc[0]));
    hc_pre_from_state_one_scratch(model,
                                  layer->hc_attn_fn,
                                  layer->hc_attn_scale,
                                  layer->hc_attn_base,
                                  scratch->attn_residual, scratch->attn_cur, post, comb,
                                  scratch->hc_flat,
                                  false);
    if (profile) t_hc = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_attn_norm_one(scratch->attn_norm, model, layer, scratch->attn_cur);
    const uint32_t ratio = cache->compress_ratio;
    layer_q_projection_with_lora_one_decode_scratch(model, layer,
                                                    scratch->attn_norm,
                                                    scratch->q,
                                                    scratch->qr_norm,
                                                    scratch);
    if (profile) t_q = now_sec() - t0;
    t0 = profile ? now_sec() : 0.0;
    layer_kv_projection_normed_one_decode_scratch(model, layer,
                                                  scratch->attn_norm,
                                                  scratch->kv,
                                                  scratch);
    if (profile) t_kv = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    rope_tail_layer_inplace(scratch->q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    rope_tail_layer_inplace(scratch->kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(scratch->kv, DS4_N_HEAD_DIM, DS4_N_ROT);

    kv_cache_push_raw(cache, scratch->kv);
    if (profile) t_rope_cache = now_sec() - t0;

    if (ratio != 0) {
        t0 = profile ? now_sec() : 0.0;
        if (compressor_decode_one_decode_scratch(scratch->comp, model,
                                                 layer->attn_compressor_kv,
                                                 layer->attn_compressor_gate,
                                                 layer->attn_compressor_ape,
                                                 layer->attn_compressor_norm,
                                                 scratch->attn_norm,
                                                 cache->attn_state_kv,
                                                 cache->attn_state_score,
                                                 DS4_N_HEAD_DIM,
                                                 ratio,
                                                 il,
                                                 pos,
                                                 scratch)) {
            kv_cache_push_comp(cache->attn_comp_kv, &cache->n_comp, cache->comp_cap, DS4_N_HEAD_DIM, scratch->comp);
        }

        if (ratio == 4) {
            if (compressor_decode_one_decode_scratch(scratch->index_comp, model,
                                                     layer->indexer_compressor_kv,
                                                     layer->indexer_compressor_gate,
                                                     layer->indexer_compressor_ape,
                                                     layer->indexer_compressor_norm,
                                                     scratch->attn_norm,
                                                     cache->index_state_kv,
                                                     cache->index_state_score,
                                                     DS4_N_INDEXER_HEAD_DIM,
                                                     ratio,
                                                     il,
                                                     pos,
                                                     scratch)) {
                kv_cache_push_comp(cache->index_comp_kv, &cache->n_index_comp, cache->comp_cap,
                                   DS4_N_INDEXER_HEAD_DIM, scratch->index_comp);
            }
            if (profile) t_compress = now_sec() - t0;
        } else if (profile) {
            t_compress = now_sec() - t0;
        }
    }
    if (ratio == 4) {
        t0 = profile ? now_sec() : 0.0;
        comp_allowed = indexer_allowed_decode_one_decode_scratch(model, layer,
                                                                 scratch->attn_norm,
                                                                 scratch->qr_norm,
                                                                 cache->index_comp_kv,
                                                                 cache->n_index_comp,
                                                                 il, pos,
                                                                 scratch);
        if (profile) t_indexer = now_sec() - t0;
    }

    t0 = profile ? now_sec() : 0.0;
    if (ratio != 0) {
        layer_attention_mixed_one_decode_scratch(scratch->heads, model, layer, scratch->q,
                                                 cache->raw_kv, cache->n_raw,
                                                 cache->attn_comp_kv, cache->n_comp,
                                                 comp_allowed,
                                                 scratch);
    } else {
        layer_attention_rows_one(scratch->heads, model, layer, scratch->q, cache->raw_kv, cache->n_raw);
    }
    if (profile) t_attn_rows = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    rope_tail_layer_inplace(scratch->heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, true);
    if (profile) t_inv_rope = now_sec() - t0;
    t0 = profile ? now_sec() : 0.0;
    layer_grouped_out_one_decode_scratch(scratch->attn_out, model, layer, scratch->heads, scratch);
    cpu_directional_steering_project_rows(scratch->attn_out, steering_dirs, il, 1, steering_attn_scale);
    if (profile) t_out = now_sec() - t0;
    t0 = profile ? now_sec() : 0.0;
    hc_post_one(scratch->after_attn_hc, scratch->attn_out, scratch->attn_residual, post, comb, DS4_N_EMBD, n_hc);
    if (profile) t_post = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_ffn_one_decode_scratch(out_hc, model, layer, scratch->after_attn_hc, il, token,
                                 steering_dirs, steering_ffn_scale, scratch);
    if (profile) t_ffn = now_sec() - t0;

    if (profile) {
        fprintf(stderr,
                "ds4: decode detail layer %u attn hc=%.3f q=%.3f kv=%.3f rope=%.3f compress=%.3f indexer=%.3f attn_rows=%.3f inv_rope=%.3f out=%.3f post=%.3f ffn=%.3f total=%.3f ms\n",
                il,
                t_hc * 1000.0,
                t_q * 1000.0,
                t_kv * 1000.0,
                t_rope_cache * 1000.0,
                t_compress * 1000.0,
                t_indexer * 1000.0,
                t_attn_rows * 1000.0,
                t_inv_rope * 1000.0,
                t_out * 1000.0,
                t_post * 1000.0,
                t_ffn * 1000.0,
                (now_sec() - t_start) * 1000.0);
    }

}

static void output_logits_one_decode_scratch(
        float                  * logits,
        const ds4_model        * model,
        const ds4_weights      * weights,
        const float            * inp_hc,
        ds4_cpu_decode_scratch * scratch);

/* CPU decode for one token through all 43 layers.  The caller owns scratch and
 * cache lifetimes so no per-token allocations are needed. */
static void forward_token_raw_swa_cpu_decode_scratch(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        ds4_kv_cache      * cache,
        int                 token,
        uint32_t            pos,
        const float       * steering_dirs,
        float               steering_attn_scale,
        float               steering_ffn_scale,
        ds4_cpu_decode_scratch * scratch) {
    float *cur = scratch->cur;
    float *next = scratch->next;

    embed_token_f16(model, weights, token, scratch->plain);
    hc_from_plain_embedding(cur, scratch->plain, DS4_N_EMBD, DS4_N_HC);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        layer_forward_raw_swa_one(next, model, &weights->layer[il], &cache->layer[il],
                                  cur, il, pos, token,
                                  steering_dirs,
                                  steering_attn_scale,
                                  steering_ffn_scale,
                                  scratch);
        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    if (logits) {
        output_logits_one_decode_scratch(logits, model, weights, cur, scratch);
    }
}

#ifndef DS4_NO_GPU
static void forward_token_raw_swa_cpu(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        ds4_kv_cache      * cache,
        int                 token,
        uint32_t            pos) {
    ds4_cpu_decode_scratch scratch;
    uint32_t ctx_guess = pos + 1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = cache->layer[il].compress_ratio;
        if (ratio != 0 && cache->layer[il].comp_cap > 2) {
            const uint32_t ctx_from_comp = (cache->layer[il].comp_cap - 2u) * ratio;
            if (ctx_guess < ctx_from_comp) ctx_guess = ctx_from_comp;
        }
    }
    cpu_decode_scratch_init(&scratch, ctx_guess);
    forward_token_raw_swa_cpu_decode_scratch(logits, model, weights, cache, token, pos,
                                             NULL, 0.0f, 0.0f, &scratch);
    cpu_decode_scratch_free(&scratch);
}
#endif

/* CPU prefill in layer-major order.  All prompt tokens pass through layer 0,
 * then layer 1, etc., which exposes batch matmul opportunities. */
static void prefill_layer_major_cpu(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        ds4_kv_cache      * cache,
        const token_vec   * prompt,
        const float       * steering_dirs,
        float               steering_attn_scale,
        float               steering_ffn_scale) {
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t n_tok = (uint64_t)prompt->len;
    float *cur = xmalloc((size_t)n_tok * hc_dim * sizeof(cur[0]));
    float *next = xmalloc((size_t)n_tok * hc_dim * sizeof(next[0]));
    float *attn = xmalloc((size_t)n_tok * hc_dim * sizeof(attn[0]));
    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(plain[0]));
    uint32_t ffn_batch = 128;
    const bool batched_attn = getenv("DS4_NO_BATCHED_ATTN") == NULL;
    const bool batched_ffn = getenv("DS4_BATCHED_FFN") != NULL;
    const bool parallel_ffn = getenv("DS4_PARALLEL_FFN") != NULL;
    const bool shared_batch_ffn = getenv("DS4_NO_SHARED_BATCH_FFN") == NULL;
    const char *batch_env = getenv("DS4_PREFILL_BATCH");
    ds4_cpu_decode_scratch decode_scratch;
    bool decode_scratch_ready = false;
    if (batch_env && batch_env[0]) {
        long v = strtol(batch_env, NULL, 10);
        if (v > 0 && v < 4096) ffn_batch = (uint32_t)v;
    }

    for (uint64_t t = 0; t < n_tok; t++) {
        embed_token_f16(model, weights, prompt->v[t], plain);
        hc_from_plain_embedding(cur + t * hc_dim, plain, DS4_N_EMBD, DS4_N_HC);
    }

    free(plain);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        fprintf(stderr, "ds4: prefill layer %u/%u\r", il + 1, (uint32_t)DS4_N_LAYER);
        fflush(stderr);

        if (batched_attn) {
            layer_attention_raw_swa_batch(attn,
                                          model,
                                          &weights->layer[il],
                                          &cache->layer[il],
                                          cur,
                                          (uint32_t)n_tok,
                                          il,
                                          0,
                                          steering_dirs,
                                          steering_attn_scale);

            if (batched_ffn) {
                for (uint64_t t = 0; t < n_tok; t += ffn_batch) {
                    uint32_t nb = (uint32_t)((n_tok - t) < ffn_batch ? (n_tok - t) : ffn_batch);
                    layer_ffn_batch(next + t * hc_dim,
                                    model,
                                    &weights->layer[il],
                                    attn + t * hc_dim,
                                    prompt->v + t,
                                    nb,
                                    il,
                                    steering_dirs,
                                    steering_ffn_scale);
                }
            } else if (shared_batch_ffn) {
                layer_ffn_shared_batch(next,
                                       model,
                                       &weights->layer[il],
                                       attn,
                                       prompt->v,
                                       (uint32_t)n_tok,
                                       il,
                                       steering_dirs,
                                       steering_ffn_scale);
            } else if (parallel_ffn) {
                layer_ffn_tokens_parallel(next,
                                          model,
                                          &weights->layer[il],
                                          attn,
                                          prompt->v,
                                          (uint32_t)n_tok,
                                          il,
                                          steering_dirs,
                                          steering_ffn_scale);
            } else {
                for (uint64_t t = 0; t < n_tok; t++) {
                    layer_ffn_one(next + t * hc_dim,
                                  model,
                                  &weights->layer[il],
                                  attn + t * hc_dim,
                                  il,
                                  prompt->v[t],
                                  steering_dirs,
                                  steering_ffn_scale,
                                  false);
                }
            }
        } else if (batched_ffn) {
            for (uint64_t t = 0; t < n_tok; t++) {
                layer_attention_raw_swa_one(attn + t * hc_dim,
                                            model,
                                            &weights->layer[il],
                                            &cache->layer[il],
                                            cur + t * hc_dim,
                                            il,
                                            (uint32_t)t,
                                            steering_dirs,
                                            steering_attn_scale);
            }

            for (uint64_t t = 0; t < n_tok; t += ffn_batch) {
                uint32_t nb = (uint32_t)((n_tok - t) < ffn_batch ? (n_tok - t) : ffn_batch);
                layer_ffn_batch(next + t * hc_dim,
                                model,
                                &weights->layer[il],
                                attn + t * hc_dim,
                                prompt->v + t,
                                nb,
                                il,
                                steering_dirs,
                                steering_ffn_scale);
            }
        } else {
            if (!decode_scratch_ready) {
                cpu_decode_scratch_init(&decode_scratch, (uint32_t)n_tok);
                decode_scratch_ready = true;
            }
            for (uint64_t t = 0; t < n_tok; t++) {
                layer_forward_raw_swa_one(next + t * hc_dim,
                                          model,
                                          &weights->layer[il],
                                          &cache->layer[il],
                                          cur + t * hc_dim,
                                          il,
                                          (uint32_t)t,
                                          prompt->v[t],
                                          steering_dirs,
                                          steering_attn_scale,
                                          steering_ffn_scale,
                                          &decode_scratch);
            }
        }

        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    kv_cache_finish_prefill_states(cache, (uint32_t)n_tok);

    if (logits) {
        output_logits_one(logits, model, weights, cur + (n_tok - 1) * hc_dim);
    }

    if (decode_scratch_ready) cpu_decode_scratch_free(&decode_scratch);
    free(next);
    free(cur);
    free(attn);
}

/* Diagnostic first-token layer without cache history: the token attends only
 * to itself, useful for checking a minimal end-to-end slice. */
static void layer_forward_self_one(
        float                   * out_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * inp_hc,
        uint32_t                  il,
        uint32_t                  pos,
        int                       token) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;

    float *attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_cur[0]));
    float *attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_norm[0]));
    float *attn_residual = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(attn_residual[0]));
    float *q = xmalloc((size_t)q_dim * sizeof(q[0]));
    float *kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(kv[0]));
    float *heads = xmalloc((size_t)q_dim * sizeof(heads[0]));
    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    float *after_attn_hc = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(after_attn_hc[0]));
    float post[4];
    float comb[16];

    memcpy(attn_residual, inp_hc, (size_t)n_hc * DS4_N_EMBD * sizeof(inp_hc[0]));
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          attn_residual, attn_cur, post, comb);

    layer_attn_norm_one(attn_norm, model, layer, attn_cur);
    layer_q_projection_normed_one(model, layer, attn_norm, q);
    layer_kv_projection_normed_one(model, layer, attn_norm, kv);
    rope_tail_layer_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    rope_tail_layer_inplace(kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv, DS4_N_HEAD_DIM, DS4_N_ROT);
    f16_round_inplace_cpu(kv, DS4_N_HEAD_DIM);

    layer_attention_one(heads, model, layer, q, kv);
    rope_tail_layer_inplace(heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, true);
    layer_grouped_out_one(attn_out, model, layer, heads);
    hc_post_one(after_attn_hc, attn_out, attn_residual, post, comb, DS4_N_EMBD, n_hc);

    layer_ffn_one(out_hc, model, layer, after_attn_hc, il, token,
                  NULL, 0.0f, false);

    free(after_attn_hc);
    free(attn_out);
    free(heads);
    free(kv);
    free(q);
    free(attn_residual);
    free(attn_norm);
    free(attn_cur);
}

static void forward_first_token_cpu(
        float             * out_hc,
        const ds4_model   * model,
        const ds4_weights * weights,
        int                 token) {
    float *plain = xmalloc((size_t)DS4_N_EMBD * sizeof(plain[0]));
    float *cur = xmalloc((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(cur[0]));
    float *next = xmalloc((size_t)DS4_N_HC * DS4_N_EMBD * sizeof(next[0]));

    embed_token_f16(model, weights, token, plain);
    hc_from_plain_embedding(cur, plain, DS4_N_EMBD, DS4_N_HC);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        layer_forward_self_one(next, model, &weights->layer[il], cur, il, 0, token);
        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    memcpy(out_hc, cur, (size_t)DS4_N_HC * DS4_N_EMBD * sizeof(out_hc[0]));

    free(next);
    free(cur);
    free(plain);
}

/* Collapse final HC streams into the ordinary embedding vector before the
 * output norm and vocabulary projection. */
static void output_hc_head_one(
        float             * out,
        const ds4_model   * model,
        const ds4_weights * weights,
        const float       * inp_hc) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * n_hc;
    float *flat = xmalloc((size_t)hc_dim * sizeof(flat[0]));
    float *pre = xmalloc((size_t)n_hc * sizeof(pre[0]));
    float *w = xmalloc((size_t)n_hc * sizeof(w[0]));

    rms_norm_no_weight(flat, inp_hc, hc_dim, DS4_RMS_EPS);
    matvec_f16(pre, model, weights->output_hc_fn, flat);

    const float *scale = tensor_data(model, weights->output_hc_scale);
    const float *base = tensor_data(model, weights->output_hc_base);
    for (uint32_t i = 0; i < n_hc; i++) {
        w[i] = sigmoid_stable(pre[i] * scale[0] + base[i]) + DS4_HC_EPS;
    }

    hc_weighted_sum_one(out, inp_hc, w, DS4_N_EMBD, n_hc);

    free(w);
    free(pre);
    free(flat);
}

/* Final language-model head: HC collapse, RMSNorm, and Q8_0 vocab projection. */
static void output_logits_one(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        const float       * inp_hc) {
    float *embd = xmalloc((size_t)DS4_N_EMBD * sizeof(embd[0]));
    float *norm = xmalloc((size_t)DS4_N_EMBD * sizeof(norm[0]));

    output_hc_head_one(embd, model, weights, inp_hc);
    rms_norm_weight(norm, embd, tensor_data(model, weights->output_norm), DS4_N_EMBD, DS4_RMS_EPS);

    matvec_q8_0(logits, model, weights->output, norm);

    free(norm);
    free(embd);
}

static void layer_glm_first_token_attention_one(
        float                   * out,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * x) {
    float *norm = xmalloc((size_t)DS4_N_EMBD * sizeof(norm[0]));
    float *kv_raw = xmalloc((size_t)layer->attn_kv_a_mqa->dim[1] * sizeof(kv_raw[0]));
    float *kv_norm = xmalloc((size_t)DS4_N_KV_LORA * sizeof(kv_norm[0]));
    float *heads = xmalloc((size_t)DS4_N_HEAD * DS4_N_VALUE_MLA * sizeof(heads[0]));
    const uint64_t kv_blocks = (DS4_N_KV_LORA + 31) / 32;
    int8_t *kvq = xmalloc((size_t)kv_blocks * 32);
    float *kvscale = xmalloc((size_t)kv_blocks * sizeof(kvscale[0]));

    if (layer->attn_kv_a_mqa->dim[1] < DS4_N_KV_LORA ||
        layer->attn_v_b->dim[0] != DS4_N_KV_LORA ||
        layer->attn_v_b->dim[1] != DS4_N_VALUE_MLA ||
        layer->attn_v_b->dim[2] != DS4_N_HEAD ||
        layer->attn_output->dim[0] != (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA ||
        layer->attn_output->dim[1] != DS4_N_EMBD) {
        ds4_die("GLM attention tensors have an unexpected layout");
    }

    rms_norm_weight(norm, x, tensor_data(model, layer->attn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    matvec_q8_0(kv_raw, model, layer->attn_kv_a_mqa, norm);
    rms_norm_weight(kv_norm, kv_raw, tensor_data(model, layer->attn_kv_a_norm),
                    DS4_N_KV_LORA, DS4_RMS_EPS);
    quantize_q8_0_activation(kv_norm, kvq, kvscale, DS4_N_KV_LORA);

    for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
        matvec_q8_0_3d_slice_prequant(heads + (uint64_t)h * DS4_N_VALUE_MLA,
                                      model,
                                      layer->attn_v_b,
                                      kvq,
                                      kvscale,
                                      h);
    }
    matvec_q8_0(out, model, layer->attn_output, heads);

    free(kvscale);
    free(kvq);
    free(heads);
    free(kv_norm);
    free(kv_raw);
    free(norm);
}

static void layer_glm_first_token_attention_one_f32_ref(
        float                   * out,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * x) {
    float *norm = xmalloc((size_t)DS4_N_EMBD * sizeof(norm[0]));
    float *kv_raw = xmalloc((size_t)layer->attn_kv_a_mqa->dim[1] * sizeof(kv_raw[0]));
    float *kv_norm = xmalloc((size_t)DS4_N_KV_LORA * sizeof(kv_norm[0]));
    float *heads = xmalloc((size_t)DS4_N_HEAD * DS4_N_VALUE_MLA * sizeof(heads[0]));

    if (layer->attn_kv_a_mqa->type != DS4_TENSOR_Q8_0 ||
        layer->attn_v_b->type != DS4_TENSOR_Q8_0 ||
        layer->attn_output->type != DS4_TENSOR_Q8_0 ||
        layer->attn_kv_a_mqa->dim[0] != DS4_N_EMBD ||
        layer->attn_kv_a_mqa->dim[1] < DS4_N_KV_LORA ||
        layer->attn_v_b->dim[0] != DS4_N_KV_LORA ||
        layer->attn_v_b->dim[1] != DS4_N_VALUE_MLA ||
        layer->attn_v_b->dim[2] != DS4_N_HEAD ||
        layer->attn_output->dim[0] != (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA ||
        layer->attn_output->dim[1] != DS4_N_EMBD) {
        ds4_die("GLM F32 attention reference found unexpected tensor layout");
    }

    rms_norm_weight(norm, x, tensor_data(model, layer->attn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    matvec_q8_0_f32_ref(kv_raw, model, layer->attn_kv_a_mqa, norm);
    rms_norm_weight(kv_norm, kv_raw, tensor_data(model, layer->attn_kv_a_norm),
                    DS4_N_KV_LORA, DS4_RMS_EPS);
    matvec_q8_0_f32_ref(heads, model, layer->attn_v_b, kv_norm);
    matvec_q8_0_f32_ref(out, model, layer->attn_output, heads);

    free(heads);
    free(kv_norm);
    free(kv_raw);
    free(norm);
}

static void glm_k_b_project_f32_ref(
        float            * out,
        const ds4_model  * model,
        const ds4_tensor * w,
        const float      * kv_norm) {
    const uint32_t q_nope = DS4_N_KEY_MLA - DS4_N_ROT;
    if (w->type != DS4_TENSOR_Q8_0 ||
        w->ndim != 3 ||
        w->dim[0] != q_nope ||
        w->dim[1] != DS4_N_KV_LORA ||
        w->dim[2] != DS4_N_HEAD) {
        ds4_die("GLM k_b reference found unexpected tensor layout");
    }

    const uint8_t *data = tensor_data(model, w);
    const uint64_t blocks = (q_nope + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    memset(out, 0, (size_t)DS4_N_HEAD * q_nope * sizeof(out[0]));

    for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
        float *dst = out + (uint64_t)h * q_nope;
        for (uint32_t j = 0; j < DS4_N_KV_LORA; j++) {
            const uint8_t *row =
                data + ((uint64_t)h * DS4_N_KV_LORA + j) * row_bytes;
            const float xj = kv_norm[j];
            for (uint64_t b = 0; b < blocks; b++) {
                uint16_t scale_bits;
                memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
                const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
                const float d = f16_to_f32(scale_bits) * xj;
                const uint32_t i0 = (uint32_t)b * 32u;
                const uint32_t n = q_nope - i0 < 32u ? q_nope - i0 : 32u;
                for (uint32_t i = 0; i < n; i++) {
                    dst[i0 + i] += d * (float)qs[i];
                }
            }
        }
    }
}

static void layer_glm_attention_prefill_f32_ref(
        float                   * out,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * x,
        uint32_t                  n_tok,
        uint32_t                  pos0,
        uint32_t                  il) {
    if (n_tok == 0) return;
    const uint32_t qk_dim = DS4_N_KEY_MLA;
    const uint32_t q_nope = qk_dim - DS4_N_ROT;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * qk_dim;
    const uint64_t heads_dim = (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA;
    const uint64_t kv_raw_dim = layer->attn_kv_a_mqa ? layer->attn_kv_a_mqa->dim[1] : 0;

    if (!layer->attn_norm ||
        !layer->attn_q_a ||
        !layer->attn_q_a_norm ||
        !layer->attn_q_b ||
        !layer->attn_kv_a_mqa ||
        !layer->attn_kv_a_norm ||
        !layer->attn_k_b ||
        !layer->attn_v_b ||
        !layer->attn_output ||
        layer->attn_q_a->type != DS4_TENSOR_Q8_0 ||
        layer->attn_q_b->type != DS4_TENSOR_Q8_0 ||
        layer->attn_kv_a_mqa->type != DS4_TENSOR_Q8_0 ||
        layer->attn_k_b->type != DS4_TENSOR_Q8_0 ||
        layer->attn_v_b->type != DS4_TENSOR_Q8_0 ||
        layer->attn_output->type != DS4_TENSOR_Q8_0 ||
        layer->attn_norm->type != DS4_TENSOR_F32 ||
        layer->attn_q_a_norm->type != DS4_TENSOR_F32 ||
        layer->attn_kv_a_norm->type != DS4_TENSOR_F32 ||
        layer->attn_q_a->dim[0] != DS4_N_EMBD ||
        layer->attn_q_a->dim[1] != DS4_N_LORA_Q ||
        layer->attn_q_a_norm->dim[0] != DS4_N_LORA_Q ||
        layer->attn_q_b->dim[0] != DS4_N_LORA_Q ||
        layer->attn_q_b->dim[1] != q_dim ||
        layer->attn_kv_a_mqa->dim[0] != DS4_N_EMBD ||
        kv_raw_dim < (uint64_t)DS4_N_KV_LORA + DS4_N_ROT ||
        layer->attn_kv_a_norm->dim[0] != DS4_N_KV_LORA ||
        layer->attn_k_b->dim[0] != q_nope ||
        layer->attn_k_b->dim[1] != DS4_N_KV_LORA ||
        layer->attn_k_b->dim[2] != DS4_N_HEAD ||
        layer->attn_v_b->dim[0] != DS4_N_KV_LORA ||
        layer->attn_v_b->dim[1] != DS4_N_VALUE_MLA ||
        layer->attn_v_b->dim[2] != DS4_N_HEAD ||
        layer->attn_output->dim[0] != heads_dim ||
        layer->attn_output->dim[1] != DS4_N_EMBD) {
        ds4_die("GLM prefill attention reference found unexpected tensor layout");
    }

    float *norm = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(norm[0]));
    float *q_rank = xmalloc((size_t)n_tok * DS4_N_LORA_Q * sizeof(q_rank[0]));
    float *q_rank_norm = xmalloc((size_t)n_tok * DS4_N_LORA_Q * sizeof(q_rank_norm[0]));
    float *q = xmalloc((size_t)n_tok * q_dim * sizeof(q[0]));
    float *kv_raw = xmalloc((size_t)n_tok * kv_raw_dim * sizeof(kv_raw[0]));
    float *kv_norm = xmalloc((size_t)n_tok * DS4_N_KV_LORA * sizeof(kv_norm[0]));
    float *k_nope = xmalloc((size_t)n_tok * DS4_N_HEAD * q_nope * sizeof(k_nope[0]));
    float *key_cache = xmalloc((size_t)n_tok * DS4_N_HEAD * qk_dim * sizeof(key_cache[0]));
    float *value_cache = xmalloc((size_t)n_tok * heads_dim * sizeof(value_cache[0]));
    float *heads = xmalloc((size_t)n_tok * heads_dim * sizeof(heads[0]));
    float *k_rot = xmalloc((size_t)DS4_N_ROT * sizeof(k_rot[0]));
    float *scores = xmalloc((size_t)n_tok * sizeof(scores[0]));

    for (uint32_t t = 0; t < n_tok; t++) {
        const float *xt = x + (uint64_t)t * DS4_N_EMBD;
        float *norm_t = norm + (uint64_t)t * DS4_N_EMBD;
        float *qr_t = q_rank + (uint64_t)t * DS4_N_LORA_Q;
        float *qrn_t = q_rank_norm + (uint64_t)t * DS4_N_LORA_Q;
        float *q_t = q + (uint64_t)t * q_dim;
        float *raw_t = kv_raw + (uint64_t)t * kv_raw_dim;
        float *kvn_t = kv_norm + (uint64_t)t * DS4_N_KV_LORA;
        float *kn_t = k_nope + (uint64_t)t * DS4_N_HEAD * q_nope;
        float *kc_t = key_cache + (uint64_t)t * DS4_N_HEAD * qk_dim;
        float *vc_t = value_cache + (uint64_t)t * heads_dim;

        rms_norm_weight(norm_t, xt, tensor_data(model, layer->attn_norm),
                        DS4_N_EMBD, DS4_RMS_EPS);
        matvec_q8_0_f32_ref(qr_t, model, layer->attn_q_a, norm_t);
        rms_norm_weight(qrn_t, qr_t, tensor_data(model, layer->attn_q_a_norm),
                        DS4_N_LORA_Q, DS4_RMS_EPS);
        matvec_q8_0_f32_ref(q_t, model, layer->attn_q_b, qrn_t);
        rope_tail_layer_inplace(q_t, DS4_N_HEAD, qk_dim, DS4_N_ROT,
                                pos0 + t, il, false);

        matvec_q8_0_f32_ref(raw_t, model, layer->attn_kv_a_mqa, norm_t);
        rms_norm_weight(kvn_t, raw_t, tensor_data(model, layer->attn_kv_a_norm),
                        DS4_N_KV_LORA, DS4_RMS_EPS);
        glm_k_b_project_f32_ref(kn_t, model, layer->attn_k_b, kvn_t);
        matvec_q8_0_f32_ref(vc_t, model, layer->attn_v_b, kvn_t);

        memcpy(k_rot, raw_t + DS4_N_KV_LORA, (size_t)DS4_N_ROT * sizeof(k_rot[0]));
        rope_tail_layer_inplace(k_rot, 1, DS4_N_ROT, DS4_N_ROT,
                                pos0 + t, il, false);
        for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
            float *kd = kc_t + (uint64_t)h * qk_dim;
            memcpy(kd, kn_t + (uint64_t)h * q_nope,
                   (size_t)q_nope * sizeof(kd[0]));
            memcpy(kd + q_nope, k_rot, (size_t)DS4_N_ROT * sizeof(kd[0]));
        }
    }

    const float scale = 1.0f / sqrtf((float)qk_dim);
    for (uint32_t t = 0; t < n_tok; t++) {
        const uint32_t visible = t + 1u;
        for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
            const float *q_h = q + ((uint64_t)t * DS4_N_HEAD + h) * qk_dim;
            float max_score = -FLT_MAX;
            for (uint32_t s = 0; s < visible; s++) {
                const float *k_h =
                    key_cache + ((uint64_t)s * DS4_N_HEAD + h) * qk_dim;
                float dot = 0.0f;
                for (uint32_t i = 0; i < qk_dim; i++) dot += q_h[i] * k_h[i];
                scores[s] = dot * scale;
                if (scores[s] > max_score) max_score = scores[s];
            }

            float denom = 0.0f;
            for (uint32_t s = 0; s < visible; s++) {
                scores[s] = expf(scores[s] - max_score);
                denom += scores[s];
            }
            if (denom < 1.0e-20f) denom = 1.0e-20f;

            float *head_out =
                heads + ((uint64_t)t * DS4_N_HEAD + h) * DS4_N_VALUE_MLA;
            for (uint32_t d = 0; d < DS4_N_VALUE_MLA; d++) {
                float acc = 0.0f;
                for (uint32_t s = 0; s < visible; s++) {
                    const float *v_h =
                        value_cache + ((uint64_t)s * DS4_N_HEAD + h) * DS4_N_VALUE_MLA;
                    acc += scores[s] * v_h[d];
                }
                head_out[d] = acc / denom;
            }
        }

        matvec_q8_0_f32_ref(out + (uint64_t)t * DS4_N_EMBD,
                            model,
                            layer->attn_output,
                            heads + (uint64_t)t * heads_dim);
    }

    free(scores);
    free(k_rot);
    free(heads);
    free(value_cache);
    free(key_cache);
    free(k_nope);
    free(kv_norm);
    free(kv_raw);
    free(q);
    free(q_rank_norm);
    free(q_rank);
    free(norm);
}

static void layer_glm_dense_ffn_one(
        float                   * out,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * x) {
    const uint64_t hidden = layer->ffn_gate->dim[1];
    const uint64_t in_dim = layer->ffn_gate->dim[0];
    const uint64_t blocks = (in_dim + 31) / 32;
    float *gate = xmalloc((size_t)hidden * sizeof(gate[0]));
    float *up = xmalloc((size_t)hidden * sizeof(up[0]));
    float *mid = xmalloc((size_t)hidden * sizeof(mid[0]));
    int8_t *xq = xmalloc((size_t)blocks * 32);
    float *xscale = xmalloc((size_t)blocks * sizeof(xscale[0]));

    if (layer->ffn_gate->type != DS4_TENSOR_Q8_0 ||
        layer->ffn_up->type != DS4_TENSOR_Q8_0 ||
        layer->ffn_down->type != DS4_TENSOR_Q8_0 ||
        layer->ffn_up->dim[0] != in_dim ||
        layer->ffn_up->dim[1] != hidden ||
        layer->ffn_down->dim[0] != hidden ||
        layer->ffn_down->dim[1] != DS4_N_EMBD) {
        ds4_die("GLM dense FFN tensors have an unexpected layout");
    }

    quantize_q8_0_activation(x, xq, xscale, in_dim);
    matvec_q8_0_pair_prequant(gate, up, model, layer->ffn_gate, layer->ffn_up, xq, xscale);
    swiglu(mid, gate, up, hidden, 0.0f);
    matvec_q8_0(out, model, layer->ffn_down, mid);

    free(xscale);
    free(xq);
    free(mid);
    free(up);
    free(gate);
}

static void layer_glm_dense_ffn_one_f32_ref(
        float                   * out,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * x) {
    const uint64_t hidden = layer->ffn_gate->dim[1];
    const uint64_t in_dim = layer->ffn_gate->dim[0];
    float *gate = xmalloc((size_t)hidden * sizeof(gate[0]));
    float *up = xmalloc((size_t)hidden * sizeof(up[0]));
    float *mid = xmalloc((size_t)hidden * sizeof(mid[0]));

    if (layer->ffn_gate->type != DS4_TENSOR_Q8_0 ||
        layer->ffn_up->type != DS4_TENSOR_Q8_0 ||
        layer->ffn_down->type != DS4_TENSOR_Q8_0 ||
        layer->ffn_up->dim[0] != in_dim ||
        layer->ffn_up->dim[1] != hidden ||
        layer->ffn_down->dim[0] != hidden ||
        layer->ffn_down->dim[1] != DS4_N_EMBD) {
        ds4_die("GLM F32 dense FFN reference found unexpected tensor layout");
    }

    matvec_q8_0_f32_ref(gate, model, layer->ffn_gate, x);
    matvec_q8_0_f32_ref(up, model, layer->ffn_up, x);
    swiglu(mid, gate, up, hidden, 0.0f);
    matvec_q8_0_f32_ref(out, model, layer->ffn_down, mid);

    free(mid);
    free(up);
    free(gate);
}

static void layer_glm_router_selected_experts(
        int                    selected[DS4_MAX_EXPERT_USED],
        float                  expert_weight[DS4_MAX_EXPERT_USED],
        const ds4_model       *model,
        const ds4_layer_weights *layer,
        const float           *x) {
    float logits[DS4_MAX_EXPERT];
    float probs[DS4_MAX_EXPERT];
    float selection[DS4_MAX_EXPERT];
    const float *bias = tensor_data(model, layer->ffn_exp_probs_b);

    matvec_any(logits, model, layer->ffn_gate_inp, x);
    for (uint32_t i = 0; i < DS4_N_EXPERT; i++) {
        probs[i] = sigmoid_stable(logits[i]);
        selection[i] = probs[i] + bias[i];
    }

    topk_desc(selection, (int)DS4_N_EXPERT, (int)DS4_N_EXPERT_USED, selected);

    float sum = 0.0f;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        if (selected[i] < 0 || (uint32_t)selected[i] >= DS4_N_EXPERT) {
            ds4_die("GLM selected expert is outside router range");
        }
        expert_weight[i] = probs[selected[i]];
        sum += expert_weight[i];
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        expert_weight[i] = expert_weight[i] / sum * DS4_EXPERT_WEIGHT_SCALE;
    }
}

typedef struct {
    float *mid;
    const float *x;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    uint32_t gate_type;
    uint32_t up_type;
    uint32_t n_expert;
} glm_routed_moe_f32_mid_ctx;

static bool glm_graph_gate_pair_type_supported(uint32_t gate_type, uint32_t up_type) {
    return gate_type == up_type &&
           (gate_type == DS4_TENSOR_IQ2_XXS ||
            gate_type == DS4_TENSOR_Q2_K ||
            gate_type == DS4_TENSOR_Q4_K ||
            gate_type == DS4_TENSOR_Q5_K ||
            gate_type == DS4_TENSOR_MXFP4);
}

static bool glm_graph_down_type_supported(uint32_t down_type) {
    return down_type == DS4_TENSOR_IQ2_XXS ||
           down_type == DS4_TENSOR_Q2_K ||
           down_type == DS4_TENSOR_Q4_K ||
           down_type == DS4_TENSOR_Q5_K ||
           down_type == DS4_TENSOR_Q6_K ||
           down_type == DS4_TENSOR_MXFP4;
}

static float glm_routed_moe_dot_f32(uint32_t type, int n, const uint8_t *row, const float *x) {
    if (type == DS4_TENSOR_IQ2_XXS) {
        return ds4_vec_dot_iq2_xxs_f32(n, (const block_iq2_xxs *)row, x);
    }
    if (type == DS4_TENSOR_Q2_K) {
        return ds4_vec_dot_q2_K_f32(n, (const block_q2_K *)row, x);
    }
    if (type == DS4_TENSOR_Q4_K) {
        return ds4_vec_dot_q4_K_f32(n, (const block_q4_K *)row, x);
    }
    if (type == DS4_TENSOR_Q5_K || type == DS4_TENSOR_Q6_K) {
        return ds4_vec_dot_q5_q6_K_f32(type, n, row, x);
    }
    if (type == DS4_TENSOR_MXFP4) {
        return ds4_vec_dot_mxfp4_f32(n, (const block_mxfp4 *)row, x);
    }
    ds4_die("GLM F32 routed-MoE reference encountered unsupported expert tensor type");
    return 0.0f;
}

static void glm_routed_moe_f32_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    glm_routed_moe_f32_mid_ctx *ctx = vctx;

    for (uint64_t idx = row0; idx < row1; idx++) {
        const uint32_t slot = (uint32_t)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        const uint8_t *gate_row = ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot];
        const uint8_t *up_row = ctx->up_base[slot] + row * ctx->up_row_bytes[slot];
        const float gate = glm_routed_moe_dot_f32(ctx->gate_type, (int)ctx->in_dim, gate_row, ctx->x);
        const float up = glm_routed_moe_dot_f32(ctx->up_type, (int)ctx->in_dim, up_row, ctx->x);
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}

typedef struct {
    float *out;
    const float *mid;
    const uint8_t *down_base[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t down_row_bytes[DS4_MAX_EXPERT_USED];
    uint32_t down_type;
    uint32_t n_expert;
} glm_routed_moe_f32_down_ctx;

static void glm_routed_moe_f32_down_worker(void *vctx, uint64_t row0, uint64_t row1) {
    glm_routed_moe_f32_down_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        float acc = 0.0f;
        for (uint32_t slot = 0; slot < ctx->n_expert; slot++) {
            const uint8_t *down_row = ctx->down_base[slot] + row * ctx->down_row_bytes[slot];
            acc += glm_routed_moe_dot_f32(ctx->down_type,
                                          (int)ctx->in_dim,
                                          down_row,
                                          ctx->mid + (uint64_t)slot * ctx->in_dim);
        }
        ctx->out[row] = acc;
    }
}

static void layer_glm_routed_moe_one_f32_ref(
        float                   *out,
        float                   *mid_all,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        const float             *x,
        const int               *selected,
        const float             *expert_weight) {
    if (!layer->ffn_gate_exps || !layer->ffn_up_exps || !layer->ffn_down_exps ||
        !tensor_is_routed_expert_type(layer->ffn_gate_exps->type) ||
        !tensor_is_routed_expert_type(layer->ffn_up_exps->type) ||
        !glm_graph_gate_pair_type_supported(layer->ffn_gate_exps->type,
                                            layer->ffn_up_exps->type) ||
        !glm_graph_down_type_supported(layer->ffn_down_exps->type)) {
        ds4_die("GLM F32 routed-MoE reference expects supported matching routed gate/up tensors and down tensors");
    }

    glm_routed_moe_f32_mid_ctx mid_ctx = {
        .mid = mid_all,
        .x = x,
        .gate_type = layer->ffn_gate_exps->type,
        .up_type = layer->ffn_up_exps->type,
        .n_expert = DS4_N_EXPERT_USED,
    };
    glm_routed_moe_f32_down_ctx down_ctx = {
        .out = out,
        .mid = mid_all,
        .down_type = layer->ffn_down_exps->type,
        .n_expert = DS4_N_EXPERT_USED,
    };

    uint64_t gate_in0 = 0, gate_out0 = 0;
    uint64_t down_in0 = 0, down_out0 = 0;
    for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
        uint64_t gate_in, gate_out, up_in, up_out, down_in, down_out;
        if (selected[slot] < 0 || (uint32_t)selected[slot] >= DS4_N_EXPERT) {
            ds4_die("GLM F32 routed-MoE reference selected expert is outside range");
        }
        mid_ctx.gate_base[slot] =
            tensor_expert_bytes(model, layer->ffn_gate_exps, (uint32_t)selected[slot],
                                &gate_in, &gate_out, &mid_ctx.gate_row_bytes[slot]);
        mid_ctx.up_base[slot] =
            tensor_expert_bytes(model, layer->ffn_up_exps, (uint32_t)selected[slot],
                                &up_in, &up_out, &mid_ctx.up_row_bytes[slot]);
        down_ctx.down_base[slot] =
            tensor_expert_bytes(model, layer->ffn_down_exps, (uint32_t)selected[slot],
                                &down_in, &down_out, &down_ctx.down_row_bytes[slot]);
        if (gate_in != up_in || gate_out != up_out ||
            down_in != gate_out || down_out != DS4_N_EMBD) {
            ds4_die("GLM F32 routed-MoE reference found mismatched expert layouts");
        }
        if (slot == 0) {
            gate_in0 = gate_in;
            gate_out0 = gate_out;
            down_in0 = down_in;
            down_out0 = down_out;
        } else if (gate_in != gate_in0 || gate_out != gate_out0 ||
                   down_in != down_in0 || down_out != down_out0) {
            ds4_die("GLM F32 routed-MoE reference expert layouts are not uniform");
        }
        mid_ctx.expert_weight[slot] = expert_weight[slot];
    }

    if (gate_in0 != DS4_N_EMBD || gate_in0 % QK_K != 0 ||
        down_in0 != DS4_N_FF_EXP || down_in0 % QK_K != 0 ||
        gate_out0 != DS4_N_FF_EXP || down_out0 != DS4_N_EMBD) {
        ds4_die("GLM F32 routed-MoE reference found unexpected GLM expert dimensions");
    }

    mid_ctx.in_dim = gate_in0;
    mid_ctx.out_dim = gate_out0;
    down_ctx.in_dim = down_in0;
    down_ctx.out_dim = down_out0;
    ds4_parallel_for((uint64_t)DS4_N_EXPERT_USED * gate_out0,
                     glm_routed_moe_f32_mid_worker,
                     &mid_ctx);
    ds4_parallel_for(down_out0, glm_routed_moe_f32_down_worker, &down_ctx);
}

static void layer_glm_shared_ffn_one_f32_ref(
        float                   *out,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        const float             *x) {
    const uint64_t in_dim = layer->ffn_gate_shexp ? layer->ffn_gate_shexp->dim[0] : 0;
    const uint64_t hidden = layer->ffn_gate_shexp ? layer->ffn_gate_shexp->dim[1] : 0;
    if (!layer->ffn_gate_shexp ||
        !layer->ffn_up_shexp ||
        !layer->ffn_down_shexp ||
        layer->ffn_gate_shexp->type != DS4_TENSOR_Q8_0 ||
        layer->ffn_up_shexp->type != DS4_TENSOR_Q8_0 ||
        layer->ffn_down_shexp->type != DS4_TENSOR_Q8_0 ||
        layer->ffn_up_shexp->dim[0] != in_dim ||
        layer->ffn_up_shexp->dim[1] != hidden ||
        layer->ffn_down_shexp->dim[0] != hidden ||
        layer->ffn_down_shexp->dim[1] != DS4_N_EMBD ||
        in_dim != DS4_N_EMBD ||
        hidden != DS4_N_FF_EXP) {
        ds4_die("GLM F32 shared expert reference found unexpected tensor layout");
    }

    float *gate = xmalloc((size_t)hidden * sizeof(gate[0]));
    float *up = xmalloc((size_t)hidden * sizeof(up[0]));
    float *mid = xmalloc((size_t)hidden * sizeof(mid[0]));

    matvec_q8_0_f32_ref(gate, model, layer->ffn_gate_shexp, x);
    matvec_q8_0_f32_ref(up, model, layer->ffn_up_shexp, x);
    swiglu(mid, gate, up, hidden, 0.0f);
    matvec_q8_0_f32_ref(out, model, layer->ffn_down_shexp, mid);

    free(mid);
    free(up);
    free(gate);
}

static void layer_glm_ffn_one_f32_ref(
        float                   *out,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        const float             *x,
        uint32_t                  il) {
    float *norm = xmalloc((size_t)DS4_N_EMBD * sizeof(norm[0]));

    rms_norm_weight(norm, x, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    if (il < DS4_N_LEADING_DENSE) {
        layer_glm_dense_ffn_one_f32_ref(out, model, layer, norm);
    } else {
        int selected[DS4_MAX_EXPERT_USED];
        float expert_weight[DS4_MAX_EXPERT_USED];
        float *mid = xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(mid[0]));
        float *moe = xmalloc((size_t)DS4_N_EMBD * sizeof(moe[0]));
        float *shared = xmalloc((size_t)DS4_N_EMBD * sizeof(shared[0]));

        layer_glm_router_selected_experts(selected, expert_weight, model, layer, norm);
        layer_glm_routed_moe_one_f32_ref(moe, mid, model, layer, norm,
                                         selected, expert_weight);
        layer_glm_shared_ffn_one_f32_ref(shared, model, layer, norm);
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) out[i] = moe[i] + shared[i];

        free(shared);
        free(moe);
        free(mid);
    }

    free(norm);
}

static void layer_glm_first_token_one_f32_ref(
        float                   *out,
        const ds4_model         *model,
        const ds4_layer_weights *layer,
        const float             *x,
        uint32_t                  il) {
    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    float *after_attn = xmalloc((size_t)DS4_N_EMBD * sizeof(after_attn[0]));
    float *ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(ffn_out[0]));

    layer_glm_first_token_attention_one_f32_ref(attn_out, model, layer, x);
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) after_attn[i] = x[i] + attn_out[i];

    layer_glm_ffn_one_f32_ref(ffn_out, model, layer, after_attn, il);
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) out[i] = after_attn[i] + ffn_out[i];

    free(ffn_out);
    free(after_attn);
    free(attn_out);
}

static void forward_glm_first_token_cpu_f32_ref(
        float             *out_hidden,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                token) {
    float *cur = xmalloc((size_t)DS4_N_EMBD * sizeof(cur[0]));
    float *next = xmalloc((size_t)DS4_N_EMBD * sizeof(next[0]));
    const uint32_t normal_layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;

    embed_token_any(model, weights, token, cur);
    for (uint32_t il = 0; il < normal_layers; il++) {
        layer_glm_first_token_one_f32_ref(next, model, &weights->layer[il], cur, il);
        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    memcpy(out_hidden, cur, (size_t)DS4_N_EMBD * sizeof(out_hidden[0]));

    free(next);
    free(cur);
}

static void output_logits_glm_one_f32_ref(
        float             *logits,
        const ds4_model   *model,
        const ds4_weights *weights,
        const float       *hidden) {
    float *norm = xmalloc((size_t)DS4_N_EMBD * sizeof(norm[0]));

    rms_norm_weight(norm, hidden, tensor_data(model, weights->output_norm), DS4_N_EMBD, DS4_RMS_EPS);
    matvec_q8_0_f32_ref(logits, model, weights->output, norm);

    free(norm);
}

static void layer_glm_routed_moe_one(
        float                   * out,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * x,
        uint32_t                  il) {
    int selected[DS4_MAX_EXPERT_USED];
    float expert_weight[DS4_MAX_EXPERT_USED];
    const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
    const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
    float *mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(mid_all[0]));
    block_q8_K *xq = xmalloc((size_t)(expert_in_dim / QK_K) * sizeof(xq[0]));
    block_q8_K *midq = xmalloc((size_t)DS4_N_EXPERT_USED * (down_in_dim / QK_K) * sizeof(midq[0]));

    if (expert_in_dim != DS4_N_EMBD || expert_in_dim % QK_K != 0 ||
        down_in_dim != DS4_N_FF_EXP || down_in_dim % QK_K != 0) {
        ds4_die("GLM routed expert tensors have an unexpected layout");
    }

    memset(out, 0, (size_t)DS4_N_EMBD * sizeof(out[0]));
    ds4_quantize_row_q8_K(x, xq, (int64_t)expert_in_dim);
    layer_glm_router_selected_experts(selected, expert_weight, model, layer, x);

    matvec_experts_mid_prequant(mid_all, model,
                                layer->ffn_gate_exps,
                                layer->ffn_up_exps,
                                xq,
                                selected,
                                expert_weight,
                                DS4_N_EXPERT_USED,
                                0.0f);
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
        ds4_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
                              midq + (uint64_t)i * (down_in_dim / QK_K),
                              (int64_t)down_in_dim);
    }
    matvec_experts_down_accum_prequant(out, model, layer->ffn_down_exps, midq,
                                       selected, DS4_N_EXPERT_USED);

    free(midq);
    free(xq);
    free(mid_all);
    (void)il;
}

static void layer_glm_sparse_ffn_one(
        float                   * out,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * x,
        uint32_t                  il) {
    float *moe = xmalloc((size_t)DS4_N_EMBD * sizeof(moe[0]));
    float *shared = xmalloc((size_t)DS4_N_EMBD * sizeof(shared[0]));

    layer_glm_routed_moe_one(moe, model, layer, x, il);
    layer_shared_ffn_one(shared, model, layer, x);
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) out[i] = moe[i] + shared[i];

    free(shared);
    free(moe);
}

static void layer_glm_ffn_one(
        float                   * out,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * x,
        uint32_t                  il) {
    float *norm = xmalloc((size_t)DS4_N_EMBD * sizeof(norm[0]));

    rms_norm_weight(norm, x, tensor_data(model, layer->ffn_norm), DS4_N_EMBD, DS4_RMS_EPS);
    if (il < DS4_N_LEADING_DENSE) {
        layer_glm_dense_ffn_one(out, model, layer, norm);
    } else {
        layer_glm_sparse_ffn_one(out, model, layer, norm, il);
    }

    free(norm);
}

static void layer_glm_first_token_one(
        float                   * out,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * x,
        uint32_t                  il) {
    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    float *after_attn = xmalloc((size_t)DS4_N_EMBD * sizeof(after_attn[0]));
    float *ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(ffn_out[0]));

    layer_glm_first_token_attention_one(attn_out, model, layer, x);
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) after_attn[i] = x[i] + attn_out[i];

    layer_glm_ffn_one(ffn_out, model, layer, after_attn, il);
    for (uint32_t i = 0; i < DS4_N_EMBD; i++) out[i] = after_attn[i] + ffn_out[i];

    free(ffn_out);
    free(after_attn);
    free(attn_out);
}

static void forward_glm_first_token_cpu(
        float             * out_hidden,
        const ds4_model   * model,
        const ds4_weights * weights,
        int                 token) {
    float *cur = xmalloc((size_t)DS4_N_EMBD * sizeof(cur[0]));
    float *next = xmalloc((size_t)DS4_N_EMBD * sizeof(next[0]));
    const uint32_t normal_layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;

    embed_token_any(model, weights, token, cur);
    for (uint32_t il = 0; il < normal_layers; il++) {
        layer_glm_first_token_one(next, model, &weights->layer[il], cur, il);
        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    memcpy(out_hidden, cur, (size_t)DS4_N_EMBD * sizeof(out_hidden[0]));

    free(next);
    free(cur);
}

static void output_logits_glm_one(
        float             * logits,
        const ds4_model   * model,
        const ds4_weights * weights,
        const float       * hidden) {
    float *norm = xmalloc((size_t)DS4_N_EMBD * sizeof(norm[0]));

    rms_norm_weight(norm, hidden, tensor_data(model, weights->output_norm), DS4_N_EMBD, DS4_RMS_EPS);
    matvec_q8_0(logits, model, weights->output, norm);

    free(norm);
}

/* Allocation-free logits head for CPU decode. */
static void output_logits_one_decode_scratch(
        float                  * logits,
        const ds4_model        * model,
        const ds4_weights      * weights,
        const float            * inp_hc,
        ds4_cpu_decode_scratch * scratch) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * n_hc;

    rms_norm_no_weight(scratch->output_flat, inp_hc, hc_dim, DS4_RMS_EPS);
    matvec_f16(scratch->output_pre, model, weights->output_hc_fn, scratch->output_flat);

    const float *scale = tensor_data(model, weights->output_hc_scale);
    const float *base = tensor_data(model, weights->output_hc_base);
    for (uint32_t i = 0; i < n_hc; i++) {
        scratch->output_weights[i] = sigmoid_stable(scratch->output_pre[i] * scale[0] + base[i]) + DS4_HC_EPS;
    }

    hc_weighted_sum_one(scratch->output_embd, inp_hc, scratch->output_weights, DS4_N_EMBD, n_hc);
    rms_norm_weight(scratch->output_norm, scratch->output_embd,
                    tensor_data(model, weights->output_norm),
                    DS4_N_EMBD, DS4_RMS_EPS);
    matvec_q8_0_decode_scratch(logits, model, weights->output, scratch->output_norm, scratch);
}

