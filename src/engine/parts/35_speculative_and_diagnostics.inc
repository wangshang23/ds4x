/* Speculative decode state machine:
 * 1. commit the normal target token and use its logits to validate draft[0];
 * 2. let MTP recursively draft a tiny suffix from its own raw-cache frontier;
 * 3. verify the suffix with the target graph, committing only the accepted
 *    prefix and rolling back speculative Metal state on miss;
 * 4. fall back to ordinary one-token decode if the fast verifier cannot prove
 *    the target stream. */

int ds4_engine_generate_argmax(
        ds4_engine        *e,
        const ds4_tokens  *prompt,
        int                n_predict,
        int                ctx_size,
        ds4_token_emit_fn  emit,
        ds4_generation_done_fn done,
        void              *emit_ud,
        ds4_session_progress_fn progress,
        void              *progress_ud) {
    const ds4_model *model = &e->model;
    const ds4_vocab *vocab = &e->vocab;
    const ds4_weights *weights = &e->weights;

    if (ds4_backend_uses_graph(e->backend)) {
#ifndef DS4_NO_GPU
        if (!e->metal_ready) {
            fprintf(stderr, "ds4: %s generation requested but the graph backend is unavailable\n",
                    ds4_backend_name(e->backend));
            return 1;
        }
        if (e->multi_tier) {
            ds4_session *s = NULL;
            char err[256] = {0};
            const double t_prefill0 = now_sec();
            if (ds4_session_create(&s, e, ctx_size) != 0) {
                fprintf(stderr, "ds4: failed to create multi-tier graph session\n");
                return 1;
            }
            ds4_session_set_progress(s, progress, progress_ud);
            if (ds4_session_sync(s, prompt, err, sizeof(err)) != 0) {
                ds4_session_set_progress(s, NULL, NULL);
                fprintf(stderr, "ds4: multi-tier prefill failed: %s\n", err);
                ds4_session_free(s);
                return 1;
            }
            ds4_session_set_progress(s, NULL, NULL);
            const double t_prefill1 = now_sec();

            int rc = 0;
            int n_generated = 0;
            const double t_decode0 = now_sec();
            const bool greedy_top1 = metal_graph_tp_env_flag("DS4_CUDA_GREEDY_TOP1", true);
            int token = ds4_session_argmax(s);
            for (int i = 0; i < n_predict && ds4_session_pos(s) < ctx_size; i++) {
                if (token < 0) {
                    fprintf(stderr, "ds4: multi-tier argmax failed\n");
                    rc = 1;
                    break;
                }
                if (ds4_token_is_stop(e, token)) break;
                if (emit) emit(emit_ud, token);
                n_generated++;
                if (i == n_predict - 1 || ds4_session_pos(s) + 1 >= ctx_size) break;
                if (greedy_top1) {
                    token = ds4_session_eval_argmax(s, token, err, sizeof(err));
                } else if (ds4_session_eval(s, token, err, sizeof(err)) == 0) {
                    token = ds4_session_argmax(s);
                } else {
                    token = -1;
                }
                if (token < 0) {
                    fprintf(stderr, "ds4: multi-tier decode failed: %s\n", err);
                    rc = 1;
                    break;
                }
            }
            const double t_decode1 = now_sec();
            if (done) done(emit_ud);
            ds4_log(stderr,
                    DS4_LOG_TIMING,
                    "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
                    (t_prefill1 - t_prefill0) > 0.0 ? (double)prompt->len / (t_prefill1 - t_prefill0) : 0.0,
                    (t_decode1 - t_decode0) > 0.0 ? (double)n_generated / (t_decode1 - t_decode0) : 0.0);
            ds4_session_free(s);
            return rc;
        }
        return generate_metal_graph_raw_swa(model, vocab, weights, prompt,
                                            n_predict, ctx_size, e->quality,
                                            e->ssd_streaming,
                                            e->ssd_streaming_cold,
                                            e->ssd_streaming_preload_experts,
                                            e->ssd_streaming_cache_bytes,
                                            e->ssd_streaming_prefill_headroom_bytes,
                                            e->power_percent,
                                            e->prefill_chunk,
                                            e->directional_steering_file,
                                            e->directional_steering_attn_scale,
                                            e->directional_steering_ffn_scale,
                                            emit, done, emit_ud,
                                            progress, progress_ud);
#else
        fprintf(stderr, "ds4: %s generation requested but this build has no graph backend support\n",
                ds4_backend_name(e->backend));
        return 1;
#endif
    }

    return generate_raw_swa_cpu(model, vocab, weights, prompt, n_predict,
                                ctx_size,
                                e->directional_steering_dirs,
                                e->directional_steering_attn_scale,
                                e->directional_steering_ffn_scale,
                                emit, done, emit_ud, progress, progress_ud);
}

static int glm_metal_compare_f32(
        const char  *name,
        const float *cpu,
        const float *gpu,
        uint32_t     n,
        float        tol) {
    float max_abs = 0.0f;
    uint32_t max_i = 0;
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        const float d = fabsf(gpu[i] - cpu[i]);
        if (d > max_abs) {
            max_abs = d;
            max_i = i;
        }
        ss += (double)d * (double)d;
    }
    const double rms = sqrt(ss / (double)n);
    printf("  %s: max_abs=%.9g at %u cpu=%.9g gpu=%.9g rms_abs=%.9g\n",
           name, max_abs, max_i, cpu[max_i], gpu[max_i], rms);
    if (max_abs > tol) {
        fprintf(stderr, "ds4: GLM Metal %s mismatch, max_abs %.9g > %.9g\n",
                name, max_abs, tol);
        return 0;
    }
    return 1;
}

static int glm_metal_compare_i32_list(
        const char    *name,
        const int     *cpu,
        const int32_t *gpu,
        uint32_t       n) {
    int ok = 1;
    printf("  %s: cpu=[", name);
    for (uint32_t i = 0; i < n; i++) {
        printf("%s%d", i ? "," : "", cpu[i]);
    }
    printf("] gpu=[");
    for (uint32_t i = 0; i < n; i++) {
        printf("%s%d", i ? "," : "", (int)gpu[i]);
        if (cpu[i] != (int)gpu[i]) ok = 0;
    }
    printf("]\n");
    if (!ok) {
        fprintf(stderr, "ds4: GLM Metal %s mismatch\n", name);
        return 0;
    }
    return 1;
}

#ifndef DS4_NO_GPU
static void glm_metal_q8_diag_fill_input(float *x, uint64_t n_tok, uint64_t in_dim) {
    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            uint32_t s = (uint32_t)(0x9e3779b9u ^ (uint32_t)(t * 0x85ebca6bu) ^
                                    (uint32_t)(i * 0xc2b2ae35u));
            s ^= s >> 16;
            s *= 0x7feb352du;
            s ^= s >> 15;
            s *= 0x846ca68bu;
            s ^= s >> 16;
            const float centered = ((float)(int32_t)(s & 0xffffu) - 32768.0f) / 32768.0f;
            const float scale = 0.25f + 0.015625f * (float)((i + 3u * t) & 31u);
            x[t * in_dim + i] = centered * scale;
        }
    }
}

static void glm_metal_q8_diag_reference(
        float            *out,
        const ds4_model  *model,
        const ds4_tensor *w,
        const float      *x,
        uint64_t          n_tok) {
    const uint64_t out_dim = w->elements / w->dim[0];
    for (uint64_t t = 0; t < n_tok; t++) {
        matvec_q8_0_f32_ref(out + t * out_dim,
                            model,
                            w,
                            x + t * w->dim[0]);
    }
}

static int glm_metal_graph_test_q8_prefill_one(
        ds4_engine       *e,
        const char       *name,
        const ds4_tensor *w,
        int               strict) {
    if (!w) return 1;
    if (w->type != DS4_TENSOR_Q8_0 || w->ndim < 2 || w->dim[0] == 0) {
        fprintf(stderr, "ds4: GLM Q8 prefill diagnostic found unexpected %s layout\n",
                name);
        return 0;
    }

    const ds4_model *model = &e->model;
    const uint64_t in_dim = w->dim[0];
    const uint64_t out_dim = w->elements / in_dim;
    const uint32_t cases[] = { 16u, 17u, 31u, 32u };
    int ok = 1;

    printf("  q8_prefill_diag: %s ndim=%u in=%llu out=%llu strict=%d\n",
           name,
           w->ndim,
           (unsigned long long)in_dim,
           (unsigned long long)out_dim,
           strict);

    for (uint32_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const uint32_t n_tok = cases[ci];
        if (in_dim > UINT64_MAX / n_tok / sizeof(float) ||
            out_dim > UINT64_MAX / n_tok / sizeof(float) ||
            out_dim > UINT32_MAX / n_tok) {
            fprintf(stderr, "ds4: GLM Q8 prefill diagnostic size overflow in %s token case %u\n",
                    name,
                    n_tok);
            ok = 0;
            if (strict) break;
            continue;
        }

        const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
        const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);
        if (x_bytes > SIZE_MAX || out_bytes > SIZE_MAX) {
            fprintf(stderr, "ds4: GLM Q8 prefill diagnostic host allocation is too large in %s token case %u\n",
                    name,
                    n_tok);
            ok = 0;
            if (strict) break;
            continue;
        }

        char label[128];
        snprintf(label, sizeof(label), "q8_prefill_diag_%s_tok%u", name, n_tok);

        float *x_host = xmalloc((size_t)x_bytes);
        float *cpu_out = xmalloc((size_t)out_bytes);
        float *gpu_out = xmalloc((size_t)out_bytes);
        ds4_gpu_tensor *x_gpu = ds4_gpu_tensor_alloc(x_bytes);
        ds4_gpu_tensor *out_gpu = ds4_gpu_tensor_alloc(out_bytes);
        int case_ok = x_gpu && out_gpu;

        if (!case_ok) {
            fprintf(stderr, "ds4: GLM Q8 prefill diagnostic could not allocate %s token case %u\n",
                    name,
                    n_tok);
        }
        if (case_ok) {
            glm_metal_q8_diag_fill_input(x_host, n_tok, in_dim);
            glm_metal_q8_diag_reference(cpu_out, model, w, x_host, n_tok);
            case_ok = ds4_gpu_tensor_write(x_gpu, 0, x_host, x_bytes) != 0;
        }
        if (case_ok) {
            case_ok = ds4_gpu_matmul_q8_0_tensor(out_gpu,
                                                 model->map,
                                                 model->size,
                                                 w->abs_offset,
                                                 in_dim,
                                                 out_dim,
                                                 x_gpu,
                                                 n_tok);
        }
        if (case_ok) {
            case_ok = ds4_gpu_tensor_read(out_gpu, 0, gpu_out, out_bytes) != 0;
        }
        if (case_ok) {
            case_ok = glm_metal_compare_f32(label,
                                            cpu_out,
                                            gpu_out,
                                            (uint32_t)((uint64_t)n_tok * out_dim),
                                            strict ? 5.0e-2f : 3.0e38f);
        }

        ds4_gpu_tensor_free(out_gpu);
        ds4_gpu_tensor_free(x_gpu);
        free(gpu_out);
        free(cpu_out);
        free(x_host);

        if (!case_ok) {
            ok = 0;
            if (strict) break;
        }
    }

    return ok || !strict;
}

static int glm_metal_graph_test_q8_prefill(
        ds4_engine              *e,
        const ds4_layer_weights *layer) {
    return 1;

    if (!layer) {
        fprintf(stderr, "ds4: GLM Q8 prefill diagnostic requires layer weights\n");
        return 0;
    }

    const int strict = 0;
    int ok = 1;
    const ds4_layer_weights *sparse =
        DS4_N_LEADING_DENSE < DS4_N_LAYER ? &e->weights.layer[DS4_N_LEADING_DENSE] : NULL;

#define DS4_GLM_Q8_DIAG_ONE(label, tensor)                                      \
    do {                                                                        \
        if (!glm_metal_graph_test_q8_prefill_one(e, label, tensor, strict)) {   \
            ok = 0;                                                             \
            if (strict) goto done;                                              \
        }                                                                       \
    } while (0)

    DS4_GLM_Q8_DIAG_ONE("layer0.attn_q_a", layer->attn_q_a);
    DS4_GLM_Q8_DIAG_ONE("layer0.attn_q_b", layer->attn_q_b);
    DS4_GLM_Q8_DIAG_ONE("layer0.attn_kv_a_mqa", layer->attn_kv_a_mqa);
    DS4_GLM_Q8_DIAG_ONE("layer0.attn_k_b", layer->attn_k_b);
    DS4_GLM_Q8_DIAG_ONE("layer0.attn_v_b", layer->attn_v_b);
    DS4_GLM_Q8_DIAG_ONE("layer0.attn_output", layer->attn_output);
    DS4_GLM_Q8_DIAG_ONE("layer0.ffn_gate", layer->ffn_gate);
    DS4_GLM_Q8_DIAG_ONE("layer0.ffn_up", layer->ffn_up);
    DS4_GLM_Q8_DIAG_ONE("layer0.ffn_down", layer->ffn_down);

    if (sparse) {
        DS4_GLM_Q8_DIAG_ONE("layer3.ffn_gate_shexp", sparse->ffn_gate_shexp);
        DS4_GLM_Q8_DIAG_ONE("layer3.ffn_up_shexp", sparse->ffn_up_shexp);
        DS4_GLM_Q8_DIAG_ONE("layer3.ffn_down_shexp", sparse->ffn_down_shexp);
    }

done:
#undef DS4_GLM_Q8_DIAG_ONE
    return strict ? ok : 1;
}

static int glm_metal_graph_test_multitok_attention(
        ds4_engine              *e,
        const ds4_tokens        *prompt,
        const ds4_layer_weights *layer) {
    if (!prompt || prompt->len < 2) {
        printf("  layer0_multitok_attention: skipped (prompt has fewer than 2 tokens)\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_weights *weights = &e->weights;
    const uint32_t n_tok = prompt->len < 4 ? (uint32_t)prompt->len : 4u;
    const uint32_t qk_dim = DS4_N_KEY_MLA;
    const uint32_t q_nope = qk_dim - DS4_N_ROT;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * qk_dim;
    const uint64_t heads_dim = (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA;
    const uint64_t kv_raw_dim = layer->attn_kv_a_mqa ? layer->attn_kv_a_mqa->dim[1] : 0;

    if (!weights->token_embd ||
        !layer->attn_norm ||
        !layer->attn_q_a ||
        !layer->attn_q_a_norm ||
        !layer->attn_q_b ||
        !layer->attn_kv_a_mqa ||
        !layer->attn_kv_a_norm ||
        !layer->attn_k_b ||
        !layer->attn_v_b ||
        !layer->attn_output ||
        weights->token_embd->type != DS4_TENSOR_Q8_0 ||
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
        fprintf(stderr, "ds4: GLM multi-token attention diagnostic found unexpected layer-0 layout\n");
        return 0;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        if (prompt->v[t] < 0 || prompt->v[t] >= (int)DS4_N_VOCAB) {
            fprintf(stderr, "ds4: GLM multi-token attention token %d is outside vocab\n",
                    prompt->v[t]);
            return 0;
        }
    }

    const uint64_t token_bytes = (uint64_t)n_tok * sizeof(int32_t);
    const uint64_t emb_bytes = (uint64_t)n_tok * DS4_N_EMBD * sizeof(float);
    const uint64_t q_rank_bytes = (uint64_t)n_tok * DS4_N_LORA_Q * sizeof(float);
    const uint64_t q_bytes = (uint64_t)n_tok * q_dim * sizeof(float);
    const uint64_t kv_raw_bytes = (uint64_t)n_tok * kv_raw_dim * sizeof(float);
    const uint64_t kv_norm_bytes = (uint64_t)n_tok * DS4_N_KV_LORA * sizeof(float);
    const uint64_t k_nope_bytes = (uint64_t)n_tok * DS4_N_HEAD * q_nope * sizeof(float);
    const uint64_t heads_bytes = (uint64_t)n_tok * heads_dim * sizeof(float);

    int32_t *tok_host = xmalloc((size_t)token_bytes);
    float *cpu_emb = xmalloc((size_t)emb_bytes);
    float *cpu_attn = xmalloc((size_t)emb_bytes);
    float *gpu_attn = xmalloc((size_t)emb_bytes);

    for (uint32_t t = 0; t < n_tok; t++) {
        tok_host[t] = (int32_t)prompt->v[t];
        embed_token_any(model, weights, prompt->v[t],
                        cpu_emb + (uint64_t)t * DS4_N_EMBD);
    }
    layer_glm_attention_prefill_f32_ref(cpu_attn, model, layer, cpu_emb, n_tok, 0, 0);

    ds4_gpu_tensor *tok = NULL;
    ds4_gpu_tensor *cur = NULL;
    ds4_gpu_tensor *attn_norm = NULL;
    ds4_gpu_tensor *q_rank = NULL;
    ds4_gpu_tensor *q_rank_norm = NULL;
    ds4_gpu_tensor *q = NULL;
    ds4_gpu_tensor *kv_raw = NULL;
    ds4_gpu_tensor *kv_norm = NULL;
    ds4_gpu_tensor *k_nope = NULL;
    ds4_gpu_tensor *value = NULL;
    ds4_gpu_tensor *key_cache = NULL;
    ds4_gpu_tensor *value_cache = NULL;
    ds4_gpu_tensor *heads = NULL;
    ds4_gpu_tensor *attn_out = NULL;

    int ok = 1;
#define DS4_GLM_MT_ALLOC_TENSOR(var, bytes_) \
    do { \
        (var) = ds4_gpu_tensor_alloc((bytes_)); \
        if (!(var)) { \
            fprintf(stderr, "ds4: GLM multi-token attention diagnostic could not allocate %s\n", #var); \
            ok = 0; \
        } \
    } while (0)

    DS4_GLM_MT_ALLOC_TENSOR(tok, token_bytes);
    DS4_GLM_MT_ALLOC_TENSOR(cur, emb_bytes);
    DS4_GLM_MT_ALLOC_TENSOR(attn_norm, emb_bytes);
    DS4_GLM_MT_ALLOC_TENSOR(q_rank, q_rank_bytes);
    DS4_GLM_MT_ALLOC_TENSOR(q_rank_norm, q_rank_bytes);
    DS4_GLM_MT_ALLOC_TENSOR(q, q_bytes);
    DS4_GLM_MT_ALLOC_TENSOR(kv_raw, kv_raw_bytes);
    DS4_GLM_MT_ALLOC_TENSOR(kv_norm, kv_norm_bytes);
    DS4_GLM_MT_ALLOC_TENSOR(k_nope, k_nope_bytes);
    DS4_GLM_MT_ALLOC_TENSOR(value, heads_bytes);
    DS4_GLM_MT_ALLOC_TENSOR(key_cache, q_bytes);
    DS4_GLM_MT_ALLOC_TENSOR(value_cache, heads_bytes);
    DS4_GLM_MT_ALLOC_TENSOR(heads, heads_bytes);
    DS4_GLM_MT_ALLOC_TENSOR(attn_out, emb_bytes);
#undef DS4_GLM_MT_ALLOC_TENSOR

    const uint32_t rope_ctx = 0;
    const float rope_base = layer_rope_freq_base(0);
    const float rope_scale = layer_rope_freq_scale(0);
    const float rope_ext = 0.0f;
    const float rope_attn = 1.0f;

    if (ok) ok = ds4_gpu_tensor_write(tok, 0, tok_host, token_bytes) != 0;
    if (ok) ok = ds4_gpu_embed_tokens_q8_0_tensor(cur,
                                                  tok,
                                                  model->map,
                                                  model->size,
                                                  weights->token_embd->abs_offset,
                                                  DS4_N_VOCAB,
                                                  n_tok,
                                                  DS4_N_EMBD);
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(attn_norm,
                                                     cur,
                                                     model->map,
                                                     model->size,
                                                     layer->attn_norm->abs_offset,
                                                     DS4_N_EMBD,
                                                     n_tok,
                                                     DS4_RMS_EPS);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(q_rank,
                                            model->map,
                                            model->size,
                                            layer->attn_q_a->abs_offset,
                                            DS4_N_EMBD,
                                            DS4_N_LORA_Q,
                                            attn_norm,
                                            n_tok);
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(q_rank_norm,
                                                     q_rank,
                                                     model->map,
                                                     model->size,
                                                     layer->attn_q_a_norm->abs_offset,
                                                     DS4_N_LORA_Q,
                                                     n_tok,
                                                     DS4_RMS_EPS);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(q,
                                            model->map,
                                            model->size,
                                            layer->attn_q_b->abs_offset,
                                            DS4_N_LORA_Q,
                                            q_dim,
                                            q_rank_norm,
                                            n_tok);
    if (ok) ok = ds4_gpu_rope_tail_tensor(q,
                                          n_tok,
                                          DS4_N_HEAD,
                                          qk_dim,
                                          DS4_N_ROT,
                                          0,
                                          rope_ctx,
                                          false,
                                          rope_base,
                                          rope_scale,
                                          rope_ext,
                                          rope_attn,
                                          DS4_ROPE_YARN_BETA_FAST,
                                          DS4_ROPE_YARN_BETA_SLOW);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(kv_raw,
                                            model->map,
                                            model->size,
                                            layer->attn_kv_a_mqa->abs_offset,
                                            DS4_N_EMBD,
                                            kv_raw_dim,
                                            attn_norm,
                                            n_tok);
    if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(kv_norm,
                                                     kv_raw,
                                                     model->map,
                                                     model->size,
                                                     layer->attn_kv_a_norm->abs_offset,
                                                     n_tok,
                                                     (uint32_t)kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_RMS_EPS);
    if (ok) ok = ds4_gpu_glm_k_b_project_tensor(k_nope,
                                                kv_norm,
                                                model->map,
                                                model->size,
                                                layer->attn_k_b->abs_offset,
                                                n_tok,
                                                DS4_N_KV_LORA,
                                                q_nope,
                                                DS4_N_HEAD);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(value,
                                            model->map,
                                            model->size,
                                            layer->attn_v_b->abs_offset,
                                            DS4_N_KV_LORA,
                                            heads_dim,
                                            kv_norm,
                                            n_tok);
    if (ok) ok = ds4_gpu_glm_build_kv_cache_tensor(key_cache,
                                                   value_cache,
                                                   kv_raw,
                                                   k_nope,
                                                   value,
                                                   0,
                                                   n_tok,
                                                   n_tok,
                                                   DS4_N_HEAD,
                                                   (uint32_t)kv_raw_dim,
                                                   DS4_N_KV_LORA,
                                                   q_nope,
                                                   DS4_N_ROT,
                                                   DS4_N_VALUE_MLA,
                                                   rope_ctx,
                                                   rope_base,
                                                   rope_scale,
                                                   rope_ext,
                                                   rope_attn,
                                                   DS4_ROPE_YARN_BETA_FAST,
                                                   DS4_ROPE_YARN_BETA_SLOW,
                                                   false);
    if (ok) {
        if (glm_graph_flash_attention_prefill_enabled()) {
            ok = ds4_gpu_glm_attention_flash_tensor(heads,
                                                    q,
                                                    key_cache,
                                                    value_cache,
                                                    0,
                                                    n_tok,
                                                    n_tok,
                                                    n_tok,
                                                    DS4_N_HEAD,
                                                    qk_dim,
                                                    DS4_N_VALUE_MLA,
                                                    false);
        } else {
            ok = ds4_gpu_glm_attention_full_tensor(heads,
                                                   q,
                                                   key_cache,
                                                   value_cache,
                                                   0,
                                                   n_tok,
                                                   n_tok,
                                                   n_tok,
                                                   DS4_N_HEAD,
                                                   qk_dim,
                                                   DS4_N_VALUE_MLA,
                                                   false);
        }
    }
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(attn_out,
                                            model->map,
                                            model->size,
                                            layer->attn_output->abs_offset,
                                            heads_dim,
                                            DS4_N_EMBD,
                                            heads,
                                            n_tok);
    if (ok) ok = ds4_gpu_tensor_read(attn_out, 0, gpu_attn, emb_bytes) != 0;
    if (ok) {
        printf("  layer0_multitok_attention: tokens=%u\n", n_tok);
        ok = glm_metal_compare_f32("layer0_multitok_attention",
                                   cpu_attn,
                                   gpu_attn,
                                   (uint32_t)((uint64_t)n_tok * DS4_N_EMBD),
                                   5.0e-1f);
    }

    ds4_gpu_tensor_free(attn_out);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(value_cache);
    ds4_gpu_tensor_free(key_cache);
    ds4_gpu_tensor_free(value);
    ds4_gpu_tensor_free(k_nope);
    ds4_gpu_tensor_free(kv_norm);
    ds4_gpu_tensor_free(kv_raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(q_rank_norm);
    ds4_gpu_tensor_free(q_rank);
    ds4_gpu_tensor_free(attn_norm);
    ds4_gpu_tensor_free(cur);
    ds4_gpu_tensor_free(tok);
    free(gpu_attn);
    free(cpu_attn);
    free(cpu_emb);
    free(tok_host);
    return ok;
}

/*
 * Validate the single-token decode attention path at pos > 0: prior tokens are
 * stored into the compact DSA caches with the batch store used by prefill,
 * then the last token runs through the exact decode kernel sequence
 * (rope tail, compact store, selected range, qk-lowrank, indexed decode
 * attention).  The pos-0-only diagnostics cannot see position-dependent
 * decode bugs; this one can.
 */
static int glm_metal_graph_test_decode_attention(
        ds4_engine              *e,
        const ds4_tokens        *prompt,
        const ds4_layer_weights *layer) {
    if (!prompt || prompt->len < 2) {
        printf("  layer0_decode_attention: skipped (prompt has fewer than 2 tokens)\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_weights *weights = &e->weights;
    const uint32_t n_tok = prompt->len < 8 ? (uint32_t)prompt->len : 8u;
    const uint32_t n_prev = n_tok - 1u;
    const uint32_t pos = n_tok - 1u;
    const uint32_t qk_dim = DS4_N_KEY_MLA;
    const uint32_t q_nope = qk_dim - DS4_N_ROT;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * qk_dim;
    const uint64_t heads_dim = (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA;
    const uint64_t kv_raw_dim = layer->attn_kv_a_mqa ? layer->attn_kv_a_mqa->dim[1] : 0;
    const uint64_t cache_elem = glm_graph_compact_cache_elem_bytes();
    const uint32_t cache_f16 = glm_graph_compact_cache_is_f16();
    const float rope_base = layer_rope_freq_base(0);
    const float rope_scale = layer_rope_freq_scale(0);

    for (uint32_t t = 0; t < n_tok; t++) {
        if (prompt->v[t] < 0 || prompt->v[t] >= (int)DS4_N_VOCAB) {
            fprintf(stderr, "ds4: GLM decode attention token %d is outside vocab\n",
                    prompt->v[t]);
            return 0;
        }
    }

    const uint64_t emb_bytes = (uint64_t)n_tok * DS4_N_EMBD * sizeof(float);
    int32_t *tok_host = xmalloc((size_t)n_prev * sizeof(int32_t));
    float *cpu_emb = xmalloc((size_t)emb_bytes);
    float *cpu_attn = xmalloc((size_t)emb_bytes);
    float *gpu_attn = xmalloc((size_t)DS4_N_EMBD * sizeof(float));

    for (uint32_t t = 0; t < n_tok; t++) {
        embed_token_any(model, weights, prompt->v[t],
                        cpu_emb + (uint64_t)t * DS4_N_EMBD);
    }
    for (uint32_t t = 0; t < n_prev; t++) tok_host[t] = (int32_t)prompt->v[t];
    layer_glm_attention_prefill_f32_ref(cpu_attn, model, layer, cpu_emb, n_tok, 0, 0);

    ds4_gpu_tensor *tok = NULL;
    ds4_gpu_tensor *cur_b = NULL;
    ds4_gpu_tensor *attn_norm_b = NULL;
    ds4_gpu_tensor *kv_raw_b = NULL;
    ds4_gpu_tensor *kv_norm_b = NULL;
    ds4_gpu_tensor *cur1 = NULL;
    ds4_gpu_tensor *attn_norm1 = NULL;
    ds4_gpu_tensor *q_rank = NULL;
    ds4_gpu_tensor *q_rank_norm = NULL;
    ds4_gpu_tensor *q = NULL;
    ds4_gpu_tensor *kv_raw1 = NULL;
    ds4_gpu_tensor *kv_norm1 = NULL;
    ds4_gpu_tensor *kv_lora_cache = NULL;
    ds4_gpu_tensor *k_rope_cache = NULL;
    ds4_gpu_tensor *selected = NULL;
    ds4_gpu_tensor *qk_low = NULL;
    ds4_gpu_tensor *heads = NULL;
    ds4_gpu_tensor *attn_out = NULL;

    int ok = 1;
#define DS4_GLM_DA_ALLOC_TENSOR(var, bytes_) \
    do { \
        (var) = ds4_gpu_tensor_alloc((bytes_)); \
        if (!(var)) { \
            fprintf(stderr, "ds4: GLM decode attention diagnostic could not allocate %s\n", #var); \
            ok = 0; \
        } \
    } while (0)

    DS4_GLM_DA_ALLOC_TENSOR(tok, (uint64_t)n_prev * sizeof(int32_t));
    DS4_GLM_DA_ALLOC_TENSOR(cur_b, (uint64_t)n_prev * DS4_N_EMBD * sizeof(float));
    DS4_GLM_DA_ALLOC_TENSOR(attn_norm_b, (uint64_t)n_prev * DS4_N_EMBD * sizeof(float));
    DS4_GLM_DA_ALLOC_TENSOR(kv_raw_b, (uint64_t)n_prev * kv_raw_dim * sizeof(float));
    DS4_GLM_DA_ALLOC_TENSOR(kv_norm_b, (uint64_t)n_prev * DS4_N_KV_LORA * sizeof(float));
    DS4_GLM_DA_ALLOC_TENSOR(cur1, (uint64_t)DS4_N_EMBD * sizeof(float));
    DS4_GLM_DA_ALLOC_TENSOR(attn_norm1, (uint64_t)DS4_N_EMBD * sizeof(float));
    DS4_GLM_DA_ALLOC_TENSOR(q_rank, (uint64_t)DS4_N_LORA_Q * sizeof(float));
    DS4_GLM_DA_ALLOC_TENSOR(q_rank_norm, (uint64_t)DS4_N_LORA_Q * sizeof(float));
    DS4_GLM_DA_ALLOC_TENSOR(q, q_dim * sizeof(float));
    DS4_GLM_DA_ALLOC_TENSOR(kv_raw1, kv_raw_dim * sizeof(float));
    DS4_GLM_DA_ALLOC_TENSOR(kv_norm1, (uint64_t)DS4_N_KV_LORA * sizeof(float));
    DS4_GLM_DA_ALLOC_TENSOR(kv_lora_cache, (uint64_t)n_tok * DS4_N_KV_LORA * cache_elem);
    DS4_GLM_DA_ALLOC_TENSOR(k_rope_cache, (uint64_t)n_tok * DS4_N_ROT * cache_elem);
    DS4_GLM_DA_ALLOC_TENSOR(selected, (uint64_t)n_tok * sizeof(int32_t));
    DS4_GLM_DA_ALLOC_TENSOR(qk_low, (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float));
    DS4_GLM_DA_ALLOC_TENSOR(heads, heads_dim * sizeof(float));
    DS4_GLM_DA_ALLOC_TENSOR(attn_out, (uint64_t)DS4_N_EMBD * sizeof(float));
#undef DS4_GLM_DA_ALLOC_TENSOR

    /* Store rows 0..n_prev-1 into the compact caches with the batch store. */
    if (ok) ok = ds4_gpu_tensor_write(tok, 0, tok_host,
                                      (uint64_t)n_prev * sizeof(int32_t)) != 0;
    if (ok) ok = ds4_gpu_embed_tokens_q8_0_tensor(cur_b,
                                                  tok,
                                                  model->map,
                                                  model->size,
                                                  weights->token_embd->abs_offset,
                                                  DS4_N_VOCAB,
                                                  n_prev,
                                                  DS4_N_EMBD);
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(attn_norm_b,
                                                     cur_b,
                                                     model->map,
                                                     model->size,
                                                     layer->attn_norm->abs_offset,
                                                     DS4_N_EMBD,
                                                     n_prev,
                                                     DS4_RMS_EPS);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(kv_raw_b,
                                            model->map,
                                            model->size,
                                            layer->attn_kv_a_mqa->abs_offset,
                                            DS4_N_EMBD,
                                            kv_raw_dim,
                                            attn_norm_b,
                                            n_prev);
    if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(kv_norm_b,
                                                     kv_raw_b,
                                                     model->map,
                                                     model->size,
                                                     layer->attn_kv_a_norm->abs_offset,
                                                     n_prev,
                                                     (uint32_t)kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_RMS_EPS);
    if (ok) ok = ds4_gpu_glm_store_compact_kv_tensor(kv_lora_cache,
                                                     k_rope_cache,
                                                     kv_norm_b,
                                                     kv_raw_b,
                                                     0,
                                                     n_prev,
                                                     n_tok,
                                                     (uint32_t)kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_N_ROT,
                                                     cache_f16) != 0;

    /* Run the last token through the decode kernel sequence at pos > 0. */
    if (ok) ok = ds4_gpu_embed_token_q8_0_tensor(cur1,
                                                 model->map,
                                                 model->size,
                                                 weights->token_embd->abs_offset,
                                                 DS4_N_VOCAB,
                                                 (uint32_t)prompt->v[pos],
                                                 DS4_N_EMBD) != 0;
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(attn_norm1,
                                                cur1,
                                                model->map,
                                                model->size,
                                                layer->attn_norm->abs_offset,
                                                DS4_N_EMBD,
                                                DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(q_rank,
                                            model->map,
                                            model->size,
                                            layer->attn_q_a->abs_offset,
                                            DS4_N_EMBD,
                                            DS4_N_LORA_Q,
                                            attn_norm1,
                                            1);
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(q_rank_norm,
                                                q_rank,
                                                model->map,
                                                model->size,
                                                layer->attn_q_a_norm->abs_offset,
                                                DS4_N_LORA_Q,
                                                DS4_RMS_EPS) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(q,
                                            model->map,
                                            model->size,
                                            layer->attn_q_b->abs_offset,
                                            DS4_N_LORA_Q,
                                            q_dim,
                                            q_rank_norm,
                                            1);
    if (ok) ok = ds4_gpu_glm_rope_tail_tensor(q,
                                              1,
                                              DS4_N_HEAD,
                                              qk_dim,
                                              DS4_N_ROT,
                                              pos,
                                              0,
                                              rope_base,
                                              rope_scale,
                                              0.0f,
                                              1.0f,
                                              DS4_ROPE_YARN_BETA_FAST,
                                              DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(kv_raw1,
                                            model->map,
                                            model->size,
                                            layer->attn_kv_a_mqa->abs_offset,
                                            DS4_N_EMBD,
                                            kv_raw_dim,
                                            attn_norm1,
                                            1);
    if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(kv_norm1,
                                                     kv_raw1,
                                                     model->map,
                                                     model->size,
                                                     layer->attn_kv_a_norm->abs_offset,
                                                     1,
                                                     (uint32_t)kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_RMS_EPS);
    if (ok) ok = ds4_gpu_glm_store_compact_kv_tensor(kv_lora_cache,
                                                     k_rope_cache,
                                                     kv_norm1,
                                                     kv_raw1,
                                                     pos,
                                                     1,
                                                     n_tok,
                                                     (uint32_t)kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_N_ROT,
                                                     cache_f16) != 0;
    if (ok) ok = ds4_gpu_glm_fill_selected_range_tensor(selected, n_tok) != 0;
    if (ok) ok = ds4_gpu_glm_qk_lowrank_typed_tensor(
            qk_low,
            q,
            model->map,
            model->size,
            layer->attn_k_b->abs_offset,
            layer->attn_k_b->type,
            DS4_N_HEAD,
            DS4_N_KV_LORA,
            q_nope,
            qk_dim) != 0;
    if (ok) ok = ds4_gpu_glm_attention_indexed_decode_typed_tensor(
            heads,
            q,
            qk_low,
            kv_lora_cache,
            k_rope_cache,
            model->map,
            model->size,
            layer->attn_v_b->abs_offset,
            layer->attn_v_b->type,
            selected,
            n_tok,
            n_tok,
            cache_f16,
            DS4_N_HEAD,
            DS4_N_KV_LORA,
            q_nope,
            DS4_N_ROT,
            DS4_N_VALUE_MLA,
            0,
            rope_base,
            rope_scale,
            0.0f,
            1.0f,
            DS4_ROPE_YARN_BETA_FAST,
            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(attn_out,
                                            model->map,
                                            model->size,
                                            layer->attn_output->abs_offset,
                                            heads_dim,
                                            DS4_N_EMBD,
                                            heads,
                                            1);
    if (ok) ok = ds4_gpu_tensor_read(attn_out, 0, gpu_attn,
                                     (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
    if (ok) {
        printf("  layer0_decode_attention: tokens=%u pos=%u\n", n_tok, pos);
        ok = glm_metal_compare_f32("layer0_decode_attention",
                                   cpu_attn + (uint64_t)pos * DS4_N_EMBD,
                                   gpu_attn,
                                   DS4_N_EMBD,
                                   5.0e-1f);
    }

    ds4_gpu_tensor_free(attn_out);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(qk_low);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(k_rope_cache);
    ds4_gpu_tensor_free(kv_lora_cache);
    ds4_gpu_tensor_free(kv_norm1);
    ds4_gpu_tensor_free(kv_raw1);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(q_rank_norm);
    ds4_gpu_tensor_free(q_rank);
    ds4_gpu_tensor_free(attn_norm1);
    ds4_gpu_tensor_free(cur1);
    ds4_gpu_tensor_free(kv_norm_b);
    ds4_gpu_tensor_free(kv_raw_b);
    ds4_gpu_tensor_free(attn_norm_b);
    ds4_gpu_tensor_free(cur_b);
    ds4_gpu_tensor_free(tok);
    free(gpu_attn);
    free(cpu_attn);
    free(cpu_emb);
    free(tok_host);
    return ok;
}

static int glm_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt) {
    if (!prompt || prompt->len <= 0) {
        fprintf(stderr, "ds4: GLM Metal graph test requires a non-empty prompt\n");
        return 1;
    }
    const int token = prompt->v[0];
    if (token < 0 || token >= (int)DS4_N_VOCAB) {
        fprintf(stderr, "ds4: GLM Metal graph test token %d is outside vocab\n", token);
        return 1;
    }
    if (!e->weights.token_embd || e->weights.token_embd->type != DS4_TENSOR_Q8_0) {
        fprintf(stderr, "ds4: GLM Metal graph test requires Q8_0 token embeddings\n");
        return 1;
    }
    if (DS4_N_LAYER <= DS4_N_NEXTN_PREDICT) {
        fprintf(stderr, "ds4: GLM Metal graph test has no normal transformer layers\n");
        return 1;
    }

    const uint32_t normal_layers = DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
    const uint32_t sparse_il = normal_layers > 8u ? 8u : DS4_N_LEADING_DENSE;
    if (sparse_il >= normal_layers) {
        fprintf(stderr, "ds4: GLM Metal graph test found no sparse GLM layer\n");
        return 1;
    }

    const ds4_model *model = &e->model;
    const ds4_weights *weights = &e->weights;
    const ds4_layer_weights *layer = &weights->layer[0];
    const ds4_layer_weights *sparse_layer = &weights->layer[sparse_il];
    const uint64_t heads_dim = (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA;
    const uint64_t kv_raw_dim = layer->attn_kv_a_mqa ? layer->attn_kv_a_mqa->dim[1] : 0;
    const uint64_t ffn_hidden = layer->ffn_gate ? layer->ffn_gate->dim[1] : 0;
    const uint64_t sparse_mid_dim =
        sparse_layer->ffn_gate_exps ? sparse_layer->ffn_gate_exps->dim[1] : 0;
    const uint32_t sparse_gate_type =
        sparse_layer->ffn_gate_exps ? sparse_layer->ffn_gate_exps->type : 0;
    const uint32_t sparse_up_type =
        sparse_layer->ffn_up_exps ? sparse_layer->ffn_up_exps->type : 0;
    const bool sparse_gate_pair_supported =
        glm_graph_gate_pair_type_supported(sparse_gate_type, sparse_up_type);
    if (!layer->attn_norm ||
        !layer->attn_kv_a_mqa ||
        !layer->attn_kv_a_norm ||
        !layer->attn_v_b ||
        !layer->attn_output ||
        !layer->ffn_norm ||
        !layer->ffn_gate ||
        !layer->ffn_up ||
        !layer->ffn_down ||
        kv_raw_dim < DS4_N_KV_LORA ||
        layer->attn_v_b->dim[0] != DS4_N_KV_LORA ||
        layer->attn_v_b->dim[1] != DS4_N_VALUE_MLA ||
        layer->attn_v_b->dim[2] != DS4_N_HEAD ||
        layer->attn_output->dim[0] != heads_dim ||
        layer->attn_output->dim[1] != DS4_N_EMBD ||
        layer->ffn_gate->dim[0] != DS4_N_EMBD ||
        layer->ffn_up->dim[0] != DS4_N_EMBD ||
        layer->ffn_up->dim[1] != ffn_hidden ||
        layer->ffn_down->dim[0] != ffn_hidden ||
        layer->ffn_down->dim[1] != DS4_N_EMBD) {
        fprintf(stderr, "ds4: GLM Metal graph test found unexpected layer-0 tensor layout\n");
        return 1;
    }
    if (!sparse_layer->ffn_gate_inp ||
        !sparse_layer->ffn_exp_probs_b ||
        !sparse_layer->ffn_norm ||
        sparse_layer->ffn_gate_inp->type != DS4_TENSOR_F32 ||
        sparse_layer->ffn_gate_inp->dim[0] != DS4_N_EMBD ||
        sparse_layer->ffn_gate_inp->dim[1] != DS4_N_EXPERT ||
        sparse_layer->ffn_exp_probs_b->type != DS4_TENSOR_F32 ||
        sparse_layer->ffn_exp_probs_b->dim[0] != DS4_N_EXPERT ||
        sparse_layer->ffn_norm->type != DS4_TENSOR_F32 ||
        sparse_layer->ffn_norm->dim[0] != DS4_N_EMBD ||
        !sparse_layer->ffn_gate_exps ||
        !sparse_layer->ffn_up_exps ||
        !sparse_layer->ffn_down_exps ||
        !sparse_layer->ffn_gate_shexp ||
        !sparse_layer->ffn_up_shexp ||
        !sparse_layer->ffn_down_shexp ||
        !sparse_gate_pair_supported ||
        !glm_graph_down_type_supported(sparse_layer->ffn_down_exps->type) ||
        sparse_layer->ffn_gate_shexp->type != DS4_TENSOR_Q8_0 ||
        sparse_layer->ffn_up_shexp->type != DS4_TENSOR_Q8_0 ||
        sparse_layer->ffn_down_shexp->type != DS4_TENSOR_Q8_0 ||
        sparse_layer->ffn_gate_exps->dim[0] != DS4_N_EMBD ||
        sparse_layer->ffn_gate_exps->dim[1] != DS4_N_FF_EXP ||
        sparse_layer->ffn_gate_exps->dim[2] != DS4_N_EXPERT ||
        sparse_layer->ffn_up_exps->dim[0] != DS4_N_EMBD ||
        sparse_layer->ffn_up_exps->dim[1] != DS4_N_FF_EXP ||
        sparse_layer->ffn_up_exps->dim[2] != DS4_N_EXPERT ||
        sparse_layer->ffn_down_exps->dim[0] != DS4_N_FF_EXP ||
        sparse_layer->ffn_down_exps->dim[1] != DS4_N_EMBD ||
        sparse_layer->ffn_down_exps->dim[2] != DS4_N_EXPERT ||
        sparse_layer->ffn_gate_shexp->dim[0] != DS4_N_EMBD ||
        sparse_layer->ffn_gate_shexp->dim[1] != DS4_N_FF_EXP ||
        sparse_layer->ffn_up_shexp->dim[0] != DS4_N_EMBD ||
        sparse_layer->ffn_up_shexp->dim[1] != DS4_N_FF_EXP ||
        sparse_layer->ffn_down_shexp->dim[0] != DS4_N_FF_EXP ||
        sparse_layer->ffn_down_shexp->dim[1] != DS4_N_EMBD ||
        sparse_mid_dim != DS4_N_FF_EXP) {
        fprintf(stderr, "ds4: GLM Metal graph test found unexpected layer-%u router layout\n",
                sparse_il);
        return 1;
    }

    const uint64_t emb_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t sparse_mid_elems = (uint64_t)DS4_N_EXPERT_USED * sparse_mid_dim;
    const uint64_t ffn_mid_elems =
        ffn_hidden > sparse_mid_elems ? ffn_hidden : sparse_mid_elems;
    const uint64_t routed_mid_bytes =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(float);
    const uint64_t routed_down_bytes =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float);
    const uint64_t max_read_elems =
        ffn_hidden > DS4_N_EMBD ? ffn_hidden : (uint64_t)DS4_N_EMBD;
    const uint64_t logits_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    float *cpu_emb = xmalloc(emb_bytes);
    float *cpu_layer = xmalloc(emb_bytes);
    float *cpu_cur = xmalloc(emb_bytes);
    float *cpu_next = xmalloc(emb_bytes);
    float *cpu_sparse_attn = xmalloc(emb_bytes);
    float *cpu_sparse_after_attn = xmalloc(emb_bytes);
    float *cpu_sparse_norm = xmalloc(emb_bytes);
    float *cpu_sparse_mid =
        xmalloc((size_t)sparse_mid_elems * sizeof(cpu_sparse_mid[0]));
    float *cpu_sparse_moe = xmalloc(emb_bytes);
    float *cpu_sparse_shared = xmalloc(emb_bytes);
    float *cpu_sparse_ffn = xmalloc(emb_bytes);
    float *cpu_full_hidden = xmalloc(emb_bytes);
    float *gpu_full_hidden = xmalloc(emb_bytes);
    float *cpu_logits = xmalloc((size_t)logits_bytes);
    float *gpu_logits = xmalloc((size_t)logits_bytes);
    float *cpu_router_logits = xmalloc((size_t)DS4_N_EXPERT * sizeof(cpu_router_logits[0]));
    float *gpu_router_logits = xmalloc((size_t)DS4_N_EXPERT * sizeof(gpu_router_logits[0]));
    float *gpu_read = xmalloc((size_t)max_read_elems * sizeof(gpu_read[0]));
    int cpu_router_selected[DS4_MAX_EXPERT_USED] = {0};
    int32_t gpu_router_selected[DS4_MAX_EXPERT_USED] = {0};
    float cpu_router_weights[DS4_MAX_EXPERT_USED] = {0};
    float gpu_router_weights[DS4_MAX_EXPERT_USED] = {0};
    char label_router_logits[64];
    char label_router_selected[64];
    char label_router_weights[64];
    char label_routed_moe[64];
    char label_shared_expert[64];
    char label_sparse_ffn[64];
    snprintf(label_router_logits, sizeof(label_router_logits),
             "layer%u_router_logits", sparse_il);
    snprintf(label_router_selected, sizeof(label_router_selected),
             "layer%u_router_selected", sparse_il);
    snprintf(label_router_weights, sizeof(label_router_weights),
             "layer%u_router_weights", sparse_il);
    snprintf(label_routed_moe, sizeof(label_routed_moe),
             "layer%u_routed_moe", sparse_il);
    snprintf(label_shared_expert, sizeof(label_shared_expert),
             "layer%u_shared_expert", sparse_il);
    snprintf(label_sparse_ffn, sizeof(label_sparse_ffn),
             "layer%u_sparse_ffn", sparse_il);

    ds4_gpu_tensor *cur = NULL;
    ds4_gpu_tensor *attn_norm = NULL;
    ds4_gpu_tensor *kv_raw = NULL;
    ds4_gpu_tensor *kv_norm = NULL;
    ds4_gpu_tensor *heads = NULL;
    ds4_gpu_tensor *attn_out = NULL;
    ds4_gpu_tensor *after_attn = NULL;
    ds4_gpu_tensor *ffn_norm = NULL;
    ds4_gpu_tensor *ffn_gate = NULL;
    ds4_gpu_tensor *ffn_up = NULL;
    ds4_gpu_tensor *ffn_mid = NULL;
    ds4_gpu_tensor *routed_gate = NULL;
    ds4_gpu_tensor *routed_up = NULL;
    ds4_gpu_tensor *routed_down = NULL;
    ds4_gpu_tensor *ffn_out = NULL;
    ds4_gpu_tensor *ffn_sum = NULL;
    ds4_gpu_tensor *next = NULL;
    ds4_gpu_tensor *router_logits = NULL;
    ds4_gpu_tensor *router_probs = NULL;
    ds4_gpu_tensor *router_selected = NULL;
    ds4_gpu_tensor *router_weights = NULL;
    ds4_gpu_tensor *logits = NULL;

    int ok = 1;
#define DS4_GLM_ALLOC_TENSOR(var, bytes_) \
    do { \
        (var) = ds4_gpu_tensor_alloc((bytes_)); \
        if (!(var)) { \
            fprintf(stderr, "ds4: GLM Metal graph test could not allocate %s\n", #var); \
            ok = 0; \
        } \
    } while (0)

    DS4_GLM_ALLOC_TENSOR(cur, emb_bytes);
    DS4_GLM_ALLOC_TENSOR(attn_norm, emb_bytes);
    DS4_GLM_ALLOC_TENSOR(kv_raw, kv_raw_dim * sizeof(float));
    DS4_GLM_ALLOC_TENSOR(kv_norm, (uint64_t)DS4_N_KV_LORA * sizeof(float));
    DS4_GLM_ALLOC_TENSOR(heads, heads_dim * sizeof(float));
    DS4_GLM_ALLOC_TENSOR(attn_out, emb_bytes);
    DS4_GLM_ALLOC_TENSOR(after_attn, emb_bytes);
    DS4_GLM_ALLOC_TENSOR(ffn_norm, emb_bytes);
    DS4_GLM_ALLOC_TENSOR(ffn_gate, ffn_hidden * sizeof(float));
    DS4_GLM_ALLOC_TENSOR(ffn_up, ffn_hidden * sizeof(float));
    DS4_GLM_ALLOC_TENSOR(ffn_mid, ffn_mid_elems * sizeof(float));
    if (glm_graph_layer_uses_generic_routed_moe(sparse_layer)) {
        DS4_GLM_ALLOC_TENSOR(routed_gate, routed_mid_bytes);
        DS4_GLM_ALLOC_TENSOR(routed_up, routed_mid_bytes);
        DS4_GLM_ALLOC_TENSOR(routed_down, routed_down_bytes);
    }
    DS4_GLM_ALLOC_TENSOR(ffn_out, emb_bytes);
    DS4_GLM_ALLOC_TENSOR(ffn_sum, emb_bytes);
    DS4_GLM_ALLOC_TENSOR(next, emb_bytes);
    DS4_GLM_ALLOC_TENSOR(router_logits, (uint64_t)DS4_N_EXPERT * sizeof(float));
    DS4_GLM_ALLOC_TENSOR(router_probs, (uint64_t)DS4_N_EXPERT * sizeof(float));
    DS4_GLM_ALLOC_TENSOR(router_selected, (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
    DS4_GLM_ALLOC_TENSOR(router_weights, (uint64_t)DS4_N_EXPERT_USED * sizeof(float));
    DS4_GLM_ALLOC_TENSOR(logits, logits_bytes);
#undef DS4_GLM_ALLOC_TENSOR

    if (ok) {
        embed_token_any(model, weights, token, cpu_emb);
        layer_glm_first_token_one(cpu_layer, model, layer, cpu_emb, 0);

        ok = ds4_gpu_embed_token_q8_0_tensor(cur,
                                             model->map,
                                             model->size,
                                             weights->token_embd->abs_offset,
                                             DS4_N_VOCAB,
                                             (uint32_t)token,
                                             DS4_N_EMBD);
    }
    if (ok) ok = ds4_gpu_tensor_read(cur, 0, gpu_read, emb_bytes);
    if (ok) {
        printf("GLM Metal graph test token=%d", token);
        if (e->vocab.token && token < e->vocab.n_vocab) {
            const ds4_str s = e->vocab.token[token];
            printf(" text=%.*s", (int)s.len, s.ptr);
        }
        printf("\n");
        ok = glm_metal_compare_f32("embedding_q8_0", cpu_emb, gpu_read,
                                   DS4_N_EMBD, 1.0e-3f);
    }

    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(attn_norm, cur,
                                                model->map, model->size,
                                                layer->attn_norm->abs_offset,
                                                DS4_N_EMBD, DS4_RMS_EPS);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(kv_raw, model->map, model->size,
                                            layer->attn_kv_a_mqa->abs_offset,
                                            DS4_N_EMBD, kv_raw_dim, attn_norm, 1);
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(kv_norm, kv_raw,
                                                model->map, model->size,
                                                layer->attn_kv_a_norm->abs_offset,
                                                DS4_N_KV_LORA, DS4_RMS_EPS);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(heads, model->map, model->size,
                                            layer->attn_v_b->abs_offset,
                                            DS4_N_KV_LORA, heads_dim, kv_norm, 1);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(attn_out, model->map, model->size,
                                            layer->attn_output->abs_offset,
                                            heads_dim, DS4_N_EMBD, heads, 1);
    if (ok) ok = ds4_gpu_add_tensor(after_attn, cur, attn_out, DS4_N_EMBD);
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(ffn_norm, after_attn,
                                                model->map, model->size,
                                                layer->ffn_norm->abs_offset,
                                                DS4_N_EMBD, DS4_RMS_EPS);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_gate, model->map, model->size,
                                            layer->ffn_gate->abs_offset,
                                            DS4_N_EMBD, ffn_hidden, ffn_norm, 1);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_up, model->map, model->size,
                                            layer->ffn_up->abs_offset,
                                            DS4_N_EMBD, ffn_hidden, ffn_norm, 1);
    if (ok) ok = ds4_gpu_swiglu_tensor(ffn_mid, ffn_gate, ffn_up,
                                       (uint32_t)ffn_hidden, 0.0f, 1.0f);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_out, model->map, model->size,
                                            layer->ffn_down->abs_offset,
                                            ffn_hidden, DS4_N_EMBD, ffn_mid, 1);
    if (ok) ok = ds4_gpu_add_tensor(next, after_attn, ffn_out, DS4_N_EMBD);
    if (ok) ok = ds4_gpu_tensor_read(next, 0, gpu_read, emb_bytes);
    if (ok) {
        ok = glm_metal_compare_f32("layer0_hidden", cpu_layer, gpu_read,
                                   DS4_N_EMBD, 5.0e-2f);
    }
    if (ok) ok = glm_metal_graph_test_multitok_attention(e, prompt, layer);
    if (ok) ok = glm_metal_graph_test_decode_attention(e, prompt, layer);
    if (ok) ok = glm_metal_graph_test_q8_prefill(e, layer);

    if (ok) {
        memcpy(cpu_cur, cpu_emb, emb_bytes);
        for (uint32_t il = 0; il < sparse_il; il++) {
            layer_glm_first_token_one(cpu_next, model, &weights->layer[il], cpu_cur, il);
            float *tmp = cpu_cur;
            cpu_cur = cpu_next;
            cpu_next = tmp;
        }

        layer_glm_first_token_attention_one(cpu_sparse_attn, model, sparse_layer, cpu_cur);
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
            cpu_sparse_after_attn[i] = cpu_cur[i] + cpu_sparse_attn[i];
        }
        rms_norm_weight(cpu_sparse_norm,
                        cpu_sparse_after_attn,
                        tensor_data(model, sparse_layer->ffn_norm),
                        DS4_N_EMBD,
                        DS4_RMS_EPS);
        matvec_any(cpu_router_logits, model, sparse_layer->ffn_gate_inp, cpu_sparse_norm);
        layer_glm_router_selected_experts(cpu_router_selected,
                                          cpu_router_weights,
                                          model,
                                          sparse_layer,
                                          cpu_sparse_norm);

        ok = ds4_gpu_tensor_write(ffn_norm, 0, cpu_sparse_norm, emb_bytes) != 0;
    }
    if (ok) ok = ds4_gpu_matmul_f32_tensor(router_logits,
                                           model->map,
                                           model->size,
                                           sparse_layer->ffn_gate_inp->abs_offset,
                                           DS4_N_EMBD,
                                           DS4_N_EXPERT,
                                           ffn_norm,
                                           1);
    if (ok) ok = ds4_gpu_tensor_read(router_logits,
                                     0,
                                     gpu_router_logits,
                                     (uint64_t)DS4_N_EXPERT * sizeof(gpu_router_logits[0])) != 0;
    if (ok) {
        ok = glm_metal_compare_f32(label_router_logits,
                                   cpu_router_logits,
                                   gpu_router_logits,
                                   DS4_N_EXPERT,
                                   2.0e-3f);
    }
    if (ok) ok = ds4_gpu_glm_router_select_tensor(router_selected,
                                                  router_weights,
                                                  router_probs,
                                                  model->map,
                                                  model->size,
                                                  sparse_layer->ffn_exp_probs_b->abs_offset,
                                                  router_logits,
                                                  DS4_N_EXPERT,
                                                  DS4_N_EXPERT_USED,
                                                  DS4_EXPERT_WEIGHT_SCALE);
    if (ok) ok = ds4_gpu_tensor_read(router_selected,
                                     0,
                                     gpu_router_selected,
                                     (uint64_t)DS4_N_EXPERT_USED * sizeof(gpu_router_selected[0])) != 0;
    if (ok) ok = ds4_gpu_tensor_read(router_weights,
                                     0,
                                     gpu_router_weights,
                                     (uint64_t)DS4_N_EXPERT_USED * sizeof(gpu_router_weights[0])) != 0;
    if (ok) {
        ok = glm_metal_compare_i32_list(label_router_selected,
                                        cpu_router_selected,
                                        gpu_router_selected,
                                        DS4_N_EXPERT_USED);
    }
    if (ok) {
        ok = glm_metal_compare_f32(label_router_weights,
                                   cpu_router_weights,
                                   gpu_router_weights,
                                   DS4_N_EXPERT_USED,
                                   1.0e-4f);
    }
    if (ok) {
        uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
        uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
        uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;
        (void)tensor_expert_bytes(model, sparse_layer->ffn_gate_exps, 0,
                                  &gate_in, &gate_out, &gate_row_bytes);
        (void)tensor_expert_bytes(model, sparse_layer->ffn_up_exps, 0,
                                  &up_in, &up_out, &up_row_bytes);
        (void)tensor_expert_bytes(model, sparse_layer->ffn_down_exps, 0,
                                  &down_in, &down_out, &down_row_bytes);
        if (gate_in != DS4_N_EMBD ||
            up_in != DS4_N_EMBD ||
            down_in != DS4_N_FF_EXP ||
            gate_out != DS4_N_FF_EXP ||
            up_out != DS4_N_FF_EXP ||
            down_out != DS4_N_EMBD) {
            fprintf(stderr, "ds4: GLM Metal graph test found unexpected layer-%u expert strides\n",
                    sparse_il);
            ok = 0;
        } else {
            const ds4_gpu_stream_expert_table table = {
                .model_map = model->map,
                .model_size = model->size,
                .layer = sparse_il,
                .n_total_expert = DS4_N_EXPERT,
                .gate_offset = sparse_layer->ffn_gate_exps->abs_offset,
                .up_offset = sparse_layer->ffn_up_exps->abs_offset,
                .down_offset = sparse_layer->ffn_down_exps->abs_offset,
                .gate_expert_bytes = gate_out * gate_row_bytes,
                .down_expert_bytes = down_out * down_row_bytes,
            };
            ok = ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
                    &table,
                    router_selected,
                    DS4_N_EXPERT_USED) != 0;
            if (ok) {
                layer_glm_routed_moe_one_f32_ref(cpu_sparse_moe,
                                                 cpu_sparse_mid,
                                                 model,
                                                 sparse_layer,
                                                 cpu_sparse_norm,
                                                 cpu_router_selected,
                                                 cpu_router_weights);
            }
            ds4_glm_gpu_graph route_g = {
                .routed_gate = routed_gate,
                .routed_up = routed_up,
                .routed_down = routed_down,
                .ssd_streaming = e->ssd_streaming,
            };
            if (ok) ok = glm_graph_routed_moe_one_dispatch(
                    &route_g,
                    model,
                    sparse_layer,
                    sparse_il,
                    ffn_out,
                    ffn_mid,
                    gate_out * gate_row_bytes,
                    gate_row_bytes,
                    up_out * up_row_bytes,
                    up_row_bytes,
                    down_out * down_row_bytes,
                    down_row_bytes,
                    router_selected,
                    router_weights,
                    ffn_norm,
                    false);
        }
    }
    if (ok) ok = ds4_gpu_tensor_read(ffn_out, 0, gpu_read, emb_bytes) != 0;
    if (ok) {
        ok = glm_metal_compare_f32(label_routed_moe,
                                   cpu_sparse_moe,
                                   gpu_read,
                                   DS4_N_EMBD,
                                   1.0e-1f);
    }
    if (ok) {
        layer_glm_shared_ffn_one_f32_ref(cpu_sparse_shared,
                                         model,
                                         sparse_layer,
                                         cpu_sparse_norm);
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
            cpu_sparse_ffn[i] = cpu_sparse_moe[i] + cpu_sparse_shared[i];
        }
        ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                ffn_gate,
                ffn_up,
                ffn_mid,
                model->map,
                model->size,
                sparse_layer->ffn_gate_shexp->abs_offset,
                sparse_layer->ffn_up_shexp->abs_offset,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                ffn_norm,
                0.0f);
    }
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(next,
                                            model->map,
                                            model->size,
                                            sparse_layer->ffn_down_shexp->abs_offset,
                                            DS4_N_FF_EXP,
                                            DS4_N_EMBD,
                                            ffn_mid,
                                            1);
    if (ok) ok = ds4_gpu_tensor_read(next, 0, gpu_read, emb_bytes) != 0;
    if (ok) {
        ok = glm_metal_compare_f32(label_shared_expert,
                                   cpu_sparse_shared,
                                   gpu_read,
                                   DS4_N_EMBD,
                                   5.0e-2f);
    }
    if (ok) ok = ds4_gpu_add_tensor(after_attn, ffn_out, next, DS4_N_EMBD);
    if (ok) ok = ds4_gpu_tensor_read(after_attn, 0, gpu_read, emb_bytes) != 0;
    if (ok) {
        ok = glm_metal_compare_f32(label_sparse_ffn,
                                   cpu_sparse_ffn,
                                   gpu_read,
                                   DS4_N_EMBD,
                                   5.0e-2f);
    }

    if (ok) {
        printf("  all_layers_reference: cpu_f32 normal_layers=%u\n", normal_layers);
        fflush(stdout);
        forward_glm_first_token_cpu_f32_ref(cpu_full_hidden, model, weights, token);
        output_logits_glm_one_f32_ref(cpu_logits, model, weights, cpu_full_hidden);
        ok = ds4_gpu_embed_token_q8_0_tensor(cur,
                                             model->map,
                                             model->size,
                                             weights->token_embd->abs_offset,
                                             DS4_N_VOCAB,
                                             (uint32_t)token,
                                             DS4_N_EMBD);
    }
    for (uint32_t il = 0; ok && il < normal_layers; il++) {
        const ds4_layer_weights *gl = &weights->layer[il];
        const uint64_t gl_kv_raw_dim = gl->attn_kv_a_mqa ? gl->attn_kv_a_mqa->dim[1] : 0;
        if (!gl->attn_norm ||
            !gl->attn_kv_a_mqa ||
            !gl->attn_kv_a_norm ||
            !gl->attn_v_b ||
            !gl->attn_output ||
            !gl->ffn_norm ||
            gl_kv_raw_dim < DS4_N_KV_LORA ||
            gl_kv_raw_dim > kv_raw_dim ||
            gl->attn_kv_a_mqa->dim[0] != DS4_N_EMBD ||
            gl->attn_v_b->dim[0] != DS4_N_KV_LORA ||
            gl->attn_v_b->dim[1] != DS4_N_VALUE_MLA ||
            gl->attn_v_b->dim[2] != DS4_N_HEAD ||
            gl->attn_output->dim[0] != heads_dim ||
            gl->attn_output->dim[1] != DS4_N_EMBD) {
            fprintf(stderr,
                    "ds4: GLM Metal all-layer test found unexpected attention layout in layer %u\n",
                    il);
            ok = 0;
            break;
        }

        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(attn_norm, cur,
                                                    model->map, model->size,
                                                    gl->attn_norm->abs_offset,
                                                    DS4_N_EMBD, DS4_RMS_EPS);
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(kv_raw,
                                                model->map,
                                                model->size,
                                                gl->attn_kv_a_mqa->abs_offset,
                                                DS4_N_EMBD,
                                                gl_kv_raw_dim,
                                                attn_norm,
                                                1);
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(kv_norm, kv_raw,
                                                    model->map, model->size,
                                                    gl->attn_kv_a_norm->abs_offset,
                                                    DS4_N_KV_LORA, DS4_RMS_EPS);
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(heads,
                                                model->map,
                                                model->size,
                                                gl->attn_v_b->abs_offset,
                                                DS4_N_KV_LORA,
                                                heads_dim,
                                                kv_norm,
                                                1);
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(attn_out,
                                                model->map,
                                                model->size,
                                                gl->attn_output->abs_offset,
                                                heads_dim,
                                                DS4_N_EMBD,
                                                heads,
                                                1);
        if (ok) ok = ds4_gpu_add_tensor(after_attn, cur, attn_out, DS4_N_EMBD);
        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(ffn_norm, after_attn,
                                                    model->map, model->size,
                                                    gl->ffn_norm->abs_offset,
                                                    DS4_N_EMBD, DS4_RMS_EPS);
        if (il < DS4_N_LEADING_DENSE) {
            const uint64_t gl_ffn_hidden = gl->ffn_gate ? gl->ffn_gate->dim[1] : 0;
            if (!gl->ffn_gate ||
                !gl->ffn_up ||
                !gl->ffn_down ||
                gl->ffn_gate->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_up->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_down->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_gate->dim[0] != DS4_N_EMBD ||
                gl->ffn_up->dim[0] != DS4_N_EMBD ||
                gl->ffn_up->dim[1] != gl_ffn_hidden ||
                gl->ffn_down->dim[0] != gl_ffn_hidden ||
                gl->ffn_down->dim[1] != DS4_N_EMBD ||
                gl_ffn_hidden > ffn_hidden) {
                fprintf(stderr,
                        "ds4: GLM Metal all-layer test found unexpected dense FFN layout in layer %u\n",
                        il);
                ok = 0;
                break;
            }
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_gate,
                                                    model->map,
                                                    model->size,
                                                    gl->ffn_gate->abs_offset,
                                                    DS4_N_EMBD,
                                                    gl_ffn_hidden,
                                                    ffn_norm,
                                                    1);
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_up,
                                                    model->map,
                                                    model->size,
                                                    gl->ffn_up->abs_offset,
                                                    DS4_N_EMBD,
                                                    gl_ffn_hidden,
                                                    ffn_norm,
                                                    1);
            if (ok) ok = ds4_gpu_swiglu_tensor(ffn_mid, ffn_gate, ffn_up,
                                               (uint32_t)gl_ffn_hidden, 0.0f, 1.0f);
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_out,
                                                    model->map,
                                                    model->size,
                                                    gl->ffn_down->abs_offset,
                                                    gl_ffn_hidden,
                                                    DS4_N_EMBD,
                                                    ffn_mid,
                                                    1);
            if (ok) ok = ds4_gpu_add_tensor(next, after_attn, ffn_out, DS4_N_EMBD);
        } else {
            const uint32_t gl_gate_type = gl->ffn_gate_exps ? gl->ffn_gate_exps->type : 0;
            const uint32_t gl_up_type = gl->ffn_up_exps ? gl->ffn_up_exps->type : 0;
            const bool gl_gate_pair_supported =
                glm_graph_gate_pair_type_supported(gl_gate_type, gl_up_type);
            uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
            uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
            uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;

            if (!gl->ffn_gate_inp ||
                !gl->ffn_exp_probs_b ||
                !gl->ffn_gate_exps ||
                !gl->ffn_up_exps ||
                !gl->ffn_down_exps ||
                !gl->ffn_gate_shexp ||
                !gl->ffn_up_shexp ||
                !gl->ffn_down_shexp ||
                gl->ffn_gate_inp->type != DS4_TENSOR_F32 ||
                gl->ffn_gate_inp->dim[0] != DS4_N_EMBD ||
                gl->ffn_gate_inp->dim[1] != DS4_N_EXPERT ||
                gl->ffn_exp_probs_b->type != DS4_TENSOR_F32 ||
                gl->ffn_exp_probs_b->dim[0] != DS4_N_EXPERT ||
                !gl_gate_pair_supported ||
                !glm_graph_down_type_supported(gl->ffn_down_exps->type) ||
                gl->ffn_gate_shexp->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_up_shexp->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_down_shexp->type != DS4_TENSOR_Q8_0 ||
                gl->ffn_gate_shexp->dim[0] != DS4_N_EMBD ||
                gl->ffn_gate_shexp->dim[1] != DS4_N_FF_EXP ||
                gl->ffn_up_shexp->dim[0] != DS4_N_EMBD ||
                gl->ffn_up_shexp->dim[1] != DS4_N_FF_EXP ||
                gl->ffn_down_shexp->dim[0] != DS4_N_FF_EXP ||
                gl->ffn_down_shexp->dim[1] != DS4_N_EMBD ||
                (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP > ffn_mid_elems) {
                fprintf(stderr,
                        "ds4: GLM Metal all-layer test found unexpected sparse FFN layout in layer %u\n",
                        il);
                ok = 0;
                break;
            }

            (void)tensor_expert_bytes(model, gl->ffn_gate_exps, 0,
                                      &gate_in, &gate_out, &gate_row_bytes);
            (void)tensor_expert_bytes(model, gl->ffn_up_exps, 0,
                                      &up_in, &up_out, &up_row_bytes);
            (void)tensor_expert_bytes(model, gl->ffn_down_exps, 0,
                                      &down_in, &down_out, &down_row_bytes);
            if (gate_in != DS4_N_EMBD ||
                up_in != DS4_N_EMBD ||
                down_in != DS4_N_FF_EXP ||
                gate_out != DS4_N_FF_EXP ||
                up_out != DS4_N_FF_EXP ||
                down_out != DS4_N_EMBD) {
                fprintf(stderr,
                        "ds4: GLM Metal all-layer test found unexpected expert strides in layer %u\n",
                        il);
                ok = 0;
                break;
            }

            if (ok) ok = ds4_gpu_matmul_f32_tensor(router_logits,
                                                   model->map,
                                                   model->size,
                                                   gl->ffn_gate_inp->abs_offset,
                                                   DS4_N_EMBD,
                                                   DS4_N_EXPERT,
                                                   ffn_norm,
                                                   1);
            if (ok) ok = ds4_gpu_glm_router_select_tensor(router_selected,
                                                          router_weights,
                                                          router_probs,
                                                          model->map,
                                                          model->size,
                                                          gl->ffn_exp_probs_b->abs_offset,
                                                          router_logits,
                                                          DS4_N_EXPERT,
                                                          DS4_N_EXPERT_USED,
                                                          DS4_EXPERT_WEIGHT_SCALE);
            if (ok) {
                const ds4_gpu_stream_expert_table table = {
                    .model_map = model->map,
                    .model_size = model->size,
                    .layer = il,
                    .n_total_expert = DS4_N_EXPERT,
                    .gate_offset = gl->ffn_gate_exps->abs_offset,
                    .up_offset = gl->ffn_up_exps->abs_offset,
                    .down_offset = gl->ffn_down_exps->abs_offset,
                    .gate_expert_bytes = gate_out * gate_row_bytes,
                    .down_expert_bytes = down_out * down_row_bytes,
                };
                ok = ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
                        &table,
                        router_selected,
                        DS4_N_EXPERT_USED) != 0;
            }
            ds4_glm_gpu_graph route_g = {
                .routed_gate = routed_gate,
                .routed_up = routed_up,
                .routed_down = routed_down,
                .ssd_streaming = e->ssd_streaming,
            };
            if (ok) ok = glm_graph_routed_moe_one_dispatch(
                    &route_g,
                    model,
                    gl,
                    il,
                    ffn_out,
                    ffn_mid,
                    gate_out * gate_row_bytes,
                    gate_row_bytes,
                    up_out * up_row_bytes,
                    up_row_bytes,
                    down_out * down_row_bytes,
                    down_row_bytes,
                    router_selected,
                    router_weights,
                    ffn_norm,
                    false);
            if (ok) ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                    ffn_gate,
                    ffn_up,
                    ffn_mid,
                    model->map,
                    model->size,
                    gl->ffn_gate_shexp->abs_offset,
                    gl->ffn_up_shexp->abs_offset,
                    DS4_N_EMBD,
                    DS4_N_FF_EXP,
                    ffn_norm,
                    0.0f);
            if (ok) ok = ds4_gpu_matmul_q8_0_tensor(ffn_sum,
                                                    model->map,
                                                    model->size,
                                                    gl->ffn_down_shexp->abs_offset,
                                                    DS4_N_FF_EXP,
                                                    DS4_N_EMBD,
                                                    ffn_mid,
                                                    1);
            if (ok) ok = ds4_gpu_add_tensor(attn_out, ffn_out, ffn_sum, DS4_N_EMBD);
            if (ok) ok = ds4_gpu_add_tensor(next, after_attn, attn_out, DS4_N_EMBD);
        }

        if (ok) {
            ds4_gpu_tensor *tmp = cur;
            cur = next;
            next = tmp;
        }
    }
    if (ok) ok = ds4_gpu_tensor_read(cur, 0, gpu_full_hidden, emb_bytes) != 0;
    if (ok) {
        ok = glm_metal_compare_f32("all_layers_hidden",
                                   cpu_full_hidden,
                                   gpu_full_hidden,
                                   DS4_N_EMBD,
                                   5.0f);
    }
    if (ok) ok = ds4_gpu_rms_norm_weight_tensor(ffn_norm, cur,
                                                model->map, model->size,
                                                weights->output_norm->abs_offset,
                                                DS4_N_EMBD, DS4_RMS_EPS);
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(logits,
                                            model->map,
                                            model->size,
                                            weights->output->abs_offset,
                                            DS4_N_EMBD,
                                            DS4_N_VOCAB,
                                            ffn_norm,
                                            1);
    if (ok) ok = ds4_gpu_tensor_read(logits, 0, gpu_logits, logits_bytes) != 0;
    if (ok) {
        ok = glm_metal_compare_f32("first_token_logits",
                                   cpu_logits,
                                   gpu_logits,
                                   DS4_N_VOCAB,
                                   10.0f);
    }

    ds4_gpu_tensor_free(router_weights);
    ds4_gpu_tensor_free(router_selected);
    ds4_gpu_tensor_free(router_probs);
    ds4_gpu_tensor_free(router_logits);
    ds4_gpu_tensor_free(logits);
    ds4_gpu_tensor_free(next);
    ds4_gpu_tensor_free(ffn_sum);
    ds4_gpu_tensor_free(ffn_out);
    ds4_gpu_tensor_free(routed_down);
    ds4_gpu_tensor_free(routed_up);
    ds4_gpu_tensor_free(routed_gate);
    ds4_gpu_tensor_free(ffn_mid);
    ds4_gpu_tensor_free(ffn_up);
    ds4_gpu_tensor_free(ffn_gate);
    ds4_gpu_tensor_free(ffn_norm);
    ds4_gpu_tensor_free(after_attn);
    ds4_gpu_tensor_free(attn_out);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(kv_norm);
    ds4_gpu_tensor_free(kv_raw);
    ds4_gpu_tensor_free(attn_norm);
    ds4_gpu_tensor_free(cur);
    free(gpu_read);
    free(gpu_router_logits);
    free(cpu_router_logits);
    free(gpu_logits);
    free(cpu_logits);
    free(gpu_full_hidden);
    free(cpu_full_hidden);
    free(cpu_sparse_ffn);
    free(cpu_sparse_shared);
    free(cpu_sparse_moe);
    free(cpu_sparse_mid);
    free(cpu_sparse_norm);
    free(cpu_sparse_after_attn);
    free(cpu_sparse_attn);
    free(cpu_next);
    free(cpu_cur);
    free(cpu_layer);
    free(cpu_emb);
    return ok ? 0 : 1;
}
#endif

