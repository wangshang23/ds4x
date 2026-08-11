#ifndef DS4_NO_GPU
static bool ds4_sessions_eval_batch_metal_supported(
        ds4_decode_item *items,
        int count,
        ds4_engine *e) {
    const char *tp_batch = getenv("DS4_METAL_TP_SESSION_BATCH");
    if (!items || count < 2 || !e || e->backend != DS4_BACKEND_METAL ||
        e->support_kind != DS4_SUPPORT_NONE ||
        (e->tp.active && tp_batch && strcmp(tp_batch, "0") == 0) ||
        getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != NULL ||
        getenv("DS4_METAL_DECODE_STAGE_PROFILE") != NULL) {
        return false;
    }
    for (int i = 0; i < count; i++) {
        ds4_session *s = items[i].session;
        if (!s || s->engine != e || s->distributed ||
            ds4_session_is_cpu(s) || !s->checkpoint_valid) {
            return false;
        }
        if (ds4_session_is_glm(s)) {
            if (!s->glm_graph_ready || s->glm_graph.ssd_streaming ||
                glm_debug_hidden_dump_layer() >= 0 ||
                e->glm_mtp ||
                getenv("DS4_GLM_MTP_PROBE") != NULL) {
                return false;
            }
        } else if (s->graph.ssd_streaming) {
            return false;
        }
    }
    return true;
}

static bool metal_graph_native_session_batch_shared_supported(
        ds4_decode_item *items,
        int count,
        const ds4_engine *e) {
    const char *enabled = getenv("DS4_METAL_SESSION_BATCH_SHARED");
    if ((enabled && enabled[0] && strcmp(enabled, "0") == 0) ||
        !items || count < 2 || !e || e->tp.active ||
        e->support_kind != DS4_SUPPORT_NONE ||
        metal_graph_use_reference_shared_down_hc() ||
        metal_graph_use_q4_selected_shared_overlap(NULL) ||
        metal_graph_use_pro_q4_cpu_router()) {
        return false;
    }

    ds4_gpu_graph *first = &items[0].session->graph;
    if (first->placement || first->ssd_streaming || first->quality ||
        !first->shared_gate_up_swiglu_fuse ||
        (uint32_t)count > first->prefill_cap ||
        !metal_graph_batch_ffn_norm(first) ||
        !metal_graph_batch_shared_gate(first) ||
        !metal_graph_batch_shared_up(first) ||
        !metal_graph_batch_shared_mid(first)) {
        return false;
    }

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &e->weights.layer[il];
        if (!layer->ffn_gate_shexp || !layer->ffn_up_shexp ||
            !layer->ffn_down_shexp ||
            layer->ffn_gate_shexp->type != DS4_TENSOR_Q8_0 ||
            layer->ffn_up_shexp->type != DS4_TENSOR_Q8_0 ||
            layer->ffn_down_shexp->type != DS4_TENSOR_Q8_0) {
            return false;
        }
    }
    for (int i = 0; i < count; i++) {
        ds4_session *s = items[i].session;
        if (!s || ds4_session_is_glm(s) || s->graph.placement ||
            s->graph.ssd_streaming || s->graph.quality ||
            s->graph.tp_world >= 2 ||
            !s->graph.shared_gate_up_swiglu_fuse ||
            s->graph.materialize_ffn_out ||
            metal_graph_directional_steering_attn_enabled(&s->graph) ||
            metal_graph_directional_steering_ffn_enabled(&s->graph)) {
            return false;
        }
    }
    return true;
}

static bool metal_graph_native_session_batch_qkv_supported(
        ds4_decode_item *items,
        int count,
        const ds4_engine *e) {
    const char *enabled = getenv("DS4_METAL_SESSION_BATCH_QKV");
    if ((enabled && enabled[0] && strcmp(enabled, "0") == 0) ||
        !items || count < 2 || !e || e->tp.active ||
        metal_graph_use_reference_qkv_norm()) {
        return false;
    }
    ds4_gpu_graph *first = &items[0].session->graph;
    if (!metal_graph_batch_attn_norm(first) ||
        !metal_graph_batch_qr(first) ||
        !metal_graph_batch_kv_raw(first)) {
        return false;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &e->weights.layer[il];
        if (!layer->attn_q_a || !layer->attn_kv ||
            layer->attn_q_a->type != DS4_TENSOR_Q8_0 ||
            layer->attn_kv->type != DS4_TENSOR_Q8_0 ||
            layer->attn_q_a->dim[0] != DS4_N_EMBD ||
            layer->attn_kv->dim[0] != DS4_N_EMBD ||
            layer->attn_kv->dim[1] != DS4_N_HEAD_DIM) {
            return false;
        }
    }
    return true;
}

static bool metal_graph_encode_native_session_batch_shared(
        ds4_decode_item *items,
        int count,
        const ds4_model *model,
        const ds4_weights *weights,
        bool batch_qkv) {
    if (!items || count < 2 || !model || !weights) return false;
    ds4_gpu_graph *batch = &items[0].session->graph;
    const uint64_t norm_row_bytes =
        (uint64_t)DS4_N_EMBD * sizeof(float);
    const char *stage = "embed";
    uint32_t failed_layer = UINT32_MAX;
    bool ok = true;

    for (int i = 0; ok && i < count; i++) {
        ds4_gpu_graph *g = &items[i].session->graph;
        metal_graph_dspark_capture_begin(g);
        ok = ds4_gpu_embed_token_hc_tensor(
                metal_graph_cur_hc(g),
                model->map,
                model->size,
                weights->token_embd->abs_offset,
                (uint32_t)weights->token_embd->dim[1],
                (uint32_t)items[i].token,
                DS4_N_EMBD,
                DS4_N_HC) != 0;
    }

    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
        const uint64_t shared_row_bytes = shared_dim * sizeof(float);
        failed_layer = il;

        if (batch_qkv) {
            stage = "decode-to-qkv";
            for (int i = 0; ok && i < count; i++) {
                ds4_session *s = items[i].session;
                ds4_gpu_graph *g = &s->graph;
                const uint32_t pos = (uint32_t)s->checkpoint.len;
                const uint32_t raw_row = pos % g->raw_cap;
                const uint32_t n_raw =
                    metal_graph_raw_span_for_batch(g, pos, 1);
                ok = metal_graph_encode_decode_layer_phase(
                        g, model, layer, il, pos,
                        g->layer_raw_cache[il], g->raw_cap, raw_row, n_raw,
                        items[i].token, METAL_DECODE_LAYER_TO_QKV);
            }
            stage = "gather-qkv";
            for (int i = 0; ok && i < count; i++) {
                ok = ds4_gpu_tensor_copy(
                        metal_graph_batch_attn_norm(batch),
                        (uint64_t)i * norm_row_bytes,
                        metal_graph_attn_norm(&items[i].session->graph),
                        0,
                        norm_row_bytes) != 0;
            }

            const uint64_t q_rank = layer->attn_q_a->dim[1];
            const uint64_t qr_row_bytes = q_rank * sizeof(float);
            const uint64_t kv_row_bytes =
                (uint64_t)DS4_N_HEAD_DIM * sizeof(float);
            ds4_gpu_tensor *qr = NULL;
            ds4_gpu_tensor *kv_raw = NULL;
            if (ok) {
                qr = ds4_gpu_tensor_view(
                        metal_graph_batch_qr(batch), 0,
                        (uint64_t)count * qr_row_bytes);
                kv_raw = ds4_gpu_tensor_view(
                        metal_graph_batch_kv_raw(batch), 0,
                        (uint64_t)count * kv_row_bytes);
                ok = qr && kv_raw;
            }
            stage = "qkv";
            if (ok) {
                ok = ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                        qr,
                        model->map,
                        model->size,
                        layer->attn_q_a->abs_offset,
                        DS4_N_EMBD,
                        q_rank,
                        metal_graph_batch_attn_norm(batch),
                        (uint32_t)count) != 0 &&
                     ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                        kv_raw,
                        model->map,
                        model->size,
                        layer->attn_kv->abs_offset,
                        DS4_N_EMBD,
                        DS4_N_HEAD_DIM,
                        metal_graph_batch_attn_norm(batch),
                        (uint32_t)count) != 0;
            }
            stage = "decode-from-qkv";
            for (int i = 0; ok && i < count; i++) {
                ds4_session *s = items[i].session;
                ds4_gpu_graph *g = &s->graph;
                const int tier = g->active_tier;
                ds4_gpu_tensor *qr_row = ds4_gpu_tensor_view(
                        qr, (uint64_t)i * qr_row_bytes, qr_row_bytes);
                ds4_gpu_tensor *kv_row = ds4_gpu_tensor_view(
                        kv_raw, (uint64_t)i * kv_row_bytes, kv_row_bytes);
                ds4_gpu_tensor *saved_qr = g->qr_by_tier[tier];
                ds4_gpu_tensor *saved_kv_raw = g->kv_raw_by_tier[tier];
                ok = qr_row && kv_row;
                if (ok) {
                    g->qr_by_tier[tier] = qr_row;
                    g->kv_raw_by_tier[tier] = kv_row;
                    const uint32_t pos = (uint32_t)s->checkpoint.len;
                    const uint32_t raw_row = pos % g->raw_cap;
                    const uint32_t n_raw =
                        metal_graph_raw_span_for_batch(g, pos, 1);
                    ok = metal_graph_encode_decode_layer_phase(
                            g, model, layer, il, pos,
                            g->layer_raw_cache[il], g->raw_cap,
                            raw_row, n_raw, items[i].token,
                            METAL_DECODE_LAYER_FROM_QA_KV_RAW_TO_SHARED_MID);
                    g->qr_by_tier[tier] = saved_qr;
                    g->kv_raw_by_tier[tier] = saved_kv_raw;
                }
                ds4_gpu_tensor_free(kv_row);
                ds4_gpu_tensor_free(qr_row);
            }
            ds4_gpu_tensor_free(kv_raw);
            ds4_gpu_tensor_free(qr);
        } else {
            stage = "decode-to-shared";
            for (int i = 0; ok && i < count; i++) {
                ds4_session *s = items[i].session;
                ds4_gpu_graph *g = &s->graph;
                const uint32_t pos = (uint32_t)s->checkpoint.len;
                const uint32_t raw_row = pos % g->raw_cap;
                const uint32_t n_raw =
                    metal_graph_raw_span_for_batch(g, pos, 1);
                ok = metal_graph_encode_decode_layer_phase(
                        g, model, layer, il, pos,
                        g->layer_raw_cache[il], g->raw_cap, raw_row, n_raw,
                        items[i].token, METAL_DECODE_LAYER_TO_SHARED_MID);
            }
        }

        stage = "gather-norm";
        for (int i = 0; ok && i < count; i++) {
            ok = ds4_gpu_tensor_copy(
                    metal_graph_batch_ffn_norm(batch),
                    (uint64_t)i * norm_row_bytes,
                    metal_graph_ffn_norm(&items[i].session->graph),
                    0,
                    norm_row_bytes) != 0;
        }

        ds4_gpu_tensor *gate = NULL;
        ds4_gpu_tensor *up = NULL;
        ds4_gpu_tensor *mid = NULL;
        if (ok) {
            gate = ds4_gpu_tensor_view(
                    metal_graph_batch_shared_gate(batch), 0,
                    (uint64_t)count * shared_row_bytes);
            up = ds4_gpu_tensor_view(
                    metal_graph_batch_shared_up(batch), 0,
                    (uint64_t)count * shared_row_bytes);
            mid = ds4_gpu_tensor_view(
                    metal_graph_batch_shared_mid(batch), 0,
                    (uint64_t)count * shared_row_bytes);
            ok = gate && up && mid;
        }
        stage = "shared-gate-up";
        if (ok) {
            ok = ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                    gate, up, mid,
                    model->map,
                    model->size,
                    layer->ffn_gate_shexp->abs_offset,
                    layer->ffn_up_shexp->abs_offset,
                    DS4_N_EMBD,
                    shared_dim,
                    metal_graph_batch_ffn_norm(batch),
                    (uint32_t)count,
                    DS4_SWIGLU_CLAMP_EXP) != 0;
        }

        stage = "shared-down";
        for (int i = 0; ok && i < count; i++) {
            ds4_gpu_graph *g = &items[i].session->graph;
            ds4_gpu_tensor *mid_row = ds4_gpu_tensor_view(
                    mid, (uint64_t)i * shared_row_bytes,
                    shared_row_bytes);
            ok = mid_row &&
                 ds4_gpu_shared_down_hc_expand_q8_0_tensor(
                         metal_graph_after_ffn_hc(g),
                         metal_graph_shared_out(g),
                         model->map,
                         model->size,
                         layer->ffn_down_shexp->abs_offset,
                         shared_dim,
                         DS4_N_EMBD,
                         mid_row,
                         metal_graph_routed_out(g),
                         metal_graph_after_attn_hc(g),
                         metal_graph_hc_split(g),
                         DS4_N_EMBD,
                         DS4_N_HC) != 0;
            ds4_gpu_tensor_free(mid_row);
        }
        for (int i = 0; ok && i < count; i++) {
            ds4_gpu_graph *g = &items[i].session->graph;
            ds4_gpu_tensor *tmp = metal_graph_cur_hc(g);
            g->cur_hc_by_tier[g->active_tier] =
                metal_graph_after_ffn_hc(g);
            g->after_ffn_hc_by_tier[g->active_tier] = tmp;
            ok = metal_graph_dspark_capture_decode_layer(g, il);
        }
        ds4_gpu_tensor_free(mid);
        ds4_gpu_tensor_free(up);
        ds4_gpu_tensor_free(gate);
    }

    stage = "output";
    for (int i = 0; ok && i < count; i++) {
        ok = metal_graph_encode_output_head(
                &items[i].session->graph, model, weights,
                weights->output->dim[1]);
    }
    if (!ok) {
        fprintf(stderr,
                "ds4: Metal native shared session batch failed "
                "stage=%s layer=%u rows=%d\n",
                stage, failed_layer, count);
    }
    return ok;
}

static ds4_tp_batch_item *ds4_sessions_tp_batch_items(
        const ds4_decode_item *items, int count) {
    if (!items || count <= 0) return NULL;
    ds4_tp_batch_item *wire =
        calloc((size_t)count, sizeof(*wire));
    if (!wire) return NULL;
    for (int i = 0; i < count; i++) {
        if (!items[i].session || items[i].session->tp_session_id == 0) {
            free(wire);
            return NULL;
        }
        wire[i].session_id = items[i].session->tp_session_id;
        wire[i].token = items[i].token;
    }
    return wire;
}

static bool ds4_sessions_tp_recv_logits(
        ds4_engine *e, ds4_session *prefill,
        ds4_decode_item *items, int count,
        char *err, size_t errlen) {
    if (!e || !e->tp.active || e->tp.rank != 0 || !e->tp.vocab_split) {
        return true;
    }
    const uint32_t vhalf = (uint32_t)DS4_N_VOCAB / 2u;
    if (prefill &&
        !ds4_tp_recv_logits_half(e->tp.ctx,
                                 prefill->logits + vhalf, vhalf)) {
        if (err && errlen) snprintf(err, errlen,
                                    "tp: worker mixed-prefill logits missing");
        return false;
    }
    for (int i = 0; i < count; i++) {
        if (!ds4_tp_recv_logits_half(e->tp.ctx,
                                     items[i].session->logits + vhalf,
                                     vhalf)) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "tp: worker batch logits missing for item %d", i);
            }
            return false;
        }
    }
    return true;
}

static int ds4_sessions_eval_batch_metal(
        ds4_decode_item *items,
        int count,
        ds4_engine *e,
        char *err,
        size_t errlen) {
    const bool mirror = e->tp.active && e->tp.rank == 0;
    if (mirror) {
        ds4_tp_batch_item *wire = ds4_sessions_tp_batch_items(items, count);
        const bool sent = wire &&
            ds4_tp_send_eval_batch(e->tp.ctx, wire, (uint32_t)count);
        free(wire);
        if (!sent) {
            if (err && errlen) snprintf(err, errlen,
                                        "tp: worker decode batch send failed");
            for (int i = 0; i < count; i++) {
                ds4_session_invalidate(items[i].session);
            }
            return 1;
        }
    }

#if defined(__APPLE__)
    if (e->tp.active) ds4_gpu_tp_set_session_batch_mode(1);
#endif
    bool ok = ds4_gpu_begin_commands() != 0;
    const bool native_shared = ok &&
        metal_graph_native_session_batch_shared_supported(items, count, e);
    const bool native_qkv = native_shared &&
        metal_graph_native_session_batch_qkv_supported(items, count, e);
    if (native_shared) {
        ok = metal_graph_encode_native_session_batch_shared(
                items, count, &e->model, &e->weights, native_qkv);
    } else {
        for (int i = 0; ok && i < count; i++) {
            ds4_session *s = items[i].session;
            const uint32_t pos = (uint32_t)s->checkpoint.len;
            if (ds4_session_is_glm(s)) {
                ok = glm_graph_forward_token(&s->glm_graph,
                                             &e->model,
                                             &e->weights,
                                             items[i].token,
                                             NULL,
                                             pos,
                                             NULL,
                                             s->logits,
                                             true);
            } else {
                ok = metal_graph_encode_token_raw_swa(&s->graph,
                                                      &e->model,
                                                      &e->weights,
                                                      items[i].token,
                                                      pos,
                                                      true,
                                                      false);
            }
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
#if defined(__APPLE__)
    if (e->tp.active) ds4_gpu_tp_set_session_batch_mode(0);
    if (ok && e->tp.active && ds4_gpu_tp_failed()) {
        if (err && errlen) snprintf(err, errlen, "tp: batch gate transport failed");
        ok = false;
    }
#endif

    for (int i = 0; ok && i < count; i++) {
        ds4_session *s = items[i].session;
        ds4_gpu_tensor *logits = ds4_session_is_glm(s)
            ? s->glm_graph.logits : metal_graph_logits(&s->graph);
        ok = ds4_gpu_tensor_read(logits,
                                 0,
                                 s->logits,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok && mirror) {
        ok = ds4_tp_wait_command_ack(e->tp.ctx, 0, "decode batch",
                                     err, errlen) &&
             ds4_sessions_tp_recv_logits(e, NULL, items, count,
                                          err, errlen);
    }
    if (!ok) {
        for (int i = 0; i < count; i++) {
            ds4_session_invalidate(items[i].session);
        }
        if (err && errlen && !err[0]) {
            snprintf(err, errlen, "Metal batched decode failed");
        }
        return 1;
    }

    for (int i = 0; i < count; i++) {
        ds4_session *s = items[i].session;
        const uint32_t pos = (uint32_t)s->checkpoint.len;
        const bool updates_dense = ds4_session_is_glm(s) &&
            glm_graph_decode_updates_dense_cache(&s->glm_graph,
                                                 pos,
                                                 s->logits);
        token_vec_push(&s->checkpoint, items[i].token);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        if (updates_dense) ds4_session_glm_note_dense_cache(s, pos, 1);
        if (!ds4_session_is_glm(s)) {
            ds4_session_dspark_capture_note_checkpoint(s);
        }
    }
    if (getenv("DS4_METAL_SESSION_BATCH_LOG") != NULL) {
        fprintf(stderr,
                "ds4: Metal session batch rows=%d family=%s "
                "native_shared=%d native_qkv=%d\n",
                count,
                ds4_session_is_glm(items[0].session) ? "glm" : "deepseek",
                native_shared ? 1 : 0,
                native_qkv ? 1 : 0);
    }
    return 0;
}

static bool ds4_sessions_eval_batch_with_prefill_metal_supported(
        ds4_decode_item *items,
        int count,
        ds4_session *prefill_session,
        const ds4_tokens *prefill_prompt) {
    if (!items || count <= 0 || !prefill_session || !prefill_prompt ||
        !prefill_session->engine) {
        return false;
    }
    ds4_engine *e = prefill_session->engine;
    const char *tp_batch = getenv("DS4_METAL_TP_SESSION_BATCH");
    if (e->backend != DS4_BACKEND_METAL ||
        e->support_kind != DS4_SUPPORT_NONE ||
        (e->tp.active && tp_batch && strcmp(tp_batch, "0") == 0) ||
        ds4_session_is_cpu(prefill_session) ||
        ds4_session_is_glm(prefill_session) ||
        prefill_session->distributed ||
        prefill_session->graph.ssd_streaming ||
        ds4_session_cancelled(prefill_session) ||
        getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != NULL ||
        getenv("DS4_METAL_GRAPH_PREFILL_SPLIT_PROFILE") != NULL ||
        getenv("DS4_METAL_LAYER_STAGE_PROFILE") != NULL) {
        return false;
    }

    const uint32_t start = (uint32_t)prefill_session->checkpoint.len;
    const uint32_t rows = (uint32_t)prefill_prompt->len - start;
    ds4_gpu_graph *g = &prefill_session->graph;
    if (rows < metal_graph_resume_prefill_min_tokens() ||
        rows > g->prefill_cap || rows > g->raw_cap ||
        (start % g->prefill_cap) + rows > g->prefill_cap) {
        return false;
    }

    for (int i = 0; i < count; i++) {
        ds4_session *s = items[i].session;
        if (!s || s->engine != e || s == prefill_session ||
            s->distributed || ds4_session_is_cpu(s) ||
            ds4_session_is_glm(s) || !s->checkpoint_valid ||
            s->graph.ssd_streaming || ds4_session_cancelled(s)) {
            return false;
        }
    }
    return true;
}

/* Interleave one resumed Metal prefill chunk and independent decode rows by
 * layer in a single command epoch. Arithmetic stays on the established
 * per-sequence paths, so every session retains private cache and checkpoint
 * state while the scheduler avoids serial prefill/decode submissions. */
static int ds4_sessions_eval_batch_with_prefill_metal(
        ds4_decode_item *items,
        int count,
        ds4_session *prefill_session,
        const ds4_tokens *prefill_prompt,
        char *err,
        size_t errlen) {
    ds4_engine *e = prefill_session->engine;
    ds4_gpu_graph *pg = &prefill_session->graph;
    const uint32_t start = (uint32_t)prefill_session->checkpoint.len;
    const uint32_t rows = (uint32_t)prefill_prompt->len - start;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const bool mirror = e->tp.active && e->tp.rank == 0;

    if (mirror) {
        ds4_tp_batch_item *wire = ds4_sessions_tp_batch_items(items, count);
        const bool sent = wire &&
            prefill_session->tp_session_id != 0 &&
            ds4_tp_send_mixed_batch(
                e->tp.ctx, prefill_session->tp_session_id,
                prefill_prompt->v, (uint32_t)prefill_prompt->len,
                wire, (uint32_t)count);
        free(wire);
        if (!sent) {
            if (err && errlen) snprintf(err, errlen,
                                        "tp: worker mixed batch send failed");
            ds4_session_invalidate(prefill_session);
            for (int i = 0; i < count; i++) {
                ds4_session_invalidate(items[i].session);
            }
            return 1;
        }
    }

    bool ok = metal_graph_set_active_tier_batch(pg, pg->emb_tier, rows) &&
              metal_graph_upload_prompt_tokens(
                      metal_graph_prefill_tokens(pg), prefill_prompt,
                      start, rows) &&
              metal_graph_warmup_prefill_kernels(
                      pg, &e->model, &e->weights, rows) &&
              metal_graph_set_active_tier_batch(pg, pg->emb_tier, rows) &&
              metal_graph_upload_prompt_embeddings_hc(
                      metal_graph_batch_cur_hc(pg),
                      metal_graph_prefill_tokens(pg),
                      &e->model, &e->weights, prefill_prompt, start, rows);
    metal_graph_dspark_capture_begin_prefill(pg);

    for (int i = 0; ok && i < count; i++) {
        ds4_gpu_graph *g = &items[i].session->graph;
        metal_graph_dspark_capture_begin(g);
        ok = metal_graph_set_active_tier_decode(g, g->emb_tier) &&
             ds4_gpu_embed_token_hc_tensor(
                     metal_graph_cur_hc(g),
                     e->model.map,
                     e->model.size,
                     e->weights.token_embd->abs_offset,
                     (uint32_t)e->weights.token_embd->dim[1],
                     (uint32_t)items[i].token,
                     DS4_N_EMBD,
                     DS4_N_HC) != 0;
    }

#if defined(__APPLE__)
    if (e->tp.active) ds4_gpu_tp_set_session_batch_mode(1);
#endif
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ok = metal_graph_encode_layer_batch(
                pg, &e->model, &e->weights.layer[il], il, start, rows);
        if (ok) {
            ok = metal_graph_dspark_capture_prefill_layer(pg, il, start, rows);
        }
    }
    /* Decode graphs retain their established session-major ordering. Some
     * decode kernels use the engine-shared prefill scratch as transient
     * reduction space, so alternating prefill and decode at layer granularity
     * would let graph-local ping-pong orientations alias the same allocation. */
    for (int i = 0; ok && i < count; i++) {
        ds4_session *s = items[i].session;
        ds4_gpu_graph *g = &s->graph;
        const uint32_t pos = (uint32_t)s->checkpoint.len;
        const uint32_t raw_row = pos % g->raw_cap;
        const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
        for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
            ok = metal_graph_encode_decode_layer(
                    g, &e->model, &e->weights.layer[il], il, pos,
                    g->layer_raw_cache[il], g->raw_cap, raw_row, n_raw,
                    items[i].token);
            if (ok) {
                ds4_gpu_tensor *tmp = metal_graph_cur_hc(g);
                g->cur_hc_by_tier[g->active_tier] =
                    metal_graph_after_ffn_hc(g);
                g->after_ffn_hc_by_tier[g->active_tier] = tmp;
                ok = metal_graph_dspark_capture_decode_layer(g, il);
            }
        }
    }

    const int prefill_src_tier = pg->active_tier;
    ds4_gpu_tensor *saved_prefill_cur = NULL;
    ds4_gpu_tensor *last_prefill_hc = NULL;
    if (ok) {
        saved_prefill_cur = pg->cur_hc_by_tier[prefill_src_tier];
        last_prefill_hc = metal_graph_tensor_row_view(
                metal_graph_batch_cur_hc(pg), rows - 1u, hc_dim);
        ok = last_prefill_hc != NULL;
    }
    if (ok) {
        pg->cur_hc_by_tier[prefill_src_tier] = last_prefill_hc;
        ok = metal_graph_encode_output_head(
                pg, &e->model, &e->weights, e->weights.output->dim[1]);
        pg->cur_hc_by_tier[prefill_src_tier] = saved_prefill_cur;
    }
    for (int i = 0; ok && i < count; i++) {
        ok = metal_graph_encode_output_head(
                &items[i].session->graph, &e->model, &e->weights,
                e->weights.output->dim[1]);
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
#if defined(__APPLE__)
    if (e->tp.active) ds4_gpu_tp_set_session_batch_mode(0);
    if (ok && e->tp.active && ds4_gpu_tp_failed()) {
        if (err && errlen) snprintf(err, errlen,
                                    "tp: mixed batch gate transport failed");
        ok = false;
    }
#endif
    if (saved_prefill_cur) {
        pg->cur_hc_by_tier[prefill_src_tier] = saved_prefill_cur;
    }
    ds4_gpu_tensor_free(last_prefill_hc);

    if (ok) {
        ok = ds4_gpu_tensor_read(
                metal_graph_logits(pg), 0, prefill_session->logits,
                (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    for (int i = 0; ok && i < count; i++) {
        ds4_session *s = items[i].session;
        ok = ds4_gpu_tensor_read(
                metal_graph_logits(&s->graph), 0, s->logits,
                (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok && mirror) {
        ok = ds4_tp_wait_command_ack(
                 e->tp.ctx, prefill_session->tp_session_id,
                 "mixed prefill/decode batch", err, errlen) &&
             ds4_sessions_tp_recv_logits(
                 e, prefill_session, items, count, err, errlen);
    }
    if (!ok) {
        ds4_session_invalidate(prefill_session);
        for (int i = 0; i < count; i++) {
            ds4_session_invalidate(items[i].session);
        }
        if (err && errlen && !err[0]) {
            snprintf(err, errlen, "Metal mixed model batch failed");
        }
        return 1;
    }

    ds4_tokens_copy(&prefill_session->checkpoint, prefill_prompt);
    prefill_session->checkpoint_valid = true;
    prefill_session->mtp_draft_valid = false;
    ds4_session_dspark_capture_note_checkpoint(prefill_session);
    session_greedy_splitkv_reset(prefill_session);
    if (prefill_session->progress) {
        prefill_session->progress(
                prefill_session->progress_ud, "prefill_chunk",
                prefill_prompt->len, prefill_prompt->len);
    }
    for (int i = 0; i < count; i++) {
        ds4_session *s = items[i].session;
        token_vec_push(&s->checkpoint, items[i].token);
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        ds4_session_dspark_capture_note_checkpoint(s);
    }
    if (getenv("DS4_METAL_SESSION_BATCH_LOG") != NULL) {
        fprintf(stderr,
                "ds4: Metal mixed batch prefill_rows=%u decode_rows=%d\n",
                rows, count);
    }
    return 0;
}
#endif
