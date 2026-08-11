#ifndef DS4_NO_GPU
/* Execute one resumed prefill quantum together with one exact decode batch.
 * Attention and non-routed FFN arithmetic remain on their original paths. */
static bool metal_graph_eval_mixed_prefill_decode(
        ds4_session *prefill_session,
        const token_vec *prompt,
        uint32_t start,
        uint32_t prefill_rows,
        ds4_decode_item *decode_items,
        int decode_count,
        const ds4_model *model,
        const ds4_weights *weights) {
    if (!metal_graph_mixed_prefill_decode_supported(
                prefill_session, prompt, start, prefill_rows,
                decode_items, decode_count, weights)) {
        return false;
    }

    const bool group_attn_core =
        metal_graph_tp_env_flag("DS4_CUDA_SESSION_BATCH_ATTN_CORE", true) &&
        metal_graph_session_batch_attn_core_supported(
                decode_items, decode_count, weights);
    const bool group_attn_post =
        group_attn_core &&
        metal_graph_tp_env_flag("DS4_CUDA_SESSION_BATCH_ATTN_POST", true) &&
        metal_graph_session_batch_attn_post_supported(
                decode_items, decode_count, weights);
    const bool group_qkv =
        group_attn_core &&
        metal_graph_tp_env_flag("DS4_CUDA_SESSION_BATCH_QKV", true) &&
        metal_graph_session_batch_qkv_supported(
                decode_items, decode_count, weights);
    const bool group_kv_store =
        group_qkv &&
        metal_graph_tp_env_flag("DS4_CUDA_SESSION_BATCH_KV_STORE", true) &&
        metal_graph_session_batch_kv_store_supported(
                decode_items, decode_count);

    ds4_gpu_graph *g = &prefill_session->graph;
    for (int i = 0; i < decode_count; i++) {
        metal_graph_align_mixed_workspace(
                g, &decode_items[i].session->graph);
    }
    g->batch_token_offset = 0;
    if (!metal_graph_set_active_tier_batch(g, g->emb_tier, prefill_rows)) {
        return false;
    }
    bool ok = metal_graph_upload_prompt_tokens(
            metal_graph_prefill_tokens(g), prompt, start, prefill_rows);
    if (ok) {
        int32_t *tokens = malloc((size_t)decode_count * sizeof(*tokens));
        if (!tokens) {
            ok = false;
        } else {
            for (int i = 0; i < decode_count; i++) {
                tokens[i] = decode_items[i].token;
            }
            ok = ds4_gpu_tensor_write(
                    metal_graph_prefill_tokens(g),
                    (uint64_t)prefill_rows * sizeof(int32_t),
                    tokens,
                    (uint64_t)decode_count * sizeof(*tokens)) != 0;
            free(tokens);
        }
    }
    if (ok) {
        ok = metal_graph_warmup_prefill_kernels(
                g, model, weights, prefill_rows);
    }
    if (ok) {
        ok = metal_graph_set_active_tier_batch(g, g->emb_tier, prefill_rows);
    }
    if (ok) {
        ok = metal_graph_upload_prompt_embeddings_hc(
                metal_graph_batch_cur_hc(g),
                metal_graph_prefill_tokens(g),
                model,
                weights,
                prompt,
                start,
                prefill_rows);
    }
    metal_graph_dspark_capture_begin_prefill(g);
    for (int i = 0; ok && i < decode_count; i++) {
        ds4_gpu_graph *dg = &decode_items[i].session->graph;
        metal_graph_dspark_capture_begin(dg);
        ok = metal_graph_set_active_tier_decode(dg, dg->emb_tier);
        if (ok) {
            ok = ds4_gpu_embed_token_hc_tensor(
                    metal_graph_cur_hc(dg),
                    model->map,
                    model->size,
                    weights->token_embd->abs_offset,
                    (uint32_t)weights->token_embd->dim[1],
                    (uint32_t)decode_items[i].token,
                    DS4_N_EMBD,
                    DS4_N_HC) != 0;
        }
    }

    if (ok) ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        const int home_tier = g->placement[il + 1u];
        ok = metal_graph_set_active_tier_batch(g, home_tier, prefill_rows) &&
             metal_graph_layer_stage_profile_start(il);
        if (ok) {
            ok = metal_graph_encode_layer_attention_batch(
                    g, model, layer, il, start, prefill_rows);
        }
        if (ok) {
            ok = metal_graph_encode_attn_pre_session_batch(
                    decode_items, decode_count, model, layer, il,
                    prefill_rows, false);
        }
        if (ok && group_qkv) {
            ok = metal_graph_encode_qkv_session_batch(
                    decode_items, decode_count, model, layer, il,
                    prefill_rows, group_kv_store);
        }
        for (int i = 0; ok && i < decode_count; i++) {
            ds4_session *s = decode_items[i].session;
            ds4_gpu_graph *dg = &s->graph;
            const uint32_t pos = (uint32_t)s->checkpoint.len;
            const uint32_t raw_row = pos % dg->raw_cap;
            const uint32_t n_raw = metal_graph_raw_span_for_batch(dg, pos, 1);
            metal_graph_attn_pre_alias alias = {0};
            ok = dg->placement[il + 1u] == home_tier &&
                 metal_graph_bind_attn_pre_batch_row(
                    dg, g, layer, prefill_rows + (uint32_t)i,
                    group_attn_core, &alias);
            if (ok) {
                ok = metal_graph_encode_decode_layer_phase(
                        dg, model, layer, il, pos,
                        dg->layer_raw_cache[il], dg->raw_cap,
                        raw_row, n_raw, decode_items[i].token,
                        !group_attn_core ?
                            METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_FFN :
                        group_kv_store ?
                            METAL_DECODE_LAYER_FROM_KV_STORE_TO_ATTN :
                        group_qkv ? METAL_DECODE_LAYER_FROM_QKV_TO_ATTN :
                                    METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_ATTN);
            }
            metal_graph_unbind_attn_pre_batch_row(dg, &alias);
        }
        if (ok && group_attn_core) {
            ok = metal_graph_encode_attention_session_batch(
                    decode_items, decode_count, model, layer, il,
                    prefill_rows);
        }
        if (ok && group_attn_post) {
            ok = metal_graph_encode_attn_post_session_batch(
                    decode_items, decode_count, model, layer, il,
                    prefill_rows);
        }
        for (int i = 0;
             ok && group_attn_core && !group_attn_post && i < decode_count;
             i++) {
            ds4_session *s = decode_items[i].session;
            ds4_gpu_graph *dg = &s->graph;
            const uint32_t pos = (uint32_t)s->checkpoint.len;
            const uint32_t raw_row = pos % dg->raw_cap;
            const uint32_t n_raw = metal_graph_raw_span_for_batch(dg, pos, 1);
            metal_graph_attn_pre_alias alias = {0};
            ok = metal_graph_bind_attn_pre_batch_row(
                    dg, g, layer, prefill_rows + (uint32_t)i,
                    true, &alias);
            if (ok) {
                ok = metal_graph_encode_decode_layer_phase(
                        dg, model, layer, il, pos,
                        dg->layer_raw_cache[il], dg->raw_cap,
                        raw_row, n_raw, decode_items[i].token,
                        METAL_DECODE_LAYER_FROM_ATTN_TO_FFN);
            }
            metal_graph_unbind_attn_pre_batch_row(dg, &alias);
        }
        if (ok) {
            ok = metal_graph_encode_ffn_pre_session_batch(
                    decode_items, decode_count, model, layer,
                    prefill_rows, !group_attn_core);
        }
        if (ok) {
            ok = metal_graph_encode_layer_ffn_batch(
                    g, model, layer, il, start, prefill_rows, NULL, 0);
        }
        if (ok) {
            ok = metal_graph_encode_routed_session_batch(
                    decode_items, decode_count, model, layer, il,
                    prefill_rows, false, false);
        }
        if (ok) {
            ok = metal_graph_encode_shared_session_batch(
                    decode_items, decode_count, model, layer,
                    prefill_rows, false);
        }
        if (ok) {
            ds4_gpu_tensor *tmp = metal_graph_batch_cur_hc(g);
            g->batch_cur_hc_by_tier[g->active_tier] = metal_graph_batch_next_hc(g);
            g->batch_next_hc_by_tier[g->active_tier] = tmp;
            ok = metal_graph_dspark_capture_prefill_layer(
                    g, il, start, prefill_rows);
        }
        for (int i = 0; ok && i < decode_count; i++) {
            ds4_gpu_graph *dg = &decode_items[i].session->graph;
            ds4_gpu_tensor *tmp = metal_graph_cur_hc(dg);
            dg->cur_hc_by_tier[dg->active_tier] = metal_graph_after_ffn_hc(dg);
            dg->after_ffn_hc_by_tier[dg->active_tier] = tmp;
            ok = metal_graph_dspark_capture_decode_layer(dg, il);
        }
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const int prefill_src_tier = g->active_tier;
    ds4_gpu_tensor *saved_prefill_cur =
        prefill_src_tier >= 0 && prefill_src_tier < DS4_MAX_GPUS
            ? g->cur_hc_by_tier[prefill_src_tier] : NULL;
    ds4_gpu_tensor *last_prefill_hc = NULL;
    if (ok && !saved_prefill_cur) ok = false;
    if (ok) {
        last_prefill_hc = metal_graph_tensor_row_view(
                metal_graph_batch_cur_hc(g), prefill_rows - 1u, hc_dim);
        ok = last_prefill_hc != NULL;
    }
    if (ok) {
        g->cur_hc_by_tier[prefill_src_tier] = last_prefill_hc;
        ok = metal_graph_encode_output_head(
                g, model, weights, weights->output->dim[1]);
        g->cur_hc_by_tier[prefill_src_tier] = saved_prefill_cur;
    }
    for (int i = 0; ok && i < decode_count; i++) {
        ds4_session *s = decode_items[i].session;
        ok = metal_graph_encode_output_head(
                &s->graph, model, weights, weights->output->dim[1]);
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (saved_prefill_cur) {
        g->cur_hc_by_tier[prefill_src_tier] = saved_prefill_cur;
    }
    ds4_gpu_tensor_free(last_prefill_hc);
    g->batch_token_offset = 0;

    if (ok && g->active_tier != prefill_src_tier) {
        ok = metal_graph_set_active_tier_no_copy(g, prefill_src_tier);
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(
                metal_graph_logits(g), 0, prefill_session->logits,
                (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    for (int i = 0; ok && i < decode_count; i++) {
        ds4_session *s = decode_items[i].session;
        ok = ds4_gpu_tensor_read(
                metal_graph_logits(&s->graph), 0, s->logits,
                (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    if (ok) ok = metal_graph_cuda_tp_attn_cache_sync_all(g);
    return ok;
}
#endif /* !DS4_NO_GPU */
