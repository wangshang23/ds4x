#ifndef DS4_NO_GPU
/* Batch Q/KV work while preserving each session's position. The optional KV
 * finalizer writes contiguous rows directly to the private raw caches. */
static bool metal_graph_encode_qkv_session_batch(
        ds4_decode_item *items,
        int count,
        const ds4_model *model,
        const ds4_layer_weights *layer,
        uint32_t il,
        uint32_t row_base,
        bool store_private_kv) {
    if (!items || count < 2 || !model || !layer || il >= DS4_N_LAYER ||
        count > (int)DS4_GPU_ATTENTION_DECODE_BATCH_MAX) {
        return false;
    }
    ds4_gpu_graph *g = &items[0].session->graph;
    const int home = g->placement[il + 1u];
    const uint32_t rows = (uint32_t)count;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    if (home < 0 || (uint64_t)row_base + rows > g->prefill_cap) {
        return false;
    }
    for (int i = 0; i < count; i++) {
        if (!metal_graph_set_active_tier_decode(
                    &items[i].session->graph, home)) {
            return false;
        }
    }

    ds4_gpu_tensor attn_norm, qr, qr_norm, q, kv_raw, kv;
    bool ok =
        metal_graph_borrow_tensor_view(
                &attn_norm, metal_graph_batch_attn_norm(g),
                (uint64_t)row_base * DS4_N_EMBD * sizeof(float),
                (uint64_t)rows * DS4_N_EMBD * sizeof(float)) &&
        metal_graph_borrow_tensor_view(
                &qr, metal_graph_batch_qr(g),
                (uint64_t)row_base * q_rank * sizeof(float),
                (uint64_t)rows * q_rank * sizeof(float)) &&
        metal_graph_borrow_tensor_view(
                &qr_norm, metal_graph_batch_qr_norm(g),
                (uint64_t)row_base * q_rank * sizeof(float),
                (uint64_t)rows * q_rank * sizeof(float)) &&
        metal_graph_borrow_tensor_view(
                &q, metal_graph_batch_q(g),
                (uint64_t)row_base * q_dim * sizeof(float),
                (uint64_t)rows * q_dim * sizeof(float)) &&
        metal_graph_borrow_tensor_view(
                &kv_raw, metal_graph_batch_kv_raw(g),
                (uint64_t)row_base * DS4_N_HEAD_DIM * sizeof(float),
                (uint64_t)rows * DS4_N_HEAD_DIM * sizeof(float)) &&
        metal_graph_borrow_tensor_view(
                &kv, metal_graph_batch_kv(g),
                (uint64_t)row_base * DS4_N_HEAD_DIM * sizeof(float),
                (uint64_t)rows * DS4_N_HEAD_DIM * sizeof(float));

    ds4_gpu_attention_decode_row positions[DS4_GPU_ATTENTION_DECODE_BATCH_MAX];
    memset(positions, 0, sizeof(positions));
    for (uint32_t i = 0; i < rows; i++) {
        positions[i].pos =
            (uint32_t)items[i].session->checkpoint.len;
    }
    const bool compressed = ds4_layer_compress_ratio(il) != 0u;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor =
        compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    if (ok) {
        ok = ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
                &qr, &kv_raw, model->map, model->size,
                layer->attn_q_a->abs_offset,
                layer->attn_kv->abs_offset,
                DS4_N_EMBD, q_rank, DS4_N_HEAD_DIM,
                &attn_norm, rows) != 0;
    }
    if (ok) {
        ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                &qr_norm, &qr, model->map, model->size,
                layer->attn_q_a_norm->abs_offset, (uint32_t)q_rank,
                &kv, &kv_raw, layer->attn_kv_a_norm->abs_offset,
                DS4_N_HEAD_DIM, rows, DS4_RMS_EPS) != 0;
    }
    if (ok) {
        ok = ds4_gpu_rope_tail_decode_rows_tensor(
                &kv, positions, rows, DS4_N_HEAD_KV, DS4_N_HEAD_DIM,
                DS4_N_ROT,
                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0u,
                false, freq_base, freq_scale, ext_factor, attn_factor,
                DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    if (ok) {
        ok = ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
                &q, model->map, model->size,
                layer->attn_q_b->abs_offset, q_rank, q_dim,
                &qr_norm, rows) != 0;
    }
    if (ok) {
        ok = ds4_gpu_head_rms_norm_tensor(
                &q, rows, DS4_N_HEAD, DS4_N_HEAD_DIM,
                DS4_RMS_EPS) != 0;
    }
    if (ok) {
        ok = ds4_gpu_rope_tail_decode_rows_tensor(
                &q, positions, rows, DS4_N_HEAD, DS4_N_HEAD_DIM,
                DS4_N_ROT,
                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0u,
                false, freq_base, freq_scale, ext_factor, attn_factor,
                DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW) != 0;
    }
    if (ok && store_private_kv) {
        ds4_gpu_tensor *raw_caches[DS4_GPU_ATTENTION_DECODE_BATCH_MAX];
        uint32_t raw_caps[DS4_GPU_ATTENTION_DECODE_BATCH_MAX];
        uint32_t raw_rows[DS4_GPU_ATTENTION_DECODE_BATCH_MAX];
        for (uint32_t i = 0; i < rows; i++) {
            ds4_gpu_graph *row_graph = &items[i].session->graph;
            raw_caches[i] = row_graph->layer_raw_cache[il];
            raw_caps[i] = row_graph->raw_cap;
            raw_rows[i] = positions[i].pos % raw_caps[i];
        }
        ok = ds4_gpu_kv_fp8_store_raw_decode_rows_tensor(
                &kv, raw_caches, raw_caps, raw_rows, rows,
                DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
    }
    return ok;
}

typedef struct {
    int tier;
    bool active;
    bool full_attention_rows;
    ds4_gpu_tensor attn_norm;
    ds4_gpu_tensor hc_split;
    ds4_gpu_tensor hc_pre;
    ds4_gpu_tensor hc_post;
    ds4_gpu_tensor hc_comb;
    ds4_gpu_tensor qr;
    ds4_gpu_tensor qr_norm;
    ds4_gpu_tensor q;
    ds4_gpu_tensor kv_raw;
    ds4_gpu_tensor kv;
    ds4_gpu_tensor comp_kv_cur;
    ds4_gpu_tensor comp_sc_cur;
    ds4_gpu_tensor indexer_q;
    ds4_gpu_tensor indexer_weights;
    ds4_gpu_tensor heads;
    ds4_gpu_tensor attn_low;
    ds4_gpu_tensor attn_out;
    ds4_gpu_tensor after_attn_hc;
    ds4_gpu_tensor *saved_attn_norm;
    ds4_gpu_tensor *saved_hc_split;
    ds4_gpu_tensor *saved_hc_pre;
    ds4_gpu_tensor *saved_hc_post;
    ds4_gpu_tensor *saved_hc_comb;
    ds4_gpu_tensor *saved_qr;
    ds4_gpu_tensor *saved_qr_norm;
    ds4_gpu_tensor *saved_q;
    ds4_gpu_tensor *saved_kv_raw;
    ds4_gpu_tensor *saved_kv;
    ds4_gpu_tensor *saved_comp_kv_cur;
    ds4_gpu_tensor *saved_comp_sc_cur;
    ds4_gpu_tensor *saved_indexer_q;
    ds4_gpu_tensor *saved_indexer_weights;
    ds4_gpu_tensor *saved_heads;
    ds4_gpu_tensor *saved_attn_low;
    ds4_gpu_tensor *saved_attn_out;
    ds4_gpu_tensor *saved_after_attn_hc;
} metal_graph_attn_pre_alias;

static bool metal_graph_bind_attn_pre_batch_row(
        ds4_gpu_graph *g,
        ds4_gpu_graph *batch_graph,
        const ds4_layer_weights *layer,
        uint32_t row,
        bool full_attention_rows,
        metal_graph_attn_pre_alias *alias) {
    if (!g || !batch_graph || !layer || !alias ||
        g->active_tier < 0 || g->active_tier != batch_graph->active_tier) {
        return false;
    }
    memset(alias, 0, sizeof(*alias));
    alias->tier = g->active_tier;
    const uint64_t mix_hc = 2ull * DS4_N_HC +
                            (uint64_t)DS4_N_HC * DS4_N_HC;
    if (!metal_graph_borrow_tensor_view(
                &alias->attn_norm,
                metal_graph_batch_attn_norm(batch_graph),
                (uint64_t)row * DS4_N_EMBD * sizeof(float),
                (uint64_t)DS4_N_EMBD * sizeof(float)) ||
        !metal_graph_borrow_tensor_view(
                &alias->hc_split,
                metal_graph_batch_hc_split(batch_graph),
                (uint64_t)row * mix_hc * sizeof(float),
                mix_hc * sizeof(float)) ||
        !metal_graph_borrow_tensor_view(
                &alias->hc_pre, &alias->hc_split, 0,
                (uint64_t)DS4_N_HC * sizeof(float)) ||
        !metal_graph_borrow_tensor_view(
                &alias->hc_post, &alias->hc_split,
                (uint64_t)DS4_N_HC * sizeof(float),
                (uint64_t)DS4_N_HC * sizeof(float)) ||
        !metal_graph_borrow_tensor_view(
                &alias->hc_comb, &alias->hc_split,
                2ull * DS4_N_HC * sizeof(float),
                (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float))) {
        return false;
    }

    if (full_attention_rows) {
        const uint64_t q_rank = layer->attn_q_a->dim[1];
        const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
        const uint64_t comp_width_max = 2ull * DS4_N_HEAD_DIM;
        const uint64_t indexer_q_dim =
            (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
        const uint64_t low_dim =
            (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
        const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
#define DS4_BIND_ATTN_BATCH_ROW(member, batch_tensor, stride)                 \
        metal_graph_borrow_tensor_view(                                      \
            &alias->member, (batch_tensor),                                  \
            (uint64_t)row * (stride) * sizeof(float),                        \
            (stride) * sizeof(float))
        if (!DS4_BIND_ATTN_BATCH_ROW(qr, metal_graph_batch_qr(batch_graph),
                                     q_rank) ||
            !DS4_BIND_ATTN_BATCH_ROW(qr_norm,
                                     metal_graph_batch_qr_norm(batch_graph),
                                     q_rank) ||
            !DS4_BIND_ATTN_BATCH_ROW(q, metal_graph_batch_q(batch_graph),
                                     q_dim) ||
            !DS4_BIND_ATTN_BATCH_ROW(kv_raw,
                                     metal_graph_batch_kv_raw(batch_graph),
                                     DS4_N_HEAD_DIM) ||
            !DS4_BIND_ATTN_BATCH_ROW(kv, metal_graph_batch_kv(batch_graph),
                                     DS4_N_HEAD_DIM) ||
            !DS4_BIND_ATTN_BATCH_ROW(comp_kv_cur,
                                     metal_graph_batch_comp_kv(batch_graph),
                                     comp_width_max) ||
            !DS4_BIND_ATTN_BATCH_ROW(comp_sc_cur,
                                     metal_graph_batch_comp_sc(batch_graph),
                                     comp_width_max) ||
            !DS4_BIND_ATTN_BATCH_ROW(indexer_q,
                                     metal_graph_batch_indexer_q(batch_graph),
                                     indexer_q_dim) ||
            !DS4_BIND_ATTN_BATCH_ROW(indexer_weights,
                                     metal_graph_batch_indexer_weights(batch_graph),
                                     DS4_N_INDEXER_HEAD) ||
            !DS4_BIND_ATTN_BATCH_ROW(heads,
                                     metal_graph_batch_heads(batch_graph),
                                     q_dim) ||
            !DS4_BIND_ATTN_BATCH_ROW(attn_low,
                                     metal_graph_batch_attn_low(batch_graph),
                                     low_dim) ||
            !DS4_BIND_ATTN_BATCH_ROW(attn_out,
                                     metal_graph_batch_attn_out(batch_graph),
                                     DS4_N_EMBD) ||
            !DS4_BIND_ATTN_BATCH_ROW(after_attn_hc,
                                     metal_graph_batch_after_attn_hc(batch_graph),
                                     hc_dim)) {
#undef DS4_BIND_ATTN_BATCH_ROW
            return false;
        }
#undef DS4_BIND_ATTN_BATCH_ROW
    }

    const int t = alias->tier;
    alias->saved_attn_norm = g->attn_norm_by_tier[t];
    alias->saved_hc_split = g->hc_split_by_tier[t];
    alias->saved_hc_pre = g->hc_pre_by_tier[t];
    alias->saved_hc_post = g->hc_post_by_tier[t];
    alias->saved_hc_comb = g->hc_comb_by_tier[t];
    g->attn_norm_by_tier[t] = &alias->attn_norm;
    g->hc_split_by_tier[t] = &alias->hc_split;
    g->hc_pre_by_tier[t] = &alias->hc_pre;
    g->hc_post_by_tier[t] = &alias->hc_post;
    g->hc_comb_by_tier[t] = &alias->hc_comb;
    if (full_attention_rows) {
        alias->saved_qr = g->qr_by_tier[t];
        alias->saved_qr_norm = g->qr_norm_by_tier[t];
        alias->saved_q = g->q_by_tier[t];
        alias->saved_kv_raw = g->kv_raw_by_tier[t];
        alias->saved_kv = g->kv_by_tier[t];
        alias->saved_comp_kv_cur = g->comp_kv_cur_by_tier[t];
        alias->saved_comp_sc_cur = g->comp_sc_cur_by_tier[t];
        alias->saved_indexer_q = g->indexer_q_by_tier[t];
        alias->saved_indexer_weights = g->indexer_weights_by_tier[t];
        alias->saved_heads = g->heads_by_tier[t];
        alias->saved_attn_low = g->attn_low_by_tier[t];
        alias->saved_attn_out = g->attn_out_by_tier[t];
        alias->saved_after_attn_hc = g->after_attn_hc_by_tier[t];
        g->qr_by_tier[t] = &alias->qr;
        g->qr_norm_by_tier[t] = &alias->qr_norm;
        g->q_by_tier[t] = &alias->q;
        g->kv_raw_by_tier[t] = &alias->kv_raw;
        g->kv_by_tier[t] = &alias->kv;
        g->comp_kv_cur_by_tier[t] = &alias->comp_kv_cur;
        g->comp_sc_cur_by_tier[t] = &alias->comp_sc_cur;
        g->indexer_q_by_tier[t] = &alias->indexer_q;
        g->indexer_weights_by_tier[t] = &alias->indexer_weights;
        g->heads_by_tier[t] = &alias->heads;
        g->attn_low_by_tier[t] = &alias->attn_low;
        g->attn_out_by_tier[t] = &alias->attn_out;
        g->after_attn_hc_by_tier[t] = &alias->after_attn_hc;
        alias->full_attention_rows = true;
    }
    alias->active = true;
    return true;
}

static void metal_graph_unbind_attn_pre_batch_row(
        ds4_gpu_graph *g,
        metal_graph_attn_pre_alias *alias) {
    if (!g || !alias || !alias->active) return;
    const int t = alias->tier;
    g->attn_norm_by_tier[t] = alias->saved_attn_norm;
    g->hc_split_by_tier[t] = alias->saved_hc_split;
    g->hc_pre_by_tier[t] = alias->saved_hc_pre;
    g->hc_post_by_tier[t] = alias->saved_hc_post;
    g->hc_comb_by_tier[t] = alias->saved_hc_comb;
    if (alias->full_attention_rows) {
        g->qr_by_tier[t] = alias->saved_qr;
        g->qr_norm_by_tier[t] = alias->saved_qr_norm;
        g->q_by_tier[t] = alias->saved_q;
        g->kv_raw_by_tier[t] = alias->saved_kv_raw;
        g->kv_by_tier[t] = alias->saved_kv;
        g->comp_kv_cur_by_tier[t] = alias->saved_comp_kv_cur;
        g->comp_sc_cur_by_tier[t] = alias->saved_comp_sc_cur;
        g->indexer_q_by_tier[t] = alias->saved_indexer_q;
        g->indexer_weights_by_tier[t] = alias->saved_indexer_weights;
        g->heads_by_tier[t] = alias->saved_heads;
        g->attn_low_by_tier[t] = alias->saved_attn_low;
        g->attn_out_by_tier[t] = alias->saved_attn_out;
        g->after_attn_hc_by_tier[t] = alias->saved_after_attn_hc;
    }
    alias->active = false;
}

static bool metal_graph_encode_attention_session_batch(
        ds4_decode_item *items,
        int count,
        const ds4_model *model,
        const ds4_layer_weights *layer,
        uint32_t il,
        uint32_t row_base) {
#if defined(__APPLE__)
    (void)items;
    (void)count;
    (void)model;
    (void)layer;
    (void)il;
    (void)row_base;
    return false;
#else
    if (!items || count < 2 || !model || !layer || il >= DS4_N_LAYER ||
        count > (int)DS4_GPU_ATTENTION_DECODE_BATCH_MAX) {
        return false;
    }
    ds4_gpu_graph *batch_graph = &items[0].session->graph;
    const int home = batch_graph->active_tier;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    ds4_gpu_tensor q_rows, head_rows;
    if (home < 0 ||
        (uint64_t)row_base + (uint32_t)count > batch_graph->prefill_cap ||
        !metal_graph_borrow_tensor_view(
            &q_rows, metal_graph_batch_q(batch_graph),
            (uint64_t)row_base * q_dim * sizeof(float),
            (uint64_t)count * q_dim * sizeof(float)) ||
        !metal_graph_borrow_tensor_view(
            &head_rows, metal_graph_batch_heads(batch_graph),
            (uint64_t)row_base * q_dim * sizeof(float),
            (uint64_t)count * q_dim * sizeof(float))) {
        return false;
    }

    ds4_gpu_attention_decode_row rows[DS4_GPU_ATTENTION_DECODE_BATCH_MAX];
    memset(rows, 0, sizeof(rows));
    for (int i = 0; i < count; i++) {
        ds4_session *s = items[i].session;
        ds4_gpu_graph *g = &s->graph;
        const uint32_t pos = (uint32_t)s->checkpoint.len;
        const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t n_comp = g->layer_n_comp[il];
        const bool indexed =
            ratio == 4u &&
            n_comp > metal_graph_decode_indexer_sparse_threshold(g) &&
            g->layer_n_index_comp[il] > DS4_N_INDEXER_TOP_K;
        const uint32_t n_selected = indexed
            ? (DS4_N_INDEXER_TOP_K < g->layer_n_index_comp[il]
                ? DS4_N_INDEXER_TOP_K : g->layer_n_index_comp[il])
            : 0u;
        ds4_gpu_tensor *selected = indexed ? metal_graph_comp_selected(g) : NULL;
        if (g->active_tier != home || !g->layer_raw_cache[il] ||
            (n_comp != 0u && !g->layer_attn_comp_cache[il]) ||
            (indexed && (!selected || n_selected == 0u))) {
            return false;
        }
        rows[i].raw_kv = (uint64_t)(uintptr_t)g->layer_raw_cache[il]->ptr;
        rows[i].comp_kv = (uint64_t)(uintptr_t)(
            n_comp ? g->layer_attn_comp_cache[il]->ptr
                   : g->layer_raw_cache[il]->ptr);
        rows[i].topk = selected
            ? (uint64_t)(uintptr_t)selected->ptr : 0u;
        rows[i].pos = pos;
        rows[i].n_raw = n_raw;
        rows[i].raw_cap = g->raw_cap;
        rows[i].raw_start = metal_graph_raw_start_for_span(g, pos, n_raw);
        rows[i].n_comp = n_comp;
        rows[i].top_k = n_selected;
        rows[i].window = indexed ? g->raw_window : 0u;
        rows[i].ratio = indexed ? ratio : 0u;
        rows[i].indexed = indexed ? 1u : 0u;
    }

    const bool compressed = ds4_layer_compress_ratio(il) != 0u;
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor =
        compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    return ds4_gpu_attention_decode_rows_rope_tensor(
        &head_rows,
        model->map,
        model->size,
        layer->attn_sinks->abs_offset,
        &q_rows,
        rows,
        (uint32_t)count,
        DS4_N_HEAD,
        DS4_N_HEAD_DIM,
        DS4_N_ROT,
        compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0u,
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        DS4_ROPE_YARN_BETA_FAST,
        DS4_ROPE_YARN_BETA_SLOW) != 0;
#endif
}

/* Keep grouped decode rows contiguous through the two TP output halves and
 * HC expansion. Each projection uses the batch-one warp reduction per row. */
static bool metal_graph_encode_attn_post_session_batch(
        ds4_decode_item *items,
        int count,
        const ds4_model *model,
        const ds4_layer_weights *layer,
        uint32_t il,
        uint32_t row_base) {
#if defined(__APPLE__)
    (void)items;
    (void)count;
    (void)model;
    (void)layer;
    (void)il;
    (void)row_base;
    return false;
#else
    if (!items || count < 2 || !model || !layer || il >= DS4_N_LAYER ||
        (DS4_N_OUT_GROUP & 1u) != 0u) {
        return false;
    }
    ds4_gpu_graph *g = &items[0].session->graph;
    const int home = g->placement[il + 1u];
    const int partner = metal_graph_cuda_tp_partner_tier(home);
    const uint32_t rows = (uint32_t)count;
    const uint32_t tp_groups = DS4_N_OUT_GROUP / 2u;
    const uint64_t group_dim =
        (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t tp_low_dim = (uint64_t)tp_groups * DS4_N_LORA_O;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc =
        2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    if (home < 0 || partner < 0 ||
        (uint64_t)row_base + rows > g->prefill_cap) {
        return false;
    }
    for (int i = 0; i < count; i++) {
        if (!metal_graph_set_active_tier_decode(
                    &items[i].session->graph, home)) {
            return false;
        }
    }

    ds4_gpu_tensor heads, home_low, home_out;
    ds4_gpu_tensor peer_low, peer_out, peer_return;
    ds4_gpu_tensor residual_hc, split, after_attn_hc;
    const uint64_t out_bytes = (uint64_t)rows * DS4_N_EMBD * sizeof(float);
    bool ok =
        metal_graph_borrow_tensor_view(
                &heads, metal_graph_batch_heads(g),
                (uint64_t)row_base * q_dim * sizeof(float),
                (uint64_t)rows * q_dim * sizeof(float)) &&
        metal_graph_borrow_tensor_view(
                &home_low, metal_graph_batch_attn_low(g),
                (uint64_t)row_base * low_dim * sizeof(float),
                (uint64_t)rows * tp_low_dim * sizeof(float)) &&
        metal_graph_borrow_tensor_view(
                &home_out, metal_graph_batch_attn_out(g),
                (uint64_t)row_base * DS4_N_EMBD * sizeof(float), out_bytes) &&
        metal_graph_borrow_tensor_view(
                &peer_low, g->batch_attn_low_by_tier[partner],
                (uint64_t)row_base * low_dim * sizeof(float),
                (uint64_t)rows * tp_low_dim * sizeof(float)) &&
        metal_graph_borrow_tensor_view(
                &peer_out, g->batch_attn_out_by_tier[partner],
                (uint64_t)row_base * DS4_N_EMBD * sizeof(float), out_bytes) &&
        metal_graph_borrow_tensor_view(
                &peer_return, metal_graph_batch_shared_out(g),
                (uint64_t)row_base * DS4_N_EMBD * sizeof(float), out_bytes) &&
        metal_graph_borrow_tensor_view(
                &residual_hc, metal_graph_batch_cur_hc(g),
                (uint64_t)row_base * hc_dim * sizeof(float),
                (uint64_t)rows * hc_dim * sizeof(float)) &&
        metal_graph_borrow_tensor_view(
                &split, metal_graph_batch_hc_split(g),
                (uint64_t)row_base * mix_hc * sizeof(float),
                (uint64_t)rows * mix_hc * sizeof(float)) &&
        metal_graph_borrow_tensor_view(
                &after_attn_hc, metal_graph_batch_after_attn_hc(g),
                (uint64_t)row_base * hc_dim * sizeof(float),
                (uint64_t)rows * hc_dim * sizeof(float));

    if (ok) ok = ds4_gpu_tensor_wait_xdev_default(&heads, partner) != 0;
    if (ok) ok = ds4_gpu_set_current_device(partner) == 0;
    if (ok) {
        ok = ds4_gpu_attention_output_low_q8_rows_exact_tensor(
                &peer_low, model->map, model->size,
                layer->attn_output_a->abs_offset,
                group_dim, DS4_N_LORA_O,
                DS4_N_OUT_GROUP, tp_groups, tp_groups,
                &heads, rows) != 0;
    }
    if (ok) {
        ok = ds4_gpu_matmul_q8_0_kslice_rows_tensor(
                &peer_out, model->map, model->size,
                layer->attn_output_b->abs_offset,
                low_dim, DS4_N_EMBD, tp_low_dim, tp_low_dim,
                &peer_low, rows) != 0;
    }
    if (ds4_gpu_set_current_device(home) != 0) ok = false;
    if (ok) {
        ok = ds4_gpu_attention_output_low_q8_rows_exact_tensor(
                &home_low, model->map, model->size,
                layer->attn_output_a->abs_offset,
                group_dim, DS4_N_LORA_O,
                DS4_N_OUT_GROUP, 0u, tp_groups,
                &heads, rows) != 0;
    }
    if (ok) {
        ok = ds4_gpu_matmul_q8_0_kslice_rows_tensor(
                &home_out, model->map, model->size,
                layer->attn_output_b->abs_offset,
                low_dim, DS4_N_EMBD, 0u, tp_low_dim,
                &home_low, rows) != 0;
    }
    if (ok) {
        ok = ds4_gpu_tensor_copy_xdev_default(
                &peer_return, &peer_out, out_bytes) != 0;
    }
    if (ok) {
        ok = ds4_gpu_hc_expand_add_split_tensor(
                &after_attn_hc, &home_out, &peer_return,
                &residual_hc, &split, DS4_N_EMBD, DS4_N_HC) != 0;
    }
    return ok;
#endif
}

static bool metal_graph_session_batch_ffn_pre_supported(
        ds4_decode_item *items,
        int count,
        const ds4_weights *weights) {
    if (!items || count < 3 || !weights) return false;

    ds4_gpu_graph *first = &items[0].session->graph;
    if ((uint32_t)count > first->prefill_cap ||
        !metal_graph_prefill_tokens(first) ||
        ds4_gpu_tensor_bytes(metal_graph_prefill_tokens(first)) <
            (uint64_t)count * sizeof(int32_t)) {
        return false;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!layer->hc_ffn_fn || !layer->ffn_gate_inp ||
            layer->hc_ffn_fn->type != DS4_TENSOR_F16 ||
            layer->ffn_gate_inp->type != DS4_TENSOR_F16 ||
            layer->ffn_gate_inp->dim[0] != 4096u ||
            layer->ffn_gate_inp->dim[1] != 256u) {
            return false;
        }
        const int home = first->placement[il + 1u];
        if (!first->batch_after_attn_hc_by_tier[home] ||
            !first->batch_flat_hc_by_tier[home] ||
            !first->batch_hc_mix_by_tier[home] ||
            !first->batch_hc_split_by_tier[home] ||
            !first->batch_ffn_cur_by_tier[home] ||
            !first->batch_ffn_norm_by_tier[home] ||
            !first->batch_router_logits_by_tier[home] ||
            !first->batch_router_probs_by_tier[home] ||
            !first->batch_router_selected_by_tier[home] ||
            !first->batch_router_weights_by_tier[home]) {
            return false;
        }
        for (int i = 1; i < count; i++) {
            ds4_gpu_graph *g = &items[i].session->graph;
            if (!g->placement || g->placement[il + 1u] != home) return false;
        }
    }
    return true;
}

/* Gather the cache-independent FFN input rows after attention, then execute
 * HC mixing, normalization, router projection, and selection as row batches.
 * Routed and shared experts consume these batch tensors directly. */
static bool metal_graph_encode_ffn_pre_session_batch(
        ds4_decode_item *items,
        int count,
        const ds4_model *model,
        const ds4_layer_weights *layer,
        uint32_t row_base,
        bool gather_after_attn_rows) {
    if (!items || count < 2 || !model || !layer) return false;

    ds4_gpu_graph *g = &items[0].session->graph;
    const int home_tier = g->active_tier;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC +
                            (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t hc_bytes = (uint64_t)count * hc_dim * sizeof(float);
    const uint64_t mix_bytes = (uint64_t)count * mix_hc * sizeof(float);
    const uint64_t norm_bytes =
        (uint64_t)count * DS4_N_EMBD * sizeof(float);
    const uint64_t router_bytes =
        (uint64_t)count * DS4_N_EXPERT * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)count * DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t weights_bytes =
        (uint64_t)count * DS4_N_EXPERT_USED * sizeof(float);

    ds4_gpu_tensor after_attn_hc, flat_hc, hc_mix, hc_split;
    ds4_gpu_tensor ffn_cur, ffn_norm, router_logits, router_probs;
    ds4_gpu_tensor router_selected, router_weights, router_tokens;
    bool ok =
        home_tier >= 0 &&
        (uint64_t)row_base + (uint32_t)count <= g->prefill_cap &&
        metal_graph_borrow_tensor_view(&after_attn_hc,
                metal_graph_batch_after_attn_hc(g),
                (uint64_t)row_base * hc_dim * sizeof(float), hc_bytes) &&
        metal_graph_borrow_tensor_view(&flat_hc,
                metal_graph_batch_flat_hc(g),
                (uint64_t)row_base * hc_dim * sizeof(float), hc_bytes) &&
        metal_graph_borrow_tensor_view(&hc_mix,
                metal_graph_batch_hc_mix(g),
                (uint64_t)row_base * mix_hc * sizeof(float), mix_bytes) &&
        metal_graph_borrow_tensor_view(&hc_split,
                metal_graph_batch_hc_split(g),
                (uint64_t)row_base * mix_hc * sizeof(float), mix_bytes) &&
        metal_graph_borrow_tensor_view(&ffn_cur,
                metal_graph_batch_ffn_cur(g),
                (uint64_t)row_base * DS4_N_EMBD * sizeof(float), norm_bytes) &&
        metal_graph_borrow_tensor_view(&ffn_norm,
                metal_graph_batch_ffn_norm(g),
                (uint64_t)row_base * DS4_N_EMBD * sizeof(float), norm_bytes) &&
        metal_graph_borrow_tensor_view(&router_logits,
                metal_graph_batch_router_logits(g),
                (uint64_t)row_base * DS4_N_EXPERT * sizeof(float), router_bytes) &&
        metal_graph_borrow_tensor_view(&router_probs,
                metal_graph_batch_router_probs(g),
                (uint64_t)row_base * DS4_N_EXPERT * sizeof(float), router_bytes) &&
        metal_graph_borrow_tensor_view(&router_selected,
                metal_graph_batch_router_selected(g),
                (uint64_t)row_base * DS4_N_EXPERT_USED * sizeof(int32_t),
                selected_bytes) &&
        metal_graph_borrow_tensor_view(&router_weights,
                metal_graph_batch_router_weights(g),
                (uint64_t)row_base * DS4_N_EXPERT_USED * sizeof(float),
                weights_bytes) &&
        metal_graph_borrow_tensor_view(&router_tokens,
                metal_graph_prefill_tokens(g),
                (uint64_t)row_base * sizeof(int32_t),
                (uint64_t)count * sizeof(int32_t));

    for (int i = 0; gather_after_attn_rows && ok && i < count; i++) {
        ds4_gpu_graph *srcg = &items[i].session->graph;
        ds4_gpu_tensor dst_row;
        ok = srcg->active_tier == home_tier &&
             metal_graph_borrow_tensor_view(
                     &dst_row, &after_attn_hc,
                     (uint64_t)i * hc_dim * sizeof(float),
                     hc_dim * sizeof(float));
        if (ok) {
            ok = ds4_gpu_tensor_copy_xdev_default(
                    &dst_row, metal_graph_after_attn_hc(srcg),
                    dst_row.bytes) != 0;
        }
    }
    if (ok) {
        ok = ds4_gpu_rms_norm_plain_rows_tensor(
                &flat_hc, &after_attn_hc, (uint32_t)hc_dim,
                (uint32_t)count, DS4_RMS_EPS) != 0;
    }
    for (int i = 0; ok && i < count; i++) {
        ds4_gpu_tensor out_row, in_row;
        ok = metal_graph_borrow_tensor_view(
                     &out_row, &hc_mix,
                     (uint64_t)i * mix_hc * sizeof(float),
                     mix_hc * sizeof(float)) &&
             metal_graph_borrow_tensor_view(
                     &in_row, &flat_hc,
                     (uint64_t)i * hc_dim * sizeof(float),
                     hc_dim * sizeof(float));
        if (ok) {
            ok = metal_graph_matmul_plain_tensor(
                    &out_row, model, layer->hc_ffn_fn,
                    hc_dim, mix_hc, &in_row, 1);
        }
    }
    if (ok) {
        ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(
                &ffn_cur,
                &ffn_norm,
                &hc_split,
                &hc_mix,
                &after_attn_hc,
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
    }
    if (ok) {
        ok = ds4_gpu_matmul_f16_router_rows_exact_tensor(
                &router_logits, model->map, model->size,
                layer->ffn_gate_inp->abs_offset, &ffn_norm,
                (uint32_t)count) != 0;
    }
    if (ok) {
        ok = ds4_gpu_router_select_batch_tensor(
                &router_selected,
                &router_weights,
                &router_probs,
                model->map,
                model->size,
                layer->ffn_exp_probs_b ?
                    layer->ffn_exp_probs_b->abs_offset : 0,
                layer->ffn_gate_tid2eid ?
                    layer->ffn_gate_tid2eid->abs_offset : 0,
                layer->ffn_gate_tid2eid ?
                    (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
                0,
                0,
                layer->ffn_exp_probs_b != NULL,
                layer->ffn_gate_tid2eid != NULL,
                &router_logits,
                &router_tokens,
                DS4_N_EXPERT,
                DS4_N_EXPERT_USED,
                DS4_EXPERT_WEIGHT_SCALE,
                (uint32_t)count) != 0;
    }
    return ok;
}

#endif /* !DS4_NO_GPU */
