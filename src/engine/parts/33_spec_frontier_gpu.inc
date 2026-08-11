static void spec_frontier_free(ds4_spec_frontier *f) {
    if (!f) return;
    memset(f, 0, sizeof(*f));
}

static bool spec_frontier_snapshot(ds4_spec_frontier *f, ds4_session *s) {
    memset(f, 0, sizeof(*f));
    ds4_gpu_graph *g = &s->graph;
    if (!metal_graph_dspark_cache_current_window_valid(g)) return false;
    f->mtp_n_raw = g->mtp_n_raw;
    f->dspark_cache_start = g->dspark_cache_start;
    f->dspark_cache_token_start = g->dspark_cache_token_start;
    f->dspark_cache_len = g->dspark_cache_len;

    bool ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        f->n_comp[il] = g->layer_n_comp[il];
        f->n_index_comp[il] = g->layer_n_index_comp[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t ab = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        ok = ds4_gpu_tensor_copy(g->spec_attn_state_kv[il], 0,
                                   g->layer_attn_state_kv[il], 0, ab) != 0 &&
             ds4_gpu_tensor_copy(g->spec_attn_state_score[il], 0,
                                   g->layer_attn_state_score[il], 0, ab) != 0;
        if (ratio == 4) {
            const uint64_t ib = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            ok = ok &&
                 ds4_gpu_tensor_copy(g->spec_index_state_kv[il], 0,
                                       g->layer_index_state_kv[il], 0, ib) != 0 &&
                 ds4_gpu_tensor_copy(g->spec_index_state_score[il], 0,
                                       g->layer_index_state_score[il], 0, ib) != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok) return true;

    spec_frontier_free(f);
    return false;
}

static bool spec_frontier_restore(ds4_spec_frontier *f, ds4_session *s) {
    ds4_gpu_graph *g = &s->graph;
    if (!metal_graph_dspark_cache_window_valid(g,
                                               f->dspark_cache_token_start,
                                               f->dspark_cache_start,
                                               f->dspark_cache_len)) {
        return false;
    }
    bool ok = ds4_gpu_begin_commands() != 0;
    g->mtp_n_raw = f->mtp_n_raw;
    if (ok) {
        ok = metal_graph_dspark_cache_set_window(g,
                                                 f->dspark_cache_token_start,
                                                 f->dspark_cache_len);
    }
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        g->layer_n_comp[il] = f->n_comp[il];
        g->layer_n_index_comp[il] = f->n_index_comp[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t ab = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        ok = ds4_gpu_tensor_copy(g->layer_attn_state_kv[il], 0,
                                   g->spec_attn_state_kv[il], 0, ab) != 0 &&
             ds4_gpu_tensor_copy(g->layer_attn_state_score[il], 0,
                                   g->spec_attn_state_score[il], 0, ab) != 0;
        if (ok && ratio == 4) {
            const uint64_t ib = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            ok = ds4_gpu_tensor_copy(g->layer_index_state_kv[il], 0,
                                       g->spec_index_state_kv[il], 0, ib) != 0 &&
                 ds4_gpu_tensor_copy(g->layer_index_state_score[il], 0,
                                       g->spec_index_state_score[il], 0, ib) != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    return ok;
}
/* Commit an intermediate state captured by a tiny speculative verifier.
 *
 * Append-only cache rows beyond the accepted prefix can remain as invisible
 * garbage. Only compressor frontiers and row counters need rewinding. */
static bool spec_frontier_commit_prefix(ds4_session *s, uint32_t prefix_len) {
    if (!s || prefix_len == 0 || prefix_len > DS4_SPEC_PREFIX_SLOTS) {
        return false;
    }
    const uint32_t slot = prefix_len - 1u;
    ds4_gpu_graph *g = &s->graph;
    bool ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;

        g->layer_n_comp[il] = g->spec_prefix_n_comp[slot][il];
        const uint64_t ab = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
        const uint64_t aoff = (uint64_t)slot * ab;
        ok = ds4_gpu_tensor_copy(g->layer_attn_state_kv[il], 0,
                                   g->spec_prefix1_attn_state_kv[il], aoff, ab) != 0 &&
             ds4_gpu_tensor_copy(g->layer_attn_state_score[il], 0,
                                   g->spec_prefix1_attn_state_score[il], aoff, ab) != 0;
        if (ok && ratio == 4) {
            g->layer_n_index_comp[il] =
                g->spec_prefix_n_index_comp[slot][il];
            const uint64_t ib = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
            const uint64_t ioff = (uint64_t)slot * ib;
            ok = ds4_gpu_tensor_copy(g->layer_index_state_kv[il], 0,
                                       g->spec_prefix1_index_state_kv[il], ioff, ib) != 0 &&
                 ds4_gpu_tensor_copy(g->layer_index_state_score[il], 0,
                                       g->spec_prefix1_index_state_score[il], ioff, ib) != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    return ok;
}

static bool spec_frontier_commit_prefix1(ds4_session *s) {
    return spec_frontier_commit_prefix(s, 1);
}

static void session_greedy_splitkv_reset(ds4_session *s) {
    if (!s) return;
    s->greedy_splitkv_segment.len = 0;
    s->greedy_splitkv_anchor_valid = false;
    s->greedy_splitkv_anchor_len = 0;
    spec_frontier_free(&s->greedy_splitkv_anchor);
}

