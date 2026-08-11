#ifndef DS4_NO_GPU
/* Long prompts are prefetched in fixed-size chunks.  Chunks bound transient
 * attention buffers while preserving the same final KV/cache state. */
static bool metal_graph_prefill_chunked(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled) {
    if (n_tokens <= 0) return false;
    return metal_graph_prefill_chunked_range(g,
                                             model,
                                             weights,
                                             prompt,
                                             0,
                                             (uint32_t)n_tokens,
                                             logits,
                                             show_progress,
                                             progress,
                                             progress_ud,
                                             display_progress,
                                             display_progress_ud,
                                             NULL,
                                             cancel,
                                             cancel_ud,
                                             cancelled);
}

typedef struct ds4_verify_suffix_timing {
    double upload_ms;
    double layer_ms;
    double head_ms;
    double read_ms;
    bool fused_head;
} ds4_verify_suffix_timing;

static bool metal_graph_dspark_verify_selected_profile_enabled(void) {
    return getenv("DS4_DSPARK_VERIFY_SELECTED_PROFILE") != NULL &&
           getenv("DS4_DSPARK_DISABLE_VERIFY_SELECTED_PROFILE") == NULL;
}

/* Layer-major speculative target verifier for tiny MTP suffixes.
 *
 * This is the first production-shaped verifier attempt: unlike repeated decode
 * it runs the target model layer-by-layer for the whole speculative suffix, and
 * unlike the diagnostic path it does not read back full logits for every row.
 * The verifier returns the row top-1 ids needed for acceptance.  The caller
 * then reads exactly one logits row: the row that becomes the new continuation
 * state.  It still reuses the existing batch layer kernels, so it is not yet
 * the final hand-written N=2/N=4 decode microbatch, but it exercises the right
 * verifier contract and removes the obvious diagnostic overheads first. */
static bool metal_graph_verify_suffix_tops_impl(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        bool                   capture_prefix1,
        bool                   capture_dspark_hidden,
        int                   *row_tops,
        float                 *row_logits,
        ds4_verify_suffix_timing *timing) {
    if (timing) memset(timing, 0, sizeof(*timing));
    if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;
    if (start > (uint32_t)prompt->len || n_tokens > (uint32_t)prompt->len - start) return false;
    const uint32_t top_rows = n_tokens > 1 ? n_tokens - 1 : 0;
    if (top_rows && !row_tops) return false;

    const double upload_t0 = timing ? now_sec() : 0.0;
    bool ok = metal_graph_upload_prompt_tokens(metal_graph_prefill_tokens(g), prompt, start, n_tokens);
    if (ok) ok = metal_graph_upload_prompt_embeddings_hc(metal_graph_batch_cur_hc(g),
                                                         metal_graph_prefill_tokens(g),
                                                         model,
                                                         weights,
                                                         prompt,
                                                         start,
                                                         n_tokens);
    if (!ok) return false;

    const bool saved_capture = g->spec_capture_prefixes;
    g->spec_capture_prefixes =
        capture_prefix1 && n_tokens > 1u &&
        n_tokens <= DS4_SPEC_PREFIX_SLOTS + 1u;
    const char *split_head_env = getenv("DS4_DSPARK_VERIFY_SPLIT_HEAD");
    const bool fuse_head =
        !split_head_env || !split_head_env[0] ||
        strcmp(split_head_env, "0") == 0;
    if (timing) timing->fused_head = fuse_head;
    if (timing) timing->upload_ms += (now_sec() - upload_t0) * 1000.0;

    const bool selected_profile =
        metal_graph_dspark_verify_selected_profile_enabled();
    if (selected_profile) {
        metal_graph_stream_prefill_selected_profile_reset(g);
    }

    /* Under TP, verify every speculative block against the two resident
     * expert halves. Both ranks encode identically, so one batch gate per
     * layer reconstructs the routed result while preserving their KV state. */
    g->tp_batch_rows = (g->tp_world == 2 &&
                        g->tp_batch_out != NULL && g->tp_batch_in != NULL &&
                        n_tokens <= (uint32_t)DS4_TP_BATCH_MAX_ROWS)
                       ? n_tokens : 0;
    const double layer_t0 = timing ? now_sec() : 0.0;
    ok = ds4_gpu_begin_commands() != 0;
    const bool dspark_capture_active =
        ok &&
        capture_dspark_hidden &&
        metal_graph_dspark_capture_verified_suffix_begin(g,
                                                         start,
                                                         n_tokens,
                                                         true);
    static int verify_profile_left = -1;
    if (verify_profile_left < 0) {
        verify_profile_left =
            getenv("DS4_DSPARK_VERIFY_PROFILE") != NULL ? 1 : 0;
    }
    const bool verify_profile = verify_profile_left > 0 && ok;
    if (verify_profile) verify_profile_left--;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        if (verify_profile) {
            ok = ds4_gpu_end_commands() != 0;
            if (ok) (void)ds4_gpu_synchronize();
            if (ok) ok = ds4_gpu_begin_commands() != 0;
            if (!ok) break;
        }
        ok = metal_graph_encode_layer_batch(g,
                                            model,
                                            &weights->layer[il],
                                            il,
                                            start,
                                            n_tokens);
        if (ok && dspark_capture_active) {
            ok = metal_graph_dspark_capture_verified_suffix_layer(g,
                                                                  il,
                                                                  start,
                                                                  n_tokens);
        }
        if (ok && selected_profile) {
            ok = ds4_gpu_end_commands() != 0 &&
                 metal_graph_selected_profile_layer_impl(
                         g,
                         &weights->layer[il],
                         il,
                         n_tokens,
                         "DSpark verifier selected profile") &&
                 ds4_gpu_begin_commands() != 0;
        }
#if defined(__APPLE__)
        /* Tiny DFlash batches are expensive enough to keep Metal busy while
         * the host encodes the next layers. Submit short prefixes to overlap
         * that work without changing the verifier's kernels or arithmetic. */
        if (ok && capture_dspark_hidden && !verify_profile &&
            !selected_profile && g->tp_world != 2 &&
            ((il + 1u) % 4u) == 0u) {
            ok = ds4_gpu_flush_commands() != 0;
        }
#endif
    }
    g->tp_batch_rows = 0;
    if (ok && fuse_head) {
        ok = metal_graph_encode_output_head_batch(g,
                                                  model,
                                                  weights,
                                                  n_tokens,
                                                  weights->output->dim[1]);
    }
    if (ok && fuse_head) {
        if (top_rows == 1) {
            ok = ds4_gpu_argmax_tensor(metal_graph_comp_selected(g),
                                       g->spec_logits,
                                       DS4_N_VOCAB) != 0;
        } else if (top_rows) {
            ok = ds4_gpu_indexer_topk_tensor(metal_graph_comp_selected(g),
                                             g->spec_logits,
                                             DS4_N_VOCAB,
                                             top_rows,
                                             1) != 0;
        }
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    g->spec_capture_prefixes = saved_capture;
    if (!ok && dspark_capture_active) {
        metal_graph_dspark_capture_invalidate(g);
    }
    if (timing) timing->layer_ms += (now_sec() - layer_t0) * 1000.0;
    if (!ok) return false;
    if (selected_profile) {
        metal_graph_selected_profile_summary_impl(
                g,
                "DSpark verifier selected profile");
    }

    if (!fuse_head) {
        const double head_t0 = timing ? now_sec() : 0.0;
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_output_head_batch(g,
                                                          model,
                                                          weights,
                                                          n_tokens,
                                                          weights->output->dim[1]);
        if (ok) {
            if (top_rows == 1) {
                /* Common K=2 verify case: top_k=1 over n_vocab → use the dedicated
                 * argmax kernel (single-block tree-reduce) instead of the legacy
                 * indexer_topk_kernel's single-thread O(n_vocab * top_k) fall-through. */
                ok = ds4_gpu_argmax_tensor(metal_graph_comp_selected(g),
                                           g->spec_logits,
                                           DS4_N_VOCAB) != 0;
            } else if (top_rows) {
                /* top-1 of each of the top_rows rows: n_tokens=top_rows, top_k=1.
                 * The order is transposed vs the indexer-score callers; a swap
                 * silently scores row 0's runner-ups instead of each row. */
                ok = ds4_gpu_indexer_topk_tensor(metal_graph_comp_selected(g),
                                                   g->spec_logits,
                                                   DS4_N_VOCAB,
                                                   top_rows,
                                                   1) != 0;
            }
        }
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        if (timing) timing->head_ms += (now_sec() - head_t0) * 1000.0;
    }
    const double read_t0 = timing ? now_sec() : 0.0;
    if (ok && top_rows) {
        ok = ds4_gpu_tensor_read(metal_graph_comp_selected(g),
                                   0,
                                   row_tops,
                                   (uint64_t)top_rows * sizeof(row_tops[0])) != 0;
        if (ok && getenv("DS4_DSPARK_VERIFY_TOPS_CHECK") != NULL) {
            float *chk = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
            for (uint32_t r = 0; r < top_rows; r++) {
                if (ds4_gpu_tensor_read(g->spec_logits,
                                        (uint64_t)r * DS4_N_VOCAB *
                                            sizeof(float),
                                        chk,
                                        (uint64_t)DS4_N_VOCAB *
                                            sizeof(float)) == 0) break;
                uint32_t am = 0;
                for (uint32_t i = 1; i < DS4_N_VOCAB; i++) {
                    if (chk[i] > chk[am]) am = i;
                }
                fprintf(stderr,
                        "ds4: verify tops-check row=%u gpu_top=%d cpu_argmax=%u "
                        "cpu_max=%.3f\n",
                        r, row_tops[r], am, chk[am]);
            }
            free(chk);
        }
    }
    if (ok && row_logits) {
        ok = ds4_gpu_tensor_read(g->spec_logits,
                                   0,
                                   row_logits,
                                   (uint64_t)n_tokens * DS4_N_VOCAB * sizeof(row_logits[0])) != 0;
    }
    if (timing) timing->read_ms += (now_sec() - read_t0) * 1000.0;
    return ok;
}

/* The verify block keeps the GPU genuinely busy, so the TP DVFS
 * keep-alive is a pure parasite for its duration — pause it. */
static bool metal_graph_verify_suffix_tops(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        bool                   capture_prefix1,
        bool                   capture_dspark_hidden,
        int                   *row_tops,
        float                 *row_logits,
        ds4_verify_suffix_timing *timing) {
    ds4_gpu_tp_keepalive_pause(1);
    const bool ok = metal_graph_verify_suffix_tops_impl(g, model, weights,
                                                        prompt, start,
                                                        n_tokens,
                                                        capture_prefix1,
                                                        capture_dspark_hidden,
                                                        row_tops, row_logits,
                                                        timing);
    ds4_gpu_tp_keepalive_pause(0);
    return ok;
}

static bool metal_graph_read_spec_logits_row(ds4_gpu_graph *g, uint32_t row, float *logits) {
    if (!g || !g->spec_logits || !logits || row >= g->prefill_cap) return false;
    const uint64_t row_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    return ds4_gpu_tensor_read(g->spec_logits,
                                 (uint64_t)row * row_bytes,
                                 logits,
                                 row_bytes) != 0;
}

/* Exact N=2 target verifier for MTP.
 *
 * The generic batch prefill path is fast, but it is not a safe substitute for
 * autoregressive decode: small row-wise differences in HC/MoE/output kernels
 * are enough to flip future greedy tokens.  This verifier keeps the exact
 * decode kernels and cache update order, but encodes the two proposed tokens
 * layer-by-layer in one command stream.  It returns the exact target top after
 * token0, and exact logits after token1. */
static bool metal_graph_verify_decode2_exact(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token0,
        int                    token1,
        uint32_t               start,
        int                   *top0,
        int                   *top1,
        float                 *logits0,
        float                 *logits1) {
    if (!g || !top0 || (!top1 && !logits1) || g->raw_cap == 0) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t hc_bytes = hc_dim * sizeof(float);
    ds4_gpu_tensor *cur0_by_tier[DS4_MAX_GPUS] = {0};
    ds4_gpu_tensor *cur1_by_tier[DS4_MAX_GPUS] = {0};
    ds4_gpu_tensor *next0_by_tier[DS4_MAX_GPUS] = {0};
    ds4_gpu_tensor *next1_by_tier[DS4_MAX_GPUS] = {0};
    ds4_gpu_tensor *saved_cur_by_tier[DS4_MAX_GPUS] = {0};
    ds4_gpu_tensor *saved_after_by_tier[DS4_MAX_GPUS] = {0};
    const int saved_active_tier = g->active_tier;
    const bool saved_capture = g->spec_capture_prefixes;

    bool ok = true;
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        saved_cur_by_tier[t] = g->cur_hc_by_tier[t];
        saved_after_by_tier[t] = g->after_ffn_hc_by_tier[t];
        if (!g->batch_cur_hc_by_tier[t] && !g->batch_next_hc_by_tier[t]) continue;
        if (!g->batch_cur_hc_by_tier[t] || !g->batch_next_hc_by_tier[t]) {
            ok = false;
            break;
        }
        cur0_by_tier[t] = metal_graph_tensor_row_view(g->batch_cur_hc_by_tier[t], 0, hc_dim);
        cur1_by_tier[t] = metal_graph_tensor_row_view(g->batch_cur_hc_by_tier[t], 1, hc_dim);
        next0_by_tier[t] = metal_graph_tensor_row_view(g->batch_next_hc_by_tier[t], 0, hc_dim);
        next1_by_tier[t] = metal_graph_tensor_row_view(g->batch_next_hc_by_tier[t], 1, hc_dim);
        if (!cur0_by_tier[t] || !cur1_by_tier[t] ||
            !next0_by_tier[t] || !next1_by_tier[t]) {
            ok = false;
            break;
        }
    }

    int cur_tier = g->emb_tier;
    if (cur_tier < 0 || cur_tier >= DS4_MAX_GPUS ||
        !cur0_by_tier[cur_tier] || !cur1_by_tier[cur_tier]) {
        ok = false;
    }
    if (ok) ok = metal_graph_set_active_tier_no_copy(g, cur_tier);
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(cur0_by_tier[cur_tier],
                                               model->map,
                                               model->size,
                                               weights->token_embd->abs_offset,
                                               (uint32_t)weights->token_embd->dim[1],
                                               (uint32_t)token0,
                                               DS4_N_EMBD,
                                               DS4_N_HC) != 0;
    if (ok) ok = ds4_gpu_embed_token_hc_tensor(cur1_by_tier[cur_tier],
                                               model->map,
                                               model->size,
                                               weights->token_embd->abs_offset,
                                               (uint32_t)weights->token_embd->dim[1],
                                               (uint32_t)token1,
                                               DS4_N_EMBD,
                                               DS4_N_HC) != 0;

    g->spec_capture_prefixes = true;
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const uint32_t pos0 = start;
        const uint32_t pos1 = start + 1u;
        const int this_tier = g->placement ? g->placement[il + 1] : cur_tier;
        if (this_tier < 0 || this_tier >= DS4_MAX_GPUS ||
            !cur0_by_tier[this_tier] || !cur1_by_tier[this_tier]) {
            ok = false;
            break;
        }
        if (this_tier != cur_tier) {
            ok = ds4_gpu_tensor_copy_xdev(cur0_by_tier[this_tier],
                                          cur0_by_tier[cur_tier],
                                          hc_bytes) != 0 &&
                 ds4_gpu_tensor_copy_xdev(cur1_by_tier[this_tier],
                                          cur1_by_tier[cur_tier],
                                          hc_bytes) != 0;
            if (!ok) break;
            cur_tier = this_tier;
        }
        ok = metal_graph_set_active_tier_no_copy(g, this_tier);
        if (!ok) break;

        g->cur_hc_by_tier[this_tier] = cur0_by_tier[this_tier];
        g->after_ffn_hc_by_tier[this_tier] = next0_by_tier[this_tier];
        ok = metal_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos0,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             pos0 % g->raw_cap,
                                             metal_graph_raw_span_for_batch(g, pos0, 1),
                                             token0);
        if (!ok) break;
        ok = metal_graph_capture_prefix_attn_state(g, il, 0) &&
             metal_graph_capture_prefix_index_state(g, il, 0);
        if (!ok) break;

        g->cur_hc_by_tier[this_tier] = cur1_by_tier[this_tier];
        g->after_ffn_hc_by_tier[this_tier] = next1_by_tier[this_tier];
        ok = metal_graph_encode_decode_layer(g,
                                             model,
                                             &weights->layer[il],
                                             il,
                                             pos1,
                                             g->layer_raw_cache[il],
                                             g->raw_cap,
                                             pos1 % g->raw_cap,
                                             metal_graph_raw_span_for_batch(g, pos1, 1),
                                             token1);
        if (!ok) break;

        ds4_gpu_tensor *tmp = cur0_by_tier[this_tier];
        cur0_by_tier[this_tier] = next0_by_tier[this_tier];
        next0_by_tier[this_tier] = tmp;
        tmp = cur1_by_tier[this_tier];
        cur1_by_tier[this_tier] = next1_by_tier[this_tier];
        next1_by_tier[this_tier] = tmp;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    g->spec_capture_prefixes = saved_capture;

    if (ok) {
        ok = metal_graph_set_active_tier_no_copy(g, cur_tier);
    }
    if (ok) {
        const bool split_top1 =
            logits0 == NULL &&
            g->cuda_tp_output &&
            metal_graph_cuda_verify_decode2_split_top1_requested();
        uint32_t output_ways = 0;
        g->cur_hc_by_tier[cur_tier] = cur0_by_tier[cur_tier];
        ok = ds4_gpu_begin_commands() != 0;
        if (ok) ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
        if (ok) ok = ds4_gpu_argmax_tensor(metal_graph_comp_selected(g),
                                           metal_graph_logits(g),
                                           DS4_N_VOCAB) != 0;
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        if (ok && split_top1) {
            ok = metal_graph_read_output_split_top1(g, output_ways, top0);
        } else if (ok) {
            ok = ds4_gpu_tensor_read(metal_graph_comp_selected(g), 0, top0, sizeof(*top0)) != 0;
        }
        if (ok && logits0) {
            ok = ds4_gpu_tensor_read(metal_graph_logits(g),
                                       0,
                                       logits0,
                                       (uint64_t)DS4_N_VOCAB * sizeof(logits0[0])) != 0;
        }
    }

    if (ok) {
        ok = metal_graph_set_active_tier_no_copy(g, cur_tier);
    }
    if (ok) {
        const bool split_top1 =
            logits1 == NULL &&
            top1 != NULL &&
            g->cuda_tp_output &&
            metal_graph_cuda_verify_decode2_split_top1_requested();
        int output_tiers[DS4_MAX_GPUS] = {0};
        uint32_t output_ways = 0;
        g->cur_hc_by_tier[cur_tier] = cur1_by_tier[cur_tier];
        ok = ds4_gpu_begin_commands() != 0;
        if (ok && split_top1) {
            ok = metal_graph_encode_output_head_split_top1(g,
                                                           model,
                                                           weights,
                                                           weights->output->dim[1],
                                                           output_tiers,
                                                           &output_ways);
        } else if (ok) {
            ok = metal_graph_encode_output_head(g, model, weights, weights->output->dim[1]);
            if (ok && top1) {
                ok = ds4_gpu_indexer_topk_tensor(metal_graph_comp_selected(g),
                                                 metal_graph_logits(g),
                                                 DS4_N_VOCAB,
                                                 1,
                                                 1) != 0;
            }
        }
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        if (ok && split_top1) {
            ok = metal_graph_read_output_split_top1(g, output_ways, top1);
        } else if (ok && top1) {
            ok = ds4_gpu_tensor_read(metal_graph_comp_selected(g), 0, top1, sizeof(*top1)) != 0;
        }
        if (ok) {
            if (logits1) {
                ok = ds4_gpu_tensor_read(metal_graph_logits(g),
                                           0,
                                           logits1,
                                           (uint64_t)DS4_N_VOCAB * sizeof(logits1[0])) != 0;
            }
        }
    }
    g->spec_capture_prefixes = saved_capture;
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        g->cur_hc_by_tier[t] = saved_cur_by_tier[t];
        g->after_ffn_hc_by_tier[t] = saved_after_by_tier[t];
    }
    if (g->placement) {
        if (saved_active_tier >= 0) {
            (void)metal_graph_set_active_tier_no_copy(g, saved_active_tier);
        } else {
            g->active_tier = saved_active_tier;
        }
    }
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        ds4_gpu_tensor_free(next1_by_tier[t]);
        ds4_gpu_tensor_free(next0_by_tier[t]);
        ds4_gpu_tensor_free(cur1_by_tier[t]);
        ds4_gpu_tensor_free(cur0_by_tier[t]);
    }
    return ok;
}

/* Pick a raw SWA cache size for Metal.  During batched prefill it must cover
 * the previous window plus the current ubatch. */
static uint32_t metal_graph_raw_cap_for_context(int ctx_size, uint32_t prefill_cap) {
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > (uint32_t)ctx_size) raw_window = (uint32_t)ctx_size;
    if (raw_window == 0) raw_window = 1;

    /*
     * During batched prefill the SWA cache must hold the current ubatch plus
     * the previous logical window. The cache is padded to a 256-row multiple
     * so the physical row order and FlashAttention block grouping match the
     * model path we compare against.
     */
    uint64_t wanted = (uint64_t)raw_window + prefill_cap;
    if (wanted > (uint32_t)ctx_size) wanted = (uint32_t)ctx_size;
    if (wanted == 0) wanted = 1;
    wanted = align_up(wanted, 256u);
    if (wanted > 8192u) wanted = 8192u;
    uint32_t raw_cap = (uint32_t)wanted;
    if (raw_cap < raw_window) raw_cap = raw_window;

#ifndef DS4_ROCM_BUILD
    const char *env = getenv("DS4_METAL_GRAPH_RAW_CAP");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env && v > 0) {
            raw_cap = (uint32_t)v;
            if (raw_cap > (uint32_t)ctx_size) raw_cap = (uint32_t)ctx_size;
            if (raw_cap > 8192u) raw_cap = 8192u;
            if (raw_cap < raw_window) raw_cap = raw_window;
        }
    }
#endif

    return raw_cap;
}

/* Choose the prefill ubatch size. Whole-batch is fastest for normal prompts.
 * Long Flash prompts default to 4096-token chunks; PRO defaults to 8192. */
static uint32_t metal_graph_prefill_cap_for_prompt(int prompt_len,
                                                   uint32_t prefill_chunk) {
    return ds4_prefill_cap_for_prompt(prompt_len, prefill_chunk);
}

/* When a server request shares a large prefix with the live checkpoint, extend
 * the KV cache with batched prefill instead of single-token decode.  On an M3
 * Max, prefill is faster from 2-token suffixes upward; keep the default at 4
 * as a conservative crossover.  The env knob remains useful for retuning. */
static uint32_t metal_graph_resume_prefill_min_tokens(void) {
#ifndef DS4_ROCM_BUILD
    const char *env = getenv("DS4_METAL_RESUME_PREFILL_MIN");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env) {
            if (v <= 0) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
#endif
    return 4u;
}

static uint32_t glm_graph_resume_prefill_min_tokens(void) {
#ifndef DS4_ROCM_BUILD
    const char *env = getenv("DS4_GLM_RESUME_PREFILL_MIN");
    if (env && env[0]) {
        char *endp = NULL;
        const long v = strtol(env, &endp, 10);
        if (endp != env) {
            if (v <= 0) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
#endif
    return 4u;
}

#define DS4_GLM_METAL_FULL_ATTN_DEFAULT_CONTEXT 4096u
#define DS4_GLM_METAL_STREAMING_FULL_ATTN_CONTEXT 8192u
#define DS4_GLM_METAL_FULL_ATTN_LAYER_FLUSH_CONTEXT 2048u
#define DS4_GLM_METAL_DISPLAY_PROGRESS_LAYER_TOKENS 32u
#define DS4_GLM_METAL_SMALL_PREFILL_STAGE_SYNC_TOKENS 0u
#define DS4_GLM_METAL_LONG_CONTEXT_THRESHOLD 65536u
#define DS4_GLM_METAL_LONG_CONTEXT_FULL_ATTN_CONTEXT 4096u
#define DS4_GLM_METAL_INDEXED_PREFILL_CHUNK_TOKENS 4096u
#define DS4_GLM_METAL_INDEXED_PREFILL_SCORE_SCRATCH_MB 256u

static uint32_t glm_graph_full_attention_cap(uint32_t ctx_size,
                                             bool     ssd_streaming);
static uint32_t glm_graph_indexed_prefill_chunk_tokens(
        uint32_t full_attention_cap,
        uint32_t compact_cap);
static uint32_t glm_graph_indexed_prefill_score_tokens(
        uint32_t indexed_prefill_cap,
        uint32_t compact_cap);

static uint64_t glm_graph_compact_cache_elem_bytes(void) {
    return DS4_GPU_GLM_COMPACT_CACHE_F16 ? sizeof(uint16_t) : sizeof(float);
}

static uint32_t glm_graph_compact_cache_is_f16(void) {
    return DS4_GPU_GLM_COMPACT_CACHE_F16 ? 1u : 0u;
}

static bool glm_graph_expanded_kv_cache_enabled(bool ssd_streaming) {
    (void)ssd_streaming;
    return false;
}

static bool glm_graph_layer_uses_full_indexer(uint32_t il) {
    if (il < DS4_N_LEADING_DENSE) return true;
    return il >= 6u && ((il - 6u) % 4u) == 0u;
}

static uint32_t glm_graph_normal_layer_count(void) {
    if (DS4_N_LAYER <= DS4_N_NEXTN_PREDICT || DS4_N_LAYER > DS4_MAX_LAYER) {
        return 0;
    }
    return DS4_N_LAYER - DS4_N_NEXTN_PREDICT;
}

static uint32_t glm_graph_full_indexer_layer_count_range(uint32_t layer_start,
                                                         uint32_t layer_end) {
    if (layer_start > layer_end) return 0;
    uint32_t n = 0;
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        if (glm_graph_layer_uses_full_indexer(il)) n++;
    }
    return n;
}

static uint64_t glm_graph_full_kv_cache_elem_bytes(void) {
    return sizeof(uint16_t);
}

static uint32_t glm_graph_indexer_top_k_limit(void) {
    return DS4_N_INDEXER_TOP_K;
}

static uint32_t glm_tp_head_split_min(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = 64;
        const char *env = getenv("DS4_GLM_TP_HEAD_SPLIT_MIN");
        if (env && env[0]) cached = atoi(env);
        if (cached < 0) cached = 0;
    }
    return (uint32_t)cached;
}

/* Correctness isolation: dump a hidden row (pre-output-norm), overwriting
 * on each call — run with -n 0 so the file ends as the final prompt row. */
static void glm_debug_dump_hidden_row(const ds4_gpu_tensor *t, uint32_t row) {
    const char *path = getenv("DS4_GLM_HIDDEN_DUMP");
    if (!path || !path[0] || !t) return;
    float *buf = malloc((size_t)DS4_N_EMBD * sizeof(float));
    if (!buf) return;
    if (ds4_gpu_tensor_read((ds4_gpu_tensor *)t,
                            (uint64_t)row * DS4_N_EMBD * sizeof(float),
                            buf,
                            (uint64_t)DS4_N_EMBD * sizeof(float))) {
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(buf, sizeof(float), (size_t)DS4_N_EMBD, f);
            fclose(f);
        }
    }
    free(buf);
}

/* Layer bisect: which layer's output hidden to dump (-1 = final/off,
 * -2 = every layer, one file per layer). */
static int glm_debug_hidden_dump_layer(void) {
    const char *v = getenv("DS4_GLM_HIDDEN_DUMP_LAYER");
    if (!v || !v[0]) return -1;
    if (strcmp(v, "all") == 0) return -2;
    return atoi(v);
}

static bool glm_debug_hidden_dump_layer_match(uint32_t il) {
    const int dl = glm_debug_hidden_dump_layer();
    return dl == -2 || dl == (int)il;
}

static void glm_debug_dump_raw_layer(const ds4_gpu_tensor *t,
                                     const char *tag,
                                     uint64_t bytes,
                                     uint32_t il,
                                     int pos) {
    const char *path = getenv("DS4_GLM_HIDDEN_DUMP");
    if (!path || !path[0] || !t) return;
    char full[1024];
    if (pos >= 0)
        snprintf(full, sizeof(full), "%s.%s.L%02u.T%02u", path, tag, il, pos);
    else
        snprintf(full, sizeof(full), "%s.%s.L%02u", path, tag, il);
    void *buf = malloc(bytes);
    if (!buf) return;
    if (ds4_gpu_tensor_read((ds4_gpu_tensor *)t, 0, buf, bytes)) {
        FILE *f = fopen(full, "wb");
        if (f) { fwrite(buf, 1, bytes, f); fclose(f); }
    }
    free(buf);
}

static void glm_debug_dump_hidden_layer(const ds4_gpu_tensor *t,
                                        uint32_t row,
                                        uint32_t il,
                                        uint32_t pos) {
    const char *path = getenv("DS4_GLM_HIDDEN_DUMP");
    if (!path || !path[0] || !t) return;
    char full[1024];
    snprintf(full, sizeof(full), "%s.L%02u.T%02u", path, il, pos);
    float *buf = malloc((size_t)DS4_N_EMBD * sizeof(float));
    if (!buf) return;
    if (ds4_gpu_tensor_read((ds4_gpu_tensor *)t,
                            (uint64_t)row * DS4_N_EMBD * sizeof(float),
                            buf,
                            (uint64_t)DS4_N_EMBD * sizeof(float))) {
        FILE *f = fopen(full, "wb");
        if (f) {
            fwrite(buf, sizeof(float), (size_t)DS4_N_EMBD, f);
            fclose(f);
        }
    }
    free(buf);
}

/* Correctness isolation: dump the post-prefill logits vector once. */
static void glm_debug_dump_prefill_logits(const float *logits) {
    static int dumped;
    const char *path = getenv("DS4_GLM_LOGIT_DUMP");
    if (!path || !path[0] || dumped || !logits) return;
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(logits, sizeof(float), (size_t)DS4_N_VOCAB, f);
    fclose(f);
    dumped = 1;
    fprintf(stderr, "ds4: prefill logits dumped to %s\n", path);
}

static bool glm_graph_indexed_prefill_trace_enabled(void) {
    return false;
}

static bool glm_graph_indexed_prefill_trace_all(void) {
    return false;
}

static uint32_t glm_graph_indexed_prefill_trace_slow_ms(void) {
    return 100u;
}

static uint32_t glm_graph_indexed_prefill_drain_interval(void) {
    return 16u;
}

static bool glm_graph_full_prefill_trace_enabled(void) {
    return false;
}

static bool glm_graph_full_prefill_trace_all(void) {
    return false;
}

static uint32_t glm_graph_full_prefill_trace_slow_ms(void) {
    return 100u;
}

static uint32_t glm_graph_full_prefill_drain_interval(void) {
    return 16u;
}

static void glm_graph_full_prefill_tracef(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ds4: GLM full prefill trace ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(ap);
}

static void glm_graph_indexed_prefill_tracef(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ds4: GLM indexed prefill trace ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    fflush(stderr);
    va_end(ap);
}

static uint32_t glm_graph_compact_cache_initial_cap(
        uint32_t ctx_size,
        uint32_t full_attention_cap) {
    if (ctx_size == 0) return 0;
    if (ctx_size <= full_attention_cap) return ctx_size;

    uint32_t cap = ctx_size;
    if (cap == 0) cap = full_attention_cap ? full_attention_cap : 1u;
    if (cap > ctx_size) cap = ctx_size;
    return cap;
}

static uint64_t glm_graph_compact_cache_bytes_for_cap(
        uint32_t normal_layers,
        uint32_t indexer_layers,
        uint32_t compact_cap) {
    if (compact_cap == 0) return 0;
    const uint64_t elem = glm_graph_compact_cache_elem_bytes();
    uint64_t total =
        (uint64_t)normal_layers *
        compact_cap *
        ((uint64_t)DS4_N_KV_LORA + DS4_N_ROT) *
        elem;
    total +=
        (uint64_t)indexer_layers *
        compact_cap *
        DS4_N_INDEXER_HEAD_DIM *
        elem;
    return total;
}

static uint64_t glm_graph_indexed_scratch_bytes_for_cap(
        uint32_t full_attention_cap,
        uint32_t compact_cap) {
    if (compact_cap == 0) return 0;
    const uint64_t indexed_rows =
        glm_graph_indexed_prefill_chunk_tokens(full_attention_cap, compact_cap);
    const uint64_t indexed_score_rows =
        glm_graph_indexed_prefill_score_tokens((uint32_t)indexed_rows,
                                               compact_cap);
    const uint64_t indexer_q_elems =
        (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t qk_low_elems =
        (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA;
    uint64_t bytes = (uint64_t)compact_cap * sizeof(float);
    const uint64_t indexer_top_k = glm_graph_indexer_top_k_limit();
    bytes += indexed_score_rows * (uint64_t)compact_cap * sizeof(float);
    bytes += indexed_rows * indexer_q_elems * sizeof(float);
    bytes += indexed_rows * DS4_N_INDEXER_HEAD * sizeof(float);
    bytes += indexed_rows * indexer_top_k * sizeof(uint32_t); /* batch_indexer_selected */
    bytes += indexed_rows * qk_low_elems * sizeof(float);     /* batch_qk_low */
    bytes += indexed_rows * qk_low_elems * sizeof(float);     /* batch_attn_lora */
    return bytes;
}

static uint64_t glm_graph_workspace_add_bytes(
        uint64_t total,
        uint64_t count,
        uint64_t elem_bytes) {
    return ds4_add_sat_u64(total, ds4_mul_sat_u64(count, elem_bytes));
}

static uint32_t glm_graph_indexed_decode_split_blocks(void);

static uint64_t glm_graph_workspace_bytes_for_cap(
        uint32_t full_attention_cap,
        uint32_t compact_cap,
        bool     ssd_streaming) {
    const bool expanded_kv =
        glm_graph_expanded_kv_cache_enabled(ssd_streaming);
    const uint64_t indexed_rows =
        compact_cap != 0 ?
            glm_graph_indexed_prefill_chunk_tokens(full_attention_cap,
                                                   compact_cap) :
            0;
    const uint64_t batch_rows =
        expanded_kv || indexed_rows == 0 ? full_attention_cap : indexed_rows;
    const uint64_t indexer_top_k = glm_graph_indexer_top_k_limit();
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA;
    const uint64_t q_nope =
        DS4_N_KEY_MLA > DS4_N_ROT ? (uint64_t)DS4_N_KEY_MLA - DS4_N_ROT : 0;
    const uint64_t heads_dim = (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA;
    const uint64_t kv_raw_dim = (uint64_t)DS4_N_KV_LORA + DS4_N_ROT;
    uint64_t dense_hidden_max =
        DS4_N_FF_DENSE > DS4_N_FF_EXP ? DS4_N_FF_DENSE : DS4_N_FF_EXP;
    if (dense_hidden_max == 0) dense_hidden_max = DS4_N_FF_EXP;
    const uint64_t sparse_mid_elems =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    const uint64_t ffn_mid_elems =
        dense_hidden_max > sparse_mid_elems ?
            dense_hidden_max :
            sparse_mid_elems;
    const uint64_t qk_low_elems =
        (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA;
    const uint64_t split_attn_blocks =
        glm_graph_indexed_decode_split_blocks();

    uint64_t bytes =
        glm_graph_indexed_scratch_bytes_for_cap(full_attention_cap,
                                                compact_cap);

    bytes = glm_graph_workspace_add_bytes(bytes, 3u, DS4_N_EMBD * sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, 2u, DS4_N_LORA_Q * sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, q_dim, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_INDEXER_HEAD_DIM, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          (uint64_t)DS4_N_INDEXER_HEAD *
                                          DS4_N_INDEXER_HEAD_DIM,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_INDEXER_HEAD, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, indexer_top_k, sizeof(uint32_t));
    bytes = glm_graph_workspace_add_bytes(bytes, qk_low_elems, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          split_attn_blocks * qk_low_elems,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          split_attn_blocks *
                                          DS4_N_HEAD * 2u,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, kv_raw_dim, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_KV_LORA, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          DS4_N_HEAD * q_nope,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, 2u * heads_dim, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, 6u, DS4_N_EMBD * sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, 2u * dense_hidden_max, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, ffn_mid_elems, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_EXPERT * 2u, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_EXPERT_USED, sizeof(int32_t));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_EXPERT_USED, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, DS4_N_VOCAB, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          (uint64_t)DS4_N_LAYER *
                                          DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                                          DS4_N_EXPERT_USED,
                                          sizeof(int32_t));

    bytes = glm_graph_workspace_add_bytes(bytes, batch_rows, sizeof(int32_t));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_EXPERT * 2u,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_EXPERT_USED,
                                          sizeof(int32_t));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_EXPERT_USED,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_EMBD * 7u,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_LORA_Q * 2u,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, batch_rows * q_dim, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_INDEXER_HEAD_DIM,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, batch_rows * kv_raw_dim, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_KV_LORA,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_HEAD * q_nope,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes, batch_rows * heads_dim * 2u, sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * dense_hidden_max * 2u,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * DS4_N_FF_EXP,
                                          sizeof(float));
    bytes = glm_graph_workspace_add_bytes(bytes,
                                          batch_rows * ffn_mid_elems,
                                          sizeof(float));
    if (indexed_rows != 0) {
        bytes = glm_graph_workspace_add_bytes(bytes,
                                              indexed_rows * qk_low_elems,
                                              sizeof(float));
    }
    return bytes;
}

static ds4_context_memory glm_graph_context_memory_estimate_for_compact_cap_slice(
        uint32_t ctx,
        uint32_t work_ctx,
        uint32_t compact_cap,
        bool     ssd_streaming,
        uint32_t layer_start,
        uint32_t layer_end) {
    ds4_context_memory m = {0};
    const uint32_t normal_layers = glm_graph_normal_layer_count();
    if (normal_layers == 0 || layer_start >= normal_layers ||
        layer_end < layer_start) {
        return m;
    }
    if (layer_end >= normal_layers) layer_end = normal_layers - 1u;
    const uint32_t layer_count = layer_end - layer_start + 1u;
    if (compact_cap > ctx) compact_cap = ctx;

    const bool expanded_kv = glm_graph_expanded_kv_cache_enabled(ssd_streaming);
    const uint32_t indexed_rows =
        compact_cap != 0 ?
            glm_graph_indexed_prefill_chunk_tokens(work_ctx, compact_cap) :
            0;
    const uint32_t batch_rows =
        expanded_kv || indexed_rows == 0 ? work_ctx : indexed_rows;

    m.prefill_cap = batch_rows;
    m.raw_cap = expanded_kv ? work_ctx : 0;
    if (expanded_kv) {
        m.raw_bytes = (uint64_t)layer_count *
                      work_ctx *
                      ((uint64_t)DS4_N_HEAD * (DS4_N_KEY_MLA + DS4_N_VALUE_MLA)) *
                      glm_graph_full_kv_cache_elem_bytes();
    }
    m.scratch_bytes =
        glm_graph_workspace_bytes_for_cap(work_ctx,
                                          compact_cap,
                                          ssd_streaming);
    if (compact_cap != 0) {
        m.comp_cap = compact_cap;
        m.compressed_bytes =
            glm_graph_compact_cache_bytes_for_cap(
                    layer_count,
                    glm_graph_full_indexer_layer_count_range(layer_start,
                                                             layer_end),
                    compact_cap);
    }
    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

static ds4_context_memory glm_graph_context_memory_estimate_for_compact_cap(
        uint32_t ctx,
        uint32_t work_ctx,
        uint32_t compact_cap,
        bool     ssd_streaming) {
    const uint32_t normal_layers = glm_graph_normal_layer_count();
    if (normal_layers == 0) {
        const ds4_context_memory empty = {0};
        return empty;
    }
    return glm_graph_context_memory_estimate_for_compact_cap_slice(
            ctx,
            work_ctx,
            compact_cap,
            ssd_streaming,
            0,
            normal_layers - 1u);
}

ds4_context_memory ds4_context_memory_estimate_with_prefill_mode(
        ds4_backend backend,
        int         ctx_size,
        uint32_t    prefill_chunk,
        bool        ssd_streaming) {
    ds4_context_memory m = {0};
    uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;

    if (ds4_backend_uses_graph(backend)) {
        if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
            const uint32_t work_ctx =
                glm_graph_full_attention_cap(ctx, ssd_streaming);
            const uint32_t compact_cap =
                glm_graph_compact_cache_initial_cap(ctx, work_ctx);
            m = glm_graph_context_memory_estimate_for_compact_cap(ctx,
                                                                  work_ctx,
                                                                  compact_cap,
                                                                  ssd_streaming);
            return m;
        }
        m.prefill_cap = metal_graph_prefill_cap_for_prompt((int)ctx,
                                                           prefill_chunk);
        m.raw_cap = metal_graph_raw_cap_for_context((int)ctx, m.prefill_cap);

        uint32_t min_ratio = UINT32_MAX;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
        }
        if (min_ratio == UINT32_MAX) min_ratio = ctx;
        m.comp_cap = ctx / min_ratio + 2u;
        if (m.comp_cap < 2u) m.comp_cap = 2u;

        m.raw_bytes = (uint64_t)DS4_N_LAYER *
                      m.raw_cap *
                      DS4_SPARK_KV_ROW_BYTES;
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t layer_comp_cap = ctx / ratio + 2u;
            m.compressed_bytes += (uint64_t)layer_comp_cap *
                                  DS4_SPARK_KV_ROW_BYTES;
            if (ratio == 4) {
                m.compressed_bytes += (uint64_t)layer_comp_cap *
                                      DS4_SPARK_INDEX_ROW_BYTES;
            }
        }
        uint64_t attn_stage_cap = (uint64_t)(m.prefill_cap / min_ratio + 2u);
        if (attn_stage_cap < 2u) attn_stage_cap = 2u;
        m.scratch_bytes = 2ull *
                          m.comp_cap *
                          m.prefill_cap *
                          sizeof(float) +
                          attn_stage_cap *
                              (DS4_N_HEAD_DIM + DS4_N_INDEXER_HEAD_DIM) *
                              sizeof(float);
    } else {
        m.raw_cap = ds4_default_raw_cap(ctx);
        m.raw_bytes = (uint64_t)DS4_N_LAYER *
                      m.raw_cap *
                      DS4_N_HEAD_DIM *
                      sizeof(float);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const uint32_t ratio = ds4_layer_compress_ratio(il);
            if (ratio == 0) continue;
            const uint32_t comp_cap = ctx / ratio + 2u;
            if (ratio == 4) m.comp_cap = comp_cap;
            m.compressed_bytes += (uint64_t)comp_cap *
                                  DS4_N_HEAD_DIM *
                                  sizeof(float);
            if (ratio == 4) {
                m.compressed_bytes += (uint64_t)comp_cap *
                                      DS4_N_INDEXER_HEAD_DIM *
                                      sizeof(float);
            }
        }
        if (m.comp_cap == 0) m.comp_cap = ctx / 4u + 2u;
        m.scratch_bytes = ((uint64_t)(m.raw_cap + m.comp_cap) * sizeof(float)) +
                          ((uint64_t)m.comp_cap * sizeof(float)) +
                          ((uint64_t)m.comp_cap * sizeof(bool));
    }

    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int         ctx_size,
        uint32_t    prefill_chunk) {
    return ds4_context_memory_estimate_with_prefill_mode(backend,
                                                         ctx_size,
                                                         prefill_chunk,
                                                         false);
}

ds4_context_memory ds4_context_memory_estimate(ds4_backend backend,
                                               int         ctx_size) {
    return ds4_context_memory_estimate_with_prefill(backend, ctx_size, 0);
}

static int metal_graph_prompt_logits_test(
        const ds4_model   *model,
        const ds4_weights *weights,
        const token_vec   *prompt,
        int                ctx_size) {
    int n_test = prompt->len;
    const char *n_test_env = getenv("DS4_METAL_GRAPH_PROMPT_TOKENS");
    if (n_test_env && n_test_env[0]) {
        char *endp = NULL;
        const long v = strtol(n_test_env, &endp, 10);
        if (endp != n_test_env && v > 0 && v <= prompt->len) n_test = (int)v;
    }

    if (n_test <= 0 || n_test > ctx_size) {
        fprintf(stderr, "ds4: Metal graph prompt test needs 1..%d prompt tokens\n", ctx_size);
        return 1;
    }

    const uint32_t raw_cap = metal_graph_raw_cap_for_context(ctx_size, (uint32_t)n_test);

    ds4_gpu_graph g;
    /* diagnostic single-tier callsite; placement=NULL. */
    bool ok = metal_graph_alloc_raw_cap(&g, weights, &weights->layer[0],
                                        raw_cap, (uint32_t)ctx_size,
                                        (uint32_t)n_test, false, NULL, false, NULL);
    if (!ok) {
        metal_graph_free(&g);
        fprintf(stderr, "ds4: failed to initialize Metal graph prompt test runtime\n");
        return 1;
    }
    const bool memory_report = getenv("DS4_METAL_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_gpu_print_memory_report("after graph alloc");

    ds4_kv_cache cpu_cache;
    kv_cache_init(&cpu_cache, (uint32_t)ctx_size, raw_cap);
    float *cpu_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *gpu_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    float *oracle_logits = NULL;

    const char *oracle_path = getenv("DS4_ORACLE_LOGITS");
    if (oracle_path && oracle_path[0]) {
        oracle_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
        if (!read_f32_binary_file(oracle_path, oracle_logits, DS4_N_VOCAB)) {
            free(oracle_logits);
            oracle_logits = NULL;
        }
    }

    for (int t = 0; t < n_test; t++) {
        const bool last = t == n_test - 1;
        forward_token_raw_swa_cpu(last ? cpu_logits : NULL,
                                  model,
                                  weights,
                                  &cpu_cache,
                                  prompt->v[t],
                                  (uint32_t)t);
    }
    ok = metal_graph_prefill_raw_swa(&g, model, weights, prompt, n_test,
                                     gpu_logits, true, NULL, NULL,
                                     NULL, NULL, NULL);
    if (memory_report) ds4_gpu_print_memory_report("after prompt graph");

    if (ok) {
        const char *dump_gpu = getenv("DS4_METAL_GRAPH_DUMP_LOGITS");
        if (dump_gpu && dump_gpu[0]) {
            if (write_f32_binary_file(dump_gpu, gpu_logits, DS4_N_VOCAB)) {
                fprintf(stderr, "ds4: wrote Metal graph logits to %s\n", dump_gpu);
            }
        }
        const char *dump_cpu = getenv("DS4_CPU_DUMP_LOGITS");
        if (dump_cpu && dump_cpu[0]) {
            if (write_f32_binary_file(dump_cpu, cpu_logits, DS4_N_VOCAB)) {
                fprintf(stderr, "ds4: wrote CPU logits to %s\n", dump_cpu);
            }
        }
        if (getenv("DS4_METAL_GRAPH_TRACE_CACHE") != NULL ||
            getenv("DS4_METAL_GRAPH_TRACE_COMP") != NULL) {
            for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
                const uint32_t n_raw = cpu_cache.layer[il].n_raw;
                if (n_raw != 0) {
                    const uint64_t raw_phys_n = (uint64_t)raw_cap * DS4_N_HEAD_DIM;
                    const uint64_t raw_logical_n = (uint64_t)n_raw * DS4_N_HEAD_DIM;
                    const uint32_t raw_start = n_raw < raw_cap ? 0u : ((uint32_t)n_test % raw_cap);
                    float *gpu_raw_phys = xmalloc((size_t)raw_phys_n * sizeof(float));
                    float *gpu_raw_logical = xmalloc((size_t)raw_logical_n * sizeof(float));
                    if (ds4_gpu_tensor_read(g.layer_raw_cache[il], 0, gpu_raw_phys, raw_phys_n * sizeof(float)) != 0) {
                        for (uint32_t r = 0; r < n_raw; r++) {
                            const uint32_t phys = (raw_start + r) % raw_cap;
                            memcpy(gpu_raw_logical + (uint64_t)r * DS4_N_HEAD_DIM,
                                   gpu_raw_phys + (uint64_t)phys * DS4_N_HEAD_DIM,
                                   (size_t)DS4_N_HEAD_DIM * sizeof(float));
                        }
                        fprintf(stderr,
                                "ds4: cache trace layer %u raw_n=%u raw_start=%u raw_max=%g raw_rms=%g\n",
                                il, n_raw, raw_start,
                                max_abs_diff(cpu_cache.layer[il].raw_kv, gpu_raw_logical, raw_logical_n),
                                rms_abs_diff(cpu_cache.layer[il].raw_kv, gpu_raw_logical, raw_logical_n));
                    }
                    free(gpu_raw_logical);
                    free(gpu_raw_phys);
                }

                const uint32_t n_comp = cpu_cache.layer[il].n_comp;
                if (n_comp == 0) continue;
                const uint64_t n = (uint64_t)n_comp * DS4_N_HEAD_DIM;
                float *gpu_comp = xmalloc((size_t)n * sizeof(float));
                bool comp_read = false;
                if (DS4_GPU_ATTN_COMP_CACHE_F16) {
                    uint16_t *gpu_comp_h = xmalloc((size_t)n * sizeof(uint16_t));
                    if (ds4_gpu_tensor_read(g.layer_attn_comp_cache[il], 0,
                                            gpu_comp_h, n * sizeof(uint16_t)) != 0) {
                        for (uint64_t i = 0; i < n; i++) gpu_comp[i] = f16_to_f32(gpu_comp_h[i]);
                        comp_read = true;
                    }
                    free(gpu_comp_h);
                } else {
                    comp_read = ds4_gpu_tensor_read(g.layer_attn_comp_cache[il], 0,
                                                    gpu_comp, n * sizeof(float)) != 0;
                }
                if (comp_read) {
                    fprintf(stderr,
                            "ds4: comp trace layer %u n=%u attn_max=%g attn_rms=%g\n",
                            il, n_comp,
                            max_abs_diff(cpu_cache.layer[il].attn_comp_kv, gpu_comp, n),
                            rms_abs_diff(cpu_cache.layer[il].attn_comp_kv, gpu_comp, n));
                }
                free(gpu_comp);

                const uint32_t n_index = cpu_cache.layer[il].n_index_comp;
                if (n_index != 0 && g.layer_index_comp_cache[il]) {
                    const uint64_t ni = (uint64_t)n_index * DS4_N_INDEXER_HEAD_DIM;
                    float *gpu_index = xmalloc((size_t)ni * sizeof(float));
                    if (ds4_gpu_tensor_read(g.layer_index_comp_cache[il], 0, gpu_index, ni * sizeof(float)) != 0) {
                        fprintf(stderr,
                                "ds4: comp trace layer %u n=%u index_max=%g index_rms=%g\n",
                                il, n_index,
                                max_abs_diff(cpu_cache.layer[il].index_comp_kv, gpu_index, ni),
                                rms_abs_diff(cpu_cache.layer[il].index_comp_kv, gpu_index, ni));
                    }
                    free(gpu_index);
                }
            }
        }
        const uint64_t cpu_top = argmax_f32(cpu_logits, DS4_N_VOCAB);
        const uint64_t gpu_top = argmax_f32(gpu_logits, DS4_N_VOCAB);
        fprintf(stderr,
                "ds4: Metal prompt graph logits: tokens=%d logits_max=%g logits_rms=%g cpu_top=%llu gpu_top=%llu cpu_top_logit=%g gpu_top_logit=%g\n",
                n_test,
                max_abs_diff(cpu_logits, gpu_logits, DS4_N_VOCAB),
                rms_abs_diff(cpu_logits, gpu_logits, DS4_N_VOCAB),
                (unsigned long long)cpu_top,
                (unsigned long long)gpu_top,
                cpu_logits[cpu_top],
                gpu_logits[gpu_top]);
        if (oracle_logits) {
            const uint64_t oracle_top = argmax_f32(oracle_logits, DS4_N_VOCAB);
            fprintf(stderr,
                    "ds4: oracle logits: tokens=%d oracle_top=%llu oracle_top_logit=%g cpu_max=%g cpu_rms=%g metal_max=%g metal_rms=%g\n",
                    n_test,
                    (unsigned long long)oracle_top,
                    oracle_logits[oracle_top],
                    max_abs_diff(cpu_logits, oracle_logits, DS4_N_VOCAB),
                    rms_abs_diff(cpu_logits, oracle_logits, DS4_N_VOCAB),
                    max_abs_diff(gpu_logits, oracle_logits, DS4_N_VOCAB),
                    rms_abs_diff(gpu_logits, oracle_logits, DS4_N_VOCAB));
        }
    } else {
        fprintf(stderr, "ds4: Metal prompt graph logits test failed\n");
        if (ds4_gpu_synchronize() == 0) {
            fprintf(stderr, "ds4: Metal synchronize after prompt graph failure also failed\n");
        }
    }

    free(gpu_logits);
    free(cpu_logits);
    free(oracle_logits);
    kv_cache_free(&cpu_cache);
    metal_graph_free(&g);
    return ok ? 0 : 1;
}

#endif /* !DS4_NO_GPU */
