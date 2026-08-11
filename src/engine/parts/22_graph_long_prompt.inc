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
    }
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
    return metal_graph_verify_suffix_tops_impl(g, model, weights,
                                               prompt, start,
                                               n_tokens,
                                               capture_prefix1,
                                               capture_dspark_hidden,
                                               row_tops, row_logits,
                                               timing);
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

#endif /* !DS4_NO_GPU */
