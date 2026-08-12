#include "engine_internal.h"

/* Graph State module. */
/* This repository has one persistent-cache ABI: the native GB10 packed
 * representation. F32 is retained only for short-lived compressor staging. */



/* Tensors that are temporary for chunked prefill and grouped multi-session
 * decode. The batched server serializes every operation that uses them, so one
 * engine-owned set can be aliased by all resident session graphs. */


/* Class P accessors retain a one-element array ABI for uniform graph code.
 * The single-GB10 runtime always uses active_tier zero. */

/* --power N GPU duty-cycle throttling helpers. --power=100 is a no-op. */

bool graph_power_throttle_enabled(const ds4_gpu_graph *g) {
    return g && g->power_percent > 0 && g->power_percent < 100;
}

static double graph_power_update_avg(double avg, double sample) {
    if (sample <= 0.0 || !isfinite(sample)) return avg;
    if (avg <= 0.0 || !isfinite(avg)) return sample;
    return avg * 0.875 + sample * 0.125;
}

static void graph_power_sleep(double work_sec, uint32_t power_percent) {
    if (power_percent == 0 || power_percent >= 100) return;
    /* Target duty cycle: work / (work + sleep) = power / 100.
     * At --power 50 this sleeps for one measured work interval; at 25 it
     * sleeps for three. */
    const double sleep = work_sec * (100.0 - (double)power_percent) /
                         (double)power_percent;
    sleep_sec(sleep);
}

void graph_power_note_prefill_layer(ds4_gpu_graph *g,
                                           uint32_t il,
                                           double elapsed_sec) {
    if (!graph_power_throttle_enabled(g)) return;
    if (il >= DS4_N_LAYER) return;
    g->prefill_layer_avg_sec[il] =
        graph_power_update_avg(g->prefill_layer_avg_sec[il], elapsed_sec);
    graph_power_sleep(g->prefill_layer_avg_sec[il], g->power_percent);
}

void graph_power_note_decode_token(ds4_gpu_graph *g, double elapsed_sec) {
    if (!graph_power_throttle_enabled(g)) return;
    g->decode_token_avg_sec =
        graph_power_update_avg(g->decode_token_avg_sec, elapsed_sec);
    graph_power_sleep(g->decode_token_avg_sec, g->power_percent);
}

static void metal_graph_free_prefill_workspace(ds4_gpu_graph *g) {
    if (!g || !g->owns_prefill_workspace) return;
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
#define DS4_FREE_PREFILL_FIELD(name)                 \
        ds4_gpu_tensor_free(g->name##_by_tier[t]);   \
        g->name##_by_tier[t] = NULL;
        DS4_GPU_PREFILL_WORKSPACE_FIELDS(DS4_FREE_PREFILL_FIELD)
#undef DS4_FREE_PREFILL_FIELD
    }
    ds4_gpu_tensor_free(g->batch_q_half);
    g->batch_q_half = NULL;
    g->owns_prefill_workspace = false;
}

/* Release every tensor owned by the whole-model CUDA graph runtime. */
void metal_graph_free(ds4_gpu_graph *g) {
    /* Captured decode-island graphs bake this graph's buffer addresses
     * into their kernel nodes; retire them before the buffers go away. */
    ds4_gpu_decode_graphs_invalidate();
    /* free every Class P slot across all DS4_MAX_GPUS tier
     * slots. Unallocated slots are NULL and ds4_gpu_tensor_free(NULL) is a
     * no-op. The hc_pre / hc_post / hc_comb views must be freed BEFORE
     * their parent hc_split — view destruction releases its own struct
     * but does not touch the parent's memory. */
    metal_graph_free_prefill_workspace(g);
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        ds4_gpu_tensor_free(g->directional_steering_dirs_by_tier[t]);
        g->directional_steering_dirs_by_tier[t] = NULL;
    }
    /* Class H free across all tier slots. Non-head slots are
     * NULL and ds4_gpu_tensor_free(NULL) is a no-op. */
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        ds4_gpu_tensor_free(g->logits_by_tier[t]);
        g->logits_by_tier[t] = NULL;
    }
    ds4_gpu_tensor_free(g->spec_logits);
    /* Class H output-head free across all tier slots. */
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        ds4_gpu_tensor_free(g->output_norm_by_tier[t]);
        g->output_norm_by_tier[t] = NULL;
        ds4_gpu_tensor_free(g->output_embd_by_tier[t]);
        g->output_embd_by_tier[t] = NULL;
        ds4_gpu_tensor_free(g->output_weights_by_tier[t]);
        g->output_weights_by_tier[t] = NULL;
        ds4_gpu_tensor_free(g->output_pre_by_tier[t]);
        g->output_pre_by_tier[t] = NULL;
    }
    /* Class P decode scratch + routed-FFN free across all
     * tier slots. ffn_out is also a Class P field freed here. */
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        ds4_gpu_tensor_free(g->after_ffn_hc_by_tier[t]);
        ds4_gpu_tensor_free(g->ffn_out_by_tier[t]);
        ds4_gpu_tensor_free(g->routed_out_by_tier[t]);
        ds4_gpu_tensor_free(g->routed_down_by_tier[t]);
        ds4_gpu_tensor_free(g->routed_mid_by_tier[t]);
        ds4_gpu_tensor_free(g->routed_up_by_tier[t]);
        ds4_gpu_tensor_free(g->routed_gate_by_tier[t]);
        ds4_gpu_tensor_free(g->router_weights_by_tier[t]);
        ds4_gpu_tensor_free(g->router_selected_by_tier[t]);
        ds4_gpu_tensor_free(g->router_probs_by_tier[t]);
        ds4_gpu_tensor_free(g->router_logits_by_tier[t]);
        ds4_gpu_tensor_free(g->shared_out_by_tier[t]);
        ds4_gpu_tensor_free(g->shared_mid_by_tier[t]);
        ds4_gpu_tensor_free(g->shared_up_by_tier[t]);
        ds4_gpu_tensor_free(g->shared_gate_by_tier[t]);
        ds4_gpu_tensor_free(g->ffn_norm_by_tier[t]);
        ds4_gpu_tensor_free(g->ffn_cur_by_tier[t]);
        ds4_gpu_tensor_free(g->after_attn_hc_by_tier[t]);
        ds4_gpu_tensor_free(g->attn_out_by_tier[t]);
        ds4_gpu_tensor_free(g->attn_low_by_tier[t]);
        ds4_gpu_tensor_free(g->heads_by_tier[t]);
        ds4_gpu_tensor_free(g->comp_sc_cur_by_tier[t]);
        ds4_gpu_tensor_free(g->index_comp_kv_cur_by_tier[t]);
        ds4_gpu_tensor_free(g->index_comp_sc_cur_by_tier[t]);
        ds4_gpu_tensor_free(g->comp_kv_cur_by_tier[t]);
        ds4_gpu_tensor_free(g->attn_comp_stage_by_tier[t]);
        ds4_gpu_tensor_free(g->index_comp_stage_by_tier[t]);
        ds4_gpu_tensor_free(g->comp_mask_by_tier[t]);
        ds4_gpu_tensor_free(g->comp_selected_by_tier[t]);
        ds4_gpu_tensor_free(g->indexer_scores_by_tier[t]);
        ds4_gpu_tensor_free(g->indexer_weights_by_tier[t]);
        ds4_gpu_tensor_free(g->indexer_q_by_tier[t]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_raw_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_kv[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_index_state_score[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->spec_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_index_state_score[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_attn_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_attn_state_score[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_index_state_kv[il]);
        ds4_gpu_tensor_free(g->spec_prefix1_index_state_score[il]);
    }
    /* Class P decode-step scratch + decode HC group free across
     * all tier slots. hc_pre / hc_post / hc_comb are VIEWS of hc_split — free
     * them before hc_split so the view struct release happens with the parent
     * still pointer-valid (view free does not touch parent memory). */
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
        ds4_gpu_tensor_free(g->kv_by_tier[t]);
        ds4_gpu_tensor_free(g->kv_raw_by_tier[t]);
        ds4_gpu_tensor_free(g->q_by_tier[t]);
        ds4_gpu_tensor_free(g->qr_norm_by_tier[t]);
        ds4_gpu_tensor_free(g->qr_by_tier[t]);
        ds4_gpu_tensor_free(g->attn_norm_by_tier[t]);
        ds4_gpu_tensor_free(g->attn_cur_by_tier[t]);
        ds4_gpu_tensor_free(g->hc_comb_by_tier[t]);
        ds4_gpu_tensor_free(g->hc_post_by_tier[t]);
        ds4_gpu_tensor_free(g->hc_pre_by_tier[t]);
        ds4_gpu_tensor_free(g->hc_split_by_tier[t]);
        ds4_gpu_tensor_free(g->hc_mix_by_tier[t]);
        ds4_gpu_tensor_free(g->flat_hc_by_tier[t]);
        ds4_gpu_tensor_free(g->cur_hc_by_tier[t]);
    }
    ds4_gpu_tensor_free(g->dspark_position_ids);
    ds4_gpu_tensor_free(g->dspark_stage_output_hc);
    ds4_gpu_tensor_free(g->dspark_stage_input_hc);
    ds4_gpu_tensor_free(g->dspark_target_hc);
    ds4_gpu_tensor_free(g->dspark_draft_hc);
    ds4_gpu_tensor_free(g->dspark_draft_tokens);
    for (uint32_t stage = 0; stage < DS4_DSPARK_MAX_STAGES; stage++) {
        ds4_gpu_tensor_free(g->dspark_raw_cache[stage]);
    }
    ds4_gpu_tensor_free(g->dspark_main_x);
    ds4_gpu_tensor_free(g->dspark_stage0_proj);
    ds4_gpu_tensor_free(g->dspark_stage0_packed);
    ds4_gpu_tensor_free(g->dspark_target_hidden_batch);
    ds4_gpu_tensor_free(g->dspark_target_hidden);
    ds4_gpu_tensor_free(g->dspark_hc_mean_rows);
    ds4_gpu_tensor_free(g->dspark_hc_mean_weights);
    memset(g, 0, sizeof(*g));
}

bool metal_tensor_fill_f32(ds4_gpu_tensor *t, float v, uint64_t n) {
    return ds4_gpu_tensor_fill_f32(t, v, n) != 0;
}

/* =========================================================================
 * Directional Steering.
 * =========================================================================
 *
 * A steering file contains one normalized 4096-wide direction per layer.  When
 * enabled, the CUDA graph edits selected block outputs in-place:
 *
 *     y = y - scale * v * dot(v, y)
 *
 * Positive scales remove the represented direction from the activation.
 * Negative scales add it.  This is deliberately explicit and opt-in; with zero
 * scales, the release graph does not allocate the direction tensor and follows
 * the normal inference path.
 */

/* directional_steering_dirs is Class P — replicated per tier.
 * The same host directions buffer is written to every tier slot the engine's
 * placement uses, then the load buffer is freed. Read-only after init, so
 * the per-tier replicas stay byte-identical and never re-sync. */
bool metal_graph_load_directional_steering(
        ds4_gpu_graph *g,
        const char      *path,
        float            attn_scale,
        float            ffn_scale) {
    if (attn_scale == 0.0f && ffn_scale == 0.0f) return true;

    if (!path || !path[0]) {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        return false;
    }

    const uint64_t n = (uint64_t)DS4_N_LAYER * DS4_N_EMBD;
    float *dirs = xmalloc((size_t)n * sizeof(dirs[0]));
    bool ok = read_f32_binary_file(path, dirs, n);
    if (ok) {
        /* Replicate the directions buffer onto every Class P tier slot that
         * has any other Class P scratch allocated (used_tier marker is the
         * presence of g->cur_hc_by_tier[t]). Single-tier: only slot 0. */
        bool any = false;
        for (int t = 0; ok && t < DS4_MAX_GPUS; t++) {
            if (!g->cur_hc_by_tier[t]) continue;
            g->directional_steering_dirs_by_tier[t] =
                ds4_gpu_tensor_alloc_ptr_on(t, n * sizeof(dirs[0]));
            ok = g->directional_steering_dirs_by_tier[t] != NULL &&
                 ds4_gpu_tensor_write(g->directional_steering_dirs_by_tier[t],
                                      0, dirs, n * sizeof(dirs[0])) != 0;
            if (ok) any = true;
        }
        if (ok && !any) {
            /* No used tiers — graph not allocated yet. This shouldn't happen
             * given the call site ordering, but bail cleanly. */
            ok = false;
        }
    }
    free(dirs);

    if (!ok) {
        fprintf(stderr, "ds4: failed to load directional steering vectors from %s\n", path);
        return false;
    }
    g->directional_steering_attn_scale = attn_scale;
    g->directional_steering_ffn_scale = ffn_scale;
    fprintf(stderr, "ds4: directional steering enabled: %s attn=%g ffn=%g\n",
            path, (double)attn_scale, (double)ffn_scale);
    return true;
}

bool metal_graph_directional_steering_attn_enabled(const ds4_gpu_graph *g) {
    return g && metal_graph_directional_steering_dirs(g) &&
           g->directional_steering_attn_scale != 0.0f;
}

bool metal_graph_directional_steering_ffn_enabled(const ds4_gpu_graph *g) {
    return g && metal_graph_directional_steering_dirs(g) &&
           g->directional_steering_ffn_scale != 0.0f;
}

static bool metal_graph_apply_directional_steering(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows,
        float             scale) {
    if (!g || !metal_graph_directional_steering_dirs(g) || scale == 0.0f) return true;
    return ds4_gpu_directional_steering_project_tensor(x,
                                            metal_graph_directional_steering_dirs(g),
                                            il,
                                            DS4_N_EMBD,
                                            rows,
                                            scale) != 0;
}

bool metal_graph_apply_directional_steering_attn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return metal_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_attn_scale : 0.0f);
}

bool metal_graph_apply_directional_steering_ffn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return metal_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_ffn_scale : 0.0f);
}

bool metal_graph_configure_dspark_capture(
        ds4_gpu_graph            *g,
        const ds4_dspark_weights *dw) {
    if (!g || !dw || dw->target_layer_count == 0) return true;
    if (dw->target_layer_count > DS4_DSPARK_MAX_TARGET_LAYERS ||
        DS4_N_HC == 0 ||
        DS4_N_HC > DS4_MAX_HC) {
        return false;
    }

    g->dspark_hc_mean_weights =
        ds4_gpu_tensor_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->dspark_hc_mean_rows =
        ds4_gpu_tensor_alloc((uint64_t)g->prefill_cap *
                             DS4_N_HC * sizeof(float));
    g->dspark_target_hidden =
        ds4_gpu_tensor_alloc((uint64_t)dw->target_layer_count *
                             DS4_N_EMBD * sizeof(float));
    g->dspark_target_hidden_batch =
        ds4_gpu_tensor_alloc((uint64_t)dw->target_layer_count *
                             g->prefill_cap *
                             DS4_N_EMBD * sizeof(float));
    if (dw->block_size != 0 && dw->block_size <= DS4_DSPARK_MAX_BLOCK_SIZE) {
        g->dspark_stage0_packed =
            ds4_gpu_tensor_alloc(((uint64_t)dw->block_size + 1u) *
                                 dw->target_layer_count *
                                 DS4_N_EMBD * sizeof(float));
    }
    g->dspark_stage0_proj =
        ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    g->dspark_main_x =
        ds4_gpu_tensor_alloc((uint64_t)DS4_N_EMBD * sizeof(float));
    if (!g->dspark_hc_mean_weights || !g->dspark_hc_mean_rows ||
        !g->dspark_target_hidden || !g->dspark_target_hidden_batch ||
        !g->dspark_stage0_proj || !g->dspark_main_x) {
        return false;
    }
    if (dw->block_size != 0 && dw->block_size <= DS4_DSPARK_MAX_BLOCK_SIZE) {
        const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
        g->dspark_draft_tokens =
            ds4_gpu_tensor_alloc((uint64_t)dw->block_size * sizeof(int32_t));
        g->dspark_draft_hc =
            ds4_gpu_tensor_alloc((uint64_t)dw->block_size * hc_dim * sizeof(float));
        g->dspark_target_hc =
            ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
        g->dspark_stage_input_hc =
            ds4_gpu_tensor_alloc((uint64_t)(dw->block_size + 1u) *
                                 hc_dim * sizeof(float));
        g->dspark_stage_output_hc =
            ds4_gpu_tensor_alloc((uint64_t)dw->block_size *
                                 hc_dim * sizeof(float));
        g->dspark_position_ids =
            ds4_gpu_tensor_alloc((uint64_t)(dw->block_size + 1u) *
                                 sizeof(int32_t));
        if (!g->dspark_draft_tokens || !g->dspark_draft_hc ||
            !g->dspark_target_hc || !g->dspark_stage_input_hc ||
            !g->dspark_stage_output_hc || !g->dspark_position_ids) {
            return false;
        }
        if (dw->n_stages != 0 && g->raw_cap != 0) {
            for (uint32_t stage = 0; stage < dw->n_stages; stage++) {
                g->dspark_raw_cache[stage] =
                    ds4_gpu_tensor_alloc((uint64_t)g->raw_cap *
                                         DS4_N_HEAD_DIM * sizeof(float));
                if (!g->dspark_raw_cache[stage]) return false;
            }
            g->dspark_cache_cap = g->raw_cap;
            g->dspark_cache_start = 0;
            g->dspark_cache_token_start = 0;
            g->dspark_cache_len = 0;
        }
        g->dspark_block_size = dw->block_size;
    }

    float mean[DS4_MAX_HC] = {0};
    const float inv_hc = 1.0f / (float)DS4_N_HC;
    for (uint32_t i = 0; i < DS4_N_HC; i++) mean[i] = inv_hc;
    if (ds4_gpu_tensor_write(g->dspark_hc_mean_weights,
                             0,
                             mean,
                             (uint64_t)DS4_N_HC * sizeof(mean[0])) == 0) {
        return false;
    }
    const uint64_t mean_rows_count = (uint64_t)g->prefill_cap * DS4_N_HC;
    if (mean_rows_count == 0 || mean_rows_count > (uint64_t)SIZE_MAX / sizeof(float)) {
        return false;
    }
    float *mean_rows = xmalloc((size_t)mean_rows_count * sizeof(mean_rows[0]));
    for (uint64_t i = 0; i < mean_rows_count; i++) mean_rows[i] = inv_hc;
    const bool mean_rows_ok =
        ds4_gpu_tensor_write(g->dspark_hc_mean_rows,
                             0,
                             mean_rows,
                             mean_rows_count * sizeof(mean_rows[0])) != 0;
    free(mean_rows);
    if (!mean_rows_ok) return false;

    g->dspark_target_layer_count = dw->target_layer_count;
    memcpy(g->dspark_target_layers,
           dw->target_layers,
           (size_t)dw->target_layer_count * sizeof(g->dspark_target_layers[0]));
    g->dspark_capture_mask = 0;
    g->dspark_capture_checkpoint_len = 0;
    g->dspark_capture_batch_mask = 0;
    g->dspark_capture_batch_start = 0;
    g->dspark_capture_batch_tokens = 0;
    g->dspark_capture_valid = false;
    g->dspark_capture_batch_valid = false;
    g->dspark_capture_enabled = true;
    return true;
}

static uint64_t metal_graph_kv_cache_bytes_for_context(uint32_t ctx_size, uint32_t raw_cap) {
    uint64_t bytes = (uint64_t)DS4_N_LAYER *
                     raw_cap *
                     DS4_SPARK_KV_ROW_BYTES;

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint64_t comp_cap = (uint64_t)(ctx_size / ratio + 2u);
        bytes += comp_cap * DS4_SPARK_KV_ROW_BYTES;
        if (ratio == 4) {
            bytes += comp_cap * DS4_SPARK_INDEX_ROW_BYTES;
        }
    }
    return bytes;
}

uint64_t metal_graph_context_bytes_for_kv_policy(
        uint32_t  ctx_size,
        uint32_t  raw_cap,
        uint32_t  prefill_cap,
        uint64_t *kv_cache_bytes_out) {
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    uint64_t comp_cap = (uint64_t)(ctx_size / min_ratio + 2u);
    if (comp_cap < 2u) comp_cap = 2u;
    const uint64_t kv_cache_bytes = metal_graph_kv_cache_bytes_for_context(ctx_size, raw_cap);
    if (kv_cache_bytes_out) *kv_cache_bytes_out = kv_cache_bytes;
    uint64_t bytes = kv_cache_bytes +
                     2ull * comp_cap * prefill_cap * sizeof(float);
    uint64_t attn_stage_cap = (uint64_t)(prefill_cap / min_ratio + 2u);
    if (attn_stage_cap < 2u) attn_stage_cap = 2u;
    bytes += attn_stage_cap * DS4_N_HEAD_DIM * sizeof(float);
    bytes += attn_stage_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    return bytes;
}

ds4_gpu_tensor *metal_graph_alloc_kv_cache_tensor_on(
        bool managed,
        int tier,
        uint64_t bytes) {
    (void)tier;
    return managed ? ds4_gpu_tensor_alloc_managed_on(tier, bytes)
                   : ds4_gpu_tensor_alloc_ptr_on(tier, bytes);
}

const metal_graph_debug_config *metal_graph_debug_get_config(void) {
    static metal_graph_debug_config cfg;
    if (!cfg.init) {
        cfg.init = 1;
        cfg.prefix = ds4_graph_env_value("DS4_METAL_GRAPH_DUMP_PREFIX");
        if (cfg.prefix && !cfg.prefix[0]) cfg.prefix = NULL;
        cfg.name = ds4_graph_env_value("DS4_METAL_GRAPH_DUMP_NAME");
        if (cfg.name && !cfg.name[0]) cfg.name = NULL;

        const char *layer_env =
            ds4_graph_env_value("DS4_METAL_GRAPH_DUMP_LAYER");
        if (layer_env && layer_env[0] && strcmp(layer_env, "all") != 0) {
            cfg.layer_set = 1;
            cfg.layer = (uint32_t)strtoul(layer_env, NULL, 10);
        }

        const char *pos_env = ds4_graph_env_value("DS4_METAL_GRAPH_DUMP_POS");
        if (pos_env && pos_env[0]) {
            cfg.pos_set = 1;
            cfg.pos = (uint32_t)strtoul(pos_env, NULL, 10);
        }
    }
    return &cfg;
}

static const char *metal_graph_debug_prefix_for(const char *name, uint32_t il, uint32_t pos) {
    const metal_graph_debug_config *cfg = metal_graph_debug_get_config();
    if (!cfg->prefix) return NULL;
    if (cfg->name && strstr(cfg->name, name) == NULL) return NULL;
    if (cfg->layer_set && cfg->layer != il) return NULL;
    if (cfg->pos_set && cfg->pos != pos) return NULL;
    return cfg->prefix;
}

bool metal_graph_debug_wants(const char *name, uint32_t il, uint32_t pos) {
    return metal_graph_debug_prefix_for(name, il, pos) != NULL;
}

void metal_graph_debug_dump_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = metal_graph_debug_prefix_for(name, il, pos);
    if (ds4_graph_env_present("DS4_METAL_GRAPH_DUMP_TRACE"))
        fprintf(stderr, "ds4: dump? name=%s il=%u pos=%u t=%p n=%llu wants=%d\n",
                name, il, pos, (void *)t, (unsigned long long)n_f32,
                metal_graph_debug_wants(name, il, pos));
    if (!t || n_f32 == 0 || !metal_graph_debug_wants(name, il, pos)) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    float *buf = xmalloc((size_t)n_f32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_f32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, buf, n_f32)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume CUDA command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

void metal_graph_debug_dump_f16_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f16,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = ds4_graph_env_value("DS4_METAL_GRAPH_DUMP_PREFIX");
    if (!t || n_f16 == 0 || !metal_graph_debug_wants(name, il, pos)) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    uint16_t *hbuf = xmalloc((size_t)n_f16 * sizeof(hbuf[0]));
    float *fbuf = xmalloc((size_t)n_f16 * sizeof(fbuf[0]));
    if (ds4_gpu_tensor_read(t, 0, hbuf, n_f16 * sizeof(hbuf[0])) != 0) {
        for (uint64_t i = 0; i < n_f16; i++) fbuf[i] = f16_to_f32(hbuf[i]);
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.bin", prefix, name, il, pos);
        if (write_f32_binary_file(path, fbuf, n_f16)) {
            fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
        }
    }
    free(fbuf);
    free(hbuf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume CUDA command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

void metal_graph_debug_dump_i32_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_i32,
        uint32_t          il,
        uint32_t          pos) {
    if (!t || n_i32 == 0) return;
    const char *prefix = metal_graph_debug_prefix_for(name, il, pos);
    if (!prefix) return;

    if (ds4_gpu_synchronize() == 0) {
        fprintf(stderr, "ds4: failed to synchronize before dumping %s layer %u pos %u\n", name, il, pos);
        return;
    }

    int32_t *buf = xmalloc((size_t)n_i32 * sizeof(buf[0]));
    if (ds4_gpu_tensor_read(t, 0, buf, n_i32 * sizeof(buf[0])) != 0) {
        char path[1024];
        snprintf(path, sizeof(path), "%s_%s-%u_pos%u.i32", prefix, name, il, pos);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            if (fwrite(buf, sizeof(buf[0]), (size_t)n_i32, fp) == (size_t)n_i32) {
                fprintf(stderr, "ds4: dumped %s layer %u pos %u to %s\n", name, il, pos, path);
            }
            fclose(fp);
        }
    }
    free(buf);

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr, "ds4: failed to resume CUDA command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

bool metal_graph_needs_ffn_out(const ds4_gpu_graph *g, uint32_t il, uint32_t pos) {
    return metal_graph_directional_steering_ffn_enabled(g) ||
           g->materialize_ffn_out ||
           metal_graph_debug_wants("ffn_out", il, pos);
}

/* tier-aware lazy allocator. The Class P ffn_out scratch is
 * created on demand the first time a layer that materializes ffn_out runs
 * on a tier; subsequent visits to the same tier reuse the existing slot.
 * Single-tier paths: active_tier == 0 always, behavior unchanged. */
bool metal_graph_ensure_ffn_out(ds4_gpu_graph *g) {
    const int t = g->active_tier;
    if (!g->ffn_out_by_tier[t]) {
        g->ffn_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(
                t, (uint64_t)DS4_N_EMBD * sizeof(float));
    }
    return g->ffn_out_by_tier[t] != NULL;
}

static bool metal_graph_ensure_batch_ffn_out_on(ds4_gpu_graph *g, int t) {
    if (t < 0 || t >= DS4_MAX_GPUS) return false;
    if (!g->batch_ffn_out_by_tier[t]) {
        g->batch_ffn_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(
                t, (uint64_t)g->prefill_cap * DS4_N_EMBD * sizeof(float));
    }
    return g->batch_ffn_out_by_tier[t] != NULL;
}

bool metal_graph_ensure_batch_ffn_out(ds4_gpu_graph *g) {
    return metal_graph_ensure_batch_ffn_out_on(g, g->active_tier);
}

static bool metal_graph_tp_env_flag(const char *name, bool dflt) {
    const char *env = getenv(name);
    if (!env || !env[0]) return dflt;
    return strcmp(env, "0") != 0;
}

bool metal_graph_cuda_greedy_splitkv_requested(void) {
    const char *no = getenv("DS4_CUDA_NO_GREEDY_SPLITKV");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_SPLITKV", false);
}

bool metal_graph_cuda_greedy_vec4_requested(void) {
    const char *no = getenv("DS4_CUDA_NO_GREEDY_VEC4");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_VEC4", false);
}

bool metal_graph_cuda_splitkv_spec_requested(void) {
    const char *no = getenv("DS4_CUDA_NO_SPLITKV_SPEC");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_SPLITKV_SPEC", false);
}

bool metal_graph_cuda_splitkv_spec_batch_verify_requested(void) {
    const char *no = getenv("DS4_CUDA_NO_SPLITKV_SPEC_BATCH_VERIFY");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_SPLITKV_SPEC_BATCH_VERIFY", false);
}

static float metal_graph_cuda_greedy_vec4_margin_threshold(void) {
    const char *env = getenv("DS4_CUDA_GREEDY_VEC4_MARGIN");
    if (env && env[0]) {
        char *end = NULL;
        double v = strtod(env, &end);
        while (end && isspace((unsigned char)*end)) end++;
        if (end != env && end && *end == '\0' && isfinite(v) && v >= 0.0) {
            return (float)v;
        }
        fprintf(stderr,
                "ds4: invalid DS4_CUDA_GREEDY_VEC4_MARGIN=%s; using 0.25\n",
                env);
    }
    return 0.25f;
}

bool metal_graph_cuda_greedy_vec4_fallback_requested(void) {
    const char *no = getenv("DS4_CUDA_NO_GREEDY_VEC4_FALLBACK");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_cuda_greedy_vec4_margin_threshold() > 0.0f;
}

static float metal_graph_cuda_greedy_splitkv_margin_threshold(void) {
    const char *env = getenv("DS4_CUDA_GREEDY_SPLITKV_MARGIN");
    if (env && env[0]) {
        char *end = NULL;
        double v = strtod(env, &end);
        while (end && isspace((unsigned char)*end)) end++;
        if (end != env && end && *end == '\0' && isfinite(v) && v >= 0.0) {
            return (float)v;
        }
        fprintf(stderr,
                "ds4: invalid DS4_CUDA_GREEDY_SPLITKV_MARGIN=%s; using 0.25\n",
                env);
    }
    return 0.25f;
}

bool metal_graph_cuda_greedy_splitkv_fallback_requested(void) {
    const char *no = getenv("DS4_CUDA_NO_GREEDY_SPLITKV_FALLBACK");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_cuda_greedy_splitkv_margin_threshold() > 0.0f;
}

bool metal_graph_cuda_q_norm_rope_fuse_requested(void) {
    return metal_graph_tp_env_flag("DS4_CUDA_Q_NORM_ROPE_FUSE", true);
}

bool metal_graph_cuda_qkv_kv_rope_fuse_requested(void) {
    const char *no = getenv("DS4_CUDA_NO_QKV_KV_ROPE_FUSE");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    if (getenv("DS4_CUDA_DISABLE_QKV_RMS_FUSED") != NULL) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_QKV_KV_ROPE_FUSE", true);
}
