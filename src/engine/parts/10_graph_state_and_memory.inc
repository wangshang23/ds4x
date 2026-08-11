#ifndef DS4_NO_GPU
/* This repository has one persistent-cache ABI: the native GB10 packed
 * representation. F32 is retained only for short-lived compressor staging. */
#define DS4_GPU_RAW_CACHE_SPARK 1
#define DS4_GPU_ATTN_COMP_CACHE_SPARK 1
#define DS4_GPU_INDEX_COMP_CACHE_SPARK 1
#define DS4_GPU_ATTN_COMP_CACHE_F16 0

#define DS4_GPU_GLM_COMPACT_CACHE_F16 DS4_GPU_ATTN_COMP_CACHE_F16

/* =========================================================================
 * Metal Release Graph State.
 * =========================================================================
 *
 * The release Metal executor owns one fixed set of tensors for single-token
 * decode and another for batched prefill.  The structure is DS4-specific:
 * tensor names follow the model stages rather than generic graph nodes.
 */

enum { DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS = 64 };

typedef struct {
    /* Class P — per-tier replicated kernel scratch buffers.
     * Each used tier has its own copy; active_tier names the slot the
     * current dispatch step reads/writes. Single-tier paths leave
     * active_tier == 0; multi-tier dispatch updates active_tier in B6.
     *
     * Decode hidden-state buffers. A generated token enters as an embedding
     * in cur_hc and leaves as logits after all 43 layers update their
     * raw/compressed/indexer caches. The hc_pre / hc_post / hc_comb views
     * are derived from hc_split per tier (see metal_graph_alloc_raw_cap). */
    ds4_gpu_tensor *cur_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *flat_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *hc_mix_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *hc_split_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *hc_pre_by_tier[DS4_MAX_GPUS];   /* views of hc_split */
    ds4_gpu_tensor *hc_post_by_tier[DS4_MAX_GPUS];  /* views of hc_split */
    ds4_gpu_tensor *hc_comb_by_tier[DS4_MAX_GPUS];  /* views of hc_split */
    ds4_gpu_tensor *attn_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *attn_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *qr_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *qr_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *q_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *kv_raw_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *kv_by_tier[DS4_MAX_GPUS];
    int active_tier;
    /* cached engine placement[] (length DS4_N_LAYER + 2) for the
     * dispatch loops. NULL in single-tier mode — active_tier stays 0 and
     * dispatch wrappers no-op the tier-switch + cross-device copy. The
     * pointer aliases e->placement; the engine outlives the graph so this
     * is safe. */
    const int *placement;

    /* Persistent KV state.  Raw KV is a sliding-window ring per layer.  Ratio-4
     * layers also keep an indexer-compressed cache; ratio-128 layers keep only
     * the attention-compressed cache.  The small state tensors are compressor
     * frontiers for the next compressed row, so they must be snapshotted with
     * the row counters whenever a checkpoint is saved or partially rewound. */
    ds4_gpu_tensor *layer_raw_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_comp_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_comp_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_raw_cache_tp[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_comp_cache_tp[DS4_MAX_LAYER];

    /* Speculative decoding scratch.  MTP is allowed to mutate graph state only
     * if the target verifier can either commit it or restore the saved
     * frontiers.  Prefix slots retain the intermediate frontiers of a tiny
     * verifier block so partial accepts do not need target-model replay. */
    ds4_gpu_tensor *spec_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_index_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_index_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_logits;
    uint32_t layer_n_comp[DS4_MAX_LAYER];
    uint32_t layer_n_index_comp[DS4_MAX_LAYER];
    uint32_t spec_prefix_n_comp[DS4_SPEC_PREFIX_SLOTS][DS4_MAX_LAYER];
    uint32_t spec_prefix_n_index_comp[DS4_SPEC_PREFIX_SLOTS][DS4_MAX_LAYER];
    bool spec_capture_prefixes;
    uint32_t raw_cap;
    /* Maximum compressed-row capacity across layers.  Shared work buffers use
     * this worst-case size because ratio-4 indexer layers can still reach it. */
    uint32_t comp_cap;
    /* Persistent compressed caches are per layer, so size them from the actual
     * layer compression ratio instead of pessimistically using the ratio-4 cap
     * for every ratio-128 layer. */
    uint32_t layer_comp_cap[DS4_MAX_LAYER];
    uint32_t attn_comp_stage_cap;

    /* Class P (per-layer work tensors). Each used tier has its
     * own replica. They are reused in place by every layer instead of
     * allocating a generic graph arena. This is why the code is verbose but
     * predictable: each pointer names an actual DS4 stage. */
    ds4_gpu_tensor *comp_kv_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *comp_sc_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *index_comp_kv_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *index_comp_sc_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *attn_comp_stage_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *index_comp_stage_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *indexer_q_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *indexer_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *indexer_scores_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *comp_mask_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *comp_selected_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *heads_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *attn_low_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *attn_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *after_attn_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *ffn_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *ffn_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *shared_gate_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *shared_up_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *shared_mid_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *shared_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *router_logits_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *router_probs_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *router_selected_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *router_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_gate_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_up_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_mid_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_down_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *ffn_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *after_ffn_hc_by_tier[DS4_MAX_GPUS];
    /* Class H — output-head buffers and logits live on the
     * head tier only. head_tier is captured at metal_graph_alloc_raw_cap
     * time from placement[DS4_N_LAYER + 1] (or 0 in single-tier /
     * diagnostic paths). Non-head slots remain NULL. Readers go through
     * the metal_graph_logits / metal_graph_output_* accessors below. */
    ds4_gpu_tensor *output_pre_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *output_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *output_embd_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *output_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *logits_by_tier[DS4_MAX_GPUS];
    int head_tier;

    /* DSpark target features.  The proposer consumes mean-over-HC rows from
     * selected target layers; keeping them on-GPU avoids adding readbacks to
     * the target path. */
    ds4_gpu_tensor *dspark_hc_mean_weights;
    ds4_gpu_tensor *dspark_hc_mean_rows;
    ds4_gpu_tensor *dspark_target_hidden;
    ds4_gpu_tensor *dspark_target_hidden_batch;
    ds4_gpu_tensor *dspark_stage0_packed;
    ds4_gpu_tensor *dspark_stage0_proj;
    ds4_gpu_tensor *dspark_main_x;
    ds4_gpu_tensor *dspark_draft_tokens;
    ds4_gpu_tensor *dspark_draft_hc;
    ds4_gpu_tensor *dspark_target_hc;
    ds4_gpu_tensor *dspark_stage_input_hc;
    ds4_gpu_tensor *dspark_stage_output_hc;
    ds4_gpu_tensor *dspark_position_ids;
    ds4_gpu_tensor *dspark_raw_cache[DS4_DSPARK_MAX_STAGES];
    uint32_t dspark_cache_cap;
    uint32_t dspark_cache_start;
    uint32_t dspark_cache_token_start;
    uint32_t dspark_cache_len;
    uint32_t dspark_target_layer_count;
    uint32_t dspark_block_size;
    uint32_t dspark_target_layers[DS4_DSPARK_MAX_TARGET_LAYERS];
    uint32_t dspark_capture_mask;
    uint32_t dspark_capture_checkpoint_len;
    uint32_t dspark_capture_batch_mask;
    uint32_t dspark_capture_batch_start;
    uint32_t dspark_capture_batch_tokens;
    bool dspark_capture_valid;
    bool dspark_capture_batch_valid;
    int dspark_exec_tier;
    bool dspark_capture_enabled;
    bool verify_small_batch_tp;
    uint32_t pipeline_capture_chunk_start;
    uint32_t pipeline_capture_chunk_len;
    bool ssd_streaming; /* glm-branch SSD streaming; always false here */

    /* Optional MTP model state.  It has its own raw cache because the drafter
     * runs on speculative future tokens; target KV state is updated only after
     * verification accepts draft tokens. */
    ds4_gpu_tensor *mtp_embed;
    ds4_gpu_tensor *mtp_enorm;
    ds4_gpu_tensor *mtp_eproj;
    ds4_gpu_tensor *mtp_eproj_hc;
    ds4_gpu_tensor *mtp_hnorm_hc;
    ds4_gpu_tensor *mtp_hproj_hc;
    ds4_gpu_tensor *mtp_input_hc;
    ds4_gpu_tensor *mtp_state_hc;
    ds4_gpu_tensor *mtp_next_hc;
    ds4_gpu_tensor *mtp_raw_cache;
    uint32_t mtp_n_raw;
    uint32_t prefill_cap;
    uint32_t raw_window;
    uint32_t batch_token_offset;

    /* Batched prefill tensors.  Prefill is layer-major: a chunk of prompt
     * tokens moves through layer 0, then layer 1, and so on, updating the same
     * persistent caches used by decode.  Keeping this separate from decode
     * avoids a slow loop of one-token graph steps for long prompts. */
    /* Class E — embedding-tier-only prompt-token integer buffer.
     * Captured at metal_graph_alloc_raw_cap time from placement[0] (or 0 in
     * single-tier / diagnostic paths). Non-embedding slots stay NULL. Readers
     * go through metal_graph_prefill_tokens() below. */
    ds4_gpu_tensor *prefill_tokens_by_tier[DS4_MAX_GPUS];
    int emb_tier;
    /* Class P batch (chunked-prefill) scratch — per-tier
     * replicated. The cur/next pair is ping-ponged per layer step on the
     * layer's active tier; tier transitions copy the active buffer across
     * boundaries via ds4_gpu_tensor_copy_xdev (handled in B6). */
    ds4_gpu_tensor *batch_cur_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_next_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_flat_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_hc_mix_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_hc_split_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_attn_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_attn_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_qr_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_qr_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_q_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_kv_raw_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_kv_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_comp_kv_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_comp_sc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_indexer_q_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_indexer_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_heads_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_attn_low_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_attn_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_group_tmp_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_low_tmp_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_after_attn_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_ffn_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_ffn_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_shared_gate_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_shared_up_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_shared_mid_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_shared_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_router_logits_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_router_probs_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_router_selected_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_router_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_gate_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_up_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_mid_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_down_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_out_by_tier[DS4_MAX_GPUS];
    bool batch_routed_mid_is_f16;
    ds4_gpu_tensor *batch_ffn_out_by_tier[DS4_MAX_GPUS];
    bool owns_prefill_workspace;
    bool materialize_ffn_out;
    /* Class P (replicated per tier — this is
     * consumed in per-layer attn/FFN kernels, NOT embedding-only). Read-only
     * after init; replicate by writing the same host directions buffer to
     * every used tier's slot during session setup. */
    ds4_gpu_tensor *directional_steering_dirs_by_tier[DS4_MAX_GPUS];
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    bool cuda_tp_decode;
    bool cuda_tp_attn;
    bool cuda_tp_attn_peer_read;
    bool cuda_tp_attn_heads;
    bool cuda_tp_attn_cache_dup;
    bool cuda_tp_moe;
    bool cuda_tp_ep;
    bool cuda_tp_ep_pack_exact;
    bool cuda_tp_moe_delay_reduce;
    bool cuda_tp_moe_copy3_handoff;
    bool cuda_tp_moe_pack_handoff;
    bool cuda_tp_moe_peer_read;
    bool cuda_tp_moe_peer_router;
    bool cuda_tp_shared;
    bool cuda_tp_shared_fold;
    bool cuda_tp_q;
    bool cuda_tp_output;
    bool cuda_tp_prefill_ffn;
    bool cuda_tp_prefill_attn_output;
    bool cuda_q_norm_rope_fuse;
    bool cuda_qkv_kv_rope_fuse;
    bool cuda_qkv_pair;
    bool cuda_tp_attn_out_hc_fuse;
    bool shared_gate_up_swiglu_fuse;
    bool decode_stage_profile;
    bool decode_index_stage_profile;
    bool output_stage_profile;
    ds4_gpu_tensor *tp_peer_tmp_by_tier[DS4_MAX_GPUS];
    uint32_t power_percent;
    double prefill_layer_avg_sec[DS4_MAX_LAYER];
    double decode_token_avg_sec;
    bool quality;
    bool mtp_enabled;
    /* Metal-only prefill helpers retained alongside the CUDA tiered workspace. */
    ds4_gpu_tensor *batch_q_half;
    ds4_gpu_tensor *prefill_seed_router_selected;
    uint32_t prefill_seed_tokens;
    uint64_t prefill_selected_profile_rows;
    uint64_t prefill_selected_profile_unique;
    uint64_t prefill_selected_profile_selected_bytes;
    uint64_t prefill_selected_profile_full_bytes;
    uint32_t prefill_selected_profile_layers;
    uint32_t prefill_selected_profile_min_unique;
    uint32_t prefill_selected_profile_max_unique;
    uint32_t streaming_preload_experts;
    bool ssd_streaming_cold;
    bool streaming_static_decode_map_current;
    float *cpu_router_norm;

    /* Metal network tensor parallelism. These views alias engine-owned
     * transport slabs except tp_logits_half, whose view object is session-owned. */
    uint32_t tp_world;
    uint32_t tp_rank;
    ds4_gpu_tensor **tp_out;
    ds4_gpu_tensor **tp_in;
    ds4_gpu_tensor **tp_batch_out;
    ds4_gpu_tensor **tp_batch_in;
    uint32_t tp_batch_rows;
    ds4_gpu_tensor *tp_zero;
    ds4_gpu_tensor *tp_logits_half;
} ds4_gpu_graph;

/* Tensors that are temporary for chunked prefill and grouped multi-session
 * decode. The batched server serializes every operation that uses them, so one
 * engine-owned set can be aliased by all resident session graphs. */
#define DS4_GPU_PREFILL_WORKSPACE_FIELDS(X) \
    X(prefill_tokens)                       \
    X(batch_ffn_out)                        \
    X(batch_routed_out)                     \
    X(batch_routed_down)                    \
    X(batch_routed_mid)                     \
    X(batch_routed_up)                      \
    X(batch_routed_gate)                    \
    X(batch_router_weights)                 \
    X(batch_router_selected)                \
    X(batch_router_probs)                   \
    X(batch_router_logits)                  \
    X(batch_shared_out)                     \
    X(batch_shared_mid)                     \
    X(batch_shared_up)                      \
    X(batch_shared_gate)                    \
    X(batch_ffn_norm)                       \
    X(batch_ffn_cur)                        \
    X(batch_after_attn_hc)                  \
    X(batch_low_tmp)                        \
    X(batch_group_tmp)                      \
    X(batch_attn_out)                       \
    X(batch_attn_low)                       \
    X(batch_heads)                          \
    X(batch_indexer_weights)                \
    X(batch_indexer_q)                      \
    X(batch_comp_sc)                        \
    X(batch_comp_kv)                        \
    X(batch_kv)                             \
    X(batch_kv_raw)                         \
    X(batch_q)                              \
    X(batch_qr_norm)                        \
    X(batch_qr)                             \
    X(batch_attn_norm)                      \
    X(batch_attn_cur)                       \
    X(batch_hc_split)                       \
    X(batch_hc_mix)                         \
    X(batch_flat_hc)                        \
    X(batch_next_hc)                        \
    X(batch_cur_hc)

/* Class H accessors. All reader sites for the output-head
 * tensors and the final logits route through these inlines, which read the
 * head_tier slot captured at allocation time. Single-tier paths set
 * head_tier == 0 and the slot is byte-identical to the legacy
 * metal_graph_logits(g) / g->output_* pointers. Multi-tier paths set head_tier
 * to placement[DS4_N_LAYER + 1]; other tier slots remain NULL. */
static inline ds4_gpu_tensor *metal_graph_logits(const ds4_gpu_graph *g) {
    return g->logits_by_tier[g->head_tier];
}
static inline ds4_gpu_tensor *metal_graph_output_pre(const ds4_gpu_graph *g) {
    return g->output_pre_by_tier[g->head_tier];
}
static inline ds4_gpu_tensor *metal_graph_output_weights(const ds4_gpu_graph *g) {
    return g->output_weights_by_tier[g->head_tier];
}
static inline ds4_gpu_tensor *metal_graph_output_embd(const ds4_gpu_graph *g) {
    return g->output_embd_by_tier[g->head_tier];
}
static inline ds4_gpu_tensor *metal_graph_output_norm(const ds4_gpu_graph *g) {
    return g->output_norm_by_tier[g->head_tier];
}

/* Class E accessor. The prompt-token integer buffer is
 * consumed by the embedding kernel on the embedding tier only. Single-tier
 * paths set emb_tier == 0 (byte-equivalent to the legacy single-tier
 * pointer). Multi-tier paths set emb_tier = placement[0]. */
static inline ds4_gpu_tensor *metal_graph_prefill_tokens(const ds4_gpu_graph *g) {
    return g->prefill_tokens_by_tier[g->emb_tier];
}

/* Class P accessors. Each Class P kernel-scratch buffer is
 * replicated across every tier the placement uses; the active_tier field
 * names the slot the current dispatch step reads/writes. Single-tier paths
 * leave active_tier == 0 (byte-equivalent to the legacy single-tier
 * pointer). Multi-tier dispatch (wired up in B6) updates active_tier with
 * the current layer's home tier before each kernel-dispatch wrapper runs. */
#define DS4_GPU_GRAPH_CLASS_P_ACCESSOR(name) \
static inline ds4_gpu_tensor *metal_graph_##name(const ds4_gpu_graph *g) { \
    return g->name##_by_tier[g->active_tier]; \
}

DS4_GPU_GRAPH_CLASS_P_ACCESSOR(cur_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(flat_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_mix)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_split)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_pre)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_post)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_comb)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(qr)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(qr_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(q)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(kv_raw)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(kv)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(comp_kv_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(comp_sc_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(index_comp_kv_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(index_comp_sc_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_comp_stage)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(index_comp_stage)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(indexer_q)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(indexer_weights)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(indexer_scores)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(comp_mask)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(comp_selected)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(heads)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_low)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(after_attn_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(ffn_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(ffn_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(shared_gate)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(shared_up)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(shared_mid)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(shared_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(router_logits)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(router_probs)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(router_selected)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(router_weights)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_gate)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_up)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_mid)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_down)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(ffn_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(after_ffn_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_cur_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_next_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_flat_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_hc_mix)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_hc_split)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_attn_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_attn_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_qr)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_qr_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_q)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_kv_raw)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_kv)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_comp_kv)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_comp_sc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_indexer_q)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_indexer_weights)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_heads)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_attn_low)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_attn_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_group_tmp)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_low_tmp)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_after_attn_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_ffn_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_ffn_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_shared_gate)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_shared_up)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_shared_mid)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_shared_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_router_logits)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_router_probs)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_router_selected)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_router_weights)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_gate)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_up)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_mid)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_down)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_ffn_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(directional_steering_dirs)

/* dispatch-loop helpers for multi-tier per-layer execution.
 *
 * Single-tier (g->placement == NULL): all helpers are no-ops; active_tier
 * stays 0 from memset; behavior is byte-equivalent to legacy.
 *
 * Multi-tier: each helper switches g->active_tier to the requested tier
 * BEFORE the next kernel-dispatch wrapper reads any Class P accessor. If
 * the source-tier Class P cur_hc (or batch_cur_hc) differs from the new
 * tier's, ds4_gpu_tensor_copy_xdev ferries the active hidden state across
 * the boundary. copy_xdev returns 1 on success, 0 on failure. The
 * destination tensor's device_id was stamped at alloc_on time and is
 * immutable.
 *
 * For decode (one token at a time): metal_graph_set_active_tier_decode
 * swaps to the requested tier and copies cur_hc across the boundary.
 *
 * For batch (chunked prefill): metal_graph_set_active_tier_batch swaps
 * tier and copies batch_cur_hc across the boundary. The next/cur pair
 * is maintained per tier — after a copy, batch_next_hc on the destination
 * tier becomes the swap target for the next layer step on that tier.
 *
 * Helpers always invoke ds4_gpu_set_current_device(tier) so the next
 * kernel-launch sees the correct CUDA device. */

/* ds4_gpu_set_current_device is declared in ds4_gpu_mgpu.h — single-tier
 * (g_n_gpus <= 1) callers no-op. Returns 0 on success. */

#ifdef DS4_NO_GPU
/* CPU-only builds do not include ds4_gpu.h: mirror the decode-graph key
 * type and stub the capture API (always eager). */
typedef struct ds4_decode_graph_key {
    uint32_t il;
    uint32_t island;
    uint32_t variant;
    uint32_t _pad;
    void    *cur_hc;
    void    *after_attn_hc;
    void    *after_ffn_hc;
    void    *attn_norm;
} ds4_decode_graph_key;
static inline int ds4_gpu_decode_graphs_supported(void) { return 0; }
static inline int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key) { (void)key; return -1; }
static inline int ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key) { (void)key; return -1; }
static inline void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key) { (void)key; }
static inline void ds4_gpu_decode_graphs_invalidate(void) {}
static inline int ds4_gpu_set_current_device(int tier) { (void)tier; return 0; }
static inline int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst,
                                            const ds4_gpu_tensor *src,
                                            uint64_t bytes) {
    (void)dst; (void)src; (void)bytes; return 1;
}
static inline int ds4_gpu_tensor_copy_xdev3(ds4_gpu_tensor       *dst0,
                                             const ds4_gpu_tensor *src0,
                                             uint64_t              bytes0,
                                             ds4_gpu_tensor       *dst1,
                                             const ds4_gpu_tensor *src1,
                                             uint64_t              bytes1,
                                             ds4_gpu_tensor       *dst2,
                                             const ds4_gpu_tensor *src2,
                                             uint64_t              bytes2) {
    (void)dst0; (void)src0; (void)bytes0;
    (void)dst1; (void)src1; (void)bytes1;
    (void)dst2; (void)src2; (void)bytes2;
    return 1;
}
static inline int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *dst,
                                                    const ds4_gpu_tensor *src,
                                                    uint64_t bytes) {
    (void)dst; (void)src; (void)bytes; return 1;
}
static inline int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *src, int dst_tier) {
    (void)src; (void)dst_tier; return 1;
}
static inline int ds4_gpu_moe_handoff_pack_tensor(
        ds4_gpu_tensor       *packed,
        const ds4_gpu_tensor *ffn_norm,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_embd,
        uint32_t              n_expert) {
    (void)packed; (void)ffn_norm; (void)selected; (void)weights;
    (void)n_embd; (void)n_expert; return 1;
}
static inline int ds4_gpu_q8_cache_suppressed(void) { return 0; }
static inline void ds4_gpu_set_q8_cache_suppressed(int suppressed) { (void)suppressed; }
static inline int ds4_gpu_set_decode_fast_attention(int enabled) {
    (void)enabled;
    return 0;
}
static inline int ds4_gpu_set_decode_score_vec4(int enabled) {
    (void)enabled;
    return 0;
}
static inline int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        uint32_t                kv_n_head,
        uint32_t                kv_head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        bool                    inverse,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   eps) {
    (void)q_out; (void)q; (void)model_map; (void)model_size;
    (void)q_weight_offset; (void)q_n; (void)kv_out; (void)kv;
    (void)kv_weight_offset; (void)kv_n; (void)rows; (void)kv_n_head;
    (void)kv_head_dim; (void)n_rot; (void)pos0; (void)n_ctx_orig;
    (void)inverse; (void)freq_base; (void)freq_scale; (void)ext_factor;
    (void)attn_factor; (void)beta_fast; (void)beta_slow; (void)eps;
    return 0;
}
#endif

/* Returns true on success. Single-tier: no-op success. Multi-tier:
 * sets the CUDA device, then if tier differs from current active_tier,
 * copies cur_hc to the destination tier and updates active_tier. */
static bool metal_graph_set_active_tier_decode(ds4_gpu_graph *g, int tier) {
    if (!g->placement) {
        /* Single-tier: just keep active_tier at 0; no device switch needed. */
        (void)tier;
        return true;
    }
    if (tier < 0 || tier >= DS4_MAX_GPUS) return false;
    if (tier == g->active_tier) return true;
    if (ds4_gpu_set_current_device(tier) != 0) return false;
    /* Boundary hop: copy cur_hc from source-tier to destination-tier slot. */
    if (g->active_tier >= 0) {
        ds4_gpu_tensor *src = g->cur_hc_by_tier[g->active_tier];
        ds4_gpu_tensor *dst = g->cur_hc_by_tier[tier];
        if (src && dst) {
            const uint64_t hc_bytes = (uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float);
            if (!ds4_gpu_tensor_copy_xdev(dst, src, hc_bytes)) return false;
        }
    }
    g->active_tier = tier;
    return true;
}

/* Returns true on success. Same semantics as the decode helper but ferries
 * batch_cur_hc (which contains chunk_tokens * hc_dim floats — variable per
 * prefill call). The caller passes the chunk size in tokens; single-tier
 * paths ignore the argument. */
static bool metal_graph_set_active_tier_batch(ds4_gpu_graph *g, int tier, uint32_t chunk_tokens) {
    if (!g->placement) {
        (void)tier;
        (void)chunk_tokens;
        return true;
    }
    if (tier < 0 || tier >= DS4_MAX_GPUS) return false;
    if (tier == g->active_tier) return true;
    if (ds4_gpu_set_current_device(tier) != 0) return false;
    if (g->active_tier >= 0) {
        ds4_gpu_tensor *src = g->batch_cur_hc_by_tier[g->active_tier];
        ds4_gpu_tensor *dst = g->batch_cur_hc_by_tier[tier];
        if (src && dst) {
            const uint64_t hc_bytes =
                (uint64_t)chunk_tokens * DS4_N_HC * DS4_N_EMBD * sizeof(float);
            if (!ds4_gpu_tensor_copy_xdev(dst, src, hc_bytes)) return false;
        }
    }
    g->active_tier = tier;
    return true;
}

static bool metal_graph_set_active_tier_no_copy(ds4_gpu_graph *g, int tier) {
    if (!g->placement) {
        (void)tier;
        return true;
    }
    if (tier < 0 || tier >= DS4_MAX_GPUS) return false;
    if (tier == g->active_tier) return true;
    if (ds4_gpu_set_current_device(tier) != 0) return false;
    g->active_tier = tier;
    return true;
}

/* Upstream: --power N GPU duty-cycle throttling helpers. The single-tier
 * --power=100 path is a no-op; multi-tier inherits the same helpers via
 * graph_power_note_prefill_layer / graph_power_note_decode_token which we
 * call from the shared encode / decode loops. */

static bool graph_power_throttle_enabled(const ds4_gpu_graph *g) {
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

static void graph_power_note_prefill_layer(ds4_gpu_graph *g,
                                           uint32_t il,
                                           double elapsed_sec) {
    if (!graph_power_throttle_enabled(g)) return;
    if (il >= DS4_N_LAYER) return;
    g->prefill_layer_avg_sec[il] =
        graph_power_update_avg(g->prefill_layer_avg_sec[il], elapsed_sec);
    graph_power_sleep(g->prefill_layer_avg_sec[il], g->power_percent);
}

static void graph_power_note_decode_token(ds4_gpu_graph *g, double elapsed_sec) {
    if (!graph_power_throttle_enabled(g)) return;
    g->decode_token_avg_sec =
        graph_power_update_avg(g->decode_token_avg_sec, elapsed_sec);
    graph_power_sleep(g->decode_token_avg_sec, g->power_percent);
}

static void metal_graph_copy_prefill_workspace_pointers(
        ds4_gpu_graph       *dst,
        const ds4_gpu_graph *src) {
#define DS4_COPY_PREFILL_FIELD(name) \
    memcpy(dst->name##_by_tier, src->name##_by_tier, \
           sizeof(dst->name##_by_tier));
    DS4_GPU_PREFILL_WORKSPACE_FIELDS(DS4_COPY_PREFILL_FIELD)
#undef DS4_COPY_PREFILL_FIELD
    dst->batch_q_half = src->batch_q_half;
    dst->prefill_seed_router_selected = src->prefill_seed_router_selected;
}

static void metal_graph_transfer_prefill_workspace(
        ds4_gpu_graph *dst,
        ds4_gpu_graph *src) {
    memset(dst, 0, sizeof(*dst));
    dst->prefill_cap = src->prefill_cap;
    dst->emb_tier = src->emb_tier;
    dst->owns_prefill_workspace = true;
    metal_graph_copy_prefill_workspace_pointers(dst, src);
    src->owns_prefill_workspace = false;
}

static uint64_t metal_graph_prefill_workspace_bytes(const ds4_gpu_graph *g) {
    uint64_t total = 0;
    for (int t = 0; t < DS4_MAX_GPUS; t++) {
#define DS4_COUNT_PREFILL_FIELD(name) \
        total += ds4_gpu_tensor_bytes(g->name##_by_tier[t]);
        DS4_GPU_PREFILL_WORKSPACE_FIELDS(DS4_COUNT_PREFILL_FIELD)
#undef DS4_COUNT_PREFILL_FIELD
    }
    total += ds4_gpu_tensor_bytes(g->batch_q_half);
    total += ds4_gpu_tensor_bytes(g->prefill_seed_router_selected);
    return total;
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
    ds4_gpu_tensor_free(g->prefill_seed_router_selected);
    g->batch_q_half = NULL;
    g->prefill_seed_router_selected = NULL;
    g->owns_prefill_workspace = false;
}

/* Release every Metal tensor owned by the whole-model graph runtime. */
static void metal_graph_free(ds4_gpu_graph *g) {
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
    ds4_gpu_tensor_free(g->mtp_raw_cache);
    ds4_gpu_tensor_free(g->mtp_next_hc);
    ds4_gpu_tensor_free(g->mtp_state_hc);
    ds4_gpu_tensor_free(g->mtp_input_hc);
    ds4_gpu_tensor_free(g->mtp_hproj_hc);
    ds4_gpu_tensor_free(g->mtp_hnorm_hc);
    ds4_gpu_tensor_free(g->mtp_eproj_hc);
    ds4_gpu_tensor_free(g->mtp_eproj);
    ds4_gpu_tensor_free(g->mtp_enorm);
    ds4_gpu_tensor_free(g->mtp_embed);
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
        ds4_gpu_tensor_free(g->tp_peer_tmp_by_tier[t]);
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
        ds4_gpu_tensor_free(g->layer_raw_cache_tp[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_comp_cache[il]);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_attn_comp_cache_tp[il]);
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
    ds4_gpu_tensor_free(g->tp_logits_half);
    free(g->cpu_router_norm);
    memset(g, 0, sizeof(*g));
}

static bool metal_tensor_fill_f32(ds4_gpu_tensor *t, float v, uint64_t n) {
    return ds4_gpu_tensor_fill_f32(t, v, n) != 0;
}

/* =========================================================================
 * Directional Steering.
 * =========================================================================
 *
 * A steering file contains one normalized 4096-wide direction per layer.  When
 * enabled, the Metal graph edits selected block outputs in-place:
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
static bool metal_graph_load_directional_steering(
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

static bool metal_graph_directional_steering_attn_enabled(const ds4_gpu_graph *g) {
    return g && metal_graph_directional_steering_dirs(g) &&
           g->directional_steering_attn_scale != 0.0f;
}

static bool metal_graph_directional_steering_ffn_enabled(const ds4_gpu_graph *g) {
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

static bool metal_graph_apply_directional_steering_attn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return metal_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_attn_scale : 0.0f);
}

static bool metal_graph_apply_directional_steering_ffn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows) {
    return metal_graph_apply_directional_steering(g, x, il, rows, g ? g->directional_steering_ffn_scale : 0.0f);
}

static bool metal_graph_configure_dspark_capture(
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

static uint64_t metal_graph_context_bytes_for_kv_policy(
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

static ds4_gpu_tensor *metal_graph_alloc_kv_cache_tensor_on(
        bool managed,
        int tier,
        uint64_t bytes) {
    if (g_n_gpus <= 1) {
        return managed ? ds4_gpu_tensor_alloc_managed(bytes)
                       : ds4_gpu_tensor_alloc(bytes);
    }
    return managed ? ds4_gpu_tensor_alloc_managed_on(tier, bytes)
                   : ds4_gpu_tensor_alloc_ptr_on(tier, bytes);
}

static ds4_gpu_tensor *metal_graph_alloc_kv_cache_tensor(bool managed, uint64_t bytes) {
    return metal_graph_alloc_kv_cache_tensor_on(managed, 0, bytes);
}

/* =========================================================================
 * Metal Diagnostic Dump Hooks.
 * =========================================================================
 *
 * The release path calls these after important stages, but they are no-ops
 * unless DS4_METAL_GRAPH_DUMP_PREFIX or DS4_ROCM_GRAPH_DUMP_PREFIX is set.
 * Dumping synchronizes and restarts the command batch, so it is intentionally
 * isolated here.
 */

typedef struct {
    int init;
    const char *prefix;
    const char *name;
    int layer_set;
    uint32_t layer;
    int pos_set;
    uint32_t pos;
} metal_graph_debug_config;

static const metal_graph_debug_config *metal_graph_debug_get_config(void) {
    static metal_graph_debug_config cfg;
    if (!cfg.init) {
        cfg.init = 1;
        cfg.prefix = glm_graph_env_value("DS4_ROCM_GRAPH_DUMP_PREFIX",
                                         "DS4_METAL_GRAPH_DUMP_PREFIX");
        if (cfg.prefix && !cfg.prefix[0]) cfg.prefix = NULL;
        cfg.name = glm_graph_env_value("DS4_ROCM_GRAPH_DUMP_NAME",
                                       "DS4_METAL_GRAPH_DUMP_NAME");
        if (cfg.name && !cfg.name[0]) cfg.name = NULL;

        const char *layer_env = glm_graph_env_value("DS4_ROCM_GRAPH_DUMP_LAYER",
                                                    "DS4_METAL_GRAPH_DUMP_LAYER");
        if (layer_env && layer_env[0] && strcmp(layer_env, "all") != 0) {
            cfg.layer_set = 1;
            cfg.layer = (uint32_t)strtoul(layer_env, NULL, 10);
        }

        const char *pos_env = glm_graph_env_value("DS4_ROCM_GRAPH_DUMP_POS",
                                                  "DS4_METAL_GRAPH_DUMP_POS");
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

static bool metal_graph_debug_wants(const char *name, uint32_t il, uint32_t pos) {
    return metal_graph_debug_prefix_for(name, il, pos) != NULL;
}

static void metal_graph_debug_dump_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = metal_graph_debug_prefix_for(name, il, pos);
    if (glm_graph_env_present("DS4_ROCM_GRAPH_DUMP_TRACE",
                              "DS4_METAL_GRAPH_DUMP_TRACE"))
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
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static void metal_graph_debug_dump_f16_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f16,
        uint32_t          il,
        uint32_t          pos) {
    const char *prefix = glm_graph_env_value("DS4_ROCM_GRAPH_DUMP_PREFIX",
                                             "DS4_METAL_GRAPH_DUMP_PREFIX");
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
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static void metal_graph_debug_dump_i32_tensor(
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
        fprintf(stderr, "ds4: failed to resume Metal command batch after dumping %s layer %u pos %u\n", name, il, pos);
    }
}

static bool metal_graph_needs_ffn_out(const ds4_gpu_graph *g, uint32_t il, uint32_t pos) {
    return metal_graph_directional_steering_ffn_enabled(g) ||
           g->materialize_ffn_out ||
           metal_graph_debug_wants("ffn_out", il, pos);
}

/* tier-aware lazy allocator. The Class P ffn_out scratch is
 * created on demand the first time a layer that materializes ffn_out runs
 * on a tier; subsequent visits to the same tier reuse the existing slot.
 * Single-tier paths: active_tier == 0 always, behavior unchanged. */
static bool metal_graph_ensure_ffn_out(ds4_gpu_graph *g) {
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

static bool metal_graph_ensure_batch_ffn_out(ds4_gpu_graph *g) {
    return metal_graph_ensure_batch_ffn_out_on(g, g->active_tier);
}

static bool metal_graph_tp_env_flag(const char *name, bool dflt) {
    const char *env = getenv(name);
    if (!env || !env[0]) return dflt;
    return strcmp(env, "0") != 0;
}

static bool metal_graph_cuda_tp_attn_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_ATTN", true);
#endif
}

static bool metal_graph_cuda_tp_attn_peer_read_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_ATTN_PEER_READ", true);
#endif
}

static bool metal_graph_cuda_tp_attn_heads_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_ATTN_HEADS", false);
#endif
}

static bool metal_graph_cuda_tp_attn_cache_dup_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_ATTN_CACHE_DUP", false);
#endif
}

static bool metal_graph_cuda_tp_moe_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_MOE", true);
#endif
}

static bool metal_graph_cuda_tp_ep_pack_exact_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_EP_PACK_EXACT", true);
#endif
}

static bool metal_graph_cuda_tp_ep_direct_return_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_EP_DIRECT_RETURN", true);
#endif
}

static bool metal_graph_cuda_tp_ep_delay_reduce_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_EP_DELAY_REDUCE", true);
#endif
}

static bool metal_graph_cuda_tp_ep_fused_hc_reduce_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_EP_FUSED_HC_REDUCE", true);
#endif
}

static DS4_MAYBE_UNUSED bool metal_graph_cuda_tp_ep_fused_shared_mid_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_EP_FUSED_SHARED_MID", true);
#endif
}

static DS4_MAYBE_UNUSED bool metal_graph_cuda_tp_ep_balanced_shared_mid_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag(
            "DS4_CUDA_TP_EP_BALANCED_SHARED_MID", true);
#endif
}

static DS4_MAYBE_UNUSED bool metal_graph_cuda_tp_ep_dual_prequant_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag(
            "DS4_CUDA_TP_EP_DUAL_PREQUANT", true);
#endif
}

static bool metal_graph_cuda_tp_moe_delay_reduce_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_MOE_DELAY_REDUCE", true);
#endif
}

static bool metal_graph_cuda_tp_moe_pack_handoff_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_MOE_PACK", false);
#endif
}

static bool metal_graph_cuda_tp_moe_copy3_handoff_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_MOE_COPY3", false);
#endif
}

static bool metal_graph_cuda_tp_moe_peer_read_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_MOE_PEER_READ", false);
#endif
}

static bool metal_graph_cuda_tp_moe_peer_router_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_MOE_PEER_ROUTER", false);
#endif
}

static bool metal_graph_cuda_tp_shared_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_SHARED", false);
#endif
}

static bool metal_graph_cuda_tp_shared_fold_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_SHARED_FOLD", true);
#endif
}

static bool metal_graph_cuda_tp_q_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_Q", false);
#endif
}

static bool metal_graph_cuda_tp_output_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_OUTPUT", true);
#endif
}

static bool metal_graph_cuda_greedy_split_top1_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_SPLIT_TOP1");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_SPLIT_TOP1", false);
#endif
}

static bool metal_graph_cuda_output_fused_top1_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_OUTPUT_FUSED_TOP1", false);
#endif
}

static bool metal_graph_cuda_verify_decode2_split_top1_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_VERIFY_DECODE2_SPLIT_TOP1");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_VERIFY_DECODE2_SPLIT_TOP1", false);
#endif
}

static bool metal_graph_cuda_greedy_splitkv_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_SPLITKV");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_SPLITKV", false);
#endif
}

static bool metal_graph_cuda_greedy_vec4_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_VEC4");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_VEC4", false);
#endif
}

static bool metal_graph_cuda_splitkv_spec_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_SPLITKV_SPEC");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_SPLITKV_SPEC", false);
#endif
}

static bool metal_graph_cuda_splitkv_spec_toponly_row0_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_SPLITKV_SPEC_TOPONLY_ROW0");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_SPLITKV_SPEC_TOPONLY_ROW0", false);
#endif
}

static bool metal_graph_cuda_splitkv_spec_batch_verify_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_SPLITKV_SPEC_BATCH_VERIFY");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_SPLITKV_SPEC_BATCH_VERIFY", false);
#endif
}

static float metal_graph_cuda_greedy_vec4_margin_threshold(void) {
#if defined(__APPLE__)
    return 0.0f;
#else
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
#endif
}

static bool metal_graph_cuda_greedy_vec4_fallback_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_VEC4_FALLBACK");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_cuda_greedy_vec4_margin_threshold() > 0.0f;
#endif
}

static float metal_graph_cuda_greedy_splitkv_margin_threshold(void) {
#if defined(__APPLE__)
    return 0.0f;
#else
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
#endif
}

static bool metal_graph_cuda_greedy_splitkv_fallback_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_SPLITKV_FALLBACK");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_cuda_greedy_splitkv_margin_threshold() > 0.0f;
#endif
}

static bool metal_graph_cuda_greedy_splitkv_top2_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_SPLITKV_TOP2");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_SPLITKV_TOP2", true);
#endif
}

static bool metal_graph_cuda_greedy_splitkv_trust_replay_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_SPLITKV_TRUST_REPLAY", false);
#endif
}

static bool metal_graph_cuda_greedy_splitkv_pair_replay_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_GREEDY_SPLITKV_PAIR_REPLAY");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_GREEDY_SPLITKV_PAIR_REPLAY", false);
#endif
}

static DS4_MAYBE_UNUSED uint32_t metal_graph_cuda_greedy_max_segment(const char *name) {
    const char *env = getenv(name);
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        while (end && isspace((unsigned char)*end)) end++;
        if (end != env && end && *end == '\0' && v <= INT32_MAX) {
            return (uint32_t)v;
        }
        fprintf(stderr,
                "ds4: invalid %s=%s; expected 0..%d, using disabled\n",
                name,
                env,
                INT32_MAX);
    }
    return 0;
}

static uint32_t metal_graph_cuda_greedy_splitkv_max_segment(void) {
#if defined(__APPLE__)
    return 0;
#else
    return metal_graph_cuda_greedy_max_segment("DS4_CUDA_GREEDY_SPLITKV_MAX_SEGMENT");
#endif
}

static uint32_t metal_graph_cuda_greedy_vec4_max_segment(void) {
#if defined(__APPLE__)
    return 0;
#else
    return metal_graph_cuda_greedy_max_segment("DS4_CUDA_GREEDY_VEC4_MAX_SEGMENT");
#endif
}

static uint32_t metal_graph_cuda_greedy_splitkv_min_score(void) {
    const char *env = getenv("DS4_CUDA_SPLITKV_MIN_SCORE");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        while (end && isspace((unsigned char)*end)) end++;
        if (end != env && end && *end == '\0' && v <= UINT32_MAX) {
            return (uint32_t)v;
        }
    }
    return metal_graph_tp_env_flag("DS4_CUDA_SPLITKV_DECODE", false) ? 0u : 512u;
}

static bool metal_graph_cuda_q_norm_rope_fuse_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_Q_NORM_ROPE_FUSE", true);
#endif
}

static bool metal_graph_cuda_qkv_kv_rope_fuse_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    const char *no = getenv("DS4_CUDA_NO_QKV_KV_ROPE_FUSE");
    if (no && no[0] && strcmp(no, "0") != 0) return false;
    if (getenv("DS4_CUDA_DISABLE_QKV_RMS_FUSED") != NULL) return false;
    return metal_graph_tp_env_flag("DS4_CUDA_QKV_KV_ROPE_FUSE", true);
#endif
}

static bool metal_graph_cuda_tp_prefill_ffn_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_PREFILL_FFN", true);
#endif
}

static bool metal_graph_cuda_tp_prefill_attn_output_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_TP_PREFILL_ATTN_OUTPUT", true);
#endif
}

static bool metal_graph_cuda_prefill_pipeline_requested(const ds4_gpu_graph *g) {
#if defined(__APPLE__)
    (void)g;
    return false;
#else
    const char *env = getenv("DS4_CUDA_PREFILL_PIPELINE");
    if (env && env[0]) return strcmp(env, "0") != 0;
    return g && g->cuda_tp_decode;
#endif
}

static bool metal_graph_cuda_prefill_pipeline_q8_cache_requested(void) {
#if defined(__APPLE__)
    return false;
#else
    return metal_graph_tp_env_flag("DS4_CUDA_PREFILL_PIPELINE_Q8_CACHE", false);
#endif
}

static uint32_t metal_graph_cuda_prefill_pipeline_microbatch(void) {
    const char *env = getenv("DS4_CUDA_PREFILL_PIPELINE_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && v > 0 && v <= UINT32_MAX) return (uint32_t)v;
    }
    return 512;
}

static int metal_graph_cuda_tp_partner_tier(int tier) {
    if (g_n_gpus < 2 || (g_n_gpus & 1) != 0) return -1;
    const int half = g_n_gpus / 2;
    if (tier < 0 || tier >= half) return -1;
    return tier + half;
}

static uint32_t metal_graph_cuda_tp_output_tiers(
        const ds4_gpu_graph *g,
        int                  tiers[DS4_MAX_GPUS]) {
    if (!g) return 0;
    return metal_graph_cuda_tp_output_tiers_for_head(g->head_tier,
                                                    g->cuda_tp_output,
                                                    g_n_gpus,
                                                    tiers);
}

static uint64_t metal_graph_q8_0_row_bytes(uint64_t in_dim) {
    return ((in_dim + 31u) / 32u) * 34u;
}

#endif /* !DS4_NO_GPU */
