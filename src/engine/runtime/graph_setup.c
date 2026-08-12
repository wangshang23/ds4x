#include "engine_internal.h"

/* Graph Setup module. */
#ifndef DS4_NO_GPU
/* Allocate the complete target runtime on the single GB10. */
bool metal_graph_alloc_raw_cap(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_dspark_verify) {
    memset(g, 0, sizeof(*g));
    g->runtime = ds4x_default_runtime_context();
    g->owns_prefill_workspace = true;
    g->active_tier = 0;
    g->emb_tier = 0;
    g->head_tier = 0;
    g->dspark_exec_tier = 0;
    g->cuda_q_norm_rope_fuse = metal_graph_cuda_q_norm_rope_fuse_requested();
    g->cuda_qkv_kv_rope_fuse = metal_graph_cuda_qkv_kv_rope_fuse_requested();
    g->cuda_qkv_pair = getenv("DS4_CUDA_NO_QKV_PAIR") == NULL;
    g->shared_gate_up_swiglu_fuse =
        getenv("DS4_CUDA_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION") == NULL;
    g->decode_stage_profile = getenv("DS4_CUDA_DECODE_STAGE_PROFILE") != NULL;
    g->decode_index_stage_profile = getenv("DS4_CUDA_INDEXER_STAGE_PROFILE") != NULL;
    g->output_stage_profile = getenv("DS4_CUDA_OUTPUT_STAGE_PROFILE") != NULL;
    const bool enable_splitkv_spec = metal_graph_cuda_splitkv_spec_requested();
    const bool enable_splitkv_batch_verify =
        enable_splitkv_spec && metal_graph_cuda_splitkv_spec_batch_verify_requested();
    const bool enable_spec_logits = enable_dspark_verify || enable_splitkv_batch_verify;
    const bool enable_prefix1_snapshot = enable_dspark_verify || enable_splitkv_spec;
    const bool enable_frontier_snapshot =
        enable_dspark_verify ||
        enable_splitkv_spec ||
        (metal_graph_cuda_greedy_splitkv_requested() &&
         metal_graph_cuda_greedy_splitkv_fallback_requested()) ||
        (metal_graph_cuda_greedy_vec4_requested() &&
         metal_graph_cuda_greedy_vec4_fallback_requested());
    g->mtp_enabled = enable_dspark_verify;
    if (raw_cap == 0) raw_cap = 1;
    if (ctx_size == 0) ctx_size = raw_cap;
    if (prefill_cap == 0) prefill_cap = 1;
    uint32_t raw_window = DS4_N_SWA;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (raw_cap < raw_window) raw_cap = raw_window;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    g->raw_cap = raw_cap;
    g->raw_window = raw_window;
    g->prefill_cap = prefill_cap;
    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!weights_layer_has_required(&weights->layer[il], il)) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx_size ? ctx_size : 1u;
    g->comp_cap = ctx_size / min_ratio + 2u;
    if (g->comp_cap < 2u) g->comp_cap = 2u;
    g->attn_comp_stage_cap = prefill_cap / min_ratio + 2u;
    if (g->attn_comp_stage_cap < 2u) g->attn_comp_stage_cap = 2u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!weights_layer_has_required(&weights->layer[il], il)) {
            g->layer_comp_cap[il] = 0;
            continue;
        }
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) {
            g->layer_comp_cap[il] = 0;
        } else {
            g->layer_comp_cap[il] = ctx_size / ratio + 2u;
            if (g->layer_comp_cap[il] < 2u) g->layer_comp_cap[il] = 2u;
        }
    }

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;
    const uint64_t group_dim = (uint64_t)DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP);
    const uint64_t shared_dim = layer->ffn_gate_shexp->dim[1];
    const uint64_t routed_mid_dim = layer->ffn_gate_exps->dim[1];
    /* Distributed coordinators do not normally own the output head. The
     * logits workspace still has a fixed model-vocabulary shape, while the
     * actual head is encoded only on a node that bound its tensors. */
    const uint64_t vocab_dim =
        weights->output ? weights->output->dim[1] : DS4_N_VOCAB;
    const uint64_t comp_width_max = 2ull * (DS4_N_HEAD_DIM > DS4_N_INDEXER_HEAD_DIM
        ? DS4_N_HEAD_DIM
        : DS4_N_INDEXER_HEAD_DIM);
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    const uint64_t pc = prefill_cap;
    uint64_t kv_cache_bytes = 0;
    const uint64_t context_bytes =
        metal_graph_context_bytes_for_kv_policy(ctx_size, raw_cap, prefill_cap, &kv_cache_bytes);
    const bool managed_kv_cache =
        ds4_gpu_should_use_managed_kv_cache(kv_cache_bytes, context_bytes) != 0;
    if (managed_kv_cache) {
        /*
         * CUDA device allocations are fastest, but a million-token KV cache is
         * large enough to starve DGX Spark's unified CPU/GPU memory once the
         * model cache and driver allocations are present.  For this one
         * long-lived cache class, managed memory restores the old demand-paged
         * behavior.  It can be slower, but it keeps oversized contexts from
         * turning memory pressure into a machine-wide lockup.
         */
        fprintf(stderr,
                "ds4: CUDA using managed KV cache for ctx=%u "
                "(kv cache %.2f GiB, context buffers %.2f GiB); "
                "this may degrade performance but is needed for very large contexts\n",
                ctx_size,
                (double)kv_cache_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0);
    }

    /* Decode scratch lives on the only supported device. */
    for (int t = 0; t < 1; t++) {
        g->cur_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, hc_dim * sizeof(float));
        g->flat_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, hc_dim * sizeof(float));
        g->hc_mix_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, mix_hc * sizeof(float));
        g->hc_split_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, mix_hc * sizeof(float));
        g->hc_pre_by_tier[t] = ds4_gpu_tensor_view(g->hc_split_by_tier[t],
                                                    0,
                                                    (uint64_t)DS4_N_HC * sizeof(float));
        g->hc_post_by_tier[t] = ds4_gpu_tensor_view(g->hc_split_by_tier[t],
                                                     (uint64_t)DS4_N_HC * sizeof(float),
                                                     (uint64_t)DS4_N_HC * sizeof(float));
        g->hc_comb_by_tier[t] = ds4_gpu_tensor_view(g->hc_split_by_tier[t],
                                                     2ull * DS4_N_HC * sizeof(float),
                                                     (uint64_t)DS4_N_HC * DS4_N_HC * sizeof(float));
        g->attn_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->attn_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->qr_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, q_rank * sizeof(float));
        g->qr_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, q_rank * sizeof(float));
        g->q_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, q_dim * sizeof(float));
        g->kv_raw_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
        g->kv_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    }
    bool state_init_ok = true;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!weights_layer_has_required(&weights->layer[il], il)) continue;
        const int layer_tier = 0;
        g->layer_raw_cache[il] = metal_graph_alloc_kv_cache_tensor_on(
                managed_kv_cache,
                layer_tier,
                (uint64_t)raw_cap * DS4_SPARK_KV_ROW_BYTES);
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t attn_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            const uint64_t attn_rows = (uint64_t)coff * ratio;
            g->layer_attn_comp_cache[il] = metal_graph_alloc_kv_cache_tensor_on(
                    managed_kv_cache,
                    layer_tier,
                    (uint64_t)g->layer_comp_cap[il] * DS4_SPARK_KV_ROW_BYTES);
            g->layer_attn_state_kv[il] = ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
            g->layer_attn_state_score[il] = ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
            if (enable_frontier_snapshot) {
                g->spec_attn_state_kv[il] =
                    ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
                g->spec_attn_state_score[il] =
                    ds4_gpu_tensor_alloc_ptr_on(layer_tier, attn_width * attn_rows * sizeof(float));
                if (enable_prefix1_snapshot) {
                    g->spec_prefix1_attn_state_kv[il] =
                        ds4_gpu_tensor_alloc_ptr_on(
                                layer_tier,
                                DS4_SPEC_PREFIX_SLOTS * attn_width * attn_rows *
                                    sizeof(float));
                    g->spec_prefix1_attn_state_score[il] =
                        ds4_gpu_tensor_alloc_ptr_on(
                                layer_tier,
                                DS4_SPEC_PREFIX_SLOTS * attn_width * attn_rows *
                                    sizeof(float));
                }
            }
            if (g->layer_attn_state_kv[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f, attn_width * attn_rows);
            }
            if (g->layer_attn_state_score[il]) {
                state_init_ok = state_init_ok &&
                                metal_tensor_fill_f32(g->layer_attn_state_score[il], DS4_NEG_INF, attn_width * attn_rows);
            }

            if (ratio == 4) {
                const uint64_t index_width = (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM;
                const uint64_t index_rows = (uint64_t)coff * ratio;
                g->layer_index_comp_cache[il] = metal_graph_alloc_kv_cache_tensor_on(
                        managed_kv_cache,
                        layer_tier,
                        (uint64_t)g->layer_comp_cap[il] * DS4_SPARK_INDEX_ROW_BYTES);
                g->layer_index_state_kv[il] = ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                g->layer_index_state_score[il] = ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                if (enable_frontier_snapshot) {
                    g->spec_index_state_kv[il] =
                        ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                    g->spec_index_state_score[il] =
                        ds4_gpu_tensor_alloc_ptr_on(layer_tier, index_width * index_rows * sizeof(float));
                    if (enable_prefix1_snapshot) {
                        g->spec_prefix1_index_state_kv[il] =
                            ds4_gpu_tensor_alloc_ptr_on(
                                    layer_tier,
                                    DS4_SPEC_PREFIX_SLOTS * index_width * index_rows *
                                        sizeof(float));
                        g->spec_prefix1_index_state_score[il] =
                            ds4_gpu_tensor_alloc_ptr_on(
                                    layer_tier,
                                    DS4_SPEC_PREFIX_SLOTS * index_width * index_rows *
                                        sizeof(float));
                    }
                }
                if (g->layer_index_state_kv[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f, index_width * index_rows);
                }
                if (g->layer_index_state_score[il]) {
                    state_init_ok = state_init_ok &&
                                    metal_tensor_fill_f32(g->layer_index_state_score[il], DS4_NEG_INF, index_width * index_rows);
                }
            }
        }
    }
    /* Per-layer decode scratch and routed-expert state. */
    for (int t = 0; t < 1; t++) {
        g->comp_kv_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, comp_width_max * sizeof(float));
        g->comp_sc_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, comp_width_max * sizeof(float));
        /* Decode-only scratch for the quad compressor projection: the indexer
         * pair outputs must not alias the attention compressor outputs inside
         * the single fused dispatch. */
        g->index_comp_kv_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, 2ull * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
        g->index_comp_sc_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, 2ull * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
        g->attn_comp_stage_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)g->attn_comp_stage_cap * DS4_N_HEAD_DIM * sizeof(float));
        g->index_comp_stage_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)g->attn_comp_stage_cap * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
        g->indexer_q_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, indexer_q_dim * sizeof(float));
        g->indexer_weights_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_INDEXER_HEAD * sizeof(float));
        g->indexer_scores_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)g->comp_cap * pc * sizeof(float));
        g->comp_mask_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)g->comp_cap * pc * sizeof(float));
        g->comp_selected_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)(DS4_N_INDEXER_TOP_K ? DS4_N_INDEXER_TOP_K : 1u) * pc * sizeof(uint32_t));
        g->heads_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, q_dim * sizeof(float));
        g->attn_low_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, low_dim * sizeof(float));
        g->attn_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->after_attn_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, hc_dim * sizeof(float));
        g->ffn_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->ffn_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->shared_gate_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, shared_dim * sizeof(float));
        g->shared_up_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, shared_dim * sizeof(float));
        g->shared_mid_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, shared_dim * sizeof(float));
        g->shared_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->router_logits_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, DS4_N_EXPERT * sizeof(float));
        g->router_probs_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, DS4_N_EXPERT * sizeof(float));
        g->router_selected_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, DS4_N_EXPERT_USED * sizeof(int));
        g->router_weights_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, DS4_N_EXPERT_USED * sizeof(float));
        g->routed_gate_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
        g->routed_up_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
        g->routed_mid_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
        g->routed_down_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t,
                (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
        g->routed_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, (uint64_t)DS4_N_EMBD * sizeof(float));
        g->after_ffn_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, hc_dim * sizeof(float));
    }
    uint64_t output_logits_elems = vocab_dim;
    g->output_pre_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier, (uint64_t)DS4_N_HC * sizeof(float));
    g->output_weights_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier, (uint64_t)DS4_N_HC * sizeof(float));
    g->output_embd_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier, (uint64_t)DS4_N_EMBD * sizeof(float));
    g->output_norm_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier, (uint64_t)DS4_N_EMBD * sizeof(float));
    g->logits_by_tier[g->head_tier] =
        ds4_gpu_tensor_alloc_ptr_on(g->head_tier,
                                    output_logits_elems * sizeof(float));
    if (enable_spec_logits) {
        g->spec_logits = ds4_gpu_tensor_alloc(
                (uint64_t)16 * DS4_N_VOCAB * sizeof(float));
    }

    /* Chunked-prefill scratch for one active request. */
    g->prefill_tokens_by_tier[0] =
        ds4_gpu_tensor_alloc_ptr_on(0, pc * sizeof(int32_t));
    for (int t = 0; t < 1; t++) {
            g->batch_cur_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * hc_dim * sizeof(float));
            g->batch_next_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * hc_dim * sizeof(float));
            g->batch_flat_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * hc_dim * sizeof(float));
            g->batch_hc_mix_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * mix_hc * sizeof(float));
            g->batch_hc_split_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * mix_hc * sizeof(float));
            g->batch_attn_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_attn_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_qr_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * q_rank * sizeof(float));
            g->batch_qr_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * q_rank * sizeof(float));
            g->batch_q_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * q_dim * sizeof(float));
            g->batch_kv_raw_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_HEAD_DIM * sizeof(float));
            g->batch_kv_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_HEAD_DIM * sizeof(float));
            g->batch_comp_kv_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * comp_width_max * sizeof(float));
            g->batch_comp_sc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * comp_width_max * sizeof(float));
            g->batch_indexer_q_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * indexer_q_dim * sizeof(float));
            g->batch_indexer_weights_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_INDEXER_HEAD * sizeof(float));
            g->batch_heads_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * q_dim * sizeof(float));
            g->batch_attn_low_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * low_dim * sizeof(float));
            g->batch_attn_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_group_tmp_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * group_dim * sizeof(float));
            g->batch_low_tmp_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_LORA_O * sizeof(float));
            g->batch_after_attn_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * hc_dim * sizeof(float));
            g->batch_ffn_cur_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_ffn_norm_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_shared_gate_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * shared_dim * sizeof(float));
            g->batch_shared_up_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * shared_dim * sizeof(float));
            g->batch_shared_mid_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * shared_dim * sizeof(float));
            g->batch_shared_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
            g->batch_router_logits_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT * sizeof(float));
            g->batch_router_probs_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT * sizeof(float));
            g->batch_router_selected_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * sizeof(int));
            g->batch_router_weights_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * sizeof(float));
            g->batch_routed_gate_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
            g->batch_routed_up_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
            g->batch_routed_mid_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * routed_mid_dim * sizeof(float));
            g->batch_routed_down_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float));
            g->batch_routed_out_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, pc * DS4_N_EMBD * sizeof(float));
    }
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        g->batch_q_half = ds4_gpu_tensor_alloc(pc * q_dim * sizeof(uint16_t));
    }

    bool layer_cache_ok = true;
    for (uint32_t il = 0; layer_cache_ok && il < DS4_N_LAYER; il++) {
        if (!weights_layer_has_required(&weights->layer[il], il)) continue;
        layer_cache_ok = g->layer_raw_cache[il] != NULL;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (layer_cache_ok && ratio != 0) {
            layer_cache_ok = g->layer_attn_comp_cache[il] != NULL &&
                             g->layer_attn_state_kv[il] != NULL &&
                             g->layer_attn_state_score[il] != NULL &&
                             (!enable_frontier_snapshot ||
                              (g->spec_attn_state_kv[il] != NULL &&
                               g->spec_attn_state_score[il] != NULL)) &&
                             (!enable_prefix1_snapshot ||
                              (g->spec_prefix1_attn_state_kv[il] != NULL &&
                               g->spec_prefix1_attn_state_score[il] != NULL));
        }
        if (layer_cache_ok && ratio == 4) {
            layer_cache_ok = g->layer_index_comp_cache[il] != NULL &&
                             g->layer_index_state_kv[il] != NULL &&
                             g->layer_index_state_score[il] != NULL &&
                             (!enable_frontier_snapshot ||
                              (g->spec_index_state_kv[il] != NULL &&
                               g->spec_index_state_score[il] != NULL)) &&
                             (!enable_prefix1_snapshot ||
                              (g->spec_prefix1_index_state_kv[il] != NULL &&
                               g->spec_prefix1_index_state_score[il] != NULL));
        }
    }

    bool class_p_ok = true;
    for (int t = 0; class_p_ok && t < 1; t++) {
        class_p_ok =
            g->cur_hc_by_tier[t] && g->flat_hc_by_tier[t] && g->hc_mix_by_tier[t] && g->hc_split_by_tier[t] &&
            g->hc_pre_by_tier[t] && g->hc_post_by_tier[t] && g->hc_comb_by_tier[t] &&
            g->attn_cur_by_tier[t] && g->attn_norm_by_tier[t] && g->qr_by_tier[t] && g->qr_norm_by_tier[t] &&
            g->q_by_tier[t] && g->kv_raw_by_tier[t] && g->kv_by_tier[t] &&
            g->comp_kv_cur_by_tier[t] && g->comp_sc_cur_by_tier[t] &&
            g->index_comp_kv_cur_by_tier[t] && g->index_comp_sc_cur_by_tier[t] &&
            g->attn_comp_stage_by_tier[t] && g->index_comp_stage_by_tier[t] &&
            g->indexer_q_by_tier[t] && g->indexer_weights_by_tier[t] && g->indexer_scores_by_tier[t] &&
            g->comp_mask_by_tier[t] && g->comp_selected_by_tier[t] &&
            g->heads_by_tier[t] && g->attn_low_by_tier[t] && g->attn_out_by_tier[t] &&
            g->after_attn_hc_by_tier[t] && g->ffn_cur_by_tier[t] && g->ffn_norm_by_tier[t] &&
            g->shared_gate_by_tier[t] && g->shared_up_by_tier[t] && g->shared_mid_by_tier[t] &&
            g->shared_out_by_tier[t] &&
            g->router_logits_by_tier[t] && g->router_probs_by_tier[t] &&
            g->router_selected_by_tier[t] && g->router_weights_by_tier[t] &&
            g->routed_gate_by_tier[t] && g->routed_up_by_tier[t] && g->routed_mid_by_tier[t] &&
            g->routed_down_by_tier[t] && g->routed_out_by_tier[t] &&
            g->after_ffn_hc_by_tier[t] &&
            g->batch_cur_hc_by_tier[t] && g->batch_next_hc_by_tier[t] && g->batch_flat_hc_by_tier[t] &&
            g->batch_hc_mix_by_tier[t] && g->batch_hc_split_by_tier[t] &&
            g->batch_attn_cur_by_tier[t] && g->batch_attn_norm_by_tier[t] &&
            g->batch_qr_by_tier[t] && g->batch_qr_norm_by_tier[t] && g->batch_q_by_tier[t] &&
            g->batch_kv_raw_by_tier[t] && g->batch_kv_by_tier[t] &&
            g->batch_comp_kv_by_tier[t] && g->batch_comp_sc_by_tier[t] &&
            g->batch_indexer_q_by_tier[t] && g->batch_indexer_weights_by_tier[t] &&
            g->batch_heads_by_tier[t] && g->batch_attn_low_by_tier[t] && g->batch_attn_out_by_tier[t] &&
            g->batch_group_tmp_by_tier[t] && g->batch_low_tmp_by_tier[t] && g->batch_after_attn_hc_by_tier[t] &&
            g->batch_ffn_cur_by_tier[t] && g->batch_ffn_norm_by_tier[t] &&
            g->batch_shared_gate_by_tier[t] && g->batch_shared_up_by_tier[t] &&
            g->batch_shared_mid_by_tier[t] && g->batch_shared_out_by_tier[t] &&
            g->batch_router_logits_by_tier[t] && g->batch_router_probs_by_tier[t] &&
            g->batch_router_selected_by_tier[t] && g->batch_router_weights_by_tier[t] &&
            g->batch_routed_gate_by_tier[t] && g->batch_routed_up_by_tier[t] &&
            g->batch_routed_mid_by_tier[t] && g->batch_routed_down_by_tier[t] &&
            g->batch_routed_out_by_tier[t];
    }
    const bool ok = state_init_ok && layer_cache_ok && class_p_ok &&
                    metal_graph_output_pre(g) && metal_graph_output_weights(g) &&
                    metal_graph_output_embd(g) && metal_graph_output_norm(g) &&
                    metal_graph_logits(g) &&
                    (!enable_spec_logits || g->spec_logits) &&
                    metal_graph_prefill_tokens(g) &&
                    (!DS4_GPU_ATTN_COMP_CACHE_F16 || g->batch_q_half);
    if (!ok) metal_graph_free(g);
    return ok;
}

#endif /* !DS4_NO_GPU */
