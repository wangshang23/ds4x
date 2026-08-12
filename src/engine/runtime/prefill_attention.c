#include "engine_internal.h"

/* Prefill Attention module. */
#ifndef DS4_NO_GPU
bool metal_graph_encode_layer_attention_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (n_tokens == 0 || n_tokens > g->prefill_cap) return false;

    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_rank = layer->attn_q_a->dim[1];
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint32_t n_groups = DS4_N_OUT_GROUP;
    const uint32_t group_heads = DS4_N_HEAD / n_groups;
    const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
    const uint32_t rank = DS4_N_LORA_O;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const bool compressed = ratio != 0;
    const bool zero_prefix = pos0 == 0;
    const bool spark_packed_prefill_batch =
        g_n_gpus <= 1 &&
        getenv("DS4_CUDA_SPARK_PREFILL_REFERENCE") == NULL;
    const bool index_stage_profile =
        getenv("DS4_CUDA_INDEXER_STAGE_PROFILE") != NULL;
    const bool layer_stage_profile = metal_graph_layer_stage_profile_enabled(il);
    const bool q_stage_profile =
        getenv("DS4_CUDA_Q_STAGE_PROFILE") != NULL;
    double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
    double q_stage_t0 = q_stage_profile ? now_sec() : 0.0;
#define DS4_METAL_PROFILE_ATTN_STAGE(name) do { \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary("attn", (name), il, pos0, n_tokens, &layer_stage_t0); \
        } \
    } while (0)
#define DS4_METAL_PROFILE_Q_STAGE(name) do { \
        if (ok && q_stage_profile) { \
            ok = metal_graph_q_stage_profile_boundary((name), il, pos0, n_tokens, &q_stage_t0); \
        } \
    } while (0)
    const float freq_base = layer_rope_freq_base(il);
    const float freq_scale = layer_rope_freq_scale(il);
    const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
    float attn_factor = 1.0f;
    if (ext_factor != 0.0f && freq_scale > 0.0f) {
        attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    enum { stack_count_cap = 16 };
    uint32_t comp_counts_stack[stack_count_cap];
    uint32_t index_counts_stack[stack_count_cap];
    uint32_t *comp_counts = NULL;
    uint32_t *index_counts = NULL;
    if (compressed) {
        if (n_tokens <= stack_count_cap) {
            memset(comp_counts_stack, 0,
                   (size_t)n_tokens * sizeof(comp_counts_stack[0]));
            comp_counts = comp_counts_stack;
        } else {
            comp_counts = xcalloc(n_tokens, sizeof(comp_counts[0]));
        }
    }
    if (ratio == 4) {
        if (n_tokens <= stack_count_cap) {
            memset(index_counts_stack, 0,
                   (size_t)n_tokens * sizeof(index_counts_stack[0]));
            index_counts = index_counts_stack;
        } else {
            index_counts = xcalloc(n_tokens, sizeof(index_counts[0]));
        }
    }
    const bool qkv_rms_fused = !metal_graph_use_reference_qkv_norm();
    ds4_gpu_tensor *hc_mix_view = ds4_gpu_tensor_view(
            metal_graph_batch_hc_mix(g), 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *hc_split_view = ds4_gpu_tensor_view(
            metal_graph_batch_hc_split(g), 0, (uint64_t)n_tokens * mix_hc * sizeof(float));
    ds4_gpu_tensor *attn_cur_view = ds4_gpu_tensor_view(
            metal_graph_batch_attn_cur(g), 0, (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float));
    ds4_gpu_tensor *after_attn_hc_view = ds4_gpu_tensor_view(
            metal_graph_batch_after_attn_hc(g), 0, (uint64_t)n_tokens * hc_dim * sizeof(float));
    bool ok = hc_mix_view && hc_split_view && attn_cur_view && after_attn_hc_view;
    const bool fuse_hc_norm = n_tokens > 1 &&
                              DS4_N_HC == 4 &&
                              !metal_graph_use_reference_hc_decode() &&
                              metal_graph_enable_batch_hc_norm_fusion();
    if (ok) ok = metal_graph_hc_rms_scale_project(hc_mix_view,
                                                    metal_graph_batch_flat_hc(g),
                                                    model,
                                                    layer->hc_attn_fn,
                                                    metal_graph_batch_cur_hc(g),
                                                    hc_dim,
                                                    n_tokens);
    if (metal_graph_use_reference_hc_decode()) {
        if (ok) ok = ds4_gpu_hc_split_sinkhorn_tensor(hc_split_view,
                                                        hc_mix_view,
                                                        model->map,
                                                        model->size,
                                                        layer->hc_attn_scale->abs_offset,
                                                        layer->hc_attn_base->abs_offset,
                                                        DS4_N_HC,
                                                        DS4_N_HC_SINKHORN_ITER,
                                                        DS4_HC_EPS) != 0;
        if (ok) ok = ds4_gpu_hc_weighted_sum_split_tensor(attn_cur_view,
                                                            metal_graph_batch_cur_hc(g),
                                                            hc_split_view,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC) != 0;
    } else if (fuse_hc_norm) {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_norm_tensor(attn_cur_view,
                                                                 metal_graph_batch_attn_norm(g),
                                                                 hc_split_view,
                                                                 hc_mix_view,
                                                                 metal_graph_batch_cur_hc(g),
                                                                 model->map,
                                                                 model->size,
                                                                 layer->hc_attn_scale->abs_offset,
                                                                 layer->hc_attn_base->abs_offset,
                                                                 layer->attn_norm->abs_offset,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_HC,
                                                                 DS4_N_HC_SINKHORN_ITER,
                                                                 DS4_HC_EPS,
                                                                 DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_hc_split_weighted_sum_tensor(attn_cur_view,
                                                            hc_split_view,
                                                            hc_mix_view,
                                                            metal_graph_batch_cur_hc(g),
                                                            model->map,
                                                            model->size,
                                                            layer->hc_attn_scale->abs_offset,
                                                            layer->hc_attn_base->abs_offset,
                                                            DS4_N_EMBD,
                                                            DS4_N_HC,
                                                            DS4_N_HC_SINKHORN_ITER,
                                                            DS4_HC_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_pre", metal_graph_batch_attn_cur(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("hc_pre");
    if (ok && !fuse_hc_norm) {
        ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_attn_norm(g),
                                                  metal_graph_batch_attn_cur(g),
                                                  model->map,
                                                  model->size,
                                                  layer->attn_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  n_tokens,
                                                  DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("attn_norm", metal_graph_batch_attn_norm(g),
                                      (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("norm");
    DS4_METAL_PROFILE_Q_STAGE("pre_q");
    if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_q_a",
                                                      il,
                                                      pos0,
                                                      metal_graph_batch_qr(g),
                                                      model,
                                                      layer->attn_q_a,
                                                      DS4_N_EMBD,
                                                      q_rank,
                                                      metal_graph_batch_attn_norm(g),
                                                      n_tokens);
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora", metal_graph_batch_qr(g),
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("q_a");
    if (qkv_rms_fused) {
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_kv",
                                                          il,
                                                          pos0,
                                                          metal_graph_batch_kv_raw(g),
                                                          model,
                                                          layer->attn_kv,
                                                          DS4_N_EMBD,
                                                          DS4_N_HEAD_DIM,
                                                          metal_graph_batch_attn_norm(g),
                                                          n_tokens);
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", metal_graph_batch_kv_raw(g),
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
        if (ok) ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(metal_graph_batch_qr_norm(g),
                                                             metal_graph_batch_qr(g),
                                                             model->map,
                                                             model->size,
                                                             layer->attn_q_a_norm->abs_offset,
                                                             (uint32_t)q_rank,
                                                             metal_graph_batch_kv(g),
                                                             metal_graph_batch_kv_raw(g),
                                                             layer->attn_kv_a_norm->abs_offset,
                                                             DS4_N_HEAD_DIM,
                                                             n_tokens,
                                                             DS4_RMS_EPS) != 0;
    } else {
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_qr_norm(g),
                                                           metal_graph_batch_qr(g),
                                                           model->map,
                                                           model->size,
                                                           layer->attn_q_a_norm->abs_offset,
                                                           (uint32_t)q_rank,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("q_lora_norm", metal_graph_batch_qr_norm(g),
                                      (uint64_t)n_tokens * q_rank, il, pos0);
    }
    if (qkv_rms_fused && ok) {
        metal_graph_debug_dump_tensor("KVnorm", metal_graph_batch_kv(g),
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    DS4_METAL_PROFILE_Q_STAGE("q_a_norm");
    const bool q_path_debug =
        metal_graph_debug_wants("Qraw", il, pos0) ||
        metal_graph_debug_wants("Qnorm", il, pos0);
    bool q_b_f16_out = false;
    if (ok && !q_path_debug && layer->attn_q_b->type == DS4_TENSOR_Q8_0) {
        q_b_f16_out = ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(metal_graph_batch_q(g),
                                                                     g->batch_q_half,
                                                                     model->map,
                                                                     model->size,
                                                                     layer->attn_q_b->abs_offset,
                                                                     q_rank,
                                                                     q_dim,
                                                                     metal_graph_batch_qr_norm(g),
                                                                     n_tokens,
                                                                     DS4_N_HEAD,
                                                                     DS4_N_HEAD_DIM,
                                                                     DS4_N_ROT,
                                                                     pos0,
                                                                     compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                                     false,
                                                                     freq_base,
                                                                     freq_scale,
                                                                     ext_factor,
                                                                     attn_factor,
                                                                     DS4_ROPE_YARN_BETA_FAST,
                                                                     DS4_ROPE_YARN_BETA_SLOW,
                                                                     DS4_RMS_EPS) != 0;
    }
    if (q_b_f16_out) {
        DS4_METAL_PROFILE_Q_STAGE("q_b");
        DS4_METAL_PROFILE_Q_STAGE("head_norm");
        if (ok) {
            metal_graph_debug_dump_tensor("Qcur", metal_graph_batch_q(g),
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
        DS4_METAL_PROFILE_Q_STAGE("rope");
    } else {
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_q_b",
                                                          il,
                                                          pos0,
                                                          metal_graph_batch_q(g),
                                                          model,
                                                          layer->attn_q_b,
                                                          q_rank,
                                                          q_dim,
                                                          metal_graph_batch_qr_norm(g),
                                                          n_tokens);
        if (ok) {
            metal_graph_debug_dump_tensor("Qraw", metal_graph_batch_q(g),
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
        DS4_METAL_PROFILE_Q_STAGE("q_b");
        const bool q_norm_debug =
            metal_graph_debug_wants("Qnorm", il, pos0);
        bool q_norm_rope_fused = false;
        if (ok && !q_norm_debug) {
            q_norm_rope_fused = ds4_gpu_head_rms_norm_rope_tail_tensor(
                metal_graph_batch_q(g),
                n_tokens,
                DS4_N_HEAD,
                DS4_N_HEAD_DIM,
                DS4_N_ROT,
                pos0,
                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                false,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                DS4_ROPE_YARN_BETA_FAST,
                DS4_ROPE_YARN_BETA_SLOW,
                DS4_RMS_EPS) != 0;
        }
        if (ok && !q_norm_rope_fused) {
            ok = ds4_gpu_head_rms_norm_tensor(
                metal_graph_batch_q(g),
                n_tokens,
                DS4_N_HEAD,
                DS4_N_HEAD_DIM,
                DS4_RMS_EPS) != 0;
        }
        if (ok && !q_norm_rope_fused) {
            metal_graph_debug_dump_tensor("Qnorm", metal_graph_batch_q(g),
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
        DS4_METAL_PROFILE_Q_STAGE("head_norm");
        if (ok && !q_norm_rope_fused) {
            ok = ds4_gpu_rope_tail_tensor(
                metal_graph_batch_q(g),
                n_tokens,
                DS4_N_HEAD,
                DS4_N_HEAD_DIM,
                DS4_N_ROT,
                pos0,
                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                false,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                DS4_ROPE_YARN_BETA_FAST,
                DS4_ROPE_YARN_BETA_SLOW) != 0;
        }
        if (ok) {
            metal_graph_debug_dump_tensor("Qcur", metal_graph_batch_q(g),
                                          (uint64_t)n_tokens * q_dim, il, pos0);
        }
        DS4_METAL_PROFILE_Q_STAGE("rope");
    }
    DS4_METAL_PROFILE_ATTN_STAGE("q_path");
    if (!qkv_rms_fused) {
        if (ok) ok = metal_graph_matmul_q8_0_named_tensor("attn_kv",
                                                          il,
                                                          pos0,
                                                          metal_graph_batch_kv_raw(g),
                                                          model,
                                                          layer->attn_kv,
                                                          DS4_N_EMBD,
                                                          DS4_N_HEAD_DIM,
                                                          metal_graph_batch_attn_norm(g),
                                                          n_tokens);
        if (ok) {
            metal_graph_debug_dump_tensor("KVraw", metal_graph_batch_kv_raw(g),
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(metal_graph_batch_kv(g),
                                                           metal_graph_batch_kv_raw(g),
                                                           model->map,
                                                           model->size,
                                                           layer->attn_kv_a_norm->abs_offset,
                                                           DS4_N_HEAD_DIM,
                                                           n_tokens,
                                                           DS4_RMS_EPS) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("KVnorm", metal_graph_batch_kv(g),
                                          (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
        }
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_batch_kv(g),
                                            n_tokens,
                                            DS4_N_HEAD_KV,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            false,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("KVrope", metal_graph_batch_kv(g),
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    if (ok) ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(metal_graph_batch_kv(g),
                                                       n_tokens,
                                                       DS4_N_HEAD_DIM,
                                                       DS4_N_ROT) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("KVcur", metal_graph_batch_kv(g),
                                      (uint64_t)n_tokens * DS4_N_HEAD_DIM, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("kv_path");
    /*
     * Static graph order is q, kv, cpy_k(raw SWA), then attention. For a
     * zero-prefix batch it is safe to store the whole batch at once: attention
     * reads the contiguous batch KV, and the ring only has to end with the last
     * SWA rows for later chunks/decode. For nonzero chunks the physical ring is
     * sized to hold the current chunk plus the previous SWA window, while the
     * attention mask still enforces the 128-token logical window.
     */
    if (ok && zero_prefix) ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                                    metal_graph_batch_kv(g),
                                                                    g->raw_cap,
                                                                    pos0,
                                                                    n_tokens,
                                                                    DS4_N_HEAD_DIM) != 0;
    if (!ok) {
        fprintf(stderr, "ds4: gpu layer %u raw KV batch store failed\n", il);
    }
    const bool raw_batch_attention = zero_prefix && ratio == 0;
    bool batch_attention_done = false;

    if (ok && raw_batch_attention) {
        ok = ds4_gpu_attention_prefill_raw_heads_tensor(
                metal_graph_batch_heads(g), model->map, model->size,
                layer->attn_sinks->abs_offset, metal_graph_batch_q(g),
                metal_graph_batch_kv(g), n_tokens, g->raw_window,
                DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
        if (ok) batch_attention_done = true;
    } else if (ok && (!DS4_GPU_RAW_CACHE_SPARK || spark_packed_prefill_batch) &&
               !zero_prefix && ratio == 0 && n_tokens <= g->raw_cap) {
        /*
         * The ubatch path stores the whole batch in the SWA cache, then runs
         * one batched attention kernel with an absolute-position causal/window
         * mask.  This avoids mixing prefill with the different single-token
         * attention path.
         */
        const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos0, n_tokens);
        /* Nonzero prompt chunks read the SWA cache as a ring.  FlashAttention
         * receives a linearized window starting at raw_start, not physical row
         * zero; otherwise wrapped chunks silently miss recent raw keys. */
        const uint32_t raw_start = metal_graph_raw_start_for_span(g,
                                                                  pos0 + n_tokens - 1u,
                                                                  n_raw);
        ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                 metal_graph_batch_kv(g),
                                                 g->raw_cap,
                                                 pos0,
                                                 n_tokens,
                                                 DS4_N_HEAD_DIM) != 0;
        if (ok) {
            metal_graph_debug_dump_tensor("raw_cache",
                                          g->layer_raw_cache[il],
                                          (uint64_t)n_raw * DS4_N_HEAD_DIM,
                                          il,
                                          pos0);
        }
        if (ok) {
            ok = ds4_gpu_attention_decode_raw_batch_heads_tensor(metal_graph_batch_heads(g),
                                                                   model->map,
                                                                   model->size,
                                                                   layer->attn_sinks->abs_offset,
                                                                   metal_graph_batch_q(g),
                                                                   g->layer_raw_cache[il],
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   g->raw_cap,
                                                                   raw_start,
                                                                   g->raw_window,
                                                                   DS4_N_HEAD,
                                                                   DS4_N_HEAD_DIM) != 0;
        }
        if (ok) batch_attention_done = true;
    } else if (ok && ratio != 0) {
        const uint32_t coff = ratio == 4 ? 2u : 1u;
        const uint32_t comp_width = coff * DS4_N_HEAD_DIM;
        const bool have_attn_comp = layer->attn_compressor_kv && layer->attn_compressor_gate &&
                                    layer->attn_compressor_ape && layer->attn_compressor_norm;
        if (!have_attn_comp) {
            fprintf(stderr, "ds4: Metal layer-major prefill needs attention compressor weights\n");
            ok = false;
        }
        if (ok) {
            ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_comp_kv(g),
                                             model->map,
                                             model->size,
                                             layer->attn_compressor_kv->abs_offset,
                                             DS4_N_EMBD,
                                             comp_width,
                                             metal_graph_batch_attn_norm(g),
                                             n_tokens) != 0;
            if (!ok) {
                fprintf(stderr, "ds4: gpu layer %u attention compressor KV projection failed\n", il);
            }
            if (ok) {
                ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_comp_sc(g),
                                                model->map,
                                                model->size,
                                                layer->attn_compressor_gate->abs_offset,
                                                DS4_N_EMBD,
                                                comp_width,
                                                metal_graph_batch_attn_norm(g),
                                                n_tokens) != 0;
                if (!ok) {
                    fprintf(stderr, "ds4: gpu layer %u attention compressor score projection failed\n", il);
                }
            }
        }
        if (ok) metal_graph_debug_dump_tensor("attn_comp_kv_raw",
                                              metal_graph_batch_comp_kv(g),
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        if (ok) metal_graph_debug_dump_tensor("attn_comp_score_raw",
                                              metal_graph_batch_comp_sc(g),
                                              (uint64_t)comp_width * n_tokens,
                                              il,
                                              pos0);
        uint32_t n_comp = g->layer_n_comp[il];
        if (zero_prefix) {
            n_comp = n_tokens / ratio;
            if (ok && n_comp > g->layer_comp_cap[il]) {
                fprintf(stderr, "ds4: Metal layer-major compressed KV cache capacity exceeded at layer %u\n", il);
                ok = false;
            }
            if (ok && n_comp > g->attn_comp_stage_cap) {
                fprintf(stderr, "ds4: Metal graph compressed KV staging capacity exceeded at layer %u\n", il);
                ok = false;
            }
            ds4_gpu_tensor *attn_comp_target = NULL;
            if (ok) {
                attn_comp_target = metal_graph_attn_comp_prefill_target(g, il, 0, n_comp);
                if (!attn_comp_target) {
                    fprintf(stderr, "ds4: gpu layer %u attention compressor target creation failed\n", il);
                    ok = false;
                }
                if (ok) ok = ds4_gpu_compressor_prefill_tensor(attn_comp_target,
                                                         g->layer_attn_state_kv[il],
                                                         g->layer_attn_state_score[il],
                                                         metal_graph_batch_comp_kv(g),
                                                         metal_graph_batch_comp_sc(g),
                                                         model->map,
                                                         model->size,
                                                         layer->attn_compressor_ape->abs_offset,
                                                         layer->attn_compressor_ape->type,
                                                         layer->attn_compressor_norm->abs_offset,
                                                         layer->attn_compressor_norm->type,
                                                         DS4_N_HEAD_DIM,
                                                         ratio,
                                                         pos0,
                                                         n_tokens,
                                                         DS4_N_ROT,
                                                         compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                         true,
                                                         freq_base,
                                                         freq_scale,
                                                         ext_factor,
                                                         attn_factor,
                                                         DS4_ROPE_YARN_BETA_FAST,
                                                         DS4_ROPE_YARN_BETA_SLOW,
                                                         DS4_RMS_EPS) != 0;
                if (!ok) {
                    fprintf(stderr, "ds4: gpu layer %u attention compressor prefill failed\n", il);
                }
                DS4_METAL_PROFILE_ATTN_STAGE("compressor_prefill");
                if (ok && n_comp != 0) {
                    ok = metal_graph_commit_attn_comp_stage(g, il, 0, n_comp);
                }
                DS4_METAL_PROFILE_ATTN_STAGE("compressor_commit");
                if (ok && ratio == 4) {
                    ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                DS4_METAL_PROFILE_ATTN_STAGE("compressor_refresh");
            }
            if (ok) {
                g->layer_n_comp[il] = n_comp;
                for (uint32_t t = 0; t < n_tokens; t++) {
                    comp_counts[t] = (pos0 + t + 1u) / ratio;
                }
                if (n_comp != 0) {
                    metal_graph_debug_dump_tensor("KVcompress",
                                                  attn_comp_target,
                                                  (uint64_t)n_comp * DS4_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                }
                metal_graph_debug_dump_tensor("attn_state_kv",
                                              g->layer_attn_state_kv[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
                metal_graph_debug_dump_tensor("attn_state_score",
                                              g->layer_attn_state_score[il],
                                              (uint64_t)comp_width * coff * ratio,
                                              il,
                                              pos0);
            }
            metal_graph_attn_comp_prefill_target_free(attn_comp_target);
        } else {
            const bool aligned_chunk =
                getenv("DS4_CUDA_NO_COMPRESSOR_PREFILL_BATCH") == NULL &&
                !g->spec_capture_prefixes &&
                (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
            if (aligned_chunk) {
                const uint32_t comp_before = g->layer_n_comp[il];
                const uint32_t comp_chunk = n_tokens / ratio;
                if (comp_before + comp_chunk > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok && comp_chunk > g->attn_comp_stage_cap) {
                    fprintf(stderr, "ds4: Metal graph compressed KV staging capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                ds4_gpu_tensor *attn_comp_target =
                    ok ? metal_graph_attn_comp_prefill_target(g, il, comp_before, comp_chunk) : NULL;
                if (ok && !attn_comp_target) ok = false;
                if (ok && ratio == 4) {
                    ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            metal_graph_batch_comp_kv(g),
                            metal_graph_batch_comp_sc(g),
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                } else if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(
                            attn_comp_target,
                            g->layer_attn_state_kv[il],
                            g->layer_attn_state_score[il],
                            metal_graph_batch_comp_kv(g),
                            metal_graph_batch_comp_sc(g),
                            model->map,
                            model->size,
                            layer->attn_compressor_ape->abs_offset,
                            layer->attn_compressor_ape->type,
                            layer->attn_compressor_norm->abs_offset,
                            layer->attn_compressor_norm->type,
                            DS4_N_HEAD_DIM,
                            ratio,
                            pos0,
                            n_tokens,
                            DS4_N_ROT,
                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                            true,
                            freq_base,
                            freq_scale,
                            ext_factor,
                            attn_factor,
                            DS4_ROPE_YARN_BETA_FAST,
                            DS4_ROPE_YARN_BETA_SLOW,
                            DS4_RMS_EPS) != 0;
                }
                if (ok && comp_chunk != 0) {
                    ok = metal_graph_commit_attn_comp_stage(g, il, comp_before, comp_chunk);
                }
                if (ok && ratio == 4) {
                    ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_attn_state_kv[il],
                                                                     g->layer_attn_state_score[il],
                                                                     layer->attn_compressor_kv,
                                                                     layer->attn_compressor_gate,
                                                                     layer->attn_compressor_ape,
                                                                     DS4_N_HEAD_DIM,
                                                                     comp_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_comp[il] = comp_before + comp_chunk;
                    if (comp_counts) {
                        for (uint32_t t = 0; t < n_tokens; t++) {
                            comp_counts[t] = (pos0 + t + 1u) / ratio;
                        }
                    }
                    metal_graph_debug_dump_tensor("KVcompress",
                                                  attn_comp_target,
                                                  (uint64_t)comp_chunk * DS4_N_HEAD_DIM,
                                                  il,
                                                  pos0);
                    metal_graph_debug_dump_tensor("attn_state_kv",
                                                  g->layer_attn_state_kv[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                    metal_graph_debug_dump_tensor("attn_state_score",
                                                  g->layer_attn_state_score[il],
                                                  (uint64_t)comp_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
                metal_graph_attn_comp_prefill_target_free(attn_comp_target);
            } else {
                for (uint32_t t = 0; ok && t < n_tokens; t++) {
                    const uint32_t pos = pos0 + t;
                    const bool emit = ((pos + 1u) % ratio) == 0u;
                    if (emit && g->layer_n_comp[il] >= g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: Metal graph compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                        break;
                    }
                    ds4_gpu_tensor *kv_view = metal_graph_tensor_row_view(metal_graph_batch_comp_kv(g), t, comp_width);
                    ds4_gpu_tensor *sc_view = metal_graph_tensor_row_view(metal_graph_batch_comp_sc(g), t, comp_width);
                    const uint32_t comp_row = g->layer_n_comp[il];
                    ok = kv_view && sc_view &&
                         ds4_gpu_compressor_update_tensor(kv_view,
                                                            sc_view,
                                                            g->layer_attn_state_kv[il],
                                                            g->layer_attn_state_score[il],
                                                            metal_graph_attn_comp_update_target(g, il),
                                                            model->map,
                                                            model->size,
                                                            layer->attn_compressor_ape->abs_offset,
                                                            layer->attn_compressor_ape->type,
                                                            layer->attn_compressor_norm->abs_offset,
                                                            layer->attn_compressor_norm->type,
                                                            DS4_N_HEAD_DIM,
                                                            ratio,
                                                            pos,
                                                            metal_graph_attn_comp_update_row(comp_row),
                                                            DS4_N_ROT,
                                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                            freq_base,
                                                            freq_scale,
                                                            ext_factor,
                                                            attn_factor,
                                                            DS4_ROPE_YARN_BETA_FAST,
                                                            DS4_ROPE_YARN_BETA_SLOW,
                                                            DS4_RMS_EPS,
                                                            false,
                                                            false,
                                                            false) != 0;
                    if (ok && emit) {
                        ds4_gpu_tensor *comp_row_view = metal_graph_attn_comp_row_view(g, il, comp_row);
                        ok = comp_row_view &&
                             ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_row_view,
                                                                   1,
                                                                   DS4_N_HEAD_DIM,
                                                                   DS4_N_ROT) != 0;
                        if (ok) {
                            metal_graph_debug_dump_tensor("KVcompress",
                                                          comp_row_view,
                                                          DS4_N_HEAD_DIM,
                                                          il,
                                                          pos);
                        }
                        ds4_gpu_tensor_free(comp_row_view);
                        if (ok) ok = metal_graph_commit_attn_comp_stage(g, il, comp_row, 1);
                    }
                    if (ok && emit) g->layer_n_comp[il]++;
                    if (comp_counts) comp_counts[t] = g->layer_n_comp[il];
                    if (ok && t + 1u < n_tokens &&
                        t < DS4_SPEC_PREFIX_SLOTS) {
                        ok = metal_graph_capture_prefix_attn_state(g, il, t);
                    }
                    ds4_gpu_tensor_free(sc_view);
                    ds4_gpu_tensor_free(kv_view);
                }
            }
            n_comp = g->layer_n_comp[il];
        }
        DS4_METAL_PROFILE_ATTN_STAGE("compressor");

        if (ok && ratio == 4) {
            const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
            if (!layer->indexer_compressor_kv || !layer->indexer_compressor_gate ||
                !layer->indexer_compressor_ape || !layer->indexer_compressor_norm ||
                !layer->indexer_attn_q_b || !layer->indexer_proj) {
                fprintf(stderr, "ds4: Metal layer-major prefill needs indexer weights\n");
                ok = false;
            }
            if (ok) {
                ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_comp_kv(g),
                                                 model->map,
                                                 model->size,
                                                 layer->indexer_compressor_kv->abs_offset,
                                                 DS4_N_EMBD,
                                                 index_width,
                                                 metal_graph_batch_attn_norm(g),
                                                 n_tokens) != 0;
                if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_comp_sc(g),
                                                         model->map,
                                                         model->size,
                                                         layer->indexer_compressor_gate->abs_offset,
                                                         DS4_N_EMBD,
                                                         index_width,
                                                         metal_graph_batch_attn_norm(g),
                                                         n_tokens) != 0;
            }
            if (ok) metal_graph_debug_dump_tensor("indexer_comp_kv_raw",
                                                  metal_graph_batch_comp_kv(g),
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            if (ok) metal_graph_debug_dump_tensor("indexer_comp_score_raw",
                                                  metal_graph_batch_comp_sc(g),
                                                  (uint64_t)index_width * n_tokens,
                                                  il,
                                                  pos0);
            if (ok) ok = metal_graph_matmul_plain_tensor(metal_graph_batch_indexer_q(g),
                                                          model,
                                                          layer->indexer_attn_q_b,
                                                          q_rank,
                                                          (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                          metal_graph_batch_qr_norm(g),
                                                          n_tokens);
            if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_batch_indexer_q(g),
                                                    n_tokens,
                                                    DS4_N_INDEXER_HEAD,
                                                    DS4_N_INDEXER_HEAD_DIM,
                                                    DS4_N_ROT,
                                                    pos0,
                                                    compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                    false,
                                                    freq_base,
                                                    freq_scale,
                                                    ext_factor,
                                                    attn_factor,
                                                    DS4_ROPE_YARN_BETA_FAST,
                                                    DS4_ROPE_YARN_BETA_SLOW) != 0;
            if (ok) ok = ds4_gpu_dsv4_indexer_qat_tensor(metal_graph_batch_indexer_q(g),
                                                          n_tokens * DS4_N_INDEXER_HEAD,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
            if (ok) ok = ds4_gpu_matmul_f16_tensor(metal_graph_batch_indexer_weights(g),
                                                     model->map,
                                                     model->size,
                                                     layer->indexer_proj->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_INDEXER_HEAD,
                                                     metal_graph_batch_attn_norm(g),
                                                     n_tokens) != 0;
            if (zero_prefix) {
                if (ok && n_comp > g->layer_comp_cap[il]) {
                    fprintf(stderr, "ds4: Metal layer-major indexer cache capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok && n_comp > g->attn_comp_stage_cap) {
                    fprintf(stderr, "ds4: Spark indexer staging capacity exceeded at layer %u\n", il);
                    ok = false;
                }
                if (ok) {
                    ok = ds4_gpu_compressor_prefill_tensor(metal_graph_index_comp_stage(g),
                                                             g->layer_index_state_kv[il],
                                                             g->layer_index_state_score[il],
                                                             metal_graph_batch_comp_kv(g),
                                                             metal_graph_batch_comp_sc(g),
                                                             model->map,
                                                             model->size,
                                                             layer->indexer_compressor_ape->abs_offset,
                                                             layer->indexer_compressor_ape->type,
                                                             layer->indexer_compressor_norm->abs_offset,
                                                             layer->indexer_compressor_norm->type,
                                                             DS4_N_INDEXER_HEAD_DIM,
                                                             ratio,
                                                             pos0,
                                                             n_tokens,
                                                             DS4_N_ROT,
                                                             compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                             false,
                                                             freq_base,
                                                             freq_scale,
                                                             ext_factor,
                                                             attn_factor,
                                                             DS4_ROPE_YARN_BETA_FAST,
                                                             DS4_ROPE_YARN_BETA_SLOW,
                                                             DS4_RMS_EPS) != 0;
                }
                if (ok && n_comp != 0) {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(metal_graph_index_comp_stage(g),
                                                          n_comp,
                                                          DS4_N_INDEXER_HEAD_DIM) != 0;
                    if (ok) {
                        ok = ds4x_graph_cache_pack(
                                g, DS4X_CACHE_INDEXER,
                                g->layer_index_comp_cache[il], 0,
                                metal_graph_index_comp_stage(g), 0, n_comp);
                    }
                }
                if (ok) {
                    ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                     model,
                                                                     g->layer_index_state_kv[il],
                                                                     g->layer_index_state_score[il],
                                                                     layer->indexer_compressor_kv,
                                                                     layer->indexer_compressor_gate,
                                                                     layer->indexer_compressor_ape,
                                                                     DS4_N_INDEXER_HEAD_DIM,
                                                                     index_width,
                                                                     pos0,
                                                                     n_tokens);
                }
                if (ok) {
                    g->layer_n_index_comp[il] = n_comp;
                    for (uint32_t t = 0; t < n_tokens; t++) {
                        index_counts[t] = (pos0 + t + 1u) / ratio;
                    }
                    if (n_comp != 0) {
                        metal_graph_debug_dump_tensor("indexer_KVcompress",
                                                      metal_graph_index_comp_stage(g),
                                                      (uint64_t)n_comp * DS4_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                    }
                    metal_graph_debug_dump_tensor("indexer_state_kv",
                                                  g->layer_index_state_kv[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                    metal_graph_debug_dump_tensor("indexer_state_score",
                                                  g->layer_index_state_score[il],
                                                  (uint64_t)index_width * coff * ratio,
                                                  il,
                                                  pos0);
                }
            } else {
                const bool aligned_chunk =
                    getenv("DS4_CUDA_NO_COMPRESSOR_PREFILL_BATCH") == NULL &&
                    !g->spec_capture_prefixes &&
                    (pos0 % ratio) == 0u && (n_tokens % ratio) == 0u;
                if (aligned_chunk) {
                    const uint32_t index_before = g->layer_n_index_comp[il];
                    const uint32_t index_chunk = n_tokens / ratio;
                    if (index_before + index_chunk > g->layer_comp_cap[il]) {
                        fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                        ok = false;
                    }
                    if (ok && index_chunk > g->attn_comp_stage_cap) {
                        fprintf(stderr, "ds4: Spark indexer staging capacity exceeded at layer %u\n", il);
                        ok = false;
                    }
                    ds4_gpu_tensor *index_view = NULL;
                    if (ok) {
                        index_view = ds4_gpu_tensor_view(
                                metal_graph_index_comp_stage(g),
                                0,
                                (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                        ok = index_view != NULL;
                    }
                    if (ok) {
                        ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                                index_view,
                                g->layer_index_state_kv[il],
                                g->layer_index_state_score[il],
                                metal_graph_batch_comp_kv(g),
                                metal_graph_batch_comp_sc(g),
                                model->map,
                                model->size,
                                layer->indexer_compressor_ape->abs_offset,
                                layer->indexer_compressor_ape->type,
                                layer->indexer_compressor_norm->abs_offset,
                                layer->indexer_compressor_norm->type,
                                DS4_N_INDEXER_HEAD_DIM,
                                pos0,
                                n_tokens,
                                DS4_N_ROT,
                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                false,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                DS4_ROPE_YARN_BETA_FAST,
                                DS4_ROPE_YARN_BETA_SLOW,
                                DS4_RMS_EPS) != 0;
                    }
                    if (ok && index_chunk != 0) {
                        ok = ds4_gpu_dsv4_indexer_qat_tensor(index_view,
                                                              index_chunk,
                                                              DS4_N_INDEXER_HEAD_DIM) != 0;
                        if (ok) {
                            ok = ds4x_graph_cache_pack(
                                    g, DS4X_CACHE_INDEXER,
                                    g->layer_index_comp_cache[il], index_before,
                                    index_view, 0, index_chunk);
                        }
                    }
                    if (ok) {
                        ok = metal_graph_refresh_ratio4_compressor_state(g,
                                                                         model,
                                                                         g->layer_index_state_kv[il],
                                                                         g->layer_index_state_score[il],
                                                                         layer->indexer_compressor_kv,
                                                                         layer->indexer_compressor_gate,
                                                                         layer->indexer_compressor_ape,
                                                                         DS4_N_INDEXER_HEAD_DIM,
                                                                         index_width,
                                                                         pos0,
                                                                         n_tokens);
                    }
                    if (ok) {
                        g->layer_n_index_comp[il] = index_before + index_chunk;
                        if (index_counts) {
                            for (uint32_t t = 0; t < n_tokens; t++) {
                                index_counts[t] = (pos0 + t + 1u) / ratio;
                            }
                        }
                        metal_graph_debug_dump_tensor("indexer_KVcompress",
                                                      index_view,
                                                      (uint64_t)index_chunk * DS4_N_INDEXER_HEAD_DIM,
                                                      il,
                                                      pos0);
                        metal_graph_debug_dump_tensor("indexer_state_kv",
                                                      g->layer_index_state_kv[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                        metal_graph_debug_dump_tensor("indexer_state_score",
                                                      g->layer_index_state_score[il],
                                                      (uint64_t)index_width * coff * ratio,
                                                      il,
                                                      pos0);
                    }
                    ds4_gpu_tensor_free(index_view);
                } else {
                    for (uint32_t t = 0; ok && t < n_tokens; t++) {
                        const uint32_t pos = pos0 + t;
                        const bool emit = ((pos + 1u) % ratio) == 0u;
                        if (emit && g->layer_n_index_comp[il] >= g->layer_comp_cap[il]) {
                            fprintf(stderr, "ds4: Metal graph indexer compressed KV cache capacity exceeded at layer %u\n", il);
                            ok = false;
                            break;
                        }
                        ds4_gpu_tensor *kv_view = metal_graph_tensor_row_view(metal_graph_batch_comp_kv(g), t, index_width);
                        ds4_gpu_tensor *sc_view = metal_graph_tensor_row_view(metal_graph_batch_comp_sc(g), t, index_width);
                        const uint32_t index_row = g->layer_n_index_comp[il];
                        ok = kv_view && sc_view &&
                             ds4_gpu_compressor_update_tensor(kv_view,
                                                                sc_view,
                                                                g->layer_index_state_kv[il],
                                                                g->layer_index_state_score[il],
                                                                metal_graph_index_comp_stage(g),
                                                                model->map,
                                                                model->size,
                                                                layer->indexer_compressor_ape->abs_offset,
                                                                layer->indexer_compressor_ape->type,
                                                                layer->indexer_compressor_norm->abs_offset,
                                                                layer->indexer_compressor_norm->type,
                                                                DS4_N_INDEXER_HEAD_DIM,
                                                                ratio,
                                                                pos,
                                                                0,
                                                                DS4_N_ROT,
                                                                compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                                                freq_base,
                                                                freq_scale,
                                                                ext_factor,
                                                                attn_factor,
                                                                DS4_ROPE_YARN_BETA_FAST,
                                                                DS4_ROPE_YARN_BETA_SLOW,
                                                                DS4_RMS_EPS,
                                                                false,
                                                                false,
                                                                false) != 0;
                        if (ok && emit) {
                            ds4_gpu_tensor *index_row_view = ds4_gpu_tensor_view(
                                    metal_graph_index_comp_stage(g),
                                    0,
                                    (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
                            if (!index_row_view) {
                                ok = false;
                            } else {
                                ok = ds4_gpu_dsv4_indexer_qat_tensor(index_row_view,
                                                                      1,
                                                                      DS4_N_INDEXER_HEAD_DIM) != 0;
                                if (ok) {
                                    ok = ds4x_graph_cache_pack(
                                            g, DS4X_CACHE_INDEXER,
                                            g->layer_index_comp_cache[il], index_row,
                                            index_row_view, 0, 1);
                                }
                                ds4_gpu_tensor_free(index_row_view);
                            }
                        }
                        if (ok && emit) g->layer_n_index_comp[il]++;
                        if (index_counts) index_counts[t] = g->layer_n_index_comp[il];
                        if (ok && t + 1u < n_tokens &&
                            t < DS4_SPEC_PREFIX_SLOTS) {
                            ok = metal_graph_capture_prefix_index_state(g, il, t);
                        }
                        ds4_gpu_tensor_free(sc_view);
                        ds4_gpu_tensor_free(kv_view);
                    }
                }
            }
        }
        if (ratio == 4) DS4_METAL_PROFILE_ATTN_STAGE("indexer_setup");

        if (ok && (!DS4_GPU_ATTN_COMP_CACHE_SPARK || spark_packed_prefill_batch) &&
            !zero_prefix && n_tokens <= g->raw_cap) {
            const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos0, n_tokens);
            /* See the raw-only branch above: batched mixed attention also
             * consumes a logical raw window, linearized out of the ring. */
            const uint32_t raw_start = metal_graph_raw_start_for_span(g,
                                                                      pos0 + n_tokens - 1u,
                                                                      n_raw);
            uint32_t use_comp_mask = 0;
            bool use_indexed_comp = false;
            double index_stage_t0 = 0.0;

            ok = ds4_gpu_store_raw_kv_batch_tensor(g->layer_raw_cache[il],
                                                     metal_graph_batch_kv(g),
                                                     g->raw_cap,
                                                     pos0,
                                                     n_tokens,
                                                     DS4_N_HEAD_DIM) != 0;
            if (ok && ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K) {
                const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                if (index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary(NULL,
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                ok = ds4x_graph_indexer(
                        g, DS4X_INDEXER_VERIFY,
                        metal_graph_indexer_scores(g),
                        metal_graph_comp_selected(g),
                        metal_graph_batch_indexer_q(g),
                        metal_graph_batch_indexer_weights(g),
                        g->layer_index_comp_cache[il], n_comp, n_tokens,
                        pos0, ratio, index_scale);
                if (ok) {
                    metal_graph_debug_dump_tensor("indexer_scores",
                                                  metal_graph_indexer_scores(g),
                                                  (uint64_t)n_comp * n_tokens,
                                                  il,
                                                  pos0);
                }
                if (ok) {
                    if (index_stage_profile) {
                        ok = metal_graph_indexer_stage_profile_boundary("score_topk",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                    if (ok) {
                        metal_graph_debug_dump_i32_tensor("indexer_topk",
                                                          metal_graph_comp_selected(g),
                                                          (uint64_t)n_tokens * DS4_N_INDEXER_TOP_K,
                                                          il,
                                                          pos0);
                    }
                }
                if (ok) {
                    use_indexed_comp = true;
                }
                use_comp_mask = 1;
            }
            if (ok) {
                if (use_indexed_comp) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(metal_graph_batch_heads(g),
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              metal_graph_batch_q(g),
                                                                              g->layer_raw_cache[il],
                                                                              g->layer_attn_comp_cache[il],
                                                                              metal_graph_attn_comp_cache_format(),
                                                                              metal_graph_comp_selected(g),
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              DS4_N_INDEXER_TOP_K,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM) != 0;
                    if (ok && index_stage_profile) {
                        ok = metal_graph_indexer_stage_profile_boundary("attention",
                                                                        il,
                                                                        pos0,
                                                                        n_tokens,
                                                                        n_comp,
                                                                        &index_stage_t0);
                    }
                } else {
                    ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(metal_graph_batch_heads(g),
                                                                             model->map,
                                                                             model->size,
                                                                             layer->attn_sinks->abs_offset,
                                                                             metal_graph_batch_q(g),
                                                                             g->layer_raw_cache[il],
                                                                             g->layer_attn_comp_cache[il],
                                                                             metal_graph_attn_comp_cache_format(),
                                                                             use_comp_mask ? metal_graph_comp_mask(g) : NULL,
                                                                             use_comp_mask,
                                                                             n_tokens,
                                                                             pos0,
                                                                             n_raw,
                                                                             g->raw_cap,
                                                                             raw_start,
                                                                             n_comp,
                                                                             g->raw_window,
                                                                             ratio,
                                                                             DS4_N_HEAD,
                                                                             DS4_N_HEAD_DIM) != 0;
                }
            }
            if (ok) batch_attention_done = true;
        }

        const bool topk_prefill_needed = ratio == 4 && n_comp > DS4_N_INDEXER_TOP_K;
        if (ok && (!DS4_GPU_ATTN_COMP_CACHE_SPARK || spark_packed_prefill_batch) &&
            zero_prefix && topk_prefill_needed && n_comp != 0) {
            const uint32_t raw_prefix_tokens = ratio - 1u;
            const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
            double index_stage_t0 = 0.0;
            if (index_stage_profile) {
                ok = metal_graph_indexer_stage_profile_boundary(NULL,
                                                                il,
                                                                pos0,
                                                                n_tokens,
                                                                n_comp,
                                                                &index_stage_t0);
            }
            ok = ds4x_graph_indexer(
                    g, DS4X_INDEXER_PREFILL,
                    metal_graph_indexer_scores(g),
                    metal_graph_comp_selected(g),
                    metal_graph_batch_indexer_q(g),
                    metal_graph_batch_indexer_weights(g),
                    g->layer_index_comp_cache[il], n_comp, n_tokens,
                    pos0, ratio, index_scale);
            if (ok) {
                metal_graph_debug_dump_tensor("indexer_scores",
                                              metal_graph_indexer_scores(g),
                                              (uint64_t)n_comp * n_tokens,
                                              il,
                                              pos0);
            }
            if (ok) {
                if (index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("score_topk",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
                if (ok) {
                    metal_graph_debug_dump_i32_tensor("indexer_topk",
                                                      metal_graph_comp_selected(g),
                                                      (uint64_t)n_tokens * DS4_N_INDEXER_TOP_K,
                                                      il,
                                                      pos0);
                }
            }
            if (ok && DS4_GPU_ATTN_COMP_CACHE_SPARK && raw_prefix_tokens != 0u) {
                ok = ds4_gpu_attention_prefill_raw_heads_tensor(
                        metal_graph_batch_heads(g),
                        model->map,
                        model->size,
                        layer->attn_sinks->abs_offset,
                        metal_graph_batch_q(g),
                        metal_graph_batch_kv(g),
                        raw_prefix_tokens,
                        g->raw_window,
                        DS4_N_HEAD,
                        DS4_N_HEAD_DIM) != 0;
            }
            if (ok && DS4_GPU_ATTN_COMP_CACHE_SPARK) {
                const uint32_t suffix_tokens = n_tokens - raw_prefix_tokens;
                ds4_gpu_tensor *suffix_heads = metal_graph_tensor_row_range_view(
                        metal_graph_batch_heads(g), raw_prefix_tokens,
                        suffix_tokens, q_dim);
                ds4_gpu_tensor *suffix_q = metal_graph_tensor_row_range_view(
                        metal_graph_batch_q(g), raw_prefix_tokens,
                        suffix_tokens, q_dim);
                ds4_gpu_tensor *suffix_topk = metal_graph_tensor_row_range_view(
                        metal_graph_comp_selected(g), raw_prefix_tokens,
                        suffix_tokens, DS4_N_INDEXER_TOP_K);
                ok = suffix_heads && suffix_q && suffix_topk &&
                     ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                            suffix_heads,
                            model->map,
                            model->size,
                            layer->attn_sinks->abs_offset,
                            suffix_q,
                            g->layer_raw_cache[il],
                            g->layer_attn_comp_cache[il],
                            metal_graph_attn_comp_cache_format(),
                            suffix_topk,
                            suffix_tokens,
                            raw_prefix_tokens,
                            n_tokens,
                            g->raw_cap,
                            0,
                            n_comp,
                            DS4_N_INDEXER_TOP_K,
                            g->raw_window,
                            ratio,
                            DS4_N_HEAD,
                            DS4_N_HEAD_DIM) != 0;
                ds4_gpu_tensor_free(suffix_topk);
                ds4_gpu_tensor_free(suffix_q);
                ds4_gpu_tensor_free(suffix_heads);
                if (ok && index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("attention",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
            } else if (ok) {
                ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(metal_graph_batch_heads(g),
                                                                          model->map,
                                                                          model->size,
                                                                          layer->attn_sinks->abs_offset,
                                                                          metal_graph_batch_q(g),
                                                                          g->layer_raw_cache[il],
                                                                          g->layer_attn_comp_cache[il],
                                                                          metal_graph_attn_comp_cache_format(),
                                                                          metal_graph_comp_selected(g),
                                                                          n_tokens,
                                                                          pos0,
                                                                          n_tokens,
                                                                          g->raw_cap,
                                                                          0,
                                                                          n_comp,
                                                                          DS4_N_INDEXER_TOP_K,
                                                                          g->raw_window,
                                                                          ratio,
                                                                          DS4_N_HEAD,
                                                                          DS4_N_HEAD_DIM) != 0;
                if (ok && index_stage_profile) {
                    ok = metal_graph_indexer_stage_profile_boundary("attention",
                                                                    il,
                                                                    pos0,
                                                                    n_tokens,
                                                                    n_comp,
                                                                    &index_stage_t0);
                }
            }
            if (ok) batch_attention_done = true;
        }
        if (ok && (!DS4_GPU_ATTN_COMP_CACHE_SPARK || spark_packed_prefill_batch) &&
            zero_prefix && !topk_prefill_needed && n_comp != 0) {
            if (DS4_GPU_ATTN_COMP_CACHE_SPARK) {
                const uint32_t raw_prefix_tokens = ratio - 1u;
                const uint32_t suffix_tokens = n_tokens - raw_prefix_tokens;
                ok = ds4_gpu_attention_prefill_raw_heads_tensor(
                        metal_graph_batch_heads(g),
                        model->map,
                        model->size,
                        layer->attn_sinks->abs_offset,
                        metal_graph_batch_q(g),
                        metal_graph_batch_kv(g),
                        raw_prefix_tokens,
                        g->raw_window,
                        DS4_N_HEAD,
                        DS4_N_HEAD_DIM) != 0;
                ds4_gpu_tensor *suffix_heads = NULL;
                ds4_gpu_tensor *suffix_q = NULL;
                if (ok) {
                    suffix_heads = metal_graph_tensor_row_range_view(
                            metal_graph_batch_heads(g), raw_prefix_tokens,
                            suffix_tokens, q_dim);
                    suffix_q = metal_graph_tensor_row_range_view(
                            metal_graph_batch_q(g), raw_prefix_tokens,
                            suffix_tokens, q_dim);
                    ok = suffix_heads && suffix_q;
                    if (ok && suffix_tokens == 1u) {
                        ok = ds4x_graph_attention_decode(
                                g, suffix_heads, model,
                                layer->attn_sinks->abs_offset, suffix_q,
                                g->layer_raw_cache[il], n_tokens, g->raw_cap, 0,
                                g->layer_attn_comp_cache[il], n_comp);
                    } else if (ok) {
                        ok = ds4_gpu_attention_decode_mixed_batch_heads_tensor(
                                suffix_heads,
                                model->map,
                                model->size,
                                layer->attn_sinks->abs_offset,
                                suffix_q,
                                g->layer_raw_cache[il],
                                g->layer_attn_comp_cache[il],
                                metal_graph_attn_comp_cache_format(),
                                NULL,
                                0,
                                suffix_tokens,
                                raw_prefix_tokens,
                                n_tokens,
                                g->raw_cap,
                                0,
                                n_comp,
                                g->raw_window,
                                ratio,
                                DS4_N_HEAD,
                                DS4_N_HEAD_DIM) != 0;
                    }
                }
                ds4_gpu_tensor_free(suffix_q);
                ds4_gpu_tensor_free(suffix_heads);
            } else {
                ok = ds4_gpu_attention_prefill_static_mixed_heads_tensor(metal_graph_batch_heads(g),
                                                                           model->map,
                                                                           model->size,
                                                                           layer->attn_sinks->abs_offset,
                                                                           metal_graph_batch_q(g),
                                                                           metal_graph_batch_kv(g),
                                                                           g->layer_attn_comp_cache[il],
                                                                           metal_graph_attn_comp_cache_format(),
                                                                           n_tokens,
                                                                           n_comp,
                                                                           g->raw_window,
                                                                           ratio,
                                                                           DS4_N_HEAD,
                                                                           DS4_N_HEAD_DIM) != 0;
            }
            if (ok) batch_attention_done = true;
        }
    }

    if (ok && !raw_batch_attention && !batch_attention_done) {
        uint32_t raw_prefix_tokens = 0;
        if (zero_prefix && ratio != 0 && n_tokens <= g->raw_cap && comp_counts != NULL) {
            while (raw_prefix_tokens < n_tokens && comp_counts[raw_prefix_tokens] == 0u) {
                raw_prefix_tokens++;
            }
        }

        if (raw_prefix_tokens != 0) {
            ok = ds4_gpu_attention_prefill_raw_heads_tensor(
                    metal_graph_batch_heads(g), model->map, model->size,
                    layer->attn_sinks->abs_offset, metal_graph_batch_q(g),
                    metal_graph_batch_kv(g), raw_prefix_tokens,
                    g->raw_window, DS4_N_HEAD, DS4_N_HEAD_DIM) != 0;
        }
        if (raw_prefix_tokens < n_tokens) {
            for (uint32_t t = raw_prefix_tokens; ok && t < n_tokens; t++) {
                const uint32_t pos = pos0 + t;
                const uint32_t n_raw = metal_graph_raw_span_for_batch(g, pos, 1);
                const uint32_t raw_start = metal_graph_raw_start_for_span(g, pos, n_raw);
                const uint32_t cur_comp = comp_counts ? comp_counts[t] : 0u;
                const uint32_t cur_index = index_counts ? index_counts[t] : 0u;
                uint32_t n_selected = 0;
                ds4_gpu_tensor *comp_mask = NULL;

                if (ratio == 4 && cur_comp > DS4_N_INDEXER_TOP_K) {
                    const float index_scale = 1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    ds4_gpu_tensor *indexer_q_view = metal_graph_tensor_row_view(
                            metal_graph_batch_indexer_q(g), t, (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM);
                    ds4_gpu_tensor *indexer_w_view = metal_graph_tensor_row_view(
                            metal_graph_batch_indexer_weights(g), t, DS4_N_INDEXER_HEAD);
                    ok = indexer_q_view && indexer_w_view &&
                         ds4x_graph_indexer(
                                 g, DS4X_INDEXER_DECODE_ONE,
                                 metal_graph_indexer_scores(g),
                                 metal_graph_comp_selected(g),
                                 indexer_q_view, indexer_w_view,
                                 g->layer_index_comp_cache[il], cur_index, 1,
                                 pos, ratio, index_scale) &&
                         ds4_gpu_dsv4_topk_mask_tensor(metal_graph_comp_mask(g),
                                                         metal_graph_comp_selected(g),
                                                         cur_index,
                                                         1,
                                                         DS4_N_INDEXER_TOP_K) != 0;
                    ds4_gpu_tensor_free(indexer_w_view);
                    ds4_gpu_tensor_free(indexer_q_view);
                    if (ok) {
                        comp_mask = metal_graph_comp_mask(g);
                        n_selected = DS4_N_INDEXER_TOP_K < cur_index
                            ? DS4_N_INDEXER_TOP_K
                            : cur_index;
                    }
                }

                ds4_gpu_tensor *q_view = metal_graph_tensor_row_view(metal_graph_batch_q(g), t, q_dim);
                ds4_gpu_tensor *kv_cache_view = metal_graph_tensor_row_view(metal_graph_batch_kv(g), t, DS4_N_HEAD_DIM);
                ds4_gpu_tensor *heads_view = metal_graph_tensor_row_view(metal_graph_batch_heads(g), t, q_dim);
                ok = ok && q_view && kv_cache_view && heads_view;
                if (ok && !zero_prefix) {
                    ok = ds4_gpu_store_raw_kv_tensor(g->layer_raw_cache[il],
                                                       kv_cache_view,
                                                       g->raw_cap,
                                                       pos % g->raw_cap,
                                                       DS4_N_HEAD_DIM) != 0;
                }
                if (ok && comp_mask != NULL && n_selected != 0) {
                    ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(heads_view,
                                                                              model->map,
                                                                              model->size,
                                                                              layer->attn_sinks->abs_offset,
                                                                              q_view,
                                                                              g->layer_raw_cache[il],
                                                                              g->layer_attn_comp_cache[il],
                                                                              metal_graph_attn_comp_cache_format(),
                                                                              metal_graph_comp_selected(g),
                                                                              1,
                                                                              pos,
                                                                              n_raw,
                                                                              g->raw_cap,
                                                                              raw_start,
                                                                              cur_comp,
                                                                              n_selected,
                                                                              g->raw_window,
                                                                              ratio,
                                                                              DS4_N_HEAD,
                                                                              DS4_N_HEAD_DIM) != 0;
                } else if (ok) {
                    ok = ds4x_graph_attention_decode(
                            g, heads_view, model,
                            layer->attn_sinks->abs_offset, q_view,
                            g->layer_raw_cache[il], n_raw, g->raw_cap,
                            raw_start,
                            cur_comp ? g->layer_attn_comp_cache[il] : NULL,
                            cur_comp);
                }
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(kv_cache_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
    }
    DS4_METAL_PROFILE_ATTN_STAGE("attention");

    if (ok) {
        metal_graph_debug_dump_tensor("kqv_out", metal_graph_batch_heads(g),
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    if (ok) ok = ds4_gpu_rope_tail_tensor(metal_graph_batch_heads(g),
                                            n_tokens,
                                            DS4_N_HEAD,
                                            DS4_N_HEAD_DIM,
                                            DS4_N_ROT,
                                            pos0,
                                            compressed ? (uint32_t)DS4_ROPE_ORIG_CTX : 0,
                                            true,
                                            freq_base,
                                            freq_scale,
                                            ext_factor,
                                            attn_factor,
                                            DS4_ROPE_YARN_BETA_FAST,
                                            DS4_ROPE_YARN_BETA_SLOW) != 0;
    if (ok) {
        metal_graph_debug_dump_tensor("kqv_back", metal_graph_batch_heads(g),
                                      (uint64_t)n_tokens * q_dim, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("inv_rope");
    const bool attn_out_debug =
        metal_graph_debug_wants("attn_low", il, pos0) ||
        metal_graph_debug_wants("attn_out", il, pos0);
    bool attn_out_f16 = false;
    if (ok &&
        !attn_out_debug &&
        layer->attn_output_a->type == DS4_TENSOR_Q8_0 &&
        layer->attn_output_b->type == DS4_TENSOR_Q8_0 &&
        !metal_graph_directional_steering_attn_enabled(g)) {
        attn_out_f16 = ds4_gpu_attention_output_q8_batch_f16_tensor(g->batch_q_half,
                                                                    metal_graph_batch_attn_low(g),
                                                                    model->map,
                                                                    model->size,
                                                                    layer->attn_output_a->abs_offset,
                                                                    layer->attn_output_b->abs_offset,
                                                                    group_dim,
                                                                    rank,
                                                                    n_groups,
                                                                    DS4_N_EMBD,
                                                                    metal_graph_batch_heads(g),
                                                                    n_tokens) != 0;
    }
    if (!attn_out_f16 && ok) {
        ok = metal_graph_attention_output_dense_quant_batch(
                metal_graph_batch_attn_out(g),
                metal_graph_batch_attn_low(g),
                g,
                model,
                layer->attn_output_a,
                layer->attn_output_b,
                group_dim,
                rank,
                n_groups,
                DS4_N_EMBD,
                metal_graph_batch_heads(g),
                n_tokens);
    }
    if (ok && !attn_out_f16) {
        metal_graph_debug_dump_tensor(
                "attn_low", metal_graph_batch_attn_low(g),
                (uint64_t)n_tokens * n_groups * rank, il, pos0);
        metal_graph_debug_dump_tensor(
                "attn_out", metal_graph_batch_attn_out(g),
                (uint64_t)n_tokens * DS4_N_EMBD, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("output_proj");
    if (ok && !attn_out_f16 && metal_graph_directional_steering_attn_enabled(g)) {
        ok = metal_graph_apply_directional_steering_attn(g, metal_graph_batch_attn_out(g), il, n_tokens);
    }
    if (ok && attn_out_f16) {
        ok = ds4_gpu_hc_expand_split_half_tensor(after_attn_hc_view,
                                                 g->batch_q_half,
                                                 metal_graph_batch_cur_hc(g),
                                                 hc_split_view,
                                                 DS4_N_EMBD,
                                                 DS4_N_HC) != 0;
    } else if (ok) {
        ok = ds4_gpu_hc_expand_split_tensor(after_attn_hc_view,
                                            metal_graph_batch_attn_out(g),
                                            metal_graph_batch_cur_hc(g),
                                            hc_split_view,
                                            DS4_N_EMBD,
                                            DS4_N_HC) != 0;
    }
    if (ok) {
        metal_graph_debug_dump_tensor("hc_attn_post", metal_graph_batch_after_attn_hc(g),
                                      (uint64_t)n_tokens * hc_dim, il, pos0);
    }
    DS4_METAL_PROFILE_ATTN_STAGE("hc_post");
    ds4_gpu_tensor_free(after_attn_hc_view);
    ds4_gpu_tensor_free(attn_cur_view);
    ds4_gpu_tensor_free(hc_split_view);
    ds4_gpu_tensor_free(hc_mix_view);
    if (index_counts != index_counts_stack) free(index_counts);
    if (comp_counts != comp_counts_stack) free(comp_counts);
#undef DS4_METAL_PROFILE_ATTN_STAGE
#undef DS4_METAL_PROFILE_Q_STAGE
    return ok;
}

#endif /* !DS4_NO_GPU */
