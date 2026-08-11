#ifndef DS4_NO_GPU
/* Timing-only skip-ablation for the GLM decode layer (comma list in
 * DS4_GLM_DECODE_ABLATE): the skipped stage's output buffer keeps stale
 * contents, so the run produces garbage text but every remaining dispatch
 * (and every TP gate) still executes. Whole-token time deltas against a
 * baseline run are the only reliable per-stage cost measurement — the
 * stage profiler's per-stage command-buffer splits inflate small stages. */
#define DS4_GLM_ABLATE_ATTN_OUT  (1u << 0)
#define DS4_GLM_ABLATE_ATTN_CORE (1u << 1)
#define DS4_GLM_ABLATE_QPATH     (1u << 2)
#define DS4_GLM_ABLATE_INDEXER   (1u << 3)
#define DS4_GLM_ABLATE_ROUTED    (1u << 4)
#define DS4_GLM_ABLATE_SHARED    (1u << 5)
#define DS4_GLM_ABLATE_QKLOW     (1u << 6)

static uint32_t glm_decode_ablate_mask(void) {
    static int cached = -1;
    if (cached < 0) {
        uint32_t mask = 0;
        const char *env = getenv("DS4_GLM_DECODE_ABLATE");
        if (env) {
            if (strstr(env, "attn_out")) mask |= DS4_GLM_ABLATE_ATTN_OUT;
            if (strstr(env, "attn_core")) mask |= DS4_GLM_ABLATE_ATTN_CORE;
            if (strstr(env, "qpath")) mask |= DS4_GLM_ABLATE_QPATH;
            if (strstr(env, "indexer")) mask |= DS4_GLM_ABLATE_INDEXER;
            if (strstr(env, "routed")) mask |= DS4_GLM_ABLATE_ROUTED;
            if (strstr(env, "shared")) mask |= DS4_GLM_ABLATE_SHARED;
            if (strstr(env, "qklow")) mask |= DS4_GLM_ABLATE_QKLOW;
            if (mask) {
                fprintf(stderr, "ds4: GLM decode ablation active (mask 0x%x) — output is garbage, timing only\n", mask);
            }
        }
        cached = (int)mask;
    }
    return (uint32_t)cached;
}

static bool glm_graph_encode_shared_swiglu_one(
        ds4_gpu_tensor          *mid,
        ds4_gpu_tensor          *gate,
        ds4_gpu_tensor          *up,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos,
        const ds4_gpu_tensor    *x,
        bool                     ssd_streaming,
        bool                     stage_profile,
        double                  *stage_t0) {
    if (!mid || !gate || !up || !model || !l || !x ||
        !l->ffn_gate_shexp || !l->ffn_up_shexp) {
        return false;
    }

    bool ok = true;
    if (glm_graph_weights_are_q8_0(model,
                                   l->ffn_gate_shexp->abs_offset,
                                   l->ffn_up_shexp->abs_offset)) {
        ok = ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                mid,
                model->map,
                model->size,
                l->ffn_gate_shexp->abs_offset,
                l->ffn_up_shexp->abs_offset,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                x,
                0.0f) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "shared_gate_up_swiglu",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
        return ok;
    }

    ok = glm_graph_matmul_q8_0_decode_profiled_tensor(gate,
                                                      model,
                                                      l->ffn_gate_shexp->abs_offset,
                                                      DS4_N_EMBD,
                                                      DS4_N_FF_EXP,
                                                      x,
                                                      il,
                                                      pos,
                                                      "shared_gate",
                                                      ssd_streaming) != 0;
    if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(up,
                                                              model,
                                                              l->ffn_up_shexp->abs_offset,
                                                              DS4_N_EMBD,
                                                              DS4_N_FF_EXP,
                                                              x,
                                                              il,
                                                              pos,
                                                              "shared_up",
                                                              ssd_streaming) != 0;
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "shared_gate_up",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (ok) ok = ds4_gpu_swiglu_tensor(mid,
                                       gate,
                                       up,
                                       DS4_N_FF_EXP,
                                       0.0f,
                                       1.0f) != 0;
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "shared_swiglu",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    return ok;
}

static bool glm_graph_encode_sparse_ffn_one(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos,
        const ds4_gpu_tensor    *ffn_norm,
        const ds4_gpu_tensor    *after_attn,
        ds4_gpu_tensor          *next,
        ds4_gpu_tensor          *ffn_gate,
        ds4_gpu_tensor          *ffn_up,
        ds4_gpu_tensor          *ffn_mid,
        ds4_gpu_tensor          *ffn_out,
        ds4_gpu_tensor          *ffn_sum,
        ds4_gpu_tensor          *tmp,
        bool                     stage_profile,
        double                  *stage_t0) {
    uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
    uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
    uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;
    (void)tensor_expert_bytes(model, l->ffn_gate_exps, 0, &gate_in, &gate_out, &gate_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_up_exps, 0, &up_in, &up_out, &up_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_down_exps, 0, &down_in, &down_out, &down_row_bytes);
    (void)gate_in;
    (void)up_in;
    (void)down_in;

    bool ok = ds4_gpu_matmul_f32_tensor(g->router_logits,
                                        model->map,
                                        model->size,
                                        l->ffn_gate_inp->abs_offset,
                                        DS4_N_EMBD,
                                        DS4_N_EXPERT,
                                        ffn_norm,
                                        1) != 0;
    if (ok) ok = ds4_gpu_glm_router_select_tensor(g->router_selected,
                                                  g->router_weights,
                                                  g->router_probs,
                                                  model->map,
                                                  model->size,
                                                  l->ffn_exp_probs_b->abs_offset,
                                                  g->router_logits,
                                                  DS4_N_EXPERT,
                                                  DS4_N_EXPERT_USED,
                                                  DS4_EXPERT_WEIGHT_SCALE) != 0;
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "router",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (ok) ok = glm_graph_profile_router_selection(g, l, il, pos);
    const bool resident_decode_layer =
        g->ssd_streaming && glm_stream_resident_decode_layer_enabled(l, il);
    const bool generic_streaming_selected_cache =
        g->ssd_streaming &&
        !resident_decode_layer &&
        glm_graph_layer_uses_generic_routed_moe(l);
    const bool uniform_streaming_selected_cache =
        g->ssd_streaming &&
        !resident_decode_layer &&
        l->ffn_gate_exps->type == l->ffn_up_exps->type &&
        l->ffn_gate_exps->type == l->ffn_down_exps->type &&
        (l->ffn_gate_exps->type == DS4_TENSOR_Q2_K ||
         l->ffn_gate_exps->type == DS4_TENSOR_Q4_K);
    const bool streaming_selected_cache =
        generic_streaming_selected_cache ||
        uniform_streaming_selected_cache;
    const bool shared_first = streaming_selected_cache;
    metal_graph_selected_async_load async_load = {0};
    bool async_load_started = false;
    const bool async_profile =
        streaming_selected_cache &&
        glm_graph_streaming_async_profile_enabled();
    if (async_profile) glm_graph_streaming_async_profile_register();
    const double stream_total_t0 =
        async_profile ? glm_graph_streaming_async_profile_ms() : 0.0;
    double stream_t0 = stream_total_t0;
    bool async_path_profiled = false;
    if (ok && streaming_selected_cache) {
        const ds4_gpu_stream_expert_table table = {
            .model_map = model->map,
            .model_size = model->size,
            .layer = il,
            .n_total_expert = DS4_N_EXPERT,
            .gate_offset = l->ffn_gate_exps->abs_offset,
            .up_offset = l->ffn_up_exps->abs_offset,
            .down_offset = l->ffn_down_exps->abs_offset,
            .gate_expert_bytes = gate_out * gate_row_bytes,
            .down_expert_bytes = down_out * down_row_bytes,
        };
        const bool async_selected_load =
#ifdef DS4_ROCM_BUILD
            streaming_selected_cache &&
#else
            glm_graph_layer_uses_generic_routed_moe(l) &&
#endif
            glm_graph_use_streaming_selected_async_load(g);
        async_path_profiled = false;
        uint64_t selected_event = 0;
        if (async_selected_load) {
            if (ds4_gpu_signal_selected_readback_ready(&selected_event) != 0) {
                async_load_started = metal_graph_selected_async_load_start_tensor(
                        &async_load,
                        g->router_selected,
                        model,
                        l,
                        il,
                        selected_event,
                        gate_out * gate_row_bytes,
                        down_out * down_row_bytes);
                async_path_profiled = async_profile && async_load_started;
            }
            if (async_profile) {
                g_glm_streaming_async_profile.async_signal_start_ms +=
                    glm_graph_streaming_async_profile_ms() - stream_t0;
                stream_t0 = glm_graph_streaming_async_profile_ms();
            }
#ifndef DS4_ROCM_BUILD
            if (ok && async_load_started) {
                ok = ds4_gpu_flush_commands() != 0;
            }
#endif
            if (async_profile) {
                g_glm_streaming_async_profile.async_flush_router_ms +=
                    glm_graph_streaming_async_profile_ms() - stream_t0;
                stream_t0 = glm_graph_streaming_async_profile_ms();
            }
        }
        if (!async_load_started) {
            if (async_selected_load && selected_event != 0) {
                ok = ds4_gpu_wait_selected_readback_ready(
                        selected_event,
                        "selected-id sync expert load fallback") != 0;
            }
            if (ok) {
                ok = ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
                        &table,
                        g->router_selected,
                        DS4_N_EXPERT_USED) != 0;
            }
            if (async_profile) {
                g_glm_streaming_async_profile.sync_early_load_ms +=
                    glm_graph_streaming_async_profile_ms() - stream_t0;
                stream_t0 = glm_graph_streaming_async_profile_ms();
            }
        }
    }
    if (ok && shared_first) {
        ok = glm_graph_encode_shared_swiglu_one(ffn_mid,
                                                ffn_gate,
                                                ffn_up,
                                                model,
                                                l,
                                                il,
                                                pos,
                                                ffn_norm,
                                                g->ssd_streaming,
                                                stage_profile,
                                                stage_t0);
        if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_sum,
                                                                  model,
                                                                  l->ffn_down_shexp->abs_offset,
                                                                  DS4_N_FF_EXP,
                                                                  DS4_N_EMBD,
                                                                  ffn_mid,
                                                                  il,
                                                                  pos,
                                                                  "shared_down",
                                                                  g->ssd_streaming) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "shared_down",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
    }
    if (async_profile) {
        const double now_ms = glm_graph_streaming_async_profile_ms();
        if (async_path_profiled) {
            g_glm_streaming_async_profile.async_shared_ms +=
                now_ms - stream_t0;
        } else {
            g_glm_streaming_async_profile.sync_shared_ms +=
                now_ms - stream_t0;
        }
        stream_t0 = now_ms;
    }
    if (async_load_started) {
        bool flush_ok = true;
#ifndef DS4_ROCM_BUILD
        flush_ok = ds4_gpu_flush_commands() != 0;
#endif
        if (async_profile) {
            g_glm_streaming_async_profile.async_flush_shared_ms +=
                glm_graph_streaming_async_profile_ms() - stream_t0;
            stream_t0 = glm_graph_streaming_async_profile_ms();
        }
        const bool finish_ok = metal_graph_selected_async_load_finish(&async_load);
        ok = ok && flush_ok && finish_ok;
        if (async_profile) {
            g_glm_streaming_async_profile.async_finish_ms +=
                glm_graph_streaming_async_profile_ms() - stream_t0;
            stream_t0 = glm_graph_streaming_async_profile_ms();
        }
    }
    /* 50/50 TP: this rank's routed partial goes straight into the
     * slab out slot, the gate exchanges it with the peer's half, and the
     * commutative add rebuilds the full routed output on both ranks
     * bit-identically.  The shared expert and everything else stay
     * replicated, so no other exchange is needed. */
    const bool tp_split_ffn = g->tp_world == 2 && g->tp_out && g->tp_in;
    const uint32_t tp_ffn_slot = il * DS4_TP_GATES_PER_LAYER + DS4_TP_GATE_FFN;
    ds4_gpu_tensor *routed_dst = tp_split_ffn ? g->tp_out[tp_ffn_slot] : ffn_out;
    if (ok && tp_split_ffn && g->ssd_streaming) {
        fprintf(stderr, "ds4: GLM tensor parallelism requires resident weights\n");
        ok = false;
    }
    if (!ok && tp_split_ffn && getenv("DS4_GLM_TP_DEBUG")) {
        fprintf(stderr, "ds4: glm sparse ffn: failed before routed dispatch (layer %u)\n", il);
    }
    if (ok && !(glm_decode_ablate_mask() & DS4_GLM_ABLATE_ROUTED)) {
        ok = glm_graph_routed_moe_one_dispatch(
            g,
            model,
            l,
            il,
            routed_dst,
            ffn_mid,
            gate_out * gate_row_bytes,
            gate_row_bytes,
            up_out * up_row_bytes,
            up_row_bytes,
            down_out * down_row_bytes,
            down_row_bytes,
            g->router_selected,
            g->router_weights,
            ffn_norm,
            resident_decode_layer) != 0;
    }
    if (ok && tp_split_ffn) {
        ok = ds4_gpu_tp_gate_encode(il, DS4_TP_GATE_FFN) != 0;
        if (ok) ok = ds4_gpu_add_tensor(ffn_out,
                                        g->tp_out[tp_ffn_slot],
                                        g->tp_in[tp_ffn_slot],
                                        DS4_N_EMBD) != 0;
        if (!ok) fprintf(stderr, "ds4: GLM TP gate/combine failed (layer %u)\n", il);
    } else if (!ok && tp_split_ffn) {
        fprintf(stderr, "ds4: GLM TP routed dispatch failed before the gate (layer %u)\n", il);
    }
    if (async_profile) {
        const double now_ms = glm_graph_streaming_async_profile_ms();
        if (async_path_profiled) {
            g_glm_streaming_async_profile.async_routed_ms +=
                now_ms - stream_t0;
        } else {
            g_glm_streaming_async_profile.sync_routed_ms +=
                now_ms - stream_t0;
        }
        stream_t0 = now_ms;
    }
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "routed_moe",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (ok && !shared_first &&
        !(glm_decode_ablate_mask() & DS4_GLM_ABLATE_SHARED)) {
        ok = glm_graph_encode_shared_swiglu_one(ffn_mid,
                                                ffn_gate,
                                                ffn_up,
                                                model,
                                                l,
                                                il,
                                                pos,
                                                ffn_norm,
                                                g->ssd_streaming,
                                                stage_profile,
                                                stage_t0);
        if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_sum,
                                                                  model,
                                                                  l->ffn_down_shexp->abs_offset,
                                                                  DS4_N_FF_EXP,
                                                                  DS4_N_EMBD,
                                                                  ffn_mid,
                                                                  il,
                                                                  pos,
                                                                  "shared_down",
                                                                  g->ssd_streaming) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "shared_down",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
    }
    if (ok && !glm_graph_disable_add3_residual()) {
        ok = ds4_gpu_add3_tensor(next,
                                 after_attn,
                                 ffn_out,
                                 ffn_sum,
                                 DS4_N_EMBD) != 0;
    } else if (ok) {
        ok = ds4_gpu_add_tensor(tmp,
                                ffn_out,
                                ffn_sum,
                                DS4_N_EMBD) != 0;
        if (ok) ok = ds4_gpu_add_tensor(next,
                                        after_attn,
                                        tmp,
                                        DS4_N_EMBD) != 0;
    }
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "residual",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (async_profile) {
        const double now_ms = glm_graph_streaming_async_profile_ms();
        if (async_path_profiled) {
            g_glm_streaming_async_profile.async_calls++;
            g_glm_streaming_async_profile.async_post_ms += now_ms - stream_t0;
            g_glm_streaming_async_profile.async_total_ms +=
                now_ms - stream_total_t0;
        } else {
            g_glm_streaming_async_profile.sync_calls++;
            g_glm_streaming_async_profile.sync_post_ms += now_ms - stream_t0;
            g_glm_streaming_async_profile.sync_total_ms +=
                now_ms - stream_total_t0;
        }
    }
    return ok;
}

static bool glm_graph_encode_ffn_one_normed_from(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos,
        const ds4_gpu_tensor    *ffn_norm,
        const ds4_gpu_tensor    *after_attn,
        ds4_gpu_tensor          *next,
        ds4_gpu_tensor          *ffn_gate,
        ds4_gpu_tensor          *ffn_up,
        ds4_gpu_tensor          *ffn_mid,
        ds4_gpu_tensor          *ffn_out,
        ds4_gpu_tensor          *ffn_sum,
        ds4_gpu_tensor          *tmp,
        bool                     stage_profile,
        double                  *stage_t0) {
    if (!g || !model || !l || !ffn_norm || !after_attn || !next ||
        !ffn_gate || !ffn_up || !ffn_mid || !ffn_out ||
        !ffn_sum || !tmp) {
        return false;
    }

    if (il < DS4_N_LEADING_DENSE) {
        const uint64_t hidden = l->ffn_gate->dim[1];
        const bool can_fuse_gate_up =
            glm_graph_weights_are_q8_0(model,
                                       l->ffn_gate->abs_offset,
                                       l->ffn_up->abs_offset);
        const bool fused_gate_up = can_fuse_gate_up &&
            ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
                    ffn_gate,
                    ffn_up,
                    ffn_mid,
                    model->map,
                    model->size,
                    l->ffn_gate->abs_offset,
                    l->ffn_up->abs_offset,
                    DS4_N_EMBD,
                    hidden,
                    ffn_norm,
                    0.0f) != 0;
        bool ok = fused_gate_up;
        if (fused_gate_up) {
            ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "dense_gate_up_swiglu",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
        } else {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_gate,
                                                              model,
                                                              l->ffn_gate->abs_offset,
                                                              DS4_N_EMBD,
                                                              hidden,
                                                              ffn_norm,
                                                              il,
                                                              pos,
                                                              "dense_gate",
                                                              g->ssd_streaming) != 0;
            if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_up,
                                                                      model,
                                                                      l->ffn_up->abs_offset,
                                                                      DS4_N_EMBD,
                                                                      hidden,
                                                                      ffn_norm,
                                                                      il,
                                                                      pos,
                                                                      "dense_up",
                                                                      g->ssd_streaming) != 0;
            if (ok) ok = glm_graph_profile_stage(stage_profile,
                                                 "glm_decode_ffn",
                                                 "dense_gate_up",
                                                 il,
                                                 pos,
                                                 1,
                                                 stage_t0);
            if (ok) ok = ds4_gpu_swiglu_tensor(ffn_mid,
                                               ffn_gate,
                                               ffn_up,
                                               (uint32_t)hidden,
                                               0.0f,
                                               1.0f) != 0;
            if (ok) ok = glm_graph_profile_stage(stage_profile,
                                                 "glm_decode_ffn",
                                                 "dense_swiglu",
                                                 il,
                                                 pos,
                                                 1,
                                                 stage_t0);
        }
        if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_out,
                                                                  model,
                                                                  l->ffn_down->abs_offset,
                                                                  hidden,
                                                                  DS4_N_EMBD,
                                                                  ffn_mid,
                                                                  il,
                                                                  pos,
                                                                  "dense_down",
                                                                  g->ssd_streaming) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "dense_down",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
        if (ok) ok = ds4_gpu_add_tensor(next,
                                        after_attn,
                                        ffn_out,
                                        DS4_N_EMBD) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "residual",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
        return ok;
    }

    return glm_graph_encode_sparse_ffn_one(g,
                                           model,
                                           l,
                                           il,
                                           pos,
                                           ffn_norm,
                                           after_attn,
                                           next,
                                           ffn_gate,
                                           ffn_up,
                                           ffn_mid,
                                           ffn_out,
                                           ffn_sum,
                                           tmp,
                                           stage_profile,
                                           stage_t0);
}

static bool glm_graph_encode_ffn_one_from(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos,
        const ds4_gpu_tensor    *after_attn,
        ds4_gpu_tensor          *next,
        ds4_gpu_tensor          *ffn_norm,
        ds4_gpu_tensor          *ffn_gate,
        ds4_gpu_tensor          *ffn_up,
        ds4_gpu_tensor          *ffn_mid,
        ds4_gpu_tensor          *ffn_out,
        ds4_gpu_tensor          *ffn_sum,
        ds4_gpu_tensor          *tmp,
        bool                     stage_profile,
        double                  *stage_t0) {
    if (!g || !model || !l || !after_attn || !next ||
        !ffn_norm || !ffn_gate || !ffn_up || !ffn_mid || !ffn_out ||
        !ffn_sum || !tmp) {
        return false;
    }

    bool ok = ds4_gpu_rms_norm_weight_tensor(ffn_norm,
                                             after_attn,
                                             model->map,
                                             model->size,
                                             l->ffn_norm->abs_offset,
                                             DS4_N_EMBD,
                                             DS4_RMS_EPS) != 0;
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "ffn_norm",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (!ok) return false;

    return glm_graph_encode_ffn_one_normed_from(g,
                                                model,
                                                l,
                                                il,
                                                pos,
                                                ffn_norm,
                                                after_attn,
                                                next,
                                                ffn_gate,
                                                ffn_up,
                                                ffn_mid,
                                                ffn_out,
                                                ffn_sum,
                                                tmp,
                                                stage_profile,
                                                stage_t0);
}

static ds4_gpu_tensor *glm_graph_tensor_row_view_strided(
        ds4_gpu_tensor *base,
        uint32_t        row,
        uint64_t        stride_values,
        uint64_t        row_values) {
    return ds4_gpu_tensor_view(base,
                               (uint64_t)row * stride_values * sizeof(float),
                               row_values * sizeof(float));
}

static uint32_t glm_graph_q8_stripe_tokens(void) {
    return 2048u;
}

static bool glm_graph_flash_attention_prefill_enabled(void) {
    return getenv("DS4_GLM_DISABLE_FLASH_PREFILL") == NULL;
}

static uint32_t glm_graph_flash_attention_prefill_min_tokens(void) {
    return 24u;
}

static bool glm_graph_use_flash_attention_prefill(uint32_t n_tokens) {
    return glm_graph_flash_attention_prefill_enabled() &&
           n_tokens >= glm_graph_flash_attention_prefill_min_tokens();
}

static bool glm_graph_use_flash_attention_staged_kv(
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_len) {
    return pos0 == 0 &&
           n_tokens == cache_len;
}

static bool glm_graph_force_indexed_decode(void) {
    return false;
}

static bool glm_graph_disable_indexed_decode(void) {
    return false;
}

static bool glm_graph_decode_uses_indexed_attention(const ds4_glm_gpu_graph *g,
                                                    uint32_t                 pos,
                                                    const float             *logits_out) {
    return g && g->compact_cache_cap != 0 &&
           (!g->full_kv_cache ||
            pos >= g->ctx_cap ||
            glm_graph_force_indexed_decode() ||
            (logits_out != NULL && !glm_graph_disable_indexed_decode()));
}

static bool glm_graph_decode_updates_dense_cache(const ds4_glm_gpu_graph *g,
                                                 uint32_t                 pos,
                                                 const float             *logits_out) {
    return g && pos < g->ctx_cap &&
           !glm_graph_decode_uses_indexed_attention(g, pos, logits_out);
}

static bool glm_graph_indexed_prefill_scalar_kernels(void) {
    return false;
}

static bool glm_graph_indexed_prefill_scalar_indexer(void) {
    return false;
}

static bool glm_graph_indexed_prefill_batch_indexer(void) {
    return true;
}

static bool glm_graph_indexed_prefill_scalar_attn(void) {
    return false;
}

static bool glm_graph_indexed_prefill_batch_qk_low(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_attn_kernel(void) {
    return true;
}

static uint32_t glm_graph_indexed_prefill_batch_attn_slice_tokens(void) {
    return 2048u;
}

static bool glm_graph_indexer_qat(void) {
    return false;
}

static bool glm_graph_indexed_prefill_batch_ffn(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_ffn_norm(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_routed_moe(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_router_select(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_residual(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_f32_rows(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_q8_rows(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_shared_expert(void) {
    return true;
}

static bool glm_graph_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens) {
    if (!out || !model || !x || n_tokens == 0) return false;

    const uint32_t q8_stripe_tokens = glm_graph_q8_stripe_tokens();
    if (n_tokens <= q8_stripe_tokens) {
        return ds4_gpu_matmul_quant_tensor(out,
                                           model->map,
                                           model->size,
                                           weight_offset,
                                           glm_graph_weight_type_for_offset(model, weight_offset),
                                           in_dim,
                                           out_dim,
                                           x,
                                           n_tokens) != 0;
    }
    if (in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float)) {
        return false;
    }

    uint32_t done = 0;
    while (done < n_tokens) {
        uint32_t chunk = n_tokens - done;
        if (chunk > q8_stripe_tokens) chunk = q8_stripe_tokens;
        /*
         * The Q8 prefill TensorOps path needs token counts divisible by 32.
         * For a final chunk like 1736 rows, keep the aligned 1728 rows on that
         * path and leave only the tiny tail to the small-batch kernel.  Avoid
         * splitting small batches where the extra launch would dominate.
         */
        if (chunk >= 256u && chunk == n_tokens - done) {
            const uint32_t tail = chunk & 31u;
            if (tail != 0u && tail <= 16u) chunk -= tail;
        }

        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
                x,
                (uint64_t)done * in_dim * sizeof(float),
                (uint64_t)chunk * in_dim * sizeof(float));
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
                out,
                (uint64_t)done * out_dim * sizeof(float),
                (uint64_t)chunk * out_dim * sizeof(float));
        const bool ok = x_view && out_view &&
            ds4_gpu_matmul_quant_tensor(out_view,
                                        model->map,
                                        model->size,
                                        weight_offset,
                                        glm_graph_weight_type_for_offset(model, weight_offset),
                                        in_dim,
                                        out_dim,
                                        x_view,
                                        chunk) != 0;
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(x_view);
        if (!ok) return false;
        done += chunk;
    }
    return true;
}

static bool glm_graph_matmul_q8_0_rows_scalar(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens) {
    if (!out || !model || !x || n_tokens == 0) return false;
    if (in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float)) {
        return false;
    }

    if (glm_graph_indexed_prefill_batch_q8_rows() &&
        ds4_gpu_matmul_quant_rows_scalar_tensor(out,
                                                model->map,
                                                model->size,
                                                weight_offset,
                                                glm_graph_weight_type_for_offset(model, weight_offset),
                                                in_dim,
                                                out_dim,
                                                x,
                                                n_tokens) != 0) {
        return true;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
                (ds4_gpu_tensor *)x,
                (uint64_t)t * in_dim * sizeof(float),
                in_dim * sizeof(float));
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
                out,
                (uint64_t)t * out_dim * sizeof(float),
                out_dim * sizeof(float));
        const bool ok = x_view && out_view &&
            ds4_gpu_matmul_quant_tensor(out_view,
                                        model->map,
                                        model->size,
                                        weight_offset,
                                        glm_graph_weight_type_for_offset(model, weight_offset),
                                        in_dim,
                                        out_dim,
                                        x_view,
                                        1) != 0;
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(x_view);
        if (!ok) return false;
    }
    return true;
}

static bool glm_graph_matmul_f32_rows_scalar(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens) {
    if (!out || !model || !x || n_tokens == 0) return false;
    if (in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float)) {
        return false;
    }

    if (glm_graph_indexed_prefill_batch_f32_rows()) {
        return ds4_gpu_matmul_f32_tensor(out,
                                         model->map,
                                         model->size,
                                         weight_offset,
                                         in_dim,
                                         out_dim,
                                         x,
                                         n_tokens) != 0;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
                (ds4_gpu_tensor *)x,
                (uint64_t)t * in_dim * sizeof(float),
                in_dim * sizeof(float));
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
                out,
                (uint64_t)t * out_dim * sizeof(float),
                out_dim * sizeof(float));
        const bool ok = x_view && out_view &&
            ds4_gpu_matmul_f32_tensor(out_view,
                                      model->map,
                                      model->size,
                                      weight_offset,
                                      in_dim,
                                      out_dim,
                                      x_view,
                                      1) != 0;
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(x_view);
        if (!ok) return false;
    }
    return true;
}

static bool glm_graph_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const ds4_model      *model,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens,
        float                 clamp) {
    if (!gate || !up || !mid || !model || !x || n_tokens == 0) return false;
    if (!glm_graph_weights_are_q8_0(model, gate_offset, up_offset)) return false;

    const uint32_t q8_stripe_tokens = glm_graph_q8_stripe_tokens();
    if (n_tokens == 1) {
        return ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(gate,
                                                                    up,
                                                                    mid,
                                                                    model->map,
                                                                    model->size,
                                                                    gate_offset,
                                                                    up_offset,
                                                                    in_dim,
                                                                    out_dim,
                                                                    x,
                                                                    clamp) != 0;
    }
    if (n_tokens <= q8_stripe_tokens) {
        return ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(gate,
                                                              up,
                                                              mid,
                                                              model->map,
                                                              model->size,
                                                              gate_offset,
                                                              up_offset,
                                                              in_dim,
                                                              out_dim,
                                                              x,
                                                              n_tokens,
                                                              clamp) != 0;
    }
    if (in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float)) {
        return false;
    }

    uint32_t done = 0;
    while (done < n_tokens) {
        uint32_t chunk = n_tokens - done;
        if (chunk > q8_stripe_tokens) chunk = q8_stripe_tokens;

        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
                x,
                (uint64_t)done * in_dim * sizeof(float),
                (uint64_t)chunk * in_dim * sizeof(float));
        ds4_gpu_tensor *gate_view = ds4_gpu_tensor_view(
                gate,
                (uint64_t)done * out_dim * sizeof(float),
                (uint64_t)chunk * out_dim * sizeof(float));
        ds4_gpu_tensor *up_view = ds4_gpu_tensor_view(
                up,
                (uint64_t)done * out_dim * sizeof(float),
                (uint64_t)chunk * out_dim * sizeof(float));
        ds4_gpu_tensor *mid_view = ds4_gpu_tensor_view(
                mid,
                (uint64_t)done * out_dim * sizeof(float),
                (uint64_t)chunk * out_dim * sizeof(float));
        const bool ok = x_view && gate_view && up_view && mid_view &&
            ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(gate_view,
                                                           up_view,
                                                           mid_view,
                                                           model->map,
                                                           model->size,
                                                           gate_offset,
                                                           up_offset,
                                                           in_dim,
                                                           out_dim,
                                                           x_view,
                                                           chunk,
                                                           clamp) != 0;
        ds4_gpu_tensor_free(mid_view);
        ds4_gpu_tensor_free(up_view);
        ds4_gpu_tensor_free(gate_view);
        ds4_gpu_tensor_free(x_view);
        if (!ok) return false;
        done += chunk;
    }
    return true;
}

static bool glm_graph_indexed_prefill_grouped_moe_default(
        const ds4_glm_gpu_graph *g) {
    return g &&
           !g->quality;
}

static uint32_t glm_graph_streaming_prefill_cache_seed_k(
        const ds4_glm_gpu_graph *g) {
    const bool enabled =
        glm_graph_env_present("DS4_ROCM_ENABLE_STREAMING_PREFILL_CACHE_SEED",
                              "DS4_METAL_ENABLE_STREAMING_PREFILL_CACHE_SEED");
    if (!g ||
        !g->ssd_streaming ||
        !enabled) {
        return 0;
    }

    uint32_t k = 1;
    const char *env = glm_graph_env_value("DS4_ROCM_STREAMING_PREFILL_CACHE_SEED_K",
                                          "DS4_METAL_STREAMING_PREFILL_CACHE_SEED_K");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0') {
            if (v == 0) return 0;
            k = v > DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS ?
                DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS : (uint32_t)v;
        }
    }
    return k;
}

static bool glm_graph_streaming_prefill_cache_seed_enabled(
        const ds4_glm_gpu_graph *g) {
    return glm_graph_streaming_prefill_cache_seed_k(g) != 0;
}

static void glm_graph_reset_prefill_seed_capture(ds4_glm_gpu_graph *g) {
    if (!g) return;
    g->prefill_seed_tokens = 0;
    memset(g->prefill_seed_layer_captured,
           0,
           sizeof(g->prefill_seed_layer_captured));
}

static bool glm_graph_streaming_expert_cache_seed_layer_expected(
        const ds4_glm_gpu_graph *g,
        const ds4_weights       *weights,
        const ds4_layer_weights *layer,
        uint32_t                 il) {
    if (!g ||
        !g->ssd_streaming ||
        g->quality ||
        !weights ||
        !layer ||
        !layer->ffn_gate_exps ||
        !layer->ffn_up_exps ||
        !layer->ffn_down_exps) {
        return false;
    }
    if (glm_stream_resident_decode_layer_enabled(layer, il)) return false;
    return glm_stream_selected_expert_cache_supported(layer, il) ||
           glm_stream_expert_cache_addr_layout_supported(weights, layer, il);
}

static bool glm_graph_capture_prefill_seed_router_selected(
        ds4_glm_gpu_graph *g,
        uint32_t           il,
        uint32_t           n_tokens) {
    uint32_t k = glm_graph_streaming_prefill_cache_seed_k(g);
    if (k == 0) return true;
    if (!g->prefill_seed_router_selected ||
        !g->batch_router_selected ||
        il >= DS4_N_LAYER ||
        il >= DS4_MAX_LAYER ||
        n_tokens == 0 ||
        DS4_N_EXPERT_USED == 0 ||
        DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return false;
    }
    if (k > n_tokens) k = n_tokens;

    const uint64_t bytes = (uint64_t)k * DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t src_off = (uint64_t)(n_tokens - k) *
                             DS4_N_EXPERT_USED * sizeof(int32_t);
    const uint64_t dst_off = (uint64_t)il *
                             DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                             DS4_N_EXPERT_USED * sizeof(int32_t);
    if (ds4_gpu_tensor_copy(g->prefill_seed_router_selected,
                            dst_off,
                            g->batch_router_selected,
                            src_off,
                            bytes) == 0) {
        return false;
    }
    g->prefill_seed_tokens = k;
    g->prefill_seed_layer_captured[il] = true;
    return true;
}

static bool glm_graph_seed_streaming_expert_cache_from_prefill(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    if (!glm_graph_streaming_prefill_cache_seed_enabled(g)) return true;
    const uint32_t seed_tokens = g ? g->prefill_seed_tokens : 0;
    if (seed_tokens == 0) return true;
    if (!g ||
        !model ||
        !weights ||
        !g->prefill_seed_router_selected ||
        seed_tokens > DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS ||
        DS4_N_LAYER > DS4_MAX_LAYER ||
        DS4_N_EXPERT_USED == 0 ||
        DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED) {
        return false;
    }
    bool any_captured = false;
    for (uint32_t il = g->layer_start;
         il <= g->layer_end && il < DS4_N_LAYER && il < DS4_MAX_LAYER;
         il++) {
        if (g->prefill_seed_layer_captured[il]) {
            any_captured = true;
            break;
        }
    }
    if (!any_captured) return true;

    int32_t selected[DS4_MAX_LAYER *
                     DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                     DS4_MAX_EXPERT_USED];
    const uint64_t bytes = (uint64_t)DS4_N_LAYER *
                           DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS *
                           DS4_N_EXPERT_USED * sizeof(selected[0]);
    if (ds4_gpu_tensor_read(g->prefill_seed_router_selected,
                            0,
                            selected,
                            bytes) == 0) {
        return false;
    }

    const bool profile =
        glm_graph_env_present("DS4_ROCM_STREAMING_PREFILL_CACHE_SEED_PROFILE",
                              "DS4_METAL_STREAMING_PREFILL_CACHE_SEED_PROFILE");
    const double t0 = profile ? now_sec() : 0.0;
    uint32_t seeded_layers = 0;
    uint32_t seeded_rows = 0;
    for (uint32_t il = g->layer_start;
         il <= g->layer_end && il < DS4_N_LAYER;
         il++) {
        if (il >= DS4_MAX_LAYER || !g->prefill_seed_layer_captured[il]) {
            continue;
        }
        const ds4_layer_weights *layer = &weights->layer[il];
        if (!glm_graph_streaming_expert_cache_seed_layer_expected(g,
                                                                  weights,
                                                                  layer,
                                                                  il)) {
            continue;
        }

        uint64_t gate_expert_bytes = 0;
        uint64_t down_expert_bytes = 0;
        if (!streaming_layer_gate_down_expert_bytes(layer,
                                                    &gate_expert_bytes,
                                                    &down_expert_bytes)) {
            fprintf(stderr,
                    "ds4: GLM prefill expert-cache seed byte size overflow at layer %u\n",
                    il);
            return false;
        }

        const ds4_gpu_stream_expert_table table =
            graph_stream_expert_table_make(model,
                                           layer,
                                           il,
                                           gate_expert_bytes,
                                           down_expert_bytes);
        for (uint32_t row = 0; row < seed_tokens; row++) {
            const size_t sel_off = ((size_t)il *
                                    DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS +
                                    row) * DS4_N_EXPERT_USED;
            if (ds4_gpu_stream_expert_cache_seed_selected(
                        &table,
                        selected + sel_off,
                        DS4_N_EXPERT_USED) == 0) {
                return false;
            }
            seeded_rows++;
        }
        seeded_layers++;
    }
    if (profile) {
        fprintf(stderr,
                "ds4: GLM streaming prefill expert-cache seed k=%u layers=%u rows=%u time=%.3f ms\n",
                seed_tokens,
                seeded_layers,
                seeded_rows,
                (now_sec() - t0) * 1000.0);
    }
    return true;
}

static bool glm_graph_seed_streaming_expert_cache_from_full_layer(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_weights       *weights,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 n_tokens,
        uint64_t                 gate_expert_bytes,
        uint64_t                 down_expert_bytes,
        bool                     full_layer_prefill) {
#ifdef DS4_ROCM_BUILD
    uint32_t seed_tokens = glm_graph_streaming_prefill_cache_seed_k(g);
    if (seed_tokens == 0) return true;
    if (!full_layer_prefill ||
        !model ||
        !weights ||
        !layer ||
        !g ||
        !g->batch_router_selected ||
        n_tokens == 0 ||
        il >= DS4_N_LAYER ||
        il >= DS4_MAX_LAYER ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        !g->prefill_seed_layer_captured[il] ||
        !glm_graph_streaming_expert_cache_seed_layer_expected(g,
                                                              weights,
                                                              layer,
                                                              il)) {
        return true;
    }
    if (seed_tokens > n_tokens) seed_tokens = n_tokens;
    const ds4_gpu_stream_expert_table table =
        graph_stream_expert_table_make(model,
                                       layer,
                                       il,
                                       gate_expert_bytes,
                                       down_expert_bytes);
    if (ds4_gpu_stream_expert_cache_seed_from_layer_selected(
                &table,
                g->batch_router_selected,
                n_tokens,
                seed_tokens,
                DS4_N_EXPERT_USED) != 0) {
        g->prefill_seed_layer_captured[il] = false;
        return true;
    }

    static bool warned = false;
    if (!warned) {
        fprintf(stderr,
                "ds4: GLM ROCm full-layer prefill expert-cache seed skipped; "
                "falling back to end-of-prefill selected seed\n");
        warned = true;
    }
    return true;
#else
    (void)g;
    (void)model;
    (void)weights;
    (void)layer;
    (void)il;
    (void)n_tokens;
    (void)gate_expert_bytes;
    (void)down_expert_bytes;
    (void)full_layer_prefill;
    return true;
#endif
}

static bool glm_graph_disable_add3_residual(void);

static bool glm_graph_encode_sparse_ffn_indexed_batch_routed_moe(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos0,
        const ds4_gpu_tensor    *after_attn,
        ds4_gpu_tensor          *next,
        uint32_t                 n_tokens,
        bool                     stage_profile,
        bool                     stage_sync,
        double                  *stage_t0) {
    if (!g || !model || !l || !after_attn || !next ||
        !g->batch_ffn_norm ||
        !g->batch_router_logits ||
        !g->batch_router_probs ||
        !g->batch_router_selected ||
        !g->batch_router_weights ||
        !g->batch_ffn_out ||
        !g->batch_ffn_mid ||
        n_tokens <= 1 ||
        il < DS4_N_LEADING_DENSE ||
        g->ffn_mid_elems > UINT32_MAX) {
        return false;
    }

    uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
    uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
    uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;
    (void)tensor_expert_bytes(model, l->ffn_gate_exps, 0, &gate_in, &gate_out, &gate_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_up_exps, 0, &up_in, &up_out, &up_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_down_exps, 0, &down_in, &down_out, &down_row_bytes);
    (void)gate_in;
    (void)up_in;
    (void)down_in;

    bool ok = glm_graph_matmul_f32_rows_scalar(g->batch_router_logits,
                                               model,
                                               l->ffn_gate_inp->abs_offset,
                                               DS4_N_EMBD,
                                               DS4_N_EXPERT,
                                               g->batch_ffn_norm,
                                               n_tokens);
    const bool use_batch_router_select =
        glm_graph_indexed_prefill_batch_router_select();
    if (ok && use_batch_router_select) {
        ok = ds4_gpu_glm_router_select_batch_tensor(g->batch_router_selected,
                                                    g->batch_router_weights,
                                                    g->batch_router_probs,
                                                    model->map,
                                                    model->size,
                                                    l->ffn_exp_probs_b->abs_offset,
                                                    g->batch_router_logits,
                                                    DS4_N_EXPERT,
                                                    DS4_N_EXPERT_USED,
                                                    DS4_EXPERT_WEIGHT_SCALE,
                                                    n_tokens) != 0;
    }
    for (uint32_t t = 0; ok && !use_batch_router_select && t < n_tokens; t++) {
        ds4_gpu_tensor *logits_view =
            glm_graph_tensor_row_view_strided(g->batch_router_logits,
                                              t,
                                              DS4_N_EXPERT,
                                              DS4_N_EXPERT);
        ds4_gpu_tensor *probs_view =
            glm_graph_tensor_row_view_strided(g->batch_router_probs,
                                              t,
                                              DS4_N_EXPERT,
                                              DS4_N_EXPERT);
        ds4_gpu_tensor *selected_view =
            ds4_gpu_tensor_view(g->batch_router_selected,
                                (uint64_t)t * DS4_N_EXPERT_USED * sizeof(int32_t),
                                (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
        ds4_gpu_tensor *weights_view =
            glm_graph_tensor_row_view_strided(g->batch_router_weights,
                                              t,
                                              DS4_N_EXPERT_USED,
                                              DS4_N_EXPERT_USED);
        ok = logits_view && probs_view && selected_view && weights_view;
        if (ok) {
            ok = ds4_gpu_glm_router_select_tensor(selected_view,
                                                  weights_view,
                                                  probs_view,
                                                  model->map,
                                                  model->size,
                                                  l->ffn_exp_probs_b->abs_offset,
                                                  logits_view,
                                                  DS4_N_EXPERT,
                                                  DS4_N_EXPERT_USED,
                                                  DS4_EXPERT_WEIGHT_SCALE) != 0;
        }
        ds4_gpu_tensor_free(weights_view);
        ds4_gpu_tensor_free(selected_view);
        ds4_gpu_tensor_free(probs_view);
        ds4_gpu_tensor_free(logits_view);
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_indexed_ffn",
                                                  "router",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok) ok = glm_graph_profile_router_selection_batch(g,
                                                          l,
                                                          il,
                                                          pos0,
                                                          n_tokens);
    if (ok) ok = glm_graph_capture_prefill_seed_router_selected(g,
                                                                il,
                                                                n_tokens);
    metal_graph_debug_dump_tensor("glm_indexed_router_logits",
                                  g->batch_router_logits,
                                  (uint64_t)n_tokens * DS4_N_EXPERT,
                                  il,
                                  pos0);
    metal_graph_debug_dump_i32_tensor("glm_indexed_router_selected",
                                      g->batch_router_selected,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED,
                                      il,
                                      pos0);
    metal_graph_debug_dump_tensor("glm_indexed_router_weights",
                                  g->batch_router_weights,
                                  (uint64_t)n_tokens * DS4_N_EXPERT_USED,
                                  il,
                                  pos0);

    const bool tp_batch_split_ffn2 = g->tp_world == 2;
    if (ok && tp_batch_split_ffn2) {
        ok = glm_graph_tp_batch_bounce_ready(g, n_tokens);
    }
    if (ok) {
        const bool use_grouped_moe =
            glm_graph_indexed_prefill_grouped_moe_default(g);
        ok = glm_graph_routed_moe_batch_dispatch(
                g,
                model,
                l,
                il,
                tp_batch_split_ffn2 ? g->tp_bounce_out : g->batch_ffn_out,
                g->batch_ffn_mid,
                gate_out * gate_row_bytes,
                gate_row_bytes,
                up_out * up_row_bytes,
                up_row_bytes,
                down_out * down_row_bytes,
                down_row_bytes,
                g->batch_router_selected,
                g->batch_router_weights,
                g->batch_ffn_norm,
                n_tokens,
                (uint32_t)g->ffn_mid_elems,
                false,
                !use_grouped_moe) != 0;
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_indexed_ffn",
                                                  "routed_moe",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    metal_graph_debug_dump_tensor("glm_indexed_routed_out",
                                  g->batch_ffn_out,
                                  (uint64_t)n_tokens * DS4_N_EMBD,
                                  il,
                                  pos0);

    const bool use_batch_residual =
        glm_graph_indexed_prefill_batch_residual();
    const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
    if (use_batch_residual && residual_elems > UINT32_MAX) return false;

    bool shared_expert_done = false;
    if (ok &&
        use_batch_residual &&
        glm_graph_indexed_prefill_batch_shared_expert() &&
        g->batch_ffn_gate &&
        g->batch_ffn_up &&
        g->batch_shared_mid &&
        glm_graph_weights_are_q8_0(model,
                                   l->ffn_gate_shexp->abs_offset,
                                   l->ffn_up_shexp->abs_offset) &&
        ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                g->batch_ffn_gate,
                g->batch_ffn_up,
                g->batch_shared_mid,
                model->map,
                model->size,
                l->ffn_gate_shexp->abs_offset,
                l->ffn_up_shexp->abs_offset,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                g->batch_ffn_norm,
                n_tokens,
                0.0f) != 0) {
        shared_expert_done =
            glm_graph_matmul_q8_0_rows_scalar(g->batch_attn_out,
                                              model,
                                              l->ffn_down_shexp->abs_offset,
                                              DS4_N_FF_EXP,
                                              DS4_N_EMBD,
                                              g->batch_shared_mid,
                                              n_tokens);
    }

    for (uint32_t t = 0; ok && !shared_expert_done && t < n_tokens; t++) {
        ds4_gpu_tensor *ffn_norm_view =
            glm_graph_tensor_row_view_strided(g->batch_ffn_norm,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD);
        ds4_gpu_tensor *shared_out_view = use_batch_residual ?
            glm_graph_tensor_row_view_strided(g->batch_attn_out,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD) :
            NULL;
        ds4_gpu_tensor *after_attn_view = !use_batch_residual ?
            glm_graph_tensor_row_view_strided((ds4_gpu_tensor *)after_attn,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD) :
            NULL;
        ds4_gpu_tensor *routed_out_view = !use_batch_residual ?
            glm_graph_tensor_row_view_strided(g->batch_ffn_out,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD) :
            NULL;
        ds4_gpu_tensor *next_view = !use_batch_residual ?
            glm_graph_tensor_row_view_strided(next,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD) :
            NULL;
        ok = ffn_norm_view &&
             (use_batch_residual ? (shared_out_view != NULL) :
                                   (after_attn_view && routed_out_view && next_view));
        if (ok && glm_graph_weights_are_q8_0(model,
                                             l->ffn_gate_shexp->abs_offset,
                                             l->ffn_up_shexp->abs_offset)) {
            ok = ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
                    g->ffn_gate,
                    g->ffn_up,
                    g->ffn_mid,
                    model->map,
                    model->size,
                    l->ffn_gate_shexp->abs_offset,
                    l->ffn_up_shexp->abs_offset,
                    DS4_N_EMBD,
                    DS4_N_FF_EXP,
                    ffn_norm_view,
                    0.0f) != 0;
        } else if (ok) {
            ok = glm_graph_matmul_q8_0_tensor(g->ffn_gate,
                                              model,
                                              l->ffn_gate_shexp->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_FF_EXP,
                                              ffn_norm_view,
                                              1);
            if (ok) ok = glm_graph_matmul_q8_0_tensor(g->ffn_up,
                                                      model,
                                                      l->ffn_up_shexp->abs_offset,
                                                      DS4_N_EMBD,
                                                      DS4_N_FF_EXP,
                                                      ffn_norm_view,
                                                      1);
            if (ok) ok = ds4_gpu_swiglu_tensor(g->ffn_mid,
                                               g->ffn_gate,
                                               g->ffn_up,
                                               DS4_N_FF_EXP,
                                               0.0f,
                                               1.0f) != 0;
        }
        if (ok) ok = glm_graph_matmul_q8_0_tensor(use_batch_residual ?
                                                      shared_out_view :
                                                      g->ffn_sum,
                                                  model,
                                                  l->ffn_down_shexp->abs_offset,
                                                  DS4_N_FF_EXP,
                                                  DS4_N_EMBD,
                                                  g->ffn_mid,
                                                  1);
        if (ok && !use_batch_residual) {
            ok = ds4_gpu_add_tensor(g->attn_out,
                                    routed_out_view,
                                    g->ffn_sum,
                                    DS4_N_EMBD) != 0;
        }
        if (ok && !use_batch_residual) {
            ok = ds4_gpu_add_tensor(next_view,
                                    after_attn_view,
                                    g->attn_out,
                                    DS4_N_EMBD) != 0;
        }

        ds4_gpu_tensor_free(next_view);
        ds4_gpu_tensor_free(routed_out_view);
        ds4_gpu_tensor_free(shared_out_view);
        ds4_gpu_tensor_free(ffn_norm_view);
        ds4_gpu_tensor_free(after_attn_view);
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_indexed_ffn",
                                                  "shared_expert",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok && use_batch_residual) {
        if (!glm_graph_disable_add3_residual()) {
            ok = ds4_gpu_add3_tensor(next,
                                     after_attn,
                                     g->batch_ffn_out,
                                     g->batch_attn_out,
                                     (uint32_t)residual_elems) != 0;
        } else {
            ok = ds4_gpu_add_tensor(g->batch_heads,
                                    g->batch_ffn_out,
                                    g->batch_attn_out,
                                    (uint32_t)residual_elems) != 0;
            if (ok) ok = ds4_gpu_add_tensor(next,
                                            after_attn,
                                            g->batch_heads,
                                            (uint32_t)residual_elems) != 0;
        }
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_indexed_ffn",
                                                  "residual",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    metal_graph_debug_dump_tensor("glm_indexed_next",
                                  next,
                                  (uint64_t)n_tokens * DS4_N_EMBD,
                                  il,
                                  pos0);

    (void)pos0;
    return ok;
}

static bool glm_graph_upload_tokens(
        ds4_gpu_tensor *out_tokens,
        const int      *tokens,
        uint32_t        n_tokens) {
    if (!out_tokens || !tokens || n_tokens == 0) return false;

    int32_t *ids = xmalloc((size_t)n_tokens * sizeof(ids[0]));
    for (uint32_t i = 0; i < n_tokens; i++) ids[i] = (int32_t)tokens[i];
    const bool ok = ds4_gpu_tensor_write(out_tokens,
                                         0,
                                         ids,
                                         (uint64_t)n_tokens * sizeof(ids[0])) != 0;
    free(ids);
    return ok;
}

static bool glm_graph_disable_add3_residual(void) {
    return false;
}

static bool glm_graph_encode_ffn_batch(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_weights       *weights,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos0,
        ds4_gpu_tensor          *after_attn,
        ds4_gpu_tensor          *next,
        uint32_t                 n_tokens,
        bool                     full_layer_prefill,
        bool                     stage_profile,
        bool                     stage_sync,
        double                  *stage_t0) {
    if (!g || !model || !weights || !l || !after_attn || !next || n_tokens == 0) return false;

    bool ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_ffn_norm,
                                                  after_attn,
                                                  model->map,
                                                  model->size,
                                                  l->ffn_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  n_tokens,
                                                  DS4_RMS_EPS) != 0;
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "ffn_norm",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok) {
        metal_graph_debug_dump_tensor("glm_ffn_norm",
                                      g->batch_ffn_norm,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
    }
    if (!ok) return false;

    if (il < DS4_N_LEADING_DENSE) {
        const uint64_t hidden = l->ffn_gate->dim[1];
        if (hidden == 0 || hidden > UINT32_MAX / n_tokens) return false;
        const uint32_t mid_elems = (uint32_t)(hidden * n_tokens);
        const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
        if (residual_elems > UINT32_MAX) return false;

        const bool fused_gate_up = glm_graph_shared_gate_up_swiglu_q8_0_tensor(
                g->batch_ffn_gate,
                g->batch_ffn_up,
                g->batch_ffn_mid,
                model,
                l->ffn_gate->abs_offset,
                l->ffn_up->abs_offset,
                DS4_N_EMBD,
                hidden,
                g->batch_ffn_norm,
                n_tokens,
                0.0f);
        if (fused_gate_up) {
            ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "dense_gate_up_swiglu",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
        } else {
            ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_gate,
                                              model,
                                              l->ffn_gate->abs_offset,
                                              DS4_N_EMBD,
                                              hidden,
                                              g->batch_ffn_norm,
                                              n_tokens);
            if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_up,
                                                      model,
                                                      l->ffn_up->abs_offset,
                                                      DS4_N_EMBD,
                                                      hidden,
                                                      g->batch_ffn_norm,
                                                      n_tokens);
            if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                          stage_sync,
                                                          "glm_ffn",
                                                          "dense_gate_up",
                                                          il,
                                                          pos0,
                                                          n_tokens,
                                                          stage_t0);
            if (ok) ok = ds4_gpu_swiglu_tensor(g->batch_ffn_mid,
                                               g->batch_ffn_gate,
                                               g->batch_ffn_up,
                                               mid_elems,
                                               0.0f,
                                               1.0f) != 0;
            if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                          stage_sync,
                                                          "glm_ffn",
                                                          "dense_swiglu",
                                                          il,
                                                          pos0,
                                                          n_tokens,
                                                          stage_t0);
        }
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_out,
                                                  model,
                                                  l->ffn_down->abs_offset,
                                                  hidden,
                                                  DS4_N_EMBD,
                                                  g->batch_ffn_mid,
                                                  n_tokens);
        if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                      stage_sync,
                                                      "glm_ffn",
                                                      "dense_down",
                                                      il,
                                                      pos0,
                                                      n_tokens,
                                                      stage_t0);
        if (ok) ok = ds4_gpu_add_tensor(next,
                                        after_attn,
                                        g->batch_ffn_out,
                                        (uint32_t)residual_elems) != 0;
        if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                      stage_sync,
                                                      "glm_ffn",
                                                      "residual",
                                                      il,
                                                      pos0,
                                                      n_tokens,
                                                      stage_t0);
        return ok;
    }

    if (g->ffn_mid_elems > UINT32_MAX ||
        g->dense_hidden_max < DS4_N_FF_EXP ||
        (uint64_t)n_tokens > UINT32_MAX / DS4_N_FF_EXP) {
        return false;
    }
    const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
    if (residual_elems > UINT32_MAX) return false;

    uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
    uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
    uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;
    (void)tensor_expert_bytes(model, l->ffn_gate_exps, 0, &gate_in, &gate_out, &gate_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_up_exps, 0, &up_in, &up_out, &up_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_down_exps, 0, &down_in, &down_out, &down_row_bytes);
    (void)gate_in;
    (void)up_in;
    (void)down_in;

    ok = ds4_gpu_matmul_f32_tensor(g->batch_router_logits,
                                   model->map,
                                   model->size,
                                   l->ffn_gate_inp->abs_offset,
                                   DS4_N_EMBD,
                                   DS4_N_EXPERT,
                                   g->batch_ffn_norm,
                                   n_tokens) != 0;
    if (ok) ok = ds4_gpu_glm_router_select_batch_tensor(g->batch_router_selected,
                                                        g->batch_router_weights,
                                                        g->batch_router_probs,
                                                        model->map,
                                                        model->size,
                                                        l->ffn_exp_probs_b->abs_offset,
                                                        g->batch_router_logits,
                                                        DS4_N_EXPERT,
                                                        DS4_N_EXPERT_USED,
                                                        DS4_EXPERT_WEIGHT_SCALE,
                                                        n_tokens) != 0;
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "router",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok) {
        metal_graph_debug_dump_tensor("glm_ffn_router_logits",
                                      g->batch_router_logits,
                                      (uint64_t)n_tokens * DS4_N_EXPERT,
                                      il,
                                      pos0);
        metal_graph_debug_dump_tensor("glm_ffn_router_probs",
                                      g->batch_router_probs,
                                      (uint64_t)n_tokens * DS4_N_EXPERT,
                                      il,
                                      pos0);
        metal_graph_debug_dump_i32_tensor("glm_ffn_router_selected",
                                          g->batch_router_selected,
                                          (uint64_t)n_tokens * DS4_N_EXPERT_USED,
                                          il,
                                          pos0);
        metal_graph_debug_dump_tensor("glm_ffn_router_weights",
                                      g->batch_router_weights,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED,
                                      il,
                                      pos0);
    }
    if (ok) ok = glm_graph_profile_router_selection_batch(g,
                                                          l,
                                                          il,
                                                          pos0,
                                                          n_tokens);
    const bool tp_batch_split_ffn = g->tp_world == 2;
    if (ok && tp_batch_split_ffn) {
        ok = glm_graph_tp_batch_bounce_ready(g, n_tokens);
    }
    if (ok) ok = glm_graph_capture_prefill_seed_router_selected(g,
                                                                il,
                                                                n_tokens);
    if (ok) ok = glm_graph_seed_streaming_expert_cache_from_full_layer(
            g,
            model,
            weights,
            l,
            il,
            n_tokens,
            gate_out * gate_row_bytes,
            down_out * down_row_bytes,
            full_layer_prefill);
    bool shared_done = false;
#define DS4_GLM_ENCODE_FFN_BATCH_SHARED() do { \
        if (ok) { \
            const bool fused_shared = glm_graph_shared_gate_up_swiglu_q8_0_tensor( \
                    g->batch_ffn_gate, \
                    g->batch_ffn_up, \
                    g->batch_shared_mid, \
                    model, \
                    l->ffn_gate_shexp->abs_offset, \
                    l->ffn_up_shexp->abs_offset, \
                    DS4_N_EMBD, \
                    DS4_N_FF_EXP, \
                    g->batch_ffn_norm, \
                    n_tokens, \
                    0.0f); \
            if (fused_shared) { \
                ok = glm_graph_prefill_stage_boundary(stage_profile, \
                                                      stage_sync, \
                                                      "glm_ffn", \
                                                      "shared_gate_up_swiglu", \
                                                      il, \
                                                      pos0, \
                                                      n_tokens, \
                                                      stage_t0); \
            } else { \
                ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_gate, \
                                                  model, \
                                                  l->ffn_gate_shexp->abs_offset, \
                                                  DS4_N_EMBD, \
                                                  DS4_N_FF_EXP, \
                                                  g->batch_ffn_norm, \
                                                  n_tokens); \
                if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_up, \
                                                          model, \
                                                          l->ffn_up_shexp->abs_offset, \
                                                          DS4_N_EMBD, \
                                                          DS4_N_FF_EXP, \
                                                          g->batch_ffn_norm, \
                                                          n_tokens); \
                if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile, \
                                                              stage_sync, \
                                                              "glm_ffn", \
                                                              "shared_gate_up", \
                                                              il, \
                                                              pos0, \
                                                              n_tokens, \
                                                              stage_t0); \
                if (ok) ok = ds4_gpu_swiglu_tensor(g->batch_shared_mid, \
                                                   g->batch_ffn_gate, \
                                                   g->batch_ffn_up, \
                                                   n_tokens * DS4_N_FF_EXP, \
                                                   0.0f, \
                                                   1.0f) != 0; \
                if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile, \
                                                              stage_sync, \
                                                              "glm_ffn", \
                                                              "shared_swiglu", \
                                                              il, \
                                                              pos0, \
                                                              n_tokens, \
                                                              stage_t0); \
            } \
        } \
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_attn_out, \
                                                  model, \
                                                  l->ffn_down_shexp->abs_offset, \
                                                  DS4_N_FF_EXP, \
                                                  DS4_N_EMBD, \
                                                  g->batch_shared_mid, \
                                                  n_tokens); \
        if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile, \
                                                      stage_sync, \
                                                      "glm_ffn", \
                                                      "shared_down", \
                                                      il, \
                                                      pos0, \
                                                      n_tokens, \
                                                      stage_t0); \
        if (ok) shared_done = true; \
    } while (0)
#ifdef DS4_ROCM_BUILD
    rocm_graph_batch_selected_async_load rocm_batch_selected_async = {0};
    bool rocm_batch_selected_async_started = false;
    const bool rocm_batch_selected_shared_overlap =
        ok &&
        g->ssd_streaming &&
        !g->quality &&
        n_tokens > 1 &&
        !full_layer_prefill &&
        !glm_graph_env_present(
                "DS4_ROCM_DISABLE_GLM_STREAMING_PREFILL_SELECTED_ASYNC_LOAD",
                "DS4_METAL_DISABLE_GLM_STREAMING_PREFILL_SELECTED_ASYNC_LOAD") &&
        !glm_graph_env_present(
                "DS4_ROCM_DISABLE_STREAMING_PREFILL_SELECTED_ASYNC_LOAD",
                "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_ASYNC_LOAD") &&
        glm_graph_stream_prefill_expert_addr_supported(weights, l, il, n_tokens);
    if (rocm_batch_selected_shared_overlap) {
        uint64_t selected_event = 0;
        if (ds4_gpu_signal_selected_readback_ready(&selected_event) != 0) {
            rocm_batch_selected_async_started =
                rocm_graph_batch_selected_async_load_start(
                    &rocm_batch_selected_async,
                    g->batch_router_selected,
                    model,
                    l,
                    il,
                    n_tokens,
                    selected_event,
                    gate_out * gate_row_bytes,
                    down_out * down_row_bytes);
        }
    }
    if (rocm_batch_selected_async_started) {
        DS4_GLM_ENCODE_FFN_BATCH_SHARED();
        const bool finish_ok =
            rocm_graph_batch_selected_async_load_finish(&rocm_batch_selected_async);
        if (!finish_ok) rocm_batch_selected_async_started = false;
    }
#endif
    if (n_tokens <= 8u && (glm_decode_ablate_mask() & DS4_GLM_ABLATE_ROUTED)) { /* ablate: keep the gate */ } else
    if (ok) ok = glm_graph_routed_moe_batch_dispatch(
            g,
            model,
            l,
            il,
            tp_batch_split_ffn ? g->tp_bounce_out : g->batch_ffn_out,
            g->batch_ffn_mid,
            gate_out * gate_row_bytes,
            gate_row_bytes,
            up_out * up_row_bytes,
            up_row_bytes,
            down_out * down_row_bytes,
            down_row_bytes,
            g->batch_router_selected,
            g->batch_router_weights,
            g->batch_ffn_norm,
            n_tokens,
            (uint32_t)g->ffn_mid_elems,
            full_layer_prefill,
            false) != 0;
    if (ok && g->tp_world == 2) {
        ok = glm_graph_tp_batch_ffn_combine(g, il, g->batch_ffn_out, n_tokens);
        if (!ok) fprintf(stderr, "ds4: GLM TP batch gate failed (layer %u)\n", il);
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "routed_moe",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok) {
        metal_graph_debug_dump_tensor("glm_ffn_routed_out",
                                      g->batch_ffn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
    }
    if (ok && !shared_done) DS4_GLM_ENCODE_FFN_BATCH_SHARED();
#undef DS4_GLM_ENCODE_FFN_BATCH_SHARED
    if (ok) {
        metal_graph_debug_dump_tensor("glm_ffn_shared_out",
                                      g->batch_attn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
    }
    if (ok && !glm_graph_disable_add3_residual()) {
        ok = ds4_gpu_add3_tensor(next,
                                 after_attn,
                                 g->batch_ffn_out,
                                 g->batch_attn_out,
                                 (uint32_t)residual_elems) != 0;
    } else if (ok) {
        ok = ds4_gpu_add_tensor(g->batch_heads,
                                g->batch_ffn_out,
                                g->batch_attn_out,
                                (uint32_t)residual_elems) != 0;
        if (ok) ok = ds4_gpu_add_tensor(next,
                                        after_attn,
                                        g->batch_heads,
                                        (uint32_t)residual_elems) != 0;
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "residual",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok) {
        metal_graph_debug_dump_tensor("glm_ffn_next",
                                      next,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
    }
    return ok;
}


static bool glm_graph_begin_commands_if_needed(void);

#endif /* !DS4_NO_GPU */
