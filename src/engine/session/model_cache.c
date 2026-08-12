#include "engine_internal.h"

/* Model Cache module. */
/* Single-GB10 engine startup and focused integration-test hooks. */

#ifdef DS4_TEST_HOOKS
int ds4_test_session_read_logits(ds4_session *s, float *out,
                                 uint64_t out_bytes) {
    if (!s || !out ||
        out_bytes < (uint64_t)DS4_N_VOCAB * sizeof(float)) {
        return 1;
    }
    return ds4_session_copy_logits(s, out, (int)DS4_N_VOCAB) ==
                   (int)DS4_N_VOCAB ? 0 : 1;
}

int ds4_test_session_seed_frontier(ds4_session *s, uint32_t pos,
                                   bool initialize_cache) {
    if (!s || !s->engine || s->engine->backend != DS4_BACKEND_CUDA ||
        pos >= (uint32_t)s->ctx_size || s->graph.raw_cap == 0u) {
        return 1;
    }

    ds4_gpu_graph *g = &s->graph;
    if (pos > (uint32_t)INT_MAX) return 1;
    if (s->checkpoint.cap < (int)pos + 1) {
        s->checkpoint.v = xrealloc(
                s->checkpoint.v,
                ((size_t)pos + 1u) * sizeof(s->checkpoint.v[0]));
        s->checkpoint.cap = (int)pos + 1;
    }
    s->checkpoint.len = (int)pos;
    s->checkpoint_valid = true;

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        const uint32_t n_comp = ratio ? pos / ratio : 0u;
        if (n_comp > g->layer_comp_cap[il]) return 1;
        g->layer_n_comp[il] = n_comp;
        g->layer_n_index_comp[il] = ratio == 4u ? n_comp : 0u;

        if (!initialize_cache) continue;
        if (!ds4x_graph_cache_zero(g, DS4X_CACHE_KV,
                                   g->layer_raw_cache[il], g->raw_cap)) {
            return 1;
        }
        if (ratio == 0u) continue;
        if (n_comp != 0u &&
            !ds4x_graph_cache_zero(g, DS4X_CACHE_KV,
                                   g->layer_attn_comp_cache[il], n_comp)) {
            return 1;
        }
        const uint32_t coff = ratio == 4u ? 2u : 1u;
        const uint64_t attn_state_count =
            (uint64_t)coff * DS4_N_HEAD_DIM * coff * ratio;
        if (!metal_tensor_fill_f32(g->layer_attn_state_kv[il], 0.0f,
                                   attn_state_count) ||
            !metal_tensor_fill_f32(g->layer_attn_state_score[il],
                                   DS4_NEG_INF, attn_state_count)) {
            return 1;
        }
        if (ratio == 4u) {
            if (n_comp != 0u &&
                !ds4x_graph_cache_zero(g, DS4X_CACHE_INDEXER,
                                       g->layer_index_comp_cache[il],
                                       n_comp)) {
                return 1;
            }
            const uint64_t index_state_count =
                (uint64_t)coff * DS4_N_INDEXER_HEAD_DIM * coff * ratio;
            if (!metal_tensor_fill_f32(g->layer_index_state_kv[il], 0.0f,
                                       index_state_count) ||
                !metal_tensor_fill_f32(g->layer_index_state_score[il],
                                       DS4_NEG_INF, index_state_count)) {
                return 1;
            }
        }
    }
    return ds4_gpu_synchronize() != 0 ? 0 : 1;
}
#endif

static bool ds4_engine_map_model(ds4_engine *e,
                                 const ds4_engine_options *opt) {
    e->startup_model_span_bytes =
        e->model.size > e->model.tensor_data_pos
            ? e->model.size - e->model.tensor_data_pos
            : 0;

#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
    (void)ds4_gpu_build_derived_artifacts(e->model.map,
                                          e->model.size,
                                          opt->model_path);
#endif
    if (!ds4_gpu_set_model_map_range(e->model.map,
                                     e->model.size,
                                     e->model.tensor_data_pos,
                                     e->startup_model_span_bytes,
                                     e->model.max_tensor_bytes)) {
        fprintf(stderr,
                "ds4: CUDA failed to map the target model; check unified-memory pressure\n");
        return false;
    }

    (void)ds4_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
    if (!accelerator_cache_model_tensors(DS4_BACKEND_CUDA,
                                         &e->model,
                                         NULL,
                                         NULL,
                                         0)) {
        fprintf(stderr, "ds4: CUDA failed to prepare the target model cache\n");
        return false;
    }
    return true;
}

static bool ds4_engine_map_dspark(ds4_engine *e) {
    if (e->support_kind != DS4_SUPPORT_DSPARK || !e->dspark) return true;
    const uint64_t bytes =
        e->support_model.size > e->support_model.tensor_data_pos
            ? e->support_model.size - e->support_model.tensor_data_pos
            : 0;
    if (!ds4_gpu_set_model_map_range(e->support_model.map,
                                     e->support_model.size,
                                     e->support_model.tensor_data_pos,
                                     bytes,
                                     e->support_model.max_tensor_bytes)) {
        fprintf(stderr, "ds4: CUDA failed to map the DSpark support model\n");
        return false;
    }
    (void)ds4_gpu_set_model_fd_for_map(e->support_model.fd, e->support_model.map);
    if (!accelerator_cache_model_tensors(DS4_BACKEND_CUDA,
                                         &e->support_model,
                                         NULL,
                                         NULL,
                                         0)) {
        fprintf(stderr, "ds4: CUDA failed to prepare the DSpark model cache\n");
        return false;
    }
    (void)ds4_gpu_set_model_fd_for_map(e->model.fd, e->model.map);
    return true;
}

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    if (!out || !opt || !opt->model_path || !opt->model_path[0]) return 1;
    *out = NULL;
    if (opt->backend != DS4_BACKEND_CUDA) {
        fprintf(stderr, "ds4: this build supports only the CUDA GB10 backend\n");
        return 1;
    }
    if (opt->dspark && (!opt->dspark_model_path || !opt->dspark_model_path[0])) {
        fprintf(stderr, "ds4: --dspark requires --dspark-model FILE\n");
        return 1;
    }

    ds4_engine *e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->support_model.fd = -1;
    e->backend = DS4_BACKEND_CUDA;
    e->quality = opt->quality;
    e->dspark = opt->dspark;
    e->dspark_strict = opt->dspark_strict;
    e->power_percent = opt->power_percent > 0 ? opt->power_percent : 100;
    if (e->power_percent > 100) e->power_percent = 100;
    e->prefill_chunk = opt->prefill_chunk;
    e->dspark_confidence_threshold =
        opt->dspark_confidence_threshold_set
            ? opt->dspark_confidence_threshold
            : 0.7f;
    e->directional_steering_attn_scale = opt->directional_steering_attn;
    e->directional_steering_ffn_scale = opt->directional_steering_ffn;
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        e->directional_steering_file =
            ds4_strdup(opt->directional_steering_file);
    } else if (opt->directional_steering_attn != 0.0f ||
               opt->directional_steering_ffn != 0.0f) {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        free(e);
        return 1;
    }
    if (opt->n_threads > 0) g_requested_threads = (uint32_t)opt->n_threads;

    ds4_acquire_instance_lock();
    if (opt->simulate_used_memory_bytes != 0 &&
        !ds4_memory_lock_acquire(&e->simulated_memory,
                                 opt->simulate_used_memory_bytes)) {
        ds4_engine_close(e);
        return 1;
    }

    ds4_linux_graph_backend_set_oom_score(DS4_BACKEND_CUDA);
    model_open(&e->model, opt->model_path, true, !opt->inspect_only);
    if (opt->warm_weights) model_warm_weights(&e->model);
    config_validate_model(&e->model);
    weights_bind(&e->weights, &e->model, false, 0, 0, false, false);
    vocab_load(&e->vocab, &e->model);

    if (opt->dspark_model_path && opt->dspark_model_path[0]) {
        model_open(&e->support_model, opt->dspark_model_path, true, !opt->inspect_only);
        ds4_dspark_summary summary = {0};
        e->support_kind =
            support_model_detect(&e->support_model, &e->support_stages, &summary);
        if (e->support_kind != DS4_SUPPORT_DSPARK) {
            fprintf(stderr,
                    "ds4: support checkpoint is %s, but this runtime keeps only DSpark\n",
                    support_kind_name(e->support_kind));
            ds4_engine_close(e);
            return 1;
        }
        dspark_weights_bind_optional(&e->dspark_weights,
                                     &e->support_model,
                                     &summary);
        fprintf(stderr,
                "ds4: DSpark support model: stages=%u block=%u markov_rank=%u "
                "tensors=%u missing=%u invalid=%u metadata_errors=%u\n",
                e->support_stages,
                summary.block_size,
                summary.markov_rank,
                e->dspark_weights.present_tensors,
                e->dspark_weights.missing_tensors,
                e->dspark_weights.invalid_tensors,
                e->dspark_weights.metadata_errors);
    }

    if (opt->inspect_only) {
        *out = e;
        return 0;
    }

    e->metal_ready = ds4_gpu_init() != 0;
    if (!e->metal_ready) {
        fprintf(stderr, "ds4: CUDA backend unavailable\n");
        ds4_engine_close(e);
        return 1;
    }
    ds4_gpu_set_quality(e->quality);
    (void)ds4_gpu_set_model_fd(e->model.fd);

    if (!ds4_engine_map_model(e, opt) || !ds4_engine_map_dspark(e)) {
        ds4_engine_close(e);
        return 1;
    }

    fprintf(stderr, "ds4: CUDA backend initialized for one GB10 and one request\n");
    ds4_engine_print_startup_memory(e, opt->context_size);
    *out = e;
    return 0;
}

void ds4_engine_summary(ds4_engine *e) {
    if (!e) return;
    model_summary(&e->model);
    if (e->support_model.map) {
        printf("\nDSpark support model (stages=%u):\n", e->support_stages);
        model_summary(&e->support_model);
        printf("support binding: tensors=%u missing=%u invalid=%u metadata_errors=%u\n",
               e->dspark_weights.present_tensors,
               e->dspark_weights.missing_tensors,
               e->dspark_weights.invalid_tensors,
               e->dspark_weights.metadata_errors);
    }
}

int ds4_engine_vocab_size(ds4_engine *e) {
    return e ? e->vocab.n_vocab : 0;
}

uint32_t ds4_engine_prefill_chunk(ds4_engine *e) {
    return e ? e->prefill_chunk : 0;
}

int ds4_engine_power(ds4_engine *e) {
    return e ? e->power_percent : 100;
}

int ds4_engine_set_power(ds4_engine *e, int power_percent) {
    if (!e || power_percent < 1 || power_percent > 100) return 1;
    e->power_percent = power_percent;
    return 0;
}

const char *ds4_engine_model_name(ds4_engine *e) {
    (void)e;
    return DS4_MODEL_SHAPE_NAME;
}

int ds4_engine_layer_count(ds4_engine *e) {
    (void)e;
    return (int)DS4_N_LAYER;
}

uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer) {
    (void)e;
    return layer < DS4_N_LAYER ? ds4_layer_compress_ratio(layer) : 0;
}

uint64_t ds4_engine_hidden_f32_values(ds4_engine *e) {
    (void)e;
    return (uint64_t)DS4_N_HC * DS4_N_EMBD;
}

int ds4_engine_model_id(ds4_engine *e) {
    (void)e;
    return (int)DS4_VARIANT_FLASH;
}
