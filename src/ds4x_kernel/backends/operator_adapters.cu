#include "ds4x/ops/attention.h"
#include "ds4x/ops/cache.h"
#include "ds4x/ops/decode_graph.h"
#include "ds4x/ops/hyper_connection.h"
#include "ds4x/ops/indexer.h"
#include "ds4x/ops/moe.h"

#include "ds4_gpu.h"

#include <cstdlib>

namespace {

bool context_supported(const ds4x_runtime_context *context) {
    return context && context->device_id == 0 && context->stream == nullptr;
}

ds4_decode_graph_key graph_key(const ds4x_decode_graph_args *args) {
    ds4_decode_graph_key key{};
    key.il = args->layer;
    key.island = args->island;
    key.variant = args->variant;
    key.cur_hc = args->current_hc;
    key.after_attn_hc = args->after_attention_hc;
    key.after_ffn_hc = args->after_ffn_hc;
    key.attn_norm = args->attention_norm;
    return key;
}

}  // namespace

extern "C" int ds4x_cache_supports(
        const ds4x_runtime_context *context,
        const ds4x_cache_args *args) {
    return context_supported(context) && args && args->dst && args->rows != 0 &&
           (args->operation == DS4X_CACHE_ZERO || args->src);
}

extern "C" uint64_t ds4x_cache_workspace_bytes(const ds4x_cache_args *) {
    return 0;
}

extern "C" int ds4x_cache_launch(
        const ds4x_runtime_context *context,
        const ds4x_cache_args *args) {
    if (!ds4x_cache_supports(context, args)) return 0;
    if (args->kind == DS4X_CACHE_KV) {
        if (args->operation == DS4X_CACHE_PACK) {
            return ds4_gpu_spark_pack_kv_rows_tensor(
                    args->dst, args->dst_row, args->src, args->src_row, args->rows);
        }
        if (args->operation == DS4X_CACHE_UNPACK) {
            return ds4_gpu_spark_unpack_kv_rows_tensor(args->dst, args->src, args->rows);
        }
        return ds4_gpu_spark_zero_kv_rows_tensor(args->dst, args->rows);
    }
    if (args->operation == DS4X_CACHE_PACK) {
        return ds4_gpu_spark_pack_index_rows_tensor(
                args->dst, args->dst_row, args->src, args->src_row, args->rows);
    }
    if (args->operation == DS4X_CACHE_UNPACK) {
        return ds4_gpu_spark_unpack_index_rows_tensor(args->dst, args->src, args->rows);
    }
    return ds4_gpu_spark_zero_index_rows_tensor(args->dst, args->rows);
}

extern "C" int ds4x_indexer_supports(
        const ds4x_runtime_context *context,
        const ds4x_indexer_args *args) {
    return context_supported(context) && args && args->scores && args->selected &&
           args->query && args->weights && args->cache && args->n_comp != 0 &&
           args->n_tokens != 0 && args->n_head != 0 && args->head_dim == 128 &&
           args->top_k != 0 && args->top_k <= args->n_comp;
}

extern "C" uint64_t ds4x_indexer_workspace_bytes(const ds4x_indexer_args *) {
    return 0;
}

extern "C" int ds4x_indexer_launch(
        const ds4x_runtime_context *context,
        const ds4x_indexer_args *args) {
    if (!ds4x_indexer_supports(context, args)) return 0;
    int ok = 0;
    if (args->mode == DS4X_INDEXER_DECODE_ONE) {
        if (args->n_tokens != 1) return 0;
        if (getenv("DS4_CUDA_NO_FUSED_INDEXER_TOPK") == NULL &&
            args->n_comp >= 65536u && args->top_k == 512u) {
            ok = ds4_gpu_indexer_score_topk_one_tensor(
                args->selected, args->query, args->weights, args->cache,
                args->n_comp, args->n_head, args->head_dim, args->top_k,
                args->scale);
            if (ok) return 1;
        }
        ok = ds4_gpu_indexer_score_one_tensor(
            args->scores, args->query, args->weights, args->cache,
            args->n_comp, args->n_head, args->head_dim, args->scale);
    } else if (args->mode == DS4X_INDEXER_PREFILL) {
        ok = ds4_gpu_indexer_scores_prefill_tensor(
                args->scores, args->query, args->weights, args->cache,
                args->n_comp, args->n_tokens, args->n_head, args->head_dim,
                args->ratio, args->scale);
    } else if (args->mode == DS4X_INDEXER_VERIFY) {
        ok = ds4_gpu_indexer_scores_decode_batch_tensor(
                args->scores, args->query, args->weights, args->cache,
                args->n_comp, args->n_tokens, args->pos0, args->n_head,
                args->head_dim, args->ratio, args->scale);
    }
    return ok && ds4_gpu_indexer_topk_tensor(
            args->selected, args->scores, args->n_comp,
            args->n_tokens, args->top_k);
}

extern "C" int ds4x_attention_decode_supports(
        const ds4x_runtime_context *context,
        const ds4x_attention_decode_args *args) {
    return context_supported(context) && args && args->heads && args->model_map &&
           args->query && args->raw_kv && args->n_raw != 0 &&
           args->raw_cap >= args->n_raw && args->raw_start < args->raw_cap &&
           args->n_head != 0 && args->head_dim == 512 &&
           (args->n_comp == 0 || args->compressed_kv) &&
           (!args->use_mask || args->compressed_mask);
}

extern "C" uint64_t ds4x_attention_decode_workspace_bytes(
        const ds4x_attention_decode_args *) {
    return 0;
}

extern "C" int ds4x_attention_decode_launch(
        const ds4x_runtime_context *context,
        const ds4x_attention_decode_args *args) {
    if (!ds4x_attention_decode_supports(context, args)) return 0;
    return ds4_gpu_attention_decode_heads_tensor(
            args->heads, args->model_map, args->model_size, args->sinks_offset,
            args->query, args->raw_kv, args->n_raw, args->raw_cap,
            args->raw_start, args->compressed_kv, DS4_GPU_CACHE_SPARK_KV,
            args->n_comp, args->compressed_mask, args->use_mask,
            args->n_head, args->head_dim);
}

extern "C" int ds4x_routed_moe_supports(
        const ds4x_runtime_context *context,
        const ds4x_routed_moe_args *args) {
    return context_supported(context) && args && args->output && args->mid &&
           args->model_map && args->selected && args->weights && args->input &&
           args->expert_in_dim != 0 && args->expert_mid_dim != 0 &&
           args->output_dim != 0 && args->total_experts != 0 &&
           args->selected_experts != 0;
}

extern "C" uint64_t ds4x_routed_moe_workspace_bytes(
        const ds4x_routed_moe_args *) {
    return 0;
}

extern "C" int ds4x_routed_moe_launch(
        const ds4x_runtime_context *context,
        const ds4x_routed_moe_args *args) {
    if (!ds4x_routed_moe_supports(context, args)) return 0;
    return ds4_gpu_routed_moe_one_tensor(
            args->output, args->gate, args->up, args->mid, args->down,
            args->model_map, args->model_size, args->gate_offset,
            args->up_offset, args->down_offset, args->gate_type,
            args->down_type, args->gate_expert_bytes, args->gate_row_bytes,
            args->down_expert_bytes, args->down_row_bytes,
            args->expert_in_dim, args->expert_mid_dim, args->output_dim,
            args->selected, args->weights, args->total_experts,
            args->selected_experts, args->clamp, args->input,
            args->add_input, args->layer,
            args->force_resident);
}

extern "C" int ds4x_output_hc_supports(
        const ds4x_runtime_context *context,
        const ds4x_output_hc_args *args) {
    return context_supported(context) && args && args->output && args->pre &&
           args->model_map && args->n_hc != 0;
}

extern "C" uint64_t ds4x_output_hc_workspace_bytes(const ds4x_output_hc_args *) {
    return 0;
}

extern "C" int ds4x_output_hc_launch(
        const ds4x_runtime_context *context,
        const ds4x_output_hc_args *args) {
    if (!ds4x_output_hc_supports(context, args)) return 0;
    return ds4_gpu_output_hc_weights_tensor(
            args->output, args->pre, args->model_map, args->model_size,
            args->scale_offset, args->base_offset, args->n_hc, args->epsilon);
}

extern "C" int ds4x_decode_graph_supports(
        const ds4x_runtime_context *context,
        const ds4x_decode_graph_args *args) {
    return context_supported(context) && args && args->island < 2 &&
           args->current_hc && args->attention_norm &&
           ds4_gpu_decode_graphs_supported();
}

extern "C" uint64_t ds4x_decode_graph_workspace_bytes(
        const ds4x_decode_graph_args *) {
    return 0;
}

extern "C" int ds4x_decode_graph_begin(
        const ds4x_runtime_context *context,
        const ds4x_decode_graph_args *args) {
    if (!ds4x_decode_graph_supports(context, args)) return -1;
    const ds4_decode_graph_key key = graph_key(args);
    return ds4_gpu_decode_graph_begin(&key);
}

extern "C" int ds4x_decode_graph_end(
        const ds4x_runtime_context *context,
        const ds4x_decode_graph_args *args) {
    if (!ds4x_decode_graph_supports(context, args)) return -1;
    const ds4_decode_graph_key key = graph_key(args);
    return ds4_gpu_decode_graph_end(&key);
}

extern "C" void ds4x_decode_graph_abort(
        const ds4x_runtime_context *context,
        const ds4x_decode_graph_args *args) {
    if (!context_supported(context) || !args) return;
    const ds4_decode_graph_key key = graph_key(args);
    ds4_gpu_decode_graph_abort(&key);
}
