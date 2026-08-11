#ifndef DS4X_OPS_ATTENTION_H
#define DS4X_OPS_ATTENTION_H

#include <stdint.h>

#include "ds4x/runtime/context.h"

typedef struct ds4_gpu_tensor ds4_gpu_tensor;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ds4_gpu_tensor *heads;
    const void *model_map;
    uint64_t model_size;
    uint64_t sinks_offset;
    const ds4_gpu_tensor *query;
    const ds4_gpu_tensor *raw_kv;
    const ds4_gpu_tensor *compressed_kv;
    const ds4_gpu_tensor *compressed_mask;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t use_mask;
    uint32_t n_head;
    uint32_t head_dim;
} ds4x_attention_decode_args;

int ds4x_attention_decode_supports(
        const ds4x_runtime_context *context,
        const ds4x_attention_decode_args *args);
uint64_t ds4x_attention_decode_workspace_bytes(
        const ds4x_attention_decode_args *args);
int ds4x_attention_decode_launch(
        const ds4x_runtime_context *context,
        const ds4x_attention_decode_args *args);

#ifdef __cplusplus
}
#endif

#endif
