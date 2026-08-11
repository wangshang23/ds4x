#ifndef DS4X_OPS_INDEXER_H
#define DS4X_OPS_INDEXER_H

#include <stdint.h>

#include "ds4x/runtime/context.h"

typedef struct ds4_gpu_tensor ds4_gpu_tensor;

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DS4X_INDEXER_DECODE_ONE = 0,
    DS4X_INDEXER_PREFILL = 1,
    DS4X_INDEXER_VERIFY = 2,
} ds4x_indexer_mode;

typedef struct {
    ds4x_indexer_mode mode;
    ds4_gpu_tensor *scores;
    ds4_gpu_tensor *selected;
    const ds4_gpu_tensor *query;
    const ds4_gpu_tensor *weights;
    const ds4_gpu_tensor *cache;
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t ratio;
    uint32_t top_k;
    float scale;
} ds4x_indexer_args;

int ds4x_indexer_supports(const ds4x_runtime_context *context,
                          const ds4x_indexer_args *args);
uint64_t ds4x_indexer_workspace_bytes(const ds4x_indexer_args *args);
int ds4x_indexer_launch(const ds4x_runtime_context *context,
                        const ds4x_indexer_args *args);

#ifdef __cplusplus
}
#endif

#endif
