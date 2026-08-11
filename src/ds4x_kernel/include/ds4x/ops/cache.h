#ifndef DS4X_OPS_CACHE_H
#define DS4X_OPS_CACHE_H

#include <stdint.h>

#include "ds4x/runtime/context.h"

typedef struct ds4_gpu_tensor ds4_gpu_tensor;

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DS4X_CACHE_KV = 0,
    DS4X_CACHE_INDEXER = 1,
} ds4x_cache_kind;

typedef enum {
    DS4X_CACHE_PACK = 0,
    DS4X_CACHE_UNPACK = 1,
    DS4X_CACHE_ZERO = 2,
} ds4x_cache_operation;

typedef struct {
    ds4x_cache_kind kind;
    ds4x_cache_operation operation;
    ds4_gpu_tensor *dst;
    const ds4_gpu_tensor *src;
    uint64_t dst_row;
    uint32_t src_row;
    uint32_t rows;
} ds4x_cache_args;

int ds4x_cache_supports(const ds4x_runtime_context *context,
                        const ds4x_cache_args *args);
uint64_t ds4x_cache_workspace_bytes(const ds4x_cache_args *args);
int ds4x_cache_launch(const ds4x_runtime_context *context,
                      const ds4x_cache_args *args);

#ifdef __cplusplus
}
#endif

#endif
