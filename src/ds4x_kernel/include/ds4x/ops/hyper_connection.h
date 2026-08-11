#ifndef DS4X_OPS_HYPER_CONNECTION_H
#define DS4X_OPS_HYPER_CONNECTION_H

#include <stdint.h>

#include "ds4x/runtime/context.h"

typedef struct ds4_gpu_tensor ds4_gpu_tensor;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ds4_gpu_tensor *output;
    const ds4_gpu_tensor *pre;
    const void *model_map;
    uint64_t model_size;
    uint64_t scale_offset;
    uint64_t base_offset;
    uint32_t n_hc;
    float epsilon;
} ds4x_output_hc_args;

int ds4x_output_hc_supports(const ds4x_runtime_context *context,
                            const ds4x_output_hc_args *args);
uint64_t ds4x_output_hc_workspace_bytes(const ds4x_output_hc_args *args);
int ds4x_output_hc_launch(const ds4x_runtime_context *context,
                          const ds4x_output_hc_args *args);

#ifdef __cplusplus
}
#endif

#endif
