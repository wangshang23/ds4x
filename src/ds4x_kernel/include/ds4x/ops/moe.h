#ifndef DS4X_OPS_MOE_H
#define DS4X_OPS_MOE_H

#include <stdbool.h>
#include <stdint.h>

#include "ds4x/runtime/context.h"

typedef struct ds4_gpu_tensor ds4_gpu_tensor;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    ds4_gpu_tensor *output;
    ds4_gpu_tensor *gate;
    ds4_gpu_tensor *up;
    ds4_gpu_tensor *mid;
    ds4_gpu_tensor *down;
    const void *model_map;
    uint64_t model_size;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint32_t gate_type;
    uint32_t down_type;
    uint64_t gate_expert_bytes;
    uint64_t gate_row_bytes;
    uint64_t down_expert_bytes;
    uint64_t down_row_bytes;
    uint32_t expert_in_dim;
    uint32_t expert_mid_dim;
    uint32_t output_dim;
    const ds4_gpu_tensor *selected;
    const ds4_gpu_tensor *weights;
    uint32_t total_experts;
    uint32_t selected_experts;
    float clamp;
    const ds4_gpu_tensor *input;
    const ds4_gpu_tensor *add_input;
    uint32_t layer;
    bool force_resident;
} ds4x_routed_moe_args;

int ds4x_routed_moe_supports(const ds4x_runtime_context *context,
                             const ds4x_routed_moe_args *args);
uint64_t ds4x_routed_moe_workspace_bytes(const ds4x_routed_moe_args *args);
int ds4x_routed_moe_launch(const ds4x_runtime_context *context,
                           const ds4x_routed_moe_args *args);

#ifdef __cplusplus
}
#endif

#endif
