#ifndef DS4X_OPS_DECODE_GRAPH_H
#define DS4X_OPS_DECODE_GRAPH_H

#include <stdint.h>

#include "ds4x/runtime/context.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t layer;
    uint32_t island;
    uint32_t variant;
    void *current_hc;
    void *after_attention_hc;
    void *after_ffn_hc;
    void *attention_norm;
} ds4x_decode_graph_args;

int ds4x_decode_graph_supports(const ds4x_runtime_context *context,
                               const ds4x_decode_graph_args *args);
uint64_t ds4x_decode_graph_workspace_bytes(const ds4x_decode_graph_args *args);
int ds4x_decode_graph_begin(const ds4x_runtime_context *context,
                            const ds4x_decode_graph_args *args);
int ds4x_decode_graph_end(const ds4x_runtime_context *context,
                          const ds4x_decode_graph_args *args);
void ds4x_decode_graph_abort(const ds4x_runtime_context *context,
                             const ds4x_decode_graph_args *args);

#ifdef __cplusplus
}
#endif

#endif
