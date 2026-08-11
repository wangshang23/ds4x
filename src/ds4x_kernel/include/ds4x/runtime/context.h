#ifndef DS4X_RUNTIME_CONTEXT_H
#define DS4X_RUNTIME_CONTEXT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int device_id;
    void *stream;
    void *workspace;
    uint64_t workspace_bytes;
} ds4x_runtime_context;

static inline ds4x_runtime_context ds4x_default_runtime_context(void) {
    ds4x_runtime_context context = {0, 0, 0, 0};
    return context;
}

#ifdef __cplusplus
}
#endif

#endif
