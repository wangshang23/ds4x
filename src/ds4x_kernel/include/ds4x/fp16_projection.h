#ifndef DS4X_FP16_PROJECTION_H
#define DS4X_FP16_PROJECTION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ds4x_fp16_projection_mode {
    DS4X_FP16_PROJECTION_PARALLEL = 0,
    DS4X_FP16_PROJECTION_SERIAL = 1,
    DS4X_FP16_PROJECTION_ORDERED = 2,
};

/* Launch-only helpers for the exact legacy projection fallbacks. Return the
 * underlying cudaError_t value as an int; allocation, pointer ownership and
 * backend selection stay in the engine-facing wrapper. */
int ds4x_launch_fp16_projection(
        float          *output,
        const uint16_t *weights,
        const float    *activations,
        uint64_t        in_dim,
        uint64_t        out_dim,
        uint64_t        tokens,
        int             mode,
        void           *stream);

int ds4x_launch_fp16_small_output_one(
        float          *output,
        const uint16_t *weights,
        const float    *activations,
        uint64_t        in_dim,
        uint64_t        out_dim,
        void           *stream);

int ds4x_launch_fp16_small_output_batch(
        float          *output,
        const uint16_t *weights,
        const float    *activations,
        uint64_t        in_dim,
        uint64_t        out_dim,
        uint64_t        tokens,
        void           *stream);

int ds4x_launch_fp16_pair_ordered(
        float          *output_a,
        float          *output_b,
        const uint16_t *weights_a,
        const uint16_t *weights_b,
        const float    *activations,
        uint64_t        in_dim,
        uint64_t        out_a_dim,
        uint64_t        out_b_dim,
        void           *stream);

#ifdef __cplusplus
}
#endif

#endif
