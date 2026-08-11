#ifndef DS4X_CUTLASS_FP16_GEMM_H
#define DS4X_CUTLASS_FP16_GEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Logical operation:
 *
 *   output[tokens, out] = activations[tokens, in] * weights[in, out]
 *
 * Activations and weights contain IEEE FP16 values. The physical weight
 * storage is row-major [out, in], which is the same byte layout as the
 * logical column-major [in, out] view consumed by GEMM. Output and
 * accumulation are FP32. The stream is a cudaStream_t passed as void * so
 * this private backend header stays usable from both C and CUDA C++ code. */
int ds4x_cutlass_fp16_gemm(
        float          *output,
        const uint16_t *activations,
        const uint16_t *weights,
        uint64_t        tokens,
        uint64_t        in_dim,
        uint64_t        out_dim,
        void           *stream);

/* Returns 1 when the dimensions and pointer alignment are supported, 0
 * otherwise. It does not enqueue GPU work. */
int ds4x_cutlass_fp16_gemm_supported(
        const float    *output,
        const uint16_t *activations,
        const uint16_t *weights,
        uint64_t        tokens,
        uint64_t        in_dim,
        uint64_t        out_dim);

const char *ds4x_cutlass_status_string(int status);

#ifdef __cplusplus
}
#endif

#endif
