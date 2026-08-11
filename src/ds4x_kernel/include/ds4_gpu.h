#ifndef DS4_GPU_H
#define DS4_GPU_H

#include <stdbool.h>
#include <stdint.h>

/* DGX Spark persistent-cache ABI. The attention row keeps seven 64-value
 * E4M3 blocks, a 64-value FP16 RoPE tail, and one UE8M0 scale byte per block.
 * Indexer rows are four 32-value E2M1 blocks plus four UE8M0 scale bytes. */
#ifndef DS4_SPARK_KV_ROW_BYTES
#define DS4_SPARK_KV_NOPE_DIM 448u
#define DS4_SPARK_KV_ROPE_DIM 64u
#define DS4_SPARK_KV_ROW_BYTES 583u
#define DS4_SPARK_INDEX_ROW_BYTES 68u
#define DS4_GPU_CACHE_F32 0u
#define DS4_GPU_CACHE_F16 1u
#define DS4_GPU_CACHE_SPARK_KV 2u
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "ds4_gpu/00_tensor_runtime.h"
#include "ds4_gpu/01_linear_and_reduction.h"
#include "ds4_gpu/02_fused_projection.h"
#include "ds4_gpu/03_attention.h"
#include "ds4_gpu/04_moe.h"
#include "ds4_gpu/05_hyper_connection.h"
#include "ds4_gpu/06_decode_graph.h"
#ifdef __cplusplus
}
#endif

#endif
