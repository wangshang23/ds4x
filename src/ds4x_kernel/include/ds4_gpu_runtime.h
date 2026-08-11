#ifndef DS4_GPU_RUNTIME_H
#define DS4_GPU_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

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

#define DS4_MAX_GPUS 1

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
    int device_id;
};
#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif

#ifndef DS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define DS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define DS4_GPU_ATTENTION_DECODE_BATCH_MAX 32u
typedef struct {
    uint64_t raw_kv;
    uint64_t comp_kv;
    uint64_t topk;
    uint32_t pos;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t window;
    uint32_t ratio;
    uint32_t indexed;
} ds4_gpu_attention_decode_row;
#endif

typedef struct {
    int device_id;
    void *stream;
    void *cublas;
    int cublas_ready;
    void *scratch;
    size_t scratch_bytes;
    size_t budget_bytes;
    size_t used_bytes;
    void *boundary_event;
} ds4_gpu_ctx;

extern ds4_gpu_ctx g_gpu[DS4_MAX_GPUS];
extern int g_n_gpus;
extern int g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS];

void ds4_gpu_cleanup(void);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int device_slot,
                                             uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int device_slot,
                                                 uint64_t bytes);
int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *tensor, int device_id,
                            uint64_t bytes);
void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_device(const ds4_gpu_tensor *tensor);
int ds4_gpu_set_current_device(int device_slot);
int ds4_gpu_set_current_device_fenced(int device_slot);
int ds4_gpu_register_model_map_no_copy(const void *model_map,
                                       uint64_t model_size);

#ifdef __cplusplus
}
#endif

#endif
