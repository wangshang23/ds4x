/* Single-GB10 tensor, command, and model-cache lifetime. */
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

int ds4_gpu_init(void);
void ds4_gpu_cleanup(void);

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base,
                                    uint64_t offset, uint64_t bytes);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value,
                            uint64_t count);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset,
                         const void *data, uint64_t bytes);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset,
                        void *data, uint64_t bytes);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                        const ds4_gpu_tensor *src, uint64_t src_offset,
                        uint64_t bytes);
int ds4_gpu_spark_pack_kv_rows_tensor(ds4_gpu_tensor *dst, uint64_t dst_row,
                                      const ds4_gpu_tensor *src,
                                      uint32_t src_row, uint32_t rows);
int ds4_gpu_spark_pack_index_rows_tensor(ds4_gpu_tensor *dst,
                                         uint64_t dst_row,
                                         const ds4_gpu_tensor *src,
                                         uint32_t src_row, uint32_t rows);
int ds4_gpu_spark_zero_kv_rows_tensor(ds4_gpu_tensor *dst, uint32_t rows);
int ds4_gpu_spark_zero_index_rows_tensor(ds4_gpu_tensor *dst, uint32_t rows);
int ds4_gpu_spark_unpack_kv_rows_tensor(ds4_gpu_tensor *dst,
                                        const ds4_gpu_tensor *src,
                                        uint32_t rows);
int ds4_gpu_spark_unpack_index_rows_tensor(ds4_gpu_tensor *dst,
                                           const ds4_gpu_tensor *src,
                                           uint32_t rows);

int ds4_gpu_pack_slot_rows_f32_tensor(ds4_gpu_tensor *out,
                                      const ds4_gpu_tensor *slots,
                                      uint32_t n_rows, uint32_t width,
                                      uint32_t n_slots, uint32_t slot_cap);

int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_end_commands(void);
int ds4_gpu_synchronize(void);

int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map);
int ds4_gpu_build_derived_artifacts(const void *model_map,
                                    uint64_t model_size,
                                    const char *model_path);
int ds4_gpu_model_range_replaced(const void *model_map, uint64_t offset,
                                 uint64_t bytes);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size,
                                uint64_t map_offset, uint64_t map_size,
                                uint64_t max_tensor_bytes);
int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size,
                              uint64_t offset, uint64_t bytes,
                              const char *label);
int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size,
                               uint64_t offset, uint64_t bytes,
                               uint64_t in_dim, uint64_t out_dim,
                               const char *label);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes,
                                        uint64_t context_bytes);
void ds4_gpu_set_quality(bool quality);
void ds4_gpu_print_memory_report(const char *label);
