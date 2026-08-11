/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
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
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count);
int ds4_gpu_spark_pack_kv_rows_tensor(ds4_gpu_tensor *dst,
                                      uint64_t dst_row,
                                      const ds4_gpu_tensor *src,
                                      uint32_t src_row,
                                      uint32_t rows);
int ds4_gpu_spark_pack_index_rows_tensor(ds4_gpu_tensor *dst,
                                         uint64_t dst_row,
                                         const ds4_gpu_tensor *src,
                                         uint32_t src_row,
                                         uint32_t rows);
int ds4_gpu_spark_zero_kv_rows_tensor(ds4_gpu_tensor *dst, uint32_t rows);
int ds4_gpu_spark_zero_index_rows_tensor(ds4_gpu_tensor *dst, uint32_t rows);
int ds4_gpu_spark_unpack_kv_rows_tensor(ds4_gpu_tensor *dst,
                                        const ds4_gpu_tensor *src,
                                        uint32_t rows);
int ds4_gpu_spark_unpack_index_rows_tensor(ds4_gpu_tensor *dst,
                                           const ds4_gpu_tensor *src,
                                           uint32_t rows);
int ds4_gpu_moe_handoff_pack_tensor(
        ds4_gpu_tensor       *packed,
        const ds4_gpu_tensor *ffn_norm,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_embd,
        uint32_t              n_expert);
int ds4_gpu_pack_slot_rows_f32_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *slots,
        uint32_t                n_rows,
        uint32_t                width,
        uint32_t                n_slots,
        uint32_t                slot_cap);

int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_encoder(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_commands_active(void);
#ifdef __APPLE__
int ds4_gpu_parallel_ffn_finish(void);
void ds4_gpu_parallel_ffn_abort(void);
int ds4_gpu_parallel_ffn_start(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        const ds4_gpu_tensor *x,
        float                 clamp);
#endif
int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value);
int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label);
int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label);
#ifdef DS4_ROCM_BUILD
int ds4_gpu_tensor_read_after_selected_event(const ds4_gpu_tensor *tensor,
                                             uint64_t offset,
                                             void *data,
                                             uint64_t bytes,
                                             uint64_t event_value,
                                             const char *label);
#endif
int ds4_gpu_end_commands(void);
int ds4_gpu_synchronize(void);

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);
int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map);
int ds4_gpu_build_derived_artifacts(const void *model_map, uint64_t model_size,
                                    const char *model_path);
int ds4_gpu_model_range_replaced(const void *model_map, uint64_t offset,
                                 uint64_t bytes);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes);
int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);
int ds4_gpu_q8_cache_suppressed(void);
void ds4_gpu_set_q8_cache_suppressed(int suppressed);
#ifdef DS4_ROCM_BUILD
void ds4_gpu_release_q8_f16_cache(void);
#endif

/* Model-file ranges assigned to CUDA devices by the multi-GPU placement
 * planner. Metal keeps these declarations for the shared engine interface. */
#ifndef DS4_MAX_GPUS
#define DS4_MAX_GPUS 16
#endif
typedef struct {
    uint64_t source_offset;
    uint64_t bytes;
    int target_device;
} ds4_tensor_range;

int ds4_gpu_device_cache_tensors(int device_id,
                                 const ds4_tensor_range *ranges,
                                 int n_ranges);
int ds4_gpu_register_support_map(const void *map, uint64_t size, uint64_t bias);
int ds4_gpu_device_cache_support_tensors(int device_id,
                                         int entry_device_id,
                                         const ds4_tensor_range *ranges,
                                         int n_ranges,
                                         int from_main_map);
uint64_t ds4_gpu_tier_free_vram(int logical_tier);
int ds4_gpu_lookup_cache(uint64_t source_offset, uint64_t bytes,
                         int *out_device_id, void **out_device_ptr);
int ds4_gpu_lookup_cache_device(uint64_t source_offset, uint64_t bytes);

int ds4_gpu_pro_q4_expert_table_auto_available(void);
int ds4_gpu_preload_q4_expert_tables(const void *model_map, uint64_t model_size,
                                     uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                                     uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
                                     uint32_t n_total_expert);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
void ds4_gpu_set_quality(bool quality);
void ds4_gpu_set_glm_model(bool enabled);
void ds4_gpu_set_ssd_streaming(bool enabled);
void ds4_gpu_set_glm_streaming_prefill_full_layer(bool enabled);
#ifdef __APPLE__
int ds4_gpu_device_is_pre_m5_apple_silicon(void);
int ds4_gpu_device_is_m5_apple_silicon(void);
int ds4_gpu_set_decode_pipeline_fast_lookup(int enabled);
/* Strict test oracle for the fixed decode mul_mv pipeline lookup cache. */
int ds4_gpu_test_decode_pipeline_fast_lookup(void);
/* Strict test oracle for the extended decode mul_mv_ext (nsg + nxpsg) cache. */
int ds4_gpu_test_decode_pipeline_fast_lookup_ext(void);
/* Strict test oracle for the generated resident-prefill MXFP4 half LUT. */
int ds4_gpu_test_mxfp4_down_half_lut(uint16_t *legacy_bits,
                                     uint16_t *lut_bits);
enum {
    DS4_GPU_TEST_MXFP4_PAIR_TAIL_CULL = 1u << 0,
    DS4_GPU_TEST_MXFP4_PAIR_COMPACT_TILE = 1u << 1,
    DS4_GPU_TEST_MXFP4_MAP_SCATTER = 1u << 2,
    DS4_GPU_TEST_MXFP4_DOWN_TAIL_CULL = 1u << 3,
    DS4_GPU_TEST_MXFP4_DOWN_HALF_LUT = 1u << 4,
    DS4_GPU_TEST_OUTPUT_HC_WEIGHTS4 = 1u << 5,
    DS4_GPU_TEST_HC_RMS_SCALE_PROJ = 1u << 6,
};
void ds4_gpu_test_set_flags(uint32_t flags);
void ds4_gpu_release_zero_prefix_prefill_mask_cache(void);
#else
static inline int ds4_gpu_device_is_pre_m5_apple_silicon(void) { return 0; }
static inline int ds4_gpu_device_is_m5_apple_silicon(void) { return 0; }
#endif
void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts);
void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes);
uint64_t ds4_gpu_recommended_working_set_size(void);
uint32_t ds4_gpu_stream_expert_cache_configured_count(void);
uint32_t ds4_gpu_stream_expert_cache_current_count(void);
typedef struct ds4_gpu_stream_expert_table {
    const void *model_map;
    uint64_t    model_size;
    uint32_t    layer;
    uint32_t    n_total_expert;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;
} ds4_gpu_stream_expert_table;
/* Reset only the prompt-local eviction heuristic.  The resident SSD expert
 * cache itself is intentionally kept warm across sessions. */
void ds4_gpu_stream_expert_cache_reset_route_hotness(void);
void ds4_gpu_stream_expert_cache_release_resident(void);
uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes);
int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected);
int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected);
int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_selected);
#ifdef __APPLE__
/* The async selected-load worker registers itself so Metal cache paths never
 * wait on command buffers from that thread (they fail the load instead and
 * the caller retries synchronously). */
void ds4_gpu_stream_expert_cache_note_service_thread(void);
#endif
#if defined(DS4_ROCM_BUILD) || (!defined(DS4_NO_GPU) && !defined(__APPLE__))
int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected);
#endif
#ifdef DS4_ROCM_BUILD
int ds4_gpu_stream_expert_cache_load_layer(
        const ds4_gpu_stream_expert_table *table);
int ds4_gpu_stream_expert_cache_seed_from_layer_selected(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor             *selected,
        uint32_t                          n_tokens,
        uint32_t                          n_seed_tokens,
        uint32_t                          n_selected);
int ds4_gpu_stream_expert_cache_finish_pending_batch(void);
int ds4_gpu_stream_expert_cache_release_layer_cache(void);
#endif
int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts);
#ifdef __APPLE__
/* Seed from mapped weights with blits appended to the active command buffer. */
int ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts);
#endif
void ds4_gpu_print_memory_report(const char *label);

/* Tensor-parallel per-layer gates (Metal only).  The encoder calls
 * ds4_gpu_tp_gate_encode() right after the kernels that produce a partial
 * block output in the TP slab: it closes the current encoder, makes the GPU
 * signal a shared event, queues the exchange on a service thread, and makes
 * the GPU wait for the CPU-signaled release before the combine kernel runs.
 * Sequence values are assigned internally and increase monotonically; both
 * ranks encode the identical gate sequence so values pair up by
 * construction.  The exchange callback runs on the service thread and must
 * return nonzero on success. */
typedef int (*ds4_gpu_tp_exchange_fn)(void *ud, uint32_t layer, uint32_t gate, uint64_t seq);
/* Bind one rank of the two-way split. slab is the transport slab tensor and
 * gpu_flags_off is the offset of its GPU-written gate-ready flag words. */
int ds4_gpu_tp_init(uint32_t rank,
                    ds4_gpu_tensor *slab, uint64_t gpu_flags_off,
                    ds4_gpu_tp_exchange_fn fn, void *ud);
void ds4_gpu_tp_shutdown(void);
/* Multi-session TP reuses slab slots across several encoded graph tapes.
 * Shared-event arrival is required in that mode to make each partial vector
 * CPU-visible before the transport thread reads it. */
void ds4_gpu_tp_set_session_batch_mode(int enabled);
/* The coordinator-only DSpark support model does not participate in TP.
 * Suspend ownership only while encoding it; base-model verification remains
 * split across both ranks. */
void ds4_gpu_tp_suspend_expert_sharding(int suspend);
int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate);
/* Verify-block batch gates: one exchange per layer moving `rows` partial
 * rows at once (speculative verify).  The callback runs on the gate service
 * thread with the same ud as the row-gate exchange fn. */
typedef int (*ds4_gpu_tp_batch_exchange_fn)(void *ud, uint32_t layer,
                                            uint32_t rows, uint64_t seq);
void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn);
int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows);
/* Prefill batch gates: the service thread exchanges `bytes` between two
 * CPU-visible bounce tensors directly (payloads far beyond slab slots). */
typedef int (*ds4_gpu_tp_big_exchange_fn)(void *ud, uint32_t layer,
                                          uint64_t seq, const void *out,
                                          void *in, uint64_t bytes);
void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn);
int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                               const ds4_gpu_tensor *out_t,
                               ds4_gpu_tensor *in_t,
                               uint64_t bytes);
/* Split big gate: kick publishes the GPU arrival marker (batch shared
 * event, whose completion semantics make the bounce payload visible to
 * the exchange thread) and queues the exchange, returning the gate seq
 * (0 on failure); wait encodes the release.  Multiple kicks may be in
 * flight; waiting on the last seq covers all earlier kicks (monotonic
 * release event, in-order service thread). */
uint64_t ds4_gpu_tp_big_gate_kick(uint32_t layer, uint32_t rows,
                                  const ds4_gpu_tensor *out_t,
                                  ds4_gpu_tensor *in_t,
                                  uint64_t bytes);
int ds4_gpu_tp_big_gate_wait(uint64_t seq);
/* Pause/resume the DVFS keep-alive around work that keeps the GPU busy.
 * No-op when TP is not bound. */
void ds4_gpu_tp_keepalive_pause(int paused);
/* Split attention heads across the two TP ranks in the GLM batch-prefill
 * attention kernels (qk-low, attention-lora, value-project). The caller
 * zeroes the unowned head range of the heads buffer and combines the
 * attn-output partials over the TP big-gate exchange. */
void ds4_gpu_tp_set_attn_head_split(int enabled);
/* Skip the whole-file model residency set (TP sharding: only the
 * owned ranges are warmed; the rest must never be paged in). Call before
 * the model is mapped. */
void ds4_gpu_model_residency_skip(int skip);
/* Nonzero after any gate exchange failed; the eval must abort. */
int ds4_gpu_tp_failed(void);

/* Tensor-parallel sliced projections (Metal decode path only).
 *
 * ds4_gpu_matmul_q8_0_kslice_tensor computes a k-range partial matvec:
 * out[out_dim] = W[:, k_off : k_off + k_cnt] @ x[x_elem_off : +k_cnt] where
 * W rows span full_in_dim quantized Q8_0 elements.  k offsets/counts must be
 * multiples of 32 (Q8_0 block).  Partial results from both ranks sum to the
 * full projection.
 *
 * ds4_gpu_attention_output_q8_tp_tensor is the group-sliced attention output
 * pair: low projection for groups [group0, group0+group_cnt) plus the
 * matching k-slice of the expand projection, producing this rank's partial
 * attention block output (n_tokens == 1 only). */
int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off);
/* CUDA multi-row variant. Each input row contains only the owned contiguous
 * K slice, while each output row spans the full projection width. */
int ds4_gpu_matmul_q8_0_kslice_rows_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              full_in_dim,
        uint64_t              out_dim,
        uint64_t              k_off,
        uint64_t              k_cnt,
        const ds4_gpu_tensor *x,
        uint64_t              n_rows);
int ds4_gpu_matmul_quant_kslice_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off);
int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads);

