#ifndef DS4X_BACKEND_COMMON_CUH
#define DS4X_BACKEND_COMMON_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include "ds4_mmq.h"
#include "ds4_repack.h"
#include "ds4x/cutlass_fp16_gemm.h"
#include "ds4x/fp16_projection.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define CUDA_QK_K 256


enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    /* 1M HCA has 8192 compressed rows plus up to 256 raw rows. */
    DS4_CUDA_ATTENTION_SCORE_CAP = 8448u,
    DS4_CUDA_ATTENTION_RAW_SCORE_CAP = 256u,
    DS4_CUDA_TOPK_MERGE_GROUP = 8u,
    /* perf-02 split-KV: fixed logical rows per chunk (shared scores = 2KB),
     * grid S = ceil(n_score / CHUNK) clamped, so block count grows with ctx. */
    DS4_CUDA_SPLITKV_CHUNK = 512u,
    DS4_CUDA_SPLITKV_SCORE_CAP = 512u,
    DS4_CUDA_SPLITKV_S_MAX = 16u,
    DS4_CUDA_SPLITKV_S_FLOOR = 4u
};



/* struct ds4_gpu_tensor is defined in ds4_gpu.h (no longer opaque as of
 * the device-aware CUDA PR). Field layout includes the new device_id
 * tag and is read by the WITH_DEVICE-wrapped tensor APIs below. */

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;



typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;



typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;



typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;
#include "ds4_gpu_runtime.h"
#include "ds4x_kernel/tables/iq2_tables.cuh"


typedef struct {
    ds4_gpu_attention_decode_row row[DS4_GPU_ATTENTION_DECODE_BATCH_MAX];
} cuda_attention_decode_row_table;



static_assert(sizeof(cuda_attention_decode_row_table) <= 3072u,
              "attention row table must fit in CUDA kernel parameters");

extern const void *g_model_host_base;


extern const char *g_model_device_base;


extern uint64_t g_model_registered_size;


extern int g_model_registered;


extern int g_model_device_owned;


extern int g_model_range_mapping_supported;


extern int g_model_hmm_direct;


extern int g_model_fd;


extern const void *g_model_fd_host_base;


extern int g_model_direct_fd;


extern uint64_t g_model_direct_align;


extern uint64_t g_model_file_size;


extern int g_model_cache_full;


extern int g_cublas_ready;


extern int g_quality_mode;


extern int g_decode_fast_attention;


extern int g_decode_score_vec4;


extern int g_cuda_disable_qkv_rms_fused;


extern int g_cuda_no_window_attention;


extern int g_cuda_decode_heads8_online;


extern int g_cuda_decode_score4;


extern int g_cuda_decode_score8;


extern int g_cuda_no_decode_value512;


extern int g_cuda_no_top1;


extern int g_cuda_end_stream_sync;


extern int g_cuda_exact_score_split_graph;


extern int g_cuda_exact_score_split_ldg;


extern int g_cuda_exact_score_split_vec4;


extern int g_cuda_exact_score_split_vec4_plain;


extern int g_cuda_exact_score_split_dim2;


extern int g_cuda_exact_score_split_fuse_inv_rope;


extern int g_cuda_moe_decode_graph;




typedef struct {
    cudaGraph_t     graph;
    cudaGraphExec_t exec;
    cudaGraphNode_t score_node;
    cudaGraphNode_t final_node;
    cudaGraphNode_t rope_node;
    uint32_t        n_head;
    uint32_t        head_dim;
    uint32_t        S;
    uint32_t        final_threads;
    uint32_t        n_rot;
    int             fuses_inv_rope;
    int             valid;
} cuda_score_split_graph_cache;

extern cuda_score_split_graph_cache g_score_split_graph[DS4_MAX_GPUS];


void attention_decode_score_split_graph_destroy_one(int logical_tier);



typedef struct {
    cudaGraph_t     graph;
    cudaGraphExec_t exec;
    cudaGraphNode_t xq_node;
    cudaGraphNode_t gate_node;
    cudaGraphNode_t midq_node;
    cudaGraphNode_t down_node;
    uint32_t        n_expert;
    uint32_t        expert_in_dim;
    uint32_t        expert_mid_dim;
    uint32_t        out_dim;
    int             valid;
} cuda_moe_decode_graph_cache;

extern cuda_moe_decode_graph_cache g_moe_decode_graph[DS4_MAX_GPUS];


int cuda_q4_mma_ok(void);

int cuda_q4_mma_tile16_shmem_ok(int which_down);

void routed_moe_decode_graph_destroy_one(int logical_tier);



static_assert(DS4_MAX_GPUS == 1, "DS4X is a single-GB10 runtime");

extern ds4_gpu_ctx g_gpu[DS4_MAX_GPUS];


extern int         g_n_gpus;


/* Internal helper: resolve a tensor's device index. -1 (untagged) is
 * treated as device 0 for legacy callers. */
int ds4_tensor_device_idx(const ds4_gpu_tensor *t);
#define WITH_DEVICE(d)                                                      \
    for (int _wd_prev = -1, _wd_first = 1;                                  \
         _wd_first;                                                          \
         _wd_first = 0,                                                      \
         (_wd_prev >= 0 ? (void)cudaSetDevice(_wd_prev) : (void)0))         \
        if (cudaGetDevice(&_wd_prev) != cudaSuccess) { /* leave */ } else   \
        if (cudaSetDevice(d)           != cudaSuccess) { /* leave */ } else


struct cuda_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered;
    int arena_allocated;
};



struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};



struct cuda_q8_f16_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
    int device_id;          /* physical CUDA device id; 0 in single-tier */
};



enum cuda_derived_kind {
    CUDA_DERIVED_IQ2_XXS_ALIGNED_MOE = 4,
    CUDA_DERIVED_Q8_0_ALIGNED_DENSE = 5,
    CUDA_DERIVED_Q2_K_ALIGNED_MOE = 6,
};



struct cuda_derived_range {
    const void *host_base;
    uint64_t source_offset;
    uint64_t source_bytes;
    uint32_t kind;
    uint64_t in_dim;
    uint64_t out_dim;
    uint32_t group_count;
    uint64_t bytes;
    char *device_ptr;
};

extern std::vector<cuda_derived_range> g_derived_ranges;


extern const void *g_derived_replace_map;


extern uint64_t g_derived_artifact_bytes;


extern double g_derived_artifact_build_secs;


extern int g_derived_replaces_complete;


extern void *g_aligned_q81_scratch;


extern int g_q8_cache_suppressed;


extern int g_q8_f16_disabled_after_oom;


extern int g_q8_f16_budget_notice_printed;

#ifdef DS4_CUDA_HAVE_MXF4

extern void *g_indexer_mxf4_scratch;


extern uint64_t g_indexer_mxf4_scratch_bytes;


extern int g_indexer_mxf4_scratch_device;

#endif

int cuda_ok(cudaError_t err, const char *what);

extern "C" void ds4_gpu_decode_graphs_invalidate(void);

const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);

__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);

int cuda_aligned_iq2_enabled(void);

int cuda_aligned_q2k_enabled(void);

int cuda_aligned_q8_enabled(void);

const char *cuda_derived_weight_ptr(
        const void *model_map,
        uint64_t source_offset,
        uint64_t source_bytes,
        uint32_t kind,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t group_count,
        uint64_t bytes);

int cuda_model_map_replaces_complete(const void *model_map);

int cuda_integrated_artifact_map(const void *model_map);

/* Startup combines adjacent tensors into cache spans.  A span is replaceable
 * when aligned IQ2/Q2K artifacts cover every tensor payload in it; GGUF
 * alignment gaps of at most 64 KiB do not need device residency. */
int cuda_span_fully_replaced(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes);

void *tt_scratch_ensure(uint64_t bytes, const char *what);

uint64_t tt_align256_u64(uint64_t x);

void *cuda_tmp_alloc_on(int logical_tier, uint64_t bytes, const char *what);

int cuda_attention_score_buffer_fits(uint32_t n_comp);

const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what);

cublasHandle_t cuda_cublas_for_tier(int logical_tier);
#define CUDA_DECODE_GRAPH_LAYERS   64u
#define CUDA_DECODE_GRAPH_ISLANDS   2u
#define CUDA_DECODE_GRAPH_VARIANTS  4u


/* Mirrors the public `struct ds4_decode_graph_key` decl in ds4_gpu.h
 * byte-for-byte (ds4_cuda.cu does not include that header; it carries
 * its own extern "C" signatures inline).  The static_assert catches
 * accidental drift: adding a field means updating both sites. */
struct ds4_decode_graph_key {
    uint32_t il;
    uint32_t island;
    uint32_t variant;
    uint32_t _pad;
    void    *cur_hc;
    void    *after_attn_hc;
    void    *after_ffn_hc;
    void    *attn_norm;
};


typedef struct ds4_decode_graph_key ds4_decode_graph_key;


static_assert(sizeof(ds4_decode_graph_key) == 48u,
              "ds4_decode_graph_key must match the ds4_gpu.h decl");



struct cuda_decode_graph_entry {
    ds4_decode_graph_key key;
    cudaGraphExec_t      exec;
    int                  state;   /* 0 empty, 1 warmed, 2 ready, 3 dead */
    uint64_t             hits;
};

extern "C" int ds4_gpu_decode_graphs_supported(void);

/* Stream the decode-island kernels launch on.  Legacy NULL stream in
 * eager mode (unchanged behavior); the capture stream while a capture
 * or replay is in flight. */
cudaStream_t cuda_decode_stream(void);

extern "C" void ds4_gpu_decode_graphs_invalidate(void);

extern "C" int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key);

extern "C" int ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key);

/* ------------------------------------------------------------------------
 * Vendored llama.cpp MMQ prefill tier (quantization/mmq/, see VENDOR.md).
 *
 * Ported from the Entrpi/ds4 batched-serving fork (fork commits 39d3877c,
 * a56e07a5, 944482d5 and the Phase 5/6 MoE wiring; author Entrpi
 * <entrpi@proton.me>).  Phase 1 takes only the RAW-layout entries: dense
 * Q8_0 GEMMs and the IQ2_XXS gate/up + Q2_K down routed-MoE pipeline for
 * n_tok >= 2 (prefill and batched verify).  The aligned-SoA / D2R /
 * producer-quantized tiers and the weight server are deliberately left
 * behind: they need the repack artifact pipeline for a further ~25% on
 * top of the ~2.5x this tier gives.  Decode (n_tok == 1) is untouched.
 * mmq changes FP32 reduction order vs the cublas+dequant pipeline, so
 * prefill logits drift at ULP scale: validated against the official
 * continuation vectors rather than byte-diffs.  DS4_CUDA_MMQ=0 restores
 * the legacy dispatch. */
int cuda_use_mmq(void);

int cuda_use_mxfp4_mmq(void);

/* mmq pipeline helpers, ported from the fork: SwiGLU + clamp + router
 * weight in the (token, slot, feature) layout ds4_mmq_*_moe writes, and
 * the guarded per-token slot sum. */
__global__ static void moe_mmq_swiglu_weighted_clamp_kernel(
        float *mid_out,
        const float *gate_buf, const float *up_buf,
        const float *weights,
        uint32_t expert_mid_dim,
        uint32_t n_tokens,
        uint32_t n_expert_used,
        float clamp);

__global__ static void moe_mmq_sum_kernel(float *out, const float *down,
        const int32_t *selected, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_tokens, uint32_t guard_nonfinite);

/* Abort an in-flight capture after an encode error inside the island.
 * Nothing was executed; the caller re-encodes eagerly. */
extern "C" void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key);

const char *cuda_resolve_weight_ptr(const void *model_map,
                                    uint64_t offset,
                                    uint64_t bytes,
                                    int logical_tier,
                                    const char *label);

int cuda_model_range_is_cached(const void *model_map, uint64_t offset, uint64_t bytes);

void cuda_q8_f16_cache_release_all(void);

uint32_t cuda_parse_u32_env_clamped(const char *name, uint32_t fallback,
                                           uint32_t min_value, uint32_t max_value,
                                           int *present);

int cuda_env_flag_enabled(const char *name, int fallback);

bool cuda_splitkv_decode_requested(void);

void cuda_q8_f16_cache_disable_after_failure(const char *what, uint64_t request_bytes);

int cuda_q8_use_dp4a(void);

unsigned cuda_q8_exact_threads(uint64_t blocks);

int cuda_q8_f16_preload_allowed(const char *label, uint64_t in_dim, uint64_t out_dim);

const __half *cuda_q8_f16_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        int expected_device,
        const char *label);

int cuda_ok(cudaError_t err, const char *what);

double cuda_wall_sec(void);

int cuda_model_prefetch_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size);

const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);

int cuda_model_copy_chunked(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size);

void cuda_model_range_release_all(void);

int cublas_ok(cublasStatus_t st, const char *what);

extern "C" int ds4_gpu_init(void);

extern "C" void ds4_gpu_cleanup(void);

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);

extern "C" int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int device_id,
                                       uint64_t bytes);

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes);

/* Heap-allocated tensor with the explicit device-slot ABI. */
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int tier, uint64_t bytes);

/* Managed-memory counterpart of ds4_gpu_tensor_alloc_ptr_on. */
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int tier, uint64_t bytes);

extern "C" int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);

extern "C" void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);

extern "C" uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);

extern "C" void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);

extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);

extern "C" int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);

extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);

extern "C" int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                     const ds4_gpu_tensor *src, uint64_t src_offset,
                                     uint64_t bytes);

extern "C" int ds4_gpu_pack_slot_rows_f32_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *slots,
        uint32_t                n_rows,
        uint32_t                width,
        uint32_t                n_slots,
        uint32_t                slot_cap);

extern "C" int ds4_gpu_begin_commands(void);

extern "C" int ds4_gpu_flush_commands(void);

extern "C" int ds4_gpu_end_commands(void);

extern "C" int ds4_gpu_synchronize(void);

extern "C" int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);

/* Register the mmap'd host model without forcing a full device copy. */
extern "C" int ds4_gpu_register_model_map_no_copy(const void *model_map, uint64_t model_size);

extern "C" int ds4_gpu_set_model_fd(int fd);

extern "C" int ds4_gpu_build_derived_artifacts(
        const void *model_map,
        uint64_t model_size,
        const char *model_path);

extern "C" int ds4_gpu_model_range_replaced(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes);

extern "C" int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);

extern "C" int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);

extern "C" void ds4_gpu_print_memory_report(const char *label);

extern "C" void ds4_gpu_set_quality(bool quality);

__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_embd, uint32_t n_hc);

__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc);

__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok);

__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc);

__global__ static void repeat_hc_rows_kernel(float *out, const float *rows, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc);

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n);

__device__ static float warp_sum_f32(float v);

__device__ static float warp_max_f32(float v);

__device__ static float dot4_f32(float4 a, float4 b);

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p);

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p);

__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b);

__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a);

__global__ static void quantize_q8_0_f32_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t in_dim,
        uint64_t blocks);

__global__ static void quantize_q8_0_group_slice_rows_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t group_dim,
        uint64_t blocks,
        uint32_t n_groups_total,
        uint32_t group0,
        uint32_t group_cnt);

__global__ static void matmul_q8_0_preq_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a);

__global__ static void matmul_q8_0_preq_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a);


__global__ static void matmul_q8_0_pair_preq_warp8_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t blocks,
        int use_dp4a);

__global__ static void matmul_q8_0_pair_preq_batch_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a);

__global__ static void matmul_q8_0_pair_preq_batch_tok2_exact_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a);

__global__ static void matmul_q8_0_hc_expand_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *block_add2,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add,
        int has_add2,
        int use_dp4a);

__global__ static void matmul_q8_0_kslice_hc_expand_add_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t slice_dim,
        uint64_t out_dim,
        uint64_t full_blocks,
        uint64_t block_start,
        uint64_t slice_blocks,
        uint32_t n_embd,
        uint32_t n_hc,
        int use_dp4a);

__global__ static void matmul_q8_0_preq_batch_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a);

__global__ static void matmul_q8_0_preq_batch_warp8_tok2_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a);

__global__ static void matmul_q8_0_preq_batch_warp8_tok4_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a);

__global__ static void matmul_q8_0_preq_batch_warp8_tok8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a);

__global__ static void matmul_q8_0_preq_batch_tok2_exact_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a);

/* ---- INT8 tensor-core exact Q8_0 batch matmul --------------------------
 * Bit-identical replacement for the exact tok2/warp8-family batched Q8_0
 * kernels. Each output element's reduction is the reference's strided
 * halving tree over T slots (T = reduction width: 32 for the warp kernels,
 * cuda_q8_exact_threads(blocks) for the exact kernels; slots >= blocks hold
 * +0.0f). The kernel decomposes that tree as: 32 streams at stride T/32
 * whose 32 terms per outer step j combine via an adjacent-pairwise static
 * register stack taken in bit-reversed stream order (== the top five strided
 * tree levels), plus per-(j&3) sequential accumulators and a fixed tail for
 * the remaining levels. Fuzz-verified bitwise against both reference
 * kernels across shapes, including blocks < T and ragged out_dim/n_tok.
 * Rollback: DS4_CUDA_NO_Q8_MMA=1. */
__device__ __forceinline__ static uint32_t ldu32_unaligned(const uint8_t *p);

__device__ __forceinline__ static void mma_m16n8k32_s8(
        int32_t &c0, int32_t &c1, int32_t &c2, int32_t &c3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1);

__device__ __forceinline__ static uint32_t bitrev5(uint32_t i);

template <uint32_t T>
__global__ static void matmul_q8_0_mma_exact_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        uint64_t a_stride_blocks, /* activation row stride in blocks (>= blocks) */
        uint64_t out_stride);

int cuda_q4_mma_ok(void);

extern int cuda_q8_mma_attr_ready[DS4_MAX_GPUS][4];


int cuda_q8_mma_try_launch(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        uint64_t a_stride_blocks,
        uint64_t out_stride,
        uint32_t T);

__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);

__global__ static void grouped_q8_0_a_preq_warp8_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a);

__global__ static void grouped_q8_0_a_preq_warp8_tok2_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a);

__global__ static void rms_norm_plain_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps);

/* Latency-optimized RMS norm for the common n==4096 decode shape: one global
 * read pass with register-batched loads, same per-thread accumulation order
 * and shared-memory tree as rms_norm_plain_kernel (bit-identical, fuzz
 * checked). */
__global__ static void rms_norm_plain_fast4096_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps);

/* Batched-load RMS norm for larger rows (n multiple of 2048, e.g. the 16384
 * HC-concatenated decode rows). Two passes like the reference kernel, but
 * eight independent loads are issued per accumulation group; the per-thread
 * accumulation order (ascending i with stride 256) is unchanged, so results
 * are bit-identical. */
__global__ static void rms_norm_plain_batch8_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps);

__global__ static void rms_norm_weight_kernel(float *out, const float *x, const float *w, uint32_t n, uint32_t rows, float eps);

__global__ static void dsv4_qkv_rms_norm_rows_kernel(
        float *q_out,
        const float *q,
        const float *q_w,
        uint32_t q_n,
        float *kv_out,
        const float *kv,
        const float *kv_w,
        uint32_t kv_n,
        uint32_t rows,
        float eps);

__global__ static void head_rms_norm_kernel(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps);

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0);

__global__ static void dsv4_qkv_rms_norm_rows_kv_rope_kernel(
        float *q_out,
        const float *q,
        const float *q_w,
        uint32_t q_n,
        float *kv_out,
        const float *kv,
        const float *kv_w,
        uint32_t kv_n,
        uint32_t rows,
        uint32_t kv_n_head,
        uint32_t kv_head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps);

__global__ static void head_rms_norm_rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps);

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0);

__global__ static void rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t pos_stride,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow);


__device__ static float dsv4_e4m3fn_value_dev(int i);

__device__ static float dsv4_e4m3fn_dequant_dev(float x);

__device__ static unsigned char dsv4_e4m3fn_encode_dev(float x);

__device__ __forceinline__ static half spark_kv_decode_half(
        const unsigned char *row, uint32_t dim);

__device__ __forceinline__ static float spark_kv_decode(
        const unsigned char *row, uint32_t dim);

__device__ static float dsv4_e2m1fn_value_dev(int i);

__device__ static float dsv4_e2m1fn_dequant_dev(float x);

__device__ static uint32_t dsv4_e2m1fn_encode_dev(float x);

__device__ static float model_scalar_dev(const void *base, uint64_t offset, uint32_t type, uint64_t idx);

__device__ static void fp8_kv_quantize_row(
        float    *xr,
        uint32_t  head_dim,
        uint32_t  n_rot,
        float    *scratch);

__global__ static void fp8_kv_quantize_kernel(
        float    *x,
        uint32_t  n_tok,
        uint32_t  head_dim,
        uint32_t  n_rot);

__device__ static void spark_pack_kv_row(
        float *src, unsigned char *dst, bool quantize_in_place);

__global__ static void spark_pack_kv_rows_kernel(
        unsigned char *dst, uint64_t dst_row,
        float *src, uint32_t src_row, uint32_t rows,
        bool quantize_in_place);

__global__ static void spark_pack_kv_ring_rows_kernel(
        unsigned char *dst, uint32_t raw_cap, uint32_t pos0,
        float *src, uint32_t rows);

__global__ static void spark_pack_kv_decode_rows_kernel(
        float *src, cuda_attention_decode_row_table rows,
        uint32_t n_rows);

__global__ static void spark_pack_index_rows_kernel(
        unsigned char *dst, uint64_t dst_row,
        const float *src, uint32_t src_row, uint32_t rows);

__global__ static void spark_zero_kv_rows_kernel(
        unsigned char *dst, uint32_t rows);

__global__ static void spark_zero_index_rows_kernel(
        unsigned char *dst, uint32_t rows);

__device__ __forceinline__ static float spark_index_decode(
        const unsigned char *packed, uint32_t dim);

__global__ static void spark_unpack_index_rows_kernel(
        float *dst, const unsigned char *src, uint32_t rows);

__global__ static void spark_unpack_kv_rows_kernel(
        float *dst,
        const unsigned char *src,
        const int32_t *rows,
        uint32_t n_rows,
        uint32_t src_cap,
        uint32_t src_start,
        uint32_t ring);

__global__ static void indexer_hadamard_fp4_kernel(float *x, uint32_t n_rows, uint32_t head_dim);

__global__ static void spark_fill_iota_i32_kernel(int32_t *dst, uint32_t n);

__global__ static void attention_prefill_raw_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim);

__global__ static void attention_prefill_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim);

__global__ static void attention_prefill_raw_softmax_kernel(
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_keys);

__global__ static void attention_prefill_mixed_softmax_kernel(
        float *scores,
        const float *sinks,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys);

__global__ static void attention_prefill_pack_mixed_kv_kernel(
        float *dst,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t head_dim);

__global__ static void attention_prefill_unpack_heads_kernel(
        float *heads,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim);

__global__ static void attention_pack_group_heads_f16_kernel(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim);

__global__ static void attention_unpack_group_low_kernel(
        float *low,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t rank);

__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t score_lanes_single);

__global__ static void attention_decode_score_split_scores_kernel(
        float *score_out,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S);

__device__ __forceinline__ float ds4_dot_scalar_ldg(
        const float *a,
        const float *b,
        uint32_t n);

__global__ static void attention_decode_score_split_scores_ldg_kernel(
        float *score_out,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S);

int cuda_matmul_q8_0_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok, const char *label);

extern "C" int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);



extern "C" int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok);



int cuda_matmul_q8_0_hc_expand_tensor_labeled(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *block_add2,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label);

extern "C" int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);

extern "C" int ds4_gpu_matmul_f16_rms_fold_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        float norm_eps);


extern "C" int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok);

extern "C" int ds4_gpu_matmul_f16_pair_compressor_store_tensor(
        ds4_gpu_tensor *out_kv,
        ds4_gpu_tensor *out_score,
        ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_kv_offset,
        uint64_t weight_score_offset,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint64_t in_dim,
        uint32_t width,
        const ds4_gpu_tensor *x,
        uint32_t ratio,
        uint32_t pos);

extern "C" int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok);

extern "C" int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row, uint32_t n_embd, uint32_t n_hc);
#define DS4_SCORE_TILE_HEADS 16u
#define DS4_SCORE_TILE_ROWS 16u
#define DS4_SCORE_TILE_STRIDE 516u /* 512 + 4 floats: 16B-aligned rows, banks shifted by 4 */

/* Multi-session form of the exact tiled score kernel. Each z-slice selects a
 * private KV table entry, while every individual score keeps the same scalar
 * ascending-d FMA chain as the one-session kernel. */
__global__ static void attention_decode_score_split_scores_tile512_rows_kernel(
        float *score_out,
        const float *q,
        cuda_attention_decode_row_table rows,
        uint32_t n_rows,
        uint32_t score_stride,
        uint32_t n_head,
        uint32_t head_dim);

__device__ float ds4_dot512_float4_ordered(
        const float *a,
        const float *b);

__device__ float ds4_dot512_float4_plain(
        const float *a,
        const float *b);

__global__ static void attention_decode_score_split_finalize_rows_kernel(
        float *heads,
        const float *sinks,
        const float *score_in,
        cuda_attention_decode_row_table rows,
        uint32_t n_rows,
        uint32_t score_stride,
        uint32_t n_head,
        uint32_t head_dim);

__global__ static void attention_decode_global_softmax_kernel(
        float *score_inout,
        float *denom_out,
        const float *sinks,
        uint32_t n_score,
        uint32_t n_head);

__global__ static void attention_decode_split_value_kernel(
        float *partials,
        const float *score_exp,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t raw_count,
        uint32_t raw_first_idx,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_score,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S);

__global__ static void attention_decode_split_value_combine_kernel(
        float *heads,
        const float *partials,
        const float *denom,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S);



typedef struct {
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
} cuda_attention_inv_rope_params;

void attention_decode_score_split_graph_destroy_one(int logical_tier);

int attention_decode_score_split_launch(
        int logical_tier,
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t final_threads,
        const cuda_attention_inv_rope_params *inv_rope);

/* ---- perf-02 split-KV / flash-decode (opt-in, default OFF) ----------------
 *
 * attention_decode_splitkv_kernel computes a partial online-softmax over a
 * contiguous chunk of the flattened logical row set [0, n_score) used by
 * attention_decode_mixed_kernel (raw rows first, then compressed rows, same
 * ascending ordering). Each block handles (t = blockIdx.x, h = blockIdx.y,
 * chunk = blockIdx.z) and writes a partial (m_j, l_j, acc_j[head_dim]) WITHOUT
 * the sink term. attention_decode_splitkv_combine_kernel merges the S partials
 * per (t,h), folds the sink once, and writes the final normalized head output.
 *
 * The math is the standard flash-attention online-softmax rescale and is
 * algebraically identical to attention_decode_mixed_kernel; it is NOT
 * guaranteed bit-identical in FP32 (different expf inputs + add/mul grouping),
 * hence default-OFF behind DS4_CUDA_SPLITKV_DECODE and the S==1 dispatch to the
 * old kernel as the bit-exact anchor (handled in the launch helper).
 *
 * Partials scratch layout (per logical tier), contiguous floats:
 *   stride = head_dim + 2
 *   base(t,h,j) = ((t*n_head + h)*S + j) * stride
 *     [0]               = m_j   (chunk running max; -INF if empty/all-masked)
 *     [1]               = l_j   (chunk denominator sum exp(s - m_j))
 *     [2 .. 2+head_dim) = acc_j[head_dim] (chunk weighted value sum)
 */
__global__ static void attention_decode_splitkv_kernel(
        float *partials,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S);

__global__ static void attention_decode_splitkv_combine_kernel(
        float *heads,
        const float *sinks,
        const float *partials,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t S);

__device__ __forceinline__ void attention_compact_topk_stable(
        uint32_t *comp_rows,
        uint32_t *comp_count,
        uint32_t *warp_offsets,
        const int32_t *topk,
        uint32_t top_k,
        uint32_t visible_comp);

__global__ static void attention_indexed_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim);

__global__ static void attention_indexed_mixed_decode_rows_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        cuda_attention_decode_row_table rows,
        uint32_t n_rows,
        uint32_t n_head,
        uint32_t head_dim);

__global__ static void attention_indexed_mixed_heads8_rb4_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim);

template <uint32_t ROWS_PER_STAGE, uint32_t HEADS_PER_GROUP>
__global__ static void __launch_bounds__(512, 2)
attention_indexed_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim);

__global__ static void attention_static_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim);




/* -------------------------------------------------------------------------
 * DS4_ATTN_TOKENTILE: token-tile HMMA indexed attention (opt-in).
 * Ported from the pass-12 standalone prototype with fixed STAGE_ROWS=32.
 * Four-token tiles reduce the selected-row union while eight heads keep the
 * 32-row MMA shape full.
 */

inline constexpr uint32_t kTTTileTokens = 4u;


inline constexpr uint32_t kTTG = 8u;


inline constexpr uint32_t kTTM = 32u;


inline constexpr uint32_t kTTStageRows = 32u;


inline constexpr uint32_t kTTRawWindow = 128u;


inline constexpr uint32_t kTTHeadDim = 512u;


inline constexpr uint32_t kTTWarps = 16u;


inline constexpr uint32_t kTTThreads = 512u;


inline constexpr uint32_t kTTScoreKQuarters = 4u;


inline constexpr uint32_t kTTScoreKSliceDim = kTTHeadDim / kTTScoreKQuarters;


inline constexpr uint32_t kTTScoreKStepsPerQuarter = kTTScoreKSliceDim / 16u;


inline constexpr uint32_t kTTRecordRingPlanes = 4u;


inline constexpr uint32_t kTTProbStride = 40u;


inline constexpr uint32_t kTTRingChunkBytes = 16u;


inline constexpr uint32_t kTTRingChunksPerRow =
    (kTTHeadDim * sizeof(half)) / kTTRingChunkBytes;


inline constexpr size_t kTTSmemHardCap = 90ull * 1024ull;



static_assert(kTTM == kTTTileTokens * kTTG, "token-tile M must be 4 tokens x G8");


static_assert(kTTProbStride == kTTStageRows + 8u, "token-tile prob stride changed");


static_assert(kTTScoreKSliceDim % 16u == 0, "token-tile score K split changed");


static_assert(kTTRingChunksPerRow == 64u, "token-tile KV ring expects 64 chunks");


static_assert(sizeof(int2) == 8u, "token-tile union record must remain 8 bytes");



template <uint32_t TT_STAGE_ROWS>
struct tt_TokentileLayout {
    static constexpr uint32_t prob_stride = TT_STAGE_ROWS + 8u;
    static constexpr uint32_t ring_plane_bytes =
        TT_STAGE_ROWS * kTTRingChunksPerRow * kTTRingChunkBytes;
    static constexpr uint32_t ring_plane_elems = ring_plane_bytes / sizeof(half);
};

__device__ static float tt_warp_sum_f32(float v);

__device__ static float tt_warp_max_f32(float v);

__device__ __forceinline__ uint32_t tt_lane_id(void);

__device__ __forceinline__ uint32_t tt_warp_id(void);

__device__ __forceinline__ int tt_mma_c_i(uint32_t lane, int l);

__device__ __forceinline__ int tt_mma_c_j(uint32_t lane, int l);

__device__ __forceinline__ unsigned tt_smem_addr(const void *p);

__device__ __forceinline__ uint32_t tt_ring_off_bytes(uint32_t row, uint32_t c);

__device__ __forceinline__ void tt_ldmatrix_x4_addr(uint32_t (&r)[4], unsigned a);

__device__ __forceinline__ void tt_ldmatrix_x2_addr(uint32_t (&r)[2], unsigned a);

__device__ __forceinline__ void tt_ldmatrix_x2_trans_addr(uint32_t (&r)[2], unsigned a);

__device__ void tt_ldmatrix_x4(uint32_t (&r)[4], const void *p);

__device__ void tt_ldmatrix_x2(uint32_t (&r)[2], const void *p);

__device__ void tt_ldmatrix_x2_trans(uint32_t (&r)[2], const void *p);

__device__ __forceinline__ void tt_mma_m16n8k16_f16_f32(
        float *d,
        const uint32_t (&a)[4],
        const uint32_t (&b)[2]);

__device__ __forceinline__ unsigned char *tt_align16(unsigned char *p);

__device__ __forceinline__ void tt_zero_16B(void *dst);

__device__ __forceinline__ void tt_zero_8B(void *dst);

__device__ __forceinline__ void tt_cp_async_16B(void *dst, const void *src, bool pred);

__device__ __forceinline__ void tt_cp_async_8B(void *dst, const void *src, bool pred);

__device__ __forceinline__ void tt_cp_async_commit(void);

template <int KeepGroups>
__device__ __forceinline__ void tt_cp_async_wait_group(void);

__device__ __forceinline__ uint32_t tt_score_partial_slot(
        uint32_t kq,
        uint32_t m,
        uint32_t r);

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_store_score_partial(
        float4 * __restrict__ partials,
        uint32_t kq,
        uint32_t m,
        uint32_t r,
        float v);

__device__ __forceinline__ float4 tt_load_score_partial_record(
        const float4 * __restrict__ p);

__device__ __forceinline__ void tt_issue_cp_async_row(
        half * __restrict__ dst,
        uint32_t rr,
        uint32_t lane16,
        const half * __restrict__ src,
        bool live);

__device__ __forceinline__ uint32_t tt_stage_raw_rows(
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count);

__global__ static void __launch_bounds__(512, 1) attention_tokentile_union_build_kernel(
        int2 *records,
        uint32_t *counts,
        const int32_t *topk,
        const int32_t *positions,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t rec_stride);

__global__ static void __launch_bounds__(256, 1) attention_tokentile_raw_mirror_kernel(
        half *dst,
        const float *raw_kv,
        const int32_t *seq_id,
        uint32_t tt_run_pos0,
        uint32_t n_tokens,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t first_raw_pos,
        uint32_t raw_row_min,
        uint32_t head_dim);

__global__ static void __launch_bounds__(256, 1) attention_tokentile_comp_mirror_kernel(
        half *dst,
        const float *comp_kv,
        uint32_t n_comp,
        uint32_t head_dim);

/* Packed prefill feeds HMMA directly through FP16 mirrors. Persistent rows
 * remain in the 583-byte cache format; no transient F32 history is created. */
__global__ static void __launch_bounds__(256, 1)
attention_tokentile_raw_packed_mirror_kernel(
        half *dst,
        const unsigned char *raw_kv,
        uint32_t tt_run_pos0,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t first_raw_pos,
        uint32_t raw_row_min,
        uint32_t head_dim);

__global__ static void __launch_bounds__(256, 1)
attention_tokentile_comp_packed_mirror_kernel(
        half *dst,
        const unsigned char *comp_kv,
        uint32_t n_comp,
        uint32_t head_dim);

/* Non-indexed layers select the causal compressed range [0, visible), so
 * their per-tile records can be built directly without a bitmap or sort. */
__global__ static void __launch_bounds__(512, 1)
attention_tokentile_dense_build_kernel(
        int2 *records,
        uint32_t *counts,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t ratio,
        uint32_t n_comp,
        uint32_t rec_stride);

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_issue_record_stage_cp_async(
        int2 * __restrict__ rec_plane,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        const int2 * __restrict__ union_records_tile);

template <uint32_t TT_STAGE_ROWS, bool USE_SMEM_RECORDS>
__device__ __forceinline__ void tt_issue_kv_stage_cp_async(
        half * __restrict__ dst,
        const int2 * __restrict__ rec_plane,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        const int2 * __restrict__ union_records_tile,
        uint32_t tile_base,
        const half * __restrict__ raw_kv,
        const half * __restrict__ comp_kv,
        uint32_t tid);

__device__ __forceinline__ void tt_load_score_q_frag(
        uint32_t (&q_frag)[kTTScoreKStepsPerQuarter][4],
        const float * __restrict__ q,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t tile_base,
        uint32_t head_base);

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_hmma_score_stage(
        float4 * __restrict__ partial_scores,
        const uint32_t (&q_frag)[kTTScoreKStepsPerQuarter][4],
        const half * __restrict__ kv_cur,
        uint32_t nr,
        float score_scale);

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_softmax_stage(
        half * __restrict__ probs,
        float * __restrict__ stage_rescale,
        float * __restrict__ max_s,
        float * __restrict__ sum_s,
        const float4 * __restrict__ scores,
        const int2 * __restrict__ records,
        uint32_t row0,
        uint32_t nr,
        uint32_t raw_union_count,
        uint32_t tile_count,
        uint32_t tile_base,
        uint32_t raw_row_min);

template <uint32_t TT_STAGE_ROWS>
__device__ __forceinline__ void tt_pv_mma_stage(
        float (&o_acc)[2u * kTTTileTokens * kTTG],
        const half * __restrict__ probs,
        const float * __restrict__ stage_rescale,
        const half * __restrict__ kv_cur);

__device__ __forceinline__ void tt_pv_mma_epilogue(
        const float (&o_acc)[2u * kTTTileTokens * kTTG],
        const float * __restrict__ final_scale,
        float * __restrict__ heads,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t tile_base,
        uint32_t head_base);

__global__ static void __launch_bounds__(512, 1) attention_tokentile_hmma_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const half *raw_kv,
        const half *comp_kv,
        const int2 *union_records,
        const uint32_t *union_counts,
        uint32_t rec_stride,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t raw_row_min);



constexpr size_t tt_align16_const(size_t x) {
    return (x + 15u) & ~size_t(15u);
}



template <uint32_t TT_STAGE_ROWS, uint32_t TT_G>
struct tt_TokentileSmemBudget {
    static constexpr uint32_t M = kTTTileTokens * TT_G;
    static constexpr uint32_t prob_stride = tt_TokentileLayout<TT_STAGE_ROWS>::prob_stride;
    static constexpr size_t q_bytes = 0;
    static constexpr size_t ring_bytes = 2ull * tt_TokentileLayout<TT_STAGE_ROWS>::ring_plane_bytes;
    static constexpr size_t p_bytes = 2ull * M * prob_stride * sizeof(half);
    static constexpr size_t partial_records = (size_t)M * TT_STAGE_ROWS;
    static constexpr size_t partial_bytes = partial_records * sizeof(float4);
    static constexpr size_t stats_bytes = 4ull * M * sizeof(float);
    static constexpr size_t record_bytes =
        (size_t)kTTRecordRingPlanes * TT_STAGE_ROWS * sizeof(int2);
    static constexpr size_t total =
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(
        tt_align16_const(q_bytes) + ring_bytes) + p_bytes) +
        partial_bytes) + stats_bytes) + record_bytes);
};



static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::p_bytes == 5120ull,
              "M32/R32 P double-buffer must use R+8 stride");


static_assert(sizeof(float4) == 16u, "score partial records must stay 16 bytes");


static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::partial_bytes == 16ull * 1024ull,
              "M32/R32 score partial records must be M*R float4");


static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::record_bytes == 1024ull,
              "M32/R32 record ring must be four R-row int2 planes");


static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::total == 88576ull,
              "M32/R32 total dynamic shared memory changed unexpectedly");


static_assert(tt_TokentileSmemBudget<kTTStageRows, kTTG>::total <= kTTSmemHardCap,
              "token-tile dynamic shared memory must stay under the 90 KiB pass gate");

int ds4_cuda_attn_tokentile_arch_ok(void);

__global__ static void __launch_bounds__(256, 4)
attention_decode_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim);

__device__ static void hc4_split_one(float *out, const float *mix, const float *scale, const float *base, uint32_t sinkhorn_iters, float epsv);

__global__ static void hc_split_sinkhorn_kernel(float *out, const float *mix, const float *scale, const float *base, uint32_t n_rows, uint32_t sinkhorn_iters, float epsv);

__global__ static void hc_weighted_sum_kernel(float *out, const float *x, const float *w, uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens, uint32_t weight_stride_f32);

__global__ static void hc_expand_kernel(
        float *out_hc,
        const float *block_out,
        const float *block_add,
        const float *block_add2,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add,
        int has_add2);

__global__ static void hc_split_weighted_sum_fused_kernel(
        float *out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv);

__global__ static void hc_split_weighted_sum_norm_fused_kernel(
        float *out,
        float *norm_out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        const float *norm_w,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps);

__global__ static void output_hc_weights_kernel(
        float *out,
        const float *pre,
        const float *scale,
        const float *base,
        uint32_t n_hc,
        uint32_t n_tokens,
        float epsv);

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);

__global__ static void compressor_store_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens);

__global__ static void compressor_set_rows_kernel(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t src0,
        uint32_t dst0,
        uint32_t rows);

__global__ static void compressor_prefill_pool_kernel(
        float *comp,
        const float *kv,
        const float *sc,
        const float *state_kv,
        const float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_comp,
        uint32_t replay);

__global__ static void compressor_update_pool_kernel(
        float *row,
        const float *state_kv,
        const float *state_score,
        uint32_t head_dim,
        uint32_t ratio);

__global__ static void compressor_shift_ratio4_kernel(float *state_kv, float *state_score, uint32_t width);

__device__ static float softplus_dev(float x);

__global__ static void router_select_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode);

__global__ static void router_select_parallel_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode);

__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi);

__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode);

__global__ static void swiglu_kernel(float *out, const float *gate, const float *up, uint32_t n, float clamp, float weight);

__global__ static void add_kernel(float *out, const float *a, const float *b, uint32_t n);

__global__ static void directional_steering_project_kernel(
        float       *x,
        const float *directions,
        uint32_t     layer,
        uint32_t     width,
        uint32_t     rows,
        float        scale);

__global__ static void zero_kernel(float *out, uint64_t n);

/* Slow, transparent B1 oracle for the persistent 68-byte indexer ABI. It is
 * only selected by the A/B regression; production uses block-scaled MMA. */
__global__ static void spark_indexer_score_one_reference_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const unsigned char *index_comp,
        uint32_t n_comp,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal);

#ifdef DS4_CUDA_HAVE_MXF4
#define DS4_MXF4_INDEXER_HEADS 64u
#define DS4_MXF4_INDEXER_DIM 128u
#endif

__device__ __forceinline__ static bool topk_score_better(float av, uint32_t ai, float bv, uint32_t bi);

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k);

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_u16_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k);

template <uint32_t SORT_N>
__global__ static void indexer_topk_chunk_pow2_kernel(
        uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_stride);

template <uint32_t SORT_N>
__global__ static void indexer_topk_merge_pow2_kernel(
        uint32_t *selected,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_count,
        uint32_t candidate_stride);

template <uint32_t SORT_N>
__global__ static void indexer_topk_tree_merge_pow2_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_sets,
        uint32_t merge_group,
        uint32_t candidate_stride,
        uint32_t out_stride);

__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens);

extern "C" int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc);

extern "C" int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

extern "C" int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale);

extern "C" int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

extern "C" int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

extern "C" int ds4_gpu_dspark_markov_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits_row,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              w1_offset,
        uint64_t              w2_offset,
        uint32_t              prev_token,
        uint32_t              vocab,
        uint32_t              rank);

extern "C" int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

extern "C" int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

extern "C" int ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_head,
        uint32_t                head_dim);

extern "C" int ds4_gpu_repeat_hc_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *rows, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc);

extern "C" int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, float eps);

extern "C" int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, uint32_t rows, float eps);

extern "C" int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps);

extern "C" int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps);

extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps);

extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        uint32_t                kv_n_head,
        uint32_t                kv_head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        bool                    inverse,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   eps);

extern "C" int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps);

extern "C" int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps);

extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot);

extern "C" int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim);

extern "C" int ds4_gpu_spark_pack_kv_rows_tensor(
        ds4_gpu_tensor *dst, uint64_t dst_row,
        const ds4_gpu_tensor *src, uint32_t src_row, uint32_t rows);

extern "C" int ds4_gpu_spark_pack_index_rows_tensor(
        ds4_gpu_tensor *dst, uint64_t dst_row,
        const ds4_gpu_tensor *src, uint32_t src_row, uint32_t rows);

extern "C" int ds4_gpu_spark_zero_kv_rows_tensor(
        ds4_gpu_tensor *dst, uint32_t rows);

extern "C" int ds4_gpu_spark_zero_index_rows_tensor(
        ds4_gpu_tensor *dst, uint32_t rows);

extern "C" int ds4_gpu_spark_unpack_kv_rows_tensor(
        ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint32_t rows);

extern "C" int ds4_gpu_spark_unpack_index_rows_tensor(
        ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint32_t rows);

extern "C" int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);


extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim);

extern "C" int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row,
        uint32_t          head_dim,
        uint32_t          n_rot);


extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim);

extern "C" int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim);

extern "C" int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens);

extern "C" int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        bool                    state_already_stored,
        bool                    decode_one_token,
        bool                    defer_finalize);

extern "C" int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

extern "C" int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

extern "C" int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0);

template <bool INDEXED>
__global__ static void __launch_bounds__(256, 2)
spark_attention_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const unsigned char *raw_kv,
        const unsigned char *comp_kv,
        const int32_t *topk,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t n_head);

__global__ static void spark_unpack_kv_rows_kernel(
        float *dst,
        const unsigned char *src,
        const int32_t *rows,
        uint32_t n_rows,
        uint32_t src_cap,
        uint32_t src_start,
        uint32_t ring);

extern "C" int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_format,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim);



extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim);

extern "C" int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

extern "C" int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_format,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

extern "C" int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_format,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);


extern "C" int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

extern "C" int ds4_gpu_attention_output_low_q8_rows_exact_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads,
        uint32_t                n_rows);

extern "C" int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads);


extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight);

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);


extern "C" int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n);

extern "C" int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

extern "C" int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits);

extern "C" int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens);

__device__ static float dev_f16_to_f32(uint16_t v);

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v);

__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid, uint32_t sign, const int8_t *q8, int32_t acc);

__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift);

__device__ static int32_t dev_dot_iq2_pair_16(uint8_t grid0, uint32_t sign0, uint8_t grid1, uint32_t sign1, const int8_t *q8);

__device__ __forceinline__ static void dev_iq2_i8x8_lut(
        const uint64_t *grid,
        const uint8_t *signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1);

__device__ static float dev_dot_iq2_xxs_q8_K_block_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs);

__device__ static float dev_dot_iq2_xxs_q8_K_block(const cuda_block_iq2_xxs *x, const cuda_block_q8_K *y);

__device__ static void dev_dot_iq2_xxs_q8_K_block8_deq_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8],
        const uint64_t *grid,
        const uint8_t *signs);

__device__ static void dev_dot_iq2_xxs_q8_K_block4(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]);

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out);

__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift);

__device__ static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *x, const cuda_block_q8_K *y);

/* Vector-load variant of dev_dot_q4_K_q8_K_block: loads the whole 144-byte
 * Q4_K block with nine 16B loads (requires a 16B-aligned tensor base; block
 * stride 144 and row strides are 16B multiples), then computes the exact same
 * integer sums and float finish. Same values in the same order, so results
 * are bit-identical; the wide loads just improve DRAM/memory-level
 * parallelism for the bandwidth-bound decode matvecs. */
__device__ __forceinline__ static void dev_dot_q4_K_q8_K_block_vec(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y,
        float *out_acc);

__device__ static void dev_dot_q4_K_q8_K_block8(
        const cuda_block_q4_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]);

__device__ static float dev_dot_q2_K_q8_K_block(const cuda_block_q2_K *x, const cuda_block_q8_K *y);

__device__ static void dev_dot_q2_K_q8_K_block4(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]);

__device__ static void dev_dot_q2_K_q8_K_block8(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]);

__device__ static float half_warp_sum_f32(float v, uint32_t lane16);

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8);

__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows);

/* Decode-only dual quantizer.  The Q8_0 half mirrors
 * quantize_q8_0_f32_kernel's 32-thread reduction and expression order, while
 * the Q8_K half remains byte-for-byte the ordinary routed-MoE quantizer. */
__global__ static void q8_K_q8_0_quantize_kernel(
        cuda_block_q8_K *out,
        int8_t *q8_0,
        float *q8_0_scale,
        const float *x,
        uint32_t in_dim,
        uint32_t n_rows);
__global__ static void q8_K_q8_0_quantize_kernel(
        cuda_block_q8_K *out,
        int8_t *q8_0,
        float *q8_0_scale,
        const float *x,
        uint32_t in_dim,
        uint32_t n_rows);





__global__ static void q8_K_quantize_sidecar_kernel(
        cuda_block_q8_K *out,
        const float *x,
        const float *amax_sidecar,
        uint32_t in_dim,
        uint32_t n_rows);
#ifndef MOE_DECODE_ROW_TILES
#define MOE_DECODE_ROW_TILES 1u
#endif
#define MOE_DECODE_ROWS_PER_BLOCK (32u * MOE_DECODE_ROW_TILES)

__global__ static void moe_gate_up_mid_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp);

__global__ static void moe_gate_up_mid_decode_lut_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp);


__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert);

__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t n_total_expert);

__global__ static void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert);

__global__ static void moe_build_expert_tile_offsets_kernel(
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        const uint32_t *counts,
        uint32_t block_m,
        uint32_t n_total_expert);

__global__ static void moe_build_expert_tiles_kernel(
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const uint32_t *tile_offsets,
        const uint32_t *counts,
        uint32_t block_m,
        uint32_t n_total_expert);


__global__ static void moe_gate_up_mid_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp);

__global__ static void moe_gate_up_mid_expert_tile4_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp);

__global__ static void moe_gate_up_mid_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp);

__global__ static void moe_gate_up_mid_expert_tile8_row2048_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp);

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp);

__global__ static void moe_gate_up_mid_sorted_p2_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t pair_count,
        float clamp);

__global__ static void moe_down_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert);

__global__ static void moe_gate_up_mid_decode_q4K_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp);

__global__ static void moe_gate_up_mid_decode_q4K_hwarp16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp);

__global__ static void moe_gate_up_mid_decode_q4K_hwarp16_row8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp);

__global__ static void moe_gate_up_mid_decode_q4K_warp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp);

__global__ static void moe_gate_up_mid_decode_q4K_warp32_noaux_kernel(
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp);


__global__ static void moe_gate_up_mid_decode_q4K_warp32_noaux_sidecar_kernel(
        float *mid_out,
        float *amax_sidecar,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp);

__global__ static void moe_gate_up_mid_decode_q4K_warp32_row16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp);

__global__ static void moe_gate_up_midq_decode_q4K_qwarp32_kernel(
        float *mid_out,
        cuda_block_q8_K *midq,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp);

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp);

__global__ static void moe_down_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim);






__global__ static void moe_down_sum3_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim);

__global__ static void moe_down_q4K_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim);






__global__ static void moe_down_q4K_sum3_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim);

__global__ static void moe_down_q4K_sum3_slotwarp_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim);

void routed_moe_decode_graph_destroy_one(int logical_tier);

int routed_moe_decode_q4_graph_launch(
        int logical_tier,
        float *out,
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_w,
        const char *up_w,
        const char *down_w,
        cuda_block_q8_K *xq,
        cuda_block_q8_K *midq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp,
        const float *x);

/* Q4_K prefill (n_tokens > 1) down kernel. Mirrors moe_down_qwarp32_kernel
 * geometry exactly; only the weight block type and dot helper differ. The
 * pair = blockIdx.y indexing means the same grid shape (out_dim/32, n_tokens*n_expert)
 * used by the IQ2 path applies here. The downstream moe_sum_kernel is
 * weight-type-agnostic and sums these per-pair outputs into the final output. */
__global__ static void moe_down_q4K_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert);

template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_expert_tile8_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert);

/* INT8 tensor-core (m8n8k16) exact MoE prefill tile kernels.
 *
 * Each warp computes an 8-token x 8-row tile. The Q4_K x Q8_K superblock dot
 * keeps its integer sums (order-invariant, exact) but computes the 32-wide
 * group dots on tensor cores; every output element keeps 8 float slot
 * accumulators (slot[b & 7] += term_b, b ascending) and reduces them with the
 * exact quarter_warp_sum_f32 grouping, so results are bit-identical to the
 * scalar expert-tile kernels (fuzz-verified). Requires sm_75+, 16B-aligned
 * expert tensors, and the staged activation-block counts (<=16 gate/up,
 * <=8 down). Rollback: DS4_CUDA_MOE_NO_Q4_MMA=1. */
__device__ __forceinline__ static void mma_m8n8k16_s8(int32_t &c0, int32_t &c1, uint32_t a, uint32_t b);

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q4K_tile8_mma_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp);

template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_tile8_mma_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert);

/* 16-pair MoE expert tile kernels on sm_80+ m16n8k32 INT8 tensor cores.
 *
 * Same per-output math and reduction order as the 8-pair expert tile
 * kernels (slot[b & 7] += term_b with b ascending, then the exact
 * quarter_warp_sum_f32 grouping), so results are bit-identical; grouping
 * 16 pairs per tile just halves how often each expert's weights are
 * streamed from DRAM. Gate and up run as two passes over the superblocks
 * to keep register pressure at the 8-pair kernel's level. */

__device__ __forceinline__ static void mma16_m16n8k32_s8(
        int32_t &c0, int32_t &c1, int32_t &c2, int32_t &c3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1);

/* One matrix pass over all superblocks for this thread's 4 C elements
 * (tokens mtokA/mtokB x rows n0/n0+1). Returns the quarter-tree-reduced
 * values in r[4] with the exact reference ordering. */
__device__ __forceinline__ static void moe_tile16_mma_pass(
        const char *wrow,            /* row0 base of this matrix */
        uint64_t row_bytes,
        const cuda_block_q8_K (*sxq)[16],
        uint32_t xq_blocks,
        uint32_t lane,
        float r[4]);

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_q4K_tile16_mma_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp);

template <uint32_t ROW_SPAN>
__global__ static void moe_down_q4K_tile16_mma_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert);

int cuda_q4_mma_tile16_shmem_ok(int which_down);

template <uint32_t ROW_SPAN>
__global__ static void moe_down_expert_tile16_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out);







extern "C" int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map,
        uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint64_t down_offset, uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *add_in,
        uint32_t layer_index);

extern "C" int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map,
        uint64_t model_size, uint64_t gate_offset, uint64_t up_offset,
        uint64_t down_offset, uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens,
        bool *mid_is_f16);

extern "C" int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t sinkhorn_iters, float eps);

extern "C" int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc);

extern "C" int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc);

extern "C" int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

extern "C" int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps);

extern "C" int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps);

extern "C" int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc);

extern "C" int ds4_gpu_hc_expand_add_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc);

extern "C" int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc);

extern "C" int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc);

extern "C" int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);



extern "C" int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Compatibility hooks for optional CUDA graph fast paths. Each unavailable
 * operation returns zero so the engine uses its established fallback. */
extern "C" int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map);

extern "C" int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t n_vocab);

extern "C" int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t weight_type,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok);


extern "C" int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor *out_h,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok);

extern "C" int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *q_half,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, float eps);



extern "C" int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor *out_h, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups,
        uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens);

extern "C" int ds4_gpu_attention_output_q4_K_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        ds4_gpu_tensor *group_tmp, ds4_gpu_tensor *low_tmp,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset, uint32_t out_b_type,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups,
        uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens);

extern "C" int ds4_gpu_attention_output_low_q4_K_slice_tensor(
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t group0, uint32_t group_cnt,
        const ds4_gpu_tensor *heads);

extern "C" int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out_h,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc);

extern "C" int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc);

#endif  // DS4X_BACKEND_COMMON_CUH
