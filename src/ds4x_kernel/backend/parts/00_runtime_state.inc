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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_CUDA_UNUSED __attribute__((unused))

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

#include "ds4_gpu_mgpu.h"
#include "ds4_iq2_tables_cuda.inc"

typedef struct {
    ds4_gpu_attention_decode_row row[DS4_GPU_ATTENTION_DECODE_BATCH_MAX];
} cuda_attention_decode_row_table;

static_assert(sizeof(cuda_attention_decode_row_table) <= 3072u,
              "attention row table must fit in CUDA kernel parameters");

static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_registered;
static thread_local bool g_glm_mtp_verify_mode;
static int g_model_device_owned;
static int g_model_range_mapping_supported = 1;
static int g_model_hmm_direct;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static int g_model_direct_fd = -1;
static uint64_t g_model_direct_align = 1;
static uint64_t g_model_file_size;
static int g_model_cache_full;
static cudaStream_t g_model_prefetch_stream;
static cudaStream_t g_model_upload_stream;
static int g_cublas_ready;
static int g_quality_mode;
static int g_decode_fast_attention;
static int g_decode_score_vec4;
static int g_xdev_sync_debug;
static int g_xdev_force_cuda_peer;
static int g_xdev_force_host_bounce;
static int g_cuda_disable_qkv_rms_fused;
static int g_cuda_no_window_attention;
static int g_cuda_decode_heads8_online;
static int g_cuda_decode_score4;
static int g_cuda_decode_score8;
static int g_cuda_no_decode_value512;
static int g_cuda_no_top1;
static int g_cuda_end_stream_sync;
static int g_cuda_no_setdevice_cache;
static int g_cuda_exact_score_split_graph;
static int g_cuda_exact_score_split_ldg;
static int g_cuda_exact_score_split_vec4;
static int g_cuda_exact_score_split_vec4_plain;
static int g_cuda_exact_score_split_dim2;
static int g_cuda_exact_score_split_fuse_inv_rope;
static int g_cuda_moe_decode_graph;
static int g_current_logical_tier = -1;
static int g_ssd_streaming_mode;

typedef struct {
    int valid;
    int logical_tier;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t slot_count;
    uint32_t compact_count;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    char *gate_ptr;
    char *up_ptr;
    char *down_ptr;
    uint64_t gate_capacity;
    uint64_t up_capacity;
    uint64_t down_capacity;
    int32_t *slot_selected_ptr;
    uint64_t slot_selected_capacity;
    ds4_gpu_tensor slot_selected_tensor;
} cuda_stream_selected_cache;

static cuda_stream_selected_cache g_stream_selected_cache;

static void cuda_stream_selected_cache_invalidate(void) {
    g_stream_selected_cache.valid = 0;
}

static void cuda_stream_selected_cache_release(void) {
    const int tier = g_stream_selected_cache.logical_tier;
    if (tier >= 0 && tier < g_n_gpus) {
        (void)ds4_gpu_set_current_device(tier);
    }
    if (g_stream_selected_cache.gate_ptr) {
        (void)cudaFree(g_stream_selected_cache.gate_ptr);
    }
    if (g_stream_selected_cache.up_ptr) {
        (void)cudaFree(g_stream_selected_cache.up_ptr);
    }
    if (g_stream_selected_cache.down_ptr) {
        (void)cudaFree(g_stream_selected_cache.down_ptr);
    }
    if (g_stream_selected_cache.slot_selected_ptr) {
        (void)cudaFree(g_stream_selected_cache.slot_selected_ptr);
    }
    memset(&g_stream_selected_cache, 0, sizeof(g_stream_selected_cache));
    g_stream_selected_cache.logical_tier = -1;
}

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

static cuda_score_split_graph_cache g_score_split_graph[DS4_MAX_GPUS];
static void attention_decode_score_split_graph_destroy_one(int logical_tier);

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

static cuda_moe_decode_graph_cache g_moe_decode_graph[DS4_MAX_GPUS];

static int cuda_q4_mma_ok(void) {
    /* Cached once: all tiers on this host are the same GPU model. */
    static int cached = -1;
    if (cached < 0) {
        if (getenv("DS4_CUDA_MOE_NO_Q4_MMA") != NULL) {
            cached = 0;
        } else {
            int dev = 0, major = 0, minor = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
            cached = (major > 7 || (major == 7 && minor >= 5)) ? 1 : 0;
        }
    }
    return cached;
}





static int cuda_q4_mma_tile16_shmem_ok(int which_down);


static void routed_moe_decode_graph_destroy_one(int logical_tier);

/* =========================================================================
 * Multi-GPU plumbing (device-aware CUDA).
 * ========================================================================= */

static_assert(DS4_MAX_GPUS == 16, "DS4_MAX_GPUS stack tables sized for 16");

ds4_gpu_ctx g_gpu[DS4_MAX_GPUS];
int         g_n_gpus = 0;
int         g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS];

/* Per-pair pinned-host bounce buffers, indexed [src][dst]. Lazily grown
 * to the largest copy seen for that pair. Each pair is its own allocation
 * so concurrent fan-out copies from a single source GPU to multiple
 * destinations cannot race for staging memory. */
static void   *g_xdev_bounce[DS4_MAX_GPUS][DS4_MAX_GPUS];
static size_t  g_xdev_bounce_bytes[DS4_MAX_GPUS][DS4_MAX_GPUS];

/* Internal helper: resolve a tensor's device index. -1 (untagged) is
 * treated as device 0 for legacy callers. */
static inline int ds4_tensor_device_idx(const ds4_gpu_tensor *t) {
    if (!t) return 0;
    int d = t->device_id;
    if (d < 0) return 0;
    return d;
}

/* Debug/override flags are read once per CUDA init. The hot decode path calls
 * the xdev helpers many times per token, so they must not re-enter getenv(). */
static void cuda_xdev_env_refresh(void) {
    g_xdev_sync_debug = getenv("DS4_CUDA_SYNC_XDEV") != NULL;
    g_xdev_force_cuda_peer = getenv("DS4_FORCE_CUDA_PEER") != NULL;
    g_xdev_force_host_bounce = getenv("DS4_FORCE_HOST_BOUNCE") != NULL;
}

static void cuda_decode_dispatch_env_refresh(void) {
    g_cuda_disable_qkv_rms_fused = getenv("DS4_CUDA_DISABLE_QKV_RMS_FUSED") != NULL;
    g_cuda_no_window_attention = getenv("DS4_CUDA_NO_WINDOW_ATTENTION") != NULL;
    g_cuda_decode_heads8_online = getenv("DS4_CUDA_DECODE_HEADS8_ONLINE") != NULL;
    g_cuda_decode_score4 = getenv("DS4_CUDA_DECODE_SCORE4") != NULL;
    g_cuda_decode_score8 = getenv("DS4_CUDA_DECODE_SCORE8") != NULL;
    g_cuda_no_decode_value512 = getenv("DS4_CUDA_NO_DECODE_VALUE512") != NULL;
    g_cuda_no_top1 = getenv("DS4_CUDA_NO_TOP1") != NULL;
    g_cuda_end_stream_sync = getenv("DS4_CUDA_END_STREAM_SYNC") != NULL;
    g_cuda_no_setdevice_cache = getenv("DS4_CUDA_NO_SETDEVICE_CACHE") != NULL;
    g_cuda_exact_score_split_graph =
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_GRAPH") != NULL;
    g_cuda_exact_score_split_ldg =
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_LDG") != NULL;
    g_cuda_exact_score_split_vec4 =
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_VEC4") != NULL;
    g_cuda_exact_score_split_vec4_plain =
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_VEC4_PLAIN") != NULL;
    g_cuda_exact_score_split_dim2 =
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_DIM2") != NULL &&
        getenv("DS4_CUDA_NO_EXACT_SCORE_SPLIT_DIM2") == NULL;
    g_cuda_exact_score_split_fuse_inv_rope =
        getenv("DS4_CUDA_EXACT_SCORE_SPLIT_FUSE_INV_ROPE") != NULL;
    g_cuda_moe_decode_graph = getenv("DS4_CUDA_MOE_DECODE_GRAPH") != NULL;
}

/* WITH_DEVICE(d) { ... } scope macro.
 *
 * Save the calling thread's current CUDA device, switch to device `d`,
 * run the body exactly once, then restore the previous device. If the
 * required CUDA calls fail, the body still runs (we don't have a clean
 * way to early-exit a containing function from a macro), but the next
 * CUDA call inside the body will surface the error naturally.
 *
 * Implementation: a for-loop with two synthetic variables. Iter 0 runs
 * the body; on iter 1, the iteration step restores the previous device
 * via cudaSetDevice and sets _wd_first = 0 so the loop exits. The
 * single-statement-body restriction of for-loops is removed by the
 * required `{ ... }` block in the call site.
 */
#define WITH_DEVICE(d)                                                      \
    for (int _wd_prev = -1, _wd_first = 1;                                  \
         _wd_first;                                                          \
         _wd_first = 0,                                                      \
         (_wd_prev >= 0 ? (void)cudaSetDevice(_wd_prev) : (void)0))         \
        if (cudaGetDevice(&_wd_prev) != cudaSuccess) { /* leave */ } else   \
        if (cudaSetDevice(d)           != cudaSuccess) { /* leave */ } else

/* =========================================================================
 * Per-device selective model cache (selective model cache).
 *
 * The public API in ds4_gpu.h declares ds4_tensor_range and the
 * device_cache_tensors / lookup_cache entry points. ds4_cuda.cu does NOT
 * include ds4_gpu.h historically (a pre-existing project convention), so
 * we redeclare the struct here with the same layout the header uses.
 * The implementation links by C linkage; struct compatibility is by
 * field layout. */
typedef struct {
    uint64_t source_offset;
    uint64_t bytes;
    int      target_device;
} ds4_tensor_range;

struct cuda_device_cache {
    void   *base;     /* device-side slab base */
    size_t  bytes;
    int     present;
};
static cuda_device_cache g_dev_cache[DS4_MAX_GPUS];

struct cache_range_entry {
    uint64_t source_offset;
    uint64_t bytes;
    int      device_id;
    void    *device_ptr;
};
static std::vector<cache_range_entry> g_cache_ranges;

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

struct cuda_q8_f32_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    float *device_ptr;
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

static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
static std::vector<cuda_q8_f32_range> g_q8_f32_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f32_by_offset;
static std::vector<cuda_derived_range> g_derived_ranges;
static const void *g_derived_replace_map;
static uint64_t g_derived_artifact_bytes;
static double g_derived_artifact_build_secs;
static int g_derived_replaces_complete;
static void *g_aligned_q81_scratch;
static uint64_t g_model_range_bytes;
static uint64_t g_q8_f16_bytes;
static uint64_t g_q8_f32_bytes;
static int g_q8_cache_suppressed;
static int g_q8_f16_disabled_after_oom;
static int g_q8_f16_budget_notice_printed;
static uint64_t g_model_load_progress_next;
static double g_model_load_progress_last;
static int g_model_load_progress_started;
static int g_model_load_progress_tty;
static void *g_cuda_tmp;
static uint64_t g_cuda_tmp_bytes;
static void *g_tt_scratch;
static uint64_t g_tt_scratch_bytes;
static int g_tt_scratch_device = -1;
#ifdef DS4_CUDA_HAVE_MXF4
static void *g_indexer_mxf4_scratch;
static uint64_t g_indexer_mxf4_scratch_bytes;
static int g_indexer_mxf4_scratch_device = -1;
#endif
static void *g_model_stage_raw[4];
static void *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_bytes;
static void *g_stream_selected_stage_raw[4];
static void *g_stream_selected_stage[4];
static cudaEvent_t g_stream_selected_stage_event[4];
static uint64_t g_stream_selected_stage_bytes;
static cudaStream_t g_stream_selected_upload_stream;

static int cuda_ok(cudaError_t err, const char *what);
extern "C" void ds4_gpu_decode_graphs_invalidate(void);
static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);

/* Forward declaration: defined later in this file. The resolver wrapper
 * below uses it for multi-tier dispatch. */
extern "C" int ds4_gpu_lookup_cache_strict(uint64_t source_offset,
                                            uint64_t bytes,
                                            int      expected_device,
                                            void   **out_device_ptr);
__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);

static int cuda_aligned_iq2_enabled(void) {
    const char *s = getenv("DS4_CUDA_MOE_NO_IQ2_ALIGNED");
    return !(s && s[0] && strcmp(s, "0") != 0);
}

static int cuda_aligned_q2k_enabled(void) {
    const char *s = getenv("DS4_CUDA_MOE_NO_Q2K_ALIGNED");
    return !(s && s[0] && strcmp(s, "0") != 0);
}

static int cuda_aligned_q8_enabled(void) {
    const char *s = getenv("DS4_CUDA_Q8_NO_ALIGNED");
    return !(s && s[0] && strcmp(s, "0") != 0);
}

static const char *cuda_derived_weight_ptr(
        const void *model_map,
        uint64_t source_offset,
        uint64_t source_bytes,
        uint32_t kind,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t group_count,
        uint64_t bytes) {
    if (getenv("DS4_CUDA_NO_DERIVED_WEIGHTS") != NULL) return NULL;
    for (const cuda_derived_range &r : g_derived_ranges) {
        if (r.host_base == model_map &&
            r.source_offset == source_offset &&
            r.source_bytes == source_bytes &&
            r.kind == kind &&
            r.in_dim == in_dim &&
            r.out_dim == out_dim &&
            r.group_count == group_count &&
            bytes <= r.bytes) {
            return r.device_ptr;
        }
    }
    return NULL;
}

static int cuda_model_map_replaces_complete(const void *model_map) {
    return g_derived_replaces_complete && model_map == g_derived_replace_map;
}

static int cuda_integrated_artifact_map(const void *model_map) {
    if (!cuda_model_map_replaces_complete(model_map)) return 0;
    int device = 0;
    int integrated = 0;
    return cudaGetDevice(&device) == cudaSuccess &&
           cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device) == cudaSuccess &&
           integrated;
}

/* Startup combines adjacent tensors into cache spans.  A span is replaceable
 * when aligned IQ2/Q2K artifacts cover every tensor payload in it; GGUF
 * alignment gaps of at most 64 KiB do not need device residency. */
static int cuda_span_fully_replaced(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes) {
    if (!cuda_model_map_replaces_complete(model_map) || bytes == 0) return 0;
    const uint64_t end = offset + bytes;
    if (end < offset) return 0;
    uint64_t cursor = offset;
    int found = 0;
    while (cursor < end) {
        uint64_t next = cursor;
        for (const cuda_derived_range &r : g_derived_ranges) {
            if (r.host_base != model_map ||
                (r.kind != CUDA_DERIVED_IQ2_XXS_ALIGNED_MOE &&
                 r.kind != CUDA_DERIVED_Q2_K_ALIGNED_MOE)) {
                continue;
            }
            const uint64_t r_end = r.source_offset + r.source_bytes;
            const uint64_t gap = found ? 65536u : 0u;
            if (r_end >= r.source_offset &&
                r.source_offset <= cursor + gap &&
                r_end > next) {
                next = r_end;
            }
        }
        if (next <= cursor) return 0;
        cursor = next;
        found = 1;
    }
    return 1;
}

static void *cuda_tmp_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_cuda_tmp_bytes >= bytes) return g_cuda_tmp;
    if (g_cuda_tmp) {
        if (!cuda_ok(cudaDeviceSynchronize(),
                     "synchronize CUDA scratch growth")) {
            return NULL;
        }
        ds4_gpu_decode_graphs_invalidate();
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA temp alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "scratch", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_cuda_tmp = ptr;
    g_cuda_tmp_bytes = bytes;
    return g_cuda_tmp;
}

static void *tt_scratch_ensure(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_tt_scratch_bytes >= bytes) return g_tt_scratch;

    int device = -1;
    if (!cuda_ok(cudaGetDevice(&device), "get device for token-tile scratch")) {
        return NULL;
    }
    if (g_tt_scratch) {
        if (!cuda_ok(cudaDeviceSynchronize(),
                     "synchronize token-tile scratch growth")) {
            return NULL;
        }
        (void)cudaFree(g_tt_scratch);
        g_tt_scratch = NULL;
        g_tt_scratch_bytes = 0;
        g_tt_scratch_device = -1;
    }
    if (!cuda_ok(cudaMalloc(&g_tt_scratch, (size_t)bytes),
                 what ? what : "allocate token-tile scratch")) {
        g_tt_scratch = NULL;
        return NULL;
    }
    g_tt_scratch_bytes = bytes;
    g_tt_scratch_device = device;
    return g_tt_scratch;
}

static inline uint64_t tt_align256_u64(uint64_t x) {
    return (x + 255ull) & ~255ull;
}

/* Per-tier scratch accessor.
 *
 * Behavior:
 *   - At g_n_gpus <= 1 (single-tier), delegates to cuda_tmp_alloc which
 *     manages the legacy g_cuda_tmp slab. This guarantees byte-identical
 *     behavior to pre-task code for the gpu_cfg == NULL case.
 *   - For multi-tier, grows the per-device g_gpu[logical_tier].scratch
 *     slab on the corresponding physical device. Cleanup is already
 *     handled by ds4_gpu_cleanup (which walks g_gpu[i].scratch).
 *
 * The legacy g_cuda_tmp slab is untouched: init-time / preload callers
 * still use cuda_tmp_alloc directly. No aliasing between g_cuda_tmp
 * and g_gpu[0].scratch — they are independently owned and freed.
 *
 * Added for multi-GPU execution (multi-GPU execution), step A3 of the
 * spec (sub-area 2). */
static void *cuda_tmp_alloc_on(int logical_tier, uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_n_gpus <= 1) {
        return cuda_tmp_alloc(bytes, what);
    }
    if (logical_tier < 0 || logical_tier >= g_n_gpus) {
        fprintf(stderr, "ds4: cuda_tmp_alloc_on: bad tier %d (n_gpus=%d, what=%s)\n",
                logical_tier, g_n_gpus, what ? what : "?");
        return NULL;
    }
    ds4_gpu_ctx *ctx = &g_gpu[logical_tier];
    if (ctx->scratch_bytes >= bytes) return ctx->scratch;
    int prev = -1;
    cudaError_t derr = cudaGetDevice(&prev);
    if (derr != cudaSuccess) {
        fprintf(stderr,
                "ds4: cudaGetDevice failed before scratch alloc on tier %d (dev=%d, what=%s): %s\n",
                logical_tier, ctx->device_id, what ? what : "scratch",
                cudaGetErrorString(derr));
        (void)cudaGetLastError();
        return NULL;
    }
    derr = cudaSetDevice(ctx->device_id);
    if (derr != cudaSuccess) {
        fprintf(stderr,
                "ds4: cudaSetDevice(%d) failed before scratch alloc on tier %d (what=%s): %s\n",
                ctx->device_id, logical_tier, what ? what : "scratch",
                cudaGetErrorString(derr));
        (void)cudaGetLastError();
        if (prev >= 0) (void)cudaSetDevice(prev);
        return NULL;
    }
    if (ctx->scratch) {
        (void)cudaFree(ctx->scratch);
        ctx->scratch = NULL;
        ctx->scratch_bytes = 0;
    }
    void *p = NULL;
    cudaError_t err = cudaMalloc(&p, (size_t)bytes);
    if (prev >= 0) (void)cudaSetDevice(prev);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA scratch alloc on tier %d (dev=%d) failed for %s (%.2f MiB): %s\n",
                logical_tier, ctx->device_id, what ? what : "scratch",
                (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    ctx->scratch = p;
    ctx->scratch_bytes = (size_t)bytes;
    return p;
}

static int cuda_attention_score_buffer_fits(uint32_t n_comp) {
    return n_comp <= DS4_CUDA_ATTENTION_SCORE_CAP - DS4_CUDA_ATTENTION_RAW_SCORE_CAP;
}

static const char *cuda_model_ptr(const void *model_map, uint64_t offset) {
    if (model_map == g_model_host_base && g_model_device_base) return g_model_device_base + offset;
    return (const char *)model_map + offset;
}

static const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    if (bytes == 0) return cuda_model_ptr(model_map, offset);
    if (g_model_device_owned || g_model_registered) return cuda_model_ptr(model_map, offset);
    if (g_model_hmm_direct &&
        getenv("DS4_CUDA_WEIGHT_CACHE") == NULL &&
        getenv("DS4_CUDA_WEIGHT_PRELOAD") == NULL) {
        return cuda_model_ptr(model_map, offset);
    }
    const char *direct_env = getenv("DS4_CUDA_DIRECT_MODEL");
    if (direct_env && direct_env[0]) return cuda_model_ptr(model_map, offset);

    const uint64_t end = offset + bytes;
    auto exact = g_model_range_by_offset.find(offset);
    if (exact != g_model_range_by_offset.end()) {
        const cuda_model_range &r = g_model_ranges[exact->second];
        if (r.host_base == model_map && end >= offset && bytes <= r.bytes) return r.device_ptr;
    }
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map && offset >= r.offset && end >= offset && end <= r.offset + r.bytes) {
            return r.device_ptr + (offset - r.offset);
        }
        if (r.host_base == model_map && r.host_registered && r.registered_base && r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return r.registered_device_base + (h0 - r0);
        }
    }

    if (getenv("DS4_CUDA_NO_FD_CACHE") == NULL) {
        const char *fd_ptr = cuda_model_range_ptr_from_fd(model_map, offset, bytes, what);
        if (fd_ptr) return fd_ptr;
    }

    cudaError_t err = cudaSuccess;
    if (g_model_range_mapping_supported) {
        const long page_sz_l = sysconf(_SC_PAGESIZE);
        const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
        const uintptr_t host_addr = (uintptr_t)((const char *)model_map + offset);
        const uintptr_t reg_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
        const uint64_t reg_delta = (uint64_t)(host_addr - reg_addr);
        const uint64_t reg_bytes = (reg_delta + bytes + page_sz - 1u) & ~(page_sz - 1u);
        void *reg_dev = NULL;
        err = cudaHostRegister((void *)reg_addr,
                               (size_t)reg_bytes,
                               cudaHostRegisterMapped | cudaHostRegisterReadOnly);
        if (err == cudaSuccess) {
            err = cudaHostGetDevicePointer(&reg_dev, (void *)reg_addr, 0);
            if (err == cudaSuccess && reg_dev) {
                char *dev_ptr = (char *)reg_dev + reg_delta;
                g_model_ranges.push_back({model_map, offset, bytes, dev_ptr, (void *)reg_addr, (char *)reg_dev, reg_bytes, 1, 0});
                g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
                if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "ds4: CUDA mapped %s %.2f MiB\n",
                            what ? what : "weights",
                            (double)bytes / 1048576.0);
                }
                return dev_ptr;
            }
            fprintf(stderr, "ds4: CUDA model range map pointer failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaHostUnregister((void *)reg_addr);
            (void)cudaGetLastError();
        } else {
            if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) g_model_range_mapping_supported = 0;
            (void)cudaGetLastError();
        }
    }

    void *dev = NULL;
    err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: CUDA model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        return NULL;
    }

    const char *src = (const char *)model_map + offset;
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t done = 0; done < bytes; done += chunk) {
        uint64_t n = bytes - done < chunk ? bytes - done : chunk;
        err = cudaMemcpy((char *)dev + done, src + done, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f/%.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)done / 1048576.0,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return NULL;
        }
    }
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

/* Per-tier cuBLAS handle. Used by kernel-dispatch wrappers; returns the
 * cuBLAS handle for the logical tier. The wrapper is expected to have
 * cudaSetDevice'd to that tier's physical device already (kernels and
 * cuBLAS calls ride the default stream and are naturally serialized).
 *
 * Added for multi-GPU execution (multi-GPU execution), sub-area 1. */
static inline cublasHandle_t cuda_cublas_for_tier(int logical_tier) {
    if (g_n_gpus <= 1) {
        return (cublasHandle_t)g_gpu[0].cublas;
    }
    /* The executing device is authoritative: GLM per-layer switching runs
     * generic launchers whose out tensors live on device 0 while the layer
     * executes elsewhere. On DS4 paths the current device always equals the
     * requested tier, so this is behavior-preserving there. */
    int cur_dev = -1;
    if (cudaGetDevice(&cur_dev) == cudaSuccess) {
        for (int t = 0; t < g_n_gpus; t++) {
            if (g_gpu[t].device_id == cur_dev) {
                return (cublasHandle_t)g_gpu[t].cublas;
            }
        }
    }
    if (logical_tier < 0 || logical_tier >= g_n_gpus) {
        return (cublasHandle_t)g_gpu[0].cublas;
    }
    return (cublasHandle_t)g_gpu[logical_tier].cublas;
}
