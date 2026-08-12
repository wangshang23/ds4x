#include "../internal/backend_internal.cuh"

/* Runtime implementation. */



#ifndef M_PI
#endif





const void *g_model_host_base;
const char *g_model_device_base;
uint64_t g_model_registered_size;
int g_model_registered;
int g_model_device_owned;
int g_model_range_mapping_supported = 1;
int g_model_hmm_direct;
int g_model_fd = -1;
const void *g_model_fd_host_base;
int g_model_direct_fd = -1;
uint64_t g_model_direct_align = 1;
uint64_t g_model_file_size;
int g_model_cache_full;
static cudaStream_t g_model_prefetch_stream;
static cudaStream_t g_model_upload_stream;
int g_cublas_ready;
int g_quality_mode;
int g_decode_fast_attention;
int g_decode_score_vec4;
int g_cuda_disable_qkv_rms_fused;
int g_cuda_no_window_attention;
int g_cuda_decode_heads8_online;
int g_cuda_decode_score4;
int g_cuda_decode_score8;
int g_cuda_no_decode_value512;
int g_cuda_no_top1;
int g_cuda_end_stream_sync;
int g_cuda_exact_score_split_graph;
int g_cuda_exact_score_split_ldg;
int g_cuda_exact_score_split_vec4;
int g_cuda_exact_score_split_vec4_plain;
int g_cuda_exact_score_split_dim2;
int g_cuda_exact_score_split_fuse_inv_rope;
int g_cuda_moe_decode_graph;

cuda_score_split_graph_cache g_score_split_graph[DS4_MAX_GPUS];

cuda_moe_decode_graph_cache g_moe_decode_graph[DS4_MAX_GPUS];

int cuda_q4_mma_ok(void) {
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

ds4_gpu_ctx g_gpu[DS4_MAX_GPUS];
int         g_n_gpus = 0;

/* Internal helper: resolve a tensor's device index. -1 (untagged) is
 * treated as device 0 for legacy callers. */
int ds4_tensor_device_idx(const ds4_gpu_tensor *t) {
    if (!t) return 0;
    int d = t->device_id;
    if (d < 0) return 0;
    return d;
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

static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
std::vector<cuda_derived_range> g_derived_ranges;
const void *g_derived_replace_map;
uint64_t g_derived_artifact_bytes;
double g_derived_artifact_build_secs;
int g_derived_replaces_complete;
void *g_aligned_q81_scratch;
static uint64_t g_model_range_bytes;
static uint64_t g_q8_f16_bytes;
int g_q8_cache_suppressed;
int g_q8_f16_disabled_after_oom;
int g_q8_f16_budget_notice_printed;
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
void *g_indexer_mxf4_scratch;
uint64_t g_indexer_mxf4_scratch_bytes;
int g_indexer_mxf4_scratch_device = -1;
#endif
static void *g_model_stage_raw[4];
static void *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_bytes;

int cuda_aligned_iq2_enabled(void) {
    const char *s = getenv("DS4_CUDA_MOE_NO_IQ2_ALIGNED");
    return !(s && s[0] && strcmp(s, "0") != 0);
}

int cuda_aligned_q2k_enabled(void) {
    const char *s = getenv("DS4_CUDA_MOE_NO_Q2K_ALIGNED");
    return !(s && s[0] && strcmp(s, "0") != 0);
}

int cuda_aligned_q8_enabled(void) {
    const char *s = getenv("DS4_CUDA_Q8_NO_ALIGNED");
    return !(s && s[0] && strcmp(s, "0") != 0);
}

const char *cuda_derived_weight_ptr(
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

int cuda_model_map_replaces_complete(const void *model_map) {
    return g_derived_replaces_complete && model_map == g_derived_replace_map;
}

int cuda_integrated_artifact_map(const void *model_map) {
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
int cuda_span_fully_replaced(
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

void *tt_scratch_ensure(uint64_t bytes, const char *what) {
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

uint64_t tt_align256_u64(uint64_t x) {
    return (x + 255ull) & ~255ull;
}

void *cuda_tmp_alloc_on(int logical_tier, uint64_t bytes, const char *what) {
    (void)logical_tier;
    return cuda_tmp_alloc(bytes, what);
}

int cuda_attention_score_buffer_fits(uint32_t n_comp) {
    return n_comp <= DS4_CUDA_ATTENTION_SCORE_CAP - DS4_CUDA_ATTENTION_RAW_SCORE_CAP;
}

static const char *cuda_model_ptr(const void *model_map, uint64_t offset) {
    if (model_map == g_model_host_base && g_model_device_base) return g_model_device_base + offset;
    return (const char *)model_map + offset;
}

const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
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

cublasHandle_t cuda_cublas_for_tier(int logical_tier) {
    (void)logical_tier;
    return (cublasHandle_t)g_gpu[0].cublas;
}

/* ------------------------------------------------------------------------
 * Decode-island CUDA graph capture.
 *
 * Design ported from the Entrpi/ds4 batched-serving fork's per-layer
 * decode graph work (fork commits abe26575, 2200c675, 9caaa508; author
 * Entrpi <entrpi@proton.me>), re-implemented against upstream's decode
 * body.  Phase 1 captures the two position-independent islands of each
 * decode layer:
 *   island 0: layer top (hc pre-norm, mixes, QKV projections), ending
 *             before the position-dependent QKV rope.
 *   island 1: attention output projection through the FFN/MoE tail.
 * Kernel launches inside the islands ride cuda_decode_stream(): the
 * legacy NULL stream in eager mode (bit-identical behavior to before),
 * or a dedicated blocking stream while capturing/replaying.  The legacy
 * stream cannot be captured; a blocking (default-flags) stream keeps the
 * implicit serialization against NULL-stream work, so the eager middle
 * section of each layer orders correctly around replays.
 *
 * Per-entry state machine: first sight of a key runs eager (warms lazy
 * allocators -- cudaMalloc is forbidden inside capture), second sight
 * captures + instantiates + launches, later sights replay.  Any failure
 * marks the entry dead and the caller re-encodes the island eagerly, so
 * a broken capture can never lose a token's work.  Graphs replay the
 * captured kernels byte-for-byte, so replayed output is bit-identical
 * to the eager encode of the same island.
 *
 * DS4_CUDA_DECODE_GRAPHS=0 (or off/no/false) disables everything. */


static cuda_decode_graph_entry
    g_decode_graphs[CUDA_DECODE_GRAPH_LAYERS][CUDA_DECODE_GRAPH_ISLANDS]
                   [CUDA_DECODE_GRAPH_VARIANTS];
static cudaStream_t g_decode_graph_stream = NULL;
static int g_decode_graph_capturing = 0;
static uint64_t g_decode_graph_replays = 0;
static uint64_t g_decode_graph_captures = 0;

extern "C" int ds4_gpu_decode_graphs_supported(void) {
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        init = 1;
        const char *s = getenv("DS4_CUDA_DECODE_GRAPHS");
        const int off = s && *s &&
            (s[0] == '0' ||
             strcmp(s, "off") == 0 || strcmp(s, "OFF") == 0 ||
             strcmp(s, "no") == 0 || strcmp(s, "NO") == 0 ||
             strcmp(s, "false") == 0 || strcmp(s, "FALSE") == 0);
        if (off) {
            fprintf(stderr, "ds4: DS4_CUDA_DECODE_GRAPHS=%s - decode graph capture disabled\n", s);
            enabled = 0;
        } else {
            enabled = 1;
        }
    }
    return enabled && g_n_gpus == 1;
}

/* Stream the decode-island kernels launch on.  Legacy NULL stream in
 * eager mode (unchanged behavior); the capture stream while a capture
 * or replay is in flight. */
cudaStream_t cuda_decode_stream(void) {
    return g_decode_graph_capturing ? g_decode_graph_stream : (cudaStream_t)0;
}

static void cuda_decode_graph_entry_kill(cuda_decode_graph_entry *e) {
    if (e->exec) {
        (void)cudaGraphExecDestroy(e->exec);
        e->exec = NULL;
    }
    e->state = 3;
}

extern "C" void ds4_gpu_decode_graphs_invalidate(void) {
    for (uint32_t il = 0; il < CUDA_DECODE_GRAPH_LAYERS; il++) {
        for (uint32_t is = 0; is < CUDA_DECODE_GRAPH_ISLANDS; is++) {
            for (uint32_t v = 0; v < CUDA_DECODE_GRAPH_VARIANTS; v++) {
                cuda_decode_graph_entry *e = &g_decode_graphs[il][is][v];
                if (e->exec) {
                    (void)cudaGraphExecDestroy(e->exec);
                    e->exec = NULL;
                }
                e->state = 0;
                e->hits = 0;
                memset(&e->key, 0, sizeof(e->key));
            }
        }
    }
}

static cuda_decode_graph_entry *cuda_decode_graph_find(
        const ds4_decode_graph_key *key) {
    if (key->il >= CUDA_DECODE_GRAPH_LAYERS ||
        key->island >= CUDA_DECODE_GRAPH_ISLANDS) return NULL;
    cuda_decode_graph_entry *slot = NULL;
    for (uint32_t v = 0; v < CUDA_DECODE_GRAPH_VARIANTS; v++) {
        cuda_decode_graph_entry *e = &g_decode_graphs[key->il][key->island][v];
        if (e->state != 0 &&
            memcmp(&e->key, key, sizeof(*key)) == 0) return e;
        if (e->state == 0 && !slot) slot = e;
    }
    if (slot) {
        memcpy(&slot->key, key, sizeof(*key));
        slot->state = 0;   /* caller advances the state machine */
        return slot;
    }
    return NULL;           /* all variants busy with other keys: stay eager */
}

extern "C" int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key) {
    if (!key || !ds4_gpu_decode_graphs_supported()) return -1;
    if (g_decode_graph_capturing) return -1;   /* no nesting */
    cuda_decode_graph_entry *e = cuda_decode_graph_find(key);
    if (!e || e->state == 3) return -1;
    if (e->state == 0) {
        /* Warm pass: run eagerly once so lazy allocators (tmp scratch,
         * cuBLAS workspaces) reach steady-state sizes before capture. */
        e->state = 1;
        return -1;
    }
    if (e->state == 2) {
        cudaError_t err = cudaGraphLaunch(e->exec, g_decode_graph_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: decode graph replay failed (il=%u island=%u): %s\n",
                    key->il, key->island, cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_decode_graph_entry_kill(e);
            return -1;     /* caller encodes eagerly; nothing was consumed */
        }
        e->hits++;
        g_decode_graph_replays++;
        return 1;
    }
    /* state == 1: capture this encode. */
    if (!g_decode_graph_stream) {
        if (!cuda_ok(cudaStreamCreate(&g_decode_graph_stream),
                     "decode graph stream create")) {
            g_decode_graph_stream = NULL;
            cuda_decode_graph_entry_kill(e);
            return -1;
        }
    }
    /* cuBLAS rides the legacy NULL stream, which cannot be captured:
     * point the handle at the capture stream for the duration. */
    (void)cublasSetStream(cuda_cublas_for_tier(0), g_decode_graph_stream);
    if (!cuda_ok(cudaStreamBeginCapture(g_decode_graph_stream,
                                        cudaStreamCaptureModeGlobal),
                 "decode graph begin capture")) {
        (void)cublasSetStream(cuda_cublas_for_tier(0), NULL);
        cuda_decode_graph_entry_kill(e);
        return -1;
    }
    g_decode_graph_capturing = 1;
    return 0;
}

extern "C" int ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key) {
    if (!key || !g_decode_graph_capturing) return -1;
    g_decode_graph_capturing = 0;
    cudaGraph_t graph = NULL;
    cudaError_t err = cudaStreamEndCapture(g_decode_graph_stream, &graph);
    (void)cublasSetStream(cuda_cublas_for_tier(0), NULL);
    cuda_decode_graph_entry *e = cuda_decode_graph_find(key);
    if (err != cudaSuccess || graph == NULL) {
        fprintf(stderr, "ds4: decode graph capture failed (il=%u island=%u): %s\n",
                key->il, key->island, cudaGetErrorString(err));
        (void)cudaGetLastError();
        if (graph) (void)cudaGraphDestroy(graph);
        if (e) cuda_decode_graph_entry_kill(e);
        return -1;         /* caller re-encodes the island eagerly */
    }
    if (!e) {              /* cannot happen: begin() found it */
        (void)cudaGraphDestroy(graph);
        return -1;
    }
    cudaGraphExec_t exec = NULL;
    err = cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
    (void)cudaGraphDestroy(graph);
    if (err != cudaSuccess || exec == NULL) {
        fprintf(stderr, "ds4: decode graph instantiate failed (il=%u island=%u): %s\n",
                key->il, key->island, cudaGetErrorString(err));
        (void)cudaGetLastError();
        cuda_decode_graph_entry_kill(e);
        return -1;
    }
    /* Capture recorded the work without executing it: launch now so this
     * token's island actually runs. */
    err = cudaGraphLaunch(exec, g_decode_graph_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: decode graph first launch failed (il=%u island=%u): %s\n",
                key->il, key->island, cudaGetErrorString(err));
        (void)cudaGetLastError();
        (void)cudaGraphExecDestroy(exec);
        cuda_decode_graph_entry_kill(e);
        return -1;
    }
    e->exec = exec;
    e->state = 2;
    g_decode_graph_captures++;
    if (getenv("DS4_CUDA_DECODE_GRAPH_LOG") != NULL) {
        fprintf(stderr, "ds4: decode graph captured il=%u island=%u (total %llu)\n",
                key->il, key->island,
                (unsigned long long)g_decode_graph_captures);
    }
    return 0;
}

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
int cuda_use_mmq(void) {
    static int init = 0;
    static int use = 0;
    if (!init) {
        init = 1;
        const char *s = getenv("DS4_CUDA_MMQ");
        const int off = (s && s[0] == '0') || g_quality_mode;
        if (off) {
            if (s && s[0] == '0') {
                fprintf(stderr, "ds4: DS4_CUDA_MMQ=0 - mmq prefill tier disabled\n");
            }
        } else if (ds4_mmq_init(0) == 0) {
            use = 1;
        } else {
            fprintf(stderr, "ds4: ds4_mmq_init failed - mmq prefill tier disabled\n");
        }
    }
    return use;
}

int cuda_use_mxfp4_mmq(void) {
    static int init = 0;
    static int use = 0;
    if (!init) {
        init = 1;
        const char *s = getenv("DS4_CUDA_MMQ");
        if (s && s[0] == '0') {
            fprintf(stderr,
                    "ds4: DS4_CUDA_MMQ=0 - MXFP4 MMQ disabled\n");
        } else {
            int device = 0;
            if (cudaGetDevice(&device) == cudaSuccess &&
                ds4_mmq_init(device) == 0) {
                use = 1;
            } else {
                fprintf(stderr,
                        "ds4: ds4_mmq_init failed - MXFP4 unavailable\n");
            }
        }
    }
    return use;
}

/* Abort an in-flight capture after an encode error inside the island.
 * Nothing was executed; the caller re-encodes eagerly. */
extern "C" void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key) {
    if (!g_decode_graph_capturing) return;
    g_decode_graph_capturing = 0;
    cudaGraph_t graph = NULL;
    (void)cudaStreamEndCapture(g_decode_graph_stream, &graph);
    if (graph) (void)cudaGraphDestroy(graph);
    (void)cublasSetStream(cuda_cublas_for_tier(0), NULL);
    (void)cudaGetLastError();
    if (key) {
        cuda_decode_graph_entry *e = cuda_decode_graph_find(key);
        if (e) cuda_decode_graph_entry_kill(e);
    }
}

/* Resolve a model slice through the single-GB10 range cache. */
const char *cuda_resolve_weight_ptr(const void *model_map,
                                            uint64_t offset,
                                            uint64_t bytes,
                                            int logical_tier,
                                            const char *label) {
    (void)logical_tier;
    return cuda_model_range_ptr(model_map, offset, bytes, label);
}

int cuda_model_range_is_cached(const void *model_map, uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return 1;
    if (g_model_device_owned || g_model_registered) return 1;

    const uint64_t end = offset + bytes;
    if (end < offset) return 0;
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map &&
            offset >= r.offset &&
            end <= r.offset + r.bytes) {
            return 1;
        }
        if (r.host_base == model_map &&
            r.host_registered &&
            r.registered_base &&
            r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return 1;
        }
    }
    return 0;
}

void cuda_q8_f16_cache_release_all(void) {
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f16_ranges.clear();
    g_q8_f16_by_offset.clear();
    g_q8_f16_bytes = 0;
}

static uint64_t cuda_parse_mib_env(const char *name, int *present) {
    const char *env = getenv(name);
    if (present) *present = 0;
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    if (present) *present = 1;
    if (v > UINT64_MAX / 1048576ull) return UINT64_MAX;
    return (uint64_t)v * 1048576ull;
}

uint32_t cuda_parse_u32_env_clamped(const char *name, uint32_t fallback,
                                           uint32_t min_value, uint32_t max_value,
                                           int *present) {
    const char *env = getenv(name);
    if (present) *present = 0;
    if (!env || !env[0]) return fallback;
    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (errno != 0 || end == env || *end != '\0') return fallback;
    if (present) *present = 1;
    if (v < min_value) return min_value;
    if (v > max_value) return max_value;
    return (uint32_t)v;
}

int cuda_env_flag_enabled(const char *name, int fallback) {
    const char *env = getenv(name);
    if (!env || !env[0]) return fallback;
    return strcmp(env, "0") != 0;
}

bool cuda_splitkv_decode_requested(void) {
    if (cuda_env_flag_enabled("DS4_CUDA_NO_SPLITKV_DECODE", 0)) return false;
    return g_decode_fast_attention ||
           cuda_env_flag_enabled("DS4_CUDA_SPLITKV_DECODE", 0);
}

static uint64_t cuda_q8_f16_cache_limit_bytes(void) {
    int present = 0;
    const uint64_t limit = cuda_parse_mib_env("DS4_CUDA_Q8_F16_CACHE_MB", &present);
    return present ? limit : UINT64_MAX;
}

static uint64_t cuda_q8_f16_cache_reserve_bytes(uint64_t total_bytes) {
    int present = 0;
    const uint64_t reserve = cuda_parse_mib_env("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB", &present);
    if (present) return reserve;

    if (total_bytes >= 112ull * 1024ull * 1024ull * 1024ull) {
        return 512ull * 1048576ull;
    }

    /* High-VRAM cards (>= 40 GiB, e.g. 48 GiB RTX 6000 Ada): use a small
     * reserve so the selective Q8->F16 cache can actually engage at tight
     * budgets (e.g. --gpu-vram 47,47, where the 81 GB model leaves only ~1.3
     * GiB free and the old 4 GiB floor rejected every cache allocation,
     * forcing the scalar DP4A prefill kernel).
     *
     * NOTE: this 768 MiB value is a *bounded cache-growth guard*, not a hard
     * guarantee that live free VRAM stays >= 768 MiB.  cuda_q8_f16_cache_has_budget
     * only blocks a *cache* allocation when free - request < reserve at that
     * moment; allocations made outside cache accounting (cuda_tmp_alloc_on
     * activation/prequant buffers, cuBLAS internal workspaces) can still dip
     * below it.  768 MiB is chosen to leave headroom above the ~0.5 GiB
     * memory-safety floor for those out-of-cache allocations; actual minimum
     * free VRAM is verified by measurement, and the disable-after-failure path
     * degrades gracefully if cuBLAS/alloc ever fails under pressure.  Set
     * DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=4096 to restore the prior behavior. */
    if (total_bytes >= 40ull * 1024ull * 1024ull * 1024ull) {
        const uint64_t hi_min_reserve = 768ull * 1048576ull;
        const uint64_t hi_pct_reserve = total_bytes / 100u; /* 1% */
        return hi_pct_reserve > hi_min_reserve ? hi_pct_reserve : hi_min_reserve;
    }

    /* Smaller cards (< 40 GiB): keep the conservative reserve.  The expanded
     * Q8->F16 cache is only an acceleration path; on a small card a sub-GiB
     * reserve would be a large fraction of total VRAM, so keep enough free for
     * cuBLAS workspaces, transient graph buffers, and driver bookkeeping. */
    const uint64_t min_reserve = 4096ull * 1048576ull;
    const uint64_t pct_reserve = total_bytes / 20u; /* 5% */
    return pct_reserve > min_reserve ? pct_reserve : min_reserve;
}

static void cuda_q8_f16_cache_budget_notice(
        const char *reason,
        uint64_t request_bytes,
        uint64_t free_bytes,
        uint64_t total_bytes,
        uint64_t reserve_bytes,
        uint64_t limit_bytes) {
    if (g_q8_f16_budget_notice_printed && getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") == NULL) return;
    g_q8_f16_budget_notice_printed = 1;
    if (limit_bytes != UINT64_MAX && free_bytes == 0 && total_bytes == 0 && reserve_bytes == 0) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0);
    } else if (limit_bytes == UINT64_MAX) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    }
}

static int cuda_q8_f16_cache_has_budget(uint64_t request_bytes, const char *label) {
    (void)label;
    const uint64_t limit = cuda_q8_f16_cache_limit_bytes();
    if (limit == 0) return 0;
    if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
        cuda_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit);
        return 0;
    }

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp16 cache memory query failed: %s; using q8 kernels\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_q8_f16_cache_reserve_bytes(total_bytes);
    if (request_bytes > free_bytes ||
        free_bytes - request_bytes < reserve_bytes) {
        cuda_q8_f16_cache_budget_notice("budget exhausted", request_bytes,
                                        free_bytes, total_bytes,
                                        reserve_bytes, limit);
        return 0;
    }
    return 1;
}

void cuda_q8_f16_cache_disable_after_failure(const char *what, uint64_t request_bytes) {
    if (!g_q8_f16_disabled_after_oom) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache disabled after %s "
                "(request=%.2f MiB cached=%.2f GiB); using q8 kernels\n",
                what ? what : "allocation failure",
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    g_q8_f16_disabled_after_oom = 1;
    if (!g_q8_f16_ranges.empty()) {
        (void)cudaDeviceSynchronize();
        cuda_q8_f16_cache_release_all();
    }
    (void)cudaGetLastError();
}

static int cuda_q8_f16_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (g_quality_mode) return 0;
    if (g_q8_cache_suppressed) return 0;
    if (g_q8_f16_disabled_after_oom) return 0;
    if (getenv("DS4_CUDA_NO_Q8_F16_CACHE") != NULL) return 0;
    if (cuda_q8_f16_cache_limit_bytes() == 0) return 0;
    if (getenv("DS4_CUDA_Q8_F16_ALL") != NULL) return 1;
    if (!label) return 0;
    if (strstr(label, "attn_output_a") != NULL ||
        strstr(label, "attn_output_b") != NULL ||
        strstr(label, "attention_output_a") != NULL ||
        strstr(label, "attention_output_b") != NULL) {
        return getenv("DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE") == NULL;
    }
    if (strstr(label, "attn_q_b") != NULL) {
        return getenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE") == NULL;
    }
    if (strstr(label, "ffn_gate_shexp") != NULL ||
        strstr(label, "ffn_up_shexp") != NULL ||
        strstr(label, "ffn_down_shexp") != NULL) {
        return 1;
    }
    return (in_dim == 4096u && out_dim == 2048u) ||
           (in_dim == 2048u && out_dim == 4096u) ||
           (in_dim == 4096u && out_dim == 1024u) ||
           (in_dim == 4096u && out_dim == 512u) ||
           (getenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE") == NULL &&
            in_dim == 1024u && out_dim == 32768u);
}

static int cuda_q8_label_is_attention_output(const char *label) {
    return label &&
           (strstr(label, "attn_output_a") != NULL ||
            strstr(label, "attn_output_b") != NULL ||
            strstr(label, "attention_output_a") != NULL ||
            strstr(label, "attention_output_b") != NULL);
}

int cuda_q8_use_dp4a(void) {
    return getenv("DS4_CUDA_NO_Q8_DP4A") == NULL;
}

unsigned cuda_q8_exact_threads(uint64_t blocks) {
    if (blocks <= 64u) return 64u;
    if (blocks <= 128u) return 128u;
    return 256u;
}

int cuda_q8_f16_preload_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (cuda_q8_label_is_attention_output(label) &&
        getenv("DS4_CUDA_ATTENTION_OUTPUT_PRELOAD") == NULL &&
        getenv("DS4_CUDA_Q8_F16_ALL") == NULL) {
        return 0;
    }
    return cuda_q8_f16_cache_allowed(label, in_dim, out_dim);
}

/* Look up or create a dequantized FP16 slice of a Q8_0 weight. */
const __half *cuda_q8_f16_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        int expected_device,
        const char *label) {
    auto exact = g_q8_f16_by_offset.find(offset);
    if (exact != g_q8_f16_by_offset.end()) {
        const cuda_q8_f16_range &r = g_q8_f16_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;

    if (in_dim != 0 && out_dim > UINT64_MAX / in_dim / sizeof(__half)) return NULL;
    const uint64_t out_bytes = in_dim * out_dim * sizeof(__half);
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) return NULL;

    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp16 cache alloc failed on device %d (%.2f MiB): %s\n",
                expected_device, (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp16 dequant launch")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("dequant launch failure", out_bytes);
        return NULL;
    }
    g_q8_f16_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev, expected_device});
    g_q8_f16_by_offset[offset] = g_q8_f16_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached q8 fp16 %.2f MiB on device %d (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0, expected_device,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    return dev;
}

int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "ds4: CUDA %s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

double cuda_wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int cuda_model_load_progress_enabled(void) {
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") != NULL) return 0;
    return 1;
}

static void cuda_model_load_progress_reset(void) {
    g_model_load_progress_next = 0;
    g_model_load_progress_last = 0.0;
    g_model_load_progress_started = 0;
    g_model_load_progress_tty = 0;
}

static void cuda_model_load_progress_note(uint64_t cached_bytes) {
    if (!cuda_model_load_progress_enabled()) return;

    const double now = cuda_wall_sec();
    if (!g_model_load_progress_started) {
        g_model_load_progress_started = 1;
        g_model_load_progress_tty = isatty(STDERR_FILENO) != 0;
        g_model_load_progress_next = (g_model_load_progress_tty ? 2ull : 16ull) *
                                     1024ull * 1024ull * 1024ull;
        g_model_load_progress_last = now;
        if (g_model_load_progress_tty) {
            fprintf(stderr, "ds4: CUDA loading model tensors into device cache: 0.00 GiB");
        } else {
            fprintf(stderr, "ds4: CUDA loading model tensors into device cache\n");
        }
    }

    if (cached_bytes < g_model_load_progress_next &&
        now - g_model_load_progress_last < (g_model_load_progress_tty ? 2.0 : 10.0)) {
        return;
    }

    if (g_model_load_progress_tty) {
        fprintf(stderr, "\rds4: CUDA loading model tensors into device cache: %.2f GiB",
                (double)cached_bytes / 1073741824.0);
    } else {
        fprintf(stderr, "ds4: CUDA loading model tensors %.2f GiB cached\n",
                (double)cached_bytes / 1073741824.0);
    }
    fflush(stderr);
    g_model_load_progress_last = now;
    const uint64_t step = (g_model_load_progress_tty ? 2ull : 16ull) *
                          1024ull * 1024ull * 1024ull;
    while (g_model_load_progress_next <= cached_bytes) {
        g_model_load_progress_next += step;
    }
}

int cuda_model_prefetch_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || map_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (getenv("DS4_CUDA_NO_MODEL_PREFETCH") != NULL ||
        getenv("DS4_CUDA_COPY_MODEL") != NULL ||
        getenv("DS4_CUDA_WEIGHT_CACHE") != NULL ||
        getenv("DS4_CUDA_WEIGHT_PRELOAD") != NULL) {
        return 0;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    int pageable = 0;
    cudaError_t err = cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device);
    if (err != cudaSuccess || !pageable) {
        (void)cudaGetLastError();
        return 0;
    }
#if CUDART_VERSION >= 13000
    cudaMemLocation loc;
    memset(&loc, 0, sizeof(loc));
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;
#else
    int loc = device;
#endif

    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t host_addr = (uintptr_t)((const char *)model_map + map_offset);
    const uintptr_t pre_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
    const uint64_t pre_delta = (uint64_t)(host_addr - pre_addr);
    const uint64_t pre_bytes = (pre_delta + map_size + page_sz - 1u) & ~(page_sz - 1u);
    void *pre_ptr = (void *)pre_addr;

    const double t0 = cuda_wall_sec();
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetReadMostly, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model read-mostly advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetPreferredLocation, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model preferred-location advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    if (!g_model_prefetch_stream) {
        err = cudaStreamCreateWithFlags(&g_model_prefetch_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch stream creation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }

#if CUDART_VERSION >= 13000
    err = cudaMemPrefetchAsync(pre_ptr, (size_t)pre_bytes, loc, 0, g_model_prefetch_stream);
#else
    err = cudaMemPrefetchAsync(pre_ptr, (size_t)pre_bytes, loc, g_model_prefetch_stream);
#endif
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model prefetch skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    if (getenv("DS4_CUDA_MODEL_PREFETCH_SYNC") != NULL) {
        err = cudaStreamSynchronize(g_model_prefetch_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch sync failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "ds4: CUDA ATS/HMM prefetch queued %.2f GiB of model tensors in %.3fs\n",
            (double)map_size / 1073741824.0,
            t1 - t0);
    g_model_hmm_direct = 1;
    return 1;
}

static uint64_t cuda_model_copy_chunk_bytes(void) {
    uint64_t mb = 64;
    const char *env = getenv("DS4_CUDA_MODEL_COPY_CHUNK_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 16) mb = 16;
    if (mb > 4096) mb = 4096;
    return mb * 1048576ull;
}

static void cuda_model_discard_source_pages(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes) {
#if defined(POSIX_MADV_DONTNEED)
    if (getenv("DS4_CUDA_KEEP_MODEL_PAGES") != NULL || !model_map || bytes == 0 || offset > model_size) return;
    if (bytes > model_size - offset) bytes = model_size - offset;
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t h1 = h0 + bytes;
    const uintptr_t p0 = h0 & ~(uintptr_t)(page_sz - 1u);
    const uintptr_t p1 = (h1 + page_sz - 1u) & ~(uintptr_t)(page_sz - 1u);
    if (p1 > p0) (void)posix_madvise((void *)p0, (size_t)(p1 - p0), POSIX_MADV_DONTNEED);
#else
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
#endif
}

static void cuda_model_drop_file_pages(uint64_t offset, uint64_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    if (g_model_fd < 0 || getenv("DS4_CUDA_KEEP_MODEL_PAGES") != NULL || bytes == 0) return;
    (void)posix_fadvise(g_model_fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
    (void)offset;
    (void)bytes;
#endif
}

static uint64_t cuda_round_down(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    return (v / align) * align;
}

static uint64_t cuda_round_up(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    const uint64_t rem = v % align;
    return rem == 0 ? v : v + (align - rem);
}

static void *cuda_align_ptr(void *ptr, uint64_t align) {
    if (align <= 1) return ptr;
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t a = (uintptr_t)align;
    return (void *)(((p + a - 1u) / a) * a);
}

static int cuda_model_stage_pool_alloc(uint64_t bytes) {
    if (g_model_stage_bytes >= bytes) return 1;
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (!g_model_upload_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_model_upload_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model upload stream creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_model_stage_raw[i], (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_model_stage[i] = cuda_align_ptr(g_model_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_model_stage_event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging event creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    g_model_stage_bytes = bytes;
    return 1;
}

static int cuda_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n_req = (bytes - done > (uint64_t)SSIZE_MAX) ? (size_t)SSIZE_MAX : (size_t)(bytes - done);
        ssize_t n = pread(fd, (char *)buf + done, n_req, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static int cuda_model_stage_read(void *stage, uint64_t stage_bytes,
                                 uint64_t offset, uint64_t bytes,
                                 const char **payload) {
    *payload = (const char *)stage;
#if defined(__linux__) && defined(O_DIRECT)
    if (g_model_direct_fd >= 0 && g_model_direct_align > 1 && g_model_file_size != 0) {
        const uint64_t aligned_off = cuda_round_down(offset, g_model_direct_align);
        const uint64_t delta = offset - aligned_off;
        uint64_t read_size = cuda_round_up(delta + bytes, g_model_direct_align);
        if (aligned_off <= g_model_file_size &&
            read_size <= stage_bytes &&
            read_size <= g_model_file_size - aligned_off) {
            const int saved_errno = errno;
            errno = 0;
            if (cuda_pread_full(g_model_direct_fd, stage, read_size, aligned_off)) {
                *payload = (const char *)stage + delta;
                errno = saved_errno;
                return 1;
            }
            const int direct_errno = errno;
            if (direct_errno == EINVAL || direct_errno == EFAULT || direct_errno == ENOTSUP || direct_errno == EOPNOTSUPP) {
                if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "ds4: CUDA direct model read disabled: %s\n", strerror(direct_errno));
                }
                (void)close(g_model_direct_fd);
                g_model_direct_fd = -1;
                g_model_direct_align = 1;
            }
            errno = direct_errno;
        }
    }
#else
    (void)stage_bytes;
#endif
    return cuda_pread_full(g_model_fd, stage, bytes, offset);
}


static uint64_t cuda_model_cache_limit_bytes(void) {
    uint64_t gb = 0;
    const char *env = getenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) gb = (uint64_t)v;
    }
    if (gb == 0) return UINT64_MAX;
    return gb * 1073741824ull;
}

static uint64_t cuda_model_arena_chunk_bytes(uint64_t need) {
    uint64_t mb = 1792;
    const char *env = getenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 256) mb = 256;
    if (mb > 8192) mb = 8192;
    uint64_t bytes = mb * 1048576ull;
    if (bytes < need) {
        const uint64_t align = 256ull * 1048576ull;
        bytes = (need + align - 1u) & ~(align - 1u);
    }
    return bytes;
}

static char *cuda_model_arena_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_model_cache_full) return NULL;
    const uint64_t align = 256u;
    const uint64_t aligned = (bytes + align - 1u) & ~(align - 1u);

    for (cuda_model_arena &a : g_model_arenas) {
        const uint64_t used = (a.used + align - 1u) & ~(align - 1u);
        if (used <= a.bytes && aligned <= a.bytes - used) {
            char *ptr = a.device_ptr + used;
            a.used = used + aligned;
            return ptr;
        }
    }

    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || aligned > limit - g_model_range_bytes) return NULL;

    const uint64_t chunk = cuda_model_arena_chunk_bytes(aligned);
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model arena alloc failed for %s (%.2f MiB chunk): %s\n",
                what ? what : "weights",
                (double)chunk / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_model_cache_full = 1;
        return NULL;
    }
    g_model_arenas.push_back({(char *)dev, chunk, aligned});
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        uint64_t arena_bytes = 0;
        for (const cuda_model_arena &a : g_model_arenas) arena_bytes += a.bytes;
        fprintf(stderr, "ds4: CUDA model arena allocated %.2f MiB (arenas %.2f GiB)\n",
                (double)chunk / 1048576.0,
                (double)arena_bytes / 1073741824.0);
    }
    return (char *)dev;
}

const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (g_model_fd < 0 || bytes == 0) return NULL;
    if (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base) return NULL;
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
            fprintf(stderr, "ds4: CUDA direct %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    (double)limit / 1073741824.0);
        }
        return cuda_model_ptr(model_map, offset);
    }

    char *dev = cuda_model_arena_alloc(bytes, what);
    if (!dev) {
        if (getenv("DS4_CUDA_STRICT_WEIGHT_CACHE") != NULL) return NULL;
        return cuda_model_ptr(model_map, offset);
    }
    cudaError_t err = cudaSuccess;

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) return NULL;

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA model staging wait failed for %s: %s\n",
                        what ? what : "weights", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return NULL;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr, "ds4: CUDA model range read failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return NULL;
        }
        err = cudaMemcpyAsync(dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging record failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, g_model_registered_size, offset + copied, n);
        copied += n;
        cuda_model_load_progress_note(g_model_range_bytes + copied);
        chunk_idx++;
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model range upload sync failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    g_model_ranges.push_back({model_map, offset, bytes, dev, NULL, NULL, 0, 0, 1});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    cuda_model_load_progress_note(g_model_range_bytes);
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA fd-cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

int cuda_model_copy_chunked(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || model_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (getenv("DS4_CUDA_NO_MODEL_COPY") != NULL ||
        getenv("DS4_CUDA_DIRECT_MODEL") != NULL ||
        getenv("DS4_CUDA_WEIGHT_CACHE") != NULL ||
        getenv("DS4_CUDA_WEIGHT_PRELOAD") != NULL) {
        return 0;
    }
    if (g_model_device_owned || g_model_registered) return 1;

    void *dev = NULL;
    const double t0 = cuda_wall_sec();
    cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    fprintf(stderr, "ds4: CUDA chunk-copying %.2f GiB model image\n",
            (double)model_size / 1073741824.0);

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    void *stage = NULL;
    err = cudaMallocHost(&stage, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return 0;
    }

    if (map_offset > 0) {
        uint64_t copied_header = 0;
        while (copied_header < map_offset) {
            const uint64_t n = (map_offset - copied_header < chunk) ? (map_offset - copied_header) : chunk;
            memcpy(stage, (const char *)model_map + copied_header, (size_t)n);
            err = cudaMemcpy((char *)dev + copied_header, stage, (size_t)n, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA model header copy failed: %s\n", cudaGetErrorString(err));
                (void)cudaFreeHost(stage);
                (void)cudaFree(dev);
                (void)cudaGetLastError();
                return 0;
            }
            copied_header += n;
        }
    }

    uint64_t copied = 0;
    double last_report = t0;
    while (copied < map_size) {
        const uint64_t n = (map_size - copied < chunk) ? (map_size - copied) : chunk;
        const uint64_t off = map_offset + copied;
        memcpy(stage, (const char *)model_map + off, (size_t)n);
        err = cudaMemcpy((char *)dev + off, stage, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model chunk copy failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, cudaGetErrorString(err));
            (void)cudaFreeHost(stage);
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_discard_source_pages(model_map, model_size, off, n);
        copied += n;
        const double now = cuda_wall_sec();
        if (getenv("DS4_CUDA_MODEL_COPY_VERBOSE") != NULL && now - last_report >= 2.0) {
            fprintf(stderr, "ds4: CUDA model chunk copy %.2f/%.2f GiB\n",
                    (double)copied / 1073741824.0,
                    (double)map_size / 1073741824.0);
            last_report = now;
        }
    }

    (void)cudaFreeHost(stage);
    g_model_device_base = (const char *)dev;
    g_model_device_owned = 1;
    g_model_hmm_direct = 0;
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "ds4: CUDA model chunk copy complete in %.3fs (%.2f GiB tensors)\n",
            t1 - t0,
            (double)map_size / 1073741824.0);
    return 1;
}

void cuda_model_range_release_all(void) {
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_registered && r.registered_base) {
            (void)cudaHostUnregister(r.registered_base);
        } else if (r.device_ptr && !r.arena_allocated) {
            (void)cudaFree(r.device_ptr);
        }
    }
    for (const cuda_model_arena &a : g_model_arenas) {
        if (a.device_ptr) (void)cudaFree(a.device_ptr);
    }
    g_model_arenas.clear();
    g_model_ranges.clear();
    g_model_range_by_offset.clear();
    g_model_range_bytes = 0;
    cuda_model_load_progress_reset();
}

static void cuda_derived_range_release_all(void) {
    ds4_mmq_set_aligned_q81_scratch(NULL, 0);
    if (g_aligned_q81_scratch) {
        (void)cudaFree(g_aligned_q81_scratch);
        g_aligned_q81_scratch = NULL;
    }
    for (const cuda_derived_range &r : g_derived_ranges) {
        if (r.device_ptr) (void)cudaFree(r.device_ptr);
    }
    g_derived_ranges.clear();
    g_derived_replace_map = NULL;
    g_derived_artifact_bytes = 0;
    g_derived_artifact_build_secs = 0.0;
    g_derived_replaces_complete = 0;
}

int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: cuBLAS %s failed: status %d\n", what, (int)st);
    return 0;
}


extern "C" int ds4_gpu_init(void) {
    cuda_decode_dispatch_env_refresh();
    memset(&g_gpu[0], 0, sizeof(g_gpu[0]));
    ds4_gpu_ctx *context = &g_gpu[0];
    context->device_id = 0;
    g_n_gpus = 1;
    if (!cuda_ok(cudaSetDevice(0), "init set device")) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        fprintf(stderr, "ds4: CUDA backend initialized on %s (sm_%d%d) dev=0\n",
                prop.name, prop.major, prop.minor);
    }
    cudaStream_t stream = NULL;
    if (!cuda_ok(cudaStreamCreate(&stream), "init stream")) return 0;
    context->stream = (void *)stream;

    cudaEvent_t event = NULL;
    if (!cuda_ok(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                 "init event")) {
        ds4_gpu_cleanup();
        return 0;
    }
    context->boundary_event = (void *)event;

    cublasHandle_t cublas = NULL;
    if (!cublas_ok(cublasCreate(&cublas), "init cublas")) {
        ds4_gpu_cleanup();
        return 0;
    }
    context->cublas = (void *)cublas;
    const cublasMath_t math_mode =
        (g_quality_mode || getenv("DS4_CUDA_NO_TF32") != NULL)
            ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH;
    (void)cublasSetMathMode(cublas, math_mode);
    context->cublas_ready = 1;
    g_cublas_ready = 1;
    return 1;
}

extern "C" void ds4_gpu_cleanup(void) {
    (void)cudaDeviceSynchronize();
    /* The single GB10 owns one stream, one cuBLAS handle, and one scratch set. */
    for (int i = 0; i < g_n_gpus; i++) {
        ds4_gpu_ctx *c = &g_gpu[i];
        (void)cudaSetDevice(c->device_id);
        attention_decode_score_split_graph_destroy_one(i);
        routed_moe_decode_graph_destroy_one(i);
        if (c->boundary_event) {
            (void)cudaEventDestroy((cudaEvent_t)c->boundary_event);
            c->boundary_event = NULL;
        }
        if (c->stream) {
            (void)cudaStreamDestroy((cudaStream_t)c->stream);
            c->stream = NULL;
        }
        if (c->cublas) {
            (void)cublasDestroy((cublasHandle_t)c->cublas);
            c->cublas = NULL;
            c->cublas_ready = 0;
        }
        if (g_tt_scratch && g_tt_scratch_device == c->device_id) {
            (void)cudaFree(g_tt_scratch);
            g_tt_scratch = NULL;
            g_tt_scratch_bytes = 0;
            g_tt_scratch_device = -1;
        }
#ifdef DS4_CUDA_HAVE_MXF4
        if (g_indexer_mxf4_scratch &&
            g_indexer_mxf4_scratch_device == c->device_id) {
            (void)cudaFree(g_indexer_mxf4_scratch);
            g_indexer_mxf4_scratch = NULL;
            g_indexer_mxf4_scratch_bytes = 0;
            g_indexer_mxf4_scratch_device = -1;
        }
#endif
    }
    g_n_gpus = 0;
    g_cublas_ready = 0;

    cuda_model_range_release_all();
    cuda_derived_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (g_model_upload_stream) {
        (void)cudaStreamDestroy(g_model_upload_stream);
        g_model_upload_stream = NULL;
    }
    if (g_model_device_owned && g_model_device_base) {
        (void)cudaFree((void *)g_model_device_base);
    }
    if (g_model_registered && g_model_host_base) {
        (void)cudaHostUnregister((void *)g_model_host_base);
    }
    g_model_host_base = NULL;
    g_model_device_base = NULL;
    g_model_registered_size = 0;
    g_model_registered = 0;
    g_model_device_owned = 0;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_fd = -1;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    g_model_file_size = 0;
    g_model_cache_full = 0;
    if (g_model_prefetch_stream) {
        (void)cudaStreamDestroy(g_model_prefetch_stream);
        g_model_prefetch_stream = NULL;
    }
}

extern "C" int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int device_id,
                                       uint64_t bytes) {
    if (!t) return 1;
    if (device_id < 0 || device_id >= g_n_gpus) return 2;
    if (bytes == 0) bytes = 1;
    int ok = 0;
    WITH_DEVICE(g_gpu[device_id].device_id) {
        ok = cuda_ok(cudaMalloc(&t->ptr, (size_t)bytes), "tensor alloc");
    }
    if (!ok) return 3;
    t->bytes = bytes;
    t->owner = 1;
    t->device_id = device_id;
    return 0;
}
