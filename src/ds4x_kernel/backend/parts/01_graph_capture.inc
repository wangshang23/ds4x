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
static inline cudaStream_t cuda_decode_stream(void) {
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
/* Producer-fold registry lookup the vendored ds4_mmq.cu entries probe for
 * pre-quantized q8_1 activations (the fork's flat-pool M2-Inc2a fold).
 * The flat-pool producers are not ported, so nothing is ever registered:
 * always miss, and the entries run their own activation quantize. */
extern "C" int ds4_cuda_q8_fold_take_q81(const void *src, uint64_t in_dim,
                                         const void **q81) {
    (void)src; (void)in_dim; (void)q81;
    return 0;
}

static int cuda_use_mmq(void) {
    static int init = 0;
    static int use = 0;
    if (!init) {
        init = 1;
        const char *s = getenv("DS4_CUDA_MMQ");
        const int off = (s && s[0] == '0') || g_quality_mode || g_n_gpus > 1;
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

/* MXFP4 has no dequant+cublas fallback, so it must retain MMQ on multi-GPU
 * placements where the optional Q8/IQ2 prefill tier stays disabled. MMQ
 * resolves the active CUDA device on every call; initialization only warms
 * its device-info singleton. The experimental global persistent scratch is
 * intentionally rejected because one pointer cannot span CUDA devices. */
static int cuda_use_mxfp4_mmq(void) {
    static int init = 0;
    static int use = 0;
    if (!init) {
        init = 1;
        const char *s = getenv("DS4_CUDA_MMQ");
        if (s && s[0] == '0') {
            fprintf(stderr,
                    "ds4: DS4_CUDA_MMQ=0 - MXFP4 MMQ disabled\n");
        } else if (g_n_gpus > 1 &&
                   getenv("DS4_CUDA_MMQ_Q81_PERSISTENT") != NULL) {
            fprintf(stderr,
                    "ds4: persistent MMQ Q8_1 scratch is unavailable with multi-GPU MXFP4\n");
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
        float clamp) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_expert_used * expert_mid_dim;
    if (gid >= n) return;
    uint64_t slot_pair = gid / expert_mid_dim;
    uint32_t tok = (uint32_t)(slot_pair / n_expert_used);
    uint32_t slot = (uint32_t)(slot_pair - (uint64_t)tok * n_expert_used);
    float g = gate_buf[gid];
    float u = up_buf[gid];
    if (!isfinite(g)) g = 0.0f;
    if (!isfinite(u)) u = 0.0f;
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }
    const float w = weights[(uint64_t)tok * n_expert_used + slot];
    const float s = g / (1.0f + expf(-g));
    mid_out[gid] = s * u * w;
}

__global__ static void moe_mmq_sum_kernel(float *out, const float *down,
        const int32_t *selected, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_tokens, uint32_t guard_nonfinite) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) {
        if (selected && selected[(uint64_t)tok * n_expert + e] < 0) continue;
        const float v = down[((uint64_t)tok * n_expert + e) * out_dim + row];
        if (!guard_nonfinite || isfinite(v)) acc += v;
    }
    out[gid] = acc;
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

/* Multi-tier-aware weight pointer resolver.
 *
 * Used by kernel-dispatch wrappers in the per-layer execution path to
 * obtain a device pointer for a weight slice on the layer's logical
 * tier. Behavior:
 *
 *   - When g_n_gpus <= 1 (single-tier engine), delegates to the existing
 *     cuda_model_range_ptr path. This short-circuit guarantees byte-
 *     identical behavior to pre-multi-tier code for the gpu_cfg == NULL
 *     case.
 *
 *   - When g_n_gpus >= 2 (multi-tier engine), translates the logical
 *     tier index to the corresponding physical CUDA device id via
 *     g_gpu[logical_tier].device_id and looks up the slice strictly
 *     in the per-device selective cache via ds4_gpu_lookup_cache_strict.
 *     On miss, logs a diagnostic and returns NULL (no host-pointer
 *     fallback — a miss here is a placement/install bug).
 *
 * The caller is responsible for cudaSetDevice'ing to the right physical
 * device before launching the kernel that consumes the returned pointer.
 * Wrappers in this file thread `int logical_tier` from the dispatch
 * caller; single-tier callers pass 0, which hits the short-circuit.
 *
 * Added for multi-GPU execution (multi-GPU execution), sub-area 3 of the
 * spec. */
/* Optional second (support) model map for speculative decoding. The strict
 * multi-tier cache is keyed by source offset only, so support tensors are
 * installed and resolved with a large disjoint offset bias. */
static const void *g_support_host_base = NULL;
static uint64_t g_support_host_size = 0;
static uint64_t g_support_offset_bias = 0;

extern "C" uint64_t ds4_gpu_tier_free_vram(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= g_n_gpus) return 0;
    int prev = -1;
    if (cudaGetDevice(&prev) != cudaSuccess) prev = -1;
    if (cudaSetDevice(g_gpu[logical_tier].device_id) != cudaSuccess) return 0;
    size_t free_b = 0, total_b = 0;
    uint64_t out = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) out = (uint64_t)free_b;
    if (prev >= 0) (void)cudaSetDevice(prev);
    return out;
}

extern "C" int ds4_gpu_register_support_map(const void *map, uint64_t size, uint64_t bias) {
    if (!map || size == 0 || bias == 0) return 0;
    g_support_host_base = map;
    g_support_host_size = size;
    g_support_offset_bias = bias;
    return 1;
}

static const char *cuda_resolve_weight_ptr(const void *model_map,
                                            uint64_t offset,
                                            uint64_t bytes,
                                            int logical_tier,
                                            const char *label) {
    if (g_n_gpus <= 1) {
        return cuda_model_range_ptr(model_map, offset, bytes, label);
    }
    if (g_support_host_base && model_map == g_support_host_base) {
        offset += g_support_offset_bias;
    }
    if (logical_tier < 0 || logical_tier >= g_n_gpus) {
        fprintf(stderr,
            "ds4: cuda_resolve_weight_ptr: bad tier %d (n_gpus=%d, label=%s)\n",
            logical_tier, g_n_gpus, label ? label : "?");
        return NULL;
    }
    const int physical_device = g_gpu[logical_tier].device_id;
    void *dev_ptr = NULL;
    if (ds4_gpu_lookup_cache_strict(offset, bytes, physical_device, &dev_ptr)
        && dev_ptr) {
        return (const char *)dev_ptr;
    }
    /* GLM multi-tier: generic launchers resolve by the OUT tensor's tier,
     * but the executing device (set per layer) is where the weights were
     * cached. Retry with the current device before declaring a miss;
     * DS4 paths never reach this (out tier == current device). */
    int cur_dev = -1;
    if (cudaGetDevice(&cur_dev) == cudaSuccess &&
        cur_dev != physical_device &&
        ds4_gpu_lookup_cache_strict(offset, bytes, cur_dev, &dev_ptr) &&
        dev_ptr) {
        return (const char *)dev_ptr;
    }
    fprintf(stderr,
        "ds4: selective-cache miss for offset=%llu bytes=%llu on "
        "logical_tier=%d (physical_device=%d, current_device=%d, "
        "label=%s); this is a placement/cache-install bug\n",
        (unsigned long long)offset, (unsigned long long)bytes,
        logical_tier, physical_device, cur_dev, label ? label : "?");
    return NULL;
}

static int cuda_model_range_is_cached(const void *model_map, uint64_t offset, uint64_t bytes) {
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

static void cuda_q8_f16_cache_release_all(void) {
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

static uint32_t cuda_parse_u32_env_clamped(const char *name, uint32_t fallback,
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

static int cuda_env_flag_enabled(const char *name, int fallback) {
    const char *env = getenv(name);
    if (!env || !env[0]) return fallback;
    return strcmp(env, "0") != 0;
}

extern "C" int ds4_gpu_set_decode_fast_attention(int enabled) {
    const int old = g_decode_fast_attention;
    g_decode_fast_attention = enabled != 0;
    return old;
}

extern "C" int ds4_gpu_set_decode_score_vec4(int enabled) {
    const int old = g_decode_score_vec4;
    g_decode_score_vec4 = enabled != 0;
    return old;
}

static bool cuda_splitkv_decode_requested(void) {
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

static void cuda_q8_f16_cache_disable_after_failure(const char *what, uint64_t request_bytes) {
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

static int cuda_q8_use_dp4a(void) {
    return getenv("DS4_CUDA_NO_Q8_DP4A") == NULL;
}

static unsigned cuda_q8_exact_threads(uint64_t blocks) {
    if (blocks <= 64u) return 64u;
    if (blocks <= 128u) return 128u;
    return 256u;
}

static int cuda_q8_f16_preload_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (cuda_q8_label_is_attention_output(label) &&
        getenv("DS4_CUDA_ATTENTION_OUTPUT_PRELOAD") == NULL &&
        getenv("DS4_CUDA_Q8_F16_ALL") == NULL) {
        return 0;
    }
    return cuda_q8_f16_cache_allowed(label, in_dim, out_dim);
}

static int cuda_q8_f32_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
#ifdef DS4_CUDA_SPARK_ONLY
    (void)label;
    (void)in_dim;
    (void)out_dim;
    return 0;
#else
    if (g_q8_cache_suppressed) return 0;
    if (getenv("DS4_CUDA_NO_Q8_F32_CACHE") != NULL) return 0;
    if (getenv("DS4_CUDA_Q8_F32_ALL") != NULL) return 1;
    if (label && strstr(label, "attn_q_b") != NULL) {
        return getenv("DS4_CUDA_ATTN_Q_B_F32_CACHE") != NULL;
    }
    return getenv("DS4_CUDA_Q8_F32_LARGE") != NULL &&
           in_dim == 1024u && out_dim == 32768u;
#endif
}

/* Look up a per-device dequantized fp16 slice of the Q8_0 weight at
 * (model_map, offset, weight_bytes, in_dim, out_dim). expected_device is a
 * PHYSICAL CUDA device id (0 in single-tier; g_gpu[logical_tier].device_id in
 * multi-tier). On hit returns the cached pointer for that device. On miss
 * cudaSetDevice's to expected_device, allocates + dequants there, stamps the
 * new entry with device_id == expected_device, restores the previous device,
 * and returns the new pointer.
 *
 * Single-tier (g_n_gpus <= 1) uses the offset-keyed map for a fast path —
 * legacy entries were stamped device_id=0, so the map remains authoritative.
 * Multi-tier linear-scans the ranges vector filtering on device_id (the same
 * offset may now legitimately map to multiple entries, one per device).
 */
static const __half *cuda_q8_f16_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        int expected_device,
        const char *label) {
    if (g_n_gpus <= 1) {
        auto exact = g_q8_f16_by_offset.find(offset);
        if (exact != g_q8_f16_by_offset.end()) {
            const cuda_q8_f16_range &r = g_q8_f16_ranges[exact->second];
            if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
                r.in_dim == in_dim && r.out_dim == out_dim) {
                return r.device_ptr;
            }
        }
    } else {
        for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
            if (r.host_base == model_map &&
                r.offset == offset &&
                r.weight_bytes == weight_bytes &&
                r.in_dim == in_dim &&
                r.out_dim == out_dim &&
                r.device_id == expected_device) {
                return r.device_ptr;
            }
        }
    }
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;

    /* Source Q8 bytes:
     *  - Single-tier (g_n_gpus <= 1): cuda_model_range_ptr — preserves the
     *    legacy behavior (FD cache, host-register, or cudaMalloc-and-copy).
     *  - Multi-tier (g_n_gpus > 1): the per-device selective cache must
     *    already contain the weight on expected_device. Use the strict
     *    lookup; on miss this is a placement bug and we hard-fail.
     */
    const char *q8;
    if (g_n_gpus <= 1) {
        q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    } else {
        void *strict_ptr = NULL;
        if (!ds4_gpu_lookup_cache_strict(offset, weight_bytes, expected_device, &strict_ptr) ||
            !strict_ptr) {
            fprintf(stderr,
                "ds4: q8 fp16 cache miss: source bytes not in selective cache for "
                "offset=%llu bytes=%llu device=%d (label=%s); placement bug\n",
                (unsigned long long)offset, (unsigned long long)weight_bytes,
                expected_device, label ? label : "?");
            return NULL;
        }
        q8 = (const char *)strict_ptr;
    }
    if (!q8) return NULL;

    if (in_dim != 0 && out_dim > UINT64_MAX / in_dim / sizeof(__half)) return NULL;
    const uint64_t out_bytes = in_dim * out_dim * sizeof(__half);
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) return NULL;

    int prev = -1;
    if (g_n_gpus > 1) {
        cudaError_t derr = cudaGetDevice(&prev);
        if (derr != cudaSuccess) {
            fprintf(stderr, "ds4: cudaGetDevice failed before q8 fp16 alloc on device %d: %s\n",
                    expected_device, cudaGetErrorString(derr));
            (void)cudaGetLastError();
            return NULL;
        }
        derr = cudaSetDevice(expected_device);
        if (derr != cudaSuccess) {
            fprintf(stderr, "ds4: cudaSetDevice(%d) failed before q8 fp16 alloc: %s\n",
                    expected_device, cudaGetErrorString(derr));
            (void)cudaGetLastError();
            if (prev >= 0) (void)cudaSetDevice(prev);
            return NULL;
        }
    }
    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp16 cache alloc failed on device %d (%.2f MiB): %s\n",
                expected_device, (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("allocation failure", out_bytes);
        if (g_n_gpus > 1 && prev >= 0) (void)cudaSetDevice(prev);
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
        if (g_n_gpus > 1 && prev >= 0) (void)cudaSetDevice(prev);
        return NULL;
    }
    g_q8_f16_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev, expected_device});
    if (g_n_gpus <= 1) {
        g_q8_f16_by_offset[offset] = g_q8_f16_ranges.size() - 1u;
    }
    g_q8_f16_bytes += out_bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached q8 fp16 %.2f MiB on device %d (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0, expected_device,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    if (g_n_gpus > 1 && prev >= 0) (void)cudaSetDevice(prev);
    return dev;
}

/* Per-device dequantized fp32 cache. Same conventions as cuda_q8_f16_ptr. */
static float *cuda_q8_f32_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        int expected_device,
        const char *label) {
    if (g_n_gpus <= 1) {
        auto exact = g_q8_f32_by_offset.find(offset);
        if (exact != g_q8_f32_by_offset.end()) {
            const cuda_q8_f32_range &r = g_q8_f32_ranges[exact->second];
            if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
                r.in_dim == in_dim && r.out_dim == out_dim) {
                return r.device_ptr;
            }
        }
    } else {
        for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
            if (r.host_base == model_map &&
                r.offset == offset &&
                r.weight_bytes == weight_bytes &&
                r.in_dim == in_dim &&
                r.out_dim == out_dim &&
                r.device_id == expected_device) {
                return r.device_ptr;
            }
        }
    }
    if (!cuda_q8_f32_cache_allowed(label, in_dim, out_dim)) return NULL;

    /* Source Q8 bytes: legacy path in single-tier; strict per-device lookup
     * in multi-tier (same rationale as cuda_q8_f16_ptr). */
    const char *q8;
    if (g_n_gpus <= 1) {
        q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, label ? label : "q8_0");
    } else {
        void *strict_ptr = NULL;
        if (!ds4_gpu_lookup_cache_strict(offset, weight_bytes, expected_device, &strict_ptr) ||
            !strict_ptr) {
            fprintf(stderr,
                "ds4: q8 fp32 cache miss: source bytes not in selective cache for "
                "offset=%llu bytes=%llu device=%d (label=%s); placement bug\n",
                (unsigned long long)offset, (unsigned long long)weight_bytes,
                expected_device, label ? label : "?");
            return NULL;
        }
        q8 = (const char *)strict_ptr;
    }
    if (!q8) return NULL;

    const uint64_t out_bytes = in_dim * out_dim * sizeof(float);
    int prev = -1;
    if (g_n_gpus > 1) {
        cudaError_t derr = cudaGetDevice(&prev);
        if (derr != cudaSuccess) {
            fprintf(stderr, "ds4: cudaGetDevice failed before q8 fp32 alloc on device %d: %s\n",
                    expected_device, cudaGetErrorString(derr));
            (void)cudaGetLastError();
            return NULL;
        }
        derr = cudaSetDevice(expected_device);
        if (derr != cudaSuccess) {
            fprintf(stderr, "ds4: cudaSetDevice(%d) failed before q8 fp32 alloc: %s\n",
                    expected_device, cudaGetErrorString(derr));
            (void)cudaGetLastError();
            if (prev >= 0) (void)cudaSetDevice(prev);
            return NULL;
        }
    }
    float *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp32 cache alloc failed on device %d (%.2f MiB): %s\n",
                expected_device, (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        if (g_n_gpus > 1 && prev >= 0) (void)cudaSetDevice(prev);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f32_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp32 dequant launch")) {
        (void)cudaFree(dev);
        if (g_n_gpus > 1 && prev >= 0) (void)cudaSetDevice(prev);
        return NULL;
    }
    g_q8_f32_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev, expected_device});
    if (g_n_gpus <= 1) {
        g_q8_f32_by_offset[offset] = g_q8_f32_ranges.size() - 1u;
    }
    g_q8_f32_bytes += out_bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached q8 fp32 %.2f MiB on device %d (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0, expected_device,
                (double)g_q8_f32_bytes / 1073741824.0);
    }
    if (g_n_gpus > 1 && prev >= 0) (void)cudaSetDevice(prev);
    return dev;
}

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "ds4: CUDA %s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static double cuda_wall_sec(void) {
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

static int cuda_model_prefetch_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
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

#if 0
static void cuda_stream_selected_stage_release(void) {
    for (size_t i = 0; i < 4; i++) {
        if (g_stream_selected_stage_event[i]) {
            (void)cudaEventDestroy(g_stream_selected_stage_event[i]);
            g_stream_selected_stage_event[i] = NULL;
        }
        if (g_stream_selected_stage_raw[i]) {
            (void)cudaFreeHost(g_stream_selected_stage_raw[i]);
            g_stream_selected_stage_raw[i] = NULL;
            g_stream_selected_stage[i] = NULL;
        }
    }
    g_stream_selected_stage_bytes = 0;
    if (g_stream_selected_upload_stream) {
        (void)cudaStreamDestroy(g_stream_selected_upload_stream);
        g_stream_selected_upload_stream = NULL;
    }
}

static int cuda_stream_selected_stage_pool_alloc(uint64_t bytes) {
    if (g_stream_selected_stage_bytes >= bytes) return 1;
    cuda_stream_selected_stage_release();
    cudaError_t err = cudaStreamCreateWithFlags(
            &g_stream_selected_upload_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected upload stream creation failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    for (size_t i = 0; i < 4; i++) {
        err = cudaMallocHost(&g_stream_selected_stage_raw[i], (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging allocation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_stream_selected_stage_release();
            return 0;
        }
        g_stream_selected_stage[i] = cuda_align_ptr(
                g_stream_selected_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_stream_selected_stage_event[i],
                                       cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging event creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            cuda_stream_selected_stage_release();
            return 0;
        }
    }
    g_stream_selected_stage_bytes = bytes;
    return 1;
}

static int cuda_model_copy_to_device_streamed(
        char *dst,
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (!dst || !model_map || offset > model_size ||
        bytes > model_size - offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    if (g_model_fd < 0 ||
        (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base)) {
        return cuda_ok(cudaMemcpy(dst,
                                  (const char *)model_map + offset,
                                  (size_t)bytes,
                                  cudaMemcpyHostToDevice),
                       what ? what : "stream selected expert copy");
    }

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes =
        chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_stream_selected_stage_pool_alloc(stage_bytes)) return 0;

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = bytes - copied < chunk ? bytes - copied : chunk;
        const uint64_t bi = chunk_idx % 4u;
        cudaError_t err;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_stream_selected_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "ds4: CUDA streaming selected staging wait failed for %s: %s\n",
                        what ? what : "expert", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return 0;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_stream_selected_stage[bi],
                                   g_stream_selected_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected read failed for %s at %.2f MiB: %s\n",
                    what ? what : "expert", (double)copied / 1048576.0,
                    strerror(errno));
            return 0;
        }
        err = cudaMemcpyAsync(dst + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice,
                              g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "expert", (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        err = cudaEventRecord(g_stream_selected_stage_event[bi],
                              g_stream_selected_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging record failed for %s: %s\n",
                    what ? what : "expert", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, model_size,
                                        offset + copied, n);
        copied += n;
        chunk_idx++;
    }

    const cudaError_t err =
        cudaStreamSynchronize(g_stream_selected_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected upload sync failed for %s: %s\n",
                what ? what : "expert", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}
#endif

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

static const char *cuda_model_range_ptr_from_fd(
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

static int cuda_model_copy_chunked(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
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

static void cuda_model_range_release_all(void) {
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

static int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: cuBLAS %s failed: status %d\n", what, (int)st);
    return 0;
}

#if 0
extern "C" int ds4_gpu_init_multi(const ds4_gpu_config *cfg) {
    if (!cfg || cfg->n_gpus < 1 || cfg->n_gpus > DS4_MAX_GPUS) return 0;
    cuda_xdev_env_refresh();
    cuda_decode_dispatch_env_refresh();
    g_current_logical_tier = -1;

    /* g_n_gpus is published incrementally so ds4_gpu_cleanup() can unwind
     * partial state on failure. We publish `i + 1` BEFORE allocating any
     * resources for context `i`, so even if (e.g.) stream creation
     * succeeds but event creation fails, cleanup still walks device `i`
     * and destroys the stream. ds4_gpu_cleanup is null-safe per field —
     * partial state is OK. */
    for (int i = 0; i < cfg->n_gpus; i++) {
        ds4_gpu_ctx *c = &g_gpu[i];
        c->device_id = cfg->device_indices[i];
        if (c->device_id < 0) return 0;
        /* Publish the in-progress device id so cleanup can target it on
         * any later failure. cudaSetDevice is also required before
         * cleanup's cudaEventDestroy / cudaStreamDestroy / cublasDestroy
         * calls hit the right context. */
        g_n_gpus = i + 1;
        if (!cuda_ok(cudaSetDevice(c->device_id), "init set device")) return 0;
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, c->device_id) == cudaSuccess) {
            fprintf(stderr, "ds4: CUDA backend initialized on %s (sm_%d%d) dev=%d\n",
                    prop.name, prop.major, prop.minor, c->device_id);
        }
        /* Per-device stream. */
        cudaStream_t s = NULL;
        if (!cuda_ok(cudaStreamCreate(&s), "init stream")) return 0;
        c->stream = (void *)s;
        /* Per-device boundary event (reusable, no timing). */
        cudaEvent_t ev = NULL;
        if (!cuda_ok(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming),
                      "init event")) return 0;
        c->boundary_event = (void *)ev;
        /* Per-device cuBLAS handle. */
        cublasHandle_t h = NULL;
        if (!cublas_ok(cublasCreate(&h), "init cublas")) return 0;
        c->cublas = (void *)h;
        const cublasMath_t math_mode =
            (g_quality_mode || getenv("DS4_CUDA_NO_TF32") != NULL)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(h, math_mode);
        c->cublas_ready = 1;
        c->budget_bytes = cfg->vram_bytes[i];
        c->used_bytes   = 0;
        c->scratch      = NULL;
        c->scratch_bytes = 0;
    }

    /* NxN peer-access matrix.
     *
     * Driver semantics: cudaDeviceCanAccessPeer + cudaDeviceEnablePeerAccess
     * can both succeed even on hardware/drivers where cudaMemcpyPeerAsync
     * silently delivers wrong data (notably RTX 6000 Ada under recent
     * NVIDIA drivers, per the v0 design doc). The corruption is
     * non-deterministic and can affect either or both directions of a
     * pair. To guard against this, we run a multi-size, multi-iteration
     * validation at init (see the loop below): write distinct known
     * patterns, peer-copy them to the destination, read back, and only
     * set peer_ok[i][j] if every probe round-trips byte-perfect. A
     * single small probe is not sufficient — it can pass while realistic
     * activation-sized copies still corrupt. On any failure the entry
     * stays at 0 and cross-device copies fall back to the pinned-host
     * bounce path automatically. */
    for (int i = 0; i < g_n_gpus; i++) {
        for (int j = 0; j < g_n_gpus; j++) {
            if (i == j) { g_gpu_peer_ok[i][j] = 1; continue; }
            int can = 0;
            (void)cudaDeviceCanAccessPeer(&can, g_gpu[i].device_id,
                                          g_gpu[j].device_id);
            if (!can) { g_gpu_peer_ok[i][j] = 0; continue; }
            (void)cudaSetDevice(g_gpu[i].device_id);
            cudaError_t e = cudaDeviceEnablePeerAccess(g_gpu[j].device_id, 0);
            int enabled = (e == cudaSuccess ||
                           e == cudaErrorPeerAccessAlreadyEnabled);
            (void)cudaGetLastError();
            if (!enabled) { g_gpu_peer_ok[i][j] = 0; continue; }

            /* Runtime validation: peer copies on RTX 6000 Ada under recent
             * NVIDIA drivers silently corrupt at realistic sizes even though
             * the API returns success. cudaDeviceCanAccessPeer and
             * cudaDeviceEnablePeerAccess can both report success while
             * cudaMemcpyPeer delivers wrong data non-deterministically.
             * Probe with multiple sizes and iterations; ALL must round-trip
             * byte-perfect or we disable peer for this pair and silently fall
             * back to the pinned-host bounce path. */
            static const size_t kValidateSizes[] = {
                4u * 1024u,
                256u * 1024u,
                1u * 1024u * 1024u,
                16u * 1024u * 1024u,
            };
            const int kValidateIters    = 4;
            const int kNValidateSizes   = (int)(sizeof(kValidateSizes) /
                                                sizeof(kValidateSizes[0]));
            const size_t kMaxValidate   = kValidateSizes[kNValidateSizes - 1];

            unsigned char *vh_src = (unsigned char *)malloc(kMaxValidate);
            unsigned char *vh_dst = (unsigned char *)malloc(kMaxValidate);
            if (!vh_src || !vh_dst) {
                free(vh_src); free(vh_dst);
                g_gpu_peer_ok[i][j] = 0; continue;
            }

            void *src_dev = NULL; void *dst_dev = NULL;
            (void)cudaSetDevice(g_gpu[i].device_id);
            if (cudaMalloc(&src_dev, kMaxValidate) != cudaSuccess) {
                (void)cudaGetLastError();
                free(vh_src); free(vh_dst);
                g_gpu_peer_ok[i][j] = 0; continue;
            }
            (void)cudaSetDevice(g_gpu[j].device_id);
            if (cudaMalloc(&dst_dev, kMaxValidate) != cudaSuccess) {
                (void)cudaGetLastError();
                (void)cudaSetDevice(g_gpu[i].device_id);
                (void)cudaFree(src_dev);
                free(vh_src); free(vh_dst);
                g_gpu_peer_ok[i][j] = 0; continue;
            }

            int    peer_validated = 1;
            size_t failed_bytes   = 0;
            int    failed_iter    = -1;
            for (int s_idx = 0;
                 s_idx < kNValidateSizes && peer_validated;
                 s_idx++) {
                size_t n = kValidateSizes[s_idx];
                for (int it = 0; it < kValidateIters && peer_validated; it++) {
                    for (size_t k = 0; k < n; k++) {
                        vh_src[k] = (unsigned char)
                            ((k * 31u + (size_t)it * 17u +
                              (size_t)s_idx * 53u + 11u) & 0xffu);
                    }
                    (void)cudaSetDevice(g_gpu[i].device_id);
                    if (cudaMemcpy(src_dev, vh_src, n,
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
                        peer_validated = 0; failed_bytes = n; failed_iter = it;
                        break;
                    }
                    cudaError_t pc = cudaMemcpyPeer(
                        dst_dev, g_gpu[j].device_id,
                        src_dev, g_gpu[i].device_id, n);
                    if (pc != cudaSuccess) {
                        peer_validated = 0; failed_bytes = n; failed_iter = it;
                        break;
                    }
                    (void)cudaSetDevice(g_gpu[j].device_id);
                    if (cudaMemcpy(vh_dst, dst_dev, n,
                                   cudaMemcpyDeviceToHost) != cudaSuccess) {
                        peer_validated = 0; failed_bytes = n; failed_iter = it;
                        break;
                    }
                    if (memcmp(vh_src, vh_dst, n) != 0) {
                        peer_validated = 0; failed_bytes = n; failed_iter = it;
                        break;
                    }
                }
            }

            (void)cudaSetDevice(g_gpu[j].device_id);
            (void)cudaFree(dst_dev);
            (void)cudaSetDevice(g_gpu[i].device_id);
            (void)cudaFree(src_dev);
            free(vh_src);
            free(vh_dst);

            g_gpu_peer_ok[i][j] = peer_validated;
            if (peer_validated) {
                fprintf(stderr,
                    "ds4: peer access %d->%d validated across %d sizes x %d"
                    " iterations (max %zu MiB)\n",
                    g_gpu[i].device_id, g_gpu[j].device_id,
                    kNValidateSizes, kValidateIters,
                    kMaxValidate / (1024u * 1024u));
            } else {
                fprintf(stderr,
                    "ds4: peer access %d->%d FAILED validation at"
                    " size=%zu iter=%d; falling back to pinned-host bounce\n",
                    g_gpu[i].device_id, g_gpu[j].device_id,
                    failed_bytes, failed_iter);
            }
        }
    }

    g_cublas_ready = 1;
    return 1;
}

#endif

extern "C" int ds4_gpu_init(void) {
    cuda_decode_dispatch_env_refresh();
    g_current_logical_tier = -1;
    memset(&g_gpu[0], 0, sizeof(g_gpu[0]));
    ds4_gpu_ctx *context = &g_gpu[0];
    context->device_id = 0;
    g_n_gpus = 1;
    g_gpu_peer_ok[0][0] = 1;

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
    g_current_logical_tier = -1;

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
        if (c->scratch) {
            (void)cudaFree(c->scratch);
            c->scratch = NULL;
            c->scratch_bytes = 0;
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

    /* Per-device selective cache teardown (selective model cache). */
    for (int d = 0; d < DS4_MAX_GPUS; d++) {
        if (!g_dev_cache[d].present) continue;
        int prev = -1;
        (void)cudaGetDevice(&prev);
        (void)cudaSetDevice(d);
        if (g_dev_cache[d].base) (void)cudaFree(g_dev_cache[d].base);
        g_dev_cache[d].base = NULL;
        g_dev_cache[d].bytes = 0;
        g_dev_cache[d].present = 0;
        if (prev >= 0) (void)cudaSetDevice(prev);
    }
    g_cache_ranges.clear();

    /* Continue with legacy global teardown below. */

    cuda_model_range_release_all();
    cuda_derived_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
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

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);

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
    g_gpu[device_id].used_bytes += bytes;
    return 0;
}
