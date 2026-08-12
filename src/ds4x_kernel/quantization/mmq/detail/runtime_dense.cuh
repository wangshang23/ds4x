// SPDX-License-Identifier: MIT

#pragma once
// ds4_mmq.cu - host wrapper around llama.cpp's vendored mul_mat_q kernels.
//
// The fused target-prefill MoE dispatcher (ds4_mmq_fused_down,
// ds4_swiglu_weighted_f32, the fused_down branches of
// ds4_mmq_moe_pair_impl, and the ds4_mmq_iq2_xxs_q2_K_moe_fused_* entry
// points) is Portions Copyright (c) 2026 Marco Palaferri (MIT), adapted
// from xangel82/DS4-GB10-GX10-DSpark-CUDA commit 910501e (v0.5 inc-9).
//
// Implements the public ds4_mmq_* entry points and explicitly instantiates
// the mul_mat_q_case<T> template for each quant type the caller needs.
//
// Status:
//   Q8_0 dense ............ implemented, parity-tested against CPU reference
//   Q2_K dense ............ pending (Phase 3)
//   IQ2_XXS dense ......... pending (Phase 3)
//   Q8_0 MoE _id .......... pending (Phase 4)
//   Q2_K MoE _id .......... pending (Phase 4)
//   IQ2_XXS MoE _id ....... pending (Phase 4)

#include "ds4_mmq.h"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"
#include "ds4_mmq_d2r.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstring>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define DS4_MMQ_HAS_NVTX 1
#endif
#endif
#ifndef DS4_MMQ_HAS_NVTX
#define DS4_MMQ_HAS_NVTX 0
#endif

static bool ds4_mmq_nvtx_requested() {
    static int enabled = -1;
    if (enabled < 0) {
        const char *nvtx = getenv("DS4_CUDA_NVTX");
        const char *capture = getenv("DS4_CUDA_NSYS_PREFILL_START_POS");
        enabled = (nvtx != nullptr && std::strcmp(nvtx, "1") == 0) ||
                  (capture != nullptr && capture[0] != '\0');
    }
    return enabled != 0;
}

static uint64_t ds4_mmq_nvtx_payload(uint32_t first, uint32_t second) {
    return ((uint64_t)first << 32) | second;
}

class ds4_mmq_nvtx_scope {
public:
    ds4_mmq_nvtx_scope(const char *name, uint64_t payload, bool enabled)
        : active_(enabled) {
#if DS4_MMQ_HAS_NVTX
        if (active_) {
            nvtxEventAttributes_t attr = {};
            attr.version = NVTX_VERSION;
            attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
            attr.payload.ullValue = payload;
            attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attr.message.ascii = name;
            (void)nvtxRangePushEx(&attr);
        }
#else
        (void)name;
        (void)payload;
        active_ = false;
#endif
    }

    ~ds4_mmq_nvtx_scope() {
#if DS4_MMQ_HAS_NVTX
        if (active_) (void)nvtxRangePop();
#endif
    }

    ds4_mmq_nvtx_scope(const ds4_mmq_nvtx_scope &) = delete;
    ds4_mmq_nvtx_scope &operator=(const ds4_mmq_nvtx_scope &) = delete;

private:
    bool active_;
};

// ----------------------------------------------------------------------------
// Init
// ----------------------------------------------------------------------------

// Step 7 task #29: experimental persistent Q8_1 scratch buffer.
//
// Hypothesis: ggml_cuda_pool_alloc inside ds4_mmq_moe_vec_impl records a
// cudaMallocAsync graph node into the captured layer graph.  At replay
// time the alloc node returns a (potentially different) address, but the
// matvec kernel's pointer argument was baked in at capture time.  Result:
// the matvec reads stale/wrong memory and produces a different output
// than eager execution, even with identical inputs.
//
// Mitigation under test: pre-allocate a persistent device buffer at
// startup via plain cudaMalloc (NOT cudaMallocAsync, NOT inside any
// capture).  When the env flag DS4_CUDA_MMQ_Q81_PERSISTENT=1 is set,
// ds4_mmq_moe_vec_impl uses this persistent buffer instead of pool_alloc.
//
// Sized for V4 Flash decode shapes: gate Q8_1 ~8 KB, down Q8_1 ~14 KB.
// 256 KB allocation gives generous headroom for short prefill batches.
static void *g_q81_scratch_ptr   = nullptr;
static size_t g_q81_scratch_bytes = 0;
static bool   g_q81_scratch_enabled = false;
static void  *g_aligned_q81_scratch_ptr = nullptr;
static size_t g_aligned_q81_scratch_bytes = 0;

extern "C" void ds4_mmq_set_aligned_q81_scratch(void *ptr, size_t bytes) {
    g_aligned_q81_scratch_ptr = ptr;
    g_aligned_q81_scratch_bytes = ptr ? bytes : 0;
}

// Read by ds4_mmq_moe_vec_impl; non-zero means use the persistent buffer.
// Set by ds4_mmq_init once based on env.  (Single-threaded GPU work; no
// atomicity needed.)
extern "C" int ds4_mmq_q81_persistent_enabled(void) {
    return g_q81_scratch_enabled ? 1 : 0;
}

extern "C" void *ds4_mmq_q81_scratch_ptr(void) {
    return g_q81_scratch_ptr;
}

// M2-Inc2a: registry of producer-emitted q8_1 activations (ds4_cuda.cu).
// A hit returns canonical block_q8_1 codes for this exact activation
// pointer (bit-exact vs quantize_row_q8_1_cuda), letting the caller skip
// its quantize prelude.  Only valid for single-token unpadded rows
// (ne10_padded == K); the registry itself guarantees freshness (slots are
// reset by the producing entry every layer and pops are one-shot).
extern "C" int ds4_cuda_q8_fold_take_q81(const void *src, uint64_t in_dim,
                                         const void **q81);
static char *ds4_mmq_folded_q81(const float *X_f32, int64_t K, int n_tokens,
                                int64_t ne10_padded) {
    if (n_tokens != 1 || ne10_padded != K) return nullptr;
    const void *p = nullptr;
    if (!ds4_cuda_q8_fold_take_q81((const void *)X_f32, (uint64_t)K, &p)) return nullptr;
    static int logged = 0;
    if (!logged) {
        logged = 1;
        fprintf(stderr, "ds4: M2-Inc2a q8_1 activation fold active (mmvq decode)\n");
    }
    return (char *)(uintptr_t)p;
}

// Default ON (2026-07-09 gated increment: same-boot ABBA 427->493 tok/s @12k,
// gsm8k 97.5 / mbpp 90). DS4_MMQ_D2R=0 is the kill switch back to the
// mul_mat_q SoA-tile down path.
static bool d2r_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_MMQ_D2R");
        cached = (env && env[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool d2r_iq2_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_MMQ_D2R_IQ2");
        cached = (env && env[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

// Blanket output zeroing on the dense/MoE-down/pair GEMM entries.  Added by
// 82b2622 as belt-and-suspenders while root-causing the cont BOS spam; the
// actual roots were fixed in the same commit (stream-K fixup write_back goes
// dense + tmp_fixup zeroed + ncols_max=ne_get_rows), after which every
// element a consumer reads is stored by the GEMM itself and the zeroing was
// ~1.0 s/12k-admission of pure memset tax.  Default OFF (2026-07-09 gated
// increment: L42 deep tensors BIT-IDENTICAL with/without, same-boot ABBA
// 641.5 -> 678 tok/s @12k, gsm8k 119/120 / mbpp 36/40 / canary=[]).
// DS4_MMQ_OUT_MEMSET=1 restores the zeroing.
static bool out_memset_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_MMQ_OUT_MEMSET");
        cached = (env && env[0] == '1') ? 1 : 0;
        if (cached) {
            fprintf(stderr, "ds4: DS4_MMQ_OUT_MEMSET=1 - blanket GEMM output zeroing restored\n");
        }
    }
    return cached != 0;
}

/* v0.5 inc-12 slice 2: Y-buffer (q8_1 activation) memset diet.  The S1.1a-era
 * zero of every quantize staging buffer before quantize_mmq_q8_1 cost ~2.3
 * s/180k of stream time (reslice10 MEMSET table: 56.6/28.3/18.9/9.45/4.7 MB
 * classes = the gateup/down/o_proj/dense/shexp Y buffers).  quantize writes
 * every valid column; only the never-written pad/slack tail is at stake, and
 * the mmq write_back masks tail lanes out of the output (the D2R kernels
 * guard their token loops outright).  Modes, same contract as the cublas ws
 * knob:
 *   DS4_MMQ_YBUF_MEMSET unset/0 -> no zero (default; bit-exact IFF no tail
 *     byte can reach an output, proven by the poison gate)
 *   =1      -> S1.1a always-zero (the old behavior)
 *   =poison -> fill 0xFF: the bit-exactness instrument.  Exact twins vs
 *     always-zero across the gate battery prove the masking claim; any
 *     drift means some path DOES leak tail bytes and OFF must not ship. */
static int ybuf_memset_mode() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("DS4_MMQ_YBUF_MEMSET");
        cached = 0;
        if (env && env[0] == '1') cached = 1;
        else if (env && (env[0] == 'p' || env[0] == 'P')) cached = 2;
        if (cached) {
            fprintf(stderr, "ds4: DS4_MMQ_YBUF_MEMSET=%s - q8_1 staging %s\n",
                    cached == 1 ? "1" : "poison",
                    cached == 1 ? "zeroing restored" : "poisoned (0xFF)");
        }
    }
    return cached;
}

static void ybuf_memset(void *ptr, size_t bytes, cudaStream_t stream) {
    const int mode = ybuf_memset_mode();
    if (mode == 0 || ptr == NULL || bytes == 0) return;
    (void)cudaMemsetAsync(ptr, mode == 1 ? 0 : 0xFF, bytes, stream);
}

/* flat-pool p5b: the direct fused gate/up path stages its input Q8 through
 * the ids_src1 column->token map inside the D2R kernel, so the activation
 * quantize runs once per TOKEN (n_tokens rows) instead of once per
 * assignment slot (n_tokens * top-k rows and bytes).  Bit-identical by
 * construction: the quantize is row-local, so a gathered slot for token t
 * holds exactly the bytes of compact row t; the kernel consumes the same
 * blocks in the same tile order through the indirection.  DS4_MMQ_NO_YIND
 * restores the slot-gathered quantize; DS4_MMQ_YIND_VERIFY byte-compares
 * the two buffers in situ (expect bad=0). */
static int moe_yind_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = getenv("DS4_MMQ_NO_YIND") == NULL ? 1 : 0;
        if (!cached) {
            fprintf(stderr, "ds4: DS4_MMQ_NO_YIND - moe gate/up y-indirect staging disabled\n");
        }
    }
    return cached;
}

static int moe_yind_verify_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = getenv("DS4_MMQ_YIND_VERIFY") != NULL ? 1 : 0;
    }
    return cached;
}

static int64_t d2r_min_cols() {
    static int64_t cached = -1;
    if (cached < 0) {
        cached = 1024;
        const char *env = getenv("DS4_MMQ_D2R_MIN_COLS");
        if (env && env[0] != '\0') {
            char *end = nullptr;
            const long v = strtol(env, &end, 10);
            if (end != env && v > 0) {
                cached = (int64_t)v;
            }
        }
    }
    return cached;
}

extern "C" size_t ds4_mmq_q81_scratch_bytes(void) {
    return g_q81_scratch_bytes;
}

extern "C" int ds4_mmq_init(int device) {
    if (device < 0) {
        fprintf(stderr, "ds4_mmq_init: invalid device %d\n", device);
        return -1;
    }
    ggml_cuda_set_device(device);
    // Trigger lazy population of the device-info singleton.
    const auto & info = ggml_cuda_info();
    if (info.device_count == 0) {
        fprintf(stderr, "ds4_mmq_init: no CUDA devices found\n");
        return -1;
    }
    if (device >= info.device_count) {
        fprintf(stderr, "ds4_mmq_init: device %d out of range (have %d)\n",
                device, info.device_count);
        return -1;
    }

    // Step 7 task #29: pre-allocate persistent Q8_1 scratch if enabled.
    // Must happen here (before any layer-graph capture) so the cudaMalloc
    // is not forbidden by capture-mode restrictions, and so the kernel
    // pointer arg baked into the captured graph stays valid at replay.
    if (getenv("DS4_CUDA_MMQ_Q81_PERSISTENT") && !g_q81_scratch_ptr) {
        const size_t bytes = 256 * 1024;
        cudaError_t err = cudaMalloc(&g_q81_scratch_ptr, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4_mmq_init: cudaMalloc(q81_scratch %zu B) failed: %s; "
                            "falling back to pool_alloc\n",
                    bytes, cudaGetErrorString(err));
            g_q81_scratch_ptr = nullptr;
            g_q81_scratch_enabled = false;
        } else {
            g_q81_scratch_bytes = bytes;
            g_q81_scratch_enabled = true;
            fprintf(stderr, "ds4_mmq_init: persistent Q8_1 scratch enabled (%zu B at %p)\n",
                    bytes, g_q81_scratch_ptr);
        }
    }
    return 0;
}

// ----------------------------------------------------------------------------
// Gating: when should the caller choose mmq over dequant+cublas?
//
// Body lifted verbatim from llama.cpp's ggml/src/ggml-cuda/mmq.cu:267-372
// (we do not vendor mmq.cu itself, since its other half talks to ggml_tensor
// and ggml_backend internals we don't carry over).
// ----------------------------------------------------------------------------

static bool ds4_should_use_mmq_impl(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_UNUSED(type); GGML_UNUSED(cc); GGML_UNUSED(ne11); GGML_UNUSED(n_experts);
    return false;
#endif

    bool mmq_supported;
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }
    if (!mmq_supported) return false;

    if (turing_mma_available(cc)) {
        return true;
    }
    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }
#ifdef GGML_CUDA_FORCE_MMQ
    GGML_UNUSED(ne11); GGML_UNUSED(n_experts);
    return true;
#endif

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }
    if (amd_mfma_available(cc)) {
        if (GGML_CUDA_CC_IS_CDNA3(cc)) return true;
        if (n_experts > 64 || ne11 <= 128) return true;
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
            type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) return true;
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) return true;
        return false;
    }
    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            if (n_experts >= 64) return true;
            switch (type) {
                case GGML_TYPE_Q2_K: return ne11 <= 128;
                case GGML_TYPE_Q6_K: return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default: return true;
            }
        }
        return true;
    }
    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

extern "C" int ds4_mmq_should_use(int type_x, int64_t ne11, int64_t n_experts) {
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    const enum ggml_type t = (enum ggml_type) type_x;
    return ds4_should_use_mmq_impl(t, cc, ne11, n_experts) ? 1 : 0;
}

// ----------------------------------------------------------------------------
// Dense matmul implementation, shared across all three quant types.
//
// Computes  out[col, row] = sum_k W[row, k] * X[k, col]   with W in the
// type-specific block layout and X / out in F32 (X K-innermost row-major,
// out column-major out[col*M + row]).
//
// Mirrors upstream mmq.cu:154-159 (the no-ids branch) but builds mmq_args
// from plain pointers + shape ints instead of ggml_tensor introspection.
// ----------------------------------------------------------------------------

// Per-device singleton context. Owns the pool for stream-K fixup scratch.
// Phase 4 will make this per-stream as well; for now a single context per
// device is sufficient for the dense path.
namespace {

__global__ static void ds4_mmq_sanitize_f32_kernel(float *p, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = p[i];
    if (!isfinite(v)) p[i] = 0.0f;
}

static void ds4_mmq_sanitize_f32(float *p, uint64_t n, cudaStream_t stream) {
    if (!p || n == 0) return;
    ds4_mmq_sanitize_f32_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, stream>>>(p, n);
}

ggml_backend_cuda_context * get_ctx_for_device(int device) {
    static ggml_backend_cuda_context * cached[GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    if (!cached[device]) {
        cached[device] = new ggml_backend_cuda_context(device);
    }
    return cached[device];
}

template <ggml_type type>
bool ds4_mmq_k_tile_supported(const char *tag, int K, int cc) {
    if constexpr (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4) {
        if (blackwell_mma_available(cc) && K % MMQ_ITER_K_FP4 != 0) {
            fprintf(stderr,
                    "%s: Blackwell FP4 K=%d must be a multiple of %d\n",
                    tag, K, MMQ_ITER_K_FP4);
            return false;
        }
    }
    return true;
}

template <ggml_type type>
int ds4_mmq_dense_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        cudaStream_t  stream) {

    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (K <= 0 || M <= 0 || N <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        // mmq requires K to be a multiple of the largest super-block size
        // it sees during the inner tile loop, which is QK_K=256.
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!ds4_mmq_k_tile_supported<type>(tag, K, cc)) return -1;

    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    /* Task #22 fix: order the pool's cudaMallocAsync/cudaFreeAsync on the SAME
     * stream the kernels below launch on.  The pool defaults to
     * cudaStreamPerThread; with kernels on the legacy stream the RAII free is
     * ordered on an EMPTY stream, so the driver can recycle/remap the scratch
     * while the in-flight quantize/GEMM still reads it -> intermittent illegal
     * access under shape churn (the batched-draft early-step crash).  The vec
     * impls already do this (graph-capture fix); the batched impls were missed. */
    ds4_pool_set_stream(stream);

    // 1. Quantize F32 activations into the format consumed by MMQ. Blackwell
    //    MXFP4/NVFP4 use native FP4 tensor cores; other paths use MMQ Q8_1.
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = N;
    const int64_t ne12         = 1;
    const int64_t ne13         = 1;

    const bool use_native_fp4 =
        (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4) &&
        blackwell_mma_available(cc);
    const size_t y_block_size = use_native_fp4
        ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4
        ? QK_FP4_MMQ : 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne13 * ne12 * ne11 * ne10_padded * y_block_size /
            y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);

    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr &&
        g_q81_scratch_bytes >= nbytes_src1_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_src1_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    // S1.1a fix: the mmq Y (activation) buffer is over-allocated for the kernel's
    // tail-tile reads (the +mmq_x_max blocks above), and ne11 columns may not fill
    // the final column tile -- but quantize_mmq_q8_1_cuda only writes the ne11 valid
    // columns.  The mmq kernel (mmq.cuh:3528) unconditionally loads the full column
    // tile, reading the never-written tail.  Pool allocs reuse stale device memory,
    // so that tail is non-deterministic: any allocator/stream perturbation (e.g. an
    // MTP draft's cudaMalloc) changes it and flips a near-threshold argmax in the
    // batched forward (confirmed by compute-sanitizer --tool initcheck on a PRO6000
    // / sm_120: 4-byte uninitialized __global__ read in mul_mat_q_process_tile).
    // The tail's dot-products are masked out by write_back, so only their
    // non-determinism matters; zero the buffer so the tail is a deterministic zero
    // (a zero q8_1 block contributes 0 to the dot product).
    ybuf_memset(src1_q8_1_ptr, nbytes_src1_q8_1, stream);

    if (use_native_fp4) {
        quantize_mmq_fp4_cuda(
            X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
            type, /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
            /*ne0=*/ne10_padded, /*ne1=*/ne11, /*ne2=*/ne12, /*ne3=*/ne13,
            stream);
    } else {
        quantize_mmq_q8_1_cuda(
            X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
            type, /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
            /*ne0=*/ne10_padded, /*ne1=*/ne11, /*ne2=*/ne12, /*ne3=*/ne13,
            stream);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    // 2. Build mmq_args. stride_row_x is in WEIGHT BLOCKS per row, which
    //    is K / blck_size(type). Q8_0 has block size 32; Q2_K and IQ2_XXS
    //    are K-quants with block size 256.
    const int64_t blck   = ggml_blck_size(type);
    const int64_t s01    = (int64_t)K / blck;
    const int64_t s1     = (int64_t)M;
    const int64_t s12    = ne11 * ne10_padded * y_block_size /
                           (y_values_per_block * sizeof(int));
    const int64_t s13    = ne12 * s12;

    const bool use_stream_k =
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc);

    if (out_memset_enabled()) {
        cudaMemsetAsync(out_f32, 0, (size_t)M * (size_t)N * sizeof(float), stream);
    }

    const mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1_ptr,
        /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr,
        /*dst=*/out_f32,
        /*ncols_x=*/ne00,    /*nrows_x=*/(int64_t)M,    /*ncols_dst=*/ne11,
        /*stride_row_x=*/s01,/*ncols_y=*/ne11,          /*nrows_dst=*/s1,
        /*nchannels_x=*/1,   /*nchannels_y=*/1,
        /*stride_channel_x=*/0, /*stride_channel_y=*/s12, /*stride_channel_dst=*/0,
        /*nsamples_x=*/1,    /*nsamples_y=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/s13, /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/ne11,
    };

    mul_mat_q_case<type>(*ctx, args, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case launch failed: %s\n", tag, cudaGetErrorString(err));
        return -3;
    }
    ds4_mmq_sanitize_f32(out_f32, (uint64_t)M * (uint64_t)N, stream);
    return 0;
}

} // anonymous namespace

