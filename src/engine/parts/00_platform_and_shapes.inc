/* =========================================================================
 * ds4.c - DeepSeek V4 inference engine.
 * =========================================================================
 *
 * This file is deliberately vertical: it owns GGUF loading, the fixed
 * DeepSeek V4 tensor layouts, CPU reference kernels, the whole-model Metal
 * graph driver, and tokenizer wiring.  Model shape selection is intentionally
 * narrow: validation accepts the known Flash and Pro layouts and fails early
 * for anything else.
 *
 * Loading is mmap based.  The loader parses only the GGUF header, metadata
 * table, and tensor directory.  Tensor data stays in the kernel page cache
 * until inference touches it, or until Metal wraps slices of the mapping as
 * no-copy MTLBuffers.
 */

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <strings.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_tp.h"

/* Wave-2 multi-GPU types are needed in every build because the engine
 * struct embeds ds4_gpu_config and the placement table. ds4_layer_pack.h
 * is included unconditionally for the same reason (engine helpers call
 * the packer in multi-tier mode, but the headers are tiny and C-safe). */
#include "ds4_layer_pack.h"
#include "ds4_gpu_mgpu.h"

#define DS4_CUDA_TP_PEER_TMP_BYTES \
    ((uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float) + 128u)

static uint32_t metal_graph_cuda_tp_output_requested_ways(void) {
    const char *env = getenv("DS4_CUDA_TP_OUTPUT_WAYS");
    if (!env || !env[0]) return 8;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end != env && *end == '\0' && v >= 2 && v <= DS4_MAX_GPUS) {
        return (uint32_t)v;
    }
    fprintf(stderr,
            "ds4: invalid DS4_CUDA_TP_OUTPUT_WAYS=%s; expected 2..%d\n",
            env, DS4_MAX_GPUS);
    return 2;
}

static uint32_t metal_graph_cuda_tp_output_tiers_for_head(
        int   head_tier,
        bool  cuda_tp_output,
        int   n_gpus,
        int   tiers[DS4_MAX_GPUS]) {
    if (!tiers || !cuda_tp_output || n_gpus < 2 || (n_gpus & 1) != 0) return 0;
    if (head_tier < 0 || head_tier >= n_gpus) return 0;
    const int half = n_gpus / 2;
    if (head_tier >= half) return 0;
    const int partner = head_tier + half;
    if (partner >= n_gpus) return 0;

    uint32_t want = metal_graph_cuda_tp_output_requested_ways();
    if (want > (uint32_t)n_gpus) want = (uint32_t)n_gpus;
    if (want < 2u) want = 2u;

    uint32_t n = 0;
    tiers[n++] = head_tier;
    if (partner != head_tier && n < want) tiers[n++] = partner;
    for (int pass = 0; pass < 2 && n < want; pass++) {
        const int start = pass == 0 ? half : 0;
        const int end = pass == 0 ? n_gpus : half;
        for (int t = start; t < end && n < want; t++) {
            bool seen = false;
            for (uint32_t i = 0; i < n; i++) {
                if (tiers[i] == t) {
                    seen = true;
                    break;
                }
            }
            if (!seen) tiers[n++] = t;
        }
    }
    return n;
}

#ifndef DS4_NO_GPU
#include "ds4_gpu.h"
#endif

/* Non-CUDA builds (Mac/Metal, CPU-only) never link ds4_cuda.cu. Provide
 * stubs for the multi-GPU plumbing multi-GPU functions and globals declared in
 * ds4_gpu_mgpu.h. These keep the linker happy on Mac/Metal and on CPU
 * builds. None of these are reached at runtime in non-CUDA builds
 * because multi_tier == 1 requires graph_backend == true which on
 * non-CUDA paths means Metal, and the multi-tier branch only fires
 * when the caller supplied a non-NULL gpu_cfg — which only the new
 * ds4_engine_create_with_gpu_config can do, and no Metal caller does.
 *
 * We key off __APPLE__ + DS4_NO_GPU rather than the inverse of CUDA
 * because there's no positive "is CUDA build" macro and ds4_cuda.cu
 * is only compiled in the non-Apple, non-DS4_NO_GPU configuration. */
/* Apple (Metal) build: ds4_cuda.cu is not linked, but the engine compiles
 * code referencing some multi-GPU CLI symbols inside dead multi-tier branches.
 * Provide stubs so the linker is happy. CPU-only (DS4_NO_GPU) builds
 * never reach the multi-tier branches and never include ds4_gpu.h, so
 * we cannot reference ds4_tensor_range there — those stubs are guarded
 * by !DS4_NO_GPU below. */
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
/* Decode-island graph capture is CUDA-only; Metal decodes eagerly. */
int ds4_gpu_decode_graphs_supported(void) { return 0; }
int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key) { (void)key; return -1; }
int ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key) { (void)key; return -1; }
void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key) { (void)key; }
void ds4_gpu_decode_graphs_invalidate(void) {}
int ds4_gpu_set_current_device(int logical_tier) { (void)logical_tier; return -1; }
int ds4_gpu_set_current_device_fenced(int logical_tier) { (void)logical_tier; return -1; }
void ds4_gpu_enable_q8_dequant_gemm(void) {}
int ds4_gpu_tensor_copy_async(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t bytes) { (void)dst; (void)src; (void)bytes; return 0; }
int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor *dst,
                                     const ds4_gpu_tensor *src,
                                     uint64_t bytes) {
    return ds4_gpu_tensor_copy(dst, 0, src, 0, bytes);
}
int ds4_gpu_tensor_copy_xdev3_default_dst(
        ds4_gpu_tensor *dst0, const ds4_gpu_tensor *src0, uint64_t bytes0,
        ds4_gpu_tensor *dst1, const ds4_gpu_tensor *src1, uint64_t bytes1,
        ds4_gpu_tensor *dst2, const ds4_gpu_tensor *src2, uint64_t bytes2) {
    return (bytes0 == 0 || ds4_gpu_tensor_copy(dst0, 0, src0, 0, bytes0)) &&
           (bytes1 == 0 || ds4_gpu_tensor_copy(dst1, 0, src1, 0, bytes1)) &&
           (bytes2 == 0 || ds4_gpu_tensor_copy(dst2, 0, src2, 0, bytes2));
}
int ds4_gpu_register_model_map_no_copy(const void *model_map, uint64_t model_size) {
    (void)model_map; (void)model_size; return 0;
}
int ds4_gpu_register_support_map(const void *map, uint64_t size, uint64_t bias) {
    (void)map; (void)size; (void)bias; return 1;
}
int ds4_gpu_device_cache_support_tensors(int device_id,
                                          int exec_device_id,
                                          const ds4_tensor_range *ranges,
                                          int n_ranges,
                                          int main_model) {
    (void)device_id; (void)exec_device_id; (void)ranges;
    (void)n_ranges; (void)main_model; return 0;
}
uint64_t ds4_gpu_tier_free_vram(int logical_tier) {
    (void)logical_tier; return 0;
}
int ds4_gpu_device_cache_tensors(int device_id,
                                  const ds4_tensor_range *ranges,
                                  int n_ranges) {
    (void)device_id; (void)ranges; (void)n_ranges; return 1;
}
int ds4_gpu_init_multi(const ds4_gpu_config *cfg) { (void)cfg; return -1; }
int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int device_id, uint64_t bytes) {
    (void)t; (void)device_id; (void)bytes; return -1;
}
ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int tier, uint64_t bytes) {
    /* Metal / CPU build has no CUDA multi-tier — short-circuit tier 0 to
     * the legacy single-device allocator so the per-tier graph
     * allocation path (e.g. g->output_pre_by_tier[head_tier]) keeps
     * working byte-equivalent to pre-multi-tier Metal. Without this,
     * every multi-tier-aware allocation in metal_graph_alloc_raw_cap
     * returns NULL, the validation chain fails, and session_create
     * silently returns 1. */
    if (tier == 0) return ds4_gpu_tensor_alloc(bytes);
    return NULL;
}
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int tier, uint64_t bytes) {
    if (tier == 0) return ds4_gpu_tensor_alloc_managed(bytes);
    return NULL;
}
void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *t) { (void)t; }
int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst,
                              const ds4_gpu_tensor *src,
                              uint64_t bytes) {
    (void)dst; (void)src; (void)bytes; return -1;
}
int ds4_gpu_tensor_copy_xdev3(ds4_gpu_tensor       *dst0,
                              const ds4_gpu_tensor *src0,
                              uint64_t              bytes0,
                              ds4_gpu_tensor       *dst1,
                              const ds4_gpu_tensor *src1,
                              uint64_t              bytes1,
                              ds4_gpu_tensor       *dst2,
                              const ds4_gpu_tensor *src2,
                              uint64_t              bytes2) {
    (void)dst0; (void)src0; (void)bytes0;
    (void)dst1; (void)src1; (void)bytes1;
    (void)dst2; (void)src2; (void)bytes2;
    return -1;
}
int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *dst,
                                      const ds4_gpu_tensor *src,
                                      uint64_t bytes) {
    (void)dst; (void)src; (void)bytes; return -1;
}
int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *src, int dst_tier) {
    (void)src; (void)dst_tier; return -1;
}
int ds4_gpu_moe_handoff_pack_tensor(
        ds4_gpu_tensor       *packed,
        const ds4_gpu_tensor *ffn_norm,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_embd,
        uint32_t              n_expert) {
    (void)packed; (void)ffn_norm; (void)selected; (void)weights;
    (void)n_embd; (void)n_expert; return -1;
}
int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t weight0_offset, uint64_t weight1_offset,
        uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
        const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_q8_0_pair_tensor(
            out0, out1, model_map, model_size,
            weight0_offset, weight1_offset,
            in_dim, out0_dim, out1_dim, x, n_rows);
}
int ds4_gpu_matmul_f16_router_rows_exact_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_f16_tensor(
            out, model_map, model_size, weight_offset,
            4096u, 256u, x, n_rows);
}
int ds4_gpu_q8_cache_suppressed(void) { return 0; }
void ds4_gpu_set_q8_cache_suppressed(int suppressed) { (void)suppressed; }
int ds4_gpu_set_decode_fast_attention(int enabled) { (void)enabled; return 0; }
int ds4_gpu_set_decode_score_vec4(int enabled) { (void)enabled; return 0; }
int ds4_gpu_indexer_top2_value_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                index_offset) {
    (void)selected; (void)values; (void)scores; (void)n_comp;
    (void)n_tokens; (void)index_offset;
    return 0;
}
int ds4_gpu_matmul_q8_0_top1_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint32_t                index_offset) {
    (void)selected; (void)values; (void)model_map; (void)model_size;
    (void)weight_offset; (void)in_dim; (void)out_dim; (void)x;
    (void)index_offset;
    return 0;
}
int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
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
        float                   eps) {
    (void)q_out; (void)q; (void)model_map; (void)model_size;
    (void)q_weight_offset; (void)q_n; (void)kv_out; (void)kv;
    (void)kv_weight_offset; (void)kv_n; (void)rows; (void)kv_n_head;
    (void)kv_head_dim; (void)n_rot; (void)pos0; (void)n_ctx_orig;
    (void)inverse; (void)freq_base; (void)freq_scale; (void)ext_factor;
    (void)attn_factor; (void)beta_fast; (void)beta_slow; (void)eps;
    return 0;
}
int ds4_gpu_attention_decode_heads_rope_tensor(
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
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        int                    *fused_inv_rope) {
    (void)n_rot; (void)pos0; (void)n_ctx_orig; (void)freq_base;
    (void)freq_scale; (void)ext_factor; (void)attn_factor;
    (void)beta_fast; (void)beta_slow;
    if (fused_inv_rope) *fused_inv_rope = 0;
    return ds4_gpu_attention_decode_heads_tensor(
            heads, model_map, model_size, sinks_offset, q, raw_kv,
            n_raw, raw_cap, raw_start, comp_kv, comp_kv_f16, n_comp,
            comp_mask, use_mask, n_head, head_dim);
}
int ds4_gpu_tensor_device(const ds4_gpu_tensor *t) { (void)t; return -1; }
ds4_gpu_ctx g_gpu[DS4_MAX_GPUS];
int         g_n_gpus = 0;
int         g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS];
#endif

#if defined(DS4_NO_GPU)
/* CPU-only build: even though no multi-tier code is reached, the engine
 * struct still embeds ds4_gpu_config and the global decls in
 * ds4_gpu_mgpu.h need matching definitions. We do not stub the
 * function symbols here because no caller references them in CPU
 * builds (every callsite is inside !DS4_NO_GPU). */
ds4_gpu_ctx g_gpu[DS4_MAX_GPUS];
int         g_n_gpus = 0;
int         g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS];
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DS4_NEG_INF (-1.0e30f)
#define DS4_POS_INF ( 1.0e30f)
#define DS4_DEFAULT_RMS_EPS ( 1.0e-6f)
#define DS4_DEFAULT_HC_EPS  ( 1.0e-6f)
#define DS4_DEFAULT_SWIGLU_CLAMP_EXP    (10.0f)
#define DS4_DEFAULT_ROPE_FREQ_BASE      (10000.0f)
#define DS4_DEFAULT_ROPE_SCALE_FACTOR   (16.0f)
#define DS4_DEFAULT_ROPE_YARN_BETA_FAST (32.0f)
#define DS4_DEFAULT_ROPE_YARN_BETA_SLOW (1.0f)
#define DS4_DEFAULT_COMPRESS_ROPE_FREQ_BASE (160000.0f)
#define DS4_DEFAULT_ROPE_ORIG_CTX       UINT64_C(65536)

static const char DS4_REASONING_EFFORT_MAX_PREFIX[] =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

/* DeepSeek recommends Think Max only with at least a 384K-token context window.
 * Below that size we keep ordinary thinking to avoid injecting a prompt that
 * asks for a reasoning budget the allocated context is not meant to hold. */
#define DS4_THINK_MAX_MIN_CONTEXT 393216u

static bool ds4_backend_uses_graph(ds4_backend backend) {
    return backend == DS4_BACKEND_METAL || backend == DS4_BACKEND_CUDA;
}

static bool ds4_backend_supports_ssd_streaming(ds4_backend backend) {
    if (backend == DS4_BACKEND_METAL) return true;
    if (backend == DS4_BACKEND_CUDA) {
#if defined(DS4_ROCM_BUILD) || (!defined(DS4_NO_GPU) && !defined(__APPLE__))
        return true;
#else
        return false;
#endif
    }
    return false;
}

static bool ds4_backend_supports_streaming_auto_cache(ds4_backend backend) {
    if (backend == DS4_BACKEND_METAL) return true;
#ifdef DS4_ROCM_BUILD
    if (backend == DS4_BACKEND_CUDA) return true;
#else
    (void)backend;
#endif
    return false;
}

static bool ds4_backend_supports_glm_streaming_full_layers(ds4_backend backend) {
    if (backend == DS4_BACKEND_METAL) return true;
#ifdef DS4_ROCM_BUILD
    if (backend == DS4_BACKEND_CUDA) return true;
#else
    (void)backend;
#endif
    return false;
}

static bool glm_graph_env_present(const char *rocm_name, const char *metal_name) {
#ifdef DS4_ROCM_BUILD
    if (rocm_name && getenv(rocm_name) != NULL) return true;
#else
    (void)rocm_name;
#endif
    return metal_name && getenv(metal_name) != NULL;
}

static const char *glm_graph_env_value(const char *rocm_name,
                                       const char *metal_name) {
#ifdef DS4_ROCM_BUILD
    const char *rocm_env = rocm_name ? getenv(rocm_name) : NULL;
    if (rocm_env && rocm_env[0]) return rocm_env;
#else
    (void)rocm_name;
#endif
    const char *metal_env = metal_name ? getenv(metal_name) : NULL;
    return (metal_env && metal_env[0]) ? metal_env : NULL;
}

/* =========================================================================
 * Model Shape Profiles.
 * =========================================================================
 *
 * The weight binder and metadata validator select one of the known model
 * profiles below.  Arrays reserve the maximum Pro dimensions; hot loops read
 * the active profile after GGUF validation.
 */

enum {
    DS4_MAX_LAYER            = 79,
    DS4_MAX_EMBD             = 7168,
    DS4_MAX_VOCAB            = 154880,
    DS4_MAX_HEAD             = 128,
    DS4_MAX_HEAD_KV          = 1,
    DS4_MAX_HEAD_DIM         = 576,
    DS4_MAX_VALUE_DIM        = 512,
    DS4_MAX_ROT              = 64,
    DS4_MAX_OUT_GROUP        = 16,
    DS4_MAX_LORA_Q           = 2048,
    DS4_MAX_LORA_O           = 1024,
    DS4_MAX_EXPERT           = 384,
    DS4_MAX_EXPERT_USED      = 8,
    DS4_MAX_EXPERT_SHARED    = 1,
    DS4_MAX_FF_EXP           = 3072,
    DS4_MAX_HASH_LAYER       = 3,
    DS4_MAX_SWA              = 128,
    DS4_MAX_INDEXER_HEAD     = 64,
    DS4_MAX_INDEXER_HEAD_DIM = 128,
    DS4_MAX_INDEXER_TOP_K    = 2048,
    DS4_MAX_HC               = 4,
    DS4_MAX_HC_SINKHORN_ITER = 20,
};

typedef enum {
    DS4_MODEL_FAMILY_DEEPSEEK4 = 0,
    DS4_MODEL_FAMILY_GLM_DSA   = 1,
} ds4_model_family;

typedef enum {
    DS4_VARIANT_FLASH = 0,
    DS4_VARIANT_PRO   = 1,
    DS4_VARIANT_GLM52 = 2,
} ds4_variant;

typedef struct {
    const char *name;
    ds4_model_family family;
    ds4_variant variant;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_head_dim;
    uint32_t n_value_dim;
    uint32_t n_rot;
    uint32_t n_out_group;
    uint32_t n_lora_q;
    uint32_t n_lora_o;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_expert_shared;
    uint32_t n_ff_exp;
    uint32_t n_ff_dense;
    uint32_t n_hash_layer;
    uint32_t n_swa;
    uint32_t n_indexer_head;
    uint32_t n_indexer_head_dim;
    uint32_t n_indexer_top_k;
    uint32_t n_hc;
    uint32_t n_hc_sinkhorn_iter;
    uint32_t n_nextn_predict;
    uint32_t n_leading_dense;
    uint32_t n_kv_lora;
    uint32_t n_key_mla;
    uint32_t n_value_mla;
    float rms_eps;
    float hc_eps;
    float expert_weight_scale;
    float swiglu_clamp_exp;
    float rope_freq_base;
    float rope_scale_factor;
    float rope_yarn_beta_fast;
    float rope_yarn_beta_slow;
    float compress_rope_freq_base;
    uint64_t rope_orig_ctx;
} ds4_shape;

static const ds4_shape DS4_SHAPE_FLASH = {
    .name = "DeepSeek V4 Flash",
    .family = DS4_MODEL_FAMILY_DEEPSEEK4,
    .variant = DS4_VARIANT_FLASH,
    .n_layer = 43,
    .n_embd = 4096,
    .n_vocab = 129280,
    .n_head = 64,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 8,
    .n_lora_q = 1024,
    .n_lora_o = 1024,
    .n_expert = 256,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_ff_exp = 2048,
    .n_hash_layer = 3,
    .n_swa = 128,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 512,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .rms_eps = DS4_DEFAULT_RMS_EPS,
    .hc_eps = DS4_DEFAULT_HC_EPS,
    .expert_weight_scale = 1.5f,
    .swiglu_clamp_exp = DS4_DEFAULT_SWIGLU_CLAMP_EXP,
    .rope_freq_base = DS4_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = DS4_DEFAULT_ROPE_SCALE_FACTOR,
    .rope_yarn_beta_fast = DS4_DEFAULT_ROPE_YARN_BETA_FAST,
    .rope_yarn_beta_slow = DS4_DEFAULT_ROPE_YARN_BETA_SLOW,
    .compress_rope_freq_base = DS4_DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    .rope_orig_ctx = DS4_DEFAULT_ROPE_ORIG_CTX,
};

static const ds4_shape DS4_SHAPE_PRO = {
    .name = "DeepSeek V4 Pro",
    .family = DS4_MODEL_FAMILY_DEEPSEEK4,
    .variant = DS4_VARIANT_PRO,
    .n_layer = 61,
    .n_embd = 7168,
    .n_vocab = 129280,
    .n_head = 128,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 16,
    .n_lora_q = 1536,
    .n_lora_o = 1024,
    .n_expert = 384,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_ff_exp = 3072,
    .n_hash_layer = 3,
    .n_swa = 128,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 1024,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .rms_eps = DS4_DEFAULT_RMS_EPS,
    .hc_eps = DS4_DEFAULT_HC_EPS,
    .expert_weight_scale = 2.5f,
    .swiglu_clamp_exp = DS4_DEFAULT_SWIGLU_CLAMP_EXP,
    .rope_freq_base = DS4_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = DS4_DEFAULT_ROPE_SCALE_FACTOR,
    .rope_yarn_beta_fast = DS4_DEFAULT_ROPE_YARN_BETA_FAST,
    .rope_yarn_beta_slow = DS4_DEFAULT_ROPE_YARN_BETA_SLOW,
    .compress_rope_freq_base = DS4_DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    .rope_orig_ctx = DS4_DEFAULT_ROPE_ORIG_CTX,
};

static const ds4_shape DS4_SHAPE_GLM52 = {
    .name = "GLM 5.2",
    .family = DS4_MODEL_FAMILY_GLM_DSA,
    .variant = DS4_VARIANT_GLM52,
    .n_layer = 79,
    .n_embd = 6144,
    .n_vocab = 154880,
    .n_head = 64,
    .n_head_kv = 1,
    .n_head_dim = 576,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 0,
    .n_lora_q = 2048,
    .n_lora_o = 0,
    .n_expert = 256,
    .n_expert_used = 8,
    .n_expert_shared = 1,
    .n_ff_exp = 2048,
    .n_ff_dense = 12288,
    .n_hash_layer = 0,
    .n_swa = 0,
    .n_indexer_head = 32,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 2048,
    .n_hc = 0,
    .n_hc_sinkhorn_iter = 0,
    .n_nextn_predict = 1,
    .n_leading_dense = 3,
    .n_kv_lora = 512,
    .n_key_mla = 256,
    .n_value_mla = 256,
    .rms_eps = 1.0e-5f,
    .hc_eps = 0.0f,
    .expert_weight_scale = 2.5f,
    .swiglu_clamp_exp = 0.0f,
    .rope_freq_base = 8000000.0f,
    .rope_scale_factor = 1.0f,
    .rope_yarn_beta_fast = 0.0f,
    .rope_yarn_beta_slow = 0.0f,
    .compress_rope_freq_base = 0.0f,
    .rope_orig_ctx = 1048576,
};

static ds4_shape g_ds4_shape = {
    .name = "DeepSeek V4 Flash",
    .family = DS4_MODEL_FAMILY_DEEPSEEK4,
    .variant = DS4_VARIANT_FLASH,
    .n_layer = 43,
    .n_embd = 4096,
    .n_vocab = 129280,
    .n_head = 64,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 8,
    .n_lora_q = 1024,
    .n_lora_o = 1024,
    .n_expert = 256,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_ff_exp = 2048,
    .n_hash_layer = 3,
    .n_swa = 128,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 512,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .rms_eps = DS4_DEFAULT_RMS_EPS,
    .hc_eps = DS4_DEFAULT_HC_EPS,
    .expert_weight_scale = 1.5f,
    .swiglu_clamp_exp = DS4_DEFAULT_SWIGLU_CLAMP_EXP,
    .rope_freq_base = DS4_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = DS4_DEFAULT_ROPE_SCALE_FACTOR,
    .rope_yarn_beta_fast = DS4_DEFAULT_ROPE_YARN_BETA_FAST,
    .rope_yarn_beta_slow = DS4_DEFAULT_ROPE_YARN_BETA_SLOW,
    .compress_rope_freq_base = DS4_DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    .rope_orig_ctx = DS4_DEFAULT_ROPE_ORIG_CTX,
};

static uint32_t g_ds4_compress_ratios[DS4_MAX_LAYER] = {0};

#define DS4_MODEL_SHAPE_NAME          (g_ds4_shape.name)
#define DS4_MODEL_FAMILY              (g_ds4_shape.family)
#define DS4_MODEL_VARIANT             (g_ds4_shape.variant)
#define DS4_N_LAYER                   (g_ds4_shape.n_layer)
#define DS4_N_EMBD                    (g_ds4_shape.n_embd)
#define DS4_N_VOCAB                   (g_ds4_shape.n_vocab)
#define DS4_N_HEAD                    (g_ds4_shape.n_head)
#define DS4_N_HEAD_KV                 (g_ds4_shape.n_head_kv)
#define DS4_N_HEAD_DIM                (g_ds4_shape.n_head_dim)
#define DS4_N_VALUE_DIM               (g_ds4_shape.n_value_dim)
#define DS4_N_ROT                     (g_ds4_shape.n_rot)
#define DS4_N_OUT_GROUP               (g_ds4_shape.n_out_group)
#define DS4_N_LORA_Q                  (g_ds4_shape.n_lora_q)
#define DS4_N_LORA_O                  (g_ds4_shape.n_lora_o)
#define DS4_N_EXPERT                  (g_ds4_shape.n_expert)
#define DS4_N_EXPERT_USED             (g_ds4_shape.n_expert_used)
#define DS4_N_EXPERT_SHARED           (g_ds4_shape.n_expert_shared)
#define DS4_N_FF_EXP                  (g_ds4_shape.n_ff_exp)
#define DS4_N_FF_DENSE                (g_ds4_shape.n_ff_dense)
#define DS4_N_HASH_LAYER              (g_ds4_shape.n_hash_layer)
#define DS4_N_SWA                     (g_ds4_shape.n_swa)
#define DS4_N_INDEXER_HEAD            (g_ds4_shape.n_indexer_head)
#define DS4_N_INDEXER_HEAD_DIM        (g_ds4_shape.n_indexer_head_dim)
#define DS4_N_INDEXER_TOP_K           (g_ds4_shape.n_indexer_top_k)
#define DS4_N_HC                      (g_ds4_shape.n_hc)
#define DS4_N_HC_SINKHORN_ITER        (g_ds4_shape.n_hc_sinkhorn_iter)
#define DS4_N_NEXTN_PREDICT           (g_ds4_shape.n_nextn_predict)
#define DS4_N_LEADING_DENSE           (g_ds4_shape.n_leading_dense)
#define DS4_N_KV_LORA                 (g_ds4_shape.n_kv_lora)
#define DS4_N_KEY_MLA                 (g_ds4_shape.n_key_mla)
#define DS4_N_VALUE_MLA               (g_ds4_shape.n_value_mla)
#define DS4_RMS_EPS                   (g_ds4_shape.rms_eps)
#define DS4_HC_EPS                    (g_ds4_shape.hc_eps)
#define DS4_EXPERT_WEIGHT_SCALE       (g_ds4_shape.expert_weight_scale)
#define DS4_SWIGLU_CLAMP_EXP          (g_ds4_shape.swiglu_clamp_exp)
#define DS4_ROPE_FREQ_BASE            (g_ds4_shape.rope_freq_base)
#define DS4_ROPE_SCALE_FACTOR         (g_ds4_shape.rope_scale_factor)
#define DS4_ROPE_YARN_BETA_FAST       (g_ds4_shape.rope_yarn_beta_fast)
#define DS4_ROPE_YARN_BETA_SLOW       (g_ds4_shape.rope_yarn_beta_slow)
#define DS4_COMPRESS_ROPE_FREQ_BASE   (g_ds4_shape.compress_rope_freq_base)
#define DS4_ROPE_ORIG_CTX             (g_ds4_shape.rope_orig_ctx)

static int g_ds4_lock_fd = -1;

#if defined(__GNUC__) || defined(__clang__)
#define DS4_MAYBE_UNUSED __attribute__((unused))
#else
#define DS4_MAYBE_UNUSED
#endif

/* =========================================================================
 * GGUF Quant Block Formats.
 * =========================================================================
 *
 * These layouts and IQ2 tables match the GGUF quantized tensor format,
 * reduced to only the formats ds4.c currently reads or sizes:
 *   - Q2_K routed down experts
 *   - Q4_K routed experts in the high-memory variant
 *   - Q5_K/Q6_K GLM routed experts
 *   - IQ2_XXS routed gate/up experts
 *   - MXFP4 routed experts preserved from native checkpoints
 *   - Q8_K temporary activation blocks for dot products
 */
#define QK_K 256
#define QK_MXFP4 32

typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
} block_q4_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[QK_K / 8];
    uint8_t  qs[QK_K / 2];
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t  scales[QK_K / 16];
    uint16_t d;
} block_q6_K;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
} block_iq2_xxs;

typedef struct {
    uint8_t e;
    uint8_t qs[QK_MXFP4 / 2];
} block_mxfp4;

#define DS4_STATIC_ASSERT(name, cond) typedef char name[(cond) ? 1 : -1]
DS4_STATIC_ASSERT(ds4_block_q2_k_size, sizeof(block_q2_K) == 84);
DS4_STATIC_ASSERT(ds4_block_q4_k_size, sizeof(block_q4_K) == 144);
DS4_STATIC_ASSERT(ds4_block_q5_k_size, sizeof(block_q5_K) == 176);
DS4_STATIC_ASSERT(ds4_block_q6_k_size, sizeof(block_q6_K) == 210);
DS4_STATIC_ASSERT(ds4_block_q8_k_size, sizeof(block_q8_K) == 292);
DS4_STATIC_ASSERT(ds4_block_iq2_xxs_size, sizeof(block_iq2_xxs) == 66);
DS4_STATIC_ASSERT(ds4_block_mxfp4_size, sizeof(block_mxfp4) == 17);

typedef struct {
    uint32_t ctx_size;
    uint32_t comp_cap;
    uint32_t attn_score_cap;
    uint32_t q8_cap;

    float *plain;
    float *cur;
    float *next;

    float *attn_cur;
    float *attn_norm;
    float *attn_residual;
    float *q;
    float *qr;
    float *qr_norm;
    float *kv_raw;
    float *kv;
    float *heads;
    float *attn_low;
    float *attn_out;
    float *after_attn_hc;
    float *attn_score;

    float *comp;
    float *index_comp;
    float *comp_kv_cur;
    float *comp_sc_cur;
    float *comp_pooled;

    bool *index_allowed;
    float *index_q;
    float *index_weights;
    float *index_scores;

    float *ffn_cur;
    float *ffn_norm;
    float *ffn_moe;
    float *ffn_shared;
    float *ffn_out;
    float *shared_gate;
    float *shared_up;
    float *shared_mid;
    float *routed_mid_all;
    block_q8_K *routed_xq;
    block_q8_K *routed_midq;
    int8_t *routed_q8_xq;
    float *routed_q8_xscale;
    int8_t *routed_q8_midq;
    float *routed_q8_midscale;

    int8_t *q8_xq;
    float *q8_xscale;

    float *hc_flat;
    float *output_flat;
    float *output_pre;
    float *output_weights;
    float *output_embd;
    float *output_norm;
} ds4_cpu_decode_scratch;

static const uint8_t kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static int8_t iq2xxs_signed_grid[256][128][8];
static int8_t iq2xxs_signs[128][8];
static pthread_once_t iq2xxs_signed_grid_once = PTHREAD_ONCE_INIT;

static void iq2xxs_signed_grid_init(void) {
    for (uint32_t s = 0; s < 128; s++) {
        const uint8_t signs = ksigns_iq2xs[s];
        for (uint32_t j = 0; j < 8; j++) {
            iq2xxs_signs[s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -1 : 1);
        }
    }

    for (uint32_t g = 0; g < 256; g++) {
        const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + g);
        for (uint32_t s = 0; s < 128; s++) {
            const uint8_t signs = ksigns_iq2xs[s];
            for (uint32_t j = 0; j < 8; j++) {
                const int v = (int)grid[j];
                iq2xxs_signed_grid[g][s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -v : v);
            }
        }
    }
}

static inline DS4_MAYBE_UNUSED int32_t dot_iq2_pair_16(const int8_t *grid0, const int8_t *grid1, const int8_t *q8) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int8x16_t gv = vcombine_s8(vld1_s8(grid0), vld1_s8(grid1));
    const int32x4_t acc = vdotq_s32(vdupq_n_s32(0), gv, vld1q_s8(q8));
    return vaddvq_s32(acc);
#elif defined(__ARM_NEON)
    const int8x16_t gv = vcombine_s8(vld1_s8(grid0), vld1_s8(grid1));
    const int8x16_t qv = vld1q_s8(q8);
    const int16x8_t p0 = vmull_s8(vget_low_s8(gv), vget_low_s8(qv));
    const int16x8_t p1 = vmull_s8(vget_high_s8(gv), vget_high_s8(qv));
    return vaddvq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
#else
    int32_t sum = 0;
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid0[i] * (int32_t)q8[i];
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid1[i] * (int32_t)q8[8 + i];
    return sum;
#endif
}

static inline DS4_MAYBE_UNUSED int32_t dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t packed = vld1q_u8(q2);
    uint8x16_t shifted;
    switch (shift) {
    case 0: shifted = packed; break;
    case 2: shifted = vshrq_n_u8(packed, 2); break;
    case 4: shifted = vshrq_n_u8(packed, 4); break;
    default: shifted = vshrq_n_u8(packed, 6); break;
    }
    const uint8x16_t vals_u = vandq_u8(shifted, vdupq_n_u8(3));
    const int8x16_t vals = vreinterpretq_s8_u8(vals_u);
    const int8x16_t q8v = vld1q_s8(q8);
    const int32x4_t acc = vdotq_s32(vdupq_n_s32(0), q8v, vals);
    return vaddvq_s32(acc);
#elif defined(__ARM_NEON)
    uint8_t vals_tmp[16];
    for (uint32_t i = 0; i < 16; i++) vals_tmp[i] = (q2[i] >> shift) & 3;
    const int8x16_t vals = vreinterpretq_s8_u8(vld1q_u8(vals_tmp));
    const int8x16_t q8v = vld1q_s8(q8);
    const int16x8_t p0 = vmull_s8(vget_low_s8(q8v), vget_low_s8(vals));
    const int16x8_t p1 = vmull_s8(vget_high_s8(q8v), vget_high_s8(vals));
    const int32x4_t s0 = vpaddlq_s16(p0);
    const int32x4_t s1 = vpaddlq_s16(p1);
    return vaddvq_s32(vaddq_s32(s0, s1));
#else
    int32_t sum = 0;
    for (uint32_t i = 0; i < 16; i++) sum += (int32_t)q8[i] * (int32_t)((q2[i] >> shift) & 3);
    return sum;
#endif
}

/* =========================================================================
 * Shared Helpers, Allocation Guards, Threads, and Cursor Reads.
 * =========================================================================
 *
 * This section holds process-wide utilities used by all later stages:
 * fatal-error helpers, allocation wrappers, the persistent CPU worker pool,
 * and the small byte cursor used to parse GGUF metadata.
 */

#define DS4_GGUF_MAGIC 0x46554747u /* "GGUF", little endian. */
#define DS4_MAX_DIMS   8

typedef struct {
    const char *ptr;
    uint64_t len;
} ds4_str;

typedef ds4_tokens token_vec;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} ds4_cursor;

static void ds4_die(const char *msg) {
    fprintf(stderr, "ds4: %s\n", msg);
    exit(1);
}

/* Attention compression is read from GGUF metadata after validating that it
 * matches the exact layout expected for the loaded model shape. */
static uint32_t ds4_layer_compress_ratio(uint32_t il) {
    if (DS4_MODEL_FAMILY != DS4_MODEL_FAMILY_DEEPSEEK4) return 0;
    if (il >= DS4_N_LAYER) ds4_die("DeepSeek4 layer index is outside the loaded model layout");
    return g_ds4_compress_ratios[il];
}

static uint32_t ds4_expected_layer_compress_ratio(uint32_t il) {
    if (il >= DS4_N_LAYER) ds4_die("DeepSeek4 layer index is outside the loaded model layout");

    switch (DS4_MODEL_VARIANT) {
    case DS4_VARIANT_FLASH:
        if (il < 2) return 0;
        return (il & 1u) == 0 ? 4u : 128u;
    case DS4_VARIANT_PRO:
        if (il < 2) return 128u;
        return (il & 1u) == 0 ? 4u : 128u;
    default:
        ds4_die("unsupported DeepSeek4 model variant");
    }
    return 0;
}

static void ds4_die_errno(const char *what, const char *path) {
    fprintf(stderr, "ds4: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

static bool ds4_streq(ds4_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}

static bool ds4_str_starts_with(ds4_str s, const char *prefix) {
    size_t n = strlen(prefix);
    return s.len >= n && memcmp(s.ptr, prefix, n) == 0;
}

static bool ds4_str_contains(ds4_str s, const char *needle) {
    size_t n = strlen(needle);
    if (n == 0) return true;
    if (s.len < n) return false;
    for (uint64_t i = 0; i <= s.len - n; i++) {
        if (memcmp(s.ptr + i, needle, n) == 0) return true;
    }
    return false;
}

static bool ds4_str_eq(ds4_str a, ds4_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, a.len) == 0;
}

static uint64_t hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static bool g_alloc_guard_enabled;
static const char *g_alloc_guard_phase;

static void ds4_alloc_guard_begin(const char *phase) {
    g_alloc_guard_phase = phase;
    g_alloc_guard_enabled = true;
}

static void ds4_alloc_guard_end(void) {
    g_alloc_guard_enabled = false;
    g_alloc_guard_phase = NULL;
}

static void ds4_alloc_guard_check(const char *op, size_t size) {
    if (!g_alloc_guard_enabled) return;
    fprintf(stderr,
            "ds4: internal allocation during %s: %s(%zu). "
            "CPU decode is expected to reuse preallocated scratch buffers.\n",
            g_alloc_guard_phase ? g_alloc_guard_phase : "guarded phase",
            op,
            size);
    exit(1);
}

static void *xcalloc(size_t n, size_t size) {
    ds4_alloc_guard_check("calloc", n * size);
    void *p = calloc(n, size);
    if (!p) ds4_die("out of memory");
    return p;
}

static void *xmalloc(size_t size) {
    ds4_alloc_guard_check("malloc", size);
    void *p = malloc(size);
    if (!p) ds4_die("out of memory");
    return p;
}

static char *ds4_strdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static void *xrealloc(void *ptr, size_t size) {
    ds4_alloc_guard_check("realloc", size);
    void *p = realloc(ptr, size);
    if (!p) ds4_die("out of memory");
    return p;
}

static void *xmalloc_zeroed(size_t n, size_t size) {
    if (size != 0 && n > SIZE_MAX / size) ds4_die("allocation size overflow");
    const size_t total = n * size;
    void *p = xmalloc(total ? total : 1);
    /*
     * This is intentionally not calloc(). Large untouched calloc ranges may be
     * represented by the VM through shared zero-page bookkeeping. The CPU decode
     * KV cache grows one token at a time, so using calloc here can move thousands
     * of first-touch faults into generation. On Darwin we have observed this end
     * in a kernel cpt_mapcnt_inc overflow panic instead of a user-space error.
     *
     * Explicitly writing the zeroes while the cache is allocated keeps those VM
     * faults out of the token loop and gives the cache private resident pages.
     */
    memset(p, 0, total);
    return p;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

/* =========================================================================
 * Metal Routed Expert Locality Profiler.
 * =========================================================================
 *
 * SSD streaming decode needs routed-expert locality data before a cache policy
 * is meaningful. This profiler is intentionally Metal-decode-only for now:
 * it reads the tiny selected-expert and route-weight tensors after router
 * selection, records histograms, and simulates per-layer latest-N unique expert
 * caches without changing normal inference when disabled.
 */

enum { DS4_EXPERT_PROFILE_MAX_CAPS = 10 };

static const uint32_t ds4_expert_profile_cap_candidates[DS4_EXPERT_PROFILE_MAX_CAPS] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 384
};

typedef struct {
    int      expert_id;
    uint64_t count;
    double   weight;
} ds4_expert_profile_sort_entry;

typedef struct {
    bool active;
    char *path;
    char *hotlist_path;
    char model_name[64];
    uint32_t n_layer;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_caps;
    uint32_t caps[DS4_EXPERT_PROFILE_MAX_CAPS];
    uint64_t layer_records[DS4_MAX_LAYER];
    uint64_t hist[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    double weight_hist[DS4_MAX_LAYER][DS4_MAX_EXPERT];
    uint64_t cache_hits[DS4_EXPERT_PROFILE_MAX_CAPS][DS4_MAX_LAYER];
    double cache_weight_hits[DS4_EXPERT_PROFILE_MAX_CAPS][DS4_MAX_LAYER];
    int cache_entries[DS4_EXPERT_PROFILE_MAX_CAPS][DS4_MAX_LAYER][DS4_MAX_EXPERT];
    uint32_t cache_count[DS4_EXPERT_PROFILE_MAX_CAPS][DS4_MAX_LAYER];
    bool prev_valid[DS4_MAX_LAYER];
    uint32_t prev_pos[DS4_MAX_LAYER];
    int prev_selected[DS4_MAX_LAYER][DS4_MAX_EXPERT_USED];
    uint64_t adjacent_pairs[DS4_MAX_LAYER];
    double adjacent_overlap_sum[DS4_MAX_LAYER];
    double adjacent_jaccard_sum[DS4_MAX_LAYER];
    bool layer_is_hash[DS4_MAX_LAYER];
    uint64_t total_records;
    uint64_t total_selections;
    double total_weight;
} ds4_expert_profile;

static ds4_expert_profile g_expert_profile;

static int ds4_expert_profile_sort_cmp(const void *a, const void *b) {
    const ds4_expert_profile_sort_entry *ea = a;
    const ds4_expert_profile_sort_entry *eb = b;
    if (ea->count < eb->count) return 1;
    if (ea->count > eb->count) return -1;
    if (ea->weight < eb->weight) return 1;
    if (ea->weight > eb->weight) return -1;
    return ea->expert_id - eb->expert_id;
}

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint64_t count;
    double weight;
} ds4_expert_hotlist_entry;

static int ds4_expert_hotlist_sort_cmp(const void *a, const void *b) {
    const ds4_expert_hotlist_entry *ea = a;
    const ds4_expert_hotlist_entry *eb = b;
    if (ea->count < eb->count) return 1;
    if (ea->count > eb->count) return -1;
    if (ea->weight < eb->weight) return 1;
    if (ea->weight > eb->weight) return -1;
    if (ea->layer != eb->layer) return ea->layer < eb->layer ? -1 : 1;
    if (ea->expert != eb->expert) return ea->expert < eb->expert ? -1 : 1;
    return 0;
}

#include "ds4_streaming_hotlist.inc"
#include "ds4_streaming_hotlist_glm52.inc"

static void ds4_json_write_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc((char)*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

static void ds4_expert_profile_init(const char *path, const char *hotlist_path) {
    if ((!path || !path[0]) && (!hotlist_path || !hotlist_path[0])) return;
    if (g_expert_profile.active) {
        free(g_expert_profile.path);
        free(g_expert_profile.hotlist_path);
        memset(&g_expert_profile, 0, sizeof(g_expert_profile));
    }

    g_expert_profile.active = true;
    if (path && path[0]) g_expert_profile.path = ds4_strdup(path);
    if (hotlist_path && hotlist_path[0]) {
        g_expert_profile.hotlist_path = ds4_strdup(hotlist_path);
    }
    snprintf(g_expert_profile.model_name,
             sizeof(g_expert_profile.model_name),
             "%s",
             DS4_MODEL_SHAPE_NAME);
    g_expert_profile.n_layer = DS4_N_LAYER;
    g_expert_profile.n_expert = DS4_N_EXPERT;
    g_expert_profile.n_expert_used = DS4_N_EXPERT_USED;

    for (uint32_t i = 0; i < DS4_EXPERT_PROFILE_MAX_CAPS; i++) {
        const uint32_t cap = ds4_expert_profile_cap_candidates[i];
        if (cap == 0 || cap > DS4_N_EXPERT) continue;
        g_expert_profile.caps[g_expert_profile.n_caps++] = cap;
    }
    fprintf(stderr,
            "ds4: Metal expert locality profiler active (profile: %s, hotlist: %s)\n",
            g_expert_profile.path ? g_expert_profile.path : "disabled",
            g_expert_profile.hotlist_path ? g_expert_profile.hotlist_path : "disabled");
}

static void ds4_expert_profile_cache_use(
        uint32_t ci,
        uint32_t il,
        int      expert,
        float    weight) {
    ds4_expert_profile *p = &g_expert_profile;
    const uint32_t cap = p->caps[ci];
    int *entries = p->cache_entries[ci][il];
    uint32_t *count = &p->cache_count[ci][il];
    uint32_t found = UINT32_MAX;

    for (uint32_t i = 0; i < *count; i++) {
        if (entries[i] == expert) {
            found = i;
            break;
        }
    }

    if (found != UINT32_MAX) {
        p->cache_hits[ci][il]++;
        p->cache_weight_hits[ci][il] += weight;
        for (uint32_t i = found; i > 0; i--) entries[i] = entries[i - 1];
        entries[0] = expert;
        return;
    }

    uint32_t n = *count;
    if (n < cap) {
        (*count)++;
    } else if (n > 0) {
        n = cap - 1;
    }
    for (uint32_t i = n; i > 0; i--) entries[i] = entries[i - 1];
    entries[0] = expert;
}

static void ds4_expert_profile_record(
        uint32_t      il,
        uint32_t      pos,
        const int32_t selected[DS4_MAX_EXPERT_USED],
        const float   weights[DS4_MAX_EXPERT_USED],
        bool          is_hash) {
    ds4_expert_profile *p = &g_expert_profile;
    if (!p->active || il >= p->n_layer) return;

    p->layer_records[il]++;
    p->total_records++;
    if (is_hash) p->layer_is_hash[il] = true;

    if (p->prev_valid[il] && p->prev_pos[il] + 1u == pos) {
        uint32_t intersection = 0;
        for (uint32_t a = 0; a < p->n_expert_used; a++) {
            for (uint32_t b = 0; b < p->n_expert_used; b++) {
                if (p->prev_selected[il][a] == selected[b]) {
                    intersection++;
                    break;
                }
            }
        }
        const uint32_t union_size = 2u * p->n_expert_used - intersection;
        p->adjacent_pairs[il]++;
        p->adjacent_overlap_sum[il] +=
            (double)intersection / (double)p->n_expert_used;
        if (union_size != 0) {
            p->adjacent_jaccard_sum[il] +=
                (double)intersection / (double)union_size;
        }
    }

    p->prev_valid[il] = true;
    p->prev_pos[il] = pos;

    for (uint32_t slot = 0; slot < p->n_expert_used; slot++) {
        const int expert = selected[slot];
        const float weight = weights[slot];
        p->prev_selected[il][slot] = expert;
        if (expert < 0 || (uint32_t)expert >= p->n_expert) continue;
        p->hist[il][expert]++;
        p->weight_hist[il][expert] += weight;
        p->total_selections++;
        p->total_weight += weight;
        for (uint32_t ci = 0; ci < p->n_caps; ci++) {
            ds4_expert_profile_cache_use(ci, il, expert, weight);
        }
    }
}

static void ds4_expert_profile_write_cache_summary(FILE *fp) {
    ds4_expert_profile *p = &g_expert_profile;
    fputs("  \"cache_summary\": [\n", fp);
    for (uint32_t ci = 0; ci < p->n_caps; ci++) {
        uint64_t hits = 0;
        double weight_hits = 0.0;
        for (uint32_t il = 0; il < p->n_layer; il++) {
            hits += p->cache_hits[ci][il];
            weight_hits += p->cache_weight_hits[ci][il];
        }
        const double hit_rate = p->total_selections ?
            (double)hits / (double)p->total_selections : 0.0;
        const double weight_hit_rate = p->total_weight > 0.0 ?
            weight_hits / p->total_weight : 0.0;
        fprintf(fp,
                "    {\"n\":%u,\"hits\":%" PRIu64 ",\"selections\":%" PRIu64
                ",\"hit_rate\":%.6f,\"weighted_hit_rate\":%.6f}%s\n",
                p->caps[ci],
                hits,
                p->total_selections,
                hit_rate,
                weight_hit_rate,
                ci + 1u == p->n_caps ? "" : ",");
    }
    fputs("  ],\n", fp);
}

static void ds4_expert_profile_write_layer(FILE *fp, uint32_t il) {
    ds4_expert_profile *p = &g_expert_profile;
    ds4_expert_profile_sort_entry entries[DS4_MAX_EXPERT];
    uint32_t unique = 0;
    uint64_t selections = 0;
    double weight_total = 0.0;

    for (uint32_t e = 0; e < p->n_expert; e++) {
        const uint64_t count = p->hist[il][e];
        selections += count;
        weight_total += p->weight_hist[il][e];
        if (count == 0) continue;
        entries[unique++] = (ds4_expert_profile_sort_entry) {
            .expert_id = (int)e,
            .count = count,
            .weight = p->weight_hist[il][e],
        };
    }
    qsort(entries, unique, sizeof(entries[0]), ds4_expert_profile_sort_cmp);

    const double avg_overlap = p->adjacent_pairs[il] ?
        p->adjacent_overlap_sum[il] / (double)p->adjacent_pairs[il] : 0.0;
    const double avg_jaccard = p->adjacent_pairs[il] ?
        p->adjacent_jaccard_sum[il] / (double)p->adjacent_pairs[il] : 0.0;

    fprintf(fp,
            "    {\"layer\":%u,\"hash_router\":%s,\"records\":%" PRIu64
            ",\"selections\":%" PRIu64 ",\"unique_experts\":%u"
            ",\"avg_adjacent_overlap\":%.6f,\"avg_adjacent_jaccard\":%.6f,\n",
            il,
            p->layer_is_hash[il] ? "true" : "false",
            p->layer_records[il],
            selections,
            unique,
            avg_overlap,
            avg_jaccard);

    fputs("      \"top_experts\": [", fp);
    const uint32_t top_n = unique < 16u ? unique : 16u;
    for (uint32_t i = 0; i < top_n; i++) {
        const double pct = selections ?
            100.0 * (double)entries[i].count / (double)selections : 0.0;
        const double weight_pct = weight_total > 0.0 ?
            100.0 * entries[i].weight / weight_total : 0.0;
        if (i) fputc(',', fp);
        fprintf(fp,
                "{\"id\":%d,\"count\":%" PRIu64 ",\"pct\":%.4f"
                ",\"weight\":%.9g,\"weight_pct\":%.4f}",
                entries[i].expert_id,
                entries[i].count,
                pct,
                entries[i].weight,
                weight_pct);
    }
    fputs("],\n", fp);

    fputs("      \"cache\": [", fp);
    for (uint32_t ci = 0; ci < p->n_caps; ci++) {
        const uint64_t hits = p->cache_hits[ci][il];
        const double weight_hits = p->cache_weight_hits[ci][il];
        const double hit_rate = selections ?
            (double)hits / (double)selections : 0.0;
        const double weight_hit_rate = weight_total > 0.0 ?
            weight_hits / weight_total : 0.0;
        if (ci) fputc(',', fp);
        fprintf(fp,
                "{\"n\":%u,\"hits\":%" PRIu64 ",\"hit_rate\":%.6f"
                ",\"weighted_hit_rate\":%.6f}",
                p->caps[ci],
                hits,
                hit_rate,
                weight_hit_rate);
    }
    fputs("]}", fp);
}

static void ds4_expert_profile_write_hotlist_file(ds4_expert_profile *p) {
    if (!p || !p->hotlist_path || !p->hotlist_path[0]) return;
    const size_t cap = (size_t)p->n_layer * p->n_expert;
    ds4_expert_hotlist_entry *entries = xmalloc(cap * sizeof(entries[0]));
    size_t n = 0;
    for (uint32_t il = 0; il < p->n_layer; il++) {
        for (uint32_t expert = 0; expert < p->n_expert; expert++) {
            entries[n++] = (ds4_expert_hotlist_entry) {
                .layer = il,
                .expert = expert,
                .count = p->hist[il][expert],
                .weight = p->weight_hist[il][expert],
            };
        }
    }
    qsort(entries, n, sizeof(entries[0]), ds4_expert_hotlist_sort_cmp);

    FILE *fp = fopen(p->hotlist_path, "wb");
    if (!fp) {
        fprintf(stderr,
                "ds4: failed to open expert hotlist output %s: %s\n",
                p->hotlist_path,
                strerror(errno));
        free(entries);
        return;
    }
    fprintf(fp,
            "# ds4 expert hotlist v1\n"
            "# model %s\n"
            "# layers %u\n"
            "# experts %u\n"
            "# layer_records %" PRIu64 "\n"
            "# selections %" PRIu64 "\n"
            "# columns: layer expert hits weight\n",
            p->model_name,
            p->n_layer,
            p->n_expert,
            p->total_records,
            p->total_selections);
    for (size_t i = 0; i < n; i++) {
        fprintf(fp,
                "%u %u %" PRIu64 " %.17g\n",
                entries[i].layer,
                entries[i].expert,
                entries[i].count,
                entries[i].weight);
    }
    free(entries);

    if (fclose(fp) != 0) {
        fprintf(stderr,
                "ds4: failed to close expert hotlist output %s: %s\n",
                p->hotlist_path,
                strerror(errno));
    } else {
        fprintf(stderr,
                "ds4: wrote Metal expert hotlist to %s "
                "(%" PRIu64 " layer records, %" PRIu64 " selections)\n",
                p->hotlist_path,
                p->total_records,
                p->total_selections);
    }
}

static void ds4_expert_profile_close(void) {
    ds4_expert_profile *p = &g_expert_profile;
    if (!p->active) return;

    if (p->path) {
        FILE *fp = fopen(p->path, "wb");
        if (!fp) {
            fprintf(stderr,
                    "ds4: failed to open expert profile output %s: %s\n",
                    p->path,
                    strerror(errno));
        } else {

            fputs("{\n", fp);
            fputs("  \"source\": \"ds4 Metal expert locality profile\",\n", fp);
            fputs("  \"model\": ", fp);
            ds4_json_write_string(fp, p->model_name);
            fputs(",\n", fp);
            fprintf(fp,
                    "  \"layers\": %u,\n"
                    "  \"experts\": %u,\n"
                    "  \"expert_used\": %u,\n"
                    "  \"layer_records\": %" PRIu64 ",\n"
                    "  \"selections\": %" PRIu64 ",\n",
                    p->n_layer,
                    p->n_expert,
                    p->n_expert_used,
                    p->total_records,
                    p->total_selections);
            fputs("  \"cache_ns\": [", fp);
            for (uint32_t ci = 0; ci < p->n_caps; ci++) {
                if (ci) fputc(',', fp);
                fprintf(fp, "%u", p->caps[ci]);
            }
            fputs("],\n", fp);
            ds4_expert_profile_write_cache_summary(fp);
            fputs("  \"layers_detail\": [\n", fp);
            for (uint32_t il = 0; il < p->n_layer; il++) {
                ds4_expert_profile_write_layer(fp, il);
                fputs(il + 1u == p->n_layer ? "\n" : ",\n", fp);
            }
            fputs("  ]\n}\n", fp);

            if (fclose(fp) != 0) {
                fprintf(stderr,
                        "ds4: failed to close expert profile output %s: %s\n",
                        p->path,
                        strerror(errno));
            } else {
                fprintf(stderr,
                        "ds4: wrote Metal expert locality profile to %s "
                        "(%" PRIu64 " layer records, %" PRIu64 " selections)\n",
                        p->path,
                        p->total_records,
                        p->total_selections);
            }
        }
    }
    ds4_expert_profile_write_hotlist_file(p);

    free(p->path);
    free(p->hotlist_path);
    memset(p, 0, sizeof(*p));
}

static void sleep_sec(double sec) {
    if (sec <= 0.0 || !isfinite(sec)) return;
    struct timespec req;
    req.tv_sec = (time_t)sec;
    req.tv_nsec = (long)((sec - (double)req.tv_sec) * 1000000000.0);
    if (req.tv_nsec < 0) req.tv_nsec = 0;
    if (req.tv_nsec >= 1000000000L) {
        req.tv_sec++;
        req.tv_nsec -= 1000000000L;
    }
    /* Do not resume after EINTR: Ctrl+C should cut through throttling sleeps. */
    (void)nanosleep(&req, &req);
}

static const char *ds4_log_color_code(ds4_log_type type) {
    switch (type) {
    case DS4_LOG_PREFILL:
    case DS4_LOG_TIMING:
        return "\x1b[36m";
    case DS4_LOG_GENERATION:
    case DS4_LOG_OK:
        return "\x1b[32m";
    case DS4_LOG_KVCACHE:
        return "\x1b[33m";
    case DS4_LOG_TOOL:
        return "\x1b[90m";
    case DS4_LOG_WARNING:
        return "\x1b[38;5;208m";
    case DS4_LOG_ERROR:
        return "\x1b[31m";
    default:
        return "";
    }
}

bool ds4_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}

static void ds4_vlog(FILE *fp, ds4_log_type type, const char *fmt, va_list ap) {
    const bool colorize = type != DS4_LOG_DEFAULT && ds4_log_is_tty(fp);
    if (colorize) fputs(ds4_log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
}

void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    ds4_vlog(fp, type, fmt, ap);
    va_end(ap);
}

static bool write_f32_binary_file(const char *path, const float *data, uint64_t n) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open %s for writing: %s\n", path, strerror(errno));
        return false;
    }
    const size_t nw = fwrite(data, sizeof(float), (size_t)n, fp);
    const bool ok = nw == (size_t)n && fclose(fp) == 0;
    if (!ok) {
        fprintf(stderr, "ds4: failed to write %s\n", path);
        return false;
    }
    return true;
}

static bool read_f32_binary_file(const char *path, float *data, uint64_t n) {
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "ds4: failed to stat %s: %s\n", path, strerror(errno));
        return false;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size != n * sizeof(float)) {
        fprintf(stderr,
                "ds4: %s has size %llu bytes, expected %llu bytes\n",
                path,
                (unsigned long long)st.st_size,
                (unsigned long long)(n * sizeof(float)));
        return false;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open %s for reading: %s\n", path, strerror(errno));
        return false;
    }
    const size_t nr = fread(data, sizeof(float), (size_t)n, fp);
    const bool ok = nr == (size_t)n && fclose(fp) == 0;
    if (!ok) {
        fprintf(stderr, "ds4: failed to read %s\n", path);
        return false;
    }
    return true;
}

static bool cpu_directional_steering_enabled(
        const float *dirs,
        float        scale);

static void cpu_directional_steering_project_rows(
        float       *x,
        const float *dirs,
        uint32_t     il,
        uint32_t     rows,
        float        scale);

typedef void (*ds4_parallel_fn)(void *ctx, uint64_t row0, uint64_t row1);

#define DS4_MAX_THREADS 32

typedef struct {
    pthread_t threads[DS4_MAX_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t work_cond;
    pthread_cond_t done_cond;
    uint32_t n_threads;
    uint32_t n_workers;
    uint32_t generation;
    uint32_t done;
    bool initialized;
    bool shutdown;
    ds4_parallel_fn fn;
    void *ctx;
    uint64_t n_rows;
} ds4_thread_pool;

static ds4_thread_pool g_pool;
static __thread int g_parallel_depth;
static uint32_t g_requested_threads;

static void *ds4_worker_main(void *arg) {
    const uint32_t tid = (uint32_t)(uintptr_t)arg;
    uint32_t seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&g_pool.mutex);
        while (seen_generation == g_pool.generation && !g_pool.shutdown) {
            pthread_cond_wait(&g_pool.work_cond, &g_pool.mutex);
        }
        if (g_pool.shutdown) {
            pthread_mutex_unlock(&g_pool.mutex);
            return NULL;
        }

        seen_generation = g_pool.generation;
        ds4_parallel_fn fn = g_pool.fn;
        void *ctx = g_pool.ctx;
        const uint64_t n_rows = g_pool.n_rows;
        const uint32_t n_threads = g_pool.n_threads;
        pthread_mutex_unlock(&g_pool.mutex);

        const uint64_t rows_per_thread = (n_rows + n_threads - 1) / n_threads;
        const uint64_t row0 = (uint64_t)tid * rows_per_thread;
        uint64_t row1 = row0 + rows_per_thread;
        if (row1 > n_rows) row1 = n_rows;
        if (row0 < row1) {
            g_parallel_depth++;
            fn(ctx, row0, row1);
            g_parallel_depth--;
        }

        pthread_mutex_lock(&g_pool.mutex);
        g_pool.done++;
        if (g_pool.done == g_pool.n_workers) {
            pthread_cond_signal(&g_pool.done_cond);
        }
        pthread_mutex_unlock(&g_pool.mutex);
    }
}

/* Create the persistent CPU worker pool.  Decode reuses these threads instead
 * of creating pthreads in the token loop. */
static void ds4_threads_init(void) {
    if (g_pool.initialized) return;

    pthread_once(&iq2xxs_signed_grid_once, iq2xxs_signed_grid_init);

    uint32_t n_threads = 12;
    const long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (online_cpus > 0) {
        n_threads = online_cpus < 12 ? (uint32_t)online_cpus : 12;
    }

    const char *env = getenv("DS4_THREADS");
    if (env && env[0]) {
        long v = strtol(env, NULL, 10);
        if (v > 0) n_threads = (uint32_t)v;
    }
    if (g_requested_threads > 0) n_threads = g_requested_threads;
    if (n_threads > DS4_MAX_THREADS) n_threads = DS4_MAX_THREADS;
    if (n_threads == 0) n_threads = 1;

    pthread_mutex_init(&g_pool.mutex, NULL);
    pthread_cond_init(&g_pool.work_cond, NULL);
    pthread_cond_init(&g_pool.done_cond, NULL);
    g_pool.n_threads = n_threads;
    g_pool.n_workers = n_threads > 0 ? n_threads - 1 : 0;
    g_pool.generation = 0;
    g_pool.done = 0;
    g_pool.shutdown = false;
    g_pool.initialized = true;

    for (uint32_t i = 1; i < n_threads; i++) {
        if (pthread_create(&g_pool.threads[i], NULL, ds4_worker_main, (void *)(uintptr_t)i) != 0) {
            ds4_die("failed to create worker thread");
        }
    }
}

static void ds4_threads_shutdown(void) {
    if (!g_pool.initialized) return;

    pthread_mutex_lock(&g_pool.mutex);
    g_pool.shutdown = true;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.work_cond);
    pthread_mutex_unlock(&g_pool.mutex);

    for (uint32_t i = 1; i < g_pool.n_threads; i++) {
        pthread_join(g_pool.threads[i], NULL);
    }

    pthread_cond_destroy(&g_pool.done_cond);
    pthread_cond_destroy(&g_pool.work_cond);
    pthread_mutex_destroy(&g_pool.mutex);
    memset(&g_pool, 0, sizeof(g_pool));
}

/* Run a row-parallel CPU kernel, falling back to serial execution for small
 * jobs or nested calls where spawning more work would only add latency. */
static void ds4_parallel_for_min_rows(uint64_t n_rows, ds4_parallel_fn fn, void *ctx, uint64_t min_parallel_rows) {
    ds4_threads_init();

    if (g_parallel_depth > 0 || g_pool.n_threads <= 1 || n_rows < min_parallel_rows) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&g_pool.mutex);
    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.n_rows = n_rows;
    g_pool.done = 0;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.work_cond);

    const uint64_t rows_per_thread = (n_rows + g_pool.n_threads - 1) / g_pool.n_threads;
    uint64_t main_row1 = rows_per_thread;
    if (main_row1 > n_rows) main_row1 = n_rows;
    pthread_mutex_unlock(&g_pool.mutex);

    if (main_row1 > 0) {
        g_parallel_depth++;
        fn(ctx, 0, main_row1);
        g_parallel_depth--;
    }

    pthread_mutex_lock(&g_pool.mutex);
    while (g_pool.done < g_pool.n_workers) {
        pthread_cond_wait(&g_pool.done_cond, &g_pool.mutex);
    }
    pthread_mutex_unlock(&g_pool.mutex);
}

static void ds4_parallel_for(uint64_t n_rows, ds4_parallel_fn fn, void *ctx) {
    ds4_parallel_for_min_rows(n_rows, fn, ctx, 512);
}

static void cursor_error(ds4_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(ds4_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_read(ds4_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(ds4_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(ds4_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(ds4_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(ds4_cursor *c, ds4_str *s) {
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

