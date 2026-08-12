#ifndef DS4_ENGINE_INTERNAL_H
#define DS4_ENGINE_INTERNAL_H

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
#include <string.h>
#include <strings.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include "ds4.h"
#include "ds4_gpu_runtime.h"
#include "ds4_gpu.h"
#include "ds4x/ops/attention.h"
#include "ds4x/ops/cache.h"
#include "ds4x/ops/decode_graph.h"
#include "ds4x/ops/hyper_connection.h"
#include "ds4x/ops/indexer.h"
#include "ds4x/ops/moe.h"
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

extern const char DS4_REASONING_EFFORT_MAX_PREFIX[];

#define DS4_THINK_MAX_MIN_CONTEXT 393216u

bool ds4_graph_env_present(const char *name);

const char *ds4_graph_env_value(const char *name);



/* =========================================================================
 * Model Shape Profiles.
 * =========================================================================
 *
 * The weight binder and metadata validator accept one fixed Flash profile.
 */

enum {
    DS4_MAX_LAYER            = 43,
    DS4_MAX_EMBD             = 4096,
    DS4_MAX_VOCAB            = 129280,
    DS4_MAX_HEAD             = 64,
    DS4_MAX_HEAD_KV          = 1,
    DS4_MAX_HEAD_DIM         = 512,
    DS4_MAX_VALUE_DIM        = 512,
    DS4_MAX_ROT              = 64,
    DS4_MAX_OUT_GROUP        = 8,
    DS4_MAX_LORA_Q           = 1024,
    DS4_MAX_LORA_O           = 1024,
    DS4_MAX_EXPERT           = 256,
    DS4_MAX_EXPERT_USED      = 6,
    DS4_MAX_EXPERT_SHARED    = 1,
    DS4_MAX_FF_EXP           = 2048,
    DS4_MAX_HASH_LAYER       = 3,
    DS4_MAX_SWA              = 128,
    DS4_MAX_INDEXER_HEAD     = 64,
    DS4_MAX_INDEXER_HEAD_DIM = 128,
    DS4_MAX_INDEXER_TOP_K    = 512,
    DS4_MAX_HC               = 4,
    DS4_MAX_HC_SINKHORN_ITER = 20,
};



typedef enum {
    DS4_VARIANT_FLASH = 0,
} ds4_variant;



typedef struct {
    const char *name;
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
    uint32_t n_hash_layer;
    uint32_t n_swa;
    uint32_t n_indexer_head;
    uint32_t n_indexer_head_dim;
    uint32_t n_indexer_top_k;
    uint32_t n_hc;
    uint32_t n_hc_sinkhorn_iter;
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

extern const ds4_shape DS4_SHAPE_FLASH;


extern ds4_shape g_ds4_shape;


extern uint32_t g_ds4_compress_ratios[DS4_MAX_LAYER];

#define DS4_MODEL_SHAPE_NAME          (g_ds4_shape.name)
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
#define DS4_N_HASH_LAYER              (g_ds4_shape.n_hash_layer)
#define DS4_N_SWA                     (g_ds4_shape.n_swa)
#define DS4_N_INDEXER_HEAD            (g_ds4_shape.n_indexer_head)
#define DS4_N_INDEXER_HEAD_DIM        (g_ds4_shape.n_indexer_head_dim)
#define DS4_N_INDEXER_TOP_K           (g_ds4_shape.n_indexer_top_k)
#define DS4_N_HC                      (g_ds4_shape.n_hc)
#define DS4_N_HC_SINKHORN_ITER        (g_ds4_shape.n_hc_sinkhorn_iter)
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

extern int g_ds4_lock_fd;

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

void ds4_die(const char *msg);

/* Attention compression is read from GGUF metadata after validating that it
 * matches the exact layout expected for the loaded model shape. */
uint32_t ds4_layer_compress_ratio(uint32_t il);

uint32_t ds4_expected_layer_compress_ratio(uint32_t il);

void ds4_die_errno(const char *what, const char *path);

bool ds4_streq(ds4_str s, const char *z);

bool ds4_str_starts_with(ds4_str s, const char *prefix);

bool ds4_str_contains(ds4_str s, const char *needle);

bool ds4_str_eq(ds4_str a, ds4_str b);

uint64_t hash_bytes(const void *ptr, uint64_t len);

void *xcalloc(size_t n, size_t size);

void *xmalloc(size_t size);

char *ds4_strdup(const char *s);

void *xrealloc(void *ptr, size_t size);

double now_sec(void);

void sleep_sec(double sec);

bool ds4_log_is_tty(FILE *fp);

void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);

bool write_f32_binary_file(const char *path, const float *data, uint64_t n);

bool read_f32_binary_file(const char *path, float *data, uint64_t n);



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

extern ds4_thread_pool g_pool;


extern uint32_t g_requested_threads;


/* Create the persistent CPU worker pool.  Decode reuses these threads instead
 * of creating pthreads in the token loop. */
void ds4_threads_init(void);

void ds4_threads_shutdown(void);

void ds4_parallel_for(uint64_t n_rows, ds4_parallel_fn fn, void *ctx);

void cursor_error(ds4_cursor *c, const char *msg);

bool cursor_read(ds4_cursor *c, void *dst, uint64_t n);

bool cursor_skip(ds4_cursor *c, uint64_t n);

bool cursor_u32(ds4_cursor *c, uint32_t *v);

bool cursor_u64(ds4_cursor *c, uint64_t *v);

bool cursor_string(ds4_cursor *c, ds4_str *s);

uint64_t align_up(uint64_t value, uint64_t alignment);

/* =========================================================================
 * GGUF Parsing and Model Mapping.
 * =========================================================================
 *
 * The loader maps the model once, records metadata/tensor descriptors, and
 * leaves tensor bytes in place.  Inference code accesses weights by adding
 * tensor offsets to the mapping instead of copying the GGUF into private
 * structures.
 */

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};



typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

enum {
    DS4_TENSOR_F32      = 0,
    DS4_TENSOR_F16      = 1,
    DS4_TENSOR_Q4_0     = 2,
    DS4_TENSOR_Q8_0     = 8,
    DS4_TENSOR_Q2_K     = 10,
    DS4_TENSOR_Q4_K     = 12,
    DS4_TENSOR_Q8_K     = 15,
    DS4_TENSOR_IQ2_XXS  = 16,
    DS4_TENSOR_I32      = 26,
    DS4_TENSOR_MXFP4    = 39,
};



typedef struct {
    ds4_str key;
    uint32_t type;
    uint64_t value_pos;
} ds4_kv;



typedef struct {
    ds4_str name;
    uint32_t ndim;
    uint64_t dim[DS4_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} ds4_tensor;



typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;

    ds4_kv *kv;
    ds4_tensor *tensors;
} ds4_model;

const gguf_type_info *tensor_type(uint32_t type);

const char *tensor_type_name(uint32_t type);

bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes);

ds4_cursor cursor_at(const ds4_model *m, uint64_t pos);

bool model_get_u32(const ds4_model *m, const char *key, uint32_t *out);

bool model_get_u64_compat(const ds4_model *m, const char *key, uint64_t *out);

bool model_get_f32_compat(const ds4_model *m, const char *key, float *out);

bool model_get_bool(const ds4_model *m, const char *key, bool *out);



typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} ds4_array_ref;

bool model_get_array(const ds4_model *m, const char *key, ds4_array_ref *out);

void model_close(ds4_model *m);

/* Open and map the GGUF once for CUDA weight-range resolution. */
void model_open(ds4_model *m, const char *path);
#define DS4_DSPARK_MAX_TARGET_LAYERS 8
#define DS4_DSPARK_MAX_STAGES 8
#define DS4_DSPARK_MAX_BLOCK_SIZE 16
#define DS4_SPEC_PREFIX_SLOTS 4


typedef struct {
    uint32_t stages;
    uint32_t block_size;
    uint32_t markov_rank;
    uint32_t noise_token_id;
    uint32_t target_layer_count;
    uint32_t target_layers[DS4_DSPARK_MAX_TARGET_LAYERS];
    bool has_metadata;
    bool has_main_proj;
    bool has_main_norm;
    bool has_markov_head;
    bool has_confidence_head;
    bool has_final_head;
    bool has_block_size;
    bool has_markov_rank;
    bool has_noise_token_id;
    bool has_target_layers;
} ds4_dspark_summary;



typedef enum {
    DS4_SUPPORT_NONE = 0,
    DS4_SUPPORT_DSPARK,
} ds4_support_kind;

void model_summary(const ds4_model *m);

ds4_tensor *model_find_tensor(const ds4_model *m, const char *name);

const char *support_kind_name(ds4_support_kind kind);

ds4_support_kind support_model_detect(
        const ds4_model *m,
        uint32_t        *stages_out,
        ds4_dspark_summary *summary_out);
typedef struct {
    uint64_t off;
    uint64_t end;
} accelerator_tensor_span;

bool accelerator_cache_model_tensors(ds4_backend backend,
                                            const ds4_model *m,
                                            const uint64_t *span_offsets,
                                            const uint64_t *span_sizes,
                                            uint32_t span_count);

/* Return the in-place tensor payload inside the mapped GGUF. */
const void *tensor_data(const ds4_model *m, const ds4_tensor *t);

/* Optional startup pass that touches tensor pages before timing generation. */
void model_warm_weights(const ds4_model *m);



/* =========================================================================
 * Scalar Conversion and Quantized Tensor Kernels.
 * =========================================================================
 *
 * These scalar helpers support validation and diagnostics. They implement only
 * the tensor formats present in the
 * DeepSeek V4 Flash GGUF: F16, F32, Q8_0, Q2_K, IQ2_XXS, and Q8_K activation
 * blocks used for expert dot products.
 */

static inline float f16_to_f32(uint16_t h) {
#if defined(__ARM_NEON)
    const float16x4_t hv = vreinterpret_f16_u16(vdup_n_u16(h));
    return vgetq_lane_f32(vcvt_f32_f16(hv), 0);
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ff;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

typedef struct {
    ds4_tensor *hc_attn_fn;
    ds4_tensor *hc_attn_scale;
    ds4_tensor *hc_attn_base;
    ds4_tensor *attn_norm;
    ds4_tensor *attn_q_a;
    ds4_tensor *attn_q_a_norm;
    ds4_tensor *attn_q_b;
    ds4_tensor *attn_kv;
    ds4_tensor *attn_kv_a_norm;
    ds4_tensor *attn_sinks;
    ds4_tensor *attn_output_a;
    ds4_tensor *attn_output_b;
    ds4_tensor *attn_compressor_ape;
    ds4_tensor *attn_compressor_kv;
    ds4_tensor *attn_compressor_gate;
    ds4_tensor *attn_compressor_norm;
    ds4_tensor *indexer_attn_q_b;
    ds4_tensor *indexer_proj;
    ds4_tensor *indexer_compressor_ape;
    ds4_tensor *indexer_compressor_kv;
    ds4_tensor *indexer_compressor_gate;
    ds4_tensor *indexer_compressor_norm;
    ds4_tensor *hc_ffn_fn;
    ds4_tensor *hc_ffn_scale;
    ds4_tensor *hc_ffn_base;
    ds4_tensor *ffn_norm;
    ds4_tensor *ffn_gate_tid2eid;
    ds4_tensor *ffn_gate_inp;
    ds4_tensor *ffn_exp_probs_b;
    ds4_tensor *ffn_gate_exps;
    ds4_tensor *ffn_up_exps;
    ds4_tensor *ffn_down_exps;
    ds4_tensor *ffn_gate_shexp;
    ds4_tensor *ffn_up_shexp;
    ds4_tensor *ffn_down_shexp;
} ds4_layer_weights;



typedef struct {
    ds4_tensor *token_embd;
    ds4_tensor *output_hc_base;
    ds4_tensor *output_hc_fn;
    ds4_tensor *output_hc_scale;
    ds4_tensor *output_norm;
    ds4_tensor *output;
    ds4_layer_weights layer[DS4_MAX_LAYER];
} ds4_weights;



typedef struct {
    ds4_tensor *main_proj;
    ds4_tensor *main_norm;
    ds4_tensor *norm;
    ds4_tensor *hc_head_base;
    ds4_tensor *hc_head_fn;
    ds4_tensor *hc_head_scale;
    ds4_tensor *markov_w1;
    ds4_tensor *markov_w2;
    ds4_tensor *confidence_proj;
    ds4_layer_weights block;
} ds4_dspark_stage_weights;



typedef struct {
    uint32_t n_stages;
    uint32_t block_size;
    uint32_t markov_rank;
    uint32_t noise_token_id;
    uint32_t target_layer_count;
    uint32_t target_layers[DS4_DSPARK_MAX_TARGET_LAYERS];
    uint32_t present_tensors;
    uint32_t missing_tensors;
    uint32_t invalid_tensors;
    uint32_t metadata_errors;
    bool has_block_size;
    bool has_markov_rank;
    bool has_noise_token_id;
    bool has_target_layers;
    ds4_dspark_stage_weights stage[DS4_DSPARK_MAX_STAGES];
} ds4_dspark_weights;

ds4_tensor *tensor_by_mtp_stage_suffix(
        const ds4_model *m,
        uint32_t         stage,
        const char      *suffix);

bool tensor_type_is_dense_quant(uint32_t type);

bool tensor_type_is_f16_or_q8_0(uint32_t type);

bool tensor_is_routed_expert_type(uint32_t type);

uint64_t routed_expert_row_bytes(const ds4_tensor *t);

uint64_t ds4_add_sat_u64(uint64_t a, uint64_t b);

double ds4_bytes_to_gib(uint64_t bytes);

bool weights_have_output_head(const ds4_weights *w);

bool weights_layer_has_required(const ds4_layer_weights *l, uint32_t il);

const ds4_layer_weights *weights_first_bound_layer(const ds4_weights *w);



typedef enum {
    DS4_DSPARK_LAYOUT_F32,
    DS4_DSPARK_LAYOUT_PLAIN,
    DS4_DSPARK_LAYOUT_DENSE,
    DS4_DSPARK_LAYOUT_ROUTED,
} ds4_dspark_layout_kind;

bool dspark_tensor_type_matches(uint32_t type,
                                       ds4_dspark_layout_kind kind);

void dspark_weights_validate_layout(ds4_dspark_weights *dw);

void config_validate_model(const ds4_model *m);

/* Bind tensor names once into the fixed DS4 layer layout.  This is the point
 * where stringly GGUF metadata becomes direct model-specific pointers. */
void weights_bind(
        ds4_weights     *w,
        const ds4_model *m,
        bool             load_slice,
        uint32_t         load_layer_start,
        uint32_t         load_layer_end,
        bool             require_output,
        bool             optional_output);



void dspark_weights_bind_optional(
        ds4_dspark_weights        *dw,
        const ds4_model           *m,
        const ds4_dspark_summary  *summary);

void weights_free(ds4_weights *w);

float layer_rope_freq_base(uint32_t il);

float layer_rope_freq_scale(uint32_t il);

float sigmoid_stable(float x);



static inline float dot_q8_0_row(
        const uint8_t *row,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks) {
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
        const int8_t *weights = (const int8_t *)(row + b * 34u + 2u);
        const uint64_t first = b * 32u;
        const uint64_t count = in_dim - first < 32u ? in_dim - first : 32u;
        int32_t dot = 0;
        for (uint64_t i = 0; i < count; i++) {
            dot += (int32_t)weights[i] * (int32_t)xq[first + i];
        }
        acc += f16_to_f32(scale_bits) * xscale[b] * (float)dot;
    }
    return acc;
}

void quantize_q8_0_activation(const float *x,
                                     int8_t *xq,
                                     float *scale,
                                     uint64_t n);

void matvec_any(float *out,
                       const ds4_model *model,
                       const ds4_tensor *weight,
                       const float *x);

uint32_t ds4_prefill_cap_for_prompt(int prompt_len,
                                           uint32_t requested_chunk);
int sample_argmax(const float *logits, uint32_t n_vocab);

/* =========================================================================
 * CUDA Reference Comparison Helpers.
 * =========================================================================
 *
 * These small scalar helpers are used only by CUDA diagnostics.
 */

float max_abs_diff(const float *a, const float *b, uint64_t n);

float rms_abs_diff(const float *a, const float *b, uint64_t n);
#define DS4_GPU_RAW_CACHE_SPARK 1
#define DS4_GPU_ATTN_COMP_CACHE_SPARK 1
#define DS4_GPU_INDEX_COMP_CACHE_SPARK 1
#define DS4_GPU_ATTN_COMP_CACHE_F16 0


/* =========================================================================
 * CUDA Graph State.
 * =========================================================================
 *
 * The CUDA executor owns one fixed set of tensors for single-token
 * decode and another for batched prefill.  The structure is DS4-specific:
 * tensor names follow the model stages rather than generic graph nodes.
 */

enum { DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS = 64 };



typedef struct {
    /* Explicit operator launch state. The GB10 runtime currently uses CUDA's
     * legacy default stream, but ownership lives with the session graph so a
     * future stream migration does not reintroduce process-global dispatch. */
    ds4x_runtime_context runtime;

    /* Class P kernel scratch buffers. DS4_MAX_GPUS is one, so active_tier
     * remains zero and the arrays preserve the tensor-accessor ABI without
     * implying a distributed execution path.
     *
     * Decode hidden-state buffers. A generated token enters as an embedding
     * in cur_hc and leaves as logits after all 43 layers update their
     * raw/compressed/indexer caches. The hc_pre / hc_post / hc_comb views
     * are derived from hc_split per tier (see metal_graph_alloc_raw_cap). */
    ds4_gpu_tensor *cur_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *flat_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *hc_mix_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *hc_split_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *hc_pre_by_tier[DS4_MAX_GPUS];   /* views of hc_split */
    ds4_gpu_tensor *hc_post_by_tier[DS4_MAX_GPUS];  /* views of hc_split */
    ds4_gpu_tensor *hc_comb_by_tier[DS4_MAX_GPUS];  /* views of hc_split */
    ds4_gpu_tensor *attn_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *attn_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *qr_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *qr_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *q_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *kv_raw_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *kv_by_tier[DS4_MAX_GPUS];
    int active_tier;

    /* Persistent KV state.  Raw KV is a sliding-window ring per layer.  Ratio-4
     * layers also keep an indexer-compressed cache; ratio-128 layers keep only
     * the attention-compressed cache.  The small state tensors are compressor
     * frontiers for the next compressed row, so they must be snapshotted with
     * the row counters whenever a checkpoint is saved or partially rewound. */
    ds4_gpu_tensor *layer_raw_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_comp_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_comp_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_index_state_score[DS4_MAX_LAYER];

    /* Speculative decoding scratch. DSpark may mutate graph state only
     * if the target verifier can either commit it or restore the saved
     * frontiers.  Prefix slots retain the intermediate frontiers of a tiny
     * verifier block so partial accepts do not need target-model replay. */
    ds4_gpu_tensor *spec_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_index_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_attn_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_attn_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_index_state_kv[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_prefix1_index_state_score[DS4_MAX_LAYER];
    ds4_gpu_tensor *spec_logits;
    uint32_t layer_n_comp[DS4_MAX_LAYER];
    uint32_t layer_n_index_comp[DS4_MAX_LAYER];
    uint32_t spec_prefix_n_comp[DS4_SPEC_PREFIX_SLOTS][DS4_MAX_LAYER];
    uint32_t spec_prefix_n_index_comp[DS4_SPEC_PREFIX_SLOTS][DS4_MAX_LAYER];
    bool spec_capture_prefixes;
    uint32_t raw_cap;
    /* Maximum compressed-row capacity across layers.  Shared work buffers use
     * this worst-case size because ratio-4 indexer layers can still reach it. */
    uint32_t comp_cap;
    /* Persistent compressed caches are per layer, so size them from the actual
     * layer compression ratio instead of pessimistically using the ratio-4 cap
     * for every ratio-128 layer. */
    uint32_t layer_comp_cap[DS4_MAX_LAYER];
    uint32_t attn_comp_stage_cap;

    /* Class P (per-layer work tensors). Each used tier has its
     * own replica. They are reused in place by every layer instead of
     * allocating a generic graph arena. This is why the code is verbose but
     * predictable: each pointer names an actual DS4 stage. */
    ds4_gpu_tensor *comp_kv_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *comp_sc_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *index_comp_kv_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *index_comp_sc_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *attn_comp_stage_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *index_comp_stage_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *indexer_q_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *indexer_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *indexer_scores_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *comp_mask_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *comp_selected_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *heads_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *attn_low_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *attn_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *after_attn_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *ffn_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *ffn_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *shared_gate_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *shared_up_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *shared_mid_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *shared_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *router_logits_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *router_probs_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *router_selected_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *router_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_gate_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_up_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_mid_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_down_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *routed_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *ffn_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *after_ffn_hc_by_tier[DS4_MAX_GPUS];
    /* Class H — output-head buffers and logits live on the
     * head tier only. head_tier is captured at metal_graph_alloc_raw_cap
     * time from placement[DS4_N_LAYER + 1] (or 0 in single-tier /
     * diagnostic paths). Non-head slots remain NULL. Readers go through
     * the metal_graph_logits / metal_graph_output_* accessors below. */
    ds4_gpu_tensor *output_pre_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *output_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *output_embd_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *output_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *logits_by_tier[DS4_MAX_GPUS];
    int head_tier;

    /* DSpark target features.  The proposer consumes mean-over-HC rows from
     * selected target layers; keeping them on-GPU avoids adding readbacks to
     * the target path. */
    ds4_gpu_tensor *dspark_hc_mean_weights;
    ds4_gpu_tensor *dspark_hc_mean_rows;
    ds4_gpu_tensor *dspark_target_hidden;
    ds4_gpu_tensor *dspark_target_hidden_batch;
    ds4_gpu_tensor *dspark_stage0_packed;
    ds4_gpu_tensor *dspark_stage0_proj;
    ds4_gpu_tensor *dspark_main_x;
    ds4_gpu_tensor *dspark_draft_tokens;
    ds4_gpu_tensor *dspark_draft_hc;
    ds4_gpu_tensor *dspark_target_hc;
    ds4_gpu_tensor *dspark_stage_input_hc;
    ds4_gpu_tensor *dspark_stage_output_hc;
    ds4_gpu_tensor *dspark_position_ids;
    ds4_gpu_tensor *dspark_raw_cache[DS4_DSPARK_MAX_STAGES];
    uint32_t dspark_cache_cap;
    uint32_t dspark_cache_start;
    uint32_t dspark_cache_token_start;
    uint32_t dspark_cache_len;
    uint32_t dspark_target_layer_count;
    uint32_t dspark_block_size;
    uint32_t dspark_target_layers[DS4_DSPARK_MAX_TARGET_LAYERS];
    uint32_t dspark_capture_mask;
    uint32_t dspark_capture_checkpoint_len;
    uint32_t dspark_capture_batch_mask;
    uint32_t dspark_capture_batch_start;
    uint32_t dspark_capture_batch_tokens;
    bool dspark_capture_valid;
    bool dspark_capture_batch_valid;
    int dspark_exec_tier;
    bool dspark_capture_enabled;

    uint32_t prefill_cap;
    uint32_t raw_window;
    uint32_t batch_token_offset;

    /* Batched prefill tensors.  Prefill is layer-major: a chunk of prompt
     * tokens moves through layer 0, then layer 1, and so on, updating the same
     * persistent caches used by decode.  Keeping this separate from decode
     * avoids a slow loop of one-token graph steps for long prompts. */
    /* Class E — embedding-tier-only prompt-token integer buffer.
     * Captured at metal_graph_alloc_raw_cap time from placement[0] (or 0 in
     * single-tier / diagnostic paths). Non-embedding slots stay NULL. Readers
     * go through metal_graph_prefill_tokens() below. */
    ds4_gpu_tensor *prefill_tokens_by_tier[DS4_MAX_GPUS];
    int emb_tier;
    /* Class P batch (chunked-prefill) scratch — per-tier
     * replicated. The cur/next pair is ping-ponged per layer step on the
     * layer's active tier; tier transitions copy the active buffer across
     * boundaries via ds4_gpu_tensor_copy_xdev (handled in B6). */
    ds4_gpu_tensor *batch_cur_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_next_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_flat_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_hc_mix_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_hc_split_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_attn_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_attn_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_qr_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_qr_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_q_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_kv_raw_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_kv_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_comp_kv_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_comp_sc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_indexer_q_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_indexer_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_heads_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_attn_low_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_attn_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_group_tmp_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_low_tmp_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_after_attn_hc_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_ffn_cur_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_ffn_norm_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_shared_gate_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_shared_up_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_shared_mid_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_shared_out_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_router_logits_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_router_probs_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_router_selected_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_router_weights_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_gate_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_up_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_mid_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_down_by_tier[DS4_MAX_GPUS];
    ds4_gpu_tensor *batch_routed_out_by_tier[DS4_MAX_GPUS];
    bool batch_routed_mid_is_f16;
    ds4_gpu_tensor *batch_ffn_out_by_tier[DS4_MAX_GPUS];
    bool owns_prefill_workspace;
    bool materialize_ffn_out;
    /* Class P (replicated per tier — this is
     * consumed in per-layer attn/FFN kernels, NOT embedding-only). Read-only
     * after init; replicate by writing the same host directions buffer to
     * every used tier's slot during session setup. */
    ds4_gpu_tensor *directional_steering_dirs_by_tier[DS4_MAX_GPUS];
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    bool cuda_q_norm_rope_fuse;
    bool cuda_qkv_kv_rope_fuse;
    bool cuda_qkv_pair;
    bool shared_gate_up_swiglu_fuse;
    bool decode_stage_profile;
    bool decode_index_stage_profile;
    bool output_stage_profile;
    uint32_t power_percent;
    double prefill_layer_avg_sec[DS4_MAX_LAYER];
    double decode_token_avg_sec;
    bool quality;
    bool mtp_enabled;
    /* FP16 prefill staging. */
    ds4_gpu_tensor *batch_q_half;
} ds4_gpu_graph;
#define DS4_GPU_PREFILL_WORKSPACE_FIELDS(X) \
    X(prefill_tokens)                       \
    X(batch_ffn_out)                        \
    X(batch_routed_out)                     \
    X(batch_routed_down)                    \
    X(batch_routed_mid)                     \
    X(batch_routed_up)                      \
    X(batch_routed_gate)                    \
    X(batch_router_weights)                 \
    X(batch_router_selected)                \
    X(batch_router_probs)                   \
    X(batch_router_logits)                  \
    X(batch_shared_out)                     \
    X(batch_shared_mid)                     \
    X(batch_shared_up)                      \
    X(batch_shared_gate)                    \
    X(batch_ffn_norm)                       \
    X(batch_ffn_cur)                        \
    X(batch_after_attn_hc)                  \
    X(batch_low_tmp)                        \
    X(batch_group_tmp)                      \
    X(batch_attn_out)                       \
    X(batch_attn_low)                       \
    X(batch_heads)                          \
    X(batch_indexer_weights)                \
    X(batch_indexer_q)                      \
    X(batch_comp_sc)                        \
    X(batch_comp_kv)                        \
    X(batch_kv)                             \
    X(batch_kv_raw)                         \
    X(batch_q)                              \
    X(batch_qr_norm)                        \
    X(batch_qr)                             \
    X(batch_attn_norm)                      \
    X(batch_attn_cur)                       \
    X(batch_hc_split)                       \
    X(batch_hc_mix)                         \
    X(batch_flat_hc)                        \
    X(batch_next_hc)                        \
    X(batch_cur_hc)


/* Class H accessors. All reader sites for the output-head
 * tensors and the final logits route through these inlines, which read the
 * head_tier slot captured at allocation time. Single-tier paths set
 * head_tier == 0 and the slot is byte-identical to the legacy
 * metal_graph_logits(g) / g->output_* pointers. Multi-tier paths set head_tier
 * to placement[DS4_N_LAYER + 1]; other tier slots remain NULL. */
static inline ds4_gpu_tensor *metal_graph_logits(const ds4_gpu_graph *g) {
    return g->logits_by_tier[g->head_tier];
}



static inline const ds4x_runtime_context *metal_graph_runtime(
        const ds4_gpu_graph *g) {
    return g ? &g->runtime : NULL;
}


static inline ds4_gpu_tensor *metal_graph_output_pre(const ds4_gpu_graph *g) {
    return g->output_pre_by_tier[g->head_tier];
}


static inline ds4_gpu_tensor *metal_graph_output_weights(const ds4_gpu_graph *g) {
    return g->output_weights_by_tier[g->head_tier];
}


static inline ds4_gpu_tensor *metal_graph_output_embd(const ds4_gpu_graph *g) {
    return g->output_embd_by_tier[g->head_tier];
}


static inline ds4_gpu_tensor *metal_graph_output_norm(const ds4_gpu_graph *g) {
    return g->output_norm_by_tier[g->head_tier];
}



/* Class E accessor. The prompt-token integer buffer is
 * consumed by the embedding kernel on the embedding tier only. Single-tier
 * paths set emb_tier == 0 (byte-equivalent to the legacy single-tier
 * pointer). Multi-tier paths set emb_tier = placement[0]. */
static inline ds4_gpu_tensor *metal_graph_prefill_tokens(const ds4_gpu_graph *g) {
    return g->prefill_tokens_by_tier[g->emb_tier];
}
#define DS4_GPU_GRAPH_CLASS_P_ACCESSOR(name) \
static inline ds4_gpu_tensor *metal_graph_##name(const ds4_gpu_graph *g) { \
    return g->name##_by_tier[g->active_tier]; \
}

DS4_GPU_GRAPH_CLASS_P_ACCESSOR(cur_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(flat_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_mix)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_split)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_pre)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_post)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(hc_comb)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(qr)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(qr_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(q)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(kv_raw)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(kv)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(comp_kv_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(comp_sc_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(index_comp_kv_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(index_comp_sc_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_comp_stage)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(index_comp_stage)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(indexer_q)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(indexer_weights)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(indexer_scores)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(comp_mask)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(comp_selected)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(heads)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_low)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(attn_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(after_attn_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(ffn_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(ffn_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(shared_gate)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(shared_up)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(shared_mid)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(shared_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(router_logits)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(router_probs)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(router_selected)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(router_weights)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_gate)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_up)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_mid)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_down)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(routed_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(ffn_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(after_ffn_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_cur_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_next_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_flat_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_hc_mix)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_hc_split)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_attn_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_attn_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_qr)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_qr_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_q)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_kv_raw)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_kv)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_comp_kv)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_comp_sc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_indexer_q)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_indexer_weights)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_heads)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_attn_low)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_attn_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_group_tmp)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_low_tmp)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_after_attn_hc)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_ffn_cur)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_ffn_norm)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_shared_gate)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_shared_up)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_shared_mid)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_shared_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_router_logits)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_router_probs)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_router_selected)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_router_weights)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_gate)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_up)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_mid)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_down)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_routed_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(batch_ffn_out)
DS4_GPU_GRAPH_CLASS_P_ACCESSOR(directional_steering_dirs)
/* --power N GPU duty-cycle throttling helpers. --power=100 is a no-op. */

bool graph_power_throttle_enabled(const ds4_gpu_graph *g);

void graph_power_note_prefill_layer(ds4_gpu_graph *g,
                                           uint32_t il,
                                           double elapsed_sec);

void graph_power_note_decode_token(ds4_gpu_graph *g, double elapsed_sec);

/* Release every tensor owned by the whole-model CUDA graph runtime. */
void metal_graph_free(ds4_gpu_graph *g);

bool metal_tensor_fill_f32(ds4_gpu_tensor *t, float v, uint64_t n);

/* =========================================================================
 * Directional Steering.
 * =========================================================================
 *
 * A steering file contains one normalized 4096-wide direction per layer.  When
 * enabled, the CUDA graph edits selected block outputs in-place:
 *
 *     y = y - scale * v * dot(v, y)
 *
 * Positive scales remove the represented direction from the activation.
 * Negative scales add it.  This is deliberately explicit and opt-in; with zero
 * scales, the release graph does not allocate the direction tensor and follows
 * the normal inference path.
 */

/* directional_steering_dirs is Class P — replicated per tier.
 * The same host directions buffer is written to every tier slot the engine's
 * placement uses, then the load buffer is freed. Read-only after init, so
 * the per-tier replicas stay byte-identical and never re-sync. */
bool metal_graph_load_directional_steering(
        ds4_gpu_graph *g,
        const char      *path,
        float            attn_scale,
        float            ffn_scale);

bool metal_graph_directional_steering_attn_enabled(const ds4_gpu_graph *g);

bool metal_graph_directional_steering_ffn_enabled(const ds4_gpu_graph *g);

bool metal_graph_apply_directional_steering_attn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows);

bool metal_graph_apply_directional_steering_ffn(
        ds4_gpu_graph  *g,
        ds4_gpu_tensor *x,
        uint32_t          il,
        uint32_t          rows);

bool metal_graph_configure_dspark_capture(
        ds4_gpu_graph            *g,
        const ds4_dspark_weights *dw);

uint64_t metal_graph_context_bytes_for_kv_policy(
        uint32_t  ctx_size,
        uint32_t  raw_cap,
        uint32_t  prefill_cap,
        uint64_t *kv_cache_bytes_out);

ds4_gpu_tensor *metal_graph_alloc_kv_cache_tensor_on(
        bool managed,
        int tier,
        uint64_t bytes);



/* =========================================================================
 * CUDA Diagnostic Dump Hooks.
 * =========================================================================
 *
 * The release path calls these after important stages, but they are no-ops
 * unless DS4_CUDA_GRAPH_DUMP_PREFIX is set.
 * Dumping synchronizes and restarts the command batch, so it is intentionally
 * isolated here.
 */

typedef struct {
    int init;
    const char *prefix;
    const char *name;
    int layer_set;
    uint32_t layer;
    int pos_set;
    uint32_t pos;
} metal_graph_debug_config;

const metal_graph_debug_config *metal_graph_debug_get_config(void);

bool metal_graph_debug_wants(const char *name, uint32_t il, uint32_t pos);

void metal_graph_debug_dump_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f32,
        uint32_t          il,
        uint32_t          pos);

void metal_graph_debug_dump_f16_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_f16,
        uint32_t          il,
        uint32_t          pos);

void metal_graph_debug_dump_i32_tensor(
        const char       *name,
        ds4_gpu_tensor *t,
        uint64_t          n_i32,
        uint32_t          il,
        uint32_t          pos);

bool metal_graph_needs_ffn_out(const ds4_gpu_graph *g, uint32_t il, uint32_t pos);

/* tier-aware lazy allocator. The Class P ffn_out scratch is
 * created on demand the first time a layer that materializes ffn_out runs
 * on a tier; subsequent visits to the same tier reuse the existing slot.
 * Single-tier paths: active_tier == 0 always, behavior unchanged. */
bool metal_graph_ensure_ffn_out(ds4_gpu_graph *g);

bool metal_graph_ensure_batch_ffn_out(ds4_gpu_graph *g);

bool metal_graph_cuda_greedy_splitkv_requested(void);

bool metal_graph_cuda_greedy_vec4_requested(void);

bool metal_graph_cuda_splitkv_spec_requested(void);

bool metal_graph_cuda_splitkv_spec_batch_verify_requested(void);

bool metal_graph_cuda_greedy_vec4_fallback_requested(void);

bool metal_graph_cuda_greedy_splitkv_fallback_requested(void);

bool metal_graph_cuda_q_norm_rope_fuse_requested(void);

bool metal_graph_cuda_qkv_kv_rope_fuse_requested(void);
/* Allocate the complete target runtime on the single GB10. */
bool metal_graph_alloc_raw_cap(
        ds4_gpu_graph *g,
        const ds4_weights     *weights,
        const ds4_layer_weights *layer,
        uint32_t                raw_cap,
        uint32_t                ctx_size,
        uint32_t                prefill_cap,
        bool                    enable_dspark_verify);

bool ds4x_graph_cache_zero(ds4_gpu_graph *g,
                                  ds4x_cache_kind kind,
                                  ds4_gpu_tensor *dst,
                                  uint32_t rows);

bool ds4x_graph_cache_pack(ds4_gpu_graph *g,
                                  ds4x_cache_kind kind,
                                  ds4_gpu_tensor *dst,
                                  uint64_t dst_row,
                                  const ds4_gpu_tensor *src,
                                  uint32_t src_row,
                                  uint32_t rows);

bool ds4x_graph_indexer(ds4_gpu_graph *g,
                               ds4x_indexer_mode mode,
                               ds4_gpu_tensor *scores,
                               ds4_gpu_tensor *selected,
                               const ds4_gpu_tensor *query,
                               const ds4_gpu_tensor *weights,
                               const ds4_gpu_tensor *cache,
                               uint32_t n_comp,
                               uint32_t n_tokens,
                               uint32_t pos0,
                               uint32_t ratio,
                               float scale);

bool ds4x_graph_attention_decode(
        ds4_gpu_graph *g,
        ds4_gpu_tensor *heads,
        const ds4_model *model,
        uint64_t sinks_offset,
        const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *raw_kv,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        const ds4_gpu_tensor *compressed_kv,
        uint32_t n_comp);

bool ds4x_graph_routed_moe(
        ds4_gpu_graph *g,
        const ds4_model *model,
        const ds4_layer_weights *layer,
        uint32_t layer_index,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t output_dim);

bool ds4x_graph_output_hc(ds4_gpu_graph *g,
                                 ds4_gpu_tensor *output,
                                 const ds4_gpu_tensor *pre,
                                 const ds4_model *model,
                                 uint64_t scale_offset,
                                 uint64_t base_offset);

ds4x_decode_graph_args ds4x_graph_decode_args(
        ds4_gpu_graph *g,
        uint32_t layer,
        uint32_t island,
        uint32_t variant);

bool metal_graph_use_reference_hc_decode(void);

bool metal_graph_use_reference_kv_decode(void);

bool metal_graph_use_reference_qkv_norm(void);

bool metal_graph_use_reference_qkv_pair_proj(void);

bool metal_graph_use_reference_compressor_pair_proj(void);

bool metal_graph_use_reference_hc_norm_decode(void);

bool metal_graph_enable_batch_hc_norm_fusion(void);

bool metal_graph_use_reference_shared_down_hc(void);

bool metal_graph_use_reference_attn_out_hc(void);

bool metal_graph_decode_hc_pre(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const ds4_model        *model,
        uint64_t                scale_offset,
        uint64_t                base_offset);

bool metal_graph_hc_norm_fusion_check_enabled(void);

bool metal_graph_check_hc_norm_fusion(
        const char            *label,
        ds4_gpu_tensor        *fused_out,
        ds4_gpu_tensor        *fused_norm,
        const ds4_gpu_tensor  *mix,
        const ds4_gpu_tensor  *residual_hc,
        const ds4_model       *model,
        uint64_t               scale_offset,
        uint64_t               base_offset,
        uint64_t               norm_weight_offset,
        uint32_t               il,
        uint32_t               pos);

bool metal_graph_decode_kv_store(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row);

uint32_t metal_graph_attn_comp_cache_format(void);

ds4_gpu_tensor *metal_graph_attn_comp_update_target(
        ds4_gpu_graph *g,
        uint32_t       il);

uint32_t metal_graph_attn_comp_update_row(uint32_t row);

bool metal_graph_commit_attn_comp_stage(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows);

ds4_gpu_tensor *metal_graph_attn_comp_row_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row);

ds4_gpu_tensor *metal_graph_attn_comp_prefill_target(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows);

void metal_graph_attn_comp_prefill_target_free(ds4_gpu_tensor *t);

/* Encode one DS4 decode layer on CUDA. This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */
bool metal_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0);

bool metal_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);

bool metal_graph_decode_stage_profile_enabled(uint32_t il);

bool metal_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

bool metal_graph_matmul_dense_quant_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

bool metal_graph_dense_quant_row_bytes(
        const ds4_tensor *w,
        uint64_t          in_dim,
        uint64_t         *row_bytes);

bool metal_graph_matmul_dense_quant_abs(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

bool metal_graph_attention_output_dense_quant_low(
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads);

bool metal_graph_attention_output_dense_quant_batch(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        const ds4_tensor       *out_b,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

uint32_t metal_graph_raw_start_for_span(
        const ds4_gpu_graph *g,
        uint32_t last_pos,
        uint32_t n_raw);

uint32_t metal_graph_raw_span_for_batch(
        const ds4_gpu_graph *g,
        uint32_t pos0,
        uint32_t n_tokens);

bool metal_graph_capture_prefix_attn_state(
        ds4_gpu_graph *g,
        uint32_t il,
        uint32_t slot);

bool metal_graph_capture_prefix_index_state(
        ds4_gpu_graph *g,
        uint32_t il,
        uint32_t slot);

uint32_t metal_graph_decode_indexer_sparse_threshold(
        const ds4_gpu_graph *g);



typedef enum {
    METAL_DECODE_LAYER_FULL = 0,
    METAL_DECODE_LAYER_TO_FFN,
    METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_FFN,
    METAL_DECODE_LAYER_FROM_ATTN_PRE_TO_ATTN,
    METAL_DECODE_LAYER_TO_QKV,
    METAL_DECODE_LAYER_FROM_QKV_TO_ATTN,
    METAL_DECODE_LAYER_FROM_KV_STORE_TO_ATTN,
    METAL_DECODE_LAYER_FROM_ATTN_TO_FFN,
} metal_decode_layer_phase;

bool metal_graph_encode_decode_layer(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token);

bool metal_graph_output_logits_head_matmul(
        ds4_gpu_graph        *g,
        const ds4_model      *model,
        const ds4_weights    *weights,
        ds4_gpu_tensor       *norm_full,
        ds4_gpu_tensor       *dst_logits,
        uint32_t              n_tokens,
        uint64_t              vocab_dim);
/* Encode the final HC collapse, output norm, and vocab projection on CUDA. */
bool metal_graph_encode_output_head(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint64_t               vocab_dim);

/* Batched output head for speculative verification.
 *
 * A target verifier only needs top-1 ids for intermediate draft rows and full
 * logits for the last accepted row.  Running the normal one-row output head in
 * a loop serializes the HC collapse, output norm, and Q8 vocab projection.  For
 * tiny DSpark suffixes we instead process all rows together and let the GPU reduce
 * each row to a top id; the CPU reads back just those ids plus the last row's
 * logits needed to continue the exact target stream. */
/* Shared vocab-head matmul pads small batches to 8 rows for the exact-mma Q8
 * kernel. */
bool metal_graph_output_logits_head_matmul(
        ds4_gpu_graph        *g,
        const ds4_model      *model,
        const ds4_weights    *weights,
        ds4_gpu_tensor       *norm_full,
        ds4_gpu_tensor       *dst_logits,
        uint32_t              n_tokens,
        uint64_t              vocab_dim);

bool metal_graph_encode_output_head_batch(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        uint32_t               n_tokens,
        uint64_t               vocab_dim);

bool metal_graph_matmul_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

bool metal_graph_dense_quant_row_bytes(
        const ds4_tensor *w,
        uint64_t          in_dim,
        uint64_t         *row_bytes);

bool metal_graph_matmul_dense_quant_abs(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

bool metal_graph_matmul_dense_quant_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

bool metal_graph_attention_output_dense_quant_low(
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads);

bool metal_graph_attention_output_dense_quant_batch(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_graph        *g,
        const ds4_model        *model,
        const ds4_tensor       *out_a,
        const ds4_tensor       *out_b,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

bool metal_graph_matmul_q8_0_named_tensor(
        const char             *module,
        uint32_t                il,
        uint32_t                pos0,
        ds4_gpu_tensor       *out,
        const ds4_model        *model,
        const ds4_tensor       *w,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);
void metal_graph_dspark_capture_invalidate(ds4_gpu_graph *g);

void metal_graph_dspark_cache_reset(ds4_gpu_graph *g);

bool metal_graph_dspark_cache_window_valid(
        const ds4_gpu_graph *g,
        uint32_t             token_start,
        uint32_t             raw_start,
        uint32_t             len);

bool metal_graph_dspark_cache_current_window_valid(
        const ds4_gpu_graph *g);

bool metal_graph_dspark_cache_set_window(ds4_gpu_graph *g,
                                                uint32_t       token_start,
                                                uint32_t       len);

bool metal_graph_dspark_cache_crop_to_prefix(ds4_gpu_graph *g,
                                                    uint32_t       prefix_len);

bool metal_graph_dspark_cache_ends_at(const ds4_gpu_graph *g,
                                             uint32_t             pos);

bool metal_graph_dspark_cache_claim_appended_row(ds4_gpu_graph *g,
                                                        uint32_t pos);

bool ds4_test_dspark_cache_window_crop(void);

void metal_graph_dspark_capture_begin_prefill(ds4_gpu_graph *g);

bool metal_graph_dspark_capture_prefill_layer(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       start,
        uint32_t       n_tokens);

bool metal_graph_dspark_capture_verified_suffix_begin(
        ds4_gpu_graph *g,
        uint32_t       start,
        uint32_t       n_tokens,
        bool           commands_open);

bool metal_graph_dspark_capture_verified_suffix_layer(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       start,
        uint32_t       n_tokens);

/* Encode a full single-token decode step on CUDA.  This is the generation
 * hot path: update caches, run all layers, then produce logits. */
bool metal_graph_encode_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        bool                   need_logits,
        bool                   allow_split_flush);

ds4_gpu_tensor *metal_graph_tensor_row_view(
        ds4_gpu_tensor *base,
        uint32_t          row,
        uint64_t          row_values);

/* Upload prompt token ids for kernels that need token-aware hash routing. */
bool metal_graph_upload_prompt_tokens(
        ds4_gpu_tensor *out_tokens,
        const token_vec  *prompt,
        uint32_t          pos0,
        uint32_t          n_tokens);

/* Rebuild ratio-4 compressor state after chunked prefill so a following decode
 * token sees the same rolling compression window. */
bool metal_graph_refresh_ratio4_compressor_state(
        ds4_gpu_graph  *g,
        const ds4_model  *model,
        ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const ds4_tensor *kv_weight,
        const ds4_tensor *score_weight,
        const ds4_tensor *ape,
        uint32_t          head_dim,
        uint32_t          width,
        uint32_t          pos0,
        uint32_t          n_tokens);

/* Seed the batched HC state from token ids: every HC stream starts as the same
 * 4096-wide embedding. Long prefill chunks use the CUDA get-rows/repeat
 * kernel so the CPU does not build and upload a large [token, HC, dim] tensor. */
bool metal_graph_upload_prompt_embeddings_hc(
        ds4_gpu_tensor   *out_hc,
        ds4_gpu_tensor   *tokens,
        const ds4_model    *model,
        const ds4_weights  *weights,
        const token_vec    *prompt,
        uint32_t            pos0,
        uint32_t            n_tokens);

bool metal_graph_hc_rms_scale_project(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_scratch,
        const ds4_model        *model,
        const ds4_tensor       *weight,
        const ds4_gpu_tensor *x,
        uint64_t                in_dim,
        uint32_t                n_tokens);

bool metal_graph_warmup_prefill_kernels(
        ds4_gpu_graph   *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           n_tokens);

/* Encode the batched prefill attention half for one layer.  It mirrors the CPU
 * layer-major path: HC pre/norm, Q/KV, cache/compression, prefix attention. */
bool metal_graph_indexer_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        uint32_t    n_comp,
        double     *stage_t0);

bool metal_graph_layer_stage_profile_enabled(uint32_t il);

bool metal_graph_decode_stage_profile_enabled(uint32_t il);

bool metal_graph_layer_stage_profile_start(uint32_t il);

/* Optional prefill stage profiler. It intentionally ends the current CUDA
 * command buffer and waits, so the printed number includes encoding plus GPU
 * execution for the stage just emitted. This is disabled by default because it
 * adds synchronization points and changes scheduling. */
bool metal_graph_layer_stage_profile_boundary(
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);

bool metal_graph_q_stage_profile_boundary(
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0);

ds4_gpu_tensor *metal_graph_tensor_row_range_view(
        ds4_gpu_tensor *base,
        uint32_t          row0,
        uint32_t          rows,
        uint64_t          row_values);
bool metal_graph_encode_layer_attention_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens);
/* Encode the multi-token prefill/verification FFN half. */
bool metal_graph_encode_layer_ffn_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens);

/* Encode one complete layer for prefill by chaining attention and FFN batches. */
bool metal_graph_encode_layer_batch(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos0,
        uint32_t                n_tokens);

/* Execute one CUDA decode token and read back logits. */
bool metal_graph_eval_token_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        int                    token,
        uint32_t               pos,
        float                 *logits);

/* Greedy verifier helper.  Speculative decoding only needs the target model's
 * top token after most accepted draft rows; the full vocabulary row is needed
 * once, for the final committed state that normal sampling will continue from.
 * Keeping intermediate rows device-resident avoids turning verification into a
 * sequence of large CPU readbacks. */
bool dspark_stage0_weights_ready(
        const ds4_gpu_graph     *g,
        const ds4_dspark_weights *dw);

bool metal_graph_eval_dspark_stage0(
        ds4_gpu_graph          *g,
        const ds4_model        *dspark_model,
        const ds4_dspark_weights *dw);

bool dspark_draft_block_ready(
        const ds4_gpu_graph      *g,
        const ds4_weights        *base_weights,
        const ds4_dspark_weights *dw,
        int                       token);

bool dspark_stage_input_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw);

bool dspark_stage_cache_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw);

bool dspark_noncausal_attention_probe_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw);

bool metal_graph_probe_dspark_noncausal_attention(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw);

bool metal_graph_prepare_dspark_setup_block(
        ds4_gpu_graph            *g,
        const ds4_model          *base_model,
        const ds4_weights        *base_weights,
        const ds4_dspark_weights *dw,
        int                       token,
        uint32_t                  pos);

bool metal_graph_prepare_dspark_stage0_setup_block(
        ds4_gpu_graph            *g,
        const ds4_model          *base_model,
        const ds4_weights        *base_weights,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        int                       token,
        uint32_t                  pos);

bool dspark_stage_block_ready(
        const ds4_gpu_graph      *g,
        const ds4_dspark_weights *dw,
        uint32_t                  stage);

bool metal_graph_seed_dspark_initial_cache_from_prefill(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  batch_start,
        uint32_t                  n_tokens,
        uint32_t                 *seeded_rows);

bool metal_graph_eval_dspark_final_hidden(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        bool                      commands_open);

bool metal_graph_eval_dspark_stage_chain(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  pos,
        bool                      encode_final_hidden,
        uint32_t                 *completed_stages,
        uint32_t                 *cache_start_out,
        uint32_t                 *cache_rows_out);
/* Keep the support KV ring aligned while the scheduler skips proposals. */
bool metal_graph_dspark_ring_maintain(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        uint32_t                  pos);

bool dspark_final_head_ready(
        const ds4_gpu_graph      *g,
        const ds4_weights        *base_weights,
        const ds4_dspark_weights *dw);

bool metal_graph_eval_dspark_base_logits(
        ds4_gpu_graph            *g,
        const ds4_model          *base_model,
        const ds4_weights        *base_weights,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw);

bool metal_graph_eval_dspark_final_hidden(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        bool                      commands_open);

bool metal_graph_eval_dspark_base_logits_from_hidden(
        ds4_gpu_graph      *g,
        const ds4_model    *base_model,
        const ds4_weights  *base_weights,
        const ds4_dspark_weights *dw);

bool dspark_markov_probe_ready(
        const ds4_dspark_weights *dw);



typedef struct {
    const uint8_t *data;
    const int8_t *xq;
    const float *xscale;
    const float *logits;
    uint64_t in_dim;
    uint64_t blocks;
    uint64_t rows_per_slot;
    uint32_t best_idx[DS4_MAX_THREADS];
    float best_val[DS4_MAX_THREADS];
} dspark_markov_q8_0_argmax_ctx;

bool dspark_disable_reuse_confidence0_markov(void);

bool dspark_apply_markov_greedy_probe(
        float                  *logits,
        const ds4_model        *dspark_model,
        const ds4_dspark_weights *dw,
        int                     first_prev_token,
        float                  *markov_state,
        float                  *markov_bias,
        int32_t                 proposal[DS4_DSPARK_MAX_BLOCK_SIZE],
        uint32_t               *proposal_len);

bool dspark_confidence_probe_ready(
        const ds4_dspark_weights *dw);

bool dspark_eval_confidence_probe(
        float                  *confidence_logits,
        const float            *hidden_rows,
        const ds4_model        *dspark_model,
        const ds4_dspark_weights *dw,
        int                     first_prev_token,
        const int32_t           proposal[DS4_DSPARK_MAX_BLOCK_SIZE],
        float                  *markov_state,
        float                  *features,
        uint32_t               *confidence_len);

bool dspark_apply_markov_confidence_lazy_runtime(
        ds4_gpu_graph          *g,
        const ds4_model        *dspark_model,
        const ds4_dspark_weights *dw,
        int                     first_prev_token,
        float                   confidence_threshold,
        float                  *logits,
        float                  *markov_bias,
        float                  *features,
        size_t                  features_cap,
        int32_t                 proposal[DS4_DSPARK_MAX_BLOCK_SIZE],
        uint32_t               *proposal_len,
        uint32_t               *confidence_len,
        uint32_t               *confidence_prefix_len,
        bool                    reuse_first_confidence,
        float                  *confidence0);

bool dspark_eval_confidence0_runtime(
        ds4_gpu_graph            *g,
        const ds4_model          *dspark_model,
        const ds4_dspark_weights *dw,
        int                       first_prev_token,
        float                    *features,
        size_t                    features_cap,
        float                    *confidence0);

uint32_t dspark_confident_prefix_len(
        const float *confidence_logits,
        uint32_t     confidence_len,
        float        threshold);

bool metal_graph_reset_prefill_state(ds4_gpu_graph *g);
bool metal_graph_prefill_raw_swa(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled);

/* Prefill a contiguous token range in fixed-size chunks.
 *
 * The common case starts at token zero, but server sessions also use this to
 * extend an existing KV cache with a long suffix.  Resumed chunks are aligned
 * to the same absolute prefill-cap boundaries used by a cold full prompt, so
 * compression windows and row finalization follow the same schedule after the
 * cached prefix.
 */
bool metal_graph_prefill_chunked_range(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled);
/* Long prompts are prefetched in fixed-size chunks.  Chunks bound transient
 * attention buffers while preserving the same final KV/cache state. */
bool metal_graph_prefill_chunked(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        int                    n_tokens,
        float                 *logits,
        bool                   show_progress,
        ds4_session_progress_fn progress,
        void                  *progress_ud,
        ds4_session_progress_fn display_progress,
        void                  *display_progress_ud,
        ds4_session_cancel_fn  cancel,
        void                  *cancel_ud,
        bool                  *cancelled);



typedef struct ds4_verify_suffix_timing {
    double upload_ms;
    double layer_ms;
    double head_ms;
    double read_ms;
    bool fused_head;
} ds4_verify_suffix_timing;

bool metal_graph_verify_suffix_tops(
        ds4_gpu_graph *g,
        const ds4_model       *model,
        const ds4_weights     *weights,
        const token_vec       *prompt,
        uint32_t               start,
        uint32_t               n_tokens,
        bool                   capture_prefix1,
        bool                   capture_dspark_hidden,
        int                   *row_tops,
        float                 *row_logits,
        ds4_verify_suffix_timing *timing);

bool metal_graph_read_spec_logits_row(ds4_gpu_graph *g, uint32_t row, float *logits);

/* Exact small-suffix target verifier for DSpark.
 *
 * The generic batch prefill path is fast, but it is not a safe substitute for
 * autoregressive decode: small row-wise differences in HC/MoE/output kernels
 * are enough to flip future greedy tokens.  This verifier keeps the exact
 * decode kernels and cache update order, but encodes the two proposed tokens
 * layer-by-layer in one command stream.  It returns the exact target top after
 * token0, and exact logits after token1. */
uint32_t metal_graph_raw_cap_for_context(int ctx_size, uint32_t prefill_cap);

/* Choose the prefill ubatch size. Whole-batch is fastest for normal prompts.
 * Long Flash prompts default to 4096-token chunks; PRO defaults to 8192. */
uint32_t metal_graph_prefill_cap_for_prompt(int prompt_len,
                                                   uint32_t prefill_chunk);

/* When a server request shares a large prefix with the live checkpoint, extend
 * the KV cache with batched prefill instead of single-token decode.  On an M3
 * Max, prefill is faster from 2-token suffixes upward; keep the default at 4
 * as a conservative crossover.  The env knob remains useful for retuning. */
uint32_t metal_graph_resume_prefill_min_tokens(void);


typedef struct ds4_vocab ds4_vocab;



/* =========================================================================
 * Tokenizer and Chat Prompt Encoding.
 * =========================================================================
 *
 * DeepSeek V4 Flash stores a GPT-2 style byte-level BPE tokenizer in GGUF.
 * The implementation below is intentionally small.  It loads token strings
 * and merge ranks from the mmaped file, builds two open-addressed hash tables,
 * and applies BPE to user text.  Chat special tokens are inserted directly by
 * ID; user text goes through BPE.
 */

typedef struct {
    ds4_str key;
    int value;
    bool used;
} str_i32_entry;



typedef struct {
    str_i32_entry *entry;
    uint64_t cap;
    uint64_t used;
} str_i32_table;

void token_vec_push(token_vec *tv, int token);

void token_vec_free(token_vec *tv);

void ds4_tokens_push(ds4_tokens *tv, int token);

void ds4_tokens_free(ds4_tokens *tv);

void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);

bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);



struct ds4_vocab {
    ds4_str *token;
    int n_vocab;
    int bos_id;
    int eos_id;
    int user_id;
    int assistant_id;
    int think_start_id;
    int think_end_id;
    int dsml_id;
    str_i32_table token_to_id;
    str_i32_table merge_rank;
};



struct ds4_engine {
    ds4_model model;
    ds4_model support_model;
    ds4_vocab vocab;
    ds4_weights weights;
    ds4_dspark_weights dspark_weights;
    ds4_backend backend;
    ds4_support_kind support_kind;
    int dspark_exec_tier;
    uint32_t support_stages;
    float dspark_confidence_threshold;
    char *directional_steering_file;
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    int power_percent;
    uint32_t prefill_chunk;
    uint64_t startup_model_span_bytes;
    ds4_memory_lock simulated_memory;
    bool quality;
    bool dspark;
    bool dspark_strict;
    bool metal_ready;
};

void ds4_engine_print_startup_memory(
        const ds4_engine *e,
        int               ctx_size);



typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

/* Load token strings, special token ids, and merge ranks from GGUF metadata. */

void vocab_load(ds4_vocab *vocab, const ds4_model *model);

void vocab_free(ds4_vocab *vocab);

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);

void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);

void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);

void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);

void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);

void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);

void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);

int ds4_token_eos(ds4_engine *e);

bool ds4_token_is_stop(ds4_engine *e, int token);

bool ds4_token_is_thinking_control(ds4_engine *e, int token);

bool ds4_token_is_stop_for_think_mode(
        ds4_engine      *e,
        int              token,
        ds4_think_mode   mode);

int ds4_token_user(ds4_engine *e);

int ds4_token_assistant(ds4_engine *e);

int argmax_f32_excluding_unrolled8(
        const float *logits,
        uint32_t     n,
        int          excluded_id);

int sample_argmax(const float *logits, uint32_t n_vocab);



typedef struct {
    int id;
    float logit;
    float prob;
} sample_candidate;

int sample_top_p_min_p(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t    *rng,
        float       *prob_scratch);
#ifdef DS4_TEST_HOOKS

int ds4_test_sample_logits(const float *logits, uint32_t n_vocab,
                           float temperature, int top_k,
                           float top_p, float min_p, uint64_t *rng,
                           float *prob_scratch);

int ds4_test_argmax_excluding_logits(const float *logits, uint32_t n_vocab,
                                     int excluded_id);
#endif

/* Packed single-GB10 context allocation estimate used by CLI diagnostics. */
ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int         ctx_size,
        uint32_t    prefill_chunk);

ds4_context_memory ds4_context_memory_estimate(ds4_backend backend,
                                               int ctx_size);

/* =========================================================================
 * Engine API and Process Lock.
 * =========================================================================
 *
 * The public entry points acquire the single instance lock, open the GGUF with
 * the backend-appropriate mmap policy, and expose tokenized prompt operations
 * to the CLI and server.
 */

const char *ds4_backend_name(ds4_backend backend);

void ds4_linux_graph_backend_set_oom_score(ds4_backend backend);

bool ds4_think_mode_enabled(ds4_think_mode mode);

const char *ds4_think_mode_name(ds4_think_mode mode);

const char *ds4_think_max_prefix(void);

uint32_t ds4_think_max_min_context(void);

ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);

void ds4_release_instance_lock(void);

/* Refuse to start a second ds4 process.  The model can map tens of GiB, so a
 * stale accidental second run is more dangerous than a normal CLI error. */
void ds4_acquire_instance_lock(void);
typedef struct {
    uint32_t n_comp[DS4_MAX_LAYER];
    uint32_t n_index_comp[DS4_MAX_LAYER];
    uint32_t dspark_cache_start;
    uint32_t dspark_cache_token_start;
    uint32_t dspark_cache_len;
    bool light;
} ds4_spec_frontier;



typedef struct ds4_dspark_spec_stats {
    uint64_t cycles;
    uint64_t first_tokens;
    uint64_t proposed_tokens;
    uint64_t accepted_draft_tokens;
    uint64_t full_accepts;
    uint64_t partial_accepts;
    uint64_t direct_full_commits;
    uint64_t direct_partial_commits;
    uint64_t replay_fallbacks;
    uint64_t first_misses;
    uint64_t no_draft;
    uint64_t no_room;
    uint64_t invalid_draft;
    uint64_t draft_len_hist[DS4_DSPARK_MAX_BLOCK_SIZE + 1u];
    uint64_t accepted_len_hist[DS4_DSPARK_MAX_BLOCK_SIZE + 1u];
    uint64_t scheduler_skips;
    uint64_t tail_skips;
    uint64_t verifier_unavailable;
    uint64_t verifier_errors;
    double target_ms;
    double saved_ms;
    double propose_ms;
    double propose_stage0_ms;
    double propose_setup_ms;
    double propose_cache_ms;
    double propose_chain_ms;
    double propose_hidden_ms;
    double propose_conf0_ms;
    double propose_logits_ms;
    double propose_markov_ms;
    double propose_confidence_ms;
    double snapshot_ms;
    double verify_ms;
    double verify_upload_ms;
    double verify_layer_ms;
    double verify_head_ms;
    double verify_read_ms;
    uint64_t verifier_fused_head;
    double replay_ms;
    double total_ms;
} ds4_dspark_spec_stats;


struct ds4_session {
    ds4_engine *engine;
    ds4_gpu_graph graph;
    ds4_spec_frontier greedy_splitkv_anchor;
    token_vec checkpoint;
    token_vec greedy_splitkv_segment;
    float *logits;
    float *sample_probs;
    int greedy_splitkv_anchor_len;
    float *spec_row_logits;
    float *dspark_markov_bias;
    float *dspark_conf_features;
    size_t dspark_conf_features_cap;
    int dspark_draft_tokens[DS4_DSPARK_MAX_BLOCK_SIZE];
    uint32_t dspark_draft_len;
    uint32_t dspark_sched_cycles;
    uint32_t dspark_sched_accepted;
    uint32_t dspark_sched_no_draft;
    uint32_t dspark_sched_skip;
    uint32_t dspark_sched_lifetime_accepted;
    double dspark_sched_life_extra_ms;
    double dspark_sched_life_saved_ms;
    double dspark_sched_extra_ms;
    double dspark_sched_saved_ms;
    double dspark_last_target_eval_ms;
    double dspark_last_propose_ms;
    float dspark_last_confidence0;
    bool dspark_draft_valid;
    bool dspark_sched_skipped_cycle;
    bool dspark_sched_long_accept_seen;
    bool dspark_last_confidence0_valid;
    ds4_dspark_spec_stats dspark_stats;
    ds4_session_progress_fn progress;
    void *progress_ud;
    ds4_session_progress_fn display_progress;
    void *display_progress_ud;
    ds4_session_cancel_fn cancel;
    void *cancel_ud;
    uint32_t prefill_cap;
    int ctx_size;
    bool checkpoint_valid;
    bool greedy_splitkv_anchor_valid;
};
bool ds4_dspark_stats_enabled(void);

void ds4_dspark_stats_note_len(
        uint64_t hist[DS4_DSPARK_MAX_BLOCK_SIZE + 1u],
        uint32_t len);

bool ds4_dspark_scheduler_enabled(void);

uint32_t ds4_dspark_scheduler_tail_min_tokens(void);

/* Timing-sensitive scheduling changes which arithmetic path advances a token.
 * Keep it opt-in so greedy DSpark output is reproducible across runs. */
bool ds4_dspark_scheduler_timing_enabled(void);

bool ds4_session_dspark_scheduler_should_skip(ds4_session *s);

void ds4_session_dspark_scheduler_note(
        ds4_session *s,
        uint32_t     accepted_drafts,
        bool         no_draft,
        double       extra_ms);
#define DS4_SESSION_IO_CHUNK (8u * 1024u * 1024u)

void payload_set_err(char *err, size_t errlen, const char *msg);

int payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes,
                               char *err, size_t errlen);

int payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes,
                              uint64_t *remaining,
                              char *err, size_t errlen);

int payload_write_u32(FILE *fp, uint32_t v,
                             char *err, size_t errlen);

int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining,
                            char *err, size_t errlen);

int payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes,
                                   char *err, size_t errlen);

uint64_t layer_attn_state_bytes(uint32_t ratio);

uint64_t layer_index_state_bytes(uint32_t ratio);

uint32_t session_raw_live_rows(const ds4_gpu_graph *g,
                                      uint32_t checkpoint_len);

uint64_t session_payload_live_tensor_bytes(
        const ds4_gpu_graph *g, uint32_t checkpoint_len);

int payload_write_tensor_span(FILE *fp,
                                     const ds4_gpu_tensor *tensor,
                                     uint64_t offset, uint64_t bytes,
                                     uint8_t *buf, size_t cap,
                                     char *err, size_t errlen);

int payload_read_tensor_span(FILE *fp, ds4_gpu_tensor *tensor,
                                    uint64_t offset, uint64_t bytes,
                                    uint8_t *buf, size_t cap,
                                    uint64_t *remaining,
                                    char *err, size_t errlen);

bool ds4_session_is_cpu(const ds4_session *s);

void ds4_session_dspark_capture_invalidate(ds4_session *s);

void ds4_session_dspark_capture_note_checkpoint(ds4_session *s);

int ds4_engine_routed_quant_bits(ds4_engine *e);

bool ds4_engine_has_output_head(ds4_engine *e);

bool ds4_engine_has_mtp(ds4_engine *e);

int ds4_engine_dspark_block_size(ds4_engine *e);

const ds4_tokens *ds4_session_tokens(ds4_session *s);

void spec_frontier_free(ds4_spec_frontier *f);

bool spec_frontier_snapshot(ds4_spec_frontier *f, ds4_session *s);

bool spec_frontier_restore(ds4_spec_frontier *f, ds4_session *s);

/* Commit an intermediate state captured by a tiny speculative verifier.
 *
 * Append-only cache rows beyond the accepted prefix can remain as invisible
 * garbage. Only compressor frontiers and row counters need rewinding. */
bool spec_frontier_commit_prefix(ds4_session *s, uint32_t prefix_len);

void session_greedy_splitkv_reset(ds4_session *s);

uint64_t ds4_session_payload_bytes(ds4_session *s);

int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);

void ds4_session_payload_file_free(ds4_session_payload_file *payload);

int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen);

int ds4_session_save_payload(ds4_session *s, FILE *fp,
                             char *err, size_t errlen);

int ds4_session_load_payload(ds4_session *s, FILE *fp,
                             uint64_t payload_bytes,
                             char *err, size_t errlen);

int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap,
                              char *err, size_t errlen);

int ds4_session_load_snapshot(ds4_session *s,
                              const ds4_session_snapshot *snap,
                              char *err, size_t errlen);

void ds4_session_snapshot_free(ds4_session_snapshot *snap);

int ds4_engine_generate_argmax(
        ds4_engine        *e,
        const ds4_tokens  *prompt,
        int                n_predict,
        int                ctx_size,
        ds4_token_emit_fn  emit,
        ds4_generation_done_fn done,
        void              *emit_ud,
        ds4_session_progress_fn progress,
        void              *progress_ud);
#ifdef DS4_TEST_HOOKS

int ds4_test_session_read_logits(ds4_session *s, float *out,
                                 uint64_t out_bytes);

int ds4_test_session_seed_frontier(ds4_session *s, uint32_t pos,
                                   bool initialize_cache);
#endif

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);

void ds4_engine_summary(ds4_engine *e);

int ds4_engine_vocab_size(ds4_engine *e);

uint32_t ds4_engine_prefill_chunk(ds4_engine *e);

int ds4_engine_power(ds4_engine *e);

int ds4_engine_set_power(ds4_engine *e, int power_percent);

const char *ds4_engine_model_name(ds4_engine *e);

int ds4_engine_layer_count(ds4_engine *e);

uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer);

uint64_t ds4_engine_hidden_f32_values(ds4_engine *e);

int ds4_engine_model_id(ds4_engine *e);

void ds4_engine_close(ds4_engine *e);

bool ds4_dspark_stats_enabled(void);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);

void ds4_session_free(ds4_session *s);

int ds4_session_power(ds4_session *s);

int ds4_session_set_power(ds4_session *s, int power_percent);

void ds4_session_set_progress(ds4_session *s,
                              ds4_session_progress_fn fn, void *ud);

void ds4_session_set_display_progress(ds4_session *s,
                                      ds4_session_progress_fn fn, void *ud);

void ds4_session_set_cancel(ds4_session *s,
                            ds4_session_cancel_fn fn, void *ud);

void ds4_session_report_progress(ds4_session *s, const char *event,
                                 int current, int total);



typedef struct {
    ds4_session *session;
    const ds4_tokens *prompt;
    ds4_session_progress_fn user;
    void *user_ud;
} ds4_sync_progress;

int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt,
                     char *err, size_t errlen);

/* Return true when canonicalization would replace already-sampled tokens.
 *
 * A DS4 session checkpoint is more than a token vector: the backend state also
 * contains raw SWA rows, compressed KV rows, indexer rows, and compressor
 * frontiers.  Replacing any part of the live tail requires restoring that whole
 * frontier first.  Extending exactly at the live end is safe; rewriting behind
 * it is not an in-place operation. */
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);

/* Replace the live suffix after a shared prefix.
 *
 * This is used after parsing a generated tool call.  The model may have emitted
 * DSML in an order that is semantically valid but not byte-for-byte equal to the
 * canonical prompt we will see on the next request.  Rewriting only the token
 * checkpoint is not enough: the backend still contains raw and compressed rows
 * for the old suffix.  Until we have a real frontier snapshot at the
 * rewrite point, any replacement behind the live end reports that a rebuild is
 * needed without mutating the session.  The server may still find an older disk KV
 * checkpoint before falling back to a full replay. */
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);

int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);

int ds4_session_argmax(ds4_session *s);

int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);

int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);

int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);

int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);

int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);

int ds4_session_copy_logits(ds4_session *s, float *out, int cap);

int ds4_session_set_logits(ds4_session *s, const float *logits, int n);

/* Pay the one-time CUDA submission and model-residency cost outside measured
 * prefill windows. */
void ds4_session_gpu_warmup(ds4_session *s);

int ds4_session_eval_probe_draft(ds4_session *s, int token, bool probe_mtp,
                                 char *err, size_t errlen);

int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);

int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);

void ds4_session_invalidate(ds4_session *s);

void ds4_session_rewind(ds4_session *s, int pos);

int ds4_session_pos(ds4_session *s);

int ds4_session_ctx(ds4_session *s);

int ds4_session_prefill_cap(ds4_session *s);

#endif
