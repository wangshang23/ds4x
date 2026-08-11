#include "ds4_gpu.h"
#include <math.h>

bool ds4_test_dspark_cache_window_crop(void);

static ds4_engine *test_engine_fast;
static ds4_engine *test_engine_quality;

static const char *test_model_path(void) {
    const char *model_path = getenv("DS4_TEST_MODEL");
    return (model_path && model_path[0]) ? model_path : "ds4flash.gguf";
}

static bool test_env_bool(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}

static uint32_t test_env_u32(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    char *end = NULL;
    unsigned long n = strtoul(v, &end, 10);
    if (end == v) return 0;
    return n > UINT32_MAX ? UINT32_MAX : (uint32_t)n;
}

static uint64_t test_env_gib(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    char *end = NULL;
    unsigned long long n = strtoull(v, &end, 10);
    if (end == v || n == 0) return 0;
    const uint64_t one_gib = 1024ull * 1024ull * 1024ull;
    if (n > UINT64_MAX / one_gib) return UINT64_MAX;
    return (uint64_t)n * one_gib;
}

static char *test_save_env(const char *name) {
    const char *value = getenv(name);
    if (!value) return NULL;
    size_t len = strlen(value);
    char *copy = malloc(len + 1);
    TEST_ASSERT(copy != NULL);
    if (!copy) return NULL;
    memcpy(copy, value, len + 1);
    return copy;
}

static void test_restore_env(const char *name, char *saved) {
    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
}

typedef struct {
    char *cold_decode;
    char *batch_selected_addr;
} test_streaming_prefill_env;

static test_streaming_prefill_env test_force_canonical_streaming_prefill(void) {
    test_streaming_prefill_env saved = {
        .cold_decode =
            test_save_env("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL"),
        .batch_selected_addr =
            test_save_env("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR"),
    };
    if (test_env_bool("DS4_TEST_SSD_STREAMING")) {
        setenv("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL", "1", 1);
        setenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR", "1", 1);
    }
    return saved;
}

static void test_restore_canonical_streaming_prefill(
        test_streaming_prefill_env saved) {
    test_restore_env("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL",
                     saved.cold_decode);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR",
                     saved.batch_selected_addr);
}

static ds4_backend test_model_backend(void) {
    const char *backend = getenv("DS4_TEST_BACKEND");
    if (backend && !strcmp(backend, "cpu")) return DS4_BACKEND_CPU;
#ifdef __APPLE__
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static ds4_engine *test_open_engine(bool quality) {
    ds4_engine *engine = NULL;
    /* DS4_TEST_MTP loads the MTP head on the fast engine so the speculative
     * verify regression can reuse it; draft=4 hits the multi-row verify path. */
    const char *mtp = getenv("DS4_TEST_MTP");
    ds4_engine_options opt = {
        .model_path = test_model_path(),
        .backend = test_model_backend(),
        .quality = quality,
        .ssd_streaming = test_env_bool("DS4_TEST_SSD_STREAMING"),
        .ssd_streaming_cold = test_env_bool("DS4_TEST_SSD_STREAMING_COLD"),
        .ssd_streaming_cache_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS"),
        .ssd_streaming_cache_bytes =
            test_env_gib("DS4_TEST_SSD_STREAMING_CACHE_GB"),
        .ssd_streaming_preload_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_PRELOAD_EXPERTS"),
        .mtp_path = (mtp && mtp[0] && !quality) ? mtp : NULL,
        .mtp_draft_tokens = (mtp && mtp[0] && !quality) ? 4 : 0,
    };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0);
    return engine;
}

static ds4_engine *test_get_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    if (*slot) return *slot;

    *slot = test_open_engine(quality);
    return *slot;
}

static void test_close_engines(void) {
    ds4_engine_close(test_engine_fast);
    ds4_engine_close(test_engine_quality);
    test_engine_fast = NULL;
    test_engine_quality = NULL;
}

static void test_close_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    ds4_engine_close(*slot);
    *slot = NULL;
}

static uint64_t test_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1) & ~(align - 1);
}

static uint32_t test_float_ordered_u32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

typedef struct {
    size_t mismatch_count;
    uint32_t max_ulp;
    float max_abs;
} test_float_compare_stats;

static test_float_compare_stats test_compare_float_bits(
        const float *reference,
        const float *actual,
        size_t count) {
    test_float_compare_stats stats = {0};
    for (size_t i = 0; i < count; i++) {
        if (memcmp(&reference[i], &actual[i], sizeof(float)) != 0) {
            stats.mismatch_count++;
        }

        const uint32_t ref_ordered = test_float_ordered_u32(reference[i]);
        const uint32_t actual_ordered = test_float_ordered_u32(actual[i]);
        const uint32_t ulp = ref_ordered > actual_ordered
            ? ref_ordered - actual_ordered
            : actual_ordered - ref_ordered;
        if (ulp > stats.max_ulp) stats.max_ulp = ulp;

        const float abs_error = fabsf(reference[i] - actual[i]);
        if (abs_error > stats.max_abs) stats.max_abs = abs_error;
    }
    return stats;
}

#if defined(__APPLE__)
static const uint32_t test_copy_f32_patterns[] = {
    0x00000000u, 0x80000000u, /* signed zero */
    0x7f800000u, 0xff800000u, /* infinities */
    0x7fc00000u, 0x7fc12345u, 0xffc54321u, 0x7fa00001u, /* NaNs */
    0x00000001u, 0x007fffffu, 0x80000001u, 0x807fffffu, /* F32 subnormals */
    0x32ffffffu, 0x33000000u, 0x33000001u,
    0x337fffffu, 0x33800000u, 0x33800001u, /* minimum F16 subnormal boundary */
    0x387fbfffu, 0x387fc000u, 0x387fffffu,
    0x38800000u, 0x38800001u, /* maximum subnormal/minimum normal boundary */
    0x3f800fffu, 0x3f801000u, 0x3f801001u,
    0x3f802fffu, 0x3f803000u, 0x3f803001u, /* round-to-nearest ties */
    0x477fdfffu, 0x477fe000u, 0x477fefffu,
    0x477ff000u, 0x477ff001u, 0x47800000u, /* maximum/overflow boundary */
    0xc77fe000u, 0xbf801000u, 0x3eaaaaabu, 0xbeaaaaabu,
};

static void test_fill_copy_f32_patterns(void *dst, uint32_t n, uint32_t salt) {
    uint8_t *bytes = dst;
    const uint32_t n_patterns =
        (uint32_t)(sizeof(test_copy_f32_patterns) / sizeof(test_copy_f32_patterns[0]));
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t bits = test_copy_f32_patterns[(i + salt) % n_patterns];
        memcpy(bytes + (uint64_t)i * sizeof(bits), &bits, sizeof(bits));
    }
}
#endif

static uint16_t test_float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };

    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float test_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void test_fill_q8_0_weights(uint8_t *weights,
                                   uint32_t in_dim,
                                   uint32_t out_dim,
                                   uint32_t seed) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = weights + (uint64_t)o * row_bytes;
        for (uint32_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t k = b * 32u + i;
                const int v = (int)((o * 17u + k * 23u + (o ^ k) * 3u +
                                     seed * 29u + ((o + seed) ^ k) * 5u) % 67u) - 33;
                vals[i] = (float)v / 96.0f;
                float av = fabsf(vals[i]);
                if (av > amax) amax = av;
            }
            const uint16_t scale_bits = test_float_to_f16(amax / 127.0f);
            const float scale = test_f16_to_f32(scale_bits);
            memcpy(row + b * 34u, &scale_bits, sizeof(scale_bits));
            int8_t *qs = (int8_t *)(row + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++) {
                int q = scale != 0.0f ? (int)lrintf(vals[i] / scale) : 0;
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                qs[i] = (int8_t)q;
            }
        }
    }
}

static void test_metal_f16_matvec_fast_nr0_4(void) {
    /*
     * This is the short regression for the long-context repetition failure.
     * Decode uses one-token F16 matvecs for several DS4 projections; the fast
     * nr0=4 variant must be numerically equivalent to the plain kernel.
     */
    const uint32_t in_dim = 4096;
    const uint32_t out_dim = 512;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16(w);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 31u) - 15) / 32.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                            in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            ref += w * x_host[i];
        }
        float err = fabsf(out_host[o] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_f16_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t weight_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((o * 11u + i * 13u + (o ^ i) * 5u) % 61u) - 30;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16((float)v / 96.0f);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 7u + i * 17u + (t ^ i) * 3u) % 73u) - 36;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                          in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float ref = 0.0f;
            for (uint32_t i = 0; i < in_dim; i++) {
                ref += test_f16_to_f32(weights[(uint64_t)o * in_dim + i]) *
                       x_host[(uint64_t)t * in_dim + i];
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_q8_0_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint8_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights(weights, in_dim, out_dim, 0);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 19u + i * 7u + (t ^ i)) % 71u) - 35;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(out, weights_raw, weight_alloc, 0,
                                           in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const uint8_t *row = weights + (uint64_t)o * row_bytes;
            float ref = 0.0f;
            for (uint32_t b = 0; b < in_dim / 32u; b++) {
                uint16_t scale_bits;
                memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
                const float scale = test_f16_to_f32(scale_bits);
                const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
                for (uint32_t i = 0; i < 32; i++) {
                    ref += scale * (float)qs[i] *
                           x_host[(uint64_t)t * in_dim + b * 32u + i];
                }
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_pack_slot_rows_f32(void) {
    const uint32_t n_rows = 3;
    const uint32_t width = 5;
    const uint32_t n_slots = 4;
    const uint32_t slot_cap = 6;
    const uint64_t slot_count = (uint64_t)n_slots * slot_cap * width;
    const uint64_t out_count = (uint64_t)n_rows * n_slots * width;
    const uint64_t slot_bytes = slot_count * sizeof(float);
    const uint64_t out_bytes = out_count * sizeof(float);

    ds4_gpu_tensor *slots = ds4_gpu_tensor_alloc(slot_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(slots != NULL);
    TEST_ASSERT(out != NULL);
    if (!slots || !out) {
        ds4_gpu_tensor_free(slots);
        ds4_gpu_tensor_free(out);
        return;
    }

    float *slot_host = malloc((size_t)slot_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(slot_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!slot_host || !out_host) {
        free(slot_host);
        free(out_host);
        ds4_gpu_tensor_free(slots);
        ds4_gpu_tensor_free(out);
        return;
    }

    for (uint32_t slot = 0; slot < n_slots; slot++) {
        for (uint32_t row = 0; row < slot_cap; row++) {
            for (uint32_t col = 0; col < width; col++) {
                slot_host[((uint64_t)slot * slot_cap + row) * width + col] =
                    (float)(slot * 1000u + row * 100u + col);
            }
        }
    }
    for (uint64_t i = 0; i < out_count; i++) out_host[i] = -1.0f;

    TEST_ASSERT(ds4_gpu_tensor_write(slots, 0, slot_host, slot_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_pack_slot_rows_f32_tensor(out,
                                                  slots,
                                                  n_rows,
                                                  width,
                                                  n_slots,
                                                  slot_cap) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t slot = 0; slot < n_slots; slot++) {
            for (uint32_t col = 0; col < width; col++) {
                const float ref =
                    slot_host[((uint64_t)slot * slot_cap + row) * width + col];
                const float got =
                    out_host[((uint64_t)row * n_slots + slot) * width + col];
                TEST_ASSERT(got == ref);
            }
        }
    }

    free(slot_host);
    free(out_host);
    ds4_gpu_tensor_free(slots);
    ds4_gpu_tensor_free(out);
}

static void test_metal_store_raw_kv_batch_wrap(void) {
    const uint32_t raw_cap = 5;
    const uint32_t head_dim = 3;
    const uint32_t n_tokens = 4;
    const uint32_t pos0 = 3;
    const uint64_t kv_count = (uint64_t)n_tokens * head_dim;
    const uint64_t raw_count = (uint64_t)raw_cap * head_dim;
    const uint64_t kv_bytes = kv_count * sizeof(float);
    const uint64_t raw_bytes = raw_count * sizeof(float);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(kv_bytes);
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(raw != NULL);
    if (!kv || !raw) {
        ds4_gpu_tensor_free(kv);
        ds4_gpu_tensor_free(raw);
        return;
    }

    float kv_host[12];
    float raw_host[15];
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t d = 0; d < head_dim; d++) {
            kv_host[(uint64_t)t * head_dim + d] = (float)(100u * t + d);
        }
    }
    for (uint64_t i = 0; i < raw_count; i++) raw_host[i] = -1.0f;

    TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, kv_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
    TEST_ASSERT(ds4_gpu_store_raw_kv_batch_tensor(raw,
                                                  kv,
                                                  raw_cap,
                                                  pos0,
                                                  n_tokens,
                                                  head_dim) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(raw, 0, raw_host, raw_bytes) != 0);

    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t row = (pos0 + t) % raw_cap;
        for (uint32_t d = 0; d < head_dim; d++) {
            const float ref = kv_host[(uint64_t)t * head_dim + d];
            const float got = raw_host[(uint64_t)row * head_dim + d];
            TEST_ASSERT(got == ref);
        }
    }
    for (uint32_t d = 0; d < head_dim; d++) {
        TEST_ASSERT(raw_host[(uint64_t)2u * head_dim + d] == -1.0f);
    }

    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(raw);
}

static void test_dspark_cache_window_crop(void) {
    TEST_ASSERT(ds4_test_dspark_cache_window_crop());
}

static void test_metal_q8_0_decode_pair_exact_case(
        uint32_t out0_dim,
        uint32_t out1_dim,
        uint32_t seed0,
        uint32_t seed1) {
    /* Exercise the Q-A/KV contract with unequal, odd output extents and
     * independently page-aligned model ranges. The paired kernel must be
     * bit-identical to two standalone decode matvec dispatches. */
    const uint32_t in_dim = 4096;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight0_bytes = (uint64_t)out0_dim * row_bytes;
    const uint64_t weight1_bytes = (uint64_t)out1_dim * row_bytes;
    const uint64_t weight1_offset = test_round_up_u64(weight0_bytes, page);
    const uint64_t weight_alloc =
        test_round_up_u64(weight1_offset + weight1_bytes, page);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)page, (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    memset(weights_raw, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights((uint8_t *)weights_raw, in_dim, out0_dim, seed0);
    test_fill_q8_0_weights((uint8_t *)weights_raw + weight1_offset,
                           in_dim, out1_dim, seed1);

    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out0_bytes = (uint64_t)out0_dim * sizeof(float);
    const uint64_t out1_bytes = (uint64_t)out1_dim * sizeof(float);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref0 = ds4_gpu_tensor_alloc(out0_bytes);
    ds4_gpu_tensor *ref1 = ds4_gpu_tensor_alloc(out1_bytes);
    ds4_gpu_tensor *pair0 = ds4_gpu_tensor_alloc(out0_bytes);
    ds4_gpu_tensor *pair1 = ds4_gpu_tensor_alloc(out1_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref0 != NULL);
    TEST_ASSERT(ref1 != NULL);
    TEST_ASSERT(pair0 != NULL);
    TEST_ASSERT(pair1 != NULL);
    if (!x || !ref0 || !ref1 || !pair0 || !pair1) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(ref0);
        ds4_gpu_tensor_free(ref1);
        ds4_gpu_tensor_free(pair0);
        ds4_gpu_tensor_free(pair1);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *ref0_host = malloc((size_t)out0_bytes);
    float *ref1_host = malloc((size_t)out1_bytes);
    float *pair0_host = malloc((size_t)out0_bytes);
    float *pair1_host = malloc((size_t)out1_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref0_host != NULL);
    TEST_ASSERT(ref1_host != NULL);
    TEST_ASSERT(pair0_host != NULL);
    TEST_ASSERT(pair1_host != NULL);
    if (!x_host || !ref0_host || !ref1_host || !pair0_host || !pair1_host) {
        free(x_host);
        free(ref0_host);
        free(ref1_host);
        free(pair0_host);
        free(pair1_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(ref0);
        ds4_gpu_tensor_free(ref1);
        ds4_gpu_tensor_free(pair0);
        ds4_gpu_tensor_free(pair1);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        const int v = (int)((i * 29u + (i ^ (i >> 3u)) * 7u) % 127u) - 63;
        x_host[i] = (float)v / 72.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(ref0, weights_raw, weight_alloc, 0,
                                           in_dim, out0_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(ref1, weights_raw, weight_alloc,
                                           weight1_offset,
                                           in_dim, out1_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_pair_tensor(pair0, pair1,
                                                weights_raw, weight_alloc,
                                                0, weight1_offset,
                                                in_dim, out0_dim, out1_dim,
                                                x, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref0, 0, ref0_host, out0_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref1, 0, ref1_host, out1_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(pair0, 0, pair0_host, out0_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(pair1, 0, pair1_host, out1_bytes) != 0);

    uint32_t mismatch0 = 0;
    uint32_t mismatch1 = 0;
    float max_abs0 = 0.0f;
    float max_abs1 = 0.0f;
    for (uint32_t i = 0; i < out0_dim; i++) {
        if (memcmp(&ref0_host[i], &pair0_host[i], sizeof(float)) != 0) mismatch0++;
        const float err = fabsf(ref0_host[i] - pair0_host[i]);
        if (err > max_abs0) max_abs0 = err;
    }
    for (uint32_t i = 0; i < out1_dim; i++) {
        if (memcmp(&ref1_host[i], &pair1_host[i], sizeof(float)) != 0) mismatch1++;
        const float err = fabsf(ref1_host[i] - pair1_host[i]);
        if (err > max_abs1) max_abs1 = err;
    }
    if (mismatch0 != 0 || mismatch1 != 0) {
        fprintf(stderr,
                "ds4-test: paired Q8_0 exactness mismatches=%u/%u max_abs=%g, %u/%u max_abs=%g\n",
                mismatch0, out0_dim, max_abs0,
                mismatch1, out1_dim, max_abs1);
    }
    TEST_ASSERT(mismatch0 == 0);
    TEST_ASSERT(mismatch1 == 0);

    free(x_host);
    free(ref0_host);
    free(ref1_host);
    free(pair0_host);
    free(pair1_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(ref0);
    ds4_gpu_tensor_free(ref1);
    ds4_gpu_tensor_free(pair0);
    ds4_gpu_tensor_free(pair1);
    free(weights_raw);
}

static void test_metal_q8_0_decode_pair_exact(void) {
    /* Cover both possible one-bank tail directions. Distinct seeds ensure a
     * mistaken A-for-B weight binding cannot compare equal by construction. */
    test_metal_q8_0_decode_pair_exact_case(77, 19, 11, 97);
    test_metal_q8_0_decode_pair_exact_case(19, 77, 23, 131);
}

