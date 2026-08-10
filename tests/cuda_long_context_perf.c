#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static float qat_value(uint64_t i) {
    static const float levels[] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float scale = ldexpf(1.0f, (int)((i / 32u) % 7u) - 4);
    const float sign = ((i * 0x9e3779b97f4a7c15ull) >> 63) ? -1.0f : 1.0f;
    return sign * scale * levels[(i * 13u + i / 17u) & 7u];
}

static int compare_f32(const char *label, const float *a, const float *b,
                       uint64_t n, float atol, float rtol) {
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    uint64_t bad = 0;
    for (uint64_t i = 0; i < n; i++) {
        const float abs_err = fabsf(a[i] - b[i]);
        const float denom = fmaxf(fmaxf(fabsf(a[i]), fabsf(b[i])), 1.0e-6f);
        const float rel_err = abs_err / denom;
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel_err > max_rel) max_rel = rel_err;
        sum_sq += (double)abs_err * abs_err;
        if (abs_err > atol + rtol * denom) bad++;
    }
    fprintf(stderr,
            "%s: max_abs=%g max_rel=%g rmse=%g bad=%llu/%llu\n",
            label, (double)max_abs, (double)max_rel,
            sqrt(sum_sq / (double)n),
            (unsigned long long)bad, (unsigned long long)n);
    return bad == 0 ? 0 : 1;
}

static double run_indexer_once(ds4_gpu_tensor *scores,
                               ds4_gpu_tensor *selected,
                               const ds4_gpu_tensor *q,
                               const ds4_gpu_tensor *weights,
                               const ds4_gpu_tensor *cache,
                               uint32_t n_comp, uint32_t iters) {
    const uint32_t pos0 = n_comp * 4u - 1u;
    const float scale = 1.0f / sqrtf(128.0f * 64.0f);
    if (!ds4_gpu_indexer_scores_decode_batch_tensor(
            scores, q, weights, cache, n_comp, 1u, pos0,
            64u, 128u, 4u, scale) ||
        !ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, 1u, 512u) ||
        !ds4_gpu_synchronize()) {
        return -1.0;
    }
    const double t0 = monotonic_seconds();
    for (uint32_t i = 0; i < iters; i++) {
        if (!ds4_gpu_indexer_scores_decode_batch_tensor(
                scores, q, weights, cache, n_comp, 1u, pos0,
                64u, 128u, 4u, scale) ||
            !ds4_gpu_indexer_topk_tensor(
                selected, scores, n_comp, 1u, 512u)) {
            return -1.0;
        }
    }
    if (!ds4_gpu_synchronize()) return -1.0;
    return (monotonic_seconds() - t0) * 1000.0 / iters;
}

static int run_indexer(uint32_t n_comp, uint32_t iters) {
    const uint64_t q_count = 64u * 128u;
    const uint64_t cache_count = (uint64_t)n_comp * 128u;
    float *q_host = (float *)malloc(q_count * sizeof(float));
    float *weights_host = (float *)malloc(64u * sizeof(float));
    float *cache_host = (float *)malloc(cache_count * sizeof(float));
    float *ref_host = (float *)malloc((uint64_t)n_comp * sizeof(float));
    float *opt_host = (float *)malloc((uint64_t)n_comp * sizeof(float));
    uint32_t *ref_topk = (uint32_t *)malloc(512u * sizeof(uint32_t));
    uint32_t *opt_topk = (uint32_t *)malloc(512u * sizeof(uint32_t));
    if (!q_host || !weights_host || !cache_host || !ref_host || !opt_host ||
        !ref_topk || !opt_topk) {
        return 1;
    }
    for (uint64_t i = 0; i < q_count; i++) q_host[i] = qat_value(i + 7u);
    for (uint32_t i = 0; i < 64u; i++) {
        weights_host[i] = 0.25f + (float)((i * 19u) % 31u) / 31.0f;
    }
    for (uint64_t i = 0; i < cache_count; i++) {
        cache_host[i] = qat_value(i + (i / 128u) * 97u + 23u);
    }

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(64u * sizeof(float));
    ds4_gpu_tensor *cache = ds4_gpu_tensor_alloc(cache_count * sizeof(float));
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(512u * sizeof(uint32_t));
    int rc = 1;
    if (!q || !weights || !cache || !scores || !selected ||
        !ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(weights, 0, weights_host, 64u * sizeof(float)) ||
        !ds4_gpu_tensor_write(cache, 0, cache_host,
                              cache_count * sizeof(float))) {
        goto cleanup;
    }

    setenv("DS4_CUDA_NO_INDEXER_MXF4", "1", 1);
    setenv("DS4_CUDA_NO_INDEXER_WMMA", "1", 1);
    unsetenv("DS4_CUDA_NO_INDEXER_DIRECT_ONE");
    const double ref_ms = run_indexer_once(
        scores, selected, q, weights, cache, n_comp, iters);
    if (ref_ms < 0.0 ||
        !ds4_gpu_tensor_read(scores, 0, ref_host,
                             (uint64_t)n_comp * sizeof(float)) ||
        !ds4_gpu_tensor_read(selected, 0, ref_topk, 512u * sizeof(uint32_t))) {
        goto cleanup;
    }

    unsetenv("DS4_CUDA_NO_INDEXER_WMMA");
    const double opt_ms = run_indexer_once(
        scores, selected, q, weights, cache, n_comp, iters);
    if (opt_ms < 0.0 ||
        !ds4_gpu_tensor_read(scores, 0, opt_host,
                             (uint64_t)n_comp * sizeof(float)) ||
        !ds4_gpu_tensor_read(selected, 0, opt_topk, 512u * sizeof(uint32_t))) {
        goto cleanup;
    }

    uint32_t topk_diff = 0;
    for (uint32_t i = 0; i < 512u; i++) {
        if (ref_topk[i] != opt_topk[i]) topk_diff++;
    }
    fprintf(stderr,
            "indexer n_comp=%u direct=%.6f ms wmma=%.6f ms speedup=%.3fx topk_diff=%u\n",
            n_comp, ref_ms, opt_ms, ref_ms / opt_ms, topk_diff);
    rc = compare_f32("indexer scores", ref_host, opt_host, n_comp,
                     2.0e-4f, 2.0e-4f);

cleanup:
    unsetenv("DS4_CUDA_NO_INDEXER_DIRECT_ONE");
    unsetenv("DS4_CUDA_NO_INDEXER_WMMA");
    unsetenv("DS4_CUDA_NO_INDEXER_MXF4");
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    ds4_gpu_tensor_free(cache);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(q);
    free(opt_topk);
    free(ref_topk);
    free(opt_host);
    free(ref_host);
    free(cache_host);
    free(weights_host);
    free(q_host);
    return rc;
}

static double run_hca_once(ds4_gpu_tensor *heads,
                           const float *sinks,
                           const ds4_gpu_tensor *q,
                           const ds4_gpu_tensor *raw,
                           const ds4_gpu_tensor *comp,
                           uint32_t n_comp, uint32_t iters) {
    if (!ds4_gpu_attention_decode_heads_tensor(
            heads, sinks, 64u * sizeof(float), 0, q, raw,
            128u, 128u, 0u, comp, 0u, n_comp, NULL, 0u, 64u, 512u) ||
        !ds4_gpu_synchronize()) {
        return -1.0;
    }
    const double t0 = monotonic_seconds();
    for (uint32_t i = 0; i < iters; i++) {
        if (!ds4_gpu_attention_decode_heads_tensor(
                heads, sinks, 64u * sizeof(float), 0, q, raw,
                128u, 128u, 0u, comp, 0u, n_comp, NULL, 0u, 64u, 512u)) {
            return -1.0;
        }
    }
    if (!ds4_gpu_synchronize()) return -1.0;
    return (monotonic_seconds() - t0) * 1000.0 / iters;
}

static int run_hca(uint32_t n_comp, uint32_t iters) {
    const uint64_t head_count = 64u * 512u;
    const uint64_t raw_count = 128u * 512u;
    const uint64_t comp_count = (uint64_t)n_comp * 512u;
    float *sinks = (float *)calloc(64u, sizeof(float));
    float *q_host = (float *)malloc(head_count * sizeof(float));
    float *raw_host = (float *)malloc(raw_count * sizeof(float));
    float *comp_host = (float *)malloc(comp_count * sizeof(float));
    float *ref_host = (float *)malloc(head_count * sizeof(float));
    float *opt_host = (float *)malloc(head_count * sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !ref_host || !opt_host) {
        return 1;
    }
    for (uint64_t i = 0; i < head_count; i++) {
        q_host[i] = sinf((float)(i % 997u) * 0.013f) * 0.125f;
    }
    for (uint64_t i = 0; i < raw_count; i++) {
        raw_host[i] = cosf((float)(i % 991u) * 0.017f) * 0.125f;
    }
    for (uint64_t i = 0; i < comp_count; i++) {
        comp_host[i] = sinf((float)(i % 983u) * 0.019f) * 0.125f;
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(head_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(head_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    int rc = 1;
    if (!heads || !q || !raw || !comp ||
        !ds4_gpu_tensor_write(q, 0, q_host, head_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(raw, 0, raw_host, raw_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(comp, 0, comp_host, comp_count * sizeof(float))) {
        goto cleanup;
    }

    setenv("DS4_CUDA_NO_SPLITKV_DECODE", "1", 1);
    const double ref_ms = run_hca_once(
        heads, sinks, q, raw, comp, n_comp, iters);
    if (ref_ms < 0.0 ||
        !ds4_gpu_tensor_read(heads, 0, ref_host,
                             head_count * sizeof(float))) {
        goto cleanup;
    }
    unsetenv("DS4_CUDA_NO_SPLITKV_DECODE");
    const double opt_ms = run_hca_once(
        heads, sinks, q, raw, comp, n_comp, iters);
    if (opt_ms < 0.0 ||
        !ds4_gpu_tensor_read(heads, 0, opt_host,
                             head_count * sizeof(float))) {
        goto cleanup;
    }
    fprintf(stderr,
            "hca n_comp=%u old=%.6f ms new=%.6f ms speedup=%.3fx\n",
            n_comp, ref_ms, opt_ms, ref_ms / opt_ms);
    rc = compare_f32("hca output", ref_host, opt_host, head_count,
                     5.0e-4f, 5.0e-4f);

cleanup:
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(opt_host);
    free(ref_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "all";
    const uint32_t n_comp = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10)
                                     : 262144u;
    const uint32_t iters = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10)
                                    : 20u;
    if (!ds4_gpu_init()) return 1;
    ds4_gpu_set_quality(false);
    int rc = 0;
    if (strcmp(mode, "indexer") == 0 || strcmp(mode, "all") == 0) {
        if (run_indexer(n_comp, iters) != 0) rc = 1;
    }
    if (strcmp(mode, "hca") == 0 || strcmp(mode, "all") == 0) {
        const uint32_t hca_comp = strcmp(mode, "hca") == 0 ? n_comp : 8192u;
        if (run_hca(hca_comp, iters) != 0) rc = 1;
    }
    ds4_gpu_cleanup();
    return rc;
}
