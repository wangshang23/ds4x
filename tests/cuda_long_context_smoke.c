#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static double getenv_seconds(const char *name, double fallback) {
    const char *s = getenv(name);
    if (!s || !s[0]) return fallback;
    char *end = NULL;
    const double v = strtod(s, &end);
    return end != s && v > 0.0 ? v : fallback;
}

static float qat_value(uint64_t i) {
    static const float levels[] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float scale = ldexpf(1.0f, (int)((i / 32u) % 7u) - 4);
    const float sign = ((i * 0x9e3779b97f4a7c15ull) >> 63) ? -1.0f : 1.0f;
    return sign * scale * levels[(i * 13u + i / 17u) & 7u];
}

static int check_large_topk(void) {
    const uint32_t n_comp = 32768;
    const uint32_t n_tokens = 32;
    const uint32_t top_k = 512;
    const uint64_t score_count = (uint64_t)n_comp * n_tokens;
    float *scores_host = (float *)malloc((size_t)score_count * sizeof(float));
    uint32_t *selected_host = (uint32_t *)malloc((size_t)n_tokens * top_k * sizeof(uint32_t));
    if (!scores_host || !selected_host) return 1;

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t i = 0; i < n_comp; i++) {
            scores_host[(uint64_t)t * n_comp + i] = (float)i;
        }
    }

    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(score_count * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
    int rc = 1;
    double elapsed = 0.0;
    if (scores && selected &&
        ds4_gpu_tensor_write(scores, 0, scores_host, score_count * sizeof(float))) {
        /* Exclude one-time CUDA module/kernel setup from the throughput guard. */
        if (!ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) ||
            !ds4_gpu_synchronize()) {
            rc = 1;
            goto cleanup;
        }
        const double t0 = monotonic_seconds();
        if (ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, n_tokens, top_k) &&
            ds4_gpu_synchronize()) {
            elapsed = monotonic_seconds() - t0;
            rc = ds4_gpu_tensor_read(selected, 0, selected_host,
                                     (uint64_t)n_tokens * top_k * sizeof(uint32_t)) ? 0 : 1;
        }
    }
    if (rc == 0) {
        for (uint32_t t = 0; t < n_tokens && rc == 0; t++) {
            for (uint32_t i = 0; i < top_k; i++) {
                const uint32_t expected = n_comp - 1u - i;
                const uint32_t got = selected_host[(uint64_t)t * top_k + i];
                if (got != expected) {
                    fprintf(stderr, "top-k mismatch token=%u rank=%u got=%u expected=%u\n",
                            t, i, got, expected);
                    rc = 1;
                    break;
                }
            }
        }
    }
    if (rc == 0) {
        const double max_seconds = getenv_seconds("DS4_CUDA_TOPK_REGRESSION_SEC", 2.0);
        fprintf(stderr, "cuda-regression: top-k n_comp=%u n_tokens=%u elapsed=%.3fs\n",
                n_comp, n_tokens, elapsed);
        if (elapsed > max_seconds) {
            fprintf(stderr, "top-k regression: %.3fs exceeds %.3fs\n", elapsed, max_seconds);
            rc = 1;
        }
    }

cleanup:
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    free(selected_host);
    free(scores_host);
    return rc;
}

static int check_decode_attention_overflow_path(void) {
    const uint32_t n_head = 8;
    const uint32_t head_dim = 512;
    const uint32_t n_raw = 128;
    const uint32_t n_comp = 8192;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;

    float *sinks = (float *)calloc(n_head, sizeof(float));
    float *q_host = (float *)calloc((size_t)q_count, sizeof(float));
    float *raw_host = (float *)calloc((size_t)raw_count, sizeof(float));
    float *comp_host = (float *)calloc((size_t)comp_count, sizeof(float));
    float *heads_host = (float *)calloc((size_t)q_count, sizeof(float));
    if (!sinks || !q_host || !raw_host || !comp_host || !heads_host) return 1;

    for (uint32_t c = 0; c < n_comp; c++) {
        comp_host[(uint64_t)c * head_dim] = 1.0f;
    }

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw_src = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp_src = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc((uint64_t)n_raw * DS4_SPARK_KV_ROW_BYTES);
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc((uint64_t)n_comp * DS4_SPARK_KV_ROW_BYTES);
    int rc = 1;
    if (heads && q && raw_src && comp_src && raw && comp &&
        ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) &&
        ds4_gpu_tensor_write(raw_src, 0, raw_host, raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp_src, 0, comp_host, comp_count * sizeof(float)) &&
        ds4_gpu_spark_pack_kv_rows_tensor(raw, 0, raw_src, 0, n_raw) &&
        ds4_gpu_spark_pack_kv_rows_tensor(comp, 0, comp_src, 0, n_comp) &&
        ds4_gpu_attention_decode_heads_tensor(heads,
                                              sinks,
                                              n_head * sizeof(float),
                                              0,
                                              q,
                                              raw,
                                              n_raw,
                                              n_raw,
                                              0,
                                              comp,
                                              DS4_GPU_CACHE_SPARK_KV,
                                              n_comp,
                                              NULL,
                                              0,
                                              n_head,
                                              head_dim) &&
        ds4_gpu_synchronize() &&
        ds4_gpu_tensor_read(heads, 0, heads_host, q_count * sizeof(float))) {
        rc = 0;
        for (uint32_t h = 0; h < n_head; h++) {
            const float v = heads_host[(uint64_t)h * head_dim];
            if (v < 0.90f) {
                fprintf(stderr, "long attention ignored compressed rows for head=%u value=%f\n",
                        h, (double)v);
                rc = 1;
            }
        }
    }

    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(comp_src);
    ds4_gpu_tensor_free(raw_src);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    free(heads_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

static int check_b1_indexer_wmma(void) {
    const uint32_t n_comp = 8192u;
    const uint32_t n_head = 64u;
    const uint32_t head_dim = 128u;
    const uint32_t top_k = 512u;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t cache_count = (uint64_t)n_comp * head_dim;
    float *q_host = malloc((size_t)q_count * sizeof(*q_host));
    float *weights_host = malloc((size_t)n_head * sizeof(*weights_host));
    float *cache_host = malloc((size_t)cache_count * sizeof(*cache_host));
    float *direct_host = malloc((size_t)n_comp * sizeof(*direct_host));
    float *wmma_host = malloc((size_t)n_comp * sizeof(*wmma_host));
    uint32_t *direct_topk = malloc((size_t)top_k * sizeof(*direct_topk));
    uint32_t *wmma_topk = malloc((size_t)top_k * sizeof(*wmma_topk));
    if (!q_host || !weights_host || !cache_host || !direct_host ||
        !wmma_host || !direct_topk || !wmma_topk) {
        free(wmma_topk);
        free(direct_topk);
        free(wmma_host);
        free(direct_host);
        free(cache_host);
        free(weights_host);
        free(q_host);
        return 1;
    }

    for (uint64_t i = 0; i < q_count; i++) {
        q_host[i] = qat_value(i + 7u);
    }
    for (uint32_t i = 0; i < n_head; i++) {
        weights_host[i] = 0.25f + (float)((i * 19u) % 31u) / 31.0f;
    }
    for (uint64_t i = 0; i < cache_count; i++) {
        cache_host[i] = qat_value(i + (i / head_dim) * 97u + 23u);
    }

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *q_packed = ds4_gpu_tensor_alloc(
        (uint64_t)n_head * DS4_SPARK_INDEX_ROW_BYTES);
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(n_head * sizeof(float));
    ds4_gpu_tensor *cache_src = ds4_gpu_tensor_alloc(cache_count * sizeof(float));
    ds4_gpu_tensor *cache = ds4_gpu_tensor_alloc((uint64_t)n_comp * DS4_SPARK_INDEX_ROW_BYTES);
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc(n_comp * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(top_k * sizeof(uint32_t));
    int rc = 1;
    if (!q || !q_packed || !weights || !cache_src || !cache || !scores || !selected ||
        !ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(weights, 0, weights_host,
                              n_head * sizeof(float)) ||
        !ds4_gpu_tensor_write(cache_src, 0, cache_host,
                              cache_count * sizeof(float)) ||
        !ds4_gpu_spark_pack_index_rows_tensor(q_packed, 0, q, 0, n_head) ||
        !ds4_gpu_spark_unpack_index_rows_tensor(q, q_packed, n_head) ||
        !ds4_gpu_spark_pack_index_rows_tensor(cache, 0, cache_src, 0, n_comp) ||
        !ds4_gpu_spark_unpack_index_rows_tensor(cache_src, cache, n_comp) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(q, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(cache_src, 0, cache_host,
                             cache_count * sizeof(float))) {
        goto cleanup;
    }

    const float score_scale = 1.0f / sqrtf(8192.0f);
    for (uint32_t c = 0; c < n_comp; c++) {
        float total = 0.0f;
        for (uint32_t h = 0; h < n_head; h++) {
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                dot += q_host[(uint64_t)h * head_dim + d] *
                       cache_host[(uint64_t)c * head_dim + d];
            }
            total += fmaxf(dot, 0.0f) * weights_host[h];
        }
        direct_host[c] = total * score_scale;
    }
    for (uint32_t k = 0; k < top_k; k++) direct_topk[k] = UINT32_MAX;
    for (uint32_t c = 0; c < n_comp; c++) {
        for (uint32_t k = 0; k < top_k; k++) {
            const uint32_t old = direct_topk[k];
            if (old == UINT32_MAX || direct_host[c] > direct_host[old] ||
                (direct_host[c] == direct_host[old] && c < old)) {
                for (uint32_t j = top_k - 1u; j > k; j--) {
                    direct_topk[j] = direct_topk[j - 1u];
                }
                direct_topk[k] = c;
                break;
            }
        }
    }
    if (!ds4_gpu_indexer_score_one_tensor(scores, q, weights, cache,
                                          n_comp, n_head, head_dim,
                                          1.0f / sqrtf(8192.0f)) ||
        !ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, 1u, top_k) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(scores, 0, wmma_host,
                             n_comp * sizeof(float)) ||
        !ds4_gpu_tensor_read(selected, 0, wmma_topk,
                             top_k * sizeof(uint32_t))) {
        goto cleanup;
    }

    float max_abs = 0.0f;
    for (uint32_t i = 0; i < n_comp; i++) {
        const float delta = fabsf(direct_host[i] - wmma_host[i]);
        if (delta > max_abs) max_abs = delta;
    }
    uint32_t topk_diff = 0u;
    for (uint32_t i = 0; i < top_k; i++) {
        if (direct_topk[i] != wmma_topk[i]) topk_diff++;
    }
    fprintf(stderr,
            "cuda-regression: B1 indexer n_comp=%u max_abs=%g topk_diff=%u\n",
            n_comp, (double)max_abs, topk_diff);
    rc = max_abs <= 5.0e-3f && topk_diff == 0u ? 0 : 1;

cleanup:
    unsetenv("DS4_CUDA_NO_INDEXER_WMMA");
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    ds4_gpu_tensor_free(cache);
    ds4_gpu_tensor_free(cache_src);
    ds4_gpu_tensor_free(q_packed);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(q);
    free(wmma_topk);
    free(direct_topk);
    free(wmma_host);
    free(direct_host);
    free(cache_host);
    free(weights_host);
    free(q_host);
    return rc;
}

static int check_packed_attention_parity(void) {
    const uint32_t n_head = 8u;
    const uint32_t head_dim = 512u;
    const uint32_t n_raw = 128u;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    const uint64_t kv_count = (uint64_t)n_raw * head_dim;
    float *sinks = calloc(n_head, sizeof(*sinks));
    float *q_host = malloc((size_t)q_count * sizeof(*q_host));
    float *kv_host = malloc((size_t)kv_count * sizeof(*kv_host));
    float *ref_host = malloc((size_t)q_count * sizeof(*ref_host));
    float *packed_host = malloc((size_t)q_count * sizeof(*packed_host));
    if (!sinks || !q_host || !kv_host || !ref_host || !packed_host) return 1;
    for (uint64_t i = 0; i < q_count; i++) q_host[i] = qat_value(i + 11u) * 0.125f;
    for (uint64_t i = 0; i < kv_count; i++) kv_host[i] = qat_value(i + 29u);

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *kv_f32 = ds4_gpu_tensor_alloc(kv_count * sizeof(float));
    ds4_gpu_tensor *kv_packed = ds4_gpu_tensor_alloc(
        (uint64_t)n_raw * DS4_SPARK_KV_ROW_BYTES);
    ds4_gpu_tensor *heads_ref = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *heads_packed = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    int rc = 1;
    if (!q || !kv_f32 || !kv_packed || !heads_ref || !heads_packed ||
        !ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(kv_f32, 0, kv_host, kv_count * sizeof(float)) ||
        !ds4_gpu_spark_pack_kv_rows_tensor(kv_packed, 0, kv_f32, 0, n_raw) ||
        !ds4_gpu_spark_unpack_kv_rows_tensor(kv_f32, kv_packed, n_raw) ||
        !ds4_gpu_attention_decode_heads_tensor(
            heads_ref, sinks, n_head * sizeof(float), 0, q, kv_f32,
            n_raw, n_raw, 0, NULL, DS4_GPU_CACHE_F32, 0,
            NULL, 0, n_head, head_dim) ||
        !ds4_gpu_attention_decode_heads_tensor(
            heads_packed, sinks, n_head * sizeof(float), 0, q, kv_packed,
            n_raw, n_raw, 0, NULL, DS4_GPU_CACHE_SPARK_KV, 0,
            NULL, 0, n_head, head_dim) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads_ref, 0, ref_host,
                             q_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(heads_packed, 0, packed_host,
                             q_count * sizeof(float))) {
        goto cleanup;
    }
    float max_abs = 0.0f;
    double sum_sq = 0.0;
    for (uint64_t i = 0; i < q_count; i++) {
        const float delta = fabsf(ref_host[i] - packed_host[i]);
        if (delta > max_abs) max_abs = delta;
        sum_sq += (double)delta * delta;
    }
    fprintf(stderr, "cuda-regression: packed attention max_abs=%g rmse=%g\n",
            (double)max_abs, sqrt(sum_sq / (double)q_count));
    rc = max_abs <= 1.0e-5f ? 0 : 1;

cleanup:
    ds4_gpu_tensor_free(heads_packed);
    ds4_gpu_tensor_free(heads_ref);
    ds4_gpu_tensor_free(kv_packed);
    ds4_gpu_tensor_free(kv_f32);
    ds4_gpu_tensor_free(q);
    free(packed_host);
    free(ref_host);
    free(kv_host);
    free(q_host);
    free(sinks);
    return rc;
}

static int check_packed_batch_attention_exact_parity(void) {
    const uint32_t n_tokens = 29u;
    const uint32_t pos0 = 3u;
    const uint32_t n_head = 8u;
    const uint32_t head_dim = 512u;
    const uint32_t n_raw = pos0 + n_tokens;
    const uint32_t n_comp = n_raw / 4u;
    const uint64_t q_stride = (uint64_t)n_head * head_dim;
    const uint64_t q_count = (uint64_t)n_tokens * q_stride;
    const uint64_t raw_count = (uint64_t)n_raw * head_dim;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    float *sinks = calloc(n_head, sizeof(*sinks));
    float *q_host = malloc((size_t)q_count * sizeof(*q_host));
    float *raw_host = malloc((size_t)raw_count * sizeof(*raw_host));
    float *comp_host = malloc((size_t)comp_count * sizeof(*comp_host));
    float *ref_host = malloc((size_t)q_count * sizeof(*ref_host));
    float *batch_host = malloc((size_t)q_count * sizeof(*batch_host));
    if (!sinks || !q_host || !raw_host || !comp_host ||
        !ref_host || !batch_host) return 1;

    for (uint64_t i = 0; i < q_count; i++) q_host[i] = qat_value(i + 101u);
    for (uint64_t i = 0; i < raw_count; i++) raw_host[i] = qat_value(i + 1009u);
    for (uint64_t i = 0; i < comp_count; i++) comp_host[i] = qat_value(i + 10007u);

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *raw_src = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    ds4_gpu_tensor *comp_src = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(
        (uint64_t)n_raw * DS4_SPARK_KV_ROW_BYTES);
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(
        (uint64_t)n_comp * DS4_SPARK_KV_ROW_BYTES);
    ds4_gpu_tensor *heads_ref = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *heads_batch = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    int rc = 1;
    if (!q || !raw_src || !comp_src || !raw || !comp ||
        !heads_ref || !heads_batch ||
        !ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(raw_src, 0, raw_host, raw_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(comp_src, 0, comp_host, comp_count * sizeof(float)) ||
        !ds4_gpu_spark_pack_kv_rows_tensor(raw, 0, raw_src, 0, n_raw) ||
        !ds4_gpu_spark_pack_kv_rows_tensor(comp, 0, comp_src, 0, n_comp)) {
        goto cleanup;
    }

    setenv("DS4_CUDA_SPARK_PREFILL_EXACT", "1", 1);
    if (!ds4_gpu_attention_decode_mixed_batch_heads_tensor(
            heads_batch, sinks, n_head * sizeof(float), 0,
            q, raw, comp, DS4_GPU_CACHE_SPARK_KV, NULL, 0,
            n_tokens, pos0, n_raw, n_raw, 0, n_comp,
            128u, 4u, n_head, head_dim)) {
        goto cleanup;
    }
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t qpos = pos0 + t;
        const uint32_t visible_comp = (qpos + 1u) / 4u;
        ds4_gpu_tensor *q_row = ds4_gpu_tensor_view(
            q, (uint64_t)t * q_stride * sizeof(float),
            q_stride * sizeof(float));
        ds4_gpu_tensor *head_row = ds4_gpu_tensor_view(
            heads_ref, (uint64_t)t * q_stride * sizeof(float),
            q_stride * sizeof(float));
        if (!q_row || !head_row ||
            !ds4_gpu_attention_decode_heads_tensor(
                head_row, sinks, n_head * sizeof(float), 0,
                q_row, raw, qpos + 1u, n_raw, 0,
                comp, DS4_GPU_CACHE_SPARK_KV, visible_comp,
                NULL, 0, n_head, head_dim)) {
            ds4_gpu_tensor_free(head_row);
            ds4_gpu_tensor_free(q_row);
            goto cleanup;
        }
        ds4_gpu_tensor_free(head_row);
        ds4_gpu_tensor_free(q_row);
    }
    if (!ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads_ref, 0, ref_host,
                             q_count * sizeof(float)) ||
        !ds4_gpu_tensor_read(heads_batch, 0, batch_host,
                             q_count * sizeof(float))) {
        goto cleanup;
    }

    float max_abs = 0.0f;
    double sum_sq = 0.0;
    for (uint64_t i = 0; i < q_count; i++) {
        const float delta = fabsf(ref_host[i] - batch_host[i]);
        if (delta > max_abs) max_abs = delta;
        sum_sq += (double)delta * delta;
    }
    fprintf(stderr,
            "cuda-regression: packed batch exact max_abs=%g rmse=%g\n",
            (double)max_abs, sqrt(sum_sq / (double)q_count));
    rc = max_abs == 0.0f ? 0 : 1;

cleanup:
    unsetenv("DS4_CUDA_SPARK_PREFILL_EXACT");
    ds4_gpu_tensor_free(heads_batch);
    ds4_gpu_tensor_free(heads_ref);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(comp_src);
    ds4_gpu_tensor_free(raw_src);
    ds4_gpu_tensor_free(q);
    free(batch_host);
    free(ref_host);
    free(comp_host);
    free(raw_host);
    free(q_host);
    free(sinks);
    return rc;
}

static int check_large_packed_indexer_zero(void) {
    const uint32_t n_comp = 262144u;
    const uint32_t n_head = 64u;
    const uint32_t head_dim = 128u;
    const uint32_t top_k = 512u;
    const uint64_t q_count = (uint64_t)n_head * head_dim;
    float *q_host = malloc((size_t)q_count * sizeof(*q_host));
    float *weights_host = malloc((size_t)n_head * sizeof(*weights_host));
    float *scores_host = malloc((size_t)n_comp * sizeof(*scores_host));
    uint32_t *topk_host = malloc((size_t)top_k * sizeof(*topk_host));
    if (!q_host || !weights_host || !scores_host || !topk_host) return 1;
    for (uint64_t i = 0; i < q_count; i++) q_host[i] = qat_value(i + 41u);
    for (uint32_t h = 0; h < n_head; h++) weights_host[h] = 1.0f;
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_count * sizeof(float));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(n_head * sizeof(float));
    ds4_gpu_tensor *cache = ds4_gpu_tensor_alloc(
        (uint64_t)n_comp * DS4_SPARK_INDEX_ROW_BYTES);
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)top_k * sizeof(uint32_t));
    int rc = 1;
    if (!q || !weights || !cache || !scores || !selected ||
        !ds4_gpu_tensor_write(q, 0, q_host, q_count * sizeof(float)) ||
        !ds4_gpu_tensor_write(weights, 0, weights_host,
                              n_head * sizeof(float)) ||
        !ds4_gpu_spark_zero_index_rows_tensor(cache, n_comp) ||
        !ds4_gpu_indexer_score_one_tensor(scores, q, weights, cache,
                                          n_comp, n_head, head_dim,
                                          1.0f / sqrtf(8192.0f)) ||
        !ds4_gpu_indexer_topk_tensor(selected, scores, n_comp, 1u, top_k) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(scores, 0, scores_host,
                             (uint64_t)n_comp * sizeof(float)) ||
        !ds4_gpu_tensor_read(selected, 0, topk_host,
                             (uint64_t)top_k * sizeof(uint32_t))) {
        goto cleanup;
    }
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < n_comp; i++) {
        const float v = fabsf(scores_host[i]);
        if (v > max_abs) max_abs = v;
    }
    uint32_t topk_diff = 0u;
    for (uint32_t i = 0; i < top_k; i++) {
        if (topk_host[i] != i) topk_diff++;
    }
    fprintf(stderr,
            "cuda-regression: large packed indexer max_abs=%g topk_diff=%u\n",
            (double)max_abs, topk_diff);
    rc = max_abs == 0.0f && topk_diff == 0u ? 0 : 1;
cleanup:
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(scores);
    ds4_gpu_tensor_free(cache);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(q);
    free(topk_host);
    free(scores_host);
    free(weights_host);
    free(q_host);
    return rc;
}

int main(void) {
    if (!ds4_gpu_init()) return 1;
    int rc = check_large_topk();
    if (check_decode_attention_overflow_path() != 0) rc = 1;
    if (check_b1_indexer_wmma() != 0) rc = 1;
    if (check_packed_attention_parity() != 0) rc = 1;
    if (check_packed_batch_attention_exact_parity() != 0) rc = 1;
    if (check_large_packed_indexer_zero() != 0) rc = 1;
    ds4_gpu_cleanup();
    if (rc == 0) puts("cuda long-context regression: OK");
    return rc;
}
