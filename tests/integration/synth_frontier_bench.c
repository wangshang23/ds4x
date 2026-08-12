#include "ds4.h"
#include "ds4_gpu.h"

#include <cuda_profiler_api.h>

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

static int compare_double(const void *a, const void *b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

static void select_legacy_path(int legacy) {
    if (getenv("DS4_SYNTH_AB_FUSED_INDEXER") != NULL) {
        if (legacy) {
            setenv("DS4_CUDA_NO_FUSED_INDEXER_TOPK", "1", 1);
        } else {
            unsetenv("DS4_CUDA_NO_FUSED_INDEXER_TOPK");
        }
        return;
    }
    if (legacy) {
        setenv("DS4_CUDA_SPARK_INDEXER_REFERENCE", "1", 1);
    } else {
        unsetenv("DS4_CUDA_SPARK_INDEXER_REFERENCE");
    }
}

static void print_samples(const char *mode, uint32_t context,
                          double *samples, uint32_t iters) {
    qsort(samples, iters, sizeof(*samples), compare_double);
    double sum = 0.0;
    for (uint32_t i = 0; i < iters; i++) sum += samples[i];
    const double median = (iters & 1u)
        ? samples[iters / 2u]
        : 0.5 * (samples[iters / 2u - 1u] + samples[iters / 2u]);
    printf("context=%u mode=%s iters=%u mean_ms=%.6f median_ms=%.6f "
           "min_ms=%.6f max_ms=%.6f median_tok_s=%.6f\n",
           context, mode, iters, sum / iters, median, samples[0],
           samples[iters - 1u], 1000.0 / median);
}

static int dump_logits_if_requested(ds4_session *session, int token,
                                    uint32_t pos, uint32_t context,
                                    int vocab, char *err, size_t errlen) {
    const char *prefix = getenv("DS4_SYNTH_DUMP_PREFIX");
    if (!prefix || !prefix[0]) return 0;
    float *logits = malloc((size_t)vocab * sizeof(*logits));
    if (!logits) return 1;
    int rc = 1;
    if (ds4_test_session_seed_frontier(session, pos, true) != 0 ||
        ds4_session_eval(session, token, err, errlen) != 0 ||
        ds4_session_copy_logits(session, logits, vocab) != vocab) {
        goto done;
    }
    char path[1024];
    if (snprintf(path, sizeof(path), "%s-%u.bin", prefix, context) >=
        (int)sizeof(path)) goto done;
    FILE *fp = fopen(path, "wb");
    if (!fp) goto done;
    const size_t wrote = fwrite(logits, sizeof(*logits), (size_t)vocab, fp);
    const int close_rc = fclose(fp);
    if (wrote != (size_t)vocab || close_rc != 0) goto done;
    printf("context=%u logits=%s argmax=%d\n",
           context, path, ds4_session_argmax(session));
    rc = 0;
done:
    free(logits);
    return rc;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        fprintf(stderr,
                "usage: %s MODEL CONTEXT|sweep|ab-sweep [WARMUP=2] [ITERS=10]\n",
                argv[0]);
        return 2;
    }
    const char *model = argv[1];
    static const uint32_t sweep_contexts[] = {
        131072u, 262144u, 524288u, 1048576u,
    };
    const int ab_sweep = strcmp(argv[2], "ab-sweep") == 0;
    const int sweep = ab_sweep || strcmp(argv[2], "sweep") == 0;
    const int forced_legacy = !ab_sweep && getenv("DS4_SYNTH_LEGACY") != NULL;
    const int profiler_capture = getenv("DS4_NSYS_CAPTURE") != NULL;
    const uint32_t one_context = sweep ? 0u :
        (uint32_t)strtoul(argv[2], NULL, 10);
    const uint32_t warmup = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 2u;
    const uint32_t iters = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 10u;
    if ((!sweep && (one_context < 2u || one_context > 1048576u)) ||
        iters == 0u) return 2;
    const uint32_t max_context = sweep ? 1048576u : one_context;

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model;
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 1;
    opt.context_size = (int)max_context;
    opt.prefill_chunk = 32u;
    opt.warm_weights = getenv("DS4_SYNTH_SKIP_WARM_WEIGHTS") == NULL;

    ds4_engine *engine = NULL;
    double *samples = calloc(iters, sizeof(*samples));
    double *legacy_samples = ab_sweep ? calloc(iters, sizeof(*legacy_samples)) : NULL;
    float *legacy_logits = NULL;
    float *optimized_logits = NULL;
    int rc = 1;
    if (!samples || (ab_sweep && !legacy_samples) ||
        ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "synth-frontier: engine initialization failed\n");
        goto cleanup;
    }
    const int vocab = ds4_engine_vocab_size(engine);
    if (ab_sweep) {
        legacy_logits = malloc((size_t)vocab * sizeof(*legacy_logits));
        optimized_logits = malloc((size_t)vocab * sizeof(*optimized_logits));
        if (!legacy_logits || !optimized_logits) goto cleanup;
    }

    const uint32_t n_contexts = sweep
        ? (uint32_t)(sizeof(sweep_contexts) / sizeof(sweep_contexts[0]))
        : 1u;
    for (uint32_t ci = 0; ci < n_contexts; ci++) {
        const uint32_t context = sweep ? sweep_contexts[ci] : one_context;
        const uint32_t pos = context - 1u;
        ds4_session *session = NULL;
        if (ds4_session_create(&session, engine, (int)context) != 0) {
            fprintf(stderr, "synth-frontier: session initialization failed\n");
            goto cleanup;
        }
        fprintf(stderr, "synth-frontier: initializing context=%u pos=%u\n",
                context, pos);
        if (ds4_test_session_seed_frontier(session, pos, true) != 0) {
            fprintf(stderr, "synth-frontier: cache initialization failed\n");
            ds4_session_free(session);
            goto cleanup;
        }
        if (getenv("DS4_SYNTH_MEMORY_REPORT") != NULL) {
            ds4_gpu_print_memory_report("synthetic frontier initialized");
        }

        char err[256] = {0};
        int token = ds4_token_assistant(engine);
        if (token < 0) token = 1;
        const uint32_t warmup_modes = ab_sweep ? 2u : 1u;
        for (uint32_t mode = 0; mode < warmup_modes; mode++) {
            select_legacy_path(forced_legacy || (ab_sweep && mode == 0u));
            for (uint32_t i = 0; i < warmup; i++) {
                if (ds4_test_session_seed_frontier(session, pos, false) != 0 ||
                    ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                    fprintf(stderr, "synth-frontier: warmup failed: %s\n", err);
                    ds4_session_free(session);
                    goto cleanup;
                }
            }
        }

        const uint32_t timed_modes = ab_sweep ? 2u : 1u;
        if (profiler_capture) (void)cudaProfilerStart();
        for (uint32_t i = 0; i < iters; i++) {
            for (uint32_t order = 0; order < timed_modes; order++) {
                const int legacy = forced_legacy ||
                    (ab_sweep && ((i + order) & 1u) == 0u);
                select_legacy_path(legacy);
                if (ds4_test_session_seed_frontier(session, pos, false) != 0) {
                    fprintf(stderr, "synth-frontier: frontier reset failed\n");
                    ds4_session_free(session);
                    goto cleanup;
                }
                const double t0 = monotonic_seconds();
                if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                    fprintf(stderr, "synth-frontier: decode failed: %s\n", err);
                    ds4_session_free(session);
                    goto cleanup;
                }
                const double elapsed = (monotonic_seconds() - t0) * 1000.0;
                (ab_sweep && legacy ? legacy_samples : samples)[i] = elapsed;
            }
        }
        if (profiler_capture) (void)cudaProfilerStop();
        if (getenv("DS4_SYNTH_MEMORY_REPORT") != NULL) {
            ds4_gpu_print_memory_report("synthetic frontier decoded");
        }

        if (ab_sweep) print_samples("legacy", context, legacy_samples, iters);
        print_samples(ab_sweep ? "optimized" :
                      (forced_legacy ? "legacy" : "current"), context,
                      samples, iters);
        if (dump_logits_if_requested(session, token, context - 2u, context,
                                     vocab, err, sizeof(err)) != 0) {
            fprintf(stderr, "synth-frontier: logits dump failed: %s\n", err);
            ds4_session_free(session);
            goto cleanup;
        }
        if (ab_sweep) {
            const uint32_t accuracy_pos = context - 2u;
            int legacy_argmax = -1;
            int optimized_argmax = -1;
            select_legacy_path(1);
            /* Decode mutates both cache bytes and compressor frontiers, so
             * each implementation must start from the same synthetic state. */
            if (ds4_test_session_seed_frontier(session, accuracy_pos, true) != 0 ||
                ds4_session_eval(session, token, err, sizeof(err)) != 0 ||
                ds4_session_copy_logits(session, legacy_logits, vocab) != vocab) {
                fprintf(stderr, "synth-frontier: legacy accuracy pass failed\n");
                ds4_session_free(session);
                goto cleanup;
            }
            legacy_argmax = ds4_session_argmax(session);
            select_legacy_path(0);
            if (ds4_test_session_seed_frontier(session, accuracy_pos, true) != 0 ||
                ds4_session_eval(session, token, err, sizeof(err)) != 0 ||
                ds4_session_copy_logits(session, optimized_logits, vocab) != vocab) {
                fprintf(stderr, "synth-frontier: optimized accuracy pass failed\n");
                ds4_session_free(session);
                goto cleanup;
            }
            optimized_argmax = ds4_session_argmax(session);
            float max_abs = 0.0f;
            double sum_sq = 0.0;
            uint32_t different = 0u;
            for (int i = 0; i < vocab; i++) {
                const float delta = fabsf(legacy_logits[i] - optimized_logits[i]);
                if (delta > max_abs) max_abs = delta;
                sum_sq += (double)delta * delta;
                if (memcmp(&legacy_logits[i], &optimized_logits[i],
                           sizeof(float)) != 0) {
                    different++;
                }
            }
            printf("context=%u accuracy_pos=%u max_abs=%g rmse=%g "
                   "different=%u/%d argmax=%d/%d\n",
                   context, accuracy_pos, (double)max_abs,
                   sqrt(sum_sq / (double)vocab), different, vocab,
                   legacy_argmax, optimized_argmax);
        }
        fflush(stdout);
        ds4_session_free(session);
    }
    rc = 0;

cleanup:
    select_legacy_path(0);
    ds4_engine_close(engine);
    free(optimized_logits);
    free(legacy_logits);
    free(legacy_samples);
    free(samples);
    return rc;
}
