#include "ds4.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s MODEL [CONTEXT=8192]\n", argv[0]);
        return 2;
    }
    const uint32_t context = argc == 3
        ? (uint32_t)strtoul(argv[2], NULL, 10)
        : 8192u;
    if (context < 4u || context > 1048576u) return 2;

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 1;
    opt.context_size = (int)context;
    opt.prefill_chunk = 32u;

    ds4_engine *engine = NULL;
    ds4_session *source = NULL;
    ds4_session *restored = NULL;
    FILE *fp = NULL;
    float *source_logits = NULL;
    float *restored_logits = NULL;
    int rc = 1;
    char err[256] = {0};

    if (ds4_engine_open(&engine, &opt) != 0 ||
        ds4_session_create(&source, engine, (int)context) != 0 ||
        ds4_session_create(&restored, engine, (int)context) != 0) {
        fprintf(stderr, "packed-checkpoint: initialization failed\n");
        goto done;
    }

    const int vocab = ds4_engine_vocab_size(engine);
    source_logits = malloc((size_t)vocab * sizeof(*source_logits));
    restored_logits = malloc((size_t)vocab * sizeof(*restored_logits));
    if (!source_logits || !restored_logits) goto done;

    const uint32_t frontier = context - 2u;
    int token = ds4_token_assistant(engine);
    if (token < 0) token = 1;
    if (ds4_test_session_seed_frontier(source, frontier, true) != 0 ||
        ds4_session_eval(source, token, err, sizeof(err)) != 0) {
        fprintf(stderr, "packed-checkpoint: source decode failed: %s\n", err);
        goto done;
    }

    const uint64_t expected_bytes = ds4_session_payload_bytes(source);
    fp = tmpfile();
    if (!fp || expected_bytes == 0u ||
        ds4_session_save_payload(source, fp, err, sizeof(err)) != 0) {
        fprintf(stderr, "packed-checkpoint: save failed: %s\n", err);
        goto done;
    }
    const long file_bytes = ftell(fp);
    if (file_bytes < 0 || (uint64_t)file_bytes != expected_bytes ||
        fseek(fp, 0, SEEK_SET) != 0 ||
        ds4_session_load_payload(restored, fp, expected_bytes,
                                 err, sizeof(err)) != 0) {
        fprintf(stderr,
                "packed-checkpoint: restore failed: %s expected=%llu file=%ld\n",
                err, (unsigned long long)expected_bytes, file_bytes);
        goto done;
    }

    if (ds4_session_pos(source) != ds4_session_pos(restored) ||
        ds4_session_argmax(source) != ds4_session_argmax(restored) ||
        ds4_session_eval(source, token, err, sizeof(err)) != 0 ||
        ds4_session_eval(restored, token, err, sizeof(err)) != 0 ||
        ds4_session_copy_logits(source, source_logits, vocab) != vocab ||
        ds4_session_copy_logits(restored, restored_logits, vocab) != vocab) {
        fprintf(stderr, "packed-checkpoint: post-restore decode failed: %s\n", err);
        goto done;
    }

    float max_abs = 0.0f;
    double sum_sq = 0.0;
    uint32_t different = 0u;
    for (int i = 0; i < vocab; i++) {
        const float delta = fabsf(source_logits[i] - restored_logits[i]);
        if (delta > max_abs) max_abs = delta;
        sum_sq += (double)delta * delta;
        different += memcmp(&source_logits[i], &restored_logits[i],
                            sizeof(float)) != 0;
    }
    const double rmse = sqrt(sum_sq / (double)vocab);
    const int source_top = ds4_session_argmax(source);
    const int restored_top = ds4_session_argmax(restored);
    printf("packed-checkpoint: context=%u payload_mib=%.3f max_abs=%g "
           "rmse=%g different=%u/%d argmax=%d/%d\n",
           context, (double)expected_bytes / 1048576.0, (double)max_abs,
           rmse, different, vocab, source_top, restored_top);
    rc = max_abs == 0.0f && source_top == restored_top ? 0 : 1;

done:
    if (fp) fclose(fp);
    free(restored_logits);
    free(source_logits);
    ds4_session_free(restored);
    ds4_session_free(source);
    ds4_engine_close(engine);
    return rc;
}
