static void test_metal_compressor_ratio4_pack_exact(void) {
    test_metal_compressor_ape_add_exact_case(128, 4, 0, 4, 1, 37, true);
    test_metal_compressor_ape_add_exact_case(512, 4, 0, 8, 1, 41, true);
    test_metal_compressor_ape_add_exact_case(512, 4, 1, 14, 0, 43, true);
}

static void test_metal_compressor_ratio4_replay_pack_exact_case(
        uint32_t head_dim,
        uint32_t n_tokens,
        uint32_t seed) {
    const uint32_t width = 2u * head_dim;
    const uint32_t n_comp = n_tokens / 4u;
    const uint64_t input_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)8u * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)4u * width * sizeof(uint16_t);
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t norm_offset = test_round_up_u64(ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *fused_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_score = ds4_gpu_tensor_alloc(state_bytes);
    float *kv_host = malloc((size_t)input_bytes);
    float *sc_host = malloc((size_t)input_bytes);
    float *state_kv_host = malloc((size_t)state_bytes);
    float *state_score_host = malloc((size_t)state_bytes);
    float *source_after_host = malloc((size_t)input_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *fused_comp_host = malloc((size_t)comp_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *fused_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *fused_state_score_host = malloc((size_t)state_bytes);
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);

    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(sc != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(fused_comp != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(fused_state_kv != NULL);
    TEST_ASSERT(fused_state_score != NULL);
    TEST_ASSERT(kv_host != NULL);
    TEST_ASSERT(sc_host != NULL);
    TEST_ASSERT(state_kv_host != NULL);
    TEST_ASSERT(state_score_host != NULL);
    TEST_ASSERT(source_after_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(fused_comp_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(fused_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(fused_state_score_host != NULL);
    TEST_ASSERT(model_raw != NULL);

    const char *ape_disable_env = "DS4_METAL_DISABLE_COMPRESSOR_APE_ADD";
    const char *pack_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_PACK_FUSION";
    char *saved_ape_disable = test_save_env(ape_disable_env);
    char *saved_pack_disable = test_save_env(pack_disable_env);
    test_float_compare_stats comp_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};

    const bool allocated = kv && sc && ref_comp && fused_comp &&
        ref_state_kv && ref_state_score && fused_state_kv &&
        fused_state_score && kv_host && sc_host && state_kv_host &&
        state_score_host && source_after_host && ref_comp_host && fused_comp_host &&
        ref_state_kv_host && fused_state_kv_host && ref_state_score_host &&
        fused_state_score_host && model_raw;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        uint16_t *ape = model_raw;
        for (uint64_t i = 0; i < (uint64_t)4u * width; i++) {
            const int value =
                (int)((i * 19u + (i ^ (i >> 3u)) * 7u + seed * 11u) %
                      113u) - 56;
            ape[i] = test_float_to_f16((float)value / 72.0f);
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.875f + (float)((i * 5u + seed) % 17u) / 64.0f;
        }
        for (uint64_t i = 0; i < input_count; i++) {
            const int kv_value =
                (int)((i * 31u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                      193u) - 96;
            const int sc_value =
                (int)((i * 37u + (i ^ (i >> 5u)) * 9u + seed * 17u) %
                      181u) - 90;
            kv_host[i] = (float)kv_value / 104.0f;
            sc_host[i] = (float)sc_value / 88.0f;
        }
        for (uint64_t i = 0; i < state_count; i++) {
            const int kv_value =
                (int)((i * 23u + (i ^ (i >> 2u)) * 3u + seed * 29u) %
                      167u) - 83;
            const int sc_value =
                (int)((i * 41u + (i ^ (i >> 6u)) * 11u + seed * 7u) %
                      157u) - 78;
            state_kv_host[i] = (float)kv_value / 80.0f;
            state_score_host[i] = (float)sc_value / 92.0f;
        }
        const uint32_t negative_zero = 0x80000000u;
        memcpy(kv_host, &negative_zero, sizeof(negative_zero));
        memcpy(state_kv_host, &negative_zero, sizeof(negative_zero));

        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(unsetenv(ape_disable_env) == 0);
        TEST_ASSERT(setenv(pack_disable_env, "1", 1) == 0);

        TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
            ref_comp, ref_state_kv, ref_state_score, kv, sc,
            model_raw, model_bytes, 0, 1, norm_offset, 0,
            head_dim, 0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        // Poison every persistent pack cell through the legacy full-fill path
        // so a missing candidate write cannot inherit the reference value.
        for (uint64_t i = 0; i < input_count; i++) {
            const uint32_t bits = 0x7fc00001u + (uint32_t)(i & 0x3ffu);
            memcpy(kv_host + i, &bits, sizeof(bits));
            memcpy(sc_host + i, &bits, sizeof(bits));
        }
        for (uint64_t i = 0; i < state_count; i++) {
            const uint32_t bits = 0x7fc00401u + (uint32_t)(i & 0x3ffu);
            memcpy(ref_state_kv_host + i, &bits, sizeof(bits));
            memcpy(ref_state_score_host + i, &bits, sizeof(bits));
        }
        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
            fused_comp, fused_state_kv, fused_state_score, kv, sc,
            model_raw, model_bytes, 0, 1, norm_offset, 0,
            head_dim, 0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        // Restore finite sources and the candidate's original replay state.
        for (uint64_t i = 0; i < input_count; i++) {
            const int kv_value =
                (int)((i * 31u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                      193u) - 96;
            const int sc_value =
                (int)((i * 37u + (i ^ (i >> 5u)) * 9u + seed * 17u) %
                      181u) - 90;
            kv_host[i] = (float)kv_value / 104.0f;
            sc_host[i] = (float)sc_value / 88.0f;
        }
        memcpy(kv_host, &negative_zero, sizeof(negative_zero));
        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(unsetenv(pack_disable_env) == 0);
        TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
            fused_comp, fused_state_kv, fused_state_score, kv, sc,
            model_raw, model_bytes, 0, 1, norm_offset, 0,
            head_dim, 0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        kv, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(kv_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        sc, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(sc_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_kv, 0, fused_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);

        comp_stats = test_compare_float_bits(
            ref_comp_host, fused_comp_host, (size_t)comp_count);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, fused_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, fused_state_score_host,
            (size_t)state_count);
    }

    test_restore_env(ape_disable_env, saved_ape_disable);
    test_restore_env(pack_disable_env, saved_pack_disable);
    fprintf(stderr,
            "ds4-test: compressor ratio4 replay pack exact head=%u "
            "tokens=%u comp=%zu/%llu state_kv=%zu/%llu "
            "state_score=%zu/%llu max_ulp=%u/%u/%u\n",
            head_dim, n_tokens,
            comp_stats.mismatch_count, (unsigned long long)comp_count,
            state_kv_stats.mismatch_count, (unsigned long long)state_count,
            state_score_stats.mismatch_count, (unsigned long long)state_count,
            comp_stats.max_ulp, state_kv_stats.max_ulp,
            state_score_stats.max_ulp);
    TEST_ASSERT(comp_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);

    free(model_raw);
    free(fused_state_score_host);
    free(ref_state_score_host);
    free(fused_state_kv_host);
    free(ref_state_kv_host);
    free(fused_comp_host);
    free(ref_comp_host);
    free(source_after_host);
    free(state_score_host);
    free(state_kv_host);
    free(sc_host);
    free(kv_host);
    ds4_gpu_tensor_free(fused_state_score);
    ds4_gpu_tensor_free(fused_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(fused_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(kv);
}

static void test_metal_compressor_ratio4_replay_pack_exact(void) {
    test_metal_compressor_ratio4_replay_pack_exact_case(128, 4, 47);
    test_metal_compressor_ratio4_replay_pack_exact_case(512, 8, 53);
}

static void test_metal_compressor_ratio4_direct_pool_exact_case(
        uint32_t head_dim,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t ape_type,
        bool replay,
        uint32_t seed) {
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t input_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(float);
    const uint64_t ape_elem_bytes = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)ratio * width * ape_elem_bytes;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t norm_offset = test_round_up_u64(ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);

    TEST_ASSERT(n_comp != 0);
    TEST_ASSERT(!replay || ((pos0 & 3u) == 0u && (n_tokens & 3u) == 0u));

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(input_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *direct_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *direct_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *direct_state_score = ds4_gpu_tensor_alloc(state_bytes);
    float *kv_host = malloc((size_t)input_bytes);
    float *sc_host = malloc((size_t)input_bytes);
    float *source_after_host = malloc((size_t)input_bytes);
    float *state_kv_host = malloc((size_t)state_bytes);
    float *state_score_host = malloc((size_t)state_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *direct_comp_host = malloc((size_t)comp_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *direct_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *direct_state_score_host = malloc((size_t)state_bytes);
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_bytes) == 0);

    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(sc != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(direct_comp != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(direct_state_kv != NULL);
    TEST_ASSERT(direct_state_score != NULL);
    TEST_ASSERT(kv_host != NULL);
    TEST_ASSERT(sc_host != NULL);
    TEST_ASSERT(source_after_host != NULL);
    TEST_ASSERT(state_kv_host != NULL);
    TEST_ASSERT(state_score_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(direct_comp_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(direct_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(direct_state_score_host != NULL);
    TEST_ASSERT(model_raw != NULL);

    const char *ape_disable_env = "DS4_METAL_DISABLE_COMPRESSOR_APE_ADD";
    const char *pack_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_PACK_FUSION";
    const char *direct_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_DIRECT_POOL";
    char *saved_ape_disable = test_save_env(ape_disable_env);
    char *saved_pack_disable = test_save_env(pack_disable_env);
    char *saved_direct_disable = test_save_env(direct_disable_env);
    test_float_compare_stats comp_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};

    const bool allocated = kv && sc && ref_comp && direct_comp &&
        ref_state_kv && ref_state_score && direct_state_kv &&
        direct_state_score && kv_host && sc_host && source_after_host &&
        state_kv_host && state_score_host && ref_comp_host &&
        direct_comp_host && ref_state_kv_host && direct_state_kv_host &&
        ref_state_score_host && direct_state_score_host && model_raw;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        if (ape_type == 1u) {
            uint16_t *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 17u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                          127u) - 63;
                ape[i] = test_float_to_f16((float)value / 80.0f);
            }
        } else {
            float *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 17u + (i ^ (i >> 4u)) * 5u + seed * 13u) %
                          127u) - 63;
                ape[i] = (float)value / 80.0f;
            }
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.75f + (float)((i * 7u + seed) % 19u) / 64.0f;
        }

        for (uint64_t i = 0; i < input_count; i++) {
            const uint32_t token = (uint32_t)(i / width);
            const uint32_t col = (uint32_t)(i - (uint64_t)token * width);
            const int kv_value =
                (int)(((uint64_t)token * 37u + (uint64_t)col * 29u +
                       (col >= head_dim ? 71u : 3u) + seed * 23u) %
                      193u) - 96;
            const int sc_value =
                (int)(((uint64_t)token * 41u + (uint64_t)col * 31u +
                       (col >= head_dim ? 17u : 83u) + seed * 11u) %
                      181u) - 90;
            kv_host[i] = (float)kv_value / 104.0f;
            sc_host[i] = (float)sc_value / 32.0f;
        }
        const uint32_t negative_zero = 0x80000000u;
        memcpy(kv_host, &negative_zero, sizeof(negative_zero));
        memcpy(sc_host + width + head_dim, &negative_zero,
               sizeof(negative_zero));

        for (uint32_t row = 0; row < state_rows; row++) {
            for (uint32_t col = 0; col < width; col++) {
                const uint64_t i = (uint64_t)row * width + col;
                if (replay && row < ratio && col < head_dim) {
                    const int kv_value =
                        (int)(((uint64_t)row * 43u + (uint64_t)col * 19u +
                               seed * 29u) % 167u) - 83;
                    const int sc_value =
                        (int)(((uint64_t)row * 47u + (uint64_t)col * 23u +
                               seed * 7u) % 157u) - 78;
                    state_kv_host[i] = (float)kv_value / 80.0f;
                    state_score_host[i] = (float)sc_value / 28.0f;
                } else {
                    const uint32_t kv_bits =
                        0x7fc00001u + (uint32_t)(i & 0x3ffu);
                    const uint32_t score_bits =
                        0x7fc00401u + (uint32_t)(i & 0x3ffu);
                    memcpy(state_kv_host + i, &kv_bits, sizeof(kv_bits));
                    memcpy(state_score_host + i, &score_bits,
                           sizeof(score_bits));
                }
            }
        }

        for (uint64_t i = 0; i < comp_count; i++) {
            const uint32_t bits = 0x7fc00801u + (uint32_t)(i & 0x3ffu);
            memcpy(ref_comp_host + i, &bits, sizeof(bits));
            memcpy(direct_comp_host + i, &bits, sizeof(bits));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        direct_comp, 0, direct_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        direct_state_kv, 0, state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        direct_state_score, 0, state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(unsetenv(ape_disable_env) == 0);
        TEST_ASSERT(setenv(direct_disable_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(pack_disable_env) == 0);
        int ref_ok;
        if (replay) {
            ref_ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                ref_comp, ref_state_kv, ref_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        } else {
            ref_ok = ds4_gpu_compressor_prefill_tensor(
                ref_comp, ref_state_kv, ref_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, ratio, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        }
        TEST_ASSERT(ref_ok != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        kv, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(kv_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        sc, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(sc_host, source_after_host,
                           (size_t)input_bytes) == 0);

        TEST_ASSERT(unsetenv(direct_disable_env) == 0);
        TEST_ASSERT(setenv(pack_disable_env, "1", 1) == 0);
        int direct_ok;
        if (replay) {
            direct_ok = ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                direct_comp, direct_state_kv, direct_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        } else {
            direct_ok = ds4_gpu_compressor_prefill_tensor(
                direct_comp, direct_state_kv, direct_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, ratio, pos0, n_tokens, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f);
        }
        TEST_ASSERT(direct_ok != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        kv, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(kv_host, source_after_host,
                           (size_t)input_bytes) == 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        sc, 0, source_after_host, input_bytes) != 0);
        TEST_ASSERT(memcmp(sc_host, source_after_host,
                           (size_t)input_bytes) == 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        direct_comp, 0, direct_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        direct_state_kv, 0, direct_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        direct_state_score, 0, direct_state_score_host,
                        state_bytes) != 0);

        comp_stats = test_compare_float_bits(
            ref_comp_host, direct_comp_host, (size_t)comp_count);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, direct_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, direct_state_score_host,
            (size_t)state_count);
    }

    test_restore_env(ape_disable_env, saved_ape_disable);
    test_restore_env(pack_disable_env, saved_pack_disable);
    test_restore_env(direct_disable_env, saved_direct_disable);
    fprintf(stderr,
            "ds4-test: compressor ratio4 direct pool exact mode=%s "
            "head=%u pos=%u tokens=%u comp_rows=%u ape=%s "
            "comp=%zu/%llu state_kv=%zu/%llu state_score=%zu/%llu "
            "max_ulp=%u/%u/%u\n",
            replay ? "replay" : "prefill",
            head_dim, pos0, n_tokens, n_comp,
            ape_type == 1u ? "f16" : "f32",
            comp_stats.mismatch_count, (unsigned long long)comp_count,
            state_kv_stats.mismatch_count, (unsigned long long)state_count,
            state_score_stats.mismatch_count,
            (unsigned long long)state_count,
            comp_stats.max_ulp, state_kv_stats.max_ulp,
            state_score_stats.max_ulp);
    TEST_ASSERT(comp_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);

    free(model_raw);
    free(direct_state_score_host);
    free(ref_state_score_host);
    free(direct_state_kv_host);
    free(ref_state_kv_host);
    free(direct_comp_host);
    free(ref_comp_host);
    free(state_score_host);
    free(state_kv_host);
    free(source_after_host);
    free(sc_host);
    free(kv_host);
    ds4_gpu_tensor_free(direct_state_score);
    ds4_gpu_tensor_free(direct_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(direct_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(kv);
}

static void test_metal_compressor_ratio4_direct_pool_exact(void) {
    /* n_comp == 1 deliberately stays on the exact GGML reduction path. */
    test_metal_compressor_ratio4_direct_pool_exact_case(
        512, 0, 4, 1, false, 59);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        512, 1, 16, 0, false, 61);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        128, 3, 14, 1, false, 67);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        512, 2048, 8, 1, true, 71);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        128, 12, 12, 0, true, 73);
    test_metal_compressor_ratio4_direct_pool_exact_case(
        128, 8, 4, 1, true, 79);
}

static void test_metal_inplace_rope_pair_exact(void) {
    typedef struct {
        uint32_t head_dim;
        uint32_t n_rot;
        uint32_t n_head;
        uint32_t n_tok;
        uint32_t pos0;
        bool inverse;
        float ext_factor;
    } rope_case;
    static const rope_case cases[] = {
        { 512, 64,  64,  1, UINT32_MAX, false, 1.0f },
        { 512, 64,   1,  1,  2047, false, 1.0f },
        { 512, 64,  64,  1, 65533,  true, 1.0f },
        { 128, 64,  64,  1,    37, false, 0.0f },
        { 512, 64,   4, 32,     0, false, 0.0f },
        { 512, 64,   7, 33,  2047,  true, 1.0f },
        { 128, 64,  64, 34, 65533, false, 1.0f },
        { 128, 64,  64, 35,    37,  true, 0.0f },
        { 128, 64,   4, 35, UINT32_MAX - 16u, true, 1.0f },
    };
    const char *disable_env = "DS4_METAL_DISABLE_INPLACE_ROPE_PAIR";
    const char *shared_disable_env =
        "DS4_METAL_DISABLE_SHARED_ROPE_COEFF";
    const char *affine_disable_env =
        "DS4_METAL_DISABLE_AFFINE_ROPE_PAIR";
    char *saved_disable = test_save_env(disable_env);
    char *saved_shared_disable = test_save_env(shared_disable_env);
    char *saved_affine_disable = test_save_env(affine_disable_env);
    size_t total_pair_mismatch = 0;
    size_t total_shared_mismatch = 0;
    size_t total_affine_mismatch = 0;
    size_t total_pair_prefix_mismatch = 0;
    size_t total_pair_tail_mismatch = 0;
    size_t total_shared_prefix_mismatch = 0;
    size_t total_shared_tail_mismatch = 0;
    size_t total_affine_prefix_mismatch = 0;
    size_t total_affine_tail_mismatch = 0;
    size_t total_elements = 0;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const rope_case *c = &cases[ci];
        const size_t elements =
            (size_t)c->n_tok * c->n_head * c->head_dim;
        const uint64_t bytes = (uint64_t)elements * sizeof(float);
        const uint32_t n_nope = c->head_dim - c->n_rot;
        const float freq_base = c->ext_factor != 0.0f ? 160000.0f : 10000.0f;
        const float freq_scale = c->ext_factor != 0.0f ? 1.0f / 16.0f : 1.0f;
        const uint32_t n_ctx_orig = c->ext_factor != 0.0f ? 65536u : 0u;
        float attn_factor = 1.0f;
        if (c->ext_factor != 0.0f) {
            attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }

        ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *pair_candidate = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *shared_candidate = ds4_gpu_tensor_alloc(bytes);
        ds4_gpu_tensor *affine_candidate = ds4_gpu_tensor_alloc(bytes);
        float *input = malloc((size_t)bytes);
        float *reference_host = malloc((size_t)bytes);
        float *pair_host = malloc((size_t)bytes);
        float *shared_host = malloc((size_t)bytes);
        float *affine_host = malloc((size_t)bytes);
        TEST_ASSERT(reference != NULL);
        TEST_ASSERT(pair_candidate != NULL);
        TEST_ASSERT(shared_candidate != NULL);
        TEST_ASSERT(affine_candidate != NULL);
        TEST_ASSERT(input != NULL);
        TEST_ASSERT(reference_host != NULL);
        TEST_ASSERT(pair_host != NULL);
        TEST_ASSERT(shared_host != NULL);
        TEST_ASSERT(affine_host != NULL);

        const bool allocated = reference && pair_candidate &&
            shared_candidate && affine_candidate && input && reference_host &&
            pair_host && shared_host && affine_host;
        if (allocated) {
            for (size_t i = 0; i < elements; i++) {
                const uint32_t key =
                    (uint32_t)(i * 37u + (i ^ (i >> 5u)) * 11u + ci * 101u);
                const int value = (int)(key % 4093u) - 2046;
                input[i] = (float)value / 1024.0f;
                if ((i + ci * 17u) % 257u == 0u) {
                    const uint32_t negative_zero = 0x80000000u;
                    memcpy(&input[i], &negative_zero, sizeof(negative_zero));
                }
            }

            TEST_ASSERT(ds4_gpu_tensor_write(reference, 0, input, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                pair_candidate, 0, input, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                shared_candidate, 0, input, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                affine_candidate, 0, input, bytes) != 0);
            ds4_gpu_set_quality(false);

            TEST_ASSERT(setenv(affine_disable_env, "1", 1) == 0);
            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
            TEST_ASSERT(setenv(shared_disable_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                reference,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(unsetenv(disable_env) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                pair_candidate,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(unsetenv(shared_disable_env) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                shared_candidate,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(setenv(shared_disable_env, "1", 1) == 0);
            TEST_ASSERT(unsetenv(affine_disable_env) == 0);
            TEST_ASSERT(ds4_gpu_rope_tail_tensor(
                affine_candidate,
                c->n_tok,
                c->n_head,
                c->head_dim,
                c->n_rot,
                c->pos0,
                n_ctx_orig,
                c->inverse,
                freq_base,
                freq_scale,
                c->ext_factor,
                attn_factor,
                32.0f,
                1.0f) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                reference, 0, reference_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                pair_candidate, 0, pair_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                shared_candidate, 0, shared_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                affine_candidate, 0, affine_host, bytes) != 0);

            const test_float_compare_stats pair_stats =
                test_compare_float_bits(
                    reference_host, pair_host, elements);
            const test_float_compare_stats shared_stats =
                test_compare_float_bits(
                    pair_host, shared_host, elements);
            const test_float_compare_stats affine_stats =
                test_compare_float_bits(
                    pair_host, affine_host, elements);
            size_t pair_prefix_mismatch = 0;
            size_t pair_tail_mismatch = 0;
            size_t shared_prefix_mismatch = 0;
            size_t shared_tail_mismatch = 0;
            size_t affine_prefix_mismatch = 0;
            size_t affine_tail_mismatch = 0;
            for (uint32_t t = 0; t < c->n_tok; t++) {
                for (uint32_t h = 0; h < c->n_head; h++) {
                    const size_t row =
                        ((size_t)t * c->n_head + h) * c->head_dim;
                    for (uint32_t d = 0; d < n_nope; d++) {
                        if (memcmp(&input[row + d],
                                   &pair_host[row + d],
                                   sizeof(float)) != 0) {
                            pair_prefix_mismatch++;
                        }
                        if (memcmp(&input[row + d],
                                   &shared_host[row + d],
                                   sizeof(float)) != 0) {
                            shared_prefix_mismatch++;
                        }
                        if (memcmp(&input[row + d],
                                   &affine_host[row + d],
                                   sizeof(float)) != 0) {
                            affine_prefix_mismatch++;
                        }
                    }
                    for (uint32_t d = n_nope; d < c->head_dim; d++) {
                        if (memcmp(&reference_host[row + d],
                                   &pair_host[row + d],
                                   sizeof(float)) != 0) {
                            pair_tail_mismatch++;
                        }
                        if (memcmp(&pair_host[row + d],
                                   &shared_host[row + d],
                                   sizeof(float)) != 0) {
                            shared_tail_mismatch++;
                        }
                        if (memcmp(&pair_host[row + d],
                                   &affine_host[row + d],
                                   sizeof(float)) != 0) {
                            affine_tail_mismatch++;
                        }
                    }
                }
            }

            fprintf(stderr,
                    "ds4-test: in-place RoPE exactness case=%zu "
                    "shape=%ux%ux%u pos=%u inverse=%d ext=%g "
                    "pair=%zu/%zu shared=%zu/%zu affine=%zu/%zu "
                    "pair_prefix=%zu pair_tail=%zu "
                    "shared_prefix=%zu shared_tail=%zu "
                    "affine_prefix=%zu affine_tail=%zu "
                    "pair_max_ulp=%u shared_max_ulp=%u affine_max_ulp=%u "
                    "pair_max_abs=%g shared_max_abs=%g affine_max_abs=%g\n",
                    ci,
                    c->n_tok,
                    c->n_head,
                    c->head_dim,
                    c->pos0,
                    c->inverse ? 1 : 0,
                    c->ext_factor,
                    pair_stats.mismatch_count,
                    elements,
                    shared_stats.mismatch_count,
                    elements,
                    affine_stats.mismatch_count,
                    elements,
                    pair_prefix_mismatch,
                    pair_tail_mismatch,
                    shared_prefix_mismatch,
                    shared_tail_mismatch,
                    affine_prefix_mismatch,
                    affine_tail_mismatch,
                    pair_stats.max_ulp,
                    shared_stats.max_ulp,
                    affine_stats.max_ulp,
                    pair_stats.max_abs,
                    shared_stats.max_abs,
                    affine_stats.max_abs);
            TEST_ASSERT(pair_stats.mismatch_count == 0);
            TEST_ASSERT(shared_stats.mismatch_count == 0);
            TEST_ASSERT(affine_stats.mismatch_count == 0);
            TEST_ASSERT(pair_prefix_mismatch == 0);
            TEST_ASSERT(pair_tail_mismatch == 0);
            TEST_ASSERT(shared_prefix_mismatch == 0);
            TEST_ASSERT(shared_tail_mismatch == 0);
            TEST_ASSERT(affine_prefix_mismatch == 0);
            TEST_ASSERT(affine_tail_mismatch == 0);
            total_pair_mismatch += pair_stats.mismatch_count;
            total_shared_mismatch += shared_stats.mismatch_count;
            total_affine_mismatch += affine_stats.mismatch_count;
            total_pair_prefix_mismatch += pair_prefix_mismatch;
            total_pair_tail_mismatch += pair_tail_mismatch;
            total_shared_prefix_mismatch += shared_prefix_mismatch;
            total_shared_tail_mismatch += shared_tail_mismatch;
            total_affine_prefix_mismatch += affine_prefix_mismatch;
            total_affine_tail_mismatch += affine_tail_mismatch;
            total_elements += elements;
        }

        free(affine_host);
        free(shared_host);
        free(pair_host);
        free(reference_host);
        free(input);
        ds4_gpu_tensor_free(affine_candidate);
        ds4_gpu_tensor_free(shared_candidate);
        ds4_gpu_tensor_free(pair_candidate);
        ds4_gpu_tensor_free(reference);
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(shared_disable_env, saved_shared_disable);
    test_restore_env(affine_disable_env, saved_affine_disable);
    fprintf(stderr,
            "ds4-test: in-place RoPE total pair=%zu/%zu shared=%zu/%zu "
            "affine=%zu/%zu "
            "pair_prefix=%zu pair_tail=%zu "
            "shared_prefix=%zu shared_tail=%zu "
            "affine_prefix=%zu affine_tail=%zu\n",
            total_pair_mismatch,
            total_elements,
            total_shared_mismatch,
            total_elements,
            total_affine_mismatch,
            total_elements,
            total_pair_prefix_mismatch,
            total_pair_tail_mismatch,
            total_shared_prefix_mismatch,
            total_shared_tail_mismatch,
            total_affine_prefix_mismatch,
            total_affine_tail_mismatch);
    TEST_ASSERT(total_pair_mismatch == 0);
    TEST_ASSERT(total_shared_mismatch == 0);
    TEST_ASSERT(total_affine_mismatch == 0);
    TEST_ASSERT(total_pair_prefix_mismatch == 0);
    TEST_ASSERT(total_pair_tail_mismatch == 0);
    TEST_ASSERT(total_shared_prefix_mismatch == 0);
    TEST_ASSERT(total_shared_tail_mismatch == 0);
    TEST_ASSERT(total_affine_prefix_mismatch == 0);
    TEST_ASSERT(total_affine_tail_mismatch == 0);
}

static void test_metal_contiguous_f32_f16_roundtrip_exact(void) {
    typedef struct {
        uint32_t n;
        uint32_t src_offset;
        uint32_t dst_offset;
    } copy_case;
    static const copy_case cases[] = {
        { 1,  0,  0 },
        { 3,  4,  2 },
        { 4, 16,  8 },
        { 5, 12,  6 },
        { 17, 20, 10 },
        { 65,  4,  2 },
    };
    const char *env_name = "DS4_METAL_DISABLE_CONTIG_F32_F16_COPY";
    char *saved_env = test_save_env(env_name);
    size_t half_mismatch = 0;
    size_t half_guard_mismatch = 0;
    size_t roundtrip_mismatch = 0;
    size_t roundtrip_guard_mismatch = 0;
    size_t half_total = 0;
    size_t roundtrip_total = 0;

    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const uint32_t n = cases[ci].n;
        const uint64_t src_bytes = cases[ci].src_offset +
                                   (uint64_t)n * sizeof(float) + 16u;
        const uint64_t half_bytes = cases[ci].dst_offset +
                                    (uint64_t)n * sizeof(uint16_t) + 16u;
        const uint32_t raw_cap = 3;
        const uint32_t raw_row = 1;
        const uint64_t raw_bytes =
            (uint64_t)raw_cap * n * sizeof(float);

        ds4_gpu_tensor *src_base = ds4_gpu_tensor_alloc(src_bytes);
        ds4_gpu_tensor *src_view = src_base
            ? ds4_gpu_tensor_view(src_base,
                                  cases[ci].src_offset,
                                  (uint64_t)n * sizeof(float))
            : NULL;
        ds4_gpu_tensor *half_ref = ds4_gpu_tensor_alloc(half_bytes);
        ds4_gpu_tensor *half_vec = ds4_gpu_tensor_alloc(half_bytes);
        ds4_gpu_tensor *raw_ref = ds4_gpu_tensor_alloc(raw_bytes);
        ds4_gpu_tensor *raw_vec = ds4_gpu_tensor_alloc(raw_bytes);
        TEST_ASSERT(src_base != NULL);
        TEST_ASSERT(src_view != NULL);
        TEST_ASSERT(half_ref != NULL);
        TEST_ASSERT(half_vec != NULL);
        TEST_ASSERT(raw_ref != NULL);
        TEST_ASSERT(raw_vec != NULL);

        uint8_t *src_host = malloc((size_t)src_bytes);
        uint16_t *half_init = malloc((size_t)half_bytes);
        uint16_t *half_ref_host = malloc((size_t)half_bytes);
        uint16_t *half_vec_host = malloc((size_t)half_bytes);
        uint32_t *raw_init = malloc((size_t)raw_bytes);
        uint32_t *raw_ref_host = malloc((size_t)raw_bytes);
        uint32_t *raw_vec_host = malloc((size_t)raw_bytes);
        TEST_ASSERT(src_host != NULL);
        TEST_ASSERT(half_init != NULL);
        TEST_ASSERT(half_ref_host != NULL);
        TEST_ASSERT(half_vec_host != NULL);
        TEST_ASSERT(raw_init != NULL);
        TEST_ASSERT(raw_ref_host != NULL);
        TEST_ASSERT(raw_vec_host != NULL);

        const bool allocated = src_base && src_view && half_ref && half_vec &&
            raw_ref && raw_vec && src_host && half_init && half_ref_host &&
            half_vec_host && raw_init && raw_ref_host && raw_vec_host;
        if (allocated) {
            memset(src_host, 0x6d, (size_t)src_bytes);
            test_fill_copy_f32_patterns(src_host + cases[ci].src_offset,
                                        n,
                                        (uint32_t)(ci * 7u));
            const size_t half_words = (size_t)(half_bytes / sizeof(uint16_t));
            for (size_t i = 0; i < half_words; i++) {
                half_init[i] = (uint16_t)(0xa55au ^ (uint16_t)(i * 73u));
            }
            const size_t raw_words = (size_t)raw_cap * n;
            for (size_t i = 0; i < raw_words; i++) {
                raw_init[i] = 0x4a000000u + (uint32_t)i;
            }

            TEST_ASSERT(ds4_gpu_tensor_write(src_base, 0, src_host, src_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(half_ref, 0, half_init, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(half_vec, 0, half_init, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(raw_ref, 0, raw_init, raw_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(raw_vec, 0, raw_init, raw_bytes) != 0);

            TEST_ASSERT(setenv(env_name, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_tensor_copy_f32_to_f16(
                half_ref,
                cases[ci].dst_offset,
                src_base,
                cases[ci].src_offset,
                n) != 0);
            TEST_ASSERT(ds4_gpu_store_raw_kv_tensor(
                raw_ref, src_view, raw_cap, raw_row, n) != 0);

            TEST_ASSERT(setenv(env_name, "0", 1) == 0);
            TEST_ASSERT(ds4_gpu_tensor_copy_f32_to_f16(
                half_vec,
                cases[ci].dst_offset,
                src_base,
                cases[ci].src_offset,
                n) != 0);
            TEST_ASSERT(ds4_gpu_store_raw_kv_tensor(
                raw_vec, src_view, raw_cap, raw_row, n) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                half_ref, 0, half_ref_host, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                half_vec, 0, half_vec_host, half_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                raw_ref, 0, raw_ref_host, raw_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                raw_vec, 0, raw_vec_host, raw_bytes) != 0);

            const size_t half_first = cases[ci].dst_offset / sizeof(uint16_t);
            const size_t half_last = half_first + n;
            for (size_t i = 0; i < half_words; i++) {
                if (half_ref_host[i] == half_vec_host[i]) continue;
                if (i >= half_first && i < half_last) {
                    half_mismatch++;
                } else {
                    half_guard_mismatch++;
                }
            }

            const size_t raw_first = (size_t)raw_row * n;
            const size_t raw_last = raw_first + n;
            for (size_t i = 0; i < raw_words; i++) {
                if (raw_ref_host[i] == raw_vec_host[i]) continue;
                if (i >= raw_first && i < raw_last) {
                    roundtrip_mismatch++;
                } else {
                    roundtrip_guard_mismatch++;
                }
            }
            half_total += n;
            roundtrip_total += n;
        }

        free(raw_vec_host);
        free(raw_ref_host);
        free(raw_init);
        free(half_vec_host);
        free(half_ref_host);
        free(half_init);
        free(src_host);
        ds4_gpu_tensor_free(raw_vec);
        ds4_gpu_tensor_free(raw_ref);
        ds4_gpu_tensor_free(half_vec);
        ds4_gpu_tensor_free(half_ref);
        ds4_gpu_tensor_free(src_view);
        ds4_gpu_tensor_free(src_base);
    }

    test_restore_env(env_name, saved_env);
    fprintf(stderr,
            "ds4-test: contiguous conversion exactness "
            "f32_f16=%zu/%zu guard=%zu, f16_f32_roundtrip=%zu/%zu guard=%zu\n",
            half_mismatch,
            half_total,
            half_guard_mismatch,
            roundtrip_mismatch,
            roundtrip_total,
            roundtrip_guard_mismatch);
    TEST_ASSERT(half_mismatch == 0);
    TEST_ASSERT(half_guard_mismatch == 0);
    TEST_ASSERT(roundtrip_mismatch == 0);
    TEST_ASSERT(roundtrip_guard_mismatch == 0);
}
