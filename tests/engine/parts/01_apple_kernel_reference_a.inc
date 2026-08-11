static void test_metal_f16_compressor_pair_state_store_exact_case(
        uint32_t width,
        uint32_t ratio,
        uint32_t pos,
        uint32_t ape_type,
        uint32_t seed,
        bool test_decode_pack) {
    const uint32_t in_dim = 4096u;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t head_dim = width / coff;
    const uint32_t state_rows = coff * ratio;
    const bool emit = ((pos + 1u) % ratio) == 0u;
    TEST_ASSERT(!test_decode_pack ||
                (ratio == 4u && emit &&
                 (head_dim == 128u || head_dim == 512u)));
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_bytes =
        (uint64_t)width * in_dim * sizeof(uint16_t);
    const uint64_t score_weight_offset =
        test_round_up_u64(weight_bytes, page);
    const uint64_t ape_offset = test_round_up_u64(
        score_weight_offset + weight_bytes, page);
    const uint64_t ape_elem_bytes = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)ratio * width * ape_elem_bytes;
    const uint64_t norm_offset =
        test_round_up_u64(ape_offset + ape_bytes, page);
    const uint64_t model_bytes = test_round_up_u64(
        norm_offset + (uint64_t)head_dim * sizeof(float), page);
    const uint64_t x_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)head_dim * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page,
                               (size_t)model_bytes) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_kv = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_score = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_kv = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_score = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_kv = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *fused_state_score = ds4_gpu_tensor_alloc(state_bytes);
    ds4_gpu_tensor *ref_comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *fused_comp = ds4_gpu_tensor_alloc(comp_bytes);

    float *x_host = malloc((size_t)x_bytes);
    float *ref_kv_host = malloc((size_t)out_bytes);
    float *ref_score_host = malloc((size_t)out_bytes);
    float *fused_kv_host = malloc((size_t)out_bytes);
    float *fused_score_host = malloc((size_t)out_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *fused_state_kv_host = malloc((size_t)state_bytes);
    float *fused_state_score_host = malloc((size_t)state_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *fused_comp_host = malloc((size_t)comp_bytes);

    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref_kv != NULL);
    TEST_ASSERT(ref_score != NULL);
    TEST_ASSERT(fused_kv != NULL);
    TEST_ASSERT(fused_score != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(fused_state_kv != NULL);
    TEST_ASSERT(fused_state_score != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(fused_comp != NULL);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref_kv_host != NULL);
    TEST_ASSERT(ref_score_host != NULL);
    TEST_ASSERT(fused_kv_host != NULL);
    TEST_ASSERT(fused_score_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(fused_state_kv_host != NULL);
    TEST_ASSERT(fused_state_score_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(fused_comp_host != NULL);

    const bool allocated = model_raw && x && ref_kv && ref_score && fused_kv &&
        fused_score && ref_state_kv && ref_state_score && fused_state_kv &&
        fused_state_score && ref_comp && fused_comp && x_host && ref_kv_host &&
        ref_score_host && fused_kv_host && fused_score_host &&
        ref_state_kv_host && ref_state_score_host && fused_state_kv_host &&
        fused_state_score_host && ref_comp_host && fused_comp_host;

    const char *pair_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ";
    const char *store_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_STORE_ONE";
    const char *decode_pack_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION";
    const char *exact_reduction_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_EXACT_REDUCTION_FUSION";
    const char *exact_reduction_poison_env =
        "DS4_METAL_TEST_POISON_COMPRESSOR_EXACT_REDUCTION_SCRATCH";
    char *saved_pair_disable = test_save_env(pair_disable_env);
    char *saved_store_disable = test_save_env(store_disable_env);
    char *saved_decode_pack_disable =
        test_save_env(decode_pack_disable_env);
    char *saved_exact_reduction_disable =
        test_save_env(exact_reduction_disable_env);
    char *saved_exact_reduction_poison =
        test_save_env(exact_reduction_poison_env);

    test_float_compare_stats kv_stats = {0};
    test_float_compare_stats score_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};
    test_float_compare_stats comp_stats = {0};

    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        uint16_t *kv_weights = model_raw;
        uint16_t *score_weights =
            (uint16_t *)((uint8_t *)model_raw + score_weight_offset);
        for (uint32_t o = 0; o < width; o++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const int kv_value =
                    (int)((o * 17u + i * 23u + (o ^ i) * 3u +
                           seed * 29u) % 67u) - 33;
                const int score_value =
                    (int)((o * 31u + i * 11u + (o ^ (i >> 2u)) * 5u +
                           seed * 19u) % 71u) - 35;
                const uint64_t wi = (uint64_t)o * in_dim + i;
                kv_weights[wi] = test_float_to_f16(
                    (float)kv_value / 96.0f);
                score_weights[wi] = test_float_to_f16(
                    (float)score_value / 104.0f);
            }
        }

        if (ape_type == 1u) {
            uint16_t *ape =
                (uint16_t *)((uint8_t *)model_raw + ape_offset);
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 13u + (i ^ (i >> 3u)) * 7u +
                           seed * 17u) % 61u) - 30;
                ape[i] = test_float_to_f16((float)value / 80.0f);
            }
        } else {
            float *ape = (float *)((uint8_t *)model_raw + ape_offset);
            for (uint64_t i = 0; i < (uint64_t)ratio * width; i++) {
                const int value =
                    (int)((i * 13u + (i ^ (i >> 3u)) * 7u +
                           seed * 17u) % 61u) - 30;
                ape[i] = (float)value / 80.0f;
            }
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.75f +
                (float)((i * 7u + seed * 3u) % 23u) / 64.0f;
        }

        for (uint32_t i = 0; i < in_dim; i++) {
            const int value =
                (int)((i * 29u + (i ^ (i >> 4u)) * 9u +
                       seed * 11u) % 127u) - 63;
            x_host[i] = (float)value / 88.0f;
        }
        for (uint32_t i = 0; i < width; i++) {
            const uint32_t poison = 0x7fc00001u + (i & 0x3ffu);
            memcpy(ref_kv_host + i, &poison, sizeof(poison));
            memcpy(ref_score_host + i, &poison, sizeof(poison));
            memcpy(fused_kv_host + i, &poison, sizeof(poison));
            memcpy(fused_score_host + i, &poison, sizeof(poison));
        }
        for (uint64_t i = 0; i < state_count; i++) {
            const int kv_value =
                (int)((i * 5u + seed * 13u) % 97u) - 48;
            const int score_value =
                (int)((i * 7u + seed * 5u) % 101u) - 50;
            ref_state_kv_host[i] = (float)kv_value / 64.0f;
            fused_state_kv_host[i] = ref_state_kv_host[i];
            ref_state_score_host[i] = (float)score_value / 72.0f;
            fused_state_score_host[i] = ref_state_score_host[i];
        }
        if (test_decode_pack) {
            static const uint32_t edge_bits[8] = {
                0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
                0x3f800000u, 0xbf800000u, 0x42a00000u, 0xc2a00000u,
            };
            for (uint32_t col = 0; col < 4u; col++) {
                for (uint32_t row = 0; row < 8u; row++) {
                    const uint64_t state_col =
                        (row >= 4u ? head_dim : 0u) + col;
                    const uint64_t state_index =
                        (uint64_t)row * width + state_col;
                    const uint32_t score_bits =
                        edge_bits[(row + col) & 7u];
                    const uint32_t kv_bits =
                        edge_bits[(7u - row + col) & 7u];
                    memcpy(ref_state_score_host + state_index,
                           &score_bits, sizeof(score_bits));
                    memcpy(fused_state_score_host + state_index,
                           &score_bits, sizeof(score_bits));
                    memcpy(ref_state_kv_host + state_index,
                           &kv_bits, sizeof(kv_bits));
                    memcpy(fused_state_kv_host + state_index,
                           &kv_bits, sizeof(kv_bits));
                }
            }
        }
        for (uint32_t i = 0; i < head_dim; i++) {
            const uint32_t poison = 0x7fc01001u + (i & 0x3ffu);
            memcpy(ref_comp_host + i, &poison, sizeof(poison));
            memcpy(fused_comp_host + i, &poison, sizeof(poison));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_kv, 0, ref_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_score, 0, ref_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_kv, 0, fused_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_score, 0, fused_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, fused_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(unsetenv(pair_disable_env) == 0);
        TEST_ASSERT(unsetenv(store_disable_env) == 0);
        TEST_ASSERT(setenv(exact_reduction_disable_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(exact_reduction_poison_env) == 0);
        if (test_decode_pack) {
            TEST_ASSERT(setenv(decode_pack_disable_env, "1", 1) == 0);
        }

        TEST_ASSERT(ds4_gpu_matmul_f16_pair_tensor(
                        ref_kv, ref_score, model_raw, model_bytes,
                        0, score_weight_offset, in_dim, width, x, 1) != 0);
        TEST_ASSERT(ds4_gpu_compressor_update_tensor(
                        ref_kv, ref_score, ref_state_kv, ref_state_score,
                        ref_comp, model_raw, model_bytes, ape_offset, ape_type,
                        norm_offset, 0, head_dim, ratio, pos, 0, 0, 0,
                        10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
                        1.0e-6f, false, test_decode_pack, false) != 0);

        TEST_ASSERT(ds4_gpu_matmul_f16_pair_compressor_store_tensor(
                        fused_kv, fused_score,
                        fused_state_kv, fused_state_score,
                        model_raw, model_bytes, 0, score_weight_offset,
                        ape_offset, ape_type, in_dim, width, x,
                        ratio, pos) == 1);
        if (test_decode_pack) {
            TEST_ASSERT(unsetenv(decode_pack_disable_env) == 0);
            TEST_ASSERT(unsetenv(exact_reduction_disable_env) == 0);
            TEST_ASSERT(setenv(exact_reduction_poison_env, "1", 1) == 0);
        }
        TEST_ASSERT(ds4_gpu_compressor_update_tensor(
                        fused_kv, fused_score,
                        fused_state_kv, fused_state_score,
                        fused_comp, model_raw, model_bytes,
                        ape_offset, ape_type, norm_offset, 0,
                        head_dim, ratio, pos, 0, 0, 0,
                        10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
                        1.0e-6f, true, test_decode_pack, false) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_kv, 0, ref_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_score, 0, ref_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_kv, 0, fused_kv_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_score, 0, fused_score_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_kv, 0, fused_state_kv_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);

        kv_stats = test_compare_float_bits(
            ref_kv_host, fused_kv_host, width);
        score_stats = test_compare_float_bits(
            ref_score_host, fused_score_host, width);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, fused_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, fused_state_score_host,
            (size_t)state_count);
        comp_stats = test_compare_float_bits(
            ref_comp_host, fused_comp_host, head_dim);
    }

    test_restore_env(pair_disable_env, saved_pair_disable);
    test_restore_env(store_disable_env, saved_store_disable);
    test_restore_env(decode_pack_disable_env, saved_decode_pack_disable);
    test_restore_env(exact_reduction_disable_env,
                     saved_exact_reduction_disable);
    test_restore_env(
        exact_reduction_poison_env, saved_exact_reduction_poison);

    fprintf(stderr,
            "ds4-test: compressor pair state-store exact width=%u ratio=%u "
            "pos=%u emit=%u ape=%s decode_pack=%u exact_reduce=%u "
            "proj=%zu/%zu state=%zu/%zu "
            "comp=%zu max_ulp=%u/%u/%u/%u/%u\n",
            width, ratio, pos, emit ? 1u : 0u,
            ape_type == 1u ? "f16" : "f32",
            test_decode_pack ? 1u : 0u,
            test_decode_pack ? 1u : 0u,
            kv_stats.mismatch_count, score_stats.mismatch_count,
            state_kv_stats.mismatch_count,
            state_score_stats.mismatch_count,
            comp_stats.mismatch_count,
            kv_stats.max_ulp, score_stats.max_ulp,
            state_kv_stats.max_ulp, state_score_stats.max_ulp,
            comp_stats.max_ulp);
    TEST_ASSERT(kv_stats.mismatch_count == 0);
    TEST_ASSERT(score_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);
    TEST_ASSERT(comp_stats.mismatch_count == 0);

    free(fused_comp_host);
    free(ref_comp_host);
    free(fused_state_score_host);
    free(fused_state_kv_host);
    free(ref_state_score_host);
    free(ref_state_kv_host);
    free(fused_score_host);
    free(fused_kv_host);
    free(ref_score_host);
    free(ref_kv_host);
    free(x_host);
    ds4_gpu_tensor_free(fused_comp);
    ds4_gpu_tensor_free(ref_comp);
    ds4_gpu_tensor_free(fused_state_score);
    ds4_gpu_tensor_free(fused_state_kv);
    ds4_gpu_tensor_free(ref_state_score);
    ds4_gpu_tensor_free(ref_state_kv);
    ds4_gpu_tensor_free(fused_score);
    ds4_gpu_tensor_free(fused_kv);
    ds4_gpu_tensor_free(ref_score);
    ds4_gpu_tensor_free(ref_kv);
    ds4_gpu_tensor_free(x);
    free(model_raw);
}

static void test_metal_f16_compressor_pair_state_store_exact(void) {
    test_metal_f16_compressor_pair_state_store_exact_case(
        256, 4, 8, 0, 17, false);
    test_metal_f16_compressor_pair_state_store_exact_case(
        256, 4, 11, 1, 23, true);
    test_metal_f16_compressor_pair_state_store_exact_case(
        1024, 4, 11, 1, 29, true);
    test_metal_f16_compressor_pair_state_store_exact_case(
        512, 128, 255, 1, 43, false);
}

static void test_metal_compressor_ape_add_exact_case(
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t ape_type,
        uint32_t seed,
        bool test_pack_fusion) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t input_count = (uint64_t)n_tokens * width;
    const uint64_t state_count = (uint64_t)state_rows * width;
    const uint64_t comp_count = (uint64_t)n_comp * head_dim;
    const uint64_t input_bytes = input_count * sizeof(float);
    const uint64_t state_bytes = state_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(float);
    const uint64_t ape_elem_bytes = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)width * ratio * ape_elem_bytes;
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
    TEST_ASSERT(kv != NULL);
    TEST_ASSERT(sc != NULL);
    TEST_ASSERT(ref_comp != NULL);
    TEST_ASSERT(fused_comp != NULL);
    TEST_ASSERT(ref_state_kv != NULL);
    TEST_ASSERT(ref_state_score != NULL);
    TEST_ASSERT(fused_state_kv != NULL);
    TEST_ASSERT(fused_state_score != NULL);

    float *kv_host = malloc((size_t)input_bytes);
    float *sc_host = malloc((size_t)input_bytes);
    float *ref_comp_host = malloc((size_t)comp_bytes);
    float *fused_comp_host = malloc((size_t)comp_bytes);
    float *ref_state_kv_host = malloc((size_t)state_bytes);
    float *ref_state_score_host = malloc((size_t)state_bytes);
    float *fused_state_kv_host = malloc((size_t)state_bytes);
    float *fused_state_score_host = malloc((size_t)state_bytes);
    const uint64_t poison_count = input_count > state_count ?
        input_count : state_count;
    float *poison_host = test_pack_fusion ?
        malloc((size_t)(poison_count * sizeof(float))) : NULL;
    TEST_ASSERT(kv_host != NULL);
    TEST_ASSERT(sc_host != NULL);
    TEST_ASSERT(ref_comp_host != NULL);
    TEST_ASSERT(fused_comp_host != NULL);
    TEST_ASSERT(ref_state_kv_host != NULL);
    TEST_ASSERT(ref_state_score_host != NULL);
    TEST_ASSERT(fused_state_kv_host != NULL);
    TEST_ASSERT(fused_state_score_host != NULL);
    TEST_ASSERT(!test_pack_fusion || poison_host != NULL);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page,
                               (size_t)model_bytes) == 0);
    const bool allocated = kv && sc && ref_comp && fused_comp && ref_state_kv &&
        ref_state_score && fused_state_kv && fused_state_score && kv_host &&
        sc_host && ref_comp_host && fused_comp_host && ref_state_kv_host &&
        ref_state_score_host && fused_state_kv_host && fused_state_score_host &&
        (!test_pack_fusion || poison_host) && model_raw;
    const char *disable_env = "DS4_METAL_DISABLE_COMPRESSOR_APE_ADD";
    const char *pack_disable_env =
        "DS4_METAL_DISABLE_COMPRESSOR_RATIO4_PACK_FUSION";
    char *saved_disable = test_save_env(disable_env);
    char *saved_pack_disable = test_save_env(pack_disable_env);
    test_float_compare_stats comp_stats = {0};
    test_float_compare_stats state_kv_stats = {0};
    test_float_compare_stats state_score_stats = {0};

    if (allocated) {
        memset(model_raw, 0, (size_t)model_bytes);
        if (ape_type == 1u) {
            uint16_t *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)width * ratio; i++) {
                const int value = (int)((i * 17u + (i ^ (i >> 4u)) * 5u +
                                         seed * 13u) % 127u) - 63;
                ape[i] = test_float_to_f16((float)value / 80.0f);
            }
        } else {
            float *ape = model_raw;
            for (uint64_t i = 0; i < (uint64_t)width * ratio; i++) {
                const int value = (int)((i * 17u + (i ^ (i >> 4u)) * 5u +
                                         seed * 13u) % 127u) - 63;
                ape[i] = (float)value / 80.0f;
            }
        }
        float *norm = (float *)((uint8_t *)model_raw + norm_offset);
        for (uint32_t i = 0; i < head_dim; i++) {
            norm[i] = 0.75f + (float)((i * 7u + seed) % 19u) / 64.0f;
        }
        for (uint64_t i = 0; i < input_count; i++) {
            const int kv_value = (int)((i * 29u + (i ^ (i >> 5u)) * 3u +
                                        seed * 23u) % 181u) - 90;
            const int sc_value = (int)((i * 31u + (i ^ (i >> 3u)) * 11u +
                                        seed * 17u) % 173u) - 86;
            kv_host[i] = (float)kv_value / 112.0f;
            sc_host[i] = (float)sc_value / 96.0f;
        }
        kv_host[0] = -0.0f;
        sc_host[0] = -0.0f;

        // Exercise exact-add edge values in the first active APE row. Using
        // bit patterns avoids host fast-math rewriting signed zeros or
        // subnormals before the legacy and fused Metal paths see them.
        const uint32_t cutoff = (n_tokens / ratio) * ratio;
        const uint32_t edge_token = cutoff < n_tokens ? cutoff : cutoff - ratio;
        const uint64_t edge_ape =
            (uint64_t)((pos0 + edge_token) % ratio) * width;
        float *edge_score = sc_host + (uint64_t)edge_token * width;
        static const uint16_t edge_f16[] = {
            0x0000u, 0x8000u, 0x0001u, 0x8001u,
            0x3c00u, 0xbc00u, 0x7bffu, 0xfbffu,
        };
        static const uint32_t edge_f32[] = {
            0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
            0x3f800000u, 0xbf800000u, 0x7f7fffffu, 0xff7fffffu,
        };
        static const uint32_t edge_score_f16[] = {
            0x80000000u, 0x00000000u, 0x00000000u, 0x80000000u,
            0xbf800000u, 0x3f800000u, 0xc77fe000u, 0x477fe000u,
        };
        static const uint32_t edge_score_f32[] = {
            0x80000000u, 0x00000000u, 0x00000000u, 0x80000000u,
            0xbf800000u, 0x3f800000u, 0xff7fffffu, 0x7f7fffffu,
        };
        if (ape_type == 1u) {
            uint16_t *ape = model_raw;
            memcpy(ape + edge_ape, edge_f16, sizeof(edge_f16));
            for (uint32_t i = 0; i < 8u; i++) {
                memcpy(edge_score + i, edge_score_f16 + i, sizeof(uint32_t));
            }
        } else {
            uint32_t *ape = model_raw;
            memcpy(ape + edge_ape, edge_f32, sizeof(edge_f32));
            for (uint32_t i = 0; i < 8u; i++) {
                memcpy(edge_score + i, edge_score_f32 + i, sizeof(uint32_t));
            }
        }
        for (uint64_t i = 0; i < state_count; i++) {
            ref_state_kv_host[i] = 1234.0f;
            fused_state_kv_host[i] = 1234.0f;
            ref_state_score_host[i] = -1234.0f;
            fused_state_score_host[i] = -1234.0f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(kv, 0, kv_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(sc, 0, sc_host, input_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_kv, 0, fused_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        ref_state_score, 0, ref_state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        fused_state_score, 0, fused_state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_bytes) != 0);
        ds4_gpu_set_quality(false);

        if (test_pack_fusion) {
            TEST_ASSERT(unsetenv(disable_env) == 0);
        } else {
            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        }
        TEST_ASSERT(setenv(pack_disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_compressor_prefill_tensor(
            ref_comp, ref_state_kv, ref_state_score, kv, sc,
            model_raw, model_bytes, 0, ape_type, norm_offset, 0,
            head_dim, ratio, pos0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        if (test_pack_fusion) {
            // Overwrite every persistent pack cell with qNaN payloads through
            // the legacy replay path. This makes an omitted candidate write
            // observable even for plane-zero padding, which normal legacy
            // packing would otherwise leave at the correct 0/-inf values.
            for (uint64_t i = 0; i < poison_count; i++) {
                const uint32_t bits =
                    0x7fc00001u + (uint32_t)(i & 0x3ffu);
                memcpy(poison_host + i, &bits, sizeof(bits));
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                            kv, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            sc, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_kv, 0, poison_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_score, 0, poison_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_compressor_prefill_ratio4_replay_tensor(
                fused_comp, fused_state_kv, fused_state_score, kv, sc,
                model_raw, model_bytes, 0, ape_type, norm_offset, 0,
                head_dim, 0, n_comp * ratio, 0, 0, false,
                10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            kv, 0, kv_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            sc, 0, sc_host, input_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_kv, 0, fused_state_kv_host,
                            state_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_state_score, 0, fused_state_score_host,
                            state_bytes) != 0);
            TEST_ASSERT(unsetenv(pack_disable_env) == 0);
        } else {
            TEST_ASSERT(unsetenv(disable_env) == 0);
        }
        TEST_ASSERT(ds4_gpu_compressor_prefill_tensor(
            fused_comp, fused_state_kv, fused_state_score, kv, sc,
            model_raw, model_bytes, 0, ape_type, norm_offset, 0,
            head_dim, ratio, pos0, n_tokens, 0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1.0e-6f) != 0);

        if (test_pack_fusion) {
            TEST_ASSERT(ds4_gpu_tensor_read(
                            kv, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(memcmp(kv_host, poison_host,
                               (size_t)input_bytes) == 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            sc, 0, poison_host, input_bytes) != 0);
            TEST_ASSERT(memcmp(sc_host, poison_host,
                               (size_t)input_bytes) == 0);
        }
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_comp, 0, ref_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_comp, 0, fused_comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_kv, 0, ref_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_kv, 0, fused_state_kv_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_state_score, 0, ref_state_score_host, state_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        fused_state_score, 0, fused_state_score_host,
                        state_bytes) != 0);

        comp_stats = test_compare_float_bits(
            ref_comp_host, fused_comp_host, (size_t)comp_count);
        state_kv_stats = test_compare_float_bits(
            ref_state_kv_host, fused_state_kv_host, (size_t)state_count);
        state_score_stats = test_compare_float_bits(
            ref_state_score_host, fused_state_score_host, (size_t)state_count);
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(pack_disable_env, saved_pack_disable);
    fprintf(stderr,
            "ds4-test: compressor %s exact head=%u ratio=%u pos=%u "
            "tokens=%u ape=%s comp=%zu/%llu state_kv=%zu/%llu "
            "state_score=%zu/%llu max_ulp=%u/%u/%u\n",
            test_pack_fusion ? "ratio4 pack" : "APE add",
            head_dim, ratio, pos0, n_tokens, ape_type == 1u ? "f16" : "f32",
            comp_stats.mismatch_count, (unsigned long long)comp_count,
            state_kv_stats.mismatch_count, (unsigned long long)state_count,
            state_score_stats.mismatch_count, (unsigned long long)state_count,
            comp_stats.max_ulp, state_kv_stats.max_ulp,
            state_score_stats.max_ulp);
    TEST_ASSERT(comp_stats.mismatch_count == 0);
    TEST_ASSERT(state_kv_stats.mismatch_count == 0);
    TEST_ASSERT(state_score_stats.mismatch_count == 0);

    free(model_raw);
    free(poison_host);
    free(fused_state_score_host);
    free(fused_state_kv_host);
    free(ref_state_score_host);
    free(ref_state_kv_host);
    free(fused_comp_host);
    free(ref_comp_host);
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

static void test_metal_compressor_ape_add_exact(void) {
    test_metal_compressor_ape_add_exact_case(128, 4, 3, 16, 1, 7, false);
    test_metal_compressor_ape_add_exact_case(512, 4, 0, 16, 1, 13, false);
    test_metal_compressor_ape_add_exact_case(512, 4, 1, 14, 0, 19, false);
    test_metal_compressor_ape_add_exact_case(512, 128, 127, 257, 1, 31, false);
}

