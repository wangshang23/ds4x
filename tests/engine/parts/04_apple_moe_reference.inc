static void test_metal_hc_split_weighted_sum_norm_batch_exact(void) {
    /* Compare the batched HC+RMSNorm fusion against the exact two-dispatch
     * sequence used by the reference path at DS4's production dimensions. */
    const uint32_t n_embd = 7168;
    const uint32_t n_hc = 4;
    const uint32_t n_rows = 3;
    const uint32_t sinkhorn_iters = 20;
    const float hc_eps = 1.0e-6f;
    const float norm_eps = 1.0e-6f;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t scale_offset = 0;
    const uint64_t base_offset = test_round_up_u64(3u * sizeof(float), page);
    const uint64_t norm_weight_offset =
        test_round_up_u64(base_offset + mix_hc * sizeof(float), page);
    const uint64_t model_alloc = test_round_up_u64(
        norm_weight_offset + (uint64_t)n_embd * sizeof(float), page);
    const uint64_t mix_bytes = (uint64_t)n_rows * mix_hc * sizeof(float);
    const uint64_t residual_bytes =
        (uint64_t)n_rows * n_hc * n_embd * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_rows * n_embd * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)model_alloc) == 0);
    if (!model_raw) return;
    memset(model_raw, 0, (size_t)model_alloc);

    float *scale = (float *)((uint8_t *)model_raw + scale_offset);
    float *base = (float *)((uint8_t *)model_raw + base_offset);
    float *norm_weight = (float *)((uint8_t *)model_raw + norm_weight_offset);
    scale[0] = 0.625f;
    scale[1] = -0.75f;
    scale[2] = 0.4375f;
    for (uint32_t i = 0; i < mix_hc; i++) {
        const int value = (int)((i * 17u + 5u) % 29u) - 14;
        base[i] = (float)value / 16.0f;
    }
    for (uint32_t i = 0; i < n_embd; i++) {
        norm_weight[i] = 0.5f + (float)((i * 13u + 7u) % 31u) / 32.0f;
    }

    ds4_gpu_tensor *mix = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc(residual_bytes);
    ds4_gpu_tensor *ref_split = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *fused_split = ds4_gpu_tensor_alloc(mix_bytes);
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *fused_norm = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(mix != NULL);
    TEST_ASSERT(residual != NULL);
    TEST_ASSERT(ref_split != NULL);
    TEST_ASSERT(fused_split != NULL);
    TEST_ASSERT(ref_out != NULL);
    TEST_ASSERT(fused_out != NULL);
    TEST_ASSERT(ref_norm != NULL);
    TEST_ASSERT(fused_norm != NULL);
    if (!mix || !residual || !ref_split || !fused_split ||
        !ref_out || !fused_out || !ref_norm || !fused_norm) {
        ds4_gpu_tensor_free(mix);
        ds4_gpu_tensor_free(residual);
        ds4_gpu_tensor_free(ref_split);
        ds4_gpu_tensor_free(fused_split);
        ds4_gpu_tensor_free(ref_out);
        ds4_gpu_tensor_free(fused_out);
        ds4_gpu_tensor_free(ref_norm);
        ds4_gpu_tensor_free(fused_norm);
        free(model_raw);
        return;
    }

    float *mix_host = malloc((size_t)mix_bytes);
    float *residual_host = malloc((size_t)residual_bytes);
    float *ref_split_host = malloc((size_t)mix_bytes);
    float *fused_split_host = malloc((size_t)mix_bytes);
    float *ref_out_host = malloc((size_t)out_bytes);
    float *fused_out_host = malloc((size_t)out_bytes);
    float *ref_norm_host = malloc((size_t)out_bytes);
    float *fused_norm_host = malloc((size_t)out_bytes);
    TEST_ASSERT(mix_host != NULL);
    TEST_ASSERT(residual_host != NULL);
    TEST_ASSERT(ref_split_host != NULL);
    TEST_ASSERT(fused_split_host != NULL);
    TEST_ASSERT(ref_out_host != NULL);
    TEST_ASSERT(fused_out_host != NULL);
    TEST_ASSERT(ref_norm_host != NULL);
    TEST_ASSERT(fused_norm_host != NULL);
    if (!mix_host || !residual_host || !ref_split_host || !fused_split_host ||
        !ref_out_host || !fused_out_host || !ref_norm_host || !fused_norm_host) {
        free(mix_host);
        free(residual_host);
        free(ref_split_host);
        free(fused_split_host);
        free(ref_out_host);
        free(fused_out_host);
        free(ref_norm_host);
        free(fused_norm_host);
        ds4_gpu_tensor_free(mix);
        ds4_gpu_tensor_free(residual);
        ds4_gpu_tensor_free(ref_split);
        ds4_gpu_tensor_free(fused_split);
        ds4_gpu_tensor_free(ref_out);
        ds4_gpu_tensor_free(fused_out);
        ds4_gpu_tensor_free(ref_norm);
        ds4_gpu_tensor_free(fused_norm);
        free(model_raw);
        return;
    }

    for (uint32_t row = 0; row < n_rows; row++) {
        for (uint32_t i = 0; i < mix_hc; i++) {
            const int value =
                (int)(((row + 1u) * 19u + i * 11u + (i ^ row) * 3u) % 47u) - 23;
            mix_host[(uint64_t)row * mix_hc + i] = (float)value / 9.0f;
        }
        for (uint32_t hc = 0; hc < n_hc; hc++) {
            for (uint32_t d = 0; d < n_embd; d++) {
                const uint32_t key =
                    d * 37u + hc * 173u + row * 997u + ((d >> 3u) ^ (d * 7u));
                const int value = (int)(key % 2047u) - 1023;
                residual_host[((uint64_t)row * n_hc + hc) * n_embd + d] =
                    (float)value / 512.0f;
            }
        }
    }

    TEST_ASSERT(ds4_gpu_tensor_write(mix, 0, mix_host, mix_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(residual, 0, residual_host, residual_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
    TEST_ASSERT(ds4_gpu_hc_split_weighted_sum_tensor(
        ref_out, ref_split, mix, residual,
        model_raw, model_alloc, scale_offset, base_offset,
        n_embd, n_hc, sinkhorn_iters, hc_eps) != 0);
    TEST_ASSERT(ds4_gpu_rms_norm_weight_rows_tensor(
        ref_norm, ref_out, model_raw, model_alloc, norm_weight_offset,
        n_embd, n_rows, norm_eps) != 0);
    TEST_ASSERT(ds4_gpu_hc_split_weighted_sum_norm_tensor(
        fused_out, fused_norm, fused_split, mix, residual,
        model_raw, model_alloc, scale_offset, base_offset, norm_weight_offset,
        n_embd, n_hc, sinkhorn_iters, hc_eps, norm_eps) != 0);

    TEST_ASSERT(ds4_gpu_tensor_read(ref_split, 0, ref_split_host, mix_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(fused_split, 0, fused_split_host, mix_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_out, 0, ref_out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(fused_out, 0, fused_out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(ref_norm, 0, ref_norm_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(fused_norm, 0, fused_norm_host, out_bytes) != 0);

    const test_float_compare_stats split_stats = test_compare_float_bits(
        ref_split_host, fused_split_host, (size_t)(n_rows * mix_hc));
    const test_float_compare_stats out_stats = test_compare_float_bits(
        ref_out_host, fused_out_host, (size_t)n_rows * n_embd);
    const test_float_compare_stats norm_stats = test_compare_float_bits(
        ref_norm_host, fused_norm_host, (size_t)n_rows * n_embd);
    fprintf(stderr,
            "ds4-test: batch HC+RMSNorm exactness rows=%u "
            "split=%zu/%llu max_ulp=%u max_abs=%g, "
            "collapse=%zu/%llu max_ulp=%u max_abs=%g, "
            "norm=%zu/%llu max_ulp=%u max_abs=%g\n",
            n_rows,
            split_stats.mismatch_count,
            (unsigned long long)(n_rows * mix_hc),
            split_stats.max_ulp,
            split_stats.max_abs,
            out_stats.mismatch_count,
            (unsigned long long)((uint64_t)n_rows * n_embd),
            out_stats.max_ulp,
            out_stats.max_abs,
            norm_stats.mismatch_count,
            (unsigned long long)((uint64_t)n_rows * n_embd),
            norm_stats.max_ulp,
            norm_stats.max_abs);
    TEST_ASSERT(split_stats.mismatch_count == 0);
    TEST_ASSERT(out_stats.mismatch_count == 0);
    TEST_ASSERT(norm_stats.mismatch_count == 0);

    free(mix_host);
    free(residual_host);
    free(ref_split_host);
    free(fused_split_host);
    free(ref_out_host);
    free(fused_out_host);
    free(ref_norm_host);
    free(fused_norm_host);
    ds4_gpu_tensor_free(mix);
    ds4_gpu_tensor_free(residual);
    ds4_gpu_tensor_free(ref_split);
    ds4_gpu_tensor_free(fused_split);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(fused_out);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(fused_norm);
    free(model_raw);
}

static void test_metal_output_hc_weights4_exact(void) {
    const uint32_t n_hc = 4;
    const float eps = 1.0e-6f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t scale_offset = 0;
    const uint64_t base_offset = page;
    const uint64_t model_alloc = 2u * page;
    const uint64_t bytes = n_hc * sizeof(float);
    const char *require_env = "DS4_METAL_REQUIRE_OUTPUT_HC_WEIGHTS4";
    char *saved_require = test_save_env(require_env);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_alloc) == 0);
    ds4_gpu_tensor *pre = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *reference = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *candidate = ds4_gpu_tensor_alloc(bytes);
    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(pre != NULL);
    TEST_ASSERT(reference != NULL);
    TEST_ASSERT(candidate != NULL);

    size_t total_mismatch = 0;
    uint32_t max_ulp = 0;
    const bool allocated = model_raw && pre && reference && candidate;
    if (allocated) {
        memset(model_raw, 0, (size_t)model_alloc);
        float *scale = (float *)((uint8_t *)model_raw + scale_offset);
        float *base = (float *)((uint8_t *)model_raw + base_offset);

        for (uint32_t ci = 0; ci < 4; ci++) {
            float pre_host[4];
            float ref_host[4];
            float candidate_host[4];
            for (uint32_t i = 0; i < 4; i++) {
                pre_host[i] =
                    ((float)((int)(ci * 17u + i * 11u) - 23)) / 8.0f;
                base[i] =
                    ((float)((int)(ci * 13u + i * 7u) - 19)) / 16.0f;
            }
            scale[0] = 0.62500012f + (float)ci * 0.125f;

            if (ci == 1) {
                /* The first lane distinguishes the required two-rounding
                 * sequence from an illegally contracted multiply-add. */
                const uint32_t pre_bits = 0x4620a541u;
                const uint32_t scale_bits = 0x462483bau;
                const uint32_t base_bits = 0xccce790eu;
                memcpy(&pre_host[0], &pre_bits, sizeof(pre_bits));
                memcpy(&scale[0], &scale_bits, sizeof(scale_bits));
                memcpy(&base[0], &base_bits, sizeof(base_bits));
            } else if (ci == 2) {
                const uint32_t pre_bits[4] = {
                    0x00000000u, 0x80000000u,
                    0x00000001u, 0x80000001u,
                };
                const uint32_t base_bits[4] = {
                    0x80000000u, 0x00000000u,
                    0x00800000u, 0x80800000u,
                };
                for (uint32_t i = 0; i < 4; i++) {
                    memcpy(&pre_host[i], &pre_bits[i], sizeof(uint32_t));
                    memcpy(&base[i], &base_bits[i], sizeof(uint32_t));
                }
                scale[0] = -1.00000012f;
            } else if (ci == 3) {
                pre_host[0] = 100.0f;
                pre_host[1] = -100.0f;
                pre_host[2] = 88.0f;
                pre_host[3] = -88.0f;
                scale[0] = 1.0f;
                for (uint32_t i = 0; i < 4; i++) base[i] = 0.0f;
            }

            memset(ref_host, 0xa5, sizeof(ref_host));
            memset(candidate_host, 0xa5, sizeof(candidate_host));
            TEST_ASSERT(ds4_gpu_tensor_write(pre, 0, pre_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            reference, 0, ref_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            candidate, 0, candidate_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
            TEST_ASSERT(unsetenv(require_env) == 0);
            ds4_gpu_test_set_flags(0);
            ds4_gpu_set_quality(true);
            TEST_ASSERT(ds4_gpu_output_hc_weights_tensor(
                            reference, pre, model_raw, model_alloc,
                            scale_offset, base_offset, n_hc, eps) != 0);

            ds4_gpu_set_quality(false);
            ds4_gpu_test_set_flags(DS4_GPU_TEST_OUTPUT_HC_WEIGHTS4);
            TEST_ASSERT(setenv(require_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_output_hc_weights_tensor(
                            candidate, pre, model_raw, model_alloc,
                            scale_offset, base_offset, n_hc, eps) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                            reference, 0, ref_host, bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            candidate, 0, candidate_host, bytes) != 0);
            const test_float_compare_stats stats =
                test_compare_float_bits(ref_host, candidate_host, n_hc);
            fprintf(stderr,
                    "ds4-test: output HC weights4 exact case=%u "
                    "mismatch=%zu/%u max_ulp=%u max_abs=%g\n",
                    ci, stats.mismatch_count, n_hc,
                    stats.max_ulp, stats.max_abs);
            TEST_ASSERT(stats.mismatch_count == 0);
            total_mismatch += stats.mismatch_count;
            if (stats.max_ulp > max_ulp) max_ulp = stats.max_ulp;
        }

        /* Quality mode prevents strict selection of the fast path. */
        TEST_ASSERT(setenv(require_env, "1", 1) == 0);
        ds4_gpu_set_quality(true);
        TEST_ASSERT(ds4_gpu_output_hc_weights_tensor(
                        candidate, pre, model_raw, model_alloc,
                        scale_offset, base_offset, n_hc, eps) == 0);
        ds4_gpu_set_quality(false);
        ds4_gpu_test_set_flags(0);
    }

    test_restore_env(require_env, saved_require);
    fprintf(stderr,
            "ds4-test: output HC weights4 total mismatch=%zu/16 max_ulp=%u\n",
            total_mismatch, max_ulp);
    TEST_ASSERT(total_mismatch == 0);
    TEST_ASSERT(max_ulp == 0);

    ds4_gpu_tensor_free(candidate);
    ds4_gpu_tensor_free(reference);
    ds4_gpu_tensor_free(pre);
    free(model_raw);
}

static void test_metal_hc_rms_scale_project_f16_exact_shape(
        uint32_t in_dim,
        uint32_t seed) {
    const uint32_t out_dim = 24;
    /* One full 32-row matmul tile plus a tail row covers both load paths. */
    const uint32_t n_rows = 33;
    const float eps = 1.0e-6f;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t weight_offset = page;
    const uint64_t weight_bytes =
        (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t model_alloc = test_round_up_u64(
        weight_offset + weight_bytes, page);
    const uint64_t x_count = (uint64_t)in_dim * n_rows;
    const uint64_t out_count = (uint64_t)out_dim * n_rows;
    const uint64_t x_bytes = x_count * sizeof(float);
    const uint64_t out_bytes = out_count * sizeof(float);
    const uint64_t scale_bytes = (uint64_t)n_rows * sizeof(float);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(
                    &model_raw, (size_t)page, (size_t)model_alloc) == 0);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_norm = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *ref_out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *scaled_out = ds4_gpu_tensor_alloc(out_bytes);
    /* Deliberately too small for the full-RMS fallback. */
    ds4_gpu_tensor *scale_scratch = ds4_gpu_tensor_alloc(scale_bytes);
    float *x_host = malloc((size_t)x_bytes);
    float *ref_norm_host = malloc((size_t)x_bytes);
    float *ref_out_host = malloc((size_t)out_bytes);
    float *scaled_out_host = malloc((size_t)out_bytes);
    float *scale_host = malloc((size_t)scale_bytes);
    float *expected_scale = malloc((size_t)scale_bytes);

    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(ref_norm != NULL);
    TEST_ASSERT(ref_out != NULL);
    TEST_ASSERT(scaled_out != NULL);
    TEST_ASSERT(scale_scratch != NULL);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(ref_norm_host != NULL);
    TEST_ASSERT(ref_out_host != NULL);
    TEST_ASSERT(scaled_out_host != NULL);
    TEST_ASSERT(scale_host != NULL);
    TEST_ASSERT(expected_scale != NULL);

    const bool allocated = model_raw && x && ref_norm && ref_out &&
        scaled_out && scale_scratch && x_host && ref_norm_host &&
        ref_out_host && scaled_out_host && scale_host && expected_scale;
    test_float_compare_stats scale_stats = {0};
    test_float_compare_stats out_stats = {0};
    if (allocated) {
        memset(model_raw, 0, (size_t)model_alloc);
        uint16_t *weights = (uint16_t *)((uint8_t *)model_raw + weight_offset);
        for (uint32_t o = 0; o < out_dim; o++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const uint32_t key = i * 37u + o * 1009u + seed * 53u +
                    ((i >> 4u) ^ (o * 17u));
                uint16_t bits;
                if (key % 257u == 0u) {
                    bits = (key & 1u) ? 0x8000u : 0x0000u;
                } else if (key % 263u == 0u) {
                    bits = (key & 1u) ? 0x8001u : 0x0001u;
                } else if (key % 269u == 0u) {
                    bits = (key & 1u) ? 0x8400u : 0x0400u;
                } else {
                    const int value = (int)(key % 127u) - 63;
                    bits = test_float_to_f16((float)value / 128.0f);
                }
                weights[(uint64_t)o * in_dim + i] = bits;
            }
        }

        static const uint32_t rounding_bits[] = {
            0x3f800fffu, 0x3f801000u, 0x3f801001u,
            0x3f802fffu, 0x3f803000u, 0x3f803001u,
            0xbf800fffu, 0xbf801000u, 0xbf801001u,
            0x3eaaaaabu, 0xbeaaaaabu,
        };
        static const int sentinel_exp[] = { -2, -10, 0, -4, -6, -1 };
        for (uint32_t row = 0; row < n_rows; row++) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const uint32_t key = i * 131u + row * 977u + seed * 71u +
                    ((i >> 3u) ^ (row * 29u));
                const float sign = (key & 1u) ? -1.0f : 1.0f;
                float value;
                switch (row % 6u) {
                    case 0:
                        value = (float)((int)(key % 4093u) - 2046) / 512.0f;
                        break;
                    case 1:
                        value = sign * ldexpf(
                            (float)(1u + (key & 7u)) / 8.0f, -19);
                        break;
                    case 2:
                        value = i % 257u == row % 257u
                            ? sign * (16.0f + (float)(key & 7u))
                            : sign * (float)(1u + (key & 31u)) / 8192.0f;
                        break;
                    case 3: {
                        const uint32_t bits = rounding_bits[
                            key % (sizeof(rounding_bits) /
                                   sizeof(rounding_bits[0]))];
                        memcpy(&value, &bits, sizeof(value));
                        break;
                    }
                    case 4:
                        if ((key & 7u) == 0u) {
                            const uint32_t bits = (key & 8u) ? 0x80000000u : 0u;
                            memcpy(&value, &bits, sizeof(value));
                        } else {
                            value = sign * ldexpf(
                                1.0f + (float)(key & 3u) * 0.25f,
                                (int)((key >> 4u) % 14u) - 10);
                        }
                        break;
                    default:
                        value = sign * (float)(1u + (key % 251u)) / 256.0f;
                        break;
                }
                x_host[(uint64_t)row * in_dim + i] = value;
            }
            x_host[(uint64_t)row * in_dim] =
                ldexpf(1.0f, sentinel_exp[row % 6u]);
        }

        for (uint64_t i = 0; i < out_count; i++) {
            const uint32_t bits = 0x7fc01000u + (uint32_t)(i & 0xfffu);
            memcpy(scaled_out_host + i, &bits, sizeof(bits));
        }
        for (uint32_t row = 0; row < n_rows; row++) {
            const uint32_t bits = 0x7fc02000u + row;
            memcpy(scale_host + row, &bits, sizeof(bits));
        }

        TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        scaled_out, 0, scaled_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        scale_scratch, 0, scale_host, scale_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_alloc) != 0);
        ds4_gpu_set_quality(false);

        int ref_begun = ds4_gpu_begin_commands();
        int ref_ok = ref_begun;
        if (ref_ok) ref_ok = ds4_gpu_rms_norm_plain_rows_tensor(
            ref_norm, x, in_dim, n_rows, eps);
        if (ref_ok) ref_ok = ds4_gpu_matmul_f16_tensor(
            ref_out, model_raw, model_alloc, weight_offset,
            in_dim, out_dim, ref_norm, n_rows);
        const int ref_end = ref_begun ? ds4_gpu_end_commands() : 0;
        TEST_ASSERT(ref_ok != 0);
        TEST_ASSERT(ref_end != 0);

        int scaled_begun = ds4_gpu_begin_commands();
        int scaled_ok = scaled_begun;
        if (scaled_ok) scaled_ok = ds4_gpu_hc_rms_scale_project_f16_tensor(
            scaled_out, scale_scratch, model_raw, model_alloc, weight_offset,
            in_dim, out_dim, x, n_rows, eps);
        const int scaled_end = scaled_begun ? ds4_gpu_end_commands() : 0;
        TEST_ASSERT(scaled_ok != 0);
        TEST_ASSERT(scaled_end != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_norm, 0, ref_norm_host, x_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        ref_out, 0, ref_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        scaled_out, 0, scaled_out_host, out_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        scale_scratch, 0, scale_host, scale_bytes) != 0);

        for (uint32_t row = 0; row < n_rows; row++) {
            expected_scale[row] = ldexpf(
                ref_norm_host[(uint64_t)row * in_dim],
                -sentinel_exp[row % 6u]);
            TEST_ASSERT(isfinite(expected_scale[row]));
            TEST_ASSERT(isfinite(scale_host[row]));
        }
        scale_stats = test_compare_float_bits(
            expected_scale, scale_host, n_rows);
        out_stats = test_compare_float_bits(
            ref_out_host, scaled_out_host, (size_t)out_count);
    }

    fprintf(stderr,
            "ds4-test: HC RMS scale F16 projection exact K=%u rows=%u "
            "scale=%zu/%u max_ulp=%u projection=%zu/%llu max_ulp=%u\n",
            in_dim, n_rows,
            scale_stats.mismatch_count, n_rows, scale_stats.max_ulp,
            out_stats.mismatch_count, (unsigned long long)out_count,
            out_stats.max_ulp);
    TEST_ASSERT(scale_stats.mismatch_count == 0);
    TEST_ASSERT(scale_stats.max_ulp == 0);
    TEST_ASSERT(out_stats.mismatch_count == 0);
    TEST_ASSERT(out_stats.max_ulp == 0);

    free(expected_scale);
    free(scale_host);
    free(scaled_out_host);
    free(ref_out_host);
    free(ref_norm_host);
    free(x_host);
    ds4_gpu_tensor_free(scale_scratch);
    ds4_gpu_tensor_free(scaled_out);
    ds4_gpu_tensor_free(ref_out);
    ds4_gpu_tensor_free(ref_norm);
    ds4_gpu_tensor_free(x);
    free(model_raw);
}

static void test_metal_hc_rms_scale_project_f16_exact(void) {
    const char *disable_env = "DS4_METAL_DISABLE_HC_RMS_SCALE_PROJ";
    char *saved_disable = test_save_env(disable_env);

    TEST_ASSERT(unsetenv(disable_env) == 0);
    ds4_gpu_test_set_flags(DS4_GPU_TEST_HC_RMS_SCALE_PROJ);
    test_metal_hc_rms_scale_project_f16_exact_shape(16384u, 59u);
    test_metal_hc_rms_scale_project_f16_exact_shape(28672u, 61u);
    ds4_gpu_test_set_flags(0);

    test_restore_env(disable_env, saved_disable);
}

static void test_metal_router_simd_finalize_exact(void) {
    typedef struct {
        const char *name;
        bool has_bias;
        uint32_t pattern;
    } router_case;
    static const router_case cases[] = {
        { "unique", false, 0 },
        { "bias", true, 0 },
        { "top6-ties", false, 1 },
        { "signed-zero-extremes", false, 2 },
        { "clamp-underflow", false, 3 },
        { "sum-rounding", true, 4 },
    };
    const uint32_t n_expert = 256;
    const uint32_t n_used = 6;
    const uint32_t modes = 4;
    const uint64_t probs_bytes = (uint64_t)n_expert * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)n_used * sizeof(int32_t);
    const uint64_t weights_bytes = (uint64_t)n_used * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();
    const char *disable_env =
        "DS4_METAL_DISABLE_PRE_M5_ROUTER_SIMD_FINALIZE";
    const char *weights_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_ROUTER_SIMD_WEIGHTS_FUSION";
    const char *transform_finalize_disable_env =
        "DS4_METAL_DISABLE_PRE_M5_ROUTER_TRANSFORM_FINALIZE_FUSION";
    const char *select_disable_env =
        "DS4_METAL_DISABLE_ROUTER_SELECT_FUSION";

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(probs_bytes);
    ds4_gpu_tensor *ref_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *simd_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *ref_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *simd_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *ref_probs = ds4_gpu_tensor_alloc(probs_bytes);
    ds4_gpu_tensor *simd_probs = ds4_gpu_tensor_alloc(probs_bytes);
    float *logits_host = malloc((size_t)probs_bytes);
    float *ref_probs_host = malloc((size_t)probs_bytes);
    float *simd_probs_host = malloc((size_t)probs_bytes);
    int32_t ref_selected_host[6];
    int32_t simd_selected_host[6];
    int32_t unique_selected_host[6];
    float ref_weights_host[6];
    float simd_weights_host[6];
    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(logits != NULL);
    TEST_ASSERT(ref_selected != NULL);
    TEST_ASSERT(simd_selected != NULL);
    TEST_ASSERT(ref_weights != NULL);
    TEST_ASSERT(simd_weights != NULL);
    TEST_ASSERT(ref_probs != NULL);
    TEST_ASSERT(simd_probs != NULL);
    TEST_ASSERT(logits_host != NULL);
    TEST_ASSERT(ref_probs_host != NULL);
    TEST_ASSERT(simd_probs_host != NULL);

    char *saved_disable = test_save_env(disable_env);
    char *saved_weights_disable = test_save_env(weights_disable_env);
    char *saved_transform_finalize_disable =
        test_save_env(transform_finalize_disable_env);
    char *saved_select_disable = test_save_env(select_disable_env);
    size_t total_selected_mismatch = 0;
    size_t total_weights_mismatch = 0;
    size_t total_probs_mismatch = 0;
    const bool allocated = model_raw && logits && ref_selected &&
        simd_selected && ref_weights && simd_weights && ref_probs &&
        simd_probs && logits_host && ref_probs_host && simd_probs_host;
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *bias = model_raw;
        for (uint32_t i = 0; i < n_used; i++) {
            bias[200u + i] = 16.0f - (float)i;
        }
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(setenv(transform_finalize_disable_env, "1", 1) == 0);
        TEST_ASSERT(unsetenv(select_disable_env) == 0);

        for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
            const router_case *c = &cases[ci];
            for (uint32_t i = 0; i < n_expert; i++) {
                const int value =
                    (int)((i * 47u + (i ^ (i >> 3u)) * 13u) % 257u) - 128;
                logits_host[i] = (float)value / 32.0f;
            }
            if (c->pattern == 1) {
                static const uint32_t tied[] = {
                    7u, 19u, 43u, 71u, 103u, 149u, 211u, 239u,
                };
                for (uint32_t i = 0; i < n_expert; i++) {
                    logits_host[i] = -7.0f - (float)(i % 17u) / 64.0f;
                }
                for (size_t i = 0; i < sizeof(tied) / sizeof(tied[0]); i++) {
                    logits_host[tied[i]] = 1.0f;
                }
            } else if (c->pattern == 2) {
                for (uint32_t i = 0; i < n_expert; i++) {
                    const uint32_t zero_bits = (i & 1u) ? 0x80000000u : 0u;
                    memcpy(&logits_host[i], &zero_bits, sizeof(zero_bits));
                }
                logits_host[3] = 80.0f;
                logits_host[17] = 40.0f;
                logits_host[61] = 20.0f;
                logits_host[127] = -20.0f;
                logits_host[193] = -40.0f;
                logits_host[251] = -80.0f;
            } else if (c->pattern == 3) {
                for (uint32_t i = 0; i < n_expert; i++) {
                    logits_host[i] = -30.0f - (float)(i % 11u);
                }
            } else if (c->pattern == 4) {
                static const float rounding_logits[6] = {
                    -0.6356699467f,
                    -0.8182631135f,
                    -2.7906901836f,
                    -3.4414808750f,
                    -3.4991359711f,
                    -3.1251864433f,
                };
                for (uint32_t i = 0; i < n_expert; i++) {
                    logits_host[i] = -20.0f;
                }
                for (uint32_t i = 0; i < n_used; i++) {
                    logits_host[200u + i] = rounding_logits[i];
                }
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                logits, 0, logits_host, probs_bytes) != 0);

            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_router_select_tensor(
                ref_selected, ref_weights, ref_probs,
                model_raw, page, 0, 0, 1, 0,
                n_expert, n_used, 1.5f, 1, 0,
                c->has_bias, false, logits) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_selected, 0, ref_selected_host, selected_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_weights, 0, ref_weights_host, weights_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_probs, 0, ref_probs_host, probs_bytes) != 0);

            if (ci == 0) {
                memcpy(unique_selected_host,
                       ref_selected_host,
                       sizeof(unique_selected_host));
            } else if (c->has_bias) {
                TEST_ASSERT(memcmp(unique_selected_host,
                                   ref_selected_host,
                                   sizeof(unique_selected_host)) != 0);
            }

            TEST_ASSERT(unsetenv(disable_env) == 0);
            for (uint32_t mode = 0; mode < modes; mode++) {
                const bool fused_weights = mode != 0;
                const bool fused_transform = mode >= 2;
                if (fused_weights) {
                    TEST_ASSERT(unsetenv(weights_disable_env) == 0);
                } else {
                    TEST_ASSERT(setenv(weights_disable_env, "1", 1) == 0);
                }
                if (fused_transform) {
                    TEST_ASSERT(unsetenv(transform_finalize_disable_env) == 0);
                } else {
                    TEST_ASSERT(setenv(
                        transform_finalize_disable_env, "1", 1) == 0);
                }
                for (uint32_t i = 0; i < n_expert; i++) {
                    const uint32_t poison_bits = 0x7fc00001u + i;
                    memcpy(&simd_probs_host[i],
                           &poison_bits,
                           sizeof(poison_bits));
                }
                TEST_ASSERT(ds4_gpu_tensor_write(
                    simd_probs, 0, simd_probs_host, probs_bytes) != 0);
                TEST_ASSERT(ds4_gpu_router_select_tensor(
                    simd_selected, simd_weights, simd_probs,
                    model_raw, page, 0, 0, 1, 0,
                    n_expert, n_used, 1.5f, 1, 0,
                    c->has_bias, false, logits) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    simd_selected, 0, simd_selected_host,
                    selected_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    simd_weights, 0, simd_weights_host,
                    weights_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    simd_probs, 0, simd_probs_host, probs_bytes) != 0);

                size_t selected_mismatch = 0;
                for (uint32_t i = 0; i < n_used; i++) {
                    if (ref_selected_host[i] != simd_selected_host[i]) {
                        selected_mismatch++;
                    }
                }
                const test_float_compare_stats weights_stats =
                    test_compare_float_bits(
                        ref_weights_host, simd_weights_host, n_used);
                const test_float_compare_stats probs_stats =
                    test_compare_float_bits(
                        ref_probs_host, simd_probs_host, n_expert);
                fprintf(stderr,
                        "ds4-test: router SIMD finalize exactness "
                        "case=%s mode=%s rep=%u selected=%zu/%u weights=%zu/%u "
                        "probs=%zu/%u max_weight_ulp=%u max_prob_ulp=%u\n",
                        c->name,
                        fused_transform ? "fused-transform" :
                        fused_weights ? "fused-weights" : "split-weights",
                        fused_transform ? mode - 2u : 0u,
                        selected_mismatch,
                        n_used,
                        weights_stats.mismatch_count,
                        n_used,
                        probs_stats.mismatch_count,
                        n_expert,
                        weights_stats.max_ulp,
                        probs_stats.max_ulp);
                TEST_ASSERT(selected_mismatch == 0);
                TEST_ASSERT(weights_stats.mismatch_count == 0);
                TEST_ASSERT(probs_stats.mismatch_count == 0);
                total_selected_mismatch += selected_mismatch;
                total_weights_mismatch += weights_stats.mismatch_count;
                total_probs_mismatch += probs_stats.mismatch_count;
            }
        }
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(weights_disable_env, saved_weights_disable);
    test_restore_env(transform_finalize_disable_env,
                     saved_transform_finalize_disable);
    test_restore_env(select_disable_env, saved_select_disable);
    fprintf(stderr,
            "ds4-test: router SIMD finalize total selected=%zu/%zu "
            "weights=%zu/%zu probs=%zu/%zu\n",
            total_selected_mismatch,
            (sizeof(cases) / sizeof(cases[0])) * modes * (size_t)n_used,
            total_weights_mismatch,
            (sizeof(cases) / sizeof(cases[0])) * modes * (size_t)n_used,
            total_probs_mismatch,
            (sizeof(cases) / sizeof(cases[0])) * modes * (size_t)n_expert);
    TEST_ASSERT(total_selected_mismatch == 0);
    TEST_ASSERT(total_weights_mismatch == 0);
    TEST_ASSERT(total_probs_mismatch == 0);

    free(simd_probs_host);
    free(ref_probs_host);
    free(logits_host);
    ds4_gpu_tensor_free(simd_probs);
    ds4_gpu_tensor_free(ref_probs);
    ds4_gpu_tensor_free(simd_weights);
    ds4_gpu_tensor_free(ref_weights);
    ds4_gpu_tensor_free(simd_selected);
    ds4_gpu_tensor_free(ref_selected);
    ds4_gpu_tensor_free(logits);
    free(model_raw);
}

static void test_metal_router_weights_batch_exact(void) {
    typedef struct {
        const char *name;
        uint32_t n_tokens;
        bool has_bias;
        bool hash_mode;
        uint32_t pattern;
    } router_batch_case;
    static const router_batch_case cases[] = {
        { "rows2-unique", 2, false, false, 0 },
        { "rows17-bias-ties", 17, true, false, 1 },
        { "rows129-hash-duplicates", 129, false, true, 2 },
        { "rows2048-typical", 2048, false, false, 0 },
        { "rows3-clamp-underflow", 3, false, false, 3 },
    };
    const uint32_t n_expert = 256;
    const uint32_t n_used = 6;
    const uint32_t max_tokens = 2048;
    const uint32_t hash_rows = 64;
    const uint32_t repeats = 2;
    const uint64_t logits_bytes =
        (uint64_t)max_tokens * n_expert * sizeof(float);
    const uint64_t selected_bytes =
        (uint64_t)max_tokens * n_used * sizeof(int32_t);
    const uint64_t weights_bytes =
        (uint64_t)max_tokens * n_used * sizeof(float);
    const uint64_t tokens_bytes = (uint64_t)max_tokens * sizeof(int32_t);
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t bias_offset = 0;
    const uint64_t hash_offset = 2048;
    const char *disable_env =
        "DS4_METAL_DISABLE_ROUTER_WEIGHTS_BATCH_FUSION";
    const char *select_disable_env =
        "DS4_METAL_DISABLE_ROUTER_SELECT_FUSION";

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(logits_bytes);
    ds4_gpu_tensor *tokens = ds4_gpu_tensor_alloc(tokens_bytes);
    ds4_gpu_tensor *ref_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *batch_selected = ds4_gpu_tensor_alloc(selected_bytes);
    ds4_gpu_tensor *ref_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *batch_weights = ds4_gpu_tensor_alloc(weights_bytes);
    ds4_gpu_tensor *ref_probs = ds4_gpu_tensor_alloc(logits_bytes);
    ds4_gpu_tensor *batch_probs = ds4_gpu_tensor_alloc(logits_bytes);
    float *logits_host = malloc((size_t)logits_bytes);
    int32_t *tokens_host = malloc((size_t)tokens_bytes);
    int32_t *ref_selected_host = malloc((size_t)selected_bytes);
    int32_t *batch_selected_host = malloc((size_t)selected_bytes);
    float *ref_weights_host = malloc((size_t)weights_bytes);
    float *batch_weights_host = malloc((size_t)weights_bytes);
    float *ref_probs_host = malloc((size_t)logits_bytes);
    float *batch_probs_host = malloc((size_t)logits_bytes);
    TEST_ASSERT(model_raw != NULL);
    TEST_ASSERT(logits != NULL);
    TEST_ASSERT(tokens != NULL);
    TEST_ASSERT(ref_selected != NULL);
    TEST_ASSERT(batch_selected != NULL);
    TEST_ASSERT(ref_weights != NULL);
    TEST_ASSERT(batch_weights != NULL);
    TEST_ASSERT(ref_probs != NULL);
    TEST_ASSERT(batch_probs != NULL);
    TEST_ASSERT(logits_host != NULL);
    TEST_ASSERT(tokens_host != NULL);
    TEST_ASSERT(ref_selected_host != NULL);
    TEST_ASSERT(batch_selected_host != NULL);
    TEST_ASSERT(ref_weights_host != NULL);
    TEST_ASSERT(batch_weights_host != NULL);
    TEST_ASSERT(ref_probs_host != NULL);
    TEST_ASSERT(batch_probs_host != NULL);

    char *saved_disable = test_save_env(disable_env);
    char *saved_select_disable = test_save_env(select_disable_env);
    size_t total_selected_mismatch = 0;
    size_t total_weights_mismatch = 0;
    size_t total_probs_mismatch = 0;
    const bool allocated = model_raw && logits && tokens && ref_selected &&
        batch_selected && ref_weights && batch_weights && ref_probs &&
        batch_probs && logits_host && tokens_host && ref_selected_host &&
        batch_selected_host && ref_weights_host && batch_weights_host &&
        ref_probs_host && batch_probs_host;
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *bias = (float *)((uint8_t *)model_raw + bias_offset);
        int32_t *hash = (int32_t *)((uint8_t *)model_raw + hash_offset);
        for (uint32_t i = 0; i < n_expert; i++) {
            bias[i] = (float)((int)((i * 29u) % 67u) - 33) / 16.0f;
        }
        for (uint32_t row = 0; row < hash_rows; row++) {
            const int32_t base = (int32_t)((row * 37u) % n_expert);
            hash[(uint64_t)row * n_used + 0u] = base;
            hash[(uint64_t)row * n_used + 1u] = base;
            hash[(uint64_t)row * n_used + 2u] = (base + 19) % (int32_t)n_expert;
            hash[(uint64_t)row * n_used + 3u] = (base + 43) % (int32_t)n_expert;
            hash[(uint64_t)row * n_used + 4u] = (base + 43) % (int32_t)n_expert;
            hash[(uint64_t)row * n_used + 5u] = (base + 101) % (int32_t)n_expert;
        }
        for (uint32_t row = 0; row < max_tokens; row++) {
            tokens_host[row] = (int32_t)((row * 13u + 7u) % hash_rows);
        }

        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);
        TEST_ASSERT(ds4_gpu_tensor_write(
            tokens, 0, tokens_host, tokens_bytes) != 0);
        TEST_ASSERT(unsetenv(select_disable_env) == 0);

        for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
            const router_batch_case *c = &cases[ci];
            const size_t prob_count = (size_t)c->n_tokens * n_expert;
            const size_t route_count = (size_t)c->n_tokens * n_used;
            const uint64_t case_probs_bytes = prob_count * sizeof(float);
            const uint64_t case_selected_bytes = route_count * sizeof(int32_t);
            const uint64_t case_weights_bytes = route_count * sizeof(float);
            for (uint32_t row = 0; row < c->n_tokens; row++) {
                for (uint32_t expert = 0; expert < n_expert; expert++) {
                    const size_t index = (size_t)row * n_expert + expert;
                    const int value =
                        (int)((row * 131u + expert * 47u +
                               (expert ^ (expert >> 3u)) * 13u) % 513u) -
                        256;
                    logits_host[index] = (float)value / 64.0f;
                    if (c->pattern == 1) {
                        logits_host[index] =
                            -7.0f - (float)((row + expert) % 17u) / 64.0f;
                    } else if (c->pattern == 3) {
                        logits_host[index] =
                            -30.0f - (float)((row * 17u + expert * 29u) % 11u);
                    }
                }
                if (c->pattern == 1) {
                    static const uint32_t tied[] = {
                        7u, 19u, 43u, 71u, 103u, 149u, 211u, 239u,
                    };
                    for (size_t i = 0; i < sizeof(tied) / sizeof(tied[0]); i++) {
                        logits_host[(size_t)row * n_expert + tied[i]] = 1.0f;
                    }
                }
            }
            TEST_ASSERT(ds4_gpu_tensor_write(
                logits, 0, logits_host, case_probs_bytes) != 0);

            TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_router_select_batch_tensor(
                ref_selected, ref_weights, ref_probs,
                model_raw, page, bias_offset, hash_offset, hash_rows,
                1, 0, c->has_bias, c->hash_mode, logits, tokens,
                n_expert, n_used, 1.5f, c->n_tokens) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_selected, 0, ref_selected_host, case_selected_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_weights, 0, ref_weights_host, case_weights_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                ref_probs, 0, ref_probs_host, case_probs_bytes) != 0);

            if (c->hash_mode) {
                for (uint32_t row = 0; row < c->n_tokens; row++) {
                    const uint32_t hash_row = (uint32_t)tokens_host[row];
                    for (uint32_t lane = 0; lane < n_used; lane++) {
                        TEST_ASSERT(
                            ref_selected_host[(size_t)row * n_used + lane] ==
                            hash[(uint64_t)hash_row * n_used + lane]);
                    }
                }
            }
            if (c->pattern == 3) {
                for (uint32_t row = 0; row < c->n_tokens; row++) {
                    float sum = 0.0f;
                    for (uint32_t lane = 0; lane < n_used; lane++) {
                        const int32_t expert =
                            ref_selected_host[(size_t)row * n_used + lane];
                        sum += ref_probs_host[(size_t)row * n_expert +
                                              (uint32_t)expert];
                    }
                    TEST_ASSERT(sum < 6.103515625e-5f);
                }
            }

            TEST_ASSERT(unsetenv(disable_env) == 0);
            for (uint32_t rep = 0; rep < repeats; rep++) {
                TEST_ASSERT(ds4_gpu_router_select_batch_tensor(
                    batch_selected, batch_weights, batch_probs,
                    model_raw, page, bias_offset, hash_offset, hash_rows,
                    1, 0, c->has_bias, c->hash_mode, logits, tokens,
                    n_expert, n_used, 1.5f, c->n_tokens) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    batch_selected, 0, batch_selected_host,
                    case_selected_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    batch_weights, 0, batch_weights_host,
                    case_weights_bytes) != 0);
                TEST_ASSERT(ds4_gpu_tensor_read(
                    batch_probs, 0, batch_probs_host,
                    case_probs_bytes) != 0);

                size_t selected_mismatch = 0;
                for (size_t i = 0; i < route_count; i++) {
                    if (ref_selected_host[i] != batch_selected_host[i]) {
                        selected_mismatch++;
                    }
                }
                const test_float_compare_stats weights_stats =
                    test_compare_float_bits(
                        ref_weights_host, batch_weights_host, route_count);
                const test_float_compare_stats probs_stats =
                    test_compare_float_bits(
                        ref_probs_host, batch_probs_host, prob_count);
                fprintf(stderr,
                        "ds4-test: router batch weights exactness "
                        "case=%s rep=%u selected=%zu/%zu weights=%zu/%zu "
                        "probs=%zu/%zu max_weight_ulp=%u max_prob_ulp=%u\n",
                        c->name,
                        rep,
                        selected_mismatch,
                        route_count,
                        weights_stats.mismatch_count,
                        route_count,
                        probs_stats.mismatch_count,
                        prob_count,
                        weights_stats.max_ulp,
                        probs_stats.max_ulp);
                TEST_ASSERT(selected_mismatch == 0);
                TEST_ASSERT(weights_stats.mismatch_count == 0);
                TEST_ASSERT(probs_stats.mismatch_count == 0);
                total_selected_mismatch += selected_mismatch;
                total_weights_mismatch += weights_stats.mismatch_count;
                total_probs_mismatch += probs_stats.mismatch_count;
            }
        }
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(select_disable_env, saved_select_disable);
    fprintf(stderr,
            "ds4-test: router batch weights total selected=%zu "
            "weights=%zu probs=%zu\n",
            total_selected_mismatch,
            total_weights_mismatch,
            total_probs_mismatch);
    TEST_ASSERT(total_selected_mismatch == 0);
    TEST_ASSERT(total_weights_mismatch == 0);
    TEST_ASSERT(total_probs_mismatch == 0);

    free(batch_probs_host);
    free(ref_probs_host);
    free(batch_weights_host);
    free(ref_weights_host);
    free(batch_selected_host);
    free(ref_selected_host);
    free(tokens_host);
    free(logits_host);
    ds4_gpu_tensor_free(batch_probs);
    ds4_gpu_tensor_free(ref_probs);
    ds4_gpu_tensor_free(batch_weights);
    ds4_gpu_tensor_free(ref_weights);
    ds4_gpu_tensor_free(batch_selected);
    ds4_gpu_tensor_free(ref_selected);
    ds4_gpu_tensor_free(tokens);
    ds4_gpu_tensor_free(logits);
    free(model_raw);
}
