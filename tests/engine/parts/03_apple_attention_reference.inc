static void test_metal_gathered_kv_stage_exact(void) {
    const uint32_t head_dim = 512;
    const uint32_t raw_cap = 7;
    const uint32_t n_raw = 5;
    const uint32_t n_comp = 3;
    const uint32_t raw_starts[] = {0, 2, 5, 6};
    const uint64_t raw_bytes =
        (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)n_comp * head_dim * sizeof(uint16_t);
    const uint64_t payload_bytes =
        ((uint64_t)n_raw + n_comp) * head_dim * sizeof(uint16_t);
    const uint64_t raw_out_bytes =
        (uint64_t)n_raw * head_dim * sizeof(uint16_t);
    const uint64_t raw_view_offset = 4;
    const uint64_t comp_view_offset = 2;
    const uint64_t dst_view_offset = 6;
    const uint64_t raw_base_bytes = raw_view_offset + raw_bytes + 12;
    const uint64_t comp_base_bytes = comp_view_offset + comp_bytes + 14;
    const uint64_t dst_base_bytes = dst_view_offset + payload_bytes + 10;
    const char *envs[] = {
        "DS4_METAL_REQUIRE_GATHERED_KV_STAGE",
        "DS4_METAL_DISABLE_CONTIG_F32_F16_COPY",
        "DS4_METAL_DISABLE_CONTIG_F16_F16_COPY",
    };
    char *saved[sizeof(envs)/sizeof(envs[0])];
    for (size_t i = 0; i < sizeof(envs)/sizeof(envs[0]); i++) {
        saved[i] = test_save_env(envs[i]);
    }

    ds4_gpu_tensor *raw_base = ds4_gpu_tensor_alloc(raw_base_bytes);
    ds4_gpu_tensor *raw = raw_base
        ? ds4_gpu_tensor_view(raw_base, raw_view_offset, raw_bytes)
        : NULL;
    ds4_gpu_tensor *comp_base = ds4_gpu_tensor_alloc(comp_base_bytes);
    ds4_gpu_tensor *comp = comp_base
        ? ds4_gpu_tensor_view(comp_base, comp_view_offset, comp_bytes)
        : NULL;
    ds4_gpu_tensor *ref_base = ds4_gpu_tensor_alloc(dst_base_bytes);
    ds4_gpu_tensor *ref = ref_base
        ? ds4_gpu_tensor_view(ref_base, dst_view_offset, payload_bytes)
        : NULL;
    ds4_gpu_tensor *fused_base = ds4_gpu_tensor_alloc(dst_base_bytes);
    ds4_gpu_tensor *fused = fused_base
        ? ds4_gpu_tensor_view(fused_base, dst_view_offset, payload_bytes)
        : NULL;
    TEST_ASSERT(raw_base != NULL);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp_base != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(ref_base != NULL);
    TEST_ASSERT(ref != NULL);
    TEST_ASSERT(fused_base != NULL);
    TEST_ASSERT(fused != NULL);

    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    uint8_t *dst_init = malloc((size_t)dst_base_bytes);
    uint8_t *ref_host = malloc((size_t)dst_base_bytes);
    uint8_t *fused_host = malloc((size_t)dst_base_bytes);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(dst_init != NULL);
    TEST_ASSERT(ref_host != NULL);
    TEST_ASSERT(fused_host != NULL);

    static const uint16_t half_patterns[] = {
        0x0000u, 0x8000u, 0x0001u, 0x03ffu, 0x0400u,
        0x3555u, 0x3c00u, 0x3c01u, 0x7bffu, 0xfbffu,
        0x7c00u, 0xfc00u, 0x7e00u, 0x7e01u, 0xfe55u,
    };
    const bool allocated = raw_base && raw && comp_base && comp &&
        ref_base && ref && fused_base && fused && raw_host && comp_host &&
        dst_init && ref_host && fused_host;
    size_t raw_mismatch = 0;
    size_t comp_mismatch = 0;
    size_t guard_mismatch = 0;
    if (allocated) {
        for (uint32_t row = 0; row < raw_cap; row++) {
            test_fill_copy_f32_patterns(
                raw_host + (uint64_t)row * head_dim,
                head_dim,
                row * 17u + 3u);
        }
        for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) {
            comp_host[i] = half_patterns[(i * 7u + (i >> 3u)) %
                (sizeof(half_patterns)/sizeof(half_patterns[0]))];
        }
        for (uint64_t i = 0; i < dst_base_bytes; i++) {
            dst_init[i] = (uint8_t)(0xa5u ^ (uint8_t)(i * 37u));
        }
        TEST_ASSERT(ds4_gpu_tensor_write(
                        raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        comp, 0, comp_host, comp_bytes) != 0);
        for (size_t ci = 0;
             ci < sizeof(raw_starts)/sizeof(raw_starts[0]);
             ci++) {
            TEST_ASSERT(ds4_gpu_tensor_write(
                            ref_base, 0, dst_init, dst_base_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_write(
                            fused_base, 0, dst_init, dst_base_bytes) != 0);

            TEST_ASSERT(unsetenv(envs[0]) == 0);
            TEST_ASSERT(unsetenv(envs[1]) == 0);
            TEST_ASSERT(unsetenv(envs[2]) == 0);
            ds4_gpu_set_quality(true);
            TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                            ref, raw, raw_cap, raw_starts[ci], n_raw,
                            comp, 1, n_comp, head_dim) != 0);

            ds4_gpu_set_quality(false);
            TEST_ASSERT(setenv(envs[0], "1", 1) == 0);
            TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                            fused, raw, raw_cap, raw_starts[ci], n_raw,
                            comp, 1, n_comp, head_dim) != 0);

            TEST_ASSERT(ds4_gpu_tensor_read(
                            ref_base, 0, ref_host, dst_base_bytes) != 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                            fused_base, 0, fused_host,
                            dst_base_bytes) != 0);
            for (uint64_t i = 0; i < dst_base_bytes; i++) {
                if (i < dst_view_offset ||
                    i >= dst_view_offset + payload_bytes) {
                    if (ref_host[i] != dst_init[i]) guard_mismatch++;
                    if (fused_host[i] != dst_init[i]) guard_mismatch++;
                } else if (ref_host[i] == fused_host[i]) {
                    continue;
                } else if (i < dst_view_offset + raw_out_bytes) {
                    raw_mismatch++;
                } else {
                    comp_mismatch++;
                }
            }
        }

        /* Component-copy diagnostics and quality mode prevent strict
         * selection of the gathered kernel. */
        TEST_ASSERT(setenv(envs[0], "1", 1) == 0);
        TEST_ASSERT(setenv(envs[1], "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                        fused, raw, raw_cap, 5, n_raw,
                        comp, 1, n_comp, head_dim) == 0);
        TEST_ASSERT(unsetenv(envs[1]) == 0);
        TEST_ASSERT(setenv(envs[2], "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                        fused, raw, raw_cap, 5, n_raw,
                        comp, 1, n_comp, head_dim) == 0);
        TEST_ASSERT(unsetenv(envs[2]) == 0);
        ds4_gpu_set_quality(true);
        TEST_ASSERT(ds4_gpu_flash_kv_stage_f16_tensor(
                        fused, raw, raw_cap, 5, n_raw,
                        comp, 1, n_comp, head_dim) == 0);
        ds4_gpu_set_quality(false);
    }

    for (size_t i = 0; i < sizeof(envs)/sizeof(envs[0]); i++) {
        test_restore_env(envs[i], saved[i]);
    }
    fprintf(stderr,
            "ds4-test: gathered KV staging exact cases=%zu "
            "raw_bytes=%zu comp_bytes=%zu guard_bytes=%zu\n",
            sizeof(raw_starts)/sizeof(raw_starts[0]),
            raw_mismatch, comp_mismatch, guard_mismatch);
    TEST_ASSERT(raw_mismatch == 0);
    TEST_ASSERT(comp_mismatch == 0);
    TEST_ASSERT(guard_mismatch == 0);

    free(fused_host);
    free(ref_host);
    free(dst_init);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(fused);
    ds4_gpu_tensor_free(fused_base);
    ds4_gpu_tensor_free(ref);
    ds4_gpu_tensor_free(ref_base);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(comp_base);
    ds4_gpu_tensor_free(raw);
    ds4_gpu_tensor_free(raw_base);
}

static void test_metal_contiguous_compressed_f16_attention_exact(void) {
    const uint32_t head_dim = 512;
    const uint32_t n_head = 2;
    const uint32_t raw_cap = 7;
    const uint32_t n_raw = 5;
    const uint32_t raw_start = 5;
    const uint32_t n_comp = 3;
    const uint64_t raw_bytes =
        (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)n_comp * head_dim * sizeof(uint16_t);
    const uint64_t q_bytes =
        (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();
    const char *env_name = "DS4_METAL_DISABLE_CONTIG_F16_F16_COPY";
    static const uint16_t half_patterns[] = {
        0x0000u, 0x8000u, 0x0001u, 0x03ffu, 0x0400u,
        0x1001u, 0x3555u, 0x3c00u, 0x3c01u, 0x4000u,
        0xbc00u, 0xc000u, 0x7bffu, 0xfbffu,
    };

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp_base = ds4_gpu_tensor_alloc(comp_bytes + 18u);
    ds4_gpu_tensor *comp = comp_base
        ? ds4_gpu_tensor_view(comp_base, 2u, comp_bytes)
        : NULL;
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *heads_blit = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *heads_compute = ds4_gpu_tensor_alloc(q_bytes);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp_base != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(heads_blit != NULL);
    TEST_ASSERT(heads_compute != NULL);

    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    float *q_host = malloc((size_t)q_bytes);
    float *blit_host = malloc((size_t)q_bytes);
    float *compute_host = malloc((size_t)q_bytes);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(blit_host != NULL);
    TEST_ASSERT(compute_host != NULL);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    const bool allocated = raw && comp_base && comp && q && heads_blit &&
        heads_compute && raw_host && comp_host && q_host && blit_host &&
        compute_host && model_raw;
    char *saved_env = test_save_env(env_name);
    test_float_compare_stats stats = {0};
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *sinks = model_raw;
        sinks[0] = -0.375f;
        sinks[1] = 0.1875f;
        for (uint64_t i = 0; i < (uint64_t)raw_cap * head_dim; i++) {
            const int value =
                (int)((i * 19u + (i ^ (i >> 5u)) * 7u) % 193u) - 96;
            raw_host[i] = (float)value / 128.0f;
        }
        for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) {
            comp_host[i] = half_patterns[(i * 5u + (i >> 4u)) %
                (sizeof(half_patterns) / sizeof(half_patterns[0]))];
        }
        for (uint32_t i = 0; i < n_head * head_dim; i++) {
            const int value = (int)((i * 37u + (i ^ (i >> 3u)) * 11u) % 251u) - 125;
            q_host[i] = (float)value / 96.0f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(comp, 0, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(setenv(env_name, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            heads_blit,
            model_raw,
            page,
            0,
            q,
            raw,
            n_raw,
            raw_cap,
            raw_start,
            comp,
            1,
            n_comp,
            NULL,
            0,
            n_head,
            head_dim) != 0);

        TEST_ASSERT(setenv(env_name, "0", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            heads_compute,
            model_raw,
            page,
            0,
            q,
            raw,
            n_raw,
            raw_cap,
            raw_start,
            comp,
            1,
            n_comp,
            NULL,
            0,
            n_head,
            head_dim) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
            heads_blit, 0, blit_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
            heads_compute, 0, compute_host, q_bytes) != 0);
        stats = test_compare_float_bits(
            blit_host, compute_host, (size_t)n_head * head_dim);
    }
    test_restore_env(env_name, saved_env);
    fprintf(stderr,
            "ds4-test: contiguous compressed-F16 staging exactness "
            "mismatches=%zu/%u max_ulp=%u max_abs=%g\n",
            stats.mismatch_count,
            n_head * head_dim,
            stats.max_ulp,
            stats.max_abs);
    TEST_ASSERT(stats.mismatch_count == 0);

    free(model_raw);
    free(compute_host);
    free(blit_host);
    free(q_host);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(heads_compute);
    ds4_gpu_tensor_free(heads_blit);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(comp_base);
    ds4_gpu_tensor_free(raw);
}

static void test_metal_persistent_zero_attention_mask_exact_case(
        uint32_t raw_cap,
        uint32_t n_raw,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t seed) {
    const uint32_t head_dim = 512;
    const uint32_t n_head = 2;
    const uint64_t raw_bytes =
        (uint64_t)raw_cap * head_dim * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)n_comp * head_dim * sizeof(uint16_t);
    const uint64_t q_bytes =
        (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t mask_bytes = (uint64_t)n_comp * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *comp_mask = ds4_gpu_tensor_alloc(mask_bytes);
    ds4_gpu_tensor *legacy = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *persistent = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *masked = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *pad_legacy = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *after_mask = ds4_gpu_tensor_alloc(q_bytes);
    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(comp_mask != NULL);
    TEST_ASSERT(legacy != NULL);
    TEST_ASSERT(persistent != NULL);
    TEST_ASSERT(masked != NULL);
    TEST_ASSERT(pad_legacy != NULL);
    TEST_ASSERT(after_mask != NULL);

    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    float *q_host = malloc((size_t)q_bytes);
    float *mask_host = malloc((size_t)mask_bytes);
    float *legacy_host = malloc((size_t)q_bytes);
    float *persistent_host = malloc((size_t)q_bytes);
    float *masked_host = malloc((size_t)q_bytes);
    float *pad_legacy_host = malloc((size_t)q_bytes);
    float *after_mask_host = malloc((size_t)q_bytes);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(mask_host != NULL);
    TEST_ASSERT(legacy_host != NULL);
    TEST_ASSERT(persistent_host != NULL);
    TEST_ASSERT(masked_host != NULL);
    TEST_ASSERT(pad_legacy_host != NULL);
    TEST_ASSERT(after_mask_host != NULL);

    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);
    const bool allocated = raw && comp && q && comp_mask && legacy &&
        persistent && masked && pad_legacy && after_mask && raw_host &&
        comp_host && q_host && mask_host && legacy_host && persistent_host &&
        masked_host && pad_legacy_host && after_mask_host &&
        model_raw;
    const char *disable_env =
        "DS4_METAL_DISABLE_PERSISTENT_ZERO_ATTN_MASK";
    const char *pad_disable_env =
        "DS4_METAL_DISABLE_GATHERED_KV_PAD_FUSION";
    const char *shared_pad_disable_env =
        "DS4_METAL_DISABLE_SHARED_KV_PAD";
    char *saved_disable = test_save_env(disable_env);
    char *saved_pad_disable = test_save_env(pad_disable_env);
    char *saved_shared_pad_disable = test_save_env(shared_pad_disable_env);
    test_float_compare_stats persistent_stats = {0};
    test_float_compare_stats masked_stats = {0};
    test_float_compare_stats pad_stats = {0};
    test_float_compare_stats after_mask_stats = {0};

    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        float *sinks = model_raw;
        sinks[0] = -0.3125f;
        sinks[1] = 0.21875f;
        for (uint64_t i = 0; i < (uint64_t)raw_cap * head_dim; i++) {
            const int value = (int)((i * 17u + (i ^ (i >> 5u)) * 11u +
                                     seed * 13u) % 211u) - 105;
            raw_host[i] = (float)value / 128.0f;
        }
        for (uint64_t i = 0; i < (uint64_t)n_comp * head_dim; i++) {
            const int value = (int)((i * 23u + (i ^ (i >> 4u)) * 7u +
                                     seed * 19u) % 193u) - 96;
            comp_host[i] = test_float_to_f16((float)value / 112.0f);
        }
        for (uint32_t i = 0; i < n_head * head_dim; i++) {
            const int value = (int)((i * 31u + (i ^ (i >> 3u)) * 5u +
                                     seed * 29u) % 227u) - 113;
            q_host[i] = (float)value / 104.0f;
        }
        for (uint32_t i = 0; i < n_comp; i++) {
            mask_host[i] = i == 0 ? -8.0f : -(float)(i + 1u) / 8.0f;
        }

        TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(comp, 0, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        comp_mask, 0, mask_host, mask_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            legacy, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            NULL, 0, n_head, head_dim) != 0);

        unsetenv(disable_env);
        unsetenv(pad_disable_env);
        unsetenv(shared_pad_disable_env);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            persistent, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            NULL, 0, n_head, head_dim) != 0);

        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            masked, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            comp_mask, 1, n_head, head_dim) != 0);

        TEST_ASSERT(setenv(pad_disable_env, "1", 1) == 0);
        TEST_ASSERT(setenv(shared_pad_disable_env, "1", 1) == 0);
        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            pad_legacy, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            comp_mask, 1, n_head, head_dim) != 0);
        unsetenv(pad_disable_env);
        unsetenv(shared_pad_disable_env);

        TEST_ASSERT(ds4_gpu_attention_decode_heads_tensor(
            after_mask, model_raw, page, 0, q, raw,
            n_raw, raw_cap, raw_start, comp, 1, n_comp,
            NULL, 0, n_head, head_dim) != 0);

        TEST_ASSERT(ds4_gpu_tensor_read(
                        legacy, 0, legacy_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        persistent, 0, persistent_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        masked, 0, masked_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        pad_legacy, 0, pad_legacy_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_read(
                        after_mask, 0, after_mask_host, q_bytes) != 0);
        persistent_stats = test_compare_float_bits(
            legacy_host, persistent_host, (size_t)n_head * head_dim);
        masked_stats = test_compare_float_bits(
            legacy_host, masked_host, (size_t)n_head * head_dim);
        pad_stats = test_compare_float_bits(
            pad_legacy_host, masked_host, (size_t)n_head * head_dim);
        after_mask_stats = test_compare_float_bits(
            legacy_host, after_mask_host, (size_t)n_head * head_dim);
    }

    test_restore_env(disable_env, saved_disable);
    test_restore_env(pad_disable_env, saved_pad_disable);
    test_restore_env(shared_pad_disable_env, saved_shared_pad_disable);
    fprintf(stderr,
            "ds4-test: persistent zero attention mask exact keys=%u "
            "candidate=%zu/%u max_ulp=%u masked_diff=%zu/%u "
            "pad_fusion=%zu/%u max_ulp=%u "
            "after_mask=%zu/%u max_ulp=%u\n",
            n_raw + n_comp,
            persistent_stats.mismatch_count, n_head * head_dim,
            persistent_stats.max_ulp,
            masked_stats.mismatch_count, n_head * head_dim,
            pad_stats.mismatch_count, n_head * head_dim,
            pad_stats.max_ulp,
            after_mask_stats.mismatch_count, n_head * head_dim,
            after_mask_stats.max_ulp);
    TEST_ASSERT(persistent_stats.mismatch_count == 0);
    TEST_ASSERT(masked_stats.mismatch_count != 0);
    TEST_ASSERT(pad_stats.mismatch_count == 0);
    TEST_ASSERT(after_mask_stats.mismatch_count == 0);

    free(model_raw);
    free(after_mask_host);
    free(pad_legacy_host);
    free(masked_host);
    free(persistent_host);
    free(legacy_host);
    free(mask_host);
    free(q_host);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(after_mask);
    ds4_gpu_tensor_free(pad_legacy);
    ds4_gpu_tensor_free(masked);
    ds4_gpu_tensor_free(persistent);
    ds4_gpu_tensor_free(legacy);
    ds4_gpu_tensor_free(comp_mask);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
}

static void test_metal_persistent_zero_attention_mask_exact(void) {
    test_metal_persistent_zero_attention_mask_exact_case(7, 5, 5, 3, 11);
    test_metal_persistent_zero_attention_mask_exact_case(37, 29, 35, 3, 23);
}

typedef enum {
    TEST_METAL_PREFILL_MASK_CACHE_RAW = 1,
    TEST_METAL_PREFILL_MASK_CACHE_RATIO4 = 2,
    TEST_METAL_PREFILL_MASK_CACHE_RATIO128 = 3,
} test_metal_prefill_mask_cache_kind;

typedef struct {
    uint32_t n_tokens;
    uint32_t n_comp;
    uint32_t window;
    uint32_t ratio;
} test_metal_prefill_mask_cache_shape;

static int test_metal_zero_prefix_prefill_mask_cache_call(
        test_metal_prefill_mask_cache_kind  kind,
        ds4_gpu_tensor                    *heads,
        const void                        *model_map,
        uint64_t                           model_size,
        const ds4_gpu_tensor              *q,
        const ds4_gpu_tensor              *raw,
        const ds4_gpu_tensor              *comp,
        const ds4_gpu_tensor              *comp_mask,
        const test_metal_prefill_mask_cache_shape *shape,
        bool                               masked,
        uint32_t                           n_head,
        uint32_t                           head_dim) {
    if (kind == TEST_METAL_PREFILL_MASK_CACHE_RAW) {
        if (masked) return 0;
        return ds4_gpu_attention_prefill_raw_heads_tensor(
            heads, model_map, model_size, 0, q, raw,
            shape->n_tokens, shape->window, n_head, head_dim);
    }

    if (masked) {
        return ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
            heads, model_map, model_size, 0, q, raw, comp, 1, comp_mask,
            shape->n_tokens, shape->n_comp, shape->window, shape->ratio,
            n_head, head_dim);
    }

    return ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        heads, model_map, model_size, 0, q, raw, comp, 1,
        shape->n_tokens, shape->n_comp, shape->window, shape->ratio,
        n_head, head_dim);
}

static bool test_metal_zero_prefix_prefill_mask_cache_run(
        test_metal_prefill_mask_cache_kind  kind,
        ds4_gpu_tensor                    *heads,
        const void                        *model_map,
        uint64_t                           model_size,
        const ds4_gpu_tensor              *q,
        const ds4_gpu_tensor              *raw,
        const ds4_gpu_tensor              *comp,
        const ds4_gpu_tensor              *comp_mask,
        const test_metal_prefill_mask_cache_shape *shape,
        bool                               masked,
        uint32_t                           n_head,
        uint32_t                           head_dim,
        float                             *host) {
    const uint64_t bytes =
        (uint64_t)shape->n_tokens * n_head * head_dim * sizeof(float);
    const int call_ok = test_metal_zero_prefix_prefill_mask_cache_call(
        kind, heads, model_map, model_size, q, raw, comp, comp_mask,
        shape, masked, n_head, head_dim);
    TEST_ASSERT(call_ok != 0);
    if (!call_ok) return false;

    const int read_ok = ds4_gpu_tensor_read(heads, 0, host, bytes);
    TEST_ASSERT(read_ok != 0);
    return read_ok != 0;
}

static void test_metal_zero_prefix_prefill_mask_cache_compare(
        const float *expected,
        const float *actual,
        size_t       count,
        size_t      *total_mismatches,
        uint32_t    *max_ulp) {
    const test_float_compare_stats stats =
        test_compare_float_bits(expected, actual, count);
    *total_mismatches += stats.mismatch_count;
    if (stats.max_ulp > *max_ulp) *max_ulp = stats.max_ulp;
}

static void test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        test_metal_prefill_mask_cache_kind kind,
        uint32_t                           seed) {
    const uint32_t head_dim = 512;
    const uint32_t n_head = 1;
    const uint32_t max_tokens = 129;
    const uint32_t max_comp = 32;
    const uint64_t raw_count = (uint64_t)max_tokens * head_dim;
    const uint64_t comp_count = (uint64_t)max_comp * head_dim;
    const uint64_t q_count = (uint64_t)max_tokens * n_head * head_dim;
    const uint64_t mask_count = (uint64_t)max_tokens * max_comp;
    const uint64_t raw_bytes = raw_count * sizeof(float);
    const uint64_t comp_bytes = comp_count * sizeof(uint16_t);
    const uint64_t q_bytes = q_count * sizeof(float);
    const uint64_t mask_bytes = mask_count * sizeof(float);
    const uint64_t page = (uint64_t)getpagesize();

    const test_metal_prefill_mask_cache_shape shape_a = {
        .n_tokens = 128,
        .n_comp = kind == TEST_METAL_PREFILL_MASK_CACHE_RAW ? 0u :
                  (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? 32u : 1u),
        .window = 128,
        .ratio = kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? 4u :
                 (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO128 ? 128u : 0u),
    };
    const test_metal_prefill_mask_cache_shape shape_b = {
        .n_tokens = 129,
        .n_comp = kind == TEST_METAL_PREFILL_MASK_CACHE_RAW ? 0u :
                  (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? 31u : 2u),
        .window = 63,
        .ratio = shape_a.ratio,
    };
    const size_t count_a =
        (size_t)shape_a.n_tokens * n_head * head_dim;
    const size_t count_b =
        (size_t)shape_b.n_tokens * n_head * head_dim;

    ds4_gpu_tensor *raw = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *comp_mask = ds4_gpu_tensor_alloc(mask_bytes);
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(q_bytes);
    float *raw_host = malloc((size_t)raw_bytes);
    uint16_t *comp_host = malloc((size_t)comp_bytes);
    float *q_host = calloc((size_t)q_count, sizeof(float));
    float *mask_host = malloc((size_t)mask_bytes);
    float *ref_a = malloc((size_t)q_bytes);
    float *ref_b = malloc((size_t)q_bytes);
    float *actual = malloc((size_t)q_bytes);
    float *masked_actual = malloc((size_t)q_bytes);
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)page, (size_t)page) == 0);

    TEST_ASSERT(raw != NULL);
    TEST_ASSERT(comp != NULL);
    TEST_ASSERT(q != NULL);
    TEST_ASSERT(comp_mask != NULL);
    TEST_ASSERT(heads != NULL);
    TEST_ASSERT(raw_host != NULL);
    TEST_ASSERT(comp_host != NULL);
    TEST_ASSERT(q_host != NULL);
    TEST_ASSERT(mask_host != NULL);
    TEST_ASSERT(ref_a != NULL);
    TEST_ASSERT(ref_b != NULL);
    TEST_ASSERT(actual != NULL);
    TEST_ASSERT(masked_actual != NULL);
    TEST_ASSERT(model_raw != NULL);

    const char *disable_env =
        "DS4_METAL_DISABLE_ZERO_PREFIX_PREFILL_MASK_CACHE";
    char *saved_disable = test_save_env(disable_env);
    size_t total_mismatches = 0;
    uint32_t max_ulp = 0;
    size_t key_difference = 0;
    size_t masked_difference = 0;

    const bool allocated = raw && comp && q && comp_mask && heads &&
        raw_host && comp_host && q_host && mask_host && ref_a && ref_b &&
        actual && masked_actual && model_raw;
    if (allocated) {
        memset(model_raw, 0, (size_t)page);
        ((float *)model_raw)[0] = -1.0f;

        for (uint32_t row = 0; row < max_tokens; row++) {
            for (uint32_t col = 0; col < head_dim; col++) {
                const int value = (int)((row * 37u + col * 17u +
                                         (col ^ (col >> 3u)) * 5u +
                                         seed * 13u) % 257u) - 128;
                raw_host[(uint64_t)row * head_dim + col] =
                    (float)value / 256.0f;
            }
        }
        for (uint32_t row = 0; row < max_comp; row++) {
            for (uint32_t col = 0; col < head_dim; col++) {
                const int value = (int)((row * 29u + col * 11u +
                                         (col ^ (col >> 4u)) * 7u +
                                         seed * 19u) % 193u) - 96;
                comp_host[(uint64_t)row * head_dim + col] =
                    test_float_to_f16(0.375f + (float)value / 384.0f);
            }
        }
        for (uint64_t i = 0; i < mask_count; i++) {
            mask_host[i] = -65504.0f;
        }
        if (shape_a.n_comp != 0) {
            for (uint32_t row = 0; row < shape_a.n_tokens; row++) {
                const uint32_t visible = (row + 1u) / shape_a.ratio;
                for (uint32_t col = 0; col < shape_a.n_comp; col++) {
                    if (col < visible) {
                        mask_host[(uint64_t)row * shape_a.n_comp + col] =
                            (col & 1u) != 0u ? -2.0f : -65504.0f;
                    }
                }
            }
        }

        TEST_ASSERT(ds4_gpu_tensor_write(raw, 0, raw_host, raw_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(comp, 0, comp_host, comp_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(q, 0, q_host, q_bytes) != 0);
        TEST_ASSERT(ds4_gpu_tensor_write(
                        comp_mask, 0, mask_host, mask_bytes) != 0);
        TEST_ASSERT(ds4_gpu_set_model_map(model_raw, page) != 0);
        ds4_gpu_set_quality(false);

        TEST_ASSERT(setenv(disable_env, "1", 1) == 0);
        const bool have_ref_a =
            test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, ref_a);
        const bool have_ref_b =
            test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_b, false, n_head, head_dim, ref_b);

        if (have_ref_a && have_ref_b) {
            const test_float_compare_stats key_stats =
                test_compare_float_bits(ref_a, ref_b, count_a);
            key_difference = key_stats.mismatch_count;
            TEST_ASSERT(key_difference != 0);
        }

        TEST_ASSERT(unsetenv(disable_env) == 0);

        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_a, actual, count_a, &total_mismatches, &max_ulp);
        }
        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_a, actual, count_a, &total_mismatches, &max_ulp);
        }

        if (kind != TEST_METAL_PREFILL_MASK_CACHE_RAW) {
            if (test_metal_zero_prefix_prefill_mask_cache_run(
                    kind, heads, model_raw, page, q, raw, comp, comp_mask,
                    &shape_a, true, n_head, head_dim, masked_actual) &&
                have_ref_a) {
                const test_float_compare_stats masked_stats =
                    test_compare_float_bits(ref_a, masked_actual, count_a);
                masked_difference = masked_stats.mismatch_count;
                TEST_ASSERT(masked_difference != 0);
            }
            if (test_metal_zero_prefix_prefill_mask_cache_run(
                    kind, heads, model_raw, page, q, raw, comp, comp_mask,
                    &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
                test_metal_zero_prefix_prefill_mask_cache_compare(
                    ref_a, actual, count_a,
                    &total_mismatches, &max_ulp);
            }
        }

        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_b, false, n_head, head_dim, actual) && have_ref_b) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_b, actual, count_b, &total_mismatches, &max_ulp);
        }
        if (test_metal_zero_prefix_prefill_mask_cache_run(
                kind, heads, model_raw, page, q, raw, comp, comp_mask,
                &shape_a, false, n_head, head_dim, actual) && have_ref_a) {
            test_metal_zero_prefix_prefill_mask_cache_compare(
                ref_a, actual, count_a, &total_mismatches, &max_ulp);
        }
    }

    test_restore_env(disable_env, saved_disable);
    const char *kind_name = kind == TEST_METAL_PREFILL_MASK_CACHE_RAW ? "raw" :
        (kind == TEST_METAL_PREFILL_MASK_CACHE_RATIO4 ? "ratio4" : "ratio128");
    fprintf(stderr,
            "ds4-test: zero-prefix prefill mask cache %s exact "
            "mismatches=%zu max_ulp=%u key_diff=%zu masked_diff=%zu\n",
            kind_name, total_mismatches, max_ulp,
            key_difference, masked_difference);
    TEST_ASSERT(total_mismatches == 0);
    TEST_ASSERT(max_ulp == 0);

    free(model_raw);
    free(masked_actual);
    free(actual);
    free(ref_b);
    free(ref_a);
    free(mask_host);
    free(q_host);
    free(comp_host);
    free(raw_host);
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(comp_mask);
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(comp);
    ds4_gpu_tensor_free(raw);
}

static void test_metal_zero_prefix_prefill_mask_cache_exact(void) {
    test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        TEST_METAL_PREFILL_MASK_CACHE_RAW, 41);
    test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        TEST_METAL_PREFILL_MASK_CACHE_RATIO4, 43);
    test_metal_zero_prefix_prefill_mask_cache_exact_kind(
        TEST_METAL_PREFILL_MASK_CACHE_RATIO128, 47);
}
