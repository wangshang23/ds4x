#include "../internal/backend_internal.cuh"

/* Storage implementation. */

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (ds4_gpu_tensor_alloc_on(t, 0, bytes) != 0) {
        free(t);
        return NULL;
    }
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    int ok = 0;
    /* Managed memory is not device-bound, but we record device 0 so that
     * subsequent ds4_gpu_tensor_free pairs with WITH_DEVICE(0) safely. */
    WITH_DEVICE(g_gpu[0].device_id) {
        ok = cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes),
                     "managed tensor alloc");
    }
    if (!ok) { free(t); return NULL; }
    t->bytes = bytes;
    t->owner = 1;
    t->device_id = 0;
    return t;
}

/* Heap-allocated tensor with the runtime's explicit device-slot ABI. */
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int tier, uint64_t bytes) {
    if (tier < 0 || tier >= g_n_gpus) {
        fprintf(stderr,
            "ds4: ds4_gpu_tensor_alloc_ptr_on: bad tier %d (n_gpus=%d)\n",
            tier, g_n_gpus);
        return NULL;
    }
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (ds4_gpu_tensor_alloc_on(t, tier, bytes) != 0) {
        free(t);
        return NULL;
    }
    return t;
}

/* Managed-memory counterpart of ds4_gpu_tensor_alloc_ptr_on. */
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int tier, uint64_t bytes) {
    if (tier < 0 || tier >= g_n_gpus) {
        fprintf(stderr,
            "ds4: ds4_gpu_tensor_alloc_managed_on: bad tier %d (n_gpus=%d)\n",
            tier, g_n_gpus);
        return NULL;
    }
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    int ok = 0;
    /* Run the cudaMallocManaged call under the home tier's device so the
     * first-touch home matches the stamped device_id; the page itself can
     * migrate freely under managed-memory semantics. */
    WITH_DEVICE(g_gpu[tier].device_id) {
        ok = cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes),
                     "managed tensor alloc (tier)");
    }
    if (!ok) { free(t); return NULL; }
    t->bytes = bytes;
    t->owner = 1;
    t->device_id = tier;
    return t;
}

static uint64_t cuda_managed_kv_reserve_bytes(uint64_t total_bytes) {
    const uint64_t min_reserve = 8ull * 1073741824ull;
    const uint64_t max_reserve = 40ull * 1073741824ull;
    uint64_t reserve = total_bytes / 4u;
    if (reserve < min_reserve) reserve = min_reserve;
    if (reserve > max_reserve) reserve = max_reserve;
    return reserve;
}

extern "C" int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    if (kv_cache_bytes == 0) return 0;

    /* Very large KV caches are where device-only cudaMalloc() can make a
     * unified-memory machine unresponsive.  Managed memory restores the old
     * demand-paged behavior for this one long-lived allocation class only. */
    const uint64_t huge_kv = 8ull * 1073741824ull;
    if (kv_cache_bytes >= huge_kv) return 1;

    const uint64_t large_context = 8ull * 1073741824ull;
    if (context_bytes < large_context) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_managed_kv_reserve_bytes(total_bytes);
    if (context_bytes > free_bytes) return 1;
    return free_bytes - context_bytes < reserve_bytes;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->ptr = (char *)base->ptr + offset;
    t->bytes = bytes;
    t->owner = 0;
    t->device_id = base->device_id;  /* inherit owning device */
    return t;
}

extern "C" void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    int d = ds4_tensor_device_idx(tensor);
    if (tensor->owner && tensor->ptr) {
        WITH_DEVICE(g_gpu[d].device_id) {
            (void)cudaFree(tensor->ptr);
        }
    }
    free(tensor);
}

extern "C" uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

extern "C" void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    /* Full-device sync preserves legacy semantics. */
    (void)cudaDeviceSynchronize();
    return tensor->ptr;
}

extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    int d = ds4_tensor_device_idx(tensor);
    int ok = 0;
    WITH_DEVICE(g_gpu[d].device_id) {
        fill_f32_kernel<<<(count + 255u) / 256u, 256>>>((float *)tensor->ptr, count, value);
        ok = cuda_ok(cudaGetLastError(), "tensor fill f32 launch");
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    int d = ds4_tensor_device_idx(tensor);
    int ok = 0;
    WITH_DEVICE(g_gpu[d].device_id) {
        ok = cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes,
                                cudaMemcpyHostToDevice),
                     "tensor write");
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    int d = ds4_tensor_device_idx(tensor);
    int ok = 0;
    WITH_DEVICE(g_gpu[d].device_id) {
        ok = cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes,
                                cudaMemcpyDeviceToHost),
                     "tensor read");
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                     const ds4_gpu_tensor *src, uint64_t src_offset,
                                     uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes || src_offset > src->bytes ||
        bytes > dst->bytes - dst_offset || bytes > src->bytes - src_offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    /* DS4X owns one GB10, so every tensor copy is device-local. */
    int d = ds4_tensor_device_idx(dst);
    int ok = 0;
    WITH_DEVICE(g_gpu[d].device_id) {
        ok = cuda_ok(cudaMemcpy((char *)dst->ptr + dst_offset,
                                (const char *)src->ptr + src_offset,
                                (size_t)bytes,
                                cudaMemcpyDeviceToDevice),
                     "tensor copy");
    }
    return ok;
}

__global__ static void pack_slot_rows_f32_kernel(float *out, const float *slots, uint32_t n_rows, uint32_t width, uint32_t n_slots, uint32_t slot_cap);

extern "C" int ds4_gpu_pack_slot_rows_f32_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *slots,
        uint32_t                n_rows,
        uint32_t                width,
        uint32_t                n_slots,
        uint32_t                slot_cap) {
    uint64_t slot_rows = 0;
    uint64_t slot_elems = 0;
    uint64_t out_rows = 0;
    uint64_t out_elems = 0;
    if (!out || !slots || n_rows == 0 || width == 0 || n_slots == 0 ||
        slot_cap == 0 || n_rows > slot_cap ||
        (uint64_t)n_slots > UINT64_MAX / slot_cap ||
        (slot_rows = (uint64_t)n_slots * slot_cap) > UINT64_MAX / width ||
        (slot_elems = slot_rows * width) > UINT64_MAX / sizeof(float) ||
        (uint64_t)n_rows > UINT64_MAX / n_slots ||
        (out_rows = (uint64_t)n_rows * n_slots) > UINT64_MAX / width ||
        (out_elems = out_rows * width) > UINT64_MAX / sizeof(float) ||
        slots->bytes < slot_elems * sizeof(float) ||
        out->bytes < out_elems * sizeof(float)) {
        return 0;
    }
    const uint64_t blocks = (out_elems + 255u) / 256u;
    if (blocks > UINT32_MAX) return 0;
    pack_slot_rows_f32_kernel<<<(unsigned)blocks, 256>>>(
            (float *)out->ptr,
            (const float *)slots->ptr,
            n_rows,
            width,
            n_slots,
            slot_cap);
    return cuda_ok(cudaGetLastError(), "pack_slot_rows_f32 launch");
}

extern "C" int ds4_gpu_begin_commands(void) { return 1; }
extern "C" int ds4_gpu_flush_commands(void) { return cuda_ok(cudaDeviceSynchronize(), "flush"); }
extern "C" int ds4_gpu_end_commands(void) {
    if (g_cuda_end_stream_sync) {
        return cuda_ok(cudaStreamSynchronize(0), "end commands stream");
    }
    return cuda_ok(cudaDeviceSynchronize(), "end commands");
}
extern "C" int ds4_gpu_synchronize(void) { return cuda_ok(cudaDeviceSynchronize(), "synchronize"); }

extern "C" int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!ds4_gpu_register_model_map_no_copy(model_map, model_size)) return 0;
    if (getenv("DS4_CUDA_COPY_MODEL_CHUNKED") != NULL &&
        !cuda_model_copy_chunked(model_map, model_size, map_offset, map_size)) {
        (void)cuda_model_prefetch_range(model_map, model_size, map_offset, map_size);
    }
    return 1;
}

/* Register the mmap'd host model without forcing a full device copy. */
extern "C" int ds4_gpu_register_model_map_no_copy(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_model_host_base == model_map && g_model_registered_size == model_size) return 1;

    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    if (g_model_device_owned && g_model_device_base) {
        (void)cudaFree((void *)g_model_device_base);
        g_model_device_owned = 0;
    }
    if (g_model_registered && g_model_host_base) {
        (void)cudaHostUnregister((void *)g_model_host_base);
        g_model_registered = 0;
    }
    g_model_host_base = model_map;
    g_model_device_base = (const char *)model_map;
    g_model_registered_size = model_size;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_cache_full = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }

    if (cuda_integrated_artifact_map(model_map)) {
        fprintf(stderr,
                "ds4: CUDA aligned artifacts replace expert residency; "
                "leaving the %.2f GiB model mmap unpinned\n",
                (double)model_size / 1073741824.0);
        return 1;
    }

    cudaError_t err = cudaHostRegister((void *)model_map, (size_t)model_size,
                                       cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    if (err == cudaSuccess) {
        void *dev = NULL;
        err = cudaHostGetDevicePointer(&dev, (void *)model_map, 0);
        if (err == cudaSuccess && dev) {
            g_model_device_base = (const char *)dev;
            g_model_registered = 1;
            fprintf(stderr,
                    "ds4: CUDA registered %.2f GiB no-copy model mapping\n",
                    (double)model_size / 1073741824.0);
        } else {
            fprintf(stderr,
                    "ds4: CUDA (no-copy) host registration pointer lookup failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    } else {
        fprintf(stderr,
                "ds4: CUDA (no-copy) host registration skipped: %s\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_fd(int fd) {
    g_model_fd = fd;
    g_model_fd_host_base = g_model_host_base;
    g_model_file_size = 0;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            g_model_file_size = (uint64_t)st.st_size;
            if (st.st_blksize > 1) g_model_direct_align = (uint64_t)st.st_blksize;
        }
#if defined(__linux__) && defined(O_DIRECT)
        if (getenv("DS4_CUDA_NO_DIRECT_IO") == NULL) {
            char proc_path[64];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
            int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
            if (direct_fd >= 0) {
                g_model_direct_fd = direct_fd;
                if (g_model_direct_align < 512) g_model_direct_align = 512;
                if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "ds4: CUDA model direct I/O enabled (align=%llu)\n",
                            (unsigned long long)g_model_direct_align);
                }
            } else if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                fprintf(stderr, "ds4: CUDA model direct I/O unavailable: %s\n", strerror(errno));
            }
        }
#endif
    }
    return 1;
}

extern "C" int ds4_gpu_build_derived_artifacts(
        const void *model_map,
        uint64_t model_size,
        const char *model_path) {
    if (!model_map || model_size == 0 || !model_path || !model_path[0]) return 0;
    if (!g_derived_ranges.empty()) return (int)g_derived_ranges.size();
    if (getenv("DS4_CUDA_NO_DERIVED_WEIGHTS") != NULL) return 0;
    const char *build = getenv("DS4_CUDA_BUILD_ARTIFACTS");
    if (build && strcmp(build, "0") == 0) return 0;

    int device = 0;
    int integrated = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device) != cudaSuccess ||
        !integrated) {
        return 0;
    }

    ds4_repack_file mapped;
    if (!ds4_repack_map_file("ds4", model_path, mapped)) return 0;
    if (mapped.size != model_size) {
        ds4_repack_unmap_file(mapped);
        fprintf(stderr, "ds4: aligned artifact build skipped: model size changed\n");
        return 0;
    }
    std::vector<ds4_repack_tensor> records;
    const bool catalog_ok =
        ds4_repack_collect_catalog("ds4", mapped, nullptr, &records);
    ds4_repack_unmap_file(mapped);
    if (!catalog_ok) return 0;

    const bool build_moe =
        cuda_aligned_iq2_enabled() && cuda_aligned_q2k_enabled();
    const bool build_q8 = cuda_aligned_q8_enabled();
    if (!build_moe && !build_q8) return 0;

    ds4_repack_build_args args;
    args.log_prefix = "ds4";
    args.model_id = "base";
    args.path = model_path;
    args.records = &records;
    args.device = device;
    args.copy_chunk_bytes = 256ull * 1048576ull;

    const double t0 = cuda_wall_sec();
    std::vector<ds4_repack_artifact> artifacts;
    uint64_t built_bytes = 0;
    uint64_t part_bytes = 0;
    bool ok = true;
    if (build_moe) {
        ok = ds4_repack_build_q2k_aligned(args, artifacts, &part_bytes);
        built_bytes += part_bytes;
    }
    if (ok && build_moe) {
        ok = ds4_repack_build_iq2_aligned(args, artifacts, &part_bytes);
        built_bytes += part_bytes;
    }
    if (ok && build_q8) {
        ok = ds4_repack_build_q8_aligned(args, artifacts, &part_bytes);
        built_bytes += part_bytes;
    }
    if (!ok || artifacts.empty()) {
        for (ds4_repack_artifact &artifact : artifacts) {
            if (artifact.dev) (void)cudaFree(artifact.dev);
        }
        fprintf(stderr, "ds4: aligned artifact build failed; using raw weights\n");
        return 0;
    }

    for (const ds4_repack_artifact &artifact : artifacts) {
        g_derived_ranges.push_back({
            model_map,
            artifact.t->off,
            artifact.t->bytes,
            artifact.kind,
            artifact.in_dim,
            artifact.out_dim,
            artifact.group_count,
            artifact.bytes,
            (char *)artifact.dev,
        });
    }

    int replaces_complete = build_moe ? 1 : 0;
    uint64_t replace_candidates = 0;
    for (const ds4_repack_tensor &tensor : records) {
        if (!ds4_repack_iq2_candidate(tensor) &&
            !ds4_repack_q2k_candidate(tensor)) {
            continue;
        }
        replace_candidates++;
        bool found = false;
        for (const ds4_repack_artifact &artifact : artifacts) {
            if (artifact.t == &tensor) {
                found = true;
                break;
            }
        }
        if (!found) replaces_complete = 0;
    }
    if (replace_candidates == 0) replaces_complete = 0;

    g_derived_replace_map = model_map;
    g_derived_replaces_complete = replaces_complete;
    g_derived_artifact_bytes = built_bytes;
    g_derived_artifact_build_secs = cuda_wall_sec() - t0;
    if (!g_aligned_q81_scratch) {
        cudaError_t scratch_err = cudaMalloc(&g_aligned_q81_scratch, 256u * 1024u);
        if (scratch_err == cudaSuccess) {
            ds4_mmq_set_aligned_q81_scratch(g_aligned_q81_scratch,
                                             256u * 1024u);
        } else {
            g_aligned_q81_scratch = NULL;
            (void)cudaGetLastError();
            fprintf(stderr,
                    "ds4: aligned decode scratch allocation failed: %s\n",
                    cudaGetErrorString(scratch_err));
        }
    }
    fprintf(stderr,
            "ds4: built %llu aligned CUDA artifacts (%.2f GiB) in %.1fs%s\n",
            (unsigned long long)g_derived_ranges.size(),
            (double)g_derived_artifact_bytes / 1073741824.0,
            g_derived_artifact_build_secs,
            replaces_complete ? "; expert raw residency replaced" : "");
    return (int)g_derived_ranges.size();
}

extern "C" int ds4_gpu_model_range_replaced(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes) {
    if (!cuda_model_map_replaces_complete(model_map) || bytes == 0) return 0;
    for (const cuda_derived_range &range : g_derived_ranges) {
        if (range.host_base == model_map &&
            range.source_offset == offset &&
            range.source_bytes == bytes &&
            (range.kind == CUDA_DERIVED_IQ2_XXS_ALIGNED_MOE ||
             range.kind == CUDA_DERIVED_Q2_K_ALIGNED_MOE)) {
            return 1;
        }
    }
    return 0;
}

extern "C" int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    if (cuda_span_fully_replaced(model_map, offset, bytes)) return 1;
    if (!cuda_model_range_ptr(model_map, offset, bytes, label ? label : "model_tensor")) return 0;
    return cuda_model_range_is_cached(model_map, offset, bytes);
}

extern "C" int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    static int optional_q8_preload_disabled = 0;
    if (optional_q8_preload_disabled) return 1;
    const char *cache_label = label ? label : "q8_0";
    if (!cuda_q8_f16_preload_allowed(cache_label, in_dim, out_dim)) return 1;
    if (cuda_q8_f16_ptr(model_map, offset, bytes, in_dim, out_dim, 0, cache_label)) return 1;
    optional_q8_preload_disabled = 1;
    return 1;
}

extern "C" void ds4_gpu_print_memory_report(const char *label) {
    size_t free_b = 0, total_b = 0;
    (void)cudaMemGetInfo(&free_b, &total_b);
    fprintf(stderr, "ds4: CUDA memory report %s: free %.2f MiB total %.2f MiB\n",
            label ? label : "", (double)free_b / 1048576.0, (double)total_b / 1048576.0);
}

extern "C" void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
    const cublasMath_t math_mode =
        (g_quality_mode || getenv("DS4_CUDA_NO_TF32") != NULL)
            ? CUBLAS_DEFAULT_MATH
            : CUBLAS_TF32_TENSOR_OP_MATH;
    if (g_gpu[0].cublas_ready && g_gpu[0].cublas) {
        cublasStatus_t st = cublasSetMathMode(
                (cublasHandle_t)g_gpu[0].cublas, math_mode);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                    "ds4: ds4_gpu_set_quality: cublasSetMathMode failed: "
                    "status %d\n",
                    (int)st);
        }
    }
}

__global__ static void pack_slot_rows_f32_kernel(float *out, const float *slots, uint32_t n_rows, uint32_t width, uint32_t n_slots, uint32_t slot_cap) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_rows * n_slots * width;
    if (i >= n) return;

    uint64_t col = i % width;
    uint64_t slot = (i / width) % n_slots;
    uint64_t row = i / ((uint64_t)n_slots * width);
    out[i] = slots[((slot * slot_cap) + row) * width + col];
}

int cuda_q8_mma_attr_ready[DS4_MAX_GPUS][4];
int cuda_q8_mma_try_launch(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        uint64_t a_stride_blocks,
        uint64_t out_stride,
        uint32_t T) {
    static int disabled = -1;
    if (disabled < 0) disabled = getenv("DS4_CUDA_NO_Q8_MMA") != NULL ? 1 : 0;
    if (disabled || !cuda_q4_mma_ok()) return 0;
    if ((in_dim & 31u) != 0u || blocks > 256u || n_tok < 8u) return 0;
    if (((uintptr_t)w & 1u) || ((uintptr_t)xq & 3u) || ((uintptr_t)xscale & 3u)) return 0;
    const size_t shmem = (size_t)(64u * blocks * 2u + 16u * blocks * 4u);
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_MAX_GPUS) return 0;
    const int ti = T == 32u ? 0 : (T == 64u ? 1 : (T == 128u ? 2 : 3));
    dim3 grid(((unsigned)out_dim + 63u) / 64u, ((unsigned)n_tok + 15u) / 16u, 1);
#define DS4_Q8_MMA_LAUNCH(TT) \
    do { \
        if (!cuda_q8_mma_attr_ready[dev][ti]) { \
            cudaFuncAttributes fn_attr; \
            if (cudaFuncGetAttributes(&fn_attr, matmul_q8_0_mma_exact_kernel<TT>) != cudaSuccess || \
                fn_attr.binaryVersion < 80) { \
                disabled = 1; \
                return 0; \
            } \
            if (cudaFuncSetAttribute(matmul_q8_0_mma_exact_kernel<TT>, \
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                     (int)(64u * 256u * 2u + 16u * 256u * 4u)) != cudaSuccess) { \
                disabled = 1; \
                return 0; \
            } \
            cuda_q8_mma_attr_ready[dev][ti] = 1; \
        } \
        matmul_q8_0_mma_exact_kernel<TT><<<grid, 256, shmem>>>( \
                out, w, xq, xscale, in_dim, out_dim, n_tok, blocks, \
                a_stride_blocks, out_stride); \
    } while (0)
    if (T == 32u) DS4_Q8_MMA_LAUNCH(32u);
    else if (T == 64u) DS4_Q8_MMA_LAUNCH(64u);
    else if (T == 128u) DS4_Q8_MMA_LAUNCH(128u);
    else DS4_Q8_MMA_LAUNCH(256u);
#undef DS4_Q8_MMA_LAUNCH
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 mma launch") ? 1 : -1;
}
