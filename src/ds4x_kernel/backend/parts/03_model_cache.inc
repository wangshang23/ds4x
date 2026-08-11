/* =========================================================================
 * Per-device selective model cache (selective model cache).
 *
 * ds4_gpu_device_cache_tensors copies the listed source ranges from the
 * host mmap onto device_id's selective slab and appends sorted lookup
 * entries. The legacy chunked-copy machinery (cuda_model_range_*) is
 * NOT disturbed — it continues to drive all existing callers. New
 * lookups fall back to it when no selective entry covers the range.
 *
 * Caller-context preference for overlap: when the same source range is
 * cached on multiple devices, ds4_gpu_lookup_cache returns the entry
 * whose device matches cudaGetDevice().
 * ========================================================================= */

extern "C" int ds4_gpu_device_cache_tensors(int device_id,
                                            const ds4_tensor_range *ranges,
                                            int n_ranges) {
    if (device_id < 0 || device_id >= DS4_MAX_GPUS) return 1;
    if (n_ranges < 0 || (!ranges && n_ranges > 0)) return 2;
    if (n_ranges == 0) return 0;

    if (!g_model_host_base || g_model_registered_size == 0) return 3;

    /* Validate ranges against the mmap'd model bounds; reject ranges
     * that overflow or extend past the mapped region. Done in a
     * separate pass before any allocation so a bad input doesn't
     * partially grow the slab. */
    uint64_t want_bytes = 0;
    for (int i = 0; i < n_ranges; i++) {
        if (ranges[i].target_device != device_id) continue;
        const uint64_t off = ranges[i].source_offset;
        const uint64_t nb  = ranges[i].bytes;
        /* Overflow-safe upper bound: off + nb must not exceed model
         * size, and the sum must not wrap. */
        if (nb == 0) continue;
        if (off > g_model_registered_size) return 8;
        if (nb > g_model_registered_size - off) return 9;
        /* Accumulate into want_bytes with overflow check. */
        if (want_bytes > UINT64_MAX - nb) return 10;
        want_bytes += nb;
    }
    if (want_bytes == 0) return 0;

    cuda_device_cache &c = g_dev_cache[device_id];

    int prev_device = -1;
    if (cudaGetDevice(&prev_device) != cudaSuccess) prev_device = -1;
    if (cudaSetDevice(device_id) != cudaSuccess) return 4;

    /* Allocate or grow the slab via cudaMalloc + d2d copy. */
    void *new_base = NULL;
    size_t new_bytes = c.bytes + want_bytes;

    /* Refuse cleanly before cudaMalloc if the device clearly cannot hold
     * the slab. The multi-tier packer reserves per-tier runtime scratch
     * before placing tensors, but it cannot predict the cudaMalloc
     * allocator's overhead (alignment, fragmentation after CUDA context
     * init, default driver-side reservations). On a borderline budget
     * that overhead pushes a "fits-by-packer-math" layout past the actual
     * free pool and the cudaMalloc below OOMs after the engine already
     * committed to the layout — same silent-late-OOM failure mode the
     * upfront refusal path was added to eliminate. Catch it here too. */
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            /* free_b already excludes the existing slab (it's still
             * allocated), so the additional cudaMalloc only needs
             * new_bytes free — not new_bytes + c.bytes. The old slab is
             * freed AFTER the d2d copy succeeds. 2 GiB safety covers what
             * the engine will allocate AFTER the cache slab in the same
             * session_create: per-tier graph scratch (the planner can't
             * predict its cumulative cudaMalloc alignment overhead),
             * cuBLAS workspace beyond the 64 MiB the packer already
             * reserves, and driver-side allocator slack. Without this
             * headroom a borderline budget that fits the slab itself can
             * still OOM at the per-tier tensor allocations a few moments
             * later — same silent-late-OOM failure mode, one layer up. */
            const size_t safety = (size_t)2ull * 1024ull * 1024ull * 1024ull;
            const size_t need = new_bytes + safety;
            if (need > free_b) {
                fprintf(stderr,
                        "ds4: device cache slab needs %.2f GiB on device %d "
                        "but only %.2f GiB free (slab=%.2f GiB + %.2f GiB safety). "
                        "Lower --gpu-vram / --ctx-max, or use --gpu-vram auto on "
                        "a host with more free VRAM. Refusing upfront to avoid "
                        "late OOM at cudaMalloc.\n",
                        (double)need / 1073741824.0,
                        device_id,
                        (double)free_b / 1073741824.0,
                        (double)new_bytes / 1073741824.0,
                        (double)safety / 1073741824.0);
                if (prev_device >= 0) (void)cudaSetDevice(prev_device);
                return 5;
            }
        }
        /* If cudaMemGetInfo itself failed, fall through; cudaMalloc's own
         * error path still catches the late case, just with a less helpful
         * message. */
    }

    if (!cuda_ok(cudaMalloc(&new_base, new_bytes), "device cache alloc")) {
        if (prev_device >= 0) (void)cudaSetDevice(prev_device);
        return 5;
    }
    if (c.present && c.bytes > 0) {
        cudaError_t e = cudaMemcpy(new_base, c.base, c.bytes,
                                   cudaMemcpyDeviceToDevice);
        if (e != cudaSuccess) {
            cuda_ok(e, "device cache grow d2d");
            (void)cudaFree(new_base);
            if (prev_device >= 0) (void)cudaSetDevice(prev_device);
            return 6;
        }
        /* Re-base existing entries on this device. */
        char *old_base = (char *)c.base;
        char *grown    = (char *)new_base;
        for (size_t k = 0; k < g_cache_ranges.size(); k++) {
            if (g_cache_ranges[k].device_id == device_id) {
                g_cache_ranges[k].device_ptr =
                    grown + ((char *)g_cache_ranges[k].device_ptr - old_base);
            }
        }
        (void)cudaFree(c.base);
    }
    c.base = new_base;
    c.bytes = new_bytes;
    c.present = 1;

    /* Copy ranges and append entries. */
    const char *host_base = (const char *)g_model_host_base;
    size_t write_off = c.bytes - want_bytes;
    for (int i = 0; i < n_ranges; i++) {
        if (ranges[i].target_device != device_id) continue;
        char *dev_ptr = (char *)c.base + write_off;
        cudaError_t e = cudaMemcpy(dev_ptr,
                                   host_base + ranges[i].source_offset,
                                   (size_t)ranges[i].bytes,
                                   cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            cuda_ok(e, "device cache range h2d");
            if (prev_device >= 0) (void)cudaSetDevice(prev_device);
            return 7;
        }
        cache_range_entry ent;
        ent.source_offset = ranges[i].source_offset;
        ent.bytes         = ranges[i].bytes;
        ent.device_id     = device_id;
        ent.device_ptr    = dev_ptr;
        g_cache_ranges.push_back(ent);
        write_off += ranges[i].bytes;
    }

    /* Keep sorted by source_offset for binary-search lookup. */
    std::sort(g_cache_ranges.begin(), g_cache_ranges.end(),
              [](const cache_range_entry &a, const cache_range_entry &b) {
                  if (a.source_offset != b.source_offset)
                      return a.source_offset < b.source_offset;
                  return a.device_id < b.device_id;
              });

    if (prev_device >= 0) (void)cudaSetDevice(prev_device);
    return 0;
}

/* Install support-model tensor ranges into device_id's strict cache,
 * copying from the registered support map and keying entries at
 * source_offset + bias. Standalone slab (does not touch the main cache
 * slab growth path). */
extern "C" int ds4_gpu_device_cache_support_tensors(int device_id,
                                                    int entry_device_id,
                                                    const ds4_tensor_range *ranges,
                                                    int n_ranges,
                                                    int from_main_map) {
    if (device_id < 0 || device_id >= DS4_MAX_GPUS) return 1;
    if (entry_device_id < 0 || entry_device_id >= DS4_MAX_GPUS) return 1;
    if (n_ranges <= 0 || !ranges) return 2;
    const char *src_base;
    uint64_t src_size;
    uint64_t key_bias;
    if (from_main_map) {
        /* Auxiliary main-model ranges (e.g. the embedding bucket for the
         * DSpark executor tier): standalone slab, unbiased offsets. */
        src_base = (const char *)g_model_host_base;
        src_size = g_model_registered_size;
        key_bias = 0;
    } else {
        src_base = (const char *)g_support_host_base;
        src_size = g_support_host_size;
        key_bias = g_support_offset_bias;
        if (key_bias == 0) return 3;
    }
    if (!src_base || src_size == 0) return 3;
    uint64_t want = 0;
    for (int i = 0; i < n_ranges; i++) {
        const uint64_t off = ranges[i].source_offset;
        const uint64_t nb = ranges[i].bytes;
        if (nb == 0) continue;
        if (off > src_size || nb > src_size - off) return 8;
        if (want > UINT64_MAX - nb) return 9;
        want += nb;
    }
    if (want == 0) return 0;
    int prev_device = -1;
    if (cudaGetDevice(&prev_device) != cudaSuccess) prev_device = -1;
    if (cudaSetDevice(device_id) != cudaSuccess) return 4;
    void *base = NULL;
    if (!cuda_ok(cudaMalloc(&base, (size_t)want), "support cache alloc")) {
        if (prev_device >= 0) (void)cudaSetDevice(prev_device);
        return 5;
    }
    const char *host_base = src_base;
    size_t write_off = 0;
    for (int i = 0; i < n_ranges; i++) {
        if (ranges[i].bytes == 0) continue;
        char *dev_ptr = (char *)base + write_off;
        cudaError_t e = cudaMemcpy(dev_ptr,
                                   host_base + ranges[i].source_offset,
                                   (size_t)ranges[i].bytes,
                                   cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            cuda_ok(e, "support cache range h2d");
            (void)cudaFree(base);
            if (prev_device >= 0) (void)cudaSetDevice(prev_device);
            return 7;
        }
        cache_range_entry ent;
        ent.source_offset = ranges[i].source_offset + key_bias;
        ent.bytes = ranges[i].bytes;
        /* Entries can claim a different (executor) device than the one the
         * slab physically lives on: strict lookups filter by entry device,
         * and peer access lets the executor's kernels dereference the
         * spilled pointer directly. */
        ent.device_id = entry_device_id;
        ent.device_ptr = dev_ptr;
        g_cache_ranges.push_back(ent);
        write_off += ranges[i].bytes;
    }
    std::sort(g_cache_ranges.begin(), g_cache_ranges.end(),
              [](const cache_range_entry &a, const cache_range_entry &b) {
                  if (a.source_offset != b.source_offset)
                      return a.source_offset < b.source_offset;
                  return a.device_id < b.device_id;
              });
    if (getenv("DS4_DSPARK_VERIFY_CACHE") != NULL) {
        /* Read back every installed range and compare with the host copy. */
        int bad = 0;
        write_off = 0;
        for (int i = 0; i < n_ranges; i++) {
            if (ranges[i].bytes == 0) continue;
            char *dev_ptr = (char *)base + write_off;
            std::vector<char> tmp((size_t)ranges[i].bytes);
            if (cudaMemcpy(tmp.data(), dev_ptr, (size_t)ranges[i].bytes,
                           cudaMemcpyDeviceToHost) != cudaSuccess ||
                memcmp(tmp.data(), host_base + ranges[i].source_offset,
                       (size_t)ranges[i].bytes) != 0) {
                fprintf(stderr,
                        "ds4: support cache VERIFY MISMATCH offset=%llu bytes=%llu dev=%d\n",
                        (unsigned long long)ranges[i].source_offset,
                        (unsigned long long)ranges[i].bytes, device_id);
                bad++;
            }
            write_off += ranges[i].bytes;
        }
        fprintf(stderr, "ds4: support cache verify dev=%d ranges=%d bad=%d\n",
                device_id, n_ranges, bad);
    }
    if (prev_device >= 0) (void)cudaSetDevice(prev_device);
    return 0;
}

extern "C" int ds4_gpu_lookup_cache(uint64_t source_offset, uint64_t bytes,
                                    int *out_device_id, void **out_device_ptr) {
    int active_device = -1;
    (void)cudaGetDevice(&active_device);

    if (!g_cache_ranges.empty()) {
        /* upper_bound: first entry with source_offset > query.
         * Candidates are at strictly earlier positions; scan all of
         * them rather than breaking on the first non-covering entry,
         * because the table allows overlap across devices. */
        auto it = std::upper_bound(
            g_cache_ranges.begin(), g_cache_ranges.end(),
            source_offset,
            [](uint64_t off, const cache_range_entry &e) {
                return off < e.source_offset;
            });
        const cache_range_entry *match_any  = NULL;
        const cache_range_entry *match_pref = NULL;
        while (it != g_cache_ranges.begin()) {
            --it;
            /* Overflow-safe coverage check:
             *   1. source_offset >= it->source_offset
             *   2. bytes <= it->bytes - (source_offset - it->source_offset)
             * The second form computes only the remaining capacity inside
             * the entry, so neither side can overflow even with bytes ==
             * UINT64_MAX. */
            if (source_offset >= it->source_offset) {
                uint64_t into = source_offset - it->source_offset;
                if (into <= it->bytes && bytes <= it->bytes - into) {
                    if (it->device_id == active_device) {
                        match_pref = &*it;
                        break;
                    }
                    if (!match_any) match_any = &*it;
                }
            }
            /* Do NOT break on non-covering: an earlier entry may still
             * cover if its bytes extend far enough. */
        }
        const cache_range_entry *m = match_pref ? match_pref : match_any;
        if (m) {
            if (out_device_id) *out_device_id = m->device_id;
            if (out_device_ptr) {
                *out_device_ptr =
                    (char *)m->device_ptr + (source_offset - m->source_offset);
            }
            return 1;
        }
    }

    /* Legacy chunk-aware fallback (device 0 only). */
    const char *p = cuda_model_range_ptr_from_fd(g_model_host_base,
                                                  source_offset, bytes,
                                                  "lookup_cache");
    if (p) {
        if (out_device_id) *out_device_id = 0;
        if (out_device_ptr) *out_device_ptr = (void *)p;
        return 1;
    }
    return 0;
}

extern "C" int ds4_gpu_lookup_cache_device(uint64_t source_offset, uint64_t bytes) {
    int d = -1;
    if (!ds4_gpu_lookup_cache(source_offset, bytes, &d, NULL)) return -1;
    return d;
}

/* Strict per-device selective-cache lookup.
 *
 * Returns 1 only if a covering entry exists whose device_id matches the
 * caller-supplied expected_device. Otherwise returns 0 with *out_device_ptr
 * untouched. Unlike ds4_gpu_lookup_cache, this variant performs NO host-
 * pointer fallback (no FD-cache, no model_range_ptr_from_fd) and NO
 * different-device fallback. It is the canonical lookup for multi-tier
 * dispatch where consuming a different device's pointer would be a
 * correctness bug. expected_device is a PHYSICAL CUDA device id (the
 * value stored in g_gpu[logical_tier].device_id, not the logical tier
 * index). The caller is expected to have cudaSetDevice'd to
 * expected_device before invoking; the returned pointer is valid to
 * consume from that device's kernel. Added for
 * multi-GPU execution (multi-GPU execution). */
extern "C" int ds4_gpu_lookup_cache_strict(uint64_t source_offset,
                                            uint64_t bytes,
                                            int      expected_device,
                                            void   **out_device_ptr) {
    if (g_cache_ranges.empty()) return 0;

    auto it = std::upper_bound(
        g_cache_ranges.begin(), g_cache_ranges.end(),
        source_offset,
        [](uint64_t off, const cache_range_entry &e) {
            return off < e.source_offset;
        });
    while (it != g_cache_ranges.begin()) {
        --it;
        if (source_offset < it->source_offset) {
            /* Should not happen given upper_bound semantics, but defensive. */
            continue;
        }
        uint64_t into = source_offset - it->source_offset;
        if (into > it->bytes) continue;
        if (bytes > it->bytes - into) continue;
        if (it->device_id != expected_device) continue;
        if (out_device_ptr) {
            *out_device_ptr =
                (char *)it->device_ptr + (source_offset - it->source_offset);
        }
        return 1;
    }
    return 0;
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
    /* Preload runs before any multi-tier dispatch. The cache entries it creates
     * are device-0 by construction; multi-tier callers in kernel wrappers will
     * miss the linear scan (device_id filter) and allocate fresh per-device
     * copies the first time they're consulted. */
    if (getenv("DS4_CUDA_Q8_F32_PRELOAD") != NULL &&
        cuda_q8_f32_cache_allowed(cache_label, in_dim, out_dim)) {
        if (cuda_q8_f32_ptr(model_map, offset, bytes, in_dim, out_dim, 0, cache_label)) return 1;
        optional_q8_preload_disabled = 1;
        return 1;
    }
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
    /* Walk every initialized per-tier handle. Single-tier (g_n_gpus == 1)
     * walks exactly one entry. On any device-switch failure,
     * skip the tier and continue — the function is void and the math-mode
     * setting is advisory, but log so misconfiguration is visible. */
    for (int i = 0; i < g_n_gpus; i++) {
        if (!g_gpu[i].cublas_ready || !g_gpu[i].cublas) continue;
        int prev = -1;
        cudaError_t derr = cudaGetDevice(&prev);
        if (derr != cudaSuccess) {
            fprintf(stderr,
                "ds4: ds4_gpu_set_quality: cudaGetDevice failed before tier %d "
                "(dev=%d): %s; skipping\n",
                i, g_gpu[i].device_id, cudaGetErrorString(derr));
            (void)cudaGetLastError();
            continue;
        }
        derr = cudaSetDevice(g_gpu[i].device_id);
        if (derr != cudaSuccess) {
            fprintf(stderr,
                "ds4: ds4_gpu_set_quality: cudaSetDevice(%d) failed for tier %d: "
                "%s; skipping\n",
                g_gpu[i].device_id, i, cudaGetErrorString(derr));
            (void)cudaGetLastError();
            if (prev >= 0) (void)cudaSetDevice(prev);
            continue;
        }
        cublasStatus_t st = cublasSetMathMode((cublasHandle_t)g_gpu[i].cublas, math_mode);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                "ds4: ds4_gpu_set_quality: cublasSetMathMode failed on tier %d "
                "(dev=%d): status %d\n",
                i, g_gpu[i].device_id, (int)st);
        }
        if (prev >= 0) (void)cudaSetDevice(prev);
    }
}

__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}

__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}

__global__ static void matmul_f16_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(wr[i]) * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_f16_serial_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok || threadIdx.x != 0) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = 0; i < in_dim; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    out[tok * out_dim + row] = sum;
}

__global__ static void matmul_f16_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    __shared__ float partial[32];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = k0; i < k1; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) total += partial[i];
        out[tok * out_dim + row] = total;
    }
}

__global__ static void matmul_f16_small_out_hx_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out_dim) return;

    __shared__ float partial[32];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    for (uint64_t i = k0; i < k1; i++) {
        const float xv = __half2float(__float2half(x[i]));
        sum += __half2float(wr[i]) * xv;
    }
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) total += partial[i];
        out[row] = total;
    }
}

__global__ static void matmul_f16_small_out_batch_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    const uint64_t tok = (uint64_t)blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (tok >= n_tok || out_dim > 32u || blockDim.x != 256u) return;

    float acc[32];
    #pragma unroll
    for (uint32_t r = 0; r < 32u; r++) acc[r] = 0.0f;

    const float *xr = x + tok * in_dim;
    for (uint64_t i = tid; i < in_dim; i += 256u) {
        const float xv = xr[i];
        #pragma unroll
        for (uint32_t r = 0; r < 32u; r++) {
            if (r < out_dim) {
                acc[r] += __half2float(w[(uint64_t)r * in_dim + i]) * xv;
            }
        }
    }

    __shared__ float partial[32 * 256];
    #pragma unroll
    for (uint32_t r = 0; r < 32u; r++) {
        if (r < out_dim) partial[r * 256u + tid] = acc[r];
    }
    __syncthreads();

    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            #pragma unroll
            for (uint32_t r = 0; r < 32u; r++) {
                if (r < out_dim) {
                    partial[r * 256u + tid] += partial[r * 256u + tid + stride];
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (uint32_t r = 0; r < 32u; r++) {
            if (r < out_dim) out[tok * out_dim + r] = partial[r * 256u];
        }
    }
}

__global__ static void matmul_f16_pair_ordered_chunks_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out0_dim && row >= out1_dim) return;

    __shared__ float partial0[32];
    __shared__ float partial1[32];
    const uint32_t tid = threadIdx.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : w0;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : w1;
    for (uint64_t i = k0; i < k1; i++) {
        const float xv = x[i];
        if (row < out0_dim) sum0 += __half2float(wr0[i]) * xv;
        if (row < out1_dim) sum1 += __half2float(wr1[i]) * xv;
    }
    partial0[tid] = sum0;
    partial1[tid] = sum1;
    __syncthreads();
    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total0 += partial0[i];
            total1 += partial1[i];
        }
        if (row < out0_dim) out0[row] = total0;
        if (row < out1_dim) out1[row] = total1;
    }
}

__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    out[i] = row[i % n_embd];
}

__global__ static void repeat_hc_rows_kernel(float *out, const float *rows, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (i >= n) return;

    uint64_t hc_row = (uint64_t)n_hc * n_embd;
    uint64_t tok = i / hc_row;
    uint64_t embd = i % n_embd;
    out[i] = rows[tok * n_embd + embd];
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

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}

__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a) {
    if (use_dp4a && n == 32u) return dot_i8x32_dp4a(a, b);
    int32_t dot = 0;
    for (uint64_t i = 0; i < n; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
}

__global__ static DS4_CUDA_UNUSED void matmul_q8_0_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const uint64_t blocks = (in_dim + 31) / 32;
    const unsigned char *wr = w + row * blocks * 34;
    const float *xr = x + tok * in_dim;
    float acc = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) amax = fmaxf(amax, fabsf(xr[i0 + i]));
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        int dot = 0;
        for (uint64_t i = 0; i < bn; i++) {
            int q = (int)lrintf(xr[i0 + i] * id);
            q = q > 127 ? 127 : (q < -128 ? -128 : q);
            dot += (int)qs[i] * q;
        }
        acc += __half2float(*scale_h) * d * (float)dot;
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void quantize_q8_0_f32_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t in_dim,
        uint64_t blocks) {
    uint64_t b = blockIdx.x;
    uint64_t tok = blockIdx.y;
    if (b >= blocks) return;
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const float *xr = x + tok * in_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) vals[threadIdx.x] = fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[tok * blocks + b] = d;
    int8_t *dst = xq + (tok * blocks + b) * 32;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

__global__ static void quantize_q8_0_group_slice_rows_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t group_dim,
        uint64_t blocks,
        uint32_t n_groups_total,
        uint32_t group0,
        uint32_t group_cnt) {
    const uint64_t b = blockIdx.x;
    const uint64_t packed_row = blockIdx.y;
    if (b >= blocks) return;
    const uint64_t token = packed_row / group_cnt;
    const uint64_t group = group0 + packed_row - token * group_cnt;
    const uint64_t i0 = b * 32u;
    const uint64_t bn = group_dim - i0 < 32u ? group_dim - i0 : 32u;
    const float *xr = x +
        (token * n_groups_total + group) * group_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            vals[threadIdx.x] =
                fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0u) xscale[packed_row * blocks + b] = d;
    int8_t *dst = xq + (packed_row * blocks + b) * 32u;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

__global__ static void matmul_q8_0_preq_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_q8_0_preq_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32u;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

__device__ __forceinline__ static uint32_t q8_top1_float_ordered_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ static uint64_t q8_top1_pack_key(float v, uint32_t idx) {
    return ((uint64_t)q8_top1_float_ordered_key(v) << 32u) |
           (uint64_t)(0xffffffffu - idx);
}

__device__ __forceinline__ static float q8_top1_unpack_value(uint32_t ordered) {
    const uint32_t u = (ordered & 0x80000000u)
        ? (ordered ^ 0x80000000u)
        : ~ordered;
    return __uint_as_float(u);
}

__global__ static void matmul_q8_0_top1_preq_warp8_kernel(
        unsigned long long *best_key,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        uint32_t index_offset,
        int use_dp4a) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + warp;
    const bool valid = row < out_dim;
    float acc = 0.0f;

    if (valid) {
        const unsigned char *wr = w + row * blocks * 34;
        for (uint64_t b = lane; b < blocks; b += 32u) {
            uint64_t i0 = b * 32;
            uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
            const __half *scale_h = (const __half *)(wr + b * 34);
            const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
            const int8_t *xqb = xq + b * 32;
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc += __half2float(*scale_h) * xscale[b] * (float)dot;
        }
    }
    acc = warp_sum_f32(acc);

    __shared__ unsigned long long keys[8];
    if (lane == 0u) {
        keys[warp] = valid
            ? (unsigned long long)q8_top1_pack_key(acc, index_offset + (uint32_t)row)
            : 0ull;
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        unsigned long long block_best = keys[0];
        #pragma unroll
        for (uint32_t i = 1u; i < 8u; i++) {
            if (keys[i] > block_best) block_best = keys[i];
        }
        (void)atomicMax(best_key, block_best);
    }
}

__global__ static void matmul_q8_0_top1_unpack_kernel(
        uint32_t *selected,
        float *values,
        const unsigned long long *best_key) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    const uint64_t key = (uint64_t)best_key[0];
    const uint32_t ordered = (uint32_t)(key >> 32u);
    const uint32_t idx = 0xffffffffu - (uint32_t)key;
    selected[0] = idx;
    values[0] = q8_top1_unpack_value(ordered);
}

__global__ static void matmul_q8_0_kslice_preq_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t slice_dim,
        uint64_t out_dim,
        uint64_t full_blocks,
        uint64_t block_start,
        uint64_t slice_blocks,
        int use_dp4a) {
    const uint64_t row =
        (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    out += tok * out_dim;
    xq += tok * slice_blocks * 32u;
    xscale += tok * slice_blocks;
    const unsigned char *wr =
        w + row * full_blocks * 34u + block_start * 34u;
    float acc = 0.0f;
    for (uint64_t b = lane; b < slice_blocks; b += 32u) {
        uint64_t i0 = b * 32u;
        uint64_t bn = slice_dim - i0 < 32u ? slice_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb = xq + b * 32u;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

__global__ static void matmul_q8_0_pair_preq_warp8_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned char *wr0 = row < out0_dim ? w0 + row * blocks * 34 : NULL;
    const unsigned char *wr1 = row < out1_dim ? w1 + row * blocks * 34 : NULL;
    const int8_t *xqr = xq + tok * blocks * 32u;
    const float *xsr = xscale + tok * blocks;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const int8_t *xqb = xqr + b * 32;
        const float xs = xsr[b];
        if (wr0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc0 += __half2float(*scale_h) * xs * (float)dot;
        }
        if (wr1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc1 += __half2float(*scale_h) * xs * (float)dot;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        if (row < out0_dim) out0[tok * out0_dim + row] = acc0;
        if (row < out1_dim) out1[tok * out1_dim + row] = acc1;
    }
}

__global__ static void shared_mid_q8_0_preq_warp8_exact_kernel(
        float *mid,
        const unsigned char *gate_w,
        const unsigned char *up_w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        float clamp,
        const int32_t *selected,
        uint32_t expert_split,
        bool home_rank,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    if (selected) {
        /* Complementary predicates select exactly one writer; ties stay on
         * the home rank to avoid an unnecessary peer store. */
        uint32_t home_count = 0u;
        uint32_t peer_count = 0u;
        #pragma unroll
        for (uint32_t i = 0; i < 6u; i++) {
            const int32_t expert = selected[i];
            if (expert >= 0 && (uint32_t)expert < expert_split) {
                home_count++;
            } else if (expert >= 0 &&
                       (uint32_t)expert < 2u * expert_split) {
                peer_count++;
            }
        }
        const bool assigned = home_rank
            ? home_count <= peer_count : peer_count < home_count;
        if (!assigned) return;
    }
    const unsigned char *gate_row = gate_w + row * blocks * 34u;
    const unsigned char *up_row = up_w + row * blocks * 34u;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const int8_t *xqb = xq + b * 32u;
        const float xs = xscale[b];
        const unsigned char *gb = gate_row + b * 34u;
        const unsigned char *ub = up_row + b * 34u;
        gate += __half2float(*(const __half *)gb) * xs *
                (float)dot_i8_block((const int8_t *)(gb + 2u), xqb, bn,
                                    use_dp4a);
        up += __half2float(*(const __half *)ub) * xs *
              (float)dot_i8_block((const int8_t *)(ub + 2u), xqb, bn,
                                  use_dp4a);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0u) {
        if (clamp > 1.0e-6f) {
            gate = fminf(gate, clamp);
            up = fminf(fmaxf(up, -clamp), clamp);
        }
        const float silu = gate / (1.0f + expf(-gate));
        mid[row] = silu * up * 1.0f;
    }
}

__global__ static void matmul_q8_0_pair_preq_batch_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint64_t tok = (uint64_t)blockIdx.y;
    if (tok >= n_tok) return;
    const int has0 = row < out0_dim;
    const int has1 = row < out1_dim;
    if (!has0 && !has1) return;

    const unsigned char *wr0 = has0 ? w0 + row * blocks * 34u : NULL;
    const unsigned char *wr1 = has1 ? w1 + row * blocks * 34u : NULL;
    const int8_t *xqr = xq + tok * blocks * 32u;
    const float *xsr = xscale + tok * blocks;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const int8_t *xqb = xqr + b * 32u;
        const float xs = xsr[b];
        if (has0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34u);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34u + 2u);
            const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc0 += __half2float(*scale_h) * xs * (float)dot;
        }
        if (has1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34u);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34u + 2u);
            const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc1 += __half2float(*scale_h) * xs * (float)dot;
        }
    }

    __shared__ float partial0[256];
    __shared__ float partial1[256];
    partial0[threadIdx.x] = acc0;
    partial1[threadIdx.x] = acc1;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial0[threadIdx.x] += partial0[threadIdx.x + stride];
            partial1[threadIdx.x] += partial1[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (has0) out0[tok * out0_dim + row] = partial0[0];
        if (has1) out1[tok * out1_dim + row] = partial1[0];
    }
}

__global__ static void matmul_q8_0_pair_preq_batch_tok2_exact_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint64_t tok0 = (uint64_t)blockIdx.y * 2u;
    if (tok0 >= n_tok) return;
    const int has0 = row < out0_dim;
    const int has1 = row < out1_dim;
    if (!has0 && !has1) return;
    const int valid1 = tok0 + 1u < n_tok;

    const unsigned char *wr0 = has0 ? w0 + row * blocks * 34u : NULL;
    const unsigned char *wr1 = has1 ? w1 + row * blocks * 34u : NULL;
    const int8_t *xqr0 = xq + tok0 * blocks * 32u;
    const int8_t *xqr1 = valid1 ? xqr0 + blocks * 32u : xqr0;
    const float *xsr0 = xscale + tok0 * blocks;
    const float *xsr1 = valid1 ? xsr0 + blocks : xsr0;
    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const int8_t *xqb0 = xqr0 + b * 32u;
        const int8_t *xqb1 = xqr1 + b * 32u;
        const float xs0 = xsr0[b];
        const float xs1 = valid1 ? xsr1[b] : 0.0f;
        if (has0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34u);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34u + 2u);
            const int dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            int dot1 = 0;
            if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
            const float ws = __half2float(*scale_h);
            acc00 += ws * xs0 * (float)dot0;
            if (valid1) acc01 += ws * xs1 * (float)dot1;
        }
        if (has1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34u);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34u + 2u);
            const int dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            int dot1 = 0;
            if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
            const float ws = __half2float(*scale_h);
            acc10 += ws * xs0 * (float)dot0;
            if (valid1) acc11 += ws * xs1 * (float)dot1;
        }
    }

    __shared__ float partial00[256];
    __shared__ float partial01[256];
    __shared__ float partial10[256];
    __shared__ float partial11[256];
    partial00[threadIdx.x] = acc00;
    partial01[threadIdx.x] = acc01;
    partial10[threadIdx.x] = acc10;
    partial11[threadIdx.x] = acc11;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial00[threadIdx.x] += partial00[threadIdx.x + stride];
            partial01[threadIdx.x] += partial01[threadIdx.x + stride];
            partial10[threadIdx.x] += partial10[threadIdx.x + stride];
            partial11[threadIdx.x] += partial11[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (has0) {
            out0[tok0 * out0_dim + row] = partial00[0];
            if (valid1) out0[(tok0 + 1u) * out0_dim + row] = partial01[0];
        }
        if (has1) {
            out1[tok0 * out1_dim + row] = partial10[0];
            if (valid1) out1[(tok0 + 1u) * out1_dim + row] = partial11[0];
        }
    }
}

__device__ static float moe_owned_packed_combine_row(
        const float *home_slots,
        const float *peer_packed,
        const int32_t *selected,
        uint32_t row,
        uint32_t out_dim,
        uint32_t expert_split);

__global__ static void matmul_q8_0_hc_expand_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *block_add2,
        const float *owned_home_slots,
        const float *owned_peer_packed,
        const int32_t *owned_selected,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add,
        int has_add2,
        int has_owned_slots,
        uint32_t owned_expert_split,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_owned_slots) {
            const float routed = moe_owned_packed_combine_row(
                    owned_home_slots,
                    owned_peer_packed,
                    owned_selected,
                    d,
                    (uint32_t)out_dim,
                    owned_expert_split);
            block_v = __fadd_rn(block_v, routed);
        } else if (has_add) {
            float add_v = block_add[d];
            if (has_add2) add_v += block_add2[d];
            block_v += add_v;
        }
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void matmul_q8_0_kslice_hc_expand_add_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t slice_dim,
        uint64_t out_dim,
        uint64_t full_blocks,
        uint64_t block_start,
        uint64_t slice_blocks,
        uint32_t n_embd,
        uint32_t n_hc,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * full_blocks * 34u + block_start * 34u;
    float acc = 0.0f;
    for (uint64_t b = lane; b < slice_blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = slice_dim - i0 < 32u ? slice_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb = xq + b * 32u;
        const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        const float block_v = acc + block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void matmul_q8_0_preq_batch_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

__global__ static void matmul_q8_0_preq_batch_warp8_tok2_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const unsigned char *wr = w + row * blocks * 34u;
    const int8_t *xqr0 = xq;
    const int8_t *xqr1 = xq + blocks * 32u;
    const float *xsr0 = xscale;
    const float *xsr1 = xscale + blocks;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb0 = xqr0 + b * 32u;
        const int8_t *xqb1 = xqr1 + b * 32u;
        int dot0 = 0;
        int dot1 = 0;
        if (use_dp4a && bn == 32u) {
#pragma unroll
            for (uint32_t i = 0; i < 32u; i += 4u) {
                const int32_t w4 = load_i8x4_i32_unaligned(qs + i);
                dot0 = __dp4a(w4, load_i8x4_i32_aligned(xqb0 + i), dot0);
                dot1 = __dp4a(w4, load_i8x4_i32_aligned(xqb1 + i), dot1);
            }
        } else {
            dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
        }
        const float ws = __half2float(*scale_h);
        acc0 += ws * xsr0[b] * (float)dot0;
        acc1 += ws * xsr1[b] * (float)dot1;
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        out[row] = acc0;
        out[out_dim + row] = acc1;
    }
}

__global__ static void matmul_q8_0_preq_batch_warp8_tok4_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok0 = (uint64_t)blockIdx.y * 4u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok0 >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr0 = xq + tok0 * blocks * 32;
    const int8_t *xqr1 = xqr0 + blocks * 32;
    const int8_t *xqr2 = xqr1 + blocks * 32;
    const int8_t *xqr3 = xqr2 + blocks * 32;
    const float *xsr0 = xscale + tok0 * blocks;
    const float *xsr1 = xsr0 + blocks;
    const float *xsr2 = xsr1 + blocks;
    const float *xsr3 = xsr2 + blocks;
    const int valid1 = tok0 + 1u < n_tok;
    const int valid2 = tok0 + 2u < n_tok;
    const int valid3 = tok0 + 3u < n_tok;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb0 = xqr0 + b * 32;
        const int8_t *xqb1 = xqr1 + b * 32;
        const int8_t *xqb2 = xqr2 + b * 32;
        const int8_t *xqb3 = xqr3 + b * 32;
        int dot0 = 0;
        int dot1 = 0;
        int dot2 = 0;
        int dot3 = 0;
        if (use_dp4a && bn == 32u) {
#pragma unroll
            for (uint32_t i = 0; i < 32u; i += 4u) {
                const int32_t w4 = load_i8x4_i32_unaligned(qs + i);
                dot0 = __dp4a(w4, load_i8x4_i32_aligned(xqb0 + i), dot0);
                if (valid1) dot1 = __dp4a(w4, load_i8x4_i32_aligned(xqb1 + i), dot1);
                if (valid2) dot2 = __dp4a(w4, load_i8x4_i32_aligned(xqb2 + i), dot2);
                if (valid3) dot3 = __dp4a(w4, load_i8x4_i32_aligned(xqb3 + i), dot3);
            }
        } else {
            dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
            if (valid2) dot2 = dot_i8_block(qs, xqb2, bn, use_dp4a);
            if (valid3) dot3 = dot_i8_block(qs, xqb3, bn, use_dp4a);
        }
        const float ws = __half2float(*scale_h);
        acc0 += ws * xsr0[b] * (float)dot0;
        if (valid1) acc1 += ws * xsr1[b] * (float)dot1;
        if (valid2) acc2 += ws * xsr2[b] * (float)dot2;
        if (valid3) acc3 += ws * xsr3[b] * (float)dot3;
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    acc2 = warp_sum_f32(acc2);
    acc3 = warp_sum_f32(acc3);
    if (lane == 0) {
        out[tok0 * out_dim + row] = acc0;
        if (valid1) out[(tok0 + 1u) * out_dim + row] = acc1;
        if (valid2) out[(tok0 + 2u) * out_dim + row] = acc2;
        if (valid3) out[(tok0 + 3u) * out_dim + row] = acc3;
    }
}

__global__ static void matmul_q8_0_preq_batch_warp8_tok8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok0 = (uint64_t)blockIdx.y * 8u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok0 >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const uint64_t xq_stride = blocks * 32u;
    const int8_t *xqr0 = xq + tok0 * xq_stride;
    const int valid1 = tok0 + 1u < n_tok;
    const int valid2 = tok0 + 2u < n_tok;
    const int valid3 = tok0 + 3u < n_tok;
    const int valid4 = tok0 + 4u < n_tok;
    const int valid5 = tok0 + 5u < n_tok;
    const int valid6 = tok0 + 6u < n_tok;
    const int valid7 = tok0 + 7u < n_tok;
    const int8_t *xqr1 = valid1 ? xqr0 + xq_stride : xqr0;
    const int8_t *xqr2 = valid2 ? xqr1 + xq_stride : xqr0;
    const int8_t *xqr3 = valid3 ? xqr2 + xq_stride : xqr0;
    const int8_t *xqr4 = valid4 ? xqr3 + xq_stride : xqr0;
    const int8_t *xqr5 = valid5 ? xqr4 + xq_stride : xqr0;
    const int8_t *xqr6 = valid6 ? xqr5 + xq_stride : xqr0;
    const int8_t *xqr7 = valid7 ? xqr6 + xq_stride : xqr0;
    const float *xsr0 = xscale + tok0 * blocks;
    const float *xsr1 = valid1 ? xsr0 + blocks : xsr0;
    const float *xsr2 = valid2 ? xsr1 + blocks : xsr0;
    const float *xsr3 = valid3 ? xsr2 + blocks : xsr0;
    const float *xsr4 = valid4 ? xsr3 + blocks : xsr0;
    const float *xsr5 = valid5 ? xsr4 + blocks : xsr0;
    const float *xsr6 = valid6 ? xsr5 + blocks : xsr0;
    const float *xsr7 = valid7 ? xsr6 + blocks : xsr0;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb0 = xqr0 + b * 32;
        const int8_t *xqb1 = xqr1 + b * 32;
        const int8_t *xqb2 = xqr2 + b * 32;
        const int8_t *xqb3 = xqr3 + b * 32;
        const int8_t *xqb4 = xqr4 + b * 32;
        const int8_t *xqb5 = xqr5 + b * 32;
        const int8_t *xqb6 = xqr6 + b * 32;
        const int8_t *xqb7 = xqr7 + b * 32;
        int dot0 = 0;
        int dot1 = 0;
        int dot2 = 0;
        int dot3 = 0;
        int dot4 = 0;
        int dot5 = 0;
        int dot6 = 0;
        int dot7 = 0;
        if (use_dp4a && bn == 32u) {
#pragma unroll
            for (uint32_t i = 0; i < 32u; i += 4u) {
                const int32_t w4 = load_i8x4_i32_unaligned(qs + i);
                dot0 = __dp4a(w4, load_i8x4_i32_aligned(xqb0 + i), dot0);
                if (valid1) dot1 = __dp4a(w4, load_i8x4_i32_aligned(xqb1 + i), dot1);
                if (valid2) dot2 = __dp4a(w4, load_i8x4_i32_aligned(xqb2 + i), dot2);
                if (valid3) dot3 = __dp4a(w4, load_i8x4_i32_aligned(xqb3 + i), dot3);
                if (valid4) dot4 = __dp4a(w4, load_i8x4_i32_aligned(xqb4 + i), dot4);
                if (valid5) dot5 = __dp4a(w4, load_i8x4_i32_aligned(xqb5 + i), dot5);
                if (valid6) dot6 = __dp4a(w4, load_i8x4_i32_aligned(xqb6 + i), dot6);
                if (valid7) dot7 = __dp4a(w4, load_i8x4_i32_aligned(xqb7 + i), dot7);
            }
        } else {
            dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
            if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
            if (valid2) dot2 = dot_i8_block(qs, xqb2, bn, use_dp4a);
            if (valid3) dot3 = dot_i8_block(qs, xqb3, bn, use_dp4a);
            if (valid4) dot4 = dot_i8_block(qs, xqb4, bn, use_dp4a);
            if (valid5) dot5 = dot_i8_block(qs, xqb5, bn, use_dp4a);
            if (valid6) dot6 = dot_i8_block(qs, xqb6, bn, use_dp4a);
            if (valid7) dot7 = dot_i8_block(qs, xqb7, bn, use_dp4a);
        }
        const float ws = __half2float(*scale_h);
        acc0 += ws * xsr0[b] * (float)dot0;
        if (valid1) acc1 += ws * xsr1[b] * (float)dot1;
        if (valid2) acc2 += ws * xsr2[b] * (float)dot2;
        if (valid3) acc3 += ws * xsr3[b] * (float)dot3;
        if (valid4) acc4 += ws * xsr4[b] * (float)dot4;
        if (valid5) acc5 += ws * xsr5[b] * (float)dot5;
        if (valid6) acc6 += ws * xsr6[b] * (float)dot6;
        if (valid7) acc7 += ws * xsr7[b] * (float)dot7;
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    acc2 = warp_sum_f32(acc2);
    acc3 = warp_sum_f32(acc3);
    acc4 = warp_sum_f32(acc4);
    acc5 = warp_sum_f32(acc5);
    acc6 = warp_sum_f32(acc6);
    acc7 = warp_sum_f32(acc7);
    if (lane == 0) {
        out[tok0 * out_dim + row] = acc0;
        if (valid1) out[(tok0 + 1u) * out_dim + row] = acc1;
        if (valid2) out[(tok0 + 2u) * out_dim + row] = acc2;
        if (valid3) out[(tok0 + 3u) * out_dim + row] = acc3;
        if (valid4) out[(tok0 + 4u) * out_dim + row] = acc4;
        if (valid5) out[(tok0 + 5u) * out_dim + row] = acc5;
        if (valid6) out[(tok0 + 6u) * out_dim + row] = acc6;
        if (valid7) out[(tok0 + 7u) * out_dim + row] = acc7;
    }
}

__global__ static void matmul_q8_0_preq_batch_tok2_exact_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint64_t tok0 = (uint64_t)blockIdx.y * 2u;
    if (row >= out_dim || tok0 >= n_tok) return;
    const int valid1 = tok0 + 1u < n_tok;
    const unsigned char *wr = w + row * blocks * 34u;
    const int8_t *xqr0 = xq + tok0 * blocks * 32u;
    const int8_t *xqr1 = valid1 ? xqr0 + blocks * 32u : xqr0;
    const float *xsr0 = xscale + tok0 * blocks;
    const float *xsr1 = valid1 ? xsr0 + blocks : xsr0;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const __half *scale_h = (const __half *)(wr + b * 34u);
        const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
        const int8_t *xqb0 = xqr0 + b * 32u;
        const int8_t *xqb1 = xqr1 + b * 32u;
        const int dot0 = dot_i8_block(qs, xqb0, bn, use_dp4a);
        int dot1 = 0;
        if (valid1) dot1 = dot_i8_block(qs, xqb1, bn, use_dp4a);
        const float ws = __half2float(*scale_h);
        acc0 += ws * xsr0[b] * (float)dot0;
        if (valid1) acc1 += ws * xsr1[b] * (float)dot1;
    }

    __shared__ float partial0[256];
    __shared__ float partial1[256];
    partial0[threadIdx.x] = acc0;
    partial1[threadIdx.x] = acc1;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial0[threadIdx.x] += partial0[threadIdx.x + stride];
            partial1[threadIdx.x] += partial1[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[tok0 * out_dim + row] = partial0[0];
        if (valid1) out[(tok0 + 1u) * out_dim + row] = partial1[0];
    }
}


/* ---- INT8 tensor-core exact Q8_0 batch matmul --------------------------
 * Bit-identical replacement for the exact tok2/warp8-family batched Q8_0
 * kernels. Each output element's reduction is the reference's strided
 * halving tree over T slots (T = reduction width: 32 for the warp kernels,
 * cuda_q8_exact_threads(blocks) for the exact kernels; slots >= blocks hold
 * +0.0f). The kernel decomposes that tree as: 32 streams at stride T/32
 * whose 32 terms per outer step j combine via an adjacent-pairwise static
 * register stack taken in bit-reversed stream order (== the top five strided
 * tree levels), plus per-(j&3) sequential accumulators and a fixed tail for
 * the remaining levels. Fuzz-verified bitwise against both reference
 * kernels across shapes, including blocks < T and ragged out_dim/n_tok.
 * Rollback: DS4_CUDA_NO_Q8_MMA=1. */
__device__ __forceinline__ static uint32_t ldu32_unaligned(const uint8_t *p) {
    const uintptr_t addr = (uintptr_t)p;
    const uint32_t *base = (const uint32_t *)(addr & ~(uintptr_t)3);
    const uint32_t lo = base[0];
    const uint32_t hi = base[1];
    return __funnelshift_r(lo, hi, (uint32_t)(addr & 3u) * 8u);
}

__device__ __forceinline__ static void mma_m16n8k32_s8(
        int32_t &c0, int32_t &c1, int32_t &c2, int32_t &c3,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
                 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
#else
    (void)a0;(void)a1;(void)a2;(void)a3;(void)b0;(void)b1;(void)c0;(void)c1;(void)c2;(void)c3;
#endif
}

__device__ __forceinline__ static uint32_t bitrev5(uint32_t i) {
    return ((i & 1u) << 4) | ((i & 2u) << 2) | (i & 4u) | ((i & 8u) >> 2) | ((i & 16u) >> 4);
}

template <uint32_t T>
__global__ static void matmul_q8_0_mma_exact_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        uint64_t a_stride_blocks, /* activation row stride in blocks (>= blocks) */
        uint64_t out_stride) {    /* output token stride in floats (>= out_dim) */
    extern __shared__ unsigned char q8mma_sh[];
    __half *sh_ws = (__half *)q8mma_sh;                    /* 64 rows x blocks */
    float *sh_xs = (float *)(q8mma_sh + 64u * blocks * 2u); /* 16 toks x blocks */
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint64_t row_base = (uint64_t)blockIdx.x * 64u;
    const uint64_t tok_base = (uint64_t)blockIdx.y * 16u;

    /* stage weight scales (64 rows) and activation scales (16 tokens) */
    for (uint32_t idx = threadIdx.x; idx < 64u * (uint32_t)blocks; idx += blockDim.x) {
        const uint32_t rl = idx / (uint32_t)blocks;
        const uint32_t b = idx - rl * (uint32_t)blocks;
        uint64_t row = row_base + rl;
        if (row >= out_dim) row = out_dim - 1u;
        sh_ws[idx] = *(const __half *)(w + row * blocks * 34u + (uint64_t)b * 34u);
    }
    for (uint32_t idx = threadIdx.x; idx < 16u * (uint32_t)blocks; idx += blockDim.x) {
        const uint32_t tl = idx / (uint32_t)blocks;
        const uint32_t b = idx - tl * (uint32_t)blocks;
        const uint64_t tok = tok_base + tl;
        sh_xs[idx] = tok < n_tok ? xscale[tok * a_stride_blocks + b] : 0.0f;
    }
    __syncthreads();

    const uint64_t row0 = row_base + (uint64_t)warp * 8u;
    /* thread's C elements: rows n0,n0+1; tokens mt0, mt0+8 */
    const uint32_t n0 = (lane & 3u) * 2u;
    const uint32_t mt0 = lane >> 2u;
    const uint64_t tokA = tok_base + mt0;
    const uint64_t tokB = tok_base + mt0 + 8u;
    /* A source rows for loads (fragment layout): rows lane>>2 and (lane>>2)+8 */
    const uint64_t a_tok_lo = tok_base + (lane >> 2u);
    const uint64_t a_tok_hi = a_tok_lo + 8u;
    const int8_t *aq_lo = xq + (a_tok_lo < n_tok ? a_tok_lo : 0u) * a_stride_blocks * 32u;
    const int8_t *aq_hi = xq + (a_tok_hi < n_tok ? a_tok_hi : 0u) * a_stride_blocks * 32u;
    const bool a_lo_ok = a_tok_lo < n_tok;
    const bool a_hi_ok = a_tok_hi < n_tok;
    /* B source row for loads: row lane>>2 within the warp tile */
    uint64_t b_row = row0 + (lane >> 2u);
    if (b_row >= out_dim) b_row = out_dim - 1u;
    const unsigned char *b_wr = w + b_row * blocks * 34u;

    /* per-element (4) x per-(j&3) accumulators */
    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;

    const uint32_t stride = T / 32u;
    const uint32_t rl_ws0 = warp * 8u + n0;      /* local row for ws of element cols */
    const uint32_t tl_xsA = mt0;                 /* local token rows for xs */
    const uint32_t tl_xsB = mt0 + 8u;

    for (uint32_t j = 0; j < stride; j++) {
        /* adjacent-pairwise static stack over 32 terms in bitrev5 m order */
        float s0e0 = 0, s1e0 = 0, s2e0 = 0, s3e0 = 0, s4e0 = 0;
        float s0e1 = 0, s1e1 = 0, s2e1 = 0, s3e1 = 0, s4e1 = 0;
        float s0e2 = 0, s1e2 = 0, s2e2 = 0, s3e2 = 0, s4e2 = 0;
        float s0e3 = 0, s1e3 = 0, s2e3 = 0, s3e3 = 0, s4e3 = 0;
#pragma unroll
        for (uint32_t i = 0; i < 32u; i++) {
            const uint32_t m = bitrev5(i);
            const uint32_t s = j + m * stride; /* slot index */
            float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
            /* slot s sums blocks {s + k*T} sequentially (multi-term when
             * blocks > T, exactly like the per-lane strided walk). */
            for (uint32_t b = s; b < blocks; b += T) {
                const uint32_t koff = (lane & 3u) * 4u;
                const int8_t *ablk_lo = aq_lo + b * 32u;
                const int8_t *ablk_hi = aq_hi + b * 32u;
                const uint32_t a0 = a_lo_ok ? *(const uint32_t *)(ablk_lo + koff) : 0u;
                const uint32_t a1 = a_hi_ok ? *(const uint32_t *)(ablk_hi + koff) : 0u;
                const uint32_t a2 = a_lo_ok ? *(const uint32_t *)(ablk_lo + 16u + koff) : 0u;
                const uint32_t a3 = a_hi_ok ? *(const uint32_t *)(ablk_hi + 16u + koff) : 0u;
                const uint8_t *bq = (const uint8_t *)(b_wr + (uint64_t)b * 34u + 2u);
                const uint32_t b0 = ldu32_unaligned(bq + koff);
                const uint32_t b1 = ldu32_unaligned(bq + 16u + koff);
                int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                mma_m16n8k32_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
                /* term = ws * xs * dot, same expression as reference */
                const float ws0 = __half2float(sh_ws[rl_ws0 * (uint32_t)blocks + b]);
                const float ws1 = __half2float(sh_ws[(rl_ws0 + 1u) * (uint32_t)blocks + b]);
                const float xsA = sh_xs[tl_xsA * (uint32_t)blocks + b];
                const float xsB = sh_xs[tl_xsB * (uint32_t)blocks + b];
                t0 += ws0 * xsA * (float)c0;
                t1 += ws1 * xsA * (float)c1;
                t2 += ws0 * xsB * (float)c2;
                t3 += ws1 * xsB * (float)c3;
            }
            /* static adjacent stack push (compile-time resolved) */
            if ((i & 1u) == 0u) { s0e0 = t0; s0e1 = t1; s0e2 = t2; s0e3 = t3; }
            else {
                t0 = s0e0 + t0; t1 = s0e1 + t1; t2 = s0e2 + t2; t3 = s0e3 + t3;
                if ((i & 2u) == 0u) { s1e0 = t0; s1e1 = t1; s1e2 = t2; s1e3 = t3; }
                else {
                    t0 = s1e0 + t0; t1 = s1e1 + t1; t2 = s1e2 + t2; t3 = s1e3 + t3;
                    if ((i & 4u) == 0u) { s2e0 = t0; s2e1 = t1; s2e2 = t2; s2e3 = t3; }
                    else {
                        t0 = s2e0 + t0; t1 = s2e1 + t1; t2 = s2e2 + t2; t3 = s2e3 + t3;
                        if ((i & 8u) == 0u) { s3e0 = t0; s3e1 = t1; s3e2 = t2; s3e3 = t3; }
                        else {
                            t0 = s3e0 + t0; t1 = s3e1 + t1; t2 = s3e2 + t2; t3 = s3e3 + t3;
                            if ((i & 16u) == 0u) { s4e0 = t0; s4e1 = t1; s4e2 = t2; s4e3 = t3; }
                            else {
                                t0 = s4e0 + t0; t1 = s4e1 + t1; t2 = s4e2 + t2; t3 = s4e3 + t3;
                                /* i == 31: t is the finished x_j */
                                switch (j & 3u) {
                                case 0u: acc00 += t0; acc10 += t1; acc20 += t2; acc30 += t3; break;
                                case 1u: acc01 += t0; acc11 += t1; acc21 += t2; acc31 += t3; break;
                                case 2u: acc02 += t0; acc12 += t1; acc22 += t2; acc32 += t3; break;
                                default: acc03 += t0; acc13 += t1; acc23 += t2; acc33 += t3; break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    /* tail combine per T */
    float r0, r1, r2, r3;
    if (T == 32u) {
        r0 = acc00; r1 = acc10; r2 = acc20; r3 = acc30;
    } else if (T == 64u) {
        r0 = acc00 + acc01; r1 = acc10 + acc11; r2 = acc20 + acc21; r3 = acc30 + acc31;
    } else {
        r0 = (acc00 + acc02) + (acc01 + acc03);
        r1 = (acc10 + acc12) + (acc11 + acc13);
        r2 = (acc20 + acc22) + (acc21 + acc23);
        r3 = (acc30 + acc32) + (acc31 + acc33);
    }
    /* writes */
    const uint64_t rowa = row0 + n0;
    const uint64_t rowb = rowa + 1u;
    if (tokA < n_tok) {
        if (rowa < out_dim) out[tokA * out_stride + rowa] = r0;
        if (rowb < out_dim) out[tokA * out_stride + rowb] = r1;
    }
    if (tokB < n_tok) {
        if (rowa < out_dim) out[tokB * out_stride + rowa] = r2;
        if (rowb < out_dim) out[tokB * out_stride + rowb] = r3;
    }
}

static int cuda_q4_mma_ok(void);
static int cuda_q8_mma_attr_ready[DS4_MAX_GPUS][4];
static int cuda_q8_mma_try_launch(
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


__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = __hmul(scale, __float2half((float)q));
}

__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const float scale = __half2float(*(const __half *)blk);
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = scale * (float)q;
}

__global__ static void grouped_q8_0_a_preq_warp8_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;

    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * blocks * 32;
    const float *xsr = xscale + xrow * blocks;
    float acc = 0.0f;

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

__global__ static void grouped_q8_0_a_preq_warp8_tok2_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a) {
    const uint32_t tid_in_tok = threadIdx.x & 255u;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (tid_in_tok >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y * 2u + (threadIdx.x >> 8u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;

    float acc = 0.0f;
    if (row < low_dim && tok < n_tokens) {
        const uint64_t group = row / rank;
        const uint64_t row_in_group = row - group * rank;
        const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34u;
        const uint64_t xrow = tok * (uint64_t)n_groups + group;
        const int8_t *xqr = xq + xrow * blocks * 32u;
        const float *xsr = xscale + xrow * blocks;

        for (uint64_t b = lane; b < blocks; b += 32u) {
            const uint64_t i0 = b * 32u;
            const uint64_t bn = group_dim - i0 < 32u ? group_dim - i0 : 32u;
            const __half *scale_h = (const __half *)(wr + b * 34u);
            const int8_t *qs = (const int8_t *)(wr + b * 34u + 2u);
            const int8_t *xqb = xqr + b * 32u;
            const int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc += __half2float(*scale_h) * xsr[b] * (float)dot;
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0 && row < low_dim && tok < n_tokens) {
        low[tok * low_dim + row] = acc;
    }
}

__global__ static void rms_norm_plain_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale;
    }
}

