/* Async D2D copy queued on the destination device's default stream —
 * ordering against the producer comes from the caller's fence; the CPU
 * does not block (unlike ds4_gpu_tensor_copy's sync cudaMemcpy). */
extern "C" int ds4_gpu_tensor_copy_async(ds4_gpu_tensor *dst,
                                         const ds4_gpu_tensor *src,
                                         uint64_t bytes) {
    if (!dst || !src || bytes > dst->bytes || bytes > src->bytes) return 0;
    if (bytes == 0) return 1;
    return cuda_ok(cudaMemcpyAsync(dst->ptr, src->ptr, (size_t)bytes,
                                   cudaMemcpyDeviceToDevice, 0),
                   "tensor copy async");
}

extern "C" void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *t) {
    if (!t) return;
    int d = ds4_tensor_device_idx(t);
    if (t->owner && t->ptr) {
        WITH_DEVICE(g_gpu[d].device_id) {
            (void)cudaFree(t->ptr);
        }
    }
    t->ptr = NULL;
    t->bytes = 0;
    t->owner = 0;
}

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

/* Heap-allocated tensor on a specific logical tier.
 *
 * Mirrors the legacy ds4_gpu_tensor_alloc ABI (returns ds4_gpu_tensor *)
 * with an explicit tier argument. Internally calls
 * ds4_gpu_tensor_alloc_on on a freshly malloc'd struct.
 *
 * The legacy ds4_gpu_tensor_alloc(bytes) (above) calls
 * ds4_gpu_tensor_alloc_on(t, 0, bytes); ds4_gpu_tensor_alloc_ptr_on(0,
 * bytes) is byte-equivalent. Single-tier callers MAY remain on the
 * legacy 1-arg helper; new multi-tier callers in ds4.c use _ptr_on. */
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

/* Heap-allocated managed-memory tensor on a specific logical tier.
 * Differs from ds4_gpu_tensor_alloc_managed only in stamping
 * tier instead of 0. Used by the per-layer KV cache when tier !=0.
 *
 * Managed-memory paging behavior: cudaMallocManaged pages between
 * devices on first-touch. In a single-tier pipeline the page lives on
 * tier 0; in a multi-tier pipeline the layer's kernels run on the
 * layer's tier so the page lives there after first-touch and stays
 * unless another device touches it. Stamping tier matches the home
 * device for free-time accounting. */
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

extern "C" int ds4_gpu_tensor_device(const ds4_gpu_tensor *t) {
    return t ? t->device_id : -1;
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
    /* Same-device fast path; for cross-device, callers should use
     * ds4_gpu_tensor_copy_xdev. We still tolerate cross-device callers
     * here by routing to D2D copy on the destination's device. */
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

__global__ static void moe_handoff_pack_kernel(
        unsigned char *packed,
        const float *ffn_norm,
        const int32_t *selected,
        const float *weights,
        uint32_t n_embd,
        uint32_t n_expert) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float *packed_norm = (float *)packed;
    int32_t *packed_selected = (int32_t *)(packed + (uint64_t)n_embd * sizeof(float));
    float *packed_weights = (float *)(packed + (uint64_t)n_embd * sizeof(float) +
                                      (uint64_t)n_expert * sizeof(int32_t));
    if (i < n_embd) packed_norm[i] = ffn_norm[i];
    if (i < n_expert) {
        packed_selected[i] = selected[i];
        packed_weights[i] = weights[i];
    }
}

extern "C" int ds4_gpu_moe_handoff_pack_tensor(
        ds4_gpu_tensor       *packed,
        const ds4_gpu_tensor *ffn_norm,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_embd,
        uint32_t              n_expert) {
    if (!packed || !ffn_norm || !selected || !weights ||
        n_embd == 0 || n_expert == 0) {
        return 0;
    }
    const uint64_t bytes = (uint64_t)n_embd * sizeof(float) +
                           (uint64_t)n_expert * sizeof(int32_t) +
                           (uint64_t)n_expert * sizeof(float);
    if (packed->bytes < bytes ||
        ffn_norm->bytes < (uint64_t)n_embd * sizeof(float) ||
        selected->bytes < (uint64_t)n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_expert * sizeof(float)) {
        return 0;
    }
    const uint32_t n = n_embd > n_expert ? n_embd : n_expert;
    moe_handoff_pack_kernel<<<(n + 255u) / 256u, 256>>>(
            (unsigned char *)packed->ptr,
            (const float *)ffn_norm->ptr,
            (const int32_t *)selected->ptr,
            (const float *)weights->ptr,
            n_embd,
            n_expert);
    return cuda_ok(cudaGetLastError(), "moe handoff pack launch");
}

#if 0
/* Cross-device copy primitive. Path selection (highest priority first):
 *   DS4_FORCE_HOST_BOUNCE=1  -> always pinned-host bounce
 *   DS4_FORCE_CUDA_PEER=1    -> always cudaMemcpyPeerAsync (manual-testing
 *                               override; bypasses g_gpu_peer_ok)
 *   otherwise                -> peer if validation passed at init, else
 *                               pinned-host bounce. */
static int ds4_gpu_tensor_copy_xdev_impl(ds4_gpu_tensor *dst,
                                         const ds4_gpu_tensor *src,
                                         uint64_t bytes,
                                         bool order_dst_before_write) {
    if (!dst || !src) return 0;
    if (bytes == 0) return 1;
    if (bytes > dst->bytes || bytes > src->bytes) return 0;
    int sd = ds4_tensor_device_idx(src);
    int dd = ds4_tensor_device_idx(dst);

    /* Same-device fast path. */
    if (sd == dd) {
        int ok = 0;
        WITH_DEVICE(g_gpu[sd].device_id) {
            cudaStream_t s = (cudaStream_t)g_gpu[sd].stream;
            ok = cuda_ok(cudaMemcpyAsync(dst->ptr, src->ptr, bytes,
                                         cudaMemcpyDeviceToDevice, s),
                          "xdev same-device copy");
            if (ok && g_xdev_sync_debug) {
                ok = cuda_ok(cudaStreamSynchronize(s), "xdev same-device sync");
            }
        }
        return ok;
    }

    int peer = g_gpu_peer_ok[sd][dd];
    if (g_xdev_force_cuda_peer)   peer = 1;
    if (g_xdev_force_host_bounce) peer = 0;

    if (peer) {
        int ok = 0;
        if (order_dst_before_write) {
            WITH_DEVICE(g_gpu[dd].device_id) {
                cudaStream_t s2 = (cudaStream_t)g_gpu[dd].stream;
                cudaEvent_t  e2 = (cudaEvent_t)g_gpu[dd].boundary_event;
                ok = cuda_ok(cudaEventRecord(e2, s2), "peer dst-ready event record");
            }
            if (!ok) return 0;
        } else {
            ok = 1;
        }
        WITH_DEVICE(g_gpu[sd].device_id) {
            cudaStream_t s = (cudaStream_t)g_gpu[sd].stream;
            cudaEvent_t  e = (cudaEvent_t)g_gpu[sd].boundary_event;
            if (order_dst_before_write) {
                ok = cuda_ok(cudaStreamWaitEvent(s, (cudaEvent_t)g_gpu[dd].boundary_event, 0),
                             "peer src wait dst-ready");
            }
            if (ok) ok = cuda_ok(cudaMemcpyPeerAsync(
                            dst->ptr, g_gpu[dd].device_id,
                            src->ptr, g_gpu[sd].device_id,
                            bytes, s),
                          "peer copy");
            if (ok) ok = cuda_ok(cudaEventRecord(e, s), "peer event record");
        }
        if (!ok) return 0;
        WITH_DEVICE(g_gpu[dd].device_id) {
            cudaStream_t s2 = (cudaStream_t)g_gpu[dd].stream;
            (void)cudaStreamWaitEvent(s2, (cudaEvent_t)g_gpu[sd].boundary_event, 0);
            if (g_xdev_sync_debug) {
                ok = cuda_ok(cudaStreamSynchronize(s2), "peer dst sync");
            }
        }
        return ok;
    }

    /* Per-pair pinned-host bounce buffer. */
    if (g_xdev_bounce_bytes[sd][dd] < bytes) {
        if (g_xdev_bounce[sd][dd]) (void)cudaFreeHost(g_xdev_bounce[sd][dd]);
        if (!cuda_ok(cudaMallocHost(&g_xdev_bounce[sd][dd], (size_t)bytes),
                     "bounce alloc")) return 0;
        g_xdev_bounce_bytes[sd][dd] = bytes;
    }
    int ok = 0;
    WITH_DEVICE(g_gpu[sd].device_id) {
        cudaStream_t s = (cudaStream_t)g_gpu[sd].stream;
        cudaEvent_t  e = (cudaEvent_t)g_gpu[sd].boundary_event;
        ok = cuda_ok(cudaMemcpyAsync(g_xdev_bounce[sd][dd], src->ptr, bytes,
                                     cudaMemcpyDeviceToHost, s),
                      "bounce d2h");
        if (ok) ok = cuda_ok(cudaEventRecord(e, s), "bounce event record");
    }
    if (!ok) return 0;
    WITH_DEVICE(g_gpu[dd].device_id) {
        cudaStream_t s2 = (cudaStream_t)g_gpu[dd].stream;
        (void)cudaStreamWaitEvent(s2, (cudaEvent_t)g_gpu[sd].boundary_event, 0);
        ok = cuda_ok(cudaMemcpyAsync(dst->ptr, g_xdev_bounce[sd][dd], bytes,
                                     cudaMemcpyHostToDevice, s2),
                      "bounce h2d");
        if (ok) ok = cuda_ok(cudaStreamSynchronize(s2), "bounce dst sync");
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst,
                                        const ds4_gpu_tensor *src,
                                        uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev_impl(dst, src, bytes, false);
}

static int ds4_gpu_tensor_copy_xdev_default_impl(ds4_gpu_tensor *dst,
                                                  const ds4_gpu_tensor *src,
                                                  uint64_t bytes) {
    if (!dst || !src || bytes > dst->bytes || bytes > src->bytes) return 0;
    if (bytes == 0u) return 1;
    const int sd = ds4_tensor_device_idx(src);
    const int dd = ds4_tensor_device_idx(dst);
    if (sd == dd) {
        int ok = 0;
        WITH_DEVICE(g_gpu[sd].device_id) {
            ok = cuda_ok(cudaMemcpyAsync(dst->ptr, src->ptr, (size_t)bytes,
                                         cudaMemcpyDeviceToDevice, 0),
                         "default-stream same-device copy");
        }
        return ok;
    }

    int peer = g_gpu_peer_ok[sd][dd];
    if (g_xdev_force_cuda_peer) peer = 1;
    if (g_xdev_force_host_bounce) peer = 0;
    if (peer) {
        int ok = 0;
        WITH_DEVICE(g_gpu[sd].device_id) {
            ok = cuda_ok(cudaMemcpyPeerAsync(
                             dst->ptr, g_gpu[dd].device_id,
                             src->ptr, g_gpu[sd].device_id,
                             (size_t)bytes, 0),
                         "default-stream peer copy");
            if (ok) {
                ok = cuda_ok(cudaEventRecord(
                                 (cudaEvent_t)g_gpu[sd].boundary_event, 0),
                             "default-stream peer event record");
            }
        }
        if (ok) {
            WITH_DEVICE(g_gpu[dd].device_id) {
                ok = cuda_ok(cudaStreamWaitEvent(
                                 0,
                                 (cudaEvent_t)g_gpu[sd].boundary_event,
                                 0),
                             "default-stream peer destination wait");
            }
        }
        return ok;
    }

    if (g_xdev_bounce_bytes[sd][dd] < bytes) {
        if (g_xdev_bounce[sd][dd]) (void)cudaFreeHost(g_xdev_bounce[sd][dd]);
        if (!cuda_ok(cudaMallocHost(&g_xdev_bounce[sd][dd], (size_t)bytes),
                     "default-stream bounce alloc")) return 0;
        g_xdev_bounce_bytes[sd][dd] = bytes;
    }
    int ok = 0;
    WITH_DEVICE(g_gpu[sd].device_id) {
        ok = cuda_ok(cudaMemcpy(g_xdev_bounce[sd][dd], src->ptr, (size_t)bytes,
                                cudaMemcpyDeviceToHost),
                     "default-stream bounce d2h");
    }
    if (ok) {
        WITH_DEVICE(g_gpu[dd].device_id) {
            ok = cuda_ok(cudaMemcpy(dst->ptr, g_xdev_bounce[sd][dd],
                                    (size_t)bytes, cudaMemcpyHostToDevice),
                         "default-stream bounce h2d");
        }
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor *dst,
                                                 const ds4_gpu_tensor *src,
                                                 uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev_default_impl(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev3_default_dst(
        ds4_gpu_tensor *dst0,
        const ds4_gpu_tensor *src0,
        uint64_t bytes0,
        ds4_gpu_tensor *dst1,
        const ds4_gpu_tensor *src1,
        uint64_t bytes1,
        ds4_gpu_tensor *dst2,
        const ds4_gpu_tensor *src2,
        uint64_t bytes2) {
    ds4_gpu_tensor *dsts[3] = {dst0, dst1, dst2};
    const ds4_gpu_tensor *srcs[3] = {src0, src1, src2};
    const uint64_t sizes[3] = {bytes0, bytes1, bytes2};
    int sd = -1;
    int dd = -1;
    for (int i = 0; i < 3; i++) {
        if (sizes[i] == 0u) continue;
        if (!dsts[i] || !srcs[i] || sizes[i] > dsts[i]->bytes ||
            sizes[i] > srcs[i]->bytes) {
            return 0;
        }
        const int this_sd = ds4_tensor_device_idx(srcs[i]);
        const int this_dd = ds4_tensor_device_idx(dsts[i]);
        if (sd < 0) {
            sd = this_sd;
            dd = this_dd;
        } else if (sd != this_sd || dd != this_dd) {
            return 0;
        }
    }
    if (sd < 0) return 1;
    if (sd == dd) {
        int ok = 1;
        WITH_DEVICE(g_gpu[sd].device_id) {
            for (int i = 0; ok && i < 3; i++) {
                if (sizes[i] == 0u) continue;
                ok = cuda_ok(cudaMemcpyAsync(
                        dsts[i]->ptr, srcs[i]->ptr, (size_t)sizes[i],
                        cudaMemcpyDeviceToDevice, 0),
                        "grouped default same-device copy");
            }
        }
        return ok;
    }

    int peer = g_gpu_peer_ok[dd][sd];
    if (g_xdev_force_cuda_peer) peer = 1;
    if (g_xdev_force_host_bounce) peer = 0;
    if (!peer) {
        for (int i = 0; i < 3; i++) {
            if (sizes[i] != 0u &&
                !ds4_gpu_tensor_copy_xdev_default_impl(
                        dsts[i], srcs[i], sizes[i])) {
                return 0;
            }
        }
        return 1;
    }

    int ok = 0;
    WITH_DEVICE(g_gpu[sd].device_id) {
        ok = cuda_ok(cudaEventRecord(
                         (cudaEvent_t)g_gpu[sd].boundary_event, 0),
                     "grouped default source-ready record");
    }
    if (!ok) return 0;
    WITH_DEVICE(g_gpu[dd].device_id) {
        ok = cuda_ok(cudaStreamWaitEvent(
                         0, (cudaEvent_t)g_gpu[sd].boundary_event, 0),
                     "grouped default destination wait");
        for (int i = 0; ok && i < 3; i++) {
            if (sizes[i] == 0u) continue;
            ok = cuda_ok(cudaMemcpyPeerAsync(
                             dsts[i]->ptr, g_gpu[dd].device_id,
                             srcs[i]->ptr, g_gpu[sd].device_id,
                             (size_t)sizes[i], 0),
                         "grouped destination-stream peer copy");
        }
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_copy_xdev3(ds4_gpu_tensor       *dst0,
                                         const ds4_gpu_tensor *src0,
                                         uint64_t              bytes0,
                                         ds4_gpu_tensor       *dst1,
                                         const ds4_gpu_tensor *src1,
                                         uint64_t              bytes1,
                                         ds4_gpu_tensor       *dst2,
                                         const ds4_gpu_tensor *src2,
                                         uint64_t              bytes2) {
    ds4_gpu_tensor *dsts[3] = {dst0, dst1, dst2};
    const ds4_gpu_tensor *srcs[3] = {src0, src1, src2};
    uint64_t bytes[3] = {bytes0, bytes1, bytes2};
    int first = -1;
    for (int i = 0; i < 3; i++) {
        if (bytes[i] == 0) continue;
        if (!dsts[i] || !srcs[i] ||
            bytes[i] > dsts[i]->bytes || bytes[i] > srcs[i]->bytes) {
            return 0;
        }
        if (first < 0) first = i;
    }
    if (first < 0) return 1;

    const int sd = ds4_tensor_device_idx(srcs[first]);
    const int dd = ds4_tensor_device_idx(dsts[first]);
    for (int i = first + 1; i < 3; i++) {
        if (bytes[i] == 0) continue;
        if (ds4_tensor_device_idx(srcs[i]) != sd ||
            ds4_tensor_device_idx(dsts[i]) != dd) {
            int ok = 1;
            for (int j = 0; ok && j < 3; j++) {
                if (bytes[j] == 0) continue;
                ok = ds4_gpu_tensor_copy_xdev(dsts[j], srcs[j], bytes[j]);
            }
            return ok;
        }
    }

    if (sd == dd) {
        int ok = 0;
        WITH_DEVICE(g_gpu[sd].device_id) {
            cudaStream_t s = (cudaStream_t)g_gpu[sd].stream;
            ok = 1;
            for (int i = 0; ok && i < 3; i++) {
                if (bytes[i] == 0) continue;
                ok = cuda_ok(cudaMemcpyAsync(dsts[i]->ptr, srcs[i]->ptr, bytes[i],
                                             cudaMemcpyDeviceToDevice, s),
                             "xdev3 same-device copy");
            }
            if (ok && g_xdev_sync_debug) {
                ok = cuda_ok(cudaStreamSynchronize(s), "xdev3 same-device sync");
            }
        }
        return ok;
    }

    int peer = g_gpu_peer_ok[sd][dd];
    if (g_xdev_force_cuda_peer)   peer = 1;
    if (g_xdev_force_host_bounce) peer = 0;
    if (!peer) {
        int ok = 1;
        for (int i = 0; ok && i < 3; i++) {
            if (bytes[i] == 0) continue;
            ok = ds4_gpu_tensor_copy_xdev(dsts[i], srcs[i], bytes[i]);
        }
        return ok;
    }

    int ok = 0;
    WITH_DEVICE(g_gpu[sd].device_id) {
        cudaStream_t s = (cudaStream_t)g_gpu[sd].stream;
        cudaEvent_t  e = (cudaEvent_t)g_gpu[sd].boundary_event;
        ok = 1;
        for (int i = 0; ok && i < 3; i++) {
            if (bytes[i] == 0) continue;
            ok = cuda_ok(cudaMemcpyPeerAsync(
                            dsts[i]->ptr, g_gpu[dd].device_id,
                            srcs[i]->ptr, g_gpu[sd].device_id,
                            bytes[i], s),
                         "peer copy3");
        }
        if (ok) ok = cuda_ok(cudaEventRecord(e, s), "peer copy3 event record");
    }
    if (!ok) return 0;
    WITH_DEVICE(g_gpu[dd].device_id) {
        cudaStream_t s2 = (cudaStream_t)g_gpu[dd].stream;
        ok = cuda_ok(cudaStreamWaitEvent(s2, (cudaEvent_t)g_gpu[sd].boundary_event, 0),
                     "peer copy3 dst wait");
        if (ok && g_xdev_sync_debug) {
            ok = cuda_ok(cudaStreamSynchronize(s2), "peer copy3 dst sync");
        }
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *dst,
                                                const ds4_gpu_tensor *src,
                                                uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev_impl(dst, src, bytes, true);
}

extern "C" int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *src, int dst_tier) {
    if (!src) return 0;
    if (dst_tier < 0 || dst_tier >= g_n_gpus) return 0;
    int sd = ds4_tensor_device_idx(src);
    int dd = dst_tier;
    if (sd == dd) return 1;
    int ok = 0;
    WITH_DEVICE(g_gpu[sd].device_id) {
        cudaStream_t s = (cudaStream_t)g_gpu[sd].stream;
        cudaEvent_t  e = (cudaEvent_t)g_gpu[sd].boundary_event;
        ok = cuda_ok(cudaEventRecord(e, s), "xdev wait source event record");
    }
    if (!ok) return 0;
    WITH_DEVICE(g_gpu[dd].device_id) {
        cudaStream_t s2 = (cudaStream_t)g_gpu[dd].stream;
        ok = cuda_ok(cudaStreamWaitEvent(s2, (cudaEvent_t)g_gpu[sd].boundary_event, 0),
                     "xdev wait destination wait");
        if (ok && g_xdev_sync_debug) {
            ok = cuda_ok(cudaStreamSynchronize(s2), "xdev wait dst sync");
        }
    }
    return ok;
}

extern "C" int ds4_gpu_tensor_wait_xdev_default(
        const ds4_gpu_tensor *src,
        int dst_tier) {
    if (!src || dst_tier < 0 || dst_tier >= g_n_gpus) return 0;
    const int sd = ds4_tensor_device_idx(src);
    const int dd = dst_tier;
    if (sd == dd) return 1;
    int ok = 0;
    WITH_DEVICE(g_gpu[sd].device_id) {
        ok = cuda_ok(cudaEventRecord(
                         (cudaEvent_t)g_gpu[sd].boundary_event, 0),
                     "default xdev wait source event record");
    }
    if (!ok) return 0;
    WITH_DEVICE(g_gpu[dd].device_id) {
        ok = cuda_ok(cudaStreamWaitEvent(
                         0,
                         (cudaEvent_t)g_gpu[sd].boundary_event,
                         0),
                     "default xdev wait destination wait");
    }
    return ok;
}
#endif

extern "C" int ds4_gpu_q8_cache_suppressed(void) {
    return g_q8_cache_suppressed;
}

extern "C" void ds4_gpu_set_q8_cache_suppressed(int suppressed) {
    g_q8_cache_suppressed = suppressed ? 1 : 0;
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

extern "C" int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_model_host_base == model_map && g_model_registered_size == model_size) return 1;
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
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

    const char *copy_env = getenv("DS4_CUDA_COPY_MODEL");
    if (copy_env && copy_env[0]) {
        void *dev = NULL;
        const double t0 = clock() / (double)CLOCKS_PER_SEC;
        cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
        if (err == cudaSuccess) {
            fprintf(stderr, "ds4: CUDA copying %.2f GiB model to device memory\n",
                    (double)model_size / 1073741824.0);
            err = cudaMemcpy(dev, model_map, (size_t)model_size, cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                g_model_device_base = (const char *)dev;
                g_model_device_owned = 1;
                const double t1 = clock() / (double)CLOCKS_PER_SEC;
                fprintf(stderr, "ds4: CUDA model copy complete in %.3fs\n", t1 - t0);
                return 1;
            }
            fprintf(stderr, "ds4: CUDA model copy failed: %s\n", cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
        } else {
            fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
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
            fprintf(stderr, "ds4: CUDA registered %.2f GiB model mapping for device access\n",
                    (double)model_size / 1073741824.0);
        } else {
            fprintf(stderr, "ds4: CUDA host registration pointer lookup failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    } else {
        fprintf(stderr, "ds4: CUDA host registration skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!ds4_gpu_register_model_map_no_copy(model_map, model_size)) return 0;
    if (getenv("DS4_CUDA_COPY_MODEL_CHUNKED") != NULL &&
        !cuda_model_copy_chunked(model_map, model_size, map_offset, map_size)) {
        (void)cuda_model_prefetch_range(model_map, model_size, map_offset, map_size);
    }
    return 1;
}

/* Register the mmap'd host model pointer for selective-cache lookups WITHOUT
 * triggering any device-side copy. Used by multi-GPU placement scaffolding's
 * multi-tier path so DS4_CUDA_COPY_MODEL cannot reintroduce a full-model
 * copy that defeats the per-device selective cache.
 *
 * This is the no-copy subset of ds4_gpu_set_model_map: same bookkeeping
 * for the host pointer plus cudaHostRegister, but skipping the
 * DS4_CUDA_COPY_MODEL branch that allocates and copies the entire model. */
extern "C" int ds4_gpu_register_model_map_no_copy(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_model_host_base == model_map && g_model_registered_size == model_size) return 1;

    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
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

    /* No DS4_CUDA_COPY_MODEL branch — that is the entire point. */

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
                    "ds4: CUDA (no-copy) registered %.2f GiB model mapping for multi-tier selective cache\n",
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

/* Set the current CUDA device by LOGICAL tier index (0..g_n_gpus-1).
 * Maps to the physical CUDA device id stored in g_gpu[].device_id.
 * Added for multi-GPU placement scaffolding (multi-GPU CLI); first executed by
 * multi-GPU execution (follow-up). */
extern "C" int ds4_gpu_set_current_device(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= g_n_gpus) return -1;
    if (!g_cuda_no_setdevice_cache && g_current_logical_tier == logical_tier) {
        return 0;
    }
    if (cudaSetDevice(g_gpu[logical_tier].device_id) == cudaSuccess) {
        g_current_logical_tier = logical_tier;
        return 0;
    }
    g_current_logical_tier = -1;
    return -1;
}

/* Fenced device switch for sequential cross-device pipelines (GLM
 * per-layer placement): work queued on the next device's default stream
 * waits for everything queued so far on the previous device's default
 * stream. Async — no host sync. Falls back to a plain switch when the
 * device does not change. */
extern "C" int ds4_gpu_set_current_device_fenced(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= g_n_gpus) return -1;
    static cudaEvent_t fence_ev[DS4_MAX_GPUS];
    /* Resolve the ACTUAL current device: WITH_DEVICE blocks and direct
     * cudaSetDevice calls can leave g_current_logical_tier stale, and a
     * false "already there" here strands work on the wrong device. */
    int cur_dev = -1;
    (void)cudaGetDevice(&cur_dev);
    int prev = -1;
    for (int t = 0; t < g_n_gpus; t++) {
        if (g_gpu[t].device_id == cur_dev) { prev = t; break; }
    }
    if (getenv("DS4_GLM_FENCE_TRACE")) {
        fprintf(stderr, "ds4: fenced switch %d -> %d\n", prev, logical_tier);
    }
    if (prev == logical_tier) {
        g_current_logical_tier = logical_tier;
        return 0;
    }
    if (prev >= 0 && prev < g_n_gpus && prev != logical_tier) {
        if (cudaSetDevice(g_gpu[prev].device_id) != cudaSuccess) return -1;
        if (!fence_ev[prev] &&
            cudaEventCreateWithFlags(&fence_ev[prev],
                                     cudaEventDisableTiming) != cudaSuccess) {
            fence_ev[prev] = NULL;
        }
        if (fence_ev[prev]) {
            (void)cudaEventRecord(fence_ev[prev], 0);
        }
        if (cudaSetDevice(g_gpu[logical_tier].device_id) != cudaSuccess) {
            g_current_logical_tier = -1;
            return -1;
        }
        g_current_logical_tier = logical_tier;
        if (fence_ev[prev]) {
            (void)cudaStreamWaitEvent(0, fence_ev[prev], 0);
        }
        return 0;
    }
    return ds4_gpu_set_current_device(logical_tier);
}
