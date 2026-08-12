#include "../internal/backend_internal.cuh"

/* Storage implementation. */
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

__global__ static void pack_slot_rows_f32_kernel(float *out, const float *slots, uint32_t n_rows, uint32_t width, uint32_t n_slots, uint32_t slot_cap) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_rows * n_slots * width;
    if (i >= n) return;

    uint64_t col = i % width;
    uint64_t slot = (i / width) % n_slots;
    uint64_t row = i / ((uint64_t)n_slots * width);
    out[i] = slots[((slot * slot_cap) + row) * width + col];
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
