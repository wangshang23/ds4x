template <ggml_type type>
static __device__ __forceinline__ float ds4_mmq_vec_dot_q8_1(
        const void * __restrict__ W,
        const block_q8_1 * __restrict__ X_q8,
        const int & kbx,
        const int & iqs) {
    if constexpr (type == GGML_TYPE_IQ2_XXS) {
        return vec_dot_iq2_xxs_q8_1(W, X_q8, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        return vec_dot_q2_K_q8_1(W, X_q8, kbx, iqs);
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        return vec_dot_q4_K_q8_1(W, X_q8, kbx, iqs);
    } else {
        static_assert(type == GGML_TYPE_MXFP4, "unsupported fused vector type");
        return vec_dot_mxfp4_q8_1(W, X_q8, kbx, iqs);
    }
}

static __device__ __forceinline__ float ds4_mmq_half_warp_sum_f32(float v) {
    const uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 16);
    }
    return v;
}

template <ggml_type type>
static __global__ void ds4_mmq_moe_down_sum6_q8_1_qwarp32_kernel(
        const void       * __restrict__ W,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        float            * __restrict__ out,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_tokens,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x) {

    constexpr int top_k = 6;
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int q8_per_k = qk / QK8_1;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int lanes_per_k = qi / vdr;
    constexpr int blocks_per_iter = vdr * 16 / qi;
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t row_lane = threadIdx.x >> 4u;
    const uint32_t tok  = blockIdx.y;
    if (tok >= n_tokens) return;

    const uint32_t blocks_per_row_x = ncols_x / qk;
    const uint32_t kbx0 = lane / lanes_per_k;
    const int kqs = vdr * (lane % lanes_per_k);

#pragma unroll
    for (uint32_t rr = 0; rr < 4u; ++rr) {
        const uint32_t row = blockIdx.x * 64u + row_lane + rr * 16u;
        if (row >= nrows_x) continue;
        float total = 0.0f;
#pragma unroll
        for (uint32_t slot = 0; slot < top_k; ++slot) {
            const uint32_t assignment = tok * top_k + slot;
            const int32_t id_raw = ids[assignment];
            const bool invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
            const uint32_t expert = invalid_id ? 0u : (uint32_t)id_raw;
            const block_q8_1 * xq = X_q8 + (uint64_t)assignment * stride_col_y;
            const int kbx_base = (int)(expert * stride_channel_x + row * stride_row_x);
            float acc = 0.0f;
            for (uint32_t b = kbx0; !invalid_id && b < blocks_per_row_x; b += blocks_per_iter) {
                acc += ds4_mmq_vec_dot_q8_1<type>(
                    W, xq + (uint64_t)b * q8_per_k, kbx_base + (int)b, kqs);
            }
            acc = ds4_mmq_half_warp_sum_f32(acc);
            if (lane == 0) {
                if (!isfinite(acc)) acc = 0.0f;
                total += acc;
            }
        }
        if (lane == 0) {
            out[(uint64_t)tok * nrows_x + row] = total;
        }
    }
}

template <ggml_type type>
static __global__ void ds4_mmq_moe_gate_up_mid_q8_1_qwarp32_kernel(
        const void       * __restrict__ W_gate,
        const void       * __restrict__ W_up,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        const float      * __restrict__ weights,
        float            * __restrict__ mid,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_tokens,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x,
        const float clamp) {

    constexpr int top_k = 6;
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int q8_per_k = qk / QK8_1;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int lanes_per_k = qi / vdr;
    constexpr int blocks_per_iter = vdr * 16 / qi;
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t row_lane = threadIdx.x >> 4u;
    const uint32_t assignment = blockIdx.y;
    const uint32_t tok = assignment / top_k;
    const uint32_t slot = assignment - tok * top_k;
    if (tok >= n_tokens) return;

    const int32_t id_raw = ids[(uint64_t)tok * top_k + slot];
    const bool invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
    const uint32_t expert = invalid_id ? 0u : (uint32_t)id_raw;
    const block_q8_1 * xq = X_q8 + (uint64_t)tok * stride_col_y;
    const uint32_t blocks_per_row_x = ncols_x / qk;
    const uint32_t kbx0 = lane / lanes_per_k;
    const int kqs = vdr * (lane % lanes_per_k);

#pragma unroll
    for (uint32_t rr = 0; rr < 4u; ++rr) {
        const uint32_t row = blockIdx.x * 64u + row_lane + rr * 16u;
        if (row >= nrows_x) continue;
        const int kbx_base = (int)(expert * stride_channel_x + row * stride_row_x);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = kbx0; !invalid_id && b < blocks_per_row_x; b += blocks_per_iter) {
            const block_q8_1 * xb = xq + (uint64_t)b * q8_per_k;
            const int kbx = kbx_base + (int)b;
            gate += ds4_mmq_vec_dot_q8_1<type>(W_gate, xb, kbx, kqs);
            up   += ds4_mmq_vec_dot_q8_1<type>(W_up,   xb, kbx, kqs);
        }
        gate = ds4_mmq_half_warp_sum_f32(gate);
        up   = ds4_mmq_half_warp_sum_f32(up);
        if (lane == 0) {
            if (!isfinite(gate)) gate = 0.0f;
            if (!isfinite(up)) up = 0.0f;
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const float silu = gate / (1.0f + expf(-gate));
            mid[(uint64_t)assignment * nrows_x + row] = silu * up * weights[(uint64_t)tok * top_k + slot];
        }
    }
}

template <ggml_type type, int c_rows_per_block>
static __global__ void ds4_mmq_moe_down_sum6_vec_kernel(
        const void       * __restrict__ W,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        float            * __restrict__ out,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_tokens,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x) {

    constexpr int top_k = 6;
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const uint32_t slot  = threadIdx.y;
    const uint32_t token = blockIdx.y;
    const uint32_t row0  = c_rows_per_block * blockIdx.x;

    if (slot >= top_k || token >= n_tokens) {
        return;
    }

    const uint32_t assignment = token * top_k + slot;
    const int32_t  id_raw     = ids[assignment];
    const bool     invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
    const uint32_t expert     = invalid_id ? 0u : (uint32_t)id_raw;

    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const block_q8_1 * y = X_q8 + (uint64_t)assignment * stride_col_y;
    const int kbx_offset = (int)(expert * stride_channel_x + row0 * stride_row_x);

    float tmp[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi / vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi / vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += ds4_mmq_vec_dot_q8_1<type>(
                W, &y[kby], kbx_offset + i * stride_row_x + kbx, kqs);
        }
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    __shared__ float partial[top_k][c_rows_per_block];
    if (threadIdx.x < c_rows_per_block) {
        const uint32_t row = row0 + threadIdx.x;
        partial[slot][threadIdx.x] = row < nrows_x ? tmp[threadIdx.x] : 0.0f;
    }
    __syncthreads();

    if (slot == 0 && threadIdx.x < c_rows_per_block) {
        const uint32_t row = row0 + threadIdx.x;
        if (row < nrows_x) {
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < top_k; ++s) {
                sum += partial[s][threadIdx.x];
            }
            out[(uint64_t)token * nrows_x + row] = sum;
        }
    }
}

template <ggml_type type>
int ds4_mmq_moe_down_sum6_vec_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        cudaStream_t    stream) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used != 6) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);

    const int n_assignments = n_tokens * n_expert_used;
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t nbytes_q8_1 = (size_t)n_assignments * ne10_padded *
                               sizeof(block_q8_1) / QK8_1;

    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1_ptr = nullptr;
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_assignments,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_assignments, /*ne3=*/1,
        stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t blck = ggml_blck_size(type);
    const uint32_t stride_row_x     = (uint32_t)((int64_t)K / blck);
    const uint32_t stride_col_y     = (uint32_t)(ne10_padded / QK8_1);
    const uint32_t stride_channel_x = (uint32_t)((int64_t)M * stride_row_x);

    const dim3 block_nums((M + 63) / 64, n_tokens);
    const dim3 block_dims(256);

    ds4_mmq_moe_down_sum6_q8_1_qwarp32_kernel<type><<<block_nums, block_dims, 0, stream>>>(
        W, (const block_q8_1 *)src1_q8_1_ptr, ids, out_f32,
        (uint32_t)K, (uint32_t)M, (uint32_t)n_tokens, (uint32_t)n_experts,
        stride_row_x, stride_col_y, stride_channel_x);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: fused down+sum launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }

    return 0;
}

template <ggml_type type, int c_rows_per_block>
static __global__ void ds4_mmq_moe_gate_up_mid_vec_kernel(
        const void       * __restrict__ W_gate,
        const void       * __restrict__ W_up,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        const float      * __restrict__ weights,
        float            * __restrict__ mid,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_tokens,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x,
        const float clamp) {

    constexpr int top_k = 6;
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const uint32_t slot  = threadIdx.y;
    const uint32_t token = blockIdx.y;
    const uint32_t row0  = c_rows_per_block * blockIdx.x;

    const uint32_t assignment = token * top_k + slot;
    const int32_t  id_raw     = ids[assignment];
    const bool     invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
    const uint32_t expert     = invalid_id ? 0u : (uint32_t)id_raw;

    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const block_q8_1 * y = X_q8 + (uint64_t)token * stride_col_y;
    const int kbx_offset = (int)(expert * stride_channel_x + row0 * stride_row_x);

    float gate[c_rows_per_block] = {0.0f};
    float up[c_rows_per_block]   = {0.0f};

    for (int kbx = threadIdx.x / (qi / vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi / vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            const int row_kbx = kbx_offset + i * stride_row_x + kbx;
            gate[i] += ds4_mmq_vec_dot_q8_1<type>(W_gate, &y[kby], row_kbx, kqs);
            up[i]   += ds4_mmq_vec_dot_q8_1<type>(W_up,   &y[kby], row_kbx, kqs);
        }
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        gate[i] = warp_reduce_sum<warp_size>(gate[i]);
        up[i]   = warp_reduce_sum<warp_size>(up[i]);
    }

    if (threadIdx.x < c_rows_per_block) {
        const uint32_t row = row0 + threadIdx.x;
        if (row < nrows_x) {
            float g = gate[threadIdx.x];
            float u = up[threadIdx.x];
            if (!isfinite(g)) g = 0.0f;
            if (!isfinite(u)) u = 0.0f;
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            const float silu = g / (1.0f + expf(-g));
            mid[(uint64_t)assignment * nrows_x + row] = silu * u * weights[assignment];
        }
    }
}

template <ggml_type type, int c_rows_per_block>
static __global__ void ds4_mmq_moe_gate_up_mid_vec_by_slot_kernel(
        const void       * __restrict__ W_gate,
        const void       * __restrict__ W_up,
        const block_q8_1 * __restrict__ X_q8,
        const int32_t    * __restrict__ ids,
        const float      * __restrict__ weights,
        float            * __restrict__ mid,
        const uint32_t ncols_x,
        const uint32_t nrows_x,
        const uint32_t n_experts,
        const uint32_t stride_row_x,
        const uint32_t stride_col_y,
        const uint32_t stride_channel_x,
        const uint32_t token0,
        const float clamp) {

    constexpr int top_k = 6;
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = ds4_mmq_vdr_mmvq_value<type>::value;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const uint32_t slot  = blockIdx.y;
    const uint32_t token = token0 + threadIdx.y;
    const uint32_t row0  = c_rows_per_block * blockIdx.x;

    const uint32_t assignment = token * top_k + slot;
    const int32_t  id_raw     = ids[assignment];
    const bool     invalid_id = id_raw < 0 || (uint32_t)id_raw >= n_experts;
    const uint32_t expert     = invalid_id ? 0u : (uint32_t)id_raw;

    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const block_q8_1 * y = X_q8 + (uint64_t)token * stride_col_y;
    const int kbx_offset = (int)(expert * stride_channel_x + row0 * stride_row_x);

    float gate[c_rows_per_block] = {0.0f};
    float up[c_rows_per_block]   = {0.0f};

    for (int kbx = threadIdx.x / (qi / vdr); !invalid_id && kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi / vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            const int row_kbx = kbx_offset + i * stride_row_x + kbx;
            gate[i] += ds4_mmq_vec_dot_q8_1<type>(W_gate, &y[kby], row_kbx, kqs);
            up[i]   += ds4_mmq_vec_dot_q8_1<type>(W_up,   &y[kby], row_kbx, kqs);
        }
    }

#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        gate[i] = warp_reduce_sum<warp_size>(gate[i]);
        up[i]   = warp_reduce_sum<warp_size>(up[i]);
    }

    if (threadIdx.x < c_rows_per_block) {
        const uint32_t row = row0 + threadIdx.x;
        if (row < nrows_x) {
            float g = gate[threadIdx.x];
            float u = up[threadIdx.x];
            if (!isfinite(g)) g = 0.0f;
            if (!isfinite(u)) u = 0.0f;
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            const float silu = g / (1.0f + expf(-g));
            mid[(uint64_t)assignment * nrows_x + row] = silu * u * weights[assignment];
        }
    }
}

template <ggml_type type>
int ds4_mmq_moe_gate_up_mid_vec_impl(
        const char    * tag,
        const void    * W_gate,
        const void    * W_up,
        const float   * X_f32,
        const int32_t * ids,
        const float   * weights,
        float         * mid_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        float           clamp,
        cudaStream_t    stream) {

    if (!W_gate || !W_up || !X_f32 || !ids || !weights || !mid_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used != 6) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    ggml_backend_cuda_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }

    ds4_pool_set_stream(stream);

    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t nbytes_q8_1 = (size_t)n_tokens * ne10_padded *
                               sizeof(block_q8_1) / QK8_1;

    ggml_cuda_pool_alloc<char> src1_q8_1_pool;
    // M2-Inc2a: the fused HC stage may have emitted this activation's q8_1
    // codes already (ffn_norm) -- take them and skip the quantize prelude.
    char *src1_q8_1_ptr = ds4_mmq_folded_q81(X_f32, K, n_tokens, ne10_padded);
    cudaError_t err;
    if (!src1_q8_1_ptr) {
    if (g_q81_scratch_enabled && g_q81_scratch_ptr && g_q81_scratch_bytes >= nbytes_q8_1) {
        src1_q8_1_ptr = (char *)g_q81_scratch_ptr;
    } else {
        src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
        src1_q8_1_ptr = src1_q8_1_pool.get();
    }

    quantize_row_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize_row_q8_1_cuda failed: %s\n",
                tag, cudaGetErrorString(err));
        return -2;
    }
    }

    const int64_t blck = ggml_blck_size(type);
    const uint32_t stride_row_x     = (uint32_t)((int64_t)K / blck);
    const uint32_t stride_col_y     = (uint32_t)(ne10_padded / QK8_1);
    const uint32_t stride_channel_x = (uint32_t)((int64_t)M * stride_row_x);

    const dim3 block_nums((M + 63) / 64, n_tokens * n_expert_used);
    const dim3 block_dims(256);
    ds4_mmq_moe_gate_up_mid_q8_1_qwarp32_kernel<type><<<block_nums, block_dims, 0, stream>>>(
        W_gate, W_up, (const block_q8_1 *)src1_q8_1_ptr, ids, weights, mid_f32,
        (uint32_t)K, (uint32_t)M, (uint32_t)n_tokens, (uint32_t)n_experts,
        stride_row_x, stride_col_y, stride_channel_x, clamp);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: fused gate+up qwarp launch failed: %s\n",
                tag, cudaGetErrorString(err));
        return -3;
    }

    return 0;
}

} // anonymous namespace

extern "C" int ds4_mmq_q8_0_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_Q8_0>(
        "ds4_mmq_q8_0_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q2_K_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_Q2_K>(
        "ds4_mmq_q2_K_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q4_K_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_mxfp4_moe_vec(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_vec_impl<GGML_TYPE_MXFP4>(
        "ds4_mmq_mxfp4_moe_vec", W, X, ids, out, M, K,
        n_tokens, n_experts, n_expert_used, stream);
}

// M1-Inc2b: exact inverse of the weight-server repack
// (repack_iq2_xxs_aligned_kernel, tools/ds4_weight_server.cu): aligned-SoA
// artifact -> raw block_iq2_xxs byte stream (66B = [half d][8 x uint2
// codes]).  Device->device fill of a raw-layout scratch so the batched/mmq
// consumers keep their layout while the raw spans stay excluded from the
// upload.  One thread per (block, pair); p==0 additionally writes the
// 2-byte scale.  Destination blocks are 66B so stores are byte-granular.
__global__ void iq2_xxs_aligned_derepack_kernel(
        unsigned char     *raw,        // [nblk * 66]
        const uint2       *qs,         // 64B-aligned code pairs
        const __half      *dq,         // block scales
        uint64_t           nblk)
{
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblk * 8ull) return;
    const uint64_t blk = i >> 3;
    const uint32_t p = (uint32_t)(i & 7u);
    unsigned char *dst = raw + blk * 66ull;
    if (p == 0u) {
        const uint16_t h = __half_as_ushort(dq[blk]);
        memcpy(dst, &h, 2u);
    }
    const uint2 v = qs[blk * 8ull + p];
    memcpy(dst + 2u + (uint64_t)p * 8u, &v, 8u);
}

