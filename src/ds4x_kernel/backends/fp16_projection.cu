#include "ds4x/fp16_projection.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ds4x::kernel {
namespace {

__global__ void fp16_projection_parallel(
        float *output,
        const __half *weights,
        const float *activations,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t tokens) {
    const uint64_t row = blockIdx.x;
    const uint64_t token = blockIdx.y;
    if (row >= out_dim || token >= tokens) return;

    float sum = 0.0f;
    const __half *weight_row = weights + row * in_dim;
    const float *activation_row = activations + token * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(weight_row[i]) * activation_row[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[token * out_dim + row] = partial[0];
}

__global__ void fp16_projection_serial(
        float *output,
        const __half *weights,
        const float *activations,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t tokens) {
    const uint64_t row = blockIdx.x;
    const uint64_t token = blockIdx.y;
    if (row >= out_dim || token >= tokens || threadIdx.x != 0) return;

    float sum = 0.0f;
    const __half *weight_row = weights + row * in_dim;
    const float *activation_row = activations + token * in_dim;
    for (uint64_t i = 0; i < in_dim; ++i) {
        sum += __half2float(weight_row[i]) * activation_row[i];
    }
    output[token * out_dim + row] = sum;
}

__global__ void fp16_projection_ordered(
        float *output,
        const __half *weights,
        const float *activations,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t tokens) {
    const uint64_t row = blockIdx.x;
    const uint64_t token = blockIdx.y;
    if (row >= out_dim || token >= tokens) return;

    __shared__ float partial[32];
    const uint32_t lane = threadIdx.x;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t begin = static_cast<uint64_t>(lane) * chunk;
    const uint64_t end = begin + chunk < in_dim ? begin + chunk : in_dim;
    const __half *weight_row = weights + row * in_dim;
    const float *activation_row = activations + token * in_dim;
    float sum = 0.0f;
    for (uint64_t i = begin; i < end; ++i) {
        sum += __half2float(weight_row[i]) * activation_row[i];
    }
    partial[lane] = sum;
    __syncthreads();
    if (lane == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; ++i) total += partial[i];
        output[token * out_dim + row] = total;
    }
}

__global__ void fp16_small_output_one(
        float *output,
        const __half *weights,
        const float *activations,
        uint64_t in_dim,
        uint64_t out_dim) {
    const uint64_t row = blockIdx.x;
    if (row >= out_dim) return;

    __shared__ float partial[32];
    const uint32_t lane = threadIdx.x;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t begin = static_cast<uint64_t>(lane) * chunk;
    const uint64_t end = begin + chunk < in_dim ? begin + chunk : in_dim;
    const __half *weight_row = weights + row * in_dim;
    float sum = 0.0f;
    for (uint64_t i = begin; i < end; ++i) {
        const float x = __half2float(__float2half(activations[i]));
        sum += __half2float(weight_row[i]) * x;
    }
    partial[lane] = sum;
    __syncthreads();
    if (lane == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; ++i) total += partial[i];
        output[row] = total;
    }
}

__global__ void fp16_small_output_batch(
        float *output,
        const __half *weights,
        const float *activations,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t tokens) {
    const uint64_t token = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (token >= tokens || out_dim > 32u || blockDim.x != 256u) return;

    float accumulators[32] = {};
    const float *activation_row = activations + token * in_dim;
    for (uint64_t i = lane; i < in_dim; i += 256u) {
        const float x = activation_row[i];
#pragma unroll
        for (uint32_t row = 0; row < 32u; ++row) {
            if (row < out_dim) {
                accumulators[row] += __half2float(weights[(uint64_t)row * in_dim + i]) * x;
            }
        }
    }

    __shared__ float partial[32 * 256];
#pragma unroll
    for (uint32_t row = 0; row < 32u; ++row) {
        if (row < out_dim) partial[row * 256u + lane] = accumulators[row];
    }
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
#pragma unroll
            for (uint32_t row = 0; row < 32u; ++row) {
                if (row < out_dim) {
                    partial[row * 256u + lane] += partial[row * 256u + lane + stride];
                }
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
#pragma unroll
        for (uint32_t row = 0; row < 32u; ++row) {
            if (row < out_dim) output[token * out_dim + row] = partial[row * 256u];
        }
    }
}

__global__ void fp16_pair_ordered(
        float *output_a,
        float *output_b,
        const __half *weights_a,
        const __half *weights_b,
        const float *activations,
        uint64_t in_dim,
        uint64_t out_a_dim,
        uint64_t out_b_dim) {
    const uint64_t row = blockIdx.x;
    if (row >= out_a_dim && row >= out_b_dim) return;

    __shared__ float partial_a[32];
    __shared__ float partial_b[32];
    const uint32_t lane = threadIdx.x;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t begin = static_cast<uint64_t>(lane) * chunk;
    const uint64_t end = begin + chunk < in_dim ? begin + chunk : in_dim;
    const __half *weight_a = row < out_a_dim ? weights_a + row * in_dim : weights_a;
    const __half *weight_b = row < out_b_dim ? weights_b + row * in_dim : weights_b;
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (uint64_t i = begin; i < end; ++i) {
        const float x = activations[i];
        if (row < out_a_dim) sum_a += __half2float(weight_a[i]) * x;
        if (row < out_b_dim) sum_b += __half2float(weight_b[i]) * x;
    }
    partial_a[lane] = sum_a;
    partial_b[lane] = sum_b;
    __syncthreads();
    if (lane == 0) {
        float total_a = 0.0f;
        float total_b = 0.0f;
        for (uint32_t i = 0; i < 32u; ++i) {
            total_a += partial_a[i];
            total_b += partial_b[i];
        }
        if (row < out_a_dim) output_a[row] = total_a;
        if (row < out_b_dim) output_b[row] = total_b;
    }
}

cudaStream_t as_stream(void *stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

}  // namespace
}  // namespace ds4x::kernel

extern "C" int ds4x_launch_fp16_projection(
        float *output,
        const uint16_t *weights,
        const float *activations,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t tokens,
        int mode,
        void *stream) {
    using namespace ds4x::kernel;
    const dim3 grid(static_cast<unsigned>(out_dim), static_cast<unsigned>(tokens));
    const __half *half_weights = reinterpret_cast<const __half *>(weights);
    switch (mode) {
    case DS4X_FP16_PROJECTION_SERIAL:
        fp16_projection_serial<<<grid, 1, 0, as_stream(stream)>>>(
                output, half_weights, activations, in_dim, out_dim, tokens);
        break;
    case DS4X_FP16_PROJECTION_ORDERED:
        fp16_projection_ordered<<<grid, 32, 0, as_stream(stream)>>>(
                output, half_weights, activations, in_dim, out_dim, tokens);
        break;
    case DS4X_FP16_PROJECTION_PARALLEL:
        fp16_projection_parallel<<<grid, 256, 0, as_stream(stream)>>>(
                output, half_weights, activations, in_dim, out_dim, tokens);
        break;
    default:
        return static_cast<int>(cudaErrorInvalidValue);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int ds4x_launch_fp16_small_output_one(
        float *output,
        const uint16_t *weights,
        const float *activations,
        uint64_t in_dim,
        uint64_t out_dim,
        void *stream) {
    using namespace ds4x::kernel;
    fp16_small_output_one<<<static_cast<unsigned>(out_dim), 32, 0, as_stream(stream)>>>(
            output, reinterpret_cast<const __half *>(weights), activations,
            in_dim, out_dim);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int ds4x_launch_fp16_small_output_batch(
        float *output,
        const uint16_t *weights,
        const float *activations,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t tokens,
        void *stream) {
    using namespace ds4x::kernel;
    fp16_small_output_batch<<<static_cast<unsigned>(tokens), 256, 0, as_stream(stream)>>>(
            output, reinterpret_cast<const __half *>(weights), activations,
            in_dim, out_dim, tokens);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int ds4x_launch_fp16_pair_ordered(
        float *output_a,
        float *output_b,
        const uint16_t *weights_a,
        const uint16_t *weights_b,
        const float *activations,
        uint64_t in_dim,
        uint64_t out_a_dim,
        uint64_t out_b_dim,
        void *stream) {
    using namespace ds4x::kernel;
    const uint64_t rows = out_a_dim > out_b_dim ? out_a_dim : out_b_dim;
    fp16_pair_ordered<<<static_cast<unsigned>(rows), 32, 0, as_stream(stream)>>>(
            output_a, output_b,
            reinterpret_cast<const __half *>(weights_a),
            reinterpret_cast<const __half *>(weights_b),
            activations, in_dim, out_a_dim, out_b_dim);
    return static_cast<int>(cudaGetLastError());
}
