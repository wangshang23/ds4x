#include "ds4x/cutlass_fp16_gemm.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do { \
    const cudaError_t status_ = (call); \
    if (status_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA failure %s:%d: %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(status_)); \
        std::exit(1); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    const cublasStatus_t status_ = (call); \
    if (status_ != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "cuBLAS failure %s:%d: %d\n", \
                     __FILE__, __LINE__, static_cast<int>(status_)); \
        std::exit(1); \
    } \
} while (0)

namespace {

struct Shape {
    const char *name;
    int tokens;
    int in_dim;
    int out_dim;
};

struct BatchedShape {
    const char *name;
    int tokens;
    int in_dim;
    int out_dim;
    int batch_count;
};

struct Buffers {
    __half *activations = nullptr;
    __half *weights = nullptr;
    float *cutlass_output = nullptr;
    float *cublas_output = nullptr;
};

void fill(std::vector<__half> &values, uint32_t seed) {
    for (auto &value : values) {
        seed = seed * 1664525u + 1013904223u;
        const float x = (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f) * 0.25f;
        value = __float2half(x);
    }
}

Buffers allocate(const Shape &shape) {
    Buffers buffers;
    const size_t activation_count = static_cast<size_t>(shape.tokens) * shape.in_dim;
    const size_t weight_count = static_cast<size_t>(shape.out_dim) * shape.in_dim;
    const size_t output_count = static_cast<size_t>(shape.tokens) * shape.out_dim;
    std::vector<__half> host_activations(activation_count);
    std::vector<__half> host_weights(weight_count);
    fill(host_activations, 0x1234u);
    fill(host_weights, 0x5678u);
    CUDA_CHECK(cudaMalloc(&buffers.activations, activation_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&buffers.weights, weight_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&buffers.cutlass_output, output_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers.cublas_output, output_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(buffers.activations, host_activations.data(),
                          activation_count * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.weights, host_weights.data(),
                          weight_count * sizeof(__half), cudaMemcpyHostToDevice));
    return buffers;
}

void release(Buffers &buffers) {
    cudaFree(buffers.cublas_output);
    cudaFree(buffers.cutlass_output);
    cudaFree(buffers.weights);
    cudaFree(buffers.activations);
}

void run_cublas(cublasHandle_t handle, const Shape &shape, const Buffers &buffers) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            shape.out_dim, shape.tokens, shape.in_dim,
            &alpha,
            buffers.weights, CUDA_R_16F, shape.in_dim,
            buffers.activations, CUDA_R_16F, shape.in_dim,
            &beta,
            buffers.cublas_output, CUDA_R_32F, shape.out_dim,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}

void run_cutlass(const Shape &shape, const Buffers &buffers) {
    const int status = ds4x_cutlass_fp16_gemm(
            buffers.cutlass_output,
            reinterpret_cast<const uint16_t *>(buffers.activations),
            reinterpret_cast<const uint16_t *>(buffers.weights),
            shape.tokens, shape.in_dim, shape.out_dim, nullptr);
    if (status != 1) {
        std::fprintf(stderr, "%s: CUTLASS backend failed: %s (%d)\n",
                     shape.name, ds4x_cutlass_status_string(status), status);
        std::exit(1);
    }
}

void check_parity(cublasHandle_t handle, const Shape &shape) {
    Buffers buffers = allocate(shape);
    run_cublas(handle, shape, buffers);
    run_cutlass(shape, buffers);
    CUDA_CHECK(cudaDeviceSynchronize());

    const size_t count = static_cast<size_t>(shape.tokens) * shape.out_dim;
    std::vector<float> cutlass_output(count);
    std::vector<float> cublas_output(count);
    CUDA_CHECK(cudaMemcpy(cutlass_output.data(), buffers.cutlass_output,
                          count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cublas_output.data(), buffers.cublas_output,
                          count * sizeof(float), cudaMemcpyDeviceToHost));
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float delta = std::fabs(cutlass_output[i] - cublas_output[i]);
        max_abs = std::max(max_abs, delta);
        max_rel = std::max(max_rel, delta / std::max(std::fabs(cublas_output[i]), 1.0e-6f));
    }
    std::printf("cutlass-parity: %-22s max_abs=%g max_rel=%g\n",
                shape.name, static_cast<double>(max_abs), static_cast<double>(max_rel));
    if (max_abs > 2.0e-3f && max_rel > 2.0e-3f) {
        std::fprintf(stderr, "%s: parity tolerance exceeded\n", shape.name);
        std::exit(1);
    }
    release(buffers);
}

template <typename Fn>
float time_ms(Fn fn, int warmup, int iterations) {
    cudaEvent_t begin;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(begin));
    for (int i = 0; i < iterations; ++i) fn();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    return elapsed / static_cast<float>(iterations);
}

void benchmark(cublasHandle_t handle, const Shape &shape) {
    Buffers buffers = allocate(shape);
    const int iterations = shape.tokens >= 1024 ? 30 : 100;
    const float cublas_ms = time_ms(
            [&] { run_cublas(handle, shape, buffers); }, 10, iterations);
    const float cutlass_ms = time_ms(
            [&] { run_cutlass(shape, buffers); }, 10, iterations);
    const double flops = 2.0 * shape.tokens * shape.in_dim * shape.out_dim;
    std::printf(
            "cutlass-bench: %-22s cuBLAS=%8.4f ms (%5.1f TF/s) "
            "CUTLASS=%8.4f ms (%5.1f TF/s) ratio=%6.3f\n",
            shape.name,
            cublas_ms, flops / (static_cast<double>(cublas_ms) * 1.0e9),
            cutlass_ms, flops / (static_cast<double>(cutlass_ms) * 1.0e9),
            cutlass_ms / cublas_ms);
    release(buffers);
}

Buffers allocate(const BatchedShape &shape) {
    Buffers buffers;
    const size_t activation_count = static_cast<size_t>(shape.batch_count) *
                                    shape.tokens * shape.in_dim;
    const size_t weight_count = static_cast<size_t>(shape.batch_count) *
                                shape.out_dim * shape.in_dim;
    const size_t output_count = static_cast<size_t>(shape.batch_count) *
                                shape.tokens * shape.out_dim;
    std::vector<__half> host_activations(activation_count);
    std::vector<__half> host_weights(weight_count);
    fill(host_activations, 0x1357u);
    fill(host_weights, 0x2468u);
    CUDA_CHECK(cudaMalloc(
            &buffers.activations, activation_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&buffers.weights, weight_count * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(
            &buffers.cutlass_output, output_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(
            &buffers.cublas_output, output_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
            buffers.activations, host_activations.data(),
            activation_count * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
            buffers.weights, host_weights.data(),
            weight_count * sizeof(__half), cudaMemcpyHostToDevice));
    return buffers;
}

void run_cublas_batched(
        cublasHandle_t handle,
        const BatchedShape &shape,
        const Buffers &buffers) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            shape.out_dim, shape.tokens, shape.in_dim,
            &alpha,
            buffers.weights, CUDA_R_16F, shape.in_dim,
            static_cast<long long>(shape.out_dim) * shape.in_dim,
            buffers.activations, CUDA_R_16F, shape.in_dim,
            static_cast<long long>(shape.tokens) * shape.in_dim,
            &beta,
            buffers.cublas_output, CUDA_R_32F, shape.out_dim,
            static_cast<long long>(shape.tokens) * shape.out_dim,
            shape.batch_count,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
}

void run_cutlass_batched(const BatchedShape &shape, const Buffers &buffers) {
    const int status = ds4x_cutlass_fp16_gemm_strided_batched(
            buffers.cutlass_output,
            reinterpret_cast<const uint16_t *>(buffers.activations),
            reinterpret_cast<const uint16_t *>(buffers.weights),
            shape.tokens, shape.in_dim, shape.out_dim, shape.batch_count,
            nullptr);
    if (status != 1) {
        std::fprintf(stderr, "%s: CUTLASS batched backend failed: %s (%d)\n",
                     shape.name, ds4x_cutlass_status_string(status), status);
        std::exit(1);
    }
}

void check_batched_parity(cublasHandle_t handle, const BatchedShape &shape) {
    Buffers buffers = allocate(shape);
    run_cublas_batched(handle, shape, buffers);
    run_cutlass_batched(shape, buffers);
    CUDA_CHECK(cudaDeviceSynchronize());

    const size_t count = static_cast<size_t>(shape.batch_count) *
                         shape.tokens * shape.out_dim;
    std::vector<float> cutlass_output(count);
    std::vector<float> cublas_output(count);
    CUDA_CHECK(cudaMemcpy(cutlass_output.data(), buffers.cutlass_output,
                          count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cublas_output.data(), buffers.cublas_output,
                          count * sizeof(float), cudaMemcpyDeviceToHost));
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float delta = std::fabs(cutlass_output[i] - cublas_output[i]);
        max_abs = std::max(max_abs, delta);
        max_rel = std::max(
                max_rel,
                delta / std::max(std::fabs(cublas_output[i]), 1.0e-6f));
    }
    std::printf("cutlass-parity: %-22s max_abs=%g max_rel=%g\n",
                shape.name, static_cast<double>(max_abs),
                static_cast<double>(max_rel));
    if (max_abs > 2.0e-3f && max_rel > 2.0e-3f) {
        std::fprintf(stderr, "%s: batched parity tolerance exceeded\n", shape.name);
        std::exit(1);
    }
    release(buffers);
}

void benchmark_batched(cublasHandle_t handle, const BatchedShape &shape) {
    Buffers buffers = allocate(shape);
    const int iterations = shape.tokens >= 1024 ? 30 : 100;
    const float cublas_ms = time_ms(
            [&] { run_cublas_batched(handle, shape, buffers); }, 10, iterations);
    const float cutlass_ms = time_ms(
            [&] { run_cutlass_batched(shape, buffers); }, 10, iterations);
    const double flops = 2.0 * shape.batch_count * shape.tokens *
                         shape.in_dim * shape.out_dim;
    std::printf(
            "cutlass-bench: %-22s cuBLAS=%8.4f ms (%5.1f TF/s) "
            "CUTLASS=%8.4f ms (%5.1f TF/s) ratio=%6.3f\n",
            shape.name,
            cublas_ms, flops / (static_cast<double>(cublas_ms) * 1.0e9),
            cutlass_ms, flops / (static_cast<double>(cutlass_ms) * 1.0e9),
            cutlass_ms / cublas_ms);
    release(buffers);
}

}  // namespace

int main(int argc, char **argv) {
    const bool run_benchmark = argc == 2 && std::strcmp(argv[1], "--benchmark") == 0;
    if (argc > 2 || (argc == 2 && !run_benchmark)) {
        std::fprintf(stderr, "usage: %s [--benchmark]\n", argv[0]);
        return 2;
    }
    const Shape parity_shapes[] = {
        {"router", 31, 4096, 256},
        {"compressor", 32, 4096, 256},
        {"indexer-weight", 32, 4096, 64},
        {"indexer-qb-128", 128, 1024, 8192},
        {"indexer-qb-256", 256, 1024, 8192},
        {"indexer-qb-1024", 1024, 1024, 8192},
        {"indexer-qb-production", 4096, 1024, 8192},
    };
    const Shape benchmark_shapes[] = {
        {"compressor-32", 32, 4096, 256},
        {"compressor-128", 128, 4096, 256},
        {"compressor-4096", 4096, 4096, 256},
        {"indexer-qb-128", 128, 1024, 8192},
        {"indexer-qb-256", 256, 1024, 8192},
        {"indexer-qb-512", 512, 1024, 8192},
        {"indexer-qb-1024", 1024, 1024, 8192},
        {"indexer-qb-2048", 2048, 1024, 8192},
        {"indexer-qb", 4096, 1024, 8192},
        {"attention-output", 4096, 4096, 8192},
    };
    const BatchedShape batched_parity_shapes[] = {
        {"attention-output-a", 31, 4096, 1024, 8},
    };
    const BatchedShape batched_benchmark_shapes[] = {
        {"attention-output-a-32", 32, 4096, 1024, 8},
        {"attention-output-a-128", 128, 4096, 1024, 8},
        {"attention-output-a-256", 256, 4096, 1024, 8},
        {"attention-output-a-512", 512, 4096, 1024, 8},
        {"attention-output-a-1024", 1024, 4096, 1024, 8},
        {"attention-output-a-4096", 4096, 4096, 1024, 8},
    };

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    if (run_benchmark) {
        for (const Shape &shape : benchmark_shapes) benchmark(handle, shape);
        for (const BatchedShape &shape : batched_benchmark_shapes) {
            benchmark_batched(handle, shape);
        }
    } else {
        for (const Shape &shape : parity_shapes) check_parity(handle, shape);
        for (const BatchedShape &shape : batched_parity_shapes) {
            check_batched_parity(handle, shape);
        }
        std::puts("CUTLASS FP16 GEMM regression: OK");
    }
    cublasDestroy(handle);
    return 0;
}
