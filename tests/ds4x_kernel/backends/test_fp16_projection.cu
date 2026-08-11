#include "ds4x/fp16_projection.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#define CUDA_CHECK(call) do { \
    const cudaError_t status_ = (call); \
    if (status_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA failure %s:%d: %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(status_)); \
        std::exit(1); \
    } \
} while (0)

namespace {

float sample(uint32_t i) {
    i = i * 1664525u + 1013904223u;
    return (static_cast<float>(i >> 8) / 16777216.0f - 0.5f) * 0.25f;
}

void require_launch(int status, const char *name) {
    if (status != static_cast<int>(cudaSuccess)) {
        std::fprintf(stderr, "%s launch failed: %s\n", name,
                     cudaGetErrorString(static_cast<cudaError_t>(status)));
        std::exit(1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void compare(const char *name, const std::vector<float> &actual,
             const std::vector<float> &expected, float tolerance = 2.0e-4f) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(actual[i] - expected[i]));
    }
    std::printf("fp16-projection: %-18s max_abs=%g\n",
                name, static_cast<double>(max_abs));
    if (max_abs > tolerance) std::exit(1);
}

std::vector<float> reference(
        const std::vector<__half> &weights,
        const std::vector<float> &activations,
        int tokens,
        int in_dim,
        int out_dim,
        bool round_activations = false) {
    std::vector<float> output(static_cast<size_t>(tokens) * out_dim);
    for (int token = 0; token < tokens; ++token) {
        for (int row = 0; row < out_dim; ++row) {
            float sum = 0.0f;
            for (int i = 0; i < in_dim; ++i) {
                float x = activations[static_cast<size_t>(token) * in_dim + i];
                if (round_activations) x = __half2float(__float2half(x));
                sum += __half2float(weights[static_cast<size_t>(row) * in_dim + i]) * x;
            }
            output[static_cast<size_t>(token) * out_dim + row] = sum;
        }
    }
    return output;
}

struct DeviceBuffers {
    __half *weights = nullptr;
    float *activations = nullptr;
    float *output = nullptr;
};

DeviceBuffers upload(const std::vector<__half> &weights,
                     const std::vector<float> &activations,
                     size_t output_count) {
    DeviceBuffers device;
    CUDA_CHECK(cudaMalloc(&device.weights, weights.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&device.activations, activations.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device.output, output_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device.weights, weights.data(), weights.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.activations, activations.data(),
                          activations.size() * sizeof(float), cudaMemcpyHostToDevice));
    return device;
}

void release(DeviceBuffers &device) {
    cudaFree(device.output);
    cudaFree(device.activations);
    cudaFree(device.weights);
}

void test_projection_modes() {
    constexpr int tokens = 3;
    constexpr int in_dim = 96;
    constexpr int out_dim = 17;
    std::vector<__half> weights(out_dim * in_dim);
    std::vector<float> activations(tokens * in_dim);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = __float2half(sample(i));
    for (size_t i = 0; i < activations.size(); ++i) activations[i] = sample(i + 10000u);
    const auto expected = reference(weights, activations, tokens, in_dim, out_dim);
    DeviceBuffers device = upload(weights, activations, expected.size());

    for (const auto &[name, mode] : std::vector<std::pair<const char *, int>>{
             {"parallel", DS4X_FP16_PROJECTION_PARALLEL},
             {"serial", DS4X_FP16_PROJECTION_SERIAL},
             {"ordered", DS4X_FP16_PROJECTION_ORDERED}}) {
        require_launch(ds4x_launch_fp16_projection(
                device.output, reinterpret_cast<const uint16_t *>(device.weights),
                device.activations, in_dim, out_dim, tokens, mode, nullptr), name);
        std::vector<float> actual(expected.size());
        CUDA_CHECK(cudaMemcpy(actual.data(), device.output, actual.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        compare(name, actual, expected);
    }
    release(device);
}

void test_small_outputs() {
    constexpr int tokens = 3;
    constexpr int in_dim = 4096;
    constexpr int out_dim = 8;
    std::vector<__half> weights(out_dim * in_dim);
    std::vector<float> activations(tokens * in_dim);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = __float2half(sample(i));
    for (size_t i = 0; i < activations.size(); ++i) activations[i] = sample(i + 20000u);
    DeviceBuffers device = upload(weights, activations, tokens * out_dim);

    require_launch(ds4x_launch_fp16_small_output_one(
            device.output, reinterpret_cast<const uint16_t *>(device.weights),
            device.activations, in_dim, out_dim, nullptr), "small-one");
    std::vector<float> actual_one(out_dim);
    CUDA_CHECK(cudaMemcpy(actual_one.data(), device.output, out_dim * sizeof(float),
                          cudaMemcpyDeviceToHost));
    auto expected_one = reference(weights, activations, 1, in_dim, out_dim, true);
    compare("small-one", actual_one, expected_one);

    require_launch(ds4x_launch_fp16_small_output_batch(
            device.output, reinterpret_cast<const uint16_t *>(device.weights),
            device.activations, in_dim, out_dim, tokens, nullptr), "small-batch");
    std::vector<float> actual_batch(tokens * out_dim);
    CUDA_CHECK(cudaMemcpy(actual_batch.data(), device.output,
                          actual_batch.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const auto expected_batch = reference(weights, activations, tokens, in_dim, out_dim);
    compare("small-batch", actual_batch, expected_batch, 5.0e-4f);
    release(device);
}

void test_pair() {
    constexpr int in_dim = 128;
    constexpr int out_a = 7;
    constexpr int out_b = 9;
    std::vector<__half> weights_a(out_a * in_dim);
    std::vector<__half> weights_b(out_b * in_dim);
    std::vector<float> activations(in_dim);
    for (size_t i = 0; i < weights_a.size(); ++i) weights_a[i] = __float2half(sample(i));
    for (size_t i = 0; i < weights_b.size(); ++i) weights_b[i] = __float2half(sample(i + 3000u));
    for (size_t i = 0; i < activations.size(); ++i) activations[i] = sample(i + 30000u);

    __half *device_a = nullptr;
    __half *device_b = nullptr;
    float *device_x = nullptr;
    float *device_out_a = nullptr;
    float *device_out_b = nullptr;
    CUDA_CHECK(cudaMalloc(&device_a, weights_a.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&device_b, weights_b.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&device_x, activations.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_out_a, out_a * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_out_b, out_b * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_a, weights_a.data(), weights_a.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, weights_b.data(), weights_b.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_x, activations.data(), activations.size() * sizeof(float), cudaMemcpyHostToDevice));
    require_launch(ds4x_launch_fp16_pair_ordered(
            device_out_a, device_out_b,
            reinterpret_cast<const uint16_t *>(device_a),
            reinterpret_cast<const uint16_t *>(device_b),
            device_x, in_dim, out_a, out_b, nullptr), "pair");

    std::vector<float> actual_a(out_a);
    std::vector<float> actual_b(out_b);
    CUDA_CHECK(cudaMemcpy(actual_a.data(), device_out_a, out_a * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(actual_b.data(), device_out_b, out_b * sizeof(float), cudaMemcpyDeviceToHost));
    compare("pair-a", actual_a, reference(weights_a, activations, 1, in_dim, out_a));
    compare("pair-b", actual_b, reference(weights_b, activations, 1, in_dim, out_b));
    cudaFree(device_out_b);
    cudaFree(device_out_a);
    cudaFree(device_x);
    cudaFree(device_b);
    cudaFree(device_a);
}

}  // namespace

int main() {
    test_projection_modes();
    test_small_outputs();
    test_pair();
    std::puts("FP16 projection regression: OK");
    return 0;
}
