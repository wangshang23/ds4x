// SPDX-License-Identifier: MIT
// Isolated B1 QKVO probe: production Q8_0 MMVQ versus native Blackwell FP4 MMQ.

#include "ds4_mmq.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr float kFp4[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

struct Q8Block {
    uint16_t d;
    int8_t qs[32];
};

struct MXFP4Block {
    uint8_t e;
    uint8_t qs[16];
};

struct NVFP4Block {
    uint8_t d[4];
    uint8_t qs[32];
};

static_assert(sizeof(Q8Block) == 34, "Q8_0 layout mismatch");
static_assert(sizeof(MXFP4Block) == 17, "MXFP4 layout mismatch");
static_assert(sizeof(NVFP4Block) == 36, "NVFP4 layout mismatch");

struct Shape {
    const char *name;
    int M;
    int K;
};

struct Metrics {
    double nrmse;
    double cosine;
    double max_abs;
};

using GemvFn = int (*)(const void *, const float *, float *, int, int, int, cudaStream_t);

bool cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    return false;
}

uint16_t f32_to_f16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = (int)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    mant += 0x1000u;
    if (mant & 0x800000u) {
        mant = 0;
        if (++exp >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

float e8m0_to_f32(uint8_t e) {
    if (e == 0) return std::ldexp(1.0f, -127);
    return std::ldexp(1.0f, (int)e - 127);
}

float ue4m3_raw_to_f32(uint8_t x) {
    if (x == 0 || x == 0x7f) return 0.0f;
    const int exp = (x >> 3) & 0xf;
    const int man = x & 0x7;
    if (exp == 0) return std::ldexp((float)man, -9);
    return std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
}

uint8_t f32_to_ue4m3(float x) {
    if (!(x > 0.0f)) return 0;
    x = std::min(x, 448.0f);
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    int exp = (int)((bits >> 23) & 0xffu) - 127 + 7;
    const int mant3 = (int)((bits >> 20) & 0x7u);
    if (exp <= 0) {
        return (uint8_t)std::max(0, std::min(7, (int)(x * 512.0f + 0.5f)));
    }
    if (exp >= 15) return 0x7e;
    int mant = mant3 + (int)((bits >> 19) & 1u);
    if (mant > 7) {
        mant = 0;
        if (++exp >= 15) return 0x7e;
    }
    return (uint8_t)((exp << 3) | mant);
}

uint8_t nearest_fp4(float value, float scale) {
    uint8_t best = 0;
    float best_err = std::fabs(value);
    for (uint8_t q = 1; q < 16; ++q) {
        const float err = std::fabs(value - scale * kFp4[q]);
        if (err < best_err) {
            best = q;
            best_err = err;
        }
    }
    return best;
}

void quantize_q8_row(const float *src, Q8Block *dst, int K) {
    for (int base = 0; base < K; base += 32) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(src[base + j]));
        const float d = amax / 127.0f;
        const float inv = d > 0.0f ? 1.0f / d : 0.0f;
        dst->d = f32_to_f16(d);
        for (int j = 0; j < 32; ++j) {
            const int q = (int)std::lrint(src[base + j] * inv);
            dst->qs[j] = (int8_t)std::max(-127, std::min(127, q));
        }
        ++dst;
    }
}

void quantize_mxfp4_row(const float *src, MXFP4Block *dst, int K) {
    for (int base = 0; base < K; base += 32) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(src[base + j]));
        int e = amax > 0.0f ? (int)std::floor(std::log2(amax)) - 2 + 127 : 0;
        e = std::max(0, std::min(254, e));
        dst->e = (uint8_t)e;
        const float d = e8m0_to_f32(dst->e);
        for (int j = 0; j < 16; ++j) {
            dst->qs[j] = nearest_fp4(src[base + j], d) |
                         (nearest_fp4(src[base + 16 + j], d) << 4);
        }
        ++dst;
    }
}

void quantize_nvfp4_row(const float *src, NVFP4Block *dst, int K) {
    for (int base = 0; base < K; base += 64) {
        for (int sub = 0; sub < 4; ++sub) {
            const float *x = src + base + sub * 16;
            float amax = 0.0f;
            for (int j = 0; j < 16; ++j) amax = std::max(amax, std::fabs(x[j]));
            dst->d[sub] = f32_to_ue4m3(amax / 6.0f);
            const float d = ue4m3_raw_to_f32(dst->d[sub]);
            for (int j = 0; j < 8; ++j) {
                dst->qs[sub * 8 + j] = nearest_fp4(x[j], d) |
                                            (nearest_fp4(x[8 + j], d) << 4);
            }
        }
        ++dst;
    }
}

Metrics compare(const std::vector<float> &got, const std::vector<double> &ref) {
    double diff2 = 0.0;
    double ref2 = 0.0;
    double got2 = 0.0;
    double dot = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = (double)got[i] - ref[i];
        diff2 += d * d;
        ref2 += ref[i] * ref[i];
        got2 += (double)got[i] * got[i];
        dot += (double)got[i] * ref[i];
        max_abs = std::max(max_abs, std::fabs(d));
    }
    return {std::sqrt(diff2 / std::max(ref2, 1e-30)),
            dot / std::sqrt(std::max(ref2 * got2, 1e-30)), max_abs};
}

double benchmark(GemvFn fn, const void *weights, size_t one_copy_bytes, int copies,
                 const float *x, float *out, int M, int N, int K, cudaStream_t stream) {
    for (int i = 0; i < std::min(copies, 8); ++i) {
        if (fn((const uint8_t *)weights + (size_t)i * one_copy_bytes,
               x, out, M, N, K, stream) != 0) return -1.0;
    }
    cudaStreamSynchronize(stream);

    std::vector<float> samples;
    samples.reserve(31);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 31; ++i) {
        const int copy = i % copies;
        cudaEventRecord(start, stream);
        if (fn((const uint8_t *)weights + (size_t)copy * one_copy_bytes,
               x, out, M, N, K, stream) != 0) return -1.0;
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        samples.push_back(ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

bool run_shape(const Shape &shape, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> weight_dist(0.0f, 0.02f);
    std::normal_distribution<float> act_dist(0.0f, 1.0f);

    std::vector<float> x(shape.K);
    for (float &v : x) v = act_dist(rng);
    std::vector<float> row(shape.K);
    std::vector<double> ref(shape.M, 0.0);
    std::vector<Q8Block> q8((size_t)shape.M * shape.K / 32);
    std::vector<MXFP4Block> mx((size_t)shape.M * shape.K / 32);
    std::vector<NVFP4Block> nv((size_t)shape.M * shape.K / 64);

    for (int m = 0; m < shape.M; ++m) {
        double sum = 0.0;
        for (int k = 0; k < shape.K; ++k) {
            row[k] = weight_dist(rng);
            sum += (double)row[k] * x[k];
        }
        ref[m] = sum;
        quantize_q8_row(row.data(), q8.data() + (size_t)m * shape.K / 32, shape.K);
        quantize_mxfp4_row(row.data(), mx.data() + (size_t)m * shape.K / 32, shape.K);
        quantize_nvfp4_row(row.data(), nv.data() + (size_t)m * shape.K / 64, shape.K);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    float *dx = nullptr;
    float *dy = nullptr;
    cudaMalloc(&dx, x.size() * sizeof(float));
    cudaMalloc(&dy, (size_t)shape.M * sizeof(float));
    cudaMemcpyAsync(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

    const size_t q8_bytes = q8.size() * sizeof(Q8Block);
    const size_t mx_bytes = mx.size() * sizeof(MXFP4Block);
    const size_t nv_bytes = nv.size() * sizeof(NVFP4Block);
    const size_t target_ring = 160ull << 20;
    const int q8_copies = std::max(1, std::min(32, (int)((target_ring + q8_bytes - 1) / q8_bytes)));
    const int mx_copies = std::max(1, std::min(32, (int)((target_ring + mx_bytes - 1) / mx_bytes)));
    const int nv_copies = std::max(1, std::min(32, (int)((target_ring + nv_bytes - 1) / nv_bytes)));
    void *dq8 = nullptr, *dmx = nullptr, *dnv = nullptr;
    cudaMalloc(&dq8, q8_bytes * q8_copies);
    cudaMalloc(&dmx, mx_bytes * mx_copies);
    cudaMalloc(&dnv, nv_bytes * nv_copies);
    for (int i = 0; i < q8_copies; ++i)
        cudaMemcpyAsync((uint8_t *)dq8 + (size_t)i * q8_bytes, q8.data(), q8_bytes, cudaMemcpyHostToDevice, stream);
    for (int i = 0; i < mx_copies; ++i)
        cudaMemcpyAsync((uint8_t *)dmx + (size_t)i * mx_bytes, mx.data(), mx_bytes, cudaMemcpyHostToDevice, stream);
    for (int i = 0; i < nv_copies; ++i)
        cudaMemcpyAsync((uint8_t *)dnv + (size_t)i * nv_bytes, nv.data(), nv_bytes, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> yq8(shape.M), ymx_vec(shape.M), ynv_vec(shape.M), ynv_tc(shape.M);
    ds4_mmq_q8_0_dense_vec(dq8, dx, dy, shape.M, 1, shape.K, stream);
    cudaMemcpyAsync(yq8.data(), dy, yq8.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    ds4_mmq_mxfp4_dense_vec(dmx, dx, dy, shape.M, 1, shape.K, stream);
    cudaMemcpyAsync(ymx_vec.data(), dy, ymx_vec.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    ds4_mmq_nvfp4_dense_vec(dnv, dx, dy, shape.M, 1, shape.K, stream);
    cudaMemcpyAsync(ynv_vec.data(), dy, ynv_vec.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    ds4_mmq_nvfp4_dense(dnv, dx, dy, shape.M, 1, shape.K, stream);
    cudaMemcpyAsync(ynv_tc.data(), dy, ynv_tc.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (!cuda_ok(cudaStreamSynchronize(stream), "accuracy sync")) return false;

    const Metrics eq8 = compare(yq8, ref);
    const Metrics emx_vec = compare(ymx_vec, ref);
    const Metrics env_vec = compare(ynv_vec, ref);
    const Metrics env_tc = compare(ynv_tc, ref);
    const double tq8 = benchmark(ds4_mmq_q8_0_dense_vec, dq8, q8_bytes, q8_copies,
                                 dx, dy, shape.M, 1, shape.K, stream);
    const double tmx_vec = benchmark(ds4_mmq_mxfp4_dense_vec, dmx, mx_bytes, mx_copies,
                                     dx, dy, shape.M, 1, shape.K, stream);
    const double tnv_vec = benchmark(ds4_mmq_nvfp4_dense_vec, dnv, nv_bytes, nv_copies,
                                     dx, dy, shape.M, 1, shape.K, stream);
    const double tnv_tc = benchmark(ds4_mmq_nvfp4_dense, dnv, nv_bytes, nv_copies,
                                    dx, dy, shape.M, 1, shape.K, stream);

    std::printf("%-4s M=%-5d K=%-5d | Q8v %7.4f ms %6.1f GB/s nrmse=%7.4f cos=%8.6f\n"
                "                       | MXv %7.4f ms %6.1f GB/s nrmse=%7.4f cos=%8.6f | "
                "NVv %7.4f ms %6.1f GB/s nrmse=%7.4f cos=%8.6f | "
                "NVtc %7.4f ms %6.1f GB/s nrmse=%7.4f cos=%8.6f\n",
                shape.name, shape.M, shape.K,
                tq8, q8_bytes / (1e6 * tq8), eq8.nrmse, eq8.cosine,
                tmx_vec, mx_bytes / (1e6 * tmx_vec), emx_vec.nrmse, emx_vec.cosine,
                tnv_vec, nv_bytes / (1e6 * tnv_vec), env_vec.nrmse, env_vec.cosine,
                tnv_tc, nv_bytes / (1e6 * tnv_tc), env_tc.nrmse, env_tc.cosine);

    if (q8_bytes >= (32ull << 20)) {
        constexpr int max_n = 8;
        std::vector<float> xb((size_t)max_n * shape.K);
        for (float &v : xb) v = act_dist(rng);
        float *dxb = nullptr;
        float *dyb = nullptr;
        cudaMalloc(&dxb, xb.size() * sizeof(float));
        cudaMalloc(&dyb, (size_t)max_n * shape.M * sizeof(float));
        cudaMemcpyAsync(dxb, xb.data(), xb.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        for (int n : {2, 4, 6, 8}) {
            const double q8n = benchmark(ds4_mmq_q8_0_dense_vec, dq8, q8_bytes, q8_copies,
                                         dxb, dyb, shape.M, n, shape.K, stream);
            const double nvv = benchmark(ds4_mmq_nvfp4_dense_vec, dnv, nv_bytes, nv_copies,
                                         dxb, dyb, shape.M, n, shape.K, stream);
            const double nvt = benchmark(ds4_mmq_nvfp4_dense, dnv, nv_bytes, nv_copies,
                                         dxb, dyb, shape.M, n, shape.K, stream);
            std::printf("                       | N=%d Q8v=%7.4f ms NVv=%7.4f ms NVtc=%7.4f ms\n",
                        n, q8n, nvv, nvt);
        }
        cudaFree(dxb);
        cudaFree(dyb);
    }

    cudaFree(dq8);
    cudaFree(dmx);
    cudaFree(dnv);
    cudaFree(dx);
    cudaFree(dy);
    cudaStreamDestroy(stream);
    return tq8 > 0.0 && tmx_vec > 0.0 && tnv_vec > 0.0 && tnv_tc > 0.0;
}

} // namespace

int main() {
    if (ds4_mmq_init(0) != 0) return 1;
    const Shape shapes[] = {
        {"Q_a",  1024, 4096},
        {"KV",    512, 4096},
        {"Q_b", 32768, 1024},
        {"O_a",  8192, 4096},
        {"O_b",  4096, 8192},
    };
    bool ok = true;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i) {
        ok &= run_shape(shapes[i], 0xD54F400u + (uint32_t)i);
    }
    return ok ? 0 : 1;
}
