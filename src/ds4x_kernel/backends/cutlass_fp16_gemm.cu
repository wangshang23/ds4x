#include "ds4x/cutlass_fp16_gemm.h"

#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cstdlib>
#include <cstring>
#include <limits>

namespace ds4x::kernel {
namespace {

using ElementInput = cutlass::half_t;
using ElementOutput = float;
using ElementAccumulator = float;

/* GB10 executes FP16 through warp-level mma.sync rather than the SM100
 * tcgen05 path. CUTLASS's SM80 TensorOp kernel is forward-compatible with
 * sm_121 and gives us an explicit, composable kernel without pretending that
 * SM120's low-precision-only CollectiveBuilder supports FP16. */
using Epilogue = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementAccumulator>;

using Gemm = cutlass::gemm::device::Gemm<
        ElementInput,
        cutlass::layout::RowMajor,
        ElementInput,
        cutlass::layout::ColumnMajor,
        ElementOutput,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        Epilogue,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3,
        8,
        8>;

template <typename ThreadblockShape, typename WarpShape, int Stages>
using BatchedGemmKernel = cutlass::gemm::device::GemmBatched<
        ElementInput,
        cutlass::layout::RowMajor,
        ElementInput,
        cutlass::layout::ColumnMajor,
        ElementOutput,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        cutlass::gemm::GemmShape<16, 8, 16>,
        Epilogue,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        Stages,
        8,
        8>;

using BatchedGemm128x128 = BatchedGemmKernel<
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        3>;
using BatchedGemm128x64 = BatchedGemmKernel<
        cutlass::gemm::GemmShape<128, 64, 32>,
        cutlass::gemm::GemmShape<64, 32, 32>,
        3>;
using BatchedGemm64x128 = BatchedGemmKernel<
        cutlass::gemm::GemmShape<64, 128, 32>,
        cutlass::gemm::GemmShape<32, 64, 32>,
        3>;

using ProblemShape = cute::Shape<int32_t, int32_t, int32_t>;

int map_status(cutlass::Status status);

bool dimensions_fit(uint64_t tokens, uint64_t in_dim, uint64_t out_dim) {
    constexpr uint64_t max = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    return tokens > 0 && in_dim > 0 && out_dim > 0 &&
           tokens <= max && in_dim <= max && out_dim <= max;
}

bool batch_fits(uint32_t batch_count) {
    return batch_count > 0 &&
           batch_count <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
}

bool aligned(const void *ptr, uintptr_t bytes) {
    return ptr && (reinterpret_cast<uintptr_t>(ptr) % bytes) == 0;
}

Gemm::Arguments make_arguments(
        float *output,
        const uint16_t *activations,
        const uint16_t *weights,
        uint64_t tokens,
        uint64_t in_dim,
        uint64_t out_dim) {
    const ProblemShape problem{
            static_cast<int32_t>(tokens),
            static_cast<int32_t>(out_dim),
            static_cast<int32_t>(in_dim)};
    const int m = cute::get<0>(problem);
    const int n = cute::get<1>(problem);
    const int k = cute::get<2>(problem);
    const auto *a = reinterpret_cast<const ElementInput *>(activations);
    const auto *b = reinterpret_cast<const ElementInput *>(weights);
    return Gemm::Arguments{
            {m, n, k},
            {a, k},
            {b, k},
            {output, n},
            {output, n},
            {1.0f, 0.0f}};
}

template <typename BatchedGemm>
typename BatchedGemm::Arguments make_batched_arguments(
        float *output,
        const uint16_t *activations,
        const uint16_t *weights,
        uint64_t tokens,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t batch_count) {
    const ProblemShape problem{
            static_cast<int32_t>(tokens),
            static_cast<int32_t>(out_dim),
            static_cast<int32_t>(in_dim)};
    const int m = cute::get<0>(problem);
    const int n = cute::get<1>(problem);
    const int k = cute::get<2>(problem);
    const auto *a = reinterpret_cast<const ElementInput *>(activations);
    const auto *b = reinterpret_cast<const ElementInput *>(weights);
    return typename BatchedGemm::Arguments{
            {m, n, k},
            {a, k},
            static_cast<int64_t>(tokens * in_dim),
            {b, k},
            static_cast<int64_t>(out_dim * in_dim),
            {output, n},
            static_cast<int64_t>(tokens * out_dim),
            {output, n},
            static_cast<int64_t>(tokens * out_dim),
            {1.0f, 0.0f},
            static_cast<int>(batch_count)};
}


template <typename BatchedGemm>
int batched_supported(
        float *output,
        const uint16_t *activations,
        const uint16_t *weights,
        uint64_t tokens,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t batch_count) {
    auto args = make_batched_arguments<BatchedGemm>(
            output, activations, weights, tokens, in_dim, out_dim,
            batch_count);
    return map_status(BatchedGemm::can_implement(args)) == 1;
}

template <typename BatchedGemm>
int batched_launch(
        float *output,
        const uint16_t *activations,
        const uint16_t *weights,
        uint64_t tokens,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t batch_count,
        cudaStream_t stream) {
    auto args = make_batched_arguments<BatchedGemm>(
            output, activations, weights, tokens, in_dim, out_dim,
            batch_count);
    BatchedGemm gemm;
    return map_status(gemm(args, nullptr, stream));
}

int batched_variant() {
    const char *variant = std::getenv("DS4X_CUTLASS_BATCHED_TILE");
    if (!variant || !variant[0] || std::strcmp(variant, "128x128") == 0) {
        return 0;
    }
    if (std::strcmp(variant, "128x64") == 0) return 1;
    if (std::strcmp(variant, "64x128") == 0) return 2;
    return 0;
}

int map_status(cutlass::Status status) {
    if (status == cutlass::Status::kSuccess) return 1;
    if (status == cutlass::Status::kErrorMisalignedOperand ||
        status == cutlass::Status::kErrorInvalidProblem ||
        status == cutlass::Status::kErrorNotSupported) {
        return 0;
    }
    return -static_cast<int>(status) - 1;
}

}  // namespace
}  // namespace ds4x::kernel

extern "C" int ds4x_cutlass_fp16_gemm_supported(
        const float *output,
        const uint16_t *activations,
        const uint16_t *weights,
        uint64_t tokens,
        uint64_t in_dim,
        uint64_t out_dim) {
    using namespace ds4x::kernel;
    if (!dimensions_fit(tokens, in_dim, out_dim) ||
        (in_dim % 8u) != 0u || !aligned(output, 16u) ||
        !aligned(activations, 16u) || !aligned(weights, 16u)) {
        return 0;
    }
    auto args = make_arguments(
            const_cast<float *>(output), activations, weights,
            tokens, in_dim, out_dim);
    return map_status(Gemm::can_implement(args)) == 1;
}

extern "C" int ds4x_cutlass_fp16_gemm(
        float *output,
        const uint16_t *activations,
        const uint16_t *weights,
        uint64_t tokens,
        uint64_t in_dim,
        uint64_t out_dim,
        void *stream) {
    using namespace ds4x::kernel;
    if (!ds4x_cutlass_fp16_gemm_supported(
            output, activations, weights, tokens, in_dim, out_dim)) {
        return 0;
    }
    auto args = make_arguments(
            output, activations, weights, tokens, in_dim, out_dim);
    Gemm gemm;
    const cutlass::Status status = gemm(
            args, nullptr, reinterpret_cast<cudaStream_t>(stream));
    return map_status(status);
}

extern "C" int ds4x_cutlass_fp16_gemm_strided_batched_supported(
        const float *output,
        const uint16_t *activations,
        const uint16_t *weights,
        uint64_t tokens,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t batch_count) {
    using namespace ds4x::kernel;
    if (!dimensions_fit(tokens, in_dim, out_dim) || !batch_fits(batch_count) ||
        (in_dim % 8u) != 0u || !aligned(output, 16u) ||
        !aligned(activations, 16u) || !aligned(weights, 16u)) {
        return 0;
    }
    switch (batched_variant()) {
        case 1:
            return batched_supported<BatchedGemm128x64>(
                    const_cast<float *>(output), activations, weights,
                    tokens, in_dim, out_dim, batch_count);
        case 2:
            return batched_supported<BatchedGemm64x128>(
                    const_cast<float *>(output), activations, weights,
                    tokens, in_dim, out_dim, batch_count);
        default:
            return batched_supported<BatchedGemm128x128>(
                    const_cast<float *>(output), activations, weights,
                    tokens, in_dim, out_dim, batch_count);
    }
}

extern "C" int ds4x_cutlass_fp16_gemm_strided_batched(
        float *output,
        const uint16_t *activations,
        const uint16_t *weights,
        uint64_t tokens,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t batch_count,
        void *stream) {
    using namespace ds4x::kernel;
    if (!ds4x_cutlass_fp16_gemm_strided_batched_supported(
            output, activations, weights, tokens, in_dim, out_dim,
            batch_count)) {
        return 0;
    }
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (batched_variant()) {
        case 1:
            return batched_launch<BatchedGemm128x64>(
                    output, activations, weights, tokens, in_dim, out_dim,
                    batch_count, cuda_stream);
        case 2:
            return batched_launch<BatchedGemm64x128>(
                    output, activations, weights, tokens, in_dim, out_dim,
                    batch_count, cuda_stream);
        default:
            return batched_launch<BatchedGemm128x128>(
                    output, activations, weights, tokens, in_dim, out_dim,
                    batch_count, cuda_stream);
    }
}

extern "C" const char *ds4x_cutlass_status_string(int status) {
    if (status == 1) return "success";
    if (status == 0) return "unsupported";
    return "cutlass launch error";
}
