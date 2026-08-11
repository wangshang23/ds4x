#include "ds4_gpu.h"
#include "ds4x/ops/attention.h"
#include "ds4x/ops/cache.h"
#include "ds4x/ops/decode_graph.h"
#include "ds4x/ops/hyper_connection.h"
#include "ds4x/ops/indexer.h"
#include "ds4x/ops/moe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int check_cache_adapter() {
    constexpr uint32_t rows = 4;
    constexpr uint32_t width = 128;
    std::vector<float> source(rows * width);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<float>((i * 17u) % 101u) / 31.0f - 1.5f;
    }

    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(source.size() * sizeof(float));
    ds4_gpu_tensor *direct = ds4_gpu_tensor_alloc(rows * DS4_SPARK_INDEX_ROW_BYTES);
    ds4_gpu_tensor *adapted = ds4_gpu_tensor_alloc(rows * DS4_SPARK_INDEX_ROW_BYTES);
    if (!input || !direct || !adapted ||
        !ds4_gpu_tensor_write(input, 0, source.data(), source.size() * sizeof(float)) ||
        !ds4_gpu_spark_pack_index_rows_tensor(direct, 0, input, 0, rows)) {
        return 1;
    }

    const ds4x_runtime_context context = ds4x_default_runtime_context();
    const ds4x_cache_args args = {
        DS4X_CACHE_INDEXER, DS4X_CACHE_PACK, adapted, input, 0, 0, rows,
    };
    if (!ds4x_cache_supports(&context, &args) ||
        ds4x_cache_workspace_bytes(&args) != 0 ||
        !ds4x_cache_launch(&context, &args) ||
        !ds4_gpu_synchronize()) {
        return 1;
    }

    std::vector<unsigned char> direct_bytes(rows * DS4_SPARK_INDEX_ROW_BYTES);
    std::vector<unsigned char> adapted_bytes(direct_bytes.size());
    const int ok = ds4_gpu_tensor_read(
                       direct, 0, direct_bytes.data(), direct_bytes.size()) &&
                   ds4_gpu_tensor_read(
                       adapted, 0, adapted_bytes.data(), adapted_bytes.size()) &&
                   direct_bytes == adapted_bytes;

    ds4_gpu_tensor_free(adapted);
    ds4_gpu_tensor_free(direct);
    ds4_gpu_tensor_free(input);
    if (!ok) std::fprintf(stderr, "operator-adapter: packed cache mismatch\n");
    return ok ? 0 : 1;
}

int check_contract_guards() {
    const ds4x_runtime_context context = ds4x_default_runtime_context();
    ds4x_runtime_context unsupported = context;
    unsupported.stream = reinterpret_cast<void *>(1);

    ds4x_indexer_args indexer{};
    ds4x_attention_decode_args attention{};
    ds4x_routed_moe_args moe{};
    ds4x_output_hc_args hc{};
    ds4x_decode_graph_args graph{};
    if (ds4x_indexer_supports(&context, &indexer) ||
        ds4x_attention_decode_supports(&context, &attention) ||
        ds4x_routed_moe_supports(&context, &moe) ||
        ds4x_output_hc_supports(&context, &hc) ||
        ds4x_decode_graph_supports(&unsupported, &graph)) {
        std::fprintf(stderr, "operator-adapter: invalid contract accepted\n");
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    if (!ds4_gpu_init()) return 1;
    const int rc = check_cache_adapter() || check_contract_guards();
    ds4_gpu_cleanup();
    if (rc == 0) std::fprintf(stderr, "DS4X operator adapters: OK\n");
    return rc;
}
