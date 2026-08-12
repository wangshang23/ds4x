#include "../internal/backend_internal.cuh"

/* Moe Quantized implementation. */


// perf-04: launch-geometry tuning for the routed-MoE gate/up decode kernels
// (moe_gate_up_mid_qwarp32 / _decode_lut_qwarp32 / _decode_q4K_qwarp32). Each
// block processes MOE_DECODE_ROW_TILES tiles of 32 rows (row_lane in [0,32)).
// The historical value was 4 (128 rows/block -> ~96 blocks, occupancy ~16%,
// "grid too small to fill the device"). Lowering it issues correspondingly more
// blocks (e.g. 1 tile -> 32 rows/block -> ~4x more blocks -> ~384) to fill the
// SMs. The per-row arithmetic is identical regardless of this value, so output
// is bit-identical; only the qgrid.x divisor must match MOE_DECODE_ROWS_PER_BLOCK.
void routed_moe_decode_graph_destroy_one(int logical_tier) {
    if (logical_tier < 0 || logical_tier >= DS4_MAX_GPUS) return;
    cuda_moe_decode_graph_cache *c = &g_moe_decode_graph[logical_tier];
    if (c->exec) (void)cudaGraphExecDestroy(c->exec);
    if (c->graph) (void)cudaGraphDestroy(c->graph);
    memset(c, 0, sizeof(*c));
}

int routed_moe_decode_q4_graph_launch(
        int logical_tier,
        float *out,
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_w,
        const char *up_w,
        const char *down_w,
        cuda_block_q8_K *xq,
        cuda_block_q8_K *midq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp,
        const float *x) {
    if (logical_tier < 0 || logical_tier >= DS4_MAX_GPUS) return 0;
    if (n_expert != 3u && n_expert != 6u) return 0;
    uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    if (xq_blocks == 0u || midq_blocks == 0u) return 0;

    cuda_moe_decode_graph_cache *c = &g_moe_decode_graph[logical_tier];
    const bool shape_match =
        c->valid &&
        c->n_expert == n_expert &&
        c->expert_in_dim == expert_in_dim &&
        c->expert_mid_dim == expert_mid_dim &&
        c->out_dim == out_dim;
    if (c->valid && !shape_match) {
        routed_moe_decode_graph_destroy_one(logical_tier);
        c = &g_moe_decode_graph[logical_tier];
    }

    uint32_t x_rows = 1u;
    uint32_t mid_rows = n_expert;
    dim3 xq_grid(xq_blocks, 1, 1);
    dim3 gate_grid((expert_mid_dim + 7u) / 8u, n_expert, 1);
    dim3 midq_grid(midq_blocks, n_expert, 1);
    dim3 down_grid((out_dim + 31u) / 32u, 1, 1);
    dim3 block(256, 1, 1);

    void *xq_args[] = { &xq, &x, &expert_in_dim, &x_rows };
    cudaKernelNodeParams xq_params;
    memset(&xq_params, 0, sizeof(xq_params));
    xq_params.func = (void *)q8_K_quantize_kernel;
    xq_params.gridDim = xq_grid;
    xq_params.blockDim = block;
    xq_params.kernelParams = xq_args;

    void *gate_args[] = {
        &gate_out, &up_out, &mid_out, &gate_w, &up_w, &xq, &selected,
        &weights, &gate_expert_bytes, &gate_row_bytes, &xq_blocks,
        &expert_mid_dim, &n_expert, &write_aux, &clamp
    };
    cudaKernelNodeParams gate_params;
    memset(&gate_params, 0, sizeof(gate_params));
    gate_params.func = (void *)moe_gate_up_mid_decode_q4K_warp32_kernel;
    gate_params.gridDim = gate_grid;
    gate_params.blockDim = block;
    gate_params.kernelParams = gate_args;

    void *midq_args[] = { &midq, &mid_out, &expert_mid_dim, &mid_rows };
    cudaKernelNodeParams midq_params;
    memset(&midq_params, 0, sizeof(midq_params));
    midq_params.func = (void *)q8_K_quantize_kernel;
    midq_params.gridDim = midq_grid;
    midq_params.blockDim = block;
    midq_params.kernelParams = midq_args;

    void *down_args[] = {
        &out, &down_w, &midq, &selected, &down_expert_bytes,
        &down_row_bytes, &midq_blocks, &out_dim
    };
    cudaKernelNodeParams down_params;
    memset(&down_params, 0, sizeof(down_params));
    down_params.func = n_expert == 6u
        ? (void *)moe_down_q4K_sum6_qwarp32_kernel
        : (void *)moe_down_q4K_sum3_qwarp32_kernel;
    down_params.gridDim = down_grid;
    down_params.blockDim = block;
    down_params.kernelParams = down_args;

    if (!c->valid) {
        cudaError_t err = cudaGraphCreate(&c->graph, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: routed MoE decode graph create failed: %s\n",
                    cudaGetErrorString(err));
            routed_moe_decode_graph_destroy_one(logical_tier);
            return -1;
        }
        err = cudaGraphAddKernelNode(&c->xq_node, c->graph, NULL, 0,
                                     &xq_params);
        if (err == cudaSuccess) {
            err = cudaGraphAddKernelNode(&c->gate_node, c->graph,
                                         &c->xq_node, 1, &gate_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphAddKernelNode(&c->midq_node, c->graph,
                                         &c->gate_node, 1, &midq_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphAddKernelNode(&c->down_node, c->graph,
                                         &c->midq_node, 1, &down_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphInstantiate(&c->exec, c->graph, NULL, NULL, 0);
        }
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: routed MoE decode graph instantiate failed: %s\n",
                    cudaGetErrorString(err));
            routed_moe_decode_graph_destroy_one(logical_tier);
            return -1;
        }
        c->n_expert = n_expert;
        c->expert_in_dim = expert_in_dim;
        c->expert_mid_dim = expert_mid_dim;
        c->out_dim = out_dim;
        c->valid = 1;
    } else {
        cudaError_t err =
            cudaGraphExecKernelNodeSetParams(c->exec, c->xq_node,
                                             &xq_params);
        if (err == cudaSuccess) {
            err = cudaGraphExecKernelNodeSetParams(c->exec, c->gate_node,
                                                   &gate_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphExecKernelNodeSetParams(c->exec, c->midq_node,
                                                   &midq_params);
        }
        if (err == cudaSuccess) {
            err = cudaGraphExecKernelNodeSetParams(c->exec, c->down_node,
                                                   &down_params);
        }
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: routed MoE decode graph update failed: %s\n",
                    cudaGetErrorString(err));
            routed_moe_decode_graph_destroy_one(logical_tier);
            return -1;
        }
    }

    cudaError_t err = cudaGraphLaunch(c->exec, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: routed MoE decode graph launch failed: %s\n",
                cudaGetErrorString(err));
        routed_moe_decode_graph_destroy_one(logical_tier);
        return -1;
    }
    return 1;
}
