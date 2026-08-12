extern "C" int ds4_mmq_q8_0_moe(

#pragma once
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_Q8_0>("ds4_mmq_q8_0_moe", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_q2_K_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_Q2_K>("ds4_mmq_q2_K_moe", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_IQ2_XXS>("ds4_mmq_iq2_xxs_moe", W, X, ids, out, M, K,
                                               n_tokens, n_experts, n_expert_used, stream);
}

/* ds4 (P4 Inc3): mmq MoE over the aligned row-pair-SoA Q2_K artifact
 * (weight server --repack-q2k-aligned) -- no raw-layout weights and no
 * derepack scratch involved; the mul_mat_q tile loader reads the SoA
 * sections directly (load_tiles_q2_K_soa, bit-identical tiles). */
extern "C" int ds4_mmq_q2_K_moe_soa(
        const void * W_soa, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    if (M <= 0 || M % 2 != 0 || K <= 0 || K % 256 != 0 || n_experts <= 0) {
        fprintf(stderr, "ds4_mmq_q2_K_moe_soa: bad shape M=%d K=%d nexp=%d\n", M, K, n_experts);
        return -1;
    }
    const int64_t npair = (int64_t)n_experts * (int64_t)(M/2) * (int64_t)(K/256);
    /* W_soa doubles as the (unused) raw pointer so the impl's null checks
     * hold.  sanitize_out=false: the routed-MoE consumers (swiglu / moe_sum)
     * sanitize at read, saving the whole-buffer pass (P3). */
    return ds4_mmq_moe_impl<GGML_TYPE_Q2_K>("ds4_mmq_q2_K_moe_soa", W_soa, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream,
                                            (const char *)W_soa, npair,
                                            /*sanitize_out=*/false);
}

extern "C" int ds4_mmq_q4_K_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_Q4_K>("ds4_mmq_q4_K_moe", W, X, ids, out, M, K,
                                            n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_mxfp4_moe(
        const void * W, const float * X, const int32_t * ids, float * out,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_impl<GGML_TYPE_MXFP4>("ds4_mmq_mxfp4_moe", W, X, ids, out, M, K,
                                             n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_iq2_xxs_moe_pair(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

/* ds4 (P4 Inc3): paired mmq MoE over the aligned-SoA IQ2_XXS gate/up
 * artifacts (weight server --repack-iq2-aligned); same contract as
 * ds4_mmq_q2_K_moe_soa. */
extern "C" int ds4_mmq_iq2_xxs_moe_pair_soa(
        const void * Wa_soa, const void * Wb_soa,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    if (M <= 0 || K <= 0 || K % 256 != 0 || n_experts <= 0) {
        fprintf(stderr, "ds4_mmq_iq2_xxs_moe_pair_soa: bad shape M=%d K=%d nexp=%d\n", M, K, n_experts);
        return -1;
    }
    const int64_t nblk = (int64_t)n_experts * (int64_t)M * (int64_t)(K/256);
    /* sanitize_out=false: see ds4_mmq_q2_K_moe_soa. */
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS>(
        "ds4_mmq_iq2_xxs_moe_pair_soa", Wa_soa, Wb_soa, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream,
        (const char *)Wa_soa, (const char *)Wb_soa, nblk,
        /*sanitize_out=*/false);
}

/* v0.5 inc-9 (F7, derived from Marco Palaferri's GB10 fork, MIT): fused
 * target-prefill MoE pipeline over the aligned-SoA artifacts.  One
 * mm_ids_helper + one input quantize serve gate/up AND down; clamp + SwiGLU +
 * router weighting run in the pair-major mid buffer, which is gathered and
 * quantized for the Q2_K down MMQ through the same ids_dst/expert_bounds.
 * gate/up/mid/down keep the standard pair-major output layout. */
extern "C" int ds4_mmq_iq2_xxs_q2_K_moe_fused_soa(
        const void * W_gate, const void * W_up, const void * W_down,
        const float * X, const int32_t * ids, const float * router_weights,
        float * gate, float * up, float * mid_f32, float * down,
        int expert_mid_dim, int expert_in_dim, int out_dim,
        int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    if (expert_mid_dim <= 0 || expert_in_dim <= 0 || out_dim <= 0 ||
        n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0 ||
        n_expert_used > n_experts || expert_in_dim % 256 != 0 ||
        expert_mid_dim % 256 != 0 || out_dim % 2 != 0) {
        return -1;
    }
    const int64_t iq2_blocks =
        (int64_t)n_experts * expert_mid_dim * (expert_in_dim / 256);
    const int64_t q2_pairs =
        (int64_t)n_experts * (out_dim / 2) * (expert_mid_dim / 256);
    const ds4_mmq_fused_down fused_down = {
        W_down,
        (const char *)W_down,
        q2_pairs,
        router_weights,
        mid_f32,
        down,
        out_dim,
        clamp,
        false,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
    };
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS, true>(
        "ds4_mmq_iq2_xxs_q2_K_moe_fused_soa",
        W_gate, W_up, X, ids, gate, up,
        expert_mid_dim, expert_in_dim, n_tokens, n_experts, n_expert_used,
        stream,
        (const char *)W_gate, (const char *)W_up, iq2_blocks,
        /*sanitize_out=*/false, &fused_down);
}

/* Aligned-artifact production fast path: gate/up accumulators stay in
 * registers, weighted SwiGLU is quantized directly into down_q8_scratch by
 * the fused D2R kernel, and only the pair-major down output is materialized.
 * Caller-owned scratch keeps the hot path free of stream-ordered pool
 * allocations; all three ranges must be distinct and sized to their LOGICAL
 * segments (never an owning arena's capacity - the overlap guard would
 * falsely cover adjacent views). */
extern "C" int ds4_mmq_iq2_xxs_q2_K_moe_fused_direct_soa(
        const void * W_gate, const void * W_up, const void * W_down,
        const float * X, const int32_t * ids, const float * router_weights,
        void * input_q8_scratch, size_t input_q8_scratch_bytes,
        void * down_q8_scratch, size_t down_q8_scratch_bytes,
        void * work_scratch, size_t work_scratch_bytes,
        const void * input_q8_ext, size_t input_q8_ext_bytes,
        float * down,
        int expert_mid_dim, int expert_in_dim, int out_dim,
        int n_tokens, int n_experts, int n_expert_used,
        float clamp, cudaStream_t stream) {
    if (expert_mid_dim <= 0 || expert_in_dim <= 0 || out_dim <= 0 ||
        n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0 ||
        n_expert_used > n_experts || expert_in_dim % 256 != 0 ||
        expert_mid_dim % 256 != 0 || out_dim % 2 != 0 ||
        !input_q8_scratch || input_q8_scratch_bytes == 0 ||
        !down_q8_scratch || down_q8_scratch_bytes == 0 ||
        !work_scratch || work_scratch_bytes == 0 || !down) {
        return -1;
    }
    const size_t down_bytes =
        (size_t)n_tokens * (size_t)n_expert_used *
        (size_t)out_dim * sizeof(float);
    if (ds4_mmq_scratch_overlaps(
            input_q8_scratch, input_q8_scratch_bytes,
            down_q8_scratch, down_q8_scratch_bytes) ||
        ds4_mmq_scratch_overlaps(
            input_q8_scratch, input_q8_scratch_bytes,
            work_scratch, work_scratch_bytes) ||
        ds4_mmq_scratch_overlaps(
            down_q8_scratch, down_q8_scratch_bytes,
            work_scratch, work_scratch_bytes) ||
        ds4_mmq_scratch_overlaps(
            input_q8_scratch, input_q8_scratch_bytes, down, down_bytes) ||
        ds4_mmq_scratch_overlaps(
            down_q8_scratch, down_q8_scratch_bytes, down, down_bytes) ||
        ds4_mmq_scratch_overlaps(
            work_scratch, work_scratch_bytes, down, down_bytes)) {
        return -1;
    }
    const int64_t iq2_blocks =
        (int64_t)n_experts * expert_mid_dim * (expert_in_dim / 256);
    const int64_t q2_pairs =
        (int64_t)n_experts * (out_dim / 2) * (expert_mid_dim / 256);
    const ds4_mmq_fused_down fused_down = {
        W_down,
        (const char *)W_down,
        q2_pairs,
        router_weights,
        nullptr,
        down,
        out_dim,
        clamp,
        true,
        input_q8_scratch,
        input_q8_scratch_bytes,
        down_q8_scratch,
        down_q8_scratch_bytes,
        work_scratch,
        work_scratch_bytes,
        input_q8_ext,
        input_q8_ext_bytes,
    };
    return ds4_mmq_moe_pair_impl<GGML_TYPE_IQ2_XXS, true>(
        "ds4_mmq_iq2_xxs_q2_K_moe_fused_direct_soa",
        W_gate, W_up, X, ids, nullptr, nullptr,
        expert_mid_dim, expert_in_dim, n_tokens, n_experts, n_expert_used,
        stream,
        (const char *)W_gate, (const char *)W_up, iq2_blocks,
        /*sanitize_out=*/false, &fused_down);
}

extern "C" int ds4_mmq_q4_K_moe_pair(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_impl<GGML_TYPE_Q4_K>(
        "ds4_mmq_q4_K_moe_pair", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

extern "C" int ds4_mmq_mxfp4_moe_pair(
        const void * W_a, const void * W_b,
        const float * X, const int32_t * ids, float * out_a, float * out_b,
        int M, int K, int n_tokens, int n_experts, int n_expert_used,
        cudaStream_t stream) {
    return ds4_mmq_moe_pair_impl<GGML_TYPE_MXFP4>(
        "ds4_mmq_mxfp4_moe_pair", W_a, W_b, X, ids, out_a, out_b,
        M, K, n_tokens, n_experts, n_expert_used, stream);
}

// ----------------------------------------------------------------------------
// mmvq-backed entry points (Step 6 of the optimization plan).
//
// mmvq is upstream's matrix-vector matmul family, optimised for the
// n_tokens <= MMVQ_MAX_BATCH_SIZE=8 regime. Unlike mmq it consumes the
// CANONICAL block_q8_1 layout (via quantize_row_q8_1_cuda), not the
// interleaved block_q8_1_mmq that quantize_mmq_q8_1_cuda produces.
//
// The single-W _moe_vec entries cover:
//   - the down matmul at decode (treating [n_tokens=1, n_expert_used=6]
//     as [n_tokens=6, n_expert_used=1])
//   - dense attention projections at decode (n_tokens=1, no MoE)
//   - any small-batch path where mmvq's per-token grid wins over mmq's
//     tile-based approach
//
// The pair-fused _moe_pair_vec entries cover the gate+up matmuls at
// decode using mmvq's built-in fusion. fusion.gate is the up_w pointer
// and fusion.glu_op is GGML_GLU_OP_SWIGLU - the kernel computes
// silu(gate@x) * (up@x) in a single launch. mmvq's fusion is supported
// only at ncols_dst=1, so n_tokens=1 is the only valid case.
// ----------------------------------------------------------------------------

#include "mmvq.cuh"

