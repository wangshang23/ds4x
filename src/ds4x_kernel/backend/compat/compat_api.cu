#include "../internal/backend_internal.cuh"

/* Compatibility implementation for optional graph fallbacks. */

extern "C" int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    const int ok = ds4_gpu_set_model_fd(fd);
    if (ok) g_model_fd_host_base = model_map;
    return ok;
}

extern "C" int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t n_vocab) {
    return ds4_gpu_indexer_topk_tensor(out_idx, logits, n_vocab, 1u, 1u);
}

extern "C" int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor *out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint32_t weight_type,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    switch (weight_type) {
    case 1u:
        return ds4_gpu_matmul_f16_tensor(out, model_map, model_size,
                                         weight_offset, in_dim, out_dim,
                                         x, n_tok);
    case 8u:
        return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                          weight_offset, in_dim, out_dim,
                                          x, n_tok);
    default:
        fprintf(stderr, "ds4: unsupported dense matrix type %u\n",
                weight_type);
        return 0;
    }
}

extern "C" int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor *out_h,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    (void)out_h; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
}

extern "C" int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *q_half,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
        uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, float eps) {
    (void)out; (void)q_half; (void)model_map; (void)model_size;
    (void)weight_offset; (void)in_dim; (void)out_dim; (void)x;
    (void)n_tok; (void)n_head; (void)head_dim; (void)n_rot; (void)pos0;
    (void)n_ctx_orig; (void)inverse; (void)freq_base; (void)freq_scale;
    (void)ext_factor; (void)attn_factor; (void)beta_fast; (void)beta_slow;
    (void)eps;
    return 0;
}

extern "C" int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor *out_h, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups,
        uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    (void)out_h; (void)low; (void)model_map; (void)model_size;
    (void)out_a_offset; (void)out_b_offset; (void)group_dim; (void)rank;
    (void)n_groups; (void)out_dim; (void)heads; (void)n_tokens;
    return 0;
}

extern "C" int ds4_gpu_attention_output_q4_K_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        ds4_gpu_tensor *group_tmp, ds4_gpu_tensor *low_tmp,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset, uint32_t out_b_type,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups,
        uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    (void)out; (void)low; (void)group_tmp; (void)low_tmp;
    (void)model_map; (void)model_size; (void)out_a_offset;
    (void)out_b_offset; (void)out_b_type; (void)group_dim; (void)rank;
    (void)n_groups; (void)out_dim; (void)heads; (void)n_tokens;
    return 0;
}

extern "C" int ds4_gpu_attention_output_low_q4_K_slice_tensor(
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t group0, uint32_t group_cnt,
        const ds4_gpu_tensor *heads) {
    (void)low; (void)model_map; (void)model_size; (void)out_a_offset;
    (void)group_dim; (void)rank; (void)group0; (void)group_cnt;
    (void)heads;
    return 0;
}

extern "C" int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out_h,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)block_out_h; (void)residual_hc; (void)split;
    (void)n_embd; (void)n_hc;
    return 0;
}

extern "C" int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)block_out; (void)block_add_h;
    (void)residual_hc; (void)split; (void)n_embd; (void)n_hc;
    return 0;
}
