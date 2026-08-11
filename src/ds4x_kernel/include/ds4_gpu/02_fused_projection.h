/* =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * The graph uses these primitives for Q/KV projections, HC/output projections,
 * attention output projections, and DS4's tail-only RoPE.
 */

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Optional fused GPU operations.
 *
 * These are acceleration hooks, not required backend primitives.  A backend
 * that does not provide the fused kernel must still define the symbol and
 * return 0.  Callers then use the portable sequence of required primitives.
 * Backends that return nonzero from a fused half-output operation must also
 * implement the matching half-input HC expansion helpers below.
 */
int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Multi-row decode projections that preserve the one-row reduction order. */
int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);
int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_offset,
        uint64_t              weight1_offset,
        uint64_t              in_dim,
        uint64_t              out0_dim,
        uint64_t              out1_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);

int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_router_shared_gate_up_q8_0_tensor(
        ds4_gpu_tensor       *router_logits,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              router_weight_offset,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              in_dim,
        uint64_t              router_out_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        float                 clamp,
        bool                  router_only);
#ifdef __APPLE__
int ds4_gpu_router_project_select_fused_tensor(
        ds4_gpu_tensor       *router_logits,
        ds4_gpu_tensor       *probs,
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              router_weight_offset,
        uint64_t              bias_offset,
        bool                  has_bias,
        const ds4_gpu_tensor *x);
#endif
int ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *prequant,
        uint32_t                expert_split,
        bool                    home_rank);

int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp);

int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* CUDA batch path: fold an input RMS normalization into the FP16 activation
 * conversion used by the following projection. Returns 0 without touching
 * out when the optimized path is unavailable. */
int ds4_gpu_matmul_f16_rms_fold_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   norm_eps);

/* Exact multi-row form of the DeepSeek 4096x256 F16 router projection. */
int ds4_gpu_matmul_f16_router_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        const ds4_gpu_tensor *x,
        uint32_t                n_rows);

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Optional Metal decode fusion. Returns 1 when the paired projection and
 * recurrent compressor-state store were encoded, 0 when the optimized path
 * is unavailable, and -1 on an attempted-path error. */
int ds4_gpu_matmul_f16_pair_compressor_store_tensor(
        ds4_gpu_tensor       *out_kv,
        ds4_gpu_tensor       *out_score,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_kv_offset,
        uint64_t                weight_score_offset,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                in_dim,
        uint32_t                width,
        const ds4_gpu_tensor *x,
        uint32_t                ratio,
        uint32_t                pos);

int ds4_gpu_matmul_f16_quad_compressor_store_tensor(
        ds4_gpu_tensor       *out0_kv,
        ds4_gpu_tensor       *out0_score,
        ds4_gpu_tensor       *out1_kv,
        ds4_gpu_tensor       *out1_score,
        ds4_gpu_tensor       *state0_kv,
        ds4_gpu_tensor       *state0_score,
        ds4_gpu_tensor       *state1_kv,
        ds4_gpu_tensor       *state1_score,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_kv_offset,
        uint64_t              weight0_score_offset,
        uint64_t              weight1_kv_offset,
        uint64_t              weight1_score_offset,
        uint64_t              ape0_offset,
        uint32_t              ape0_type,
        uint64_t              ape1_offset,
        uint32_t              ape1_type,
        uint64_t              in_dim,
        uint32_t              width0,
        uint32_t              width1,
        const ds4_gpu_tensor *x,
        uint32_t              ratio,
        uint32_t              pos);

/* Decode-only M5 fusion: emit-path compressor row finalize (norm + rope +
 * fp8/commit + indexer qat) in one dispatch.  Bit-exact vs the separate
 * dispatches.  Returns 1 when fused, 0 to fall back. */
int ds4_gpu_dsv4_comp_row_finalize_tensor(
        ds4_gpu_tensor       *attn_stage,
        ds4_gpu_tensor       *attn_cache,
        uint32_t              attn_comp_row,
        uint64_t              attn_norm_offset,
        ds4_gpu_tensor       *index_cache,
        uint32_t              index_comp_row,
        uint64_t              index_norm_offset,
        ds4_gpu_tensor       *attn_state_kv,
        ds4_gpu_tensor       *attn_state_score,
        ds4_gpu_tensor       *index_state_kv,
        ds4_gpu_tensor       *index_state_score,
        const void           *model_map,
        uint64_t              model_size,
        uint32_t              pos,
        uint32_t              n_rot,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 rms_eps);

/* Decode-only M5 fusion: q_a/kv Q8 pair projection + F16 quad compressor
 * projection/store in one dispatch.  Bit-exact vs the separate dispatches.
 * Returns 1 when fused, 0 to fall back, -1 on error. */
int ds4_gpu_qkv_pair_quad_compressor_store_tensor(
        ds4_gpu_tensor       *qr,
        ds4_gpu_tensor       *kv_raw,
        ds4_gpu_tensor       *out0_kv,
        ds4_gpu_tensor       *out0_score,
        ds4_gpu_tensor       *out1_kv,
        ds4_gpu_tensor       *out1_score,
        ds4_gpu_tensor       *state0_kv,
        ds4_gpu_tensor       *state0_score,
        ds4_gpu_tensor       *state1_kv,
        ds4_gpu_tensor       *state1_score,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_a_offset,
        uint64_t              kv_offset,
        uint64_t              weight0_kv_offset,
        uint64_t              weight0_score_offset,
        uint64_t              weight1_kv_offset,
        uint64_t              weight1_score_offset,
        uint64_t              ape0_offset,
        uint32_t              ape0_type,
        uint64_t              ape1_offset,
        uint32_t              ape1_type,
        uint32_t              in_dim,
        uint32_t              q_rank,
        uint32_t              kv_dim,
        uint32_t              width0,
        uint32_t              width1,
        const ds4_gpu_tensor *x,
        uint32_t              ratio,
        uint32_t              pos);

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_repeat_hc_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *rows,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t              kv_weight_offset,
        uint32_t              kv_n,
        ds4_gpu_tensor       *raw_cache,
        uint64_t              raw_cap,
        uint32_t              raw_row,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps);

int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        uint32_t                kv_n_head,
        uint32_t                kv_head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        bool                    inverse,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   eps);

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps);

int ds4_gpu_head_rms_norm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow,
        float             eps);

int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *q_half,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tok,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        bool                  inverse,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps);

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot);

int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim);



int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow);

int ds4_gpu_glm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow);

int ds4_gpu_glm_kv_lora_rms_norm_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_raw,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        float                 eps);

int ds4_gpu_glm_store_compact_kv_tensor(
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_norm,
        const ds4_gpu_tensor *kv_raw,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16);

int ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_raw,
        uint64_t              kv_weight_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16,
        float                 eps);

int ds4_gpu_glm_store_indexer_k_tensor(
        ds4_gpu_tensor       *indexer_key_cache,
        const ds4_gpu_tensor *raw_k,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              bias_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              n_ctx_orig,
        float                 eps,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16);

int ds4_gpu_glm_build_kv_cache_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16);

int ds4_gpu_glm_build_kv_cache_flash_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16);

int ds4_gpu_glm_attention_full_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16);

int ds4_gpu_glm_fill_selected_range_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_selected);

int ds4_gpu_glm_fill_selected_range_batch_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_tokens,
        uint32_t        pos0,
        uint32_t        n_selected,
        uint32_t        pad_row);

int ds4_gpu_glm_indexer_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow);

int ds4_gpu_glm_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16);

int ds4_gpu_glm_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16);

int ds4_gpu_glm_qk_lowrank_q8_0_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_qk_lowrank_typed_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim);

int ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim);

int ds4_gpu_glm_attention_indexed_decode_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_rope_tail_decode_rows_tensor(
        ds4_gpu_tensor                     *x,
        const ds4_gpu_attention_decode_row *rows,
        uint32_t                            n_rows,
        uint32_t                            n_head,
        uint32_t                            head_dim,
        uint32_t                            n_rot,
        uint32_t                            n_ctx_orig,
        bool                                inverse,
        float                               freq_base,
        float                               freq_scale,
        float                               ext_factor,
        float                               attn_factor,
        float                               beta_fast,
        float                               beta_slow);

int ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_sort_i32_rows_asc_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        uint32_t              row_width,
        uint32_t              n_rows);

int ds4_gpu_glm_attention_indexed_batch_lora_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_flash_staged_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16);

int ds4_gpu_glm_attention_flash_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16);

/* Release decode fused KV finalizer: after the standalone RoPE kernel, this
 * performs DS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
 * attention cache row in one dispatch. */
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot);

/* Exact multi-session form of the decode KV finalizer. KV rows are
 * contiguous, while each output row is written to its session-private cache. */
int ds4_gpu_kv_fp8_store_raw_decode_rows_tensor(
        ds4_gpu_tensor        *kv,
        ds4_gpu_tensor *const *raw_caches,
        const uint32_t        *raw_caps,
        const uint32_t        *raw_rows,
        uint32_t               n_rows,
        uint32_t               head_dim,
        uint32_t               n_rot);

/* Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * ds4_gpu_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim);

int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim);
