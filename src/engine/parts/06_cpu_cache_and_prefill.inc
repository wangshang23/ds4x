/* =========================================================================
 * KV Cache, Compressors, and CPU Layer Execution.
 * =========================================================================
 *
 * The CPU path is the correctness reference.  It maintains raw SWA KV rows,
 * optional compressed KV rows, the indexer mask for ratio-4 layers, and a
 * reusable decode scratch arena so token generation does not allocate in the
 * hot loop.
 */

typedef struct {
    float *raw_kv;
    uint32_t n_raw;
    uint32_t cap_raw;

    uint32_t compress_ratio;
    uint32_t comp_cap;
    uint32_t n_comp;
    float *attn_comp_kv;
    float *attn_state_kv;
    float *attn_state_score;

    uint32_t n_index_comp;
    float *index_comp_kv;
    float *index_state_kv;
    float *index_state_score;
} ds4_layer_cache;

typedef struct {
    ds4_layer_cache layer[DS4_MAX_LAYER];
    uint32_t head_dim;
} ds4_kv_cache;

static uint32_t ds4_default_raw_cap(uint32_t ctx_size) {
    uint32_t raw_cap = DS4_N_SWA;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    return raw_cap;
}

#define DS4_CUDA_TP_DEFAULT_PREFILL_CHUNK 2048u

static uint32_t ds4_effective_prefill_chunk(bool cuda_tensor_parallel,
                                            uint32_t requested_chunk) {
    if (requested_chunk != 0) return requested_chunk;
    return cuda_tensor_parallel ? DS4_CUDA_TP_DEFAULT_PREFILL_CHUNK : 0;
}

static uint32_t ds4_prefill_cap_for_prompt(int prompt_len,
                                           uint32_t requested_chunk) {
    if (prompt_len <= 0) return 1;
    uint32_t cap = (uint32_t)prompt_len;

    if (requested_chunk != 0) {
        cap = requested_chunk;
    } else {
        const char *env = getenv("DS4_METAL_PREFILL_CHUNK");
        if (env && env[0]) {
            char *endp = NULL;
            const long v = strtol(env, &endp, 10);
            if (endp != env) {
                if (v <= 0) return cap;
                cap = (uint32_t)v;
            }
        } else if (prompt_len > 4096) {
            cap = DS4_MODEL_VARIANT == DS4_VARIANT_PRO ? 8192u : 4096u;
        }
    }

    if (cap == 0) cap = 1;
    if (cap > (uint32_t)prompt_len) cap = (uint32_t)prompt_len;
    return cap;
}

/* Allocate all CPU decode temporaries once.  This keeps generation deterministic
 * from the VM's point of view and makes accidental hot-loop malloc visible. */
static void cpu_decode_scratch_init(ds4_cpu_decode_scratch *scratch, uint32_t ctx_size) {
    memset(scratch, 0, sizeof(*scratch));
    if (ctx_size == 0) ctx_size = 1;
    const uint32_t raw_cap = ds4_default_raw_cap(ctx_size);
    const uint32_t comp_cap = ctx_size / 4 + 2;
    const uint32_t attn_score_cap = raw_cap + comp_cap;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t q8_cap = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t q8_blocks = (q8_cap + 31u) / 32u;
    if ((DS4_N_EMBD % 32u) != 0 || (DS4_N_FF_EXP % 32u) != 0) {
        ds4_die("Q8_0 routed decode scratch dimensions are not QK8_0 aligned");
    }
    const uint64_t routed_q8_x_blocks = DS4_N_EMBD / 32u;
    const uint64_t routed_q8_mid_blocks = DS4_N_FF_EXP / 32u;

    /*
     * The CPU decode path used to malloc/free dozens of medium-sized buffers
     * for every layer of every generated token. On macOS this can drive the VM
     * system through repeated map/unmap bookkeeping while the huge model mmap is
     * also being streamed, and we have observed kernel panics in VM accounting.
     * Keep decode scratch resident for the whole generation instead.
     */
    scratch->ctx_size = ctx_size;
    scratch->comp_cap = comp_cap;
    scratch->attn_score_cap = attn_score_cap;
    scratch->q8_cap = (uint32_t)q8_cap;

    scratch->plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->cur = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->next = xmalloc((size_t)hc_dim * sizeof(float));

    scratch->attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->attn_residual = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->q = xmalloc((size_t)q_dim * sizeof(float));
    scratch->qr = xmalloc((size_t)DS4_N_LORA_Q * sizeof(float));
    scratch->qr_norm = xmalloc((size_t)DS4_N_LORA_Q * sizeof(float));
    scratch->kv_raw = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    scratch->kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    scratch->heads = xmalloc((size_t)q_dim * sizeof(float));
    scratch->attn_low = xmalloc((size_t)DS4_N_OUT_GROUP * DS4_N_LORA_O * sizeof(float));
    scratch->attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->attn_score = xmalloc((size_t)attn_score_cap * sizeof(float));

    scratch->comp = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    scratch->index_comp = xmalloc((size_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
    scratch->comp_kv_cur = xmalloc((size_t)2u * DS4_N_HEAD_DIM * sizeof(float));
    scratch->comp_sc_cur = xmalloc((size_t)2u * DS4_N_HEAD_DIM * sizeof(float));
    scratch->comp_pooled = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));

    scratch->index_allowed = xmalloc((size_t)comp_cap * sizeof(bool));
    scratch->index_q = xmalloc((size_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
    scratch->index_weights = xmalloc((size_t)DS4_N_INDEXER_HEAD * sizeof(float));
    scratch->index_scores = xmalloc((size_t)comp_cap * sizeof(float));

    scratch->ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->ffn_moe = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->ffn_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->shared_gate = xmalloc((size_t)DS4_N_FF_EXP * sizeof(float));
    scratch->shared_up = xmalloc((size_t)DS4_N_FF_EXP * sizeof(float));
    scratch->shared_mid = xmalloc((size_t)DS4_N_FF_EXP * sizeof(float));
    scratch->routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(float));
    scratch->routed_xq = xmalloc((size_t)(DS4_N_EMBD / QK_K) * sizeof(block_q8_K));
    scratch->routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (DS4_N_FF_EXP / QK_K) * sizeof(block_q8_K));
    scratch->routed_q8_xq = xmalloc((size_t)routed_q8_x_blocks * 32u);
    scratch->routed_q8_xscale = xmalloc((size_t)routed_q8_x_blocks * sizeof(float));
    scratch->routed_q8_midq = xmalloc((size_t)DS4_N_EXPERT_USED * routed_q8_mid_blocks * 32u);
    scratch->routed_q8_midscale = xmalloc((size_t)DS4_N_EXPERT_USED * routed_q8_mid_blocks * sizeof(float));

    scratch->q8_xq = xmalloc((size_t)q8_blocks * 32u);
    scratch->q8_xscale = xmalloc((size_t)q8_blocks * sizeof(float));

    scratch->hc_flat = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->output_flat = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->output_pre = xmalloc((size_t)DS4_N_HC * sizeof(float));
    scratch->output_weights = xmalloc((size_t)DS4_N_HC * sizeof(float));
    scratch->output_embd = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->output_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
}

static void cpu_decode_scratch_free(ds4_cpu_decode_scratch *scratch) {
    if (!scratch) return;
    free(scratch->output_norm);
    free(scratch->output_embd);
    free(scratch->output_weights);
    free(scratch->output_pre);
    free(scratch->output_flat);
    free(scratch->hc_flat);
    free(scratch->q8_xscale);
    free(scratch->q8_xq);
    free(scratch->routed_q8_midscale);
    free(scratch->routed_q8_midq);
    free(scratch->routed_q8_xscale);
    free(scratch->routed_q8_xq);
    free(scratch->routed_midq);
    free(scratch->routed_xq);
    free(scratch->routed_mid_all);
    free(scratch->shared_mid);
    free(scratch->shared_up);
    free(scratch->shared_gate);
    free(scratch->ffn_out);
    free(scratch->ffn_shared);
    free(scratch->ffn_moe);
    free(scratch->ffn_norm);
    free(scratch->ffn_cur);
    free(scratch->index_scores);
    free(scratch->index_weights);
    free(scratch->index_q);
    free(scratch->index_allowed);
    free(scratch->comp_pooled);
    free(scratch->comp_sc_cur);
    free(scratch->comp_kv_cur);
    free(scratch->index_comp);
    free(scratch->comp);
    free(scratch->attn_score);
    free(scratch->after_attn_hc);
    free(scratch->attn_out);
    free(scratch->attn_low);
    free(scratch->heads);
    free(scratch->kv);
    free(scratch->kv_raw);
    free(scratch->qr_norm);
    free(scratch->qr);
    free(scratch->q);
    free(scratch->attn_residual);
    free(scratch->attn_norm);
    free(scratch->attn_cur);
    free(scratch->next);
    free(scratch->cur);
    free(scratch->plain);
    memset(scratch, 0, sizeof(*scratch));
}

/* Allocate per-layer KV state: a raw sliding window for all layers, plus
 * compressed attention/indexer caches for layers whose ratio is nonzero. */
static void kv_cache_init(ds4_kv_cache *cache, uint32_t ctx_size, uint32_t raw_cap) {
    memset(cache, 0, sizeof(*cache));
    if (raw_cap == 0) raw_cap = ds4_default_raw_cap(ctx_size);
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;

    cache->head_dim = DS4_N_HEAD_DIM;

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        cache->layer[il].cap_raw = raw_cap;
        cache->layer[il].raw_kv = xmalloc_zeroed((size_t)raw_cap * DS4_N_HEAD_DIM, sizeof(float));
        cache->layer[il].compress_ratio = ratio;

        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint32_t comp_cap = ctx_size / ratio + 2;
            const uint32_t attn_width = coff * DS4_N_HEAD_DIM;
            const uint32_t attn_rows = coff * ratio;

            cache->layer[il].comp_cap = comp_cap;
            cache->layer[il].attn_comp_kv = xmalloc_zeroed((size_t)comp_cap * DS4_N_HEAD_DIM, sizeof(float));
            cache->layer[il].attn_state_kv = xmalloc_zeroed((size_t)attn_width * attn_rows, sizeof(float));
            cache->layer[il].attn_state_score = xmalloc((size_t)attn_width * attn_rows * sizeof(float));
            for (uint64_t i = 0; i < (uint64_t)attn_width * attn_rows; i++) {
                cache->layer[il].attn_state_score[i] = DS4_NEG_INF;
            }

            if (ratio == 4) {
                const uint32_t index_width = coff * DS4_N_INDEXER_HEAD_DIM;
                const uint32_t index_rows = coff * ratio;
                cache->layer[il].index_comp_kv = xmalloc_zeroed((size_t)comp_cap * DS4_N_INDEXER_HEAD_DIM, sizeof(float));
                cache->layer[il].index_state_kv = xmalloc_zeroed((size_t)index_width * index_rows, sizeof(float));
                cache->layer[il].index_state_score = xmalloc((size_t)index_width * index_rows * sizeof(float));
                for (uint64_t i = 0; i < (uint64_t)index_width * index_rows; i++) {
                    cache->layer[il].index_state_score[i] = DS4_NEG_INF;
                }
            }
        }
    }
}

static void kv_cache_free(ds4_kv_cache *cache) {
    if (!cache) return;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        free(cache->layer[il].raw_kv);
        free(cache->layer[il].attn_comp_kv);
        free(cache->layer[il].attn_state_kv);
        free(cache->layer[il].attn_state_score);
        free(cache->layer[il].index_comp_kv);
        free(cache->layer[il].index_state_kv);
        free(cache->layer[il].index_state_score);
    }
    memset(cache, 0, sizeof(*cache));
}

/* Append to the raw SWA cache.  Once full, it slides by one row. */
static void kv_cache_push_raw(ds4_layer_cache *cache, const float *kv) {
    if (cache->n_raw < cache->cap_raw) {
        float *dst = cache->raw_kv + (uint64_t)cache->n_raw * DS4_N_HEAD_DIM;
        for (uint32_t i = 0; i < DS4_N_HEAD_DIM; i++) dst[i] = f16_to_f32(f32_to_f16(kv[i]));
        cache->n_raw++;
        return;
    }

    memmove(cache->raw_kv,
            cache->raw_kv + DS4_N_HEAD_DIM,
            (size_t)(cache->cap_raw - 1) * DS4_N_HEAD_DIM * sizeof(cache->raw_kv[0]));
    float *dst = cache->raw_kv + (uint64_t)(cache->cap_raw - 1) * DS4_N_HEAD_DIM;
    for (uint32_t i = 0; i < DS4_N_HEAD_DIM; i++) dst[i] = f16_to_f32(f32_to_f16(kv[i]));
}

static void kv_cache_push_comp(float *rows, uint32_t *n_rows, uint32_t cap_rows, uint32_t row_dim, const float *kv) {
    if (*n_rows >= cap_rows) ds4_die("compressed KV cache capacity exceeded");
    float *dst = rows + (uint64_t)(*n_rows) * row_dim;
    for (uint32_t i = 0; i < row_dim; i++) dst[i] = f16_to_f32(f32_to_f16(kv[i]));
    (*n_rows)++;
}

/* After prefill, clear unused compressor state rows so decode starts from the
 * same partial-window state the streaming path would have produced. */
static void compressor_finish_prefill_state_cpu(
        float    * state_kv,
        float    * state_score,
        uint32_t   head_dim,
        uint32_t   compress_ratio,
        uint32_t   n_tokens) {
    if (!state_kv || !state_score || head_dim == 0 || compress_ratio == 0) return;

    const uint32_t coff = compress_ratio == 4 ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t rem = n_tokens % compress_ratio;
    const uint32_t clear_start = compress_ratio == 4 ? compress_ratio + rem : rem;
    const uint32_t clear_end = compress_ratio == 4 ? 2u * compress_ratio : compress_ratio;

    for (uint32_t row = clear_start; row < clear_end; row++) {
        float *kv = state_kv + (uint64_t)row * width;
        float *score = state_score + (uint64_t)row * width;
        memset(kv, 0, (size_t)width * sizeof(kv[0]));
        for (uint32_t i = 0; i < width; i++) score[i] = DS4_NEG_INF;
    }
}

static void kv_cache_finish_prefill_states(ds4_kv_cache *cache, uint32_t n_tokens) {
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_layer_cache *layer = &cache->layer[il];
        const uint32_t ratio = layer->compress_ratio;
        if (ratio == 0) continue;

        compressor_finish_prefill_state_cpu(layer->attn_state_kv,
                                            layer->attn_state_score,
                                            DS4_N_HEAD_DIM,
                                            ratio,
                                            n_tokens);
        if (ratio == 4) {
            compressor_finish_prefill_state_cpu(layer->index_state_kv,
                                                layer->index_state_score,
                                                DS4_N_INDEXER_HEAD_DIM,
                                                ratio,
                                                n_tokens);
        }
    }
}

/* Pool the current compression window with a softmax over per-dimension scores.
 * Ratio-4 layers keep two lanes: attention compression and indexer compression. */
static void compressor_pool_decode_state(
        float    * out,
        float    * state_kv,
        float    * state_score,
        uint32_t   head_dim,
        uint32_t   compress_ratio) {
    const uint32_t coff = compress_ratio == 4 ? 2u : 1u;
    const uint32_t width = coff * head_dim;

    for (uint32_t j = 0; j < head_dim; j++) {
        float max_score = DS4_NEG_INF;

        if (compress_ratio == 4) {
            for (uint32_t r = 0; r < compress_ratio; r++) {
                const float sp = state_score[(uint64_t)r * width + j];
                const float sc = state_score[(uint64_t)(compress_ratio + r) * width + head_dim + j];
                if (sp > max_score) max_score = sp;
                if (sc > max_score) max_score = sc;
            }
        } else {
            for (uint32_t r = 0; r < compress_ratio; r++) {
                const float s = state_score[(uint64_t)r * width + j];
                if (s > max_score) max_score = s;
            }
        }

        if (max_score <= DS4_NEG_INF * 0.5f) {
            out[j] = 0.0f;
            continue;
        }

        float denom = 0.0f;
        float sum = 0.0f;
        if (compress_ratio == 4) {
            for (uint32_t r = 0; r < compress_ratio; r++) {
                const float wp = expf(state_score[(uint64_t)r * width + j] - max_score);
                const float wc = expf(state_score[(uint64_t)(compress_ratio + r) * width + head_dim + j] - max_score);
                denom += wp + wc;
                sum += wp * state_kv[(uint64_t)r * width + j];
                sum += wc * state_kv[(uint64_t)(compress_ratio + r) * width + head_dim + j];
            }
        } else {
            for (uint32_t r = 0; r < compress_ratio; r++) {
                const float w = expf(state_score[(uint64_t)r * width + j] - max_score);
                denom += w;
                sum += w * state_kv[(uint64_t)r * width + j];
            }
        }

        out[j] = denom > 0.0f ? sum / denom : 0.0f;
    }
}

/* Streaming compressor update for one token.  It projects kv/score rows,
 * updates the rolling state, and emits a compressed KV row on ratio boundaries. */
static bool compressor_decode_one(
        float                   * out_comp,
        const ds4_model         * model,
        const ds4_tensor        * wkv,
        const ds4_tensor        * wgate,
        const ds4_tensor        * ape,
        const ds4_tensor        * norm,
        const float             * x,
        float                   * state_kv,
        float                   * state_score,
        uint32_t                  head_dim,
        uint32_t                  compress_ratio,
        uint32_t                  il,
        uint32_t                  pos) {
    const uint32_t coff = compress_ratio == 4 ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t pos_mod = pos % compress_ratio;
    const uint32_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
    const bool should_compress = ((pos + 1) % compress_ratio) == 0;

    float *kv_cur = xmalloc((size_t)width * sizeof(kv_cur[0]));
    float *sc_cur = xmalloc((size_t)width * sizeof(sc_cur[0]));
    if (wkv->type == 8 &&
        wgate->type == 8 &&
        wkv->ndim == 2 &&
        wgate->ndim == 2 &&
        wkv->dim[0] == wgate->dim[0]) {
        const uint64_t in_dim = wkv->dim[0];
        const uint64_t blocks = (in_dim + 31) / 32;
        int8_t *xq = xmalloc((size_t)blocks * 32);
        float *xscale = xmalloc((size_t)blocks * sizeof(xscale[0]));

        quantize_q8_0_activation(x, xq, xscale, in_dim);
        matvec_q8_0_pair_prequant(kv_cur, sc_cur, model, wkv, wgate, xq, xscale);

        free(xscale);
        free(xq);
    } else {
        matvec_any(kv_cur, model, wkv, x);
        matvec_any(sc_cur, model, wgate, x);
    }

    for (uint32_t j = 0; j < width; j++) {
        sc_cur[j] += tensor_2d_value(model, ape, j, pos_mod);
    }

    memcpy(state_kv + (uint64_t)row * width, kv_cur, (size_t)width * sizeof(kv_cur[0]));
    memcpy(state_score + (uint64_t)row * width, sc_cur, (size_t)width * sizeof(sc_cur[0]));

    free(sc_cur);
    free(kv_cur);

    if (!should_compress) {
        return false;
    }

    float *pooled = xmalloc((size_t)head_dim * sizeof(pooled[0]));
    compressor_pool_decode_state(pooled, state_kv, state_score, head_dim, compress_ratio);

    double ss = 0.0;
    for (uint32_t i = 0; i < head_dim; i++) ss += (double)pooled[i] * pooled[i];
    const float rms = 1.0f / sqrtf((float)(ss / (double)head_dim) + DS4_RMS_EPS);
    for (uint32_t i = 0; i < head_dim; i++) {
        out_comp[i] = pooled[i] * rms * tensor_1d_value(model, norm, i);
    }

    const uint32_t comp_pos = pos + 1 - compress_ratio;
    rope_tail_layer_inplace(out_comp, 1, head_dim, DS4_N_ROT, comp_pos, il, false);
    if (head_dim == DS4_N_HEAD_DIM) {
        dsv4_fp8_kv_quantize_row_inplace_cpu(out_comp, head_dim, DS4_N_ROT);
    } else if (head_dim == DS4_N_INDEXER_HEAD_DIM) {
        dsv4_indexer_qat_row_inplace_cpu(out_comp, head_dim);
    }

    if (compress_ratio == 4) {
        for (uint32_t r = 0; r < compress_ratio; r++) {
            memcpy(state_kv + (uint64_t)r * width,
                   state_kv + (uint64_t)(compress_ratio + r) * width,
                   (size_t)width * sizeof(state_kv[0]));
            memcpy(state_score + (uint64_t)r * width,
                   state_score + (uint64_t)(compress_ratio + r) * width,
                   (size_t)width * sizeof(state_score[0]));
        }
        for (uint32_t r = 0; r < compress_ratio; r++) {
            memcpy(state_kv + (uint64_t)(compress_ratio + r) * width,
                   state_kv + (uint64_t)r * width,
                   (size_t)width * sizeof(state_kv[0]));
            memcpy(state_score + (uint64_t)(compress_ratio + r) * width,
                   state_score + (uint64_t)r * width,
                   (size_t)width * sizeof(state_score[0]));
        }
    }

    free(pooled);
    return true;
}

static bool compressor_decode_one_decode_scratch(
        float                  * out_comp,
        const ds4_model        * model,
        const ds4_tensor       * wkv,
        const ds4_tensor       * wgate,
        const ds4_tensor       * ape,
        const ds4_tensor       * norm,
        const float            * x,
        float                  * state_kv,
        float                  * state_score,
        uint32_t                 head_dim,
        uint32_t                 compress_ratio,
        uint32_t                 il,
        uint32_t                 pos,
        ds4_cpu_decode_scratch * scratch) {
    const uint32_t coff = compress_ratio == 4 ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t pos_mod = pos % compress_ratio;
    const uint32_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
    const bool should_compress = ((pos + 1) % compress_ratio) == 0;

    if (width > 2u * DS4_N_HEAD_DIM) ds4_die("compressor scratch width is outside the fixed model layout");
    float *kv_cur = scratch->comp_kv_cur;
    float *sc_cur = scratch->comp_sc_cur;

    if (wkv->type == 8 &&
        wgate->type == 8 &&
        wkv->ndim == 2 &&
        wgate->ndim == 2 &&
        wkv->dim[0] == wgate->dim[0]) {
        matvec_q8_0_pair_decode_scratch(kv_cur, sc_cur, model, wkv, wgate, x, scratch);
    } else {
        matvec_any_decode_scratch(kv_cur, model, wkv, x, scratch);
        matvec_any_decode_scratch(sc_cur, model, wgate, x, scratch);
    }

    for (uint32_t j = 0; j < width; j++) {
        sc_cur[j] += tensor_2d_value(model, ape, j, pos_mod);
    }

    memcpy(state_kv + (uint64_t)row * width, kv_cur, (size_t)width * sizeof(kv_cur[0]));
    memcpy(state_score + (uint64_t)row * width, sc_cur, (size_t)width * sizeof(sc_cur[0]));

    if (!should_compress) {
        return false;
    }

    float *pooled = scratch->comp_pooled;
    compressor_pool_decode_state(pooled, state_kv, state_score, head_dim, compress_ratio);

    double ss = 0.0;
    for (uint32_t i = 0; i < head_dim; i++) ss += (double)pooled[i] * pooled[i];
    const float rms = 1.0f / sqrtf((float)(ss / (double)head_dim) + DS4_RMS_EPS);
    for (uint32_t i = 0; i < head_dim; i++) {
        out_comp[i] = pooled[i] * rms * tensor_1d_value(model, norm, i);
    }

    const uint32_t comp_pos = pos + 1 - compress_ratio;
    rope_tail_layer_inplace(out_comp, 1, head_dim, DS4_N_ROT, comp_pos, il, false);
    if (head_dim == DS4_N_HEAD_DIM) {
        dsv4_fp8_kv_quantize_row_inplace_cpu(out_comp, head_dim, DS4_N_ROT);
    } else if (head_dim == DS4_N_INDEXER_HEAD_DIM) {
        dsv4_indexer_qat_row_inplace_cpu(out_comp, head_dim);
    }

    if (compress_ratio == 4) {
        for (uint32_t r = 0; r < compress_ratio; r++) {
            memcpy(state_kv + (uint64_t)r * width,
                   state_kv + (uint64_t)(compress_ratio + r) * width,
                   (size_t)width * sizeof(state_kv[0]));
            memcpy(state_score + (uint64_t)r * width,
                   state_score + (uint64_t)(compress_ratio + r) * width,
                   (size_t)width * sizeof(state_score[0]));
        }
        for (uint32_t r = 0; r < compress_ratio; r++) {
            memcpy(state_kv + (uint64_t)(compress_ratio + r) * width,
                   state_kv + (uint64_t)r * width,
                   (size_t)width * sizeof(state_kv[0]));
            memcpy(state_score + (uint64_t)(compress_ratio + r) * width,
                   state_score + (uint64_t)r * width,
                   (size_t)width * sizeof(state_score[0]));
        }
    }

    return true;
}

/* Attention over raw SWA rows plus optional compressed rows.  Ratio-4 layers
 * pass an indexer mask to hide compressed rows not selected for this token. */
static void layer_attention_mixed_one(
        float             * out_heads,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * q,
        const float       * raw_kv,
        uint32_t            n_raw,
        const float       * comp_kv,
        uint32_t            n_comp,
        const bool        * comp_allowed) {
    const float *sinks = tensor_data(model, layer->attn_sinks);
    const float kq_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    const uint32_t n_total = n_raw + n_comp;
    float score_stack[512];
    float *score = n_total <= 512 ? score_stack : xmalloc((size_t)n_total * sizeof(score[0]));

    for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
        const float *qh = q + (uint64_t)h * DS4_N_HEAD_DIM;
        float max_score = sinks[h];
        uint32_t idx = 0;

        for (uint32_t r = 0; r < n_raw; r++, idx++) {
            const float *kv = raw_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            score[idx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[idx] > max_score) max_score = score[idx];
        }
        for (uint32_t r = 0; r < n_comp; r++, idx++) {
            if (comp_allowed && !comp_allowed[r]) {
                score[idx] = DS4_NEG_INF;
                continue;
            }
            const float *kv = comp_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            score[idx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[idx] > max_score) max_score = score[idx];
        }

        float *oh = out_heads + (uint64_t)h * DS4_N_HEAD_DIM;
        memset(oh, 0, (size_t)DS4_N_HEAD_DIM * sizeof(oh[0]));

        float denom = expf(sinks[h] - max_score);
        idx = 0;
        for (uint32_t r = 0; r < n_raw; r++, idx++) {
            const float weight = expf(score[idx] - max_score);
            const float *kv = raw_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }
        for (uint32_t r = 0; r < n_comp; r++, idx++) {
            if (score[idx] <= DS4_NEG_INF * 0.5f) continue;
            const float weight = expf(score[idx] - max_score);
            const float *kv = comp_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }

        const float inv = 1.0f / denom;
        scale_f32(oh, inv, DS4_N_HEAD_DIM);
    }

    if (score != score_stack) free(score);
}

static void layer_attention_mixed_one_decode_scratch(
        float                  * out_heads,
        const ds4_model        * model,
        const ds4_layer_weights * layer,
        const float            * q,
        const float            * raw_kv,
        uint32_t                 n_raw,
        const float            * comp_kv,
        uint32_t                 n_comp,
        const bool             * comp_allowed,
        ds4_cpu_decode_scratch * scratch) {
    const float *sinks = tensor_data(model, layer->attn_sinks);
    const float kq_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    const uint32_t n_total = n_raw + n_comp;
    if (n_total > scratch->attn_score_cap) ds4_die("CPU decode attention score scratch buffer is too small");
    float *score = scratch->attn_score;

    for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
        const float *qh = q + (uint64_t)h * DS4_N_HEAD_DIM;
        float max_score = sinks[h];
        uint32_t idx = 0;

        for (uint32_t r = 0; r < n_raw; r++, idx++) {
            const float *kv = raw_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            score[idx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[idx] > max_score) max_score = score[idx];
        }
        for (uint32_t r = 0; r < n_comp; r++, idx++) {
            if (comp_allowed && !comp_allowed[r]) {
                score[idx] = DS4_NEG_INF;
                continue;
            }
            const float *kv = comp_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            score[idx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[idx] > max_score) max_score = score[idx];
        }

        float *oh = out_heads + (uint64_t)h * DS4_N_HEAD_DIM;
        memset(oh, 0, (size_t)DS4_N_HEAD_DIM * sizeof(oh[0]));

        float denom = expf(sinks[h] - max_score);
        idx = 0;
        for (uint32_t r = 0; r < n_raw; r++, idx++) {
            const float weight = expf(score[idx] - max_score);
            const float *kv = raw_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }
        for (uint32_t r = 0; r < n_comp; r++, idx++) {
            if (score[idx] <= DS4_NEG_INF * 0.5f) continue;
            const float weight = expf(score[idx] - max_score);
            const float *kv = comp_kv + (uint64_t)r * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }

        const float inv = 1.0f / denom;
        scale_f32(oh, inv, DS4_N_HEAD_DIM);
    }
}

typedef struct {
    float             * out_heads;
    const ds4_model   * model;
    const ds4_layer_weights * layer;
    const float       * q;
    const float       * raw_kv;
    const float       * comp_kv;
    const uint32_t    * comp_counts;
    const uint8_t     * allowed_mask;
    const uint8_t     * allowed_bits;
    uint64_t            allowed_stride;
    uint32_t            n_tok;
    uint32_t            raw_cap;
} layer_attention_prefix_batch_ctx;

static inline bool attention_prefix_comp_allowed(
        const layer_attention_prefix_batch_ctx *ctx,
        uint32_t                                t,
        uint32_t                                c) {
    if (!ctx->allowed_bits || !ctx->allowed_mask || !ctx->allowed_mask[t]) return true;
    const uint8_t *bits = ctx->allowed_bits + (uint64_t)t * ctx->allowed_stride;
    return (bits[c >> 3] & (uint8_t)(1u << (c & 7u))) != 0;
}

static void layer_attention_prefix_batch_worker(void *vctx, uint64_t r0, uint64_t r1) {
    layer_attention_prefix_batch_ctx *ctx = vctx;
    const float *sinks = tensor_data(ctx->model, ctx->layer->attn_sinks);
    const float kq_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    const uint32_t max_comp = ctx->comp_counts ? ctx->comp_counts[ctx->n_tok - 1] : 0;
    const uint32_t max_total = ctx->raw_cap + max_comp;
    float score_stack[2048];
    float *score = max_total <= 2048 ? score_stack : xmalloc((size_t)max_total * sizeof(score[0]));

    for (uint64_t idx = r0; idx < r1; idx++) {
        const uint32_t t = (uint32_t)(idx / DS4_N_HEAD);
        const uint32_t h = (uint32_t)(idx - (uint64_t)t * DS4_N_HEAD);
        const uint32_t raw_count = t + 1 < ctx->raw_cap ? t + 1 : ctx->raw_cap;
        const uint32_t raw_start = t + 1 - raw_count;
        const uint32_t comp_count = ctx->comp_counts ? ctx->comp_counts[t] : 0;
        const float *qh = ctx->q + (uint64_t)t * DS4_N_HEAD * DS4_N_HEAD_DIM + (uint64_t)h * DS4_N_HEAD_DIM;

        float max_score = sinks[h];
        uint32_t sidx = 0;
        for (uint32_t r = 0; r < raw_count; r++, sidx++) {
            const float *kv = ctx->raw_kv + (uint64_t)(raw_start + r) * DS4_N_HEAD_DIM;
            score[sidx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[sidx] > max_score) max_score = score[sidx];
        }
        for (uint32_t c = 0; c < comp_count; c++, sidx++) {
            if (!attention_prefix_comp_allowed(ctx, t, c)) {
                score[sidx] = DS4_NEG_INF;
                continue;
            }
            const float *kv = ctx->comp_kv + (uint64_t)c * DS4_N_HEAD_DIM;
            score[sidx] = dot_f32(qh, kv, DS4_N_HEAD_DIM) * kq_scale;
            if (score[sidx] > max_score) max_score = score[sidx];
        }

        float *oh = ctx->out_heads + (uint64_t)t * DS4_N_HEAD * DS4_N_HEAD_DIM + (uint64_t)h * DS4_N_HEAD_DIM;
        memset(oh, 0, (size_t)DS4_N_HEAD_DIM * sizeof(oh[0]));

        float denom = expf(sinks[h] - max_score);
        sidx = 0;
        for (uint32_t r = 0; r < raw_count; r++, sidx++) {
            const float weight = expf(score[sidx] - max_score);
            const float *kv = ctx->raw_kv + (uint64_t)(raw_start + r) * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }
        for (uint32_t c = 0; c < comp_count; c++, sidx++) {
            if (score[sidx] <= DS4_NEG_INF * 0.5f) continue;
            const float weight = expf(score[sidx] - max_score);
            const float *kv = ctx->comp_kv + (uint64_t)c * DS4_N_HEAD_DIM;
            denom += weight;
            axpy_f32(oh, kv, weight, DS4_N_HEAD_DIM);
        }

        scale_f32(oh, 1.0f / denom, DS4_N_HEAD_DIM);
    }

    if (score != score_stack) free(score);
}

/* Prefix prefill attention for a fresh prompt.  It computes each token's view
 * of the raw window and compressed rows without running the decode loop. */
static void layer_attention_prefix_batch(
        float                   * out_heads,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * q,
        const float             * raw_kv,
        const float             * comp_kv,
        const uint32_t          * comp_counts,
        const uint8_t           * allowed_mask,
        const uint8_t           * allowed_bits,
        uint64_t                  allowed_stride,
        uint32_t                  n_tok,
        uint32_t                  raw_cap) {
    layer_attention_prefix_batch_ctx ctx = {
        .out_heads = out_heads,
        .model = model,
        .layer = layer,
        .q = q,
        .raw_kv = raw_kv,
        .comp_kv = comp_kv,
        .comp_counts = comp_counts,
        .allowed_mask = allowed_mask,
        .allowed_bits = allowed_bits,
        .allowed_stride = allowed_stride,
        .n_tok = n_tok,
        .raw_cap = raw_cap,
    };
    ds4_parallel_for_min_rows((uint64_t)n_tok * DS4_N_HEAD,
                              layer_attention_prefix_batch_worker,
                              &ctx,
                              1);
}

/* Ratio-4 layers use an auxiliary indexer to select which compressed rows are
 * visible to attention.  This is the CPU allocation-owning helper. */
static bool *indexer_allowed_decode_one(
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * cur,
        const float             * qr_norm,
        const float             * index_comp,
        uint32_t                  n_comp,
        uint32_t                  il,
        uint32_t                  pos) {
    if (n_comp == 0) return NULL;

    bool *allowed = xcalloc(n_comp, sizeof(allowed[0]));
    const uint32_t top_k = DS4_N_INDEXER_TOP_K < n_comp ? DS4_N_INDEXER_TOP_K : n_comp;
    if (top_k == n_comp) {
        for (uint32_t i = 0; i < n_comp; i++) allowed[i] = true;
        return allowed;
    }

    const uint32_t head_dim = DS4_N_INDEXER_HEAD_DIM;
    const uint32_t n_head = DS4_N_INDEXER_HEAD;
    float *q = xmalloc((size_t)head_dim * n_head * sizeof(q[0]));
    float *weights = xmalloc((size_t)n_head * sizeof(weights[0]));
    float *scores = xmalloc((size_t)n_comp * sizeof(scores[0]));

    matvec_any(q, model, layer->indexer_attn_q_b, qr_norm);
    rope_tail_layer_inplace(q, n_head, head_dim, DS4_N_ROT, pos, il, false);
    dsv4_indexer_qat_rows_inplace_cpu(q, n_head, head_dim);

    matvec_any(weights, model, layer->indexer_proj, cur);
    const float scale = 1.0f / sqrtf((float)(head_dim * n_head));
    for (uint32_t h = 0; h < n_head; h++) weights[h] *= scale;

    for (uint32_t c = 0; c < n_comp; c++) {
        const float *kv = index_comp + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + (uint64_t)h * head_dim;
            float dot = dot_f32(kv, qh, head_dim);
            if (dot < 0.0f) dot = 0.0f;
            s += dot * weights[h];
        }
        scores[c] = s;
    }

    for (uint32_t k = 0; k < top_k; k++) {
        uint32_t best = 0;
        float best_score = DS4_NEG_INF;
        for (uint32_t c = 0; c < n_comp; c++) {
            if (!allowed[c] && scores[c] > best_score) {
                best = c;
                best_score = scores[c];
            }
        }
        allowed[best] = true;
    }

    free(scores);
    free(weights);
    free(q);
    return allowed;
}

/* Scratch-backed indexer selection for decode. */
static bool *indexer_allowed_decode_one_decode_scratch(
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        const float             * cur,
        const float             * qr_norm,
        const float             * index_comp,
        uint32_t                  n_comp,
        uint32_t                  il,
        uint32_t                  pos,
        ds4_cpu_decode_scratch  * scratch) {
    if (n_comp == 0) return NULL;
    if (n_comp > scratch->comp_cap) ds4_die("CPU decode indexer scratch buffer is too small");

    bool *allowed = scratch->index_allowed;
    memset(allowed, 0, (size_t)n_comp * sizeof(allowed[0]));
    const uint32_t top_k = DS4_N_INDEXER_TOP_K < n_comp ? DS4_N_INDEXER_TOP_K : n_comp;
    if (top_k == n_comp) {
        for (uint32_t i = 0; i < n_comp; i++) allowed[i] = true;
        return allowed;
    }

    const uint32_t head_dim = DS4_N_INDEXER_HEAD_DIM;
    const uint32_t n_head = DS4_N_INDEXER_HEAD;
    float *q = scratch->index_q;
    float *weights = scratch->index_weights;
    float *scores = scratch->index_scores;

    matvec_any_decode_scratch(q, model, layer->indexer_attn_q_b, qr_norm, scratch);
    rope_tail_layer_inplace(q, n_head, head_dim, DS4_N_ROT, pos, il, false);
    dsv4_indexer_qat_rows_inplace_cpu(q, n_head, head_dim);

    matvec_any_decode_scratch(weights, model, layer->indexer_proj, cur, scratch);
    const float scale = 1.0f / sqrtf((float)(head_dim * n_head));
    for (uint32_t h = 0; h < n_head; h++) weights[h] *= scale;

    for (uint32_t c = 0; c < n_comp; c++) {
        const float *kv = index_comp + (uint64_t)c * head_dim;
        float s = 0.0f;
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + (uint64_t)h * head_dim;
            float dot = dot_f32(kv, qh, head_dim);
            if (dot < 0.0f) dot = 0.0f;
            s += dot * weights[h];
        }
        scores[c] = s;
    }

    for (uint32_t k = 0; k < top_k; k++) {
        uint32_t best = 0;
        float best_score = DS4_NEG_INF;
        for (uint32_t c = 0; c < n_comp; c++) {
            if (!allowed[c] && scores[c] > best_score) {
                best = c;
                best_score = scores[c];
            }
        }
        allowed[best] = true;
    }

    return allowed;
}

/* Single-token attention sublayer with raw SWA cache and DS4 compression. */
static void layer_attention_raw_swa_one(
        float                   * after_attn_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        ds4_layer_cache         * cache,
        const float             * inp_hc,
        uint32_t                  il,
        uint32_t                  pos,
        const float             * steering_dirs,
        float                     steering_scale) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;

    float *attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_cur[0]));
    float *attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_norm[0]));
    float *attn_residual = xmalloc((size_t)n_hc * DS4_N_EMBD * sizeof(attn_residual[0]));
    float *q = xmalloc((size_t)q_dim * sizeof(q[0]));
    float *qr_norm = xmalloc((size_t)DS4_N_LORA_Q * sizeof(qr_norm[0]));
    float *kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(kv[0]));
    float *heads = xmalloc((size_t)q_dim * sizeof(heads[0]));
    float *attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(attn_out[0]));
    bool *comp_allowed = NULL;
    float post[4];
    float comb[16];

    memcpy(attn_residual, inp_hc, (size_t)n_hc * DS4_N_EMBD * sizeof(inp_hc[0]));
    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          attn_residual, attn_cur, post, comb);

    layer_attn_norm_one(attn_norm, model, layer, attn_cur);
    layer_q_projection_with_lora_one(model, layer, attn_norm, q, qr_norm);
    layer_kv_projection_normed_one(model, layer, attn_norm, kv);

    rope_tail_layer_inplace(q, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    rope_tail_layer_inplace(kv, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
    dsv4_fp8_kv_quantize_row_inplace_cpu(kv, DS4_N_HEAD_DIM, DS4_N_ROT);

    kv_cache_push_raw(cache, kv);

    const uint32_t ratio = cache->compress_ratio;
    if (ratio != 0) {
        float *comp = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(comp[0]));
        if (compressor_decode_one(comp, model,
                                  layer->attn_compressor_kv,
                                  layer->attn_compressor_gate,
                                  layer->attn_compressor_ape,
                                  layer->attn_compressor_norm,
                                  attn_norm,
                                  cache->attn_state_kv,
                                  cache->attn_state_score,
                                  DS4_N_HEAD_DIM,
                                  ratio,
                                  il,
                                  pos)) {
            kv_cache_push_comp(cache->attn_comp_kv, &cache->n_comp, cache->comp_cap, DS4_N_HEAD_DIM, comp);
        }
        free(comp);

        if (ratio == 4) {
            float *index_comp = xmalloc((size_t)DS4_N_INDEXER_HEAD_DIM * sizeof(index_comp[0]));
            if (compressor_decode_one(index_comp, model,
                                      layer->indexer_compressor_kv,
                                      layer->indexer_compressor_gate,
                                      layer->indexer_compressor_ape,
                                      layer->indexer_compressor_norm,
                                      attn_norm,
                                      cache->index_state_kv,
                                      cache->index_state_score,
                                      DS4_N_INDEXER_HEAD_DIM,
                                      ratio,
                                      il,
                                      pos)) {
                kv_cache_push_comp(cache->index_comp_kv, &cache->n_index_comp, cache->comp_cap, DS4_N_INDEXER_HEAD_DIM, index_comp);
            }
            free(index_comp);

            comp_allowed = indexer_allowed_decode_one(model, layer,
                                                      attn_norm, qr_norm,
                                                      cache->index_comp_kv,
                                                      cache->n_index_comp,
                                                      il, pos);
        }

        layer_attention_mixed_one(heads, model, layer, q,
                                  cache->raw_kv, cache->n_raw,
                                  cache->attn_comp_kv, cache->n_comp,
                                  comp_allowed);
    } else {
        layer_attention_rows_one(heads, model, layer, q, cache->raw_kv, cache->n_raw);
    }

    rope_tail_layer_inplace(heads, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, true);
    layer_grouped_out_one(attn_out, model, layer, heads);
    cpu_directional_steering_project_rows(attn_out, steering_dirs, il, 1, steering_scale);
    hc_post_one(after_attn_hc, attn_out, attn_residual, post, comb, DS4_N_EMBD, n_hc);

    free(comp_allowed);
    free(attn_out);
    free(heads);
    free(kv);
    free(qr_norm);
    free(q);
    free(attn_residual);
    free(attn_norm);
    free(attn_cur);
}

/* Batched prefill attention.  It projects Q/KV for all tokens, streams them
 * through the same raw/compressed cache updates, then runs prefix attention. */
static void layer_attention_raw_swa_batch(
        float                   * after_attn_hc,
        const ds4_model         * model,
        const ds4_layer_weights * layer,
        ds4_layer_cache         * cache,
        const float             * inp_hc,
        uint32_t                  n_tok,
        uint32_t                  il,
        uint32_t                  pos0,
        const float             * steering_dirs,
        float                     steering_scale) {
    const bool profile = getenv("DS4_PREFILL_PROFILE_DETAIL") != NULL;
    const double t_start = profile ? now_sec() : 0.0;
    double t_hc_norm = 0.0;
    double t_q = 0.0;
    double t_kv = 0.0;
    double t_token_loop = 0.0;
    double t_tl_rope_cache = 0.0;
    double t_tl_compress = 0.0;
    double t_tl_indexer = 0.0;
    double t_tl_attn_rows = 0.0;
    double t_tl_inv_rope = 0.0;
    double t_out = 0.0;
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)n_hc * DS4_N_EMBD;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;

    float *attn_cur = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(attn_cur[0]));
    float *attn_norm = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(attn_norm[0]));
    float *attn_residual = xmalloc((size_t)n_tok * hc_dim * sizeof(attn_residual[0]));
    const uint32_t q_rank = DS4_N_LORA_Q;
    float *qr = xmalloc((size_t)n_tok * q_rank * sizeof(qr[0]));
    float *qr_norm = xmalloc((size_t)n_tok * q_rank * sizeof(qr_norm[0]));
    float *q = xmalloc((size_t)n_tok * q_dim * sizeof(q[0]));
    float *kv_raw = xmalloc((size_t)n_tok * DS4_N_HEAD_DIM * sizeof(kv_raw[0]));
    float *kv = xmalloc((size_t)n_tok * DS4_N_HEAD_DIM * sizeof(kv[0]));
    float *heads = NULL;
    float *attn_out = xmalloc((size_t)n_tok * DS4_N_EMBD * sizeof(attn_out[0]));
    float *post = xmalloc((size_t)n_tok * n_hc * sizeof(post[0]));
    float *comb = xmalloc((size_t)n_tok * n_hc * n_hc * sizeof(comb[0]));

    const float *q_a_norm = tensor_data(model, layer->attn_q_a_norm);
    const float *kv_norm = tensor_data(model, layer->attn_kv_a_norm);

    double t0 = profile ? now_sec() : 0.0;
    hc_pre_norm_batch(model,
                      layer->hc_attn_fn,
                      layer->hc_attn_scale,
                      layer->hc_attn_base,
                      layer->attn_norm,
                      inp_hc,
                      attn_residual,
                      attn_cur,
                      attn_norm,
                      post,
                      comb,
                      n_tok);
    if (profile) t_hc_norm = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    matmul_q8_0_batch(qr, model, layer->attn_q_a, attn_norm, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        rms_norm_weight(qr_norm + (uint64_t)t * q_rank,
                        qr + (uint64_t)t * q_rank,
                        q_a_norm,
                        q_rank,
                        DS4_RMS_EPS);
    }
    matmul_q8_0_batch(q, model, layer->attn_q_b, qr_norm, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        head_rms_norm_inplace(q + (uint64_t)t * q_dim,
                              DS4_N_HEAD,
                              DS4_N_HEAD_DIM,
                              DS4_RMS_EPS);
    }
    if (profile) t_q = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    matmul_q8_0_batch(kv_raw, model, layer->attn_kv, attn_norm, n_tok);
    for (uint32_t t = 0; t < n_tok; t++) {
        rms_norm_weight(kv + (uint64_t)t * DS4_N_HEAD_DIM,
                        kv_raw + (uint64_t)t * DS4_N_HEAD_DIM,
                        kv_norm,
                        DS4_N_HEAD_DIM,
                        DS4_RMS_EPS);
    }
    if (profile) t_kv = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    const uint32_t ratio = cache->compress_ratio;
    const bool prefer_parallel_attn = getenv("DS4_PARALLEL_ATTN_ROWS") != NULL;
    const bool prefix_batch_attn =
        prefer_parallel_attn &&
        getenv("DS4_NO_PARALLEL_ATTN_ROWS") == NULL &&
        cache->n_raw == 0 &&
        pos0 == 0;
    if (!prefix_batch_attn) {
        heads = xmalloc((size_t)n_tok * q_dim * sizeof(heads[0]));
    }
    uint32_t batch_rope_max = 4096;
    const char *batch_rope_max_env = getenv("DS4_BATCHED_ROPE_MAX");
    if (batch_rope_max_env && batch_rope_max_env[0]) {
        long v = strtol(batch_rope_max_env, NULL, 10);
        if (v >= 0 && v <= 65536) batch_rope_max = (uint32_t)v;
    }
    const bool batch_prefix_rope =
        prefix_batch_attn &&
        getenv("DS4_NO_BATCHED_ROPE") == NULL &&
        n_tok <= batch_rope_max;
    uint32_t *comp_counts = prefix_batch_attn ?
        xcalloc((size_t)n_tok, sizeof(comp_counts[0])) : NULL;
    uint8_t *allowed_mask = prefix_batch_attn && ratio == 4 ?
        xcalloc((size_t)n_tok, sizeof(allowed_mask[0])) : NULL;
    uint8_t *allowed_bits = NULL;
    const uint64_t allowed_stride = ratio == 4 ? ((uint64_t)cache->comp_cap + 7u) / 8u : 0;
    float *comp_scratch = NULL;
    float *index_comp_scratch = NULL;

    if (ratio != 0) {
        comp_scratch = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(comp_scratch[0]));

        if (ratio == 4) {
            index_comp_scratch = xmalloc((size_t)DS4_N_INDEXER_HEAD_DIM * sizeof(index_comp_scratch[0]));
        }
    }

    if (batch_prefix_rope) {
        double tx = profile ? now_sec() : 0.0;
        rope_tail_layer_batch_inplace(q,
                                      q_dim,
                                      DS4_N_HEAD,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_ROT,
                                      pos0,
                                      il,
                                      false,
                                      n_tok);
        rope_tail_layer_batch_inplace(kv,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_HEAD_KV,
                                      DS4_N_HEAD_DIM,
                                      DS4_N_ROT,
                                      pos0,
                                      il,
                                      false,
                                      n_tok);
        if (profile) t_tl_rope_cache += now_sec() - tx;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        const uint32_t pos = pos0 + t;
        float *q_t = q + (uint64_t)t * q_dim;
        float *kv_t = kv + (uint64_t)t * DS4_N_HEAD_DIM;
        bool *comp_allowed = NULL;

        double tx = profile ? now_sec() : 0.0;
        if (!batch_prefix_rope) {
            rope_tail_layer_inplace(q_t, DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
            rope_tail_layer_inplace(kv_t, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_ROT, pos, il, false);
        }
        dsv4_fp8_kv_quantize_row_inplace_cpu(kv_t, DS4_N_HEAD_DIM, DS4_N_ROT);

        kv_cache_push_raw(cache, kv_t);
        if (profile) t_tl_rope_cache += now_sec() - tx;

        if (ratio != 0) {
            tx = profile ? now_sec() : 0.0;
            float *comp = comp_scratch;
            const bool have_comp = compressor_decode_one(comp, model,
                                                         layer->attn_compressor_kv,
                                                         layer->attn_compressor_gate,
                                                         layer->attn_compressor_ape,
                                                         layer->attn_compressor_norm,
                                                         attn_norm + (uint64_t)t * DS4_N_EMBD,
                                                         cache->attn_state_kv,
                                                         cache->attn_state_score,
                                                         DS4_N_HEAD_DIM,
                                                         ratio,
                                                         il,
                                                         pos);
            if (have_comp) {
                kv_cache_push_comp(cache->attn_comp_kv, &cache->n_comp, cache->comp_cap, DS4_N_HEAD_DIM, comp);
            }

            if (ratio == 4) {
                float *index_comp = index_comp_scratch;
                const bool have_index_comp = compressor_decode_one(index_comp, model,
                                                                   layer->indexer_compressor_kv,
                                                                   layer->indexer_compressor_gate,
                                                                   layer->indexer_compressor_ape,
                                                                   layer->indexer_compressor_norm,
                                                                   attn_norm + (uint64_t)t * DS4_N_EMBD,
                                                                   cache->index_state_kv,
                                                                   cache->index_state_score,
                                                                   DS4_N_INDEXER_HEAD_DIM,
                                                                   ratio,
                                                                   il,
                                                                   pos);
                if (have_index_comp) {
                    kv_cache_push_comp(cache->index_comp_kv, &cache->n_index_comp, cache->comp_cap, DS4_N_INDEXER_HEAD_DIM, index_comp);
                }
                if (profile) t_tl_compress += now_sec() - tx;

                tx = profile ? now_sec() : 0.0;
                comp_allowed = indexer_allowed_decode_one(model, layer,
                                                          attn_norm + (uint64_t)t * DS4_N_EMBD,
                                                          qr_norm + (uint64_t)t * q_rank,
                                                          cache->index_comp_kv,
                                                          cache->n_index_comp,
                                                          il, pos);
                if (profile) t_tl_indexer += now_sec() - tx;
            } else {
                if (profile) t_tl_compress += now_sec() - tx;
            }

            if (comp_counts) comp_counts[t] = cache->n_comp;
            if (prefix_batch_attn && comp_allowed) {
                if (!allowed_bits) {
                    allowed_bits = xcalloc((size_t)n_tok * allowed_stride, sizeof(allowed_bits[0]));
                }
                allowed_mask[t] = 1;
                uint8_t *bits = allowed_bits + (uint64_t)t * allowed_stride;
                for (uint32_t c = 0; c < cache->n_comp; c++) {
                    if (comp_allowed[c]) bits[c >> 3] |= (uint8_t)(1u << (c & 7u));
                }
            }

            if (!prefix_batch_attn) {
                tx = profile ? now_sec() : 0.0;
                layer_attention_mixed_one(heads + (uint64_t)t * q_dim, model, layer, q_t,
                                          cache->raw_kv, cache->n_raw,
                                          cache->attn_comp_kv, cache->n_comp,
                                          comp_allowed);
                if (profile) t_tl_attn_rows += now_sec() - tx;
            }
        } else {
            if (!prefix_batch_attn) {
                tx = profile ? now_sec() : 0.0;
                layer_attention_rows_one(heads + (uint64_t)t * q_dim, model, layer, q_t, cache->raw_kv, cache->n_raw);
                if (profile) t_tl_attn_rows += now_sec() - tx;
            }
        }

        if (!prefix_batch_attn) {
            tx = profile ? now_sec() : 0.0;
            rope_tail_layer_inplace(heads + (uint64_t)t * q_dim,
                                    DS4_N_HEAD,
                                    DS4_N_HEAD_DIM,
                                    DS4_N_ROT,
                                    pos,
                                    il,
                                    true);
            if (profile) t_tl_inv_rope += now_sec() - tx;
        }

        free(comp_allowed);
    }

    if (prefix_batch_attn) {
        double tx = profile ? now_sec() : 0.0;
        const float *comp_kv_for_prefix = cache->attn_comp_kv ? cache->attn_comp_kv : kv;
        if (!heads) {
            heads = xmalloc((size_t)n_tok * q_dim * sizeof(heads[0]));
        }
        layer_attention_prefix_batch(heads, model, layer,
                                     q,
                                     kv,
                                     comp_kv_for_prefix,
                                     comp_counts,
                                     allowed_mask,
                                     allowed_bits,
                                     allowed_stride,
                                     n_tok,
                                     cache->cap_raw);
        if (profile) t_tl_attn_rows += now_sec() - tx;
        tx = profile ? now_sec() : 0.0;
        if (batch_prefix_rope) {
            rope_tail_layer_batch_inplace(heads,
                                          q_dim,
                                          DS4_N_HEAD,
                                          DS4_N_HEAD_DIM,
                                          DS4_N_ROT,
                                          pos0,
                                          il,
                                          true,
                                          n_tok);
        } else {
            for (uint32_t t = 0; t < n_tok; t++) {
                rope_tail_layer_inplace(heads + (uint64_t)t * q_dim,
                                        DS4_N_HEAD,
                                        DS4_N_HEAD_DIM,
                                        DS4_N_ROT,
                                        pos0 + t,
                                        il,
                                        true);
            }
        }
        if (profile) t_tl_inv_rope += now_sec() - tx;
    }
    if (profile) t_token_loop = now_sec() - t0;

    t0 = profile ? now_sec() : 0.0;
    layer_grouped_out_batch(attn_out, model, layer, heads, n_tok);
    cpu_directional_steering_project_rows(attn_out, steering_dirs, il, n_tok, steering_scale);

    hc_post_batch(after_attn_hc,
                  attn_out,
                  attn_residual,
                  post,
                  comb,
                  n_tok,
                  DS4_N_EMBD,
                  n_hc);
    if (profile) t_out = now_sec() - t0;

    if (profile) {
        fprintf(stderr,
                "ds4: prefill detail layer %u attn hc_norm=%.3f q=%.3f kv=%.3f token_loop=%.3f out=%.3f total=%.3f\n",
                il, t_hc_norm, t_q, t_kv, t_token_loop, t_out, now_sec() - t_start);
        if (getenv("DS4_PREFILL_PROFILE_TOKEN") != NULL) {
            fprintf(stderr,
                    "ds4: prefill token detail layer %u rope_cache=%.3f compress=%.3f indexer=%.3f attn_rows=%.3f inv_rope=%.3f\n",
                    il, t_tl_rope_cache, t_tl_compress, t_tl_indexer, t_tl_attn_rows, t_tl_inv_rope);
        }
    }

    free(allowed_bits);
    free(allowed_mask);
    free(comp_counts);
    free(index_comp_scratch);
    free(comp_scratch);
    free(comb);
    free(post);
    free(attn_out);
    free(heads);
    free(kv);
    free(kv_raw);
    free(q);
    free(qr_norm);
    free(qr);
    free(attn_residual);
    free(attn_norm);
    free(attn_cur);
}

