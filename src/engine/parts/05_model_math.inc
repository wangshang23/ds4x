static float layer_rope_freq_base(uint32_t il) {
    return ds4_layer_compress_ratio(il) != 0 &&
                   DS4_COMPRESS_ROPE_FREQ_BASE > 0.0f
               ? DS4_COMPRESS_ROPE_FREQ_BASE
               : DS4_ROPE_FREQ_BASE;
}

static float layer_rope_freq_scale(uint32_t il) {
    if (ds4_layer_compress_ratio(il) == 0 ||
        DS4_ROPE_SCALE_FACTOR <= 0.0f) {
        return 1.0f;
    }
    return 1.0f / DS4_ROPE_SCALE_FACTOR;
}

static float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

static float softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void layer_hash_selected_experts(
        int selected[DS4_MAX_EXPERT_USED],
        const ds4_model *model,
        const ds4_layer_weights *layer,
        int token) {
    const ds4_tensor *table_tensor = layer->ffn_gate_tid2eid;
    if (!table_tensor || table_tensor->type != 26u ||
        table_tensor->ndim != 2 ||
        table_tensor->dim[0] != DS4_N_EXPERT_USED) {
        ds4_die("hash routing table has an unexpected layout");
    }
    if (token < 0 || (uint64_t)token >= table_tensor->dim[1]) {
        ds4_die("token id is outside the hash routing table");
    }
    const int32_t *table = tensor_data(model, table_tensor);
    const int32_t *row = table + (uint64_t)token * DS4_N_EXPERT_USED;
    for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) selected[i] = row[i];
}

static inline float dot_q8_0_row(
        const uint8_t *row,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks) {
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
        const int8_t *weights = (const int8_t *)(row + b * 34u + 2u);
        const uint64_t first = b * 32u;
        const uint64_t count = in_dim - first < 32u ? in_dim - first : 32u;
        int32_t dot = 0;
        for (uint64_t i = 0; i < count; i++) {
            dot += (int32_t)weights[i] * (int32_t)xq[first + i];
        }
        acc += f16_to_f32(scale_bits) * xscale[b] * (float)dot;
    }
    return acc;
}

static void quantize_q8_0_activation(const float *x,
                                     int8_t *xq,
                                     float *scale,
                                     uint64_t n) {
    const uint64_t blocks = (n + 31u) / 32u;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t first = b * 32u;
        const uint64_t count = n - first < 32u ? n - first : 32u;
        float max_abs = 0.0f;
        for (uint64_t i = 0; i < count; i++) {
            const float value = fabsf(x[first + i]);
            if (value > max_abs) max_abs = value;
        }
        const float d = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        scale[b] = d;
        for (uint64_t i = 0; i < count; i++) {
            int value = (int)lrintf(x[first + i] / d);
            if (value < -127) value = -127;
            if (value > 127) value = 127;
            xq[first + i] = (int8_t)value;
        }
        for (uint64_t i = count; i < 32u; i++) xq[first + i] = 0;
    }
}

static void matvec_any(float *out,
                       const ds4_model *model,
                       const ds4_tensor *weight,
                       const float *x) {
    const uint64_t in_dim = weight->dim[0];
    const uint64_t out_dim = weight->dim[1];
    if (weight->type == DS4_TENSOR_F16) {
        const uint16_t *rows = tensor_data(model, weight);
        for (uint64_t row = 0; row < out_dim; row++) {
            float sum = 0.0f;
            const uint16_t *w = rows + row * in_dim;
            for (uint64_t i = 0; i < in_dim; i++) {
                sum += f16_to_f32(w[i]) * x[i];
            }
            out[row] = sum;
        }
        return;
    }
    if (weight->type == DS4_TENSOR_Q8_0) {
        const uint64_t blocks = (in_dim + 31u) / 32u;
        int8_t *xq = xmalloc((size_t)blocks * 32u);
        float *scale = xmalloc((size_t)blocks * sizeof(scale[0]));
        quantize_q8_0_activation(x, xq, scale, in_dim);
        const uint8_t *rows = tensor_data(model, weight);
        const uint64_t row_bytes = blocks * 34u;
        for (uint64_t row = 0; row < out_dim; row++) {
            out[row] = dot_q8_0_row(rows + row * row_bytes,
                                    xq, scale, in_dim, blocks);
        }
        free(scale);
        free(xq);
        return;
    }
    ds4_die("unsupported DSpark dense tensor type");
}

static uint32_t ds4_prefill_cap_for_prompt(int prompt_len,
                                           uint32_t requested_chunk) {
    if (prompt_len <= 0) return 1u;
    uint32_t cap = (uint32_t)prompt_len;
    if (requested_chunk != 0u) {
        cap = requested_chunk;
    } else {
        const char *env = getenv("DS4_CUDA_PREFILL_CHUNK");
        if (!env || !env[0]) env = getenv("DS4_METAL_PREFILL_CHUNK");
        if (env && env[0]) {
            char *end = NULL;
            const long value = strtol(env, &end, 10);
            if (end != env && value > 0) cap = (uint32_t)value;
        } else if (prompt_len > 4096) {
            cap = 4096u;
        }
    }
    if (cap == 0u) cap = 1u;
    if (cap > (uint32_t)prompt_len) cap = (uint32_t)prompt_len;
    return cap;
}
