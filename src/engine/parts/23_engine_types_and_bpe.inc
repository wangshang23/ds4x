
typedef struct ds4_vocab ds4_vocab;

static void embed_prompt(
        const ds4_model   * model,
        const ds4_weights * weights,
        const token_vec   * tokens,
        uint32_t            n_embd,
        float             * out) {
    for (int i = 0; i < tokens->len; i++) {
        embed_token_any(model, weights, tokens->v[i], out + (uint64_t)i * n_embd);
    }
}

/* =========================================================================
 * Tokenizer and Chat Prompt Encoding.
 * =========================================================================
 *
 * DeepSeek V4 Flash stores a GPT-2 style byte-level BPE tokenizer in GGUF.
 * The implementation below is intentionally small.  It loads token strings
 * and merge ranks from the mmaped file, builds two open-addressed hash tables,
 * and applies BPE to user text.  Chat special tokens are inserted directly by
 * ID; user text goes through BPE.
 */

typedef struct {
    ds4_str key;
    int value;
    bool used;
} str_i32_entry;

typedef struct {
    str_i32_entry *entry;
    uint64_t cap;
    uint64_t used;
} str_i32_table;

static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->used = 0;
    t->entry = xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}

static void table_free(str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void table_put(str_i32_table *t, ds4_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;

    while (t->entry[i].used) {
        if (ds4_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }

    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}

static bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    if (t->cap == 0) return false;

    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(ptr, len) & mask;

    while (t->entry[i].used) {
        ds4_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

static void token_vec_push(token_vec *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}

static void token_vec_free(token_vec *tv) {
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}

void ds4_tokens_push(ds4_tokens *tv, int token) {
    token_vec_push(tv, token);
}

void ds4_tokens_free(ds4_tokens *tv) {
    token_vec_free(tv);
}

void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) token_vec_push(dst, src->v[i]);
}

bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix) {
    if (prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}

struct ds4_vocab {
    ds4_str *token;
    int n_vocab;
    int bos_id;
    int eos_id;
    int system_id;
    int user_id;
    int assistant_id;
    int observation_id;
    int sop_id;
    int think_start_id;
    int think_end_id;
    int tool_call_start_id;
    int tool_call_end_id;
    int tool_response_start_id;
    int tool_response_end_id;
    int arg_key_start_id;
    int arg_key_end_id;
    int arg_value_start_id;
    int arg_value_end_id;
    int dsml_id;
    str_i32_table token_to_id;
    str_i32_table merge_rank;
};

/* Engine-side tensor-parallel state.  The transport context is owned by the
 * frontend (CLI leader or ds4_tp_worker_run); the engine owns the GPU slab,
 * the per-slot views and the gate machinery lifetime. */
typedef struct {
    struct ds4_tp *ctx;
    ds4_gpu_tensor *slab;
    ds4_gpu_tensor **out_views;
    ds4_gpu_tensor **in_views;
    ds4_gpu_tensor **batch_out_views;   /* [layer] verify-block row partials */
    ds4_gpu_tensor **batch_in_views;
    ds4_gpu_tensor *zero_vec;
    uint64_t eval_seq;          /* leader: mirrored eval counter */
    uint64_t next_session_id;   /* leader: stable worker-session handle */
    int rank;
    bool vocab_split;           /* DS4-only: logits halves cross the wire */
    bool active;
} ds4_engine_tp_state;

struct ds4_engine {
    ds4_model model;
    ds4_model mtp_model;
    ds4_vocab vocab;
    ds4_weights weights;
    ds4_mtp_weights mtp_weights;
    ds4_dspark_weights dspark_weights;
    ds4_backend backend;
    ds4_support_kind support_kind;
    int dspark_exec_tier;
    uint32_t support_stages;
    int mtp_draft_tokens;
    float mtp_margin;
    float dspark_confidence_threshold;
    char *directional_steering_file;
    float *directional_steering_dirs;
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    int power_percent;
    uint32_t prefill_chunk;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint64_t ssd_streaming_prefill_headroom_bytes;
    uint64_t ssd_streaming_full_layer_bytes;
    uint32_t ssd_streaming_full_layers;
    uint32_t ssd_streaming_preload_experts;
    uint64_t startup_model_span_bytes;
    ds4_ssd_memory_lock simulated_memory;
    bool quality;
    bool glm_mtp;
    bool glm_mtp_timing;
    bool dspark;
    bool dspark_strict;
    bool cuda_tensor_parallel;
    bool glm_tp_token_prefill;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
    ds4_distributed_options distributed;
    ds4_engine_tp_state tp;
    bool metal_ready;
    bool mtp_ready;
    bool share_session_prefill_workspace;
#ifndef DS4_NO_GPU
    bool shared_prefill_workspace_ready;
    ds4_gpu_graph shared_prefill_workspace;
#endif

    /* Wave-2 multi-GPU placement scaffolding: optional multi-GPU placement
     * state. Zero-initialized for every existing caller (gpu_cfg == NULL)
     * via xcalloc, so the single-tier path observes identical engine
     * state to pre-multi-GPU CLI main. multi_tier == 1 is the gate for all
     * new code paths. */
    ds4_gpu_config gpu_cfg;
    int            placement[DS4_MAX_LAYER + 2];
    int            n_placement_entries;
    int            multi_tier;

    /* Max-context hint copied from
     * ds4_engine_options.placement_ctx_hint. Used by
     * engine_compute_entry_bytes for per-layer KV estimation.
     * Zero / negative = legacy 4096 fallback (single-tier paths and any
     * caller that doesn't set the option observe the prior behavior). */
    int            placement_ctx_hint;
    int            placement_session_count_hint;
};

static uint64_t ds4_engine_dynamic_expert_cache_bytes(
        const ds4_engine *e) {
    if (!e || !e->ssd_streaming) return 0;
    if (e->ssd_streaming_cache_bytes != 0) {
        return e->ssd_streaming_cache_bytes;
    }
    if (e->ssd_streaming_cache_experts == 0) return 0;

    uint64_t per_expert_bytes = 0;
    if (!ds4_streaming_routed_expert_bytes(&e->weights,
                                           &per_expert_bytes)) {
        return 0;
    }
    if (e->ssd_streaming_cache_experts > UINT64_MAX / per_expert_bytes) {
        return UINT64_MAX;
    }
    return (uint64_t)e->ssd_streaming_cache_experts * per_expert_bytes;
}

static uint64_t ds4_engine_streaming_transient_guard_bytes(
        const ds4_engine *e) {
    if (!e || !e->ssd_streaming) return 0;
    uint64_t total = ds4_engine_dynamic_expert_cache_bytes(e);
    total = ds4_add_sat_u64(total, e->ssd_streaming_full_layer_bytes);
    total = ds4_add_sat_u64(total, e->ssd_streaming_prefill_headroom_bytes);
    return total;
}

static void ds4_engine_print_startup_memory(
        const ds4_engine *e,
        int               ctx_size) {
    if (!e || ctx_size <= 0) return;

    ds4_context_memory mem;
#ifndef DS4_NO_GPU
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA &&
        e->distributed.role != DS4_DISTRIBUTED_NONE &&
        e->distributed.layers.set) {
        const uint32_t normal_layers = glm_graph_normal_layer_count();
        const uint32_t layer_end = e->distributed.layers.has_output ?
            (normal_layers ? normal_layers - 1u : 0u) :
            e->distributed.layers.end;
        const uint32_t ctx = (uint32_t)ctx_size;
        const uint32_t work_ctx =
            glm_graph_full_attention_cap(ctx, e->ssd_streaming);
        const uint32_t compact_cap =
            glm_graph_compact_cache_initial_cap(ctx, work_ctx);
        mem = glm_graph_context_memory_estimate_for_compact_cap_slice(
                ctx,
                work_ctx,
                compact_cap,
                e->ssd_streaming,
                e->distributed.layers.start,
                layer_end);
    } else {
#endif
        mem = ds4_context_memory_estimate_with_prefill_mode(e->backend,
                                                            ctx_size,
                                                            e->prefill_chunk,
                                                            e->ssd_streaming);
#ifndef DS4_NO_GPU
    }
#endif
    const uint64_t kv_bytes =
        ds4_add_sat_u64(mem.raw_bytes, mem.compressed_bytes);
    const uint64_t dynamic_expert_cache_bytes =
        ds4_engine_dynamic_expert_cache_bytes(e);
    const uint64_t expert_reserved_bytes =
        e->ssd_streaming_prefill_headroom_bytes;
    uint64_t total = kv_bytes;
    total = ds4_add_sat_u64(total, mem.scratch_bytes);
    total = ds4_add_sat_u64(total, e->startup_model_span_bytes);
    total = ds4_add_sat_u64(total, dynamic_expert_cache_bytes);
    total = ds4_add_sat_u64(total, e->ssd_streaming_full_layer_bytes);
    total = ds4_add_sat_u64(total, expert_reserved_bytes);

    const bool color = ds4_log_is_tty(stderr);
    const char *green = color ? "\x1b[32m" : "";
    const char *bright_green = color ? "\x1b[1;32m" : "";
    const char *reset = color ? "\x1b[0m" : "";

    fprintf(stderr,
            "%sds4: memory: KV %.2f GiB (raw %.2f + compressed %.2f) "
            "+ buffers %.2f GiB + resident model %.2f GiB",
            green,
            ds4_bytes_to_gib(kv_bytes),
            ds4_bytes_to_gib(mem.raw_bytes),
            ds4_bytes_to_gib(mem.compressed_bytes),
            ds4_bytes_to_gib(mem.scratch_bytes),
            ds4_bytes_to_gib(e->startup_model_span_bytes));
    if (e->ssd_streaming_full_layer_bytes != 0) {
        fprintf(stderr,
                " + full-layer experts %.2f GiB",
                ds4_bytes_to_gib(e->ssd_streaming_full_layer_bytes));
    }
    if (dynamic_expert_cache_bytes != 0) {
        fprintf(stderr,
                " + expert cache %.2f GiB",
                ds4_bytes_to_gib(dynamic_expert_cache_bytes));
    }
    if (expert_reserved_bytes != 0) {
        fprintf(stderr,
                " + prefill expert reserve %.2f GiB",
                ds4_bytes_to_gib(expert_reserved_bytes));
    }
    fprintf(stderr,
            " = %s%.2f GiB planned%s\n",
            bright_green,
            ds4_bytes_to_gib(total),
            reset);

    fprintf(stderr,
            "%sds4: memory detail: ctx=%d prefill_cap=%u raw_kv_rows=%u "
            "compressed_kv_rows=%u backend=%s%s\n",
            green,
            ctx_size,
            mem.prefill_cap,
            mem.raw_cap,
            mem.comp_cap,
            ds4_backend_name(e->backend),
            reset);
}

static bool cpu_directional_steering_enabled(
        const float *dirs,
        float        scale) {
    return dirs && scale != 0.0f;
}

static void cpu_directional_steering_project_rows(
        float       *x,
        const float *dirs,
        uint32_t     il,
        uint32_t     rows,
        float        scale) {
    if (!cpu_directional_steering_enabled(dirs, scale) || !x || rows == 0) return;

    const float *dir = dirs + (uint64_t)il * DS4_N_EMBD;
    for (uint32_t row = 0; row < rows; row++) {
        float *xr = x + (uint64_t)row * DS4_N_EMBD;
        float dot = 0.0f;
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
            dot += xr[i] * dir[i];
        }
        const float coeff = scale * dot;
        for (uint32_t i = 0; i < DS4_N_EMBD; i++) {
            xr[i] -= coeff * dir[i];
        }
    }
}

static bool cpu_load_directional_steering(ds4_engine *e) {
    if (!e ||
        (e->directional_steering_attn_scale == 0.0f &&
         e->directional_steering_ffn_scale == 0.0f)) {
        return true;
    }

    const char *path = e->directional_steering_file;
    if (!path || !path[0]) {
        fprintf(stderr, "ds4: directional steering needs --dir-steering-file\n");
        return false;
    }

    const uint64_t n = (uint64_t)DS4_N_LAYER * DS4_N_EMBD;
    e->directional_steering_dirs = xmalloc((size_t)n * sizeof(e->directional_steering_dirs[0]));
    if (!read_f32_binary_file(path, e->directional_steering_dirs, n)) {
        free(e->directional_steering_dirs);
        e->directional_steering_dirs = NULL;
        fprintf(stderr, "ds4: failed to load directional steering vectors from %s\n", path);
        return false;
    }
    fprintf(stderr, "ds4: CPU directional steering enabled: %s attn=%g ffn=%g\n",
            path,
            (double)e->directional_steering_attn_scale,
            (double)e->directional_steering_ffn_scale);
    return true;
}

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
        return b;
    }

    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) {
            continue;
        }
        if (x == b) return 256 + n;
        n++;
    }
    return b;
}

/* GPT-2 byte-level BPE first maps raw bytes to printable Unicode codepoints
 * so merges can operate on UTF-8 strings without losing byte identity. */
static char *byte_encode(ds4_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4 + 1);
    char *p = out;

    for (uint64_t i = 0; i < in.len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

/* Look up the merge rank for two adjacent BPE symbols. */
static int bpe_rank(const ds4_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);

    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);

    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);

    if (buf != stack) free(buf);
    return rank;
}

/* Apply byte-level BPE to one regex-like pre-tokenized piece and emit token ids. */
static void bpe_emit_piece(const ds4_vocab *vocab, ds4_str raw_piece, token_vec *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);

    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;

        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }

        if (best_i < 0) break;

        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);

        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;

        for (int j = best_i + 1; j + 1 < n_sym; j++) {
            sym[j] = sym[j + 1];
        }
        n_sym--;
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            token_vec_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    token_vec_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }

    free(sym);
    free(encoded);
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

static bool joyai_ascii_punct_symbol(uint8_t c) {
    return (c >= '!' && c <= '/') ||
           (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}

static bool utf8_is_cjk_hira_kata(uint32_t cp) {
    return (cp >= 0x4e00 && cp <= 0x9fa5) ||
           (cp >= 0x3040 && cp <= 0x309f) ||
           (cp >= 0x30a0 && cp <= 0x30ff);
}

static uint32_t utf8_peek_one(const char *s, uint64_t len, uint64_t pos, uint64_t *next) {
    const uint8_t c0 = (uint8_t)s[pos];
    int n = utf8_len_from_first_byte(c0);
    if (pos + (uint64_t)n > len) n = 1;
    *next = pos + (uint64_t)n;

    if (n == 1) return c0;
    if (n == 2) {
        return ((uint32_t)(c0 & 0x1f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f));
    }
    if (n == 3) {
        return ((uint32_t)(c0 & 0x0f) << 12) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 2] & 0x3f));
    }
    return ((uint32_t)(c0 & 0x07) << 18) |
           ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 12) |
           ((uint32_t)((uint8_t)s[pos + 2] & 0x3f) << 6) |
           ((uint32_t)((uint8_t)s[pos + 3] & 0x3f));
}

static bool joyai_letter_like_at(const char *s, uint64_t len, uint64_t pos) {
    (void)len;
    uint8_t c = (uint8_t)s[pos];
    if (c < 128) return ascii_alpha(c);

    /*
     * The JoyAI tokenizer maps Unicode letters into a collapsed regex alphabet before
     * applying the JoyAI pre-tokenizer.  The prompts we care about are mostly
     * ASCII, but treating non-ASCII non-control bytes as letters preserves the
     * useful behavior for ordinary UTF-8 text such as Italian accents.  CJK and
     * kana are isolated by the JoyAI pre-tokenizer before the generic letter
     * rule, below.
     */
    return true;
}

static uint64_t joyai_consume_letters(const char *s, uint64_t len, uint64_t pos) {
    while (pos < len && joyai_letter_like_at(s, len, pos)) {
        pos = next_utf8_char(s, len, pos);
    }
    return pos;
}

static bool joyai_cjk_at(const char *s, uint64_t len, uint64_t pos) {
    if ((uint8_t)s[pos] < 128) return false;
    uint64_t next = pos;
    uint32_t cp = utf8_peek_one(s, len, pos, &next);
    return utf8_is_cjk_hira_kata(cp);
}

typedef struct {
    uint32_t cp;
    uint64_t next;
    bool valid;
    bool is_letter;
    bool is_number;
    bool is_whitespace;
} glm4_char_info;

static bool glm4_unicode_whitespace(uint32_t cp) {
    if (cp < 128) return ascii_space((uint8_t)cp);
    return cp == 0x0085 ||
           cp == 0x00a0 ||
           cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) ||
           cp == 0x2028 ||
           cp == 0x2029 ||
           cp == 0x202f ||
           cp == 0x205f ||
           cp == 0x3000;
}

static bool glm4_unicode_number(uint32_t cp) {
    if (cp < 128) return ascii_digit((uint8_t)cp);
    return (cp >= 0x0660 && cp <= 0x0669) ||
           (cp >= 0x06f0 && cp <= 0x06f9) ||
           (cp >= 0x07c0 && cp <= 0x07c9) ||
           (cp >= 0x0966 && cp <= 0x096f) ||
           (cp >= 0x09e6 && cp <= 0x09ef) ||
           (cp >= 0x0a66 && cp <= 0x0a6f) ||
           (cp >= 0x0ae6 && cp <= 0x0aef) ||
           (cp >= 0x0b66 && cp <= 0x0b6f) ||
           (cp >= 0x0be6 && cp <= 0x0bef) ||
           (cp >= 0x0c66 && cp <= 0x0c6f) ||
           (cp >= 0x0ce6 && cp <= 0x0cef) ||
           (cp >= 0x0d66 && cp <= 0x0d6f) ||
           (cp >= 0x0de6 && cp <= 0x0def) ||
           (cp >= 0x0e50 && cp <= 0x0e59) ||
           (cp >= 0x0ed0 && cp <= 0x0ed9) ||
           (cp >= 0x0f20 && cp <= 0x0f29) ||
           (cp >= 0x1040 && cp <= 0x1049) ||
           (cp >= 0x1090 && cp <= 0x1099) ||
           (cp >= 0x17e0 && cp <= 0x17e9) ||
           (cp >= 0x1810 && cp <= 0x1819) ||
           (cp >= 0xff10 && cp <= 0xff19);
}

static bool glm4_unicode_punct_symbol(uint32_t cp) {
    if (cp < 128) return joyai_ascii_punct_symbol((uint8_t)cp);
    return (cp >= 0x00a1 && cp <= 0x00a9) ||
           (cp >= 0x00ab && cp <= 0x00ac) ||
           (cp >= 0x00ae && cp <= 0x00b1) ||
           cp == 0x00b4 ||
           (cp >= 0x00b6 && cp <= 0x00b8) ||
           cp == 0x00bb ||
           cp == 0x00bf ||
           cp == 0x00d7 ||
           cp == 0x00f7 ||
           (cp >= 0x02c2 && cp <= 0x02df) ||
           (cp >= 0x02e5 && cp <= 0x02eb) ||
           (cp >= 0x02ed && cp <= 0x02ff) ||
           (cp >= 0x0375 && cp <= 0x037e) ||
           (cp >= 0x0384 && cp <= 0x0385) ||
           cp == 0x0387 ||
           (cp >= 0x055a && cp <= 0x055f) ||
           (cp >= 0x0589 && cp <= 0x058a) ||
           (cp >= 0x05be && cp <= 0x05c0) ||
           cp == 0x05c3 ||
           (cp >= 0x05c6 && cp <= 0x05c7) ||
           (cp >= 0x0609 && cp <= 0x060a) ||
           (cp >= 0x060c && cp <= 0x060d) ||
           cp == 0x061b ||
           (cp >= 0x061e && cp <= 0x061f) ||
           cp == 0x066a ||
           cp == 0x066d ||
           cp == 0x06d4 ||
           (cp >= 0x2000 && cp <= 0x206f) ||
           (cp >= 0x20a0 && cp <= 0x20cf) ||
           (cp >= 0x2100 && cp <= 0x214f) ||
           (cp >= 0x2190 && cp <= 0x23ff) ||
           (cp >= 0x2460 && cp <= 0x24ff) ||
           (cp >= 0x2500 && cp <= 0x2775) ||
           (cp >= 0x2794 && cp <= 0x2bff) ||
           (cp >= 0x2e00 && cp <= 0x2e7f) ||
           (cp >= 0x3000 && cp <= 0x303f) ||
           (cp >= 0xfd3e && cp <= 0xfd3f) ||
           (cp >= 0xfe10 && cp <= 0xfe6f) ||
           (cp >= 0xff01 && cp <= 0xff0f) ||
           (cp >= 0xff1a && cp <= 0xff20) ||
           (cp >= 0xff3b && cp <= 0xff40) ||
           (cp >= 0xff5b && cp <= 0xff65) ||
           (cp >= 0x1f000 && cp <= 0x1faff);
}

static glm4_char_info glm4_char_at(const char *s, uint64_t len, uint64_t pos) {
    glm4_char_info info;
    memset(&info, 0, sizeof(info));
    if (pos >= len) return info;

    info.valid = true;
    info.cp = utf8_peek_one(s, len, pos, &info.next);
    info.is_whitespace = glm4_unicode_whitespace(info.cp);
    info.is_number = glm4_unicode_number(info.cp);
    if (info.cp < 128) {
        info.is_letter = ascii_alpha((uint8_t)info.cp);
    } else {
        info.is_letter =
            !info.is_whitespace &&
            !info.is_number &&
            !glm4_unicode_punct_symbol(info.cp);
    }
    return info;
}

static uint32_t ascii_tolower_cp(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + ('a' - 'A');
    return cp;
}

/* ChatGLM4/GLM pre-tokenization.  GLM GGUFs use tokenizer.ggml.pre="glm4",
 * which shares the llama3-style split shape used by llama.cpp's CHATGLM4 path. */
static void bpe_tokenize_text_glm4(const ds4_vocab *vocab, const char *text, token_vec *out) {
    const uint64_t len = strlen(text);
    uint64_t pos = 0;

    while (pos < len) {
        uint64_t start = pos;
        glm4_char_info cur = glm4_char_at(text, len, pos);

        if (!cur.valid) break;

        if (cur.cp == '\'' && cur.next < len) {
            glm4_char_info next = glm4_char_at(text, len, cur.next);
            uint32_t n1 = ascii_tolower_cp(next.cp);
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                pos = next.next;
                bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
                continue;
            }
            if (next.valid && next.next < len) {
                glm4_char_info next2 = glm4_char_at(text, len, next.next);
                uint32_t n2 = ascii_tolower_cp(next2.cp);
                if ((n1 == 'r' && n2 == 'e') ||
                    (n1 == 'v' && n2 == 'e') ||
                    (n1 == 'l' && n2 == 'l')) {
                    pos = next2.next;
                    bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
                    continue;
                }
            }
        }

        if (!(cur.cp == '\r' || cur.cp == '\n' || cur.is_number)) {
            glm4_char_info next = glm4_char_at(text, len, cur.next);
            if (cur.is_letter || next.is_letter) {
                pos = cur.next;
                while (pos < len) {
                    glm4_char_info scan = glm4_char_at(text, len, pos);
                    if (!scan.valid || !scan.is_letter) break;
                    pos = scan.next;
                }
                bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
                continue;
            }
        }

        if (cur.is_number) {
            int ndigits = 0;
            while (pos < len && ndigits < 3) {
                glm4_char_info scan = glm4_char_at(text, len, pos);
                if (!scan.valid || !scan.is_number) break;
                pos = scan.next;
                ndigits++;
            }
            bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
            continue;
        }

        glm4_char_info punct = cur;
        uint64_t punct_pos = pos;
        if (cur.cp == ' ') {
            punct_pos = cur.next;
            punct = glm4_char_at(text, len, punct_pos);
        }
        if (punct.valid &&
            !punct.is_whitespace &&
            !punct.is_letter &&
            !punct.is_number) {
            pos = punct_pos;
            while (pos < len) {
                glm4_char_info scan = glm4_char_at(text, len, pos);
                if (!scan.valid ||
                    scan.is_whitespace ||
                    scan.is_letter ||
                    scan.is_number) {
                    break;
                }
                pos = scan.next;
            }
            while (pos < len) {
                glm4_char_info scan = glm4_char_at(text, len, pos);
                if (!scan.valid || !(scan.cp == '\r' || scan.cp == '\n')) break;
                pos = scan.next;
            }
            bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
            continue;
        }

        if (cur.is_whitespace) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            uint64_t last_ws_start = pos;
            int nspace = 0;
            while (p < len) {
                glm4_char_info scan = glm4_char_at(text, len, p);
                if (!scan.valid || !scan.is_whitespace) break;
                last_ws_start = p;
                if (scan.cp == '\r' || scan.cp == '\n') last_newline_end = scan.next;
                p = scan.next;
                nspace++;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (nspace > 1 && p < len) {
                pos = last_ws_start;
            } else {
                pos = p;
            }
            bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
            continue;
        }

        pos = cur.next;
        if (pos == start) pos = next_utf8_char(text, len, pos);
        bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
    }
}

/*
 * DeepSeek V4 Flash declares tokenizer.ggml.pre = "joyai-llm".  The split
 * below mirrors the JoyAI BPE pre-tokenizer for the cases this model
 * uses in normal text and source-code prompts:
 *
 *   \p{N}{1,3}
 *   [CJK/Hiragana/Katakana]+
 *   [P/S][A-Za-z]+
 *   [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
 *    ?[\p{P}\p{S}]+[\r\n]*
 *   \s*[\r\n]+
 *   \s+(?!\S)
 *   \s+
 *
 * The punctuation rule intentionally keeps trailing newlines in the same BPE
 * word (for example ">;\n").  Splitting those newlines separately changes the
 * token stream for code prompts and produces wrong long-context logits.
 */
static void bpe_tokenize_text(const ds4_vocab *vocab, const char *text, token_vec *out) {
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        bpe_tokenize_text_glm4(vocab, text, out);
        return;
    }

    const uint64_t len = strlen(text);
    uint64_t pos = 0;

    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];

        if (ascii_digit(c)) {
            int ndigits = 0;
            while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                pos++;
                ndigits++;
            }
        } else if (joyai_cjk_at(text, len, pos)) {
            do {
                pos = next_utf8_char(text, len, pos);
            } while (pos < len && joyai_cjk_at(text, len, pos));
        } else if (joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   ascii_alpha((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && ascii_alpha((uint8_t)text[pos])) pos++;
        } else if (joyai_letter_like_at(text, len, pos)) {
            pos = joyai_consume_letters(text, len, pos);
        } else if (!ascii_newline(c) &&
                   !joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   joyai_letter_like_at(text, len, pos + 1)) {
            pos++;
            pos = joyai_consume_letters(text, len, pos);
        } else if (c == ' ' &&
                   pos + 1 < len &&
                   joyai_ascii_punct_symbol((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (joyai_ascii_punct_symbol(c)) {
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (p < len && p > pos + 1 &&
                       (joyai_letter_like_at(text, len, p) ||
                        joyai_ascii_punct_symbol((uint8_t)text[p]))) {
                /*
                 * JoyAI lets a single leading space join the following word or
                 * punctuation run.  For "    int", the pre-tokenizer therefore emits
                 * "   " then " int", not "    " then "int".
                 */
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(text, len, pos);
        }

        if (pos == start) pos = next_utf8_char(text, len, pos);
        bpe_emit_piece(vocab, (ds4_str){ text + start, pos - start }, out);
    }
}

static int vocab_lookup(const ds4_vocab *vocab, const char *text) {
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) {
        fprintf(stderr, "ds4: required tokenizer token is missing: %s\n", text);
        exit(1);
    }
    return token;
}

static int vocab_lookup_optional(const ds4_vocab *vocab, const char *text) {
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) return -1;
    return token;
}

