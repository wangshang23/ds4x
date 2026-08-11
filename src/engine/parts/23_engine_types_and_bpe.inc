
typedef struct ds4_vocab ds4_vocab;

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
    int user_id;
    int assistant_id;
    int think_start_id;
    int think_end_id;
    int dsml_id;
    str_i32_table token_to_id;
    str_i32_table merge_rank;
};

struct ds4_engine {
    ds4_model model;
    ds4_model support_model;
    ds4_vocab vocab;
    ds4_weights weights;
    ds4_dspark_weights dspark_weights;
    ds4_backend backend;
    ds4_support_kind support_kind;
    int dspark_exec_tier;
    uint32_t support_stages;
    float dspark_confidence_threshold;
    char *directional_steering_file;
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    int power_percent;
    uint32_t prefill_chunk;
    uint64_t startup_model_span_bytes;
    ds4_memory_lock simulated_memory;
    bool quality;
    bool dspark;
    bool dspark_strict;
    bool metal_ready;
};

static void ds4_engine_print_startup_memory(
        const ds4_engine *e,
        int               ctx_size) {
    if (!e || ctx_size <= 0) return;

    ds4_context_memory mem;
    mem = ds4_context_memory_estimate_with_prefill(e->backend, ctx_size,
                                                   e->prefill_chunk);
    const uint64_t kv_bytes =
        ds4_add_sat_u64(mem.raw_bytes, mem.compressed_bytes);
    uint64_t total = kv_bytes;
    total = ds4_add_sat_u64(total, mem.scratch_bytes);
    total = ds4_add_sat_u64(total, e->startup_model_span_bytes);

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
