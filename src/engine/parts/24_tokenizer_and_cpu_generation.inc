/* Load token strings, special token ids, and merge ranks from GGUF metadata. */

static void vocab_load(ds4_vocab *vocab, const ds4_model *model) {
    memset(vocab, 0, sizeof(*vocab));

    ds4_array_ref tokens;
    ds4_array_ref merges;
    if (!model_get_array(model, "tokenizer.ggml.tokens", &tokens) ||
        tokens.type != GGUF_VALUE_STRING ||
        tokens.len > INT32_MAX) {
        ds4_die("GGUF tokenizer token table is missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.merges", &merges) ||
        merges.type != GGUF_VALUE_STRING) {
        ds4_die("GGUF tokenizer merge table is missing or invalid");
    }

    vocab->n_vocab = (int)tokens.len;
    vocab->token = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    table_init(&vocab->token_to_id, tokens.len);

    ds4_cursor c = cursor_at(model, tokens.data_pos);
    for (int i = 0; i < vocab->n_vocab; i++) {
        if (!cursor_string(&c, &vocab->token[i])) ds4_die(c.error);
        table_put(&vocab->token_to_id, vocab->token[i], i);
    }

    table_init(&vocab->merge_rank, merges.len);
    c = cursor_at(model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        ds4_str merge;
        if (!cursor_string(&c, &merge)) ds4_die(c.error);
        table_put(&vocab->merge_rank, merge, (int)i);
    }

    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        if (!model_get_token_id(model, "tokenizer.ggml.bos_token_id", &vocab->bos_id)) {
            vocab->bos_id = vocab_lookup_optional(vocab, "<sop>");
        }
        if (!model_get_token_id(model, "tokenizer.ggml.eos_token_id", &vocab->eos_id)) {
            vocab->eos_id = vocab_lookup_optional(vocab, "<|endoftext|>");
        }
        vocab->system_id = vocab_lookup_optional(vocab, "<|system|>");
        vocab->user_id = vocab_lookup_optional(vocab, "<|user|>");
        vocab->assistant_id = vocab_lookup_optional(vocab, "<|assistant|>");
        vocab->observation_id = vocab_lookup_optional(vocab, "<|observation|>");
        vocab->sop_id = vocab_lookup_optional(vocab, "<sop>");
        vocab->think_start_id = vocab_lookup_optional(vocab, "<think>");
        vocab->think_end_id = vocab_lookup_optional(vocab, "</think>");
        vocab->tool_call_start_id = vocab_lookup_optional(vocab, "<tool_call>");
        vocab->tool_call_end_id = vocab_lookup_optional(vocab, "</tool_call>");
        vocab->tool_response_start_id = vocab_lookup_optional(vocab, "<tool_response>");
        vocab->tool_response_end_id = vocab_lookup_optional(vocab, "</tool_response>");
        vocab->arg_key_start_id = vocab_lookup_optional(vocab, "<arg_key>");
        vocab->arg_key_end_id = vocab_lookup_optional(vocab, "</arg_key>");
        vocab->arg_value_start_id = vocab_lookup_optional(vocab, "<arg_value>");
        vocab->arg_value_end_id = vocab_lookup_optional(vocab, "</arg_value>");
        vocab->dsml_id = -1;
        return;
    }

    vocab->bos_id       = vocab_lookup(vocab, "<｜begin▁of▁sentence｜>");
    vocab->eos_id       = vocab_lookup(vocab, "<｜end▁of▁sentence｜>");
    vocab->system_id    = -1;
    vocab->user_id      = vocab_lookup(vocab, "<｜User｜>");
    vocab->assistant_id = vocab_lookup(vocab, "<｜Assistant｜>");
    vocab->observation_id = -1;
    vocab->sop_id = -1;
    vocab->think_start_id = vocab_lookup(vocab, "<think>");
    vocab->think_end_id = vocab_lookup(vocab, "</think>");
    vocab->tool_call_start_id = -1;
    vocab->tool_call_end_id = -1;
    vocab->tool_response_start_id = -1;
    vocab->tool_response_end_id = -1;
    vocab->arg_key_start_id = -1;
    vocab->arg_key_end_id = -1;
    vocab->arg_value_start_id = -1;
    vocab->arg_value_end_id = -1;
    vocab->dsml_id = vocab_lookup(vocab, "｜DSML｜");
}

static void vocab_free(ds4_vocab *vocab) {
    free(vocab->token);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}

/* Build the DS4 chat prompt: BOS, optional system text, user prompt, assistant
 * marker, and either <think> or </think> depending on the requested mode.  Max
 * thinking is only a prompt prefix: the model still enters through <think>. */
static void chat_push_bos_sequence(const ds4_vocab *vocab, token_vec *out) {
    token_vec_push(out, vocab->bos_id);
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA && vocab->sop_id >= 0)
        token_vec_push(out, vocab->sop_id);
}

const char *ds4_glm_reasoning_effort_text(ds4_think_mode mode) {
    switch (mode) {
    case DS4_THINK_HIGH: return "Reasoning Effort: High";
    case DS4_THINK_MAX:  return "Reasoning Effort: Max";
    case DS4_THINK_NONE: return NULL;
    }
    return NULL;
}

static void chat_push_think_prefix(const ds4_vocab *vocab,
                                   ds4_think_mode   think_mode,
                                   token_vec       *out) {
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        const char *effort = ds4_glm_reasoning_effort_text(think_mode);
        if (effort) {
            token_vec_push(out, vocab->system_id);
            bpe_tokenize_text(vocab, effort, out);
        }
    } else if (think_mode == DS4_THINK_MAX) {
        bpe_tokenize_text(vocab, DS4_REASONING_EFFORT_MAX_PREFIX, out);
    }
}

static void encode_chat_prompt(
        const ds4_vocab *vocab,
        const char      *system,
        const char      *prompt,
        ds4_think_mode   think_mode,
        token_vec       *out) {
    const bool need_think_start =
        ds4_think_mode_enabled(think_mode) ||
        DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA;
    if (vocab->bos_id < 0 ||
        vocab->user_id < 0 ||
        vocab->assistant_id < 0 ||
        vocab->think_end_id < 0 ||
        (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA && vocab->system_id < 0) ||
        (need_think_start && vocab->think_start_id < 0)) {
        ds4_die("this tokenizer does not provide the DeepSeek chat markers; use raw prompt tokenization");
    }

    chat_push_bos_sequence(vocab, out);
    chat_push_think_prefix(vocab, think_mode, out);
    if (system && system[0]) {
        if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA)
            token_vec_push(out, vocab->system_id);
        bpe_tokenize_text(vocab, system, out);
    }
    token_vec_push(out, vocab->user_id);
    bpe_tokenize_text(vocab, prompt, out);
    token_vec_push(out, vocab->assistant_id);
    if (ds4_think_mode_enabled(think_mode)) {
        token_vec_push(out, vocab->think_start_id);
    } else if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        token_vec_push(out, vocab->think_start_id);
        token_vec_push(out, vocab->think_end_id);
    } else {
        token_vec_push(out, vocab->think_end_id);
    }
}

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out) {
    bpe_tokenize_text(&e->vocab, text ? text : "", out);
}

static bool special_token_at(const ds4_vocab *vocab, const char *p, int *token, size_t *len) {
    struct special {
        const char *text;
        int token;
    } specials[] = {
        {"<｜begin▁of▁sentence｜>", vocab->bos_id},
        {"<｜end▁of▁sentence｜>",   vocab->eos_id},
        {"[gMASK]",                vocab->bos_id},
        {"<sop>",                  vocab->sop_id},
        {"<|system|>",             vocab->system_id},
        {"<｜User｜>",              vocab->user_id},
        {"<｜Assistant｜>",         vocab->assistant_id},
        {"<|user|>",               vocab->user_id},
        {"<|assistant|>",          vocab->assistant_id},
        {"<|observation|>",        vocab->observation_id},
        {"<think>",                vocab->think_start_id},
        {"</think>",               vocab->think_end_id},
        {"<tool_call>",            vocab->tool_call_start_id},
        {"</tool_call>",           vocab->tool_call_end_id},
        {"<tool_response>",        vocab->tool_response_start_id},
        {"</tool_response>",       vocab->tool_response_end_id},
        {"<arg_key>",              vocab->arg_key_start_id},
        {"</arg_key>",             vocab->arg_key_end_id},
        {"<arg_value>",            vocab->arg_value_start_id},
        {"</arg_value>",           vocab->arg_value_end_id},
        {"｜DSML｜",                vocab->dsml_id},
    };

    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        if (specials[i].token < 0) continue;
        size_t n = strlen(specials[i].text);
        if (!strncmp(p, specials[i].text, n)) {
            *token = specials[i].token;
            *len = n;
            return true;
        }
    }
    return false;
}

static void tokenize_span(const ds4_vocab *vocab, const char *p, size_t n, token_vec *out) {
    if (!n) return;
    char *tmp = xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    bpe_tokenize_text(vocab, tmp, out);
    free(tmp);
}




static void tokenize_rendered_chat_vocab(const ds4_vocab *vocab, const char *text,
                                         token_vec *out) {
    if (!text) text = "";

    const char *span = text;
    const char *p = text;
    while (*p) {
        int token = -1;
        size_t len = 0;
        if (special_token_at(vocab, p, &token, &len)) {
            tokenize_span(vocab, span, (size_t)(p - span), out);
            token_vec_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    tokenize_span(vocab, span, (size_t)(p - span), out);
}

void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out) {
    tokenize_rendered_chat_vocab(&e->vocab, text, out);
}

void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens) {
    chat_push_bos_sequence(&e->vocab, tokens);
}

void ds4_encode_chat_prompt(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out) {
    encode_chat_prompt(&e->vocab, system, prompt ? prompt : "", think_mode, out);
}

void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens) {
    bpe_tokenize_text(&e->vocab, DS4_REASONING_EFFORT_MAX_PREFIX, tokens);
}

static void bpe_tokenize_wrapped_payload_text(ds4_vocab *vocab, const char *content,
                                              const char *end, token_vec *out) {
    /* Tool output is plain data inside the model-family wrapper.
     * Preserve literal '<', '>' and '&' so shell output and file snippets stay
     * intact, but escape the exact closing sentinel so a malicious or accidental
     * tool payload cannot terminate the wrapper early. */
    const size_t endlen = strlen(end);
    const char *span = content ? content : "";
    const char *p = span;
    while (*p) {
        if (!strncmp(p, end, endlen)) {
            tokenize_span(vocab, span, (size_t)(p - span), out);
            bpe_tokenize_text(vocab, "&lt;", out);
            p++;
            span = p;
        } else {
            p++;
        }
    }
    tokenize_span(vocab, span, (size_t)(p - span), out);
}

static void bpe_tokenize_tool_result_text(ds4_vocab *vocab, const char *content, token_vec *out) {
    bpe_tokenize_wrapped_payload_text(vocab, content, "</tool_result>", out);
}

static void bpe_tokenize_tool_response_text(ds4_vocab *vocab, const char *content, token_vec *out) {
    bpe_tokenize_wrapped_payload_text(vocab, content, "</tool_response>", out);
}

void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content) {
    ds4_vocab *vocab = &e->vocab;
    if (!role) role = "user";
    if (!content) content = "";

    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        if (!strcmp(role, "system") || !strcmp(role, "developer")) {
            if (vocab->system_id >= 0) token_vec_push(tokens, vocab->system_id);
            tokenize_rendered_chat_vocab(vocab, content, tokens);
        } else if (!strcmp(role, "assistant")) {
            token_vec_push(tokens, vocab->assistant_id);
            if (strncmp(content, "<think>", 7) != 0 &&
                strncmp(content, "</think>", 8) != 0) {
                token_vec_push(tokens, vocab->think_start_id);
                token_vec_push(tokens, vocab->think_end_id);
            }
            tokenize_rendered_chat_vocab(vocab, content, tokens);
        } else if (!strcmp(role, "tool") || !strcmp(role, "function")) {
            if (vocab->observation_id >= 0) token_vec_push(tokens, vocab->observation_id);
            tokenize_rendered_chat_vocab(vocab, "<tool_response>", tokens);
            bpe_tokenize_tool_response_text(vocab, content, tokens);
            tokenize_rendered_chat_vocab(vocab, "</tool_response>", tokens);
        } else {
            token_vec_push(tokens, vocab->user_id);
            bpe_tokenize_text(vocab, content, tokens);
        }
        return;
    }

    if (!strcmp(role, "system") || !strcmp(role, "developer")) {
        bpe_tokenize_text(vocab, content, tokens);
    } else if (!strcmp(role, "assistant")) {
        token_vec_push(tokens, vocab->assistant_id);
        if (strncmp(content, "<think>", 7) != 0 && strncmp(content, "</think>", 8) != 0) {
            token_vec_push(tokens, vocab->think_end_id);
        }
        bpe_tokenize_text(vocab, content, tokens);
    } else if (!strcmp(role, "tool") || !strcmp(role, "function")) {
        token_vec_push(tokens, vocab->user_id);
        bpe_tokenize_text(vocab, "<tool_result>", tokens);
        bpe_tokenize_tool_result_text(vocab, content, tokens);
        bpe_tokenize_text(vocab, "</tool_result>", tokens);
    } else {
        token_vec_push(tokens, vocab->user_id);
        bpe_tokenize_text(vocab, content, tokens);
    }
}


void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode) {
    token_vec_push(tokens, e->vocab.assistant_id);
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA &&
        !ds4_think_mode_enabled(think_mode)) {
        token_vec_push(tokens, e->vocab.think_start_id);
        token_vec_push(tokens, e->vocab.think_end_id);
        return;
    }
    token_vec_push(tokens, ds4_think_mode_enabled(think_mode) ?
                   e->vocab.think_start_id : e->vocab.think_end_id);
}

static void dump_tokens_fp(FILE *fp, const ds4_vocab *vocab, const token_vec *tokens) {
    fprintf(fp, "[");
    for (int i = 0; i < tokens->len; i++) {
        if (i) fprintf(fp, ", ");
        fprintf(fp, "%d", tokens->v[i]);
    }
    fprintf(fp, "]\n");

    for (int i = 0; i < tokens->len; i++) {
        int id = tokens->v[i];
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%6d  %.*s\n", id, (int)vocab->token[id].len, vocab->token[id].ptr);
        }
    }
}

static void dump_tokens(const ds4_vocab *vocab, const token_vec *tokens) {
    dump_tokens_fp(stdout, vocab, tokens);
}

static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    const uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}

static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }

    uint32_t n = 0;
    for (uint32_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) {
            continue;
        }
        if (cp == 256 + n) return (int)b;
        n++;
    }
    return -1;
}

static bool vocab_token_is_literal_special(ds4_str s) {
    const unsigned char bar[] = {0xef, 0xbd, 0x9c}; /* U+FF5C fullwidth vertical bar. */
    if (s.len < sizeof(bar)) return false;
    for (uint64_t i = 0; i + sizeof(bar) <= s.len; i++) {
        if (!memcmp(s.ptr + i, bar, sizeof(bar))) return true;
    }
    return false;
}

char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    ds4_vocab *vocab = &e->vocab;
    if (token < 0 || token >= vocab->n_vocab) {
        if (len) *len = 0;
        char *out = xmalloc(1);
        out[0] = '\0';
        return out;
    }

    ds4_str s = vocab->token[token];
    char *out = xmalloc((size_t)s.len + 1);
    if (vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        out[s.len] = '\0';
        if (len) *len = (size_t)s.len;
        return out;
    }

    size_t n = 0;
    uint64_t pos = 0;
    while (pos < s.len) {
        uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

static bool vocab_token_is_generation_stop(const ds4_vocab *vocab, int token) {
    if (!vocab || token < 0) return false;
    if (token == vocab->eos_id) return true;
    if (DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_GLM_DSA) {
        return (vocab->system_id >= 0 && token == vocab->system_id) ||
               (vocab->user_id >= 0 && token == vocab->user_id) ||
               (vocab->assistant_id >= 0 && token == vocab->assistant_id) ||
               (vocab->observation_id >= 0 && token == vocab->observation_id);
    }
    return false;
}

int ds4_token_eos(ds4_engine *e) {
    return e->vocab.eos_id;
}

bool ds4_token_is_stop(ds4_engine *e, int token) {
    return e ? vocab_token_is_generation_stop(&e->vocab, token) : false;
}

bool ds4_token_is_thinking_control(ds4_engine *e, int token) {
    if (!e || token < 0) return false;
    return (e->vocab.think_start_id >= 0 &&
            token == e->vocab.think_start_id) ||
           (e->vocab.think_end_id >= 0 &&
            token == e->vocab.think_end_id);
}

bool ds4_token_is_stop_for_think_mode(
        ds4_engine      *e,
        int              token,
        ds4_think_mode   mode) {
    if (ds4_token_is_stop(e, token)) return true;
    /* In no-thinking mode the prompt already supplied the protocol close tag.
     * If the model emits another thinking tag, do not print or feed it back:
     * it is a control marker, not assistant content. */
    if (!ds4_think_mode_enabled(mode) &&
        ds4_token_is_thinking_control(e, token)) {
        return true;
    }
    return false;
}

int ds4_token_user(ds4_engine *e) {
    return e->vocab.user_id;
}

int ds4_token_assistant(ds4_engine *e) {
    return e->vocab.assistant_id;
}

static inline void argmax_f32_unrolled8_range(
        const float *logits,
        uint32_t     begin,
        uint32_t     end,
        int         *best,
        float       *best_v) {
    uint32_t i = begin;
    int b0 = *best, b1 = *best, b2 = *best, b3 = *best;
    int b4 = *best, b5 = *best, b6 = *best, b7 = *best;
    float v0 = *best_v, v1 = *best_v, v2 = *best_v, v3 = *best_v;
    float v4 = *best_v, v5 = *best_v, v6 = *best_v, v7 = *best_v;

    while (end - i >= 8u) {
        const float x0 = logits[i + 0u];
        const float x1 = logits[i + 1u];
        const float x2 = logits[i + 2u];
        const float x3 = logits[i + 3u];
        const float x4 = logits[i + 4u];
        const float x5 = logits[i + 5u];
        const float x6 = logits[i + 6u];
        const float x7 = logits[i + 7u];
        if (x0 > v0) { v0 = x0; b0 = (int)(i + 0u); }
        if (x1 > v1) { v1 = x1; b1 = (int)(i + 1u); }
        if (x2 > v2) { v2 = x2; b2 = (int)(i + 2u); }
        if (x3 > v3) { v3 = x3; b3 = (int)(i + 3u); }
        if (x4 > v4) { v4 = x4; b4 = (int)(i + 4u); }
        if (x5 > v5) { v5 = x5; b5 = (int)(i + 5u); }
        if (x6 > v6) { v6 = x6; b6 = (int)(i + 6u); }
        if (x7 > v7) { v7 = x7; b7 = (int)(i + 7u); }
        i += 8u;
    }

#define DS4_ARGMAX_MERGE_LANE(b, v) \
    do { \
        if ((v) > *best_v || ((v) == *best_v && (b) < *best)) { \
            *best_v = (v); \
            *best = (b); \
        } \
    } while (0)
    DS4_ARGMAX_MERGE_LANE(b0, v0);
    DS4_ARGMAX_MERGE_LANE(b1, v1);
    DS4_ARGMAX_MERGE_LANE(b2, v2);
    DS4_ARGMAX_MERGE_LANE(b3, v3);
    DS4_ARGMAX_MERGE_LANE(b4, v4);
    DS4_ARGMAX_MERGE_LANE(b5, v5);
    DS4_ARGMAX_MERGE_LANE(b6, v6);
    DS4_ARGMAX_MERGE_LANE(b7, v7);
#undef DS4_ARGMAX_MERGE_LANE

    for (; i < end; i++) {
        const float v = logits[i];
        if (v > *best_v) {
            *best_v = v;
            *best = (int)i;
        }
    }
}

static int sample_argmax_unrolled8(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = DS4_NEG_INF;
    argmax_f32_unrolled8_range(logits, 0, n_vocab, &best, &best_v);
    return best;
}

static int argmax_f32_excluding_unrolled8(
        const float *logits,
        uint32_t     n,
        int          excluded_id) {
    const uint32_t first = excluded_id == 0 ? 1u : 0u;
    if (first >= n) return -1;

    int best = (int)first;
    float best_v = logits[first];
    const uint32_t begin = first + 1u;
    if (excluded_id >= 0 &&
        (uint32_t)excluded_id >= begin &&
        (uint32_t)excluded_id < n) {
        const uint32_t excluded = (uint32_t)excluded_id;
        argmax_f32_unrolled8_range(logits, begin, excluded, &best, &best_v);
        argmax_f32_unrolled8_range(logits, excluded + 1u, n, &best, &best_v);
    } else {
        argmax_f32_unrolled8_range(logits, begin, n, &best, &best_v);
    }
    return best;
}

static int sample_argmax(const float *logits, uint32_t n_vocab) {
    if (getenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == NULL) {
        return sample_argmax_unrolled8(logits, n_vocab);
    }
    int best = 0;
    float best_v = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}

static DS4_MAYBE_UNUSED void logits_top2(const float *logits, uint32_t n_vocab,
                        int *top0, float *logit0,
                        int *top1, float *logit1) {
    int b0 = -1, b1 = -1;
    float v0 = DS4_NEG_INF, v1 = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > v0) {
            b1 = b0; v1 = v0;
            b0 = (int)i; v0 = v;
        } else if (v > v1) {
            b1 = (int)i; v1 = v;
        }
    }
    if (top0) *top0 = b0;
    if (logit0) *logit0 = v0;
    if (top1) *top1 = b1;
    if (logit1) *logit1 = v1;
}

static uint64_t sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dULL;
}

static float sample_rng_f32(uint64_t *state) {
    const uint64_t x = sample_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

typedef struct {
    int id;
    float logit;
    float prob;
} sample_candidate;

static int sample_candidate_cmp_desc(const void *a, const void *b) {
    const sample_candidate *ca = a;
    const sample_candidate *cb = b;
    const int logit_order =
        (cb->logit > ca->logit) - (cb->logit < ca->logit);
    if (logit_order != 0) return logit_order;
    return (ca->id > cb->id) - (ca->id < cb->id);
}

static bool sample_candidate_gt(sample_candidate a, sample_candidate b) {
    if (a.logit != b.logit) return a.logit > b.logit;
    return a.id < b.id;
}

static void sample_heap_sift_up(sample_candidate *heap, uint32_t idx) {
    while (idx > 0) {
        const uint32_t parent = (idx - 1u) / 2u;
        if (!sample_candidate_gt(heap[parent], heap[idx])) break;
        sample_candidate tmp = heap[parent];
        heap[parent] = heap[idx];
        heap[idx] = tmp;
        idx = parent;
    }
}

static void sample_heap_sift_down(sample_candidate *heap, uint32_t n, uint32_t idx) {
    for (;;) {
        const uint32_t left = idx * 2u + 1u;
        const uint32_t right = left + 1u;
        uint32_t smallest = idx;
        if (left < n && sample_candidate_gt(heap[smallest], heap[left])) {
            smallest = left;
        }
        if (right < n && sample_candidate_gt(heap[smallest], heap[right])) {
            smallest = right;
        }
        if (smallest == idx) break;
        sample_candidate tmp = heap[idx];
        heap[idx] = heap[smallest];
        heap[smallest] = tmp;
        idx = smallest;
    }
}

static bool sample_fast_top_p(
        const float *logits,
        uint32_t     n_vocab,
        uint32_t     finite,
        float        max_logit,
        int          best,
        float        temperature,
        float        top_p,
        float        min_p,
        uint64_t    *rng,
        int         *token_out) {
    enum { SAMPLE_FAST_TOP_P_CAP = 512 };
    if (!logits || !rng || !token_out || finite == 0) return false;
    if (finite > SAMPLE_FAST_TOP_P_CAP && top_p >= 0.999f) return false;

    const uint32_t cap = finite < SAMPLE_FAST_TOP_P_CAP ?
        finite : (uint32_t)SAMPLE_FAST_TOP_P_CAP;
    sample_candidate heap[SAMPLE_FAST_TOP_P_CAP];
    uint32_t n = 0;
    float sum = 0.0f;
    float heap_sum = 0.0f;

    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float p = expf((v - max_logit) / temperature);
        sum += p;
        sample_candidate cand = {.id = (int)i, .logit = v, .prob = p};
        if (n < cap) {
            heap[n] = cand;
            heap_sum += p;
            sample_heap_sift_up(heap, n);
            n++;
        } else if (sample_candidate_gt(cand, heap[0])) {
            heap_sum -= heap[0].prob;
            heap[0] = cand;
            heap_sum += p;
            sample_heap_sift_down(heap, n, 0);
        }
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        *token_out = best;
        return true;
    }

    if (n < finite && heap_sum < top_p * sum) {
        return false;
    }

    qsort(heap, n, sizeof(heap[0]), sample_candidate_cmp_desc);
    const float min_prob = (heap[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    const float min_prob_raw = heap[0].prob * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    bool stopped_by_min_p = false;
    for (uint32_t i = 0; i < n; i++) {
        const float p = heap[i].prob / sum;
        if (i > 0 && p < min_prob) {
            stopped_by_min_p = true;
            break;
        }
        filtered_sum += heap[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (n < finite &&
        stopped_by_min_p &&
        min_p > 0.0f &&
        heap[n - 1u].prob >= min_prob_raw) {
        return false;
    }
    if (filtered == 0) {
        *token_out = best;
        return true;
    }

    float r = sample_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= heap[i].prob;
        if (r <= 0.0f) {
            *token_out = heap[i].id;
            return true;
        }
    }
    *token_out = heap[filtered - 1u].id;
    return true;
}

static int sample_full_vocab(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        float        top_p,
        float        min_p,
        uint64_t    *rng,
        float       *prob_scratch) {
    float max_logit = DS4_NEG_INF;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return sample_argmax(logits, n_vocab);

    int fast_token = best;
    if (top_p < 1.0f &&
        sample_fast_top_p(logits,
                          n_vocab,
                          finite,
                          max_logit,
                          best,
                          temperature,
                          top_p,
                          min_p,
                          rng,
                          &fast_token)) {
        return fast_token;
    }

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        if (min_rel > 1.0f) return best;

        /* Find a conservative log-space rejection boundary using the same
         * expf implementation as the probability path. Values below this
         * boundary are guaranteed to fail min-p, avoiding an expf for the
         * overwhelming majority of a large vocabulary. Near-boundary values
         * still take the ordinary expf comparison. */
        float reject_scaled = DS4_NEG_INF;
        bool have_reject_scaled = false;
        if (min_rel > 0.0f && isfinite(min_rel)) {
            float cutoff = logf(min_rel);
            for (int i = 0; i < 8 && isfinite(cutoff); i++) {
                cutoff = nextafterf(cutoff, -FLT_MAX);
                if (expf(cutoff) < min_rel) {
                    reject_scaled = cutoff;
                    have_reject_scaled = true;
                    break;
                }
            }
        }

        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            prob_scratch[i] = -1.0f;
            if (!isfinite(v)) continue;
            const float scaled = (v - max_logit) / temperature;
            if (have_reject_scaled && scaled <= reject_scaled) continue;
            const float p = expf(scaled);
            if (p < min_rel) continue;
            prob_scratch[i] = p;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = sample_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float p = prob_scratch[i];
            if (p < 0.0f) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    uint32_t n = 0;
    float sum = 0.0f;
    sample_candidate *cand = NULL;
    if (min_p > 0.0f && min_p <= 1.0f) {
        /* The later min-p comparison is equivalent to
         * exp((logit-max)/temperature) >= min_p; its normalization cancels.
         * Still compute the full softmax sum in the original order, then sort
         * only candidates that can survive. This preserves the nucleus mass
         * and RNG semantics while avoiding a full-vocabulary qsort. */
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            prob_scratch[i] = -1.0f;
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            prob_scratch[i] = p;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;

        const float min_prob = (1.0f / sum) * min_p;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float p = prob_scratch[i];
            if (p < 0.0f || p / sum < min_prob) continue;
            n++;
        }
        if (n == 0) return best;
        cand = xmalloc((size_t)n * sizeof(cand[0]));
        uint32_t out = 0;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float p = prob_scratch[i];
            if (p < 0.0f || p / sum < min_prob) continue;
            cand[out++] = (sample_candidate){
                .id = (int)i, .logit = logits[i], .prob = p
            };
        }
    } else {
        cand = xmalloc((size_t)finite * sizeof(cand[0]));
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            cand[n++] = (sample_candidate){.id = (int)i, .logit = v, .prob = p};
            sum += p;
        }
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        free(cand);
        return best;
    }

    qsort(cand, n, sizeof(cand[0]), sample_candidate_cmp_desc);
    const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = cand[i].prob / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) {
        free(cand);
        return best;
    }

    float r = sample_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= cand[i].prob;
        if (r <= 0.0f) {
            const int id = cand[i].id;
            free(cand);
            return id;
        }
    }
    const int id = cand[filtered - 1].id;
    free(cand);
    return id;
}

static int sample_top_p_min_p(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t    *rng,
        float       *prob_scratch) {
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) {
        const bool owned_scratch = prob_scratch == NULL;
        if (owned_scratch) {
            prob_scratch = xmalloc((size_t)n_vocab * sizeof(prob_scratch[0]));
        }
        const int token = sample_full_vocab(logits, n_vocab, temperature,
                                            top_p, min_p, rng, prob_scratch);
        if (owned_scratch) free(prob_scratch);
        return token;
    }
    if (top_k > 1024) top_k = 1024;
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    int ids[1024];
    float vals[1024];
    int n = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == top_k && v <= vals[n - 1]) continue;
        int j = n < top_k ? n++ : n - 1;
        while (j > 0 && vals[j - 1] < v) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        vals[j] = v;
        ids[j] = (int)i;
    }
    if (n == 0) return sample_argmax(logits, n_vocab);

    float probs[1024];
    const float max_logit = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((vals[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) return ids[0];

    const float min_prob = (probs[0] / sum) * min_p;
    float filtered_sum = 0.0f;
    int filtered = 0;
    for (int i = 0; i < n; i++) {
        float p = probs[i] / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += probs[i];
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered <= 0) return ids[0];

    float r = sample_rng_f32(rng) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        r -= probs[i];
        if (r <= 0.0f) return ids[i];
    }
    return ids[filtered - 1];
}

#ifdef DS4_TEST_HOOKS
int ds4_test_sample_logits(const float *logits, uint32_t n_vocab,
                           float temperature, int top_k,
                           float top_p, float min_p, uint64_t *rng,
                           float *prob_scratch) {
    if (!logits || !rng || n_vocab == 0) return -1;
    return sample_top_p_min_p(logits, n_vocab, temperature, top_k,
                              top_p, min_p, rng, prob_scratch);
}

int ds4_test_argmax_excluding_logits(const float *logits, uint32_t n_vocab,
                                     int excluded_id) {
    if (!logits) return -1;
    if (getenv("DS4_CPU_DISABLE_UNROLLED_ARGMAX") == NULL) {
        return argmax_f32_excluding_unrolled8(logits, n_vocab, excluded_id);
    }
    int best = -1;
    float best_v = DS4_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        if ((int)i == excluded_id) continue;
        const float v = logits[i];
        if (best < 0 || v > best_v) {
            best = (int)i;
            best_v = v;
        }
    }
    return best;
}
#endif

static void print_top_logits(
        FILE          * fp,
        const char    * label,
        const ds4_vocab * vocab,
        const float   * logits,
        uint32_t        n_vocab,
        int             k) {
    int best[16];
    if (k > 16) k = 16;
    for (int i = 0; i < k; i++) best[i] = -1;

    for (uint32_t i = 0; i < n_vocab; i++) {
        for (int j = 0; j < k; j++) {
            if (best[j] < 0 || logits[i] > logits[best[j]]) {
                for (int l = k - 1; l > j; l--) best[l] = best[l - 1];
                best[j] = (int)i;
                break;
            }
        }
    }

    fprintf(fp, "ds4: top logits %s:\n", label);
    for (int i = 0; i < k && best[i] >= 0; i++) {
        const int id = best[i];
        fprintf(fp, "  %2d %7d % .9g  ", i, id, logits[id]);
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%.*s", (int)vocab->token[id].len, vocab->token[id].ptr);
        }
        fputc('\n', fp);
    }
}

/* CPU generation entry point.  It runs layer-major prefill once, then decodes
 * one token at a time using the persistent KV cache and scratch arena. */
static int generate_raw_swa_cpu(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        const float       * directional_steering_dirs,
        float               directional_steering_attn,
        float               directional_steering_ffn,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud,
        ds4_session_progress_fn progress,
        void              * progress_ud) {
    (void)progress;
    (void)progress_ud;
    fprintf(stderr, "ds4: using CPU generation with layer-major prefill\n");

    ds4_kv_cache cache;
    kv_cache_init(&cache, (uint32_t)ctx_size, 0);
    ds4_cpu_decode_scratch decode_scratch;
    cpu_decode_scratch_init(&decode_scratch, (uint32_t)ctx_size);

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    int pos = prompt->len;
    const bool trace_top = getenv("DS4_TRACE_TOP") != NULL;
    const double t_prefill0 = now_sec();

    if (prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        free(logits);
        cpu_decode_scratch_free(&decode_scratch);
        kv_cache_free(&cache);
        return 1;
    }

    prefill_layer_major_cpu(logits, model, weights, &cache, prompt,
                            directional_steering_dirs,
                            directional_steering_attn,
                            directional_steering_ffn);

    const double t_prefill1 = now_sec();
    fprintf(stderr, "ds4: prefill %d/%d done\n", prompt->len, prompt->len);
    const char *dump_prefill_logits = getenv("DS4_CPU_DUMP_PREFILL_LOGITS");
    if (dump_prefill_logits && dump_prefill_logits[0]) {
        if (!write_f32_binary_file(dump_prefill_logits, logits, DS4_N_VOCAB)) {
            free(logits);
            cpu_decode_scratch_free(&decode_scratch);
            kv_cache_free(&cache);
            return 1;
        }
        fprintf(stderr, "ds4: wrote CPU prefill logits to %s\n", dump_prefill_logits);
    }

    int n_generated = 0;
    int n_decode_eval = 0;
    const bool token_timing = getenv("DS4_TOKEN_TIMING") != NULL;
    const double t_decode0 = now_sec();
    for (int i = 0; i < n_predict && pos < ctx_size; i++) {
        if (trace_top) {
            char label[64];
            snprintf(label, sizeof(label), "step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }

        int token = sample_argmax(logits, DS4_N_VOCAB);
        if (vocab_token_is_generation_stop(vocab, token)) break;

        if (emit) emit(emit_ud, token);
        n_generated++;

        if (i == n_predict - 1 || pos + 1 >= ctx_size) {
            pos++;
            break;
        }

        const double t_eval0 = token_timing ? now_sec() : 0.0;
        /* The CPU decode step is expected to reuse buffers from
         * cpu_decode_scratch.  Keep the allocation guard tightly scoped to the
         * decode math itself; sampling, token emission, tracing, and callbacks
         * may allocate small temporary strings without invalidating that
         * guarantee. */
        ds4_alloc_guard_begin("CPU token decode");
        forward_token_raw_swa_cpu_decode_scratch(logits, model, weights, &cache, token, (uint32_t)pos,
                                                 directional_steering_dirs,
                                                 directional_steering_attn,
                                                 directional_steering_ffn,
                                                 &decode_scratch);
        ds4_alloc_guard_end();
        if (token_timing) {
            const double t_eval1 = now_sec();
            fprintf(stderr, "ds4: decode eval %d took %.3f ms\n", n_decode_eval + 1, (t_eval1 - t_eval0) * 1000.0);
        }
        n_decode_eval++;
        pos++;
    }
    const double t_decode1 = now_sec();
    if (done) done(emit_ud);

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)n_generated / decode_s : 0.0);

    free(logits);
    cpu_decode_scratch_free(&decode_scratch);
    kv_cache_free(&cache);
    return 0;
}

