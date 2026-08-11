static void plain_set_thinking_color(bool use_color) {
    if (use_color) fputs(ANSI_DIM, stdout);
}

static void plain_reset_color(bool use_color) {
    if (use_color) fputs(ANSI_RESET, stdout);
}

static const char *eval_system_prompt(void) {
    return "You are solving a hard benchmark question. Reason carefully. "
           "The final answer must follow the requested format exactly.";
}

static char *build_question_prompt(const eval_case *tc) {
    byte_buf b = {0};
    int nchoices = eval_case_nchoices(tc);
    buf_appendf(&b, "%s\n", tc->question);
    if (nchoices > 0) {
        buf_append(&b, "\nChoices:\n", strlen("\nChoices:\n"));
        for (int i = 0; i < nchoices; i++) {
            buf_appendf(&b, "%c. %s\n", 'A' + i, tc->choice[i]);
        }
        buf_append(&b,
            "\nSolve the question. At the end, write exactly one final line in this "
            "format and do not write anything after it:\n"
            "Answer: <letter>",
            strlen("\nSolve the question. At the end, write exactly one final line in this "
                   "format and do not write anything after it:\n"
                   "Answer: <letter>"));
    } else if (eval_case_is_compsec(tc)) {
        buf_append(&b,
            "\nAt the end, write exactly one final line in this format and do not "
            "write anything after it:\n"
            "Answer: <line number or comma-separated line numbers>",
            strlen("\nAt the end, write exactly one final line in this format and do not "
                   "write anything after it:\n"
                   "Answer: <line number or comma-separated line numbers>"));
    } else {
        buf_append(&b,
            "\nSolve the problem. At the end, write exactly one final line in this "
            "format and do not write anything after it:\n"
            "Answer: <integer>",
            strlen("\nSolve the problem. At the end, write exactly one final line in this "
                   "format and do not write anything after it:\n"
                   "Answer: <integer>"));
    }
    if (b.v) return b.v;
    char *empty = malloc(1);
    if (empty) empty[0] = '\0';
    return empty;
}

static int eval_max_prompt_tokens(ds4_engine *engine,
                                  const eval_config *cfg,
                                  const eval_case *cases,
                                  int ncases,
                                  int ctx_for_think_mode,
                                  int *max_case_out)
{
    int max_prompt = 0;
    int max_case = -1;
    const ds4_think_mode think_mode =
        ds4_think_mode_for_context(cfg->think_mode, ctx_for_think_mode);

    for (int i = 0; i < ncases; i++) {
        char *question = build_question_prompt(&cases[i]);
        if (!question) {
            fprintf(stderr, "ds4-eval: failed to allocate prompt\n");
            exit(1);
        }
        ds4_tokens prompt = {0};
        ds4_encode_chat_prompt(engine, eval_system_prompt(), question, think_mode, &prompt);
        if (prompt.len > max_prompt) {
            max_prompt = prompt.len;
            max_case = i;
        }
        ds4_tokens_free(&prompt);
        free(question);
    }
    if (max_case_out) *max_case_out = max_case;
    return max_prompt;
}

static int eval_auto_context_size(ds4_engine *engine,
                                  eval_config *cfg,
                                  const eval_case *cases,
                                  int ncases,
                                  int *max_prompt_out,
                                  int *max_case_out)
{
    int ctx = EVAL_MAX_CONTEXT;
    int max_prompt = 0;
    int max_case = -1;
    const int min_ctx = cfg->think_mode == DS4_THINK_MAX ?
                        (int)ds4_think_max_min_context() : 1;

    /* Think Max downgrades to normal thinking under its minimum context.  Size
     * the prompts iteratively so the prompt tokenizer sees the same effective
     * thinking mode that the actual run will use. */
    for (int iter = 0; iter < 3; iter++) {
        max_prompt = eval_max_prompt_tokens(engine, cfg, cases, ncases, ctx, &max_case);
        long long required = (long long)max_prompt + (long long)cfg->max_tokens;
        if (required < min_ctx) required = min_ctx;
        if (required > EVAL_MAX_CONTEXT) {
            fprintf(stderr,
                    "ds4-eval: largest prompt (%d tokens, case %d) + --tokens (%d) exceeds the %d token context cap\n",
                    max_prompt, max_case + 1, cfg->max_tokens, EVAL_MAX_CONTEXT);
            exit(2);
        }
        if ((int)required == ctx) break;
        ctx = (int)required;
    }

    if (max_prompt_out) *max_prompt_out = max_prompt;
    if (max_case_out) *max_case_out = max_case;
    return ctx;
}

static void eval_warn_think_max_downgraded(const eval_config *cfg) {
    if (cfg->think_mode != DS4_THINK_MAX ||
        ds4_think_mode_for_context(cfg->think_mode, cfg->ctx_size) == DS4_THINK_MAX) {
        return;
    }
    fprintf(stderr,
            "ds4-eval: warning: --think-max needs --ctx >= %u; ctx=%d uses normal thinking instead\n",
            ds4_think_max_min_context(),
            cfg->ctx_size);
}

static void eval_warn_context_budget(const eval_config *cfg, int max_prompt_tokens, int max_prompt_case) {
    if (max_prompt_tokens >= cfg->ctx_size) {
        fprintf(stderr,
                "ds4-eval: warning: largest prompt (%d tokens, case=%d) does not fit ctx=%d\n",
                max_prompt_tokens,
                max_prompt_case + 1,
                cfg->ctx_size);
        return;
    }

    const int room = cfg->ctx_size - max_prompt_tokens;
    if (room < cfg->max_tokens) {
        fprintf(stderr,
                "ds4-eval: warning: largest prompt (%d tokens, case=%d) leaves %d generation tokens in ctx=%d; requested %d\n",
                max_prompt_tokens,
                max_prompt_case + 1,
                room,
                cfg->ctx_size,
                cfg->max_tokens);
    }
}

static void trace_write_block(FILE *trace, const char *label, const char *text) {
    if (!trace) return;
    size_t len = text ? strlen(text) : 0;
    /* Counted blocks make regrading robust against model output that happens
     * to contain trace-looking delimiters or embedded CASE headers. */
    fprintf(trace, "%s_BEGIN bytes=%zu\n", label, len);
    if (len) {
        fwrite(text, 1, len, trace);
        if (text[len - 1] != '\n') fputc('\n', trace);
    }
    fprintf(trace, "%s_END\n", label);
}

static const char *think_close_kind_name(eval_think_close_kind kind) {
    switch (kind) {
    case EVAL_THINK_CLOSE_NATURAL: return "natural";
    case EVAL_THINK_CLOSE_SOFT: return "soft_forced";
    case EVAL_THINK_CLOSE_HARD: return "hard_forced";
    case EVAL_THINK_CLOSE_NONE:
    default: return "none";
    }
}

static int token_rank_in_top(ds4_session *session, int token, int max_rank) {
    if (token < 0 || max_rank <= 0) return 0;
    ds4_token_score *top = malloc((size_t)max_rank * sizeof(*top));
    if (!top) return 0;
    int n = ds4_session_top_logprobs(session, top, max_rank);
    int rank = 0;
    for (int i = 0; i < n; i++) {
        if (top[i].id == token) {
            rank = i + 1;
            break;
        }
    }
    free(top);
    return rank;
}

static void trace_write_header(FILE *trace, const eval_config *cfg,
                               const char *model_name,
                               int ncases,
                               int max_prompt_tokens) {
    if (!trace) return;
    fprintf(trace,
            "# ds4-eval trace\n"
            "started_unix: %lld\n"
            "model: %s\n"
            "model_shape: %s\n"
            "backend: %s\n"
            "ctx: %d\n"
            "max_tokens: %d\n"
            "max_prompt_tokens: %d\n"
            "questions: %d\n"
            "temperature: %.6g\n"
            "top_p: %.6g\n"
            "min_p: %.6g\n"
            "seed: %llu\n"
            "think_mode_requested: %s\n"
            "soft_limit_reply_budget: %d\n"
            "hard_limit_reply_budget: %d\n"
            "soft_limit_think_close_rank: %d\n"
            "\n",
            (long long)time(NULL),
            cfg->model_path,
            model_name ? model_name : "unknown",
            ds4_backend_name(cfg->backend),
            cfg->ctx_size,
            cfg->max_tokens,
            max_prompt_tokens,
            ncases,
            cfg->temperature,
            cfg->top_p,
            cfg->min_p,
            (unsigned long long)cfg->seed,
            ds4_think_mode_name(cfg->think_mode),
            cfg->soft_limit_reply_budget,
            cfg->hard_limit_reply_budget,
            cfg->soft_limit_think_close_rank);
    fflush(trace);
}

static void trace_write_case(FILE *trace,
                             const eval_config *cfg,
                             const eval_case *tc,
                             int idx,
                             int ncases,
                             const char *status,
                             const char *error,
                             const char *system_prompt,
                             const char *question_prompt,
                             const char *model_output,
                             ds4_think_mode effective_think_mode,
                             int prompt_tokens,
                             int generated_tokens,
                             double elapsed_sec,
                             const char *picked,
                             const eval_think_close_info *think_close) {
    if (!trace) return;
    int nchoices = eval_case_nchoices(tc);
    fprintf(trace,
            "===== CASE %d/%d %s/%s =====\n"
            "timestamp_unix: %lld\n"
            "source: %s\n"
            "id: %s\n"
            "domain: %s\n"
            "title: %s\n"
            "status: %s\n"
            "picked: %s\n"
            "expected: %s\n"
            "prompt_tokens: %d\n"
            "generated_tokens: %d\n"
            "elapsed_sec: %.3f\n"
            "temperature: %.6g\n"
            "top_p: %.6g\n"
            "min_p: %.6g\n"
            "think_mode_effective: %s\n",
            idx + 1, ncases, tc->source, tc->id,
            (long long)time(NULL),
            tc->source,
            tc->id,
            tc->domain,
            tc->title,
            status,
            picked && *picked ? picked : "?",
            tc->answer,
            prompt_tokens,
            generated_tokens,
            elapsed_sec,
            cfg->temperature,
            cfg->top_p,
            cfg->min_p,
            ds4_think_mode_name(effective_think_mode));
    if (think_close && think_close->kind != EVAL_THINK_CLOSE_NONE) {
        fprintf(trace,
                "think_close: %s\n"
                "think_close_token_index: %d\n"
                "think_close_remaining_budget: %d\n"
                "think_close_rank: %d\n",
                think_close_kind_name(think_close->kind),
                think_close->token_index,
                think_close->remaining_budget,
                think_close->rank);
    } else {
        fprintf(trace, "think_close: none\n");
    }
    if (error && *error) fprintf(trace, "error: %s\n", error);

    if (nchoices > 0) {
        fprintf(trace, "choices:\n");
        for (int i = 0; i < nchoices; i++) {
            fprintf(trace, "  %c. %s\n", 'A' + i, tc->choice[i]);
        }
    } else {
        fprintf(trace, "answer_kind: exact_integer\n");
    }
    trace_write_block(trace, "SYSTEM_PROMPT", system_prompt);
    trace_write_block(trace, "QUESTION_PROMPT", question_prompt);
    trace_write_block(trace, "MODEL_OUTPUT", model_output);
    fputc('\n', trace);
    fflush(trace);
}

/* Model outputs can contain provisional "answer" text after a forced
 * </think> and then a later final line.  A strict final Answer: marker is the
 * best grading target; outputs without one keep the original loose fallback. */
static char *strcasestr_local(const char *hay, const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0) return (char *)hay;
    for (; *hay; hay++) {
        if (tolower((unsigned char)*hay) == tolower((unsigned char)needle[0]) &&
            strncasecmp(hay, needle, nlen) == 0)
            return (char *)hay;
    }
    return NULL;
}

static bool is_letter_boundary(char before, char after) {
    return !isalpha((unsigned char)before) && !isalpha((unsigned char)after);
}

static char *find_last_answer_marker(const char *visible) {
    char *last = NULL;
    const size_t nlen = strlen("answer");

    for (const char *p = visible; *p; p++) {
        if (tolower((unsigned char)*p) != 'a' || strncasecmp(p, "answer", nlen) != 0)
            continue;
        char before = p == visible ? ' ' : p[-1];
        char after = p[nlen];
        if (!is_letter_boundary(before, after)) continue;

        const char *q = p + nlen;
        while (*q && isspace((unsigned char)*q)) q++;
        if (*q == ':') last = (char *)p;
    }
    return last ? last : strcasestr_local(visible, "answer");
}

/* True when the in-range capital at `letter` is the object of an explicit
 * rejection earlier on the same answer line -- "not B", "isn't B",
 * "rules out C", "eliminate E" -- so it is a distractor the model is
 * discarding, not its pick.  This rewrites only the lexical "first valid
 * letter wins" default for clear elimination phrasing; the cue set is kept
 * small and high-precision on purpose, and there is no general
 * selection-vs-rejection sentence parsing (issue #321).  It never looks before
 * `start` or across a newline, so "D, not B" still grades D: the pick is
 * reached and accepted before the rejected distractor is ever inspected. */
static bool mc_letter_is_negated(const char *start, const char *letter) {
    const char *p = letter;
    /* Step back over the gap to the previous word: spaces and light separating
     * punctuation only, and never across a line break. */
    while (p > start) {
        char c = p[-1];
        if (c == '\n') return false;
        if (c == ' ' || c == '\t' || c == ',' || c == ';') p--;
        else break;
    }
    /* Read the immediately preceding word (letters/apostrophe), lowercased. */
    const char *wend = p;
    while (p > start && (isalpha((unsigned char)p[-1]) || p[-1] == '\'')) p--;
    size_t wlen = (size_t)(wend - p);
    if (wlen == 0 || wlen >= 16) return false;
    char w[16];
    for (size_t i = 0; i < wlen; i++) w[i] = (char)tolower((unsigned char)p[i]);
    w[wlen] = '\0';

    /* Contraction form: isn't / aren't / wasn't / doesn't / won't / can't. */
    if (wlen >= 3 && strcmp(w + wlen - 3, "n't") == 0) return true;

    static const char *cues[] = {
        "not", "except", "excluding", "exclude", "excludes",
        "eliminate", "eliminates", "eliminated",
        "reject", "rejects", "rejected", "rejecting", NULL
    };
    for (int i = 0; cues[i]; i++) if (strcmp(w, cues[i]) == 0) return true;

    /* Two-word cue: "rule out" / "rules out" / "ruled out". */
    if (strcmp(w, "out") == 0) {
        const char *q = p;
        while (q > start && (q[-1] == ' ' || q[-1] == '\t')) q--;
        const char *rend = q;
        while (q > start && isalpha((unsigned char)q[-1])) q--;
        size_t rl = (size_t)(rend - q);
        if (rl && rl < 8) {
            char r[8];
            for (size_t i = 0; i < rl; i++) r[i] = (char)tolower((unsigned char)q[i]);
            r[rl] = '\0';
            if (!strcmp(r, "rule") || !strcmp(r, "rules") || !strcmp(r, "ruled"))
                return true;
        }
    }
    return false;
}

static char find_answer_letter(const char *generated, int nchoices) {
    if (nchoices <= 0) return '?';
    const char *visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;
    char max_answer = (char)('A' + nchoices - 1);

    char *answer = find_last_answer_marker(visible);
    if (answer) {
        const char *end = answer + strlen(answer);
        if (strlen(answer) > 96) end = answer + 96;
        for (const char *p = answer; p < end && *p; p++) {
            char c = (char)toupper((unsigned char)*p);
            if (c >= 'A' && c <= max_answer) {
                char before = p == visible ? ' ' : p[-1];
                char after = p[1];
                if (!is_letter_boundary(before, after)) continue;
                /* A standalone capital that begins a same-line English word
                 * ("A careful ...") or a contraction ("I'll ...") is prose,
                 * not the model's pick: the real letter comes later on the
                 * line. Skip it; the reverse scan below still recovers it if
                 * it was the only candidate (e.g. "Answer: A is correct"). */
                if (after == '\'') continue;
                if (after == ' ' || after == '\t') {
                    const char *w = p + 1;
                    while (*w == ' ' || *w == '\t') w++;
                    if (islower((unsigned char)*w)) continue;
                }
                /* A distractor explicitly rejected before the pick on the same
                 * line ("not B, ... D") must not win over the real choice. */
                if (mc_letter_is_negated(answer, p)) continue;
                return c;
            }
        }
    }

    for (const char *p = visible + strlen(visible); p > visible; p--) {
        char c = (char)toupper((unsigned char)p[-1]);
        if (c >= 'A' && c <= max_answer) {
            char before = p - 1 == visible ? ' ' : p[-2];
            char after = p[0];
            if (is_letter_boundary(before, after)) return c;
        }
    }
    return '?';
}

static void normalize_integer_answer(const char *p, size_t len,
                                     char *dst, size_t dstlen) {
    while (len > 1 && *p == '0') {
        p++;
        len--;
    }
    if (dstlen == 0) return;
    size_t n = len < dstlen - 1 ? len : dstlen - 1;
    memcpy(dst, p, n);
    dst[n] = '\0';
}

static bool scan_first_integer(const char *start, const char *end,
                               char *dst, size_t dstlen) {
    const char *p = start;
    while (p < end && *p) {
        if (isdigit((unsigned char)*p)) {
            const char *q = p + 1;
            while (q < end && isdigit((unsigned char)*q)) q++;
            normalize_integer_answer(p, (size_t)(q - p), dst, dstlen);
            return true;
        }
        p++;
    }
    return false;
}

static void find_integer_answer(const char *generated, char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    snprintf(dst, dstlen, "?");
    const char *visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;

    char *answer = find_last_answer_marker(visible);
    if (answer) {
        const char *end = answer + strlen(answer);
        if (strlen(answer) > 160) end = answer + 160;
        /* Restrict to the final answer line: digits after it (continued
         * reasoning, footnotes, years) must not override the answer. */
        const char *nl = memchr(answer, '\n', (size_t)(end - answer));
        if (nl) end = nl;
        /* When the line shows arithmetic ("m+n = 256+37 = 293") the stated
         * result is the right-hand side of the LAST '='. Otherwise the first
         * integer on the line is the answer (keeps "Final answer: 082"). */
        const char *eq = NULL;
        for (const char *r = answer; r < end; r++) if (*r == '=') eq = r;
        if (scan_first_integer(eq ? eq + 1 : answer, end, dst, dstlen)) return;
        if (eq && scan_first_integer(answer, end, dst, dstlen)) return;
    }

    const char *last_start = NULL;
    const char *last_end = NULL;
    for (const char *p = visible; *p; p++) {
        if (isdigit((unsigned char)*p)) {
            const char *q = p + 1;
            while (isdigit((unsigned char)*q)) q++;
            last_start = p;
            last_end = q;
            p = q - 1;
        }
    }
    if (last_start && last_end) {
        normalize_integer_answer(last_start, (size_t)(last_end - last_start),
                                 dst, dstlen);
    }
}

static void normalize_compsec_line_spec(const char *p, const char *end,
                                        char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    size_t n = 0;

    /* The model is instructed to finish with:
     *
     *     Answer: <line number or comma-separated line numbers>
     *
     * We still accept harmless surface variants in that final line: "line 9",
     * "9 and 15", "9, 15", or "20-22".  The hidden key decides what set of
     * exact line numbers is acceptable; this routine only normalizes the model
     * text into the same compact line-spec syntax.
     */
    for (; p < end && *p; p++) {
        if (!isdigit((unsigned char)*p)) continue;

        if (n > 0 && n + 1 < dstlen) dst[n++] = ',';
        while (p < end && isdigit((unsigned char)*p)) {
            if (n + 1 < dstlen) dst[n++] = *p;
            p++;
        }
        while (p < end && isspace((unsigned char)*p)) p++;

        if (p < end && *p == '-') {
            if (n + 1 < dstlen) dst[n++] = '-';
            p++;
            while (p < end && isspace((unsigned char)*p)) p++;
            while (p < end && isdigit((unsigned char)*p)) {
                if (n + 1 < dstlen) dst[n++] = *p;
                p++;
            }
        }
        if (p >= end || !*p) break;
    }
    while (n > 0 && (dst[n - 1] == ',' || dst[n - 1] == '-')) n--;
    dst[n] = '\0';
    if (n == 0) snprintf(dst, dstlen, "?");
}

static void find_compsec_answer(const char *generated, char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    snprintf(dst, dstlen, "?");
    const char *visible = strstr(generated, "</think>");
    visible = visible ? visible + 8 : generated;

    char *answer = find_last_answer_marker(visible);
    if (answer) {
        const char *end = answer + strlen(answer);
        if (strlen(answer) > 160) end = answer + 160;
        const char *newline = memchr(answer, '\n', (size_t)(end - answer));
        if (newline) end = newline;
        normalize_compsec_line_spec(answer, end, dst, dstlen);
        if (strcmp(dst, "?") != 0) return;
    }
    find_integer_answer(generated, dst, dstlen);
}

static bool parse_line_spec(const char *spec, bool *set, size_t setlen) {
    bool any = false;
    const char *p = spec;
    while (p && *p) {
        while (*p && !isdigit((unsigned char)*p)) p++;
        if (!*p) break;
        char *end = NULL;
        long a = strtol(p, &end, 10);
        long b = a;
        p = end;
        if (*p == '-') {
            p++;
            b = strtol(p, &end, 10);
            p = end;
        }
        if (a > b) {
            long tmp = a;
            a = b;
            b = tmp;
        }
        if (a < 0) a = 0;
        if (b >= (long)setlen) b = (long)setlen - 1;
        for (long i = a; i <= b; i++) {
            set[i] = true;
            any = true;
        }
    }
    return any;
}

static bool compsec_answer_matches(const char *expected_spec, const char *got_spec) {
    bool expected[256] = {0};
    bool got[256] = {0};
    if (!parse_line_spec(expected_spec, expected, sizeof(expected))) return false;
    if (!parse_line_spec(got_spec, got, sizeof(got))) return false;

    bool hit = false;
    for (size_t i = 0; i < sizeof(got); i++) {
        if (!got[i]) continue;
        if (!expected[i]) return false;
        hit = true;
    }
    /* The prompt asks the model for the single best line, or the smallest exact
     * set when the bug cannot be localized to one line.  The hidden expected
     * answer may be a small audited range when adjacent lines are equivalent
     * locations for the same bug.  Any model-supplied line must be inside that
     * accepted set, and at least one accepted line must be present. */
    return hit;
}

static void find_case_answer(const eval_case *tc, const char *generated,
                             char *dst, size_t dstlen) {
    if (dstlen == 0) return;
    if (eval_case_is_multiple_choice(tc)) {
        dst[0] = find_answer_letter(generated, eval_case_nchoices(tc));
        if (dstlen > 1) dst[1] = '\0';
    } else if (eval_case_is_compsec(tc)) {
        find_compsec_answer(generated, dst, dstlen);
    } else {
        find_integer_answer(generated, dst, dstlen);
    }
}

static bool answer_matches(const eval_case *tc, const char *got) {
    if (eval_case_is_multiple_choice(tc)) {
        return got && got[0] && tc->answer && got[0] == tc->answer[0];
    }
    if (eval_case_is_compsec(tc)) {
        return got && tc->answer && compsec_answer_matches(tc->answer, got);
    }
    char expected[EVAL_ANSWER_MAX];
    normalize_integer_answer(tc->answer, strlen(tc->answer), expected, sizeof(expected));
    return got && strcmp(got, expected) == 0;
}

static char *read_text_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-eval: cannot open trace '%s': %s\n",
                path, strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4-eval: cannot seek trace '%s': %s\n",
                path, strerror(errno));
        fclose(fp);
        return NULL;
    }
    long len_long = ftell(fp);
    if (len_long < 0) {
        fprintf(stderr, "ds4-eval: cannot tell trace size '%s': %s\n",
                path, strerror(errno));
        fclose(fp);
        return NULL;
    }
    rewind(fp);

    size_t len = (size_t)len_long;
    char *buf = malloc(len + 1);
    if (!buf) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        fclose(fp);
        return NULL;
    }
    if (len && fread(buf, 1, len, fp) != len) {
        fprintf(stderr, "ds4-eval: cannot read trace '%s': %s\n",
                path, ferror(fp) ? strerror(errno) : "short read");
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    buf[len] = '\0';
    if (len_out) *len_out = len;
    return buf;
}

static const char *bounded_strstr(const char *start, const char *end,
                                  const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0) return start;
    if ((size_t)(end - start) < nlen) return NULL;
    for (const char *p = start; p + nlen <= end; p++) {
        if (!memcmp(p, needle, nlen)) return p;
    }
    return NULL;
}

static void copy_span(char *dst, size_t dstlen, const char *start, const char *end) {
    if (dstlen == 0) return;
    while (end > start && (end[-1] == '\r' || end[-1] == '\n')) end--;
    size_t n = (size_t)(end - start);
    if (n >= dstlen) n = dstlen - 1;
    memcpy(dst, start, n);
    dst[n] = '\0';
}

static const char *trace_skip_counted_block(const char *line,
                                            const char *line_end,
                                            const char *end) {
    /*
     * Trace regrading must scan metadata lines without being confused by model
     * output.  Prefer the byte count written by trace_write_block(); the older
     * delimiter-only parser is still accepted in trace_copy_model_output for
     * compatibility with existing trace files.
     */
    const char *begin = bounded_strstr(line, line_end, "_BEGIN bytes=");
    if (!begin || begin == line) return NULL;

    size_t label_len = (size_t)(begin - line);
    if (label_len > 96) return NULL;

    char end_marker[128];
    int marker_len = snprintf(end_marker, sizeof(end_marker), "%.*s_END",
                              (int)label_len, line);
    if (marker_len <= 0 || (size_t)marker_len >= sizeof(end_marker)) return NULL;

    const char *bytes = begin + strlen("_BEGIN bytes=");
    char *endptr = NULL;
    unsigned long long declared = strtoull(bytes, &endptr, 10);
    if (endptr == bytes || endptr > line_end) return NULL;

    const char *content = line_end < end ? line_end + 1 : end;
    if (declared > (unsigned long long)(end - content)) return NULL;

    const char *marker = content + (size_t)declared;
    if (marker < end && *marker == '\n') marker++;
    if ((size_t)(end - marker) < (size_t)marker_len ||
        memcmp(marker, end_marker, (size_t)marker_len) != 0)
        return NULL;

    const char *after_marker = marker + marker_len;
    const char *after_line = memchr(after_marker, '\n', (size_t)(end - after_marker));
    return after_line ? after_line + 1 : end;
}

static const char *trace_find_next_case(const char *start, const char *end) {
    const char *p = start;
    while (p < end) {
        const char *line_end = memchr(p, '\n', (size_t)(end - p));
        if (!line_end) line_end = end;

        const char *skip = trace_skip_counted_block(p, line_end, end);
        if (skip && skip > p) {
            p = skip;
            continue;
        }

        const char *case_marker = "===== CASE ";
        size_t marker_len = strlen(case_marker);
        if ((size_t)(line_end - p) >= marker_len &&
            !memcmp(p, case_marker, marker_len))
            return p;

        p = line_end < end ? line_end + 1 : end;
    }
    return NULL;
}

static const char *trace_find_block_begin(const char *start, const char *end,
                                          const char *label) {
    size_t label_len = strlen(label);
    const char *p = start;
    while (p < end) {
        const char *line_end = memchr(p, '\n', (size_t)(end - p));
        if (!line_end) line_end = end;

        if ((size_t)(line_end - p) >= label_len &&
            !memcmp(p, label, label_len))
            return p;

        const char *skip = trace_skip_counted_block(p, line_end, end);
        if (skip && skip > p) {
            p = skip;
            continue;
        }

        p = line_end < end ? line_end + 1 : end;
    }
    return NULL;
}

static bool trace_get_line_field(const char *start, const char *end,
                                 const char *key, char *dst, size_t dstlen) {
    size_t keylen = strlen(key);
    if (dstlen > 0) dst[0] = '\0';

    const char *p = start;
    while (p < end) {
        const char *line_end = memchr(p, '\n', (size_t)(end - p));
        if (!line_end) line_end = end;
        if ((size_t)(line_end - p) >= keylen && !memcmp(p, key, keylen)) {
            copy_span(dst, dstlen, p + keylen, line_end);
            return true;
        }
        p = line_end < end ? line_end + 1 : end;
    }
    return false;
}

static const eval_case *find_eval_case_by_source_id(const char *source,
                                                    const char *id) {
    size_t ncases = sizeof(eval_cases) / sizeof(eval_cases[0]);
    for (size_t i = 0; i < ncases; i++) {
        if (eval_cases[i].source && eval_cases[i].id &&
            !strcmp(eval_cases[i].source, source) &&
            !strcmp(eval_cases[i].id, id))
            return &eval_cases[i];
    }
    return NULL;
}

static char *trace_copy_model_output(const char *case_start, const char *case_end) {
    const char *begin = trace_find_block_begin(case_start, case_end, "MODEL_OUTPUT_BEGIN");
    if (!begin) return NULL;
    const char *line_end = memchr(begin, '\n', (size_t)(case_end - begin));
    if (!line_end) return NULL;
    const char *content = line_end + 1;

    size_t len = 0;
    const char *bytes = bounded_strstr(begin, line_end, "bytes=");
    if (bytes) {
        char *endptr = NULL;
        unsigned long long declared = strtoull(bytes + 6, &endptr, 10);
        if (endptr == bytes + 6 || endptr > line_end ||
            declared > (unsigned long long)(case_end - content))
            return NULL;
        len = (size_t)declared;
        const char *marker = content + len;
        if (marker < case_end && *marker == '\n') marker++;
        if ((size_t)(case_end - marker) < strlen("MODEL_OUTPUT_END") ||
            memcmp(marker, "MODEL_OUTPUT_END", strlen("MODEL_OUTPUT_END")) != 0)
            return NULL;
    } else {
        const char *finish = bounded_strstr(content, case_end, "\nMODEL_OUTPUT_END");
        if (!finish) return NULL;
        len = (size_t)(finish - content);
    }

    char *out = malloc(len + 1);
    if (!out) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        return NULL;
    }
    memcpy(out, content, len);
    out[len] = '\0';
    return out;
}

