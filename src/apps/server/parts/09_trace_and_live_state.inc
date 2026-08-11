/* =========================================================================
 * Trace Diagnostics.
 * =========================================================================
 *
 * The human transcript is not enough to debug prompt-cache misses.  The model
 * may generate text that is semantically accepted as a tool call, while the
 * next OpenAI request re-renders a slightly different canonical DSML block.
 * That creates a token mismatch even if the conversation "looks" continuous.
 *
 * When --trace is enabled we therefore record the exact cache decision and a
 * small token window around the first mismatch between the live KV checkpoint
 * and the incoming prompt.  Normal server logs stay compact; trace files get
 * enough data to diagnose tokenizer-boundary and canonicalization problems.
 */

#define TRACE_CACHE_BEFORE 8
#define TRACE_CACHE_AFTER  8
#define TRACE_CACHE_WINDOW (TRACE_CACHE_BEFORE + 1 + TRACE_CACHE_AFTER)

typedef struct {
    bool valid;
    int old_pos;
    int prompt_len;
    int common;
    int rewind_to;
    int start;
    int count;
    int live_id[TRACE_CACHE_WINDOW];
    int prompt_id[TRACE_CACHE_WINDOW];
} trace_cache_diag;

static void trace_cache_capture(
        trace_cache_diag *d,
        const ds4_tokens *live,
        const ds4_tokens *prompt,
        int old_pos,
        int common)
{
    memset(d, 0, sizeof(*d));
    d->valid = true;
    d->rewind_to = -1;
    d->old_pos = old_pos;
    d->prompt_len = prompt ? prompt->len : 0;
    d->common = common;

    const int live_len = live ? live->len : 0;
    const int prompt_len = prompt ? prompt->len : 0;
    int max_len = live_len > prompt_len ? live_len : prompt_len;
    int start = common - TRACE_CACHE_BEFORE;
    if (start < 0) start = 0;
    int end = common + TRACE_CACHE_AFTER + 1;
    if (end > max_len) end = max_len;
    if (end < start) end = start;

    d->start = start;
    d->count = end - start;
    if (d->count > TRACE_CACHE_WINDOW) d->count = TRACE_CACHE_WINDOW;
    for (int i = 0; i < d->count; i++) {
        int pos = start + i;
        d->live_id[i] = live && pos < live->len ? live->v[pos] : -1;
        d->prompt_id[i] = prompt && pos < prompt->len ? prompt->v[pos] : -1;
    }
}

static const char *trace_cache_miss_reason(const trace_cache_diag *d) {
    if (!d || !d->valid) return "unknown";
    if (d->old_pos == 0) return "no-live-checkpoint";
    if (d->rewind_to >= 0) return "live-prefix-rewind";
    if (d->common != d->old_pos) return "token-mismatch";
    if (d->prompt_len < d->old_pos) return "incoming-prompt-shorter-than-live-checkpoint";
    return "live-prefix-match";
}

static bool trace_cache_memory_reusable(const trace_cache_diag *d) {
    return d && d->valid &&
           (d->rewind_to >= 0 ||
            (d->old_pos > 0 && d->common == d->old_pos &&
             d->prompt_len >= d->old_pos));
}

static void trace_write_escaped_bytes(FILE *fp, const char *p, size_t len) {
    static const char hex[] = "0123456789abcdef";
    fputc('"', fp);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)p[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20 || c == 0x7f) {
            fputs("\\x", fp);
            fputc(hex[c >> 4], fp);
            fputc(hex[c & 15], fp);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

static void trace_write_token(FILE *fp, ds4_engine *engine, int token) {
    if (token < 0) {
        fputs("- <none>", fp);
        return;
    }
    size_t len = 0;
    char *piece = ds4_token_text(engine, token, &len);
    fprintf(fp, "%d ", token);
    trace_write_escaped_bytes(fp, piece, len);
    free(piece);
}

static void trace_write_cache_diag(
        server *s,
        const trace_cache_diag *d,
        const tool_replay_stats *tool_replay,
        int cached,
        const char *cache_source,
        int disk_cached,
        const char *disk_path)
{
    fprintf(s->trace,
            "\n--- cache decision ---\n"
            "live_tokens_before: %d\n"
            "prompt_tokens: %d\n"
            "live_prompt_common: %d\n"
            "memory_token_reusable: %d\n"
            "memory_miss_reason: %s\n"
            "tool_replay: mem=%d disk=%d canonical=%d missing_ids=%d\n"
            "cache_source: %s\n"
            "cached_tokens: %d\n"
            "disk_cached_tokens: %d\n",
            d && d->valid ? d->old_pos : 0,
            d && d->valid ? d->prompt_len : 0,
            d && d->valid ? d->common : 0,
            trace_cache_memory_reusable(d) ? 1 : 0,
            trace_cache_miss_reason(d),
            tool_replay ? tool_replay->mem : 0,
            tool_replay ? tool_replay->disk : 0,
            tool_replay ? tool_replay->canonical : 0,
            tool_replay ? tool_replay->missing_ids : 0,
            cache_source ? cache_source : "none",
            cached,
            disk_cached);
    if (disk_path && disk_path[0]) fprintf(s->trace, "disk_cache_file: %s\n", disk_path);

    if (!d || !d->valid || d->old_pos == 0 ||
        trace_cache_memory_reusable(d))
    {
        return;
    }

    fprintf(s->trace,
            "\nfirst_mismatch_token: %d\n"
            "token_window: [%d..%d)\n",
            d->common,
            d->start,
            d->start + d->count);
    for (int i = 0; i < d->count; i++) {
        int pos = d->start + i;
        int live = d->live_id[i];
        int prompt = d->prompt_id[i];
        const char *mark;
        if (live < 0) mark = "prompt-only";
        else if (prompt < 0) mark = "live-only";
        else mark = live == prompt ? "==" : "!=";

        fprintf(s->trace, "%7d %-11s live ", pos, mark);
        trace_write_token(s->trace, s->engine, live);
        fputs(" | prompt ", s->trace);
        trace_write_token(s->trace, s->engine, prompt);
        fputc('\n', s->trace);
    }
}

static int live_prefix_rewind_target(bool backend_can_rewind,
                                     int old_pos, int prompt_len, int common) {
    if (!backend_can_rewind || prompt_len <= 1 || prompt_len >= old_pos) return -1;
    if (common != prompt_len) return -1;
    return prompt_len - 1;
}

static void trace_time(FILE *fp) {
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    fputs(buf, fp);
}

static uint64_t trace_begin(
        server *s,
        const job *j,
        int cached,
        int effective_prompt_tokens,
        const trace_cache_diag *cache_diag,
        const char *cache_source,
        int disk_cached,
        const char *disk_path) {
    if (!s->trace) return 0;

    pthread_mutex_lock(&s->trace_mu);
    uint64_t id = ++s->trace_seq;
    fprintf(s->trace, "\n===== request %llu ", (unsigned long long)id);
    trace_time(s->trace);
    fprintf(s->trace,
            " =====\nkind: %s\nmodel: %s\nstream: %d\ntools: %d\nthink_mode: %s\nprompt_tokens: %d\neffective_prompt_tokens: %d\ncached_tokens: %d\nmax_tokens: %d\ntemperature: %.3f\ntop_k: %d\ntop_p: %.3f\nmin_p: %.3f\nseed: %llu\n",
            j->req.kind == REQ_CHAT ? "chat" : "completion",
            j->req.model ? j->req.model : "",
            j->req.stream ? 1 : 0,
            j->req.has_tools ? 1 : 0,
            ds4_think_mode_name(j->req.think_mode),
            j->req.prompt.len,
            effective_prompt_tokens,
            cached,
            j->req.max_tokens,
            j->req.temperature,
            j->req.top_k,
            j->req.top_p,
            j->req.min_p,
            (unsigned long long)j->req.seed);
    fprintf(s->trace, "stream_include_usage: %d\n",
            j->req.stream_include_usage ? 1 : 0);
    trace_write_cache_diag(s, cache_diag, &j->req.tool_replay, cached,
                           cache_source, disk_cached, disk_path);
    if (j->req.raw_body) {
        fputs("\n--- raw request json ---\n", s->trace);
        fputs(j->req.raw_body, s->trace);
        if (!j->req.raw_body[0] || j->req.raw_body[strlen(j->req.raw_body) - 1] != '\n') {
            fputc('\n', s->trace);
        }
    }
    if (j->req.prompt_text) {
        fputs("\n--- rendered prompt ---\n", s->trace);
        fputs(j->req.prompt_text, s->trace);
        if (!j->req.prompt_text[0] || j->req.prompt_text[strlen(j->req.prompt_text) - 1] != '\n') {
            fputc('\n', s->trace);
        }
    }
    fputs("\n--- generated text ---\n", s->trace);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
    return id;
}

static void trace_piece(server *s, uint64_t id, const char *piece, size_t len) {
    if (!s->trace || !id || !piece || !len) return;
    pthread_mutex_lock(&s->trace_mu);
    fwrite(piece, 1, len, s->trace);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

static void trace_event(server *s, uint64_t id, const char *fmt, ...) {
    if (!s->trace || !id) return;
    pthread_mutex_lock(&s->trace_mu);
    fputs("\n\n--- trace: ", s->trace);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(s->trace, fmt, ap);
    va_end(ap);
    fputs(" ---\n\n", s->trace);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

static void trace_finish(
        server *s,
        uint64_t id,
        const request *r,
        const char *final_finish,
        int completion,
        bool saw_tool_start,
        bool saw_tool_end,
        const char *parsed_content,
        const char *parsed_reasoning,
        const tool_calls *parsed_calls,
        double elapsed) {
    if (!s->trace || !id) return;

    pthread_mutex_lock(&s->trace_mu);
    fprintf(s->trace,
            "\n\n--- parsed message ---\nfinish: %s\ngenerated_tokens: %d\ndsml_start: %d\ndsml_end: %d\nelapsed_sec: %.3f\n",
            final_finish,
            completion,
            saw_tool_start ? 1 : 0,
            saw_tool_end ? 1 : 0,
            elapsed);
    if (r->kind == REQ_CHAT) {
        if (parsed_reasoning && parsed_reasoning[0]) {
            fputs("\nreasoning:\n", s->trace);
            fputs(parsed_reasoning, s->trace);
            fputc('\n', s->trace);
        }
        if (parsed_content && parsed_content[0]) {
            fputs("\ncontent:\n", s->trace);
            fputs(parsed_content, s->trace);
            fputc('\n', s->trace);
        }
        for (int i = 0; i < parsed_calls->len; i++) {
            const tool_call *tc = &parsed_calls->v[i];
            fprintf(s->trace, "\ntool_call[%d]:\nid: %s\nname: %s\narguments:\n%s\n",
                    i,
                    tc->id ? tc->id : "",
                    tc->name ? tc->name : "",
                    tc->arguments ? tc->arguments : "");
        }
    }
    fprintf(s->trace, "\n===== end request %llu =====\n", (unsigned long long)id);
    fflush(s->trace);
    pthread_mutex_unlock(&s->trace_mu);
}

typedef struct {
    server *srv;
    server_slot *slot;
    req_kind kind;
    int prompt_tokens;
    int cached_tokens;
    char ctx[48];
    const char *phase;
    bool has_tools;
    bool responses_protocol;
    double t0;
    double last_t;
    int last_current;
    bool seen;
    /* SSE keepalive during long prefill: send HTTP/SSE headers ahead of
     * generation and emit a `:` comment line every few seconds so HTTP/TCP
     * idle timeouts on the client side don't close the connection while the
     * server is busy doing prefill. */
    int fd;
    bool stream;
    bool enable_cors;
    bool headers_sent;
    bool stream_failed;
    double last_keepalive;
    job *request_job;
} server_prefill_progress;

static void request_ctx_span(char *buf, size_t len, int cached, int prompt) {
    int suffix = prompt - cached;
    if (suffix < 0) suffix = 0;
    snprintf(buf, len, "%d..%d:%d", cached, prompt, suffix);
}

static void log_flags(char *buf, size_t len, bool responses_protocol,
                      bool tools, bool thinking,
                      bool dsml_start, bool dsml_end) {
    size_t used = 0;
    buf[0] = '\0';
#define ADD_FLAG(name) do { \
    int n = snprintf(buf + used, used < len ? len - used : 0, "%s%s", used ? " " : "", name); \
    if (n > 0) used += (size_t)n; \
} while (0)
    if (responses_protocol) ADD_FLAG("RESPPROTO");
    if (tools) ADD_FLAG("TOOLS");
    if (thinking) ADD_FLAG("THINKING");
    if (dsml_start) ADD_FLAG("DSML_START");
    if (dsml_end) ADD_FLAG("DSML_END");
#undef ADD_FLAG
}

static void log_decode_progress(req_kind kind, int prompt_tokens, int completion,
                                bool responses_protocol,
                                bool tools, bool thinking,
                                bool dsml_start, bool dsml_end,
                                double decode_t0,
                                double *last_t, int *last_completion) {
    const double now = now_sec();
    const double elapsed = now - decode_t0;
    const double interval_s = now - *last_t;
    const int interval_tokens = completion - *last_completion;
    const double chunk_tps = interval_s > 0.0 ? (double)interval_tokens / interval_s : 0.0;
    const double avg_tps = elapsed > 0.0 ? (double)completion / elapsed : 0.0;
    char ctx[48];
    request_ctx_span(ctx, sizeof(ctx),
                     prompt_tokens + *last_completion,
                     prompt_tokens + completion);
    char flags[80];
    log_flags(flags, sizeof(flags), responses_protocol,
              tools, thinking, dsml_start, dsml_end);
    server_log(DS4_LOG_GENERATION,
               "ds4-server: %s ctx=%s gen=%d%s%s decoding chunk=%.2f t/s avg=%.2f t/s %.3fs",
               kind == REQ_CHAT ? "chat" : "completion",
               ctx,
               completion,
               flags[0] ? " " : "",
               flags,
               chunk_tps,
               avg_tps,
               elapsed);
    *last_t = now;
    *last_completion = completion;
}

typedef struct {
    bool inside;
    char tail[8]; /* Long enough for "</think>". */
    int tail_len;
} thinking_state;

static bool thinking_tail_ends_with(const thinking_state *st, const char *s) {
    int n = (int)strlen(s);
    return st->tail_len >= n && !memcmp(st->tail + st->tail_len - n, s, (size_t)n);
}

static void thinking_state_feed(thinking_state *st, const char *p, size_t len) {
    if (!st || !p) return;
    for (size_t i = 0; i < len; i++) {
        if (st->tail_len == (int)sizeof(st->tail)) {
            memmove(st->tail, st->tail + 1, sizeof(st->tail) - 1);
            st->tail_len--;
        }
        st->tail[st->tail_len++] = p[i];
        if (thinking_tail_ends_with(st, "<think>")) st->inside = true;
        else if (thinking_tail_ends_with(st, "</think>")) st->inside = false;
    }
}

static thinking_state thinking_state_from_prompt(const request *r) {
    thinking_state st = {0};
    if (r && r->prompt_text) {
        thinking_state_feed(&st, r->prompt_text, strlen(r->prompt_text));
    } else if (r && ds4_think_mode_enabled(r->think_mode)) {
        st.inside = true;
    }
    return st;
}

/* A completed tool block inside unclosed reasoning can be recovered without
 * predicting what the model will emit after an injected close marker. Keep a
 * short overlap until the opening appears, then wait for its matching end. */
static bool complete_tool_call_inside_thinking(const char *text, size_t len,
                                               size_t *scan_from) {
    if (!text || !scan_from) return false;
    if (*scan_from > len) *scan_from = len;
    const char *start = find_any_tool_start(text + *scan_from);
    if (!start) {
        const size_t hold = 80;
        *scan_from = len > hold ? len - hold : 0;
        return false;
    }
    *scan_from = (size_t)(start - text);
    return find_any_tool_end(start) != NULL;
}

static int server_eval_token(server *s, server_slot *slot, int token,
                             char *err, size_t errlen);

static char *rendered_chat_system_region(const char *prompt_text) {
    if (!prompt_text) return xstrdup("");
    const char *p = prompt_text;
    const char *bos = "<｜begin▁of▁sentence｜>";
    const size_t bos_len = strlen(bos);
    if (!strncmp(p, bos, bos_len)) p += bos_len;
    const char *max_prefix = ds4_think_max_prefix();
    const size_t max_prefix_len = strlen(max_prefix);
    if (max_prefix_len && !strncmp(p, max_prefix, max_prefix_len)) {
        p += max_prefix_len;
    }
    while (*p && isspace((unsigned char)*p)) p++;

    const char *user = strstr(p, "<｜User｜>");
    const char *assistant = strstr(p, "<｜Assistant｜>");
    const char *end = NULL;
    if (user && assistant) end = user < assistant ? user : assistant;
    else end = user ? user : assistant;
    if (!end) end = p + strlen(p);
    while (end > p && isspace((unsigned char)end[-1])) end--;
    return xstrndup(p, (size_t)(end - p));
}

static char *build_invalid_dsml_tool_error_suffix(const request *r,
                                                  const thinking_state *thinking,
                                                  const char *detail) {
    char *system = rendered_chat_system_region(r ? r->prompt_text : NULL);
    buf tool_error = {0};
    buf_puts(&tool_error, "Tool error: invalid DSML tool call");
    if (detail && detail[0]) {
        buf_puts(&tool_error, ": ");
        buf_puts(&tool_error, detail);
    }
    buf_puts(&tool_error,
             "\nThe previous assistant output was not executed because the DSML syntax was malformed. "
             "Emit a new valid DSML tool call, or answer normally if no tool is needed.");
    if (system && system[0]) {
        buf_puts(&tool_error, "\n\nSystem prompt reminder:\n");
        buf_puts(&tool_error, system);
    }

    buf suffix = {0};
    if (r && ds4_think_mode_enabled(r->think_mode) && thinking && thinking->inside) {
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, "<｜end▁of▁sentence｜><｜User｜><tool_result>");
    append_tool_result_text(&suffix, tool_error.ptr ? tool_error.ptr : "");
    buf_puts(&suffix, "</tool_result><｜Assistant｜>");
    buf_puts(&suffix, r && ds4_think_mode_enabled(r->think_mode) ? "<think>" : "</think>");

    free(system);
    buf_free(&tool_error);
    return buf_take(&suffix);
}

static char *build_invalid_glm_tool_error_suffix(const request *r,
                                                 const thinking_state *thinking,
                                                 const char *detail) {
    buf tool_error = {0};
    buf_puts(&tool_error, "Tool error: invalid GLM tool call");
    if (detail && detail[0]) {
        buf_puts(&tool_error, ": ");
        buf_puts(&tool_error, detail);
    }
    buf_puts(&tool_error,
             "\nThe previous assistant output was not executed because the "
             "<tool_call> syntax was malformed. Emit a new valid <tool_call>, "
             "or answer normally if no tool is needed.");

    buf suffix = {0};
    if (r && ds4_think_mode_enabled(r->think_mode) && thinking && thinking->inside) {
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, "<|observation|><tool_response>");
    append_glm_tool_response_text(&suffix, tool_error.ptr ? tool_error.ptr : "");
    buf_puts(&suffix, "</tool_response><|assistant|>");
    buf_puts(&suffix,
             r && ds4_think_mode_enabled(r->think_mode) ?
             "<think>" : "<think></think>");

    buf_free(&tool_error);
    return buf_take(&suffix);
}

static char *build_invalid_tool_call_error_suffix(const request *r,
                                                  const thinking_state *thinking,
                                                  const char *detail) {
    if (r && r->model_syntax == SERVER_MODEL_SYNTAX_GLM) {
        return build_invalid_glm_tool_error_suffix(r, thinking, detail);
    }
    return build_invalid_dsml_tool_error_suffix(r, thinking, detail);
}

static int server_next_prefill_slot_locked(const server *s) {
    if (!s || s->slot_count <= 0) return -1;
    for (int n = 1; n <= s->slot_count; n++) {
        int id = (s->last_prefill_slot + n) % s->slot_count;
        if (s->slots[id].prefill_waiting) return id;
    }
    return -1;
}

static bool server_prefill_enter(server *s, server_slot *slot) {
    if (!s || !slot || slot_job_cancelled(slot)) return false;
    if (!s->batched_mode) {
        pthread_mutex_lock(&s->inference_mu);
        if (slot_job_cancelled(slot)) {
            pthread_mutex_unlock(&s->inference_mu);
            return false;
        }
        return true;
    }

    pthread_mutex_lock(&s->model_mu);
    slot->prefill_waiting = true;
    pthread_cond_broadcast(&s->model_cv);
    while (!g_stop_requested && !slot_job_cancelled(slot) &&
           (s->model_busy || s->decode_pending > 0 ||
            server_next_prefill_slot_locked(s) != slot->id)) {
        pthread_cond_wait(&s->model_cv, &s->model_mu);
    }
    if (g_stop_requested || slot_job_cancelled(slot)) {
        slot->prefill_waiting = false;
        pthread_cond_broadcast(&s->model_cv);
        pthread_mutex_unlock(&s->model_mu);
        return false;
    }
    slot->prefill_waiting = false;
    s->last_prefill_slot = slot->id;
    s->model_busy = true;
    pthread_mutex_unlock(&s->model_mu);
    pthread_mutex_lock(&s->inference_mu);
    return true;
}

static void server_prefill_leave(server *s) {
    if (!s) return;
    pthread_mutex_unlock(&s->inference_mu);
    if (!s->batched_mode) return;
    pthread_mutex_lock(&s->model_mu);
    s->model_busy = false;
    pthread_cond_broadcast(&s->model_cv);
    pthread_mutex_unlock(&s->model_mu);
}

static int server_prefill_quantum_for(const server *s,
                                      bool generation_active) {
    return generation_active ? s->mixed_prefill_quantum : 2048;
}

static int server_prefill_quantum(server *s) {
    pthread_mutex_lock(&s->model_mu);
    bool generation_active = s->active_generations > 0;
    pthread_mutex_unlock(&s->model_mu);
    return server_prefill_quantum_for(s, generation_active);
}

/* Synchronize one resident slot without monopolizing the model executor.  A
 * non-matching prompt is first rebuilt to one quantum; a matching checkpoint
 * advances from its current frontier. Absolute positions remain session-local,
 * so compressor alignment is independent of scheduler order. */
static int server_session_sync(server *s, server_slot *slot,
                               const ds4_tokens *prompt,
                               char *err, size_t errlen) {
    if (!s || !slot || !prompt) return 1;
    if (!s->batched_mode) {
        if (!server_prefill_enter(s, slot)) return DS4_SESSION_SYNC_INTERRUPTED;
        int rc = ds4_session_sync(slot->session, prompt, err, errlen);
        server_prefill_leave(s);
        return rc;
    }

    pthread_mutex_lock(&s->inference_mu);
    int live = ds4_session_pos(slot->session);
    int common = ds4_session_common_prefix(slot->session, prompt);
    pthread_mutex_unlock(&s->inference_mu);
    int done = common == live && prompt->len >= live ? live : 0;
    bool called = false;

    while (!g_stop_requested && !slot_job_cancelled(slot) &&
           (!called || done < prompt->len)) {
        int quantum = server_prefill_quantum(s);
        int target = done + quantum;
        if (target > prompt->len || target < done) target = prompt->len;
        if (target <= 0) target = prompt->len;

        ds4_tokens prefix = *prompt;
        prefix.len = target;
        if (!server_prefill_enter(s, slot)) return DS4_SESSION_SYNC_INTERRUPTED;
        int rc = ds4_session_sync(slot->session, &prefix, err, errlen);
        if (rc == 0) done = ds4_session_pos(slot->session);
        server_prefill_leave(s);
        called = true;
        if (rc != 0) return rc;
        if (done >= prompt->len) return 0;
        if (done < target) {
            if (err && errlen) snprintf(err, errlen, "prefill made no progress");
            return 1;
        }
    }
    return (g_stop_requested || slot_job_cancelled(slot)) ?
           DS4_SESSION_SYNC_INTERRUPTED : 0;
}

static bool append_rendered_suffix_to_live_session(server *s, server_slot *slot,
                                                   const char *suffix,
                                                   int *tokens_appended,
                                                   char *err, size_t errlen) {
    if (tokens_appended) *tokens_appended = 0;
    if (!s || !slot || !suffix || !suffix[0]) return true;
    const ds4_tokens *live = ds4_session_tokens(slot->session);
    if (!live) {
        if (err && errlen) snprintf(err, errlen, "live session is unavailable");
        return false;
    }

    ds4_tokens target = {0};
    build_prompt_from_exact_prefix_and_text_suffix(s->engine, live, suffix, &target);
    const int before = ds4_session_pos(slot->session);
    bool ok = server_session_sync(s, slot, &target, err, errlen) == 0;
    if (ok && tokens_appended) {
        int delta = ds4_session_pos(slot->session) - before;
        *tokens_appended = delta > 0 ? delta : 0;
    }
    ds4_tokens_free(&target);
    return ok;
}

static bool continue_after_invalid_dsml(server *s, server_slot *slot,
                                        const request *r,
                                        const thinking_state *thinking,
                                        const char *detail,
                                        int *tokens_appended,
                                        char *err, size_t errlen) {
    char *suffix = build_invalid_tool_call_error_suffix(r, thinking, detail);
    bool ok = append_rendered_suffix_to_live_session(s, slot, suffix,
                                                     tokens_appended,
                                                     err, errlen);
    free(suffix);
    return ok;
}

static bool should_remember_thinking_checkpoint(const request *r,
                                                const thinking_state *thinking,
                                                const char *finish) {
    if (!r || r->kind != REQ_CHAT || r->has_tools) return false;
    if (r->prompt_preserves_reasoning) return false;
    if (!ds4_think_mode_enabled(r->think_mode)) return false;
    if (finish && (!strcmp(finish, "error") || !strcmp(finish, "length"))) return false;
    if (thinking && thinking->inside) return false;
    return true;
}

static void log_tool_calls_summary(const char *ctx, const tool_calls *calls,
                                   bool responses_protocol) {
    if (!calls || calls->len == 0) return;
    buf names = {0};
    buf ids = {0};
    for (int i = 0; i < calls->len; i++) {
        if (i) buf_putc(&names, ',');
        if (i) buf_putc(&ids, ',');
        buf_puts(&names, calls->v[i].name ? calls->v[i].name : "?");
        buf_puts(&ids, calls->v[i].id ? calls->v[i].id : "?");
    }
    char flags[32];
    log_flags(flags, sizeof(flags), responses_protocol, false, false, false, false);
    server_log(DS4_LOG_TOOL,
               "ds4-server: tool calls ctx=%s%s%s n=%d raw_tool_text=%d ids=[%s] names=[%s]",
               ctx,
               flags[0] ? " " : "",
               flags,
               calls->len,
               calls->raw_tool_text && calls->raw_tool_text[0] ? 1 : 0,
               ids.ptr ? ids.ptr : "",
               names.ptr ? names.ptr : "");
    buf_free(&ids);
    buf_free(&names);
}

static void server_progress_cb(void *ud, const char *event, int current, int total) {
    server_prefill_progress *p = ud;
    if (!p || !event || job_cancelled(p->request_job)) return;
    const bool is_chunk = strcmp(event, "prefill_chunk") == 0;
    const bool is_display = strcmp(event, "prefill_display") == 0;
    if (!is_chunk && !is_display) return;

    double now = now_sec();
    /* Keep the HTTP/SSE connection alive while prefill runs.  We write the SSE
     * response headers the first time the callback fires and then emit a
     * comment line (`:` prefix, ignored by SSE clients) every few seconds.
     * A failed write marks the job cancelled; the session callback then stops
     * prefill at the next backend-safe boundary. */
    if (p->stream && p->fd >= 0 && !p->stream_failed) {
        if (!p->headers_sent) {
            p->headers_sent = true;
            if (sse_headers(p->fd, p->enable_cors)) {
                p->last_keepalive = now;
            } else {
                p->stream_failed = true;
                job_mark_cancelled(p->request_job);
                return;
            }
        } else if (now - p->last_keepalive >= 5.0) {
            static const char ka[] = ": prefill\n\n";
            if (send_all(p->fd, ka, sizeof(ka) - 1)) {
                p->last_keepalive = now;
            } else {
                p->stream_failed = true;
                job_mark_cancelled(p->request_job);
                return;
            }
        }
    }
    if (is_display) return;
    double elapsed = now - p->t0;
    if (p->seen && current == p->last_current) {
        if (p->srv && p->slot && current > p->cached_tokens) {
            kv_cache_maybe_store_continued(p->srv, p->slot);
        }
        return;
    }
    int display_start = p->cached_tokens;
    if (display_start < 0 || display_start > p->prompt_tokens) display_start = 0;
    int display_total = p->prompt_tokens - display_start;
    if (display_total <= 0) {
        display_start = 0;
        display_total = p->prompt_tokens > total ? p->prompt_tokens : total;
    }
    int display_current = current - display_start;
    if (display_current < 0) display_current = 0;
    if (display_current > display_total) display_current = display_total;
    double pct = display_total > 0 ? 100.0 * (double)display_current / (double)display_total : 100.0;
    double avg_tps = elapsed > 0.0 ? (double)display_current / elapsed : 0.0;
    int interval_tokens = p->seen ? current - p->last_current : 0;
    if (interval_tokens < 0) interval_tokens = 0;
    double interval_s = p->seen ? now - p->last_t : 0.0;
    double chunk_tps = interval_s > 0.0 ? (double)interval_tokens / interval_s : 0.0;
    p->last_current = current;
    p->last_t = now;
    p->seen = true;
    char flags[64];
    log_flags(flags, sizeof(flags), p->responses_protocol,
              p->has_tools, false, false, false);
    const char *phase = p->phase ? p->phase : "prefill";
    server_log(DS4_LOG_PREFILL,
               "ds4-server: %s ctx=%s%s%s %s chunk %d/%d (%.1f%%) chunk=%.2f t/s avg=%.2f t/s %.3fs",
               p->kind == REQ_CHAT ? "chat" : "completion",
               p->ctx,
               flags[0] ? " " : "",
               flags,
               phase,
               display_current,
               display_total,
               pct,
               chunk_tps,
               avg_tps,
               elapsed);
    if (p->srv && p->slot && current > p->cached_tokens) {
        kv_cache_maybe_store_continued(p->srv, p->slot);
    }
}

static void send_prefill_failure_response(server *s, const job *j,
                                          const server_prefill_progress *progress,
                                          const char *ctx, const char *flags,
                                          const char *err) {
    const char *kind = j->req.kind == REQ_CHAT ? "chat" : "completion";
    if (j->req.stream && progress && progress->headers_sent) {
        if (progress->stream_failed) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s%s%s prefill failed after stream closed: %s",
                       kind, ctx, flags && flags[0] ? " " : "",
                       flags && flags[0] ? flags : "", err);
            return;
        }
        if (!sse_error_event(j->fd, &j->req, err)) {
            server_log(DS4_LOG_GENERATION,
                       "ds4-server: %s ctx=%s%s%s prefill SSE error failed: %s",
                       kind, ctx, flags && flags[0] ? " " : "",
                       flags && flags[0] ? flags : "", err);
        }
        return;
    }
    http_error(j->fd, s->enable_cors, 500, err);
}

static char *build_tool_checkpoint_suffix(const request *r, const char *content,
                                          const char *reasoning, const tool_calls *calls) {
    const server_model_syntax syntax =
        r ? r->model_syntax : SERVER_MODEL_SYNTAX_DEEPSEEK;
    buf suffix = {0};
    if (r && ds4_think_mode_enabled(r->think_mode)) {
        buf_puts(&suffix, reasoning ? reasoning : "");
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, content ? content : "");
    append_tool_calls_text_for_syntax(&suffix, syntax, calls,
                                      r ? &r->tool_orders : NULL);
    if (syntax != SERVER_MODEL_SYNTAX_GLM) {
        buf_puts(&suffix, "<｜end▁of▁sentence｜>");
    }
    return buf_take(&suffix);
}

static char *build_responses_visible_assistant_suffix(const request *r,
                                                      const char *content,
                                                      const char *reasoning,
                                                      const tool_calls *calls) {
    const server_model_syntax syntax =
        r ? r->model_syntax : SERVER_MODEL_SYNTAX_DEEPSEEK;
    buf suffix = {0};
    /* This suffix mirrors what a Responses client can replay, not necessarily
     * every token in KV.  Hidden reasoning stays live in the session unless the
     * next client replay is expected to include it.  In practice, pi replays
     * reasoning summaries for tool-call turns, but not for final assistant
     * answers; Codex currently requests no summaries at all.  So only include
     * reasoning in the remembered visible prefix when this assistant turn ended
     * in tool calls.  A client that does replay final-answer reasoning will not
     * match this visible shortcut and can still use exact token-prefix replay. */
    if (r && ds4_think_mode_enabled(r->think_mode)) {
        if (r->reasoning_summary_emit && calls && calls->len > 0) {
            buf_puts(&suffix, reasoning ? reasoning : "");
        }
        buf_puts(&suffix, "</think>");
    }
    buf_puts(&suffix, content ? content : "");
    append_tool_calls_text_for_syntax(&suffix, syntax, calls,
                                      r ? &r->tool_orders : NULL);
    if (syntax != SERVER_MODEL_SYNTAX_GLM) {
        buf_puts(&suffix, "<｜end▁of▁sentence｜>");
    }
    return buf_take(&suffix);
}

/* In thinking mode without tools, old assistant reasoning is intentionally not
 * rendered back into later prompts.  The sampled live graph still contains the
 * reasoning bytes, so the next request would miss the session cache even though
 * the visible conversation prefix is logically the same.
 *
 *   prompt-without-final-<think> + </think> + visible-content + eos
 *
 * is exactly the visible prefix that render_chat_prompt_text() will produce on
 * the next turn.  Do not rebuild the KV cache to erase hidden reasoning here:
 * that caused long post-answer pauses and threw away useful sampled state.
 * Instead, remember the visible bytes as a key for the current sampled frontier.
 * The next request can then continue from live KV while tokenizing only the new
 * visible suffix. */
static char *build_toolless_thinking_visible_text(const request *r,
                                                  const char *content) {
    if (!r || !r->prompt_text) return NULL;
    if (!ds4_think_mode_enabled(r->think_mode)) return NULL;

    size_t pt_len = strlen(r->prompt_text);
    const char *think_tag = "<think>";
    size_t tag_len = strlen(think_tag);
    if (pt_len < tag_len ||
        memcmp(r->prompt_text + pt_len - tag_len, think_tag, tag_len) != 0) {
        return NULL;
    }

    buf visible = {0};
    buf_append(&visible, r->prompt_text, pt_len - tag_len);
    buf_puts(&visible, "</think>");
    buf_puts(&visible, content ? content : "");
    buf_puts(&visible, "<｜end▁of▁sentence｜>");
    return buf_take(&visible);
}

static void remember_thinking_checkpoint(server *s, server_slot *slot,
                                         const job *j, const char *ctx,
                                         uint64_t trace_id, const char *content) {
    char *visible = build_toolless_thinking_visible_text(&j->req, content);
    if (!visible) return;

    thinking_live_remember(s, slot, visible);
    server_log(DS4_LOG_KVCACHE,
               "ds4-server: thinking live checkpoint remembered ctx=%s live=%d visible=%zu",
               ctx, ds4_session_pos(slot->session), strlen(visible));
    trace_event(s, trace_id,
                "thinking live checkpoint remembered: live=%d visible=%zu",
                ds4_session_pos(slot->session), strlen(visible));
    free(visible);
}

/* After a successful tool-call finish, make the live checkpoint match what the
 * next request will render.  Usually that is just the exact DSML remembered by
 * tool id.  If a client sends a tool call without an id we know, the fallback
 * renderer still builds valid DSML from JSON, and this function either rewrites
 * the short suffix in place or reloads an older disk checkpoint before replay. */
static void canonicalize_tool_checkpoint(server *s, server_slot *slot,
                                         job *j, const char *ctx,
                                         uint64_t trace_id, const char *content,
                                         const char *reasoning, const tool_calls *calls) {
    if (!calls || calls->len == 0 || !j->req.prompt_text) return;

    char *suffix_text = build_tool_checkpoint_suffix(&j->req, content, reasoning, calls);

    buf rendered = {0};
    buf_puts(&rendered, j->req.prompt_text);
    buf_puts(&rendered, suffix_text);

    ds4_tokens canonical = {0};
    ds4_tokenize_rendered_chat(s->engine, rendered.ptr ? rendered.ptr : "", &canonical);
    const int live_len = ds4_session_pos(slot->session);
    const int common = ds4_session_common_prefix(slot->session, &canonical);
    if (common == live_len && canonical.len == live_len) goto done;

    size_t live_text_len = 0;
    char *live_text = render_tokens_text(s->engine,
                                         ds4_session_tokens(slot->session),
                                         &live_text_len);
    if (live_text_len == rendered.len &&
        (live_text_len == 0 || memcmp(live_text, rendered.ptr, live_text_len) == 0))
    {
        /* The graph already represents the bytes the next request will render.
         * Token-level canonicalization would only replace a valid sampled
         * history with a different BPE spelling of the same transcript. */
        free(live_text);
        goto done;
    }
    free(live_text);

    if (common < j->req.prompt.len) {
        trace_event(s, trace_id,
                    "tool checkpoint canonicalization skipped: common=%d prompt=%d live=%d canonical=%d",
                    common, j->req.prompt.len, live_len, canonical.len);
        goto done;
    }

    char err[160] = {0};
    pthread_mutex_lock(&s->inference_mu);
    ds4_session_rewrite_result rr =
        ds4_session_rewrite_from_common(slot->session, &canonical, common,
                                        err, sizeof(err));
    pthread_mutex_unlock(&s->inference_mu);
    if (rr == DS4_SESSION_REWRITE_OK) {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: tool checkpoint canonicalized ctx=%s common=%d live=%d canonical=%d",
                   ctx, common, live_len, canonical.len);
        trace_event(s, trace_id,
                    "tool checkpoint canonicalized: common=%d live=%d canonical=%d",
                    common, live_len, canonical.len);
    } else if (rr == DS4_SESSION_REWRITE_REBUILD_NEEDED) {
        /* The generated DSML suffix and the canonical prompt share a prefix,
         * but the generated tail is too large to overwrite safely inside the
         * live raw-window ring.  Prefer an older disk checkpoint over replaying
         * a very long conversation from token zero. */
        char *path = NULL;
        ds4_tokens effective = {0};
        int loaded = kv_cache_try_load_text(s, slot,
                                            rendered.ptr ? rendered.ptr : "",
                                            &effective, &path, NULL, false);
        if (loaded == 0) {
            pthread_mutex_lock(&s->inference_mu);
            ds4_session_invalidate(slot->session);
            pthread_mutex_unlock(&s->inference_mu);
        }

        char sync_err[160] = {0};
        const ds4_tokens *sync_prompt = loaded > 0 ? &effective : &canonical;
        char rebuild_ctx[48];
        request_ctx_span(rebuild_ctx, sizeof(rebuild_ctx), loaded, sync_prompt->len);
        int replay_tokens = sync_prompt->len - loaded;
        if (replay_tokens < 0) replay_tokens = sync_prompt->len;
        int canonical_tail_tokens = canonical.len - common;
        if (canonical_tail_tokens < 0) canonical_tail_tokens = canonical.len;
        int discarded_live_tokens = live_len - common;
        if (discarded_live_tokens < 0) discarded_live_tokens = 0;
        const char *source = loaded > 0 ? "disk" : "full";
        const double rebuild_t0 = now_sec();
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: tool checkpoint canonicalization needs %d tokens rebuild ctx=%s request_ctx=%s reason=canonical-tail-rewrite tail=%d discard=%d common=%d live=%d target=%d cached=%d source=%s%s%s",
                   replay_tokens,
                   rebuild_ctx,
                   ctx,
                   canonical_tail_tokens,
                   discarded_live_tokens,
                   common,
                   live_len,
                   canonical.len,
                   loaded,
                   source,
                   path ? " file=" : "",
                   path ? path : "");
        server_prefill_progress rebuild_progress = {
            .srv = s,
            .slot = slot,
            .kind = j->req.kind,
            .prompt_tokens = sync_prompt->len,
            .cached_tokens = loaded,
            .phase = "tool checkpoint rebuild",
            .has_tools = j->req.has_tools,
            .t0 = rebuild_t0,
            .fd = j->fd,
            .stream = j->req.stream,
            .enable_cors = s->enable_cors,
            .request_job = j,
            /* Tool checkpoint rebuild only runs after the response stream is
             * already in flight, so the SSE headers were sent long ago.
             * Pre-arm the flag so the progress callback only emits keepalive
             * comments and never tries to write a second set of headers. */
            .headers_sent = true,
        };
        snprintf(rebuild_progress.ctx, sizeof(rebuild_progress.ctx), "%s", rebuild_ctx);
        ds4_session_set_progress(slot->session, server_progress_cb, &rebuild_progress);
        ds4_session_set_display_progress(slot->session, server_progress_cb, &rebuild_progress);
        if (server_session_sync(s, slot, sync_prompt,
                                sync_err, sizeof(sync_err)) == 0) {
            ds4_session_set_progress(slot->session, NULL, NULL);
            ds4_session_set_display_progress(slot->session, NULL, NULL);
            const double rebuild_sec = now_sec() - rebuild_t0;
            if (loaded > 0) {
                server_log(DS4_LOG_KVCACHE,
                           "ds4-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=disk cached=%d replay=%d target=%d %.3fs",
                           rebuild_ctx, ctx, loaded, replay_tokens, canonical.len, rebuild_sec);
                trace_event(s, trace_id,
                            "tool checkpoint canonicalized via disk: common=%d live=%d canonical=%d cached=%d file=%s",
                            common, live_len, canonical.len, loaded, path ? path : "");
            } else {
                server_log(DS4_LOG_KVCACHE,
                           "ds4-server: tool checkpoint rebuild done ctx=%s request_ctx=%s source=full cached=0 replay=%d target=%d %.3fs",
                           rebuild_ctx, ctx, replay_tokens, canonical.len, rebuild_sec);
                trace_event(s, trace_id,
                            "tool checkpoint canonicalized via rebuild: common=%d live=%d canonical=%d reason=%s",
                            common, live_len, canonical.len, err);
            }
        } else {
            ds4_session_set_progress(slot->session, NULL, NULL);
            ds4_session_set_display_progress(slot->session, NULL, NULL);
            server_log(DS4_LOG_KVCACHE,
                       "ds4-server: tool checkpoint rebuild failed ctx=%s request_ctx=%s source=%s cached=%d replay=%d target=%d error=\"%s\"",
                       rebuild_ctx, ctx, source, loaded, replay_tokens,
                       canonical.len, sync_err);
            trace_event(s, trace_id, "tool checkpoint canonicalization failed after rebuild request: %s", sync_err);
        }
        ds4_tokens_free(&effective);
        free(path);
    } else {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: tool checkpoint canonicalization failed ctx=%s common=%d live=%d canonical=%d error=\"%s\"",
                   ctx, common, live_len, canonical.len, err);
        trace_event(s, trace_id, "tool checkpoint canonicalization failed: %s", err);
    }

done:
    ds4_tokens_free(&canonical);
    buf_free(&rendered);
    free(suffix_text);
}

static bool should_canonicalize_tool_checkpoint(const server *s, const tool_calls *calls) {
    if (!calls || calls->len == 0) return false;
    if (s && !s->disable_exact_dsml_tool_replay &&
        calls->raw_tool_text && calls->raw_tool_text[0])
    {
        return false;
    }
    return true;
}

static void server_generation_enter(server *s) {
    if (!s || !s->batched_mode) return;
    pthread_mutex_lock(&s->model_mu);
    s->active_generations++;
    pthread_cond_broadcast(&s->model_cv);
    pthread_mutex_unlock(&s->model_mu);
}

static void server_generation_leave(server *s) {
    if (!s || !s->batched_mode) return;
    pthread_mutex_lock(&s->model_mu);
    if (s->active_generations > 0) s->active_generations--;
    pthread_cond_broadcast(&s->model_cv);
    pthread_mutex_unlock(&s->model_mu);
}

/* model_mu must be held. An in-flight batch owns the session until it
 * completes, but a pending token can be withdrawn without touching backend
 * state. */
static bool server_cancel_pending_decode_locked(server *s, server_slot *slot) {
    if (!s || !slot || !slot->decode_pending || slot->decode_in_flight) return false;
    slot->decode_pending = false;
    s->decode_pending--;
    slot->decode_rc = DS4_SESSION_SYNC_INTERRUPTED;
    snprintf(slot->decode_err, sizeof(slot->decode_err), "client disconnected");
    slot->decode_done = true;
    pthread_cond_broadcast(&s->model_cv);
    return true;
}

static int server_eval_token(server *s, server_slot *slot, int token,
                             char *err, size_t errlen) {
    if (!s || !slot) return 1;
    if (!s->batched_mode) {
        if (g_stop_requested || slot_job_cancelled(slot)) {
            if (err && errlen) snprintf(err, errlen, "%s",
                                        g_stop_requested ? "shutdown requested" :
                                                           "client disconnected");
            return DS4_SESSION_SYNC_INTERRUPTED;
        }
        pthread_mutex_lock(&s->inference_mu);
        int rc = ds4_session_eval(slot->session, token, err, errlen);
        pthread_mutex_unlock(&s->inference_mu);
        return rc;
    }

    pthread_mutex_lock(&s->model_mu);
    if (g_stop_requested || slot_job_cancelled(slot)) {
        pthread_mutex_unlock(&s->model_mu);
        if (err && errlen) snprintf(err, errlen, "%s",
                                    g_stop_requested ? "shutdown requested" :
                                                       "client disconnected");
        return DS4_SESSION_SYNC_INTERRUPTED;
    }
    if (slot->decode_pending || slot->decode_in_flight) {
        pthread_mutex_unlock(&s->model_mu);
        if (err && errlen) snprintf(err, errlen, "session already has a decode in flight");
        return 1;
    }
    slot->decode_token = token;
    slot->decode_rc = 1;
    slot->decode_err[0] = '\0';
    slot->decode_done = false;
    slot->decode_pending = true;
    s->decode_pending++;
    pthread_cond_broadcast(&s->model_cv);
    while (!slot->decode_done) {
        const bool client_cancelled = slot_job_cancelled(slot);
        if ((client_cancelled || g_stop_requested) &&
            server_cancel_pending_decode_locked(s, slot)) {
            if (!client_cancelled) {
                snprintf(slot->decode_err, sizeof(slot->decode_err),
                         "shutdown requested");
            }
            break;
        }
        /* An in-flight backend call still owns the session. Even during
         * shutdown or client cancellation, wait for that safe boundary before
         * the stack-owned job and its cancellation callback can be released. */
        pthread_cond_wait(&s->model_cv, &s->model_mu);
    }
    int rc = slot->decode_rc;
    if (g_stop_requested && rc == 0) rc = DS4_SESSION_SYNC_INTERRUPTED;
    if (rc != 0 && err && errlen) {
        snprintf(err, errlen, "%s",
                 g_stop_requested ? "shutdown requested" :
                 (slot->decode_err[0] ? slot->decode_err : "decode interrupted"));
    }
    slot->decode_done = false;
    pthread_mutex_unlock(&s->model_mu);
    return rc;
}

static long server_decode_coalesce_us(void) {
    long us = 2000;
    const char *env = getenv("DS4_SERVER_DECODE_COALESCE_US");
    if (env && env[0]) {
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (end != env && *end == '\0' && v >= 0 && v <= 100000) us = v;
    }
    return us;
}

static void timespec_add_us(struct timespec *ts, long us) {
    if (!ts || us <= 0) return;
    ts->tv_nsec += (us % 1000000L) * 1000L;
    ts->tv_sec += us / 1000000L + ts->tv_nsec / 1000000000L;
    ts->tv_nsec %= 1000000000L;
}

static void *decode_worker_main(void *arg) {
    server *s = arg;
    ds4_decode_item *items = xmalloc((size_t)s->slot_count * sizeof(*items));
    server_slot **members = xmalloc((size_t)s->slot_count * sizeof(*members));
    const long coalesce_us = server_decode_coalesce_us();
    const bool log_batches = getenv("DS4_SERVER_BATCH_LOG") != NULL;

    pthread_mutex_lock(&s->model_mu);
    for (;;) {
        while (s->decode_pending == 0 && !s->model_stopping) {
            pthread_cond_wait(&s->model_cv, &s->model_mu);
        }
        if (s->decode_pending == 0 && s->model_stopping) break;

        if (coalesce_us > 0 && s->decode_pending < s->slot_count) {
            struct timespec deadline;
            clock_gettime(CLOCK_REALTIME, &deadline);
            timespec_add_us(&deadline, coalesce_us);
            int observed = s->decode_pending;
            while (!s->model_stopping && s->decode_pending < s->slot_count &&
                   s->decode_pending < s->active_generations) {
                int rc = pthread_cond_timedwait(&s->model_cv, &s->model_mu,
                                                &deadline);
                if (rc == ETIMEDOUT) break;
                if (s->decode_pending == observed && rc != 0) break;
                observed = s->decode_pending;
            }
        }

        while (s->model_busy) {
            pthread_cond_wait(&s->model_cv, &s->model_mu);
        }
        if (s->model_stopping && s->decode_pending == 0) break;

        int count = 0;
        for (int i = 0; i < s->slot_count; i++) {
            server_slot *slot = &s->slots[i];
            if (!slot->decode_pending) continue;
            slot->decode_pending = false;
            slot->decode_in_flight = true;
            s->decode_pending--;
            members[count] = slot;
            items[count].session = slot->session;
            items[count].token = slot->decode_token;
            count++;
        }
        if (count == 0) continue;
        s->model_busy = true;
        pthread_mutex_unlock(&s->model_mu);

        char batch_err[160] = {0};
        const double batch_t0 = log_batches ? now_sec() : 0.0;
        pthread_mutex_lock(&s->inference_mu);
        int rc = ds4_sessions_eval_batch(items, count,
                                         batch_err, sizeof(batch_err));
        pthread_mutex_unlock(&s->inference_mu);
        if (log_batches) {
            server_log(DS4_LOG_DEFAULT,
                       "ds4-server: decode batch count=%d elapsed=%.3f ms status=%s",
                       count, (now_sec() - batch_t0) * 1000.0,
                       rc == 0 ? "ok" : "error");
        }

        pthread_mutex_lock(&s->model_mu);
        s->model_busy = false;
        for (int i = 0; i < count; i++) {
            server_slot *slot = members[i];
            slot->decode_in_flight = false;
            slot->decode_rc = rc;
            if (rc != 0) {
                snprintf(slot->decode_err, sizeof(slot->decode_err), "%s",
                         batch_err[0] ? batch_err : "batched decode failed");
            }
            slot->decode_done = true;
        }
        pthread_cond_broadcast(&s->model_cv);
    }
    pthread_mutex_unlock(&s->model_mu);
    free(members);
    free(items);
    return NULL;
}

static uint64_t server_next_sequence(server *s) {
    pthread_mutex_lock(&s->mu);
    uint64_t seq = ++s->seq;
    pthread_mutex_unlock(&s->mu);
    return seq;
}

