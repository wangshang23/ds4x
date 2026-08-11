static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static pthread_mutex_t server_log_mu = PTHREAD_MUTEX_INITIALIZER;

static void server_log(ds4_log_type type, const char *fmt, ...) {
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    char ts[16];
    strftime(ts, sizeof(ts), "%m%d %H:%M:%S", &tm);

    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    int n = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);

    pthread_mutex_lock(&server_log_mu);
    fprintf(stderr, "%s ", ts);
    if (n < 0) {
        ds4_log(stderr, type, "%s", fmt);
    } else {
        char *line = xmalloc((size_t)n + 1);
        vsnprintf(line, (size_t)n + 1, fmt, ap);
        ds4_log(stderr, type, "%s", line);
        free(line);
    }
    va_end(ap);
    fputc('\n', stderr);
    pthread_mutex_unlock(&server_log_mu);
}

typedef struct job job;
typedef struct server_slot server_slot;

typedef ds4_kvstore_entry kv_entry;
typedef ds4_kvstore_options kv_cache_options;
typedef ds4_kvstore kv_disk_cache;

typedef enum {
    TOOL_MEMORY_RAM = 0,
    TOOL_MEMORY_DISK = 1,
} tool_memory_source;

typedef struct tool_memory_entry tool_memory_entry;

typedef struct {
    char *dsml;
    size_t len;
    size_t bytes;
    int refs;
    uint64_t seen;
    tool_memory_entry *entries;
} tool_memory_block;

struct tool_memory_entry {
    char *id;
    tool_memory_block *block;
    size_t bytes;
    uint64_t stamp;
    tool_memory_source source;
    tool_memory_entry *prev;
    tool_memory_entry *next;
    tool_memory_entry *block_next;
};

typedef struct {
    rax *by_id;
    rax *by_block;
    tool_memory_entry *head;
    tool_memory_entry *tail;
    int entries;
    int max_entries;
    size_t bytes;
    size_t max_bytes;
    uint64_t clock;
    uint64_t scan_clock;
} tool_memory;

typedef struct {
    bool valid;
    /* Token frontier of a live assistant tool-call turn. Continuing from this
     * point preserves hidden thinking and sampled DSML bytes that are not
     * necessarily present in the client-visible replay. */
    int live_tokens;
    /* Optional rendered conversation text that the client is expected to replay.
     * Responses uses this because visible replay can omit hidden reasoning.
     * Anthropic currently uses only the call-id side of the state. */
    char *visible_text;
    size_t visible_len;
    /* Tool-call ids generated at the same live frontier. A following tool
     * result for these ids is a direct protocol continuation and should not
     * trigger prompt-prefix matching or checkpoint canonicalization. */
    stop_list call_ids;
} live_tool_state;

typedef struct {
    bool valid;
    /* Token frontier of the live sampled session.  The visible text below is
     * what clients will replay, but the payload at this frontier may also
     * contain hidden thinking tokens that are intentionally absent from that
     * visible replay. */
    int live_tokens;
    char *visible_text;
    size_t visible_len;
} visible_live_state;

struct server_slot {
    server *srv;
    int id;
    ds4_session *session;
    live_tool_state responses_live;
    live_tool_state anthropic_live;
    visible_live_state thinking_live;
    int continued_last_store_tokens;

    job *assigned;
    job *running;
    bool busy;
    bool prefill_waiting;

    bool decode_pending;
    bool decode_in_flight;
    bool decode_done;
    int decode_token;
    int decode_rc;
    char decode_err[160];
};

static bool id_list_contains(const stop_list *ids, const char *id);
static void id_list_push_unique(stop_list *ids, const char *id);

struct server {
    ds4_engine *engine;
    server_slot *slots;
    int slot_count;
    int ctx_size;
    bool batched_mode;
    pthread_t *slot_threads;
    pthread_t decode_thread;
    int default_tokens;
    kv_disk_cache kv;
    tool_memory tool_mem;
    bool disable_exact_dsml_tool_replay;
    bool enable_cors;
    pthread_mutex_t tool_mu;
    pthread_mutex_t kv_mu;
    pthread_mutex_t inference_mu;
    pthread_mutex_t model_mu;
    pthread_cond_t model_cv;
    bool model_busy;
    bool model_stopping;
    int decode_pending;
    int active_generations;
    int mixed_prefill_quantum;
    int last_prefill_slot;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_cond_t clients_cv;
    job *head;
    job *tail;
    bool stopping;
    int clients;
    uint64_t seq;
    FILE *trace;
    pthread_mutex_t trace_mu;
    uint64_t trace_seq;
};

/* Jobs are stack-owned by the client thread.  A resident-slot worker signals
 * completion after it has written the response, so request data and the socket
 * remain valid without heap-allocating per-request job objects. */
struct job {
    int fd;
    request req;
    bool done;
    bool cancelled;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    job *next;
};

static bool job_cancelled(void *ud) {
    job *j = ud;
    if (!j) return false;
    pthread_mutex_lock(&j->mu);
    bool cancelled = j->cancelled;
    pthread_mutex_unlock(&j->mu);
    return cancelled;
}

static void job_mark_cancelled(job *j) {
    if (!j) return;
    pthread_mutex_lock(&j->mu);
    j->cancelled = true;
    pthread_mutex_unlock(&j->mu);
}

static void job_complete(job *j) {
    pthread_mutex_lock(&j->mu);
    j->done = true;
    pthread_cond_signal(&j->cv);
    pthread_mutex_unlock(&j->mu);
}

static bool slot_job_cancelled(const server_slot *slot) {
    return slot && slot->running && job_cancelled(slot->running);
}

/* =========================================================================
 * Tool Call Text Memory.
 * =========================================================================
 *
 * The model speaks DSML, while OpenAI and Anthropic clients round-trip tool
 * calls as JSON.  Re-rendering that JSON is not always the same byte sequence:
 * clients may preserve, sort, or rebuild object keys differently.  Tool call
 * ids are the bridge between both worlds.  For every generated tool call we
 * remember the exact DSML block sampled by the model under a random id.  When
 * the client later sends the same id back in conversation history, we replay
 * the sampled DSML verbatim and keep the KV cache aligned with the live model
 * state.
 */

#define DS4_TOOL_MEMORY_DEFAULT_MAX_IDS 100000
#define DS4_TOOL_MEMORY_MAX_BYTES (512u * 1024u * 1024u)

static int tool_memory_max_entries(const tool_memory *m) {
    return m && m->max_entries > 0 ? m->max_entries : DS4_TOOL_MEMORY_DEFAULT_MAX_IDS;
}

static size_t tool_memory_max_bytes(const tool_memory *m) {
    return m && m->max_bytes > 0 ? m->max_bytes : DS4_TOOL_MEMORY_MAX_BYTES;
}

static void tool_memory_init_locked(tool_memory *m) {
    if (m->by_id && m->by_block) return;
    m->by_id = raxNew();
    m->by_block = raxNew();
    if (!m->by_id || !m->by_block) die("out of memory");
}

static void tool_memory_link_head(tool_memory *m, tool_memory_entry *e) {
    e->prev = NULL;
    e->next = m->head;
    if (m->head) m->head->prev = e;
    else m->tail = e;
    m->head = e;
}

static void tool_memory_unlink(tool_memory *m, tool_memory_entry *e) {
    if (e->prev) e->prev->next = e->next;
    else m->head = e->next;
    if (e->next) e->next->prev = e->prev;
    else m->tail = e->prev;
    e->prev = e->next = NULL;
}

static void tool_memory_touch(tool_memory *m, tool_memory_entry *e) {
    e->stamp = ++m->clock;
    if (m->head == e) return;
    tool_memory_unlink(m, e);
    tool_memory_link_head(m, e);
}

static void tool_block_unlink_entry(tool_memory_block *b, tool_memory_entry *e) {
    tool_memory_entry **p = &b->entries;
    while (*p) {
        if (*p == e) {
            *p = e->block_next;
            e->block_next = NULL;
            return;
        }
        p = &(*p)->block_next;
    }
}

static tool_memory_block *tool_memory_find_block_locked(tool_memory *m,
                                                        const char *dsml,
                                                        size_t len) {
    if (!m->by_block || !dsml || len == 0) return NULL;
    void *v = raxFind(m->by_block, (unsigned char *)dsml, len);
    return v == raxNotFound ? NULL : v;
}

static tool_memory_block *tool_memory_get_block_locked(tool_memory *m,
                                                       const char *dsml,
                                                       size_t len) {
    tool_memory_block *b = tool_memory_find_block_locked(m, dsml, len);
    if (b) return b;

    b = xmalloc(sizeof(*b));
    memset(b, 0, sizeof(*b));
    b->dsml = xstrndup(dsml, len);
    b->len = len;
    b->bytes = len + 1 + sizeof(*b);
    if (!raxInsert(m->by_block, (unsigned char *)b->dsml, b->len, b, NULL)) {
        free(b->dsml);
        free(b);
        die("out of memory");
    }
    m->bytes += b->bytes;
    return b;
}

static void tool_memory_release_block_locked(tool_memory *m, tool_memory_block *b) {
    if (!b) return;
    if (--b->refs > 0) return;
    if (m->by_block) {
        void *old = NULL;
        (void)raxRemove(m->by_block, (unsigned char *)b->dsml, b->len, &old);
    }
    if (m->bytes >= b->bytes) m->bytes -= b->bytes;
    else m->bytes = 0;
    free(b->dsml);
    free(b);
}

static void tool_memory_remove_entry_locked(tool_memory *m, tool_memory_entry *e) {
    if (!e) return;
    if (m->by_id && e->id) {
        void *old = NULL;
        (void)raxRemove(m->by_id, (unsigned char *)e->id, strlen(e->id), &old);
    }
    tool_memory_unlink(m, e);
    if (e->block) tool_block_unlink_entry(e->block, e);
    if (m->bytes >= e->bytes) m->bytes -= e->bytes;
    else m->bytes = 0;
    if (m->entries > 0) m->entries--;
    free(e->id);
    tool_memory_release_block_locked(m, e->block);
    free(e);
}

static void tool_memory_prune_locked(tool_memory *m) {
    while ((m->entries > tool_memory_max_entries(m) ||
            m->bytes > tool_memory_max_bytes(m)) && m->tail)
    {
        tool_memory_remove_entry_locked(m, m->tail);
    }
}

static tool_memory_entry *tool_memory_find_entry_locked(tool_memory *m,
                                                        const char *id) {
    if (!m->by_id || !id || !id[0]) return NULL;
    void *v = raxFind(m->by_id, (unsigned char *)id, strlen(id));
    return v == raxNotFound ? NULL : v;
}

static void tool_memory_put_locked(tool_memory *m, const char *id,
                                   const char *dsml, tool_memory_source source) {
    if (!id || !id[0] || !dsml || !dsml[0]) return;
    tool_memory_init_locked(m);

    size_t dsml_len = strlen(dsml);
    tool_memory_entry *old = tool_memory_find_entry_locked(m, id);
    if (old && old->block && old->block->len == dsml_len &&
        !memcmp(old->block->dsml, dsml, dsml_len))
    {
        if (source == TOOL_MEMORY_RAM) old->source = TOOL_MEMORY_RAM;
        tool_memory_touch(m, old);
        tool_memory_prune_locked(m);
        return;
    }
    if (old) tool_memory_remove_entry_locked(m, old);

    tool_memory_block *b = tool_memory_get_block_locked(m, dsml, dsml_len);
    tool_memory_entry *e = xmalloc(sizeof(*e));
    memset(e, 0, sizeof(*e));
    e->id = xstrdup(id);
    e->block = b;
    e->bytes = strlen(id) + 1 + sizeof(*e);
    e->stamp = ++m->clock;
    e->source = source;
    e->block_next = b->entries;
    b->entries = e;
    b->refs++;

    if (!raxInsert(m->by_id, (unsigned char *)e->id, strlen(e->id), e, NULL)) {
        tool_block_unlink_entry(b, e);
        free(e->id);
        free(e);
        tool_memory_release_block_locked(m, b);
        die("out of memory");
    }
    tool_memory_link_head(m, e);
    m->entries++;
    m->bytes += e->bytes;
    tool_memory_prune_locked(m);
}

static void tool_memory_free(tool_memory *m) {
    while (m->tail) tool_memory_remove_entry_locked(m, m->tail);
    if (m->by_id) raxFree(m->by_id);
    if (m->by_block) raxFree(m->by_block);
    memset(m, 0, sizeof(*m));
}

/* Single live protocol-tool state.
 *
 * This is not an implementation of durable remote conversation storage.  It is
 * only an in-memory binding from protocol tool-call IDs to the current sampled
 * KV frontier.  If it does not match, DS4 falls back to the same prefix and
 * disk-cache machinery used by chat/completions, or returns a clear error for
 * tool-result-only requests that have no replayable prefix. */
static void live_tool_state_clear_locked(live_tool_state *st) {
    if (!st) return;
    stop_list_clear(&st->call_ids);
    free(st->visible_text);
    st->visible_text = NULL;
    st->visible_len = 0;
    st->valid = false;
    st->live_tokens = 0;
}

static void live_tool_state_free(live_tool_state *st) {
    if (!st) return;
    live_tool_state_clear_locked(st);
    free(st->call_ids.v);
    memset(st, 0, sizeof(*st));
}

static void visible_live_clear_locked(visible_live_state *st) {
    if (!st) return;
    free(st->visible_text);
    st->visible_text = NULL;
    st->visible_len = 0;
    st->live_tokens = 0;
    st->valid = false;
}

static void visible_live_free(visible_live_state *st) {
    if (!st) return;
    visible_live_clear_locked(st);
    memset(st, 0, sizeof(*st));
}

static void thinking_live_clear(server *s, server_slot *slot) {
    if (!s || !slot) return;
    pthread_mutex_lock(&s->tool_mu);
    visible_live_clear_locked(&slot->thinking_live);
    pthread_mutex_unlock(&s->tool_mu);
}

static void thinking_live_remember(server *s, server_slot *slot,
                                   const char *visible_text) {
    if (!s || !slot || !visible_text || !visible_text[0]) return;
    pthread_mutex_lock(&s->tool_mu);
    visible_live_clear_locked(&slot->thinking_live);
    slot->thinking_live.visible_text = xstrdup(visible_text);
    slot->thinking_live.visible_len = strlen(visible_text);
    slot->thinking_live.live_tokens = ds4_session_pos(slot->session);
    slot->thinking_live.valid = true;
    pthread_mutex_unlock(&s->tool_mu);
}

static void responses_live_remember(server *s, server_slot *slot,
                                    const char *visible_text,
                                    const tool_calls *calls) {
    if (!s || !slot || !visible_text || !visible_text[0]) return;
    pthread_mutex_lock(&s->tool_mu);
    live_tool_state_clear_locked(&slot->responses_live);
    slot->responses_live.visible_text = xstrdup(visible_text);
    slot->responses_live.visible_len = strlen(visible_text);
    if (calls) {
        for (int i = 0; i < calls->len; i++) {
            id_list_push_unique(&slot->responses_live.call_ids, calls->v[i].id);
        }
    }
    slot->responses_live.live_tokens = ds4_session_pos(slot->session);
    slot->responses_live.valid = true;
    pthread_mutex_unlock(&s->tool_mu);
}

static void anthropic_live_remember(server *s, server_slot *slot,
                                    const tool_calls *calls) {
    if (!s || !slot || !calls || calls->len == 0) return;
    pthread_mutex_lock(&s->tool_mu);
    live_tool_state_clear_locked(&slot->anthropic_live);
    for (int i = 0; i < calls->len; i++) {
        id_list_push_unique(&slot->anthropic_live.call_ids, calls->v[i].id);
    }
    slot->anthropic_live.live_tokens = ds4_session_pos(slot->session);
    slot->anthropic_live.valid = slot->anthropic_live.call_ids.len > 0;
    pthread_mutex_unlock(&s->tool_mu);
}

static void responses_live_clear(server *s, server_slot *slot) {
    if (!s || !slot) return;
    pthread_mutex_lock(&s->tool_mu);
    live_tool_state_clear_locked(&slot->responses_live);
    pthread_mutex_unlock(&s->tool_mu);
}

static void anthropic_live_clear(server *s, server_slot *slot) {
    if (!s || !slot) return;
    pthread_mutex_lock(&s->tool_mu);
    live_tool_state_clear_locked(&slot->anthropic_live);
    pthread_mutex_unlock(&s->tool_mu);
}

static void request_live_state_clear(server *s, server_slot *slot) {
    responses_live_clear(s, slot);
    anthropic_live_clear(s, slot);
    thinking_live_clear(s, slot);
}

static bool responses_live_has_call_id(server *s, const char *id) {
    if (!s || !id || !id[0]) return false;
    pthread_mutex_lock(&s->tool_mu);
    bool found = false;
    for (int i = 0; i < s->slot_count && !found; i++) {
        found = s->slots[i].responses_live.valid &&
                id_list_contains(&s->slots[i].responses_live.call_ids, id);
    }
    pthread_mutex_unlock(&s->tool_mu);
    return found;
}

static bool anthropic_live_has_call_id(server *s, const char *id) {
    if (!s || !id || !id[0]) return false;
    pthread_mutex_lock(&s->tool_mu);
    bool found = false;
    for (int i = 0; i < s->slot_count && !found; i++) {
        found = s->slots[i].anthropic_live.valid &&
                id_list_contains(&s->slots[i].anthropic_live.call_ids, id);
    }
    pthread_mutex_unlock(&s->tool_mu);
    return found;
}

static bool responses_live_matches_request(server *s, server_slot *slot,
                                           const stop_list *ids,
                                           int live_tokens) {
    if (!s || !slot || !ids || ids->len == 0) return false;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = slot->responses_live.valid &&
              slot->responses_live.live_tokens == live_tokens &&
              slot->responses_live.call_ids.len == ids->len;
    for (int i = 0; ok && i < ids->len; i++) {
        ok = id_list_contains(&slot->responses_live.call_ids, ids->v[i]);
    }
    pthread_mutex_unlock(&s->tool_mu);
    return ok;
}

static bool anthropic_live_matches_request(server *s, server_slot *slot,
                                           const stop_list *ids,
                                           int live_tokens) {
    if (!s || !slot || !ids || ids->len == 0) return false;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = slot->anthropic_live.valid &&
              slot->anthropic_live.live_tokens == live_tokens &&
              slot->anthropic_live.call_ids.len == ids->len;
    for (int i = 0; ok && i < ids->len; i++) {
        ok = id_list_contains(&slot->anthropic_live.call_ids, ids->v[i]);
    }
    pthread_mutex_unlock(&s->tool_mu);
    return ok;
}

static bool tool_memory_has_id(server *s, const char *id) {
    if (!s || s->disable_exact_dsml_tool_replay || !id || !id[0]) return false;
    pthread_mutex_lock(&s->tool_mu);
    bool found = tool_memory_find_entry_locked(&s->tool_mem, id) != NULL;
    pthread_mutex_unlock(&s->tool_mu);
    return found;
}

static const char *tool_memory_lookup_locked(tool_memory *m, const char *id,
                                             tool_memory_source *source,
                                             tool_memory_block **block) {
    tool_memory_entry *e = tool_memory_find_entry_locked(m, id);
    if (!e || !e->block) return NULL;
    tool_memory_touch(m, e);
    if (source) *source = e->source;
    if (block) *block = e->block;
    return e->block->dsml;
}

static void tool_memory_remember(server *s, const tool_calls *calls) {
    if (!s || s->disable_exact_dsml_tool_replay ||
        !calls || !calls->raw_tool_text || !calls->raw_tool_text[0]) return;
    pthread_mutex_lock(&s->tool_mu);
    for (int i = 0; i < calls->len; i++) {
        tool_memory_put_locked(&s->tool_mem, calls->v[i].id, calls->raw_tool_text,
                               TOOL_MEMORY_RAM);
    }
    pthread_mutex_unlock(&s->tool_mu);
}

static void tool_memory_put_source(server *s, const char *id, const char *dsml,
                                   tool_memory_source source) {
    if (!s || s->disable_exact_dsml_tool_replay ||
        !id || !id[0] || !dsml || !dsml[0]) return;
    pthread_mutex_lock(&s->tool_mu);
    tool_memory_put_locked(&s->tool_mem, id, dsml, source);
    pthread_mutex_unlock(&s->tool_mu);
}

#ifdef DS4_SERVER_TEST
static void tool_memory_put(server *s, const char *id, const char *dsml) {
    tool_memory_put_source(s, id, dsml, TOOL_MEMORY_RAM);
}
#endif

static void tool_memory_attach_to_messages(server *s, chat_msgs *msgs,
                                           tool_replay_stats *stats) {
    if (!msgs) return;
    if (!s || s->disable_exact_dsml_tool_replay) {
        if (stats) {
            for (int i = 0; i < msgs->len; i++) {
                tool_calls *calls = &msgs->v[i].calls;
                if (calls->len == 0 || calls->raw_tool_text) continue;
                stats->canonical++;
                stats->missing_ids += calls->len;
            }
        }
        return;
    }
    pthread_mutex_lock(&s->tool_mu);
    for (int i = 0; i < msgs->len; i++) {
        tool_calls *calls = &msgs->v[i].calls;
        if (calls->len == 0 || calls->raw_tool_text) continue;
        tool_memory_block *matched = NULL;
        tool_memory_source matched_source = TOOL_MEMORY_DISK;
        bool exact = true;
        int missing = 0;
        for (int j = 0; j < calls->len; j++) {
            tool_memory_source source = TOOL_MEMORY_DISK;
            tool_memory_block *block = NULL;
            const char *dsml =
                tool_memory_lookup_locked(&s->tool_mem, calls->v[j].id,
                                          &source, &block);
            if (!dsml) {
                exact = false;
                missing++;
                continue;
            }
            if (!matched) {
                matched = block;
                matched_source = source;
            } else if (matched != block) {
                exact = false;
            }
            if (source == TOOL_MEMORY_RAM) matched_source = TOOL_MEMORY_RAM;
        }
        if (exact && matched) {
            calls->raw_tool_text = xstrdup(matched->dsml);
            if (stats) {
                if (matched_source == TOOL_MEMORY_RAM) stats->mem++;
                else stats->disk++;
            }
        } else if (stats) {
            stats->canonical++;
            stats->missing_ids += missing;
        }
    }
    pthread_mutex_unlock(&s->tool_mu);
}

static bool tool_calls_contains_id(const tool_calls *calls, const char *id, int upto) {
    if (!calls || !id || !id[0]) return false;
    if (upto > calls->len) upto = calls->len;
    for (int i = 0; i < upto; i++) {
        if (calls->v[i].id && !strcmp(calls->v[i].id, id)) return true;
    }
    return false;
}

static void assign_tool_call_ids(server *s, tool_calls *calls, api_style api) {
    if (!calls) return;
    for (int i = 0; i < calls->len; i++) {
        if (calls->v[i].id && calls->v[i].id[0]) continue;
        char id[64];
        for (;;) {
            random_tool_id(id, sizeof(id), api);
            if (!tool_calls_contains_id(calls, id, i) && !tool_memory_has_id(s, id)) break;
        }
        calls->v[i].id = xstrdup(id);
    }
}

static void apply_openai_stream_tool_ids(tool_calls *calls,
                                         const openai_stream *st) {
    if (!calls || !st) return;
    int n = calls->len < st->tool.ids_cap ? calls->len : st->tool.ids_cap;
    for (int i = 0; i < n; i++) {
        if (calls->v[i].id && calls->v[i].id[0]) continue;
        if (st->tool.ids[i] && st->tool.ids[i][0]) calls->v[i].id = xstrdup(st->tool.ids[i]);
    }
}

static void apply_anthropic_stream_tool_ids(tool_calls *calls,
                                            const anthropic_stream *st) {
    if (!calls || !st) return;
    /* The SSE stream may have exposed tool ids before final DSML parsing.  The
     * parsed calls must inherit those ids before assign_tool_call_ids() and
     * tool_memory_remember(), otherwise the client returns a tool_result for an
     * id that the continuation fast path does not know. */
    int n = calls->len < st->tool.ids_cap ? calls->len : st->tool.ids_cap;
    for (int i = 0; i < n; i++) {
        if (calls->v[i].id && calls->v[i].id[0]) continue;
        if (st->tool.ids[i] && st->tool.ids[i][0]) calls->v[i].id = xstrdup(st->tool.ids[i]);
    }
}

/* =========================================================================
 * KV Cache.
 * =========================================================================
 *
 * The server has one live Metal session.  We persist reusable DS4 session
 * snapshots when a cold prompt reaches a useful prefix, when a long continued
 * conversation has grown far enough, and when a request evicts the live session.
 * The cache key is the SHA1 of the rendered byte prefix.  The payload still
 * stores exact token IDs and graph state; the filename only selects a checkpoint
 * whose decoded transcript bytes are a prefix of the next rendered request.
 *
 * Files are loaded with plain read/write I/O into the existing graph tensors;
 * mmap is deliberately avoided here so cache restore cannot add more VM
 * mappings to a process that already maps a very large GGUF.
 *
 * Stores are created only when the live graph is already at the checkpoint we
 * want to persist.  For long cold prompts this means prefill reaches the stable
 * boundary first, writes that prefix, and then continues with the suffix.  We
 * never roll the session backward just to build a disk cache entry: that would
 * turn cache population into a second hidden prefill.
 *
 * File layout:
 *
 *   "KVC" version
 *   quant bits, save reason, token count, hit count, context size
 *   creation time, last-used time, payload byte count
 *   rendered text byte count + rendered text for human inspection
 *   DS4 engine payload written by ds4_session_save_payload()
 *   optional tool-id map section
 *
 * The filename is SHA1(cache text bytes), not SHA1(token ids).  For ordinary
 * checkpoints the cache text is the rendered token prefix.  For live hidden
 * state it can instead be the client-visible transcript: the payload still
 * contains sampled reasoning KV, but the lookup key must be what the client can
 * replay after a process restart or session switch.
 *
 * The optional tool-id map is not part of model state, but it is needed to
 * render future client JSON back to the exact DSML sampled by the model.  We
 * persist only mappings whose DSML block appears in the saved cache text.
 */

#define KV_CACHE_FIXED_HEADER DS4_KVSTORE_FIXED_HEADER
#define KV_CACHE_HIT_HALF_LIFE_SECONDS DS4_KVSTORE_HIT_HALF_LIFE_SECONDS
#define KV_EXT_TOOL_MAP DS4_KVSTORE_EXT_TOOL_MAP
#define KV_EXT_RESPONSES_VISIBLE DS4_KVSTORE_EXT_RESPONSES_VISIBLE
#define KV_EXT_THINKING_VISIBLE DS4_KVSTORE_EXT_THINKING_VISIBLE
#define KV_TOOL_MAP_MAGIC0 'K'
#define KV_TOOL_MAP_MAGIC1 'T'
#define KV_TOOL_MAP_MAGIC2 'M'
#define KV_TOOL_MAP_VERSION 1u
#define KV_TOOL_MAP_HEADER 8u

typedef enum {
    KV_REASON_UNKNOWN   = DS4_KVSTORE_REASON_UNKNOWN,
    KV_REASON_COLD      = DS4_KVSTORE_REASON_COLD,
    KV_REASON_CONTINUED = DS4_KVSTORE_REASON_CONTINUED,
    KV_REASON_EVICT     = DS4_KVSTORE_REASON_EVICT,
    KV_REASON_SHUTDOWN  = DS4_KVSTORE_REASON_SHUTDOWN,
} kv_cache_reason;


static kv_cache_options kv_cache_default_options(void) {
    return ds4_kvstore_default_options();
}

static void le_put32(uint8_t *p, uint32_t v) {
    ds4_kvstore_le_put32(p, v);
}


static uint32_t le_get32(const uint8_t *p) {
    return ds4_kvstore_le_get32(p);
}


#ifdef DS4_SERVER_TEST
static void sha1_bytes_hex(const void *ptr, size_t len, char out[41]) {
    ds4_kvstore_sha1_bytes_hex(ptr, len, out);
}
#endif

static bool id_list_contains(const stop_list *ids, const char *id) {
    if (!ids || !id || !id[0]) return false;
    for (int i = 0; i < ids->len; i++) {
        if (ids->v[i] && !strcmp(ids->v[i], id)) return true;
    }
    return false;
}

static void id_list_push_unique(stop_list *ids, const char *id) {
    if (!ids || !id || !id[0] || id_list_contains(ids, id)) return;
    stop_list_push(ids, xstrdup(id));
}

static void id_list_free(stop_list *ids) {
    stop_list_clear(ids);
    free(ids->v);
    memset(ids, 0, sizeof(*ids));
}

static void collect_tool_call_ids(const chat_msgs *msgs, stop_list *ids) {
    if (!msgs || !ids) return;
    for (int i = 0; i < msgs->len; i++) {
        id_list_push_unique(ids, msgs->v[i].tool_call_id);
        for (int j = 0; j < msgs->v[i].tool_call_ids_len; j++) {
            id_list_push_unique(ids, msgs->v[i].tool_call_ids[j]);
        }
        const tool_calls *calls = &msgs->v[i].calls;
        for (int j = 0; j < calls->len; j++) {
            id_list_push_unique(ids, calls->v[j].id);
        }
    }
}

static bool sha_hex_name(const char *name, char sha[41]) {
    return ds4_kvstore_sha_hex_name(name, sha);
}

static char *path_join(const char *dir, const char *name) {
    return ds4_kvstore_path_join(dir, name);
}






static const char *find_next_dsml_tool_block(const char *p, const char **end_out) {
    struct block_form {
        const char *start;
        const char *end;
    } forms[] = {
        {"\n\n" DS4_TOOL_CALLS_START, DS4_TOOL_CALLS_END},
        {DS4_TOOL_CALLS_START, DS4_TOOL_CALLS_END},
        {"\n\n" DS4_TOOL_CALLS_START_SHORT, DS4_TOOL_CALLS_END_SHORT},
        {DS4_TOOL_CALLS_START_SHORT, DS4_TOOL_CALLS_END_SHORT},
        {"\n\n<tool_calls>", "</tool_calls>"},
        {"<tool_calls>", "</tool_calls>"},
    };

    const char *best = NULL;
    const char *best_end = NULL;
    for (size_t i = 0; i < sizeof(forms) / sizeof(forms[0]); i++) {
        const char *s = strstr(p, forms[i].start);
        if (!s || (best && s >= best)) continue;
        const char *e = strstr(s, forms[i].end);
        if (!e) continue;
        best = s;
        best_end = e + strlen(forms[i].end);
    }
    if (end_out) *end_out = best_end;
    return best;
}


static bool kv_tool_map_measure_locked(server *s, const char *text,
                                       uint32_t *count_out,
                                       uint64_t *bytes_out) {
    uint32_t count = 0;
    uint64_t bytes = KV_TOOL_MAP_HEADER;
    uint64_t scan = ++s->tool_mem.scan_clock;
    const char *p = text;
    for (;;) {
        const char *end = NULL;
        const char *start = find_next_dsml_tool_block(p, &end);
        if (!start || !end) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b && b->seen != scan) {
            b->seen = scan;
            for (tool_memory_entry *e = b->entries; e; e = e->block_next) {
                size_t id_len = strlen(e->id);
                size_t dsml_len = b->len;
                if (id_len > UINT32_MAX || dsml_len > UINT32_MAX) continue;
                if (count == UINT32_MAX) return false;
                if (UINT64_MAX - bytes < 8u ||
                    UINT64_MAX - bytes - 8u < (uint64_t)id_len ||
                    UINT64_MAX - bytes - 8u - (uint64_t)id_len < (uint64_t)dsml_len)
                    return false;
                count++;
                bytes += 8u + (uint64_t)id_len + (uint64_t)dsml_len;
            }
        }
        p = end;
    }
    if (count == 0) bytes = 0;
    if (count_out) *count_out = count;
    if (bytes_out) *bytes_out = bytes;
    return true;
}

static bool kv_tool_map_serialized_size(server *s, const char *text,
                                        uint64_t *bytes_out) {
    if (bytes_out) *bytes_out = 0;
    if (!s || s->disable_exact_dsml_tool_replay || !text || !text[0]) return true;

    pthread_mutex_lock(&s->tool_mu);
    bool ok = kv_tool_map_measure_locked(s, text, NULL, bytes_out);
    pthread_mutex_unlock(&s->tool_mu);
    return ok;
}

static bool kv_tool_map_write(server *s, FILE *fp, const char *text,
                              uint64_t *written_bytes) {
    if (written_bytes) *written_bytes = 0;
    if (!s || s->disable_exact_dsml_tool_replay || !fp || !text || !text[0]) return true;

    pthread_mutex_lock(&s->tool_mu);
    uint32_t count = 0;
    uint64_t bytes = 0;
    bool ok = kv_tool_map_measure_locked(s, text, &count, &bytes);
    if (!ok) {
        pthread_mutex_unlock(&s->tool_mu);
        return false;
    }
    if (count == 0) {
        pthread_mutex_unlock(&s->tool_mu);
        return true;
    }

    uint8_t h[KV_TOOL_MAP_HEADER];
    h[0] = KV_TOOL_MAP_MAGIC0;
    h[1] = KV_TOOL_MAP_MAGIC1;
    h[2] = KV_TOOL_MAP_MAGIC2;
    h[3] = KV_TOOL_MAP_VERSION;
    le_put32(h + 4, count);
    ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h);

    uint64_t scan = ++s->tool_mem.scan_clock;
    const char *p = text;
    for (;;) {
        const char *end = NULL;
        const char *start = find_next_dsml_tool_block(p, &end);
        if (!start || !end || !ok) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b && b->seen != scan) {
            b->seen = scan;
            for (tool_memory_entry *e = b->entries; ok && e; e = e->block_next) {
                size_t id_len = strlen(e->id);
                size_t dsml_len = b->len;
                if (id_len > UINT32_MAX || dsml_len > UINT32_MAX) continue;
                uint8_t lens[8];
                le_put32(lens, (uint32_t)id_len);
                le_put32(lens + 4, (uint32_t)dsml_len);
                ok = fwrite(lens, 1, sizeof(lens), fp) == sizeof(lens) &&
                     fwrite(e->id, 1, id_len, fp) == id_len &&
                     fwrite(b->dsml, 1, dsml_len, fp) == dsml_len;
            }
        }
        p = end;
    }
    pthread_mutex_unlock(&s->tool_mu);

    if (ok && written_bytes) *written_bytes = bytes;
    return ok;
}

static int kv_tool_map_load_from_pos(server *s, FILE *fp, const stop_list *wanted) {
    if (!s || s->disable_exact_dsml_tool_replay || !fp) return 0;
    uint8_t h[KV_TOOL_MAP_HEADER];
    size_t n = fread(h, 1, sizeof(h), fp);
    if (n == 0 && feof(fp)) return 0;
    if (n != sizeof(h)) return 0;
    if (h[0] != KV_TOOL_MAP_MAGIC0 || h[1] != KV_TOOL_MAP_MAGIC1 ||
        h[2] != KV_TOOL_MAP_MAGIC2 || h[3] != KV_TOOL_MAP_VERSION) return 0;

    uint32_t count = le_get32(h + 4);
    if ((uint64_t)count > (uint64_t)tool_memory_max_entries(&s->tool_mem) * 4u) return 0;
    int loaded = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint8_t lens[8];
        if (fread(lens, 1, sizeof(lens), fp) != sizeof(lens)) return loaded;
        uint32_t id_len = le_get32(lens);
        uint32_t dsml_len = le_get32(lens + 4);
        if (id_len == 0 || id_len > 256 || dsml_len == 0 ||
            dsml_len > DS4_TOOL_MEMORY_MAX_BYTES) return loaded;
        char *id = xmalloc((size_t)id_len + 1);
        char *dsml = xmalloc((size_t)dsml_len + 1);
        bool ok = fread(id, 1, id_len, fp) == id_len &&
                  fread(dsml, 1, dsml_len, fp) == dsml_len;
        id[id_len] = '\0';
        dsml[dsml_len] = '\0';
        if (ok && (!wanted || id_list_contains(wanted, id))) {
            tool_memory_put_source(s, id, dsml, TOOL_MEMORY_DISK);
            loaded++;
        }
        free(id);
        free(dsml);
        if (!ok) return loaded;
    }
    return loaded;
}

#ifdef DS4_SERVER_TEST
static void kv_fill_header(uint8_t h[KV_CACHE_FIXED_HEADER], uint8_t quant_bits,
                           uint8_t reason, uint8_t ext_flags,
                           uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                           uint64_t created_at, uint64_t last_used,
                           uint64_t payload_bytes) {
    ds4_kvstore_fill_header(h, 0, quant_bits, reason, ext_flags, tokens, hits,
                            ctx_size, created_at, last_used, payload_bytes);
}
#endif

static bool kv_read_header(FILE *fp, kv_entry *e, uint32_t *text_bytes) {
    return ds4_kvstore_read_header(fp, e, text_bytes);
}




static void kv_cache_restore_tool_memory_for_messages(server *s, const chat_msgs *msgs) {
    if (!s || s->disable_exact_dsml_tool_replay || !s->kv.enabled || !msgs) return;
    stop_list wanted = {0};
    collect_tool_call_ids(msgs, &wanted);
    if (wanted.len == 0) return;
    /* Tool replay payloads are stored next to KV checkpoints; keep them model
     * scoped too, since token positions and graph state are not portable across
     * Flash/Pro shapes even when the rendered chat text is identical. */
    uint8_t model_id = s->engine ? (uint8_t)ds4_engine_model_id(s->engine) : 0;

    DIR *d = opendir(s->kv.dir);
    if (!d) {
        id_list_free(&wanted);
        return;
    }
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!sha_hex_name(de->d_name, sha)) continue;
        (void)sha;
        char *path = path_join(s->kv.dir, de->d_name);
        FILE *fp = fopen(path, "rb");
        free(path);
        if (!fp) continue;

        kv_entry hdr = {0};
        uint32_t text_bytes = 0;
        bool ok = kv_read_header(fp, &hdr, &text_bytes);
        uint64_t skip = (uint64_t)text_bytes + hdr.payload_bytes;
        if (ok && hdr.model_id == model_id && (hdr.ext_flags & KV_EXT_TOOL_MAP) &&
            skip <= (uint64_t)INT64_MAX &&
            fseeko(fp, (off_t)skip, SEEK_CUR) == 0)
        {
            kv_tool_map_load_from_pos(s, fp, &wanted);
        }
        fclose(fp);
    }
    closedir(d);
    id_list_free(&wanted);
}

#ifdef DS4_SERVER_TEST
static double kv_entry_eviction_score(const kv_entry *e, const ds4_tokens *live,
                                      uint64_t now,
                                      const ds4_kvstore_eviction_context *incoming) {
    return ds4_kvstore_entry_eviction_score(e, live, now, incoming);
}
#endif

#ifdef DS4_SERVER_TEST
static void kv_cache_evict(kv_disk_cache *kc, const ds4_tokens *live,
                           uint64_t extra_bytes,
                           const ds4_kvstore_eviction_context *incoming) {
    ds4_kvstore_evict(kc, live, extra_bytes, incoming);
}
#endif

static void kv_cache_log_cb(void *ud, ds4_kvstore_log_type type, const char *msg) {
    (void)ud;
    ds4_log_type stype = DS4_LOG_KVCACHE;
    if (type == DS4_KVSTORE_LOG_DEFAULT) stype = DS4_LOG_DEFAULT;
    else if (type == DS4_KVSTORE_LOG_WARNING) stype = DS4_LOG_WARNING;
    server_log(stype, "%s", msg);
}

static bool kv_cache_open(kv_disk_cache *kc, const char *dir, uint64_t budget_mb,
                          bool reject_different_quant, kv_cache_options opt) {
    return ds4_kvstore_open(kc, dir, budget_mb, reject_different_quant, opt,
                            "ds4-server", kv_cache_log_cb, NULL);
}

static void kv_cache_close(kv_disk_cache *kc) {
    ds4_kvstore_close(kc);
}

static char *render_tokens_text(ds4_engine *engine, const ds4_tokens *tokens, size_t *out_len) {
    return ds4_kvstore_render_tokens_text(engine, tokens, out_len);
}

static bool byte_prefix_match(const char *text, size_t text_len,
                              const char *prefix, size_t prefix_len) {
    return ds4_kvstore_byte_prefix_match(text, text_len, prefix, prefix_len);
}


static void tokens_copy_prefix(ds4_tokens *dst, const ds4_tokens *src, int n) {
    ds4_kvstore_tokens_copy_prefix(dst, src, n);
}


static void build_prompt_from_exact_prefix_and_text_suffix(
        ds4_engine *engine,
        const ds4_tokens *exact_prefix,
        const char *suffix_text,
        ds4_tokens *out)
{
    ds4_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        engine, exact_prefix, suffix_text, out);
}

static int kv_cache_store_len(const kv_disk_cache *kc, int tokens) {
    return ds4_kvstore_store_len(kc, tokens);
}

static int kv_cache_chat_anchor_pos(const kv_disk_cache *kc,
                                    const ds4_tokens *prompt,
                                    int user_token_id,
                                    int assistant_token_id) {
    return ds4_kvstore_chat_anchor_pos(kc, prompt, user_token_id, assistant_token_id);
}


static int kv_cache_continued_store_target(const kv_disk_cache *kc, int live_tokens) {
    return ds4_kvstore_continued_store_target(kc, live_tokens);
}

/* A same-text-prefix file can be reused by a larger context, but not by a
 * smaller one: the payload was validated against the context capacity recorded
 * in the file.  If the existing file cannot be used by this server, replace it
 * so this context can still populate its own cache. */



#ifdef DS4_SERVER_TEST
static bool kv_cache_file_size_fits(const kv_disk_cache *kc,
                                    uint64_t text_bytes,
                                    uint64_t payload_bytes,
                                    uint64_t tool_map_bytes,
                                    uint64_t *file_bytes_out,
                                    uint64_t *required_bytes_out) {
    return ds4_kvstore_file_size_fits(kc, text_bytes, payload_bytes,
                                      tool_map_bytes, file_bytes_out,
                                      required_bytes_out);
}
#endif



static bool kv_cache_tool_map_size_cb(void *ud, const char *text,
                                      uint64_t *bytes_out) {
    return kv_tool_map_serialized_size((server *)ud, text, bytes_out);
}

static bool kv_cache_tool_map_write_cb(void *ud, FILE *fp, const char *text,
                                       uint64_t *written_bytes) {
    return kv_tool_map_write((server *)ud, fp, text, written_bytes);
}

static int kv_cache_tool_map_load_cb(void *ud, FILE *fp, const void *wanted) {
    return kv_tool_map_load_from_pos((server *)ud, fp, (const stop_list *)wanted);
}

static ds4_kvstore_trailer_hooks kv_cache_tool_map_hooks(server *s,
                                                         const stop_list *wanted) {
    return (ds4_kvstore_trailer_hooks){
        .ud = s,
        .ext_flag = KV_EXT_TOOL_MAP,
        .serialized_size = kv_cache_tool_map_size_cb,
        .write = kv_cache_tool_map_write_cb,
        .load = kv_cache_tool_map_load_cb,
        .load_wanted = wanted,
    };
}

static bool kv_cache_store_live_prefix_text(server *s, server_slot *slot,
                                            const ds4_tokens *tokens,
                                            int store_len, const char *reason,
                                            const char *cache_text_override,
                                            uint8_t cache_text_ext,
                                            const char *cache_text_key) {
    if (!s || !slot) return false;
    char err[160] = {0};
    ds4_kvstore_trailer_hooks hooks = kv_cache_tool_map_hooks(s, NULL);
    pthread_mutex_lock(&s->inference_mu);
    pthread_mutex_lock(&s->kv_mu);
    bool ok = ds4_kvstore_store_live_prefix_text(&s->kv, s->engine,
                                                  slot->session,
                                                  tokens, store_len, reason,
                                                  cache_text_override,
                                                  cache_text_ext,
                                                  cache_text_key,
                                                  &hooks, err, sizeof(err));
    pthread_mutex_unlock(&s->kv_mu);
    pthread_mutex_unlock(&s->inference_mu);
    return ok;
}

static bool kv_cache_store_live_prefix(server *s, server_slot *slot,
                                       const ds4_tokens *tokens,
                                       int store_len, const char *reason) {
    return kv_cache_store_live_prefix_text(s, slot, tokens, store_len, reason,
                                           NULL, 0, NULL);
}

static void kv_cache_store_current(server *s, server_slot *slot,
                                   const char *reason) {
    if (!s || !slot) return;
    const ds4_tokens *tokens = ds4_session_tokens(slot->session);
    if (!tokens) return;

    char *visible_text = NULL;
    uint8_t visible_ext = 0;
    const char *visible_key = NULL;
    pthread_mutex_lock(&s->tool_mu);
    if (slot->responses_live.valid &&
        slot->responses_live.live_tokens == tokens->len &&
        slot->responses_live.visible_text &&
        slot->responses_live.visible_text[0])
    {
        visible_text = xstrdup(slot->responses_live.visible_text);
        visible_ext = KV_EXT_RESPONSES_VISIBLE;
        visible_key = "responses-visible";
    } else if (slot->thinking_live.valid &&
               slot->thinking_live.live_tokens == tokens->len &&
               slot->thinking_live.visible_text &&
               slot->thinking_live.visible_text[0])
    {
        visible_text = xstrdup(slot->thinking_live.visible_text);
        visible_ext = KV_EXT_THINKING_VISIBLE;
        visible_key = "thinking-visible";
    }
    pthread_mutex_unlock(&s->tool_mu);

    /* A visible live checkpoint can contain hidden reasoning that the client
     * intentionally does not replay.  For disk recovery after a session switch,
     * key that payload by the visible protocol transcript, not by rendering the
     * hidden sampled tokens.  On load, DS4 restores the hidden KV payload and
     * tokenizes only the visible suffix that follows this key. */
    if (visible_text) {
        kv_cache_store_live_prefix_text(s, slot, tokens, tokens->len, reason,
                                        visible_text, visible_ext, visible_key);
        free(visible_text);
    } else {
        kv_cache_store_live_prefix(s, slot, tokens, tokens->len, reason);
    }
}

#ifdef DS4_SERVER_TEST
static int kv_cache_suppress_continued_store(kv_disk_cache *kc, int tokens) {
    return ds4_kvstore_suppress_continued_store(kc, tokens);
}

static void kv_cache_restore_suppressed_continued(kv_disk_cache *kc,
                                                  int old_tokens,
                                                  int suppressed_tokens) {
    ds4_kvstore_restore_suppressed_continued(kc, old_tokens, suppressed_tokens);
}
#endif

static int kv_cache_slot_continued_target(server *s, server_slot *slot,
                                          int live_tokens) {
    if (!s || !slot) return 0;
    kv_disk_cache view = s->kv;
    view.continued_last_store_tokens = slot->continued_last_store_tokens;
    return kv_cache_continued_store_target(&view, live_tokens);
}

static void kv_cache_slot_note_store(server_slot *slot, int tokens) {
    if (slot && tokens > slot->continued_last_store_tokens) {
        slot->continued_last_store_tokens = tokens;
    }
}

static int kv_cache_slot_suppress_continued(server *s, server_slot *slot,
                                             int tokens) {
    if (kv_cache_slot_continued_target(s, slot, tokens) != tokens) return -1;
    int old = slot->continued_last_store_tokens;
    kv_cache_slot_note_store(slot, tokens);
    return old;
}

static void kv_cache_slot_restore_suppressed(server_slot *slot,
                                              int old_tokens,
                                              int suppressed_tokens) {
    if (slot && old_tokens >= 0 &&
        slot->continued_last_store_tokens == suppressed_tokens) {
        slot->continued_last_store_tokens = old_tokens;
    }
}

static void kv_cache_discard_failed_disk_entry(server *s, server_slot *slot,
                                                const char *path) {
    if (!s || !slot || !path) return;
    pthread_mutex_lock(&s->kv_mu);
    if (unlink(path) == 0) {
        server_log(DS4_LOG_KVCACHE,
                   "ds4-server: kv cache discarded reason=prefill-failed file=%s",
                   path);
    } else if (errno != ENOENT) {
        server_log(DS4_LOG_WARNING,
                   "ds4-server: kv cache failed to discard prefill-failed file=%s: %s",
                   path, strerror(errno));
    }
    pthread_mutex_unlock(&s->kv_mu);
    slot->continued_last_store_tokens = 0;
    pthread_mutex_lock(&s->inference_mu);
    ds4_session_invalidate(slot->session);
    pthread_mutex_unlock(&s->inference_mu);
}

static void kv_cache_maybe_store_continued(server *s, server_slot *slot) {
    if (!s || !slot) return;
    kv_disk_cache *kc = &s->kv;
    const ds4_tokens *tokens = ds4_session_tokens(slot->session);
    if (!tokens) return;
    const int target = kv_cache_slot_continued_target(s, slot, tokens->len);
    if (target == 0) return;
    if (kv_cache_store_live_prefix(s, slot, tokens, target, "continued")) {
        (void)kc;
        kv_cache_slot_note_store(slot, target);
    }
}

#ifdef DS4_SERVER_TEST
static int kv_cache_find_text_prefix(kv_disk_cache *kc, const char *prompt_text,
                                     int quant_bits, int ctx_size) {
    return ds4_kvstore_find_text_prefix(kc, prompt_text, 0, quant_bits, ctx_size);
}
#endif

static int kv_cache_try_load_text(server *s, server_slot *slot,
                                  const char *prompt_text,
                                  ds4_tokens *effective_prompt,
                                  char **loaded_path_out,
                                  uint8_t *loaded_ext_flags_out,
                                  bool responses_protocol) {
    if (!s || !slot) return 0;
    if (loaded_path_out) *loaded_path_out = NULL;
    if (loaded_ext_flags_out) *loaded_ext_flags_out = 0;
    ds4_kvstore_load_result lr = {0};
    ds4_kvstore_trailer_hooks hooks = kv_cache_tool_map_hooks(s, NULL);
    pthread_mutex_lock(&s->inference_mu);
    pthread_mutex_lock(&s->kv_mu);
    int loaded = ds4_kvstore_try_load_text(&s->kv, s->engine, slot->session,
                                           prompt_text, effective_prompt, &lr,
                                           &hooks, responses_protocol);
    pthread_mutex_unlock(&s->kv_mu);
    pthread_mutex_unlock(&s->inference_mu);
    if (loaded > 0) {
        if (loaded_path_out && lr.path) *loaded_path_out = xstrdup(lr.path);
        if (loaded_ext_flags_out) *loaded_ext_flags_out = lr.ext_flags;
    }
    ds4_kvstore_load_result_free(&lr);
    return loaded;
}

static int kv_cache_try_load(server *s, server_slot *slot, const request *req,
                             ds4_tokens *effective_prompt,
                             char **loaded_path_out,
                             uint8_t *loaded_ext_flags_out) {
    return kv_cache_try_load_text(s, slot, req ? req->prompt_text : NULL,
                                  effective_prompt,
                                  loaded_path_out,
                                  loaded_ext_flags_out,
                                  req && req->api == API_RESPONSES);
}

static int live_text_prefix_prompt(server *s, server_slot *slot,
                                   const request *req,
                                   ds4_tokens *effective_prompt) {
    if (!s || !slot || !req || !req->prompt_text || !effective_prompt) return 0;
    const ds4_tokens *live_tokens = ds4_session_tokens(slot->session);
    if (!live_tokens || live_tokens->len <= 0) return 0;

    size_t live_text_len = 0;
    char *live_text = render_tokens_text(s->engine, live_tokens, &live_text_len);
    const size_t prompt_text_len = strlen(req->prompt_text);
    if (!byte_prefix_match(req->prompt_text, prompt_text_len,
                           live_text, live_text_len))
    {
        free(live_text);
        return 0;
    }

    /* This is the core text-prefix case.  The live graph is authoritative, so
     * keep its sampled tokenization and tokenize only the request bytes that
     * come after it.  Reusing req->prompt's token suffix would be wrong: full
     * prompt BPE may have merged across this byte boundary. */
    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->prompt_text + live_text_len,
        effective_prompt);
    free(live_text);
    return live_tokens->len;
}

/* Tool-output-only Responses continuation.
 *
 * Some clients send just the new tool outputs after a tool call.  There is no
 * long visible prefix to match in that shape; the call_id itself is the
 * protocol binding to the previous live assistant output.  Use it only when the
 * remembered live frontier and call-id set match exactly. */
static int responses_live_continuation_prompt(server *s, server_slot *slot,
                                              const request *req,
                                              int live_pos,
                                              ds4_tokens *effective_prompt,
                                              int *matched_ids) {
    if (!s || !slot || !req || !effective_prompt) return 0;
    if (req->api != API_RESPONSES || !req->responses_live_suffix_text) return 0;
    if (req->responses_live_call_ids.len == 0) return 0;
    if (!responses_live_matches_request(s, slot,
                                        &req->responses_live_call_ids,
                                        live_pos)) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(slot->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->responses_live_suffix_text,
        effective_prompt);
    if (matched_ids) *matched_ids = req->responses_live_call_ids.len;
    return live_tokens->len;
}

/* Tool-result Anthropic continuation.
 *
 * /v1/messages has no server-side response object like the OpenAI Responses
 * API, but its tool_use_id is still a precise continuation handle inside a live
 * local agent loop.  When the IDs and live token frontier match, continue from
 * the sampled DSML state and append only the user tool_result suffix. */
static int anthropic_live_continuation_prompt(server *s, server_slot *slot,
                                              const request *req,
                                              int live_pos,
                                              ds4_tokens *effective_prompt,
                                              int *matched_ids) {
    if (!s || !slot || !req || !effective_prompt) return 0;
    if (req->api != API_ANTHROPIC || !req->anthropic_live_suffix_text) return 0;
    if (req->anthropic_live_call_ids.len == 0) return 0;
    if (!anthropic_live_matches_request(s, slot,
                                        &req->anthropic_live_call_ids,
                                        live_pos)) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(slot->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->anthropic_live_suffix_text,
        effective_prompt);
    if (matched_ids) *matched_ids = req->anthropic_live_call_ids.len;
    return live_tokens->len;
}

/* Visible-replay Responses continuation.
 *
 * Other clients send the full visible transcript on every turn even though the
 * API semantics still make the request a continuation.  For Responses, exact
 * token-prefix matching is the wrong first question: hidden reasoning may be
 * live in KV but absent from the replay by design.  Instead, verify that the
 * request's rendered text begins with the visible transcript remembered at the
 * live frontier.  If it does, continue from the live token prefix and tokenize
 * only the bytes after that visible boundary.
 *
 * If this check fails, DS4 has no special Responses state to trust.  The caller
 * then uses normal token/text/disk matching, which is the correct fallback for
 * cold starts, edits, restarts, or cross-client replays. */
static int responses_live_visible_prefix_prompt(server *s, server_slot *slot,
                                                const request *req,
                                                int live_pos,
                                                ds4_tokens *effective_prompt) {
    if (!s || !slot || !req || !req->prompt_text || !effective_prompt) return 0;
    if (req->api != API_RESPONSES) return 0;

    const size_t prompt_len = strlen(req->prompt_text);
    size_t visible_len = 0;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = slot->responses_live.valid &&
              slot->responses_live.live_tokens == live_pos &&
              slot->responses_live.visible_text &&
              slot->responses_live.visible_len < prompt_len &&
              byte_prefix_match(req->prompt_text, prompt_len,
                                slot->responses_live.visible_text,
                                slot->responses_live.visible_len);
    if (ok) visible_len = slot->responses_live.visible_len;
    pthread_mutex_unlock(&s->tool_mu);
    if (!ok) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(slot->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->prompt_text + visible_len,
        effective_prompt);
    return live_tokens->len;
}

/* Tool-less thinking continuation.
 *
 * Chat/completions and Anthropic do not have a previous_response_id object that
 * binds a later request to the last sampled turn.  Still, after a normal
 * tool-less thinking answer, the next prompt renderer intentionally omits that
 * hidden reasoning.  The live KV state is richer than the visible transcript.
 *
 * Remembering the visible transcript as a key lets us keep the sampled hidden
 * KV when the next request clearly extends that same visible history.  This is
 * the same byte-prefix idea used by the disk cache: the client-visible text
 * selects the checkpoint, while the payload stays the exact sampled token
 * frontier.  If the visible key does not match, callers fall back to ordinary
 * token/text/disk matching. */
static int thinking_live_visible_prefix_prompt(server *s, server_slot *slot,
                                               const request *req,
                                               int live_pos,
                                               ds4_tokens *effective_prompt) {
    if (!s || !slot || !req || !req->prompt_text || !effective_prompt) return 0;
    if (req->kind != REQ_CHAT || req->api == API_RESPONSES) return 0;

    const size_t prompt_len = strlen(req->prompt_text);
    size_t visible_len = 0;
    pthread_mutex_lock(&s->tool_mu);
    bool ok = slot->thinking_live.valid &&
              slot->thinking_live.live_tokens == live_pos &&
              slot->thinking_live.visible_text &&
              slot->thinking_live.visible_len < prompt_len &&
              byte_prefix_match(req->prompt_text, prompt_len,
                                slot->thinking_live.visible_text,
                                slot->thinking_live.visible_len);
    if (ok) visible_len = slot->thinking_live.visible_len;
    pthread_mutex_unlock(&s->tool_mu);
    if (!ok) return 0;

    const ds4_tokens *live_tokens = ds4_session_tokens(slot->session);
    if (!live_tokens || live_tokens->len != live_pos) return 0;

    build_prompt_from_exact_prefix_and_text_suffix(
        s->engine, live_tokens, req->prompt_text + visible_len,
        effective_prompt);
    return live_tokens->len;
}

