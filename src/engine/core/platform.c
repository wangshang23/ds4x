#include "engine_internal.h"

/* Platform module. */
/* =========================================================================
 * ds4.c - DeepSeek V4 inference engine.
 * =========================================================================
 *
 * This file owns the fixed DeepSeek V4 tensor layouts and shared engine
 * helpers. Model validation accepts only the DeepSeek V4 Flash layout.
 *
 * Loading is mmap based. The loader parses only the GGUF header, metadata
 * table, and tensor directory; CUDA resolves tensor ranges from that mapping.
 */

const char DS4_REASONING_EFFORT_MAX_PREFIX[] =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

/* DeepSeek recommends Think Max only with at least a 384K-token context window.
 * Below that size we keep ordinary thinking to avoid injecting a prompt that
 * asks for a reasoning budget the allocated context is not meant to hold. */

bool ds4_graph_env_present(const char *name) {
    return name && getenv(name) != NULL;
}

const char *ds4_graph_env_value(const char *name) {
    const char *value = name ? getenv(name) : NULL;
    return (value && value[0]) ? value : NULL;
}

const ds4_shape DS4_SHAPE_FLASH = {
    .name = "DeepSeek V4 Flash",
    .variant = DS4_VARIANT_FLASH,
    .n_layer = 43,
    .n_embd = 4096,
    .n_vocab = 129280,
    .n_head = 64,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 8,
    .n_lora_q = 1024,
    .n_lora_o = 1024,
    .n_expert = 256,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_ff_exp = 2048,
    .n_hash_layer = 3,
    .n_swa = 128,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 512,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .rms_eps = DS4_DEFAULT_RMS_EPS,
    .hc_eps = DS4_DEFAULT_HC_EPS,
    .expert_weight_scale = 1.5f,
    .swiglu_clamp_exp = DS4_DEFAULT_SWIGLU_CLAMP_EXP,
    .rope_freq_base = DS4_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = DS4_DEFAULT_ROPE_SCALE_FACTOR,
    .rope_yarn_beta_fast = DS4_DEFAULT_ROPE_YARN_BETA_FAST,
    .rope_yarn_beta_slow = DS4_DEFAULT_ROPE_YARN_BETA_SLOW,
    .compress_rope_freq_base = DS4_DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    .rope_orig_ctx = DS4_DEFAULT_ROPE_ORIG_CTX,
};

ds4_shape g_ds4_shape = {
    .name = "DeepSeek V4 Flash",
    .variant = DS4_VARIANT_FLASH,
    .n_layer = 43,
    .n_embd = 4096,
    .n_vocab = 129280,
    .n_head = 64,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 8,
    .n_lora_q = 1024,
    .n_lora_o = 1024,
    .n_expert = 256,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_ff_exp = 2048,
    .n_hash_layer = 3,
    .n_swa = 128,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 512,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .rms_eps = DS4_DEFAULT_RMS_EPS,
    .hc_eps = DS4_DEFAULT_HC_EPS,
    .expert_weight_scale = 1.5f,
    .swiglu_clamp_exp = DS4_DEFAULT_SWIGLU_CLAMP_EXP,
    .rope_freq_base = DS4_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = DS4_DEFAULT_ROPE_SCALE_FACTOR,
    .rope_yarn_beta_fast = DS4_DEFAULT_ROPE_YARN_BETA_FAST,
    .rope_yarn_beta_slow = DS4_DEFAULT_ROPE_YARN_BETA_SLOW,
    .compress_rope_freq_base = DS4_DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    .rope_orig_ctx = DS4_DEFAULT_ROPE_ORIG_CTX,
};

uint32_t g_ds4_compress_ratios[DS4_MAX_LAYER] = {0};


int g_ds4_lock_fd = -1;

/* =========================================================================
 * GGUF Quant Block Formats.
 * =========================================================================
 *
 * These layouts and IQ2 tables match the GGUF quantized tensor format,
 * reduced to only the formats ds4.c currently reads or sizes:
 *   - Q2_K routed down experts
 *   - Q4_K routed experts in compatible Flash checkpoints
 *   - IQ2_XXS routed gate/up experts
 *   - MXFP4 routed experts preserved from native checkpoints
 *   - Q8_K temporary activation blocks for dot products
 */




/* =========================================================================
 * Shared Helpers, Allocation Guards, Threads, and Cursor Reads.
 * =========================================================================
 *
 * This section holds process-wide utilities used by all later stages:
 * fatal-error helpers, allocation wrappers, the persistent CPU worker pool,
 * and the small byte cursor used to parse GGUF metadata.
 */



void ds4_die(const char *msg) {
    fprintf(stderr, "ds4: %s\n", msg);
    exit(1);
}

/* Attention compression is read from GGUF metadata after validating that it
 * matches the exact layout expected for the loaded model shape. */
uint32_t ds4_layer_compress_ratio(uint32_t il) {
    if (il >= DS4_N_LAYER) ds4_die("DeepSeek4 layer index is outside the loaded model layout");
    return g_ds4_compress_ratios[il];
}

uint32_t ds4_expected_layer_compress_ratio(uint32_t il) {
    if (il >= DS4_N_LAYER) ds4_die("DeepSeek4 layer index is outside the loaded model layout");

    if (il < 2) return 0;
    return (il & 1u) == 0 ? 4u : 128u;
}

void ds4_die_errno(const char *what, const char *path) {
    fprintf(stderr, "ds4: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

bool ds4_streq(ds4_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}

bool ds4_str_starts_with(ds4_str s, const char *prefix) {
    size_t n = strlen(prefix);
    return s.len >= n && memcmp(s.ptr, prefix, n) == 0;
}

bool ds4_str_contains(ds4_str s, const char *needle) {
    size_t n = strlen(needle);
    if (n == 0) return true;
    if (s.len < n) return false;
    for (uint64_t i = 0; i <= s.len - n; i++) {
        if (memcmp(s.ptr + i, needle, n) == 0) return true;
    }
    return false;
}

bool ds4_str_eq(ds4_str a, ds4_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, a.len) == 0;
}

uint64_t hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) ds4_die("out of memory");
    return p;
}

void *xmalloc(size_t size) {
    void *p = malloc(size);
    if (!p) ds4_die("out of memory");
    return p;
}

char *ds4_strdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

void *xrealloc(void *ptr, size_t size) {
    void *p = realloc(ptr, size);
    if (!p) ds4_die("out of memory");
    return p;
}

double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

void sleep_sec(double sec) {
    if (sec <= 0.0 || !isfinite(sec)) return;
    struct timespec req;
    req.tv_sec = (time_t)sec;
    req.tv_nsec = (long)((sec - (double)req.tv_sec) * 1000000000.0);
    if (req.tv_nsec < 0) req.tv_nsec = 0;
    if (req.tv_nsec >= 1000000000L) {
        req.tv_sec++;
        req.tv_nsec -= 1000000000L;
    }
    /* Do not resume after EINTR: Ctrl+C should cut through throttling sleeps. */
    (void)nanosleep(&req, &req);
}

static const char *ds4_log_color_code(ds4_log_type type) {
    switch (type) {
    case DS4_LOG_PREFILL:
    case DS4_LOG_TIMING:
        return "\x1b[36m";
    case DS4_LOG_GENERATION:
    case DS4_LOG_OK:
        return "\x1b[32m";
    case DS4_LOG_KVCACHE:
        return "\x1b[33m";
    case DS4_LOG_TOOL:
        return "\x1b[90m";
    case DS4_LOG_WARNING:
        return "\x1b[38;5;208m";
    case DS4_LOG_ERROR:
        return "\x1b[31m";
    default:
        return "";
    }
}

bool ds4_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}

static void ds4_vlog(FILE *fp, ds4_log_type type, const char *fmt, va_list ap) {
    const bool colorize = type != DS4_LOG_DEFAULT && ds4_log_is_tty(fp);
    if (colorize) fputs(ds4_log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
}

void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    ds4_vlog(fp, type, fmt, ap);
    va_end(ap);
}

bool write_f32_binary_file(const char *path, const float *data, uint64_t n) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open %s for writing: %s\n", path, strerror(errno));
        return false;
    }
    const size_t nw = fwrite(data, sizeof(float), (size_t)n, fp);
    const bool ok = nw == (size_t)n && fclose(fp) == 0;
    if (!ok) {
        fprintf(stderr, "ds4: failed to write %s\n", path);
        return false;
    }
    return true;
}

bool read_f32_binary_file(const char *path, float *data, uint64_t n) {
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "ds4: failed to stat %s: %s\n", path, strerror(errno));
        return false;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size != n * sizeof(float)) {
        fprintf(stderr,
                "ds4: %s has size %llu bytes, expected %llu bytes\n",
                path,
                (unsigned long long)st.st_size,
                (unsigned long long)(n * sizeof(float)));
        return false;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open %s for reading: %s\n", path, strerror(errno));
        return false;
    }
    const size_t nr = fread(data, sizeof(float), (size_t)n, fp);
    const bool ok = nr == (size_t)n && fclose(fp) == 0;
    if (!ok) {
        fprintf(stderr, "ds4: failed to read %s\n", path);
        return false;
    }
    return true;
}



ds4_thread_pool g_pool;
static __thread int g_parallel_depth;
uint32_t g_requested_threads;

static void *ds4_worker_main(void *arg) {
    const uint32_t tid = (uint32_t)(uintptr_t)arg;
    uint32_t seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&g_pool.mutex);
        while (seen_generation == g_pool.generation && !g_pool.shutdown) {
            pthread_cond_wait(&g_pool.work_cond, &g_pool.mutex);
        }
        if (g_pool.shutdown) {
            pthread_mutex_unlock(&g_pool.mutex);
            return NULL;
        }

        seen_generation = g_pool.generation;
        ds4_parallel_fn fn = g_pool.fn;
        void *ctx = g_pool.ctx;
        const uint64_t n_rows = g_pool.n_rows;
        const uint32_t n_threads = g_pool.n_threads;
        pthread_mutex_unlock(&g_pool.mutex);

        const uint64_t rows_per_thread = (n_rows + n_threads - 1) / n_threads;
        const uint64_t row0 = (uint64_t)tid * rows_per_thread;
        uint64_t row1 = row0 + rows_per_thread;
        if (row1 > n_rows) row1 = n_rows;
        if (row0 < row1) {
            g_parallel_depth++;
            fn(ctx, row0, row1);
            g_parallel_depth--;
        }

        pthread_mutex_lock(&g_pool.mutex);
        g_pool.done++;
        if (g_pool.done == g_pool.n_workers) {
            pthread_cond_signal(&g_pool.done_cond);
        }
        pthread_mutex_unlock(&g_pool.mutex);
    }
}

/* Create the persistent CPU worker pool.  Decode reuses these threads instead
 * of creating pthreads in the token loop. */
void ds4_threads_init(void) {
    if (g_pool.initialized) return;

    uint32_t n_threads = 12;
    const long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (online_cpus > 0) {
        n_threads = online_cpus < 12 ? (uint32_t)online_cpus : 12;
    }

    const char *env = getenv("DS4_THREADS");
    if (env && env[0]) {
        long v = strtol(env, NULL, 10);
        if (v > 0) n_threads = (uint32_t)v;
    }
    if (g_requested_threads > 0) n_threads = g_requested_threads;
    if (n_threads > DS4_MAX_THREADS) n_threads = DS4_MAX_THREADS;
    if (n_threads == 0) n_threads = 1;

    pthread_mutex_init(&g_pool.mutex, NULL);
    pthread_cond_init(&g_pool.work_cond, NULL);
    pthread_cond_init(&g_pool.done_cond, NULL);
    g_pool.n_threads = n_threads;
    g_pool.n_workers = n_threads > 0 ? n_threads - 1 : 0;
    g_pool.generation = 0;
    g_pool.done = 0;
    g_pool.shutdown = false;
    g_pool.initialized = true;

    for (uint32_t i = 1; i < n_threads; i++) {
        if (pthread_create(&g_pool.threads[i], NULL, ds4_worker_main, (void *)(uintptr_t)i) != 0) {
            ds4_die("failed to create worker thread");
        }
    }
}

void ds4_threads_shutdown(void) {
    if (!g_pool.initialized) return;

    pthread_mutex_lock(&g_pool.mutex);
    g_pool.shutdown = true;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.work_cond);
    pthread_mutex_unlock(&g_pool.mutex);

    for (uint32_t i = 1; i < g_pool.n_threads; i++) {
        pthread_join(g_pool.threads[i], NULL);
    }

    pthread_cond_destroy(&g_pool.done_cond);
    pthread_cond_destroy(&g_pool.work_cond);
    pthread_mutex_destroy(&g_pool.mutex);
    memset(&g_pool, 0, sizeof(g_pool));
}

/* Run a row-parallel CPU kernel, falling back to serial execution for small
 * jobs or nested calls where spawning more work would only add latency. */
static void ds4_parallel_for_min_rows(uint64_t n_rows, ds4_parallel_fn fn, void *ctx, uint64_t min_parallel_rows) {
    ds4_threads_init();

    if (g_parallel_depth > 0 || g_pool.n_threads <= 1 || n_rows < min_parallel_rows) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&g_pool.mutex);
    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.n_rows = n_rows;
    g_pool.done = 0;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.work_cond);

    const uint64_t rows_per_thread = (n_rows + g_pool.n_threads - 1) / g_pool.n_threads;
    uint64_t main_row1 = rows_per_thread;
    if (main_row1 > n_rows) main_row1 = n_rows;
    pthread_mutex_unlock(&g_pool.mutex);

    if (main_row1 > 0) {
        g_parallel_depth++;
        fn(ctx, 0, main_row1);
        g_parallel_depth--;
    }

    pthread_mutex_lock(&g_pool.mutex);
    while (g_pool.done < g_pool.n_workers) {
        pthread_cond_wait(&g_pool.done_cond, &g_pool.mutex);
    }
    pthread_mutex_unlock(&g_pool.mutex);
}

void ds4_parallel_for(uint64_t n_rows, ds4_parallel_fn fn, void *ctx) {
    ds4_parallel_for_min_rows(n_rows, fn, ctx, 512);
}

void cursor_error(ds4_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(ds4_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

bool cursor_read(ds4_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

bool cursor_skip(ds4_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

bool cursor_u32(ds4_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

bool cursor_u64(ds4_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

bool cursor_string(ds4_cursor *c, ds4_str *s) {
    uint64_t len;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}
