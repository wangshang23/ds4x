/* =========================================================================
 * Engine API and Process Lock.
 * =========================================================================
 *
 * The public entry points acquire the single instance lock, open the GGUF with
 * the backend-appropriate mmap policy, and expose tokenized prompt operations
 * to the CLI and server.
 */

const char *ds4_backend_name(ds4_backend backend) {
    switch (backend) {
    case DS4_BACKEND_METAL: return "metal";
    case DS4_BACKEND_CUDA:
#ifdef DS4_ROCM_BUILD
        return "rocm";
#else
        return "cuda";
#endif
    case DS4_BACKEND_CPU:   return "cpu";
    }
    return "unknown";
}

static void ds4_linux_graph_backend_set_oom_score(ds4_backend backend) {
#if defined(__linux__) && !defined(DS4_NO_GPU)
    static bool attempted = false;
    if (attempted) return;
    attempted = true;

    const int score = 1000;
    FILE *fp = fopen("/proc/self/oom_score_adj", "w");
    if (!fp) {
        fprintf(stderr,
                "ds4: failed to set Linux %s backend oom_score_adj=%d: %s\n",
                ds4_backend_name(backend),
                score,
                strerror(errno));
        return;
    }
    if (fprintf(fp, "%d\n", score) < 0) {
        const int err = errno;
        fclose(fp);
        fprintf(stderr,
                "ds4: failed to write Linux %s backend oom_score_adj=%d: %s\n",
                ds4_backend_name(backend),
                score,
                strerror(err));
        return;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr,
                "ds4: failed to close Linux %s backend oom_score_adj=%d: %s\n",
                ds4_backend_name(backend),
                score,
                strerror(errno));
        return;
    }
    fprintf(stderr,
            "ds4: Linux %s backend set oom_score_adj=%d\n",
            ds4_backend_name(backend),
            score);
#else
    (void)backend;
#endif
}

bool ds4_think_mode_enabled(ds4_think_mode mode) {
    return mode == DS4_THINK_HIGH || mode == DS4_THINK_MAX;
}

const char *ds4_think_mode_name(ds4_think_mode mode) {
    switch (mode) {
    case DS4_THINK_NONE: return "none";
    case DS4_THINK_HIGH: return "high";
    case DS4_THINK_MAX:  return "max";
    }
    return "unknown";
}

const char *ds4_think_max_prefix(void) {
    return DS4_REASONING_EFFORT_MAX_PREFIX;
}

uint32_t ds4_think_max_min_context(void) {
    return DS4_THINK_MAX_MIN_CONTEXT;
}

ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size) {
    if (mode == DS4_THINK_MAX && (uint32_t)(ctx_size > 0 ? ctx_size : 0) < DS4_THINK_MAX_MIN_CONTEXT) {
        return DS4_THINK_HIGH;
    }
    return mode;
}

static void ds4_release_instance_lock(void) {
    if (g_ds4_lock_fd >= 0) {
        close(g_ds4_lock_fd);
        g_ds4_lock_fd = -1;
    }
}

/* Refuse to start a second ds4 process.  The model can map tens of GiB, so a
 * stale accidental second run is more dangerous than a normal CLI error. */
static void ds4_acquire_instance_lock(void) {
    const char *path = getenv("DS4_LOCK_FILE");
    if (!path || !path[0]) path = "/tmp/ds4.lock";

    const int fd = open(path, O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        fprintf(stderr, "ds4: failed to open lock file %s: %s\n", path, strerror(errno));
        exit(2);
    }
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            char buf[64];
            const ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
            long owner = -1;
            if (n > 0) {
                buf[n] = '\0';
                char *end = NULL;
                owner = strtol(buf, &end, 10);
            }
            if (owner > 0) {
                fprintf(stderr, "ds4: another ds4 process is already running (pid %ld); refusing to start\n", owner);
            } else {
                fprintf(stderr, "ds4: another ds4 process is already running; refusing to start\n");
            }
            close(fd);
            exit(2);
        }
        fprintf(stderr, "ds4: failed to lock %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }

    if (ftruncate(fd, 0) != 0) {
        fprintf(stderr, "ds4: failed to truncate lock file %s: %s\n", path, strerror(errno));
        close(fd);
        exit(2);
    }
    dprintf(fd, "%ld\n", (long)getpid());
    g_ds4_lock_fd = fd;
    atexit(ds4_release_instance_lock);
}

#ifndef DS4_NO_GPU
typedef struct {
    uint32_t n_comp[DS4_MAX_LAYER];
    uint32_t n_index_comp[DS4_MAX_LAYER];
    uint32_t dspark_cache_start;
    uint32_t dspark_cache_token_start;
    uint32_t dspark_cache_len;
    bool light;
} ds4_spec_frontier;

typedef struct ds4_dspark_spec_stats {
    uint64_t cycles;
    uint64_t first_tokens;
    uint64_t proposed_tokens;
    uint64_t accepted_draft_tokens;
    uint64_t full_accepts;
    uint64_t partial_accepts;
    uint64_t direct_full_commits;
    uint64_t direct_partial_commits;
    uint64_t replay_fallbacks;
    uint64_t first_misses;
    uint64_t no_draft;
    uint64_t no_room;
    uint64_t invalid_draft;
    uint64_t draft_len_hist[DS4_DSPARK_MAX_BLOCK_SIZE + 1u];
    uint64_t accepted_len_hist[DS4_DSPARK_MAX_BLOCK_SIZE + 1u];
    uint64_t scheduler_skips;
    uint64_t tail_skips;
    uint64_t verifier_unavailable;
    uint64_t verifier_errors;
    double target_ms;
    double saved_ms;
    double propose_ms;
    double propose_stage0_ms;
    double propose_setup_ms;
    double propose_cache_ms;
    double propose_chain_ms;
    double propose_hidden_ms;
    double propose_conf0_ms;
    double propose_logits_ms;
    double propose_markov_ms;
    double propose_confidence_ms;
    double snapshot_ms;
    double verify_ms;
    double verify_upload_ms;
    double verify_layer_ms;
    double verify_head_ms;
    double verify_read_ms;
    uint64_t verifier_fused_head;
    double replay_ms;
    double total_ms;
} ds4_dspark_spec_stats;

#endif

struct ds4_session {
    ds4_engine *engine;
#ifndef DS4_NO_GPU
    ds4_gpu_graph graph;
    ds4_spec_frontier greedy_splitkv_anchor;
#endif
    token_vec checkpoint;
    token_vec greedy_splitkv_segment;
    float *logits;
    float *sample_probs;
    int greedy_splitkv_anchor_len;
#ifndef DS4_NO_GPU
    float *spec_row_logits;
    float *dspark_markov_bias;
    float *dspark_conf_features;
    size_t dspark_conf_features_cap;
#endif
#ifndef DS4_NO_GPU
    int dspark_draft_tokens[DS4_DSPARK_MAX_BLOCK_SIZE];
    uint32_t dspark_draft_len;
    uint32_t dspark_sched_cycles;
    uint32_t dspark_sched_accepted;
    uint32_t dspark_sched_no_draft;
    uint32_t dspark_sched_skip;
    uint32_t dspark_sched_lifetime_accepted;
    double dspark_sched_life_extra_ms;
    double dspark_sched_life_saved_ms;
    double dspark_sched_extra_ms;
    double dspark_sched_saved_ms;
    double dspark_last_target_eval_ms;
    double dspark_last_propose_ms;
    float dspark_last_confidence0;
    bool dspark_draft_valid;
    bool dspark_sched_skipped_cycle;
    bool dspark_sched_long_accept_seen;
    bool dspark_last_confidence0_valid;
    ds4_dspark_spec_stats dspark_stats;
#endif
    ds4_session_progress_fn progress;
    void *progress_ud;
    ds4_session_progress_fn display_progress;
    void *display_progress_ud;
    ds4_session_cancel_fn cancel;
    void *cancel_ud;
    uint32_t prefill_cap;
    int ctx_size;
    bool checkpoint_valid;
    bool greedy_splitkv_anchor_valid;
};

#ifndef DS4_NO_GPU
static bool ds4_dspark_stats_enabled(void);

static void ds4_dspark_stats_note_len(
        uint64_t hist[DS4_DSPARK_MAX_BLOCK_SIZE + 1u],
        uint32_t len) {
    if (len > DS4_DSPARK_MAX_BLOCK_SIZE) len = DS4_DSPARK_MAX_BLOCK_SIZE;
    hist[len]++;
}

static uint32_t ds4_dspark_env_u32(const char *name, uint32_t fallback) {
    const char *env = getenv(name);
    if (!env || !env[0]) return fallback;
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || errno != 0 || v > UINT32_MAX) return fallback;
    return (uint32_t)v;
}

static bool ds4_dspark_scheduler_enabled(void) {
    const char *env = getenv("DS4_DSPARK_SCHEDULER");
    return !env || !env[0] || strcmp(env, "0") != 0;
}

static uint32_t ds4_dspark_scheduler_window(void) {
    uint32_t v = ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_WINDOW", 4);
    return v ? v : 4;
}

static uint32_t ds4_dspark_scheduler_skip_cycles(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_SKIP", 2);
}

static uint32_t ds4_dspark_scheduler_slow_skip_cycles(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_SLOW_SKIP", 4);
}

static uint32_t ds4_dspark_scheduler_min_avg_milli(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_MIN_AVG_MILLI", 1500);
}

static uint32_t ds4_dspark_scheduler_max_ms_per_accept_milli(void) {
    return ds4_dspark_env_u32(
            "DS4_DSPARK_SCHEDULER_MAX_MS_PER_ACCEPT_MILLI", 0);
}

static uint32_t ds4_dspark_scheduler_max_extra_saved_ratio_milli(void) {
    return ds4_dspark_env_u32(
            "DS4_DSPARK_SCHEDULER_MAX_EXTRA_SAVED_RATIO_MILLI", 0);
}

static uint32_t ds4_dspark_scheduler_break_even_window(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_BREAK_EVEN_WINDOW", 0);
}

static uint32_t ds4_dspark_scheduler_no_draft_skip_cycles(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP", 3);
}

static uint32_t ds4_dspark_scheduler_short_accept_no_draft_skip_cycles(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_SHORT_ACCEPT_NO_DRAFT_SKIP", 4);
}

static uint32_t ds4_dspark_scheduler_cold_low_confidence_skip_cycles(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_SKIP", 7);
}

static uint32_t ds4_dspark_scheduler_tail_min_tokens(void) {
    return ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS", 10);
}

static float ds4_dspark_scheduler_cold_low_confidence_threshold(void) {
    return (float)ds4_dspark_env_u32("DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_MILLI", 500) / 1000.0f;
}

/* Timing-sensitive scheduling changes which arithmetic path advances a token.
 * Keep it opt-in so greedy DSpark output is reproducible across runs. */
static bool ds4_dspark_scheduler_timing_enabled(void) {
    return ds4_dspark_scheduler_max_ms_per_accept_milli() != 0 ||
           ds4_dspark_scheduler_max_extra_saved_ratio_milli() != 0;
}

static void ds4_session_dspark_scheduler_reset(ds4_session *s) {
    if (!s) return;
    s->dspark_sched_cycles = 0;
    s->dspark_sched_accepted = 0;
    s->dspark_sched_no_draft = 0;
    s->dspark_sched_extra_ms = 0.0;
    s->dspark_sched_saved_ms = 0.0;
}

static bool ds4_session_dspark_scheduler_should_skip(ds4_session *s) {
    if (!s || !ds4_dspark_scheduler_enabled()) return false;
    s->dspark_sched_skipped_cycle = false;
    if (s->dspark_sched_skip == 0) return false;
    s->dspark_sched_skip--;
    s->dspark_sched_skipped_cycle = true;
    s->dspark_stats.scheduler_skips++;
    if (getenv("DS4_DSPARK_SPEC_LOG") != NULL) {
        fprintf(stderr,
                "ds4: DSpark scheduler skip remaining=%u\n",
                s->dspark_sched_skip);
    }
    return true;
}

static void ds4_session_dspark_scheduler_note(
        ds4_session *s,
        uint32_t     accepted_drafts,
        bool         no_draft,
        double       extra_ms) {
    if (!s || !ds4_dspark_scheduler_enabled()) return;
    if (s->dspark_sched_skipped_cycle) {
        s->dspark_sched_skipped_cycle = false;
        return;
    }

    s->dspark_sched_cycles++;
    s->dspark_sched_accepted += accepted_drafts;
    if (accepted_drafts != 0) {
        if (s->dspark_sched_lifetime_accepted <=
            UINT32_MAX - accepted_drafts) {
            s->dspark_sched_lifetime_accepted += accepted_drafts;
        } else {
            s->dspark_sched_lifetime_accepted = UINT32_MAX;
        }
        if (accepted_drafts > 2u) {
            s->dspark_sched_long_accept_seen = true;
        }
    }
    if (no_draft) s->dspark_sched_no_draft++;
    if (extra_ms > 0.0 && isfinite(extra_ms)) {
        s->dspark_sched_extra_ms += extra_ms;
    }
    if (accepted_drafts != 0 &&
        s->dspark_last_target_eval_ms > 0.0 &&
        isfinite(s->dspark_last_target_eval_ms)) {
        const double saved_ms =
            s->dspark_last_target_eval_ms * (double)accepted_drafts;
        s->dspark_sched_saved_ms += saved_ms;
        if (ds4_dspark_stats_enabled()) {
            s->dspark_stats.saved_ms += saved_ms;
        }
    }

    const uint32_t no_draft_skip =
        ds4_dspark_scheduler_no_draft_skip_cycles();
    if (no_draft && no_draft_skip != 0) {
        uint32_t skip = no_draft_skip;
        if (s->dspark_sched_lifetime_accepted != 0 &&
            !s->dspark_sched_long_accept_seen) {
            const uint32_t short_accept_skip =
                ds4_dspark_scheduler_short_accept_no_draft_skip_cycles();
            if (skip < short_accept_skip) skip = short_accept_skip;
        } else if (s->dspark_sched_lifetime_accepted == 0 &&
                   s->dspark_last_confidence0_valid &&
                   s->dspark_last_confidence0 <=
                       ds4_dspark_scheduler_cold_low_confidence_threshold()) {
            const uint32_t cold_low_conf_skip =
                ds4_dspark_scheduler_cold_low_confidence_skip_cycles();
            if (skip < cold_low_conf_skip) skip = cold_low_conf_skip;
        }
        if (s->dspark_sched_skip < skip) {
            s->dspark_sched_skip = skip;
        }
        if (getenv("DS4_DSPARK_SPEC_LOG") != NULL) {
            fprintf(stderr,
                    "ds4: DSpark scheduler no-draft pause skip=%u "
                    "accepted_total=%u long_accept=%d confidence0=%s%.3f\n",
                    s->dspark_sched_skip,
                    s->dspark_sched_lifetime_accepted,
                    s->dspark_sched_long_accept_seen ? 1 : 0,
                    s->dspark_last_confidence0_valid ? "" : "n/a:",
                    s->dspark_last_confidence0);
        }
    }

    const uint32_t window = ds4_dspark_scheduler_window();
    const uint32_t break_even_window =
        ds4_dspark_scheduler_break_even_window();

    const uint32_t max_extra_saved_ratio_milli =
        ds4_dspark_scheduler_max_extra_saved_ratio_milli();
    const bool measured_unprofitable =
        max_extra_saved_ratio_milli != 0 &&
        s->dspark_sched_accepted != 0 &&
        s->dspark_sched_saved_ms > 0.0 &&
        s->dspark_sched_extra_ms * 1000.0 >
            s->dspark_sched_saved_ms *
            (double)max_extra_saved_ratio_milli;

    if (break_even_window != 0 &&
        s->dspark_sched_cycles >= break_even_window &&
        measured_unprofitable) {
        s->dspark_sched_skip = ds4_dspark_scheduler_slow_skip_cycles();
        if (getenv("DS4_DSPARK_SPEC_LOG") != NULL) {
            fprintf(stderr,
                    "ds4: DSpark scheduler break-even pause cycles=%u "
                    "accepted=%u saved=%.3fms extra=%.3fms skip=%u\n",
                    s->dspark_sched_cycles,
                    s->dspark_sched_accepted,
                    s->dspark_sched_saved_ms,
                    s->dspark_sched_extra_ms,
                    s->dspark_sched_skip);
        }
        ds4_session_dspark_scheduler_reset(s);
        return;
    }

    if (s->dspark_sched_cycles < window) return;

    const uint64_t avg_milli =
        ((uint64_t)s->dspark_sched_accepted * 1000ull) /
        (uint64_t)s->dspark_sched_cycles;
    const uint32_t min_avg_milli =
        ds4_dspark_scheduler_min_avg_milli();
    const bool low_accept = avg_milli < min_avg_milli;
    const bool many_no_draft =
        s->dspark_sched_no_draft * 2u >= s->dspark_sched_cycles;
    const uint32_t max_ms_per_accept_milli =
        ds4_dspark_scheduler_max_ms_per_accept_milli();
    const double extra_per_accept_ms =
        s->dspark_sched_accepted != 0 ?
        s->dspark_sched_extra_ms / (double)s->dspark_sched_accepted : 0.0;
    const bool slow_accept =
        max_ms_per_accept_milli != 0 &&
        s->dspark_sched_accepted != 0 &&
        extra_per_accept_ms * 1000.0 > (double)max_ms_per_accept_milli;
    if (low_accept || many_no_draft || slow_accept || measured_unprofitable) {
        s->dspark_sched_skip = ds4_dspark_scheduler_skip_cycles();
        if (many_no_draft || slow_accept || measured_unprofitable) {
            const uint32_t slow_skip = ds4_dspark_scheduler_slow_skip_cycles();
            if (s->dspark_sched_skip < slow_skip) {
                s->dspark_sched_skip = slow_skip;
            }
        }
        if (getenv("DS4_DSPARK_SPEC_LOG") != NULL) {
            fprintf(stderr,
                    "ds4: DSpark scheduler pause cycles=%u accepted=%u "
                    "avg=%.3f no_draft=%u extra_per_accept=%.3fms "
                    "saved=%.3fms extra=%.3fms skip=%u\n",
                    s->dspark_sched_cycles,
                    s->dspark_sched_accepted,
                    (double)avg_milli / 1000.0,
                    s->dspark_sched_no_draft,
                    extra_per_accept_ms,
                    s->dspark_sched_saved_ms,
                    s->dspark_sched_extra_ms,
                    s->dspark_sched_skip);
        }
    }
    ds4_session_dspark_scheduler_reset(s);
}
#endif
