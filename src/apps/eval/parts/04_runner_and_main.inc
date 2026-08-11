static bool tui_has_switch_request(eval_ui *ui, int running_idx) {
    return ui->enabled &&
           ui->requested_case >= 0 &&
           ui->requested_case < ui->ncases &&
           ui->requested_case != running_idx;
}

static bool tui_has_quit_request(eval_ui *ui) {
    return ui->enabled && ui->quit_requested;
}

static void mark_case_pending(eval_ui *ui, int idx) {
    ui->status[idx] = EVAL_PENDING;
    ui->guess[idx][0] = '\0';
}

static double tui_wait_if_paused(eval_ui *ui, const char *phase) {
    if (!ui->enabled || !ui->paused) return 0.0;
    bool was_running = ui->run_clock_active;
    if (was_running) tui_run_clock_stop(ui);
    double start = now_sec();
    tui_refresh(ui, phase);
    while (ui->paused) {
        usleep(50000);
        tui_consume_input(ui);
        tui_refresh(ui, phase);
    }
    if (was_running) tui_run_clock_start(ui);
    tui_refresh(ui, phase);
    return now_sec() - start;
}

static void eval_prefill_progress(void *ud, const char *event, int current, int total) {
    eval_ui *ui = ud;
    if (!ui || !event) return;
    if (strcmp(event, "prefill_chunk") && strcmp(event, "prefill_display"))
        return;
    tui_consume_input(ui);
    tui_run_clock_tick(ui);
    ui->prefill_current = current;
    ui->prefill_total = total;
    double elapsed = now_sec() - ui->phase_start_sec;
    ui->speed_tps = elapsed > 0.001 ? (double)current / elapsed : 0.0;
    tui_refresh(ui, "prefill");
    double paused_sec = tui_wait_if_paused(ui, "prefill");
    if (paused_sec > 0.0) ui->phase_start_sec += paused_sec;
}

static eval_run_result run_one_case(ds4_engine *engine, ds4_session *session,
                                    const eval_config *cfg, eval_ui *ui,
                                    FILE *trace, int idx, uint64_t *rng) {
    const eval_case *tc = &eval_cases[idx];
    const bool tty = ui->enabled;
    const bool use_plain_color = !tty && isatty(STDOUT_FILENO);
    const ds4_think_mode think_mode = ds4_think_mode_for_context(cfg->think_mode, cfg->ctx_size);
    const char *system = eval_system_prompt();

    char *question = build_question_prompt(tc);
    if (!question) {
        fprintf(stderr, "ds4-eval: failed to allocate prompt\n");
        return EVAL_RUN_ERROR;
    }

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, system, question, think_mode, &prompt);
    ui->prompt_tokens[idx] = prompt.len;
    ui->generated_tokens[idx] = 0;

    if (prompt.len >= cfg->ctx_size) {
        ui->active_case = idx;
        if (!ui->selection_active) ui->selected_case = idx;
        ui->guess[idx][0] = '\0';
        ui->status[idx] = EVAL_SKIPPED;
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "SKIPPED",
                         "prompt does not fit context", system, question, "",
                         think_mode, prompt.len, 0, 0.0, "?", NULL);
        if (!tty) {
            printf("\n[%d/%d] SKIPPED %s/%s prompt=%d ctx=%d\n",
                   idx + 1, ui->ncases, tc->source, tc->id, prompt.len, cfg->ctx_size);
        }
        free(question);
        ds4_tokens_free(&prompt);
        return EVAL_RUN_OK;
    }

    int generation_limit = cfg->max_tokens;
    int ctx_generation_limit = cfg->ctx_size - prompt.len;
    if (ctx_generation_limit < generation_limit) generation_limit = ctx_generation_limit;
    if (generation_limit < 1) {
        ui->active_case = idx;
        if (!ui->selection_active) ui->selected_case = idx;
        ui->guess[idx][0] = '\0';
        ui->status[idx] = EVAL_SKIPPED;
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "SKIPPED",
                         "prompt leaves no generation room", system, question, "",
                         think_mode, prompt.len, 0, 0.0, "?", NULL);
        if (!tty) {
            printf("\n[%d/%d] SKIPPED %s/%s prompt=%d ctx=%d\n",
                   idx + 1, ui->ncases, tc->source, tc->id, prompt.len, cfg->ctx_size);
        }
        free(question);
        ds4_tokens_free(&prompt);
        return EVAL_RUN_OK;
    }

    ui->active_case = idx;
    if (!ui->selection_active) ui->selected_case = idx;
    ui->requested_case = -1;
    ui->guess[idx][0] = '\0';
    ui->status[idx] = EVAL_PREFILL;
    ui->max_tokens = generation_limit;
    ui->think_max_tokens = generation_limit - cfg->hard_limit_reply_budget;
    if (ui->think_max_tokens < 0) ui->think_max_tokens = 0;
    tui_run_clock_start(ui);
    if (tty) {
        tui_reset_stream(ui, tc, ds4_think_mode_enabled(think_mode));
        tui_refresh(ui, "prefill");
    } else {
        printf("\n%s[%d/%d] %s%s%s/%s%s%s %s%s\n",
               use_plain_color ? ANSI_BOLD : "",
               idx + 1, ui->ncases,
               use_plain_color ? ANSI_CYAN : "", tc->source,
               use_plain_color ? ANSI_RESET : "",
               use_plain_color ? ANSI_CYAN : "", tc->domain,
               use_plain_color ? ANSI_RESET : "", tc->title,
               use_plain_color ? ANSI_RESET : "");
        fflush(stdout);
    }

    char err[256];
    ds4_session_set_progress(session, eval_prefill_progress, ui);
    ds4_session_set_display_progress(session, eval_prefill_progress, ui);
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        ds4_session_set_progress(session, NULL, NULL);
        ds4_session_set_display_progress(session, NULL, NULL);
        tui_run_clock_stop(ui);
        fprintf(stderr, "ds4-eval: prefill failed for %s: %s\n", tc->id, err);
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "ERROR", err,
                         system, question, "", think_mode, prompt.len, 0, 0.0, "?", NULL);
        free(question);
        ds4_tokens_free(&prompt);
        return EVAL_RUN_ERROR;
    }
    ds4_session_set_progress(session, NULL, NULL);
    ds4_session_set_display_progress(session, NULL, NULL);
    int prompt_tokens = prompt.len;
    ui->prompt_tokens[idx] = prompt_tokens;
    ds4_tokens_free(&prompt);

    tui_consume_input(ui);
    tui_wait_if_paused(ui, "prefill");
    if (tui_has_quit_request(ui)) {
        ui->status[idx] = EVAL_STOPPED;
        ui->generated_tokens[idx] = 0;
        tui_run_clock_stop(ui);
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "STOPPED", NULL,
                         system, question, "", think_mode, prompt_tokens, 0, 0.0, "?", NULL);
        free(question);
        return EVAL_RUN_QUIT;
    }
    if (tui_has_switch_request(ui, idx)) {
        mark_case_pending(ui, idx);
        tui_run_clock_stop(ui);
        tui_refresh(ui, "idle");
        trace_write_case(trace, cfg, tc, idx, ui->ncases, "SWITCHED", NULL,
                         system, question, "", think_mode, prompt_tokens, 0, 0.0, "?", NULL);
        free(question);
        return EVAL_RUN_SWITCH;
    }

    ui->status[idx] = EVAL_THINKING;
    ui->generated = 0;
    ui->phase_start_sec = now_sec();
    ui->speed_tps = 0.0;
    byte_buf raw = {0};
    bool plain_in_think = ds4_think_mode_enabled(think_mode);
    bool generation_in_think = ds4_think_mode_enabled(think_mode);
    eval_think_close_info think_close = {0};
    ds4_tokens think_close_tokens = {0};
    if (generation_in_think) ds4_tokenize_text(engine, "</think>", &think_close_tokens);
    if (!tty && plain_in_think) plain_set_thinking_color(use_plain_color);
    tui_refresh(ui, "thinking");

    double t0 = ui->phase_start_sec;
    int forced_close_pos = -1;
    for (int i = 0; i < generation_limit; i++) {
        if (tty) {
            tui_consume_input(ui);
            if (tui_has_quit_request(ui)) {
                ui->status[idx] = EVAL_STOPPED;
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "STOPPED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_QUIT;
            }
            if (tui_has_switch_request(ui, idx)) {
                mark_case_pending(ui, idx);
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "SWITCHED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_SWITCH;
            }
            double paused_sec = tui_wait_if_paused(ui, ui->in_think ? "thinking" : "answer");
            if (paused_sec > 0.0) {
                ui->phase_start_sec += paused_sec;
                t0 += paused_sec;
            }
            if (tui_has_quit_request(ui)) {
                ui->status[idx] = EVAL_STOPPED;
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "STOPPED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_QUIT;
            }
            if (tui_has_switch_request(ui, idx)) {
                mark_case_pending(ui, idx);
                ui->generated_tokens[idx] = ui->generated;
                tui_run_clock_stop(ui);
                tui_refresh(ui, "idle");
                trace_write_case(trace, cfg, tc, idx, ui->ncases, "SWITCHED", NULL,
                                 system, question, raw.v ? raw.v : "", think_mode,
                                 prompt_tokens, ui->generated, now_sec() - t0, "?",
                                 &think_close);
                free(question);
                ds4_tokens_free(&think_close_tokens);
                buf_free(&raw);
                return EVAL_RUN_SWITCH;
            }
        }

        int remaining_budget = generation_limit - ui->generated;
        int close_rank = 0;
        int token = -1;
        eval_think_close_kind close_kind = EVAL_THINK_CLOSE_NONE;

        /* Benchmarks usually cap generation length, but DeepSeek can spend the
         * entire budget in <think>.  This controller only acts while the model is
         * still in thinking mode.  The soft limit is conservative: it accepts the
         * model's own desire to end thinking when </think> is already near the
         * top of the distribution.  The hard limit is a uniform benchmark rule:
         * leave a fixed answer reserve instead of failing a case because all
         * tokens were spent before the visible answer could start. */
        if (generation_in_think && think_close_tokens.len > 0) {
            if (forced_close_pos >= 0) {
                token = think_close_tokens.v[forced_close_pos++];
                if (forced_close_pos >= think_close_tokens.len) forced_close_pos = -1;
            } else if (remaining_budget <= cfg->hard_limit_reply_budget) {
                close_kind = EVAL_THINK_CLOSE_HARD;
                token = think_close_tokens.v[0];
                forced_close_pos = think_close_tokens.len > 1 ? 1 : -1;
            } else if (remaining_budget <= cfg->soft_limit_reply_budget &&
                       think_close_tokens.len == 1) {
                close_rank = token_rank_in_top(session, think_close_tokens.v[0],
                                               cfg->soft_limit_think_close_rank);
                if (close_rank > 0) {
                    close_kind = EVAL_THINK_CLOSE_SOFT;
                    token = think_close_tokens.v[0];
                }
            }
        }
        if (token < 0)
            token = ds4_session_sample(session, cfg->temperature, 0,
                                       cfg->top_p, cfg->min_p, rng);
        if (ds4_token_is_stop(engine, token)) break;
        if (close_kind != EVAL_THINK_CLOSE_NONE &&
            think_close.kind == EVAL_THINK_CLOSE_NONE) {
            think_close.kind = close_kind;
            think_close.token_index = ui->generated + 1;
            think_close.remaining_budget = remaining_budget;
            think_close.rank = close_rank;
        }
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            plain_reset_color(use_plain_color);
            ui->generated_tokens[idx] = ui->generated;
            tui_run_clock_stop(ui);
            fprintf(stderr, "ds4-eval: decode failed for %s: %s\n", tc->id, err);
            trace_write_case(trace, cfg, tc, idx, ui->ncases, "ERROR", err,
                             system, question, raw.v ? raw.v : "", think_mode,
                             prompt_tokens, ui->generated, now_sec() - t0, "?",
                             &think_close);
            free(question);
            ds4_tokens_free(&think_close_tokens);
            buf_free(&raw);
            return EVAL_RUN_ERROR;
        }

        size_t len = 0;
        char *text = ds4_token_text(engine, token, &len);
        buf_append(&raw, text, len);
        ui->generated++;
        ui->generated_tokens[idx] = ui->generated;
        tui_run_clock_tick(ui);
        if (generation_in_think && raw.v && strstr(raw.v, "</think>")) {
            generation_in_think = false;
            if (think_close.kind == EVAL_THINK_CLOSE_NONE) {
                think_close.kind = EVAL_THINK_CLOSE_NATURAL;
                think_close.token_index = ui->generated;
                think_close.remaining_budget = remaining_budget;
                think_close.rank = 0;
            }
        }
        double elapsed = now_sec() - ui->phase_start_sec;
        ui->speed_tps = elapsed > 0.001 ? (double)ui->generated / elapsed : 0.0;

        if (tty) {
            stream_append_token_text(ui, text, len, false);
            tui_refresh(ui, ui->in_think ? "thinking" : "answer");
        } else {
            if (plain_in_think && strstr(raw.v ? raw.v : "", "</think>")) {
                plain_in_think = false;
                plain_reset_color(use_plain_color);
            }
            fwrite(text, 1, len, stdout);
            fflush(stdout);
        }
        free(text);
    }
    if (tty) {
        stream_append_token_text(ui, NULL, 0, true);
        tui_draw_stream(ui);
    } else {
        plain_reset_color(use_plain_color);
        if (!raw.v || raw.len == 0 || raw.v[raw.len - 1] != '\n') fputc('\n', stdout);
    }

    char got[EVAL_ANSWER_MAX];
    find_case_answer(tc, raw.v ? raw.v : "", got, sizeof(got));
    snprintf(ui->guess[idx], EVAL_ANSWER_MAX, "%s", got);
    bool pass = answer_matches(tc, got);
    ui->status[idx] = pass ? EVAL_PASSED : EVAL_FAILED;
    ui->generated_tokens[idx] = ui->generated;
    tui_run_clock_stop(ui);
    double sec = now_sec() - t0;
    tui_refresh(ui, pass ? "passed" : "failed");
    trace_write_case(trace, cfg, tc, idx, ui->ncases, pass ? "PASSED" : "FAILED", NULL,
                     system, question, raw.v ? raw.v : "", think_mode, prompt_tokens,
                     ui->generated, sec, got, &think_close);

    if (!tty) {
        printf("%s%s%s got %s expected %s (%.1fs, %d tokens)\n",
               use_plain_color ? (pass ? ANSI_GREEN : ANSI_RED) : "",
               pass ? "PASSED" : "FAIL",
               use_plain_color ? ANSI_RESET : "",
               got, tc->answer, sec, ui->generated);
    }

    if (tty && cfg->pause_ms > 0) usleep((useconds_t)cfg->pause_ms * 1000);
    free(question);
    ds4_tokens_free(&think_close_tokens);
    buf_free(&raw);
    return EVAL_RUN_OK;
}

static int next_pending_case(eval_ui *ui, int start) {
    if (ui->ncases <= 0) return -1;
    for (int off = 0; off < ui->ncases; off++) {
        int i = (start + off) % ui->ncases;
        if (ui->status[i] == EVAL_PENDING) return i;
    }
    return -1;
}

static int parse_case_sequence(const char *arg, int ncases, int **seq_out, int *len_out) {
    int cap = 8;
    int len = 0;
    int *seq = malloc((size_t)cap * sizeof(*seq));
    if (!seq) {
        fprintf(stderr, "ds4-eval: out of memory while parsing --case-sequence\n");
        return -1;
    }

    const char *p = arg;
    while (p && *p) {
        while (isspace((unsigned char)*p)) p++;
        if (!*p) break;
        errno = 0;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || errno != 0 || v < 1 || v > ncases) {
            fprintf(stderr,
                    "ds4-eval: invalid --case-sequence entry near '%s' "
                    "(valid range: 1..%d)\n",
                    p, ncases);
            free(seq);
            return -1;
        }
        if (len == cap) {
            cap *= 2;
            int *new_seq = realloc(seq, (size_t)cap * sizeof(*seq));
            if (!new_seq) {
                fprintf(stderr, "ds4-eval: out of memory while parsing --case-sequence\n");
                free(seq);
                return -1;
            }
            seq = new_seq;
        }
        seq[len++] = (int)v - 1;
        p = end;
        while (isspace((unsigned char)*p)) p++;
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p) {
            fprintf(stderr, "ds4-eval: expected comma in --case-sequence near '%s'\n", p);
            free(seq);
            return -1;
        }
    }
    if (len == 0) {
        fprintf(stderr, "ds4-eval: --case-sequence cannot be empty\n");
        free(seq);
        return -1;
    }
    *seq_out = seq;
    *len_out = len;
    return 0;
}

static void log_context_memory(ds4_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk,
                               bool        ssd_streaming) {
    ds4_context_memory m =
        ds4_context_memory_estimate_with_prefill_mode(backend,
                                                      ctx_size,
                                                      prefill_chunk,
                                                      ssd_streaming);
    fprintf(stderr,
            "ds4-eval: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static int wait_distributed_route(ds4_session *session) {
    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4-eval: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4-eval: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }
        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4-eval: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static const char *report_status_name(eval_status st) {
    switch (st) {
    case EVAL_PASSED: return "PASSED";
    case EVAL_FAILED: return "FAILED";
    case EVAL_SKIPPED: return "SKIPPED";
    case EVAL_STOPPED: return "STOPPED";
    case EVAL_PREFILL: return "PREFILL";
    case EVAL_THINKING: return "RUNNING";
    case EVAL_PENDING:
    default: return "PENDING";
    }
}

static void print_eval_report(const eval_ui *ui, int ncases, int passed, int failed) {
    char elapsed[32];
    format_run_elapsed(elapsed, sizeof(elapsed), tui_run_clock_visible_sec(ui));

    printf("ds4-eval: %d/%d passed", passed, ncases);
    if (failed) printf(", %d failed", failed);
    printf(", runtime %s\n", elapsed);
    printf("%-3s %-8s %8s %8s %8s %-8s %-8s %s\n",
           "#", "state", "prompt", "gen", "total", "given", "correct", "test");
    for (int i = 0; i < ncases; i++) {
        int prompt_tokens = ui->prompt_tokens ? ui->prompt_tokens[i] : 0;
        int generated_tokens = ui->generated_tokens ? ui->generated_tokens[i] : 0;
        int total_tokens = prompt_tokens + generated_tokens;
        const char *given = ui->guess && ui->guess[i][0] ? ui->guess[i] : "-";
        printf("%3d %-8s %8d %8d %8d %-8s %-8s %s/%s\n",
               i + 1,
               report_status_name(ui->status[i]),
               prompt_tokens,
               generated_tokens,
               total_tokens,
               given,
               ui->cases[i].answer,
               ui->cases[i].source,
               ui->cases[i].id);
    }
}

int main(int argc, char **argv) {
    eval_config cfg = parse_options(argc, argv);
    if (cfg.self_test_extractors) return run_extractor_self_tests();
    if (cfg.regrade_trace_path) return regrade_trace_file(cfg.regrade_trace_path);

    int ncases = (int)(sizeof(eval_cases) / sizeof(eval_cases[0]));
    if (cfg.question_limit > 0 && cfg.question_limit < ncases) ncases = cfg.question_limit;
    if (cfg.question_limit > (int)(sizeof(eval_cases) / sizeof(eval_cases[0]))) {
        fprintf(stderr, "ds4-eval: only %zu questions are embedded\n",
                sizeof(eval_cases) / sizeof(eval_cases[0]));
        return 2;
    }
    int *case_sequence = NULL;
    int case_sequence_len = 0;
    if (cfg.case_sequence &&
        parse_case_sequence(cfg.case_sequence, ncases, &case_sequence, &case_sequence_len) != 0) {
        return 2;
    }
    if (!cfg.seed) {
        cfg.seed = (uint64_t)time(NULL) ^
                   ((uint64_t)getpid() << 32) ^
                   (uint64_t)clock();
    }

    FILE *trace = NULL;
    if (cfg.trace_path) {
        trace = fopen(cfg.trace_path, "w");
        if (!trace) {
            fprintf(stderr, "ds4-eval: cannot open trace '%s': %s\n",
                    cfg.trace_path, strerror(errno));
            free(case_sequence);
            return 2;
        }
    }

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .mtp_path = cfg.mtp_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .context_size = cfg.ctx_size > 0 ? cfg.ctx_size : 0,
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
        .power_percent = cfg.power_percent,
        .prefill_chunk = cfg.prefill_chunk,
        .ssd_streaming_cache_experts = cfg.ssd_streaming_cache_experts,
        .ssd_streaming_cache_bytes = cfg.ssd_streaming_cache_bytes,
        .ssd_streaming_full_layers = cfg.ssd_streaming_full_layers,
        .ssd_streaming_preload_experts = cfg.ssd_streaming_preload_experts,
        .simulate_used_memory_bytes = cfg.simulate_used_memory_bytes,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .ssd_streaming = cfg.ssd_streaming,
        .ssd_streaming_cold = cfg.ssd_streaming_cold,
        .ssd_streaming_full_layers_set = cfg.ssd_streaming_full_layers_set,
        .distributed = cfg.dist,
    };
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&cfg.dist, &opt, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-eval: %s\n", dist_err);
        if (trace) fclose(trace);
        free(case_sequence);
        return 2;
    }

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) {
        if (trace) fclose(trace);
        free(case_sequence);
        return 1;
    }

    int max_prompt_tokens = 0;
    int max_prompt_case = -1;
    const bool auto_ctx = cfg.ctx_size <= 0;
    if (auto_ctx) {
        cfg.ctx_size = eval_auto_context_size(engine, &cfg, eval_cases, ncases,
                                              &max_prompt_tokens, &max_prompt_case);
        fprintf(stderr,
                "ds4-eval: context auto-sized to %d tokens "
                "(largest prompt=%d tokens, case=%d, generation budget=%d)\n",
                cfg.ctx_size, max_prompt_tokens, max_prompt_case + 1, cfg.max_tokens);
    } else {
        max_prompt_tokens = eval_max_prompt_tokens(engine, &cfg, eval_cases, ncases,
                                                   cfg.ctx_size, &max_prompt_case);
        fprintf(stderr,
                "ds4-eval: context set to %d tokens "
                "(largest prompt=%d tokens, case=%d, generation budget=%d)\n",
                cfg.ctx_size, max_prompt_tokens, max_prompt_case + 1, cfg.max_tokens);
        eval_warn_context_budget(&cfg, max_prompt_tokens, max_prompt_case);
    }
    fprintf(stderr, "ds4-eval: model shape %s\n", ds4_engine_model_name(engine));
    eval_warn_think_max_downgraded(&cfg);
    trace_write_header(trace, &cfg, ds4_engine_model_name(engine), ncases, max_prompt_tokens);
    log_context_memory(cfg.backend,
                       cfg.ctx_size,
                       cfg.prefill_chunk,
                       cfg.ssd_streaming);

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_size) != 0) {
        fprintf(stderr, "ds4-eval: failed to create session\n");
        if (trace) fclose(trace);
        ds4_engine_close(engine);
        free(case_sequence);
        return 1;
    }
    if (cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR &&
        wait_distributed_route(session) != 0)
    {
        ds4_session_free(session);
        if (trace) fclose(trace);
        ds4_engine_close(engine);
        free(case_sequence);
        return 1;
    }

    eval_ui ui;
    bool split_ui = !cfg.plain && isatty(STDOUT_FILENO);
    tui_start(&ui, eval_cases, ncases, cfg.max_tokens, split_ui);

    uint64_t rng = cfg.seed;
    int rc = 0;
    int next = 0;
    int sequence_pos = 0;
    while (next >= 0) {
        if (case_sequence_len > 0) {
            if (sequence_pos >= case_sequence_len) break;
            next = case_sequence[sequence_pos++];
            ui.selected_case = next;
            ui.selection_active = false;
        }
        tui_consume_input(&ui);
        tui_wait_if_paused(&ui, "idle");
        if (ui.quit_requested) break;
        if (case_sequence_len == 0 && ui.requested_case >= 0) {
            next = ui.requested_case;
            ui.requested_case = -1;
            ui.selection_active = false;
            ui.selected_case = next;
        }

        eval_run_result result = run_one_case(engine, session, &cfg, &ui, trace, next, &rng);
        if (result == EVAL_RUN_ERROR) {
            rc = 1;
            break;
        }
        if (result == EVAL_RUN_QUIT) break;
        /* A successful case should advance to the next pending benchmark.  If a
         * stale quit flag ever survives here, clear it; real q handling either
         * returns EVAL_RUN_QUIT from run_one_case() or is consumed at the top of
         * the next idle iteration. */
        ui.quit_requested = false;
        if (case_sequence_len > 0) continue;
        if (ui.requested_case >= 0) {
            next = ui.requested_case;
            ui.requested_case = -1;
            ui.selection_active = false;
            ui.selected_case = next;
            continue;
        }
        next = next_pending_case(&ui, next + 1);
        if (next >= 0) {
            ui.selected_case = next;
            ui.selection_active = false;
        }
    }

    int passed = 0;
    int failed = 0;
    for (int i = 0; i < ncases; i++) {
        if (ui.status[i] == EVAL_PASSED) passed++;
        else if (ui.status[i] == EVAL_FAILED) failed++;
    }

    if (ui.active) tui_restore();
    print_eval_report(&ui, ncases, passed, failed);
    if (trace) {
        fprintf(trace,
                "===== SUMMARY =====\n"
                "passed: %d\n"
                "failed: %d\n"
                "total: %d\n",
                passed, failed, ncases);
        fflush(trace);
    }

    tui_free(&ui);
    ds4_session_free(session);
    ds4_engine_close(engine);
    if (trace) fclose(trace);
    free(case_sequence);
    return rc || failed ? 1 : 0;
}
