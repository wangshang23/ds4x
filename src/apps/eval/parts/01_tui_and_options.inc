static double run_clock_sec(void);

static void tui_run_clock_tick(eval_ui *ui) {
    if (!ui || !ui->run_clock_active) return;
    double now = run_clock_sec();
    double delta = now - ui->run_last_sec;
    if (delta > 0.0) ui->run_elapsed_sec += delta;
    ui->run_last_sec = now;
}

static void tui_run_clock_start(eval_ui *ui) {
    if (!ui || ui->run_clock_active) return;
    ui->run_last_sec = run_clock_sec();
    ui->run_clock_active = true;
}

static void tui_run_clock_stop(eval_ui *ui) {
    if (!ui || !ui->run_clock_active) return;
    tui_run_clock_tick(ui);
    ui->run_clock_active = false;
}

static double tui_run_clock_visible_sec(const eval_ui *ui) {
    if (!ui) return 0.0;
    double elapsed = ui->run_elapsed_sec;
    if (ui->run_clock_active) {
        double delta = run_clock_sec() - ui->run_last_sec;
        if (delta > 0.0) elapsed += delta;
    }
    return elapsed;
}

static void format_run_elapsed(char *dst, size_t dstlen, double sec) {
    if (sec < 0.0) sec = 0.0;
    unsigned long long minutes = (unsigned long long)(sec / 60.0);
    unsigned long long hours = minutes / 60ull;
    minutes %= 60ull;
    snprintf(dst, dstlen, "%02lluh:%02llum", hours, minutes);
}

typedef struct {
    bool enabled;
    bool raw_mode;
    bool thread_started;
    volatile sig_atomic_t running;
    pthread_t thread;
    pthread_mutex_t mu;
    struct termios orig_termios;
    int move_delta;
    bool enter_pressed;
    bool pause_pressed;
    bool quit_pressed;
} eval_input;

static eval_input global_input = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
};

static void buf_append(byte_buf *b, const char *p, size_t n) {
    if (n == 0) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 1024;
        while (cap < b->len + n + 1) cap *= 2;
        char *v = realloc(b->v, cap);
        if (!v) {
            fprintf(stderr, "ds4-eval: out of memory\n");
            exit(1);
        }
        b->v = v;
        b->cap = cap;
    }
    memcpy(b->v + b->len, p, n);
    b->len += n;
    b->v[b->len] = '\0';
}

static void buf_appendf(byte_buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    int n = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);
    if (n < 0) {
        va_end(ap);
        return;
    }
    char *tmp = malloc((size_t)n + 1);
    if (!tmp) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        exit(1);
    }
    vsnprintf(tmp, (size_t)n + 1, fmt, ap);
    va_end(ap);
    buf_append(b, tmp, (size_t)n);
    free(tmp);
}

static int eval_case_nchoices(const eval_case *tc) {
    int n = 0;
    while (n < EVAL_MAX_CHOICES && tc->choice[n]) n++;
    return n;
}

static bool eval_case_is_multiple_choice(const eval_case *tc) {
    return eval_case_nchoices(tc) > 0;
}

static bool eval_case_is_compsec(const eval_case *tc) {
    return tc->source && !strcmp(tc->source, "COMPSEC");
}

static void style_append(style_buf *b, unsigned char style, size_t n) {
    if (n == 0) return;
    if (b->len + n > b->cap) {
        size_t cap = b->cap ? b->cap : 1024;
        while (cap < b->len + n) cap *= 2;
        unsigned char *v = realloc(b->v, cap);
        if (!v) {
            fprintf(stderr, "ds4-eval: out of memory\n");
            exit(1);
        }
        b->v = v;
        b->cap = cap;
    }
    memset(b->v + b->len, style, n);
    b->len += n;
}

static void buf_free(byte_buf *b) {
    free(b->v);
    memset(b, 0, sizeof(*b));
}

static void style_free(style_buf *b) {
    free(b->v);
    memset(b, 0, sizeof(*b));
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static double run_clock_sec(void) {
    struct timespec ts;
#ifdef CLOCK_UPTIME_RAW
    /* On Darwin this clock excludes system sleep, which is exactly what the TUI
     * elapsed counter wants: benchmark runtime, not lid-closed wall time. */
    clock_gettime(CLOCK_UPTIME_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int parse_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int_arg(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_u64_arg(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_arg(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-eval: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-eval: invalid value for %s: %s\n", opt, s);
#ifdef DS4_ROCM_BUILD
    fprintf(stderr, "ds4-eval: valid backends are: metal, rocm, cpu\n");
#else
    fprintf(stderr, "ds4-eval: valid backends are: metal, cuda, cpu\n");
#endif
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_EVAL, topic);
}

static eval_config parse_options(int argc, char **argv) {
    eval_config c = {
        .model_path = "ds4flash.gguf",
        .backend = default_backend(),
        .max_tokens = 16000,
        .top_p = DS4_DEFAULT_TOP_P,
        .min_p = DS4_DEFAULT_MIN_P,
        .pause_ms = 350,
        .soft_limit_reply_budget = 1024,
        .hard_limit_reply_budget = 512,
        .soft_limit_think_close_rank = 3,
        .think_mode = DS4_THINK_HIGH,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg,
                                   &i,
                                   argc,
                                   argv,
                                   &c.dist,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-eval: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.ctx_size = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.max_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--questions")) {
            c.question_limit = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--case-sequence")) {
            c.case_sequence = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--temp")) {
            c.temperature = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.top_p = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.min_p = parse_float_arg(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--seed")) {
            c.seed = parse_u64_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--regrade-trace")) {
            c.regrade_trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--soft-limit-reply-budget")) {
            c.soft_limit_reply_budget = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--hard-limit-reply-budget")) {
            c.hard_limit_reply_budget = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--soft-limit-think-close-rank")) {
            c.soft_limit_think_close_rank = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--pause-ms")) {
            c.pause_ms = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4-eval: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.ssd_streaming_cache_experts = experts;
            c.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            int v = parse_nonnegative_int_arg(need_arg(&i, argc, argv, arg), arg);
            c.ssd_streaming_full_layers = (uint32_t)v;
            c.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-eval: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4-eval: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            int v = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-eval: --prefill-chunk must be positive\n");
                exit(2);
            }
            c.prefill_chunk = (uint32_t)v;
        } else if (!strcmp(arg, "--power")) {
            c.power_percent = parse_int_arg(need_arg(&i, argc, argv, arg), arg);
            if (c.power_percent < 1 || c.power_percent > 100) {
                fprintf(stderr, "ds4-eval: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--think")) {
            c.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.think_mode = DS4_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--plain")) {
            c.plain = true;
        } else if (!strcmp(arg, "--self-test-extractors")) {
            c.self_test_extractors = true;
        } else {
            fprintf(stderr, "ds4-eval: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }
    if (c.self_test_extractors || c.regrade_trace_path) return c;

    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.dist, NULL, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-eval: %s\n", dist_err);
        exit(2);
    }
    if (c.dist.role == DS4_DISTRIBUTED_WORKER) {
        fprintf(stderr, "ds4-eval: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }

    if (c.max_tokens > EVAL_MAX_CONTEXT) {
        fprintf(stderr,
                "ds4-eval: --tokens (%d) exceeds the %d token context cap\n",
                c.max_tokens, EVAL_MAX_CONTEXT);
        exit(2);
    }
    if (c.ctx_size > EVAL_MAX_CONTEXT) {
        fprintf(stderr,
                "ds4-eval: --ctx (%d) exceeds the %d token context cap\n",
                c.ctx_size, EVAL_MAX_CONTEXT);
        exit(2);
    }
    if (ds4_think_mode_enabled(c.think_mode) &&
        c.hard_limit_reply_budget >= c.max_tokens) {
        fprintf(stderr,
                "ds4-eval: --hard-limit-reply-budget (%d) must be smaller than --tokens (%d)\n",
                c.hard_limit_reply_budget, c.max_tokens);
        exit(2);
    }
    if (ds4_think_mode_enabled(c.think_mode) &&
        c.soft_limit_reply_budget < c.hard_limit_reply_budget) {
        fprintf(stderr,
                "ds4-eval: --soft-limit-reply-budget (%d) must be >= --hard-limit-reply-budget (%d)\n",
                c.soft_limit_reply_budget, c.hard_limit_reply_budget);
        exit(2);
    }
    return c;
}

static int terminal_size(int *cols, int *rows) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != 0 || ws.ws_col == 0 || ws.ws_row == 0) {
        *cols = 80;
        *rows = 24;
        return -1;
    }
    *cols = ws.ws_col;
    *rows = ws.ws_row;
    return 0;
}

static void term_move(int row, int col) {
    printf("\x1b[%d;%dH", row, col);
}

static void term_clear_to_eol(void) {
    fputs("\x1b[K", stdout);
}

static int tui_right_text_w(const eval_ui *ui) {
    /* Avoid writing the physical last column.  Many terminals defer wrapping
     * until the next byte after the last column, which would spill right-pane
     * status text into the next row's left pane. */
    return ui->right_w > 1 ? ui->right_w - 1 : ui->right_w;
}

static void tui_clear_left_line(eval_ui *ui, int row) {
    term_move(row, 1);
    for (int i = 0; i < ui->left_w; i++) fputc(' ', stdout);
    term_move(row, ui->left_w + 1);
    fputs(ANSI_DIM "|" ANSI_RESET, stdout);
    term_move(row, 1);
}

static void term_set_color_for_status(eval_status st) {
    switch (st) {
    case EVAL_PENDING:  fputs(ANSI_DIM, stdout); break;
    case EVAL_PREFILL:  fputs(ANSI_CYAN, stdout); break;
    case EVAL_THINKING: fputs(ANSI_YELLOW, stdout); break;
    case EVAL_SKIPPED:  fputs(ANSI_DIM, stdout); break;
    case EVAL_STOPPED:  fputs(ANSI_YELLOW ANSI_BOLD, stdout); break;
    case EVAL_PASSED:   fputs(ANSI_GREEN ANSI_BOLD, stdout); break;
    case EVAL_FAILED:   fputs(ANSI_RED ANSI_BOLD, stdout); break;
    }
}

static const char *status_name(eval_status st) {
    switch (st) {
    case EVAL_PENDING: return "PEND";
    case EVAL_PREFILL: return "FILL";
    case EVAL_THINKING: return "RUN";
    case EVAL_SKIPPED: return "SKIP";
    case EVAL_STOPPED: return "STOP";
    case EVAL_PASSED: return "PASS";
    case EVAL_FAILED: return "FAIL";
    }
    return "?";
}

static bool eval_status_running(eval_status st) {
    return st == EVAL_PREFILL || st == EVAL_THINKING;
}

static void print_trimmed(const char *s, int width) {
    if (width <= 0) return;
    int len = (int)strlen(s);
    if (len <= width) {
        fputs(s, stdout);
        return;
    }
    if (width <= 3) {
        for (int i = 0; i < width; i++) fputc('.', stdout);
        return;
    }
    fwrite(s, 1, (size_t)width - 3, stdout);
    fputs("...", stdout);
}

static void tui_draw_question_preview(eval_ui *ui, const eval_case *tc) {
    const char *q = tc->question;
    size_t pos = 0;
    const int width = tui_right_text_w(ui);

    for (int row = 0; row < 3; row++) {
        term_move(2 + row, ui->right_x);
        term_clear_to_eol();
        fputs(ANSI_BLUE, stdout);

        int printed = 0;
        bool last_space = true;
        while (q[pos] && printed < width) {
            unsigned char c = (unsigned char)q[pos++];
            if (isspace(c)) {
                if (last_space) continue;
                c = ' ';
                last_space = true;
            } else {
                last_space = false;
            }
            fputc(c, stdout);
            printed++;
        }

        if (row == 2) {
            while (q[pos] && isspace((unsigned char)q[pos])) pos++;
            if (q[pos] && width >= 3) {
                int ell_col = ui->right_x + (printed >= 3 ? printed - 3 : 0);
                term_move(2 + row, ell_col);
                fputs("...", stdout);
            }
        }
        fputs(ANSI_RESET, stdout);
    }

    term_move(5, ui->right_x);
    term_clear_to_eol();
}

static void input_queue_key(int key) {
    pthread_mutex_lock(&global_input.mu);
    if (key == 'A') {
        global_input.move_delta--;
    } else if (key == 'B') {
        global_input.move_delta++;
    } else if (key == '\r' || key == '\n') {
        global_input.enter_pressed = true;
    } else if (key == 'p' || key == 'P') {
        global_input.pause_pressed = true;
    } else if (key == 'q' || key == 'Q') {
        global_input.quit_pressed = true;
    }
    pthread_mutex_unlock(&global_input.mu);
}

static void *input_thread_main(void *arg) {
    (void)arg;
    int esc_state = 0;
    char buf[64];

    while (global_input.running) {
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n <= 0) {
            usleep(5000);
            continue;
        }
        for (ssize_t i = 0; i < n; i++) {
            unsigned char c = (unsigned char)buf[i];
            if (esc_state == 0) {
                if (c == 27) esc_state = 1;
                else if (c == 'p' || c == 'P' || c == 'q' || c == 'Q') input_queue_key(c);
                else if (c == '\r' || c == '\n') input_queue_key(c);
            } else if (esc_state == 1) {
                esc_state = (c == '[' || c == 'O') ? 2 : 0;
            } else {
                if (c == 'A' || c == 'B') input_queue_key(c);
                esc_state = 0;
            }
        }
    }
    return NULL;
}

static void tui_start_input(void) {
    if (!isatty(STDIN_FILENO)) return;
    if (tcgetattr(STDIN_FILENO, &global_input.orig_termios) != 0) return;

    /* The input thread is intentionally boring: it never writes to the terminal,
     * it only queues navigation/control state.  Rendering remains owned by the
     * main thread and follows the same full-frame redraw path as the
     * noninteractive UI. Keep ISIG set so Ctrl-C still restores the alternate
     * screen. */
    struct termios raw = global_input.orig_termios;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) return;

    pthread_mutex_lock(&global_input.mu);
    global_input.move_delta = 0;
    global_input.enter_pressed = false;
    global_input.pause_pressed = false;
    global_input.quit_pressed = false;
    pthread_mutex_unlock(&global_input.mu);

    global_input.raw_mode = true;
    global_input.enabled = true;
    global_input.running = 1;
    if (pthread_create(&global_input.thread, NULL, input_thread_main, NULL) == 0) {
        global_input.thread_started = true;
    } else {
        global_input.running = 0;
        global_input.enabled = false;
        tcsetattr(STDIN_FILENO, TCSANOW, &global_input.orig_termios);
        global_input.raw_mode = false;
    }
}

static void tui_stop_input(void) {
    if (global_input.thread_started) {
        global_input.running = 0;
        pthread_join(global_input.thread, NULL);
        global_input.thread_started = false;
    }
    if (global_input.raw_mode) {
        tcsetattr(STDIN_FILENO, TCSANOW, &global_input.orig_termios);
        global_input.raw_mode = false;
    }
    global_input.enabled = false;
}

static void tui_consume_input(eval_ui *ui) {
    if (!ui->enabled || !global_input.enabled) return;

    pthread_mutex_lock(&global_input.mu);
    int move = global_input.move_delta;
    bool enter = global_input.enter_pressed;
    bool pause = global_input.pause_pressed;
    bool quit = global_input.quit_pressed;
    global_input.move_delta = 0;
    global_input.enter_pressed = false;
    global_input.pause_pressed = false;
    global_input.quit_pressed = false;
    pthread_mutex_unlock(&global_input.mu);

    if (quit) {
        ui->quit_requested = true;
        ui->paused = false;
        return;
    }

    if (pause) ui->paused = !ui->paused;

    if (move) {
        if (!ui->selection_active) {
            ui->selected_case = ui->active_case;
            ui->selection_active = true;
        }
        int selected = ui->selected_case + move;
        if (selected < 0) selected = 0;
        if (selected >= ui->ncases) selected = ui->ncases - 1;
        ui->selected_case = selected;
    }

    if (enter) {
        if (!ui->selection_active) {
            ui->selected_case = ui->active_case;
            ui->selection_active = true;
        }
        if (ui->selected_case != ui->active_case ||
            !eval_status_running(ui->status[ui->active_case]))
        {
            ui->requested_case = ui->selected_case;
        }
    }
}

static void tui_restore(void) {
    eval_ui *ui = global_ui;
    tui_stop_input();
    if (!ui || !ui->active) return;
    fputs(ANSI_RESET "\x1b[?25h\x1b[?1049l", stdout);
    fflush(stdout);
    ui->active = false;
}

static void tui_signal_restore(int sig) {
    /* Signal handlers cannot safely run the full tui_restore() path: that path
     * joins the input thread and uses stdio.  For Ctrl-C / termination, do the
     * minimal terminal repair directly, then re-raise the signal with its
     * default action so the process status remains correct. */
    eval_ui *ui = global_ui;
    global_input.running = 0;
    if (global_input.raw_mode) {
        tcsetattr(STDIN_FILENO, TCSANOW, &global_input.orig_termios);
        global_input.raw_mode = false;
    }
    if (ui && ui->active) {
        const char restore[] = ANSI_RESET "\x1b[?25h\x1b[?1049l";
        if (write(STDOUT_FILENO, restore, sizeof(restore) - 1) == -1) {}
        ui->active = false;
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

static void tui_draw_title(eval_ui *ui) {
    term_move(1, 1);
    tui_clear_left_line(ui, 1);
    char elapsed[32];
    format_run_elapsed(elapsed, sizeof(elapsed), tui_run_clock_visible_sec(ui));
    fputs("ds4-eval (" ANSI_BOLD "p" ANSI_RESET ")ause (" ANSI_BOLD "q" ANSI_RESET ")uit", stdout);
    printf(" %s", elapsed);
    if (ui->paused) {
        fputs(" " ANSI_RED ANSI_BOLD "PAUSED" ANSI_RESET, stdout);
    }
}

static void tui_draw_frame(eval_ui *ui) {
    fputs("\x1b[2J", stdout);
    tui_draw_title(ui);
    term_move(1, ui->right_x);
    fputs(ANSI_BOLD "live sampled tokens" ANSI_RESET, stdout);

    for (int row = 1; row <= ui->rows; row++) {
        term_move(row, ui->left_w + 1);
        fputs(ANSI_DIM "|" ANSI_RESET, stdout);
    }
}

static void tui_draw_left(eval_ui *ui) {
    int passed = 0;
    int failed = 0;
    for (int i = 0; i < ui->ncases; i++) {
        if (ui->status[i] == EVAL_PASSED) passed++;
        else if (ui->status[i] == EVAL_FAILED) failed++;
    }

    tui_draw_title(ui);

    term_move(2, 1);
    tui_clear_left_line(ui, 2);
    printf("score %s%d%s/%d  failed %s%d%s",
           ANSI_GREEN, passed, ANSI_RESET,
           ui->ncases,
           failed ? ANSI_RED : ANSI_DIM, failed, ANSI_RESET);

    const int first_row = 4;
    const int visible_rows = ui->rows >= first_row ? ui->rows - first_row + 1 : 0;
    const int shown = ui->ncases < visible_rows ? ui->ncases : visible_rows;
    int first = 0;
    if (shown > 0 && ui->ncases > shown) {
        /* The list follows the selection cursor, not the running test.  This
         * makes arrow navigation visible immediately.  When normal execution
         * advances, main() moves the selection to the new active case so the
         * viewport naturally follows the work again. */
        int anchor = ui->selected_case;
        if (anchor < 0) anchor = ui->active_case;
        if (anchor >= ui->ncases) anchor = ui->ncases - 1;
        first = anchor - shown / 2;
        if (first < 0) first = 0;
        if (first > ui->ncases - shown) first = ui->ncases - shown;
    }

    for (int row = 0; row < visible_rows; row++) {
        const int screen_row = first_row + row;
        const int i = first + row;
        term_move(screen_row, 1);
        tui_clear_left_line(ui, screen_row);
        if (i >= ui->ncases) continue;

        if (i == ui->active_case) fputs(ANSI_BOLD, stdout);
        term_set_color_for_status(ui->status[i]);
        printf("%c%2d ", ui->selection_active && i == ui->selected_case ? '>' : ' ', i + 1);
        printf("%-4s", status_name(ui->status[i]));
        fputs(ANSI_RESET, stdout);

        const int answer_w = 18;
        const int answer_col = ui->left_w - answer_w + 1;
        int title_w = answer_col - 11;
        if (title_w > 0) {
            fputc(' ', stdout);
            char title[512];
            snprintf(title, sizeof(title), "%s: %s",
                     ui->cases[i].source, ui->cases[i].title);
            print_trimmed(title, title_w);
        }
        if (ui->status[i] == EVAL_FAILED || ui->status[i] == EVAL_PASSED) {
            char answers[64];
            snprintf(answers, sizeof(answers), "%s/%s",
                     ui->guess[i][0] ? ui->guess[i] : "?",
                     ui->cases[i].answer);
            term_move(screen_row, answer_col);
            print_trimmed(answers, answer_w);
        }
        fputs(ANSI_RESET, stdout);
    }
}

static void tui_reset_stream(eval_ui *ui, const eval_case *tc, bool in_think) {
    ui->stream.len = 0;
    if (ui->stream.v) ui->stream.v[0] = '\0';
    ui->styles.len = 0;
    ui->in_think = in_think;
    ui->pending_tag_len = 0;
    ui->generated = 0;
    ui->prefill_current = 0;
    ui->prefill_total = 0;
    ui->phase_start_sec = now_sec();
    ui->speed_tps = 0.0;

    for (int row = 2; row <= ui->rows; row++) {
        term_move(row, ui->right_x);
        term_clear_to_eol();
    }
    tui_draw_question_preview(ui, tc);
}

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void stream_append_visible(eval_ui *ui, const char *p, size_t n) {
    buf_append(&ui->stream, p, n);
    style_append(&ui->styles, ui->in_think ? 1 : 0, n);
}

static void stream_append_token_text(eval_ui *ui, const char *text, size_t len, bool finish) {
    const char *open = "<think>";
    const char *close = "</think>";
    size_t total = ui->pending_tag_len + len;
    char *tmp = malloc(total ? total : 1);
    if (!tmp) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        exit(1);
    }
    if (ui->pending_tag_len) memcpy(tmp, ui->pending_tag, ui->pending_tag_len);
    if (len) memcpy(tmp + ui->pending_tag_len, text, len);
    ui->pending_tag_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = tmp + i;
        size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, open)) {
            ui->in_think = true;
            i += strlen(open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, close)) {
            ui->in_think = false;
            stream_append_visible(ui, "\n", 1);
            i += strlen(close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, open) ||
             bytes_is_partial_prefix(cur, rem, close)))
        {
            if (rem < sizeof(ui->pending_tag)) {
                memcpy(ui->pending_tag, cur, rem);
                ui->pending_tag_len = rem;
            }
            break;
        }
        stream_append_visible(ui, cur, 1);
        i++;
    }
    free(tmp);
}

typedef struct {
    size_t start;
    size_t end;
} line_span;

static void tui_draw_stream(eval_ui *ui) {
    if (!ui->enabled) return;
    const int width = tui_right_text_w(ui);

    line_span *lines = NULL;
    int line_len = 0;
    int line_cap = 0;
    size_t line_start = 0;
    int col = 0;

    for (size_t i = 0; i < ui->stream.len; i++) {
        char c = ui->stream.v[i];
        bool end_line = false;
        size_t end = i;
        if (c == '\n') {
            end_line = true;
            end = i;
        } else {
            col++;
            if (col >= width) {
                end_line = true;
                end = i + 1;
            }
        }
        if (end_line) {
            if (line_len == line_cap) {
                line_cap = line_cap ? line_cap * 2 : 64;
                line_span *v = realloc(lines, (size_t)line_cap * sizeof(*lines));
                if (!v) {
                    fprintf(stderr, "ds4-eval: out of memory\n");
                    exit(1);
                }
                lines = v;
            }
            lines[line_len++] = (line_span){line_start, end};
            line_start = i + 1;
            col = 0;
        }
    }
    if (line_start <= ui->stream.len) {
        if (line_len == line_cap) {
            line_cap = line_cap ? line_cap * 2 : 64;
            line_span *v = realloc(lines, (size_t)line_cap * sizeof(*lines));
            if (!v) {
                fprintf(stderr, "ds4-eval: out of memory\n");
                exit(1);
            }
            lines = v;
        }
        lines[line_len++] = (line_span){line_start, ui->stream.len};
    }

    int start = line_len > ui->body_h ? line_len - ui->body_h : 0;
    for (int row = 0; row < ui->body_h; row++) {
        term_move(ui->body_y + row, ui->right_x);
        term_clear_to_eol();
        int li = start + row;
        if (li >= line_len) continue;
        unsigned char cur_style = 255;
        int printed = 0;
        for (size_t i = lines[li].start; i < lines[li].end && printed < width; i++) {
            unsigned char st = ui->styles.v[i];
            if (st != cur_style) {
                fputs(st ? ANSI_DIM : ANSI_RESET, stdout);
                cur_style = st;
            }
            char c = ui->stream.v[i];
            if (c == '\r' || c == '\n') continue;
            if (c == '\t') c = ' ';
            fputc((unsigned char)c, stdout);
            printed++;
        }
        fputs(ANSI_RESET, stdout);
    }
    free(lines);
}

static void format_short_count(char *dst, size_t dstlen, int n) {
    if (n >= 1000000) {
        snprintf(dst, dstlen, "%dm", (n + 500000) / 1000000);
    } else if (n >= 10000) {
        snprintf(dst, dstlen, "%dk", (n + 500) / 1000);
    } else {
        snprintf(dst, dstlen, "%d", n);
    }
}

static const char *short_phase_name(const char *phase) {
    if (!strcmp(phase, "prefill")) return "FILL";
    if (!strcmp(phase, "thinking")) return "THINK";
    if (!strcmp(phase, "answer")) return "ANS";
    if (!strcmp(phase, "passed")) return "PASS";
    if (!strcmp(phase, "failed")) return "FAIL";
    return "RUN";
}

static void tui_draw_right_status(eval_ui *ui, const char *phase) {
    term_move(1, ui->right_x);
    term_clear_to_eol();
    const eval_case *tc = &ui->cases[ui->active_case];
    char id[13];
    char gen[16], max[16], cur[16], total[16];
    char line[512];
    snprintf(id, sizeof(id), "%.12s", tc->id);
    if (ui->prefill_total > 0 && ui->status[ui->active_case] == EVAL_PREFILL) {
        double pct = 100.0 * (double)ui->prefill_current / (double)ui->prefill_total;
        format_short_count(cur, sizeof(cur), ui->prefill_current);
        format_short_count(total, sizeof(total), ui->prefill_total);
        snprintf(line, sizeof(line),
                 "Speed %.2f t/s  %s %s/%s %.0f%%  %s/%s",
                 ui->speed_tps, short_phase_name(phase), cur, total, pct,
                 tc->source, id);
    } else {
        int phase_max = !strcmp(phase, "thinking") ? ui->think_max_tokens : ui->max_tokens;
        int phase_gen = ui->generated;
        if (phase_max < 0) phase_max = 0;
        if (!strcmp(phase, "thinking") && phase_gen > phase_max) phase_gen = phase_max;
        format_short_count(gen, sizeof(gen), phase_gen);
        format_short_count(max, sizeof(max), phase_max);
        snprintf(line, sizeof(line),
                 "Speed %.2f t/s  %s %s/%s  %s/%s",
                 ui->speed_tps, short_phase_name(phase), gen, max,
                 tc->source, id);
    }
    print_trimmed(line, tui_right_text_w(ui));
}

static void tui_refresh(eval_ui *ui, const char *phase) {
    if (!ui->enabled) return;
    tui_draw_left(ui);
    tui_draw_right_status(ui, phase);
    tui_draw_stream(ui);
    fflush(stdout);
}

static void tui_start(eval_ui *ui, const eval_case *cases, int ncases, int max_tokens,
                      bool enabled) {
    memset(ui, 0, sizeof(*ui));
    ui->enabled = enabled;
    ui->cases = cases;
    ui->ncases = ncases;
    ui->max_tokens = max_tokens;
    ui->selected_case = 0;
    ui->requested_case = -1;
    ui->quit_requested = false;
    ui->status = calloc((size_t)ncases, sizeof(*ui->status));
    ui->guess = calloc((size_t)ncases, sizeof(*ui->guess));
    ui->prompt_tokens = calloc((size_t)ncases, sizeof(*ui->prompt_tokens));
    ui->generated_tokens = calloc((size_t)ncases, sizeof(*ui->generated_tokens));
    if (!ui->status || !ui->guess || !ui->prompt_tokens || !ui->generated_tokens) {
        fprintf(stderr, "ds4-eval: out of memory\n");
        exit(1);
    }
    if (!enabled) return;

    terminal_size(&ui->cols, &ui->rows);
    if (ui->cols < 90 || ui->rows < 18) {
        ui->enabled = false;
        return;
    }
    ui->left_w = ui->cols / 2;
    if (ui->left_w < 42) ui->left_w = 42;
    if (ui->left_w > 72) ui->left_w = 72;
    ui->right_x = ui->left_w + 3;
    ui->right_w = ui->cols - ui->right_x + 1;
    ui->body_y = 6;
    ui->body_h = ui->rows - ui->body_y + 1;

    global_ui = ui;
    atexit(tui_restore);
    signal(SIGINT, tui_signal_restore);
    signal(SIGTERM, tui_signal_restore);
#ifdef SIGHUP
    signal(SIGHUP, tui_signal_restore);
#endif
#ifdef SIGQUIT
    signal(SIGQUIT, tui_signal_restore);
#endif

    fputs("\x1b[?1049h\x1b[?25l", stdout);
    ui->active = true;
    tui_start_input();
    tui_draw_frame(ui);
    tui_refresh(ui, "idle");
}

static void tui_free(eval_ui *ui) {
    if (ui->active) tui_restore();
    free(ui->status);
    free(ui->guess);
    free(ui->prompt_tokens);
    free(ui->generated_tokens);
    buf_free(&ui->stream);
    style_free(&ui->styles);
    memset(ui, 0, sizeof(*ui));
}

