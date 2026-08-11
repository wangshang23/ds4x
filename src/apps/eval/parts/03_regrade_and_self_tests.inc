typedef enum {
    REGRADE_NOT_GRADED,
    REGRADE_PASSED,
    REGRADE_FAILED,
} regrade_outcome;

/* Decide how one trace case regrades. Only PASSED/FAILED traces were graded by
 * a completed run; STOPPED/SKIPPED/SWITCHED/ERROR cases carry partial or no
 * model output and must be reported as not-graded rather than counted, since
 * grading them inflates the totals and raises spurious "changed" drift. An
 * empty status keeps legacy traces (written before the status line) working. */
static regrade_outcome regrade_case_outcome(const eval_case *tc,
                                            const char *traced_status,
                                            const char *model_output,
                                            char *got, size_t gotlen) {
    bool gradeable = traced_status[0] == '\0' ||
                     !strcmp(traced_status, "PASSED") ||
                     !strcmp(traced_status, "FAILED");
    if (!gradeable) {
        if (gotlen) got[0] = '\0';
        return REGRADE_NOT_GRADED;
    }
    find_case_answer(tc, model_output, got, gotlen);
    return answer_matches(tc, got) ? REGRADE_PASSED : REGRADE_FAILED;
}

static int regrade_trace_file(const char *path) {
    size_t len = 0;
    char *text = read_text_file(path, &len);
    if (!text) return 2;

    const char *start = text;
    const char *end = text + len;
    int total = 0;
    int passed = 0;
    int failed = 0;
    int changed = 0;
    int unknown = 0;
    int parse_errors = 0;
    int not_graded = 0;

    while (true) {
        const char *case_start = trace_find_next_case(start, end);
        if (!case_start) break;
        const char *after_header = memchr(case_start, '\n', (size_t)(end - case_start));
        after_header = after_header ? after_header + 1 : end;
        const char *case_end = trace_find_next_case(after_header, end);
        if (!case_end) case_end = end;
        start = case_end;
        total++;

        char source[64];
        char id[128];
        char traced_status[32];
        char traced_pick[EVAL_ANSWER_MAX];
        if (!trace_get_line_field(case_start, case_end, "source: ", source, sizeof(source)) ||
            !trace_get_line_field(case_start, case_end, "id: ", id, sizeof(id))) {
            fprintf(stderr, "ds4-eval: trace case %d is missing source/id\n", total);
            parse_errors++;
            continue;
        }
        trace_get_line_field(case_start, case_end, "status: ", traced_status, sizeof(traced_status));
        trace_get_line_field(case_start, case_end, "picked: ", traced_pick, sizeof(traced_pick));

        const eval_case *tc = find_eval_case_by_source_id(source, id);
        if (!tc) {
            fprintf(stderr, "ds4-eval: trace case %d not found in embedded cases: %s/%s\n",
                    total, source, id);
            unknown++;
            continue;
        }

        char *model_output = trace_copy_model_output(case_start, case_end);
        if (!model_output) {
            fprintf(stderr, "ds4-eval: trace case %d is missing MODEL_OUTPUT block: %s/%s\n",
                    total, source, id);
            parse_errors++;
            continue;
        }

        char got[EVAL_ANSWER_MAX];
        regrade_outcome outcome =
            regrade_case_outcome(tc, traced_status, model_output, got, sizeof(got));
        if (outcome == REGRADE_NOT_GRADED) {
            not_graded++;
            free(model_output);
            continue;
        }
        bool ok = (outcome == REGRADE_PASSED);
        if (ok) passed++;
        else failed++;

        bool traced_ok = !strcmp(traced_status, "PASSED");
        if ((traced_status[0] && ok != traced_ok) ||
            (traced_pick[0] && strcmp(got, traced_pick) != 0)) {
            changed++;
            printf("case %d %s/%s: trace %s picked=%s -> regrade %s picked=%s expected=%s\n",
                   total, source, id,
                   traced_status[0] ? traced_status : "?",
                   traced_pick[0] ? traced_pick : "?",
                   ok ? "PASSED" : "FAILED", got, tc->answer ? tc->answer : "?");
        }
        free(model_output);
    }

    printf("ds4-eval: regraded %d cases from %s: passed=%d failed=%d changed=%d not_graded=%d unknown=%d parse_errors=%d\n",
           total, path, passed, failed, changed, not_graded, unknown, parse_errors);
    free(text);
    return (unknown || parse_errors || total == 0) ? 1 : 0;
}

static int extractor_self_test_case(const char *name, const eval_case *tc,
                                    const char *generated,
                                    const char *expected_extract) {
    char got[EVAL_ANSWER_MAX];
    find_case_answer(tc, generated, got, sizeof(got));
    if (strcmp(got, expected_extract) == 0 && answer_matches(tc, got)) return 0;

    fprintf(stderr,
            "ds4-eval: extractor self-test failed: %s (got %s, expected %s, key %s)\n",
            name, got, expected_extract, tc->answer ? tc->answer : "?");
    return 1;
}

static int trace_copy_self_test_case(void) {
    const char *prompt_output =
        "Prompt text may mention\n"
        "MODEL_OUTPUT_BEGIN without being the model block.\n";
    const char *model_output =
        "</think>Trace payload may mention\n"
        "MODEL_OUTPUT_END before the real marker.\n"
        "===== CASE not a real case =====\n"
        "Answer: F";
    char trace_case[1024];
    int n = snprintf(trace_case, sizeof(trace_case),
                     "===== CASE 1/2 SuperGPQA/trace-copy-self-test =====\n"
                     "source: SuperGPQA\n"
                     "id: trace-copy-self-test\n"
                     "QUESTION_PROMPT_BEGIN bytes=%zu\n"
                     "%s\n"
                     "QUESTION_PROMPT_END\n"
                     "MODEL_OUTPUT_BEGIN bytes=%zu\n"
                     "%s\n"
                     "MODEL_OUTPUT_END\n"
                     "\n"
                     "===== CASE 2/2 SuperGPQA/trace-copy-self-test-2 =====\n",
                     strlen(prompt_output), prompt_output,
                     strlen(model_output), model_output);
    if (n < 0 || (size_t)n >= sizeof(trace_case)) {
        fprintf(stderr, "ds4-eval: trace self-test setup failed\n");
        return 1;
    }

    const char *trace_start = trace_case;
    const char *trace_end = trace_case + strlen(trace_case);
    const char *first = trace_find_next_case(trace_start, trace_end);
    const char *after_header = first ? memchr(first, '\n', (size_t)(trace_end - first)) : NULL;
    after_header = after_header ? after_header + 1 : trace_end;
    const char *second = first ? trace_find_next_case(after_header, trace_end) : NULL;
    if (!first || !second || !strstr(second, "===== CASE 2/2")) {
        fprintf(stderr, "ds4-eval: trace self-test failed: embedded case marker skip\n");
        return 1;
    }

    char *copied = trace_copy_model_output(first, second);
    if (!copied || strcmp(copied, model_output) != 0) {
        fprintf(stderr, "ds4-eval: trace self-test failed: MODEL_OUTPUT byte copy\n");
        free(copied);
        return 1;
    }
    free(copied);
    return 0;
}

static int regrade_status_self_test_case(void) {
    int failed = 0;
    const eval_case integer = {.source = "AIME2025", .answer = "82"};
    char got[EVAL_ANSWER_MAX];

    if (regrade_case_outcome(&integer, "PASSED", "</think>Answer: 82",
                             got, sizeof(got)) != REGRADE_PASSED) {
        fprintf(stderr, "ds4-eval: regrade self-test failed: PASSED trace not regraded\n");
        failed++;
    }
    /* An interrupted run whose partial output happens to look correct must not
     * be counted or flagged as drift. */
    if (regrade_case_outcome(&integer, "STOPPED", "</think>Answer: 82",
                             got, sizeof(got)) != REGRADE_NOT_GRADED) {
        fprintf(stderr, "ds4-eval: regrade self-test failed: STOPPED trace was graded\n");
        failed++;
    }
    /* Legacy traces without a status line stay gradeable. */
    if (regrade_case_outcome(&integer, "", "</think>Answer: 82",
                             got, sizeof(got)) != REGRADE_PASSED) {
        fprintf(stderr, "ds4-eval: regrade self-test failed: legacy empty status not graded\n");
        failed++;
    }
    return failed;
}

static int run_extractor_self_tests(void) {
    int failed = 0;

    failed += trace_copy_self_test_case();
    failed += regrade_status_self_test_case();

    const eval_case mc = {
        .source = "SuperGPQA",
        .choice[0] = "A",
        .choice[1] = "B",
        .choice[2] = "C",
        .choice[3] = "D",
        .choice[4] = "E",
        .choice[5] = "F",
        .choice[6] = "G",
        .choice[7] = "H",
        .choice[8] = "I",
        .choice[9] = "J",
        .answer = "F",
    };
    failed += extractor_self_test_case(
        "multiple-choice prefers final answer marker",
        &mc,
        "</think>So answer is 0.716 H+/O2. That corresponds to option F.\n"
        "Thus final answer: F.</think>The visible explanation repeats the calculation.\n"
        "Answer: F",
        "F");
    failed += extractor_self_test_case(
        "multiple-choice prefers Answer-colon over later prose",
        &mc,
        "</think>Answer: F\nThis answer is final; option H is a tempting distractor.",
        "F");
    failed += extractor_self_test_case(
        "multiple-choice preserves loose-answer fallback",
        &mc,
        "</think>The answer is F. This answer is final; option H is tempting.",
        "F");

    const eval_case integer = {
        .source = "AIME2025",
        .answer = "82",
    };
    failed += extractor_self_test_case(
        "integer prefers final answer marker",
        &integer,
        "</think>I first thought the answer was 80.\nFinal answer: 082",
        "82");
    failed += extractor_self_test_case(
        "integer preserves loose-answer fallback",
        &integer,
        "</think>The answer is 082. This answer comes from AIME 2025.",
        "82");

    const eval_case compsec = {
        .source = "COMPSEC",
        .answer = "9-10",
    };
    failed += extractor_self_test_case(
        "COMPSEC prefers final answer marker",
        &compsec,
        "</think>I think the answer should be line 10, because CWE-122 may apply.\n"
        "**Answer:** 10</think>The primary write is at line 10.\n"
        "Answer: 10",
        "10");

    /* --- Regression cases for answer-extractor false negatives. These guard
     *     against the grader under-reporting model accuracy on well-formed
     *     final-answer lines. --- */

    /* Multiple choice: a standalone in-range capital that merely begins an
     * English word ("I think", "A careful") or a contraction ("I'll") must
     * not shadow the choice the model actually states later on the line.
     * Only reachable when the stray letter is itself a valid option, i.e.
     * 10-choice (A-J) cases, of which the embedded set has 24. */
    const eval_case mc_c = {
        .source = "SuperGPQA",
        .choice = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"},
        .answer = "C",
    };
    failed += extractor_self_test_case(
        "MC: leading pronoun 'I' does not shadow the chosen letter",
        &mc_c, "</think>Answer: I think it is C", "C");
    failed += extractor_self_test_case(
        "MC: contraction I'll does not shadow the chosen letter",
        &mc_c, "</think>Answer: I'll go with C.", "C");
    failed += extractor_self_test_case(
        "MC: leading article 'A' does not shadow the chosen letter",
        &mc_c, "</think>Answer: A careful reading shows C.", "C");

    const eval_case mc_i = {
        .source = "SuperGPQA",
        .choice = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"},
        .answer = "I",
    };
    failed += extractor_self_test_case(
        "MC: a genuine standalone 'I' answer is still picked",
        &mc_i, "</think>Answer: I.", "I");

    const eval_case mc4_d = {
        .source = "SuperGPQA",
        .choice = {"A", "B", "C", "D"},
        .answer = "D",
    };
    failed += extractor_self_test_case(
        "MC: out-of-range pronoun is harmless on 4-choice cases",
        &mc4_d, "</think>Answer: I think it is D", "D");

    /* A distractor the model explicitly rejects *before* stating its pick on
     * the same line must not shadow the real choice ("not B, ... D" /
     * "rules out C, leaving D"). The bare first-valid-letter rule grabbed the
     * rejected letter; a small, high-precision negation-cue skip fixes it.
     * "D, not B" is the guard against the naive "take the last letter" fix:
     * the pick is reached and accepted before the rejected distractor. #321 */
    const eval_case mc_d = {
        .source = "SuperGPQA",
        .choice = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"},
        .answer = "D",
    };
    failed += extractor_self_test_case(
        "MC: 'not B' distractor before the pick does not shadow it",
        &mc_d, "</think>Answer: It is not B, the answer is D", "D");
    failed += extractor_self_test_case(
        "MC: 'rules out C, leaving D' grades the surviving choice",
        &mc_d, "</think>Answer: rules out C, leaving D", "D");
    failed += extractor_self_test_case(
        "MC: contraction negation (isn't B) before the pick is skipped",
        &mc_d, "</think>Answer: It isn't B, so D", "D");
    failed += extractor_self_test_case(
        "MC: a leading pick followed by 'not B' is still graded as the pick",
        &mc_d, "</think>Answer: D, not B", "D");

    /* Integer: when the answer line shows the arithmetic, the graded value
     * must be the stated result (right of the last '='), not the first
     * summand/factor; digits on later lines must not leak in either. Many
     * embedded AIME2025 cases ask for a derived sum (m+n, a+b+c, ...). */
    const eval_case int_293 = {.source = "AIME2025", .answer = "293"};
    failed += extractor_self_test_case(
        "integer: sum line grades the total, not the first summand",
        &int_293, "</think>Answer: m+n = 256+37 = 293", "293");
    const eval_case int_62 = {.source = "AIME2025", .answer = "62"};
    failed += extractor_self_test_case(
        "integer: three-term sum line grades the total",
        &int_62, "</think>Answer: a+b+c = 12+25+25 = 62", "62");
    const eval_case int_81 = {.source = "AIME2025", .answer = "81"};
    failed += extractor_self_test_case(
        "integer: product-sum line grades the result, not the first factor",
        &int_81, "</think>Answer: 2*7 + 3*6 + 5*4 + 7*5 = 81", "81");
    const eval_case int_82 = {.source = "AIME2025", .answer = "82"};
    failed += extractor_self_test_case(
        "integer: digits on a later line do not override the answer line",
        &int_82, "</think>Answer: 82\nThe value 2025 is just the year.", "82");

    if (failed) return 1;
    printf("ds4-eval: answer extractor self-tests passed\n");
    return 0;
}

