static void test_thinking_checkpoint_canonical_matches_future_prompt(void) {
    /* Simulate: user sends a single message, thinking mode on, no tools.
     * Model generates reasoning + content.  The next request will drop the
     * reasoning from this turn.  Verify that:
     *   prompt_text[:-len("<think>")] + "</think>" + content + "<|eos|>"
     * equals what render_chat_prompt_text produces for the history. */

    chat_msgs prefix_msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("What is 2+2?");
    chat_msgs_push(&prefix_msgs, user1);

    /* This is what prompt_text looks like for the first generation */
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, NULL, NULL, DS4_THINK_HIGH);
    /* prompt_text should end with <think> */
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(pt_len >= 7);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 7, "<think>", 7));

    /* The model generates: reasoning + </think> + content */
    const char *reasoning = "Let me think... 2+2 = 4";
    const char *content = "The answer is 4.";

    /* Build the canonical checkpoint text (what we'd produce after canonicalization) */
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 7);  /* strip <think> */
    buf_puts(&canonical, "</think>");
    buf_puts(&canonical, content);
    buf_puts(&canonical, "<" "\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">");

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = DS4_THINK_HIGH;
    r.prompt_text = xstrdup(prompt_text);
    char *visible = build_toolless_thinking_visible_text(&r, content);
    TEST_ASSERT(visible != NULL);
    TEST_ASSERT(!strcmp(visible, canonical.ptr));
    free(visible);
    request_free(&r);

    /* Now build what the NEXT request would render: history includes this
     * assistant message, plus a new user message.  Extract just the prefix
     * up to and including the eos of the assistant turn. */
    chat_msgs history_msgs = {0};
    chat_msg h_user1 = {0};
    h_user1.role = xstrdup("user");
    h_user1.content = xstrdup("What is 2+2?");
    chat_msgs_push(&history_msgs, h_user1);
    chat_msg h_asst = {0};
    h_asst.role = xstrdup("assistant");
    h_asst.reasoning = xstrdup(reasoning);
    h_asst.content = xstrdup(content);
    chat_msgs_push(&history_msgs, h_asst);
    chat_msg h_user2 = {0};
    h_user2.role = xstrdup("user");
    h_user2.content = xstrdup("Thanks!");
    chat_msgs_push(&history_msgs, h_user2);

    char *future_prompt = render_chat_prompt_text(&history_msgs, NULL, NULL, DS4_THINK_HIGH);

    /* The future prompt should START with our canonical text */
    size_t clen = canonical.len;
    TEST_ASSERT(strlen(future_prompt) > clen);
    TEST_ASSERT(!memcmp(future_prompt, canonical.ptr, clen));

    /* And what comes after is the new user turn + assistant prefix */
    const char *rest = future_prompt + clen;
    TEST_ASSERT(strstr(rest, "Thanks!") != NULL);
    TEST_ASSERT(strstr(rest, "<think>") != NULL);  /* new turn starts thinking */

    /* Verify reasoning is NOT in the future prompt for this turn */
    const char *asst_turn = strstr(future_prompt, "<" "\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c" ">");
    TEST_ASSERT(asst_turn != NULL);
    TEST_ASSERT(strstr(future_prompt, reasoning) == NULL);  /* reasoning dropped */

    free(future_prompt);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&prefix_msgs);
    chat_msgs_free(&history_msgs);
}

static void test_thinking_canonical_empty_content(void) {
    /* Edge case: model thinks but produces empty content (e.g. tool-less
     * thinking where answer is entirely in reasoning).  Canonical should
     * still be valid: prompt_text[:-7] + "</think><|eos|>" */
    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Think about life");
    chat_msgs_push(&msgs, user);

    char *prompt_text = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_HIGH);
    size_t pt_len = strlen(prompt_text);

    /* Build canonical with empty content */
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 7);
    buf_puts(&canonical, "</think>");
    /* empty content */
    buf_puts(&canonical, "<" "\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">");

    /* Future prompt with empty content assistant message */
    chat_msgs history = {0};
    chat_msg h_u = {0};
    h_u.role = xstrdup("user");
    h_u.content = xstrdup("Think about life");
    chat_msgs_push(&history, h_u);
    chat_msg h_a = {0};
    h_a.role = xstrdup("assistant");
    h_a.reasoning = xstrdup("Deep thoughts about existence...");
    h_a.content = xstrdup("");
    chat_msgs_push(&history, h_a);
    chat_msg h_u2 = {0};
    h_u2.role = xstrdup("user");
    h_u2.content = xstrdup("Continue");
    chat_msgs_push(&history, h_u2);

    char *future = render_chat_prompt_text(&history, NULL, NULL, DS4_THINK_HIGH);
    TEST_ASSERT(strlen(future) > canonical.len);
    TEST_ASSERT(!memcmp(future, canonical.ptr, canonical.len));
    /* reasoning dropped */
    TEST_ASSERT(strstr(future, "Deep thoughts") == NULL);

    free(future);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&msgs);
    chat_msgs_free(&history);
}

static void test_thinking_canonical_multi_turn(void) {
    /* Multi-turn: 3 user messages, 2 assistant responses with reasoning.
     * Both prior assistant turns should have reasoning dropped.
     * The canonical after the SECOND generation should produce text that
     * matches the start of a 3rd-turn future prompt. */
    chat_msgs turn2_prefix = {0};
    chat_msg u1 = {0};
    u1.role = xstrdup("user");
    u1.content = xstrdup("Hello");
    chat_msgs_push(&turn2_prefix, u1);
    chat_msg a1 = {0};
    a1.role = xstrdup("assistant");
    a1.reasoning = xstrdup("first reasoning");
    a1.content = xstrdup("Hi there");
    chat_msgs_push(&turn2_prefix, a1);
    chat_msg u2 = {0};
    u2.role = xstrdup("user");
    u2.content = xstrdup("How are you?");
    chat_msgs_push(&turn2_prefix, u2);

    /* prompt_text for the 2nd generation (includes 1st assistant turn) */
    char *prompt_text = render_chat_prompt_text(&turn2_prefix, NULL, NULL, DS4_THINK_HIGH);
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 7, "<think>", 7));

    /* 1st turn reasoning is already dropped in this prompt_text */
    TEST_ASSERT(strstr(prompt_text, "first reasoning") == NULL);
    TEST_ASSERT(strstr(prompt_text, "Hi there") != NULL);

    /* After 2nd generation: canonical drops 2nd reasoning too */
    const char *content2 = "I'm doing well";
    buf canonical = {0};
    buf_append(&canonical, prompt_text, pt_len - 7);
    buf_puts(&canonical, "</think>");
    buf_puts(&canonical, content2);
    buf_puts(&canonical, "<" "\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c" ">");

    /* Future: 3rd user message arrives */
    chat_msgs future_msgs = {0};
    chat_msg fu1 = {0}; fu1.role = xstrdup("user"); fu1.content = xstrdup("Hello");
    chat_msgs_push(&future_msgs, fu1);
    chat_msg fa1 = {0}; fa1.role = xstrdup("assistant");
    fa1.reasoning = xstrdup("first reasoning");
    fa1.content = xstrdup("Hi there");
    chat_msgs_push(&future_msgs, fa1);
    chat_msg fu2 = {0}; fu2.role = xstrdup("user"); fu2.content = xstrdup("How are you?");
    chat_msgs_push(&future_msgs, fu2);
    chat_msg fa2 = {0}; fa2.role = xstrdup("assistant");
    fa2.reasoning = xstrdup("second reasoning");
    fa2.content = xstrdup(content2);
    chat_msgs_push(&future_msgs, fa2);
    chat_msg fu3 = {0}; fu3.role = xstrdup("user"); fu3.content = xstrdup("Great");
    chat_msgs_push(&future_msgs, fu3);

    char *future = render_chat_prompt_text(&future_msgs, NULL, NULL, DS4_THINK_HIGH);
    /* Both reasonings dropped */
    TEST_ASSERT(strstr(future, "first reasoning") == NULL);
    TEST_ASSERT(strstr(future, "second reasoning") == NULL);
    /* Canonical is a prefix of future */
    TEST_ASSERT(strlen(future) > canonical.len);
    TEST_ASSERT(!memcmp(future, canonical.ptr, canonical.len));

    free(future);
    buf_free(&canonical);
    free(prompt_text);
    chat_msgs_free(&turn2_prefix);
    chat_msgs_free(&future_msgs);
}

static void test_thinking_canonical_with_tools_preserves_reasoning(void) {
    /* When tools ARE present, reasoning is preserved in re-render.
     * The toolless thinking live binding should NOT fire (has_tools gate),
     * and the tool-call replay path handles it.  Verify the template
     * preserves reasoning when tool_context is true. */
    const char *tool_schemas = "{\"name\":\"bash\"}";

    chat_msgs msgs = {0};
    chat_msg u = {0};
    u.role = xstrdup("user");
    u.content = xstrdup("run ls");
    chat_msgs_push(&msgs, u);

    char *prompt_text = render_chat_prompt_text(&msgs, tool_schemas, NULL, DS4_THINK_HIGH);
    size_t pt_len = strlen(prompt_text);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 7, "<think>", 7));

    /* With tools, next render KEEPS reasoning */
    chat_msgs history = {0};
    chat_msg hu = {0}; hu.role = xstrdup("user"); hu.content = xstrdup("run ls");
    chat_msgs_push(&history, hu);
    chat_msg ha = {0}; ha.role = xstrdup("assistant");
    ha.reasoning = xstrdup("I should run bash");
    ha.content = xstrdup("Here you go");
    chat_msgs_push(&history, ha);
    chat_msg hu2 = {0}; hu2.role = xstrdup("user"); hu2.content = xstrdup("thanks");
    chat_msgs_push(&history, hu2);

    char *future = render_chat_prompt_text(&history, tool_schemas, NULL, DS4_THINK_HIGH);
    /* Reasoning IS preserved when tools present */
    TEST_ASSERT(strstr(future, "I should run bash") != NULL);
    TEST_ASSERT(strstr(future, "<think>I should run bash</think>") != NULL);

    free(future);
    free(prompt_text);
    chat_msgs_free(&msgs);
    chat_msgs_free(&history);
}

static void test_thinking_canonical_non_thinking_mode_noop(void) {
    /* When thinking is disabled (deepseek-chat), prompt_text ends with
     * </think> not <think>.  The toolless thinking live binding is a no-op
     * (early return on memcmp check). */
    chat_msgs msgs = {0};
    chat_msg u = {0};
    u.role = xstrdup("user");
    u.content = xstrdup("Hello");
    chat_msgs_push(&msgs, u);

    char *prompt_text = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_NONE);
    size_t pt_len = strlen(prompt_text);
    /* Should end with </think>, not <think> */
    TEST_ASSERT(pt_len >= 8);
    TEST_ASSERT(!memcmp(prompt_text + pt_len - 8, "</think>", 8));
    /* Does NOT end with <think> */
    TEST_ASSERT(memcmp(prompt_text + pt_len - 7, "<think>", 7) != 0);

    free(prompt_text);
    chat_msgs_free(&msgs);
}

static void ds4_server_unit_tests_run(void) {
    test_batched_prefill_round_robin();
    test_mixed_prefill_quantum_option();
    test_batched_live_continuation_slot_binding();
    test_request_defaults_use_min_p_filtering();
    test_reasoning_effort_mapping();
    test_model_alias_thinking_controls();
    test_api_thinking_controls_parse();
    test_render_think_max_prompt_prefix();
    test_render_non_thinking_prompt_closes_think();
    test_render_drops_old_reasoning_without_tools();
    test_render_preserves_reasoning_with_tools();
    test_render_chat_prompt_text_renders_tools_before_system();
    test_render_glm_chat_prompt_text();
    test_render_glm_drops_old_reasoning_without_tools();
    test_render_glm_preserves_reasoning_with_tools();
    test_tool_schema_order_from_anthropic_schema();
    test_tool_schema_order_from_openai_tools();
    test_tool_schema_order_from_responses_tool_search();
    test_responses_function_named_tool_search_stays_function_call();
    test_responses_namespace_tool_schemas_restore_wire_namespace();
    test_responses_input_tool_search_output_loads_tools();
    test_responses_input_tool_search_output_rejects_bad_tools();
    test_responses_input_function_call_namespace_round_trips_to_dsml();
    test_responses_output_sends_tool_search_call_item();
    test_dsml_tool_args_preserve_call_order();
    test_openai_tool_args_preserve_call_order();
    test_anthropic_thinking_and_tool_args_preserve_call_order();
    test_context_length_error_uses_protocol_standard_shape();
    test_cors_headers_are_opt_in();
    test_cors_preflight_response_is_no_content();
    test_cors_sse_headers();
    test_anthropic_live_stream_sends_incremental_blocks();
    test_anthropic_stream_reroutes_second_reasoning_pass();
    test_anthropic_usage_reports_cache_details();
    test_anthropic_tool_stream_sends_live_tool_use();
    test_openai_tool_stream_sends_incremental_text();
    test_openai_stream_reroutes_second_reasoning_pass();
    test_openai_stream_usage_reports_cache_details();
    test_responses_usage_reports_cache_details();
    test_openai_chat_stream_splits_reasoning_without_tools();
    test_openai_tool_stream_sends_partial_arguments();
    test_openai_glm_tool_stream_suppresses_raw_tool_call();
    test_openai_tool_stream_waits_for_incomplete_tool_tags();
    test_openai_tool_stream_sends_partial_raw_arguments();
    test_openai_tool_stream_holds_partial_dsml_entities();
    test_openai_tool_stream_holds_partial_utf8_arguments();
    test_openai_tool_stream_handles_multiple_calls();
    test_streaming_holds_partial_utf8();
    test_parse_short_dsml_and_canonical_suffix();
    test_parse_glm_tool_call_message();
    test_dsml_parser_recovers_loose_nested_parameters();
    test_dsml_repair_produces_parseable_calls();
    test_tool_parse_failure_returns_recoverable_finish();
    test_invalid_dsml_tool_error_suffix_includes_system_prompt();
    test_invalid_glm_tool_error_suffix();
    test_thinking_dsml_is_not_executable_before_think_close();
    test_thinking_dsml_after_think_close_is_executable();
    test_tool_checkpoint_suffix_is_future_prompt_canonical();
    test_glm_tool_checkpoint_suffix_is_canonical();
    test_tool_checkpoint_minifies_json_parameters();
    test_tool_memory_replays_sampled_dsml();
    test_anthropic_tool_memory_replays_sampled_dsml();
    test_anthropic_live_tail_renders_tool_results_only();
    test_anthropic_tool_result_id_validation();
    test_anthropic_full_replay_allows_unknown_live_id();
    test_anthropic_tool_use_parses_before_role();
    test_tool_checkpoint_canonicalization_gate_exact_replay();
    test_responses_live_tail_renders_tool_outputs_only();
    test_responses_tool_output_id_validation();
    test_responses_stateless_tool_replay_requires_reasoning();
    test_responses_visible_suffix_matches_client_replay();
    test_exact_dsml_tool_replay_can_be_disabled();
    test_dsml_decode_state_separates_structure_and_payload();
    test_tool_memory_max_ids_prunes_oldest();
    test_kv_tool_map_filters_by_dsml_text();
    test_kv_tool_map_restores_before_prompt_render();
    test_thinking_checkpoint_canonical_matches_future_prompt();
    test_thinking_canonical_empty_content();
    test_thinking_canonical_multi_turn();
    test_thinking_canonical_with_tools_preserves_reasoning();
    test_thinking_canonical_non_thinking_mode_noop();
    test_tool_separator_whitespace_is_not_content();
    test_dsml_prompt_escapes_tool_supplied_text();
    test_stop_list_parses_all_sequences();
    test_stop_list_streaming_holds_and_trims_stop_text();
    test_json_skip_has_nesting_limit();
    test_request_parsers_reject_malformed_duplicate_owned_fields();
    test_json_parser_handles_tool_heavy_requests();
    test_json_string_handles_surrogates();
    test_json_int_handles_non_finite_values();
    test_tool_history_validation_handles_large_replays();
    test_model_metadata_clamps_completion_to_context();
    test_live_prefix_rewind_target();
    test_client_socket_nonblocking_flag();
    test_client_disconnect_probe();
    test_cancelled_progress_callback_is_inert();
    test_waiting_job_cancels_on_client_close();
    test_cancel_unlinks_queued_jobs();
    test_cancel_detaches_assigned_job();
    test_cancel_running_job_keeps_worker_ownership();
    test_cancel_withdraws_only_pending_decode();
    test_thinking_state_tracks_prompt_and_generated_tags();
    test_thinking_checkpoint_remember_gate();
    test_tool_marker_state_ignores_orphan_end();
    test_canonical_rewrite_rebuilds_when_live_tail_changes();
    test_kv_cache_store_len_uses_configured_boundary();
    test_kv_cache_chat_anchor_uses_last_user_before_assistant();
    test_kv_cache_chat_anchor_ignores_multiturn_tail();
    test_kv_cache_continued_uses_aligned_frontiers();
    test_kv_cache_cold_store_suppresses_duplicate_continued_boundary();
    test_kv_cache_file_size_must_fit_budget();
    test_sha1_bytes_hex_matches_known_vector();
    test_kv_cache_lookup_uses_longest_text_prefix();
    test_kv_cache_lookup_rejects_wrong_model();
    test_kv_cache_lookup_rejects_stale_payload_abi();
    test_kv_cache_eviction_values_fresh_snapshots();
    test_kv_cache_eviction_prefers_anchor_reason();
    test_kv_cache_eviction_makes_room_before_store();
    test_kv_cache_eviction_ignores_oversize_incoming();
    test_kv_cache_eviction_prefers_superseded_continued_prefix();
    test_kv_cache_eviction_keeps_smaller_context_prefix();
    test_kv_cache_eviction_score_decays_stale_hits();
    test_kv_cache_eviction_decayed_hits_tie_break_by_age();
    test_kv_cache_eviction_keeps_aligned_continued_frontiers();
}

#ifndef DS4_SERVER_TEST_NO_MAIN
int main(void) {
    ds4_server_unit_tests_run();
    if (test_failures) {
        fprintf(stderr, "ds4-server tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ds4-server tests: ok");
    return 0;
}
#endif

