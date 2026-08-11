static void test_render_non_thinking_prompt_closes_think(void) {
    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_NONE);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, ds4_think_max_prefix()) == NULL);
    TEST_ASSERT(strstr(prompt, "<｜User｜>Hello<｜Assistant｜></think>") != NULL);
    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_render_drops_old_reasoning_without_tools(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("first");
    chat_msgs_push(&msgs, user1);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup("old hidden reasoning");
    assistant.content = xstrdup("first answer");
    chat_msgs_push(&msgs, assistant);
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("second");
    chat_msgs_push(&msgs, user2);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "old hidden reasoning") == NULL);
    TEST_ASSERT(strstr(prompt, "<｜Assistant｜></think>first answer") != NULL);
    TEST_ASSERT(strstr(prompt, "<｜User｜>second<｜Assistant｜><think>") != NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_render_preserves_reasoning_with_tools(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("first");
    chat_msgs_push(&msgs, user1);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup("tool reasoning");
    assistant.content = xstrdup("");
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    char *prompt = render_chat_prompt_text(&msgs, "{}", NULL, DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "<think>tool reasoning</think>") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_result>/tmp</tool_result>") != NULL);
    free(prompt);

    prompt = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "<think>tool reasoning</think>") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_result>/tmp</tool_result>") != NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_render_chat_prompt_text_renders_tools_before_system(void) {
    /* The tool-schema block must sit at the head of the system region so the
     * client's system content stays at the tail, right before <｜User｜>.
     * That keeps a per-request dynamic tail (e.g. a timestamp) out of the
     * cached prefix without losing the tool schemas to the trim. */
    chat_msgs msgs = {0};
    chat_msg sys = {0};
    sys.role = xstrdup("system");
    sys.content = xstrdup("CLIENT_SYSTEM_MARKER");
    chat_msgs_push(&msgs, sys);
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("hello");
    chat_msgs_push(&msgs, user);

    char *prompt = render_chat_prompt_text(&msgs, "TOOL_SCHEMA_MARKER", NULL,
                                           DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    const char *tools  = strstr(prompt, "## Tools");
    const char *client = strstr(prompt, "CLIENT_SYSTEM_MARKER");
    const char *user_m = strstr(prompt, "<｜User｜>");
    TEST_ASSERT(tools && client && user_m);
    TEST_ASSERT(tools  < client);
    TEST_ASSERT(client < user_m);
    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_render_glm_chat_prompt_text(void) {
    chat_msgs msgs = {0};
    chat_msg sys = {0};
    sys.role = xstrdup("system");
    sys.content = xstrdup("You are terse.");
    chat_msgs_push(&msgs, sys);
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("Hello");
    chat_msgs_push(&msgs, user);

    tool_schema_orders orders = make_bash_order();
    const char *tool_schemas =
        "{\"name\":\"bash\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{}}}}";
    char *prompt = render_chat_prompt_text_for_syntax(
        SERVER_MODEL_SYNTAX_GLM, &msgs, tool_schemas, &orders, DS4_THINK_HIGH);

    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(!strncmp(prompt, "[gMASK]<sop>", strlen("[gMASK]<sop>")));
    TEST_ASSERT(strstr(prompt, "<|system|>Reasoning Effort: High") != NULL);
    TEST_ASSERT(strstr(prompt, "<tools>") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_call>{function-name}") != NULL);
    TEST_ASSERT(strstr(prompt, "<|system|>You are terse.") != NULL);
    TEST_ASSERT(strstr(prompt, "<|user|>Hello<|assistant|><think>") != NULL);
    TEST_ASSERT(strstr(prompt, "DSML") == NULL);

    free(prompt);
    tool_schema_orders_free(&orders);
    chat_msgs_free(&msgs);
}

static void test_render_glm_drops_old_reasoning_without_tools(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("first");
    chat_msgs_push(&msgs, user1);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup("old hidden reasoning");
    assistant.content = xstrdup("first answer");
    chat_msgs_push(&msgs, assistant);
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("second");
    chat_msgs_push(&msgs, user2);

    char *prompt = render_chat_prompt_text_for_syntax(
        SERVER_MODEL_SYNTAX_GLM, &msgs, NULL, NULL, DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "old hidden reasoning") == NULL);
    TEST_ASSERT(strstr(prompt, "<|assistant|><think></think>first answer") != NULL);
    TEST_ASSERT(strstr(prompt, "<|user|>second<|assistant|><think>") != NULL);

    free(prompt);
    chat_msgs_free(&msgs);
}

static void test_render_glm_preserves_reasoning_with_tools(void) {
    chat_msgs msgs = {0};
    chat_msg user1 = {0};
    user1.role = xstrdup("user");
    user1.content = xstrdup("first");
    chat_msgs_push(&msgs, user1);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup("tool reasoning");
    assistant.content = xstrdup("");
    tool_call tc = {0};
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    tool_schema_orders orders = make_bash_order();
    char *prompt = render_chat_prompt_text_for_syntax(
        SERVER_MODEL_SYNTAX_GLM, &msgs, NULL, &orders, DS4_THINK_HIGH);
    TEST_ASSERT(prompt != NULL);
    TEST_ASSERT(strstr(prompt, "<think>tool reasoning</think>") != NULL);
    TEST_ASSERT(strstr(prompt, "<tool_call>bash") != NULL);
    TEST_ASSERT(strstr(prompt, "<|observation|><tool_response>/tmp</tool_response>") != NULL);

    free(prompt);
    tool_schema_orders_free(&orders);
    chat_msgs_free(&msgs);
}

static void test_dsml_tool_args_preserve_call_order(void) {
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_dsml_tool_calls_text(&b, &calls);
    const char *command = strstr(b.ptr, "name=\"command\"");
    const char *description = strstr(b.ptr, "name=\"description\"");
    const char *timeout = strstr(b.ptr, "name=\"timeout\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(command < timeout);
    buf_free(&b);
    tool_calls_free(&calls);
}

static void test_openai_tool_args_preserve_call_order(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.tool_orders = make_bash_order();
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_tool_calls_json(&b, &calls, "test", &r.tool_orders);
    const char *command = strstr(b.ptr, "\\\"command\\\"");
    const char *description = strstr(b.ptr, "\\\"description\\\"");
    const char *timeout = strstr(b.ptr, "\\\"timeout\\\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(command < timeout);
    buf_free(&b);
    tool_calls_free(&calls);
    request_free(&r);
}

static void test_anthropic_thinking_and_tool_args_preserve_call_order(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.tool_orders = make_bash_order();
    tool_calls calls = make_swapped_bash_call();
    buf b = {0};
    append_anthropic_content(&b, "done", "thinking text", &calls, "msg_1", &r.tool_orders);
    const char *thinking = strstr(b.ptr, "\"type\":\"thinking\"");
    const char *text = strstr(b.ptr, "\"type\":\"text\"");
    const char *tool = strstr(b.ptr, "\"type\":\"tool_use\"");
    const char *command = strstr(b.ptr, "\"command\"");
    const char *description = strstr(b.ptr, "\"description\"");
    TEST_ASSERT(thinking != NULL);
    TEST_ASSERT(text != NULL);
    TEST_ASSERT(tool != NULL);
    TEST_ASSERT(thinking < text);
    TEST_ASSERT(text < tool);
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(description < command);
    buf_free(&b);
    tool_calls_free(&calls);
    request_free(&r);
}

static void test_parse_short_dsml_and_canonical_suffix(void) {
    const char *generated =
        "<think>need a tool</think>"
        "<DSML｜tool_calls>\n"
        "<DSML｜invoke name=\"bash\">\n"
        "<DSML｜parameter name=\"description\" string=\"true\">list files</DSML｜parameter>\n"
        "<DSML｜parameter name=\"command\" string=\"true\">ls -la</DSML｜parameter>\n"
        "</DSML｜invoke>\n"
        "</DSML｜tool_calls>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need a tool"));
    TEST_ASSERT(content && content[0] == '\0');
    TEST_ASSERT(calls.len == 1);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = DS4_THINK_HIGH;
    r.tool_orders = make_bash_order();
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    const char *command = strstr(suffix, "name=\"command\"");
    const char *description = strstr(suffix, "name=\"description\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(description < command);
    TEST_ASSERT(strstr(suffix, "</think>") != NULL);
    TEST_ASSERT(strstr(suffix, "<｜end▁of▁sentence｜>") != NULL);

    free(suffix);
    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    request_free(&r);
}

static void test_parse_glm_tool_call_message(void) {
    const char *generated =
        "<think>need bash</think>OK\n\n"
        "<tool_call>bash"
        "<arg_key>command</arg_key><arg_value>echo hi</arg_value>"
        "<arg_key>timeout</arg_key><arg_value>10</arg_value>"
        "</tool_call>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};

    TEST_ASSERT(parse_generated_message_ex_for_syntax(
        SERVER_MODEL_SYNTAX_GLM, generated, true,
        &content, &reasoning, &calls));
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need bash"));
    TEST_ASSERT(content && !strcmp(content, "OK"));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"echo hi\"") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"timeout\": \"10\"") != NULL);
    TEST_ASSERT(calls.raw_tool_text &&
                !strncmp(calls.raw_tool_text, "\n\n<tool_call>bash",
                         strlen("\n\n<tool_call>bash")));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

static void test_dsml_parser_recovers_loose_nested_parameters(void) {
    const char *generated =
        "review done\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n"
        DS4_PARAM_START " name=\"path\">/private/tmp/tetris.c" DS4_PARAM_END "\n"
        DS4_PARAM_START " name=\"edits\">\n"
        DS4_PARAM_START " name=\"oldText\" string=\"true\">old &lt;text&gt;" DS4_PARAM_END "\n"
        DS4_PARAM_START " name=\"newText\" string=\"true\">new text" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(content && !strcmp(content, "review done"));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "edit"));
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"path\": \"/private/tmp/tetris.c\"") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"edits\": {") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"oldText\":\"old <text>\"") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"newText\":\"new text\"") != NULL);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

/* Verify that try_repair_dsml + parse_generated_message produces structurally
   valid tool calls for all three DSML styles and multiple truncation scenarios.
   Balanced but malformed DSML is not repaired: the model must retry it.
   This tests repair ACCURACY, not just that it doesn't crash. */
static void test_dsml_repair_produces_parseable_calls(void) {
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    buf repaired = {0};

    /* === TEST 1: Full DSML - missing </tool_calls> === */
    {
        const char *broken =
            "thinking done\n\n"
            DS4_TOOL_CALLS_START "\n"
            DS4_INVOKE_START " name=\"bash\">\n"
            DS4_PARAM_START " name=\"command\" string=\"true\">ls -la" DS4_PARAM_END "\n"
            DS4_INVOKE_END "\n";
        /* Missing: DS4_TOOL_CALLS_END */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"ls -la\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 2: Full DSML - missing </invoke> and </tool_calls> === */
    {
        const char *broken =
            "\n\n"
            DS4_TOOL_CALLS_START "\n"
            DS4_INVOKE_START " name=\"edit\">\n"
            DS4_PARAM_START " name=\"path\" string=\"true\">/tmp/test.c" DS4_PARAM_END "\n";
        /* Missing: DS4_INVOKE_END, DS4_TOOL_CALLS_END */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "edit"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"path\": \"/tmp/test.c\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 3: Full DSML - missing </parameter> === */
    {
        const char *broken =
            "\n\n"
            DS4_TOOL_CALLS_START "\n"
            DS4_INVOKE_START " name=\"bash\">\n"
            DS4_PARAM_START " name=\"command\" string=\"true\">echo hello";
        /* Missing: DS4_PARAM_END, DS4_INVOKE_END, DS4_TOOL_CALLS_END */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"echo hello\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 4: Short DSML - missing closing tags === */
    {
        const char *broken =
            "\n\n"
            DS4_TOOL_CALLS_START_SHORT "\n"
            DS4_INVOKE_START_SHORT " name=\"write_file\">\n"
            DS4_PARAM_START_SHORT " name=\"path\" string=\"true\">/tmp/out.txt" DS4_PARAM_END_SHORT "\n"
            DS4_PARAM_START_SHORT " name=\"content\" string=\"true\">hello world" DS4_PARAM_END_SHORT "\n"
            DS4_INVOKE_END_SHORT "\n";
        /* Missing: DS4_TOOL_CALLS_END_SHORT */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "write_file"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"path\": \"/tmp/out.txt\"") != NULL);
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"content\": \"hello world\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 5: Plain XML - missing closing tags === */
    {
        const char *broken =
            "\n\n"
            "<tool_calls>\n"
            "<invoke name=\"execute_command\">\n"
            "<parameter name=\"command\" string=\"true\">pwd</parameter>\n"
            "</invoke>\n";
        /* Missing: </tool_calls> */

        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken, strlen(broken), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, false, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "execute_command"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"pwd\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    /* === TEST 6: Balanced text should NOT be modified === */
    {
        const char *balanced =
            "\n\n"
            DS4_TOOL_CALLS_START "\n"
            DS4_INVOKE_START " name=\"bash\">\n"
            DS4_PARAM_START " name=\"command\" string=\"true\">ls" DS4_PARAM_END "\n"
            DS4_INVOKE_END "\n"
            DS4_TOOL_CALLS_END;

        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(balanced, strlen(balanced), &repaired));
        /* No repair needed */
    }

    /* === TEST 7: No DSML tags should return false === */
    {
        const char *no_dsml = "just plain text, no tools";
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(no_dsml, strlen(no_dsml), &repaired));
    }

    /* === TEST 8: Balanced DSML with no invoke is not repaired === */
    {
        const char *balanced_no_invoke =
            "Let me analyze this.\n\n"
            DS4_TOOL_CALLS_START
            "The write tool truncates this too, at what looks like the same content location."
            DS4_TOOL_CALLS_END;
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(balanced_no_invoke, strlen(balanced_no_invoke), &repaired));
    }

    /* === TEST 9: Balanced short DSML with no invoke is not repaired === */
    {
        const char *balanced_short_no_invoke =
            "thinking...\n\n"
            DS4_TOOL_CALLS_START_SHORT
            "some content here"
            DS4_TOOL_CALLS_END_SHORT;
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(balanced_short_no_invoke, strlen(balanced_short_no_invoke), &repaired));
    }

    /* === TEST 10: Balanced plain XML DSML with no invoke is not repaired === */
    {
        const char *balanced_xml_no_invoke =
            "Let me think.\n\n"
            "<tool_calls>"
            "I need to use a tool but I don't know which one."
            "</tool_calls>";
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(balanced_xml_no_invoke, strlen(balanced_xml_no_invoke), &repaired));
    }

    /* === TEST 11: DSML mentioned inside thinking is not repaired === */
    {
        const char *thinking_quote =
            "<think>The protocol uses "
            DS4_TOOL_CALLS_START
            "some explanatory text"
            DS4_TOOL_CALLS_END
            ", but this is only a quote.</think>\nFinal answer.";
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(thinking_quote, strlen(thinking_quote), &repaired));
    }

    /* === TEST 12: Extra closing tags are unrecoverable, not truncation === */
    {
        const char *orphan_close =
            "done\n\n"
            DS4_TOOL_CALLS_START
            DS4_TOOL_CALLS_END
            DS4_TOOL_CALLS_END;
        buf_free(&repaired);
        TEST_ASSERT(!try_repair_dsml(orphan_close, strlen(orphan_close), &repaired));
    }

    /* === TEST 13: Real DSML after thinking still repairs normally === */
    {
        const char *broken_after_think =
            "<think>"
            DS4_TOOL_CALLS_START
            "quoted DSML, not executable"
            DS4_TOOL_CALLS_END
            "</think>\n\n"
            DS4_TOOL_CALLS_START "\n"
            DS4_INVOKE_START " name=\"bash\">\n"
            DS4_PARAM_START " name=\"command\" string=\"true\">date" DS4_PARAM_END "\n"
            DS4_INVOKE_END "\n";
        buf_free(&repaired);
        TEST_ASSERT(try_repair_dsml(broken_after_think, strlen(broken_after_think), &repaired));
        TEST_ASSERT(parse_generated_message_ex(repaired.ptr, true, &content, &reasoning, &calls));
        TEST_ASSERT(calls.len == 1);
        TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
        TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"date\"") != NULL);
        free(content); free(reasoning); tool_calls_free(&calls);
    }

    buf_free(&repaired);
}

static void test_tool_parse_failure_returns_recoverable_finish(void) {
    const char *generated =
        "trying a tool\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START ">\n"
        DS4_TOOL_CALLS_END;

    char err[128] = {0};
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    const char *finish = "tool_calls";
    bool recovered = false;

    TEST_ASSERT(!parse_generated_message_for_response(generated,
                                                       true,
                                                       true,
                                                       false,
                                                       &finish,
                                                       err,
                                                       sizeof(err),
                                                       &content,
                                                       &reasoning,
                                                       &calls,
                                                       &recovered));
    TEST_ASSERT(recovered);
    TEST_ASSERT(!strcmp(finish, "stop"));
    TEST_ASSERT(!strcmp(err, "invalid tool call"));
    TEST_ASSERT(content && strstr(content, DS4_TOOL_CALLS_START) != NULL);
    TEST_ASSERT(reasoning == NULL);
    TEST_ASSERT(calls.len == 0);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

static void test_invalid_dsml_tool_error_suffix_includes_system_prompt(void) {
    request r = {0};
    r.think_mode = DS4_THINK_HIGH;
    r.prompt_text = xstrdup(
        "<｜begin▁of▁sentence｜>"
        "## Tools\nschema\n\nSystem rule\n\n"
        "<｜User｜>Hi<｜Assistant｜><think>");
    thinking_state st = {.inside = true};

    char *suffix = build_invalid_dsml_tool_error_suffix(&r, &st, "missing invoke name");
    TEST_ASSERT(suffix != NULL);
    TEST_ASSERT(strstr(suffix, "</think><｜end▁of▁sentence｜><｜User｜><tool_result>") == suffix);
    TEST_ASSERT(strstr(suffix, "Tool error: invalid DSML tool call: missing invoke name") != NULL);
    TEST_ASSERT(strstr(suffix, "The previous assistant output was not executed") != NULL);
    TEST_ASSERT(strstr(suffix, "System prompt reminder:\n## Tools\nschema\n\nSystem rule") != NULL);
    TEST_ASSERT(strstr(suffix, "<｜User｜>Hi") == NULL);
    TEST_ASSERT(strstr(suffix, "</tool_result><｜Assistant｜><think>") != NULL);

    free(suffix);
    free(r.prompt_text);
}

static void test_invalid_glm_tool_error_suffix(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.model_syntax = SERVER_MODEL_SYNTAX_GLM;
    r.think_mode = DS4_THINK_HIGH;
    thinking_state st = {.inside = true};

    char *suffix = build_invalid_tool_call_error_suffix(&r, &st,
                                                        "missing arg_value");
    TEST_ASSERT(suffix != NULL);
    TEST_ASSERT(strstr(suffix, "</think><|observation|><tool_response>") == suffix);
    TEST_ASSERT(strstr(suffix,
                       "Tool error: invalid GLM tool call: missing arg_value") != NULL);
    TEST_ASSERT(strstr(suffix, "</tool_response><|assistant|><think>") != NULL);
    TEST_ASSERT(strstr(suffix, "DSML") == NULL);
    TEST_ASSERT(strstr(suffix, "<｜end▁of▁sentence｜>") == NULL);

    free(suffix);
    request_free(&r);
}

static void test_thinking_dsml_is_not_executable_before_think_close(void) {
    const char *generated =
        "<think>I might mention a malformed or tentative tool call here:\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"bash\">\n"
        DS4_PARAM_START " name=\"command\" string=\"true\">true" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END
        "\nBut it is still reasoning, not an assistant action.</think>Final answer.";

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, true,
                                           &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 0);
    TEST_ASSERT(reasoning && strstr(reasoning, DS4_TOOL_CALLS_START) != NULL);
    TEST_ASSERT(content && !strcmp(content, "Final answer."));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

static void test_thinking_dsml_after_think_close_is_executable(void) {
    const char *generated =
        "<think>need a shell check</think>\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"bash\">\n"
        DS4_PARAM_START " name=\"command\" string=\"true\">pwd" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, true,
                                           &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(reasoning && !strcmp(reasoning, "need a shell check"));
    TEST_ASSERT(content && content[0] == '\0');
    TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "bash"));
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"command\": \"pwd\"") != NULL);

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
}

static void test_tool_checkpoint_suffix_is_future_prompt_canonical(void) {
    tool_schema_orders orders = make_bash_order();
    const char *tool_schemas =
        "{\"name\":\"bash\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{},\"description\":{},\"timeout\":{}}}}";

    chat_msgs prefix_msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("inspect");
    chat_msgs_push(&prefix_msgs, user);
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, tool_schemas,
                                                &orders, DS4_THINK_HIGH);

    const char *generated =
        "need a tool</think>\n\n"
        DS4_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">cd /tmp && git diff 2>/dev/null</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"timeout\" string=\"false\">10</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(strstr(calls.v[0].arguments, "cd /tmp && git diff 2>/dev/null") != NULL);
    TEST_ASSERT(strstr(calls.v[0].arguments, "&amp;&amp;") == NULL);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = DS4_THINK_HIGH;
    r.tool_orders = orders;
    memset(&orders, 0, sizeof(orders));
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    TEST_ASSERT(strstr(suffix, "cd /tmp && git diff 2>/dev/null") != NULL);
    TEST_ASSERT(strstr(suffix, "&amp;&amp;") == NULL);
    TEST_ASSERT(strstr(suffix, "2&gt;/dev/null") == NULL);
    buf canonical = {0};
    buf_puts(&canonical, prompt_text);
    buf_puts(&canonical, suffix);

    chat_msgs history_msgs = {0};
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("inspect");
    chat_msgs_push(&history_msgs, user2);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    assistant.calls = calls;
    memset(&calls, 0, sizeof(calls));
    chat_msgs_push(&history_msgs, assistant);
    char *future_prompt = render_chat_prompt_text(&history_msgs, tool_schemas,
                                                  &r.tool_orders, DS4_THINK_HIGH);

    TEST_ASSERT(!strcmp(canonical.ptr, future_prompt));

    free(future_prompt);
    buf_free(&canonical);
    free(suffix);
    free(prompt_text);
    free(content);
    free(reasoning);
    chat_msgs_free(&history_msgs);
    chat_msgs_free(&prefix_msgs);
    tool_calls_free(&calls);
    request_free(&r);
    tool_schema_orders_free(&orders);
}

static void test_glm_tool_checkpoint_suffix_is_canonical(void) {
    tool_schema_orders orders = make_bash_order();
    const char *tool_schemas =
        "{\"name\":\"bash\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"command\":{},\"description\":{}}}}";

    chat_msgs prefix_msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("inspect");
    chat_msgs_push(&prefix_msgs, user);
    char *prompt_text = render_chat_prompt_text_for_syntax(
        SERVER_MODEL_SYNTAX_GLM, &prefix_msgs, tool_schemas,
        &orders, DS4_THINK_HIGH);

    const char *generated =
        "need bash</think>done\n\n"
        "<tool_call>bash"
        "<arg_key>description</arg_key><arg_value>list files</arg_value>"
        "<arg_key>command</arg_key><arg_value>ls -la</arg_value>"
        "</tool_call>";
    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex_for_syntax(
        SERVER_MODEL_SYNTAX_GLM, generated, false,
        &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.model_syntax = SERVER_MODEL_SYNTAX_GLM;
    r.think_mode = DS4_THINK_HIGH;
    r.tool_orders = orders;
    memset(&orders, 0, sizeof(orders));

    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    TEST_ASSERT(strstr(suffix, "need bash</think>done\n\n<tool_call>bash") != NULL);
    TEST_ASSERT(strstr(suffix, "<｜end▁of▁sentence｜>") == NULL);
    TEST_ASSERT(strstr(suffix, "DSML") == NULL);

    buf canonical = {0};
    buf_puts(&canonical, prompt_text);
    buf_puts(&canonical, suffix);

    chat_msgs history_msgs = {0};
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("inspect");
    chat_msgs_push(&history_msgs, user2);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    assistant.calls = calls;
    memset(&calls, 0, sizeof(calls));
    chat_msgs_push(&history_msgs, assistant);

    char *future_prompt = render_chat_prompt_text_for_syntax(
        SERVER_MODEL_SYNTAX_GLM, &history_msgs, tool_schemas,
        &r.tool_orders, DS4_THINK_HIGH);
    TEST_ASSERT(!strcmp(canonical.ptr, future_prompt));

    free(future_prompt);
    buf_free(&canonical);
    free(suffix);
    free(prompt_text);
    free(content);
    free(reasoning);
    chat_msgs_free(&history_msgs);
    chat_msgs_free(&prefix_msgs);
    tool_calls_free(&calls);
    request_free(&r);
    tool_schema_orders_free(&orders);
}

static void test_tool_checkpoint_minifies_json_parameters(void) {
    tool_schema_orders orders = {0};
    tool_schema_orders_add_json(&orders,
        "{\"name\":\"edit\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{},\"edits\":{}}}}");
    const char *tool_schemas =
        "{\"name\":\"edit\",\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"path\":{},\"edits\":{}}}}";

    chat_msgs prefix_msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("edit");
    chat_msgs_push(&prefix_msgs, user);
    char *prompt_text = render_chat_prompt_text(&prefix_msgs, tool_schemas,
                                                &orders, DS4_THINK_HIGH);

    const char *generated =
        "need edit</think>\n\n"
        DS4_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"edit\">\n"
        "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/file</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"edits\" string=\"false\">"
        "[{\"oldText\": \"status=created\", \"newText\": \"status=created\\nstatus2=resumed\"}]"
        "</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);

    request r;
    request_init(&r, REQ_CHAT, 128);
    r.think_mode = DS4_THINK_HIGH;
    r.tool_orders = orders;
    memset(&orders, 0, sizeof(orders));
    char *suffix = build_tool_checkpoint_suffix(&r, content, reasoning, &calls);
    buf canonical = {0};
    buf_puts(&canonical, prompt_text);
    buf_puts(&canonical, suffix);

    chat_msgs history_msgs = {0};
    chat_msg user2 = {0};
    user2.role = xstrdup("user");
    user2.content = xstrdup("edit");
    chat_msgs_push(&history_msgs, user2);
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    assistant.calls = calls;
    memset(&calls, 0, sizeof(calls));
    chat_msgs_push(&history_msgs, assistant);
    char *future_prompt = render_chat_prompt_text(&history_msgs, tool_schemas,
                                                  &r.tool_orders, DS4_THINK_HIGH);

    TEST_ASSERT(!strcmp(canonical.ptr, future_prompt));

    free(future_prompt);
    buf_free(&canonical);
    free(suffix);
    free(prompt_text);
    free(content);
    free(reasoning);
    chat_msgs_free(&history_msgs);
    chat_msgs_free(&prefix_msgs);
    tool_calls_free(&calls);
    request_free(&r);
    tool_schema_orders_free(&orders);
}

static void test_tool_memory_replays_sampled_dsml(void) {
    const char *generated =
        "<think>need shell</think>\n\n"
        DS4_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"timeout\" string=\"false\">10</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">list files</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls sampled = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, false, &content, &reasoning, &sampled));
    TEST_ASSERT(sampled.len == 1);

    server s;
    memset(&s, 0, sizeof(s));
    pthread_mutex_init(&s.tool_mu, NULL);
    assign_tool_call_ids(&s, &sampled, API_OPENAI);
    TEST_ASSERT(sampled.v[0].id != NULL);
    TEST_ASSERT(!strncmp(sampled.v[0].id, "call_", 5));
    tool_memory_remember(&s, &sampled);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    assistant.reasoning = xstrdup(reasoning ? reasoning : "");
    assistant.content = xstrdup(content ? content : "");
    tool_call tc = {0};
    tc.id = xstrdup(sampled.v[0].id);
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"description\":\"list files\",\"command\":\"ls -la\",\"timeout\":10}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&s, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_tool_text != NULL);
    TEST_ASSERT(stats.mem == 1);
    TEST_ASSERT(stats.disk == 0);
    TEST_ASSERT(stats.canonical == 0);
    TEST_ASSERT(stats.missing_ids == 0);
    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_HIGH);
    const char *command = strstr(prompt, "name=\"command\"");
    const char *timeout = strstr(prompt, "name=\"timeout\"");
    const char *description = strstr(prompt, "name=\"description\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(timeout != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(command < timeout);
    TEST_ASSERT(timeout < description);

    free(prompt);
    chat_msgs_free(&msgs);
    free(content);
    free(reasoning);
    tool_calls_free(&sampled);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_anthropic_tool_memory_replays_sampled_dsml(void) {
    const char *sampled_dsml =
        "\n\n" DS4_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"Bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">ls -la</｜DSML｜parameter>\n"
        "<｜DSML｜parameter name=\"description\" string=\"true\">list files</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        DS4_TOOL_CALLS_END;

    server s;
    memset(&s, 0, sizeof(s));
    pthread_mutex_init(&s.tool_mu, NULL);
    tool_memory_put(&s, "toolu_exact", sampled_dsml);

    const char *json =
        "["
        "{\"role\":\"assistant\",\"content\":["
        "{\"type\":\"tool_use\",\"id\":\"toolu_exact\",\"name\":\"Bash\","
        "\"input\":{\"description\":\"list files\",\"command\":\"ls -la\"}}"
        "]},"
        "{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_exact\",\"content\":\"ok\"}"
        "]}"
        "]";
    const char *p = json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 2);
    TEST_ASSERT(msgs.v[1].tool_call_id && !strcmp(msgs.v[1].tool_call_id, "toolu_exact"));

    stop_list ids = {0};
    collect_tool_call_ids(&msgs, &ids);
    TEST_ASSERT(id_list_contains(&ids, "toolu_exact"));
    id_list_free(&ids);

    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&s, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_tool_text != NULL);
    TEST_ASSERT(stats.mem == 1);
    TEST_ASSERT(stats.canonical == 0);

    char *prompt = render_chat_prompt_text(&msgs, NULL, NULL, DS4_THINK_HIGH);
    const char *command = strstr(prompt, "name=\"command\"");
    const char *description = strstr(prompt, "name=\"description\"");
    TEST_ASSERT(command != NULL);
    TEST_ASSERT(description != NULL);
    TEST_ASSERT(command < description);

    free(prompt);
    chat_msgs_free(&msgs);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_anthropic_live_tail_renders_tool_results_only(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_ANTHROPIC;
    r.think_mode = DS4_THINK_HIGH;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("toolu_live");
    tc.name = xstrdup("Bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("<tool_result>/tmp</tool_result>");
    chat_msg_add_tool_call_id(&user, "toolu_live");
    chat_msgs_push(&msgs, user);

    /* Anthropic system text is parsed separately and appended to chat_msgs for
     * rendering.  The live-tail finder must ignore it when locating the final
     * tool_result run. */
    chat_msg system = {0};
    system.role = xstrdup("system");
    system.content = xstrdup("You are terse.");
    chat_msgs_push(&msgs, system);

    anthropic_prepare_live_continuation(&r, &msgs);
    TEST_ASSERT(r.anthropic_live_call_ids.len == 1);
    TEST_ASSERT(!strcmp(r.anthropic_live_call_ids.v[0], "toolu_live"));
    TEST_ASSERT(r.anthropic_live_suffix_text != NULL);
    TEST_ASSERT(!strncmp(r.anthropic_live_suffix_text,
                         "<｜end▁of▁sentence｜><｜User｜><tool_result>",
                         strlen("<｜end▁of▁sentence｜><｜User｜><tool_result>")));
    TEST_ASSERT(strstr(r.anthropic_live_suffix_text, "/tmp</tool_result>") != NULL);
    TEST_ASSERT(strstr(r.anthropic_live_suffix_text, "<｜Assistant｜><think>") != NULL);
    TEST_ASSERT(strstr(r.anthropic_live_suffix_text, "Bash") == NULL);

    chat_msgs_free(&msgs);
    request_free(&r);
}

static void test_anthropic_tool_result_id_validation(void) {
    server s = {0};
    server_slot slot;
    test_server_bind_slot(&s, &slot);
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("<tool_result>out</tool_result>");
    chat_msg_add_tool_call_id(&user, "toolu_missing");
    chat_msgs_push(&msgs, user);

    char err[160] = {0};
    TEST_ASSERT(!anthropic_validate_tool_results(&s, &msgs, NULL,
                                                 err, sizeof(err)));
    TEST_ASSERT(strstr(err, "Anthropic continuation state is not available") != NULL);

    pthread_mutex_lock(&s.tool_mu);
    slot.anthropic_live.valid = true;
    slot.anthropic_live.live_tokens = 10;
    id_list_push_unique(&slot.anthropic_live.call_ids, "toolu_missing");
    pthread_mutex_unlock(&s.tool_mu);
    bool needs_live_tool_state = false;
    err[0] = '\0';
    TEST_ASSERT(anthropic_validate_tool_results(&s, &msgs,
                                                &needs_live_tool_state,
                                                err, sizeof(err)));
    TEST_ASSERT(needs_live_tool_state);

    chat_msgs_free(&msgs);
    live_tool_state_free(&slot.anthropic_live);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_anthropic_full_replay_allows_unknown_live_id(void) {
    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("toolu_replay");
    tc.name = xstrdup("Bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg user = {0};
    user.role = xstrdup("user");
    user.content = xstrdup("<tool_result>/tmp</tool_result>");
    chat_msg_add_tool_call_id(&user, "toolu_replay");
    chat_msgs_push(&msgs, user);

    bool needs_live_tool_state = false;
    char err[160] = {0};
    TEST_ASSERT(anthropic_validate_tool_results(&s, &msgs,
                                                &needs_live_tool_state,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);

    chat_msgs_free(&msgs);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_anthropic_tool_use_parses_before_role(void) {
    server s = {0};
    server_slot slot;
    test_server_bind_slot(&s, &slot);
    pthread_mutex_init(&s.tool_mu, NULL);

    /* GitHub #127 regression: Crush can replay full Anthropic history with
     * message objects serialized as {"content": ..., "role": ...}.  The parser
     * must still remember prior assistant tool_use ids, otherwise old
     * tool_result blocks are mistaken for live-only continuations and rejected
     * once the live frontier has moved on to newer tool calls. */
    pthread_mutex_lock(&s.tool_mu);
    slot.anthropic_live.valid = true;
    slot.anthropic_live.live_tokens = 100;
    id_list_push_unique(&slot.anthropic_live.call_ids, "toolu_current");
    pthread_mutex_unlock(&s.tool_mu);

    const char *json =
        "["
        "{\"content\":["
        "{\"type\":\"tool_use\",\"id\":\"toolu_old\",\"name\":\"Bash\","
        "\"input\":{\"command\":\"ls\"}}"
        "],\"role\":\"assistant\"},"
        "{\"role\":\"user\",\"content\":["
        "{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_old\",\"content\":\"ok\"}"
        "]},"
        "{\"role\":\"user\",\"content\":\"continue\"}"
        "]";
    const char *p = json;
    chat_msgs msgs = {0};
    TEST_ASSERT(parse_anthropic_messages(&p, &msgs));
    TEST_ASSERT(msgs.len == 3);
    TEST_ASSERT(msgs.v[0].calls.len == 1);
    TEST_ASSERT(msgs.v[0].calls.v[0].id &&
                !strcmp(msgs.v[0].calls.v[0].id, "toolu_old"));

    bool needs_live_tool_state = false;
    char err[160] = {0};
    TEST_ASSERT(anthropic_validate_tool_results(&s, &msgs,
                                                &needs_live_tool_state,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);

    chat_msgs_free(&msgs);
    live_tool_state_free(&slot.anthropic_live);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_tool_checkpoint_canonicalization_gate_exact_replay(void) {
    server s;
    memset(&s, 0, sizeof(s));

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_exact");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{}");
    tool_calls_push(&calls, tc);
    calls.raw_tool_text = xstrdup(
        "\n\n" DS4_TOOL_CALLS_START "\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "</｜DSML｜invoke>\n"
        DS4_TOOL_CALLS_END);

    TEST_ASSERT(!should_canonicalize_tool_checkpoint(&s, &calls));

    s.disable_exact_dsml_tool_replay = true;
    TEST_ASSERT(should_canonicalize_tool_checkpoint(&s, &calls));

    s.disable_exact_dsml_tool_replay = false;
    free(calls.raw_tool_text);
    calls.raw_tool_text = NULL;
    TEST_ASSERT(should_canonicalize_tool_checkpoint(&s, &calls));

    tool_calls_free(&calls);
}

static void test_responses_live_tail_renders_tool_outputs_only(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.think_mode = DS4_THINK_HIGH;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_live");
    tc.name = xstrdup("exec_command");
    tc.arguments = xstrdup("{\"cmd\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_live");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    responses_prepare_live_continuation(&r, &msgs);
    TEST_ASSERT(r.responses_live_call_ids.len == 1);
    TEST_ASSERT(!strcmp(r.responses_live_call_ids.v[0], "call_live"));
    TEST_ASSERT(r.responses_live_suffix_text != NULL);
    TEST_ASSERT(!strncmp(r.responses_live_suffix_text,
                         "<｜end▁of▁sentence｜><｜User｜><tool_result>",
                         strlen("<｜end▁of▁sentence｜><｜User｜><tool_result>")));
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "/tmp</tool_result>") != NULL);
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "<｜Assistant｜><think>") != NULL);
    TEST_ASSERT(strstr(r.responses_live_suffix_text, "exec_command") == NULL);

    chat_msgs_free(&msgs);
    request_free(&r);
}

static void test_responses_tool_output_id_validation(void) {
    server s = {0};
    server_slot slot;
    test_server_bind_slot(&s, &slot);
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_missing");
    tool.content = xstrdup("out");
    chat_msgs_push(&msgs, tool);

    char err[160] = {0};
    TEST_ASSERT(!responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH, NULL, NULL,
                                                 err, sizeof(err)));
    TEST_ASSERT(strstr(err, "Responses continuation state is not available") != NULL);

    pthread_mutex_lock(&s.tool_mu);
    slot.responses_live.valid = true;
    slot.responses_live.live_tokens = 10;
    id_list_push_unique(&slot.responses_live.call_ids, "call_missing");
    pthread_mutex_unlock(&s.tool_mu);
    err[0] = '\0';
    bool needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH,
                                                &needs_live_tool_state, NULL,
                                                err, sizeof(err)));
    TEST_ASSERT(needs_live_tool_state);

    chat_msgs_free(&msgs);
    live_tool_state_free(&slot.responses_live);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_responses_stateless_tool_replay_requires_reasoning(void) {
    server s = {0};
    server_slot slot;
    test_server_bind_slot(&s, &slot);
    pthread_mutex_init(&s.tool_mu, NULL);

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_replay");
    tc.name = xstrdup("exec_command");
    tc.arguments = xstrdup("{\"cmd\":\"pwd\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    chat_msg tool = {0};
    tool.role = xstrdup("tool");
    tool.tool_call_id = xstrdup("call_replay");
    tool.content = xstrdup("/tmp");
    chat_msgs_push(&msgs, tool);

    char err[160] = {0};
    bool needs_live_reasoning = false;
    bool needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(needs_live_reasoning);

    pthread_mutex_lock(&s.tool_mu);
    slot.responses_live.valid = true;
    slot.responses_live.live_tokens = 123;
    id_list_push_unique(&slot.responses_live.call_ids, "call_replay");
    pthread_mutex_unlock(&s.tool_mu);
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(needs_live_reasoning);

    free(msgs.v[0].reasoning);
    msgs.v[0].reasoning = xstrdup("replayed hidden reasoning");
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_HIGH,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(!needs_live_reasoning);

    free(msgs.v[0].reasoning);
    msgs.v[0].reasoning = NULL;
    err[0] = '\0';
    needs_live_reasoning = false;
    needs_live_tool_state = false;
    TEST_ASSERT(responses_validate_tool_outputs(&s, &msgs, DS4_THINK_NONE,
                                                &needs_live_tool_state,
                                                &needs_live_reasoning,
                                                err, sizeof(err)));
    TEST_ASSERT(!needs_live_tool_state);
    TEST_ASSERT(!needs_live_reasoning);

    chat_msgs_free(&msgs);
    live_tool_state_free(&slot.responses_live);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_responses_visible_suffix_matches_client_replay(void) {
    request r;
    request_init(&r, REQ_CHAT, 128);
    r.api = API_RESPONSES;
    r.think_mode = DS4_THINK_HIGH;
    r.reasoning_summary_emit = true;

    char *suffix = build_responses_visible_assistant_suffix(&r, "5",
                                                            "hidden summary",
                                                            NULL);
    TEST_ASSERT(strstr(suffix, "hidden summary") == NULL);
    TEST_ASSERT(strstr(suffix, "</think>5") != NULL);
    free(suffix);

    tool_calls calls = {0};
    tool_call tc = {0};
    tc.id = xstrdup("call_live");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"pwd\"}");
    tool_calls_push(&calls, tc);

    suffix = build_responses_visible_assistant_suffix(&r, "",
                                                      "tool summary",
                                                      &calls);
    TEST_ASSERT(strstr(suffix, "tool summary</think>") != NULL);
    TEST_ASSERT(strstr(suffix, "<｜DSML｜tool_calls>") != NULL);
    free(suffix);

    tool_calls_free(&calls);
    request_free(&r);
}

static void test_exact_dsml_tool_replay_can_be_disabled(void) {
    const char *dsml =
        "\n\n<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"bash\">\n"
        "<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>";

    server s = {0};
    pthread_mutex_init(&s.tool_mu, NULL);
    tool_memory_put(&s, "call_disabled", dsml);
    s.disable_exact_dsml_tool_replay = true;

    chat_msgs msgs = {0};
    chat_msg assistant = {0};
    assistant.role = xstrdup("assistant");
    tool_call tc = {0};
    tc.id = xstrdup("call_disabled");
    tc.name = xstrdup("bash");
    tc.arguments = xstrdup("{\"command\":\"canonical\"}");
    tool_calls_push(&assistant.calls, tc);
    chat_msgs_push(&msgs, assistant);

    tool_replay_stats stats = {0};
    tool_memory_attach_to_messages(&s, &msgs, &stats);
    TEST_ASSERT(msgs.v[0].calls.raw_tool_text == NULL);
    TEST_ASSERT(stats.canonical == 1);
    TEST_ASSERT(stats.missing_ids == 1);

    FILE *fp = tmpfile();
    TEST_ASSERT(fp != NULL);
    uint64_t bytes = 123;
    TEST_ASSERT(kv_tool_map_write(&s, fp, dsml, &bytes));
    TEST_ASSERT(bytes == 0);

    if (fp) fclose(fp);
    chat_msgs_free(&msgs);
    tool_memory_free(&s.tool_mem);
    pthread_mutex_destroy(&s.tool_mu);
}

static void test_dsml_decode_state_separates_structure_and_payload(void) {
    dsml_decode_tracker tracker;
    dsml_decode_tracker_init(&tracker);

    const char *prefix =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n";
    TEST_ASSERT(dsml_decode_state_for_text(prefix, strlen(prefix)) ==
                DSML_DECODE_STRUCTURAL);
    dsml_decode_tracker_update(&tracker, prefix, strlen(prefix));
    TEST_ASSERT(tracker.decode == DSML_DECODE_STRUCTURAL);

    const char *path_param =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n"
        DS4_PARAM_START " name=\"path\" string=\"true\">/tmp/a.py";
    TEST_ASSERT(dsml_decode_state_for_text(path_param, strlen(path_param)) ==
                DSML_DECODE_STRING_BODY);
    dsml_decode_tracker_update(&tracker, path_param, strlen(path_param));
    TEST_ASSERT(tracker.decode == DSML_DECODE_STRING_BODY);

    const char *path_closing =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n"
        DS4_PARAM_START " name=\"path\" string=\"true\">/tmp/a.py</";
    TEST_ASSERT(dsml_decode_state_for_text(path_closing, strlen(path_closing)) ==
                DSML_DECODE_STRUCTURAL);
    dsml_decode_tracker_update(&tracker, path_closing, strlen(path_closing));
    TEST_ASSERT(tracker.decode == DSML_DECODE_STRUCTURAL);

    const char *json_struct =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n"
        DS4_PARAM_START " name=\"edits\" string=\"false\">[{";
    TEST_ASSERT(dsml_decode_state_for_text(json_struct, strlen(json_struct)) ==
                DSML_DECODE_JSON_STRUCTURAL);
    dsml_decode_tracker_init(&tracker);
    dsml_decode_tracker_update(&tracker, json_struct, strlen(json_struct));
    TEST_ASSERT(tracker.decode == DSML_DECODE_JSON_STRUCTURAL);

    const char *json_string =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n"
        DS4_PARAM_START " name=\"edits\" string=\"false\">[{\"newText\":\"for i in";
    TEST_ASSERT(dsml_decode_state_for_text(json_string, strlen(json_string)) ==
                DSML_DECODE_JSON_STRING);
    dsml_decode_tracker_update(&tracker, json_string, strlen(json_string));
    TEST_ASSERT(tracker.decode == DSML_DECODE_JSON_STRING);

    const char *done =
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"edit\">\n"
        DS4_PARAM_START " name=\"edits\" string=\"false\">[]"
        DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END;
    TEST_ASSERT(dsml_decode_state_for_text(done, strlen(done)) ==
                DSML_DECODE_OUTSIDE);
    dsml_decode_tracker_init(&tracker);
    dsml_decode_tracker_update(&tracker, done, strlen(done));
    TEST_ASSERT(tracker.decode == DSML_DECODE_OUTSIDE);
}

