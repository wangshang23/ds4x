int ds4_engine_generate_argmax(
        ds4_engine        *e,
        const ds4_tokens  *prompt,
        int                n_predict,
        int                ctx_size,
        ds4_token_emit_fn  emit,
        ds4_generation_done_fn done,
        void              *emit_ud,
        ds4_session_progress_fn progress,
        void              *progress_ud) {
    if (!e || !prompt || n_predict < 0 || ctx_size <= 0) return 1;

    ds4_session *session = NULL;
    if (ds4_session_create(&session, e, ctx_size) != 0) {
        fprintf(stderr, "ds4: failed to create CUDA session\n");
        return 1;
    }
    ds4_session_set_progress(session, progress, progress_ud);

    char err[256] = {0};
    const double prefill_t0 = now_sec();
    int rc = ds4_session_sync(session, prompt, err, sizeof(err));
    const double prefill_t1 = now_sec();
    ds4_session_set_progress(session, NULL, NULL);
    if (rc != 0) {
        fprintf(stderr, "ds4: prefill failed: %s\n", err);
        ds4_session_free(session);
        return rc;
    }

    int generated = 0;
    const double decode_t0 = now_sec();
    while (generated < n_predict &&
           ds4_session_pos(session) + 1 < ds4_session_ctx(session)) {
        const int token = ds4_session_argmax(session);
        if (token < 0) {
            fprintf(stderr, "ds4: argmax failed\n");
            rc = 1;
            break;
        }
        if (ds4_token_is_stop(e, token)) break;
        if (emit) emit(emit_ud, token);
        generated++;
        if (generated >= n_predict ||
            ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
            break;
        }
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4: decode failed: %s\n", err);
            rc = 1;
            break;
        }
    }
    const double decode_t1 = now_sec();
    if (done) done(emit_ud);
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: prefill %.2f tok/s, generation %.2f tok/s\n",
            prefill_t1 > prefill_t0
                ? (double)prompt->len / (prefill_t1 - prefill_t0)
                : 0.0,
            decode_t1 > decode_t0
                ? (double)generated / (decode_t1 - decode_t0)
                : 0.0);
    ds4_session_free(session);
    return rc;
}
