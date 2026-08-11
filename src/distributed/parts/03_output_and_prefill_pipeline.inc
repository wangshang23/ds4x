/* =========================================================================
 * One-Shot Coordinator Generation Utilities
 * ========================================================================= */

static bool dist_prompt_is_rendered_chat(const char *prompt) {
    const char *bos = "<｜begin▁of▁sentence｜>";
    return prompt && strncmp(prompt, bos, strlen(bos)) == 0;
}

static bool dist_json_utf8_valid(const char *s, size_t n) {
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            i++;
            continue;
        }
        int need = 0;
        if ((c & 0xe0) == 0xc0) need = 2;
        else if ((c & 0xf0) == 0xe0) need = 3;
        else if ((c & 0xf8) == 0xf0) need = 4;
        else return false;
        if (i + (size_t)need > n) return false;
        unsigned char c1 = (unsigned char)s[i + 1u];
        if ((c1 & 0xc0) != 0x80) return false;
        if (need == 2 && c < 0xc2) return false;
        if (need == 3 && c == 0xe0 && c1 < 0xa0) return false;
        if (need == 3 && c == 0xed && c1 >= 0xa0) return false;
        if (need == 4 && c == 0xf0 && c1 < 0x90) return false;
        if (need == 4 && c == 0xf4 && c1 >= 0x90) return false;
        for (int j = 2; j < need; j++) {
            if ((((unsigned char)s[i + (size_t)j]) & 0xc0) != 0x80) return false;
        }
        i += (size_t)need;
    }
    return true;
}

static void dist_json_write_string(FILE *fp, const char *s, size_t n) {
    const bool valid_utf8 = dist_json_utf8_valid(s, n);
    fputc('"', fp);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            fputc('\\', fp);
            fputc((char)c, fp);
        } else if (c == '\n') {
            fputs("\\n", fp);
        } else if (c == '\r') {
            fputs("\\r", fp);
        } else if (c == '\t') {
            fputs("\\t", fp);
        } else if (c < 0x20) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else if (!valid_utf8 && c >= 0x80) {
            fprintf(fp, "\\u%04x", (unsigned)c);
        } else {
            fputc((char)c, fp);
        }
    }
    fputc('"', fp);
}

static void dist_json_write_token(FILE *fp, ds4_engine *engine, int token) {
    size_t n = 0;
    char *text = ds4_token_text(engine, token, &n);
    fprintf(fp, "{\"id\":%d,\"text\":", token);
    dist_json_write_string(fp, text ? text : "", text ? n : 0);
    fputs(",\"bytes\":[", fp);
    if (text) {
        for (size_t i = 0; i < n; i++) {
            if (i) fputc(',', fp);
            fprintf(fp, "%u", (unsigned)(unsigned char)text[i]);
        }
    }
    fputc(']', fp);
    fputc('}', fp);
    free(text);
}

static int dist_write_logits_dump(
        ds4_dist_coordinator_state *state,
        const ds4_dist_generation_options *gen,
        const ds4_tokens *prompt,
        const ds4_dist_route_plan *plan,
        const float *logits) {
    FILE *fp = fopen(gen->dump_logits_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open distributed --dump-logits file: %s\n",
                gen->dump_logits_path);
        return 1;
    }

    const int vocab = ds4_engine_vocab_size(state->engine);
    const int argmax = dist_logits_argmax(logits, vocab);
    fprintf(fp,
            "{\n"
            "  \"source\":\"ds4-distributed\",\n"
            "  \"quant_bits\":%d,\n"
            "  \"prompt_tokens\":%d,\n"
            "  \"ctx\":%d,\n"
            "  \"vocab\":%d,\n"
            "  \"route_count\":%u,\n"
            "  \"argmax_token\":",
            ds4_engine_routed_quant_bits(state->engine),
            prompt->len,
            gen->ctx_size,
            vocab,
            plan->count);
    dist_json_write_token(fp, state->engine, argmax);
    fprintf(fp, ",\n  \"argmax_logit\":%.9g,\n  \"logits\":[", logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close distributed --dump-logits file: %s\n",
                gen->dump_logits_path);
        return 1;
    }
    return 0;
}

static int dist_logits_top_logprobs(const float *logits, int vocab, ds4_dist_logprob *scores, int k) {
    if (k <= 0) return 0;
    for (int i = 0; i < k; i++) {
        scores[i].id = -1;
        scores[i].logit = -FLT_MAX;
        scores[i].logprob = -FLT_MAX;
    }

    float max_logit = -FLT_MAX;
    for (int i = 0; i < vocab; i++) {
        if (isfinite(logits[i]) && logits[i] > max_logit) max_logit = logits[i];
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (int i = 0; i < vocab; i++) {
        if (isfinite(logits[i])) sum += exp((double)logits[i] - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);

    int n = 0;
    for (int i = 0; i < vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == k && v <= scores[k - 1].logit) continue;
        int pos = n < k ? n++ : k - 1;
        while (pos > 0 && v > scores[pos - 1].logit) {
            scores[pos] = scores[pos - 1];
            pos--;
        }
        scores[pos].id = i;
        scores[pos].logit = v;
        scores[pos].logprob = (float)((double)v - logsum);
    }
    return n;
}

static int dist_coordinator_rebuild_from_transcript(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        ds4_dist_route_plan *plan,
        const ds4_tokens *transcript,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        uint64_t *plan_generation,
        bool forget_route,
        char *err,
        size_t errlen) {
    DIST_COORD_DEBUG(state,
                     "ds4: distributed coordinator: replaying %d tokens after distributed %s\n",
                     transcript->len,
                     forget_route ? "route failure" : "KV mismatch");
    if (forget_route) {
        dist_coordinator_forget_route_workers(state, plan);
        dist_route_plan_free(plan);
        uint64_t generation = 0;
        if (!dist_coordinator_ensure_route(state, plan, &generation, err, errlen)) return 1;
        if (plan_generation) *plan_generation = generation;
    } else if (plan->count == 0) {
        uint64_t generation = 0;
        if (!dist_coordinator_ensure_route(state, plan, &generation, err, errlen)) return 1;
        if (plan_generation) *plan_generation = generation;
    }
    if (dist_coordinator_prefill_prompt(state,
                                        session,
                                        plan,
                                        transcript,
                                        session_id,
                                        request_id,
                                        logits,
                                        err,
                                        errlen) != 0) {
        return 1;
    }
    return 0;
}

static int dist_write_logprobs_dump(
        ds4_dist_coordinator_state *state,
        const ds4_dist_generation_options *gen,
        const ds4_tokens *prompt,
        ds4_dist_route_plan *plan,
        ds4_session *session,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits) {
    FILE *fp = fopen(gen->dump_logprobs_path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open distributed --dump-logprobs file: %s\n",
                gen->dump_logprobs_path);
        return 1;
    }

    int k = gen->dump_logprobs_top_k > 0 ? gen->dump_logprobs_top_k : 20;
    if (k > 128) k = 128;
    ds4_dist_logprob *scores = calloc((size_t)k, sizeof(scores[0]));
    if (!scores) {
        fclose(fp);
        return 1;
    }

    ds4_tokens transcript = {0};
    ds4_tokens_copy(&transcript, prompt);

    int max_tokens = gen->n_predict;
    int room = gen->ctx_size - prompt->len;
    if (room <= 1) max_tokens = 0;
    else if (max_tokens > room - 1) max_tokens = room - 1;

    fprintf(fp,
            "{\n"
            "  \"source\":\"ds4-distributed\",\n"
            "  \"prompt_tokens\":%d,\n"
            "  \"ctx\":%d,\n"
            "  \"top_k\":%d,\n"
            "  \"route_count\":%u,\n"
            "  \"steps\":[\n",
            prompt->len,
            gen->ctx_size,
            k,
            plan->count);

    char err[256];
    int rc = 0;
    const int eos = ds4_token_eos(state->engine);
    for (int generated = 0; generated < max_tokens; generated++) {
        const int n = dist_logits_top_logprobs(logits, ds4_engine_vocab_size(state->engine), scores, k);
        const int token = dist_logits_argmax(logits, ds4_engine_vocab_size(state->engine));
        if (generated) fputs(",\n", fp);
        fprintf(fp, "    {\"step\":%d,\"selected\":", generated);
        dist_json_write_token(fp, state->engine, token);
        fputs(",\"top_logprobs\":[", fp);
        for (int i = 0; i < n; i++) {
            if (i) fputc(',', fp);
            fputs("{\"token\":", fp);
            dist_json_write_token(fp, state->engine, scores[i].id);
            fprintf(fp, ",\"logit\":%.9g,\"logprob\":%.9g}", scores[i].logit, scores[i].logprob);
        }
        fputs("]}", fp);

        if (token == eos) break;
        const uint32_t token_pos = (uint32_t)prompt->len + (uint32_t)generated;
        ds4_tokens_push(&transcript, token);
        if (dist_coordinator_eval_span(state, session, plan,
                                       &token, 1, token_pos,
                                       session_id, (*request_id)++,
                                       false, logits, err, sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4: distributed decode failed while dumping logprobs: %s\n",
                    err);
            if (dist_coordinator_rebuild_from_transcript(state,
                                                         session,
                                                         plan,
                                                         &transcript,
                                                         session_id,
                                                         request_id,
                                                         logits,
                                                         NULL,
                                                         true,
                                                         err,
                                                         sizeof(err)) != 0) {
                fprintf(stderr,
                        "ds4: distributed recovery failed while dumping logprobs: %s\n",
                        err);
                rc = 1;
                break;
            }
        }
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4: failed to close distributed --dump-logprobs file: %s\n",
                gen->dump_logprobs_path);
        rc = 1;
    }
    ds4_tokens_free(&transcript);
    free(scores);
    return rc;
}

/* =========================================================================
 * Pipelined Prefill
 * =========================================================================
 *
 * Long prompt ingestion is chunked so the coordinator can compute its local
 * slice for chunk N+1 while downstream workers process chunk N. Intermediate
 * chunks are ACK-only; only the final chunk needs to return hidden state or
 * logits to the coordinator.
 */

static int dist_prefill_sender_init(
        ds4_dist_prefill_sender *sender,
        ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint64_t session_id,
        int fd,
        uint32_t chunk_count,
        uint32_t max_hidden_bytes,
        char *err,
        size_t errlen) {
    memset(sender, 0, sizeof(*sender));
    sender->state = state;
    sender->plan = plan;
    sender->prompt = prompt;
    sender->session_id = session_id;
    sender->fd = fd;
    sender->slot_count = dist_prefill_send_depth(chunk_count);
    pthread_mutex_init(&sender->mu, NULL);
    pthread_cond_init(&sender->can_enqueue, NULL);
    pthread_cond_init(&sender->can_dequeue, NULL);

    sender->slots = calloc(sender->slot_count, sizeof(sender->slots[0]));
    if (!sender->slots) {
        if (errlen) snprintf(err, errlen, "out of memory allocating prefill sender slots");
        return 1;
    }
    for (uint32_t i = 0; i < sender->slot_count; i++) {
        sender->slots[i].hidden = malloc(max_hidden_bytes);
        if (!sender->slots[i].hidden) {
            if (errlen) snprintf(err, errlen, "out of memory allocating prefill sender hidden-state buffers");
            return 1;
        }
    }
    return 0;
}

static void dist_prefill_sender_destroy(ds4_dist_prefill_sender *sender) {
    if (!sender) return;
    if (sender->slots) {
        for (uint32_t i = 0; i < sender->slot_count; i++) {
            free(sender->slots[i].hidden);
        }
        free(sender->slots);
    }
    pthread_cond_destroy(&sender->can_dequeue);
    pthread_cond_destroy(&sender->can_enqueue);
    pthread_mutex_destroy(&sender->mu);
}

static ds4_dist_prefill_send_slot *dist_prefill_sender_acquire_slot(
        ds4_dist_prefill_sender *sender,
        char *err,
        size_t errlen) {
    pthread_mutex_lock(&sender->mu);
    while (!sender->stop && sender->queued == sender->slot_count) {
        pthread_cond_wait(&sender->can_enqueue, &sender->mu);
    }
    if (sender->stop || sender->rc != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             sender->err[0] ? sender->err : "distributed prefill sender stopped");
        pthread_mutex_unlock(&sender->mu);
        return NULL;
    }
    ds4_dist_prefill_send_slot *slot = &sender->slots[sender->tail];
    pthread_mutex_unlock(&sender->mu);
    return slot;
}

static int dist_prefill_sender_enqueue_slot(
        ds4_dist_prefill_sender *sender,
        char *err,
        size_t errlen) {
    pthread_mutex_lock(&sender->mu);
    if (sender->stop || sender->rc != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             sender->err[0] ? sender->err : "distributed prefill sender stopped");
        pthread_mutex_unlock(&sender->mu);
        return 1;
    }
    sender->tail = (sender->tail + 1u) % sender->slot_count;
    sender->queued++;
    pthread_cond_signal(&sender->can_dequeue);
    pthread_mutex_unlock(&sender->mu);
    return 0;
}

static void dist_prefill_sender_finish(ds4_dist_prefill_sender *sender) {
    pthread_mutex_lock(&sender->mu);
    sender->producer_done = true;
    pthread_cond_signal(&sender->can_dequeue);
    pthread_mutex_unlock(&sender->mu);
}

static void dist_prefill_sender_cancel(ds4_dist_prefill_sender *sender) {
    pthread_mutex_lock(&sender->mu);
    sender->producer_done = true;
    sender->stop = true;
    pthread_cond_broadcast(&sender->can_enqueue);
    pthread_cond_broadcast(&sender->can_dequeue);
    pthread_mutex_unlock(&sender->mu);
    shutdown(sender->fd, SHUT_RDWR);
}

static void *dist_prefill_sender_main(void *arg) {
    ds4_dist_prefill_sender *sender = arg;
    for (;;) {
        pthread_mutex_lock(&sender->mu);
        while (!sender->stop && sender->queued == 0 && !sender->producer_done) {
            pthread_cond_wait(&sender->can_dequeue, &sender->mu);
        }
        if (sender->stop || (sender->queued == 0 && sender->producer_done)) {
            pthread_mutex_unlock(&sender->mu);
            break;
        }
        ds4_dist_prefill_send_slot *slot = &sender->slots[sender->head];
        pthread_mutex_unlock(&sender->mu);

        char send_err[256];
        const double send_t0 = dist_now_sec();
        int rc = dist_coordinator_send_remote_work_on_fd(sender->state,
                                                         sender->plan,
                                                         sender->fd,
                                                         sender->prompt->v + slot->pos,
                                                         slot->n_tokens,
                                                         slot->pos,
                                                         sender->session_id,
                                                         slot->request_id,
                                                         slot->prefix_hash,
                                                         slot->result_hash,
                                                         slot->reset_session,
                                                         slot->ack_only,
                                                         slot->hidden,
                                                         slot->hidden_bytes,
                                                         send_err,
                                                         sizeof(send_err));
        const double send_t1 = dist_now_sec();

        pthread_mutex_lock(&sender->mu);
        uint32_t slot_hidden_wire_bytes = slot->hidden_bytes;
        (void)dist_activation_wire_bytes_from_f32_bytes(sender->state->activation_bits,
                                                        slot->hidden_bytes,
                                                        &slot_hidden_wire_bytes);
        sender->send_sec += send_t1 - send_t0;
        sender->send_bytes += (uint64_t)sizeof(ds4_dist_work_fixed) +
                              (uint64_t)slot->n_tokens * sizeof(uint32_t) +
                              (uint64_t)slot_hidden_wire_bytes +
                              sender->plan->blob_bytes;
        if (rc != 0) {
            sender->rc = 1;
            snprintf(sender->err, sizeof(sender->err), "%s", send_err);
            sender->stop = true;
            pthread_cond_broadcast(&sender->can_enqueue);
            pthread_cond_broadcast(&sender->can_dequeue);
            pthread_mutex_unlock(&sender->mu);
            shutdown(sender->fd, SHUT_RDWR);
            break;
        }
        sender->head = (sender->head + 1u) % sender->slot_count;
        sender->queued--;
        pthread_cond_signal(&sender->can_enqueue);
        pthread_mutex_unlock(&sender->mu);
    }
    return NULL;
}

static void dist_prefill_reader_signal_progress(
        ds4_dist_prefill_result_reader *reader,
        uint32_t completed,
        bool done) {
    pthread_mutex_lock(&reader->progress_mu);
    if (completed > reader->progress_completed) reader->progress_completed = completed;
    if (done) reader->progress_done = true;
    pthread_cond_broadcast(&reader->progress_cv);
    pthread_mutex_unlock(&reader->progress_mu);
}

static void dist_prefill_reader_emit_progress(
        ds4_dist_prefill_result_reader *reader,
        uint32_t *reported) {
    if (!reader || !reported || !reader->progress_session) return;

    pthread_mutex_lock(&reader->progress_mu);
    uint32_t completed = reader->progress_completed;
    pthread_mutex_unlock(&reader->progress_mu);

    while (*reported < completed) {
        (*reported)++;
        uint32_t rel = (*reported) * reader->chunk_cap;
        if (rel > reader->total_tokens) rel = reader->total_tokens;
        uint32_t current = reader->progress_base + rel;
        if (current > reader->progress_total) current = reader->progress_total;
        ds4_session_report_progress(reader->progress_session,
                                    "prefill_chunk",
                                    (int)current,
                                    (int)reader->progress_total);
    }
}

static bool dist_prefill_reader_wait_emit_progress(
        ds4_dist_prefill_result_reader *reader,
        uint32_t *reported) {
    if (!reader || !reported) return true;

    pthread_mutex_lock(&reader->progress_mu);
    while (reader->progress_completed <= *reported && !reader->progress_done) {
        pthread_cond_wait(&reader->progress_cv, &reader->progress_mu);
    }
    const bool done = reader->progress_done;
    pthread_mutex_unlock(&reader->progress_mu);

    dist_prefill_reader_emit_progress(reader, reported);
    pthread_mutex_lock(&reader->progress_mu);
    const bool finished = done && *reported >= reader->progress_completed;
    pthread_mutex_unlock(&reader->progress_mu);
    return finished;
}

static bool dist_prefill_reader_wait_flow_window(
        ds4_dist_prefill_result_reader *reader,
        uint32_t submitted,
        uint32_t window,
        uint32_t *reported) {
    if (!reader || window == 0) return true;

    for (;;) {
        pthread_mutex_lock(&reader->progress_mu);
        const uint32_t completed = reader->progress_completed;
        const bool done = reader->progress_done;
        const bool has_room = submitted < completed + window;
        if (done || has_room) {
            pthread_mutex_unlock(&reader->progress_mu);
            dist_prefill_reader_emit_progress(reader, reported);
            return !done && has_room;
        }
        pthread_cond_wait(&reader->progress_cv, &reader->progress_mu);
        pthread_mutex_unlock(&reader->progress_mu);
        dist_prefill_reader_emit_progress(reader, reported);
    }
}

static void *dist_prefill_result_reader_main(void *arg) {
    ds4_dist_prefill_result_reader *reader = arg;
    reader->rc = 0;
    reader->err[0] = '\0';
    reader->final_kind = 0;
    reader->final_payload = NULL;
    reader->final_payload_bytes = 0;

    const uint32_t logits_bytes =
        (uint32_t)((uint64_t)ds4_engine_vocab_size(reader->state->engine) * sizeof(float));
    for (uint32_t i = 0; i < reader->count; i++) {
        const uint64_t request_id = reader->first_request_id + (uint64_t)i;
        uint32_t kind = 0;
        uint32_t payload_bytes = 0;
        uint64_t result_hash = 0;
        void *payload = NULL;
        int recv_rc = dist_recv_result_alloc(reader->fd,
                                             reader->state,
                                             request_id,
                                             &kind,
                                             &result_hash,
                                             &payload,
                                             &payload_bytes,
                                             reader->err,
                                             sizeof(reader->err));
        if (recv_rc != 0) {
            reader->rc = recv_rc;
            free(payload);
            shutdown(reader->fd, SHUT_RDWR);
            dist_prefill_reader_signal_progress(reader, i, true);
            return NULL;
        }
        if (reader->expected_hashes && result_hash != reader->expected_hashes[i]) {
            snprintf(reader->err,
                     sizeof(reader->err),
                     "distributed pipelined prefill prefix hash mismatch");
            reader->rc = 1;
            free(payload);
            shutdown(reader->fd, SHUT_RDWR);
            dist_prefill_reader_signal_progress(reader, i, true);
            return NULL;
        }
        const uint32_t pos0 = i * reader->chunk_cap;
        const uint32_t remaining = reader->total_tokens - pos0;
        const uint32_t chunk = remaining < reader->chunk_cap ? remaining : reader->chunk_cap;
        const uint64_t hidden_bytes64 = (uint64_t)chunk * reader->hc_values * sizeof(float);
        const bool final_chunk = i + 1u == reader->count;
        const bool valid_ack = !final_chunk &&
                               kind == DS4_DIST_RESULT_ACK &&
                               payload_bytes == 0;
        const bool valid_logits = kind == DS4_DIST_RESULT_LOGITS && payload_bytes == logits_bytes;
        const bool valid_hidden = reader->allow_hidden &&
                                  hidden_bytes64 <= UINT32_MAX &&
                                  kind == DS4_DIST_RESULT_HIDDEN_STATE &&
                                  payload_bytes == (uint32_t)hidden_bytes64;
        if (!valid_ack && !valid_logits && !valid_hidden) {
            snprintf(reader->err,
                     sizeof(reader->err),
                     "distributed pipelined prefill returned invalid result");
            reader->rc = 1;
            free(payload);
            shutdown(reader->fd, SHUT_RDWR);
            dist_prefill_reader_signal_progress(reader, i, true);
            return NULL;
        }
        if (final_chunk) {
            reader->final_kind = kind;
            reader->final_payload = payload;
            reader->final_payload_bytes = payload_bytes;
            payload = NULL;
        }
        free(payload);
        dist_prefill_reader_signal_progress(reader, i + 1u, final_chunk);
    }
    dist_prefill_reader_signal_progress(reader, reader->count, true);
    return NULL;
}

static bool dist_coordinator_can_pipeline_prefill(
        const ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        ds4_session *session,
        uint32_t n_tokens,
        uint32_t chunk_cap) {
    if (getenv("DS4_DIST_DISABLE_PREFILL_PIPELINE")) return false;
    if (!state || !plan) return false;
    (void)session;
    if (chunk_cap == 0 || n_tokens <= chunk_cap) return false;
    if (plan->count == 0) return false;
    if (plan->entry[0].fd < 0) return false;
    const ds4_dist_route_entry *final = &plan->entry[plan->count - 1u];
    if ((final->flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) == 0) {
        return final->layer_end + 1u == state->n_layers &&
               state->local_can_output_head;
    }
    return true;
}

static int dist_coordinator_prefill_chunk_cap(
        const ds4_dist_coordinator_state *state,
        ds4_session *session,
        uint32_t *chunk_cap,
        char *err,
        size_t errlen) {
    if (!chunk_cap) return 1;
    const int prefill_cap_i = ds4_session_prefill_cap(session);
    if (prefill_cap_i <= 0) {
        if (errlen) snprintf(err, errlen, "distributed coordinator has no prefill capacity");
        return 1;
    }
    const uint32_t prefill_cap = (uint32_t)prefill_cap_i;
    uint32_t requested = state ? state->prefill_chunk : 0u;
    const char *env = getenv("DS4_DIST_PREFILL_CHUNK");
    if (requested == 0 && env && env[0]) {
        if (!dist_parse_positive_u32(env, "DS4_DIST_PREFILL_CHUNK", &requested, err, errlen)) {
            return 1;
        }
    }
    if (requested == 0) requested = prefill_cap;
    if (requested > prefill_cap) {
        if (errlen) {
            snprintf(err,
                     errlen,
                     "distributed prefill chunk %u exceeds session prefill cap %u",
                     requested,
                     prefill_cap);
        }
        return 1;
    }
    *chunk_cap = requested;
    return 0;
}

static int dist_coordinator_prefill_window(
        const ds4_dist_coordinator_state *state,
        const ds4_dist_route_plan *plan,
        uint32_t chunk_count,
        uint32_t *window,
        char *err,
        size_t errlen) {
    if (!window) return 1;
    uint32_t requested = state ? state->prefill_window : 0u;
    const char *env = getenv("DS4_DIST_PREFILL_WINDOW");
    if (requested == 0 && env && env[0]) {
        if (!dist_parse_positive_u32(env, "DS4_DIST_PREFILL_WINDOW", &requested, err, errlen)) {
            return 1;
        }
    }
    if (requested > 64u) {
        if (errlen) snprintf(err, errlen, "distributed prefill window %u exceeds limit 64", requested);
        return 1;
    }
    if (requested == 0) {
        const uint32_t remote_stages = plan ? plan->count : 0u;
        requested = remote_stages + 2u;
        if (requested < 2u) requested = 2u;
        if (requested > 8u) requested = 8u;
    }
    if (chunk_count != 0 && requested > chunk_count) requested = chunk_count;
    *window = requested ? requested : 1u;
    return 0;
}

static void dist_report_prefill_progress(ds4_session *session, uint32_t current, uint32_t total) {
    if (!session) return;
    if (current > (uint32_t)INT_MAX) current = (uint32_t)INT_MAX;
    if (total > (uint32_t)INT_MAX) total = (uint32_t)INT_MAX;
    ds4_session_report_progress(session, "prefill_chunk", (int)current, (int)total);
}

static int dist_coordinator_prefill_prompt_pipelined(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint32_t span_start,
        uint32_t n_tokens,
        bool reset_first_chunk,
        uint32_t chunk_cap,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        char *err,
        size_t errlen) {
    const uint32_t total = n_tokens;
    if (!prompt ||
        span_start > (uint32_t)prompt->len ||
        n_tokens == 0 ||
        n_tokens > (uint32_t)prompt->len - span_start) {
        if (errlen) snprintf(err, errlen, "invalid distributed pipelined prefill span");
        return 1;
    }
    const uint32_t span_end = span_start + n_tokens;
    const uint32_t chunk_count = (total + chunk_cap - 1u) / chunk_cap;
    const uint64_t hc_values = ds4_engine_hidden_f32_values(state->engine);
    const uint64_t max_hidden_bytes64 = (uint64_t)chunk_cap * hc_values * sizeof(float);
    if (max_hidden_bytes64 > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed coordinator hidden-state chunk is too large");
        return 1;
    }
    const uint32_t max_hidden_bytes = (uint32_t)max_hidden_bytes64;
    uint32_t flow_window = 0;
    if (dist_coordinator_prefill_window(state,
                                        plan,
                                        chunk_count,
                                        &flow_window,
                                        err,
                                        errlen) != 0) {
        return 1;
    }

    ds4_dist_prefill_sender sender;
    if (dist_prefill_sender_init(&sender,
                                 state,
                                 plan,
                                 prompt,
                                 session_id,
                                 plan->entry[0].fd,
                                 chunk_count,
                                 max_hidden_bytes,
                                 err,
                                 errlen) != 0) {
        dist_prefill_sender_destroy(&sender);
        return 1;
    }

    ds4_dist_prefill_result_reader reader;
    memset(&reader, 0, sizeof(reader));
    reader.state = state;
    reader.fd = plan->entry[0].fd;
    reader.progress_session = session;
    reader.first_request_id = *request_id;
    reader.count = chunk_count;
    reader.total_tokens = total;
    reader.chunk_cap = chunk_cap;
    reader.progress_base = span_start;
    reader.progress_total = (uint32_t)prompt->len;
    reader.hc_values = hc_values;
    reader.allow_hidden =
        (plan->entry[plan->count - 1u].flags & DS4_DIST_ROUTE_F_OUTPUT_LOGITS) == 0;
    pthread_mutex_init(&reader.progress_mu, NULL);
    pthread_cond_init(&reader.progress_cv, NULL);
    reader.expected_hashes = calloc(chunk_count, sizeof(reader.expected_hashes[0]));
    if (!reader.expected_hashes) {
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        dist_prefill_sender_destroy(&sender);
        if (errlen) snprintf(err, errlen, "out of memory allocating distributed prefill hashes");
        return 1;
    }
    uint64_t chunk_prefix_hash = dist_token_hash_prefix(prompt->v, span_start);
    for (uint32_t i = 0, hash_pos = span_start; i < chunk_count; i++) {
        const uint32_t remaining = span_end - hash_pos;
        const uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
        chunk_prefix_hash = dist_token_hash_update_span(chunk_prefix_hash,
                                                        prompt->v + hash_pos,
                                                        chunk);
        reader.expected_hashes[i] = chunk_prefix_hash;
        hash_pos += chunk;
    }

    pthread_t reader_tid;
    if (pthread_create(&reader_tid, NULL, dist_prefill_result_reader_main, &reader) != 0) {
        free(reader.expected_hashes);
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        dist_prefill_sender_destroy(&sender);
        if (errlen) snprintf(err, errlen, "failed to start distributed prefill result reader");
        return 1;
    }
    pthread_t sender_tid;
    if (pthread_create(&sender_tid, NULL, dist_prefill_sender_main, &sender) != 0) {
        dist_prefill_sender_cancel(&sender);
        pthread_join(reader_tid, NULL);
        free(reader.expected_hashes);
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        dist_prefill_sender_destroy(&sender);
        if (errlen) snprintf(err, errlen, "failed to start distributed prefill sender");
        return 1;
    }

    DIST_COORD_DEBUG(state,
                     "ds4: distributed coordinator: pipelined prefill %u chunks of up to %u tokens through %u worker%s, first hop %s:%u, send depth %u, flow window %u\n",
                     chunk_count,
                     chunk_cap,
                     plan->count,
                     plan->count == 1u ? "" : "s",
                     plan->entry[0].host,
                     plan->entry[0].port,
                     sender.slot_count,
                     flow_window);

    int rc = 0;
    double local_eval_sec = 0.0;
    const double pipeline_t0 = dist_now_sec();
    uint32_t pos = span_start;
    uint64_t next_prefix_hash = dist_token_hash_prefix(prompt->v, span_start);
    uint32_t reported_chunks = 0;
    uint32_t submitted_chunks = 0;
    while (pos < span_end) {
        if (!dist_prefill_reader_wait_flow_window(&reader,
                                                  submitted_chunks,
                                                  flow_window,
                                                  &reported_chunks)) {
            if (errlen) snprintf(err, errlen, "distributed prefill result reader stopped");
            rc = 1;
            break;
        }
        const uint32_t remaining = span_end - pos;
        const uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
        const uint64_t hidden_bytes64 = (uint64_t)chunk * hc_values * sizeof(float);
        const uint32_t hidden_bytes = (uint32_t)hidden_bytes64;
        ds4_dist_prefill_send_slot *slot =
            dist_prefill_sender_acquire_slot(&sender, err, errlen);
        if (!slot) {
            rc = 1;
            break;
        }
        if (pos == span_start &&
            reset_first_chunk &&
            ds4_session_layer_slice_reset(session, err, errlen) != 0) {
            rc = 1;
            break;
        }
        const double local_t0 = dist_now_sec();
        rc = ds4_session_eval_layer_slice(session,
                                          prompt->v + pos,
                                          chunk,
                                          pos,
                                          state->local_start,
                                          state->local_end,
                                          NULL,
                                          slot->hidden,
                                          false,
                                          NULL,
                                          err,
                                          errlen);
        const double local_t1 = dist_now_sec();
        local_eval_sec += local_t1 - local_t0;
        if (rc != 0) break;

        slot->pos = pos;
        slot->n_tokens = chunk;
        slot->hidden_bytes = hidden_bytes;
        slot->request_id = *request_id;
        slot->prefix_hash = next_prefix_hash;
        slot->result_hash = reader.expected_hashes[submitted_chunks];
        slot->reset_session = reset_first_chunk && pos == span_start;
        slot->ack_only = !getenv("DS4_DIST_DISABLE_PREFILL_ACK_ONLY") &&
                         pos + chunk < span_end;
        rc = dist_prefill_sender_enqueue_slot(&sender, err, errlen);
        if (rc != 0) break;

        dist_prefill_reader_emit_progress(&reader, &reported_chunks);
        (*request_id)++;
        submitted_chunks++;
        next_prefix_hash = slot->result_hash;
        pos += chunk;
    }

    if (rc == 0) dist_prefill_sender_finish(&sender);
    else dist_prefill_sender_cancel(&sender);
    pthread_join(sender_tid, NULL);
    if (rc == 0 && sender.rc != 0) {
        if (errlen) snprintf(err, errlen, "%s",
                             sender.err[0] ? sender.err : "distributed prefill sender failed");
        rc = 1;
    }
    if (rc != 0) {
        shutdown(plan->entry[0].fd, SHUT_RDWR);
    }
    if (rc == 0) {
        while (!dist_prefill_reader_wait_emit_progress(&reader, &reported_chunks)) {
            ;
        }
    }
    pthread_join(reader_tid, NULL);
    const double pipeline_t1 = dist_now_sec();
    if (rc == 0 && reader.rc == 0) {
        const double total_sec = pipeline_t1 - pipeline_t0;
        DIST_COORD_DEBUG(state,
                         "ds4: distributed coordinator: pipelined prefill done tokens=%u chunks=%u total=%.3fs %.2f t/s local=%.3fs send=%.3fs %.2f MiB/s\n",
                         total,
                         chunk_count,
                         total_sec,
                         total_sec > 0.0 ? (double)total / total_sec : 0.0,
                         local_eval_sec,
                         sender.send_sec,
                         sender.send_sec > 0.0
                             ? ((double)sender.send_bytes / (1024.0 * 1024.0)) / sender.send_sec
                             : 0.0);
    }
    if (reader.rc != 0) {
        if (errlen) snprintf(err, errlen, "%s", reader.err[0] ? reader.err : "distributed pipelined prefill failed");
        int reader_rc = reader.rc;
        free(reader.final_payload);
        free(reader.expected_hashes);
        dist_prefill_sender_destroy(&sender);
        pthread_cond_destroy(&reader.progress_cv);
        pthread_mutex_destroy(&reader.progress_mu);
        return reader_rc;
    }
    dist_prefill_sender_destroy(&sender);
    free(reader.expected_hashes);
    pthread_cond_destroy(&reader.progress_cv);
    pthread_mutex_destroy(&reader.progress_mu);
    if (rc != 0) {
        free(reader.final_payload);
        return 1;
    }
    const uint32_t logits_bytes =
        (uint32_t)((uint64_t)ds4_engine_vocab_size(state->engine) * sizeof(float));
    if (reader.final_kind == DS4_DIST_RESULT_LOGITS &&
        reader.final_payload_bytes == logits_bytes) {
        memcpy(logits, reader.final_payload, logits_bytes);
        free(reader.final_payload);
        return 0;
    }
    if (reader.final_kind == DS4_DIST_RESULT_HIDDEN_STATE &&
        reader.final_payload) {
        const uint32_t last_pos = (chunk_count - 1u) * chunk_cap;
        const uint32_t last_tokens = total - last_pos;
        int head_rc = ds4_session_eval_output_head_from_hc(session,
                                                           reader.final_payload,
                                                           last_tokens,
                                                           logits,
                                                           err,
                                                           errlen);
        free(reader.final_payload);
        return head_rc;
    }
    free(reader.final_payload);
    if (errlen) snprintf(err, errlen, "distributed pipelined prefill did not return a final result");
    return 1;
}

static int dist_coordinator_prefill_prompt(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        char *err,
        size_t errlen) {
    uint32_t chunk_cap = 0;
    if (dist_coordinator_prefill_chunk_cap(state, session, &chunk_cap, err, errlen) != 0) {
        return 1;
    }
    const uint32_t prompt_len = (uint32_t)prompt->len;
    if (dist_coordinator_can_pipeline_prefill(state, plan, session, prompt_len, chunk_cap)) {
        return dist_coordinator_prefill_prompt_pipelined(state,
                                                        session,
                                                        plan,
                                                        prompt,
                                                        0,
                                                        prompt_len,
                                                        true,
                                                        chunk_cap,
                                                        session_id,
                                                        request_id,
                                                        logits,
                                                        err,
                                                        errlen);
    }

    uint32_t pos = 0;
    while (pos < prompt_len) {
        uint32_t remaining = prompt_len - pos;
        uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
        int eval_rc = dist_coordinator_eval_span(state, session, plan,
                                                 prompt->v + pos, chunk, pos,
                                                 session_id, (*request_id)++,
                                                 pos == 0, logits, err, errlen);
        if (eval_rc != 0) {
            return eval_rc;
        }
        pos += chunk;
        dist_report_prefill_progress(session, pos, prompt_len);
    }
    return 0;
}

