/* =========================================================================
 * ds4_distributed.c - Distributed inference runtime.
 * =========================================================================
 *
 * This module owns the DS4 distributed transport and orchestration layer. The
 * rest of the engine still sees a normal ds4_session: when distributed mode is
 * active, ds4.c delegates sync/eval/save/load to the coordinator session API in
 * this file.
 *
 * Workers execute contiguous model slices with the same graph-slice entry
 * points used by the local engine. KV snapshots remain topology-independent:
 * save gathers worker-owned layer tensors into the normal DSV4 payload, and
 * load splits a normal DSV4 payload across the currently registered route.
 */

#include "ds4_distributed.h"

#include <arpa/inet.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <netdb.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* =========================================================================
 * Protocol Constants And Wire Records
 * ========================================================================= */

#define DS4_DIST_MAGIC 0x44533444u /* DS4D */
#define DS4_DIST_MSG_HELLO 1u
#define DS4_DIST_MSG_ERROR 2u
#define DS4_DIST_MSG_WORK 3u
#define DS4_DIST_MSG_RESULT 4u
#define DS4_DIST_MSG_SNAPSHOT_SAVE_REQ 5u
#define DS4_DIST_MSG_SNAPSHOT_BEGIN 6u
#define DS4_DIST_MSG_SNAPSHOT_CHUNK 7u
#define DS4_DIST_MSG_SNAPSHOT_DONE 8u
#define DS4_DIST_MSG_SNAPSHOT_LOAD_BEGIN 9u
#define DS4_DIST_MAX_MODEL_NAME 127u
#define DS4_DIST_WORK_F_INPUT_HC 0x00000001u
#define DS4_DIST_WORK_F_OUTPUT_LOGITS 0x00000002u
#define DS4_DIST_WORK_F_RESET_SESSION 0x00000004u
#define DS4_DIST_WORK_F_ACK_ONLY 0x00000008u
#define DS4_DIST_WORK_F_VALID_MASK \
    (DS4_DIST_WORK_F_INPUT_HC | DS4_DIST_WORK_F_OUTPUT_LOGITS | \
     DS4_DIST_WORK_F_RESET_SESSION | DS4_DIST_WORK_F_ACK_ONLY)
#define DS4_DIST_RESULT_ACK 0u
#define DS4_DIST_RESULT_HIDDEN_STATE 1u
#define DS4_DIST_RESULT_LOGITS 2u
#define DS4_DIST_ACTIVATION_BITS_DEFAULT 32u
#define DS4_DIST_ROUTE_F_OUTPUT_LOGITS 0x00000001u
#define DS4_DIST_ROUTE_RETURN_UPSTREAM 1u
#define DS4_DIST_RECV_TRANSPORT_ERROR 1
#define DS4_DIST_RECV_REMOTE_ERROR 2
#define DS4_DIST_SNAPSHOT_CHUNK_BYTES (8u * 1024u * 1024u)

typedef struct {
    uint32_t magic;
    uint32_t type;
    uint32_t bytes;
} ds4_dist_frame_header;

typedef struct {
    uint32_t model_id;
    uint32_t quant_bits;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t has_output;
    uint32_t has_hidden;
    uint32_t ctx_size;
    uint32_t n_layers;
    uint32_t listen_port;
    uint32_t model_name_len;
} ds4_dist_hello_fixed;

typedef struct {
    uint32_t model_id;
    uint32_t session_hi;
    uint32_t session_lo;
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t prefix_hash_hi;
    uint32_t prefix_hash_lo;
    uint32_t result_hash_hi;
    uint32_t result_hash_lo;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t flags;
    uint32_t token_bytes;
    uint32_t input_hc_bytes;
    uint32_t input_hc_bits;
    uint32_t route_count;
    uint32_t route_index;
    uint32_t route_bytes;
} ds4_dist_work_fixed;

typedef struct {
    uint32_t host_len;
    uint32_t port;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t flags;
} ds4_dist_route_fixed;

typedef struct {
    uint32_t kind;
    uint32_t host_len;
    uint32_t port;
} ds4_dist_route_return_fixed;

typedef struct {
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t result_hash_hi;
    uint32_t result_hash_lo;
    uint32_t status;
    uint32_t result_kind;
    uint32_t telemetry_count;
    uint32_t telemetry_bytes;
    uint32_t payload_bytes;
    uint32_t payload_bits;
} ds4_dist_result_fixed;

typedef struct {
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t route_index;
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t eval_usec;
    uint32_t downstream_wait_usec;
    uint32_t forward_send_usec;
    uint32_t input_bytes;
    uint32_t output_bytes;
} ds4_dist_telemetry_fixed;

typedef struct {
    uint32_t model_id;
    uint32_t session_hi;
    uint32_t session_lo;
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t token_hash_hi;
    uint32_t token_hash_lo;
    uint32_t token_count;
    uint32_t layer_start;
    uint32_t layer_end;
} ds4_dist_snapshot_req_fixed;

typedef struct {
    uint32_t model_id;
    uint32_t session_hi;
    uint32_t session_lo;
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t token_hash_hi;
    uint32_t token_hash_lo;
    uint32_t token_count;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t payload_hi;
    uint32_t payload_lo;
    uint32_t status;
    uint32_t token_bytes;
    uint32_t message_bytes;
} ds4_dist_snapshot_begin_fixed;

typedef struct {
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t chunk_bytes;
} ds4_dist_snapshot_chunk_fixed;

typedef struct {
    uint32_t request_hi;
    uint32_t request_lo;
    uint32_t status;
    uint32_t message_bytes;
} ds4_dist_snapshot_done_fixed;

/* =========================================================================
 * Runtime State
 * =========================================================================
 *
 * The coordinator registry is shared by the accept thread and by the session
 * calls made from the main inference thread. Workers keep per-session KV state
 * keyed by the coordinator-provided session ID so independent callers do not
 * share token timelines by accident.
 */

typedef struct ds4_dist_worker_entry {
    int fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
    char model_name[DS4_DIST_MAX_MODEL_NAME + 1u];
    uint32_t model_id;
    uint32_t quant_bits;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t has_output;
    uint32_t has_hidden;
    uint32_t ctx_size;
    uint32_t n_layers;
    uint32_t listen_port;
    struct ds4_dist_worker_entry *next;
} ds4_dist_worker_entry;

typedef struct {
    ds4_engine *engine;
    uint32_t model_id;
    uint32_t n_layers;
    uint32_t local_start;
    uint32_t local_end;
    uint32_t ctx_size;
    bool local_has_output;
    bool local_can_output_head;
    bool replay_check;
    bool debug;
    bool use_control_for_work;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    uint64_t generation;
    pthread_mutex_t mu;
    ds4_dist_worker_entry *workers;
    bool shutting_down;
} ds4_dist_coordinator_state;

typedef struct {
    ds4_dist_coordinator_state *state;
    int fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
} ds4_dist_client_ctx;

typedef struct {
    ds4_dist_coordinator_state *state;
    int listen_fd;
} ds4_dist_accept_ctx;

typedef struct ds4_dist_worker_session {
    uint64_t session_id;
    uint64_t token_hash;
    bool token_hash_valid;
    ds4_session *session;
    struct ds4_dist_worker_session *next;
} ds4_dist_worker_session;

typedef struct {
    ds4_engine *engine;
    uint32_t model_id;
    uint32_t layer_start;
    uint32_t layer_end;
    bool has_output;
    int ctx_size;
    int listen_fd;
    pthread_mutex_t mu;
    ds4_dist_worker_session *sessions;
} ds4_dist_worker_state;

typedef struct ds4_dist_worker_upstream ds4_dist_worker_upstream;

typedef struct ds4_dist_pending_request {
    uint64_t request_id;
    double downstream_t0;
    ds4_dist_telemetry_fixed telemetry;
    struct ds4_dist_pending_request *next;
} ds4_dist_pending_request;

typedef struct ds4_dist_worker_forwarder {
    ds4_dist_worker_upstream *upstream;
    char host[NI_MAXHOST];
    uint32_t port;
    int fd;
    pthread_t tid;
    bool thread_started;
    pthread_mutex_t send_mu;
    pthread_mutex_t queue_mu;
    pthread_cond_t queue_not_full;
    ds4_dist_pending_request *pending_head;
    ds4_dist_pending_request *pending_tail;
    uint32_t pending_count;
    uint32_t pending_depth;
    bool closing;
    struct ds4_dist_worker_forwarder *next;
} ds4_dist_worker_forwarder;

struct ds4_dist_worker_upstream {
    ds4_dist_worker_state *state;
    int fd;
    pthread_mutex_t write_mu;
    pthread_mutex_t forward_mu;
    ds4_dist_worker_forwarder *forwarders;
};

typedef struct ds4_dist_worker_job {
    void *payload;
    uint32_t bytes;
    struct ds4_dist_worker_job *next;
} ds4_dist_worker_job;

typedef struct {
    ds4_dist_worker_state *state;
    ds4_dist_worker_upstream *upstream;
    pthread_mutex_t mu;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
    ds4_dist_worker_job *head;
    ds4_dist_worker_job *tail;
    uint32_t queued;
    uint32_t depth;
    bool closed;
    bool canceled;
    int rc;
} ds4_dist_worker_job_queue;

typedef struct {
    ds4_dist_worker_state *state;
    int fd;
    char peer_host[NI_MAXHOST];
    char peer_port[NI_MAXSERV];
} ds4_dist_data_client_ctx;

typedef struct {
    char host[NI_MAXHOST];
    uint32_t port;
    uint32_t kind;
} ds4_dist_route_return;

typedef struct {
    char host[NI_MAXHOST];
    uint32_t port;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t flags;
    int fd;
} ds4_dist_route_entry;

typedef struct {
    ds4_dist_route_entry *entry;
    uint32_t count;
    void *blob;
    uint32_t blob_bytes;
} ds4_dist_route_plan;

typedef struct {
    uint32_t ctx;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t raw_window;
    uint32_t comp_cap;
    uint32_t token_count;
    uint32_t n_layers;
    uint32_t head_dim;
    uint32_t indexer_head_dim;
    uint32_t vocab;
    uint32_t raw_live;
} ds4_dist_kv_layout;

typedef struct {
    FILE *fp;
    uint64_t bytes;
    uint32_t layer_start;
    uint32_t layer_end;
    uint64_t tensor_offset;
    uint64_t tensor_bytes;
} ds4_dist_kv_shard_file;

struct ds4_dist_session {
    ds4_dist_coordinator_state state;
    int listen_fd;
    pthread_t accept_tid;
    bool accept_started;
    ds4_dist_accept_ctx accept_ctx;
    ds4_dist_route_plan plan;
    bool plan_ready;
    uint64_t plan_generation;
    uint64_t session_id;
    uint64_t request_id;
    uint64_t snapshot_request_id;
};

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_dist_logprob;

typedef struct {
    ds4_dist_coordinator_state *state;
    int fd;
    ds4_session *progress_session;
    uint64_t first_request_id;
    uint64_t *expected_hashes;
    uint32_t count;
    uint32_t total_tokens;
    uint32_t chunk_cap;
    uint32_t progress_base;
    uint32_t progress_total;
    uint32_t progress_completed;
    bool progress_done;
    uint64_t hc_values;
    bool allow_hidden;
    uint32_t final_kind;
    void *final_payload;
    uint32_t final_payload_bytes;
    int rc;
    char err[256];
    pthread_mutex_t progress_mu;
    pthread_cond_t progress_cv;
} ds4_dist_prefill_result_reader;

typedef struct {
    uint32_t pos;
    uint32_t n_tokens;
    uint32_t hidden_bytes;
    uint64_t request_id;
    uint64_t prefix_hash;
    uint64_t result_hash;
    bool reset_session;
    bool ack_only;
    float *hidden;
} ds4_dist_prefill_send_slot;

typedef struct {
    ds4_dist_coordinator_state *state;
    const ds4_dist_route_plan *plan;
    const ds4_tokens *prompt;
    uint64_t session_id;
    int fd;
    ds4_dist_prefill_send_slot *slots;
    uint32_t slot_count;
    uint32_t head;
    uint32_t tail;
    uint32_t queued;
    bool producer_done;
    bool stop;
    int rc;
    double send_sec;
    uint64_t send_bytes;
    char err[256];
    pthread_mutex_t mu;
    pthread_cond_t can_enqueue;
    pthread_cond_t can_dequeue;
} ds4_dist_prefill_sender;

/* =========================================================================
 * Small Utilities And Forward Declarations
 * ========================================================================= */

static uint32_t dist_prefill_send_depth(uint32_t chunk_count) {
    uint32_t depth = 2;
    const char *env = getenv("DS4_DIST_PREFILL_SEND_DEPTH");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 1 && v <= 8) {
            depth = (uint32_t)v;
        }
    }
    if (chunk_count != 0 && depth > chunk_count) depth = chunk_count;
    return depth ? depth : 1;
}

static int dist_send_work_frame(
        int fd,
        const ds4_dist_work_fixed *work,
        const int *tokens,
        const float *input_hc,
        const void *route_blob);
static int dist_write_full(int fd, const void *buf, size_t len);
static int dist_send_snapshot_file_chunks(
        int fd,
        uint64_t request_id,
        FILE *fp,
        uint64_t bytes);

static int dist_worker_handle_work(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes);
static int dist_worker_handle_snapshot_save(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes);
static int dist_worker_handle_snapshot_load(
        ds4_dist_worker_state *state,
        ds4_dist_worker_upstream *upstream,
        uint32_t bytes);
static void dist_worker_upstream_init(
        ds4_dist_worker_upstream *upstream,
        ds4_dist_worker_state *state,
        int fd);
static void dist_worker_upstream_destroy(ds4_dist_worker_upstream *upstream);
static int dist_worker_upstream_send_work_error(
        ds4_dist_worker_upstream *upstream,
        uint64_t request_id,
        const char *msg);
static int dist_coordinator_prefill_prompt(
        ds4_dist_coordinator_state *state,
        ds4_session *session,
        const ds4_dist_route_plan *plan,
        const ds4_tokens *prompt,
        uint64_t session_id,
        uint64_t *request_id,
        float *logits,
        char *err,
        size_t errlen);
static int dist_validate_options(const ds4_dist_options *opt, char *err, size_t errlen);

static uint32_t dist_resolved_layer_end(const ds4_dist_options *opt, uint32_t n_layers) {
    if (opt->layers.has_output) return n_layers - 1u;
    return opt->layers.end;
}

static const char *dist_role_name(ds4_distributed_role role) {
    switch (role) {
    case DS4_DISTRIBUTED_NONE:        return "none";
    case DS4_DISTRIBUTED_COORDINATOR: return "coordinator";
    case DS4_DISTRIBUTED_WORKER:      return "worker";
    }
    return "unknown";
}

static void dist_sleep_reconnect(void) {
    sleep(1);
}

static double dist_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

/* =========================================================================
 * Local File And Size Helpers
 * ========================================================================= */

static int dist_payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes, char *err, size_t errlen) {
    const uint8_t *p = ptr;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fwrite(p, 1, n, fp) != n) {
            if (errlen) snprintf(err, errlen, "failed to write distributed payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    return 0;
}

static int dist_payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes, uint64_t *remaining, char *err, size_t errlen) {
    if (remaining && *remaining < bytes) {
        if (errlen) snprintf(err, errlen, "truncated distributed payload");
        return 1;
    }
    uint8_t *p = ptr;
    uint64_t original = bytes;
    while (bytes != 0) {
        const size_t n = bytes > (uint64_t)SIZE_MAX ? SIZE_MAX : (size_t)bytes;
        if (fread(p, 1, n, fp) != n) {
            if (errlen) snprintf(err, errlen, "failed to read distributed payload");
            return 1;
        }
        p += n;
        bytes -= n;
    }
    if (remaining) *remaining -= original;
    return 0;
}

static int dist_payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    uint8_t b[4] = {
        (uint8_t)v,
        (uint8_t)(v >> 8),
        (uint8_t)(v >> 16),
        (uint8_t)(v >> 24),
    };
    return dist_payload_write_bytes(fp, b, sizeof(b), err, errlen);
}

static int dist_payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining, char *err, size_t errlen) {
    uint8_t b[4];
    if (dist_payload_read_bytes(fp, b, sizeof(b), remaining, err, errlen) != 0) return 1;
    *v = (uint32_t)b[0] |
         ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) |
         ((uint32_t)b[3] << 24);
    return 0;
}

static int dist_payload_copy_bytes(
        FILE *src,
        FILE *dst,
        uint64_t bytes,
        uint64_t *remaining,
        char *err,
        size_t errlen) {
    if (remaining && *remaining < bytes) {
        if (errlen) snprintf(err, errlen, "truncated distributed payload");
        return 1;
    }
    uint8_t *buf = malloc(DS4_DIST_SNAPSHOT_CHUNK_BYTES);
    if (!buf) {
        if (errlen) snprintf(err, errlen, "out of memory copying distributed payload");
        return 1;
    }
    int rc = 0;
    uint64_t left = bytes;
    while (left != 0) {
        const size_t n = left > DS4_DIST_SNAPSHOT_CHUNK_BYTES ?
            DS4_DIST_SNAPSHOT_CHUNK_BYTES : (size_t)left;
        if (fread(buf, 1, n, src) != n) {
            if (errlen) snprintf(err, errlen, "failed to read distributed payload");
            rc = 1;
            break;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            if (errlen) snprintf(err, errlen, "failed to write distributed payload");
            rc = 1;
            break;
        }
        left -= n;
    }
    free(buf);
    if (rc == 0 && remaining) *remaining -= bytes;
    return rc;
}

static int dist_copy_file_range(
        FILE *src,
        uint64_t offset,
        uint64_t bytes,
        FILE *dst,
        char *err,
        size_t errlen) {
    if (offset > (uint64_t)LLONG_MAX || fseeko(src, (off_t)offset, SEEK_SET) != 0) {
        if (errlen) snprintf(err, errlen, "failed to seek distributed KV shard");
        return 1;
    }
    return dist_payload_copy_bytes(src, dst, bytes, NULL, err, errlen);
}

static int dist_rewind_file(FILE *fp, const char *what, char *err, size_t errlen) {
    if (fflush(fp) != 0 || fseeko(fp, 0, SEEK_SET) != 0) {
        if (errlen) snprintf(err, errlen, "failed to rewind %s", what);
        return 1;
    }
    return 0;
}

static int dist_measure_file(FILE *fp, uint64_t *bytes, const char *what, char *err, size_t errlen) {
    if (!bytes) return 1;
    if (fflush(fp) != 0) {
        if (errlen) snprintf(err, errlen, "failed to flush %s", what);
        return 1;
    }
    off_t pos = ftello(fp);
    if (pos < 0) {
        if (errlen) snprintf(err, errlen, "failed to measure %s", what);
        return 1;
    }
    *bytes = (uint64_t)pos;
    return 0;
}

static FILE *dist_tmpfile_or_err(const char *what, char *err, size_t errlen) {
    FILE *fp = tmpfile();
    if (!fp && errlen) snprintf(err, errlen, "failed to create %s temp file: %s",
                                what, strerror(errno));
    return fp;
}

static bool dist_u64_add(uint64_t *acc, uint64_t add) {
    if (!acc || *acc > UINT64_MAX - add) return false;
    *acc += add;
    return true;
}

static bool dist_u64_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out) return false;
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

/* =========================================================================
 * Tunable Limits
 * ========================================================================= */

static int dist_socket_buffer_bytes(void) {
    int mb = 128;
    const char *env = getenv("DS4_DIST_SOCKET_BUFFER_MB");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 0 && v <= 512) {
            mb = (int)v;
        }
    }
    return mb > 0 ? mb * 1024 * 1024 : 0;
}

static uint32_t dist_worker_prefetch_depth(void) {
    uint32_t depth = 2;
    const char *env = getenv("DS4_DIST_WORKER_PREFETCH_DEPTH");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 1 && v <= 8) {
            depth = (uint32_t)v;
        }
    }
    return depth;
}

static uint32_t dist_worker_forward_window(void) {
    uint32_t depth = 4;
    const char *env = getenv("DS4_DIST_WORKER_FORWARD_WINDOW");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && v >= 1 && v <= 64) {
            depth = (uint32_t)v;
        }
    }
    return depth;
}

static bool dist_decode_profile_enabled(void) {
    return getenv("DS4_DIST_DECODE_PROFILE") != NULL;
}

static bool dist_parse_positive_u32(
        const char *s,
        const char *name,
        uint32_t *out,
        char *err,
        size_t errlen) {
    if (!s || !out) {
        if (errlen) snprintf(err, errlen, "%s requires a positive integer", name);
        return false;
    }
    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || s[0] == '\0' || *end != '\0' || v == 0 || v > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "invalid value for %s: %s", name, s);
        return false;
    }
    *out = (uint32_t)v;
    return true;
}

/* =========================================================================
 * Activation Transport
 * =========================================================================
 *
 * The graph-slice APIs exchange float buffers. Distributed transport can leave
 * those buffers as 32-bit floats or pack them to 16/8 bits on the wire; workers
 * decode back to float before executing the next slice.
 */

static uint32_t dist_activation_bits_or_default(uint32_t bits) {
    return bits ? bits : DS4_DIST_ACTIVATION_BITS_DEFAULT;
}

static bool dist_activation_bits_valid(uint32_t bits) {
    bits = dist_activation_bits_or_default(bits);
    return bits == 32u || bits == 16u || bits == 8u;
}

static bool dist_activation_wire_bytes(uint32_t bits, uint64_t values, uint32_t *out) {
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits) || (bits % 8u) != 0) return false;
    const uint64_t bytes = values * (uint64_t)(bits / 8u);
    if (bytes > UINT32_MAX) return false;
    if (out) *out = (uint32_t)bytes;
    return true;
}

static bool dist_activation_values_from_wire_bytes(uint32_t bits, uint32_t bytes, uint64_t *out) {
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits) || (bits % 8u) != 0) return false;
    const uint32_t bytes_per_value = bits / 8u;
    if (bytes_per_value == 0 || (bytes % bytes_per_value) != 0) return false;
    if (out) *out = bytes / bytes_per_value;
    return true;
}

static bool dist_activation_wire_bytes_from_f32_bytes(uint32_t bits, uint32_t f32_bytes, uint32_t *out) {
    if ((f32_bytes % (uint32_t)sizeof(float)) != 0) return false;
    return dist_activation_wire_bytes(bits, f32_bytes / (uint32_t)sizeof(float), out);
}

static uint16_t dist_f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }

    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

static float dist_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int32_t exp = (int32_t)((h >> 10) & 0x1fu);
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint8_t dist_f32_to_f8_e4m3(float f) {
    const uint8_t sign = signbit(f) ? 0x80u : 0u;
    float a = fabsf(f);
    if (a == 0.0f) return sign;
    if (!isfinite(a) || a >= 240.0f) return (uint8_t)(sign | 0x77u);

    if (a < 0.001953125f) {
        int mant = (int)floorf(a * 512.0f + 0.5f);
        if (mant <= 0) return sign;
        if (mant > 7) mant = 7;
        return (uint8_t)(sign | (uint8_t)mant);
    }

    int exp2 = 0;
    (void)frexpf(a, &exp2);
    int exp = exp2 - 1 + 7;
    if (exp <= 0) {
        int mant = (int)floorf(a * 512.0f + 0.5f);
        if (mant <= 0) return sign;
        if (mant > 7) mant = 7;
        return (uint8_t)(sign | (uint8_t)mant);
    }

    float base = ldexpf(1.0f, exp2 - 1);
    int mant = (int)floorf(((a / base) - 1.0f) * 8.0f + 0.5f);
    if (mant >= 8) {
        mant = 0;
        exp++;
    }
    if (exp >= 15) return (uint8_t)(sign | 0x77u);
    return (uint8_t)(sign | (uint8_t)(exp << 3) | (uint8_t)mant);
}

static float dist_f8_e4m3_to_f32(uint8_t h) {
    const float sign = (h & 0x80u) ? -1.0f : 1.0f;
    const uint32_t exp = (h >> 3) & 0x0fu;
    const uint32_t mant = h & 0x07u;
    if (exp == 0) {
        return sign * (float)mant * 0.001953125f;
    }
    if (exp >= 15u) {
        return sign * 240.0f;
    }
    return sign * ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
}

static int dist_write_activation_payload(
        int fd,
        const float *src,
        uint64_t values,
        uint32_t bits) {
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits)) return -1;
    if (values == 0) return 0;
    if (!src) return -1;
    if (bits == 32u) {
        uint32_t bytes = 0;
        if (!dist_activation_wire_bytes(bits, values, &bytes)) return -1;
        return dist_write_full(fd, src, bytes);
    }

    const uint64_t max_values = 1024u * 1024u;
    uint64_t cap = values < max_values ? values : max_values;
    void *buf = malloc((size_t)cap * (size_t)(bits / 8u));
    if (!buf) return -1;
    uint64_t done = 0;
    int rc = 0;
    while (done < values) {
        uint64_t n = values - done;
        if (n > cap) n = cap;
        if (bits == 16u) {
            uint16_t *dst = buf;
            for (uint64_t i = 0; i < n; i++) dst[i] = dist_f32_to_f16(src[done + i]);
        } else {
            uint8_t *dst = buf;
            for (uint64_t i = 0; i < n; i++) dst[i] = dist_f32_to_f8_e4m3(src[done + i]);
        }
        if (dist_write_full(fd, buf, (size_t)n * (size_t)(bits / 8u)) != 0) {
            rc = -1;
            break;
        }
        done += n;
    }
    free(buf);
    return rc;
}

static int dist_decode_activation_payload(
        const void *wire,
        uint32_t bits,
        uint32_t wire_bytes,
        float **out,
        uint32_t *out_f32_bytes,
        bool *out_uses_wire,
        char *err,
        size_t errlen) {
    if (out) *out = NULL;
    if (out_f32_bytes) *out_f32_bytes = 0;
    if (out_uses_wire) *out_uses_wire = false;
    bits = dist_activation_bits_or_default(bits);
    if (!dist_activation_bits_valid(bits)) {
        if (errlen) snprintf(err, errlen, "invalid distributed activation width: %u bits", bits);
        return 1;
    }
    if (wire_bytes != 0 && !wire) {
        if (errlen) snprintf(err, errlen, "missing distributed activation payload");
        return 1;
    }

    uint64_t values = 0;
    if (!dist_activation_values_from_wire_bytes(bits, wire_bytes, &values)) {
        if (errlen) snprintf(err, errlen, "invalid distributed activation payload size");
        return 1;
    }
    const uint64_t f32_bytes64 = values * sizeof(float);
    if (f32_bytes64 > UINT32_MAX) {
        if (errlen) snprintf(err, errlen, "distributed activation payload is too large");
        return 1;
    }
    const uint32_t f32_bytes = (uint32_t)f32_bytes64;
    if (bits == 32u) {
        if (out) *out = (float *)(void *)wire;
        if (out_f32_bytes) *out_f32_bytes = f32_bytes;
        if (out_uses_wire) *out_uses_wire = true;
        return 0;
    }

    float *dst = f32_bytes ? malloc(f32_bytes) : NULL;
    if (f32_bytes && !dst) {
        if (errlen) snprintf(err, errlen, "out of memory decoding distributed activations");
        return 1;
    }
    if (bits == 16u) {
        const uint16_t *src = wire;
        for (uint64_t i = 0; i < values; i++) dst[i] = dist_f16_to_f32(src[i]);
    } else {
        const uint8_t *src = wire;
        for (uint64_t i = 0; i < values; i++) dst[i] = dist_f8_e4m3_to_f32(src[i]);
    }
    if (out) *out = dst;
    if (out_f32_bytes) *out_f32_bytes = f32_bytes;
    return 0;
}

