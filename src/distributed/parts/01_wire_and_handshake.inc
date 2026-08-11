/* =========================================================================
 * TCP Framing And Connections
 * ========================================================================= */

static int dist_set_socket_low_latency(int fd) {
    int one = 1;
    int rc = 0;
    int buffer_bytes = dist_socket_buffer_bytes();
    int send_timeout_sec = 60;
    const char *timeout_env = getenv("DS4_DIST_SOCKET_TIMEOUT_SEC");
    if (timeout_env && timeout_env[0]) {
        char *end = NULL;
        long v = strtol(timeout_env, &end, 10);
        if (end != timeout_env && *end == '\0' && v > 0 && v <= 3600)
            send_timeout_sec = (int)v;
    }
    struct timeval send_tv = {
        .tv_sec = send_timeout_sec,
        .tv_usec = 0,
    };
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0) rc = -1;
    if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one)) != 0) rc = -1;
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &send_tv, sizeof(send_tv)) != 0) rc = -1;
    /* Do not install a receive timeout by default.  Worker control sockets can
     * be legitimately idle while a KV snapshot is transferred on a separate
     * data connection; timing out that read drops an otherwise healthy route. */
    const char *recv_timeout_env = getenv("DS4_DIST_SOCKET_RECV_TIMEOUT_SEC");
    if (recv_timeout_env && recv_timeout_env[0]) {
        char *end = NULL;
        long v = strtol(recv_timeout_env, &end, 10);
        if (end != recv_timeout_env && *end == '\0' && v > 0 && v <= 3600) {
            struct timeval recv_tv = {
                .tv_sec = (int)v,
                .tv_usec = 0,
            };
            if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                           &recv_tv, sizeof(recv_tv)) != 0) rc = -1;
        }
    }
    if (buffer_bytes > 0 &&
        setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buffer_bytes, sizeof(buffer_bytes)) != 0) rc = -1;
    if (buffer_bytes > 0 &&
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffer_bytes, sizeof(buffer_bytes)) != 0) rc = -1;
#ifdef SO_NOSIGPIPE
    if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)) != 0) rc = -1;
#endif
    return rc;
}

#ifdef DS4_DIST_TRACE
#define DIST_DEBUG(...) do { \
    fprintf(stderr, "ds4: distributed debug: " __VA_ARGS__); \
    fputc('\n', stderr); \
} while (0)
#else
#define DIST_DEBUG(...) ((void)0)
#endif

static int dist_write_full(int fd, const void *buf, size_t len) {
    const unsigned char *p = buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int dist_read_full(int fd, void *buf, size_t len) {
    unsigned char *p = buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return 0;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 1;
}

static bool dist_connect_trace_enabled(void) {
    return getenv("DS4_DIST_CONNECT_TRACE") != NULL;
}

static void dist_sockaddr_string(const struct sockaddr *sa, socklen_t salen, char *buf, size_t buflen) {
    if (buflen) buf[0] = '\0';
    if (!sa || !buf || buflen == 0) return;
    char host[NI_MAXHOST];
    char service[NI_MAXSERV];
    if (getnameinfo(sa, salen,
                    host, sizeof(host),
                    service, sizeof(service),
                    NI_NUMERICHOST | NI_NUMERICSERV) == 0) {
        snprintf(buf, buflen, "%s:%s", host, service);
        return;
    }
    snprintf(buf, buflen, "unknown");
}

static int dist_write_frame_header(int fd, uint32_t type, uint32_t bytes) {
    ds4_dist_frame_header h = {
        htonl(DS4_DIST_MAGIC),
        htonl(type),
        htonl(bytes)
    };
    return dist_write_full(fd, &h, sizeof(h));
}

static int dist_read_frame_header(int fd, uint32_t *type, uint32_t *bytes, char *err, size_t errlen) {
    ds4_dist_frame_header h;
    int rc = dist_read_full(fd, &h, sizeof(h));
    if (rc < 0 && errlen) snprintf(err, errlen, "failed to read frame header: %s", strerror(errno));
    if (rc <= 0) return rc;

    uint32_t magic = ntohl(h.magic);
    if (magic != DS4_DIST_MAGIC) {
        if (errlen) snprintf(err, errlen, "bad frame magic 0x%08x", magic);
        return -1;
    }

    *type = ntohl(h.type);
    *bytes = ntohl(h.bytes);
    return 1;
}

static int dist_discard_bytes(int fd, uint32_t bytes) {
    unsigned char buf[4096];
    while (bytes > 0) {
        size_t n = bytes < sizeof(buf) ? bytes : sizeof(buf);
        int rc = dist_read_full(fd, buf, n);
        if (rc <= 0) return rc == 0 ? 0 : -1;
        bytes -= (uint32_t)n;
    }
    return 1;
}

static int dist_send_error(int fd, const char *msg) {
    if (!msg) msg = "distributed protocol error";
    size_t len = strlen(msg);
    if (len > UINT32_MAX) len = UINT32_MAX;
    if (dist_write_frame_header(fd, DS4_DIST_MSG_ERROR, (uint32_t)len) != 0) return -1;
    return dist_write_full(fd, msg, len);
}

static void dist_peer_name(int fd, char *host, size_t hostlen, char *port, size_t portlen) {
    if (hostlen) host[0] = '\0';
    if (portlen) port[0] = '\0';

    struct sockaddr_storage ss;
    socklen_t slen = sizeof(ss);
    if (getpeername(fd, (struct sockaddr *)&ss, &slen) == 0) {
        if (getnameinfo((struct sockaddr *)&ss, slen,
                        host, (socklen_t)hostlen,
                        port, (socklen_t)portlen,
                        NI_NUMERICHOST | NI_NUMERICSERV) == 0) {
            return;
        }
    }
    if (hostlen) snprintf(host, hostlen, "unknown");
    if (portlen) snprintf(port, portlen, "0");
}

static int dist_open_listener(const char *host, int port, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    const char *host_display = host ? host : "*";

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    hints.ai_flags = AI_PASSIVE;

    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, portbuf, &hints, &res);
    if (gai != 0) {
        if (errlen) snprintf(err, errlen, "getaddrinfo(%s:%s): %s", host_display, portbuf, gai_strerror(gai));
        return -1;
    }

    int listen_fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;

        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        dist_set_socket_low_latency(fd);

        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 &&
            listen(fd, 64) == 0) {
            listen_fd = fd;
            break;
        }

        close(fd);
    }

    freeaddrinfo(res);
    if (listen_fd < 0 && errlen) {
        snprintf(err, errlen, "unable to listen on %s:%d: %s", host_display, port, strerror(errno));
    }
    return listen_fd;
}

static int dist_listener_port(int fd) {
    struct sockaddr_storage ss;
    socklen_t slen = sizeof(ss);
    if (getsockname(fd, (struct sockaddr *)&ss, &slen) != 0) return 0;
    char service[NI_MAXSERV];
    if (getnameinfo((struct sockaddr *)&ss, slen,
                    NULL, 0,
                    service, sizeof(service),
                    NI_NUMERICSERV) != 0) {
        return 0;
    }
    char *end = NULL;
    unsigned long v = strtoul(service, &end, 10);
    if (end == service || *end != '\0' || v > 65535ul) return 0;
    return (int)v;
}

static bool dist_connect_errno_retryable(int e) {
    return e == ECONNREFUSED ||
           e == EHOSTUNREACH ||
           e == ENETUNREACH ||
           e == ETIMEDOUT ||
           e == EADDRNOTAVAIL;
}

static int dist_bind_connect_source(int fd, int family, const char *host, int *bind_errno) {
    if (bind_errno) *bind_errno = 0;
    if (!host || !host[0]) return 0;

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = family;

    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, "0", &hints, &res);
    if (gai != 0) {
        if (bind_errno) *bind_errno = EINVAL;
        return -1;
    }

    int rc = -1;
    int saved_errno = EADDRNOTAVAIL;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            rc = 0;
            saved_errno = 0;
            break;
        }
        saved_errno = errno;
    }
    freeaddrinfo(res);

    if (bind_errno) *bind_errno = saved_errno;
    return rc;
}

static int dist_bind_connect_interface(int fd, int family, const char *ifname, int *bind_errno) {
    (void)fd;
    if (bind_errno) *bind_errno = 0;
    if (!ifname || !ifname[0]) return 0;

    unsigned int ifindex = if_nametoindex(ifname);
    if (ifindex == 0) {
        if (bind_errno) *bind_errno = errno ? errno : ENODEV;
        return -1;
    }

    int rc = 0;
    if (family == AF_INET) {
#ifdef IP_BOUND_IF
        rc = setsockopt(fd, IPPROTO_IP, IP_BOUND_IF, &ifindex, sizeof(ifindex));
#else
        errno = ENOTSUP;
        rc = -1;
#endif
    } else if (family == AF_INET6) {
#ifdef IPV6_BOUND_IF
        rc = setsockopt(fd, IPPROTO_IPV6, IPV6_BOUND_IF, &ifindex, sizeof(ifindex));
#else
        errno = ENOTSUP;
        rc = -1;
#endif
    }

    if (rc != 0) {
        if (bind_errno) *bind_errno = errno ? errno : EIO;
        return -1;
    }
    return 0;
}

static int dist_connect_endpoint_once(const char *host, int port, int *last_errno, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    if (last_errno) *last_errno = 0;
    const char *bind_host = getenv("DS4_DIST_CONNECT_BIND_HOST");
    if (bind_host && bind_host[0] == '\0') bind_host = NULL;
    const char *bind_if = getenv("DS4_DIST_CONNECT_BIND_IF");
    if (bind_if && bind_if[0] == '\0') bind_if = NULL;
    const bool trace = dist_connect_trace_enabled();

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;

    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, portbuf, &hints, &res);
    if (gai != 0) {
        if (errlen) snprintf(err, errlen, "getaddrinfo(%s:%s): %s", host, portbuf, gai_strerror(gai));
        if (last_errno) *last_errno = EINVAL;
        return -1;
    }

    int fd = -1;
    int saved_errno = 0;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        char dst_desc[NI_MAXHOST + NI_MAXSERV + 8];
        if (trace) {
            dist_sockaddr_string(ai->ai_addr, ai->ai_addrlen, dst_desc, sizeof(dst_desc));
            fprintf(stderr, "ds4: distributed connect trace: candidate dst=%s family=%d bind=%s if=%s\n",
                    dst_desc, ai->ai_family, bind_host ? bind_host : "(none)",
                    bind_if ? bind_if : "(none)");
        }
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            saved_errno = errno;
            if (trace) {
                fprintf(stderr, "ds4: distributed connect trace: socket failed: %s\n",
                        strerror(saved_errno));
            }
            continue;
        }
        dist_set_socket_low_latency(fd);
        if (bind_if) {
            int bind_errno = 0;
            if (dist_bind_connect_interface(fd, ai->ai_family, bind_if, &bind_errno) != 0) {
                saved_errno = bind_errno ? bind_errno : ENODEV;
                if (trace) {
                    fprintf(stderr, "ds4: distributed connect trace: bind interface %s failed: %s\n",
                            bind_if, strerror(saved_errno));
                }
                close(fd);
                fd = -1;
                continue;
            }
            if (trace) {
                fprintf(stderr, "ds4: distributed connect trace: bound interface=%s\n", bind_if);
            }
        }
        if (bind_host) {
            int bind_errno = 0;
            if (dist_bind_connect_source(fd, ai->ai_family, bind_host, &bind_errno) != 0) {
                saved_errno = bind_errno ? bind_errno : EADDRNOTAVAIL;
                if (trace) {
                    fprintf(stderr, "ds4: distributed connect trace: bind source %s failed: %s\n",
                            bind_host, strerror(saved_errno));
                }
                close(fd);
                fd = -1;
                continue;
            }
            if (trace) {
                struct sockaddr_storage local_ss;
                socklen_t local_len = sizeof(local_ss);
                char local_desc[NI_MAXHOST + NI_MAXSERV + 8];
                if (getsockname(fd, (struct sockaddr *)&local_ss, &local_len) == 0) {
                    dist_sockaddr_string((const struct sockaddr *)&local_ss, local_len,
                                         local_desc, sizeof(local_desc));
                    fprintf(stderr, "ds4: distributed connect trace: bound source=%s\n",
                            local_desc);
                }
            }
        }
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            if (trace) {
                struct sockaddr_storage local_ss;
                socklen_t local_len = sizeof(local_ss);
                char local_desc[NI_MAXHOST + NI_MAXSERV + 8];
                if (getsockname(fd, (struct sockaddr *)&local_ss, &local_len) == 0) {
                    dist_sockaddr_string((const struct sockaddr *)&local_ss, local_len,
                                         local_desc, sizeof(local_desc));
                    fprintf(stderr, "ds4: distributed connect trace: connected source=%s dst=%s\n",
                            local_desc, dst_desc);
                }
            }
            break;
        }
        saved_errno = errno;
        if (trace) {
            fprintf(stderr, "ds4: distributed connect trace: connect to %s failed: %s\n",
                    dst_desc, strerror(saved_errno));
        }
        close(fd);
        fd = -1;
    }

    freeaddrinfo(res);
    if (fd < 0 && errlen) {
        if (saved_errno == 0) saved_errno = EIO;
        snprintf(err, errlen, "unable to connect to %s:%d: %s", host, port, strerror(saved_errno));
    }
    if (fd < 0 && last_errno) *last_errno = saved_errno ? saved_errno : EIO;
    return fd;
}

static int dist_connect_endpoint(const char *host, int port, char *err, size_t errlen) {
    int last_errno = 0;
    for (int attempt = 0; attempt < 200; attempt++) {
        int fd = dist_connect_endpoint_once(host, port, &last_errno, err, errlen);
        if (fd >= 0) return fd;
        if (!dist_connect_errno_retryable(last_errno)) break;
        struct timespec ts = {0, 25 * 1000 * 1000};
        nanosleep(&ts, NULL);
    }
    return -1;
}

/* =========================================================================
 * Wire Encoding Helpers
 * ========================================================================= */

static void dist_hello_to_wire(ds4_dist_hello_fixed *h) {
    h->model_id = htonl(h->model_id);
    h->quant_bits = htonl(h->quant_bits);
    h->layer_start = htonl(h->layer_start);
    h->layer_end = htonl(h->layer_end);
    h->has_output = htonl(h->has_output);
    h->has_hidden = htonl(h->has_hidden);
    h->ctx_size = htonl(h->ctx_size);
    h->n_layers = htonl(h->n_layers);
    h->listen_port = htonl(h->listen_port);
    h->model_name_len = htonl(h->model_name_len);
}

static void dist_hello_from_wire(ds4_dist_hello_fixed *h) {
    h->model_id = ntohl(h->model_id);
    h->quant_bits = ntohl(h->quant_bits);
    h->layer_start = ntohl(h->layer_start);
    h->layer_end = ntohl(h->layer_end);
    h->has_output = ntohl(h->has_output);
    h->has_hidden = ntohl(h->has_hidden);
    h->ctx_size = ntohl(h->ctx_size);
    h->n_layers = ntohl(h->n_layers);
    h->listen_port = ntohl(h->listen_port);
    h->model_name_len = ntohl(h->model_name_len);
}

static uint64_t dist_u64_from_halves(uint32_t hi, uint32_t lo) {
    return ((uint64_t)hi << 32) | lo;
}

/* FNV-1a over little-endian token IDs.  This is not a security primitive; it is
 * a compact session invariant so distributed workers can reject same-position
 * but different-prefix KV state before doing layer work. */
#define DS4_DIST_TOKEN_HASH_INIT 1469598103934665603ull
#define DS4_DIST_TOKEN_HASH_PRIME 1099511628211ull

static uint64_t dist_token_hash_update(uint64_t h, int token) {
    uint32_t t = (uint32_t)token;
    for (int i = 0; i < 4; i++) {
        h ^= (uint64_t)((t >> (i * 8)) & 0xffu);
        h *= DS4_DIST_TOKEN_HASH_PRIME;
    }
    return h;
}

static uint64_t dist_token_hash_update_span(uint64_t h, const int *tokens, uint32_t n_tokens) {
    for (uint32_t i = 0; i < n_tokens; i++) h = dist_token_hash_update(h, tokens[i]);
    return h;
}

static uint64_t dist_token_hash_prefix(const int *tokens, uint32_t n_tokens) {
    return dist_token_hash_update_span(DS4_DIST_TOKEN_HASH_INIT, tokens, n_tokens);
}

static int dist_session_token_hash_prefix(
        ds4_session *session,
        uint32_t n_tokens,
        uint64_t *hash,
        char *err,
        size_t errlen) {
    const ds4_tokens *checkpoint = ds4_session_tokens(session);
    if (!hash || !checkpoint || checkpoint->len < 0 || (uint32_t)checkpoint->len < n_tokens) {
        if (errlen) snprintf(err, errlen, "distributed session has no %u-token prefix", n_tokens);
        return 1;
    }
    *hash = dist_token_hash_prefix(checkpoint->v, n_tokens);
    return 0;
}

static bool dist_bytes_have_nul(const void *p, uint32_t len) {
    return len != 0 && memchr(p, '\0', len) != NULL;
}

static void dist_u64_to_halves(uint64_t v, uint32_t *hi, uint32_t *lo) {
    *hi = (uint32_t)(v >> 32);
    *lo = (uint32_t)v;
}

static void dist_work_from_wire(ds4_dist_work_fixed *w) {
    w->model_id = ntohl(w->model_id);
    w->session_hi = ntohl(w->session_hi);
    w->session_lo = ntohl(w->session_lo);
    w->request_hi = ntohl(w->request_hi);
    w->request_lo = ntohl(w->request_lo);
    w->prefix_hash_hi = ntohl(w->prefix_hash_hi);
    w->prefix_hash_lo = ntohl(w->prefix_hash_lo);
    w->result_hash_hi = ntohl(w->result_hash_hi);
    w->result_hash_lo = ntohl(w->result_hash_lo);
    w->pos0 = ntohl(w->pos0);
    w->n_tokens = ntohl(w->n_tokens);
    w->layer_start = ntohl(w->layer_start);
    w->layer_end = ntohl(w->layer_end);
    w->flags = ntohl(w->flags);
    w->token_bytes = ntohl(w->token_bytes);
    w->input_hc_bytes = ntohl(w->input_hc_bytes);
    w->input_hc_bits = ntohl(w->input_hc_bits);
    w->route_count = ntohl(w->route_count);
    w->route_index = ntohl(w->route_index);
    w->route_bytes = ntohl(w->route_bytes);
}

static void dist_work_to_wire(ds4_dist_work_fixed *w) {
    w->model_id = htonl(w->model_id);
    w->session_hi = htonl(w->session_hi);
    w->session_lo = htonl(w->session_lo);
    w->request_hi = htonl(w->request_hi);
    w->request_lo = htonl(w->request_lo);
    w->prefix_hash_hi = htonl(w->prefix_hash_hi);
    w->prefix_hash_lo = htonl(w->prefix_hash_lo);
    w->result_hash_hi = htonl(w->result_hash_hi);
    w->result_hash_lo = htonl(w->result_hash_lo);
    w->pos0 = htonl(w->pos0);
    w->n_tokens = htonl(w->n_tokens);
    w->layer_start = htonl(w->layer_start);
    w->layer_end = htonl(w->layer_end);
    w->flags = htonl(w->flags);
    w->token_bytes = htonl(w->token_bytes);
    w->input_hc_bytes = htonl(w->input_hc_bytes);
    w->input_hc_bits = htonl(w->input_hc_bits);
    w->route_count = htonl(w->route_count);
    w->route_index = htonl(w->route_index);
    w->route_bytes = htonl(w->route_bytes);
}

static void dist_route_from_wire(ds4_dist_route_fixed *r) {
    r->host_len = ntohl(r->host_len);
    r->port = ntohl(r->port);
    r->layer_start = ntohl(r->layer_start);
    r->layer_end = ntohl(r->layer_end);
    r->flags = ntohl(r->flags);
}

static void dist_route_to_wire(ds4_dist_route_fixed *r) {
    r->host_len = htonl(r->host_len);
    r->port = htonl(r->port);
    r->layer_start = htonl(r->layer_start);
    r->layer_end = htonl(r->layer_end);
    r->flags = htonl(r->flags);
}

static void dist_route_return_from_wire(ds4_dist_route_return_fixed *r) {
    r->kind = ntohl(r->kind);
    r->host_len = ntohl(r->host_len);
    r->port = ntohl(r->port);
}

static void dist_route_return_to_wire(ds4_dist_route_return_fixed *r) {
    r->kind = htonl(r->kind);
    r->host_len = htonl(r->host_len);
    r->port = htonl(r->port);
}

static void dist_result_to_wire(ds4_dist_result_fixed *r) {
    r->request_hi = htonl(r->request_hi);
    r->request_lo = htonl(r->request_lo);
    r->result_hash_hi = htonl(r->result_hash_hi);
    r->result_hash_lo = htonl(r->result_hash_lo);
    r->status = htonl(r->status);
    r->result_kind = htonl(r->result_kind);
    r->telemetry_count = htonl(r->telemetry_count);
    r->telemetry_bytes = htonl(r->telemetry_bytes);
    r->payload_bytes = htonl(r->payload_bytes);
    r->payload_bits = htonl(r->payload_bits);
}

static void dist_result_from_wire(ds4_dist_result_fixed *r) {
    r->request_hi = ntohl(r->request_hi);
    r->request_lo = ntohl(r->request_lo);
    r->result_hash_hi = ntohl(r->result_hash_hi);
    r->result_hash_lo = ntohl(r->result_hash_lo);
    r->status = ntohl(r->status);
    r->result_kind = ntohl(r->result_kind);
    r->telemetry_count = ntohl(r->telemetry_count);
    r->telemetry_bytes = ntohl(r->telemetry_bytes);
    r->payload_bytes = ntohl(r->payload_bytes);
    r->payload_bits = ntohl(r->payload_bits);
}

static void dist_snapshot_req_to_wire(ds4_dist_snapshot_req_fixed *s) {
    s->model_id = htonl(s->model_id);
    s->session_hi = htonl(s->session_hi);
    s->session_lo = htonl(s->session_lo);
    s->request_hi = htonl(s->request_hi);
    s->request_lo = htonl(s->request_lo);
    s->token_hash_hi = htonl(s->token_hash_hi);
    s->token_hash_lo = htonl(s->token_hash_lo);
    s->token_count = htonl(s->token_count);
    s->layer_start = htonl(s->layer_start);
    s->layer_end = htonl(s->layer_end);
}

static void dist_snapshot_req_from_wire(ds4_dist_snapshot_req_fixed *s) {
    s->model_id = ntohl(s->model_id);
    s->session_hi = ntohl(s->session_hi);
    s->session_lo = ntohl(s->session_lo);
    s->request_hi = ntohl(s->request_hi);
    s->request_lo = ntohl(s->request_lo);
    s->token_hash_hi = ntohl(s->token_hash_hi);
    s->token_hash_lo = ntohl(s->token_hash_lo);
    s->token_count = ntohl(s->token_count);
    s->layer_start = ntohl(s->layer_start);
    s->layer_end = ntohl(s->layer_end);
}

static void dist_snapshot_begin_to_wire(ds4_dist_snapshot_begin_fixed *s) {
    s->model_id = htonl(s->model_id);
    s->session_hi = htonl(s->session_hi);
    s->session_lo = htonl(s->session_lo);
    s->request_hi = htonl(s->request_hi);
    s->request_lo = htonl(s->request_lo);
    s->token_hash_hi = htonl(s->token_hash_hi);
    s->token_hash_lo = htonl(s->token_hash_lo);
    s->token_count = htonl(s->token_count);
    s->layer_start = htonl(s->layer_start);
    s->layer_end = htonl(s->layer_end);
    s->payload_hi = htonl(s->payload_hi);
    s->payload_lo = htonl(s->payload_lo);
    s->status = htonl(s->status);
    s->token_bytes = htonl(s->token_bytes);
    s->message_bytes = htonl(s->message_bytes);
}

static void dist_snapshot_begin_from_wire(ds4_dist_snapshot_begin_fixed *s) {
    s->model_id = ntohl(s->model_id);
    s->session_hi = ntohl(s->session_hi);
    s->session_lo = ntohl(s->session_lo);
    s->request_hi = ntohl(s->request_hi);
    s->request_lo = ntohl(s->request_lo);
    s->token_hash_hi = ntohl(s->token_hash_hi);
    s->token_hash_lo = ntohl(s->token_hash_lo);
    s->token_count = ntohl(s->token_count);
    s->layer_start = ntohl(s->layer_start);
    s->layer_end = ntohl(s->layer_end);
    s->payload_hi = ntohl(s->payload_hi);
    s->payload_lo = ntohl(s->payload_lo);
    s->status = ntohl(s->status);
    s->token_bytes = ntohl(s->token_bytes);
    s->message_bytes = ntohl(s->message_bytes);
}

static void dist_snapshot_chunk_to_wire(ds4_dist_snapshot_chunk_fixed *s) {
    s->request_hi = htonl(s->request_hi);
    s->request_lo = htonl(s->request_lo);
    s->chunk_bytes = htonl(s->chunk_bytes);
}

static void dist_snapshot_chunk_from_wire(ds4_dist_snapshot_chunk_fixed *s) {
    s->request_hi = ntohl(s->request_hi);
    s->request_lo = ntohl(s->request_lo);
    s->chunk_bytes = ntohl(s->chunk_bytes);
}

static void dist_snapshot_done_to_wire(ds4_dist_snapshot_done_fixed *s) {
    s->request_hi = htonl(s->request_hi);
    s->request_lo = htonl(s->request_lo);
    s->status = htonl(s->status);
    s->message_bytes = htonl(s->message_bytes);
}

static void dist_snapshot_done_from_wire(ds4_dist_snapshot_done_fixed *s) {
    s->request_hi = ntohl(s->request_hi);
    s->request_lo = ntohl(s->request_lo);
    s->status = ntohl(s->status);
    s->message_bytes = ntohl(s->message_bytes);
}

static void dist_telemetry_to_wire(ds4_dist_telemetry_fixed *t) {
    t->layer_start = htonl(t->layer_start);
    t->layer_end = htonl(t->layer_end);
    t->route_index = htonl(t->route_index);
    t->pos0 = htonl(t->pos0);
    t->n_tokens = htonl(t->n_tokens);
    t->eval_usec = htonl(t->eval_usec);
    t->downstream_wait_usec = htonl(t->downstream_wait_usec);
    t->forward_send_usec = htonl(t->forward_send_usec);
    t->input_bytes = htonl(t->input_bytes);
    t->output_bytes = htonl(t->output_bytes);
}

static void dist_telemetry_from_wire(ds4_dist_telemetry_fixed *t) {
    t->layer_start = ntohl(t->layer_start);
    t->layer_end = ntohl(t->layer_end);
    t->route_index = ntohl(t->route_index);
    t->pos0 = ntohl(t->pos0);
    t->n_tokens = ntohl(t->n_tokens);
    t->eval_usec = ntohl(t->eval_usec);
    t->downstream_wait_usec = ntohl(t->downstream_wait_usec);
    t->forward_send_usec = ntohl(t->forward_send_usec);
    t->input_bytes = ntohl(t->input_bytes);
    t->output_bytes = ntohl(t->output_bytes);
}

static uint32_t dist_usec_since(double t0, double t1) {
    if (t1 <= t0) return 0;
    const double usec = (t1 - t0) * 1000000.0;
    if (usec >= (double)UINT32_MAX) return UINT32_MAX;
    return (uint32_t)(usec + 0.5);
}

/* =========================================================================
 * Worker Registration
 * ========================================================================= */

static int dist_send_hello(ds4_engine *engine, const ds4_dist_options *opt, int ctx_size, uint32_t listen_port, int fd) {
    uint32_t n_layers = (uint32_t)ds4_engine_layer_count(engine);
    const char *model_name = ds4_engine_model_name(engine);
    if (!model_name) model_name = "unknown";
    size_t model_name_len = strlen(model_name);
    if (model_name_len > DS4_DIST_MAX_MODEL_NAME) model_name_len = DS4_DIST_MAX_MODEL_NAME;

    ds4_dist_hello_fixed h = {
        (uint32_t)ds4_engine_model_id(engine),
        (uint32_t)ds4_engine_routed_quant_bits(engine),
        opt->layers.start,
        dist_resolved_layer_end(opt, n_layers),
        opt->layers.has_output ? 1u : 0u,
        1u,
        ctx_size > 0 ? (uint32_t)ctx_size : 0u,
        n_layers,
        listen_port,
        (uint32_t)model_name_len
    };
    ds4_dist_hello_fixed wire = h;
    dist_hello_to_wire(&wire);

    uint32_t bytes = (uint32_t)sizeof(wire) + (uint32_t)model_name_len;
    if (dist_write_frame_header(fd, DS4_DIST_MSG_HELLO, bytes) != 0) return -1;
    if (dist_write_full(fd, &wire, sizeof(wire)) != 0) return -1;
    if (model_name_len && dist_write_full(fd, model_name, model_name_len) != 0) return -1;
    return 0;
}

static int dist_recv_hello(int fd, ds4_dist_hello_fixed *hello, char *model_name, size_t model_name_cap, char *err, size_t errlen) {
    uint32_t type = 0, bytes = 0;
    int rc = dist_read_frame_header(fd, &type, &bytes, err, errlen);
    if (rc <= 0) return rc;
    if (type != DS4_DIST_MSG_HELLO) {
        if (errlen) snprintf(err, errlen, "expected HELLO frame, got type %u", type);
        dist_discard_bytes(fd, bytes);
        return -1;
    }
    if (bytes < sizeof(*hello) || bytes > sizeof(*hello) + DS4_DIST_MAX_MODEL_NAME) {
        if (errlen) snprintf(err, errlen, "invalid HELLO payload length %u", bytes);
        dist_discard_bytes(fd, bytes);
        return -1;
    }

    ds4_dist_hello_fixed wire;
    rc = dist_read_full(fd, &wire, sizeof(wire));
    if (rc <= 0) return rc == 0 ? 0 : -1;
    dist_hello_from_wire(&wire);

    uint32_t remaining = bytes - (uint32_t)sizeof(wire);
    if (wire.model_name_len != remaining || wire.model_name_len > DS4_DIST_MAX_MODEL_NAME) {
        if (errlen) snprintf(err, errlen, "invalid HELLO model name length %u", wire.model_name_len);
        dist_discard_bytes(fd, remaining);
        return -1;
    }

    if (model_name_cap) model_name[0] = '\0';
    if (wire.model_name_len) {
        char tmp[DS4_DIST_MAX_MODEL_NAME + 1u];
        rc = dist_read_full(fd, tmp, wire.model_name_len);
        if (rc <= 0) return rc == 0 ? 0 : -1;
        if (dist_bytes_have_nul(tmp, wire.model_name_len)) {
            if (errlen) snprintf(err, errlen, "HELLO model family contains NUL bytes");
            return -1;
        }
        tmp[wire.model_name_len] = '\0';
        if (model_name_cap) {
            snprintf(model_name, model_name_cap, "%s", tmp);
        }
    }

    *hello = wire;
    return 1;
}

static void dist_coordinator_report_plan(ds4_dist_coordinator_state *state);

static bool dist_coordinator_debug_enabled(const ds4_dist_coordinator_state *state) {
    return state && state->debug;
}

#define DIST_COORD_DEBUG(state, ...) do { \
    if (dist_coordinator_debug_enabled(state)) fprintf(stderr, __VA_ARGS__); \
} while (0)

