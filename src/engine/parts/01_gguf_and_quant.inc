/* =========================================================================
 * GGUF Parsing and Model Mapping.
 * =========================================================================
 *
 * The loader maps the model once, records metadata/tensor descriptors, and
 * leaves tensor bytes in place.  Inference code accesses weights by adding
 * tensor offsets to the mapping instead of copying the GGUF into private
 * structures.
 */

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

static const gguf_type_info gguf_types[] = {
    [0]  = {"f32",      1,   4},
    [1]  = {"f16",      1,   2},
    [2]  = {"q4_0",    32,  18},
    [3]  = {"q4_1",    32,  20},
    [6]  = {"q5_0",    32,  22},
    [7]  = {"q5_1",    32,  24},
    [8]  = {"q8_0",    32,  34},
    [9]  = {"q8_1",    32,  40},
    [10] = {"q2_k",   256,  84},
    [11] = {"q3_k",   256, 110},
    [12] = {"q4_k",   256, 144},
    [13] = {"q5_k",   256, 176},
    [14] = {"q6_k",   256, 210},
    [15] = {"q8_k",   256, 292},
    [16] = {"iq2_xxs",256,  66},
    [17] = {"iq2_xs", 256,  74},
    [18] = {"iq3_xxs",256,  98},
    [19] = {"iq1_s",  256, 110},
    [20] = {"iq4_nl", 256,  50},
    [21] = {"iq3_s",  256, 110},
    [22] = {"iq2_s",  256,  82},
    [23] = {"iq4_xs", 256, 136},
    [24] = {"i8",       1,   1},
    [25] = {"i16",      1,   2},
    [26] = {"i32",      1,   4},
    [27] = {"i64",      1,   8},
    [28] = {"f64",      1,   8},
    [29] = {"iq1_m",  256,  56},
    [30] = {"bf16",     1,   2},
    [39] = {"mxfp4",   32,  17},
};

enum {
    DS4_TENSOR_F32      = 0,
    DS4_TENSOR_F16      = 1,
    DS4_TENSOR_Q4_0     = 2,
    DS4_TENSOR_Q8_0     = 8,
    DS4_TENSOR_Q2_K     = 10,
    DS4_TENSOR_Q4_K     = 12,
    DS4_TENSOR_Q5_K     = 13,
    DS4_TENSOR_Q6_K     = 14,
    DS4_TENSOR_Q8_K     = 15,
    DS4_TENSOR_IQ2_XXS  = 16,
    DS4_TENSOR_I32      = 26,
    DS4_TENSOR_MXFP4    = 39,
};

typedef struct {
    ds4_str key;
    uint32_t type;
    uint64_t value_pos;
} ds4_kv;

typedef struct {
    ds4_str name;
    uint32_t ndim;
    uint64_t dim[DS4_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} ds4_tensor;

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;

    ds4_kv *kv;
    ds4_tensor *tensors;
} ds4_model;

static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(ds4_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return cursor_skip(c, scalar);

    if (type == GGUF_VALUE_STRING) {
        ds4_str ignored;
        return cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type;
        uint64_t len;

        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;

        uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }

        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}

static const gguf_type_info *tensor_type(uint32_t type) {
    uint32_t n = sizeof(gguf_types) / sizeof(gguf_types[0]);
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    return &gguf_types[type];
}

static const char *tensor_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type(type);
    return info ? info->name : "unknown";
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type(type);
    if (!info || info->block_elems == 0) return false;
    uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

static ds4_cursor cursor_at(const ds4_model *m, uint64_t pos) {
    ds4_cursor c = {
        .base = m->map,
        .size = m->size,
        .pos = pos,
        .error = {0},
    };
    return c;
}

static ds4_kv *model_find_kv(const ds4_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (ds4_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}

static bool model_get_string(const ds4_model *m, const char *key, ds4_str *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}

static bool model_get_u32(const ds4_model *m, const char *key, uint32_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

static bool model_get_token_id(const ds4_model *m, const char *key, int *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;

    ds4_cursor c = cursor_at(m, kv->value_pos);
    switch (kv->type) {
    case GGUF_VALUE_UINT32: {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v) || v > (uint32_t)INT_MAX) return false;
        *out = (int)v;
        return true;
    }
    case GGUF_VALUE_INT32: {
        int32_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v)) || v < 0) return false;
        *out = (int)v;
        return true;
    }
    case GGUF_VALUE_UINT64: {
        uint64_t v = 0;
        if (!cursor_u64(&c, &v) || v > (uint64_t)INT_MAX) return false;
        *out = (int)v;
        return true;
    }
    case GGUF_VALUE_INT64: {
        int64_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v)) || v < 0 || v > (int64_t)INT_MAX) return false;
        *out = (int)v;
        return true;
    }
    default:
        return false;
    }
}

static bool model_get_u64_compat(const ds4_model *m, const char *key, uint64_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_UINT64) {
        return cursor_u64(&c, out);
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = v;
        return true;
    }
    return false;
}

static bool model_get_f32_compat(const ds4_model *m, const char *key, float *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) {
        return cursor_read(&c, out, sizeof(*out));
    }
    if (kv->type == GGUF_VALUE_FLOAT64) {
        double v = 0.0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_INT32) {
        int32_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    return false;
}

static bool model_get_bool(const ds4_model *m, const char *key, bool *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_BOOL) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    uint8_t v = 0;
    if (!cursor_read(&c, &v, sizeof(v))) return false;
    *out = v != 0;
    return true;
}

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} ds4_array_ref;

static bool model_get_array(const ds4_model *m, const char *key, ds4_array_ref *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;

    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.pos;
    return true;
}

static void model_close(ds4_model *m) {
    if (!m) return;
    free(m->kv);
    free(m->tensors);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

static void model_prefetch_cpu_mapping(const ds4_model *m) {
    if (!m || !m->map || m->size == 0) return;

    /*
     * CPU generation touches expert weights according to router decisions, so a
     * long decode can fault in model pages that the prompt never touched. On
     * current Darwin kernels we have seen those late file-backed faults trigger
     * an OS-level VM panic in map-count accounting. This hint does not copy or
     * pin the GGUF; it just asks the kernel to start bringing the read-only
     * mapping into the page cache before token generation reaches it.
     */
#if defined(POSIX_MADV_WILLNEED)
    const int rc = posix_madvise((void *)m->map, (size_t)m->size, POSIX_MADV_WILLNEED);
    if (rc != 0) {
        ds4_log(stderr,
                DS4_LOG_WARNING,
                "ds4: warning: POSIX_MADV_WILLNEED failed for CPU model mapping: %s\n",
                strerror(rc));
    }
#else
    (void)m;
#endif
}

/* Read the GGUF metadata table.  Values stay in the mmap; we store offsets so
 * later validation can decode only the keys it needs. */
static void parse_metadata(ds4_model *m, ds4_cursor *c) {
    /* n_kv comes from the header. Every entry consumes at least one byte in the
     * file, so a count larger than the bytes remaining cannot be real; reject it
     * before calloc so a tiny file can't request an enormous allocation. */
    if (m->n_kv > c->size - c->pos) ds4_die("GGUF metadata count exceeds file size");
    m->kv = calloc((size_t)m->n_kv, sizeof(m->kv[0]));
    if (!m->kv) ds4_die("out of memory while allocating metadata table");

    m->alignment = 32;

    for (uint64_t i = 0; i < m->n_kv; i++) {
        ds4_kv *kv = &m->kv[i];

        if (!cursor_string(c, &kv->key)) ds4_die(c->error);
        if (!cursor_u32(c, &kv->type)) ds4_die(c->error);

        kv->value_pos = c->pos;

        if (ds4_streq(kv->key, "general.alignment") &&
            kv->type == GGUF_VALUE_UINT32)
        {
            ds4_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t alignment;
            if (cursor_u32(&tmp, &alignment) && alignment != 0) {
                m->alignment = alignment;
            }
        }

        if (!skip_value(c, kv->type, 0)) ds4_die(c->error);
    }
}

/* Read the tensor directory and convert relative GGUF offsets to absolute
 * mmap offsets.  Tensor bytes are still never copied here. */
static void parse_tensors(ds4_model *m, ds4_cursor *c) {
    /* As in parse_metadata: each tensor directory entry needs at least one byte
     * in the file, so reject a count larger than the bytes remaining before the
     * allocation. */
    if (m->n_tensors > c->size - c->pos) ds4_die("GGUF tensor count exceeds file size");
    m->tensors = calloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    if (!m->tensors) ds4_die("out of memory while allocating tensor table");

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];

        if (!cursor_string(c, &t->name)) ds4_die(c->error);
        if (!cursor_u32(c, &t->ndim)) ds4_die(c->error);
        if (t->ndim == 0 || t->ndim > DS4_MAX_DIMS) {
            ds4_die("tensor has an unsupported number of dimensions");
        }

        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) ds4_die(c->error);
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
                ds4_die("tensor element count overflow");
            }
            t->elements *= t->dim[d];
        }

        if (!cursor_u32(c, &t->type)) ds4_die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) ds4_die(c->error);

        if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
            ds4_log(stderr,
                DS4_LOG_WARNING,
                "ds4: warning: tensor %.*s has unsupported GGUF type %u\n",
                (int)t->name.len, t->name.ptr, t->type);
        }
    }

    m->tensor_data_pos = align_up(c->pos, m->alignment);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) {
            ds4_die("tensor offset overflow");
        }
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->bytes != 0 &&
            (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset))
        {
            ds4_die("tensor points outside GGUF file");
        }
        if (t->bytes > m->max_tensor_bytes) {
            m->max_tensor_bytes = t->bytes;
        }
    }
}

/* Open and map the GGUF once.  Metal needs a shared mapping for no-copy
 * MTLBuffers; CPU uses a private read-only mapping to avoid Darwin VM stress.
 * Tokenizer-only callers pass prefetch_cpu=false so inspecting tokens never
 * walks the huge tensor payload. */
static void model_open(ds4_model *m, const char *path, bool metal_mapping,
                       bool prefetch_cpu) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd == -1) ds4_die_errno("cannot open model", path);

    struct stat st;
    if (fstat(fd, &st) == -1) ds4_die_errno("cannot stat model", path);
    if (st.st_size < 32) ds4_die("model file is too small to be GGUF");

    /*
     * Metal wraps slices of this mapping as no-copy MTLBuffers, so the Metal
     * path keeps the file-backed shared mapping. The CPU path only reads the
     * weights through normal pointers and should not inherit Metal's VM policy:
     * use a private read-only mapping there.
     *
     * This is deliberately defensive against an OS-level Darwin VM bug observed
     * while the CPU backend streams the very large GGUF through a shared mmap:
     * the kernel can panic in VM map-count accounting instead of returning a
     * normal user-space failure. Keeping CPU inference off the shared mapping
     * avoids that VM accounting path while preserving normal file-backed reads.
     */
    const int mmap_flags = metal_mapping ? MAP_SHARED : MAP_PRIVATE;
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, mmap_flags, fd, 0);
    if (map == MAP_FAILED) ds4_die_errno("cannot mmap model", path);

    m->fd = fd;
    m->map = map;
    m->size = (uint64_t)st.st_size;

    ds4_cursor c = cursor_at(m, 0);
    uint32_t magic;
    if (!cursor_u32(&c, &magic)) ds4_die(c.error);
    if (magic != DS4_GGUF_MAGIC) ds4_die("model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_tensors)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_kv)) ds4_die(c.error);

    if (m->version != 3) ds4_die("only GGUF v3 is supported");

    parse_metadata(m, &c);
    parse_tensors(m, &c);

    if (!metal_mapping && prefetch_cpu) model_prefetch_cpu_mapping(m);
}

static void print_size(uint64_t bytes) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    printf("%.2f GiB", (double)bytes / gib);
}

#define DS4_DSPARK_MAX_TARGET_LAYERS 8
#define DS4_DSPARK_MAX_STAGES 8
#define DS4_DSPARK_MAX_BLOCK_SIZE 16
#define DS4_SPEC_PREFIX_SLOTS 4

typedef struct {
    uint32_t stages;
    uint32_t block_size;
    uint32_t markov_rank;
    uint32_t noise_token_id;
    uint32_t target_layer_count;
    uint32_t target_layers[DS4_DSPARK_MAX_TARGET_LAYERS];
    bool has_metadata;
    bool has_main_proj;
    bool has_main_norm;
    bool has_markov_head;
    bool has_confidence_head;
    bool has_final_head;
    bool has_block_size;
    bool has_markov_rank;
    bool has_noise_token_id;
    bool has_target_layers;
} ds4_dspark_summary;

typedef enum {
    DS4_SUPPORT_NONE = 0,
    DS4_SUPPORT_MTP_LEGACY,
    DS4_SUPPORT_DSPARK,
} ds4_support_kind;

static bool model_get_u32_any(const ds4_model *m, const char *const *keys,
                              size_t nkeys, uint32_t *out) {
    for (size_t i = 0; i < nkeys; i++) {
        if (model_get_u32(m, keys[i], out)) return true;
    }
    return false;
}

static bool model_get_u32_array_any(const ds4_model *m, const char *const *keys,
                                    size_t nkeys, uint32_t *out,
                                    uint32_t cap, uint32_t *n_out) {
    for (size_t ik = 0; ik < nkeys; ik++) {
        ds4_array_ref arr = {0};
        if (!model_get_array(m, keys[ik], &arr)) continue;
        if (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32) continue;

        ds4_cursor c = cursor_at(m, arr.data_pos);
        uint32_t n = arr.len < cap ? (uint32_t)arr.len : cap;
        for (uint32_t i = 0; i < n; i++) {
            if (arr.type == GGUF_VALUE_UINT32) {
                if (!cursor_u32(&c, &out[i])) return false;
            } else {
                int32_t v = 0;
                if (!cursor_read(&c, &v, sizeof(v))) return false;
                if (v < 0) return false;
                out[i] = (uint32_t)v;
            }
        }
        *n_out = n;
        return true;
    }
    return false;
}

static bool ds4_tensor_mtp_stage(ds4_str name, uint32_t *stage) {
    if (!ds4_str_starts_with(name, "mtp.")) return false;
    uint64_t pos = 4;
    if (pos >= name.len || !isdigit((unsigned char)name.ptr[pos])) return false;

    uint32_t value = 0;
    while (pos < name.len && isdigit((unsigned char)name.ptr[pos])) {
        uint32_t digit = (uint32_t)(name.ptr[pos] - '0');
        if (value > (UINT32_MAX - digit) / 10u) return false;
        value = value * 10u + digit;
        pos++;
    }
    if (pos >= name.len || name.ptr[pos] != '.') return false;
    *stage = value;
    return true;
}

static ds4_dspark_summary model_dspark_summary(const ds4_model *m) {
    static const char *const block_keys[] = {
        "deepseek4.dspark.block_size",
        "deepseek4.dspark_block_size",
        "dspark.block_size",
    };
    static const char *const markov_keys[] = {
        "deepseek4.dspark.markov_rank",
        "deepseek4.dspark_markov_rank",
        "dspark.markov_rank",
    };
    static const char *const noise_keys[] = {
        "deepseek4.dspark.noise_token_id",
        "deepseek4.dspark_noise_token_id",
        "dspark.noise_token_id",
    };
    static const char *const target_keys[] = {
        "deepseek4.dspark.target_layer_ids",
        "deepseek4.dspark_target_layer_ids",
        "dspark.target_layer_ids",
    };

    ds4_dspark_summary s = {0};
    if (model_get_u32_any(m, block_keys, sizeof(block_keys) / sizeof(block_keys[0]),
                          &s.block_size)) {
        s.has_metadata = true;
        s.has_block_size = true;
    }
    if (model_get_u32_any(m, markov_keys, sizeof(markov_keys) / sizeof(markov_keys[0]),
                          &s.markov_rank)) {
        s.has_metadata = true;
        s.has_markov_rank = true;
    }
    if (model_get_u32_any(m, noise_keys, sizeof(noise_keys) / sizeof(noise_keys[0]),
                          &s.noise_token_id)) {
        s.has_metadata = true;
        s.has_noise_token_id = true;
    }
    if (model_get_u32_array_any(m,
                                target_keys,
                                sizeof(target_keys) / sizeof(target_keys[0]),
                                s.target_layers,
                                DS4_DSPARK_MAX_TARGET_LAYERS,
                                &s.target_layer_count)) {
        s.has_metadata = true;
        s.has_target_layers = true;
    }

    uint32_t max_stage = 0;
    bool have_stage = false;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_str name = m->tensors[i].name;
        uint32_t stage = 0;
        if (!ds4_tensor_mtp_stage(name, &stage)) continue;
        if (!have_stage || stage > max_stage) max_stage = stage;
        have_stage = true;

        if (ds4_str_contains(name, ".main_proj.")) s.has_main_proj = true;
        if (ds4_str_contains(name, ".main_norm.")) s.has_main_norm = true;
        if (ds4_str_contains(name, ".markov_head.")) s.has_markov_head = true;
        if (ds4_str_contains(name, ".confidence_head.")) s.has_confidence_head = true;
        if (ds4_str_contains(name, ".hc_head_") ||
            ds4_str_contains(name, ".norm.weight")) {
            s.has_final_head = true;
        }
    }
    if (have_stage) s.stages = max_stage + 1u;
    return s;
}

static void model_print_dspark_summary(const ds4_model *m) {
    ds4_dspark_summary s = model_dspark_summary(m);
    if (!s.stages && !s.has_metadata) return;

    printf("mtp/dspark: stages=%u", s.stages);
    if (s.block_size) printf(" block=%u", s.block_size);
    if (s.markov_rank) printf(" markov_rank=%u", s.markov_rank);
    if (s.noise_token_id) printf(" noise_token=%u", s.noise_token_id);
    if (s.target_layer_count) {
        printf(" target_layers=");
        for (uint32_t i = 0; i < s.target_layer_count; i++) {
            printf("%s%u", i == 0 ? "" : ",", s.target_layers[i]);
        }
    }
    printf("\n");
    if (s.has_main_proj || s.has_main_norm || s.has_markov_head ||
        s.has_confidence_head || s.has_final_head) {
        printf("mtp/dspark tensors: main_proj=%s main_norm=%s markov=%s confidence=%s final_head=%s\n",
               s.has_main_proj ? "yes" : "no",
               s.has_main_norm ? "yes" : "no",
               s.has_markov_head ? "yes" : "no",
               s.has_confidence_head ? "yes" : "no",
               s.has_final_head ? "yes" : "no");
    }
}

static void model_summary(const ds4_model *m) {
    ds4_str name = {0};
    ds4_str arch = {0};
    uint32_t layers = 0;
    uint64_t ctx_train = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_swa = 0;
    uint32_t indexer_heads = 0;
    uint32_t indexer_head_dim = 0;
    uint32_t indexer_top_k = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    uint64_t tensor_bytes = 0;
    uint64_t params = 0;

    model_get_string(m, "general.name", &name);
    model_get_string(m, "general.architecture", &arch);
    if (!model_get_u32(m, "deepseek4.block_count", &layers)) {
        model_get_u32(m, "glm-dsa.block_count", &layers);
    }
    if (!model_get_u64_compat(m, "deepseek4.context_length", &ctx_train)) {
        model_get_u64_compat(m, "glm-dsa.context_length", &ctx_train);
    }
    if (!model_get_u32(m, "deepseek4.attention.head_count", &n_head)) {
        model_get_u32(m, "glm-dsa.attention.head_count", &n_head);
    }
    if (!model_get_u32(m, "deepseek4.attention.head_count_kv", &n_head_kv)) {
        model_get_u32(m, "glm-dsa.attention.head_count_kv", &n_head_kv);
    }
    if (!model_get_u32(m, "deepseek4.attention.key_length", &head_dim)) {
        model_get_u32(m, "glm-dsa.attention.key_length", &head_dim);
    }
    model_get_u32(m, "deepseek4.attention.sliding_window", &n_swa);
    if (!model_get_u32(m, "deepseek4.attention.indexer.head_count", &indexer_heads)) {
        model_get_u32(m, "glm-dsa.attention.indexer.head_count", &indexer_heads);
    }
    if (!model_get_u32(m, "deepseek4.attention.indexer.key_length", &indexer_head_dim)) {
        model_get_u32(m, "glm-dsa.attention.indexer.key_length", &indexer_head_dim);
    }
    if (!model_get_u32(m, "deepseek4.attention.indexer.top_k", &indexer_top_k)) {
        model_get_u32(m, "glm-dsa.attention.indexer.top_k", &indexer_top_k);
    }
    if (!model_get_u32(m, "deepseek4.expert_count", &n_expert)) {
        model_get_u32(m, "glm-dsa.expert_count", &n_expert);
    }
    if (!model_get_u32(m, "deepseek4.expert_used_count", &n_expert_used)) {
        model_get_u32(m, "glm-dsa.expert_used_count", &n_expert_used);
    }
    if (!model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups)) {
        model_get_u32(m, "glm-dsa.expert_group_count", &n_expert_groups);
    }
    if (!model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used)) {
        model_get_u32(m, "glm-dsa.expert_group_used_count", &n_group_used);
    }

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        tensor_bytes += m->tensors[i].bytes;
        params += m->tensors[i].elements;
    }

    printf("model: %.*s\n", (int)name.len, name.ptr);
    printf("arch:  %.*s\n", (int)arch.len, arch.ptr);
    printf("gguf:  v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n",
        m->version, m->n_kv, m->n_tensors);
    if (layers) printf("layers: %u\n", layers);
    if (ctx_train) printf("train context: %" PRIu64 "\n", ctx_train);
    if (n_head || n_head_kv || head_dim || n_swa) {
        printf("attention: heads=%u kv_heads=%u head_dim=%u swa=%u\n",
               n_head, n_head_kv, head_dim, n_swa);
    }
    if (indexer_heads || indexer_head_dim || indexer_top_k) {
        printf("indexer: heads=%u head_dim=%u top_k=%u\n",
               indexer_heads, indexer_head_dim, indexer_top_k);
    }
    if (n_expert || n_expert_used || n_expert_groups || n_group_used) {
        printf("experts: count=%u used=%u groups=%u groups_used=%u\n",
               n_expert, n_expert_used, n_expert_groups, n_group_used);
    }
    model_print_dspark_summary(m);
    printf("file size: ");
    print_size(m->size);
    printf("\n");
    printf("tensor bytes described by GGUF: ");
    print_size(tensor_bytes);
    printf("\n");
    printf("logical parameters: %.2f B\n", (double)params / 1000000000.0);

    printf("tensor types:\n");
    for (uint32_t type = 0; type < sizeof(gguf_types)/sizeof(gguf_types[0]); type++) {
        uint64_t count = 0;
        uint64_t bytes = 0;
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            if (m->tensors[i].type == type) {
                count++;
                bytes += m->tensors[i].bytes;
            }
        }
        if (count != 0) {
            printf("  %-8s %5" PRIu64 " tensors, ", tensor_type_name(type), count);
            print_size(bytes);
            printf("\n");
        }
    }

}

static ds4_tensor *model_find_tensor(const ds4_model *m, const char *name) {
    const size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == len &&
            memcmp(m->tensors[i].name.ptr, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

static const char *support_kind_name(ds4_support_kind kind) {
    switch (kind) {
    case DS4_SUPPORT_MTP_LEGACY: return "legacy MTP";
    case DS4_SUPPORT_DSPARK:     return "DSpark";
    case DS4_SUPPORT_NONE:       return "none";
    }
    return "unknown";
}

static ds4_support_kind support_model_detect(
        const ds4_model *m,
        uint32_t        *stages_out,
        ds4_dspark_summary *summary_out) {
    if (stages_out) *stages_out = 0;
    if (summary_out) memset(summary_out, 0, sizeof(*summary_out));
    if (!m) return DS4_SUPPORT_NONE;

    ds4_dspark_summary s = model_dspark_summary(m);
    if (summary_out) *summary_out = s;
    if (stages_out) *stages_out = s.stages;

    if (s.stages >= 3 &&
        s.has_main_proj &&
        s.has_markov_head &&
        s.has_confidence_head) {
        return DS4_SUPPORT_DSPARK;
    }

    if (model_find_tensor(m, "mtp.0.e_proj.weight") &&
        model_find_tensor(m, "mtp.0.h_proj.weight") &&
        model_find_tensor(m, "mtp.0.hc_head_base.weight")) {
        if (stages_out) *stages_out = s.stages ? s.stages : 1u;
        return DS4_SUPPORT_MTP_LEGACY;
    }

    return DS4_SUPPORT_NONE;
}

#ifndef DS4_NO_GPU
#ifndef __APPLE__
typedef struct {
    uint64_t off;
    uint64_t end;
} accelerator_tensor_span;

static int accelerator_tensor_span_cmp(const void *a, const void *b) {
    const accelerator_tensor_span *sa = a;
    const accelerator_tensor_span *sb = b;
    if (sa->off < sb->off) return -1;
    if (sa->off > sb->off) return 1;
    if (sa->end < sb->end) return -1;
    if (sa->end > sb->end) return 1;
    return 0;
}

static uint64_t accelerator_cuda_preload_span_bytes(void) {
    uint64_t mb = 1024;
#ifndef DS4_ROCM_BUILD
    const char *env = getenv("DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
#endif
    if (mb < 64) mb = 64;
    if (mb > 4096) mb = 4096;
    return mb * 1048576ull;
}

static bool accelerator_span_filter_contains(uint64_t off,
                                             uint64_t bytes,
                                             const uint64_t *span_offsets,
                                             const uint64_t *span_sizes,
                                             uint32_t span_count) {
    if (span_count == 0) return true;
    if (bytes == 0) return true;
    const uint64_t end = off + bytes;
    if (end < off) return false;
    for (uint32_t i = 0; i < span_count; i++) {
        const uint64_t span_end = span_offsets[i] + span_sizes[i];
        if (span_end < span_offsets[i]) return false;
        if (off >= span_offsets[i] && end <= span_end) return true;
    }
    return false;
}

static bool accelerator_prepare_model_tensor_spans(const ds4_model *m,
                                                   const uint64_t *span_offsets,
                                                   const uint64_t *span_sizes,
                                                   uint32_t span_count,
                                                   uint64_t *prepared_out) {
    uint64_t cap = m->n_tensors;
    if (cap == 0) {
        if (prepared_out) *prepared_out = 0;
        return true;
    }

    accelerator_tensor_span *spans = xmalloc((size_t)cap * sizeof(spans[0]));
    uint64_t nspan = 0;
    for (uint32_t i = 0; i < span_count; i++) {
        if (span_offsets[i] > m->size ||
            span_sizes[i] == 0 ||
            span_sizes[i] > m->size - span_offsets[i]) {
            free(spans);
            return false;
        }
    }
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        if (t->bytes == 0) continue;
        if (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset) {
            free(spans);
            return false;
        }
        if (!accelerator_span_filter_contains(t->abs_offset, t->bytes,
                                              span_offsets, span_sizes, span_count)) {
            continue;
        }
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
        if (ds4_gpu_model_range_replaced(m->map, t->abs_offset, t->bytes)) {
            continue;
        }
#endif
        spans[nspan++] = (accelerator_tensor_span){
            .off = t->abs_offset,
            .end = t->abs_offset + t->bytes,
        };
    }
    if (nspan == 0) {
        free(spans);
        if (prepared_out) *prepared_out = 0;
        return true;
    }

    qsort(spans, (size_t)nspan, sizeof(spans[0]), accelerator_tensor_span_cmp);

    const uint64_t max_span = accelerator_cuda_preload_span_bytes();
    const int tty = ds4_log_is_tty(stderr);
    const uint64_t progress_step = (tty ? 2ull : 16ull) * 1073741824ull;
    uint64_t next_progress = progress_step;
    double last_progress = now_sec();
    uint64_t prepared = 0;
    uint64_t merged = 0;

#ifdef DS4_ROCM_BUILD
    const char *accelerator_name = "ROCm";
#else
    const char *accelerator_name = "CUDA";
#endif

    fprintf(stderr, "%sds4: %s preparing model tensor mappings%s",
            tty ? "\r\033[K" : "",
            accelerator_name,
            tty ? ": 0.00 GiB" : "\n");
    fflush(stderr);

    for (uint64_t i = 0; i < nspan;) {
        uint64_t off = spans[i].off;
        uint64_t end = spans[i].end;
        i++;
        while (i < nspan &&
               spans[i].off <= end + 65536u &&
               spans[i].end - off <= max_span) {
            if (spans[i].end > end) end = spans[i].end;
            i++;
        }
        char label[96];
        snprintf(label, sizeof(label), "tensor-span:%" PRIu64, merged);
        if (ds4_gpu_cache_model_range(m->map, m->size, off, end - off, label) == 0) {
            if (tty) fputc('\n', stderr);
            fprintf(stderr,
                    "ds4: accelerator failed to prepare model tensor span %" PRIu64
                    " at offset %" PRIu64 "\n",
                    merged, off);
            free(spans);
            return false;
        }
        prepared += end - off;
        merged++;

        const double now = now_sec();
        if (prepared >= next_progress || now - last_progress >= (tty ? 2.0 : 10.0)) {
            if (tty) {
                fprintf(stderr, "\r\033[Kds4: %s preparing model tensor mappings: %.2f GiB",
                        accelerator_name,
                        (double)prepared / 1073741824.0);
            } else {
                fprintf(stderr, "ds4: %s prepared model tensor mappings %.2f GiB\n",
                        accelerator_name,
                        (double)prepared / 1073741824.0);
            }
            fflush(stderr);
            last_progress = now;
            while (next_progress <= prepared) next_progress += progress_step;
        }
    }

    if (tty) fputc('\n', stderr);
    free(spans);
    if (prepared_out) *prepared_out = prepared;
    return true;
}

static bool accelerator_cache_q8_tensors(const ds4_model *m,
                                         const uint64_t *span_offsets,
                                         const uint64_t *span_sizes,
                                         uint32_t span_count) {
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        if (t->bytes == 0) continue;
        if (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset) return false;
        if (!accelerator_span_filter_contains(t->abs_offset, t->bytes,
                                              span_offsets, span_sizes, span_count)) {
            continue;
        }
        char label[128];
        snprintf(label, sizeof(label), "tensor:%.*s", (int)t->name.len, t->name.ptr);
        if (t->type == DS4_TENSOR_Q8_0 && t->ndim == 2 &&
            ds4_gpu_cache_q8_f16_range(m->map, m->size, t->abs_offset, t->bytes, t->dim[0], t->dim[1], label) == 0) {
            fprintf(stderr, "ds4: accelerator failed to cache dequantized Q8 tensor %.*s\n",
                    (int)t->name.len, t->name.ptr);
            return false;
        }
    }
    return true;
}

static bool accelerator_cache_model_tensors(ds4_backend backend,
                                            const ds4_model *m,
                                            const uint64_t *span_offsets,
                                            const uint64_t *span_sizes,
                                            uint32_t span_count) {
    if (backend != DS4_BACKEND_CUDA) return true;
    if (!m || !m->map || m->size == 0) return false;
#ifndef DS4_ROCM_BUILD
    if (getenv("DS4_CUDA_DIRECT_MODEL") != NULL) {
        return true;
    }
#endif

    const double t0 = now_sec();
    uint64_t prepared = 0;
    if (!accelerator_prepare_model_tensor_spans(m, span_offsets, span_sizes, span_count, &prepared)) {
        return false;
    }
    if (!accelerator_cache_q8_tensors(m, span_offsets, span_sizes, span_count)) return false;
    const double t1 = now_sec();
#ifdef DS4_ROCM_BUILD
    const char *accelerator_name = "ROCm";
#else
    const char *accelerator_name = "CUDA";
#endif
    fprintf(stderr,
            "ds4: %s startup model preparation covered %.2f GiB of tensor spans in %.3fs\n",
            accelerator_name, (double)prepared / 1073741824.0, t1 - t0);
    return true;
}
#else
static bool accelerator_cache_model_tensors(ds4_backend backend,
                                            const ds4_model *m,
                                            const uint64_t *span_offsets,
                                            const uint64_t *span_sizes,
                                            uint32_t span_count) {
    (void)backend;
    (void)m;
    (void)span_offsets;
    (void)span_sizes;
    (void)span_count;
    return true;
}
#endif
#endif

/* Return the in-place tensor payload inside the mapped GGUF. */
static const void *tensor_data(const ds4_model *m, const ds4_tensor *t) {
    return m->map + t->abs_offset;
}

/* Optional startup pass that touches tensor pages before timing generation. */
static void model_warm_weights(const ds4_model *m) {
    const uint64_t start = m->tensor_data_pos;
    const uint64_t end = m->size;
    if (start >= end) return;

    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint8_t *p = m->map;
    volatile uint64_t checksum = 0;
    const double t0 = now_sec();

    fprintf(stderr, "ds4: warming mapped tensor pages: %.2f GiB\n",
            (double)(end - start) / (1024.0 * 1024.0 * 1024.0));

#if defined(POSIX_MADV_WILLNEED)
    (void)posix_madvise((void *)(p + start), (size_t)(end - start), POSIX_MADV_WILLNEED);
#endif

    for (uint64_t off = start; off < end; off += page) {
        checksum += p[off];
    }
    checksum += p[end - 1];

    const double t1 = now_sec();
    fprintf(stderr, "ds4: warmed tensor pages in %.3fs (checksum=%llu)\n",
            t1 - t0, (unsigned long long)checksum);
}

/* =========================================================================
 * Scalar Conversion and Quantized Tensor Kernels.
 * =========================================================================
 *
 * These functions are the CPU reference math used by the C backend and by
 * Metal diagnostics.  They implement only the tensor formats present in the
 * DeepSeek V4 Flash GGUF: F16, F32, Q8_0, Q2_K, IQ2_XXS, and Q8_K activation
 * blocks used for expert dot products.
 */

static inline float f16_to_f32(uint16_t h) {
#if defined(__ARM_NEON)
    const float16x4_t hv = vreinterpret_f16_u16(vdup_n_u16(h));
    return vgetq_lane_f32(vcvt_f32_f16(hv), 0);
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ff;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

static inline uint16_t f32_to_f16(float f) {
#if defined(__ARM_NEON)
    const float32x4_t fv = vdupq_n_f32(f);
    const float16x4_t hv = vcvt_f16_f32(fv);
    return vget_lane_u16(vreinterpret_u16_f16(hv), 0);
#else
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
#endif
}

static void f16_round_inplace_cpu(float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) x[i] = f16_to_f32(f32_to_f16(x[i]));
}

static float dsv4_e4m3fn_value_cpu(int i) {
    static const float exp_scale[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f,
        0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f,
        32.0f, 64.0f, 128.0f, 256.0f,
    };

    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}

static float dsv4_e4m3fn_dequant_cpu(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);

    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_cpu(mid) <= ax) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - dsv4_e4m3fn_value_cpu(best));
        const float next_diff = fabsf(ax - dsv4_e4m3fn_value_cpu(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) {
            best++;
        }
    }

    return sign * dsv4_e4m3fn_value_cpu(best);
}

/* DeepSeek V4 stores the non-RoPE part of compressed KV through an E4M3-style
 * round trip.  Keeping this in the CPU reference makes cache values comparable
 * to the Metal graph's compressed-cache behavior. */
static void dsv4_fp8_kv_quantize_row_inplace_cpu(float *x, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 64; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }

        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < 64; i++) {
            float v = x[off + i] / scale;
            if (v > 448.0f) v = 448.0f;
            if (v < -448.0f) v = -448.0f;
            x[off + i] = dsv4_e4m3fn_dequant_cpu(v) * scale;
        }
    }
}

static float dsv4_e2m1fn_value_cpu(int i) {
    static const float values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    return values[i & 7];
}

static float dsv4_e2m1fn_dequant_cpu(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_cpu(0));
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - dsv4_e2m1fn_value_cpu(i));
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_value_cpu(best);
}

static void dsv4_hadamard128_inplace_cpu(float *x) {
    for (uint32_t stride = 1; stride < 128; stride <<= 1) {
        for (uint32_t base = 0; base < 128; base += 2u * stride) {
            for (uint32_t i = 0; i < stride; i++) {
                const float a = x[base + i];
                const float b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
        }
    }
    const float scale = 0.08838834764831845f;
    for (uint32_t i = 0; i < 128; i++) x[i] *= scale;
}

static void dsv4_fp4_act_quantize_row_inplace_cpu(float *x, uint32_t n) {
    if ((n % 32u) != 0) ds4_die("DSV4 FP4 activation quantization requires 32-aligned rows");
    for (uint32_t off = 0; off < n; off += 32) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 32; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }

        if (amax < 7.052966104933725e-38f) amax = 7.052966104933725e-38f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 6.0f)));
        for (uint32_t i = 0; i < 32; i++) {
            float v = x[off + i] / scale;
            if (v > 6.0f) v = 6.0f;
            if (v < -6.0f) v = -6.0f;
            x[off + i] = dsv4_e2m1fn_dequant_cpu(v) * scale;
        }
    }
}

/* The official DeepSeek V4 graph rotates indexer activations with a 128-wide
 * Hadamard transform and immediately runs the FP4 activation-simulation
 * round trip. This applies to both indexer Q and the indexer compressor KV;
 * without it, the top-k compressed-row selection is not the model's graph. */
static void dsv4_indexer_qat_row_inplace_cpu(float *x, uint32_t head_dim) {
    if (head_dim != 128) ds4_die("DSV4 indexer QAT expects 128-wide indexer rows");
    dsv4_hadamard128_inplace_cpu(x);
    dsv4_fp4_act_quantize_row_inplace_cpu(x, head_dim);
}

static void dsv4_indexer_qat_rows_inplace_cpu(float *x, uint32_t rows, uint32_t head_dim) {
    for (uint32_t r = 0; r < rows; r++) {
        dsv4_indexer_qat_row_inplace_cpu(x + (uint64_t)r * head_dim, head_dim);
    }
}

/* Quantize a float activation into Q8_K blocks so GGUF Q2_K/IQ2_XXS expert
 * kernels can reuse the same activation for many expert rows. */
static void ds4_quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k) {
    if (k % QK_K != 0) ds4_die("Q8_K quantization length is not QK_K aligned");
    const int64_t nb = k / QK_K;

    for (int64_t b = 0; b < nb; b++) {
        float max = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            const float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }

        if (amax == 0.0f) {
            y[b].d = 0.0f;
            memset(y[b].qs, 0, sizeof(y[b].qs));
            memset(y[b].bsums, 0, sizeof(y[b].bsums));
            x += QK_K;
            continue;
        }

        const float iscale = -127.0f / max;
        for (int j = 0; j < QK_K; j++) {
            int v = (int)lrintf(iscale * x[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (int j = 0; j < QK_K / 16; j++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / iscale;
        x += QK_K;
    }
}

static void ds4_vec_dot_q2_K_q8_K(int n, float *s, const block_q2_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m3 = vdupq_n_u8(0x03);
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    const int32x4_t zero = vdupq_n_s32(0);
    float sum = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * f16_to_f32(x[i].d);
        const float dmin = -y[i].d * f16_to_f32(x[i].dmin);

        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        const uint8x16_t mins_and_scales = vld1q_u8(sc);
        const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
        uint8_t scale_lanes[16];
        vst1q_u8(scale_lanes, scales);

        const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
        const int16x8x2_t q8sums = vld1q_s16_x2(y[i].bsums);
        const int16x8x2_t mins16 = {{
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))),
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins))),
        }};
        const int32x4_t s0 = vaddq_s32(
            vmull_s16(vget_low_s16(mins16.val[0]), vget_low_s16(q8sums.val[0])),
            vmull_s16(vget_high_s16(mins16.val[0]), vget_high_s16(q8sums.val[0])));
        const int32x4_t s1 = vaddq_s32(
            vmull_s16(vget_low_s16(mins16.val[1]), vget_low_s16(q8sums.val[1])),
            vmull_s16(vget_high_s16(mins16.val[1]), vget_high_s16(q8sums.val[1])));
        sum += dmin * (float)vaddvq_s32(vaddq_s32(s0, s1));

        int isum = 0;
        int is = 0;
        for (int j = 0; j < QK_K / 128; j++) {
            const uint8x16x2_t q2bits = vld1q_u8_x2(q2);
            q2 += 32;

#define DS4_Q2_DOT_NOSHIFT(scale_index) do {                                           \
                const int8x16x2_t q8bytes = vld1q_s8_x2(q8);                           \
                q8 += 32;                                                              \
                const int8x16_t q2lo = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));\
                const int8x16_t q2hi = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));\
                isum += vaddvq_s32(vdotq_s32(zero, q2lo, q8bytes.val[0])) *            \
                        scale_lanes[is + (scale_index)];                               \
                isum += vaddvq_s32(vdotq_s32(zero, q2hi, q8bytes.val[1])) *            \
                        scale_lanes[is + 1 + (scale_index)];                           \
            } while (0)

#define DS4_Q2_DOT_SHIFT(shift, scale_index) do {                                      \
                const int8x16x2_t q8bytes = vld1q_s8_x2(q8);                           \
                q8 += 32;                                                              \
                const int8x16_t q2lo = vreinterpretq_s8_u8(                            \
                    vandq_u8(vshrq_n_u8(q2bits.val[0], (shift)), m3));                 \
                const int8x16_t q2hi = vreinterpretq_s8_u8(                            \
                    vandq_u8(vshrq_n_u8(q2bits.val[1], (shift)), m3));                 \
                isum += vaddvq_s32(vdotq_s32(zero, q2lo, q8bytes.val[0])) *            \
                        scale_lanes[is + (scale_index)];                               \
                isum += vaddvq_s32(vdotq_s32(zero, q2hi, q8bytes.val[1])) *            \
                        scale_lanes[is + 1 + (scale_index)];                           \
            } while (0)

            DS4_Q2_DOT_NOSHIFT(0);
            DS4_Q2_DOT_SHIFT(2, 2);
            DS4_Q2_DOT_SHIFT(4, 4);
            DS4_Q2_DOT_SHIFT(6, 6);
            is += 8;

#undef DS4_Q2_DOT_NOSHIFT
#undef DS4_Q2_DOT_SHIFT
        }

        sum += d * (float)isum;
    }

    *s = sum;
#else
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; j++) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * f16_to_f32(x[i].d);
        const float dmin = y[i].d * f16_to_f32(x[i].dmin);

        int isum = 0;
        int is = 0;
        for (int k = 0; k < QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                int isuml = dot_q2_16(q2, q8, shift);
                isum += d * isuml;

                d = sc[is++] & 0x0f;
                isuml = dot_q2_16(q2 + 16, q8 + 16, shift);
                isum += d * isuml;

                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    *s = sumf;
#endif
}

static inline float q2_k_value_f32(const block_q2_K *blocks, uint32_t k) {
    const uint32_t block = k / QK_K;
    const uint32_t idx = k - block * QK_K;
    const block_q2_K *xb = blocks + block;
    const uint32_t group = idx / 16u;
    const uint32_t l = idx - group * 16u;
    const uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
    const uint32_t shift = ((group / 2u) & 3u) * 2u;
    const uint32_t q = ((uint32_t)xb->qs[q_base + l] >> shift) & 0x03u;
    const uint32_t sc = xb->scales[group];
    return f16_to_f32(xb->d) * (float)(sc & 0x0fu) * (float)q -
           f16_to_f32(xb->dmin) * (float)(sc >> 4u);
}

static float ds4_vec_dot_q2_K_f32(int n, const block_q2_K *x, const float *y) {
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        sum += q2_k_value_f32(x, (uint32_t)k) * y[k];
    }
    return sum;
}

static inline void q4_k_get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

static void ds4_vec_dot_q4_K_q8_K(int n, float *s, const block_q4_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int32x4_t zero = vdupq_n_s32(0);
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d  = y[i].d * f16_to_f32(x[i].d);
        const float dm = -y[i].d * f16_to_f32(x[i].dmin);

        const uint8_t *qs = x[i].qs;
        const uint8_t *sc = x[i].scales;
        const int8_t  *q8 = y[i].qs;

        int32_t summs = 0;
        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(j, sc, &sc_val, &m_val);
            int32_t gsum = (int32_t)y[i].bsums[j * 2] + (int32_t)y[i].bsums[j * 2 + 1];
            summs += m_val * gsum;
        }

        int isum = 0;
        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(j, sc, &sc_val, &m_val);

            const int byte_off = (j >> 1) * 32;
            const int shift = (j & 1) * 4;

            /* Load 32 q8 values for this group */
            const int8x16x2_t q8v = vld1q_s8_x2(q8 + j * 32);

            /* Unpack 32 q4 values from 32 bytes at qs[byte_off] with shift */
            uint8_t q4_u[32];
            if (shift == 0) {
                for (int l = 0; l < 32; l++) q4_u[l] = qs[byte_off + l] & 0xF;
            } else {
                for (int l = 0; l < 32; l++) q4_u[l] = qs[byte_off + l] >> 4;
            }

            const int8x16_t q4a = vreinterpretq_s8_u8(vld1q_u8(q4_u));
            const int8x16_t q4b = vreinterpretq_s8_u8(vld1q_u8(q4_u + 16));

            isum += vaddvq_s32(vdotq_s32(zero, q4a, q8v.val[0])) * sc_val;
            isum += vaddvq_s32(vdotq_s32(zero, q4b, q8v.val[1])) * sc_val;
        }

        sumf += d * (float)isum + dm * (float)summs;
    }

    *s = sumf;
#else
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d  = y[i].d * f16_to_f32(x[i].d);
        const float dm = -y[i].d * f16_to_f32(x[i].dmin);

        const uint8_t *qs = x[i].qs;
        const uint8_t *sc = x[i].scales;
        const int8_t  *q8 = y[i].qs;

        int summs = 0;
        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(j, sc, &sc_val, &m_val);
            int32_t gsum = (int32_t)y[i].bsums[j * 2] + (int32_t)y[i].bsums[j * 2 + 1];
            summs += m_val * gsum;
        }

        int isum = 0;
        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(j, sc, &sc_val, &m_val);

            const int byte_off = (j >> 1) * 32;
            const int shift = (j & 1) * 4;

            for (int l = 0; l < 32; l++) {
                isum += ((qs[byte_off + l] >> shift) & 0xF) * (int)q8[j * 32 + l] * sc_val;
            }
        }

        sumf += d * (float)isum + dm * (float)summs;
    }

    *s = sumf;
#endif
}

static void ds4_vec_dot_q5_K_q8_K(int n, float *s, const block_q5_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * f16_to_f32(x[i].d);
        const float dmin = y[i].d * f16_to_f32(x[i].dmin);
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const int8_t *q8 = y[i].qs;
        const uint8_t *scales = x[i].scales;

        int64_t isum = 0;
        int64_t summs = 0;
        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(is, scales, &sc_val, &m_val);
            summs += (int64_t)m_val * ((int32_t)y[i].bsums[2 * is] + (int32_t)y[i].bsums[2 * is + 1]);
            for (int l = 0; l < 32; l++) {
                const int q = (int)(ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                isum += (int64_t)sc_val * q * (int)q8[j + l];
            }

            q4_k_get_scale_min(is + 1, scales, &sc_val, &m_val);
            summs += (int64_t)m_val * ((int32_t)y[i].bsums[2 * (is + 1)] + (int32_t)y[i].bsums[2 * (is + 1) + 1]);
            for (int l = 0; l < 32; l++) {
                const int q = (int)(ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                isum += (int64_t)sc_val * q * (int)q8[j + 32 + l];
            }

            ql += 32;
            is += 2;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }

        sumf += d * (float)isum - dmin * (float)summs;
    }

    *s = sumf;
}

static void ds4_vec_dot_q6_K_q8_K(int n, float *s, const block_q6_K *x, const block_q8_K *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = y[i].d * f16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *scales = x[i].scales;
        const int8_t *q8 = y[i].qs;
        int64_t isum = 0;

        for (int n128 = 0; n128 < QK_K; n128 += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = ((int)(ql[l + 0]  & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((int)(ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((int)(ql[l + 0]  >> 4)    | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((int)(ql[l + 32] >> 4)    | (((qh[l] >> 6) & 3) << 4)) - 32;

                isum += (int64_t)scales[is + 0] * q1 * (int)q8[n128 + l + 0];
                isum += (int64_t)scales[is + 2] * q2 * (int)q8[n128 + l + 32];
                isum += (int64_t)scales[is + 4] * q3 * (int)q8[n128 + l + 64];
                isum += (int64_t)scales[is + 6] * q4 * (int)q8[n128 + l + 96];
            }

            ql += 64;
            qh += 32;
            scales += 8;
        }

        sumf += d * (float)isum;
    }

    *s = sumf;
}

static float ds4_vec_dot_q4_K_f32(int n, const block_q4_K *x, const float *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const float dmin = f16_to_f32(x[i].dmin);
        const uint8_t *qs = x[i].qs;
        const uint8_t *scales = x[i].scales;
        const float *yb = y + (uint64_t)i * QK_K;

        for (int j = 0; j < QK_K / 32; j++) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(j, scales, &sc_val, &m_val);

            const int byte_off = (j >> 1) * 32;
            const int shift = (j & 1) * 4;
            const float scale = d * (float)sc_val;
            const float minv = dmin * (float)m_val;
            for (int l = 0; l < 32; l++) {
                const int q = (qs[byte_off + l] >> shift) & 0x0F;
                sumf += (scale * (float)q - minv) * yb[j * 32 + l];
            }
        }
    }

    return sumf;
}

static inline float ds4_e8m0_to_f32(uint8_t e) {
    const uint32_t bits = e == 0 ? 0x00400000u : (uint32_t)e << 23;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static const float ds4_mxfp4_values[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static float ds4_vec_dot_mxfp4_f32(int n, const block_mxfp4 *x, const float *y) {
    const int nb = n / QK_MXFP4;
    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ib++) {
        const float d = ds4_e8m0_to_f32(x[ib].e);
        const float *yb = y + (uint64_t)ib * QK_MXFP4;
        for (int j = 0; j < QK_MXFP4 / 2; j++) {
            const uint8_t q = x[ib].qs[j];
            sumf += d * ds4_mxfp4_values[q & 0x0fu] * yb[j];
            sumf += d * ds4_mxfp4_values[q >> 4] * yb[j + QK_MXFP4 / 2];
        }
    }

    return sumf;
}

static float ds4_vec_dot_q5_K_f32(int n, const block_q5_K *x, const float *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const float dmin = f16_to_f32(x[i].dmin);
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *scales = x[i].scales;
        const float *yb = y + (uint64_t)i * QK_K;

        for (int group = 0; group < QK_K / 32; group++) {
            uint8_t sc_val, m_val;
            q4_k_get_scale_min(group, scales, &sc_val, &m_val);

            const int ql_base = (group >> 1) * 32;
            const int shift = (group & 1) * 4;
            const uint8_t hmask = (uint8_t)(1u << group);
            const float scale = d * (float)sc_val;
            const float minv = dmin * (float)m_val;
            for (int l = 0; l < 32; l++) {
                const int q = ((ql[ql_base + l] >> shift) & 0x0F) +
                              ((qh[l] & hmask) ? 16 : 0);
                sumf += (scale * (float)q - minv) * yb[group * 32 + l];
            }
        }
    }

    return sumf;
}

static float ds4_vec_dot_q6_K_f32(int n, const block_q6_K *x, const float *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *scales = x[i].scales;
        const float *yb = y + (uint64_t)i * QK_K;

        for (int n128 = 0; n128 < QK_K; n128 += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = ((int)(ql[l + 0]  & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((int)(ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((int)(ql[l + 0]  >> 4)    | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((int)(ql[l + 32] >> 4)    | (((qh[l] >> 6) & 3) << 4)) - 32;

                sumf += d * (float)scales[is + 0] * (float)q1 * yb[n128 + l + 0];
                sumf += d * (float)scales[is + 2] * (float)q2 * yb[n128 + l + 32];
                sumf += d * (float)scales[is + 4] * (float)q3 * yb[n128 + l + 64];
                sumf += d * (float)scales[is + 6] * (float)q4 * yb[n128 + l + 96];
            }

            ql += 64;
            qh += 32;
            scales += 8;
        }
    }

    return sumf;
}

static inline float ds4_vec_dot_q5_q6_K_f32(uint32_t type, int n, const uint8_t *x, const float *y) {
    if (type == DS4_TENSOR_Q5_K) {
        return ds4_vec_dot_q5_K_f32(n, (const block_q5_K *)x, y);
    } else if (type == DS4_TENSOR_Q6_K) {
        return ds4_vec_dot_q6_K_f32(n, (const block_q6_K *)x, y);
    } else {
        ds4_die("expected a Q5_K or Q6_K tensor");
    }
    return 0.0f;
}

static float ds4_vec_dot_iq2_xxs_f32(int n, const block_iq2_xxs *x, const float *y) {
    pthread_once(&iq2xxs_signed_grid_once, iq2xxs_signed_grid_init);

    const int nb = n / QK_K;
    float sumf = 0.0f;
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const uint16_t *q2 = x[i].qs;
        const float *yb = y + (uint64_t)i * QK_K;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;

            const float scale = 0.125f * d * (float)(2u * (aux32[1] >> 28) + 1u);
            const uint32_t base = (uint32_t)ib32 * 32u;
            for (int l = 0; l < 4; l++) {
                const uint32_t sign_idx = (aux32[1] >> (7 * l)) & 127u;
                const int8_t *grid = iq2xxs_signed_grid[aux8[l]][sign_idx];
                const float *yf = yb + base + (uint32_t)l * 8u;
                for (int j = 0; j < 8; j++) {
                    sumf += scale * (float)grid[j] * yf[j];
                }
            }
        }
    }

    return sumf;
}

static void ds4_vec_dot_q8_K_q8_K(int n, float *s,
                                  const block_q8_K *x,
                                  const block_q8_K *y) {
    const int nb = n / QK_K;
    float sum = 0.0f;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    for (int i = 0; i < nb; i++) {
        int32x4_t isum = vdupq_n_s32(0);
        for (int j = 0; j < QK_K; j += 16) {
            isum = vdotq_s32(isum, vld1q_s8(x[i].qs + j),
                             vld1q_s8(y[i].qs + j));
        }
        sum += x[i].d * y[i].d * (float)vaddvq_s32(isum);
    }
#else
    for (int i = 0; i < nb; i++) {
        int isum = 0;
        for (int j = 0; j < QK_K; j++) {
            isum += (int)x[i].qs[j] * (int)y[i].qs[j];
        }
        sum += x[i].d * y[i].d * (float)isum;
    }
#endif
    *s = sum;
}

static void ds4_vec_dot_q8_K_pair_q8_K(
        int n, float *s0, float *s1,
        const block_q8_K *x0, const block_q8_K *x1,
        const block_q8_K *y) {
    const int nb = n / QK_K;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    for (int i = 0; i < nb; i++) {
        int32x4_t isum0 = vdupq_n_s32(0);
        int32x4_t isum1 = vdupq_n_s32(0);
        for (int j = 0; j < QK_K; j += 16) {
            const int8x16_t yv = vld1q_s8(y[i].qs + j);
            isum0 = vdotq_s32(isum0, vld1q_s8(x0[i].qs + j), yv);
            isum1 = vdotq_s32(isum1, vld1q_s8(x1[i].qs + j), yv);
        }
        sum0 += x0[i].d * y[i].d * (float)vaddvq_s32(isum0);
        sum1 += x1[i].d * y[i].d * (float)vaddvq_s32(isum1);
    }
#else
    for (int i = 0; i < nb; i++) {
        int isum0 = 0;
        int isum1 = 0;
        for (int j = 0; j < QK_K; j++) {
            const int yv = (int)y[i].qs[j];
            isum0 += (int)x0[i].qs[j] * yv;
            isum1 += (int)x1[i].qs[j] * yv;
        }
        sum0 += x0[i].d * y[i].d * (float)isum0;
        sum1 += x1[i].d * y[i].d * (float)isum1;
    }
#endif
    *s0 = sum0;
    *s1 = sum1;
}

static DS4_MAYBE_UNUSED void ds4_vec_dot_iq2_xxs_q8_K(int n, float *s, const block_iq2_xxs *x, const block_q8_K *y) {
    const int nb = n / QK_K;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        float sumf1 = 0.0f;
        float sumf2 = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;

            uint32_t aux32[4];
            memcpy(aux32, q2, sizeof(aux32));
            q2 += 8;
            const uint8_t *aux8 = (const uint8_t *)aux32;

            int8x16_t q2u0 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[0])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[1])));
            int8x16_t q2u1 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[2])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[3])));
            int8x16_t q2u2 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[8])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[9])));
            int8x16_t q2u3 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + aux8[10])),
                                          vld1_s8((const int8_t *)(iq2xxs_grid + aux8[11])));

            const int8x16_t q2s0 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[1] >>  0) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[1] >>  7) & 127]));
            const int8x16_t q2s1 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[1] >> 14) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[1] >> 21) & 127]));
            const int8x16_t q2s2 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[3] >>  0) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[3] >>  7) & 127]));
            const int8x16_t q2s3 = vcombine_s8(vld1_s8(iq2xxs_signs[(aux32[3] >> 14) & 127]),
                                               vld1_s8(iq2xxs_signs[(aux32[3] >> 21) & 127]));

            q2u0 = vmulq_s8(q2u0, q2s0);
            q2u1 = vmulq_s8(q2u1, q2s1);
            q2u2 = vmulq_s8(q2u2, q2s2);
            q2u3 = vmulq_s8(q2u3, q2s3);

            const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q2u0, q8b.val[0]), q2u1, q8b.val[1]);
            const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q2u2, q8b.val[2]), q2u3, q8b.val[3]);

            sumf1 += (float)vaddvq_s32(p1) * (0.5f + (float)(aux32[1] >> 28));
            sumf2 += (float)vaddvq_s32(p2) * (0.5f + (float)(aux32[3] >> 28));
        }

        sumf += d * (sumf1 + sumf2);
    }

    *s = 0.25f * sumf;
#else
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        int32_t bsum = 0;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;

            const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; l += 2) {
                const uint32_t sign_idx0 = (aux32[1] >> (7 * l)) & 127;
                const uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127;
                sumi += dot_iq2_pair_16(iq2xxs_signed_grid[aux8[l]][sign_idx0],
                                        iq2xxs_signed_grid[aux8[l + 1]][sign_idx1],
                                        q8);
                q8 += 16;
            }
            bsum += sumi * (int32_t)ls;
        }
        sumf += d * (float)bsum;
    }
    *s = 0.125f * sumf;
#endif
}

static void ds4_vec_dot_iq2_xxs_pair_q8_K(
        int n,
        float *s0,
        float *s1,
        const block_iq2_xxs *x0,
        const block_iq2_xxs *x1,
        const block_q8_K *y) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int nb = n / QK_K;
    float total0 = 0.0f;
    float total1 = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d0 = f16_to_f32(x0[i].d) * y[i].d;
        const float d1 = f16_to_f32(x1[i].d) * y[i].d;
        const uint16_t *q20 = x0[i].qs;
        const uint16_t *q21 = x1[i].qs;
        const int8_t *q8 = y[i].qs;
        float sum01 = 0.0f;
        float sum02 = 0.0f;
        float sum11 = 0.0f;
        float sum12 = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;

            uint32_t aux0[4];
            uint32_t aux1[4];
            memcpy(aux0, q20, sizeof(aux0));
            memcpy(aux1, q21, sizeof(aux1));
            q20 += 8;
            q21 += 8;
            const uint8_t *a0 = (const uint8_t *)aux0;
            const uint8_t *a1 = (const uint8_t *)aux1;

#define DS4_IQ2_PAIR_DOT(aux, aux8, accum_a, accum_b) do {                                              \
                int8x16_t u0 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[0])),          \
                                           vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[1])));          \
                int8x16_t u1 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[2])),          \
                                           vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[3])));          \
                int8x16_t u2 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[8])),          \
                                           vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[9])));          \
                int8x16_t u3 = vcombine_s8(vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[10])),         \
                                           vld1_s8((const int8_t *)(iq2xxs_grid + (aux8)[11])));         \
                const int8x16_t sgn0 = vcombine_s8(vld1_s8(iq2xxs_signs[((aux)[1] >>  0) & 127]),       \
                                                   vld1_s8(iq2xxs_signs[((aux)[1] >>  7) & 127]));      \
                const int8x16_t sgn1 = vcombine_s8(vld1_s8(iq2xxs_signs[((aux)[1] >> 14) & 127]),       \
                                                   vld1_s8(iq2xxs_signs[((aux)[1] >> 21) & 127]));      \
                const int8x16_t sgn2 = vcombine_s8(vld1_s8(iq2xxs_signs[((aux)[3] >>  0) & 127]),       \
                                                   vld1_s8(iq2xxs_signs[((aux)[3] >>  7) & 127]));      \
                const int8x16_t sgn3 = vcombine_s8(vld1_s8(iq2xxs_signs[((aux)[3] >> 14) & 127]),       \
                                                   vld1_s8(iq2xxs_signs[((aux)[3] >> 21) & 127]));      \
                u0 = vmulq_s8(u0, sgn0);                                                               \
                u1 = vmulq_s8(u1, sgn1);                                                               \
                u2 = vmulq_s8(u2, sgn2);                                                               \
                u3 = vmulq_s8(u3, sgn3);                                                               \
                const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), u0, q8b.val[0]), u1, q8b.val[1]); \
                const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), u2, q8b.val[2]), u3, q8b.val[3]); \
                (accum_a) += (float)vaddvq_s32(p1) * (0.5f + (float)((aux)[1] >> 28));                  \
                (accum_b) += (float)vaddvq_s32(p2) * (0.5f + (float)((aux)[3] >> 28));                  \
            } while (0)

            DS4_IQ2_PAIR_DOT(aux0, a0, sum01, sum02);
            DS4_IQ2_PAIR_DOT(aux1, a1, sum11, sum12);

#undef DS4_IQ2_PAIR_DOT
        }

        total0 += d0 * (sum01 + sum02);
        total1 += d1 * (sum11 + sum12);
    }

    *s0 = 0.25f * total0;
    *s1 = 0.25f * total1;
#else
    ds4_vec_dot_iq2_xxs_q8_K(n, s0, x0, y);
    ds4_vec_dot_iq2_xxs_q8_K(n, s1, x1, y);
#endif
}

typedef struct {
    ds4_tensor *hc_attn_fn;
    ds4_tensor *hc_attn_scale;
    ds4_tensor *hc_attn_base;
    ds4_tensor *attn_norm;
    ds4_tensor *attn_q_a;
    ds4_tensor *attn_q_a_norm;
    ds4_tensor *attn_q_b;
    ds4_tensor *attn_kv;
    ds4_tensor *attn_kv_a_norm;
    ds4_tensor *attn_sinks;
    ds4_tensor *attn_output_a;
    ds4_tensor *attn_output_b;
    ds4_tensor *attn_compressor_ape;
    ds4_tensor *attn_compressor_kv;
    ds4_tensor *attn_compressor_gate;
    ds4_tensor *attn_compressor_norm;
    ds4_tensor *indexer_attn_q_b;
    ds4_tensor *indexer_proj;
    ds4_tensor *indexer_compressor_ape;
    ds4_tensor *indexer_compressor_kv;
    ds4_tensor *indexer_compressor_gate;
    ds4_tensor *indexer_compressor_norm;
    ds4_tensor *hc_ffn_fn;
    ds4_tensor *hc_ffn_scale;
    ds4_tensor *hc_ffn_base;
    ds4_tensor *ffn_norm;
    ds4_tensor *ffn_gate_tid2eid;
    ds4_tensor *ffn_gate_inp;
    ds4_tensor *ffn_exp_probs_b;
    ds4_tensor *ffn_gate_exps;
    ds4_tensor *ffn_up_exps;
    ds4_tensor *ffn_down_exps;
    ds4_tensor *ffn_gate_shexp;
    ds4_tensor *ffn_up_shexp;
    ds4_tensor *ffn_down_shexp;
} ds4_layer_weights;

typedef struct {
    ds4_tensor *token_embd;
    ds4_tensor *output_hc_base;
    ds4_tensor *output_hc_fn;
    ds4_tensor *output_hc_scale;
    ds4_tensor *output_norm;
    ds4_tensor *output;
    ds4_layer_weights layer[DS4_MAX_LAYER];
} ds4_weights;

typedef struct {
    ds4_tensor *main_proj;
    ds4_tensor *main_norm;
    ds4_tensor *norm;
    ds4_tensor *hc_head_base;
    ds4_tensor *hc_head_fn;
    ds4_tensor *hc_head_scale;
    ds4_tensor *markov_w1;
    ds4_tensor *markov_w2;
    ds4_tensor *confidence_proj;
    ds4_layer_weights block;
} ds4_dspark_stage_weights;

typedef struct {
    uint32_t n_stages;
    uint32_t block_size;
    uint32_t markov_rank;
    uint32_t noise_token_id;
    uint32_t target_layer_count;
    uint32_t target_layers[DS4_DSPARK_MAX_TARGET_LAYERS];
    uint32_t present_tensors;
    uint32_t missing_tensors;
    uint32_t invalid_tensors;
    uint32_t metadata_errors;
    bool has_block_size;
    bool has_markov_rank;
    bool has_noise_token_id;
    bool has_target_layers;
    ds4_dspark_stage_weights stage[DS4_DSPARK_MAX_STAGES];
} ds4_dspark_weights;
