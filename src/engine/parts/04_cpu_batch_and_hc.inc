/* =========================================================================
 * Q4_K routed expert matrix-vector products.
 * ========================================================================= */

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[DS4_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q4_k_mid_ctx;

static void matvec_q4_k_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q4_k_mid_ctx *ctx = vctx;

    for (uint64_t idx = row0; idx < row1; idx++) {
        const int slot = (int)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        float gate = 0.0f;
        float up = 0.0f;

        const block_q4_K *gate_row = (const block_q4_K *)(ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot]);
        ds4_vec_dot_q4_K_q8_K((int)ctx->in_dim, &gate, gate_row, ctx->xq);

        const block_q4_K *up_row = (const block_q4_K *)(ctx->up_base[slot] + row * ctx->up_row_bytes[slot]);
        ds4_vec_dot_q4_K_q8_K((int)ctx->in_dim, &up, up_row, ctx->xq);

        if (ctx->clamp > 1.0e-6f) {
            if (gate > ctx->clamp) gate = ctx->clamp;
            if (up > ctx->clamp) up = ctx->clamp;
            if (up < -ctx->clamp) up = -ctx->clamp;
        }
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}

static void matvec_q4_k_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type != DS4_TENSOR_Q4_K || up_w->type != DS4_TENSOR_Q4_K)
        ds4_die("expected Q4_K expert tensors");
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_q4_k_mid_ctx ctx = {
        .mid = mid,
        .xq = xq,
        .clamp = clamp,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
                                               &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
        ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
                                             &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
        if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
            ds4_die("paired Q4_K expert tensors do not match");
        }
        if (i == 0) {
            in_dim0 = gate_in_dim;
            out_dim0 = gate_out_dim;
        } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
            ds4_die("Q4_K expert tensors do not share a layout");
        }
        ctx.expert_weight[i] = expert_weight[i];
    }
    if (in_dim0 % QK_K != 0) ds4_die("Q4_K expert row is not QK_K aligned");

    ctx.in_dim = in_dim0;
    ctx.out_dim = out_dim0;
    ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_q4_k_mid_worker, &ctx);
}

typedef struct {
    float *out;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];
    int n_expert;
} matvec_q4_k_accum_ctx;

static void matvec_q4_k_accum_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q4_k_accum_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        float acc = 0.0f;
        for (int i = 0; i < ctx->n_expert; i++) {
            float v = 0.0f;
            const block_q4_K *br = (const block_q4_K *)(ctx->base[i] + row * ctx->row_bytes[i]);
            ds4_vec_dot_q4_K_q8_K((int)ctx->in_dim, &v, br, ctx->xq[i]);
            acc += v;
        }
        ctx->out[row] = acc;
    }
}

static void matvec_q4_k_experts_accum_prequant(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert) {
    if (w->type != DS4_TENSOR_Q4_K) ds4_die("expected a Q4_K expert tensor");
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];

    for (int i = 0; i < n_expert; i++) {
        uint64_t in_dim, out_dim;
        base[i] = tensor_expert_bytes(m, w, (uint32_t)selected[i], &in_dim, &out_dim, &row_bytes[i]);
        if (i == 0) {
            in_dim0 = in_dim;
            out_dim0 = out_dim;
        } else if (in_dim != in_dim0 || out_dim != out_dim0) {
            ds4_die("Q4_K expert tensors do not share a layout");
        }
    }
    if (in_dim0 % QK_K != 0) ds4_die("Q4_K expert row is not QK_K aligned");

    const uint64_t n_blocks = in_dim0 / QK_K;
    matvec_q4_k_accum_ctx ctx = {
        .out = out,
        .in_dim = in_dim0,
        .n_expert = n_expert,
    };
    for (int i = 0; i < n_expert; i++) {
        ctx.base[i] = base[i];
        ctx.row_bytes[i] = row_bytes[i];
        ctx.xq[i] = xq + (uint64_t)i * n_blocks;
    }

    ds4_parallel_for(out_dim0, matvec_q4_k_accum_worker, &ctx);
}

static inline void ds4_vec_dot_q5_q6_K_q8_K(
        uint32_t          type,
        int               n,
        float            *s,
        const uint8_t    *x,
        const block_q8_K *y) {
    if (type == DS4_TENSOR_Q5_K) {
        ds4_vec_dot_q5_K_q8_K(n, s, (const block_q5_K *)x, y);
    } else if (type == DS4_TENSOR_Q6_K) {
        ds4_vec_dot_q6_K_q8_K(n, s, (const block_q6_K *)x, y);
    } else {
        ds4_die("expected a Q5_K or Q6_K tensor");
    }
}

typedef struct {
    float *out;
    const uint8_t *base;
    const block_q8_K *xq;
    uint64_t in_dim;
    uint64_t row_bytes;
    uint32_t type;
} matvec_q5_q6_k_ctx;

static void matvec_q5_q6_k_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q5_q6_k_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        ds4_vec_dot_q5_q6_K_q8_K(ctx->type, (int)ctx->in_dim, &ctx->out[row],
                                 ctx->base + row * ctx->row_bytes, ctx->xq);
    }
}

typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT_USED];
    const uint8_t *up_base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq;
    float expert_weight[DS4_MAX_EXPERT_USED];
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT_USED];
    uint64_t up_row_bytes[DS4_MAX_EXPERT_USED];
    uint32_t gate_type;
    uint32_t up_type;
    int n_expert;
} matvec_q5_q6_k_mid_ctx;

static void matvec_q5_q6_k_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q5_q6_k_mid_ctx *ctx = vctx;

    for (uint64_t idx = row0; idx < row1; idx++) {
        const int slot = (int)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
        float gate = 0.0f;
        float up = 0.0f;

        ds4_vec_dot_q5_q6_K_q8_K(ctx->gate_type, (int)ctx->in_dim, &gate,
                                 ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot],
                                 ctx->xq);
        ds4_vec_dot_q5_q6_K_q8_K(ctx->up_type, (int)ctx->in_dim, &up,
                                 ctx->up_base[slot] + row * ctx->up_row_bytes[slot],
                                 ctx->xq);

        if (ctx->clamp > 1.0e-6f) {
            if (gate > ctx->clamp) gate = ctx->clamp;
            if (up > ctx->clamp) up = ctx->clamp;
            if (up < -ctx->clamp) up = -ctx->clamp;
        }
        ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
    }
}

static void matvec_q5_q6_k_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if ((gate_w->type != DS4_TENSOR_Q5_K && gate_w->type != DS4_TENSOR_Q6_K) ||
        (up_w->type != DS4_TENSOR_Q5_K && up_w->type != DS4_TENSOR_Q6_K)) {
        ds4_die("expected Q5_K/Q6_K expert tensors");
    }
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    matvec_q5_q6_k_mid_ctx ctx = {
        .mid = mid,
        .xq = xq,
        .clamp = clamp,
        .gate_type = gate_w->type,
        .up_type = up_w->type,
        .n_expert = n_expert,
    };

    for (int i = 0; i < n_expert; i++) {
        uint64_t gate_in_dim, gate_out_dim;
        uint64_t up_in_dim, up_out_dim;
        ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
                                               &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
        ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
                                             &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
        if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
            ds4_die("paired Q5_K/Q6_K expert tensors do not match");
        }
        if (i == 0) {
            in_dim0 = gate_in_dim;
            out_dim0 = gate_out_dim;
        } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
            ds4_die("Q5_K/Q6_K expert tensors do not share a layout");
        }
        ctx.expert_weight[i] = expert_weight[i];
    }
    if (in_dim0 % QK_K != 0) ds4_die("Q5_K/Q6_K expert row is not QK_K aligned");

    ctx.in_dim = in_dim0;
    ctx.out_dim = out_dim0;
    ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_q5_q6_k_mid_worker, &ctx);
}

static void matvec_q5_q6_k_expert(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const float      *x,
        uint32_t          expert) {
    if (w->type != DS4_TENSOR_Q5_K && w->type != DS4_TENSOR_Q6_K) {
        ds4_die("expected a Q5_K or Q6_K expert tensor");
    }

    uint64_t in_dim, out_dim, row_bytes;
    const uint8_t *base = tensor_expert_bytes(m, w, expert, &in_dim, &out_dim, &row_bytes);
    if (in_dim % QK_K != 0) ds4_die("Q5_K/Q6_K expert row is not QK_K aligned");

    block_q8_K *xq = xmalloc((size_t)(in_dim / QK_K) * sizeof(xq[0]));
    ds4_quantize_row_q8_K(x, xq, (int64_t)in_dim);

    matvec_q5_q6_k_ctx ctx = {
        .out = out,
        .base = base,
        .xq = xq,
        .in_dim = in_dim,
        .row_bytes = row_bytes,
        .type = w->type,
    };
    ds4_parallel_for(out_dim, matvec_q5_q6_k_worker, &ctx);

    free(xq);
}

typedef struct {
    float *out;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    const block_q8_K *xq[DS4_MAX_EXPERT_USED];
    uint64_t in_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];
    uint32_t type;
    int n_expert;
} matvec_q5_q6_k_accum_ctx;

static void matvec_q5_q6_k_accum_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q5_q6_k_accum_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        float acc = 0.0f;
        for (int i = 0; i < ctx->n_expert; i++) {
            float v = 0.0f;
            ds4_vec_dot_q5_q6_K_q8_K(ctx->type, (int)ctx->in_dim, &v,
                                     ctx->base[i] + row * ctx->row_bytes[i],
                                     ctx->xq[i]);
            acc += v;
        }
        ctx->out[row] = acc;
    }
}

static void matvec_q5_q6_k_experts_accum_prequant(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert) {
    if (w->type != DS4_TENSOR_Q5_K && w->type != DS4_TENSOR_Q6_K) {
        ds4_die("expected a Q5_K or Q6_K expert tensor");
    }
    if (n_expert < 1 || (uint32_t)n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");

    uint64_t in_dim0 = 0;
    uint64_t out_dim0 = 0;
    const uint8_t *base[DS4_MAX_EXPERT_USED];
    uint64_t row_bytes[DS4_MAX_EXPERT_USED];

    for (int i = 0; i < n_expert; i++) {
        uint64_t in_dim, out_dim;
        base[i] = tensor_expert_bytes(m, w, (uint32_t)selected[i], &in_dim, &out_dim, &row_bytes[i]);
        if (i == 0) {
            in_dim0 = in_dim;
            out_dim0 = out_dim;
        } else if (in_dim != in_dim0 || out_dim != out_dim0) {
            ds4_die("Q5_K/Q6_K expert tensors do not share a layout");
        }
    }
    if (in_dim0 % QK_K != 0) ds4_die("Q5_K/Q6_K expert row is not QK_K aligned");

    const uint64_t n_blocks = in_dim0 / QK_K;
    matvec_q5_q6_k_accum_ctx ctx = {
        .out = out,
        .in_dim = in_dim0,
        .type = w->type,
        .n_expert = n_expert,
    };
    for (int i = 0; i < n_expert; i++) {
        ctx.base[i] = base[i];
        ctx.row_bytes[i] = row_bytes[i];
        ctx.xq[i] = xq + (uint64_t)i * n_blocks;
    }

    ds4_parallel_for(out_dim0, matvec_q5_q6_k_accum_worker, &ctx);
}

/* Q4_K batch mid worker: same structure as IQ2_XXS batch but uses Q4_K dot. */
typedef struct {
    float *mid;
    const uint8_t *gate_base[DS4_MAX_EXPERT];
    const uint8_t *up_base[DS4_MAX_EXPERT];
    const block_q8_K *xq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    const float *pair_weight;
    float clamp;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t gate_row_bytes[DS4_MAX_EXPERT];
    uint64_t up_row_bytes[DS4_MAX_EXPERT];
    uint64_t xq_blocks;
} matvec_q4_k_batch_mid_ctx;

static void matvec_q4_k_batch_mid_worker(void *vctx, uint64_t task0, uint64_t task1) {
    matvec_q4_k_batch_mid_ctx *ctx = vctx;

    for (uint64_t task = task0; task < task1; task++) {
        const uint32_t active_idx = (uint32_t)(task / ctx->out_dim);
        const uint64_t row = task - (uint64_t)active_idx * ctx->out_dim;
        const uint32_t expert = ctx->active_expert[active_idx];
        const uint32_t begin = ctx->expert_offset[expert];
        const uint32_t end = ctx->expert_offset[expert + 1];

        const block_q4_K *gate_row = (const block_q4_K *)(ctx->gate_base[expert] + row * ctx->gate_row_bytes[expert]);
        const block_q4_K *up_row = (const block_q4_K *)(ctx->up_base[expert] + row * ctx->up_row_bytes[expert]);

        for (uint32_t i = begin; i < end; i++) {
            const uint32_t pair_id = ctx->pair_ids[i];
            const ds4_expert_pair pair = ctx->pairs[pair_id];
            const block_q8_K *xq = ctx->xq + (uint64_t)pair.token * ctx->xq_blocks;
            float gate = 0.0f;
            float up = 0.0f;

            ds4_vec_dot_q4_K_q8_K((int)ctx->in_dim, &gate, gate_row, xq);
            ds4_vec_dot_q4_K_q8_K((int)ctx->in_dim, &up, up_row, xq);

            if (ctx->clamp > 1.0e-6f) {
                if (gate > ctx->clamp) gate = ctx->clamp;
                if (up > ctx->clamp) up = ctx->clamp;
                if (up < -ctx->clamp) up = -ctx->clamp;
            }

            ctx->mid[(uint64_t)pair_id * ctx->out_dim + row] = silu(gate) * up * ctx->pair_weight[pair_id];
        }
    }
}

/* Q4_K batch down accum worker: same structure as Q2_K batch but uses Q4_K dot. */
typedef struct {
    float *moe;
    const uint8_t *base[DS4_MAX_EXPERT];
    const block_q8_K *midq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_q4_k_batch_accum_rows_ctx;

static void matvec_q4_k_batch_accum_rows_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q4_k_batch_accum_rows_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            ctx->moe[(uint64_t)t * ctx->out_dim + row] = 0.0f;
        }

        for (uint32_t ai = 0; ai < ctx->n_active; ai++) {
            const uint32_t expert = ctx->active_expert[ai];
            const uint32_t begin = ctx->expert_offset[expert];
            const uint32_t end = ctx->expert_offset[expert + 1];
            const block_q4_K *br = (const block_q4_K *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

            for (uint32_t i = begin; i < end; i++) {
                const uint32_t pair_id = ctx->pair_ids[i];
                const ds4_expert_pair pair = ctx->pairs[pair_id];
                const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
                float v = 0.0f;

                ds4_vec_dot_q4_K_q8_K((int)ctx->in_dim, &v, br, xq);
                ctx->moe[(uint64_t)pair.token * ctx->out_dim + row] += v;
            }
        }
    }
}

typedef struct {
    float *moe;
    const uint8_t *base[DS4_MAX_EXPERT];
    const block_q8_K *midq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_iq2_xxs_batch_accum_rows_ctx;

static void matvec_iq2_xxs_batch_accum_rows_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_iq2_xxs_batch_accum_rows_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            ctx->moe[(uint64_t)t * ctx->out_dim + row] = 0.0f;
        }

        for (uint32_t ai = 0; ai < ctx->n_active; ai++) {
            const uint32_t expert = ctx->active_expert[ai];
            const uint32_t begin = ctx->expert_offset[expert];
            const uint32_t end = ctx->expert_offset[expert + 1];
            const block_iq2_xxs *br = (const block_iq2_xxs *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

            for (uint32_t i = begin; i < end; i++) {
                const uint32_t pair_id = ctx->pair_ids[i];
                const ds4_expert_pair pair = ctx->pairs[pair_id];
                const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
                float v = 0.0f;

                ds4_vec_dot_iq2_xxs_q8_K((int)ctx->in_dim, &v, br, xq);
                ctx->moe[(uint64_t)pair.token * ctx->out_dim + row] += v;
            }
        }
    }
}

/* Dispatch: call the right gate/up mid builder based on tensor type. */
static void matvec_experts_mid_prequant(
        float            *mid,
        const ds4_model  *m,
        const ds4_tensor *gate_w,
        const ds4_tensor *up_w,
        const block_q8_K *xq,
        const int        *selected,
        const float      *expert_weight,
        int               n_expert,
        float             clamp) {
    if (gate_w->type == DS4_TENSOR_IQ2_XXS) {
        matvec_iq2_xxs_experts_mid_prequant(mid, m, gate_w, up_w, xq,
                                            selected, expert_weight, n_expert, clamp);
    } else if (gate_w->type == DS4_TENSOR_Q2_K) {
        matvec_q2_k_experts_mid_prequant(mid, m, gate_w, up_w, xq,
                                         selected, expert_weight, n_expert, clamp);
    } else if (gate_w->type == DS4_TENSOR_Q4_K) {
        matvec_q4_k_experts_mid_prequant(mid, m, gate_w, up_w, xq,
                                         selected, expert_weight, n_expert, clamp);
    } else if (gate_w->type == DS4_TENSOR_Q5_K || gate_w->type == DS4_TENSOR_Q6_K) {
        matvec_q5_q6_k_experts_mid_prequant(mid, m, gate_w, up_w, xq,
                                            selected, expert_weight, n_expert, clamp);
    } else {
        ds4_die("unsupported gate/up expert tensor type");
    }
}

/* Dispatch: call the right down-projection accumulator based on tensor type. */
static void matvec_experts_down_accum_prequant(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const block_q8_K *xq,
        const int        *selected,
        int               n_expert) {
    if (w->type == DS4_TENSOR_IQ2_XXS) {
        matvec_iq2_xxs_experts_accum_prequant(out, m, w, xq, selected, n_expert);
    } else if (w->type == DS4_TENSOR_Q2_K) {
        matvec_q2_k_experts_accum_prequant(out, m, w, xq, selected, n_expert);
    } else if (w->type == DS4_TENSOR_Q4_K) {
        matvec_q4_k_experts_accum_prequant(out, m, w, xq, selected, n_expert);
    } else if (w->type == DS4_TENSOR_Q5_K || w->type == DS4_TENSOR_Q6_K) {
        matvec_q5_q6_k_experts_accum_prequant(out, m, w, xq, selected, n_expert);
    } else {
        ds4_die("unsupported down expert tensor type");
    }
}

/* Dispatch: single-expert gate/up pair for tracing. */
static void matvec_expert_pair_prequant(
        float            *out0,
        float            *out1,
        const ds4_model  *m,
        const ds4_tensor *w0,
        const ds4_tensor *w1,
        const block_q8_K *xq,
        uint32_t          expert) {
    if (w0->type == DS4_TENSOR_IQ2_XXS) {
        matvec_iq2_xxs_expert_pair_prequant(out0, out1, m, w0, w1, xq, expert);
    } else if (w0->type == DS4_TENSOR_Q2_K) {
        uint64_t in_dim0, out_dim0, rb0;
        uint64_t in_dim1, out_dim1, rb1;
        const uint8_t *base0 = tensor_expert_bytes(m, w0, expert, &in_dim0, &out_dim0, &rb0);
        const uint8_t *base1 = tensor_expert_bytes(m, w1, expert, &in_dim1, &out_dim1, &rb1);
        if (w1->type != DS4_TENSOR_Q2_K ||
            in_dim0 != in_dim1 ||
            out_dim0 != out_dim1) {
            ds4_die("paired Q2_K expert tensors do not match");
        }

        for (uint64_t row = 0; row < out_dim0; row++) {
            const block_q2_K *gr = (const block_q2_K *)(base0 + row * rb0);
            ds4_vec_dot_q2_K_q8_K((int)in_dim0, &out0[row], gr, xq);
            const block_q2_K *ur = (const block_q2_K *)(base1 + row * rb1);
            ds4_vec_dot_q2_K_q8_K((int)in_dim0, &out1[row], ur, xq);
        }
    } else if (w0->type == DS4_TENSOR_Q4_K) {
        uint64_t in_dim0, out_dim0, rb0;
        uint64_t in_dim1, out_dim1, rb1;
        const uint8_t *base0 = tensor_expert_bytes(m, w0, expert, &in_dim0, &out_dim0, &rb0);
        const uint8_t *base1 = tensor_expert_bytes(m, w1, expert, &in_dim1, &out_dim1, &rb1);
        if (in_dim0 != in_dim1 || out_dim0 != out_dim1) ds4_die("paired Q4_K expert tensors do not match");

        for (uint64_t row = 0; row < out_dim0; row++) {
            const block_q4_K *gr = (const block_q4_K *)(base0 + row * rb0);
            ds4_vec_dot_q4_K_q8_K((int)in_dim0, &out0[row], gr, xq);
            const block_q4_K *ur = (const block_q4_K *)(base1 + row * rb1);
            ds4_vec_dot_q4_K_q8_K((int)in_dim0, &out1[row], ur, xq);
        }
    } else if (w0->type == DS4_TENSOR_Q5_K || w0->type == DS4_TENSOR_Q6_K) {
        uint64_t in_dim0, out_dim0, rb0;
        uint64_t in_dim1, out_dim1, rb1;
        const uint8_t *base0 = tensor_expert_bytes(m, w0, expert, &in_dim0, &out_dim0, &rb0);
        const uint8_t *base1 = tensor_expert_bytes(m, w1, expert, &in_dim1, &out_dim1, &rb1);
        if (in_dim0 != in_dim1 || out_dim0 != out_dim1) ds4_die("paired Q5_K/Q6_K expert tensors do not match");

        for (uint64_t row = 0; row < out_dim0; row++) {
            ds4_vec_dot_q5_q6_K_q8_K(w0->type, (int)in_dim0, &out0[row], base0 + row * rb0, xq);
            ds4_vec_dot_q5_q6_K_q8_K(w1->type, (int)in_dim0, &out1[row], base1 + row * rb1, xq);
        }
    } else {
        ds4_die("unsupported gate/up expert tensor type");
    }
}

/* Dispatch: single-expert down projection for tracing. */
static void matvec_expert_down(
        float            *out,
        const ds4_model  *m,
        const ds4_tensor *w,
        const float      *x,
        uint32_t          expert) {
    if (w->type == DS4_TENSOR_IQ2_XXS) {
        uint64_t in_dim, out_dim, row_bytes;
        const uint8_t *base = tensor_expert_bytes(m, w, expert, &in_dim, &out_dim, &row_bytes);
        if (in_dim % QK_K != 0) ds4_die("IQ2_XXS expert row is not QK_K aligned");

        block_q8_K *xq = xmalloc((size_t)(in_dim / QK_K) * sizeof(xq[0]));
        ds4_quantize_row_q8_K(x, xq, (int64_t)in_dim);

        for (uint64_t row = 0; row < out_dim; row++) {
            const block_iq2_xxs *br = (const block_iq2_xxs *)(base + row * row_bytes);
            ds4_vec_dot_iq2_xxs_q8_K((int)in_dim, &out[row], br, xq);
        }
        free(xq);
    } else if (w->type == DS4_TENSOR_Q2_K) {
        matvec_q2_k_expert(out, m, w, x, expert);
    } else if (w->type == DS4_TENSOR_Q4_K) {
        uint64_t in_dim, out_dim, row_bytes;
        const uint8_t *base = tensor_expert_bytes(m, w, expert, &in_dim, &out_dim, &row_bytes);
        if (in_dim % QK_K != 0) ds4_die("Q4_K expert row is not QK_K aligned");

        block_q8_K *xq = xmalloc((size_t)(in_dim / QK_K) * sizeof(xq[0]));
        ds4_quantize_row_q8_K(x, xq, (int64_t)in_dim);

        for (uint64_t row = 0; row < out_dim; row++) {
            const block_q4_K *br = (const block_q4_K *)(base + row * row_bytes);
            ds4_vec_dot_q4_K_q8_K((int)in_dim, &out[row], br, xq);
        }
        free(xq);
    } else if (w->type == DS4_TENSOR_Q5_K || w->type == DS4_TENSOR_Q6_K) {
        matvec_q5_q6_k_expert(out, m, w, x, expert);
    } else {
        ds4_die("unsupported down expert tensor type");
    }
}

typedef struct {
    float *moe;
    const uint8_t *base[DS4_MAX_EXPERT];
    const block_q8_K *midq;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t midq_blocks;
} matvec_q8_k_batch_accum_rows_ctx;

static void matvec_q8_k_batch_accum_rows_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q8_k_batch_accum_rows_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            ctx->moe[(uint64_t)t * ctx->out_dim + row] = 0.0f;
        }

        for (uint32_t ai = 0; ai < ctx->n_active; ai++) {
            const uint32_t expert = ctx->active_expert[ai];
            const uint32_t begin = ctx->expert_offset[expert];
            const uint32_t end = ctx->expert_offset[expert + 1];
            const block_q8_K *br = (const block_q8_K *)(ctx->base[expert] + row * ctx->row_bytes[expert]);

            for (uint32_t i = begin; i < end; i++) {
                const uint32_t pair_id = ctx->pair_ids[i];
                const ds4_expert_pair pair = ctx->pairs[pair_id];
                const block_q8_K *xq = ctx->midq + (uint64_t)pair_id * ctx->midq_blocks;
                float v = 0.0f;

                ds4_vec_dot_q8_K_q8_K((int)ctx->in_dim, &v, br, xq);
                ctx->moe[(uint64_t)pair.token * ctx->out_dim + row] += v;
            }
        }
    }
}

typedef struct {
    float *moe;
    const uint8_t *base[DS4_MAX_EXPERT];
    const int8_t *midq;
    const float *midscale;
    const ds4_expert_pair *pairs;
    const uint32_t *pair_ids;
    const uint32_t *expert_offset;
    const uint32_t *active_expert;
    uint32_t n_active;
    uint32_t n_tok;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t row_bytes[DS4_MAX_EXPERT];
    uint64_t blocks;
} matvec_q8_0_batch_accum_rows_ctx;

static void matvec_q8_0_batch_accum_rows_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_q8_0_batch_accum_rows_ctx *ctx = vctx;

    for (uint64_t row = row0; row < row1; row++) {
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            ctx->moe[(uint64_t)t * ctx->out_dim + row] = 0.0f;
        }

        for (uint32_t ai = 0; ai < ctx->n_active; ai++) {
            const uint32_t expert = ctx->active_expert[ai];
            const uint32_t begin = ctx->expert_offset[expert];
            const uint32_t end = ctx->expert_offset[expert + 1];
            const uint8_t *br = ctx->base[expert] + row * ctx->row_bytes[expert];

            for (uint32_t i = begin; i < end; i++) {
                const uint32_t pair_id = ctx->pair_ids[i];
                const ds4_expert_pair pair = ctx->pairs[pair_id];
                const int8_t *xq = ctx->midq + (uint64_t)pair_id * ctx->blocks * 32u;
                const float *xscale = ctx->midscale + (uint64_t)pair_id * ctx->blocks;
                const float v = dot_q8_0_row(br, xq, xscale, ctx->in_dim, ctx->blocks);
                ctx->moe[(uint64_t)pair.token * ctx->out_dim + row] += v;
            }
        }
    }
}

typedef struct {
    float *moe;
    const float *down_pair;
    uint32_t n_tok;
    uint64_t out_dim;
} sum_down_pairs_ctx;

static DS4_MAYBE_UNUSED void sum_down_pairs_worker(void *vctx, uint64_t row0, uint64_t row1) {
    sum_down_pairs_ctx *ctx = vctx;
    for (uint64_t idx = row0; idx < row1; idx++) {
        const uint32_t token = (uint32_t)(idx / ctx->out_dim);
        const uint64_t row = idx - (uint64_t)token * ctx->out_dim;
        float acc = 0.0f;
        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            const uint64_t pair_id = (uint64_t)token * DS4_N_EXPERT_USED + slot;
            acc += ctx->down_pair[pair_id * ctx->out_dim + row];
        }
        ctx->moe[idx] = acc;
    }
}

/* =========================================================================
 * Hyper-Connection Transforms.
 * =========================================================================
 *
 * DeepSeek V4 Flash keeps four hyper-connection streams per token.  Before
 * attention or FFN, a learned small projection chooses how to reduce the HC
 * state into the 4096-wide sublayer input.  After the sublayer, the post and
 * combine weights expand the result back into the four-stream HC state.
 */

/* Decode the HC control projection.  The output contains pre weights, post
 * gates, and a small doubly-normalized combine matrix. */
static void hc_split_sinkhorn_one(
        float       * out,
        const float * mix,
        const float * scale,
        const float * base,
        int           n_hc,
        int           iters,
        float         eps) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < n_hc; i++) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }

    for (int i = 0; i < n_hc; i++) {
        const int off = n_hc + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[16 * 16];

    for (int dst = 0; dst < n_hc; dst++) {
        float row_max = DS4_NEG_INF;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const int off = 2 * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }

        float row_sum = 0.0f;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv = 1.0f / row_sum;
        for (int src = 0; src < n_hc; src++) {
            const int idx = src + dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }

    for (int src = 0; src < n_hc; src++) {
        float sum = 0.0f;
        for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];

        const float inv = 1.0f / (sum + eps);
        for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
    }

    for (int iter = 1; iter < iters; iter++) {
        for (int dst = 0; dst < n_hc; dst++) {
            float sum = 0.0f;
            for (int src = 0; src < n_hc; src++) sum += c[src + dst * n_hc];

            const float inv = 1.0f / (sum + eps);
            for (int src = 0; src < n_hc; src++) c[src + dst * n_hc] *= inv;
        }

        for (int src = 0; src < n_hc; src++) {
            float sum = 0.0f;
            for (int dst = 0; dst < n_hc; dst++) sum += c[src + dst * n_hc];

            const float inv = 1.0f / (sum + eps);
            for (int dst = 0; dst < n_hc; dst++) c[src + dst * n_hc] *= inv;
        }
    }

    for (int i = 0; i < n_hc * n_hc; i++) out[2 * n_hc + i] = c[i];
}

/* Reduce the four HC streams into the plain embedding vector consumed by a
 * normal attention or FFN sublayer. */
static void hc_weighted_sum_one(
        float       * out,
        const float * x,
        const float * weights,
        uint32_t      n_embd,
        uint32_t      n_hc) {
    for (uint32_t d = 0; d < n_embd; d++) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < n_hc; h++) {
            acc += x[(uint64_t)h * n_embd + d] * weights[h];
        }
        out[d] = acc;
    }
}

/* HC pre step for one token.  It normalizes the HC state, projects the control
 * vector, runs the Sinkhorn split, and emits the sublayer input plus post data. */
static void hc_pre_from_state_one_scratch(
        const ds4_model   * model,
        const ds4_tensor  * fn,
        const ds4_tensor  * scale_tensor,
        const ds4_tensor  * base_tensor,
        const float       * residual_hc,
        float             * out,
        float             * post,
        float             * comb,
        float             * flat,
        bool                serial_fn) {
    const uint32_t n_hc = DS4_N_HC;
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * n_hc;

    float mix[24];
    float split[24];

    rms_norm_no_weight(flat, residual_hc, hc_dim, DS4_RMS_EPS);
    if (serial_fn) {
        matvec_f16_serial(mix, model, fn, flat);
    } else {
        matvec_f16(mix, model, fn, flat);
    }

    const float *scale = tensor_data(model, scale_tensor);
    const float *base = tensor_data(model, base_tensor);
    hc_split_sinkhorn_one(split, mix, scale, base, (int)n_hc, DS4_N_HC_SINKHORN_ITER, 1.0e-6f);
    hc_weighted_sum_one(out, residual_hc, split, DS4_N_EMBD, n_hc);

    memcpy(post, split + n_hc, n_hc * sizeof(post[0]));
    memcpy(comb, split + 2 * n_hc, n_hc * n_hc * sizeof(comb[0]));
}

static void hc_pre_from_state_one(
        const ds4_model   * model,
        const ds4_tensor  * fn,
        const ds4_tensor  * scale_tensor,
        const ds4_tensor  * base_tensor,
        const float       * residual_hc,
        float             * out,
        float             * post,
        float             * comb) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    float *flat = xmalloc((size_t)hc_dim * sizeof(flat[0]));

    hc_pre_from_state_one_scratch(model,
                                  fn, scale_tensor, base_tensor,
                                  residual_hc, out, post, comb,
                                  flat, false);
    free(flat);
}

static void layer_attn_pre_one(
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * token_embd,
        float             * out,
        float             * residual_hc,
        float             * post,
        float             * comb) {
    const uint32_t n_hc = DS4_N_HC;

    for (uint32_t h = 0; h < n_hc; h++) {
        memcpy(residual_hc + (uint64_t)h * DS4_N_EMBD, token_embd, (size_t)DS4_N_EMBD * sizeof(token_embd[0]));
    }

    hc_pre_from_state_one(model,
                          layer->hc_attn_fn,
                          layer->hc_attn_scale,
                          layer->hc_attn_base,
                          residual_hc, out, post, comb);
}

/* The input embedding starts all HC streams with the same token vector. */
static void hc_from_plain_embedding(float *out_hc, const float *x, uint32_t n_embd, uint32_t n_hc) {
    for (uint32_t h = 0; h < n_hc; h++) {
        memcpy(out_hc + (uint64_t)h * n_embd, x, (size_t)n_embd * sizeof(x[0]));
    }
}

/* HC post step for one sublayer output.  It injects the new block output and
 * mixes the previous HC streams through the learned combine matrix. */
static void hc_post_one(
        float       * out_hc,
        const float * block_out,
        const float * residual_hc,
        const float * post,
        const float * comb,
        uint32_t      n_embd,
        uint32_t      n_hc) {
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = block_out[d] * post[dst];

            for (uint32_t src = 0; src < n_hc; src++) {
                /* The HC combine matrix is addressed as [dst_hc, src_hc]. */
                acc += comb[dst + src * n_hc] * residual_hc[(uint64_t)src * n_embd + d];
            }

            out_hc[(uint64_t)dst * n_embd + d] = acc;
        }
    }
}

typedef struct {
    float       *out_hc;
    const float *block_out;
    const float *residual_hc;
    const float *post;
    const float *comb;
    uint64_t     hc_dim;
    uint32_t     n_embd;
    uint32_t     n_hc;
} hc_post_batch_ctx;

static void hc_post_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    hc_post_batch_ctx *ctx = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        hc_post_one(ctx->out_hc + t * ctx->hc_dim,
                    ctx->block_out + t * ctx->n_embd,
                    ctx->residual_hc + t * ctx->hc_dim,
                    ctx->post + t * ctx->n_hc,
                    ctx->comb + t * ctx->n_hc * ctx->n_hc,
                    ctx->n_embd,
                    ctx->n_hc);
    }
}

static void hc_post_batch(
        float       * out_hc,
        const float * block_out,
        const float * residual_hc,
        const float * post,
        const float * comb,
        uint32_t      n_tok,
        uint32_t      n_embd,
        uint32_t      n_hc) {
    hc_post_batch_ctx ctx = {
        .out_hc = out_hc,
        .block_out = block_out,
        .residual_hc = residual_hc,
        .post = post,
        .comb = comb,
        .hc_dim = (uint64_t)n_hc * n_embd,
        .n_embd = n_embd,
        .n_hc = n_hc,
    };
    ds4_parallel_for_min_rows(n_tok, hc_post_batch_worker, &ctx, 1);
}

typedef struct {
    float       *out_hc;
    const float *moe;
    const float *shared;
    const float *residual_hc;
    const float *post;
    const float *comb;
    uint64_t     hc_dim;
    uint32_t     n_embd;
    uint32_t     n_hc;
} hc_post_sum_batch_ctx;

static void hc_post_sum_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    hc_post_sum_batch_ctx *ctx = vctx;
    for (uint64_t t = t0; t < t1; t++) {
        const float *moe = ctx->moe + t * ctx->n_embd;
        const float *shared = ctx->shared + t * ctx->n_embd;
        const float *residual = ctx->residual_hc + t * ctx->hc_dim;
        const float *post = ctx->post + t * ctx->n_hc;
        const float *comb = ctx->comb + t * ctx->n_hc * ctx->n_hc;
        float *out = ctx->out_hc + t * ctx->hc_dim;

        for (uint32_t dst = 0; dst < ctx->n_hc; dst++) {
            for (uint32_t d = 0; d < ctx->n_embd; d++) {
                float acc = (moe[d] + shared[d]) * post[dst];
                for (uint32_t src = 0; src < ctx->n_hc; src++) {
                    acc += comb[dst + src * ctx->n_hc] *
                        residual[(uint64_t)src * ctx->n_embd + d];
                }
                out[(uint64_t)dst * ctx->n_embd + d] = acc;
            }
        }
    }
}

static void hc_post_sum_batch(
        float       * out_hc,
        const float * moe,
        const float * shared,
        const float * residual_hc,
        const float * post,
        const float * comb,
        uint32_t      n_tok,
        uint32_t      n_embd,
        uint32_t      n_hc) {
    hc_post_sum_batch_ctx ctx = {
        .out_hc = out_hc,
        .moe = moe,
        .shared = shared,
        .residual_hc = residual_hc,
        .post = post,
        .comb = comb,
        .hc_dim = (uint64_t)n_hc * n_embd,
        .n_embd = n_embd,
        .n_hc = n_hc,
    };
    ds4_parallel_for_min_rows(n_tok, hc_post_sum_batch_worker, &ctx, 1);
}

typedef struct {
    const ds4_model *model;
    const ds4_tensor *fn;
    const ds4_tensor *scale;
    const ds4_tensor *base;
    const ds4_tensor *norm_w;
    const float *inp_hc;
    float *residual_hc;
    float *cur;
    float *norm;
    float *post;
    float *comb;
    uint64_t hc_dim;
    uint32_t n_hc;
} hc_pre_norm_batch_ctx;

static void hc_pre_norm_batch_worker(void *vctx, uint64_t t0, uint64_t t1) {
    hc_pre_norm_batch_ctx *ctx = vctx;
    const float *norm_w = tensor_data(ctx->model, ctx->norm_w);
    float *flat = xmalloc((size_t)ctx->hc_dim * sizeof(flat[0]));

    for (uint64_t t = t0; t < t1; t++) {
        const float *residual = ctx->inp_hc + t * ctx->hc_dim;
        if (ctx->residual_hc) {
            float *dst = ctx->residual_hc + t * ctx->hc_dim;
            memcpy(dst, residual, (size_t)ctx->hc_dim * sizeof(dst[0]));
            residual = dst;
        }

        hc_pre_from_state_one_scratch(ctx->model,
                                      ctx->fn,
                                      ctx->scale,
                                      ctx->base,
                                      residual,
                                      ctx->cur + t * DS4_N_EMBD,
                                      ctx->post + t * ctx->n_hc,
                                      ctx->comb + t * ctx->n_hc * ctx->n_hc,
                                      flat,
                                      true);
        rms_norm_weight(ctx->norm + t * DS4_N_EMBD,
                        ctx->cur + t * DS4_N_EMBD,
                        norm_w,
                        DS4_N_EMBD,
                        DS4_RMS_EPS);
    }

    free(flat);
}

/* Batched HC pre plus RMSNorm.  Prefill uses this to keep the layer-major
 * token batch in contiguous arrays. */
static void hc_pre_norm_batch(
        const ds4_model  * model,
        const ds4_tensor * fn,
        const ds4_tensor * scale,
        const ds4_tensor * base,
        const ds4_tensor * norm_w,
        const float      * inp_hc,
        float            * residual_hc,
        float            * cur,
        float            * norm,
        float            * post,
        float            * comb,
        uint32_t           n_tok) {
    hc_pre_norm_batch_ctx ctx = {
        .model = model,
        .fn = fn,
        .scale = scale,
        .base = base,
        .norm_w = norm_w,
        .inp_hc = inp_hc,
        .residual_hc = residual_hc,
        .cur = cur,
        .norm = norm,
        .post = post,
        .comb = comb,
        .hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD,
        .n_hc = DS4_N_HC,
    };
    ds4_parallel_for_min_rows(n_tok, hc_pre_norm_batch_worker, &ctx, 1);
}

static void layer_attn_norm_one(
        float             * out,
        const ds4_model   * model,
        const ds4_layer_weights * layer,
        const float       * x) {
    const float *attn_norm = tensor_data(model, layer->attn_norm);
    rms_norm_weight(out, x, attn_norm, DS4_N_EMBD, DS4_RMS_EPS);
}

