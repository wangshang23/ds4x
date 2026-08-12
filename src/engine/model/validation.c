#include "engine_internal.h"

/* Model Validation module. */
/* =========================================================================
 * Fixed Weight Binding and Model Validation.
 * =========================================================================
 *
 * The GGUF tensor directory is converted into a DS4-specific pointer table.
 * After this section, the rest of the program addresses tensors by semantic
 * fields such as layer->attn_q_a or layer->ffn_gate_exps rather than by string
 * lookup.  Shape validation is intentionally strict.
 */

static uint32_t required_u32(const ds4_model *m, const char *key) {
    uint32_t v = 0;
    if (!model_get_u32(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static float required_f32(const ds4_model *m, const char *key) {
    float v = 0.0f;
    if (!model_get_f32_compat(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static bool required_bool(const ds4_model *m, const char *key) {
    bool v = false;
    if (!model_get_bool(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static ds4_tensor *required_tensor(const ds4_model *m, const char *name) {
    ds4_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "ds4: required tensor is missing: %s\n", name);
        exit(1);
    }
    return t;
}

static ds4_tensor *tensor_by_namef(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return model_find_tensor(m, name);
}

static ds4_tensor *required_tensorf(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return required_tensor(m, name);
}

ds4_tensor *tensor_by_mtp_stage_suffix(
        const ds4_model *m,
        uint32_t         stage,
        const char      *suffix) {
    char name[160];
    int n = snprintf(name, sizeof(name), "mtp.%u.%s", stage, suffix);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return model_find_tensor(m, name);
}

static void tensor_expect_layout(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating layout");
    if (t->type != type) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected %s\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type),
                tensor_type_name(type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}

bool tensor_type_is_dense_quant(uint32_t type) {
    return type == DS4_TENSOR_Q8_0 ||
           type == DS4_TENSOR_Q4_K ||
           type == DS4_TENSOR_Q4_0;
}

static void tensor_expect_dense_quant_layout(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating dense quant layout");
    if (!tensor_type_is_dense_quant(t->type)) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected q8_0, q4_K, or q4_0\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_optional(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (t) tensor_expect_layout(t, type, ndim, d0, d1, d2);
}

bool tensor_type_is_f16_or_q8_0(uint32_t type) {
    return type == DS4_TENSOR_F16 || type == DS4_TENSOR_Q8_0;
}

static void tensor_expect_f16_or_q8_0_layout(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing tensor while validating layout");
    if (!tensor_type_is_f16_or_q8_0(t->type)) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected f16 or q8_0\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

bool tensor_is_routed_expert_type(uint32_t type) {
    return type == DS4_TENSOR_Q8_0 ||
           type == DS4_TENSOR_IQ2_XXS ||
           type == DS4_TENSOR_Q2_K ||
           type == DS4_TENSOR_Q4_K ||
           type == DS4_TENSOR_MXFP4;
}

static uint64_t routed_expert_block_bytes(uint32_t type) {
    switch (type) {
    case DS4_TENSOR_Q8_0:    return 34;
    case DS4_TENSOR_IQ2_XXS: return sizeof(block_iq2_xxs);
    case DS4_TENSOR_Q2_K:    return sizeof(block_q2_K);
    case DS4_TENSOR_Q4_K:    return sizeof(block_q4_K);
    case DS4_TENSOR_MXFP4:   return sizeof(block_mxfp4);
    default:                 ds4_die("unsupported routed expert tensor type");
    }
    return 0;
}

uint64_t routed_expert_row_bytes(const ds4_tensor *t) {
    const gguf_type_info *info = tensor_type(t->type);
    if (!info || info->block_elems == 0) ds4_die("unsupported routed expert tensor type");
    if ((t->dim[0] % info->block_elems) != 0) ds4_die("routed expert row is not quant block aligned");
    return (t->dim[0] / info->block_elems) * routed_expert_block_bytes(t->type);
}

uint64_t ds4_add_sat_u64(uint64_t a, uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

double ds4_bytes_to_gib(uint64_t bytes) {
    return (double)bytes / 1073741824.0;
}

static void tensor_expect_routed_expert(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) ds4_die("internal error: missing routed expert tensor while validating layout");
    if (!tensor_is_routed_expert_type(t->type)) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %u (%s), expected a routed expert quant type\n",
                (int)t->name.len,
                t->name.ptr,
                t->type,
                tensor_type_name(t->type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}

bool weights_have_output_head(const ds4_weights *w) {
    return w &&
           w->output_hc_base &&
           w->output_hc_fn &&
           w->output_hc_scale &&
           w->output_norm &&
           w->output;
}

static bool weights_have_partial_output_head(const ds4_weights *w) {
    return w &&
           (w->output_hc_base ||
            w->output_hc_fn ||
            w->output_hc_scale ||
            w->output_norm ||
            w->output);
}

bool weights_layer_has_required(const ds4_layer_weights *l, uint32_t il) {
    if (!l) return false;
    if (!l->hc_attn_fn ||
        !l->hc_attn_scale ||
        !l->hc_attn_base ||
        !l->attn_norm ||
        !l->attn_q_a ||
        !l->attn_q_a_norm ||
        !l->attn_q_b ||
        !l->attn_kv ||
        !l->attn_kv_a_norm ||
        !l->attn_sinks ||
        !l->attn_output_a ||
        !l->attn_output_b ||
        !l->hc_ffn_fn ||
        !l->hc_ffn_scale ||
        !l->hc_ffn_base ||
        !l->ffn_norm ||
        !l->ffn_gate_inp ||
        !l->ffn_gate_exps ||
        !l->ffn_up_exps ||
        !l->ffn_down_exps ||
        !l->ffn_gate_shexp ||
        !l->ffn_up_shexp ||
        !l->ffn_down_shexp)
    {
        return false;
    }

    const uint32_t ratio = ds4_layer_compress_ratio(il);
    if (ratio != 0 &&
        (!l->attn_compressor_ape ||
         !l->attn_compressor_kv ||
         !l->attn_compressor_gate ||
         !l->attn_compressor_norm))
    {
        return false;
    }
    if (ratio == 4 &&
        (!l->indexer_attn_q_b ||
         !l->indexer_proj ||
         !l->indexer_compressor_ape ||
         !l->indexer_compressor_kv ||
         !l->indexer_compressor_gate ||
         !l->indexer_compressor_norm))
    {
        return false;
    }
    if (il < DS4_N_HASH_LAYER && !l->ffn_gate_tid2eid) return false;
    return true;
}

const ds4_layer_weights *weights_first_bound_layer(const ds4_weights *w) {
    if (!w) return NULL;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (weights_layer_has_required(&w->layer[il], il)) return &w->layer[il];
    }
    return NULL;
}

/* Verify every tensor type and dimension used by the specialized pipeline.
 * For distributed sliced GGUFs, only the advertised local layer range is
 * required; token embedding and output head are validated when present. */
static void weights_validate_layout(
        const ds4_weights *w,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               require_token_embd,
        bool               require_output) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;

    if (!w) ds4_die("internal error: missing weights while validating layout");
    if (layer_start >= DS4_N_LAYER) ds4_die("invalid first layer in weight layout validation");
    if (layer_end == UINT32_MAX) layer_end = DS4_N_LAYER - 1u;
    if (layer_end >= DS4_N_LAYER || layer_end < layer_start) {
        ds4_die("invalid layer range in weight layout validation");
    }

    if (require_token_embd && !w->token_embd) ds4_die("required token embedding tensor is missing");
    if (w->token_embd) {
        tensor_expect_layout(w->token_embd, DS4_TENSOR_F16, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    }

    const bool have_output = weights_have_output_head(w);
    if (require_output && !have_output) ds4_die("required output head tensors are missing");
    if (weights_have_partial_output_head(w) && !have_output) ds4_die("partial output head in GGUF");
    if (have_output) {
        tensor_expect_layout(w->output_hc_base,  DS4_TENSOR_F32,  1, DS4_N_HC, 0, 0);
        tensor_expect_layout(w->output_hc_fn,    DS4_TENSOR_F16,  2, hc_dim, DS4_N_HC, 0);
        tensor_expect_layout(w->output_hc_scale, DS4_TENSOR_F32,  1, 1, 0, 0);
        tensor_expect_layout(w->output_norm,     DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_dense_quant_layout(w->output,          2, DS4_N_EMBD, DS4_N_VOCAB, 0);
    }

    for (uint32_t il = layer_start; il <= layer_end; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!weights_layer_has_required(l, il)) {
            fprintf(stderr, "ds4: required tensors for layer %u are missing\n", il);
            exit(1);
        }

        tensor_expect_layout(l->hc_attn_fn,     DS4_TENSOR_F16,  2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_attn_scale,  DS4_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_attn_base,   DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->attn_norm,      DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_dense_quant_layout(l->attn_q_a,       2, DS4_N_EMBD, DS4_N_LORA_Q, 0);
        tensor_expect_layout(l->attn_q_a_norm,  DS4_TENSOR_F32,  1, DS4_N_LORA_Q, 0, 0);
        tensor_expect_dense_quant_layout(l->attn_q_b,       2, DS4_N_LORA_Q, q_dim, 0);
        tensor_expect_dense_quant_layout(l->attn_kv,        2, DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
        tensor_expect_layout(l->attn_kv_a_norm, DS4_TENSOR_F32,  1, DS4_N_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->attn_sinks,     DS4_TENSOR_F32,  1, DS4_N_HEAD, 0, 0);
        tensor_expect_dense_quant_layout(l->attn_output_a,  2, DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low_dim, 0);
        tensor_expect_dense_quant_layout(l->attn_output_b,  2, out_low_dim, DS4_N_EMBD, 0);

        if (ratio != 0) {
            const uint32_t coff = ratio == 4 ? 2u : 1u;
            const uint64_t comp_width = (uint64_t)coff * DS4_N_HEAD_DIM;
            tensor_expect_layout(l->attn_compressor_ape,  DS4_TENSOR_F16, 2, comp_width, ratio, 0);
            tensor_expect_layout(l->attn_compressor_kv,   DS4_TENSOR_F16, 2, DS4_N_EMBD, comp_width, 0);
            tensor_expect_layout(l->attn_compressor_gate, DS4_TENSOR_F16, 2, DS4_N_EMBD, comp_width, 0);
            tensor_expect_layout(l->attn_compressor_norm, DS4_TENSOR_F32, 1, DS4_N_HEAD_DIM, 0, 0);
        }
        if (ratio == 4) {
            const uint64_t index_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_width = 2u * DS4_N_INDEXER_HEAD_DIM;
            tensor_expect_f16_or_q8_0_layout(l->indexer_attn_q_b, 2, DS4_N_LORA_Q, index_q_dim, 0);
            tensor_expect_layout(l->indexer_proj,              DS4_TENSOR_F16, 2, DS4_N_EMBD, DS4_N_INDEXER_HEAD, 0);
            tensor_expect_layout(l->indexer_compressor_ape,    DS4_TENSOR_F16, 2, index_width, ratio, 0);
            tensor_expect_layout(l->indexer_compressor_kv,     DS4_TENSOR_F16, 2, DS4_N_EMBD, index_width, 0);
            tensor_expect_layout(l->indexer_compressor_gate,   DS4_TENSOR_F16, 2, DS4_N_EMBD, index_width, 0);
            tensor_expect_layout(l->indexer_compressor_norm,   DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0);
        }

        tensor_expect_layout(l->hc_ffn_fn,      DS4_TENSOR_F16,  2, hc_dim, hc_mix_dim, 0);
        tensor_expect_layout(l->hc_ffn_scale,   DS4_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_ffn_base,    DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->ffn_norm,       DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_layout(l->ffn_gate_inp,   DS4_TENSOR_F16,  2, DS4_N_EMBD, DS4_N_EXPERT, 0);
        tensor_expect_optional(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0);
        tensor_expect_routed_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
        tensor_expect_routed_expert(l->ffn_up_exps,   3, DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
        tensor_expect_routed_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
        if (l->ffn_gate_exps->type != l->ffn_up_exps->type) {
            fprintf(stderr, "ds4: routed gate/up experts use different quant types in layer %u\n", il);
            exit(1);
        }
        tensor_expect_dense_quant_layout(l->ffn_gate_shexp, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
        tensor_expect_dense_quant_layout(l->ffn_up_shexp,   2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
        tensor_expect_dense_quant_layout(l->ffn_down_shexp, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
        if (il < DS4_N_HASH_LAYER) {
            tensor_expect_layout(l->ffn_gate_tid2eid, DS4_TENSOR_I32, 2, DS4_N_EXPERT_USED, DS4_N_VOCAB, 0);
        }
    }
}

static const char *dspark_layout_kind_name(ds4_dspark_layout_kind kind) {
    switch (kind) {
    case DS4_DSPARK_LAYOUT_F32:    return "F32";
    case DS4_DSPARK_LAYOUT_PLAIN:  return "F16 or F32";
    case DS4_DSPARK_LAYOUT_DENSE:  return "F16, F32, or Q8_0";
    case DS4_DSPARK_LAYOUT_ROUTED: return "routed expert quant";
    }
    return "unknown";
}

bool dspark_tensor_type_matches(uint32_t type,
                                       ds4_dspark_layout_kind kind) {
    switch (kind) {
    case DS4_DSPARK_LAYOUT_F32:
        return type == DS4_TENSOR_F32;
    case DS4_DSPARK_LAYOUT_PLAIN:
        return type == DS4_TENSOR_F16 || type == DS4_TENSOR_F32;
    case DS4_DSPARK_LAYOUT_DENSE:
        return type == DS4_TENSOR_F16 ||
               type == DS4_TENSOR_F32 ||
               type == DS4_TENSOR_Q8_0;
    case DS4_DSPARK_LAYOUT_ROUTED:
        return tensor_is_routed_expert_type(type);
    }
    return false;
}

static void dspark_validate_tensor_layout(
        ds4_dspark_weights       *dw,
        const ds4_tensor         *t,
        const char               *role,
        ds4_dspark_layout_kind    kind,
        uint32_t                  ndim,
        uint64_t                  d0,
        uint64_t                  d1,
        uint64_t                  d2) {
    if (!dw || !t) return;

    bool ok = true;
    if (!dspark_tensor_type_matches(t->type, kind)) {
        fprintf(stderr,
                "ds4: DSpark tensor %.*s (%s) has type %s, expected %s\n",
                (int)t->name.len,
                t->name.ptr,
                role,
                tensor_type_name(t->type),
                dspark_layout_kind_name(kind));
        ok = false;
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: DSpark tensor %.*s (%s) has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                role,
                t->ndim,
                ndim);
        ok = false;
    }

    const uint64_t want[3] = { d0, d1, d2 };
    const uint32_t n = t->ndim < ndim ? t->ndim : ndim;
    for (uint32_t i = 0; i < n; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: DSpark tensor %.*s (%s) has dim[%u]=%" PRIu64
                ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                role,
                i,
                t->dim[i],
                want[i]);
        ok = false;
    }
    if (!ok) dw->invalid_tensors++;
}

static void dspark_weights_note_metadata_error(
        ds4_dspark_weights *dw,
        const char         *msg) {
    if (!dw) return;
    fprintf(stderr, "ds4: DSpark metadata error: %s\n", msg);
    dw->metadata_errors++;
}

static void dspark_weights_validate_metadata(ds4_dspark_weights *dw) {
    if (!dw) return;
    if (!dw->has_block_size || dw->block_size == 0) {
        dspark_weights_note_metadata_error(dw, "missing or zero block size");
    } else if (dw->block_size > DS4_DSPARK_MAX_BLOCK_SIZE) {
        dspark_weights_note_metadata_error(dw, "block size exceeds runtime limit");
    }
    if (!dw->has_markov_rank || dw->markov_rank == 0) {
        dspark_weights_note_metadata_error(dw, "missing or zero Markov rank");
    }
    if (!dw->has_noise_token_id || dw->noise_token_id >= DS4_N_VOCAB) {
        dspark_weights_note_metadata_error(dw, "missing or out-of-range noise token");
    }
    if (!dw->has_target_layers || dw->target_layer_count == 0) {
        dspark_weights_note_metadata_error(dw, "missing target layer list");
        return;
    }

    uint32_t prev = UINT32_MAX;
    for (uint32_t i = 0; i < dw->target_layer_count; i++) {
        const uint32_t layer = dw->target_layers[i];
        if (layer >= DS4_N_LAYER) {
            dspark_weights_note_metadata_error(dw, "target layer is outside the target model");
        }
        if (i != 0 && layer <= prev) {
            dspark_weights_note_metadata_error(dw, "target layers are not strictly increasing");
        }
        prev = layer;
    }
}

static void dspark_weights_validate_block_layout(
        ds4_dspark_weights *dw,
        const ds4_layer_weights *l) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;

    dspark_validate_tensor_layout(dw, l->hc_attn_fn, "hc_attn_fn",
                                  DS4_DSPARK_LAYOUT_PLAIN, 2,
                                  hc_dim, hc_mix_dim, 0);
    dspark_validate_tensor_layout(dw, l->hc_attn_scale, "hc_attn_scale",
                                  DS4_DSPARK_LAYOUT_F32, 1, 3, 0, 0);
    dspark_validate_tensor_layout(dw, l->hc_attn_base, "hc_attn_base",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  hc_mix_dim, 0, 0);
    dspark_validate_tensor_layout(dw, l->attn_norm, "attn_norm",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_EMBD, 0, 0);
    dspark_validate_tensor_layout(dw, l->attn_q_a, "attn_q_a",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_EMBD, DS4_N_LORA_Q, 0);
    dspark_validate_tensor_layout(dw, l->attn_q_a_norm, "attn_q_a_norm",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_LORA_Q, 0, 0);
    dspark_validate_tensor_layout(dw, l->attn_q_b, "attn_q_b",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_LORA_Q, q_dim, 0);
    dspark_validate_tensor_layout(dw, l->attn_kv, "attn_kv",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
    dspark_validate_tensor_layout(dw, l->attn_kv_a_norm, "attn_kv_a_norm",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_HEAD_DIM, 0, 0);
    dspark_validate_tensor_layout(dw, l->attn_sinks, "attn_sinks",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_HEAD, 0, 0);
    dspark_validate_tensor_layout(dw, l->attn_output_a, "attn_output_a",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP),
                                  out_low_dim, 0);
    dspark_validate_tensor_layout(dw, l->attn_output_b, "attn_output_b",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  out_low_dim, DS4_N_EMBD, 0);

    dspark_validate_tensor_layout(dw, l->hc_ffn_fn, "hc_ffn_fn",
                                  DS4_DSPARK_LAYOUT_PLAIN, 2,
                                  hc_dim, hc_mix_dim, 0);
    dspark_validate_tensor_layout(dw, l->hc_ffn_scale, "hc_ffn_scale",
                                  DS4_DSPARK_LAYOUT_F32, 1, 3, 0, 0);
    dspark_validate_tensor_layout(dw, l->hc_ffn_base, "hc_ffn_base",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  hc_mix_dim, 0, 0);
    dspark_validate_tensor_layout(dw, l->ffn_norm, "ffn_norm",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_EMBD, 0, 0);
    dspark_validate_tensor_layout(dw, l->ffn_gate_inp, "ffn_gate_inp",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_EMBD, DS4_N_EXPERT, 0);
    dspark_validate_tensor_layout(dw, l->ffn_exp_probs_b, "exp_probs_b",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_EXPERT, 0, 0);
    dspark_validate_tensor_layout(dw, l->ffn_gate_exps, "ffn_gate_exps",
                                  DS4_DSPARK_LAYOUT_ROUTED, 3,
                                  DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    dspark_validate_tensor_layout(dw, l->ffn_up_exps, "ffn_up_exps",
                                  DS4_DSPARK_LAYOUT_ROUTED, 3,
                                  DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
    dspark_validate_tensor_layout(dw, l->ffn_down_exps, "ffn_down_exps",
                                  DS4_DSPARK_LAYOUT_ROUTED, 3,
                                  DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
    if (l->ffn_gate_exps &&
        l->ffn_up_exps &&
        l->ffn_gate_exps->type != l->ffn_up_exps->type) {
        fprintf(stderr,
                "ds4: DSpark routed gate/up experts use different quant types\n");
        dw->invalid_tensors++;
    }
    dspark_validate_tensor_layout(dw, l->ffn_gate_shexp, "ffn_gate_shexp",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_EMBD, DS4_N_FF_EXP, 0);
    dspark_validate_tensor_layout(dw, l->ffn_up_shexp, "ffn_up_shexp",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_EMBD, DS4_N_FF_EXP, 0);
    dspark_validate_tensor_layout(dw, l->ffn_down_shexp, "ffn_down_shexp",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  DS4_N_FF_EXP, DS4_N_EMBD, 0);
}

void dspark_weights_validate_layout(ds4_dspark_weights *dw) {
    if (!dw) return;
    dspark_weights_validate_metadata(dw);

    for (uint32_t stage = 0; stage < dw->n_stages; stage++) {
        ds4_dspark_stage_weights *sw = &dw->stage[stage];
        dspark_weights_validate_block_layout(dw, &sw->block);
        if (stage == 0) {
            dspark_validate_tensor_layout(dw, sw->main_proj, "main_proj",
                                          DS4_DSPARK_LAYOUT_DENSE, 2,
                                          (uint64_t)dw->target_layer_count *
                                              DS4_N_EMBD,
                                          DS4_N_EMBD, 0);
            dspark_validate_tensor_layout(dw, sw->main_norm, "main_norm",
                                          DS4_DSPARK_LAYOUT_F32, 1,
                                          DS4_N_EMBD, 0, 0);
        }
    }

    if (dw->n_stages == 0) return;
    ds4_dspark_stage_weights *final = &dw->stage[dw->n_stages - 1u];
    dspark_validate_tensor_layout(dw, final->norm, "norm",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_EMBD, 0, 0);
    dspark_validate_tensor_layout(dw, final->hc_head_base, "hc_head_base",
                                  DS4_DSPARK_LAYOUT_F32, 1,
                                  DS4_N_HC, 0, 0);
    dspark_validate_tensor_layout(dw, final->hc_head_fn, "hc_head_fn",
                                  DS4_DSPARK_LAYOUT_PLAIN, 2,
                                  (uint64_t)DS4_N_EMBD * DS4_N_HC,
                                  DS4_N_HC, 0);
    dspark_validate_tensor_layout(dw, final->hc_head_scale, "hc_head_scale",
                                  DS4_DSPARK_LAYOUT_F32, 1, 1, 0, 0);
    dspark_validate_tensor_layout(dw, final->markov_w1, "markov_w1",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  dw->markov_rank, DS4_N_VOCAB, 0);
    dspark_validate_tensor_layout(dw, final->markov_w2, "markov_w2",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  dw->markov_rank, DS4_N_VOCAB, 0);
    dspark_validate_tensor_layout(dw, final->confidence_proj,
                                  "confidence_proj",
                                  DS4_DSPARK_LAYOUT_DENSE, 2,
                                  (uint64_t)DS4_N_EMBD + dw->markov_rank,
                                  1, 0);
}

static bool ds4_shape_matches_metadata(
        const ds4_shape *s,
        uint32_t n_layer,
        uint32_t n_embd,
        uint32_t n_vocab,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_head_dim,
        uint32_t n_value_dim,
        uint32_t n_rot,
        uint32_t n_lora_q,
        uint32_t n_lora_o,
        uint32_t n_out_group,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_ff_exp,
        uint32_t n_expert_shared,
        uint32_t n_hash_layer,
        uint32_t n_swa,
        uint32_t n_indexer_head,
        uint32_t n_indexer_head_dim,
        uint32_t n_indexer_top_k,
        uint32_t n_hc,
        uint32_t n_hc_sinkhorn_iter) {
    return s->n_layer == n_layer &&
           s->n_embd == n_embd &&
           s->n_vocab == n_vocab &&
           s->n_head == n_head &&
           s->n_head_kv == n_head_kv &&
           s->n_head_dim == n_head_dim &&
           s->n_value_dim == n_value_dim &&
           s->n_rot == n_rot &&
           s->n_lora_q == n_lora_q &&
           s->n_lora_o == n_lora_o &&
           s->n_out_group == n_out_group &&
           s->n_expert == n_expert &&
           s->n_expert_used == n_expert_used &&
           s->n_ff_exp == n_ff_exp &&
           s->n_expert_shared == n_expert_shared &&
           s->n_hash_layer == n_hash_layer &&
           s->n_swa == n_swa &&
           s->n_indexer_head == n_indexer_head &&
           s->n_indexer_head_dim == n_indexer_head_dim &&
           s->n_indexer_top_k == n_indexer_top_k &&
           s->n_hc == n_hc &&
           s->n_hc_sinkhorn_iter == n_hc_sinkhorn_iter;
}

static void ds4_select_shape_from_metadata(
        uint32_t n_layer,
        uint32_t n_embd,
        uint32_t n_vocab,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_head_dim,
        uint32_t n_value_dim,
        uint32_t n_rot,
        uint32_t n_lora_q,
        uint32_t n_lora_o,
        uint32_t n_out_group,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_ff_exp,
        uint32_t n_expert_shared,
        uint32_t n_hash_layer,
        uint32_t n_swa,
        uint32_t n_indexer_head,
        uint32_t n_indexer_head_dim,
        uint32_t n_indexer_top_k,
        uint32_t n_hc,
        uint32_t n_hc_sinkhorn_iter) {
    if (ds4_shape_matches_metadata(&DS4_SHAPE_FLASH,
                                   n_layer, n_embd, n_vocab, n_head, n_head_kv,
                                   n_head_dim, n_value_dim, n_rot, n_lora_q,
                                   n_lora_o, n_out_group, n_expert,
                                   n_expert_used, n_ff_exp, n_expert_shared,
                                   n_hash_layer, n_swa, n_indexer_head,
                                   n_indexer_head_dim, n_indexer_top_k, n_hc,
                                   n_hc_sinkhorn_iter)) {
        g_ds4_shape = DS4_SHAPE_FLASH;
        return;
    }
    fprintf(stderr,
            "ds4: unsupported DeepSeek4 shape: layers=%u embd=%u heads=%u "
            "q_lora=%u out_groups=%u experts=%u ff_exp=%u indexer_top_k=%u\n",
            n_layer,
            n_embd,
            n_head,
            n_lora_q,
            n_out_group,
            n_expert,
            n_ff_exp,
            n_indexer_top_k);
    exit(1);
}

static void validate_compress_ratio_metadata(const ds4_model *m) {
    const char *key = "deepseek4.attention.compress_ratios";
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)) {
        fprintf(stderr, "ds4: required int32/uint32 array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < DS4_N_LAYER) {
        ds4_die("deepseek4.attention.compress_ratios is shorter than the layer count");
    }

    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        uint32_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &got)) ds4_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            if (v < 0) ds4_die("metadata array contains a negative value");
            got = (uint32_t)v;
        }

        const uint32_t expected = ds4_expected_layer_compress_ratio(il);
        if (got != expected) {
            fprintf(stderr,
                    "ds4: unexpected DeepSeek4 compression ratio at layer %u for %s: got %u, expected %u\n",
                    il, DS4_MODEL_SHAPE_NAME, got, expected);
            exit(1);
        }
        g_ds4_compress_ratios[il] = got;
    }
}

static void config_expect_f32(const char *name, float got, float expected);

static void validate_swiglu_clamp_metadata(const ds4_model *m) {
    const char *key = "deepseek4.swiglu_clamp_exp";
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_FLOAT32 && arr.type != GGUF_VALUE_FLOAT64)) {
        fprintf(stderr, "ds4: required float array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < DS4_N_LAYER) {
        ds4_die("deepseek4.swiglu_clamp_exp is shorter than the layer count");
    }

    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t i = 0; i < DS4_N_LAYER; i++) {
        float got = 0.0f;
        if (arr.type == GGUF_VALUE_FLOAT32) {
            if (!cursor_read(&c, &got, sizeof(got))) ds4_die(c.error);
        } else {
            double v = 0.0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            got = (float)v;
        }
        config_expect_f32("swiglu_clamp_exp", got, DS4_SWIGLU_CLAMP_EXP);
    }
}

static void config_expect_u32(const char *name, uint32_t got, uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%u for %s, got %u\n",
            name, expected, DS4_MODEL_SHAPE_NAME, got);
    exit(1);
}

static void config_expect_f32(const char *name, float got, float expected) {
    const float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-6f) return;
    fprintf(stderr, "ds4: expected %s=%.9g for %s, got %.9g\n",
            name, (double)expected, DS4_MODEL_SHAPE_NAME, (double)got);
    exit(1);
}

static void config_expect_bool(const char *name, bool got, bool expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%s for %s, got %s\n",
            name, expected ? "true" : "false", DS4_MODEL_SHAPE_NAME, got ? "true" : "false");
    exit(1);
}

static void config_validate_fixed_shape(uint32_t n_layer) {
    config_expect_u32("block_count",                  n_layer,                 DS4_N_LAYER);
}

/* Validate metadata values that affect semantics: attention shape, HC count,
 * expert routing, RoPE scaling, compression ratios, and SwiGLU clamp. */
static void config_validate_deepseek4_model(const ds4_model *m) {
    const uint32_t n_layer = required_u32(m, "deepseek4.block_count");
    const uint32_t n_embd = required_u32(m, "deepseek4.embedding_length");
    const uint32_t n_vocab = required_u32(m, "deepseek4.vocab_size");
    const uint32_t n_head = required_u32(m, "deepseek4.attention.head_count");
    const uint32_t n_head_kv = required_u32(m, "deepseek4.attention.head_count_kv");
    const uint32_t n_head_dim = required_u32(m, "deepseek4.attention.key_length");
    const uint32_t n_value_dim = required_u32(m, "deepseek4.attention.value_length");
    const uint32_t n_rot = required_u32(m, "deepseek4.rope.dimension_count");
    const uint32_t n_lora_q = required_u32(m, "deepseek4.attention.q_lora_rank");
    const uint32_t n_lora_o = required_u32(m, "deepseek4.attention.output_lora_rank");
    const uint32_t n_out_group = required_u32(m, "deepseek4.attention.output_group_count");
    const uint32_t n_expert = required_u32(m, "deepseek4.expert_count");
    const uint32_t n_expert_used = required_u32(m, "deepseek4.expert_used_count");
    const uint32_t n_ff_exp = required_u32(m, "deepseek4.expert_feed_forward_length");
    const uint32_t n_expert_shared = required_u32(m, "deepseek4.expert_shared_count");
    const uint32_t n_hash_layer = required_u32(m, "deepseek4.hash_layer_count");
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups);
    model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used);
    const uint32_t n_swa = required_u32(m, "deepseek4.attention.sliding_window");
    const uint32_t n_indexer_head = required_u32(m, "deepseek4.attention.indexer.head_count");
    const uint32_t n_indexer_head_dim = required_u32(m, "deepseek4.attention.indexer.key_length");
    const uint32_t n_indexer_top_k = required_u32(m, "deepseek4.attention.indexer.top_k");
    const uint32_t n_hc = required_u32(m, "deepseek4.hyper_connection.count");
    const uint32_t n_hc_sinkhorn_iter = required_u32(m, "deepseek4.hyper_connection.sinkhorn_iterations");

    ds4_select_shape_from_metadata(n_layer,
                                   n_embd,
                                   n_vocab,
                                   n_head,
                                   n_head_kv,
                                   n_head_dim,
                                   n_value_dim,
                                   n_rot,
                                   n_lora_q,
                                   n_lora_o,
                                   n_out_group,
                                   n_expert,
                                   n_expert_used,
                                   n_ff_exp,
                                   n_expert_shared,
                                   n_hash_layer,
                                   n_swa,
                                   n_indexer_head,
                                   n_indexer_head_dim,
                                   n_indexer_top_k,
                                   n_hc,
                                   n_hc_sinkhorn_iter);

    config_expect_u32("embedding_length",            n_embd,         DS4_N_EMBD);
    config_expect_u32("vocab_size",                  n_vocab,        DS4_N_VOCAB);
    config_expect_u32("attention.head_count",        n_head,         DS4_N_HEAD);
    config_expect_u32("attention.key_length",        n_head_dim,     DS4_N_HEAD_DIM);
    config_expect_u32("attention.head_count_kv",     n_head_kv,      DS4_N_HEAD_KV);
    config_expect_u32("attention.value_length",      n_value_dim,    DS4_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count",        n_rot,          DS4_N_ROT);
    config_expect_u32("attention.output_group_count", n_out_group,    DS4_N_OUT_GROUP);
    config_expect_u32("attention.q_lora_rank",       n_lora_q,        DS4_N_LORA_Q);
    config_expect_u32("attention.output_lora_rank",  n_lora_o,        DS4_N_LORA_O);
    config_expect_u32("expert_count",               n_expert,        DS4_N_EXPERT);
    config_expect_u32("expert_used_count",          n_expert_used,   DS4_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length", n_ff_exp,        DS4_N_FF_EXP);
    config_expect_u32("expert_shared_count",         n_expert_shared, DS4_N_EXPERT_SHARED);
    config_expect_u32("hash_layer_count",            n_hash_layer,    DS4_N_HASH_LAYER);
    config_expect_u32("expert_group_count",         n_expert_groups, 0);
    config_expect_u32("expert_group_used_count",    n_group_used,    0);

    config_expect_u32("attention.sliding_window",     n_swa,                   DS4_N_SWA);
    config_expect_u32("attention.indexer.head_count", n_indexer_head,     DS4_N_INDEXER_HEAD);
    config_expect_u32("attention.indexer.key_length", n_indexer_head_dim, DS4_N_INDEXER_HEAD_DIM);
    config_expect_u32("attention.indexer.top_k",      n_indexer_top_k,    DS4_N_INDEXER_TOP_K);
    config_expect_u32("hyper_connection.count", n_hc, DS4_N_HC);
    config_expect_u32("hyper_connection.sinkhorn_iterations", n_hc_sinkhorn_iter, DS4_N_HC_SINKHORN_ITER);

    config_validate_fixed_shape(n_layer);
    validate_compress_ratio_metadata(m);

    validate_swiglu_clamp_metadata(m);

    uint64_t rope_orig_ctx = DS4_ROPE_ORIG_CTX;
    model_get_u64_compat(m, "deepseek4.rope.scaling.original_context_length", &rope_orig_ctx);
    if (rope_orig_ctx != DS4_ROPE_ORIG_CTX) {
        fprintf(stderr, "ds4: expected rope.scaling.original_context_length=%" PRIu64
                " for %s, got %" PRIu64 "\n",
                (uint64_t)DS4_ROPE_ORIG_CTX, DS4_MODEL_SHAPE_NAME, rope_orig_ctx);
        exit(1);
    }
    const float rope_freq_base = required_f32(m, "deepseek4.rope.freq_base");
    config_expect_f32("rope.freq_base", rope_freq_base, DS4_ROPE_FREQ_BASE);
    float rope_scale_factor = DS4_ROPE_SCALE_FACTOR;
    model_get_f32_compat(m, "deepseek4.rope.scaling.factor", &rope_scale_factor);
    config_expect_f32("rope.scaling.factor", rope_scale_factor, DS4_ROPE_SCALE_FACTOR);
    float rope_yarn_beta_fast = DS4_ROPE_YARN_BETA_FAST;
    model_get_f32_compat(m, "deepseek4.rope.scaling.yarn_beta_fast", &rope_yarn_beta_fast);
    config_expect_f32("rope.scaling.yarn_beta_fast", rope_yarn_beta_fast, DS4_ROPE_YARN_BETA_FAST);
    float rope_yarn_beta_slow = DS4_ROPE_YARN_BETA_SLOW;
    model_get_f32_compat(m, "deepseek4.rope.scaling.yarn_beta_slow", &rope_yarn_beta_slow);
    config_expect_f32("rope.scaling.yarn_beta_slow", rope_yarn_beta_slow, DS4_ROPE_YARN_BETA_SLOW);
    const float compress_rope_freq_base = required_f32(m, "deepseek4.attention.compress_rope_freq_base");
    config_expect_f32("attention.compress_rope_freq_base", compress_rope_freq_base, DS4_COMPRESS_ROPE_FREQ_BASE);
    const float expert_weight_scale = required_f32(m, "deepseek4.expert_weights_scale");
    config_expect_f32("expert_weights_scale", expert_weight_scale, DS4_EXPERT_WEIGHT_SCALE);
    const float rms_eps = required_f32(m, "deepseek4.attention.layer_norm_rms_epsilon");
    config_expect_f32("attention.layer_norm_rms_epsilon", rms_eps, DS4_RMS_EPS);
    const float hc_eps = required_f32(m, "deepseek4.hyper_connection.epsilon");
    config_expect_f32("hyper_connection.epsilon", hc_eps, DS4_HC_EPS);
    const bool expert_weight_norm = required_bool(m, "deepseek4.expert_weights_norm");
    config_expect_bool("expert_weights_norm", expert_weight_norm, true);
}

void config_validate_model(const ds4_model *m) {
    config_validate_deepseek4_model(m);
}

static void weights_bind_output(
        ds4_weights     *w,
        const ds4_model *m,
        bool             required,
        bool             optional) {
    if (required) {
        w->output_hc_base   = required_tensor(m, "output_hc_base.weight");
        w->output_hc_fn     = required_tensor(m, "output_hc_fn.weight");
        w->output_hc_scale  = required_tensor(m, "output_hc_scale.weight");
        w->output_norm      = required_tensor(m, "output_norm.weight");
        w->output           = required_tensor(m, "output.weight");
    } else if (optional) {
        w->output_hc_base   = model_find_tensor(m, "output_hc_base.weight");
        w->output_hc_fn     = model_find_tensor(m, "output_hc_fn.weight");
        w->output_hc_scale  = model_find_tensor(m, "output_hc_scale.weight");
        w->output_norm      = model_find_tensor(m, "output_norm.weight");
        w->output           = model_find_tensor(m, "output.weight");
    }

    if (optional &&
        weights_have_partial_output_head(w) &&
        !weights_have_output_head(w)) {
        ds4_die("partial output head in GGUF");
    }
}

static void weights_bind_layer(ds4_layer_weights *l, const ds4_model *m, uint32_t il) {
    const uint32_t compress_ratio = ds4_layer_compress_ratio(il);

    l->hc_attn_fn      = required_tensorf(m, "blk.%u.hc_attn_fn.weight", il);
    l->hc_attn_scale   = required_tensorf(m, "blk.%u.hc_attn_scale.weight", il);
    l->hc_attn_base    = required_tensorf(m, "blk.%u.hc_attn_base.weight", il);
    l->attn_norm       = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    l->attn_q_a        = required_tensorf(m, "blk.%u.attn_q_a.weight", il);
    l->attn_q_a_norm   = required_tensorf(m, "blk.%u.attn_q_a_norm.weight", il);
    l->attn_q_b        = required_tensorf(m, "blk.%u.attn_q_b.weight", il);
    l->attn_kv         = required_tensorf(m, "blk.%u.attn_kv.weight", il);
    l->attn_kv_a_norm  = required_tensorf(m, "blk.%u.attn_kv_a_norm.weight", il);
    l->attn_sinks      = required_tensorf(m, "blk.%u.attn_sinks.weight", il);
    l->attn_output_a   = required_tensorf(m, "blk.%u.attn_output_a.weight", il);
    l->attn_output_b   = required_tensorf(m, "blk.%u.attn_output_b.weight", il);
    if (compress_ratio != 0) {
        l->attn_compressor_ape  = required_tensorf(m, "blk.%u.attn_compressor_ape.weight", il);
        l->attn_compressor_kv   = required_tensorf(m, "blk.%u.attn_compressor_kv.weight", il);
        l->attn_compressor_gate = required_tensorf(m, "blk.%u.attn_compressor_gate.weight", il);
        l->attn_compressor_norm = required_tensorf(m, "blk.%u.attn_compressor_norm.weight", il);
    }
    if (compress_ratio == 4) {
        l->indexer_attn_q_b = required_tensorf(m, "blk.%u.indexer.attn_q_b.weight", il);
        l->indexer_proj     = required_tensorf(m, "blk.%u.indexer.proj.weight", il);
        l->indexer_compressor_ape  = required_tensorf(m, "blk.%u.indexer_compressor_ape.weight", il);
        l->indexer_compressor_kv   = required_tensorf(m, "blk.%u.indexer_compressor_kv.weight", il);
        l->indexer_compressor_gate = required_tensorf(m, "blk.%u.indexer_compressor_gate.weight", il);
        l->indexer_compressor_norm = required_tensorf(m, "blk.%u.indexer_compressor_norm.weight", il);
    }
    l->hc_ffn_fn       = required_tensorf(m, "blk.%u.hc_ffn_fn.weight", il);
    l->hc_ffn_scale    = required_tensorf(m, "blk.%u.hc_ffn_scale.weight", il);
    l->hc_ffn_base     = required_tensorf(m, "blk.%u.hc_ffn_base.weight", il);
    l->ffn_norm        = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
    l->ffn_gate_inp    = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
    l->ffn_exp_probs_b = tensor_by_namef(m, "blk.%u.exp_probs_b.bias", il);
    l->ffn_gate_exps   = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
    l->ffn_up_exps     = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
    l->ffn_down_exps   = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
    l->ffn_gate_shexp  = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
    l->ffn_up_shexp    = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
    l->ffn_down_shexp  = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);

    if (il < DS4_N_HASH_LAYER) {
        l->ffn_gate_tid2eid = required_tensorf(m, "blk.%u.ffn_gate_tid2eid.weight", il);
    }
}

/* Bind tensor names once into the fixed DS4 layer layout.  This is the point
 * where stringly GGUF metadata becomes direct model-specific pointers. */
void weights_bind(
        ds4_weights     *w,
        const ds4_model *m,
        bool             load_slice,
        uint32_t         load_layer_start,
        uint32_t         load_layer_end,
        bool             require_output,
        bool             optional_output) {
    memset(w, 0, sizeof(*w));

    uint32_t start = 0;
    uint32_t end = DS4_N_LAYER - 1u;
    bool require_token_embd = true;
    if (load_slice) {
        if (load_layer_start >= DS4_N_LAYER) ds4_die("invalid model load layer slice");
        start = load_layer_start;
        end = load_layer_end == UINT32_MAX ? DS4_N_LAYER - 1u : load_layer_end;
        if (end >= DS4_N_LAYER || end < start) ds4_die("invalid model load layer slice");
        require_token_embd = start == 0;
    } else {
        require_output = true;
        optional_output = false;
    }

    if (require_token_embd) {
        w->token_embd = required_tensor(m, "token_embd.weight");
    } else {
        w->token_embd = model_find_tensor(m, "token_embd.weight");
    }
    weights_bind_output(w, m, require_output, optional_output);

    for (uint32_t il = start; il <= end; il++) {
        weights_bind_layer(&w->layer[il], m, il);
    }
    weights_validate_layout(w, start, end, require_token_embd, require_output);
}
