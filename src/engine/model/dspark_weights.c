#include "engine_internal.h"

/* Dspark Weights module. */
static ds4_tensor *dspark_bind_tensor(
        ds4_dspark_weights *dw,
        const ds4_model *m,
        uint32_t stage,
        const char *suffix,
        bool required) {
    ds4_tensor *tensor = tensor_by_mtp_stage_suffix(m, stage, suffix);
    if (tensor) {
        dw->present_tensors++;
    } else if (required) {
        dw->missing_tensors++;
    }
    return tensor;
}

static void dspark_bind_block(
        ds4_dspark_weights      *dw,
        ds4_layer_weights       *l,
        const ds4_model         *m,
        uint32_t                 stage) {
    l->hc_attn_fn      = dspark_bind_tensor(dw, m, stage, "hc_attn_fn.weight", true);
    l->hc_attn_scale   = dspark_bind_tensor(dw, m, stage, "hc_attn_scale.weight", true);
    l->hc_attn_base    = dspark_bind_tensor(dw, m, stage, "hc_attn_base.weight", true);
    l->attn_norm       = dspark_bind_tensor(dw, m, stage, "attn_norm.weight", true);
    l->attn_q_a        = dspark_bind_tensor(dw, m, stage, "attn_q_a.weight", true);
    l->attn_q_a_norm   = dspark_bind_tensor(dw, m, stage, "attn_q_a_norm.weight", true);
    l->attn_q_b        = dspark_bind_tensor(dw, m, stage, "attn_q_b.weight", true);
    l->attn_kv         = dspark_bind_tensor(dw, m, stage, "attn_kv.weight", true);
    l->attn_kv_a_norm  = dspark_bind_tensor(dw, m, stage, "attn_kv_a_norm.weight", true);
    l->attn_sinks      = dspark_bind_tensor(dw, m, stage, "attn_sinks.weight", true);
    l->attn_output_a   = dspark_bind_tensor(dw, m, stage, "attn_output_a.weight", true);
    l->attn_output_b   = dspark_bind_tensor(dw, m, stage, "attn_output_b.weight", true);
    l->hc_ffn_fn       = dspark_bind_tensor(dw, m, stage, "hc_ffn_fn.weight", true);
    l->hc_ffn_scale    = dspark_bind_tensor(dw, m, stage, "hc_ffn_scale.weight", true);
    l->hc_ffn_base     = dspark_bind_tensor(dw, m, stage, "hc_ffn_base.weight", true);
    l->ffn_norm        = dspark_bind_tensor(dw, m, stage, "ffn_norm.weight", true);
    l->ffn_gate_inp    = dspark_bind_tensor(dw, m, stage, "ffn_gate_inp.weight", true);
    l->ffn_exp_probs_b = dspark_bind_tensor(dw, m, stage, "exp_probs_b.bias", true);
    l->ffn_gate_exps   = dspark_bind_tensor(dw, m, stage, "ffn_gate_exps.weight", true);
    l->ffn_up_exps     = dspark_bind_tensor(dw, m, stage, "ffn_up_exps.weight", true);
    l->ffn_down_exps   = dspark_bind_tensor(dw, m, stage, "ffn_down_exps.weight", true);
    l->ffn_gate_shexp  = dspark_bind_tensor(dw, m, stage, "ffn_gate_shexp.weight", true);
    l->ffn_up_shexp    = dspark_bind_tensor(dw, m, stage, "ffn_up_shexp.weight", true);
    l->ffn_down_shexp  = dspark_bind_tensor(dw, m, stage, "ffn_down_shexp.weight", true);
}

void dspark_weights_bind_optional(
        ds4_dspark_weights        *dw,
        const ds4_model           *m,
        const ds4_dspark_summary  *summary) {
    memset(dw, 0, sizeof(*dw));
    if (!m || !summary) return;

    dw->n_stages = summary->stages < DS4_DSPARK_MAX_STAGES ?
                   summary->stages : DS4_DSPARK_MAX_STAGES;
    dw->block_size = summary->block_size;
    dw->markov_rank = summary->markov_rank;
    dw->noise_token_id = summary->noise_token_id;
    dw->target_layer_count = summary->target_layer_count;
    dw->has_block_size = summary->has_block_size;
    dw->has_markov_rank = summary->has_markov_rank;
    dw->has_noise_token_id = summary->has_noise_token_id;
    dw->has_target_layers = summary->has_target_layers;
    memcpy(dw->target_layers,
           summary->target_layers,
           (size_t)dw->target_layer_count * sizeof(dw->target_layers[0]));
    if (summary->stages > DS4_DSPARK_MAX_STAGES) dw->missing_tensors++;

    for (uint32_t stage = 0; stage < dw->n_stages; stage++) {
        ds4_dspark_stage_weights *sw = &dw->stage[stage];
        dspark_bind_block(dw, &sw->block, m, stage);
        if (stage == 0) {
            sw->main_proj = dspark_bind_tensor(dw, m, stage, "main_proj.weight", true);
            sw->main_norm = dspark_bind_tensor(dw, m, stage, "main_norm.weight", true);
        }
    }

    if (dw->n_stages != 0) {
        const uint32_t final_stage = dw->n_stages - 1u;
        ds4_dspark_stage_weights *sw = &dw->stage[final_stage];
        sw->norm = dspark_bind_tensor(dw, m, final_stage, "norm.weight", true);
        sw->hc_head_base =
            dspark_bind_tensor(dw, m, final_stage, "hc_head_base.weight", true);
        sw->hc_head_fn =
            dspark_bind_tensor(dw, m, final_stage, "hc_head_fn.weight", true);
        sw->hc_head_scale =
            dspark_bind_tensor(dw, m, final_stage, "hc_head_scale.weight", true);
        sw->markov_w1 =
            dspark_bind_tensor(dw, m, final_stage, "markov_head.markov_w1.weight", true);
        sw->markov_w2 =
            dspark_bind_tensor(dw, m, final_stage, "markov_head.markov_w2.weight", true);
        sw->confidence_proj =
            dspark_bind_tensor(dw, m, final_stage, "confidence_head.proj.weight", true);
    }

    dspark_weights_validate_layout(dw);
}

void weights_free(ds4_weights *w) {
    memset(w, 0, sizeof(*w));
}

/* Load one token embedding row and expand it to float activations. */
