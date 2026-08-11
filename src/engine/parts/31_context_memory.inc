/* Packed single-GB10 context allocation estimate used by CLI diagnostics. */
ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int         ctx_size,
        uint32_t    prefill_chunk) {
    (void)backend;

    ds4_context_memory m = {0};
    const uint32_t ctx = ctx_size > 0 ? (uint32_t)ctx_size : 1u;
    m.prefill_cap = metal_graph_prefill_cap_for_prompt((int)ctx,
                                                        prefill_chunk);
    m.raw_cap = metal_graph_raw_cap_for_context((int)ctx, m.prefill_cap);

    uint32_t min_ratio = UINT32_MAX;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio != 0 && ratio < min_ratio) min_ratio = ratio;
    }
    if (min_ratio == UINT32_MAX) min_ratio = ctx;

    m.comp_cap = ctx / min_ratio + 2u;
    if (m.comp_cap < 2u) m.comp_cap = 2u;
    m.raw_bytes = (uint64_t)DS4_N_LAYER * m.raw_cap *
                  DS4_SPARK_KV_ROW_BYTES;

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio == 0) continue;
        const uint32_t rows = ctx / ratio + 2u;
        m.compressed_bytes += (uint64_t)rows * DS4_SPARK_KV_ROW_BYTES;
        if (ratio == 4u) {
            m.compressed_bytes += (uint64_t)rows *
                                  DS4_SPARK_INDEX_ROW_BYTES;
        }
    }

    uint64_t stage_rows = m.prefill_cap / min_ratio + 2u;
    if (stage_rows < 2u) stage_rows = 2u;
    m.scratch_bytes = 2ull * m.comp_cap * m.prefill_cap * sizeof(float) +
                      stage_rows *
                          (DS4_N_HEAD_DIM + DS4_N_INDEXER_HEAD_DIM) *
                          sizeof(float);
    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

ds4_context_memory ds4_context_memory_estimate(ds4_backend backend,
                                               int ctx_size) {
    return ds4_context_memory_estimate_with_prefill(backend, ctx_size, 0u);
}
