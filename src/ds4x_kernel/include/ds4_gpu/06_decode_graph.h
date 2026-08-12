/* Decode-island CUDA graph capture. Design ported from the Entrpi/ds4 batched-serving
 * fork's per-layer decode graph capture.  The key identifies a captured
 * island: layer, island index, and the activation buffers whose addresses
 * the captured kernels bake in.  ds4_cuda.cu mirrors this struct
 * byte-for-byte (it does not include this header); keep both in sync. */
typedef struct ds4_decode_graph_key {
    uint32_t il;
    uint32_t island;    /* 0: layer top to pre-rope; 1: attn-out to layer end */
    uint32_t variant;
    uint32_t _pad;
    void    *cur_hc;
    void    *after_attn_hc;
    void    *after_ffn_hc;
    void    *attn_norm;
} ds4_decode_graph_key;

int  ds4_gpu_decode_graphs_supported(void);
/* 1: replayed (island already executed; skip encoding it)
 * 0: capturing (encode the island, then call _end)
 * -1: run eagerly */
int  ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key);
/* 0: capture committed and launched; -1: capture failed (entry retired;
 * the caller must re-encode the island eagerly -- no work was executed). */
int  ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key);
void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key);
void ds4_gpu_decode_graphs_invalidate(void);
