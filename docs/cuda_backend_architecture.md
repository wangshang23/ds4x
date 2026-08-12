# CUDA backend architecture

DS4X keeps the engine-facing `ds4_gpu_*` C ABI stable while moving kernel
implementation toward ordinary, independently testable CUDA C++ translation
units. This document defines that boundary so source cleanup does not silently
change model semantics or DGX Spark performance.

The complete migration order, including the later DSpark and Codex adapter
phases, is defined in [`roadmap.md`](roadmap.md). This document covers only the
device/backend boundary.

## Layers

```text
engine C code
    -> ds4_gpu_* compatibility and resource wrappers
        -> measured backend dispatch
            -> cuBLAS / existing packed kernels
            -> standalone CUDA C++ operators
            -> CUTLASS/CuTe operators
```

The temporary compatibility layer owns model offsets, cache residency,
temporary-buffer allocation and user-visible diagnostics. A standalone
operator receives resolved device pointers, dimensions, one stream and an
explicit workspace. It does not know about GGUF, sessions, placement, DSpark
acceptance or Codex protocols.

As of the Phase 2 release, every model-visible target hot path enters this
operator boundary. Some adapters deliberately call the already validated
packed kernel modules under `backend/ops/`; migration means the engine no
longer depends on private source ordering or untyped argument lists, not that a
slower replacement kernel must be selected.

## Source ownership

- `backend/runtime/` owns process/device state, graph capture, memory and model
  cache lifetime.
- `backend/ops/` owns independently compiled linear, attention, indexer and MoE
  kernels; `backend/compat/` owns only the stable compatibility ABI.
- `backend/internal/backend_internal.cuh` contains private declarations,
  templates and force-inlined device helpers needed across translation units.
- `backends/` contains independent CUDA C++ implementations. Files expose a
  narrow launch ABI through `include/ds4x/` and have focused tests under
  `tests/ds4x_kernel/backends/`.
- `backends/operator_adapters.cu` is the typed bridge for packed cache,
  indexer, attention, routed MoE, hyper-connection and graph-capture paths.
- `quantization/mmq/` retains the audited llama.cpp-derived quantized kernels
  as explicit template-instantiation units with private `detail/*.cuh` headers,
  plus its vendor record.

## FP16 projection dispatch

The logical operation is:

```text
output[tokens, out] = activations[tokens, in] * weights[in, out]
```

GGUF stores the FP16 weight bytes as row-major `[out, in]`. That is the same
physical layout as the column-major `[in, out]` view used by cuBLAS and
CUTLASS. Activations are converted from FP32 to FP16 once before either GEMM;
accumulation and output remain FP32.

The default backend policy is empirical:

- small compressor, router and output projections remain on cuBLAS or their
  exact fallback kernels;
- `in=1024`, `out=8192`, `tokens>=128` uses CUTLASS, matching the indexer
  `q_b` shapes that win or tie on GB10;
- attention output-A uses a CuTe-shaped CUTLASS strided-batched GEMM only for
  the measured Flash shape `batch=8`, `in=4096`, `out=1024`, and
  `512<=tokens<=1024`; smaller and 4096-token chunks remain on cuBLAS;
- unsupported or failed CUTLASS launches fall back to cuBLAS;
- `DS4_CUDA_F16_BACKEND=cublas` disables CUTLASS, while `cutlass` is a research
  override that attempts it for every supported dense shape;
- `DS4_CUDA_ATTN_OUTPUT_A_BACKEND=cublas|cutlass` provides the corresponding
  attention output-A A/B override.

No runtime autotuning occurs in inference. It would add synchronization and
first-request latency, and a noisy first sample is not a safe production
policy. The checked-in threshold comes from the reproducible backend benchmark.

## Blackwell boundary

DGX Spark GB10 is `sm_121`. Its FP16 Tensor Core path is warp-level
`mma.sync`, not the SM100 data-center `tcgen05` path. CUTLASS 4.6.2 supports
SM120/SM121, but its SM120 CollectiveBuilder currently restricts the dense
non-block-scaled mainloop to F8/F6/F4 inputs. DS4X therefore uses:

- an SM80-tagged, forward-compatible CUTLASS FP16 TensorOp composition for
  `mma.sync`;
- native SM121/CuTe or CUTLASS SM120 features only when their input formats and
  hardware contracts actually match;
- cluster shape `1x1x1`, because GB10 does not expose the data-center
  multicast assumptions used by SM100 kernels.

## CSA exact top-k fusion

Batch-one CSA keeps the SM121 block-scaled MXFP4 MMA scorer and changes only
the score handoff at long context. A CTA scores 2,048 packed indexer rows,
converts each result to a 64-bit sortable key and retains its exact local
top-512. The key encodes the float total-order value in the high 32 bits and
`0xffffffff-index` in the low bits, preserving the existing lower-index tie
break. Four-way hierarchical merges then produce the global 512 indices.

The fused dispatch starts at `n_comp=65536`, corresponding to a 256K CSA
frontier. The threshold is empirical: 128K had no repeatable end-to-end gain,
while 50-sample alternating A/B runs improved median TPOT by 1.57%, 1.05% and
2.55% at 256K, 512K and 1M. All four full-logit comparisons were bit-identical.
`DS4_CUDA_NO_FUSED_INDEXER_TOPK=1` remains the deterministic fallback.

The CUTLASS/CuTe MXFP4 atom was also evaluated against the current native
SM120 block-scaled MMA sequence. Both compiled to 69 registers and 37,888
bytes of shared memory, with only 0.1%-0.3% microbenchmark variation, so the
native scorer remains selected. A CUTLASS Example 41 SWA experiment was also
rejected: batch-one changed F32-Q/output numerics, and the 128-token prefill
shape was about 30% slower than DS4X's token-tile HMMA kernel.

HCA online split-KV prototypes were not promoted. The pre-existing F32
prototype was roughly 4x slower across 1,024-8,192 compressed rows. A packed
16-head split prototype reached 1.32x isolated HCA speedup at 8,192 rows, but
its changed floating-point reduction order produced non-identical 1M model
logits and a different argmax. The exact-order packed variant was 3x slower.
The production HCA path therefore remains unchanged.

## Change gate

Every dispatch or kernel change must pass all of the following on DGX Spark:

1. focused numerical parity for the operator;
2. `make kernel-test` for packed cache, attention, indexer and MMQ coverage;
3. full-logit or task-level comparison for any model-visible path;
4. real prefill benchmarking when a batch kernel changes;
5. the 128K, 256K, 512K and 1M synthetic decode sweep;
6. fallback verification with the relevant backend override.

A backend is enabled by default only for shapes where measured end-to-end
performance does not regress. Faster isolated kernels are insufficient if they
add conversion, workspace, synchronization or graph-capture overhead.
