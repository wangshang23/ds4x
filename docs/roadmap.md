# DS4X roadmap

DS4X is converging on one production path: DeepSeek-V4-Flash-Q2, one GB10,
and one active request. The implementation order is deliberately strict:

1. finish the single-device target runtime;
2. migrate the target runtime to modern CUDA C++ and CUTLASS/CuTe modules;
3. add the quantized DSpark draft path on the same session;
4. expose the session as a local Codex inference provider.

DSpark and Codex integration do not precede the target-runtime regression
gate. Otherwise a numerical or performance change cannot be attributed to the
target kernels, speculative scheduling, or the external adapter.

## Target architecture

```text
Codex adapter
    -> ds4x::Session (prefill, decode, verify, cancel, checkpoint)
        -> ds4x::Runtime (model, cache, stream, graph lifetime)
            -> operator dispatch
                -> CUTLASS/CuTe operators
                -> audited packed CUDA operators
                -> measured cuBLAS fallback
```

The intended source ownership is:

```text
include/ds4x/
  runtime/       public runtime, session, tensor and stream contracts
  ops/           typed operator arguments and launch interfaces
src/
  runtime/       model loading, session state and decode orchestration
  ops/
    linear/      Q8/FP16 and future FP4 projections
    indexer/     packed CSA score and top-k
    attention/   packed CSA/HCA prefill and decode
    moe/         router, expert dispatch and reduction
    cache/       packed KV/indexer storage and checkpoint ABI
  adapters/
    codex/       local provider boundary; no kernel or model ownership
tests/
  unit/          one operator and one invariant per test binary
  integration/   full-logit, checkpoint and end-to-end session tests
  performance/   reproducible GB10 regression benchmarks
```

The packed runtime now lives in independently compiled modules under
`src/ds4x_kernel/backend/`. New code must not recreate textual `.inc`
aggregation. Template or force-inlined device implementation belongs in a
private `.cuh`; stateful launch and resource ownership belongs in one `.cu`.

## Phase 0: single-device runtime

Status: complete for the supported GB10 target.

- Remove TP/EP, placement tiers, peer copies and multi-device configuration.
- Remove Metal, ROCm, CPU inference, unrelated model families and server
  batching.
- Keep multi-token prefill because it is required for usable prompt ingestion.
- Keep a single-request verification primitive for later DSpark integration.
- Retain only `ds4` and `ds4-bench` as executable frontends.

Completion requires a clean single-device link with no dormant TP or remote
transport symbols.

The release build initializes device 0 directly and exposes no TP/EP,
placement, peer-copy, SSD expert-streaming, Metal, ROCm, CPU-inference, server
batching, GLM/Pro, or legacy MTP surface. Multi-token prefill and the
single-request target verifier remain intentionally available.

## Phase 1: modern CUDA foundation

Status: complete at the engine/operator boundary.

- Replace process-global device, stream, cuBLAS and scratch state with one
  explicit `RuntimeContext` owned by the engine.
- Replace tier-aware tensor accessors with a move-only device tensor/view type.
- Pass `cudaStream_t` and explicit workspace to every operator launch.
- Use RAII for CUDA resources and keep the compatibility C ABI only at the
  engine boundary while the C engine is being retired.
- Build standalone CUDA C++ translation units with C++17 or newer; remove
  private symbol ordering as a source-layout requirement.

CuTe describes shapes, strides, tiling and data movement. CUTLASS supplies a
kernel only where the GB10 benchmark beats the existing path. cuBLAS remains a
valid measured fallback; replacing it is not itself a project goal.

The engine now owns an explicit `ds4x_runtime_context`, and typed operator
adapters receive that context, resolved tensors, dimensions, stream, and
workspace contracts. The stable `ds4_gpu_*` C ABI still encapsulates validated
legacy kernels and a small amount of process-level CUDA bookkeeping; this is a
compatibility boundary, not an alternate execution path.

## Phase 2: operator migration

Status: complete for every model-visible target hot path.

The migration order follows dependency and regression risk:

1. dense FP16/Q8 projection dispatch;
2. packed CSA indexer score and top-k;
3. packed CSA/HCA attention;
4. routed MoE and shared experts;
5. hyper-connection, output head and decode graph capture;
6. prefill orchestration and packed checkpoint I/O.

Each operator owns a typed argument structure, a support predicate, workspace
calculation, launch function, reference implementation, numerical test and
GB10 benchmark. Runtime autotuning is avoided in the request path; checked-in
dispatch policy comes from reproducible measurements.

The packed cache, CSA score/top-k, CSA/HCA attention, routed MoE,
hyper-connection/output, decode graph capture, prefill, verifier, and packed
checkpoint paths all cross typed adapters. FP16 projection and CUTLASS/CuTe
GEMM are standalone CUDA C++ translation units with focused tests. Audited
packed kernels remain behind the same typed boundary where replacing them did
not improve GB10 performance or numerical confidence.

## Phase 3: quantized DSpark

DSpark is a second model executed inside the same single-request session, not a
second serving runtime. The integration boundary consists of:

- target hidden-state capture;
- quantized draft stages and their cache;
- candidate generation;
- one target verification call for the candidate block;
- acceptance, rollback and cache commit;
- target-only fallback when the draft path is unavailable or rejected.

Draft weights and persistent state should use the selected packed low-precision
format. Temporary accumulation precision is chosen from numerical tests, not
from the storage format. Target-only and DSpark modes must share the same
session/checkpoint invariants.

## Phase 4: Codex adapter

Codex support belongs above the inference runtime. The core contract must be
usable without an HTTP server:

```text
load model -> create session -> prefill prompt -> stream decode tokens
           -> cancel/resume -> save or restore checkpoint
```

A thin local adapter can then translate the Codex provider protocol into this
contract. Token streaming, cancellation, context limits, deterministic error
reporting and session reuse are part of the adapter boundary. Generic
multi-user scheduling, distributed serving and unrelated API compatibility
remain out of scope unless the deployment target changes.

## Regression gate

Every phase must preserve:

- packed KV and indexer physical storage;
- the model's 512-row CSA selection and exact attention semantics;
- packed-checkpoint round-trip parity;
- full-logit or stable-greedy parity for every model-visible fast path;
- real prefill throughput at 4K and 16K;
- steady target-only decode at 128K, 256K, 512K and 1M;
- peak-memory reporting at the 1M frontier;
- a fallback A/B for each newly selected CUTLASS/CuTe path.

The current no-regression centers are documented in the top-level README.
Performance wins from DSpark are reported separately from target-only kernel
wins.
