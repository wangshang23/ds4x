# Source layout

DS4X uses explicit ownership boundaries between the single-request frontend,
inference runtime, device operators, and packed checkpoint storage.

```text
src/
  apps/          CLI and benchmark entry points
  engine/        model loading, one session and graph execution
  ds4x_kernel/   CUDA-only GB10 kernels and their C ABI
  storage/       packed checkpoint persistence
  support/       embedded utility libraries
```

The host engine is split into ordinary, independently compiled C modules under
`engine/core`, `engine/model`, `engine/runtime`, and `engine/session`. Its
private cross-module ABI is centralized in `engine/internal`; public callers
only include `engine/include/ds4.h`.

Packed CUDA kernels are split into independent runtime, linear, attention,
indexer, MoE and compatibility translation units inside `ds4x_kernel/backend`.
Template and force-inlined device implementation stays in private `.cuh`
headers, following CUTLASS/CuTe's header-instantiation model. New standalone
CUDA work uses ordinary `.cu` translation units under `ds4x_kernel/backends/`
with narrow C launch interfaces; the FP16 projection module is the reference
structure.

Phase 2 routes every model-visible target operation through typed adapters.
The packed CUDA modules remain audited implementations behind that boundary,
not engine-owned APIs or alternate platform backends.

Vendored llama.cpp headers retain their upstream layout to keep source audits
and re-syncs practical.

The supported production path is CUDA-only and targets `sm_121a`.
The staged CUTLASS/CuTe, DSpark and Codex migration is documented in
`docs/roadmap.md`.
