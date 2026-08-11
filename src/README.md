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

Historical C/CUDA implementations use a thin aggregation translation unit plus
semantic `parts/*.inc` files. This preserves private/static symbol ordering
while those subsystems are migrated. New self-contained CUDA work uses ordinary
`.cu` translation units under `ds4x_kernel/backends/` with narrow C launch
interfaces; the FP16 projection module is the reference structure.

Phase 2 routes every model-visible target operation through typed adapters.
The remaining aggregation parts are audited implementations behind that
boundary, not engine-owned APIs or alternate platform backends.

Vendored llama.cpp headers retain their upstream layout to keep source audits
and re-syncs practical.

The supported production path is CUDA-only and targets `sm_121a`.
The staged CUTLASS/CuTe, DSpark and Codex migration is documented in
`docs/roadmap.md`.
