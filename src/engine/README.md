# Inference engine

The engine owns host-side model and request execution. It does not define CUDA
kernels.

```text
engine/
  ds4_engine.c   thin aggregation translation unit
  include/       public `ds4_*` C API
  config/        GPU/runtime option parsing shared by applications
  model/         layer packing and model-layout helpers
  parts/         GGUF, CPU reference, graph, prefill, decode, DSpark and batch
                 execution subsystems
```

The ordered `parts/*.inc` files preserve the original single-translation-unit
static symbol visibility. They are grouped by lifecycle: model validation and
reference operations, graph construction, decode/prefill execution, engine and
session state, then batching and distributed integration.

Device work is accessed only through headers from `src/ds4x_kernel/include`.
New host scheduling logic belongs here; new CUDA implementation belongs in
`src/ds4x_kernel`.
