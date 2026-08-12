# Inference engine

The engine owns host-side model and request execution. It does not define CUDA
kernels.

```text
engine/
  core/          platform and process-level runtime support
  include/       public `ds4_*` C API
  internal/      private types, declarations and hot-path inline accessors
  model/         GGUF loading, validation, tokenizer and model memory policy
  runtime/       graph setup, prefill, decode, verification and DSpark runtime
  session/       engine/session lifecycle, generation, checkpoint and batching
```

Every `.c` file is an independent translation unit and produces its own object
file. Shared private declarations live in `internal/engine_internal.h`; they
are not part of the installed API. Small graph accessors remain `static inline`
there so splitting the former aggregation unit does not add calls to the decode
hot path.

Modules are grouped by ownership rather than source-order dependencies. New
engine code should keep private helpers in its owning `.c` file and expose only
the narrow declarations needed by another engine module through the internal
header.

Device work is accessed only through headers from `src/ds4x_kernel/include`.
New host scheduling logic belongs here; new CUDA implementation belongs in
`src/ds4x_kernel`.
