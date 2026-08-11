# Source layout

DS4X follows the same broad separation used by serving runtimes such as
SGLang: process frontends, an inference engine, device kernels, distributed
coordination, and storage are separate ownership domains.

```text
src/
  apps/          CLI, server, benchmark and evaluation entry points
  engine/        model loading, scheduling, sessions and graph execution
  ds4x_kernel/   CUDA-only GB10 kernels and their C ABI
  distributed/   tensor-parallel transport, routing and remote execution
  storage/       KV checkpoint persistence and model streaming
  support/       embedded utility libraries
```

Large C/CUDA implementations use a thin aggregation translation unit plus
semantic `parts/*.inc` files. This keeps private/static symbols in one
translation unit, so the existing ABI and compiler optimization boundaries do
not change, while each subsystem remains small enough to review independently.

The long streaming hotlists under `storage/` are generated data tables and are
the only intentionally large include files. Vendored llama.cpp headers retain
their upstream layout to keep source audits and re-syncs practical.

The supported production path is CUDA-only and targets `sm_121a`.
