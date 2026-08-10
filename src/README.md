# Source layout

The runtime is organized by responsibility rather than by executable:

- `apps/` owns process entry points and user-facing argument handling.
- `core/` owns GGUF loading, model/session state, graph scheduling, and the
  DeepSeek-V4-Flash execution pipeline.
- `cuda/` owns the GB10 backend and all device-facing APIs. `cuda/mmq/`
  contains the adapted quantized matrix kernels and their standalone probes.
- `distributed/` owns tensor-parallel coordination and remote layer routing.
- `storage/` owns persistent KV payloads, SSD streaming, and expert hotlists.
- `support/` contains small embedded third-party support libraries.

Public headers are currently internal to this repository. Build targets use
the include search paths declared in the top-level `Makefile`; generated
objects and dependency files always go to `build/obj/<module>/`.

The supported production path remains CUDA-only and `sm_121a`-specific.
