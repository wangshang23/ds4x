# DS4X kernel layer

This directory owns all CUDA code and device-facing interfaces for the GB10
runtime.

```text
ds4x_kernel/
  backend/
    ds4x_kernel.cu   thin aggregation translation unit
    parts/           runtime, cache, attention, projection and MoE kernels
  include/           C ABI used by the engine and standalone tests
  quantization/mmq/  adapted llama.cpp quantized matrix kernels
  tables/            CUDA lookup tables
```

`backend/parts/` is ordered by dependency: runtime state and memory management
come first, followed by attention/indexer kernels, projections, MoE, GLM, and
the compatibility API. Persistent KV and indexer rows remain physically
packed; this layout must not be changed without a checkpoint ABI bump.

The kernel layer can be built and tested without linking an application:

```bash
make kernel-test
make kernel-perf
make qkvo-fp4-probe
```

Every production kernel change requires numerical parity and a 128K-1M
frontier run on DGX Spark.
