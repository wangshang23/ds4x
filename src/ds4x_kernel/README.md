# DS4X kernel layer

This directory owns all CUDA code and device-facing interfaces for the GB10
runtime.

```text
ds4x_kernel/
  backend/
    ds4x_kernel.cu   thin aggregation translation unit
    parts/           runtime, cache, attention, projection and MoE kernels
  backends/          independent modern CUDA C++ operator implementations
  include/           C ABI used by the engine and standalone tests
  quantization/mmq/  adapted llama.cpp quantized matrix kernels
  tables/            CUDA lookup tables
```

`backend/parts/` is ordered by dependency: runtime state and memory management
come first, followed by attention/indexer kernels, projections, MoE, GLM, and
the compatibility API. Persistent KV and indexer rows remain physically
packed; this layout must not be changed without a checkpoint ABI bump.

New self-contained work belongs in `backends/` rather than adding another
large textual include. `fp16_projection.cu` is the migration example: kernels
and launch wrappers live in one ordinary CUDA C++ translation unit, while the
engine-facing wrapper retains allocation, model-cache and dispatch ownership.
`cutlass_fp16_gemm.cu` composes the pinned CUTLASS kernel and uses CuTe for its
logical problem shape. See `docs/cuda_backend_architecture.md` for the boundary
and performance-gating rules.

`operator_adapters.cu` completes the Phase 2 hot-path boundary for packed
cache, CSA indexer, CSA/HCA attention, routed MoE, output hyper-connections and
decode graph capture. These adapters preserve the proven packed kernels while
making their contracts explicit and independently testable.

The kernel layer can be built and tested without linking an application:

```bash
make kernel-test
make kernel-perf
make kernel-cutlass-bench
make qkvo-fp4-probe
```

Every production kernel change requires numerical parity and a 128K-1M
frontier run on DGX Spark.
