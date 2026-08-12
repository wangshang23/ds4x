# DS4X kernel layer

This directory owns all CUDA code and device-facing interfaces for the GB10
runtime.

```text
ds4x_kernel/
  backend/
    runtime/         CUDA context, graph, tensor and model-cache ownership
    ops/             linear, attention, indexer and routed-MoE modules
    compat/          stable engine-facing compatibility ABI
    internal/        private types, declarations and shared device helpers
  backends/          independent modern CUDA C++ operator implementations
  include/           C ABI used by the engine and standalone tests
  quantization/mmq/  adapted llama.cpp quantized template instantiations
  tables/            private CUDA lookup-table headers
```

The packed backend is compiled as nine ordinary CUDA translation units. Their
shared private contract lives in `backend/internal/backend_internal.cuh`.
Small device helpers that must remain visible for inlining stay in that private
header; stateful host functions and kernels have one owning `.cu` module.

Cross-module kernels and force-inlined device helpers remain private
header-visible definitions. This avoids relocatable device code: CUDA 13
device LTO currently lowers the SM121a block-scaled MXFP4 instructions to an
SM121 target, which rejects those instructions. Host state and launch APIs are
still independently compiled, while each caller instantiates the exact local
kernel variant it launches.

Persistent KV and indexer rows remain physically packed; this layout must not
be changed without a checkpoint ABI bump.

New self-contained work belongs in `backends/` or an owning `backend/ops`
module. `fp16_projection.cu` is the migration example: kernels
and launch wrappers live in one ordinary CUDA C++ translation unit, while the
engine-facing wrapper retains allocation, model-cache and dispatch ownership.
`cutlass_fp16_gemm.cu` composes the pinned CUTLASS dense and strided-batched
kernels and uses CuTe for their logical problem shapes. Production dispatch
currently covers the indexer `q_b` projection and selected attention output-A
prefill chunks. See `docs/cuda_backend_architecture.md` for the boundary and
performance-gating rules.

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
