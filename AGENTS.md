# Repository Scope

DS4X is a CUDA-only DeepSeek-V4-Flash-Q2 inference runtime for the
NVIDIA GB10 in DGX Spark. It intentionally targets `sm_121a`; portability to
Metal, ROCm, older CUDA architectures, or other model families is out of scope.

# Performance Rules

- Persistent KV and indexer caches must remain physically packed.
- Do not add an expanded Q8-to-FP16 weight cache.
- Preserve the model's 512-row CSA top-k and exact attention semantics.
- Optimize steady-state decode after prefill; benchmark prefill separately.
- Every kernel change needs a numerical comparison and a 128K-1M benchmark.
- Keep Nsight markers and synthetic-frontier benchmarks reproducible.

# Validation

- `make smoke` validates CUDA cache/attention/indexer kernels.
- `make perf` builds long-context microbenchmarks and the full-model synthetic
  frontier benchmark.
- Full-model results use the repository's documented Q2 GGUF and must report
  model hash, driver, CUDA version, context, TPOT, throughput, and peak memory.
