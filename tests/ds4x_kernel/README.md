# Standalone kernel tests

These tests link the CUDA kernel objects directly and do not depend on a CLI or
HTTP server.

- `cuda_long_context_smoke.c` checks packed cache, indexer and attention
  correctness at representative long-context shapes.
- `cuda_long_context_perf.c` provides reproducible kernel performance probes.
- `mmq/test_mmq_parity.cu` compares adapted quantized matrix kernels with CPU
  references.
- `qkvo_fp4_probe.cu` is the isolated QKVO FP4 feasibility experiment.

Run the independent suite with:

```bash
make kernel-test
```

Build performance binaries without running a full-model benchmark with:

```bash
make kernel-perf
```
