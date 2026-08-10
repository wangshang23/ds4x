# DS4X

DS4X is a narrow CUDA inference runtime for one deployment target:
DeepSeek-V4-Flash-Q2 on the NVIDIA GB10 in DGX Spark. It intentionally trades
portability for predictable long-context decode behavior on `sm_121a`.

The existing `ds4_*` C API, source names, and executable names are retained for
compatibility; DS4X is the repository and project name.

It is not a llama.cpp backend and it is not intended to run arbitrary GGUF
architectures. Metal, ROCm, older CUDA architectures, and other model families
are outside the supported build.

## Origin and attribution

DS4X is derived from [antirez/ds4](https://github.com/antirez/ds4), with the
optimization work developed against upstream commit
[`84cc882`](https://github.com/antirez/ds4/commit/84cc882352757baf628a1776badf7cc54d584e28).
DS4 provides the original GGUF loader, DeepSeek-V4-Flash execution pipeline,
CLI/server frontends, session and checkpoint machinery, and the `ds4_*` API
retained in this repository.

DS4X extracts that runtime into a CUDA-only, GB10-specific project and adds the
packed-cache and long-context kernel work described below. Selected quantized
matrix kernels are adapted from llama.cpp's `ggml-cuda` backend; the exact
upstream pin and local modifications are documented in
[`src/cuda/mmq/VENDOR.md`](src/cuda/mmq/VENDOR.md).

The DS4 and llama.cpp copyright notices remain in `LICENSE`. DS4X is an
independent community project and is not affiliated with DeepSeek or NVIDIA.

## What DS4X changes

| Area | DS4 CUDA path used as the baseline | DS4X |
|---|---|---|
| Deployment target | General multi-backend runtime | CUDA-only GB10 build, compiled specifically for `sm_121a` |
| Persistent attention history | Expanded F32 cache slots | Native 583-byte packed rows: E4M3 non-RoPE values, FP16 RoPE values, and UE8M0 scales |
| CSA indexer history | 128 F32 values per compressed row | 68-byte MXFP4 rows with four UE8M0 block scales |
| Batch-one CSA scoring | Scalar/direct path over expanded history | Direct packed-row SM121 block-scaled MMA scorer; no full-history re-encode per generated token |
| HCA long-context attention | Optimized path limited to 7,936 selected rows | Exact score-split attention without the old row ceiling, including the 1M frontier |
| Q8 weight handling | Optional expanded Q8-to-FP16/F32 cache | Expansion is disabled at compile time, preserving roughly 10 GiB for model and KV state |
| Quantized matrix multiply | Original DS4 dispatch | Self-contained adapters for selected llama.cpp Q8_0, Q2_K, IQ2_XXS, dense, and routed-MoE CUDA kernels |
| Persistent checkpoints | F32-cache payloads | Native packed-cache payloads with explicit ABI versioning and rejection of incompatible files |
| Validation focus | General runtime coverage | Exact packed-kernel parity plus reproducible 128K, 256K, 512K, and 1M decode-frontier tests |

The result is not intended as a replacement for upstream DS4. It is a focused
deployment fork for users who prefer a smaller support matrix in exchange for
lower cache memory, a usable 1M-token allocation, and stronger GB10
long-context performance.

## Optimization details

- Persistent attention KV uses the model's packed low-precision layout rather
  than F32 slots.
- Persistent CSA indexer keys use packed MXFP4 rather than F32 slots.
- The B1 indexer reads the 68-byte persistent rows directly and executes the
  SM121 block-scaled MXFP4 MMA path; it does not re-encode the full history for
  every generated token.
- HCA uses exact score-split attention without the old 7,936-row fast-path
  ceiling, including at a 1M-token frontier.
- Q8 weights remain Q8. The optional Q8-to-FP16/F32 expanded weight caches are
  compile-time disabled so roughly 10 GiB remains available for KV state.
- Disk checkpoints store the native packed rows. Payload ABI version 3 and
  distributed layer-payload ABI version 2 deliberately reject old F32-cache
  checkpoint files.
- The quantized matmul adapter calls the selected CUDA kernels directly through
  a small raw-argument ABI, without requiring the full ggml graph runtime.
- CUDA, core runtime, storage, distributed execution, frontends, and support
  libraries are separated into explicit source modules with one standalone
  Makefile build.

Some graph-driver functions retain historical `metal_graph_*` names from the
source runtime. The Makefile only builds the CUDA implementation.

## Supported configuration

- GPU: NVIDIA GB10, compute capability 12.1
- CUDA target: `compute_121a` / `sm_121a`
- Model family: DeepSeek-V4-Flash
- Routed weights: IQ2_XXS/Q2_K model used by the benchmark below
- Attention projection/shared-expert/output tensors: Q8 in the benchmark GGUF
- Maximum tested allocated context: 1,048,576 tokens

The build intentionally defines `DS4_CUDA_HAVE_MXF4=1` and
`DS4_CUDA_SPARK_ONLY=1`. Compilation should fail rather than silently emit a
portable fallback for another GPU generation.

## Build

Requirements are an aarch64 DGX Spark host, a CUDA toolkit that recognizes
`sm_121a`, GNU make, and a C99 compiler.

```bash
make -j4 spark smoke perf ds4_test
```

The main programs are `ds4`, `ds4-server`, `ds4-bench`, and `ds4-eval`.

## Project layout

```text
src/
  README.md      module ownership and source-layout notes
  apps/          CLI, HTTP server, benchmark and evaluation frontends
  core/          model loader, graph/session runtime and layer packing
  cuda/          GB10 backend, packed-cache kernels and GPU configuration
    mmq/         vendored/adapted quantized CUDA matrix kernels
  distributed/   tensor-parallel and distributed execution support
  storage/       KV checkpoint store, SSD streaming and expert hotlists
  support/       small embedded support libraries (linenoise and rax)
tests/            CUDA, checkpoint, synthetic-frontier and semantic tests
build/obj/        generated dependency files and intermediate objects
```

Only top-level build metadata, documentation and final executables remain in
the repository root. `make clean` removes generated binaries and `build/`.

Example target-only run:

```bash
./ds4 \
  -m /path/to/DeepSeek-V4-Flash-Q2.gguf \
  --cuda --ctx 1048576 --prefill-chunk 32 \
  --temp 0 -n 128 -p "Explain compressed sparse attention."
```

No environment variable is needed to disable the expanded Q8 cache. It cannot
be enabled in this Spark-only build.

## Packed cache ABI

Each attention row contains 512 elements:

```text
KV row = 448 E4M3 non-RoPE bytes
       +  64 FP16 RoPE values * 2 bytes
       +   7 UE8M0 block scales
       = 583 bytes
```

The seven scales correspond to seven 64-value non-RoPE blocks. The RoPE tail
stays FP16.

Each 128-element indexer row is:

```text
indexer row = 128 E2M1 values / 2 values per byte
            +   4 UE8M0 block scales
            = 64 + 4
            = 68 bytes
```

The four scales correspond to four 32-value blocks. The constants live in
`src/cuda/ds4_gpu.h`; pack/unpack and the SM121 scorer live in
`src/cuda/ds4_cuda.cu`.

For Flash, layers 0 and 1 have no compressed history, 21 layers use CSA ratio
4, and 20 layers use HCA ratio 128. With the benchmark's 32-token prefill chunk,
the raw ring is padded to 256 rows. The persistent KV allocation is therefore:

```text
KV_bytes(L)
  = 43 * 256 * 583
  + 21 * (floor(L / 4) + 2) * (583 + 68)
  + 20 * (floor(L / 128) + 2) * 583
```

At `L = 1,048,576`:

```text
raw rings                 =     6,417,664 bytes
compressed KV + indexer   = 3,679,340,006 bytes
total persistent KV       = 3,685,757,670 bytes
                          = 3.433 GiB
```

The observed startup plan was:

| Allocation | GiB |
|---|---:|
| Packed KV and indexer | 3.43 |
| Runtime buffers | 0.06 |
| Resident Q2/Q8 model | 80.76 |
| Total planned | 84.25 |

The small compressor frontier tensors remain F32. They are state, not the
history-sized persistent KV/indexer storage.

## Validation commands

Kernel regression:

```bash
make smoke
```

Packed checkpoint round trip:

```bash
make checkpoint-smoke DS4_TEST_MODEL=/path/to/model.gguf
```

Long-context synthetic decode sweep:

```bash
DS4_SYNTH_SKIP_WARM_WEIGHTS=1 \
  ./tests/synth_frontier_bench /path/to/model.gguf sweep 3 30
```

Packed reference-versus-optimized accuracy sweep:

```bash
./tests/synth_frontier_bench /path/to/model.gguf ab-sweep 1 3
```

Real text fact-recall regression:

```bash
make long-context DS4_TEST_MODEL=/path/to/model.gguf
```

## Test methodology

The measured system was:

| Item | Value |
|---|---|
| Host | DGX Spark, aarch64 |
| GPU | NVIDIA GB10, compute capability 12.1 |
| Driver | 580.82.09 |
| CUDA compiler | 13.0 (`cuda_13.0.r13.0`) |
| Nsight Systems | 2025.3.2.474 |
| GGUF size | 86,720,111,488 bytes (80.76 GiB) |
| GGUF SHA-256 | `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668` |

The model file was
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.

Performance numbers use target-only decode, the packed cache path, and no
Q8-to-FP16 weight cache. `synth_frontier_bench` allocates the requested context
and materializes a deterministic zero-cache frontier without performing a
million-token prefill. It then measures one exact decode step. The table
reports the median of 30 samples after 3 warmups. The optional startup pass
that touches all mapped model pages was disabled; this does not remove the
per-context decode warmups or the resident aligned CUDA artifacts.

This isolates steady-state decode cost versus cache length. It is not a prefill
benchmark and the synthetic zero history is not a language-quality workload.

The table below is the release-validation run from 2026-08-10: a fresh build,
three warmups, and the median of 30 timed decode steps at each frontier.

At 1M, `cudaMemGetInfo` reported 122,572 MiB total and 12,930 MiB free after
decode, or 107.07 GiB system-wide used. That is an observed unified-memory
machine total, not a process-only allocation: it includes the Linux page cache,
driver state, and any other system use. The runtime's own allocation plan is
the 84.25 GiB total shown earlier.

| Context | Median TPOT (ms) | Throughput (tok/s) |
|---:|---:|---:|
| 128K | 69.570 | 14.374 |
| 256K | 74.057 | 13.503 |
| 512K | 80.419 | 12.435 |
| 1M | 93.993 | 10.639 |

The release A/B run alternated the legacy F32-slot path and packed path from
the same process and reports the median of three timed steps per path:

| Context | F32-slot tok/s | Packed tok/s | Speedup |
|---:|---:|---:|---:|
| 128K | 12.744 | 14.166 | 1.112x |
| 256K | 10.910 | 13.169 | 1.207x |
| 512K | 8.759 | 12.225 | 1.396x |
| 1M | 6.234 | 10.517 | 1.687x |

## Accuracy results

The CUDA kernel smoke test passes all of the following:

```text
B1 indexer n_comp=8192: max_abs=0, topk_diff=0
packed attention: max_abs=0, rmse=0
large packed indexer: max_abs=0, topk_diff=0
```

At 128K, 256K, 512K, and 1M, the optimized SM121 indexer scorer and a scalar B1
oracle that decodes the same persistent 68-byte rows produced bit-identical
129,280-element full-model logits on the same initialized frontier
(`max_abs=0`, `rmse=0`, identical argmax). Packed attention is covered
separately by the exact kernel smoke comparison above.

The 8K packed checkpoint test produced a 42.614 MiB payload and bit-identical
post-restore decode logits (`max_abs=0`, `rmse=0`, identical argmax).

The real-text regression prefills a 30,474-token generated story, asks for the
embedded facts, and passes all assignments:

```text
ds4-test: long-context prefill 30474/30474
long-context: OK
```

The same release run passed the clean `sm_121a` build, CUDA kernel smoke suite,
packed checkpoint round trip, all four full-logit A/B frontiers, and the real
text recall test.

An old F32-slot build and this packed build need not produce identical logits:
the persistent formats are intentionally different. In particular, a
synthetic all-zero million-token history is numerically fragile because tiny
quantization changes can alter downstream MoE routing. The acceptance gates
are packed-format reference parity, independent pack/unpack tests, checkpoint
round-trip parity, and real-text recall.

## Nsight Systems

The 1M trace was captured around one profiled decode step. Profiling
raises TPOT to 103.376 ms. The largest GPU kernel groups were:

| Kernel group | GPU time (ms) |
|---|---:|
| Score-split finalize | 12.253 |
| SM121 indexer MXFP4 score | 7.817 |
| Score-split score tiles | 5.553 |
| Indexed attention | 2.823 |
| Packed KV unpack | 1.800 |

Raw benchmark and Nsight artifacts are intentionally not committed. Nsight
reports can contain user names, home directories, host names, internal network
addresses, environment variables, command lines, and unrelated process data.
The reproducible commands and sanitized aggregate results remain documented
here.

Packed KV unpack accounts for about 3% of captured GPU kernel time. At 1M the
remaining bottlenecks are the exact HCA score/reduction work and CSA
score/top-k, not persistent-cache expansion.

## Current boundaries

- Only the documented DeepSeek-V4-Flash layout is supported.
- The performance table is target-only and batch one.
- Packed prefill is functional. Multi-token compressed attention currently
  uses an exact per-token fallback while Q/KV projections and FFN work remain
  batched.
- The 1M result is a decode-frontier test, not a 1M-token prefill result.
- Checkpoint files from the old F32 persistent-cache ABI are intentionally
  incompatible and must be regenerated.
- The repository does not claim Apple or AMD compatibility; those backends are
  deliberately absent from the build.

## License

MIT. See `LICENSE`.
