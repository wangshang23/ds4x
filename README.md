# DS4X

DS4X is a narrow CUDA inference runtime for one deployment target:
DeepSeek-V4-Flash-Q2 on the NVIDIA GB10 in DGX Spark. It intentionally trades
portability for predictable long-context decode behavior on `sm_121a`.

The existing `ds4_*` C API, source names, and executable names are retained for
compatibility; DS4X is the repository and project name.

It is not a llama.cpp backend and it is not intended to run arbitrary GGUF
architectures. Metal, ROCm, older CUDA architectures, and other model families
are outside the supported build.

## Performance

DeepSeek-V4-Flash-Q2, batch 1, target-only inference on one DGX Spark GB10
(128 GB unified memory, 273 GB/s memory bandwidth). The upstream baseline is
an unmodified `antirez/ds4@84cc882` build on the same machine and GGUF.

| Context | Upstream prefill latency (s) | DS4X prefill latency (s) | DS4X latency delta | Upstream steady decode (tok/s) | DS4X steady decode (tok/s) | Decode speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 5.452 | 4.910 | -9.9% | 14.68 | 15.72 | 1.071x |
| 8K | 9.403 | 9.465 | +0.7% | 14.33 | 15.59 | 1.088x |
| 16K | 18.731 | 18.751 | +0.1% | 14.26 | 15.45 | 1.084x |
| 32K | 37.430 | 37.423 | 0.0% | 13.70 | 15.09 | 1.102x |
| 64K | 76.019 | 78.459 | +3.2% | 13.04 | 14.80 | 1.135x |
| 128K | 159.041 | 163.564 | +2.8% | 12.00 | 14.12 | 1.177x |

Each prefill row is a separate process with a real full-context prefill. DS4X
uses the same selective Q8-weight-to-FP16 cuBLAS prefill path as upstream while
keeping persistent KV/indexer history packed. The decode columns retain the
established 31-token steady-state measurements; this change does not alter the
B1 decode path. See [Benchmark details](#benchmark-details) for the exact
prompt and methodology.

Long-context target-only decode uses an actually allocated packed KV/indexer
cache at each frontier. These measurements seed deterministic cache contents
instead of performing a 128K-1M prompt prefill, then report the median of 30
decode steps after 3 warmups:

| Context | Upstream TPOT (ms) | DS4X TPOT (ms) | Upstream (tok/s) | DS4X (tok/s) | Decode speedup |
|---:|---:|---:|---:|---:|---:|
| 128K | 81.248 | 70.918 | 12.308 | 14.101 | 1.146x |
| 256K | 92.640 | 75.092 | 10.794 | 13.317 | 1.234x |
| 512K | 115.246 | 81.434 | 8.677 | 12.280 | 1.415x |
| 1M | 363.286 | 94.206 | 2.753 | 10.615 | 3.856x |

The 1M row allocates the full 3.43 GiB packed persistent cache. It validates
decode at a million-token frontier; it is not a million-token prefill result.

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
[`src/ds4x_kernel/quantization/mmq/VENDOR.md`](src/ds4x_kernel/quantization/mmq/VENDOR.md).

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
| Q8 weight handling | Selective Q8-to-FP16 prefill cache | Same selective FP16/cuBLAS prefill path; no activation-Q8 exact-MMA path when the cache is available |
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
- Checkpoint weights remain Q8. Prefill lazily creates the same selective FP16
  weight mirrors as upstream so activations feed cuBLAS/HMMA directly instead
  of being requantized for the exact INT8 Tensor Core path.
- Disk checkpoints store the native packed rows. Payload ABI version 3 and
  distributed layer-payload ABI version 2 deliberately reject old F32-cache
  checkpoint files.
- The quantized matmul adapter calls the selected CUDA kernels directly through
  a small raw-argument ABI, without requiring the full ggml graph runtime.
- CUDA kernels, inference engine, storage, distributed execution, frontends,
  and support libraries are separated into explicit source modules. Large
  translation units are thin aggregators over subsystem-sized parts.

Some graph-driver functions retain historical `metal_graph_*` names from the
source runtime. The Makefile only builds the CUDA implementation.

## Supported configuration

- GPU: NVIDIA GB10, compute capability 12.1
- CUDA target: `compute_121a` / `sm_121a`
- Model family: DeepSeek-V4-Flash
- Benchmark checkpoint: [`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`](https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf)
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

Focused design studies:

- [QKVO FP4 and quantized DSpark feasibility](docs/qkvo_fp4_dspark_feasibility.md)

## Project layout

```text
src/
  README.md       module ownership and source-layout notes
  apps/           CLI, HTTP server, benchmark and evaluation frontends
  engine/         model loading, graph/session runtime and layer packing
  ds4x_kernel/    GB10 CUDA backend and device-facing API
    backend/      packed cache, attention, projection and MoE kernels
    quantization/ adapted quantized CUDA matrix kernels
    include/      kernel ABI consumed by the inference engine
  distributed/   tensor-parallel and distributed execution support
  storage/       KV checkpoint store, SSD streaming and expert hotlists
  support/       small embedded support libraries (linenoise and rax)
tests/
  ds4x_kernel/    standalone kernel parity, smoke and performance tests
  engine/         engine and protocol semantic tests
  integration/    packed-checkpoint and full-model frontier tests
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
`src/ds4x_kernel/include/ds4_gpu.h`; pack/unpack and the SM121 scorer live
under `src/ds4x_kernel/backend/parts/`.

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

The startup plan excludes lazily created selective Q8-to-FP16 prefill mirrors.
Short-context prefill can populate roughly 10 GiB of mirrors, matching the
upstream DS4 performance path. The cache has a free-memory reserve and may fall
back to quantized weights when a long-context allocation leaves insufficient
headroom.

The small compressor frontier tensors remain F32. They are state, not the
history-sized persistent KV/indexer storage.

## Validation commands

Kernel regression:

```bash
make kernel-test
```

Packed checkpoint round trip:

```bash
make checkpoint-smoke DS4_TEST_MODEL=/path/to/model.gguf
```

Long-context synthetic decode sweep:

```bash
DS4_SYNTH_SKIP_WARM_WEIGHTS=1 \
  ./tests/integration/synth_frontier_bench /path/to/model.gguf sweep 3 30
```

Packed reference-versus-optimized accuracy sweep:

```bash
./tests/integration/synth_frontier_bench /path/to/model.gguf ab-sweep 1 3
```

Real text fact-recall regression:

```bash
make long-context DS4_TEST_MODEL=/path/to/model.gguf
```

## Benchmark details

### Refactor regression

The responsibility-based source split was rebuilt and checked on DGX Spark on
2026-08-11. `make kernel-test` passed the packed-cache/attention/indexer smoke
suite and every MMQ parity case; `./ds4_test --server` and all four application
help paths passed. The packed checkpoint round trip at 8K reported
`max_abs=0`, `rmse=0`, and identical argmax logits.

The latest synthetic frontier validation, rerun after aligning the prefill
weight path with upstream, used three warmups and 30 timed steps. It exercises
the same packed target-only decode path as the comparison table.

| Context | Median TPOT (ms) | Throughput (tok/s) |
|---:|---:|---:|
| 128K | 70.918 | 14.101 |
| 256K | 75.092 | 13.317 |
| 512K | 81.434 | 12.280 |
| 1M | 94.206 | 10.615 |

### Test methodology

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

DS4X synthetic performance numbers use target-only decode and the packed cache
path. `synth_frontier_bench` does not invoke prefill, so it does not populate
the selective Q8-to-FP16 prefill cache. It allocates the requested context and
materializes a deterministic zero-cache frontier before measuring one exact
decode step.
The table reports the median of 30 samples after 3 warmups. The optional
startup pass that touches all mapped model pages was disabled; this does not
remove the per-context decode warmups or the resident aligned CUDA artifacts.

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

The release A/B run alternated two scorers over the same persistent 68-byte
packed indexer rows: a transparent scalar oracle that decodes each row to F32,
and the production SM121 block-scaled MXFP4 MMA scorer. It reports the median
of three timed steps per path. This is an indexer-kernel comparison, not an
upstream DS4 comparison.

| Context | Packed scalar oracle (tok/s) | SM121 MXFP4 scorer (tok/s) | Speedup |
|---:|---:|---:|---:|
| 128K | 12.744 | 14.166 | 1.112x |
| 256K | 10.910 | 13.169 | 1.207x |
| 512K | 8.759 | 12.225 | 1.396x |
| 1M | 6.234 | 10.517 | 1.687x |

### Pristine upstream DS4 comparison

The 2026-08-11 comparison below uses an unmodified checkout of
[`antirez/ds4@84cc882`](https://github.com/antirez/ds4/commit/84cc882352757baf628a1776badf7cc54d584e28)
and the current DS4X build on the same DGX Spark. Both use their native Spark
target, the same Q2/Q8 GGUF, `--prefill-chunk 4096`, and 32 greedy target-only
generation steps. Each row is a separate process that actually prefills the
full context; model loading and aligned-weight construction are outside the
prefill timer. Each prefill latency is one complete measured pass. `steady
decode` excludes the first generated token and averages the remaining 31
tokens. The resulting 4K-128K values are summarized in the top-level
Performance table.

The preceding packed-online implementation was 23.8%-55.6% slower in prefill.
Direct packed-to-FP16 token-tile mirrors removed the attention-history
overhead. Restoring upstream's selective Q8-weight-to-FP16 cuBLAS path then
removed the remaining activation requantization and exact INT8-MMA overhead.
The resulting 8K-32K prefill latency is within 0.7% of upstream while persistent
KV/indexer history remains packed.

The raw prompt is a deterministic 6,000,000-byte repetition of:

```text
DS4X reproducible prefill benchmark. The quick brown fox jumps over the lazy dog. Compressed sparse attention and mixture of experts are measured on NVIDIA GB10.
```

Its SHA-256 is
`ee9ef79daa0f0adbe3972bebe6a404c19a65c829f19e1c34d84fb6f12c4ff645`.
Prefill throughput is prompt-dependent because token routing changes which MoE
experts are touched, so comparisons against another prompt fixture are not
directly interchangeable.

### Long-context synthetic decode

Actual prefill beyond 128K is intentionally not used for this comparison. To
isolate decode scaling, both runtimes instead initialize a deterministic zero
cache at the requested frontier, perform three warmups, and report the median
of 30 exact target-only decode steps. The upstream measurement adds only a
test-only frontier-seeding hook; its production `ds4_cuda.cu` remains
byte-identical to `84cc882`.

| Context | Upstream TPOT (ms) | DS4X TPOT (ms) | Upstream (tok/s) | DS4X (tok/s) | Decode speedup |
|---:|---:|---:|---:|---:|---:|
| 128K | 81.248 | 70.918 | 12.308 | 14.101 | 1.146x |
| 256K | 92.640 | 75.092 | 10.794 | 13.317 | 1.234x |
| 512K | 115.246 | 81.434 | 8.677 | 12.280 | 1.415x |
| 1M | 363.286 | 94.206 | 2.753 | 10.615 | 3.856x |

At 128K, the synthetic results are within 2.6% of upstream's real-prefill
decode and within 1.0% of DS4X's real-prefill decode. At 1M, upstream allocates
13.46 GiB of F32 KV through managed memory and shows substantial paging
variance (236.7-540.4 ms across the timed steps); DS4X uses 3.43 GiB of packed
KV and remains on the stable packed decode path.

Raw benchmark logs and CSV files remain outside the public repository.

### Packed prefill implementation

Packed prefill keeps batch score/top-k and multi-token attention active on one
DGX Spark GPU. The default path decodes visible packed raw and compressed KV
directly into transient FP16 mirrors and feeds the existing 4-token x 8-head
HMMA token-tile kernel. It does not materialize transient F32 history.
Persistent KV and checkpoint files remain in the 583-byte KV and 68-byte
indexer formats.

The first compressed row is not available until token 3 in CSA or token 127 in
HCA. Those short zero-prefix spans continue to read the current batch's F32 KV
directly; round-tripping them through the persistent packed cache changed early
attention outputs and could be amplified by downstream MoE routing. The
remaining rows use the packed multi-token dense or indexed path.

Three modes are available:

- default: direct packed-to-FP16 token-tile HMMA attention;
- `DS4_CUDA_SPARK_PREFILL_EXACT=1`: 32-row grouped score/finalize kernels that
  use the reusable F32 fallback and are bit-identical to the promoted exact
  decode reduction order;
- `DS4_CUDA_SPARK_PREFILL_REFERENCE=1`: the old per-token unpack and attention
  path, retained as the slow A/B oracle.

At 8K on the repeated-text fixture, the direct-FP16 default path reached 710.11
tok/s. The unchanged grouped-exact and old reference measurements were 283.03
and 172.88 tok/s respectively. The default is 4.11x faster than the old packed
reference path.

## Accuracy results

The CUDA kernel smoke test passes all of the following:

```text
B1 indexer n_comp=8192: max_abs=0, topk_diff=0
packed attention: max_abs=0, rmse=0
packed batch exact: max_abs=0, rmse=0
large packed indexer: max_abs=0, topk_diff=0
```

The grouped exact prefill mode matches the old packed reference at 128 and
8,192 tokens over the full 129,280-element vocabulary (`max_abs=0`, `rmse=0`,
identical argmax). The default online mode deliberately changes floating-point
reduction order. On the adversarial 8K repeated-token fixture, its full logits
had `max_abs=2.62764` and `rmse=0.509800` versus the bit-exact reference, while
the argmax and all 32 subsequent greedy tokens remained identical. A layer-2
attention dump before MoE amplification measured `max_abs=4.52995e-6` and
`rmse=1.37528e-7`.

At 128K, 256K, 512K, and 1M, the optimized SM121 indexer scorer and a scalar B1
oracle that decodes the same persistent 68-byte rows produced bit-identical
129,280-element full-model logits on the same initialized frontier
(`max_abs=0`, `rmse=0`, identical argmax). Packed attention is covered
separately by the exact kernel smoke comparison above.

The 8K packed checkpoint test produced a 42.614 MiB payload and bit-identical
post-restore decode logits (`max_abs=0`, `rmse=0`, identical argmax).

The real-text regression prefills a 30,474-token generated story, asks for the
embedded facts, and passes all assignments on the default fast path:

```text
ds4-test: long-context prefill 30474/30474
long-context: OK
```

The same release run passed the clean `sm_121a` build, CUDA kernel smoke suite,
packed checkpoint round trip, all four decode full-logit A/B frontiers, the
8K exact-prefill full-logit A/B, and the default-path real-text recall test.

An old F32-slot build and this packed build need not produce identical logits:
the persistent formats are intentionally different. In particular, a
synthetic all-zero million-token history is numerically fragile because tiny
quantization changes can alter downstream MoE routing. Exact-mode gates are
packed-format reference parity and independent pack/unpack tests. Default
fast-mode gates are stable greedy output, checkpoint round-trip parity, and
real-text recall; use exact mode when bit-reproducible prefill logits are
required.

## Nsight Systems

The latest 16K trace uses the same model, prompt, 4,096-token chunk, and zero
generated tokens. Nsight measured 853.20 tok/s; the unprofiled run reached
873.75 tok/s versus upstream's 874.74 tok/s. All 170 eligible batch attention
calls use token-tile HMMA. The old packed-to-F32 unpack kernel, activation
`quantize_q8_0_f32`, and exact INT8 `matmul_q8_0_mma_exact` kernels are absent:

| Kernel group | Instances | GPU time |
|---|---:|---:|
| Token-tile HMMA attention | 170 | 2.072 s |
| Packed raw KV to FP16 mirror | 170 | 0.0057 s |
| Packed compressed KV to FP16 mirror | 164 | 0.0021 s |
| FP16 attention-output cuBLAS/HMMA | 172 | 1.024 s |
| Activation F32-to-Q8 plus exact INT8 MMA | 0 | 0 s |
| Packed KV to F32 history | 0 | 0 s |

The preceding online implementation spent 6.213 s in indexed plus dense
attention at 16K. The direct-FP16 HMMA path reduces that attention time by
about 3x. Reports and generated SQLite databases remain outside the repository.

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
- Packed multi-token prefill is enabled only for one CUDA GPU. Multi-GPU
  placement retains the old fallback until packed row-table ownership and
  communication are implemented.
- Default prefill prioritizes throughput and is not bit-identical to the exact
  reduction order. Set `DS4_CUDA_SPARK_PREFILL_EXACT=1` for grouped bit-exact
  attention or `DS4_CUDA_SPARK_PREFILL_REFERENCE=1` for the old oracle.
- The 1M result is a decode-frontier test, not a 1M-token prefill result.
- Checkpoint files from the old F32 persistent-cache ABI are intentionally
  incompatible and must be regenerated.
- The repository does not claim Apple or AMD compatibility; those backends are
  deliberately absent from the build.

## License

MIT. See `LICENSE`.
