# DS4X

DS4X is a narrow CUDA inference runtime for one deployment target:
DeepSeek-V4-Flash-Q2 on the NVIDIA GB10 in DGX Spark. It intentionally trades
portability for predictable long-context decode behavior on `sm_121a`.

The existing `ds4_*` C API, source names, and executable names are retained for
compatibility during the CUDA C++ migration; DS4X is the repository and
project name.

It is not a llama.cpp backend and it is not intended to run arbitrary GGUF
architectures. Metal, ROCm, older CUDA architectures, and other model families
are outside the supported build.

## Performance

DeepSeek-V4-Flash-Q2, batch 1, target-only inference on one DGX Spark GB10
(128 GB unified memory, 273 GB/s memory bandwidth). The upstream baseline is
an unmodified `antirez/ds4@84cc882` build on the same machine and GGUF.

| Context | Upstream prefill latency (s) | DS4X prefill latency (s) | DS4X latency delta | Upstream steady decode (tok/s) | DS4X steady decode (tok/s) | Decode speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 4K | 5.452 | 4.928 | -9.6% | 14.68 | 15.79 | 1.076x |
| 16K | 18.731 | 18.984 | +1.3% | 14.26 | 15.36 | 1.077x |

Each row is a separate process with a real full-context prefill and 32 greedy
decode steps. The decode rate excludes the first token and averages the final
31. DS4X uses the same selective Q8-weight-to-FP16 prefill strategy as
upstream while keeping persistent KV/indexer history packed. The reported DS4X
rows are the warmed Phase 2 release runs; immediate same-machine checks of the
pre-refactor DS4X snapshot measured 831.82 and 847.63 prefill tok/s at 4K and
16K, versus 831.13 and 863.05 tok/s for the release build. Single-pass prefill
is sensitive to page-cache and thermal state, so paired runs matter more than
an isolated sample. See [Benchmark details](#benchmark-details).

Long-context target-only decode uses an actually allocated packed KV/indexer
cache at each frontier. These measurements seed deterministic cache contents
instead of performing a 128K-1M prompt prefill. The final DS4X column reports
50 alternating A/B samples after 3 warmups; the frozen upstream column retains
its original 30-sample measurement:

| Context | Upstream TPOT (ms) | DS4X TPOT (ms) | Upstream (tok/s) | DS4X (tok/s) | Decode speedup |
|---:|---:|---:|---:|---:|---:|
| 128K | 81.248 | 70.155 | 12.308 | 14.254 | 1.158x |
| 256K | 92.640 | 72.757 | 10.794 | 13.744 | 1.273x |
| 512K | 115.246 | 80.090 | 8.677 | 12.486 | 1.439x |
| 1M | 363.286 | 92.190 | 2.753 | 10.847 | 3.940x |

The 1M row allocates the full 3.43 GiB packed persistent cache. It validates
decode at a million-token frontier; it is not a million-token prefill result.

## Origin and attribution

DS4X is derived from [antirez/ds4](https://github.com/antirez/ds4), with the
optimization work developed against upstream commit
[`84cc882`](https://github.com/antirez/ds4/commit/84cc882352757baf628a1776badf7cc54d584e28).
DS4 provides the original GGUF loader, DeepSeek-V4-Flash execution pipeline,
session and checkpoint machinery, and the `ds4_*` API retained during the
current migration.

DS4X extracts that runtime into a CUDA-only, GB10-specific project and adds the
packed-cache and long-context kernel work described below. Selected quantized
matrix kernels are adapted from llama.cpp's `ggml-cuda` backend; the exact
upstream pin and local modifications are documented in
[`src/ds4x_kernel/quantization/mmq/VENDOR.md`](src/ds4x_kernel/quantization/mmq/VENDOR.md).

The DS4 and llama.cpp copyright notices remain in `LICENSE`. DS4X is an
independent community project and is not affiliated with DeepSeek or NVIDIA.
NVIDIA CUTLASS is consumed as a pinned submodule under its BSD-3-Clause
license; DS4X does not vendor or modify its source.

## What DS4X changes

| Area | DS4 CUDA path used as the baseline | DS4X |
|---|---|---|
| Deployment target | General multi-backend runtime | CUDA-only GB10 build, compiled specifically for `sm_121a` |
| Persistent attention history | Expanded F32 cache slots | Native 583-byte packed rows: E4M3 non-RoPE values, FP16 RoPE values, and UE8M0 scales |
| CSA indexer history | 128 F32 values per compressed row | 68-byte MXFP4 rows with four UE8M0 block scales |
| Batch-one CSA scoring | Scalar/direct path over expanded history | Direct packed-row SM121 block-scaled MMA scorer; no full-history re-encode per generated token |
| CSA score/top-k handoff | Full F32 score array, then a separate exact top-512 pass | At 256K+ context, each scorer CTA retains an exact local top-512 and hierarchical key merges avoid materializing the full score array |
| HCA long-context attention | Optimized path limited to 7,936 selected rows | Exact score-split attention without the old row ceiling, including the 1M frontier |
| Q8 weight handling | Selective Q8-to-FP16 prefill cache | Same selective FP16/cuBLAS prefill path; no activation-Q8 exact-MMA path when the cache is available |
| FP16 GEMM dispatch | cuBLAS for every Tensor Core-sized projection | Measured cuBLAS/CUTLASS dispatch; CuTe describes the logical layouts and CUTLASS only takes the production indexer shape where it wins |
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
- At `n_comp >= 65536` (256K context), B1 scoring emits the exact sortable
  `(score,index)` keys directly, retains 512 candidates per 2,048-row CTA and
  hierarchically merges them to the model's exact top-512. The float score
  ordering and lower-index tie break are unchanged, while the full F32 score
  array and its second read are removed.
- HCA uses exact score-split attention without the old 7,936-row fast-path
  ceiling, including at a 1M-token frontier.
- Checkpoint weights remain Q8. Prefill lazily creates the same selective FP16
  weight mirrors as upstream so activations feed cuBLAS/HMMA directly instead
  of being requantized for the exact INT8 Tensor Core path.
- Disk checkpoints store the native packed rows. Payload ABI version 3
  deliberately rejects old F32-cache checkpoint files.
- The quantized matmul adapter calls the selected CUDA kernels directly through
  a small raw-argument ABI, without requiring the full ggml graph runtime.
- Modern CUDA C++ operators live in independent translation units under
  `src/ds4x_kernel/backends/`. The first migrated operator owns the FP16
  projection fallbacks and a pinned CUTLASS/CuTe GEMM backend; the stable C ABI
  and the existing cuBLAS path remain available as fallbacks.
- CUDA kernels, the single-request inference engine, storage, frontends, and
  support libraries are separated into explicit source modules. The packed
  backend builds as nine ordinary CUDA translation units; shared templates and
  force-inlined device helpers remain in private `.cuh` headers.

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
`sm_121a`, GNU make, a C99 compiler, and the pinned CUTLASS submodule.

```bash
git submodule update --init --recursive
make -j4 spark smoke perf ds4_test
```

The main programs are `ds4` and `ds4-bench`.

Focused design studies:

- [QKVO FP4 and quantized DSpark feasibility](docs/qkvo_fp4_dspark_feasibility.md)
- [Modern CUDA, DSpark, and Codex roadmap](docs/roadmap.md)

## Project layout

```text
src/
  README.md       module ownership and source-layout notes
  apps/           single-request CLI and benchmark frontends
  engine/         model loading and single-session graph runtime
    core/         platform support
    model/        GGUF, validation, tokenizer and memory policy
    runtime/      graph, prefill, decode and verification
    session/      lifecycle, generation and checkpoints
    internal/     private cross-module declarations
  ds4x_kernel/    GB10 CUDA backend and device-facing API
    backend/      independently compiled packed runtime and operator modules
    backends/     standalone modern CUDA C++ operators and dispatch backends
    quantization/ adapted quantized CUDA matrix kernels
    include/      kernel ABI consumed by the inference engine
  storage/       packed checkpoint storage
  support/       small embedded support libraries (linenoise and rax)
tests/
  ds4x_kernel/    standalone kernel parity, smoke and performance tests
  engine/         engine and session semantic tests
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
`src/ds4x_kernel/include/ds4_gpu.h`; pack/unpack and the SM121 scorer live in
the independent modules under `src/ds4x_kernel/backend/ops/`.

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

### Phase 2 release regression

The final single-device build was rebuilt from scratch and checked on DGX
Spark on 2026-08-12. `make kernel-test` passed packed cache, attention,
indexer, CUTLASS FP16, projection, operator-adapter and every MMQ parity case.
The 8K packed-checkpoint round trip reported a 42.614 MiB payload,
`max_abs=0`, `rmse=0`, `different=0/129280`, and identical argmax logits.

The final synthetic frontier validation used three warmups and 50 timed steps
per path, alternating the fused and fallback CSA dispatch in one process. It
exercises the release build's packed target-only decode path.

| Context | Median TPOT (ms) | Throughput (tok/s) |
|---:|---:|---:|
| 128K | 70.155 | 14.254 |
| 256K | 72.757 | 13.744 |
| 512K | 80.090 | 12.486 |
| 1M | 92.190 | 10.847 |

The 128K row stays on the established score-then-top-k path. At 256K, 512K
and 1M, exact CSA score/top-k fusion improves median TPOT by 1.57%, 1.05% and
2.55% over the same-process fallback.

### CUTLASS/CuTe dispatch regression

CUTLASS 4.6.2 is pinned at commit `6c65a175`. GB10's FP16 path uses
warp-level `mma.sync`; CUTLASS's SM120 CollectiveBuilder is currently aimed at
F8/F6/F4 inputs, so DS4X deliberately uses its forward-compatible SM80 FP16
TensorOp composition rather than mislabeling an SM100 `tcgen05` path as
available on `sm_121`. CuTe owns the logical `[tokens, in] x [in, out]`
problem shape and CUTLASS owns the explicit kernel composition.

The DGX Spark microbenchmark keeps cuBLAS for latency-sized compressor/router
projections and selects CUTLASS for indexer `q_b` from 128 tokens upward. The
measured CUTLASS/cuBLAS latency ratios were `0.899x`, `0.810x`, `0.956x`, and
`0.843x` at 128, 256, 1024, and 4096 tokens respectively. Output-A also has a
CuTe-shaped strided-batched CUTLASS path for the model's eight
`[tokens,4096] x [4096,1024]` GEMMs: it measured `0.899x` at 512 tokens and
`0.947x` at 1024, but `1.255x` at 4096. Production therefore enables it only
for 512-1024 token chunks and keeps the common 4096-token chunk on cuBLAS.

The focused attention output-A parity test is bit-identical to cuBLAS. A full
835-token prompt A/B at prefill chunk 512 also produced byte-identical
129,280-logit JSON. At chunk 1024, per-layer profiling reduced cumulative
output-A time from `496.149 ms` to `479.963 ms` (`3.3%`); whole-prefill impact
is smaller because attention output-A is only one stage of every layer.

Whole-model A/B used the same binary with `DS4_CUDA_F16_BACKEND=cublas` versus
the default measured dispatch. On a 6,000,000-byte repeated-token fixture, 4K
prefill was `851.26` versus `854.66 tok/s`, and warmed 16K prefill was `874.15`
versus `885.94 tok/s`. The 4K full-logit JSON files were byte-identical. These
numbers are dispatch regression checks, not replacements for the published
prompt-dependent performance table.

`DS4_CUDA_F16_BACKEND=cublas` disables CUTLASS dispatch for reproducible
fallback runs. `DS4_CUDA_F16_BACKEND=cutlass` force-attempts CUTLASS on all
supported FP16 batch shapes and is intended only for kernel research; the
default auto policy is the measured production setting.
`DS4_CUDA_ATTN_OUTPUT_A_BACKEND=cublas|cutlass` separately controls the
strided-batched attention output-A path.

### CSA score/top-512 fusion

The B1 CSA scorer now fuses packed MXFP4 scoring with exact top-512 selection
for `n_comp >= 65536`. Every 2,048-row CTA produces 512 ordered 64-bit keys;
four-way hierarchical merges retain the same float total order and
lower-index tie break as the separate top-k implementation. Smaller contexts
keep the legacy two-stage path because their end-to-end measurements did not
show a repeatable gain.

The final same-process A/B alternated legacy and fused dispatch over identical
frontiers, with 3 warmups and 50 timed samples per path. Full vocabulary
logits were bit-identical at every context (`max_abs=0`, `rmse=0`, identical
argmax).

| Context | Legacy TPOT (ms) | Fused TPOT (ms) | TPOT improvement |
|---:|---:|---:|---:|
| 128K | 70.101 | 70.155 | threshold not active |
| 256K | 73.915 | 72.757 | 1.57% |
| 512K | 80.938 | 80.090 | 1.05% |
| 1M | 94.605 | 92.190 | 2.55% |

Reproduce it with:

```bash
DS4_SYNTH_SKIP_WARM_WEIGHTS=1 \
DS4_SYNTH_AB_FUSED_INDEXER=1 \
./tests/integration/synth_frontier_bench MODEL ab-sweep 3 50
```

`DS4_CUDA_NO_FUSED_INDEXER_TOPK=1` forces the legacy score-then-top-k path.
The fused implementation is not a relaxed or approximate selector.

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
The final DS4X table reports the median of 50 alternating samples after 3
warmups. The optional startup pass that touches all mapped model pages was
disabled; this does not remove the per-context decode warmups or the resident
aligned CUDA artifacts.

This isolates steady-state decode cost versus cache length. It is not a prefill
benchmark and the synthetic zero history is not a language-quality workload.

The release-validation run is the fresh-build, three-warmup, 50-sample table
in [Phase 2 release regression](#phase-2-release-regression). At 1M the runtime
planned 84.25 GiB: 80.76 GiB resident model, 3.43 GiB packed KV/indexer, and
0.06 GiB runtime buffers. System-wide unified-memory usage also includes the
Linux page cache, driver state, and unrelated processes, so it is not used as
a process allocation figure.

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
tokens. The paired 4K and 16K release values are summarized in the top-level
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
| 128K | 81.248 | 70.155 | 12.308 | 14.254 | 1.158x |
| 256K | 92.640 | 72.757 | 10.794 | 13.744 | 1.273x |
| 512K | 115.246 | 80.090 | 8.677 | 12.486 | 1.439x |
| 1M | 363.286 | 92.190 | 2.753 | 10.847 | 3.940x |

At 1M, upstream allocates 13.46 GiB of F32 KV through managed memory and shows
substantial paging variance (236.7-540.4 ms across the timed steps); DS4X uses
3.43 GiB of packed KV and remains on the stable packed decode path.

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

The Phase 2 release build also produced a byte-identical 4K full-logit JSON
against the last validated pre-refactor snapshot (SHA-256
`b26645e9d08c1e84005634e4a9b50353e32ddc67ba8eef088eb72b70070e6474`).
At 128K, 256K, 512K, and 1M, the optimized and scalar-oracle runs reported
`max_abs=0`, `rmse=0`, `different=0/129280`, and identical argmax values.

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
- Packed multi-token prefill and decode run on one CUDA GPU and one active
  request. Multi-GPU placement, TP/EP and server batching are out of scope.
- The target verifier/checkpoint and DSpark cache-window invariant are tested,
  but no compatible DSpark support GGUF was available for an end-to-end draft
  model run. Quantized DSpark integration remains Phase 3 work.
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
