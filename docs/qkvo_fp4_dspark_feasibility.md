# DS4X QKVO FP4 与量化 DSpark 可行性研究

本文记录 2026-08-10 在 DGX Spark GB10 上完成的第一性原理分析、独立
QKVO kernel 实验、长上下文显存实测，以及下一阶段接入量化 DSpark 的工程边界。

## 结论

1. QKVO 可以从 Q8_0 改成 NVFP4，但在当前 GB10 decode 场景中，正确方向是
   **NVFP4 权重 + Q8_1 activation 的 W4A8 向量 kernel**，不是直接调用通用
   W4A4 Tensor Core MMQ。
2. 对 B1 的五个真实 QKVO shape，W4A8 NVFP4 的单层合计时间为
   `0.3039 ms`，Q8_0 W8A8 为 `0.5662 ms`，孤立 projection 加速为
   `1.863x`。三个 34 MiB 大 projection 的单项加速为 `1.84x-1.96x`。
3. 当前通用 NVFP4 Tensor Core MMQ 在 `N=1..8` 都不合适：真实大 projection
   约为 `1.27-1.48 ms`，比 W4A8 NVFP4 慢一个数量级。该结果说明当前 kernel
   的 tile、stream-K/fixup 和调度路径不适合小 N，不代表 GB10 的 FP4
   Tensor Core 本身无效。
4. 孤立随机矩阵实验中，NVFP4 W4A8 的输出 NRMSE 约为 `8.8%-10.5%`，
   cosine 约为 `0.9945-0.9961`；Q8_0 约为 `0.7%-0.8%`。因此性能方向成立，
   但**模型精度尚未验证，不能直接默认全量替换**。
5. 当前 Q8 target 在 128K、256K、512K、1M decode 后分别实测剩余
   `16.56/16.26/14.79/13.43 GiB` unified memory。按现有量化 DSpark
   `5.6168 GiB` 估算，1M 稳态仍剩约 `7.81 GiB`；若 target QKVO 同时改成
   NVFP4，可再释放 `2.1416 GiB`，约剩 `9.95 GiB`。
6. 量化 DSpark 的**稳态容量可行**，更大的风险是启动阶段同时存在 support
   GGUF page cache、原始布局和 aligned/repacked 权重造成的峰值。加载器需要
   流式重排并及时回收源页面。

## 1. 为什么 QKVO 仍然是 memory-bound

对一个 `M x K` 权重矩阵和 B1 activation，矩阵向量乘的主要工作为：

```text
FLOP = 2 * M * K

T_projection
  = max(T_weight_memory, T_compute)
  + T_activation_quant
  + T_launch
  + T_epilogue
```

Q8_0 每 32 个权重使用 32 bytes value 和 2 bytes FP16 scale：

```text
Q8_0 bytes/weight = 34/32 = 1.0625
Q8_0 arithmetic intensity at B1
  = 2 / 1.0625
  = 1.882 FLOP/byte
```

DS4X 测试的 GGML NVFP4 block 每 64 个权重使用 32 bytes E2M1 value 和
4 bytes UE4M3 scale：

```text
NVFP4 bytes/weight = 36/64 = 0.5625
NVFP4 arithmetic intensity at B1
  = 2 / 0.5625
  = 3.556 FLOP/byte

NVFP4 / Q8_0 bytes
  = (36/64) / (34/32)
  = 9/17
  = 0.5294118
```

即使改成 NVFP4，B1 的 arithmetic intensity 仍然很低，远未达到 GB10
从 memory-bound 转成 compute-bound 的平衡点。因此这里最重要的是减少权重
读取并保持较高有效带宽，而不是追求峰值 FP4 TFLOPS。

MXFP4 每 32 个权重使用 16 bytes value 和 1 byte E8M0 scale：

```text
MXFP4 bytes/weight = 17/32 = 0.53125
```

MXFP4 比 NVFP4 少约 5.6% bytes，但 scale group 是 32，而 NVFP4 是 16。
实测 NVFP4 的误差更低，并且大矩阵向量 kernel 的有效带宽更高，所以本项目
优先研究 NVFP4。

## 2. QKVO shape 与容量收益

V4-Flash 每层五个主要 attention projection 为：

| Projection | `M x K` | Q8_0 | NVFP4 |
|---|---:|---:|---:|
| Q_a | `1024 x 4096` | 4.250 MiB | 2.250 MiB |
| KV | `512 x 4096` | 2.125 MiB | 1.125 MiB |
| Q_b | `32768 x 1024` | 34.000 MiB | 18.000 MiB |
| O_a | `8192 x 4096` | 34.000 MiB | 18.000 MiB |
| O_b | `4096 x 8192` | 34.000 MiB | 18.000 MiB |
| **每层** | | **108.375 MiB** | **57.375 MiB** |

43 层总节省：

```text
QKVO saving
  = 43 * (108.375 - 57.375) MiB
  = 2,193 MiB
  = 2.141602 GiB

converted target size estimate
  = 80.76 - 2.141602
  = 78.618398 GiB
```

## 3. 当前代码覆盖与缺口

代码证据：

- `src/engine/model/validation.c` 的 Flash layout validation 当前要求这五类 projection 使用
  dense quant layout，现有模型实际为 Q8_0。
- `src/ds4x_kernel/quantization/mmq/mma.cuh` 已包含
  `mma.sync.aligned.kind::mxf4nvf4...e2m1...ue4m3` 原生 PTX。
- `src/ds4x_kernel/quantization/mmq/mmq.cuh` 已包含 NVFP4 weight loader、FP4 MMA trait 和
  Q8_1 vector dot。
- `src/ds4x_kernel/quantization/mmq/quantize.cu` 已包含 MXFP4/NVFP4 activation quantizer。
- 本次实验补齐了 `ds4_mmq_nvfp4_dense`、MXFP4/NVFP4 dense-vector probe
  入口和可复现实验程序。

尚未完成的整模型接入：

- `src/engine/model/gguf.c` 尚未把 GGUF type 40 注册为 DS4 dense NVFP4 类型。
- `tensor_type_is_dense_quant()` 和 DSpark dense validation 尚未接受 NVFP4。
- `ds4_gpu_matmul_quant_tensor()` 尚未把 type 40 分派到 NVFP4 vector kernel。
- 当前 Q8 production path 已有 Q_a+KV、Q_b+norm+RoPE、grouped O_a 和
  O_b+HC 等融合；NVFP4 版本必须重建这些融合，否则会丢掉部分字节收益。
- 尚无 Q8 QKVO 到 NVFP4 QKVO 的离线 GGUF converter。

## 4. 实验方法

实验程序为 `tests/ds4x_kernel/qkvo_fp4_probe.cu`，构建和运行：

```bash
make qkvo-fp4-probe
```

测试口径：

- 硬件：DGX Spark，NVIDIA GB10，`sm_121a`。
- CUDA：13.0，driver 580.82.09。
- shape：使用上表五个真实 B1 QKVO shape。
- 权重和 activation：固定 seed 的正态随机 F32 数据，分别量化为 Q8_0、
  MXFP4 和 NVFP4。
- 精度 reference：CPU F64 accumulation 的原始 F32 `W*x`。
- 性能：每项 31 次，报告 median。
- 为避免重复命中 L2，每种格式轮换约 160 MiB 的权重副本。
- 使用 persistent activation scratch，计时仍包含每步 activation quantization、
  matmul、output sanitize 和 kernel launch，但不包含 `cudaMallocAsync`。
- `Q8v/MXv/NVv` 为 Q8_1 activation 的 vector path；`NVtc` 为 activation
  也量化到 FP4 的 native Tensor Core MMQ path。

这不是模型质量实验。随机矩阵只用于检查 kernel 数值误差量级和真实性能，
不能代替 full-logit、生成一致性和任务集评测。

## 5. B1 实测结果

| Shape | Q8v ms | NVv ms | 加速 | Q8v GB/s | NVv GB/s | NVv NRMSE | NVv cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q_a | 0.0311 | 0.0207 | 1.502x | 143.1 | 114.0 | 0.1044 | 0.994557 |
| KV | 0.0207 | 0.0147 | 1.408x | 107.6 | 80.1 | 0.0879 | 0.996131 |
| Q_b | 0.1610 | 0.0874 | 1.842x | 221.5 | 215.9 | 0.1035 | 0.994641 |
| O_a | 0.1750 | 0.0903 | 1.938x | 203.7 | 208.9 | 0.1027 | 0.994718 |
| O_b | 0.1784 | 0.0908 | 1.965x | 199.9 | 207.8 | 0.1050 | 0.994476 |
| **每层合计** | **0.5662** | **0.3039** | **1.863x** | | | | |

Q8v 的 NRMSE 为 `0.0070-0.0077`，cosine 约为 `0.99997`。MXFP4 W4A8
的 NRMSE 为 `0.107-0.116`，NVFP4 W4A8 为 `0.088-0.105`，说明 NVFP4
的 16-value scale group 在该实验中明显优于 MXFP4。

分 projection 的 43 层理论节省为：

| Projection | 43 层节省 |
|---|---:|
| Q_a | 0.447 ms/token |
| KV | 0.258 ms/token |
| Q_b | 3.165 ms/token |
| O_a | 3.642 ms/token |
| O_b | 3.767 ms/token |
| **合计** | **11.279 ms/token** |

Q_a+KV 只贡献约 6.3% 的总时间收益。若精度实验迫使混合量化，真正值得优先
争取的是 Q_b、O_a 和 O_b；但 Q/K 会影响 attention score，O 会直接进入
residual stream，二者都必须通过 full-model ablation 决定，不能只按敏感性猜测。

## 6. 小 batch 与 DSpark verification

三个 34 MiB projection 的 `N=1..8` 实测如下，单位 ms：

| Shape | N | Q8v | NVv W4A8 | NVtc W4A4 |
|---|---:|---:|---:|---:|
| Q_b | 1 | 0.1610 | 0.0874 | 1.4678 |
| Q_b | 2 | 0.1692 | 0.0916 | 1.2723 |
| Q_b | 4 | 0.1722 | 0.1323 | 1.2686 |
| Q_b | 6 | 0.1825 | 0.1384 | 1.2660 |
| Q_b | 8 | 0.2184 | 0.1815 | 1.2708 |
| O_a | 1 | 0.1750 | 0.0903 | 1.4793 |
| O_a | 2 | 0.1793 | 0.0905 | 1.4744 |
| O_a | 4 | 0.1815 | 0.0893 | 1.4777 |
| O_a | 6 | 0.1866 | 0.0952 | 1.4852 |
| O_a | 8 | 0.1845 | 0.0974 | 1.4784 |
| O_b | 1 | 0.1784 | 0.0908 | 1.4817 |
| O_b | 2 | 0.1824 | 0.0933 | 1.4761 |
| O_b | 4 | 0.1879 | 0.0904 | 1.4719 |
| O_b | 6 | 0.1906 | 0.0956 | 1.4751 |
| O_b | 8 | 0.1857 | 0.0973 | 1.4813 |

DSpark 一轮 target verification 的常见 query 数不超过 6，因此当前证据支持：

```text
N <= 8: use NVFP4-weight + Q8_1-activation vector path
N large: separately benchmark native W4A4 MMQ before enabling it
```

不能仅依据 `N=8` 已填满 `m16n8` 的 N tile 就认为 native MMQ 应该变快。
当前实现还存在 stream-K、fixup、tile selection 和 wrapper overhead，实测没有出现
交叉点。后续应把 crossover probe 扩展到 N=16/32/64/128。

## 7. 整模型吞吐边界

8K release baseline 约为：

```text
TPOT_Q8 = 64.185 ms/token
throughput_Q8 = 15.58 tok/s
```

若 43 层都能保留现有 Q8 production fusion，并兑现 isolated W4A8 NVFP4
的时间差：

```text
T_QKVO_Q8 = 43 * 0.5662 = 24.347 ms/token
T_QKVO_NV = 43 * 0.3039 = 13.068 ms/token
saving     = 11.279 ms/token

TPOT_NV upper-bound
  = 64.185 - 11.279
  = 52.906 ms/token

throughput_NV upper-bound
  = 1000 / 52.906
  = 18.90 tok/s
```

独立的纯字节上界为：

```text
traffic saving = 2,193 MiB/token
at 224.5 GB/s streaming bandwidth:

T_saved
  = 2,193 * 1.048576 / 224.5
  = 10.24 ms/token
```

两个方法给出相近数量级。由于当前 FP4 probe 尚未实现 production fusion，可信的
工程区间应保守写为：

```text
8K B1: 约 17.5-18.9 tok/s，18.9 是 kernel-level 上界，不是整模型实测
1M B1: 约 11.7-12.1 tok/s，基于 94.502 ms Phase 2 release TPOT 扣除固定 QKVO 时间
```

长上下文时 CSA/HCA history cost 增大，固定 QKVO 优化在绝对时间上仍近似固定，
所以相对加速比例会下降。

## 8. 长上下文显存实测

命令口径：单独为每个 context 创建进程，跳过全模型 page warmup，创建 synthetic
frontier，执行一次 decode 后读取 `cudaMemGetInfo`：

```bash
DS4_SYNTH_SKIP_WARM_WEIGHTS=1 \
DS4_SYNTH_MEMORY_REPORT=1 \
./tests/integration/synth_frontier_bench MODEL CONTEXT 0 1
```

性能数值是冷启动单步，不用于吞吐结论；这里只使用 decode 后 free memory。

| Context | Packed KV | Planned total | 实测 free after decode | 加 Q2 DSpark 后 | 再把 target QKVO 改 NVFP4 |
|---:|---:|---:|---:|---:|---:|
| 128K | 0.43 GiB | 81.20 GiB | 16.56 GiB | 10.94 GiB | 13.09 GiB |
| 256K | 0.86 GiB | 81.64 GiB | 16.26 GiB | 10.64 GiB | 12.78 GiB |
| 512K | 1.72 GiB | 82.51 GiB | 14.79 GiB | 9.17 GiB | 11.31 GiB |
| 1M | 3.43 GiB | 84.25 GiB | 13.43 GiB | 7.81 GiB | 9.95 GiB |

`cudaMemGetInfo` 在 GB10 unified-memory 系统上是机器级观察值，受 Linux page
cache、driver 和其他进程影响，不是严格的进程 RSS。表中后两列是从实测 free
分别减去 `5.6168 GiB`、再加回 `2.141602 GiB` 的容量推演，不是已加载
DSpark 后的实测。

## 9. 量化 DSpark 的可行边界

当前规划中的 Q2 DSpark 不是“所有 tensor 都是 2 bit”，而是：

- routed gate/up 使用 IQ2_XXS；
- routed down 使用 Q2_K；
- dense attention、main projection、Markov 等非 routed tensor 保持现有
  Q8/F16/F32 支持格式；
- target output head 共享，不增加 resident capacity。

容量推导：

```text
routed DSpark
  = 256 experts * 3 layers * 6.75 MiB
  = 5.0625 GiB

non-routed DSpark
  = 595,182,108 bytes
  = 0.5543 GiB

additional resident DSpark
  = 5.0625 + 0.5543
  = 5.6168 GiB
```

DS4X 已有 `--dspark`、`--dspark-strict`、confidence pruning、multi-row target
verification 和 DSpark regression hook，但 DGX Spark 当前没有可用的量化
DSpark GGUF，因此本轮只能验证 runtime 代码和容量，不能给出 acceptance 或
端到端 speculative throughput 实测。

若要求“所有 DSpark 权重都 2 bit”，还需要扩展 dense projection 的低 bit
格式、CUDA dispatch、CPU reference 和 converter。这种方案可能继续省显存，
但 attention/main projection/Markov 的精度下降会直接影响 draft acceptance，
未必提高最终吞吐。

### 推荐加载顺序

1. 先用 `--dspark-strict` 只加载 support model，不开启 speculation，实测
   startup peak、steady-state free memory 和 target-only logits 是否不变。
2. support GGUF 直接保存最终 aligned packed layout，或逐 tensor 重排后立即
   `madvise(MADV_DONTNEED)` 源页，避免 raw + aligned 双份长期共存。
3. 确认 1M 下加载完成后至少保留 4-6 GiB allocator/driver slack，再打开 draft。
4. 分别记录 draft time、target verify time、平均接受长度、平均接受率和 fallback
   比例；不能只报告 aggregate tok/s。

## 10. DGX Spark 极致优化路线

按当前证据排序：

### P0：QKVO NVFP4 W4A8 production fusion

- 增加 GGUF type 40、row-byte validation 和离线 Q8-to-NVFP4 converter。
- B1/B2-B8 dispatch 到 NVFP4-weight + Q8_1-activation vector kernel。
- 融合 Q_a+KV，共用一次 activation quantization。
- 重建 Q_b + RMSNorm + RoPE、grouped O_a、O_b + HC epilogue。
- 使用 persistent scratch 和 CUDA Graph，不允许 projection 内动态 allocation。
- 每个 tensor 保留 runtime fallback，可做 Q_a/KV/Q_b/O_a/O_b 独立 ablation。

### P0：full-model 精度门槛

- 单层 output RMSE/cosine 和 attention score/top-k agreement。
- 43 层 MoE expert-route agreement。
- full-logit RMSE、argmax agreement 和 128-token greedy token agreement。
- 128K-1M needle/recall 与长文本生成。
- DSpark acceptance length/rate；target 和 draft 的量化误差会共同改变接受率。
- 若全量 NVFP4 不过线，尝试 outlier row 保持 Q8、其余 row NVFP4 的混合布局。

### P1：CSA score + top-k 合并

- packed indexer score 直接进入 streaming top-512，避免完整 score buffer 和独立
  finalize/top-k pass。
- 1M trace 中 score-split finalize、indexer score 和 score tile 仍是主要 kernel
  group，收益高于继续优化 packed-KV unpack。

### P1：HCA exact attention 合并

- 把 score tile、online softmax 和 value reduction 做成更少的 persistent CTA
  pipeline，减少中间 global-memory score traffic。
- 保持无 7,936-row 限制和 exact attention 语义。

### P1：MoE selected-expert megakernel

- 继续优化 IQ2_XXS gate/up + SwiGLU + Q2_K down 的 decode-to-register chain。
- 对 top-6 expert 做共享 activation staging、expert worklist 合并和更少 launch。
- 当前 FFN/MoE 仍是全模型主要剩余时间之一，QKVO 完成后其占比会进一步上升。

### P2：native W4A4 crossover

- 把 NVFP4 native MMQ 的 benchmark 扩展到 N=16/32/64/128。
- 用 Nsight Compute 检查 DRAM、L2、Tensor Core issue、occupancy、stream-K
  fixup 和每次 launch 的有效 tile 数。
- 只有找到真实 crossover 后，才允许大 batch/prefill 使用 W4A4；DSpark 的
  `q<=6` 不应走当前 native MMQ。

### P2：DSpark 调度

- native batching 同时处理多请求 draft/verify，不能循环调用 B 次 B1。
- 按 context、batch、confidence 动态选择 verification depth，并在预计收益小于
  target-only 时回退。
- 把 support model 的权重布局和 target 的 aligned artifact 管理统一，避免第二套
  allocator 和重复 page residency。

## 11. 验证状态

- `make smoke`：通过，packed attention/indexer parity 为 0，large packed
  indexer regression 通过。
- 修改后 target-only synthetic frontier：128K/256K/512K/1M median 分别为
  `13.243/13.450/12.360/10.320 tok/s`。该短跑使用 1 warmup、3 samples，
  仅用于确认默认 Q8 full-model 路径没有功能回归，不替代 README 的 release run。
- FP4 数值结果仅为 isolated synthetic matrix 实验。
- 尚未生成 NVFP4 target GGUF，尚未完成 full-model 精度验证。
- DGX Spark 尚无量化 DSpark GGUF，尚未实测 speculative acceptance 和吞吐。
