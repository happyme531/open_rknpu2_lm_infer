# Qwen3 W4A8 Gate/up：两波 CPU 归约与 SwiGLU 一次消费

2026-10-22。真实 Qwen3-4B layer2 输入、权重，RK3588 单个完整 FFN 测量。

**保留原 FP64 SwiGLU 算法，只合并两波 CPU 消费，完整 FFN 从 9.419 降至 8.835 ms，时延降低 6.20%，middle 与最终输出逐字节一致。沿用已有 FP32 SwiGLU 后，融合版本为 8.372 ms，较原版降低 11.12%；Gate/up 阶段降低 15.62%。**

后续 [SwiGLU 累计 maxabs、Down 复用 scale](QWEN3_W4A8_SWIGLU_MAXABS_FUSION_20261022.md) 在同批对照中进一步将完整 FFN 从 8.410 降到 8.200 ms，增量约 2.50%，与两波融合基线逐字节一致。

收益来自省掉一整轮 INT32 accumulator 的写回和读取。两份 K partial 在同一次 CPU 收尾中直接完成整数重建、correction、scale 和 SwiGLU，不增加 NPU 工作或中间搬运。

## 范围和协议

- Gate/up：M128/K2560/N19456，输出 F9728；panel8、Mtile128、Ntile4096、Ktile1472，对应两个 K wave：1472/1088。
- Down：M128/K9728/N2560，Ktile1888、六波；所有比较模式使用同一个原有 Down workspace。
- 权重是同一个 checkpoint layer2 的原始 codes/scales；输入为已有模型运行捕获的 `p0-l2-post.f32`。
- 同一进程内共用 Gate/up 权重、executor、输入/partial DMA 缓冲、accumulator 分配、输出缓冲，以及 Down 权重和 workspace；只切换 CPU consumer。
- CPU4–7、四线程、2.256 GHz；三核 NPU、1 GHz；DDR 未控制。使用共享 NPU 锁、`OMP_WAIT_POLICY=PASSIVE`、频率恢复工具。
- 两个进程，每进程 30 个对称周期，六种模式按 `A B C D E F F E D C B A` 运行；每模式共 120 次完整调用。逐事件记录关闭。周期属于进程内重复，不视作独立板卡实验。

CPU 原版归约来自 `-O2` 静态库。其它 consumer 按 `-O3` 编译，其中保留同算法的 FP64/O3 控制。公共执行器由同一个诊断对象提供，支持两波消费的 CPU 读窗口；各比较模式使用相同执行器构建。

## 主要结果

以下为六模式交错测试的均值，单位 ms。

| CPU 收尾方式 | Gate/up | Down | 完整 FFN | 相对原版时延变化 |
|---|---:|---:|---:|---:|
| 原版 FP64 / O2 | 6.628 | 2.791 | 9.419 | 基线 |
| 同算法 FP64 / O3 | 6.736 | 2.789 | 9.526 | +1.13% |
| 原有 FP32 / O3 | 6.363 | 2.795 | 9.158 | −2.77% |
| FP32 四路交错 / O3 | 6.291 | 2.780 | 9.071 | −3.69% |
| 两波融合＋原 FP64 / O3 | 6.048 | 2.787 | **8.835** | **−6.20%** |
| 两波融合＋FP32 四路交错 / O3 | **5.593** | 2.779 | **8.372** | **−11.12%** |

两波融合＋FP32 相比同数学表达式的 FP32 四路交错控制，再降低完整 FFN 时延 **7.71%**。两个进程分别为 9.076 → 8.368 ms、9.067 → 8.376 ms；各 30/30 个配对周期均更快。

FP64 融合相对原版的两个进程分别为 9.393 → 8.822 ms、9.445 → 8.848 ms；同样各 30/30 个配对周期更快。相对同编译选项 FP64/O3 控制，它也更快，因此收益不依赖降低激活计算精度。

另一批五模式对照中，原版为 9.423 ms、FP32 四路交错为 9.062 ms、FP32 两波融合为 8.396 ms，融合前后的排序一致。该批数据保留为独立结果，不与主要六模式样本合并。

## 执行流程与减少的工作

原来的 CPU 数据路径：

```text
同步 P0 → correction + 16*P0_hi + P0_lo → 写 INT32 accumulator
同步 P1 → 读 accumulator + 16*P1_hi + P1_lo → scale → SwiGLU → middle
```

两波融合路径：

```text
第二波 GEMM 执行期间，同步 P0 并保持 CPU 读窗口
第二波完成后，同步 P1
同一个 CPU kernel 读取 P0、P1 → correction/整数重建 → scale → SwiGLU → middle
释放两个 CPU 读窗口
```

两份 partial 原本就在独立的双缓冲中。本实验不增加 partial 数量，执行器负责同步和读窗口的生命周期。原有 accumulator 分配保留，以保证同 workspace 对照，但融合模式不读写它。

M128/N19456 的每份 high/low INT16 partial 为 9.5 MiB，INT32 accumulator 同样为 9.5 MiB；middle 为 4.75 MiB。主要 CPU 张量逻辑读写为：

| 项目 | 原路径 MiB | 两波融合 MiB |
|---|---:|---:|
| 读取 P0、P1 | 19 | 19 |
| 写回、重读 accumulator | 19 | 0 |
| 写 middle | 4.75 | 4.75 |
| 合计 | **42.75** | **23.75** |

因此每次删除约 **19 MiB** 的 CPU 逻辑读写。这不是实测 DDR 流量，也不包含 scale、临时栈数据、cache write-allocate 等访问。

同 FP32 数学表达式的控制中，累计 CPU reduce/finish 跨度从 **3.565 降到 2.441 ms**；Gate/up 完整时延从 **6.291 降到 5.593 ms**。省下的 CPU 跨度大于完整调用收益，因为原来的第一波归约有一部分与第二波 GEMM 重叠。融合路径保留第一波同步的重叠，但不再提前执行第一波整数归约。

输出同步累计仅从约 1.110 降到 1.090 ms。此项结果没有依赖省略驱动缓存同步。

## CPU 算术对照

原 FP64 激活已经交错执行四条独立 Horner 链。FP32 原型使用七阶多项式，新的四路版本将 16 个值分成四条独立 FP32 链，保持相同 range reduction、系数、除法与边界回退。

单独增加指令交错的收益较小，且批次间变化：最初四模式对照的完整 FFN 为原版 9.375、FP32 9.089、FP32 四路 9.066 ms。主要六模式批次为 9.419、9.158、9.071 ms。不能用单次四路改动的细小差异推断大幅算术余量。

已有实际输入的离线整数重建中，仅约 0.385% 的四元素组因 `abs(gate)>16` 回退，因此 FP32 路径的收益小并非由大量回退造成。CPU 编译选项也需要单独控制：本次同算法从 O2 改 O3 反而慢约 1.13%。

## 正确性与适用边界

每进程在计时前检查真实输入、全零输入、四倍真实输入，并反复复用同一组 workspace：

- FP64 两波融合的 middle 和最终 Down 输出，均通过与原版 CPU 的 `memcmp` 检查。
- FP32 两波融合及四路交错的 middle 和 Down 输出，均通过与原 FP32 consumer 的 `memcmp` 检查。融合不额外改变 FP32 原型的数值。
- 在本次真实输入上，FP32 相对原 FP64 的 middle relative L2 约 `1.754e-8`，最终 Down relative L2 约 `1.127e-8`；统计到的 Down A8 codes 改变比例为零，row scale 有约 `1.019e-7` 的最大相对变化。
- 主要两个进程均退出成功，频率恢复检查通过。

当前验收覆盖这一组真实 layer2、M128、两 K 波 Gate/up 的完整 FFN。它不能直接外推为全模型速度，也不能直接推广到 Down 的六 K 波或任意 M/N 几何。生产默认策略尚未使用此诊断路径。

这次结果支持继续减少 CPU 中间数据：例如在 SwiGLU 输出时累计每行 maxabs，并把完整行就绪后的量化打包与 Down 输入准备衔接。完整行的 scale 依赖仍须保留；该方向不属于本报告已经测得的收益。

## 实现与证据

- [融合内核](../native/rk_npu_matmul/tests/tools/w4a8_cpu_fused2/fused_two.cpp)、[诊断构建](../native/rk_npu_matmul/tests/tools/w4a8_cpu_fused2/build.py)、[板端运行与统计](../native/rk_npu_matmul/tests/tools/w4a8_cpu_fused2/run.py)。工具均在算子库仓库中。
- [主要六模式结果](evidence/w4a8_cpu_finish_20261022/fused-f64-paired/summary.json)、[协议](evidence/w4a8_cpu_finish_20261022/fused-f64-paired/protocol.json)、[原始进程 0 CSV](evidence/w4a8_cpu_finish_20261022/fused-f64-paired/pair-0.csv)、[原始进程 1 CSV](evidence/w4a8_cpu_finish_20261022/fused-f64-paired/pair-1.csv)。同目录保留数值检查日志与 clocks JSON。
- [独立五模式对照](evidence/w4a8_cpu_finish_20261022/fused-paired/summary.json)、[最初四模式 CPU 对照](evidence/w4a8_cpu_finish_20261022/paired/summary.json)。
- [既有真实输入 FP32/NPU SwiGLU 对照](evidence/w4a8_swiglu_npu_20261020/real-m128/protocol.json)、[NPU 尾部归约结果](QWEN3_W4A8_NPU_REDUCE_TAIL_20261022.md)、[两波融合的原始流程分析](QWEN3_W4A8_PIPELINE_REVIEW_20261020.md)。

板端从外层仓库根目录复现：

```bash
python3 native/rk_npu_matmul/tests/tools/w4a8_cpu_fused2/build.py \
  --library native/rk_npu_matmul/build/w4a8-swiglu-npu/library/librk_npu_matmul.a

python3 native/rk_npu_matmul/tests/tools/w4a8_cpu_fused2/run.py \
  <BOARD_HOME>/mnt/3090x4/tmp/w4a8-swiglu-npu-20261020 \
  <BOARD_HOME>/mnt/3090x4/tmp/w4a8-swiglu-npu-20261020/trace128/p0-l2-post.f32 \
  llm_generated_docs/evidence/w4a8_cpu_finish_20261022/replay \
  --cycles 30 --processes 2
```
