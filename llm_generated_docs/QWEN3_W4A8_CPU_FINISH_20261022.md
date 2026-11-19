# Qwen3 W4A8：CPU 收尾对照与可减少的数据访问

2026-10-22。真实 layer2 输入、同一组缓冲与权重的完整单 FFN 测试。

**纯 CPU 路径仍有收益：FP32 SwiGLU 将完整 FFN 从 9.375 降到 9.089 ms；四路交错版为 9.066 ms，较原版快 3.30%。四路交错相对既有 FP32 版本只改善 0.25%，不能把它当作大幅突破。两次 NPU 卸载的负结果没有证明 CPU 已达到硬件上限。**

## 同缓冲对照

复用另一组 FFN 探针的真实 layer2 W4 codes/scales 与捕获输入 `trace128/p0-l2-post.f32`。M128/H2560/F9728；Gate/up Ktile1472，Down Ktile1888；CPU4–7、四线程、2.256 GHz，NPU 三核 1 GHz，DDR 未控制。

所有模式共用 Gate/up executor、DMA 输入与 partial 缓冲、INT32 accumulator、Down workspace 和权重。只切换 Gate/up 的 CPU consumer。包含 Gate/up、SwiGLU、Down 的完整调用；不是完整模型推理。

原库的 CPU output 编译选项为 `-O2`，原有 FP32 探针为 `-O3`，因此补充 FP64/O3 控制。每批两个进程、每进程 30 个正反顺序周期，每种方法合计 120 次调用；逐周期配对保留在原始数据中。

| CPU 收尾 | Gate/up ms | CPU reduce/finish 累计 ms | 完整 FFN ms | 相对原版 |
|---|---:|---:|---:|---:|
| 原版 FP64，O2 | 6.605 | 3.885 | 9.375 | 基线 |
| 相同 FP64，O3 | 6.736 | 4.018 | 9.502 | 慢 1.35% |
| 既有 FP32，O3 | 6.317 | 3.601 | 9.089 | 快 3.06% |
| FP32 四路交错，O3 | 6.300 | 3.582 | 9.066 | 快 3.30% |

四路交错保留原有 degree-7 FP32 多项式、range reduction、精确向量除法与超范围 fallback，每次处理 16 个元素。原 FP64 实现已经交错四条独立 Horner 链；FP32 原型没有相同的显式交错结构。

四路交错相对既有 FP32 的进程内配对时延分别降低 0.342%/0.158%，更快周期为 23/30、16/30，属于小幅变化。单独切换 O3 没有带来收益；不能将原 FP32 探针的改善归给编译等级。

## 数值与另一项算术对照

真实输入、零输入、原输入乘 4 三种模式均完成完整输出比较。四路交错与原 FP32 版本的 middle 和 Down 输出通过逐字节比较。

真实输入上，两者相对原 FP64 基线的 middle 相对 L2 为 `1.754e-8`，Down 相对 L2 为 `1.127e-8`，Down A8 codes 没有改变。这只覆盖本次捕获输入及上述变体，不代表完整模型质量结论。

另外一批将交错版中的向量除法换为 reciprocal estimate 加两次 Newton 修正，结果为：

| 同批比较 | 完整 FFN ms |
|---|---:|
| 原版 FP64/O2 | 9.370 |
| 既有 FP32/O3 | 9.069 |
| 交错 FP32＋倒数迭代 | 9.311 |

倒数迭代比同批 FP32 慢 2.66%，60 个配对周期均未更快。它使真实输入的两个 Down A8 codes 改变，Down 相对 L2 增至 `2.083e-4`，不作为保留的性能候选。

## 大段 CPU 时间不能全部归给 SiLU

另一组真实输入测试的原始 CSV 中，CPU reduce/finish 累计约 3.901 ms；卸载激活后，CPU 仍需约 3.163 ms 完成整数重建、反量化和独立 gate/up 输出。两条路径的布局、融合和写回不同，差值不是纯 SiLU 的算术耗时，也不是其严格上限。

同样，CPU 阶段墙钟时间包含多核计算与汇合、访存等待；缓存同步另有约 1.1 ms。不能凭时间长就判断有多少可通过 SIMD 删除的工作，也不能在没有 PMU/DDR 计数器时认定唯一瓶颈是 DDR 带宽。

但源码可以确定存在以下数据访问量。M128/N19456 下，主要张量的 CPU 逻辑读写为：

| 当前两波路径 | MiB |
|---|---:|
| 第一波 high/low partial 读取 | 9.5 |
| INT32 accumulator 写入 | 9.5 |
| 最终波 partial 读取 | 9.5 |
| INT32 accumulator 读取 | 9.5 |
| FP32 middle 写入 | 4.75 |
| 合计 | 42.75 |

这不含 scale、临时数组、cache-line write allocation 等，不是物理 DDR 流量。SwiGLU 出来的 4.75 MiB middle 随后还会被 Down 的 maxabs 扫描和 quant/pack 各读取一次。

## 下一步的两个结构候选

### 两波 partial 直接融合消费

当前 Gate/up 恰好两波，可在最终收尾直接读取两份原生 INT16 partial，计算 correction 与两个 high/low 重建项，然后保持既有 scale/SwiGLU 顺序。两份 partial 已经各有缓冲。

它能够取消完整 INT32 accumulator 的写回和读取，将上述主要张量逻辑访问从 42.75 减到 23.75 MiB，减少 19 MiB。这个字节数是确定的；时延收益尚未测量。必须保留第一波 partial 的提前同步，并比较失去第一波 CPU 归约与第二波 GEMM 重叠的代价。

历史普通 Linear 的两波合并曾因输出尾部增长而变慢，不能把字节数减少直接换成同比例加速。当前真实 Gate/up 的两波 fused SwiGLU 是单独的验证对象。

### SwiGLU 输出时统计 Down 的逐行 maxabs

在生成 middle 时累计每行最大绝对值，通过很小的线程间归并形成完整行的 scale，可取消 Down 对整个 middle 的独立 maxabs 读取。既有真实输入数据中，Down scan 约 0.221 ms，quant/pack 累计约 0.823 ms。

优先只融合 maxabs，继续保留当前 Down 的逐 K-wave 打包流水。完整行的 maxabs 仍是量化依赖，不能在只生成部分 N 通道时就确定 scale。进一步按小行块在缓存中完成量化打包需要另测：过早准备全部 K wave 也可能失去原有打包与 GEMM 的重叠。

这两个候选减少必须执行的步骤，数值和流水代价都可由同一单 FFN 测试验证；当前没有数十个百分点收益或 3T 的实测承诺。

## 工具与证据

- [CPU 对照构建](../native/rk_npu_matmul/tests/tools/w4a8_cpu_finish/build.py)、[交错内核](../native/rk_npu_matmul/tests/tools/w4a8_cpu_finish/interleaved.h)、[运行和配对统计](../native/rk_npu_matmul/tests/tools/w4a8_cpu_finish/run.py)。
- [同缓冲四模式结果](evidence/w4a8_cpu_finish_20261022/paired/summary.json)、[正确性日志](evidence/w4a8_cpu_finish_20261022/paired/pair-0.log)。
- [倒数迭代对照](evidence/w4a8_cpu_finish_20261022/reciprocal/summary.json)、[对应精度日志](evidence/w4a8_cpu_finish_20261022/reciprocal/pair-0.log)。
- [已有 NPU SwiGLU 原始真实输入 CSV](evidence/w4a8_swiglu_npu_20261020/real-m128/pair-0.csv)、[旧 NPU reduce 尾部实验](QWEN3_W4A8_NPU_REDUCE_TAIL_20261022.md)、[流水分析](QWEN3_W4A8_PIPELINE_REVIEW_20261020.md)。

频率恢复检查全部通过。原有 SwiGLU agent 的工具和生产源码未修改；新增 CPU 探针从生产 output.cpp 与已有 FFN 测试生成独立构建。

板端从外层仓库根目录复现：

```bash
python3 native/rk_npu_matmul/tests/tools/w4a8_cpu_finish/build.py \
  --library native/rk_npu_matmul/build/w4a8-swiglu-npu/library/librk_npu_matmul.a
python3 native/rk_npu_matmul/tests/tools/w4a8_cpu_finish/run.py \
  <BOARD_HOME>/mnt/3090x4/tmp/w4a8-swiglu-npu-20261020 \
  <BOARD_HOME>/mnt/3090x4/tmp/w4a8-swiglu-npu-20261020/trace128/p0-l2-post.f32 \
  llm_generated_docs/evidence/w4a8_cpu_finish_20261022/replay \
  --cycles 30 --processes 2
```
