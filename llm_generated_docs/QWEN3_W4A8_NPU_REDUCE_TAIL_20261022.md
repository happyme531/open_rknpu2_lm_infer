# W4A8 Gate/up：CPU 主体与 NPU 尾部归约

2026-10-22。RK3588 真实板端单算子完整调用测试。

**尾部 NPU reduce 可以与 CPU 主体有效重叠，但复用现有普通 Conv 归约没有加速。关闭事件记录的三组交错测试中，最好的 F2048 尾部仍慢 5.73%。时间线显示，NPU 归约及其输出同步已在 CPU 主体结束前完成；并发期间 CPU 阶段变慢，抵消了减少 CPU 整数归约的收益。**

## 测量范围

真实 Qwen3 混合量化 checkpoint 的 layer1 Gate/up codes/scales，固定合成 FP32 输入；逻辑 M128/K2560/N19456，SwiGLU 输出 F9728。输入不是模型真实 activation，本实验不是完整模型测试。

保留原有 panel8、Mtile128、Ntile4096、Ktile1472，两个 K wave 为 1472/1088。每波六个 GEMM task，三核 NPU，不划出专用归约核。CPU4–7、四个 OpenMP 线程、2.256 GHz；NPU 1 GHz；DDR 未控制。使用共享板锁与频率恢复工具。

尾部 F 表示最终输出的最后 F 个通道，同时包含其 gate/up 两半各 F 列。主体和尾部保持相同的 correction、FP32 scale 乘法顺序与既有 SwiGLU 算法。

## 执行方式

现有 [W4A8NpuReduce](../native/rk_npu_matmul/src/rk_npu_w4a8_npu_reduce.cpp) 使用对角权重 16/1 的普通 INT16 Conv，将 high/low partial 重建为 INT32。原路径在所有 GEMM 结束后归约全部输出，correction、反量化和输出转换仍在 CPU；公共接口也不支持其融合 SwiGLU。

本实验只将一段尾部整数工作交给 NPU：

1. 原来的两波 GEMM 照常执行，仍写原来的两个 native INT16 partial 缓冲。
2. CPU 对主体通道执行第一波整数归约；尾部不写 CPU accumulator。
3. 最后一波 partial 同步完成、CPU 开始主体 SwiGLU 时，唤醒 NPU 尾部 worker。
4. NPU 直接读取两份 partial 的尾部，对每个 N32 gate/up 区间分别做 high/low Conv 归约，再使用已存在的 offline INT32 ADD 配方合并两波。每个有数据依赖的 Conv/Conv/ADD 序列位于同一个核的 PC chain。
5. CPU 同时完成主体 SwiGLU；worker 完成尾部 NPU 结果的缓存同步。
6. CPU 读取一份已合并的 INT32 尾部，完成 correction、scale 与 SwiGLU，然后返回完整输出。

没有运行时权重转换或 CPU partial 复制。原有 GEMM 的形状与 task 数保持相同，增加一次三核归约 submit。F1024 的这次提交包含 192 个硬件 task；F2048 包含 384 个，并非同等数量的 ioctl。

普通 Conv 原路径对大 crop pitch 有已知实验限制。这里直接读取原始两个 BO 的连续 N32 片段，单个 Conv 使用 rows512、单波、不 crop 的配置；两波间通过 offline ADD 连接。没有使用未经验证的 flying INT32 ADD。

对照包含三个模式：

- `baseline`：原来的全 CPU 归约与融合 SwiGLU。
- `reduce_serial`：相同 CPU 主体/NPU 尾部分工，等待 NPU 尾部归约和同步完成，再做 CPU 主体最终输出。
- `reduce_overlap`：NPU 尾部与 CPU 主体最终输出同时进行。

## 完整调用结果

以下为关闭事件记录的主要结果。每种尾部大小各两个进程；每进程 40 个 ABCCBA 周期，每种方法合计 160 次完整调用。配对周期在进程内，不将调用次数当作独立板卡实验数。

| 尾部 F | CPU 基线 ms | NPU 尾部串行 ms | NPU 尾部重叠 ms | 重叠版相对 CPU 基线 |
|---|---:|---:|---:|---:|
| 512 | 6.683 | 7.105 | 7.294 | 慢 9.14% |
| 1024 | 6.668 | 7.176 | 7.261 | 慢 8.89% |
| 2048 | 6.740 | 7.602 | 7.126 | 慢 5.73% |

三组各自的 80 个配对周期中，重叠版均没有快于 CPU 基线的周期。F2048 的重叠确实比同分工串行版本节省约 0.475 ms，但不足以超过原始 CPU 路径。

## 启动开销已经隐藏，CPU 并发阶段仍变慢

用单独的带事件 F1024 测试解释机制，每模式十次调用。其绝对耗时不与上面的无事件批次混作 A/B：

| 阶段 | CPU 基线 ms | NPU 尾部串行 ms | NPU 尾部重叠 ms |
|---|---:|---:|---:|
| CPU 主体第一波整数归约 | 1.074 | 0.964 | 0.956 |
| CPU 主体最终归约/SwiGLU | 2.945 | 2.559 | 2.924 |
| CPU 尾部 correction/scale/SwiGLU | — | 0.275 | 0.278 |
| NPU 尾部阻塞提交区间 | — | 0.542 | 0.742 |
| NPU 尾部输出同步 | — | 0.056 | 0.073 |
| 完整调用 | 6.787 | 7.185 | 6.939 |

重叠版的十个样本中，NPU 尾部提交区间全部落在 CPU 主体最终输出期间；输出同步也全部在主体结束前完成，平均提前约 2.096 ms。CPU 没有因为尾部 NPU 未完成而等待。

但相同主体工作，串行控制约 2.559 ms，重叠时增至 2.924 ms；NPU 尾部自身也从 0.542 增至 0.742 ms。增加的 CPU 阶段耗时抵消了少做整数归约的收益。提前到最后 GEMM 返回就启动 NPU 归约，会把这类变慢移到前一波 CPU 整数归约；后移到 SwiGLU 没有消除它。

把 NPU 提交/同步 worker 单独固定到额外的 A55 CPU3，F2048 小样本仍为 6.742 → 7.079 ms，慢约 5.00%。该项为一进程十个周期、有事件记录，A55 频率未固定；它不是主要性能结果，只说明简单移走 worker 没有逆转排序。

## 为什么只卸载整数归约仍有成本

这个 Conv 映射会写包含空通道的 INT32 中间结果，之后 ADD 再读取并生成紧凑结果。以 F1024 为例，按描述符计算的新增 NPU 逻辑数据访问为：

| 项目 | MiB |
|---|---:|
| 读取两波尾部 high/low partial | 2 |
| Conv 写两波、含填充通道的 INT32 | 4 |
| ADD 读取两波有效 INT32 | 2 |
| ADD 写最终 INT32 | 1 |
| 合计 | 9 |

这是逻辑访问量，不是物理 DDR 总线流量；不含预取、事务放大等。原来的 GEMM partial 写入是共同成本，没有计入新增量。CPU 对原有 partial BO 的整块同步也仍然存在，另有 NPU 结果同步与单独的 CPU 尾部输出阶段。

因此，NPU 空闲时间可以容纳归约执行，但不能消除它新增的数据访问。串行/重叠对照证明并发会改变阶段成本；现有测量没有硬件 stall counter，尚不能进一步判定 DDR、缓存维护、调度等各占多少。

SwiGLU 本身仍在 CPU。若继续研究，应以更少中间写回、更轻的归约映射，或归约与 correction/scale/激活的完整融合为目标；仅把当前 Conv reduce 搬到尾部，没有实测收益。

## 正确性与边界

主要性能测试的零输入、改变输入、重复 workspace 复用以及每个计时调用的完整输出均与原 CPU 路径逐元素一致；各次主要测试的频率恢复检查通过。

普通 INT16 Conv 会读取零权重对应的补齐 C16 平面，需要 partial 后留映射 guard。本实验为其两个 partial BO 各增加 256 KiB guard；这是一项一次性分配，没有每次调用的复制。调试时未留 guard 的 F256 用例曾触发 IOMMU 读取 fault 与驱动自动复位，该数据不纳入性能结论；CPU 恢复 canary 随后通过，补上 guard 后的小尾部、主要测试及后续测试未再出现新的驱动 fault。

本次修改均位于算子库测试工具及其生成的诊断构建中，生产源码和默认策略未改。

## 工具与证据

- [尾部实现](../native/rk_npu_matmul/tests/tools/w4a8_utilization/npu_tail_reduce.cpp)、[单算子入口](../native/rk_npu_matmul/tests/profile_w4a8_utilization.cpp)。
- [运行工具](../native/rk_npu_matmul/tests/tools/w4a8_utilization/run_reduce_tail.py)、[时间线复算](../native/rk_npu_matmul/tests/tools/w4a8_utilization/analyze_reduce_tail.py)、[诊断构建](../native/rk_npu_matmul/tests/tools/w4a8_utilization/build.py)。
- 无事件交错测试：[F512](evidence/w4a8_npu_tail_reduce_20261022/paired-f512/summary.json)、[F1024](evidence/w4a8_npu_tail_reduce_20261022/paired-f1024/summary.json)、[F2048](evidence/w4a8_npu_tail_reduce_20261022/paired-f2048/summary.json)。目录内保留原始 CSV、clock report、协议与日志。
- [F1024 阶段分析](evidence/w4a8_npu_tail_reduce_20261022/deferred-f1024/tail-analysis.json)、[A55 worker 对照](evidence/w4a8_npu_tail_reduce_20261022/little-f2048/tail-analysis.json)。
- [CPU 恢复 canary](evidence/w4a8_npu_tail_reduce_20261022/recovery-clocks.json)、[guard 用例](evidence/w4a8_npu_tail_reduce_20261022/guarded-f256/pair-0/single-clocks.json)。
- 原有机制与历史结果：[CPU/NPU 流水复核](QWEN3_W4A8_PIPELINE_REVIEW_20261020.md)、[lowlevel NPU reduce 测量](../../lowlevel/docs/w4a8_npu_reduce_2026-09-20.md)、[2+1 分核实验](../../lowlevel/docs/w4a8_split_reduce_2026-09-22.md)。

诊断 archive 使用原有实验的冻结 base archive，替换当前源码生成的 W4A8、I4 plan/executor、core、output 和新增实验对象；同一二进制内的全部比较模式使用相同构建选项。

板端从外层仓库根目录复现：

```bash
python3 native/rk_npu_matmul/tests/tools/w4a8_utilization/build.py \
  --base-archive llm_generated_docs/evidence/w4a8_utilization_20261020/librk_npu_matmul.a \
  --output-dir native/rk_npu_matmul/build/w4a8-npu-tail-reduce

python3 native/rk_npu_matmul/tests/tools/w4a8_utilization/run_reduce_tail.py \
  <BOARD_HOME>/mnt/3090x4/tmp/w4a8-utilization-20261020 \
  llm_generated_docs/evidence/w4a8_npu_tail_reduce_20261022/replay-f2048 \
  --runner scripts/run_board.py --tail-f 2048 --cycles 40 --processes 2 --no-profile
```
