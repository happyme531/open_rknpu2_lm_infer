# Qwen3 FFN down：关键路径与 3T 预算

2026-10-20。停止新增板端测试后，复核已有单算子时间线与源码。

**Down 的主要调度限制是长 K、多 wave 下的 CPU 输入准备与 partial 处理。最终尾部只占约 9%；CPU 控制路径上串行经过的阶段却占约 90%。NPU 块内有效吞吐也低于峰值，但不能把这段与 CPU 大量重叠的设备时间，直接算成可独立消除的延迟。**

## 形状与范围

真实 Qwen3 混合量化 checkpoint 的 layer2 Down codes/scales，固定合成 FP32 输入。M128/K9728/N2560，panel8，M/K/N tile 为 128/1888/896，K 分成 `1888 × 5 + 288`。CPU4–7、四线程、2.256 GHz；NPU 三核、1 GHz；DDR 未受控。

每次有效工作量 6.37534208 GOP。没有 SwiGLU，输出是 FP32 `[128,2560]`。

本文区分两批数据：带事件的样本用于解释时间线；无事件记录的同进程交错测试用于报告完整调用性能，不将不同批次绝对耗时混成一个 A/B。

## 完整调用的时间分解

以 `down-cpu-smoke` 中十个原始基线调用为例，平均 3.000 ms。下面的区间互不重叠，可以相加：

| 区间 | ms | 占完整调用 |
|---|---:|---:|
| NPU 驱动设备执行区间 | 1.943 | 64.8% |
| submit 区间内其余开销 | 0.118 | 3.9% |
| 第一次提交前的准备 | 0.445 | 14.8% |
| 相邻 submit 之间空档 | 0.235 | 7.8% |
| 最后 submit 返回后的尾部 | 0.259 | 8.6% |

末尾 0.259 ms 中，最终输出同步约 0.046 ms，最终归约/反量化/布局转换约 0.196 ms，前一波 CPU 工作越过最后提交返回的部分约 0.007 ms，其余是交接等开销。

另外两组既有 baseline 时间线分别为 2.940/3.087 ms，末尾为 0.242/0.282 ms，结论一致。即便完全消掉最后 CPU 收尾，也无法解释到 6T 的大部分差距。

## CPU 在主干波次已经来不及准备下一次提交

[I4Executor::run](../native/rk_npu_matmul/src/rk_npu_i4_executor.cpp) 当前每波的顺序是：

```text
启动 NPU(w)
CPU pack(w+1)
CPU 同步 partial(w-1)
CPU reduce(w-1)
等当前提交完成，再发 w+1
```

后三个 CPU 阶段在同一控制线程上串行。partial 只有两个槽，前一份结果还没消费完时，下一个 NPU wave 不能覆盖它。

原始基线的两个典型中间波如下；波号从 0 开始：

| 当前 wave | 打包下一波 ms | 同步上一波 ms | 归约上一波 ms | CPU 三段合计 ms | 当前 submit ms | CPU 工作越过 submit 返回 ms |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.205 | 0.104 | 0.146 | 0.455 | 0.383 | 0.069 |
| 3 | 0.255 | 0.080 | 0.133 | 0.468 | 0.364 | 0.100 |

下一次 submit 的实际空档分别为 0.077/0.107 ms，大部分由 CPU 工作未完成解释。全部中间空档 0.235 ms 中，约 0.175 ms 可以直接归到这类 CPU 延迟，剩余约 0.060 ms 为交接等开销。

还可以从 CPU 控制路径本身看：

| CPU 阶段，墙钟跨度 | 累计 ms |
|---|---:|
| 全 K activation maxabs 扫描 | 0.287 |
| 六波 quant/pack，含输入 clean | 1.052 |
| 六份输出 partial 同步 | 0.510 |
| 全部整数归约及最终输出 | 0.847 |
| 合计 | **2.697** |
| 控制线程在 submit wait 中停留 | 0.265 |
| 其余控制开销 | 0.039 |

已逐事件检查前四类阶段都在同一控制线程上、互不重叠。它们是该线程经历的执行/同步跨度，不是四个 CPU 核的 core-ms，也不意味着同步期间都在执行算术指令。

所以，约 3 ms 的调用中，CPU 串行路径本身已约 2.70 ms。保持这些阶段耗时不变，单独令 NPU 更快，也无法把完整调用压到 2.1 ms。改变并发后缓存和访存成本可能变化，因此这是一项调度预算，不是不可突破的硬件上限。

## 与 Gate/up 不同的根源

| 数据形状 | Gate/up＋SwiGLU | Down |
|---|---:|---:|
| K | 2560 | 9728 |
| GEMM N | 19456 | 2560 |
| 每次 FP32 输入 | 1.25 MiB | 4.75 MiB |
| K wave 数 | 2 | 6 |
| 每份 INT16 high/low partial | 9.5 MiB | 1.25 MiB |
| 末尾是否有 SwiGLU | 有 | 无 |

Down 输入更长、输出更窄。全 K 扫描与量化要处理约 3.8 倍的输入元素，但 GEMM 有效工作量只有上述 Gate/up 的一半，因此输入准备在完整调用中更重。

当前 [quant.cpp](../native/rk_npu_matmul/src/rk_npu_w4a8_quant.cpp) 先扫描全部输入，再按 wave 量化和打包。逻辑上两次读取 FP32 输入合计 9.5 MiB，还要写 native INT4 输入、做缓存同步。另一方面，六波各写一份完整 M×N partial，K 变短不会让该份输出变小；CPU 随后反复读取 partial 和更新 INT32 accumulator。

已有 CPU 频率 A/B 也支持 CPU 路径的重要性。这是较早那组同协议实验，不与本轮绝对耗时混合：CPU 从 2.256 降到 1.2 GHz 时，Down 完整调用从 **3.132 增至 5.483 ms**，而设备区间仅 **1.841 → 1.899 ms**。显著增加的是前置准备、中间空档和 CPU 收尾。

## NPU 块内效率仍有损失，但尚不能归因到某一个硬件 stall

十个原始基线样本中：

- 整体设备区间折算有效吞吐约 **3.28T**。
- 前五个完整 K1888 wave 合计约 **3.44T**；单波约 3.1–3.7T。
- 最后一波 K288 约 **1.31T**，但只占约 2.96% 的 MAC 工作量。

因此低吞吐并非仅由最后一个小 K 波造成。当前 Down 每核 N 只有 896/832/832；Gate/up 每核包含更宽的 N tile。按 [当前寄存器生成代码](../native/rk_npu_matmul/src/rk_npu_i4_regs.cpp) 与 [panel 布局计算](../native/rk_npu_matmul/src/rk_npu_i4_layout.h)，Down 的物理 M256/K1888 使用 8 个 data bank、4 个 weight bank；Gate/up 的第一波 K1472 为 6/6。这些是不同的数据复用和片上缓冲条件。

访存、CBUF 分配、任务启动/排空及 CPU/NPU 并发争用都是合理调查方向，但现有时间线没有硬件 stall counter，不能据此认定唯一原因是 DDR 带宽或某个 bank 配置。`hw_elapse_time` 本身包含设备访存、任务转换和完成中断延迟，并非纯 MAC 周期。

## 已完成改动的结果

以下实验均在停止指令之前完成。保留在算子库测试工具及诊断构建中，未改生产默认策略。

CPU 与尾 K 实验各两个进程、每进程 40 个 ABCCBA 周期，每种方法合计 160 次完整调用；性能表关闭逐事件记录。样本来自进程内循环，不是 160 次独立板卡实验。零输入、改变输入和每次测量的完整输出均通过逐元素对照，频率恢复成功。

| 同批比较 | 基线 ms | 改动后 ms | 完整时延变化 |
|---|---:|---:|---:|
| CPU 非最终归约均分 | 2.881 | 2.880 | −0.03%，无稳定收益 |
| 均分＋输出同步重叠 | 2.881 | 2.799 | **−2.83%** |
| 最后 K288 改 W8，预展开权重 | 2.873 | 2.936 | +2.19% |
| 最后 K288 改 W8，每次动态展开 | 2.873 | 2.977 | +3.62% |

同步重叠在两个进程分别降低约 3.36%/2.27% 的周期配对时延，80 个配对周期中 73 个更快。它把部分同步移到下一波打包期间，并尝试在前一波 CPU 归约时同步最后一份已完成的 partial。这里尚未单独分离“同步重叠而不均分”的收益。

两组 N 流水的小样本对照为 2.938 → 3.261 ms，慢约 10.96%；同 N 分组串行控制为 4.522 ms。各 N 组复用同一次 scale 扫描和每波 A 打包，仍因更小的 N 任务、更多提交/阶段交接而增加成本。该项仅一个进程、五个周期，不作为精确性能幅度的最终估计。

尾 K W8 路径仍需与前五波 accumulator 相加、应用固定 correction 调整以及反量化/输出转换；它不能消除这张完整输出的处理。其展开权重为 0.703 MiB。预展开版本已经变慢，说明权重转换并非该原型没有收益的唯一原因。

## CPU kernel 优化与接近 3T 的可行性

工程上，扫描、量化、打包和整数归约比实现新的 NPU 激活映射更直接，但当前实现已经使用 NEON 和四线程，不能把“加 SIMD”当作尚未使用的简单收益。

完整调用的目标时间为：

| 有效吞吐目标 | 所需完整调用时间 |
|---|---:|
| 2.5T | 2.550 ms |
| 2.6T | 2.452 ms |
| 3.0T | 2.125 ms |

当前无插桩基线约 2.881 ms，即约 2.21T；同步重叠原型约 2.799 ms，即约 2.28T。达到 3T 还需在该原型上减少约 24.1% 的完整时延。

按上述时间线保持当前设备成本不变，设备区间约 1.943 ms，所有 submit 区间累计约 2.061 ms；3T 的 2.125 ms 预算只剩约 **0.064 ms** 留给 submit 以外的启动、间隙和收尾。当前前置准备与最终尾部就合计约 0.704 ms。因此只优化中间整数归约，或只消掉中间空档，都不足以达到 3T。

CPU 优化首先应关注最大项 `maxabs scan + quant/pack` 的数据访问和阶段组织，其次是输出同步与消费。按行衔接完整 K 扫描和首波打包，有机会减少并行区域交接、改善局部性，但全 K scale 的数学依赖仍在，不能说扫描可以直接省去。进一步的 kernel 优化可能减少并发争用，从而同时改善设备区间；这部分尚未测量。

2.5–2.6T 可以作为下一阶段的验证目标，尚无实测保证。接近 3T 需要输入准备、输出处理和设备效率共同接近预算，或者把启动/收尾跨独立调用覆盖，不能仅凭现有 CPU 阶段跨度承诺。

## 证据与复算

- [逐波分析 JSON](evidence/w4a8_utilization_20261020/down-flow-analysis.json)，保留三组既有 baseline 的逐调用/逐波数据。
- [仅读取文件的分析工具](../native/rk_npu_matmul/tests/tools/w4a8_utilization/analyze_down_flow.py)。
- [CPU 无插桩交错测试](evidence/w4a8_utilization_20261020/down-cpu-unrecorded/summary.json)、[尾 K W8 无插桩交错测试](evidence/w4a8_utilization_20261020/down-ktail-unrecorded/summary.json)。
- [CPU 时间线](evidence/w4a8_utilization_20261020/down-cpu-smoke/pair-0/)、[N 分组时间线](evidence/w4a8_utilization_20261020/down-n-smoke/pair-0/)、[早期 CPU 频率 A/B](evidence/w4a8_utilization_20261020/single-analysis.json)。
- [单算子入口](../native/rk_npu_matmul/tests/profile_w4a8_utilization.cpp)、[运行协议工具](../native/rk_npu_matmul/tests/tools/w4a8_utilization/run_down.py)、[诊断构建](../native/rk_npu_matmul/tests/tools/w4a8_utilization/build.py)。

复核本报告时只需分析已有文件，无需再上板：

```bash
python3 native/rk_npu_matmul/tests/tools/w4a8_utilization/analyze_down_flow.py \
  llm_generated_docs/evidence/w4a8_utilization_20261020/down-cpu-smoke/pair-0 \
  llm_generated_docs/evidence/w4a8_utilization_20261020/down-ktail-smoke/pair-0 \
  llm_generated_docs/evidence/w4a8_utilization_20261020/down-n-smoke/pair-0 \
  --output llm_generated_docs/evidence/w4a8_utilization_20261020/down-flow-analysis.json
```
