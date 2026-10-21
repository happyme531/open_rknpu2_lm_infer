# W4A8 Gate/up：真正的N分组流水实验

2026-10-20。**N流水可以实现计算与CPU finish的重叠，但这个形状的完整调用只加速约1.38%。** 不记录事件的同进程交错测试：原始K流水6.731372 ms，两组N流水6.638631 ms；输出逐字节一致。当前保留为算子库测试原型，没有接入生产调度或调优缓存。

对源码和这些原始时间线的[完整流程与依赖复核](QWEN3_W4A8_PIPELINE_REVIEW_20261020.md)进一步确认：首批结果到达后，CPU输出同步和归约构成关键路径。后续潜力主要在同步与消费流水、非最终归约均分及减少accumulator访存；单独提前发出更多NPU任务的延迟余量很小。

## 实现与计时范围

- 仅运行C++单算子。真实Qwen3-4B checkpoint的layer1 Gate/up，原始codes/scales；固定合成FP32输入。
- M128/K2560/N19456，融合SwiGLU，panel8，M tile128，K tile1472，因此K有1472/1088两个wave。N tile4096。
- CPU4–7、4线程，固定2.256 GHz；三NPU核mask7、固定1 GHz；DDR未受控。计时包含完整量化、打包、提交、同步、归约与SwiGLU，不包含权重/workspace创建、频率切换、输出比较和CSV写入。
- 按SwiGLU输出通道把F9728拆成两个4864通道组；每组包含相同通道的gate和up。组内仍均衡使用三核。
- 每次调用只扫描一次全K scale、只打包两个A wave各一次，供所有N组复用；逐事件计数已核验。原始路径每次2个submit，N分组路径每次4个submit。
- 各N组使用独立的原生partial缓冲和累加器，避免CPU读取上一组时与NPU写下一组发生所有权冲突。缓冲、任务链和submit线程均提前创建，测量中不分配tensor缓冲。

两组N流水的执行顺序：

1. 预先打包A的两个K wave，执行N0/K0。
2. NPU执行N0/K1，CPU归约N0/K0。
3. NPU执行N1/K0，CPU完成N0的反量化、输出和SwiGLU。
4. NPU执行N1/K1，CPU归约N1/K0。
5. CPU完成N1输出。

整数MSD重建、correction、scale及FP64多项式SwiGLU继续使用原有实现，只改变数据分组和调度。

## 同进程交错结果

准备原始、分组串行、分组流水三个独立常驻workspace，复用同一权重和输入。每周期依次执行 `baseline, serial, pipeline, pipeline, serial, baseline`，对称位置使三种方法的平均采样时间点一致。两个进程，每进程30周期，每种方法每进程60次调用；每次输出都与原始路径逐元素比较，最终输出文件也逐字节一致。

以下是不记录事件的CSV完整调用时间，每种方法共120次调用；有效工作量每次12.750684160 GOP。

| 方法 | 平均 ms | 中位数 ms | 有效 TOPS |
|---|---:|---:|---:|
| 原始K-wave流水 | 6.731372 | 6.736675 | 1.894 |
| 同样两组N，串行执行 | 7.892892 | 7.894309 | 1.615 |
| 两组N流水 | 6.638631 | 6.633279 | 1.921 |

相对原始K流水，完整时延降低 **1.378%**，吞吐提升约1.397%。相对相同N分组的串行控制，时延降低15.891%，说明分组后的重叠有效；原始K流水本身已经覆盖了一部分设备计算，因此不能把串行控制的较大差距当作已有实现的收益。

| 独立进程 | 原始均值 ms | N流水均值 ms | 周期配对时延降低均值 | N流水更快的周期 |
|---|---:|---:|---:|---:|
| pair-0 | 6.761030 | 6.647852 | 1.673% | 30/30 |
| pair-1 | 6.701714 | 6.629409 | 1.080% | 29/30 |

周期/调用来自同一个进程的连续样本，不能当作独立板卡实验。原始记录保留全部样本，没有删除长尾。

带逐事件插桩的另一组同样交错测试，时延降低1.380%；不记录事件的对照为1.378%，本次小收益没有依赖插桩。

## 为什么尾部减半，完整调用只快约1.4%

以下来自插桩交错测试，每种方法120次调用取平均，单位ms。各行存在包含与重叠关系，不能全部相加。

| 时间区间 | 原始K流水 | 两组N流水 |
|---|---:|---:|
| 完整调用 | 6.737317 | 6.643716 |
| 驱动设备执行区间 | 2.703060 | 3.059944 |
| submit ioctl区间总和 | 2.750459 | 3.161204 |
| submit之外的暴露时间 | 3.986858 | 3.482511 |
| 最后一次submit返回后的尾部 | 3.775084 | 1.881240 |
| CPU归约/finish跨度总和 | 3.989376 | 4.272319 |

N组最终finish与后续submit区间实际重叠 **0.541243 ms**。CPU归约还会与同N组的第二K wave重叠。

按不重叠边界分解：submit之外的时间减少0.504347 ms，但submit区间总和增加0.410745 ms，净减少0.093601 ms。设备区间增长反映了更小的N任务及并发数据流的成本；CPU归约/finish跨度也略增长。最终尾部虽然约减半，CPU总工作量并没有下降，一部分等待转移到了组间。

驱动`hw_elapse_time`包含设备访存、任务链转换和中断延迟，不是纯MAC周期；本实验没有用硬件stall counter进一步区分这些损失。

## 分组数量

2/3/4/6组均通过完整输出对照。探索性单进程分组扫描中，增加组数可以进一步缩短最后尾部，但设备执行区间和CPU跨度逐步增加，2组的完整时间最好。正式交错复核只对2组进行，未把不同进程中的探索性时延直接当作最终收益。

只在最后一个K wave内部沿N流水的变体也完成了输出验证与时间线测量；它保留了整N前一wave的归约依赖，没有稳定超过原始路径。完整N流水同时把后续N组的第一个K wave覆盖到前一组的SwiGLU下，因此更适合本次比较。

## 范围与代码

该原型的完整N模式目前只支持两K wave的FP32融合SwiGLU；测试覆盖上述真实权重形状。没有验证整模型收益，也没有改默认策略。原型保留了原workspace中未使用的缓冲，又新增约28.5 MiB分组partial/累加器及少量控制存储；这些分配均在计时之外，生产化时需要统一内存规划。

- [调度原型](../native/rk_npu_matmul/tests/tools/w4a8_utilization/n_pipeline.cpp)、[声明](../native/rk_npu_matmul/tests/tools/w4a8_utilization/n_pipeline.h)。
- [C++单算子入口](../native/rk_npu_matmul/tests/profile_w4a8_utilization.cpp)、[副本构建](../native/rk_npu_matmul/tests/tools/w4a8_utilization/build.py)。构建只生成诊断源码副本，不修改算子库生产源文件。
- [交错运行工具](../native/rk_npu_matmul/tests/tools/w4a8_utilization/run_n_pairs.py)、[周期统计](../native/rk_npu_matmul/tests/tools/w4a8_utilization/summarize_pairs.py)。
- [不记录事件的协议](evidence/w4a8_utilization_20261020/n-pipeline-unrecorded/protocol.json)、[原始CSV与频率记录](evidence/w4a8_utilization_20261020/n-pipeline-unrecorded/)、[统计JSON](evidence/w4a8_utilization_20261020/n-pipeline-unrecorded/summary.json)。
- [插桩时间线与统计](evidence/w4a8_utilization_20261020/n-pipeline-paired/)、[完整汇总](evidence/w4a8_utilization_20261020/n-pipeline-summary.json)。全部运行退出成功、频率上下限恢复，`restore_errors=[]`。

算子库根目录的板端复现命令：

```bash
python3 tests/tools/w4a8_utilization/build.py --base-archive /path/to/librk_npu_matmul.a
python3 tests/tools/w4a8_utilization/run_n_pairs.py /path/to/exported_weights /path/to/results --groups 2 --cycles 30 --no-profile --runner /path/to/run_board.py
python3 tests/tools/w4a8_utilization/summarize_pairs.py /path/to/results
```
