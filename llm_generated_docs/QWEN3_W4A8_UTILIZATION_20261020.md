# Qwen3 W4A8：设备空档与尾块实验

2026-10-20。

**主要原因是CPU最终归约/反量化/融合finish造成的设备空档；块内执行效率也有损失。** 已有正式样本的W4调用合计1871.074 ms，驱动设备区间943.930 ms，完整有效算力1.943T，设备区间内折算3.852T。没有任何submit执行的时间为885.560 ms，其中最后一次submit返回后的尾部为730.877 ms，两个submit之间仅24.654 ms。这个比例只覆盖W4调用。

后续实验全部使用算子库内的C++单算子探针，没有继续运行整模型测试。

## 单算子协议

- 备用1，CPU4–7、4线程；NPU三核mask7固定1 GHz；DDR未受控。
- 真实Qwen3-4B混合量化checkpoint：Gate/up和O取layer1，Down取W4的layer2；原始codes/scales不改、不重新量化。
- 固定合成FP32激活；逻辑M128，panel8，pipeline1。Gate/up K2560/N19456，O K4096/N2560，Down K9728/N2560。Ktile按真实权重的安全界限选择。
- Gate/up的M/K/N tile为128/1472/4096；O为128/1536/896；Down为128/1888/896。
- CPU对照按2.256/1.2/1.2/2.256 GHz执行，每轮预热后测5次，每频率10个调用。
- M及尾N实验在CPU2.256 GHz下，每配置2轮，每轮预热后测10次；基线置于实验前后。各配置输出文件逐字节相同。所有测试频率均恢复，`restore_errors=[]`。

## 原因：CPU尾部与块内效率都有损失

下表为CPU频率对照中2.256 GHz的10次调用均值。TOPS按有效操作量除以平均时间计算。

| 单算子 | 完整调用 ms | 驱动设备区间 ms | 最终CPU尾部 ms | 完整有效T | 设备区间内有效T |
|---|---:|---:|---:|---:|---:|
| Gate/up普通输出 | 7.200 | 2.615 | 4.255 | 1.771 | 4.877 |
| Gate/up＋SwiGLU | 8.683 | 2.613 | 5.704 | 1.468 | 4.881 |
| O | 1.561 | 0.820 | 0.359 | 1.719 | 3.273 |
| Down | 3.132 | 1.841 | 0.290 | 2.036 | 3.463 |

Gate/up设备执行期间已经接近4.9T；完整调用只有约1.5T，主要是最终finish没有被覆盖。O、Down的设备区间吞吐只有约3.3–3.5T，块内效率也需要改善。

CPU降至1.2 GHz、NPU仍为1 GHz时，融合Gate/up完整调用从8.683增至11.658 ms，而设备区间为2.613/2.614 ms，几乎不变。增加的时间在CPU阶段和最终收尾，支持上述归因。

驱动`hw_elapse_time`是提交到最后完成中断的区间，包含DMA、访存、PC chain转换、流水停顿和中断延迟；不是纯MAC周期。所有submit都有三核任务。

## 缩小M tile

Gate/up只有两个K wave，本实验在两者中都改变M tile；逻辑M仍为128。下表各列均为调用中位数；两次M128基线合并为40个样本，其余配置各20个样本。

| M tile | 完整调用 ms | 驱动设备区间 ms | 最终CPU尾部 ms |
|---|---:|---:|---:|
| 128 | 7.303 | 2.611 | 4.378 |
| 64 | 7.506 | 3.107 | 4.108 |
| 32 | 8.576 | 4.411 | 3.832 |
| 16 | 11.705 | 7.730 | 3.557 |

M越小，尾部能略微缩短，但任务数和设备区间耗时明显增加，完整调用没有稳定收益。M64的均值曾略低于基线，但中位数更慢；原始长尾样本全部保留。

## 只缩小尾块N，前面的4096列大块不变

在诊断库副本里修改每核N切分，只拆最后一块，保持逻辑M128/N19456、M tile128、Ktile1472和前面的4096列块不变。三个核的N分片分别为6528、6464、6464列：

- 原始：核0为4096＋2432；核1/2为4096＋2368。
- 尾部二分：核0为4096＋1216＋1216；核1/2为4096＋1216＋1152。
- 尾部三分：核0为4096＋832＋832＋768；核1/2为4096＋832＋768＋768。

每个K wave的task数从每核2个增加为3个或4个。实际tile序列记录在`profile.n_tile`事件中。两次基线合并为40个调用，其余各20个。下表各列均为中位数，第一列用核0表示。

| 每核N分块 | 完整调用 ms | 驱动设备区间 ms | 最终CPU尾部 ms |
|---|---:|---:|---:|
| 4096＋2432 | 7.152 | 2.598 | 4.255 |
| 4096＋1216＋1216 | 7.276 | 2.668 | 4.336 |
| 4096＋832＋832＋768 | 7.318 | 2.744 | 4.308 |

**只缩尾块N也没有降低等待或完整耗时。** 当前执行器一次提交整个wave的PC chain，等整个wave完成后才消费结果；CPU最终仍要处理全部M×N。把PC chain内部的尾task切窄，不会提前CPU消费的时间点，只增加了任务转换。

如果要让更细的尾块覆盖CPU收尾，需要改变提交/消费粒度，使某些结果完成后可被CPU消费，并与后续NPU任务重叠。本实验仅改变内部tile几何，没有实现这种执行时序。

后续[真正的N分组流水实验](QWEN3_W4A8_N_PIPELINE_20261020.md)已实现并验证输出：两组N流水在同进程交错、无事件记录的测试中，将该Gate/up融合完整调用从6.731 ms降至6.639 ms，约快1.38%。尾部约减半，但小块设备耗时和CPU处理跨度增加，净收益较小；原型保留在算子库测试目录，未接入默认策略。

## 工具与数据

- C++入口：[profile_w4a8_utilization.cpp](../native/rk_npu_matmul/tests/profile_w4a8_utilization.cpp)。
- 工具目录：[w4a8_utilization](../native/rk_npu_matmul/tests/tools/w4a8_utilization/)；包含真实权重导出、诊断副本构建、tile对照及时间区间分析。生产库源码和主项目Rust入口均未修改。
- [单算子CSV](evidence/w4a8_utilization_20261020/single.csv)、[逐事件原始时间线](evidence/w4a8_utilization_20261020/single-events.json)、[单算子分析](evidence/w4a8_utilization_20261020/single-analysis.json)。
- [M实验](evidence/w4a8_utilization_20261020/m-tiles/)、[尾N实验协议](evidence/w4a8_utilization_20261020/tail-n/protocol.json)、[尾N实验数据](evidence/w4a8_utilization_20261020/tail-n/)。每轮目录保留完整CSV、事件、输出和频率恢复记录。
- [已有模型调用的时间归因](evidence/w4a8_utilization_20261020/analysis.json)。模型真实中间激活与单算子固定合成激活的计时分别报告，不按比例混用。

构建及复现（在算子库仓库根目录、板端执行；先用`export_weights.py`导出权重）：

```bash
python3 tests/tools/w4a8_utilization/build.py --base-archive /path/to/librk_npu_matmul.a
python3 tests/tools/w4a8_utilization/run_tiles.py /path/to/exported_weights /path/to/results --axis tail-n --runner /path/to/run_board.py
python3 tests/tools/w4a8_utilization/analyze.py /path/to/results/tail-nt2
```

`build.py`只在`build/w4a8-utilization/source/`生成插桩和尾块切分副本，再替换诊断archive中的相关对象。真实权重导出物放在共享模型盘临时目录，不进入算子库源码。
