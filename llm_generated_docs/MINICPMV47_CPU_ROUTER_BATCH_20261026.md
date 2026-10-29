# MiniCPM-V 4.7 CPU router / LES 批处理对照

## 结论

**局部优化有效，尚未测出稳定的端到端提升。** 正常 allocator、ABBA 顺序、每实现两次对照，DS3 从 **7.8394→7.8777 tok/s**，变化 **+0.49%**。每例生成文本、35 轮验证、139 行、93/104 接受、9 行 replay 均相同。本轮没有通过 TF32 截断或路由变化换取速度。

单算子 M=4 完整调用节省约 24%，M=128 节省约 33%；M=1 持平。端到端收益仍以整段配对测量为准。默认源码保留本轮 CPU 批处理，未修改专家算子内部和另一位 agent 的工作。

## 改动与条件

- `Cpu::dense_head_batch` 对整批 row-major logits 只调用一次 `pool.install`，并展开输出任务；每个输出调用原 `dense_dot`。BF16 权重、FP32 累加、NEON reduction、scalar tail 全部保持原样。
- 实际 router 与 LES 共用同一份 logits scratch，按原行序执行原 top-8 / softmax。保持实际需求入队→LES 投影→预测入队→专家执行的时序。
- shared gate、draft、head、GDN、专家计算 kernel 均未修改；未增加 CLI 参数。
- H88K 8 GiB，现有 GM7 NVMe 和转接板；CPU4–7 2.4 GHz、NPU mask7 / 1 GHz、DDR2112 MHz。共享锁独占，每例恢复时钟。
- 同一 noSQ 模型、stream、每层 40 槽、LES、O_DIRECT、2 个 read worker、1 个 import worker；未加载完整专家。
- pp512、NPU ubatch128、tg128，CLI n-predict129 含 prefill 首 token；DSpark draft=3。CPU 逻辑 prefill 的 router 一次处理 512 行，ubatch128 是 NPU 投影/专家执行分块，不能把它说成 CPU router M128。
- native 统一为上一轮 `811b2a3` 加原导入补丁的冻结构建：`/home/firefly/.cache/minicpmv47-native-update-20261026/native/rk_npu_matmul`。基线 `stage_infer`，本轮 `batch_infer`；不使用上一轮 TF32 或 CPU 截断控制二进制。

## 单算子完整调用

真实 router 第 0/10/20/39 层，BF16 [256,2048]，4 线程，输入为确定性合成浮点数，并非实测模型激活。8 次 warmup，7 组×32 次，交替先后顺序；下表为四层均值，包含线程池调度，不含模型加载。20 个实际权重组合全部逐位相同。

| M | 逐行 μs | 批处理 μs | 逐行/批处理 |
|---:|---:|---:|---:|
| 1 | 34.619 | 34.714 | 0.997× |
| 4 | 135.994 | 103.078 | 1.319× |
| 8 | 272.579 | 194.692 | 1.400× |
| 16 | 555.763 | 381.633 | 1.456× |
| 128 | 4464.199 | 2982.977 | 1.497× |

## 正常 allocator 端到端

固定顺序 CPU A→batch B→batch C→CPU D，均为独立进程，空专家热槽起步。表中的 trace 样本另列，不并入正常对照均值。均值是各次 tok/s 的算术平均；总生成时间包含 prepare，排除模型加载。AR 只有新实现一次观察，不作为配对加速结论。

| 样本 | prefill s | decode tok/s | pp+tg s | 峰值 RSS GiB | 最大 VmSwap MiB |
|---|---:|---:|---:|---:|---:|
| batch_ar_0 | 12.208 | 5.3470 | 36.176 | 5.217 | 0.0 |
| batch_ds3_B | 12.975 | 8.2066 | 28.636 | 6.447 | 128.1 |
| batch_ds3_C | 13.642 | 7.5488 | 30.917 | 6.323 | 126.2 |
| batch_ds3_detail_trace | 14.029 | 7.6768 | 30.845 | 6.191 | 137.5 |
| cpu_ds3_A | 13.525 | 7.7655 | 30.119 | 6.273 | 173.7 |
| cpu_ds3_D | 13.355 | 7.9133 | 29.584 | 6.404 | 143.2 |
| cpu_ds3_detail_trace | 14.535 | 7.3748 | 31.998 | 5.981 | 140.0 |

批处理正常样本为 8.2066 和 7.5488 tok/s，波动明显大于约 0.5% 的均值差，不能据此认定稳定加速。慢样本的 draft 为 2619.58 ms，其他正常样本约 2204.84–2263.08 ms；这段没有改动。telemetry 也观察到 decode 期间系统 swap-in，提示内存压力与系统波动，但不足以独立解释所有差异。

DS3 两次均值：prefill **13.440→13.308 s**，pp+tg **29.852→29.776 s**。样本量只有两次/实现；进程存在不同程度 swap，未调整现有 zram/swappiness。峰值 VmSwap 覆盖完整进程，不等于 decode 的换页量。原始 telemetry、物理链路快照、各次结果均保留。

## 配对 trace

沿用原 `RKNPU_EXPERT_TIMELINE`，诊断单独使用 `MALLOC_MMAP_THRESHOLD_=131072`。共享同一时间轴，未开启保存张量的慢 trace。以下除以 128 输出 token，router/LES 和其他 target 阶段只统计 verify span；expert/I/O 统计验证窗口，属于不同层级，不能直接把所有行再相加。

| 阶段 ms/output | 逐行 | 批处理 |
|---|---:|---:|
| target router | 3.909 | 2.647 |
| target LES router | 6.107 | 3.646 |
| target demand bookkeeping | 0.737 | 0.736 |
| target LES enqueue | 0.660 | 0.699 |
| target verify | 111.815 | 106.477 |
| wait record | 24.207 | 21.474 |
| import | 4.726 | 5.085 |
| NPU experts | 27.804 | 27.985 |
| routed orchestration | 7.755 | 7.795 |
| wait 与 pread 重叠 | 16.820 | 14.551 |
| wait 与 import_service 重叠 | 5.770 | 5.306 |

| 流量与缓存 | 逐行 trace | 批处理 trace |
|---|---:|---:|
| 冷需求记录数 | 12926.000 | 12926.000 |
| 必要 cold MB/output | 161.316 | 161.316 |
| 实际读取 MB/output | 200.304 | 202.201 |
| 热缓存 assignment hit % | 65.450 | 65.450 |
| 热缓存 unique hit % | 56.602 | 56.602 |
| LES 冷需求覆盖 % | 65.960 | 65.960 |
| 提前 RAM-ready / cold % | 42.550 | 44.291 |
| 窗口平均 GB/s | 1.490 | 1.569 |

实际 router 的完整专家 ID 序列共 **1440** 次、LES 序列共 **1365** 次，两份 trace 全相同。缓存替换和物理读取仍可能受异步完成顺序影响，所以单算子省时不必等于总延迟差值。`router_trace.json` 给出 M3/M4/M512 的实际调用均值。

## 后续判断

配对 trace 的 router+LES 节省约 3.72 ms/output；record 等待也从 24.21 降至 21.47 ms/output，主线程 import 从 4.73 升至 5.09 ms/output。这轮没有观察到等待增量把路由加速完全抵消，不能套用上一轮 TF32 的因果解释。单次 trace 的总速度提升约 4.1%，正常 allocator 的均值提升仅约 0.5%，两种口径应分开。

40 槽的 cache assignment hit 65.45%、unique hit 56.60%，冷需求记录数和 LES 命中个数保持相同。提前完成路由不会减少每个 token 必须读取的约 161.3 MB 冷载荷。后台导入约 26.6 ms/output，大部分有重叠；不能把它全部再算进关键路径。仍值得优先分析主线程分页整理的约 7.8 ms/output 和必要复制，并控制 DSpark 在 8 GiB 板上的内存/换页波动；专家 kernel 内部保持由另一个任务处理。

## 验证与复现

- ARM release build、all-target Clippy 通过。BF16/F16/F32、尾部形状 [7,19] 与实际 [256,2048]、M1/4/8/16/128 的逐位等价测试通过。
- 20 组真实权重 probe 逐位相同，7 个完整 pp512/tg128 样本全部匹配旧 CPU AR 参考文本，无 speculation fallback。没有重跑 PPL：点积数学不变，逐位单算子和完整路由/LES 序列均已检查；此处不把有限输入测试扩展为全分布质量证明。
- 每例时钟恢复，物理槽位/staging promotion 审计、缓存重建、trace 字节与统计对账全部通过。自动验证总计 55 项。
- 首次启动因 root 所有的共享锁导致权限错误，尚未推理就退出；随后以原 sudo 方式重启。保存首错日志，没有改锁权限。
- 没有 commit、stash、push；保留本轮前的所有 dirty 改动。

证据目录：`llm_generated_docs/evidence/minicpmv47_cpu_router_batch_20261026/`。

- `integration.patch`、`production_sources.tar.gz`、`baseline_cpu.rs`、`baseline_moe.rs`：本轮窄范围改动与基线。
- `router_probe.rs`、`build_batch.sh`、`batch_infer`、`batch_probe_test`、`cpu_batch_build.log`：ARM 构建及实际测量程序。probe hook 只存在于 `compiled_cpu.rs` 的实验构建副本，默认生产文件不依赖 evidence。
- `run_batch_suite.sh`、`run_batch_case.py`、`results/*.json` / `*.log` / `*.output.txt`：完整命令、原始时间、telemetry、trace、生成文本。
- `summary.json`、`probe_summary.json`、`router_trace.json`、`route_equality.json`、`*.analysis.json`、`trace_analysis.json`、`validation.json`：复算、配对与审计；同目录 Python 脚本可复现汇总。
