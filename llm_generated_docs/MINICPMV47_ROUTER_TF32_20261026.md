# MiniCPM-V 4.7 router / LES 的 CPU 与 NPU TF32 对照

## 结论

**当前 DS3 流式 decode 不建议默认切到这版 NPU TF32 router。** 两轮正常 allocator 对照中，CPU FP32 为 7.9302 tok/s，NPU TF32 为 7.7196 tok/s，低约 2.66%。普通 AR 单次对照为 5.3646→5.1778 tok/s。PPL 在相同 2040 个目标 token 上从 6.75969→6.81353，增加 0.80%。

这不是 NPU 算不动 router：热缓存单算子 M=8/16/128 更快，M=128 约 2.92 倍；但 M=1 明显慢，M=4 接近 CPU。真实 DS3 trace 中 router+LES 从约 9.93 降到 4.75 ms/output，record 等待却从约 22.60 增到 28.29 ms/output。原 CPU 工作有读取重叠；同时数值变化增加了验证工作和冷专家流量，局部加速没有变成整体加速。

**默认源码已恢复到本轮前的 CPU 实现。** NPU 原型、完整源码快照、可应用补丁、二进制及结果都保存在本报告的证据目录。没有修改专家算子内部、原流式优化、模型文件或另一个任务的改动，没有新增默认 CLI 开关。

## 条件与实现

- H88K 8 GiB、GM7 NVMe、现有 PCIe Gen3×4 和转接板；内核、挂载、zram、swappiness 保持原状。
- 固定 CPU4–7 2.4 GHz、NPU mask7 / 1 GHz、DDR2112 MHz；独占原 NPU 锁，每例恢复频率。
- 相同 noSQ 模型、stream、每层 40 槽、LES、O_DIRECT、2 个 read worker、1 个 import worker，绝不加载全量专家。
- pp512、ubatch128、tg128；CLI n-predict129 包含 prefill 产生的首 token。正常对照不启用 trace；诊断 trace 沿用原 allocator 环境变量。
- 所有构建固定使用上一轮 `811b2a3` 加原导入补丁的 native 快照：`/home/firefly/.cache/minicpmv47-native-update-20261026/native/rk_npu_matmul`。不混入并行进行的专家算子修改。
- 原型复用 Ling3tiny 的 raw TF32×TF32→FP32 ABI，使用常驻 packed-B、按 M 复用 workspace。router 的 BF16 权重先无损展开至 FP32 container；top-8、softmax、入队仍在 CPU。
- 40 份 router packed-B 总计 **80 MiB**。实际 router 与 LES 通过 `Rc` 引用同一份权重，没有为 39 个 LES 再存一套。权重计入 native weight payload；实际差额为 83886080 bytes。
- 原型按 ubatch128 分块计算 prefill router，保留完整逻辑路由集合和专家加载策略；decode/verify 按实际行数执行。未改 shared gate、GDN 或专家 kernel。

## 单算子完整调用

真实层 0/10/20/39 的 BF16 router，K=2048、N=256；输入是确定性的合成浮点数，尺度接近归一化激活，并非实测模型激活。CPU 为现行逐行 `dense_head`；NPU 包含 A packing、sync、submit、输出 unpack。权重及 workspace 已预热，3 次 warmup，5 组×32 次，交替先后顺序。下表为四层均值，NPU 使用 mask7。

| M | CPU μs/call | NPU TF32 μs/call | CPU/NPU |
|---:|---:|---:|---:|
| 1 | 37.22 | 142.57 | 0.26× |
| 4 | 134.11 | 147.74 | 0.91× |
| 8 | 285.78 | 174.28 | 1.64× |
| 16 | 545.68 | 199.43 | 2.74× |
| 128 | 4259.72 | 1460.74 | 2.92× |

mask1 的原始结果也保留在 `router_probe.json`；本形状下 mask7 更快。不能用大 M 的吞吐推断 M=1/4 的延迟。

40 个 layer/mask/M 组合均通过 TF32 截断参考校验，最大绝对误差 2.56e-6。1256 行的 top-8 集合全一致，顺序完全一致 1252 行；与 CPU FP32 相比最大 NRMSE 为 0.0632%。这些输入上的 top-8 一致并不保证整模型 PPL 或生成文本一致。

## 正常 allocator 端到端

CPU 与 NPU DS3 各两次，顺序为 CPU A、NPU B、NPU C、CPU D；中间另有 AR/PPL 正确性测量。均为独立进程、空专家热槽起步。CPU 截断控制只有一次，不并入原 CPU 的均值。

| 样本 | prefill s | decode tok/s | pp+tg s | 峰值 RSS GiB | 最大 VmSwap MiB |
|---|---:|---:|---:|---:|---:|
| cpu_ar_0 | 12.937 | 5.3646 | 36.826 | 5.233 | 0.0 |
| cpu_ds3_a | 14.396 | 7.9096 | 30.733 | 6.085 | 208.7 |
| cpu_ds3_d | 13.071 | 7.9508 | 29.232 | 6.380 | 126.5 |
| cpu_tf32_ds3_x | 13.393 | 8.1076 | 29.301 | 5.953 | 173.4 |
| npu_ar_0 | 12.126 | 5.1778 | 36.875 | 5.270 | 0.0 |
| npu_ds3_b | 13.154 | 7.7367 | 29.874 | 6.018 | 160.8 |
| npu_ds3_c | 14.192 | 7.7024 | 31.346 | 6.253 | 168.8 |

DS3 均值：CPU **7.9302**，NPU **7.7196 tok/s**，变化 **-2.66%**。pp+tg 总时间分别为 29.982、30.610 秒。DS3 prefill 均值几乎相同，单次波动超过 1 秒；不能据此宣称稳定 prefill 加速。AR 单次 prefill 从 12.937 降至 12.126 秒，属于单次观察。

### 投机工作量与文本

| 路由实现 | 正常 DS3 样本数 | 轮数 | verify 行数 | 接受率 |
|---|---:|---:|---:|---:|
| CPU FP32 | 2 | 35 | 139 | 89.423% |
| NPU TF32 | 2 | 36 | 143 | 85.981% |
| CPU、输入截断至 TF32 | 1 | 35 | 139 | 89.423% |

NPU 的两个正常 DS3 和诊断 trace 都匹配 NPU AR 的生成文本。NPU AR 与原 CPU AR 的文本不同。CPU 截断控制在这个提示词上也生成相同的 NPU AR 文本，但 draft 特征/建议和接受前缀不同。因此，表中 E2E 差值同时包含硬件调用、路由变化、接受路径和读取量，不能当作纯 matmul 加速比。

## PPL 与同精度消融

相同八个不重叠 512-token WikiText 窗口，逐窗清空 KV，评分目标位置 257..511，共 2040 个目标；不添加 BOS/EOS 或 chat template。三份报告的 protocol 字段完全相同。时间只覆盖评分循环，排除模型加载。

| 路由实现 | PPL | 相对 CPU FP32 | 平均 NLL | 评分循环 s |
|---|---:|---:|---:|---:|
| CPU FP32 | 6.75968795 | +0.000% | 1.91097673 | 122.228 |
| NPU TF32 | 6.81352711 | +0.796% | 1.91890992 | 114.105 |
| CPU FP32，router 输入截断至 TF32 | 6.83945063 | +1.180% | 1.92270741 | 121.971 |

CPU 截断控制只对实际 router 和 LES 的输入执行 `bits & !0x1fff`，仍用原 BF16 CPU dot，专家输入保持原值。它也出现 PPL 退步，支持 TF32 截断本身会改变本模型结果。

CPU 截断与 NPU 并非逐位等价：FP32 累加顺序仍不同，整模型输出及接受路径可以继续分岔。因此这个控制是精度消融，不是完全等价的硬件速度对照。2040 个目标是有限样本；这里报告实际变化，不宣称质量中性，也不推断所有数据上的固定损失。

## trace 中时间去了哪里

单位均为 ms/输出 token，除以 128。原 CPU FP32 是上一轮拆分 trace，本轮为 CPU 截断与 NPU TF32 各一份。它们的验证工作量、路由和缓存轨迹不完全相同，表内用于定位量级和重叠关系，不是严格因果相减。

| 阶段 | 原 CPU FP32 | CPU 截断控制 | NPU TF32 |
|---|---:|---:|---:|
| 实际 router | 3.854 | 3.932 | 2.414 |
| LES router | 6.074 | 6.180 | 2.334 |
| router + LES | 9.928 | 10.112 | 4.748 |
| record 等待 | 22.602 | 18.468 | 28.291 |
| 主线程 import/swap | 4.455 | 4.829 | 5.824 |
| 专家 complete-call | 27.669 | 27.886 | 28.661 |
| 等待与 pread 重叠 | 15.490 | 12.144 | 19.916 |
| 等待与后台导入重叠 | 5.396 | 4.803 | 6.481 |

原 CPU router/LES 的一部分执行时间与 SSD 读取重叠。把这一段加速后，主线程更早进入冷记录消费，原来重叠的部分可转成显式等待；这与当前 trace 结果一致。NPU 同时还改变了实际路由和投机工作量，不能把等待增量全部归因于这一种机制。

| 流量/缓存指标 | 原 CPU FP32 | CPU 截断控制 | NPU TF32 |
|---|---:|---:|---:|
| 必要 cold MB/output | 161.316 | 162.914 | 169.453 |
| 实际读取 MB/output | 201.028 | 204.772 | 209.215 |
| 冷需求记录数 | 12926.000 | 13054.000 | 13578.000 |
| 提前 RAM-ready 的 cold 占比 % | 42.890 | 46.239 | 41.015 |
| 窗口平均 GB/s | 1.525 | 1.574 | 1.524 |

NPU 的必要 cold 载荷约 169.45 MB/output，原 CPU 约 161.32；实际读取约 209.21 对 201.03 MB/output。局部路由投影快了一倍，整个 I/O 工作量却没有保持相同。减少 CPU 计时不能直接等价为缩短关键路径。

## 后续方向与当前源码

先保留 CPU FP32 router。下一步应测每个 M 一次提交 CPU 线程池，替换逐行 `pool.install`，并保留原 dot 的累加顺序；同时继续分析分页数据整理和必要复制。任何收益仍应通过完整流水线配对验证，不能只加总独立算子时间。

这版通用 raw TF32 路径对大 M 有价值，但 M≤4 的完整调用没有显著优势。更高效的 NPU 提交、router/LES 联合调度可能改变结论，当前实验没有实现或验证这些方案。没有通过量化 router、改 top-k 或丢弃专家来换取速度。

NPU 原型已从默认源码撤回，恢复前后逐文件比较了本轮涉及的五个原文件；此前的 trace/I/O 改动和其他 dirty work 保留。没有执行 git stash、commit 或 push。

## 验证和限制

- ARM release 编译及 all-target Clippy 通过；40 个实际权重 shape/mask 组合完成。
- 9 个完整 pp512/tg128 样本、3 份 2040-target PPL 全部正常退出；每例时钟恢复，无 fallback。
- 两份本轮 trace 共 417920 次物理槽位检查、21979 次 promotion 检查，缓存重建和读取计数均无违规。
- NPU DS3 与其 AR 参考文本相同；CPU 截断样本匹配 NPU AR 文本，这不构成全数据分布的逐 token 等价证明。
- 两轮 DS3 对照数量有限。8 GiB 板进程有不同程度的 VmSwap；表中为整个进程的峰值，不能视为 decode 换页量。固定保留现有 swap 策略。
- 使用现有转接板，配置读取仍偶见错误；链路快照为 Gen3×4，实验未改 PCIe 策略。这与路由/接受率变化共同限制对几个百分点差值的因果解释。

## 复现文件

目录：`llm_generated_docs/evidence/minicpmv47_router_tf32_20261026/`。

- `integration.patch`：仓库相对路径的完整 NPU 原型，含共享常驻权重及 probe；已通过 `git apply --check`，按本轮前 CPU runtime 生成。
- `cpu_tf32_control.patch`：CPU 截断精度控制；与 NPU patch 分别应用。
- `integration_sources_final.tar.gz`、`cpu_baseline_sources.tar.gz`、`cpu_tf32_moe.rs`：源码快照。
- `router_tf32_infer`、`cpu_tf32_infer`、`router_probe_test`：本轮 ARM 二进制。板端分别为 `bench/tf32_infer`、`bench/ctf32_infer`、`bench/router_probe_test`；原 CPU `bench/stage_infer` 保留。
- `results/*.case.json`、`*.clocks.json`、`*.telemetry.json`、`*.timeline.json`：全部原始结果。`router_probe.json` 保存每组 5 次样本；三份 `*_ppl.json` 保存逐目标 NLL。
- `run_router_case.py`、`run_integration.sh`、`run_cpu_tf32.sh`、`prepare_ppl.py`、`build_*.sh`：完整命令与构建；`results/router_ppl_windows.json` 为实际 token 窗口。
- `summary.json`、`probe_summary.json`、`*.analysis.json`、`trace_analysis.json`、`validation.json`：汇总与审计。
- `summarize.py`、`analyze.py`、`audit_traces.py`、`validate.py`、`write_report.py`：复算脚本。
