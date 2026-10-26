# MiniCPM-V 4.7 分页工作区复用与时序分析

## 结果

**prefill 平均减少约 348 ms，decode 未测到提升。** H88K 8 GiB、pp512 / NPU ubatch128 / tg128、DSpark draft3、40 槽、LES 开启，正常分配器 ABBA 每实现两次：

| 指标 | CPU 批处理基线 | 共享分页工作区 |
|---|---:|---:|
| prefill s | 13.1205 | 12.7721 |
| decode tok/s | 7.6925 | 7.6517 |
| 总生成 s，含 prepare、排除模型加载 | 29.8483 | 29.6373 |
| 峰值 RSS GiB，各次峰值的均值 | 6.285 | 6.304 |

prefill 时间变化 -2.66%；decode 各次 tok/s 算术均值变化 -0.53%，小于样本波动。保留改动作为 prefill / 缓冲复用改进，不宣称流式 decode 稳定加速。

## 实现

`src/model/experts/paged.rs` 把贡献数组、收集输入、执行输出、物理 expert ID、单位路由权重、assignment 列表放进所有层共用的一份 `Workspace`，通过 `Paging` 持有。层按顺序执行，一次只借用一个工作区。保留原 hot→cold 顺序、缓存替换、需求/LES 时序和 FP32 按原 token/rank 的 FMA 归约。

每个有效 assignment 在归约前都由 hot 或 cold 执行覆盖；因此贡献区已存在的元素无需逐次清零，Vec 扩容仍做正常初始化。必要的输入展开和输出回写仍保留，复制的有效字节数没有改变。这轮省的是重复分配、初始化及其缺页/释放开销。

工作区保留历史峰值容量；pp512 的贡献数据为 32 MiB，执行输入/输出各最多 1 MiB（ubatch128），另有小型列表。只保留一份，不按 40 层复制。更长逻辑 prompt 会增加这份峰值工作区。没有增加 CLI 参数。

新增 opt-in timeline 标签：prepare、jobs、gather、scatter、execute、cache plan、slot selection、cache touch、reduction、cache snapshot。未改 C++、Rust NPU 算子封装、专家计算、路由精度、模型、缓存槽数、读线程数或 import 线程数。

## 测试条件与原始样本

- H88K 8 GiB，`user@fe80::1828%enp3s0`，现有 GM7 NVMe/转接板；stream / O_DIRECT / 2 个 read worker / 1 个 import worker。
- 相同 noSQ target 与 DSpark；cache 15.625%=每层 40 槽，LES 开启。完整专家始终在盘上。
- CPU4–7 2.4 GHz、NPU mask7 / 1 GHz、DDR2112 MHz，共享 NPU 锁；每例恢复时钟，未改 zram/swappiness。
- 基线是上一轮 CPU router 批处理 `batch_infer`；本轮 `page_opt`。都固定使用 `811b2a3` 加原异步导入补丁的 native 快照：`/home/firefly/.cache/minicpmv47-native-update-20261026/native/rk_npu_matmul`。
- CLI n-predict129 包含 prefill 首 token，计时 decode 128 token。CPU 逻辑 router prefill M512，NPU ubatch128。
- 正常样本为独立进程、空热槽起步，顺序 baseline A→optimized B→optimized C→baseline D；均未设置 trace 或 allocator 环境覆盖。

| 样本 | prefill s | decode tok/s | 总生成 s | 峰值 RSS GiB | VmSwap 峰值 MiB |
|---|---:|---:|---:|---:|---:|
| baseline_ds3_A | 13.150 | 7.6358 | 29.978 | 6.173 | 164.7 |
| optimized_ds3_B | 12.649 | 7.8025 | 29.152 | 6.286 | 182.3 |
| optimized_ds3_C | 12.895 | 7.5009 | 30.123 | 6.322 | 177.4 |
| baseline_ds3_D | 13.092 | 7.7493 | 29.718 | 6.397 | 165.2 |

## 默认分配器下的分页明细

下表来自独立的同口径诊断 trace，除以 128 输出 token；未设置 `MALLOC_MMAP_THRESHOLD_`。trace 和 smaps 采样会增加内存、CPU 和锁开销，绝对数不与正常测速直接混合。

| 分页阶段 ms/output | 基线 | 工作区复用 |
|---|---:|---:|
| paged prepare | 0.458 | 0.301 |
| paged jobs | 0.157 | 0.071 |
| paged gather | 1.739 | 0.665 |
| paged scatter | 0.418 | 0.438 |
| paged reduction | 1.275 | 1.370 |
| paged cache plan | 0.422 | 0.406 |
| paged slot selection | 0.227 | 0.164 |
| paged cache touch | 0.025 | 0.025 |
| paged cache snapshot | 0.067 | 0.076 |
| execute 包装/释放等剩余时间 | 0.359 | 0.403 |
| routed 扣除 wait/import/NPU 后的全部剩余时间 | 6.309 | 5.554 |

输入准备的变化最明显；输出回写和归约基本没有变化。全部分页剩余时间只少约 0.76 ms/output，所以不能从局部复制计时推断大幅 E2E 加速。

### 分配器控制

同一对二进制另采固定 `MALLOC_MMAP_THRESHOLD_=131072` 的 trace。这个条件明显放大了基线的临时数组分配和首次写入成本，不能作为默认运行的提速结论。

| allocator / 实现 | gather ms/output | scatter ms/output | execute 包装/释放 ms/output | 全部分页剩余 ms/output |
|---|---:|---:|---:|---:|
| 固定128KiB / 基线 | 2.836 | 2.330 | 2.264 | 12.194 |
| 固定128KiB / 复用 | 0.649 | 0.706 | 0.394 | 6.749 |
| 默认 / 基线 | 1.739 | 0.418 | 0.359 | 6.309 |
| 默认 / 复用 | 0.665 | 0.438 | 0.403 | 5.554 |

默认分配器 baseline 的 scatter 为 0.42 ms/output，固定阈值为 2.33 ms/output；该差值发生在完全相同的基线二进制。固定阈值 trace 的 decode minor faults 为 495177，而正常 baseline 两次约 1.2–2.1 万。分配器和记录行为显著改变了观察到的开销，必须保留正常运行的端到端对照。

## I/O、缓存和下一步

以下仍使用默认分配器的诊断 pair。

| 项目 | 基线 | 复用 |
|---|---:|---:|
| wait record ms/output | 27.350 | 23.180 |
| import ms/output | 7.065 | 7.172 |
| NPU experts ms/output | 28.577 | 29.384 |
| 必要 cold MB/output | 161.316 | 161.316 |
| 实际读取 MB/output | 199.717 | 201.614 |
| 读窗口平均 GB/s | 1.473 | 1.526 |
| 至少一个 reader 忙时的合计 GB/s | 1.987 | 2.118 |
| LES cold 需求覆盖 % | 65.960 | 65.960 |
| 提前 RAM-ready / cold % | 41.328 | 43.347 |

四份 trace 的 assignment cache hit 均为 65.4496%，unique cache hit 均为 56.6023%；cold 记录数 12926，LES 预测请求 13091、命中冷需求 8526。完整专家 ID 与 LES 序列保持一致，后台完成顺序带来少量额外预取流量变化。

复用不会减少约 161.3 MB/output 的必要冷权重。新实现默认 trace 仍有约 23.2 ms/output record 等待、7.2 ms/output 主线程 import，而临时缓冲优化只有约 0.76 ms/output。优先分析读入/导入关键路径更合适；可做的下一轮调度实验是按 ready 集合分小批执行冷专家，评估额外 submit 能否换来与剩余读取/导入的重叠，收益尚未验证。

## 内存与 swap 的计时边界

以第一个和最后一个生成文本输出为界统计生成期，再单列输出完成后的释放阶段；采样间隔约 100 ms。系统 swap-in 包含全板活动，进程 major fault 也可能包含文件页缺页，两者不等于 DSpark 的计算停顿。

| 正常样本均值 | 基线 | 复用 |
|---|---:|---:|
| 生成期系统 swap-in MiB，近似 | 28.08 | 30.13 |
| 输出完成后系统 swap-in MiB，近似 | 81.34 | 73.39 |
| 进程 VmSwap 峰值 MiB | 164.93 | 179.85 |
| 生成期进程 major faults，近似 | 6303.5 | 7424.5 |

正常样本生成期系统 swap-in 约 20.7–35.5 MiB，输出后另有约 72–83 MiB。整段进程统计会把释放时换入也计算进去；应按阶段判断影响。smaps 中 NPU `/dev/dri/card1` 映射约 4.74 GiB、Swap=0，DSpark checkpoint 文件页约 0.5–0.6 GiB、Swap=0、可回收，swap 主要来自匿名堆/匿名映射。这轮没有证明内存占用降低，也没有证据将速度波动全部归因于 draft 换页。

## 正确性、限制与复现

- ARM release 与 all-target Clippy 通过。
- 同一 20 槽 stream 配置下，128-token prefill 加 8 步 decode 的完整词表 logits 共 {teacher['logit_values']} 个 FP32 值、{teacher['bytes']} bytes，前后逐位一致。该质量测试与 40 槽测速分开。
- 8 个 pp512/tg128 样本（4 正常、4 诊断）全部匹配旧 CPU AR 文本；35 轮、139 verify 行、93/104 接受、9 replay 行，均无 fallback。4 份 trace 的 1440 次实际路由、1365 次 LES 序列全部相同。
- 所有时钟恢复；物理槽位与 staging promotion 审计、缓存重建、I/O bytes 对账通过。自动验证全部通过，逐项结果保存在证据目录的 `validation.json`。
- 未重跑 PPL；使用全 logits 逐位对照和整段路由/文本一致性验证本轮未改数学。只有两次正常样本/实现，decode 差值属于当前波动量级；诊断 trace 自身增加内存和开销。
- 保留共享工作区和可选细分 trace；没有 commit、stash、push。其他 dirty 修改及并行算子任务保持原样。

证据：`llm_generated_docs/evidence/minicpmv47_paging_cpu_20261026/`。

- `integration.patch` / `baseline_paged.rs` / `profile_paged.rs` / `optimized_paged.rs`：本轮窄范围修改和两份诊断实现。
- `frozen_runtime_sources.tar.gz` / `production_sources.tar.gz`：实际 ARM 构建源与相关生产文件；native 沿用上述冻结快照。
- `page_opt` / `page_test` / `paged_infer` / `build_*.sh` / `paged_*build.log`：本轮 ARM 二进制和构建。
- `run_paging_case.py` / `run_teacher.sh` / `run_formal.sh` / `run_default_traces.sh`：所有命令、采样与对照条件。
- `results/`：原始 JSON、stdout/stderr、smaps、telemetry、timeline、时钟恢复和全 logits；`teacher_comparison.json` 保存逐位结果。
- `summary.json` / `*.paging.json` / `*.analysis.json` / `*.memory.json` / `memory_summary.json` / `trace_analysis.json` / `route_equality.json` / `validation.json`：全部汇总、分解与审计。同目录 Python 脚本可复算。
- `prior_run_swap_boundary.json`：上一轮 4 个正常样本按末 token 重新划分的系统 swap-in，约 1.9–24 MiB 在生成期，约 81–91 MiB 在输出之后。
