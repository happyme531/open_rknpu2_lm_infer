# MiniCPM-V 4.7：record 交接与目标验证排查

## 结论

约 55 ms/output 的剩余验证时间分散在多个阶段：投影 14.07 ms、router/LES/请求管理 11.36 ms、分页编排 8.28 ms、CPU GDN 6.94 ms、共享专家 5.89 ms、lm_head 5.15 ms。优先做 CPU router/LES 行批处理的配对 probe，以及分页缓冲复用；11.36 ms 不能全部视为线程池开销。

record 等待主要落在读取服务和已开始的导入服务。两份 trace 的 pread 重叠为 13.56–15.49 ms/output、后台导入重叠为 5.10–5.40 ms；导入排队仅 0.38–0.41 ms，就绪后交接约 0.51 ms。现有 ready-first、确认需求后导入、主线程接管未开始导入的策略可以保留。

后台导入自身约 27 ms/output，其中约 83% 是 memcpy、约 15% 是 sync，大部分已被重叠隐藏。继续增加 import worker 的优先级较低。直接读入 NPU 缓冲仍需验证 O_DIRECT 与该映射兼容，不能仅靠获得 vaddr 就认定零拷贝可用。

本轮只增加 runtime/I/O 时间线计时，没有修改专家 complete-call 内部、算子库、缓存策略、投机算法或权重。

## 条件与计时口径

- H88K 8 GiB，`user@fe80::1828%enp3s0`，GM7 NVMe；保持既有 Gen3×4、内核、挂载和 zram。
- CPU4–7 2.4 GHz、NPU mask7 / 1 GHz、DDR2112 MHz，每例通过原 harness 恢复。
- noSQ 模型，stream、每层 40 槽、LES、O_DIRECT、2 个 read worker、1 个 import worker；没有全量驻留专家。
- pp512、ubatch128、tg128；CLI n-predict129 包含 prefill 的首 token。DS3 共 35 轮、139 个逻辑验证输入、128 个计时输出。
- native 固定为上一轮已验证的 `811b2a3` 加原异步导入补丁快照，位于 `/home/firefly/.cache/minicpmv47-native-update-20261026/native/rk_npu_matmul`。
- 正常控制使用默认 allocator；trace 使用原 `RKNPU_EXPERT_TIMELINE` 和 `MALLOC_MMAP_THRESHOLD_=131072`。
- 下表只含 decode 验证，不混入 prefill 或 draft head。所有 ms/output 除以 128；不能解释成单行算子延迟。

## 端到端与 trace 扰动

| 样本 | decode tok/s | ms/output | prefill s | 峰值 RSS GiB | 最大 VmSwap MiB |
|---|---:|---:|---:|---:|---:|
| ds3_control | 7.6801 | 130.226 | 13.551 | 6.204 | 184.3 |
| ds3_detail_trace | 7.5072 | 133.266 | 14.727 | 6.269 | 86.4 |
| ds3_split_trace | 7.4739 | 133.816 | 14.793 | 6.258 | 93.9 |

三例文本均匹配上一轮新 native 的 AR 参考；两份 trace 的全部 actual routes、cold 列表、LES 预测数组一致。trace 比本轮控制慢约 2–3%，且影响 read/import 线程竞争和 takeover 数量。因此用两份 trace 判断稳定瓶颈，不把诊断速度当作优化收益。

## 第 3 项：剩余验证的互斥分解

第二份 trace 完整验证 109.914 ms/output；扣除专家 complete-call 27.669、record 等待 22.602、主线程 import/swap 4.455，剩余 **55.189 ms/output**。

| 部分（互斥） | ms/output | 占剩余验证 |
|---|---:|---:|
| GDN/Attention 输入及输出投影 | 14.070 | 25.5% |
| 实际 router、LES、请求管理 | 11.358 | 20.6% |
| 分页专家编排与数据整理 | 8.281 | 15.0% |
| GDN 状态计算 | 6.944 | 12.6% |
| 共享专家完整路径 | 5.890 | 10.7% |
| 目标 lm_head | 5.150 | 9.3% |
| Attention 核心、QKV、gate | 1.822 | 3.3% |
| 其他 CPU 与调用开销 | 1.674 | 3.0% |

分页编排包含物理调用外的规划、临时贡献数组、输入展开、输出收集、路由归并、记录生命周期与入口/退出；8.28 ms 还不能全部归因于 allocator。

| 具体阶段 | 调用数 | 累计 ms | ms/output | 平均 ms/call |
|---|---:|---:|---:|---:|
| embedding | 35 | 8.420 | 0.066 | 0.2406 |
| input norm | 1400 | 16.374 | 0.128 | 0.0117 |
| GDN input projection | 1050 | 1047.190 | 8.181 | 0.9973 |
| GDN state | 1050 | 888.818 | 6.944 | 0.8465 |
| output projection | 1400 | 555.961 | 4.343 | 0.3971 |
| residual norm | 1400 | 22.439 | 0.175 | 0.0160 |
| router | 1400 | 493.279 | 3.854 | 0.3523 |
| demand bookkeeping | 1400 | 95.680 | 0.747 | 0.0683 |
| LES router | 1365 | 777.510 | 6.074 | 0.5696 |
| LES enqueue | 1365 | 87.403 | 0.683 | 0.0640 |
| shared gateup | 1400 | 344.040 | 2.688 | 0.2457 |
| shared activation | 1400 | 78.017 | 0.610 | 0.0557 |
| shared down | 1400 | 208.792 | 1.631 | 0.1491 |
| shared gate merge | 1400 | 123.046 | 0.961 | 0.0879 |
| FFN residual | 1400 | 6.810 | 0.053 | 0.0049 |
| feature record | 1400 | 2.586 | 0.020 | 0.0018 |
| attention input projection | 350 | 197.810 | 1.545 | 0.5652 |
| QKV norm rope | 350 | 21.173 | 0.165 | 0.0605 |
| attention | 350 | 161.836 | 1.264 | 0.4624 |
| attention gate | 350 | 50.144 | 0.392 | 0.1433 |
| head preparation | 35 | 5.919 | 0.046 | 0.1691 |
| head | 35 | 659.187 | 5.150 | 18.8339 |
| logits finite check | 35 | 30.049 | 0.235 | 0.8585 |

源码对应与含义：

- `qwen35/moe.rs`：实际 router 逐行 `dense_head`；LES 再对下一层 router 逐行计算。实际 router 3.854 ms/output、LES router 6.074；demand 记账 0.747、预测入队 0.683。主要时间在计算，不在入队。
- `model/cpu.rs`：每次 `dense_head` 都调用 `pool.install`。真实 router、shared gate、GDN a/b 已核实为 BF16；router 256×2048，已有 BF16 NEON 路径，不是 F16 scalar fallback。线程池、缓存及并发带宽各占多少，仍需配对 probe。
- `model/projection.rs` / `qwen35/load.rs`：GDN qkv/z 是 W4，a/b 是受保护 BF16；存在受保护模块时分部执行。GDN 输入投影的 8.18 ms 包含这些调用和布局整理，不能全算成单次 W4 GEMM。
- `layers/gated_delta/state.rs`：验证执行 CPU FP32 `compute`；endpoint 通过 swap 保存，没有第二遍完整 endpoint memcpy。每层状态约 2 MiB，共 30 个 GDN 层；6.94 ms 包括卷积、递推、norm、紧凑日志等。
- 共享专家每 token 一次，gate/up、CPU SwiGLU、down、sigmoid gate 合并共 5.89 ms；shared gate 合并约 0.96 ms。
- `speculative/driver.rs` 的 verify 计时包含整块 logits `to_vec`；与模型内计时及时间线收尾的差额约 0.52 ms/output，优先级低。

## 第 2 项：record 等待与导入

| 等待区间 | 第一份 trace ms/output | 第二份 trace ms/output |
|---|---:|---:|
| 选中记录读取开始前 | 0.050 | 0.086 |
| 与选中记录 pread 重叠 | 13.557 | 15.490 |
| pread 完成到 RAM ready | 0.227 | 0.228 |
| RAM ready 到后台导入开始 | 0.382 | 0.409 |
| 与后台导入服务重叠 | 5.103 | 5.396 |
| 导入完成/旁路接管到发布 ready | 0.490 | 0.483 |
| ready 发布到主线程消费 | 0.506 | 0.509 |
| **总计** | **20.314** | **22.602** |

对每次 `take_any` 消费记录的 pread、RAM ready、stage import、read ready 时间戳与 wait 区间求交，完整覆盖 wait，不叠加线程时间。两份 trace 中所消费记录的 ready 都等于 wanted 集合内最早的 ready 发布时刻，没有把其他 ready 记录的机会算成读取等待。区间归因仍不是独立的 SSD/CPU/DDR 因果消融；pread complete-call 也包含缓冲获取和系统调用服务。

| 导入自身时间（不直接相加到 E2E） | 第一份 ms/output | 第二份 ms/output |
|---|---:|---:|
| 后台 copy_ms | 22.664 | 22.465 |
| 后台 validate_ms | 0.439 | 0.445 |
| 后台 sync_ms | 3.859 | 3.995 |
| 后台 total_ms | 26.974 | 26.917 |
| 主线程 main copy + sync | 4.483 | 4.323 |
| 主线程 slot swap | 0.143 | 0.132 |

第二份 trace 中 10992 个 cold 记录通过后台导入再换槽，1934 个由主线程接管复制，合计 12926。slot swap 仅 0.132 ms/output。后台导入区间共 27.278 ms/output，只有 5.396 ms 与 record 等待重合，约 80% 已在该等待之外；不能把 27 ms 全加到 E2E 上。

理想地完全消除等待内的后台导入和主线程复制/同步，暴露预算约 9.72 ms/output，对应这份诊断 trace 上 7.47→8.06 tok/s 的理论上限。实际减少 memcpy 仍保留部分 sync/元数据处理，CPU/DDR 竞争还会联动，这不是实测收益。

### 带宽与流量

| 指标 | 第一份 | 第二份 |
|---|---:|---:|
| 实际读取 MB/output | 202.014 | 201.028 |
| 必要 cold 记录 MB/output | 161.316 | 161.316 |
| 验证窗口平均 GB/s | 1.533 | 1.525 |
| 至少一条 pread 活跃时平均 GB/s | 2.164 | 2.119 |
| 两条读线程均活跃的窗口比例 % | 64.077 | 65.488 |
| 无 pread 活跃的窗口比例 % | 29.152 | 28.048 |

窗口为第一轮目标验证入口到最后一轮验证结束，包含轮间 draft/commit。同步 pread 区间按两线程并集/交集计；批量 preadv 多记录共享区间只计一次线程忙时，字节逐记录相加。约 28–29% 时间无 pread，不能称整段 decode 持续跑满 SSD。存在读取时约 2.12–2.16 GB/s，高于全窗口约 1.53 GB/s；CPU/NPU/导入同时竞争内存，不能用裸读峰值相减得到可免费追回的时间。

两份 trace 的 cache assignment hit 65.450%，unique-demand cache hit 56.602%，冷需求预测覆盖 65.960%。cold 提前 RAM-ready 42.890–45.296%。实际读约 201–202 MB/output，必要 cold 161.32 MB/output，多读约 40 MB/output。错预测/取消造成的多余磁盘读取仍在；确认需求后导入避免把这些错误预测全部复制进 NPU。

## 优化顺序

1. **CPU router/LES 行批处理 probe。** 保持现有 BF16 dot 的累加顺序，比较逐行 `pool.install` 与一次提交多行的 complete-call，覆盖 M=1/4/8/16；单输出 shared gate 比较直接 dot 与线程池。通过输出/路由一致性后再跑默认 allocator 的端到端配对实验。预算是约 10 ms 的实际/预测 router 计算及约 1 ms 的 shared gate，不能预先保证全省下来。
2. **分页层复用贡献、展开输入和输出缓冲。** 分页编排约 8.28 ms/output，先拆归并/布局/分配；保留 hot/cold 分组与读取重叠。已有 read buffer pool 生效：两份 trace 都只分配 536 次、复用超过 2 万次，不是每条记录都重新分配 1.6 MB。
3. **直接读 NPU 映射的最小兼容性 probe。** 现状为对齐 RAM record → memcpy 到 gate/down NPU 存储 → sync。gate/down/元数据分散存放，需要验证 O_DIRECT 地址/长度对齐，以及该映射能否供直接 I/O pin；通过后再扩展借槽 API 与所有权。暴露导入预算约 9.7 ms，不能按后台总计 27 ms 承诺收益。
4. **后续算子方向是投影和 GDN。** 非路由投影约 14.07、CPU GDN 6.94、head 5.15 ms/output，全 Attention 核心只有约 1.26。专家 complete-call 内部留给另一项任务。

导入排队约 0.4 ms/output；扩 import worker 或把错预测提前复制进 NPU，不是这份数据支持的优先方案。此前推测性导入的额外复制代价见 `MINICPMV47_ASYNC_IMPORT_20261025.md`。

## 验证与文件

ARM release 编译及 lib Clippy 均通过。三次 pp512/tg128 正常退出、文本匹配 AR、无 fallback、时钟恢复；两份 trace 共 416640 次物理槽位检查、21918 次 promotion 检查及 cache 重建均无违规。没有新 PPL 测量。VmSwap 为整个运行期峰值，不能据此推断 decode 正在换页。

证据：`llm_generated_docs/evidence/minicpmv47_stage_profile_20261026/`。

- `results/ds3_control.*`、`ds3_detail_trace.*`、`ds3_split_trace.*`：控制、两份完整 timeline 与系统 telemetry。
- `ds3_*.analysis.json`、`trace_analysis.json`、`validation.json`：阶段明细与审计。
- `analyze.py`、`audit_traces.py`、`validate.py`：复算脚本。
- `runtime_profile_sources.tar.gz`、`runtime_profile_split_sources.tar.gz`、`runtime_profile.patch`：两版计时源码及相对冻结 runtime 的本轮 diff。
- `stage_profile.log`、`stage_profile_split.log`、`build.sh`：ARM 编译与 Clippy 日志。
- `floating_metadata.json`：真实 safetensors dtype/shape。

源码仍为未提交改动。沿用 `RKNPU_EXPERT_TIMELINE` 输出事件；常规 stats 累计口径保持不变，没有增加 CLI 参数。
