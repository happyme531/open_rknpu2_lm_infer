# MiniCPM-V 4.7：异步 NPU 导入实验

实验启动于 2026-10-25，完成于 2026-10-26。沿用上一轮已验证的双路 O_DIRECT、对齐缓冲复用、prefill 分组直接填充 hot bank；本轮只调整专家的流式加载和绑定，不改矩阵计算或量化。

最终选择：LES 提前把专家读到 RAM，实际路由确认后异步导入独立 NPU 暂存槽，完成后交换槽位的存储视图。相同工作负载下 decode 从 **5.4108 → 5.5442 tok/s，约 +2.47%**；与同样分配 96 MiB 暂存权重、关闭异步导入的对照相比约 **+3.41%**。这是小幅收益，未达到之前把导入时间完全隐藏时推算的 6 tok/s 以上。

**测试口径**

- 目标板：`user@fe80::1828%enp3s0`，H88K，约 8 GiB 内存；GM7 NVMe，经现有转接板运行 PCIe Gen3 ×4。未修改 PCIe、内核、挂载、swap 策略。
- CPU4–7 2.4 GHz、NPU mask 7 / 1 GHz、DDR 2112 MHz；每例独占测试锁，结束后恢复时钟策略。
- noSQ MiniCPM-V 4.7 35B-A3B，routed gate/up/down 均 W4A8，shared FFN 与 lm_head W8A8。模型和 `experts.npu` 均在本地 SSD。
- 固定真实 WikiText 的 640 个 teacher-forced token：pp512、tg128，physical ubatch 128，logical prefill 512，context 1024；始终使用 stream，每层 40 个逻辑缓存槽，开启 LES 预取。
- prefill 不算 lm_head；decode 包含 lm_head 和 logits 捕获。排除模型加载、预留、prepare、统计与结果文件写入。每例独立进程、空专家缓存。
- 正式样本使用原分配器环境；trace 单独设置 `MALLOC_MMAP_THRESHOLD_=131072`，不混入正式速度均值。
- 组速度按平均耗时换算，RSS 为组内最大 HWM，所有 GiB/MiB 为二进制单位。每组两例的结果只代表本 fixture，几百分点差异仍需注意运行波动。

**完整消融结果**

| 实现 | 正式样本数 | pp tok/s | tg tok/s | 峰值 RSS GiB | 暂存权重 MiB | 进程最大换出 MiB |
|---|---:|---:|---:|---:|---:|---:|
| 原方案，本轮重测 `baseline` | 2 | 39.246 | 5.4108 | 5.185 | 0 | 0 |
| 同内存同步对照 `control` | 2 | 39.135 | 5.3315 | 5.670 | 480 | 0 |
| 每层 8 暂存槽、读线程内导入 `async` | 2 | 38.647 | 5.3177 | 5.660 | 480 | 103.63 |
| 共享槽池、读线程内导入 `shared` | 1 | 38.605 | 5.4378 | 5.250 | 96 | 0 |
| 同内存同步对照 `fixed_control` | 2 | 39.277 | 5.3616 | 5.259 | 96 | 0 |
| 独立导入线程，推测性导入 `fixed` | 2 | 39.385 | 5.5428 | 5.268 | 96 | 0 |
| **独立导入线程，确认需求后导入 `confirmed`** | **2** | **39.063** | **5.5442** | **5.245** | **96** | **0** |

`shared` 第二例中途返回参数错误，未获得完整测量，不计入均值。原因及修复见下文；该实现只保留为实验记录。初版 `async` 的两例分别为 5.1887、5.4533 tok/s，其中第一例发生换出，不能据此把差异全部归因于导入调度。

最终 `confirmed` 两次 decode 为 **5.4942、5.5952 tok/s**；原方案两次为 **5.3459、5.4773 tok/s**。prefill 没有新的导入流水线，组间小幅差异不足以宣称 prefill 加速。

**decode 明细：每 token**

| 指标 | 原方案 | 推测性导入 | 确认需求后导入 |
|---|---:|---:|---:|
| 端到端 ms | 184.815 | 180.415 | 180.367 |
| 主线程 record 等待 ms | 16.208 | 21.019 | 25.112 |
| 主线程导入/交换 ms | 21.120 | 7.697 | 3.545 |
| 等待＋主线程导入 ms | 37.329 | 28.715 | 28.657 |
| expert complete-call ms | 45.541 | 46.701 | 47.564 |
| 后台导入 service ms | 0 | 26.747 | 23.264 |
| 其中后台 memcpy ms | 0 | 22.367 | 19.983 |
| 其中后台 sync ms | 0 | 3.921 | 2.870 |
| 实际磁盘读取 MB | 189.902 | 189.684 | 189.128 |
| 相对必需 miss 的额外读取 MB | 45.995 | 45.777 | 45.221 |
| 异步导入并被使用的专家数 | 0 | 57.742 | 76.430 |
| 未被使用的异步导入数 | 0 | 23.406 | **0** |

异步模式的 record 等待包含尚未完成的 NPU 导入，不能全部解释为 SSD 等待。后台 service 时间与主线程计算、等待重叠，不能再直接相加到端到端耗时。

最终方案把主线程导入减少约 17.6 ms/token，但等待增加约 8.9 ms，expert 调用增加约 2.0 ms，其余主线程阶段也有变化，最终净省约 **4.45 ms/token**。原生并发探针也观察到了更慢的 memcpy 和 sync；这些数据支持并发资源竞争抵消部分收益，但没有单独测量出 CPU、DDR、driver 各自的占比。

三种实现的逻辑缓存命中率均为 **71.8481%**，冷专家需求约 **90.0859 个/token**，LES 对冷 miss 的预测覆盖率均为 **66.5944%**。最终方案约 76.43 个 miss 使用已导入的暂存槽，其余约 13.66 个同步导入。这里的异步完成数包括路由确认后加载的未预测 miss，不能当作 LES 的预测准确率。

最终方案仍有约 45 MB/token 的额外 SSD 读取。归零的是**错误预测造成的多余 NPU 导入复制**，不是磁盘过读。推测性导入每 token 多复制约 23.4 份专家记录，约 37.4 MB 的记录载荷；确认需求后导入避免了这部分 CPU/内存工作。

`predicted_ready` 在最终方案表示实际路由确定时已经读到 RAM 的预测专家，与原方案口径相同；在推测性导入版本表示整个导入流水线已完成，比较时应使用它单独提供的 `predicted_ram_ready`。初期版本没有独立 RAM-ready 计数，不补猜测值。

**保留的实现**

1. 保留两条 I/O 线程，增加一条独立导入线程；读线程不会被 memcpy 和 cache sync 阻塞。
2. 各 NPU Backend/context 共享一个 32 槽的暂存 bank。本模型使用两个 bank，共 96 MiB；没有为 40 层各分配一套。
3. LES 只提前发起 SSD→RAM 读取。实际 router 确认后，RAM 中已就绪的冷专家可进入导入队列；尚未完成的实际 demand 读取返回后也可进入队列。计算当前层的 hot 专家时可与导入重叠。
4. 主线程需要某记录，而导入尚未开始时，可直接接管 RAM 记录并同步导入；已经开始的导入等待完成。错误预测不进入 NPU 导入队列。
5. 导入完成后，交换 hot/staging 的权重视图、FP32 scales、INT4 correction 和存储所有权；不再复制整份权重。逻辑缓存映射只在完成后发布。
6. 暂存槽以独占 loan 管理，消费、取消、reset 自动归还；reader 先停止并 join 所有线程，再释放暂存 bank/context。未把 `Rc<Backend>` 整体标记为 Send。

沿用原 `prefetch` 开关，不新增用户参数。关闭预取时不分配这些暂存权重；实验用的编译常量已经清理。生产源码保留在 `open_rknpu2_lm_infer-minicpmv47` 工作树，未提交。

**原生导入探针和并发修复**

真实 H=2048、F=512 专家记录，64 轮 import→swap→run，与同步参考输出逐字节相同；销毁 staging bank 后 hot 结果也相同。

| 原生操作 | 独立运行 μs/专家 | 与 hot NPU run 并发 μs/专家 |
|---|---:|---:|
| memcpy | 156.313 | 197.598 |
| scale 校验 | 4.041 | 4.261 |
| gate sync | 22.044 | 39.481 |
| down sync | 10.465 | 14.575 |
| 完整导入 | 192.991 | 256.064 |

槽位交换平均约 **0.154 μs/专家**。这里消除的是暂存槽→hot 槽的第二次复制；SSD 记录→暂存 NPU buffer 的 memcpy 仍然存在。

共享槽池的失败样本暴露了本轮新增并发场景下的缺陷：`rk_npu_moe_quant_expert_blob_bytes` 通过 0 号槽的 vector 元数据计算记录长度；主线程交换 0 号槽时，导入其他槽的线程可能读到中间状态。独立压力测试在不重新读 SSD 的情况下复现 `blob size raced with slot-zero promotion`，因此不把这次错误归因于转接板。

修复为 bank 创建完成后固定记录长度，后续查询只读不可变字段。修复后的探针完成 **2048 次导入与 10,734,665 次并发交换**，导出的记录及计算输出全部一致。失败日志、修复前二进制和修复后日志均保留。

**验证和证据**

- 最终算法及各成功对照的全部 128-token FP32 logits 与原 resident 参考逐字节一致。
- 5 份 trace 共审计 **1,024,000 次物理槽位绑定**、**25,532 次暂存导入完成后交换**，均 0 违规；共享暂存池的 trace 带全局 bank ID。
- 专家队列/存储/窗口的 11 项单元测试通过；ARM 板上 release 构建、`cargo fmt --check`、`cargo clippy --all-targets -- -D warnings`、C ABI header 编译通过。
- 选中的两例正式测量及 trace 均未发生进程换出；没有尝试让完整权重驻留在 8 GiB 板上。
- 移除实验编译开关、清理内部字段名后，再次独立编译并跑完整 pp512/tg128：**pp 39.0659、tg 5.5327 tok/s、峰值 RSS 5.2544 GiB、进程换出 0、全部 logits 精确一致**。结果保存在 `slots40_on_production_verification.*`，与上述两例正式均值分开。

原始证据在 [evidence/minicpmv47_async_import_20261025](evidence/minicpmv47_async_import_20261025/)：`summary.json`、`trace_audit.json`、`results/*.json`、`*.log`、构建脚本、各实现的源码归档与板端二进制。

可重新汇总已有数据：

```bash
python3 llm_generated_docs/evidence/minicpmv47_async_import_20261025/summarize.py
python3 llm_generated_docs/evidence/minicpmv47_async_import_20261025/audit_traces.py
```

本次证明了异步导入、槽位交换的时序可以正确工作，也显示只把 memcpy 移到后台的收益有限。更大的潜在收益需要继续验证能否直接读入 NPU 可用的缓冲，减少 memcpy 本身；该路径本轮未实现、未测量。
