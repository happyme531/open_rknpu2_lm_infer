# MiniCPM-V 4.7 专家缓存与预取时序

2026-10-21，RK3588，分支 `feat/minicpmv47`。

本报告使用此前带 SQ 的 checkpoint 与 native v1 专家缓存。自定义 SQ 适配现已单独 stash，当前工作树采用不含 input scale 的 v2 记录；下述数据保留为旧权重的时序验证结果。

旧板端权重与 v1 缓存已清理。当前无 SQ 标准权重、移植后的调度和完整 pp512/tg128 结果见 [新报告](MINICPMV47_NOSQ_BENCHMARK.md)。

## 结果

eMMC 上的专家流式加载与 LES 预取通过时序检查。每个选中专家在计算前已导入正确的物理槽位，读取完成后才交接和导入，已有需求队列优先于预测队列。开启/关闭预取、调整调度顺序前后，9 组完整 FP32 logits 的文件逐字节一致。

采用真实 WikiText token 窗口：先 prefill 128 token 建立缓存，再执行固定的 8 个后续 token。每层 20 个热槽位、共享 16 个冷槽位，物理 ubatch 128。GDN 使用 CPU FP32。现有 C4 checkpoint 的 routed gate/up 和 down 均为 W4，共享专家与 LM head 为 W8。

CPU 固定 2.4 GHz，NPU 1 GHz，DDR 2112 MHz；使用共享 NPU 锁，所有本轮时序运行均正常恢复频率。未启用 autotune。数值检查与时序分析针对这组输入，不作为长文本生成速度的稳定基准。

| 8 步 decode，最终调度 | 关闭预取 | 开启预取 |
|---|---:|---:|
| 完整前向耗时，含 LM head | 6.324 s | 5.770 s |
| 等待专家记录 | 4.970 s | 4.389 s |
| I/O 与主线程非等待阶段重叠 | 0.235 s | 1.181 s |
| 专家完整调用时间 | 0.325 s | 0.320 s |
| 热缓存命中 | 1727 / 2560，67.46% | 同左 |
| 下一层 top-8 预测命中 | — | 2093 / 2496，83.85% |

开启预取后这组 decode 用时减少 8.8%。两次单独前向测量都包含相同的目标 token，未使用 DSpark。

## 时序

当前执行顺序：

1. 当前层实际 router 产生 top-k，当前层冷专家需求先入队。
2. 用当前层 FFN 输入通过下一层 router 生成 LES 预测，只为尚未驻留的专家排预取。
3. 一个后台线程优先读取需求队列，其后才读取预测队列。
4. 主线程先计算热专家；读取线程同时准备冷专家记录。
5. 主线程等待记录完成，导入冷 bank 并同步，再提交专家计算。
6. 每个贡献保留到原始 top-k 顺序进行 FP32 FMA 归约。

时间线暴露的调度问题已修复：当前层的已知冷需求应先于下一层预测入队。主线程等待期间与其他层预测读重叠的时间由 81.85 ms 降至 0.64 ms。剩余重叠位于当前读取已完成后的唤醒与下一次读取启动交界；队列优先级审计没有发现预测任务越过排队中的需求任务。

最终两组时序分别检查了 43,520 次专家 assignment 的物理槽位归属、6,404 次导入，以及读取交接顺序，均无违规。旧版与新版、预取关闭与开启的四份 logits 全部逐字节相同。

LES 命中不等于记录在需求发生前完整就绪。本次 eMMC 运行中，很多记录只是提前开始读取，需求发生后仍需等其剩余部分。即使如此，I/O 与主线程计算/导入等非等待阶段重叠约 1.18 s，减少了暴露的等待。

## 存储

板端本地路径：

- 原始模型：`<BOARD_HOME>/.cache/minicpmv47-model/MiniCPM-V-4.7-35B-A3B-C4-W4A8-Mixed`，20,193,257,915 bytes，110 个文件。
- NPU 专家缓存：`<BOARD_HOME>/.cache/minicpmv47-experts-emmc/experts.npu`，16,441,675,776 bytes。

复制保留原文件修改时间，原生缓存可直接使用。测试中权重和专家记录均从本地 eMMC 加载。1 GiB、8 MiB 块、`O_DIRECT` 顺序读取实测约 293–295 MB/s。

开启预取的最终 decode 实际读取 1,403,322,368 bytes，约 175.4 MB/token。其中实际完成但未使用的预测读为 65,830,912 bytes；263 个未命中预测大多在发出读取前已取消。实际 I/O 请求共 874 条，冷需求 833 条。

若以 2 GB/s 仅计算这条 trace 的实际传输量，下界约 88 ms/token；这不是 SSD 实测，也不是完整 decode 速度。更快的存储可能让更多错误预测在取消前完成，不能直接按带宽比例缩放本次整体耗时。

## 复现与证据

源码：`src/model/experts/{store,io,paged,timeline}.rs` 与 `src/model/qwen35/moe.rs`。时间线通过 `RKNPU_EXPERT_TIMELINE` 启用；默认关闭。测试入口为 `minicpm_prefetch_timeline`，独立 I/O 队列测试覆盖预测提升为需求、未来层请求保留、重复需求提交和读取错误。

板端构建与测试目录为 `<BOARD_HOME>/.cache/minicpmv47-build`。原始脚本和报告位于 `llm_generated_docs/evidence/minicpmv47_support_20261020/`：

- `paging_timeline_fixed.log` / `.exit`：Clippy、76 项单元测试、两次完整时序测试及 logits 文件比较，退出码 0。
- `paging_emmc_fixed_{0,1}.json`：前向耗时、缓存与读取计数、进程内存快照。
- `paging_emmc_fixed_{0,1}.clocks.json`：固定频率、退出码与恢复记录。
- `paging_emmc_fixed_{0,1}.timeline.json`：可用 Perfetto 或 Chrome tracing 打开的时间线。
- `paging_emmc_fixed_{0,1}.audit.json`：时序与槽位归属审计。
- `paging_emmc_fixed_{0,1}.png`：首个 decode 的前 180 ms。

分析命令：

```bash
python3 tests/tools/analyze_expert_timeline.py \
  llm_generated_docs/evidence/minicpmv47_support_20261020/paging_emmc_fixed_1.timeline.json \
  --report llm_generated_docs/evidence/minicpmv47_support_20261020/paging_emmc_fixed_1.audit.json \
  --plot llm_generated_docs/evidence/minicpmv47_support_20261020/paging_emmc_fixed_1.png
```

## 限制

- prefill 按实际路由去重读取，本轮没有为 prefill 启用 LES；长 prompt 的不同 ubatch 会重新读取部分冷权重。
- 单个已提交的存储读取无法由队列优先级抢占；错误预测会在路由确认后取消尚未提交的请求，已完成或正在进行的读取单独计数。
- 进程内存快照受加载时 CPU 堆保留影响，本轮最终 VmHWM 约 6.2–6.6 GiB。NPU 权重 payload 约 2.54 GB 加冷 bank 25 MB，不代表进程总内存；内存稳定性需要单独分析。
