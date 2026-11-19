# MiniCPM-V 4.7 分支盘点与提交划分

盘点日期：2026-10-26。以下状态来自当前 Git、实际源码差异及各轮实验报告；本次未修改实现、暂存或提交代码，没有重新测速。

## 当前状态

- 工作树：`<HOST_HOME>/rk3588-nn/expr/open_rknpu2_lm_infer-minicpmv47`。
- 分支：`feat/minicpmv47`；HEAD `8bf5568`，与本地跟踪的 `origin/feat/minicpmv47` 相同。
- 相对本地 `origin/main`，本分支独有 3 个提交，main 独有 10 个提交。这里未 fetch，也未执行 merge/rebase。
- 盘点开始时：14 个 Rust 文件和 1 个子模块指针修改，合计 1278 行增加、304 行删除；暂存区为空。另有 16 份未跟踪报告，本盘点文档不计入该数。
- `native/rk_npu_matmul` 位于 `feat/minicpmv47-moe-w4`，HEAD `811b2a3`；本地跟踪的 native `origin/main` 也是此提交。3 个已跟踪文件有改动，另有 `tests/bench_moe_import.cpp` 未跟踪。
- 父仓库记录的 native 是 `0fd946e`；工作树已切到 `811b2a3` 并叠加未提交 ABI 改动。本地父仓库 `origin/main` 已记录 `811b2a3`。

基础 MiniCPM 支持、无 SQ checkpoint 加载、磁盘专家文件、缓存容量参数、原始 LES、DSpark 结构及此前清理，主要已在 `8704842` 提交；`07a2b7c` 合并过当时的 main，`8bf5568` 补了视觉回归测试的配置初始化。不能将旧清理报告列出的全部文件当作当前新增改动。

## 实验归档与取舍

各行数字都是当轮条件下的对照，不能把不同时期、算子版本、量化配置和路由轨迹的结果当成一条严格累加的加速链。

| 实验 | 结果与当前状态 | 提交建议 |
|---|---|---|
| 独立 GEMM 累加、融合 MoE FFN 估算 | 初期为 W4 gate/up、W8 down 的估算；当前实际 routed 专家为全 W4。报告已跟踪 | 历史参照，不能当当前端到端结果；没有对应的新 kernel 改动待提交 |
| 自定义 SmoothQuant、W4 gate/up + W8 down | 已移出默认源码，仍保存在 stash | 不纳入本轮提交，不恢复 stash |
| 基础 noSQ stream、Strata/Omni 调度、20/40/64 槽缓存参数 | 基础实现和缓存/NFS/noSQ 报告已在 `8704842` | 不重复提交；补当前说明即可 |
| eMMC、NFS 千兆/万兆、GM7 链路、并发读取和容量实验 | 形成硬件与内存边界；没有应加入 runtime 的系统配置修改 | 可提交简明报告；模型、裸盘数据、二进制和完整 trace 留本地 |
| 对齐 buffer pool、双 reader、prefill prompt-tail 直接进入 hot 槽 | 当轮选定逐组方案：prefill 吞吐约 +7.4%，AR decode 约 +8.0% | 提交最终实现和回归测试；不保留各消融分支 |
| 实际需求确认后异步导入、共享 staging bank、槽位视图交换 | AR 5.4108 → 5.5442 tok/s；保留共享池及记录长度并发修复 | 提交 C ABI、存储所有权、Rust staging 和生命周期管理 |
| 错误预测也提前导入 NPU | 额外复制代价较高；最终方案只预测读入 RAM | 仅留消融记录，不进入默认实现 |
| DSpark 与 M≤16 预取 | draft3 的调度对照 6.1164 → 8.0838 tok/s；同轮 AR 为 5.3738 | 提交 M≤16 LES、ready-first 消费及专家并集超过槽容量的处理；DSpark 本体已有提交 |
| 更新 native 至 `811b2a3` | 同轮 DS3 8.0838 → 7.8008；文本、接受率和路由也变化 | 保留版本记录；不把差值归为纯计算性能变化。上游改动无需重复提交 |
| NPU TF32 router | DS3 -2.66%，有限样本 PPL +0.796%；原型已撤回 | 只提交结论性报告，不提交原型或 CPU 截断控制 |
| CPU router/LES 一次提交多行 | M4 完整调用约 -24%，M128 约 -33%；E2E +0.49%，未测出稳定整体收益 | 提交简洁实现及逐位等价测试，保持准确的收益表述 |
| 全层共享分页 Workspace | prefill 均值减少约 348 ms；decode 未测出收益 | 提交工作区复用，不宣称 decode 加速 |
| 冷专家 ready 分组 | 小组 7.601 → 7.849；约两组 7.707 → 7.934 tok/s（+2.95%） | 提交约两组策略及非阻塞 ready 领取；小组变体留作对照 |
| 细粒度 target / paging trace、缓存与 I/O 审计 | 用于区分 record 等待、导入和完整算子调用，当前源码保留 | 独立可观测性提交，沿用环境变量开关 |
| SSD 直接读入 NPU buffer | 尚未实现或验证 | 不列为已完成功能；当前省掉的是 staging→hot 的第二次权重复制 |
| Decode 策略离线分析 | 缓存接纳、热专家与冷专家复用等候选分析，没有相应在线实现 | 研究记录，不混入生产功能声明 |

主要报告：

- [流式 I/O](MINICPMV47_STREAM_IO_OPTIMIZATION_20261025.md)
- [异步导入](MINICPMV47_ASYNC_IMPORT_20261025.md)
- [DSpark 与 M16](MINICPMV47_DSPARK_H88K_20261026.md)
- [算子库更新](MINICPMV47_NATIVE_UPDATE_20261026.md)
- [TF32 router](MINICPMV47_ROUTER_TF32_20261026.md)
- [CPU router 批处理](MINICPMV47_CPU_ROUTER_BATCH_20261026.md)
- [分页工作区](MINICPMV47_PAGING_WORKSPACE_20261026.md)
- [冷专家分组](MINICPMV47_READY_GROUPS_20261026.md)
- [后续策略分析](MINICPMV47_DECODE_STRATEGY_20261026.md)

## 建议的提交边界

建议为 1 个 native 提交和 5 个 runtime 提交。下表是逻辑边界，部分变更在同一文件或同一 hunk，不能简单按整文件依次 add。

| 顺序 | 建议主题 | 内容与依赖 |
|---|---|---|
| N1，算子库 | `feat: support staged expert import and slot promotion` | timed import、swap slots、交换后的 backing storage 所有权、不可变 packed-record 长度，以及正式并发/所有权回归测试 |
| R1，父仓库 | `perf: pipeline streamed expert loading for batched verification` | buffer pool、双 reader、有界队列、需求优先/取消/reset、prefill 直接 seed、共享 staging、主线程接管、M≤16 LES 和容量溢出分组；包含 `src/npu.rs`、`src/npu/moe.rs` 的导出/包装以及指向 N1 的子模块更新 |
| R2 | `perf: batch CPU router and LES scheduling` | `src/model/cpu.rs::dense_head_batch`、router/LES 调用，以及 BF16/F16/F32 的逐位等价测试 |
| R3 | `perf: reuse paged expert workspace across layers` | `Paging` 共享贡献、输入、输出、physical ID、routes 和 jobs 缓冲；保留原 token/rank 归约顺序 |
| R4 | `perf: overlap cold expert groups with loading` | `Reader::try_take_any`、约两组冷执行、ready 非阻塞消费及相应测试；依赖 R1/R3 |
| R5 | `feat: trace target verification stages` | `Timings` 的 timeline/context、target 与 paging 事件、reset 后保留 timeline；配套时序分析说明 |

R1 的 NPU 导出涉及 `src/npu.rs` 和 `src/npu/moe.rs`。加载与预取主要位于 `src/model/experts/{io,paged,store}.rs`、`experts.rs`、`qwen35/moe.rs`。R5 还涉及 `qwen35/{decoder,decoder/mixers,load,runtime,timings,transaction}.rs`。文档随对应功能提交，并更新统一的 SUPPORT 说明。

**N1 必须先落地。** `811b2a3` 本身没有新增的 timed-import/swap ABI；当前 Rust 已调用这些符号。只提交父仓库 Rust、或者只记录现有 `811b2a3` 指针，干净检出都不能重建当前实现。父仓库需指向实际包含 N1 的提交；需要对外推送时也应先确保 native 提交可取得。

## 提交前的收尾

1. **整理 native 回归测试。** 当前 `tests/bench_moe_import.cpp` 同时包含计时和重要正确性校验：并发导入/交换、固定 blob 长度、销毁 staging bank 后 hot 存储仍有效。把这些校验迁入现有 `tests/test_moe_quant.cpp` 或同类正式测试。硬编码真实模型记录布局的计时探针留本地；`CMakeLists.txt` 新增 benchmark 目标也应对应排除，不能只排除 cpp 留下失效目标。
2. **更新当前用户文档。** `MINICPMV47_SUPPORT.md` 仍描述单 reader、计算线程负责全部导入、DSpark E2E 尚未完成，已经过时。应写明双 reader/单 importer、M≤16、shared staging、冷专家分组及当前验证范围。
3. **校正文档模板。** `MINICPMV47_PAGING_WORKSPACE_20261026.md` 含未展开的检查数量表达式；发布报告前修正。精简重复报告可以保留最终结果和必要的负结果，不把所有过程记录当作使用文档。
4. **保留关键验证的可复现入口。** M4/M16 超缓存容量及完整词表 logits 的证据当前主要在忽略的 evidence 中。提交小型回归测试/复现说明，外部 checkpoint 仍通过环境变量提供；不提交 checkpoint、构建产物和大 trace。
5. **完成提交组合的正式回归。** 各轮已有 ARM release 构建、Clippy、I/O 测试、原生并发探针和模型对照；最新两组策略有 M1 的2233296个、批量 M4/M16 的1488864个FP32 logits逐位一致，以及8个正式/3个trace样本。但分拆后的提交仍应按仓库规范完成 release fmt/Clippy/测试和相关板端回归，无需重跑所有历史性能消融。
6. **保留数值验证边界。** 更新 native 后的文本和投机轨迹变化已有记录，尚无旧新算子库完整 PPL 对照。当前调度的逐位一致性是相对同一新版算子快照，不能扩大为跨算子版本数值不变。

## 明确保留在提交之外的内容

- 模型、专家文件、板端拷贝、构建目录、可执行文件及大体积原始 trace。
- `llm_generated_docs/evidence/` 中的新实验目录；现有三个已跟踪 Qwen35 MTP JSON 属于历史提交，不在本次清理范围。
- TF32 router、截断控制、推测性 NPU 导入、小组粒度及各 reader 数量的消融实现。
- native 的 `WIP: custom SmoothQuant adapters`、`WIP: unused MoE W4 gate/up with W8 down`，以及父仓库同名 SQ stash。
- 其他工作树的 main、Ling、dual-ubatch 等改动；本次不处理这些工作树和 stash。

本轮只执行了只读盘点、现有验证记录核查和 `git diff --check`，新增此盘点文档。暂存区保持为空。
