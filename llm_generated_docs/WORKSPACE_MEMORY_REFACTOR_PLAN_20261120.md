# RK3588 推理临时缓冲重构计划

## 目标和基线

首版优化单模型实例连续处理文本请求时的临时内存管理。使用审查 worktree 的 `5930ead` 和 native `ec363d9` 建立对照。主验收对象为 Qwen3-4B AR、Qwen3.5-4B MTP1 单与双 ubatch。初始矩阵覆盖直接模型接口和 API 使用的 prepared prefill 路径。连续 batching 在资源所有权和预算接口稳定后接入。

| 基线观测 | 设计目标 |
|---|---|
| Qwen3.5 双 ubatch 在第二条同样的 512-token 请求里创建 874 个 GEM，prepare 墙钟 831 ms，prefill/首 token 阶段 GEM 释放累计 370 ms | 常用形状在预算内跨请求复用，消除相同大形状的反复创建和销毁 |
| Qwen3-4B 连续请求也重建 541 个 GEM，稳态 AR 则基本复用已有缓冲 | 优先处理请求边界，保住稳态 decode 行为 |
| Linear 的 FP32 padding 和 native A/partial 随形状分别持有 | 按同时活跃的缓冲需求规划容量，减少常驻保留的内存代价 |
| Qwen3.5 MTP1 后续 14 个输出申请约 62.8 MiB Rust 堆，MTP domain 有 96 次创建和 96 次销毁 | 单独消除小 M 的 workspace 换形和词表级数据复制，分别测量收益 |
| 默认 prepared 512-token 请求保存两份前缀快照共 100.5 MiB，KV reset 清理整块容量 | 用独立的快照字节预算和 KV 清理策略处理跨请求状态 |

[完整审计](TEMP_BUFFER_MEMORY_AUDIT_20261120.md) · 板端阶段数据（本地归档 `evidence/temp_memory_audit_20261120/summary.json`）

## 内存所有权

| 对象 | 所有者 | 复用和生命周期 |
|---|---|---|
| 权重及策略 | 模型实例和所属 native backend | 权重按 domain 放置，策略按既有形状和调优配置查询，主副 lane 共享权重 |
| Linear 的 FP32 padding | 每个 backend/lane 一组 input/output | 同一 lane 的 Linear 依次使用。两个缓冲各按实际尾块的最大需求扩容 |
| Native GEMM 数据 scratch | 每个 IOMMU domain、每条执行 lane | 输入、部分和及关联 scratch 根据活跃区间决定可复用的范围。提交等待、同步和结果消费完成后可覆盖 |
| 形状专属 plan 和任务描述符 | 各 domain/lane 的形状缓存 | 保留策略、tiling、命令和任务对象。一个 plan 只绑定当前有效的数据缓冲。任务缓冲沿用驱动要求的独立基对象 |
| GDN、GQA/MLA 等专用 scratch | 对应算子和 lane | 先沿用已有算子接口，纳入统一预算和计数。GDN 满块与实测常用尾块按预算持有 |
| KV、GDN/KDA canonical state、事务状态和前缀快照 | 模型的序列状态及前缀缓存 | 这些对象需要活过单次算子执行。有单独的容量与恢复语义。连续 batching 接入时由请求槽位或缓存持有 |

同一块共享数据缓冲的容量取各可复用形状所需的**最大同时占用**，包含对齐、padding 和归约 guard。W4 流水在设备读当前 slot 时还可能填充下一 slot。计算峰值时要包括两个活跃 slot。融合 FFN 的 gate/up → down 跨调用 scale、middle 输出需要保持有效，直到 down 用完。两条并行 ubatch lane 的容量按总和计入同一个 domain。 [W4 plan](../native/rk_npu_matmul/src/rk_npu_i4_plan.cpp#L125) · [FFN scale 生命周期](../native/rk_npu_matmul/src/rk_npu_w4a8.cpp#L354) · [GEM 子视图](../native/rk_npu_matmul/include/rk_npu_common.h#L164)

## 阶段 0 固定对照与内存口径

- 基于已有 阶段探针（本地归档 `evidence/temp_memory_audit_20261120/probe.rs`） 增加来源标记，把 Linear、GDN、Attention、draft 各自的 GEM owner 数、容量和创建/销毁时间按 domain 记录。计数跟随 owning GEM，多个子视图只计一次。区分当前和峰值。IOVA 对齐、碎片及内核页开销分别记录其可观测范围。
- 每次跑一组关闭探针的端到端对照，用于比较用户可见时延。阶段记录含 reset、reserve、prepare、prefill/首 token、首个 decode 间隔、后续 inter-token latency 和请求总耗时。RSS/PSS、Rust 申请字节、native GEM 容量和同步累计时间各自报告。
- 固定模型、调优缓存、输入、编译器、绑核和 NPU/DDR 时钟。同板交替运行基线和改动版，输出每次样本与离散程度。性能用真实请求时序，正确性另用固定 token 对照。调优或策略回退变化随样本报告。

## 阶段 1 Rust staging 共享与常用 workspace 保留

**1a 把尾块 staging 提到 backend/lane。** 从 `Work` 移出 Rust FP32 input/output，放入 `BackendInner` 的共享 scratch。通过现有 backend mutex 保证独占使用。M 恰好相等继续直通 caller 缓冲。尾块按当前 M、K、输出宽度取有效区间，清零需要参与计算的 padding 行，按逻辑行数回拷。检查普通 W4/W8，以及 fused SwiGLU 的输出宽度与 stride。resize 按实用容量增长，减少在 prefill 大满块为所有形状申请 staging。 [Linear 执行与 workspace](../src/npu.rs#L1035) · [padding 路径](../src/npu.rs#L1169)

**1b 在资源预算内保留热形状。** 将 `Workspaces::enter_decode` 的阶段切换与待释放对象的选择分开。首个 head 完成后，已结束执行的 prefill 形状进入可复用集合。优先保留当前配置的常用 M128 Linear/GDN，以及 decode/verify 实际使用的小 M。偶用尾块在资源申请确实需要空间时先驱逐。首轮策略沿用已有行数规划，避免把改变算子策略和内存生命周期放在同一次对照里。 [阶段切换](../src/model/workspace.rs#L27) · [Qwen3.5 workspace 路径](../src/model/qwen35/runtime.rs#L63) · [执行 lane](../src/npu/lane.rs)

**按 domain 做容量准入。** 在原有权重、KV、workspace 与运行时必须项之上，使用各 domain 的预算估算常驻可用空间，包括两个 lane 和已知额外 native 分配。保留地址对齐和碎片的余量，实际 GEM 分配结果作为最终判据。申请前回收该 domain 内已完成且未被活跃 plan 引用的非必要对象，预算不足时可以采用现行的分阶段保留方式。不同 domain 的空闲容量分别核算。分配出错时释放本次成功创建且未发布的对象，保留已提交 KV 与事务语义。提交失败沿用对应算子的 poisoned/recovery 处理。

**验收。** 同进程重复 pp512→tg128 和 `512→129→512`，RSS 与分 domain 占用在预热后进入稳定范围。预算允许且已预热的常用 Linear/GDN 形状，在直接 pp512 的下一请求中创建和销毁目标均为 0。单独核查 prepared 的尾块、draft 与 Attention 的事件。默认路径首 token 前的 prepare 与回收合计下降，同时验证相同配置的 decode 和峰值占用。覆盖保留充足、预算紧张及中途分配失败三种路径。

## 阶段 2 Native W4A8 跨形状共享数据缓冲

1. **先做少量形状的原型。** 从现有 W4 `I4Plan` 开始，把输入、部分和的容量查询与 GEM owner 分开。保持原来的物理布局及 tiling，使用 `rk_npu_mem_view` 绑定按 domain/lane 创建的共享 allocation。为流水 slot 分配互不重叠的范围。若归约模式要保留多波 partial/guard，按该策略的真实活跃范围计算。形状专属 regcmd/task 沿用原实现。
2. **绑定稳定基址。** 优先在 prepare 知道形状集合时一次规划。扩容时等待该 lane 的相关提交完成，使引用旧基址的 plan 重新绑定，再释放旧 GEM owner。预算估算包含扩容期间旧、新 owner 同时存活的峰值。创建或绑定失败时，原有可用 plan 可以继续按原方式使用。native 内部独占由 lane/执行生命周期保证，Rust 锁和 ABI 不允许同一 scratch 并发重入。
3. **保留算子策略并扩大覆盖。** 首测 M1/M128、至少两组 K/N 交替、串行和双 lane。随后覆盖 split-K、NPU reduce 的保留缓冲以及 fused SwiGLU/FFN 的交接场景。W4 chain 或其他计划需要独立核对其活跃区间，再纳入共享。首版已有 ABI 调用者可继续使用独占 workspace，新路径用已有形状策略运行。W8/FP16 后续依据对象与字节的实测贡献接入。
4. **最后做相应的控制面复用。** 如果数据 scratch 共享后仍有持续的 GEM 对象或 worker 创建成本，再单独测 regcmd 分配和提交线程复用。各自有独立计数和性能证据。

**验收。** 用 native 定长输入逐位对比当前策略与共享 scratch 输出。覆盖重复切换、扩容重绑、两 lane 分别提交、异常后恢复及 IOMMU domain 隔离。报告 data GEM owner 数、每 lane 数据容量、整个 domain 峰值、第一条和连续请求时延。目标是同 lane 的数据容量按最大活跃需求增长。形状专属控制对象的峰值单列报告。无形状首次切换时的性能差异应通过独立测量解释，再变化默认路径。 [I4 plan](../native/rk_npu_matmul/src/rk_npu_i4_plan.cpp#L125) · [W4 workspace](../native/rk_npu_matmul/src/rk_npu_w4a8.cpp#L26)

## 阶段 3 MTP 热路径和状态容量

这阶段的三项可以分别提交和归因。

| 工作 | 做法 | 验收 |
|---|---|---|
| MTP 浮点 workspace | 先用计数确认小 M 的切换来源。为常用 M1/M2 保留有界 workspace，prefill 的大 M 单独按预算保留。后续实测决定 DFlash/DSpark 是否共享此策略 | 相同上下文与 MTP1 验证行数下，小 M 预热后 GEM create/destroy 接近 0。对照 IOVA 与 steady decode 时延 |
| sampler 与验证 logits | 把 proposal/target 的临时 history 和候选分配提到可复用 scratch，保留 RNG、惩罚与采样顺序。其后改 verify 输出的所有权，先复用输出缓冲，再确认何时能只保留已消费位置所需行 | greedy 输出一致，固定 seed 随机输出及接受/回放轨迹一致，覆盖 EOS、stop、取消、提前终止。报告每 round Rust 申请量与吞吐 |
| 前缀快照与 KV reset | 快照增加字节预算，检验命中率、重算成本与最坏占用。KV reset 单独实现按有效范围清理的原型，保留 LUT V/分母、padding、prefix sum/absmax、tail epoch 和 poisoned 恢复 | 测 512/4K/支持的长上下文。fresh、reset 后及 truncate 后对照完整 logits，覆盖窗口边界、相邻缓存重用和异常恢复。快照驱逐后仍能正确重算 |

[浮点工作区](../src/npu/floating.rs#L18) · [投机 driver](../src/speculative/driver.rs#L60) · [前缀快照](../src/model/qwen35/cache.rs) · [KV 重置与截断](../native/rk_npu_matmul/src/rk_npu_attention_f16_cache.cpp)

## 板端验收矩阵与接入顺序

| 场景 | 验收内容 |
|---|---|
| 单次冷请求、重复 pp512→tg128、短长交替 | 第一条与后续 TTFT、prepare、首个 decode 间隔、总耗时。确认指向常驻、驱逐或换形的事件计数 |
| Qwen3-4B AR、Qwen3.5-4B AR/MTP1，单/双 ubatch | 固定 token 逐位 logits 对照。双 ubatch 统计确认实际执行两条 lane。默认生成与 prepared 路径分别给出结果 |
| 前缀命中、全重置、短尾、长上下文和取消 | 状态恢复和继续推理正确。内存增长归到真实上下文、已接纳 workspace 或字节预算内的快照 |
| 低余量与支持的流专家/视觉路径 | 验证 domain 内预算、瞬时峰值、驱逐和分配失败恢复。对默认启用相关改动的每条路径做真实板端测量 |
| 后续连续 batching | KV/GDN 和事务状态仍按槽位管理。每条实际执行 lane 持有自己的算子 scratch，槽位调度给同一 lane 的请求依次复用。用真实混合阶段的并行需求扩充预算，复验背压、取消、请求隔离和掉出时的释放 |

建议先交付阶段 0 与 1a 的小改动，然后用阶段 1b 的结果确定保留收益与 RAM/IOVA 代价，再进入 native W4A8。阶段 1a 的收益主要看容量和尾块行为，阶段 1b 的收益主要看重复请求的 TTFT。阶段 2 负责降低常驻成本。性能通过线以同板基线的抖动和样本分布为准。只有对应的日志、资源占用和正确性证据齐全的路径进入默认配置。
