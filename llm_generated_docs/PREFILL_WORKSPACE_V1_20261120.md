# Prefill workspace 跨请求复用首版

基于运行时 `5930ead` 和 native `ec363d9`，在独立 worktree 实现。首轮性能对照覆盖 Dense 及 dense Qwen3.5 AR/MTP；后续接入了 Ling3Tiny 的 TF32 router 和 FP16 mixer 工作区。连续 batching 和 native 跨形状 GEM 数据共享仍沿用各自当前实现。 [审查基线与重构计划](WORKSPACE_MEMORY_REFACTOR_PLAN_20261120.md)

## 内存生命周期

- 量化 Linear 的 padding input/output 由实际执行的 backend/lane 共享。只有逻辑行数小于物理桶时才申请和使用 staging。填充、native 执行和结果回拷都在同一 backend 锁内。正常满桶直接使用调用者的缓冲。两条预填充 lane 保留独立的 staging。 [Linear 执行](../src/npu.rs) · [独立 lane](../src/npu/lane.rs)
- 首个输出头前，目标模型丢弃本轮的尾块 workspace，保留 decode/verify 桶，并评估已创建的常用整块。每个 domain 独立判断，上游 weights、KV、GDN、Attention 与两条 lane 的活跃 owning GEM 容量一起计入。GDN 配合保留该 domain 的整块 scratch。多余形状、超阈值的整块以及长行数 LM head workspace 释放。只有已经创建的整块可以保留，短请求不会为缓存预建一份新整块。 [阶段切换](../src/model/workspace.rs) · [Qwen3.5 GDN](../src/model/qwen35/runtime.rs)
- `RK_NPU_PREFILL_CACHE_DOMAIN_MIB` 可设为 0–1792，默认 1792，用于已加载模型在阶段边界的保留决策。设为 0 时，阶段结束时释放可回收的 prefill 整块；请求内仍可同时使用整块与尾块，Linear staging 与 GEM 计数继续工作。分配出现内存压力时，native 清退当前 domain 中闲置的 workspace 并在释放成功后重试。显式调优、Qwen3.5 外部草稿模型和无预先规划的输入沿用阶段释放。 [Dense 生命周期](../src/model/dense.rs) · [Qwen3.5 选择](../src/model/qwen35/load.rs)
- Native 统计按 context/domain 记录 owning GEM 的 `live_bytes`、`peak_bytes`、`creates` 和 `destroys`。导入 dma-buf 计登记跨度，子视图不重复计入。共享 context 的主、副 lane 显示同一份记录，汇总按 domain 去重。保留阈值基于这个进程通过该 context 登记的 GEM 容量，因此和整个驱动的 IOVA 剩余空间、物理内存和进程 RSS 有各自口径。 [Native API](../native/rk_npu_matmul/include/rk_npu_common.h) · [状态接口](../src/npu.rs)

## 板端对照

在同一块 RK3588 主板上对照，CPU4–7 固定 2.4 GHz、NPU 1 GHz、DDR 2.112 GHz。两版使用相同的 C7 权重、编译器、调优缓存、token 与解码配置。每个进程依次跑两条 512-token 直接请求，再跑 prepared `512→129→512`，所有请求都重置序列状态。首 token 时间为模型侧 reset、reserve、prepare、prefill 含 head 和贪心取样的和。探针在两版使用相同的 logits 捕获边界，散列对比在计时外。

| 第二条 512-token 请求 | Prepare 墙钟 | 模型侧首 token | 目标 domain 15/14 创建 / 销毁 GEM |
|---|---:|---:|---:|
| Qwen3-4B 单 ubatch | 640.4 → 0.02 ms | 3469.9 → 2544.1 ms | 541 / 541 → 0 / 0 |
| Qwen3.5-4B MTP1 单 ubatch | 425.1 → 1.8 ms | 3851.6 → 3216.3 ms | 437 / 437 → 0 / 0 |
| Qwen3.5-4B MTP1 双 ubatch | 828.0 → 1.8 ms | 4092.9 → 2912.5 ms | 874 / 874 → 0 / 0 |

三组每组五条输入的 FP32 logits 和输出 token 一致，共对比 **229,407,872** 个 logits。双 ubatch 首次准备的工作区记账量由 **1473.5 MiB 降到 968.7 MiB**，满块 Linear padding 为 0。第二条请求在首 token 边界的 RSS 为 3953 → 4166 MiB，反映已复用整块的常驻代价。Prepared 双 ubatch 的最后一次 512-token 请求，首 token 为 **4384.7 → 3343.9 ms**，prepare 中仍创建 291 个目标 GEM。127 步 decode 墙钟变化在这三组单样本中均低于 1%。

用同一新版二进制调节保留阈值，对照 Qwen3.5 双 ubatch 的第二条 512-token 请求。

| 阈值 | 保留 domain | 准备创建 / 首 token 前销毁 | 模型侧首 token | 首 token 时进程 RSS |
|---|---|---:|---:|---:|
| 0 MiB | 空 | 874 / 874 | 3948.6 ms | 3486 MiB |
| 1280 MiB | 14 | 402 / 402 | 3408.3 ms | 3853 MiB |
| 1792 MiB | 14、15 | 0 / 0 | 2912.5 ms | 4166 MiB |

两个低预算配置各核对 91,381,760 个 logits，与基线按位一致。阈值控制当前 context 的保留决策。它允许为当前请求临时创建工作区，并在 phase 退出时释放超阈值部分。RSS 为探针进程的采样点数据。三组主对照加两组预算对照的时钟恢复正常。

首轮板端 release 单测通过 174 项。真实设备另外通过 W4/W8 尾块和 SwiGLU、双 lane 共享权重、Native GEM / 子视图 / dma-buf 内存计数，以及 Qwen3.5 MTP 前缀恢复和 GDN 尾块复用测试。各运行步骤已恢复时钟与 governor。 测试日志（本地归档 `evidence/prefill_workspace_v1_20261120/tests.log`）

板端原始阶段统计（本地归档 `evidence/prefill_workspace_v1_20261120/summary.json`） · 运行脚本（本地归档 `evidence/prefill_workspace_v1_20261120/run.sh`）


## 后续回归

Qwen3.5 的 prepare、finish 与 abort 共用同一份资源清单。整合后的板端 release 单测通过 176 项；双 lane 在 0 与 1792 MiB 下均通过取消、未规划追加及同范围旧计划替换测试。MTP（C7）和 DFlash（未旋转的 C4）均通过前缀恢复与分支 logits 对照。Ling resident 与 stream 路径也有整块保留与序列状态恢复的板端用例。上面的耗时和 RSS 取自首轮对照。

分配故障注入已在 prepare、Dense decode 和 MTP verify 三个场景验证。注入一次 ENOMEM 后，检查目标 domain 的清退计数、另一个 domain 的计数和 logits 散列；verify 连续两次分配失败后，下一请求重新 prefill 和验证。可在 RK3588 的仓库根目录运行以下入口，模型与调优缓存取自同一配置。

```sh
tests/board/workspace_reclaim.sh prepare "$QWEN35_C7" "$QWEN35_CACHE"
tests/board/workspace_reclaim.sh verify "$QWEN35_C7" "$QWEN35_CACHE"
tests/board/workspace_reclaim.sh dense "$QWEN3_C7" "$QWEN3_CACHE"
```

[故障回归入口](../tests/board/workspace_reclaim.sh) · [模型断言](../tests/tools/workspace_reclaim.rs) · [原生 Attention 失败路径](../native/rk_npu_matmul/tests/test_attention_f16_pipeline.cpp)

## 本版边界

Prepared prefill 的少见尾块仍按实际规划分配和释放。固定最大 prompt 的重复请求经过预热后可复用常用整块。Native 每个 shape plan 的数据缓冲仍独立持有，下一阶段可以通过跨形状共享进一步降低保留成本。MTP 的浮点 workspace 换形、采样缓冲、验证 logits 复制及前缀快照字节预算各自有独立的后续工作。
