# RK3588 推理运行时的中间张量与临时缓冲审计

审查基线为 `main` 的 `5930ead` 和它锁定的 native `ec363d9`。该版本的 API 使用一个模型 worker 串行执行请求。双 ubatch 并行的是同一请求的两个完整物理块。以下板测在 Firefly RK3588 主板完成，覆盖 Qwen3-4B 和 Qwen3.5-4B MTP1，其他模型路径依据该版本源码。 [请求 worker](../src/engine.rs#L134) · [ubatch 调度](../src/model/prefill.rs) · [层流水](../src/model/pipeline.rs)

**常规 decoder 的 CPU 激活大多已经跨层复用。现有高开销来自 NPU workspace 的生命周期、算子内部频繁换形，以及部分热路径用堆分配和拷贝传递中间结果。** 这些资源分别由模型、backend 和 native 算子管理。`Workspaces` 统一了行数桶规划和 prefill→decode 的保留规则，实际存储仍由各对象独立持有。量化 Linear 的同形状复用已经存在，跨形状和跨阶段复用空间较大。 [`Workspaces`](../src/model/workspace.rs) · [`BackendInner`](../src/npu.rs#L276)

## 内存由谁持有

| 对象 | 存储与复用 | 何时改变 |
|---|---|---|
| 模型权重 | Safetensors 文件 mmap。量化层加载时解包、重排并复制到所属 NPU IOMMU domain。运行时保留原 mmap、native packed 权重和调优所需小量元数据。双 lane 复用权重和 domain | 模型加载时建立。转换中的解包数组释放。部分节点对文件页调用 `MADV_DONTNEED`，之后读取可重新缺页。启用可选 W8 压缩时，布局变化还有独立的重打包和 `malloc_trim` 路径 |
| CPU decoder 激活 | Dense 的 `Batch`、Qwen3.5 的 `Buffers` 按 ubatch 分配，跨层、跨 token、跨请求复用。双 lane 各持一份。Ling 有独立的模型和 mixer scratch | 加载或首次需要更大容量时建立。模型在 decode 仍保留以 prefill 尺寸分配的 CPU 向量 |
| 量化 Linear | 每个 backend/lane 为 `(bits,K,N,Ktile,bucket(M))` 持有 `Work`。同形状的不同层复用。每项包含 FP32 staging 和 native plan。native 另有输入、部分和、控制缓冲、CPU reduce scratch，部分策略拥有提交线程 | 首次 `prepare` 或调用创建。行数桶不在保留集合时同步析构。SwiGLU 的线程数或归约策略与普通 GEMM 不兼容时可再建一份 native workspace |
| GQA / MLA 的执行 scratch | GQA 每 backend/lane 一份 host FP16 staging 与 native Attention workspace，多层切换各自的 KV handle。native 复用 Q、score、partial、regcmd、descriptor 等高水位缓冲，并维护内部流水 slot。Ling 的 MLA 集成有 layer-owned native workspace 与可增长的 host packing scratch | GQA 行数、可见 KV span、mask 和 LUT 路径变化时可能重建 Graph，旧缓冲能复用的继续用，尺寸不足才换 GEM。执行 scratch 通常跨请求保留 |
| GDN / KDA | 每层各自持有 canonical recurrence、卷积历史和事务所需状态。GDN 普通路径原位推进，CPU scratch 跨层复用。支持的 Qwen3.5 GDN prefill 为每 lane 每个实际行数建立 native workspace，把 canonical FP32 状态导入、导出。Ling KDA 使用共享 CPU execution scratch | GDN decode 小批走 CPU，进入 decode 后释放大行数 NPU scratch。备选状态、更新日志在 verify / rollback / replay 使用，按缓存高水位保留 |
| GQA / MLA KV 和前缀快照 | KV 每层持有。Dense GQA 以较小容量创建并按需要 reserve，Qwen3.5 GQA 分层按上下文容量预留。前缀复用保留原 KV 并 truncate，并为 GDN/KDA 保存 CPU recurrence 和卷积副本 | 冷请求 reset，前缀命中则截断或恢复。KV 容量不因 reset 缩小。快照按上下文位置和条目数管理，默认上限 32 |
| MTP / DFlash / DSpark | 独立 draft KV、浮点 Linear workspace 和模型 scratch。浮点 `FloatBackend` 一个 `(K,N)` 只保留一个精确 M 的 workspace。Qwen3.5 MTP 在 target prefill 中 catch-up，decode 中两边各有事务暂存 | M 变化可在正在执行的请求内新建 workspace 并销毁旧项。DFlash/DSpark layer.forward 还持有调用级临时 Vec |
| MoE 与视觉 | MoE grouped 算子在 backend 保留 `moe_work` 和自己的固定容量 scratch。流专家有热槽、共享冷槽、有限的读缓冲和 staging 池。ViT 每次实际编码分配一组 FP16/FP32 激活，跨视觉 blocks 复用，并建 image Attention。图像特征另有 32 MiB 缓存 | MoE 的 `moe_work` 不跟随普通 Linear 的行数裁剪。流专家 staging 按请求和层复用。视觉激活在编码结束离开作用域，浮点 Linear 和已缓存视觉特征有各自生命周期 |

[模型权重 mmap](../src/model/checkpoint.rs#L132) · [Dense buffer](../src/model/dense/batch.rs#L17) · [Qwen3.5 buffer](../src/model/qwen35/decoder.rs#L59) · [执行 lane](../src/npu/lane.rs) · [GDN 状态](../src/layers/gated_delta/state.rs#L138) · [GDN workspace](../src/npu/gated_delta.rs) · [Ling scratch](../src/model/ling3tiny/mixer/scratch.rs) · [MLA](../src/npu/mla.rs) · [MoE](../src/npu/moe.rs) · [流专家缓冲](../src/model/experts/paged.rs) · [视觉激活](../src/model/qwen35/vision/encoder.rs#L114)

## 从一条请求到下一条

1. **加载**。建立模型权重、CPU 激活、各后端和部分小行数 workspace。量化权重按 domain 切分。Dense 单 lane 的权重加 KV 预算为 2 GiB 减 128 MiB，双 lane 为 1 GiB。Qwen3.5 dense/streamed 使用 1 GiB 权重与 KV 预算，resident MoE 另用 2 GiB 减 256 MiB 的策略。实际 IOVA 消耗还受对齐、GEM 分布、其他 native scratch 和跨 lane 共享影响。权重预算和总 RSS 均不足以直接计算各 domain 的剩余空间。 [Dense 加载](../src/model/dense.rs#L234) · [Qwen3.5 加载](../src/model/qwen35/load.rs)
2. **reset / reserve**。API worker 在无法复用前缀时 reset，再 reserve `prompt + max_tokens`。Dense 的 KV 只增不减，首次增长会申请新 K/V 并清零，保留有效历史时还会复制有效数据。Qwen3.5 的 target GQA 在加载时预留整个上下文。现行 native reset 通过 `cache_load(count=0)` 清零并同步整块已经分配的 K/V，重置 LUT 元数据。GDN 清 canonical state 和卷积历史，丢弃快照。 [引擎请求路径](../src/engine.rs#L299) · [Attention reset](../src/npu/attention.rs#L227) · [native Cache reserve/load](../native/rk_npu_matmul/src/rk_npu_attention_f16_cache.cpp#L65)
3. **prepare**。将 prompt 计划转换为物理行数集合，加入 decode 与 verify 行数。普通量化 Linear 将行数向上取桶，并为该 backend 已加载的各几何准备所需桶。普通生成的 LM head 只准备小行数。第二 lane 只准备满块。GDN 的 NPU workspace 按实际 M 管理。设置了前缀检查点时，512 行 Qwen3.5 会按 `128 + 128 + 124 + 128 + 4` 执行，在位置 380 和 508 保存检查点。Linear 的 124 行落入 M128，GDN 建 M124 workspace。该计划只形成一对双 lane 满块。 [物理行数](../src/model/workspace.rs) · [Qwen3.5 prepare](../src/model/qwen35/runtime.rs#L63) · [前缀计划](../src/model/prefix_cache/policy.rs) · [执行检查点](../src/model/qwen35/multimodal.rs)
4. **prefill 和首 token**。中间激活在 `Batch/Buffers` 内切片传递。Q/K/V 拆分、转置、FP32→FP16 打包及 NPU cache ownership 同步仍会产生流量。量化 Linear 的 `M == rows` 直接传 caller 的 FP32 指针，尾块才复制到 staging、填充和回拷。每个 `Work` 仍预分配了独立的 FP32 input/output，即使常见的 M128 整块直通。Qwen3.5 的 MTP catch-up 另外复制 embedding、target hidden、shifted seed 和位置。生成路径在首个 head 前同步回收不再使用的大行数 Linear/GDN workspace。 [Linear staging](../src/npu.rs#L1169) · [GQA bridge](../src/npu/attention.rs#L333) · [MTP catch-up](../src/model/qwen35/mtp/mod.rs#L219) · [回收调用](../src/model/qwen35/head.rs#L41)
5. **decode 和请求结束**。常规 AR 的 Linear M1、激活及 logits 主缓冲可复用。Attention 随 KV 可见长度变化重绑或重建 Graph，复用其底层大缓冲。投机解码还会复制验证 logits、fork 两份 sampler 并准备相应 M 的 draft workspace。请求结束后，模型 worker 保留已提交状态和可复用前缀，直到下一条请求决定复用或 reset。下一条冷请求的 prepare 会重新构造首 token 前已释放的大行数 workspace。 [投机解码](../src/speculative/driver.rs#L60) · [sampler](../src/sampling.rs#L88) · [Attention Graph 重建](../native/rk_npu_matmul/src/rk_npu_attention_f16.cpp#L940) · [引擎提交状态](../src/engine.rs#L434)

## 容易漏算的开销

- **准备阶段会做与真实算子可达性无关的工作。** 当前枚举按 backend 的已注册几何 × 需要的行数桶建立计划。Native W4 的数据缓冲和任务/寄存器缓冲跟着各 shape 计划走，同一 lane 内顺序执行的几何彼此独立持有。提交线程也跟随部分 W4 workspace 生命周期。现有的 quantized padding 缓冲可以按 lane/backend 复用，因为持有该 backend mutex 的调用已经串行。Native 的 A/partial 可以在继续独立持有形状描述符的条件下按同时活跃区间复用。Fused FFN 使用的 scale 等跨算子活跃数据需要保留。 [`get_work`](../src/npu.rs) · [W4 plan](../native/rk_npu_matmul/src/rk_npu_i4_plan.cpp#L127) · [W4 executor](../native/rk_npu_matmul/src/rk_npu_i4_executor.cpp#L9) · [FFN 数据依赖](../native/rk_npu_matmul/src/rk_npu_w4a8.cpp#L354)
- **malloc 次数和内存流量是两类成本。** Dense 和 Qwen3.5 的多数激活 Vec 已复用，但合并 QKV 拆分、规范化后的复制、GQA host staging 与 native pack 仍逐层发生。Qwen3.5 还为 trace/非融合路径预留 `gate_up` 和 scatter。DFlash/DSpark 的浮点 SplitK/SplitN 会在各次调用创建 packed input 或 partial，draft layer 有逐层 Vec，MiniCPM streamed logical prefill 则按逻辑块新建 hidden/normalized/routed 大数组。Ling 已经把许多 KDA/MLA 计算缓冲提到长期 scratch。 [Dense QKV 拆分](../src/model/dense/batch.rs#L123) · [Qwen3.5 QKV](../src/model/qwen35/decoder/mixers.rs#L96) · [SplitK/SplitN](../src/model/projection.rs#L139) · [DFlash layer](../src/model/dflash/layer.rs) · [MoE logical block](../src/model/qwen35/prefill.rs)
- **MTP1 的热循环同时有 host 和 native 的周转。** `Sampler::clone` 复制全词表计数并预留候选数组。每轮 fork draft sampler、clone target history，再把验证 logits 整体复制到 `VerifiedRound`。以 vocab 248320、一次两行验证为例，这几项合计请求约 7.58 MiB 堆容量，包含只预留容量的数组。engine 最终取被消费位置的一行 logits，并维护完整的停止/取消语义。MTP 浮点 backend 在同几何切换精确行数时，会在运行中换 workspace。 [投机 driver](../src/speculative/driver.rs#L73) · [sampler 内存](../src/sampling.rs#L88) · [engine 取有效 logits](../src/engine.rs#L434) · [FloatBackend](../src/npu/floating.rs#L18)
- **KV reset 全量清零的成本跟容量走。** 现有 truncate 处理 LUT 正偏置、V 的分母通道、prefix sum/absmax 和 tail epoch。若缩小 reset 清理范围，需要覆盖这些约束、旧的 padding 数据以及 poisoned cache 的恢复。直接把 reset 替换成只写一个 length 会改变这些条件。 [native truncate](../native/rk_npu_matmul/src/rk_npu_attention_f16_cache.cpp#L200)
- **模型的 `--stats` 是分项估计。** `workspace_payload_bytes` 混合 Linear native memory-query 的估算值、Rust staging capacity 和 grouped MoE workspace。它遗漏了独立的浮点 workspace、GQA native Graph、GDN native scratch 等实际用量。`workspace_count` 只数普通 Linear map 条目。`attention_cache_payload_bytes` 是 KV 数据和 tail 的 payload，`mapped_weight_bytes` 是文件映射跨度，`rss_kib` 是进程 RSS。三者加总无法得到 IOVA 峰值。 [backend bytes](../src/npu.rs#L716) · [模型聚合](../src/model/metrics.rs) · [Qwen3.5 stats](../src/model/qwen35/stats.rs)

## 板端分阶段测量

测量代码在隔离快照中记录 Rust `GlobalAlloc` 的申请次数和请求字节，并在 `rk_npu_mem_alloc/free/sync` 记录各 domain 的成功次数、返回容量和调用时间。GEM 销毁分开记录 `munmap` 与 `MEM_DESTROY`。模型和 native 控制流保持一致，计数与时钟读取有少量观测开销。Rust 申请量包括 `Vec::with_capacity` 与 realloc 新尺寸，不代表写入字节或 RSS。Native C++ 普通 heap、导入 dma-buf、内核内存及 IOVA 碎片分别需要其他观测方法。多线程的调用累计时间也可能与同阶段墙钟重叠。 采集脚本（本地归档 `evidence/temp_memory_audit_20261120/instrument.py`） · 复现实验（本地归档 `evidence/temp_memory_audit_20261120/run.sh`）

CPU4–7 固定 2.4 GHz、NPU 1 GHz、DDR 2.112 GHz，物理 ubatch128，沿用 W4A8 C7 权重与已有策略缓存。每种配置在一个进程顺序执行两次直接 512 行、一次正常 prepared 512 行、一次 prepared 129 行、再一次 prepared 512 行，每次请求 reset 并生成 17 个 token。各配置的同输入同路径输出相同。下表取第二条直接 512 行请求，已经过首条请求但再次建立自己的 KV 历史。三组在该样本的在线调优及策略回退计数均为 0。

| 配置 | KV reset | prepare 墙钟 | prepare 创建 GEM 数 / MiB | GEM 创建调用累计 | prefill 含首 token 阶段释放 GEM 数 / MiB | GEM 释放调用累计 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B 单 ubatch | 8.4 ms | 625.7 ms | 541 / 501.2 MiB | 462.2 ms | 541 / 501.2 MiB | 272.7 ms |
| Qwen3.5-4B MTP1 单 ubatch | 15.2 ms | 431.2 ms | 437 / 343.0 MiB | 322.4 ms | 453 / 343.6 MiB | 187.9 ms |
| Qwen3.5-4B MTP1 双 ubatch | 14.8 ms | 830.8 ms | 874 / 686.0 MiB | 611.2 ms | 890 / 686.6 MiB | 369.7 ms |

- **Qwen3.5 双 ubatch 的重复周转主要集中在目标模型两个 domain。** Domain 15 和 14 分别创建并释放 402 和 472 个对象，约 686 MiB。prefill 含首 token 阶段的另外 16 次释放来自 MTP domain。全阶段的释放时间中，`MEM_DESTROY` 占 337.8 ms，`munmap` 占 31.8 ms。仅 GEM 创建和释放就累计约 **981 ms**。之前的 README 测法在 prepare 完成后才计 prefill，约 831 ms 的 prepare 墙钟对该吞吐口径不可见。
- **稳态 AR 的激活和执行缓冲已经得到复用。** Qwen3 第二次 prefill 含 head 的 Rust 堆申请约 0.02 MiB，仍销毁 541 个 GEM。随后的 14 步单 token decode，Rust alloc/realloc 共 3 次、GEM create/destroy 均为 0。Qwen3.5 MTP1 的后续 14 个输出则有约 62.76 MiB Rust 堆请求，以及 96 次 GEM 创建和 96 次销毁，全部 create/destroy 记录在 MTP 的 domain 11。该组的小 GEM create/free 合计约 5 ms，host 堆请求量的拷贝和执行代价另需做单变量对照。
- **检查点会增加大行数 workspace 集合。** Qwen3.5 双 ubatch 的 prepared 512 行在连续请求中每次创建 1,165 个 GEM，`prepare` 约 701–712 ms，之后释放 1,223–1,235 个 GEM。两份 GDN/MTP 前缀快照共 **100.5 MiB**，单条请求新增的快照保存墙钟在两个样本分别为 **46.3 ms 和 10.0 ms**。该输入还触发 48 次尾形状策略回退，不能用整段 prefill 的差值单独估算快照成本。按每份约 50.3 MiB 估计，32 份等大小快照的 payload 约 1.57 GiB，实际条目数取决于前缀规划与复用。
- **还存在复制和同步带宽成本。** Qwen3.5 双 ubatch 的第二次直接 prefill 含首 token 阶段记录 6,483 次 GEM sync，累计约 427 ms。其 steady decode 的 5,034 次 sync 累计约 24 ms。sync 统计覆盖所有 native 算子的 CPU/device 交接，包含两 lane 可能重叠的时间。两组 KV reset 样本约 8–15 ms，分别保留约 131 MiB 和 137 MiB KV。该结果对应本次容量，长上下文的全量清零另需测量。

全部样本、计数和时钟恢复（本地归档 `evidence/temp_memory_audit_20261120/summary.json`） · Qwen3.5 双 ubatch 按 domain 原始记录（本地归档 `evidence/temp_memory_audit_20261120/qwen35-dual.json`） · Qwen3.5 单 ubatch（本地归档 `evidence/temp_memory_audit_20261120/qwen35-single.json`） · Qwen3 单 ubatch（本地归档 `evidence/temp_memory_audit_20261120/qwen3-single.json`）

## 改动顺序

1. **先减少每请求 GEM 创建和销毁。** 引入按 domain 和 lane 的常驻预算，优先保留常用 Linear 桶和 GDN 满块 scratch。逐阶段区分候选使用形状、首次冷启动与已有工作区。对容量紧张的配置，允许在分配 admission 时驱逐闲置项。以同进程连续等长及短长交替请求验证 `prepare`、TTFT、首次 token 间隔、GEM 数、每 domain 容量和 RSS。权重、KV、live workspace 和 in-flight 提交都要纳入容量条件。
2. **把常用 workspace 的数据量压下来。** 先把量化 Linear 的 Rust padding staging 提到 backend/lane，按实际尾块申请。随后从 W4A8 入手，把数据缓冲所有权从独立 shape plan 中拆出，按同 domain/lane 的最大活跃需求分配 A/partial 与 CPU reduce scratch，利用现有 `rk_npu_mem_view`。形状仍决定 tiling、控制描述符和数值路径。任务缓冲保持 driver 所需的独立基对象。先测 M1/M128 和两个几何交替，再测双 lane 与融合 FFN。 [native 子视图规则](../native/rk_npu_matmul/include/rk_npu_common.h#L164)
3. **针对稳态的独立周转源改小范围接口。** 浮点 draft backend 可保留实测高频的小 M，并明确其 IOVA 上限。复用 sampler 的候选和历史 scratch，验证 logits 用 caller 管理的可复用缓冲，或在保证 abort、stop、EOS 语义的时点仅保留所需行。正交验证 greedy token 与随机抽样复现。再处理 Attention reset 的清理范围，以及大 C++ 临时向量和非连续拷贝热点。
4. **把预算和实测口径补齐。** 区分实际 GEM owner 容量、跨对象 subview、driver 地址空间、Rust/native host heap、文件映射与驻留页。报告 prepare 与回收 wall time、按 domain 的 GEM create/destroy/sync 次数和高水位。前缀快照的缓存要有字节观测。长上下文、流专家、多模态和连续 batching 在各自真实配置中验收。

本轮的诊断范围为当前已提交基线的串行请求调度。测量覆盖上述两个 C7 checkpoint 的文本生成，静态盘点另外覆盖 Llama/Granite 所走的 Dense 路径、MiniCPM/Qwen3.5 MoE、可选 Ling、DFlash/DSpark 和 ViT。各路径的绝对耗时、长期并发占用及拟议缓冲复用收益需用相应用例测量。
