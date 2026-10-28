# Runtime 模块组织

日期：2026-10-17，Asia/Shanghai。

2026-10-28 模型开关边界：Cargo 只提供 `model-*` 家族选择和 `all-models` 组合。条件编译集中在 `model.rs` 的模型入口、模型注册表及模型专属测试工具；共享算子、层、投影、专家调度、预填充、视觉处理和草稿公共实现统一编译，接口和字段不随目标模型选择变化。模型目录保留配置、权重映射和能力接入；运行时判断所选模型是否支持对应能力。新增模型不得向共享模块添加模型家族条件。

2026-10-18视觉扩展：通用输入、图像预处理、LayerNorm/GELU、FP16 Linear分别新增于 `input.rs`、`vision/`、`layers/layer_norm.rs` / `ops/gelu.rs`、`npu/floating.rs`；Qwen3.5目录只放ViT配置/权重/编排。API请求解析及前缀身份管理分别拆至 `api/request.rs`、`engine/context.rs`，视觉功能与验证版本边界见 [视觉记录](QWEN35_VISION_PROGRESS.md)。下文重构数字与验证是10月17日的历史记录。

独立测试和诊断工具源码统一放在 `tests/tools/`，通过 Cargo.toml 显式注册，沿用 `--example NAME` 构建/运行入口；默认测试不执行这些需要外部数据或设备的程序。历史消融和插桩工具移至 `scripts/archive/`，目录与依赖见 [FP16/ViT 实验归档](FP16_VIT_EXPERIMENT_ARCHIVE.md)。

2026-10-24 SSM 前缀缓存：`model/prefix_cache.rs` 和 `model/prefix_cache/` 统一管理 Qwen3.5 与 Ling3tiny 的请求边界调度、同位置快照复用、容量淘汰和统计。模型私有的 `qwen35/cache.rs`、`ling3tiny/cache.rs` 分别负责 GDN/MTP 与 KDA/卷积快照的保存和恢复；Attention/MLA KV 保留分配并截断。模型前向使用同一份 `PreparedPrefill` 计划完成 workspace 准备和实际执行。

ViT 的 FP16 CPU 算子统一在 `ops/vision_half.rs` 和 `ops/vision_half/`，使用 Rust NEON/窄范围内联汇编及模型自身的 Rayon 线程池。此前新增的 `native/cpu_kernels/vision.cpp`、`vision_rotary.cpp` 和对应 FFI 已移除；`native/cpu_kernels/` 只保留已有的 GDN 内核。

本轮将 Qwen3.5 接入时放在模型目录内的公共实现提取出来。依赖方向为模型适配层 → 可复用层 → 数值算子；公共算子和 GDN 层不导入模型配置、checkpoint、NPU backend 或 engine。

| 目录/模块 | 职责与边界 |
|---|---|
| `src/ops/activation.rs` | 稳定 scalar sigmoid/softplus/SiLU、NEON SiLU、sigmoid gate |
| `src/ops/norm.rs` | FP32 L2/RMSNorm、zero-centered 参数、SiLU gated norm |
| `src/ops/causal_conv.rs`、`causal_conv/` | 带历史的 depthwise causal convolution + SiLU；通道数和 kernel 长度由调用者传入 |
| `src/ops/gated_delta.rs`、`gated_delta/` | 归一化 Q/K 后的 recurrence、形状/数值检查、scalar oracle、无 query replay；不负责卷积或权重命名 |
| `src/ops/rope.rs` | 显式维度/theta/sections/interleaved 参数的 partial 三轴 M-RoPE；不解析 JSON |
| `native/cpu_kernels/` | 独立 C++ recurrence、NEON 实现与窄 C ABI；不依赖 Rust、模型或 NPU |
| `src/layers/gated_delta/` | 卷积、Q/K norm、gate 构造、recurrence、output norm 的组合；每层状态、verify/commit/abort、compact log 与跨层 scratch |
| `src/model/output_head.rs` | Dense 和 Qwen3.5 共用的量化/浮点输出投影及 backend 生命周期；不选择 tied 权重，不做最终 norm/模型缩放 |
| `src/model/trace.rs` | 两个模型共用的可选 FP32 tensor trace 写出 |
| `src/model/projection.rs`、`cpu.rs`、`metadata.rs`、`metrics.rs` | 模型共用的量化投影、CPU 线程池/Tensor 适配、元数据与 backend 统计 |
| `src/model/qwen35/` | Qwen 配置/权重映射、RoPE 默认值、Q/gate 布局、混合 decoder 编排、LanguageModel 实现和模型统计 |

保留 `model/cpu.rs` 是因为它直接消费 checkpoint `Tensor`，提供线程池和模型公共执行适配。其原有优化 norm/RoPE 与新提取的固定 scalar reduction 路径具有不同浮点顺序，本次不强行合并算术。每个模型的最终 norm、tied/explicit 输出权重优先级、head row 准备和输出缩放保持原有语义。

`layers/gated_delta` 是 gated delta 架构的公共层，其他线性注意力算法需要自己的层实现；不引入暂时没有消费者的通用 mixer trait。`LayerState` 的事务只覆盖本层 recurrence/卷积，整模型 KV/hidden/MTP 原子提交仍属于后续工作。

公共测试入口为 `tests/cpu_layers.rs` / `tests/cpu_layers/`，仅通过 `ops`、`layers` API 测试算子和状态；Qwen 的配置和 Q/gate 布局测试留在模型内。现有 Transformers fixture 位于 `evidence/qwen35_mtp/`，保留生成来源，并不要求公共实现依赖 Qwen 模型。

独立工具入口为 `tests/tools/gdn_bench.rs`、`scripts/gdn/ggml_reference/`、`scripts/gdn/cpu_reference/`、`scripts/gdn/check_case.py`、`export_cases.py` 和 `summarize.py`。原11个 NPZ、CPU 源码快照与历史性能记录保持原哈希；NPU kernel 实验可继续使用原数据包，新维护入口见 [GDN 复现说明](QWEN35_GDN_KERNEL_REPRO.md)。

Qwen3.5 模型目录由21个 Rust 文件、2,857行缩至12个文件、1,510行；本次重组的算子、GDN 层、输出头、trace 与剩余 Qwen 模块均不超过275行。

重构验证结果记录在 [evidence/module_reorg_20261017/](evidence/module_reorg_20261017/)，包括 Release 检查、独立 GDN 数据校验和两种模型的重构前后数值对照。原始 logits 保留在共享实验目录，仓库保存哈希、逐 token NLL、命令、日志和时钟恢复证据。

## 重构验证

备用1，CPU4 / `0xf0` / 2256 MHz、NPU mask7 / 1 GHz；DDR 未受控。旧二进制 SHA256 为 `90770d45181b40a619e2613b8f876f9f78111f3e397f15e462153e2c930e8876`，新二进制为 `fbd441571915c2c6729cb690e6d40ed5590192a3986c7db20b7ae1d5dc642693`，均为 Release。

| 检查 | 结果 |
|---|---|
| fmt、全 target release Clippy `-D warnings` | PASS |
| release 默认测试 | 54 passed、0 failed、7 ignored；其中42个库测试、11个公共 CPU 层集成测试、1个 OpenMP startup 测试 |
| 额外显式 NPU 算子测试 | 4 passed；包含 C7 D256 GQA 和 N32/N64 投影 |
| Qwen3.5：129-token prompt + 4-token continuation | 5行、1,241,600个 logits 位级一致 |
| Qwen3：129-token prompt + 4-token continuation | 5行、759,680个 logits 位级一致 |
| 两模型批量评分：各2个512-token窗口 | 各510项 NLL 逐项一致；覆盖 head-free prefix、128行输出头和尾块 |
| 独立 CPU GDN reference | 原11个 NPZ 全部通过；x86_64 scalar fallback 使用原数值门槛 |
| 新路径导出数据包 | 11个 NPZ 哈希与原包完全一致；导出包独立编译及 stress128 校验通过 |
| 时钟恢复 | 全11次 runner 正常退出，受控频率上下限和 governor 恢复 |

Qwen3.5 评分输入取原32窗协议的前2窗，并确认旧二进制重跑结果与原记录逐项一致。Qwen3 使用同一输入取 `%10000` 的固定合法 token IDs，只验证同模型重构前后等价，不用于跨模型 PPL 质量比较。两模型输出头的权重位宽、tuning/reload 行为保持不变。

汇总入口：[validation.json](evidence/module_reorg_20261017/validation.json)。当前77个源码/构建/工具文件哈希与二进制身份见 [source-manifest.json](evidence/module_reorg_20261017/source-manifest.json)；recurrence 源码仅变更路径/标识符的对照见 [kernel-source-equivalence.json](evidence/module_reorg_20261017/kernel-source-equivalence.json)。

完整32窗 PPL 与完整性能矩阵沿用此前带独立二进制身份的测量记录；本轮只做上述重构回归，不将2窗对照当成新的32窗验证或性能测量。
