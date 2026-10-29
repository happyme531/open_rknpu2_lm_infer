# 模型运行时通用优化补齐，2026-10-28

基线为 `95047c8`，原生算子库固定为 `3396f77b810a3888684f8648703c4592ea8b3329`。本轮修改不增加 Cargo feature；共享算子和辅助模块始终编译，模型开关仍只负责模型装配。

## 修改

1. **GDN 卷积状态复用。** `ConvState::clone_from()` 复用已有 history 容量，并同步 geometry，供跨层 scratch 使用；GDN checkpoint 恢复调用严格检查 geometry 的 `restore()`，原地复制历史。回归检查 state/history 地址、内容和恢复后的计算。
2. **Ling 专家裁剪参数。** Ling 目前没有实现 `expert_min_weight`，加载配置后立即拒绝非零值，避免参数被静默忽略。默认值 0 的计算保持原样。
3. **统一工作区生命周期。** `src/model/workspace.rs` 提供实际物理行数规划和 prefill/decode 阶段切换。Dense、Qwen3.5/MiniCPM、Ling 只预热实际会执行的 prefill 行数；生成首个 logits 时释放多行 prefill 工作区，只保留 decode/verification 所需形状。GDN 的 NPU prefill 工作区和 Ling 的 TF32 router 工作区同步释放。Ling FP16 投影继续按 geometry 保留一个形状，预热首个实际 batch，后续按需切换。Qwen 已处于 decode 时在构造行数列表前直接返回，避免每步分配临时列表。
4. **恢复 prepared 流式 MoE 的逻辑块。** Qwen3.5/MiniCPM 的流式、无 MTP 路径可将连续物理批次合并为最多 1024 行的逻辑块，复用已有 layer-wise expert walk。检查点标记必定开启新块；user/tail 保存位置及物理 ubatch 不变。模型使用 MTP 或常驻专家时保留原调度。
5. **Ling 模型级 scratch。** hidden、norm、mixer、FFN、router 输出及选择缓冲按模型复用；各层继续独立持有 KDA recurrence、三路卷积状态和 MLA KV。MLA 公共封装复用 FP16 打包缓冲。计算公式、精度和算子执行次序保持原样，增加 `cpu_scratch_bytes` 观测。

## 验证口径

两版均使用同一原生算子源码、板端 Rust 1.95.0 工具链和 release 配置构建。测试独占 `/tmp/rknpu_lowlevel_submit.lock`，未调整时钟，本记录只用于数值和资源生命周期验证，不作吞吐结论。

- 主机 release：144 项库测试、26 项集成测试通过；ARM64 六种单模型及 Ling+Qwen3 组合的 all-targets Clippy `-D warnings` 通过。
- 主板最终源码：严格 Clippy、144 项库测试通过，213 份源码/fixture 哈希与工作区一致。
- GDN、GQA、MLA 三项原生 NPU 回归通过；MLA 增加 33 行 prefill 到单行 decode 的打包缓冲地址/容量保持断言。
- Ling 常驻/流式两种真实模型前缀回归通过，检查完整 logits、KDA/卷积、MLA 分配、20/4 行 FP16 workspace 句柄、分支、取消、reset、容量与禁用状态。
- Qwen3.5 C4 的 AR/MTP/DFlash 三种模式均通过完整前缀与 draft-state 回归，包含计划形状无额外 GDN workspace 创建、decode 释放、NPU TF32/CPU FP32 分支重放。
- MiniCPM 流式前缀回归与 decode 后追加多行 prompt 的专家预取回归通过；prepared 计划确认含超过物理 ubatch 的逻辑块。

固定 token A/B 使用 600-token prompt、`ubatch=128` 和固定续写 `[42,123,89]`，比较 4 行完整词表 FP32 logits，无采样。五组均逐字节一致：

| 路径 | 比较的 FP32 值数 | workspace payload 旧版 bytes | 新版 bytes |
|---|---:|---:|---:|
| Qwen3-4B | 607744 | 508436976 | 7155248 |
| Qwen3.5-4B C4 | 993280 | 13240672 | 17057600 |
| Ling resident | 628736 | 397444512 | 173637168 |
| Ling stream | 628736 | 280679088 | 33247440 |
| MiniCPM stream | 992576 | 20852284 | 20852284 |

payload 取 eval 结束时的 `stats.workspace_payload_bytes`，包含 backend 的 host staging 与 native workspace，不是总 RSS，也不能据此推导全部设备 IOVA 占用。Qwen3.5 本次额外确认 GDN workspace 行数从旧版 `[88,128]` 变为 `[]`。MiniCPM 原本已走 Qwen 的量化工作区释放路径，该字段保持原样。

### 验证发现的已有域预算问题

多次复验时，Qwen3.5 AR/MTP 在准备第二组 `32/64/128` 行桶时出现 `cannot allocate W4 workspace`；内核报告 `rknpu_gem_get_pages: dma map 9437184 fail`。未经本轮修改的 `95047c8` 基线也在相同位置复现，故不属于新的 checkpoint 数值差异。失败后没有 reset 或重启板子。

旧加载策略对 resident 只预留 128 MiB 工作区预算，C4 decoder 的约 1.7 GiB 权重 payload 集中在一个 2 GiB 地址域。现在 resident/stream 均采用原 stream 路径已有的 512 MiB 预留，提前把权重分到后续域，为多个计划形状及 GDN workspace 留余量。修正后的 AR/MTP/DFlash 完整回归均通过此前失败的形状组合。失败记录、基线复现和内核日志保存在 `evidence/model_runtime_20261028/`。

分域可能增加常驻 workspace 和 tuning material 的总量，目标是降低单域地址压力。W4 调优缓存路径带域号，分域改变后应重新核对 cache hit/miss 并补齐调优；本轮使用 `--no-tune`，不作调优后速度结论。

NFS 模型使用主板已有挂载；C4/C7 各约 4.4 GiB，DFlash 约 1.2 GiB。本轮整模型 Qwen3.5 使用 C4，主板 30 GiB 内存、开始时约 25 GiB available，AR 回归观测 RSS 约 4.5 GiB。所有模型串行加载，MiniCPM 在 7.5 GiB 板上使用已有流式专家库。

GDN TF32 在改变物理 prefill 分块时仍可能存在既有数值差异；本轮没有修改该精度策略。前缀恢复验证要求相同执行路径的 logits/state 精确一致。

板端原始产物目录：主板 `/home/firefly/.cache/model-optimization-audit-20261028`；MiniCPM 板 `/mnt/nvme/minicpmv47/results/model_optimization_audit_20261028`。构建快照与辅助脚本位于本机 `target/model-optimization-audit/`，不进入生产提交。

最终程序 SHA256 为 `c66d6bd298176b973e1a10e4b629561895555c1d1b292cb25b6b35be774b8dba`；测试程序为 `ff9b3336bd47589ea728bfeb8e4211848b544d1e9c3754ab23942cbecbab2245`。本地验证汇总为 `evidence/model_runtime_20261028/validation-summary.json`，最终日志集中在 `evidence/model_runtime_20261028/final/`；这些原始验证产物不纳入生产提交。
