# 2026-10-21 当前未提交代码清理

审查范围为本轮开始时的外层仓库与 `native/rk_npu_matmul` 子模块未提交代码，包括 MiniCPM-V 4.7 MoE、专家分页与预取、DSpark、原生专家 blob，以及上一轮 N8128 限制。本轮清理实际修改外层 14 个源文件；子模块源码与本轮开始时一致。没有提交或暂存。

本轮开始前已保存 75 个变更源文件的副本和 SHA256。下面的 [cleanup.patch](evidence/code_cleanup_20261021/cleanup.patch) **仅包含本轮清理相对于开始快照的差异**；完整 `git diff` 还包含原有功能开发，不能将其全部算作这次清理。

## 已处理

| 位置 | 问题 | 清理结果 |
| --- | --- | --- |
| `src/model/qwen35/moe.rs` | 关闭专家预取后仍加载下一层 router，并做 LES router/top-k 计算，最后才丢弃预测 | 只有启用预取才构造预测器 |
| `src/speculative/driver.rs` | 每个草稿 token 都复制整行词表 logits；并行结果已有可独占修改的缓冲 | 并行行就地修正，串行 MTP 直接借用返回值；保留分布与采样语义 |
| `src/cli.rs` | CLI 单独复制 7/15 上限，后面又调用完整配置校验 | 在读取模型配置前统一调用 `SpeculativeConfig::validate`；保留早失败，并补充现有测试中的对应断言 |
| `src/model/qwen35/load.rs` | 重复实现量化 Linear 加载，并为 dense FFN 估算复制整个列表 | 复用 `projection::Linear::load` 和过滤迭代；dense down 同样遵循 protected floating linear 的加载规则 |
| `src/model/experts/{prepare,store}.rs` | 有效专家缓存先打开并校验一次，又重新打开、解析、校验 | 复用已校验句柄，再设置 O_DIRECT；保留双重检查锁和错误传播 |
| `src/model/trace.rs` | 仅测试使用的位置筛选也进入生产结构和保存路径 | 字段与分支限定为 `cfg(test)`；改为命名字段 |
| `src/model/dflash/{config,runtime}.rs`、`src/model/qwen35/features.rs` | DFlash 的 anchor 行与 DSpark 提案行偏移在多处分别判断；DSpark 还预热用不到的第 16 行 | 集中为 `proposal_offset` / `max_proposals`，按实际最大行数预热；移除重复只读 unwrap |
| `tests/tools/minicpmv47_check.rs` | 将某次旧 C4 PPL 常数硬编码为任意当前 checkpoint 的参考，输出缺乏匹配条件的相对差值 | 删除两个参考比较字段；保留当前模型实测 PPL |
| 模型配置与错误信息、DSpark 加载注释 | 仍声称仅支持 dense Qwen3.5；DSpark 使用 domain 2/3 的原因不明确 | 与实际支持范围一致；解释 domain 1 已用于冷专家 staging |

## 核查后保留

- 原生 W4A8 的 K512 分波有 INT16 累加范围依据；专家 bank 的 32MiB 分配边界用于控制多专家 IOVA 分配。两者不是无依据的魔数。
- 专家 blob 版本、slot 所有权与同步、分页 generation/publish 检查，以及 demand 优先于 prediction 的调度保护均有实际调用路径。
- `Linear::SplitN` 用于 DSpark 248144 词表，受原生浮点算子单次 N 上限约束。
- GDN compensated 包装用于诊断比较；旧 C4 加载器仍有明确兼容入口及测试。没有按“当前默认路径没用到”直接删掉功能。
- N8128 限制及 INT4→INT8 canary 保持原样。

## 验证

- `cargo fmt --check`、外层与子模块 `git diff --check` 通过。
- x86 主机 `cargo test --release --locked --lib --tests`：100 通过、18 个硬件/模型测试按默认忽略。日志：[host_tests_latest.log](evidence/code_cleanup_20261021/host_tests_latest.log)。主机仍有 4 条原有 CPU kernel `unused_mut` 警告，本轮没有修改这些无关算子。
- ARM64 `cargo clippy --all-targets --release --locked -- -D warnings` 通过；release benchmark 构建通过。板端自动测试 100 通过，最后一次 CLI 校验位置调整后又单独复验 7 项 CLI 测试，全部通过。日志：[board_final.log](evidence/code_cleanup_20261021/board_final.log)、[final_verification.log](evidence/code_cleanup_20261021/final_verification.log)。
- 最终板端 staging 的 75 个变更源文件与本地 SHA256 全部一致；本轮清理修改列表见 [changed_files.json](evidence/code_cleanup_20261021/changed_files.json)，子模块文件均与开始快照一致。

真实模型对照使用主板 RK3588、CPU 2.4GHz、NPU 1GHz、DMC 2112MHz；模型为 `MiniCPM-V-4.7-35B-A3B-QQQ-W4A8-Mixed-CT`，复用相同已准备专家缓存和 Wiki 文本。每次新建进程，stream 模式、ubatch 128、129-token prefill、生成 4 token。清理前基线使用开始时冻结的旧 benchmark 二进制；其构建目录源码已与开始快照核对。命令和二进制 SHA256 分别记录在 `*.command.json`。

| 配置 | 与清理前基线的输出 token | 与清理前基线的全部 logits |
| --- | --- | --- |
| 清理后，关闭预取 | 4 个完全一致 | 992,576 个 F32 值逐字节一致 |
| 清理后，开启预取 | 4 个完全一致 | 992,576 个 F32 值逐字节一致 |

对照数据见 [after_no_prefetch.comparison.json](evidence/code_cleanup_20261021/after_no_prefetch.comparison.json)、[after_prefetch.comparison.json](evidence/code_cleanup_20261021/after_prefetch.comparison.json)。本轮只据此确认数值一致，不给出性能提升结论。

所有最终板端测试和对照的退出码为 0，`restore_errors=[]`，CPU/NPU/DMC 的频率边界及 governor 均恢复到各次运行前的值。汇总见 [validation.json](evidence/code_cleanup_20261021/validation.json)。原始日志、快照与对照数据位于 [code_cleanup_20261021](evidence/code_cleanup_20261021/)。

真实权重对照只覆盖 MiniCPM-V 4.7 target；本轮没有重新跑 DSpark 草稿 checkpoint 或 dense Qwen3.5 checkpoint 的端到端测试。草稿行数与协议清理覆盖到现有配置和协议测试，不扩大 DSpark 质量或性能结论。
