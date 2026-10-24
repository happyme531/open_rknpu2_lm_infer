# Qwen3.5 NPU GDN prefill

Qwen3.5 的实际调用窗口 `M > 12` 使用算子库 NPU GDN，`M <= 12` 使用现有 CPU FP32 recurrence。路由依据每次 decoder chunk 的实际行数：长 prompt 的短尾部仍可走 CPU，单 token decode 保持 CPU。

当前算子库支持 Hk16/Hv32、Dk128/Dv128、K4 Conv，覆盖 C7。其他维度继续使用通用 CPU mixer。所有 canonical state 保持 FP32 `[value_head,value_channel,key_channel]`，key 连续，value head `h` 使用 key head `h/2`。

## 执行与所有权

`src/npu/gated_delta.rs` 封装公开 `rk_npu_gdn_run_projected_strided` ABI，复用模型最后一个权重 domain 的 Backend。各层共享 Scratch 中的工作区。模型 prepare 按本次执行计划预建全部所需块长，下一次 prepare 释放不再使用的形状；直接 forward 遇到未准备的尾块时，沿用保留主窗口并替换旧尾块的按需策略。不进行在线调优。

合并投影通过带步长视图直接提供 QKV/Z/a/b，输出 gated RMSNorm 直接写入 decoder 的输出投影缓冲区。输入为原始 projected QKV、K4 Conv 权重、FP32 log decay／sigmoid beta、卷积历史与 canonical state。算子库的 CPU producer 将 Conv＋SiLU＋固定顺序 Q/K L2 直接写入 native 输入；NPU TF32 乘法计算 Gram、state prediction、输出与状态增量。衰减系数、三角求解和输出收集由算子库 CPU 部分完成；投影后 gate 变换、output gated RMSNorm 仍在 Rust。

使用 ordinary TF32，state 存储和 DPU 状态合并为 FP32；这不等于全精度 FP32 GEMM。内部块长使用 `next_power_of_two(M).clamp(16,64)`：M13..16 为 chunk16，M17..32 为 chunk32，更长窗口为 chunk64；CPU worker 和 NPU mask 沿用模型配置。每次调用都导入当前层的 canonical state 和历史，成功后导出。NPU prefill 后可直接交给 CPU decode，不保留跨层隐式设备状态。native 输出失败时不提交 state/history，workspace 会释放，错误交给模型已有的 reset 路径。

verify／部分接受 replay 保持 CPU，沿用现有事务日志。prompt checkpoint 在 prefill 边界保存，与验证事务日志独立。

## 统计

- `gdn_backend`：`npu-tf32-prefill/cpu-fp32-decode`，不支持上述形状时为原 CPU backend。
- `gdn_cpu_max_rows`：12。
- `gdn_npu_precision`：`tf32`。
- `gdn_npu_calls/tokens`、`gdn_cpu_calls/tokens`：本次模型 reset 后正常 forward 成功执行的调用/行数，累计所有 GDN 层；verify/replay 不计入。
- `gdn_npu_ms`：native 完整调用累计时间，含 CPU producer、状态导入导出和同步；`gdn_npu_hardware_ms` 是 native hardware 时间。
- `gdn_npu_prepare_ms/coefficients_ms/solve_ms/collect_ms`：native run 内的 CPU producer、系数构造、三角求解、输出收集计时；`prepare_ms` 不含工作区创建。
- `gdn_split_ms` / `gdn_output_copy_ms`：trace 开启时 QKV 紧凑化，以及输出复制的累计时间。正常推理直接读取合并投影视图，输出直接写入调用方缓冲区，两项均为 0。
- `gdn_layer_prepare_ms`：形状/状态检查及当前层卷积历史复制；`gdn_buffers_ms` 是 mixer 输出和 gate 缓冲区 resize 时间。
- `gdn_gates_ms` / `gdn_output_norm_ms`：log decay/beta 生成，以及输出 gated RMSNorm 的累计时间。
- `gdn_npu_workspace_rows`：当前驻留的实际窗口形状；准备好的计划可以同时保留多个尾块。
- `gdn_npu_workspace_creations`：本次模型 reset 后成功创建工作区的次数；执行前后取差可检查初始化是否落入 prefill，与 native run 内的 `gdn_npu_prepare_ms` 分开。
- `gdn_npu_chunks`：各驻留窗口的实际内部块长。

原有 `gdn_ms` 仍是完整 mixer 阶段，包括投影拆分和输出 gated RMSNorm。NPU GDN 的内部 CPU/device 分配未计入原 Backend 的量化 workspace 字节字段。

模型热路径不扫描 projected QKV/Z/a/b 或 CPU recurrence 的 Q/K/V 来检查有限值。保留形状/长度检查、verify 事务状态检查和小尺寸 gate 范围检查；静态权重在加载时检查。

## 前缀缓存

前缀检查点的位置与淘汰规则参考 llama.cpp `7894e8b92481`。在逻辑批次执行前保存，优先覆盖最后一条 user 消息起点及距 prompt 末尾 `4+ubatch`、`4` 个 token 的位置。其他 user 边界受最小间距约束；最后一条 user 消息和尾部检查点可以绕过间距限制。实际执行按 physical ubatch 拆分，不按每个 microbatch 保存。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `serve --cache-prompt` / `--no-cache-prompt` | 开启 | 设置请求的前缀复用默认值 |
| `--ctx-checkpoints`，别名 `-ctxcp`、`--swa-checkpoints` | 32 | 检查点数量上限；非正值关闭保存 |
| `--checkpoint-min-step`，别名 `-cms` | 8192 | 最小间距；0 表示不限制间距 |

服务请求的 `cache_prompt` 为 `true/false` 时覆盖服务默认值；省略或 `null` 时继承默认值。关闭复用仍保留本次计算的新状态，后续请求可以重新启用复用。环境变量使用上游名称 `LLAMA_ARG_CACHE_PROMPT`、`LLAMA_ARG_CTX_CHECKPOINTS`、`LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT`，命令行覆盖环境变量。兼容旧的 `LLAMA_ARG_NO_CACHE_PROMPT`，其存在即关闭默认复用；命令行开关按最后一个生效。现有 context/ubatch 参数也接受 `LLAMA_ARG_CTX_SIZE`、`LLAMA_ARG_UBATCH`。

CPU/NPU 共用检查点中的 FP32 GDN 状态、卷积历史和位置；MTP 同时恢复 seed，target/draft KV 保持驻留并截断到恢复位置。相同活动分支的同位置快照直接复用并刷新请求标记，避免重复复制与误淘汰更早的检查点。C7 的一个 GDN/卷积快照载荷约 50.25 MiB，数量上限不代表预先分配。

准备阶段生成一份借用输入的执行计划，执行阶段直接消费。ChatML user/tool 分隔符在 tokenizer 加载时编码；仅消费这些边界且开启检查点的模型扫描消息边界。当前服务仍由单个推理 worker 持有一份活动模型状态。

**数值 NOTE：** 检查点会改变实际 prefill 分块。已验证 CPU FP32 与 NPU TF32 的恢复结果，各自与同一分块路径重算的完整 logits、GDN/卷积状态一致。NPU TF32 在不同分块路径下可能出现 greedy 输出差异；已复现的 427/428-token 公共前缀分支中，CPU FP32 GDN 对照没有该差异。本次保留缓存策略并记录此限制，没有修改 GDN 数值算法。

`prefix_restore_matches_all_logits_and_draft_state` 覆盖 AR/MTP/DFlash、CPU/NPU 状态恢复、`[468,596]` 的同位置复用和后续分支恢复 468。工作区回归覆盖 `[20,74,118,128]`、`[20,52,128]`、`[84,128]` 三组计划，检查 prefill 创建计数增量为 0，以及旧尾块回收。另有软件回归覆盖调度位置、分隔符优先级、参数覆盖和非法 gate 不修改状态。

缓存验证使用 spare1、CPU2.256GHz/NPU1GHz、DDR 未固定、`--no-tune`。原始请求、输出和状态诊断保存在忽略目录 `llm_generated_docs/evidence/ssm_prefix_cache_20261024/`、`cache_controls_20261024/`、`cache_review_20261024/`、`gdn_workspace_20261024/`。这些是功能与数值验证，下面的历史调优性能数据不代表新增缓存的吞吐基准。

## 验证

`npu_prefill_threshold_state_handoff_and_layer_reuse` 覆盖 M12 的 CPU 输出/state 逐 bit 一致；M13/32/64/128 以及短窗口交错、两层共享、不同 Conv 权重、canonical state 和 history、NPU 后的 CPU decode、verify/commit、非法派生 gate 失败不提交状态。CPU 接续从相同导出状态开始时逐 bit 一致。该按需调用序列保持至多两个窗口。

真实模型对照以外层 `6143940`、算子库 `a9a2e06` 的冻结 CPU GDN 二进制为基线，同 C7 权重和同一调优缓存。精度用真实文本 128-token prompt＋16 个固定 continuation 的全量 logits，以及原有 C7 PPL 数据的前两个 512-token 窗口、每窗 255 个目标，合计 510 个 NLL。

性能使用固定 CPU2.4GHz／NPU1GHz／DDR2.112GHz、CPU4 `0xf0`、NPU7，以 `CPU128 → NPU128 → NPU256 → NPU256 → NPU128 → CPU128` 顺序配对，数字为外层 ubatch。每个进程每 case 预热一次、正式三次，每种模式合计六个正式样本。输入为 bench 固定种子的合成 token IDs。prefill 不跑 lm_head，decode 包含输出头；不计模型加载、prepare 或采样。

## GDN 参数搜索

固定 ordinary TF32，搜索 chunk16/32/64/128 × CPU1/2/3/4 × NPU1/2/3 核（mask1/3/7），每个窗口 48 组，共 288 组。数据为 C7 维度的合成 projected QKV、Conv 权重、gate 和非零初始 state，预建 24 组输入/state；每组策略预热一次、正式六次，轮换 layer0/7/15/23。计时包含完整 native 调用和状态导入导出，不含工作区创建和外部 output gated RMSNorm。

| 实际 M | 最快 chunk / CPU / NPU mask | 最快完整调用 | chunk64 / CPU4 / mask7 |
|---:|---|---:|---:|
| 13 | 16 / 4 / 7 | 2.442 ms | 3.272 ms |
| 16 | 16 / 4 / 7 | 2.265 ms | 3.344 ms |
| 32 | 32 / 4 / 7 | 2.929 ms | 3.787 ms |
| 64 | 64 / 4 / 7 | 4.362 ms | 4.362 ms |
| 128 | 64 / 4 / 7 | 7.750 ms | 7.750 ms |
| 256 | 64 / 4 / 7 | 14.589 ms | 14.589 ms |

长窗口使用的 chunk64 经过搜索仍最快。短窗口缩小块长减少填充和块内 CPU 工作。表中为算子参数搜索，真实 C7 的整体收益和精度单独测量。

## 调优后完整模型性能

本节比较保留输入有限值扫描时的 CPU128/NPU128/NPU256 配置；默认 NPU128 移除扫描后的 pp512 配对结果和 CPU 分项见下节。

量化 GEMM 调优缓存为 `.cache/qwen35-gdn-prefill`，包含 124 份策略。ubatch128 完成 92 次搜索；ubatch256 复用已有策略，补齐其余策略。大词表输出头在整模型常驻前单独搜索，以限制调优峰值内存。正式测试重新加载模型，CPU128/NPU128 各 92 次缓存命中，NPU256 为 120 次；全部 case 均为 **0 miss、0 在线搜索、0 回退调用**。

以下每个数值为六个正式样本的中位数。CPU128 和 NPU128 的外层 ubatch 都为 128，NPU256 的外层 ubatch 为 256；内部 GDN 块长按前述规则选择。

| 场景 | CPU128 | NPU128 | NPU256 |
|---|---:|---:|---:|
| pp12 | 186.369 ms | 187.042 ms | 186.535 ms |
| pp13 | 191.643 ms | 212.514 ms | 213.368 ms |
| pp128 | 929.911 ms | 831.630 ms | 839.717 ms |
| pp512 | 3741.726 ms | 3354.496 ms | 3338.284 ms |
| 连续 pp128 + tg16：prefill | 935.282 ms | 831.168 ms | 839.000 ms |
| 连续 pp128 + tg16：16 次 decode 总计 | 1656.574 ms | 1657.897 ms | 1660.369 ms |

pp512 吞吐 **136.835 → 152.631 token/s（NPU128，+11.5%）**，耗时减少 10.3%。NPU256 为 **153.372 token/s**，比 NPU128 仅快 0.5%，量化 workspace 从 730.3 MiB 增至 1454.3 MiB；默认外层 ubatch 继续为 128，用户可通过现有 `--ubatch-size 256` 测更大窗口。

pp12 的 GDN 均走 CPU，时间差为测量波动。pp13 虽已改用 chunk16，NPU128 仍比 CPU128 慢 10.9%；`M>12` 是指定路由条件，不能作为实测最佳性能分界线。

### pp512 阶段分解

| 阶段 | CPU128 | NPU128 | NPU256 |
|---|---:|---:|---:|
| Linear：FFN、输入/输出投影等完整调用 | 1884.425 ms | 1856.650 ms | 1888.604 ms |
| GDN mixer：Conv、gate、recurrence、状态处理和输出 gated RMSNorm | 1331.827 ms | 972.097 ms | 920.207 ms |
| Full Attention | 151.525 ms | 152.058 ms | 149.178 ms |
| 其余 CPU 操作 | 373.268 ms | 372.550 ms | 379.756 ms |
| 完整 prefill | 3741.726 ms | 3354.496 ms | 3338.284 ms |

各阶段分别取中位数，与总中位数有少量差异。NPU128 的 GDN mixer 减少 359.730 ms（27.0%）；当前耗时占比为 Linear 55.3%、GDN 29.0%、Full Attention 4.5%、其余 CPU 11.1%。

NPU256 将每层的外部 GDN 调用由四次减为两次，总调用数 96 → 48；每层内部仍有八个 chunk64，NPU hardware 总时间基本相同。它主要节省状态导入导出等边界开销，但线性层完整调用变慢，整体收益有限。

### 移除全量扫描后的 pp512

对同一 NPU128 配置进行 `含扫描 → 移除扫描 → 移除扫描 → 含扫描` 四进程配对，每个进程预热一次、正式三次。固定频率、模型、输入和调优缓存均与前述测试一致，所有正式运行均为 0 miss、0 在线搜索、0 回退。移除扫描版同时记录 CPU 操作的分项时间，计算公式和精度保持不变。下表各项分别取六样本中位数。

| 阶段 | 含扫描 | 移除扫描 |
|---|---:|---:|
| Linear 完整调用 | 1864.944 ms | 1897.523 ms |
| GDN mixer | 972.667 ms | 844.550 ms |
| Full Attention | 151.204 ms | 152.719 ms |
| 其余 CPU | 373.205 ms | 373.564 ms |
| 完整 pp512 | 3362.428 ms | 3268.886 ms |
| pp512 吞吐 | 152.271 token/s | 156.628 token/s |

GDN 减少 128.118 ms（13.2%），完整 pp512 中位数减少 93.542 ms（2.8%）。含扫描版两个进程的 pp512 中位数分别为 3326.483 / 3396.328 ms，移除版为 3270.192 / 3267.555 ms；线性层的进程间差异较大，整体收益的稳定性弱于 GDN 分项收益。

固定 128-token prompt＋16-token continuation 的 17×248320 共 4,221,440 个 logits，与移除扫描前的 NPU 结果逐 bit 一致。

### 边界数据流优化前的 CPU 操作

以下为移除扫描版的六样本中位数，累计 24 层、每层四个外层窗口。各分项独立取中位数，合计与 GDN 总中位数有少量差异。

| 操作 | 作用 | 耗时 |
|---|---|---:|
| 投影拆分复制 | 将合并投影按行拆成连续 QKV/Z/a/b；累计复制 579 MiB 数据 | 70.594 ms |
| 输出 gated RMSNorm | 每个 value head 的 128 维 FP32 归约、缩放并乘 `SiLU(Z)`，累计 50,331,648 个元素 | 54.259 ms |
| gate 生成 | `log_decay=-exp(a_log)*softplus(a+dt_bias)`、`beta=sigmoid(b)`，各 393,216 个值 | 18.828 ms |
| 输出复制 | 将归一化结果写入输出投影输入，累计 192 MiB 数据 | 15.853 ms |
| 层 prepare | 形状/状态检查和卷积历史复制，历史数据累计 9 MiB | 1.474 ms |
| 缓冲区 resize | gate、core、output 复用已有容量 | 0.025 ms |
| 未分项余量（差值） | 完整 GDN 减去以上各项及 native 完整调用 | 4.939 ms |
| native 完整调用 | 含内部 CPU 数值处理、状态布局、同步及 NPU 运算 | 678.604 ms |
| GDN 总计 | 完整 mixer | 844.550 ms |

调用前后的 CPU 合计为 165.954 ms（逐样本 `gdn_ms-gdn_npu_ms` 再取中位数）。这些时间来自明确的张量处理；未分项余量仍为差值，不能视作单独测得的 FFI 成本。余量范围内还有 workspace 查找、状态位置/统计更新、trace 分支，以及 native `total_us` 计时区间外的 `getrusage`。

该基线的投影拆分为连续切片 ABI 准备输入；末尾复制来自 mixer 自有输出缓冲区和 decoder 的 `attended` 缓冲区分离。带步长输入、调用方输出和融合 gated RMSNorm 的实现与测量见 [边界数据流优化](QWEN35_GDN_BOUNDARY_DATAFLOW.md)。

### native 内部分解

同一批移除扫描后的六样本，分别取中位数。

| native 内部部分 | 耗时 |
|---|---:|
| CPU Conv + SiLU + Q/K L2 producer | 171.330 ms |
| CPU 衰减/系数构造 | 50.340 ms |
| CPU 三角求解 | 85.801 ms |
| CPU 输出收集 | 41.497 ms |
| 状态布局、同步及其余开销（差值） | 152.193 ms |
| NPU hardware | 178.437 ms |
| native 完整调用 | 678.604 ms |

native 余量由每个样本的完整调用减去四个 CPU 分项和 hardware 后取中位数。源码中的状态导入/导出在普通布局和 native 布局间转换，pp512 各方向累计 192 MiB payload；各计算阶段之间还有显式 buffer sync。当前计时尚未进一步区分状态布局转换和同步各自的占比。

NPU hardware 占完整 GDN 阶段约 21.1%；主要剩余成本仍是 CPU 数值处理、布局和同步。chunk64 已是本轮长窗口搜索最快组合。

## 调优后数值对照

默认 ubatch128，CPU FP32 GDN 对比 NPU ordinary TF32 GDN；其余量化算子均加载同一批调优策略。eval/PPL 子进程全部成功，全部 logits 有限，精度测试同样为 0 cache miss、0 在线搜索、0 回退调用。

| 指标 | 结果 |
|---|---:|
| logits 相对 L2 | 7.9954% |
| 去除每行均值后的 logits 相对 L2 | 11.1225% |
| logits 最大绝对差 | 2.260162 |
| top-1 一致 | 16 / 17 行 |
| CPU GDN PPL | 9.262681 |
| NPU GDN PPL | 9.185992 |
| PPL 相对变化 | -0.8279% |
| 评分目标数 | 510 |

唯一 top-1 变化位于第六行：CPU token6584、NPU token11528，CPU 两候选 margin 为 0.010315，NPU margin 为 0.001387。该组 510 个目标上的 PPL 没有恶化，不能据此认定完整数据集精度提升。ubatch256 此轮只测性能，数值表对应默认 ubatch128。

普通 TF32 与 CPU FP32 recurrence 的乘法精度、块算法不同，量化模型还会传播中间变化；本轮没有逐层消融来确认 logits 偏移的具体主因。使用的模式是 `RK_NPU_GDN_TF32`。库中的高精度 `RK_NPU_GDN_COMPENSATED_TF32` 会做高/高、高/残差、残差/高三个乘积并以 FP32 求和；两种模式的 canonical state 都存 FP32。

## 复现与检查

完整参数搜索与对照驱动保存在 [scripts/archive/gdn_prefill_20261018](../scripts/archive/gdn_prefill_20261018/)：

- `sweep.rs` / `sweep.py`：通过生成的公共 C ABI bindings 编译独立 Rust 探针；默认搜索 M13/128/256，`small 16 32 64` 搜索短窗口。
- `head_tune.rs` / `head_tune.py`：整模型常驻前单独准备 C7 大词表头 M256 策略。
- `performance.py`：六进程配对，断言所有 case 缓存命中且没有在线搜索或回退。
- `accuracy.py`：默认 ubatch128 的固定 token logits 与 510-target PPL 对照。
- `cpu_processing.py`：默认 ubatch128 移除全量扫描前后的四进程配对、CPU 分项计时及固定 token logits 逐 bit 对照；需冻结移除前二进制到 `logs/gdn-cpu-breakdown/before`。

这些是本次 C7 实验驱动，运行目录为项目根目录；CPU 基线使用 `logs/gdn-prefill-cpu-baseline/` 中冻结的 `6143940` 二进制。模型、原始 token 输入、生成的 bindings、静态库和板端环境需与本报告一致。先在板上以 `CARGO_TARGET_DIR=target/float-autotune cargo build --release --locked` 构建。

量化策略准备顺序为：ubatch128 的 `tune --batches 1,12,13,128`，独立 `head_tune.py`，再进行 ubatch256 的 `tune --batches 1,128,256`；共用 `--tune-cache .cache/qwen35-gdn-prefill`。所有搜索和测试由 `scripts/run_board.py --wait-lock` 持独占板锁、固定频率，子进程运行在 CPU4..7。

自动检查中 70 项测试通过、11 项硬件测试默认忽略；调优后的实现重新通过 51 项库测试、fmt 和 Clippy 全目标 `-D warnings`，并显式通过含 M13/32/64/128 的状态接续板测。所有最终频率报告均 `restore_errors=[]`，governor 与频率上下限恢复。

移除扫描及新增 CPU 分项计时后，重新通过全部 70 项常规测试、上述 NPU 状态接续板测、fmt 和 Clippy 全目标 `-D warnings`。pp512 四进程配对与 logits 对照的 runner 返回 0，固定频率设置完整恢复。

原始 JSON、搜索样本、调优缓存、基线版本与频率记录保存在忽略目录 `llm_generated_docs/evidence/qwen35_npu_gdn_prefill_20261018/`。性能与精度汇总分别为 `performance.json`、`accuracy.json`。精度日志的 wall time 不作为性能结论。

移除扫描前后的 pp512 原始样本、分项、输出一致性和频率记录位于该目录的 `cpu-processing/`，汇总为 `performance.json`、`accuracy.json`、`checks.json`。
