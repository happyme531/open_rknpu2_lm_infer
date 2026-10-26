# MiniCPM-V 4.7 低权重路由跳过实验（2026-10-27）

## 实现

新增 `--expert-min-weight T`，默认 `0`，合法范围 `[0, 1]`。每个 token 先计算原有 top-8 路由及其归一化权重，再跳过权重严格小于 T 的分配；最高分专家始终保留。留下的系数不重新归一化，共享专家照常执行。

例如 top-4 权重 `[0.50, 0.25, 0.125, 0.125]`，阈值 `0.25` 得到 `[0.50, 0.25, 0, 0]`，不是 `[2/3, 1/3, 0, 0]`。这是改变模型计算的有损近似；默认关闭。

参考 [Strata `ds4_model.cu`](https://github.com/Niko1221/Strata/blob/4b9e05a2a806a3fe24f7b2560d98626c37dade35/src/dsv41/ds4_model.cu#L6508) 的 `STRATA_DS4_DISK_SKIP_W`，源码版本 `4b9e05a2a806a3fe24f7b2560d98626c37dade35`。Strata 该模式只在 decode/verify 跳过不在 RAM 的低权重专家；本实现扩展到 prefill/decode 的冷热专家，并增加至少保留最高分专家的约束。两者都不重新归一化。许可证和出处记在根目录 THIRD_PARTY_NOTICES.md 与 src/model/experts/Strata-LICENSE。

流式执行在以下位置过滤：

- 实际 demand 的去重、冷读和 staging 队列；
- LES 的预测集合，使用相同阈值，但预测错误仍可能产生浪费读取；
- 热/冷专家的 token 分配列表和 NPU gather，避免批量中某个专家仍被其他 token 使用时多算被跳过的行；
- prefill 尾部缓存填充和 decode LRU touch；
- 输出 reduction，跳过未写入的 contribution，防止复用 workspace 时读取旧值。

常驻 C++ MoE 在 GEMM 构建前忽略零权重分配，输出合并也跳过对应槽。没有 ABI、checkpoint 或专家存储格式变化。固定 top-8 workspace 及 40 槽缓存仍按原容量保留，因此不能把任务减少解释为缓存内存下降。

统计字段：`expert_min_weight`、`expert_pruning_renormalize=false`；每层 `pruning.all/decode` 包含 rows、原始 assignments、skipped_assignments、dropped_weight。`dropped_weight / rows` 是平均每 token 每层丢失的路由概率质量。原有 paged `assignments` 改为实际保留分配数，并增加 skipped_assignments；cache hit 分母和 LES hit 分母也只含保留分配。现有 `decode` 统计沿用 `M<=16` 的小批次口径，PPL 的跨阶段统计使用 all。

## 验证协议

H88K / RK3588 / 8 GiB，GM7 SSD，流式专家，40 槽/层（15.625%），LES 开启，两个 reader，O_DIRECT。CPU4–7 2.4 GHz、NPU 三核 1 GHz、DDR 2112 MHz，由 run_board.py 保存和恢复。目标模型 gate/up/down W4A8，共享专家及输出头 W8A8；ubatch 128、context 1024。

端到端使用同一 WikiText 512-token prompt，DSpark 3 草稿，greedy，运行 129 输出 token，报告内部 decode 的 128 token 计时。开启近似后生成轨迹和投机接受率可能变化，因此 E2E 速度不是固定路由下的纯算子节省。

质量使用固定 teacher-forced WikiText token 窗口，2×512 token，每窗前 384 token 为上下文、后 128 token 计 NLL，共 256 个评分目标。仅作为小样本筛查，不能代表完整语料质量。

原始脚本、trace、clock report、stdout/stderr 和结构化统计保存在忽略目录 `llm_generated_docs/evidence/minicpmv47_expert_skip_20261027/`；板端对应 `/mnt/nvme/minicpmv47/results/expert_skip_20261027/`。

## 实测结果

正常测速按 `0 → 0.05 → 0.10`、反向 `0.10 → 0.05 → 0` 各跑一次；中间完成三档 PPL。TPS 为两次算术平均，括号保留原始样本。不同阈值的生成轨迹不同，但每档两次路由统计相同，且三档均为 35 轮、139 行有效 verify、93/104 草稿接受（89.42%）。不把这一点解释为更多任务上的接受率结论。

| 阈值 | 平均执行专家/层/行 | 跳过分配 | 丢失概率质量 | 必要冷读 MB/输出 token | Decode tok/s（原始样本） | PPL（256 targets） |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 8.000 | 0% | 0% | 161.32 | 7.817（7.783 / 7.852） | 5.324853 |
| 0.05 | 7.880 | 1.50% | 0.44% | 158.22 | 8.082（8.206 / 7.959） | 5.508891 |
| 0.10 | 4.851 | 39.36% | 25.44% | 76.39 | 10.218（10.393 / 10.042） | 5.625887 |

必要冷读 = decode/small-M 阶段 cold_experts × 1,597,440 字节/record ÷ 128 输出 token，使用十进制 MB。它不含预测错误/取消读的浪费；下表另外列出完整 prefill+decode 的实际 reader 读取总量。跳过比例按 token-expert 分配计，不能直接等同于独立专家数或磁盘字节减少比例。

| 阈值 | Prefill 秒（两次） | 完整运行 reader 读取 GB（两次） | 其中丢弃读取 GB（两次） | Decode 热分配命中率 |
|---|---:|---:|---:|---:|
| 0 | 17.268 / 14.341 | 35.650 / 35.577 | 4.650 / 4.577 | 65.45% |
| 0.05 | 13.087 / 13.797 | 35.022 / 34.853 | 4.677 / 4.508 | 65.63% |
| 0.10 | 10.812 / 10.743 | 19.928 / 19.826 | 3.161 / 3.059 | 73.95% |

首次关闭状态的 prefill 较慢，进程出现过 swap；六次运行峰值进程 swap 约 30–224 MiB。因此保留原始 prefill 数据，不把均值当作无内存压力下的算法加速。缓存容量、workspace 预留和量化配置相同；这次没有调内存策略。所有正常测速都关闭 timeline。

阈值 0.10 的 decode 平均提升约 30.7%，必要冷读减少约 52.6%；PPL 增加约 5.65%。阈值 0.05 只跳过约 1.5% decode 分配，流量节省很小，约 3% TPS 差异仍有运行波动，PPL 已增加约 3.46%。

两档近似生成文本均与基线不同；阈值 0 的两次文本均与旧 AR 基线一致。这不构成语义质量评测，质量判断只使用固定输入 PPL；256 targets 仍然很小，不能直接认定实际任务损失只有表中的百分比。

## 验证和交付

- ARM `cargo fmt --all --check`、release 全目标 clippy（`-D warnings`）通过。
- 常规测试 116 passed、21 ignored；忽略项需要特定板端或模型。
- 原生 W4 MoE 的正常→部分零权重→全部零权重→恢复调用，输出与逐路独立计算结果逐元素相同；gather tasks 为 20→10→0→20，GEMM tasks 为 15→15→0→15。部分跳过仍涉及相同专家集合，因此 GEMM task 数没有下降，但 gather/实际行数减半。
- 原生混合 W4/W8 共享专家、槽交换/并发导入、W8 两种输入布局与 CPU oracle 检查通过；零路由不会跳过共享专家。
- 默认关闭的整模型全 logits 对照通过：单行序列及 M4/M16/M1 混合批次，均包含 M128 prompt；分别导出 2,233,296 / 1,488,864 个 FP32 logits，与 commit_20261026 固定参考逐字节相同。
- 开启 0.10 的诊断 trace 共 208,320 个原始分配，跳过 93,745 个，NPU 实际执行 114,575 个。逐层、逐专家的执行计数与保留分配计数完全相同，确认没有继续提交被删掉的分配。
- 诊断 trace 的首个 draft network 到末个 target verify 间，按 pread 开始时间计实际读 12.923 GB，即 100.96 MB/输出 token；必要冷读为 76.39 MB/token。多出的部分含错误/取消预取和边界在途读，不能把必要冷读直接当成实际 SSD 流量。预取标记读约 8.704 GB，这个数字不是预取命中字节。
- 诊断 trace 的 prefill/decode 冷需求分别为 4,375 / 6,121 条 record；该 trace 不纳入正常测速均值。
- 14 份 clock report（13 份 H88K、1 份 Firefly 常规测试）检查通过，均无 restore_errors；run/followup/verify/native 退出标记均为 0。

使用方式是在原命令上添加 `--expert-min-weight 0.1`，省略或设置 `0` 则关闭。实现默认关闭；算子库零权重调度固定至 `4ac2a395a180afa643d610af4024bc2b8edf52f2`。
