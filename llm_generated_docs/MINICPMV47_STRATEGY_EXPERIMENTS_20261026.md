# MiniCPM-V 4.7 流式 decode 优化分层实验

2026-10-26。实测确认了阶段内存回收和大窗口目标验证的潜力；没有测出“增加 48 槽、简单频率替换、共享专家提前计算”的稳定端到端加速。保留内存生命周期、domain 地址预留和 trace 改进，缓存与调度原型留在 evidence。

## 条件与比较口径

- H88K 8 GiB，GM7 本地 SSD；PCIe 8.0 GT/s ×4。转接链路的既有问题未在本轮调整。
- MiniCPM-V-4.7-35B-A3B noSQ，routed gate/up/down 全 W4A8，shared FFN/lm_head W8A8，CPU BF16 router，FP16 DSpark。
- CPU4–7 2.4GHz、NPU mask7/1GHz、DDR2112MHz；默认 allocator，`--no-tune`，两个 reader、一个确认需求后 importer、LES 开启。
- 正常生成 pp512→129 个输出，其中首 token 属于 prefill，decode 吞吐分母为后续 128 个 token。ubatch128，逻辑 prefill512，context1024，贪心、seed42。
- 原始代码为 runtime `9c75e06`、native `0f588e4`。正式 cache/scheduling 对照统一采用 3 个 target weight domains；实验分组用各自交错的对照，不能混用不同轮次的平均值。
- 正常测速与诊断 trace 分开；容量及调度对照各两次，置信度筛选各一次。所有生成输出与 AR 参考一致；同为 DS3 的 cache/scheduling 对照接受前缀一致，不同窗口的接受统计分别报告。

## 一、按阶段释放 workspace

生成预热现在只给输出头准备 decode/verify 所需行数；128 行评分通过 `token_nll()` 按需创建大 head workspace。进入 decode 时释放 prefill 专用 Linear workspace，保留权重、tune strategy、KV 和 GDN 状态。再次 prefill/评分可以重新分配。

独立目标模型实验在同一进程、40 槽下测得：

| 阶段 | workspace 计数 MiB | 进程 PSS MiB |
|---|---:|---:|
| prefill 后 | 738.36 | 5167.82 |
| 释放 prefill workspace 后 | 26.48 | 4838.33 |
| 128 行评分后 | 982.08 | 5632.66 |
| 返回 decode 后 | 26.48 | 5183.09 |

直接释放约 **329.49MiB PSS**，一次切换耗时 164.81ms。128 行评分、再返回生成均通过。workspace 计数包含未驻留的 host 数组，不能把计数减少量直接称为物理内存回收；评分后的大 logits 容量和 allocator 保留页也未全部释放。

初始两 domain 对照为 7.65 / 8.17→7.97 / 7.68 tok/s，未确认速度收益。保留此改动的依据是实际内存回收及减少内存压力。

扩大缓存还暴露了 IOVA 预算问题：48 槽在第 19 层分配失败，40 槽也复现过失败；内核为 `failed to allocate IOVA: -12`。已把 streaming domain 的非权重地址预留从 128MiB 增至 512MiB，为 staging、prefill workspace 及分配对齐留空间。本轮 40/48 槽的统一 3-domain 对照均成功；这不是按可用物理内存自动规划地址空间。

## 二、容量、替换和缓存接纳

### 正常端到端样本

| 实验 | 对照 tok/s | 实验 tok/s | 必要冷读变化 | 结论 |
|---|---|---|---:|---|
| 40→48 槽 | 8.00 / 7.75 | 7.76 / 8.06 | 161.32→142.96 MB/output，−11.38% | 未确认加速，48 槽换页更多 |
| 40 槽，历史频率每轮衰减 0.8 | 8.07 / 8.23 | 8.15 / 7.79 | 161.32→148.15，−8.16% | 未确认稳定加速 |
| 40 槽，共享专家提前计算 | 8.07 / 8.23 | 8.15 / 7.89 | 161.32→161.32 | 未确认稳定加速 |

48 槽额外增加 480MiB expert device payload。decode 采样窗口内，40 槽换入约 0.13 / 0.61MiB；48 槽约 20.89 / 2.80MiB。两档都未观察到该窗口内新增换出。这里是系统计数的阶段差值，不能用全进程 VmSwap 峰值代替 decode I/O。

### 命中与流量

以下采用三份独立诊断 trace。assignment 命中按 token/top-k assignment 计，unique 命中按每次层调用去重后的专家计；LES 覆盖与 RAM ready 都除以实际冷专家需求。

| 40 槽策略 | assignment 命中 | unique 命中 | 必要冷读 MB/output | 实际专家读取 MB/output | LES 冷需求覆盖 | 需求确认时 RAM 已 ready |
|---|---:|---:|---:|---:|---:|---:|
| 40 槽 LRU | 65.45% | 56.60% | 161.32 | 199.27 | 65.96% | 40.91% |
| 40 槽频率替换 | 69.29% | 60.14% | 148.15 | 187.46 | 62.86% | 40.44% |
| 40 槽共享专家提前 | 65.45% | 56.60% | 161.32 | 197.95 | 65.96% | 38.66% |

频率策略保存了更多热门专家，但错误预取没有相同比例下降：unused predictions 为 4565→4523，所以实际流量只从约 199.27 降到 187.46MB/output。LES 覆盖率的分母也变成了更难预测的剩余冷集，不能把该比例下降直接解读为预测器退化。

### 多输入离线回放

四段非重叠 WikiText 的频率策略分别减少 8.96%、12.22%、4.06%、6.56% 必要冷读。另补中文工程说明、Rust 代码、合成数学题解及英文 WikiText，每段 384 token prefix＋128 teacher-forced token、verify4、40 槽。每组各 5120 次层调用的 LRU cold 数均逐次复现，零差异；两组英文首段重合，共七段不同输入。

| 输入 | 当前 MB/input token | 频率 0.8 | 频率变化 | 频率＋临时执行/接纳变化 |
|---|---:|---:|---:|---:|
| 中文工程说明 | 123.13 | 107.79 | -12.46% | -14.62% |
| Rust 源代码 | 103.68 | 91.37 | -11.88% | -14.00% |
| 合成数学题解 | 197.78 | 183.88 | -7.03% | -8.15% |
| 英文 WikiText | 123.51 | 112.44 | -8.96% | -8.56% |

临时执行/接纳回放只使用过去和当前频率，从“当前 cache＋本批需求”中保留高分专家，其余需求必须用临时槽执行。相比单纯频率替换，多数只再省约 1–2 个百分点的冷流量，尚未计入临时执行和调度成本，未实现为 runtime。知道未来的 oracle 可省约 28%–45%，但这个差距不能当作现有在线策略能直接兑现的收益。

## 三、共享专家调度与完整调用

原型在实际需求与 LES 入队后，先计算共享 FFN，再执行 routed hot/cold；最终保持 routed 归约与 shared gate 合并顺序。未增加共享 FFN 调用，保留现有 ready 分组。

| 独立诊断 trace | 等记录 | 主线程导入 | 专家 complete-call | routed 合计 | shared FFN | 总 ms/output |
|---|---:|---:|---:|---:|---:|---:|
| spread_infer | 20.64 | 5.70 | 31.78 | 63.72 | 6.26 | 131.76 |
| frequency_v2 | 19.25 | 4.92 | 32.23 | 61.75 | 6.02 | 129.71 |
| shared_v2 | 18.69 | 4.46 | 30.24 | 58.90 | 9.02 | 129.89 |

共享提前让专家调用从 3908 降到 3772 次，routed 部分省约 4.83ms/output，但 shared 自身从 6.26 增至 9.02ms，抵消约 2.76ms。时序与后台导入/读取争用资源相符；本轮没有总线计数器，不能进一步归因为某一种 DDR 或 CPU 资源。诊断总时间只省约 1.88ms，正常测速尚未确认收益，因此未替换默认顺序。

接出了现有 native timings，控制 trace 中的完整调用内部为：

| native 部分 | ms/output |
|---|---:|
| 输入量化与 W4 打包 | 2.266 |
| expert/row 分组 | 0.279 |
| 命令构建 | 1.640 |
| 显式 buffer sync | 0.671 |
| gather 提交及等待 | 2.272 |
| gate/up 提交及等待 | 12.336 |
| CPU SwiGLU 与 A8 量化 | 2.434 |
| down 提交及等待 | 6.385 |
| 反量化与输出合并 | 1.953 |
| native complete-call 合计 | 30.260 |

这些是 native 软件阶段，其中 gate/down 包含驱动提交和等待，不是纯硬件 busy 时间。native 总计约 30.26ms，外层带 trace 的专家 span 约 31.78ms。CPU middle 只有约 2.43ms，单独挪走它的预算有限；若继续改执行器，应瞄准 gate/down 的调用边界、原始 token 输入复用及 sparse job 调度。

## 四、正确未来 token 的真实 verify 曲线

先由同版本单步 AR 产生参考，再用完全相同的 512-token prompt 和 128 个已知正确后续 token，实际运行 target verify＋GDN/KV commit。每档提前准备对应 workspace，统计包含目标输出头、专家加载与所有目标层。

| verify 行数 | target token/s | 必要冷读 MB/input token | 实际读 MB/input token | 相对 AR 全词表校验 |
|---|---:|---:|---:|---|
| 1 | 5.404 | 164.54 | 226.26 | 128/128；NRMSE=0 |
| 4 | 10.619 | 153.69 | 188.00 | 128/128；NRMSE=0 |
| 8 | 13.372 | 143.03 | 164.65 | 128/128；NRMSE=0 |
| 16 | 15.065 | 131.35 | 142.66 | 128/128；NRMSE=0 |

四档各 31,762,432 个 FP32 logits 全部逐位一致。**目标侧大窗口有实际潜力**：verify16 约 15.07 token/s，但冷读仍是每 token 131.35MB，不能把权重读取除以 16。

该表没有草稿网络、proposal 采样、拒绝后缀及 drafter context 更新，数值比对也在目标计时之外；它是 target 上限实验，不能称为 DSpark 端到端吞吐。现有 DS7 要验证 182 行才输出 128，DS3 是 139 行；直接加长窗口会把目标侧摊薄收益消耗在拒绝计算上。

### 现有置信度截断的端到端实验

使用现有 `--spec-draft-p-min`，其余配置保持 40 槽、LES、默认 allocator。各项目前一条完整生成样本，作为方向筛选；均与 AR 参考文本一致。

| 配置 | tok/s | 轮数 | verify 输入行 | 接受率 | confidence stops | 必要冷读 MB/output |
|---|---:|---:|---:|---:|---:|---:|
| DS7，p_min=0 | 7.497 | 24 | 182 | 65.82% | 0 | 177.14 |
| DS7，p_min=0.8 | 7.983 | 34 | 138 | 90.38% | 31 | 156.97 |
| DS7，p_min=0.95 | 6.848 | 50 | 130 | 97.50% | 49 | 157.78 |
| DS3，p_min=0 | 7.760 | 35 | 139 | 89.42% | 0 | 161.32 |

0.8 阈值把 DS7 的必要冷读从 177.14 降到 156.97MB/output、verify 从 182 降到 138 行；速度从 7.50 到 7.98 tok/s。它的实际工作量已接近 DS3 的 139 行，仅比本轮 DS3 多省一轮验证。0.95 虽得到 97.5% 接受率，却需 50 轮，速度降到 6.85；优化目标应是每单位时间的有效产出。0.8 相对 DS3 的约 2.9% 单样本差异尚不足以设成默认，需要跨输入复测。

草稿预算的独立拆分：网络约 7.01ms/output，共享大词表 head 约 5.37ms，Markov 投影约 4.89ms；driver 的全部 draft 阶段约 20.17ms/output，余量包括采样、分配及外层处理。两个词表投影合计比五层草稿网络还贵，不能只优化草稿 transformer。

## 下一步投入

优先围绕更便宜的草稿词表投影、DS7 的有效产出和窗口选择开展实验。目标侧 verify8/16 的真实曲线提供了依据，实际接受的 token 数及必要冷流量应和 draft 成本一起作为指标。

执行器方向可以独立推进原始 token＋稀疏 expert job 调度。当前分页将 token/top-k 展开为 assignment 行；shared 与 routed 的 intermediate 都是 512，底层已有 mixed grouped 实现，但直接给现有 top-1 assignment 路径加 shared 会重复执行共享专家。后续接口需要同时处理稀疏路由、共享 gate、槽位所有权和原 rank FP32 归约，并以完整调用验证。

现有 cache/prefetch 仍保留；本轮不增加 reader/import worker，也不以几乎相同的离线接纳规则扩展 runtime。从约 8 到 12 tok/s 需要省约 42ms/output，当前约 2ms 的局部改进达不到这个量级。

## 保留的代码与验证

保留：生成/评分分开准备输出头、进入 decode 回收 prefill workspace、streaming domain 地址预留、native/draft 阶段 profile、以借用事件流式导出 timeline。频率替换与共享提前原型保存在 evidence，并另存为 Git stash；48 槽未设为默认。C++ 算子库未改动，仍固定 `0f588e4`。2026-10-27 将内存回收与地址预留整理为 `fb22ac1`，trace 改进单独提交；未 push。

- ARM release 格式、all-target Clippy 和完整 Cargo 测试通过：112 passed、21 ignored。
- 最终保留代码在 20 槽固定 token 单行及 M4/M16 路径：3,722,160 个 FP32 logits 与原参考逐位一致。
- 两个调度/cache 原型各自同样通过 20 槽下 3,722,160 值比对；40 槽正确未来 verify1/4/8/16 共比较 127,049,728 值。
- 128 行评分后恢复 decode 已实际运行；全部完成的板端 clock report 恢复正常。
- 本轮是短上下文文本、固定模型和本地 SSD 的实验；没有重新做完整 PPL、长上下文或视觉质量评估。正常样本数有限，小于几个百分点的速度变化不作稳定收益结论。

证据目录：[strategy experiments](evidence/minicpmv47_strategy_experiments_20261026/)。汇总：[experiment_summary.json](evidence/minicpmv47_strategy_experiments_20261026/experiment_summary.json)；原始结果在 `results/`。离线脚本 `replay_corpus.py` 支持 `corpus` 与 `multitext`，正常测量入口为 `run_v2.sh`、容量为 `run_capacity.sh`、正确未来为 `run_replays.sh`、置信度为 `run_adaptive.sh`。初始建议见 [decode strategy](MINICPMV47_DECODE_STRATEGY_20261026.md)。
