# PR 5 采样优化

对照 PR [#5](https://github.com/happyme531/open_rknpu2_lm_infer/pull/5) 的 `9b4a56722be97c0e8f29c93bfd28a1244fc79663`，本地修改保留既有 seed 序列、候选顺序和 RNG 消耗规则。对照基线为 `9ad7a6d`。

采样改动通过194项自动测试及四组完整模型对照。真实词表的 top-p/min-p CPU 采样快2.75–3.44倍；本次固定 prompt 的完整 decode 耗时在普通 nucleus、MTP nucleus 和 MTP 概率草稿配置下降低8.4%、5.5%和13.1%。贪心 MTP 未见明确收益。

## 行为与实现

无截断抽样继续使用按处理后 logit 排序的分布。PR 的词表顺序累计抽样具有相同概率律，但相同 seed 的 token 序列不同；单候选时消耗 RNG 的行为也与既有实现不同。

当 `top_k=0` 且启用 top-p 或 min-p 时，先按原始处理后 logit 选择并排序候选前缀，候选量从64按4倍增长。指数权重不参与排序，避免不同 logits 经 FP32 指数舍入后成为相同权重、改变截断边界。未排序的全词表权重和只用于确定前缀是否足够；若累加误差可能影响 top-p 的保留集合，则使用完整排序。最终概率始终按已保留候选的原有顺序重新求和、归一化。

对于平坦分布，使用最大、最小 logit 的权重范围判断是否需要保留大量候选：当 min-p 保留全部 token，且 top-p 至少需要约四分之一词表时，直接完整排序，避免重复选择宽前缀。

贪心 `sample` 和 MTP 验证路径直接执行带历史惩罚的 argmax，保留最先出现的并列 token，无须构造单元素分布。`sample_via` 使用验证历史的处理器与调用方的 RNG；概率草稿继续使用目标分布、接受概率和 rejection residual。

`OPEN_RKNPU2_TUNE_SEARCH` 未加入。该开关会在普通推理中开放整个模型的通用搜索；MTP 宽浮点投影曾触发板端 timeout，且搜索后选择的 K 分块可能不在既有 resident weights 中。缓存命中数不能代替同输入的端到端正确性和耗时验证。

当前源码的静态检查也确认：[floating.rs](../src/npu/floating.rs) 在 `prepare_for_weights` 中先执行通用搜索，再检查 decoder 的 Ntile 上限和 resident weights 是否匹配；`load` 已在此前创建权重。因此现有检查不能约束搜索阶段的候选，也不能保证新搜到的配置实际用于推理。

## 验证入口

测试以原有完整排序实现为参考，逐项比较概率、连续 seed 抽样和历史更新；包含 top-p/min-p、重复惩罚、相同与相近 logits、单候选，以及使用独立历史的 MTP 抽样。

板端为备用板1 `备用板 1`，驱动0.9.6。CPU4–7 固定2.256GHz、NPU1GHz；无 DMC devfreq，DDR 未控制。共享锁与时钟恢复由 `scripts/run_board.py` 管理。

编译使用 Rust1.95.0、GCC12.2、all-features release，native 验证快照为 `c1f4bc5`；父仓库的 submodule 指针未修改。`cargo fmt --all --check`、all-target/all-feature release Clippy（`-D warnings`）及 release tests 均通过，共194项通过、39项硬件或模型依赖测试忽略。

真实词表检查使用此前捕获的 Qwen3.5 C7 W4 head logits（position8/9，vocab248320）。另有相同词表大小的均匀输入对照。生成对照使用原始 C7 W8 head，context512、ubatch128、4线程、CPU mask0xf0、NPU mask7、seed42、48个输出 token、禁用调优缓存（`--no-tune`），分别检查普通 nucleus、MTP greedy、MTP nucleus、MTP probabilistic。每组按基线、修复、修复、基线顺序运行。

原始检查结果、逐次耗时与命令统一保存于 [证据目录](evidence/pr5_20261103/)。采样改动已签名提交为 `b03d12c` 并进入远端 main；PR #5 已于2026-11-04按修订版替代关闭，原分支未合并。[关闭说明](https://github.com/happyme531/open_rknpu2_lm_infer/pull/5#issuecomment-6611268898)。

## 真实词表采样耗时

两个捕获的真实 logits 均逐项比较了完整保留概率，并核对32次带历史更新的连续 seed 抽样。之后每个实现按交错顺序计时4批，每批8次调用。表内范围对应两个输入；单位为 ms/次，仅为 CPU 采样耗时。

| 配置 | 基线 | 修复版 | 基线/修复 |
| --- | ---: | ---: | ---: |
| Greedy | 0.555–0.556 | 0.477–0.478 | 1.16–1.17 |
| 无截断 | 14.830–14.871 | 14.833–14.835 | 1.00 |
| top-p0.95，temperature0.7 | 13.625–13.648 | 4.142–4.946 | 2.75–3.29 |
| top-p0.5，temperature1 | 13.603–13.619 | 3.957–4.132 | 3.30–3.44 |
| min-p0.05，temperature1 | 13.597–13.638 | 3.958–4.133 | 3.30–3.44 |
| top-k20，top-p0.95 | 1.722–1.828 | 1.732–1.839 | 0.99 |

相同词表大小的均匀输入对照中，三种 top-p/min-p 配置为基线13.347–14.042ms、修复版13.470–14.156ms，约1%的额外耗时。概率与连续抽样序列也完全一致。无截断路径和既有 top-k 路径的性能基本保持原样。

## 完整生成对照

每个配置均为48个输出 token、seed42，按基线、修复、修复、基线顺序运行。表中为两次完整 decode 墙钟时间的均值，包含模型调用、采样和推测验证后的处理；不包含加载和首 token prefill。四组的生成文本、结束原因、token 数量、接受/提议计数、接受前缀直方图和验证行数均相同。

| 配置 | 基线 decode（秒） | 修复版（秒） | 耗时降低 | 接受长度 |
| --- | ---: | ---: | ---: | ---: |
| 普通 nucleus，top-k0/top-p0.95/temp0.7 | 5.785 | 5.300 | 8.4% | — |
| MTP，target greedy、draft greedy | 6.982 | 6.937 | 0.6% | 2.350 |
| MTP，target nucleus、draft greedy | 9.125 | 8.623 | 5.5% | 1.958 |
| MTP，target nucleus、draft probabilistic | 3.713 | 3.225 | 13.1% | 5.222 |

生成矩阵按已完成报告续跑：首段完成7个模型进程，剩余9个由新 runner 完成；续跑显式检查原命令。首段批处理在下一进程启动处退出1，未终止正在执行的 NPU 调用；续跑退出0。16个模型进程均成功，三份时钟报告均未超时、`restore_errors=[]`，恢复后的 min/max/governor 逐项匹配各段开始状态。

这些是单一短 prompt、每个实现两次测量的结果，DDR 未控制，不能推及所有输入或默认采样设置。MTP greedy 的0.6%变化小于组内波动；无截断和 top-k20 路径没有对应的采样加速。原始逐次结果见 [汇总](evidence/pr5_20261103/summary.json)、[生成报告](evidence/pr5_20261103/e2e/summary.json) 和 [采样结果](evidence/pr5_20261103/sampling.json)。
