# Attention 保守优化：基于 c96930f 的正式整理

本轮只整理分块、图复用和 CPU 处理路径，不接入重算 QK、扩维 K、设备 ReduceMax 或新的 LUT 配方。实现位于独立算子库工作树 `open_rknpu2_lm_infer-attention-nomask/native/rk_npu_matmul`，分支 `perf/attention-nomask`。

最终对照：Qwen3 的512/2048-token完整 head-free prefill 延迟下降3.52%/7.18%；C7三档完整ViT下降12.48%–16.11%。所测两模型 logits 和全部ViT features 均与基线逐位一致。

下文性能表来自提交前数值诊断清理之前的快照；清理后的回归单独记录于文末，不把删除诊断检查的独立收益混入这些数字。

## Rebase 与改动边界

已将该分支从 `a9a2e068a8f8767c26b7009f6a6bb703fbd5fb7e` rebase 到 `c96930f4dac9d9a978712df0fbdce918f5f02cee`。原实验仅有两份未提交改动，使用原基点显式 `rebase --onto`，随后无冲突恢复。浅克隆中的历史截断没有导致重放旧库提交。

保留了 `backup/attention-nomask-before-c96930f-20261023` 和 stash `b9d2149bcfda1320e4fd8b58c204fce07b304fa5`；原始补丁与 rebase 元数据在 [evidence/attention_cleanup_20261023](evidence/attention_cleanup_20261023/)。主工作树中的 MoE 改动未参与本轮源码修改。

## 正式实现

- stable、NO_MASK、MHA 优先使用 128 行 query；几何和 QK/PV/LUT 合计任务数不允许时降到 64/32。复用现有 `lut_tasks_fit`，预留 split 提交模式的每核 setup。GQA、masked、fixed-shift 保留原先 32 行起始分块策略。
- causal LUT 的内部 query segments 复用 PV-tile 对齐的 span，但严格限制在**当前公共请求的可见前缀**内。不会因预加载未来 KV 而超过 `query()` 的内存估算，也不会越过任务数边界切换成 CPU exp。CPU-exp 的短/长序列与 V 幅值分派继续按可见前缀决定。
- centering 的完整八元素组直接执行 FP16 subtract/clamp，边界与 Boolean mask 标量路径不变。LUT 表、carrier 修正、PV、最终 FP32 累加和归一化不变。
- 多行且物理 M>16 时按 KV head 并行 pack Q；单 token 不启用并行 pack。causal V 前缀修正使用硬件 half→float 转换，保持原有逐 token 的 FP32 累加顺序。

公共 C ABI 没有变化。MHA 的 peak scratch 会随较大 query tile 增长，调用方继续使用 `query()` 返回的大小；没有新增一份 KV cache。

## 正确性回归

主机 Attention/GQA sizing、causal 参数和 MLA 三项 CTest，以及公共 C 头文件构建通过。板端另覆盖：

- 全部有限 FP16 score 与 13 种代表性 row max 的 centering 舍入对照，包含子正规数、正负零、clamp 和有限极值。
- MHA mask1/2/4/3/7，长度129/513/1024/8192，尾行、任务预算缩块、大 V CPU fallback、FP16/FP32 输出 guard；与 32 行调用逐位一致，随后 INT8 canary 正确。
- pingpong/split 两种提交模式的 causal batch/AR 等价检查，包括高未来 V、CPU exp 边界、descriptor 边界、truncate/abort、bulk load/append/checkpoint。
- 预加载512个KV、公共请求只使用前128/256/384个时，不越过 sizing 边界，结果与逐 token 执行一致。
- 通用 GQA 几何、三种 mask、cache 重绑定、reserve/grow、Attention/INT8 混跑及 stable MLA/expanded MLA 回归。

初次完整 mask 测试发现旧参考计算漏传实际 KV head 数；其默认值8不适用于测试中的1/3/32 heads。已修正参考参数并重跑，不放宽精度阈值。初次日志保留在 `attempt1/`。

真实模型使用各自 WikiText 固定 token 序列，1537-token prompt 加7个 continuation，逐字节比较全部8个位置的 logits：Qwen3 为1,215,488个FP32值，C7为1,986,560个FP32值，均与基线完全一致。两边 cache miss、在线调优及策略 fallback 均为0。本轮没有重新跑PPL，也不将这两个固定序列的检查外推为所有输入保证。

## 单算子完整热态调用

基线为未经修改的 `c96930f`，实验仅包含本轮改动。备用板1，driver0.9.6，CPU4–7/四线程2.256GHz，NPU三核1GHz，DDR未受控。每个几何三轮ABBA、每进程预热4次/正式12次；每种方法72个正式样本。表中为各进程 median 的均值。包括公共 run 内的分段、构图更新、pack、计算、同步与finish；不含首次KV load。Q128/KV4096不是完整4096-token prefill。

| 几何 | Q/KV | 基线 ms | 当前 ms | 延迟下降 |
|---|---|---:|---:|---:|
| Qwen3 GQA | 128/512 | 5.925 | 5.500 | 7.17% |
| Qwen3 GQA | 128/4096 | 28.933 | 22.695 | 21.56% |
| Qwen3 decode | 1/4096 | 0.866 | 0.863 | 0.33% |
| C7 GQA | 128/512 | 4.591 | 3.931 | 14.38% |
| C7 GQA | 128/4096 | 18.764 | 16.534 | 11.88% |
| C7 verify | 4/4096 | 1.356 | 1.270 | 6.35% |
| MHA H16/D64 | 128/256 | 1.546 | 0.915 | 40.81% |
| MHA H16/D64 | 128/1024 | 3.773 | 2.208 | 41.49% |
| 真实 C7 捕获 | 4/1025 | 0.556 | 0.489 | 12.01% |

所有输出与基线逐字节一致。decode 视为持平，不把0.33%解读为稳定提速。Qwen3长KV三轮下降17.18%–24.57%，保留该波动；不能将这次与旧批次的百分比差异归因于某一项整理。

## 完整模型与 ViT 性能

Qwen3 使用已有 WikiText 前2048个 token 的固定前缀，ctx2048、ubatch128。四个独立进程按ABBA运行，每进程每个长度预热一次、正式两次；每方法4个样本，取算术均值。计时包含所有 decoder 层和KV追加，不含模型加载、reset/prepare、tokenization、stats采样与LM head。四个进程均为86 cache hits，cache miss、在线调优和策略 fallback 为0。

| Qwen3 head-free prefill | 基线 ms | 当前 ms | 延迟下降 |
|---|---:|---:|---:|
| 512 tokens | 2969.629 | 2865.184 | 3.52% |
| 2048 tokens | 14458.543 | 13420.719 | 7.18% |

ViT 复用已有同板 FP16 GEMM 策略缓存，使用相同C7 checkpoint和固定图片；每档ABBA四进程、每进程预热一次和3次正式调用。清空图片特征缓存后测完整 encoder，包含全部CPU/NPU阶段，排除图片预处理、模型加载和文本推理。表中为每方法6次调用的中位数，进程内重复不当作独立板卡实验。三档全部features逐位一致，对既有FP32 gold的误差也相同。

| C7 ViT 输入 | 基线 ms | 当前 ms | 延迟下降 |
|---|---:|---:|---:|
| 256×256 | 317.463 | 277.221 | 12.68% |
| 384×256 | 500.185 | 437.745 | 12.48% |
| 512×512 | 1593.734 | 1336.909 | 16.11% |

DDR未受控，保留了全部样本和进程配对范围。单算子热态调用、完整文本prefill、完整ViT分别报告，收益不相加；本轮数字也不直接与旧基底的批次相减归因。

最终24份板端时钟报告均 `exit_code=0, restore_errors=[]`。末次原库canary的重复输出、guard和独立FP32参考行检查通过。板端RTC偏旧，文件日期按主机命名；计时使用单调时钟。

## 复现与状态

板端快照：`<BOARD_HOME>/mnt/3090x4/tmp/attention-cleanup-20261023`；主机映射：`<HOST_HOME>/mnt/3090x4/tmp/attention-cleanup-20261023`。`base/` 来自精确 Git archive，`candidate/` 叠加本分支改动。文本/ViT 测试使用主工作树 `b6ffd49` 的同一 Rust 源码快照，只有固定 token benchmark 输入适配；两套程序分别链接基线和实验库。

测试及比较工具在算子库 `tests/tools/attention_cleanup/`，实验数据与二进制不属于正式源码提交范围。源码、测试和接口文档已签名提交为 `61762012b0a145e2286169526f1abf142695cb11`，父提交为 `c96930f`；没有合并到主分支或发布。

归档包含基线提交、正式源码补丁、编译/测试日志、原始计时JSON、运行脚本，以及两套库/程序和全部输出张量的SHA256。较大的输出张量和二进制留在上述共享快照，不重复放入报告目录。

## 提交前热路径清理

按用户要求，删除 stable LUT centering 和 CPU softmax 的逐行 `isfinite` 检查、两个原子错误标志及三处最终汇总检查；同时去掉长序列分支内恒真的 `span>1024` 判断和扫描 lambda 的重复参数。没有增加诊断开关或备用实现。

Q/K/V 有限、缩放后的 QK 不溢出 FP16 是调用前提，已写入算子接口文档。行最大值归约和 `-INFINITY` 哨兵用于 softmax、padding 与全遮罩语义，仍保留；V 幅值统计控制 LUT/CPU exp 的数值分派，也不能当作调试扫描删除。公共参数、buffer/domain、任务预算、BUSY 和提交失败状态管理均保留。

清理前后快照分别为共享目录 `attention-cleanup-20261023/candidate` 与 `attention-hotpath-cleanup-20261023/candidate`。后者的回归以清理前版本为直接对照；源码补丁和哈希单独保存在 [evidence/attention_hotpath_cleanup_20261023](evidence/attention_hotpath_cleanup_20261023/)，不覆盖原始性能证据。

最终清理版重新通过主机三项 CTest 和公共 C 头文件构建，以及板端全部14项算子任务。另跑9组完整调用的三轮ABBA（每方法每组72个正式样本，共1296次），输出全部逐字节一致。平均延迟相对清理前为慢1.36%到快2.64%，轮间波动明显，DDR仍未受控；这次诊断清理按性能基本持平记录，不宣称独立提速。

Qwen3/C7 的同一固定序列再次分别比较1,215,488/1,986,560个FP32 logits，全部逐字节一致，哈希也与本报告前一轮相同。两边Qwen3为86 cache hits、C7为64 hits，miss/在线调优/策略fallback均为0。三档ViT features也全部逐字节一致，完整encoder延迟差异均小于1%；仍复用相同预计算缓存，vision工具没有导出cache miss计数。本轮未额外重跑完整文本prefill性能或PPL。

清理后全部20份时钟报告均 `exit_code=0, restore_errors=[]`。提交只含7份正式源码、测试和文档文件，GPG签名验证通过；实验工具、170份二进制/输出张量哈希所对应的产物及证据未进入该提交。
