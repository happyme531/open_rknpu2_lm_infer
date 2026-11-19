# Qwen3-4B 真实 prefill 首 token 与 GQA 分组提交实验

在 RK3588 主板上，沿用逐行真实最大值的 Attention 优化，使 Qwen3-4B 的 4096-token 首 token 延迟从 29.914 s 降到 27.888 s，512-token 从 3.251 s 降到 3.163 s。模型的线性层策略全部命中。进一步把 8 个 KV head 分成 3、3、2 提前处理，三层复测的 4K 平均收益只有 0.19%–0.60%，正反顺序的收益符号相反。适合保留前一项作为后续基线。[整模型数据](evidence/gqa_model_wave_20261118/model-full/summary.json) · [三层分组对照](evidence/gqa_model_wave_20261118/whole/summary.json)

## 整模型测量

原库取 native `3be24de`，开启双 slot 和二维 LUT。优化版在同一提交上使用上一轮实验补丁。它在 M128 causal 快路径上连续扫描真实可见的 score，求每行最大值，写出与原 DPU 曲线相同的指数到独立的 write-combine 缓冲，并按 KV head 并行合并输出。新提出的分组提交只进了单算子探针。整模型两侧使用同一份 Rust 可执行文件，仅替换 Attention 所在的动态库。[优化版补丁](evidence/gqa_stream_prefill_20261118/prefill.patch) · [运行时测量补丁](evidence/gqa_model_wave_20261118/runtime-probe.patch) · [二进制与输入散列](evidence/gqa_model_wave_20261118/binary-sha256.json)

| 输入 | 原库 → 优化版 TTFT | TTFT 降幅，两对交错结果 | 原库 → 优化版 Attention 累计 | Attention 降幅 | 模型 forward 吞吐 |
|---|---:|---:|---:|---:|---:|
| 512 | 3251.4 → 3163.1 ms | 2.72%，2.64% / 2.79% | 680.6 → 546.7 ms | 19.68% | 175.1 → 181.3 tok/s |
| 4096 | 29913.7 → 27887.8 ms | 6.77%，6.66% / 6.88% | 12804.9 → 10554.2 ms | 17.58% | 138.4 → 148.6 tok/s |

4K 的 Attention 累计节省 2.251 s，首 token 提前 2.026 s。同批线性层累计 15.170 → 15.402 s，这 0.232 s 的变化没有单独做因果归因。按 36 层、每个可见 query-key 对计算 `4 × Hq × D` FLOP，分母取模型 Attention 累计时间及项目三核 FP16 峰值 3 TFLOP/s，512 的有效利用率从 3.79% 提高到 4.72%，4K 从 12.88% 提高到 15.63%。这些比例包括算子与运行时开销，硬件 MAC 忙碌比例需要单独的计数器。[每进程、每阶段结果](evidence/gqa_model_wave_20261118/model-full/summary.json) · [项目利用率口径](ATTENTION_PREFILL_LIMITS_20261118.md)

模型为 `Qwen3-4B-W4A8-Mixed-LMHeadW8`，36 层，Hq/Hkv=32/8，D128，ubatch128，Attention KV tile512。Rust 取外层 `e133194`，构建使用 Rust 1.95、Clang 22、native Release -O2，两个库均关闭 tracing。主板 RKNPU 0.9.8，CPU4–7 为 2.4 GHz，三核 NPU 1 GHz，DDR 2.112 GHz。每次 ABBA 持有同一把板锁。以 WikiText 的同一串 token IDs 取前 512 和 4096 项。每个独立进程常驻模型，两种长度各预热一次，再各测两次。每种库每个长度有 4 个正式请求，先取进程内中位数，再平均两个进程。TTFT 从引擎开始处理请求算至第一段非空 token 文本，包含清空 KV、prepare、prefill、LM head 和采样，权重加载、输入分词及队列等待在计时外。全部请求的前缀复用数、decode 调用数和运行时搜索次数均为零。[请求代码](evidence/gqa_model_wave_20261118/model_measure.rs) · [交错运行与依赖检查](evidence/gqa_model_wave_20261118/measure-full.py) · [正式锁与时钟记录](evidence/gqa_model_wave_20261118/model-full-clocks.json)

最早一轮沿用了旧命名的策略文件，启动时命中 14 项、回退 78 项，测得 512/4K 的 TTFT 降幅为 4.11%/9.35%。随后从原驱动版本的已有策略记录恢复与当前 runtime 匹配的文件名，保留文件内容，并用本次 CLI 为缺项调优。上表的四个进程均命中 92 项，零 miss、零 fallback，旧缓存数据单列。[准备缓存](evidence/gqa_model_wave_20261118/prepare-cache.py) · [调优记录](evidence/gqa_model_wave_20261118/tune.json) · [旧缓存对照](evidence/gqa_model_wave_20261118/model/summary.json)

## 数值、分派与回退

每种长度导出首步全部 151,936 个 FP32 logits。两版的所有测量进程与各自的预热、重复执行均逐位一致，非有限值为零，top-1 token ID 分别为 553 和 330。两种策略缓存得到的首步 logits 散列也相同。此处验证的是这两个真实前缀，未测语料级 PPL。[逐位复核](evidence/gqa_model_wave_20261118/model-full/logits.json) · [不同缓存的散列](evidence/gqa_model_wave_20261118/logits-sha256.json)

另外用旧缓存、同一模型输入，在优化版的实际 `Graph::process_scores` 入口统计分派，计数运行的 logits 再次与原库相同。512 的 576 次处理全部经过 M128 快路径，覆盖 36 层。4K 的 4611 次处理中有 4487 次走快路径、2 次走其他 LUT 路径、122 次因 V 幅值保护切换，触发保护的有 2 个不同层缓存。按物理 score 元素加权，快路径覆盖 96.09%。这些比例只描述本输入的分派，计数运行与正式延迟测量独立。[计数代码](evidence/gqa_model_wave_20261118/coverage.cpp) · [计数与分母](evidence/gqa_model_wave_20261118/coverage/summary.json)

两个动态库都通过原有 causal 板测，覆盖预加载未来 KV、大小 V、短/长分派、截断、恢复和多核 mask。原 pipeline 测试在优化版要求注入第 10 次 submit，此几何实际只提交 8 次，原脚本在这一项报错。验证补丁在新 workspace 上先数实际调用，再逐个提交点注入故障，同时覆盖第 2、4 和末次同步及原有 BUSY 检查。原库 12 个提交点、优化版 8 个提交点全部通过，两版观测到的同步次数分别为 80、12。九份板测报告均核对了 CPU、DDR 和 NPU 的频率范围及 governor 恢复。[原脚本失败记录](evidence/gqa_model_wave_20261118/contracts/stream-pipeline.log) · [测试补丁](evidence/gqa_model_wave_20261118/pipeline-observed.patch) · [优化版逐项记录](evidence/gqa_model_wave_20261118/contracts/stream-observed.log) · [时钟恢复核对](evidence/gqa_model_wave_20261118/clock-check.json)

## 提前按 KV head 交付 QK

在第 17 层真实 post-RoPE Q/K 与 V 上，从空 KV 增长到目标长度。四种方法依次正序、逆序，各进程预热两次、正式六次，另跑一次 trace。QK 分组后轮流在三个核上分配 head，前两组各三个核，末组两个核。每个 head 都看到完整可见上下文，PV 等待全部 8 个 head 的概率就绪后提交。按 key 分段的版本先求各段最大值，再取真实行最大值，随后用原曲线写指数。若 CPU 接手时多个组已就绪，会合并剩余工作。[实现生成器](evidence/gqa_model_wave_20261118/prepare-wave.py) · [CPU 分段实现](evidence/gqa_model_wave_20261118/wave_keys.inc) · [原始样本与 trace 汇总](evidence/gqa_model_wave_20261118/wave/summary.json)

| 第 17 层筛选 | 512 增长 ms | 4K 增长 ms | 4K QK / PV 提交数 | CPU 处理和 QK 重叠 ms | CPU 处理在全部提交外 ms |
|---|---:|---:|---:|---:|---:|
| 上轮优化版，整组提交 | 9.379 | 269.080 | 128 / 128 | 30.19 | 74.76 |
| QK 拆三次，CPU 等全部完成 | 9.880 | 268.790 | 384 / 128 | 32.16 | 72.06 |
| 组就绪后按 head 并行 | 9.609 | 270.534 | 384 / 128 | 6.09 | 94.38 |
| 组就绪后按 key 分段 | 10.143 | 264.001 | 384 / 128 | 5.54 | 86.86 |

4K 的物理 task 数保持 5632，总 submit 从 256 增至 512。按 key 分段时，CPU 处理累计从 116.33 ms 降至 105.04 ms；在 QK 提交期间处理未满 8 个 head 的组只覆盖 4.03 ms。原来双 slot 已让上一块 CPU 处理与下一块 QK 重叠，缩短交付粒度后，两者总重叠从 30.19 ms 降到 5.54 ms。设备 QK、PV 执行时间始终约为 55、78 ms。上述重叠取 CPU slice 与驱动提交区间的交集，墙钟使用独立的无 trace 样本，不能把几列直接相加。三种分组实现都对齐原库的增长全输出，以及两个末尾 Q128 块合计 18,874,368 个 FP16 PV 输入。[trace 查询](evidence/gqa_model_wave_20261118/wave_detail.sql) · [PV 和增长输出逐位核对](evidence/gqa_model_wave_20261118/wave-validation.json)

## 单独改变 CPU 分工的对照

为区分处理次序与额外 QK 提交的作用，又在第 0、17、35 层各做独立正反序测试。表内为从空 KV 增长到 4K 的单层毫秒数。“整组按 key 分段”和“CPU 分 3/3/2 遍”都等完整 QK 到齐，NPU 侧沿用原来的整组 QK 提交。最后一列重测筛选中最快的提前交付版本。[对照实现](evidence/gqa_model_wave_20261118/prepare-whole.py) · [三个层的逐进程样本](evidence/gqa_model_wave_20261118/whole/summary.json)

| 层 | 整组基线 | 整组按 key 分段 | CPU 分 3/3/2 遍，各组按 key 分段 | QK 分组提前交付并按 key 分段 | 提前交付的正 / 反向降幅 |
|---|---:|---:|---:|---:|---:|
| 0 | 265.153 | 279.681 | 281.180 | 264.655 | +1.27% / −0.92% |
| 17 | 252.331 | 264.672 | 264.510 | 251.758 | +1.50% / −1.07% |
| 35 | 243.660 | 252.077 | 254.811 | 242.199 | +2.31% / −1.16% |

在这三次配对里，CPU 单独改成按 key 分段使 4K 慢 3.45%–5.48%。同时改 CPU 与 QK 的三个均值只快 0.19%、0.23%、0.60%，512 则慢 4.71%、9.31%、3.02%。同一第 17 层基线在初筛和复核两批分别为 269.080 和 252.331 ms，绝对值有批间漂移。性能判断取批内配对。所有三层、四种方法的 512/4K 全增长输出及末块输出均已与冻结的原库输出逐位比较。[无 trace 样本、分顺序收益](evidence/gqa_model_wave_20261118/whole/summary.json) · [板上比较与参考散列](evidence/gqa_model_wave_20261118/whole/validation.json)

## 下一步判断

这批数据支持把连续扫描、独立概率缓冲及并行收尾的版本作为 prefill 基线。3/3/2 分组增加提交数，完整模型没有必要为这点单层波动增加另一条调度路径。本轮只在隔离构建中测量。正式集成前还有一处静态代码边界值得处理。实验补丁的 CPU 表覆盖中心化后 −16 至 0 的 FP16 输入；NEON 快路径直接用幅值位模式索引，若输入或 QK 转成非有限值，索引可能超出该域。需要给异常 score 明确定义检查或回退，并验证；本次实测模型 logits 和有效 PV 输入均有限。[实验版查表源码](<HOST_HOME>/mnt/3090x4/tmp/gqa-model-wave-20261118/stream/attention.cpp)

若目标是再明显缩短 4K，下一项可先量 PV 的物理输出宽度成本。当前 128 个 V 通道再加一列分母，硬件布局补齐到 160。用同一真实概率块、相同 tile 与三核分工对比 N160 与 N128 的 PV 设备时间，然后把产生相同 tile 分母、原舍入顺序、输出合并、必要数据搬运的代价全部加回完整调用。以本轮约 78 ms 的 PV 做线性宽度估算，160→128 对应约 16 ms 的设备时间差；真实差额受带宽、分块和重叠影响，这个数只是决定实验成本的量级。即使按此数全部兑现，也解释不了成倍加速。当前数据支持排除这一种分组调度，仍没有测出 RK3588 的硬件极限。[PV 布局与既有约束](ATTENTION_PREFILL_LIMITS_20261118.md) · [频率与 NoC 实测](ATTENTION_STREAMING_PREFILL_20261118.md)

[冻结源码、构建产物、完整 logits 和原始回放输出](<HOST_HOME>/mnt/3090x4/tmp/gqa-model-wave-20261118) · [本轮可复查证据](evidence/gqa_model_wave_20261118)
