# RK3588 causal GQA prefill 的连续扫描、并行收尾与概率缓冲实验

保留每行真实最大值和当前 DPU 指数曲线后，Qwen3-4B 三个真实层的单层 4K prefill 从 297–300 ms 降到 232.5–233.5 ms，相对原库降低 21.6%–22.1%，相对上轮 CPU 查表版再降低 17.7%–18.1%。512 token 相对原库降低 33.8%–34.4%。新结果来自完整 KV 增长的公共 API 回放。三层、两种长度的全部 56,623,104 个输出与原库逐位一致。[上一轮实验](ATTENTION_CPU_LOOKUP_PREFILL_20261118.md) · [本轮数值核对](evidence/gqa_stream_prefill_20261118/validation.json)

## 真实输入和完整调用

A 为 native `3be24de`，启用双 slot 和 2D LUT。B 是上轮原位 CPU 查表。C 把全部 query 都可见的前缀与因果边界分开处理，保持原生 M128 顺序连续扫描，以四个大核并行完成各 KV head 的输出合并。QK 写入可缓存的 score，CPU 读取 score，把指数顺序写入独立的 write-combine 概率缓冲，PV 直接读取该缓冲。[以 A 为基准的 C 实验补丁](evidence/gqa_stream_prefill_20261118/prefill.patch)

| WikiText 层 | 512 token，A / B / C，ms | C 较 A 耗时下降 | 4096 token，A / B / C，ms | C 较 A 耗时下降 | C 较 B 耗时下降，4K |
|---|---:|---:|---:|---:|---:|
| 0 | 12.869 / 10.622 / 8.469 | 34.19% | 299.539 / 285.181 / 233.487 | 22.05% | 18.13% |
| 17 | 12.808 / 10.587 / 8.476 | 33.82% | 296.958 / 283.326 / 232.527 | 21.70% | 17.93% |
| 35 | 12.935 / 10.588 / 8.491 | 34.36% | 296.904 / 282.995 / 232.821 | 21.58% | 17.73% |

每层交错跑 A、B、C，再按反向顺序重复，各层轮换首个版本。每进程预热两次，正式六次，每种方法取两个进程中位数的平均值。六组正反顺序比较中，C 相对 A 的 4K 降幅为 21.48%–22.25%。在预装 KV4096 后只执行末尾 Q128，A 为 15.78–15.94 ms，C 为 12.66–12.70 ms。按项目 3 TFLOP/s 峰值，以及可见因果 QK+PV 共 137.47 GFLOP 折算，4K 完整 prefill 的有效利用率从 15.3%–15.4% 提高到 19.6%–19.7%。[第 0 层原始样本和 trace](evidence/gqa_stream_prefill_20261118/confirm-0/summary.json) · [第 17 层](evidence/gqa_stream_prefill_20261118/confirm-17/summary.json) · [第 35 层](evidence/gqa_stream_prefill_20261118/confirm-35/summary.json)

## 分开测量三个改动

第一批用同一份第 17 层数据，七个变体顺序各跑一次，再以相反顺序各跑一次。表内为完整 prefill 耗时，单位 ms。[筛选样本和 trace](evidence/gqa_stream_prefill_20261118/screen/summary.json)

| 实现 | 512 token | 4096 token |
|---|---:|---:|
| 原库 A | 12.93 | 299.65 |
| 上轮查表 B | 10.64 | 286.74 |
| B 加按 KV head 并行收尾 | 9.60 | 276.38 |
| B 加前缀专门循环，连续扫描 128 行 | 10.06 | 263.36 |
| 上述两项组合 | 8.74 | 253.35 |
| B 加前缀专门循环，每次只扫 16 行 | 10.31 | 339.54 |
| 16 行扫描与并行收尾组合 | 9.13 | 328.79 |

前缀长度由真实 query 位置决定。对这部分，CPU 两遍循环先计算真实行最大值，然后执行 FP16 减法、截断和逐位相同的指数查表。只在对角部分生成按 lane 的因果 mask。当前原生布局每个 key 八元组连续排列 128 行。缩至 16 行时，相邻小条之间要重新跨步遍历整段 KV；这一版单层慢约 76 ms，访问次序影响很大。收尾时每个线程使用私有的 partial 累加和 V 前缀 scratch，同一输出的 FP32 tile 累加次序保持一致。

第二批控制改动为概率缓冲的写法。第 17 层 4K 的独立可缓存缓冲为 242.31 ms，改用 write-combine 为 234.36 ms。这两个版本的两个进程中位数各自相差不到 1.3 ms。同批原位写的两个进程为 311.43 和 253.13 ms，第一批独立测得 252.91 和 253.80 ms。原位对照中有一个慢进程，不能据此把整段差额归功于分配缓冲。C 的总收益取上表三层的独立确认批次。[缓冲写法对照](evidence/gqa_stream_prefill_20261118/buffer/summary.json)

C 保留 QK 写完后的 FROM_DEVICE 同步。独立的概率缓冲使用驱动的 page-backed、IOMMU、write-combine 映射，CPU 工作线程在写完后执行存储完成屏障，再交给 PV。4K 下该缓冲每 slot 逻辑占 8 MiB，双 slot 相比 B 增加 16 MiB，原库 A 自身也保留独立 exponent 缓冲。专门循环限于这次的 causal、M128、每组 4 个 query head、每组 32 行几何。其他几何沿用上轮探针的通用路径。[驱动内存映射](../../lowlevel/refs/allbilly__rknpu_driver/rknpu_gem.c) · [缓冲实验源码生成器](evidence/gqa_stream_prefill_20261118/buffer.py)

## 4K 剩余时间和流量

第 17 层独立诊断 trace，单位 ms。提交外时间从对应 CPU 区间中扣除与所有 NPU 提交的交集，设备段使用驱动返回的执行时间。

| 4K 增长 | 原库 A | 上轮 B | 本轮 C |
|---|---:|---:|---:|
| QK / LUT / PV 设备区间 | 54.3 / 67.1 / 78.0 | 54.1 / 0 / 78.8 | 55.9 / 0 / 79.6 |
| CPU 减 max 阶段累计，B、C 含指数查表 | 110.2 | 149.5 | 128.2 |
| 其中暴露在 NPU 提交外 | 34.4 | 85.8 | 63.5 |
| CPU 输出收尾累计 / 提交外 | 53.0 / 16.5 | 58.4 / 16.7 | 31.7 / 5.0 |
| 概率输入的 TO_DEVICE 阶段 | 28.0 | 18.3 | 0.17 |

C 的 TO_DEVICE trace 区间只保留记录开销；CPU 写合并缓冲的存储屏障计在处理阶段内。QK 和 PV 没有因这次修改加速，收益出在 CPU 扫描、输出收尾和交给 PV 前的缓存维护。各阶段可以重叠，以未开 trace 的完整墙钟判断最终收益。[确认批次分段数据](evidence/gqa_stream_prefill_20261118/confirm-17/summary.json)

真实第 17 层另跑固定时钟的 NoC payload 计数。每个窗口只包一块 128 token 的 KV append 和 Attention 全调用，累加 32 个窗口得到 4K。每个版本两进程，每进程三个计数样本。表中以十进制 GB 表示总字节。DSU 和系统计数包括窗口内的其他活动，NPU 计数按该设备的四个数据端口过滤。[逐次计数与寄存器恢复检查](evidence/gqa_stream_prefill_20261118/traffic/summary.json) · [项目 NoC 计数器说明](../../lowlevel/ddr_bw/README.md)

| 实现 | NPU 端口 GB | CPU/DSU GB | 全系统 GB |
|---|---:|---:|---:|
| 原库 A | 3.697 | 1.870 | 5.716 |
| 上轮 B | 2.583 | 1.953 | 4.679 |
| 并行收尾、连续扫描、独立可缓存概率缓冲 | 2.583 | 1.846 | 4.550 |
| 本轮 C，独立 write-combine 概率缓冲 | 2.583 | 1.761 | 4.460 |

A→B 已去掉约 1.11 GB 的 NPU 端口流量。B→C 的 NPU 端口字节几乎相同，CPU/DSU 字节降低约 9.9%。同轮 768 个计数窗口的最大单端口读数为 83.23 MB，八个计数进程均读回确认寄存器恢复。正式计时独立运行。

对 C 每次只降低一个时钟，以正向和反向两轮测量。显著的 DDR 和 CPU 频率敏感性同时存在。降 DDR 后，QK 从 55.5 增至 68.4 ms、PV 从 79.7 增至 92.3 ms、CPU 处理从 127.9 增至 149.8 ms；降 CPU 后，QK/PV 基本持平，CPU 处理增至 157.1 ms。上述数据定位了对频率敏感的阶段，不能由此推出 DDR 已饱和。[降频对照、原始样本、时钟和 trace](evidence/gqa_stream_prefill_20261118/frequency/summary.json)

| 第 17 层 C，单项变化 | 完整 4K ms | 延迟增加 |
|---|---:|---:|
| CPU 2.4 GHz、NPU 1 GHz、DDR 2.112 GHz | 231.91 | — |
| CPU 2.4 → 1.608 GHz | 269.03 | 16.01% |
| NPU 1.0 → 0.7 GHz | 240.07 | 3.52% |
| DDR 2.112 → 1.560 GHz | 272.24 | 17.39% |

## 工程判断

C 已值得在真实模型试接，优先核对 512/4096 的 Attention 累计时间、TTFT、首步 logits，以及回退路径。继续做单算子时，最大的已识别项仍是提交外的 63.5 ms score 处理。一个具体实验是按三个 NPU 核把 KV heads 分成 3、3、2 三组，每组独立产出完整上下文的 QK。CPU 可以先处理已完成组的真实行 max 和 exponent，同时让 NPU 继续计算下一组。应检查高频提交和分段同步是否吃掉重叠收益，并与 C 做完整 4K 配对。这一调度尚未实测。

若把原库的 297 ms 减半作为目标，同层 C 的 QK+PV 驱动区间已占约 135.5 ms。在两段执行时间和串行设备依赖固定的假设下，只剩约 13 ms 给全部暴露的 CPU、提交和同步。当前测试支持进一步优化 CPU 和数据搬运，也给出了单靠调流水接近两倍加速所需满足的时间预算。

## 协议和边界

主板，RK3588，RKNPU 0.9.8。Clang 22.1.3、O2、CPU4–7，三核 NPU，沿用上轮的 Qwen3-4B-W4A8-Mixed-LMHeadW8 post-RoPE Q/K 和 V、公共 C API、Hq/Hkv=32/8、D128、Q128、KV tile512 及相同的依赖对象。所有增长用例从空 KV 逐块真实追加，输入准备、首次容量预留、KV reset 在计时外。三个长文本层的增长输出逐位一致且全部有限。第 17 层两个热块的 18,874,368 个 FP16 PV 输入也逐位一致；另一个 22-token 对话的 36 层输出和 PV 输入均一致。[数值记录](evidence/gqa_stream_prefill_20261118/validation.json) · [确认脚本](evidence/gqa_stream_prefill_20261118/confirm.sh) · [锁与时钟恢复](evidence/gqa_stream_prefill_20261118/confirm-clocks.json)

结果限于这一模型、所选输入和单层算子。整模型 TTFT、PPL、不同驱动的 write-combine 支持和其他几何的性能仍待验证。实验补丁以完整原库为参照，剔除了用于捕获中间量的调试代码；实际编译和复核脚本、CSV、Perfetto trace 已归档。[实验目录](evidence/gqa_stream_prefill_20261118) · [含冻结依赖、可执行文件与原始输出的共享工作区](<HOST_HOME>/mnt/3090x4/tmp/gqa-causal-stream-20261118)
