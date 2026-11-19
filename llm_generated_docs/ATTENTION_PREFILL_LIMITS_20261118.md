# RK3588 stable GQA prefill 的利用率和下一步实验

Qwen3-4B 几何下，当前 native 库从空 KV 增长到 512 和 4096 token，单层 Attention 的有效 FP16 利用率分别为 **5.4% 和 14.9%**。在 4K 上，要把 308 ms 减半，保留现有 QK、PV 执行时间后，整层只剩约 21 ms 做 softmax、KV 追加、打包、同步和输出。这个预算解释了继续调 task 数或只优化 CPU 行最大值的局限。尚无证据把 308 ms 认定为 RK3588 的硬件极限。

## 当前基线

外层仓库 HEAD `e133194` 记录的 submodule 是 `20b682d`；实际 native 工作树和本地 native main 均为 `3be24de`，包含 M128 的 2D LUT。当前分支另有未提交的 fixed-shift 图复用改动。下列正式性能只取同批次、策略标记为 `stable` 的第 0、7 个进程。探针使用 Qwen3 几何的合成 FP16 Q/K/V，Hq/Hkv=32/8，D=128，ubatch128，KV tile512，三核，CPU/NPU/DDR 为 2.4/1/2.112 GHz。每进程五次正式计时，取各自中位数后平均。[运行汇总](evidence/gqa_fixed_shift_20261105/summary.json) · [探针](../native/rk_npu_matmul/tests/gqa_device_softmax/fixed_shift.cpp)

| 单层 causal Attention | 有效 QK+PV GFLOP | 墙钟 ms | 有效 TFLOP/s | 占 3 TFLOP/s |
|---|---:|---:|---:|---:|
| 从空 KV 增长到 512 | 2.152 | 13.302 | 0.162 | 5.4% |
| KV512 上的末尾 Q128 | 0.941 | 3.446 | 0.273 | 9.1% |
| 从空 KV 增长到 4096 | 137.473 | 307.704 | 0.447 | 14.9% |
| KV4096 上的末尾 Q128 | 8.457 | 16.237 | 0.521 | 17.4% |

分子为 `4 × Hq × D × 可见 query-key 对数`，FMA 计两次，完整因果前缀取 `S(S+1)/2`。分母为公共调用墙钟时间乘项目沿用的三核 FP16 峰值。增长计入实际 append，KV 容量预留、reset 和测试程序准备输入在计时外；热调用预装 KV。QKV/O 投影均在算子外。译成硬件问题，这个比例同时受额外的物理运算、访存、DPU、CPU 以及流水空档影响。历史整模型 pp512 的 Attention 阶段为 0.099 有效 TFLOP/s，折合 3.3%；它有独立的软件版本和运行边界。[历史模型剖析](QWEN3_PP512_EFFECTIVE_TOPS_20261020.md)

## 一次 4K prefill 的时间花在哪里

每个 KV head 把四个 query head 各 32 行叠成物理 M128。每个公共 Q128 调用包含四块。NPU 写原生布局的 FP16 QK score；CPU 同步读，按因果前缀求每行真实最大值，第二遍减 max、截断并写回；DPU LUT 把它变成带正偏置的指数。CPU 在 PV 前清零未来位置的偏置；NPU 分 KV tile 同时算分子和分母，CPU 以 FP32 合并 partial、扣除偏置并归一化。两个 slot 交错执行，最多有一个 NPU submit 在途。最后一个 Q128 的矩形 score 总共 32 MiB，分四块处理。[图构造、centering 和 finish](../native/rk_npu_matmul/src/rk_npu_attention_f16.cpp) · [流水调度](../native/rk_npu_matmul/src/rk_npu_attention_f16_pipeline.cpp)

下表重新解析同版本两个 stable 进程各自独立的第六次 trace，列出两者范围。对 host slice 取与全部 `driver.submit` 时间区间的交集，从自身墙钟中扣除交集得到“提交外”时间。对 QK/LUT/PV 读取对应 `graph_submit` 下驱动返回的 `hardware_us`。正式墙钟用上表的无 trace 样本。[stable 进程 0](evidence/gqa_fixed_shift_20261105/paired0/growth-4096.pftrace) · [stable 进程 7](evidence/gqa_fixed_shift_20261105/paired7/growth-4096.pftrace) · [slice 查询](evidence/gqa_trace_20261104/slices.sql)

| 从空 KV 增长至 4096，单层 | 累计 ms | 其中在 NPU 提交外的 ms |
|---|---:|---:|
| trace 墙钟 | 312.64–312.97 | 110.95–111.70 |
| QK，驱动区间 | 54.27–54.40 | — |
| LUT，驱动区间 | 63.64–63.88 | — |
| PV，驱动区间 | 78.14–78.40 | — |
| CPU 求 max、减 max、截断 | 108.39–109.50 | 34.87–35.75 |
| score FROM 同步 | 18.53–18.74 | 9.10–9.17 |
| score TO 同步 | 31.23–31.44 | 12.45–12.52 |
| CPU 输出收尾 | 52.96–53.24 | 18.01–18.28 |
| 图几何及 tail 准备 | 16.92–17.21 | 7.43–7.61 |
| Q 打包 | 12.64–12.79 | 8.66–8.80 |
| KV 追加 | 9.35–9.36 | 9.35–9.36 |

三个依次提交的阶段合计约 196 ms。在设备时长和串行依赖固定的假设下，即使所有其他时间都能隐藏，308/196 也只有约 1.57×；这是对当前调度的条件估算。每个矩阵乘各贡献约 68.74 GFLOP 的可见工作。按各自驱动区间折算，QK 约 1.26 TFLOP/s，PV 约 0.88 TFLOP/s；PV 还计算分母并写 tile partial。流水已覆盖约三分之二的 CPU centering。即使实现能免费完成 centering，也不能直接从 308 ms 中减去 109 ms；“提交外”同样只是时间线归属，修改调度会改变重叠和带宽竞争。512 增长的两份 trace 则显示，finish 暴露 2.28–2.33 ms，centering 只暴露 0.60–0.93 ms。[512 trace A](evidence/gqa_fixed_shift_20261105/paired0/growth-512.pftrace) · [512 trace B](evidence/gqa_fixed_shift_20261105/paired7/growth-512.pftrace)

## 历史实验已经回答的问题

| 改动 | 板上结果 | 对下一步的约束 |
|---|---|---|
| 两个 slot 交错执行，保留算术顺序 | 当时 Qwen3 4K 热调用 22.73→17.39 ms。真实整模型 4K head-free prefill 32.85→29.67 s，Attention 16.88→14.27 s。[正式集成](ATTENTION_PIPELINE_INTEGRATION_20261024.md) | 应与开启双 slot 的稳定路径比较 |
| M128 LUT 采用二维连续大块 | 同库配对，单层 4K 热调用降低 3.35%，4K 增长降低 5.82%，增长输出逐位一致。[性能](evidence/gqa_lut_geometry_20261105/summary.json) · [正确性](evidence/gqa_lut_commit_20261105/verification.json) | 已进入本地 native main，不能重复计收益 |
| 整链驻留 NPU，保留精确 max | QK→ReduceMax→广播减法→LUT→精确 mask→PV。所测输出逐位一致，KV512/4096 热调用分别慢 5.65%/13.20%；4K 为约 17.89→20.25 ms，实验端还免计建链和预制 mask。[数据](evidence/gqa_device_softmax_20261104/summary.json) | 省 CPU 往返的价值有上限，添加多个全表设备阶段会吃掉它 |
| CPU 只回传 max；或每行单独融合减法和 LUT | 前者把热 4K 回传从 32 MiB 降到 8 KiB，完整 4K 增长慢 4.70%。后者热 4K 任务由 544 增到 4384，15.91→35.27 ms。[紧凑 max](evidence/gqa_hybrid_max_20261105/summary.json) · [逐行融合](evidence/gqa_fused_sub_lut_20261105/summary.json) | 需同时控制实际遍历次数、原生布局与任务粒度 |
| 保留真 max，第二遍扩维 QK 融合减法和 LUT | 旧基线的热块能快 25%。Qwen3 4K 整模型复测 Attention 降低 11.55%、head-free prefill 降低 3.88%；首轮有慢样本。Qwen3 8160-target PPL 相对升 0.245%。[重算原型](ATTENTION_RECOMPUTE_PROBE_20261022.md) · [模型验证](ATTENTION_STREAM_20261023.md) | 两遍都稳定仍可能改变 FP16 舍入。缺相对当前 2D LUT 加双 slot 的直接优势 |
| 直接切现成 CPU exp、换 PPU 或改跑 GPU | 开流水时 CPU exp 的 512 增长快 13.86%，4K 增长慢 10.73%，数值配方也变了。官方 512×128 大窗口 max 图出现 681 个 PPU 任务加四个卷积。优化后的 Mali 实现在其 4K 热对照中为 74.48 ms，同场 NPU 13.88 ms。[CPU exp](evidence/gqa_exp_paths_20261105/summary.json) · [PPU](evidence/gqa_ppu_20261104/summary.json) · [Mali](evidence/gqa_vulkan_opt_20261104/final-summary.json) | 分别约束这些具体实现；换执行单元本身不创造收益 |
| 全局 fixed shift | 真实 Qwen3 的 22-token、36 层 QKV 回放，shift=4 有 36 层出现非有限值，shift=12 仍有 12 层。[数值记录](evidence/gqa_fixed_shift_20261105/summary.json) | 排除这条路线，不计入稳定实现的可交付增益 |

旧版 Q32/KV4096 的同板三核降频对照还给出硬件线索：保持 NPU 1 GHz，把 DDR 2112→1068 MHz，causal 块从 2.679 增至 3.965 ms；保持 DDR 2112 MHz，把 NPU 1000→600 MHz，只增至 3.085 ms。该原型采用早期 exp 配方。这证明当时共享数据路径对速度重要，当前 stable 版本是否饱和 DDR 仍需按阶段测 NoC 读写字节、有效带宽及频率敏感性。官方 RKNN 2.3.2 的 Q32/KV4096 对照采用单层 exSDPAttention NPUTime，25.69 ms；同轮原型的 3.73 ms 包括提交、输出同步和收尾，双方输入准备边界及数值也不同。[频率实验](../../lowlevel/tools/attention_prefill_probe/MULTICORE.md) · [官方单层对照](../../lowlevel/tools/attention_prefill_probe/README.md)

## 最值得做的一次实验

**保留真实行最大值，让 CPU 在第二遍读 score 时直接写出与当前 DPU LUT 逐位相同的 PV 输入。** 现在 CPU 已经在第二遍得到舍入为 FP16 的 `clamp(score − max, −16, 0)`。该区间只有 19,457 个不同幅值的 FP16 输入。用幅值的 half bits 查一张 38,914 B 的只读表，直接得到最终 FP16 exponent。可见位置保留当前 LUT 的正偏置，不可见位置当场写真正的零，沿用现有 PV 顺序、分母通道、前缀修正和 FP32 finish。这可以复用 score 缓冲，去掉独立 LUT 的整表读写、一次提交以及 LUT 后的 causal 尾带处理。QK→CPU 和 CPU→PV 的所有权同步仍需支付。[当前查表配方](../native/rk_npu_matmul/src/rk_npu_attention_f16_lut.h) · [当前 causal 清零与同步](../native/rk_npu_matmul/src/rk_npu_attention_f16.cpp)

本次在主机按当前 gain、bias、索引和整数插值公式重建输出，用 lowlevel 保存的板上曲线枚举核对，19,457 项的 FP16 bits 全部相同，正负零共用零下标。历史 CPU emulation 已验证这种重现硬件数值的思路，但其控制实现逐行遍历、做标量算术并用临时数组；另有实验使用补偿表。原型应直接对齐本 checkout 的实际 LUT 配方。这一次验证仅覆盖标量曲线，完整 PV 输出和板上速度仍待测。[原始曲线](../../lowlevel/tools/npu_cpu_offload_probe/results/2026-10-17/attention-precision/curve.csv) · [硬件曲线模型](../../lowlevel/tools/npu_cpu_offload_probe/ATTENTION_PRECISION.md) · [历史控制实现](../../lowlevel/tools/npu_cpu_offload_probe/attention_probability_probe.h)

**为什么有机会，又为什么可能失败。** 最后一个 Q128/KV4096 块有 16.78M 个矩形元素；当前独立 LUT 的驱动区间为 3.90–3.92 ms。按四个 2.4 GHz 大核均摊，大约只值每元素 **2.2 个新增 CPU 周期**。这是只把 LUT 时间当预算的数量级估算。并发重叠、尾带删除和不同的缓存争用都能移动盈亏点。A76 的间接查表吞吐和双 slot 同时读写内存，足以令它失败。先用相同原生布局、四大核和并发 QK 对照现有 centering 与融合写回，再测公共 API；单独测热缓存中的查表循环会高估收益。

单层实验以当前三核、双 slot、2D LUT 为 A，以只改第二遍和 PV 输入的原型为 B。两边跑 512/4096 增长和末尾热块，ABBA 交错；核对有效位置、斜线边界、padding 和真实 QKV 上 PV 输入及最终输出的所有位。只有输出吻合且完整增长配对稳定变快，才进入真实 Qwen3 的 512/4096 prefill 和 logits。若只能在短前缀获益，就限定结论到该范围。未提出指标要求时，性能是否值得集成以实际模型耗时和实现成本判断。

## 更高收益的条件

短 prefill 的低风险候选是 `Graph::finish`。它现在串行遍历独立 KV heads，共用 `sum` 和前缀 scratch；按 head 划分私有 scratch 可保留每行每 tile 的相加顺序。先检查 512 增长中 2.3 ms 的暴露收尾能减掉多少，并把线程启动、与 NPU 同时访存和复用缓冲的成本都算入。这一项独自解释不了 4K 的倍数级收益。

长 prefill 若追求接近两倍，先量出 **同形状 QK + PV 的地板**。预置已验证的 exponent，保持 QK、PV 的实际 shape、三核分工、地址布局、tile、输出和缓存状态，对照真实完整调用，并测 NoC 流量。预置输入的实验只用于估计地板。按现有 trace 两段串行设备时间合计约 133 ms，2× 目标为 154 ms，留给其他阶段的 21 ms 甚至低于当前仅暴露的 centering。若设备段本身也要缩短，可看 PV 的物理 N160（128 个 V 通道加一列分母再对齐）和每 tile 写 partial 成本；单纯删除分母列会改变归一化，需要把相同分母的获取及舍入成本算回来。

算法上还可以在 KV tile 内求真实局部 max，形成 `(m, sum_exp, sum_exp_V)`，用 FP32 按各 tile 最大值差缩放合并。这个方向成立的条件是原型确实少写或少读了一整份大中间量，而且局部 LUT 截断、正偏置、causal 处理和 FP16 partial 的误差能通过真实模型验证。RKNPU2 已验证的 CBUF 服务于矩阵取数，当前缺少让 DPU 输出直接成为下一次 GEMM 输入的通用片上通路。单纯缩小 tile 后仍逐阶段写 DDR，会增加任务和合并成本。先对一个真实 M128/KV512 tile 比较完整链的字节与时间；有可测的跨阶段复用，再扩成 4K 增长。[项目硬件记录](../../lowlevel/docs/kernel-writing/01-hardware-behavior.md) · [已有分块设计与约束](ATTENTION_FIRST_PRINCIPLES_20261022.md) · [FlashAttention 的 IO 前提](https://arxiv.org/abs/2205.14135)

本次结论来自既有板测、原始 trace 重算、当前源码和主机穷举。新候选没有板端性能结果。
