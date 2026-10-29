# Ling3tiny 分阶段耗时与后续优化 20261023

Prefill 的主要瓶颈是 routed MoE 与 KDA recurrence，两者占完整阶段的 **61.98%**。Decode 的 routed/shared experts 加 LM head 占 **54.58%**。后续优先优化 MoE 的 CPU 中间处理及输出合并、KDA state 行更新，以及 decode 的 grouped FFN 调度；router 选择和 RoPE 已降到较小占比。

## 测量条件

当前源码包含 KDA Q/K/V/F/G 融合投影、批量 Norm/gate 并行、M1 卷积直接执行、router scratch 复用和 RoPE 角度共享。主板 zt-firefly；checkpoint 为 `Ling-3.0-tiny-QQQ-W4A8-Mixed-CT`；resident、ubatch128、context640；CPU4–7/4 线程 @2.4 GHz，NPU mask7 @1 GHz，DDR @2.112 GHz。输入为 MT19937 seed42 的固定 640 个 token，连续 prefill512→decode128，decode 历史深度 512…639。Prefill 不执行输出头；完整调用排除加载、prepare、tokenization、sampling 和 stats。

三份 `llm,rknpu` 主 trace 提供阶段均值；一份 `rknpu.detail` 提供算子内部诊断。采集前后分别运行同一二进制关闭采集的三次对照。

| 采集方式 | Prefill512 平均 ms | Decode128 平均 ms | 相对关闭采集的差值 |
|---|---:|---:|---|
| 关闭采集 前后共六次 | 2900.487 | 6041.728 | — |
| 主 trace 三次 | 2880.009 | 6242.360 | -0.71% / +3.32% |
| detail 一次 | 2949.504 | 6442.619 | +1.69% / +6.64% |

Prefill 的小幅反向差异属于测量波动，不作为加速收益。主表按主 trace 的完整阶段 wall time 计算百分比；模型叶 span 互不重叠，各行合计 100%。NPU 项包含 CPU 准备、同步、提交等待和收集，不表示纯硬件 MAC 时间。detail 内部 span 与其模型父 span 不相加。

## 完整阶段分解

| 部分 | Prefill ms | 占比 | Decode ms | 占比 |
|---|---:|---:|---:|---:|
| Routed MoE FFN 完整调用 | 1166.06 | 40.49% | 1674.97 | 26.83% |
| CPU KDA recurrence 调用 | 619.07 | 21.50% | 257.24 | 4.12% |
| MLA absorbed core 完整调用 | 289.06 | 10.04% | 469.91 | 7.53% |
| KDA NPU 输入及输出投影 | 187.25 | 6.50% | 997.31 | 15.98% |
| KDA Norm 与 gate 及 CPU beta 投影 | 171.19 | 5.94% | 252.79 | 4.05% |
| 共享专家 FFN 完整调用 | 101.37 | 3.52% | 662.14 | 10.61% |
| TF32 router 投影及 CPU 选择 | 99.35 | 3.45% | 227.24 | 3.64% |
| CPU causal convolution 与 SiLU | 68.27 | 2.37% | 160.10 | 2.56% |
| MLA NPU 前投影及输出投影 | 51.48 | 1.79% | 317.78 | 5.09% |
| MLA CPU Norm 与 RoPE | 9.03 | 0.31% | 5.56 | 0.09% |
| LM head 含最终 Norm 与输出检查 | 0.00 | 0.00% | 1070.26 | 17.15% |
| 其他层 Norm 残差 首层 dense 分配及控制 | 117.88 | 4.09% | 147.06 | 2.36% |

KDA recurrence span 包含 Rust 输入值扫描、OpenMP 组织和 C++ 递推。KDA Norm/gate/beta 的 prefill 细分为 99.00 / 57.81 / 14.39 ms，decode 为 96.05 / 57.37 / 99.36 ms。Router 的 NPU 投影为 58.92 / 207.31 ms，CPU 选择为 40.43 / 19.93 ms。

完整叶阶段及逐层数据保存在 `trace-raw.json`、`trace-summary.json`、`layers.csv`。逐层列出的耗时为该层叶 span 总和，不包含模型外层分配/控制以及 LM head。

## Routed MoE 内部

复用 `rk_npu_moe_w8_timings`，三份主 trace 取均值。这里细分上表 routed MoE，不增加到主表总数。

| MoE 子阶段 | Prefill ms | 占整个 prefill | Decode ms | 占整个 decode |
|---|---:|---:|---:|---:|
| 输入动态 A8 与 bitplane 打包 | 13.01 | 0.45% | 68.34 | 1.09% |
| 路由 job 调度及命令构建 | 28.47 | 0.99% | 84.61 | 1.36% |
| NPU gather 完整调用 | 55.44 | 1.92% | 0.31 | 0.00% |
| Gate/up 提交与等待 | 251.55 | 8.73% | 748.93 | 12.00% |
| CPU 反量化 SwiGLU 再量化及打包 | 388.15 | 13.48% | 123.04 | 1.97% |
| Down 提交与等待 | 125.85 | 4.37% | 399.30 | 6.40% |
| CPU 输出反量化及路由合并 | 241.34 | 8.38% | 177.35 | 2.84% |
| 设备内存同步 | 58.15 | 2.02% | 48.67 | 0.78% |

MoE CPU 输入、调度、构建、中间处理和合并合计 **670.97 ms / 23.30% prefill**、**453.34 ms / 7.26% decode**。其中 middle+combine 为 **629.50 ms / 21.86% prefill**。当前 `middle_npu_ms=0`。ABI total 为 1162.03 / 1652.32 ms，与模型父 span 的差额来自封装及 trace 事件等开销。

## NPU 调用内的诊断

一次 detail trace 中：

- LM head 的 W8 完整调用为 **1062.21 ms**，其中 `w8.wave_submit` 为 **1024.55 ms**，CPU pack+reduce 为 **26.89 ms**。submit 是包含等待的墙钟区间，不等于纯计算时间。
- KDA 融合输入投影的 prefill W4 完整调用为 **172.11 ms**，`i4.wave_reduce` 为 **85.36 ms**。这一内部 CPU 收集/反量化值得做真实 N10240 形状的 complete-call 检查。detail 对该调用的扰动明显，不能将其与主 trace 的 137.47 ms 直接相减。
- MLA prefill 的 `attention.output_finish` 为 **86.85 ms**，query pack 为 **24.24 ms**；attention 内核父 span 为 **174.86 ms**。外层 absorbed query/value 投影与相关搬运还包含在模型的 MLA core 中。
- 共享专家 decode 包含 **5888 次 W8 Linear**；detail 的 gate/up、down 完整调用分别为 **431.77 / 315.87 ms**。CPU pack+reduce 为 **63.23 ms**，因此不能仅靠优化激活函数消除整个共享专家成本。

## 与优化前的历史 trace 对照

前值来自 `results/ling-cpu-profile-summary.json`，与本次相同模型、频率、ubatch 和主 trace categories。下面用于定位变化，单项因果和回退需配对单算子复核；正式整体收益仍以支持文档中的 A/B/B/A 为准。

| 部分 | Prefill 前 → 当前 ms | Decode 前 → 当前 ms |
|---|---:|---:|
| KDA Q/K/V/F/G 投影 | 217.66 → 137.47 | 1186.67 → 749.79 |
| KDA Q/K norm 与 decay/beta | 253.65 → 99.00 | 72.58 → 96.05 |
| KDA 输出 Norm 与 gate | 215.48 → 57.81 | 58.48 → 57.37 |
| KDA 卷积 | 66.23 → 68.27 | 297.30 → 160.10 |
| Router CPU 选择 | 85.33 → 40.43 | 33.13 → 19.93 |
| MLA Norm/RoPE | 49.77 → 9.03 | 16.54 → 5.56 |
| KDA recurrence 调用 | 569.58 → 619.07 | 259.97 → 257.24 |

历史对照提示两个需要复核的局部变化：prefill KDA recurrence 增加约 49.49 ms，decode Q/K norm 与 gate 增加约 23.47 ms。下一轮可用相同数据分别隔离 recurrence、Rust 输入扫描和 strided gate，检查工作集、缓存与调度影响。

## 后续优化顺序

1. **先优化 routed MoE 的 CPU middle/combine。** `native/rk_npu_matmul/src/rk_npu_moe_w8.cpp:725` 仍逐元素调用 `std::exp`；反量化、激活、max-abs、A8 量化及 bitplane 打包经过多个循环。优先比较 SIMD exp、合并可兼容的遍历以及 scratch 数据布局。Combine 已有 NEON，但 `:752` 按 token 分配 workers，M1 只有一个 worker 工作；应比较按输出通道分块和直接串行路径。保持动态 A8 与路由加权顺序，用真实 H1536/F512/top8 complete-call 检查收益和误差。若 middle+combine 耗时降低 25%，在其余阶段不变的估算下，prefill 可少约 157 ms，约 5.5% 总时延；这不是实测收益。

2. **KDA 保持 CPU，做 state 行更新融合。** `native/cpu_kernels/kda.cpp:24` 分开执行 decay、dot、mad、dot；`vec.h::update_row` 已有 scalar-decay 的“两遍访问”实现。可按同样的 reduction/FMA 顺序扩展 vector decay，在寄存器中复用 state 行，避免四次独立遍历。当前 dot/mad 已使用 NEON，优化目标是减少 state 访问次数。先隔离 Rust 检查和 OpenMP 组织，再比较单/四线程及 M1/M128 的 state/output 等价性与完整调用耗时。

3. **Decode 优先研究 routed/shared FFN 的共同调度及 LM head 的权重路径。** Routed 与 shared 合计 **37.44% decode**。可以评估共享输入量化、混合 W4/W8 task chain 与按工作量分核，减少单独 shared gate/up、down 的串行调用。现有 `RoutedW4SharedW8` 代码可参考，但当前 Ling 使用 certified K-sliced W4，不能直接切换到旧 full-K mixed 路径；需要保留 K 分片的数值边界。LM head 为 `[1,1536]×[1536,157184]` W8：每 token 逻辑权重 241434624 bytes，按当前完整 head 耗时约对应 **28.87 GB/s logical-byte estimate**，并非实测 DDR 带宽。结合其 submit/wait 占比，应先查权重流量、分块与提交区间，不把小量 CPU pack/reduce 视作主要突破口。改 LM-head 位宽属于单独的量化与质量任务。

4. **随后检查 MLA finish 与融合 W4 输出收集。** MLA core 占 10.04% prefill，其中 detail 指向 output_finish；应检查 FP16 partial widening、归一化和布局转换的融合机会。融合 KDA 投影的 W4 reduce 也是明确的 CPU 区间。两者都先使用已有真实形状的单算子 complete-call，对比包括 sync/layout/收集后的总成本，再决定是否集成。

Router CPU 选择当前只占 1.40% prefill / 0.32% decode，MLA Norm/RoPE 为 0.31% / 0.09%；不应继续将它们放在优化首位。直接把 SwiGLU 改成 NPU LUT 也需包含新增同步、布局和再量化成本，不以孤立激活耗时决定是否迁移。

## 完整性与复现

四份 trace 的非 info health counters 均为空。每份主 trace 的 prefill/decode：KDA recurrence 72/2304 次、卷积 216/6912 次、MLA core 24/768 次、routed MoE 92/2944 次；decode head 128 次。全部 benchmark head-free 检查通过，prefill head_ms=0；cache miss/search/fallback/weight reload 均为零，四个 runner 的 restore_errors=[]。

原始证据位于 `results/ling-trace-20261023/`：`primary.case0.rep{0,1,2}.pftrace`、`detail.case0.rep0.pftrace`、`{control,primary,detail,control-after}.json` 与对应时钟报告。`trace-raw.json` 保存全部查询；`trace-summary.json` 保存阶段均值及内部归因；`layers.csv` 保存每层每阶段明细。

采集命令为同目录 `run.sh` 和 `control-after.sh`。诊断二进制为板端 `/tmp/ling-review-20261023/trace`；普通非 trace 构建已恢复到 `/tmp/ling3tiny-target-20261022/release/open_rknpu2_lm_infer`。本次没有修改推理算术。

```sh
python3 tests/tools/ling3tiny_profile.py results/ling-trace-20261023
.cache/perfetto-v58.2/trace_processor_shell query \
  -f llm_generated_docs/LING3TINY_TRACE_20261023.sql \
  results/ling-trace-20261023/detail.case0.rep0.pftrace
```
