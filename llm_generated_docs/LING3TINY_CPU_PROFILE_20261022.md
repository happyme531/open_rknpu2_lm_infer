# Ling3tiny 当前路径的 CPU 热点，2026-10-22

当前实现为 router NPU TF32、MLA q_a/q_b/kv_a/g_proj NPU FP16，dense 输出
投影仍是 checkpoint 的 NPU W4；本报告排除 KDA recurrence 核心。

## 口径与完整性

主板 zt-firefly，CPU4–7/4 线程 @2.4 GHz、NPU mask7 @1 GHz、DDR @2.112 GHz，
resident、context640、ubatch128。使用既有固定 MT19937 seed42 的 640 个 token，
连续 pp512→tg128，head-free prefill，decode 含 LM head；不计加载、prepare、
tokenization、sampling 和 stats。

主表取三次 `llm,rknpu` trace 的均值。Prefill wall 为 3289.109 ms，decode
为 6804.316 ms；同二进制关闭采集的三次对照分别平均 3260.695 / 6512.015 ms，
采集开销为 +0.871% / +4.489%。表中百分比以各自完整阶段 wall time 为分母，
CPU 行是该阶段的 elapsed wall time，包含其并行调度/等待，并非累计 CPU cycles。
不把 NPU submit 等待计为 CPU 算术。

另采集一次全量 `rknpu.detail` trace，补查 NPU 调用内的 CPU pack/reduce/finish。
该次 wall 为 3328.023 / 7100.934 ms，较关闭采集对照为 +2.065% / +9.044%，
只用作细分诊断，不替代正式生产速度。三份主 trace 和一份细分 trace 的
非 info health stats 均为空，无 loss/overrun；MoE 调用数每份均为 prefill92、
decode2944，KDA recurrence/conv、MLA core、head 调用数均匹配真实层数和 token 数。

## 主要 CPU 阶段

下面互不重叠，均排除 KDA recurrence；不会把完整 NPU 算子当作 CPU 时间。

| CPU 阶段 | Prefill512 ms | 占比 | Decode128 ms | 占比 |
|---|---:|---:|---:|---:|
| 专家 CPU 中间处理：INT32 反量化、SwiGLU、A8 再量化及打包 | 386.46 | 11.75% | 119.97 | 1.76% |
| KDA Q/K L2 norm、decay/beta 激活 | 253.65 | 7.71% | 72.58 | 1.07% |
| 专家 CPU 输出：反量化、路由加权和合并 | 233.52 | 7.10% | 182.97 | 2.69% |
| KDA 输出 RMSNorm + sigmoid gate | 215.48 | 6.55% | 58.48 | 0.86% |
| Router sigmoid、group top-k、权重归一化 | 85.33 | 2.59% | 33.13 | 0.49% |
| KDA causal convolution + SiLU + history 更新 | 66.23 | 2.01% | 297.30 | 4.37% |
| 层 RMSNorm 与 residual add | 63.20 | 1.92% | 30.51 | 0.45% |
| MLA q/kv RMSNorm、RoPE | 49.77 | 1.51% | 16.54 | 0.24% |
| 专家输入 A8/bitplane、路由 job 调度、动态命令构建 | 41.15 | 1.25% | 150.83 | 2.22% |
| KDA beta 的保护浮点 CPU 投影 | 13.80 | 0.42% | 87.31 | 1.28% |

专家 CPU 各项合计 **661.13 ms / 20.10% prefill**、**453.77 ms / 6.67% decode**。
KDA 门控/归一化加 beta 投影合计 **482.93 ms / 14.68% prefill**、
**218.37 ms / 3.21% decode**，不含 recurrence 和卷积。

## NPU 调用内部的额外 CPU 部分

一次完整细分 trace 中，普通 W4/W8 投影的 CPU 部分为：

| CPU 部分 | Prefill512 ms | Decode128 ms |
|---|---:|---:|
| W4 activation scan、input pack、输出 reduce/dequant | 161.99 | 180.58 |
| W8 input pack/quant、输出 reduce/dequant | 35.71 | 90.30 |
| MLA attention query/cache pack、center/softmax、output finish | 115.01 | 43.46 |

这些与主表 MoE 的专用 grouped 算子 CPU 计时互不重复。W8 span 位于其 worker
track，查询包含全部 track；不能只查模型主线程而漏掉它们。没有相加父/子 span，
例如 W4 SwiGLU epilogue 已在 wave_reduce 内，不再另加。异步流水中的 CPU span
可能与 NPU 执行重叠，所以它们不是可直接从端到端时延中扣掉的收益。
这张表也不是所有 NPU 调用 CPU 开销的穷尽统计：FP16/TF32 host IO 等未单独分项。
另外的 sync/submit/blocked wait 单独保留在原始数据，不冒充 CPU 算术。

MLA 的 115.01 ms 中，attention.output_finish 为 81.91 ms、query_pack 24.91 ms、
center_scores 6.50 ms、cache_pack 1.69 ms。output_finish 包含 FP16 partial
还原/求和、carrier 修正、分母归一化和输出 layout 整理；MLA 前投影移到 NPU
之后，这一段仍然在 CPU。

## 优化优先级与源码依据

1. **Prefill：先看专家中间处理/合并、KDA 门控/归一化。**
   `rk_npu_moe_w8.cpp` 的 middle 当前逐元素 `std::exp` 计算 SwiGLU，之后再 A8
   量化/bitplane 打包；combine 则对 top8 的 INT32 输出反量化并加权。KDA 的
   sigmoid/normalization 也在 Rust 循环中，尚未使用统一的批量 SIMD gate 路径。
   优化这些 CPU 路径或核对融合机会之前，不应仅以 NPU activation 的局部耗时
   宣称端到端收益，必须包括新增 sync/layout/downstream quantization。
2. **Decode：卷积的单 token 调度、专家 combine、保护 beta 投影。**
   `ConvState::run_parallel_strided` 按 token 分 Rayon chunk；M1 只有一个任务，
   仍经过 pool.install 和调度。卷积 K=4 与 SiLU 已有 NEON，不能说缺 SIMD。
   专家 combine 的 workers 也按 token 分配，M1 时实际只有一个 worker 有输出
   token。这些是源码可证实的调度问题；各自有多少耗时能消除仍需单算子配对测试。
3. **MLA RoPE 可去掉重复函数计算，但优先级较低。**
   同一 position 的 angle/powf/sin_cos 在 KV 和 16 个 Q heads 中重复；
   可考虑复用频率及 sin/cos。49.77 ms 是 Norm+RoPE 合计，不能全归因于三角函数。
4. Router 剩余 CPU 只有选择和归一化；不再包含 dense projection。Prefill 占
   2.59%，decode 0.49%，低于以上主要目标。

本次没有修改推理算术。复用 native 已有 `rk_npu_moe_w8_timings`，只新增
`moe.timings` Perfetto 事件导出其结果；producer shared buffer 从默认值增为
16 MiB，解决已证实的 SMB overflow 丢包，普通关闭 trace 的构建不受影响。
正常推理二进制保留，诊断二进制为 `/tmp/ling3tiny-tf32-trace`。

## 证据与复现

- 三份主 trace：`results/ling-cpu-primary.case0.rep{0,1,2}.pftrace`。
- 算子细分：`results/ling-cpu-detail.case0.rep0.pftrace`。
- 原始查询：`results/ling-cpu-profile-raw.json`；汇总：`results/ling-cpu-profile-summary.json`。
- benchmark/control：`results/ling-cpu-{primary,detail,control}.json` 及对应时钟报告。
- SQL：[LING3TINY_CPU_PROFILE.sql](LING3TINY_CPU_PROFILE.sql)。

编译仍用原有 `RK_NPU_ENABLE_TRACE=ON` 和 pinned Perfetto v58.2 SDK，命令沿用
[原阶段报告](LING3TINY_PHASE_PROFILE_20261022.md)；主采集设置
`RK_NPU_TRACE_CATEGORIES=llm,rknpu`，细分另加入 `rknpu.detail`，输出路径改为
`OPEN_RKNPU2_PROFILE_TRACE=results/ling-cpu-primary.pftrace` / `ling-cpu-detail.pftrace`。
所有定频 runner 的 restore_errors 均为空，三个 benchmark 的 head-free 检查通过，
cache miss/search/fallback/weight reload 均为零。

SQL 已使用 Perfetto v58.2 执行通过，可复现：

```sh
.cache/perfetto-v58.2/trace_processor_shell query \
  -f llm_generated_docs/LING3TINY_CPU_PROFILE.sql \
  results/ling-cpu-detail.case0.rep0.pftrace
```
