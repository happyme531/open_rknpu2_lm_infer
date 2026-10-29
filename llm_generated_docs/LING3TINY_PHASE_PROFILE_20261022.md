# Ling3tiny prefill/decode 耗时归因

本报告记录 router 和 MLA 前投影仍在 CPU 时的版本，作为迁移到 NPU 的
基线。后续源码中 router 已改为 NPU TF32，MLA 前投影已改为 NPU FP16；
以下原始 trace 和数值保持不变。

已调优的 resident 路径，主板 `zt-firefly`，CPU4–7/4 线程 @2.4 GHz，
NPU mask7 @1 GHz，DDR @2.112 GHz。ubatch128、context640，连续
prefill512→decode128；沿用此前 MT19937 seed42 的同一组 640 个 token。
一次 warmup、三次采集，计时不含加载、prepare、tokenization、sampling 和统计。
Prefill 不执行输出头；decode 每个输入 token 执行一次输出头。

复用算子库现有 Perfetto session 与 `llm` category，在公共
`src/model/trace.rs` 接入 Rust span，使用现有 `bench` 入口。
只采集 `llm` 阶段；三份 trace 的 loss/overrun/错误统计均为空。
KDA recurrence、MLA core、routed expert 和 output-head 调用次数均与
18/6/23 层及 512/128 token 数量一致。每个阶段使用不重叠的叶 span，
下表百分比以整个阶段的 wall time 为分母，合计 100%。

所有 NPU 项均为完整调用，包含 CPU 准备、量化、同步、等待及输出收集，
不代表纯硬件计算时间。总时延相对同二进制关闭采集的对照：prefill +0.816%、
decode +2.402%。缓存命中 86，miss/search/fallback 均为 0；head-free logits
检查通过，定频 runner 正常恢复频率。

## Prefill512

三次 wall time：4000.266630、3998.033310、4000.441341 ms；平均 3999.580427 ms。

| 按耗时排序 | ms | 占比 |
|---|---:|---:|
| 路由专家 FFN，W4 完整调用 | 1164.61 | 29.12% |
| CPU 路由：投影、分组 top-k、权重 | 592.52 | 14.81% |
| CPU KDA recurrence | 568.25 | 14.21% |
| KDA CPU 门控/归一化，含 beta 投影 | 484.04 | 12.10% |
| MLA CPU 前投影及 Norm/RoPE | 337.76 | 8.44% |
| MLA NPU 核心及输出投影 | 309.47 | 7.74% |
| KDA NPU Q/K/V、f/g、o 投影 | 261.77 | 6.54% |
| 其他：层 norm/残差、首层 dense、分配/控制及计时差额 | 117.71 | 2.94% |
| 共享专家 FFN，W8 完整调用 | 97.30 | 2.43% |
| KDA CPU causal convolution | 66.14 | 1.65% |

KDA 门控/归一化细分：Q/K L2 norm、decay/beta 激活 253.32 ms（6.33%）；
输出 RMSNorm + sigmoid gate 217.07 ms（5.43%）；CPU beta 投影 13.64 ms（0.34%）。
MLA CPU 前投影：q_a 50.31 ms、q_b 119.23 ms、kv_a 113.93 ms、g_proj 3.75 ms；
Norm/RoPE 50.54 ms。MLA NPU core 292.29 ms，输出投影 17.18 ms。

## Decode128，历史深度 512…639

三次 wall time：6726.981819、6729.107512、6717.933332 ms；平均 6724.674221 ms。

| 按耗时排序 | ms | 占比 |
|---|---:|---:|
| 路由专家 FFN，W4 完整调用 | 1664.15 | 24.75% |
| KDA NPU Q/K/V、f/g、o 投影 | 1352.56 | 20.11% |
| LM head，W8，含最终 norm 与输出检查 | 1060.85 | 15.78% |
| 共享专家 FFN，W8 完整调用 | 647.65 | 9.63% |
| MLA NPU 核心及输出投影 | 543.18 | 8.08% |
| MLA CPU 前投影及 Norm/RoPE | 315.26 | 4.69% |
| KDA CPU causal convolution | 291.84 | 4.34% |
| CPU KDA recurrence | 237.69 | 3.53% |
| KDA CPU 门控/归一化，含 beta 投影 | 229.95 | 3.42% |
| CPU 路由：投影、分组 top-k、权重 | 219.01 | 3.26% |
| 其他：层 norm/残差、首层 dense、分配/控制及计时差额 | 162.53 | 2.42% |

KDA 投影细分：Q/K/V 688.59 ms（10.24%）、f/g 438.46 ms（6.52%）、
o 225.51 ms（3.35%）。MLA NPU core 465.52 ms（6.92%），输出投影
77.66 ms（1.15%）。

## 源码含义

1. Prefill 的 CPU 路由 + KDA recurrence + KDA 门控/归一化合计 41.12%。
   路由当前逐 token 调用 CPU `dense_head` 并分组排序，512 token ×23 层
   共 11776 次；没有使用按整个 ubatch 的矩阵乘。CPU dot 已有 NEON。
2. KDA 的 Norm/gate 在 Rust 中执行，recurrence 在独立 CPU kernel 中执行。
   Prefill 的门控/归一化与 recurrence 成本相近，decode recurrence 只有 3.53%。
3. MLA core 使用吸收式 NPU 路径。checkpoint 的 q_a/q_b/kv_a/g_proj 都保留
   BF16 权重，当前公共 `Linear::load` 将这些浮点投影映射到 `Host`，所以前投影
   仍在 CPU；这部分及 Norm/RoPE 占 prefill 8.44%、decode 4.69%。
4. Decode 的 routed experts + KDA 线性投影 + lm_head 合计 60.64%。此前的
   自动调优覆盖普通线性层，grouped MoE、CPU KDA 与 MLA core 使用专用实现。

以上是阶段归因；没有用阶段 wall time 推断 NPU MAC 上限或实测 DDR 流量。

## 复现与原始数据

编译时使用 `RK_NPU_ENABLE_TRACE=ON`；本机已存在的 pinned SDK 为：
`expr/lowlevel/rk_npu_matmul/third_party/perfetto-v58.2`，通过
`RK_NPU_PERFETTO_ROOT` 指定。正常构建默认关闭 trace，Rust spans 编译为空操作。

在先前调优测速的相同命令前增加：

```sh
RK_NPU_TRACE_CATEGORIES=llm \
OPEN_RKNPU2_PROFILE_TRACE=results/ling-stages.pftrace \
python3 scripts/run_board.py --report results/ling-profile-stages-clock.json -- \
  /tmp/ling3tiny-target-20261022/release/open_rknpu2_lm_infer \
  --model /home/firefly/mnt/spark-test/Ling-3.0-tiny-QQQ-W4A8-Mixed-CT \
  --ctx-size 640 --ubatch-size 128 \
  --expert-store /tmp/ling3tiny-expert-store-20261022 --expert-buffered-io \
  --tune-cache /home/firefly/.cache/open_rknpu2_lm_infer/ling3tiny-20261022 \
  bench -p 0 -n 0 -pg 512,128 -r 3 --report results/ling-profile-stages.json
```

三份 trace 为 `results/ling-stages.case0.rep{0,1,2}.pftrace`。
Perfetto v58.2 查询见 [LING3TINY_PROFILE.sql](LING3TINY_PROFILE.sql)。
各叶阶段与逐层查询原始结果：`results/ling-stage-trace-raw.json`；
细分汇总：`results/ling-stage-profile.json`；分组汇总：
`results/ling-stage-profile-grouped.json`。对照为 `results/ling-profile-control.json`。
trace 开/关两种构建的全 targets Clippy 均通过。
