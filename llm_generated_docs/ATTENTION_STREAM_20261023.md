# Attention 重算 QK：增量缓存和真实模型验证

重算 QK 已从上一轮的热态工作块扩展到真实模型：增量维护扩维 K、更新 causal mask、复用任务链，并处理缓存切换和截断。**2K Qwen3 的 Attention 阶段降低 8.34%，完整 head-free prefill 降低 2.99%；C7 分别降低 8.58% 和 0.71%。Qwen3 4K 复测分别降低 11.55% 和 3.88%，首轮存在未解释的慢样本。** 这证实数据路径改变有效，也说明热态块的 25% 提升不能当作整模型收益。

实现仍是实验快照，没有默认启用。它改变 FP16 舍入，并额外保存扩维 K。8160-target PPL 已完成：Qwen3 相对增加 0.245%，C7 相对降低 0.034%；不能据此宣称数值无损。

## 真实模型结果

板子为备用板 1 `备用板 1`，RKNPU driver 0.9.6，NPU 三核 1 GHz，CPU4–7 固定 2256 MHz，4 线程。板子没有可用的 DMC devfreq，DDR 频率未受控。

两边使用**同一个可执行文件**，通过实验环境变量 `RK_NPU_ATTN_RECOMPUTE=0/1` 切换；均包含前一轮的 CPU 优化和本轮 span 复用。基线已经优化过，不是未优化的旧库。

计时覆盖全部 decoder 层的 head-free prefill，包含真实 QKV 投影、RoPE、KV append、Attention、输出投影、FFN，以及 C7 的 GDN。模型加载、tokenization、`reset/prepare` 和 LM head 不在计时内。ubatch128，从空 KV 开始，每进程预热一次、正式两次，进程顺序 ABBA。每个方法每个长度 4 个正式样本，表中为均值；配对范围是 A0/B1 与 A3/B2 的进程均值比较，不是置信区间。

| 模型 / prompt | 基线 ms | 重算 ms | 总延迟下降 | 两组配对下降 | Attention ms：基线 → 重算 |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B / 512 | 2860.418 | 2760.401 | 3.50% | 3.23%–3.76% | 856.810 → 799.790 |
| Qwen3-4B / 2048 | 13381.308 | 12981.726 | 2.99% | 2.71%–3.26% | 5353.997 → 4907.285 |
| Qwen3.5 C7 / 512 | 3383.466 | 3256.707 | 3.75% | −0.75%–7.75% | 166.683 → 156.173 |
| Qwen3.5 C7 / 2048 | 13260.975 | 13166.229 | 0.71% | 0.61%–0.82% | 952.277 → 870.608 |

C7 的 512-token 总延迟波动明显，基线末次进程尤其慢；Linear 和 GDN 也随之变化。这个平均值不足以证明 3.75% 的稳定因果收益。2K 结果更容易解释：基线 Attention 仅占 7.18%，它降低 8.58%，单独可贡献约 0.62% 的总延迟下降，接近实测 0.71%。C7 有 24 层 GDN、8 层 FullAttention，只优化后者无法带来大幅整模型提速。

Qwen3 的 2K 基线中，Linear 7290.457 ms、Attention 5353.997 ms、其他 CPU 735.113 ms。Attention 占 40.01%，降低 8.34% 对总时间贡献约 3.34%；实测总下降 2.99%，同时 Linear 增加约 60.7 ms。后续优化应针对完整调用剩下的开销。

### 4K 补测与异常样本

第一次 4K ABBA 每进程预热一次、正式一次，原始样本如下。首个重算样本明显变慢，Linear 与其他 CPU 阶段也同时变慢；没有将它删除。

| 进程顺序 | 方法 | 完整 prefill ms |
|---|---|---:|
| A0 | baseline | 32429.845 |
| B1 | stream | 41578.301 |
| B2 | stream | 30922.683 |
| A3 | baseline | 32282.396 |

这一轮均值为 32356.121 → 36250.492 ms，实验平均**慢 12.04%**；两组配对分别为慢 28.21% 和快 4.21%。慢样本的 Linear 为 21647.290 ms，而其他正常 4K 进程约 14–15 秒。事后频率/温度快照、可用内存和进程检查没有解释原因；没有证据将其确定归因于外部负载，也不能排除新路径改变内存行为。

因此追加一轮完整 ABBA，每进程预热一次、正式两次，使用同一冻结二进制、同一上下文缓存和 token，额外每秒记录频率、温度、CPU/memory pressure 和进程 stat。没有重新调优或更改实现。

| Qwen3 / 4096 复测 | 基线 ms | 重算 ms | 延迟下降 |
|---|---:|---:|---:|
| 完整 head-free prefill | 32181.420 | 30932.888 | 3.88% |
| Attention 阶段 | 16161.143 | 14295.026 | 11.55% |
| Linear 阶段 | 14539.484 | 15158.580 | −4.26% |

四个基线样本为 32194.654、32219.574、32164.573、32146.878 ms；四个重算样本为 30909.002、30910.925、30908.724、31002.900 ms。两组配对均值分别快 4.03% 和 3.73%。698 个采样点的 CPU4/6 policy 均为 2256000 kHz、NPU 均为 1 GHz，温度最高 65.615℃。DDR 未受控，不能用这些采样排除所有运行干扰。

这轮支持“本条件下完整 prefill 约快 4%”，不支持没有性能波动的保证。Attention 省下约 1.866 秒，Linear 增加约 0.619 秒；后者的原因本轮尚未分解。完整调用的净收益才是当前可交付的性能结论。首轮和复测分别保存在 `models-bench-4k`、`models-bench-4k-repeat`，未混合聚合。

速度报告中的 `tune_cache_misses/tune_calls/strategy_fallback_calls` 均为 0；2K Qwen3 cache hits 为 86。每个上下文使用单独的缓存，先显式 `tune --batches 1,128`，再运行 ABBA。head-free 路径与包含 prompt head 的路径，其后续 token logits 在各自方法内逐字节一致。

Qwen3 输入取自已有 WikiText 测试 token 文件的前 4096 个 ID；C7 使用已有 WikiText 512-token 窗口前八个拼接的固定序列。512/2048 测试使用相同序列的前缀，不经过聊天模板。实际 ID 随报告保留，C7 的拼接输入不等同于连续原文。

## 公共 Attention 接口的完整成本

以下是合成 QKV（seed20261023，标准差 0.5）的单层测试。按 128-token chunk 从空 KV 依次执行公开的 `cache_append + attention_run`，包括增量扩维 K、mask 更新、构图/重绑定、提交/同步和 CPU 收尾。wall time 还包含测试程序的输出拼接和 guard 检查；预先准备紧凑输入片段、清空缓存不在计时内。

每种几何 3 个独立进程，每进程两边各 4 轮、交替顺序。第 0 轮保留为首次调用，后 3 轮使用保留的容量，但每轮逻辑 KV 重新清空。正式结果为三个进程内 median 的均值，每方法 9 个样本。首次调用固定 baseline 在前，只作开销参考。

| 几何 / 总 tokens | 基线 ms | 重算 ms | 延迟下降 | 首次基线 / 重算 ms |
|---|---:|---:|---:|---:|
| Qwen3 H32/KV8/D128 / 512 | 19.432 | 18.212 | 6.28% | 33.143 / 34.770 |
| Qwen3 H32/KV8/D128 / 4096 | 463.064 | 379.725 | 18.00% | 542.152 / 474.878 |
| C7 H16/KV4/D256 / 512 | 15.154 | 15.661 | −3.35% | 26.401 / 29.522 |
| C7 H16/KV4/D256 / 4096 | 314.601 | 282.018 | 10.36% | 388.473 / 361.790 |

初次扩容、构图以及短 KV 的准备成本会吃掉热态块的收益。完整增量调用保留了长 KV 收益，但不是所有形状都更快。这些数值也不能乘层数直接代替真实模型测量。

Qwen3 单层 4K 的 9 个正式样本按算术均值汇总，driver hardware 区间从 190.629 ms **增至** 247.931 ms（约 +30%），wall time 却从 463.611 ms 降至 379.459 ms。`wall − hardware` 的剩余时间从 272.982 ms 降至 131.528 ms；它包含 CPU 处理、同步、构图及其他暴露间隙，不能全称为纯 CPU 算术时间。prepare 反而从 16.333 ms 增至 39.818 ms，mask 更新约 2.149 ms。收益来自删去更昂贵的数据往返，代价是额外计算与状态准备；这比仅看 TOPS 更能说明瓶颈。此段使用样本均值，与上表的进程内 median 聚合略有不同。

## 数值质量与状态正确性

PPL 为 32 个独立的 512-token 窗口，每个窗口清空状态，评分位置 257–511，共 8160 个 target。不额外插入 BOS/chat token。两边使用相同 token、模型和量化路径。

| 模型 | 基线 PPL | 重算 PPL | 相对变化 | 配对窗口近似 95% 区间 |
|---|---:|---:|---:|---:|
| Qwen3-4B | 14.958254672 | 14.994868512 | +0.2448% | −1.0297% ～ +1.5357% |
| Qwen3.5 C7 | 8.574409061 | 8.571474311 | −0.0342% | −0.3842% ～ +0.3169% |

区间由 32 个窗口的配对平均 NLL 差，使用 `mean ± 1.96*stdev/sqrt(32)` 后指数变换得到；它只描述这个样本，且没有校正同一语料窗口之间的相关性。Qwen3 有 20/32 个窗口 NLL 增加，区间还不足以保证退化低于 1%。这是实际模型质量测量，但不是长上下文质量或其他数据集的保证。

Qwen3 PPL 两边均有相同的输出头 M128 策略缺项：`tune_cache_misses=1`、`strategy_fallbacks=1`、`strategy_fallback_calls=64`、`tune_calls=0`，使用相同 builtin 评分策略。C7 PPL 的这些计数均为 0。PPL 不用于速度比较，也不能混淆 Linear 调优 fallback 与 Attention 选择原路径的计数。

另用同一速度输入执行长 prefill，导出最后一个位置的全部 FP32 logits：

| 模型 / 长度 | logits relative L2 | 去均值后 relative L2 | max abs | KL(base∥stream) | top-1 / top-10 交集 |
|---|---:|---:|---:|---:|---:|
| Qwen3 / 4096 | 14.7015% | 15.5671% | 1.633362 | 0.00222663 | 相同 / 9 |
| C7 / 2048 | 5.9312% | 7.7458% | 0.995682 | 0.00579815 | 相同 / 10 |

每个模型这里只取一个末尾位置，不能代替长上下文 PPL。虽然 top-1 一致，完整 logits 已发生可见变化，且去均值后差异仍在；单层 Attention 约 0.03% 的误差不能推导为整模型无损。速度和这些质量测量均是同一冻结二进制完成，长 logits 测试的调优缺项、调优调用及策略 fallback 均为 0。

已通过的公共接口检查：

- 逐块增长到 4096；同方法多轮输出逐字节一致。
- 截断末尾 17 个 token 后更换 K/V；多层缓存 A/B/A 切换后，A 的输出恢复逐字节一致。
- 511/513 非整块尾部，以及 MHA H16/KV16/D64。
- 无 mask 的完整 KV 读取；FP16 输出等于同方法 FP32 输出逐元素转换。
- 大 V 振幅触发原 CPU 路径，与基线逐字节一致。

正式合成增长测试 relative L2 最大 0.0347%；这只比较重算与 CPU 优化后的基线。早期 growth4096 检查可能与编译重叠，其时间未作为性能数据。没有宣称覆盖任意 FP16 极值、非有限值、Boolean mask、MLA 或所有 public API 组合。

## 实现与额外开销

数据路径延续上一轮：第一遍 QK 加 causal mask，再用原生 ReduceMax 取得每行最大值；CPU 只读最大值，构造 `[Q_scaled,-max]`；第二遍与 `[K,1]` 相乘，融合 LUT，必要的尾块精确 mask 后执行 PV，沿用原来的 FP32 partial 累加和 carrier 修正。

第一遍仍向 DDR 写完整 score，ReduceMax 再读取它。消除的是整张 score 的 CPU 同步、扫描和减最大值；设备侧仍有 score 流量，并增加一次 QK。额外 MAC 不能计作有用模型 FLOPs 来美化利用率。

本轮新增的状态维护：

- 扩维 K 只补充新 append 的位置，容量增长时才迁移旧数据；load/truncate 分别清空或截断有效前缀。
- 每个 workspace 只保留一份可共享的 mask，并仅更新发生变化的 key 区间。
- 几何及地址稳定时复用任务链；不同缓存 identity/generation/tail 变化时修改 K/V 地址绑定。
- 保留原路径选择条件。当前重算仅用于 LUT 可用、query rows≥16、物理 M≤128、span≤4096、kv_tile512、K/V dim 相同的非 MLA、非 Boolean 路径。

4096-token 增长两轮的检查中，`key_rows=8241 = 2*4096 + 17 + 32`，分别对应两次增长、截断重写和缓存 B。返回缓存 A 没有重打包全部 K，说明增量路径确实被执行。该检查共 259 次重算调用、67 次构图、67 次绑定。

| 上下文 / 模型 | 额外扩维 K 的逻辑存储 |
|---|---:|
| Qwen3 / 2048，36 层×8 KV heads×160 channels×FP16 | 180 MiB |
| Qwen3 / 4096 | 360 MiB |
| C7 / 2048，8 层×4 KV heads×288 channels×FP16 | 36 MiB |

还会增加 Q、max、mask 和任务描述符等 workspace；表中不是总内存增量。2K Qwen3 进程 peak RSS 从约 3047.9 MiB 增至 3230.8 MiB（约 +183 MiB）。4K 首轮与复测分别增加约 363/365 MiB。C7 的峰值容易被模型加载阶段覆盖，两次实验进程约 3764.4/3699.3 MiB，不能用它反推实际新增 GEM 内存。

当前 runtime 的 Attention 内存预算/query **没有计入扩维 K**。生产化前必须补齐预算，否则上下文上限和 IOVA 域分配预估不可靠。若两遍 QK 共用扩维 K，可删掉原 K 副本；但第一遍也会多做对齐通道的 MAC，需要另做完整调用对照。

Qwen3 两遍共用 K160 时，4K 全模型 K 存储可由当前的 288+360=648 MiB 降至 360 MiB，较原库的净增加将变成 72 MiB。这个方案尚未实现；它让第一遍 QK 也从 D128 计算到 K160，不能只根据少分配内存就认定更快。

## 代码与复现

本轮代码均位于算子库实验目录：

- [stream.cpp](../native/rk_npu_matmul/tests/tools/attention_pipeline/stream.cpp)：增量 K、mask、任务链与重绑定。
- [prepare_stream.py](../native/rk_npu_matmul/tests/tools/attention_pipeline/prepare_stream.py)：在独立快照注入实现和共用 CPU 优化。
- [stream_probe.cpp](../native/rk_npu_matmul/tests/tools/attention_pipeline/stream_probe.cpp)、[run_stream_probe.py](../native/rk_npu_matmul/tests/tools/attention_pipeline/run_stream_probe.py)：公共接口正确性与完整 prefill。
- [prepare_stream_app.py](../native/rk_npu_matmul/tests/tools/attention_pipeline/prepare_stream_app.py)、[model_suite.py](../native/rk_npu_matmul/tests/tools/attention_pipeline/model_suite.py)：固定真实 token 的模型快照和调优、ABBA、PPL。
- [long_eval.py](../native/rk_npu_matmul/tests/tools/attention_pipeline/long_eval.py)、[summarize_stream.py](../native/rk_npu_matmul/tests/tools/attention_pipeline/summarize_stream.py)：长上下文 logits 比较与汇总。

准备快照时 native HEAD 为 `6df8ec899f5e8b7757b73911da25baff9178ef9e`。快照同时包含冻结时工作区已有的 W4 修改，两边完全共用；不能只凭该 revision 重现最终二进制。归档时仓库已推进，生成差分所用 native 基底为 `a3d424bec63c46ee8e466694c74808c5058af899`，app 基底为 `b6ffd4985ccb69b6685996cd2162d756be975bd7`；应用归档 patch 时应以 `snapshot-bases.json` 为准。测量期间冻结的实现和可执行文件没有随工作区推进而替换。实际编译 native 使用 GCC 12.2、O2、`-march=armv8.2-a+fp16`；Rust 为 release 构建。快照差异、源码/输入/二进制哈希随证据保留。

完整板端快照：`<BOARD_HOME>/mnt/3090x4/tmp/attention-stream-20261022`；主机映射为 `<HOST_HOME>/mnt/3090x4/tmp/attention-stream-20261022`。在已构建快照中运行：

```sh
python3 model_suite.py --root . --stage ppl
python3 model_suite.py --root . --stage bench
python3 model_suite.py --root . --stage bench --models qwen3 \
  --context 4096 --prompts 4096 --repetitions 1 --tag 4k
python3 long_eval.py --root .
ATTN_BENCH_TELEMETRY=1 python3 model_suite.py --root . --stage bench \
  --models qwen3 --context 4096 --prompts 4096 --repetitions 2 \
  --tag 4k-repeat --prepared-cache
python3 summarize_stream.py --root . --output summary.json
```

原始正式报告、样本、token IDs、状态计数、时钟与恢复记录位于 [evidence/attention_stream_20261023](evidence/attention_stream_20261023/)。前几次缓存不匹配的 smoke 和编译重叠检查没有进入正式速度表。4K 补测第一次被共享锁拒绝，未提交 NPU，记录单独保留；随后使用 `--wait-lock` 串行运行。

归档还包括各上下文的调优缓存，以及 `native-snapshot.patch`、`app-snapshot.patch`、`snapshot-bases.json` 和 `sha256.json`。全部 36 份正式/正确性/调优/恢复时钟报告均 `exit_code=0, restore_errors=[]`；最终原库 canary 的重复输出、guard 和独立 FP32 参考行检查通过。板端 RTC 未同步，原始报告中的 Unix 时间对应板端旧日期；实验目录和本文日期按主机记录，不能将板端时间戳当作真实日历时间。

已在临时目录从记录的 Git 基底应用归档 patch，重建 native 103 个文件、app 105 个文件；文件集合和逐文件 SHA256 全部与冻结快照一致，见 [verification.json](evidence/attention_stream_20261023/verification.json)。

## 后续取舍

1. 优先以长 prefill 为候选继续评估，短验证/decode 保留原路径。C7 的整模型瓶颈仍是 Linear/GDN，继续只磨 FullAttention 的收益上限较低。
2. 生产化前先解决扩维 K 的重复存储及内存预算，并扩大质量验证；当前实现不适合直接默认开启。
3. 下一条更值得优先对照的是：保留第一遍已舍入的 FP16 QK score，在设备端做 row-max 广播、subtract 与 LUT，再接 PV。相邻 lowlevel 的 [native Softmax 记录](../../lowlevel/docs/fp16_native_softmax_2026-08-27.md) 已验证相关基本阶段；这里只提取稳定化与 exp，不恢复 ReduceSum/Div 全套。它可能避免重算引入的舍入位置变化及扩维 K，但广播仍会引入全 score 流量，融合后的数值和速度均需重新验证。先检查完整 score/概率与当前路径的差异，再做完整 append+run 和模型对照；若额外 DPU 扫描抵消收益就停止，不继续堆任务参数。
4. QK→ReduceMax flying 仍是另一个设备侧流量候选，但上一轮映射未通过数值验证，不能视作现成方案。额外 MAC 的忙碌率不等于有用 Attention 吞吐。
