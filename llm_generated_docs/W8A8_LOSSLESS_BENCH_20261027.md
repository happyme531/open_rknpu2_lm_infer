# W8A8 无损压缩：内存和整模型速度

2026-10-27，`zt-firefly` RK3588，压缩支持接入后的首次 release 测量，尚未启用自动 CPU 堆回收。当前自动回收实现及结果见[堆回收报告](W8A8_COMPRESSION_HEAP_CLEANUP_20261027.md)。

`Qwen3-4B-W8A8-Baseline` 的 145 个 W8A8Linear 全部参与。压缩使 NPU GEM 分配减少 **459.82 MiB**，但进程稳定 RSS 增加 **115.47 MiB**；稳定推理吞吐基本持平，没有测到加速。

## 结果

| 指标 | 原始 | `--compress-w8a8` | 变化 |
|---|---:|---:|---:|
| 稳态进程 RSS | 4168.02 MiB | 4283.49 MiB | +115.47 MiB，+2.77% |
| NPU GEM 总分配 | 4034.50 MiB | 3574.67 MiB | −459.82 MiB，−11.40% |
| CPU 匿名常驻内存 | 96.93 MiB | 672.13 MiB | +575.20 MiB |
| 进程峰值 RSS | 4716.12 MiB | 4894.87 MiB | +178.76 MiB，+3.79% |
| pp512 耗时 | 3997.93 ms | 4007.87 ms | +0.25% |
| pp512 吞吐 | 128.066 token/s | 127.749 token/s | −0.25% |
| decode32 @512 耗时 | 6175.36 ms | 6204.72 ms | +0.48% |
| decode 吞吐 | 5.182 token/s | 5.157 token/s | −0.47% |
| 模型加载 | 15.29 s | 56.08 s | +40.79 s，3.67 倍 |
| 正式计时前全部启动准备 | 29.01 s | 146.45 s | +117.44 s |

最后一行包含进程启动、模型加载、head-free 一致性检查和一轮预热，不属于稳定推理时间。

NPU GEM 已计入 RSS，二者不能相加。GEM 数字来自运行进程 `/proc/PID/smaps` 中 `/dev/dri/card1` 映射的 `Size` 总和，包含权重、workspace 和 KV；对应 resident 大小也保存在摘要。各进程的 workspace 和 KV payload 相同，因此 GEM 差值反映权重存储变化。没有使用仍返回 raw 大小的 `weight_payload_bytes` 估算压缩收益。

两次 raw 的最终 RSS 分别为 4168.55、4167.49 MiB；两次压缩为 4283.61、4283.36 MiB。额外 CPU 匿名常驻内存抵消了 NPU 存储节省；堆诊断表明其中的大头是 glibc 保留的空闲堆页。

## 匿名内存来源

压缩准备与首次执行布局转换会创建大型临时 CPU 数组。`I8KnWeights::build_compressed_layout` 使用 `packed`、`part`、`DcompEncoded::data` 和聚合 `payload`；它们在函数结束时释放。glibc 2.31 的 malloc 会保留已释放的空闲堆空间，页面仍可能计入匿名 RSS。已有快照中，额外内存集中在多个约50–64 MiB的匿名堆段。

同模型另做8-token prefill和2-token continuation、context128的诊断运行：在第一次 logits 写出时读取 `mallinfo` / `malloc_info`，执行一次 `malloc_trim(0)`，随后继续两步 decode。诊断通过 LD_PRELOAD 加载，生产二进制保持原样。

| 同一诊断进程 | trim 前 | trim 后 |
|---|---:|---:|
| malloc 在用块 `uordblks` | 45.92 MiB | 45.92 MiB |
| malloc 空闲块 `fordblks` | 624.18 MiB | 579.56 MiB |
| 匿名 RSS | 645.67 MiB | 328.66 MiB |
| 总 RSS | 4089.67 MiB | 3772.66 MiB |

一次 trim 归还 **317.01 MiB** 实际物理页，NPU/共享映射的 resident 大小保持不变。后续两步 decode 后总 RSS 为3772.86 MiB、匿名 RSS为328.72 MiB，没有重新涨回；全部455,808个 FP32 logits 与原始未压缩参考逐位一致且有限。时钟恢复通过。

这确认了空闲堆驻留是匿名内存膨胀的主要来源。`fordblks` 表示分配器内的空闲空间，不等于可以一次归还的 resident 大小；本次 trim 仅回收一部分。诊断运行的 context 和序列长度不同，其绝对 RSS 不直接与上面的 pp512 ABBA 表相减。

- [堆诊断摘要与 logits 校验](evidence/w8a8_lossless_20261027/heap_probe/summary.json)
- [诊断代码](evidence/w8a8_lossless_20261027/heap_probe/probe.c) / [复核代码](evidence/w8a8_lossless_20261027/heap_probe/verify.py)
- 同目录保存 trim 前后和 decode 后的 `malloc_info` XML、`mallinfo` 及 `smaps_rollup` 原始记录。

## 测量口径

- CPU4–7 固定 2.4 GHz、NPU 1 GHz、DDR 2.112 GHz；4 线程、CPU mask `0xf0`、NPU mask7。
- 同一模型目录 `/home/firefly/mnt/3090x4/llm/w4a8_quant/Qwen3-4B-W8A8-Baseline`，context1024、ubatch128、单 prefill ubatch、`--no-tune`，使用相同内置策略。没有宣称已调优的峰值性能。
- ABBA 顺序：raw → compressed → compressed → raw，串行、独占共享 NPU 锁。
- 每进程使用 `bench -p 0 -n 0 --n-prompt-gen 512,32 -r 6`，每项预热一次。内存快照在第1轮测量后读取；排除第1、2轮，防止快照读取干扰，保留第3–6轮。每种模式汇总8个计时样本并取中位数；内存和加载时间取两个进程均值。
- pp512 不计算 LM head；decode 在这512个 token 的 KV 后逐个消费32个 token，包含完整模型和 LM head，不含采样、分词、加载和准备。输入为程序的 `MT19937(42) % vocab_size`。
- 四进程均退出0，head-free continuation logits 检查均通过；时钟上下限和 governor 已恢复，`restore_errors=[]`。稳态温度快照最高50.846°C。

prefill 的进程间波动大于0.25%的汇总差异，不能将这点差异解读为稳定的性能变化。decode 的汇总吞吐下降约0.47%。该结果针对这个全 W8A8 checkpoint 和策略，不代表其他权重分布、分块或模型的压缩率和速度。

## 原始记录

- [汇总、逐进程内存和全部保留样本](evidence/w8a8_lossless_20261027/summary.json)
- [精确命令、运行顺序和进程耗时](evidence/w8a8_lossless_20261027/runs.json)
- [频率、温度和恢复验证](evidence/w8a8_lossless_20261027/clocks.json)
- [raw 1](evidence/w8a8_lossless_20261027/1-raw.json) / [compressed 1](evidence/w8a8_lossless_20261027/2-compressed.json) / [compressed 2](evidence/w8a8_lossless_20261027/3-compressed.json) / [raw 2](evidence/w8a8_lossless_20261027/4-raw.json)
- [运行脚本](evidence/w8a8_lossless_20261027/run.py) / [汇总脚本](evidence/w8a8_lossless_20261027/summarize.py)

同目录保留各进程的 `.smaps`、`.smaps_rollup`、`.status` 和原始日志。
