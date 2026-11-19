# Qwen3 / Qwen3.5-4B 主开发板性能与 PPL 重测

日期：2026-10-25。主板，RKNPU driver 0.9.8。

运行时 `05dc87dadec3fcb9be77a7b2e72154e6cc99a277`，算子库 `811b2a37579499d746fd3bdbc2abdfd5de46ac9d`。GCC 10.2.1、Rust 1.95.0 release 构建。

CPU4–7 固定 2.4 GHz、NPU 1 GHz、DDR 2.112 GHz，CPU 4 线程 / mask `0xf0`、NPU mask7、Attention tile512。
性能使用 ubatch128、单 prefill ubatch；每项预热一次、计时三轮，保留全部样本。视觉编码沿用历史协议：不同图片预热一次、正式测量一次，不命中特征缓存。

## 性能

| 模型 | 测试 | token/s（均值 ± 样本标准差） | 相对 SDK 历史基线 |
|---|---|---:|---:|
| Qwen3-4B | pp512 | 191.053 ± 0.154 | +81.89% |
| Qwen3-4B | pp1024 | 182.039 ± 0.020 | +92.39% |
| Qwen3-4B | pp4096 | 141.488 ± 0.047 | +182.23% |
| Qwen3-4B | decode128 @512 | 10.726 ± 0.001 | +85.82% |
| Qwen3-4B | decode128 @1024 | 10.311 ± 0.006 | +101.20% |
| Qwen3-4B | decode128 @3800 | 8.544 ± 0.002 | +269.86% |
| Qwen3.5-4B MTP1 | pp512 | 159.266 ± 0.330 | +172.96% |
| Qwen3.5-4B MTP1 | decode128 @512 | 11.453 ± 0.001 | +106.81% |
| Qwen3.5-4B MTP1 | pp1024 | 159.215 ± 0.172 | +175.25% |
| Qwen3.5-4B MTP1 | decode128 @1024 | 11.409 ± 0.003 | +110.66% |
| Qwen3.5-4B MTP1 | pp3800 | 153.458 ± 0.126 | +184.03% |
| Qwen3.5-4B MTP1 | decode128 @3800 | 10.661 ± 0.002 | +133.53% |

448×448 视觉编码：**862.25 ms**；正式图片编码一次、feature cache hit 为0。

Qwen3 的性能与 PPL 统一使用 `selection/models/C7-Rotate-QQQ`。性能 context8192；输入为 `MT19937(42) % vocab_size`。prefill 不计算末行输出头；decode 为完整的128次模型调用，包含输出头，不含采样，历史 KV 在计时前建立。无输出头路径的 continuation logits 一致性检查通过。

Qwen3.5 性能 checkpoint 为 `transfer/qwen35/models/C7`，context4096；使用原512/1024/3800-token真实文本提示、greedy、seed42、MTP1。prefill 包含末行输出头；请求输出128个 token，首token由prefill产生，decode计时覆盖其余127个 token，包含draft/verify/replay及采样。每组重复样本的输出IDs相同。

两个模型的性能与PPL均使用各自同一C7 checkpoint。SDK基线保留原记录，未在本轮重跑。大输出头与MTP浮点辅助投影在独立阶段准备调优缓存，正式测量加载缓存。

## Perplexity

| 模型 | 当前 PPL | SDK 历史 W8A8 | 相对 SDK | 目标数 |
|---|---:|---:|---:|---:|
| Qwen3-4B C7-Rotate-QQQ | 12.591253456 | 12.8709 | -2.1727% | 8160 |
| Qwen3.5-4B C7 | 8.574409061 | 8.5197 | +0.6421% | 8160 |

沿用各模型 tokenizer 对 WikiText-2 raw test 得到的前32个独立512-token窗口；评分零起始target位置257–511，每窗重置KV/GDN/卷积状态，不加BOS/EOS/chat模板，关闭MTP。Qwen3为 `selection/models/C7-Rotate-QQQ`、ubatch64；Qwen3.5为C7、ubatch128；context均为512。

两个模型各32窗均完成，逐token NLL全部有限。独立对每模型8160项NLL做FP64求和并计算 `exp(sum_nll / 8160)`，与运行时报告一致。这是32窗前缀PPL，不代表全量WikiText-2评分。

## 调优覆盖

Qwen3.5 MTP 的 FP16 K2560/N9216 gate/up 投影使用运行时内置策略，Ntile3072；该形状的通用搜索仍含会触发硬件超时的候选。正式测量使用已验证的内置策略，该分支产生的cache miss保留在统计中，不计入现有 `strategy_fallback_calls`。其余MTP配置和context2048视觉请求所需的量化配置分别预调优。

## 验证与原始记录

Qwen3性能与两个模型PPL的 `tune_cache_misses`、`tune_calls`、`strategy_fallback_calls`、`weight_reload_calls` 均为0。Qwen3.5文本/视觉请求仅MTP浮点domain12有上述已知cache miss；在线调优、计数器记录的fallback和权重重载均为0，其余domain无miss。正式矩阵的11个板端步骤退出0，频率上下限和governor恢复，`restore_errors=[]`。

Qwen3 pp4096进程峰值RSS为 3.885 GiB，包含进程启动和模型加载高水位。

- [完整汇总与检查结果](evidence/readme_rebench_20261025/summary.json)
- [测试配置、模型配置和输入身份](evidence/readme_rebench_20261025/protocol.json)
- [构建与二进制身份](evidence/readme_rebench_20261025/build.json)
- [步骤、精确命令与退出状态](evidence/readme_rebench_20261025/completed.json)
- [Qwen3 prefill](evidence/readme_rebench_20261025/qwen3-prefill.json) / [decode](evidence/readme_rebench_20261025/qwen3-decode.json)
- [Qwen3.5 MTP1文本](evidence/readme_rebench_20261025/qwen35-text/native.json) / [视觉请求](evidence/readme_rebench_20261025/qwen35-vision/measured.json)
- [Qwen3 PPL与逐token NLL](evidence/readme_rebench_20261025/qwen3-ppl.json) / [Qwen3.5 PPL与逐token NLL](evidence/readme_rebench_20261025/qwen35-ppl.json)
- [板端运行脚本](evidence/readme_rebench_20261025/run.py) / [结果核验脚本](evidence/readme_rebench_20261025/summarize.py)

## 调优进度显示

视觉和浮点MM搜索现在在stderr显示候选进度、百分比、累计耗时与分阶段ETA。FP16分别显示coarse/refine阶段，BF16显示单阶段搜索；阶段完成3个候选后开始估算ETA。终端常规刷新间隔250ms，重定向日志为5秒；阶段开始、首次ETA和阶段结束始终输出。缓存命中不进入搜索显示。

板端release构建通过。真实FP16/BF16小矩阵调优、TTY原地刷新、非TTY日志、ETA有限值与阶段100%完成、JSON stdout解析、缓存命中不搜索均通过；时钟恢复成功。该显示改动在模型性能/PPL测量完成后实现。

- [进度显示验证结果](evidence/readme_rebench_20261025/progress-ui/validation.json)
- [实际调优进度日志](evidence/readme_rebench_20261025/progress-ui/log.stderr)
