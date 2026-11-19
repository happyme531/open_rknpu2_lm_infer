# C7 双 ubatch 重叠调度速度复测

主板，CPU4–7 2.4 GHz、NPU1 GHz、DDR2.112 GHz，4线程、NPU mask7、ubatch128。
本轮启用 `--prefill-ubatches 2`，每项预热一次、计时3轮；对照为上一轮同主板、同C7、同输入的单ubatch记录。未运行PPL、logits对比或其他精度测试。

| 模型 | 测试 | 单 ubatch token/s | 双 ubatch token/s | 变化 |
|---|---|---:|---:|---:|
| Qwen3-4B C7 | pp512 | 191.053 | 222.079 ± 0.569 | +16.24% |
| Qwen3-4B C7 | pp1024 | 182.039 | 211.770 ± 0.123 | +16.33% |
| Qwen3-4B C7 | pp4096 | 141.488 | 163.069 ± 0.008 | +15.25% |
| Qwen3-4B C7 | decode128 @512 | 10.726 | 10.667 ± 0.051 | -0.55% |
| Qwen3-4B C7 | decode128 @1024 | 10.311 | 10.284 ± 0.002 | -0.27% |
| Qwen3-4B C7 | decode128 @3800 | 8.544 | 8.525 ± 0.000 | -0.22% |
| Qwen3.5-4B C7 MTP1 | pp512 | 159.266 | 178.532 ± 0.409 | +12.10% |
| Qwen3.5-4B C7 MTP1 | decode128 @512 | 11.453 | 11.471 ± 0.002 | +0.16% |
| Qwen3.5-4B C7 MTP1 | pp1024 | 159.215 | 179.157 ± 0.230 | +12.53% |
| Qwen3.5-4B C7 MTP1 | decode128 @1024 | 11.409 | 11.436 ± 0.002 | +0.24% |
| Qwen3.5-4B C7 MTP1 | pp3800 | 153.458 | 171.250 ± 0.242 | +11.59% |
| Qwen3.5-4B C7 MTP1 | decode128 @3800 | 10.661 | 10.674 ± 0.006 | +0.13% |

Qwen3使用 `selection/models/C7-Rotate-QQQ`、context8192、固定MT19937输入，prefill不含输出头，decode为128次完整模型调用。
Qwen3.5使用 `transfer/qwen35/models/C7`、context4096、同一真实文本提示、greedy MTP1。prefill含输出头；128个输出token中，首token来自prefill，decode计时覆盖剩余127个token，包含draft/verify/replay和采样。

两模型运行统计均确认 `prefill_ubatches=2`、`dual_prefill_groups>0`。在线调优、计数器记录的fallback和权重重载均为0；Qwen3.5的MTP浮点辅助投影保留上一轮已说明的默认策略及cache miss。
此为前后两轮记录比较，没有重新运行单ubatch的ABBA对照。启用双ubatch也会改变工作区与IOMMU domain布局，结果衡量整个选项的实际效果。

峰值RSS为进程高水位，包含载入：

| 进程 | 单ubatch GiB | 双ubatch GiB |
|---|---:|---:|
| qwen3-prefill | 3.885 | 5.189 |
| qwen3-decode | 4.143 | 5.523 |
| qwen35 | 3.815 | 4.594 |

- [汇总、原始样本与计数器](evidence/dual_ubatch_c7_speed_20261025/summary.json)
- [固定频率及恢复记录](evidence/dual_ubatch_c7_speed_20261025/clocks.json)
- [精确命令与二进制身份](evidence/dual_ubatch_c7_speed_20261025/completed.json)
- [上一轮单ubatch报告](README_REBENCH_20261025.md)
