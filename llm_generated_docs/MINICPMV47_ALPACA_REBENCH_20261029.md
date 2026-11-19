# MiniCPM V4.7 35B A3B Alpaca 性能

2026-10-29。Alpaca 两条日常指令上，全驻留 DSpark 3 草稿 decode 为 **10.25 / 10.10 tok/s**，默认 20 槽 SSD 流式为 **6.15 / 6.00 tok/s**。SSD 开启 0.10 专家跳过后为 **7.72 / 7.73 tok/s**；同样开启跳过的 7 草稿为 **6.05 / 6.18 tok/s**，比 3 草稿慢 **21.6% / 20.1%**。

## 四组结果

每格 P / D 分别为 prefill / decode，单位 tok/s；每项一次独立进程。所有八项均生成 129 token，其中 128 token 计入 decode。

| 配置 | 问答 23 token，P / D | 摘要 150 token，P / D | 峰值 RSS，问答 / 摘要 GiB |
| --- | ---: | ---: | ---: |
| 全驻留，DSpark 3 草稿 | 29.85 / 10.25 | 51.06 / 10.10 | 19.14 / 19.64 |
| SSD 流式，DSpark 3 草稿 | 8.32 / 6.15 | 22.27 / 6.00 | 4.68 / 5.19 |
| SSD 流式，DSpark 3 草稿，跳过 0.10 | 12.74 / 7.72 | 29.24 / 7.73 | 4.70 / 5.14 |
| SSD 流式，DSpark 7 草稿，跳过 0.10 | 12.55 / 6.05 | 27.17 / 6.18 | 4.73 / 5.18 |

流式统一使用默认 8% 缓存，即每层 20/256 槽，LES 和 O_DIRECT 开启，2 个 reader、1 个导入线程。外部采样中所有流式进程 VmSwap 均为 0，主板未启用 swap。峰值 RSS 包含模型加载、草稿、KV 和工作区。

DS3 未跳过时，两条指令的草稿接受率为 **60.74% / 60.29%**，每轮输出均为 **2.783 token**。开启跳过后为 **53.74% / 55.24%**、每轮 **2.612 token**；DS7 跳过组为 **28.91% / 29.82%**、每轮 **2.977 token**。DS7 的每轮产出略增，整体 decode 仍更慢。

0.10 跳过相对同一条输入的无跳过 SSD DS3 提升 25.6% / 28.9%，属于改变模型计算的有损近似，本轮没有重测质量。两条输入上，全驻留与无跳过流式的 DS3 生成文本一致；跳过组 DS3 与 DS7 文本一致；开启跳过后的文本与无跳过不同。

## 输入和测量口径

源文件为 [DeepSpec 官方 Alpaca 评测语料](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/eval_datasets/alpaca.jsonl)，共 52,002 条。以下两条在测速前固定，保留完整原题，没有根据结果筛选、截断或填充：

- 零起始序号 272：解释个人电脑与服务器的区别，含 chat 模板共 23 token。
- 零起始序号 3584：总结 AI 应用文章的要点，含原文和 chat 模板共 150 token。

Alpaca 原题偏短，这轮保持自然长度。上一轮 Arena-Hard 的两个代码输入为 498 / 2110 token，SSD DS3 为 4.92 / 5.74 tok/s；本轮日常指令为 6.15 / 6.00 tok/s。任务内容和输入长度同时变化，这些单样本结果不能作为题目难度的因果对照或整套数据集均值。短输入的 prefill tok/s 也受固定开销摊销影响。

两轮使用相同固定二进制、模型、专家存储和板端设置，本轮没有重新编译或修改推理代码。模型为无 SQ MiniCPM-V-4.7-35B-A3B-QQQ-W4A8-Mixed-CT；路由专家 W4A8，共享 FFN/输出头 W8A8，默认 CPU GDN，DSpark 5 层草稿运行时 FP16。

ctx4096、ubatch128、单 prefill lane、no-tune、W8 压缩关闭，thinking 关闭；greedy、temp0、top-k0、top-p1、seed42。每次从空请求状态开始，cached_tokens 全为 0。自然 EOS，最多129输出 token；本轮八次均到达上限。

prefill 计完整 prompt forward 和最终输出头；decode 按实际输出128 token / 完整 decode_ms 计算，包含草稿、验证、拒绝、commit/replay、采样和流式输出。加载、预分配与统计写盘排除；正式运行关闭 timeline、使用默认分配器。

## 环境与验证

全驻留使用 32GB Firefly ITX-3588J、kernel 6.1.118-zt；流式使用 8GB H88K、kernel 6.1.115-vendor-rk35xx、GM7 本地 NVMe。均为 RKNPU 0.9.8，CPU4–7 固定2.4GHz、NPU三核1GHz、DDR2112MHz，由共享锁 runner 管理。八次成功退出，八份 clock report 的频率上下限均精确恢复；四组生成文本对照通过。

沿用 [Arena-Hard 测试](MINICPMV47_DSPARK_REBENCH_20261029.md) 的两个固定二进制：全驻留使用含驻留 MoE 域预算修复的 `resident-infer`，流式使用 `current-infer`，native 均为 `f872a946e4ce8dfba51317fc6aad68bfff311f46`。该修复仅改变驻留加载预算；两轮软件条件一致。全驻留与流式跨板，不构成同板存储替换的因果 A/B。

证据目录：[evidence/minicpmv47_alpaca_20261029/](evidence/minicpmv47_alpaca_20261029/)。包含完整源数据、输入与 token IDs、固定参数、脚本、原始输出/统计、时钟、遥测、`summary.json` 与 `output-comparison.json`。板端流式文件另存 `/mnt/nvme/minicpmv47/results/alpaca_20261029/`。可执行 `python3 llm_generated_docs/evidence/minicpmv47_alpaca_20261029/summarize.py` 汇总已有文件，不会重新测速。
