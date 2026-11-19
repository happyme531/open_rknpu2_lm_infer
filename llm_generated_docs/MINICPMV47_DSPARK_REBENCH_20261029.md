# MiniCPM V4.7 35B A3B DSpark 板端性能

2026-10-29，当前 main 工作区。Arena-Hard-v2 两条完整开发请求上，全驻留 DSpark 3 草稿的 decode 为 **8.41 / 9.31 tok/s**。默认 20 槽 SSD 流式为 **4.92 / 5.74 tok/s**；开启 0.10 专家跳过后为 **7.89 / 8.03 tok/s**。同样开启跳过时，7 草稿比 3 草稿慢 **13.1% / 15.7%**。

## 四组结果

每格 P / D 分别表示 prefill / decode，单位 tok/s。每项一次独立进程，均完成 129 个输出 token，其中 128 个计入 decode。

| 配置 | 短输入 498 token，P / D | 长输入 2110 token，P / D | 峰值 RSS，短 / 长 GiB |
| --- | ---: | ---: | ---: |
| 全驻留，DSpark 3 草稿 | 59.00 / 8.41 | 63.06 / 9.31 | 19.67 / 20.12 |
| SSD 流式，DSpark 3 草稿 | 33.98 / 4.92 | 44.44 / 5.74 | 5.46 / 5.88 |
| SSD 流式，DSpark 3 草稿，跳过 0.10 | 41.19 / 7.89 | 54.78 / 8.03 | 5.34 / 5.69 |
| SSD 流式，DSpark 7 草稿，跳过 0.10 | 41.98 / 6.85 | 54.76 / 6.76 | 5.57 / 5.94 |

流式均为默认 `--expert-cache-percent 8`，即每层 20/256 槽，LES、O_DIRECT 开启，2 个读线程、1 个后台导入线程。峰值 RSS 包含加载、目标模型、草稿、KV 和工作区。流式 DS3 跳过组的长输入出现最大 17.50 MiB 进程 VmSwap，其余流式样本为 0；主板未启用 swap。

不跳过时，DS3 短/长输入的草稿接受率为 44.79% / 54.86%，每轮输出 2.327 / 2.612 token。跳过后，DS3 为 62.41% / 61.65%、2.844 / 2.783 token；DS7 为 36.80% / 35.86%、3.556 / 3.368 token。DS7 每轮产出增加，但整段 decode 更慢。

专家跳过是有损近似：删除归一化 top-8 中权重严格小于 0.10 的分配，最高分专家保留，剩余权重不重新归一化。它同时影响 prefill 和 decode，本轮没有重新评估质量。同一输入下，全驻留与无跳过流式的 DS3 生成文本一致；跳过组 DS3 与 DS7 文本一致；开启跳过后的文本与未跳过时不同。这些检查覆盖本轮生成文本，不代表完整 logits 或任务质量一致。

## 语料和计时

输入来自 [DeepSpec 官方 Arena-Hard-v2 评测文件](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/eval_datasets/arena-hard-v2.jsonl)，按完整指令长度选择，没有根据测速结果筛选、截断或填充：

- 短输入：零起始序号 171，Unity C# 集合序列化与 `IList<T>` 编译问题，套模板后 498 token。
- 长输入：零起始序号 104，修改 MarkdownBlock 的引号文本高亮，套模板后 2110 token。

使用模型 chat 模板，thinking 关闭，greedy、temperature=0、top-k=0、top-p=1、seed=42。上下文容量统一为 4096，物理 ubatch128，单 prefill lane、no-tune，W8 压缩关闭。两个输入是不同任务，长度档间差异同时包含任务内容变化。

模型为无 SQ MiniCPM-V-4.7-35B-A3B-QQQ-W4A8-Mixed-CT，路由专家 gate/up/down W4A8，共享 FFN 与输出头 W8A8，GDN 使用默认 CPU 路径；DSpark 为对应 5 层草稿，运行时 FP16。

prefill 包含完整 prompt forward 与最终输出头；decode 按实际 128 个输出 token / `decode_ms` 计算，包含草稿、目标验证、拒绝、commit/replay、采样和流式输出。权重加载、预分配和统计写盘均不计入 P/D。各进程从空请求状态开始，实际 cached_tokens 全为 0。正式运行无 timeline、使用默认分配器。

[MiniCPM DSpark 模型卡](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark#evaluation)只披露文本 Math、Code、General 三类，未给出具体文本子集名；其 15 草稿、自然 EOS、8192 输出上限与本轮不同。本轮是从 DSpark 官方公开评测语料取样的板端速度测试，不复现模型卡的接受长度，也不代表整套 Arena-Hard-v2 均值。

## 板端和源码

全驻留使用 32GB Firefly ITX-3588J，kernel 6.1.118-zt；SSD 流式使用 8GB HINLINK H88K，kernel 6.1.115-vendor-rk35xx，GM7 本地 NVMe，ext4 `/mnt/nvme`。两板均为 RKNPU 0.9.8，CPU4–7 固定 2.4GHz、NPU mask7 固定 1GHz、DDR 固定 2112MHz。各样本经共享独占锁执行，八份正式 clock report 均退出 0、恢复无错误。两种存储结果跨板，不作为同板替换存储的因果 A/B。

Rust 基线 `49ab166a3f3af9d715bee4183f752f28bae659a0`，native 固定 `f872a946e4ce8dfba51317fc6aad68bfff311f46`，默认 all-models release 构建。当前工作区另有既存改动，构建所用 Rust/Cargo 差异保存于 evidence。

全驻留包含一处加载预算修复：仅驻留 MoE 使用每域 2GiB 减 256MiB 的预算，避免统一 1GiB 预估上限导致 40 层专家耗尽域数量；dense 和流式仍为 1GiB。驻留使用含此修复的构建，流式使用原构建，native 相同。驻留两档均完成，且与流式 DS3 文本一致；该预算的本轮板端验证范围为 ctx4096、ubatch128、DS3。ARM release 构建及板端 `cargo fmt --all --check` 通过。未提交代码。

## 原始记录

[evidence/minicpmv47_rebench_20261029/](evidence/minicpmv47_rebench_20261029/) 保存 `protocol.json`、两条完整输入和 token IDs、源数据及 revision、两个固定二进制、构建日志、运行脚本、每次 stdout/stderr、clock report、原始统计、遥测、`summary.json` 和 `output-comparison.json`。峰值 RSS 使用模型末尾报告的进程 VmHWM，并与可用的外部采样取最大值。

H88K 原始文件另在 `/mnt/nvme/minicpmv47/results/rebench_20261029/`。可在主机执行 `python3 llm_generated_docs/evidence/minicpmv47_rebench_20261029/summarize.py` 重新汇总已保存结果，不触发新的板端运行。
