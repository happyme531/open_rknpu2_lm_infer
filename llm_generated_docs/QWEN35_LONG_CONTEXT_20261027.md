# Qwen3.5-4B 长上下文性能（2026-10-27）

RK3588 `主板`，Qwen3.5-4B C7（`w4a8_quant/transfer/qwen35/models/C7`），CPU 大核 2.4 GHz、NPU 1.0 GHz、DDR 2.112 GHz。ubatch128、单 ubatch，4 个 CPU 线程、CPU mask `0xf0`、NPU mask `7`、Attention tile512；关闭 MTP、关闭 thinking，greedy 生成。每档独立加载模型，无 warmup，每档单样本。

| 输入 token 数 | Prefill (tok/s) | Prefill 耗时 (s) | Decode (tok/s) | 峰值 RSS (GiB) | 检索 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K (4,096) | 158.3 | 25.9 | 9.21 | 3.69 | 3/3 |
| 8K (8,192) | 149.0 | 55.0 | 8.57 | 3.70 | 3/3 |
| 16K (16,384) | 135.6 | 120.9 | 7.59 | 4.02 | 3/3 |
| 32K (32,768) | 115.3 | 284.3 | 6.19 | 4.79 | 3/3 |
| 64K (65,536) | 82.0 | 798.8 | 4.42 | 6.49 | 3/3 |

输入使用聊天模板前后缀与语料 token 拼接，在约 10%、50%、90% 深度插入三个项目代码；各档全部找回。此检索探针不替代长上下文 PPL 或完整任务集质量评估。

Prefill 时间为完整 1024-token 模型 prefill/forward 调用的累计时间（内部 ubatch128），包含最终输出头及首 token 的 greedy 扫描。Decode 为后续 127 次单 token 模型调用及 greedy 有限 logits 扫描的累计时间；总输出为 128 token。两者均排除模型加载、进度、统计与报告 I/O。峰值 RSS 来自进程 `VmHWM`。64K 模型调用累计 prefill 为 798.769 秒，包含记录开销的 prefill 墙钟时间为 813.724 秒（80.54 tok/s）。

## 调优与验证边界

| 输入 | tune_cache_hits | tune_cache_misses | tune_calls | strategy_fallback_calls |
| --- | ---: | ---: | ---: | ---: |
| 4K | 64 | 0 | 0 | 0 |
| 8K | 68 | 14 | 0 | 1337 |
| 16K | 78 | 20 | 0 | 4080 |
| 32K | 84 | 16 | 0 | 9575 |
| 64K | 62 | 84 | 0 | 60066 |

长上下文部分形状使用内置策略，以上数据不代表所有形状均已调优的性能上限。64K 结果使用 KV 先于投影权重预分配、每 IOMMU 域按 1 GiB payload 预算规划的加载路径。实测最大输入为 65,536 token，配置容量为 65,664 token；128K 无完整 prefill/decode 数据。板端频率恢复成功，`restore_errors=[]`。

## 数据来源

- Codex 会话 `01a17ae4-5795-7f23-a2ee-fdd28522b38f`，2026-10-27，末尾性能表与原始实验汇总一致。
- 本地原始记录目录：`llm_generated_docs/evidence/qwen35_long_context_20261027/`（Git 忽略目录）。`summary.json` 保存汇总；`pp4096.json`、`pp8192.json`、`pp16384.json`、`pp32768.json`、`pp65536.json` 保存逐段和逐 token 原始耗时；`clocks.json` 保存固定频率及恢复记录；`long_context.rs` 和 `analyze.py` 定义计时与汇总口径。
- 实验记录的 runtime 基线：`74d940bd8ad405be1635b25db2577391eeb2b5f1`，native 基线：`811b2a37579499d746fd3bdbc2abdfd5de46ac9d`。实验另含上下文扩展和 KV 预分配改动，不能把基线 commit 单独视为完整复现版本；相关补丁保存在同目录 `runtime.patch` 和 `kv-preallocation.patch`。
