# W8A8 压缩准备阶段的 CPU 堆回收

2026-10-27，`zt-firefly` RK3588，按 M bucket 标记回收的 release 构建。当前按 native 布局 generation 触发的实现和回归见[布局回收说明](W8A8_COMPRESSION_LAYOUT_REVIEW_20261028.md)。本文保留当日内存、吞吐与调用次数的原始测量。

`--compress-w8a8` 自动回收压缩准备阶段已经释放的 CPU 堆页。真实 Qwen3-4B 全 W8A8 模型的稳定 RSS 为 **3944.18 MiB**：比同一二进制屏蔽回收的对照少 **349.61 MiB**，比原始未压缩路径少 **223.36 MiB（5.36%）**。

## 实现

压缩权重创建，以及每个 batch bucket 首次成功执行时，标记有准备阶段内存待回收。Dense、Qwen3.5 在加载结束、整组 decoder 层执行结束后统一处理标记；共享输出头在首次布局执行结束后独立收尾。

双 ubatch 的执行通道共享标记，回收发生在并发任务结束后。每个 Linear 用位图记录已经使用的 batch bucket；重复 batch 不再请求回收。新 bucket 可能需要新的 N 分块，因此首次使用时保守地请求一次回收。

实际回收使用 `malloc_trim(0)`，作用于分配器中的空闲页。仍在使用的权重、KV、workspace 和数值计算路径保持原有生命周期。没有新增命令行参数，也没有修改全局 malloc 策略。

## 内存实测

同模型目录 `w4a8_quant/Qwen3-4B-W8A8-Baseline`，145 个 W8A8Linear，context1024、ubatch128、单 prefill ubatch、`--no-tune`。CPU4–7 固定2.4 GHz、NPU 1 GHz、DDR 2.112 GHz，4线程、CPU mask `0xf0`、NPU mask7。

| 指标 | 未压缩 | 压缩，屏蔽回收 | 压缩，自动回收 |
|---|---:|---:|---:|
| 稳定 RSS | 4167.54 MiB | 4293.79 MiB | **3944.18 MiB** |
| 进程峰值 RSS | 4716.22 MiB | 4903.24 MiB | **4660.64 MiB** |
| NPU GEM 总分配 | 4034.50 MiB | 3574.67 MiB | 3574.67 MiB |

NPU 分配已包含在 RSS 中，不能相加。两个压缩运行的 NPU GEM 总分配相同，内存差值来自 CPU 一侧。自动回收后仍存在部分匿名堆驻留；本改动不保证归还 malloc 的全部空闲空间。

三个运行使用同一应用二进制。诊断动态库记录 `malloc_trim` 调用，屏蔽对照仅让该函数返回0，以隔离回收效果；这个开关只存在于测试工具中。正式程序直接调用 libc。

## 调用次数与速度

每进程先进行 head-free continuation logits 检查和一轮预热，再测6轮 `pp512 + tg32 @512`。第1轮后读取内存快照，丢弃第1、2轮，保留第3–6轮并取中位数。

自动回收共调用 **4次**：加载结束、decoder 首次 M128、LM head 首次 M1、decoder 首次 M1。实际 `malloc_trim` 合计 **37.87 ms**。6轮正式测量期间调用数为 **0**，覆盖192次 decode 模型调用。未压缩路径调用数为0。

| 吞吐 | 未压缩 | 压缩，屏蔽回收 | 压缩，自动回收 |
|---|---:|---:|---:|
| head-free pp512 | 128.084 token/s | 126.195 token/s | 124.823 token/s |
| 完整 decode32 @512 | 5.179 token/s | 5.137 token/s | 5.151 token/s |

自动回收相对屏蔽对照的名义变化为 prefill −1.09%、decode +0.28%。这是每模式一个进程、4个保留样本的配对，不能据此宣称零开销或性能收益。审计确认了稳定阶段没有执行 `malloc_trim`。

## 验证

- 板端 release 构建、全目标 Clippy `-D warnings` 通过，本次修改的 Rust 文件通过 rustfmt。
- 102项普通测试通过；公开压缩接口的9组形状/核掩码组合输出逐位一致，检查加载、新 batch 回收及重复 batch 不回收。
- 双执行通道测试通过：共享回收标记，第二通道首次使用新 batch 后可由主通道统一回收，后续并发调用不重复触发。
- 真实模型8-token prefill及2-token continuation：**455,808个有限 FP32 logits 与原始参考逐位一致**。
- 所有运行退出0，频率上下限及 governor 恢复，`restore_errors=[]`。

## 记录

- [验证汇总、同二进制对照及内存差值](evidence/w8a8_lossless_20261027/cleanup/verified-summary.json)
- [正常回收原始报告](evidence/w8a8_lossless_20261027/cleanup/compressed.json) / [屏蔽回收对照](evidence/w8a8_lossless_20261027/cleanup/control/compressed.json)
- [正常运行频率和恢复](evidence/w8a8_lossless_20261027/cleanup/clocks.json) / [对照频率和恢复](evidence/w8a8_lossless_20261027/cleanup/control-clocks.json)
- [实际回收调用记录](evidence/w8a8_lossless_20261027/cleanup/compressed.trim.jsonl)
- [运行和复核代码](evidence/w8a8_lossless_20261027/cleanup/run.py) / [调用审计工具](evidence/w8a8_lossless_20261027/cleanup/audit.c)

同目录保存构建、Clippy、测试日志、内存映射及完整 logits。清理前的 ABBA 结果和 malloc 诊断见[初始报告](W8A8_LOSSLESS_BENCH_20261027.md)。
