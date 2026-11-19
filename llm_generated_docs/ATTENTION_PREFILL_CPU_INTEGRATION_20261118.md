# RK3588 causal GQA prefill 合入与 CPU 成本

Qwen3-4B 的 4K prefill 在这块板上平均提前 2.38 秒给出首 token，每个请求多耗费 5.82 CPU 核秒，平均多占用 0.38 个大核。512-token 请求平均多耗费 0.06 核秒。算子库本地提交 `2043c24`，外层推理仓库本地提交 `d794558` 更新子模块指针。

| 真实请求 | 首 token 秒，原库 → 集成版 | 整请求 CPU 核秒 | 整请求平均大核数 | Attention 墙钟秒 | Attention 区间 CPU 核秒 | Attention 区间平均大核数 |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 3.300 → 3.138 | 6.278 → 6.337 | 1.902 → 2.020 | 0.685 → 0.552 | 0.959 → 1.001 | 1.401 → 1.811 |
| 4096 | 30.430 → 28.055 | 61.105 → 66.921 | 2.008 → 2.385 | 12.948 → 10.792 | 21.701 → 27.319 | 1.676 → 2.532 |

按四个可用大核计算，4K 整请求平均占用从 50.2% 升到 59.6%。这 5.82 个额外核秒中，5.62 个发生在 Attention 区间。两对正反顺序下，4K 整请求 CPU 核秒分别增加 10.21%、8.82%，首 token 延迟分别下降 9.13%、6.43%。512 两对的 CPU 核秒变化为 +1.99%、−0.11%，总体变化接近进程间波动。[逐进程指标与汇总](evidence/gqa_prefill_cpu_integration_20261118/summary.json)

## 测量方法

在 主板 RK3588、RKNPU 0.9.8 上运行 `Qwen3-4B-W4A8-Mixed-LMHeadW8`。CPU4–7 固定 2.4 GHz，NPU 三核 1 GHz，DDR 2.112 GHz，OpenMP 四线程、PASSIVE，ubatch128，同一进程只执行一个请求。原库为 `3be24de`，开启双 slot、LUT、pingpong。用原库、集成版、集成版、原库的顺序运行同一二进制，每进程对同一 WikiText 输入的 512/4096 两种长度各预热一次，再分别正式跑两次。每次清空 KV、生成一个 token，92 个线性层策略全部命中。每种实现每个长度共 4 个正式请求，先取进程内中位数，再取两个进程的平均。校验所有正式请求首步完整 logits 与预热相同，再进行跨进程、跨版本对照。[运行参数与采样](evidence/gqa_prefill_cpu_integration_20261118/measure.py) · [固定时钟与恢复](evidence/gqa_prefill_cpu_integration_20261118/model-clocks.json)

用 Linux `CLOCK_PROCESS_CPUTIME_ID` 读取进程全部线程实际消耗的 CPU 时间。整请求从引擎开始执行到首个非空 token，模型加载和 logits 导出在计时外。Attention 在每次真实 `layer.attention.run` 调用前后读同一个进程时钟，累加该请求的所有层、所有块。平均核数为对应区间的 CPU 核秒除以墙钟秒。该实验测量单请求平均占用，未测同板多服务争用、瞬时峰值或能耗。[测量代码差异](evidence/gqa_prefill_cpu_integration_20261118/cpu-measure.patch) · [统计脚本](evidence/gqa_prefill_cpu_integration_20261118/summarize.py)

## 合入范围与复核

在默认 pingpong 模式下，AArch64 causal GQA 的每组四个 query head、每段 32 行满足 LUT 条件时，CPU 保留每行真实最大值，以原生布局顺序写出硬件指数曲线对应的 FP16 概率。四个大核按 head 分担扫描与输出合并。NPU 以整组方式执行 QK 和 PV，两者之间复用独立 write-combine 缓冲。既有 V 幅值和任务数保护继续生效。某段出现非有限行最大值时，原 score 仍在，整段退回 DPU 的指数与屏蔽流程。显式 `RK_NPU_ATTN_LUT_SUBMIT=pc` 或 `split` 继续使用 DPU。[算子说明](../native/rk_npu_matmul/docs/ATTENTION.md) · [上一轮分 head 的实测结果](ATTENTION_MODEL_TTFT_AND_HEAD_GROUPS_20261118.md)

上表 A/B 使用集成中的同一默认计算路径。最终提交补齐了实发 task 数的统计，保留显式 PC/split 的 DPU 行为。最终动态库另外运行了默认 pingpong 和显式 pc 两种整模型验证，每个长度各一次预热和一次正式请求。具体数值见下表。两个前缀各有 151,936 个 FP32 logits，两种模式均与旧库逐位一致。板上穷举的 19,457 项 CPU 指数表也与原始 DPU 测量逐位相同。故障注入、KV 截断/重载、掩码和并发测试通过，同一图上的可见 Inf、NaN 与恢复至有限值也已对照 DPU 的实际概率缓冲。[验证结果](evidence/gqa_prefill_cpu_integration_20261118/validation.json) · [pipeline 板测](evidence/gqa_prefill_cpu_integration_20261118/release/pipeline.log)

| 最终库单次复核 | 512 TTFT / CPU 核秒 | 4K TTFT / CPU 核秒 |
|---|---:|---:|
| 默认 pingpong | 3.164 s / 6.403 | 28.207 s / 67.853 |
| 显式 pc | 3.284 s / 6.396 | 30.097 s / 61.767 |
