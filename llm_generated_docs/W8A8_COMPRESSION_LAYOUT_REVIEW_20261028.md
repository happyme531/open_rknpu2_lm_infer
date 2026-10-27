# 压缩布局回收触发与输出头检查

## 回收触发

native 的压缩布局缓存按每个权重对象的 `n_tile` 管理。`rk_npu_f32i8f32_weights_compressed_layout_generation()` 提供线程安全的 generation：创建 full-N seed 后为1，成功发布新布局时递增，缓存命中不改变它。淘汰未使用的 seed 不回退 generation；以后重建该布局仍会递增。raw 权重和空指针返回0。

Rust 在权重加载时记录初始 generation，运行后只在 generation 前进时设置待回收标记。不同 M 共享布局、首次使用命中 seed 时均不重复回收。观察值通过 atomic fetch-max 保持单调，执行通道共享同一个 Linear 与回收标记，整组任务结束后统一处理。

## LM head

输出头在调用回收 helper 前检查 Linear 的不可变压缩状态。仅启用 W8 压缩的 head 进入该路径；W4、未压缩 W8 和浮点 head 都跳过回收检查。检查压缩状态不获取 backend 锁。

## 回归验证

板端102项普通测试通过，另有两项压缩回归和一项跨执行通道回归通过；全目标 Clippy `-D warnings`、release 构建和本次修改文件的 rustfmt 通过。

- 小尺寸 seed 命中、不同 M 复用同一 N 布局，不请求回收。
- 同一 M 改用新的 `n_tile`，generation 递增且请求一次回收；再次复用不回收。
- 未使用的 seed 被淘汰后，重建该分块仍能触发回收。
- 第二执行通道使用新 M 但命中缓存时不回收；创建新 N 布局后可由主通道统一回收。
- W4 在全局压缩开关打开或关闭时，压缩状态均为 false；未压缩 W8 同样为 false。
- 所有形状、尾块、split-K 和并发调用继续保持 raw/DCOMP 输出逐位一致。

真实 `Qwen3-4B-W8A8-Baseline` 的145个 W8 Linear，`--no-tune`、context128、ubatch128、固定 CPU2.4 GHz / NPU1 GHz / DDR2.112 GHz：8-token prefill及2-token continuation的455,808个有限 FP32 logits与原始参考逐位一致。审计记录3次回收，分别对应加载、decoder新布局和输出头新布局；decoder切换到M1时命中相同布局，没有新增回收。所有板端步骤退出0，频率设置恢复通过。

详见[复核摘要](evidence/w8a8_layout_review_20261028/verified-summary.json)、[回收记录](evidence/w8a8_layout_review_20261028/trim.jsonl)和[真实模型验证脚本](evidence/w8a8_layout_review_20261028/model.py)。

记录目录：[w8a8_layout_review_20261028](evidence/w8a8_layout_review_20261028)。

2026-10-27 的[内存和吞吐报告](W8A8_COMPRESSION_HEAP_CLEANUP_20261027.md)记录当时按 M bucket 标记回收的版本，不能将其中4次初始化回收的计数当作当前布局 generation 方案的固定计数。本次不重测吞吐或压缩率。
