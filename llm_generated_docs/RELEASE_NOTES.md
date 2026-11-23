# v0.13.0-rc.1

- OpenAI 兼容服务新增固定槽位连续批处理。`serve --parallel N` 可将多路文本请求的 decode 和分块 prefill 合并到同一个模型步。支持 Qwen3、Llama、Granite 和 Qwen3.5 系列的文本请求。
- 每个槽位独立持有 KV、GDN 和采样状态，按 `--ctx-size` 预留 KV 容量。慢连接的输出背压按请求处理，调用端可控制请求额度。批处理要求 `--prefill-ubatches 1 --spec-type none`，前缀复用、投机解码和多模态请使用单请求模式。
- 服务按墙钟窗口分别显示 prefill 和 decode 吞吐、缓存命中和运行槽位。启用投机解码时显示草稿提议、接受数和接受率。
- RK3588 上验证了 Qwen3-4B 和 Qwen3.5-4B C7 的 2 槽、4 槽 logits、槽位回收，以及 C7 的真实 SSE 并发和单请求 MTP 计数。
