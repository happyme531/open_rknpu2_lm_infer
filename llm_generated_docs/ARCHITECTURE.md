# Architecture

`npu` 通过 bindgen 生成现有公共 C ABI 的绑定。Backend 持有 context/domain、几何调优策略和共享工作区；Linear 持有权重和 Backend 引用；Attention 持有 KV 和 Backend 引用。所有对象在专用推理线程中运行，Rust 的析构顺序和所有权负责释放资源。

ViT 使用独立的 `FullAttention`：按当前图像的 patch 数分配临时 K/V 与 Attention scratch，每层完整覆写 K/V，完成图像的 Transformer blocks 后释放。图像 buffer 不受文本 `ctx-size` 或独立的固定 patch 容量约束；图像特征仍可由 Encoder 的 LRU 缓存复用。

`model::checkpoint` 使用 safetensors 与 memmap2 验证 checkpoint 并加载 W4/W8。`model::dense` 共用 Qwen3、Llama 和 Granite 的 CPU 运算与 NPU 调度。Qwen3 执行 Q/K Norm；Llama 和 Granite 直接对投影结果执行 RoPE。Granite 的 embedding、attention、residual 和 logits 缩放由配置提供。相邻同位宽的 QKV、gate/up 合并投影；量化 lm_head 单独分配 domain，生成使用 M1，PPL 使用批量工作区。

`LanguageModel` 定义模型信息、reset、KV reserve、prepare、forward、可取消 forward、KV position 和 stats。新增架构在 `model::load` 注册，并独立实现此接口。模型类管理其 KV 和 scratch，外层生成引擎不依赖 Qwen3 层结构。

稠密模型按最大上下文的权重/KV 预算划分 domain，KV 在请求开始前按 prompt 加最大输出预留；容量增长后保留供后续请求复用。调优和计时前完成预留，prefill/decode 内不增长本请求的 KV。

`Engine` 在单一专用线程中加载模型，使用有界请求通道。Tokenizer 和 checkpoint 自带的 Jinja 聊天模板负责 prompt 格式；Sampler 实现 greedy、temperature、top-k、top-p、min-p 和 repetition/presence/frequency penalties。StopFilter 在流式文本中保留可能匹配 stop 的后缀。

聊天模板使用 Transformers 的空白裁剪规则，缺省 tools 作为未定义值。API 根据实际生成提示是否已开启 `<think>` 初始化 reasoning 解析，支持模型把开始标签放进提示词的格式。

每个请求持有取消标记和事件通道。GenerationHandle 析构时设置取消；模型在 prefill 块和 decode 步之间检查标记。事件发送具有回压且可取消。相同 token 前缀通过原生 KV truncate 复用，末端 token 重新计算以获得请求的正确 logits；失败或中断重置缓存。

`cli` 使用 clap，`tui` 使用 ratatui/crossterm，`api` 使用 axum/tokio 和现成 SSE response。三个入口共用 Engine。HTTP 线程不持有或直接调用 NPU 对象。

进程持有 `/tmp/rknpu_lowlevel_submit.lock`；板端 runner 可通过 `OPEN_RKNPU2_LOCK_FD` 或 `LLM_INFER_LOCK_FD` 传入已持有的锁。模型销毁后权重、workspace、KV、domain 和 context 按所有权释放。
