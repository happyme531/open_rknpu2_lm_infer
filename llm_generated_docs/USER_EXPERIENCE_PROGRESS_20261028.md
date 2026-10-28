# 推理进度与调优提示

`run`、`chat`、`serve` 默认显示推理进度。状态使用独立的 Tokio watch 通道，接收端在终端 IO 前复制快照并释放读锁，不占用正文 token 队列。TTY 在状态刷新时缓存格式化、换行后的文本，正文片段移动状态区时整块复用缓存；宽度改变时重建换行。

## 显示与计数

- Prefill 按模型实际完成的 ubatch/group 更新，包括双 ubatch 和 checkpoint 切分；显示已处理/待处理 token、缓存命中数、速度及 ETA。视觉编码使用单独的阶段提示。
- Decode 显示输出数、最近约一秒完整轮次的速度，以及全程平均速度。速度复用 `decode_consumed_tokens / decode_model_ms` 的口径，不包含 prefill 产生的首 token。
- MTP/DFlash 增加接受/提议数、接受率、平均每轮产出、轮数、draft/verify 平均耗时及置信度截断次数。接受长度包含验证器额外产出，与最终 `spec_accept_length` 保持一致。
- `run` 的正文写 stdout，状态写 stderr；TTY 状态显示在正文下方。chat 使用独立状态区，提示保留在可滚动的界面记录中，不进入模型消息。
- server 使用完整日志行，HTTP/SSE 响应格式保持兼容。非 TTY 每五秒更新，阶段切换和结束立即记录；TTY run 最多每 250ms 更新，TTY server 每秒记录。开启 INFO/DEBUG 日志时，run 使用完整状态行，避免日志干扰光标位置。
- `tune --vision` 显示整体 shape 进度与当前投影的 M/K/N，同时复用 native 的候选搜索进度和 ETA。缓存命中也推进整体计数。

## 未调优提示

W4/W8 和浮点后端仅在实际选用默认策略且缺少可用缓存时记录提示，整个进程合计显示一次。target、DFlash 和视觉命令保留模型、缓存目录、线程数和 CPU/NPU mask。MTP 浮点投影由加载器明确标记来源，提示当前 CLI 未提供对应调优入口。

尾块上下文覆盖普通 prefill、checkpoint 部分块、DFlash 上下文，以及 MTP 首轮少一行 hidden state 的情况。尾块不会消耗提示次数；decode 和推测验证的小 batch 正常检查。候选策略探测、显式 tuning 搜索及 `--no-tune` 不触发提示。

Rust 接口新增 `Engine::subscribe_progress()` 和可选的 `LanguageModel::forward_prepared_with_progress()`；原有生成事件、HTTP/SSE 格式和 native C ABI 保持兼容。

## 验证

板端为 `zt-firefly`，aarch64，Rust 1.95。构建产物为板端 `/tmp/prefill-schedule-target-20261025/release/open_rknpu2_lm_infer`。

| 检查 | 结果 |
| --- | --- |
| 全 target 编译、release 构建、Rust 格式和 diff 空白检查 | PASS |
| 库测试 | 104 通过；26 个设备/模型测试按默认规则忽略 |
| API mock 回归 | 46 通过；5 个真实 tokenizer 测试按默认规则忽略 |
| 独立板端尾块检查 | 尾块 miss 不提示；普通 shape miss 后提示；跨后端只提示一次；提示保留指定缓存路径 |
| 真实 CLI 请求 | Dense Qwen3 C7、Qwen3.5 C7、MTP1、C4+DFlash3 均通过；638-token 输入、24-token 输出上限；均保留双 ubatch 执行，`tune_calls=0` |
| server | 普通 completion、SSE completion、真实猫图 chat 请求通过；进程中只出现一次未调优提示 |
| 视觉 tuning | 隔离的小型配置完成 7 个 shape 搜索；热缓存复用 7 次，搜索数为 0；整体进度到 100% |
| 真实 PTY | 100/48 列 run、chat 缩放、取消和重置恢复通过；run 正文与直接输出的 VT 渲染逐行一致 |

视觉搜索检查使用 hidden=64、intermediate=128、out=64 的独立配置和临时缓存，验证搜索、缓存与进度显示。真实图像推理检查使用生产 Qwen3.5 C7 和 `cats.png`；未重新调优生产 ViT。

成功的 server/PTY 检查退出码均为 0，时钟恢复记录均为 `restore_errors=[]`。

原始记录：

- [构建与回归](evidence/ux_progress_20261028/validation.json)
- [四种真实 CLI 路径](evidence/ux_progress_20261028/cli-smoke.json)
- [server 与视觉 tuning](evidence/ux_progress_20261028/server-smoke.json)
- [终端检查](evidence/ux_progress_20261028/terminal.json)
- [run 实时统计](evidence/ux_progress_20261028/run-narrow-decode.png)
- [chat 实时统计](evidence/ux_progress_20261028/chat-terminal-decode.png)
- [chat 窄终端取消](evidence/ux_progress_20261028/chat-terminal-cancelled-narrow.png)
