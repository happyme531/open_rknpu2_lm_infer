# 进度显示与调优提示复核

本次处理用户体验功能中的 MTP 提示来源、TTY 状态输出和不可见 Loading 状态。submodule 的 slot 内存所有权、零 route 同步，以及 `call_llm.py` 保留待单独处理。

## 当前行为

- 调优提示显式区分 target、视觉、MTP 和 DFlash。MTP 加载器使用专用来源标记，即使通过模型 API 启用 MTP、配置中的 speculative kind 仍为 none，也能识别 MTP 浮点投影。该来源提示当前 CLI 不支持对应调优，不给出只能覆盖 target 的命令。
- target、视觉和 DFlash 保留各自可用的调优命令及当前缓存、CPU/NPU 参数。
- TTY 缓存已格式化、换行的状态文本和行数。正文片段移动状态区时复用缓存，以整块 `write_all` 写入；状态刷新、新提示或宽度改变时重新生成文本。数据刷新继续受 250ms 节流控制。
- 同步启动在模型就绪后返回，进度从 Ready 开始；移除无法通过当前订阅接口观察到的 Loading 状态及发布。

## 验证

板端：`zt-firefly`，aarch64，Rust 1.95。

| 检查 | 结果 |
| --- | --- |
| 全 target 编译、release 构建、格式与 diff 空白检查 | PASS |
| 库测试 | 108 通过；28 个需要设备、模型或 TTY 的测试默认忽略 |
| API mock 回归 | 46 通过；5 个真实 tokenizer 测试默认忽略 |
| MTP 提示 | 独立进程实际创建浮点默认 workspace；缺失缓存时提示 MTP 来源及 CLI 限制，无 target 调优建议，后续不重复 |
| 状态缓存 | 重复写入保持已渲染快照；每次重画使用一个状态 payload 写入；宽度和提示变化重建换行 |
| 真实 PTY | 100/48 列均通过；确定性正文与直接输出的 VT 渲染逐行一致；节流期内连续正文片段不重建状态缓存 |

本轮验证范围为提示来源与终端渲染，没有测量端到端吞吐变化，也没有重新运行完整模型的性能或精度测试。

- [构建、回归及板端提示检查](evidence/ux_progress_review_20261028/validation.json)
- [MTP 提示输出](evidence/ux_progress_review_20261028/mtp-notice.log)
- [PTY 检查结果](evidence/ux_progress_review_20261028/tty.json)
- [100 列终端画面](evidence/ux_progress_review_20261028/tty-100.txt)
- [48 列终端画面](evidence/ux_progress_review_20261028/tty-48.txt)
