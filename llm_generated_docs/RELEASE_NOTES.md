# v0.11.0

- 修复调优缓存默认路径：优先使用 systemd 的 `CACHE_DIRECTORY`，缺少用户缓存目录时使用模型目录下的持久缓存；启动时显示缓存路径、命中和回退统计。
- 显式调优时覆盖请求的输出头 batch，包括推测解码验证使用的批次；普通生成仍只准备所需的输出头形状。
- 优化 top-p 和 min-p 截断采样，保留宽分布和边界不明确时的完整排序回退，并保持固定种子采样序列和随机数状态。
- 改进 Qwen3-Coder 工具调用解析，以及聊天模板中 JSON 历史和数字格式与 Transformers 的兼容性。
- 更新 NPU 算子库：改进 RKNPU DRM 设备发现和验证，修复 Attention 命令缓冲区分页及刷新问题，以及 Clang OpenMP 下的 query offset 捕获问题。
