# Qwen3.5 静态图像接入

2026-10-18，分支 `feat/qwen35-vision`，基线 `d8bd33b`。

本轮后续已增加正式 `tune --vision [--image PATH]`、移除正式运行的全局板锁，并完成真实照片的 OpenAI 服务和实际 PTY 多轮交互。当前用法、70项测试和 API/TUI 原始回答见 [用户流程验证](VISION_USER_FLOW_VALIDATION.md)；以下保留首版实现与早期验证记录。

目标：复用 C7 的 BF16 ViT、patch merger 和现有量化文本 decoder，支持单图、多图、图文交错、多轮上下文、CLI/TUI 与 OpenAI chat API。首版不含视频；API 只接受内联 PNG/JPEG data URL。

## 模块边界

- `vision/`：图像解码、受限 resize、归一化、patch 排列及二维位置；通用 LayerNorm/GELU 分别位于 `layers/layer_norm.rs` 和 `ops/gelu.rs`。
- `input.rs`：展开后的 token、图像内容身份、图像 span、三轴位置和上下文预算；逻辑 KV 行与 RoPE 坐标分别维护。
- `npu/floating.rs`：公共 C ABI 的 FP16 Linear；独立于量化 Linear。无因果 Attention 复用公开 native 算子，`FullAttention` 按实际 patch 数准备本图临时 K/V 与 scratch，跨层覆写复用，blocks 完成后释放，不保留视觉 KV 状态。
- `model/qwen35/vision/`：C7 结构、权重映射、24层 ViT、merger；按需加载，输出已在 C7 residual 坐标系，不再旋转。
- `engine/context.rs`：前缀身份与位置匹配；视觉 feature cache 限定容量。
- `api/request.rs`：请求解析和 content parts 规范化。

固定参考：Transformers `09281b8d56b08199ce9f397c694f38853cc45278` 的 Qwen3.5 和 Qwen2-VL processor。ViT MLP 使用 tanh GELU，merger 使用精确 GELU，二者不可混用。图像 resize 以该版本的 PIL backend 对照；像素面积上下限读取 checkpoint 的 `size.shortest_edge` / `size.longest_edge`，不另设运行时 patch/token 上限。C7 配置为65536–16777216像素，保持纵横比并将宽高对齐到32像素。处理参数参与图像身份。

## 使用

需要 C7 这类同时包含 `model.visual.*` BF16 权重、processor 和原生 chat template 的 checkpoint。视觉权重按需上传，文本请求不加载 ViT。C7 merger 已折入文本 residual 旋转，不应重复旋转。

```sh
open_rknpu2_lm_infer --model MODEL --ctx-size 4096 run \
  --image photo.png -p '描述这张图片。'
open_rknpu2_lm_infer --model MODEL run \
  --image first.jpg --image second.png -p '比较两张图。'
```

`chat` 中输入 `/image PATH` 追加附件，再发送文字；可重复追加，`/reset` 清理历史和待发附件。CLI `--image` 自动使用 checkpoint 的 chat template。

`/v1/chat/completions` 的用户消息保留图文交错顺序：

```json
{
  "model": "C7",
  "enable_thinking": false,
  "max_tokens": 64,
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "请描述："},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BASE64_BYTES"}}
  ]}]
}
```

图像支持 PNG/JPEG，`image_url.detail` 只支持省略或 `auto`；远程 URL、视频和 system/assistant 图像返回明确错误。多轮请求由调用者发送完整 messages，历史图像也用相同 content parts 传回。API 不自动删除历史；展开后的视觉 token 与文字 token 一起占用上下文，再预留 `max_tokens`，超限返回400。

每张图的视觉 token 数为 resize 后 patch 数除以4，与文字及生成预算共同占用上下文。每会话最多8图，单图编码数据≤8 MiB、请求图像数据合计≤16 MiB，HTTP body≤24 MiB；解码维度和内存另有限制。特征 LRU cache 上限32 MiB，缓存身份包括处理后的像素、grid及归一化参数。KV 前缀逐行比较token、三轴位置及图像身份；新图不能通过相同 image-pad IDs 复用旧 KV。当前 GDN 不能回退至任意非零前缀，分叉历史通过 reset/recompute 恢复，已有视觉 features 仍可复用。

## 已完成的验证

- processor：5组确定性图像覆盖上下采样、方形/非方形，与固定版本 PIL processor 的 resize、归一化、patch 排列逐值一致。
- 公共数值：LayerNorm、精确/tanh GELU、位置插值、二维轴向RoPE与固定参考一致；多图展开、三轴位置和不合法输入测试通过。
- 备用1、Release：FP16 Linear 的 M3/M129 输入与尾块对照通过，max abs分别0.000238/0.000244；NO_MASK Attention max abs `1.45e-5`，再次调用相同输入逐值一致。
- 完整24层 ViT：C7 BF16权重转FP16计算，FP32参考关闭TF32。64-token红图的最终特征 cosine `0.9999825`、relative L2 `0.5928%`；96-token长方形蓝图为 `0.9999909` / `0.4278%`。各层数据和完整trace保留在共享实验目录；重复图像命中特征缓存。
- 完整API：红图/蓝图、相同尺寸且相同占位IDs的换图、双图SSE、多轮图像历史、上下文超限、远程/坏图像错误、断开SSE后的下一次请求、图像后的纯文字请求均通过，服务正常关闭。图像输出为 `red`、`blue`、`red, blue`，历史问题仍回答 `red`，最后纯文字回答 `OK`。
- CLI 单图也已在板端生成 `red`。三种图文输入的三轴位置和decode offset与固定版本的 `get_rope_index` 完全一致，logical tokens分别94/126/197、RoPE delta为−56/−84/−140。
- 固定continuation共10行、2,483,200个图文logits，全部top-1与参考一致；但**严格分布检查仍未通过**：红图首位置 KL `0.131749` 超过预先设置的0.1线，失败留在 `comparison.json`，不扩大阈值。
- 误差拆分：固定同一个文本整数参考，只把FP32 ViT特征换成板端特征，10行KL最大 `0.024159`；固定板端特征，文本参考与原生执行之间KL最大 `0.101192`。两部分误差不是可加分解，不能直接归为某一个算子的误差。原生文本原有FP16 Attention/A8 rounding等差异仍存在；这不证明通用VQA精度等价。
- 纯文字129-token prompt +4-token continuation：5行、1,241,600个logits与原运行时逐位一致。视觉接入没有改变这组文本回归输出。

这些图像是数值与状态回归样本，不代表通用视觉问答准确率。带trace的单次encoder耗时包含I/O且部分验证时有编译任务，不作为正式性能基线；FP16调优交接另见 [QWEN35_VISION_FP16_HANDOFF.md](QWEN35_VISION_FP16_HANDOFF.md)。

最终固定快照：Release默认测试62项通过、8项忽略，额外5项NPU oracle通过；fmt与全target Release Clippy（`-D warnings`）通过。JPEG相机方向校正与PNG完整预处理的固定PIL patch哈希测试通过。8次runner均正常退出，受控频率上下限及governor恢复。汇总见 `validation.json`。严格图文logits分布问题独立保留，未称作通过；本阶段交付静态图像功能和可复现数值记录。

## 并行开发的版本边界

本报告的板端结果来自 `/tmp/qwen35-runtime` 的固定FP16默认实现，未启用自动调优。其他agent随后开始修改共享工作树的 `src/npu/floating.rs`、`build.rs` 和算子子模块；这些改动保留且不计入上述验证。固定wrapper和构建入口的源码副本位于证据目录 `baseline-source/`，完整可构建的默认策略快照保存在共享目录 `fixed-vision-source.tar.gz`，源码/归档哈希见 `source-manifest.json`；测试CLI/probe哈希见 `binary-manifest.json`，实测程序留在共享目录 `baseline-cli` / `baseline-probe`。

新调优实现合入后，应复跑FP16 oracle、ViT features、图文logits与API状态验证，不能沿用本报告作为新策略的通过记录。

## 复现与证据

小型结果位于 [evidence/qwen35_vision_20261018/](evidence/qwen35_vision_20261018/)。大型权重、FP32 patches/features/logits 不入仓库；完整数据位于 `/home/zt/mnt/3090x4/tmp/qwen35-vision-20261018/`，备用1对应前缀为 `/home/firefly/mnt/3090x4/`。板载时钟仍为2024年，日志时间只作原始记录，本报告日期以主机为准。

固定参考的源码、哈希与运行环境见 `reference-manifest.json`；参考源码为 Transformers Apache-2.0：

- [Qwen3.5模型](https://github.com/huggingface/transformers/blob/09281b8d56b08199ce9f397c694f38853cc45278/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [视觉位置工具](https://github.com/huggingface/transformers/blob/09281b8d56b08199ce9f397c694f38853cc45278/src/transformers/vision_utils.py)
- [PIL processor](https://github.com/huggingface/transformers/blob/09281b8d56b08199ce9f397c694f38853cc45278/src/transformers/models/qwen2_vl/image_processing_pil_qwen2_vl.py)

`scripts/qwen35/vision_reference.py` 只提取原始定义、提供轻量框架包装与标准Torch SDPA，不改量化工程、checkpoint或Python环境；`vision_model_reference.py` 复用量化工程现有的 Transformers5.3文本整数参考，并显式传入三轴位置和逐个continuation位置。两部分的版本分别记录，不声称整模型都运行了新版Transformers。
