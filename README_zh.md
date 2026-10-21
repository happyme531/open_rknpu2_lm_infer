<div align="center">

# open_rknpu2_lm_infer

**完全开源的高性能 RK3588 NPU 大模型推理引擎**

![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Rust](https://img.shields.io/badge/rust-1.95-orange)
![Platform](https://img.shields.io/badge/platform-RK3588-green)

[![Proudly Vibe Coded - Plasma Mix](https://vibecoded.fyi/badges/flat/main/proudly-vibe-coded-plasma-mix.svg)](https://vibecoded.fyi/)
[![Coded with Codex](https://vibecoded.fyi/badges/flat/agents/codex.svg)](https://vibecoded.fyi/)

[English](README.md) | [简体中文](README_zh.md)

</div>

## 亮点

- 🚀 **比原厂更快**：采用自动调优，异构流水执行，数据流优化，零拷贝等先进策略，平均推理速度比原厂 rknn-llm SDK 高50%以上，上下文越长优势越大，prefill 最高 **2.59×**，decode 最高 **3.56×**。在 3800-token 历史上，Qwen3-4B 生成 128 个 token 的时间从 55.4 秒降到 15.6 秒。
- 🎯 **比原厂更准**：采用先进的量化算法，Qwen3-4B 和 MiniCPM5-2B 的 WikiText-2 PPL 分别比 rknn-llm W8A8 低 2.64% 和 2.28%。
- 🔓 **完全开源**：基于自研高性能开源 RKNPU2 算子库 [`rknpu2_matmul_open`](https://github.com/happyme531/rknpu2_matmul_open/)，整个推理流程无任何原厂闭源库依赖，可随意研究/优化/添加新模型支持。
- 📦 **混合 W4/W8 量化**：4-bit/8-bit 权重配合 8-bit 动态激活的混合量化，相比原厂W8A8量化占用内存更低。权重 mmap 按需加载，Qwen3-4B 在 4K 上下文下峰值 RSS 约 3.4 GiB。
- 🧩 **开箱即用**：一个二进制同时提供 CLI、终端聊天 TUI 和 OpenAI 兼容 API，支持流式输出、thinking、function calling 和前缀缓存。

## 性能

### Qwen3.5-4B

| 测试 | rknn-llm 1.3.1 | open_rknpu2_lm_infer (MTP1) | 性能提升 |
|---|---:|---:|---:|
| 视觉编码 448×448 (ms) | 1863 | 1112.38 | +67% |
| pp512 (tps) | 58.347 | 147.832 | +153.4% |
| pp1024 | 57.844 | 148.166 | +156.1% |
| pp3800 | 54.029 | 139.917 | +159.0% |
| decode128 @512 (tps) | 5.538 | 10.611 | +91.6% |
| decode128 @1024 | 5.416 | 10.379 | +91.7% |
| decode128 @3800 | 4.565 | 9.896 | +116.8% |

### Qwen3-4B

| 测试 | rknn-llm 1.3.1 | open_rknpu2_lm_infer | 性能提升 |
|---|---:|---:|---:|
| pp512 | 105.037 ± 0.031 | 169.711 ± 0.127 | +61.6% |
| pp1024 | 94.619 ± 0.021 | 157.033 ± 0.160 | +66.0% |
| pp4096 | 50.133 ± 0.084 | 116.973 ± 0.050 | +133.3% |
| decode128 @512 | 5.772 ± 0.002 | 10.497 ± 0.003 | +81.8% |
| decode128 @1024 | 5.125 ± 0.004 | 10.090 ± 0.004 | +96.9% |
| decode128 @3800 | 2.310 ± 0.005 | 8.226 ± 0.003 | +256.0% |

### Granite-4.2-3B

| 测试 | rknn-llm 1.3.1 | open_rknpu2_lm_infer | 性能提升 |
|---|---:|---:|---:|
| pp512 | 120.459 ± 0.107 | 182.476 ± 0.114 | +51.5% |
| pp1024 | 106.277 ± 0.019 | 164.162 ± 0.240 | +54.5% |
| pp4096 | 54.404 ± 0.004 | 110.307 ± 0.066 | +102.8% |
| decode128 @512 | 6.806 ± 0.003 | 9.295 ± 0.012 | +36.6% |
| decode128 @1024 | 6.370 ± 0.006 | 9.054 ± 0.004 | +42.1% |
| decode128 @3800 | 3.932 ± 0.001 | 7.727 ± 0.003 | +96.5% |

### MiniCPM5-2B

| 测试 | rknn-llm 1.3.1 | open_rknpu2_lm_infer | 性能提升 |
|---|---:|---:|---:|
| pp512 | 211.052 ± 0.484 | 245.465 ± 0.279 | +16.3% |
| pp1024 | 188.643 ± 0.082 | 219.865 ± 0.220 | +16.6% |
| pp4096 | 98.874 ± 0.009 | 173.578 ± 0.034 | +75.6% |
| decode128 @512 | 9.431 ± 0.006 | 13.151 ± 0.029 | +39.4% |
| decode128 @1024 | 8.985 ± 0.003 | 12.954 ± 0.005 | +44.2% |
| decode128 @3800 | 7.117 ± 0.003 | 11.613 ± 0.012 | +63.2% |

## Perplexity

| 模型 | rknn-llm W8A8 | open_rknpu2_lm_infer 混合量化 | PPL 变化（越低越好） |
|---|---:|---:|---:|
| Qwen3.5-4B | 8.5197 | 8.5790 | +0.70% |
| Qwen3-4B | 12.8709 | 12.5317 | -2.64% |
| Granite-4.2-3B | 15.7448 | 16.0519 | +1.95% |
| MiniCPM5-2B | 14.1293 | 13.8073 | -2.28% |

## 支持的模型

| 架构 | 已验证 checkpoint |
|---|---|
| qwen3 | [Qwen3-4B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/Qwen3-4B-W4A8-Mixed-QQQ-Rot) |
| granite | [Granite-4.2-3B-W4A8-Mixed-QQQ-SQ](https://huggingface.co/happyme531/Granite-4.2-3B-W4A8-Mixed-QQQ-SQ) |
| llama | [MiniCPM5-2B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/MiniCPM5-2B-W4A8-Mixed-QQQ-Rot) |
| qwen3_5 / qwen3_5_text | [Qwen3.5-4B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot) |

程序支持线性层使用w4a8或w8a8(可以混合使用)per-channel per-token对称量化的 compressed-tensors 格式的模型。因此你也可以根据以上示例模型，尝试加载其他大小的模型，试验不同的混合量化配置和量化算法。

## 快速开始

你需要一个 Linux 系统的 RK3588 设备。目前版本只支持原厂`rknpu2`内核驱动，且版本必须>=0.9.6。
使用root权限运行以下命令检查驱动版本:
```bash
> cat /sys/kernel/debug/rknpu/version
RKNPU driver: v0.9.8
```
之后，复制粘贴并执行以下命令，下载最新 [Release](https://github.com/happyme531/open_rknpu2_lm_infer/releases/latest) 和 Qwen3-4B 模型后直接进入终端聊天。适用于 Debian 11 / glibc 2.31 或更新的系统，无需编译。

```bash
sudo apt install curl jq python3-pip libgomp1 libstdc++6

curl -fL "$(curl -fsSL https://api.github.com/repos/happyme531/open_rknpu2_lm_infer/releases/latest | jq -r '.assets[] | select(.name | endswith("-linux-aarch64.tar.gz")) | .browser_download_url')" | tar -xz --strip-components=1
pip3 install 'huggingface_hub[cli]>=0.34,<1'
hf download happyme531/Qwen3-4B-W4A8-Mixed-QQQ-Rot --local-dir models/Qwen3-4B-W4A8-Mixed-QQQ-Rot
# 将大核CPU频率固定为最高。可选，但如果不操作会大幅影响推理速度。可能需要root权限。
# echo "performance" > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
# echo "performance" > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
./bin/open_rknpu2_lm_infer --model models/Qwen3-4B-W4A8-Mixed-QQQ-Rot chat
```

将最后的 `chat` 换成 `serve --port 8080` 即可启动 OpenAI 兼容的 LLM API 服务。

### 图像输入与MTP

将 `MODEL` 指向包含视觉与 MTP 权重的 Qwen3.5-4B 模型目录。以下使用图文基准中的 K1 配置：

```sh
./bin/open_rknpu2_lm_infer --model models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot run --spec-type draft-mtp --spec-draft-n-max 1 --image photo.jpg --temp 0 -n 128 --stats -p '请详细描述这张照片。'
```

图像按原图分辨率输入模型，除非超出了模型允许的上限。可以动态输入不同分辨率的图片，不过如果对应的视觉token数量没有调优，视觉编码器速度可能会较慢。

## 跑出完整调优性能

推理程序使用的默认的推理策略并非最优性能。可以针对指定的模型手动调优一次。

```sh
# 将大核CPU频率固定为最高。强烈建议执行。
echo "performance" > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo "performance" > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
./bin/open_rknpu2_lm_infer tune --model models/Qwen3-4B-W4A8-Mixed-QQQ-Rot
# 调优推测解码相关的batch，提升推测解码的速度
#./bin/open_rknpu2_lm_infer tune --model models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot --batches 1,2,4,128
# 使用指定输入分辨率调优视觉编码器
#./bin/open_rknpu2_lm_infer tune --model models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot --vision --image photo.jpg
```

首次调优可能需要数分钟，会显示进度和最佳候选耗时；之后重复运行会复用有效缓存。

## 架构

```
   CLI  ·  TUI  ·  OpenAI API (axum)
                 │
   Engine ── 专用推理线程 · 请求队列 · 采样 · 前缀 KV 复用
                 │
   LanguageModel ── Qwen3 / Llama / Granite / Qwen3.5（视觉 + MTP）
        │                         │
   CPU (Rust + C++)          NPU (rk_npu_matmul, C ABI)
   Norm · RoPE · Conv · GDN   W4A8/W8A8 Linear · Attention · ViT · GDN
                                  │
                         开源 RKNPU2 内核驱动
```


通用 CPU 算子位于 `src/ops/`，可复用 GDN 层和状态事务位于 `src/layers/`；`src/model/` 负责 checkpoint 与模型执行。Dense 和实验性 Qwen3.5 共用输出投影、trace 和量化投影组件。职责边界见 [模块组织说明](llm_generated_docs/RUNTIME_MODULE_LAYOUT.md)。

## 开发

### 从源码构建

在 RK3588 Linux 板上构建（rustup 会按 `rust-toolchain.toml` 自动安装 Rust 1.95）：

```sh
sudo apt install build-essential cmake libclang-dev git
git clone --depth 1 https://github.com/happyme531/open_rknpu2_lm_infer.git
cd open_rknpu2_lm_infer
git submodule update --init --depth 1
cargo build --release --locked
```

指定模型目录后即可使用：

```sh
export RKNPU_MODEL=/path/to/Qwen3-4B-W4A8-Mixed-QQQ-Rot

# 单次流式生成
target/release/open_rknpu2_lm_infer run '你好，请介绍一下自己。' --conversation -n 128
# 终端聊天
target/release/open_rknpu2_lm_infer chat
# OpenAI 兼容服务
target/release/open_rknpu2_lm_infer serve --port 8080
```

服务启动后，可以直接用标准 OpenAI SDK 调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
stream = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "用两句话介绍 RK3588。"}],
    stream=True,
    extra_body={"enable_thinking": False},
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## TODO

- 支持上游的`rocket`内核态驱动。可能需要等到相关的NPU DVFS完善。
- 目前的Prefill MFU较好情况下可以达到20%左右，仍有改进空间。
- 支持更多模型，包括非LLM的Transformer模型。
- 你也可以动手来改进这个项目。欢迎fork和pr！

## 许可证

[AGPL-3.0-only](LICENSE)。算子库的来源与许可边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
