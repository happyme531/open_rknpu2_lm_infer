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

- 🚀 **比原厂更快**：采用自动调优，异构流水执行，数据流优化，零拷贝等先进策略，平均推理速度比原厂 rknn-llm SDK 高50%以上，上下文越长优势越大，prefill 最高 **3.25×**（双 ubatch），decode 最高 **3.70×**。在 3800-token 历史上，Qwen3-4B 的 128 步 decode 耗时从 55.4 秒降到 15.0 秒。
- 🎯 **比原厂更准**：采用先进的量化算法，Qwen3-4B 和 MiniCPM5-2B 的 WikiText-2 PPL 分别比 rknn-llm W8A8 低 2.17% 和 2.28%。
- 🔓 **完全开源**：基于自研高性能开源 RKNPU2 算子库 [`rknpu2_matmul_open`](https://github.com/happyme531/rknpu2_matmul_open/)，整个推理流程无任何原厂闭源库依赖，可随意研究/优化/添加新模型支持。
- 📦 **混合 W4/W8 量化**：4-bit/8-bit 权重配合 8-bit 动态激活的混合量化，相比原厂W8A8量化占用内存更低。权重 mmap 按需加载，Qwen3-4B 在 4K 上下文下峰值 RSS 约 3.9 GiB。
- 🧠 **低内存运行 MoE**：兼容的 MoE 模型可选择专家常驻，或从磁盘流式加载打包后的专家权重，并使用逐层 NPU 缓存与预取。
- ⚡ **投机解码**：使用 Qwen3.5 MTP 或匹配的 DFlash/DSpark 草稿模型生成候选 token，再由目标模型验证。
- 🧩 **开箱即用**：一个二进制同时提供 CLI、终端聊天 TUI 和 OpenAI 兼容 API，支持流式输出、thinking、function calling 和前缀缓存。

## 性能

### Qwen3.5-4B

| 测试 | rknn-llm 1.3.1 | open_rknpu2_lm_infer(MTP1) | 相比原厂 SDK 性能提升 | +双 ubatch 调度 | 性能提升 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 视觉编码 448×448 (ms) | 1863 | 862.25 | +116.1% | — | — |
| pp512 (tps) | 58.347 | 159.266 ± 0.330 | +173.0% | 178.532 ± 0.409 | +205.98% |
| pp1024 | 57.844 | 159.215 ± 0.172 | +175.2% | 179.157 ± 0.230 | +209.72% |
| pp3800 | 54.029 | 153.458 ± 0.126 | +184.0% | 171.250 ± 0.242 | +216.96% |
| decode128 @512 (tps) | 5.538 | 11.453 ± 0.001 | +106.8% | 11.471 ± 0.002 | +107.14% |
| decode128 @1024 | 5.416 | 11.409 ± 0.003 | +110.7% | 11.436 ± 0.002 | +111.16% |
| decode128 @3800 | 4.565 | 10.661 ± 0.002 | +133.5% | 10.674 ± 0.006 | +133.83% |

#### 长上下文（关闭 MTP）

| 上下文长度 | Prefill tps | Decode128 tps | 峰值内存占用 (RSS GiB) |
| --- | ---: | ---: |  ---: |
| 4K | 158.3 | 9.21 | 3.69 |
| 8K | 149.0 | 8.57 | 3.70 |
| 16K | 135.6 | 7.59 | 4.02 |
| 32K | 115.3 | 6.19 | 4.79 |
| 64K | 82.0 | 4.42 | 6.49 |

(提示: rknn-llm框架最大仅支持到16K上下文)

### Qwen3-4B

| 测试 | rknn-llm 1.3.1 | open_rknpu2_lm_infer | 相比原厂 SDK 性能提升 | +双 ubatch 调度 | 性能提升 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pp512 | 105.037 ± 0.031 | 191.053 ± 0.154 | +81.9% | 222.079 ± 0.569 | +111.43% |
| pp1024 | 94.619 ± 0.021 | 182.039 ± 0.020 | +92.4% | 211.770 ± 0.123 | +123.81% |
| pp4096 | 50.133 ± 0.084 | 141.488 ± 0.047 | +182.2% | 163.069 ± 0.008 | +225.27% |
| decode128 @512 | 5.772 ± 0.002 | 10.726 ± 0.001 | +85.8% | 10.667 ± 0.051 | +84.81% |
| decode128 @1024 | 5.125 ± 0.004 | 10.311 ± 0.006 | +101.2% | 10.284 ± 0.002 | +100.66% |
| decode128 @3800 | 2.310 ± 0.005 | 8.544 ± 0.002 | +269.9% | 8.525 ± 0.000 | +269.05% |

### MiniCPM-V 4.7 35B-A3B（文本，DSpark）

使用 Arena-Hard-v2 的两条完整代码请求，含聊天模板分别为 498 / 2110 token，每项单次。路由专家 W4A8，共享 FFN 与输出头 W8A8，DSpark 草稿 FP16。均为默认操作。速度单位为 tok/s。

| 配置 | pp498 | decode128 @498 | pp2110 | decode128 @2110 | 峰值 RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全驻留＋DSpark 3 | 59.00 | 8.41 | 63.06 | 9.31 | 19.67 / 20.12 |
| 流式加载＋DSpark 3 | 33.98 | 4.92 | 44.44 | 5.74 | 5.46 / 5.88 |
| 流式加载＋DSpark 3＋专家跳过 | 41.19 | 7.89 | 54.78 | 8.02 | 5.34 / 5.69 |

<details>
<summary>历史评测数据</summary>

## 历史数据（当前版本可能更快）

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

</details>

## Perplexity

| 模型 | rknn-llm W8A8 | open_rknpu2_lm_infer 混合量化 | PPL 变化（越低越好） |
|---|---:|---:|---:|
| Qwen3.5-4B | 8.5197 | 8.5744 | +0.64% |
| Qwen3-4B | 12.8709 | 12.5913 | -2.17% |
| Granite-4.2-3B | 15.7448 | 16.0519 | +1.95% |
| MiniCPM5-2B | 14.1293 | 13.8073 | -2.28% |

Qwen3/Qwen3.5 PPL 于 2026-10-25 重测，使用 WikiText-2 raw test 的前 32 个独立 512-token 窗口，评分 target 位置 257–511（共 8160 个目标），每窗重置状态并关闭推测解码。[Checkpoint 与逐 token NLL](llm_generated_docs/README_REBENCH_20261025.md#perplexity)。

## 支持的模型

| 架构(model_type) | checkpoint / 状态 |
|---|---|
| qwen3 | [Qwen3-4B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/Qwen3-4B-W4A8-Mixed-QQQ-Rot) |
| qwen3_5 / qwen3_5_text | [Qwen3.5-4B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot) |
| qwen3_5_moe (仅文本部分) / qwen3_5_moe_text |  |
| minicpmv4_7 (仅文本部分) | [MiniCPM-V-4.7-35B-A3B-QQQ-W4A8-Mixed-CT](llm_generated_docs/MINICPMV47_SUPPORT.md)（仅文本） |
| llama | [MiniCPM5-2B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/MiniCPM5-2B-W4A8-Mixed-QQQ-Rot) |
| granite | [Granite-4.2-3B-W4A8-Mixed-QQQ-SQ](https://huggingface.co/happyme531/Granite-4.2-3B-W4A8-Mixed-QQQ-SQ) |

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

可选 `--compress-w8a8` 对所有 W8A8 线性层权重做无损压缩，仅在 tile 体积严格变小时使用压缩存储，详见 [W8A8 权重无损压缩](llm_generated_docs/W8A8_LOSSLESS_COMPRESSION.md)。

## 功能一览

### 图像输入与MTP

将 `MODEL` 指向包含视觉与 MTP 权重的 Qwen3.5-4B 模型目录。以下使用图文基准中的 K1 配置：

```sh
./bin/open_rknpu2_lm_infer --model models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot run --spec-type draft-mtp --spec-draft-n-max 1 --image photo.jpg --temp 0 -n 128 --stats -p '请详细描述这张照片。'
```

图像按原图分辨率输入模型，除非超出了模型允许的上限。可以动态输入不同分辨率的图片，不过如果对应的视觉token数量没有调优，视觉编码器速度可能会较慢。

### 调优

推理程序使用的默认的推理策略并非最优性能。可以针对指定的模型手动调优一次。

```sh
# 将大核CPU频率固定为最高。强烈建议执行。
echo "performance" > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo "performance" > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
./bin/open_rknpu2_lm_infer tune --model models/Qwen3-4B-W4A8-Mixed-QQQ-Rot
# 调优推测解码相关的batch，提升推测解码的速度
#./bin/open_rknpu2_lm_infer tune --model models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot --batches 1,2,4,8,128
# 使用指定输入分辨率调优视觉编码器
#./bin/open_rknpu2_lm_infer tune --model models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot --vision --image photo.jpg
```

首次调优可能需要数分钟，会显示进度和最佳候选耗时；之后重复运行会复用有效缓存。

`--batches` 同时准备 decoder 和量化输出头的调优配置。开启投机解码时，应包含验证实际使用的 bucket，例如最多6个 draft token 需要 M=8 的输出头配置。普通生成只准备 decode 和 verify 使用的输出头 shape。

调优和推理应使用相同的 `--tune-cache PATH`。未显式指定时，依次使用 systemd 的 `CACHE_DIRECTORY`（配置多个目录时取第一个）、`$XDG_CACHE_HOME/open_rknpu2_lm_infer`、`$HOME/.cache/open_rknpu2_lm_infer`；这些变量均未设置时，使用 `MODEL/.cache/open_rknpu2_lm_infer`。启动日志会显示实际路径以及初始缓存命中、缺失和 fallback shape 数量。

systemd 服务可在 `[Service]` 中设置 `CacheDirectory=open_rknpu2_lm_infer`，提供持久且可写的缓存。使用服务用户执行 `tune`，指定 `--tune-cache /var/cache/open_rknpu2_lm_infer`，并保持 CPU/NPU 设置一致。模型目录只读时，应设置 `CacheDirectory` 或显式指定可写的 `--tune-cache`。推理读取已有配置，缺失配置由 `tune` 生成。

`run`、`chat` 和 `serve` 默认显示 prefill 进度、缓存命中 token 数、处理速度及 ETA，并在 decode 时显示近期和平均速度。开启投机解码时，还会显示接受率、平均每轮产出、轮数及 draft/verify 平均耗时。DSpark 目前在进度显示中标为 `DFlash`，`--stats` 中的类型为 `draft-dspark`。速度按完整模型调用统计，decode 不包含 prefill 产生的首 token；近期速度取最近约一秒的完整轮次。

`run` 的状态写入 stderr，生成正文写入 stdout；`chat` 在独立状态区显示；`serve` 在服务端终端记录状态，HTTP/SSE 格式保持不变。重定向日志使用每五秒一次的纯文本更新，同时记录阶段切换和结束状态。视觉调优同时显示整体 shape 进度和当前 shape 的候选搜索进度。

发现实际使用的 shape 缺少可用调优缓存时，每个进程只提示一次。target、DFlash 和视觉投影提示对应调优命令；MTP 浮点投影会说明当前 CLI 未提供对应调优入口。prefill 尾块及 checkpoint 切分产生的部分块不触发提示；`--no-tune` 主动禁用调优时也不提示。

### MoE 专家存储与预取

Qwen3.5 MoE、MiniCPM-V 4.7 等模型支持两种专家存储模式：

- **常驻**（默认）：将路由专家加载到内存。
- **流式**：只在内存保留部分路由专家，其余专家按需从磁盘加载。大幅降低专家权重的内存需求。

例如，使用兼容的 MiniCPM-V 4.7 量化 checkpoint：

```sh
MODEL=/path/to/MiniCPM-V-4.7-35B-A3B-QQQ-W4A8-Mixed-CT
EXPERT_STORE=/path/to/expert-store # 强烈建议使用m.2 pcie3.0 x4接口连接的高速硬盘
./bin/open_rknpu2_lm_infer -m "$MODEL" --expert-storage stream \
  --expert-store "$EXPERT_STORE" --expert-cache-percent 8 \
  run --conversation --temp 0 -n 128 --stats -p '请介绍一下自己。'
```

流式模式必须指定 `--expert-store`。首次启动会从 checkpoint 准备 NPU 打包记录，需要额外时间和磁盘空间；之后会复用匹配的专家库。已测试的 MiniCPM-V 4.7 35B-A3B checkpoint 的专家记录约占 15.2 GiB，原模型文件还需另计；实际空间取决于模型。

`--expert-cache-percent` 控制缓存的路由专家百分比。默认为 `8`。增大缓存会减少读盘，提高性能，但占用更多内存。

### 前缀缓存

Prompt 前缀复用独立于磁盘专家库和 NPU 专家缓存。服务默认启用前缀缓存；可用 `serve --no-cache-prompt` 对服务关闭，或在单个 OpenAI 兼容请求中设置 `extra_body={"cache_prompt": False}`。实际复用的 token 数在 `usage.prompt_tokens_details.cached_tokens` 中返回。

对于含SSM层的模型，`--ctx-checkpoints` 设置状态检查点数量上限（默认 32），`--checkpoint-min-step` 设置最小间距（默认 8192 token）。`--ctx-checkpoints 0` 只关闭这些状态检查点，不会关闭服务的整个 prompt 缓存。

### 投机解码

投机解码默认关闭，可以为 `run`、`chat` 或 `serve` 选择兼容的目标模型与草稿。

| `--spec-type` | 目标模型与草稿 |
|---|---|
| `draft-mtp` | 包含受支持 MTP 层的 Qwen3.5 checkpoint，不需要单独的草稿目录。 |
| `draft-dflash` | 未旋转的 Qwen3.5 目标模型和匹配的 DFlash 草稿。已验证的组合是 Qwen3.5-4B C4 与 [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash)；模型表中的 QQQ-Rot checkpoint 不兼容该草稿。 |
| `draft-dspark` | MiniCPM-V 4.7 35B-A3B 文本目标模型和匹配的 [DSpark 草稿](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark)。 |

例如，用包含 MTP 权重的 Qwen3.5-4B 运行：

```sh
./bin/open_rknpu2_lm_infer -m models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot \
  --spec-type draft-mtp --spec-draft-n-max 1 \
  run --conversation --temp 0 -n 128 --stats -p '请介绍一下自己。'
```

使用 DFlash 时，先准备兼容的**未旋转** Qwen3.5-4B C4 量化目标模型，再下载项目验证使用的草稿版本：

```sh
hf download z-lab/Qwen3.5-4B-DFlash --revision 9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf --local-dir models/Qwen3.5-4B-DFlash
./bin/open_rknpu2_lm_infer -m /path/to/Qwen3.5-4B-C4 \
  --ctx-size 4096 --spec-type draft-dflash \
  --spec-draft-model models/Qwen3.5-4B-DFlash --spec-draft-n-max 3 \
  run --conversation --temp 0 -n 128 --stats -p '请介绍一下自己。'
```

使用 DSpark 时，沿用上面的 MiniCPM 量化目标模型与专家库：

```sh
hf download openbmb/MiniCPM-V-4.7-35B-A3B-DSpark --local-dir models/MiniCPM-V-4.7-35B-A3B-DSpark
MODEL=/path/to/MiniCPM-V-4.7-35B-A3B-QQQ-W4A8-Mixed-CT
EXPERT_STORE=/path/to/expert-store
./bin/open_rknpu2_lm_infer -m "$MODEL" --expert-storage stream --expert-store "$EXPERT_STORE" \
  --spec-type draft-dspark --spec-draft-model models/MiniCPM-V-4.7-35B-A3B-DSpark \
  --spec-draft-n-max 3 run --conversation --temp 0 -n 128 --stats -p '请介绍一下自己。'
```

`--spec-draft-n-max` 设置每轮最多提议的 token 数（默认 3，设为 0 关闭投机解码）。MTP 和 DFlash 最多为 7，DSpark 最多为 15；该值还必须小于 `--ubatch-size`，并符合草稿 checkpoint 的限制。`--spec-draft-model` 也可写作 `--model-draft` 或 `-md`。`--spec-draft-p-min` 可截断低置信度提议；`--spec-draft-sampling` 可选 `greedy`（默认）或 `probabilistic` 草稿采样。`run --stats` 输出提议数、接受数及草稿和验证耗时。

### 双 ubatch prefill 调度

使用 `--prefill-ubatches 2`，可以按层流水线重叠执行同一 prompt 的两个物理 ubatch。每个 ubatch 的 token 数仍由 `--ubatch-size` 控制。

```sh
./bin/open_rknpu2_lm_infer --model MODEL --ubatch-size 128 --prefill-ubatches 2 bench -p 4096 -n 0
```

开启后可以显著提升 prefill 速度，但会增加额外的激活值和内部缓冲区内存占用（典型 ~1.5 GB），因此默认不启用。Qwen3.5 MoE、MiniCPM-V 4.7 和 Ling 要求 `--prefill-ubatches 1`。

## 架构

```
   CLI  ·  TUI  ·  OpenAI API (axum)
                 │
   Engine ── 专用推理线程 · 请求队列 · 采样 · 前缀 KV 复用
                 │
   LanguageModel ── Dense / Qwen3.5 / Qwen3.5 MoE / MiniCPM-V 4.7 / Ling*
        │                         │
   CPU (Rust + C++)          NPU (rk_npu_matmul, C ABI)
   Norm · RoPE · Conv · GDN   W4A8/W8A8 Linear · MoE · Attention · ViT · GDN
                                  │
                         开源 RKNPU2 内核驱动
```


*Ling 需要可选的构建 feature。通用 CPU 算子位于 `src/ops/`，可复用 GDN 层和状态事务位于 `src/layers/`；`src/model/` 负责 checkpoint 与模型执行。职责边界见 [模块组织说明](llm_generated_docs/RUNTIME_MODULE_LAYOUT.md)。

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

如需在默认模型家族之外加入 Ling，使用 `cargo build --release --locked --features model-ling3tiny` 构建。

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
