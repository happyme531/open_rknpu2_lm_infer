<div align="center">

# open_rknpu2_lm_infer

**A fully open-source, high-performance LLM inference engine for the RK3588 NPU**

![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Rust](https://img.shields.io/badge/rust-1.95-orange)
![Platform](https://img.shields.io/badge/platform-RK3588-green)

[![Proudly Vibe Coded - Plasma Mix](https://vibecoded.fyi/badges/flat/main/proudly-vibe-coded-plasma-mix.svg)](https://vibecoded.fyi/)
[![Coded with Codex](https://vibecoded.fyi/badges/flat/agents/codex.svg)](https://vibecoded.fyi/)

[English](README.md) | [简体中文](README_zh.md)

</div>

## Highlights

- 🚀 **Faster than the vendor SDK**: Autotuning, heterogeneous pipelining, dataflow optimization, and zero-copy execution deliver over 50% higher average inference speed than the vendor's rknn-llm SDK, with larger gains at longer contexts: up to **3.25×** for prefill (with dual ubatches) and **3.70×** for decode. With a 3,800-token history, Qwen3-4B generates 128 tokens in 15.0 seconds, down from 55.4 seconds.
- 🎯 **Better accuracy than the vendor SDK**: Advanced quantization algorithms reduce WikiText-2 perplexity by 2.17% for Qwen3-4B and 2.28% for MiniCPM5-2B compared with rknn-llm W8A8.
- 🔓 **Fully open source**: Built on our high-performance, open-source RKNPU2 operator library, [`rknpu2_matmul_open`](https://github.com/happyme531/rknpu2_matmul_open/). The entire inference pipeline runs without proprietary vendor libraries, so you can study it, optimize it, and add support for new models.
- 📦 **Mixed W4/W8 quantization**: A mix of 4-bit and 8-bit weights with dynamic 8-bit activations uses less memory than the vendor's W8A8 quantization. Weights are loaded on demand through mmap; Qwen3-4B reaches a peak RSS of approximately 3.9 GiB with a 4K context.
- 🧩 **Ready to use**: A single binary provides a CLI, a terminal chat TUI, and an OpenAI-compatible API, with support for streaming, thinking, function calling, and prefix caching.

## Performance

### Qwen3.5-4B

| Test | rknn-llm 1.3.1 | open_rknpu2_lm_infer (MTP1) | Speedup over vendor SDK | +Dual ubatch scheduling | Speedup over vendor SDK |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vision encoding 448×448 (ms) | 1863 | 862.25 | +116.1% | — | — |
| pp512 (tps) | 58.347 | 159.266 ± 0.330 | +173.0% | 178.532 ± 0.409 | +205.98% |
| pp1024 | 57.844 | 159.215 ± 0.172 | +175.2% | 179.157 ± 0.230 | +209.72% |
| pp3800 | 54.029 | 153.458 ± 0.126 | +184.0% | 171.250 ± 0.242 | +216.96% |
| decode128 @512 (tps) | 5.538 | 11.453 ± 0.001 | +106.8% | 11.471 ± 0.002 | +107.14% |
| decode128 @1024 | 5.416 | 11.409 ± 0.003 | +110.7% | 11.436 ± 0.002 | +111.16% |
| decode128 @3800 | 4.565 | 10.661 ± 0.002 | +133.5% | 10.674 ± 0.006 | +133.83% |

### Qwen3-4B

| Test | rknn-llm 1.3.1 | open_rknpu2_lm_infer | Speedup over vendor SDK | +Dual ubatch scheduling | Speedup over vendor SDK |
| --- | ---: | ---: | ---: | ---: | ---: |
| pp512 | 105.037 ± 0.031 | 191.053 ± 0.154 | +81.9% | 222.079 ± 0.569 | +111.43% |
| pp1024 | 94.619 ± 0.021 | 182.039 ± 0.020 | +92.4% | 211.770 ± 0.123 | +123.81% |
| pp4096 | 50.133 ± 0.084 | 141.488 ± 0.047 | +182.2% | 163.069 ± 0.008 | +225.27% |
| decode128 @512 | 5.772 ± 0.002 | 10.726 ± 0.001 | +85.8% | 10.667 ± 0.051 | +84.81% |
| decode128 @1024 | 5.125 ± 0.004 | 10.311 ± 0.006 | +101.2% | 10.284 ± 0.002 | +100.66% |
| decode128 @3800 | 2.310 ± 0.005 | 8.544 ± 0.002 | +269.9% | 8.525 ± 0.000 | +269.05% |

<details>
<summary>Historical benchmark results</summary>

## Historical Results (the current version may be faster)

### Granite-4.2-3B

| Test | rknn-llm 1.3.1 | open_rknpu2_lm_infer | Speedup |
|---|---:|---:|---:|
| pp512 | 120.459 ± 0.107 | 182.476 ± 0.114 | +51.5% |
| pp1024 | 106.277 ± 0.019 | 164.162 ± 0.240 | +54.5% |
| pp4096 | 54.404 ± 0.004 | 110.307 ± 0.066 | +102.8% |
| decode128 @512 | 6.806 ± 0.003 | 9.295 ± 0.012 | +36.6% |
| decode128 @1024 | 6.370 ± 0.006 | 9.054 ± 0.004 | +42.1% |
| decode128 @3800 | 3.932 ± 0.001 | 7.727 ± 0.003 | +96.5% |

### MiniCPM5-2B

| Test | rknn-llm 1.3.1 | open_rknpu2_lm_infer | Speedup |
|---|---:|---:|---:|
| pp512 | 211.052 ± 0.484 | 245.465 ± 0.279 | +16.3% |
| pp1024 | 188.643 ± 0.082 | 219.865 ± 0.220 | +16.6% |
| pp4096 | 98.874 ± 0.009 | 173.578 ± 0.034 | +75.6% |
| decode128 @512 | 9.431 ± 0.006 | 13.151 ± 0.029 | +39.4% |
| decode128 @1024 | 8.985 ± 0.003 | 12.954 ± 0.005 | +44.2% |
| decode128 @3800 | 7.117 ± 0.003 | 11.613 ± 0.012 | +63.2% |

</details>

## Perplexity

| Model | rknn-llm W8A8 | open_rknpu2_lm_infer mixed quantization | PPL change (lower is better) |
|---|---:|---:|---:|
| Qwen3.5-4B | 8.5197 | 8.5744 | +0.64% |
| Qwen3-4B | 12.8709 | 12.5913 | -2.17% |
| Granite-4.2-3B | 15.7448 | 16.0519 | +1.95% |
| MiniCPM5-2B | 14.1293 | 13.8073 | -2.28% |

Qwen3/Qwen3.5 PPL was remeasured on 2026-10-25 using the first 32 independent 512-token windows of WikiText-2 raw test, scoring target positions 257–511 (8,160 targets in total), with state reset for each window and speculative decoding disabled. See [checkpoints and per-token NLL](llm_generated_docs/README_REBENCH_20261025.md#perplexity).

## Supported Models

| Architecture | Validated checkpoint |
|---|---|
| qwen3 | [Qwen3-4B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/Qwen3-4B-W4A8-Mixed-QQQ-Rot) |
| granite | [Granite-4.2-3B-W4A8-Mixed-QQQ-SQ](https://huggingface.co/happyme531/Granite-4.2-3B-W4A8-Mixed-QQQ-SQ) |
| llama | [MiniCPM5-2B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/MiniCPM5-2B-W4A8-Mixed-QQQ-Rot) |
| qwen3_5 / qwen3_5_text | [Qwen3.5-4B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot) |
| bailing_hybrid (Ling3tiny) | [Ling-3.0-tiny-QQQ-W4A8-Mixed-CT](llm_generated_docs/LING3TINY_SUPPORT.md) |

The engine supports models in the compressed-tensors format with W4A8 or W8A8 linear layers, including mixed configurations, using symmetric per-channel weight and per-token activation quantization. Use the models above as examples to try other model sizes, mixed quantization configurations, and quantization algorithms.

## Quick Start

You need an RK3588 device running Linux. The current version supports only the vendor's `rknpu2` kernel driver, version 0.9.6 or newer.

Run the following command as root to check the driver version:

```bash
cat /sys/kernel/debug/rknpu/version
RKNPU driver: v0.9.8
```

Then copy and paste the following commands to download the latest [release](https://github.com/happyme531/open_rknpu2_lm_infer/releases/latest) and the Qwen3-4B model, then start a terminal chat session. Supports Debian 11 / glibc 2.31 or newer, with no compilation required.

```bash
sudo apt install curl jq python3-pip libgomp1 libstdc++6

curl -fL "$(curl -fsSL https://api.github.com/repos/happyme531/open_rknpu2_lm_infer/releases/latest | jq -r '.assets[] | select(.name | endswith("-linux-aarch64.tar.gz")) | .browser_download_url')" | tar -xz --strip-components=1
pip3 install 'huggingface_hub[cli]>=0.34,<1'
hf download happyme531/Qwen3-4B-W4A8-Mixed-QQQ-Rot --local-dir models/Qwen3-4B-W4A8-Mixed-QQQ-Rot
# Set the big CPU cores to their maximum frequency. Optional, but skipping this can significantly reduce inference speed. Root privileges may be required.
# echo "performance" > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
# echo "performance" > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
./bin/open_rknpu2_lm_infer --model models/Qwen3-4B-W4A8-Mixed-QQQ-Rot chat
```

Replace `chat` in the last command with `serve --port 8080` to start an OpenAI-compatible LLM API server.

Use `--compress-w8a8` to losslessly compress all W8A8 linear weights. Tiles stay raw unless their encoded size is smaller. See [W8A8 weight compression](llm_generated_docs/W8A8_LOSSLESS_COMPRESSION.md) for startup and memory accounting details.

### Image Input and MTP

Use a Qwen3.5-4B model directory containing both vision and MTP weights. The following command uses the K1 configuration from the image-text benchmark:

```sh
./bin/open_rknpu2_lm_infer --model models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot run --spec-type draft-mtp --spec-draft-n-max 1 --image photo.jpg --temp 0 -n 128 --stats -p 'Describe this photo in detail.'
```

Images are passed to the model at their original resolution unless they exceed the model's limits. Different resolutions are supported dynamically, but vision encoding may be slower if the corresponding number of vision tokens has not been tuned.

## Getting full tuned performance

The default inference strategies do not deliver optimal performance. You can manually tune once for a specific model.

```sh
# Set the big CPU cores to their maximum frequency. Strongly recommended.
echo "performance" > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo "performance" > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
./bin/open_rknpu2_lm_infer tune --model models/Qwen3-4B-W4A8-Mixed-QQQ-Rot
# Tune batches used by speculative decoding to improve its speed
#./bin/open_rknpu2_lm_infer tune --model models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot --batches 1,2,4,128
# Tune the vision encoder for a specific input resolution
#./bin/open_rknpu2_lm_infer tune --model models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot --vision --image photo.jpg
```

The first tuning run can take several minutes and shows progress and the best candidate time. Valid cache entries are reused on subsequent runs.

`run`, `chat`, and `serve` show prefill progress, cached token counts, throughput, and ETA by default, followed by recent and average decode speed. With MTP/DFlash enabled, the display also includes acceptance, tokens per round, round count, and average draft/verify time. Throughput measures complete model calls; decode excludes the first token produced by prefill. Recent speed uses complete rounds from roughly the last second.

`run` writes status to stderr and generated text to stdout; `chat` uses a separate status area. `serve` logs progress on the server, preserving the HTTP/SSE response format. Redirected logs use plain text at five-second intervals, plus phase transitions and completion. Vision tuning shows overall shape progress alongside the native candidate search progress.

A shape used without an applicable tuning profile triggers one notice per process. Target, DFlash, and vision projections include a tuning command; MTP floating projections identify the missing CLI tuning support. Partial prefill batches, including checkpoint splits, do not trigger the notice. Explicit `--no-tune` also suppresses it.

### Dual Ubatch Prefill Scheduling

Use `--prefill-ubatches 2` to overlap two physical ubatches from the same prompt in a layer-wise pipeline. The number of tokens in each ubatch is still controlled by `--ubatch-size`.

```sh
./bin/open_rknpu2_lm_infer --model MODEL --ubatch-size 128 --prefill-ubatches 2 bench -p 4096 -n 0
```

This can significantly improve prefill speed, but requires additional memory for activations and internal buffers (typically ~1.5 GB), so it is disabled by default.

## Architecture

```
   CLI  ·  TUI  ·  OpenAI API (axum)
                 │
   Engine ── Dedicated inference thread · Request queue · Sampling · Prefix KV reuse
                 │
   LanguageModel ── Qwen3 / Llama / Granite / Qwen3.5 (vision + MTP)
        │                         │
   CPU (Rust + C++)          NPU (rk_npu_matmul, C ABI)
   Norm · RoPE · Conv · GDN   W4A8/W8A8 Linear · Attention · ViT · GDN
                                  │
                         Open-source RKNPU2 kernel driver
```


Shared CPU operators live in `src/ops/`, reusable GDN layers and state transactions in `src/layers/`, and checkpoint loading and model execution in `src/model/`. Dense and experimental Qwen3.5 share output projections, tracing and quantized projection components. See the [module layout](llm_generated_docs/RUNTIME_MODULE_LAYOUT.md).

## Development

### Build from Source

Build on an RK3588 Linux board (rustup automatically installs Rust 1.95 as specified in `rust-toolchain.toml`):

```sh
sudo apt install build-essential cmake libclang-dev git
git clone --depth 1 https://github.com/happyme531/open_rknpu2_lm_infer.git
cd open_rknpu2_lm_infer
git submodule update --init --depth 1
cargo build --release --locked
```

Set the model directory, then run:

```sh
export RKNPU_MODEL=/path/to/Qwen3-4B-W4A8-Mixed-QQQ-Rot

# Single streaming generation
target/release/open_rknpu2_lm_infer run 'Hello, please introduce yourself.' --conversation -n 128
# Terminal chat
target/release/open_rknpu2_lm_infer chat
# OpenAI-compatible server
target/release/open_rknpu2_lm_infer serve --port 8080
```

Once the server is running, call it with the standard OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
stream = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "Introduce the RK3588 in two sentences."}],
    stream=True,
    extra_body={"enable_thinking": False},
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## TODO

- Support the upstream `rocket` kernel driver. This may need to wait until NPU DVFS support is more complete.
- Prefill MFU currently reaches around 20% under favorable conditions, with room for improvement.
- Support more models, including Transformer models beyond LLMs.
- Contributions are welcome: fork the project and submit a PR!

## License

[AGPL-3.0-only](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the operator library's provenance and licensing boundaries.
