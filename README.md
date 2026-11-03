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
- 🧠 **MoE with lower memory use**: Compatible MoE models can keep experts resident or stream packed expert weights from disk, with per-layer NPU caching and prefetching.
- ⚡ **Speculative decoding**: Generate candidate tokens with Qwen3.5 MTP or a matching DFlash/DSpark draft model, then verify them with the target model.
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

#### Long Context (MTP Disabled)

| Context length | Prefill tps | Decode128 tps | Peak RSS (GiB) |
| --- | ---: | ---: | ---: |
| 4K | 158.3 | 9.21 | 3.69 |
| 8K | 149.0 | 8.57 | 3.70 |
| 16K | 135.6 | 7.59 | 4.02 |
| 32K | 115.3 | 6.19 | 4.79 |
| 64K | 82.0 | 4.42 | 6.49 |

Note: rknn-llm supports a maximum context length of 16K.

### Qwen3-4B

| Test | rknn-llm 1.3.1 | open_rknpu2_lm_infer | Speedup over vendor SDK | +Dual ubatch scheduling | Speedup over vendor SDK |
| --- | ---: | ---: | ---: | ---: | ---: |
| pp512 | 105.037 ± 0.031 | 191.053 ± 0.154 | +81.9% | 222.079 ± 0.569 | +111.43% |
| pp1024 | 94.619 ± 0.021 | 182.039 ± 0.020 | +92.4% | 211.770 ± 0.123 | +123.81% |
| pp4096 | 50.133 ± 0.084 | 141.488 ± 0.047 | +182.2% | 163.069 ± 0.008 | +225.27% |
| decode128 @512 | 5.772 ± 0.002 | 10.726 ± 0.001 | +85.8% | 10.667 ± 0.051 | +84.81% |
| decode128 @1024 | 5.125 ± 0.004 | 10.311 ± 0.006 | +101.2% | 10.284 ± 0.002 | +100.66% |
| decode128 @3800 | 2.310 ± 0.005 | 8.544 ± 0.002 | +269.9% | 8.525 ± 0.000 | +269.05% |

### MiniCPM-V 4.7 35B-A3B (Text, DSpark)

Measured with two complete coding requests from Arena-Hard-v2, containing 498 and 2110 tokens including the chat template, with one run per configuration and prompt. Routed experts use W4A8, the shared FFN and output head use W8A8, and the DSpark draft uses FP16. All runs use default operators. Throughput is in tok/s.

| Configuration | pp498 | decode128 @498 | pp2110 | decode128 @2110 | Peak RSS (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| All experts resident + DSpark 3 | 59.00 | 8.41 | 63.06 | 9.31 | 19.67 / 20.12 |
| Streaming + DSpark 3 | 33.98 | 4.92 | 44.44 | 5.74 | 5.46 / 5.88 |
| Streaming + DSpark 3 + expert skipping | 41.19 | 7.89 | 54.78 | 8.02 | 5.34 / 5.69 |

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

| Architecture (`model_type`) | Checkpoint / status |
|---|---|
| qwen3 | [Qwen3-4B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/Qwen3-4B-W4A8-Mixed-QQQ-Rot) |
| qwen3_5 / qwen3_5_text | [Qwen3.5-4B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot) |
| qwen3_5_moe (text component only) / qwen3_5_moe_text |  |
| minicpmv4_7 (text component only) | [MiniCPM-V-4.7-35B-A3B-QQQ-W4A8-Mixed-CT](llm_generated_docs/MINICPMV47_SUPPORT.md) (text only) |
| llama | [MiniCPM5-2B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/MiniCPM5-2B-W4A8-Mixed-QQQ-Rot) |
| granite | [Granite-4.2-3B-W4A8-Mixed-QQQ-SQ](https://huggingface.co/happyme531/Granite-4.2-3B-W4A8-Mixed-QQQ-SQ) |

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

## Features

### Image Input and MTP

Use a Qwen3.5-4B model directory containing both vision and MTP weights. The following command uses the K1 configuration from the image-text benchmark:

```sh
./bin/open_rknpu2_lm_infer --model models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot run --spec-type draft-mtp --spec-draft-n-max 1 --image photo.jpg --temp 0 -n 128 --stats -p 'Describe this photo in detail.'
```

Images are passed to the model at their original resolution unless they exceed the model's limits. Different resolutions are supported dynamically, but vision encoding may be slower if the corresponding number of vision tokens has not been tuned.

### Tuning

The default inference strategies do not deliver optimal performance. You can manually tune once for a specific model.

```sh
# Set the big CPU cores to their maximum frequency. Strongly recommended.
echo "performance" > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo "performance" > /sys/devices/system/cpu/cpufreq/policy6/scaling_governor
./bin/open_rknpu2_lm_infer tune --model models/Qwen3-4B-W4A8-Mixed-QQQ-Rot
# Tune batches used by speculative decoding to improve its speed
#./bin/open_rknpu2_lm_infer tune --model models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot --batches 1,2,4,8,128
# Tune the vision encoder for a specific input resolution
#./bin/open_rknpu2_lm_infer tune --model models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot --vision --image photo.jpg
```

The first tuning run can take several minutes and shows progress and the best candidate time. Valid cache entries are reused on subsequent runs.

Use the same `--tune-cache PATH` for tuning and inference. Without an explicit path, the program uses systemd's `CACHE_DIRECTORY` (the first directory when several are listed), then `$XDG_CACHE_HOME/open_rknpu2_lm_infer`, then `$HOME/.cache/open_rknpu2_lm_infer`. With none of these set, it uses `MODEL/.cache/open_rknpu2_lm_infer`.

### MoE Expert Storage and Prefetching

Models such as Qwen3.5 MoE and MiniCPM-V 4.7 support two expert storage modes:

- **Resident** (default): Load routed experts into memory.
- **Streaming**: Keep only some routed experts in memory and load the others from disk on demand, substantially reducing memory use for expert weights.

For example, with a compatible quantized MiniCPM-V 4.7 checkpoint:

```sh
MODEL=/path/to/MiniCPM-V-4.7-35B-A3B-QQQ-W4A8-Mixed-CT
EXPERT_STORE=/path/to/expert-store # A fast drive connected via M.2 PCIe 3.0 x4 is strongly recommended
./bin/open_rknpu2_lm_infer -m "$MODEL" --expert-storage stream \
  --expert-store "$EXPERT_STORE" --expert-cache-percent 8 \
  run --conversation --temp 0 -n 128 --stats -p 'Please introduce yourself.'
```

Streaming mode requires `--expert-store`. The first startup prepares packed NPU records from the checkpoint, requiring extra time and disk space; subsequent runs reuse a matching expert store. Expert records for the tested MiniCPM-V 4.7 35B-A3B checkpoint occupy approximately 15.2 GiB, in addition to the original model files. Actual disk use depends on the model.

`--expert-cache-percent` controls the percentage of routed experts to cache and defaults to `8`. A larger cache reduces disk reads and improves performance at the cost of more memory.

### Prefix Caching

Prompt prefix reuse is independent of the disk expert store and NPU expert cache. The server enables prefix caching by default. Disable it server-wide with `serve --no-cache-prompt`, or for an individual OpenAI-compatible request with `extra_body={"cache_prompt": False}`. The number of reused tokens is returned in `usage.prompt_tokens_details.cached_tokens`.

For models with SSM layers, `--ctx-checkpoints` sets the maximum number of state checkpoints (default: 32), and `--checkpoint-min-step` sets their minimum spacing (default: 8192 tokens). `--ctx-checkpoints 0` disables these state checkpoints while leaving the server's prompt cache enabled.

### Speculative Decoding

Speculative decoding is disabled by default. Select a compatible target model and draft for `run`, `chat`, or `serve`.

| `--spec-type` | Target model and draft |
|---|---|
| `draft-mtp` | A Qwen3.5 checkpoint containing supported MTP layers; no separate draft directory is needed. |
| `draft-dflash` | An unrotated Qwen3.5 target model and a matching DFlash draft. The validated pairing is Qwen3.5-4B C4 with [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash); the QQQ-Rot checkpoint in the model table is incompatible with this draft. |
| `draft-dspark` | A MiniCPM-V 4.7 35B-A3B text target model and the matching [DSpark draft](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark). |

For example, with a Qwen3.5-4B checkpoint containing MTP weights:

```sh
./bin/open_rknpu2_lm_infer -m models/Qwen3.5-4B-W4A8-Mixed-QQQ-Rot \
  --spec-type draft-mtp --spec-draft-n-max 1 \
  run --conversation --temp 0 -n 128 --stats -p 'Please introduce yourself.'
```

For DFlash, first prepare a compatible **unrotated** quantized Qwen3.5-4B C4 target model, then download the draft revision validated by this project:

```sh
hf download z-lab/Qwen3.5-4B-DFlash --revision 9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf --local-dir models/Qwen3.5-4B-DFlash
./bin/open_rknpu2_lm_infer -m /path/to/Qwen3.5-4B-C4 \
  --ctx-size 4096 --spec-type draft-dflash \
  --spec-draft-model models/Qwen3.5-4B-DFlash --spec-draft-n-max 3 \
  run --conversation --temp 0 -n 128 --stats -p 'Please introduce yourself.'
```

For DSpark, reuse the MiniCPM quantized target model and expert store from above:

```sh
hf download openbmb/MiniCPM-V-4.7-35B-A3B-DSpark --local-dir models/MiniCPM-V-4.7-35B-A3B-DSpark
MODEL=/path/to/MiniCPM-V-4.7-35B-A3B-QQQ-W4A8-Mixed-CT
EXPERT_STORE=/path/to/expert-store
./bin/open_rknpu2_lm_infer -m "$MODEL" --expert-storage stream --expert-store "$EXPERT_STORE" \
  --spec-type draft-dspark --spec-draft-model models/MiniCPM-V-4.7-35B-A3B-DSpark \
  --spec-draft-n-max 3 run --conversation --temp 0 -n 128 --stats -p 'Please introduce yourself.'
```

`--spec-draft-n-max` sets the maximum number of proposed tokens per round (default: 3; set to 0 to disable speculative decoding). MTP and DFlash support up to 7, and DSpark up to 15; the value must also be smaller than `--ubatch-size` and comply with the draft checkpoint's limits. `--spec-draft-model` also accepts the aliases `--model-draft` and `-md`. Use `--spec-draft-p-min` to truncate low-confidence proposals, and `--spec-draft-sampling` to select `greedy` (default) or `probabilistic` draft sampling. `run --stats` reports proposed and accepted token counts, along with draft and verification times.

### Dual Ubatch Prefill Scheduling

Use `--prefill-ubatches 2` to overlap two physical ubatches from the same prompt in a layer-wise pipeline. The number of tokens in each ubatch is still controlled by `--ubatch-size`.

```sh
./bin/open_rknpu2_lm_infer --model MODEL --ubatch-size 128 --prefill-ubatches 2 bench -p 4096 -n 0
```

This can significantly improve prefill speed, but requires additional memory for activations and internal buffers (typically ~1.5 GB), so it is disabled by default. Qwen3.5 MoE, MiniCPM-V 4.7, and Ling require `--prefill-ubatches 1`.

### Hardware-Accelerated INT8 Weight Compression

Use `--compress-w8a8` to enable lossless compression of W8A8 linear-layer weights. This can substantially reduce memory use without affecting inference accuracy, but may slightly reduce inference speed and significantly increase server startup time.

## Architecture

```
   CLI  ·  TUI  ·  OpenAI API (axum)
                 │
   Engine ── Dedicated inference thread · Request queue · Sampling · Prefix KV reuse
                 │
   LanguageModel ── Dense / Qwen3.5 / Qwen3.5 MoE / MiniCPM-V 4.7 / Ling*
        │                         │
   CPU (Rust + C++)          NPU (rk_npu_matmul, C ABI)
   Norm · RoPE · Conv · GDN   W4A8/W8A8 Linear · MoE · Attention · ViT · GDN
                                  │
                         Open-source RKNPU2 kernel driver
```


*Ling requires an optional build feature. Shared CPU operators live in `src/ops/`, reusable GDN layers and state transactions in `src/layers/`, and checkpoint loading and model execution in `src/model/`. See the [module layout](llm_generated_docs/RUNTIME_MODULE_LAYOUT.md).

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

To add Ling alongside the default model families, build with `cargo build --release --locked --features model-ling3tiny`.

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
