# Third-party notices

## Transformers vision references

The Qwen3.5 image patch ordering, vision positions and encoder structure are
implemented against the Qwen team / Hugging Face Transformers reference at
revision `09281b8d56b08199ce9f397c694f38853cc45278` (Apache License 2.0).
`scripts/qwen35/vision_reference.py` and the position checks execute selected
original definitions with lightweight framework wrappers. Downloaded upstream
files retain their copyright/license headers; they are not bundled here.
Source URLs and hashes are in the vision evidence manifest. The Rust image
resize implements the Pillow RGB bicubic arithmetic for processor parity.

Copyright 2024-2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
Upstream license: https://www.apache.org/licenses/LICENSE-2.0

## llama.cpp / ggml

Resident recurrent prefix checkpoints follow `llama-memory-recurrent.cpp` and
`tools/server/server-context.cpp` at upstream revision
`7894e8b92481a8bb8ffe94f73b656ffe0dd01e10`: snapshot recurrent/convolution
state, retain ordinary attention KV, and resume from a checkpoint within the
shared prefix. The Rust implementation also restores its MTP seed and draft KV.
ChatML user/tool delimiter matching follows `common/chat.cpp` and
`common/parsers/qwen3-coder.cpp` at the same revision.
Prompt-cache flag/environment precedence and per-request `cache_prompt`
overrides follow `common/arg.cpp` and `tools/server/server-schema.cpp` at
that revision. Disabling reuse still retains the newly computed prompt state.
Same-position snapshot reuse and current-task refresh adapt the proposal in
[llama.cpp PR #30683](https://github.com/ggml-org/llama.cpp/pull/30683), head
`435211bd898abdaf86655a421ce9e5de83c4d39e`. Restore invalidates future-branch
snapshots before reuse; this retains earlier checkpoints without copying the
same recurrent state again.

The Qwen3.5 NextN graph and normalized hidden-state handoff were checked
against `src/models/qwen35.cpp` and `common/speculative.cpp` at local reference
revision `de3ebb93bd8828496faefb44ed45059ec7ed3ea5`:
[Qwen3.5 graph](https://github.com/ggml-org/llama.cpp/blob/a17e94d229fc7fed1567a799baebd11b771fe5af/src/models/qwen35.cpp)
(unchanged between that upstream revision and the local reference).
The Rust MTP adapter uses the same embedding-first concatenation and gated
full-attention block. Prompt shift and canonical draft repair are additionally
checked against the supplied C7 Transformers reference. The implementation
shares its target decoder and public NPU operators rather than copying ggml.
The speculative CLI names, top-10 draft candidate chain, confidence stop,
greedy sample-and-match and optional probabilistic rejection follow the same
`common/speculative.cpp` reference. Target sampling uses this runtime's RNG.

The FP32 gated delta recurrence in `native/cpu_kernels/`, called through
`src/ops/gated_delta.rs` and `src/ops/gated_delta/native.rs`, adapts the head/window loop and transposed state
organization of `ggml_compute_forward_gated_delta_net_one_chunk` from
[llama.cpp, revision a17e94d229fc7fed1567a799baebd11b771fe5af](https://github.com/ggml-org/llama.cpp/blob/a17e94d229fc7fed1567a799baebd11b771fe5af/ggml/src/ggml-cpu/ops.cpp).
The adapter uses safetensors head ordering, rectangular state
dimensions, FP32 state and compact replay.

`native/cpu_kernels/kda.cpp` and `src/ops/kda.rs` adapt the same operator's
KDA branch: per-key-channel exponential decay, delta update and transposed
FP32 state. The Ling decoder follows the supplied inclusionAI
`Ling-3.0-tiny/modeling_bailing_moe_v3.py` for KDA gate activation,
adjacent-pair RoPE and bias-corrected group-limited sigmoid routing.
That reference is Copyright 2025 Antgroup and The HuggingFace Inc. team,
licensed under the Apache License, Version 2.0:
https://www.apache.org/licenses/LICENSE-2.0

The four-accumulator NEON reduction in `native/cpu_kernels/vec.h` follows
`ggml-cpu/vec.cpp` and `simd-mappings.h` at the same revision. The NEON
SiLU/exp polynomial in `src/ops/activation.rs` adapts
`ggml-cpu/vec.h`, retaining its exact FP32 coefficients; extreme inputs
use the stable scalar implementation. `scripts/gdn/ggml_reference/`
links and calls the actual ggml CPU operator for numerical and timing checks.
`native/cpu_kernels/gdn.cpp` retains both the four-pass ggml loop and a
two-pass fusion, with FP32 state and the same NEON reduction arithmetic.

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## DFlash

The Qwen3.5 DFlash adapter follows z-lab/dflash, revision
`07ebd93db9f472af339b644bb70221ad8428328a`. The original notice is retained
in `src/model/dflash/LICENSE` and reproduced here for binary distributions.

MIT License

Copyright (c) 2026 Z Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
