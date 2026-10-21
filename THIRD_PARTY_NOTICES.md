# Third-party notices

Project contributions are licensed under AGPL-3.0-only; see [LICENSE](LICENSE).

The executable statically links the separately supplied `rk_npu_matmul` library. Preserve that library's [license](native/rk_npu_matmul/LICENSE) and [third-party notices](native/rk_npu_matmul/THIRD_PARTY_NOTICES.md), including the recorded attribution and permission boundary for adapted allbilly/rk3588 code. This project does not change those permissions.

Rust dependencies are pinned in Cargo.lock. Their upstream license and copyright notices must be retained in source or binary distributions. Major dependencies include Hugging Face tokenizers and safetensors, memmap2, half, Rayon, Tokio, axum, clap, ratatui, crossterm, MiniJinja, serde, rand, tracing, bindgen and cmake. Cargo package metadata and each dependency's distributed license files define their respective terms.

The CPU kernels preserve arithmetic from the adjacent native inference demo and the open operator library. Model checkpoints and tokenizer assets are supplied separately under their own licenses. Model weights, vendor runtime binaries and reference documents are excluded from this source distribution.

The Qwen3.5 C++ CPU module in `native/cpu_kernels/` adapts the gated delta loop and NEON helpers from llama.cpp / ggml. The Rust vector SiLU also adapts ggml arithmetic. Their MIT copyright and permission notice is retained in [llm_generated_docs/THIRD_PARTY_NOTICES.md](llm_generated_docs/THIRD_PARTY_NOTICES.md) and must accompany distributions of this code.

The DFlash adapter retains the Z Lab MIT notice in `src/model/dflash/LICENSE` and [llm_generated_docs/THIRD_PARTY_NOTICES.md](llm_generated_docs/THIRD_PARTY_NOTICES.md), which accompanies binary distributions.

The streamed MoE prompt scheduler and adjacent-record staging adapt Strata
`src/prefill/prefill.cpp` and `src/core/expert_source.cpp`, revision
`cbfd5b926b794b356195d3cb600191b0c3567540` (https://github.com/Niko1221/Strata).
The per-layer decode LRU, request protection and prediction-aware eviction
adapt OpenBMB/llama.cpp-omni `src/llama-moe-slot.cpp`, revision
`a64932ec615e1463158129feaf2f0752d0c07a2e`
(https://github.com/OpenBMB/llama.cpp-omni). Their MIT notices are retained in
[src/model/experts/Strata-LICENSE](src/model/experts/Strata-LICENSE) and
[src/model/experts/Omni-LICENSE](src/model/experts/Omni-LICENSE).

The DSpark Markov correction and confidence head follow DeepSeek's DeepSpec
`deepspec/modeling/dspark/markov_head.py` and `common.py`
(https://github.com/deepseek-ai/DeepSpec). Its MIT notice is retained in
[src/model/dflash/DeepSpec-LICENSE](src/model/dflash/DeepSpec-LICENSE).
