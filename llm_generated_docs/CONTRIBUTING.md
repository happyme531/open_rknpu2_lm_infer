# Contributing

模型、通用 CPU 运算、调度和用户界面使用 Rust；NPU 运算通过 `rk_npu_matmul` 公共 C ABI 接入。复用 llama.cpp 算法的 GDN CPU kernel 独立放在 `native/cpu_kernels/`，由 Rust 校验输入后经窄 C ABI 调用，并保留上游许可。新增模型实现 `LanguageModel` 并在 `model::load` 注册。数值算子放 `src/ops/`，无 checkpoint 依赖的可复用层放 `src/layers/`，配置、权重映射与模型执行放 `src/model/`；详见 [RUNTIME_MODULE_LAYOUT.md](RUNTIME_MODULE_LAYOUT.md)。

算子库通过 `native/rk_npu_matmul` Git submodule 固定 [上游提交](https://github.com/happyme531/rknpu2_matmul_open)。克隆后运行 `git submodule update --init --depth 1`。算子实现和硬件布局调度维护在上游库中；本地联调可通过 `RK_NPU_MATMUL_ROOT` 指定 checkout。

独立测试、benchmark、调优和诊断工具放在 `tests/tools/`，不作为使用例程。Cargo.toml 显式注册这些独立可执行目标；仍使用 `cargo run --release --example NAME -- ARGS` 或 `cargo build --release --example NAME`，产物仍在 `target/release/examples/`。这些工具不会在默认 `cargo test` 中执行；新工具也应显式注册并设置 `test = false`。

测试必需的小型图片和数值参考放在 `tests/fixtures/`，不要仅放进忽略的实验报告目录。视觉测试的六份 fixture 来自 2026-10-18 的 `qwen35_vision_20261018` 验证记录，原字节与固定参考哈希保持一致；参考生成入口为 `scripts/qwen35/vision_reference.py`。

在 RK3588 Linux 板上验证代码修改：

```sh
cargo fmt --check
cargo clippy --all-targets --release --locked -- -D warnings
cargo test --release --locked
```

涉及 NPU、KV 或模型前向的修改还需运行相关板端算子测试，并使用固定 token IDs 比较模型 logits。板端测试与测速共享 `/tmp/rknpu_lowlevel_submit.lock`；固定频率和恢复由 `scripts/run_board.py` 管理。

`tests/board/regression.py` 提供配置驱动的单板/多板自动回归，覆盖算子、基线 logits、prefix、专家 prefetch、压缩和推测解码，并支持超时与断点续跑。配置和运行方式见 [BOARD_REGRESSION.md](BOARD_REGRESSION.md)。

全局板锁只用于测试、评估和测速，工具类型为 `validation::BoardLock`；正式 `run`、`chat`、`serve` 和 `tune` 不获取它。真实图片的服务/TUI 回归入口为 `tests/board/vision_api.py`、`vision_tui.py`；由外层 runner 持锁，子应用不继承锁描述符和 bypass 环境变量，以同时验证正式路径不依赖全局锁。TUI 测试可按 `tests/board/requirements.txt` 在隔离目录安装测试依赖。

性能报告包含模型格式、上下文、chunk、CPU/NPU mask、固定频率、调优缓存状态及原始样本。分别报告 prefill、decode 和历史 KV 深度；prefill 测完整主干调用，不执行最终 norm/输出头，decode 包含输出头。连续 `-pg PP,TG` 在同一 KV 上下文先 prefill 再 decode。结果与复现方式见 [VALIDATION.md](VALIDATION.md)。

PR 说明具体行为变化和验证结果。提交仅包含相关源码、文档与小型证据；模型文件、构建产物和实验目录由 `.gitignore` 排除。贡献按工程的 AGPL-3.0-only 许可证提供，第三方来源保留原许可和通知。

GitHub ARM64 CI 执行上述自动检查并生成可下载的包；版本 tag 触发 Release CI。构建环境、附件内容和发布步骤见 [RELEASING.md](RELEASING.md)。
