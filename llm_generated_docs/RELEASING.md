# CI and releases

CI 在 GitHub ARM64 runner 的 Debian 11 容器中构建，使用 rust-toolchain.toml 的固定版本和 Cargo.lock。每次 main push、PR 或手动运行执行格式检查、Clippy、自动测试、release 构建、CLI 启动和打包。Cargo 依赖及构建产物按 toolchain 和 lockfile 缓存。

Release CI 由 v 开头的 tag 触发，复用同一套检查和构建。tag 必须与 Cargo.toml 的版本一致；全部检查完成后发布 GitHub Release。

    git tag -s v0.1.0 -m 'Release v0.1.0'
    git push origin v0.1.0

发布包含两个附件：

| 附件 | 内容 |
|---|---|
| open_rknpu2_lm_infer-vVERSION-linux-aarch64.tar.gz | ARM64 可执行程序、使用文档、固定频率脚本、工程和第三方许可证 |
| open_rknpu2_lm_infer-vVERSION-source.tar.gz | Rust 源码、CI 定义及从上游初始化的固定算子库 submodule |

二进制使用 glibc 2.31 构建，适用于 Debian 11 或更新的 ARM64 Linux。系统需要 libstdc++6、libgomp1 和可用的 RK3588 NPU DRM 驱动。解压后通过 bin/open_rknpu2_lm_infer 运行。

源码包保留 submodule 的 Git 元数据和固定提交，可直接执行 cargo build --release --locked。依赖仍由上游仓库维护。源码与文档只收录Git跟踪的文件，本地未跟踪实验不进入附件；二进制附件包含scripts/run_board.py，以便按README显式调优和固定频率测速。

## 板端验证

GitHub runner 没有 RK3588 NPU。3项硬件测试、完整模型数值与性能在 RK3588 上运行；普通 CI 执行自动测试，并保留这3项为 ignored。发布二进制还需在板上检查启动、模型加载和固定 token 的 logits，口径见 [VALIDATION.md](VALIDATION.md)。


## 按模型家族构建交付版本

默认构建启用全部已实现家族以及 DFlash/DSpark。客户版通过 Cargo features 选择家族；features 是叠加开关，因此单家族构建必须使用 `--no-default-features`。

```sh
cargo build --release --locked --no-default-features --features model-ling3tiny --bin open_rknpu2_lm_infer
target/release/open_rknpu2_lm_infer models
python3 scripts/package_release.py --binary target/release/open_rknpu2_lm_infer --models ling3tiny --output dist
```

| 家族 | Cargo feature | config.json 的 model_type |
|---|---|---|
| Qwen3 | `model-qwen3` | `qwen3` |
| Llama | `model-llama` | `llama` |
| Granite | `model-granite` | `granite` |
| Qwen3.5 | `model-qwen35` | `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text` |
| MiniCPM-V 4.7 | `model-minicpmv47` | `minicpmv4_7` |
| Ling3tiny | `model-ling3tiny` | `bailing_hybrid` |

多个家族用逗号组合，例如 `--features model-qwen35,model-ling3tiny`。`all-models` 选择全部家族；`draft-dflash` 额外编入 DFlash/DSpark 草稿实现，需要同时选择 Qwen3.5 或 MiniCPM-V 4.7。单家族开关不会隐式启用草稿模型。

Qwen3.5 与 MiniCPM-V 4.7 共享 Qwen3.5 解码器架构，选择其中一个会编译这套公共实现，但不会注册另一个家族。MiniCPM-V 4.7 仍仅支持已实现的文本路径，不会因架构复用而启用尚未实现的视觉。Qwen3、Llama、Granite 同样共享 dense 解码器。内部 `arch-*` 与 `vision` features 由家族选择自动带入，交付时通常只选择 `model-*`。

未选择的模型模块通过 `#[cfg]` 排除，不参与 Rust 编译；视觉解码依赖和不需要的 CPU recurrence 也按能力裁剪。模型注册表只包含已启用的家族，运行时在加载 tokenizer、权重和设备前拒绝其他家族。共享 NPU 算子库继续复用同一份实现。

`models` 不需要模型目录或 NPU，输出程序的版本、家族及能力清单。包内 `build-info.json` 来自实际可执行文件；`--models` 对这个清单做精确匹配检查，不会代替构建。裁剪包文件名包含家族名称，完整默认包保留原命名。`--binary` 可指定独立输出路径；未指定时遵循 Cargo 的 target directory，包括 `CARGO_TARGET_DIR`。

模型家族由 `model_type` 识别；此选项不绑定某一份权重文件。

### 设置交付名称

在 ARM64 构建环境中，`--name` 同时指定交付可执行文件、CLI、终端聊天界面和默认缓存目录的名称。可与模型裁剪一起使用：

```sh
python3 scripts/build_release.py --name customer_infer --models ling3tiny
dist/customer_infer/customer_infer --version
dist/customer_infer/customer_infer models
python3 scripts/package_release.py --binary dist/customer_infer/customer_infer --name customer_infer --models ling3tiny --output dist/packages
```

构建脚本使用 release 优化并 strip，默认输出到 `dist/NAME/`，可用 `--output` 指定目录。不传 `--models` 时使用默认完整构建；不传 `--name` 时依次使用 `RKNPU_PROJECT_NAME` 或 Cargo 包名。名称以 ASCII 字母或数字开头，可包含字母、数字、`.`、`_` 和 `-`。

交付名称在编译时写入程序；`--help`、`--version`、`models` 的 `project_name` 和 `build-info.json` 保持一致。发布包自动从实际程序读取名称，输出 `NAME-vVERSION-FAMILIES-linux-aarch64.tar.gz` 和对应源码包，包内程序为 `bin/NAME`。打包时的 `--name` 用于核对名称，名称不符会报错。

内部 Cargo package/crate 名称固定，代码导入和 Cargo 原始 target 文件名保持稳定。直接使用 Cargo 时通过 `RKNPU_PROJECT_NAME=customer_infer cargo build ...` 设置显示名称，再由构建/打包脚本输出交付文件名。默认缓存路径为 `$XDG_CACHE_HOME/NAME` 或 `$HOME/.cache/NAME`；复用已有调优缓存时显式传入 `--tune-cache`。

板端已验证默认名称和 `ling3tiny` 自定义名称，各 3 项集成测试通过，覆盖帮助/版本/默认缓存目录、构建清单和排除模型；默认完整构建与自定义 Ling 构建均通过 all-targets release Clippy。示例产物为 `dist/ling3tiny-named/ling3tiny`，实际推理回复“你好。”，`tune_calls=0`，频率设置已恢复。原始记录位于 `results/project-name-20261023/`。

### 2026-10-23 Ling3tiny 专用版验证

产物：`dist/ling3tiny/open_rknpu2_lm_infer`，ARM64 ELF，release（opt-level 3、thin LTO、codegen-units 1），关闭 trace 并 strip。同目录的 `build-info.json` 由实际程序生成，只注册 `ling3tiny`，视觉与 DFlash 能力均关闭。大小为 8,371,000 字节（7.98 MiB），同配置完整版本为 9,653,104 字节（9.21 MiB），缩小 13.3%。

Cargo dep-info 确认不包含 dense、Qwen3.5、DFlash、视觉处理和 GDN 实现，未编译 `image` 依赖；CPU 静态库只有 `kda.cpp.o`，不含 `gdn.cpp.o`。六个单家族、MiniCPM+DFlash、Ling+Qwen3 和默认完整构建均通过 release Clippy（`-D warnings`）。完整版本自动测试 123 项通过、22 项硬件测试忽略；Ling 专用版 88 项通过、11 项忽略。排除家族在加载 tokenizer/权重前被拒绝，打包脚本也能拒绝与实际二进制不符的 `--models`。

在 RK3588 上使用 `Ling-3.0-tiny-QQQ-W4A8-Mixed-CT`、resident 专家、ubatch 128 和已有 `tune-optimized` 缓存验证，未额外调优。固定 CPU 2.4 GHz、NPU 1 GHz、DMC 2.112 GHz。与裁剪前 `adbff99` / native `4e57e12` 的冻结程序比较，137 个输入 token 加 3 个 continuation token 输出的 628,736 个 FP32 logits 逐字节一致。

最终产物的 OpenAI API 短问答、四轮记忆/更名和 SSE 共 7 次请求通过；SSE、非流式及 CLI 的同题输出一致。多轮复用 49/72/97 个 token，23 层 joint MoE 路径保持启用。CLI 统计为 `tune_cache_hits=110`、`tune_cache_misses=54`、`tune_calls=0`、`strategy_fallback_calls=44`、`weight_reload_calls=0`；cache miss 使用默认 NPU 策略。缓存文件保持不变，服务正常退出，频率设置已恢复。此处是功能冒烟验证，不作为新的性能基准。

原始结果保存在 `results/model-features-20261023/`，包括 `build-check.json`、`before/after.json`、对应 logits、`api/report.json` 与频率记录。这里只生成并验证了独立可执行文件，未生成发布归档。
