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

## GHCR 镜像

Release 附件发布后，容器工作流从该附件构建 ARM64 scratch 镜像，验证 CLI 启动后推送至
`ghcr.io/happyme531/open_rknpu2_lm_infer:VERSION`。当前最新稳定 Release 同时发布 `latest`。
使用 Actions 的 `GITHUB_TOKEN` 和 job 级 `packages: write` 权限，不需要另配 PAT。

补发已有 Release 的镜像时，手动运行 CI，并填写 `container_version`；该模式只执行镜像发布：

```sh
gh workflow run ci.yml -f container_version=0.11.0
```

首次发布后确认 GHCR package 的 visibility 为 public，用户即可免登录拉取。
部署及本地构建方式见 [DOCKER.md](DOCKER.md)。

源码包保留 submodule 的 Git 元数据和固定提交，可直接执行 cargo build --release --locked。依赖仍由上游仓库维护。源码与文档只收录Git跟踪的文件，本地未跟踪实验不进入附件；二进制附件包含scripts/run_board.py，以便按README显式调优和固定频率测速。

## 板端验证

GitHub runner 没有 RK3588 NPU。3项硬件测试、完整模型数值与性能在 RK3588 上运行；普通 CI 执行自动测试，并保留这3项为 ignored。发布二进制还需在板上检查启动、模型加载和固定 token 的 logits，口径见 [VALIDATION.md](VALIDATION.md)。


## 按模型家族构建交付版本

默认构建启用 Qwen3、Llama、Granite、Qwen3.5 和 MiniCPM-V 4.7 五个家族，GitHub Release 使用同一默认配置。Cargo features 只选择目标模型家族；共享算子、视觉处理和推测解码公共模块统一编译。features 是叠加开关，因此单家族构建必须使用 `--no-default-features`。

```sh
cargo build --release --locked --no-default-features --features model-qwen3 --bin open_rknpu2_lm_infer
target/release/open_rknpu2_lm_infer models
python3 scripts/package_release.py --binary target/release/open_rknpu2_lm_infer --models qwen3 --output dist
```

| 家族 | Cargo feature | config.json 的 model_type |
|---|---|---|
| Qwen3 | `model-qwen3` | `qwen3` |
| Llama | `model-llama` | `llama` |
| Granite | `model-granite` | `granite` |
| Qwen3.5 | `model-qwen35` | `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text` |
| MiniCPM-V 4.7 | `model-minicpmv47` | `minicpmv4_7` |

多个家族用逗号组合，例如 `--features model-qwen35,model-qwen3`。没有独立的架构、视觉或草稿功能开关；MTP、DFlash/DSpark 是否适用于当前模型，由模型适配、checkpoint 和运行参数决定。Qwen3.5/MiniCPM 单家族版本同样可使用其已实现的推测解码路径。

Qwen3.5 与 MiniCPM-V 4.7 共享 Qwen3.5 解码器架构，选择其中一个会编译这套实现，但不会注册另一个家族。MiniCPM-V 4.7 仍仅支持已实现的文本路径，不会因架构复用而启用尚未实现的视觉。Qwen3、Llama、Granite 同样共享 dense 解码器。

模型条件编译集中在模块入口、注册表和模型专属测试工具。共享的 NPU/CPU 算子、投影、专家调度和预填充接口不枚举模型家族；`image` 依赖及 GDN/KDA CPU 内核统一编译。没有消费者的实现由链接器按可达性裁剪，最终体积需以实际产物为准。模型注册表只包含已启用的家族，运行时在加载 tokenizer、权重和设备前拒绝其他家族。

`models` 不需要模型目录或 NPU，输出程序的版本、家族及能力清单。`capabilities` 表示公共实现已编入，不代表每个目标模型均支持这些功能。包内 `build-info.json` 来自实际可执行文件；`--models` 对家族清单做精确匹配检查，不会代替构建。与默认家族集合不同的包文件名包含家族名称，默认包保留原命名。`--binary` 可指定独立输出路径；未指定时遵循 Cargo 的 target directory，包括 `CARGO_TARGET_DIR`。

模型家族由 `model_type` 识别；此选项不绑定某一份权重文件。

### 设置交付名称

在 ARM64 构建环境中，`--name` 同时指定交付可执行文件、CLI、终端聊天界面和默认缓存目录的名称。可与模型裁剪一起使用：

```sh
python3 scripts/build_release.py --name customer_infer --models qwen3
dist/customer_infer/customer_infer --version
dist/customer_infer/customer_infer models
python3 scripts/package_release.py --binary dist/customer_infer/customer_infer --name customer_infer --models qwen3 --output dist/packages
```

构建脚本使用 release 优化并 strip，默认输出到 `dist/NAME/`，可用 `--output` 指定目录。不传 `--models` 时使用默认家族集合；不传 `--name` 时依次使用 `RKNPU_PROJECT_NAME` 或 Cargo 包名。名称以 ASCII 字母或数字开头，可包含字母、数字、`.`、`_` 和 `-`。

交付名称在编译时写入程序；`--help`、`--version`、`models` 的 `project_name` 和 `build-info.json` 保持一致。发布包自动从实际程序读取名称，输出 `NAME-vVERSION-FAMILIES-linux-aarch64.tar.gz` 和对应源码包，包内程序为 `bin/NAME`。打包时的 `--name` 用于核对名称，名称不符会报错。

内部 Cargo package/crate 名称固定，代码导入和 Cargo 原始 target 文件名保持稳定。直接使用 Cargo 时通过 `RKNPU_PROJECT_NAME=customer_infer cargo build ...` 设置显示名称，再由构建/打包脚本输出交付文件名。默认缓存依次使用 systemd 的 `CACHE_DIRECTORY`、`$XDG_CACHE_HOME/NAME`、`$HOME/.cache/NAME`，这些变量均未设置时使用 `MODEL/.cache/NAME`。systemd 配置的目录直接使用；其他位置按交付名称区分。复用已有调优缓存时显式传入 `--tune-cache`。
