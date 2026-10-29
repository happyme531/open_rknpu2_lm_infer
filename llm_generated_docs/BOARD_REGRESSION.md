# 板端自动回归

`tests/board/regression.py` 从配置完成板端回归，按 case 保存 stdout、日志、时钟和
`summary.json`，失败时返回非零。所有板端调用复用 `scripts/run_board.py` 的共享锁和
固定频率，默认单 case 超时 600 秒；超时会终止命令进程组并恢复时钟。

## 单板

复制 `tests/board/regression.example.json`，填写本板模型、固定 token IDs、continuation
和对照源码路径，删除本板不测的模型，然后运行：

```sh
python3 tests/board/regression.py --config /path/to/config.json --output llm_generated_docs/evidence/regression
```

默认编译当前 checkout 的 release CLI、libtest，以及 `sources` 中指定的对照 checkout，
依次保存独立二进制。对照和当前版本均使用当前 checkout 的 native 算子库，以比较
runtime 改动。源码 checkout 需提前具备 Cargo 依赖和 native submodule。
可用 `target_dir` 指定已有 Cargo 构建缓存。普通 fmt、Clippy、单元测试和单模型 feature
检查仍由 CI 自动执行。

已有构建产物时，用 `binaries` 代替相应源码构建，适合没有编译环境的板：

```json
{
  "binaries": {"current": "/path/to/current-infer", "tests": "/path/to/current-libtest", "feature": "/path/to/feature-infer"},
  "tokens": "/path/to/tokens.json",
  "continuation": "/path/to/continuation.json",
  "operators": false,
  "models": {
    "minicpmv47": {
      "path": "/path/to/minicpm-model",
      "store": "/path/to/expert-store",
      "baseline": "feature",
      "draft": "/path/to/dspark",
      "prompt": "/path/to/wiki512.txt",
      "windows": "/path/to/router_ppl_windows.json"
    }
  }
}
```

`tokens` 是 JSON token ID 数组或现有 `eval --tokens-file` 支持的输入；当前标准配置使用
600 个 prompt tokens，续写文件含至少一个 token，context=640。MiniCPM 生成 case 使用
1024 context、128 ubatch、40 hot slots、greedy tg33。每次更换模型或 expert store 内容时
使用新输出目录。`--resume` 复用同一配置、同一可执行文件及 token 输入的成功 case；
所有比较仍重新执行。失败 case 的旧日志和结果移到 `attempts/` 后再重试。
两个板可以分别设置 `cpu_khz` 和 `skip_dmc`。

只运行底层算子时可省略 `models` 和 token 文件，保留 `operators: true`。

## 多板一次运行

主机配置顶层 `boards`，每项包含 SSH 地址、本板结果目录和上述单板 `config`：

```json
{
  "build_on": "primary",
  "boards": {
    "primary": {
      "ssh": "firefly@192.168.10.183",
      "source": "/path/on/board/to/current-checkout",
      "output": "/path/on/board/to/regression-results",
      "config": {"models": {}, "operators": true}
    },
    "h88k": {
      "ssh": "user@fe80::1828%enp3s0",
      "sudo": true,
      "output": "/mnt/nvme/regression-results",
      "config": {"binaries": {"current": "/path/to/infer", "tests": "/path/to/libtest"}, "operators": true}
    }
  }
}
```

`sudo: true` 用于板上共享锁或模型文件由 root 持有的情况，需要无密码 sudo。
设置 `build_on` 后，入口先在指定板构建并收集 CLI、libtest 和对照二进制，再把同一组
产物部署到所有板，随后并行回归。该板的配置应包含所有需要的 baseline 标签。
省略 `build_on` 时各板自行使用配置的构建方式或已有二进制。
模型使用各板已有路径，原始日志和 `summary.json` 保存在各板的 `output`。
主机保存 `artifacts/`、各板日志和总汇总 `boards.json`，可使用与主板共享的 NFS 目录：

```sh
python3 tests/board/regression.py --config /path/to/boards.json --output llm_generated_docs/evidence/boards --resume
```

## 覆盖范围

- 公共算子：GQA、GDN、MLA、TF32、独立 execution lanes。
- Qwen3：单/双路 prefill、基线完整 logits、W8A8 无损压缩。
- Qwen3.5：单/双路完整 logits、recurrent state 与 features、AR/MTP/DFlash prefix restore。
- Ling3tiny：resident/stream 对照 logits、两条路径的 KDA/MLA prefix restore、无损压缩。
- MiniCPM-V 4.7：streaming 对照 logits、逻辑 prompt 块与 prefix restore、single/batched/append
  prefetch、无损压缩、AR/DSpark greedy 输出及基线比较。

模型通过 `models` 显式选择。`baseline` 选择对照二进制名；省略时只做当前版本内部一致性
与状态测试。省略 `draft` 时不运行需要该 draft 模型的 DFlash/DSpark case。
过滤器匹配零个测试、空输出、非有限 logits、任何字节差异、命令超时或时钟恢复失败都会
使该次回归失败，并在汇总中保留失败位置。
