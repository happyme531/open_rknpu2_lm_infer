# 调优缓存和输出头覆盖修复

2026-11-03，issue #6 的默认缓存路径与 issue #7 的输出头调优覆盖已完成修复，并通过备用板 1 的实际 CLI 回归。PR #5 的采样实现保持未合入，已加入三个对应的行为回归测试；运行时开放全模型调优的环境变量入口也保持未合入。

## 默认缓存路径

`--tune-cache PATH` 优先于环境变量。未显式指定时，依次采用：

1. systemd 的 `CACHE_DIRECTORY`，多个目录时使用第一个非空目录。
2. `$XDG_CACHE_HOME/open_rknpu2_lm_infer`。
3. `$HOME/.cache/open_rknpu2_lm_infer`。
4. `MODEL/.cache/open_rknpu2_lm_infer`。

空环境变量按未设置处理。systemd 提供的路径直接使用；其余路径末尾使用编译时项目名称。服务可配置 `CacheDirectory=open_rknpu2_lm_infer`，调优和服务应使用相同的目录、用户与 CPU/NPU 设置。

`run`、`chat` 和 `serve` 的引擎启动后会输出实际缓存路径、初始命中与缺失数量，以及 fallback shape 数量。普通推理继续只读取缓存，缺失配置由显式 `tune` 生成。

## 输出头调优覆盖

显式调优将实际 prefill batch 加入输出头准备集合。Dense、Qwen3.5/MiniCPM-V 和 Ling3tiny 共用同一个行数规划函数；普通推理只准备 decode 和 verification 使用的输出头 workspace。

`tune --batches 1,2,4,8,128` 同时覆盖 decoder 和量化输出头。Qwen3.5 的 MTP target 验证因此可以读取 M=2/4/8 的 head profile，无需逐次修改 `--ubatch-size`。此改动没有开放 MTP 辅助浮点投影的通用搜索。

## 备用板验证

板端为 `备用板 1`，RKNPU 驱动0.9.6。使用 Rust1.95.0、GCC12.2；CPU4–7 固定2.256GHz、NPU1GHz、mask7，板端无 DMC devfreq，DDR 未控制。构建工具、libclang 和依赖 vendor 均置于隔离目录。

验证基线为父仓库 `0f29dfe`，native 使用工作区已存在的 `c1f4bc5`。本轮没有修改该 native 提交或父仓库的 submodule 指针。

四个小型单层检查点分别覆盖 Llama/Qwen3.5 与 W4/W8 head，hidden64、vocab320。它们用于检验真实 loader、tuner、缓存文件和推理进程，数值与性能结果不代表真实大模型。

| 检查点 | 基线 head M=2/4/8 | 修复后 head M=1/2/4/8/128 | 新进程重复调优 | 普通生成 head workspace |
| --- | --- | --- | --- | --- |
| Llama W4 | 全部缺失 | 全部生成 | 0搜索、0缺失 | 1 |
| Llama W8 | 全部缺失 | 全部生成 | 0搜索、0缺失 | 1 |
| Qwen3.5 W4 | 全部缺失 | 全部生成 | 0搜索、0缺失 | 1 |
| Qwen3.5 W8 | 全部缺失 | 全部生成 | 0搜索、0缺失 | 1 |

各检查点调优后的普通生成均为零 cache miss、零 tune call、零权重重载。另验证了三个缓存部署场景：缺少 HOME/XDG/CACHE_DIRECTORY 时在模型目录生成缓存并由新进程命中；`CACHE_DIRECTORY` 供调优与推理共同使用；显式 `--tune-cache` 覆盖服务缓存目录。

四个检查点使用固定输入 token IDs `[1,2]`，分别运行原版默认策略、修复版默认策略和修复版已有调优缓存。每种运行方式合计12,672个 trace 元素，全部 FP32 中间结果与 logits 均逐位一致。

`cargo fmt --all --check`、all-target/all-feature release Clippy 和 release tests 通过。自动测试共192项通过、39项硬件或模型依赖测试忽略。基线与修复版硬件回归均经共享板锁执行，`exit_code=0`、`timed_out=false`、`restore_errors=[]`。

## PR 5 采样和运行时调优

当前 sampler 的三个回归测试固定以下行为：

- 无截断采样保留已有 seed42 的 token 序列。
- `logits=[0,1e-8]`、temperature1、top_p0.5 时，按原始 logit 顺序保留 token1，即使两个 FP32 指数舍入相同。
- 单候选步骤不推进 RNG，后续均匀抽样序列保持一致。

这三个测试在当前实现上通过；PR #5 提交 `9b4a567` 的对应路径与这些行为不一致。采样分布等价与相同 seed 的输出序列一致是不同要求。

MTP 浮点调优需要在权重上传前完成受约束的离线搜索，并确认新进程实际采用 profile。PR 的环境变量会开启整个模型的通用搜索，decoder 的宽 N 检查发生在搜索之后，且新 K 分块可能不在已上传的 resident weights 中。当前工作没有执行已知会超时的 MTP 宽投影搜索。

## 修改与证据

- 缓存解析：`src/cli/cache.rs`、`src/cli.rs`。
- 启动缓存统计：`src/engine.rs`。
- head 行数规划：`src/model/workspace.rs` 及 Dense、Qwen3.5、Ling3tiny 的 prepare 路径。
- 采样回归：`src/sampling.rs`。
- 中英文 README 与交付文档已同步。
- [验证汇总](evidence/issues67_20261103/summary.json)、[基线结果](evidence/issues67_20261103/baseline/summary.json)、[修复版结果](evidence/issues67_20261103/modified/summary.json)。
- [检查点生成工具](evidence/issues67_20261103/create_fixtures.py)、[板端回归入口](evidence/issues67_20261103/board_regression.py)。完整构建和缓存位于 `<HOST_HOME>/mnt/3090x4/tmp/issues67-pr5-20261103/`，其 `baseline` 子目录为实际应用修复后的验证源码，`original-files` 保留修改前文件，`modified` 保留格式化后的修复快照。

缓存和 head 调优修复已签名提交为 `9ad7a6d`，尚未 push。PR #5 的采样测试与后续优化保持独立，PR 尚未合并。工作区中原有的 AGENTS、Ling 文档、native 指针和其他实验文件保持独立。
