# feat/minicpmv47 合并验证

合并来源：main `3b901e9`、feat/minicpmv47 `2224b40`，native 算子库固定为
`f872a94`。通过验证的 runtime tree 为 `6e42b6dd2d45b5825e59bc8c1d188568c53b833d`。
自动回归入口和测试基础设施单独提交，未改变该 runtime tree 的 Rust/native 源码。

## 执行边界

- 保留 main 的 0.8.0 版本、可选 W8A8 无损压缩、独立执行 lane 和 Qwen3.5 先预分配 KV 的加载顺序。
- Dense 与 Qwen3.5 dense 支持双 ubatch；Qwen MoE、MiniCPM、Ling 在加载前拒绝双 ubatch。
- 流式 MiniCPM/Qwen MoE 保留逻辑 prompt 块，mixer 使用物理 ubatch；checkpoint 边界开启新块。
- `Paging` 由模型持有，每层 `Paged` 只持有自己的热专家 bank、窗口和统计。
  串行层执行显式借用 `&Paging`，共享分页状态不进入双路 worker。
- prefill 工作区按实际物理行数准备，进入 decode 时释放多行 scratch；两条执行 lane
  和新增模型路径均参与相同生命周期管理。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| Rust 格式检查 | 通过 |
| 默认 all-target release Clippy | 通过 |
| 七组单模型/组合 feature Clippy | 通过 |
| 默认 Rust 测试 | 185 通过、0 失败、39 ignored |
| Python 测试基础设施 | 7 通过 |
| 主板模型回归 | 22 个 case 通过 |
| H88K MiniCPM 回归 | 12 个 case 通过 |
| 两板底层算子 | 每板 5 个 case 通过 |
| 完整输出比较 | 17 组逐字节一致 |

主板为 32 GB RK3588，kernel 6.1.118-zt；H88K 为 8 GB RK3588，kernel
6.1.115-vendor-rk35xx。两板均为 RKNPU 0.9.8。板端 case 由共享锁和固定时钟 runner
管理，所有最终成功 case 的时钟恢复无错误。

模型对照使用固定的 600 个输入 token IDs 和 3 个 continuation IDs，比较 4 行完整
FP32 logits，context=640、ubatch=128、no-tune。runtime 对照二进制使用同一个 native pin。

- Qwen3：main/集成版单路、双路与压缩 logits，以及单路/双路和无损压缩内部一致性。
- Qwen3.5：main/集成版单路、双路 logits；双路 recurrent state、features、decode；
  单路/双路 AR、MTP、DFlash 的 prefix restore 状态与 logits。
- Ling3tiny：feature/集成版 resident、stream logits；两种存储路径的 KDA、卷积、MLA
  prefix restore；压缩前后 logits。
- MiniCPM：feature/集成版 streaming logits、无损压缩、逻辑 prompt 与 prefix restore、
  single/batched/append prefetch、AR/DSpark greedy 输出。生成 case 使用 wiki512、tg33、
  context=1024、40 hot slots、temperature=0、seed=42。
- 公共算子：GQA、GDN、MLA、TF32、独立 execution lanes，均在两板执行。

## 自动入口

`tests/board/regression.py` 支持在指定 ARM 板构建，向其他板分发同一组产物，
并行运行不同板、保存独立 case 日志、精确比较输出和断点续跑。主机总报告为
`boards.json`，板端报告为 `summary.json`；失败重试前归档旧结果。
构建、部署和报告可使用共享 NFS 目录。运行方式见 [BOARD_REGRESSION.md](BOARD_REGRESSION.md)。

跨板分发后的续跑验证，两板各 5 个 case 全部显示 `REUSE`，没有重复提交 NPU 测试。
命令超时清理另在 H88K 验证：返回期望的 124，`timed_out=true`，时钟恢复无错误。
默认 CI 同时执行测试基础设施的 Python 单元测试。

原始证据保留在集成 worktree 的
`llm_generated_docs/evidence/minicpmv47_merge_20261028/`：
`validation-summary.json`、`regression-model/`、`regression-remaining/`、`h88-final/`、
`regression-deploy-smoke/`。本机双板配置为该目录下的 `regression-config.json`。

## 已知异常

旧 main 对照程序重复加载时发生过一次 certified W4 权重页映射分配失败。
失败日志已保留，断点续跑只重试该项后通过，对照 logits 逐字节一致。
此前 H88K 的秒级 CMA 重试尚未复现，按要求停止继续调查；没有改动 CMA 配置、
native 分配 flags 或运行时域预算来规避该现象。
