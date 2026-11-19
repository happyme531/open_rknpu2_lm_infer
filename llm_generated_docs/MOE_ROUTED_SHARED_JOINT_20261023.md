# Routed W4 / shared W8 共同调度，2026-10-23

算子库签名提交：`7a996a2858025d5f545da71d3fcb8625a2776b5d`，分支 `perf/moe-ffn`。提交只含 5 个源码/文档/回归测试文件，实验工具与结果未提交。

## 结果与范围

在 `c96930f` 的 CPU 融合基础上，扩展算子库已有 private mixed MoE 路径：routed gate/up 保持 K≤512 分片，shared W8 按 N 分到三核，同阶段任务进入同一次 mixed 提交。真实 Ling layer1 权重与输入的完整 routed+shared FFN 对照，单 token 形状时延下降约 19–21%，M9 均值下降 7.48%，M25 的配对收益中位数为 5.55%。M1/M9 为 40/40 配对周期更快，M25 为 39/40。

本轮仅实现并验证算子库路径。没有改动 Ling runtime 工作树，没有提前执行 shared，没有 CPU/NPU 跨分支流水。固定路由由调用者提供，计时不包含 TF32 router 或 CPU top-k；结果不是整模型吞吐提升。

## 实现

- 原 `RoutedW4SharedW8` 的 routed gate/up 要求完整 K 通过 INT16 bounds；现与纯 W4 共用 K≤512 的权重检查、打包、partial 存储和重建。共享 W8 仍保留原 codes/scales、完整 K。
- Routed 权重沿用 ≤16 MiB 的 bank 划分，shared W8 独立分配；不动态转换权重。
- 联合执行只进行一次输入 A8 量化。所有专家沿用 MoE nearest-even 规则，CPU 中间处理复用本轮之前已合入的 FP32 SwiGLU/maxabs/Down packing。
- Shared 的 gate/up 与 Down 分别按 N 切片，B 的 N32 native 子区间、C 的 N4 native 子区间直接引用原缓冲，无输出拼接搬运。任务追加到各核的 mixed INT4/INT8 chain。
- 分核成本估计不再将 shared 的整块行工作量记到一个核，避免给已分布到各核的 shared 重复计费、扭曲后续 routed 分配。
- N slice 至少 64 列，满足 M1 时原生 INT8 输出 view 至少 256 bytes 的要求。初版切到 N32 被公共 API 以 `buffer too small` 拒绝；修正后小形状和完整真实形状回归通过。这不是 submit 超时或设备失效。
- 最终 CPU combine 在 routed 顺序之后加 shared，移除独立 shared 输出数组与后续 add 的生产路径需求。

修改集中在 `src/rk_npu_moe_w8.cpp`、private mixed 头文件、MoE 文档与测试。没有新增运行时 debug/NaN 扫描或自动 fallback。

提交前清理了旧的 `RK_NPU_MOE_W4_MIXED` 环境变量、分开提交分支和单核 shared 调度分支，mixed 入口固定采用联合调度。Shared 任务描述符仅按 shared M 分块数分配，不再按全部 routed job 数分配。公共 C 头文件、ABI 和 private mixed 函数签名均不变；公共输入及资源校验保留。

## 对照设计

板卡 `备用板 1`，CPU4–7/4 threads @2.256 GHz，NPU mask7 @1 GHz，DDR 未固定。共享板端锁，所有成功运行的时钟恢复记录均为 `restore_errors=[]`。

原始 checkpoint：`<HOST_HOME>/mnt/spark-test/Ling-3.0-tiny-QQQ-W4A8-Mixed-CT`。Layer1，H1536/I512，128 routed W4 experts、top8、1 shared W8 expert。保持原始量化 codes/scales。

输入复用 `open_rknpu2_lm_infer-ling3tiny/results/ling-real-stream-trace/p*-l1-post.f32` 的 25 行真实 post-norm 捕获。M9 是首个 prefill 捕获；M25 拼接全部捕获行用于批处理测试，不是一段捕获的 25-token prefill。M1 取 offset0/8/24 三个切片。路由从 TF32 截断后的 router 操作数在 CPU 重建并固定，不是 runtime 路由 dump。

| 版本 | 执行方式 |
|---|---|
| serial_dense | 纯 W4 routed FFN → 独立 W8 gate/up → FP32 SIMD SwiGLU → W8 Down → add |
| serial_fused | 纯 W4 routed FFN → 单专家 W8 grouped FFN → add；CPU 中间核与联合路径一致 |
| joint | K-sliced routed W4 + N-sliced shared W8 联合调度 |

Dense W8 在每个进程内先用库 autotune 选择策略，权重加载、调优和正确性检查均在计时外。Dense 对照也使用新的 FP32 SIMD 激活；按 runtime 阈值，activation≤1024 元素、add≤16384 元素直接串行。它是算子级对照，不是 Rust runtime 的逐指令重放：未模拟 runtime 的 M bucket、Rayon 与分配开销，也未使用旧 FP64 activation。

两个独立进程，每进程每个用例 20 个 `A B C C B A` 周期；每版本每用例 80 次调用、40 个配对周期。每次计时后检查输出与该版本首次输出逐位一致。

下述完整性能采样在提交前清理之前完成。清理后重跑主机、板端所有相关路径及真实权重对照；没有重复完整性能采样。

## 正式完整调用结果

单位 ms。收益均以 `serial_dense` 为分母。

| M | offset | serial_dense | serial_fused | joint | 时延下降 | 更快周期 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.797178 | 0.860011 | 0.644267 | 19.18% | 40/40 |
| 1 | 8 | 0.807810 | 0.855359 | 0.641431 | 20.60% | 40/40 |
| 1 | 24 | 0.794816 | 0.854130 | 0.637519 | 19.79% | 40/40 |
| 9 | 0 | 2.836676 | 2.861680 | 2.624415 | 7.48% | 40/40 |
| 25 | 0 | 4.501438 | 4.408688 | 4.117347 | 8.53% | 39/40 |

M25 第二个进程的 dense 对照出现一次 16.581 ms 长尾，所有样本均保留，均值收益受到它影响。以每个 ABC/CBA 周期内的均值计算相对收益，再取中位数，M25 为 **5.55%**，更适合表述典型收益；M1 三个切片为 19.68%/20.37%/20.68%，M9 为 8.01%。

相对 CPU 核相同的 `serial_fused`，joint 的 M1 均值快 25.0–25.4%，M9 快 8.29%，M25 快 6.61%，也都是 40/40 周期更快。M1/M9 单独把 shared 改成单专家 grouped FFN 并没有胜过 tuned dense；联合收益不能归因于 shared 更换 CPU 激活核。

## 收益来自哪里

M1/offset0，`serial_fused → joint` 的内部计时均值：

| 阶段 | 原 us | 新 us |
|---|---:|---:|
| 输入量化/打包 | 60.109 | 36.892 |
| 命令构建 | 25.317 | 31.384 |
| gather 完整调用 | 11.926 | 0.036 |
| gate/up 提交等待 | 398.478 | 298.538 |
| CPU middle | 50.028 | 37.614 |
| Down 提交等待 | 223.736 | 169.106 |
| 输出 combine | 64.532 | 52.683 |
| 内存同步 | 17.598 | 13.428 |

NPU GEMM task 数从 34 增至 38，因为 shared 按 N 分核。收益来自共同提交、三核分担 shared、共享 CPU 前后处理及 M1 的直接输入，不是减少逻辑 MAC 数。提交等待包含驱动/调度/硬件，不能解释成纯 MAC 耗时。完整调用是最终裁决。

首版 shared 未按 N 分核时，M1 初测仅约 9–10% 收益；该记录是早期探针，且 dense 小张量线程阈值尚未对齐，不用于正式收益或 N 分块的独立因果估计。最终以 `paired-balanced/` 为准；`paired-final/` 是修正 shared 调度成本估计之前的中间记录。

## 正确性与测试

- 主机 `moe_cpu/i4_cpu/w4a8_swiglu_cpu/w4a8_ffn_cpu` 全部通过，公共 C header 编译通过。
- 板端 W8 normal/native、原 mixed、K512+64 mixed、纯 W4/slot、dense W4A8 FFN 均通过；mixed K 分片同时覆盖合并提交和 `RK_NPU_MOE_W4_MIXED=0` 分开提交。
- 真实权重的原输入、全零、4 倍、取负输入，三个版本两两 relative L2 均为 0；各版本重复输出逐位稳定。这不等于认证任意输入的 FP32/量化舍入完全相同。
- 最终版单/双/三核的 K 分片 mixed 用例全部通过，另存于 `cores/`。
- 提交前清理后，W8 normal/native、mixed、K 分片 mixed 单/双/三核、纯 W4/slot、NPU middle 和 dense W4A8 再次通过；真实权重 20 组输入比较仍为 relative L2=0。记录位于 `commit-cleanup/`。之前的分开提交模式已经移除。
- 未做 PPL 或整模型性能测试，也未覆盖不同 shared/routed 中间宽度。

## 证据与下一步

- 正式数据：[`paired-balanced/summary.json`](evidence/moe_joint_20261023/paired-balanced/summary.json)、[`protocol.json`](evidence/moe_joint_20261023/paired-balanced/protocol.json)、[`clocks.json`](evidence/moe_joint_20261023/paired-balanced/clocks.json)，同目录 `joint-{0,1}.csv/log`。
- 最终库回归：`evidence/moe_joint_20261023/paired-balanced/`；核数验证：`evidence/moe_joint_20261023/cores/`。
- 提交前清理回归与真实权重复核：`evidence/moe_joint_20261023/commit-cleanup/`。
- 工具：算子库 `tests/tools/moe_ffn/{joint.cpp,run_joint.py,export_shared.py,summarize_joint.py,validate_cores.py}`。实验代码、数据、二进制不进入正式提交。
- 下一步可将 Ling 的 resident routed bank 与 shared 权重接入联合入口；slot/paged 模式还需要给共享专家提供独立的驻留生命周期，不能把 shared 当成可驱逐的 routed slot。再以完整调用验证集成收益，之后才考虑 shared 提前执行和 CPU/NPU 交错。
