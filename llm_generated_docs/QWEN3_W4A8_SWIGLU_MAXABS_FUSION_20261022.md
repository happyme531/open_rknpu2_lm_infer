# Qwen3 W4A8：SwiGLU 累计 maxabs，Down 复用 scale

2026-10-22。真实 layer2 输入、原始 codes/scales，RK3588 单个完整 FFN。

**在 FP32 两波融合基线上，完整 FFN 从 8.410 降至 8.200 ms，时延进一步降低约 2.50%。80 个进程内配对周期中，79 个更快。scale、inverse、middle 和最终输出保持逐字节一致。**

SwiGLU 在结果仍位于 SIMD 寄存器时累计每行 maxabs。Down 直接复用生成的 scale/inverse，从而省掉一次完整 middle 扫描；量化、打包、GEMM 与输出归约继续使用原来的实现。

## 完整调用结果

三种模式共用权重、Gate/up executor、输入/partial 缓冲、middle/output 和 Down workspace：

| 模式 | Gate/up ms | Down ms | 完整 FFN ms |
|---|---:|---:|---:|
| 两波融合基线，Down 自己扫描 | 5.599 | 2.811 | **8.410** |
| SwiGLU 计算 maxabs，Down 仍扫描 | 5.598 | 2.785 | 8.384 |
| SwiGLU 计算 maxabs，Down 复用 scale | 5.597 | 2.602 | **8.200** |

主要结果关闭事件记录；两个进程，每进程 40 个 ABCCBA 周期，每种模式合计 160 次完整调用。两个进程分别为：

| 进程 | 基线 ms | 复用 scale ms | 更快的配对周期 |
|---|---:|---:|---:|
| 0 | 8.448 | 8.215 | 40/40 |
| 1 | 8.373 | 8.184 | 39/40 |

Gate/up 为 M128/K2560/N19456，Mtile128、Ntile4096、Ktile1472，两 K 波；Down 为 M128/K9728/N2560，Ktile1888、六波。CPU4–7、四线程、2.256 GHz，三核 NPU 1 GHz，DDR 未控制。测试使用共享板锁与频率恢复工具，周期不是独立板卡实验。

这批基线已经包含[两波 CPU 融合](QWEN3_W4A8_CPU_TWO_WAVE_FUSION_20261022.md)。这里的 2.50% 是本批增量，不把不同批次的百分比直接相加。

## 消除的阶段和新增的工作

原路径：

```text
两波归约 + SwiGLU → 写 middle
Down 扫描完整 middle → 每行 scale/inverse
Down 量化打包 → GEMM/归约
```

当前路径：

```text
两波归约 + SwiGLU → 写 middle，同时从输出寄存器累计 maxabs
归并每行 maxabs → scale/inverse
Down 复用 scale/inverse → 原量化打包/GEMM/归约
```

M128/F9728 的 middle 为 4.75 MiB，复用 scale 删除了对它的一次独立扫描。CPU 按 N 块分工，各线程持有自己的行最大值；结束后归并四份最大值，再执行与原扫描相同的 `maxabs/127`、`1/scale`、全零行处理和 `FLT_MIN` 下限。每行全部 9728 列仍共用一个 scale。

最大值使用清除符号位后的 FP32 位模式进行无符号 max。对于有限正值，这与浮点绝对值的大小顺序一致；NaN/Inf 的位模式也会保留到最终合法性检查。SwiGLU 的算术、输出精度和回退条件保持原样。

每线程 128 行最大值之间保留一条缓存行间隔。额外状态为 2.25 KiB 的线程局部最大值，以及 1 KiB 的 scale/inverse，均预先分配。Down 的诊断入口将两组元数据复制到原 workspace，随后使用原 producer 和执行器。

## 阶段成本

| 阶段，累计墙钟跨度 | 基线 ms | 复用 scale ms |
|---|---:|---:|
| Gate/up CPU reduce/finish | 2.417 | 2.437 |
| Down scale 准备 | 0.224 | 0.0008 |
| Down quant/pack | 0.807 | 0.912 |
| 完整 FFN | 8.410 | 8.200 |

寄存器内收集最大值只增加约 0.020 ms 的 CPU 收尾跨度，独立扫描变成不足 1 μs 的元数据复制。CSV 的 `down_scan_us` 字段在复用模式下记录的是这段复制成本。

Down 的 quant/pack 跨度增加约 0.105 ms，与省略预扫描后缓存状态发生变化相符，但尚未用硬件计数器细分原因。pack、submit、reduce 存在重叠，不能将各段差值直接相加；收益以上面的完整 FFN 交错测试为准。

maxabs 从激活结果寄存器直接形成，不回读刚写出的 middle。源码中的这种衔接对性能很重要：收集逻辑需要与激活写回共同生成，而不是在每个小输出块后另加一次读取循环。

## 正确性和边界

- 计时前覆盖真实输入、全零、四倍真实输入、取负真实输入。
- 每种输入上，融合得到的 scale 和 inverse 与原 `w4a8_scales` 扫描结果逐字节一致；middle 和最终 Down 输出也与两波融合基线逐字节一致。
- 每个计时调用返回后，在计时区间外核对完整 middle 和最终输出；所有调用通过。
- 两个主要测试进程均正常退出，频率恢复检查通过。

本次没有改变已有 FP32 激活算法的误差，因此结果继承 FP32 两波融合基线的数值边界。验证范围是这一组真实 layer2/M128 单个 FFN，不是全模型速度或质量验收。

Down 阶段的改善依赖于上游提供 scale；生成它的成本已包含在完整 FFN 中，不能把这个 Down 时间解释为任意独立输入都无需扫描。本报告前半部分记录诊断实现，位于算子库工具及生成的源码中；后续正式实现与回归见下一节。

## 正式实现与回归

算子库签名提交：`905ee2bf55bd5111e49d0eac9c42fc39595f46f2`，分支 `perf/w4a8-fused-ffn`。提交只含正式源码、接口文档和回归测试；实验工具、数据与构建产物均未提交。

后续已将两波消费、默认 FP32 SwiGLU、寄存器 maxabs 收集接入算子库正式源码，并增加 `rk_npu_w4a8_run_ffn_f32` 公共 C ABI。按用户明确要求，W4A8 直接使用新 FP32 激活，不保留旧 FP64 策略开关；宽范围数值回退继续复用共享实现。新 FFN 入口持有两个 workspace 的 BUSY 状态，在库内传递行 scale/inverse。原 SwiGLU 入口也自动使用两波融合；单独调用普通 Down 时仍自行扫描。

正式库和旧静态库的真实 layer2/M128 配对结果如下。两进程各 40 个 ABCCBA 周期，每模式 160 次调用，CPU4–7/2.256 GHz、NPU mask7/1 GHz、DDR 未控制，与前述原型采用相同原始权重和捕获输入。

| 正式入口 | 完整 FFN ms | Gate/up ms | Down ms |
|---|---:|---:|---:|
| 旧库 SwiGLU + Down | 9.328 | 6.546 | 2.781 |
| 新库 SwiGLU + 普通 Down | 8.252 | 5.497 | 2.753 |
| 新库完整 FFN，复用 scale | **8.135** | 5.565 | 2.568 |

新 FFN 相对旧库降低 12.78%，80/80 个配对周期更快；相对新库分开调用降低 1.42%，79/80 个周期更快。真实输入、全零、4 倍与取负输入均通过新分开调用/完整 FFN 的 middle、Down 逐位对照；每个计时调用也在计时外核对。真实输入的新旧最终输出 relative L2 为 `1.12736703e-8`。这仍是单个 FFN 结果，不是全模型速度或质量验收。

主机相关 CPU 测试、板端 SwiGLU/完整 FFN/普通 W4A8/chain/I4/量化/归约测试通过；完整 FFN 还覆盖非有限 middle 在 Down 前拒绝、相同 workspace 重用，以及两 partial consumer 返回错误或抛异常后的恢复。频率恢复检查通过。证据见 [汇总](evidence/w4a8_production_20261022/summary.json)、[协议与二进制哈希](evidence/w4a8_production_20261022/protocol.json)、[原始 CSV 0](evidence/w4a8_production_20261022/real-0.csv)、[原始 CSV 1](evidence/w4a8_production_20261022/real-1.csv)。实验工具与数据不进入算子库提交。

## 移除回退后的最终实现

按用户要求移除 W4A8 激活的 FP64/四组回退、重复的私有几何/指针/stride 检查与重复空指针判断。Down 直接引用被锁定的上游 scale/inverse，不再复制，`activation_scan_us=0`。精度取舍在源码与公共头文件中保留 NOTE。

全部激活采用同一套 FP32 运算；指数用 SIMD 无分支饱和至 [-80,80]，gate<-80 输出带符号零，避免指数位重建发生回绕。这是近似公式的范围处理，不切换到其他激活核。直接删除范围处理的初次对照在 4 倍真实输入上出现约 146.08 的最终输出 relative L2；该负面记录保留在 `evidence/w4a8_cleanup_20261022/without-saturation/`，不作为最终正确性或性能结果。

最终主机六项 CPU 测试、公共 C 头文件、板端 SwiGLU/FFN/普通 W4A8/chain/量化/归约、动态与静态 FFN（含 NPU Down 归约）均通过。激活在 [-80,80] 及大幅有限值样本的最大归一化相对误差约 `2.058e-7`。真实、零、4 倍和取负输入的新分开调用/完整 FFN middle 与 Down 逐位一致；相对旧库最终输出 relative L2 分别为 `1.12736703e-8`、0、`3.00984697e-10`、0。单个真实 FFN 回归不替代全模型质量验收。

最终性能独立一批：两个进程各 20 个 ABCCBA 周期，每模式 80 次调用，同样 CPU4–7/2.256 GHz、NPU mask7/1 GHz、DDR 未控制。旧库 9.308 ms，新分开调用 8.212 ms，新完整 FFN 8.084 ms；完整 FFN 相对旧库降低 13.15%，相对新分开调用降低 1.56%，两组比较均 40/40 个周期更快。频率恢复通过。这里不把跨批次的时延差归因于清理本身。

最终签名提交：`6df8ec899f5e8b7757b73911da25baff9178ef9e`，amend 替换了清理前的 `905ee2b`，没有新增提交。实验工具与数据均未提交。完整记录：[协议](evidence/w4a8_cleanup_20261022/protocol.json)、[汇总](evidence/w4a8_cleanup_20261022/summary.json)、[CSV 0](evidence/w4a8_cleanup_20261022/real-0.csv)、[CSV 1](evidence/w4a8_cleanup_20261022/real-1.csv)。

## 接入前的收敛判断与边界

2026-10-22 补充静态检查：本轮停止扩展完整 NPU 收尾链，优先整理已经测得收益的 CPU 路径。这个判断不是认定硬件做不到，而是现有实现尚不足以构成保持当前量化语义、且有明确性能优势的替代链。

- **NPU 动态量化输出与 Down 输入不匹配。** 已验证的链为 FP16 block8 → INT8 block16；当前 W4A8 的 [输入 producer](../native/rk_npu_matmul/src/rk_npu_w4a8_quant.cpp) 还要把 A8 拆成 high/low 两路 A4，按 panel8 写入两倍物理行。现有链没有这段转换。退回 CPU 转换会重新引入同步与张量读写。
- **完整行 scale 依赖仍然存在。** [公共量化契约](../native/rk_npu_matmul/include/rk_npu_w4a8.h) 要求每行全部 K 使用同一个动态 scale，不能只就绪一块 N 就独立量化。lowlevel 的 [已有动态量化链](../../lowlevel/docs/npu_cpu_offload_todo.md) 使用八个任务；M32/C2560 的历史测试为 249.96 μs 完整调用，对照 CPU fused quant/pack 为 21.95 μs。这是另一形状上的负面成本证据，不能直接外推当前 M128/F9728 的时延。
- **可复用的反量化改变数值契约。** [I8 DPU chain](../native/rk_npu_matmul/src/rk_npu_i8_dpu_chain.cpp) 把行与通道 scale 转为 FP16；它不是当前 FP32 scale 路径的逐位替代。之前真实输入的 NPU SwiGLU 完整 FFN 为 10.793 ms，对照 CPU 为 9.416 ms；这也不能直接充当尚未实现的全 NPU 融合链性能。

正式接入应分开处理三项：

1. 两 K 波直接消费放进算子库 executor/reducer，明确两个 partial 的 CPU 读窗口生命周期，不保留诊断构建的源码替换或 consumer 标记。保留原 FP64 激活的版本已测得 6.20% 完整 FFN 收益，并通过所测输入的逐位对照；先限定已支持的几何，其他路径回退原实现。
2. FP32 激活单独形成可选数值策略。当前公共 SwiGLU 契约写明 AArch64 使用 FP64 多项式；真实 layer2 的约 1e-8 相对误差不能替代其他输入、形状或全模型验证。不能在整理代码时顺带全局修改共享激活默认值。
3. maxabs/scale 复用需要正式的元数据交接接口。当前 Rust 的 `run_swiglu` 与 Down `run` 是两个独立调用，单把收集最大值的 kernel 合入不会自动获得本报告的 2.50%。算子库负责生成和消费行 scale，调用侧传递其所属输出、有效行数与生命周期；本轮未修改公共 ABI 或 runtime。

剩余最值得小范围验证的是**按融合调用重新调分块**：[现有 tuner](../native/rk_npu_matmul/src/rk_npu_w4a8_tune_data.cpp) 实际测量 `run_f32/run_f16`，而 [runtime](../src/npu.rs) 复用相同策略运行 `run_swiglu`。两波融合改变了 CPU/NPU 重叠，旧分块不保证仍最优，但当前没有实测收益。

按行就绪后继续融合量化打包也有潜力，不过本批 Down quant/pack 的约 0.91 ms 是与 NPU 重叠的累计跨度，不能当成可直接删除的完整调用时延。它还涉及当前按 N 分工与整行 maxabs 的协调，优先级低于整理已验证实现。当前没有发现另一项像删除 19 MiB accumulator 读写那样明确、低成本的大收益；这不构成“已经最优”的结论。

以上为接入前的收敛检查记录；后续默认精度和正式接口按用户指示调整，实际实现与回归见前一节。

## 实现与复现

- [融合内核与行最大值归并](../native/rk_npu_matmul/tests/tools/w4a8_cpu_fused2/fused_two.cpp)、[元数据结构](../native/rk_npu_matmul/tests/tools/w4a8_cpu_fused2/fused_two.h)。
- [诊断构建](../native/rk_npu_matmul/tests/tools/w4a8_cpu_fused2/build.py)：生成寄存器 maxabs 激活函数以及接收预计算 scale 的 W4A8 诊断入口。
- [完整 FFN 对照](../native/rk_npu_matmul/tests/tools/w4a8_cpu_fused2/maxabs_probe.cpp)、[板端运行与汇总](../native/rk_npu_matmul/tests/tools/w4a8_cpu_fused2/run_maxabs.py)。
- [主要汇总](evidence/w4a8_maxabs_fusion_20261022/register-paired/summary.json)、[协议](evidence/w4a8_maxabs_fusion_20261022/register-paired/protocol.json)、[进程 0 原始 CSV](evidence/w4a8_maxabs_fusion_20261022/register-paired/pair-0.csv)、[进程 1 原始 CSV](evidence/w4a8_maxabs_fusion_20261022/register-paired/pair-1.csv)。同目录保留逐项正确性日志和 clocks JSON。

板端从外层仓库根目录执行：

```bash
python3 native/rk_npu_matmul/tests/tools/w4a8_cpu_fused2/build.py --maxabs \
  --library native/rk_npu_matmul/build/w4a8-swiglu-npu/library/librk_npu_matmul.a

python3 native/rk_npu_matmul/tests/tools/w4a8_cpu_fused2/run_maxabs.py \
  <BOARD_HOME>/mnt/3090x4/tmp/w4a8-swiglu-npu-20261020 \
  <BOARD_HOME>/mnt/3090x4/tmp/w4a8-swiglu-npu-20261020/trace128/p0-l2-post.f32 \
  llm_generated_docs/evidence/w4a8_maxabs_fusion_20261022/replay \
  --cycles 40 --processes 2
```
