# W8/W4 MoE CPU 中间处理融合，2026-10-23

本轮在同步后的算子库上优化 MoE CPU middle：FP32 反量化、SwiGLU 与寄存器 maxabs 分块融合，随后直接量化打包到 Down 输入。W8 普通/native-A8 和 W4 high/low A4 共用这条 CPU 路径；dense W4A8 复用同一份 FP32 激活实现。普通 dense W8 FFN 接口不属于本轮改动。

算子库签名优化提交：`c96930f4dac9d9a978712df0fbdce918f5f02cee`，分支 `perf/moe-ffn`；已 amend 纳入提交前清理。

## 同步与实现

- 算子库合并提交 `7d653da`，同时保留 W4A8 融合 `6df8ec8` 和 Ling 最新 `b917811`（含全 W4 MoE、专家 slot bank、N8128 修正和 trace 计时）。Ling 外层 runtime 的未提交改动未复制。
- 公共 MoE C ABI、路由、任务调度与 device 同步顺序不变；CPU middle 默认直接采用新实现。
- 将 W4A8 的 FP32 激活提取为 `src/rk_npu_swiglu_fp32_neon.h`，保留无分支指数饱和及精度 NOTE。没有旧激活核回退。
- `src/rk_npu_moe_cpu.cpp` 每次只反量化 16 列 gate/up，并从激活结果寄存器累计 maxabs。保留一行 hidden；删除独立 maxabs 扫描和 A8 临时数组中转。
- MoE 保留原有 nearest-even 量化与全零行 scale=1；不套用 dense W4A8 的 ties-away 或 FLT_MIN 规则。内部激活及非零 scale/倒数的可表示性作为数值前提写成 NOTE，不在中间处理内验证。
- 提交前清理了内部 NaN/Inf 检查、hidden scale 缓冲/重复扫描、错误标记分支，以及 `RK_NPU_MOE_TRACE` 的逐次 stderr 输出和 stage 参数。公共 API 的输入、尺寸、路由与资源校验仍保留；既有可选 Perfetto trace 不变。

## 测量范围

板卡 备用板 1，CPU4–7/4 线程 @2.256 GHz，NPU mask7 @1 GHz，DDR 未控制。独立算子 API 完整调用，不运行整模型。权重准备、加载和正确性检查不计时。共享 NPU 锁；全部时钟恢复检查通过。

每种模式两个进程，每进程 20 个 ABBA 周期；每个 M/offset/版本共 80 次调用、40 个进程内配对周期。以下是提交前清理内部验证逻辑之前的性能数据，按输入切片分别列出，不把周期视作独立板卡实验。清理后重跑了正确性回归与真实权重对照，没有重跑这套完整性能采样。

- W8：既有 Ling layer1 原始权重生成的 W8 codes/scales，包含 128 个 routed 和一个 shared expert，H1536/I512/top8。输入为原 checkpoint 前置层的 FP32 重建，office corpus，**不是当前 runtime 捕获**；使用 offset0/1792 两个切片，路由固定。来源 `<HOST_HOME>/mnt/3090x4/llm/Ling-3.0-tiny-moe-probe/fixture/metadata.json`。
- W4：当前 `Ling-3.0-tiny-QQQ-W4A8-Mixed-CT` layer1 原始 routed W4 codes/scales，128 个 routed expert，排除单独的 shared W8 expert。输入是既有 runtime trace 的 25 行真实 post-norm 张量，按 token 顺序拼接；路由从 TF32 截断的 router 操作数在 CPU 重建后固定，**不是捕获的路由 dump**。使用其中 M1/9/25。来源 `tests/tools/moe_ffn/export_qqq.py` 生成的 fixture metadata。

## 完整调用结果

单位 ms；“更快”列是 optimized 完整调用均值小于 baseline 的配对周期数。

| 模式 | M | offset | 原完整调用 | 新完整调用 | 时延降低 | 原 middle | 新 middle | 更快 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w8 | 1 | 0 | 0.984 | 0.968 | 1.63% | 0.039 | 0.026 | 25/40 |
| w8 | 1 | 1792 | 0.964 | 0.954 | 1.02% | 0.039 | 0.024 | 27/40 |
| w8 | 128 | 0 | 16.399 | 14.920 | 9.02% | 2.352 | 0.953 | 40/40 |
| w8 | 128 | 1792 | 15.468 | 14.177 | 8.35% | 2.499 | 1.310 | 40/40 |
| w8 | 256 | 0 | 26.044 | 23.599 | 9.39% | 4.505 | 2.173 | 39/40 |
| w8 | 256 | 1792 | 25.367 | 22.697 | 10.53% | 4.657 | 2.175 | 40/40 |
| w8native | 1 | 0 | 0.977 | 0.967 | 0.99% | 0.044 | 0.028 | 25/40 |
| w8native | 1 | 1792 | 0.975 | 0.965 | 0.99% | 0.037 | 0.023 | 26/40 |
| w8native | 128 | 0 | 15.917 | 14.718 | 7.53% | 2.101 | 0.982 | 40/40 |
| w8native | 128 | 1792 | 15.241 | 13.883 | 8.91% | 2.191 | 0.925 | 40/40 |
| w8native | 256 | 0 | 25.507 | 22.889 | 10.26% | 4.249 | 1.769 | 40/40 |
| w8native | 256 | 1792 | 25.030 | 22.282 | 10.98% | 4.342 | 1.849 | 40/40 |
| w4 | 1 | 0 | 0.559 | 0.544 | 2.56% | 0.046 | 0.033 | 36/40 |
| w4 | 9 | 0 | 2.637 | 2.525 | 4.24% | 0.222 | 0.136 | 40/40 |
| w4 | 25 | 0 | 3.903 | 3.557 | 8.87% | 0.656 | 0.394 | 40/40 |

W8 M128/256 的完整 MoE 时延降低约 7.5–11.0%，middle 跨度明显下降。M1 的 W8 收益仅约 1%，且多数对照只有 25–27/40 周期更快，不宜描述成稳定的大幅 decode 加速。W4 捕获输入 M25 的完整 routed MoE 降低 8.87%。这些不是整模型加速比例。

## 正确性与边界

- 主机 `moe_cpu/i4_cpu/w4a8_swiglu_cpu/w4a8_ffn_cpu` 通过；新增 CPU 用例覆盖 W8/W4、多 K partial、三种 Down 布局、逐行 scale/code、零值与行外 guard。测试发现并修正了非 AArch64 scalar native-A8 的 lane 地址偏移。
- 板端 `test_moe_cpu`、W8 normal/native、混合 W4/W8、全 W4（含 slot 替换/排列）、NPU middle 及 dense W4A8 CPU/完整 FFN 回归通过。全 W4 测试链接静态库，其他主要路径链接动态库。
- 真实权重对照覆盖原输入、全零、4 倍与取负输入；所有计时调用的输出均在计时外与对应版本首次输出逐位比较，重复性检查通过。
- 相对旧 CPU 实现，最大完整输出 relative L2：W8 `3.17625727e-05`、native W8 `3.17625727e-05`、W4 `2.54379927e-08`。量化舍入规则未改变，FP32 激活近似可使靠近量化阈值的值改变代码。未做全模型 PPL/质量验收。
- 提交前清理后，上述主机和板端回归再次通过。真实权重复核覆盖 W8 normal/native 的 M128、offset0/1792 与 W4 M25，均包含原输入、全零、4 倍和取负；最大 relative L2 与上面一致。

## 证据

- [配对汇总](evidence/moe_ffn_20261023/paired/summary.json)、[协议与库哈希](evidence/moe_ffn_20261023/paired/protocol.json)、[时钟恢复](evidence/moe_ffn_20261023/paired/clocks.json)。
- 原始 CSV/log：`evidence/moe_ffn_20261023/paired/{w8,w8native,w4}-{0,1}.{csv,log}`。
- 基线与优化回归日志分别位于 `evidence/moe_ffn_20261023/{base,optimized}/`。
- 提交前清理后的编译、回归和真实权重对照日志位于 `evidence/moe_ffn_20261023/cleanup/`，同目录 `clocks.json` 记录恢复状态。
- 单算子工具保留在算子库 `tests/tools/moe_ffn/`；实验代码、输入、结果和构建产物不进入正式提交。
