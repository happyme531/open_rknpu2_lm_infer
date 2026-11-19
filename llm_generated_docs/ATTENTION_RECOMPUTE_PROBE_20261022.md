# Attention 重算 QK 的板端实验

额外计算一次 QK、把减行最大值并入第二遍 GEMM、在 DPU epilogue 融合 LUT，可以用更多 NPU 计算换掉 score 的 CPU 往返。三核实验在已经优化 CPU 部分的基线上，Qwen3 长 KV prefill 块降低 25.0%，Qwen3.5 降低 18.7%，ViT KV1024 降低 33.2%。短验证与 decode 不适合这条路径。

这是单算子原型。重算改变 FP16 舍入，真实 C7 输出相对基线的 relative L2 为 0.0696%；尚未做整模型 PPL 或端到端吞吐验证。

后续的增量 KV、公共 prefill、真实模型与 PPL 验证见 [ATTENTION_STREAM_20261023.md](ATTENTION_STREAM_20261023.md)。下面保留本轮热态原型的原始计时边界。

## 实际数据路径

第一阶段是一个设备任务链：

`QK（causal 时融合 additive mask）→ 原生 DPU ReduceMax`

CPU 只读取每行的 FP16 最大值，把已缩放 Q 的附加通道填为 `-max`。K 的附加通道为常数 1，其余 K32 对齐通道为零。第二阶段是另一个设备任务链：

`[Q_scaled,-max] × [K,1] → 融合 LUT → 必要的尾块 mask → PV`

随后沿用现有 FP32 partial 累加、carrier 修正和归一化。Qwen3 D128 的附加 K 对齐为 160；Qwen3.5 D256 对齐为 288；ViT D64 对齐为 96。

causal 的行最大值从已屏蔽未来位置的 QK 取得。LUT 后的正 carrier 在 PV 前由独立 DPU multiply-mask 任务精确归零；只对含不可见位置的 PV tile 添加该任务。有效位置继续使用现有 `g=255/256`、`delta=1/2048` 的修正。padding 也在这条路径中处理。

当前第一遍 QK 仍写完整 score 到 DDR，由 ReduceMax 读取。没有宣称消除了设备侧 score 张量，也没有把 CBUF 当作可编程共享 SRAM。已移除的是正常热态执行中的整张 score CPU 同步、max/subtract 和独立 LUT 遍历；CPU max oracle 仅在首个、未计时的预热调用中运行。

## 对照与计时范围

- native 基底：`6df8ec899f5e8b7757b73911da25baff9178ef9e` 的冻结静态 archive。
- 板子：备用板 1 `备用板 1`，driver 0.9.6；GCC 12.2/O2，CPU4–7 固定 2256 MHz，NPU 三核固定 1 GHz。DDR 无 DMC devfreq，未受控。
- 两边均启用上一轮的 CPU 半精度 centering、硬件 FP16 转换、四线程 Q pack；两边共用相同的 pack/finish 实现。MHA 基线已经使用 128 行工作块。
- baseline 为原有 `Workspace::run`，实验为两个已准备任务链的完整执行。计时从紧凑 Q 开始，到 FP32 O 返回，包含 Q pack、max 向量同步、扩维 Q 更新/同步、设备提交、输出同步和 CPU finish。
- 这是一个内部工作块，不包含公共 `run_gqa` 的分段规划；不含首次构图、KV load、扩维 K 和 mask 的准备。两边均预热三轮。
- 每个几何运行三个独立进程，每进程交替 AB/BA，两个方法各 30 次，合计 1440 个正式样本。表中每种方法取三个进程内 median 的均值。
- 七组为固定种子 `20261022`、标准差 0.5 的合成 FP16 Q/K/V，匹配模型几何；C7 一组来自真实 `prefix1021/R4` 捕获。

这与上一份报告的公共 Q128 调用边界不同，不能将下面的 Q32 时间直接当作完整 prefill，也不能不计状态更新就乘四外推。

## 配对结果

| 几何 | Hq/Hkv，D | Q / KV | CPU 优化后基线 ms | 重算路径 ms | 延迟下降 | 三轮下降范围 |
|---|---|---|---:|---:|---:|---:|
| Qwen3 causal | 32/8，128 | 32 / 512 | 1.174 | 0.964 | 17.90% | 11.01%–21.21% |
| Qwen3 causal | 32/8，128 | 32 / 4096 | 5.939 | 4.453 | 25.02% | 20.69%–31.74% |
| Qwen3.5 causal | 16/4，256 | 32 / 512 | 0.924 | 0.783 | 15.25% | 15.06%–15.51% |
| Qwen3.5 causal | 16/4，256 | 32 / 4096 | 3.904 | 3.176 | 18.67% | 15.01%–24.28% |
| ViT no mask | 16/16，64 | 128 / 256 | 0.942 | 0.729 | 22.67% | 14.44%–26.63% |
| ViT no mask | 16/16，64 | 128 / 1024 | 2.518 | 1.682 | 33.23% | 28.95%–40.27% |
| 真实 C7 causal | 16/4，256 | 4 / 1025 | 0.512 | 0.527 | −2.90% | −7.50%–2.34% |
| Qwen3 decode | 32/8，128 | 1 / 4096 | 0.896 | 1.164 | −29.93% | −31.63%–−26.91% |

范围是实际三轮结果，不是置信区间。CPU 阶段仍有波动；六个 prefill 几何每一轮均快于基线。C7 短验证的差值跨零，decode 的退化稳定。

原始样本和逐轮命令在 [formal/summary.json](evidence/attention_recompute_20261022/formal/summary.json)，时钟和恢复状态在 [formal-clocks.json](evidence/attention_recompute_20261022/formal-clocks.json)。

## 提速机制的证据

Qwen3 Q32/KV4096 的全部样本中，原路径 driver hardware median 约 2.831 ms；重算的两段设备区间分别约 1.397 和 2.272 ms，合计约 3.669 ms。NPU 执行区间增加约 30%，完整工作块仍加速约 25%。

重算路径的阶段 median 约为：Q pack 0.080 ms、QK+max submit 1.407 ms、max 同步及扩维 Q 0.050 ms、第二遍 QK/LUT/mask/PV submit 2.279 ms、输出同步及收尾 0.633 ms。阶段 median 与三个进程 median 的均值属于不同聚合，不能要求精确相加。

该几何的 score 为 8 MiB，而回到 CPU 的 max 只有 `8*128*2 = 2048 bytes`。扩维 Q 的逻辑尺寸为 320 KiB。设备侧仍支付第一遍 score 写/read、第二遍 QK、mask 和 KV 流量；收益来自避免 CPU 处理大 score，以及把 exp 合入矩阵乘的后处理。这是算子内数据路径改变的证据，不是 DDR 带宽或真实 bytes 的计数器测量。

## 数值与运行状态

- 各路径在同一进程内重复输出逐字节一致，三个独立进程之间的完整输出也逐字节一致。
- 新旧路径之间存在数值差异。七组合成数据 relative L2 最大约 0.0431%；真实 C7 为 0.0696%，max abs 0.004343。
- 每个几何的第一轮都将设备行 max 与 CPU 对本次原生 QK 的全量扫描逐值比较。causal 路径同时检查全部未来概率在 PV 前精确为零。
- 额外 22 个正确性用例通过：M4/16/32/64/128 的随机、零和全负 QK max；causal/no-mask 的 KV129 非整块尾部，分别使用随机、零 query、全负 logits；真实 C7 回放。
- 按最终构建脚本重编译后的真实 C7 完整输出，与正式测量文件逐字节一致。
- 最后的原库 Attention canary 通过；正式测量、正确性检查和 canary 的时钟均恢复，`restore_errors=[]`。

检查明细：[checks/summary.json](evidence/attention_recompute_20261022/checks/summary.json)、[checks-clocks.json](evidence/attention_recompute_20261022/checks-clocks.json)、[final-canary.jsonl](evidence/attention_recompute_20261022/final-canary.jsonl)。上述局部差异不构成整模型 PPL/采样分布的保证。

原生 ReduceMax 的必要维度字段已从官方抓取核对：

`DPU_FEATURE_MODE = 0x020001e5 | (align_up(M,8) << 6)`

M4/8/16/32/128 分别对应 `0x020003e5/0x020003e5/0x020005e5/0x020009e5/0x020021e5`。使用这些配置时输出是紧凑的 M 个 FP16 标量；不能把 M32 的配置常量用于 M128。抓取文件的 git revision 和字段证据在 [provenance.json](evidence/attention_recompute_20261022/provenance.json)。

## 实验代码

- [device_common.h](../native/rk_npu_matmul/tests/tools/attention_pipeline/device_common.h)：任务链、三核分区及完整 ReduceMax body。
- [device_primitives.cpp](../native/rk_npu_matmul/tests/tools/attention_pipeline/device_primitives.cpp)：独立 QK/ReduceMax 正确性与输出 guard。`flying`/`flying-convert` 是未通过数值验证的显式诊断模式，不在验收脚本中使用。
- [recompute.cpp](../native/rk_npu_matmul/tests/tools/attention_pipeline/recompute.cpp)：扩维 K/Q、fused QK/LUT、causal mask、PV 与交替测量。
- [build_device.py](../native/rk_npu_matmul/tests/tools/attention_pipeline/build_device.py)：板端构建；[run_recompute.py](../native/rk_npu_matmul/tests/tools/attention_pipeline/run_recompute.py)：配对测量；[run_device_checks.py](../native/rk_npu_matmul/tests/tools/attention_pipeline/run_device_checks.py)：22 项正确性检查。

实验文件、冻结 archive 和输出位于主机 `<HOST_HOME>/mnt/3090x4/tmp/attention-firstprinciples-20261022/`，备用板路径为 `<BOARD_HOME>/mnt/3090x4/tmp/attention-firstprinciples-20261022/`。源码改动均在 tests 下，没有新增生产 API 或默认开关。

在板端快照目录复现，先完成构建，再运行测量：

```sh
python3 native/tests/tools/attention_pipeline/build.py \
  --base-archive build/librk_npu_matmul.a --output-dir diagnostic
python3 native/tests/tools/attention_pipeline/build_device.py \
  --base-archive build/librk_npu_matmul.a \
  --cpu-opt-object diagnostic/attention.o --output-dir device/final-bin
python3 run_board.py --cpu-khz 2256000 --skip-dmc --report clocks.json -- \
  taskset -c 4-7 python3 native/tests/tools/attention_pipeline/run_recompute.py \
  --binary device/final-bin/recompute-opt --output-dir replay \
  --fixture ../qwen35-dev/mtp-20261018/cpu-softmax-20261019/phase2-p1021-r4
```

## 进入生产路径之前

1. **先完成变长缓存成本。** 当前扩维 K 和 mask 在准备阶段构建。生产中需要增量 append K 的常数通道、更新 mask 和复用图；不能每次 query 重新打包全量 K。第一遍使用原 K、第二遍使用扩维 K，目前还有重复存储。可比较两遍共用扩维 K、第一遍附加 Q=0，以较多 MAC 换较小缓存的方案。
2. **以长 prefill 为候选，保留短调用原路径。** MHA 在已经使用 Q128 的基线上仍有收益；GQA Q32/S4096 也有稳定收益。真实短验证和单 token decode 没有采用依据。现有八个测点不足以拟合通用阈值。
3. **再做真实长 prefill 和 PPL。** 更新KV、mask、图和分段规划全部计时，随后做真实长 QKV、完整 logits、固定 8160-target PPL 及模型 prefill A/B。只有这些条件满足后才值得默认启用。

直接从 QK flying 输出得到 row max 的映射尚未通过数值 gate；当前可用实现是 QK 后接独立 ReduceMax。布尔 mask、长于 4096 的 QK 分片、非有限输入及生产 fallback 的集成不属于本轮已验收范围。
