# W4A8 完整执行流程与流水潜力复核

2026-10-20。依据当前算子库源码、已有单算子原始时间线复核；本次没有新增板端测量。

FFN down 的后续单算子结果与逐波分析见 [Down 关键路径与 3T 预算](QWEN3_FFN_DOWN_BOTTLENECK_20261020.md)：该形状主要受主干 K 波中的 CPU 输入准备/partial 处理限制，最终尾部占比明显低于本页主要分析的 Gate/up。

**M128 Gate/up 的关键路径已经落在 CPU 的输出同步、整数归约和 SwiGLU 上。下一步最值得尝试的是输出同步与 CPU 消费之间的流水，同时修正非最终归约的工作分配。单独提前提交更多 NPU wave，能够减少设备空档，但按当前阶段耗时几乎不能缩短完整调用。**

## 1. 当前算子究竟执行什么

本次主要形状是逻辑 M128、K2560、N19456，融合 SwiGLU 后输出 F9728。真实 layer1 codes/scales、固定合成 FP32 输入，四个大核 CPU 线程、三核 NPU。数据和频率协议见 [N 流水报告](QWEN3_W4A8_N_PIPELINE_20261020.md)。

### 一次性准备

[rk_npu_w4a8.cpp](../native/rk_npu_matmul/src/rk_npu_w4a8.cpp) 创建权重时完成：

- 选择有整数范围保证的 Ktile，本形状为 1472，因此有 1472/1088 两个 K wave。
- 将 B 的 INT4 codes 按 K wave、N64 block 打包到常驻设备内存。
- 保存每输出通道的 weight scale，并预计算 `correction[n] = 8 * sum_k B[k,n]`。
- workspace 预先分配输入/partial 双缓冲、CPU INT32 accumulator、命令链，建立 gate/up 配对描述符，并创建 submit worker。

这些步骤不在稳态调用中重复。当前问题并非每次重新打包权重或重新调优。

### 每次调用

1. **完整 K 的逐行 scale 扫描。** 对每个 token 计算 `s[m] = max_k(abs(A[m,k])) / 127`，处理全零行并检查非有限输入。
2. **逐 K wave 量化并打包。** `q = round(A/s)`，范围为 -127..127，拆成两个有符号 INT4 数，使 `q = 16*h + l + 8`。每个逻辑 M 行变成 high/low 两个物理行，所以 NPU 实际 M 为 256。
3. **输入 cache clean 后提交三核 GEMM。** N 沿三个核切分；每核内部是一条包含其全部 M/N tile 的 PC chain。本形状每核两个 task，一次提交共六个 task。
4. **wave 完成后 invalidate partial。** partial 是原生布局 `N8 × physical_M × 8` 的 INT16 数据，CPU 必须等对应 wave 完成并同步后才能读。
5. **非最终 wave 做整数归约。** 第一波写入 `acc = correction + 16*P_high + P_low`；后续非最终波向 acc 累加。
6. **最终 wave 融合归约、scale、输出。** 普通 linear 直接生成 FP32/FP16；Gate/up 路径同时取同一输出通道的 gate、up，反量化并计算 SwiGLU，直接写 `[M,F]`，省去了完整 gate/up 浮点中间张量。

相关实现分别在 [quant.cpp](../native/rk_npu_matmul/src/rk_npu_w4a8_quant.cpp)、[i4_plan.cpp](../native/rk_npu_matmul/src/rk_npu_i4_plan.cpp)、[output.cpp](../native/rk_npu_matmul/src/rk_npu_w4a8_output.cpp)。当前 SwiGLU 使用 [既有 FP64 degree-12 exp 实现](../native/rk_npu_matmul/src/rk_npu_swiglu_neon.h)；这里讨论的调度不改变数值算法。

### 已经存在的重叠

[I4Executor::run](../native/rk_npu_matmul/src/rk_npu_i4_executor.cpp) 的 `pipeline=1` 沿 **K wave** 流水：

```text
pack(0)
对每个 w：
    submit_worker 启动 NPU(w)
    CPU pack(w+1)，若存在
    CPU sync/reduce(w-1)，若存在
    等待 NPU(w) 完成
CPU sync/reduce/finish(最后一波)
```

因此，两个 wave 的本形状已重叠 `NPU(K0) / pack(K1)`，以及 `NPU(K1) / sync(P0)+reduce(P0)`。M/N tile 目前是 wave 内部的存储和硬件任务切分，单纯改变 tile 大小不会产生新的 CPU 可消费边界。

输入与输出各两个槽。多 K wave 时，在再次写同一个 partial 槽前，前一份 CPU 消费必须结束。这是真实的缓冲复用依赖；添加缓冲可以提前提交，但是否降低延迟取决于 CPU 是否还在关键路径上。

## 2. 关键路径：为什么 NPU 更忙不等于调用更快

重新关联已有同进程交错测试中的 CSV、submit 和 CPU 阶段事件，每种方法各 120 次。以下完整时间来自带插桩的那组；不插桩正式结果仍是 6.731/6.639 ms。

| 可相加的边界分解，ms | 原始 K 流水 | 两组 N 流水 |
|---|---:|---:|
| 调用开始至第一份 partial 可用 | 1.662 | 1.077 |
| 之后 CPU 输出同步累计 | 1.073 | 1.267 |
| 之后 CPU 归约及 finish 累计 | 3.991 | 4.274 |
| 其余交接、计时及调度差额 | 0.011 | 0.025 |
| 完整调用 | 6.736 | 6.643 |

这里 CPU 阶段是调用线程经历的墙钟跨度，包含 OpenMP 团队执行/汇合，不是四核 CPU time 之和。

**两种方法的 120/120 个样本，在初始填充结束后，各次 CPU 工作结束时，对应需要等待的 NPU 提交都已返回。** 也就是说，后续没有实质的“CPU 计算完还在等 NPU”区间。

原始路径的平均时间边界：

| 工作 | 开始 ms | 结束 ms |
|---|---:|---:|
| NPU K0 | 0.197 | 1.662 |
| NPU K1 | 1.677 | 2.962 |
| CPU P0 同步 | 约 1.675 | 2.374 |
| CPU P0 整数归约 | 2.374 | 3.438 |
| CPU P1 同步 | 约 3.440 | 3.809 |
| CPU P1 最终归约及 SwiGLU | 3.809 | 6.734 |

两个输出同步分别约 0.699/0.369 ms；非最终归约约 1.064 ms；最终归约与 SwiGLU 约 2.925 ms。最后 3.78 ms 的设备空档包含 CPU 等待数据可见、整数处理和激活，并非还有一个可直接提前的 GEMM 依赖。

N 流水使第一份 partial 提早约 0.585 ms，但输出同步与归约跨度增加约 0.478 ms，其余交接增加约 0.015 ms，最后只剩约 0.093 ms 收益。

保持实测 CPU 各段耗时及首份结果时刻不变，原始/N 流水的串行 CPU 路径分别需要约 **6.725/6.618 ms**；距离各自实测完整时间仅 **0.011/0.025 ms**。这是用于判断调度余量的固定成本模型，不是改变同步、缓存、工作划分之后的性能上限。

### N 原型存在人为提交依赖，但它目前不在主要延迟路径上

[n_pipeline.cpp](../native/rk_npu_matmul/tests/tools/w4a8_utilization/n_pipeline.cpp) 会在 `finish(N0)` 期间提交 `N1/K0`，等 finish 返回后才发 `N1/K1`。两组有独立 partial，两个 A wave 都已打包，所以数学上并不需要这次等待。

时间线上，N1/K0 平均在 2.854 ms 结束，N1/K1 到 3.997 ms 才开始，中间约 1.143 ms 空档。可以让 submit worker 自动推进已就绪任务，把这个空档消掉；但 CPU 此时正在 finish N0。随后 CPU 继续同步/归约 N1，现有 N1/K1 已经能在 CPU 需要它之前完成。

所以，自动推进是合理的调度结构，也是更深流水的基础，**单独改它不应被预期能省下这 1.143 ms**。它可能只是把更多设备访存移到 CPU 正在运行的区间。

同样的两组 N、相同 CPU 算法，串行控制的输出同步/归约是 0.821/3.872 ms，流水时为 1.267/4.274 ms。并发时两个阶段合计变慢约 0.848 ms。这支持存在共享资源争用和缓存行为变化，但现有事件不足以进一步分解 DDR、cache maintenance、线程调度各占多少。

## 3. 优先尝试：输出同步与 CPU 消费之间的流水

当前 [I4Plan::consume](../native/rk_npu_matmul/src/rk_npu_i4_plan.cpp) 严格执行：

```text
整个 partial FROM_DEVICE 同步完成
    -> 四线程归约/finish 整个 partial
    -> 返回
```

[DomainDataBuffer::begin_cpu_read](../native/rk_npu_matmul/src/rk_npu_core.cpp) 通过同步 ioctl 完成整块缓存维护。CPU 消费线程在这时被阻塞，随后才启动 OpenMP 消费；读操作的 `end_cpu_access()` 不会再做一次 clean。

本形状每份 partial 为 **9.5 MiB**，两波需要同步 19 MiB。相比之下，两波输入同步总共只有约 0.046 ms。因此应瞄准输出同步，而非继续优化十几微秒的输入交接。

可以保留当前大块 NPU 任务，把**已经完成的** partial 切成 CPU 消费区间：同步 worker 处理下一段，CPU 团队处理上一段。已有 `rk_npu_mem_view` 与 sync 的 offset/size 能表达范围，无需改变 GEMM tile 或增加 NPU submit。

具体可分两步：

1. 在 CPU 仍归约 P0 时，一旦 NPU K1 完成，另一个 worker 提前同步独立的 P1。原始时间线里 K1 在 2.962 ms 完成，P0 归约到 3.438 ms 才结束，约有 0.476 ms 窗口，而 P1 同步本身约 0.369 ms。两者量级匹配，值得先做最小探针。
2. 将 P0/P1 各自的完整同步也分成较大的区间，使本段的同步与上一段的归约或 SwiGLU 重叠。Gate/up 必须让两个配对区间都就绪后才消费；不能读取尚未完成或未同步的区域。

先测 0.5–2 MiB 量级的消费区间，避免每 N8/N32 发一个 ioctl。同步 worker 和消费团队通过 ready 状态交接；BO 在消费者结束前不能重用。两组 N 原型已有独立 BO，原始两 K 路径也有两个独立输出槽。

第一步的 0.369 ms、全部输出同步的 1.073 ms 是潜在可覆盖工作量，**不是加速承诺**：额外 CPU 核分配、缓存维护与计算并发、范围 ioctl 和缓存局部性都会改变成本。需要比较同样分段的串行/流水版本，并按完整调用计时。

## 4. 先修正一个明确的 CPU 分工问题

[output.cpp 的 `reduce_tiles<..., Final=false>`](../native/rk_npu_matmul/src/rk_npu_w4a8_output.cpp) 仍对 `tiles.size()` 做 OpenMP static 分配。最终 finish 已按更细的 CPU 工作量切分，非最终归约却直接沿用 NPU tile。

当前六个 tile 的 N 宽度：

```text
4096, 2432, 4096, 2368, 4096, 2368
```

按四线程常规连续 static 分配，线程得到的列数是：

```text
6528, 6464, 4096, 2368    理想均分：4864
```

两组 N 原型中，每组六个 tile 则变成：

```text
1664, 1664, 1600, 1600, 1600, 1600
线程工作量：3328, 3200, 1600, 1600    理想均分：2432
```

所以缩小 N 并没有消除这个不均。应保持 NPU 的几何不动，将 CPU 非最终归约按连续 N8/若干行的工作区间均分；每个线程拥有独立 cache line，继续使用现有连续 NEON kernel。

仅按工作量线性估算，原始非最终归约的阶段时延有约 25.5% 的均分空间，对应 1.064 ms 中约 0.27 ms。它不包含带宽饱和及缓存变化，尚未板测。这个问题是源码确定的分工限制，具体收益需要测量。

建议先完成此项小改动，再测同步流水，避免把 CPU 负载均衡收益误记成流水收益。

## 5. 两个 K partial 可以直接融合消费，减少流水中的工作量

当前两波先把 P0 重建成完整 INT32 accumulator，再在最终 finish 中读 accumulator 和 P1。对这一个常见的两 K 波形状，可直接计算：

```text
correction + 16*P0_high + P0_low + 16*P1_high + P1_low
    -> 原有 scale 乘法顺序
    -> 原有 SwiGLU
```

整数逐项累加不改变结果，scale 和激活算法可保持原样。该路径可以省掉一遍 9.5 MiB accumulator 写入和一遍 9.5 MiB 读取，合计 **19 MiB CPU 访存指令覆盖的数据量**；不是实测 DDR 总线流量。两个 partial 本来就都有独立槽，并不必然要求更多 partial 存储。

代价是不能再提前完成单独的 P0 重建，需要等两个 partial 都就绪才能做融合消费。单独把它放在全部 GEMM 之后未必更快；适合与输出同步预取、N 组或完整 M 组的流水一起比较。它的价值是减轻 CPU 阶段，而不是单纯增加重叠比例。

## 6. 保留大块效率的另外两条结构路线

### Gate/up 成对打包，减少 N 流水额外增加的 task

N 原型仍保留 `[gate(F), up(F)]` 权重列顺序。每个核、每个 N 组都分别发 gate 和 up 两个较窄 task：两组、两 K 波一共 **24 个硬件 task**，原始路径是 **12 个**。

如果一次性把权重按 `gate32, up32` 配对重排，每个 N64 block 正好包含一对输出，scale/correction 和 CPU 配对描述符同步重排。每组 N9728 可以切为 3264/3264/3200，三核各一个 task；两组、两 K 波总 task 数恢复到 12，submit 仍为四次。CPU 最终按原始通道顺序写回。

这保留整数运算和输出语义，不做运行时权重搬运。它可能减轻小任务和跨远距离 gate/up 读取的成本，但不会自动消除 CPU 关键路径；优先级在同步流水和 CPU 分工之后。

### 真正的 M 组流水，保留每组 M128

之前的 M 实验是把一次调用内部的 tile 缩成 64/32/16，仍在同一个 wave 的 PC chain 里完成。这不同于将一个大 M 调用分成多个**完整 M128 组**：某组所有 K 都完成后，CPU 消费该组，同时 NPU 执行下一组。

M 行之间没有 scale 或归约依赖，每行 scale 只要求本行完整 K。若一次单算子已经有 M256/M512 输入，可保留 M128 的 NPU tile，按 M 组改变完成边界，不必付出把 M128 硬件 tile 缩小的效率损失。权重常驻并复用，CPU/设备至少需要两个组的独立状态。

这改善的是多组的填充/排空摊销。当前孤立 M128 没有后续完整 M128 可以覆盖，不能直接套用多组收益。当前模型在 [dense.rs](../src/model/dense.rs) 中按 chunk 遍历所有层，尚不是同层四个 M128 连续发射；未来接入需要单独调整外层调度。先在算子库中做 M512 对四次 M128 的完整输出与时间对照即可判断算子层潜力。

跨算子若考虑 Gate/up → Down，还有一个硬依赖：Down 的动态激活 scale 要扫描每行完整 F。只有若干 N 列的 middle 就绪，不能直接按现有量化语义开始 Down；完整 M 行组就绪则可以。当前 SwiGLU 按 N32 pair 再遍历 M 行消费，若要早交付完整行，还需调整 CPU 消费顺序及接口。

## 7. 其它 W4 linear 的重点不同

同一执行器中，O 有三 K 波，Down 有六 K 波。既有 CPU2.256GHz 单算子样本的阶段跨度如下，来自较早那组实验，不能与上述交错实验的绝对时间混成同一个 A/B：

| 算子 | A scan ms | A pack 累计 ms | 输出 finish ms | submit 之间空档 ms |
|---|---:|---:|---:|---:|
| O | 0.103 | 0.421 | 0.256 | 0.076 |
| Down | 0.265 | 1.189 | 0.219 | 0.439 |

Down 的 pack 比 Gate/up 更值得关注。在多波循环中，`pack(w+1)` 与 `sync/reduce(w-1)` 都在同一条 CPU 控制路径上串行，随后才允许发下一波。可以研究提前生产 A、预先打包更多 wave 或为 producer/reducer 分配不同 CPU 资源；但多开两个各占四个大核的 OpenMP team 会争用，不能默认获得加速。

Gate/up 的 pack 总共约 0.167 ms，且 K1 的 pack 已被 K0 覆盖，因此此方向对当前 Gate/up 的优先级低。all-K chain 的现有实现减少提交次数，却把 CPU 消费整体放到设备任务完成后，不会凭空覆盖最后的 CPU finish。

## 8. 建议的最小验证顺序

1. **只均分非最终 CPU 归约。** 保留硬件任务、提交次数、同步粒度和数值算法，测它能否降低约 1.064 ms 的阶段跨度与完整调用。
2. **只提前同步下一份已经完成的 partial。** 先瞄准 P0 归约期间的 P1 同步窗口，随后再做区间同步与消费流水。通过相同分段的串行控制区分局部性变化与真实重叠。
3. **两 K partial 直接融合消费。** 与前两项配合，检查节省 accumulator 访存是否补偿失去的提前归约；完整输出逐字节对齐。
4. CPU 路径有实质下降后，再复核成对权重布局的 N 流水和 M128 组流水，使设备生产速度与新的 CPU 消费速度匹配。

都可以在算子库的 C++ 单算子测试中完成。正式时间仍使用同进程交错、关闭事件记录的完整调用；阶段 trace 单独解释机制，包含同步和最终排空。

### 证据与复算

- [新增依赖统计](evidence/w4a8_utilization_20261020/dependency-analysis.json)：保留每次调用、相对 submit/reduce 区间、初始填充后的等待和固定 CPU 成本模型。
- [复算工具](../native/rk_npu_matmul/tests/tools/w4a8_utilization/analyze_dependencies.py)：仅读取既有文件，无板端执行。
- [已有 N 流水报告](QWEN3_W4A8_N_PIPELINE_20261020.md)、[原始时间线目录](evidence/w4a8_utilization_20261020/n-pipeline-paired/)、[无事件记录对照](evidence/w4a8_utilization_20261020/n-pipeline-unrecorded/)。
- 参考 lowlevel 的[缓存同步实验](../../lowlevel/docs/cache_sync_experiments_2026-09-17.md)：输入 clean、输出 invalidate、同步范围及并发的历史结果。上述建议继续使用驱动的 FROM_DEVICE 同步，不假定同步可以省略。

从外层仓库根目录复算：

```bash
python3 native/rk_npu_matmul/tests/tools/w4a8_utilization/analyze_dependencies.py \
  llm_generated_docs/evidence/w4a8_utilization_20261020/n-pipeline-paired \
  --output llm_generated_docs/evidence/w4a8_utilization_20261020/dependency-analysis.json
```
