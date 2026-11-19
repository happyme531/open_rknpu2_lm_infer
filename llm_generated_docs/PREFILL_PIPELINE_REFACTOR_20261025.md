# 共用层级 prefill 流水与算子库版本对照

## 实现

`--prefill-ubatches 1|2` 默认保持 `1`。Dense 和 Qwen3.5 使用同一调度器，前一批
完成整层后，后一批进入该层；两个批次可并发执行相邻层。没有增加 CPU 阶段锁或
算子级调度策略。使用说明见 [PREFILL_UBATCH_SCHEDULING.md](PREFILL_UBATCH_SCHEDULING.md)。

- `src/model/pipeline.rs`：唯一层对象的存储、双线程依赖、错误与取消时排空。
- `src/model/prefill.rs`：完整相邻物理批的配对；尾批保持串行，不跨 checkpoint 边界。
- `src/npu/lane.rs`：为同一算子在执行作用域中选择独立 workspace。模型层、设备权重、
  context/domain 和 canonical state 共用；激活和 scratch 独立。
- Dense 的逐层回调和 Qwen3.5 的批次提交分别留在模型适配层。移除了复制 Layer 和
  Dense 独有的握手逻辑，也移除了 Attention 中途交接回调。
- Qwen3.5 的 GDN 状态、卷积历史和 KV 各只有一份，按原批次顺序更新。流水排空后，
  MTP catch-up、特征提交和 DFlash context 更新仍按原来的物理批顺序执行。
  prepared 输入继续保留三轴位置和注入特征。

逐 token decode、逐行 NLL 和 speculative verification 保持单批。关闭双批时不创建
第二组 workspace。阶段计时在双批时会重叠，不能相加解释为端到端 wall time。
本轮性能对象为量化 decoder；FloatBackend 沿用自身的互斥 workspace，未验证浮点
decoder 的并发收益。prepared 特征回归不是完整视觉 encoder 或 DFlash 端到端性能测试。

## 兼容性验证

原始证据：[prefill_refactor_20261024](evidence/prefill_refactor_20261024/validation.log)。
最终 `validation.exit=0`，固定时钟恢复无错误。

| 检查 | 结果 |
|---|---|
| 普通测试 | 123 个通过，硬件测试单独执行 |
| Release all-target Clippy / fmt | 通过 |
| 调度器 | 层状态顺序、跨层并发、任一侧失败释放、取消排空通过 |
| native 执行 workspace | W4/W8，NPU mask 1/3/7，并发结果逐位一致 |
| Dense 单批/双批 | 9 组历史/输入长度，63 行完整词表 logits 逐位一致；包括 pp4096、尾批、后续 decode、truncate |
| Qwen3.5 单批/双批 | MTP 开/关各 7 组；logits、GDN state、卷积历史、特征、后续 decode 逐位一致 |
| prepared 输入 | 三轴位置和跨越物理批边界的注入特征一致 |
| 前缀缓存 | CPU/NPU GDN、MTP 开/关的 restore、分支、checkpoint 重用通过；prefill 内 workspace 创建数为 0 |
| 重构前后默认路径 | Qwen3 的 1736 份和 Qwen3.5 的 968 份逐层张量 hash 全部一致 |

前缀缓存 CPU 强制路径的测试最初只替换了第一条 lane 的 scratch，导致 CPU/NPU 混用。
已将测试夹具改为同时替换两条 lane，未放宽数值容差；CPU 和 NPU 两条测试均通过。
上面的 hash 是逐层数值回归证据，开启 trace 的运行不参与性能统计。

## 性能测量方法

仅在主板 测试：RK3588，四个大核 2.4 GHz、NPU 1 GHz / mask 7、
DDR 2112 MHz；CPU mask `0xf0`、4 threads，`OMP_WAIT_POLICY=PASSIVE`、
`GOMP_SPINCOUNT=0`。固定输入 token ID 为 1000 起递增，ubatch128、context4224，
空 KV，分别测 pp512 和 pp4096。

计时为完整 decoder 的 `LanguageModel::prefill` 调用，含 embedding、全部层、状态更新，
不含模型加载、tokenization、reset、prepare、最终 norm/LM head。MTP/DFlash 关闭。
同一实例预分配两组 workspace，再切换 lane 数量；两边使用相同的 domain 分布和权重。
因此这是调度的受控对照，并非分别按默认单批/双批内存预算新建模型的 CLI 对照。

每个模型、长度、算子库版本先分别预热单批/双批，并比较 prefill 末行与后续两行
完整词表 logits；随后执行两轮 `[1,2,2,1]`，每模式 4 个正式样本。
cache 搜索在正式测量前完成，正式窗口核对 cache hits/misses、tune calls 和 fallback calls。

CPU 占用 = 进程 CPU 时间 / wall time / 4；100% 表示四个大核全忙。
同时用 `/proc/stat` 核对整机大核占用，审计额外 CPU 任务。
NPU 占用来自内核 `rknpu/load` 的三个核心 busy 百分比，不代表 MAC 利用率。
50 ms 采样，丢弃每次计时开始后的 2.05 秒以避开一秒刷新延迟，再按持续时间加权。
pp512 剩余采样窗口短，NPU 数字仅作辅助；pp4096 更有代表性。

10 月 24 日的受干扰记录保留在旧证据目录，不纳入最终结果。10 月 25 日第一轮构建
期间共享 native HEAD 被外部更新，尚未产生性能样本，已归档并改用固定 commit 的独立
源码目录。正式两版均记录 runtime/native 源码 hash，构建与调优日志、原始样本、CPU/NPU
监控和时钟恢复报告保留在 [prefill_schedule_perf_20261025](evidence/prefill_schedule_perf_20261025/)。

## 对照版本

- A：当前重构代码、单批、旧算子库 `4f36ab66ddd6a2d2a12dd034dc29754bdb02235a`。
- B：相同代码、双批、相同旧算子库。
- C：相同代码、双批、新 `main @ 811b2a37579499d746fd3bdbc2abdfd5de46ac9d`。
- D（归因补充）：相同代码、单批、新 `main`，区分算子库自身与模型调度的收益。

旧版测量结束后，固定新 main 到另一独立源码目录。新库仅更新 Attention，复用相同
GEMM 策略缓存，正式测量仍要求零 cache miss / tune / fallback / weight reload。
两次构建使用相同 runtime 源码，hash 已核对。

## 吞吐对照

单位 tok/s；百分比均为吞吐提升，并非延迟下降。A/B/C 为三组主对照。

| 模型 / prompt | A 旧库单批 | B 旧库双批 | C 新库双批 | B/A | C/A | C/B |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B / 512 | 186.52 | 217.99 | 221.94 | +16.87% | +18.99% | +1.81% |
| Qwen3-4B / 4096 | 129.49 | 157.21 | 163.42 | +21.41% | +26.21% | +3.95% |
| Qwen3.5-4B C7 / 512 | 167.00 | 186.06 | 185.81 | +11.41% | +11.26% | -0.14% |
| Qwen3.5-4B C7 / 4096 | 157.76 | 174.83 | 177.92 | +10.82% | +12.78% | +1.77% |

归因补充：D 仅更新算子库并保持单批。新库变化为 stable Attention 的 query-block
CPU/NPU 双 slot 流水，GEMM 策略缓存复用旧库的相同内容。

| 模型 / prompt | D 新库单批 tok/s | D/A 仅更新库 | C/D 新库上的双批收益 |
|---|---:|---:|---:|
| Qwen3-4B / 512 | 192.56 | +3.23% | +15.26% |
| Qwen3-4B / 4096 | 141.40 | +9.20% | +15.57% |
| Qwen3.5-4B C7 / 512 | 167.34 | +0.20% | +11.04% |
| Qwen3.5-4B C7 / 4096 | 159.98 | +1.41% | +11.22% |

每模式 4 次的延迟均值 ± 样本标准差，单位 ms。原始值保留在
[comparison.json](evidence/prefill_schedule_perf_20261025/comparison.json)。
ABBA 用于每个库版本内的单/双批；两个库版本按用户要求先旧后新，不是跨库 ABBA。
这里的标准差不是置信区间。

| 模型 / prompt | A | B | C | D |
|---|---:|---:|---:|---:|
| Qwen3-4B / 512 | 2744.95 ± 3.07 | 2348.74 ± 6.05 | 2306.91 ± 8.01 | 2658.98 ± 3.50 |
| Qwen3-4B / 4096 | 31631.48 ± 13.29 | 26054.07 ± 11.98 | 25063.55 ± 2.94 | 28966.62 ± 21.97 |
| Qwen3.5-4B C7 / 512 | 3065.81 ± 4.83 | 2751.80 ± 3.02 | 2755.54 ± 7.01 | 3059.63 ± 2.34 |
| Qwen3.5-4B C7 / 4096 | 25964.11 ± 9.66 | 23428.60 ± 31.42 | 23021.84 ± 22.82 | 25603.94 ± 13.91 |

## CPU / NPU 占用

下面各格均按 **A / B / C / D** 排列。CPU 为四大核归一化后的进程 CPU 时间占比，
NPU 为三核 busy 的平均值。pp512 的 NPU 采样窗口短，仅作辅助，不用于判断微小增量。

| 模型 / prompt | CPU %，A / B / C / D | NPU 三核平均 %，A / B / C / D |
|---|---|---|
| Qwen3-4B / 512 | 51.4 / 67.6 / 69.8 / 53.3 | 33.3 / 40.5 / 41.6 / 34.5 |
| Qwen3-4B / 4096 | 44.1 / 61.1 / 66.9 / 51.2 | 37.7 / 48.2 / 52.3 / 43.4 |
| Qwen3.5-4B C7 / 512 | 53.8 / 69.8 / 70.8 / 54.3 | 32.4 / 38.4 / 38.3 / 32.5 |
| Qwen3.5-4B C7 / 4096 | 51.9 / 67.2 / 69.9 / 53.5 | 33.4 / 39.2 / 40.2 / 34.1 |

pp4096 的 NPU 各核心 busy：

| 模型 | 版本 | Core0 / Core1 / Core2 % |
|---|---|---|
| Qwen3-4B | A | 36.0 / 41.7 / 35.5 |
| Qwen3-4B | B | 46.0 / 53.3 / 45.5 |
| Qwen3-4B | C | 49.9 / 57.7 / 49.2 |
| Qwen3-4B | D | 41.7 / 47.9 / 40.6 |
| Qwen3.5-4B C7 | A | 32.0 / 34.1 / 33.9 |
| Qwen3.5-4B C7 | B | 37.3 / 40.3 / 40.0 |
| Qwen3.5-4B C7 | C | 38.3 / 41.4 / 41.0 |
| Qwen3.5-4B C7 | D | 32.7 / 35.0 / 34.7 |

利用率提高还包含 CPU 成本增加，不能直接等同于有效计算同比例增加。
每次 pp4096 的进程 CPU core-seconds 如下；它与 wall time 的计时范围相同。

| 模型 | A | B | C | D |
|---|---:|---:|---:|---:|
| Qwen3-4B | 55.83 | 63.67 | 67.08 | 59.37 |
| Qwen3.5-4B C7 | 53.88 | 62.97 | 64.39 | 54.80 |

## 结果解释与后续边界

1. 模型双批有稳定收益：旧库上 Qwen3 为 16.9%–21.4%，Qwen3.5 为 10.8%–11.4%。
   新库上仍分别有 15.3%–15.6% 和约 11.0%–11.2% 的收益；共用调度没有让混合模型失效。
2. 两级流水的收益不能直接相加。Qwen3 pp4096 仅更新库提升 9.2%，在已有双批上额外
   提升 4.0%。这与两条路径覆盖了部分相同等待相符，但本轮没有用新的时间线拆分每个气泡。
3. Qwen3.5 pp512 的库更新增量为 −0.14%，小于该组样本波动，不视为稳定回退或收益。
   pp4096 有 1.77% 的额外吞吐收益；整体合计为 12.78%。
4. 单批 pp4096 中，旧库 Qwen3 的 Attention 为 16.220 / 31.631 秒，约占 51.3%；新库
   降至 13.780 秒，阶段降低 15.0%。Qwen3.5 则仅为 2.624 / 25.964 秒，约占 10.1%，
   新库降至 2.185 秒，阶段降低 16.7%。Qwen3.5 的 24 层 GDN 和主要 Linear 路径没有
   被本次 Attention 更新直接加速，因此整模型收益小得多。原始阶段均值见
   [phases.json](evidence/prefill_schedule_perf_20261025/phases.json)。
5. 最新双批 pp4096 下，Qwen3 CPU/NPU 为 66.9%/52.3%，Qwen3.5 为 69.9%/40.2%，
   仍有明显空闲；单一平均值不能区分依赖等待、算子 CPU 阶段、队列等待和资源竞争。
   后续应先对 Linear/GDN 的完整调用定位暴露等待，再决定算子内部流水如何调整。
   当前保留简单的整层交接，不继续增加模型层调度策略。

## 数据与验证状态

- [旧库 summary.json](evidence/prefill_schedule_perf_20261025/old/summary.json)、
  [新库 summary.json](evidence/prefill_schedule_perf_20261025/new/summary.json)：计时、CPU、
  NPU、标准差及原始样本；各模型 `analyzed.json` 保留逐样本审计结果。
- 两库的全部正式窗口均通过：实际采样频率固定、外部 CPU 差值小于 5 个百分点、
  无编译进程、零 cache miss / tune / fallback / weight reload，以及实际双批组数核对。
  Qwen3 cache hits=120，Qwen3.5=100。
- [weight-layout-audit.json](evidence/prefill_schedule_perf_20261025/weight-layout-audit.json)：
  两版的 domain、resident 权重布局、mapped/payload 字节数一致。
- 每版的 `source-manifest.json` / `machine.json` / `clocks.json` / `runner.log`，以及各模型
  的 `samples.jsonl`、`load.jsonl` 保留可复查的源码与机器来源。
- 新库普通回归 123 个通过；原生 Attention pipeline 板端契约测试通过。
- 旧库退出码 0、时钟恢复无错误。Qwen3 监控子进程首次启动缺少 `sys` 导入，在模型
  加载期间修复；第一个正式计时前已开始采样，全部计时窗口覆盖率通过审计。
- 新库与重构前默认串行路径逐层 hash 全部一致：Qwen3 1736 份、Qwen3.5 968 份。
  这也核对了本次 Attention 更新后的完整默认路径。新库退出码 0，时钟恢复无错误。
  新旧库 `clocks.json` 均为 `exit_code=0`、`restore_errors=[]`。

实现和测量已完成，代码未提交；原有无关工作区改动保留。
