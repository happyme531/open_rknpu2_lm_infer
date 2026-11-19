# MiniCPM V 4.7 35B A3B decode 流程与优化方向

2026-10-26。分析对象为 H88K 8 GiB、本地 GM7 SSD、noSQ 模型、40 槽/层、DSpark draft3 的流式 decode。核心判断：继续调整分配、线程池和队列，很难再复现此前约 50% 的收益。更值得投入的是**按推理阶段管理内存并增加有效专家缓存、减少低复用冷专家的读入与缓存污染、把小 M 的专家完整调用和投机验证作为整体优化**。

当前没有证据支持某一个小改动能把约 8 tok/s 翻倍。若目标是 12 tok/s，需要把约 125 ms/output 降到 83 ms，省约 42 ms；16 tok/s 需要省约 62.5 ms。这个量级必须同时改变多个主要成本，或提高每轮产生的有效 token 数。

## 当前 decode 流程

目标模型共 40 层，hidden 2048；30 层 GDN、10 层完整 Attention。每层 256 个 routed experts、top-8、expert intermediate 512，另有共享专家。路由专家 gate/up/down 为 W4A8，shared FFN 和输出头为 W8A8；router 使用 BF16 权重、CPU FP32 点积。模型结构来自[本地配置](../tests/fixtures/minicpmv47/config.json)，量化及运行条件以对应 benchmark 的实际模型为准。

1. **DSpark 提议。** 收取已经提交的目标层特征，5 层 FP16 草稿网络并行生成候选 hidden；复用目标模型的输出头得到 logits，然后逐候选执行 Markov 修正与采样。draft3 通常得到 3 个 proposal，加上 anchor，目标模型验证 4 行。普通 AR 直接执行 1 行。
2. **每层 mixer。** norm → GDN/Attention 输入投影 → CPU GDN 状态计算或 NPU Attention → 输出投影 → 残差与 FFN norm。GDN 验证保存可提交的状态，不能随意把不同位置或层重排。
3. **实际路由先发需求。** CPU 一次批量计算本层 router，逐行选 top-8；分页模块对专家去重，查本层热缓存，先将真实 miss 入队。
4. **预测下一层。** 在当前 FFN 输入上执行下一层 router，作为 LES 预测；对多行预测取并集，过滤下一层已缓存专家，提前读盘。当前小批量路径覆盖 M≤16。
5. **后台搬运。** 两个 O_DIRECT reader，需求优先、最多 32 条 ready/in-flight 记录，读到可复用的页对齐 RAM buffer。只有真实需求确认后，才由一个后台 import worker 拷贝到 NPU staging 并 sync；错误预测不提前做 NPU 导入。每份磁盘记录 1,597,440 bytes。
6. **本层专家执行。** 先执行当前 hot 专家；随后 `take_any` 消费冷记录，已后台导入的记录交换到 hot 槽，未开始导入的记录可由主线程接管。冻结基线会收齐一组冷专家才执行；DS3 专家并集不超过 32，因此一般是整层 cold 收齐后的一次执行。输出保留原 token/rank，最终按原顺序做 FP32 FMA 归约。
7. **共享专家与层结束。** 当前源码在 routed 路径之后执行共享 gate/up、SwiGLU、down、sigmoid gate 合并，再残差进入下一层。
8. **验证与提交。** 目标输出头给每行 logits；接受连续 proposal 前缀，提交 GDN 状态、截断未接受的 Attention KV、更新 DSpark 上下文，然后进入下一轮。

主要入口：[speculative driver](../src/speculative/driver.rs)、[DSpark features](../src/model/qwen35/features.rs)、[DSpark runtime](../src/model/dflash/runtime.rs)、[Markov](../src/model/dflash/markov.rs)、[transaction](../src/model/qwen35/transaction.rs)、[mixer](../src/model/qwen35/decoder/mixers.rs)、[MoE](../src/model/qwen35/moe.rs)、[paged](../src/model/experts/paged.rs)、[reader](../src/model/experts/io.rs)。

底层已经按 expert 聚合行、按三个 core 排任务，并批量构建命令；不能把当前实现描述成“每个专家各做一次普通 matmul”。但 Rust 分页层把 token/top-k 展成 assignment 行，hot/cold 分开调用；底层每次调用仍经过输入量化、gather、gate/up、CPU 中间激活/量化、down、输出同步与归约。见[原生 MoE 实现](../native/rk_npu_matmul/src/rk_npu_moe_w8.cpp)。

## 指定对话已经证明了什么

已核对对话 `01a17188-3766-7b93-bbb1-4de58e42b704` 的 10 月 25 日记录及 10 月 26 日续接记录。关键结果应按各自配对基线理解，不能拼成同一次严格消融。

| 实验 | decode 结果 | 对下一步的含义 |
|---|---:|---|
| 读 buffer pool、双 reader、减少重复导入 | 4.97 → 5.37 tok/s | 基础流式加载优化已有约 8% 收益 |
| 确认需求后异步导入 | 5.41 → 5.54 | 主线程 import 大幅下降，但 E2E 只有约 2.5%，存在等待与资源竞争迁移 |
| DSpark 与 M≤16 LES/持久缓存 | AR 5.37，DS3 8.08 | 这轮约 50% 是已经证明的较大收益 |
| 算子库从 0fd946e 更新至 811b2a3，均含本地补丁 | DS3 8.08 → 7.80 | 接受率、文本和路由轨迹变了，不能当成纯算子性能回退 |
| router 改 NPU TF32 | CPU 7.93，NPU 7.72 | 小 M 完整调用、重叠和精度同时影响结果；未保留为默认 |
| CPU router/LES 批处理 | 7.84 → 7.88 | 单算子快，E2E 未测出稳定提升 |
| 共享分页工作区 | 7.69 → 7.65 | prefill 改善，decode 没有稳定收益 |

对应报告：[流式 I/O](MINICPMV47_STREAM_IO_OPTIMIZATION_20261025.md)、[异步导入](MINICPMV47_ASYNC_IMPORT_20261025.md)、[DSpark](MINICPMV47_DSPARK_H88K_20261026.md)、[native 更新](MINICPMV47_NATIVE_UPDATE_20261026.md)、[TF32](MINICPMV47_ROUTER_TF32_20261026.md)、[CPU router](MINICPMV47_CPU_ROUTER_BATCH_20261026.md)、[分页工作区](MINICPMV47_PAGING_WORKSPACE_20261026.md)。

最新已完成分页实验的正常样本为 7.80、7.50 tok/s。以下耗时采用同实验**默认 allocator 的独立诊断 trace**，避免沿用固定 mmap 阈值放大分配成本的结论。诊断本身约 7.41 tok/s，不能代替正常测速。

## 当前时间预算

条件为 pp512 后 tg128、ubatch128、CPU4–7 2.4 GHz、NPU mask7/1 GHz、DDR2112 MHz、40 槽、LES、2 read workers、1 import worker、DS3。所有数值均为每实际输出 token 的时间，包含被拒绝 proposal 的工作。

| 互斥部分 | ms/output |
|---|---:|
| 路由专家 complete-call | 29.38 |
| record 等待 | 23.18 |
| 主线程导入与槽交换 | 7.17 |
| mixer 输入与输出投影 | 14.97 |
| router、LES、请求记账 | 8.25 |
| CPU GDN | 7.20 |
| 共享专家 | 6.55 |
| 目标输出头 | 5.20 |
| 分页剩余编排 | 5.55 |
| 其他验证，包括 Attention 和验证外包装差额 | 4.11 |
| 草稿生成 | 19.37 |
| commit | 2.87 |
| 外层采样输出等差额 | 1.19 |
| **总计** | **135.00** |

后台 import service 另有约 27.42 ms/output，但与上表重叠，不能再相加。record 等待中与读取服务重叠约 14.88 ms、与后台导入重叠约 5.98 ms；单纯导入排队仅约 0.25 ms。依据为[原始阶段分析](evidence/minicpmv47_paging_cpu_20261026/optimized_default_ds3_detail_trace.analysis.json)与[本轮复算](evidence/minicpmv47_decode_strategy_20261026/summary.json)。

以该 trace 的其他阶段不变计算，清零当前全部 record 等待，135.00 → 111.82 ms，即 7.41 → 8.94 tok/s，约 +20.7%；连主线程 import 也清零，为约 9.56 tok/s。这只是当前可见等待的条件上限，既不是实测速度，也不是改变缓存、计算或 DDR 竞争后的共同上限。

这解释了为什么继续精修队列很难翻倍：一项 0.5–1 ms 的整理优化，只占总预算不到 1%；即使把约 5.55 ms 的分页剩余工作全部清零，吞吐也只提高约 4.3%。

## 新的离线证据

本轮复用默认 allocator trace，重建 prefill 尾部缓存、每轮需求和 LES 保护集合。当前预测保护 LRU 在全部 **1400 次层调用**的 cold 数上逐次复现，零差异，总计 12926 条。再固定同一份路由和初始缓存策略，只替换缓存算法或容量。

### 冷权重的跨 token 复用仍然很低

DS3 实际验证 139 行，产出 128 token。全部 routed assignment 为 44480，去重后有 29785 个层调用内专家；冷专家 12926 个，对应 15368 个 assignment。

- 每个冷专家平均只服务 **1.189 行**。
- **84.36%** 的冷专家在该次验证只服务一行，只有 56 个冷专家同时服务四行。
- 现有必要冷读为 **161.32 MB/output**，实际读取 **201.61 MB/output**；额外约 40.30 MB 主要来自预测读入后未使用。
- 只属于未接受后缀的冷记录为 1015 个，即 **12.67 MB/output**。这是事后归因；即便事先知道这些后缀无用，尚需重新模拟缓存轨迹，不能直接把它当成一定能省下的总流量。

因此，DS3 的主要收益不能解释成专家权重读取被除以四。它更多地摊薄了投影、输出头和固定调用成本。冷专家恰好是最缺乏复用的部分；单纯扩大验证 M，容易增加低复用读入和缓存污染。

### 同容量下的缓存策略空间

| 每层 40 槽，固定路由回放 | 必要冷读 MB/output | 相对当前冷读 |
|---|---:|---:|
| 当前 LES 保护 LRU，精确复现 | 161.32 | — |
| 普通 LRU | 162.15 | +0.5% |
| 历史频率每轮乘 0.8，再加入当前 assignment 频率 | 148.15 | -8.2% |
| 知道未来的替换，仍要求整批需求留在 cache | 111.72 | -30.7% |
| 知道未来且允许一次性专家不进入持久 cache | 95.15 | -41.0% |

频率策略以 prompt 计数初始化，只使用已见 prefix 和当前已知需求；0.5/0.95/1.0 的结果分别为 155.38/148.34/148.82 MB/output。这里只是一条 trace 上的探索，不能认为 0.8 已经泛化。两个未来 oracle 不可在线直接实现，只说明避免错误保留/驱逐的潜在范围；允许不入缓存还需要独立临时槽及相应执行接口。

此前“总 1600 槽不变，改各层 32/40/48/56/64 分配”只减少约 0.82% cold，是**容量分配**的实验；不等于“缓存策略只剩 0.82% 空间”。本轮变化的是槽内留谁、是否必须永久接纳每个一次性需求。

### 如果能增加可用缓存

为清晰隔离容量，这张表统一用普通 LRU、各容量对应的 prompt-tail 初始集合，不能与线上预测保护 LRU 混称配对实测。

| 每层槽数 | 必要冷读 MB/output | 相对 40 槽增加的原生权重 payload |
|---|---:|---:|
| 40 | 162.15 | 0 |
| 48 | 143.45 | 480 MiB |
| 56 | 125.94 | 960 MiB |
| 64 | 108.96 | 1440 MiB |

一个原生专家设备权重 payload 为 1.5 MiB，40 层每层增加一槽约需 60 MiB，另有元数据和可能增加的 domain/workspace 成本。以上是读量回放，不是内存可行性或 tok/s 预测。

### 更大投机窗口的理想读量也并不按 M 缩小

把已经提交的 128 行路由重组为批次，假设提前知道正确 token、全部接受、零草稿成本，按 hit-first、每个冷专家每批只载入一次、批尾保留最近专家的服务模型回放：

| 理想验证行数 | 必要冷读 MB/output |
|---|---:|
| 1 | 164.79 |
| 4 | 154.51 |
| 8 | 142.76 |
| 16 | 125.44 |

这不是实际大 M 的数值执行；超过 40 槽的并集需要分组或额外暂存。真实新版 DS7 验证 182 行才输出 128 token，DS15 trace 为 267 行输出 128。DS3 只有 139 行。更大窗口的权重复用会被拒绝成本抵消，因此现在不能靠直接改成 draft15 期待大提速。

## 优先方向一 按阶段管理 workspace 并改善缓存接纳

这是本轮最值得先做成一个系统工作包的方向：**减少 decode 不再需要的常驻内存，把释放的实际物理内存转成有效专家缓存，同时避免一次性冷需求污染热集。** 它同时影响必要读盘、导入、DDR 流量以及内存压力，覆盖面大于减少一次 Vec 分配。

源码与原始统计暴露了一个具体入口：

- [runtime prepare](../src/model/qwen35/runtime.rs) 把 prefill 的 M128 和投机 M1–4 一起传给所有 backend，**也包括输出头**。
- [Backend prepare/get_work](../src/npu.rs) 对每个 `(geometry, bucket(M))` 保留独立 native workspace 和 FP32 input/output 数组。M3 bucket 到 M4，因此本例留下 M1/M2/M4/M128 四档；`retain_workspaces` 只在 prepare 阶段调用。
- [生成用 output_head](../src/model/qwen35/head.rs) 在 prefill 最后只计算一行；DS3 的 target head 为 M≤4、draft head 为 M≤3。这个生成请求并没有 M128 输出头需求，PPL 的 `score()` 才会需要整块 logits。
- 正常样本 `workspace_payload_bytes = 1,012,575,900`，约 **965.67 MiB**；其中 head domain 合计 **257.46 MiB**。仅 M128 的 FP32 head output 数组容量就约 121 MiB，另有 native 输出等缓冲。

这些数字是 workspace 分配计数，包含 host 数组和 device buffer；不是全都已驻留，也不是全都可释放。不能承诺直接省 1 GiB。FP16 drafter 的 `FloatBackend` 持有独立权重/工作区，当前 `Backend::stats` 也没有完整汇入它们；本例 dflash domains 的量化 weight/workspace 计数为零，不能据此认为草稿不占 NPU 内存。

建议先分清生成、评分、prefill、verify 的实际 shape 生命周期，取消生成阶段无用的 head M128 预热，并在进入 decode 时回收 prefill 专用工作区。然后评估按 domain 共用 scratch arena，保留权重、strategy、KV 和 GDN 状态。以 smaps/BO 实际回收量为准，再决定能否从 40 槽升到 48 槽。多轮对话下一次 prefill 可以重新申请，不能破坏评分路径。

缓存侧先用历史频率/近期性作为强基线，进一步区分“临时执行一次”和“长期保留”。当前 LES 预测本轮当前层的需求，不是在预测该专家下一轮还会不会被使用；它对读提前量有价值，对长期保留价值的判断有限。更长远的方向是根据复用概率、草稿置信度、读入成本决定 admission/eviction，但不能改变真实 top-8 或忽略专家贡献。

第一道 gate：固定路由回放覆盖多条不同任务，必要冷读稳定降低至少约 10%，或实测回收足够支持额外缓存的物理内存。之后才实现新的缓存接纳接口。最终以同 8 GiB 预算、无新增 decode 换页、正常 allocator 的配对 E2E 衡量；不能把读量降幅当成吞吐降幅。

## 优先方向二 将流式 MoE 调度与小 M 完整调用一起改

短期可以打破“等所有冷专家到齐再执行”的屏障：热专家、已 ready 冷专家、共享专家都是实际路由后可安排的工作。共享 FFN 只依赖当前归一化输入，可使用独立输出缓冲，在冷记录未齐时执行，最终保持原始合并顺序。下一层依赖本层完整输出，不能无条件跨层提前算。

必须同时约束额外 submit：当前 DS3 trace 有 **2790 次专家 complete-call，约 21.80 次/output**；每次内部已 grouped。简单按每个 ready 专家立刻提交，会把目前每层通常两批变成很多小批，可能牺牲三核占用并增加量化、gather、build、sync、CPU middle 和输出处理。应按 ready 数量、可隐藏等待、预计计算时长选择批次，并允许等待时插入共享专家工作。

当前工作树的另一个既有实验目录 [ready groups](evidence/minicpmv47_ready_groups_20261026/experiment.json) 已在做“至少取三条，再 drain ready 集合”的原型。该实验在本轮阅读时尚无完整正式汇总；其收益不计入本报告，也不需要重复开启同一小实验。

若目标是超过 30%，这个方向需延伸到底层完整执行路径：

1. 在真实专家分组上接出已有 `rk_npu_moe_w8_timings`，拆 gate、down、CPU middle、gather、build、sync、reduce，和正常 E2E 对齐。
2. 复用原始 token 的量化输入和路由索引，减少 assignment 展开后重复处理；评估可复用命令模板和任务 patch，保持槽位所有权正确。
3. 评估将兼容的 hot/cold/shared 工作并入一次调度，或让 gate/middle/down 按可用任务流水执行，减少每批共同等待的边界。CPU SwiGLU/dynamic-A8 的精度和全 K scale 不能因流水而改变。

预算是专家约 29 ms、投影约 15 ms、共享专家约 6.5 ms，三项合计约 51 ms/output。但没有 gate/down 等内部证据，不能宣称这些都是硬件 GEMM 或某一种带宽损失。即使这三项整体快一倍，在其他时间冻结的当前 trace 上也只是约 23% E2E 提升；更大提升还需要流量、重叠或有效 token 数一起改善。

## 优先方向三 提高有效验证产出并降低草稿成本

现有 DS3 只有 11/139 行属于未提交后缀，且 35 轮输出 128 token，平均 3.657。固定每轮成本不变、每轮最多输出四个时，即使接受率达到 100%，轮数收益也只约 `4 / 3.657 = 1.094`。所以单独把 DS3 接受率从 89% 修到 100%，不会翻倍。

更有价值的实验是：已知正确未来 token 的真实 teacher-forced verify4/8/16，包含 GDN 提交、专家缓存/读取及所有 target 成本；先确定目标模型本身的有效验证吞吐曲线。仅有离线路由重组不足以证明更大窗口的端到端上限。

然后拆草稿约 19 ms/output 的三部分：5 层 FP16 网络、共享的巨大词表 head、逐 proposal 的 Markov 投影。Markov rank256→vocab248144，每个 proposal 单独调用；shared target head 也在 draft 阶段使用，所以只看表中的 target-head 5.2 ms 会漏掉另一部分词表计算。

可研究更便宜且接受率足够高的草稿、按收益动态选择窗口，以及量化草稿后让出的内存用于专家缓存。目标模型与验证规则保持不变时，草稿质量的主要成本体现为接受率；仍需检查目标实现的准确接受逻辑。量化草稿不能仅看 draft kernel 加速，还要看净有效输出、额外 cold bytes、PPL/采样正确性和内存。

更激进的“草稿预测目标专家，使读盘提前多层/提前一轮”需要训练或校准预测器。可以从既有目标特征和路由轨迹评估，但必须把未命中/错误读入、预测计算、内存和延迟一并计入。当前 LES 冷需求覆盖约 66%、提前 RAM ready 约 43%，说明提前量有改进空间；清零当前等待的约 20.7% 条件上限也说明预取本身不是翻倍方案。

## 不再作为主线的方向

- **再加 IO/import worker 或扩大队列。** 两个 reader 的独立回放已接近当前链路上限；实际 ready/import 排队很短，未证明队列容量持续卡住 DS3。多读错预测还会增加 DDR 竞争。
- **单独优化 memcpy。** 直接读 NPU buffer 可以探索，但 O_DIRECT pin/对齐、record 格式、meta 与 gate/down 分散布局、sync 及所有权都要解决。应计算暴露部分和 DDR 竞争，不能把后台 27 ms 全当成可直接追回的时间。
- **继续换 router 硬件。** 当前 M4 的 CPU/NPU complete-call 对照和 TF32 精度结果不支持默认替换。
- **优先优化完整 Attention。** 当前约 512–640 token KV 长度下，目标 Attention 核心仅约 1.3 ms/output。长上下文需重新看排名，本结论不外推到数万 token。
- **把固定 mmap 阈值 trace 中的分配热点当主瓶颈。** 默认 allocator 下分页缓冲复用只减少约 0.76 ms/output，正常 decode 未提升。
- **只调整各层槽数，或盲目把 draft3 增到 draft15。** 两者已有反证或很弱的收益依据；应分别转向接纳/替换价值与有效验证产出。

## 建议执行顺序

1. 固定一份数值参考，沿用当前 CPU router 和同一 native 补丁；把 811b2a3 跨版本数值问题与调度实验分开。所有性能比较用同路由/同接受前缀，不能把接受率变化归为 kernel 快慢。
2. 审计并回收阶段无用 workspace，明确实际可释放 BO/RSS；同期完成多 prompt 的 cache admission/replacement 离线回放。这个工作包的产出是更少的必要 cold bytes 和更大的有效热集。
3. 收尾已有 ready-groups 原型，再根据额外 submit 成本决定是否纳入 shared FFN 调度；同时细拆真实小 M 专家 complete-call，形成统一 MoE executor 的设计，而非继续增加独立小开关。
4. 做正确未来 token 的 verify 上限实验，再决定投入草稿压缩、窗口自适应或新的更强草稿。不要先以接受率或 batch 大小为优化目标。

还有一个基线边界：正常样本明确 `--no-tune`，`tune_cache_hits=0`、`tune_cache_misses=0`、`tune_calls=0`，`strategy_fallback_calls=7481`。这是缺少 tune 时使用预设策略，不是模型/CPU 意外 fallback。当前没有验证计算策略已经最优；在声明硬件瓶颈或做大 backend 改造前，应补真实 M1/M4/M8 的 tune 对照，但不预设它能带来大收益。

所有 gate 都应以正常 allocator、完整 decode、固定频率、阶段内内存统计、固定模型与 token 输入为准；局部耗时、流量和 oracle 仅用于决定是否继续投入。约 8 tok/s 目前是这一模型、输入、内存预算和软件路径上的测量点，不是 RK3588 或流式 MoE 的硬件上限。

## 复现材料

本轮未改运行时代码、未构建或上板测速。新增材料仅是分析文档、离线回放脚本及结果。

- [回放脚本](evidence/minicpmv47_decode_strategy_20261026/replay.py)，从仓库根目录运行；只读取已有 trace，输出同目录 summary。
- [回放与阶段结果](evidence/minicpmv47_decode_strategy_20261026/summary.json)，含源 trace SHA256、1400 次 cold 对账、容量/策略/理想重组结果及限制。
- [默认 allocator 原始 trace](evidence/minicpmv47_paging_cpu_20261026/results/optimized_default_ds3_detail_trace.timeline.json)。
- [正常运行原始统计](evidence/minicpmv47_paging_cpu_20261026/results/optimized_ds3_B.json)，包括 shape/workspace/tune 计数。
- [指定对话原始记录]([本地会话记录]
