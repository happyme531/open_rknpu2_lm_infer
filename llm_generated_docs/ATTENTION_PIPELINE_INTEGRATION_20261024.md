# Attention 双 slot：默认集成与 Qwen3-4B 端到端验证

主板验证日期：2026-10-25。

算子库main已合入 `811b2a3`；prepare收尾提交为 `59aa26a`（仅准备首轮两个graph，后续buffer按需扩容，并删除无效MLA状态复制）。该收尾按要求未重新编译或测试，下述性能与数值验证仍对应 `8774a0c`。

实现基于 native 提交 `61762012b0a145e2286169526f1abf142695cb11`，位于 `perf/attention-nomask`。Stable Attention 的多布局分段默认启用双slot流水；设置 `RK_NPU_ATTN_PIPELINE=0` 可使用串行路径。环境策略在workspace创建时读取，使用前应先设置环境变量并重新创建workspace。

正式库签名提交：`8774a0c7a49dc06f102588e1819729f89b6d0b4b`。双slot默认启用；主板算子、两模型logits、ViT features及512/4K配对均通过。实验脚本、数据与二进制不在该提交中。

## 正式实现

- 新增 `src/rk_npu_attention_f16_pipeline.cpp`。一个持久工作线程执行原有阻塞submit；任意时刻最多一个NPU submit在运行。QK(next)覆盖CPU scores(current)，LUT/PV、finish及后续pack进入同一流水。先invalidate已完成的scores，再启动下一块QK。
- 两个slot各自持有Q、scores、LUT exponent、PV partial、寄存器/描述符、CPU scratch及partial tail。完整K/V与value-prefix checkpoint共享只读。串行与流水复用相同的score同步、softmax、PV/carrier清理和output同步实现。
- GQA保留原32行起始分块。默认NO_MASK MHA使用64行slot，关闭流水恢复128行起始策略；几何或4095任务预算可进一步缩块。只有请求确实包含多个布局分段时才创建/调度worker；单块verify/decode、fixed-shift和MLA保持串行提交。
- Tail staging从cache固定缓冲抽为可独立使用的 `Tail`。复用同时检查cache identity、epoch和geometry，防止共享workspace在相同epoch的不同层间复用错误V。
- 增量更新还要求 `dirty_append` 明确标记本次变更是append。Load/truncate完整重打包tail中的清零padding。回归覆盖truncate及同tail几何的缩短reload。
- 错误路径先排空在途阻塞submit再返回，随后workspace标记为poisoned。析构先停止并join工作线程，再释放graph、tail和domain引用。正式热路径没有新增NaN/Inf扫描或实验性验证逻辑。

公共C结构体与函数ABI保持兼容。`query_bytes/score_bytes/mask_bytes/partial_bytes/tail_bytes` 汇总活动slot，`physical_rows` 为每slot行数，key/value/output不乘二。Causal tail上界也覆盖较早CPU-exp分段的partial tail。LUT exponent另外需要至多一份合计 `score_bytes`；寄存器/描述符、CPU scratch、线程栈、allocator余量和历史保留峰值不在这些逻辑尺寸中。

流水的阶段计时会重叠，不能相加还原总耗时；`total_us` 是完整公共调用的墙钟时间，`hardware_us` 是driver区间之和，不是纯MAC时间。

## 正确性验证

四项CPU CTest在主板默认/禁用两种策略均通过：Attention、causal、pipeline policy/sizing、MLA；公共C头文件构建通过。板端覆盖默认/禁用策略、core mask1/2/4/3/7、causal pingpong/split、MHA/GQA/三种mask、MLA stable/expanded、后续INT8 canary。

新增pipeline板端契约测试覆盖同epoch跨层cache重绑定、三/五块和尾行、预加载未来高V、FP16/FP32 guards、reserve、增量append、truncate、同tail几何的缩短reload，以及带query offset/key stride的Boolean mask和全隐藏行。通过测试程序拦截syscall，在进入驱动前安全注入6种submit/sync失败，验证排空、poison、重建与BUSY锁；正式库不含该注入代码。

固定WikiText序列使用1537-token prompt和7个continuation，比较全部8个位置的logits：Qwen3共1,215,488个FP32值，C7共1,986,560个。默认、禁用以及已提交基线均逐字节一致。此为固定序列验证，本轮没有重新跑PPL。

## 端到端口径

模型为 `Qwen3-4B-W4A8-Mixed-LMHeadW8`。Rust runtime固定为上一轮验证快照 `b6ffd49`，仅有固定token输入的 `src/bench.rs` 适配；已与Git archive逐文件核对。两组运行同一个新构建程序，仅切换Attention环境变量。

主板（<BOARD_IP>），kernel `6.1.118-zt`、RKNPU driver `0.9.8`。CPU4–7/四线程2.4GHz，NPU三核1GHz，DDR2.112GHz，三个频率固定。ubatch128，同一WikiText前缀。ctx-size统一4160，为4096-token prompt后连续处理32个token预留容量；测量的两个历史规模仍是512/4096。

4K使用四个独立进程按 off/default/default/off 运行，每进程预热1次、正式2次，每方法4个样本。512采用整轮持有同一测试锁的独立ABBA，每进程预热1次、正式3次，每方法6个样本；每2秒记录进程活动，204份快照中未看到编译器/构建进程。原始完整ABBA的512样本另列于下表，未与专门的512测量混合。Prefill包含所有decoder层和KV追加，不含最终norm/LM head、模型加载、reset/prepare、tokenization及stats采样。Decode在同一KV上连续处理32个固定token，包含每步输出头，不包含采样。表中使用耗时的算术均值；吞吐由token数除以平均耗时计算。

正式测量使用独立的主板策略缓存。4160容量对应的两个Attention domain及所有Linear策略预先准备完成；每个正式进程要求92 hits、0 miss、0在线调优、0策略fallback。两组仅改变Attention流水。

## 结果与状态

### 单算子完整调用

每几何3轮ABBA，各进程12次正式调用，共1584次。表中先取每进程中位数，再对同一方法的6个进程取均值；时长为公共API完整调用，KV预加载不计时。Append行在计时内执行真实append128，计时外truncate到固定历史。各方法输出逐字节一致，另通过独立FP32抽样oracle。

| 用例 | 禁用 / ms | 默认 / ms | 耗时下降 |
|---|---:|---:|---:|
| qwen3_512 | 4.768 | 3.746 | 21.44% |
| qwen3_4096 | 22.725 | 17.393 | 23.46% |
| qwen3_decode4096 | 0.880 | 0.880 | 0.01% |
| c7_512 | 3.382 | 2.594 | 23.31% |
| c7_4096 | 14.701 | 11.427 | 22.27% |
| c7_verify | 1.283 | 1.283 | -0.01% |
| vit256 | 0.764 | 0.664 | 13.06% |
| vit1024 | 2.194 | 1.924 | 12.30% |
| c7_real | 0.477 | 0.477 | -0.02% |
| qwen3_append4096 | 26.881 | 21.447 | 20.22% |
| c7_append4096 | 21.415 | 17.044 | 20.41% |

Qwen3裸Attention 4K三个配对块分别下降22.72%/25.11%/22.49%。包含append的Qwen3三个块为20.98%/34.79%/7.15%，C7为28.27%/19.32%/13.09%；append行波动较大，表中保留全部配对块。Decode与四行verify均基本持平。

### Qwen3-4B 完整模型

| 上下文 | 正式样本/方法 | Prefill禁用 / s | Prefill默认 / s | 耗时下降 | 吞吐禁用→默认 / tok/s |
|---|---:|---:|---:|---:|---:|
| 512 | 6 | 2.758 | 2.660 | 3.56% | 185.62 → 192.48 |
| 4096 | 4 | 32.849 | 29.668 | 9.68% | 124.69 → 138.06 |

| 上下文 | Decode32禁用 / s | Decode32默认 / s | Prefill+decode禁用 / s | Prefill+decode默认 / s | 合计耗时下降 |
|---|---:|---:|---:|---:|---:|
| 512 | 3.026 | 3.028 | 5.785 | 5.688 | 1.67% |
| 4096 | 3.978 | 3.888 | 36.827 | 33.556 | 8.88% |

512的两个prefill配对分别下降2.77%/4.36%，4K为10.68%/8.64%。512 decode均值仅变化+0.06%；4K decode的两个配对符号不同，未形成一致收益。

| 上下文 | Attention禁用→默认 / s | Attention耗时下降 | Linear禁用→默认 / s |
|---|---:|---:|---:|
| 512 | 0.824 → 0.740 | 10.26% | 1.748 → 1.735 |
| 4096 | 16.876 → 14.267 | 15.46% | 14.417 → 13.902 |

4K的总prefill减少3.181秒，其中Attention阶段减少2.609秒，约占82%；Linear也有约0.515秒波动。因此端到端观测收益为9.68%，不能把全部差值都归因到Attention。512的Attention阶段下降10.26%，完整prefill下降3.56%，符合其在总耗时中的占比。真实prefill包含不同KV前缀，单个Q128/KV4096的局部比例不能代替整模型收益。

### 原始512完整ABBA样本

该轮第一个默认进程与主板其他项目的构建/测试时段重叠。所有方法和样本完整保留；阶段明细与时间记录见 `models/` 和 `run1-overlap.json`。下表不用于估计稳定的512收益。

| 顺序 | 策略 | Prefill均值 / ms | Attention / ms | Linear / ms | CPU / ms |
|---:|---|---:|---:|---:|---:|
| 0 | off | 2760.321 | 826.381 | 1749.051 | 184.330 |
| 1 | default | 5323.313 | 1186.291 | 3730.533 | 405.885 |
| 2 | default | 2644.860 | 710.918 | 1743.221 | 190.168 |
| 3 | off | 2872.472 | 894.715 | 1789.736 | 187.470 |

原始512四进程全样本均值为2816.397→3984.087ms。其第一个默认进程中Linear从约1.75秒升到3.73秒，CPU阶段从约0.18秒升到0.41秒；第二个默认进程恢复至2644.860ms。独立512 ABBA中Linear差异仅0.75%、CPU差异0.33%，并保留全部12个正式样本。

### ViT features与encoder完整调用

同一FP16存储/原有FFN实现、三张图分别ABBA，每进程预热1次、正式3次。计时覆盖完整encoder，不含模型加载与图像预处理；每次清空feature cache。所有默认/禁用features逐字节一致，与既有FP32参考的差异保持一致。

| 图像 | 禁用 / ms | 默认 / ms | 耗时下降 |
|---|---:|---:|---:|
| red | 380.107 | 379.054 | 0.28% |
| blue | 568.472 | 568.563 | -0.02% |
| red512 | 1659.189 | 1611.203 | 2.89% |

小图完整encoder基本持平，512×512图约下降2.89%。本轮视觉工具不输出策略miss/fallback计数，因此未对其声明零miss。

### 完成状态与内存边界

主板17项板端任务、双slot专门契约测试、1584次单算子计时、两模型完整logits、三图features均通过。29份正式任务clock报告均核对固定频率、结束频率及原min/max恢复；512补测的204份进程快照没有发现编译器/构建进程。最终10个源码/测试/文档文件与已测candidate逐文件一致。

大KV GQA的主要scratch增加一份slot，score/exponent/partial等接近两份；完整K/V与prefix checkpoint不复制。以Qwen3 4K几何为例，原型两个slot的GEM scratch合计约38.88MiB/Backend，可用于判断量级；本轮没有测完整driver内存峰值，实际高水位还取决于workspace复用、cache tail、寄存器与CPU缓冲。

结论适用于本报告固定模型、ubatch128与CPU/NPU/DDR频率。Prefill不含最终norm/LM head；同KV的decode32包含输出头。没有重新跑PPL，也未把单算子改善等同于decode提升。

完整统计与原始报告索引：[summary.json](evidence/attention_pipeline_integration_20261024/summary.json)。对应脚本、原始JSON/log、clock记录与源码patch均在同目录；大型f32与二进制保留在板端实验目录。

主板实验根目录：`<BOARD_HOME>/mnt/3090x4/tmp/attention-pipeline-mainboard-20261024`；主机映射位于 `<HOST_HOME>/mnt/3090x4/tmp/attention-pipeline-mainboard-20261024`。备用板数据独立保留在 `attention-pipeline-integration-20261024`，不参与主板性能统计。

实验工具、二进制和大体积原始输出不进入正式源码提交。
