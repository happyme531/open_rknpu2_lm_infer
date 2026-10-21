# MiniCPM-V 4.7 35B-A3B

支持文本常驻推理、专家流式加载与 LES 预取；DSpark 接口为实验功能，视觉尚未接入。

当前文本路径已支持标准 compressed-tensors 的无 SQ QQQ 权重，并移植 Strata 的逻辑 prefill chunk／专家分组读取和 llama.cpp-omni 的逐层 LRU 调度。[pp512/tg128 实测](MINICPMV47_NOSQ_BENCHMARK.md)、[万兆 NFS 对照](MINICPMV47_NFS10G_BENCHMARK.md)、[扩大专家缓存](MINICPMV47_CACHE_BUDGET.md)。

执行路径不做自定义 input-scale 除法；C4 加载器拒绝非单位 `input_scale`。标准 compressed-tensors 无 SQ 权重已完成板端数值与性能检查。

## 权重与执行

当前测试模型为 MiniCPM-V-4.7-35B-A3B-QQQ-W4A8-Mixed-CT。标准格式复用 `src/model/checkpoint.rs` 中的 compressed-tensors 解包、矩阵转置及 NPU 权重装载，独立 gate/up 专家在装载时拼接。

| 项目 | 测试权重 |
|---|---|
| 文本主干 | 40 层，30 GDN + 10 GQA，hidden 2048 |
| 路由专家 | 每层 256，top 8，intermediate 512 |
| 专家 gate/up、down | W4，无自定义 input scale |
| 共享专家 | W8，intermediate 512，独立 sigmoid gate |
| LM head | W8，248144 × 2048，非共享 embedding |
| GDN | CPU FP32 状态更新 |

普通量化投影按相邻模块的位宽和输入宽度 K 进行拼接，C4 与 compressed-tensors 共用此路径。包含受保护浮点层时保留独立 Linear 执行；C4 中非单位 `input_scale` 仍被拒绝。

同一 128-token 输入在 CPU GDN 下，ubatch 128 与逐 token 的整模型 logits 逐值相等。TF32 GDN 的舍入差异会经过激活量化和 MoE 路由放大；本模型默认使用 CPU GDN。

## 专家存储

`--expert-storage resident` 为默认行为。指定 `--expert-store DIR` 可保存并复用磁盘上的 NPU 原生打包记录；首次准备按层进行，无需先装入整个模型。

`--expert-storage stream --expert-store DIR` 使用每层独立的持久专家窗口。容量通过 `--expert-cache-percent P` 设置，默认 8，范围为 `0 < P <= 100`，支持小数。按每层路由专家总数乘以百分比向下取整，至少保留单 token 的 top-k 个专家；共享专家不计入分母。这是专家数比例，不是进程内存百分比或 RSS 上限。

对于当前 256 专家、top-8 模型：

| 参数 | 实际每层槽数 |
|---|---:|
| 默认 8% | 20 |
| 15.625% | 40 |
| 25% | 64 |
| 1% | 8，受 top-k 下限约束 |
| 100% | 256 |

```sh
open_rknpu2_lm_infer -m MODEL --expert-storage stream \
  --expert-store EXPERT_STORE --expert-cache-percent 25 \
  --ubatch-size 128 run --conversation --stats -p '你好' -n 128
```

启动日志和 `--stats` 输出均显示请求百分比、实际槽数及有效比例。stream 模式必须提供 `--expert-store DIR`。原目录参数 `--expert-cache` 已更名为 `--expert-store`，主程序和测试工具均使用新名。decode 的真实 miss 写入本层窗口，按 LRU 淘汰，并先保护本步所需及预测集中的专家。prefill 使用共享的 16 槽 bank 流式计算专家并集，持久窗口保留逻辑 chunk 结束时的最近访问集合。

文件中的一个专家记录包含 gate/up、down、输出通道 scales 和 INT4 correction。native v2 的 H2048/F512 W4 记录每条 1,597,440 bytes，10240 条记录的文件合计 16,357,789,696 bytes。v2 不复用 v1 缓存。缓存头记录模型几何、打包版本、源文件大小与修改时间；原始 checkpoint 改变后重新准备缓存。

checkpoint 文件发现及来源信息由 `src/model/checkpoint.rs` 负责：`TensorStore` 记录实际打开的单文件或分片及索引，`QuantLoader` 记录配置和实际使用的量化清单。专家缓存接收来源快照，在打开缓存和发布新记录前校验；校验仅查询文件元数据，不重复解析索引。`experts/store.rs` 负责原生记录格式及读写，不依赖 checkpoint 文件名或分片方式。

读取默认使用 `O_DIRECT`；`--expert-buffered-io` 允许系统页缓存。一个后台线程执行定位读取，需求请求优先于预测请求，最多保留 32 条已读取记录。计算线程负责槽位导入、同步和 NPU 提交。prefill 按实际路由去重；decode 使用当前层 FFN 输入通过下一层 router 预测，`--no-expert-prefetch` 关闭 LES。

各路由贡献按原始 top-k 顺序进行 FP32 FMA 归约，避免读入顺序改变累加结果。缓存命中、预测召回、预测及时率、读取 bytes、等待时间分别统计。IO 计数区分正确预测到来时的 `predicted_ready`、`predicted_in_flight`、`predicted_queued`，以及未使用预测的 `canceled_before_io`、`canceled_in_flight`、`canceled_ready`。`discarded_read_bytes` 同时计入读完后取消的记录与在途取消后丢弃的读取。

当前 prefill 的物理 microbatch 为 128，stream 路径的逻辑 chunk 默认最多 1024 token。在一个逻辑 chunk 内先逐层计算物理 microbatch，再按专家并集读取和计算；同一专家的权重服务该逻辑 chunk 的全部选中行。相邻记录最多 8 条合并为一次向量读取。跨逻辑 chunk 仍可能重读冷权重。NFS 的实测吞吐不代表本地 SSD 吞吐。

## 验证记录

- 标准无 SQ 权重的 resident/stream pp512/tg128 完整 logits 逐字节一致；129-token 非整 ubatch 检查通过。
- 缓存比例 8%、15.625%、25% 分别解析为每层 20、40、64 槽；25% 的 pp512/tg128 完整 logits 与既有参考一致。
- 64 槽 trace 完成 204800 次物理槽位检查，无违规；预取状态计数分解一致。
- 单文件与分片 checkpoint 的来源快照、缓存失效及索引不重复解析的回归检查通过；原有 v2 专家文件直接复用。
- C4 合成小矩阵单核板测中，1/5/128 行的融合与独立输出逐值一致，浮点保护层路径保留。
- 正式 `run --conversation --stats` 路径确认 25% 对应 64 槽，并完成简单算术生成。
- 板端 fmt、Clippy 和自动测试通过；固定频率测试正常恢复配置。逐次时延与协议见三个性能报告；完整原始 JSON、trace 和源码快照保留在本地实验目录，未随仓库发布。

## DSpark

草稿 checkpoint 为五层全注意力网络，从目标层 `[3,11,19,27,35]` 收集 post-block 状态，复用目标 embedding 与 LM head。DSpark 的第零输出行预测 anchor 后的第一个 token；逐步 Markov 修正使用前一个已提议 token，并在修正后的分布上采样和验证。置信度头供 `--spec-draft-p-min` 使用。

接口为 `--spec-type draft-dspark --spec-draft-model PATH --spec-draft-n-max 15`。完整草稿、head、目标验证与 commit 均计入推测解码流程。数值与端到端验证仍在进行。

Markov 与置信度头公式参考 [DeepSpec](https://github.com/deepseek-ai/DeepSpec/tree/main/deepspec/modeling/dspark)，保留其 MIT 许可于 `src/model/dflash/DeepSpec-LICENSE`。

## 尚未完成

长文本流式性能验证、CPU GDN 默认路径 PPL、DSpark 端到端验证，以及视觉接入。
