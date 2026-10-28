# v0.7.0

- 新增按模型模板处理的工具调用解析，支持 MiniCPM5 XML 工具调用及流式 OpenAI 兼容 API 输出。
- W4 gate/up 融合扩展到小批次与 decode，按调用策略复用或单独缓存 SwiGLU 工作区；普通 GEMM 的调优策略保持独立。
- 稠密 FFN 的融合与独立激活统一使用算子库的 FP32 SwiGLU，实现不承诺与旧 FP64 路径逐位一致。
- 优化有限 top-k 采样，先选择候选再排序；贪心采样改为单次扫描。复用生成缓冲区，移除重复的输出头准备和 release 热路径数值诊断。
- CLI 与 HTTP API 的默认温度统一为 **1**，默认输出上限统一为 **2048 tokens**。需要贪心生成时显式指定 `--temp 0` 或 API 的 `temperature: 0`。
- 更新 NPU 算子库至 `4f36ab6`，包含 W4A8 和 Attention 执行优化；普通推理继续读取已有调优缓存，需要时使用 `tune` 命令显式调优。
- 已完成 RK3588 上的 105 项自动测试、W4/W8 padding 与融合 SwiGLU 板端检查，以及 Qwen3-4B、Qwen3.5-4B MTP1 和 448×448 视觉编码的 README 口径复测。GitHub Release 另执行 ARM64 构建、Clippy、自动测试、启动和打包检查。

ARM64 附件适用于 Debian 11 / glibc 2.31 或更新系统，依赖 libstdc++6、libgomp1 和 RK3588 NPU DRM 驱动。模型独立下载。

# v0.6.1

- 修复 Qwen3.5 图片超过1024个patch时出现 `invalid full FP16 attention shape` 的问题；视觉 Attention 按预处理后的实际图像长度分配工作区。
- 使用 `FullAttention` 临时 K/V buffer，在图像的 Transformer blocks 间覆写复用，blocks 完成后释放，独立于文本上下文容量。
- 同步算子库，移除 Attention 对32768行容量的固定拒绝条件，保留调用方分配边界及实际算子形状、任务数量校验。
- 板端验证包含原图 FP16/FP32 编码、CLI 问答、固定token文本logits及小图features逐位回归；89项默认测试、2项视觉NPU测试和native Attention CPU contract通过。

# v0.6.0

- 新增 Qwen3.5 混合文本主干、静态图片与多图片输入，CLI、终端聊天和 OpenAI 兼容 API 共用图文历史与前缀缓存。
- 新增 Qwen3.5 MTP 推测解码，以及独立 DFlash 草稿模型的并行起草；共用验证事务、采样、停止条件和流式输出。MTP 与 DFlash 均需显式启用。
- 新增 NPU GDN prefill、带步长投影输入、视觉 FP16 GEMM/bias/GELU 融合与浮点算子调优；普通推理只读取已有策略缓存。
- 修复 causal Attention 的 masked-future、CPU softmax 分段与回滚数值边界，使已验证的批量 target 校验及提交后的 logits 与逐 token AR 一致。
- 同步中英文 README 的 Qwen3.5 性能、PPL、图像输入和调优示例。实验数据与 trace 不再纳入源码及发布附件，本地文件保留；测试必需的输入继续随源码提供。
- 板端已有 release 构建、全 target Clippy、数值与 CLI/API/TUI 回归记录；最新 DFlash 检查包含 68 项库测试、6 项输入测试和 19 项真实 CLI/API 功能检查。

DFlash 当前验证组合为未旋转的 Qwen3.5-4B C4 与对应草稿，旋转 C7 尚不支持 DFlash；上下文不超过草稿的 4096-token 窗口。两种推测解码当前最多 7 个提案。ARM64 附件适用于 Debian 11 / glibc 2.31 或更新系统，模型单独下载。

# v0.3.1

- 默认ubatch改为128；默认 `tune` 准备decode与当前主块，即 `1,128`，显式修改ubatch时自动跟随。
- 应用支持ubatch1–256，新增真实192行Linear档位；Attention独立按最多128行调用并继续内部拆块，无须修改算子库。
- 扩展Linear CPU oracle、融合SwiGLU及Attention分段板测。Qwen3的128/192/256完整模型小测未见更大档位的明确吞吐收益，固定输入及后续8步的全部logits逐位一致。
- 中英文README与使用说明同步，历史chunk64性能复现命令显式指定64。原始样本和验证边界见[小测报告](UBATCH_SWEEP_20261015.md)。

# v0.3.0

- 普通推理与评测只查询已有策略，缺失时使用合法默认配置；首次新长度不再触发在线性能搜索或调优缓存写入。
- 新增显式 `tune` 命令，支持选择batches和输出JSON，显示形状、缓存命中、候选进度、最佳耗时与累计时间。
- W4认证profile和缓存身份在权重创建时保留；读取策略不再reload代表权重或重复扫描原码，workspace淘汰独立于小量策略驻留。原有调优缓存格式保持兼容。
- 新增在线search/reload/实际fallback计数。三款模型缓存与回退路径完整logits检查一致；短prefill和已完成稳态数据见SHORT_PREFILL_TUNING_RESULTS。
- README中英文均补充完整性能准备与验证步骤。二进制附件提供固定频率脚本，源码/文档附件只收录版本控制内的文件。
- 本版本应用ubatch范围仍为1–128、默认64；更大ubatch后续独立验证。thinking与聊天模板语义保持原状。

# v0.2.0

- 各命令统一使用 llama.cpp 风格的参数：`-c/--ctx-size`、`-ub/--ubatch-size`、`-t/--threads`、`-n/--n-predict`、`-p/--prompt`、`-f/--file`、`--temp`、`-s/--seed`、`-cnv/--conversation` 和 `-sys/--system-prompt`。
- bench 使用 llama-bench 风格的 `-p/--n-prompt`、`-n/--n-gen`、`-d/--n-depth` 和 `-r/--repetitions`，支持逗号分隔的测试长度。
- `bench -p 0 -n 0 -pg 512,128` 在同一 KV 上连续执行 prefill512 和 decode128，分别报告两阶段及合计。
- `-c 0` 在 bench 中按测试需求自动设置容量，其他命令按模型支持的上下文上限设置容量，最多32768。
- 文档和板端比较脚本同步使用新参数名。原 `--context`、`--chunk`、`--max-tokens`、`--prompt-file`、`--chat-template` 及 bench 的 `--prompt`、`--generate`、`--depth` 已移除。

ARM64 二进制适用于 Debian 11 / glibc 2.31 或更新的系统，依赖 libstdc++6、libgomp1 和 RK3588 NPU DRM 驱动；模型独立提供。附件包含二进制和固定依赖的源码包。
