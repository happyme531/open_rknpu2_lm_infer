# MiniCPM-V 4.7 35B-A3B：RK3588 算子累加估算

完整融合MoE FFN的后续实测与更新估算见[融合MoE报告](MINICPMV47_35A3_FUSED_MOE_20261019.md)。本文保留独立GEMM相加的原估算作对照。

日期：2026-10-19。对象为语言主干，prefill ubatch=128，单序列自回归 decode，无投机采样。
精度：FFN down（路由专家、共享专家）与 LM head 为 W8A8；其余线性投影为 W4A8。
GDN prefill 使用现有 TF32/FP32 状态算子，decode 使用现有 CPU FP32 recurrence；全注意力使用现有 FP16 stable-softmax 算子。

## 结果

这是合成矩阵的板端实测时间与路由数据相结合的累加模型，没有加载 MiniCPM-V checkpoint 执行完整推理。
权重常驻内存，创建/打包权重与 workspace 的一次性成本不计入；动态量化、NPU 提交/同步、split-K 归并、反量化和紧凑输出包含在线性算子时间内。

| 范围 | 时间 | 折算吞吐 |
|---|---:|---:|
| prefill，仅线性算子，128 tokens | 1975.2 ms | 64.80 token/s |
| prefill，加 GDN/注意力，chunk 结束 KV=128 | 2237.5 ms/128 tokens | 57.21 token/s |
| decode，加 GDN/注意力，KV=128 | 108.69 ms/token | 9.20 token/s |
| prefill，加 GDN/注意力，chunk 结束 KV=512 | 2253.2 ms/128 tokens | 56.81 token/s |
| decode，加 GDN/注意力，KV=512 | 109.23 ms/token | 9.15 token/s |
| prefill，加 GDN/注意力，chunk 结束 KV=2048 | 2318.1 ms/128 tokens | 55.22 token/s |
| decode，加 GDN/注意力，KV=2048 | 111.10 ms/token | 9.00 token/s |
| prefill，加 GDN/注意力，chunk 结束 KV=4096 | 2405.2 ms/128 tokens | 53.22 token/s |
| decode，加 GDN/注意力，KV=4096 | 113.60 ms/token | 8.80 token/s |

上表 KV 是当前注意力调用可见的总长度。prefill128/512/2048/4096 是同一个128-token chunk 位于不同 KV 深度的成本，不是更换 ubatch，也不是整段 prompt 的累计时间。prefill 不对每个 prompt token 执行 LM head；首 token 还需一次末尾 norm/head，head 本身约16.63 ms。

## 路由依据与建模

主表采用same_arch_alpaca_proxy，作为有公开同架构统计约束的非均匀参考分布；单prompt prefill另列集中度敏感性范围。

另检查 [Shorthill 的公开 prefill captures](https://huggingface.co/datasets/ec75hash/routing-entropy-token-position)：来源说明为Qwen3.5-35B-A3B-Base BF16、Transformers 5.14、WikiText-103 train、无聊天模板。选取16个源文件（00000至00300，每隔20个），逐个检查发现专家ID均严格以128 token为周期重复。捕获语义尚未核实，因此不把这些文件当作可信长文本路由实测；每文件只保留第一个ubatch128，合计16个代理窗口，用作敏感性对照。

这个周期性代理每层平均激活 **140.45/256** 个专家，最热专家平均 **85.66** 个token，平均Gini **0.769**。每层1024次选择守恒，单个token选8个互不重复的专家，但这些结构检查不能证明捕获正确性。

它是同架构、不同checkpoint/任务/精度且捕获语义未核验的路由代理，不能声称是MiniCPM-V的真实路由，也不能证明混合量化后的路由不变。原数据依CC-BY-4.0使用；作者J. Shorthill，论文 Read Routing Entropy at a Fixed Position, v2.0, doi:10.5281/zenodo.22151499。所选原始npz与SOURCE_README保存在证据目录。

参考 [ReXpert 原论文，表II和表VI](https://arxiv.org/html/2608.13962v1)：Qwen3.5-35B-A3B 的128请求decode，Alpaca平均触及205个专家，最热/平均=6.77，Gini=0.491；HumanEval触及202.5个，最热/平均=9.89，Gini=0.522。据此构造非均匀参考分布，并与集中度情形比较；多请求decode的路由不能等同于单prompt prefill。proxy直方图拟合这些摘要统计；其中code top8/top16的分配是模型选择，未声称来自原始trace。U128/U160是敏感性情形，不是公开测量值。

| 分布 | 平均激活专家 | 平均最热专家token数 | Gini | 线性算子合计 | 加GDN/注意力，KV128 |
|---|---:|---:|---:|---:|---:|
| same_arch_alpaca_proxy | 205.00 | 27.00 | 0.491 | 1975.2 ms | 57.21 token/s |
| same_arch_code_proxy | 203.00 | 40.00 | 0.522 | 1957.1 ms | 57.67 token/s |
| concentrated_U160 | 160.00 | 40.00 | 0.640 | 1687.4 ms | 65.65 token/s |
| concentrated_U128 | 128.00 | 48.00 | 0.730 | 1476.9 ms | 73.60 token/s |
| uniform_four_each | 256.00 | 4.00 | 0.000 | 2265.6 ms | 50.63 token/s |
| public_periodic_prefill_proxy_ub128 | 140.45 | 85.66 | 0.769 | 1527.6 ms | 71.51 token/s |

计算规则：`sum_layer sum_expert [T_gate_up(M_e)+T_down(M_e)]`，M_e=0时跳过。
其余投影按层数累加，共享专家每层一次M=128；decode每层执行8个路由专家及1个共享专家，均M=1。
专家顺序执行，每个专家使用可用的三个NPU核；没有假设跨专家/跨层并行，未实现新的grouped MoE调度。
因此这里路由集中通常减少权重读取和调用次数，不应把多GPU专家并行的“热门专家拖慢”结论直接搬过来。

专家M=1..16逐个实测；另实测20/24/32/40/48/64/80/96/112/128，间隔档位线性插值。小M不会按大M的吞吐简单外推。

## 时间构成

| 项目 | prefill128 | decode/token |
|---|---:|---:|
| 30层GDN输入、10层全注意力输入、40层输出与router/shared gate | 298.72 ms | 28.79 ms |
| 路由专家 | 1644.98 ms | 50.87 ms |
| 共享专家 | 31.49 ms | 6.36 ms |
| LM head | 仅最终输出时一次 | 16.63 ms |

## 矩阵形状

模型配置取自 [OpenBMB config.json](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/blob/main/config.json)，原件保存在证据目录。

| 投影 | K → N | 权重 | 每层/模型次数 |
|---|---:|---|---|
| GDN QKV/Z/a/b合并 | 2048 → 12352 | W4 | 30层 |
| 全注意力Q/gate/K/V合并 | 2048 → 9216 | W4 | 10层 |
| GDN/全注意力output | 4096 → 2048 | W4 | 40层 |
| router | 2048 → 256 | W4 | 40层 |
| shared expert gate | 2048 → 1 | W4 | 40层，单NPU核 |
| 每专家gate/up合并 | 2048 → 1024 | W4 | 每层256选8，另1共享 |
| 每专家down | 512 → 2048 | W8 | 同上 |
| LM head | 2048 → 248144 | W8 | 每decode token一次 |

GDN的QK/PV/state等乘法已在完整GDN调用内，不重复加算。

## 板端实测

- 板：zt-firefly，32GB RK3588；CPU4–7/4线程，NPU mask7，小shared gate为mask1；注意力2个KV head限制实际NPU核数。
- 固定NPU 1GHz、DDR 2112MHz、大核2400MHz；共享板锁；两次runner均exit_code=0、restore_errors=[]，逐字段核对频率上下限/governor已恢复。
- 独立从当前算子源码构建，不修改runtime或现有benchmark构建目录。
- 合成权重：固定种子、W4 codes均匀[-8,7]、W8 codes均匀[-128,127]、每输出通道scale=0.005。W4自动选择权重认证Ktile，未放宽INT16界；无DCOMP。
- 合成激活：固定种子正态FP32；线性算子共63种形状，每种在少量有效tile/layout/线程候选中用3次pilot选配方，再3次warmup、15个完整调用取中位数。Ktile不做跨形状搜索，策略不代表穷尽最优。
- 每形状抽查2行×3列，对照CPU整数dot和同样的动态量化；通过。真实checkpoint精度/PPL未测。
- GDN附加测量包括projected Conv/SiLU/L2和状态导入导出；decode GDN使用CPU recurrence，包含非原地状态复制，不含完整mixer。
- 注意力实测是已装载native KV上的run调用，含内部pack/sync/softmax/output；KV追加、RoPE、QK norm和模型调度另计。

## 适用边界

尚未计入视觉编码、tokenizer、embedding、RMSNorm/RoPE、router softmax/top8与分桶、专家token gather/scatter、SwiGLU、路由加权合并、残差、部分GDN门控/输出norm、KV追加、采样和runtime调度。
没有为这些项目随意指定一个百分比，所以“加GDN/注意力”仍是已覆盖算子的累加估算，不是承诺的端到端速度。
合成矩阵重复运行没有复现整模型权重轮转与实际MoE内存访问顺序。实际权重量化分布会改变安全Ktile；W4精度可用性还需真实模型验证。
专家gate/up合并是通常的SwiGLU两路同输入投影；若具体实现拆成两次调用，调用开销会增加。
全体路由专家的W4 gate/up+W8 down权重约20GiB，不包含主干、head、embedding及运行时内存；这里的常驻估算适用于32GB板，存储流式加载须额外建模。

## 证据与复现

证据目录：`llm_generated_docs/evidence/minicpmv47_35a3_20261019/`（仓库按约定忽略）。

- `bench.cpp` / `run.sh`：板端测速；从仓库根目录 `remoterun 'bash llm_generated_docs/evidence/minicpmv47_35a3_20261019/run.sh'`。
- `linear.log` / `linear.csv`：候选pilot、选定策略、63个形状的全部正式样本。
- `other.log`：GDN/注意力11个调用配置的全部正式样本；GDN compensated-TF32另测作敏感性，主估算使用runtime现行TF32。
- `clocks_linear.json` / `clocks_other.json`：定频、温度、恢复记录；`job.exit`=0。
- `routing.py` / `routing_provenance.json` / `routing_raw/` / `routing_counts/`：来源、抽样规则、原始专家ID与ubatch128直方图。
- `summarize.py` / `summary.json`：算子插值与累加，含分布、KV深度、precision敏感性；`report.py`生成本文。

重新汇总：先运行`routing.py`，再运行`summarize.py`和`report.py`。
