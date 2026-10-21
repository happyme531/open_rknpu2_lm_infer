# MiniCPM-V 4.7 35B-A3B：融合 MoE FFN 实测与主干累加估算

2026-10-19。RK3588 32GB；CPU4–7、2.4GHz；NPU三核1GHz；DDR2112MHz；ubatch128。
使用现有MoE调度、NPU gather、每核专家任务链、CPU SwiGLU和route combine。
路由/共享专家均为gate/up W4A8、down W8A8，LM head W8A8，其余投影W4A8。

## 完整融合 FFN 实测

hidden=2048、intermediate=512、256路由专家、top-8、1共享专家。每层权重514MiB，workspace约35MiB。
一次FFN调用包含输入量化、路由分组、任务生成、gather、W4 gate/up、SwiGLU/中间激活量化、W8 down、加权合并及全部同步。
router logits、softmax/top8、共享专家sigmoid门控及外部残差在此调用之外。

| 场景 | 行数 | 完整FFN中位数 | 正式样本P10–P90 |
|---|---:|---:|---:|
| decode | 1 | 0.874 ms | 0.868–0.880 ms |
| same_arch_alpaca_proxy | 128 | 28.892 ms | 28.787–29.031 ms |
| concentrated_U160 | 128 | 25.841 ms | 25.793–25.987 ms |
| concentrated_U128 | 128 | 24.376 ms | 24.357–24.429 ms |
| uniform_four_each | 128 | 29.944 ms | 29.901–30.062 ms |

各prefill分布与原估算使用同一组专家负载计数，每token选8个不同专家，每层1024次选择守恒。由专家计数构造有效token→expert分配，按remaining count贪心实现，未声称是真实MiniCPM路由。decode先后各测15样本，覆盖128→1 workspace切换；prefill每情形15样本，均另有3次warmup。

## 更新后的语言主干累加

| 路由情形，ubatch128 | 原拆分估算，KV128 | 融合FFN累加，KV128 | 融合FFN累加，KV4096 |
|---|---:|---:|---:|
| same_arch_alpaca_proxy | 57.21 token/s | 74.56 token/s | 67.93 token/s |
| concentrated_U160 | 65.65 token/s | 80.27 token/s | 72.63 token/s |
| concentrated_U128 | 73.60 token/s | 83.33 token/s | 75.13 token/s |
| uniform_four_each | 50.63 token/s | 72.78 token/s | 66.44 token/s |

Decode，KV=128：**86.42 ms/token，11.57 token/s**；原拆分估算9.20 token/s。

Decode，KV=512：**86.97 ms/token，11.50 token/s**；原拆分估算9.15 token/s。

Decode，KV=2048：**88.83 ms/token，11.26 token/s**；原拆分估算9.00 token/s。

Decode，KV=4096：**91.33 ms/token，10.95 token/s**；原拆分估算8.80 token/s。

这是替换FFN项后的算子累加更新。原FFN项只计GEMM，现项包含完整FFN；两次合成权重与Ktile配方也不同，故表中变化不能当作同权重、同覆盖范围的完整模型A/B加速比。

## 详细构成（205专家参考分布，KV128）

| 项目 | prefill单次/层 | 次数 | prefill合计 | decode单次/层 | 次数 | decode合计 |
|---|---:|---:|---:|---:|---:|---:|
| GDN输入投影 W4 | 5.864 ms | 30 | 175.930 ms | 0.487 ms | 30 | 14.604 ms |
| 全注意力输入投影 W4 | 4.139 ms | 10 | 41.394 ms | 0.370 ms | 10 | 3.698 ms |
| 输出投影 W4 | 1.416 ms | 40 | 56.642 ms | 0.184 ms | 40 | 7.373 ms |
| router线性 W4 | 0.368 ms | 40 | 14.712 ms | 0.048 ms | 40 | 1.925 ms |
| 共享门控线性 W4 | 0.251 ms | 40 | 10.045 ms | 0.030 ms | 40 | 1.190 ms |
| 完整融合MoE FFN | 28.892 ms | 40 | 1155.667 ms | 0.874 ms | 40 | 34.960 ms |
| GDN核心及prefill Conv等 | 7.915 ms | 30 | 237.452 ms | 0.177 ms | 30 | 5.320 ms |
| 全注意力核心 | 2.486 ms | 10 | 24.865 ms | 0.072 ms | 10 | 0.723 ms |
| LM head W8 | 末尾输出时一次 | — | 主干prefill不计 | 16.627 ms | 1 | 16.627 ms |
| **已覆盖算子合计** | — | — | **1716.706 ms/128 tokens** | — | — | **86.420 ms/token** |

## 融合FFN内部阶段（单层中位数）

| 阶段 | prefill128，205专家参考 | decode1 |
|---|---:|---:|
| 一次输入量化/拆INT4 | 0.197 ms | 0.023 ms |
| 专家分组及任务分配 | 0.194 ms | 0.006 ms |
| 任务链生成 | 0.821 ms | 0.032 ms |
| 缓存同步合计 | 1.220 ms | 0.025 ms |
| NPU gather | 0.868 ms | 0.000 ms |
| W4 gate/up NPU任务链 | 8.059 ms | 0.348 ms |
| CPU split-K还原、SwiGLU、中间A8打包 | 5.244 ms | 0.040 ms |
| W8 down NPU任务链 | 7.609 ms | 0.347 ms |
| route加权及合并 | 4.647 ms | 0.051 ms |

总时间采用外部wall time；阶段取各自中位数，因此阶段中位数之和未必等于总中位数。decode直接共享量化输入，gather任务为0。

## 实现与验证

- 移除hidden≤1536、routed_experts≤128两处软件限制，目标2048/256实际跑通。
- 原全W8 MoE及“路由全W4、共享W8”接口保留；新增实验内部入口`rk_npu_moe_w4w8_weights_create/workspace_create`，共享原run/free/timings API。
- gate/up按K512分片，创建权重时严格检查INT16范围；一个每核任务链提交多个专家/多个K片，CPU用INT32精确重建并归并，未移除数值正确性检查。down使用W8，包含共享专家。
- max_rows=128 workspace按当前有效rows同步每个K片，避免decode同步整块prefill最大缓冲。
- 目标形状decode全行、每个prefill分布的0/63/127行，全部2048列对照独立CPU整数dot、nearest-even量化、SwiGLU及路由合并；本次抽查全部逐值一致（relL2=0，max_abs=0）。这不等同于真实checkpoint质量验证。
- 原W8、原W4、NPU-middle及新增W4/W8路径的公共API回归通过，覆盖变长、重绑定、重复专家ID、零系数、空专家、输入量化等。
- 运行持有共享板锁、固定频率；`clocks_fused.json`和`clocks_moe_regression.json`记录恢复；`fused.exit=0`。

## 准确度与边界

| 层次 | 证据与可信范围 |
|---|---|
| 固定合成权重/路由下的单层融合FFN | 完整板端调用实测，含所有内部数据搬运、量化、激活和combine；比独立GEMM相加直接得多 |
| 40层FFN项 | 同层几何乘40；未运行整模型20GiB专家权重的层间轮换 |
| 单prompt真实路由 | 未取得MiniCPM checkpoint路由；205/160/128/均匀是参考及敏感性情形，不是置信区间 |
| 真实权重量化 | 合成W4/W8 codes，无真实模型精度/PPL；实际分布可能允许不同Ktile |
| 完整语言推理 | 未包含embedding、norm/RoPE、router softmax/top8、共享专家外部门控、残差、KV追加、采样和runtime调度，不能给出已验证的端到端误差百分比 |
| 视觉输入 | 不包含ViT与图片预处理；图像token进入语言部分后的成本可用同ubatch模型估算 |

专家分布、原始单算子样本和旧估算见[拆分算子报告](MINICPMV47_35A3_OPERATOR_ESTIMATE_20261019.md)。
证据目录`llm_generated_docs/evidence/minicpmv47_35a3_20261019/`：`bench_fused.cpp`、`run_fused.sh`、`fused.log`、`fused_summary.json`、`moe_regression.log`、定频报告与生成的有效路由文件。改动尚未commit，未改动runtime模型路径。
