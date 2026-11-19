# RK3588 stable causal GQA prefill：CPU 直接查表写入 PV 的单算子结果

Qwen3-4B 在 WikiText 前 4096 个 token 上的第 0、17、35 层，保留每行真实最大值，由 CPU 在减 max 的同一遍直接生成 PV 输入。单层从空 KV 增长至 512 的耗时下降 **15.39%–16.92%**，增长至 4096 下降 **4.20%–4.83%**。三个层、两种长度的全部 **56,623,104** 个 FP32 增长输出与原库逐位一致。4K 的时间线上，更多 CPU 减 max 和查表工作落在 NPU 提交区间之外。本实验仅构建隔离探针，生产库未合入改动。

## 真实 QKV 单层回放

| 层号 | 增长至 512，原版 → 实验版 ms | 耗时下降 | 增长至 4096，原版 → 实验版 ms | 耗时下降 |
|---|---:|---:|---:|---:|
| 0 | 12.811 → 10.684 | 16.61% | 300.169 → 287.014 | 4.38% |
| 17 | 12.887 → 10.707 | 16.92% | 299.031 → 286.475 | 4.20% |
| 35 | 12.687 → 10.735 | 15.39% | 299.662 → 285.194 | 4.83% |

每个层、每个长度的两组进程配对均支持增长加速。逐对下降范围在 512 为 14.67%–17.49%，在 4K 为 3.91%–5.23%。层号从 0 开始，三个点分别取自模型开头、中部和末尾。[第 0 层样本与 trace 统计](evidence/gqa_cpu_lut_20261118/wiki-l0/summary.json) · [第 17 层](evidence/gqa_cpu_lut_20261118/wiki-l17/summary.json) · [第 35 层](evidence/gqa_cpu_lut_20261118/wiki-l35/summary.json)

| 层号 | KV512 上末尾 Q128，原版 → 实验版 ms | 耗时下降 | KV4096 上末尾 Q128，原版 → 实验版 ms | 耗时下降 |
|---|---:|---:|---:|---:|
| 0 | 3.366 → 3.046 | 9.51% | 15.902 → 15.397 | 3.18% |
| 17 | 3.301 → 2.839 | 13.99% | 15.907 → 15.401 | 3.18% |
| 35 | 3.270 → 3.277 | −0.21% | 15.894 → 15.254 | 4.03% |

第 35 层的 KV512 热调用首次两对分别慢 13.63%、快 13.21%。使用相同输入和二进制另跑一轮 ABBA，把每进程正式次数从 3 增至 12，耗时从 3.339 降至 3.195 ms（4.33%），两对降幅为 7.53% 和 0.87%。这个毫秒级热调用的收益幅度仍有波动。同轮从空 KV 增长至 512 和 4096 分别下降 18.73% 和 4.52%。热调用预载全部 KV，增长测试以 128 token 为一块逐次追加、逐次执行 Attention。[独立复测样本和 trace](evidence/gqa_cpu_lut_20261118/wiki-l35-repeat/summary.json) · [复测时钟](evidence/gqa_cpu_lut_20261118/wiki-l35-repeat/clocks.json)

## 实现和数值

原版 A 取 native 提交 `3be24de` 的 stable 路径，启用双 slot 和 M128 2D LUT。实验版 B 在同一份源码上保留 QK、每行 max、FP16 减法和截断。CPU 以中心化 FP16 的幅值位模式为索引，通过 38,914 字节的表，原位写回 FP16 exponent。可见位置保留现有 LUT 正偏置，真实行中的未来位置和 padding key 写零；物理补齐行按原始 DPU 行为填充。PV 改读 score 缓冲，原有 FP16 PV、分母辅助列、偏置修正、FP32 partial 合并和归一化顺序均沿用。causal LUT 路径的设备查表、独立 exponent 分配和尾带同步被移除。CPU↔NPU 的 score 所有权同步照常计时。[实验补丁](evidence/gqa_cpu_lut_20261118/candidate.patch) · [构建与 PV 捕获代码](evidence/gqa_cpu_lut_20261118/prepare.py)

- 对实验版查表函数进行主机编译，枚举 19,457 个 FP16 幅值，与已有 RK3588 DPU 实测曲线逐项核对，0 项不同。[标量核对](evidence/gqa_cpu_lut_20261118/curve-check.json)
- 在本轮板端，合成 QKV 的 512/4K 热调用比较了 36 MiB 的实际 PV 输入，逐位一致；两次增长的全部 FP32 输出共 72 MiB，逐位一致。真实 22-token 对话的全部 36 层，PV 输入及输出均逐位一致。[逐文件比对](evidence/gqa_cpu_lut_20261118/validation.json)
- WikiText 三个层的 512/4K 增长输出逐位比较，并检查所有值有限。另对第 17 层 512/4K 热调用的实际 PV 输入逐项比较，18,874,368 个 FP16 值全部一致。中间量捕获、输出保存和比对使用独立运行。[逐层输出与 PV 输入](evidence/gqa_cpu_lut_20261118/wiki-validation.json)

## 为什么 4K 只兑现几个百分点

用相同种子的 FP16 合成 QKV 做八进程配对。512 增长从 12.932 降至 10.915 ms，降幅 15.60%；4K 增长从 301.934 降至 291.961 ms，降幅 3.30%，四组配对分别为 3.06%、4.00%、2.87%、3.28%。末尾 Q128 热调用在 512 和 4K 的平均降幅分别为 9.75% 和 1.70%；512 热调用有一组反向。[无 trace 样本、配对和分阶段数据](evidence/gqa_cpu_lut_20261118/synthetic/summary.json)

| 完整合成 4K 增长的诊断项 | 原版 | 实验版 |
|---|---:|---:|
| 提交次数 / 实际 task 数 | 384 / 10,240 | 256 / 5,632 |
| QK 驱动区间 ms | 54.40 | 54.14 |
| LUT 驱动区间 ms | 66.07 | 0 |
| PV 驱动区间 ms | 78.12 | 78.81 |
| CPU centering 阶段累计 ms（B 包含查表） | 108.09 | 150.73 |
| 其中落在所有 NPU 提交区间之外的 ms | 33.43 | 88.37 |
| 整个调用在 NPU 提交区间外的 ms | 102.72 | 160.91 |

移除 LUT 后，QK 和 PV 的时间基本持平。CPU 第二阶段增加约 43 ms，同时调度中可供 CPU 重叠的 NPU 工作缩短，centering 暴露时间增加约 55 ms。原来能隐藏的 CPU 工作现在出现在关键路径上。这两项有重叠，不能相加扣费。真实三个层的 4K trace 也有同样形态，原版 LUT 为 66.7–66.9 ms，centering 提交外时间从 34–35 ms 增至 87–89 ms。以项目采用的 3 TFLOP/s FP16 峰值和可见因果 QK+PV 工作量折算，真实三层 4K 增长的有效利用率约从 **15.3%** 升至 **16.0%**。设备区间来自驱动时间戳，CPU 重叠来自 Perfetto 时间区间交集；以上数据没有测量 DDR 流量或缓存 miss。[时间线分析脚本](evidence/gqa_cpu_lut_20261118/analyze.py)

## 协议与复现

主板，Linux 6.1.118-zt，RKNPU 0.9.8，Clang 22.1.3、O2、ARMv8.2 FP16、LLVM OpenMP。固定 CPU4–7 为 2.4 GHz，NPU 三核 1 GHz，DDR 2.112 GHz，Hq/Hkv=32/8，D128，KV tile512，ubatch128。A、B 分别编译 Attention 对象，并链接相同的 trace 库和其余 native 对象。两边通过公共 C API 执行。归档中的 probe 沿用旧探针的 CSV `shift` 列；本轮两边均选择 stable，固定平移模式没有参与。[硬件与链接记录](evidence/gqa_cpu_lut_20261118/system.txt) · [构建命令](evidence/gqa_cpu_lut_20261118/build.sh)

增长计时包含已预留容量上的 KV append、公共 Attention 全调用；输入转换、构造 chunk、KV reset 均在计时外。合成输入用种子 20261104、标准差 0.5；进程顺序 ABBA/BAAB，每进程每用例预热 2 次，正式 5 次。真实输入取 WikiText 前 4096 个 token，同次模型推理捕获 post-RoPE Q/K 和 V；每层 ABBA，预热 2 次、正式 3 次，每进程另跑一次 trace。先取每进程无 trace 样本中位数，再按方法平均。百分比表示延迟下降。增长复验含每个中间 chunk 的全部 query 输出。正式计时运行和板锁由 `run_board.py` 管理，已核对退出状态及 CPU/NPU/DDR min/max 和 governor 恢复。[合成时钟](evidence/gqa_cpu_lut_20261118/synthetic/clocks.json) · [捕获时钟](evidence/gqa_cpu_lut_20261118/capture-clocks.json) · [真实回放时钟](evidence/gqa_cpu_lut_20261118/replay-clocks.json)

真实 QKV 来自已归档的 Qwen3-4B-W4A8-Mixed-LMHeadW8 捕获程序，运行时版本 `e944563`。捕获作业的模型策略缓存有 16 个 miss、576 次 fallback，在线调优为 0；A/B 共用捕获结果，捕获时间不参与比较。三层 4K 的 FP16 V 最大幅值分别为 0.559、36.25、48.875，均在此 LUT 路径幅值范围内。本次比较回答这一 checkpoint、这段文本、所选三层的单算子问题。整模型 TTFT 和 PPL 未测。[输入与幅值](evidence/gqa_cpu_lut_20261118/fixture.json) · [模型捕获记录](evidence/gqa_cpu_lut_20261118/capture-model.json)

板端完整快照位于 `<BOARD_HOME>/mnt/3090x4/tmp/gqa-cpu-lut-20261118`，主机可访问[同一快照](<HOST_HOME>/mnt/3090x4/tmp/gqa-cpu-lut-20261118)。其中包含 A/B 可执行文件、冻结的源码与依赖、原始 QKV、每进程 CSV/trace 和逐位比较的输出。固定频率后由 [measure.sh](evidence/gqa_cpu_lut_20261118/measure.sh) 运行合成实验，[measure_replay.sh](evidence/gqa_cpu_lut_20261118/measure_replay.sh) 运行真实回放，[verify_outputs.py](evidence/gqa_cpu_lut_20261118/verify_outputs.py) 核对原始输出。
