# RK3588：IRQ 亲和性、W8A8 prefill 与层内 TP

2026-11-02，主板 / Firefly ITX-3588J，Linux 6.1.118-zt，GCC 10.2.1。

结论：IRQ 大核亲和性使整模型 tg32 decode 提高 2.33%，对 M128 FFN 没有收益；prefill 改 W8A8 的已测方案没有超过当前 M128 W4A8；标准 TP2 更慢，非均匀三路 FFN 在双方都选择更快的 M1 提交配置后仅改善约 0.4%，不值得据此改造整层 TP。

## 测量范围

- CPU4–7 2.4 GHz，NPU 1 GHz，DDR 2.112 GHz，CPU mask `0xf0`；使用共享 `/tmp/rknpu_lowlevel_submit.lock` 串行占用板端 NPU。OpenMP passive wait，`GOMP_SPINCOUNT=0`。
- 单算子库从当前源码重建，Release `-O2`、trace 关闭；探针同为 `-O2`，链接时包装 `ioctl`，记录 submit 次数、submit 墙钟及驱动返回的 `hw_elapse_time`。源码版本及输入来源在 [provenance.json](evidence/idea_eval_20261102/provenance.json)。
- Qwen3-4B 原始混合量化 checkpoint 第 2 层，同层 gate/up/down 的原始 signed W4 codes 与 FP32 channel scales。输入来自 `trace128/p0-l2-post.f32`，M1/M64 取前 1/64 行。H=2560，F=9728，gate/up 为 K2560/N19456，down 为 K9728/N2560。
- 完整 FFN 墙钟包括动态 A8、打包、缓存同步、NPU 提交/等待、归约、SwiGLU 和 down。生产 W4 基线使用融合 `rk_npu_w4a8_run_ffn_f32`，包含 SwiGLU 两 wave 归约融合及 down scale 复用。
- M64/M128 W4 使用已验证的 panel8 / pipeline 配置，自动选择严格安全 Ktile；gate/up Nt4096，down Nt896。它不是本次重新穷举得到的最优配置。M1 最终 TP 对照为基线比较 serial/pipeline/chain 与 1/4 CPU threads 的 36 种 gate/down 组合，各 TP rank 也比较 9 种 gate/down 提交组合；双方均选择 chain。W8 分别调用现有 autotuner；TP 分片比较窄/宽两个 N 分块后再计时。
- W8/TP 每种 M 做 2 个独立进程，每进程 30 个对称周期，每模式每周期 2 次，共 120 次完整调用。M128 对两个 M64 的补测为 2×40 个 ABBA 周期，每模式 160 次。
- 正确性覆盖真实、零、4 倍、负输入；W4、CPU 归约 W8、DCOMP W8 均要求输出逐位一致；TP 并发与相同分片串行执行要求逐位一致；最终输出检查 guard。NPU 反量化及全局 scale TP 有独立误差阈值。

`hw_elapse_time` 是设备提交至完成中断的驱动区间，包含内存访问与中断开销。它不能直接解释为 MAC 忙碌时间。多路 TP 的 submit/hardware 数据是各并发提交的累计量，可大于完整墙钟；阶段累计量同样不能直接相加。

## a. NPU IRQ 亲和性

官方 SDK V2.3.2 用户指南第 5.3.3 节建议把应用及 NPU 中断放在同一 CPU 大核，以减少跨核处理成本。参考本地 [官方用户指南](../../../rknn-toolkit2/doc/02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2_CN.pdf)，文档中的操作对象是 `/proc/irq/<irq>/smp_affinity_list`。

板端 IRQ42/43/44 的初始请求 mask 是 `0-7`，实际 effective CPU 均为 CPU0。本实验固定提交线程在 CPU4；CPU 量化/归约仍允许使用 CPU4–7。比较原配置、全部 IRQ→CPU4、全部 IRQ→CPU7、IRQ 分散至 CPU4/5/6。每个进程按正反顺序各跑一遍，共两个进程，每种配置/形状有 800 次调用。

| 完整调用 | 原配置 | IRQ→CPU4 | 墙钟变化 |
|---|---:|---:|---:|
| 小 FFN，M1/H192/F96，2 次 submit | 94.25 µs | 78.32 µs | -16.9% |
| 真实 FFN，M1，8 次 submit | 1533.46 µs | 1445.86 µs | -5.71% |
| 真实 FFN，M128，8 次 submit | 8118.54 µs | 8150.14 µs | +0.39% |

M1 的两次独立进程分别改善 5.74% / 5.68%；M128 分别增加 0.26% / 0.51%。M1 submit 累计墙钟由 1404.69 降至 1315.66 µs，驱动 hardware 累计区间由 1350.03 降至 1275.49 µs。它证明提交/完成通知路径存在可减少的开销，尚不能把这部分全部归因于启动本身。

把 IRQ 放到 CPU7 或分散到 CPU4/5/6 也能改善小任务，但 M128 的进程间波动较大；它们没有显示比 CPU4 更稳定的真实 FFN 收益。CPU 工作与 IRQ 的竞争需要实测。当前数据不支持把 IRQ 调整作为 prefill 加速手段。

原始数据：[IRQ 汇总](evidence/idea_eval_20261102/irq/summary.json)、[各进程对照](evidence/idea_eval_20261102/analysis.json)、[IRQ 恢复](evidence/idea_eval_20261102/irq/protocol.json)、[时钟恢复](evidence/idea_eval_20261102/irq-clocks.json)。

### 整模型 decode 对照

Qwen3-4B，固定 synthetic token IDs，空 KV，tg32，context512，ubatch128，无投机、无 checkpoint；每组进程复用同一个现有 `target/release/open_rknpu2_lm_infer` v0.8.0 二进制（mtime 2026-10-28）。这是现有二进制的 IRQ 控制实验，单算子测试则链接本日重建的当前源码。

| 完整 decoder + lm_head，tg32 | 平均时延 | 吞吐 |
|---|---:|---:|
| 原 IRQ 配置 | 2863.931 ms | 11.1735 tok/s |
| IRQ42/43/44→CPU4 | 2798.771 ms | 11.4336 tok/s |
| 变化 | **-2.275%** | **+2.328%** |

顺序为 ABBA（原配置、CPU4、CPU4、原配置），每进程两个正式 tg32 样本，共每配置 4 个样本。进程 0/3 原配置平均约 2863.518/2864.345 ms；进程 1/2 CPU4 约 2799.826/2797.716 ms。实际 runtime 仅限制 CPU mask `0xf0`，没有使用单算子探针的提交线程强制绑核包装。

每个进程先执行一次相同 case，以吸收 bench 的 head-free prefill 正确性自检；随后第二 case 的 warmup/测量期间，`tune_cache_misses`、`tune_calls`、`strategy_fallback_calls`、`weight_reload_calls` 增量全部为 0。启动自检的非 M1 fallback 留在累计统计中，没有计入正式 tg32 时延。

计时包含所有 decoder 层、Attention 和每 token 的输出 head；不含加载、准备、分词、sampling 或 HTTP 请求开销。head-free 自检的 continuation logits exact。该短上下文实验不证明长上下文或 HTTP 端到端同幅度提升。

原始数据：[ABBA 协议和计数器](evidence/idea_eval_20261102/e2e-validated/protocol.json)、[统计](evidence/idea_eval_20261102/analysis.json)、[时钟恢复](evidence/idea_eval_20261102/e2e-validated-clocks.json)。

## b. prefill 改用 W8A8

本实验把相同 W4 数值扩展为 INT8，保留原 scale，因此无需重新量化模型。W8 使用 INT32 partial，W4A8 的 MSD/INT16 partial 重建被去掉；但实际 W8 计算和权重传输、额外 gate/up 浮点中间张量，以及独立 SwiGLU 也要计入。

| M128 完整 FFN，同组配对 | W4 基线 | W8 方案 | 墙钟变化 |
|---|---:|---:|---:|
| 未压缩 W8，CPU 归约 | 8.236 ms | 11.872 ms | +44.1% |
| 未压缩 W8，NPU INT32 归约 | 8.236 ms | 11.437 ms | +38.9% |
| 未压缩 W8，NPU 归约与反量化 | 8.236 ms | 16.371 ms | +98.8% |
| DCOMP W8，CPU 归约 | 8.339 ms | 12.038 ms | +44.4% |
| 两个 M64 DCOMP W8 对一个 M128 W4 | 7.978 ms | 8.662 ms | +8.57% |

NPU 归约把两次 GEMM 的总 submit 数从 7 降到 2，仍未缩短完整 FFN 至 W4 以下；较长 PC chain 及输出处理抵消了减少提交次数的收益。NPU 反量化使用 FP16-rounded coefficients，是精度变化路径；实际误差检查通过，但不属于与原 W4 逐位一致的方案。

DCOMP 的边界值得保留：**同为 M64** 时 W4 4.421 ms、DCOMP W8 4.294 ms，改善 2.86%，60 个配对周期中有 57 个更快；M1 即使与固定 pipeline 的 W4 控制相比也增加 29.1%。将 M64 的结果推广到当前 M128 prefill 前，直接补测了两次 M64；完整处理相同 128 行仍比 W4 慢 8.57%，80 个周期中仅 1 个更快。因此本轮不接入 W8 prefill，也不跑 W8 整模型速度测试。

不同进程的 M128 W8 抖动明显，表格使用各自同组 W4 控制，不能交叉选择不同进程的最快数字。无论使用原始 W8、现有 NPU 归约/反量化，还是 DCOMP+M64 切块，都没有形成替换当前 M128 W4 FFN 的证据。

未压缩 W8 的 gate/up+down 常驻 packed payload 为 74,799,104 bytes/层，W4 packed payload 为 37,355,520 bytes/层，另有很小的 scale/correction 元数据。如果 decode 保留 W4、prefill 额外常驻未压缩 W8，36 层 FFN 额外约 **2.51 GiB**，尚未计 Q/K/V/O、workspace 和 allocator 开销。这个数字不适用于 DCOMP 存储。

单层原始 W8 权重创建/打包在两次进程中为 98.9 / 105.1 ms，明显大于一次 FFN 的约 8 ms。此数不含从 checkpoint nibble 解包的成本，也不是完整模型加载时间。常驻预打包可摊销该成本；逐请求重新构造的收益会更差。当前连预打包后的稳态调用都没有收益，不存在通过频繁重打包弥补的空间。

原始数据：[原始 W8/NPU 归约](evidence/idea_eval_20261102/compare/summary.json)、[DCOMP](evidence/idea_eval_20261102/w8/summary.json)、[M64×2](evidence/idea_eval_20261102/split/summary.json)。

## c. 从算子内多核改为 Megatron TP

按 [Megatron 原论文的 MLP 划分](https://research.nvidia.com/labs/adlr/MegatronLM/)，gate/up 按输出列划分，SwiGLU 留在分片，down 按输入 K 划分，最后求和。本实验由持久 CPU 线程分别驱动固定 NPU core，权重和 workspace 独立，单板共享内存求和计入墙钟。

- TP2：F=4864/4864，使用 NPU core0/1，每个 rank 一个 CPU 大核；与现有三 NPU core、四 CPU 大核路径比较。
- TP3：F=3264/3264/3200，使用三个 NPU core、三个 CPU 大核；不是等宽 TP3。主线程执行全局 scale 归并及输出求和。
- 严格数值版本在各 rank 的 SwiGLU 后求全局 row maxabs，使用全 F 共用的 A8 scale，再运行各自 down 并求和。测试包括这些同步、扫描及求和。

| 配置 | 真实 M1 FFN：基线→TP | 变化 | 真实 M128 FFN：基线→TP | 变化 |
|---|---:|---:|---:|---:|
| TP2，全局 A8 scale | 1.338→1.801 ms | +34.5% | 8.767→12.506 ms | +42.6% |
| 非均匀 TP3，全局 A8 scale | 1.339→1.334 ms | **-0.40%** | 8.163→9.893 ms | +21.2% |

M1 的基线 gate/up 与 down 均选择完整 PC chain，共两次 submit；三个 TP rank 各自也使用 chain，总计六次并发 submit。保留全局 A8 scale 后，TP3 M1 的 60 个配对周期有 45 个更快，周期 bootstrap 的时延变化区间约为 -0.54% 至 -0.26%；这只是两个进程内的探索性区间，不能外推成整模型增益。M64 的全局 scale TP3 约慢 10.8%。

直接让各分片独立量化 down 输入会省掉全局 max 同步，但原始输入上的 FFN 输出相对 L2 误差为 5.44%，不能用于声称同数值加速。全局 scale 版本的误差约 10^-7，并发结果与相同分片串行执行逐位一致；剩余微小差异来自各分片 FP32 反量化后求和的舍入。

M1 提交/等待占比高，但现有 chain 已经将每个 GEMM 的多 K waves 合并为一次提交。TP3 剩下的计算/分片收益基本被全局 scale 同步、调度及输出求和抵消。M128 则要付出更多分片 CPU 处理、全局 scale 同步和输出求和成本，使完整 FFN 更慢。

标准 TP2 没有收益，TP3 的约 5 µs 单层 M1 收益不足以支持整层改造，本轮不接入或跑 TP 整模型测试。Qwen3-4B 的 Q heads=32、KV heads=8，不能直接等分为三个 attention rank；非均匀切分或 padding 可实现，但其 Attention/O 投影、同步、内存及整模型收益尚未测试。IRQ 与实际提交线程的一一绑定及其 TP3 组合，见[逐核绑定实验](IRQ_CPU_THREAD_PAIRING_20261102.md)。

原始数据：M1 [TP2](evidence/idea_eval_20261102/tp2-final/summary.json)、[TP3](evidence/idea_eval_20261102/tp3-final/summary.json)、[配置选择及数值日志](evidence/idea_eval_20261102/tp3-final/p0-m1.log)；M64/M128 [TP2](evidence/idea_eval_20261102/tp2/summary.json)、[TP3](evidence/idea_eval_20261102/tp3/summary.json)。表中每个形状均使用同一组内的配对基线。

## 适用边界与交付状态

算子实验覆盖第 2 层实际权重和捕获输入，不是逐层扫描，也没有证明所有模型、形状或未来 W8 kernel 都无收益。真正从浮点权重重新量化的 W8 checkpoint 质量不在本实验范围；本轮 W8 采用原 W4 数值的无损改格式。W8 和 TP 的判断依据是完整 FFN 单元测试，未将局部结果称为全模型 prefill/decode 加速。

生产源码未修改。IRQ 及 CPU/NPU/DDR 时钟上下限恢复至实验前设置；运行协议、逐次 CSV、正确性日志与恢复报告全部保留。

[交付核验](evidence/idea_eval_20261102/verification.json)通过：正式算子套件的正确性日志均 PASS，7 组正式板端时钟报告退出 0 且恢复完成，ABBA 正式 case 的计数器增量均为 0。板端再次读取三个 NPU IRQ：请求列表均恢复为 `0-7`，effective CPU 均为 0。

## 复现

代码：[完整调用探针](../native/rk_npu_matmul/tests/idea_eval/probe.cpp)、[板端驱动](../native/rk_npu_matmul/tests/idea_eval/run.py)、[统计](../native/rk_npu_matmul/tests/idea_eval/analyze.py)、[整模型 IRQ 对照](../native/rk_npu_matmul/tests/idea_eval/e2e.py)。生产源码未改动。

在板端项目根目录构建：

```bash
cmake -S native/rk_npu_matmul -B native/rk_npu_matmul/build/idea-eval -DBUILD_TESTING=OFF
cmake --build native/rk_npu_matmul/build/idea-eval --target rk_npu_matmul_static -j4
g++ -O2 -fopenmp -march=armv8.2-a+fp16 -std=c++17 \
  -I native/rk_npu_matmul/include -I native/rk_npu_matmul/src \
  native/rk_npu_matmul/tests/idea_eval/probe.cpp \
  native/rk_npu_matmul/build/idea-eval/librk_npu_matmul.a \
  -Wl,--wrap=ioctl -ldl -pthread -o native/rk_npu_matmul/build/idea-eval/probe

python3 scripts/run_board.py --wait-lock --report /tmp/irq-clocks.json -- \
  python3 native/rk_npu_matmul/tests/idea_eval/run.py irq \
  --rows 1 128 --cycles 100 --processes 2 --output /tmp/irq-eval

python3 scripts/run_board.py --wait-lock --report /tmp/w8-clocks.json -- \
  python3 native/rk_npu_matmul/tests/idea_eval/run.py compare --mode w8 \
  --rows 1 64 128 --cycles 30 --processes 2 --output /tmp/w8-eval

python3 scripts/run_board.py --wait-lock --report /tmp/split-clocks.json -- \
  python3 native/rk_npu_matmul/tests/idea_eval/run.py compare --mode split \
  --rows 128 --cycles 40 --processes 2 --output /tmp/split-eval

python3 scripts/run_board.py --wait-lock --report /tmp/tp-clocks.json -- \
  python3 native/rk_npu_matmul/tests/idea_eval/run.py compare --mode tp --tp-ranks 3 \
  --rows 1 64 128 --cycles 30 --processes 2 --output /tmp/tp3-eval
```

TP2 将最后一条中的 `--tp-ranks` 改为 2；W8 NPU 归约/反量化对照使用 `--mode compare`。路径默认指向本板共享 fixture，可通过 `--weights` 指定。原始 CSV、正确性日志、时钟/IRQ 恢复报告保存在输出目录。
