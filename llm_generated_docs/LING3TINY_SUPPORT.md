# Ling-3.0-tiny

分支 `feat/ling3tiny`，基于 `feat/minicpmv47` 的 `8bf5568`。

主开发板 `zt-firefly` 已验证 checkpoint：
`/home/firefly/mnt/spark-test/Ling-3.0-tiny-QQQ-W4A8-Mixed-CT`。

## 执行路径

`bailing_hybrid` 通过 `model::load` 加载，使用标准 CLI、TUI、API 和
`LanguageModel` 接口。配置、权重映射和执行位于 `src/model/ling3tiny/`。
支持 compressed-tensors per-channel W4/W8、动态 per-token A8。MLA 的
q_a/q_b/kv_a/g_proj 使用公共 NPU FP16 Linear，MoE router 使用公共 raw MM
的 TF32×TF32→FP32；KDA 的保护
浮点投影仍沿用公共 CPU 路径。首层使用 dense FFN，其余层使用 routed/shared experts。
专家权重支持 resident 与 `--expert-storage stream --expert-store DIR`。

KDA 的 Q/K/V/F/G 通过公共 `Projection` 按位宽合并，卷积、forget gate 和
输出 gate 读取合并结果的 strided view。批量 L2 norm、decay/beta 和
RMSNorm+sigmoid 按 head 并行，保持每个 head 的 FP32 求和顺序；输出门控
原地完成。单 token 卷积直接调用同一 NEON 内核。Router 复用选择 scratch，
组内 top-2 使用线性扫描，最终 top-k 保持分数降序及相同分数时 expert ID
升序。MLA RoPE 预计算频率，同一位置的 Q heads 与共享 K 复用 sin/cos。

| 部件 | 实现 |
|---|---|
| KDA | `native/cpu_kernels/kda.cpp`，ggml KDA recurrence，OpenMP/NEON，FP32 state |
| CPU ABI | `src/ops/kda.rs` 校验维度、buffer 和 gate/beta |
| KDA 层 | Q/K/V causal conv + SiLU、L2 norm、safe vector gate、delta update、sigmoid gated RMSNorm |
| MLA | q_a/q_b/kv_a/g_proj 为 NPU FP16，dense 输出投影保留 checkpoint 的 NPU W4/W8；`src/npu/mla.rs` 调用公共 `rk_npu_mla_f16` absorbed 模式 |
| MLA KV | 逻辑状态为 normalized latent512 和 rotated key64；设备采用共享 K576/V512 布局，分配量由 cache-info ABI 报告 |
| RoPE | 相邻 pair 交错，theta=6,000,000，仅作用于 64 个 rotary channels |
| Router | 整个 ubatch 使用 NPU TF32 GEMM，FP32 输出；CPU sigmoid、expert bias、分组 top-k、归一化并乘 2.5 |
| Shared expert | 直接加到 routed 输出 |

官方配置为 24 层、16 heads、head_dim128、18 个 KDA 层和 6 个 MLA 层。
首层 dense intermediate4608；其余层 128 experts/top8，routed/shared
intermediate512。每个 KDA 层保留 1 MiB FP32 recurrence state。
MLA 以最多 32 行调用 absorbed attention，外层 ubatch 保持运行时设置。
非零前缀缩短由 engine reset/recompute；reset 同时清除 recurrence、卷积和 MLA KV。

`kv_b_proj` 浮点权重或 W4/W8 权重在加载时还原一次，并交给 FP16 MLA
权重打包器。其 absorbed 路径的舍入位置与 expanded/dynamic-A8 路径不同，
实际模型的 logits/质量验证需要使用同一份转换 checkpoint。

## 验证入口

板端本地构建目录可设为 `/tmp/ling3tiny-target-20261022`。

```sh
export CARGO_TARGET_DIR=/tmp/ling3tiny-target-20261022
cargo fmt --check
cargo clippy --all-targets --release --locked -- -D warnings
cargo test --release --locked
python3 scripts/run_board.py --report results/ling-mla-clock.json -- \
  cargo test --release --locked --lib \
  npu::mla::tests::absorbed_cache_matches_expanded_oracle_across_prefill_decode_and_reset \
  -- --ignored --nocapture
```

`tests/fixtures/ling3tiny/config.json` 来自原始 inclusionAI Ling-3.0-tiny
checkpoint。CPU KDA 单元测试使用独立 FP64 recurrence，检查 vector decay、
非零初始 state、单线程/四线程、分块和 reset。MLA 测试使用可解析的稀疏
投影和 expanded CPU oracle，覆盖 33 行 prefill、decode、KV 扩容和 reset。

小型端到端 fixture 保留真实的 KDA/MLA head geometry，使用 H64、四层、
八个专家。NumPy 参考使用 key-major FP64 recurrence、expanded MLA 和
动态 A8/W4 专家，与运行时实现独立。

```sh
OPENBLAS_NUM_THREADS=4 python3 tests/tools/ling3tiny_reference.py \
  --output results/ling3tiny-fixture
cargo build --release --locked --example ling3tiny_check
python3 scripts/run_board.py --report results/ling-fixture-clock.json -- \
  "$CARGO_TARGET_DIR/release/examples/ling3tiny_check" \
  --model results/ling3tiny-fixture \
  --tokens 17,33,6,45,128,57,91,3,88 --continuation 42,123,89 \
  --reference results/ling3tiny-fixture/reference.f32 \
  --report results/ling-fixture-resident.json
```

同一检查增加 `--storage stream --expert-store results/ling3tiny-experts`
即可验证 streaming。检查器保存固定 token IDs 的 FP32 logits、参考误差、
chunk1/3/7、head-free prefill、取消后 reset/replay 和每层 cache-info。

`ling3tiny_reference.py --quantize-projections` 还会量化 attention/output
投影，覆盖窄 W4 门控和量化 `kv_b_proj` 的加载。真实 checkpoint 的独立
NumPy 参考入口为 `--checkpoint DIR --output logits.f32 --tokens ID,...`；
可用 `--trace-dir DIR` 保存逐层参考。参考采用 FP64 BLAS 计算整数 dot：
整数乘积与部分和均小于 2^53，因此 INT32 矩阵结果保持精确。

## 主板验证

本节及下方原始性能表为 router/MLA 前投影迁移到 NPU 之前的记录。
迁移后的验证与性能结果单独追加，原始数据保留。

CPU4–7 四线程、NPU mask7；板端 runner 使用共享提交锁并固定 CPU/NPU/DDR
频率，退出后恢复。源码通过 fmt、全 targets release Clippy `-D warnings`；
默认测试合计 113 passed、21 ignored，另显式运行 MLA NPU 测试通过。

| 检查 | 结果 |
|---|---|
| CPU KDA 对独立 FP64 recurrence | PASS，含非零初始 state、vector decay、分块、单/四线程和 reset |
| MLA 对 expanded CPU oracle | PASS，33 行 prefill、decode、reserve 和 reset |
| 小型模型对独立 NumPy 参考 | 浮点保护投影 NRMSE 1.9809e-5；27 个 W4 投影 NRMSE 0.0019069；均 4/4 top-1 一致 |
| 真实模型 resident / streaming | 4 行、628736 个 logits 逐位一致；streaming 每层 10/128 个 hot slots |
| 真实模型 chunk1/3/7、head-free prefill、取消后重放 | 全部逐位一致 |
| 真实模型 KV | 6 层均 absorbed；容量 256 时合计 5332992 bytes，KDA FP32 state 共 18 MiB |
| OpenAI API 两轮真实文本 | 正确解释天空的蓝光散射；算术题回答“小明现在有15个苹果。”，均在 EOS 停止 |
| WikiText-2，8×512 窗口，2040 个评分 token | PPL 11.956034；随包参考报告 11.907040 |

固定 logits 输入为 `17,33,6,45,128,57,91,3,88`，teacher-forced continuation
为 `42,123,89`。PPL 使用 `spark-test/eval-data/wiki.test.raw`、当前 checkpoint
tokenizer，无额外 BOS/EOS；窗口起点为 0,4096,…,28672，每窗保留前 256 个
context token、评分目标位置 257…511。运行时使用 FP32 CPU 层与 maxabs/127
动态 A8；随包参考采用 BF16 输出与 maxabs/127.5 动态 A8。PPL 差为 +0.4115%，
两者的数值路径不同。

证据保存在本 worktree 的 `results/ling-real-resident.json`、
`results/ling-real-stream.json`、`results/ling-real-api-final.json`、`results/ling-real-ppl.json` 及对应
`*-clock.json`。所有成功验证的 runner 均记录 `restore_errors=[]`。
`tests/tools/ling3tiny_api_check.py` 可独立重跑两条 API 文本检查。

板端 `/tmp/ling3tiny-expert-store-20261022` 已准备原生 expert records。
tmpfs 的读取使用 `--expert-buffered-io`；例如：

```sh
MODEL=/home/firefly/mnt/spark-test/Ling-3.0-tiny-QQQ-W4A8-Mixed-CT
/tmp/ling3tiny-target-20261022/release/open_rknpu2_lm_infer \
  --model "$MODEL" --expert-store /tmp/ling3tiny-expert-store-20261022 \
  --expert-buffered-io run --temp 0 --conversation -n 48 \
  -p '用一句话解释为什么天空是蓝色的。'
```

## Prefill512 / decode128

主板、resident experts、ubatch128、context640，连续 pp512→tg128；CPU4–7
2.4 GHz、NPU 三核 1 GHz、DDR 2.112 GHz。内置 MT19937 seed42 随机 token，
1 次 warmup、3 次计时，表内速度为均值 ± 样本标准差。

| 阶段 | 平均耗时 | tokens/s |
|---|---:|---:|
| Head-free prefill512 | 4.028974 s | 127.0797 ± 0.2002 |
| Decode128，历史深度 512…639，含 lm_head | 6.961592 s | 18.3866 ± 0.0324 |

Prefill 原始 ms：4026.848297、4036.117282、4023.957554；decode 原始 ms：
6949.214058、6961.825292、6973.737977。Decode 输出头约 1.0784 s，其余
decoder 约 5.8829 s。计时包含完整模型调用，排除加载、准备、tokenization、
sampling 和 stats。128-token head-free 检查的 continuation logits 逐位一致，
prefill 输出头耗时为零。

本次显式 `--no-tune`，使用内置 fallback 策略；`tune_cache_hits=0`、
`tune_cache_misses=0`、`tune_calls=0`、`strategy_fallback_calls=86699`，
`weight_reload_calls=0`。原始报告为
`results/ling-pp512-tg128-default.json` 和 `results/ling-pp512-tg128-default-clock.json`，
时钟恢复无错误。复现：

```sh
python3 scripts/run_board.py --report results/ling-pp512-tg128-default-clock.json -- \
  /tmp/ling3tiny-target-20261022/release/open_rknpu2_lm_infer \
  --model "$MODEL" --ctx-size 640 --ubatch-size 128 \
  --expert-store /tmp/ling3tiny-expert-store-20261022 --expert-storage resident \
  --expert-buffered-io --no-tune bench -p 0 -n 0 -pg 512,128 -r 3 \
  --report results/ling-pp512-tg128-default.json
```

## 线性层自动调优

2026-10-22 的线性层自动调优覆盖 M=1/128，保存到
`/home/firefly/.cache/open_rknpu2_lm_infer/ling3tiny-20261022`。
调优阶段含加载用时 42.16 s：66 次搜索、20 次缓存命中、0 fallback。
正式测速读取保存的配置：两组均为 `tune_cache_hits=86`、
`tune_cache_misses=0`、`tune_calls=0`、`strategy_fallback_calls=0`、
`weight_reload_calls=0`。KDA、MLA 与 grouped MoE 使用现有专用实现。

沿用上一节相同的模型、频率、resident、640 个固定随机 token 和连续
pp512→tg128 协议；未调优/已调优各跑两组，每组 1 次 warmup、3 次采样。
下表统计全部 6 个样本，保留所有原始值：

| 阶段 | 未调优中位数 tok/s | 调优后中位数 tok/s | 变化 |
|---|---:|---:|---|
| Prefill512 | 127.2450 | 128.1611 | 基本不变 |
| Decode128 @512 | 18.3818 | 19.4595 | +5.8629% |

全部样本的均值 ± 标准差：prefill 未调优 127.2862 ± 0.3100、调优后
126.5871 ± 4.1882 tok/s；decode 未调优 18.3850 ± 0.0279、调优后
19.4708 ± 0.0786 tok/s。调优后的 prefill 含一个 4337.155303 ms 样本，
其余五个为 3978.907502…3997.318896 ms，当前证据没有显示明显 prefill 收益。

| 配置/阶段 | 六个原始耗时 ms |
|---|---|
| 未调优 prefill | 4026.848297, 4036.117282, 4023.957554, 4023.508261, 4017.480895, 4006.797305 |
| 调优 prefill | 3997.318896, 4337.155303, 3993.049426, 3978.907502, 3987.967655, 3996.894807 |
| 未调优 decode | 6949.214058, 6961.825292, 6973.737977, 6972.814748, 6965.023069, 6950.640224 |
| 调优 decode | 6529.515311, 6605.791754, 6562.568845, 6584.199404, 6590.673317, 6571.369412 |

所有正式测量的 head-free 检查通过，continuation logits 逐位一致，
时钟恢复无错误。报告为 `results/ling-tune-1-128.json`、
`results/ling-pp512-tg128-tuned.json`、`results/ling-pp512-tg128-default-recheck.json`、
`results/ling-pp512-tg128-tuned-recheck.json` 及对应时钟报告；汇总为
`results/ling-pp512-tg128-tuning-comparison.json`。

复现调优和测速时，使用上一节命令的相同参数，将 `--no-tune` 替换为
`--tune-cache /home/firefly/.cache/open_rknpu2_lm_infer/ling3tiny-20261022`。
首次搜索使用 `tune --batches 1,128 --report results/ling-tune-1-128.json`；
测速仍使用 `bench -p 0 -n 0 -pg 512,128 -r 3`。

## 验证边界

真实 checkpoint 的四行 logits 对 FP32 expanded NumPy 参考 NRMSE 为
0.038388，4/4 top-1 一致；该参考与 FP16 absorbed MLA 不保证逐位相同。
独立算子 oracle、真实模型路径间逐位对照和 PPL 分别验证算子算术、状态/
存储等价与模型质量。当前质量证据限于上述短文本和 8 窗口协议；未测
长上下文、完整质量基准或吞吐调优。

## Router TF32 / MLA FP16 迁移，2026-10-22

- Router：公共 raw MM 的 TF32×TF32→FP32，checkpoint BF16 权重展开为 FP32
  容器后直接打包常驻；整批 `[M,1536]×[1536,128]` 只提交一次完整调用。
  CPU 保留 sigmoid、bias、分组 top-k 与贡献权重计算。没有 FP16 输入/输出
  中转；TF32 的 MAC 输入截断语义见 `rk_npu_mm.h`，不等同于完整 FP32 GEMM。
- MLA：q_a/q_b/kv_a/g_proj 使用现有 `FloatBackend::for_decoder` 的 FP16
  NPU GEMM；dense 输出投影仍使用 checkpoint 的 NPU W4。kv_b 吸收权重和
  absorbed cache 布局不变，Norm/RoPE 仍在 CPU。
- TF32 Rust 封装位于 `src/npu/tf32.rs`，调用现有 plan/packed-B/workspace ABI，
  每个 domain 共享相同形状的 workspace，并在 prepare 时保留所需 M buckets。
  采用固定 N 切块、无 split-K；现有浮点 autotuner 只支持 FP16/BF16，TF32
  router 没有伪装成 autotuner 命中。

验证：fmt、全 targets release Clippy `-D warnings` 通过，库测试 92 passed、
22 ignored；另显式运行真实 router 几何的 TF32 板端测试，对照独立 FP64
求和及 TF32 输入截断，覆盖 NPU mask1/3/7 和 M1/17/128/3/1、共享 workspace
切换权重、M1 重放与 FP32 输出。固定真实模型 token 的 chunk1/3/7、head-free
prefill、取消后 reset/replay 均 NRMSE=0。相对旧 CPU 投影版本，4 行 logits
NRMSE=0.04115169、最大绝对误差 0.9116311、top-1 4/4。

未调优新路径同样 8×512 窗口/2040 targets 的 PPL 为 **11.97287615**，相对
旧路径 11.95603409 增加 **0.14087%**。这是短窗口质量检查，不能外推完整基准。
原始报告为 `results/ling-tf32-fixed.json`、`results/ling-tf32-ppl.json`、
`results/ling-tf32-oracle-clock.json` 及相应日志/时钟报告。

调优后的同一 PPL 协议复跑为 **11.97287615**，全部 2040 个逐 token NLL
与未调优新路径完全一致，报告 `results/ling-tf32-ppl-tuned.json`。新路径
resident/streaming 的 628736 个 logits 逐位一致；streaming 的分块、head-free、
取消重放也全部通过，报告 `results/ling-tf32-stream.json`。6 层 MLA 仍全部
为 absorbed，capacity256 的 KV payload 保持 5332992 bytes。
API 两条文本检查通过：天空回答包含“短波长蓝光被大气中的分子散射得更多”，
算术题回答“小明现在有15个苹果。”，报告 `results/ling-tf32-api.json`。

显式调优新增 8 个 FP16 M1/128 profiles：8 次搜索、126 次命中、8 次 miss、
0 fallback、0 weight reload。缓存沿用前述目录的 `floating/` 子目录，
报告 `results/ling-tf32-tune.json`。

同板定频 A/B/B/A：A 为旧 CPU router/MLA 前投影，B 为 TF32 router + FP16
MLA 前投影。两者都加载既有调优缓存。CPU4–7/4 线程 @2.4 GHz，NPU mask7
@1 GHz，DDR @2.112 GHz，resident、context640、ubatch128；连续 pp512→tg128，
相同 MT19937 seed42 的 640 个输入 token，每组一次 warmup、三次测量，
各配置合计六个样本，全部保留。以中位耗时计算吞吐：

| 阶段 | A 中位耗时 ms | B 中位耗时 ms | A tok/s | B tok/s | 吞吐变化 |
|---|---:|---:|---:|---:|---:|
| Prefill512 | 4008.734405 | 3271.236095 | 127.7211 | 156.5158 | +22.5449% |
| Decode128 @512 | 6579.516062 | 6510.250501 | 19.4543 | 19.6613 | +1.0639% |

收益主要在 prefill，decode 改善约 1%。两组 B 均为 374 cache hits、0 miss、
0 在线搜索、0 fallback、0 weight reload；两组 A 均为 86 hits，其他上述计数
同为 0。四组 head-free 检查通过，128-token prompt 后 continuation logits
逐位一致，prefill head_ms=0。完整调用计时排除加载、prepare、tokenization、
sampling 和 stats；decode 开头懒创建的 FP16 M1 workspace 仍在计时内。
峰值 RSS：A 为 4648116–4648452 KiB，B 为 4691808–4695088 KiB。

| 配置/阶段 | 六个原始耗时 ms |
|---|---|
| A prefill | 4010.652137, 4006.816673, 4157.282651, 4000.752557, 4012.736413, 3988.106610 |
| B prefill | 3266.741018, 3269.673763, 3267.708780, 3272.798426, 3275.495501, 3274.339320 |
| A decode | 6591.371874, 6588.670425, 6683.119087, 6570.361699, 6562.852065, 6547.329665 |
| B decode | 6509.867538, 6506.195993, 6505.592235, 6510.633464, 6517.288504, 6511.987980 |

报告为 `results/ling-tf32-abba-{a1,b1,b2,a2}.json`，汇总
`results/ling-tf32-performance-comparison.json`，二进制哈希
`results/ling-tf32-binaries.sha256`。四组时钟报告的 restore_errors 均为空。

迁移后的 CPU 热点、MoE 既有计时细分及原始 trace，见
[当前 CPU profile](LING3TINY_CPU_PROFILE_20261022.md)。

## KDA 数据流与 CPU 路径优化，2026-10-23

当前实现合并 KDA Q/K/V/F/G 投影，使用带 stride 的卷积/门控输入；批量 Norm/门控按 head 并行，M1 卷积直接执行，router 复用选择 scratch，MLA RoPE 共享角度。C++ KDA recurrence、beta 的保护浮点投影及输入值检查保持原有实现。

主板 zt-firefly，CPU4–7 四线程 @2.4 GHz、NPU mask7 @1 GHz、DDR @2.112 GHz；同一 QQQ W4A8 mixed checkpoint，resident、ubatch128、context640，MT19937 seed42 的 640 个固定 token，连续 pp512→tg128。Prefill 不含输出头，decode 含输出头；完整调用排除加载、prepare、tokenization、sampling 和 stats。

A1 在修改前运行并冻结旧二进制；完成修改和调优后依次运行 B1、B2、A2。每组一次 warmup、三次测量，每个版本保留全部六个样本。表中速度按中位耗时计算。

| 阶段 | Baseline ms | 当前 ms | Baseline tok/s | 当前 tok/s | 吞吐变化 |
|---|---:|---:|---:|---:|---:|
| Prefill512 | 3280.167897 | 2932.179850 | 156.0896 | 174.6141 | +11.8679% |
| Decode128 @512 | 6514.575136 | 6054.465383 | 19.6482 | 21.1414 | +7.5995% |

| 配置/阶段 | 六个原始耗时 ms |
|---|---|
| baseline prefill | 3284.293687, 3280.195720, 3284.094768, 3280.140074, 3274.188836, 3274.307254 |
| baseline decode | 6504.767599, 6668.249331, 6496.898338, 6526.861452, 6516.262157, 6512.888115 |
| optimized prefill | 2944.461790, 2956.744024, 2943.320777, 2904.149760, 2921.038923, 2913.459290 |
| optimized decode | 6071.075126, 6061.720096, 6067.831462, 6045.649068, 6047.210670, 6042.430487 |

新增融合形状显式调优 M1/128：12 次搜索、122 次缓存命中、12 次 miss。四组正式测速均为 374 cache hits，miss/search/fallback/weight reload 均为零；head-free 检查通过，prefill head_ms=0。基线与当前版本分别使用独立调优缓存。

workspace payload 从 322047708 增至 398154660 bytes（+72.58 MiB）；峰值 RSS 从 4691648–4691804 KiB 增至 4719272–4719648 KiB，增加约 27 MiB。

验证：fmt、全 targets release Clippy `-D warnings` 通过；完整测试 118 passed、22 ignored。137-token 固定 prompt 加 3 个 continuation 的四行 logits（628736 个 FP32 值），当前 resident/streaming 与基线逐位一致；chunk1/3/7、head-free、取消后 reset/replay 也逐位一致。浮点投影/量化投影两种小型 fixture 对独立 NumPy 参考 NRMSE 分别为 2.50357e-5 / 1.83774e-5，均 4/4 top-1 一致。

WikiText-2 原有 8×512 窗口、2040 targets 协议的 PPL 为 **11.97287615259297**；所有逐 token NLL 与 `results/ling-tf32-ppl-tuned.json` 完全一致。全部 11 个定频 runner 均 exit_code=0、restore_errors=[]。

证据目录：`results/ling-review-20261023/`。`comparison.json` 汇总性能；`{a1,b1,b2,a2}.json` 保留原始测速；`baseline-fixed.json`、`optimized-fixed.json`、`stream.json` 和对应 `.f32` 保留 logits 对照；`ppl.json` 保留逐 token NLL。`baseline.sh`、`after.sh`、`quality.sh` 记录本次完整命令，`sources-before.tar.gz` 和 `changes.patch` 保存本次修改范围。

板端冻结二进制为 `/tmp/ling-review-20261023/{baseline,optimized}`，检查器为同目录 `{baseline-check,optimized-check}`；调优缓存为 `{tune,tune-optimized}`。正式输出也已更新到 `/tmp/ling3tiny-target-20261022/release/`。

优化后的分阶段 trace、MoE 内部明细及后续方向见 [2026-10-23 阶段分析](LING3TINY_TRACE_20261023.md)。

KDA 四行融合与等价输入扫描的性能及验证见 [KDA CPU 优化](LING3TINY_KDA_CPU_20261023.md)：本次配对 prefill 吞吐 +9.77%，decode +0.62%，真实 logits 与 2040 个 NLL 保持一致。

## 算子库更新与 resident MoE 联合调度，2026-10-23

本轮核对提交图并 fetch 远端时，`origin/main` 仍为 `0fd946e`。较新的本地生产线是 `perf/moe-ffn`：`7d653da` 合并 `b917811` 的 MoE/trace 与 `6df8ec8` 的 W4A8 FFN 优化，随后为 `c96930f` 的 FP32 middle 优化和 `7a996a2` 的 routed W4/shared W8 联合调度。本次在 `7a996a2` 上补齐集成所需的 cache-slot API 和 gather 修复，算子库最终指向 `4e57e12b2fb865b65dfe38ed59e10e7a82a5f62e`。

```text
* 4e57e12 (HEAD -> perf/moe-ffn) feat: support cached routed W4 with jointly scheduled shared W8
* 7a996a2 (local-main/perf/moe-ffn) perf: jointly schedule routed W4 and shared W8 FFNs (2026-10-23 03:34 +0800)
* c96930f perf: fuse MoE CPU SwiGLU and Down packing (2026-10-23 02:56 +0800)
*   7d653da Merge latest MoE library changes from Ling worktree
|\
| * b917811 (feat/minicpmv47-moe-w4) perf: expose MoE stage timings in configurable traces
| * 0fd946e (origin/main, origin/feat/minicpmv47-moe-w4, origin/HEAD) feat: add all-W4 MoE and expert slot banks (2026-10-21 10:23 +0800)
| * f7ba205 fix: cap INT4 task width at 8128 (2026-10-21 10:23 +0800)
* | 6df8ec8 perf: fuse W4A8 CPU epilogue and FFN scales (2026-10-22 17:34 +0800)
|/
* 332fa0d fix: honor caller-sized Attention capacity (2026-10-21 04:30 +0800)
```

符合条件的 resident 层使用同一 MoE 调用执行 routed W4 和 shared W8：共享输入量化、调度和输出 combine，删除对应的独立 shared FFN 调用与中间输出加法。条件为 shared gate/up/down 均 W8、shared/routed intermediate 相同；当前真实模型的 23 个 MoE 层全部命中，`joint_moe_shared_layers=23`。protected floating 或其他 shared 格式沿用各自原有投影；没有通过捕获运行错误切换执行方式。streaming 仍按专家分批执行，保持 shared 每个 token 独立计算一次，避免在 assignment 批次中重复贡献。

算子库新增 `rk_npu_moe_w4_slots_create`，将常驻 shared W8 与可导入 routed W4 slots 放入同一权重对象。routed blob 保持 version 2，现有 expert store 可直接使用；shared 不可作为 routed slot 导入/导出。Rust workspace key 区分是否带 shared。新增 trace 阶段 `ling.moe_routed_shared_experts`，汇总脚本同时兼容联合与独立阶段。

新增测试还发现原 gather 固定合并 5 行：H2048/M5 会超过既有 7680-channel copy 限制，返回 PARAM。已按该字节上限合并连续行，并拆分更宽的单行；未放宽底层 copy 的合法范围。H2048 的批量/逐行对照和 H8128 独立 oracle 已通过。

### 三版本配对性能

zt-firefly，CPU4–7 四线程 @2.4 GHz、NPU mask7 @1 GHz、DDR @2.112 GHz，trace 关闭。同一 Ling-3.0-tiny QQQ W4A8 mixed checkpoint、已有 expert store 与 tune cache，resident、context640、ubatch128，MT19937 seed42 的固定 token，连续 pp512→tg128。Prefill 不含输出头；decode 每个 token 含输出头。完整调用排除加载、prepare、tokenization、sampling 与 stats。

A=更新前（outer `518c0d4` + native `b917811`，已包含本轮 KDA 优化）；B=只升级算子库（outer `518c0d4` + native `7a996a2`，未接入 shared）；C=本节联合调度实现。执行顺序 A1→B1→C1→C2→B2→A2，每组一次 warmup、三次测量，每个版本六个样本，全部保留。以中位耗时计算吞吐。

| 版本 | Prefill512 ms | Prefill tok/s | Decode128 @512 ms | Decode tok/s | pp512+tg128 ms |
|---|---:|---:|---:|---:|---:|
| A 更新前 | 2628.367532 | 194.7977 | 5989.699242 | 21.3700 | 8609.993049 |
| B 只升级库 | 2538.412046 | 201.7009 | 5982.728177 | 21.3949 | 8519.931979 |
| C 联合调度接入 | 2523.735343 | 202.8739 | 5656.338061 | 22.6295 | 8176.479006 |

相对 A，C 的 prefill/decode 吞吐分别 **+4.15% / +5.89%**，组合吞吐 +5.30%。相对 B，接入本身为 **+0.58% / +5.77%**，组合吞吐 +4.20%。接入收益主要在 decode；prefill 的增量很小。各组 prefill 有漂移，未删除较慢样本。这是含新 FP32 数值行为的整模型版本对比，不能单凭这些数据归因到某一个内部阶段。

| 版本/阶段 | 六个原始耗时 ms |
|---|---|
| A 更新前 prefill | 2646.148189, 2662.428348, 2660.563409, 2605.203807, 2610.586875, 2586.783664 |
| A 更新前 decode | 5981.710102, 6000.236997, 5994.677178, 5986.924000, 5980.995509, 5992.474485 |
| B 只升级库 prefill | 2576.380952, 2568.894943, 2575.550566, 2506.011415, 2500.789060, 2507.929148 |
| B 只升级库 decode | 6002.038645, 6006.762244, 6010.377205, 5944.592143, 5963.417710, 5953.725798 |
| C 联合调度接入 prefill | 2545.524570, 2567.484133, 2567.309423, 2501.946115, 2477.276936, 2498.449864 |
| C 联合调度接入 decode | 5656.412583, 5656.263539, 5658.872239, 5649.074743, 5660.913348, 5650.967974 |

A/B 每组 374 cache hits，C 每组 350；全部正式测速的 miss、在线搜索、strategy fallback、weight reload 均为 0。六组 head-free continuation 检查逐位一致，prefill head_ms=0。C 的普通 W8 linear 数量从 47 降为 1，shared 权重由 mixed MoE 持有。

B→C 的设备 weight payload 为 3961225216→3960989696 bytes，workspace payload 为 398230104→397002552 bytes；设备 payload 不等于进程 RSS。峰值 RSS 为 4719432–4719476→4737036–4737216 KiB（约增加 17 MiB），测后 RSS 为 4500956–4501260→4505516–4506512 KiB。模型加载约 13.1–13.3 秒，不计入上表。

### 验证结果与数值边界

fmt、all-target release Clippy `-D warnings`、119 项常规测试通过，22 项 ignored。新增 mixed slot 测试对照独立打包权重，覆盖 slot 置换、重复路由、非单位 route 总和、M1/5/128、1/2/3 NPU 核，结果逐位相同，并对照单行调用；H8128 的独立整数/浮点 oracle 通过。W8 normal/native、mixed K512+64、NPU-middle 回归及公共 C header 编译也通过。

**模型数值一致性检查未通过，不能称为无损替换。** 137 个固定 token 加 3 个 continuation 的 4 行 logits（628736 个 FP32 值）：C 相对 A 的 NRMSE 为 0.02220324，相对 B 为 0.02159579，top-1 均 4/4 相同；C 的 chunk1/3/7 相对 chunk128 的 NRMSE 为 0.02331670。head-free 和取消后 reset/replay 仍逐位一致。streaming 的基准 logits 与 B 逐位一致，分块误差仍为 0.02174686，因此当前 resident/streaming 不再逐位一致。

这些报告保留 `--max-nrmse 0`，`fixed` 和 `stream` 命令均以 1 退出；分块误差也超过工具默认 0.01 门槛，未提高阈值掩盖失败。库的 degree-7 FP32 SwiGLU 与原 CPU exp 数值规则不同；上层稠密 FFN 仍按 M≥32 选择 fused 路径，小分块使用原激活。该路径差异与分块现象吻合，尚未逐层隔离所有误差来源。联合 shared W8 还采用库的 middle/量化规则，不能假设与原独立 W8 FFN 逐位相同。

同一 WikiText-2 8×512 窗口、score_start257、2040 targets 的 PPL：

| 版本 | PPL |
|---|---:|
| A 更新前 | 11.972876152593 |
| B 只升级库 | 11.963005828261 |
| C 联合调度接入 | 11.975940817067 |

C 相对 A 为 +0.0256%，相对 B 为 +0.1081%。逐 token NLL 不再相同；这只是短窗口质量检查，不能外推完整基准或称为质量等价。后续若要求跨分块和存储模式一致，需要统一各路径的激活与量化数值规则。

所有本轮时钟 runner 的 restore_errors 均为空。最终 native 回归、六组测速和 PPL 的退出码均为 0；模型固定输入的严格比较失败如上，原始失败结果保留。

最终证据为 `results/ling-moe-joint-20261023/`：`comparison.json`、`{a1,b1,c1,c2,b2,a2}.json`、`fixed.json`、`stream.json`、`ppl.json` 及 `.f32`、clock/log 文件。`run.sh`、`paired.sh`、`extra-native-after.sh` 与 `summarize.py` 保留命令和汇总方式。最终 native 回归日志为 `native.log`、`native-wide.log`、`native-cpu.log`、`native-npu.log`、`native-split.log`。中间版本升级结果分别保留在 `results/ling-native-update-20261023/`（c96930f）和 `results/ling-native-final-20261023/`（7a996a2，未接入）。

冻结二进制 SHA-256：

```text
cec5dda606e3faa35a114390aa2278a278a88f703d84a0d61382d9326948fb42  /tmp/ling-kda-optimized
05d58841704d13fe278677faba148d72ff5293e419fa1a545723c14ff8a2c497  /tmp/ling-native-7a996a2
ca5b936725e8471839cd6e37bee17984aa2ca63d079c843597ff6d3e582fea37  /tmp/ling-moe-joint
```

## Chat / serve 采样与默认参数（2026-10-23）

模型裁剪和交付命名已签名提交为 `540ce85`。采样器对有限 top-k 使用部分选择，只排序保留的 k 个候选；temperature0 的贪心路径合并为一次扫描，保持相同分数时优先选择较小 token ID。CLI、HTTP 和 MTP 测速工具的默认温度统一为 1，CLI `--n-predict` 与 HTTP 输出上限默认 2048。显式 temperature0 的贪心行为保留。

主板 zt-firefly、Ling QQQ W4A8 mixed checkpoint，CPU4–7/4线程 @2.4 GHz，NPU mask7 @1 GHz，DDR @2.112 GHz，context4096、ubatch128、resident experts，trace关闭。配对项目复用原 expert store 和 `tune-optimized` 缓存，seed42，关闭 thinking，同一48-token聊天 prompt生成128个token；decode分子为首token之后的127个，含模型和采样/文本处理。bench 使用相同深度48、128个固定随机token，1次warmup、2次采样，不含采样。TUI 通过真实PTY运行；HTTP同时测试非流式和SSE。

固定温度0.6、top_k20、top_p0.95，排除默认值变化的影响：

| 路径 | 优化前 decode tok/s | 优化后 decode tok/s |
|---|---:|---:|
| bench | 23.0978 | 23.0712 |
| CLI run | 19.2536 | 22.4228 |
| 终端 chat | 19.7300 | 22.3884 |
| serve 非流式 | 19.8118 | 22.4340 |
| serve SSE | 19.7012 | 22.4065 |

bench 原始两次速度为 23.1092/23.0864 → 23.0876/23.0548 tok/s；其他行各为一次128-token冒烟测速。这里未复现10 tok/s，测得的默认chat差距主要来自原采样器对157184个logits全量排序后再保留20个的额外工作；终端刷新与HTTP/SSE没有造成速度减半。单位测试对完整排序参考逐位核对概率及固定seed抽样，覆盖同分候选、惩罚项、k=0/1/20/词表边界。配对CLI/HTTP文本、同参数的新版CLI/API/SSE文本均一致。

新默认温度1的结果：

| 路径 | decode tok/s | 配置 |
|---|---:|---|
| chat，128个输出token | 22.3100 | 原store/cache，top_k20、top_p0.95 |
| chat，未指定store/cache/温度/生成上限 | 20.4923 | 直接加载权重，默认2048上限，实际输出43个token |
| serve，非流式 / SSE | 18.3591 / 18.9868 | 默认top_k0、top_p1，完整词表采样，128个输出token |
| serve，非流式 / SSE | 22.4541 / 22.3853 | top_k20、top_p0.95，128个输出token |
| serve，未指定温度和生成上限 | 18.4570 | 默认2048上限，实际输出62个token |

未指定store/cache的chat与未指定HTTP生成上限的请求使用39-token短prompt；其余新默认项目沿用48-token prompt。HTTP默认完整词表分布比有限top-k耗时更高；对比chat时需显式使用相同top-k/top-p。默认加载路径的cache hits/misses为0/164，使用已有缓存的项目为110/54，所有项目 `tune_calls=0`。服务正常退出，所有频率设置已恢复。

默认完整版本128项自动测试通过、22项ignored；Ling专用版本93项通过、11项ignored，二者all-targets release Clippy `-D warnings`通过。新产物为 `dist/ling3tiny-named/ling3tiny`（8,399,672字节），仅注册Ling3tiny；原始日志、PTY输出、统计与配对汇总位于 `results/chat-perf-20261023/`，最终汇总为 `comparison.json`。

### 最新专用版 prefill512 复测

同一 `ling3tiny-named/ling3tiny`、模型与定频配置，context640、ubatch128、resident、已有store/cache，`bench -p 512 -n 0 -r 3`：1次warmup后，head-free prefill512平均 **204.4029 ± 1.0859 tok/s**，三次分别为203.3101、204.4168、205.4817 tok/s，耗时2518.320142、2504.686015、2491.705814 ms。计时不含输出头、加载和prepare。Head-free continuation logits逐位一致，23层joint MoE启用，cache hits182、miss0、tune_calls0、strategy fallback0，频率设置恢复正常。原始记录为 `results/chat-perf-20261023/latest-pp512.json` 与对应clock/log文件。

### Release 热路径清理

KDA的Q/K/V、decay/beta值域扫描，Ling与其他解码器的logits/router扫描，以及采样器逐元素finite检查改为debug断言，release不执行这些数值诊断。加载阶段的权重/配置检查、请求参数与token范围检查、FFI缓冲区尺寸/线程数检查、取消和设备错误处理保留。计算公式及数值精度不变。

输出头直接调用可惰性准备workspace的Linear，移除每步重复的全backend `prepare`；浮点Linear只在出错时格式化M/K/N信息。生成循环复用logits和事件缓冲区，以切片借用speculative输出并用栈数组保存单token；Ling复用last_hidden缓冲区，MLA移除零缓冲区的无用clone。浮点投影的重复尺寸检查由实际FloatLinear入口统一完成。

与已包含采样优化及1/2048默认值的冻结版本比较，继续使用上面的主板/模型/频率/store/cache配置，context640、ubatch128，连续pp512→tg128，ABBA顺序，每组1次warmup和3次采样，每个版本共6个样本。下表用中位耗时换算速度，排除加载、prepare、tokenization和sampling，prefill不含输出头、decode含输出头。

| 阶段 | 清理前中位 ms / tok/s | 清理后中位 ms / tok/s |
|---|---:|---:|
| Prefill512 | 2547.9560 / 200.9454 | 2538.0804 / 201.7273 |
| Decode128 @512 | 5660.6999 / 22.6120 | 5611.7375 / 22.8093 |

性能基本稳定，prefill/decode分别改善约0.39%/0.87%。137-token输入加3个continuation的628736个FP32 logits逐字节一致；四组head-free检查均逐位一致，在线调优、cache miss和strategy fallback均为0。默认完整版本128项、Ling专用版93项自动测试通过，all-targets release Clippy `-D warnings`通过。

最终release的7项HTTP短问答、多轮记忆/更名与SSE测试通过，同题CLI/HTTP/SSE输出一致；真实终端chat在温度1、top_k20、seed42、128-token测试中为22.2355 tok/s。频率设置恢复正常，`tune_calls=0`。`dist/ling3tiny-named/ling3tiny`已更新为清理版，8,375,096字节；原始记录与六样本汇总位于 `results/hotpath-cleanup-20261023/`，最终汇总为 `comparison.json`。

## 2026-10-24：SSM 前缀检查点

`feat/ling3tiny` 已通过 merge commit `33503e5` 合并本地 main 至 `7d72396`，包含 main 的跨请求前缀缓存；该 main 包含拉取时的远端 main `8b186ef`。算子库先更新至远端 main `4f36ab6`，接口提交 `860c369` 在其上增加公共 `rk_npu_mla_f16_cache_truncate`，复用现有 Attention cache 的截断实现，保留 absorbed KV 的分配及有效前缀。

`src/model/prefix_cache.rs` 与 `prefix_cache/policy.rs` 供 Qwen3.5 和 Ling 共用：按 user 消息边界和 prompt 尾块安排检查点，复用同位置快照，沿用 main 的当前请求保护、间隔压缩和 FIFO 容量限制。Ling 每个快照只保存各 KDA 层的 FP32 recurrence 和 Q/K/V 三路因果卷积历史。恢复时选择不超过共享前缀长度的最近检查点，恢复上述状态、截断所有 MLA KV，清除旧 logits/hidden，并移除未来分支的检查点；没有检查点时由 engine reset/recompute。

`prepare_prepared` 生成的同一份计划用于准备所有实际 batch 行数和执行前向。取消或执行错误会重置模型和检查点，避免复用部分更新的状态。`stats` 暴露 `prefix_checkpoint_positions/count/bytes/saves/hits/save_ms`、容量和最小间距。

真实 Bailing V3 模板使用 `<role>HUMAN</role>`；`TextCodec` 已支持这个用户边界，排除 OBSERVATION 工具消息以及以 `<tool_response>` 开头的旧式用户工具响应，同时保留 ChatML 的识别规则。

沿用 main 的参数：`--ctx-checkpoints`（默认 32，0 或 -1 关闭）、`--checkpoint-min-step`（默认 8192）；别名为 `--swa-checkpoints`、`-ctxcp` 和 `-cms`。服务级 `--no-cache-prompt` 和请求级 `cache_prompt` 控制前缀复用。快照占用 CPU 内存，由数量限制约束；本次没有进行吞吐或 PPL 测量。

未完成的 UBTree 改动已独立 stash，包含未跟踪的源码、fixture 和验证工具。后续 agent 使用 `git stash apply ac6ce1cb7318e48a9ecfcfbad8d9dc7f97f0c50e` 继续，需要处理与本次缓存适配的重叠。合并 main 前的第一份保险备份为 `cd50dbd186ccebd30c6a5a6024cc4c2e7359ca04`；本次不继续 UBTree。

验证记录位于 `results/ling-ssm-20261024/`，最终汇总为 `validation-summary.json`。主板默认 154 项、Ling 专用 117 项自动测试及两种构建的 all-targets Release Clippy `-D warnings` 通过。板端执行时的 333 份源码/fixture 与主机哈希一致，快照见 `board-tested-source-manifest.json`；测试二进制已冻结并校验备份哈希。

MLA 公共接口的 expanded/absorbed 两种模式在 mask1/3/7 下通过：保留长度 0/1/17/31/32/33/63/64/65/129，截断后追加与全新加载前缀输出逐位一致，分配不变，非法长度被拒绝。

最新真实权重 resident 和 streaming 两种模式均通过：600-token 固定输入、3 个 continuation、从 596 恢复，以及在 500 处分叉后从 468 重算，完整 logits 逐位一致，KDA recurrence 与三路卷积历史比较相同。同时覆盖同位置快照复用、容量限制、miss/reset、取消、过期计划及禁用缓存。真实 Bailing 模板的两条用户消息与工具消息边界断言也通过。Streaming 重建后的专家记录位于主板 `/home/firefly/ling-ssm-20261024/expert-store`。

验证期间主板曾出现 `switch iommu domain time out ... id: 15`，原有 dense GEMM canary 也超时。按用户要求向 `/sys/kernel/debug/rknpu/reset` 写入 `1\n`，内核确认执行 soft reset，但未恢复；用户随后手动重启主板。重启后 canary、mask1/3/7 MLA 回归和两种专家模式的整模型回归全部通过。六份 `reboot*-clock.json` 均为 `exit_code=0`、`restore_errors=[]`，NPU 最终负载读数为 0%。故障前和 debugfs reset 的记录保留作历史证据，最终通过记录为 `board-artifacts/results/ssm/reboot-*.log`。

最后修正了纯 dense 构建的兼容边界：未启用循环模型时不编译检查点计划字段，并始终构建所有模型共用的 CPU SwiGLU，避免缺少 `rknpu2_cpu_swiglu` 链接符号。此三文件修改不改变默认/Ling 的执行分支；最终主机默认 154 项、Ling 117 项、Qwen3-only 97 项测试均通过。CI 的八种单模型/组合 feature 配置均通过主机 Release Clippy，命令仅额外允许既有的非 AArch64 `unused_mut` 警告；严格 ARM 检查为前述默认/Ling 构建。最终源码哈希及与板端冻结版本的三个 feature/build 文件差异见 `final-source-manifest.json`。

恢复 UBTree stash 后，还需让前缀保存/恢复避开验证事务，并同步 draft/Features 状态及 MLA 的 UBTree host KV mirror；本次验证的 Ling speculation 配置为 None。

## 2026-10-25：SSM review 修正

Workspace 准备仅保留实际 prefill 计划的行数，并为量化/TF32 backend 保留 decode 的 1 行。FP16 backend 每个 K/N 只保留一个行数，因此预热计划的首个实际 batch：例如首批 20 行则准备 20 行，缓存恢复仅需 4 行则准备 4 行；后续 batch 和 decode 按实际行数切换。

卷积历史新增 `ConvState::restore()`：校验 channels/kernel 后使用 `copy_from_slice()` 原地恢复；Ling 检查点恢复改用该方法。主机隔离测量 18 层×3 路卷积：派生 `clone_from()` 为 54 次分配、54 次释放并替换 54 个 buffer；新方法分配、释放、realloc 和 buffer 替换均为 0，history 内容逐位相同。探针调用实际 crate 的公开 API，源码、构建日志和结果保存在 `results/ling-ssm-review-20261025/`，未加入生产目标。

新增单元回归覆盖 history 地址/容量保持、kernel=1、2048 通道、恢复后的连续计算，以及同长度异维度快照拒绝。板端回归增加计划首批 20 行与短重放 4 行的检查，直接核对每个 FP16 workspace 的逻辑行数与句柄，要求执行后保持同一 workspace。总 backend fallback 计数包含量化投影首次建策略，不用它代替 FP16 workspace 的观测。

验证：最终主机默认 155 项、板端 Ling-only Release 118 项自动测试及严格 Release Clippy 通过；resident/streaming 两种真实权重回归均通过 20/4 行 workspace 行数与句柄保持断言，完整 logits 逐位一致。两次 runner 均 exit_code=0、restore_errors=[]。观测方法限定为 cfg(test)，当前板端编译源码与主机 333 份源码/fixture 哈希一致。最终汇总为 `results/ling-ssm-review-20261025/validation-summary.json`。

## 2026-10-28：统一工作区与模型级 scratch

Ling 与其他 decoder 共用实际行数规划及 prefill/decode 阶段切换，进入 decode 时释放多行量化和 TF32 工作区。mixer、FFN、router 与输出头的临时 host 缓冲按模型复用；KDA、卷积和 MLA KV 继续逐层持有，MLA 的 FP16 打包缓冲也复用容量。`cpu_scratch_bytes` 单独记录 host 临时缓冲。

Ling 当前不支持 `--expert-min-weight` 非零值，加载时直接报错；默认值 0 保持原计算。resident/streaming 的真实模型回归均通过完整 logits/state 和 20/4 行 FP16 workspace 句柄检查。修改与验证口径见[通用优化补齐记录](MODEL_RUNTIME_OPTIMIZATIONS_20261028.md)。
