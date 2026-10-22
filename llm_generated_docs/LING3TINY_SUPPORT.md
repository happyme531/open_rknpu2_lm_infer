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
