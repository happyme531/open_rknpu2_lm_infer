# 推理框架与算子库数值边界扫描

日期：2026-10-20，Asia/Shanghai。

最值得先试的是框架的 `ubatch<=256`、MTP 的 `draft<=7 / verify<=8`，以及 FP16 decoder 投影在 `K>4096` 时强制拆分。这三处直接限制当前推理路径，而底层实现已有更一般的行数或 K 分块处理。GDN 的窗口上限、FlatQuant 的预处理上限也值得逐项放开。以下结论是源码分析得到的试验候选，不代表越界形状已经通过板端验证。

## 扫描范围

- 推理框架：`src/`、`native/cpu_kernels/`，主仓 HEAD 为 `d1be888d640ea256d23ada6aef3fba9387c2eb8f`。
- 实际链接的算子库：`native/rk_npu_matmul/{src,include,tests,docs}`，子模块 HEAD 为 `2b3b438cdceaf8fb8c62f73f09f0a4fb38b51537`。
- 对照目录：`../lowlevel/rk_npu_matmul/`。其中 FP16 fused clamp、FlatQuant 上限、Attention 容量上限和 INT8 M tile 上限存在相同实现；该 checkout 没有当前子模块的 `rk_npu_gdn.cpp`。
- 主仓和两个算子库 checkout 均存在已有修改。MoE 条目按当前工作区源码描述，不把未提交实现当成固定版本。

示例提交已移除的小 batch 补到 8 行和视觉 `MAX_PATCHES=1024` 不重复列为问题。扫描重点是逻辑 shape 的任意上下限、强制拆分与寄存器 stride 的饱和截断。

## 优先试删的限制

### 1 全局 ubatch 和 Linear 行数最多 256

位置：`src/npu.rs:52`、`src/npu.rs:154`。

`configure_cpu()` 把 `options.chunk` 限制到 `1..=256`，`bucket()` 又把每次 Linear 的逻辑行数限制到同一区间。后者在 128 行以上已经使用 `div_ceil(64)*64`，没有只能表示 256 行的数组或枚举。

底层 W4A8 的 `backend_config()` 接受总 `M<=INT_MAX/2`，随后把逻辑行数变成 INT4 的 `2*M`，按 `m_tile` 拆任务；`64/256` 限制的是每个 tile 的逻辑行数，而非整次调用的 M。W8 的 `query_i8_kn_memory()` 同样按 K/N 波次和 M tile 规划。Attention 在框架中另有 query 分块，不要求整个 ubatch 与单次 query 上限相同。

建议一并去掉两处固定 256 上限，保留正数、ABI 可表示范围、底层 query 和内存检查。保留现有 bucket 规则作为第一步，避免同时改变缓存策略和行数限制。

验证首先覆盖 `M=257/320/511/512` 的 W4/W8 complete call，再跑真实模型 prefill。W8 可选 NPU reduction 还有单独的 `M<=128` 条件，不能把总 M 放开解释成所有执行策略都能使用。GDN 的 2048 上限是后续独立边界。

### 2 MTP 最多 7 个 draft 和 8 行 verify

位置：`src/cli.rs:139`、`src/model.rs:51`、`src/model/qwen35/runtime.rs:373`、`src/model/qwen35/transaction.rs:30`、`src/model/qwen35/load.rs:186`。

CLI、ModelOptions、运行时分别拒绝 `draft>7`，target verification 拒绝 `rows>8`，加载 GDN 状态时又固定传入 `LayerState::new(..., 8)`。这五处构成同一个限制。

`RoundPlan`、MTP 的递归 draft、verification logits 和更新日志均按实际行数处理；`LayerState::new()` 本身接受通用 `max_verify_rows`。算法需要的是 `verify_rows=draft+1`、窗口不超过 ubatch/context/output budget，源码没有要求窗口必须最多 8 行。

建议去掉固定 7/8，让 GDN verification 容量跟随实际窗口。不能只删 CLI 检查：8 行状态容量必须同步调整；`enable_mtp()` 后续改变 draft 长度时也要满足容量要求。

验证 `draft=8/15/31`，逐项覆盖全接受、零接受、部分接受、abort、EOS/stop 提前截断。尤其比较每个接受前缀后的 GDN 状态、KV 和后续 logits 与普通 AR 是否一致。更长 draft 是否更快取决于接受率，不由放开限制本身保证。

### 3 FP16 投影在 K 大于 4096 时强制由框架拆 K

位置：`src/model/projection.rs:45`。

`Linear::floating()` 遇到 `K>4096` 就创建多个浮点 Linear；每次调用重新收集各段输入、提交各段计算，再把已经 FP16 舍入的部分输出加到 FP32。

该路径使用 `FloatBackend::for_decoder()`，dtype 是 FP16。底层明确拒绝 `part_k>4096` 的条件只针对 TF32，位于 `native/rk_npu_matmul/src/rk_npu_mm.cpp:159` 和 `src/rk_npu_matmul_f16.cpp:619`。FP16 实现还明确讨论 K=5120 的 CBUF 切块，`tests/test_batch_f16.cpp:159` 有 `M26/N64/K5120` 的 native-A 用例。库内浮点 autotuner 本身也会搜索 K 分块。

建议首先让完整 FP16 K 进入库内 planner/autotuner，比较库选择的 full-K/K-tiled 路径与现有框架 SplitK。它有机会同时简化代码并减少输入复制，但删除后会改变部分和的舍入位置，需要对真实 MTP 的 K=5120 和 FFN down 投影分别验证输出及最终 logits。

这里不连带删除 decoder 的 `N tile<=4096` 规则；后者与 FP16/INT8 混跑记录有关。

### 4 GDN 固定窗口最多 2048 tokens

位置：`src/npu/gated_delta.rs:43`、`native/rk_npu_matmul/src/rk_npu_gdn.cpp:46`。

Rust wrapper 与 C ABI 都硬拒绝 `T>2048`。实际 `Prefill` 使用按 `blocks=ceil(T/chunk)` 分配的 buffer、vector 和任务链，没有固定长度为 2048 的 token 数组。C ABI 在同一个 validate 中还检查真实任务数：

```text
ceil(T/chunk_size) * 16 * products <= 4095
products = 1 (TF32) 或 5 (compensated TF32)
```

因此默认 TF32、chunk64、T4096 只对应 1024 个 Gram 任务；单从这个任务数条件推不出 2048 上限。预测任务按 block 分链。

建议去掉两层固定上限，保留任务数和分配失败检查。先验证 `T=2049/3072/4096` 的 projected、strided 和归一化输入接口，比较输出、最终 state、history。Compensated 模式受任务数约束更紧，不能沿用 TF32 的窗口判断。

全模型实验要先处理第 1 项，否则 `ubatch<=256` 会先拒绝输入。更长窗口也会增加 Gram buffer 和预构建预测链的内存。

### 5 FlatQuant transform batch 最多 256 和 ubatch 最多 32

位置：`native/rk_npu_matmul/src/rk_npu_flatquant.cpp:84`、`src/rk_npu_w4a4_linear.cpp:54`。

两层 query 重复拒绝 `transform_batch>256`、`transform_ubatch>32`。但 preprocessing batch 的实际大小已经取 `min(M, requested_batch)`；group、双缓冲、命令和任务都按所选大小构建，单次硬件 M 又由 `tile1()/tile2()` 按四个 CBUF bank 拆分。源码中没有与这两个常量对应的固定数组。

建议先放开这两个参数上限，保留非负数、任务/寄存器尺寸及内存约束。测试 `batch=257/512`、`ubatch=33/64` 和不整除尾块，并对比完整 transform→quantize→INT4 GEMM→dequant 调用。

它们控制预处理的资源和流水方式。允许更大值并不意味着默认值应调大；当前默认 batch64 的优劣要由 complete-call 数据决定。

## 需要更完整验证的限制

### 6 框架和 Attention 同时限制 context 为 32768

位置：`src/model/metadata.rs:14`、`src/npu.rs:56`、`src/model/dense.rs:177`、`src/model/qwen35/config.rs:7,188`，以及算子库 `src/rk_npu_attention_f16_cache.cpp:34`、`src/rk_npu_mla_f16.cpp:18`。

metadata 把 checkpoint 的 `max_position_embeddings` 静默压到 32768，框架和底层随后重复检查。底层的 KV buffer 按 capacity 分配，QK 已按 4096 channels 拆块、PV 按 kv_tile 拆块，并按实际形状计算任务数；这些处理没有直接给出总容量必须是 32768 的公式。

这是值得验证的历史支持包络，优先级低于前三项。应同时处理各层检查，保留 checkpoint context、任务数、DMA/寄存器和内存限制。先在小 head 数、低内存配置验证 `32769→align32=32800`、49152，然后测真实模型。不能只删 metadata 的 min，否则默认配置会在更深层失败。

### 7 FP16 fused MUL 和 ADD 遗留 group cap 13

位置：`native/rk_npu_matmul/src/rk_npu_internal.h:125`、`src/rk_npu_matmul_f16.cpp:256`。

legacy fused 分支仍然使用：

```cpp
line_stride = 4 * min(ceil_div(eff_k, 32), 13);
notch_val   = 8 * min(align_out / 32, 13) - 1;
```

这会在相应宽度超过 416 时停止增长。pack/layout 使用实际 aligned K/N，因而值得验证饱和后的寄存器步长是否仍与实际 buffer 一致。

INT8 路径已经移除了同名 cap，并在 `src/rk_npu_matmul_i8.cpp:182` 明确记录旧 clamp 在 N>416 的多行 tile 中造成错误。这个记录不能直接证明 FP16 fused 也错，但为专项验证提供了具体依据。当前 fused 设备回归 `tests/test_compress.cpp:274` 使用的是 M1/N32/K64，未覆盖这条边界。

建议隔离验证 fused API 的 M1/2/8，N/K416/448/512，再按实际 layout 推导 stride/notch。它还使用特殊的 C0 half-surface 布局，不能直接复制 plain matmul 公式。当前 Rust 推理路径没有调用 legacy fused MUL/ADD，预期收益是修正公共算子支持范围。

## 可继续研究但不优先试删

- **FlatQuant 的 L/R≤128**：`src/rk_npu_flatquant.cpp:38`。权重和 CPU layout 均按 L/R 处理，128 未给出明确硬件推导。不过放开会改变两个 GEMM 的权重 bank、fold 和 DMA 几何；应先测 144/160，再决定怎样用实际 CBUF/寄存器条件替代。
- **MoE 的 top_k≤8 和 max_rows≤256**：`src/rk_npu_moe_w8.cpp:306`。assignment、buffer 和分组按 `rows*(top_k+shared)` 分配，值得检查是否只是既有模型/回归范围。扩大前需重新核对 gather 的 4095-task 边界、命令容量和内存。该文件有正在进行的未提交修改。
- **INT8 单 tile 的 M≤128**：`src/rk_npu_matmul_i8.cpp:32,132`。代码明确标成保守且待调优，已有解释；适合在小 K、既定 CBUF budget 内另做 192/256 行实验，不应把大 K 的特殊配方一并撤掉。
- **浮点 fallback 的 physical M=128**：`src/rk_npu_mm_tune.cpp:148`。即使实际 M 很小也按 128 行准备；decoder 已自行缩小。可比较库层改成 `min(M,128)`，并核对权重复用、layout 对齐和小 batch 输出。它是默认策略问题，不是输入上限。
- **FP16 autotune 的 m_tile≥M/8**：`src/rk_npu_mm_tune.cpp:191`。这是控制搜索量的剪枝，没有拒绝应用输入；只有当实测最优 tile 可能落在被排除范围时才值得扩大搜索。
- **实验工具的 batch≤512、共同 rows≤4096**：`src/npu/ubatch_probe.rs:16,29`。只影响 ignored probe，避免把工具的资源范围当成生产后端限制。

## 有实际依据的边界

| 边界 | 保留依据 |
| --- | --- |
| TF32 每个 K 分片≤4096 | `src/rk_npu_mm.cpp:157`、`docs/MM.md:126` 记录 K5120 未拆分发生 submit timeout。 |
| Attention QK tile4096 和 decoder 宽 N 防线 | `src/rk_npu_attention_f16_internal.h:19`、`docs/ATTENTION.md:270` 记录 native FP16 N8192 本次输出正确仍可让后续 INT8 超时。decoder 的规则沿用这一混跑风险背景，若要放开，必须验证后继 INT8。 |
| 4095 个任务 | PC/提交链边界；GDN 和 Attention 都按实际任务数检查。不能用删除固定 shape 上限替代它。 |
| Attention Hkv≤32 | `value_absmax[32]`、`std::array<uint16_t,32>` 及每个 head 的 uint32 位图。需要修改存储/位图才能扩大。 |
| Attention query≤128 | 有按 128 行处理的固定临时数组和分块逻辑；框架已对更长 query 做外部分块。 |
| MoE intermediate≤512 | 当前 CPU 路径仍有 `gate/up/hidden/hidden_q[512]`，NPU middle recipe 也固定 I512。只删校验会越界。 |
| 浮点 MM 维度≤65504 | `src/rk_npu_mm.cpp:117` 明确关联 legacy 16-bit 字段和 signed-int 任务运算；需要逐字段重审，不是把数改大。 |
| Panel 输出 notch≤8191 | FP16 `valid_config()` 直接检查实际 Panel 输出 pitch。GELU 入口的 N≤32768 不能代表所有 N 都能执行，仍要满足底层更紧的 Panel8 条件。 |
| W4A8 NPU reduce 的 storage≤4096 | `src/rk_npu_w4a8_npu_reduce.cpp:103` 记录 8192 pitch 即使无 overlap 仍失败。 |
| 量化范围、概率范围和 FP16 有限值 | INT4 [-8,7]、对称 INT8 [-127,127]、beta [0,1]、FP16 幅度65504、norm epsilon 等有格式或数学依据。 |

HTTP 图片数量/编码字节/解码内存、stop 数量，以及 sampler 的 temperature≤2、presence/frequency penalty 范围属于接口或资源策略。temperature 和 penalty 的固定范围不是 softmax 的数学上限，但 HTTP parser 也明确使用相同区间；如需扩大 CLI 的采样自由度，可单独调整，优先级低于算子 shape 限制。概率本身的 [0,1] 区间应保留。

## 建议试验顺序与范围

先独立处理全局 ubatch256、MTP7/8、框架 FP16 SplitK4096；再处理 GDN2048 和 FlatQuant 的 batch/ubatch。context32768 与 legacy fused stride 分别单独验证。

每个改动先做超出旧边界的完整算子调用，对比输出及会受影响的状态，再运行相应真实模型路径。浮点或状态变化要比较最终 logits；性能统计包含 pack、sync、submit、collect。固定频率下做配对测量，并检查缓存命中、调优调用和 fallback，避免把策略重选或运行时搜索算成放开限制的收益。

本次只增加扫描报告，没有修改运行时代码或执行板端试删实验。没有声称任何候选已经通过更大形状的正确性或性能验证。
