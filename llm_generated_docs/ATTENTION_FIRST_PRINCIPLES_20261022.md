# RK3588 Attention 瓶颈与结构优化实验

当前 FP16 Attention 的主要问题随场景变化。单 token GQA decode 的算术强度很低，提升 MAC 利用率不是合理的独立目标；prefill 则被 score 张量的反复处理、CPU/NPU 同步、分段图维护和串行收尾共同限制。

本轮单算子对照中，实验组合使 GQA prefill 的完整调用降低 6%–15%，视觉 MHA 降低 36%–37%，真实 Qwen3.5 C7 四行验证降低 11.7%，decode 增加 1.1%。这支持按场景优化，不支持统一开启全部变体。若要进一步追求约两倍的算子加速，优先研究移除整张 score 的 CPU 往返；继续调整 LUT 表或少量任务参数没有足够的收益空间。

## 测量范围

- 基线：当前 native revision `6df8ec899f5e8b7757b73911da25baff9178ef9e` 的原始静态库，GCC 12.2、`-O2`、ARMv8.2 FP16。正式基线没有阶段插桩。
- 板子：备用板 1，`备用板 1`，CPU4–7 固定 2256 MHz，NPU 三核固定 1 GHz。该板没有 DMC devfreq，DDR 未受控。
- 计时：公共 `rk_npu_attention_f16_run` 的完整热态调用，包含内部 query 分段、构图更新、pack、QK、softmax/LUT、同步、PV 和 finish；不含初始 KV load、进程启动和首次预热。阶段诊断另行采样。
- 每个几何三轮 ABBA，每进程预热四次、正式十二次，共 72 个基线和 72 个实验样本；九种几何合计 1296 个正式样本。表中每种方法取六个进程内 median 的均值。
- `Q128/KV4096` 表示已有 KV 上最后 128 个 query 的调用，不是完整 4096-token prefill，也不是整模型 TTFT。
- 八种数据使用相同种子 `20261022`、标准差 0.5 的 FP16 Q/K/V，匹配模型几何；另有真实 C7 首个 FullAttention 层的 `prefix1021/R4` 捕获数据。真实数据未用合成值替换。
- 每进程检查完整输出重复性、guard 和有限性，并以独立 FP32 stable-softmax 抽查 16 个分散的 head/query 行。所有进程的完整输出文件与同几何基线逐字节比较。

时钟 runner 全部正常退出，`restore_errors=[]`。正式对照期间未编译或运行其他 NPU 测试。原始样本及配置见 [final/summary.json](evidence/attention_firstprinciples_20261022/final/summary.json) 和 [final-clocks.json](evidence/attention_firstprinciples_20261022/final-clocks.json)。

## 完整调用结果

| 场景 | Hq/Hkv，D | Q / KV | 原库 ms | 实验 ms | 延迟下降 | 三轮配对下降范围 |
|---|---|---|---:|---:|---:|---:|
| Qwen3 prefill chunk | 32/8，128 | 128 / 512 | 5.906 | 5.555 | 5.95% | 4.6%–6.7% |
| Qwen3 prefill chunk | 32/8，128 | 128 / 4096 | 28.960 | 24.960 | 13.81% | 3.9%–20.4% |
| Qwen3 decode | 32/8，128 | 1 / 4096 | 0.862 | 0.872 | −1.09% | −1.2%–−0.9% |
| Qwen3.5 prefill chunk | 16/4，256 | 128 / 512 | 4.599 | 3.910 | 15.00% | 11.9%–18.7% |
| Qwen3.5 prefill chunk | 16/4，256 | 128 / 4096 | 18.650 | 16.651 | 10.72% | 7.7%–14.7% |
| Qwen3.5 verify | 16/4，256 | 4 / 4096 | 1.357 | 1.292 | 4.77% | 4.6%–4.9% |
| ViT MHA | 16/16，64 | 128 / 256 | 1.458 | 0.913 | 37.35% | 32.3%–40.7% |
| ViT MHA | 16/16，64 | 128 / 1024 | 3.717 | 2.393 | 35.61% | 32.1%–39.4% |
| 真实 C7 verify | 16/4，256 | 4 / 1025 | 0.568 | 0.501 | 11.72% | 10.4%–13.3% |

全部完整输出逐字节一致。合成数据抽查的 relative L2 不超过 0.0444%；真实 C7 为 0.0835%，max abs 0.003624，实验与原库的误差相同。配对范围是三轮实际结果，不是置信区间。长上下文 CPU 阶段有明显波动，应以配对结果而非单次扫描决定是否合入。

实验组合包含四项：

1. 稳定、无 mask 的 MHA 将内部 query block 从 32 扩到 128；GQA 保持原来的 32。
2. causal 多行调用将计算 span 向 KV tile 对齐、截在已加载 KV 长度内，以复用分段 graph。保留按可见前缀判断数值模式的逻辑和 PV 前的未来 carrier 清零。
3. CPU centering 的完整八元素向量使用 FP16 subtract/clamp，代替先升 FP32 再降 FP16。边界和 mask 标量路径保持原计算。
4. causal V 前缀修正使用硬件 FP16→FP32 转换；query pack 按 KV head 使用四个 CPU 线程。

这些变体仅存在于诊断构建生成的 attention 源码中，尚未作为生产策略集成。decode 不应采用此组合。

单因素筛查表明：把 centering 的 128 行块缩到 32/8 并没有改善，多个长上下文测点明显变慢；单独换半精度 subtract 的完整调用收益也不稳定。MHA query 合并有明确作用，span 复用则会增加不可见区域的 QK/LUT/mask 工作，不能只以建图次数下降作为成功依据。筛查样本见 [matrix](evidence/attention_firstprinciples_20261022/matrix/summary.json) 和 [arithmetic](evidence/attention_firstprinciples_20261022/arithmetic/summary.json)。

## 瓶颈在哪里

当前 [query_gqa](../native/rk_npu_matmul/src/rk_npu_attention_f16.cpp) 仍统一将内部 query 行数限制为 32。GQA 已经把同一 KV head 的 query heads 合并到物理 M 维：Qwen3 的 G=4、R=32 得到 M=128；不能再把“把四个 query head 合起来”当作新优化。MHA 的 G=1 则只有 M=32，因此合并 query 的余地更大。

稳定 causal LUT 的每个内部图实际执行：QK → score FROM sync → CPU max/subtract → score TO sync → LUT → exponent 尾带 FROM sync → CPU 清未来 carrier → 尾带 TO sync → PV → partial FROM sync → CPU 修正/归一化。一个 128 行 GQA 调用对应四个这样的图，共十二次 blocking submit。

下表为 Qwen3、Q128/KV4096、原分块策略的独立诊断样本均值，阶段总调用约 30.95 ms。它用于解释路径，不与上表另一轮未插桩基线的 28.96 ms 混用。

| 互斥阶段 | ms | 含义 |
|---|---:|---|
| QK submit/wait | 2.910 | 含设备执行和提交等待 |
| LUT submit/wait | 3.898 | DPU exponent 单独占用执行阶段 |
| PV submit/wait | 4.547 | 包括各 KV tile 的 partial |
| CPU score centering | 8.500 | 行最大值与减最大值 |
| score FROM + TO sync | 3.001 | 两次方向的 cache ownership 同步 |
| 几何准备与 tail staging | 3.885 | 其中 graph prepare 3.294 ms |
| CPU 输出收尾 | 2.659 | partial 累加、V 前缀 carrier 修正、归一化、整理 |
| Q pack | 0.602 | 缩放、布局转换和输入同步 |
| causal carrier 清零及尾带同步 | 0.405 | 不能直接删，否则改变未来 token 对 PV 舍入的影响 |
| 输出同步 | 0.406 | partial FROM_DEVICE |

`hardware_us` 是驱动报告的设备区间，不是纯 MAC 时间；`submit_us` 更包含 CPU softmax 和同步，不能拿它当 ioctl 开销。设备区间中 QK/LUT/PV 分别约 2.860/3.851/4.521 ms。LUT 不做矩阵乘，但同样占用 NPU 的串行流水线时间。

历史 [LUT 驻留实验](../../lowlevel/tools/npu_cpu_offload_probe/LUT_RESIDENT.md) 已测得刷新表的边际成本很小；历史 [native Softmax](../../lowlevel/docs/fp16_native_softmax_2026-08-27.md) 已验证行 max 和 max broadcast/subtract，同时表明完整官方 Softmax 的多个全张量阶段并不便宜。本轮优化应针对剩下的数据路径。

## 第一性原理的约束

### Decode 的低 MAC 利用率可以是正常结果

令 query 数为 R，KV 长度为 S，head 维度为 D，G=Hq/Hkv。两次 matmul 的逻辑工作量约为 `4*Hq*R*S*D` FLOPs。仅理想读取一次 FP16 K/V，就需要 `4*Hkv*S*D` bytes，所以只考虑 KV 的算术强度为 `G*R` FLOP/byte。

Qwen3 单 token 的 G=4，因此只有 4 FLOP/byte。S4096 时，约 67.1 MFLOPs 对应至少 16 MiB 逻辑 K/V。假设可用带宽为 20 GB/s，仅这份 K/V 就需约 0.839 ms，与本轮 0.862 ms 完整调用同量级。20 GB/s 是量级假设，16 MiB 也不含 V 的 denominator/padding、score 和缓存维护；这不是 DDR 流量实测或饱和证明。

因此 decode 的实质问题是 KV 字节数与有效读取速度。若需要明显提速，应验证 KV 低精度及其解码成本，或让更多 query 共享一次 KV 读取；后者需要 batch/MTP 等上层提供真实工作量。单纯延长任务链不会改变这个下限。

### Prefill 的中间张量足以压垮有效算术强度

Qwen3 Q128/KV4096 的完整矩形 score 为约 32 MiB。现有路径包含 QK 写、CPU max 读、CPU subtract 读写、LUT 读写、PV 读等七份全量逻辑访问，合计约 224 MiB，尚未计 K/V、partial、mask 和缓存维护。实际 causal 分段略小；CPU cache 命中也会改变 DDR 请求数。这是 logical-byte estimate，不是 NoC/DDR counter 结果。

这解释了“矩阵乘算得不多，整次却很慢”：CPU 与 NPU 都在围绕同一中间矩阵读写，LUT 的 DPU-only 阶段还会令同核 MAC 阵列空闲。CBUF 是受约束的卷积流水线缓冲区，不能直接当成 GPU 的可编程共享 SRAM。

Qwen3.5 还有按 KV head 分核的不均衡：四个 KV heads 分成 2/1/1，若每 head 同耗时，仅此划分的理想计算利用率上限为 4/6；Qwen3 的 3/3/2 则是 8/9。这是分工上限，不是完整调用可直接获得的加速倍数。改为按 query 或 KV tile 分核必须把重复读 KV、额外任务和合并成本算进去。

## 值得继续验证的结构改动

### 首选：让 NPU 完成局部稳定化，CPU 只处理小结果

针对每个 KV tile，在设备上完成 QK、row max、subtract、LUT、精确 mask 和 PV，输出小的局部状态 `(m_b,l_b,u_b)`，其中 `l_b=sum(exp(s-m_b))`，`u_b=exp(s-m_b)@V`。CPU 在 FP32 中只合并这些状态：

`m=max_b(m_b)`

`O=sum_b(exp(m_b-m)*u_b) / sum_b(exp(m_b-m)*l_b)`

分块 softmax 合并是已有方法；[FlashAttention 2](https://arxiv.org/abs/2307.08691) 和 [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) 提供算法依据。RK3588 的工作是把它映射到已经验证的 DPU/CNA 布局和任务状态，并用设备侧任务成本抵消 CPU 全 score 往返的删除收益，不能照搬 GPU kernel 的速度结论。

最小实验应只拼接已验证的 ReduceMax、max broadcast/subtract 与当前 LUT/PV；保留延后归一化，不恢复完整七阶段 Softmax。causal mask 必须在 PV 前得到真正的零，先用独立 NPU 后置 mask 任务建立正确性，再考虑融合。阶段判断以相同 QKV 的完整调用为准，不能只计 exp→PV 驻留链。

如果增加的 DPU 扫描与广播仍吃掉节省的 CPU 时间，则停止扩展此配方，转向下面的重算方案。

### 更激进的候选：额外计算一次 QK，替代 score 搬运与 max 广播

这是待验证的硬件映射假设：

1. 第一遍 QK 求行最大值；优先验证能否让 QK flying 输出经 DPU minmax 直接落成小向量。若必须先写全 score 再 ReduceMax，则把这份额外流量计入。
2. 将已缩放的 Q 扩成 `[Q_scaled,-m]`，K 扩成 `[K,1]`，第二遍 GEMM 在实数算术下直接产生 `Q_scaled@K^T-m`。
3. 用第二遍 QK 的 DPU epilogue 执行 LUT，再进入 PV，避免 CPU 扫描、改写 score，也避免动态行 max 的全张量 broadcast。

RKNPU2 的 K32 对齐使 D128 变成 K160，增加 25% 的第二遍 K 工作量；D256→288 增加 12.5%，D64→96 增加 50%。还要支付第一遍 QK 和 max 的成本。应先选 D128/256，而非盲目用于全部模型。

本轮 Qwen3 诊断中 QK submit 约 2.9 ms，CPU centering+score sync 约 11.5 ms，独立 LUT 约 3.9 ms，给出了“多算少搬”值得测试的量级依据。只有确实删除相应中间读写和独立阶段后，这个比较才成立。

三个 gate：flying max 的输出布局/数值正确；augmented GEMM→LUT 的状态与范围正确；含最终归一化的完整调用明显优于当前最佳路径。移动减 max 的位置会改变 FP16 舍入，即使实数代数相同，也必须重新做真实 QKV、logits/PPL 验证，不能要求或宣称与现实现逐位相同。

### 低风险备选：CPU/NPU 双缓冲重叠

保留原数值顺序，用两个 score/workspace 槽，让 NPU 处理下一组 QK 时 CPU 处理上一组 score，随后调度 LUT/PV。它不减少主要字节数，但可能隐藏部分 CPU 等待。按 tile/head 重排依赖，避免跨核同一缓存范围争用。收益上限受两侧共享 DDR 限制；仅把 blocking ioctl 放到线程里并不能保证重叠。

## 实现与复现

新增诊断入口 [profile_attention_pipeline.cpp](../native/rk_npu_matmul/tests/profile_attention_pipeline.cpp)，构建脚本 [build.py](../native/rk_npu_matmul/tests/tools/attention_pipeline/build.py)，筛查脚本 [run_matrix.py](../native/rk_npu_matmul/tests/tools/attention_pipeline/run_matrix.py)，配对脚本 [run_paired.py](../native/rk_npu_matmul/tests/tools/attention_pipeline/run_paired.py)。构建脚本在输出目录生成修改后的 Attention 编译单元，并链接当前原库静态 archive；原库基线另行直接链接同一 archive。

板端快照与完整输出：`<BOARD_HOME>/mnt/3090x4/tmp/attention-firstprinciples-20261022/`。主机对应 `<HOST_HOME>/mnt/3090x4/tmp/attention-firstprinciples-20261022/`。真实 fixture 来自相邻 `qwen35-dev/mtp-20261018/cpu-softmax-20261019/phase2-p1021-r4-{q,k,v}.f16`。

在板端快照根目录，先构建，再执行；测量期间不要编译：

```sh
cmake -S native -B build -DBUILD_TESTING=OFF
cmake --build build --target rk_npu_matmul_static -j3
python3 native/tests/tools/attention_pipeline/build.py \
  --base-archive build/librk_npu_matmul.a --output-dir diagnostic
python3 run_board.py --cpu-khz 2256000 --skip-dmc --report clocks.json -- \
  taskset -c 4-7 python3 native/tests/tools/attention_pipeline/run_paired.py \
  --binary diagnostic/profile_attention_pipeline \
  --baseline-binary diagnostic/profile_attention_baseline \
  --output-dir paired-replay \
  --fixture ../qwen35-dev/mtp-20261018/cpu-softmax-20261019/phase2-p1021-r4
```

## 验证边界

本轮没有完整 prefill/TTFT、decode generation 或整模型 PPL A/B；表中百分比仅适用于所测单算子调用。实验 span 复用使用热态 graph，在真实从空 KV 增长的 prefill 中仍会跨桶重建，需要单独计入。没有把局部加速外推为整模型收益，也没有测量实际 DDR bytes。NPU 局部稳定化、QK 重算和双缓冲仍是下一阶段实验方案。
