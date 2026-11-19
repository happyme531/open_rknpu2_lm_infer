# MTP 浮点投影调优入口与性能影响

可以补齐，现有 FP16 complete-call tuner、缓存与执行接口都能复用。需要补充 MTP 的实际 shape 枚举和适合混合 FP16/INT8 运行的候选约束。根据现有板端记录，主要价值是调优覆盖和可操作性；尚无证据表明补齐入口能带来明显的 decode 加速。

本次核对当前源码，并重新汇总 2026-10-25 的原始数据；没有修改运行时代码，没有重新运行板端搜索或性能测试。

## 当前缺口

1. `src/cli.rs` 的 `validate_command()` 禁止 `tune --spec-type draft-mtp`。普通 target tuning 不加载 MTP。
2. `Qwen35::prepare_rows()` 只遍历 target 和输出头 backend；MTP 的 `Predictor::load()` 上传浮点权重，实际投影在执行时才准备 workspace。因此只解除 CLI 限制，随后调用 `model.prepare()`，仍不会搜索 MTP 浮点投影。
3. `FloatBackend::prepare_for_weights()` 已支持搜索和缓存，但通用搜索的候选集不受 decoder 的 Ntile 限制约束。运行时对宽于 4096 的已选策略进行拒绝，发生在搜索之后；它不能避免搜索期间的超时，也可能让搜索结果无法被 decoder 使用。
4. resident weights 在加载时收集缓存已有的 K 分片。加载权重后才搜索出新的 K 分片，会因 `rk_npu_mm_tuned_weights_match()` 不匹配而退回默认策略。应在权重上传前完成离线搜索，之后用新进程验证加载与执行。

内置 decoder 策略已经使用 `m_tile=min(M,32)`，并按可用核心数均分 N、按 32 对齐、限制 Ntile 不超过 4096。缺少 profile 不等于使用原始通用 M128 策略；现有小 batch 的手工策略已经接近多个历史搜索结果。

## 实际工作负载

当前 Qwen3.5-4B C7 为 hidden=2560、intermediate=9216、16 个 query heads、4 个 KV heads、head_dim=256。按当前 `projection::Linear::floating()` 的外部分 K 逻辑，每个完整 draft step 有以下调用：

| 投影 | 原始 K / N | 实际浮点 MM 的 K / N | 每个 draft step 的次数 |
| --- | --- | --- | ---: |
| fc | 5120 / 2560 | 2560 / 2560 | 2 |
| q | 2560 / 8192 | 2560 / 8192 | 1 |
| k、v | 2560 / 1024 | 2560 / 1024 | 2 |
| o | 4096 / 2560 | 4096 / 2560 | 1 |
| gate、up | 2560 / 9216 | 2560 / 9216 | 2 |
| down | 9216 / 2560 | 3072 / 2560 | 3 |

只有六种独立 K/N。draft 每次是 M1；teacher-forced prefill 和验证后 catch-up 只需要 fc、k、v，不执行完整 MTP block。后面三组投影不需要按所有 prefill batch 搜索。

fc、k、v 还需要覆盖实际 prefill rows、首块少一行的 shifted rows，以及验证后 catch-up 的 `1..=max_draft+1`。尾块/checkpoint rows 可按请求的 batch 枚举，不能只准备 nominal ubatch。

## 现有性能证据

两组均来自主板：CPU4–7 2.4 GHz、NPU 1 GHz、DDR 2.112 GHz、4 线程、CPU mask0xf0、NPU mask7、ubatch128、单 prefill ubatch、C7、greedy MTP1、context4096。各 prompt 预热一次，测量三次；输出128 tokens，decode 计时覆盖首 token 之后的127 tokens。

基准组的 MTP 使用内置策略；第二组补齐14个安全 shape/M 配置，gate/up K2560/N9216 仍使用内置 Ntile3072。九对输出 token IDs、接受/提议序列均相同。这是顺序运行的历史数据，不是交错 ABBA。

| Prompt tokens | 内置策略 decode tok/s | 补齐14个配置后 tok/s | 变化 | 第二组累计 MTP decoder / decode 时间 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 11.5090 | 11.4530 | -0.49% | 8.10% |
| 1024 | 11.4661 | 11.4093 | -0.50% | 8.40% |
| 3800 | 10.7055 | 10.6608 | -0.42% | 10.12% |

这些数据没有显示安全配置补齐后的加速，也不足以把约0.5%的变化归因于调优。

`draft_ms` 占 decode 约20%，但它包含共享输出头，不能作为 MTP 浮点投影的耗时占比。表中的 `mtp_decoder_ms` 包含完整 MTP block、CPU/attention、catch-up 和 prefill；拿它除以 decode 时间已经是对可优化部分的宽松上界。即使把整个累计 MTP decoder 时间减半，估算 decode 吞吐提升也只有约4.2%～5.3%；只优化缺失的 gate/up，其范围更小。该估算不是实际调优收益。

历史 K2560/N9216、M1 通用搜索返回 `rc=-5`，dmesg 记录约500ms job timeout 和 NPU soft reset。记录未定位具体失败候选，不能仅凭 Ntile 限制就宣称新搜索已安全。现有 M1/M2 gate/up 内置策略的运行记录正常。

## 补齐方式

1. 为 MTP 增加形状规划，复用投影的 K 分片规则，按实际用途去重 M/K/N。让现有 `tune --spec-type draft-mtp` 覆盖 target 和 MTP，报告分别列出命中、搜索和实际采用的策略。
2. 先在算子库中为 decoder 使用受约束的候选集：包含当前已验证的内置策略，避免已知不适合混合运行的宽 N task；候选约束、缓存校验和运行时选择保持一致。先验证 gate/up 的完整单算子调用，再开放完整搜索。
3. 离线调优先于 resident 权重创建；新进程验证 profile 能被实际采用。比较 logits 和 greedy token/接受序列，并用固定频率、相同 prompt 的 ABBA 比较完整 decode。统计需同时确认 cache miss、搜索、fallback 和权重重载，不能仅以搜索成功或 cache hit 判断性能收益。

优先补入口和受约束搜索，性能目标应以实测决定。现有证据不支持把它列为大幅提升 decode 的主要手段。

## 原始记录

- [内置策略完整模型数据](evidence/readme_rebench_20261025/mtp-default-profiles/native.json)
- [补齐安全配置后的完整模型数据](evidence/readme_rebench_20261025/qwen35-text/native.json)
- [协议与模型配置](evidence/readme_rebench_20261025/protocol.json)
- [14个安全配置及保留的默认 shape](evidence/readme_rebench_20261025/mtp-profiles.json)
- [宽 shape 搜索结果](evidence/readme_rebench_20261025/mtp-profiles/m1-k2560-n9216.json)
- [硬件超时记录](evidence/readme_rebench_20261025/mm-timeout-dmesg.txt)
