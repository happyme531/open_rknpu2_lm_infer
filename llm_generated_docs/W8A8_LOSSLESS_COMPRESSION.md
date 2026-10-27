# W8A8 权重无损压缩

`--compress-w8a8` 默认关闭，可用于所有子命令，例如：

```sh
./bin/open_rknpu2_lm_infer --model MODEL --compress-w8a8 chat
./bin/open_rknpu2_lm_infer --model MODEL --compress-w8a8 serve --port 8080
```

参数覆盖所有通过 `npu::Backend::load` 创建的 W8A8Linear，包括 W8 LM head 和 Qwen3.5 的 W8 投影。W4A8 与浮点线性层保持各自的存储方式。Rust 调用方可设置 `ModelOptions::compress_w8a8`。

实现使用算子库的 [`rk_npu_f32i8f32_weights_create_compress`](../native/rk_npu_matmul/docs/API.md#compressed-weights)，直接接收已有 INT8 codes 与 FP32 scales，不重新量化。算子库逐个 K/N tile 比较包含编码头部和对齐的压缩大小，只有严格小于原始 native INT8 tile 时才使用 DCOMP，否则保留 raw tile。执行时由 NPU inline 解码，沿用原有 workspace、run 和 free 接口。

压缩增加加载时的 CPU 编码成本。每种新的执行 N 分块会在首次使用时准备并缓存布局；同一权重使用多种 N 分块会保留多份布局。压缩后的体积与实际分块有关，节省存储不保证推理更快。

开启参数后，运行时会在模型加载结束，以及 native 实际创建新的压缩布局后，集中调用 `malloc_trim(0)` 回收准备阶段已经释放的 CPU 堆页。native 通过 layout generation 通知布局新增；Rust 记录创建时 seed 的初始 generation，不按 M bucket 推断。不同 M 复用同一 `n_tile`、首次执行命中 seed 时均不重复请求回收。整组 decoder 层与 LM head 分别收尾；双 ubatch 的两个执行通道共享待回收标记，并在并发执行结束后统一处理。W4 和未压缩 W8 的 LM head 直接跳过回收检查。清理不改变仍在使用的权重、KV 和 workspace；实际可归还的页数受堆布局影响。

算子库当前的 NPU split-K reduce 链仅支持未压缩 W8 权重，不能与 `RK_NPU_I8_NPU_REDUCE=1` 同时使用；该组合在创建 backend 时会明确报错。沿用现有 CPU INT32 归约和 FP32 反量化路径。

`stats.compress_w8a8` 表示开关状态。启用压缩且加载了 W8 权重时，`stats.weight_payload_bytes_estimated` 为 `true`：`weight_payload_bytes` 沿用算子库 shape-only query 的未压缩布局估算，用于现有 IOMMU domain 容量规划，不代表压缩后的实际占用；多布局缓存也未计入此估算。原始 checkpoint 文件不修改。

## 验证

2026-10-27，在 `zt-firefly` RK3588 板端完成：

- `cargo test --release --locked --lib`：102 passed，24 ignored；包含所有子命令的参数默认值、全局位置和 `ModelOptions` 传递检查。
- `lossless_w8a8_matches_raw_across_batches_and_partitions`：K/N 为 32/32、257/288、4353/257，核掩码 1/3/7，M 为 1/3/128/129/256 并重复 M1；raw/DCOMP 的 FP32 输出全部逐位一致，覆盖不缩小的小块、混合 tile、尾块和 split-K。
- `independent_execution_lanes_share_weights`：W4、raw W8、压缩 W8 在核掩码 1/3/7 下共享权重、使用独立 workspace 并发执行，输出逐位一致。
- `cargo clippy --release --locked --all-targets -- -D warnings`：使用板端本地临时 target 目录通过。
- `cargo build --release --locked`：通过。本次修改的 Rust 文件通过 rustfmt；全项目格式检查仍报告既有未提交改动 `src/model/qwen35/load.rs` 中的格式差异。
- 真实 `Qwen3-4B-W8A8-Baseline` checkpoint：145 个 W8A8Linear（含 LM head），`--no-tune --ctx-size 128 --ubatch-size 128`，相同 8-token prefill 与 2-token continuation。开关前后 455,808 个有限 FP32 logits 逐位一致；压缩统计开关和内存估算标记均符合预期。摘要为 `model-comparison.json`。

板端正确性验证日志保存在 `/tmp/w8a8-compression-20261027/`。[布局回收触发与输出头检查](W8A8_COMPRESSION_LAYOUT_REVIEW_20261028.md)记录当前实现和回归；[自动堆回收和内存实测](W8A8_COMPRESSION_HEAP_CLEANUP_20261027.md)包含2026-10-27版本的真实模型回归、回收调用审计和同二进制对照；[初始内存和整模型性能对比](W8A8_LOSSLESS_BENCH_20261027.md)保留回收接入前的 ABBA 测量及原始样本。未进行 PPL 测试。
