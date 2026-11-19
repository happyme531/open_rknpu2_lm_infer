# Dense 新版 SwiGLU runtime 接入，2026-10-23

## 接入行为

当前工作树的 W4 dense gate/up 融合入口不再按 `rows<32`、用户线程数少于2或普通 GEMM tuner 选择单线程而返回 false。

- M1 的融合 CPU 工作使用一个线程。
- M>1 使用用户配置的 CPU 线程数，默认四线程，遵守 `--threads`/CPU mask 的限制。
- 复用普通 GEMM 已选的 K/M/N 分块、布局、pipeline 和 NPU 核配置；融合明确采用 CPU epilogue。
- 普通 Linear 的策略和 workspace 不被改写。其资源策略已经匹配时直接复用；不匹配时为该 geometry/bucket 按需创建一个独立的 SwiGLU workspace，后续复用。
- 普通和融合路径共用原有输入/输出 staging；补齐 bucket 的输入尾行清零，融合输出仍按 `[rows,F]` 拷贝实际行。
- 新增 workspace 的内存计入已有 `workspace_payload_bytes`，`swiglu_workspace_count` 记录额外的融合 workspace。`retain_workspaces` 随普通条目一起释放它们。

本次接入的是 `rk_npu_w4a8_run_swiglu_f32`。Down 仍通过现有独立 Linear 调用执行；没有在这一轮额外接入完整 FFN/scale 复用入口，也没有改 routed/shared 联合调度的调用方。

## 删除旧实现

`src/model/cpu.rs` 的 degree12 FP64 指数、旧四 lane SwiGLU 和超出 ±80 时的逐元素回退已删除。独立激活通过 `native/cpu_kernels/swiglu.cpp` 调用与 W4/MoE 相同的 FP32 SIMD 头文件，不再复制另一套公式。

W8、浮点或多 part 投影，以及需要保存 gate/up 中间张量的 tensor trace，仍通过独立激活完成；它们也都使用新内核。这些是格式/观测需求分支，不是旧精度策略回退。W4 融合仍要求单 part、N 为64的倍数，这是算子接口的几何要求。

已无生产调用者的 `native/rk_npu_matmul/src/rk_npu_swiglu_neon.h` 也已删除。历史 FP64 参考仅保留在未提交的实验工具中，不进入生产构建。

NOTE：AArch64 使用 degree7 FP32 exp，指数无分支饱和至 [-80,80]，gate<-80 映射为带符号零；不承诺原 expf/FP64 的逐位结果和非有限语义。不足16个元素的尾部补到同一 SIMD 内核计算，仅写回有效元素，不转回旧精度核。非 AArch64 主机构建保留 FP32 标量实现用于可移植构建。

## 改动范围

- `src/npu.rs`：小 M 融合接入、独立线程策略与 workspace 复用、扩展板端集成测试。
- `src/model/cpu.rs`：删除旧 FP64 算法，改用私有 C++ ABI，增加 SIMD 尾部输出保护区测试。
- `native/cpu_kernels/{CMakeLists.txt,swiglu.cpp}`、`build.rs`：接入共享 FP32 内核，同时支持 `RK_NPU_MATMUL_ROOT` 指定的算子库根目录。
- 算子库仅删除未使用的旧 FP64 头文件；公共 C API/ABI 没有变化。

已签名提交：runtime `3b9ab0de79bf22374e2603f18f34342e8d2753ac`（`perf/dense-swiglu-runtime`），算子库旧头文件清理 `511b9aeeeecc895e57db9df630a04c54c24f9145`（`perf/moe-ffn`）。提交只含生产源码、相关测试与子模块版本更新；本报告、实验工具、证据和二进制未提交。未修改其他工作树的 runtime，外层仓库已有的 AGENTS/报告修改和其他 agent 的实验文件保持原样。

## 验证范围

新增/扩展的检查包括：

- 独立 CPU 激活的 decode/prefill、±90 输入、F9728 宽行和非16倍数的尾部；使用 scalar 公式作为数值参考。
- 实际 Rust→C++→NPU 调用，W4/W8 普通 Linear，M1/2/3/4/5/8/9/16/17/31/32/33/65/127/128/129/191/192/193/255/256。
- 所有支持的 W4 小 M 都接受融合，W8 明确走独立路径；普通输出及融合中间输出对独立整数/浮点 oracle。
- 模拟 M8 普通 GEMM 缓存选择 CPU1，并先移除 M5 已建立的同 bucket workspace，再验证融合没有改写原策略。
- 普通/融合交替调用、重复输出和分配量保持稳定、workspace retain、cache-only 路径不重载权重或启动调优。

主机和板端记录保存在 `evidence/dense_swiglu_integration_20261023/`。板端使用独立 target/cache 目录、共享 NPU 锁及固定频率；没有跑全模型 PPL 或将之前 C++ 单算子收益当作此次 runtime 实测加速。

最终结果：

| 检查 | 结果 |
|---|---|
| 主机 `cargo test --release --locked --lib` | 69 passed，14 个板端/模型用例按声明 ignored |
| 板端 `model::cpu::tests::` | 6 passed，含新 FP32 激活与尾部 guards |
| 板端 `quantized_linear_padding_and_swiglu` | passed，覆盖所有上述 M 及普通/融合交替 |
| AArch64 `cargo clippy --release --locked --offline --all-targets -- -D warnings` | passed |
| AArch64 正常 CLI release 构建与 `--help` | passed |
| `cargo fmt --check`、`git diff --check` | passed |
| 测试与 CLI 构建 runner | exit0，restore_errors=[] |

板端验证使用实际 runtime 源码和私有 C++ 绑定，不是上一轮的 C++ 算子模拟。正常 CLI 保存在独立构建目录 `/tmp/dense-swiglu-runtime-20261023/release/`，并复制到证据目录；没有替换原有生产二进制。

证据：[主机测试](evidence/dense_swiglu_integration_20261023/host-tests.log)、[板端 CPU](evidence/dense_swiglu_integration_20261023/cpu-tests.log)、[板端 NPU](evidence/dense_swiglu_integration_20261023/npu-swiglu.log)、[Clippy](evidence/dense_swiglu_integration_20261023/clippy.log)、[CLI 构建](evidence/dense_swiglu_integration_20261023/build-cli.log)。
