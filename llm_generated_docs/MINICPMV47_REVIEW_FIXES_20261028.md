# MiniCPM-V 4.7 流式加载修复（2026-10-28）

## 行为

- staging 请求按当前层的当前需求重建。消费成功或失败的 record 都清除对应标记；新一轮 demand 清除该层旧标记，其他层不受影响。importer 取队首，并在导入前确认 generation、pending 和 staging 请求仍有效。
- Reader 在启动第一个线程前建立所有权。后续 I/O worker 或 importer 创建失败时，已有线程由 Reader 的析构路径停止、唤醒并 join。
- 专家跳过阈值为零时，不复制原始路由权重，不扫描剪枝或丢弃统计，直接累加行数和分配数。启用张量 trace 时，原始权重引用当前未修改的权重；非零阈值保留原有剪枝和统计语义。
- 原生 CPU 聚合构建目标包含 Attention pipeline 与 causal 两个测试 executable，覆盖已注册的 CTest 项。README 不再保留过时的测试数量。

## 回归覆盖

- 注入第二个 reader 和 importer 创建失败，检查错误返回以及已启动线程释放所有 Store 引用。
- 成功/失败消费均清除 staging 标记；新一轮需求清除本层标记并保留其他层标记。
- 固定 token 序列执行 M128、M1、M4、M16；追加 prompt 回归使用 `[128, 4, 16, 32, 4, 32, 1]`，断言每个大于 16 行的调用不增加 staged_records。
- 使用干净 CMake 构建目录，只构建 `rknpu2_matmul_open_cpu_tests`，随后执行完整 CTest。

## 验证结果

- ARM release 格式检查和全目标 Clippy（`-D warnings`）通过。常规测试合计 118 passed、22 ignored，包含两种线程创建失败注入与 staging 生命周期检查。
- ARM 干净目录仅构建 CPU 聚合目标后，24/24 个 CTest 全部通过，包含 Attention causal 和 pipeline。
- H88K / RK3588 / 8 GiB，GM7 本地 NVMe，noSQ MiniCPM-V 4.7 35B-A3B；20 槽/层，ubatch128，CPU4–7 2.4GHz、NPU 三核 1GHz、DDR2112MHz。四次模型回归与常规测试的五份 clock report 均恢复成功。
- 单行及 M4/M16 路径分别比较 2,233,296 / 1,488,864 个 FP32 logits，与 `commit_20261026` 固定参考逐字节相同。
- 追加 prompt 路径的 1,737,008 个 FP32 logits 在预取关闭/开启两条路径逐字节相同。
- 开启预取时，第一次 M32 追加调用前后 staged_records 均为 1605，第二次均为 1922；两个追加 prefill 均没有额外 staging 导入。M4/M16/M1 仍正常使用 staging。

原始脚本、完整日志、clock report 和追加 prompt 的 timeline 保存在忽略目录 `llm_generated_docs/evidence/minicpmv47_review_fixes_20261028/`；完整 logits 留在板端 `/mnt/nvme/minicpmv47/results/review_fixes_20261028/`。本轮不改变缓存容量、模型数学或默认运行参数，未重新测量吞吐。
