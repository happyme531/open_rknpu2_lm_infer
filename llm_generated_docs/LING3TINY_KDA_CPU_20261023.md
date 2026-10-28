# Ling3tiny KDA CPU 优化 20261023

KDA 的 FP32 state 更新复用 GDN 四行融合 helper，合并 decay+dot 与 mad+dot；每个输出行保留原有四个 NEON 累加器及归约顺序。Rust 输入值检查改为可向量化的布尔归约，形状、有限值、decay≤0 与 beta∈[0,1] 的接受条件不变。

## 范围与算子库基线

算子库 trace 支持已签名提交为 `b917811d1f09d02bab9d76969c2bce00bbd861aa`，标题 `perf: expose MoE stage timings in configurable traces`；仅修改 `src/rk_npu_moe_w8.cpp` 与 `src/rk_npu_trace.cpp`。本轮 KDA 对照固定使用该提交的 native snapshot，不包含其他 MoE 优化。

KDA 修改集中在 `native/cpu_kernels/{kda.cpp,vec.h,block.h}` 和 `src/ops/kda.rs`。scalar decay 的 GDN 与 vector decay 的 KDA 共享模板，未增加运行时后端选择。FP32 state、log-decay 的 `std::exp`、FMA/归约顺序及 C ABI 保持不变。

## 真实输入单算子

从冻结二进制 `/tmp/ling-review-20261023/optimized` 的第 0 个 KDA 层捕获 Q/K/V、log_decay、beta 和初始 state。覆盖 M128 零/非零 state 及 M1 非零 state。GDB 捕获时的 benchmark 数值不作为性能数据。重放为 CPU4–7、1/4 线程 @2.4 GHz；编译使用系统 C++、`-O3 -DNDEBUG -fPIC -fopenmp -mtune=cortex-a76 -std=c++17`，没有 fast-math。

下表为八个正反顺序样本的中位数。每次样本恢复初始 state 后计时，多次调用连续更新 state；恢复开销不计入。native 计时包含 OpenMP、scratch、exp 和递推，不含 Rust 输入检查。所有候选及生产实现与原 kernel 的输出、最终 state 均逐位一致，重复调用后仍一致。

| 输入 | 线程 | 原 native µs | 当前 native µs | 时延下降 |
|---|---:|---:|---:|---:|
| m128_zero | 1 | 25763.48 | 16533.52 | 35.83% |
| m128_zero | 4 | 6642.39 | 4220.07 | 36.47% |
| m128_nonzero | 1 | 25723.92 | 16217.89 | 36.95% |
| m128_nonzero | 4 | 6585.99 | 4173.66 | 36.63% |
| m1_nonzero | 1 | 199.84 | 135.34 | 32.28% |
| m1_nonzero | 4 | 68.30 | 51.71 | 24.28% |

生产实现采用四行融合。独立 row、block4、tile8 的原始样本均保留在 `production-probe.json`；没有加入 token 分块的运行时切换分支。

## Rust 输入检查

原先 `chain(...).all(...)` 的逐元素提前退出改成每个 slice 内的完整布尔归约。有效输入的归约由 LLVM 自动向量化；AArch64 反汇编包含向量 load、比较、AND 和 UMINV。NaN、正负无穷、非法 gate/beta 仍被拒绝，±0 与 beta 边界保持原语义。

| 输入 | 原扫描 µs | 当前扫描 µs | 时延下降 |
|---|---:|---:|---:|
| m128_zero | 1331.54 | 329.51 | 75.25% |
| m128_nonzero | 1343.15 | 324.32 | 75.85% |
| m1_nonzero | 10.41 | 2.32 | 77.70% |

扫描重放固定 CPU4 @2.4 GHz，原始样本见 `check-fold.jsonl`。此表是独立检查开销，不与 native 微基准直接相加推导整模型速度。

## 整模型对照

主板 zt-firefly，`Ling-3.0-tiny-QQQ-W4A8-Mixed-CT`，resident、context640、ubatch128；CPU4–7/4 线程 @2.4 GHz、NPU mask7 @1 GHz、DDR @2.112 GHz。固定 MT19937 seed42 的 640 个 token，连续 pp512→tg128。Head-free prefill；decode 历史深度 512…639，包含输出头。计时排除加载、prepare、tokenization、sampling 和 stats。

A/B/B/A，每组一次 warmup、三次测量，两版本各保留六个样本；下表吞吐由中位耗时计算。

| 阶段 | Baseline ms | 当前 ms | Baseline tok/s | 当前 tok/s | 吞吐变化 |
|---|---:|---:|---:|---:|---:|
| Prefill512 | 2884.723138 | 2628.015075 | 177.4867 | 194.8238 | +9.7681% |
| Decode128 @512 | 6045.429693 | 6008.074574 | 21.1730 | 21.3047 | +0.6217% |

收益主要在 prefill；decode 本次为小幅改善。四组正式测量均为 374 cache hits，miss/search/fallback/weight reload 为零；head-free logits 检查通过，prefill head_ms=0。

| 配置/阶段 | 六个原始耗时 ms |
|---|---|
| baseline prefill | 2860.781174, 2873.162057, 2846.774063, 2897.772613, 2896.284219, 2907.038395 |
| baseline decode | 6044.579214, 6039.677924, 6032.221082, 6060.877396, 6046.280172, 6050.838105 |
| optimized prefill | 2613.367248, 2629.818618, 2617.355547, 2632.413025, 2637.400005, 2626.211532 |
| optimized decode | 6035.153826, 6012.201417, 6010.366227, 6004.296276, 6005.782920, 6003.455683 |

## 正确性与复现

fmt、全 targets release Clippy 及扫描工具 Clippy 通过。完整回归 **119 passed、22 ignored**，包含 GDN scalar decay、KDA 独立 FP64 oracle、非零初始 state、D7/D17/D128、M128、单/四线程、分块/reset 和输入边界。

137-token prompt 加 3 个 continuation 的 628736 个 logits，resident/streaming 与基线逐位一致；chunk1/3/7、head-free、取消重放均逐位一致。WikiText 原有 8×512 窗口/2040 targets：PPL **11.97287615259297**，全部逐 token NLL 与 `results/ling-review-20261023/ppl.json` 完全一致。全部定频 runner 的 exit_code=0、restore_errors=[]。

证据目录为 `results/ling-kda-20261023/`：`component-summary.json`、`comparison.json`、`validation.json` 保存汇总；`inputs/` 保存实际 kernel 输入；`production-probe.json`、`check-fold.jsonl`、`{a1,b1,b2,a2}.json`、`fixed.json`、`stream.json`、`ppl.json` 保存原始结果。`native-snapshot/` 固定在算子库提交 b917811，`runtime-snapshot/` 保存本轮隔离构建源码。

新二进制为板端 `/tmp/ling-kda-optimized`，检查器为 `/tmp/ling-kda-check`；baseline 为 `/tmp/ling-review-20261023/optimized`。构建与完整模型复现命令保存在 `build.sh`、`baseline.sh`、`after.sh`。独立 kernel 对照可在板端运行：

```sh
c++ -O3 -DNDEBUG -fPIC -fopenmp -mtune=cortex-a76 -std=c++17 \
  -I native/cpu_kernels tests/tools/kda_cpu_probe.cpp \
  native/cpu_kernels/kda.cpp -o /tmp/ling-kda-probe-production
python3 scripts/run_board.py --report results/kda-replay-clock.json -- \
  taskset -c 4-7 /tmp/ling-kda-probe-production results/ling-kda-20261023/inputs
```
