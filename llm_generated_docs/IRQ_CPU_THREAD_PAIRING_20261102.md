# NPU IRQ 与提交线程逐核绑定

2026-11-02，主板 RK3588，当前算子库、Qwen3-4B 第 2 层真实权重及捕获输入。CPU4–7 2.4 GHz、NPU 1 GHz、DDR 2.112 GHz，共享 NPU lock，OpenMP passive wait。

**逐核匹配相对 IRQ 全集中 CPU4 的收益很小：三路提交的 M1 FFN 改善 0.18%，M128 改善 0.53%。** 匹配后与现有三核单提交路径相比，M1 只改善 0.55%，M128 增加 32.44%。这组结果不支持为亲和性收益改造提交结构。

## 绑定方案与核验

使用单核 canary 实测 IRQ 对应关系，再配置 CPU affinity：

| NPU core | core mask | 实测 IRQ | 三路提交/等待线程所在 CPU | 匹配 IRQ CPU | 循环错配 IRQ CPU |
|---|---:|---:|---:|---:|---:|
| 0 | 1 | 42 | 4 | 4 | 5 |
| 1 | 2 | 43 | 5 | 5 | 6 |
| 2 | 4 | 44 | 6 | 6 | 4 |

三个 canary 分别只引起一个对应 IRQ 的计数增加。正式运行每次 ioctl 读取实际 `sched_getcpu()`，验证 mask1/2/4 的提交分别发生在 CPU4/5/6。M128 pipeline 内部的实际提交 worker 同样绑定，不能仅依赖外层控制线程的 affinity。

现有算子内多核路径使用 mask7，由一个实际提交/等待线程驱动三个 NPU core，该线程固定 CPU4；CPU 量化/归约团队固定 CPU4–7。它没有三条可与 NPU core 一一对应的提交线程。

四种 IRQ 配置：

- `original`：保持初始 requested `0-7`，effective CPU0。
- `all4`：IRQ42/43/44 全部 CPU4。
- `matched`：IRQ42/43/44 分别 CPU4/5/6。
- `rotated`：IRQ42/43/44 分别 CPU5/6/4，CPU 集合相同，刻意打乱对应关系。

对每个非 original 区间，读取 `/proc/interrupts` 增量，确认各 IRQ 只在配置的 CPU 上计数。所有区间的 IRQ CPU 及实际提交 CPU 核验通过。

## 固定配置与计时

同一进程中复用相同权重、workspace 和线程，测试途中只调整 IRQ，不重新调优：

- M1 现有 FFN：gate/down 均 chain，gate CPU threads4、down threads1，Nt4096/896，共两次 submit。
- M128 现有 FFN：panel8/pipeline，CPU threads4，Nt4096/896，共八次 submit。
- 三路 FFN：F=3264/3264/3200，各 rank 一个 CPU 大核和一个 NPU core，gate Nt=F_rank，down Nt2560。M1 使用 chain，M128 使用 pipeline。
- 三路 FFN 在 SwiGLU 后归并全局 row maxabs，使用与未分片 FFN 相同的 A8 scale，最后求和。全局 scale 同步和输出求和计入墙钟。

两个独立进程，每进程、每形状依次测试四种 IRQ 配置再逆序测试；第二个进程反转起始顺序。每个 IRQ 区间做 20 个 `W4→TP→TP→W4` 周期，每种形状/布局/路径共 160 次调用，共 2560 次完整 FFN 调用。

每次切换 IRQ 后先做相同的预热及数值检查。真实、零、放大和负输入检查通过；每种 IRQ 配置下 TP 全局 scale 输出相对基线误差保持约 10^-7，并发/串行相同分片结果逐位一致。计时包括量化、打包、cache sync、提交/等待、归约、激活及最终求和。

## 完整 FFN 结果

| 形状、路径 | original | all4 | matched | rotated |
|---|---:|---:|---:|---:|
| M1，现有单提交三 NPU 核 | 1337.239 µs | 1322.165 µs | 1324.536 µs | 1324.170 µs |
| M1，三路独立提交＋全局 scale | 1334.169 µs | 1317.195 µs | 1314.882 µs | 1317.040 µs |
| M128，现有单提交三 NPU 核 | 8040.693 µs | 8037.753 µs | 8030.776 µs | 8027.380 µs |
| M128，三路独立提交＋全局 scale | 10640.038 µs | 10702.067 µs | 10645.100 µs | 10700.926 µs |

三路提交的 matched 相对 all4：

| 形状 | 汇总时延变化 | 进程 0 | 进程 1 |
|---|---:|---:|---:|
| M1 | -0.176% | -0.123% | -0.228% |
| M128 | -0.532% | -0.972% | -0.085% |

matched 相对同样使用三个大核的 rotated，M1 改善 0.164%，M128 改善 0.522%。相对 original，三路提交的 matched 在 M1 改善约 1.45%，M128 基本相同。大部分 M1 收益来自 IRQ 从 CPU0 移到大核；严格一一对应只贡献很小的额外差异。

现有单提交路径的 M1 all4 比 original 改善 1.13%；matched 反而比 all4 慢 0.18%。此处 M1 使用两次 submit 的 chain 配置，与八次 submit 的 pipeline 探针对 IRQ 开销的敏感程度不同。

比较可采用的两种完整方案：

| 形状 | 现有路径＋IRQ all4 | 三路提交＋IRQ matched | 三路时延变化 |
|---|---:|---:|---:|
| M1 | 1322.165 µs | 1314.882 µs | -0.55% |
| M128 | 8037.753 µs | 10645.100 µs | +32.44% |

## 结论及边界

已有三个独立提交线程时，将对应 IRQ 与线程同核是合理布局，但这批真实 FFN 测试中的额外收益不足 1%。现有算子内多核路径采用集中 IRQ＋固定提交线程即可作为已测配置；这组实验没有证明它对所有工作负载最优。

结果限于第 2 层 FFN 的 M1/M128，以及固定的线程/分块配置。没有新增整模型 TP 或亲和性 E2E 测试，也未将这组局部结果外推为模型 TPS。此前整模型 all4 相对默认 IRQ 的 +2.33% 属于另一个明确标注的测量，见[原报告](IRQ_W8A8_TP_EVAL_20261102.md)。

生产代码未修改。IRQ requested 列表已恢复为 `0-7`，effective CPU0；CPU/NPU/DDR 的 min/max 和 governor 恢复完成。

## 证据与复现

- [原始 CSV 与正确性日志目录](evidence/irq_pair_20261102/paired/)
- [canary 映射、逐区间提交 CPU 与 IRQ CPU 计数、恢复协议](evidence/irq_pair_20261102/paired/protocol.json)
- [原始均值](evidence/irq_pair_20261102/paired/summary.json)、[逐进程比较与核验](evidence/irq_pair_20261102/paired/analysis.json)
- [定频与恢复](evidence/irq_pair_20261102/paired-clocks.json)
- [探针](../native/rk_npu_matmul/tests/idea_eval/probe.cpp)、[控制脚本](../native/rk_npu_matmul/tests/idea_eval/irq_pair.py)、[统计脚本](../native/rk_npu_matmul/tests/idea_eval/analyze_irq_pair.py)

按原报告构建静态库及探针，将输出二进制命名为 `native/rk_npu_matmul/build/idea-eval/probe-affinity`，在板端项目根目录运行：

```bash
python3 scripts/run_board.py --wait-lock --report /tmp/irq-pair-clocks.json -- \
  python3 native/rk_npu_matmul/tests/idea_eval/irq_pair.py \
  --rows 1 128 --cycles 20 --processes 2 --output /tmp/irq-pair

python3 native/rk_npu_matmul/tests/idea_eval/analyze_irq_pair.py /tmp/irq-pair
```
