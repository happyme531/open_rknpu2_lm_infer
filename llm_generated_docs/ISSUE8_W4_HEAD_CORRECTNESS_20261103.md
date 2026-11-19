# W4 输出头默认策略数值排查

2026-11-03，使用作者提供的转换脚本和完整48×512-token语料，在 RKNPU 0.9.8 主板上运行正式 v0.10.0 发布二进制。空缓存与仅含作者 M=1 tune 文件的两组 PPL 均为 **12.453118137984458**，全部12,240个逐 token NLL完全相同，未复现 [issue #8](https://github.com/happyme531/open_rknpu2_lm_infer/issues/8) 的数值异常。

备用板 1 的 RKNPU 0.9.6 另完成267次单算子对照，全部匹配独立 CPU 整数参考；正式发布二进制两次实际模型 head 调用的全部496,640个 logits 也逐位匹配参考。作者已确认问题来自错误转换的权重和旧报告复用，并关闭 issue；生产 fallback 保持原样。

## 作者完整复现材料对照

主板为 `主板`，内核 `6.1.118-zt`、RKNPU 驱动0.9.8；作者板端内核为 `6.1.115-vendor-rk35xx`。CPU4–7 固定2.4GHz，NPU1GHz，DDR2.112GHz。转换使用作者的 `requant_head_w4.py`，仅修改输入和输出路径，保留其 FP32 反量化、`np.rint` 和 safetensors 写法。语料为作者 gist 中的48个512-token窗口。

评估使用官方 v0.10.0 二进制，context2048、ubatch128、4线程、CPU mask0xf0、NPU mask7、score-start257。两组使用全新的报告路径；缓存分别为空目录路径和仅含作者 M=1 文件的目录。

| 缓存 | PPL | Head 缓存命中与缺失 | Head fallback 调用 |
| --- | ---: | --- | ---: |
| 空缓存 | 12.453118137984458 | 0命中，2缺失 | 96 |
| 仅作者的 M=1 tune 文件 | 12.453118137984458 | 1命中，1缺失 | 96 |

作者提供的 `fp5f16f04664f10d70` tune 文件被接受，确认重建 head 与该权重 fingerprint 相符。全部12,240个 NLL逐项相等，最大差为0。两组均正常结束，受控时钟上下限及 governor 均恢复。

这条 PPL 命令实际使用 **M=128 head bucket**。每个窗口的前256个 token 通过 head-free prefill 消费，剩余255个评分输入分成128和127行，两者均执行 bucket128，因此恰好48×2=96次 head 调用。M=1 profile/workspace 在加载时准备，但不参与这些评分矩阵乘法。源码见 `src/ppl.rs` 的 prefill/token_nll 路径及 `src/model/qwen35/runtime.rs::score`。

完整语料结果位于 [author/summary.json](evidence/issue8_20261103/author/summary.json)，两份原始 PPL 报告和时钟报告位于同一目录。

## 单算子和真实调用验证环境

| 项目 | 本次配置 |
| --- | --- |
| 板端 | `备用板 1`，`备用板 1` |
| RKNPU 驱动 | `0.9.6` |
| CPU | CPU4–7，固定 2.256GHz，4 线程；单线程对照使用同一 affinity |
| NPU | 固定 1GHz；默认 mask 7，另含 mask 1 对照 |
| DDR | 板端没有 DMC devfreq 接口，使用 `--skip-dmc`，DDR 未控制 |
| 源码 | 父仓库 `0f29dfe59d5302131680f5443322a42e4aa1be49`，native `af3cbc4bae623978314a5006e8c25bc0ee060a1c` |
| 单算子编译 | 将 native HEAD 导出到独立目录，在板端用 GCC 12.2、Release `-O2` 重建静态库 |
| 发布版验证 | GitHub 发布资产 `open_rknpu2_lm_infer-v0.10.0-linux-aarch64.tar.gz` 中的原始二进制 |

所有硬件阶段通过现有 `run_board.py --wait-lock` 串行执行。有效阶段均 `exit_code=0`、`timed_out=false`、`restore_errors=[]`。内核日志节选未见 timeout、reset 或 IOMMU fault。

## 单算子权重和输入

源模型为 `<HOST_HOME>/mnt/3090x4/llm/w4a8_quant/transfer/qwen35/models/C7`。输出头形状为 `K=2560,N=248320`。从原始 W8 head 解包带偏置的 pack-quantized codes，按输出通道重新量化：

```text
factor[n] = max(abs(q8[n,:])) / 7
q4[n,k]  = clamp(round_away(q8[n,k] / factor[n]), -7, 7)
scale4[n] = scale8[n] * factor[n]
```

全零通道的 factor 使用正数下界。单算子数据存为 `B[K,N]` signed INT8 codes 和 FP32 scales；模型副本将同一组 codes 重新打包为 W4 safetensors，并把 `lm_head` 移到 W4 配置组。原模型文件保持原样。

输入包含合成有限值、已有 Qwen3.5 C7 与 MTP 调试中捕获的真实 `head_input`、全零和全正负极值。M=8 的真实输入由四个已捕获行重复一次组成。来源分别记录于 `head-source.json` 和 `input-source.json`。

独立 CPU 参考按每行完整 K 计算动态 A8 `maxabs/127`、round-away 和 `[-127,127]` clamp，再执行整数点积、输入 scale 与权重 scale 乘法。K=2560 时整数和的绝对值上界小于 `2^22`，INT32 精确容纳全部结果。核对覆盖完整词表，并检查输出前后的 guard。

## 单算子结果

| 对照组 | 显式调用数 | 对比元素数 | 不同元素数 |
| --- | ---: | ---: | ---: |
| 合成权重与输入，M=1 | 24 | 5,959,680 | 0 |
| 重建 W4 head 与合成输入，M=1/2/8 | 72 | 65,556,480 | 0 |
| 重建 W4 head 与真实输入及零值、极值 | 144 | 83,435,520 | 0 |
| M=1 正式调优与默认策略对照 | 27 | 6,704,640 | 0 |
| 合计 | 267 | 161,656,320 | 0 |

每个配置重复三次。八种手工对照分别为默认策略、串行执行、单 CPU 线程、CHAIN、Ntile512、Ntile2048、Ntile4096，以及单 NPU 核。默认配置与 Rust fallback 相同：native 输入、Mtile=min(M,64)、Ntile1024、4 CPU 线程、pipeline=1、NPU mask7。

重建的真实 head 经权重认证后使用 Ktile2048，`bound_relaxed=0`、multiplier1；合成权重的认证 Ktile 为1056。

正式 M=1 调优固定相同的认证 K 分块，使用当前 W4 tuner 产生真实 tune 文件。选中的配置为 Ktile2048、Ntile4096、Mtile1、4线程、CHAIN、mask7。该配置和默认配置对真实捕获输入均逐位匹配 CPU 参考。表中不计入调优器内部的校准与计时调用；本轮没有建立端到端性能结论。

## 发布版实际调用

v0.10.0 发布二进制加载独立 W4 模型副本，使用空缓存路径，context256、ubatch32、temperature0、关闭投机。输入为 `Explain what an NPU does.`，生成2个 token。通过运行时 trace 取得每次实际输入和 logits，再独立计算完整词表的 CPU 参考。

| Head 调用 | 元素数 | 不同元素数 | 相对 L2 | 最大绝对误差 | Top1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| position8 | 248,320 | 0 | 0 | 0 | 271 |
| position9 | 248,320 | 0 | 0 | 0 | 248068 |

运行时统计确认 `lm_head_weight_bits=4`、`tune_cache_hits=0`、`tune_cache_misses=139`、`tune_calls=0`、`tuning_search_enabled=false`。因此上述结果覆盖发布版实际采用默认 head 策略的路径。

## 结案

本轮证据支持：在驱动0.9.6的备用板和驱动0.9.8的主板上，已覆盖的单算子、真实调用和作者完整语料评估均未出现所述异常。

作者随后在自己的0.9.8板端完成了全新报告 A/B，两组 PPL 同为12.453118137984458。他确认最初的转换脚本在 NumPy 2 下错误转换负数，产生了错误 W4 权重；修正脚本后沿用旧的 `--report` 路径，程序返回了旧结果，导致误归因于 M=1 tune 文件。

[作者确认并关闭 issue 的评论](https://github.com/happyme531/open_rknpu2_lm_infer/issues/8#issuecomment-6601791556)。作者建议为已完成报告的复用增加提示；这与 head fallback 数值正确性是独立问题。

## 证据和复现入口

- [汇总](evidence/issue8_20261103/summary.json)、[逐配置结果](evidence/issue8_20261103/native-summary.json)、[发布版逐调用结果与统计](evidence/issue8_20261103/release-summary.json)。
- [单算子探针](../native/rk_npu_matmul/tests/idea_eval/issue8_head.cpp)。
- 原始权重、模型副本、发布包及完整 trace 位于 `<HOST_HOME>/mnt/3090x4/tmp/issue8-20261103/`；板端对应 `<BOARD_HOME>/mnt/3090x4/tmp/issue8-20261103/`。
- 板端工作目录内的 `run-synthetic.sh`、`run-real.sh`、`run-capture.sh`、`run-tuned.sh`、`run-release.sh`、`run-oracle.sh` 分别对应表中阶段；副本保存在证据目录。

在该板端工作目录中，以下命令复用已生成的权重和真实输入，独立比较默认策略、各执行配置与正式调优结果：

```bash
python3 run_board.py --wait-lock --cpu-khz 2256000 --skip-dmc \
  --timeout 300 --report rerun-clocks.json -- \
  taskset -c 4-7 ./issue8_head . real 1 248320 head-inputs.f32 --tune
```

本轮新增诊断探针和报告，未修改生产 fallback、量化器或模型源文件。
