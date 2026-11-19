# 无 mask ViT Attention：备用板子1

内部 query 上限由32行扩大到128行，C7三档完整ViT的本轮中位数加速为1.033×/1.126×/1.140×；features与同板旧路径逐bit相同，没有新增精度偏移。当前实现只扩大stable-softmax、NO_MASK、MHA的合法分块，并保留原生几何和LUT任务预算约束。其他mask/GQA/fixed-shift策略不变。

2026-10-18。独立worktree：`open_rknpu2_lm_infer-attention-nomask`；外层分支`perf/vision-attention-nomask`，算子库分支`perf/attention-nomask`；基点`6143940`/`a9a2e06`。本轮只修改新worktree，原目录中的并行改动保持原样，当前研究分支未提交。

## 同板完整ViT

备用1：`备用板 1`，rknpu0.9.6；4个A76/CPU mask0xf0，2256MHz；NPU mask7、1GHz。DMC无调频节点，DDR未锁频；clk provider的一次读取为2.112GHz。两条路径均为本板GCC12.2与Rust1.95编译，FFN默认融合GELU、FP16张量、FP32 LayerNorm统计与Attention末端CPU归约。

普通GEMM三档21个HOST_STORAGE shape在本板重新调优，包含原有K tiling；后续仅加载缓存，确认21 hits、0 miss、0 search。性能对照共用此缓存，不能用主板0.9.8/CPU2.4GHz的旧数值作分母。

每档按旧/新/新/旧运行4个独立进程；每个预热1次、正式3次，以下是每条路径6个调用的中位数。计时是权重驻留、清空图片特征缓存后的完整encoder，包含全部CPU/NPU阶段；不包含图片预处理、模型加载或文本解码。精度仅比较ViT features及同C7 PyTorch全FP32 gold。

| 输入 | 原32行 / ms | 新128行 / ms | 加速 | 对FP32 gold L2，两者相同 | features |
|---|---:|---:|---:|---:|---|
| 256×256 | 309.115 | 299.376 | 1.033× | 1.8129% | 逐bit相同 |
| 384×256 | 482.435 | 428.430 | 1.126× | 1.0615% | 逐bit相同 |
| 512×512 | 1586.811 | 1391.832 | 1.140× | 2.5484% | 逐bit相同 |

| 输入 | Attention原/新 / ms | down原/新 / ms | 原/新整体MFU | 原/新Attention有效FU |
|---|---:|---:|---:|---:|
| 256×256 | 118.697/92.984 | 66.569/83.146 | 17.83%/18.41% | 1.81%/2.31% |
| 384×256 | 204.956/151.142 | 104.635/104.444 | 17.47%/19.67% | 2.36%/3.20% |
| 512×512 | 883.031/661.018 | 261.594/291.700 | 15.52%/17.69% | 3.89%/5.20% |

MFU沿用3TFLOPS、FMA=2的口径：`F(M)=620756992*M+98304*M²`；Attention分子仅为`24*4*M²*1024`。时间仍包含LUT、CPU归约/打包/同步；padding、分母辅助列等额外硬件工作不计入模型FLOPs。有效FU不是纯MAC利用率。

512² Attention累计883→661ms，下降约25%；全ViT下降约12.3%，Attention仍占新总时间约47.5%。down在部分对照中变慢，尤其小图抵消了Attention收益；down代码与profile未改变，目前不能把该变化归因于某个寄存器或DDR机制。

## Query分块的组件对照

原`query_gqa`无条件`min(rows,32)`。ViT外层每128行一次API调用，内部再拆4段，每段QK和LUT+PV各一次submit。扩大到128后，每图256/384/1024 patch的每层submit数从16/24/64降到4/6/16；数学、LUT表、KV布局和PV归约不变，模型FLOPs及缓存中的逻辑KV数据量不变。

组件包括FP16 KV装载、整图query、同步与最终FP16收集；合成可重复Q/K/V，16 heads、D64、KV tile512。各2个独立进程，预热2次、正式7次，顺序32/64/128/128/64/32。独立NumPy FP32参考在计时之外计算。

| patch数 | 32行 / ms | 64行 / ms | 128行 / ms | 32/128 | 数值 |
|---|---:|---:|---:|---:|---|
| 256 | 4.404 | 3.469 | 3.063 | 1.438× | 全部逐bit相同 |
| 384 | 7.291 | 5.966 | 5.349 | 1.363× | 全部逐bit相同 |
| 1024 | 30.338 | 28.721 | 25.660 | 1.182× | 全部逐bit相同 |

128行的三档相对组件FP32参考L2约0.0511%/0.0516%/0.0507%，与原路径相同。该组件不是实际图片QKV重放，不能替代完整ViT结果。

## 其他参数

同一个支持大query的实验二进制，按128/256/512/512/256/128配对；256/512行只用于隔离探针，没有扩大正式公共API的128行上限。

| patch数 | 128行 / ms | 256行 / ms | 512行 / ms |
|---|---:|---:|---:|
| 256 | 3.037 | 2.877 | 2.892 |
| 384 | 5.100 | 5.793 | 5.020 |
| 1024 | 22.422 | 25.464 | 25.544 |

384个patch采用256行会在256与128行之间反复重建几何；512行的大图虽然NPU时间更低，完整调用反而变慢。当前保留128上限。

在query128不变时，指数LUT每任务元素数4096/16384/32768，以及width128/多channel二维cube均已比较。表值、headroom和PV校正不变，三档输出与原路径逐bit相同。

| patch数 | LUT4096 / ms | LUT16384 / ms | LUT32768 / ms | 二维cube / ms |
|---|---:|---:|---:|---:|
| 256 | 3.188 | 3.201 | 3.061 | 3.163 |
| 384 | 5.349 | 5.323 | 5.325 | 5.250 |
| 1024 | 25.777 | 22.109 | 25.342 | 25.679 |

1024-patch任务数分别4480/1408/896/512；任务数下降没有同比变成时延收益。CPU score扫描块8/16/32/64也做了配对，未找到跨尺寸稳定优于现有128扫描块的方案；均保持数值一致。LUT和CPU扫描块的实验配置不进入当前实现。

## 剩余开销的直接诊断

只在隔离库中给QK提交、score sync FROM、CPU centering、score sync TO、LUT+PV提交加时间计数；热路径不打印日志，结束后一次读取。下表为合成QKV、1024 patch、完整单层的阶段中位数；每格单位ms。

| 阶段 | 原32行 | 128行 | 128行＋LUT16384 | 128行＋二维cube |
|---|---:|---:|---:|---:|
| QK submit/wait | 5.579 | 2.419 | 2.425 | 2.421 |
| score sync FROM | 1.182 | 0.924 | 0.948 | 0.901 |
| CPU score centering | 5.107 | 5.755 | 5.008 | 7.995 |
| score sync TO | 1.899 | 1.502 | 1.541 | 1.498 |
| LUT+PV submit/wait | 10.128 | 6.716 | 6.327 | 6.204 |

QK/PV两格含提交和等待，不能视作纯设备MAC时间。128行中，CPU centering约5.75ms，score同步约2.43ms，合计约完整组件时间的36%；QK与LUT+PV提交等待合计约9.14ms。进一步合并LUT主要只减少其中较小一段；二维cube配置甚至出现centering更慢的跨阶段变化。没有据此断言CPU算法变了，也未将这些合成组件百分比当成真实图片的逐项占比。

## 当前实现与验证

- 只对STABLE_SOFTMAX＋NO_MASK＋Hq=Hkv尝试最多128行；CBUF几何不支持时沿用缩小分块。
- 选择大于32行时检查QK/PV任务加LUT任务不超过4095，预留可选每核LUT setup；长序列会缩小到64/32，避免新引入CPU exp回退。旧32行本就需要回退的情况保持原策略。
- causal、boolean、GQA与fixed-shift的分块策略保持不变；不添加环境变量、模型参数或新的CPU数学实现。
- native CPU sizing/预算测试通过；原generic-GQA板测覆盖D32/64/96/128/160/192/256、不同GQA组、grow/truncate/append/reset和mask，全部通过。
- mask1/2/4/3/7、FP16/FP32输出、M3/M129尾块均通过独立FP32参考与重复性检查。大V=586.5时按原策略回退CPU exp，两种输出类型均通过，relative L2约0.0327%。
- 最终带预算保护的重编ViT三档输出与旧基线逐bit相同；352×224、14×22网格（308个patch）的完整ViT尾块对照也逐bit相同。
- 21个同板GEMM profile仅加载为21 hits/0 miss/0 search。所有板端runner频率上下限及governor恢复，`restore_errors=[]`。

源码入口：算子库`src/rk_npu_attention_f16.cpp`的`query_gqa`，CPU contract在`tests/test_attention_f16.cpp`；外层Rust不用改。可复用组件探针是`tests/board/attention_nomask_probe.py`，普通模型计时仍用`vision_profile`。

```sh
# 备用板1的共享工作树内；工具链见logs/build-vision-final.sh。
cmake --build build/attention --target test_attention_f16 -j4
build/attention/test_attention_f16 --cpu
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python3 scripts/run_board.py \
  --wait-lock --cpu-khz 2256000 --skip-dmc --report logs/repeat-clocks.json -- \
  taskset -c 4-7 python3 tests/board/attention_nomask_probe.py \
  --library build/attention/librk_npu_matmul.so --output logs/repeat.json
```

完整ViT配对命令在`logs/vit-rows128.sh`，同板调优与加载确认在`logs/prepare-matched.sh`、`logs/load-only.json`。隔离任务几何生成器和编译产物保留于本worktree的`logs/`；小型原始JSON、逐样本时延、CPU/板测结果和频率恢复记录归档于[evidence/attention_nomask_spare1_20261018](evidence/attention_nomask_spare1_20261018/)。

## 限制

DDR未锁频，进程之间有波动；尤其小图的端到端收益较小，当前不足以给出显著性结论。每条路径只有两个独立性能进程，进程内3次调用不当作6个独立实验。不同阶段的内存/缓存交互尚未单独归因；任务数、驱动load和模型有效FU不能混用。

一维65536元素LUT在1024-token输入上触发一次NPU超时与驱动软复位，该配置不进入有效路径。之后core mask1/2/4稠密FP16 identity GEMM各重复3次精确输出，恢复检查及后续板测全部通过。TRM宽度字段的13-bit范围不等于完整数据通路支持该配置。

本次等价调度没有新增features偏移；局部/最终features一致仍不替代视觉任务准确率。当前未提交，未修改原工作树或推送新分支。
