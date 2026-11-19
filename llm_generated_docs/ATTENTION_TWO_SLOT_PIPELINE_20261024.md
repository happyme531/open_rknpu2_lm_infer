# Attention 双 slot 流水试验

这个方向在多块 prefill 上成立。相对已提交的 `6176201`，Qwen3 Q128/KV4096 的完整热态 Attention 调用下降18.69%，C7下降25.38%；计入128行 KV append 后分别下降19.18%/23.54%。相同双slot布局的串行控制组也显著慢于流水组，收益有实际时间线支持。

当前交付是可运行的实验原型，正式库仍为 `61762012b0a145e2286169526f1abf142695cb11`。没有接入之前的 QK 重算、扩维 K 或设备 ReduceMax 方案，没有修改 LUT/数值配方，也没有将原型提交或默认启用。此次目标是验证用户提出的双slot方向，完整模型集成与E2E收益尚未验证。

## 实现及依赖

代码位于 [probe.cpp](../native/rk_npu_matmul/tests/tools/attention_slots/probe.cpp)，控制脚本为 [run.py](../native/rk_npu_matmul/tests/tools/attention_slots/run.py)，证据校验与绘图为 [summarize.py](../native/rk_npu_matmul/tests/tools/attention_slots/summarize.py)。算术直接复用库内 `Graph::pack_q/stable_softmax/submit/finish`，公共串行入口作为数值基准。

- 两个独立slot各自持有Q、scores、exponent、PV partial、寄存器/描述符、CPU scratch与partial KV tail。程序实际检查了两个slot全部GEM地址范围不相交。完整K/V和前缀checkpoint共享只读，没有额外复制完整KV。
- 每个slot保留自己的 `Workspace` 与tail staging元数据。实验通过临时装入/恢复cache tail调用既有staging实现；实际NPU任务引用的是各自独立tail地址。cache busy锁覆盖完整调用，更新只在所有任务结束后执行。
- 一个持久工作线程执行原有阻塞submit；任意时刻最多一个NPU submit在运行，不依赖未经验证的异步fence ABI。CPU scores使用原有四线程，整个进程仍限制在CPU4–7，没有增加可用CPU核。
- 对一对query块：先完成QK(0)，然后QK(1)覆盖CPU scores(0)，PV(0)覆盖CPU scores(1)，下一块的LUT/PV阶段覆盖finish(0)，后续QK覆盖finish(1)。奇数尾块按同一依赖收尾。
- 先对已完成块执行score FROM_DEVICE，再启动下一块QK。初版在长KV上让QK完全落在invalidate区间内；调整后，QK确实覆盖实际centering，而不仅是cache同步。
- 每个分段仍经过原有 `causal_rows()` 数值模式规划，保留1024、V幅值和descriptor边界；预加载未来KV不能改变可见前缀的分派和结果。

## 重叠证据

所有正式样本关闭阶段记录，另跑独立trace。图的两条轨道按逻辑阶段区分；NPU轨道是阻塞submit的用户态调用区间，灰色还包含worker上的future-carrier清理，不把整条轨道当作纯硬件执行。

![同一时间尺度下的串行与流水阶段](evidence/attention_slots_20261024/timeline.png)

在一份最终Qwen3 4K检查trace中，QK(1)的完整849.923µs submit落在CPU scores(0)的区间内，driver返回840.006µs设备区间；QK(3)也完整落在CPU scores(2)内。另有2.452ms的实际PV submit与另一块CPU scores重叠，以及1.493ms的finish与另一块QK/LUT/PV submit重叠。原始时间戳见 [summary.json](evidence/attention_slots_20261024/summary.json) 和各trace。

同布局串行/流水的独立对照trace中，首个到末个submit窗口内部、相邻阻塞submit之间的host空窗如下。这是单次归因trace，不能替代配对性能统计，也不是纯MAC利用率：

| Q128/KV4096 | 同布局串行空窗 ms | 流水空窗 ms | 串行driver区间合计 ms | 流水driver区间合计 ms |
|---|---:|---:|---:|---:|
| Qwen3 | 11.272 | 5.950 | 11.335 | 12.192 |
| C7 | 9.014 | 2.958 | 8.143 | 8.412 |

设备区间没有缩短，完整调用仍提速；证据支持“提前产生就绪任务并隐藏host工作”，而不是GEMM变快。CPU与NPU并发访问DDR仍可能拉长单个阶段。

## 完整调用配对结果

板端为 `备用板 1`，kernel5.10.198、RKNPU driver0.9.6（本轮实时读取），CPU4–7/四线程2.256GHz，NPU mask7/1GHz；DDR没有DMC控制，仍未受控。每个几何三轮 `A B C C B A`，每进程预热4次、正式12次；每方法每几何72个正式样本，共2808次。表中是各进程median的均值。

A是公共串行run；B与C均使用双slot、相同分块和scratch布局，B串行调度，C流水。定时包含规划、geometry/tail更新、pack、计算、同步、归一化和finish；不含首次KV load、线程创建、磁盘输出和正确性检查。Q128/KV4096不是完整4096-token prefill。

| 几何 | A公共串行 ms | B双slot串行 ms | C流水 ms | C相对A延迟下降 | C相对B延迟下降 |
|---|---:|---:|---:|---:|---:|
| Qwen3 Q128/KV512 | 5.324 | 5.644 | 4.092 | 23.13% | 27.49% |
| Qwen3 Q128/KV4096 | 25.920 | 25.099 | 21.075 | 18.69% | 16.03% |
| C7 Q128/KV512 | 3.940 | 4.009 | 2.736 | 30.56% | 31.75% |
| C7 Q128/KV4096 | 17.278 | 17.200 | 12.892 | 25.38% | 25.05% |
| C7 Q4/KV4096 | 1.286 | 1.285 | 1.302 | -1.24% | -1.31% |
| MHA H16/D64 Q128/KV256，流水块64 | 0.917 | 1.065 | 0.830 | 9.48% | 22.06% |
| MHA H16/D64 Q128/KV1024，流水块64 | 2.406 | 2.981 | 2.142 | 10.99% | 28.14% |
| MHA H16/D64 Q128/KV1024，流水块32 | 2.426 | 3.616 | 2.565 | -5.75% | 29.05% |
| Qwen3 Q128/KV4096，CPU exp | 25.616 | 25.689 | 23.683 | 7.55% | 7.81% |
| 真实C7捕获，prefix1021/Q4 | 0.487 | 0.494 | 0.496 | -1.72% | -0.38% |
| Qwen3 decode Q1/KV4096 | 0.868 | 0.869 | 0.886 | -2.05% | -1.99% |
| Qwen3 append128 + Q128/KV4096 | 23.871 | 25.170 | 19.293 | 19.18% | 23.35% |
| C7 append128 + Q128/KV4096 | 16.516 | 16.214 | 12.628 | 23.54% | 22.12% |

append两行在每次定时前rollback到固定历史前缀，定时内执行真实bulk append128及完整Attention；rollback不计时，不把它当作生产开销。每次epoch变化均强制刷新两份tail，避免只测静态cache复用。Qwen3 append三个配对块分别下降18.73%/19.83%/18.94%，C7为27.42%/19.84%/22.97%。各轮和原始样本全部保留，不与前一天批次直接相减归因。

MHA的公共A使用既有128行起始策略；B/C显式限制为64或32行。因此B用于分离分块代价与流水收益：32行虽然比其自身串行快，仍不及原公共128行。64行是本组形状的较好折中，不是通用最优值。

真实C7捕获使用已有 `phase2-p1021-r4-{q,k,v}.f16`；其余输入为固定seed生成。真实捕获、Q4 verify和Q1 decode在这里都只有一个分段，线程交接不能产生有意义的跨块重叠，按回退/持平看待，不能据此宣称decode提速。

## 正确性与成本

最终28项板端用例全部与公共串行入口逐字节一致，包括输出guard、FP16输出、三/五块和尾行、跨partial-tail边界、预加载未来高V、1024分派边界、V=65000归一化fallback、关闭LUT、Boolean全遮罩、truncate/append/reserve、core mask1/2/4/3/7、split/pingpong和真实C7捕获。formal的13个几何也逐进程比较全部输出。末次原库Attention/INT8交替canary通过，四份最终时钟报告均 `exit_code=0, restore_errors=[]`。

初次批量runner未继承锁FD，在进入NPU测试前报错；修正 `pass_fds` 后全量重跑。早期日志和调整顺序前的trace保留在共享目录 `attempt1/attempt2/`，不混入最终统计。

Qwen3 4K双slot的GEM scratch合计约38.88MiB，C7约21.72MiB；这里仅统计两份slot的原生Q/score/exponent/partial/寄存器/描述符/tail，不含共享K/V、CPU vector、线程栈及独立公共基准workspace。长KV GQA的主要scratch接近原来两份，这是本方案的明确成本；没有扩维K。线程创建和冷构图开销未纳入热态结果。

## 状态与复现

该原型已验证提出的两slot流水机制及其收益边界。正式库入口仍保持 `6176201`；完整模型prefill、logits/PPL、长期并发与内存生命周期尚未进行这版原型的集成验证，不能把上述百分比当作完整模型或ViT encoder提速。

板端快照为 `<BOARD_HOME>/mnt/3090x4/tmp/attention-slots-20261024`，主机映射为 `<HOST_HOME>/mnt/3090x4/tmp/attention-slots-20261024`。它链接上一轮清理版的静态库；构建命令、运行命令、原始样本、完整输出哈希和trace均归档于 [evidence/attention_slots_20261024](evidence/attention_slots_20261024/)。可用归档中的 `build.sh/checks.sh/bench.sh/traces.sh` 复现；三个probe模式分别为 `--mode 0/1/2`。trace另跑，正式计时不带trace。
