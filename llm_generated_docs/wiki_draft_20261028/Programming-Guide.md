# 第二部分：怎样给 RKNPU2 写一个新 kernel

[首页](https://github.com/happyme531/rknpu2_matmul_open/wiki) · [硬件行为](https://github.com/happyme531/rknpu2_matmul_open/wiki/Hardware-Behavior) · [编程方法](https://github.com/happyme531/rknpu2_matmul_open/wiki/Programming-Guide)


第一部分讲硬件"是什么样"，本文讲"怎么做"。这里的方法是从本项目几十个 kernel/probe 里总结出来的：
INT8/FP16/BF16/TF32/INT4 GEMM、split-K 归约、requant/dequant 融合、Add+RMSNorm、Softmax、
SiLU/GELU LUT、stable Attention、gather/复制、KDA decode 和分块 GDN prefill 等。

内容基线：2026-10-28，公开库提交
[`3396f77`](https://github.com/happyme531/rknpu2_matmul_open/commit/3396f77b810a3888684f8648703c4592ea8b3329)。

核心方法：**把问题改写成硬件原本就会做的事（卷积 + 固定后处理），
从已经验证过的配置出发，一次只改一个变量，用能区分对错的数据验证数值与状态，
最后用完整调用时间而不是硬件时间判断值不值得。**

---

## 1. 心智模型

### 1.1 kernel 是什么

在这个 NPU 上，"kernel" 不是一段在 NPU 上执行的程序，而是：

```text
一组 task（每个 task = 一串寄存器写入 + 4 qword 的 PC 尾）
  + 已按硬件布局打包好的数据（A、B、系数、LUT）
  + 按数据依赖安排的一次或多次 submit、同步和 CPU 阶段
```

每个 task 就是"配置一次固定流水线并启动它"。所以写 kernel 要回答的问题是：

1. 这个计算能拆成哪几次**流水线调用**（GEMM/卷积、带后处理的 GEMM、DPU-only 逐元素/归约、复制）？
2. 每次调用的**数据布局**是什么？相邻两次调用能不能零拷贝衔接？
3. 哪些部分留给 CPU 更划算？

### 1.2 硬件能直接做的"原语"

| 原语 | 用什么实现 | 参考实现 |
|---|---|---|
| 矩阵乘 / 1×1 卷积 | CNA+CORE+DPU，op_en `0xd` | `rk_npu_matmul_i8.cpp`、`rk_npu_matmul_f16.cpp`、`rk_npu_i4_regs.cpp` |
| GEMM + 逐通道 scale/bias/requant | 同一 task 里 DPU 的 BS/BN/OUT_CVT | DPU BS/BN 与 OUT_CVT 配置 |
| GEMM + 逐元素张量运算 | flying EW + 双 surface RDMA，op_en `0x1d` | 本库 FP16 fused MUL/ADD |
| GEMM + 激活 | 同一 task 用 LUT，装表可放在首 task 的前缀 | 融合 bias/GELU，`rk_npu_mm_gelu.cpp` |
| 逐元素 A∘B（张量） | offline DPU：MRDMA 主 + ERDMA 操作数，op_en `0x18` | INT32 add、FP32 BS/EW |
| 逐通道缩放/偏置 | BS/BN 外部系数（BRDMA/NRDMA） | dequant row task |
| 标量缩放/偏置 | BS/BN 寄存器操作数 | 同上 |
| 行最大/最小 | DPU `minmax_ctl` | Softmax ReduceMax |
| 行求和 / 均值 | CNA/CORE 全 1 权重卷积 | Softmax、RMSNorm |
| 标量→行广播 | CNA/CORE 单位/稀疏权重卷积 | Softmax、RMSNorm |
| 非线性函数 | LUT（513×2 项），或特定函数的位运算近似 | stable Attention exp、GELU、SiLU；历史 rsqrt 近似 |
| 复制、改布局、按 CPU 给的地址 gather | offline DPU 纯 bypass，每行一个 task | `copy_i8_body` |
| 行抽取 | MRDMA `SURF_NOTCH` 跳行 | dequant row task |

**本项目尚未获得通过验证配置的功能**：根据数据值计算地址的动态 gather/scatter、scatter-add、
ERDMA 标量广播、保留完整 FP32 精度的乘数、任意转置、跨 task 的 CACC 续加。
遇到这些要改写算法，或者留给 CPU。

---

## 2. 先判断值不值得上 NPU

先用第一部分 §11 的历史样本估算，再以实际板端测量修正模型。若预估成本高于 CPU 基线，
应先检查驻留、流水线或算子融合是否能改变数据搬运量。

### 2.1 GEMM 类

```text
t_npu ≈ Σ_task (B_tile_bytes / measured_B_bandwidth) + 流水线开销
       + submit (~20–30 µs) + CPU 打包/量化/sync/后处理
```

- M 很小（decode）时 B 流量决定一切：`M=1` 的 GEMM 基本等于"把权重读一遍"，
  应比较相同操作数、压缩格式、核数与频率下的完整调用；多核并行和释放 CPU 容量可能有价值。
- K 很大时先算 Mtile（CBUF 预算），再判断要不要沿 K 拆分。
- N 太小（≤32）时固定开销占主导。

### 2.2 非 GEMM 类

- 逐元素或归约（Softmax、Norm、量化）：每个 task 要读写一遍 DDR，
  链内的额外 task 虽然只有约 0.3–0.9 µs，但**每多一个中间张量就多一次 DDR 往返**。
- 对比基线必须是**优化过的 NEON/OpenMP**，NumPy 可以用于功能对照，但不能代表优化后的 CPU 性能上限。本项目多次出现"比 NumPy 快，比 NEON 慢"。
- NPU 的价值往往在于：张量本来就在 NPU 上（省掉一次 CPU 往返）、CPU 已经满载、或 M 足够大能摊薄开销。

### 2.3 必须算进来的隐藏成本

- CPU 打包/量化：动态量化加 native 打包，在 `K=8192` 时累计几百 µs（可以与 NPU 重叠）。
- cache sync：约 58 µs/MiB。
- DPU task 与 CORE 串行：放在 NPU 上的后处理会推迟下一个 GEMM。
- 共享 DDR：CPU 和 NPU 同时运行时双方都会变慢（CPU 归约 1.2 ms→2.1 ms）。
- LUT：区分独立装表提交、嵌入首 task 的表前缀，以及 CPU 构图成本。
  当前 Attention 的链内装表增量是微秒量级，不能拿历史约 60 µs 的独立装表值乘以调用次数。
- 流水 scratch：重叠两个块通常需要独立输入、中间结果、描述符和尾块缓冲；共享只读权重或 KV
  可以减少复制，但不能让在途 task 的地址指向已被下一个块覆盖的数据。

---

## 3. 开发流程

### 步骤 1：把算法改写成硬件形状

- 能做成 GEMM 的就做成 GEMM，而且**尽量做成大 GEMM**。KDA 把多个 head 放进块对角的 B，
  利用原本就要补零的 K/N 位置，任务数从 64 降到 21。
- 卷积的空间维就是矩阵的 M，通道维是 K/N。M 可以映射成 `H×W` 的二维形状，
  换一种排布就能改变访存模式（panel 布局在部分形状上快 1.2–1.6 倍）。
- 标量广播、行求和都用卷积做；"除以行和"只能写成 `main / external` 的顺序（没有操作数交换位）。
- 利用代数重排把运算塞进已有 task：RMSNorm 把 gamma 乘法并进广播 ConvMul；
  Softmax 把"减最大值"改写成 `BS×(-1)` 加 EW Add，从而和广播合并。
- 允许消费者完成数值修正：Attention 让 LUT 输出 `g*exp(x)+delta`，由 PV 后的去偏置和归一化
  抵消公共增益；GELU 用 FP32 buffer 承载 Q13 表值，在输出收集时还原倍率。

### 步骤 2：确定每条边的 dtype 和布局

- 画出数据流图，每条边标 dtype、native 块大小（K8/K16/N4…）、M/N 的 padding。
- 首选**生产者的输出布局正好是消费者的输入布局**（INT8 requant→native A、FP16 native D→native A、
  QK 输出→Softmax→PV）。这是 NPU 能赢 CPU 的主要来源之一。
- 注意 K 对齐：INT8/FP16/BF16/INT4 GEMM 的 K 按 32 对齐，TF32 GEMM 按 16 对齐；FP16 紧凑张量常以 8 通道为 atom。
  可以用 `CNA_DATA_SIZE1` 高/低 16 位分离"DMA 跨度"和"计算 K"来衔接（见第一部分 §6.2）。

### 步骤 3：选择最接近的已验证配置

优先从能检查到具体实现和适用范围的材料开始：

1. **本库寄存器生成器**：[INT8 GEMM](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/src/rk_npu_matmul_i8.cpp)、
   [FP16/BF16/TF32 GEMM](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/src/rk_npu_matmul_f16.cpp)、
   [INT4 GEMM](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/src/rk_npu_i4_regs.cpp)、
   [MoE 行复制与 DPU 后处理](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/src/rk_npu_moe_regs.h)
   以及 [INT8 split-K DPU 链](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/src/rk_npu_i8_dpu_chain.cpp)。
   非线性融合可参考 [Attention LUT](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/src/rk_npu_attention_f16_lut.h)
   和 [GEMM+bias+GELU](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/src/rk_npu_mm_gelu.cpp)。
   先确认 dtype、输入/输出布局、CBUF 预算、模块使能和 submit flags，再选择相近配置。
2. **Mesa Rocket**：[寄存器生成实现](https://github.com/chaotic-cx/mesa-mirror/blob/main/src/gallium/drivers/rocket/rkt_regcmd.c)
   和 [寄存器定义](https://github.com/chaotic-cx/mesa-mirror/blob/main/src/gallium/drivers/rocket/registers.xml)。
   与本库的布局和驱动契约逐项对照，不直接套用不同几何的参数。
3. **TRM**：用于查位域、编码和规定的配置顺序。没有板端记录的组合仍需从最小测试验证。

已有配置只保证原来验证的条件。改变几何、数据类型、前驱 task 或驱动后，重新检查状态依赖和数值行为。

### 步骤 4：做最小 canary，一次只改一个变量

- 从能工作的配置开始，形状取最小（M=4、K=N=32/64）。
  M<4 在部分路径上是特殊情况，M=1 常常需要单独的配置。
- **一次只改一个寄存器或字段**，每次都上板。已有配置中的关键参数通过单变量对照检查：
  EW cvt 位、32 bit 槽的低 16 位、双 surface、submit flags。
- 新配置先在 1 个 task、1 个 core 上跑通，再扩展到链、M/N tile、多核。

### 步骤 5：用能区分对错的数据验证

**恒等数据不能单独证明操作数生效**：乘 1、加 0、单位矩阵都无法区分"运算生效"和"操作数被忽略"。

- **判别性数据**：非平凡的 `A@B` 加非零常数操作数，或者让每个元素都不同（`buf[i]=i` 的 marker）。
- **marker 验证布局**：令 A=0（或 B=单位阵）、操作数 `buf[i]=i`，读回输出就能看到每个元素来自哪里。
  FP16 half-surface 布局、BRDMA 系数的 64 字节分块、INT32 半覆盖问题都是这样定位的。
- **参考计算要模拟硬件的舍入**：TF32 截断低 13 bit、FP16 系数、FP16 分阶段舍入。
  整数路径要求逐位一致。浮点路径先定义累加精度、操作顺序和舍入边界；
  可逐位复现时检查逐位一致，否则使用明确的误差容限，并另行报告相对理想 FP32 的误差。
- **边界覆盖**：M/N/K 尾块、M mod tile、极值（-128、±65504）、全零、正负抵消、重复索引。
- **重复运行**：同一个 prepared chain 连跑几百到上千次，还要 A/B/A 交替输入。
  有的配置第一次正确，重复后才出错（panel 布局、内存复用 bug）。
- **跨算子转换**：在目标 kernel 前后运行完整初始化的 INT8/FP16 canary，尤其覆盖 LUT、DCOMP
  和精度切换。INT4 N8192 的自身输出正确，却影响后续 INT8；单算子 oracle 无法发现这种问题。
- **局部误差与模型误差分开**：LUT 近似或浮点 K 分块先核对算子，再看最终 features/logits/PPL。
  相对旧实现的误差与相对独立高精度参考的误差分别报告。
- **guard**：每个输出缓冲后面填 `0xa5`，每次运行后检查。
- **先跑 NPU，再跑 CPU 参考**，避免 CPU 先写出的正确值掩盖 NPU 没写的问题。

### 步骤 6：扩展到真实形状

- **M tile**：按 CBUF 预算计算（第一部分 §6.3），不要超出已验证的 bank 数。
- **N tile**：整块 N 可作为基线；多核拆分、notch 位宽或其他几何限制可能需要 N 分块。
  各候选应比较包含布局转换和同步的完整调用时间。
- **K 拆分**：`Kt` 取让 Mtile 达到上限的值（INT8 normal、M=128 时为 2048）。
  INT8 可用 CPU 精确归约并与下一波 NPU 重叠；FP16 还需比较设备侧 partial/归约的完整调用成本，
  并检查分块写回及每层归约的舍入。非线性必须放在所有 K partial 合并之后。
- **多核**：按 N 均分成连续的 task 区间，每个 core 一条封闭的链；
  本库使用 mask 1/2/4/3/7。核数不能写死：历史样本中 `N=128` 单核较快，
  较大的 N 才可能摊薄多核开销；阈值还受 M/K、布局和 CPU 配置影响。
- **长链**：所有 body 放在同一个 regcmd buffer，所有数据放在同一个 IOMMU domain。

### 步骤 7：生产化

- **prepare 一次，运行时只 patch 地址**：建 plan 时 DMA 地址填 0，记下
  `CNA_FEATURE_DATA_ADDR / DCOMP_ADDR0 / DPU_DST_BASE / RDMA_SRC_BASE / RDMA_EW_BASE`
  所在的 qword 下标，run 时只改这些字（`find_patch_indices()`）。
  不要读 non-cacheable 的 regcmd 来做读-改-写，直接覆盖整个 qword。
- 尾部链接（PC 尾）在 prepare 时生成；多核重新分区时重建所有尾，避免残留旧的结束尾。
- 正确性测试可预填 marker 检查完整写回；若实现保证覆盖全部有效输出，性能测量避免在每次调用前无谓清零。
- 数据缓冲用 cacheable（0x13）并显式 sync；CPU 的中间结果不要写进 non-cacheable 内存。
- 模板的 relocation 只允许出现在白名单里的地址寄存器上。
  "寄存器值落在旧 BO 地址范围内就当成地址"会把恰好相等的常数也改掉。
- 内存复用：graph 输入默认保活到结束，只有显式 donate 才能复用。
- 需要 autotune 时，先建立独立参考和数值契约。整数候选要求精确一致；浮点候选允许改变 K 分块时，
  应使用明确的容差与高精度参考，不能要求任意累加顺序都逐位相同。
  任何 submit 失败都立即停止本次 tune。最终决胜用**包含打包/同步/后处理的完整调用时间**。
- 将搜索与执行分开：准备阶段显式调优并上传所需权重布局，运行阶段用常驻权重及 load-only
  profile。W4 的 immutable weight profile 可在释放原始 CPU codes/scales 后继续查找新 M 的策略。
  热路径不扫描权重、不临时调优；测量时记录 hit、miss 和实际 fallback。

---

## 4. 代码骨架

### 4.1 C++：基于库内部接口

下面的辅助函数展示单核 GEMM/DPU 链的组织方式。调用者提供完整 body 和每个 task 的
模块使能值，并负责 A/B/C 缓冲的分配、地址绑定、缓存同步与生命周期。
它使用私有 `src/` 接口，应与同一版本的库一起编译。

```cpp
#include "rk_npu_internal.h"
#include <cstring>

using namespace rknpu2_matmul_open::detail;

int submit_bodies(rk_npu_ctx* ctx, rk_npu_iommu_domain* domain,
                  const std::vector<std::vector<uint64_t>>& bodies,
                  const std::vector<uint32_t>& enables) {
    if (!ctx || !domain || bodies.empty() || bodies.size() > 4095 ||
        bodies.size() != enables.size()) return RK_NPU_ERR_PARAM;
    if (domain->ctx != ctx) return RK_NPU_ERR_DOMAIN;

    std::vector<int> base;
    uint64_t qwords = 0;
    constexpr uint64_t max_qwords = UINT32_MAX / 8;
    for (size_t i = 0; i < bodies.size(); ++i) {
        if (bodies[i].empty() || bodies[i].size() > max_qwords - 5 ||
            (enables[i] != 0xd && enables[i] != 0x1d && enables[i] != 0x18))
            return RK_NPU_ERR_PARAM;
        const uint64_t slot = (bodies[i].size() + 5) & ~uint64_t(1);
        if (qwords + slot > max_qwords) return RK_NPU_ERR_PARAM;
        base.push_back(static_cast<int>(qwords));
        qwords += slot;  // body + 四个 PC 尾字，向两个 qword 对齐
    }

    rk_npu_mem cmd{}, tsk{};
    int rc = rk_npu_mem_alloc(domain, qwords * 8, RK_NPU_MEM_NON_CACHEABLE, &cmd);
    if (rc != RK_NPU_OK) return rc;
    rc = rk_npu_mem_alloc(domain, bodies.size() * sizeof(rknpu_task),
                          RK_NPU_MEM_KERNEL_MAPPING, &tsk);
    if (rc != RK_NPU_OK) {
        rk_npu_mem_free(ctx, &cmd);
        return rc;
    }
    std::memset(tsk.vaddr, 0, static_cast<size_t>(tsk.size));
    auto* commands = static_cast<uint64_t*>(cmd.vaddr);
    auto* tasks = static_cast<rknpu_task*>(tsk.vaddr);
    write_chain(commands, tasks, cmd.dma_addr, bodies, base, {0xd, 0, 0xd});
    for (size_t i = 0; i < bodies.size(); ++i) {
        commands[base[i] + bodies[i].size() + 3] =
            E(T_PC, R_OPERATION_ENABLE, enables[i]);
        tasks[i].op_idx = enables[i] == 0xd ? 0 : 1;
        tasks[i].enable_mask = enables[i];
    }
    rc = do_submit(ctx->fd, tsk.obj_addr, static_cast<int>(bodies.size()),
                   domain->id, 2000);
    rk_npu_mem_free(ctx, &tsk);
    rk_npu_mem_free(ctx, &cmd);
    return rc;
}
```

`do_submit()` 的 flags 固定为 `PC|PINGPONG`，失败后会请求 reset。
它不适用于需要 PC-only 的初始化链。此类诊断应使用自己明确控制 flags 和恢复行为的封装，
失败后停止测试并保留日志。多核提交还需要在 `write_chain()` 生成时分别封闭各核的链，
不能把以上单核链直接当成多核描述符提交。

### 4.2 Python：检查编码和布局

Python 适合生成 marker、计算参考布局和检查 64 bit 寄存器命令。
下面的例子只验证编码；它不包含设备打开、DMA 分配、缓存同步或 NPU 提交。

```python
def regcmd(target, offset, value):
    if not (0 <= target < (1 << 16) and 0 <= offset < (1 << 16)
            and 0 <= value < (1 << 32)):
        raise ValueError("register command field outside encoded range")
    return (target << 48) | (value << 16) | offset

# DPU S_POINTER，指令格式见第一部分 §2.3。
word = regcmd(0x1001, 0x4004, 0xE)
assert word == 0x10010000000E4004
```

板端执行仍需遵循第 2、3 节的内存与链契约。已有可构建、可运行的入口见
[测试指南](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/tests/README.md)。

### 4.3 一个 offline DPU task 的完整模板

下例列出已测试的 FP32 native `[C/4, rows, 4]` 逐元素加配置。
修改数据类型、几何或前驱模块时，仍须检查完整初始化和共享状态；C 按 4 通道填充。

```text
DPU  0x4004=0xe  0x400c=0x1e5(offline)  0x4010=0xb4000005(FP32 in/proc/out)  0x4014=0
     0x4020=dst  0x4024=rows*16  0x4030=rows-1  0x4034=0  0x4038=0  0x403c=(C-1)<<16|(C-1)
     0x4040=0x53 0x4044=0 0x4048=0 0x404c=0  0x4050=0x2  0x4054=0  0x4058=C-1  0x405c=rows-1
     0x4060=0x53 0x4064=0 0x4068=0 0x406c=0
     0x4070=0x10c203c0(EW ADD, 32bit, cvt bypass) 0x4074=0 0x4078=1 0x407c=0
     0x4080=0 0x4084=1 0x4088=0  0x4090..0x40ac=0  0x40c0=rows*16  0x40c4=0
RDMA 0x5004=0xe  0x500c=rows-1  0x5010=0  0x5014=C-1  0x5018=src_a
     0x501c=0 0x5020=0 0x5028=0 0x502c=0
     0x5034=0x4000000c(per-pixel 32bit)  0x5038=src_b  0x5040=rows*16
     0x5044=0x2f8a1(FP32, offline)  0x5048=0 0x504c=0 0x5064=0 0x5068=0x01010101 0x506c=0
op_en = 0x18
```

改成乘法时，把 `0x4070` 改成 `0x10c003c4`，操作数改成 FP16-in-32bit 槽；
改成逐通道广播时，把 `0x5034` 改成 `0xc`、`0x4070` 的 data_mode 改成 0、`0x5040=16`。

---

## 5. 常用技巧

| 技巧 | 要点 | 对应实现或用途 |
|---|---|---|
| 沿 K 拆分 + CPU 归约 | 让每一波的 Mtile 达到上限，减少 B 重读；partial 与下一波重叠 | `rk_npu_kn_plan.cpp` |
| 输出 native C | 大 N 下是最大的单项性能收益 | CPU 布局转换与 GEMM 配置 |
| requant 串接 | INT8 GEMM 直出 `[N/16,M,16]`，下一个 GEMM 直接读 | DPU requant 与下一算子的布局契约 |
| 块对角 B 批处理小矩阵 | 把多个 head 塞进一个 GEMM 的补零区域 | KDA 多头批处理 |
| 卷积做广播和求和 | 稀疏单位权重广播，全 1 权重求和（BF16 求和可提高精度） | Softmax/RMSNorm |
| 每行一个 task + patch 寄存器 | 行尺度写进 BS_MUL 寄存器，每次运行只改 M 个字 | dequant row task |
| MRDMA notch 抽行 | `SURF_NOTCH=(M-1)*16` 从 native 布局抽一行 | dequant、MoE |
| DPU 当 DMA 用 | 纯 bypass 复制，目标 stride 决定 normal/native | `copy_i8_body` |
| 位运算近似 | FP16 exp、BF16 rsqrt 种子，用整数加乘操作位模式 | Softmax/RMSNorm 近似 |
| 把倍率留在数据流里 | SiLU LUT 的 ×1024 倍率合并进后续量化 scale，删掉恢复 task | MoE 中间阶段 |
| 代数重排合并 task | gamma 并进广播；`x - max` 改成 `BS(-1)` 加 EW Add | Add+RMSNorm、Softmax |
| 预生成模板 | 固定几何与完整寄存器配置 → 标记地址字段 → 重绑定缓冲 → 独立数值验证 | `rk_npu_add_rmsnorm_templates.inc` |
| 仿射 LUT 输出 | 用正偏置绕开零码、用公共增益留出表值余量，消费者去偏置后归一化 | stable Attention |
| FP32 承载定点激活结果 | 保留零码，收集时还原倍率并做最终 FP16 舍入 | 融合 GELU |
| 双 slot 隐藏 CPU 阶段 | 完整 KV 共享，scratch 独立，QK(next) 覆盖 centering(current) | Attention prefill |
| CPU 收尾融合 | 两份 partial 一次消费，直接生成 SwiGLU middle 并累计下游 scale | W4A8 FFN |
| 递推分块 | 用 Gram/state/output 矩阵乘摊薄逐 token 状态读写和提交 | GDN prefill |

---

## 6. 安全规则（板子是共享资源）

- 使用自己的板端 SSH/执行工具，并用 `timeout` 限时；多个实验共用设备时，应约定同一个锁来串行化 NPU 提交。
- probe 的 submit timeout 设为 1–2 s，缩短失败循环。
- 每次超时之后：先看 dmesg，再跑一个**完整初始化**的已知 canary，然后才继续。
  不要在同一个进程里 reset 后接着重试失败的配置。
- 已知会导致超时的配置要在代码里**在提交前拒绝**，不要留成可以随手触发的 CLI 选项。
- **禁止写 NPU runtime-PM control**（曾导致 kernel panic）。改频率时保存原 governor，在 finally 里恢复并读回确认。
- SSH 失联或系统异常时暂停测试，先恢复设备并采集故障信息，避免自动重试继续提交任务。
- 模型和大型测试数据放在独立目录，记录来源和校验值，避免混入源码仓库。

---

## 7. 调试清单

**submit 超时**
1. 看 dmesg 的 task counter，定位是第几个 task。
2. 检查 flags 是否匹配具体模板；Attention 的 LUT exp→PV 已有 PINGPONG 配置，部分历史模板仍需 PC-only。
   同时检查 core mask、`subcore_task` 槽位。
3. 检查链：每个 core 的最后一个 task 是否有结束尾；所有 body 是否在同一个 allocation。
4. flying EW：MRDMA 和 ERDMA 两个地址是否都改了；M=1 是否用了特殊配置。
5. 精度字段是否一致：CNA in/proc、CORE proc、DPU in/proc、RDMA in/proc。
6. 数值是否超出已验证范围：DPU 单 task 通道数、M tile bank 预算、notch 位宽。

**结果错但不超时**
1. 用 marker 数据找出错位规律：只有第一个 surface → `size_e`；块重叠 → `SURFACE_ADD`；
   一半通道 → atomic 不匹配；间隔读取 → EW cvt 位。
2. 换一个前驱 kernel 再跑，或者重启后先跑本 kernel：结果变化提示状态依赖；同时检查缓冲生命周期与同步，再补齐缺少的初始化。
3. 检查 relocation：是否有地址没改（仍指向旧 BO），是否把常数误当成地址改了。
4. 检查 native split 的 DMA 配置、`DATA_SIZE1` 高低位、feature_grains。
5. 检查 cache sync 方向和范围；检查 CPU 是否读了 NPU 还没写完的缓冲。

**性能不如预期**
1. 先比较 driver 的 `hw_elapse_time` 和 wall time，再用阶段 trace 区分 CPU、sync、submit 和调度等待。
   流水中的两者不是可以直接相减的独占阶段。
2. 只有每个测试前缀都完整初始化且输出可验证时，才用前缀差分估算增量成本；
   重排 task、删除初始化或改变频率时，这种差值不能直接解释为单个 task 的成本。
3. 用 B 流量模型检查 GEMM：task 数 × B tile 字节 / 带宽。
4. 测量时固定频率，以设备支持的 OPP 和实测读回值为准。同一进程内交替运行各方案，
   奇偶轮反转顺序。重叠的阶段时间不能相加。

---

## 8. 案例：几个 kernel 是怎么做出来的

### 8.1 INT8 GEMM

INT8 GEMM 路径的来源署名保留在
[源码](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/src/rk_npu_matmul_i8.cpp)
和 [第三方说明](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/THIRD_PARTY_NOTICES.md) 中。
配置检查包括 INT8→INT32/FP32 输出精度、大 K 的 M tile/grains、native A split-M，
以及 native C 输出、split-K 和三核 N 分片。prepare 阶段生成 regcmd，运行时重绑定地址；
收益需要用包含 CPU 打包、同步和归约的完整调用时间确认。

### 8.2 融合 GEMM + 张量加法

在已测试的 flying GEMM 几何中，仅启用 ERDMA 会超时，双 surface 的 MRDMA/ERDMA 配置通过。
恒等操作数不能证明 FP16 加法真正生效：CVT 设置错误时，第二操作数可能被忽略。
INT8 flying 加法的另一个问题是仅覆盖部分通道，该路径改为 offline INT32 DPU 加法。
应区分超时、操作数未生效和通道覆盖不完整，并用非恒等输入分别定位。

### 8.3 NPU dequant（两轴 scale）
逐行 task 最初超时。拆成单变量对照后（FP32 copy → BS 标量乘 → BS+EW → native copy → notch 抽行），
发现是 EW 乘数的粒度问题：32 bit 槽的低 16 位要放 FP16。修正后每行一个 task，BS 乘行尺度、
EW 乘通道尺度，直接输出紧凑 FP32。

### 8.4 固定形状 Add+RMSNorm

本库的固定形状实现使用预先准备的寄存器模板。运行时生成 gamma、归约权重和广播权重，
按已知地址字段重绑定缓冲。算子顺序、近似 rsqrt 的迭代次数以及 gamma 的融合位置
都要按各自输入范围做数值验证，不能把一个形状下等价的变换推广到任意形状。

实现见 [Add+RMSNorm](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/src/rk_npu_add_rmsnorm_f16.cpp)
及 [固定形状模板](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/src/rk_npu_add_rmsnorm_templates.inc)。
当前公开接口仅覆盖 `M=1/128, D=4096, eps=1e-5`，不是通用 RMSNorm 内核。

### 8.5 KDA decode：已记录的性能限制
复用 TF32 GEMM 生成器和 FP32 DPU 配置，状态常驻 NPU，布局 `[Dk/4,Dv,4]`
同时满足 DPU 输出和下一次 GEMM 的 native A。通过块对角合并 head，任务数从 64 降到 21，
但在该历史样本中仍比四核 NEON 慢 2.9 倍：瓶颈是状态的反复读写，而不是任务数。
`Dk=Dv=128`、FP32 状态时，每个 head 占 64 KiB，16 个 head 合计 1 MiB；
每 token 的总流量还取决于这些状态在多少个 task 中被重复读取。
尝试 flying 融合超时后，记录未通过的配置，未继续扩大测试范围。
**教训：先用流量模型判断上限；任务数减少不等于更快。**

### 8.6 Stable Attention：LUT、消费者修正与双 slot 流水

Attention 同时包含适合 NPU 的 QK/PV 矩阵乘，以及行最大值、mask、指数和归一化。
当前 stable 路径的计算分工如下：

| 阶段 | 执行位置 | 输出和用途 |
|---|---|---|
| QK | NPU GEMM | native FP16 scores |
| 有效 key 筛选、行最大值和 centering | CPU | 减最大值并夹取后的 FP16 scores；完整八元素组可用 FP16 SIMD |
| 指数近似 | NPU LUT | 带正偏置和公共增益的 FP16 指数表示 |
| PV 和分母列 | NPU GEMM | 分子与分母 partial |
| 去偏置、归一化、输出收集 | CPU FP32 | 最终 Attention 输出 |

指数表示使用 `P'≈g*exp(x)+delta`，`g=255/256`、`delta=16/32768`。
在延后归一化路径中，分子减去 `delta*sum(V_visible)`，分母减去 `delta*visible_key_count`，
相除时 g 抵消。这样可避免 LUT 零码异常，并在 INT16 表值顶部为偏置留余量。
寄存器与范围见第一部分 §7.5.1。

**修正公式要求有效集合完全一致。** 预加载未来 KV 不应改变 causal 行的可见前缀、分派模式或输出。
Boolean mask、全遮罩行、KV 尾块、truncate 后的替换后缀都要单独验证。
当前实现按 V 幅值与任务预算决定是否启用 LUT，不适合的行段保留 CPU softmax。

数值配方的模型检查必须独立于性能检查。2026-10-16 的 Qwen3-4B-W4A8-Mixed-LMHeadW8 对照，
使用 WikiText-2 raw test 前 32 个独立 512-token 窗口，每窗重置 KV，评分 target257–511，
共 8160 个目标；同一个可执行文件用 FP64 累加 NLL：

| 指数路径 | PPL | 相对 CPU FP32 |
|---|---:|---:|
| CPU FP32 | 14.885053 | — |
| LUT，g=1 | 14.972756 | +0.5892% |
| LUT，g=255/256 | 14.953214 | +0.4579% |

该配方在已测语料前缀上仍有数值代价；两种 LUT 的配对窗口区间包含零，不能由点估计确认整体质量改善。
固定 36 层真实 QKV 回放中，g=255/256 相对 CPU softmax 的输出 relative L2 均值/最大值为
0.04449%/0.05114%。局部误差小也不能替代模型检查。
详见 [数值记录](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/ATTENTION.md#stable-lut-exponential)。

多 query 块还可以保持相同的 LUT 配方，改变调度。两个 scratch slot 分别保存各块的中间结果：
完成 QK(0) 并 invalidate scores 后，提交 QK(1)，同时由 CPU 处理 scores(0)。
随后 LUT/PV、输出 finish 和下一块 packing 继续交错。完整 K/V 共享，slot 的 scores、partial、
描述符及 partial KV tail 独立。任意时刻最多一个阻塞 submit 在途，不需要异步 fence ABI。

验证流水时至少比较公共串行路径、相同 slot/分块布局的串行控制和流水路径。
前者给出实际收益，后两者用于分离布局代价与重叠收益。trace 应显示真实 QK/PV 提交覆盖 CPU
计算区间，而不只是覆盖 invalidate；输出还要经过多块、奇数尾块、跨层 cache 重绑定和错误收尾检查。

2026-10-25 的 `8774a0c` 板测中，固定 Qwen3 与 C7 序列的全部受测 logits，以及三组 ViT features，
在串行与流水之间逐字节一致。该轮未重跑 PPL。Qwen3-4B 的模型 prefill 配对结果如下：

| Prompt tokens | 串行 / s | 流水 / s | 耗时下降 |
|---|---:|---:|---:|
| 512 | 2.758 | 2.660 | 3.56% |
| 4096 | 32.849 | 29.668 | 9.68% |

模型为 Qwen3-4B-W4A8-Mixed-LMHeadW8，ubatch128；RK3588、kernel 6.1.118-zt、rknpu 0.9.8，
四个 A76 2.4 GHz、NPU 三核 1 GHz、DDR 2.112 GHz。两组使用同一程序，仅切换流水配置，
512/4096 分别为每方法 6/4 个正式样本的耗时均值，采用独立进程 ABBA。
计时包含所有 decoder 层与 KV append，不含模型加载、准备、tokenization、最终 norm/LM head。
各进程策略缓存均为 92 hits、0 miss、0 在线调优、0 策略 fallback。

4K prefill 总差值中约 82% 来自 Attention 阶段，其他阶段也有波动；表中是模型路径的观测结果，
不能把全部差值归因于 Attention。单块 decode/verify 不获得跨块重叠，未观察到一致的 decode 收益。
主要代价是增加一份 slot scratch，完整 KV 不翻倍。

实现与接口见 [pipeline](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/src/rk_npu_attention_f16_pipeline.cpp)
和 [Attention 工作区](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/ATTENTION.md#query-block-pipeline)。
上述性能验证对应 `8774a0c`；后续只准备首轮 graph 的 `59aa26a` 调整不包含在这组测量中。

### 8.7 GEMM+bias+GELU：把非线性放到正确的舍入位置

对于 `Y=GELU(A@B+bias)`，融合路径将 GEMM 的 CORE 输出直接送入 DPU：BS 沿 M 广播
FP32 bias，BN 缩放索引，LUT 计算 tanh GELU 近似，FP32 writer 写回 Q13 表值。
收集时乘以 `1/8192`，再按调用者需要写 FP32 或舍入到 FP16。

| 设计决定 | 原因 |
|---|---|
| bias 仅存 N 个 FP32 值，由 BRDMA 读取 | 不需要构造 M×N 的 bias 张量 |
| DPU 处理与写回使用 FP32 | 避免 BN 索引缩放饱和，以及直接 FP16 LUT 输出的零码异常 |
| 各核首 task 带装表前缀 | 不依赖前一算子或 runtime PM 之后的 LUT 内容 |
| 复用常驻完整 K 权重，运行时 patch B 地址 | 多层相同形状可复用 workspace，无需重传完整权重 |
| 固定完整 K | `GELU(sum(partials)+bias)` 不能拆成各 partial 的 GELU 后求和 |

当前几何采用 panel8，M 补齐到 8，N/K 补齐到 32；N tile 按允许的 core 分配。
FP32 输出的 surface 数、地址步长和 notch 按新布局配置；公共返回值是借用的映射视图，
要在下一次 run 或 workspace 销毁前完成收集。

这条路径改变了舍入位置：普通 FP16 GEMM 先舍入输出，再由 CPU 加 bias/激活；融合路径
在最终 FP16 舍入之前做 bias 和 LUT。因此验证要同时覆盖独立高精度 GELU 参考、原有实现、
真实模型最终特征，不能仅要求与原实现逐位一致，也不能用局部 LUT 误差替代任务准确率。
LUT 覆盖 `[-4,4]`，正尾近似恒等、负尾近似零；它不是完整 FP32 GELU。

这是在已有 GEMM 写回前增加后处理的案例。若 K 很大必须分块，应先构造并验证归约后的
激活阶段；不能直接把这个 full-K 模板附到每个 partial 上。普通 GEMM 的调优成绩也不能代表融合路径。

接口见 [MM GELU](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/MM.md#fused-fp16-gemm--bias--gelu)，
源码见 [rk_npu_mm_gelu.cpp](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/src/rk_npu_mm_gelu.cpp)，
板端入口为 [test_mm_gelu --board](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/tests/test_mm_gelu.cpp)。

### 8.8 GDN prefill：把逐 token 递推改写成分块矩阵乘

标量门控 DeltaNet 每个 token/head 的数学边界为：

```text
a = exp(g)
S = a * S
delta = beta * (v - S @ k)
S = S + delta @ transpose(k)
o = S @ q / sqrt(128)
```

逐 token 的状态读写与小任务提交可能盖过计算收益，见 §8.5 的 KDA decode 样本。
GDN prefill 使用分块形式，将多个 token 的关联写成 Gram、state prediction、state increment
和 block output 矩阵乘。CPU 构造衰减系数、求解三角系统并收集输出；NPU 执行 TF32 矩阵乘，
FP32 DPU 合并状态。

状态的逻辑定义保持 FP32。设备内使用 `[head,key_channel/4,value_channel,4]` 布局，
以每 head 一个标量表示 `S=scale*Z`，减少反复对整张状态做衰减；状态在块之间驻留于设备，
导出时恢复标量与公共布局。带 Conv 的入口将 CPU Conv/SiLU/QK normalization 的结果直接写成
NPU native 输入，减少一轮中间张量的读写。

TF32 的 FP32 容器不会保留乘法器输入的全部有效位。普通模式使用一次 TF32 乘积，硬件截去
13 个低位；补偿模式将操作数拆成 high 与 residual，执行 high/high、high/residual、residual/high
三组乘积后用 FP32 合并。补偿提高精度，同时增加 task 和执行成本，不能视为免费的 FP32 运算。

当前接口覆盖一条序列、Hk16/Hv32、Dk=Dv128，窗口 1..2048、chunk16/32/64/128，
并显式区分 grouped/interleaved head 映射。补偿模式的小 chunk 还受 4095-task 预算限制。
单 token decode 仍是独立 CPU 路径。

2026-10-18 的库接入验证包含独立 FP64 递推、两种 head 映射和精度、2048-token 窗口、
原地 state/history，以及 142 项相对前一 kernel 的逐位对照。
这些组件检查用于确认接口与实现边界；模型质量仍需由集成方验证，组件时间不能当作模型吞吐。
本例可复用的方法是：保持规范状态定义，按块增加 GEMM 的有效工作量，明确每个 CPU/NPU 阶段的职责。

见 [GDN 算法、布局与验证](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/GDN.md)
及 [公共接口](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/include/rk_npu_gdn.h)。

### 8.9 W4A8 FFN：在 CPU 收尾时省掉中间张量

W4A8 的 NPU partial 仍需整数重建、correction、scale 和激活。Gate/up 权重按 `[gate(F),up(F)]`
排列时，可以在最终收尾中直接生成紧凑 FP32 `[M,F]` SwiGLU middle，无需先写完整 `[M,2F]`
浮点 gate/up 张量。恰好两个 K wave 的路径进一步同时读取两份 native partial，省掉中间
INT32 accumulator 的写回与重读。

SwiGLU 输出就绪时累计每行 maxabs，完整行的 scale 可交给 Down，避免 Down 再扫描一次 middle。
这里的依赖是“完整行已完成”，不能用某个 tile 的局部最大值替代全行 scale。
Gate/up 与 Down workspace 在整个 FFN 调用期间保持占用，middle 仍由调用者持有。

当前 AArch64 SwiGLU 使用 FP32 指数多项式与 `[-80,80]` 饱和；它与标量 `expf` 或早期 FP64
实现的舍入、极小负尾及非有限语义不同。评价调度/访存融合时，应固定相同的数学实现做对照，
再单独报告改变近似算法的效果。

这个案例说明 CPU 后处理本身也可以融合。收益来自减少中间写回、重复扫描和 packing，
最终要测完整 gate/up→SwiGLU→Down；仅比较 CPU kernel 的用时，可能把原本已被 NPU 隐藏的工作算成收益。
见 [FFN 接口与数值说明](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/include/rk_npu_w4a8.h)
和 [实现概览](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/README.md#w4a8-fused-swiglu)。

---

## 9. 文档与证据的写法

新 kernel 的记录应包括：

- 环境（驱动、runtime、编译器版本，频率）和形状范围；
- 最终寄存器配置中与已知配置**不同的字段**，以及每个字段为什么这样设；
- 失败的配置和现象（超时、错误值、半覆盖），并注明"未证明硬件不支持"；
- 正确性：参考计算定义、逐位还是容差、边界用例、重复次数；
- 性能：完整调用与硬件时间分开报告，写明对照基线（NEON 还是 NumPy）；
- 结果目录里保存源码 hash、原始样本和 governor 恢复记录。

可以按“环境与范围 → 数据布局 → 寄存器差异 → 正确性 → 完整调用性能 → 未解决问题”组织报告。
公开时使用可访问的源码或数据链接；历史样本、主机参考、板端结果和推断分别标明，
不能用源码实现代替板端验证，也不能用设备计时代替端到端延迟。

---

[首页](https://github.com/happyme531/rknpu2_matmul_open/wiki) · [硬件行为](https://github.com/happyme531/rknpu2_matmul_open/wiki/Hardware-Behavior) · [编程方法](https://github.com/happyme531/rknpu2_matmul_open/wiki/Programming-Guide)
