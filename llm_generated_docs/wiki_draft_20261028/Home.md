# RKNPU2 kernel 编程手册

面向在 RK3588 NPU 上开发新 kernel 的开发者。这里的 kernel 指直接生成寄存器命令流（regcmd），
通过 rknpu DRM ioctl 提交，不依赖 librknnrt 或 RKNN 编译器。

RKNPU2 提供卷积、矩阵乘和固定后处理流水线。开发 kernel 的关键是确定每个阶段的
数据布局、数值表示和状态依赖，再决定哪些工作放到 NPU、哪些由 CPU 配合完成。
本手册覆盖基础 GEMM，也介绍 stable Attention、融合 GELU 和分块 GDN prefill。

| 文档 | 内容 |
|---|---|
| [第一部分：硬件实际行为](https://github.com/happyme531/rknpu2_matmul_open/wiki/Hardware-Behavior) | 驱动怎样消费 task、PC 链编码、内存与同步、寄存器状态残留、数据布局、CNA/CORE/DPU/RDMA 各寄存器的实测语义、数值行为、性能数字、问题清单、未解决问题 |
| [第二部分：编程方法](https://github.com/happyme531/rknpu2_matmul_open/wiki/Programming-Guide) | 是否值得上 NPU 的估算、从算法到 task 链的设计流程、公开实现参考、验证方法、生产化、代码骨架、常用技巧、安全规则、调试清单、案例 |

## 阅读顺序

1. 第一部分 §1–§5：建立整体模型，理解 task、PC 链、状态残留和布局。
2. 第二部分 §1–§3：设计流程。
3. 动手时查第一部分 §6–§8 的寄存器细节和 §12 的问题清单。
4. 第二部分 §8.6–§8.8：查看 Attention、GELU、GDN 的完整设计案例；§8.9 介绍 CPU 收尾融合。

## 常见开发问题

| 问题 | 阅读位置 |
|---|---|
| 为什么单算子通过，接着运行另一个算子却失败？ | 第一部分 §4，尤其是 INT4→INT8 的状态验证 |
| LUT 能否与 GEMM 串成 PINGPONG 链？ | 第一部分 §2.5、§4.3、§7.5；第二部分 §8.6 |
| 如何用 LUT 表达指数函数或 GELU，同时处理零值和范围？ | 第一部分 §7.5；第二部分 §8.6、§8.7 |
| K 分块为什么影响浮点结果？ | 第一部分 §10、§11.4 |
| 如何重叠 CPU 后处理与 NPU 计算？ | 第一部分 §11.5；第二部分 §8.6、§8.9 |
| 怎样把递推计算改写成较大的矩阵乘？ | 第二部分 §8.8 |

## 库接口

直接使用算子时，优先阅读仓库内的接口说明：
[MM/BMM 与浮点调优](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/MM.md)、
[量化 GEMM](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/API.md)、
[Attention](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/ATTENTION.md)、
[MLA](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/MLA.md)、
[GDN](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/GDN.md)、
[MoE](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/MOE.md)。
Wiki 解释硬件行为与实现方法；接口生命周期、支持形状及运行配置以对应版本的头文件和接口文档为准。

## 适用范围

内容基线：2026-10-28，公开库提交
[`3396f77`](https://github.com/happyme531/rknpu2_matmul_open/commit/3396f77b810a3888684f8648703c4592ea8b3329)。
手册整理既有代码和历史板端记录；各实验的验证版本与环境分别注明。
寄存器参数、性能数字和失败现象必须结合对应的驱动、形状、数据布局及频率理解。
公开源码以 [rknpu2_matmul_open](https://github.com/happyme531/rknpu2_matmul_open) 为入口；
文内源码链接用于检查实现，不作为全部历史数值测量的原始记录。

本手册原创内容适用项目的 [AGPL-3.0-only](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/LICENSE)。
第三方署名和权利说明见 [THIRD_PARTY_NOTICES.md](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/THIRD_PARTY_NOTICES.md)。
