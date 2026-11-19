# RKNPU2 Wiki 修订稿审阅入口

本目录包含完整的 Wiki 候选页面，保留原有“硬件行为＋编程方法”两篇结构。
当前仅供本地审阅，尚未提交或发布。

| 稿件 | 建议阅读位置 |
|---|---|
| [首页](Home.md) | 阅读顺序、常见开发问题和库接口入口 |
| [硬件行为](Hardware-Behavior.md) | §2.5 提交 flags；§4.3 LUT 生命周期；§4.4 跨精度状态；§7.5 LUT 数值表示；§11.4–§11.5 调优与流水 |
| [编程方法](Programming-Guide.md) | §3 验证与生产化；§8.6 Attention；§8.7 GELU；§8.8 GDN；§8.9 W4A8 FFN |
| [侧栏](_Sidebar.md) | 补充各算子的 API 文档导航 |

以上五个 Wiki 文件（另含 `_Footer.md`）保留线上导航格式；正文中的 Wiki 导航链接会打开现有线上页面。
审阅本地版本请使用本页的文件链接。

## 修订重点

1. 将 LUT flags 从“混合链用 PC-only”改为具体模板契约，说明 stable Attention 的 PINGPONG 实现。
2. 区分独立装表、链内装表和 CPU 构图成本；保留请求级 LUT 驻留复用未通过模型数值检查的边界。
3. 补入 INT4 N8192→INT8、FP16 QK N8192→INT8 的状态问题与现有安全分块，不推断共同根因。
4. 说明 Attention 正偏置/增益修正与 GELU FP32/Q13 输出，避免把 FP32 容器误写成 FP32 激活精度。
5. 更新浮点 K 分块、独立 NPU 归约树、完整调用调优及 load-only 策略；明确分块舍入。
6. 加入 Attention、GELU、GDN 三个完整案例，以及 W4A8 CPU 收尾融合案例。
7. 用完整调用与模型 prefill 的不同计时口径解释流水收益，保留频率、样本数和验证版本。
8. 修正非零 IOMMU domain 的软件版本门槛，并将 DPU 几何限制按具体模板描述。

## 版本与证据

- 对照的线上 Wiki 提交：`fa0448d7c08af2f71af3b10d925c86490ef48dc2`，2026-10-14。
- 正文源码基线：公开主线 `3396f77b810a3888684f8648703c4592ea8b3329`，2026-10-28。
- Wiki 正文指向本库的文件链接已固定到该公开提交，避免后续主线变化使源码与描述错位。
- 本地 HEAD 比上述公开主线多出的两项提交未作为本次新增结论的依据。
- 本轮仅编辑、核对文档，没有重新执行板端实验。

下表用于本地审阅，不属于发布到 Wiki 的页面。

| 正文内容 | 证据入口 | 使用边界 |
|---|---|---|
| LUT 装表成本、休眠失效和驻留复用 | [LUT_RESIDENT.md](<HOST_HOME>/rk3588-nn/expr/lowlevel/tools/npu_cpu_offload_probe/LUT_RESIDENT.md) | 三核热态 Attention 配对；构图单列；模型驻留复用未通过 |
| LUT 正偏置、gain 和 PPL | [公开 Attention 数值记录](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/ATTENTION.md#stable-lut-exponential)；[本地 gain 记录](<HOST_HOME>/rk3588-nn/expr/lowlevel/tools/npu_cpu_offload_probe/LUT_GAIN.md) | 8160 个评分目标；局部回放与 PPL 分开；两种 LUT 的配对区间包含零 |
| Attention 双 slot 机制与收益 | [正式集成报告](../ATTENTION_PIPELINE_INTEGRATION_20261024.md)；[原始统计](../evidence/attention_pipeline_integration_20261024/summary.json) | 数值和性能对应 `8774a0c`，不把后续 `59aa26a` prepare 调整写成已测 |
| 双 slot 同布局串行控制 | [原型报告](../ATTENTION_TWO_SLOT_PIPELINE_20261024.md) | 用于解释对照设计；正文性能表采用正式集成报告的数据 |
| GELU 寄存器与 FP32 carrier | [公开 MM 文档](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/MM.md#fused-fp16-gemm--bias--gelu)；[本地实验](../NPU_FFN_GELU_RESEARCH.md) | 区分独立 INT16 writer 实验与正式融合 FP32 writer；正文未采用跨批次 ViT 加速比 |
| 浮点 K 分块与舍入 | [公开 MM 文档](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/MM.md)；[舍入诊断](../FLOAT_MM_ROUNDING_DIAGNOSIS.md) | 相对原实现一致性与相对高精度参考误差分开 |
| INT4 单 task N 上限 | [修复提交](https://github.com/happyme531/rknpu2_matmul_open/commit/f7ba205bced540fc48fdea50228479913644251e) | Ntile≤8128，逻辑 N 可更大，不能解释为寄存器编码上限 |
| FP16 QK N 上限 | [公开 Attention 文档](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/ATTENTION.md) | 私有 QK 安全 tile4096，独立于 PV kv_tile，不改变公开 GEMM 默认策略 |
| GDN 分块、补偿 TF32 与接入验证 | [公开 GDN 文档](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/docs/GDN.md) | 组件验证，未作为模型吞吐或全模型质量结论 |
| W4A8 收尾融合 | [公开接口](https://github.com/happyme531/rknpu2_matmul_open/blob/3396f77b810a3888684f8648703c4592ea8b3329/include/rk_npu_w4a8.h)；[两波实验](../QWEN3_W4A8_CPU_TWO_WAVE_FUSION_20261022.md) | 区分数学近似与访存/调度变化；正文说明现有实现，不外推局部百分比 |

## 文档检查

五个 Wiki 页面已通过 Markdown 表格、代码围栏、导航目标和公开文件路径检查。
正文引用的 33 个本库文件均存在于 GitHub 基线提交；公开页面中未发现本机路径、私有地址或工具说明。
保留的第三方链接沿用原 Wiki，本轮未重新测试所有外部站点。

未修改原 Wiki checkout、`lowlevel/docs/kernel-writing/` 或算子源码；当前交付是本目录中的审阅稿。
