# INT4 单 task N8128 上限

2026-10-21，按主机日期记录。用户指定采用 `N <= 8128` 的简单限制。

`I4_N_TILE_MAX` 设为 8128，INT4/W4A8 的 `n_tile` 仍要求 64 对齐。内存查询和寄存器生成均拒绝 `n_tile=8192`，在提交前返回参数错误。逻辑矩阵 N 不受此限制：例如 `N=8192,n_tile=8128` 使用 `8128+64` 两个 task。MoE 直接调用同一寄存器生成器，沿用同一上限。

## 排查结果

原实现仅按 DPU channel 字段的可编码范围放宽到 8192。主板已有失败记录；本次在备用1独立复现：

- 单 task `N=8128,M=2,K=64` 的 INT4 及随后 INT8 正常；`N=8192` 的 INT4 逐值正确，但随后 INT8 的 192 个输出均从 `-384` 变为 `-192`。逻辑 N8191 补齐至 8192 后也触发。
- `K=32/64/128/256` 均可触发，因此限制 K 不能解决本次问题。
- 基向量输入表明这不是简单的输出缩放：K 的每个 16 元素段切换时，正确输出在偶数/奇数 N 通道之间交替，其余通道为零。
- 补齐 INT8 遗漏寄存器、增加 CBUF/精度前序写入、清 ping-pong 指针、显式设置 kernel_group 均未解决。具体硬件内部机制仍未定因；本次仅排除已复现的 N8192 task。

原始诊断见 [boundary.log](evidence/i4_i8_transition_20261021/boundary.log)、[km.log](evidence/i4_i8_transition_20261021/km.log)、[basis.log](evidence/i4_i8_transition_20261021/basis.log)。修复前诊断源码归档在同一证据目录的 `probe_i4_i8_transition_before_limit.cpp`。

## 验证

备用1 `备用板 1`，Linux 5.10.198，RKNPU 0.9.6。共享板锁，CPU4–7 / 2256MHz、NPU 1GHz，DDR 未控制。本次不提供性能结论；主板 0.9.8 的修复后回归未重跑。

| 检查 | 结果 |
|---|---|
| `test_i4_cpu`：8128 寄存器字段、8192 拒绝、逻辑 N8192 两 task 查询 | PASS |
| `test_i4` 完整回归，新增 N8128 / N8192 分块后立即接 INT8；串行、流水 | PASS |
| `test_w4a8`，加载本次新库 | PASS |
| `test_moe_w8 --w4`，加载本次新库 | PASS |

没有通过 reset 或重试使新增边界用例通过。测试使用冻结的新共享库，SHA256 为 `8949924ca830d3f5ec357650f31b65331d9352cdec300dff0bd08ef7f4d47325`；W4A8/MoE 的既有测试可执行文件通过 `LD_LIBRARY_PATH` 加载该库。命令、可执行文件及库哈希见 [limited_results.json](evidence/i4_i8_transition_20261021/limited_results.json)。

[limited_i4.log](evidence/i4_i8_transition_20261021/limited_i4.log) 保留完整 INT4/INT8 输出；[limited.clocks.json](evidence/i4_i8_transition_20261021/limited.clocks.json) 记录退出码 0、`restore_errors=[]`，全部频率上下限恢复。
