# H88K GM7 SSD：连接清洁前后与 PCIe 代际对照

2026-10-25，`<BOARD_SSH>`，HINLINK H88K 8GB，Predator GM7 512GB，经转接板或延长线连接。

## 清洁并重启后的结果

| 读取模式 | 清洁前 GB/s | 清洁后 GB/s |
|---|---:|---:|
| 裸盘顺序 1MiB，QD1，单任务 | 0.878 | 2.788 |
| 裸盘顺序 1MiB，QD8，单任务 | 1.076 | 3.272 |
| 裸盘顺序 1MiB，QD32，单任务 | 1.034 | 3.285 |
| 真实 decode 专家序列，单 worker | 约 1.06 | 2.019 |

裸盘为 `/dev/nvme0n1`，所有 fio 测试使用 `--readonly --direct=1`，libaio，单任务，大核 CPU4–7，1 秒预热、5 秒计时。清洁前每个 fio 配置一次，清洁后按 QD1/8/32/32/8/1 各两次取均值。数据为十进制 GB/s。

专家回放读取同一段历史 trace 的前 4096 条记录，每条 1,597,440 B，共 6,543,114,240 B；O_DIRECT，单线程，复用对齐缓冲区，不包含 NPU 导入或模型计算。清洁后的读取耗时 3.240199 秒。

两轮均为内核 `6.1.115-vendor-rk35xx`、NPU 驱动 0.9.8，测试期间 CPU 大核 2.4GHz、DDR2112MHz，结束后频率策略恢复成功。SSD 链路为 8.0GT/s ×4，MPS128。清洁后没有调整 PCIe、APST 或 SSD 参数。

## 链路稳定性

| 条件 | 裸盘 1MiB QD8 GB/s | Endpoint 配置读取返回全 ff |
|---|---:|---:|
| 清洁前 Gen3 ×4 | 1.022 | 37/100 |
| 清洁前临时 Gen2 ×4 | 1.664 | 0/100 |
| 清洁前恢复 Gen3 ×4 | 1.342 | 57/100 |
| 清洁后 Gen3 ×4，吞吐测试前采样 | — | 30/1000 |
| 清洁后 Gen3 ×4，吞吐测试后采样 | — | 18/1000 |

配置采样读取 endpoint `0000:01:00.0` 的 Device Control/Status DWORD，返回全 ff 不代表相同比例的文件读失败。Root Port `0000:00:00.0` 的前后两批各 1000 次配置读取均无全 ff。

清洁后的 endpoint AER Correctable Status 从 `0x00000081` 变为 `0x000010c1`，新增 Bad TLP 与 Replay Timer Timeout 状态；Uncorrectable Status 始终为 0。测试后 SMART 为 44°C，介质错误和错误日志计数均为 0，未记录温度节流。

吞吐已恢复至约 3.3GB/s，但 Gen3 链路仍有残留异常。Gen2 更稳定，以及清洁/重插并重启后吞吐提高、配置读取异常减少，支持优先排查转接、延长连接及其 Gen3 信号质量；尚未将具体故障定位到某个连接器、线缆、板端 PHY 或 SSD 控制器。

## 其他控制与当前状态

- 清洁前 ext4 文件与裸盘吞吐相近，QD32、io_uring、四任务均没有消除当时约 1GB/s 的限制；CPU 使用率较低。
- 关闭 APST 的只读对照没有改善吞吐。MPS 对照未获得有效速度样本，不据此归因；PCIe Device Control 与 APST 已恢复原值，之后板子重启。
- 临时 Gen2 对照期间先卸载 SSD 文件系统，测试后重新挂载并恢复 Gen3；SSD 上的模型及专家文件保留。
- 清洁后的正式推理测试已完成：20/40/56 槽、预取开关各两次，共 12 组；另有两份完整 trace。结果见 [NVMe 正式测试报告](MINICPMV47_NVME_H88K_BENCHMARK.md)。清洁前单个样本仍保留作历史记录。

证据位于 [minicpmv47_nvme_h88k_20261025](evidence/minicpmv47_nvme_h88k_20261025/)：`storage_diag/`、`storage_tune/`、`link_diagnostic/` 与 `after_clean_1716/`。包含原始 fio JSON、配置采样、SMART、内核日志、时钟控制及恢复记录。清洁后结果以 `after_clean_1716/summary.json` 为准。
