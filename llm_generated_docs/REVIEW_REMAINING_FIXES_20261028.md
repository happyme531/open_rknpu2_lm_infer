# Slot 内存所有权、零 route 同步与调用脚本修复

## 改动

`native/rk_npu_matmul/src/rk_npu_moe_w8.cpp` 的 W4 bank 分配数组改为构造函数局部变量。构造完成后，仅各 slot 的 owner 保留设备 buffer；slot 交换同步转移 view、元数据和 owner，不再有原始 bank 数组保留无用分配。

`weights_device_bytes()` 按当前 owner 引用到的独立分配去重统计。部分 slot 交换时，一个 handle 可能保留两边的整块分配，因此统计会暂时增加；全部 slot 交换后回落。多个 handle 若仍共享同一分配，各自都会计入该完整分配，不能直接将这些数相加作为全局唯一分配量。

MoE 中间 buffer 的同步范围改为实际 assignments 数，包括共享 expert 的 assignments。无 shared expert 且所有 route 为零时，在完成输入校验后清零输出，直接返回，不构建或提交任务，也不调用设备同步。部分 route 被跳过时，gate、middle 和 down 的同步只覆盖当前实际写入区域，保留原有 wave stride。

`call_llm.py` 在读取提示词前处理 `--list-models`。显式 `--think` 和 `--no-think` 分别发送 `chat_template_kwargs.enable_thinking=true/false`；未指定时省略该字段，跟随服务默认值。普通和 SSE 请求共用这一行为。

## 验证范围

native 回归在 RK3588 主板 `zt-firefly` 执行，使用独立 release 构建。设备测试由 `scripts/run_board.py` 获取共享锁并固定频率，检查退出码和频率恢复。

- `test_moe_quant` 用链接包装记录实际设备分配、释放和同步。检查部分交换时的去重统计、完整交换后销毁 staging 的实际释放、保留 hot 的数值结果，以及并发导入/其他 slot 交换。
- 同一 workspace 连续执行全部、部分、全零、全部 route，比较独立 route 结果；检查部分 route 的同步字节下降，全零 route 无同步调用。
- `test_moe_w8` 覆盖 W8、NPU LUT middle、W4 和 split-K mixed 路径，检查独立数值 oracle，以及全零 route 清除旧输出并继续拒绝非法 scale。
- `test_moe_cpu` 检查共享 CPU middle 内核。
- 调用脚本使用本地 HTTP/SSE fixture，执行真实 CLI：无提示词时列模型，以及普通/流式请求的默认、开启、关闭思考，共7项。

以上检查全部通过；6项 native 测试退出码均为0，频率恢复无错误。量化 MoE 用例的同步字节数为：全部 route 261,120、部分 route 135,680、全零 route 0；保留 route 的输出与独立执行结果一致。staging 销毁的实际释放量与该 bank 的 payload 一致，hot 继续运行的输出保持一致。

未执行完整模型性能或 PPL 测试；未改动 MTP 调优逻辑。

结果与复现入口：

- [native 构建和回归](evidence/review_remaining_20261028/native.json)
- [设备测试脚本](evidence/review_remaining_20261028/check_native.py)
- [脚本 HTTP/SSE 检查结果](evidence/review_remaining_20261028/client.json)
- [脚本 HTTP/SSE 检查入口](evidence/review_remaining_20261028/check_client.py)
