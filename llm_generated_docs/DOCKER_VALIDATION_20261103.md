# RK3588 scratch 镜像验证

2026-11-03，在 `zt-firefly` RK3588 板上验证 v0.10.0 发布版 scratch 镜像。
宿主 RKNPU 驱动为 0.9.8，Docker Engine 为 20.10.5。镜像未压缩大小约 22.3 MB，
使用自带 glibc、libstdc++ 和 libgomp，不依赖宿主发行版或 `librknnrt.so`。

## 配置

- 模型：`Qwen3-4B-W4A8-Mixed-LMHeadW8`，从板端已有 checkpoint 复制到本地临时目录。
- context=512、ubatch=128、CPU mask=0xf0、threads=4、NPU mask=7。
- 外层 `scripts/run_board.py` 持有共享板锁，CPU 2.4 GHz、NPU 1 GHz、DDR 2.112 GHz。
- 模型只读挂载，调优缓存使用独立 Docker volume，不使用 `--privileged`。
- `/dev/dri/card1` 和 `/dev/dri/renderD129` 的 sysfs device 均为 `fdab0000.npu`。
- 默认使用宿主 `/dev/dri/renderD129`，映射到 v0.10.0 二进制打开的容器 `/dev/dri/card1`。

## 结果

| 检查 | 实测结果 |
|---|---|
| ARM64 镜像加载、程序启动 | 通过 |
| prefill + 4 步 decode 完整 logits | `[5,151936]`，759680 个 FP32 值；原生和容器逐字节相同，最大绝对差 0 |
| render 节点完整 logits | 宿主 renderD129 映射后，与相同原生参考逐字节一致 |
| greedy CLI 生成 | 原生与容器输出逐字节相同 |
| API 普通响应、SSE | 均通过，输出一致；`1+1` 返回 `2` |
| API 中文生成 | 与原生 CLI 一致，生成 20 tokens |
| 板端时钟恢复 | `restore_errors=[]` |

中文生成输入：`请用一句中文解释什么是人工智能。`

输出：

> 人工智能是指让机器具备人类智能的能力，如学习、推理、感知和决策等。

原始 evidence 在 `llm_generated_docs/evidence/docker_scratch_rk3588_20261103/`：
`results/` 保存原生/容器完整 logits 和 CLI 输出；`render-results/` 保存 render 节点 logits；
`release-api-results/` 保存 API JSON、SSE、服务日志及成功的时钟报告。

验证覆盖这份 checkpoint 的文本推理和服务路径。没有据此宣称其他模型家族、视觉或专家流式存储路径均已通过。

## render 节点自动发现版本

另外在板端从相同 Rust runtime 源码重新构建，接入当前算子库的设备发现改动；
验证镜像为 `open-rknpu2-lm-infer:device-discovery-20261103`，不作为已发布版本分发。
设备发现实现对应算子库提交 `c1f4bc5`，已核对 `rk_npu_core.cpp` 与构建快照完全一致。
容器只映射 `/dev/dri/renderD129:/dev/dri/renderD129`，未提供 `/dev/dri/card1`。

- `rk_npu_open(NULL)` 正常发现 NPU 并完成模型加载。
- 759680 个 FP32 logits 与相同原生参考逐字节一致。
- API 普通响应与 SSE 一致，均返回 `2`。
- `discovery-results/clocks.json` 的 `exit_code=0`、`restore_errors=[]`。

设备发现源码快照、补丁和结果在同一 evidence 目录的 `operator-source/`、
`device-discovery.patch`、`discovery-results/`。正式 Dockerfile 继续下载 Release；
包含新发现逻辑的 Release 发布后，再同步镜像版本和容器内 render 节点映射。
