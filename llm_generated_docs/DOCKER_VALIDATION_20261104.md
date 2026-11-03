# v0.11.0 ARM64 容器验证

验证日期：2026-11-04。镜像：`ghcr.io/happyme531/open_rknpu2_lm_infer:0.11.0`。
镜像使用 v0.11.0 的正式 ARM64 Release 附件，未压缩约 22.4 MB；最终运行层为 scratch，
包含程序、动态库和许可证，不包含 shell 或包管理器。

## 发布与拉取

- GitHub Actions 发布任务：[41681193410](https://github.com/happyme531/open_rknpu2_lm_infer/actions/runs/41681193410)。
- 版本标签 `0.11.0` 与 `latest` 指向同一个 manifest：
  `sha256:f0dcf6b4485716cae4182b08bf8db696d8a6e12a997c74fd371c26c8bf286376`。
- 已使用空 Docker 客户端配置匿名拉取，启动输出 `open_rknpu2_lm_infer 0.11.0`。
- 正常 Release 流程在上传附件后发布 GHCR；已有 Release 可通过 CI 的 `container_version` 手动补发镜像。

## 实板配置

- 板：`zt-firefly`，RK3588，RKNPU 驱动 0.9.8，Docker Engine 20.10.5。
- 模型：`Qwen3-4B-W4A8-Mixed-LMHeadW8`，复用板端已有的本地只读 checkpoint。
- context=512、ubatch=128、threads=4、CPU mask=0xf0、NPU mask=7。
- 固定 CPU 2.4 GHz、NPU 1 GHz、DDR 2.112 GHz；外层 `scripts/run_board.py` 获取共享板锁并恢复时钟。
- 仅映射 `/dev/dri/renderD129:/dev/dri/renderD129`，未提供 `card1`，未启用 privileged。
- 调优缓存使用独立 Docker volume，模型只读挂载。

## v0.11.0 发布二进制与本地镜像

| 检查 | 结果 |
|---|---|
| 模型加载和 NPU 设备发现 | 通过 |
| prefill + 4 步 decode 完整 logits | `[5,151936]`，759680 个 FP32 值；原生与容器逐字节相同，最大绝对差 0 |
| greedy CLI 文本生成 | 原生与容器逐字节相同 |
| API 普通响应、SSE | 输出相同；`1+1` 返回 `2` |
| 时钟恢复 | `exit_code=0`、`restore_errors=[]` |

输入：`请用一句中文解释什么是人工智能。`

输出：

> 人工智能是指让机器具备人类智能的能力，如学习、推理、感知和决策等。

## GHCR 预构建镜像实板结果

从 GHCR 匿名拉取的镜像单独导入板端，并重新执行 NPU 测试：

- 759680 个 FP32 logits 与原生 v0.11.0 参考逐字节相同。
- API 普通响应与 SSE 相同，均返回 `2`。
- 仅暴露 renderD129，设备发现、模型加载和硬件执行均正常。
- `ghcr-results/clocks.json` 中 `exit_code=0`、`restore_errors=[]`、`timed_out=false`。

## 范围与运行环境

验证覆盖这份 checkpoint 的文本推理；不代表其他模型、视觉和专家流式存储路径均已验证。
板端 Docker daemon 的既有代理连接失败，因此 GHCR 镜像由主机匿名拉取后导入板端；未改动板端 Docker 服务配置。
在线拉取需要宿主 Docker daemon 具备可用的网络/代理配置，离线部署可使用配套 `image.tar`。

原始日志、完整 logits、时钟报告和部署包位于
`llm_generated_docs/evidence/docker_scratch_rk3588_20261104/`；`results/` 对应本地构建的发布包镜像，
`ghcr-results/` 对应从 GHCR 拉取的预构建镜像。
