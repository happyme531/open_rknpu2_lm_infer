# v0.12.0

- 跨请求复用预填充整块工作区，减少重复申请 NPU 内存。分配遇到内存压力时可回收当前 domain 的空闲工作区。可用 `RK_NPU_PREFILL_CACHE_DOMAIN_MIB` 调整保留预算，设为 `0` 可关闭跨请求保留。
- 优化因果 GQA 预填充、Qwen3.5 的注意力输出门控和 GDN 预填充。Qwen3.5 MoE 中满足布局条件的专家使用 full-K，旧专家缓存会在首次使用时重建。
- 优化自回归 logits 的采样路径。HTTP API 读取模型的 `generation_config.json` 作为采样默认值，请求参数可逐字段覆盖。
- 修复 Rust 构建与原生 CMake 所选 OpenMP 运行库的链接不一致。
- 发布 ARM64 Docker 镜像至 GHCR，提供 Compose 模板和 NPU render node 映射。镜像标签为 `0.12.0` 和 `latest`。
