# ViT Attention 临时工作区

2026-10-21，RK3588 主板，Qwen3.5 C7。

ViT 的 `FullAttention` 按当前图像的实际 patch 数分配临时 K/V 和查询工作区。24 个 Transformer blocks 覆写并复用同一组 buffer，最后一个 block 完成后释放；图像特征仍由 Encoder 的 LRU 缓存复用。视觉路径不保留自回归 KV 状态，也不使用文本 `ctx-size` 限制 patch 数。

K/V 物理行数按 native 布局对齐到32；分配边界来自实际输入。移除了视觉后端固定1024行的配置，以及 native Attention 对 `max_capacity > 32768` 的固定拒绝条件。调用方的分配边界、任务数量和实际算子形状校验继续生效。

## 验证

- 700×527测试图片预处理为704×512，grid `[1,32,44]`，1408个patch、352个视觉token。
- 修改前版本 `0b5ffcd` 复现 `invalid full FP16 attention shape`；修改后 FP16、FP32 完整ViT及CLI问答均通过，两条路径各输出901120个有限特征值，重复输入命中特征缓存。
- 修改前后的固定token文本logits、蓝色小图features逐位一致。
- 图像Attention板测覆盖1408、3072、132行；132行仅分配160行物理buffer，没有保留前一张大图的容量。随机输入CPU oracle也通过。
- 65536行K/V、1行query的板端计算通过；该结果验证长K/V维度，不代表完整65536-patch ViT已验证。
- Release Clippy全targets、89项默认测试通过，14项硬件/模型测试默认忽略；本次显式运行的2项视觉NPU测试及native Attention CPU contract通过。
- 所有正式算子、模型及基线runner正常退出，受控频率设置恢复。
