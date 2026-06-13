# EdgeKVLife
面向边缘多负载推理的质量分级 KV/Cache 在线生命周期调度

## 当前设备与环境结论（2026-06-13）

当前设备在不升级 NVIDIA driver 的前提下，不能完整完成项目正式版目标，但可以完成 H0/H1/H2/H4/H5 仿真主线、H3 策略层原型、小模型 smoke 和前期论文证据。

### 当前可用环境

- Conda 环境：`h3-lmcache-blog`
- 环境路径：`/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache-blog`
- GPU：`3 x NVIDIA GeForce RTX 2080 Ti`，每张约 11 GiB 显存，compute capability 7.5
- Driver：`525.105.17`
- `nvidia-smi` 显示 CUDA：`12.0`
- Python：`3.12.13`
- PyTorch：`2.5.1+cu124`，`torch.cuda.is_available() == True`
- vLLM：`0.6.6.post1`
- LMCache：`0.3.15`，已从源码编译安装
- `nvcc`：`12.4.131`
- `lmcache.c_ops`：可导入

### 能完成的实验

当前设备可以完成不依赖真实 `vLLM + LMCacheConnectorV1` 的实验：

1. H0 trace 回放与指标闭环。
2. H1 生命周期策略仿真。
3. H2 restore-vs-recompute 仿真。
4. H4 quality 维度收益仿真。
5. H5 质量-带宽相变面仿真。
6. H3 策略层 contract、hook schema、H0/H3 事件语义样例和策略开销旁路验证。
7. 小模型 `vLLM-only` smoke。
8. LMCache CUDA 扩展可编译性证明，即 `lmcache.c_ops` 在当前环境可用。

### 不能完成的正式实验

当前设备不能可靠完成正式版真实系统实验：

1. 真实 `vLLM + LMCacheConnectorV1` 联调。
2. 真实 LMCache KV events 采集。
3. 真实 `LMCache default` baseline。
4. 真实 `LMCache + EdgeKVTiers` 控制面干预。
5. E2 正式主实验。
6. E3 正式消融实验。
7. E4 使用真实引擎代价标定的核心技术验证。
8. E5 端侧真实验证。

### 原因链

不能完成正式真实系统实验的原因最终收敛到驱动和设备资源：

1. 本机 NVIDIA driver 是 `525.105.17`，`nvidia-smi` 只暴露 CUDA `12.0`。
2. LMCache 新版本和新 vLLM connector 生态偏向 CUDA 12.9 / CUDA 13。此前 `lmcache 0.4.6` 路线失败的根因是缺少 `libcudart.so.13`。
3. 不升级 driver 时，用户态安装 CUDA toolkit 或 `nvcc` 只能提供编译工具和部分用户态库，不能替代内核态 driver 对 CUDA 12.9/13 runtime 的支持。
4. 当前可行的折中环境退到 `torch 2.5.1+cu124 + vllm 0.6.6.post1 + lmcache 0.3.15`，并通过源码编译解决了 `lmcache.c_ops` 的 ABI 问题。
5. 但 `LMCache 0.3.15` 的 vLLM connector 代码要求 `vllm.distributed.kv_transfer.kv_connector.v1`，而 `vllm 0.6.6.post1` 没有该 API。
6. 因此当前环境处在版本夹缝中：旧 vLLM 可在低驱动环境运行但缺 connector v1，新 vLLM/新 LMCache 更可能需要更高 CUDA runtime 和 driver。
7. 设备显存只有 11 GiB/卡，且经常已有进程占用，不适合承担规划中的 `Qwen2.5-7B-Instruct`、`Llama-3.1-8B-Instruct`、多 workload、3x3 预算矩阵、每 cell 5 次重复的正式主实验。

### 推荐推进方式

1. 当前设备继续完成 H0-H5 全量仿真、H3 策略层 contract、policy overhead 和 E2 仿真版矩阵。
2. 真实系统实验单独迁移到 driver 更新、显存更大且 GPU 空闲的机器。
3. 如果坚持在当前设备且不升级 driver，只能尝试手动改写 LMCache connector 适配 `vllm 0.6.6.post1` 老接口；这属于额外工程攻关，不能保证完成正式主实验。
