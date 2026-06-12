# H3 Work Log

## 2026-06-12 - H3 已实现内容与缺口

### 已完成

1. 新增独立 `h3/` 目录，用于承载 H3 策略层适配代码和 H3 专属输出。
2. 将原 H0 侧 adapter 移动到 `h3/lmcache_adapter.py`。
3. 新增 `h3/run_h3.py`，用于生成 H3 adapter smoke 输出。
4. H3 输出已放到 `h3/out/...`，与 H0 的 `h0/out/...` 分离。
5. `h3/lmcache_adapter.py` 已实现 H3 策略层契约：
   - `EdgeKVTiersPolicy`
   - `HookResult`
   - `JsonlHookLogger`
   - `attach_to_cache(cache, policy)`
   - `h0_event_to_h3(...)`
   - `write_h3_contract(...)`
   - `write_h3_event_sample(...)`
6. 已实现 COP/TMS/LPE/RRS 到 hook 的映射：
   - `on_admit -> COP -> cop_update`
   - `on_reuse -> RRS -> rrs_on_reuse`
   - `on_pressure -> TMS -> tms_then_lpe`
   - `on_evict -> LPE -> lpe_choose_victim`
7. 已生成并保留 H3 adapter contract 与 hook 样例：
   - `h3/out/h1245_h3_synthetic_smoke/h3_adapter_contract.json`
   - `h3/out/h1245_h3_synthetic_smoke/h3_hook_events.sample.jsonl`
8. H3 代码可编译：

```bash
python3 -m py_compile h3/run_h3.py h3/lmcache_adapter.py
```

### 已实现结果的性质

当前 H3 已完成的是“策略层插件契约 + H0/H3 语义样例”。它证明了 H0 模拟器事件可以转换为 H3 hook schema，也给出了未来挂到 LMCache/vLLM 外围时需要满足的字段契约。

当前样例覆盖情况：

1. `h3/out/adapter_smoke/h3_hook_events.sample.jsonl`
   - rows: 20
   - `on_admit`: 16
   - `on_reuse`: 4
   - `on_pressure`: 0
   - `on_evict`: 0
   - `semantic_match`: 20/20 true
   - `T_policy_ms` p95: 0.0 ms
2. `h3/out/h1245_h3_synthetic_smoke/h3_hook_events.sample.jsonl`
   - rows: 40
   - `on_admit`: 26
   - `on_reuse`: 14
   - `on_pressure`: 0
   - `on_evict`: 0
   - `semantic_match`: 40/40 true
   - `T_policy_ms` p95: 0.0 ms

### 未完成

1. 尚未完成真实 LMCache/vLLM GPU connector 接入。
2. 尚未产生真实 LMCache KV events。
3. 尚未完成 `LMCache default`、`LMCache + LPE/RRS`、`LMCache + COP/TMS/LPE/RRS` 的真实对照实验。
4. 当前样例没有触发 `on_pressure/on_evict`，因此尚未覆盖真实 pressure/eviction 路径。
5. 当前 `T_policy_ms=0.0` 来自 H0 转换样例，不是真实 hook 计时结果。
6. 当前 `c_mig_ms=0.0` 仍是占位，真实值需要接入压缩、量化或稀疏化机制后替换。
7. 尚未达到 H3 gate：`T_policy p95 < 1 ms`、完整 hook 日志、真实语义对齐、服务器/端侧 smoke。

### 当前可复现命令

生成 H3 adapter smoke：

```bash
python3 h3/run_h3.py \
  --trace-source synthetic \
  --max-requests 30 \
  --event-limit 20 \
  --out h3/out/adapter_smoke
```

在已配置环境中运行 H3 adapter smoke：

```bash
conda run -n h3-lmcache python h3/run_h3.py \
  --trace-source synthetic \
  --max-requests 10 \
  --event-limit 5 \
  --out h3/out/env_install_smoke_final
```

## 2026-06-12 - 本机环境配置结果与限制

### 已完成

1. 创建 conda 环境：`h3-lmcache`。
2. 环境路径：`/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache`。
3. Python: `3.12.13`。
4. PyTorch: `2.6.0+cu124`。
5. vLLM: `0.8.5.post1`。
6. LMCache: `0.4.6`。
7. 安装 `libstdcxx-ng` 到 conda 环境。
8. 记录环境检查结果：`h3/out/env_check.json`。
9. 验证 PyTorch CUDA 可用：
   - `torch.cuda.is_available() == True`
   - GPU: `NVIDIA GeForce RTX 2080 Ti`
10. 验证 vLLM 可用：
   - `import vllm` 通过
   - vLLM CLI/API server help 可用
11. 验证 H3 adapter smoke 可在该环境中运行：`h3/out/env_install_smoke_final`。
12. `import lmcache` 在设置 `LD_LIBRARY_PATH` 后可通过。

运行该环境时建议先设置：

```bash
export LD_LIBRARY_PATH=/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib:/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib/python3.12/site-packages/torch/lib:/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
```

### 本地补丁

为绕过 LMCache 0.4.6 在无有效 OpenTelemetry LoggerProvider 时导入崩溃的问题，已对当前 conda 环境中的 LMCache logging 做最小补丁。

补丁文件：

`/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib/python3.12/site-packages/lmcache/logging.py`

备份文件：

`/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib/python3.12/site-packages/lmcache/logging.py.bak_h3`

补丁逻辑：仅当 OpenTelemetry logger provider 不是 `ProxyLoggerProvider` 时才添加 `LoggingHandler`。

### 未完成

1. LMCache CUDA `c_ops` backend 未通过。
2. 真实 `vLLM + LMCache GPU connector` 尚不可用。
3. 真实 `LMCache default` baseline 尚不可跑。
4. 真实 `LMCache + EdgeKVTiers` H3 主实验尚不可跑。
5. 当前只能执行：
   - `vLLM only` smoke / baseline
   - H3 adapter contract
   - trace tooling
   - 不依赖真实 LMCache GPU backend 的策略旁路计时

### 为什么当前不能实现真实 LMCache GPU connector

1. 本机 driver: `525.105.17`。
2. `nvidia-smi` 显示 CUDA Version: `12.0`。
3. 当前可用 PyTorch 栈是 `torch 2.6.0+cu124`。
4. LMCache 0.4.6 的 CUDA backend 需要 CUDA 13 runtime，导入 `lmcache.c_ops` 时失败：

```text
libcudart.so.13: cannot open shared object file
```

5. 尝试安装 `cupy-cuda13x` 后仍未解决动态链接问题，并且会破坏 vLLM 0.8.5 需要的 numpy/OpenTelemetry 版本约束，因此已回退。
6. 本机没有 `nvcc`，不能稳妥走源码编译 CUDA ops。
7. 本机没有 Docker/Podman，不能走官方容器路线。
8. 本服务器不可重启，因此不能采用升级 NVIDIA driver 的路线。

### 可行的实现方式

1. 当前本机可继续推进 `vLLM only` baseline 和 H3 工具链。
2. 若必须在本机完成真实 LMCache GPU connector，有两个理论可行方向：
   - 找到与 `torch 2.6.0+cu124` ABI 匹配、且不要求 CUDA 13 runtime 的 LMCache wheel。
   - 在本机补齐 CUDA toolkit、`nvcc`、编译工具链后，从 LMCache 源码本地编译 c_ops；该路线风险高，因为当前系统 Ubuntu 18.04、GCC 旧、无 `nvcc`。
3. 若允许换运行条件，最稳妥路线是使用可重启/可升级 driver 的机器，或带 Docker/Podman 的机器，部署支持 CUDA 13 runtime 的 LMCache/vLLM 环境。
4. 在当前不可重启服务器上，真实 LMCache GPU connector 实验应标记为 blocked；不要用当前环境产出 LMCache 主实验结论。
