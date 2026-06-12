# H3 Work Log

## 2026-06-12 - Adapter Contract and Output Separation

### 已完成

1. 新增独立 `h3/` 目录，用于承载 H3 真实接入与策略层适配代码。
2. 将原先位于 H0 侧的 H3 adapter 移动到 `h3/lmcache_adapter.py`。
3. 新增 `h3/run_h3.py`，用于生成 H3 adapter smoke 输出。
4. 将 H3 输出放到 `h3/out/...`，与 H0 的 `h0/out/...` 分离。
5. 保留 H0/H1/H2/H4/H5 解析仿真与可视化输出在 `h0/out/...`。

### 当前代码能力

`h3/lmcache_adapter.py` 已实现：

1. `EdgeKVTiersPolicy`
   - 策略层对象，不直接 import LMCache 或 vLLM。
   - 支持 `COP/TMS/LPE/RRS` 四类策略回调。

2. `attach_to_cache(cache, policy)`
   - 优先支持 `cache.register_hook(name, fn)`。
   - 回退支持 `cache.on_admit/on_reuse/on_pressure/on_evict` 属性赋值。

3. `HookResult`
   - 统一记录 hook、action、object_id、tier 变化和策略开销字段。

4. `JsonlHookLogger`
   - 支持把策略回调结果写入 JSONL。

5. H0 到 H3 的事件转换
   - `h0_event_to_h3(...)` 将 H0 enriched event 转为 H3 hook schema。
   - `write_h3_contract(...)` 输出接入契约。
   - `write_h3_event_sample(...)` 输出 H3 hook 语义样例。

### Hook 与模块映射

1. `on_admit -> COP`
   - 入口：`EdgeKVTiersPolicy.cop_update`
   - 当前记录 admission、object id、当前 tier。

2. `on_reuse -> RRS`
   - 入口：`EdgeKVTiersPolicy.rrs_on_reuse`
   - 当前按 `restore` 与 `recompute` 代价选择动作。

3. `on_pressure -> TMS`
   - 入口：`EdgeKVTiersPolicy.tms_then_lpe`
   - 当前按 H0 keep score 选择降级候选。

4. `on_evict -> LPE`
   - 入口：`EdgeKVTiersPolicy.lpe_choose_victim`
   - 当前按 keep score 选择 victim，并根据复用概率选择 offload/drop。

### 已生成输出

1. `h3/out/adapter_smoke/h3_adapter_contract.json`
2. `h3/out/adapter_smoke/h3_hook_events.sample.jsonl`
3. `h3/out/adapter_smoke/summary.json`
4. `h3/out/h1245_h3_synthetic_smoke/h3_adapter_contract.json`
5. `h3/out/h1245_h3_synthetic_smoke/h3_hook_events.sample.jsonl`

### 当前输出统计

`h3/out/adapter_smoke/h3_hook_events.sample.jsonl`：

1. rows: 20
2. hook 覆盖：`on_admit` 16 次，`on_reuse` 4 次
3. `on_pressure`: 0 次
4. `on_evict`: 0 次
5. `tms_action`: 全部 `none`
6. `lpe_action`: 全部 `resident`
7. `semantic_match`: 20/20 为 true
8. `T_policy_ms` p95: 0.0 ms
9. `M_peak` max: 227.64 MB

`h3/out/h1245_h3_synthetic_smoke/h3_hook_events.sample.jsonl`：

1. rows: 40
2. hook 覆盖：`on_admit` 26 次，`on_reuse` 14 次
3. `on_pressure`: 0 次
4. `on_evict`: 0 次
5. `tms_action`: 全部 `none`
6. `lpe_action`: 全部 `resident`
7. `semantic_match`: 40/40 为 true
8. `T_policy_ms` p95: 0.0 ms
9. `M_peak` max: 347.28 MB

### 与 H3 预期的差距

当前状态应判定为“部分完成 / 未通过 H3 gate”。

已完成的是：

1. 策略层 adapter contract。
2. H3 hook 字段契约。
3. H0 enriched events 到 H3 hook sample 的语义转换。
4. H3 输出目录独立化。

尚未完成的是：

1. 尚未接入真实 LMCache/vLLM 运行。
2. 尚未产生真实 KV events。
3. 尚未比较 `LMCache default` 与 `EdgeKVTiers policy` 的收益。
4. 尚未触发 `on_pressure/on_evict`。
5. `T_policy_ms=0.0` 目前来自转换样例，不是真实 hook 计时。
6. `c_mig_ms=0.0` 仍是占位，真实值需要接入压缩、量化或稀疏化路径后替换。
7. 尚未完成服务器端和端侧 smoke。

### 当前可复现命令

生成 H3 adapter smoke：

```bash
python3 h3/run_h3.py \
  --trace-source synthetic \
  --max-requests 30 \
  --event-limit 20 \
  --out h3/out/adapter_smoke
```

编译检查：

```bash
python3 -m py_compile h3/run_h3.py h3/lmcache_adapter.py
```

## 下一步

优先执行 `h3/TODO.md` 中的 P0-P3：

1. 真实 LMCache/vLLM smoke。
2. 真实 KV events 订阅与 H3 schema 转换。
3. 策略旁路计时，补齐真实 `T_policy_ms`。
4. 压力型 smoke，强制覆盖 `on_pressure/on_evict`。

## 2026-06-12 - Environment Preparation Attempt

### 已完成

1. 创建独立 conda 环境 `h3-lmcache`。
2. 环境路径：`/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache`。
3. Python 版本：`3.12.13`。
4. 记录环境检查结果到 `h3/out/env_check.json`。

### 当前主机状态

1. GPU: 3 x `NVIDIA GeForce RTX 2080 Ti`。
2. GPU compute capability: `7.5`，满足 vLLM 小模型 smoke 的最低硬件要求。
3. 单卡显存：11264 MiB。
4. Driver: `525.105.17`。
5. `nvidia-smi` CUDA Version: `12.0`。
6. `nvcc` 不存在。
7. `docker` / `podman` 不存在。
8. 当前新环境尚未安装 `torch`、`transformers`、`vllm`、`lmcache`。

### 阻塞原因

真实 LMCache/vLLM 环境尚未完成。原因是当前主机软件栈过旧：LMCache 当前官方安装路线要求 CUDA 12.1+，主推 CUDA 12.9/13.0 wheel；当前 driver 只暴露 CUDA 12.0，且没有 `nvcc` 或容器工具可作为 fallback。

### 后续动作

1. 优先升级 NVIDIA driver 到与目标 CUDA wheel 兼容的版本，或切换到已满足 CUDA 12.1+/12.9+/13.0 的服务器。
2. 若有管理员支持，可安装 Docker/Podman 并使用官方容器路线。
3. driver/container 就绪后，在 `h3-lmcache` 环境中继续安装 `torch`、`transformers`、`vllm`、`lmcache`。
4. 完成安装后重新生成 `h3/out/env_check.json`，并运行真实 H3 smoke。

## 2026-06-12 - Downgrade Compatibility Check

### 结论

降低版本有一定可行性，但不是确定可用。当前最有希望的组合是：

1. `torch==2.5.1` / `torchvision==0.20.1`
2. PyTorch CUDA wheel 优先尝试 `cu118`
3. `vllm==0.6.6.post1`
4. `lmcache==0.3.15`

### 依据

1. `vllm==0.6.6.post1` 有 `cp38-abi3-manylinux1_x86_64` wheel，可被 Python 3.12 环境使用。
2. `vllm==0.6.6.post1` 元数据固定依赖 `torch==2.5.1`、`torchvision==0.20.1`、`xformers==0.0.28.post3`。
3. `lmcache==0.3.15` 有 `cp312` wheel，可被当前 `h3-lmcache` 环境使用。
4. `lmcache==0.3.15` 元数据仍包含 `cupy-cuda12x`、`cufile-python`、`nixl` 等依赖，因此即使 vLLM/PyTorch 走 cu118，LMCache 自身仍可能拉 CUDA 12.x 组件。

### 风险

1. 当前 driver `525.105.17` 对 CUDA 12.x 新 wheel 支持不足。
2. `cupy-cuda12x` 和 `cufile-python` 可能要求比当前 driver 更高的软件栈。
3. 本机没有 `nvcc`，源码编译 fallback 不现实。
4. 本机没有 Docker/Podman，官方容器 fallback 暂不可用。

### 建议尝试命令

```bash
conda run -n h3-lmcache python -m pip install   torch==2.5.1 torchvision==0.20.1   --index-url https://download.pytorch.org/whl/cu118

conda run -n h3-lmcache python -m pip install   vllm==0.6.6.post1 lmcache==0.3.15

conda run -n h3-lmcache python - <<PY
import torch
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
import vllm
print("vllm", vllm.__version__)
import lmcache
print("lmcache", getattr(lmcache, "__version__", "unknown"))
PY
```

若上述安装或导入失败，应优先升级 driver 或切换到兼容服务器，不建议在当前主机上源码编译 vLLM/LMCache。

## 2026-06-12 - Local Downgrade Install Result

### 已安装

1. Conda env: `h3-lmcache`
2. Python: `3.12.13`
3. PyTorch: `2.6.0+cu124`
4. vLLM: `0.8.5.post1`
5. LMCache: `0.4.6`
6. `libstdcxx-ng` installed in the conda env.

### 已验证通过

1. `torch.cuda.is_available() == True`
2. GPU 可见：`NVIDIA GeForce RTX 2080 Ti`
3. `import vllm` 通过。
4. vLLM CLI / API server help 可用。
5. 当前 H3 adapter smoke 在该环境下通过：`h3/out/env_install_smoke_final`。
6. `import lmcache` 在设置 `LD_LIBRARY_PATH` 后通过。

运行该环境时建议带上：

```bash
export LD_LIBRARY_PATH=/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib:/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib/python3.12/site-packages/torch/lib:/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
```

### 本地补丁

为绕过 LMCache 0.4.6 在无有效 OpenTelemetry LoggerProvider 时导入崩溃的问题，已对以下环境文件做最小补丁：

`/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib/python3.12/site-packages/lmcache/logging.py`

备份文件：

`/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib/python3.12/site-packages/lmcache/logging.py.bak_h3`

补丁逻辑：仅当 OpenTelemetry logger provider 不是 `ProxyLoggerProvider` 时才添加 `LoggingHandler`。

### 仍未通过

1. `lmcache.c_ops` CUDA backend 未通过。
2. 失败原因：`libcudart.so.13: cannot open shared object file`。
3. 安装 `cupy-cuda13x` 后仍未解决 `libcudart.so.13` 动态链接问题，并且会破坏 vLLM 0.8.5 所需的 numpy/OpenTelemetry 版本约束，因此已回退。
4. 当前不能宣称真实 LMCache GPU connector 已可用；只能宣称本机 vLLM 环境可用、LMCache Python 包可导入但 GPU backend 未就绪。

### 后续建议

1. 先用本环境做 `vLLM only` smoke 和 H3 adapter/trace tooling。
2. LMCache 真实 GPU connector 需要继续解决 CUDA13 runtime 或寻找/编译与 `torch 2.6.0+cu124` ABI 匹配的 LMCache wheel。
3. 若必须继续在本机完成 LMCache connector，可考虑从 LMCache 源码在该 conda env 中本地编译 c_ops，但需要先补齐 build 工具链和 CUDA headers。当前主机没有 `nvcc`，风险较高。

## 2026-06-12 - No-Reboot Constraint

### 结论

本服务器不可重启，因此不能采用升级 NVIDIA driver 的路线来满足 LMCache CUDA 13 runtime 需求。

### 当前可用边界

1. 保留并使用 `h3-lmcache` 环境。
2. 可继续执行 `vLLM only` smoke、trace 回放工具、H3 adapter contract 和策略旁路计时。
3. 暂不能执行真实 `vLLM + LMCache GPU connector` 主实验。
4. LMCache 当前状态是 Python 包可导入，但 CUDA backend 未就绪：`lmcache.c_ops` 需要 `libcudart.so.13`。

### 后续策略

1. 在本机不可重启约束下，优先完成不依赖 LMCache GPU backend 的实验基础设施。
2. 将 `LMCache default`、`LMCache + EdgeKVTiers` 真实实验标记为 blocked。
3. 如必须继续本机完成 LMCache connector，只能尝试源码编译 LMCache CUDA ops 或寻找不依赖 CUDA13 runtime 的兼容 wheel；当前无 `nvcc`，风险较高。

