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

### 具体报错记录

导入 LMCache CUDA backend 的验证命令：

```bash
LD_LIBRARY_PATH=/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib:/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib/python3.12/site-packages/torch/lib:/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH \
  conda run -n h3-lmcache python -c "import lmcache.c_ops"
```

当前输出中的关键失败信息：

```text
LMCache INFO: torch_dev=<module 'torch.cuda' from '/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache/lib/python3.12/site-packages/torch/cuda/__init__.py'>, torch_device_type=cuda
LMCache INFO: Skipping backend lmcache.xpu_ops: predicate returned False
LMCache WARNING: Failed to import backend lmcache.c_ops: libcudart.so.13: cannot open shared object file: No such file or directory
```

依赖一致性检查命令：

```bash
conda run -n h3-lmcache python -m pip check
```

当前输出：

```text
lmcache 0.4.6 requires cupy-cuda13x, which is not installed.
lmcache 0.4.6 requires opentelemetry-exporter-prometheus, which is not installed.
google-api-core 2.31.0 has requirement protobuf<8.0.0,>=5.29.6, but you have protobuf 4.25.9.
ERROR conda.cli.main_run:execute(148): `conda run python -m pip check` failed. (See above for error)
```

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


## 2026-06-13 - 当前设备环境最终分析与实验可行性

### 最新环境状态

旧环境 `h3-lmcache` 已不再作为当前工作环境使用。当前可用环境为：

1. Conda 环境：`h3-lmcache-blog`。
2. 环境路径：`/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache-blog`。
3. Python：`3.12.13`。
4. PyTorch：`2.5.1+cu124`。
5. vLLM：`0.6.6.post1`。
6. LMCache：`0.3.15`，从源码编译安装。
7. `nvcc`：`12.4.131`。
8. GPU：`3 x NVIDIA GeForce RTX 2080 Ti`，每张约 11 GiB 显存，compute capability 7.5。
9. NVIDIA driver：`525.105.17`。
10. `nvidia-smi` 显示 CUDA Version：`12.0`。

运行 LMCache/vLLM 检查时建议设置：

```bash
export LD_LIBRARY_PATH=/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache-blog/lib:/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache-blog/targets/x86_64-linux/lib:/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache-blog/lib/python3.12/site-packages/torch/lib:/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache-blog/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
```

### 已验证通过

1. `torch.cuda.is_available() == True`。
2. `import vllm` 通过。
3. `import lmcache.c_ops` 通过。
4. `pip check` 通过。
5. `python3 -m py_compile h0/run_h0.py h0/run_h1245.py h3/run_h3.py h3/lmcache_adapter.py` 通过。
6. H1/H2 小规模探针通过，输出：`h0/out/current_env_h1245_probe`。
7. H4/H5 小规模探针通过，输出：`h0/out/current_env_h45_probe`。
8. H3 adapter 探针通过，输出：`h3/out/current_env_h3_probe`。
9. LMCache 源码编译路线已解决此前预编译 wheel 的 `c_ops` ABI 问题。

LMCache 源码编译的关键原因和参数：

1. 当前 `torch._C._GLIBCXX_USE_CXX11_ABI == False`。
2. 预编译 LMCache wheel 的 C++ ABI 与当前 torch 不匹配，会导致 `lmcache.c_ops` undefined symbol。
3. 源码编译时使用 `ENABLE_CXX11_ABI=0` 匹配当前 torch。
4. RTX 2080 Ti 使用 `TORCH_CUDA_ARCH_LIST=7.5`。
5. 当前环境使用 conda 内 `nvcc 12.4.131` 和 CUDA runtime/library。
6. `setup.py` 中补入了 CUDA include、library_dirs 和 rpath，解决最终链接阶段 `ld: cannot find -lcudart` 的问题。

### 当前可以进行的实验

当前设备可以进行不依赖真实 `vLLM + LMCacheConnectorV1` 的实验：

1. H0 trace 回放与指标闭环。
2. H1 生命周期策略仿真。
3. H2 restore-vs-recompute 仿真。
4. H4 quality 维度收益仿真。
5. H5 质量-带宽相变面仿真。
6. H3 策略层 contract、hook schema、H0/H3 事件语义样例。
7. H3 策略旁路计时，即不干预真实 LMCache 控制面，只记录 `T_policy_ms`。
8. E2 主实验的仿真版或先导版矩阵。
9. 小模型 `vLLM-only` smoke，例如 0.5B/0.6B/1.5B 级模型。
10. LMCache CUDA extension 可编译性证明。

这些实验可以作为论文前期证据和方法趋势验证，但不能替代正式真实系统主实验。

### 当前不能进行的实验

当前设备不能可靠完成以下正式实验：

1. 真实 `vLLM + LMCacheConnectorV1` 联调。
2. 真实 LMCache KV events 采集。
3. 真实 `LMCache default` baseline。
4. 真实 `LMCache + LPE/RRS` baseline。
5. 真实 `LMCache + COP/TMS/LPE/RRS` 完整 H3 主实验。
6. E2 正式主实验。
7. E3 正式消融实验。
8. E4 使用真实引擎代价标定的核心技术验证。
9. E5 端侧真实验证。
10. 规划中以 `Qwen2.5-7B-Instruct`、`Llama-3.1-8B-Instruct` 或 `DeepSeek-R1-Distill-Qwen-7B` 为主模型的完整多 workload、多预算、重复实验矩阵。

### 不可进行实验的原因链

#### 1. 真实 LMCache/vLLM connector 不可用

1. 当前可运行的低驱动折中环境是 `torch 2.5.1+cu124 + vllm 0.6.6.post1 + lmcache 0.3.15`。
2. LMCache `c_ops` 已通过源码编译解决，说明 CUDA extension 本身可以在当前用户态环境中工作。
3. 但 LMCache 0.3.15 安装后的 connector 代码引用：

```text
vllm.distributed.kv_transfer.kv_connector.v1
```

4. 当前 `vllm 0.6.6.post1` 只有旧路径，没有 `kv_connector.v1` API。
5. 因此导入 `LMCacheConnectorV1` 会失败：

```text
ModuleNotFoundError: No module named 'vllm.distributed.kv_transfer.kv_connector.v1'
```

6. 这个问题不是 `lmcache.c_ops` 问题，而是 LMCache connector 与 vLLM API 版本不匹配。

#### 2. 为什么不能直接升级到新 LMCache/vLLM 组合

1. 本机 driver 是 `525.105.17`。
2. `nvidia-smi` 显示 CUDA Version 是 `12.0`。
3. LMCache 新版本和新 vLLM connector 生态偏向 CUDA 12.9 / CUDA 13。
4. 旧环境 `lmcache 0.4.6` 路线失败时，关键错误是：

```text
libcudart.so.13: cannot open shared object file
```

5. 这说明新版 LMCache CUDA backend 依赖 CUDA 13 runtime。
6. 不更新 driver 时，用户态安装 CUDA toolkit、`nvcc` 或 pip/conda CUDA runtime 只能补齐编译工具和部分动态库，不能替代内核态 NVIDIA driver 对 CUDA 12.9/13 runtime 的支持。
7. 因此，当前设备无法稳定采用新版 LMCache/vLLM 官方推荐路线。
8. 根因最终收敛到：driver `525.105.17` 过低，`nvidia-smi` 暴露能力只到 CUDA `12.0`，无法支撑新版 LMCache CUDA 13 runtime 路线。

#### 3. 为什么正式 E2/E3/E4/E5 不可完成

1. 正式 E2 要求真实 `vLLM + LMCache`、真实 KV events、真实 offload/cache hook 和统一 trace 日志。
2. 当前真实 connector 未打通，因此无法产生真实 LMCache KV events，也无法运行 `LMCache default` 或 `LMCache + EdgeKVTiers` 对照组。
3. E3 消融依赖 E2 的真实代表性 cell；E2 未完成时，E3 正式版没有可信输入。
4. E4 正式版要求真实引擎测得的 `c_restore/c_recomp/c_mig`；当前只能用仿真或占位值，不能作为正式真实系统验证。
5. E5 端侧验证要求真实引擎或边缘设备；当前服务器不是端侧设备，且真实 connector 未打通。
6. 因此 E2/E3/E4/E5 的正式版均不可在当前设备上完成。
7. 该限制的底层原因仍然回到驱动和版本链：低 driver 限制新版 LMCache/vLLM 路线，旧 vLLM 又缺 LMCache connector v1 API。

#### 4. 为什么 7B/8B 正式主模型矩阵不可完成

1. 当前 GPU 是 RTX 2080 Ti，每张约 11 GiB 显存。
2. 规划中的正式主实验要求 `Qwen2.5-7B-Instruct`，并至少补充 `Llama-3.1-8B-Instruct` 或 `DeepSeek-R1-Distill-Qwen-7B` 的代表性 cell。
3. 7B/8B 模型在 11 GiB 显存上即使勉强运行，也通常需要量化、低 batch、短上下文和严格显存控制。
4. 正式实验还要求 ShareGPT/RAG 多 workload、3x3 `(M_budget, epsilon)`、每 cell 多次重复、记录完整 trace 和 cache/offload 行为。
5. 当前三张 GPU 经常已有进程占用，显存和算力都不足以稳定承担正式矩阵。
6. 因此当前设备最多适合小模型 smoke，不适合正式主实验矩阵。

### 可行推进方案

1. 继续在当前设备完成 H0-H5 全量仿真。
2. 继续完善 H3 策略层 contract、hook schema、policy overhead 和 H0/H3 语义对齐样例。
3. 将 E2 做成仿真版或先导版矩阵，用于验证趋势和生成论文前期证据。
4. 小模型只用于 `vLLM-only` smoke，不宣称真实 LMCache 主实验完成。
5. 真实系统实验迁移到 driver 更新、CUDA runtime 匹配、显存更大且 GPU 空闲的机器。
6. 若必须在当前设备且不升级 driver 上继续攻关，只能尝试手动改写 LMCache connector 适配 `vllm 0.6.6.post1` 老接口；该路线属于额外工程风险，不应作为正式论文主路径。
