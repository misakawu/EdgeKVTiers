# H3 TODO

## 目标

把当前 H3 从“策略层契约 + H0/H3 语义样例”推进到“真实 LMCache/vLLM 可部署性证据”。

H3 通过标准：

1. `COP/TMS/LPE/RRS` 至少以最小功能接入 LMCache/vLLM 或等价 cache layer。
2. 真实 workload 可跑完，错误率可控，事件日志完整。
3. `T_policy_ms` p95 < 1 ms，且明显小于 cache 命中或避免重算带来的收益。
4. 服务器端和边缘端均完成 smoke test。
5. H0 模拟器事件与 H3 真实 hook/event 语义一致，不一致样本有解释。

## 环境要求与当前状态

官方参考：

1. vLLM GPU 安装文档：`https://docs.vllm.ai/en/latest/getting_started/installation/gpu/`
2. LMCache 安装文档：`https://docs.lmcache.ai/getting_started/installation.html`
3. LMCache quickstart：`https://docs.lmcache.ai/getting_started/quickstart.html`

最低要求：

1. OS: Linux。
2. Python: vLLM 文档支持 3.10-3.13；LMCache 官方安装示例使用 Python 3.12。
3. NVIDIA GPU: vLLM 要求 compute capability >= 7.5。
4. CUDA/PyTorch/vLLM/LMCache 版本必须匹配。LMCache 文档当前主推 CUDA 13.0 / CUDA 12.9 wheel，并给出 vLLM/LMCache 兼容矩阵。
5. 需要安装 `torch`、`transformers`、`vllm`、`lmcache`。
6. 真实 H3 smoke 需要能启动 vLLM OpenAI-compatible server，并通过 `LMCacheConnectorV1` 或 `LMCacheMPConnector` 接入 LMCache。

当前环境检查结果（2026-06-12）：

1. Python: `Python 3.13.13`
2. Python path: `/DATACENTER3/zhenxiang.wang/miniforge3/bin/python3`
3. GPU: 3 x `NVIDIA GeForce RTX 2080 Ti`
4. GPU compute capability: `7.5`
5. GPU memory: 每张 11264 MiB
6. Driver: `525.105.17`
7. `nvidia-smi` 显示 CUDA Version: `12.0`
8. 当前 GPU 均已有进程占用，且部分 GPU utilization 较高。
9. 当前 Python 环境未安装：`torch`、`transformers`、`vllm`、`lmcache`
10. `import torch` 失败：`ModuleNotFoundError: No module named torch`

当前判定：

1. 硬件 compute capability 满足 vLLM 最低要求，RTX 2080 Ti 可用于小模型 smoke。
2. 当前 Python 3.13 可满足 vLLM 文档范围，但不建议作为 LMCache 主环境；LMCache 官方示例使用 Python 3.12，后续应新建 Python 3.12 独立环境。
3. 当前环境不满足真实 LMCache/vLLM 运行要求，因为核心包未安装。
4. 当前驱动较旧，`nvidia-smi` 只显示 CUDA 12.0；直接安装 LMCache 当前 CUDA 12.9/13.0 wheel 可能不匹配。优先路线是新建 Python 3.12 环境后安装与驱动可用 CUDA 版本兼容的 `torch/vllm/lmcache`，若 CUDA 12.9 wheel 不能运行，需要升级 NVIDIA driver 或使用容器/另一台环境。
5. 11GB 显存适合 `Qwen/Qwen3-0.6B`、小型 Qwen/Llama 系列或量化模型 smoke，不适合直接跑 7B/8B 全精度主实验。

环境准备 TODO：

- [x] 新建独立 Python 3.12 环境：`conda env h3-lmcache`，路径 `/DATACENTER3/zhenxiang.wang/miniforge3/envs/h3-lmcache`。
- [x] 安装并验证 PyTorch CUDA：`torch 2.6.0+cu124`，`torch.cuda.is_available() == True`。
- [x] 安装并验证 vLLM：`vllm 0.8.5.post1`，CLI/API server help 可用。
- [ ] 部分安装 LMCache：`lmcache 0.4.6` 顶层可导入，但 CUDA `c_ops` 因 `libcudart.so.13` 缺失未通过；当前只能作为 Python 包/接口检查，不能宣称 LMCache GPU connector 已可用。
- [x] 记录当前版本和阻塞原因到 `h3/out/env_check.json`。
- [ ] 当前阻塞：host driver `525.105.17` / `nvidia-smi` CUDA `12.0`，且本服务器不可重启，无法采用升级 driver 路线；真实 LMCache GPU connector 暂时 blocked。
- [ ] 可尝试降级路线：`torch==2.5.1+cu118`、`torchvision==0.20.1+cu118`、`vllm==0.6.6.post1`、`lmcache==0.3.15`；该组合有 cp312 wheel，但 LMCache 仍依赖 `cupy-cuda12x/cufile-python`，必须实际安装验证。
- [ ] 为 smoke 预留一张空闲 RTX 2080 Ti；当前三张卡均有进程占用，部分 utilization 较高。
- [ ] 本机继续 LMCache 真实 GPU connector 测试前，需要解决 CUDA13 runtime 或找到与 torch/vLLM ABI 匹配且不要求 CUDA13 的 LMCache wheel/source build。

## P0 - 真实 LMCache/vLLM Smoke

- [ ] 新增 `h3/configs/lmcache_h3.yaml`，开启 LMCache event 输出和可复现实验参数。
- [ ] 新增 `h3/run_lmcache_trace.py`，按 H0 trace 顺序调用 vLLM OpenAI-compatible API。
- [ ] 新增 `h3/lmcache_event_subscriber.py`，订阅 vLLM/LMCache KV events 并写入 H3 JSONL。
- [ ] 跑通 `vLLM only` baseline，输出 `h3/out/lmcache_real_smoke/vllm_only/*`。
- [ ] 跑通 `vLLM + LMCache default` baseline，输出 `h3/out/lmcache_real_smoke/lmcache_default/*`。
- [ ] 记录 `config.resolved.json`、`requests.jsonl`、`metrics.csv`、`summary.json`。

建议先用小模型和短 trace：

```bash
LMCACHE_CONFIG_FILE=h3/configs/lmcache_h3.yaml \
vllm serve Qwen/Qwen3-0.6B \
  --port 8000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --disable-log-requests \
  --kv-events-config '{"enable_kv_cache_events":"True","publisher":"zmq","topic":"kv-events"}'
```

## P1 - H0/H3 事件语义对齐

- [ ] 将真实 KV events 映射到 `h3/lmcache_adapter.py` 中的 H3 字段契约。
- [ ] 输出 `h3/out/lmcache_real_smoke/h3_hook_events.real.jsonl`。
- [ ] 生成 `h0_h3_alignment.csv`，对齐 `h0_event_id`、`lmcache_event_id`、`object_id`、`hook`、`hit`、`ttft_ms`。
- [ ] 统计并解释不一致样本，例如 chunk 粒度和对象粒度不一致、prefix hash 不一致、异步 eviction 延迟。
- [ ] 保证 `semantic_match=true` 的样本占主体，不一致项必须写入 summary。

## P2 - 策略旁路计时

- [ ] 让 `EdgeKVTiersPolicy` 在真实事件流上旁路执行，不直接干预 LMCache。
- [ ] 记录每次 `cop_update`、`rrs_on_reuse`、`tms_then_lpe`、`lpe_choose_victim` 的 `T_policy_ms`。
- [ ] 输出 `policy_overhead.csv`，至少包含 p50、p95、p99、max。
- [ ] 通过门槛：`T_policy_ms` p95 < 1 ms。

## P3 - 压力型 H3 Smoke

- [ ] 构造低显存预算或高请求量 trace，强制触发 pressure/eviction。
- [ ] 确认样例中覆盖 `on_admit`、`on_reuse`、`on_pressure`、`on_evict`。
- [ ] 确认事件中出现 `tms_action=downgrade` 或 `lpe_action=offload/drop`。
- [ ] 重新生成 `h3/out/pressure_smoke/*`。

## P4 - 控制面干预

- [ ] 先接 LPE：根据 keep score 对高价值对象执行 pin/keep，对低价值对象允许 evict/clear。
- [ ] 再接 RRS：根据 `c_restore_ms <= c_recomp_ms` 记录或控制 restore/recompute 路径。
- [ ] 最后接 TMS：用 LMCache compress/serde 或 KIVI/H2O 等价机制承载 `full -> int8 -> int4 -> sparse-k`。
- [ ] 记录真实 `c_mig_ms`，替换当前样例中的 0.0 占位值。

## P5 - H3 对照实验

对照组：

1. `vLLM only`
2. `vLLM + LMCache default`
3. `vLLM + LMCache LRU/LFU`
4. `vLLM + LMCache + EdgeKVTiers LPE/RRS`
5. `vLLM + LMCache + EdgeKVTiers COP/TMS/LPE/RRS`

指标：

1. p50/p95 TTFT
2. hit rate / hit tokens
3. `M_peak`
4. restore latency
5. `T_policy_ms`
6. `c_mig_ms`
7. qloss 或 proxy quality loss
8. error_count

输出：

- [ ] `h3/out/h3_lmcache_compare/summary.csv`
- [ ] `h3/out/h3_lmcache_compare/summary.json`
- [ ] `h3/out/h3_lmcache_compare/h3_hook_events.real.jsonl`
- [ ] `h3/out/h3_lmcache_compare/policy_overhead.csv`

## P6 - 后续主实验衔接

- [ ] 将 H3 真实接入结果接到 E2 主实验配置。
- [ ] 在服务器限显存环境重跑关键配置。
- [ ] 在 Jetson 或边缘 GPU 上重跑缩小 trace。
- [ ] 输出端侧拆解表：p95 TTFT、qloss、`M_peak`、`T_policy_ms`、`c_mig_ms`。
- [ ] 若真实 hook 不足以支持完整策略，保留“trace 仿真 + 少量真实验证”的 B3 缩小方案。
