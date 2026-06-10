# H0 实验入口

`h0` 目录现在包含两类入口：

1. `run_h0.py`：执行 H0 trace 回放与指标闭环 smoke。
2. `run_h1245.py`：基于同一套 H0 trace 加载逻辑，执行 H1/H2/H4/H5 实验。

核心仿真器仍复用 `pre实验/sim.py`，H0 负责统一真实 trace 接入、结果输出和配置快照。

## ShareGPT trace 路径

默认 ShareGPT trace 路径：

```text
/DATACENTER3/zhenxiang.wang/data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json
```

该路径已经写入 `run_h0.py` 与 `run_h1245.py` 的默认配置。运行时也可以用 `--trace-path` 覆盖。

## 运行 H0

H0 需要 JSON 配置文件。示例命令：

```bash
python3 h0/run_h0.py --config h0/configs/sharegpt_server_edge.json --out out/h0_sharegpt_server_edge
```

H0 输出：

- `events.jsonl`：聚合事件日志。
- `summary.csv`：多设备汇总指标。
- `config.resolved.json`：实际执行配置快照。
- `validation.json`：H0 闭环检查结果。
- 每个设备子目录也会输出自己的 `events.jsonl`、`summary.csv`、`config.resolved.json`。

## 运行 H1/H2/H4/H5

默认一次运行全部 H1/H2/H4/H5：

```bash
python3 h0/run_h1245.py --out out/h1245
```

只运行部分实验：

```bash
python3 h0/run_h1245.py --experiments h1,h2 --out out/h1245_quick
```

使用 synthetic trace 做快速检查：

```bash
python3 h0/run_h1245.py --trace-source synthetic --max-requests 240 --out out/h1245_synthetic_smoke
```

使用 ShareGPT trace 并限制请求数：

```bash
python3 h0/run_h1245.py --trace-source sharegpt --max-sessions 200 --max-requests 500 --out out/h1245_sharegpt
```

## H3：LMCache/vLLM 策略层适配

`h3_lmcache_adapter.py` 提供 H3 所需的策略层插件接口，不直接修改 vLLM 或 LMCache 内核。它暴露以下 hook：

- `on_admit -> COP`：对象画像更新与 admission 记录。
- `on_reuse -> RRS`：restore/recompute 判决。
- `on_pressure -> TMS`：内存压力下的降级尝试。
- `on_evict -> LPE`：offload/drop 受害者选择。

适配器支持两种绑定方式：

```python
from h3_lmcache_adapter import EdgeKVTiersPolicy, attach_to_cache

policy = EdgeKVTiersPolicy(cfg, policy="tiered", rrs_mode="rrs")
cache = attach_to_cache(cache, policy)
```

如果 cache 对象有 `register_hook(name, fn)`，会优先使用注册式 hook；否则会写入 `cache.on_admit`、`cache.on_reuse`、`cache.on_pressure`、`cache.on_evict` 属性。

`run_h1245.py` 可输出 H3 接入契约：

```bash
python3 h0/run_h1245.py --experiments h1 --h3-contract --out out/h1245_h3_contract
```

生成 H0 到 H3 hook 语义对齐样例：

```bash
python3 h0/run_h1245.py --experiments h1 --h3-contract --emit-h3-events --h3-event-limit 100 --out out/h1245_h3_contract
```

新增 H3 文件：

- `h3_adapter_contract.json`：LMCache/vLLM hook、字段和验收契约。
- `h3_hook_events.sample.jsonl`：由 H0 事件转换得到的 H3 hook 日志样例。

## H1/H2/H4/H5 输出

`run_h1245.py` 会在输出根目录下生成：

- `trace.resolved.json`：本次 trace 的来源、对象数、请求数和 `token_ref`。
- `summary.json`：各实验运行耗时和输出目录。
- `h1/h1_results.csv`、`h1/h1_summary.csv`、`h1/config.json`。
- `h2/h2_results.csv`、`h2/h2_summary.csv`、`h2/config.json`。
- `h4/h4_results.csv`、`h4/h4_summary.csv`、`h4/config.json`。
- `h5/h5_grid.csv`、`h5/h5_objects.csv`、`h5/h5_tau.csv`、`h5/config.json`。

## 当前 H0 ShareGPT smoke 结果

最近一次 H0 ShareGPT server/edge 回放结果：

- `passed = true`
- 设备：`server_sharegpt`、`edge_sharegpt`
- 对象数：`120`
- 请求数：`800`
- 聚合事件数：`1600`
- `token_ref = 116173`
- server p95 TTFT：`135.93 ms`
- edge p95 TTFT：`152.763 ms`
- 两端 `epsilon_ok = true`

结果解释：

- server 与 edge 使用同一 ShareGPT trace 和同一 `token_ref`，指标可直接对照。
- 两端都满足 `epsilon_norm = 0.0002` 的质量预算。
- server 显存峰值为 `1199.82 MB / 1200 MB`，edge 为 `519.87 MB / 520 MB`，内存压力足够且未超限。
- edge 的 resident hit rate 更低、offload hit rate 更高，符合边缘侧显存更紧的预期。
- edge p95/mean TTFT 更高，主要来自更低带宽和更高反序列化成本。

## 版本控制约定

仓库 `.gitignore` 已设置为默认只发现新增 `.py`、`.md` 和 `.gitignore` 文件。实验输出、cache 和临时结果不会上传：

- `out/`
- `__pycache__/`
- `*.pyc`
