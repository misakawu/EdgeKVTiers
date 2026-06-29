# §9 实际执行进度与结果（2026-06-29）

## 9.1 探路：放大工作集，报告 hit 纹丝不动（异常信号）
固定 `h1_lru`、`budget 0.85`、TP2、Qwen2.5-7B、float16，逐档放大工作集探路（chunk_words≤320，未触发 >2032 过滤）：

| 档 | global/warm/cold 池 | warm/cold ratio | chunk_words | requests | warm/cold 唯一选中 | 报告 hit_rate | evictions |
|----|------|------|------|------|------|------|------|
| ws1 | 5/40/300 | 0.60/0.20 | 200 | 512 | 40/92 | 0.9414 | — |
| ws2 | 5/80/600 | 0.60/0.20 | 320 | 768 | 80/152 | 0.9372 | 5668 |
| ws3r | 5/800/800 | **0.95/0.95** | 320 | 768 | **730/730**(对象级复用=0) | 0.9279 | 14950 |

工作集放大 5 倍、对象级复用打到 0、驱逐爆炸到 14950，报告 hit 只从 0.941 微降到 0.928。
**异常铁证**：ws3r 的 `gpu_prefix_cache_lookup_misses = 768`，恰好等于请求数（每条只记 1 个 miss）。

## 9.2 决定性发现：`hit_rate` 口径 bug
- vLLM `find_longest_cache_hit`（`single_type_kv_cache_manager.py`）逐块查前缀，**在第一个 miss 处 `break`**。
- `h1/sitecustomize.py` 的 hook 因此**每个请求只记 1 个 miss**（分叉那一块），分叉之后的唯一 tail block 根本不被 lookup、不计 miss。
- 于是 `hit_rate = matched_blocks/(matched_blocks+请求数)` = **「前缀匹配长度效率」**，被共享 Global 段顶高到 ~0.93，**对 tail 驱逐完全失明**。这正是上一条 commit「hit结果不达标，正在代码审查」要查的根因。
- **正确口径 = token 级覆盖率** `matched_tokens / 总prompt_tokens`。用此口径离线重算三档：**ws1=0.69、ws2=0.62、ws3r=0.38**，随工作集放大单调下降，ws1/ws2 已在 [0.5,0.85] 窗内。

## 9.3 代码修复（改用 vLLM 原生 token 级口径）
hook `KVCacheManager.get_computed_blocks`，累加 vLLM 原生 `queries(=prompt tokens)` / `hits(=matched tokens)`，summary 的 `hit_rate` 改用 `native_hits/native_queries`；block 级旧值保留为诊断 `block_lookup_hit_rate`。涉及文件：
- `h1/sitecustomize.py`：新增 `native_queries/native_hits/native_requests` 计数 + `_install_edgekv_native_hit_rate_patch()`；`get_edgekv_gpu_cache_stats()` 优先输出原生口径，`hit_source=vllm_native_token_coverage`。
- `h1/run_h1_vllm0110_real.py`：`merge_gpu_cache_stats` 汇总原生计数并据此算 `hit_rate`；summary 增列 `native_hit_rate/block_lookup_hit_rate/gpu_prefix_cache_native_{queries,hits,requests}`，`hit_source` 不再硬编码。
- `h1/aggregate_h1_serving_bench.py`：同步原生口径。
- **gating**：原生记录仅在 `EDGEKV_H1_GPU_POLICY ∈ {h1_lru,h1_lfu,h1_lpe}` 时生效（与既有 hook 一致）。
- **验证**：ws2×0.85 h1_lru → `hit_rate=0.6303`、`hit_source=vllm_native_token_coverage`、`block_lookup_hit_rate=0.9372`（旧坏值保留）、`native_requests=768`、`qwr=4.1e-5`。

## 9.4 锁定 ws2 做策略对比（实验配置）
- **trace**：`data/edgekv_traces/source_ablation/hotqa_ws2.jsonl`（池 5/80/600，warm/cold ratio 0.60/0.20，chunk_words 320，768 请求，random_seed 2026，hotpotqa-max-examples 150 生成）。
- **策略**：`h1_lru`（≈vLLM 原生 LRU 基线）/ `h1_lfu` / `h1_lpe`，**每策略独立进程、`EDGEKV_H1_GPU_POLICY` 启动注入**。
- **budget**：0.85（主）/ 0.90（对照）；**reps=3**（确定性回放，std≈0）。
- **引擎/负载**：TP2、`models/Qwen2.5-7B-Instruct`、float16、max-model-len 2048、max-tokens 16、replay-batch-size 8、batch-order original、visible-devices 0,1；代价模型 c-re 0.12 / bw 1.0 / d-deser 3.0。
- **驱动脚本**：`scripts/run_ws2_policy_cmp.sh`（顺序跑 3×2×3=18 runs）；**聚合出图**：`scripts/aggregate_ws2_policy_cmp.py`。
- **产物**：`h1/out/hotqa三级trace_有效窗口/policy_cmp/`，含 `{policy}_rep{n}/{budget}_{policy}_summary.json`、`ws2_policy_cmp_summary.csv`、`ws2_hit_rate.png`、`ws2_latency_p95.png`。

## 9.5 结果

| policy | budget | **hit_rate(原生)** | evictions | queue_wait_p95_ratio | latency_p95(ms, mean±std) |
|--------|--------|------|------|------|------|
| h1_lru | 0.85 | **0.6303** | 5668 | 4.7e-5 | 901±53 |
| h1_lfu | 0.85 | 0.6297 | 5783 | 5.2e-5 | 802±73 |
| h1_lpe | 0.85 | 0.6298 | 5786 | 4.4e-5 | 919±103 |
| h1_lru | 0.90 | **0.6976** | 3367 | 5.2e-5 | 777±37 |
| h1_lfu | 0.90 | 0.6847 | 3699 | 4.9e-5 | 817±33 |
| h1_lpe | 0.90 | 0.6843 | 3696 | 4.6e-5 | 856±29 |

## 9.6 结论（诚实）
1. ✅ **方法学成立 / 进窗达标**：全部 6 点 hit∈[0.63,0.70]⊂[0.5,0.85]，`queue_wait_p95_ratio`≈5e-5«0.5（非饱和），evict>0。§2 的「进窗成功」判据满足。
2. ⚠️ **策略未拉开（驱逐不是瓶颈）**：0.85 档三策略 Δ≤0.001（几乎重合）；0.90 档 `h1_lru`(0.698) 反而**略高于** `h1_lfu/h1_lpe`(0.685)。该工作集 reuse 以近期性为主，LRU 已近最优，LFU/LPE 无命中收益、LPE 延迟更高。
3. 根因是 trace 的 reuse 结构（近期性主导 ⇒ LRU 难被超越），非 budget 可解。要让策略分离，需「缓存明显装不下 + 频率/分层偏斜复用」的 regime（待决策：再调 trace 求分离 / 扫复用比梯度 / 接受此诚实结论出图）。

## 9.7 ws2 × xLRU 五档 budget 快速测试配置（改用 max_model_len=1024）

### 背景
原计划在 ws2 上扫 `budget=0.75/0.80/0.85/0.90/0.95` 五档、`h1_lru` 单次执行。保持 `max_model_len=2048` 时，低两档无法初始化：
- `0.75`：vLLM 报 `Available KV cache memory: -0.53 GiB`，没有可用 KV cache block。
- `0.80`：可用 KV cache 约 `0.01 GiB`，但服务 1 条 `max_model_len=2048` 请求至少需要约 `0.05 GiB`，vLLM 估计最大可支持长度约 `368`。

因此后续五档 xLRU 快速测试统一改为 `max_model_len=1024`。临时验证结果显示低两档可启动并完成 768 请求：

| policy | budget | max_model_len | hit_rate(原生) | block_lookup_hit_rate | evictions | queue_wait_p95_ratio | latency_p95(ms) |
|--------|--------|---------------|----------------|-----------------------|-----------|----------------------|-----------------|
| h1_lru | 0.75 | 1024 | 0.5454 | 0.9278 | 7905 | 6.8e-5 | 519 |
| h1_lru | 0.80 | 1024 | 0.6339 | 0.9377 | 5234 | 8.6e-5 | 492 |

### 固化启动入口
后续五档快速测试改为使用：

```bash
python3 h1/run_test.py --force
```

`h1/run_test.py` 当前固定配置：
- trace：`data/edgekv_traces/source_ablation/hotqa_ws2.jsonl`
- policy：`h1_lru`（xLRU/LRU baseline）
- budget：`0.75 0.80 0.85 0.90 0.95`
- model：`models/Qwen2.5-7B-Instruct`
- dtype：`float16`
- TP：`2`
- visible devices：`0,1`
- `max_model_len=1024`
- `max_tokens=16`
- `replay_batch_size=8`
- `max_num_batched_tokens=8192`
- `batch_order=original`
- 输出目录：`h1/out/hotqa三级trace_有效窗口/run_test_ws2_lru_budget5_maxlen1024/ws2_lru_budget5_maxlen1024/`

代码变更：
- `h1/run_step3_budget_tiers.py`：新增 `max_model_len` 参数透传，避免只能使用模块常量 `2048`。
- `h1/run_test.py`：改为 ws2 × `h1_lru` × 五档 budget × `max_model_len=1024` 的快速测试入口。

### 当前快速结论
`max_model_len=1024` 后，`0.75` 和 `0.80` 两档已经进入有效窗口且非饱和：`hit_rate=0.545/0.634`，`queue_wait_p95_ratio<1e-4`，eviction 明显存在。五档完整趋势应使用 `h1/run_test.py` 统一重跑后，以同一输出目录的 `step3_summary.csv` 为准。
