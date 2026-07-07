# H1 完整实验：2 数据集 × 3 预算 × 4 策略（24 cell）

## 背景

已完成 sharedgpt_v5 hit 跳跃修复与 hotqa_ws2 标定，两条 trace 均已进入可用窗口。现在跑一次正式的 H1 策略对比完整实验，比较四种 GPU 驱逐策略（LPE / LRU / LFU / vLLM 默认）在三档预算下、两个工作负载上的表现（hit_rate、TTFT、evictions 等）。

矩阵：`2 数据集 × 3 预算 × 4 策略 = 24 个 cell`，每个 cell 启动一次 vLLM 引擎。

- 数据集：
  - `data/edgekv_traces/有效实验数据/sharedgpt_v5.jsonl`（1536 请求，prompt≤692 tok）
  - `data/edgekv_traces/有效实验数据/hotqa_ws2.jsonl`（768 请求，prompt≤795 tok）
- 预算（gpu_memory_utilization）：`0.75 / 0.825 / 0.9`（均在 11GB 卡启动下限 ≈0.72 之上）
- 策略：`h1_lpe / h1_lru / h1_lfu / vllm_default`
- 启动器：`h1/run_test.py`
- 结果三层文件夹：`实验数据 / 预算 / 策略`

## 关键验证（已确认）

- 两条 trace 的 `max(prompt_est_tokens)+16` 分别为 708 / 811，均 ≤ `max_model_len=1024`，无需改 max-len。
- `run_test.py` 已委托 `run_step3_budget_tiers.run_step3`，后者在 `env_overrides` 中按策略注入 `EDGEKV_H1_GPU_POLICY`（`run_step3_budget_tiers.py:108`），策略生效路径正确，不会 no-op。
- `run_step3` 产出目录天然是 `base_out/<tier>/<budget>/<policy>`；令 `tier=数据集名` 即得到「数据集/预算/策略」三层结构。
- `run_test.py` 已有 `--tier / --replay-trace / --num-prompts / --out-dir / --force` CLI 参数；**仅** `BUDGETS` 与 `POLICIES` 是硬编码，需要改动。
- 注意：`--out-dir` 传值不能带前导 `/`（`Path("h1/out")/"/x"` 会被 pathlib 解析成绝对路径 `/x`）。用相对名如 `full_experiment`。

## 改动（唯一改文件：`h1/run_test.py`）

1. 将硬编码常量改为本实验值：
   - `BUDGETS = ["0.75", "0.825", "0.9"]`
   - `POLICIES = ["h1_lpe", "h1_lru", "h1_lfu", "vllm_default"]`
2. 新增可选 CLI 覆盖参数（对齐 `run_step3_budget_tiers.py` 的空格分隔风格），保持脚本可复用、非破坏：
   - `--budgets`，默认 `" ".join(BUDGETS)`，运行时 `args.budgets.split()`
   - `--policies`，默认 `" ".join(POLICIES)`，运行时 `args.policies.split()`
   - 在 `step3.run_step3(...)` 调用里把 `budgets=`/`policies=` 换成解析后的列表。
3. 顶部 docstring 简单更新为「预算×策略矩阵、可 --replay-trace/--tier 重定向」的说明（一行即可）。

其余参数不变：`REPLAY_BATCH_SIZE=8`、`MAX_MODEL_LEN=1024`、`MAX_NUM_BATCHED_TOKENS=8192`、`keep_cells=True`。

## 运行（异步后台，两数据集顺序执行——共用 GPU 0,1 不能并行）

按仓库约定长任务后台异步、结束后再检查，不轮询。两次调用串到一条后台命令里（`&&`），共用一个输出根 `h1/out/full_experiment`：

```bash
cd /DATACENTER3/zhenxiang.wang/work/EdgeKVTiers
python3 h1/run_test.py \
  --replay-trace data/edgekv_traces/有效实验数据/sharedgpt_v5.jsonl \
  --tier sharedgpt_v5 --num-prompts 1536 \
  --out-dir full_experiment --force \
&& python3 h1/run_test.py \
  --replay-trace data/edgekv_traces/有效实验数据/hotqa_ws2.jsonl \
  --tier hotqa_ws2 --num-prompts 768 \
  --out-dir full_experiment --force
```

产出结构：

```text
h1/out/full_experiment/
  sharedgpt_v5/
    {0.75,0.825,0.9}/{h1_lpe,h1_lru,h1_lfu,vllm_default}/...
    step3_summary.csv        # 该数据集 3×4 汇总
  hotqa_ws2/
    {0.75,0.825,0.9}/{h1_lpe,h1_lru,h1_lfu,vllm_default}/...
    step3_summary.csv
```

## 验收 / 检查

进程结束（或报错唤醒）后：

- 确认 24 个 cell 均成功（无 `RuntimeError: real replay cell failed`）。
- 读两个 `step3_summary.csv`：
  - 每个数据集 12 行（3 预算 × 4 策略）。
  - `hit_rate` 随 budget 单调上升；四策略在同一 budget 下可区分。
  - `hit_rate_delta_vs_lru` / `p95_ttft_gain_pct_vs_lru` 有非零值（策略确实生效，非 no-op）。
  - `gpu_prefix_cache_evictions` 随 budget 下降。
- 若某 cell 因 OOM 在 0.75 启动失败，回看是否触及 11GB 卡下限（0.75 应安全，>0.72）。

---

# 后续优化方案：batch_size=16 + 真实异步 QPS 到达注入

## 背景（为什么要做）

在 `h1/out/H1完整实验_batch_size_8` 里，`queue_wait` 延迟分段几乎可忽略（均值 ~0.016–0.021 ms，p95 ~0.033–0.047 ms；`queue_wait_p95/p95_ttft ≈ 4e-5`），prefill 占 99.996% 延迟。

**根因是结构性的，不是调参问题**（已在代码确认）：`h1/run_h1_vllm0110_real.py` 是**同步**回放——每个 batch 调 `llm.generate(prompts, sampling)` 阻塞到整批跑完（`:719`），且引擎按 `max_num_seqs = args.replay_batch_size` 构建（`:558`）。一个 batch 的请求在提交瞬间全部被准入，调度器从不持有等待队列，batch 之间是硬屏障，所以 `queue_wait` 只能 ≈0。

**当前没有任何真正的"请求到达速度"机制**：流经 `run_step3_budget_tiers.run_step3()` 的 `request_rate` **只**传给 `summarize_step3_budget_tiers.py --request-rate`（`:138`）作为事后"saturated"标记，从不进入 vLLM，也不改变请求何时注入。

目标（两部分）：
1. **Part A** — 把 `batch_size` 8→16，重跑 24-cell 矩阵。
2. **Part B** — 加入**真实异步 QPS 到达注入器**（AsyncLLMEngine + 按速率 `add_request`，Poisson/constant 到达间隔），让请求随时间到达、在并发上限后排队，使 `queue_wait` 成为有意义的分段，从而把饱和度 `queue_wait_p95/p95_ttft` 驱进实验目标窗口 ~[0.5, 0.85]（见记忆 `h1-budget-floor-2080ti-11gb`：窗口是负载/旋钮2 属性，不是 budget 属性）。

> 提醒：Part A 单独**不会**明显抬高 `queue_wait`——因为 `max_num_seqs` 跟随 `replay_batch_size`（16 个仍一次性全准入），且 `max_tokens=16` 让 decode 极短。Part B（异步到达、并发与到达解耦）才是真正制造排队的杠杆。Part A 仍作为吞吐/KV 压力点与 Part B 的干净基线。

## 可行性（已核实）

- 计时补丁挂在 `vllm.v1.engine.output_processor.RequestState._new_request_output`（`h1/sitecustomize.py:82-117`）——v1 同步与异步引擎共用，异步路径下 `output.metrics`（arrival/queued/scheduled/first_token）照常填充。
- edgekv 策略 + 命中率钩子挂在核心 `BlockPool`/`KVCacheManager`/`SingleTypeKVCacheManager`（`sitecustomize.py:1222-1572`）——与引擎无关，异步下 hit_rate/eviction 统计照常。
- `EDGEKV_H1_GPU_POLICY` 在 `build_llm` 构建引擎前设置（`run_h1_vllm0110_real.py:551`）；异步构建器必须同样先设置。

## Part A — batch_size 8 → 16（快、低风险）

改 `h1/run_test.py`：
- `REPLAY_BATCH_SIZE = 16`（`:24`）。
- `MAX_NUM_BATCHED_TOKENS` 保持 `8192`（**不**用 `MAX_MODEL_LEN * REPLAY_BATCH_SIZE = 16384` 的乘积），以避开 11GB/TP=2 引擎启动下限（~0.72，budget 0.75 逼近）。`resolved_max_num_batched_tokens`（`run_h1_vllm0110_real.py:530-547`）接受 `[replay_batch_size, max_model_len*replay_batch_size]=[16,16384]` 内任意值，8192 合法且 chunked-prefill 可处理 16 seq。若 0.75 能干净启动，可再试 16384 测更高 prefill 并行点。

运行（两次调用，一数据集一次，与 `batch_size_8` 产生方式一致；后台异步、不轮询）：
- ShareGPT：`python h1/run_test.py --out-dir H1完整实验_batch_size_16`
- HotQA：`python h1/run_test.py --out-dir H1完整实验_batch_size_16 --tier hotqa_ws2 --replay-trace data/edgekv_traces/有效实验数据/hotqa_ws2.jsonl --num-prompts 768`

## Part B — 异步 QPS 到达注入器（主要工作）

改动集中在 `h1/run_h1_vllm0110_real.py`，经 `run_step3_budget_tiers.py`、`run_test.py` 透传。默认路径保持 **sync**，使既有 24-cell 复现逐字节一致。

### 新增 CLI（`parse_args`，~`:1095-1145`）
- `--arrival-mode {sync,async}` 默认 `sync`。
- `--request-rate FLOAT` 默认 `0.0`（req/s；`0`/`inf` = 一次性全提交）。
- `--arrival-dist {poisson,constant}` 默认 `poisson`。
- `--max-num-seqs INT` 默认 `0`→回退到 `replay_batch_size`。异步模式下这是**并发上限**，与到达解耦，是到达速度超过服务速度时制造等待队列的关键。
- `--arrival-seed INT` 默认 `0`，用于可复现的 Poisson（`random.seed`）。

### 抽取复用（避免复制行/plumbing 逻辑）
把当前同步循环（`:676-759`）里的每请求逻辑抽成两个 helper，供两种模式共用：
- `build_sampling(trace_fields, args)` → `SamplingParams(..., extra_args={'edgekv_h1': {...}})`（即 `:692-717` 的字典，须**原样保留**以保证 edgekv 策略输入一致）。
- `build_request_row(item, output, reuse_info, trace_fields, latency_ms, args, ...)` → `:736-759` 的行字典（含 `queue_wait_ms`/`prefill_ms`/`ttft_proxy_ms`，经既有 `request_output_timing_ms`）。

### 异步构建器
`build_async_llm(args, policy, gpu_memory_utilization)` 镜像 `build_llm`（`:550-565`）：先设 `EDGEKV_H1_GPU_POLICY`，再用**相同** kwargs 构建 `AsyncLLMEngine.from_engine_args(AsyncEngineArgs(...))`（model、dtype、gpu_memory_utilization、max_model_len、`enforce_eager=True`、`max_num_seqs = args.max_num_seqs or args.replay_batch_size`、`max_num_batched_tokens = resolved_max_num_batched_tokens(args)`、tensor_parallel_size、`enable_prefix_caching=True`、`swap_space=0`、`cpu_offload_gb=0`、`disable_log_stats=False`）。
- 实现第 0 步：运行时探测确切异步类/构造方法（`vllm.AsyncLLMEngine` vs `vllm.v1.engine.async_llm.AsyncLLM`；`from_engine_args` vs `from_vllm_config`）——规划时 Bash 分类器临时故障没跑成。`.generate(prompt, sampling, request_id)` 是异步生成器，须消费到 `output.finished` 才拿到完整 metrics。

### 异步驱动 `run_async_replay(...)`
`asyncio.run` 一个注入协程：
- 沿用 `order_trace_for_batches` 排序，按序遍历请求。
- 第 `i` 个请求：算到达间隔——constant `1/rate`，或 Poisson `random.expovariate(rate)`（rate `0`/`inf` → 不 sleep，突发提交）；`await asyncio.sleep(delay)`；起一个 consumer task 把 `engine.generate(prompt, sampling, request_id=f'req-{i}')` 消费到最终输出。
- `await asyncio.gather(...)` 所有 consumer；用 `build_request_row` 组行（`latency_ms` 用每请求端到端墙钟；`ttft_proxy_ms` 不变）。
- 复用**不变**的下游汇总/CSV/JSON/gpu-stats 代码（只消费 rows 列表）。
- 异步路径同样保留构建前的 `patch_edgekv_vllm_request_metrics()` 与 `reset_edgekv_gpu_cache_stats()`（当前 `:648-658`）。

### 透传
- `run_step3_budget_tiers.py`：加模块常量 + `run_step3(...)` 参数（`arrival_mode`/`request_rate`/`arrival_dist`/`max_num_seqs`），并在 `cell_args()`（`:50-76`）追加对应 flag。把**同一** `request_rate` 传给汇总脚本（`:138`），使其"saturated"列基于真实注入速率计算。
- `run_test.py`：加 `ARRIVAL_MODE`/`REQUEST_RATE`/`ARRIVAL_DIST`/`MAX_NUM_SEQS` 常量 + CLI，转发进 `step3.run_step3(...)`。

## 改动文件
- `h1/run_test.py` — Part A 常量；Part B 旋钮常量 + 透传。
- `h1/run_h1_vllm0110_real.py` — 新增 args、`build_async_llm`、`run_async_replay`、抽 `build_sampling`/`build_request_row`、按 `--arrival-mode` 分派。
- `h1/run_step3_budget_tiers.py` — 新增 `run_step3` 参数 + `cell_args` flag。
- （`sitecustomize.py` 与 `summarize_step3_budget_tiers.py` 无需改。）

## 验收 / 检查（后台异步、不轮询）
1. **运行时 API 探测**（第 0 步）：Bash 分类器恢复后确认异步引擎类与 `.generate` 签名。
2. **异步单 cell 冒烟**：跑一个 `policy×budget`（如 `h1_lru`@0.75），`--arrival-mode async`，小切片(~64 prompts)、适中 `--request-rate`、`--max-num-seqs 8`。断言输出 CSV/JSON：引擎在 11GB 下限启动；`queue_wait_ms` 现已明显 >0 且 `queue_wait_p95/p95_ttft` 非平凡；`native_hit_rate` 仍非零（edgekv 路径完好）。
3. **batch16 全量**：确认 `H1完整实验_batch_size_16` 产出全 24 cell，与 `batch_size_8` 对比 p95/吞吐。
4. **速率扫描**：扫 `--request-rate`（几个点）找到让 `queue_wait_p95/p95_ttft` 落进 ~[0.5,0.85] 的值，再以该速率跑完整异步 24-cell 到新 out-dir（如 `H1完整实验_async_rateN`）。
5. 确认默认 `--arrival-mode sync` 复现既有 `batch_size_8` 数值不变（向后兼容护栏）。
