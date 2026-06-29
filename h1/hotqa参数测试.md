# HotQA 参数测试方案 v2：把命中率打进「有效窗口」（DDL 2026-07-01 出图）

## 0. 一句话目标
让**缓存替换策略的好坏成为真正的瓶颈**——即把 GPU 前缀缓存命中率从当前的「死区」拉进
**有效窗口（命中率 0.5~0.85 且 `queue_wait_p95_ratio < 0.5`）**，这样不同策略才会拉开命中率/延迟梯度。

## 1. 第一步（Plan A：复用当前 trace + 扫 batch/请求数）已失败 —— 结论与证据
上一步 `hotqa_param_np1024_bs8`（`hotqa_x4.jsonl` 1024 行、`replay_batch_size=8`、budgets 0.75/0.80/0.85/0.90）实测：

| budget | gpu_mem_util | 引擎 | hit_rate | evictions | queue_wait_p95_ratio | ttft_p95(ms) | prefill_p95(ms) |
|--------|-------------|------|----------|-----------|----------------------|--------------|------------------|
| 0.75   | 0.75        | **OOM 启动失败** | — | — | — | — | — |
| 0.80   | 0.80        | **OOM 启动失败** | — | — | — | — | — |
| 0.85   | 0.85        | ok   | **0.9451** | 3673 | 3.9e-5 | 1067 | 1067 |
| 0.90   | 0.90        | ok   | **0.9874** | 330  | 4.6e-5 | 919  | 919  |

**三条硬结论：**
1. **budget 旋钮已到底**：本机（2080Ti/11GB × TP2）0.80 及以下引擎直接 OOM 启动失败，与记忆中的「11GB 卡引擎启动下限 ≈0.72」一致，实测在 TP2+7B+float16 下连 0.80 都起不来。**Plan B 里「往下扫 budget 到 0.30~0.60」在本机物理不可行。**
2. **没有饱和问题**：`queue_wait_p95_ratio ≈ 4e-5`（远小于 0.5），p95 几乎全是 prefill，不是排队。所以**不需要降并发去解饱和**——饱和不是当前矛盾。
3. **死区才是矛盾**：能启动的两档命中率都 ≥0.945，工作集太小（唯一 chunk 池 5+20+100、`chunk_words` 短），缓存几乎全装得下，驱逐谁都不影响命中。

→ 既然 budget 旋钮压不下去，**唯一可行路径是另一个旋钮：把工作集做大 / 把复用率降下来**，在能启动的 budget（0.85 主，0.90 作对照）上把命中率拉进 0.5~0.85。

## 2. 操作定义（判据，重跑时逐点核对）
- **死区**：`hit_rate ≥ ~0.95`。缓存几乎装得下整个工作集，驱逐谁都不影响命中 → 任何策略都一样。**当前两个可启动档都在死区。**
- **饱和区**：到达速度 > 系统吞吐，请求堆在调度队列，p95 主要是排队。判据：`queue_wait_p95_ratio > 0.5` → 缓存省的 prefill 被排队淹没。**当前不在饱和区（≈4e-5）。**
- **有效窗口（要打到这里）**：`hit_rate ∈ [0.5, 0.85]` 且 `queue_wait_p95_ratio < 0.5`。这时命中率由驱逐决策决定，策略好坏才看得出来。

## 3. 两个旋钮（本机适配版）
| 旋钮 | 现状 | 本方案设定 | 目的 |
|------|------|-----------|------|
| 缓存显存预算 budget | 0.85 已是可启动下限，0.80 OOM | **不再往下扫**（物理到底）。固定 `budget = 0.85`（主，制造最大可行驱逐压力）+ `0.90`（对照） | 取本机能给的最大驱逐压力 |
| 工作集 / 复用率（**主旋钮**） | 池 5+20+100、`chunk_words=120`、复用率高 | **重生成 trace**：扩大 cold/warm 池、加长 `chunk_words`、降低 warm/cold 复用比 → 唯一工作集 KV 远超 0.85 budget | 把命中率从 ≥0.95 拉进 0.5~0.85 |
| 负载/到达率 | 未饱和（无需动） | 固定 `replay_batch_size=8`、`batch_order=original`；每点跑完核对 `queue_wait_p95_ratio < 0.5`，若越界再降并发 | 保持 TTFT 反映 prefill 而非排队 |
| 控制变量 | — | 同一条重生成 trace、同模型 Qwen2.5-7B、同后端、同到达序列；每点 `reps ≥ 3` | 跨策略 / 跨档可比 |

## 4. 主旋钮怎么拧：重生成 trace（`scripts/hotqa_jsonl_generation.py`）
该脚本已暴露所需全部旋钮（无需改代码）：
`--global-pool-size`(5) `--warm-pool-size`(20) `--cold-pool-size`(100)
`--warm-request-ratio`(0.60) `--cold-request-ratio`(0.20) `--chunk-words`(120) `--requests`(256)

**扫描思路：沿「工作集大小」单调放大，找让 hit 落进 0.5~0.85 的那一档。** 同时放大池子 + 加长 chunk（两者都增加唯一 KV 字节数）：

| 档位 | global/warm/cold 池 | chunk_words | requests | 产物文件 |
|------|---------------------|-------------|----------|----------|
| ws1 | 5 / 40 / 300   | 200 | 512  | `data/edgekv_traces/source_ablation/hotqa_ws1.jsonl` |
| ws2 | 5 / 80 / 600   | 320 | 768  | `data/edgekv_traces/source_ablation/hotqa_ws2.jsonl` |
| ws3 | 5 / 160 / 1200 | 480 | 1024 | `data/edgekv_traces/source_ablation/hotqa_ws3.jsonl` |

> 若降复用更有效，可在某档基础上把 `--warm-request-ratio`/`--cold-request-ratio` 调低（更多请求落到唯一 cold chunk）。**先放大池子，hit 仍 >0.85 再降复用比；二选一进窗即可。**
> 受限于 HotpotQA 唯一 chunk 总量：若 `pool_size+query_pool` 超过可用唯一 chunk 数，脚本会报 `only built N unique chunks`，需配合 `--hotpotqa-max-examples` 扩大原始语料或下调池子。

生成命令（每档一条，按需调参）：
```bash
conda run --no-capture-output -n edgekv-vllm0110 python scripts/hotqa_jsonl_generation.py \
  --out data/edgekv_traces/source_ablation/hotqa_ws1.jsonl \
  --global-pool-size 5 --warm-pool-size 40 --cold-pool-size 300 \
  --warm-request-ratio 0.60 --cold-request-ratio 0.20 \
  --chunk-words 200 --requests 512 --hotpotqa-path data/hotpotqa
```

## 5. 实验矩阵
- 工作集档：`ws1 → ws2 → ws3`（按需停在第一个进窗的档）。
- budget：`0.85`（主）、`0.90`（对照）。**不扫 0.30~0.60（OOM）。**
- 每点 `reps ≥ 3`（取均值 ± 标准差）。
- 先用 `ws1 × 0.85` 探路：进窗则在该 ws 上跑满 budget×reps；未进窗（hit 仍 >0.85）则升到 ws2，再 ws3。

## 6. 运行命令（每档一条，后台异步执行，跑完唤醒再跑下一条；不轮询）
```bash
PYTHONPATH=.:h1:h0 EDGEKV_H1_GPU_POLICY=h1_lru \
conda run --no-capture-output -n edgekv-vllm0110 \
  python h1/run_h1_vllm0110_real.py \
  --out "h1/out/hotqa三级trace_有效窗口/ws1_bud0.85_rep1" \
  --model models/Qwen2.5-7B-Instruct --dtype float16 \
  --replay-trace data/edgekv_traces/source_ablation/hotqa_ws1.jsonl \
  --workload rag --hotpotqa-path data/hotpotqa --hotpotqa-max-examples 5 \
  --rag-requests 512 --max-requests 512 --max-sessions 512 \
  --sharegpt-order longest --timeout-s 120 \
  --policies h1_lru --budgets 0.85 \
  --tensor-parallel-size 2 --max-model-len 2048 --max-tokens 16 \
  --replay-batch-size 8 --batch-order original --warmup-batches 0 \
  --c-re-ms-per-token 0.12 --bw-gbps 1.0 --d-deser-ms 3.0 \
  --visible-devices 0,1
```
- `--max-requests/--max-sessions/--rag-requests` 跟随该档 trace 行数。
- 其余点仅改 `--out`、`--replay-trace`、`--budgets`，并按 reps 改 `_rep{n}` 后缀。

### 输出目录命名
- 顶层文件夹：`h1/out/hotqa三级trace_有效窗口/`
- 子实验：`ws{1,2,3}_bud{0.85,0.90}_rep{1..3}`，例：
  `h1/out/hotqa三级trace_有效窗口/ws2_bud0.85_rep3`

## 7. CSV 要记的列（在现有 summary.json schema 上补齐，每点取 reps 均值±std）
均从各 `*_summary.json` 直接读：
- 标识：`workload_set`(ws1/2/3)、`budget`、`gpu_memory_utilization`、`rep`、`ok`、`error`
- 工作集描述：global/warm/cold 池大小、`chunk_words`、warm/cold 复用比、唯一 chunk 数、`requests`
- 命中（核心）：`hit_rate`、`gpu_prefix_cache_lookup_total/_hits/_misses`、`gpu_prefix_cache_cached_blocks`
- 驱逐压力：`gpu_prefix_cache_evictions`、`eviction_count`、`admission_count`、`admission_rejection_count`、`high_reuse_eviction_count`、`low_score_evictions`
- 饱和判据（D4）：`queue_wait_p95_ms`、`ttft_proxy_p95_ms`、**`queue_wait_p95_ratio`（<0.5 才有效）**、`prefill_p95_ms`、`prefill_p95_ratio`
- 延迟/资源：`latency_p95_ms`、`latency_mean_ms`、`gpu_memory_peak_mib`、`elapsed_s`

## 8. 成功 / 停止判据
- **成功（进窗）**：某 `ws × budget` 档满足 `hit_rate ∈ [0.5, 0.85]` 且 `queue_wait_p95_ratio < 0.5` 且 `evictions > 0`。
  → 在该档锁定控制变量，跑满 reps≥3，进入策略对比（h1_lru vs 其它策略）出图。
- **升档**：若 hit 仍 >0.85，按 ws1→ws2→ws3 放大工作集（或降复用比）继续。
- **停止/换路**：若 ws3（最大可生成档）仍 hit >0.85，说明 HotpotQA 唯一 chunk 总量不足以撑出窗口，
  需换更大语料（提高 `--hotpotqa-max-examples` 或换数据源），或接受在「次优窗口」附近出图并在报告中诚实标注。
- **越界保护**：任一点若 `queue_wait_p95_ratio > 0.5`（进了饱和区），下调 `replay_batch_size`（8→4→2）回到非饱和再比策略。

## 9. 执行约定
- 遵循全局规则：每条命令后台异步执行，进程结束或报错后唤醒，再跑下一条；不轮询。
- budget 不再下扫（本机 OOM 物理约束，见 §1）。
- DDL：**2026-07-01 出图**（今天 2026-06-29，剩 2 天，优先 ws1/ws2 × 0.85 探路）。
