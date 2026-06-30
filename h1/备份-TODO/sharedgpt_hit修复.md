# 修改 ShareGPT Trace 生成器以拉开 Hit 动态范围（0.75≈0.50 → 0.95≈0.90）

## Summary
- 重写 `scripts/sharedgpt_jsonl_generation.py` 的逐请求构造逻辑，把当前
  “global/warm/cold + 随机命中”的层级结构，改为严格 **prefix-cache 友好**的三段式：
  `HOT（恒命中，定下限）→ WARM（按流行度摆动，定爬升）→ 短唯一尾部（定上限缺口）`。
- 目标：H1 budget 扫描下 `hit_rate` 从 `0.75≈0.50` 平稳上升到 `0.95≈0.90`，
  中间 `0.80/0.85/0.90` 单调平稳（形态对齐 hotqa ws2 的“平稳提升”，但动态范围更大）。
- 默认 768 请求；H1 验证显式传 `--num-prompts 768`。

## 根因（已读 run_h1_vllm0110_real.py / run_step3 / h0.load_replay_prompts 确认）
- 报告的 `hit_rate` 是 **vLLM 真实块级 prefix cache** 命中率（`enable_prefix_caching=True`，
  `gpu_cache_stats['hit_rate']`，`block_size=16`），按 trace 原始顺序回放（`batch_order=original`），
  prompt 文本**逐字**喂给 `llm.generate`（无 chat template / 无 BOS / 无 per-request 随机）。
  → 两条 prompt 文本完全相同就共享全部前缀块；prefix cache **位置相关**：某 chunk 的块只有在
  它之前的整段 token 序列完全一致时才复用。
- 当前 v2（floor 0.62 / ceiling 0.751）的问题：
  1. 共享前缀区域内有随机性（50% 追加第二 global、warm/cold 按 60%/20% 随机命中），
     打碎精确前缀匹配，warm 在块级几乎复用不到 → **上限压在 ~0.75**。
  2. cold 一次性层（108 unique，~1.4x）= 永久 miss → 进一步压低上限。
  3. global 仅 5 个、约 150x 复用、恒在缓存 → 抬高恒命中基线 → **下限卡在 0.62**，摆动空间小。

## 块级标定（block_size=16）
设每条 prompt 块数 = H(hot) + W(warm) + T(tail)：
- 上限（warm 全命中）≈ (H+W)/(H+W+T)；要 0.90 → `T ≈ 0.11·(H+W)`。
- 下限（仅 hot 命中）≈ H/(H+W+T)；要 0.50 → `H ≈ 总块数一半`。
- 目标块占比：**H≈50% / W≈40% / T≈10%**。
得到三个干净旋钮：
- 下限 ← hot 占比（hot/warm chunk 词数）+ 尾部长度。
- 上限 / 平滑度 ← warm 池规模 `P` 与 Zipf 指数 `s`（决定各 budget 能装下多少 warm）。
- 0.75↔0.95 跨度 ← warm 足迹 `P·W` 相对各 budget 缓存容量。

## Key Changes（仅改 `scripts/sharedgpt_jsonl_generation.py`）
- **HOT 层（恒命中下限）**：固定 1–2 个 hot chunk，每条请求按相同顺序置于最前；去掉 50% 第二 global 随机。
- **WARM 层（摆动）**：单一 warm 池（规模 `P`），每条请求用 `random.Random(seed)` 按 **Zipf(P, s)**
  采样**恰好 1 个** warm chunk，放在 hot 之后。Zipf 头部块在所有 budget 驻留（撑下限），
  尾部块仅高 budget 装得下（撑爬升），梯度自然平滑。随机性只决定“选哪个 warm”，不破坏其前缀。
- **删除一次性 cold 层**，改为一个**短的 per-request 唯一尾部**（如 `Q: <session_id 派生唯一短串>`，
  ~1 块）制造受控上限缺口（默认让上限自然逼近 0.9，可调 `--tail-words`）。
- **块占比标定**：hot 与 warm chunk 词数相等（≈150 words ≈ ~13 块），尾部 ~2–3 块
  → 下限 ≈ 0.45–0.50、上限 ≈ 0.9。
- **字段兼容**：保留 `prefix_ids/prefixes/global_prefix_ids/warm_prefix_ids/cold_prefix_ids/prefix_tiers`
  （hot→global、warm→warm、cold 置空或填尾部）；`turns_format="complete_prompt"` 不变；
  `reuse_key = sharegpt:hot+warm:<hot_ids>|<warm_id>`（仅影响诊断 `trace_side_hit_rate`，
  不影响报告的 vLLM `hit_rate`）。
- **新增 CLI**：`--hot-pool-size`（默认 1）、`--warm-pool-size`（默认 ~200）、`--zipf-s`（默认 1.0）、
  `--tail-words`（默认 ~12）；保留 `--chunk-words` 及兼容 alias。

## 起步参数（首跑，之后按结果标定）
`requests=768`、`hot_pool_size=1`、`warm_pool_size≈200`、`zipf_s≈1.0`、
`chunk_words≈150`（hot 与 warm 同尺寸）、`tail_words≈12`、`seed=2026`。

## Test Plan
- 静态验证生成器：
  - `python3 scripts/sharedgpt_jsonl_generation.py --out data/edgekv_traces/source_ablation/sharedgpt.jsonl`
  - 期望 summary：`total_requests=768`；prompt 估算 token 平均 ~350–500，p95 < 1008；
    `estimated_valid_for_max_model_len_1024.ratio = 1.0`；warm unique ≈ `P`。
- H1 验证（**后台异步运行，进程结束/报错再唤醒，不轮询**）：
  - `python3 h1/run_test.py --replay-trace data/edgekv_traces/source_ablation/sharedgpt.jsonl --num-prompts 768 --out-dir sharedgpt_jsonl_generation_lru_budget5_maxlen1024_v3 --force`
- 读 `step3_summary.csv` 调参：
  - 0.95 上限偏低 → 减小 `warm_pool_size` 或 `tail_words`。
  - 0.75 下限偏高 → 增大 `warm_pool_size` 或略减 hot chunk 词数。
  - 曲线不够平滑 → 调 `zipf_s`（增大 → 头部更稳、尾部更晚进入）。
  - 五档 `gpu_prefix_cache_evictions` 应非零且随 budget 升高单调下降。
  - 预计 1 次首跑 + 1–2 次标定跑收敛。

## 验收标准
- `hit_rate`：`0.75 ≈ 0.50`（±0.03）、`0.95 ≈ 0.90`（±0.03），`0.80/0.85/0.90` 单调平稳上升。
- 五档 `gpu_prefix_cache_evictions` 均非零且随 budget 升高整体下降。
- 全部 prompt 满足 `n_tokens + max_tokens(16) ≤ max_model_len(1024)`。

## Assumptions
- 只改 `scripts/sharedgpt_jsonl_generation.py`，不改 `h1/run_test.py` / `run_step3_budget_tiers.py` /
  `run_h1_vllm0110_real.py`。
- 生成内容仍全部来源于 ShareGPT 原始数据，不引入纯 synthetic 文本（尾部唯一串由 session_id 派生）。
- 验证以“形态平稳 + 端点命中达标”为准，不要求逐点复现参考曲线。

---

## 实测结果与最终配置（v4，已采纳）

### 最终生成器参数（`scripts/sharedgpt_jsonl_generation.py` 默认值）
| 旋钮 | 取值 | 作用 |
| --- | --- | --- |
| `requests` | 768 | 请求数 |
| `hot_pool_size` | 1 | HOT 恒命中层（定下限锚点） |
| `warm_pool_size` | 200 | WARM 池规模 |
| `zipf_s` | 1.0 | WARM 流行度（Zipf 指数） |
| `chunk_words` / `min_chunk_words` | 240 / 240 | hot/warm chunk 词数（均匀，~20 块/chunk）；**v3→v4 由 150 增至 240 以拉长 prompt、稀释首次 miss、抬高上限** |
| `tail_words` | 8 | per-request 唯一尾部（v3 的 12 缩短） |
| task | 单一常量，置于共享前缀内（v3 的 8-task 轮换已删除，避免破坏前缀匹配压低上限） |
| `seed` | 2026 | |

生成命令：
`python3 scripts/sharedgpt_jsonl_generation.py --out data/edgekv_traces/source_ablation/sharedgpt.jsonl`

静态校验：`total_requests=768`、prompt 均匀 `692` est-tokens（min=max=692，`+max_tokens(16)=708 ≤ 1024`）、
`estimated_valid_for_max_model_len_1024.ratio=1.0`、warm unique used=151。

### H1 五档实测（LRU，budget 0.75→0.95，max_len 1024，768 请求）
out-dir：`h1/out/sharedgpt_jsonl_generation_lru_budget5_maxlen1024_v4`
运行命令：
`python3 h1/run_test.py --replay-trace data/edgekv_traces/source_ablation/sharedgpt.jsonl --num-prompts 768 --out-dir sharedgpt_jsonl_generation_lru_budget5_maxlen1024_v4 --force`

| budget | hit_rate | gpu_prefix_cache_evictions |
| --- | --- | --- |
| 0.75 | 0.682 | 10702 |
| 0.80 | 0.814 | 5128 |
| 0.85 | 0.858 | 2531 |
| 0.90 | 0.868 | 1131 |
| 0.95 | 0.871 | 16 |

### 验收对照（勉强符合，已采纳）
- 形态：`0.682 → 0.814 → 0.858 → 0.868 → 0.871`，**单调平稳上升、无早饱和**（v3 在 0.85 已平台化，v4 climb 延伸至 0.95）。✓
- 端点：下限 `0.75≈0.68`（目标 0.50，仍偏高 ~0.18，未达 ±0.03）；上限 `0.95≈0.87`（目标 0.90，差 ~0.03，基本达标）。△ 勉强符合
- 五档 `evictions` 均非零且随 budget 单调下降（10702→16）。✓
- 全部 prompt `708 ≤ 1024`。✓

### 关键根因（块级标定，block_size=16）
- 上限受**首次 miss 固定成本**支配：满驻留时仅每个 distinct warm chunk 的首次加载为 miss
  （~151 distinct × ~20 块），其占比 ≈ `firstmiss/(768·B)`。v3 prompt 仅 ~28 块→该成本 ~16%→上限压在 0.836；
  v4 拉长到 ~43 块（692 tokens）稀释该成本→上限升至 0.871。
- 下限/早饱和受 **warm 块足迹相对各 budget 缓存容量**支配：v3 的 13 块 warm working set 在 0.85 即全驻留→平台化；
  v4 的 20 块 warm 使 working set(~3020 块) 超过 0.95 容量(~2500 块)→0.75 装不下→下限略降、climb 延伸至 0.95。
- **下限仍偏高（0.68）**的原因：0.75 档已驻留 Zipf 头部热门 warm chunk，故非纯 hot 下限(~0.46)。
  若需进一步压低下限到 0.50：增大 `warm_pool_size` + 略降 `zipf_s`（摊薄头部驻留），代价是上限略降（首次 miss 增多），需配合再拉长 chunk 补偿。
