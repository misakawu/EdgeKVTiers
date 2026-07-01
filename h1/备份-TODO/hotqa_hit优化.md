# HotQA hit 曲线优化：0.95 档从 0.73 拉到 ~0.90（0.75 档保持 ~0.545）

## Context（为什么做）
参考结果 `h1/备份-运行结果/有效结果/hotqa_hit平稳提升/run_test_ws2_lru_budget5_maxlen1024/ws2_lru_budget5_maxlen1024/step3_summary.csv`：

| budget | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
| --- | --- | --- | --- | --- | --- |
| hit_rate | 0.545 | 0.634 | 0.698 | 0.733 | **0.734** |
| evictions | 7905 | 5234 | 2991 | 1329 | 229 |

0.90 与 0.95 几乎相同（0.733≈0.734）= **早饱和**，上限卡在 ~0.734。目标：**0.75 档保持 ~0.545（软约束 0.50–0.58），0.95 档升到 ~0.90**，中间单调平稳。

### 根因（与 `h1/备份-TODO/sharedgpt_hit修复.md` 同源）
`hit_rate` 是 vLLM 真实块级 prefix cache 的 token 级命中（`enable_prefix_caching=True`，`block_size=16`，按 trace 原序回放，prompt 逐字喂 `llm.generate`，位置相关）。当前 `scripts/hotqa_jsonl_generation.py` 用旧的 global/warm/cold + 随机命中结构，三处压低上限/造成早饱和：
1. `if rng.random() < 0.5: selected_global.append(...)`（`hotqa_jsonl_generation.py:176-177`）—— 随机第二个 global 把 warm 块的前缀位置打乱，半数请求 warm 复用不到。
2. **cold 层**（20% 请求 × 600 池，`:183-185`）= 永久 miss，直接把上限钉在 ~0.73。
3. warm 仅 60% 请求、`request_index % warm_pool_size` 轮询（非 Zipf），warm 足迹在 0.90 即全驻留 → 0.90≈0.95 早饱和。

`scripts/sharedgpt_jsonl_generation.py` 已用 prefix-clean 结构（`HOT 恒定 → WARM 单块 Zipf → 短唯一 TAIL`）证明可把上限拉到 ~0.87、climb 延伸至 0.95。本方案把该结构移植到 hotqa。

## 方案：仅改 `scripts/hotqa_jsonl_generation.py`
把逐请求构造改为与 sharedgpt 同构的三段式（源数据仍是 HotpotQA chunk）：

- **HOT（定下限锚点）**：每条请求取整个 `hot_pool`（`hot_pool_size` 默认 1），相同顺序置于最前。**删除 50% 随机第二 global**。
- **WARM（定爬升）**：用 `random.Random(seed)` 按 `Zipf(warm_pool_size, zipf_s)` 采样**恰好 1 个** warm chunk，紧跟 HOT 之后。Zipf 头部块在所有 budget 驻留（撑下限），尾部块仅高 budget 装下（撑 climb），梯度平滑。每条请求都有 warm（取消 60% 覆盖语义）。
- **删除 cold 层**，改为**短的 per-request 唯一尾部**（复用 sharedgpt 的 `build_unique_tail`，由 `session_id`+`request_index` 派生 + 确定性 filler 补到 `tail_words`），制造受控上限缺口。
- prompt 模板沿用现有 `build_hotqa_prompt`，把 cold 段替换为 `Q: <tail>`，task/section 头保持常量在共享前缀内。

### 复用现有实现
- Zipf：移植 `sharedgpt_jsonl_generation.py:214-215` 的 `zipf_weights` + `:258` 的 `rng.choices(...)`。
- 唯一尾部：移植 `sharedgpt_jsonl_generation.py:181-193` 的 `build_unique_tail` + `TAIL_FILLER`。
- chunk 源：保持 `load_unique_hotqa_chunks`（`hotqa_jsonl_generation.py:76-100`）；`max_examples=150` 已能产 ~1500 unique chunk，足够 hot+warm 池。
- 输出字段：保留 `prefix_tiers/global_prefix_ids/warm_prefix_ids/cold_prefix_ids`（hot→global、warm→warm、cold 置空）、`turns_format="complete_prompt"`、`reuse_key`，与 `run_h1_vllm0110_real.py` 兼容。

### CLI（`parse_args`）
- 新增：`--zipf-s`（默认 0.8）、`--tail-words`（默认 8）。
- `--global-pool-size` 复用为 hot 池大小（默认 1）；`--warm-pool-size`（默认 300）、`--chunk-words`（默认 320）、`--requests`、`--random-seed`、`--hotpotqa-max-examples` 保留。
- `--cold-pool-size` / `--warm-request-ratio` / `--cold-request-ratio` 改为**接受但忽略**（deprecated），避免旧命令报错。
- summary 增加 `zipf_s` / `tail_words` / `warm_unique_used` 字段，便于标定。

### 起步参数（首跑）
`requests=768`、`hot_pool_size=1`、`warm_pool_size=300`、`zipf_s=0.8`、`chunk_words=320`、`tail_words=8`、`seed=2026`。
块级估算（chunk≈430 tok≈27 块）：单请求 ≈ HOT27 + WARM27 + TAIL1 ≈ 55 块 → 纯 hot 下限 ≈ 27/55≈0.49；上限 ≈ 1 − tail(~0.02) − warm 首次 miss 摊销 ≈ 0.88–0.90。`warm_pool=300` + `zipf_s=0.8` 把 warm 足迹拉大，使 0.75 仅头部驻留（压住下限不抬到 sharedgpt 的 0.68），working set 跨过 0.90 容量、climb 延伸到 0.95。

## 执行步骤
1. 改写 `scripts/hotqa_jsonl_generation.py`（唯一代码改动）。
2. 生成 v2 trace：
   `python3 scripts/hotqa_jsonl_generation.py --out data/edgekv_traces/source_ablation/hotqa_ws2_v2.jsonl --requests 768 --warm-pool-size 300 --zipf-s 0.8 --chunk-words 320 --tail-words 8`
   静态校验 summary：`total_requests=768`、`max_prompt_est_tokens+16 ≤ 1024`、`warm_unique_used≈150`。
3. **后台异步**跑 H1 五档（进程结束/报错再唤醒，不轮询；按 [[async-not-polling]] 与 [[real-harness-policy-env]]，本次仅 LRU 无需注策略 env）：
   `python3 h1/run_test.py --replay-trace data/edgekv_traces/source_ablation/hotqa_ws2_v2.jsonl --tier ws2_lru_budget5_maxlen1024_v2 --num-prompts 768 --out-dir run_test_ws2_lru_budget5_maxlen1024_v2 --force`
4. 读 `h1/out/run_test_ws2_lru_budget5_maxlen1024_v2/ws2_lru_budget5_maxlen1024_v2/step3_summary.csv` 标定（注意按 [[hit-rate-metric-bug]] 看 `hit_rate` 列即真实口径）：
   - 0.95 上限偏低 → 减小 `warm_pool_size` 或 `tail_words`，或加大 `chunk_words`（稀释首次 miss）。
   - 0.75 下限偏高（>0.58）→ 增大 `warm_pool_size` 或减小 `zipf_s`（摊薄头部驻留）。
   - 0.75 偏低（<0.50）→ 反向；曲线不平滑 → 调 `zipf_s`。
   - 预计 1 首跑 + 1–2 标定跑收敛。

## 验收标准
- `hit_rate`：`0.75 ≈ 0.50–0.58`、`0.95 ≈ 0.90`（±0.03），`0.80/0.85/0.90` 单调平稳上升、无早饱和（0.90 与 0.95 明显拉开）。
- 五档 `gpu_prefix_cache_evictions` 均非零且随 budget 单调下降。
- 全部 prompt `n_tokens + 16 ≤ 1024`。

## Assumptions
- 只改 `scripts/hotqa_jsonl_generation.py`，不动 `h1/run_test.py` / `run_step3_budget_tiers.py` / `run_h1_vllm0110_real.py` / `sitecustomize.py`。
- 内容仍全部来自 HotpotQA 原始 chunk，尾部唯一串由 session_id 派生（无纯 synthetic 文档）。
- 旧 `hotqa_ws2.jsonl` 与参考结果保持不动；新结果写入 v2 out-dir。

---

## 执行进度（2026/06/30）

### 已完成代码改动（`scripts/hotqa_jsonl_generation.py`）
1. 重写为 sharedgpt 同构三段式：HOT(整 hot_pool,恒定)→WARM(1 个 Zipf chunk)→唯一 TAIL。删除 50% 随机第二 global、cold 层、60% warm 覆盖。
2. 新增 CLI：`--zipf-s` / `--tail-words` / `--min-chunk-words`；`--global-pool-size` 复用为 hot 池；`--cold-pool-size`/`--warm-request-ratio`/`--cold-request-ratio` 改为接受即忽略；summary 增 `zipf_s`/`tail_words`/`warm_unique_used`。
3. **关键修正**：`load_unique_hotqa_chunks` 改为把 HotpotQA 段落文本拼接后按 `chunk_words`/`min_chunk_words` 重切成**均匀大 chunk**（仿 sharedgpt），解决小 chunk 导致的 ceiling 墙。内容仍全部 HotpotQA。

口径：只看 `native_hit_rate`(=`hit_rate`)，`block_lookup_hit_rate`(~0.93) 是已知假象列（[[hit-rate-metric-bug]]）。

### 三次跑结果（LRU 五档）
| 版本 | 参数 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 | 评估 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2 | req768 pool300 zipf0.8 chunk320(小段落) | 0.537 | 0.689 | 0.748 | 0.760 | **0.761** | floor✓；ceiling 低、早饱和(0.90≈0.95)、0.95 evict=0 |
| v3 | req1536 pool400 zipf0.8 chunk320(小段落) | 0.521 | 0.666 | 0.737 | 0.770 | **0.789** | 早饱和已修、五档 evict 全非零且单调、单调✓；ceiling 仍 0.789 |
| v4 | req1536 pool280 zipf0.8 **均匀240词大chunk** | 运行中(已被手动关闭) | | | | | 目标 ceiling≈0.90 |

evict 单调：v2 7162→0；v3 15370→9669→6342→4249→2659（全非零✓）。

### 根因与结构墙（详见 memory [[hotqa-ceiling-warm-first-touch-amortization]]）
- ceiling 被 **warm 首touch miss** 钉住：小段落 chunk(~150tok≈9blk)，226 unique warm × 9blk ≈ 73% 永久 miss。
- floor 被 HOT=1 小 chunk 钉住（HOT 越大 floor 越高，HOT=2 会破 0.58 上界），故 HOT=1 时理论 ceiling ≈ (HOT+WARM)/(HOT+WARM+tail) ≈ 0.90，仅当 warm 首touch 完全摊销才到。
- 杠杆：①加 requests 摊销 warm 首touch（v3 验证 ceiling↑）；②加 warm_pool 防早饱和（v3 验证 0.90/0.95 拉开）；③**加 chunk 词数无效**（HotpotQA 段落本就 <320 词）——故 v4 改为拼接段落成均匀 240 词大 chunk，降低 tail/首touch 占比把 ceiling 顶到 ~0.90。

### v4 参数与产物
- 生成命令：`python3 scripts/hotqa_jsonl_generation.py --out data/edgekv_traces/source_ablation/hotqa_ws2_v4.jsonl --requests 1536 --warm-pool-size 280 --zipf-s 0.8 --chunk-words 240 --min-chunk-words 240 --tail-words 8 --hotpotqa-max-examples 200`
- 静态校验：均匀 689 est tokens/prompt（real≈868 ≤1008 安全）、warm_unique_used=248/280（复用~6×）、281 chunk 恰好够 hot1+warm280。
- 跑命令（被手动关闭）：`python3 h1/run_test.py --replay-trace data/edgekv_traces/source_ablation/hotqa_ws2_v4.jsonl --tier ws2_lru_budget5_maxlen1024_v4 --num-prompts 1536 --out-dir run_test_ws2_lru_budget5_maxlen1024_v4 --force`

### 下一步（恢复时）
1. 重跑 v4（trace `hotqa_ws2_v4.jsonl` 已生成、代码已就绪，直接跑上面的 run_test 命令即可）。
2. 读 `h1/out/run_test_ws2_lru_budget5_maxlen1024_v4/.../step3_summary.csv` 标定：
   - ceiling 偏低/0.95 evict 仍大 → 工作集太大没在 0.95 装下，**减小 warm_pool**(280→~240)。
   - 0.90≈0.95 早饱和 → 工作集太小提前装满，**加 warm_pool**。
   - floor<0.50 → 减 requests 或抬 zipf_s；floor>0.58 → 反向。
3. 预计 1–2 次微调（仅 warm_pool）即可收敛到验收标准。
