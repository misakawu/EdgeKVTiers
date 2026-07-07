# 大幅重构：structured_conversation_v2 trace + 回放器真实统计（具体 plan）

## Context（为什么做）

当前 `run_test.py` 走的 replay trace（`data/edgekv_traces/source_ablation/sharedgpt_token_v1.jsonl`，`trace_format=sharegpt_token_delta_v1`）把「拼装累积 Prompt + apply_chat_template + 分词 + 前缀复用计数」全部**预计算并写死在 trace 里**（`prompt_token_ids` / `reused_prefix_token_count` / `prefix_hash`）。问题：

1. Trace 与模型强绑定（换 tokenizer/模板就得重生成），不可读、不可移植。
2. 命中/复用统计走的是 **trace-side 模拟**（`trace_side_reuse` + `reuse_seen`：`reused_prefix_token_count>0` 或 `reuse_key` 见过即命中）。这是"按 trace 猜"的假命中，与真实缓存行为脱节（见 memory: hit-rate 口径 bug）。

目标（按 `h1/trace格式修复.md` + 用户决策）：

- **Trace 只存"发生了什么"**：原始 ShareGPT 多轮对话消息（role/content/turn_index），不含任何 token_ids/哈希/复用计数。
- **回放器负责"怎么算"**：动态累积历史 → `apply_chat_template` 渲染 Prompt **纯文本** → 喂 vLLM。
- **完全抛弃 trace-side 命中模拟**；所有 hit/复用统计**只来自回放器真实执行**（sitecustomize 已测的 `native_hits/native_queries` 真实缓存命中）。
- 本次**只做纯 ShareGPT 多轮**；`messages` 结构天然为 RAG 预留扩展位（下次注入检索上下文为特殊消息）。

关键前提（已核实，决定了本方案可行且低风险）：

- 引擎内策略 `_edgekv_refresh_profile_reuse`（`h1/sitecustomize.py:1101`）**自行**用 profile 真实 `hit_count/access_count` + `p_reuse_prior` + 类型先验在线估计 `p_reuse`，**不依赖**回放器传入的模拟 hit。回放器传的 `p_reuse/score` 仅作初始 seed。
- 真实缓存命中已由 sitecustomize 在调度器处采集：`native_hits += num_new_computed_tokens`（`h1/sitecustomize.py:1548`），经 `get_edgekv_gpu_cache_stats()` + `merge_gpu_cache_stats()` 汇总。这就是"通过回放器获取"的真实统计。
- 引擎 `_edgekv_infer_object_id`（`sitecustomize.py:870`）用 `meta.object_id/reuse_key/request_id` 定位对象；只要回放器继续传 `object_id=reuse_key=session_id + n_tokens + object_type`，对象级 profiling 照常工作。

---

## 新 Trace 格式 v2 规范

JSONL，每行一个 Session：

```json
{
  "session_id": "eIA0TuW_0",
  "trace_format": "structured_conversation_v2",
  "source": "sharegpt",
  "system_prompt": null,
  "messages": [
    {"role": "user", "content": "...", "turn_index": 0},
    {"role": "assistant", "content": "...", "turn_index": 0},
    {"role": "user", "content": "...", "turn_index": 1},
    {"role": "assistant", "content": "...", "turn_index": 1}
  ],
  "reuse_key": "eIA0TuW_0",
  "temperature": "hot"
}
```

- user/assistant 成对；`turn_index` 显式标轮次。
- **不含** `prompt_token_ids` / `reused_prefix_token_count` / `prefix_hash` / `prompt` 文本。
- 扩展位：`messages` 可容纳未来的检索上下文消息（如 `{"role":"system"/"context","content":...}`）；`reuse_key`/可选 `rag_reuse_key` 保留。

---

## Stage 1 — 新增独立生成脚本 `scripts/sharedgpt_structured_trace.py`

新建，复用 `scripts/sharedgpt_token_trace.py` 里已验证的工具，**不动**旧脚本（旧 token_delta 实验仍可复现）：

- 复用 `load_sharegpt_rows` / `iter_clean_messages` / `normalize_role` / `conversation_length`（照抄或 import）。
- 复用筛选/排序：`user_count < 2` 跳过、`--order longest`（`conversation_length` 排序）、`--max-sessions/--max-requests` 截断。
- Hot/cold 打标：产出 `temperature` 字段（沿用现有判定；纯 ShareGPT 下可先固定或按长度分桶，作为 `p_reuse_prior` 来源的钩子）。
- 逐 session 组装 v2 记录（messages 保持原始顺序、带 turn_index），`write_jsonl` 落盘（复用现成实现）。
- CLI：`--sharegpt`（默认 `DEFAULT_SHAREGPT_TRACE_PATH`）、`--out`（默认 `data/edgekv_traces/source_ablation/sharegpt_structured_v2.jsonl`）、`--order`、`--max-sessions`、`--max-requests`、`--system-prompt`。
- **不加载 tokenizer、不 apply_chat_template**（渲染职责移交回放器）。

---

## Stage 2 — 新加载器（`h0/run_h0_vllm.py`）

1. 新增 `load_structured_conversation(session, tokenizer) -> List[dict]`：
   - `history = [{"role":"system","content": session.system_prompt or DEFAULT_SYSTEM_PROMPT}]`。
   - 遍历 `session["messages"]`：
     - `role=="user"`：`prompt = tokenizer.apply_chat_template(history+[msg], tokenize=False, add_generation_prompt=True)`；产出 request 行；然后 `history.append(msg)`。
     - `role=="assistant"`（及未来其它 role）：`history.append(msg)`，不产出请求。
   - request 行字段（对齐现有 schema，供下游 CSV/引擎 meta 使用）：`request_id`(`{sid}:turn:{i:03d}`)、`session_id`、`turn_index`、`prompt`(文本)、`prompt_chars`、`prompt_est_tokens`(`estimate_tokens`)、`n_tokens`(`count_tokens(prompt, tokenizer)` — **max_model_len 过滤必需**)、`reuse_key`(=session_id，仅作对象标识/扩展位，**不再用于命中模拟**)、`workload`、`object_type`、`history_format="structured_conversation_v2"`、`history_turns`、`replay_trace_format="structured_conversation_v2"`、`replay_source="frozen_replay_trace"`、`temperature`。
   - **不产出** `prompt_token_ids` / `reused_prefix_token_count` / `prefix_hash`（run_cell 因缺 `prompt_token_ids` 自动改喂 `str(item['prompt'])`，`run_h1_vllm0110_real.py:746-750` 已支持）。
2. `replay_sessions_to_prompts`（`:1120`）分派表最前面加分支：
   ```python
   if trace_format == "structured_conversation_v2":
       session_prompts = load_structured_conversation(session, tokenizer)
   elif trace_format == "sharegpt_token_delta_v1": ...  # 旧路径保留
   ```
   其余旧加载器（token_delta / cumulative_user / rag / sharegpt）**原样保留**，向后兼容。

---

## Stage 3 — 回放器改造（`h1/run_h1_vllm0110_real.py`）核心

**移除 trace-side 命中模拟，改为只报真实统计：**

1. 删除对 `trace_side_reuse` 的调用（`:751`）与 `reuse_seen` 累积循环（`:849-857`）及 `reuse_seen` 变量（`:677`）；`trace_side_reuse` 函数（`:519`）可删除或标注废弃。
2. `request_trace_fields`（`:533`）：移除 `hit` / `hit_source` / `rag_hit` / `rag_hit_source` 这些**模拟命中**字段；不再把模拟 hit 传入 `cop.update_from_item`。COP 保留用于产出对象成本列（`c_recomp_ms`/`c_restore_ms`/`risk_exp`/`size_mb`/`object_id`）与 `p_reuse_prior`，但 `update_from_item` 的 `hit` 恒传 `False`（或改为只 seed 先验），因为**在线 p_reuse 由引擎从真实访问自算**。
3. `extra_args['edgekv_h1']`（`:768-786`）：保留 `policy/request_id/object_id/reuse_key/object_type/workload/n_tokens/c_recomp_ms/c_restore_ms/risk_exp/temperature`；**去掉伪造的 `p_reuse`/`score`**，改传 `p_reuse_prior`，让引擎 `_edgekv_profile_from_values`/`_edgekv_refresh_profile_reuse` 用真实访问+先验自估。
4. 汇总段（`:890-928`）：删除 `trace_hit_count`、基于模拟 hit 的 `workload_hit_counts`/`workload_hit_rates`；命中率**只保留** `gpu_cache_stats` 的真实口径 —— `native_hits/native_queries`（真实缓存命中率）与块级 `lookup_*`（附带、标注口径）。summary 里把主命中率字段指向真实值。
5. per-request CSV：`hit` 列移除或改为真实来源（若要 per-request 真实命中需 sitecustomize 按 request_id 归因，非本次范围 —— 本次 per-request 不出模拟 hit，命中率在 cell 级由真实 gpu stats 给出）。

**保持不变**：`run_cell` 推理循环、`request_output_timing_ms`（D4 时延指标）、`build_llm`（`enable_prefix_caching=True`）、`replay_batches`（对 `structured_conversation_v2` 同样按 session 切分，`:487` 的判定加入新 history_format）、GPU 内存监控、真实 `gpu_cache_stats` 采集与 `merge_gpu_cache_stats`。

> 注：`replay_batches`（`:479`）的 `history_format in {'cumulative_user','token_delta_chatml'}` 需追加 `'structured_conversation_v2'`，保证同一 session 多轮不进同一 batch（前缀复用要求顺序执行）。

---

## Stage 4 — 编排层 wiring（`h1/run_test.py`）

- `REPLAY_TRACE` 指向 v2 文件：`data/edgekv_traces/source_ablation/sharegpt_structured_v2.jsonl`；`TIER`/`--out-dir` 改名（如 `sharegpt_structured_v2`）避免与旧结果目录混淆。
- 生成为独立手动步骤：先跑 `scripts/sharedgpt_structured_trace.py` 产出 v2 jsonl，再跑 `run_test.py`。
- `run_step3_budget_tiers.py` / `_runner.py` **无需改**（只透传 `--replay-trace`）。
- 下游汇总 `summarize_step3_budget_tiers.py` / `summarize_step3_real.py`：核查是否引用了被删列（`hit`/`workload_hit_rates`/`trace_hit_count`），如有则改指向真实 `native_hit_rate`/`gpu_hit_rate`。

---

## 待改文件清单

- 新增 `scripts/sharedgpt_structured_trace.py`（生成器）
- `h0/run_h0_vllm.py`：新增 `load_structured_conversation` + 分派分支（`:1120`）
- `h1/run_h1_vllm0110_real.py`：移除 trace-side 命中模拟、改 extra_args、改汇总口径、`replay_batches` 加新 history_format
- `h1/run_test.py`：`REPLAY_TRACE`/`TIER`/`out-dir` 指向 v2
- `h1/summarize_step3_*.py`：核查并改列引用（按需）

---

## 验证（端到端）

1. **生成**：`python scripts/sharedgpt_structured_trace.py --max-requests 64 --out data/edgekv_traces/source_ablation/sharegpt_structured_v2.jsonl`；`head -1` 人工核对：有 `messages`、无 `prompt_token_ids/prefix_hash`。
2. **单测**：加/改 `h1/test_h1_rag_trace.py` 或 `test_h1_refactor_modules.py` —— 断言 `load_structured_conversation` 对一个小 session：user 轮数=产出请求数、每请求 `prompt` 非空且以 chat 模板结尾、无 `prompt_token_ids`、`n_tokens>0`、多轮 prompt 前缀单调增长。
3. **冒烟回放**（异步后台，遵循 async-not-polling）：小样本 `python h1/run_test.py --num-prompts 16 --budgets 0.95 --policies h1_lpe --visible-devices 1,2`；进程结束后检查：
   - 无 `trace_side_*` 报错；requests CSV 无 `hit`/`hit_source` 模拟列。
   - summary JSON 出现真实 `native_hits/native_queries` 且命中率合理（多轮 session 第 2+ 轮应有真实前缀命中 → `native_hits>0`）。
   - LPE 正常评分驱逐（`runtime_monitor.jsonl` 有 profile、`p_reuse` 非退化）。
4. **对照**：确认 vLLM 内部前缀缓存对纯文本 prompt 仍命中（同 tokenizer 渲染确定性）；`native_hit_rate` 随 budget 升高而升高，符合预期趋势。

---

## 执行进度（2026-07-07，代码已落地，尚未验证）

按用户要求「先不验证」，四个 Stage 的代码改动全部完成，`py_compile` 通过；端到端验证（生成/单测/冒烟回放/对照）尚未运行。

### Stage 1 —— 生成脚本（✅ 完成）

- 新建 `scripts/sharedgpt_structured_trace.py`：产出 `structured_conversation_v2` JSONL。
  - 每行一个 session：`messages`（role/content/turn_index 成对）、`system_prompt=null`、`reuse_key`、`temperature`、`workload`/`object_type`。
  - **不加载 tokenizer、不 apply_chat_template**；**不含** `prompt_token_ids`/`prefix_hash`/`reused_prefix_token_count`。
  - 复用旧脚本的筛选/排序逻辑：`user_count<2` 跳过、`--order longest`、`--max-sessions`/`--max-requests` 截断。
  - `temperature` 按 user 轮数分桶（≥4 为 `hot`，否则 `warm`）作为 `p_reuse_prior` 钩子。
  - CLI：`--sharegpt`/`--out`（默认 `data/edgekv_traces/source_ablation/sharegpt_structured_v2.jsonl`）/`--order`/`--max-sessions`/`--max-requests`/`--system-prompt`。

### Stage 2 —— h0 加载器（✅ 完成）

- `h0/run_h0_vllm.py` 新增常量 `DEFAULT_SYSTEM_PROMPT`（与生成器一致："You are a helpful, accurate assistant."）。
- 新增 `load_structured_conversation(session, tokenizer)`：动态累积历史 → `apply_chat_template(tokenize=False, add_generation_prompt=True)` 渲染纯文本 Prompt。
  - 每个 user 轮产出一条 request；`assistant`（及未来其它 role）只累积历史不产出请求。
  - request 行含 `n_tokens=count_tokens(prompt, tokenizer)`（**max_model_len 过滤必需**，无 token_ids 时唯一长度依据）、`history_format="structured_conversation_v2"`、`replay_trace_format`/`replay_source`/`reuse_key`/`temperature` 等；**不产出** `prompt_token_ids`/`prefix_hash`。
- `replay_sessions_to_prompts` 分派表最前面加 `trace_format == "structured_conversation_v2"` 分支；其余旧加载器原样保留，向后兼容。

### Stage 3 —— 回放器改造（✅ 完成）

- `h1/run_h1_vllm0110_real.py`：
  - **删除** `trace_side_reuse` 函数、`reuse_seen` 变量与其累积循环、`trace_reuse` 计算。
  - `request_trace_fields` 精简：移除 `hit`/`hit_source`/`rag_hit`/`rag_hit_source`/`p_reuse`/`score` 模拟字段；`cop.update_from_item` 的 `hit` 恒传 `False`；保留对象成本列（`c_recomp_ms`/`c_restore_ms`/`risk_exp`/`size_mb`/`object_id`）与 `p_reuse_prior`。
  - `extra_args['edgekv_h1']`：去掉伪造的 `p_reuse`/`score`，改传 `p_reuse_prior`，在线 `p_reuse`/`score` 由引擎从真实访问自算。
  - 汇总段：删除 `trace_hit_count`、`workload_hit_counts`/`workload_hit_rates`、`trace_side_hit_rate`/`trace_side_hit_source`；主命中率 `hit_rate` 改指向真实 `native_hit_rate`，`hit_source='gpu_prefix_cache_native'`；块级 lookup 命中率降级为附带口径（`block_lookup_hit_rate`/`gpu_block_lookup_hit_rate`）；新增 `hit_rate_source='real_gpu_prefix_cache_native'`。
  - `replay_batches` 的 `history_format` 判定集合追加 `'structured_conversation_v2'`（两处），保证同一 session 多轮不进同一 batch。
  - `[cell-done]` 日志把 `trace_side_hit_rate` 换成 `block_lookup_hit_rate`。
  - **保持不变**：`run_cell` 推理循环、`request_output_timing_ms`、`build_llm`（`enable_prefix_caching=True`）、GPU 内存监控、真实 `gpu_cache_stats` 采集与 `merge_gpu_cache_stats`。

### Stage 4 —— 编排层 wiring（✅ 完成）

- `h1/run_test.py`：`TIER`/`REPLAY_TRACE`/`--out-dir` 全部指向 `sharegpt_structured_v2`（trace 路径 `data/edgekv_traces/source_ablation/sharegpt_structured_v2.jsonl`）。
- `run_step3_budget_tiers.py`/`_runner.py` 无需改。
- `summarize_step3_*.py`：核查确认只引用了 `hit_rate`（仍存在，现已指向真实值），**无需改列引用**。

### 遗留待办（验证阶段处理）

- ⚠️ `h1/test_h1_rag_trace.py` 仍引用已删除的 `trace_side_reuse` 与旧版 `request_trace_fields(hit=..., rag_hit=...)` 签名 / `fields["hit"]` 断言（约 line 325、360–414），**会失败**。需按验证 Stage 2 重写：改为断言 `load_structured_conversation` 行为 + 新版 `request_trace_fields` 不再产出模拟 hit。
- 端到端验证（生成 v2 trace → 单测 → 冒烟回放 → budget 趋势对照）尚未运行。

