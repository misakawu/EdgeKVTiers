# H1 Token-Physical Trace 方案

## Summary

构建新的 ShareGPT trace 生成器，放弃旧 `cumulative_user` 文本堆叠和人工 `reuse_key` 命中口径，改为以模型 tokenizer 生成的 `prompt_token_ids` 为主数据。H1 回放器直接把 token IDs 交给 vLLM，让 prefix cache 基于真实 token 前缀匹配，同时记录 LCP、prefix hash、实际新增 prefill token 等指标。

## Key Changes

- 新增 token trace 生成器，默认使用 Qwen2.5 tokenizer 的 `apply_chat_template`：
  - 每行一个 ShareGPT session。
  - 顶层包含 `trace_format="sharegpt_token_delta_v1"`、`tokenizer_name_or_path`、`chat_template_source`、`system_prompt`、`system_fingerprint`、`session_group_id`。
  - 每个 turn 保存 `delta_messages`、`prompt_token_ids`、`prompt_token_count`、`reused_prefix_token_count`、`new_prefill_token_count`、`prefix_hash`。
  - 不再写 `reuse_key`；仅保留 `session_group_id` 作为日志聚合字段，不参与 hit 判定。
- prompt 构造规则：
  - 固定 system message 放在所有 session 头部。
  - turn k 的请求 token 为 `[system] + user0 + assistant0 + ... + user_k + generation_prompt`。
  - 历史 assistant 内容来自 ShareGPT 原始 assistant 消息，不使用本次 vLLM 生成结果。
  - 跳过空文本和少于 2 个有效 user turn 的 session。
- H0/H1 replay loader 增加一等支持：
  - 识别 `trace_format="sharegpt_token_delta_v1"`。
  - 校验 `prompt_token_ids` 非空、`prompt_token_count` 与实际长度一致、turn index 单调。
  - 展开后保留 `prompt_token_ids`，并补充 `history_format="token_delta_chatml"`、`replay_trace_format="sharegpt_token_delta_v1"`。
  - legacy `cumulative_user`、flat prompt、`complete_prompt` 继续兼容。
- H1 `run_h1_vllm0110_real.py` 回放路径调整：
  - 若 item 有 `prompt_token_ids`，调用 vLLM 时传 token prompt input，而不是字符串 prompt。
  - `n_tokens` 以 `len(prompt_token_ids)` 为准。
  - `trace_side_hit_rate` 改为基于 `reused_prefix_token_count > 0` 的诊断值；真实结果仍以 vLLM native prefix cache stats 为准。
  - request CSV/summary 新增 `token_prefix_hash`、`reused_prefix_token_count`、`new_prefill_token_count`、`system_prefix_token_count`。
- H1 `run_test.py` 启动器调整：
  - 默认 trace 指向新 token trace，例如 `data/edgekv_traces/有效实验数据/sharedgpt_token_v1.jsonl`。
  - 默认 `REPLAY_BATCH_SIZE = 1`，`MAX_NUM_BATCHED_TOKENS = MAX_MODEL_LEN`，保证同一 session 前一 turn 完成后后一 turn 才复用。
  - 保留 CLI 覆盖 batch size、trace path、budgets、policies。
  - tier 改名为 `sharedgpt_token_v1`，输出目录与旧结果隔离。

## Test Plan

- 生成器单测：
  - 新 trace 无顶层 `prompt`、无 `reuse_key`。
  - 每个 turn 有 `prompt_token_ids`，且 `prompt_token_count == len(prompt_token_ids)`。
  - 相邻 turn 的 `reused_prefix_token_count` 等于两个 token 数组 LCP。
  - system prefix token 在所有请求开头完全一致。
- loader 单测：
  - token trace 正确展开为 H1 replay item。
  - 过长 prompt 仍按 `len(prompt_token_ids) + max_tokens <= max_model_len` 过滤。
  - legacy `cumulative_user` 测试保持通过。
- H1 dry/smoke 验证：
  - mock `LLM.generate` 确认 token trace 走 token input，不走字符串 prompt。
  - `trace_used.jsonl` 保留 token 计量字段。
  - 小规模 smoke 用 `--replay-batch-size 1 --batch-order original`，检查 native hit 不再依赖 `reuse_key`。
  - 确认 `use_tqdm=False` 行为不回退。

## Assumptions

- 默认模型/tokenizer 为当前实验使用的 `models/Qwen2.5-7B-Instruct`。
- 默认使用 tokenizer 自带 `chat_template`；若 tokenizer 没有 template，则生成器直接报错，不静默退回旧 `User:/Assistant:` 模板。
- 新 trace 以准确模拟 vLLM token 物理结构优先，接受 trace 与 tokenizer 绑定。
- 后续实现时再新增文件、修改 loader 和启动器。

## Implementation Progress (2026-07-07)

### Done

- 新增 `scripts/sharedgpt_token_trace.py`。
  - 使用 `AutoTokenizer.from_pretrained(models/Qwen2.5-7B-Instruct)` 和 tokenizer 自带 `apply_chat_template` 生成 token-physical trace。
  - 输出格式为 `trace_format="sharegpt_token_delta_v1"`。
  - 默认 compact 输出只保留 vLLM replay 和校验需要的数据：顶层 `trace_format`、`system_prefix_token_count`、`session_group_id`、`session_id`、`turns`；turn 内 `i`、`request_id`、`prompt_token_ids`、`prompt_token_count`、`reused_prefix_token_count`、`new_prefill_token_count`、`prefix_hash`。
  - 默认不再写管理员查看用的 `prompt`、`delta_messages`、`system_prompt`、`system_fingerprint`、`tokenizer_name_or_path`、`chat_template_source`、`source`、`workload`、`object_type`、`source_index`，也不写 `reuse_key`。
  - 如需人工排查，可用 `--include-debug-fields` 临时恢复 prompt/messages/provenance 字段。
  - 默认输出路径为 `data/edgekv_traces/source_ablation/sharedgpt_token_v1.jsonl`。

- 更新 `h0/run_h0_vllm.py` 共享 replay loader。
  - 新增 `lcp_token_count`、`token_prefix_hash`。
  - 新增 `token_delta_session_to_prompts`。
  - `replay_sessions_to_prompts` 可识别 `trace_format="sharegpt_token_delta_v1"`。
  - loader 会校验 token id 非空、`prompt_token_count`、turn index 单调、LCP、`new_prefill_token_count`。
  - 展开后的 item 保留 `prompt_token_ids`，并补齐 `history_format="token_delta_chatml"`、`replay_trace_format="sharegpt_token_delta_v1"`。

- 更新 `h1/run_h1_vllm0110_real.py`。
  - `load_trace` 对 token trace 使用 `len(prompt_token_ids)` 作为 `n_tokens` 和过长过滤依据。
  - `order_trace_for_batches` 的长度桶使用 token id 长度。
  - `replay_batches` 对 `token_delta_chatml` 与 `cumulative_user` 一样拆分同 session 相邻 turn，避免同 batch 内复用顺序不确定。
  - vLLM `LLM.generate` 输入在存在 `prompt_token_ids` 时改为 `{"prompt_token_ids": [...]}`，否则继续使用字符串 prompt。
  - token trace 的 trace-side hit 诊断改为 `reused_prefix_token_count > 0`，`hit_source="trace_side_token_lcp"`。
  - request CSV 不写完整 `prompt_token_ids`，但保留 `token_prefix_hash`、`reused_prefix_token_count`、`new_prefill_token_count`、`system_prefix_token_count` 等计量字段。
  - summary 新增 `token_delta_sessions`、`token_delta_turns`。

- 更新 `h1/run_test.py` 启动器。
  - 默认 `TIER="sharedgpt_token_v1"`。
  - 默认 trace 指向 `data/edgekv_traces/有效实验数据/sharedgpt_token_v1.jsonl`。
  - 默认 `REPLAY_BATCH_SIZE=1`。
  - 默认 `MAX_NUM_BATCHED_TOKENS=MAX_MODEL_LEN`。
  - 新增 CLI 覆盖 `--replay-batch-size`、`--max-model-len`、`--max-num-batched-tokens`。
  - 默认输出目录改为 `h1/out/sharedgpt_token_v1/...`，与旧结果隔离。

- 更新 `h1/test_h1_rag_trace.py`。
  - 增加生成器契约测试。
  - 增加 token trace loader 展开测试。
  - 增加 token trace replay batch 拆分测试。
  - 增加 token trace trace-side hit 使用 LCP 的测试。

### Generated Trace

已生成默认 trace：

```bash
conda run -n edgekv-vllm0110 python scripts/sharedgpt_token_trace.py   --out data/edgekv_traces/有效实验数据/sharedgpt_token_v1.jsonl   --max-requests 1536
```

生成器输出：

```json
{"out": "data/edgekv_traces/有效实验数据/sharedgpt_token_v1.jsonl", "requests": 1562, "sessions": 36, "tokenizer_name_or_path": "/DATACENTER3/zhenxiang.wang/work/EdgeKVTiers/models/Qwen2.5-7B-Instruct", "trace_format": "sharegpt_token_delta_v1"}
```

说明：生成器按 session 写完整行，因此达到 `--max-requests 1536` 后保留最后一个完整 session，实际 trace 有 1562 个请求。`h1/run_test.py` 默认 `--num-prompts 1536` 会在 loader 展开后截取前 1536 个请求。

当前 compact 文件大小：约 8.5 MiB。参考旧 debug 版约 16 MiB。

结构简要说明：

  {
    "trace_format": "sharegpt_token_delta_v1",
    "session_id": "...",
    "session_group_id": "...",
    "system_prefix_token_count": 13,
    "turns": [...]
  }

  顶层字段描述：

  - trace_format：新格式标识，loader 用它识别 token trace。
  - session_id / session_group_id：会话聚合 ID；不作为真实 hit 判定依据。
  - system_prefix_token_count：固定 system 前缀长度。
  - turns：该 session 内每个 user turn 对应一次 replay request。

  每个 turn 类似：

  {
    "i": 1,
    "request_id": "eIA0TuW_0:turn:001",
    "prompt_token_ids": [151644, 8948, ...],
    "prompt_token_count": 218,
    "reused_prefix_token_count": 172,
    "new_prefill_token_count": 46,
    "prefix_hash": "5ed829dd261fad5c"
  }

  关键点：

  - prompt_token_ids 是 vLLM 实际回放用的主数据。
  - prompt 和 delta_messages 默认不写入 trace；它们只是可读调试数据，不参与 vLLM replay。
  - reused_prefix_token_count 是和同 session 上一个 turn 的 token LCP 长度。
  - new_prefill_token_count = prompt_token_count - reused_prefix_token_count。
  - prefix_hash 是整个 prompt token 序列的短 hash，用于日志定位。

### Verification

已通过语法检查：

```bash
conda run -n edgekv-vllm0110 python -m py_compile   scripts/sharedgpt_token_trace.py   h0/run_h0_vllm.py   h1/run_h1_vllm0110_real.py   h1/run_test.py   h1/test_h1_rag_trace.py
```

已通过 H1 相关单测：

```bash
conda run -n edgekv-vllm0110 python -m pytest h1/test_h1_rag_trace.py -q
```

结果：

```text
29 passed, 2 warnings in 7.75s
```

已做生成器小样本验证：

```bash
conda run -n edgekv-vllm0110 python scripts/sharedgpt_token_trace.py   --sharegpt /tmp/sharegpt_fixture.json   --max-sessions 1   --out /tmp/sharedgpt_token_test.jsonl
```

结果：生成 1 个 session / 2 个 requests，格式为 `sharegpt_token_delta_v1`。

### Remaining

- 尚未运行真实 GPU H1 smoke。建议命令形态：

```bash
conda run -n edgekv-vllm0110 python h1/run_test.py   --num-prompts 16   --budgets 0.95   --policies h1_lpe   --replay-batch-size 1   --max-model-len 4096   --out-dir sharedgpt_token_v1_smoke   --force
```

- GPU smoke 重点检查：`trace_used.jsonl`/request CSV 中保留 token 计量字段，vLLM native prefix cache stats 不再依赖 `reuse_key`，且 `use_tqdm=False` 没有回退。

## Trace Structure Notes

### `session_id` / `session_group_id`

`session_id` 和 `session_group_id` 表示同一个 ShareGPT 多轮对话的会话聚合 ID。当前 JSONL 中每一行是一个 ShareGPT session，`turns` 数组里保存该 session 内的多次 user turn replay request。

它们主要用于：

- 日志聚合：可以按 session 查看第 0、1、2... 个 turn 的表现。
- batch 拆分：H1 replay 会避免同一个 session 的相邻 turn 放进同一个 batch，保证前一个 turn 完成后，后一个 turn 才可能复用 prefix cache。
- 诊断展示：CSV/summary 中可以按 session 观察 token prefix 增长。

它们不再作为真实 hit 判定依据。旧 trace 里常用 `reuse_key=session_id` 推断“同一 session 后续 turn 就算 hit”，这个口径不准确。新 trace 改为看真实 token 前缀：是否能复用，由相邻请求 `prompt_token_ids` 的最长公共前缀决定。

示意：

```text
turn 0 prompt_token_ids = A
turn 1 prompt_token_ids = A + assistant0 + user1 + generation_prompt
reused_prefix_token_count = LCP(turn0, turn1)
```

### Current Trace Generation Logic

当前 `sharedgpt_token_v1.jsonl` 的生成思路：

1. 读取 ShareGPT 原始会话，每个原始会话生成 JSONL 的一行。
2. 跳过空文本、少于 2 个有效 user turn 的会话。
3. 每个 session 开头加固定 system message。
4. 对第 `k` 个 user turn，构造完整 chat prompt：

```text
system
user0
assistant0
...
user_k
assistant generation prompt
```

5. 用 Qwen2.5 tokenizer 的 `apply_chat_template` 生成真实 `prompt_token_ids`。
6. 对同一 session 内相邻 turn 计算 token LCP：
   - 第 0 turn：`reused_prefix_token_count = 0`。
   - 第 1 turn：和第 0 turn 比较 token 前缀。
   - 第 2 turn：和第 1 turn 比较 token 前缀。
7. 写出每个 turn 的 token 计量字段：
   - `prompt_token_count`
   - `reused_prefix_token_count`
   - `new_prefill_token_count`
   - `prefix_hash`

核心目标：让 replay 输入尽量贴近真实 LLM chat 请求的 token 结构，让 vLLM prefix cache 按真实 token 前缀自然命中，而不是靠人工 `reuse_key` 统计命中。
