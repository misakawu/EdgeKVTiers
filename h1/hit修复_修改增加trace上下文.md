# hit 修复：ShareGPT 累计上下文 trace 与真实 KV cache 复用

## 背景

当前顺序 ShareGPT trace 的真实 hit 低，不是因为 vLLM 没有生成输出，也不是因为没有生成 KV cache。vLLM 每个请求都会正常 prefill、decode，并为输入 token 计算 KV cache。

问题在于：后续请求的 prompt 没有包含同一会话的历史上下文，因此 prefix cache 找不到相同的 token 前缀。`session_id` 只是 trace 元数据，vLLM 不会因为两个请求有相同 `session_id` 就自动续上下文。

真实多轮聊天的请求通常是累计上下文：

```text
turn0:
User: user0
Assistant:

turn1:
User: user0
Assistant: assistant0
User: user1
Assistant:

turn2:
User: user0
Assistant: assistant0
User: user1
Assistant: assistant1
User: user2
Assistant:
```

而当前 `sharedgpt_ordered.jsonl` 更接近：

```text
turn0:
user0

turn1:
user1

turn2:
user2
```

这种格式虽然 trace-side 可以用 `reuse_key=session_id` 统计出较高命中，但真实 vLLM prefix cache 看的是 prompt 开头 token 是否一致。同一会话的相邻请求如果没有共同历史前缀，真实 hit 就会很低。

另一个问题是 H1 日志目前按每次 `LLM.generate()` 输出 vLLM tqdm：

```text
Adding requests: 100%|...
Processed prompts:   0%|...
```

这会在 batch 较多时刷屏，并且这些日志只描述单个 batch，不利于观察整个测试任务的进度和结果。

## 最终方案

取消“换 vLLM 调用接口”的方案。仅切换到 `LLM.chat` 或 OpenAI-compatible chat API 不能解决问题，因为这些接口同样不会自动保存会话上下文。无论使用 `generate` 还是 `chat`，历史都必须由回放器或 trace 显式传入。

最终采用：

1. 修改三个 ShareGPT 相关 trace 生成器，按原始 ShareGPT 会话结构生成累计上下文 trace。
2. 同步修改 H0/H1 回放器，使其明确识别、展开、校验累计上下文格式。
3. 继续使用现有 vLLM `LLM.generate` 回放路径，不引入自动会话状态。
4. 调整 H1 回放时序，确保需要复用的前一轮请求已经完成并进入 prefix cache。
5. 禁用 vLLM per-batch tqdm，改由 H1 输出任务级进度和摘要日志。

目标数据流：

```text
原始 ShareGPT conversations
-> 生成器输出 cumulative session JSONL
-> 回放器按 session/turn 展开请求
-> H1 按可复用时序提交 generate
-> vLLM generate 回放完整历史 prompt
-> prefix cache 基于真实历史前缀命中
```

默认选择：

- 继续使用 `LLM.generate`，不切换 `LLM.chat`。
- ShareGPT prompt 默认使用当前 `User:/Assistant:` 文本模板。
- ShareGPT hit 修复 smoke/真实 hit 验证默认使用 `--replay-batch-size 1 --batch-order original`。
- 保留 legacy flat/`complete_prompt` trace 兼容。

## 生成器修改

需要同步修改三个入口：

- `scripts/sharedgpt_order_trace.py`
- `scripts/sharedgpt_jsonl_generation.py`
- `scripts/optimize_h0_pressure_trace.py` 的 ShareGPT 分支

三个生成器的 ShareGPT 路径统一输出会话型 JSONL：每行一个原始 ShareGPT session，而不是每行一个扁平 prompt。

每个 session 的基本结构：

```json
{
  "session_id": "原始 ShareGPT 会话 ID",
  "source": "sharegpt",
  "object_type": "sharegpt_session_prefix",
  "reuse_key": "同 session_id",
  "turns_format": "cumulative_user",
  "turns": [
    {
      "i": 0,
      "user": "User: user0\nAssistant:"
    },
    {
      "i": 1,
      "user": "User: user0\nAssistant: assistant0\nUser: user1\nAssistant:"
    }
  ]
}
```

生成规则：

- 按原始 ShareGPT `conversations` 的会话边界生成。
- 会话内保持原始 user/assistant 顺序。
- 跳过空文本。
- 至少保留 2 个有效 user/human turn 的会话。
- 每个 turn 的 `user` 字段存完整累计 prompt。
- 历史 assistant 文本来自原始 ShareGPT 数据，不使用本次 vLLM 生成输出。
- `reuse_key` 使用原始 `session_id`，表示该会话内的历史 prefix 可复用。
- `max_turns` 表示展开后的最大请求数；必要时可以截断最后一个 session 的 `turns`，保证展开请求数不超过上限。

`sharedgpt_order_trace.py` 的变化：

- 从“按原始顺序输出当前 user message”改成“按原始顺序输出 cumulative session”。
- 原有 `max_turns` 语义保留为展开后的最大请求数。
- 不再输出顶层 `prompt` 扁平请求。

`sharedgpt_jsonl_generation.py` 的变化：

- 不再默认生成 `HOT+WARM+TAIL` 合成层级 prefix trace。
- 改为从原始 ShareGPT 会话生成累计上下文 trace。
- 旧的 hot/warm/Zipf 参数保留 CLI 兼容，但 summary 中标记为 `ignored_for_cumulative_sharegpt=true`。
- summary 中使用 `prefix_layout="original_conversation_cumulative"`。

`optimize_h0_pressure_trace.py` 的变化：

- ShareGPT source-mode 改用同一套 cumulative session 构造逻辑。
- mixed/pressure trace 中 ShareGPT 部分保持真实多 turn 会话结构。
- RAG/HotQA 部分不强行改成 ShareGPT 会话格式，继续使用现有 complete prompt/RAG 格式。
- mixed trace 允许同一 JSONL 内同时存在 ShareGPT cumulative session 与 HotQA/RAG session。
- `original_sharegpt_random_rag` 旧 flat 分支保留为 legacy mode，但不作为 hit 修复默认路径。
- summary 中区分 `sharegpt_cumulative_sessions`、`sharegpt_cumulative_turns` 和 RAG/HotQA 合成请求。

## 回放器修改

需要同步适配 H0/H1 replay loader。即使现有代码已经部分兼容 `turns_format="cumulative_user"`，也要把它提升为明确支持的 trace 格式，并补齐校验、字段和 summary。

H0 replay loader 修改点：

- 将 `turns_format="cumulative_user"` 作为一等格式处理。
- 严格校验每个 cumulative session：
  - 必须有非空 `turns` list。
  - 每个 turn 必须有整数兼容的 `i`。
  - 每个 turn 必须有非空 `user`。
- 展开时保留：
  - `session_id`
  - `request_id`
  - `turn_index`
  - `reuse_key`
  - `object_type`
  - `workload`
  - `replay_source`
- 展开请求时增加诊断字段：
  - `history_format="cumulative_user"`
  - `history_turns=turn_index + 1`
  - `history_prompt_chars=len(prompt)`
  - `replay_trace_format="session_cumulative_user"`
- 对 cumulative trace，展开后的 `prompt` 必须严格等于 turn 中的累计 `user` 字段加可选 RAG 前缀。

H1 回放器修改点：

- `load_trace()` 继续调用共享 replay loader，但 `trace_used.jsonl` 要保留新增历史字段。
- request CSV 中保留 `history_format`、`history_turns`、`history_prompt_chars`、`replay_trace_format`。
- summary 增加：
  - `replay_trace_format`
  - `history_format`
  - `cumulative_sessions`
  - `cumulative_turns`
  - `trace_side_hit_rate`
  - `native_hit_rate`
- 保持真实 hit 统计来源为 vLLM native prefix cache/block lookup。
- CSV/summary 中明确区分：
  - trace-side hit：由 `reuse_key` 推断。
  - native hit：由 vLLM 真实 prefix cache 统计。

## 回放时序与 batch 策略

原“批处理策略保持现状”的方案不够安全。原因是如果同一 session 的 turn0 和 turn1 在同一个 `LLM.generate(prompts, sampling)` batch 内提交，turn1 查询 prefix cache 时，turn0 的 KV block 可能还没有完成并进入可复用缓存。这样即使 prompt 有共同历史前缀，也不一定产生真实 native hit。

修复后的策略：

- ShareGPT hit 修复的正确性验证默认使用：
  - `--replay-batch-size 1`
  - `--batch-order original`
- 如果使用 batch size > 1，必须保证同一 session 的 turn k+1 不与 turn k 在同一个 `LLM.generate` batch 内。
- 可选新增 `session_completed` 调度策略：
  - 按原始展开顺序执行。
  - 遇到同一 session 依赖上一 turn 时强制切 batch。
  - 保证前一轮请求完成后，后一轮才可能复用其 KV blocks。
- 不切换 `LLM.chat`。
- 不实现模型生成输出自动进入下一轮上下文。

## 日志优化

H1 现有日志中大量重复的：

```text
Adding requests: ...
Processed prompts: ...
```

来自每次 `LLM.generate()` 的 vLLM tqdm，描述的是单个 batch，不是整个测试任务。修复后默认禁用这类 per-batch 进度条，改用任务级日志。

修改点：

- warmup 和 replay 的 `llm.generate()` 都传入 `use_tqdm=False`。
- 若当前 vLLM 版本签名不支持 `use_tqdm`，使用兼容 helper 检测签名后 fallback 调用，避免破坏旧版本。
- 保留 vLLM engine INFO 日志，例如 prefix cache hit rate、KV usage；这些是全局状态信息，不属于重复 batch tqdm 噪声。
- H1 自己输出任务级日志：
  - cell 开始：policy、budget、requests、batch size、batch count、trace path。
  - 运行中：每 N 个 batch 或每 10% 输出一行 `[replay]` 进度。
  - cell 结束：requests、elapsed、native hit rate、trace-side hit rate、ttft p95、GPU memory peak。
- summary/config 记录：
  - `vllm_generate_tqdm=false`
  - `replay_progress_log="task_level"`

## 验证计划

trace 结构验证：

- 新 trace 每行都有 `turns`。
- 新 trace 不再输出顶层 `prompt` 扁平请求。
- `turns_format` 全部为 `cumulative_user`。
- 每个有效 session 至少包含 2 个 turn。
- 同一 session 相邻 turn 的最长公共前缀中位数显著大于 0。

loader 验证：

- `replay_sessions_to_prompts()` 能正确展开 cumulative session。
- 展开后的 `prompt` 等于 turn 中的累计 `user` 字段。
- `reuse_key=session_id` 保持不变。
- `turn_index` 和原始会话 turn 顺序一致。
- `history_format`、`history_turns`、`history_prompt_chars`、`replay_trace_format` 存在。
- 超长 prompt 仍由 H1 `max_model_len` 过滤。
- legacy flat prompt 和 `complete_prompt` trace 仍可加载。

日志验证：

- mock `LLM.generate`，确认 warmup/replay 调用传入 `use_tqdm=False` 或走兼容 fallback。
- 检查 H1 日志中不再出现 `Adding requests:` 和 `Processed prompts:`。
- 检查日志中存在任务级 `[replay]` 进度与 cell 结束摘要。

H1 smoke 验证：

- 使用新 trace 跑 64 或 128 requests。
- 单 budget、单 policy 即可。
- 默认使用 `--replay-batch-size 1 --batch-order original`。
- 检查 `trace_used.jsonl` 中 turn1/turn2 prompt 包含历史 assistant。
- 对比旧 `sharedgpt_ordered.jsonl`，真实 `native_hit_rate` 应明显提升。
- 同时报告 trace-side hit 与 native hit，不用 trace-side hit 证明真实 KV 复用。

## 预期结果

修复后，顺序 ShareGPT 实验不再只是 `session_id` 层面的 trace-side reuse，而是让 vLLM 看到真实累计上下文前缀，并在正确的请求完成时序下复用 KV blocks。

预期变化：

- 同会话后续 turn 的 prompt 共享历史 token 前缀。
- 前序 turn 完成后，vLLM prefix cache 可以复用其 KV blocks。
- 真实 `native_hit_rate` 相比当前顺序扁平 trace 明显提升。
- trace-side hit 与 native hit 的含义更清晰，不再把“同 session”误解为“真实 prefix 已复用”。
- H1 日志不再被 per-batch tqdm 刷屏，而是输出整个测试任务维度的进度和摘要。
