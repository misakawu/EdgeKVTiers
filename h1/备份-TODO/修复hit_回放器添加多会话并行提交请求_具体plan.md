# 修复 hit 回放器：多会话并行提交请求具体计划

## 1. 目标

把 `h1/run_test.py` 驱动的 vLLM 回放链路从旧的单请求/原始顺序 batch，改为新的多会话并行 batch：

- 每次提交给 `LLM.generate()` 的 batch，尽量包含来自不同 `session_id` 的一条请求。
- 同一会话内部仍保持原始 turn 顺序，不能把后续轮次排到前面。
- 默认运行就使用新并行 batch，不再依赖旧的 `REPLAY_BATCH_SIZE = 1` 单请求设置。

实际排序和 batch 生成发生在底层回放器 `h1/run_h1_vllm0110_real.py`，`h1/run_test.py` 是入口配置，需要把默认参数切到新模式。

## 2. 实现改动

### 2.1 新增 round-robin 排序

修改 `h1/run_h1_vllm0110_real.py` 的 `order_trace_for_batches()`：

- 保留现有 `original` 和 `length_bucket` 能力，新增 `round_robin`。
- `round_robin` 逻辑：
  1. 遍历过滤后的 `trace`。
  2. 有非空 `session_id` 的请求按 session 分组。
  3. 每个 session 组内按 `_trace_original_index` 排序；如果缺失该字段，则用遍历顺序兜底。
  4. 按 session 出现顺序做轮询，每轮从每个未耗尽 session 中取一条请求。
  5. 无 `session_id` 的请求作为 orphan，保持原始顺序追加到最后。
- 这样重排后的序列形如：

```text
s0:t0, s1:t0, s2:t0, s0:t1, s1:t1, s2:t1, ...
```

而不是：

```text
s0:t0, s0:t1, s0:t2, s1:t0, s1:t1, ...
```

### 2.2 保留 batch 拆分保护

`replay_batches()` 保持现有行为：

- 当当前 batch 达到 `replay_batch_size` 时切批。
- 当当前 batch 已经包含同一个 `session_id` 时强制切批。

这层保护可以保证即使某些 session 已经耗尽、round-robin 后段出现同 session 相邻请求，也不会把同一会话的多轮请求并行提交到同一个 batch。

### 2.3 扩展 CLI batch-order

修改 `h1/run_h1_vllm0110_real.py` 的 argparse：

```python
choices=('original', 'length_bucket', 'round_robin')
```

help 文案说明：

- `original`：保持 trace 顺序。
- `length_bucket`：按长度桶重排。
- `round_robin`：按 session 轮询，使 batch 尽量包含不同会话。

### 2.4 修改 step3 参数传递

修改 `h1/run_step3_budget_tiers.py`：

- 新增常量：

```python
BATCH_ORDER = "round_robin"
```

- `cell_args()` 固定追加：

```python
"--batch-order", BATCH_ORDER,
```

- `run_step3()` 增加可选参数 `batch_order=BATCH_ORDER`，并传给 `cell_args()`。
- 日志中打印 `order={batch_order}`，便于确认运行配置。

### 2.5 修改 h1/run_test.py 默认并行配置

修改 `h1/run_test.py`：

- 删除旧的单请求默认：

```python
REPLAY_BATCH_SIZE = 1
MAX_NUM_BATCHED_TOKENS = MAX_MODEL_LEN
```

- 改为新并行默认：

```python
REPLAY_BATCH_SIZE = 8
MAX_NUM_BATCHED_TOKENS = None
BATCH_ORDER = "round_robin"
```

- argparse 增加：

```python
parser.add_argument("--batch-order", choices=("round_robin",), default=BATCH_ORDER)
```

这里入口只开放新并行模式，避免继续误用旧 batch 顺序。

- 调用 `step3.run_step3()` 时传入：

```python
batch_order=args.batch_order
```

- `max_num_batched_tokens` 默认计算改为：

```python
args.max_num_batched_tokens
if args.max_num_batched_tokens is not None
else args.max_model_len * args.replay_batch_size
```

默认 `max_model_len=4096`、`replay_batch_size=8` 时，`max_num_batched_tokens=32768`，避免 token 上限把实际并行压低。

## 3. 行为示例

输入 trace：

```text
s0:t0, s0:t1, s0:t2,
s1:t0, s1:t1,
s2:t0, s2:t1
```

`round_robin` 后：

```text
s0:t0, s1:t0, s2:t0,
s0:t1, s1:t1, s2:t1,
s0:t2
```

`replay_batch_size=3` 时，batch 为：

```text
[s0:t0, s1:t0, s2:t0]
[s0:t1, s1:t1, s2:t1]
[s0:t2]
```

`replay_batch_size=8` 且 session 数足够时，每次 `LLM.generate()` 会并行提交最多 8 个不同会话的一条请求。

## 4. 测试计划

在 `h1/test_h1_rag_trace.py` 增加测试：

1. `test_h1_order_trace_round_robin_interleaves_sessions`
   - 构造 3 个 session、每个 2 轮。
   - 验证输出顺序为 `s0:t0, s1:t0, s2:t0, s0:t1, s1:t1, s2:t1`。

2. `test_h1_order_trace_round_robin_handles_uneven_sessions`
   - 构造 session 长度不均：`s0` 三轮，`s1` 一轮，`s2` 两轮。
   - 验证短 session 耗尽后，长 session 剩余请求继续输出，且会话内 turn 顺序不变。

3. `test_h1_order_trace_round_robin_appends_orphans`
   - 构造有 `session_id` 和无 `session_id` 的混合请求。
   - 验证 orphan 请求保持原始顺序追加到末尾。

4. `test_h1_round_robin_batches_have_unique_sessions`
   - 对 round-robin 后的 trace 调用 `replay_batches(trace, replay_batch_size=8)`。
   - 验证每个 batch 内没有重复 `session_id`。

运行验证：

```bash
python -m pytest h1/test_h1_rag_trace.py
python -m py_compile h1/run_test.py h1/run_step3_budget_tiers.py h1/run_h1_vllm0110_real.py
```

## 5. 验收标准

- `h1/run_test.py` 默认运行时 summary/config 中记录：
  - `batch_order = "round_robin"`
  - `replay_batch_size = 8`
  - `max_num_batched_tokens = 32768`，除非用户显式覆盖。
- 生成的 requests CSV 中：
  - 同一 `batch_start_index` 对应的一组请求不应包含重复 `session_id`。
  - `event_index` 顺序体现 session 轮询。
- vLLM 每次 `LLM.generate()` 收到的是不同会话组成的 prompt 列表，从而更接近真实多用户并发提交。

## 6. 默认假设

- 新并行 batch 默认大小使用 8，和原方案示例一致，压力足够且比 16 更稳妥。
- 不修改 vLLM 引擎调用方式；仍然通过一次 `LLM.generate(prompts, sampling)` 提交一个 batch。
- 只把 `h1/run_test.py` 所在链路默认切换到新并行模式；其他入口如果复用 `run_step3_budget_tiers.py` 会继承该默认，否则只获得底层 `round_robin` 兼容能力。
