# 修复 hit 回放器：窗口化 round_robin 计划

## 1. Summary

将现有 `round_robin` 从“全 trace 所有 session 轮询”改为“按活跃 session 窗口轮询”。

窗口大小默认等于 `replay_batch_size`，使 batch 仍由不同 session 组成，但同一组
session 会先完成各自多轮请求，再进入下一组 session，避免当前
`s0:t0 ... s246:t0, s0:t1` 这种超长复用间隔导致 prefix cache 被覆盖。

## 2. Key Changes

- 修改 `h1/run_h1_vllm0110_real.py`：
  - `order_trace_for_batches()` 增加可选参数：

    ```python
    replay_batch_size: int | None = None
    ```

  - `batch_order == "round_robin"` 的语义改为 windowed round-robin：
    1. 先按 session 出现顺序分组。
    2. 每个 session 内按 `_trace_original_index` 排序，缺失时用遍历顺序兜底。
    3. 每次取 `window_size = replay_batch_size` 个 session 作为活跃窗口。
    4. 在窗口内轮询输出：

       ```text
       s0:t0, s1:t0, ..., s0:t1, s1:t1, ...
       ```

    5. 当前窗口所有 session 耗尽后，再处理下一窗口。
    6. 无 `session_id` 的 orphan 请求仍按原始顺序追加到末尾。

- 修改 `run_cell()` 调用：

  ```python
  replay_trace = order_trace_for_batches(
      trace,
      args.batch_order,
      args.replay_batch_size,
  )
  ```

- 更新 argparse help：
  - `round_robin` 表示按 replay batch size 分组的 session 窗口轮询。

- 保持 `replay_batches()` 现有保护不变：
  - batch 达到 `replay_batch_size` 切批。
  - batch 内出现重复 `session_id` 强制切批。
  - 这保证窗口内不规则 session 长度不会把同 session 多轮塞进同一个 batch。

- 保持 `h1/run_test.py` 入口仍只开放 `round_robin`：
  - 不新增新模式名。
  - 不修改当前已调好的 `REPLAY_BATCH_SIZE/MAX_MODEL_LEN/MAX_NUM_BATCHED_TOKENS` 配置。
  - 默认行为直接变为窗口轮询。

## 3. Test Plan

- 更新现有 round-robin 测试期望：
  - `replay_batch_size=3`、3 个 session 每个 2 轮时，顺序仍为：

    ```text
    s0:t0, s1:t0, s2:t0, s0:t1, s1:t1, s2:t1
    ```

  - `replay_batch_size=2`、3 个 session 每个 2 轮时，顺序应为：

    ```text
    s0:t0, s1:t0, s0:t1, s1:t1, s2:t0, s2:t1
    ```

  - session 长度不均时，窗口内短 session 耗尽后长 session 继续输出，且不会乱序。
  - orphan 请求保持原始顺序追加到末尾。
  - 对 windowed round-robin 输出调用 `replay_batches(..., replay_batch_size=N)`，断言每个 batch 内 session 唯一。

- 运行验证：

  ```bash
  python -m pytest h1/test_h1_rag_trace.py -k 'round_robin'
  python -m py_compile h1/run_test.py h1/run_step3_budget_tiers.py h1/run_h1_vllm0110_real.py
  ```

- 如需完整实验，再运行：

  ```bash
  python h1/run_test.py --force
  ```

  验收重点：
  - summary 中 `batch_order=round_robin`。
  - requests CSV 中每个 `batch_start_index` 不含重复 `session_id`。
  - 同一 session 的相邻 turn 间隔应接近活跃窗口大小，而不是全 trace session 数。

## 4. Assumptions

- 直接替换 `round_robin` 语义，不保留旧的全局 round-robin 模式。
- 活跃窗口大小固定为 `replay_batch_size`，不新增 CLI 参数。
- 本次只修 replay 排序逻辑，不改 vLLM KV cache 管理、不改预算计算、不改 summary 指标口径。
