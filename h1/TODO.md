当前 `ttft_proxy_ms` 被错误地设置为整个 batch 的生成耗时（wall time），而不是每个请求自身的首次 token 延迟。当 `--replay-batch-size 64` 时，这个值会被复制给 batch 内的所有请求，导致 TTFT 严重虚高（p95 ~10.5s）。同时，`queue_wait_ms` 和 `prefill_ms` 虽然尝试从 `RequestOutput.metrics` 提取，但目前**并未用于计算 TTFT**，所以修正 TTFT 是修复该指标的第一步。

### ✅ 推荐的修复方案
为每个请求独立计算 TTFT，充分利用 vLLM `RequestOutput.metrics` 中的时间戳（如 `first_token_time` 和 `arrival_time` 或 `scheduled_time`）。我们需要修改两个地方：

1. **增强 `request_output_timing_ms()`**：增加 `ttft_ms` 输出。
2. **在 `run_cell()` 中**：使用每个请求的 `ttft_ms` 替代整个 batch 的墙钟时间。

### 📝 具体代码修改（基于 `run_h1_vllm0110_real.py`）

#### 修改 `request_output_timing_ms` 函数（约第 88 行）
```python
def request_output_timing_ms(output: Any) -> dict[str, float | str | bool]:
    """Extract vLLM RequestMetrics timing fields, falling back to zeros."""
    metrics = getattr(output, 'metrics', None)
    if metrics is None:
        return {
            'queue_wait_ms': 0.0,
            'prefill_ms': 0.0,
            'ttft_ms': 0.0,
            'd4_metrics_available': False,
            'd4_metrics_source': 'request_output.metrics_missing',
        }

    def ts(name: str) -> float | None:
        value = getattr(metrics, name, None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    arrival = ts('arrival_time')
    queued = ts('queued_ts')
    scheduled = ts('first_scheduled_time')
    if scheduled is None:
        scheduled = ts('scheduled_time') or ts('scheduled_ts')
    first_token = ts('first_token_time') or ts('first_token_ts')
    time_in_queue = ts('time_in_queue')

    # 计算 queue_wait_ms
    if time_in_queue is not None:
        queue_wait_ms = max(0.0, time_in_queue * 1000.0)
    elif queued is not None and scheduled is not None:
        queue_wait_ms = max(0.0, (scheduled - queued) * 1000.0)
    elif arrival is not None and scheduled is not None:
        queue_wait_ms = max(0.0, (scheduled - arrival) * 1000.0)
    else:
        queue_wait_ms = 0.0

    # 计算 prefill_ms
    prefill_ms = (
        max(0.0, (first_token - scheduled) * 1000.0)
        if scheduled is not None and first_token is not None
        else 0.0
    )

    # 计算 TTFT（从 arrival 到 first_token）
    ttft_ms = 0.0
    if arrival is not None and first_token is not None:
        ttft_ms = max(0.0, (first_token - arrival) * 1000.0)
    elif queued is not None and first_token is not None:
        # 如果 arrival 缺失，使用 queued 作为起点
        ttft_ms = max(0.0, (first_token - queued) * 1000.0)
    elif scheduled is not None and first_token is not None:
        ttft_ms = max(0.0, (first_token - scheduled) * 1000.0)

    available = (
        (time_in_queue is not None or (queued is not None and scheduled is not None)
         or (arrival is not None and scheduled is not None))
        and scheduled is not None
        and first_token is not None
    )
    missing = [
        name for name, value in (
            ('queue_wait_source', time_in_queue if time_in_queue is not None else queued if queued is not None else arrival),
            ('first_scheduled_time', scheduled),
            ('first_token_time', first_token),
        ) if value is None
    ]
    return {
        'queue_wait_ms': queue_wait_ms,
        'prefill_ms': prefill_ms,
        'ttft_ms': ttft_ms,
        'd4_metrics_available': available,
        'd4_metrics_source': (
            'request_output.metrics'
            if available
            else f"request_output.metrics_missing:{','.join(missing)}"
        ),
    }
```

#### 修改 `run_cell` 中的请求处理部分（约第 355 行）
在 `llm.generate` 之后，对每个 output：
```python
timing = request_output_timing_ms(output)
ttft_ms = timing.get('ttft_ms', latency_ms)  # 如果 metrics 不可用，回退到 batch 墙钟
...
row.update({
    'ttft_proxy_ms': round(ttft_ms, 6),
    'queue_wait_ms': round(float(timing['queue_wait_ms']), 6),
    'prefill_ms': round(float(timing['prefill_ms']), 6),
    ...
})
```

### 🔧 确保 `sitecustomize` 补丁生效
`h1/sitecustomize.py` 中的 `patch_edgekv_vllm_request_metrics()` 会在 vLLM 导入后自动将 v1 内部统计挂载到 `RequestOutput.metrics`。该补丁通过 `sitecustomize` 机制自动加载（只要 `h1/` 在 Python 路径中）。请在运行前确认环境变量 `PYTHONPATH` 包含了 `h1`（您的脚本已设置 `PYTHONPATH=.:h1:h0`，所以通常没问题）。

### 📊 预期效果
- **单个请求的 TTFT** 将反映真实的端到端首 token 延迟（通常数百毫秒）。
- **汇总统计（p50/p95）** 将基于真实值，不再被 batch 墙钟污染。
- `queue_wait_ms` 和 `prefill_ms` 依然可以使用，但您现在会看到它们与 TTFT 的关系更加合理（例如，TTFT ≈ queue_wait + prefill）。

### 📌 关于 “每个 trace 添加 TTFT 监控”
如果您希望得到整个 trace（即整个实验运行）的总体 TTFT 分布，修改后 `h1_vllm_real_summary.csv` 中的 `ttft_proxy_p50_ms` 和 `ttft_proxy_p95_ms` 就已经是基于所有请求的独立 TTFT 计算得出的，这正好是您需要的“每个 trace 的 TTFT 监控”。

### 🚀 快速验证
应用修改后，用相同的 `--replay-batch-size 64` 重新运行，观察新的 p95 TTFT 是否落到合理区间（例如 < 2000 ms），同时 `queue_wait_p95` 和 `prefill_p95` 依然反映真实排队和预填充耗时。

如果仍有部分请求的 `metrics` 缺失（`d4_metrics_available_ratio` < 1），您可能需要进一步调试 `sitecustomize` 补丁是否真正拦截了 `RequestState._new_request_output` 调用。

如有需要，我可以提供完整的 `diff` 补丁。请告知您的运行环境是否还有其他定制需求。