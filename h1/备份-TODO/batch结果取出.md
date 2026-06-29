你的代码已经正确实现了逐请求 TTFT 的采集。`llm.generate()` 返回一个 `list`，每个元素是一个 `vllm.RequestOutput` 对象，它对应输入 `prompts` 列表中相同位置的提示词。从每个 `RequestOutput` 中提取 TTFT 的核心逻辑集中在 `request_output_timing_ms()` 函数（第 94～148 行），你可以直接复用。

---

### 如何从 `outputs` 中提取每个 trace 的 TTFT

1. **调用 `llm.generate`**  
   ```python
   outputs = llm.generate(prompts, sampling_params_list)
   # outputs 是一个列表，len(outputs) == len(prompts)
   ```

2. **逐请求提取 TTFT**  
   对每个 `output` 调用 `request_output_timing_ms(output)`，返回的字典中包含 `'ttft_ms'` 键（单位毫秒）。  
   ```python
   for idx, output in enumerate(outputs):
       timing = request_output_timing_ms(output)
       ttft_ms = timing['ttft_ms']          # float，优先使用 metrics 中的 first_token_time - arrival_time
       metrics_avail = timing['d4_metrics_available']   # bool，指示 metrics 是否完整
   ```

3. **TTFT 的计算逻辑**（代码第 130～140 行）  
   - 如果 `arrival_time` 和 `first_token_time` 都存在，则 `ttft_ms = (first_token_time - arrival_time) * 1000`。  
   - 否则尝试用 `queued_ts` 或 `time_in_queue` 等后备方案。  
   - 如果所有时间戳都缺失，则回退为 batch 级别的端到端延迟（`latency_ms`），此时 `d4_metrics_available` 为 `False`。

4. **TTFT 的最终记录**（第 639 行）  
   ```python
   ttft_proxy_ms = timing_ttft_ms if metrics_available and timing_ttft_ms > 0.0 else latency_ms
   ```
   在你的实验中，`ttft_proxy_ms` 就是每个请求实际使用的 TTFT 值，已经写入每行 CSV 的 `ttft_proxy_ms` 字段。

---

### 注意事项

- **Metrics 可用性**：`vLLM` 的 `RequestOutput.metrics` 在某些版本或配置下可能不完整。你可以通过 `timing['d4_metrics_available']` 判断当前请求是否使用了真正的逐请求计时。  
- **vLLM 版本**：你的代码已针对 vLLM 0.11.0 做了适配，包括 `first_scheduled_time`、`queued_ts` 等字段的防御性获取。  
- **Batch 级别 Fallback**：如果 metrics 缺失或 `ttft_ms <= 0`，则用整个 batch 的端到端延迟 `latency_ms` 作为代理（代码 639 行），此时同一 batch 内所有请求的 `ttft_proxy_ms` 相同。

