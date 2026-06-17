# H0 vLLM Prefix Caching Replay - Qwen2.5-7B TP=2 max_model_len=1024

This output follows the H0 requirements in `论文规划/00_王祯祥_论文工作规划.md`.

## Run configuration

- Model: `models/Qwen2.5-7B-Instruct`
- vLLM: `0.6.6.post1`
- GPUs: RTX 2080 Ti, `CUDA_VISIBLE_DEVICES=0,1`
- Tensor parallel size: `2`
- vLLM flags: `--enable-prefix-caching --max-model-len 1024 --dtype half --gpu-memory-utilization 0.95`
- Replay: ShareGPT, `200` sessions, `max_tokens=8`

## Outputs

- `events.jsonl`: per-request H0 trace with `event`, `hit`, `n_tokens`, `size_mb`, `t_policy_ms`, TTFT, latency, and skip/error fields.
- `summary.csv`: p50/p95 TTFT, hit rate, GPU memory peak, success rate, and replay counts.
- `gpu_memory_samples.jsonl`: sampled `nvidia-smi` memory/utilization records.
- `config.resolved.json`: resolved runner arguments and summary.
- `vllm_server.log`: vLLM server startup and runtime log.

## Important field notes

- `hit` is a trace-side prefix reuse inference: later turns in the same ShareGPT session are marked as reusable prefix hits. vLLM 0.6.6 OpenAI responses do not expose exact per-request prefix-cache hit flags in this environment.
- `n_tokens` is counted with the local Qwen tokenizer when available.
- `size_mb` is a per-GPU logical full-KV estimate from model config and TP=2. It is not a direct allocator measurement.
- `t_policy_ms` is runner-side bookkeeping time. H0 uses vLLM default prefix caching, so no external strategy policy is applied.
- Requests whose `n_tokens + max_tokens > 1024` are written as `event=skip_overlength` instead of being sent to vLLM.

## Summary

- `sessions_requested`: 200
- `requests_total`: 894
- `requests_attempted`: 557
- `requests_skipped_overlength`: 337
- `requests_measured`: 552
- `success_rate`: 1.0
- `hit_rate`: 0.653986
- `ttft_p95_ms`: 212.071688
- `gpu_memory_peak_mib`: 11007.0
