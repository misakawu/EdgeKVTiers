# H0 实验日志

## 2026-06-19：H0 vLLM Prefix Cache 回放实验

### 实验目的

本次实验使用 vLLM 真实推理服务，在开启 prefix caching 的条件下回放两份固定 JSONL trace，比较独立压力回放和弱关联 RAG 回放的延迟、trace-side 命中率、显存峰值与超长请求跳过情况。

### 服务端设置

- Conda 环境：`h3-lmcache-blog`
- vLLM 版本：`0.8.5.post1`
- transformers 版本：`5.11.0`
- 启动命令前缀：`CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=h0 conda run -n h3-lmcache-blog python -m vllm.entrypoints.openai.api_server`
- 模型：`/DATACENTER3/zhenxiang.wang/work/EdgeKVTiers/models/Qwen2.5-7B-Instruct`
- 服务地址：`http://127.0.0.1:8000`
- 使用 GPU：`0,1`
- Tensor parallel size：`2`
- dtype：`half`
- `max_model_len`：`2048`
- `max_num_seqs`：`8`
- `max_num_batched_tokens`：`4096`
- `gpu_memory_utilization`：`0.9`
- Prefix caching：开启

兼容性说明：vLLM 服务端启动时必须带 `PYTHONPATH=h0`，以加载 `h0/sitecustomize.py`。该文件修补了 vLLM `0.8.5.post1` 在 transformers `5.11.0` 下访问 `Qwen2Tokenizer.all_special_tokens_extended` 的兼容性问题。

### 回放设置

- 主 workload：`mixed`
- ShareGPT 源文件：`/DATACENTER3/zhenxiang.wang/data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json`
- HotpotQA 源目录：`/DATACENTER3/zhenxiang.wang/data/hotpotqa`
- `max_sessions`：`200`
- `max_requests`：`1024`
- `rag_requests`：`100`
- `hotpotqa_max_examples`：`5`
- `rag_chunk_words`：`56`
- `rag_chunks_per_query`：`2`
- `rag_query_repeats`：`4`
- `sharegpt_order`：`longest`
- 每请求最大生成 token 数：`16`
- 并发数：`4`
- 预热请求数：`2`，统计指标中排除
- tokenizer 计数来源：`transformers_auto`
- 命中率来源：trace 侧 `reuse_key` 推断，不是 vLLM 每请求内部真实 hit flag

### 回放输入

1. 独立压力 trace
   - 输入文件：`/DATACENTER3/zhenxiang.wang/data/edgekv_traces/h0_sharegpt_hotpotqa_200sessions_pressure.jsonl`
   - 结构：ShareGPT session prefix 与 HotpotQA RAG chunk-set 独立交错。
   - 输出目录：`h0/results/h0_vllm_prefix_cache_qwen25_7b_pressure`

2. 弱关联 trace
   - 输入文件：`/DATACENTER3/zhenxiang.wang/data/edgekv_traces/h0_sharegpt_hotpotqa_200sessions_weak_link.jsonl`
   - 结构：将 HotpotQA RAG context 以弱关联方式挂到 ShareGPT turn 前缀中。
   - 输出目录：`h0/results/h0_vllm_prefix_cache_qwen25_7b_weak_link`

### 汇总结果

| Trace | 总请求数 | 实际发送 | 超长跳过 | 有效统计请求 | 成功率 | Trace-side hit rate | TTFT p50 ms | TTFT p95 ms | Latency p50 ms | Latency p95 ms | GPU 显存峰值 MiB | 总耗时 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pressure` | 1010 | 1007 | 3 | 1005 | 1.0 | 0.486567 | 226.087760 | 820.294527 | 449.599873 | 1103.559360 | 11007.0 | 135.973456 |
| `weak_link` | 910 | 891 | 19 | 889 | 1.0 | 0.506187 | 239.334509 | 905.244579 | 429.405779 | 1206.307560 | 11007.0 | 131.660064 |

### Workload 细分

#### Pressure trace

- `rag_chunk_reuse`
  - 有效统计请求数：`100`
  - hit rate：`0.31`
- `sharegpt_session_prefix`
  - 有效统计请求数：`905`
  - hit rate：`0.506077`
- 事件行统计：
  - `events.jsonl` 总行数：`1010`
  - `ok`：`1007`
  - `skip_overlength`：`3`
  - 错误：`0`
  - `miss`：`518`
  - `hit`：`489`

#### Weak-link trace

- `sharegpt_hotpotqa_weak_link`
  - 有效统计请求数：`82`
  - hit rate：`0.243902`
- `sharegpt_session_prefix`
  - 有效统计请求数：`807`
  - hit rate：`0.532838`
- 事件行统计：
  - `events.jsonl` 总行数：`910`
  - `ok`：`891`
  - `skip_overlength`：`19`
  - 错误：`0`
  - `miss`：`441`
  - `hit`：`450`

### 保留文件

本次只保留两组实验结果，均位于 `h0/results/` 下：

- `h0/results/h0_vllm_prefix_cache_qwen25_7b_pressure`
- `h0/results/h0_vllm_prefix_cache_qwen25_7b_weak_link`

每个结果目录包含：

- `events.jsonl`：逐请求事件、命中推断、延迟和错误信息
- `gpu_memory_samples.jsonl`：GPU 显存采样
- `summary.csv`：单行汇总指标
- `config.resolved.json`：运行参数和汇总结果

已清理 `h0/out` 下旧实验输出，避免混淆本次结果。
