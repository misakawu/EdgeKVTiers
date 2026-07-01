
## 概述

`h0/` 目录包含 H0 阶段的实验工具，主要用于：

- 构建 ShareGPT 和 HotpotQA 的混合 replay trace（`build_h0_replay_trace.py`）
- 在真实的 vLLM 服务上回放这些 trace，测量 TTFT、命中率、GPU 显存峰值（`run_h0_vllm.py`）
- 运行一个离线的异构性分析探针，验证 LPE 分数在 homogeneous/heterogeneous 对象上的行为（`run_h0_hetero_probe.py`）
- 提供运行时补丁（`sitecustomize.py`）和单元测试

---

## 文件说明

### `build_h0_replay_trace.py`

**功能**：将 ShareGPT 对话和 HotpotQA 检索语料混合，生成一个冻结的 JSONL 格式 replay trace，供后续实验重复使用。

**主要参数**（通过命令行传入）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--trace-path` | ShareGPT JSON 文件路径 | `data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json` |
| `--hotpotqa-path` | HotpotQA 数据目录或文件 | `data/hotpotqa` |
| `--download-hotpotqa` | 若 HotpotQA 文件缺失则从 HuggingFace 下载 | 不启用 |
| `--out` | 输出 JSONL 文件路径 | `data/edgekv_traces/sharegpt_hotpotqa_session.jsonl` |
| `--workload` | 选择 `sharegpt`, `rag`, `mixed` | `mixed` |
| `--max-sessions` | 最大 ShareGPT 对话数 | `2000` |
| `--max-requests` | 最大总请求数 | `10000` |
| `--rag-requests` | RAG 请求数 | `500` |
| `--hotpotqa-max-examples` | HotpotQA 最大样例数 | `50` |
| `--rag-chunk-words` | 每个 RAG chunk 的单词数 | `56` |
| `--rag-chunks-per-query` | 每查询 chunk 数 | `2` |
| `--rag-query-repeats` | 每个 RAG 查询重复次数 | `4` |
| `--sharegpt-order` | ShareGPT 排序方式 `file` / `longest` | `file` |
| `--link-mode` | ShareGPT 与 RAG 链接方式 `independent` / `weak` | `independent` |
| `--weak-rag-repeat` | weak 模式下每个 RAG 单元重复次数 | `2` |
| `--timeout-s` | 下载超时（秒） | `120.0` |

**用法示例**：

```bash
python h0/build_h0_replay_trace.py --workload mixed --max-requests 512 --out my_trace.jsonl
```

---

### `run_h0_vllm.py`

**功能**：核心回放脚本，向正在运行的 vLLM 服务器（需启用 `--enable-prefix-caching`）发送请求，记录 TTFT、端到端延迟、GPU 显存峰值，并收集 CO Profiler 统计信息。

**主要参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--endpoint` | vLLM 服务器地址 | `http://127.0.0.1:8000` |
| `--model` | 模型名称（自动从 `/v1/models` 获取） | 空 |
| `--trace-path` | ShareGPT 原始 JSON 路径 | 同 `DEFAULT_SHAREGPT_TRACE_PATH` |
| `--replay-trace` | 预生成的重放 trace（JSONL） | 同 `DEFAULT_REPLAY_TRACE_PATH` |
| `--workload` | `sharegpt`, `rag`, `mixed` | `mixed` |
| `--max-requests` | 最大请求数 | `40` |
| `--max-sessions` | ShareGPT 最大会话数 | `20` |
| `--rag-requests` | RAG 请求数 | `16` |
| `--max-tokens` | 生成最大 token 数 | `16` |
| `--concurrency` | 并发度 | `1` |
| `--tensor-parallel-size` | TP 大小 | `1` |
| `--out` | 输出目录 | `h0/out/h0_vllm_prefix_cache` |
| `--c-re-ms-per-token` | 每 token 重计算时间（ms） | `0.12` |
| `--bw-gbps` | 带宽（GB/s） | `1.0` |
| `--d-deser-ms` | 反序列化延迟（ms） | `3.0` |

**用法示例**：

```bash
# 先启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf --enable-prefix-caching

# 在另一个终端运行回放
python h0/run_h0_vllm.py --endpoint http://127.0.0.1:8000 --max-requests 100 --concurrency 4 --out ./results
```

---

### `run_h0_hetero_probe.py`

**功能**：离线分析性探测器，用于验证 LPE 分数在 homogeneous / heterogeneous 对象集合上的行为，生成摘要 CSV、事件 JSONL 和可视化图表，并输出结论（`conclusion.md`）。

**主要参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--out-dir` | 输出目录 | `h0/results/h0_hetero_probe` |
| `--seed` | 随机种子 | `20260701` |
| `--reps` | 重复次数 | `3` |
| `--trace-len` | 模拟 trace 长度 | `1800` |
| `--budgets` | 缓存预算分数列表 | `0.25 0.35 0.45 0.60` |

**用法**：

```bash
python h0/run_h0_hetero_probe.py --budgets 0.3 0.5 0.7
```

输出目录包含 `summary.csv`, `events.jsonl`, 各场景的 png 图以及 `conclusion.md`。

---

### `sitecustomize.py`

为 vLLM 的某些版本（如 `transformers` 缺少 `all_special_tokens_extended` 属性）提供运行时兼容性补丁，在导入时自动生效。

---

### 测试文件

- `test_h0_hetero_probe.py`：测试异质探针的基本功能
- `test_h0_rag_trace.py`：测试 RAG trace 构建逻辑
- `test_vllm_offload_policies.py`：旧版 CPU 卸载策略测试（默认跳过，需设置环境变量 `EDGEKV_ENABLE_LEGACY_CPU_OFFLOAD_TESTS=1`）

---