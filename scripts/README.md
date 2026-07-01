
## 概述

`scripts/` 目录包含用于生成和管理实验 trace 的脚本，以及一些专用的聚合工具。

这些脚本独立于 h0/h1 实验框架，用于创建具有可控重用模式的 JSONL trace，供回放实验使用。

---

## 文件说明

### `optimize_h0_pressure_trace.py`

**功能**：生成一个预算敏感的压力 trace，混合 ShareGPT 和 HotpotQA 对象。它通过控制 hot/cold 对象的重复访问和排序，使得不同 KV 缓存预算下命中率有明显阶梯变化，用于测试缓存策略的预算敏感性。

**主要参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sharegpt-path` | ShareGPT JSON 路径 | `data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json` |
| `--hotpotqa-path` | HotpotQA 路径 | `data/hotpotqa` |
| `--out` | 输出 JSONL 路径 | `data/edgekv_traces/sharegpt_hotpotqa_session.jsonl` |
| `--source-mode` | 数据源 `mixed` / `sharegpt` / `hotpotqa` | `mixed` |
| `--sharegpt-groups` | ShareGPT 对象数 | `96` |
| `--hot-ratio` | hot 对象比例 | `0.20` |
| `--hot-repeats` | hot 对象重复次数 | `4` |
| `--hot-context-words` | hot 上下文单词数 | `300` |
| `--cold-context-words` | cold 上下文单词数 | `800` |
| `--rag-requests` | RAG 请求数（仅 mixed） | `128` |
| `--scan-hot-objects` | 预算阶梯中 hot 对象数 | `16` |
| `--scan-cold-objects` | 每轮 cold 对象数 | `24` |
| `--scan-probe-rounds` | 探针轮数 | `3` |
| `--reuse-schedule` | 调度方式 `budget_ladder` / `segmented_ladder` / `original_sharegpt_random_rag` | `budget_ladder` |
| `--ladder-segments` | 分段数（`segmented_ladder` 模式） | `3` |
| `--total-requests` | 总请求数（`original_sharegpt_random_rag` 模式） | `1024` |
| `--rag-share` | RAG 请求占比 | `0.20` |
| `--sharegpt-order` | ShareGPT 排序 `file` / `longest` | `file` |
| `--sharegpt-human-merge` | 合并方式 `first` / `all` / `session` | `first` |
| `--random-seed` | 随机种子 | `2026` |
| `--download-hotpotqa` | 下载 HotpotQA | 否 |

**用法**：

```bash
# 生成混合压力 trace（budget_ladder 模式）
python scripts/optimize_h0_pressure_trace.py --out my_trace.jsonl --sharegpt-groups 50 --rag-requests 80

# 纯 ShareGPT 随机插入 RAG 模式（original_sharegpt_random_rag）
python scripts/optimize_h0_pressure_trace.py --reuse-schedule original_sharegpt_random_rag --total-requests 512 --rag-share 0.15
```

---

### `hotqa_jsonl_generation.py`

**功能**：专门生成 HotpotQA 的 prefix-cache 友好 trace。采用 **HOT + WARM + TAIL** 三层结构：

- HOT：固定不变的全局前缀（`global-pool-size` 个 chunk）
- WARM：从 `warm-pool-size` 个候选 chunk 中按 Zipf 分布抽取一个
- TAIL：每请求唯一短尾

此结构确保共享 HOT 前缀的重用，且热门 WARM 前缀重复出现，产生清晰的分层命中率曲线。

**主要参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--out` | 输出 JSONL | `data/edgekv_traces/source_ablation/hotqa.jsonl` |
| `--requests` / `--num-prompts` | 请求数 | `768` |
| `--random-seed` | 随机种子 | `2026` |
| `--hotpotqa-path` | HotpotQA 源数据 | `data/hotpotqa` |
| `--download-hotpotqa` | 下载 | 否 |
| `--global-pool-size` | HOT 池大小 | `1` |
| `--warm-pool-size` | WARM 池大小 | `300` |
| `--zipf-s` | Zipf 参数 | `0.8` |
| `--tail-words` | 尾部的单词数 | `8` |
| `--chunk-words` | 每个 chunk 的单词数 | `240` |
| `--min-chunk-words` | 最小 chunk 单词数 | `240` |
| `--hotpotqa-max-examples` | 最大样例数 | `200` |

**用法**：

```bash
python scripts/hotqa_jsonl_generation.py --requests 1024 --global-pool-size 2 --warm-pool-size 200 --zipf-s 1.0
```

---

### `sharedgpt_jsonl_generation.py`

**功能**：与 `hotqa_jsonl_generation.py` 类似，但数据源为 ShareGPT。将长对话分段成 chunk，同样采用 **HOT + WARM + TAIL** 结构。

**主要参数**（与 hotqa 版本类似）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--out` | 输出路径 | `data/edgekv_traces/source_ablation/sharedgpt.jsonl` |
| `--requests` | 请求数 | `1536` |
| `--sharegpt-path` | ShareGPT JSON | 默认 |
| `--hot-pool-size` | HOT 池大小 | `1` |
| `--warm-pool-size` | WARM 池大小 | `300` |
| `--zipf-s` | Zipf 参数 | `0.8` |
| `--tail-words` | 尾部单词数 | `8` |
| `--chunk-words` | 每个 chunk 单词数 | `240` |
| `--min-chunk-words` | 最小 chunk 数 | `240` |
| `--max-source-sessions` | 最大源会话数 | `4096` |

**用法**：

```bash
python scripts/sharedgpt_jsonl_generation.py --requests 2048 --warm-pool-size 400 --zipf-s 0.9
```

---

### `aggregate_ws2_policy_cmp.py`

**功能**：专用聚合脚本，用于分析 `hotqa三级trace_有效窗口/policy_cmp` 下的策略比较结果。读取各策略、各预算、各 rep 的 summary JSON，计算均值±标准差，输出 CSV 并绘制命中率和 p95 延迟的柱状图。

**用法**：

```bash
python scripts/aggregate_ws2_policy_cmp.py
```

输出：

- `ws2_policy_cmp_summary.csv`
- `ws2_hit_rate.png`
- `ws2_latency_p95.png`

无需参数，结果路径硬编码为 `h1/out/hotqa三级trace_有效窗口/policy_cmp`。

---

### 其他脚本

- `README.md`：本文件（原有，现被覆盖）
- 无其他脚本

---

## 环境依赖

所有脚本均需在 `edgekv-vllm0110` conda 环境下运行（或具有相同依赖）。运行前请激活环境：

```bash
conda activate edgekv-vllm0110
```

某些脚本（如 `optimize_h0_pressure_trace.py`）会调用 `h0/run_h0_vllm.py` 中的函数，因此需确保 `h0/` 在 `PYTHONPATH` 中。通常顶层 `run_*.py` 已处理好导入路径。