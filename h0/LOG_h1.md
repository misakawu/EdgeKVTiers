# H0 实验日志

## 2026-07-01：H0 异质探针验证 LPE score 唯一出路

### 实验目的

本次实验不使用 vLLM、不占 GPU，只用解析仿真器验证 `score = p_reuse * c_recomp / size` 在同质 KV 对象下是否退化为只看 `p_reuse`，以及引入异质对象后 LPE-score 是否能稳定优于 LRU。

### 实现与验证

- 新增脚本：`h0/run_h0_hetero_probe.py`
- 新增测试：`h0/test_h0_hetero_probe.py`
- 输出目录：`h0/results/h0_hetero_probe/20260701_081410`
- 运行命令：`python3 h0/run_h0_hetero_probe.py --out-dir h0/results/h0_hetero_probe --seed 20260701 --reps 3`
- 验证命令：
  - `python3 -m py_compile h0/run_h0_hetero_probe.py h0/test_h0_hetero_probe.py`
  - `python3 h0/test_h0_hetero_probe.py`

### 场景与策略

- 策略：`LRU`、`LFU`、`LPE-score`
- Budget：`0.25`、`0.35`、`0.45`、`0.60`
- Reps：`3`
- 同质场景：所有对象 `size_factor=1.0`、`recompute_factor=1.0`
- 异质场景：
  - `fp16_session_kv`：`size_factor=1.0`、`recompute_factor=1.0`
  - `int4_quant_block`：`size_factor=0.25`、`recompute_factor=0.9`
  - `rag_chunk`：按短/中/长分档设置 size 与 recompute factor

### 关键结果

同质场景中 LPE-score 与 LRU 的 `p95_cost` 接近但略差，说明当 `c_recomp/size` 基本为常数时，score 没有提供额外区分度。

| budget | LRU p95_cost | LPE p95_cost | LPE-LRU |
| ---: | ---: | ---: | ---: |
| 0.25 | 14.877060 | 14.960308 | 0.083247 |
| 0.35 | 14.689199 | 14.787109 | 0.097911 |
| 0.45 | 14.483095 | 14.554274 | 0.071178 |
| 0.60 | 14.094985 | 14.156611 | 0.061626 |

异质场景中 4 个 budget 都落在 `hit_rate 0.5~0.85` 有效窗口，且 LPE-score 的 `p95_cost` 均低于 LRU。

| budget | LRU hit_rate | LRU p95_cost | LPE p95_cost | LRU-LPE |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.615556 | 12.354928 | 11.508375 | 0.846553 |
| 0.35 | 0.695370 | 11.675097 | 11.172170 | 0.502927 |
| 0.45 | 0.760926 | 11.387728 | 11.077018 | 0.310710 |
| 0.60 | 0.823519 | 11.051629 | 10.804274 | 0.247355 |

### 保留文件

- `summary.csv`：`scenario,budget,policy,rep,...` 汇总指标
- `events.jsonl`：逐访问事件，包含 `p_reuse`、`score`、`hit`、`action`、`evicted`
- `homogeneous_p95_cost.png` / `homogeneous_hit_rate.png`
- `heterogeneous_p95_cost.png` / `heterogeneous_hit_rate.png`
- `homogeneous_score_vs_p_reuse.png` / `heterogeneous_score_vs_p_reuse.png`
- `conclusion.md`：一页结论

### 结论

本次探针复现了同质 KV 下 LPE-score 的退化：`score` 排序等价于 `p_reuse` 排序，无法稳定优于 LRU。异质对象加入后，尤其是 `int4_quant_block` 让 `c_recomp/size` 高于 fp16 基线，LPE-score 在有效窗口内 4 个 budget 档均优于 LRU。结论是：LPE 的 score 需要异质对象才有意义，后续应把证据链转向 H4 的质量/量化维度。
