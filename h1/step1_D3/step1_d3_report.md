# H1 Step1 D3 Runtime Monitor

数据来源：`EDGEKV_H1_RUNTIME_MONITOR=1` 写出的真实运行时 JSONL。图中点只来自 `n_tokens>0` 且 `c_recomp>0` 的事件。

## tight
- monitor_files: h1/step1_D3/runtime/tight/prefix_128/runtime_monitor.jsonl, h1/step1_D3/runtime/tight/prefix_256/runtime_monitor.jsonl, h1/step1_D3/runtime/tight/prefix_768/runtime_monitor.jsonl, h1/step1_D3/runtime/tight/runtime_monitor.jsonl
- events: 5479, plotted_points: 4307
- lpe_action_counts: `{"admit": 1101, "evict": 270, "lookup_hit": 1822, "lookup_miss": 80, "reorder_candidate": 384, "touch": 1822}`
- hit_counts: `{"False": 1451, "True": 1822}`
- p_reuse: mean=0.924168, variance=0.0194406, std=0.13943
- score: mean=138.542, variance=26969.7, std=164.225

## mid
- monitor_files: h1/step1_D3/runtime/mid/prefix_128/runtime_monitor.jsonl, h1/step1_D3/runtime/mid/prefix_256/runtime_monitor.jsonl, h1/step1_D3/runtime/mid/prefix_768/runtime_monitor.jsonl, h1/step1_D3/runtime/mid/runtime_monitor.jsonl
- events: 4866, plotted_points: 4143
- lpe_action_counts: `{"admit": 1060, "lookup_hit": 1863, "lookup_miss": 80, "touch": 1863}`
- hit_counts: `{"False": 1140, "True": 1863}`
- p_reuse: mean=0.922694, variance=0.0199961, std=0.141407
- score: mean=132.569, variance=17633.8, std=132.792

## 计算口径
`c_recomp_ms = c_re * n_tokens`，除非请求元信息显式传入 `c_recomp_ms`。默认 `c_re` 来自 `EDGEKV_C_RE_MS_PER_TOKEN`，当前默认值为 0.12 ms/token。
`p_reuse` 由命中频率、未命中 recency 项和对象类型先验加权得到，随后 clamp 到 `[0.01, 0.99]`。
`score = p_reuse * c_recomp_ms / size_mb`；当对象尚无可用 KV size 时 score 为 0。
LPE 驱逐实际作用在 vLLM prefix cache block/free queue 上；profile 是对象级，驱逐候选和 `_maybe_evict_cached_block` 都是 block 粒度。
