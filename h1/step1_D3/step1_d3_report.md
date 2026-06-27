# H1 Step1 D3 复盘：LPE 真实运行时监控

## 结论
- `c_recomp` 在实际代码中是对象 profile 字段 `c_recomp_ms`，默认按 `c_re * n_tokens` 线性计算。
- KVCache 的 LPE 驱逐执行粒度是 vLLM prefix-cache block，不是对象整体。对象级 profile 只提供 `p_reuse/c_recomp/score` 给 block 排序和诊断。
- `score = p_reuse * c_recomp_ms / size_mb`；`p_reuse` 由命中频率、miss recency 和对象类型先验加权得到。
- 图和 summary 使用 `h1/step1_D3/runtime/{tight,mid}/**/runtime_monitor.jsonl` 的真实运行时事件，不从聚合 stats 复制散点。

## 代码依据：c_recomp 如何计算
位置：`h1/sitecustomize.py`，对象 profile 更新逻辑，当前约第 715 行。

`c_recomp` 实际写出的值就是 `profile['c_recomp_ms']`。如果请求 meta 没有显式 `c_recomp_ms`，代码按 `c_re * n_tokens` 计算。

```python
c_re = _edgekv_env_float('EDGEKV_C_RE_MS_PER_TOKEN', 0.12)
n_tokens = max(int(n_tokens), 1)
c_recomp_ms = float(meta.get('c_recomp_ms', 0.0) or 0.0) or (c_re * n_tokens)
profile.update({'n_tokens': n_tokens, 'c_recomp_ms': c_recomp_ms})
```

## 代码依据：p_reuse 与 score 如何计算
位置：`h1/sitecustomize.py`，对象 profile 评分逻辑，当前约第 760-792 行。

`p_reuse` 是命中比例、miss recency 项、对象类型先验三者加权；默认权重为 `0.55/0.30/0.15`。`score` 在对象有 resident size 后按收益/容量计算。

```python
p_freq = _edgekv_clamp(hit_count / access_count, 0.0, 1.0)
p_recency = 1.0 / (1.0 + math.log1p(misses))
p_type = _edgekv_object_type_prior(str(profile.get('object_type', 'unknown')))
p_reuse = ((w_freq * p_freq) + (w_recency * p_recency) + (w_type * p_type)) / weight_sum
profile['p_reuse'] = _edgekv_clamp(p_reuse, 0.01, 0.99)
profile['score'] = profile['p_reuse'] * profile['c_recomp_ms'] / max(size_mb, 1e-9)
```

## 代码依据：LPE 按 block 驱逐
位置：`h1/sitecustomize.py`，free queue 候选记录当前约第 1163 行；`_maybe_evict_cached_block()` hook 当前约第 1398 行。

驱逐候选从 free queue/rank heap 中选出的是 `block`，真正 evict hook 是 `_maybe_evict_cached_block(self, block)`。因此执行粒度是 block；对象 profile 通过 block 映射提供分数。

```python
block = _edgekv_heap_valid_block(pool, item)
block_id = _edgekv_block_id(block)
_edgekv_record_lpe_monitor('reorder_candidate', block_id=block_id, profile=_edgekv_block_profile(pool, block_id))

def _maybe_evict_cached_block(self, block):
    evicted = original_maybe_evict_cached_block(self, block)
    block_id = _edgekv_block_id(block)
    profile = _edgekv_block_profile(self, block_id, kv_cache_group_id)
    _edgekv_record_lpe_monitor('evict', block_id=block_id, profile=profile, hit=False, evicted=True)
```

## 输出文件
- 图片：`h1/step1_D3/out/step1_D3_c_recomp_vs_n.png`
- 图片：`h1/step1_D3/out/step1_D3_score_histogram.png`
- 图片：`h1/step1_D3/out/step1_D3_p_reuse_histogram.png`
- summary：`h1/step1_D3/out/step1_D3_summary.json`
- runtime 原始数据：`h1/step1_D3/runtime/{tight,mid}/**/runtime_monitor.jsonl`

### 三张图片解释

1. step1_D3_c_recomp_vs_n.png

  这个图显示 c_recomp_ms 和对象 token 数 n_tokens 基本是严格线性关系：

  - tight: c_re=0.12，c_recomp_ms p50=61.44，对应 512 tokens * 0.12
  - mid: 同样 c_re=0.12，c_recomp_ms p50=61.44
  - p95 都是 92.16，对应 768 tokens * 0.12

  含义：当前代码里的重算成本不是实测出来的复杂代价，而是 0.12 ms/token * n_tokens。所以 c_recomp 本身没有揭示“某些对象特别难重算”的额外信息，只是在表达对象长度。

2. step1_D3_p_reuse_histogram.png

  p_reuse 分布高度集中：

  - tight: mean 0.924，p50 0.97，p95 0.97
  - mid: mean 0.923，p50 0.97，p95 0.97
  - min 都是 0.5

  含义：大多数 runtime 事件的 p_reuse 已经饱和到接近 0.97。这说明 LPE 的复用概率估计在这个 workload 下区分度不强：很多对象都被判成“高复用”，排序时很难靠 p_reuse 拉开差距。

  这也解释了为什么 LPE 可能退化成类似 LRU/LFU 的行为：如果 p_reuse 大面积饱和，策略主要只是在已有热点对象之间做很粗的排序。

3. step1_D3_score_histogram.png

  score 分布是“主体集中 + 少量长尾”：

  - tight: score p50 119.19，p95 152.57，max 3814.20
  - mid: score p50 119.19，p95 122.88，max 2949.12

  含义：大多数对象的 score 其实也挤在一个较窄区间，说明有效排序信号有限；但有少量极高 score outlier。这些 outlier 通常不是因为 p_reuse 更准，而是因为公式分母 size_mb 较小，导致 p_reuse * c_recomp / size_mb 被放大。

综合看，三张图支持这个结论：当前 LPE 的 c_recomp 主要是长度线性函数，p_reuse 又高度饱和，最终 score 对大多数对象区分度有限，只在少量小resident size 对象上产生长尾高分。这就是 D3 所说“退化需要被坐实”的主要证据。


## 三行验收日志
- `c_recomp/n_tokens`: 见 `c_recomp_ms` 与 `c_re` 统计，以及 `step1_D3_c_recomp_vs_n.png`。
- `eviction_granularity`: LPE 执行粒度为 `vLLM prefix-cache block`，runtime 事件包含 `block_id/block_start/block_end`。
- `score/p_reuse histogram`: 见 `step1_D3_score_histogram.png` 与 `step1_D3_p_reuse_histogram.png`。

## 真实运行数据
### tight
- runtime monitor files: 4
- events=5479, plotted_points=4307
- n_tokens: count=4307, mean=553.369, variance=37181.1, std=192.824, min=128, p50=512, p95=768, max=768
- c_recomp_ms: count=4307, mean=66.4043, variance=535.408, std=23.1389, min=15.36, p50=61.44, p95=92.16, max=92.16
- c_re: count=4307, mean=0.12, variance=0, std=0, min=0.12, p50=0.12, p95=0.12, max=0.12
- p_reuse: count=4307, mean=0.924168, variance=0.0194406, std=0.13943, min=0.5, p50=0.97, p95=0.97, max=0.97
- score: count=4307, mean=138.542, variance=26969.7, std=164.225, min=61.44, p50=119.194, p95=152.568, max=3814.2
- lpe_action_counts: `{"admit": 1101, "evict": 270, "lookup_hit": 1822, "lookup_miss": 80, "reorder_candidate": 384, "touch": 1822}`
- hit_counts: `{"False": 1451, "True": 1822}`

runtime files:
- `h1/step1_D3/runtime/tight/prefix_128/runtime_monitor.jsonl`
- `h1/step1_D3/runtime/tight/prefix_256/runtime_monitor.jsonl`
- `h1/step1_D3/runtime/tight/prefix_768/runtime_monitor.jsonl`
- `h1/step1_D3/runtime/tight/runtime_monitor.jsonl`

### mid
- runtime monitor files: 4
- events=4866, plotted_points=4143
- n_tokens: count=4143, mean=549.075, variance=37539, std=193.75, min=128, p50=512, p95=768, max=768
- c_recomp_ms: count=4143, mean=65.889, variance=540.561, std=23.25, min=15.36, p50=61.44, p95=92.16, max=92.16
- c_re: count=4143, mean=0.12, variance=0, std=0, min=0.12, p50=0.12, p95=0.12, max=0.12
- p_reuse: count=4143, mean=0.922694, variance=0.0199961, std=0.141407, min=0.5, p50=0.97, p95=0.97, max=0.97
- score: count=4143, mean=132.569, variance=17633.8, std=132.792, min=61.44, p50=119.194, p95=122.88, max=2949.12
- lpe_action_counts: `{"admit": 1060, "lookup_hit": 1863, "lookup_miss": 80, "touch": 1863}`
- hit_counts: `{"False": 1140, "True": 1863}`

runtime files:
- `h1/step1_D3/runtime/mid/prefix_128/runtime_monitor.jsonl`
- `h1/step1_D3/runtime/mid/prefix_256/runtime_monitor.jsonl`
- `h1/step1_D3/runtime/mid/prefix_768/runtime_monitor.jsonl`
- `h1/step1_D3/runtime/mid/runtime_monitor.jsonl`

## 代码依据：runtime 监控字段如何写出
位置：`h1/sitecustomize.py`，函数 `_edgekv_record_lpe_monitor()`，当前约第 231 行。

`_edgekv_record_lpe_monitor()` 在 LPE 策略开启且设置 `EDGEKV_H1_RUNTIME_MONITOR_PATH` 后写 JSONL。`n_tokens/c_recomp/p_reuse/score` 来自真实运行中的 object profile；`lpe_action/hit/block_id` 来自当前 hook 事件。

```python
c_re = _edgekv_env_float('EDGEKV_C_RE_MS_PER_TOKEN', 0.12)
record = {'lpe_action': str(lpe_action), 'hit': hit, 'block_id': block_id, 'c_re': c_re}
if profile is not None:
    c_recomp_ms = float(profile.get('c_recomp_ms', 0.0) or 0.0)
    record.update({'n_tokens': int(profile.get('n_tokens', 0) or 0), 'c_recomp': c_recomp_ms, 'p_reuse': float(profile.get('p_reuse', 0.0) or 0.0), 'score': float(profile.get('score', 0.0) or 0.0)})
```


## 生成脚本口径
位置：`h1/step1_D3/build_step1_d3.py`，`collect_budget()` 中 `real_points` 过滤逻辑，当前约第 100 行。

`build_step1_d3.py` 只保留 `n_tokens>0` 且 `c_recomp_ms/c_recomp>0` 的真实事件点；没有真实事件时默认报错。

```python
real_points = [p for row in events if f(row, 'n_tokens') > 0.0 and f(row, 'c_recomp_ms', f(row, 'c_recomp', 0.0)) > 0.0]
```
