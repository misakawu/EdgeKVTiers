# H1 Step1 D3 复盘：对象级 COP 监控（路径 A）

## 结论
- `c_recomp / p_reuse / score` 由 **COP 模块 `edgekv_cop.py`** 计算（`COPProfiler.update_from_item` → `ObjectProfile.recompute` / `estimate_reuse`），即设计 §6.2 Algorithm 1，对象粒度。
- 数据来源是 `run_h1_vllm0110_real.py` 写出的 per-request 行，`score_source == object_level_cop`，不再从 sitecustomize 的 block 级内联 profile 取值。
- `c_recomp = c_re * n_tokens`（线性）；`score = p_reuse * c_recomp / size_mb`，其中 `size_mb = μ_kv * n_tokens` → `score = p_reuse * (c_re/μ_kv)`，n 被约掉，**score 与 p_reuse 同序**（坐实 §2 退化）。
- `p_reuse` 由 COP 的访问历史估计（近期命中率/频次/recency 加权 + 可选先验），在真实混合 trace 下有显著方差，因此本轮 D3 的退化结论靠 p_reuse 的真实分布坐实，而非饱和值。
- 引擎实际驱逐粒度仍是 vLLM prefix-cache block（见各档 `eviction_granularity`）；COP 只在对象级提供 `p_reuse/c_recomp/score` 画像与打分。
- COP 画像仅由 trace 回放顺序（每请求 `update_from_item`，hit=trace 复用）决定，与 GPU 显存预算无关，因此 tight/mid 两档的 `c_recomp/p_reuse/score` 分布一致；预算只改变引擎侧 summary（hit_rate/evictions）。

## 输出文件
- 图片：`h1/step1_D3/out/step1_D3_c_recomp_vs_n.png`
- 图片：`h1/step1_D3/out/step1_D3_score_histogram.png`
- 图片：`h1/step1_D3/out/step1_D3_p_reuse_histogram.png`
- summary：`h1/step1_D3/out/step1_D3_summary.json`
- COP 原始数据：`h1/step1_D3/runtime/{tight,mid}/**/*_h1_lpe_requests.csv`

## 三行验收日志
- `c_recomp/n_tokens`: 见 `c_recomp_ms` 与 `c_re` 统计，以及 `step1_D3_c_recomp_vs_n.png`（COP `ObjectProfile.recompute`）。
- `eviction_granularity`: 引擎驱逐为 `vLLM prefix-cache block`（各档 run summary 字段）；COP 画像为对象级。
- `score/p_reuse histogram`: 见 `step1_D3_score_histogram.png` 与 `step1_D3_p_reuse_histogram.png`（COP `estimate_reuse` + `score`）。

## 真实运行数据
### tight (gpu_memory_utilization=0.720)
- cop requests csv files: 1
- cop_events=128, plotted_points=128
- n_tokens: count=128, mean=531.305, variance=62657.5, std=250.315, min=112, p50=432, p95=901, max=1139
- c_recomp_ms: count=128, mean=63.7566, variance=902.267, std=30.0378, min=13.44, p50=51.84, p95=108.12, max=136.68
- c_re (=c_recomp/n): count=128, mean=0.12, variance=4.47511e-35, std=6.68962e-18, min=0.12, p50=0.12, p95=0.12, max=0.12
- p_reuse: count=128, mean=0.328343, variance=0.128809, std=0.3589, min=0.052718, p50=0.052718, p95=0.862718, max=0.862718
- score: count=128, mean=1.44096, variance=2.4808, std=1.57506, min=0.231359, p50=0.231359, p95=3.7861, max=3.7861
- size_mb: count=128, mean=14.5279, variance=46.8478, std=6.84454, min=3.0625, p50=11.8125, p95=24.6367, max=31.1445
- object_type_counts: `{"rag_chunk_set": 29, "sharegpt_cold_context": 60, "sharegpt_hot_context": 39}`
- hit_counts: `{"False": 96, "True": 32}`
- run summary: c_recomp_model=`linear_c_re_ms_per_token_times_n_tokens`, eviction_granularity=`vllm_prefix_cache_block`, cop_object_profile_count=96, cop_avg_p_reuse=0.187718, cop_avg_score=0.823815972, hit_rate=0.685504

cop csv files:
- `h1/step1_D3/runtime/tight/tight_h1_lpe_requests.csv`

### mid (gpu_memory_utilization=0.735)
- cop requests csv files: 1
- cop_events=128, plotted_points=128
- n_tokens: count=128, mean=531.305, variance=62657.5, std=250.315, min=112, p50=432, p95=901, max=1139
- c_recomp_ms: count=128, mean=63.7566, variance=902.267, std=30.0378, min=13.44, p50=51.84, p95=108.12, max=136.68
- c_re (=c_recomp/n): count=128, mean=0.12, variance=4.47511e-35, std=6.68962e-18, min=0.12, p50=0.12, p95=0.12, max=0.12
- p_reuse: count=128, mean=0.328343, variance=0.128809, std=0.3589, min=0.052718, p50=0.052718, p95=0.862718, max=0.862718
- score: count=128, mean=1.44096, variance=2.4808, std=1.57506, min=0.231359, p50=0.231359, p95=3.7861, max=3.7861
- size_mb: count=128, mean=14.5279, variance=46.8478, std=6.84454, min=3.0625, p50=11.8125, p95=24.6367, max=31.1445
- object_type_counts: `{"rag_chunk_set": 29, "sharegpt_cold_context": 60, "sharegpt_hot_context": 39}`
- hit_counts: `{"False": 96, "True": 32}`
- run summary: c_recomp_model=`linear_c_re_ms_per_token_times_n_tokens`, eviction_granularity=`vllm_prefix_cache_block`, cop_object_profile_count=96, cop_avg_p_reuse=0.187718, cop_avg_score=0.823815972, hit_rate=0.692308

cop csv files:
- `h1/step1_D3/runtime/mid/mid_h1_lpe_requests.csv`

### 三张图片解释

> 数据来自 object-level COP（`score_source=object_level_cop`），tight / mid 两档 COP 分布一致（见上），下述数值取代表档 `tight`。

1. step1_D3_c_recomp_vs_n.png

  `c_recomp_ms` 与 `n_tokens` 严格线性：`c_re` 恒为 0.12 ms/token（std≈6.7e-18），`c_recomp` p50=51.84、p95=108.1，对应 n p50=432、p95=901。

  含义：重算成本只是 `c_re * n_tokens`，不含"某些对象特别难重算"的额外信息，本质是在表达对象长度。

2. step1_D3_p_reuse_histogram.png

  与旧 path-B 的饱和分布不同，COP 的 `p_reuse` 有真实区分度：mean=0.328、std=0.359、min=0.0527、p50=0.0527、max=0.863，呈 hot/cold 双峰（按 object_type 分化）。

  含义：COP 的复用概率不再大面积饱和，对象之间能拉开差距——这正是用真实分布坐实退化所需要的前提（旧 path-B 因 p_reuse≈0.97 看不出区分度）。

3. step1_D3_score_histogram.png

  `score` mean=1.44、std=1.58、min=0.231、max=3.79。 由于 `size_mb=μ_kv*n`、`c_recomp=c_re*n`，n 被约掉，`score = p_reuse * (c_re/μ_kv) ≈ p_reuse * 4.39`。

  含义：score 只是 p_reuse 的线性缩放（score_var/p_reuse_var ≈ (c_re/μ_kv)² = 19.3，实测 2.48/0.129=19.3），排序信息与 p_reuse 完全等价，不提供额外区分度。

综合：c_recomp 是长度线性、score 是 p_reuse 的常数倍 → 在同质 KV 下 `score` 退化为按 `p_reuse` 排序（§2 结论）。区别在于本轮用 COP 的真实 p_reuse 分布坐实，而非旧 path-B 的饱和值；唯一打破退化的途径是引入异质性（量化/对象类型，见 §5/H4）。

## 代码依据：COP 模块如何计算各变量
位置：`edgekv_cop.py`，`ObjectProfile.recompute()` 与 `ObjectProfile.estimate_reuse()`；入口 `COPProfiler.update_from_item()`。

`c_recomp_ms = c_re * n_tokens`，`size_mb = μ_kv * n_tokens`，`score = p_reuse * c_recomp_ms / size_mb`；`p_reuse` 由访问历史估计并可与先验融合。

```python
# edgekv_cop.py ObjectProfile.recompute()
self.size_mb = mu_kv * n_tokens
self.c_recomp_ms = c_re * n_tokens
self.c_restore_ms = self.size_mb / bw_gbps + self.d_deser_ms
self.score = (self.p_reuse * self.c_recomp_ms) / max(self.size_mb, 1e-9)

# edgekv_cop.py ObjectProfile.estimate_reuse()
recent_rate = recent_hits / len(self.access_history)
freq_rate = self.hit_count / self.access_count
recency_bonus = 1.0 / (1.0 + math.log1p(max(self.access_count - self.hit_count, 0)))
estimate = 0.55 * recent_rate + 0.35 * freq_rate + 0.10 * recency_bonus
# 若有 p_reuse_prior，再按 p_reuse_prior_weight 融合
```

## 代码依据：COP 记录如何写出
位置：`h1/run_h1_vllm0110_real.py`，`request_trace_fields()`，调用 `cop.update_from_item(...)` 并落 `score_source='object_level_cop'`。

```python
profile = cop.update_from_item(item, hit=bool(trace_hit or rag_hit), access_index=event_index)
fields.update({
    'p_reuse': round(profile.p_reuse, 6),
    'c_recomp_ms': round(profile.c_recomp_ms, 6),
    'score': round(profile.score, 9),
    'score_source': 'object_level_cop',
    'size_mb': round(profile.size_mb, 6),
})
```

## 生成脚本口径
位置：`h1/step1_D3/build_step1_d3.py`，`collect_budget()`。

只保留 `score_source == 'object_level_cop'` 且 `n_tokens>0`、`c_recomp_ms>0` 的 COP 行；没有 COP 行时默认报错（`--allow-fallback` 仅调试）。

```python
cop_events = [r for r in events if str(r.get('score_source', '')) == 'object_level_cop']
real_points = [p for r in cop_events if f(r, 'n_tokens') > 0.0 and f(r, 'c_recomp_ms') > 0.0]
```
