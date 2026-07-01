# H0 TODO：异质探针验证 LPE 唯一出路

来源：`h1/H1_复盘与重跑配置单.md` 第 5 节“第三步（并行）：异质探针验证唯一出路”。

## 目标

- [x] 用解析仿真器验证：同质 KV 下 `score = p_reuse * c_recomp / size` 会退化为只看 `p_reuse`；只有引入异质对象，让 `c_re/μ_kv` 不再是常数，LPE 才可能明显反超 LRU。
- [x] 不占真机，不接 vLLM 服务端；先用仿真器把结论跑出来，作为是否把重点转向 H4 异质/量化维度的证据。

## 具体任务

- [x] 新建或扩展一个 H0/H1 可复用的解析仿真器，输入为对象访问 trace，输出每种策略的 hit rate、miss/recompute cost、eviction 次数和总/分位延迟 proxy。
- [x] 在仿真器中实现至少 3 种驱逐策略：`LRU`、`LFU`、`LPE-score`。
- [x] 实现 LPE 分数：

```text
score(o) = p_reuse(o) * c_recomp(o) / size(o)
```

- [x] 为每个对象维护 `p_reuse` 估计，先用简单规则即可：LRU-K、频次衰减或访问间隔 EMA，保证 LRU/LFU/LPE 使用同一条 trace 可比。
- [x] 构造同质基线场景：所有对象 `size_factor = 1.0`、`recompute_factor = 1.0`，确认 LPE 不应稳定优于 LRU。
- [x] 构造异质探针场景，至少包含以下对象类：

| 对象类 | size 因子 | 重算因子 | 目的 |
| --- | ---: | ---: | --- |
| `fp16_session_kv` | `1.0` | `1.0` | 同质基线 |
| `int4_quant_block` | `0.25` | `0.9` | 模拟 KIVI/H2O：显存降约 4x，重算成本近似不变，拉开 `c_re/μ_kv` |
| `rag_chunk` | 按长度或固定档位 | 按长度或固定档位 | 模拟不同复用分布/对象类型 |

- [x] 设计一条可控 trace：同时包含 session prefix 局部性、RAG chunk 热点复用、冷对象干扰，并固定随机种子。
- [x] 扫缓存预算，至少覆盖低/中/高 3 档；优先让 hit rate 落到 `0.5~0.85` 的有效窗口。
- [x] 对每个预算和场景运行 `LRU/LFU/LPE`，每点 `reps >= 3`。
- [x] 输出 CSV：`scenario,budget,policy,rep,hit_rate,miss_cost,p95_cost,evictions,cached_objects,score_mean,score_std,p_reuse_mean,p_reuse_std`。
- [x] 输出图表：
  - 同质场景：`budget -> p95_cost/hit_rate`，验证 LPE 与 LRU 接近或更差。
  - 异质场景：`budget -> p95_cost/hit_rate`，观察 LPE 是否明显反超 LRU。
  - `score` 与 `p_reuse` 分布图，说明异质场景下 score 是否获得额外区分度。

## 验收判据

- [x] 若异质场景下 LPE 在有效窗口内相对 LRU 的 `p95_cost` 稳定下降，且至少 2 个 budget 档成立，则结论为：LPE 的 score 需要异质对象才有意义，后续应转向 H4 质量/量化维度。
- [ ] 若异质场景下 LPE 仍与 LRU 打平或更差，则结论为：当前 score 公式本身需要重审，不能直接作为 H4 的调度依据。
- [x] 最终交付一页结论：同质退化是否复现、异质探针是否有效、是否建议转 H4。
