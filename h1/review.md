# H1 复盘进度 Review

> 对照基准：`h1/H1_复盘与重跑配置单.md`
> 更新日期：2026-06-27
> 一句话现状：**第一步（D1-D4 四个测量/实现修复）已全部完成**；第二步 B 已从前置扫描推进到“换到正确区间重跑”。已生成并固定合理的真实 trace：`data/edgekv_traces/sharegpt_hotpotqa_session.jsonl`（summary 同目录保留），默认 H0/H1 运行入口已切到该文件。**下一步按三档 budget × 两策略（`h1_lru`/`h1_lpe`）正式测试**。

---

## 1. 进度对照表

| 配置单步骤 | 内容 | 状态 | 证据 / 产物 |
| --- | --- | --- | --- |
| D1 | 决策计时器修复（`policy_time_us` 非 0、有分布） | ✅ 完成 | summary `policy_time_us_avg=232.7`、`eviction_decision_time_us_avg=1579`（不再为 0） |
| D2 | 全量重排 -> 增量小顶堆 | ✅ 实现完成（验收待 B 步多负载确认） | `sitecustomize.py` rank heap 增量选择 + `_edgekv_compact_rank_heap` 周期压缩 + `H1_LPE_REORDER_WINDOW` 有界窗口 |
| D3 | 坐实退化：c_recomp 线性 / 驱逐粒度 / score+p_reuse 直方图 | ✅ 完成（本轮升级为 COP 路径 A） | `h1/step1_D3/`（三图 + summary + 报告） |
| D4 | TTFT 拆 `queue_wait_ms` + `prefill_ms` | ✅ 完成 | summary `d4_metrics_available_ratio=1.0`、`prefill_p95_ratio=0.9999`、`queue_wait_p95_ratio=0.0001` |
| B-1 | budget 下扫找有效窗口（旋钮1） | ⚠️ 已跑，未得到可用窗口 | `h1/out/find_interval/`：0.30/0.40/0.50/0.60/0.65 均 `ok=false`，引擎初始化失败；报告中的 hit=0 是失败占位，不是实验数据 |
| B-2 | 固定 budget 去饱和（旋钮2） | ⚠️ 已跑，去饱和成功但仍在死区 | `h1/out/find_load/`：bs2/4/8 queue_wait≈0，但 hit≈0.952>0.85；bs16 接近排队边界；无推荐工作点 |
| B-3 | workload 旋钮造有效窗口 | ✅ 已完成并固化为正式 trace | `data/edgekv_traces/sharegpt_hotpotqa_session.jsonl`：真实 prompt 第一行均为 `[TRACE_OBJECT ...]`；RAG 使用 `complete_prompt`，replay 后 marker 保持第一行；summary 显示 cold scan 每轮约 18k-20k tokens，hot 工作集约 6.5k tokens |
| B-4 | 正确区间正式重跑 | ⏳ 当前进行中 / 下一步 | 使用固定 trace `sharegpt_hotpotqa_session.jsonl`，跑三档 budget（tight/mid/loose 或配置单指定三档）× 两策略 `h1_lru`、`h1_lpe`；保持同一到达序列、同一 batch 设置，优先 reps>=3 |
| C | 异质探针仿真验证唯一出路 | ⏳ 未开始 | — |
| 决策树 | 2026-07-04 组会给 Go/A/C 结论 | ⏳ 未到 | — |

---

## 2. 已完成事项详述

### D1 — 决策计时器（已修）
- `sitecustomize.py` 用 `time.perf_counter_ns()` 实现 `_edgekv_note_policy_time()` 与 `_edgekv_note_eviction_decision_time()`，由 `EDGEKV_H1_PROFILE_POLICY_TIME` 开关。
- 验收：本轮真实运行 `policy_time_us_avg=232.7`、`eviction_decision_time_us_avg=1579.0`，均非 0 且有分布（旧三档曾全为 0）。

### D2 — 增量小顶堆（已实现）
- `sitecustomize.py` 维护 `_edgekv_h1_rank_heap`，每次 touch/cache/free 增量入堆（懒删除），`_edgekv_compact_rank_heap()` 周期重建去除陈旧项；重排走有界窗口 `H1_LPE_REORDER_WINDOW`（默认 128），注释标注 “Incremental selection (D2)”。
- 当前一档观测：`free_queue_reorder_calls=247`、`blocks=5020`、`window=272`、`time_ms=381.9`。
- ⚠️ 验收口径「reorder 时间基本不随负载涨」属跨负载结论，需在 B 步多负载档位对比后才能最终判定。

### D3 — 坐实退化（本轮主要工作：切到 COP 路径 A）
**背景问题**：原 step1_D3 的 `c_recomp/p_reuse/score` 取自 `sitecustomize.py` 的 **block 级内联画像（路径 B）**，与设计要求「这些量在 COP 模块 `edgekv_cop.py` 计算」不一致，且其 `p_reuse` 公式（含 object_type 先验、0.55/0.30/0.15 加权）与 COP 不同，数值不可与 `score_source=object_level_cop` 互证。

**本轮改动**：将 step1_D3 整条管线切换为 **路径 A — COP 模块**：
- `run_step1_D3_monitor.sh` 改为驱动 `run_h1_vllm0110_real.py`（policy `h1_lpe`，buckets tight=0.720 / mid=0.735，经 conda `edgekv-vllm0110`）。
- `build_step1_d3.py` 改为读取其 per-request CSV 中 `score_source==object_level_cop` 的行——`c_recomp/p_reuse/score` 全部来自 `edgekv_cop.py`（`COPProfiler.update_from_item -> ObjectProfile.recompute / estimate_reuse`，即设计 §6.2 Algorithm 1，对象粒度）。
- 旧 path-B 数据备份至 `h1/step1_D3/runtime_pathB_backup_20260627/`。

**结果（真实重跑，128 对象/档）**：
- ① `c_recomp` 严格线性：`c_re`≡0.12 ms/token（std≈6.7e-18），`c_recomp=c_re·n`，只编码长度。
- ② 驱逐粒度：引擎实际驱逐为 `vllm_prefix_cache_block`（block 级），COP 在对象级提供画像/打分——报告已如实区分两者。
- ③ `p_reuse` 有真实区分度：mean=0.328、std=0.359，呈 hot/cold 双峰（按 object_type 分化），不再是旧 path-B 饱和的 ≈0.97。
- ③ `score` 退化被精确坐实：`score=p_reuse·(c_re/μ_kv)≈p_reuse·4.39`，`score_var/p_reuse_var=2.48/0.129=19.3=(c_re/μ_kv)^2`——score 与 p_reuse 排序完全等价，零额外信息（§2 结论成立）。
- COP 画像仅由 trace 回放顺序决定、与显存预算无关 -> tight/mid 两档 COP 分布一致，预算只改引擎侧 summary（hit_rate 0.686 vs 0.692、evictions 3249 vs 2863）。

**产物**：
- `h1/step1_D3/out/step1_D3_c_recomp_vs_n.png`
- `h1/step1_D3/out/step1_D3_p_reuse_histogram.png`
- `h1/step1_D3/out/step1_D3_score_histogram.png`
- `h1/step1_D3/out/step1_D3_summary.json`
- `h1/step1_D3/step1_d3_report.md`（含「三张图片解释」数据驱动小节）
- COP 原始数据：`h1/step1_D3/runtime/{tight,mid}/*_h1_lpe_requests.csv`

### D4 — TTFT 拆分（已完成）
- `run_h1_vllm0110_real.py` 的 `request_output_timing_ms()` 从 `RequestOutput.metrics` 拆出 `queue_wait_ms` 与 `prefill_ms` 落盘。
- 验收：本轮 `d4_metrics_available_ratio=1.0`（128/128 全可算），`prefill_p95_ratio=0.9999`、`queue_wait_p95_ratio=0.0001` -> 该工作点是 **prefill 主导、未饱和**，能正确归因排队 vs prefill。

---

## 3. 关键结论（D 阶段产出）

1. **退化已被代码与真实数据双重坐实**：同质 KV 下 `score=p_reuse·(c_re/μ_kv)` 是常数倍缩放，按 score 驱逐 ≡ 按 p_reuse 驱逐（实测 score/p_reuse 方差比 = (c_re/μ_kv)^2，严格相等）。这与 §2 的数学预测一致，且本轮用 **COP 真实 p_reuse 分布**（非饱和值）给出，证据更硬。
2. **打破退化的唯一途径是引入异质性**（量化精度 / 对象类型让 `c_re/μ_kv` 不再是常数）——即配置单 §5 探针 C 与合并稿 H4 的方向。

---

## 4. 第二步 B 最新进展（截至 `h1/out/find_workload`）

**目标不变**：把「缓存策略质量」逼成真正瓶颈——命中率落入 **0.5-0.85** 且 `queue_wait/p95 < 50%` 的有效窗口，四策略（LRU/LFU/LPE/default）出「命中率-p95」主图 + 「budget-命中率」健全性图 + 有效窗口内降幅表。

**旋钮1：budget 下扫（`h1/out/find_interval`）**
- 已扫 0.30/0.40/0.50/0.60，并额外有 0.65 产物。
- 这些 cell 的 summary 均为 `ok=false` / `requests=0` / `RuntimeError: Engine core initialization failed`，`hit_rate=0`、`p95=0` 是失败占位。
- 结论：当前 **3×RTX 2080 Ti（11GB）+ Qwen2.5-7B/TP=2** 下，继续靠降低 `gpu_memory_utilization` 进入有效窗口不可行；`run_find_interval.py` 已改为过滤 `ok=false` cell，失败档单列 failed，不参与窗口判定。

**旋钮2：固定 budget=0.735 扫并发（`h1/out/find_load`）**

| batch_size | policy | hit_rate | p95 TTFT(ms) | qwait/p95 | 窗口判定 |
| ---: | --- | ---: | ---: | ---: | --- |
| 16 | h1_lru | 0.951923 | 2303.192 | 0.492111 | hit 过高，仍不在窗口 |
| 16 | h1_lpe | 0.951923 | 2519.512 | 0.500607 | hit 过高且略饱和 |
| 8 | h1_lru | 0.952191 | 1190.558 | 0.000029 | 已去饱和，但 hit 过高 |
| 8 | h1_lpe | 0.951907 | 1322.977 | 0.000028 | 已去饱和，但 hit 过高 |
| 4 | h1_lru | 0.952064 | 601.240 | 0.000133 | 已去饱和，但 hit 过高 |
| 4 | h1_lpe | 0.951907 | 656.331 | 0.000046 | 已去饱和，但 hit 过高 |
| 2 | h1_lru | 0.952032 | 265.174 | 0.000111 | 已去饱和，但 hit 过高 |
| 2 | h1_lpe | 0.952207 | 287.483 | 0.000139 | 已去饱和，但 hit 过高 |

**判读**：
1. 降 batch_size 能把排队项压到近似 0（bs2/4/8 的 `qwait/p95≈0`），D4 指标可用。
2. 命中率几乎不随 batch_size 变化，稳定在约 0.952，仍高于有效窗口上界 0.85；因此当前 workload 仍是死区，不能直接进入四策略主图。
3. 已扫档位里 LPE 的 p95 全部慢于 LRU（bs2 慢 8.4%、bs4 慢 9.2%、bs8 慢 11.1%、bs16 慢 9.4%），且 hit 基本打平；这与 D3 的同质退化结论一致，但由于还未进入有效窗口，不能把它当作 B 的最终判定。
4. `run_find_load.py` 顶部过期注释已修正：0.735 当前仍是 hit 过高的 workload 死区，batch 扫描只负责去饱和。

**旋钮3：固定 budget=0.735、batch_size=8 扫 workload unique prefix（`h1/out/find_workload`）**

先做了一个 metadata-only 负控：只改 `reuse_key/session_id/rag_reuse_key`，不改 prompt 文本，`uf250/500/750/900` 的 GPU hit 仍全部约 0.952。结论：vLLM GPU prefix cache 的真实 hit 由 prompt token 前缀决定，不由 COP/CSV metadata 决定。随后将扫描器改为对选中请求在 prompt 开头注入 per-request unique marker，再重跑 LRU。

| workload knob | policy | hit_rate | p95 TTFT(ms) | qwait/p95 | 窗口判定 |
| --- | --- | ---: | ---: | ---: | --- |
| uf250 | h1_lru | 0.918169 | 1097.694 | 0.000036 | hit 仍过高 |
| uf500 | h1_lru | 0.851698 | 1199.308 | 0.000035 | 贴近上界但略高于 0.85 |
| uf750 | h1_lru | 0.686856 | 1125.238 | 0.000038 | ✅ 有效窗口 |
| uf900 | h1_lru | 0.566836 | 1092.128 | 0.000038 | ✅ 有效窗口 |

**当前 B 阶段结论**：前置扫描已经证明，metadata-only 改动无效，真实 prompt prefix 才能改变 vLLM GPU prefix-cache hit。现在已不再使用 `uf750/uf900` 临时派生 trace，而是固化为正式 trace：`data/edgekv_traces/sharegpt_hotpotqa_session.jsonl`。该 trace 的结构是 budget-sensitive 的 hot prime -> cold scan -> hot probe ladder：`estimated_hot_working_set_tokens=6512`，`estimated_cold_scan_tokens_per_round=[18454, 18369, 20593]`，预期低 budget 更容易驱逐 hot prefix，中/高 budget 保留更多 hot prefix。

**当前所在步骤**：`h1/H1_复盘与重跑配置单.md` 第 4 节「第二步：换到正确区间重跑」。下一步不再继续造 trace，而是用固定 trace 做正式策略对比。

**下一步运行矩阵**：三档 budget × 两策略：

| 维度 | 设定 |
| --- | --- |
| trace | `data/edgekv_traces/sharegpt_hotpotqa_session.jsonl` |
| budget | 三档：优先沿用 `tight` / `mid` / `loose`；如要贴合配置单，可替换为当前机器可启动的三个数值档 |
| policies | `h1_lru`, `h1_lpe` |
| batch/order | 固定同一 `replay_batch_size`、`batch_order=original`、同一到达序列 |
| requests | trace 全量 281，或固定 `--num-prompts 281` 保证每档一致 |
| reps | 建议 `reps>=3`；若先 smoke，可每档每策略 1 次确认可启动 |

推荐直接使用 `h1/run_step3_budget_tiers.py` 或在 `h1/run_test.py` 的两策略 smoke 基础上扩展三档。当前各运行入口默认 trace 已改为 `sharegpt_hotpotqa_session.jsonl`，因此不再需要额外传旧的 pressure trace 路径。

---

## 5. 下一步任务清单

### P0 — 修正扫描器报告语义
- [x] `run_find_interval.py` / 汇总逻辑过滤 `ok=false`、`requests=0` 的 summary；失败 cell 单独列为 failed，不参与窗口判定。
- [x] 修正 `run_find_load.py` 注释：budget=0.735 在最新实测中 **未** 进入 hit 窗口，当前问题是 workload 死区而非仅仅饱和。

### P0 — 造出有效窗口
- [x] 固定一个去饱和并发作为基准，优先用 `batch_size=8`（p95 仍有统计量、`qwait/p95≈0`）。
- [x] 调整 trace/workload 降复用率：增加真实 unique prompt prefix；LRU 已在 `uf750`/`uf900` 跑到 `hit_rate=0.5-0.85`。
- [x] 找到窗口并固化正式 trace：`data/edgekv_traces/sharegpt_hotpotqa_session.jsonl`；旧 trace 已清理，只保留 JSONL 和 summary。
- [ ] 用固定 trace、同一到达序列、同一 batch 设置，跑三档 budget × 两策略：`h1_lru`、`h1_lpe`，建议每点 `reps>=3`。

### P1 — B 阶段正式交付
- [ ] 生成「hit_rate/budget 或 workload knob - p95 TTFT」主图，标出有效窗口。
- [ ] 生成「knob - hit_rate」健全性图。
- [ ] 生成三档内 `LPE vs LRU` 的 p95 降幅表，同时附 `queue_wait/p95` 确认未饱和，并记录 hit 是否随 budget 增加。
- [ ] 用 B 的多负载/多工作点结果补 D2 验收：`free_queue_reorder_time_ms` 是否不随负载显著上涨；当前 `find_load` 中 LPE reorder time 约 409-463ms/256req，仍需归一化到 per decision / per request 再判断。

### P1 — C 探针与 7-04 决策
- [ ] 开始 C：在 COP/仿真器里加入异质对象（fp16 KV、int4 量化块、RAG chunk），让 `c_re/μ_kv` 不再是常数，验证 score 是否能反超 LRU。
- [ ] 7-04 组会按决策树给结论：若 B 仍无信号且 C 有效，则走「同质 KV 负结论 + 转 H4 异质/量化」；若 B 意外有信号，再继续推 H1 ≥20%。

---

## 6. 本轮文件改动清单

| 文件 | 改动 |
| --- | --- |
| `h1/step1_D3/build_step1_d3.py` | 数据源 runtime_monitor.jsonl -> COP per-request CSV（`score_source=object_level_cop`）；新增 `_figure_interpretation()` 数据驱动图解 |
| `h1/step1_D3/run_step1_D3_monitor.sh` | 改为驱动 `run_h1_vllm0110_real.py`（COP 路径），buckets tight/mid |
| `h1/step1_D3/step1_d3_report.md` | 重新生成为「对象级 COP（路径 A）」报告 |
| `h1/step1_D3/out/*`、`runtime/{tight,mid}/*` | 真实重跑产物刷新 |
| `h1/step1_D3/runtime_pathB_backup_20260627/` | 旧 path-B 数据备份 |
| `h1/out/find_interval/*` | B-1 budget 下扫产物；当前为失败占位，需要过滤 `ok=false` 后重报 |
| `h1/out/find_load/*` | B-2 batch 扫描产物；确认去饱和成功但 hit 仍约 0.952 |
| `h1/run_find_interval.py` | 过滤失败 cell：`ok=false` / `requests=0` 单列 failed，不参与窗口判定 |
| `h1/run_find_load.py` | 修正过期注释：0.735 未进入 hit 窗口，batch 只负责去饱和 |
| `h1/run_find_workload.py` | 新增 workload 扫描器：生成 unique prompt prefix 派生 trace，固定 bs8 找有效窗口，并支持 `--four-policies` / `--reps` |
| `scripts/optimize_h0_pressure_trace.py` | 固化正式 budget-sensitive trace 生成器：对象 marker 首行、RAG `complete_prompt`、budget ladder 排列、summary 输出，并默认清理旧 trace |
| `data/edgekv_traces/sharegpt_hotpotqa_session.jsonl`、`.summary.json` | B-3 正式 trace；旧 `h0_*pressure` / `h0_budget_sensitive*` 已清理 |
| `h0/run_h0_vllm.py`、`h1/run_h1_vllm0110_real.py`、`h1/run_step3_budget_tiers.py`、`h1/run_step3_real.py`、相关 shell/test | 默认 replay trace 名称统一为 `sharegpt_hotpotqa_session.jsonl` |
| `h1/out/find_workload/*`、`data/edgekv_traces/h1_workload_sweep/*` | B-3 workload 扫描历史产物；`uf750`/`uf900` 证明真实 unique prefix 可进入有效窗口，现已被正式 trace 替代 |

---

## 7. 待办 / 风险

- [x] **B 步阻塞点**：原两个旋钮没有命中有效窗口；已通过真实 unique prompt prefix workload 扫描造出有效窗口。
- [x] **报告风险**：`find_interval` 目前把失败 summary 写入窗口报告，容易误读为 hit=0；已在扫描器逻辑中修正。
- [ ] **B 步下一动作**：使用 `sharegpt_hotpotqa_session.jsonl` 跑三档 budget × 两策略（`h1_lru`/`h1_lpe`），建议 reps>=3，并生成主图/表。
- [ ] **D2 收尾验收**：B 步多 workload 下确认 reorder 时间不随负载显著上涨。
- [ ] **C 探针**：异质对象仿真（DDL 07-04）。
- [ ] **决策建议**：B 大概率「无信号」（同质退化已坐实），按决策树倾向 A（诚实负结论 + 边界讨论）并把火力转 C/H4；LPE 在规划中本就是「支撑性贡献」，非失败。
