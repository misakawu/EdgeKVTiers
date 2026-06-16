# H01245 TODO 与规划原文对照

本文档用于对齐 `h0/` 当前实现与论文规划中对 `H0/H1/H2/H4/H5` 的要求。结论先写清楚：

- 当前 `h0` 已实现的是基于 `pre实验/sim.py` 的解析仿真实验闭环：`H0` 多设备 trace replay、`H1/H2/H4/H5` 批处理、CSV/JSON 结果与可视化。
- 当前 `h0` 尚未实现的是规划中 D2-D4 要求的真实标定、真实引擎接入、正式 H4/H5 证据、失败案例与 Go/No-Go 报告。
- `H01245` 的下一步重点不是继续堆图，而是把解析趋势升级为真实系统证据链。

## 当前已实现

1. 解析仿真内核复用：`h0/run_h0.py` 动态加载 `pre实验/sim.py`。
2. H0 多设备配置：支持 `devices` / `device_profile`，可输出全局与设备级 `events.jsonl`、`summary.csv`、`config.resolved.json`、`validation.json`。
3. H0 基础校验：检查 replay 完整性、同 trace、同 token_ref、显存预算、epsilon 预算、summary 必要字段。
4. H1：`LRU/LFU/score/tiered` 在多档 `M_budget` 下输出 `h1_results.csv` 和 `h1_summary.csv`。
5. H2：`always-restore/always-recompute/rrs` 在多档 `BW` 下输出对比结果。
6. H4：三档 `M_budget` × 三档 `epsilon_norm` × 多 baseline × repeat 的解析矩阵。
7. H5：`BW × epsilon_norm` 35 网格、对象级 `q_star_offline/q_star_pred`、Kendall tau 表。
8. 可视化：`visualize_h01245.py` 生成 H0/H1/H2/H4/H5 图。

## 尚未实现

### P0：真实 H0 回放与标定

- [ ] 接 `vLLM` 默认 prefix caching，使用统一 ShareGPT 回放文件真实发送请求。
  - 原文对照：`§0.5 H0`：“ShareGPT 多轮 trace 在 vLLM 上回放成功，p95 TTFT/命中率/显存峰值可算、trace 落盘。”

- [ ] 采集真实 `TTFT_p95`、cache hit rate、GPU memory peak。
  - 原文对照：`§0.5 H0`：“测什么 + trace 字段：event、hit、n_tokens、size_mb、t_policy_ms；外加 p95 TTFT/显存峰值（CSV）。”

- [ ] 标定真实 `c_re`、`mu_kv_mb_per_token`、`d_deser_ms`。
  - 原文对照：`D2`：“标定 c_re/μ_kv/d_deser。”

- [ ] 将真实标定值写入 `h0` 配置和 `config.resolved.json`，并让 `run_h1245.py` 能读取。
  - 原文对照：`§0.5`：“H1/H2/H4/H5 在仿真器里就能出趋势 → H0/H3 才接真机钩子。”`D2`：“用真标定值在三档 M × 三档 ε 网格上跑……四 baseline。”

### P0：KIVI/H2O quality 三元组

- [ ] 接入 KIVI/KVQuant 等价 int8/int4 机制，测 `size_factor`、`quality_loss_per_token`、`restore_factor`。
  - 原文对照：`D2`：“用 KIVI/H2O 在 LongBench 上实测每精度 (α,β,ℓ) 替换示意三元组。”

- [ ] 接入 H2O/SnapKV 等价 sparse-k 机制，测 `sparse_k` 三元组。
  - 原文对照：`§0.5 H4`：“用 KIVI/H2O 实测 (size_factor, quality_loss_per_token, restore_factor) 三元组。”

- [ ] 在 LongBench 或等价任务上报告 perplexity / Rouge-L / accuracy 变化。
  - 原文对照：`§0.5 H4`：“对每精度，跑 perplexity 与 LongBench 任务。”

- [ ] 用真实三元组替换 `sim.TIERS` 的示意值，并保存标定表。
  - 原文对照：`D2`：“用 KIVI/H2O 在 LongBench 上实测每精度 (α,β,ℓ) 替换示意三元组。”`W2`：“真标定值 + 真三元组。”

### P1：H4 正式结论

- [ ] 用真实标定值重跑 `3 x 3` 的 `(M_budget, epsilon_norm)` 矩阵。
  - 原文对照：`D2`：“用真标定值在三档 M × 三档 ε 网格上跑 RAGCache/static-int4/LPE/TMS 四 baseline。”

- [ ] 明确 `RAGCache/PGDSF` 与当前 `pgdsf` 近似实现的差异，并补等价说明。
  - 原文对照：`§0.5 H4`：“对照：RAGCache(PGDSF 仿真，二元 cache)。”

- [ ] 补 `static-sparse-k` baseline，覆盖 H2O 类质量基线。
  - 原文对照：`实验方案 §3.1`：“强 quality 基线：static-int4 (KIVI)、static-sparse-k (H2O)。”

- [ ] 对每个 cell 报 `TTFT_p95`、`QLoss_abs`、`QLoss_norm`、`M_peak`、`N_mig`、`T_policy`、`c_mig`。
  - 原文对照：`实验方案 §3.1`：“主要指标改为 p95 TTFT、QLoss_abs、QLoss_norm……次要：cache hit rate、GPU 显存峰值、restore latency、recompute ratio；调度开销：N_mig、T_policy、c_mig。”

- [ ] 给出 H4 是否通过 P1 门槛：TMS 同 `epsilon_norm` 下相比次优 baseline 是否达到 25% 收益。
  - 原文对照：`§0.5 H4`：“TMS 比 RAGCache 二元策略降 p95 TTFT ≥25%；或同 p95 下显存降 ≥30%。”`8.3`：“P1 CCF-A 路线门槛：H4 quality 收益 ≥25% + H5 相变 Kendall-tau ≥0.8 + H1/H2/H3 通过。”

### P1：H5 相变面正式验证

- [ ] 将 `q_star_pred` 与 `q_star_offline` 解耦：一个来自定理 1 解析预测，一个来自独立代价穷举或真实测量。
  - 原文对照：`§0.5 H5`：“扫 (BW, ε) 网格，最优精度 q* 在曲面两侧明显切换，与定理 1 预测吻合。”

- [ ] 用真实或校准后的 `c_restore/c_recomp` 重跑 35 网格。
  - 原文对照：`D3`：“扫 (BW,ε) 35 网格点求每对象 q*，画相变热力图、算与定理 1 的 Kendall-tau。”

- [ ] 输出相变热力图、Kendall tau 表、agreement ratio，并解释 tau 高但 agreement 低的情况。
  - 原文对照：`§0.5 H5`：“画热力图 (BW, ε) → q*，验证块状结构。”`D3`：“H5 相变热力图 + Kendall-tau 表。”

- [ ] 标记相变边界附近的抖动 cell，作为失败案例来源。
  - 原文对照：`D3`：“从 H4/H5 trace 中挖 3-5 个失败案例（降级反弹、q* 跨边界抖动、I/O 反噬）。”

### P1：H2 巩固

- [ ] 用少量真实 offload 校准解析 `BW` 档位。
  - 原文对照：`W6`：“扫带宽（仿真为主 + 少量真实 offload 校准）比较 always-restore/recompute/RRS。”

- [ ] 增加 `restore_latency`、`recompute_latency`、`rrs_action` 的事件级字段。
  - 原文对照：`§0.5 H2`：“测什么 + trace 字段：rrs_action、c_restore_ms、c_recomp_ms、bw_gbps、hit；p95 latency（CSV）。”

- [ ] 检查 RRS 是否确实不劣于固定策略下包络，而不是只比较 summary 均值。
  - 原文对照：`§0.5 H2`：“RRS 取下包络，p95 在全带宽段不劣于两固定策略。”`D3`：“扫 BW 比 always-restore/always-recompute/RRS（H2）。”

### P1：失败案例与 Go/No-Go

- [ ] 从 H4/H5 trace 中自动抽取 3-5 个失败案例：降级反弹、`q*` 边界抖动、I/O 反噬、画像失准、epsilon 耗尽。
  - 原文对照：`D3`：“从 H4/H5 trace 中挖 3-5 个失败案例（降级反弹、q* 跨边界抖动、I/O 反噬）。”`实验方案 §7`：“案例分类：降级反弹、q* 跨边界抖动、I/O 反噬、画像失准、ε 耗尽。”

- [ ] 新增失败案例分析脚本，输出案例 ID、现象、根因、影响指标、修复建议。
  - 原文对照：`实验方案 §7`：“失败案例表（案例 ID、现象、根因、修复措施、是否已修复）。”

- [ ] 新增 Go/No-Go 报告模板，自动汇总 H0/H1/H2/H4/H5 结果和 P1 判定。
  - 原文对照：`D4`：“按 §0.5.4 骨架写两页 Go/No-Go 报告（瓶颈图 + H4 矩阵 + H5 热力图 + H2 表 + 失败案例 + 判定）。”

### P1：与 H3 的接口对齐

- [ ] 保证 `h0` 事件字段能与 `h3` hook 事件字段逐项对齐。
  - 原文对照：`实验方案 §1.7`：“对比 H0 事件语义与 H3 hook 事件语义。”

- [ ] 增加 `t_policy_ms`、`c_mig_ms`、`hook_name`、`module_decision` 等字段预留。
  - 原文对照：`D4`：“接 LMCache/vLLM/KIVI 钩子，把 COP/TMS/LPE/RRS 挂上去。”`实验方案 §1.7`：“记录 hook 事件、T_policy、c_mig、TTFT_p95、M_peak。”

- [ ] 明确 `h0` 解析事件与 `h3` 真实 hook 事件的语义差异。
  - 原文对照：`D4`：“服务器 + 端侧各跑一遍 H4 的 trace 对比接/不接策略层。”`实验方案 §1.8`：“H0 无 LMCache 服务器端/边缘端环境闭环；H3 接入 LMCache 的可部署性结果。”

## 建议实现顺序

1. 先补真实标定输入：`c_re/mu_kv/d_deser` 和 KIVI/H2O 三元组。
2. 用真实标定值重跑 H4，判断 quality 维度是否继续成立。
3. 重做 H5，使定理预测与离线/实测最优解真正解耦。
4. 补失败案例抽取和 Go/No-Go 报告。
5. 再与 `h3/` 对齐 hook 事件语义，进入真实系统接入。

