# H4 — Quality 维度核心验证

## 1. 实验定位

~~`H4` 在 `H0` 的无 `LMCache` 服务器端/边缘端模拟器中完成，用于验证 `TMS` 把 quality 作为一等调度维度后，是否能在同一质量预算 `epsilon` 下优于二元 cache、静态低精度和纯生命周期策略。~~ 修改方案：`H4` 仍在 `H0` 的无 `LMCache` 服务器端/边缘端模拟器中完成，但内部执行统一使用绝对预算 `epsilon_abs`，对外报告、图表与通过标准统一追加归一化预算 `epsilon_norm = epsilon_abs / token_ref`，主比较表述改为“同一 `epsilon_norm` 下是否优于二元 cache、静态低精度和纯生命周期策略”。

~~`H4` 不负责证明 `LMCache` 可部署性；该问题由 `H3` 负责。`H4` 的核心是算法有效性：自变量 `epsilon` 和 `M_budget` 改变时，因变量 `p95 TTFT`、`QLoss`、显存峰值是否按预期变化。~~ 修改方案：`H4` 仍不负责证明 `LMCache` 可部署性；核心变为在 `M_budget` 与 `epsilon_norm` 改变时观察 `p95 TTFT`、`QLoss_norm`、显存峰值和迁移行为，内部策略继续使用 `epsilon_abs` 与 `QLoss_abs` 判断可行性。

## 2. 回答什么 / 卡住后续什么

本实验要回答：

1. ~~同 `epsilon` 下，`TMS-tiered` 是否比 `RAGCache-like` 二元 cache、`static-int4`、`LPE-only` 降低 `p95 TTFT`。~~ 修改方案：同 `epsilon_norm` 下比较 `TMS-tiered` 与 `RAGCache-like`、`static-int4`、`LPE-only` 的 `p95 TTFT`，并在每个 cell 同时给出对应的 `epsilon_abs`。
2. ~~质量预算收紧或放宽时，`TMS` 的精度迁移行为是否可解释。~~ 修改方案：质量预算变化统一描述为 `epsilon_norm` 收紧或放宽，并通过 `token_ref` 解释对应的 `epsilon_abs` 如何改变可降级空间。
3. 在服务器端和边缘端两类设备画像下，quality 维度收益是否方向一致。

若 `H4` 不通过，论文主创新不足，应考虑退回纯 lifecycle 或系统工程路线。

## 3. 前置依赖

1. `pre` 已看到 `TMS-tiered` 的初步趋势。
2. `H0` 无 `LMCache` 模拟器已可在服务器端和边缘端运行。
3. `H0` 已支持精度等级、质量损失、显存预算和迁移动作记录。

## 4. 使用环境与资源

1. 平台：`H0` 模拟器，分别使用服务器端画像和边缘端画像。
2. 数据与负载：统一 `ShareGPT + RAG chunk` trace。
3. 精度三元组：使用 `pre` 选定并在 `H0` 固化的 `(alpha, beta, ell)` 表；真实 kernel 标定留到主实验前补充，不得阻塞本实验。
4. 对照方法：`RAGCache-like` 二元 cache、`static-int4`、`LPE-only`、`TMS-tiered`。
5. ~~自变量：`M_budget` 三档、`epsilon` 三档、设备画像两档。~~ 修改方案：自变量更新为 `M_budget` 三档、`epsilon_norm` 三档、设备画像两档；每个 `epsilon_norm` 在运行前按 `epsilon_abs = epsilon_norm * token_ref` 换算。
6. ~~因变量：`TTFT_p95`、`QLoss`、`M_peak`、`N_mig`、`hit_rate`。~~ 修改方案：因变量扩展为 `TTFT_p95`、`QLoss_abs`、`QLoss_norm`、`M_peak`、`N_mig`、`hit_rate`，其中预算合规以 `QLoss_norm <= epsilon_norm` 报告。

## 5. 输入与记录字段

1. 精度字段：`q_before`、`q_after`、`alpha(q)`、`beta(q)`、`ell(o,q)`
2. ~~预算字段：`M_budget`、`epsilon`、`epsilon_remaining`~~ 修改方案：预算字段扩展为 `M_budget`、`epsilon_abs`、`epsilon_norm`、`epsilon_remaining_abs`、`token_ref`；内部执行仍消费 `epsilon_remaining_abs`，对外展示 `epsilon_norm`。
3. 策略字段：`policy`、`tms_action`、`tier_migrations_total`
4. ~~结果字段：`TTFT_p95`、`QLoss`、`M_peak`、`hit_rate`~~ 修改方案：结果字段扩展为 `TTFT_p95`、`QLoss_abs`、`QLoss_norm`、`M_peak`、`hit_rate`，并要求 summary 同时保存 `epsilon_abs` 与 `epsilon_norm`。

## 6. 实验执行步骤

1. 固定同一份 trace 和同一组精度三元组。
2. ~~在服务器端画像下跑 `3 x 3` 的 `(M_budget, epsilon)` 矩阵。~~ 修改方案：在服务器端画像下内部执行 `3 x 3` 的 `(M_budget, epsilon_abs)` 矩阵，但主结果矩阵、图表和表格统一按 `(M_budget, epsilon_norm)` 呈现。
3. 在边缘端画像下重复同一矩阵，必要时缩小 trace 规模但不改变 trace 结构。
4. 对每个 cell 运行四种方法：`RAGCache-like`、`static-int4`、`LPE-only`、`TMS-tiered`。
5. ~~汇总 `TTFT_p95`、`QLoss`、`M_peak`、`N_mig`。~~ 修改方案：汇总 `TTFT_p95`、`QLoss_abs`、`QLoss_norm`、`M_peak`、`N_mig`，并为每个 cell 附带 `token_ref` 与 `epsilon_abs`。
6. ~~绘制 `epsilon -> p95 TTFT` 曲线和 `M_budget x epsilon` 结果矩阵。~~ 修改方案：绘制 `epsilon_norm -> p95 TTFT` 曲线和 `M_budget x epsilon_norm` 结果矩阵；若需要兼容历史记录，在图注中补充对应的 `epsilon_abs`。

## 6.1 对应伪代码

```python
for device in ["server", "edge"]:
    for m_budget in M_BUDGETS:
        for epsilon_norm in EPSILON_NORMS:
            for policy in ["ragcache_like", "static_int4", "lpe_only", "tms_tiered"]:
                result = run_h0_sim(
                    trace=trace,
                    device_profile=device,
                    policy=policy,
                    m_budget=m_budget,
                    epsilon_norm=epsilon_norm,
                    tier_table=tier_table,
                )
                write_summary("h4.csv", result)
```
修改方案：伪代码参数入口统一改为 `epsilon_norm`，runner 在每个 cell 开始时结合 `token_ref` 换算 `epsilon_abs`，并在输出里同时保存 `epsilon_abs/epsilon_norm/QLoss_abs/QLoss_norm`。

## 7. 产出物

1. `H4` 主结果矩阵。
2. ~~`epsilon-p95` Pareto 曲线。~~ 修改方案：主图改为 `epsilon_norm-p95` Pareto 曲线，并在附表保留 `epsilon_abs` 映射，避免与历史 pre 结果断裂。
3. 服务器端与边缘端趋势对照表。
4. 质量预算合规报告。

## 8. 通过标准

1. ~~多数 `(M_budget, epsilon)` cell 中，`TMS-tiered` 相比次优基线有显著 `p95 TTFT` 收益。~~ 修改方案：多数 `(M_budget, epsilon_norm)` cell 中，`TMS-tiered` 相比次优基线有显著 `p95 TTFT` 收益，并在每个 cell 记录对应的 `epsilon_abs`。
2. ~~`QLoss <= epsilon`，且违规样本可解释。~~ 修改方案：正式通过标准写为 `QLoss_norm <= epsilon_norm`，实现层保留等价校验 `QLoss_abs <= epsilon_abs`；违规样本必须同时给出 `token_ref` 以解释预算失效原因。
3. 服务器端和边缘端的收益方向一致，边缘端在紧显存下应更明显。

## 9. 与后续实验衔接

`H4` 通过后，quality 维度可以进入 `H3` 的可部署性验证和后续主实验。若未通过，先回到 `pre/H0` 调整质量预算、trace 结构或精度三元组，不应直接接 `LMCache`。
