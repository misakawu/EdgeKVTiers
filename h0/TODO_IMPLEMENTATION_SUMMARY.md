# H0 TODO 当前可实现项总结

## 已落地

- H0 事件字段补齐：`c_restore_ms`、`c_recomp_ms`、`restore_latency_ms`、`recompute_latency_ms`、`T_policy_ms`、`c_mig_ms`、`hook_name`、`module_decision`。
- H2 巩固：输出每个 `BW x rrs_mode` 的事件样本，支持检查 restore/recompute 事件级行为。
- H4 正式结论支撑：加入 `static-sparse-k` baseline，结果表补 `TTFT_p95`、`QLoss_abs`、`QLoss_norm`、`M_peak`、`N_mig`、`T_policy`、`c_mig`，并输出 P1 25% gate 判定。
- H5 相变面验证：保留 `q_star_offline` 与 `q_star_pred` 的独立代价函数，新增 `phase_boundary_cell` 与 `agreement_note`，解释 tau 高但 exact agreement 低的情况。
- 失败案例与 Go/No-Go：自动输出 `failure_cases.csv`、`go_nogo_report.json`、`go_nogo_report.md`。
- 可视化更新：新增 H4 P1 gate 图、H5 Kendall tau 图、H5 相变边界标记、Go/No-Go 总览图。

## 当前仍不能真实完成

- vLLM 默认 prefix caching 真实回放。
- 真实 `TTFT_p95`、cache hit rate、GPU memory peak 采集。
- 真实 `c_re`、`mu_kv_mb_per_token`、`d_deser_ms` 标定。
- KIVI/KVQuant、H2O/SnapKV 三元组实测。
- LongBench/perplexity/Rouge/accuracy 质量评估。

## 判定口径

当前输出仍是解析仿真证据链。所有配置中新增的 `calibration.status` 都标记为 `placeholder_until_real_calibration`，Go/No-Go 默认要求 `real_calibration_present=false` 时保持 `NO-GO`。
