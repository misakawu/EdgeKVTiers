# H2 — RRS 恢复 vs 重算阈值

## 1. 实验定位

`H2` 在 `H0` 无 `LMCache` 模拟器中完成，用于验证 `RRS` 在不同带宽下是否能避免盲目 restore 造成的 I/O 反噬。`pre` 只判断趋势，`H2` 在服务器端和边缘端模拟环境中给算法有效性证据。

## 2. 回答什么 / 卡住后续什么

本实验要回答：

1. `c_restore <= c_recomp` 的阈值判据是否能稳定选择 restore/recompute。
2. `BW` 变化时，`RRS` 是否接近 `always-restore` 和 `always-recompute` 的较优下包络。
3. 边缘端低带宽画像下，`RRS` 是否能明显减少 I/O 反噬。

若 `H2` 不成立，offload 后复用动作缺少可靠判据，`RRS` 不能进入 `H3` 和主实验。

## 3. 前置依赖

1. `pre` 已看到 `BW` 扫描的初步趋势。
2. `H0` 模拟器支持 `offload`、`restore`、`recompute` 状态。
3. 统一 trace 中有被驱逐后再次复用的对象。

## 4. 使用环境与资源

1. 平台：`H0` 模拟器，服务器端画像和边缘端画像各跑一遍。
2. Cache 层：不接入 `LMCache`。
3. 数据与负载：统一 `ShareGPT + RAG chunk` trace，重点分析 offloaded 后复用对象。
4. 对照策略：`always-restore`、`always-recompute`、`RRS`。
5. 自变量：`BW={0.5,1,2,4,8,16}` GB/s，可按设备画像扩展。
6. 因变量：`TTFT_p95`、`restore_latency`、`recompute_ratio`、错误选择比例。

## 5. 输入与记录字段

1. `device_profile`
2. `BW`
3. `object_id`
4. `q`
5. `c_restore`
6. `c_recomp`
7. `rrs_action`
8. `TTFT_p95`
9. `recompute_ratio`

## 6. 实验执行步骤

1. 固定同一条 trace 和 offload 对象集合。
2. 对每个 `BW` 档运行三种复用策略。
3. 记录逐对象 `c_restore`、`c_recomp` 和 `rrs_action`。
4. 汇总 `TTFT_p95`、`recompute_ratio` 和 restore 次数。
5. 分别输出服务器端和边缘端带宽敏感性曲线。
6. 检查 `RRS` 是否贴近两种固定策略的较优下包络。

## 6.1 对应伪代码

```python
def rrs(o, q, bw, mu_kv, c_re, d_deser, tier_table):
    alpha, _, beta = tier_table[q]
    c_restore = alpha * mu_kv * o.n_tokens / bw + d_deser
    c_recomp = beta * c_re * o.n_tokens
    return "restore" if c_restore <= c_recomp else "recompute"
```

## 7. 产出物

1. 带宽到 `p95 TTFT` 曲线。
2. `RRS` 判据效果表。
3. 低带宽 I/O 反噬样本表。
4. 服务器端/边缘端趋势对照表。

## 8. 通过标准

1. 多数带宽档下，`RRS` 不劣于两种固定策略中的较优者。
2. 低带宽档下，`RRS` 明显优于 `always-restore`。
3. 逐对象决策能由 `c_restore` 与 `c_recomp` 解释。

## 9. 与后续实验衔接

`H2` 通过后，`RRS` 可进入 `H3` 的 `LMCache` 接入验证。若失败，应先在 `H0` 中修正带宽估计、反序列化开销或批量 restore 模型。
