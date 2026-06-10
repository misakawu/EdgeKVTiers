# H0 — 无 LMCache 的服务器端/边缘端模拟器

## 1. 实验定位

`H0` 不是 `LMCache` 接入实验，也不是主实验。它的任务是在服务器端和边缘端部署一个不接入 `LMCache` 的可复现实验模拟器，为 `H1/H2/H4/H5` 测试本项目四个算法 `COP/TMS/LPE/RRS` 的有效性提供统一环境。

`pre` 只做纯 Python 趋势判断；`H0` 把同一套 trace、参数和策略状态机固化成可在不同设备上运行的模拟器。`H3` 才负责接入 `LMCache` 验证可部署性。

## 2. 回答什么 / 卡住后续什么

本实验要回答：

1. 同一份 trace 是否能在服务器端和边缘端模拟器中一致回放。
2. 模拟器是否能记录 `COP/TMS/LPE/RRS` 所需的对象画像、策略动作和指标。
3. ~~`M_budget/BW/epsilon` 等自变量是否能稳定控制，并让 `H1/H2/H4/H5` 复用同一实验环境。~~ 修改方案：`M_budget/BW/epsilon_abs` 继续作为内部执行自变量稳定控制，同时所有 `H1/H2/H4/H5` 输出必须追加 `epsilon_norm = epsilon_abs / token_ref` 与 `token_ref`，使后续实验复用统一的外部质量预算口径。

若 `H0` 不通过，`H1/H2/H4/H5` 的算法有效性验证没有统一平台，不能进入 `H3` 或主实验。

## 3. 前置依赖

1. `pre` 已完成，明确需要扫描的参数范围和 trace 字段。
2. 已准备统一 trace：ShareGPT 多轮 prefix 复用、RAG chunk 复用或等价自建 trace。
3. 已准备服务器端与边缘端设备，边缘端可用真实边缘 GPU，也可用服务器限显存/限带宽方式模拟。

## 4. 使用环境与资源

1. 平台：服务器端模拟器 + 边缘端模拟器。
2. Cache 层：不接入 `LMCache`，模拟器内部维护 `resident/offloaded/dropped` 状态。
3. 模型层：不要求真实 LLM 推理；可用解析代价模型或轻量 mock engine 产生 `TTFT`、命中、恢复、重算和质量损失。
4. 设备画像：服务器端和边缘端分别配置 `M_budget`、`BW_profile`、`d_deser`、`c_re`、`mu_kv`。
5. 算法接口：模拟器必须暴露 `COP.update`、`TMS.on_pressure`、`LPE.evict`、`RRS.on_reuse` 的调用点。
6. 记录格式：所有事件写入统一 JSONL，保证 `H1/H2/H4/H5` 可复用同一后处理脚本。

## 5. 输入与记录字段

1. trace 字段：`request_id`、`session_id`、`object_id`、`object_type`、`n_tokens`、`arrival_ts`
2. ~~环境字段：`device_profile`、`M_budget`、`BW`、`epsilon`、`mu_kv`、`c_re`、`d_deser`~~ 修改方案：环境字段扩展为 `device_profile`、`M_budget`、`BW`、`epsilon_abs`、`epsilon_norm`、`token_ref`、`mu_kv`、`c_re`、`d_deser`，其中内部执行读取 `epsilon_abs`，报告口径统一用 `epsilon_norm`。
3. 决策字段：`policy`、`q_before`、`q_after`、`lpe_action`、`rrs_action`、`tms_action`
4. ~~指标字段：`ttft_ms`、`hit`、`M_peak`、`qloss_total`、`recompute_ratio`、`t_policy_ms`~~ 修改方案：指标字段扩展为 `ttft_ms`、`hit`、`M_peak`、`qloss_total_abs`、`qloss_total_norm`、`recompute_ratio`、`t_policy_ms`，并保留 `epsilon_abs` 与 `epsilon_norm` 以支持等价预算校验。

## 6. 实验执行步骤

1. 将 `pre` 的纯 Python 仿真循环封装为可配置模拟器。
2. 固定 trace 格式和配置文件格式，保证服务器端/边缘端使用同一份输入。
3. 实现无 `LMCache` 的缓存状态管理：准入、命中、降级、驱逐、offload、restore/recompute。
4. 加入设备画像：服务器端高带宽/较大显存，边缘端低带宽/小显存。
5. 跑 smoke test：同一 trace 在两类设备画像下都能输出完整 JSONL 和 summary CSV。
6. ~~校验指标单调性：显存越紧命中率应下降，带宽越低 restore 代价应上升，`epsilon` 越紧可用低精度空间应收缩。~~ 修改方案：单调性校验改为“显存越紧命中率应下降，带宽越低 restore 代价应上升，`epsilon_norm` 越紧可用低精度空间应收缩”，内部仍以 `epsilon_abs` 驱动策略，但分析和图表统一看 `epsilon_norm`。

## 6.1 最小接口骨架

```python
class SimEngine:
    def __init__(self, device_profile, tier_table, policy):
        self.resident = {}
        self.offloaded = {}
        self.profile = device_profile
        self.tier_table = tier_table
        self.policy = policy

    def replay(self, trace):
        for req in trace:
            obj = self.policy.cop_update(req)
            if obj.id in self.resident:
                event = self.policy.on_hit(obj, self.resident[obj.id])
            elif obj.id in self.offloaded:
                event = self.policy.rrs_on_reuse(obj, self.offloaded[obj.id])
            else:
                event = self.policy.on_miss(obj)
            self.policy.tms_then_lpe(self.resident, self.offloaded)
            self.log(req, event)
```

## 7. 产出物

1. 服务器端和边缘端均可运行的无 `LMCache` 模拟器。
2. 统一 trace、统一配置文件、统一 JSONL 日志格式。
3. 服务器端/边缘端 baseline summary。
4. `H1/H2/H4/H5` 可直接调用的运行脚本或配置模板。

## 8. 通过标准

1. 同一 trace 在服务器端和边缘端均能完整回放。
2. 所有 `COP/TMS/LPE/RRS` 决策字段可记录、可后处理、可复现。
3. ~~关键自变量 `M_budget/BW/epsilon` 能被配置控制，并对指标产生合理方向变化。~~ 修改方案：关键自变量改为 `M_budget/BW/epsilon_abs` 可被配置控制，同时必须证明其对应的 `epsilon_norm` 与指标方向关系合理，作为后续 H4/H5/E2 的统一口径基础。
4. 全程不依赖 `LMCache`。

## 9. 与后续实验衔接

`H0` 通过后，`H1/H2/H4/H5` 均在该环境中验证算法有效性。`H3` 在此基础上接入 `LMCache`，检查策略层能否落到真实 cache layer，并为主实验环境做准备。
