# TODO：让 budget–hit 曲线从「台阶」变「平缓斜坡」

## 目标
让 `data/edgekv_traces/sharegpt_hotpotqa_session.jsonl` 的 budget–hit 曲线平缓爬升：
**budget≈0.77 时 hit≈0.5，budget≈0.90 时 hit≈0.9（≈结构天花板）**，中间各档落在 0.6/0.7/0.8 附近。
当前曲线极陡：0.77 时 hit≈0.42–0.51，到 0.80 直接饱和到 ~0.903，0.80–0.90 全平
（见 `h1/备份-运行结果/新trace-budget区间0.77-0.90/step3_summary.csv`）。

这是一次 **trace 重标定**（改本目录 `optimize_h0_pressure_trace.py` 的生成参数 + 经验迭代），
**不改 LPE/LRU 策略代码**，也不改 runner。

## 根因（已定量验证）
`optimize_h0_pressure_trace.py` 的 `budget_ladder` 结构：prime N 个 hot 前缀 →
每轮「扫 cold（约 19k token）→ 复测全部 hot」共 3 轮。每轮 hit 取决于 cold 扫描后有多少 hot 前缀在 LRU 里幸存。

- hot working set ≈ **9.6k** token（summary `estimated_hot_working_set_tokens`）
- cold 单轮 ≈ **19k** token（`estimated_cold_scan_tokens_per_round`）
- KV 容量（Qwen2.5-7B / TP=2 / 11GB×2，budget 直接当 gpu_memory_utilization）粗算：
  **0.77≈17k token，0.80≈30k，0.90≈71k**

只有两个区间、几乎无过渡：0.77 时 cold(19k) 装不下 → hot 全被挤掉（floor）；
0.80 时 cold+hot≈29k 全部装下 → hot 全幸存（天花板）。台阶卡在 0.77→0.80。

**决定 ramp 宽度的唯一关键量：hot working set 的绝对 token 数。**
ramp 宽度 ≈ hot_working_set_tokens ÷ (每单位 budget 的 KV token 增量)。现在 9.6k 太小 → 压成一个台阶。
要铺满 0.77→0.90（KV 从 ~17k 到 ~71k，Δ≈54k），需把 hot working set 放大到 **~45–50k token（约 4–5×）**，
同时保持 cold 扫描大致不变以锚住 0.77 的 floor。hot 前缀数量越多，斜坡颗粒越细。

## 方案：只调 trace 生成参数（已有 CLI 旋钮，无需改代码）

第一版参数（v1）：

| 参数 | 现值 | v1 | 作用 |
|---|---|---|---|
| `--scan-hot-objects` | 35 | **70** | hot 前缀更多 → 斜坡颗粒更细、ramp 更宽 |
| `--hot-context-words` | 300 | **500** | 单个 hot 前缀更大 → 抬高 hot working set |
| `--sharegpt-groups` | 96 | **200** | 提供足够多 hot/cold 候选对象 |
| `--hot-ratio` | 0.20 | **0.30** | sharegpt hot≈60 |
| `--rag-requests` | 128 | **160** | rag hot≈20，凑够 ≥70 hot keys |
| `--scan-cold-objects` | 24 | **24（保持）** | cold 扫描量不变 → 锚住 0.77 floor≈0.5 |
| `--cold-context-words` | 800 | **800（保持）** | 同上 |
| `--scan-probe-rounds` | 3 | **3（保持）** | — |

预期：hot working set 9.6k → ~45k（70×~650token），把饱和点从 0.80 顶到 ~0.90；cold 不变，0.77 仍 ramp 下段。
生成约束已核对可满足（hot keys 80≥70；unique cold 220 ≥ scan-cold-objects×rounds=72；hot 前缀 ≤2048 max_model_len）。

> 生成器对短 session 会按实际长度截断，真实 hot working set 可能 < 估算。
> **生成后必须读 summary 的 `estimated_hot_working_set_tokens` 与 `estimated_cold_scan_tokens_per_round` 核对**，作为下一轮反馈。

## 执行步骤

**Step 0 — 备份现有 trace**（生成器默认会清理旧 trace）
```bash
cp data/edgekv_traces/sharegpt_hotpotqa_session.jsonl* data/edgekv_traces/备份-实验数据/
```

**Step 1 — 重生成 trace（异步）**
```bash
python3 scripts/optimize_h0_pressure_trace.py \
  --out data/edgekv_traces/sharegpt_hotpotqa_session.jsonl \
  --sharegpt-groups 200 --hot-ratio 0.30 --hot-repeats 4 \
  --hot-context-words 500 --cold-context-words 800 \
  --scan-hot-objects 70 --scan-cold-objects 24 --scan-probe-rounds 3 \
  --rag-requests 160 --rag-hot-ratio 0.20 --rag-hot-repeats 4
```
生成后读 `*.summary.json` 确认 hot working set ≈ 40–50k、cold/round ≈ 18–20k。

**Step 2 — 跑 6 点 LRU 标定曲线（异步，复用现有脚本）**
```bash
python3 h1/run_find_interval_2.py --visible-devices 0,1
```
默认粗扫 `0.77/0.80/0.83/0.86/0.88/0.90`，每档 LRU+LPE。
看 `h1/out/find_interval_2/find_interval_2_report.csv` 的 LRU hit 列，画 budget→hit。

**Step 3 — 经验迭代（2–4 轮，调参规则）**
- **饱和太早**（如 0.83 就到 0.9）→ hot working set 不够大：`--scan-hot-objects` ↑ 或 `--hot-context-words` ↑。
- **饱和不到 0.9**（0.90 仍 <0.85）→ 反向略减 hot working set。
- **0.77 floor 偏高/偏低**（≠0.5）→ 调 cold：偏高则 `--scan-cold-objects`/`--cold-context-words` ↑，偏低则 ↓。
- **斜坡有跳变** → `--scan-hot-objects` ↑ 增颗粒度。
每轮只回 Step 1+2。

## 验收标准
- LRU hit 随 budget **单调非降**，0.77→0.90 至少 4 个中间档可区分（不是两段平台）。
- budget≈0.77 → hit≈0.5（±0.05）；budget≈0.90 → hit≈0.9（≈该 trace 结构天花板，±0.05）。
- queue_wait/p95 比例不显著恶化（report 里 `qwait_ratio` 保持 <~0.55，避免饱和区伪信号）。

## 风险 / 注意
- budget→KV-token 斜率只能经验标定（11GB 卡、引擎 floor≈0.72），v1 数值是起点而非终点，需 2–4 轮迭代。
- 放大 hot 会抬高结构天花板（可命中请求变多），"0.9" 语义随之微移——以「0.90 档≈饱和」为准，而非固定 0.903。
- 数据可得性：`--scan-hot-objects 70` 需足够多够长的 sharegpt session；若报「only built N objects」则下调对象数或上调 `--sharegpt-groups`。
- trace 变大 → replay 变慢，可接受；标定一律后台异步、完成再唤醒。
- 跑 LPE 对比时启动环境须注入 `EDGEKV_H1_GPU_POLICY`（`run_find_interval_2.py` 已封装，无需手动）。

## 关键文件
- `scripts/optimize_h0_pressure_trace.py` — trace 生成器（只调 CLI 参数，不改码）
- `data/edgekv_traces/sharegpt_hotpotqa_session.jsonl(.summary.json)` — 产物 + 反馈信号
- `h1/run_find_interval_2.py` → `h1/run_step3_budget_tiers.py` → `h1/run_h1_vllm0110_real.py` — 标定运行链（不改）
