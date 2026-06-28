# TODO: H1 新 trace 参数扫实验

## 目标

验证 `scripts/README.md` 的新判断：当前失败不是 hot working set 不够，而是
`budget_ladder` 的容量压力过于集中；继续放大 hot 只会移动陡崖。

本轮只做 trace 重标定实验：

- 采用 one-factor-at-a-time：每次 trace 只改一个生成参数。
- 每个参数最多 3 个取值。
- 不使用 `h1/备份-运行结果/trace生产参数测试` 的备份数据。
- 不改 LPE/LRU/runner 代码。

## 固定基准 F0

先生成一个新基准 trace `F0`。供给参数一次性放宽，避免后续调整
`scan-hot-objects` 时还要同步改供给参数。

固定生成参数：

| 参数 | 值 |
|---|---:|
| `scan-hot-objects` | 90 |
| `hot-context-words` | 500 |
| `scan-cold-objects` | 24 |
| `cold-context-words` | 800 |
| `scan-probe-rounds` | 3 |
| `hot-repeats` | 4 |
| `rag-hot-repeats` | 4 |
| `random-seed` | 2026 |
| `sharegpt-groups` | 360 |
| `hot-ratio` | 0.34 |
| `rag-requests` | 260 |
| `rag-hot-ratio` | 0.20 |

所有 trace 写到独立目录，不覆盖默认 trace：

```bash
data/edgekv_traces/h1_param_sweep/F0.jsonl
```

生成命令模板：

```bash
python3 scripts/optimize_h0_pressure_trace.py \
  --out data/edgekv_traces/h1_param_sweep/F0.jsonl \
  --sharegpt-groups 360 --hot-ratio 0.34 --hot-repeats 4 \
  --hot-context-words 500 --cold-context-words 800 \
  --scan-hot-objects 90 --scan-cold-objects 24 --scan-probe-rounds 3 \
  --rag-requests 260 --rag-hot-ratio 0.20 --rag-hot-repeats 4 \
  --random-seed 2026
```

生成后先检查 summary：

- `estimated_hot_working_set_tokens`
- `estimated_cold_scan_tokens_per_round`
- `avg_hot_prompt_est_tokens`
- `avg_cold_prompt_est_tokens`

LRU 标定命令模板：

```bash
python3 h1/run_step3_budget_tiers.py \
  --tier F0 \
  --base-out h1/out/trace_param_sweep \
  --replay-trace data/edgekv_traces/h1_param_sweep/F0.jsonl \
  --budgets "0.77 0.80 0.83 0.86 0.87 0.88 0.89 0.90" \
  --policies h1_lru \
  --no-finalize --keep-cells --force
```

实验只跑LRU策略，同一实验的子实验输出放在一个文件夹下，不同实验建立不同文件夹。所有子实验应有总结summary

## 实验 2: hot 单对象粒度

只改 `hot-context-words`，其余参数完全等于 `F0`。

| trace | `hot-context-words` |
|---|---:|
| `H500` | 500 |
| `H425` | 425 |
| `H350` | 350 |

评估预算：

```text
0.77 0.80 0.83 0.86 0.87 0.88 0.89 0.90
```

选择规则：

- 优先选择 `0.77-0.90` 内 distinct LRU hit 档位最多的设置。
- 要求 `hit@0.90 >= 0.85`。

## 实验 3: hot working set 补偿

在实验 2 的优胜设置上，只改 `scan-hot-objects`。

| trace | `scan-hot-objects` |
|---|---:|
| `S90` | 90 |
| `S100` | 100 |
| `S110` | 110 |

目的：验证“更小 hot 对象 + 更多对象”是否比单纯放大 hot 更平缓。

选择规则：

- `hit@0.77` 接近 `0.45-0.55`。
- `hit@0.90` 接近 `0.85-0.95`。
- 相邻 budget 最大跳变尽量小。

## 实验 4: cold floor 修正

只在需要时调整 cold floor。一次只改一个 cold 参数，优先改
`scan-cold-objects`。

- 若 `hit@0.77 > 0.55`：测试 `scan-cold-objects=28`。
- 若 `hit@0.77 < 0.45`：测试 `scan-cold-objects=20`。
- 若 `scan-cold-objects` 调整过粗，再测试 `cold-context-words`。
- `cold-context-words` 最多从 `700/800/900` 中选相关的两个方向值。
- 不同时调整 `scan-cold-objects` 和 `cold-context-words`。

## LRU 验收标准

- `0.77-0.90` 至少 4 个可区分 hit 档位。
- `hit@0.77 = 0.50 ± 0.05`。
- `hit@0.90 = 0.90 ± 0.05`。
- 最大相邻跳变建议 `< 0.20`。
- 若仍出现一步跳到高平台，判定该参数只移动陡崖。
- `qwait_ratio < 0.55`，避免系统排队成为主信号。

## 最终候选

只有 LRU 曲线达标后，才跑最终候选的 LRU/LFU/LPE 对比：

```bash
python3 h1/run_step3_budget_tiers.py \
  --tier <candidate> \
  --base-out h1/out/trace_param_sweep_final \
  --replay-trace data/edgekv_traces/h1_param_sweep/<candidate>.jsonl \
  --budgets "0.77 0.80 0.83 0.86 0.88 0.90" \
  --policies h1_lru h1_lfu h1_lpe \
  --no-finalize --keep-cells --force
```

最终再看 LPE 相对 LRU/LFU 的 hit rate 与 p95 TTFT 优势。

## 记录要求

- 所有 trace 生成到 `data/edgekv_traces/h1_param_sweep/`。
- 所有标定输出写到 `h1/out/trace_param_sweep/`。
- 长任务按异步方式运行，日志写到对应实验目录。
- 每组完成后汇总 CSV/JSON，再决定下一组参数。

## 下一轮判断

如果以上 OFAT 实验仍只有陡崖移动，下一轮不要继续加大 hot。
应改生成器访问结构，例如把当前 `budget_ladder` 拆成分段 ladder：
`cold 小批 -> probe 子集`，让容量压力分层释放。
