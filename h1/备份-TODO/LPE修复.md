# 彻底禁用 LPE runtime monitor + 方案三历史记录

## Context(为什么改)

LPE 的诊断 monitor(`_edgekv_record_lpe_monitor` → `runtime_monitor.jsonl`)默认对 h1_lpe 开启,且每条记录同步 `mkdir+open+write+close`,是 `eviction_decision_time`/`policy_time` 虚高、与 LFU/LRU 口径不公平的主要来源之一。决定:**彻底禁用 monitor 逐事件 trace,只保留评估策略用的核心指标**。

关键前提(已核实):TTFT/prefill/每请求 hit 来自 `run_h1_vllm0110_real.py` 的 `RequestOutput.metrics`(写 `*_requests.csv`/`*_summary.json`);hit_rate/eviction_decision_time/policy_time/`evicted_score_avg`/`score_p50` 来自 `sitecustomize` flush(写 `edgekv_gpu_stats/`);显存来自 `*_gpu_memory_samples.csv`。**这些全部不经过 monitor**。禁用 monitor 只让 `runtime_monitor.jsonl` 这一个产物消失,核心指标一律保留。

同时收尾:上一轮方案三(`promote_head` 修复驱逐方向)已实现。后续 H1 LPE 只保留 `promote_head` 作为正式执行路径,旧 `reorder_tail` 方向错误,不再作为 runner 或文档推荐的 A/B 策略。

预期结果:正式跑零 monitor overhead;`evicted_score_avg vs score_p50` 作方向验收(常开零成本);需要 per-event 时用 `EDGEKV_H1_RUNTIME_MONITOR=1` 临时开一次。

## 改动明细(3 个 py 文件)

### 文件 1:`h1/sitecustomize.py`

**改动 A — 新增进程级 monitor 总闸(默认关)**
在 `_edgekv_env_bool`(262 定义)之后、`_edgekv_record_lpe_monitor`(288)之前(约 286 行,紧接 `_edgekv_lpe_monitor_path` 之后)新增模块级常量:
```python
# monitor 总闸:进程启动求值一次。默认关 → 正式跑零 monitor 开销;
# 只认 EDGEKV_H1_RUNTIME_MONITOR(不受 RUNTIME_MONITOR_PATH 影响),
# 临时开时用 EDGEKV_H1_RUNTIME_MONITOR=1(需在子进程 import 前由 env 传入)。
_EDGEKV_LPE_MONITOR_ENABLED = _edgekv_env_bool('EDGEKV_H1_RUNTIME_MONITOR', False)
```
作用:统一的、免疫 PATH 陷阱的开关。import 时求值,record 热路径只读一个 bool 全局,零成本。

**改动 B — `_edgekv_record_lpe_monitor` 首部加总闸早退**
在现有 `if not _EDGEKV_GPU_POLICY_IS_LPE: return`(297-298)之后插入:
```python
    if not _EDGEKV_LPE_MONITOR_ENABLED:
        return
```
作用:总闸关时 O(1) 早退,不读 env(`_edgekv_lpe_monitor_path`)、不构造 record dict、不落盘。

**改动 C — 固定方案三 victim 路径**
`get_new_blocks` wrapper 固定调用 `_edgekv_promote_lowest_score_victims`。旧 `reorder_tail` 逻辑方向错误,不再由 `EDGEKV_H1_LPE_VICTIM_MODE` 暴露为正式策略开关。
作用:始终把低分 victim 搬到 free queue 头部,让 vLLM `popleft_n()` 实际驱逐它们。

### 文件 2:`h1/_runner.py`

`lpe_runtime_env_overrides`(51-60):删掉两行注入 —
```python
        "EDGEKV_H1_RUNTIME_MONITOR": "1",              # 删
        "EDGEKV_H1_RUNTIME_MONITOR_PATH": str(...),    # 删
```
**保留** `"EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES": "1"`(喂验收用的 `score_p50/score_min`,来自 `_EDGEKV_GPU_LPE_PROFILES`)。
作用:堵住 `run_test → run_step3 → _runner` 这条路径的 monitor 默认开启源头。

### 文件 3:`h1/run_h1_vllm0110_real.py`

`if policy == 'h1_lpe':` 块(731-739):删掉 monitor 相关 —
```python
        os.environ.setdefault('EDGEKV_H1_RUNTIME_MONITOR', '1')     # 删(733)
        monitor_path = ( ... )                                      # 删(734-738)
        os.environ.setdefault('EDGEKV_H1_RUNTIME_MONITOR_PATH', ...)# 删(739)
```
**保留** `os.environ.setdefault('EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES', '1')`(732)。
作用:堵住第二个源头(最终 python 入口的 setdefault)。`monitor_path` 变量删除安全(仅 734/739 引用,之后不再用)。

## 保留 / 不动(边界)

- `_maybe_evict_cached_block` 里 `_edgekv_block_profile` 查询与 `evicted_score_avg` 累加器:**保留**,验收方向要用,且只累加进 gpu_stats、不写 monitor。
- 所有 serving/requests/memory/gpu_stats 记录:本就独立于 monitor,不动。
- `_edgekv_record_lpe_monitor` 函数体与各调用点(touch/evict/admit/lookup):代码保留,仅被总闸短路,便于临时开与回退。
- admission 的 `_edgekv_min_resident_object_score` O(N) 扫描等其它 overhead:本次不涉及。

## 临时开 monitor(验收用)

正式跑:什么都不设 → 总闸 False → 无 monitor。
临时开一次:`EDGEKV_H1_RUNTIME_MONITOR=1 python h1/run_test.py --policies h1_lpe --budgets 0.75`
- `EDGEKV_H1_` 前缀被 `_runner.py:78` 白名单透传到 cell 子进程;子进程 import `sitecustomize` 时总闸求值 True。
- path 无需手动给:`_edgekv_lpe_monitor_path` 在 RUNTIME_MONITOR 为真且未设 PATH 时 fallback 到 `stats_dir/edgekv_lpe_runtime_monitor.jsonl`。
- 注意:必须设 `RUNTIME_MONITOR=1`(不能只设 PATH),因为总闸只认前者。

## 验收方式

- **方向**(方案三是否修复):每个 cell 的 `edgekv_gpu_stats/` 里 `evicted_score_avg` 应 < 全体 `score_p50`(旧 reorder_tail 是 110.4 > p50,反的)。常开零成本。
- **策略表现**:TTFT/prefill/hit(requests.csv/summary)、hit_rate/eviction_decision_time/policy_time(gpu_stats)、显存(memory_samples)。
- **方向**:默认路径固定为 `promote_head`;不再用 `reorder_tail` 作为正式 A/B 对照。

## 下游影响

- `validate_d3.py:120` 有 `if monitor_path.exists()` 保护 → 禁用后 `runtime_monitor_events=0`,不报错,D3 实质校验(c_recomp 线性/score 分布,来自 object_profiles)不受影响。
- 无其它代码强依赖 `runtime_monitor.jsonl` 存在。

## Verification(执行后)

1. 语法:`python3 -m py_compile h1/sitecustomize.py h1/_runner.py h1/run_h1_vllm0110_real.py`。
2. 默认关验证(小样本):`python h1/run_test.py --policies h1_lpe --budgets 0.75 --num-prompts <小值>`,确认 cell 目录**不生成** `runtime_monitor.jsonl`,但 `*_requests.csv`/`*_summary.json`/`edgekv_gpu_stats/` 齐全,且 gpu_stats 含 `evicted_score_avg`、`score_p50`。
3. 临时开验证:同命令前加 `EDGEKV_H1_RUNTIME_MONITOR=1`,确认 `runtime_monitor.jsonl` 重新出现。
4. 方向验收(方案三):确认默认下 `evicted_score_avg < score_p50` 且 0.75 档 hit_rate 回升。
