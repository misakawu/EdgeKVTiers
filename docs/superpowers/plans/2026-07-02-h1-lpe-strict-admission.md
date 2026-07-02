# H1 LPE Strict Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 H1 LPE `strict` admission 在无真实缓存压力时物理拒绝对象的问题，使高 budget 不再因为早期 resident prefix 阈值而误拒绝后续 RAG chunk。

**Architecture:** `strict` admission 分成诊断决策和物理跳过两层：无压力时只保留诊断结果并完整缓存；有压力时才执行对象级 all-or-none physical skip。压力判断复用 `_edgekv_queue_pressure(pool, queue, num_blocks)`，避免引入第二套压力定义。

**Tech Stack:** Python 3.10, vLLM 0.11 runtime monkey patch, pytest, conda env `edgekv-vllm0110`。

---

## 文件结构

- Modify: `h1/sitecustomize.py`
  - 添加 strict admission pressure bypass 统计。
  - 给 `_edgekv_build_strict_admission_plan(...)` 增加 `under_pressure` 参数。
  - 在 `cache_full_blocks(...)` 中调用 `_edgekv_queue_pressure(...)`，只在压力期生成 physical strict plan。
  - 重新计算 pressure-time physical decision，避免 stale diagnostic reject 永久生效。
- Modify: `h1/test_h1_rag_trace.py`
  - 添加 no-pressure bypass regression test。
  - 添加 stale diagnostic reject recompute regression test。
  - 保留 pressure-time skip 与 fallback 测试。
- Modify: `h1/summarize_step3_budget_tiers.py`
  - 输出 `strict_admission_pressure_bypass_count`、`strict_cached_block_skip_count`、`strict_admission_rejection_count`、`strict_admission_fallback_count`。
- Do not modify: `h1/run_test.py`
  - 该文件已有工作区改动，本修复不覆盖。

---

### Task 1: 写 no-pressure strict bypass 回归测试

**Files:**
- Modify: `h1/test_h1_rag_trace.py`

- [ ] **Step 1: 添加失败测试**

在 strict plan 测试区添加：

```python
def test_sitecustomize_strict_admission_bypasses_physical_skip_without_pressure(monkeypatch) -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()
    monkeypatch.setenv("H1_LPE_ADMISSION_MODE", "strict")

    low = sitecustomize._edgekv_profile_from_values(
        "obj-low", "prefix", 16, {"p_reuse": 0.1, "c_recomp_ms": 1.0}
    )
    low["admission_decision"] = "reject"
    low["admission_rejected"] = True

    plan = sitecustomize._edgekv_build_strict_admission_plan(
        [FakeBlock(block_id=30), FakeBlock(block_id=31)],
        [("obj-low", low), ("obj-low", low)],
        pinned=False,
        under_pressure=False,
    )

    assert plan.fallback_reason == ""
    assert plan.accepted_runs == [(0, 2)]
    assert plan.skipped_blocks == 0
    assert low["admission_decision"] == "reject"
    assert low["strict_cache_skipped"] is False
    assert low["admission_mode"] == "diagnostic"
    stats = sitecustomize.get_edgekv_gpu_cache_stats()
    assert stats["strict_admission_pressure_bypass_count"] == 2
    assert stats["strict_cached_block_skip_count"] == 0
```

- [ ] **Step 2: 验证测试先失败**

Run:

```bash
conda run -n edgekv-vllm0110 python -m pytest h1/test_h1_rag_trace.py::test_sitecustomize_strict_admission_bypasses_physical_skip_without_pressure -q
```

Expected: FAIL，原因是 `_edgekv_build_strict_admission_plan()` 尚不接受 `under_pressure` 或无压力时仍 skip。

---

### Task 2: 写 stale diagnostic reject 回归测试

**Files:**
- Modify: `h1/test_h1_rag_trace.py`

- [ ] **Step 1: 添加失败测试**

添加：

```python
def test_sitecustomize_strict_admission_recomputes_stale_diagnostic_reject(monkeypatch) -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()
    monkeypatch.setenv("H1_LPE_ADMISSION_MODE", "strict")

    candidate = sitecustomize._edgekv_profile_from_values(
        "obj-candidate", "prefix", 16, {"p_reuse": 0.1, "c_recomp_ms": 1.0}
    )
    candidate["admission_decision"] = "reject"
    candidate["admission_rejected"] = True

    plan = sitecustomize._edgekv_build_strict_admission_plan(
        [FakeBlock(block_id=40)],
        [("obj-candidate", candidate)],
        pinned=False,
        under_pressure=True,
    )

    assert plan.fallback_reason == ""
    assert plan.accepted_runs == [(0, 1)]
    assert plan.skipped_blocks == 0
    assert candidate["strict_cache_skipped"] is False
    stats = sitecustomize.get_edgekv_gpu_cache_stats()
    assert stats["strict_cached_block_skip_count"] == 0
```

- [ ] **Step 2: 验证测试先失败**

Run:

```bash
conda run -n edgekv-vllm0110 python -m pytest h1/test_h1_rag_trace.py::test_sitecustomize_strict_admission_recomputes_stale_diagnostic_reject -q
```

Expected: FAIL，当前实现会复用旧 `admission_decision='reject'`。

---

### Task 3: 实现 strict pressure bypass

**Files:**
- Modify: `h1/sitecustomize.py`

- [ ] **Step 1: 添加统计字段**

在 `_EDGEKV_GPU_STATS` 增加：

```python
'strict_admission_pressure_bypasses': 0,
```

在 `get_edgekv_gpu_cache_stats()` 增加：

```python
stats['strict_admission_pressure_bypass_count'] = int(
    stats.get('strict_admission_pressure_bypasses', 0) or 0
)
```

- [ ] **Step 2: 增加当前 admission decision helper**

在 `_edgekv_min_resident_object_score(...)` 后添加：

```python
def _edgekv_current_object_admission_decision(profile: dict[str, Any], pinned: bool) -> str:
    object_id = str(profile.get('object_id', ''))
    new_score = float(profile.get('score', 0.0) or 0.0)
    min_resident = _edgekv_min_resident_object_score(exclude_object_id=object_id)
    if pinned or min_resident is None or new_score > min_resident:
        return 'accept'
    return 'reject'
```

- [ ] **Step 3: 给 strict plan 增加 pressure 参数**

修改签名：

```python
def _edgekv_build_strict_admission_plan(
    blocks: list[Any],
    entries: list[tuple[str, dict[str, Any] | None]],
    pinned: bool,
    under_pressure: bool = True,
) -> _EdgeKVStrictAdmissionPlan:
```

在 mode 判断后加入：

```python
if not under_pressure:
    bypassed_blocks = 0
    for block, (_, profile) in zip(blocks, entries):
        if getattr(block, 'is_null', False):
            continue
        bypassed_blocks += 1
        if profile is not None:
            profile['admission_mode'] = 'diagnostic'
            profile['strict_cache_skipped'] = False
            profile['strict_fallback_reason'] = ''
    if bypassed_blocks:
        _edgekv_note_gpu_stat('strict_admission_pressure_bypasses', bypassed_blocks)
    return _EdgeKVStrictAdmissionPlan(accepted_runs=[(0, len(blocks))] if blocks else [])
```

- [ ] **Step 4: pressure-time 不复用 stale reject**

把 strict plan 循环中的 decision 逻辑改为：

```python
previous_decision = str(profile.get('admission_decision', '') or '')
decision = _edgekv_current_object_admission_decision(profile, pinned)
if not previous_decision:
    profile['admission_decision'] = ''
    decision = _edgekv_evaluate_object_admission(profile, pinned) or str(
        profile.get('admission_decision', '') or 'accept'
    )
elif decision != previous_decision:
    profile['admission_decision'] = decision
    profile['admission_rejected'] = (decision == 'reject')
    min_resident = _edgekv_min_resident_object_score(
        exclude_object_id=str(profile.get('object_id', ''))
    )
    profile['admission_min_resident_score'] = (
        float(min_resident) if min_resident is not None else ''
    )
```

- [ ] **Step 5: 在 cache path 中接入统一压力判断**

在 `cache_full_blocks(...)` strict 分支中调用：

```python
strict_under_pressure = _edgekv_queue_pressure(
    self,
    self.free_block_queue,
    len(new_full_blocks),
)
strict_plan = _edgekv_build_strict_admission_plan(
    new_full_blocks,
    lpe_entries,
    pinned,
    under_pressure=strict_under_pressure,
)
```

- [ ] **Step 6: 运行回归测试**

Run:

```bash
conda run -n edgekv-vllm0110 python -m pytest h1/test_h1_rag_trace.py -q
```

Expected: `28 passed`。

---

### Task 4: 汇总 strict 统计

**Files:**
- Modify: `h1/summarize_step3_budget_tiers.py`

- [ ] **Step 1: 扩展 METRICS**

在 `METRICS` 中添加：

```python
'strict_admission_pressure_bypass_count',
'strict_cached_block_skip_count',
'strict_admission_rejection_count',
'strict_admission_fallback_count',
```

- [ ] **Step 2: 映射 real summary JSON 字段**

在 `row_from_real_summary(...)` 的 `mapped` 中添加：

```python
'strict_admission_pressure_bypass_count': data.get('strict_admission_pressure_bypass_count', ''),
'strict_cached_block_skip_count': data.get('strict_cached_block_skip_count', ''),
'strict_admission_rejection_count': data.get('strict_admission_rejection_count', ''),
'strict_admission_fallback_count': data.get('strict_admission_fallback_count', ''),
```

- [ ] **Step 3: 编译检查**

Run:

```bash
conda run -n edgekv-vllm0110 python -m py_compile h1/sitecustomize.py h1/test_h1_rag_trace.py h1/summarize_step3_budget_tiers.py
```

Expected: exit code 0。

---

### Task 5: Strict smoke 验收

**Files:**
- Read: strict smoke 输出目录中的 summary JSON/CSV。
- Do not modify: `h1/run_test.py`。

- [ ] **Step 1: 运行 strict smoke**

Run:

```bash
H1_LPE_ADMISSION_MODE=strict python3 h1/run_test.py --out-dir h1/out/strict_pressure_bypass_smoke --force
```

Expected: run exits 0。

- [ ] **Step 2: 检查 0.95 budget cached blocks**

打开对应 summary，确认：

```text
gpu_prefix_cache_cached_blocks 不再塌缩到 64
strict_admission_pressure_bypass_count > 0
strict_cached_block_skip_count 只在压力期增长
```

- [ ] **Step 3: 生成 step3 summary**

Run:

```bash
python3 h1/summarize_step3_budget_tiers.py --out h1/out/strict_pressure_bypass_smoke --summary h1/out/strict_pressure_bypass_smoke/step3_summary.csv
```

Expected: `step3_summary.csv` 包含以下列：

```text
strict_admission_pressure_bypass_count
strict_cached_block_skip_count
strict_admission_rejection_count
strict_admission_fallback_count
```

---

## 自检清单

- Spec coverage:
  - 无压力不 physical skip: Task 1 + Task 3。
  - 压力期仍 strict skip: Task 3 保留原 strict plan 行为，既有测试覆盖。
  - stale diagnostic reject 不永久生效: Task 2 + Task 3。
  - strict 统计汇总: Task 4。
  - 不修改 `h1/run_test.py`: 文件结构和 Task 5 明确限制。
- Placeholder scan:
  - 无 TBD/TODO/implement later。
  - 每个测试和代码步骤包含具体代码或具体命令。
- Type consistency:
  - 新参数名统一为 `under_pressure`。
  - 新统计源字段为 `strict_admission_pressure_bypasses`。
  - 对外 summary 字段为 `strict_admission_pressure_bypass_count`。

