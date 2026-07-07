#!/usr/bin/env python3
"""对比两次 LPE 实验的确定性统计字段，判断行为是否不变。

只比较由 trace + 策略逻辑决定的确定性字段；排除时间相关字段
(policy_time_*、TTFT、reorder_time、pid)。退出码 0 = 一致，非 0 = 有差异。

用法: python h1/tools/compare_lpe_stats.py <baseline> <candidate>
参数可为单个 json 文件，或包含 edgekv_gpu_stats_*.json 的目录。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DETERMINISTIC_FIELDS = [
    "hit_rate", "native_hit_rate", "native_hits", "native_queries", "native_requests",
    "block_lookup_hit_rate", "lookup_hits", "lookup_misses", "lookup_total",
    "evictions", "touches", "cached_blocks",
    "admissions", "admission_accepts", "admission_rejections",
    "lpe_profile_count", "cop_object_profile_count",
    "avg_score", "avg_p_reuse", "score_p50", "score_p95", "score_mean",
    "evicted_score_avg", "evicted_p_reuse_avg", "evicted_score_count", "evicted_p_reuse_count",
    "low_score_evictions", "hot_prefix_evictions", "queue_reorders",
]

_SUM_FIELDS = {
    "native_hits", "native_queries", "native_requests", "lookup_hits", "lookup_misses",
    "evictions", "touches", "cached_blocks", "admissions", "admission_accepts",
    "admission_rejections", "low_score_evictions", "hot_prefix_evictions", "queue_reorders",
    "evicted_score_count", "evicted_p_reuse_count",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(target: Path) -> dict:
    if target.is_file():
        return _load(target)
    files = sorted(target.rglob("edgekv_gpu_stats_*.json"))
    if not files:
        summ = sorted(target.rglob("*_summary.json"))
        if summ:
            return _load(summ[-1])
        raise SystemExit(f"未在 {target} 找到 edgekv_gpu_stats_*.json 或 *_summary.json")
    merged: dict = {}
    for f in files:
        data = _load(f)
        for k in _SUM_FIELDS:
            if k in data:
                merged[k] = merged.get(k, 0) + data[k]
    if len(files) == 1:
        one = _load(files[0])
        for k in DETERMINISTIC_FIELDS:
            if k not in _SUM_FIELDS and k in one:
                merged[k] = one[k]
    return merged


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    base = _aggregate(Path(sys.argv[1]))
    cand = _aggregate(Path(sys.argv[2]))
    diffs = []
    for field in DETERMINISTIC_FIELDS:
        if field not in base and field not in cand:
            continue
        a, b = base.get(field), cand.get(field)
        if a != b:  # 严格相等：迁移只移位置不改算式，应逐位相同
            diffs.append((field, a, b))
    if diffs:
        print("行为已改变 —— 确定性字段存在差异：")
        for field, a, b in diffs:
            print(f"  {field}: baseline={a!r}  candidate={b!r}")
        return 1
    print("行为不变 —— 所有确定性字段一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
