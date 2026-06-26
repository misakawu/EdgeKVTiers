#!/usr/bin/env python3
"""Validate H1 D3 diagnostics from EdgeKV GPU stats JSON files.

Checks the three D3 questions from H1_复盘与重跑配置单:
1. c_recomp is linear in n_tokens.
2. eviction granularity is vLLM prefix-cache block level.
3. score and p_reuse distributions/correlation show the expected degeneration.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def corrcoef(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    cov = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denom = math.sqrt(left_var * right_var)
    return cov / denom if denom > 0.0 else 0.0


def histogram(values: list[float], buckets: int = 10) -> dict[str, Any]:
    if not values:
        return {'min': 0.0, 'max': 0.0, 'buckets': []}
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {'min': low, 'max': high, 'buckets': [{'lo': low, 'hi': high, 'count': len(values)}]}
    bucket_count = max(int(buckets), 1)
    width = (high - low) / bucket_count
    counts = [0] * bucket_count
    for value in values:
        idx = min(int((value - low) / width), bucket_count - 1)
        counts[idx] += 1
    return {
        'min': low,
        'max': high,
        'buckets': [
            {'lo': low + idx * width, 'hi': low + (idx + 1) * width, 'count': count}
            for idx, count in enumerate(counts)
        ],
    }


def load_stats(stats_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    paths = sorted(stats_dir.glob('edgekv_gpu_stats_*.json'))
    if not paths:
        paths = sorted(stats_dir.glob('**/edgekv_gpu_stats_*.json'))
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            print(f'[warn] skip unreadable stats file {path}: {exc}')
            continue
        rows.append(row)
        object_profiles = row.get('object_profiles', {})
        if isinstance(object_profiles, dict):
            for object_id, profile in object_profiles.items():
                if isinstance(profile, dict):
                    profiles.append({'object_id': str(object_id), **profile})
    return rows, profiles


def validate(stats_dir: Path, out_json: Path | None = None) -> dict[str, Any]:
    rows, profiles = load_stats(stats_dir)
    scores = [fnum(profile.get('score')) for profile in profiles]
    p_reuses = [fnum(profile.get('p_reuse'), 0.5) for profile in profiles]
    n_tokens = [fnum(profile.get('n_tokens'), 1.0) for profile in profiles]
    c_recomps = [fnum(profile.get('c_recomp_ms', profile.get('c_recomp'))) for profile in profiles]
    size_sources: dict[str, int] = {}
    for profile in profiles:
        source = str(profile.get('size_source') or 'unknown')
        size_sources[source] = size_sources.get(source, 0) + 1
    ratios = [c_recomp / n_token for n_token, c_recomp in zip(n_tokens, c_recomps) if n_token > 0]
    granularities = sorted({str(row.get('eviction_granularity', '')) for row in rows if row.get('eviction_granularity')})
    block_mapping_count = sum(int(row.get('block_object_mapping_count', 0) or 0) for row in rows)
    runtime_monitor_events = 0
    monitor_path = stats_dir.parent / 'runtime_monitor.jsonl'
    if monitor_path.exists():
        runtime_monitor_events = sum(1 for _ in monitor_path.open(encoding='utf-8'))

    result = {
        'stats_dir': str(stats_dir),
        'stats_files': len(rows),
        'object_profile_samples': len(profiles),
        'c_recomp_per_token_mean': mean(ratios),
        'c_recomp_per_token_std': stddev(ratios),
        'c_recomp_linear_ok': bool(ratios and stddev(ratios) < 1e-5),
        'eviction_granularity_values': granularities,
        'block_object_mapping_count_total': block_mapping_count,
        'runtime_monitor_events': runtime_monitor_events,
        'block_granularity_ok': 'vllm_prefix_cache_block' in granularities,
        'score_p_reuse_corr': corrcoef(scores, p_reuses),
        'score_mean': mean(scores),
        'score_std': stddev(scores),
        'score_p50': percentile(scores, 50),
        'score_p95': percentile(scores, 95),
        'score_nonzero_ok': bool(percentile(scores, 50) > 0.0),
        'p_reuse_mean': mean(p_reuses),
        'p_reuse_std': stddev(p_reuses),
        'p_reuse_p50': percentile(p_reuses, 50),
        'p_reuse_p95': percentile(p_reuses, 95),
        'score_histogram': histogram(scores),
        'p_reuse_histogram': histogram(p_reuses),
        'size_source_counts': dict(sorted(size_sources.items())),
        'score_p_reuse_degenerate_ok': bool(
            len(scores) > 1 and percentile(scores, 50) > 0.0 and corrcoef(scores, p_reuses) > 0.95
        ),
    }
    if out_json is None:
        out_json = stats_dir.parent / 'd3_validation.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stats_dir', type=Path)
    parser.add_argument('--out-json', type=Path, default=None)
    args = parser.parse_args()
    result = validate(args.stats_dir, args.out_json)

    print('=== D3 validation ===')
    print(f"stats_files = {result['stats_files']}")
    print(f"object_profile_samples = {result['object_profile_samples']}")
    print(
        'c_recomp/n_tokens: '
        f"mean={result['c_recomp_per_token_mean']:.9f}, "
        f"std={result['c_recomp_per_token_std']:.9f}, "
        f"ok={result['c_recomp_linear_ok']}"
    )
    print(
        'eviction_granularity: '
        f"values={result['eviction_granularity_values']}, "
        f"block_mappings={result['block_object_mapping_count_total']}, "
        f"runtime_events={result['runtime_monitor_events']}, "
        f"ok={result['block_granularity_ok']}"
    )
    print(
        'score_vs_p_reuse: '
        f"corr={result['score_p_reuse_corr']:.9f}, "
        f"score_p50={result['score_p50']:.9f}, "
        f"score_std={result['score_std']:.9f}, "
        f"p_reuse_std={result['p_reuse_std']:.9f}, "
        f"nonzero_ok={result['score_nonzero_ok']}, "
        f"ok={result['score_p_reuse_degenerate_ok']}"
    )
    print(f"size_source_counts = {result['size_source_counts']}")
    print(f"wrote {Path(args.out_json) if args.out_json else args.stats_dir.parent / 'd3_validation.json'}")


if __name__ == '__main__':
    main()
