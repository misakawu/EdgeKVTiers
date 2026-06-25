#!/usr/bin/env python3
"""Aggregate vLLM serving benchmark results with EdgeKV GPU cache stats."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


INT_STAT_KEYS = (
    'lookup_hits',
    'lookup_misses',
    'touches',
    'cached_blocks',
    'evictions',
    'queue_reorders',
    'free_queue_reorder_calls',
    'free_queue_reorder_blocks',
    'free_queue_reorder_skipped',
    'free_queue_reorder_window',
    'admissions',
    'admission_rejections',
    'evict_high_reuse',
    'evict_offload',
    'evict_drop',
    'evicted_score_count',
    'evicted_p_reuse_count',
    'low_score_evictions',
    'hot_prefix_evictions',
    'pinned_eviction_attempts',
    'lpe_profile_count',
    'policy_timing_samples',
    'cop_object_profile_count',
    'block_object_mapping_count',
    'cop_resident_size_bytes_total',
    'cop_resident_block_count_total',
    'kv_cache_group_count',
)

FLOAT_STAT_KEYS = (
    'cop_resident_size_mb_total',
    'cop_resident_size_mb_avg',
    'policy_time_ms_total',
    'free_queue_reorder_time_ms',
    'evicted_score_avg',
    'evicted_p_reuse_avg',
    'score_min',
    'score_p50',
    'score_p95',
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {'data': data}


def metric(data: Any, names: tuple[str, ...]) -> Any:
    if isinstance(data, dict):
        lowered = {str(k).lower(): v for k, v in data.items()}
        for name in names:
            if name.lower() in lowered:
                return lowered[name.lower()]
        for value in data.values():
            found = metric(value, names)
            if found not in (None, ''):
                return found
    elif isinstance(data, list):
        for value in data:
            found = metric(value, names)
            if found not in (None, ''):
                return found
    return None


def fmetric(data: dict[str, Any], names: tuple[str, ...]) -> float:
    value = metric(data, names)
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def merge_stats(stats_dir: Path) -> dict[str, Any]:
    sums = {key: 0 for key in INT_STAT_KEYS}
    float_sums = {key: 0.0 for key in FLOAT_STAT_KEYS}
    weighted_p_reuse = 0.0
    weighted_score = 0.0
    weighted_evicted_score = 0.0
    weighted_evicted_p_reuse = 0.0
    score_mins: list[float] = []
    score_p50s: list[float] = []
    score_p95s: list[float] = []
    page_size_mins: list[int] = []
    page_size_maxs: list[int] = []
    files = 0
    for path in sorted(stats_dir.glob('edgekv_gpu_stats_*.json')):
        try:
            row = load_json(path)
        except Exception:
            continue
        files += 1
        for key in INT_STAT_KEYS:
            sums[key] += int(row.get(key, 0) or 0)
        for key in FLOAT_STAT_KEYS:
            float_sums[key] += float(row.get(key, 0.0) or 0.0)
        page_min = int(row.get('kv_cache_page_size_bytes_min', 0) or 0)
        page_max = int(row.get('kv_cache_page_size_bytes_max', 0) or 0)
        if page_min:
            page_size_mins.append(page_min)
        if page_max:
            page_size_maxs.append(page_max)
        count = int(row.get('lpe_profile_count', 0) or 0)
        weighted_p_reuse += float(row.get('avg_p_reuse', 0.0) or 0.0) * count
        weighted_score += float(row.get('avg_score', 0.0) or 0.0) * count
        evicted_score_count = int(row.get('evicted_score_count', 0) or 0)
        evicted_p_reuse_count = int(row.get('evicted_p_reuse_count', 0) or 0)
        weighted_evicted_score += float(row.get('evicted_score_avg', 0.0) or 0.0) * evicted_score_count
        weighted_evicted_p_reuse += float(row.get('evicted_p_reuse_avg', 0.0) or 0.0) * evicted_p_reuse_count
        if count:
            score_mins.append(float(row.get('score_min', 0.0) or 0.0))
            score_p50s.append(float(row.get('score_p50', 0.0) or 0.0))
            score_p95s.append(float(row.get('score_p95', 0.0) or 0.0))

    lookups = sums['lookup_hits'] + sums['lookup_misses']
    return {
        **sums,
        **{key: round(value, 6) for key, value in float_sums.items()},
        'lookup_total': lookups,
        'hit_rate': round(sums['lookup_hits'] / lookups, 6) if lookups else 0.0,
        'avg_p_reuse': round(weighted_p_reuse / sums['lpe_profile_count'], 6)
        if sums['lpe_profile_count'] else 0.0,
        'avg_score': round(weighted_score / sums['lpe_profile_count'], 9)
        if sums['lpe_profile_count'] else 0.0,
        'policy_time_us_avg': round(
            (float_sums['policy_time_ms_total'] * 1000.0) / sums['policy_timing_samples'],
            6,
        ) if sums['policy_timing_samples'] else 0.0,
        'evicted_score_avg': round(
            weighted_evicted_score / sums['evicted_score_count'], 9,
        ) if sums['evicted_score_count'] else 0.0,
        'evicted_p_reuse_avg': round(
            weighted_evicted_p_reuse / sums['evicted_p_reuse_count'], 6,
        ) if sums['evicted_p_reuse_count'] else 0.0,
        'score_min': round(min(score_mins), 9) if score_mins else 0.0,
        'score_p50': round(sum(score_p50s) / len(score_p50s), 9) if score_p50s else 0.0,
        'score_p95': round(max(score_p95s), 9) if score_p95s else 0.0,
        'kv_cache_page_size_bytes_min': min(page_size_mins) if page_size_mins else 0,
        'kv_cache_page_size_bytes_max': max(page_size_maxs) if page_size_maxs else 0,
        'stats_files': files,
    }


def summarize_cell(cell_dir: Path) -> dict[str, Any]:
    bench = load_json(cell_dir / 'result.json')
    stats = merge_stats(cell_dir / 'stats')
    budget = cell_dir.name
    summary = {
        'budget': budget,
        'ok': bool(bench),
        'p50_ttft_ms': fmetric(bench, ('p50_ttft_ms', 'median_ttft_ms', 'ttft_p50_ms')),
        'p95_ttft_ms': fmetric(bench, ('p95_ttft_ms', 'ttft_p95_ms')),
        'p99_ttft_ms': fmetric(bench, ('p99_ttft_ms', 'ttft_p99_ms')),
        'mean_ttft_ms': fmetric(bench, ('mean_ttft_ms', 'avg_ttft_ms')),
        'p95_e2el_ms': fmetric(bench, ('p95_e2el_ms', 'e2el_p95_ms', 'p95_latency_ms')),
        'request_throughput': fmetric(bench, ('request_throughput', 'requests_per_second')),
        'output_throughput': fmetric(bench, ('output_throughput', 'output_tokens_per_second')),
        'total_input_tokens': fmetric(bench, ('total_input_tokens',)),
        'total_output_tokens': fmetric(bench, ('total_output_tokens',)),
        'result_json': str(cell_dir / 'result.json'),
        'stats_dir': str(cell_dir / 'stats'),
    }
    summary.update({
        'gpu_prefix_cache_lookup_total': stats['lookup_total'],
        'gpu_prefix_cache_lookup_hits': stats['lookup_hits'],
        'gpu_prefix_cache_lookup_misses': stats['lookup_misses'],
        'hit_rate': stats['hit_rate'],
        'gpu_prefix_cache_evictions': stats['evictions'],
        'gpu_prefix_cache_cached_blocks': stats['cached_blocks'],
        'gpu_prefix_cache_touches': stats['touches'],
        'gpu_prefix_cache_queue_reorders': stats['queue_reorders'],
        'free_queue_reorder_calls': stats.get('free_queue_reorder_calls', 0),
        'free_queue_reorder_blocks': stats.get('free_queue_reorder_blocks', 0),
        'free_queue_reorder_skipped': stats.get('free_queue_reorder_skipped', 0),
        'free_queue_reorder_window': stats.get('free_queue_reorder_window', 0),
        'free_queue_reorder_time_ms': stats.get('free_queue_reorder_time_ms', 0.0),
        'gpu_prefix_cache_high_reuse_evictions': stats['evict_high_reuse'] + stats['evict_offload'],
        'gpu_prefix_cache_drop_evictions': stats['evict_drop'],
        'evicted_score_avg': stats.get('evicted_score_avg', 0.0),
        'evicted_p_reuse_avg': stats.get('evicted_p_reuse_avg', 0.0),
        'low_score_evictions': stats.get('low_score_evictions', 0),
        'hot_prefix_evictions': stats.get('hot_prefix_evictions', 0),
        'pinned_eviction_attempts': stats.get('pinned_eviction_attempts', 0),
        'score_min': stats.get('score_min', 0.0),
        'score_p50': stats.get('score_p50', 0.0),
        'score_p95': stats.get('score_p95', 0.0),
        'avg_p_reuse': stats['avg_p_reuse'],
        'avg_score': stats['avg_score'],
        'lpe_profile_count': stats['lpe_profile_count'],
        'policy_timing_samples': stats['policy_timing_samples'],
        'policy_time_ms_total': stats['policy_time_ms_total'],
        'policy_time_us_avg': stats['policy_time_us_avg'],
        'cop_object_profile_count': stats['cop_object_profile_count'],
        'block_object_mapping_count': stats['block_object_mapping_count'],
        'cop_resident_size_bytes_total': stats['cop_resident_size_bytes_total'],
        'cop_resident_size_mb_total': stats['cop_resident_size_mb_total'],
        'cop_resident_size_mb_avg': stats['cop_resident_size_mb_avg'],
        'cop_resident_block_count_total': stats['cop_resident_block_count_total'],
        'kv_cache_group_count': stats['kv_cache_group_count'],
        'kv_cache_page_size_bytes_min': stats['kv_cache_page_size_bytes_min'],
        'kv_cache_page_size_bytes_max': stats['kv_cache_page_size_bytes_max'],
        'stats_files': stats['stats_files'],
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--budgets', nargs='+', default=['tight', 'mid', 'loose'])
    args = parser.parse_args()

    out_dir = Path(args.out)
    rows = [summarize_cell(out_dir / budget) for budget in args.budgets]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    csv_path = out_dir / 'aggregate.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({'aggregate_csv': str(csv_path), 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
