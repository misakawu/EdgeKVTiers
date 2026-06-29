#!/usr/bin/env python3
"""Aggregate vLLM serving benchmark results with EdgeKV GPU cache stats."""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


INT_STAT_KEYS = (
    'lookup_hits',
    'lookup_misses',
    'native_queries',
    'native_hits',
    'native_requests',
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
    'eviction_decision_timing_samples',
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
    'eviction_decision_time_ms_total',
    'eviction_decision_time_us_avg',
    'free_queue_reorder_time_ms',
    'evicted_score_avg',
    'evicted_p_reuse_avg',
    'score_min',
    'score_p50',
    'score_p95',
    'score_mean',
    'score_std',
    'p_reuse_min',
    'p_reuse_p50',
    'p_reuse_p95',
    'p_reuse_std',
    'c_recomp_ms_min',
    'c_recomp_ms_p50',
    'c_recomp_ms_p95',
    'c_recomp_ms_per_token',
)

STRING_STAT_KEYS = (
    'c_recomp_model',
    'eviction_granularity',
)

QUEUE_METRIC_HINTS = ('queue_time', 'queue_wait', 'waiting_time', 'scheduler_wait')
PREFILL_METRIC_HINTS = ('prefill_time', 'prompt_time')


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


def scrape_metrics(url: str, out: Path) -> None:
    request = urllib.request.Request(url, method='GET')
    with urllib.request.urlopen(request, timeout=10.0) as response:
        body = response.read().decode('utf-8', errors='replace')
    out.write_text(body, encoding='utf-8')


def parse_prometheus_histograms(text: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[tuple[float, float]]] = {}
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    bucket_re = re.compile(
        r'^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)_bucket\{[^}]*\ble="(?P<le>[^"]+)"[^}]*\}\s+(?P<value>[-+0-9.eE]+)'
    )
    sum_re = re.compile(r'^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)_sum(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)')
    count_re = re.compile(r'^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)_count(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)')
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = bucket_re.match(line)
        if match:
            le_raw = match.group('le')
            le = float('inf') if le_raw == '+Inf' else float(le_raw)
            buckets.setdefault(match.group('name'), []).append((le, float(match.group('value'))))
            continue
        match = sum_re.match(line)
        if match:
            sums[match.group('name')] = sums.get(match.group('name'), 0.0) + float(match.group('value'))
            continue
        match = count_re.match(line)
        if match:
            counts[match.group('name')] = counts.get(match.group('name'), 0.0) + float(match.group('value'))
    result: dict[str, dict[str, Any]] = {}
    for name, values in buckets.items():
        ordered = sorted(values, key=lambda item: item[0])
        total = ordered[-1][1] if ordered else 0.0
        target = total * 0.95
        p95 = 0.0
        for upper, cumulative in ordered:
            if cumulative >= target:
                p95 = upper
                break
        if math_is_inf(p95) and len(ordered) > 1:
            p95 = ordered[-2][0]
        count = counts.get(name, total)
        mean = (sums.get(name, 0.0) / count) if count else 0.0
        result[name] = {'p95_s': p95, 'mean_s': mean, 'count': count}
    return result


def math_is_inf(value: float) -> bool:
    return value == float('inf') or value == float('-inf')


def metric_matches(name: str, hints: tuple[str, ...]) -> bool:
    normalized = name.lower().replace(':', '_')
    return any(hint in normalized for hint in hints)


def split_latency_from_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return {
            'queue_wait_ms': 0.0,
            'prefill_ms': 0.0,
            'queue_wait_p95_ms': 0.0,
            'prefill_p95_ms': 0.0,
            'queue_prefill_source': 'metrics_unavailable',
        }
    histograms = parse_prometheus_histograms(metrics_path.read_text(encoding='utf-8', errors='replace'))
    queue = next((value for name, value in histograms.items() if metric_matches(name, QUEUE_METRIC_HINTS)), None)
    prefill = next((value for name, value in histograms.items() if metric_matches(name, PREFILL_METRIC_HINTS)), None)
    source_parts = []
    if queue:
        source_parts.append('queue_histogram')
    if prefill:
        source_parts.append('prefill_histogram')
    return {
        'queue_wait_ms': round(float(queue.get('mean_s', 0.0)) * 1000.0, 6) if queue else 0.0,
        'prefill_ms': round(float(prefill.get('mean_s', 0.0)) * 1000.0, 6) if prefill else 0.0,
        'queue_wait_p95_ms': round(float(queue.get('p95_s', 0.0)) * 1000.0, 6) if queue else 0.0,
        'prefill_p95_ms': round(float(prefill.get('p95_s', 0.0)) * 1000.0, 6) if prefill else 0.0,
        'queue_prefill_source': '+'.join(source_parts) if source_parts else 'metrics_missing_queue_prefill_histograms',
    }


def merge_stats(stats_dir: Path) -> dict[str, Any]:
    sums = {key: 0 for key in INT_STAT_KEYS}
    float_sums = {key: 0.0 for key in FLOAT_STAT_KEYS}
    strings = {key: '' for key in STRING_STAT_KEYS}
    weighted_p_reuse = 0.0
    weighted_score = 0.0
    weighted_evicted_score = 0.0
    weighted_evicted_p_reuse = 0.0
    score_mins: list[float] = []
    score_p50s: list[float] = []
    score_p95s: list[float] = []
    p_reuse_mins: list[float] = []
    p_reuse_p50s: list[float] = []
    p_reuse_p95s: list[float] = []
    score_stds: list[float] = []
    p_reuse_stds: list[float] = []
    c_recomp_mins: list[float] = []
    c_recomp_p50s: list[float] = []
    c_recomp_p95s: list[float] = []
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
        for key in STRING_STAT_KEYS:
            if not strings[key] and row.get(key):
                strings[key] = str(row.get(key))
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
            p_reuse_mins.append(float(row.get('p_reuse_min', 0.0) or 0.0))
            p_reuse_p50s.append(float(row.get('p_reuse_p50', 0.0) or 0.0))
            p_reuse_p95s.append(float(row.get('p_reuse_p95', 0.0) or 0.0))
            score_stds.append(float(row.get('score_std', 0.0) or 0.0))
            p_reuse_stds.append(float(row.get('p_reuse_std', 0.0) or 0.0))
            c_recomp_mins.append(float(row.get('c_recomp_ms_min', 0.0) or 0.0))
            c_recomp_p50s.append(float(row.get('c_recomp_ms_p50', 0.0) or 0.0))
            c_recomp_p95s.append(float(row.get('c_recomp_ms_p95', 0.0) or 0.0))

    lookups = sums['lookup_hits'] + sums['lookup_misses']
    native_q = sums['native_queries']
    native_h = sums['native_hits']
    native_hit_rate = (native_h / native_q) if native_q else 0.0
    block_hit_rate = (sums['lookup_hits'] / lookups) if lookups else 0.0
    return {
        **sums,
        **{key: round(value, 6) for key, value in float_sums.items()},
        **strings,
        'lookup_total': lookups,
        'native_hit_rate': round(native_hit_rate, 6),
        'block_lookup_hit_rate': round(block_hit_rate, 6),
        'hit_rate': round(native_hit_rate if native_q else block_hit_rate, 6),
        'hit_source': 'vllm_native_token_coverage' if native_q else 'gpu_prefix_cache_block_lookup',
        'avg_p_reuse': round(weighted_p_reuse / sums['lpe_profile_count'], 6)
        if sums['lpe_profile_count'] else 0.0,
        'avg_score': round(weighted_score / sums['lpe_profile_count'], 9)
        if sums['lpe_profile_count'] else 0.0,
        'policy_time_us_avg': round(
            (float_sums['policy_time_ms_total'] * 1000.0) / sums['policy_timing_samples'],
            6,
        ) if sums['policy_timing_samples'] else 0.0,
        'eviction_decision_time_us_avg': round(
            (float_sums['eviction_decision_time_ms_total'] * 1000.0)
            / sums['eviction_decision_timing_samples'],
            6,
        ) if sums['eviction_decision_timing_samples'] else 0.0,
        'evicted_score_avg': round(
            weighted_evicted_score / sums['evicted_score_count'], 9,
        ) if sums['evicted_score_count'] else 0.0,
        'evicted_p_reuse_avg': round(
            weighted_evicted_p_reuse / sums['evicted_p_reuse_count'], 6,
        ) if sums['evicted_p_reuse_count'] else 0.0,
        'score_min': round(min(score_mins), 9) if score_mins else 0.0,
        'score_p50': round(sum(score_p50s) / len(score_p50s), 9) if score_p50s else 0.0,
        'score_p95': round(max(score_p95s), 9) if score_p95s else 0.0,
        'score_std': round(sum(score_stds) / len(score_stds), 9) if score_stds else 0.0,
        'p_reuse_min': round(min(p_reuse_mins), 6) if p_reuse_mins else 0.0,
        'p_reuse_p50': round(sum(p_reuse_p50s) / len(p_reuse_p50s), 6) if p_reuse_p50s else 0.0,
        'p_reuse_p95': round(max(p_reuse_p95s), 6) if p_reuse_p95s else 0.0,
        'p_reuse_std': round(sum(p_reuse_stds) / len(p_reuse_stds), 6) if p_reuse_stds else 0.0,
        'c_recomp_ms_min': round(min(c_recomp_mins), 6) if c_recomp_mins else 0.0,
        'c_recomp_ms_p50': round(sum(c_recomp_p50s) / len(c_recomp_p50s), 6) if c_recomp_p50s else 0.0,
        'c_recomp_ms_p95': round(max(c_recomp_p95s), 6) if c_recomp_p95s else 0.0,
        'kv_cache_page_size_bytes_min': min(page_size_mins) if page_size_mins else 0,
        'kv_cache_page_size_bytes_max': max(page_size_maxs) if page_size_maxs else 0,
        'stats_files': files,
    }


def summarize_cell(cell_dir: Path) -> dict[str, Any]:
    bench = load_json(cell_dir / 'result.json')
    stats = merge_stats(cell_dir / 'stats')
    split_latency = split_latency_from_metrics(cell_dir / 'metrics.txt')
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
    summary.update(split_latency)
    summary.update({
        'gpu_prefix_cache_lookup_total': stats['lookup_total'],
        'gpu_prefix_cache_lookup_hits': stats['lookup_hits'],
        'gpu_prefix_cache_lookup_misses': stats['lookup_misses'],
        'gpu_prefix_cache_native_queries': stats.get('native_queries', 0),
        'gpu_prefix_cache_native_hits': stats.get('native_hits', 0),
        'gpu_prefix_cache_native_requests': stats.get('native_requests', 0),
        'native_hit_rate': stats.get('native_hit_rate', 0.0),
        'block_lookup_hit_rate': stats.get('block_lookup_hit_rate', 0.0),
        'hit_rate': stats['hit_rate'],
        'hit_source': stats.get('hit_source', 'gpu_prefix_cache_block_lookup'),
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
        'score_mean': stats.get('score_mean', 0.0),
        'score_std': stats.get('score_std', 0.0),
        'p_reuse_min': stats.get('p_reuse_min', 0.0),
        'p_reuse_p50': stats.get('p_reuse_p50', 0.0),
        'p_reuse_p95': stats.get('p_reuse_p95', 0.0),
        'p_reuse_std': stats.get('p_reuse_std', 0.0),
        'c_recomp_ms_min': stats.get('c_recomp_ms_min', 0.0),
        'c_recomp_ms_p50': stats.get('c_recomp_ms_p50', 0.0),
        'c_recomp_ms_p95': stats.get('c_recomp_ms_p95', 0.0),
        'c_recomp_ms_per_token': stats.get('c_recomp_ms_per_token', 0.0),
        'c_recomp_model': stats.get('c_recomp_model', ''),
        'eviction_granularity': stats.get('eviction_granularity', ''),
        'avg_p_reuse': stats['avg_p_reuse'],
        'avg_score': stats['avg_score'],
        'lpe_profile_count': stats['lpe_profile_count'],
        'policy_timing_samples': stats['policy_timing_samples'],
        'eviction_decision_timing_samples': stats.get('eviction_decision_timing_samples', 0),
        'policy_time_ms_total': stats['policy_time_ms_total'],
        'policy_time_us_avg': stats['policy_time_us_avg'],
        'eviction_decision_time_ms_total': stats.get('eviction_decision_time_ms_total', 0.0),
        'eviction_decision_time_us_avg': stats.get('eviction_decision_time_us_avg', 0.0),
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
    diagnostics = {
        'cell': str(cell_dir),
        'c_recomp_model': summary['c_recomp_model'],
        'eviction_granularity': summary['eviction_granularity'],
        'score': {
            'min': summary['score_min'],
            'p50': summary['score_p50'],
            'p95': summary['score_p95'],
            'mean': summary['score_mean'],
            'std': summary['score_std'],
        },
        'p_reuse': {
            'min': summary['p_reuse_min'],
            'p50': summary['p_reuse_p50'],
            'p95': summary['p_reuse_p95'],
            'std': summary['p_reuse_std'],
        },
        'c_recomp_ms': {
            'min': summary['c_recomp_ms_min'],
            'p50': summary['c_recomp_ms_p50'],
            'p95': summary['c_recomp_ms_p95'],
            'per_token': summary['c_recomp_ms_per_token'],
        },
    }
    try:
        (cell_dir / 'lpe_diagnostics.json').write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True),
            encoding='utf-8',
        )
    except Exception:
        pass
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out')
    parser.add_argument('--budgets', nargs='+', default=['tight', 'mid', 'loose'])
    parser.add_argument('--scrape-metrics', default='')
    parser.add_argument('--metrics-out', default='')
    args = parser.parse_args()

    if args.scrape_metrics:
        if not args.metrics_out:
            parser.error('--metrics-out is required with --scrape-metrics')
        scrape_metrics(args.scrape_metrics, Path(args.metrics_out))
        print(json.dumps({'metrics_out': args.metrics_out}, indent=2))
        return

    if not args.out:
        parser.error('--out is required unless --scrape-metrics is used')

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
