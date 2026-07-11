#!/usr/bin/env python3
"""汇总 Step3 repeat 协议：按 point/budget/policy 对多个 rep 取中位数。

同时支持旧 serving-bench aggregate.csv 布局，以及
h1/run_h1_vllm0110_real.py 输出的 pressure-replay summary JSON 布局。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    'p95_ttft_ms',
    'p50_ttft_ms',
    'mean_ttft_ms',
    'hit_rate',
    'gpu_prefix_cache_evictions',
    'gpu_prefix_cache_cached_blocks',
    'queue_wait_ms',
    'prefill_ms',
    'queue_wait_p95_ms',
    'prefill_p95_ms',
    'free_queue_reorder_calls',
    'free_queue_reorder_blocks',
    'free_queue_reorder_skipped',
    'free_queue_reorder_window',
    'free_queue_reorder_time_ms',
    'policy_time_us_avg',
    'eviction_decision_time_us_avg',
    'score_std',
    'p_reuse_std',
    'c_recomp_ms_p50',
    'request_throughput',
    'replay_batch_size',
)


def read_first_row(path: Path) -> dict[str, str]:
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def fnum(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def row_from_real_summary(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    elapsed = fnum(data.get('elapsed_s'))
    requests = fnum(data.get('requests'))
    throughput = requests / elapsed if elapsed > 0 else 0.0
    mapped = {
        'p95_ttft_ms': data.get('ttft_proxy_p95_ms', ''),
        'p50_ttft_ms': data.get('ttft_proxy_p50_ms', ''),
        'mean_ttft_ms': data.get('latency_mean_ms', ''),
        'hit_rate': data.get('hit_rate', ''),
        'gpu_prefix_cache_evictions': data.get('gpu_prefix_cache_evictions', ''),
        'gpu_prefix_cache_cached_blocks': data.get('gpu_prefix_cache_cached_blocks', ''),
        'queue_wait_ms': data.get('queue_wait_ms', ''),
        'prefill_ms': data.get('prefill_ms', ''),
        'queue_wait_p95_ms': data.get('queue_wait_p95_ms', ''),
        'prefill_p95_ms': data.get('prefill_p95_ms', ''),
        'free_queue_reorder_calls': data.get('free_queue_reorder_calls', ''),
        'free_queue_reorder_blocks': data.get('free_queue_reorder_blocks', ''),
        'free_queue_reorder_skipped': data.get('free_queue_reorder_skipped', ''),
        'free_queue_reorder_window': data.get('free_queue_reorder_window', ''),
        'free_queue_reorder_time_ms': data.get('free_queue_reorder_time_ms', ''),
        'policy_time_us_avg': data.get('policy_time_us_avg', ''),
        'eviction_decision_time_us_avg': data.get('eviction_decision_time_us_avg', ''),
        'score_std': data.get('score_std', ''),
        'p_reuse_std': data.get('p_reuse_std', ''),
        'c_recomp_ms_p50': data.get('c_recomp_ms_p50', ''),
        'request_throughput': throughput,
        'replay_batch_size': data.get('replay_batch_size', ''),
    }
    return {key: str(value) for key, value in mapped.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='h1/out/step3_repeat')
    parser.add_argument('--summary', default='h1/out/step3_repeat/step3_repeat_summary.csv')
    args = parser.parse_args()

    base = Path(args.base)
    collected: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for aggregate in sorted(base.glob('*/rep*/*/*/*/aggregate.csv')):
        policy = aggregate.parent.name
        budget = aggregate.parent.parent.name
        point = aggregate.parents[4].name
        source = read_first_row(aggregate)
        for metric in METRICS:
            collected[(point, budget, policy)][metric].append(fnum(source.get(metric)))

    for summary_json in sorted(base.glob('*/rep*/*/*/*_summary.json')):
        policy = summary_json.parent.name
        budget = summary_json.parent.parent.name
        point = summary_json.parents[4].name
        source = row_from_real_summary(summary_json)
        for metric in METRICS:
            collected[(point, budget, policy)][metric].append(fnum(source.get(metric)))

    rows: list[dict[str, str]] = []
    for (point, budget, policy), metric_vals in sorted(collected.items()):
        reps = max((len(v) for v in metric_vals.values()), default=0)
        row = {'point': point, 'budget': budget, 'policy': policy, 'reps': reps}
        for metric in METRICS:
            vals = metric_vals.get(metric, [])
            row[metric] = round(statistics.median(vals), 6) if vals else ''
        rows.append(row)

    fields = ['point', 'budget', 'policy', 'reps', *METRICS]
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f'wrote {summary_path} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
