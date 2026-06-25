#!/usr/bin/env python3
"""Summarize H1 step3 budget-tier experiment outputs (one tier dir).

Reads <tier_dir>/<budget>/<policy>/aggregate.csv (written by aggregate_h1_serving_bench.py)
and writes one concise CSV: one row per (budget, policy) with core TTFT / cache-policy
metrics plus LPE-vs-LRU p95/mean gains per budget. No path columns.
"""

from __future__ import annotations

import argparse
import csv
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
    'free_queue_reorder_time_ms',
    'policy_time_us_avg',
    'eviction_decision_time_us_avg',
    'score_std',
    'p_reuse_std',
    'c_recomp_ms_p50',
    'request_throughput',
)


def read_first_row(path: Path) -> dict[str, str]:
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, '') or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='h1/out/step3/tight',
                        help='tier directory containing <budget>/<policy>/aggregate.csv')
    parser.add_argument('--summary', default='h1/out/step3/tight/step3_summary.csv')
    parser.add_argument('--request-rate', type=float, default=18.5)
    args = parser.parse_args()

    out_dir = Path(args.out)
    rows: list[dict[str, str]] = []
    by_budget: dict[str, dict[str, dict[str, str]]] = {}
    for aggregate in sorted(out_dir.glob('*/*/aggregate.csv')):
        policy = aggregate.parent.name
        budget = aggregate.parent.parent.name
        source = read_first_row(aggregate)
        by_budget.setdefault(budget, {})[policy] = source
        throughput = as_float(source, 'request_throughput')
        row = {
            'budget': budget,
            'policy': policy,
            'saturated': str(throughput < 0.8 * args.request_rate).lower(),
        }
        for metric in METRICS:
            row[metric] = source.get(metric, '')
        rows.append(row)

    for row in rows:
        if row['policy'] != 'h1_lpe':
            row.update({
                'p95_ttft_gain_pct_vs_lru': '',
                'mean_ttft_gain_pct_vs_lru': '',
                'hit_rate_delta_vs_lru': '',
                'eviction_delta_vs_lru': '',
            })
            continue
        cell = by_budget.get(row['budget'], {})
        lru = cell.get('h1_lru')
        lpe = cell.get('h1_lpe')
        if not lru or not lpe:
            continue
        lru_p95 = as_float(lru, 'p95_ttft_ms')
        lpe_p95 = as_float(lpe, 'p95_ttft_ms')
        lru_mean = as_float(lru, 'mean_ttft_ms')
        lpe_mean = as_float(lpe, 'mean_ttft_ms')
        row['p95_ttft_gain_pct_vs_lru'] = (
            f'{((lru_p95 - lpe_p95) / lru_p95 * 100.0):.6f}' if lru_p95 else ''
        )
        row['mean_ttft_gain_pct_vs_lru'] = (
            f'{((lru_mean - lpe_mean) / lru_mean * 100.0):.6f}' if lru_mean else ''
        )
        row['hit_rate_delta_vs_lru'] = f'{(as_float(lpe, "hit_rate") - as_float(lru, "hit_rate")):.6f}'
        row['eviction_delta_vs_lru'] = (
            f'{(as_float(lpe, "gpu_prefix_cache_evictions") - as_float(lru, "gpu_prefix_cache_evictions")):.6f}'
        )

    fields = [
        'budget',
        'policy',
        *METRICS,
        'p95_ttft_gain_pct_vs_lru',
        'mean_ttft_gain_pct_vs_lru',
        'hit_rate_delta_vs_lru',
        'eviction_delta_vs_lru',
        'saturated',
    ]
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f'wrote {summary_path} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
