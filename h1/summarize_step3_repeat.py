#!/usr/bin/env python3
"""Summarize the Step3 repeat protocol: median across reps per (point, budget, policy).

Reads <base>/<point>/rep*/<point>/<budget>/<policy>/aggregate.csv (the layout produced by
run_step3_repeat_protocol.sh, which drives run_step3_budget_tiers.sh once per rep) and writes
one concise CSV with the median of each core TTFT / cache-policy metric across reps. No path
columns.
"""

from __future__ import annotations

import argparse
import csv
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
    'free_queue_reorder_time_ms',
    'policy_time_us_avg',
    'request_throughput',
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='h1/out/step3_repeat')
    parser.add_argument('--summary', default='h1/out/step3_repeat/step3_repeat_summary.csv')
    args = parser.parse_args()

    base = Path(args.base)
    # key -> metric -> [values across reps]
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
