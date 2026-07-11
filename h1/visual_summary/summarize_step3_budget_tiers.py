#!/usr/bin/env python3
"""汇总 H1 第三步预算档实验输出（单个档目录）。

同时支持旧服务压测布局（<budget>/<policy>/aggregate.csv）和
压力回放布局（<budget>/<policy>/<budget>_<policy>_summary.json）。
写出一份紧凑 CSV，包含核心 TTFT/缓存策略指标以及 LPE 相对 LRU 的收益。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    'p95_ttft_ms',
    'p50_ttft_ms',
    'mean_ttft_ms',
    'replay_batch_size',
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


def row_from_real_summary(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    elapsed = as_float(data, 'elapsed_s')
    requests = as_float(data, 'requests')
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
    parser.add_argument('--out', default='h1/out/step3/tight',
                        help='tier directory containing <budget>/<policy>/aggregate.csv')
    parser.add_argument('--summary', default='h1/out/step3/tight/step3_summary.csv')
    parser.add_argument('--request-rate', type=float, default=18.5)
    args = parser.parse_args()

    out_dir = Path(args.out)
    rows: list[dict[str, str]] = []
    by_budget: dict[str, dict[str, dict[str, str]]] = {}
    sources: list[tuple[str, str, dict[str, str]]] = []
    for aggregate in sorted(out_dir.glob('*/*/aggregate.csv')):
        sources.append((aggregate.parent.parent.name, aggregate.parent.name, read_first_row(aggregate)))
    for summary_json in sorted(out_dir.glob('*/*/*_summary.json')):
        sources.append((summary_json.parent.parent.name, summary_json.parent.name, row_from_real_summary(summary_json)))

    for budget, policy, source in sources:
        if not source:
            continue
        by_budget.setdefault(budget, {})[policy] = source
        throughput = as_float(source, 'request_throughput')
        row = {
            'budget': budget,
            'policy': policy,
            'saturated': str(bool(args.request_rate and throughput < 0.8 * args.request_rate)).lower(),
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
