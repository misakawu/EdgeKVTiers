#!/usr/bin/env python3
"""Summarize the real Step3 (ShareGPT+HotpotQA) matrix: median across reps.

Reads <base>/rep*/<budget>_<policy>/<budget>_<policy>_summary.json (written by
h1/run_h1_vllm0110_real.py) and writes one concise CSV with the median of each core
metric across reps per (budget, policy). request_throughput is not emitted by the
real harness, so it is derived per rep as requests / elapsed_s and then medianed.
No path columns. Mirrors summarize_step3_repeat.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

# (csv_column, source_summary.json_key)
METRICS = [
    ('p95_ttft_ms', 'ttft_proxy_p95_ms'),
    ('p50_ttft_ms', 'ttft_proxy_p50_ms'),
    ('hit_rate', 'hit_rate'),
    ('gpu_prefix_cache_evictions', 'gpu_prefix_cache_evictions'),
    ('gpu_prefix_cache_cached_blocks', 'gpu_prefix_cache_cached_blocks'),
    ('free_queue_reorder_time_ms', 'free_queue_reorder_time_ms'),
    ('policy_time_us_avg', 'policy_time_us_avg'),
    ('eviction_decision_time_us_avg', 'eviction_decision_time_us_avg'),
    ('score_std', 'score_std'),
    ('p_reuse_std', 'p_reuse_std'),
    ('c_recomp_ms_p50', 'c_recomp_ms_p50'),
]
DERIVED = ('request_throughput',)


def fnum(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='h1/out/step3_real')
    parser.add_argument('--summary', default='h1/out/step3_real/step3_real_summary.csv')
    args = parser.parse_args()

    base = Path(args.base)
    # (budget, policy, replay_batch_size) -> column -> [values across reps].
    # replay_batch_size is a key so a --batch-sweep run keeps each concurrency separate;
    # a normal (single-concurrency) run just yields one batch value per (budget, policy).
    collected: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sjson in sorted(base.glob('rep*/*/*_summary.json')):
        try:
            data = json.loads(sjson.read_text(encoding='utf-8'))
        except Exception:
            continue
        if not data.get('ok', False):
            continue
        budget = str(data.get('budget', ''))
        policy = str(data.get('policy', ''))
        batch = str(data.get('replay_batch_size', ''))
        key = (budget, policy, batch)
        for col, src in METRICS:
            collected[key][col].append(fnum(data.get(src)))
        elapsed = fnum(data.get('elapsed_s'))
        throughput = fnum(data.get('requests')) / elapsed if elapsed > 0 else 0.0
        collected[key]['request_throughput'].append(throughput)

    columns = [c for c, _ in METRICS] + list(DERIVED)
    rows: list[dict[str, object]] = []
    for (budget, policy, batch), metric_vals in sorted(collected.items()):
        reps = max((len(v) for v in metric_vals.values()), default=0)
        row: dict[str, object] = {
            'budget': budget, 'policy': policy, 'replay_batch_size': batch, 'reps': reps,
        }
        for col in columns:
            vals = metric_vals.get(col, [])
            row[col] = round(statistics.median(vals), 6) if vals else ''
        rows.append(row)

    fields = ['budget', 'policy', 'replay_batch_size', 'reps', *columns]
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f'wrote {summary_path} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
