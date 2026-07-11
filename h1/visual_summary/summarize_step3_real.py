#!/usr/bin/env python3
"""汇总真实 Step3（ShareGPT+HotpotQA）矩阵：对多个 rep 取中位数。

读取 <base>/rep*/<budget>_<policy>/<budget>_<policy>_summary.json（由
h1/run_h1_vllm0110_real.py 写出），并为每个 (budget, policy) 写出核心指标
跨 rep 中位数的紧凑 CSV。real harness 不直接输出 request_throughput，因此按
每个 rep 的 requests / elapsed_s 派生后再取中位数。不输出路径列。与
summarize_step3_repeat.py 保持一致。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

# (CSV 列名, 源 summary.json 键名)
METRICS = [
    ('p95_ttft_ms', 'ttft_proxy_p95_ms'),
    ('p50_ttft_ms', 'ttft_proxy_p50_ms'),
    ('queue_wait_ms', 'queue_wait_ms'),
    ('prefill_ms', 'prefill_ms'),
    ('queue_wait_ratio_mean', 'queue_wait_ratio_mean'),
    ('queue_wait_p95_ms', 'queue_wait_p95_ms'),
    ('prefill_p95_ms', 'prefill_p95_ms'),
    ('batch_queue_span_p95_ms', 'batch_queue_span_p95_ms'),
    ('hit_rate', 'hit_rate'),
    ('gpu_prefix_cache_evictions', 'gpu_prefix_cache_evictions'),
    ('gpu_prefix_cache_cached_blocks', 'gpu_prefix_cache_cached_blocks'),
    ('free_queue_reorder_calls', 'free_queue_reorder_calls'),
    ('free_queue_reorder_blocks', 'free_queue_reorder_blocks'),
    ('free_queue_reorder_skipped', 'free_queue_reorder_skipped'),
    ('free_queue_reorder_window', 'free_queue_reorder_window'),
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
    # (budget, policy, replay_batch_size, batch_order, warmup_batches) ->
    # 列名 -> [跨 rep 的取值]。
    # replay_batch_size 作为 key，保证 --batch-sweep 运行能区分不同并发；
    # 普通单并发运行只会为每个 (budget, policy) 产生一个 batch 值。
    collected: dict[tuple[str, str, str, str, str], dict[str, list[float]]] = defaultdict(
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
        batch_order = str(data.get('batch_order', 'original'))
        warmup_batches = str(data.get('warmup_batches', '0'))
        key = (budget, policy, batch, batch_order, warmup_batches)
        for col, src in METRICS:
            collected[key][col].append(fnum(data.get(src)))
        elapsed = fnum(data.get('elapsed_s'))
        throughput = fnum(data.get('requests')) / elapsed if elapsed > 0 else 0.0
        collected[key]['request_throughput'].append(throughput)

    columns = [c for c, _ in METRICS] + list(DERIVED)
    rows: list[dict[str, object]] = []
    for (budget, policy, batch, batch_order, warmup_batches), metric_vals in sorted(collected.items()):
        reps = max((len(v) for v in metric_vals.values()), default=0)
        row: dict[str, object] = {
            'budget': budget,
            'policy': policy,
            'replay_batch_size': batch,
            'batch_order': batch_order,
            'warmup_batches': warmup_batches,
            'reps': reps,
        }
        for col in columns:
            vals = metric_vals.get(col, [])
            row[col] = round(statistics.median(vals), 6) if vals else ''
        rows.append(row)

    fields = [
        'budget',
        'policy',
        'replay_batch_size',
        'batch_order',
        'warmup_batches',
        'reps',
        *columns,
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
