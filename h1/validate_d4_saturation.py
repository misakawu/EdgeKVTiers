#!/usr/bin/env python3
"""在不依赖 vLLM RequestOutput.metrics 的情况下验证 H1 D4 饱和度。

Offline ``LLM.generate`` 可能让请求时间指标保持为 0。本脚本使用 run summary
和请求级 token 计数，估计实测吞吐是否显著低于理想的仅 prefill 吞吐；这是
队列等待主导型饱和的有效 D4 信号。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_tokens(requests_path: Path) -> list[float]:
    with requests_path.open(encoding='utf-8', newline='') as f:
        rows = csv.DictReader(f)
        values = [fnum(row.get('n_tokens')) for row in rows]
    return [value for value in values if value > 0.0]


def check_saturation(
    summary_path: Path,
    requests_path: Path,
    *,
    c_re_ms_per_token: float | None = None,
    threshold: float = 0.60,
    out_json: Path | None = None,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    tokens = load_tokens(requests_path)
    avg_tokens = statistics.mean(tokens) if tokens else 0.0
    c_re = (
        float(c_re_ms_per_token)
        if c_re_ms_per_token is not None
        else fnum(summary.get('c_re_ms_per_token'), 0.12)
    )
    avg_prefill_ms = avg_tokens * c_re
    requests = int(summary.get('requests', 0) or 0)
    elapsed_s = fnum(summary.get('elapsed_s'))
    actual_throughput = requests / elapsed_s if elapsed_s > 0.0 else 0.0
    theoretical_throughput = 1000.0 / avg_prefill_ms if avg_prefill_ms > 0.0 else 0.0
    saturation_ratio = (
        actual_throughput / theoretical_throughput
        if theoretical_throughput > 0.0 else 0.0
    )
    result = {
        'summary_path': str(summary_path),
        'requests_path': str(requests_path),
        'requests': requests,
        'elapsed_s': elapsed_s,
        'avg_tokens': avg_tokens,
        'c_re_ms_per_token': c_re,
        'avg_prefill_ms': avg_prefill_ms,
        'actual_throughput_req_s': actual_throughput,
        'theoretical_prefill_throughput_req_s': theoretical_throughput,
        'saturation_ratio': saturation_ratio,
        'saturation_threshold': threshold,
        'is_saturated': bool(saturation_ratio < threshold),
        'decision': 'saturated_queue_wait_dominant'
        if saturation_ratio < threshold else 'not_saturated_prefill_dominant',
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding='utf-8',
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary_json', type=Path)
    parser.add_argument('requests_csv', type=Path)
    parser.add_argument('--c-re-ms-per-token', type=float, default=None)
    parser.add_argument('--threshold', type=float, default=0.60)
    parser.add_argument('--out-json', type=Path, default=None)
    args = parser.parse_args()

    result = check_saturation(
        args.summary_json,
        args.requests_csv,
        c_re_ms_per_token=args.c_re_ms_per_token,
        threshold=args.threshold,
        out_json=args.out_json,
    )
    print('=== D4 saturation validation ===')
    print(f"avg_tokens={result['avg_tokens']:.1f}")
    print(f"avg_prefill_ms={result['avg_prefill_ms']:.3f}")
    print(
        f"actual_throughput={result['actual_throughput_req_s']:.3f} req/s, "
        f"theoretical={result['theoretical_prefill_throughput_req_s']:.3f} req/s"
    )
    print(f"saturation_ratio={result['saturation_ratio']:.3f}")
    print(f"decision={result['decision']}")
    if args.out_json is not None:
        print(f"wrote {args.out_json}")


if __name__ == '__main__':
    main()
