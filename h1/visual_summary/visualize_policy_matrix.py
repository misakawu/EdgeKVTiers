#!/usr/bin/env python3
"""可视化一个或多个第三步策略矩阵汇总。

输入汇总由 h1/run_test.py 生成，路径形如：
  h1/out/<out-dir>/<tier>/step3_summary.csv

图形布局为每个数据集/档位一行、每个指标一列。每个面板展示三档预算，并按
四种策略分组。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POL_KEYS = ["h1_lpe", "h1_lru", "h1_lfu", "vllm_default"]
POL_LABELS = ["LPE", "LRU", "LFU", "vLLM"]
POL_COLORS = ["#d95319", "#55a868", "#c4a000", "#4c72b0"]
DEFAULT_BUDGETS = ["0.75", "0.825", "0.95"]
METRICS = [
    ("p95_ttft_ms", "p95 TTFT proxy", "ms", "{:.0f}"),
    ("hit_rate", "GPU prefix-cache hit rate", "hit rate", "{:.3f}"),
    ("gpu_prefix_cache_evictions", "GPU prefix-cache evictions", "evictions", "{:.0f}"),
    ("request_throughput", "Request throughput", "req/s", "{:.2f}"),
]


def read_summary(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    table: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("budget", ""), row.get("policy", ""))
            vals: dict[str, float] = {}
            for col, *_ in METRICS:
                try:
                    vals[col] = float(row.get(col, "") or "nan")
                except ValueError:
                    vals[col] = float("nan")
            table[key] = vals
    return table


def parse_input(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label, Path(path)
    path = Path(spec)
    return path.parent.name, path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", action="append", required=True,
                    help="label=path/to/step3_summary.csv; repeat for multiple datasets")
    ap.add_argument("--out", required=True, help="output image stem or .png path")
    ap.add_argument("--budgets", default=" ".join(DEFAULT_BUDGETS))
    args = ap.parse_args()

    inputs = [parse_input(s) for s in args.summary]
    budgets = args.budgets.split()
    missing = [str(p) for _, p in inputs if not p.is_file()]
    if missing:
        raise SystemExit("missing summary CSV(s): " + ", ".join(missing))

    tables = [(label, read_summary(path)) for label, path in inputs]
    n_rows = len(tables)
    n_cols = len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.5 * n_rows), squeeze=False)
    x = np.arange(len(budgets))
    width = 0.82 / len(POL_KEYS)

    for row_idx, (dataset, table) in enumerate(tables):
        for col_idx, (metric, title, ylabel, fmt) in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            for pol_idx, (policy, label, color) in enumerate(zip(POL_KEYS, POL_LABELS, POL_COLORS)):
                vals = [table.get((budget, policy), {}).get(metric, float("nan")) for budget in budgets]
                offs = x + (pol_idx - (len(POL_KEYS) - 1) / 2) * width
                bars = ax.bar(offs, vals, width, label=label, color=color, edgecolor="black", linewidth=0.35)
                ax.bar_label(bars, labels=[fmt.format(v) if v == v else "" for v in vals], fontsize=7, padding=2)
            ax.set_xticks(x)
            ax.set_xticklabels([f"budget\n{b}" for b in budgets], fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", linestyle=":", alpha=0.55)
            ax.margins(y=0.18)
            # if col_idx == 0:
            ax.text(-0.25, 1.04, dataset, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
            # if row_idx == 0 and col_idx == n_cols - 1:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("H1 replay trace policy comparison (3 budgets x 4 policies)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(args.out)
    stem = out.with_suffix("") if out.suffix else out
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    fig.savefig(png, dpi=160)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
