#!/usr/bin/env python3
"""Visualize h1/out/run_all_cell_3x/run_all_cell_3x_average.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POL_KEYS = ["h1_lpe", "h1_lru", "h1_lfu", "vllm_default"]
POL_LABELS = {
    "h1_lpe": "LPE",
    "h1_lru": "LRU",
    "h1_lfu": "LFU",
    "vllm_default": "vLLM default",
}
POL_COLORS = {
    "h1_lpe": "#d95319",
    "h1_lru": "#55a868",
    "h1_lfu": "#c4a000",
    "vllm_default": "#4c72b0",
}
POL_MARKERS = {
    "h1_lpe": "o",
    "h1_lru": "s",
    "h1_lfu": "^",
    "vllm_default": "D",
}
METRICS = [
    ("p95_ttft_ms", "p95 TTFT", "ms", "{:.0f}"),
    ("mean_ttft_ms", "Mean TTFT", "ms", "{:.0f}"),
    ("hit_rate", "GPU prefix-cache hit rate", "hit rate", "{:.3f}"),
    ("gpu_prefix_cache_evictions", "GPU prefix-cache evictions", "evictions", "{:.0f}"),
    ("request_throughput", "Request throughput", "req/s", "{:.2f}"),
    ("policy_time_us_avg", "Policy time", "us avg", "{:.0f}"),
]


def read_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float | str] = {"policy": row["policy"]}
            for key, value in row.items():
                if key == "policy":
                    continue
                parsed[key] = float(value) if value else float("nan")
            rows.append(parsed)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        default="h1/out/run_all_cell_3x/run_all_cell_3x_average.csv",
        help="input average CSV",
    )
    ap.add_argument(
        "--out",
        default="h1/out/run_all_cell_3x/run_all_cell_3x_average_visualization",
        help="output image stem or .png path",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"missing CSV: {csv_path}")

    rows = read_rows(csv_path)
    budgets = sorted({float(r["budget"]) for r in rows})
    by_key = {(float(r["budget"]), str(r["policy"])): r for r in rows}

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8), squeeze=False)
    for ax, (metric, title, ylabel, fmt) in zip(axes.ravel(), METRICS):
        for policy in POL_KEYS:
            vals = [
                float(by_key.get((budget, policy), {}).get(metric, float("nan")))
                for budget in budgets
            ]
            ax.plot(
                budgets,
                vals,
                label=POL_LABELS[policy],
                color=POL_COLORS[policy],
                marker=POL_MARKERS[policy],
                linewidth=2.0,
                markersize=5.0,
            )
            if metric in {"p95_ttft_ms", "hit_rate", "request_throughput"}:
                for x, y in zip(budgets[::3], vals[::3]):
                    if y == y:
                        ax.annotate(fmt.format(y), (x, y), textcoords="offset points", xytext=(0, 7),
                                    ha="center", fontsize=7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("budget")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.set_xticks(budgets)
        ax.tick_params(axis="x", labelrotation=35)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("H1 run_all_cell_3x average policy comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out = Path(args.out)
    stem = out.with_suffix("") if out.suffix else out
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    fig.savefig(png, dpi=180)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
