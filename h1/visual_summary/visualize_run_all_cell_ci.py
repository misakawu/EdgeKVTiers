#!/usr/bin/env python3
"""Visualize H1 run_all_cell CI long-table results."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
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
LINE_METRICS = [
    ("p95_ttft_ms", "p95 TTFT", "ms", "{:.0f}"),
    ("prefill_p95_ms", "p95 Prefill time", "ms", "{:.0f}"),
]
BAR_METRIC = ("hit_rate", "GPU prefix-cache hit rate", "hit rate", "{:.3f}")


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        parsed = float(value)
    except ValueError:
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def sort_budget(value: str) -> tuple[int, float | str]:
    parsed = parse_float(value)
    return (0, parsed) if parsed == parsed else (1, value)


def read_rows(path: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "tier": row["tier"],
                    "budget": row["budget"],
                    "budget_num": parse_float(row["budget"]),
                    "policy": row["policy"],
                    "metric": row["metric"],
                    "value": parse_float(row.get("value")),
                    "ci95_low": parse_float(row.get("ci95_low")),
                    "ci95_high": parse_float(row.get("ci95_high")),
                }
            )
    return rows


def yerr(
    values: list[float],
    lows: list[float],
    highs: list[float],
) -> list[list[float]]:
    lower: list[float] = []
    upper: list[float] = []
    for value, low, high in zip(values, lows, highs):
        if value == value and low == low and high == high:
            lower.append(max(0.0, value - low))
            upper.append(max(0.0, high - value))
        else:
            lower.append(0.0)
            upper.append(0.0)
    return [lower, upper]


def values_for(
    by_key: dict[tuple[str, str, str], dict[str, str | float]],
    budgets: list[str],
    policy: str,
    metric: str,
) -> tuple[list[float], list[float], list[float]]:
    values: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for budget in budgets:
        row = by_key.get((budget, policy, metric), {})
        values.append(float(row.get("value", math.nan)))
        lows.append(float(row.get("ci95_low", math.nan)))
        highs.append(float(row.get("ci95_high", math.nan)))
    return values, lows, highs


def annotate_points(ax, xs: list[float], ys: list[float], fmt: str) -> None:
    for x, y in zip(xs, ys):
        if y == y:
            ax.annotate(
                fmt.format(y),
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
            )


def plot_tier(tier: str, rows: list[dict[str, str | float]], out: Path) -> Path:
    budgets = sorted({str(row["budget"]) for row in rows}, key=sort_budget)
    x_positions = [float(i) for i in range(len(budgets))]
    by_key = {
        (str(row["budget"]), str(row["policy"]), str(row["metric"])): row
        for row in rows
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), squeeze=False)
    line_axes = axes[0][:2]
    bar_ax = axes[0][2]

    for ax, (metric, title, ylabel, fmt) in zip(line_axes, LINE_METRICS):
        for policy in POL_KEYS:
            values, lows, highs = values_for(by_key, budgets, policy, metric)
            ax.errorbar(
                x_positions,
                values,
                yerr=yerr(values, lows, highs),
                label=POL_LABELS[policy],
                color=POL_COLORS[policy],
                marker=POL_MARKERS[policy],
                linewidth=2.0,
                markersize=5.0,
                capsize=3.0,
                elinewidth=1.1,
            )
            annotate_points(ax, x_positions, values, fmt)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("budget")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions, budgets, rotation=35)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(loc="best", fontsize=8, frameon=False)

    metric, title, ylabel, fmt = BAR_METRIC
    width = 0.18
    offsets = {
        policy: (index - (len(POL_KEYS) - 1) / 2.0) * width
        for index, policy in enumerate(POL_KEYS)
    }
    for policy in POL_KEYS:
        values, lows, highs = values_for(by_key, budgets, policy, metric)
        xs = [x + offsets[policy] for x in x_positions]
        bar_ax.bar(
            xs,
            values,
            width=width,
            yerr=yerr(values, lows, highs),
            label=POL_LABELS[policy],
            color=POL_COLORS[policy],
            capsize=3.0,
            error_kw={"elinewidth": 1.0},
        )
        annotate_points(bar_ax, xs, values, fmt)
    bar_ax.set_title(title, fontsize=11, fontweight="bold")
    bar_ax.set_xlabel("budget")
    bar_ax.set_ylabel(ylabel)
    bar_ax.set_xticks(x_positions, budgets, rotation=35)
    bar_ax.set_ylim(bottom=0.0)
    bar_ax.grid(True, axis="y", linestyle=":", alpha=0.55)
    bar_ax.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(f"H1 run_all_cell CI policy comparison ({tier})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def output_path(base: Path, tier: str, tier_count: int) -> Path:
    path = base if base.suffix else base.with_suffix(".png")
    if tier_count <= 1:
        return path
    safe_tier = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tier)
    return path.with_name(f"{path.stem}_{safe_tier}{path.suffix}")


def regenerate_csv(args: argparse.Namespace, csv_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from h1 import aggregate_run_all_cell_ci as agg

    if args.bootstrap_iters < 1:
        raise SystemExit("--bootstrap-iters must be >= 1")
    if args.min_samples < 1:
        raise SystemExit("--min-samples must be >= 1")

    sources, warnings = agg.discover_sources(args.base_glob, args.tier)
    samples, read_warnings = agg.read_samples(sources)
    warnings.extend(read_warnings)
    rows, aggregate_warnings = agg.aggregate(
        samples,
        args.min_samples,
        args.bootstrap_iters,
        args.bootstrap_seed,
    )
    warnings.extend(aggregate_warnings)
    agg.write_csv(csv_path, rows)

    print(f"wrote {csv_path} ({len(rows)} rows, {len(sources)} source CSVs, {len(samples)} samples)")
    if warnings:
        unique_warnings = list(dict.fromkeys(warnings))
        print(f"warnings: {len(unique_warnings)}", file=sys.stderr)
        for warning in unique_warnings[:40]:
            print(f"[warn] {warning}", file=sys.stderr)
        if len(unique_warnings) > 40:
            print(f"[warn] ... {len(unique_warnings) - 40} more", file=sys.stderr)
        if args.strict:
            raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        default="h1/out/run_all_cell_ci/h1_visualization_data.csv",
        help="input CI long-table CSV",
    )
    ap.add_argument(
        "--regenerate-csv",
        action="store_true",
        help="regenerate the CI long-table CSV before plotting",
    )
    ap.add_argument(
        "--base-glob",
        default="h1/out/run_all_cell_3x_rep*",
        help="run_all_cell root glob used when regenerating CSV",
    )
    ap.add_argument(
        "--tier",
        default=None,
        help="tier directory name used when regenerating CSV, for example sharegpt_batch_8",
    )
    ap.add_argument(
        "--bootstrap-iters",
        type=int,
        default=10000,
        help="bootstrap iterations per metric group used when regenerating CSV",
    )
    ap.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260711,
        help="base seed for deterministic bootstrap resampling",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="minimum valid samples before adding a warning",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if CSV regeneration produces any warning",
    )
    ap.add_argument(
        "--out",
        default="h1/out/run_all_cell_ci/h1_ci_visualization.png",
        help="output PNG path; multiple tiers append _<tier>",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if args.regenerate_csv or not csv_path.is_file():
        regenerate_csv(args, csv_path)
    if not csv_path.is_file():
        raise SystemExit(f"missing CSV: {csv_path}")

    rows = read_rows(csv_path)
    by_tier: dict[str, list[dict[str, str | float]]] = defaultdict(list)
    wanted = {metric for metric, *_ in LINE_METRICS}
    wanted.add(BAR_METRIC[0])
    for row in rows:
        if str(row["metric"]) in wanted:
            by_tier[str(row["tier"])].append(row)

    if not by_tier:
        raise SystemExit(f"no target metrics found in {csv_path}")

    out_base = Path(args.out)
    for tier in sorted(by_tier):
        png = plot_tier(tier, by_tier[tier], output_path(out_base, tier, len(by_tier)))
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
