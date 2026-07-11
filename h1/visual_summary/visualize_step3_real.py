#!/usr/bin/env python3
"""可视化真实 Step3（ShareGPT+HotpotQA）4 策略 x 3 预算对比。

读取 run_step3_real.py / summarize_step3_real.py 生成的跨重复轮次中位数汇总，并在三档
GPU 显存预算下绘制 4 策略对比（vLLM / LRU / LFU / LPE）：

用法（无需环境变量）：
    python h1/visualize_step3_real.py
    python h1/visualize_step3_real.py --summary <path> --out <png_stem>

默认在 summary CSV 旁保存 <out>.png。仅依赖 matplotlib + numpy
（stdlib csv 负责读取；不依赖 pandas）。改写自 visualize_lpe_scenarios.py。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无头渲染
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# 固定顺序：预算 tight,mid,loose；策略 vLLM,LRU,LFU,LPE
BUDGET_KEYS = ["tight", "mid", "loose"]
BUDGET_LABELS = ["tight\n", "mid\n", "loose\n"]
POL_KEYS = ["vllm_default", "h1_lru", "h1_lfu", "h1_lpe"]
POL_LABELS = ["vLLM", "LRU", "LFU", "LPE"]
POL_COLORS = ["#4c72b0", "#55a868", "#c4a000", "#d95319"]  # LPE = 橙红色

# 待绘制指标：(列名, 面板标题, y 轴标签, 数值格式)
METRICS = [
    ("p95_ttft_ms", "p95 TTFT proxy (batch-latency proxy, not real TTFT)", "p95 batch-latency proxy (ms)", "{:.0f}"),
    ("hit_rate", "GPU prefix-cache hit rate", "hit rate", "{:.3f}"),
    ("gpu_prefix_cache_evictions", "GPU prefix-cache evictions", "evictions", "{:.0f}"),
    ("request_throughput", "Request throughput (requests/elapsed_s)", "req/s", "{:.2f}"),
]


def load(summary: Path) -> dict[tuple[str, str], dict[str, float]]:
    """(budget, policy) -> {column: float}。"""
    table: dict[tuple[str, str], dict[str, float]] = {}
    with summary.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["budget"], row["policy"])
            vals = {}
            for col, *_ in METRICS:
                try:
                    vals[col] = float(row[col])
                except (TypeError, ValueError, KeyError):
                    vals[col] = float("nan")
            table[key] = vals
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=str(HERE / "out/step3_real/step3_real_summary.csv"))
    ap.add_argument("--out", default="", help="output image path stem (default: next to summary)")
    args = ap.parse_args()

    summary = Path(args.summary)
    if not summary.is_file():
        raise SystemExit(f"summary CSV not found: {summary}\nRun run_step3_real.py first.")
    print(f"reading {summary}")
    table = load(summary)

    out_stem = Path(args.out) if args.out else summary.with_name("step3_real_scenarios")

    n_b, n_p = len(BUDGET_KEYS), len(POL_KEYS)
    x = np.arange(n_b)
    width = 0.8 / n_p

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for ax, (col, title, ylabel, fmt) in zip(axes.flat, METRICS):
        for j, (pol, label, color) in enumerate(zip(POL_KEYS, POL_LABELS, POL_COLORS)):
            vals = [table.get((b, pol), {}).get(col, float("nan")) for b in BUDGET_KEYS]
            offs = x + (j - (n_p - 1) / 2) * width
            bars = ax.bar(offs, vals, width, label=label, color=color,
                          edgecolor="black", linewidth=0.4)
            ax.bar_label(bars, labels=[fmt.format(v) if v == v else "" for v in vals],
                         fontsize=7, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(BUDGET_LABELS, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.margins(y=0.15)

    axes.flat[0].legend(loc="upper left", fontsize=9, ncol=2)
    fig.suptitle("LPE vs LRU/LFU/vLLM on real ShareGPT+HotpotQA mixed workload "
                 "(4 policies x 3 budgets, cross-rep median)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    png = out_stem.with_suffix(".png")
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
