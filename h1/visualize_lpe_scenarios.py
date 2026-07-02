#!/usr/bin/env python3
"""可视化 LPE_NOT_GOOD A/B/C 场景对比（Python / matplotlib）。

读取 run_step3_repeat.py / summarize_step3_repeat.py 生成的 Step3 repeat 协议汇总
（跨重复轮次中位数），并在三个场景下绘制 4 策略对比（vLLM / LRU / LFU / LPE）：

    A  press_0720_n200   0.720 / n=200   有压力但未饱和（主场景）
    B  press_0730_n200   0.730 / n=200   未饱和
    C  sat_0710_n400     0.710 / n=400   饱和

用法（无需环境变量）：
    python h1/visualize_lpe_scenarios.py
    python h1/visualize_lpe_scenarios.py --summary <path> --out <png>

默认在 summary CSV 旁保存 <out>.png 和 <out>.pdf。
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

# 固定顺序：场景 A,B,C；策略 vLLM,LRU,LFU,LPE
SCEN_KEYS = ["press_0720_n200", "press_0730_n200", "sat_0710_n400"]
SCEN_LABELS = ["A: 0.720/n200\n(pressured)", "B: 0.730/n200\n(non-sat)",
               "C: 0.710/n400\n(saturated)"]
POL_KEYS = ["vllm_default", "h1_lru", "h1_lfu", "h1_lpe"]
POL_LABELS = ["vLLM", "LRU", "LFU", "LPE"]
POL_COLORS = ["#4c72b0", "#55a868", "#c4a000", "#d95319"]  # LPE = 橙红色

# 待绘制指标：(列名, 面板标题, y 轴标签, 数值格式)
METRICS = [
    ("p95_ttft_ms", "p95 TTFT (median across reps)", "p95 TTFT (ms)", "{:.0f}"),
    ("hit_rate", "GPU prefix-cache hit rate", "hit rate", "{:.3f}"),
    ("gpu_prefix_cache_evictions", "GPU prefix-cache evictions", "evictions", "{:.0f}"),
    ("request_throughput", "Request throughput", "req/s", "{:.2f}"),
]


def load(summary: Path) -> dict[tuple[str, str], dict[str, float]]:
    """(point, policy) -> {column: float}。"""
    table: dict[tuple[str, str], dict[str, float]] = {}
    with summary.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["point"], row["policy"])
            vals = {}
            for col, *_ in METRICS:
                try:
                    vals[col] = float(row[col])
                except (TypeError, ValueError):
                    vals[col] = float("nan")
            table[key] = vals
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=str(HERE / "out/step3_repeat/step3_repeat_summary.csv"))
    ap.add_argument("--out", default="", help="output image path stem (default: next to summary)")
    args = ap.parse_args()

    summary = Path(args.summary)
    if not summary.is_file():
        raise SystemExit(f"summary CSV not found: {summary}\nRun run_step3_repeat.py first.")
    print(f"reading {summary}")
    table = load(summary)

    out_stem = Path(args.out) if args.out else summary.with_name("lpe_scenarios")

    n_s, n_p = len(SCEN_KEYS), len(POL_KEYS)
    x = np.arange(n_s)
    width = 0.8 / n_p

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for ax, (col, title, ylabel, fmt) in zip(axes.flat, METRICS):
        for j, (pol, label, color) in enumerate(zip(POL_KEYS, POL_LABELS, POL_COLORS)):
            vals = [table.get((s, pol), {}).get(col, float("nan")) for s in SCEN_KEYS]
            offs = x + (j - (n_p - 1) / 2) * width
            bars = ax.bar(offs, vals, width, label=label, color=color,
                          edgecolor="black", linewidth=0.4)
            ax.bar_label(bars, labels=[fmt.format(v) if v == v else "" for v in vals],
                         fontsize=7, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(SCEN_LABELS, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.margins(y=0.15)

    axes.flat[0].legend(loc="upper left", fontsize=9, ncol=2)
    fig.suptitle("LPE vs LRU/LFU/vLLM across LPE_NOT_GOOD scenarios A/B/C "
                 "(cross-rep median)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    png = out_stem.with_suffix(".png")
    pdf = out_stem.with_suffix(".pdf")
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
