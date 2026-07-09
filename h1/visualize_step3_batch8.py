#!/usr/bin/env python3
"""可视化 batch_size=8 完整实验的 step3_summary.csv（ShareGPT_v5 与 HotQA_ws2）。

两个数据集分开绘制，每个数据集绘制三张图：ttft(p95)、hit_rate、prefill_ms。
横轴为 3 档 GPU 显存预算(0.75/0.825/0.9)，每档 4 策略并排(vLLM/LRU/LFU/LPE)。

用法：
    python h1/visualize_step3_batch8.py
    python h1/visualize_step3_batch8.py --out <dir>

仅依赖 matplotlib + numpy（stdlib csv 读取，不依赖 pandas）。
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
BASE = HERE / "备份-运行结果" / "H1完整实验_batch_size_8"

# 待绘制的两个数据集：(标题, summary CSV 路径)
DATASETS = [
    ("ShareGPT_v5", BASE / "sharedgpt_v5" / "step3_summary.csv"),
    ("HotQA_ws2", BASE / "hotqa_ws2" / "step3_summary.csv"),
]

# 策略顺序与配色：vLLM/LRU/LFU/LPE，LPE = 橙红色
POL_KEYS = ["vllm_default", "h1_lru", "h1_lfu", "h1_lpe"]
POL_LABELS = ["vLLM", "LRU", "LFU", "LPE"]
POL_COLORS = ["#4c72b0", "#55a868", "#c4a000", "#d95319"]

# 三张图：(列名, 面板标题, y 轴标签, 数值格式)
METRICS = [
    ("p95_ttft_ms", "p95 TTFT", "p95 TTFT (ms)", "{:.0f}"),
    ("hit_rate", "GPU prefix-cache hit rate", "hit rate", "{:.3f}"),
    ("prefill_ms", "Prefill time", "prefill (ms)", "{:.0f}"),
]


def load(summary: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, float]]]:
    """返回 (排序后的 budget 列表, {(budget, policy): {column: float}})。"""
    table: dict[tuple[str, str], dict[str, float]] = {}
    budgets: set[str] = set()
    with summary.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            budgets.add(row["budget"])
            vals = {}
            for col, *_ in METRICS:
                try:
                    vals[col] = float(row[col])
                except (TypeError, ValueError, KeyError):
                    vals[col] = float("nan")
            table[(row["budget"], row["policy"])] = vals
    budget_keys = sorted(budgets, key=float)
    return budget_keys, table


def plot_dataset(name: str, summary: Path, out_dir: Path) -> None:
    if not summary.is_file():
        raise SystemExit(f"summary CSV not found: {summary}")
    print(f"reading {summary}")
    budget_keys, table = load(summary)

    x = np.arange(len(budget_keys))
    n_p = len(POL_KEYS)
    width = 0.8 / n_p

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (col, title, ylabel, fmt) in zip(axes, METRICS):
        for j, (pol, label, color) in enumerate(zip(POL_KEYS, POL_LABELS, POL_COLORS)):
            vals = [table.get((b, pol), {}).get(col, float("nan")) for b in budget_keys]
            offs = x + (j - (n_p - 1) / 2) * width
            bars = ax.bar(offs, vals, width, label=label, color=color,
                          edgecolor="black", linewidth=0.4)
            ax.bar_label(bars, labels=[fmt.format(v) if v == v else "" for v in vals],
                         fontsize=7, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"budget={b}" for b in budget_keys], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.margins(y=0.15)

    axes[0].legend(loc="upper right", fontsize=9, ncol=2)
    fig.suptitle(f"{name}  —  step3 (batch_size=8, 4 policies x 3 budgets)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"step3_{name}"
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.png')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "out" / "step3_viz_batch8"),
                    help="输出目录")
    args = ap.parse_args()
    out_dir = Path(args.out)
    for name, summary in DATASETS:
        plot_dataset(name, summary, out_dir)


if __name__ == "__main__":
    main()
