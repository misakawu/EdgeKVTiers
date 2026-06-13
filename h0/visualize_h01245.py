#!/usr/bin/env python3
"""Visualize H0/H1/H2/H4/H5 result CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def ensure_out(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_h0(h0_dir: Path, out_dir: Path, plt) -> None:
    rows = read_csv(h0_dir / "summary.csv")
    labels = [row["device_profile"] for row in rows]
    p95 = [as_float(row, "ttft_p95_ms") for row in rows]
    mean = [as_float(row, "ttft_mean_ms") for row in rows]
    memory = [as_float(row, "memory_peak_mb") for row in rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    xs = range(len(labels))
    ax1.bar([x - 0.18 for x in xs], p95, width=0.36, label="p95 TTFT ms")
    ax1.bar([x + 0.18 for x in xs], mean, width=0.36, label="mean TTFT ms")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(labels)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(list(xs), memory, color="black", marker="o", label="memory peak MB")
    ax2.set_ylabel("Memory peak (MB)")
    ax2.legend(loc="upper right")
    ax1.set_title("H0 ShareGPT server/edge replay")
    fig.tight_layout()
    fig.savefig(out_dir / "h0_server_edge_ttft_memory.png", dpi=180)
    plt.close(fig)


def plot_h1(h1245_dir: Path, out_dir: Path, plt) -> None:
    rows = read_csv(h1245_dir / "h1" / "h1_results.csv")
    policies = ["lru", "lfu", "score", "tiered"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for policy in policies:
        sub = [row for row in rows if row["policy"] == policy]
        sub.sort(key=lambda row: as_float(row, "m_budget_mb"))
        ax.plot(
            [as_float(row, "m_budget_mb") for row in sub],
            [as_float(row, "ttft_p95_ms") for row in sub],
            marker="o",
            label="LPE-score" if policy == "score" else policy,
        )
    ax.set_xlabel("M_budget (MB)")
    ax.set_ylabel("p95 TTFT (ms)")
    ax.set_title("H1 lifecycle policy effectiveness")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "h1_lifecycle_p95.png", dpi=180)
    plt.close(fig)


def plot_h2(h1245_dir: Path, out_dir: Path, plt) -> None:
    rows = read_csv(h1245_dir / "h2" / "h2_results.csv")
    modes = ["always-restore", "always-recompute", "rrs"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode in modes:
        sub = [row for row in rows if row["rrs_mode"] == mode]
        sub.sort(key=lambda row: as_float(row, "bw_gbps"))
        ax.plot(
            [as_float(row, "bw_gbps") for row in sub],
            [as_float(row, "ttft_p95_ms") for row in sub],
            marker="o",
            label=mode,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("BW (GB/s)")
    ax.set_ylabel("p95 TTFT (ms)")
    ax.set_title("H2 RRS restore/recompute threshold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "h2_rrs_bw_p95.png", dpi=180)
    plt.close(fig)


def plot_h4(h1245_dir: Path, out_dir: Path, plt) -> None:
    rows = read_csv(h1245_dir / "h4" / "h4_summary.csv")
    budgets = sorted({as_float(row, "m_budget_mb") for row in rows})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, budget in enumerate(budgets):
        color = colors[idx % len(colors)]
        sub = [row for row in rows if as_float(row, "m_budget_mb") == budget]
        sub.sort(key=lambda row: as_float(row, "epsilon_norm"))
        axes[0].plot(
            [as_float(row, "epsilon_norm") for row in sub],
            [as_float(row, "tms_mean_p95_ms") for row in sub],
            marker="o",
            color=color,
            label=f"TMS {int(budget)} MB",
        )
        axes[0].plot(
            [as_float(row, "epsilon_norm") for row in sub],
            [as_float(row, "score_only_mean_p95_ms") for row in sub],
            linestyle="--",
            marker="x",
            color=color,
            label=f"score {int(budget)} MB",
        )
        axes[1].plot(
            [as_float(row, "epsilon_norm") for row in sub],
            [as_float(row, "tms_improvement_vs_score_pct") for row in sub],
            marker="o",
            color=color,
            label=f"{int(budget)} MB",
        )
    axes[0].set_xlabel("epsilon_norm")
    axes[0].set_ylabel("p95 TTFT (ms)")
    axes[0].set_title("H4 TMS vs score-only")
    axes[1].set_xlabel("epsilon_norm")
    axes[1].set_ylabel("Improvement vs score-only (%)")
    axes[1].set_title("H4 quality-budget benefit")
    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "h4_quality_benefit.png", dpi=180)
    plt.close(fig)


def plot_h5(h1245_dir: Path, out_dir: Path, plt) -> None:
    rows = read_csv(h1245_dir / "h5" / "h5_grid.csv")
    tier_rank = {"full": 0, "int8": 1, "int4": 2, "sparse_k": 3}
    bws = sorted({as_float(row, "bw_gbps") for row in rows})
    eps = sorted({as_float(row, "epsilon_norm") for row in rows})
    by_cell: Dict[tuple, dict] = {
        (as_float(row, "bw_gbps"), as_float(row, "epsilon_norm")): row for row in rows
    }
    data = []
    for epsilon in eps:
        data.append([tier_rank[by_cell[(bw, epsilon)]["dominant_q_offline"]] for bw in bws])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(data, aspect="auto", origin="lower")
    ax.set_xticks(range(len(bws)))
    ax.set_xticklabels([str(v) for v in bws])
    ax.set_yticks(range(len(eps)))
    ax.set_yticklabels([str(v) for v in eps])
    ax.set_xlabel("BW (GB/s)")
    ax.set_ylabel("epsilon_norm")
    ax.set_title("H5 quality-bandwidth phase surface")
    cbar = fig.colorbar(image, ax=ax, ticks=list(tier_rank.values()))
    cbar.ax.set_yticklabels(list(tier_rank.keys()))
    fig.tight_layout()
    fig.savefig(out_dir / "h5_phase_surface.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize H0/H1245 result summaries.")
    parser.add_argument("--h0-dir", default=str(REPO_ROOT / "h0" / "out" / "h0_sharegpt_server_edge"))
    parser.add_argument("--h1245-dir", default=str(REPO_ROOT / "h0" / "out" / "h1245_sharegpt"))
    parser.add_argument("--out", default=str(REPO_ROOT / "h0" / "out" / "h01245_visuals"))
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    h0_dir = Path(args.h0_dir).expanduser()
    h1245_dir = Path(args.h1245_dir).expanduser()
    out_dir = ensure_out(Path(args.out).expanduser())
    plot_h0(h0_dir, out_dir, plt)
    plot_h1(h1245_dir, out_dir, plt)
    plot_h2(h1245_dir, out_dir, plt)
    plot_h4(h1245_dir, out_dir, plt)
    plot_h5(h1245_dir, out_dir, plt)
    print(f"visualizations written to {out_dir}")


if __name__ == "__main__":
    main()
