#!/usr/bin/env python3
"""Visualize H0/H1/H2/H4/H5 result CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
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
    if not rows:
        return
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
    if not rows:
        return
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
    if not rows:
        return
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
    if not rows:
        return
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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, budget in enumerate(budgets):
        color = colors[idx % len(colors)]
        sub = [row for row in rows if as_float(row, "m_budget_mb") == budget]
        sub.sort(key=lambda row: as_float(row, "epsilon_norm"))
        ax.plot(
            [as_float(row, "epsilon_norm") for row in sub],
            [as_float(row, "tms_improvement_vs_best_pct") for row in sub],
            marker="o",
            color=color,
            label=f"{int(budget)} MB",
        )
    ax.axhline(25.0, color="black", linestyle="--", linewidth=1.0, label="P1 gate 25%")
    ax.set_xlabel("epsilon_norm")
    ax.set_ylabel("Improvement vs best valid baseline (%)")
    ax.set_title("H4 P1 gate check")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "h4_p1_gate.png", dpi=180)
    plt.close(fig)


def plot_h5(h1245_dir: Path, out_dir: Path, plt) -> None:
    rows = read_csv(h1245_dir / "h5" / "h5_grid.csv")
    if not rows:
        return
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
    for y_idx, epsilon in enumerate(eps):
        for x_idx, bw in enumerate(bws):
            cell = by_cell[(bw, epsilon)]
            if str(cell.get("phase_boundary_cell", "")).lower() == "true":
                ax.scatter([x_idx], [y_idx], marker="x", color="white", linewidths=1.8)
    ax.set_title("H5 quality-bandwidth phase surface")
    cbar = fig.colorbar(image, ax=ax, ticks=list(tier_rank.values()))
    cbar.ax.set_yticklabels(list(tier_rank.keys()))
    fig.tight_layout()
    fig.savefig(out_dir / "h5_phase_surface.png", dpi=180)
    plt.close(fig)

    tau_rows = read_csv(h1245_dir / "h5" / "h5_tau.csv")
    if tau_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for epsilon in eps:
            sub = [row for row in tau_rows if as_float(row, "epsilon_norm") == epsilon]
            sub.sort(key=lambda row: as_float(row, "bw_gbps"))
            ax.plot(
                [as_float(row, "bw_gbps") for row in sub],
                [as_float(row, "kendall_tau") for row in sub],
                marker="o",
                label=f"eps {epsilon:g}",
            )
        ax.axhline(0.8, color="black", linestyle="--", linewidth=1.0, label="tau gate 0.8")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("BW (GB/s)")
        ax.set_ylabel("Kendall tau")
        ax.set_title("H5 prediction rank agreement")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "h5_kendall_tau.png", dpi=180)
        plt.close(fig)


def plot_go_nogo(h1245_dir: Path, out_dir: Path, plt) -> None:
    report_path = h1245_dir / "go_nogo_report.json"
    failures = read_csv(h1245_dir / "failure_cases.csv")
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gates = report.get("gates", {})
    labels = list(gates.keys())
    values = [1 if gates[key] else 0 for key in labels]
    colors = ["#2f9e44" if value else "#c92a2a" for value in values]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))
    axes[0].barh(range(len(labels)), values, color=colors)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlim(0, 1)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["fail", "pass"])
    axes[0].set_title(f"Go/No-Go: {report.get('status', 'unknown')}")

    by_type: Dict[str, int] = {}
    for row in failures:
        kind = str(row.get("case_id", "UNKNOWN")).split("-")[0]
        by_type[kind] = by_type.get(kind, 0) + 1
    if by_type:
        axes[1].bar(list(by_type.keys()), list(by_type.values()), color="#4c6ef5")
    axes[1].set_ylabel("cases")
    axes[1].set_title("Extracted failure cases")
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "go_nogo_summary.png", dpi=180)
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
    plot_go_nogo(h1245_dir, out_dir, plt)
    print(f"visualizations written to {out_dir}")


if __name__ == "__main__":
    main()
