#!/usr/bin/env python3
"""Aggregate ws2 policy-comparison runs and emit CSV + figures.

Reads h1/out/hotqa三级trace_有效窗口/policy_cmp/{policy}_rep{n}/{budget}_{policy}_summary.json,
groups by (policy, budget), computes mean +/- std over reps for the key metrics,
writes a tidy CSV and two PNG figures (true native hit_rate and p95 latency).
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path("h1/out/hotqa三级trace_有效窗口/policy_cmp")
POLICIES = ["h1_lru", "h1_lfu", "h1_lpe"]
BUDGETS = ["0.85", "0.90"]
REPS = [1, 2, 3]

METRICS = [
    "hit_rate",            # native token coverage (true)
    "native_hit_rate",
    "block_lookup_hit_rate",
    "gpu_prefix_cache_evictions",
    "queue_wait_p95_ratio",
    "latency_p95_ms",
    "ttft_proxy_p95_ms",
    "prefill_p95_ms",
    "gpu_memory_peak_mib",
    "elapsed_s",
]


def load(policy: str, budget: str, rep: int) -> dict | None:
    p = ROOT / f"{policy}_rep{rep}" / f"{budget}_{policy}_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def mean_std(vals: list[float]) -> tuple[float, float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return (float("nan"), float("nan"))
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return (m, s)


def main() -> None:
    rows = []
    for policy in POLICIES:
        for budget in BUDGETS:
            per_metric: dict[str, list[float]] = {k: [] for k in METRICS}
            n_ok = 0
            for rep in REPS:
                d = load(policy, budget, rep)
                if not d or not d.get("ok"):
                    continue
                n_ok += 1
                for k in METRICS:
                    v = d.get(k)
                    per_metric[k].append(float(v) if v is not None else None)
            row = {"policy": policy, "budget": budget, "reps_ok": n_ok}
            for k in METRICS:
                m, s = mean_std(per_metric[k])
                row[f"{k}_mean"] = round(m, 6) if not math.isnan(m) else ""
                row[f"{k}_std"] = round(s, 6) if not math.isnan(s) else ""
            rows.append(row)

    out_csv = ROOT / "ws2_policy_cmp_summary.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {out_csv}")
    # console table
    for r in rows:
        print(
            f"{r['policy']:7s} bud={r['budget']} reps={r['reps_ok']} "
            f"hit={r['hit_rate_mean']}±{r['hit_rate_std']} "
            f"evict={r['gpu_prefix_cache_evictions_mean']} "
            f"qwr={r['queue_wait_p95_ratio_mean']} "
            f"lat_p95={r['latency_p95_ms_mean']}±{r['latency_p95_ms_std']}"
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable, skipped figures: {e}")
        return

    x = list(range(len(POLICIES)))
    width = 0.35
    # Figure 1: native hit_rate by policy x budget
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, budget in enumerate(BUDGETS):
        means, stds = [], []
        for policy in POLICIES:
            r = next((rr for rr in rows if rr["policy"] == policy and rr["budget"] == budget), None)
            means.append(r["hit_rate_mean"] if r and r["hit_rate_mean"] != "" else 0.0)
            stds.append(r["hit_rate_std"] if r and r["hit_rate_std"] != "" else 0.0)
        ax.bar([xi + (i - 0.5) * width for xi in x], means, width, yerr=stds, capsize=4,
               label=f"budget {budget}")
    ax.axhspan(0.5, 0.85, color="green", alpha=0.08, label="effective window")
    ax.set_xticks(x)
    ax.set_xticklabels(POLICIES)
    ax.set_ylabel("native prefix-cache hit_rate (token coverage)")
    ax.set_title("ws2 policy comparison — true hit_rate (mean±std, reps)")
    ax.legend()
    fig.tight_layout()
    f1 = ROOT / "ws2_hit_rate.png"
    fig.savefig(f1, dpi=160)
    print(f"wrote {f1}")

    # Figure 2: p95 latency by policy x budget
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, budget in enumerate(BUDGETS):
        means, stds = [], []
        for policy in POLICIES:
            r = next((rr for rr in rows if rr["policy"] == policy and rr["budget"] == budget), None)
            means.append(r["latency_p95_ms_mean"] if r and r["latency_p95_ms_mean"] != "" else 0.0)
            stds.append(r["latency_p95_ms_std"] if r and r["latency_p95_ms_std"] != "" else 0.0)
        ax.bar([xi + (i - 0.5) * width for xi in x], means, width, yerr=stds, capsize=4,
               label=f"budget {budget}")
    ax.set_xticks(x)
    ax.set_xticklabels(POLICIES)
    ax.set_ylabel("p95 latency / TTFT proxy (ms)")
    ax.set_title("ws2 policy comparison — p95 latency (mean±std, reps)")
    ax.legend()
    fig.tight_layout()
    f2 = ROOT / "ws2_latency_p95.png"
    fig.savefig(f2, dpi=160)
    print(f"wrote {f2}")


if __name__ == "__main__":
    main()
