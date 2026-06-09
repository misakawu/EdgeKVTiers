#!/usr/bin/env python3
"""H4: quality dimension benefit validation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import sim


BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "out" / "h4"
TIER_CONFIG_PATH = BASE_DIR / "out" / "tier_config.json"
MAX_SESSIONS = 200
SEED = 4
TRACE_REQUESTS = 500
M_BUDGETS = [1600.0, 2200.0, 3000.0]
EPSILON_NORMS = [0.002, 0.004, 0.008]
REPEATS = [1, 2, 3, 4, 5]
BASELINES = [
    {
        "method": "ragcache_pgdsf_approx",
        "policy": "pgdsf",
        "offload_keep_threshold": 1.1,
    },
    {"method": "static-int4", "policy": "static-int4", "offload_keep_threshold": 1.1},
    {"method": "lpe-only", "policy": "score", "offload_keep_threshold": 1.1},
    {"method": "tms-tiered", "policy": "tiered", "offload_keep_threshold": 1.1},
]
BASE_CFG = sim.SimConfig(seed=SEED, trace_requests=TRACE_REQUESTS, bw_gbps=8.0)
METHOD_LABELS = {
    "ragcache_pgdsf_approx": "RAGCache-PGDSF approx",
    "static-int4": "Static int4",
    "lpe-only": "Score-only / LPE",
    "tms-tiered": "Tiered / TMS",
}


def limit_trace(
    objects: Sequence[sim.KVObject],
    requests: Sequence[sim.Request],
    max_requests: int,
) -> Tuple[List[sim.KVObject], List[sim.Request]]:
    limited_requests = list(requests[:max_requests])
    used_ids = set()
    for req in limited_requests:
        used_ids.add(req.object_id)
        if req.admit_object_id:
            used_ids.add(req.admit_object_id)
    limited_objects = [obj for obj in objects if obj.object_id in used_ids]
    return limited_objects, limited_requests


def load_tiers() -> str:
    if not TIER_CONFIG_PATH.exists():
        return "default_demo"
    with TIER_CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tiers = data.get("tiers", data)
    for q, spec in tiers.items():
        if q in sim.TIERS:
            sim.TIERS[q].update(
                {
                    "size_factor": float(spec["size_factor"]),
                    "qloss_per_token": float(spec["qloss_per_token"]),
                    "restore_factor": float(spec["restore_factor"]),
                }
            )
    return str(TIER_CONFIG_PATH)


def build_trace(seed: int) -> Tuple[List[sim.KVObject], List[sim.Request], str]:
    if sim.DEFAULT_SHAREGPT_PATH.exists():
        objects, requests = sim.load_sharegpt_trace(
            sim.DEFAULT_SHAREGPT_PATH,
            max_sessions=MAX_SESSIONS,
        )
        objects, requests = limit_trace(objects, requests, TRACE_REQUESTS)
        return objects, requests, str(sim.DEFAULT_SHAREGPT_PATH)
    objects, requests = sim.make_synthetic_trace(seed, TRACE_REQUESTS)
    return objects, requests, "synthetic"


def run_matrix() -> Tuple[List[dict], str, int, int, int]:
    rows: List[dict] = []
    base_objects, base_requests, trace_source = build_trace(SEED)
    object_count = len(base_objects)
    request_count = len(base_requests)
    token_ref = sim.token_ref_for_objects(base_objects)
    for repeat_seed in REPEATS:
        if trace_source == "synthetic":
            objects, requests, _ = build_trace(SEED + repeat_seed)
        else:
            objects, requests = base_objects, base_requests
        for m_budget in M_BUDGETS:
            for epsilon_norm in EPSILON_NORMS:
                epsilon_abs = sim.denormalize_epsilon(epsilon_norm, token_ref)
                for baseline in BASELINES:
                    cfg = sim.config_with(
                        BASE_CFG,
                        seed=SEED + repeat_seed,
                        m_budget_mb=m_budget,
                        epsilon=epsilon_abs,
                        offload_keep_threshold=baseline["offload_keep_threshold"],
                    )
                    metrics, _ = sim.run_one(
                        objects,
                        requests,
                        cfg,
                        policy=baseline["policy"],
                    )
                    row = sim.metrics_row(metrics, "H4")
                    row["repeat_seed"] = repeat_seed
                    row["method"] = baseline["method"]
                    row["epsilon_norm_cfg"] = round(epsilon_norm, 9)
                    row["qloss_within_budget"] = row["qloss_peak_abs"] <= epsilon_abs + 1e-9
                    rows.append(row)
    return rows, trace_source, object_count, request_count, token_ref


def summarize(rows: Sequence[dict]) -> List[dict]:
    summaries: List[dict] = []
    for m_budget in M_BUDGETS:
        for epsilon_norm in EPSILON_NORMS:
            cell = [
                row
                for row in rows
                if row["m_budget_mb"] == m_budget and row["epsilon_norm_cfg"] == round(epsilon_norm, 9)
            ]
            method_means = {}
            for baseline in BASELINES:
                method = baseline["method"]
                values = [row["ttft_p95_ms"] for row in cell if row["method"] == method]
                q_ok = all(row["qloss_within_budget"] for row in cell if row["method"] == method)
                method_means[method] = {
                    "p95": sum(values) / max(len(values), 1),
                    "qloss_ok": q_ok,
                }
            tms_p95 = method_means["tms-tiered"]["p95"]
            score_only_p95 = method_means["lpe-only"]["p95"]
            improvement_vs_score = (score_only_p95 - tms_p95) / score_only_p95 * 100.0
            valid_baselines = [
                value["p95"]
                for method, value in method_means.items()
                if method != "tms-tiered" and value["qloss_ok"]
            ]
            baseline_best = min(valid_baselines) if valid_baselines else float("inf")
            improvement = (
                (baseline_best - tms_p95) / baseline_best * 100.0
                if baseline_best != float("inf")
                else 0.0
            )
            summaries.append(
                {
                    "experiment": "H4",
                    "m_budget_mb": m_budget,
                    "epsilon_abs": round(cell[0]["epsilon_abs"], 6) if cell else 0.0,
                    "epsilon_norm": round(epsilon_norm, 9),
                    "score_only_mean_p95_ms": round(score_only_p95, 6),
                    "tms_mean_p95_ms": round(tms_p95, 6),
                    "tms_improvement_vs_score_pct": round(improvement_vs_score, 6),
                    "best_baseline_mean_p95_ms": round(baseline_best, 6),
                    "tms_improvement_vs_best_pct": round(improvement, 6),
                    "tms_qloss_ok_all_repeats": method_means["tms-tiered"]["qloss_ok"],
                    "passes_tiered_below_score": tms_p95 < score_only_p95,
                    "passes_25pct_gate": baseline_best != float("inf") and tms_p95 <= baseline_best * 0.75,
                }
            )
    return summaries


def plot(rows: Sequence[dict]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows=len(M_BUDGETS),
        ncols=2,
        figsize=(12, 10),
    )
    for row_idx, m_budget in enumerate(M_BUDGETS):
        ax_p95 = axes[row_idx][0]
        ax_q = axes[row_idx][1]
        for baseline in BASELINES:
            method = baseline["method"]
            label = METHOD_LABELS.get(method, method)
            p95_values = []
            qloss_values = []
            for epsilon_norm in EPSILON_NORMS:
                values = [
                    row["ttft_p95_ms"]
                    for row in rows
                    if row["method"] == method
                    and row["m_budget_mb"] == m_budget
                    and row["epsilon_norm_cfg"] == round(epsilon_norm, 9)
                ]
                q_values = [
                    row["qloss_peak_norm"]
                    for row in rows
                    if row["method"] == method
                    and row["m_budget_mb"] == m_budget
                    and row["epsilon_norm_cfg"] == round(epsilon_norm, 9)
                ]
                p95_values.append(sum(values) / max(len(values), 1))
                qloss_values.append(sum(q_values) / max(len(q_values), 1))
            ax_p95.plot(EPSILON_NORMS, p95_values, marker="o", label=label)
            ax_q.plot(EPSILON_NORMS, qloss_values, marker="o", label=label)
        ax_p95.set_title(f"Memory budget = {int(m_budget)} MB | p95 TTFT")
        ax_q.set_title(f"Memory budget = {int(m_budget)} MB | QLoss norm")
        ax_p95.set_xlabel("epsilon_norm")
        ax_q.set_xlabel("epsilon_norm")
        ax_p95.set_ylabel("p95 TTFT (ms)")
        ax_q.set_ylabel("QLoss peak / token_ref")
        ax_p95.set_xticks(EPSILON_NORMS)
        ax_q.set_xticks(EPSILON_NORMS)
        ax_p95.tick_params(axis="both", labelsize=8)
        ax_q.tick_params(axis="both", labelsize=8)
        ax_p95.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax_q.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax_q.plot(
            EPSILON_NORMS,
            EPSILON_NORMS,
            linestyle="--",
            color="black",
            linewidth=1,
            label="epsilon_norm",
        )
        ax_p95.legend(fontsize=7, loc="best")
        ax_q.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "h4_summary_3x2.png", dpi=180)
    plt.close(fig)


def main() -> None:
    tier_source = load_tiers()
    rows, trace_source, object_count, request_count, token_ref = run_matrix()
    summary_rows = summarize(rows)
    sim.write_csv(OUT_DIR / "h4_results.csv", rows)
    sim.write_csv(OUT_DIR / "h4_summary.csv", summary_rows)
    sim.write_json(
        OUT_DIR / "config.json",
        {
            "experiment": "H4",
            "trace_source": trace_source,
            "objects": object_count,
            "requests": request_count,
            "token_ref": token_ref,
            "tier_source": tier_source,
            "tiers": sim.TIERS,
            "m_budgets": M_BUDGETS,
            "epsilon_norms": EPSILON_NORMS,
            "epsilon_abs_values": [
                round(sim.denormalize_epsilon(value, token_ref), 6) for value in EPSILON_NORMS
            ],
            "repeats": REPEATS,
            "baselines": BASELINES,
            "base_config": asdict(BASE_CFG),
            "pass_gate": "TMS p95 improves >=25% vs best baseline and qloss_peak_norm <= epsilon_norm",
        },
    )
    plot(rows)
    print(json.dumps({"experiment": "H4", "rows": len(rows), "out_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
