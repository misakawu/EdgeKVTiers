#!/usr/bin/env python3
"""H1: lifecycle policy effectiveness.

Run directly with no command-line arguments:
    python h1.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import sim


BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "out" / "h1"
MAX_SESSIONS = 200
SEED = 1
TRACE_REQUESTS = 500
M_BUDGETS = [3000.0, 3400.0, 3800.0]
POLICIES = ["lru", "lfu", "score", "tiered"]
BASE_CFG = sim.SimConfig(
    seed=SEED,
    trace_requests=TRACE_REQUESTS,
    epsilon=4.0,
    offload_keep_threshold=1.1,
)


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


def build_trace() -> Tuple[List[sim.KVObject], List[sim.Request], str]:
    if sim.DEFAULT_SHAREGPT_PATH.exists():
        objects, requests = sim.load_sharegpt_trace(
            sim.DEFAULT_SHAREGPT_PATH,
            max_sessions=MAX_SESSIONS,
        )
        objects, requests = limit_trace(objects, requests, TRACE_REQUESTS)
        return objects, requests, str(sim.DEFAULT_SHAREGPT_PATH)
    objects, requests = sim.make_synthetic_trace(SEED, TRACE_REQUESTS)
    return objects, requests, "synthetic"


def run_matrix(
    objects: Sequence[sim.KVObject],
    requests: Sequence[sim.Request],
) -> List[dict]:
    rows: List[dict] = []
    for m_budget in M_BUDGETS:
        for policy in POLICIES:
            cfg = sim.config_with(BASE_CFG, m_budget_mb=m_budget)
            metrics, _ = sim.run_one(objects, requests, cfg, policy=policy)
            row = sim.metrics_row(metrics, "H1")
            row["lpe_label"] = "LPE-score" if policy == "score" else policy
            rows.append(row)
    return rows


def summarize(rows: Sequence[dict]) -> List[dict]:
    summaries: List[dict] = []
    for m_budget in M_BUDGETS:
        cell = [row for row in rows if row["m_budget_mb"] == m_budget]
        by_policy = {row["policy"]: row for row in cell}
        score_p95 = by_policy["score"]["ttft_p95_ms"]
        lru_p95 = by_policy["lru"]["ttft_p95_ms"]
        lfu_p95 = by_policy["lfu"]["ttft_p95_ms"]
        best_baseline = min(lru_p95, lfu_p95)
        summaries.append(
            {
                "experiment": "H1",
                "m_budget_mb": m_budget,
                "score_p95_ms": score_p95,
                "lru_p95_ms": lru_p95,
                "lfu_p95_ms": lfu_p95,
                "improvement_vs_lru_pct": round((lru_p95 - score_p95) / lru_p95 * 100.0, 6),
                "improvement_vs_best_lru_lfu_pct": round(
                    (best_baseline - score_p95) / best_baseline * 100.0,
                    6,
                ),
                "passes_20pct_lru_gate": score_p95 <= lru_p95 * 0.80,
            }
        )
    return summaries


def plot(rows: Sequence[dict]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for policy in POLICIES:
        sub = [row for row in rows if row["policy"] == policy]
        sub.sort(key=lambda row: row["m_budget_mb"])
        plt.plot(
            [row["m_budget_mb"] for row in sub],
            [row["ttft_p95_ms"] for row in sub],
            marker="o",
            label="LPE-score" if policy == "score" else policy,
        )
    plt.xlabel("M_budget (MB)")
    plt.ylabel("p95 TTFT (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "h1_memory_vs_p95.png", dpi=160)
    plt.close()


def main() -> None:
    objects, requests, trace_source = build_trace()
    token_ref = sim.token_ref_for_objects(objects)
    rows = run_matrix(objects, requests)
    summary_rows = summarize(rows)
    sim.write_csv(OUT_DIR / "h1_results.csv", rows)
    sim.write_csv(OUT_DIR / "h1_summary.csv", summary_rows)
    sim.write_json(
        OUT_DIR / "config.json",
        {
            "experiment": "H1",
            "trace_source": trace_source,
            "objects": len(objects),
            "requests": len(requests),
            "token_ref": token_ref,
            "policies": POLICIES,
            "m_budgets": M_BUDGETS,
            "epsilon_abs": BASE_CFG.epsilon,
            "epsilon_norm": sim.normalize_qloss(BASE_CFG.epsilon, token_ref),
            "base_config": asdict(BASE_CFG),
            "pass_gate": "score p95 <= 0.8 * lru p95 in tight-memory cells; report epsilon_abs and epsilon_norm together",
        },
    )
    plot(rows)
    print(json.dumps({"experiment": "H1", "rows": len(rows), "out_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
