#!/usr/bin/env python3
"""H2：restore-vs-recompute 阈值验证。"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import sim


BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "out" / "h2"
MAX_SESSIONS = 200
SEED = 2
TRACE_REQUESTS = 500
BW_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
RRS_MODES = ["always-restore", "always-recompute", "rrs"]
BASE_CFG = sim.SimConfig(
    seed=SEED,
    trace_requests=TRACE_REQUESTS,
    m_budget_mb=420.0,
    epsilon=4.0,
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


def cost_snapshot(objects: Sequence[sim.KVObject], cfg: sim.SimConfig) -> dict:
    sample = list(objects)[: min(len(objects), 200)]
    if not sample:
        return {"avg_restore_full_ms": 0.0, "avg_recomp_full_ms": 0.0}
    return {
        "avg_restore_full_ms": round(
            statistics.mean(sim.c_restore_ms(obj, "full", cfg) for obj in sample),
            6,
        ),
        "avg_recomp_full_ms": round(
            statistics.mean(sim.c_recomp_ms(obj, "full", cfg) for obj in sample),
            6,
        ),
    }


def run_matrix(
    objects: Sequence[sim.KVObject],
    requests: Sequence[sim.Request],
) -> List[dict]:
    rows: List[dict] = []
    for bw in BW_GRID:
        cfg = sim.config_with(BASE_CFG, bw_gbps=bw)
        costs = cost_snapshot(objects, cfg)
        for rrs_mode in RRS_MODES:
            metrics, _ = sim.run_one(
                objects,
                requests,
                cfg,
                policy="tiered",
                rrs_mode=rrs_mode,
            )
            row = sim.metrics_row(metrics, "H2")
            row.update(costs)
            row["critical_bw_gbps"] = round(
                cfg.mu_kv_mb_per_token / max(cfg.c_re_ms_per_token, 1e-9),
                6,
            )
            rows.append(row)
    return rows


def summarize(rows: Sequence[dict]) -> List[dict]:
    summaries: List[dict] = []
    for bw in BW_GRID:
        cell = [row for row in rows if row["bw_gbps"] == bw]
        by_mode = {row["rrs_mode"]: row for row in cell}
        fixed_best = min(
            by_mode["always-restore"]["ttft_p95_ms"],
            by_mode["always-recompute"]["ttft_p95_ms"],
        )
        rrs_p95 = by_mode["rrs"]["ttft_p95_ms"]
        summaries.append(
            {
                "experiment": "H2",
                "bw_gbps": bw,
                "rrs_p95_ms": rrs_p95,
                "best_fixed_p95_ms": fixed_best,
                "rrs_not_worse_than_best_fixed": rrs_p95 <= fixed_best + 1e-9,
                "rrs_gap_vs_best_fixed_pct": round((rrs_p95 - fixed_best) / fixed_best * 100.0, 6),
                "rrs_recompute_ratio": by_mode["rrs"]["recompute_ratio"],
                "rrs_restore_ratio": by_mode["rrs"]["restore_ratio"],
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
    for mode in RRS_MODES:
        sub = [row for row in rows if row["rrs_mode"] == mode]
        sub.sort(key=lambda row: row["bw_gbps"])
        plt.plot(
            [row["bw_gbps"] for row in sub],
            [row["ttft_p95_ms"] for row in sub],
            marker="o",
            label=mode,
        )
    plt.xlabel("BW (GB/s)")
    plt.ylabel("p95 TTFT (ms)")
    plt.xscale("log", base=2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "h2_bw_vs_p95.png", dpi=160)
    plt.close()


def main() -> None:
    objects, requests, trace_source = build_trace()
    token_ref = sim.token_ref_for_objects(objects)
    rows = run_matrix(objects, requests)
    summary_rows = summarize(rows)
    sim.write_csv(OUT_DIR / "h2_results.csv", rows)
    sim.write_csv(OUT_DIR / "h2_summary.csv", summary_rows)
    sim.write_json(
        OUT_DIR / "config.json",
        {
            "experiment": "H2",
            "trace_source": trace_source,
            "objects": len(objects),
            "requests": len(requests),
            "token_ref": token_ref,
            "bw_grid": BW_GRID,
            "rrs_modes": RRS_MODES,
            "epsilon_abs": BASE_CFG.epsilon,
            "epsilon_norm": sim.normalize_qloss(BASE_CFG.epsilon, token_ref),
            "base_config": asdict(BASE_CFG),
            "pass_gate": "RRS p95 is not worse than the best fixed mode in most bandwidth cells; report epsilon_abs and epsilon_norm together",
        },
    )
    plot(rows)
    print(json.dumps({"experiment": "H2", "rows": len(rows), "out_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
