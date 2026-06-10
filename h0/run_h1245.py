#!/usr/bin/env python3
"""Run H1/H2/H4/H5 experiments on top of the H0 trace replay stack."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import run_h0
import h3_lmcache_adapter


REPO_ROOT = run_h0.REPO_ROOT
sim = run_h0.sim

DEFAULT_OUT_DIR = REPO_ROOT / "out" / "h1245"
DEFAULT_MAX_SESSIONS = 200
DEFAULT_MAX_REQUESTS = 500
DEFAULT_H3_EVENT_SAMPLE = 200

H1_M_BUDGETS = [3000.0, 3400.0, 3800.0]
H1_POLICIES = ["lru", "lfu", "score", "tiered"]

H2_BW_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
H2_RRS_MODES = ["always-restore", "always-recompute", "rrs"]

H4_M_BUDGETS = [1600.0, 2200.0, 3000.0]
H4_EPSILON_NORMS = [0.002, 0.004, 0.008]
H4_REPEATS = [1, 2, 3, 4, 5]
H4_BASELINES = [
    {"method": "ragcache_pgdsf_approx", "policy": "pgdsf", "offload_keep_threshold": 1.1},
    {"method": "static-int4", "policy": "static-int4", "offload_keep_threshold": 1.1},
    {"method": "lpe-only", "policy": "score", "offload_keep_threshold": 1.1},
    {"method": "tms-tiered", "policy": "tiered", "offload_keep_threshold": 1.1},
]

H5_BW_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
H5_EPSILON_NORM_GRID = [0.0005, 0.001, 0.002, 0.004, 0.008]
H5_OBJECT_SAMPLE = 200
TIER_RANK = {q: i for i, q in enumerate(sim.TIER_ORDER)}


def parse_experiments(value: str) -> List[str]:
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        return ["h1", "h2", "h4", "h5"]
    valid = {"h1", "h2", "h4", "h5"}
    unknown = [item for item in requested if item not in valid]
    if unknown:
        raise ValueError(f"unknown experiments: {', '.join(unknown)}")
    return requested


def trace_config_from_args(args: argparse.Namespace, seed: int) -> dict:
    cfg = {
        "source": args.trace_source,
        "seed": seed,
        "requests": args.max_requests,
        "max_requests": args.max_requests,
        "max_sessions": args.max_sessions,
    }
    if args.trace_path:
        cfg["path"] = args.trace_path
    return cfg


def load_h0_trace(args: argparse.Namespace, seed: int):
    cfg = sim.SimConfig(seed=seed, trace_requests=args.max_requests)
    return run_h0.load_trace(trace_config_from_args(args, seed), cfg)


def write_config(out_dir: Path, data: dict) -> None:
    run_h0.write_json(out_dir / "config.json", data)


def write_h3_outputs(
    *,
    out_root: Path,
    objects: Sequence[object],
    requests: Sequence[object],
    trace_source: str,
    sample_events: bool,
    sample_limit: int,
) -> None:
    token_ref = sim.token_ref_for_objects(objects)
    h3_lmcache_adapter.write_h3_contract(out_root, trace_source=trace_source, token_ref=token_ref)
    if not sample_events:
        return

    cfg = sim.SimConfig(
        seed=3,
        trace_requests=min(len(requests), sample_limit),
        m_budget_mb=520.0,
        bw_gbps=2.0,
        epsilon=sim.denormalize_epsilon(0.0002, token_ref),
        d_deser_ms=6.0,
        warmup_requests=0,
    )
    _, raw_events = sim.run_one(
        objects,
        list(requests)[:sample_limit],
        cfg,
        policy="tiered",
        rrs_mode="rrs",
        emit_events=True,
    )
    enriched = run_h0.enrich_events(raw_events, objects, cfg, "h3_edge_sample")
    h3_lmcache_adapter.write_h3_event_sample(out_root, enriched, limit=sample_limit)


def run_h1(objects: Sequence[object], requests: Sequence[object], trace_source: str, out_dir: Path) -> dict:
    started = time.perf_counter()
    base_cfg = sim.SimConfig(
        seed=1,
        trace_requests=len(requests),
        epsilon=4.0,
        offload_keep_threshold=1.1,
    )
    rows: List[dict] = []
    for m_budget in H1_M_BUDGETS:
        for policy in H1_POLICIES:
            cfg = sim.config_with(base_cfg, m_budget_mb=m_budget)
            metrics, _ = sim.run_one(objects, requests, cfg, policy=policy)
            row = sim.metrics_row(metrics, "H1")
            row["lpe_label"] = "LPE-score" if policy == "score" else policy
            rows.append(row)

    summary_rows: List[dict] = []
    for m_budget in H1_M_BUDGETS:
        cell = [row for row in rows if row["m_budget_mb"] == m_budget]
        by_policy = {row["policy"]: row for row in cell}
        score_p95 = by_policy["score"]["ttft_p95_ms"]
        lru_p95 = by_policy["lru"]["ttft_p95_ms"]
        lfu_p95 = by_policy["lfu"]["ttft_p95_ms"]
        best_baseline = min(lru_p95, lfu_p95)
        summary_rows.append(
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

    token_ref = sim.token_ref_for_objects(objects)
    run_h0.write_csv(out_dir / "h1_results.csv", rows)
    run_h0.write_csv(out_dir / "h1_summary.csv", summary_rows)
    write_config(
        out_dir,
        {
            "experiment": "H1",
            "trace_source": trace_source,
            "objects": len(objects),
            "requests": len(requests),
            "token_ref": token_ref,
            "policies": H1_POLICIES,
            "m_budgets": H1_M_BUDGETS,
            "epsilon_abs": base_cfg.epsilon,
            "epsilon_norm": sim.normalize_qloss(base_cfg.epsilon, token_ref),
            "base_config": asdict(base_cfg),
        },
    )
    return {"experiment": "H1", "rows": len(rows), "summary_rows": len(summary_rows), "elapsed_s": round(time.perf_counter() - started, 6)}


def cost_snapshot(objects: Sequence[object], cfg) -> dict:
    sample = list(objects)[: min(len(objects), 200)]
    if not sample:
        return {"avg_restore_full_ms": 0.0, "avg_recomp_full_ms": 0.0}
    return {
        "avg_restore_full_ms": round(statistics.mean(sim.c_restore_ms(obj, "full", cfg) for obj in sample), 6),
        "avg_recomp_full_ms": round(statistics.mean(sim.c_recomp_ms(obj, "full", cfg) for obj in sample), 6),
    }


def run_h2(objects: Sequence[object], requests: Sequence[object], trace_source: str, out_dir: Path) -> dict:
    started = time.perf_counter()
    base_cfg = sim.SimConfig(seed=2, trace_requests=len(requests), m_budget_mb=420.0, epsilon=4.0)
    rows: List[dict] = []
    for bw in H2_BW_GRID:
        cfg = sim.config_with(base_cfg, bw_gbps=bw)
        costs = cost_snapshot(objects, cfg)
        for rrs_mode in H2_RRS_MODES:
            metrics, _ = sim.run_one(objects, requests, cfg, policy="tiered", rrs_mode=rrs_mode)
            row = sim.metrics_row(metrics, "H2")
            row.update(costs)
            row["critical_bw_gbps"] = round(cfg.mu_kv_mb_per_token / max(cfg.c_re_ms_per_token, 1e-9), 6)
            rows.append(row)

    summary_rows: List[dict] = []
    for bw in H2_BW_GRID:
        cell = [row for row in rows if row["bw_gbps"] == bw]
        by_mode = {row["rrs_mode"]: row for row in cell}
        fixed_best = min(
            by_mode["always-restore"]["ttft_p95_ms"],
            by_mode["always-recompute"]["ttft_p95_ms"],
        )
        rrs_p95 = by_mode["rrs"]["ttft_p95_ms"]
        summary_rows.append(
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

    token_ref = sim.token_ref_for_objects(objects)
    run_h0.write_csv(out_dir / "h2_results.csv", rows)
    run_h0.write_csv(out_dir / "h2_summary.csv", summary_rows)
    write_config(
        out_dir,
        {
            "experiment": "H2",
            "trace_source": trace_source,
            "objects": len(objects),
            "requests": len(requests),
            "token_ref": token_ref,
            "bw_grid": H2_BW_GRID,
            "rrs_modes": H2_RRS_MODES,
            "epsilon_abs": base_cfg.epsilon,
            "epsilon_norm": sim.normalize_qloss(base_cfg.epsilon, token_ref),
            "base_config": asdict(base_cfg),
        },
    )
    return {"experiment": "H2", "rows": len(rows), "summary_rows": len(summary_rows), "elapsed_s": round(time.perf_counter() - started, 6)}


def run_h4(objects: Sequence[object], requests: Sequence[object], trace_source: str, out_dir: Path) -> dict:
    started = time.perf_counter()
    base_cfg = sim.SimConfig(seed=4, trace_requests=len(requests), bw_gbps=8.0)
    token_ref = sim.token_ref_for_objects(objects)
    rows: List[dict] = []
    for repeat_seed in H4_REPEATS:
        for m_budget in H4_M_BUDGETS:
            for epsilon_norm in H4_EPSILON_NORMS:
                epsilon_abs = sim.denormalize_epsilon(epsilon_norm, token_ref)
                for baseline in H4_BASELINES:
                    cfg = sim.config_with(
                        base_cfg,
                        seed=4 + repeat_seed,
                        m_budget_mb=m_budget,
                        epsilon=epsilon_abs,
                        offload_keep_threshold=baseline["offload_keep_threshold"],
                    )
                    metrics, _ = sim.run_one(objects, requests, cfg, policy=baseline["policy"])
                    row = sim.metrics_row(metrics, "H4")
                    row["repeat_seed"] = repeat_seed
                    row["method"] = baseline["method"]
                    row["epsilon_norm_cfg"] = round(epsilon_norm, 9)
                    row["qloss_within_budget"] = row["qloss_peak_abs"] <= epsilon_abs + 1e-9
                    rows.append(row)

    summary_rows: List[dict] = []
    for m_budget in H4_M_BUDGETS:
        for epsilon_norm in H4_EPSILON_NORMS:
            cell = [
                row
                for row in rows
                if row["m_budget_mb"] == m_budget and row["epsilon_norm_cfg"] == round(epsilon_norm, 9)
            ]
            method_means = {}
            for baseline in H4_BASELINES:
                method = baseline["method"]
                values = [row["ttft_p95_ms"] for row in cell if row["method"] == method]
                q_ok = all(row["qloss_within_budget"] for row in cell if row["method"] == method)
                method_means[method] = {"p95": sum(values) / max(len(values), 1), "qloss_ok": q_ok}
            tms_p95 = method_means["tms-tiered"]["p95"]
            score_only_p95 = method_means["lpe-only"]["p95"]
            valid_baselines = [
                value["p95"]
                for method, value in method_means.items()
                if method != "tms-tiered" and value["qloss_ok"]
            ]
            baseline_best = min(valid_baselines) if valid_baselines else float("inf")
            summary_rows.append(
                {
                    "experiment": "H4",
                    "m_budget_mb": m_budget,
                    "epsilon_abs": round(cell[0]["epsilon_abs"], 6) if cell else 0.0,
                    "epsilon_norm": round(epsilon_norm, 9),
                    "score_only_mean_p95_ms": round(score_only_p95, 6),
                    "tms_mean_p95_ms": round(tms_p95, 6),
                    "tms_improvement_vs_score_pct": round((score_only_p95 - tms_p95) / score_only_p95 * 100.0, 6),
                    "best_baseline_mean_p95_ms": round(baseline_best, 6),
                    "tms_improvement_vs_best_pct": round(
                        (baseline_best - tms_p95) / baseline_best * 100.0
                        if baseline_best != float("inf")
                        else 0.0,
                        6,
                    ),
                    "tms_qloss_ok_all_repeats": method_means["tms-tiered"]["qloss_ok"],
                    "passes_tiered_below_score": tms_p95 < score_only_p95,
                    "passes_25pct_gate": baseline_best != float("inf") and tms_p95 <= baseline_best * 0.75,
                }
            )

    run_h0.write_csv(out_dir / "h4_results.csv", rows)
    run_h0.write_csv(out_dir / "h4_summary.csv", summary_rows)
    write_config(
        out_dir,
        {
            "experiment": "H4",
            "trace_source": trace_source,
            "objects": len(objects),
            "requests": len(requests),
            "token_ref": token_ref,
            "m_budgets": H4_M_BUDGETS,
            "epsilon_norms": H4_EPSILON_NORMS,
            "epsilon_abs_values": [round(sim.denormalize_epsilon(value, token_ref), 6) for value in H4_EPSILON_NORMS],
            "repeats": H4_REPEATS,
            "baselines": H4_BASELINES,
            "base_config": asdict(base_cfg),
        },
    )
    return {"experiment": "H4", "rows": len(rows), "summary_rows": len(summary_rows), "elapsed_s": round(time.perf_counter() - started, 6)}


def representative_objects(objects: Sequence[object]) -> List[object]:
    ranked = sorted(objects, key=lambda obj: obj.p_reuse * obj.n_tokens, reverse=True)
    return ranked[: min(H5_OBJECT_SAMPLE, len(ranked))]


def feasible_tiers(obj, epsilon: float) -> List[str]:
    tiers = [q for q in sim.TIER_ORDER if sim.qloss(obj, q) <= epsilon + 1e-9]
    return tiers or ["full"]


def offline_cost(obj, q: str, cfg, epsilon: float) -> float:
    action_cost = obj.p_reuse * min(sim.c_restore_ms(obj, q, cfg), sim.c_recomp_ms(obj, q, cfg))
    memory_pressure = (1.0 - obj.p_reuse) * sim.size_mb(obj, q, cfg) / max(cfg.bw_gbps, 1e-9)
    quality_penalty = sim.qloss(obj, q) * sim.c_recomp_ms(obj, "full", cfg) / max(epsilon, 1e-9)
    return action_cost + memory_pressure + quality_penalty


def predicted_cost(obj, q: str, cfg, epsilon: float) -> float:
    restore_or_recompute = min(sim.c_restore_ms(obj, q, cfg), sim.c_recomp_ms(obj, q, cfg))
    compression_bonus = sim.size_mb(obj, "full", cfg) - sim.size_mb(obj, q, cfg)
    bw_weight = 1.0 / (cfg.bw_gbps + 1.0)
    quality_weight = sim.c_recomp_ms(obj, "full", cfg) / max(math.sqrt(epsilon), 1e-9)
    return restore_or_recompute - compression_bonus * bw_weight + sim.qloss(obj, q) * quality_weight


def qstar_offline(obj, cfg, epsilon: float) -> str:
    return min(feasible_tiers(obj, epsilon), key=lambda q: offline_cost(obj, q, cfg, epsilon))


def qstar_pred(obj, cfg, epsilon: float) -> str:
    return min(feasible_tiers(obj, epsilon), key=lambda q: predicted_cost(obj, q, cfg, epsilon))


def kendall_tau(xs: Sequence[int], ys: Sequence[int]) -> float:
    concordant = discordant = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return 1.0 if denom == 0 else (concordant - discordant) / denom


def dominant_tier(tiers: Sequence[str]) -> str:
    counts: Dict[str, int] = {}
    for q in tiers:
        counts[q] = counts.get(q, 0) + 1
    return max(counts, key=lambda q: (counts[q], -TIER_RANK[q]))


def run_h5(objects: Sequence[object], requests: Sequence[object], trace_source: str, out_dir: Path) -> dict:
    started = time.perf_counter()
    base_cfg = sim.SimConfig(seed=5, trace_requests=len(requests))
    sample = representative_objects(objects)
    token_ref = sim.token_ref_for_objects(sample)
    object_rows: List[dict] = []
    grid_rows: List[dict] = []
    tau_rows: List[dict] = []

    for bw in H5_BW_GRID:
        for epsilon_norm in H5_EPSILON_NORM_GRID:
            epsilon_abs = sim.denormalize_epsilon(epsilon_norm, token_ref)
            cfg = sim.config_with(base_cfg, bw_gbps=bw, epsilon=epsilon_abs)
            offline_ranks: List[int] = []
            pred_ranks: List[int] = []
            offline_tiers: List[str] = []
            pred_tiers: List[str] = []
            matches = 0
            for obj in sample:
                q_offline = qstar_offline(obj, cfg, epsilon_abs)
                q_pred = qstar_pred(obj, cfg, epsilon_abs)
                offline_tiers.append(q_offline)
                pred_tiers.append(q_pred)
                offline_ranks.append(TIER_RANK[q_offline])
                pred_ranks.append(TIER_RANK[q_pred])
                matches += int(q_offline == q_pred)
                object_rows.append(
                    {
                        "experiment": "H5",
                        "bw_gbps": bw,
                        "epsilon_abs": round(epsilon_abs, 6),
                        "epsilon_norm": round(epsilon_norm, 9),
                        "token_ref": token_ref,
                        "object_id": obj.object_id,
                        "object_type": obj.object_type,
                        "n_tokens": obj.n_tokens,
                        "p_reuse": round(obj.p_reuse, 6),
                        "q_star_offline": q_offline,
                        "q_star_pred": q_pred,
                        "offline_cost": round(offline_cost(obj, q_offline, cfg, epsilon_abs), 6),
                        "predicted_cost": round(predicted_cost(obj, q_pred, cfg, epsilon_abs), 6),
                    }
                )
            tau = kendall_tau(offline_ranks, pred_ranks)
            agreement = matches / max(len(sample), 1)
            grid_rows.append(
                {
                    "experiment": "H5",
                    "bw_gbps": bw,
                    "epsilon_abs": round(epsilon_abs, 6),
                    "epsilon_norm": round(epsilon_norm, 9),
                    "token_ref": token_ref,
                    "dominant_q_offline": dominant_tier(offline_tiers),
                    "dominant_q_pred": dominant_tier(pred_tiers),
                    "kendall_tau": round(tau, 6),
                    "agreement_ratio": round(agreement, 6),
                }
            )
            tau_rows.append(
                {
                    "experiment": "H5",
                    "bw_gbps": bw,
                    "epsilon_abs": round(epsilon_abs, 6),
                    "epsilon_norm": round(epsilon_norm, 9),
                    "token_ref": token_ref,
                    "kendall_tau": round(tau, 6),
                    "agreement_ratio": round(agreement, 6),
                    "passes_08_tau_gate": tau >= 0.8,
                }
            )

    run_h0.write_csv(out_dir / "h5_grid.csv", grid_rows)
    run_h0.write_csv(out_dir / "h5_objects.csv", object_rows)
    run_h0.write_csv(out_dir / "h5_tau.csv", tau_rows)
    write_config(
        out_dir,
        {
            "experiment": "H5",
            "trace_source": trace_source,
            "objects_total": len(objects),
            "requests": len(requests),
            "objects_sampled": len(sample),
            "token_ref": token_ref,
            "bw_grid": H5_BW_GRID,
            "epsilon_norm_grid": H5_EPSILON_NORM_GRID,
            "epsilon_abs_grid": [round(sim.denormalize_epsilon(value, token_ref), 6) for value in H5_EPSILON_NORM_GRID],
            "base_config": asdict(base_cfg),
        },
    )
    return {"experiment": "H5", "rows": len(grid_rows), "object_rows": len(object_rows), "elapsed_s": round(time.perf_counter() - started, 6)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1/H2/H4/H5 experiments using the H0 trace stack.")
    parser.add_argument("--experiments", default="all", help="Comma-separated subset: h1,h2,h4,h5, or all.")
    parser.add_argument("--trace-source", default="sharegpt", choices=["sharegpt", "synthetic", "jsonl"])
    parser.add_argument("--trace-path", default=str(run_h0.DEFAULT_SHAREGPT_TRACE_PATH))
    parser.add_argument("--max-sessions", type=int, default=DEFAULT_MAX_SESSIONS)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument("--h3-contract", action="store_true", help="Write LMCache/vLLM H3 adapter contract files.")
    parser.add_argument("--emit-h3-events", action="store_true", help="Write an H0-to-H3 hook event sample JSONL.")
    parser.add_argument("--h3-event-limit", type=int, default=DEFAULT_H3_EVENT_SAMPLE)
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    experiments = parse_experiments(args.experiments)
    out_root = Path(args.out).expanduser()
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root

    objects, requests, trace_source = load_h0_trace(args, seed=1)
    token_ref = sim.token_ref_for_objects(objects)
    run_h0.write_json(
        out_root / "trace.resolved.json",
        {
            "trace_source": trace_source,
            "objects": len(objects),
            "requests": len(requests),
            "token_ref": token_ref,
            "max_sessions": args.max_sessions,
            "max_requests": args.max_requests,
        },
    )
    if args.h3_contract or args.emit_h3_events:
        write_h3_outputs(
            out_root=out_root,
            objects=objects,
            requests=requests,
            trace_source=trace_source,
            sample_events=args.emit_h3_events,
            sample_limit=args.h3_event_limit,
        )

    runners = {
        "h1": run_h1,
        "h2": run_h2,
        "h4": run_h4,
        "h5": run_h5,
    }
    summaries = []
    for experiment in experiments:
        experiment_out = out_root / experiment
        result = runners[experiment](objects, requests, trace_source, experiment_out)
        result["out_dir"] = str(experiment_out)
        summaries.append(result)

    run_h0.write_json(out_root / "summary.json", {"experiments": summaries})
    print(json.dumps({"out_dir": str(out_root), "experiments": summaries}, indent=2))


if __name__ == "__main__":
    main()
