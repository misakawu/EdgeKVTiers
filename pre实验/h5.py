#!/usr/bin/env python3
"""H5：quality-bandwidth 相变面验证。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sim


BASE_DIR = Path.cwd()
OUT_DIR = BASE_DIR / "out" / "h5"
TIER_CONFIG_PATH = BASE_DIR / "out" / "tier_config.json"
MAX_SESSIONS = 200
SEED = 5
TRACE_REQUESTS = 500
OBJECT_SAMPLE = 200
BW_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
EPS_NORM_GRID = [0.0005, 0.001, 0.002, 0.004, 0.008]
BASE_CFG = sim.SimConfig(seed=SEED, trace_requests=TRACE_REQUESTS)
TIER_RANK = {q: i for i, q in enumerate(sim.TIER_ORDER)}


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


def representative_objects(objects: Sequence[sim.KVObject]) -> List[sim.KVObject]:
    ranked = sorted(objects, key=lambda obj: obj.p_reuse * obj.n_tokens, reverse=True)
    return ranked[: min(OBJECT_SAMPLE, len(ranked))]


def feasible_tiers(obj: sim.KVObject, epsilon: float) -> List[str]:
    tiers = [q for q in sim.TIER_ORDER if sim.qloss(obj, q) <= epsilon + 1e-9]
    return tiers or ["full"]


def offline_cost(obj: sim.KVObject, q: str, cfg: sim.SimConfig, epsilon: float) -> float:
    action_cost = obj.p_reuse * min(
        sim.c_restore_ms(obj, q, cfg),
        sim.c_recomp_ms(obj, q, cfg),
    )
    memory_pressure = (1.0 - obj.p_reuse) * sim.size_mb(obj, q, cfg) / max(cfg.bw_gbps, 1e-9)
    quality_penalty = sim.qloss(obj, q) * sim.c_recomp_ms(obj, "full", cfg) / max(epsilon, 1e-9)
    return action_cost + memory_pressure + quality_penalty


def predicted_cost(obj: sim.KVObject, q: str, cfg: sim.SimConfig, epsilon: float) -> float:
    restore_or_recompute = min(
        sim.c_restore_ms(obj, q, cfg),
        sim.c_recomp_ms(obj, q, cfg),
    )
    compression_bonus = sim.size_mb(obj, "full", cfg) - sim.size_mb(obj, q, cfg)
    bw_weight = 1.0 / (cfg.bw_gbps + 1.0)
    quality_weight = sim.c_recomp_ms(obj, "full", cfg) / max(math.sqrt(epsilon), 1e-9)
    return restore_or_recompute - compression_bonus * bw_weight + sim.qloss(obj, q) * quality_weight


def qstar_offline(obj: sim.KVObject, cfg: sim.SimConfig, epsilon: float) -> str:
    return min(feasible_tiers(obj, epsilon), key=lambda q: offline_cost(obj, q, cfg, epsilon))


def qstar_pred(obj: sim.KVObject, cfg: sim.SimConfig, epsilon: float) -> str:
    return min(feasible_tiers(obj, epsilon), key=lambda q: predicted_cost(obj, q, cfg, epsilon))


def kendall_tau(xs: Sequence[int], ys: Sequence[int]) -> float:
    concordant = discordant = 0
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 1.0
    return (concordant - discordant) / denom


def dominant_tier(tiers: Sequence[str]) -> str:
    counts: Dict[str, int] = {}
    for q in tiers:
        counts[q] = counts.get(q, 0) + 1
    return max(counts, key=lambda q: (counts[q], -TIER_RANK[q]))


def run_grid(objects: Sequence[sim.KVObject]) -> Tuple[List[dict], List[dict], List[dict]]:
    object_rows: List[dict] = []
    grid_rows: List[dict] = []
    tau_rows: List[dict] = []
    token_ref = sim.token_ref_for_objects(objects)
    for bw in BW_GRID:
        for epsilon_norm in EPS_NORM_GRID:
            epsilon_abs = sim.denormalize_epsilon(epsilon_norm, token_ref)
            cfg = sim.config_with(BASE_CFG, bw_gbps=bw, epsilon=epsilon_abs)
            offline_ranks: List[int] = []
            pred_ranks: List[int] = []
            offline_tiers: List[str] = []
            pred_tiers: List[str] = []
            matches = 0
            for obj in objects:
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
            agreement = matches / max(len(objects), 1)
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
    return grid_rows, object_rows, tau_rows


def plot(grid_rows: Sequence[dict]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = []
    for epsilon_norm in EPS_NORM_GRID:
        row = []
        for bw in BW_GRID:
            cell = next(
                r
                for r in grid_rows
                if r["bw_gbps"] == bw and r["epsilon_norm"] == round(epsilon_norm, 9)
            )
            row.append(TIER_RANK[cell["dominant_q_offline"]])
        data.append(row)

    plt.figure(figsize=(8, 4.5))
    im = plt.imshow(data, aspect="auto", origin="lower")
    plt.xticks(range(len(BW_GRID)), [str(v) for v in BW_GRID])
    plt.yticks(range(len(EPS_NORM_GRID)), [str(v) for v in EPS_NORM_GRID])
    plt.xlabel("BW (GB/s)")
    plt.ylabel("epsilon_norm")
    cbar = plt.colorbar(im, ticks=list(TIER_RANK.values()))
    cbar.ax.set_yticklabels(list(TIER_RANK.keys()))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "h5_phase_heatmap.png", dpi=160)
    plt.close()


def main() -> None:
    tier_source = load_tiers()
    objects, requests, trace_source = build_trace()
    sample = representative_objects(objects)
    token_ref = sim.token_ref_for_objects(sample)
    grid_rows, object_rows, tau_rows = run_grid(sample)
    sim.write_csv(OUT_DIR / "h5_grid.csv", grid_rows)
    sim.write_csv(OUT_DIR / "h5_objects.csv", object_rows)
    sim.write_csv(OUT_DIR / "h5_tau.csv", tau_rows)
    sim.write_json(
        OUT_DIR / "config.json",
        {
            "experiment": "H5",
            "trace_source": trace_source,
            "objects_total": len(objects),
            "requests": len(requests),
            "objects_sampled": len(sample),
            "token_ref": token_ref,
            "tier_source": tier_source,
            "tiers": sim.TIERS,
            "bw_grid": BW_GRID,
            "epsilon_norm_grid": EPS_NORM_GRID,
            "epsilon_abs_grid": [
                round(sim.denormalize_epsilon(value, token_ref), 6) for value in EPS_NORM_GRID
            ],
            "base_config": asdict(BASE_CFG),
            "pass_gate": "Kendall-tau >= 0.8 and heatmap shows block-like regions under epsilon_norm grid",
        },
    )
    plot(grid_rows)
    print(json.dumps({"experiment": "H5", "rows": len(grid_rows), "out_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
