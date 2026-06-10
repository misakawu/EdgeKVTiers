#!/usr/bin/env python3
"""Run the H0 trace replay and metrics-closure smoke experiment.

H0 is intentionally a thin, configurable wrapper around the current analytical
simulator in ``pre实验/sim.py``. It produces the stable artifacts later H1/H2/H4/H5
experiments should be able to consume: event JSONL, summary CSV, and a resolved
configuration snapshot.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PRE_SIM_PATH = REPO_ROOT / "pre实验" / "sim.py"


def load_pre_sim():
    spec = importlib.util.spec_from_file_location("edgekv_pre_sim", str(PRE_SIM_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load pre simulator module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sim = load_pre_sim()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def limit_trace(objects: Sequence[object], requests: Sequence[object], max_requests: int):
    limited_requests = list(requests[:max_requests])
    used_ids = set()
    for req in limited_requests:
        used_ids.add(req.object_id)
        if req.admit_object_id:
            used_ids.add(req.admit_object_id)
    limited_objects = [obj for obj in objects if obj.object_id in used_ids]
    return limited_objects, limited_requests


def load_trace(trace_cfg: dict, sim_cfg) -> Tuple[List[object], List[object], str]:
    source = trace_cfg.get("source", "synthetic")
    max_requests = int(trace_cfg.get("max_requests", sim_cfg.trace_requests))

    if source == "synthetic":
        objects, requests = sim.make_synthetic_trace(
            int(trace_cfg.get("seed", sim_cfg.seed)),
            int(trace_cfg.get("requests", sim_cfg.trace_requests)),
        )
        trace_source = "synthetic"
    elif source == "sharegpt":
        path = Path(trace_cfg.get("path", str(sim.DEFAULT_SHAREGPT_PATH))).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        objects, requests = sim.load_sharegpt_trace(
            path,
            max_sessions=int(trace_cfg.get("max_sessions", 200)),
        )
        trace_source = str(path)
    elif source == "jsonl":
        path = Path(str(trace_cfg["path"])).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        objects, requests = sim.load_jsonl_trace(path)
        trace_source = str(path)
    else:
        raise ValueError(f"unknown trace source: {source}")

    return (*limit_trace(objects, requests, max_requests), trace_source)


def build_sim_config(config: dict, token_ref: int):
    device = dict(config.get("device_profile", {}))
    trace = dict(config.get("trace", {}))
    epsilon_abs = config.get("epsilon_abs")
    epsilon_norm = config.get("epsilon_norm")
    if epsilon_abs is None:
        if epsilon_norm is None:
            epsilon_abs = 4.0
        else:
            epsilon_abs = sim.denormalize_epsilon(float(epsilon_norm), token_ref)

    return sim.SimConfig(
        mu_kv_mb_per_token=float(device.get("mu_kv_mb_per_token", 0.12)),
        c_re_ms_per_token=float(device.get("c_re_ms_per_token", 0.12)),
        d_deser_ms=float(device.get("d_deser_ms", 3.0)),
        seed=int(trace.get("seed", config.get("seed", 1))),
        m_budget_mb=float(device.get("m_budget_mb", 800.0)),
        bw_gbps=float(device.get("bw_gbps", 8.0)),
        epsilon=float(epsilon_abs),
        offload_keep_threshold=float(config.get("offload_keep_threshold", 0.35)),
        trace_requests=int(trace.get("requests", config.get("trace_requests", 500))),
        warmup_requests=int(config.get("warmup_requests", 50)),
    )


def lpe_action_for(event: dict) -> str:
    q_after = str(event.get("q_after", ""))
    if q_after == "drop":
        return "drop"
    if q_after.startswith("offload:"):
        return "offload"
    return "resident"


def tms_action_for(event: dict) -> str:
    q_before = str(event.get("q_before", ""))
    q_after = str(event.get("q_after", ""))
    if q_after in sim.TIER_ORDER and q_before in sim.TIER_ORDER and q_after != q_before:
        return "downgrade"
    return "none"


def enrich_events(
    raw_events: Sequence[dict],
    objects: Sequence[object],
    cfg,
    device_name: str,
) -> List[dict]:
    object_by_id: Dict[str, object] = {obj.object_id: obj for obj in objects}
    rows: List[dict] = []
    memory_peak_so_far = 0.0
    for idx, event in enumerate(raw_events):
        obj = object_by_id[event["object_id"]]
        memory_current = float(event.get("memory_current_mb", 0.0))
        memory_peak_so_far = max(memory_peak_so_far, memory_current)
        row = dict(event)
        row.update(
            {
                "event_index": idx,
                "device_profile": device_name,
                "object_type": obj.object_type,
                "n_tokens": obj.n_tokens,
                "M_budget": round(cfg.m_budget_mb, 6),
                "BW": round(cfg.bw_gbps, 6),
                "mu_kv": round(cfg.mu_kv_mb_per_token, 9),
                "c_re": round(cfg.c_re_ms_per_token, 9),
                "d_deser": round(cfg.d_deser_ms, 6),
                "lpe_action": lpe_action_for(event),
                "tms_action": tms_action_for(event),
                "t_policy_ms": 0.0,
                "M_peak": round(memory_peak_so_far, 6),
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H0 simulator smoke test.")
    parser.add_argument("--config", required=True, help="Path to H0 JSON config.")
    parser.add_argument("--out", required=True, help="Output directory.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    out_dir = Path(args.out).expanduser()
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    config = read_json(config_path)
    seed_cfg = sim.SimConfig(seed=int(config.get("seed", 1)))
    objects, requests, trace_source = load_trace(dict(config.get("trace", {})), seed_cfg)
    token_ref = sim.token_ref_for_objects(objects)
    cfg = build_sim_config(config, token_ref)

    policy = str(config.get("policy", "tiered"))
    rrs_mode = str(config.get("rrs_mode", "rrs"))
    device_name = str(config.get("device_profile", {}).get("name", "unknown"))

    started = time.perf_counter()
    metrics, raw_events = sim.run_one(
        objects,
        requests,
        cfg,
        policy=policy,
        rrs_mode=rrs_mode,
        emit_events=True,
    )
    elapsed_s = round(time.perf_counter() - started, 6)

    events = enrich_events(raw_events, objects, cfg, device_name)
    summary = sim.metrics_row(metrics, "H0")
    summary.update(
        {
            "device_profile": device_name,
            "trace_source": trace_source,
            "objects": len(objects),
            "total_requests": len(requests),
            "elapsed_s": elapsed_s,
        }
    )

    resolved = {
        "experiment": "H0",
        "config_path": str(config_path),
        "trace_source": trace_source,
        "device_profile": config.get("device_profile", {}),
        "policy": policy,
        "rrs_mode": rrs_mode,
        "objects": len(objects),
        "requests": len(requests),
        "token_ref": token_ref,
        "epsilon_abs": cfg.epsilon,
        "epsilon_norm": sim.normalize_qloss(cfg.epsilon, token_ref),
        "sim_config": {
            "mu_kv_mb_per_token": cfg.mu_kv_mb_per_token,
            "c_re_ms_per_token": cfg.c_re_ms_per_token,
            "d_deser_ms": cfg.d_deser_ms,
            "m_budget_mb": cfg.m_budget_mb,
            "bw_gbps": cfg.bw_gbps,
            "warmup_requests": cfg.warmup_requests,
            "offload_keep_threshold": cfg.offload_keep_threshold,
        },
        "outputs": {
            "events_jsonl": "events.jsonl",
            "summary_csv": "summary.csv",
            "config_resolved_json": "config.resolved.json",
        },
    }

    write_jsonl(out_dir / "events.jsonl", events)
    write_csv(out_dir / "summary.csv", [summary])
    write_json(out_dir / "config.resolved.json", resolved)

    print(
        json.dumps(
            {
                "experiment": "H0",
                "device_profile": device_name,
                "policy": policy,
                "rrs_mode": rrs_mode,
                "events": len(events),
                "out_dir": str(out_dir),
                "ttft_p95_ms": round(metrics.ttft_p95_ms, 6),
                "epsilon_ok": metrics.epsilon_ok,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
