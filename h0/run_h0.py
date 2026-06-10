#!/usr/bin/env python3
"""Run the H0 trace replay and metrics-closure smoke experiment.

H0 is intentionally a configurable wrapper around the analytical simulator in
``pre实验/sim.py``. It produces the stable artifacts later H1/H2/H4/H5
experiments consume: event JSONL, summary CSV, resolved configuration snapshots,
and a small validation report. The runner supports both the legacy single
``device_profile`` config and the H0-required multi-device ``devices`` config.
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
DEFAULT_SHAREGPT_TRACE_PATH = Path(
    "/DATACENTER3/zhenxiang.wang/data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
)


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
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        path = Path(trace_cfg.get("path", str(DEFAULT_SHAREGPT_TRACE_PATH))).expanduser()
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


def device_profiles(config: dict) -> List[dict]:
    if "devices" in config:
        devices = config["devices"]
        if not isinstance(devices, list) or not devices:
            raise ValueError("config.devices must be a non-empty list")
        return [dict(device) for device in devices]
    return [dict(config.get("device_profile", {}))]


def build_sim_config(config: dict, device: dict, token_ref: int):
    trace = dict(config.get("trace", {}))
    epsilon_abs = device.get("epsilon_abs", config.get("epsilon_abs"))
    epsilon_norm = device.get("epsilon_norm", config.get("epsilon_norm"))
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


def cache_event_for(event: dict) -> str:
    if not bool(event.get("hit", False)):
        return "miss"
    action = str(event.get("rrs_action", "none"))
    if action == "none":
        return "resident_hit"
    if action == "restore":
        return "offload_restore"
    if action == "recompute":
        return "offload_recompute"
    return "hit"


def enrich_events(
    raw_events: Sequence[dict],
    objects: Sequence[object],
    cfg,
    device_name: str,
) -> List[dict]:
    object_by_id: Dict[str, object] = {obj.object_id: obj for obj in objects}
    rows: List[dict] = []
    memory_peak_so_far = 0.0
    token_ref = sim.token_ref_for_objects(objects)
    for idx, event in enumerate(raw_events):
        obj = object_by_id[event["object_id"]]
        memory_current = float(event.get("memory_current_mb", 0.0))
        memory_peak_so_far = max(memory_peak_so_far, memory_current)
        row = dict(event)
        row.update(
            {
                "event_index": idx,
                "device_profile": device_name,
                "event": cache_event_for(event),
                "object_type": obj.object_type,
                "n_tokens": obj.n_tokens,
                "M_budget": round(cfg.m_budget_mb, 6),
                "BW": round(cfg.bw_gbps, 6),
                "epsilon_abs": round(cfg.epsilon, 6),
                "epsilon_norm": round(sim.normalize_qloss(cfg.epsilon, token_ref), 9),
                "mu_kv": round(cfg.mu_kv_mb_per_token, 9),
                "c_re": round(cfg.c_re_ms_per_token, 9),
                "d_deser": round(cfg.d_deser_ms, 6),
                "lpe_action": lpe_action_for(event),
                "tms_action": tms_action_for(event),
                "qloss_total_abs": row.get("qloss_current_abs", row.get("qloss_current", 0.0)),
                "qloss_total_norm": row.get("qloss_current_norm", 0.0),
                "t_policy_ms": 0.0,
                "M_peak": round(memory_peak_so_far, 6),
            }
        )
        rows.append(row)
    return rows


def resolved_config(
    *,
    config_path: Path,
    trace_source: str,
    device: dict,
    policy: str,
    rrs_mode: str,
    objects: Sequence[object],
    requests: Sequence[object],
    cfg,
) -> dict:
    token_ref = sim.token_ref_for_objects(objects)
    return {
        "experiment": "H0",
        "config_path": str(config_path),
        "trace_source": trace_source,
        "device_profile": device,
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


def run_device(
    *,
    config: dict,
    config_path: Path,
    out_dir: Path,
    objects: Sequence[object],
    requests: Sequence[object],
    trace_source: str,
    device: dict,
) -> Tuple[dict, dict, List[dict]]:
    cfg = build_sim_config(config, device, sim.token_ref_for_objects(objects))
    policy = str(device.get("policy", config.get("policy", "tiered")))
    rrs_mode = str(device.get("rrs_mode", config.get("rrs_mode", "rrs")))
    device_name = str(device.get("name", "unknown"))

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
            "qloss_total_abs": summary.get("qloss_peak_abs", summary.get("qloss_current_abs", 0.0)),
            "qloss_total_norm": summary.get("qloss_peak_norm", summary.get("qloss_current_norm", 0.0)),
            "device_profile": device_name,
            "trace_source": trace_source,
            "objects": len(objects),
            "total_requests": len(requests),
            "elapsed_s": elapsed_s,
        }
    )

    resolved = resolved_config(
        config_path=config_path,
        trace_source=trace_source,
        device=device,
        policy=policy,
        rrs_mode=rrs_mode,
        objects=objects,
        requests=requests,
        cfg=cfg,
    )

    device_out_dir = out_dir / device_name
    write_jsonl(device_out_dir / "events.jsonl", events)
    write_csv(device_out_dir / "summary.csv", [summary])
    write_json(device_out_dir / "config.resolved.json", resolved)
    return summary, resolved, events


def validation_report(summaries: Sequence[dict], token_ref: int) -> dict:
    complete = bool(summaries) and all(int(row.get("total_requests", 0)) > 0 for row in summaries)
    epsilon_ok = bool(summaries) and all(bool(row.get("epsilon_ok", False)) for row in summaries)
    same_trace = len({row.get("trace_source") for row in summaries}) <= 1
    same_token_ref = len({row.get("token_ref") for row in summaries}) <= 1
    memory_controlled = all(
        float(row.get("memory_peak_mb", 0.0)) <= float(row.get("m_budget_mb", 0.0)) + 1e-6
        for row in summaries
    )
    required = ("policy", "rrs_mode", "epsilon_abs", "epsilon_norm", "token_ref")
    has_required = all(all(key in row for key in required) for row in summaries)
    return {
        "experiment": "H0",
        "passed": all(
            [
                complete,
                epsilon_ok,
                same_trace,
                same_token_ref,
                memory_controlled,
                has_required,
            ]
        ),
        "checks": {
            "complete_replay": complete,
            "same_trace_across_devices": same_trace,
            "same_token_ref_across_devices": same_token_ref,
            "memory_peak_within_budget": memory_controlled,
            "epsilon_budget_respected": epsilon_ok,
            "required_summary_fields_present": has_required,
        },
        "token_ref": token_ref,
        "devices": [row.get("device_profile") for row in summaries],
    }


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

    summaries: List[dict] = []
    resolved_runs: List[dict] = []
    all_events: List[dict] = []
    for device in device_profiles(config):
        summary, resolved, events = run_device(
            config=config,
            config_path=config_path,
            out_dir=out_dir,
            objects=objects,
            requests=requests,
            trace_source=trace_source,
            device=device,
        )
        summaries.append(summary)
        resolved_runs.append(resolved)
        all_events.extend(events)

    validation = validation_report(summaries, token_ref)
    resolved_all = {
        "experiment": "H0",
        "config_path": str(config_path),
        "trace_source": trace_source,
        "objects": len(objects),
        "requests": len(requests),
        "token_ref": token_ref,
        "runs": resolved_runs,
        "validation": validation,
        "outputs": {
            "events_jsonl": "events.jsonl",
            "summary_csv": "summary.csv",
            "config_resolved_json": "config.resolved.json",
            "validation_json": "validation.json",
        },
    }

    write_jsonl(out_dir / "events.jsonl", all_events)
    write_csv(out_dir / "summary.csv", summaries)
    write_json(out_dir / "config.resolved.json", resolved_all)
    write_json(out_dir / "validation.json", validation)

    print(
        json.dumps(
            {
                "experiment": "H0",
                "devices": [row["device_profile"] for row in summaries],
                "events": len(all_events),
                "out_dir": str(out_dir),
                "ttft_p95_ms": {
                    row["device_profile"]: row["ttft_p95_ms"] for row in summaries
                },
                "epsilon_ok": {
                    row["device_profile"]: row["epsilon_ok"] for row in summaries
                },
                "passed": validation["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
