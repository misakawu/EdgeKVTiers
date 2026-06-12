#!/usr/bin/env python3
"""Run H3 strategy-layer adapter smoke for LMCache/vLLM integration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "h0"))
sys.path.insert(0, str(REPO_ROOT))

import run_h0
from h3 import lmcache_adapter


sim = run_h0.sim
DEFAULT_OUT_DIR = REPO_ROOT / "h3" / "out" / "adapter_smoke"
DEFAULT_MAX_SESSIONS = 200
DEFAULT_MAX_REQUESTS = 120
DEFAULT_EVENT_LIMIT = 120


def trace_config(args: argparse.Namespace) -> dict:
    cfg = {
        "source": args.trace_source,
        "seed": args.seed,
        "requests": args.max_requests,
        "max_requests": args.max_requests,
        "max_sessions": args.max_sessions,
    }
    if args.trace_path:
        cfg["path"] = args.trace_path
    return cfg


def load_trace(args: argparse.Namespace):
    cfg = sim.SimConfig(seed=args.seed, trace_requests=args.max_requests)
    return run_h0.load_trace(trace_config(args), cfg)


def write_h3_outputs(
    *,
    out_dir: Path,
    objects: Sequence[object],
    requests: Sequence[object],
    trace_source: str,
    event_limit: int,
    m_budget_mb: float,
    bw_gbps: float,
    epsilon_norm: float,
    d_deser_ms: float,
    seed: int,
) -> dict:
    token_ref = sim.token_ref_for_objects(objects)
    cfg = sim.SimConfig(
        seed=seed,
        trace_requests=min(len(requests), event_limit),
        m_budget_mb=m_budget_mb,
        bw_gbps=bw_gbps,
        epsilon=sim.denormalize_epsilon(epsilon_norm, token_ref),
        d_deser_ms=d_deser_ms,
        warmup_requests=0,
    )
    _, raw_events = sim.run_one(
        objects,
        list(requests)[:event_limit],
        cfg,
        policy="tiered",
        rrs_mode="rrs",
        emit_events=True,
    )
    enriched = run_h0.enrich_events(raw_events, objects, cfg, "h3_adapter_smoke")
    lmcache_adapter.write_h3_contract(out_dir, trace_source=trace_source, token_ref=token_ref)
    lmcache_adapter.write_h3_event_sample(out_dir, enriched, limit=event_limit)
    summary = {
        "experiment": "H3",
        "trace_source": trace_source,
        "objects": len(objects),
        "requests": len(requests),
        "event_limit": event_limit,
        "token_ref": token_ref,
        "m_budget_mb": m_budget_mb,
        "bw_gbps": bw_gbps,
        "epsilon_norm": epsilon_norm,
        "epsilon_abs": cfg.epsilon,
        "d_deser_ms": d_deser_ms,
        "outputs": {
            "contract": "h3_adapter_contract.json",
            "events": "h3_hook_events.sample.jsonl",
        },
    }
    run_h0.write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H3 LMCache/vLLM adapter smoke.")
    parser.add_argument("--trace-source", default="sharegpt", choices=["sharegpt", "synthetic", "jsonl"])
    parser.add_argument("--trace-path", default=str(run_h0.DEFAULT_SHAREGPT_TRACE_PATH))
    parser.add_argument("--max-sessions", type=int, default=DEFAULT_MAX_SESSIONS)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument("--event-limit", type=int, default=DEFAULT_EVENT_LIMIT)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--m-budget-mb", type=float, default=520.0)
    parser.add_argument("--bw-gbps", type=float, default=2.0)
    parser.add_argument("--epsilon-norm", type=float, default=0.0002)
    parser.add_argument("--d-deser-ms", type=float, default=6.0)
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser()
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    objects, requests, trace_source = load_trace(args)
    summary = write_h3_outputs(
        out_dir=out_dir,
        objects=objects,
        requests=requests,
        trace_source=trace_source,
        event_limit=args.event_limit,
        m_budget_mb=args.m_budget_mb,
        bw_gbps=args.bw_gbps,
        epsilon_norm=args.epsilon_norm,
        d_deser_ms=args.d_deser_ms,
        seed=args.seed,
    )
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
