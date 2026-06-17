#!/usr/bin/env python3
"""Run H1 custom eviction policies on top of H0 vLLM replay.

The runner has two modes:
- shadow: works in the current vLLM 0.6.6 environment and logs policy decisions
  beside normal OpenAI replay, but does not claim real KV offload.
- real-v1: requires vLLM V1 KV connector support and fails fast otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "h0"))

import run_h0_vllm as h0
from edgekv_v1_offload.env_check import check_v1_offload_env, write_env_status
from edgekv_v1_offload.policy import H1Policy, JsonlDecisionLogger, decisions_to_rows


DEFAULT_CONFIG = REPO_ROOT / "h0" / "configs" / "vllm_qwen25_7b_h1_offload.json"
DEFAULT_OUT_DIR = REPO_ROOT / "h0" / "out" / "h1_vllm_offload_qwen25_7b"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[float], pct: float) -> float:
    return h0.percentile(values, pct)


def resolve_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H1 vLLM V1 KV offload policy experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--trace-path", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--policy", choices=sorted(H1Policy.SUPPORTED), default=None)
    parser.add_argument("--gpu-budget-mb", type=float, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--warmup-requests", type=int, default=None)
    parser.add_argument("--c-re-ms-per-token", type=float, default=None)
    parser.add_argument("--theta-keep", type=float, default=None)
    parser.add_argument("--reserve-mb", type=float, default=None)
    parser.add_argument("--mode", choices=["shadow", "real-v1", "env-check"], default=None)
    return parser.parse_args()


def config_value(args: argparse.Namespace, cfg: dict, key: str, default):
    cli_value = getattr(args, key.replace("-", "_"), None)
    if cli_value is not None:
        return cli_value
    return cfg.get(key, default)


def build_object_id(item: dict) -> str:
    return str(item.get("session_id", item.get("request_id", "unknown")))


def latest_gpu_memory(samples: list[dict]) -> float:
    if not samples:
        return 0.0
    return float(samples[-1].get("memory_used_mib", 0.0))


def run_once(args: argparse.Namespace, cfg: dict) -> dict:
    mode = config_value(args, cfg, "mode", "shadow")
    out_dir = Path(config_value(args, cfg, "out", str(DEFAULT_OUT_DIR))).expanduser()
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    policy_name = config_value(args, cfg, "policy", "lpe-score")
    gpu_budget_mb = float(config_value(args, cfg, "gpu-budget-mb", cfg.get("gpu_budget_mb", 2048.0)))

    env_status = check_v1_offload_env()
    write_env_status(out_dir / "v1_env_status.json", env_status)
    if mode == "env-check":
        return {"mode": mode, "out_dir": str(out_dir), "env_ok": env_status.ok, "message": env_status.message}
    if mode == "real-v1" and not env_status.ok:
        raise RuntimeError(env_status.message)

    endpoint = config_value(args, cfg, "endpoint", "http://127.0.0.1:8000")
    trace_path = Path(config_value(args, cfg, "trace-path", cfg.get("trace_path", str(h0.DEFAULT_SHAREGPT_TRACE_PATH)))).expanduser()
    max_sessions = int(config_value(args, cfg, "max-sessions", cfg.get("max_sessions", 20)))
    max_requests = int(config_value(args, cfg, "max-requests", cfg.get("max_requests", 40)))
    max_model_len = int(config_value(args, cfg, "max-model-len", cfg.get("max_model_len", 1024)))
    max_tokens = int(config_value(args, cfg, "max-tokens", cfg.get("max_tokens", 16)))
    tensor_parallel_size = int(config_value(args, cfg, "tensor-parallel-size", cfg.get("tensor_parallel_size", 1)))
    temperature = float(config_value(args, cfg, "temperature", cfg.get("temperature", 0.0)))
    timeout_s = float(config_value(args, cfg, "timeout-s", cfg.get("timeout_s", 120.0)))
    warmup_requests = int(config_value(args, cfg, "warmup-requests", cfg.get("warmup_requests", 2)))
    c_re_ms_per_token = float(config_value(args, cfg, "c-re-ms-per-token", cfg.get("c_re_ms_per_token", 0.08)))
    theta_keep = float(config_value(args, cfg, "theta-keep", cfg.get("theta_keep", 0.5)))
    reserve_mb = float(config_value(args, cfg, "reserve-mb", cfg.get("reserve_mb", 0.0)))

    model = args.model or cfg.get("model") or h0.get_default_model(endpoint, timeout_s)
    tokenizer = h0.load_tokenizer(model)
    model_config = h0.load_model_config(model)
    kv_mib_per_token = h0.kv_size_mib_per_token(model_config, tensor_parallel_size)
    prompts = h0.load_sharegpt_prompts(trace_path, max_sessions, max_requests, tokenizer)
    if not prompts:
        raise RuntimeError("no replay prompts were built from trace")

    decision_logger = JsonlDecisionLogger(out_dir / "policy_decisions.jsonl")
    policy = H1Policy(
        policy=policy_name,
        gpu_budget_mb=gpu_budget_mb,
        c_re_ms_per_token=c_re_ms_per_token,
        theta_keep=theta_keep,
        reserve_mb=reserve_mb,
        logger=decision_logger,
    )
    monitor = h0.GpuMemoryMonitor()
    monitor.start()
    events: List[dict] = []
    all_decisions: list[dict] = []
    session_seen: set[str] = set()
    started = time.perf_counter()
    try:
        for idx, item in enumerate(prompts):
            n_tokens = int(item["n_tokens"])
            overlength = n_tokens + max_tokens > max_model_len
            trace_hit = item["session_id"] in session_seen and item["turn_index"] > 0
            size_mb = n_tokens * kv_mib_per_token
            row = dict(item)
            prompt = row.pop("prompt")
            row.update(
                {
                    "event_index": idx,
                    "experiment": "H1_vLLM_V1_KV_offload",
                    "mode": mode,
                    "policy": policy_name,
                    "event": "hit" if trace_hit else "miss",
                    "hit": trace_hit,
                    "hit_source": "trace_side_session_prefix_reuse" if mode == "shadow" else "vllm_v1_connector_or_trace",
                    "object_id": build_object_id(item),
                    "object_type": "session_prefix",
                    "size_mb": round(size_mb, 6),
                    "size_scope": "per_gpu_logical_full_kv_estimate",
                    "kv_mib_per_token": round(kv_mib_per_token, 9),
                    "model": model,
                    "max_tokens": max_tokens,
                    "max_model_len": max_model_len,
                    "temperature": temperature,
                    "gpu_budget_mb": gpu_budget_mb,
                    "c_re_ms_per_token": c_re_ms_per_token,
                    "theta_keep": theta_keep,
                    "reserve_mb": reserve_mb,
                }
            )
            if overlength:
                row.update(
                    {
                        "ok": False,
                        "skipped": True,
                        "event": "skip_overlength",
                        "error": f"n_tokens + max_tokens exceeds max_model_len ({n_tokens} + {max_tokens} > {max_model_len})",
                        "ttft_ms": 0.0,
                        "latency_ms": 0.0,
                    }
                )
                decisions = policy.observe_request(
                    request_id=str(item["request_id"]),
                    session_id=str(item["session_id"]),
                    object_id=build_object_id(item),
                    object_type="session_prefix",
                    n_tokens=n_tokens,
                    size_mb=size_mb,
                    hit=trace_hit,
                    ttft_ms=0.0,
                    gpu_memory_mb=latest_gpu_memory(monitor.samples),
                )
                decision_rows = decisions_to_rows(decisions)
                row.update(_decision_summary(decision_rows))
                all_decisions.extend(decision_rows)
                events.append(row)
                session_seen.add(item["session_id"])
                continue
            try:
                ttft_ms, latency_ms, chunks, text = h0.stream_completion(
                    endpoint,
                    model,
                    prompt,
                    max_tokens,
                    temperature,
                    timeout_s,
                )
                row.update(
                    {
                        "ok": True,
                        "ttft_ms": round(ttft_ms, 6),
                        "latency_ms": round(latency_ms, 6),
                        "stream_chunks": chunks,
                        "output_chars": len(text),
                    }
                )
            except Exception as exc:
                row.update({"ok": False, "error": h0.completion_error_message(exc), "ttft_ms": 0.0, "latency_ms": 0.0})

            decisions = policy.observe_request(
                request_id=str(item["request_id"]),
                session_id=str(item["session_id"]),
                object_id=build_object_id(item),
                object_type="session_prefix",
                n_tokens=n_tokens,
                size_mb=size_mb,
                hit=trace_hit,
                ttft_ms=float(row.get("ttft_ms", 0.0)),
                gpu_memory_mb=latest_gpu_memory(monitor.samples),
            )
            decision_rows = decisions_to_rows(decisions)
            row.update(_decision_summary(decision_rows))
            all_decisions.extend(decision_rows)
            events.append(row)
            session_seen.add(item["session_id"])
    finally:
        monitor.stop()

    elapsed_s = time.perf_counter() - started
    summary = build_summary(
        events,
        monitor.samples,
        policy.snapshot(),
        mode=mode,
        policy_name=policy_name,
        endpoint=endpoint,
        model=model,
        trace_path=trace_path,
        max_sessions=max_sessions,
        warmup_requests=warmup_requests,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        kv_mib_per_token=kv_mib_per_token,
        elapsed_s=elapsed_s,
        env_ok=env_status.ok,
    )
    h0.write_jsonl(out_dir / "events.jsonl", events)
    write_csv(out_dir / "summary.csv", [summary])
    h0.write_json(
        out_dir / "config.resolved.json",
        {
            "args": vars(args),
            "config": cfg,
            "summary": summary,
            "v1_env_status": env_status.__dict__,
            "policy_snapshot": policy.snapshot(),
        },
    )
    h0.write_jsonl(out_dir / "gpu_memory_samples.jsonl", monitor.samples)
    write_readme(out_dir, summary, mode, env_status.message)
    return {"out_dir": str(out_dir), "summary": summary, "decisions": len(all_decisions)}


def _decision_summary(decisions: list[dict]) -> dict:
    if not decisions:
        return {"p_reuse": 0.0, "score": 0.0, "lpe_action": "none", "policy_actions": ""}
    main = decisions[0]
    actions = [row["action"] for row in decisions]
    lpe_action = "none"
    if "offload" in actions:
        lpe_action = "offload"
    elif "drop" in actions:
        lpe_action = "drop"
    return {
        "p_reuse": main.get("p_reuse", 0.0),
        "score": main.get("score", 0.0),
        "lpe_action": lpe_action,
        "policy_actions": ",".join(actions),
        "resident_mb_after": decisions[-1].get("resident_mb_after", 0.0),
    }


def build_summary(
    events: list[dict],
    gpu_samples: list[dict],
    policy_snapshot: dict,
    *,
    mode: str,
    policy_name: str,
    endpoint: str,
    model: str,
    trace_path: Path,
    max_sessions: int,
    warmup_requests: int,
    max_model_len: int,
    max_tokens: int,
    tensor_parallel_size: int,
    kv_mib_per_token: float,
    elapsed_s: float,
    env_ok: bool,
) -> dict:
    attempted = [row for row in events if not row.get("skipped")]
    measured = [row for row in attempted[warmup_requests:] if row.get("ok")]
    ttfts = [float(row["ttft_ms"]) for row in measured]
    latencies = [float(row["latency_ms"]) for row in measured]
    hit_rows = [row for row in measured if row.get("hit")]
    return {
        "experiment": "H1_vLLM_V1_KV_offload",
        "mode": mode,
        "real_v1_env_ok": env_ok,
        "policy": policy_name,
        "endpoint": endpoint,
        "model": model,
        "trace_source": str(trace_path),
        "sessions_requested": max_sessions,
        "requests_total": len(events),
        "requests_attempted": len(attempted),
        "requests_skipped_overlength": sum(1 for row in events if row.get("skipped")),
        "requests_measured": len(measured),
        "success_rate": round(sum(1 for row in attempted if row.get("ok")) / max(len(attempted), 1), 6),
        "warmup_requests": warmup_requests,
        "hit_rate": round(len(hit_rows) / max(len(measured), 1), 6),
        "ttft_p50_ms": round(percentile(ttfts, 50), 6),
        "ttft_p95_ms": round(percentile(ttfts, 95), 6),
        "ttft_mean_ms": round(statistics.mean(ttfts), 6) if ttfts else 0.0,
        "latency_p50_ms": round(percentile(latencies, 50), 6),
        "latency_p95_ms": round(percentile(latencies, 95), 6),
        "gpu_memory_peak_mib": round(max((row["memory_used_mib"] for row in gpu_samples), default=0.0), 6),
        "gpu_budget_mb": policy_snapshot["gpu_budget_mb"],
        "policy_resident_mb": policy_snapshot["resident_mb"],
        "policy_resident_objects": policy_snapshot["resident_objects"],
        "elapsed_s": round(elapsed_s, 6),
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "tensor_parallel_size": tensor_parallel_size,
        "kv_mib_per_token": round(kv_mib_per_token, 9),
    }


def write_readme(out_dir: Path, summary: dict, mode: str, env_message: str) -> None:
    text = f"""# H1 vLLM V1 KV Offload

- mode: `{mode}`
- policy: `{summary['policy']}`
- real_v1_env_ok: `{summary['real_v1_env_ok']}`
- p95 TTFT: `{summary['ttft_p95_ms']}` ms
- hit rate: `{summary['hit_rate']}`
- GPU peak: `{summary['gpu_memory_peak_mib']}` MiB

Environment note: {env_message}

Files:

- `events.jsonl`: request-level replay events with H1 policy fields.
- `policy_decisions.jsonl`: custom eviction/offload decisions.
- `summary.csv`: aggregate metrics.
- `config.resolved.json`: resolved configuration and environment gate.
- `gpu_memory_samples.jsonl`: `nvidia-smi` samples.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = resolve_args()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = load_json(cfg_path)
    result = run_once(args, cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
