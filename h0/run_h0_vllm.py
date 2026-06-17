#!/usr/bin/env python3
"""Replay ShareGPT prompts against a vLLM OpenAI-compatible server.

This is the real-engine H0 runner: it assumes vLLM is already serving with
``--enable-prefix-caching`` and measures request TTFT, end-to-end latency, and
GPU memory peak while replaying multi-turn ShareGPT prompts that share prefixes.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAREGPT_TRACE_PATH = Path(
    "/DATACENTER3/zhenxiang.wang/data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "h0" / "out" / "h0_vllm_prefix_cache"
BYTES_PER_DTYPE = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


def estimate_tokens(text: str) -> int:
    # Keep this runner tokenizer-free. The exact token count is not used for
    # scheduling; it only gives a stable trace-side size estimate.
    return max(1, int(len((text or "").split()) * 1.35))


def load_tokenizer(model_path: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return None


def count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return estimate_tokens(text)
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_model_config(model_path: str) -> dict:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def kv_size_mib_per_token(config: dict, tensor_parallel_size: int = 1) -> float:
    hidden_size = int(config.get("hidden_size", 0) or 0)
    num_attention_heads = int(config.get("num_attention_heads", 0) or 0)
    num_key_value_heads = int(config.get("num_key_value_heads", num_attention_heads) or num_attention_heads or 0)
    num_hidden_layers = int(config.get("num_hidden_layers", 0) or 0)
    dtype = str(config.get("torch_dtype", "float16")).replace("torch.", "")
    bytes_per_value = BYTES_PER_DTYPE.get(dtype, 2)
    if hidden_size <= 0 or num_attention_heads <= 0 or num_key_value_heads <= 0 or num_hidden_layers <= 0:
        return 0.0
    head_dim = hidden_size // num_attention_heads
    bytes_per_token = num_hidden_layers * 2 * num_key_value_heads * head_dim * bytes_per_value
    per_gpu = bytes_per_token / max(int(tensor_parallel_size), 1)
    return per_gpu / (1024.0 * 1024.0)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    pos = (len(data) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    if lo == hi:
        return data[lo]
    weight = pos - lo
    return data[lo] * (1.0 - weight) + data[hi] * weight


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
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def message_role(message: dict) -> str:
    return str(message.get("from", message.get("role", ""))).lower()


def message_text(message: dict) -> str:
    value = message.get("value", message.get("content", ""))
    return value if isinstance(value, str) else str(value)


def load_sharegpt_prompts(path: Path, max_sessions: int, max_requests: int, tokenizer=None) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    prompts: List[dict] = []
    sessions = 0
    for raw_session_idx, row in enumerate(rows):
        conversations = row.get("conversations", [])
        if not isinstance(conversations, list):
            continue
        human_count = sum(1 for msg in conversations if message_role(msg) in {"human", "user"})
        if human_count < 2:
            continue

        session_id = str(row.get("id", f"sharegpt_{raw_session_idx:06d}"))
        transcript: List[str] = []
        human_turn = 0
        for msg in conversations:
            role = message_role(msg)
            text = message_text(msg).strip()
            if not text:
                continue
            if role in {"human", "user"}:
                prompt = "\n".join(transcript + [f"User: {text}", "Assistant:"])
                n_tokens = count_tokens(prompt, tokenizer)
                prompts.append(
                    {
                        "request_id": f"{session_id}:turn:{human_turn:03d}",
                        "session_id": session_id,
                        "turn_index": human_turn,
                        "prompt": prompt,
                        "prompt_chars": len(prompt),
                        "prompt_est_tokens": estimate_tokens(prompt),
                        "n_tokens": n_tokens,
                    }
                )
                transcript.append(f"User: {text}")
                human_turn += 1
                if len(prompts) >= max_requests:
                    return prompts
            elif role in {"gpt", "assistant"}:
                transcript.append(f"Assistant: {text}")
        sessions += 1
        if sessions >= max_sessions:
            break
    return prompts[:max_requests]


def get_default_model(endpoint: str, timeout_s: float) -> str:
    url = endpoint.rstrip("/") + "/v1/models"
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data", [])
    if not data:
        raise RuntimeError("vLLM /v1/models returned no models")
    return str(data[0]["id"])


def stream_completion(endpoint: str, model: str, prompt: str, max_tokens: int, temperature: float, timeout_s: float) -> tuple[float, float, int, str]:
    url = endpoint.rstrip("/") + "/v1/completions"
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_s = None
    chunks = 0
    text_parts: List[str] = []
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            if first_token_s is None:
                first_token_s = time.perf_counter()
            chunks += 1
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = payload.get("choices", [])
            if choices:
                text_parts.append(str(choices[0].get("text", "")))
    ended = time.perf_counter()
    ttft_ms = ((first_token_s or ended) - started) * 1000.0
    latency_ms = (ended - started) * 1000.0
    return ttft_ms, latency_ms, chunks, "".join(text_parts)


class GpuMemoryMonitor:
    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self.samples: List[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                ts = time.time()
                for line in proc.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 4:
                        self.samples.append(
                            {
                                "ts": ts,
                                "gpu_index": int(parts[0]),
                                "memory_used_mib": float(parts[1]),
                                "memory_total_mib": float(parts[2]),
                                "utilization_gpu_pct": float(parts[3]),
                            }
                        )
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def peak_mib(self) -> float:
        return max((row["memory_used_mib"] for row in self.samples), default=0.0)


def completion_error_message(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return f"{repr(exc)} {body}".strip()
    return repr(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ShareGPT prompts against vLLM prefix caching server.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="", help="Served model name. Defaults to /v1/models first id.")
    parser.add_argument("--trace-path", default=str(DEFAULT_SHAREGPT_TRACE_PATH))
    parser.add_argument("--max-sessions", type=int, default=20)
    parser.add_argument("--max-requests", type=int, default=40)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser()
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    trace_path = Path(args.trace_path).expanduser()
    model = args.model or get_default_model(args.endpoint, args.timeout_s)
    tokenizer = load_tokenizer(model)
    model_config = load_model_config(model)
    kv_mib_per_token = kv_size_mib_per_token(model_config, args.tensor_parallel_size)
    prompts = load_sharegpt_prompts(trace_path, args.max_sessions, args.max_requests, tokenizer)
    if not prompts:
        raise RuntimeError("no replay prompts were built from trace")

    monitor = GpuMemoryMonitor()
    monitor.start()
    events: List[dict] = []
    session_seen: set[str] = set()
    started = time.perf_counter()
    try:
        for idx, item in enumerate(prompts):
            policy_started = time.perf_counter()
            n_tokens = int(item["n_tokens"])
            overlength = n_tokens + args.max_tokens > args.max_model_len
            trace_hit = item["session_id"] in session_seen and item["turn_index"] > 0
            event_name = "hit" if trace_hit else "miss"
            size_mb = n_tokens * kv_mib_per_token
            t_policy_ms = (time.perf_counter() - policy_started) * 1000.0
            row = dict(item)
            row.pop("prompt", None)
            row.update(
                {
                    "event_index": idx,
                    "event": event_name,
                    "hit": trace_hit,
                    "hit_source": "trace_side_session_prefix_reuse",
                    "size_mb": round(size_mb, 6),
                    "size_scope": "per_gpu_logical_full_kv_estimate",
                    "kv_mib_per_token": round(kv_mib_per_token, 9),
                    "t_policy_ms": round(t_policy_ms, 6),
                    "model": model,
                    "max_tokens": args.max_tokens,
                    "max_model_len": args.max_model_len,
                    "temperature": args.temperature,
                }
            )
            if overlength:
                row.update(
                    {
                        "ok": False,
                        "skipped": True,
                        "event": "skip_overlength",
                        "error": f"n_tokens + max_tokens exceeds max_model_len ({n_tokens} + {args.max_tokens} > {args.max_model_len})",
                        "ttft_ms": 0.0,
                        "latency_ms": 0.0,
                    }
                )
                events.append(row)
                session_seen.add(item["session_id"])
                continue
            try:
                ttft_ms, latency_ms, chunks, text = stream_completion(
                    args.endpoint,
                    model,
                    item["prompt"],
                    args.max_tokens,
                    args.temperature,
                    args.timeout_s,
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
                row.update({"ok": False, "error": completion_error_message(exc), "ttft_ms": 0.0, "latency_ms": 0.0})
            events.append(row)
            session_seen.add(item["session_id"])
    finally:
        monitor.stop()
    elapsed_s = time.perf_counter() - started

    attempted = [row for row in events if not row.get("skipped")]
    measured = [row for row in attempted[args.warmup_requests :] if row.get("ok")]
    ttfts = [float(row["ttft_ms"]) for row in measured]
    latencies = [float(row["latency_ms"]) for row in measured]
    hit_rows = [row for row in measured if row.get("hit")]
    summary = {
        "experiment": "H0_vLLM_prefix_cache",
        "endpoint": args.endpoint,
        "model": model,
        "trace_source": str(trace_path),
        "sessions_requested": args.max_sessions,
        "requests_total": len(events),
        "requests_attempted": len(attempted),
        "requests_skipped_overlength": sum(1 for row in events if row.get("skipped")),
        "requests_measured": len(measured),
        "success_rate": round(sum(1 for row in attempted if row.get("ok")) / max(len(attempted), 1), 6),
        "warmup_requests": args.warmup_requests,
        "hit_rate": round(len(hit_rows) / max(len(measured), 1), 6),
        "hit_source": "trace_side_session_prefix_reuse",
        "ttft_p50_ms": round(percentile(ttfts, 50), 6),
        "ttft_p95_ms": round(percentile(ttfts, 95), 6),
        "ttft_mean_ms": round(statistics.mean(ttfts), 6) if ttfts else 0.0,
        "latency_p50_ms": round(percentile(latencies, 50), 6),
        "latency_p95_ms": round(percentile(latencies, 95), 6),
        "gpu_memory_peak_mib": round(monitor.peak_mib(), 6),
        "elapsed_s": round(elapsed_s, 6),
        "prefix_caching": "enabled_by_server_configuration",
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "kv_mib_per_token": round(kv_mib_per_token, 9),
        "tokenizer_source": "transformers_auto" if tokenizer is not None else "whitespace_estimate",
    }

    write_jsonl(out_dir / "events.jsonl", events)
    write_csv(out_dir / "summary.csv", [summary])
    write_json(out_dir / "config.resolved.json", {"args": vars(args), "summary": summary})
    write_jsonl(out_dir / "gpu_memory_samples.jsonl", monitor.samples)
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
