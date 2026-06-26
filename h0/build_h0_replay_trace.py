#!/usr/bin/env python3
"""Build the frozen H0 replay JSONL required by pre-experiment 0.5.0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_h0_vllm import (
    DEFAULT_HOTPOTQA_PATH,
    DEFAULT_REPLAY_TRACE_PATH,
    DEFAULT_SHAREGPT_TRACE_PATH,
    build_replay_sessions,
    write_replay_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen ShareGPT+HotpotQA H0 replay trace.")
    parser.add_argument("--trace-path", default=str(DEFAULT_SHAREGPT_TRACE_PATH))
    parser.add_argument("--hotpotqa-path", type=Path, default=DEFAULT_HOTPOTQA_PATH)
    parser.add_argument("--download-hotpotqa", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPLAY_TRACE_PATH)
    parser.add_argument("--workload", choices=("sharegpt", "rag", "mixed"), default="mixed")
    parser.add_argument("--max-sessions", type=int, default=2000)
    parser.add_argument("--max-requests", type=int, default=10000)
    parser.add_argument("--rag-requests", type=int, default=500)
    parser.add_argument("--hotpotqa-max-examples", type=int, default=50)
    parser.add_argument("--rag-chunk-words", type=int, default=56)
    parser.add_argument("--rag-chunks-per-query", type=int, default=2)
    parser.add_argument("--rag-query-repeats", type=int, default=4)
    parser.add_argument("--sharegpt-order", choices=("file", "longest"), default="file")
    parser.add_argument("--link-mode", choices=("independent", "weak"), default="independent")
    parser.add_argument("--weak-rag-repeat", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out.expanduser()
    sessions = build_replay_sessions(args)
    write_replay_trace(out, sessions)
    sharegpt_sessions = sum(1 for row in sessions if row.get("source") == "sharegpt")
    rag_sessions = sum(1 for row in sessions if row.get("source") == "hotpotqa")
    turns = sum(len(row.get("turns", [])) for row in sessions)
    summary = {
        "out": str(out),
        "sessions": len(sessions),
        "sharegpt_sessions": sharegpt_sessions,
        "rag_sessions": rag_sessions,
        "turns": turns,
        "args": vars(args),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
