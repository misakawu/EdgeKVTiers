#!/usr/bin/env python3
"""Build a pressure trace with stronger real prefix-cache reuse.

The original pressure trace reused session ids, but each ShareGPT turn often
changed the prompt near the beginning. This builder converts long ShareGPT
messages into stable-context sessions: every turn in a group starts with the
same long context and only changes the final task line. That pattern gives
vLLM's prefix cache identical leading token blocks while still using real
ShareGPT text as the source material.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
H0_DIR = REPO_ROOT / "h0"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(H0_DIR) not in sys.path:
    sys.path.insert(0, str(H0_DIR))

from run_h0_vllm import (  # noqa: E402
    DEFAULT_HOTPOTQA_PATH,
    DEFAULT_SHAREGPT_TRACE_PATH,
    build_rag_sessions,
    estimate_tokens,
    load_sharegpt_sessions,
    write_replay_trace,
)


DEFAULT_OUT = (
    REPO_ROOT
    / "data"
    / "edgekv_traces"
    / "h0_sharegpt_hotpotqa_200sessions_pressure.jsonl"
)

# 扩充任务池
TASKS = (
    "Summarize the context in 3 concise bullets.",
    "Extract the main entities and their roles.",
    "Write a short title for this context.",
    "List the likely user intent behind the context.",
    "Give one practical next step based on the context.",
    "Rewrite the key point in plain language.",
    "Identify the potential risks mentioned.",
    "Suggest three alternative approaches.",
    "Explain the cause-and-effect chain.",
    "Propose a solution to the main challenge.",
    "Compare this situation to a similar case.",
    "Formulate a question to clarify the ambiguity.",
    "Outline the assumptions made.",
    "Evaluate the credibility of the information.",
    "Translate the key insight into a metaphor.",
    "Predict the future trend implied.",
    "Highlight the most critical sentence.",
    "Simplify the jargon into everyday terms.",
    "Create a checklist for action.",
    "Summarize in one sentence for a busy executive.",
)


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def bounded_context(text: str, max_words: int) -> str:
    words = normalize_text(text).split()
    return " ".join(words[:max_words])


def build_reuse_focused_sharegpt_sessions(
    sharegpt_path: Path,
    groups: int,
    turns_per_group: int,
    context_words: int,
    reuse_window_size: int,
) -> list[dict[str, Any]]:
    source_sessions = load_sharegpt_sessions(
        sharegpt_path,
        max_sessions=max(groups * 4, groups),
        order="longest",
    )
    grouped_sessions: list[dict[str, Any]] = []
    built_groups = 0
    for group_index, source in enumerate(source_sessions):
        if built_groups >= groups:
            break
        turns = source.get("turns", [])
        if not turns:
            continue
        seed_text = str(turns[0].get("user", "")).strip()
        context = bounded_context(seed_text, context_words)
        if len(context.split()) < 128:
            continue

        reuse_key = f"sharegpt_hot_prefix_{group_index:04d}"
        prompt_prefix = (
            "Use the following ShareGPT-derived context as the fixed working memory.\n"
            "Context:\n"
            f"{context}\n\n"
            "Task:"
        )
        grouped_sessions.append(
            {
                "session_id": reuse_key,
                "source": "sharegpt",
                "object_type": "sharegpt_stable_context_prefix",
                "reuse_key": reuse_key,
                "turns_format": "cumulative_user",
                "source_index": source.get("source_index", group_index),
                "source_session_id": source.get("session_id", ""),
                "trace_shape": "stable_sharegpt_context_with_task_variants",
                "reuse_group_index": group_index,
                "reuse_group_size": turns_per_group,
                "context_words": len(context.split()),
                "prompt_est_tokens": estimate_tokens(prompt_prefix),
                "turns": [
                    {
                        "i": turn_index,
                        "user": f"{prompt_prefix} {TASKS[turn_index % len(TASKS)]}",
                    }
                    for turn_index in range(turns_per_group)
                ],
            }
        )
        built_groups += 1
    if built_groups < groups:
        raise RuntimeError(
            f"only built {built_groups} ShareGPT reuse groups; requested {groups}"
        )
    sessions: list[dict[str, Any]] = []
    window_size = max(1, reuse_window_size)
    for window_start in range(0, len(grouped_sessions), window_size):
        sessions.extend(grouped_sessions[window_start : window_start + window_size])
    return sessions


def interleave_sessions(
    sharegpt_sessions: list[dict[str, Any]],
    rag_sessions: list[dict[str, Any]],
    rag_every: int,
) -> list[dict[str, Any]]:
    if not rag_sessions:
        return sharegpt_sessions
    mixed: list[dict[str, Any]] = []
    rag_pos = 0
    for idx, session in enumerate(sharegpt_sessions):
        mixed.append(session)
        if rag_every > 0 and (idx + 1) % rag_every == 0 and rag_pos < len(rag_sessions):
            mixed.append(rag_sessions[rag_pos])
            rag_pos += 1
    mixed.extend(rag_sessions[rag_pos:])
    return mixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the H0/H1 pressure replay trace for higher KV-cache reuse."
    )
    parser.add_argument("--sharegpt-path", type=Path, default=DEFAULT_SHAREGPT_TRACE_PATH)
    parser.add_argument("--hotpotqa-path", type=Path, default=DEFAULT_HOTPOTQA_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sharegpt-groups", type=int, default=80)
    # parser.add_argument("--turns-per-group", type=int, default=8)
    parser.add_argument("--turns-per-group", type=int, default=12)
    parser.add_argument("--context-words", type=int, default=700)
    parser.add_argument(
        "--reuse-window-size",
        type=int,
        default=16,
        help="Number of hot ShareGPT reuse keys to cycle before repeating; match H1 replay_batch_size.",
    )
    parser.add_argument("--rag-requests", type=int, default=120)
    parser.add_argument("--rag-every", type=int, default=4)
    parser.add_argument("--hotpotqa-max-examples", type=int, default=56)
    parser.add_argument("--rag-chunk-words", type=int, default=56)
    parser.add_argument("--rag-chunks-per-query", type=int, default=2)
    parser.add_argument("--rag-query-repeats", type=int, default=4)
    parser.add_argument("--download-hotpotqa", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sharegpt_sessions = build_reuse_focused_sharegpt_sessions(
        args.sharegpt_path.expanduser(),
        args.sharegpt_groups,
        args.turns_per_group,
        args.context_words,
        args.reuse_window_size,
    )
    rag_sessions = build_rag_sessions(
        max_requests=args.rag_requests,
        hotpotqa_path=args.hotpotqa_path.expanduser(),
        download_hotpotqa=args.download_hotpotqa,
        chunk_words_count=args.rag_chunk_words,
        chunks_per_query=args.rag_chunks_per_query,
        query_repeats=args.rag_query_repeats,
        max_examples=args.hotpotqa_max_examples,
        timeout_s=args.timeout_s,
    )
    sessions = interleave_sessions(sharegpt_sessions, rag_sessions, args.rag_every)
    write_replay_trace(args.out.expanduser(), sessions)
    summary = {
        "out": str(args.out.expanduser()),
        "sessions": len(sessions),
        "sharegpt_sessions": len(sharegpt_sessions),
        "rag_sessions": len(rag_sessions),
        "turns": sum(len(row.get("turns", [])) for row in sessions),
        "sharegpt_turns": sum(len(row.get("turns", [])) for row in sharegpt_sessions),
        "rag_turns": sum(len(row.get("turns", [])) for row in rag_sessions),
        "reuse_window_size": args.reuse_window_size,
        "expected_sharegpt_trace_side_hit_rate_after_first_window": round(
            (args.turns_per_group - 1) / max(args.turns_per_group, 1), 6
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
