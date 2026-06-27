#!/usr/bin/env python3
"""Build a ShareGPT + HotpotQA pressure trace with explicit hot/cold objects.

Hot objects are intentionally small and accessed repeatedly. Cold objects are
larger and accessed once. The generated JSONL remains compatible with the H0/H1
frozen replay loader while avoiding multi-turn sessions that hide interleaving
pressure inside one row.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
H0_DIR = REPO_ROOT / "h0"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(H0_DIR) not in sys.path:
    sys.path.insert(0, str(H0_DIR))

from run_h0_vllm import (  # noqa: E402
    DEFAULT_HOTPOTQA_PATH,
    DEFAULT_SHAREGPT_TRACE_PATH,
    estimate_tokens,
    load_hotpotqa_chunk_groups,
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


HOT_P_REUSE_PRIOR = 0.95
COLD_P_REUSE_PRIOR = 0.05


def reuse_prior_for_temperature(value: str) -> float:
    return HOT_P_REUSE_PRIOR if str(value).lower() == "hot" else COLD_P_REUSE_PRIOR


def with_temperature_prior(row: dict[str, Any], hot_or_cold: str) -> dict[str, Any]:
    row["temperature"] = hot_or_cold
    row["p_reuse_prior"] = reuse_prior_for_temperature(hot_or_cold)
    return row


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def bounded_context(text: str, max_words: int) -> str:
    words = normalize_text(text).split()
    return " ".join(words[:max_words])


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def bounded_ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def average(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def evenly_insert_hot_copies(
    cold_list: Sequence[dict[str, Any]], hot_copies: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not hot_copies:
        return list(cold_list)
    if not cold_list:
        return list(hot_copies)
    mixed = list(cold_list)
    step = max(1, len(cold_list) // len(hot_copies))
    offset = 0
    for hot in hot_copies:
        insert_at = min(len(mixed), offset)
        mixed.insert(insert_at, hot)
        offset += step + 1
    return mixed


def choose_hot_count(total: int, ratio: float) -> int:
    if total <= 0 or ratio <= 0.0:
        return 0
    return min(total, max(1, int(round(total * ratio))))


def build_sharegpt_prompt(context: str, task_index: int) -> str:
    return (
        "Use the following ShareGPT-derived context as the fixed working memory.\n"
        "Context:\n"
        f"{context}\n\n"
        f"Task: {TASKS[task_index % len(TASKS)]}\n"
        "Assistant:"
    )


def build_hot_cold_sharegpt_sessions(
    sharegpt_path: Path,
    groups: int,
    hot_ratio: float,
    hot_repeats: int,
    hot_context_words: int,
    cold_context_words: int,
    min_context_words: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    source_sessions = load_sharegpt_sessions(
        sharegpt_path,
        max_sessions=max(groups * 4, groups),
        order="longest",
    )
    rng = random.Random(random_seed)
    rng.shuffle(source_sessions)

    candidates: list[dict[str, Any]] = []
    max_context_words = max(hot_context_words, cold_context_words, min_context_words)
    for source in source_sessions:
        if len(candidates) >= groups:
            break
        turns = source.get("turns", [])
        if not turns:
            continue
        seed_text = str(turns[0].get("user", "")).strip()
        max_context = bounded_context(seed_text, max_context_words)
        if len(max_context.split()) < min_context_words:
            continue
        candidates.append({**source, "seed_text": seed_text})
    if len(candidates) < groups:
        raise RuntimeError(
            f"only built {len(candidates)} ShareGPT objects with at least "
            f"{min_context_words} words; requested {groups}"
        )

    hot_count = choose_hot_count(groups, hot_ratio)
    hot_objects = candidates[:hot_count]
    cold_objects = candidates[hot_count:groups]

    hot_copies: list[dict[str, Any]] = []
    cold_list: list[dict[str, Any]] = []

    for object_index, source in enumerate(hot_objects):
        context = bounded_context(str(source["seed_text"]), hot_context_words)
        reuse_key = f"sharegpt:hot:w{len(context.split())}:obj:{object_index:04d}"
        prompt = build_sharegpt_prompt(context, object_index)
        for repeat_index in range(hot_repeats):
            hot_copies.append(
                with_temperature_prior(
                    {
                        "session_id": reuse_key,
                        "source": "sharegpt",
                        "object_type": "sharegpt_hot_context",
                        "reuse_key": reuse_key,
                        "turns_format": "cumulative_user",
                        "source_index": source.get("source_index", object_index),
                        "source_session_id": source.get("session_id", ""),
                        "reuse_group_index": object_index,
                        "reuse_group_size": hot_repeats,
                        "repeat_index": repeat_index,
                        "context_words": len(context.split()),
                        "prompt_est_tokens": estimate_tokens(prompt),
                        "turns": [{"i": 0, "user": prompt}],
                    },
                    "hot",
                )
            )

    for cold_index, source in enumerate(cold_objects):
        object_index = hot_count + cold_index
        context = bounded_context(str(source["seed_text"]), cold_context_words)
        reuse_key = f"sharegpt:cold:w{len(context.split())}:obj:{object_index:04d}"
        prompt = build_sharegpt_prompt(context, object_index)
        cold_list.append(
            with_temperature_prior(
                {
                    "session_id": f"{reuse_key}:access:000",
                    "source": "sharegpt",
                    "object_type": "sharegpt_cold_context",
                    "reuse_key": reuse_key,
                    "turns_format": "cumulative_user",
                    "source_index": source.get("source_index", object_index),
                    "source_session_id": source.get("session_id", ""),
                    "reuse_group_index": object_index,
                    "reuse_group_size": 1,
                    "repeat_index": 0,
                    "context_words": len(context.split()),
                    "prompt_est_tokens": estimate_tokens(prompt),
                    "turns": [{"i": 0, "user": prompt}],
                },
                "cold",
            )
        )

    return evenly_insert_hot_copies(cold_list, hot_copies)


def target_distinct_count(requests: int, hot_ratio: float, hot_repeats: int) -> int:
    denominator = hot_ratio * hot_repeats + (1.0 - hot_ratio)
    if denominator <= 0.0:
        return requests
    return max(1, int(math.ceil(requests / denominator)))


def exact_hot_cold_counts(requests: int, hot_ratio: float, hot_repeats: int) -> tuple[int, int]:
    if requests <= 0:
        return 0, 0
    if hot_ratio <= 0.0:
        return 0, requests
    if hot_repeats <= 1:
        hot_count = choose_hot_count(requests, hot_ratio)
        return hot_count, requests - hot_count

    # Derived from h / (h + c) ~= hot_ratio and requests = h * repeats + c.
    hot_count = int(round((hot_ratio * requests) / (1.0 + hot_ratio * (hot_repeats - 1))))
    hot_count = max(1, min(hot_count, requests // hot_repeats))
    cold_count = requests - hot_count * hot_repeats
    return hot_count, cold_count


def build_rag_query_session(
    group_row: dict[str, Any],
    group_index: int,
    repeat_index: int,
    hot_or_cold: str,
    chunk_words_count: int,
    chunks_per_query: int,
) -> dict[str, Any]:
    chunks = group_row["chunks"]
    chunk_ids = "|".join(row["chunk_id"] for row in chunks)
    chunk_words_total = sum(len(str(row.get("text", "")).split()) for row in chunks)
    reuse_key = (
        f"rag:hotpotqa:{hot_or_cold}:chunk_words:{chunk_words_count}:"
        f"chunks:{chunks_per_query}:{chunk_ids}"
    )
    query = str(group_row["question"])
    if repeat_index > 0:
        query = f"{query} Answer in a different concise wording. Variant {repeat_index}."
    return with_temperature_prior(
        {
            "session_id": f"rag_{hot_or_cold}_{group_index:06d}_access_{repeat_index:03d}",
            "source": "hotpotqa",
            "object_type": f"rag_{hot_or_cold}_chunk_set",
            "reuse_key": reuse_key,
            "dataset": "hotpotqa",
            "hotpotqa_example_id": group_row["example_id"],
            "hotpotqa_source_path": group_row["source_path"],
            "answer": group_row["answer"],
            "chunks": chunks,
            "reuse_group_index": group_index,
            "reuse_group_size": 1 if hot_or_cold == "cold" else None,
            "repeat_index": repeat_index,
            "chunk_words": chunk_words_total,
            "chunk_words_count": chunk_words_count,
            "chunks_per_query": chunks_per_query,
            "turns": [{"i": 0, "user": query}],
        },
        hot_or_cold,
    )


def build_hot_cold_rag_sessions(
    max_requests: int,
    hotpotqa_path: Path,
    download_hotpotqa: bool,
    hot_ratio: float,
    hot_repeats: int,
    hot_chunk_words: int,
    cold_chunk_words: int,
    hot_chunks_per_query: int,
    cold_chunks_per_query: int,
    max_examples: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    if max_requests <= 0:
        raise ValueError("--rag-requests must be > 0 for the pressure trace")

    distinct_count = target_distinct_count(max_requests, hot_ratio, hot_repeats)
    hot_count, cold_count = exact_hot_cold_counts(max_requests, hot_ratio, hot_repeats)

    hot_groups = load_hotpotqa_chunk_groups(
        hotpotqa_path,
        download_hotpotqa,
        max_examples,
        hot_chunk_words,
        hot_chunks_per_query,
        timeout_s,
    )
    cold_groups = load_hotpotqa_chunk_groups(
        hotpotqa_path,
        download_hotpotqa,
        max_examples,
        cold_chunk_words,
        cold_chunks_per_query,
        timeout_s,
    )
    if hot_count > len(hot_groups):
        raise RuntimeError(f"only built {len(hot_groups)} hot RAG objects; requested {hot_count}")
    if cold_count > len(cold_groups):
        raise RuntimeError(f"only built {len(cold_groups)} cold RAG objects; requested {cold_count}")

    hot_copies: list[dict[str, Any]] = []
    cold_list: list[dict[str, Any]] = []
    for hot_index, group_row in enumerate(hot_groups[:hot_count]):
        for repeat_index in range(hot_repeats):
            session = build_rag_query_session(
                group_row,
                hot_index,
                repeat_index,
                "hot",
                hot_chunk_words,
                hot_chunks_per_query,
            )
            session["reuse_group_size"] = hot_repeats
            hot_copies.append(session)

    for cold_index, group_row in enumerate(cold_groups[:cold_count]):
        session = build_rag_query_session(
            group_row,
            hot_count + cold_index,
            0,
            "cold",
            cold_chunk_words,
            cold_chunks_per_query,
        )
        cold_list.append(session)

    sessions = evenly_insert_hot_copies(cold_list, hot_copies)
    if len(sessions) != max_requests:
        raise RuntimeError(
            f"built {len(sessions)} RAG accesses for target {max_requests}; "
            f"formula distinct target was {distinct_count}"
        )
    return sessions


def temperature_of(row: dict[str, Any]) -> str:
    return str(row.get("temperature", "")).lower()


def scan_resistant_prefix(
    sessions: Sequence[dict[str, Any]],
    hot_objects: int,
    cold_scan_objects: int,
    probe_rounds: int,
) -> tuple[list[dict[str, Any]], set[int]]:
    if hot_objects <= 0 or cold_scan_objects <= 0 or probe_rounds <= 0:
        return [], set()

    hot_by_key: dict[str, list[dict[str, Any]]] = {}
    cold_once: list[dict[str, Any]] = []
    cold_seen: set[str] = set()
    for row in sessions:
        temp = temperature_of(row)
        reuse_key = str(row.get("reuse_key", row.get("session_id", "")))
        if temp == "hot":
            hot_by_key.setdefault(reuse_key, []).append(row)
        elif temp == "cold" and reuse_key not in cold_seen:
            cold_seen.add(reuse_key)
            cold_once.append(row)

    hot_keys = [key for key, rows in hot_by_key.items() if len(rows) >= probe_rounds + 1]
    hot_keys = hot_keys[:hot_objects]
    required_cold = cold_scan_objects * probe_rounds
    if len(hot_keys) < hot_objects or len(cold_once) < required_cold:
        return [], set()

    prefix: list[dict[str, Any]] = []
    used: set[int] = set()

    def add(row: dict[str, Any]) -> None:
        prefix.append(row)
        used.add(id(row))

    for key in hot_keys:
        add(hot_by_key[key][0])

    cold_pos = 0
    for probe_index in range(probe_rounds):
        for row in cold_once[cold_pos:cold_pos + cold_scan_objects]:
            add(row)
        cold_pos += cold_scan_objects
        for key in hot_keys:
            add(hot_by_key[key][probe_index + 1])

    return prefix, used


def order_scan_resistant_sessions(
    sessions: list[dict[str, Any]],
    hot_objects: int,
    cold_scan_objects: int,
    probe_rounds: int,
) -> list[dict[str, Any]]:
    prefix, used = scan_resistant_prefix(
        sessions,
        hot_objects=hot_objects,
        cold_scan_objects=cold_scan_objects,
        probe_rounds=probe_rounds,
    )
    if not prefix:
        return sessions
    return prefix + [row for row in sessions if id(row) not in used]


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
    parser.add_argument("--sharegpt-groups", type=positive_int, default=80)
    parser.add_argument("--hot-ratio", type=bounded_ratio, default=0.25)
    parser.add_argument("--hot-repeats", type=positive_int, default=8)
    parser.add_argument("--hot-context-words", type=positive_int, default=300)
    parser.add_argument("--cold-context-words", type=positive_int, default=700)
    parser.add_argument("--min-context-words", type=positive_int, default=128)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--turns-per-group", type=positive_int, default=None)
    parser.add_argument("--context-words", type=positive_int, default=None)
    parser.add_argument(
        "--reuse-window-size",
        type=int,
        default=16,
        help="Compatibility no-op retained for old commands.",
    )
    parser.add_argument("--rag-requests", type=positive_int, default=120)
    parser.add_argument("--rag-every", type=nonnegative_int, default=4)
    parser.add_argument("--scan-hot-objects", type=nonnegative_int, default=16)
    parser.add_argument("--scan-cold-objects", type=nonnegative_int, default=40)
    parser.add_argument("--scan-probe-rounds", type=nonnegative_int, default=2)
    parser.add_argument("--hotpotqa-max-examples", type=positive_int, default=56)
    parser.add_argument("--rag-hot-ratio", type=bounded_ratio, default=0.25)
    parser.add_argument("--rag-hot-repeats", type=positive_int, default=8)
    parser.add_argument("--rag-hot-chunk-words", type=positive_int, default=100)
    parser.add_argument("--rag-cold-chunk-words", type=positive_int, default=300)
    parser.add_argument("--rag-hot-chunks-per-query", type=positive_int, default=1)
    parser.add_argument("--rag-cold-chunks-per-query", type=positive_int, default=3)
    parser.add_argument("--rag-chunk-words", type=positive_int, default=None)
    parser.add_argument("--rag-chunks-per-query", type=positive_int, default=None)
    parser.add_argument("--rag-query-repeats", type=positive_int, default=None)
    parser.add_argument("--download-hotpotqa", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    if args.context_words is not None:
        args.cold_context_words = args.context_words
    if args.turns_per_group is not None:
        args.hot_repeats = args.turns_per_group
    if args.rag_chunk_words is not None:
        args.rag_cold_chunk_words = args.rag_chunk_words
    if args.rag_chunks_per_query is not None:
        args.rag_cold_chunks_per_query = args.rag_chunks_per_query
    if args.rag_query_repeats is not None:
        args.rag_hot_repeats = args.rag_query_repeats
    return args


def main() -> None:
    args = parse_args()
    sharegpt_sessions = build_hot_cold_sharegpt_sessions(
        args.sharegpt_path.expanduser(),
        args.sharegpt_groups,
        args.hot_ratio,
        args.hot_repeats,
        args.hot_context_words,
        args.cold_context_words,
        args.min_context_words,
        args.random_seed,
    )
    rag_sessions = build_hot_cold_rag_sessions(
        max_requests=args.rag_requests,
        hotpotqa_path=args.hotpotqa_path.expanduser(),
        download_hotpotqa=args.download_hotpotqa,
        hot_ratio=args.rag_hot_ratio,
        hot_repeats=args.rag_hot_repeats,
        hot_chunk_words=args.rag_hot_chunk_words,
        cold_chunk_words=args.rag_cold_chunk_words,
        hot_chunks_per_query=args.rag_hot_chunks_per_query,
        cold_chunks_per_query=args.rag_cold_chunks_per_query,
        max_examples=args.hotpotqa_max_examples,
        timeout_s=args.timeout_s,
    )
    sessions = interleave_sessions(sharegpt_sessions, rag_sessions, args.rag_every)
    sessions = order_scan_resistant_sessions(
        sessions,
        hot_objects=args.scan_hot_objects,
        cold_scan_objects=args.scan_cold_objects,
        probe_rounds=args.scan_probe_rounds,
    )
    write_replay_trace(args.out.expanduser(), sessions)
    sg_hot = [row for row in sharegpt_sessions if row.get("temperature") == "hot"]
    sg_cold = [row for row in sharegpt_sessions if row.get("temperature") == "cold"]
    rag_hot = [row for row in rag_sessions if row.get("temperature") == "hot"]
    rag_cold = [row for row in rag_sessions if row.get("temperature") == "cold"]
    summary = {
        "out": str(args.out.expanduser()),
        "sessions": len(sessions),
        "turns": sum(len(row.get("turns", [])) for row in sessions),
        "sharegpt": {
            "hot_objects": len({row["reuse_key"] for row in sg_hot}),
            "cold_objects": len({row["reuse_key"] for row in sg_cold}),
            "hot_accesses": len(sg_hot),
            "cold_accesses": len(sg_cold),
            "avg_hot_context_words": round(average([int(row["context_words"]) for row in sg_hot]), 2),
            "avg_cold_context_words": round(average([int(row["context_words"]) for row in sg_cold]), 2),
        },
        "rag": {
            "hot_objects": len({row["reuse_key"] for row in rag_hot}),
            "cold_objects": len({row["reuse_key"] for row in rag_cold}),
            "hot_accesses": len(rag_hot),
            "cold_accesses": len(rag_cold),
            "avg_hot_chunk_words": round(average([int(row["chunk_words"]) for row in rag_hot]), 2),
            "avg_cold_chunk_words": round(average([int(row["chunk_words"]) for row in rag_cold]), 2),
        },
        "scan_resistant_prefix": {
            "hot_objects": args.scan_hot_objects,
            "cold_scan_objects": args.scan_cold_objects,
            "probe_rounds": args.scan_probe_rounds,
            "requests": args.scan_hot_objects + args.scan_probe_rounds * (args.scan_cold_objects + args.scan_hot_objects),
        },
        "compat": {
            "reuse_window_size": args.reuse_window_size,
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
