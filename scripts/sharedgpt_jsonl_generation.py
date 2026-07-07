#!/usr/bin/env python3
"""Generate a prefix-cache-friendly ShareGPT replay trace.

Layout per request (strictly position-stable so vLLM block-level prefix cache
can reuse):

    HOT  (constant for every request, sets the floor)
    WARM (exactly one chunk, chosen by Zipf popularity, drives the climb)
    TAIL (short per-request unique string, caps the ceiling)

Only "which warm chunk" is random; the warm chunk always sits immediately
after the identical HOT region, so two requests that pick the same warm chunk
share the entire `HOT + WARM` token prefix and reuse its blocks. The unique
tail is the only never-reused region, so it sets the ceiling gap.
"""

from __future__ import annotations

import argparse
import json
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
    DEFAULT_SHAREGPT_TRACE_PATH,
    build_sharegpt_cumulative_sessions,
    estimate_tokens,
    load_sharegpt_sessions,
    write_jsonl,
)


DEFAULT_OUT = (
    REPO_ROOT
    / "data"
    / "edgekv_traces"
    / "source_ablation"
    / "sharedgpt.jsonl"
)
DEFAULT_REQUESTS = 1536
DEFAULT_RANDOM_SEED = 2026
DEFAULT_HOT_POOL_SIZE = 1
DEFAULT_WARM_POOL_SIZE = 300
DEFAULT_ZIPF_S = 0.8
DEFAULT_TAIL_WORDS = 8
DEFAULT_CHUNK_WORDS = 240
DEFAULT_MIN_CHUNK_WORDS = 240
DEFAULT_MAX_SOURCE_SESSIONS = 4096
DEFAULT_MIN_HUMAN_TURNS = 2

# Single constant task, kept in the SHARED prefix (before the unique tail) so it
# never breaks prefix-cache matching. Per-request task rotation would inflate the
# always-miss region and depress the ceiling.
CONSTANT_TASK = "Summarize the recurring context and answer using the shared prefix."

# Deterministic filler vocabulary for padding the per-request unique tail to a
# fixed word budget. Sourced words are still ShareGPT-derived (session id), the
# filler only pads to a stable token length.
TAIL_FILLER = (
    "context", "detail", "note", "item", "step",
    "case", "point", "topic", "aspect", "factor",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def average(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[int], pct: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    pos = (len(data) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    if lo == hi:
        return float(data[lo])
    weight = pos - lo
    return data[lo] * (1.0 - weight) + data[hi] * weight


def coverage_at_counts(counts: Sequence[int], top_n: int) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return sum(sorted(counts, reverse=True)[:top_n]) / total


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def chunk_words(text: str, chunk_word_count: int, min_chunk_words: int) -> list[str]:
    words = normalize_text(text).split()
    chunks: list[str] = []
    for start in range(0, len(words), chunk_word_count):
        chunk = words[start:start + chunk_word_count]
        if len(chunk) < min_chunk_words:
            continue
        chunks.append(" ".join(chunk))
    return chunks


def session_transcript(session: dict[str, Any]) -> str:
    lines: list[str] = []
    for turn in session.get("turns", []):
        user = normalize_text(str(turn.get("user", "")))
        assistant = normalize_text(str(turn.get("assistant", "")))
        if user:
            lines.append(f"User: {user}")
        if assistant:
            lines.append(f"Assistant: {assistant}")
    return "\n".join(lines)


def load_sharegpt_prefix_chunks(
    sharegpt_path: Path,
    max_source_sessions: int,
    chunk_word_count: int,
    min_chunk_words: int,
) -> list[dict[str, Any]]:
    sessions = load_sharegpt_sessions(
        sharegpt_path,
        max_sessions=max_source_sessions,
        order="longest",
    )
    prefixes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for session in sessions:
        session_id = str(session.get("session_id", ""))
        if not session_id or session_id in seen_ids:
            continue
        transcript = session_transcript(session)
        word_count = len(normalize_text(transcript).split())
        if word_count < min_chunk_words:
            continue
        seen_ids.add(session_id)
        source_index = int(session.get("source_index", len(prefixes)))
        for chunk_index, context in enumerate(
            chunk_words(transcript, chunk_word_count, min_chunk_words)
        ):
            words_in_chunk = len(context.split())
            prefixes.append(
                {
                    "prefix_id": f"sharegpt_{source_index:06d}:chunk:{chunk_index:03d}",
                    "source_session_id": session_id,
                    "source_index": source_index,
                    "chunk_index": chunk_index,
                    "turn_count": len(session.get("turns", [])),
                    "total_user_chars": int(session.get("total_user_chars", 0)),
                    "source_words": word_count,
                    "chunk_words": words_in_chunk,
                    "context_words": words_in_chunk,
                    "context": context,
                }
            )
    return prefixes


def format_prefix_context(tier: str, row: dict[str, Any]) -> str:
    return (
        f"[{tier} {row['prefix_id']} session={row['source_session_id']} "
        f"turns={row['turn_count']}] {row['context']}"
    )


def build_unique_tail(session_id: str, request_index: int, tail_words: int) -> str:
    """Per-request unique short string (sets the ceiling gap).

    Uniqueness is guaranteed by embedding the request index; the remaining
    budget is padded with deterministic filler to a stable token length.
    """
    head = ["query", session_id, f"r{request_index:06d}"]
    words = list(head)
    cursor = 0
    while len(words) < tail_words:
        words.append(TAIL_FILLER[(request_index + cursor) % len(TAIL_FILLER)])
        cursor += 1
    return " ".join(words)


def build_sharegpt_prompt(
    hot_prefixes: Sequence[dict[str, Any]],
    warm_prefixes: Sequence[dict[str, Any]],
    task: str,
    tail_text: str,
) -> str:
    sections: list[str] = ["Hot Prefix:"]
    sections.extend(format_prefix_context("HOT", row) for row in hot_prefixes)
    sections.append("")
    sections.append("Warm Prefix:")
    sections.extend(format_prefix_context("WARM", row) for row in warm_prefixes)
    sections.append("")
    sections.append(f"Task: {task}")
    sections.append(f"Q: {tail_text}")
    sections.append("Assistant:")
    return "\n".join(sections)


def zipf_weights(pool_size: int, s: float) -> list[float]:
    return [1.0 / ((rank + 1) ** s) for rank in range(pool_size)]


def build_sharegpt_hierarchical_sessions(
    request_count: int = DEFAULT_REQUESTS,
    sharegpt_path: Path = DEFAULT_SHAREGPT_TRACE_PATH,
    random_seed: int = DEFAULT_RANDOM_SEED,
    hot_pool_size: int = DEFAULT_HOT_POOL_SIZE,
    warm_pool_size: int = DEFAULT_WARM_POOL_SIZE,
    zipf_s: float = DEFAULT_ZIPF_S,
    tail_words: int = DEFAULT_TAIL_WORDS,
    chunk_word_count: int = DEFAULT_CHUNK_WORDS,
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    max_source_sessions: int = DEFAULT_MAX_SOURCE_SESSIONS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request_count <= 0:
        raise ValueError("request_count must be > 0")
    pool_size = hot_pool_size + warm_pool_size
    prefixes = load_sharegpt_prefix_chunks(
        sharegpt_path,
        max(max_source_sessions, pool_size),
        chunk_word_count,
        min_chunk_words,
    )
    if len(prefixes) < pool_size:
        raise RuntimeError(
            f"only built {len(prefixes)} usable ShareGPT prefix chunks; need {pool_size} "
            f"({hot_pool_size} hot + {warm_pool_size} warm)"
        )

    hot_pool = prefixes[:hot_pool_size]
    warm_pool = prefixes[hot_pool_size:hot_pool_size + warm_pool_size]

    rng = random.Random(random_seed)
    weights = zipf_weights(warm_pool_size, zipf_s)
    sessions: list[dict[str, Any]] = []

    for request_index in range(request_count):
        # HOT: entire hot pool, identical order, every request -> constant floor.
        selected_hot = list(hot_pool)

        # WARM: exactly one chunk chosen by Zipf popularity, placed right after
        # the identical hot region so its blocks reuse on repeat picks.
        warm_index = rng.choices(range(warm_pool_size), weights=weights, k=1)[0]
        selected_warm = [warm_pool[warm_index]]

        task = CONSTANT_TASK
        session_id = f"sharegpt_hier_{request_index:06d}"
        tail_text = build_unique_tail(session_id, request_index, tail_words)
        prompt = build_sharegpt_prompt(selected_hot, selected_warm, task, tail_text)

        selected_prefixes = selected_hot + selected_warm
        hot_ids = [str(row["prefix_id"]) for row in selected_hot]
        warm_ids = [str(row["prefix_id"]) for row in selected_warm]
        prefix_ids = hot_ids + warm_ids
        reuse_key = "sharegpt:hot+warm:" + "|".join(hot_ids) + "|" + "|".join(warm_ids)
        session = {
            "session_id": session_id,
            "source": "sharegpt",
            "workload": "conversation_prefix_reuse",
            "object_type": "sharegpt_hierarchical_prefix",
            "reuse_key": reuse_key,
            "turns_format": "complete_prompt",
            "dataset": "sharegpt",
            "task": task,
            "prefix_ids": prefix_ids,
            "source_session_ids": [str(row["source_session_id"]) for row in selected_prefixes],
            "prefixes": [
                {
                    "prefix_id": str(row["prefix_id"]),
                    "source_session_id": str(row["source_session_id"]),
                    "source_index": int(row["source_index"]),
                    "chunk_index": int(row["chunk_index"]),
                    "turn_count": int(row["turn_count"]),
                    "source_words": int(row["source_words"]),
                    "chunk_words": int(row["chunk_words"]),
                    "context_words": int(row["context_words"]),
                    "context": str(row["context"]),
                }
                for row in selected_prefixes
            ],
            "prefix_layout": "hot_warm_tail",
            "global_prefix_ids": hot_ids,
            "warm_prefix_ids": warm_ids,
            "cold_prefix_ids": [],
            "prefix_tiers": {
                "global": hot_ids,
                "warm": warm_ids,
                "cold": [],
            },
            "warm_zipf_rank": warm_index,
            "tail_text": tail_text,
            "prompt_est_tokens": estimate_tokens(prompt),
            "prompt_is_complete": True,
            "turns": [{"i": 0, "user": prompt, "prompt_is_complete": True}],
        }
        sessions.append(session)

    meta = {
        "hot_pool_ids": [str(row["prefix_id"]) for row in hot_pool],
        "warm_pool_ids": [str(row["prefix_id"]) for row in warm_pool],
        "random_seed": random_seed,
        "hot_pool_size": hot_pool_size,
        "warm_pool_size": warm_pool_size,
        "zipf_s": zipf_s,
        "tail_words": tail_words,
        "chunk_words": chunk_word_count,
        "min_chunk_words": min_chunk_words,
        "max_source_sessions": max_source_sessions,
    }
    return sessions, meta


def build_sharegpt_hierarchical_summary(
    out_path: Path,
    sessions: Sequence[dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    request_count = len(sessions)
    warm_requests = sum(1 for row in sessions if row.get("warm_prefix_ids"))
    prompt_tokens = [int(row.get("prompt_est_tokens", 0)) for row in sessions]
    prefix_counts = [len(row.get("prefix_ids", [])) for row in sessions]
    selected_prefixes = [
        prefix
        for row in sessions
        for prefix in row.get("prefixes", [])
    ]
    context_words = [int(prefix.get("context_words", 0)) for prefix in selected_prefixes]
    source_words = [int(prefix.get("source_words", 0)) for prefix in selected_prefixes]
    warm_ranks = [int(row.get("warm_zipf_rank", 0)) for row in sessions]
    warm_rank_histogram: dict[str, int] = {}
    for rank in warm_ranks:
        key = str(rank)
        warm_rank_histogram[key] = warm_rank_histogram.get(key, 0) + 1
    warm_counts_by_id: dict[str, int] = {}
    warm_reuse_distances: list[int] = []
    last_seen_warm: dict[str, int] = {}
    for index, row in enumerate(sessions):
        for warm_id in row.get("warm_prefix_ids", []):
            warm_key = str(warm_id)
            warm_counts_by_id[warm_key] = warm_counts_by_id.get(warm_key, 0) + 1
            previous_index = last_seen_warm.get(warm_key)
            if previous_index is not None:
                warm_reuse_distances.append(index - previous_index)
            last_seen_warm[warm_key] = index
    warm_counts = list(warm_counts_by_id.values())
    estimated_hot_warm_words = 0
    if sessions:
        first_prefixes = sessions[0].get("prefixes", [])
        estimated_hot_warm_words = sum(
            int(prefix.get("context_words", 0)) for prefix in first_prefixes[:2]
        )
    return {
        "out": str(out_path),
        "source": "sharegpt",
        "workload": "conversation_prefix_reuse",
        "prefix_layout": "hot_warm_tail",
        "selection": "longest_dialogues_chunked_first",
        "requests": request_count,
        "total_requests": request_count,
        "tier_pool_counts": {
            "hot": len(meta.get("hot_pool_ids", [])),
            "warm": len(meta.get("warm_pool_ids", [])),
        },
        "tier_request_counts": {
            "hot": request_count,
            "warm": warm_requests,
        },
        "unique_selected_tier_prefixes": {
            "hot": len({cid for row in sessions for cid in row.get("global_prefix_ids", [])}),
            "warm": len({cid for row in sessions for cid in row.get("warm_prefix_ids", [])}),
        },
        "warm_unique_used": len({cid for row in sessions for cid in row.get("warm_prefix_ids", [])}),
        "warm_top32_coverage": round(coverage_at_counts(warm_counts, 32), 6),
        "warm_top48_coverage": round(coverage_at_counts(warm_counts, 48), 6),
        "warm_top64_coverage": round(coverage_at_counts(warm_counts, 64), 6),
        "warm_reuse_distance_p50": round(percentile(warm_reuse_distances, 50), 2),
        "warm_reuse_distance_p90": round(percentile(warm_reuse_distances, 90), 2),
        "warm_reuse_distance_p95": round(percentile(warm_reuse_distances, 95), 2),
        "estimated_hot_warm_words": estimated_hot_warm_words,
        "warm_zipf_rank_top10": {
            str(rank): warm_rank_histogram.get(str(rank), 0) for rank in range(10)
        },
        "unique_reuse_keys": len({str(row.get("reuse_key", "")) for row in sessions}),
        "prefix_count_distribution": {
            str(count): prefix_counts.count(count)
            for count in sorted(set(prefix_counts))
        },
        "avg_prompt_est_tokens": round(average(prompt_tokens), 2),
        "min_prompt_est_tokens": min(prompt_tokens) if prompt_tokens else 0,
        "p50_prompt_est_tokens": round(percentile(prompt_tokens, 50), 2),
        "p95_prompt_est_tokens": round(percentile(prompt_tokens, 95), 2),
        "max_prompt_est_tokens": max(prompt_tokens) if prompt_tokens else 0,
        "estimated_valid_for_max_model_len_1024": {
            "count": sum(1 for value in prompt_tokens if value <= 1024),
            "ratio": round(
                sum(1 for value in prompt_tokens if value <= 1024) / request_count,
                6,
            ) if request_count else 0.0,
        },
        "avg_selected_context_words": round(average(context_words), 2),
        "avg_selected_source_words": round(average(source_words), 2),
        "random_seed": meta.get("random_seed"),
        "hot_pool_size": meta.get("hot_pool_size"),
        "warm_pool_size": meta.get("warm_pool_size"),
        "zipf_s": meta.get("zipf_s"),
        "tail_words": meta.get("tail_words"),
        "chunk_words": meta.get("chunk_words"),
        "min_chunk_words": meta.get("min_chunk_words"),
        "context_words": meta.get("chunk_words"),
        "min_context_words": meta.get("min_chunk_words"),
        "max_source_sessions": meta.get("max_source_sessions"),
        "hot_pool_ids": meta.get("hot_pool_ids", []),
        "warm_pool_ids": meta.get("warm_pool_ids", []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a cumulative-context ShareGPT replay trace."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--requests", type=positive_int, default=DEFAULT_REQUESTS)
    parser.add_argument(
        "--num-prompts",
        type=positive_int,
        default=None,
        help="Compatibility alias for --requests.",
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--sharegpt-path", type=Path, default=DEFAULT_SHAREGPT_TRACE_PATH)
    parser.add_argument("--hot-pool-size", type=positive_int, default=DEFAULT_HOT_POOL_SIZE)
    parser.add_argument("--warm-pool-size", type=positive_int, default=DEFAULT_WARM_POOL_SIZE)
    parser.add_argument("--zipf-s", type=positive_float, default=DEFAULT_ZIPF_S)
    parser.add_argument("--tail-words", type=positive_int, default=DEFAULT_TAIL_WORDS)
    parser.add_argument("--chunk-words", type=positive_int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument(
        "--context-words",
        type=positive_int,
        default=None,
        help="Compatibility alias for --chunk-words.",
    )
    parser.add_argument("--min-chunk-words", type=positive_int, default=DEFAULT_MIN_CHUNK_WORDS)
    parser.add_argument(
        "--min-context-words",
        type=positive_int,
        default=None,
        help="Compatibility alias for --min-chunk-words.",
    )
    parser.add_argument("--max-source-sessions", type=positive_int, default=DEFAULT_MAX_SOURCE_SESSIONS)
    parser.add_argument("--min-human-turns", type=positive_int, default=DEFAULT_MIN_HUMAN_TURNS)
    parser.add_argument("--sharegpt-order", choices=("file", "longest"), default="file")
    parser.add_argument(
        "--max-prompt-est-tokens",
        type=int,
        default=0,
        help="Drop the rest of a session once a cumulative prompt exceeds this estimate; 0 disables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out.expanduser()
    sessions, summary = build_sharegpt_cumulative_sessions(
        args.sharegpt_path.expanduser(),
        max_turns=args.num_prompts or args.requests,
        min_human_turns=args.min_human_turns,
        order=args.sharegpt_order,
        max_prompt_est_tokens=args.max_prompt_est_tokens,
    )
    write_jsonl(out_path, sessions)
    summary.update(
        {
            "out": str(out_path),
            "out_path": str(out_path),
            "workload": "conversation_prefix_reuse",
            "prefix_layout": "original_conversation_cumulative",
            "trace_format": "session_cumulative_user",
            "total_requests": summary.get("written_requests", 0),
            "sharegpt_cumulative_sessions": len(sessions),
            "sharegpt_cumulative_turns": summary.get("written_requests", 0),
            "ignored_for_cumulative_sharegpt": {
                "random_seed": args.random_seed,
                "hot_pool_size": args.hot_pool_size,
                "warm_pool_size": args.warm_pool_size,
                "zipf_s": args.zipf_s,
                "tail_words": args.tail_words,
                "chunk_words": args.context_words or args.chunk_words,
                "min_chunk_words": args.min_context_words or args.min_chunk_words,
                "max_source_sessions": args.max_source_sessions,
            },
        }
    )
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
