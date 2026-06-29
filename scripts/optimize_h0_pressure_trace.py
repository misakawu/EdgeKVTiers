#!/usr/bin/env python3
"""Build a budget-sensitive ShareGPT + HotpotQA pressure trace.

The trace is shaped for real vLLM prefix-cache behavior: every prompt starts
with an object-level marker, hot objects repeat the same marker, and cold
objects use unique markers. The request order primes hot objects, scans cold
objects, then probes the hot objects again so retention improves as KV capacity
increases.
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
    DEFAULT_HOTPOTQA_PATH,
    DEFAULT_SHAREGPT_TRACE_PATH,
    estimate_tokens,
    hotpotqa_high_frequency_group_order,
    load_hotpotqa_chunk_groups,
    load_sharegpt_sessions,
    write_jsonl,
    write_replay_trace,
)
from hotqa_jsonl_generation import (  # noqa: E402
    build_hotqa_hierarchical_sessions,
    build_hotqa_hierarchical_summary,
)


DEFAULT_OUT = (
    REPO_ROOT
    / "data"
    / "edgekv_traces"
    / "sharegpt_hotpotqa_session.jsonl"
)

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


def choose_hot_count(total: int, ratio: float) -> int:
    if total <= 0 or ratio <= 0.0:
        return 0
    return min(total, max(1, int(round(total * ratio))))


def exact_hot_cold_counts(requests: int, hot_ratio: float, hot_repeats: int) -> tuple[int, int]:
    if requests <= 0:
        return 0, 0
    if hot_ratio <= 0.0:
        return 0, requests
    if hot_repeats <= 1:
        hot_count = choose_hot_count(requests, hot_ratio)
        return hot_count, requests - hot_count

    # Derived from h / (h * repeats + c) ~= hot_ratio and requests = h * repeats + c.
    hot_count = int(round((hot_ratio * requests) / (1.0 + hot_ratio * (hot_repeats - 1))))
    hot_count = max(1, min(hot_count, requests // hot_repeats))
    cold_count = requests - hot_count * hot_repeats
    return hot_count, cold_count


def build_object_marker(source: str, hot_or_cold: str, object_index: int) -> str:
    return f"[TRACE_OBJECT {source}_{hot_or_cold}_{object_index:04d}]"


def build_sharegpt_prompt(marker: str, context: str, task_index: int) -> str:
    return (
        f"{marker}\n"
        "Context:\n"
        f"{context}\n\n"
        f"Task: {TASKS[task_index % len(TASKS)]}\n"
        "Assistant:"
    )


def build_rag_prompt(marker: str, chunks: Sequence[dict[str, Any]], query: str) -> str:
    context_lines = [
        f"[{row['chunk_id']}] {row['title']}: {row['text']}"
        for row in chunks
    ]
    return (
        f"{marker}\n"
        "Retrieved context:\n"
        + "\n".join(context_lines)
        + "\n\n"
        f"Question: {query}\n"
        "Answer:"
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
    prompt_prefix_mode: str,
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

    sessions: list[dict[str, Any]] = []

    for object_index, source in enumerate(hot_objects):
        context = bounded_context(str(source["seed_text"]), hot_context_words)
        marker = build_object_marker("sharegpt", "hot", object_index)
        reuse_key = f"sharegpt:hot:w{len(context.split())}:obj:{object_index:04d}"
        prompt = build_sharegpt_prompt(marker, context, object_index)
        for repeat_index in range(hot_repeats):
            sessions.append(
                with_temperature_prior(
                    {
                        "session_id": f"{reuse_key}:access:{repeat_index:03d}",
                        "source": "sharegpt",
                        "object_type": "sharegpt_hot_context",
                        "reuse_key": reuse_key,
                        "turns_format": "cumulative_user",
                        "source_index": source.get("source_index", object_index),
                        "source_session_id": source.get("session_id", ""),
                        "reuse_group_index": object_index,
                        "reuse_group_size": hot_repeats,
                        "repeat_index": repeat_index,
                        "prompt_prefix_mode": prompt_prefix_mode,
                        "object_marker": marker,
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
        marker = build_object_marker("sharegpt", "cold", object_index)
        reuse_key = f"sharegpt:cold:w{len(context.split())}:obj:{object_index:04d}"
        prompt = build_sharegpt_prompt(marker, context, object_index)
        sessions.append(
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
                    "prompt_prefix_mode": prompt_prefix_mode,
                    "object_marker": marker,
                    "context_words": len(context.split()),
                    "prompt_est_tokens": estimate_tokens(prompt),
                    "turns": [{"i": 0, "user": prompt}],
                },
                "cold",
            )
        )

    return sessions


def build_original_sharegpt_flat_rows(
    sharegpt_path: Path,
    request_count: int,
    order: str,
) -> list[dict[str, Any]]:
    if request_count <= 0:
        return []
    sessions = load_sharegpt_sessions(
        sharegpt_path,
        max_sessions=request_count,
        order=order,
    )
    rows: list[dict[str, Any]] = []
    for session in sessions:
        session_id = str(session["session_id"])
        source_index = int(session.get("source_index", len(rows)))
        turns = session.get("turns", [])
        if not turns:
            continue
        turn = turns[0]
        turn_index = int(turn.get("i", 0))
        user_text = str(turn.get("user", "")).strip()
        if not user_text:
            continue
        prompt = f"User: {user_text}\nAssistant:"
        rows.append(
            {
                "request_id": f"{session_id}:turn:{turn_index:03d}",
                "session_id": session_id,
                "source": "sharegpt",
                "workload": "sharegpt_original_file_order",
                "object_type": "sharegpt_session_prefix",
                "reuse_key": f"{session_id}:turn:{turn_index:03d}",
                "turn_index": turn_index,
                "source_index": source_index,
                "sharegpt_order": order,
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "prompt_est_tokens": estimate_tokens(prompt),
                "replay_source": "original_sharegpt_random_rag",
            }
        )
        if len(rows) >= request_count:
            break
    if len(rows) < request_count:
        raise RuntimeError(f"only built {len(rows)} ShareGPT requests; requested {request_count}")
    return rows


def build_rag_query_session(
    group_row: dict[str, Any],
    group_index: int,
    repeat_index: int,
    hot_or_cold: str,
    chunk_words_count: int,
    chunks_per_query: int,
    prompt_prefix_mode: str,
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
    marker = build_object_marker("hotpotqa", hot_or_cold, group_index)
    prompt = build_rag_prompt(marker, chunks, query)
    return with_temperature_prior(
        {
            "session_id": f"rag_{hot_or_cold}_{group_index:06d}_access_{repeat_index:03d}",
            "source": "hotpotqa",
            "workload": "rag_chunk_reuse",
            "object_type": f"rag_{hot_or_cold}_chunk_set",
            "reuse_key": reuse_key,
            "turns_format": "complete_prompt",
            "dataset": "hotpotqa",
            "hotpotqa_example_id": group_row["example_id"],
            "hotpotqa_source_path": group_row["source_path"],
            "query": query,
            "answer": group_row["answer"],
            "chunks": chunks,
            "chunk_ids": [row["chunk_id"] for row in chunks],
            "doc_ids": sorted({row["doc_id"] for row in chunks}),
            "reuse_group_index": group_index,
            "reuse_group_size": 1 if hot_or_cold == "cold" else None,
            "repeat_index": repeat_index,
            "prompt_prefix_mode": prompt_prefix_mode,
            "object_marker": marker,
            "chunk_words": chunk_words_total,
            "chunk_words_count": chunk_words_count,
            "chunks_per_query": chunks_per_query,
            "prompt_est_tokens": estimate_tokens(prompt),
            "prompt_is_complete": True,
            "turns": [{"i": 0, "user": prompt, "prompt_is_complete": True}],
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
    prompt_prefix_mode: str,
) -> list[dict[str, Any]]:
    if max_requests <= 0:
        raise ValueError("--rag-requests must be > 0 for the pressure trace")

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

    sessions: list[dict[str, Any]] = []
    for hot_index, group_row in enumerate(hot_groups[:hot_count]):
        for repeat_index in range(hot_repeats):
            session = build_rag_query_session(
                group_row,
                hot_index,
                repeat_index,
                "hot",
                hot_chunk_words,
                hot_chunks_per_query,
                prompt_prefix_mode,
            )
            session["reuse_group_size"] = hot_repeats
            sessions.append(session)

    for cold_index, group_row in enumerate(cold_groups[:cold_count]):
        session = build_rag_query_session(
            group_row,
            hot_count + cold_index,
            0,
            "cold",
            cold_chunk_words,
            cold_chunks_per_query,
            prompt_prefix_mode,
        )
        sessions.append(session)

    if len(sessions) != max_requests:
        raise RuntimeError(f"built {len(sessions)} RAG accesses for target {max_requests}")
    return sessions


def build_8020_rag_flat_rows(
    max_requests: int,
    hotpotqa_path: Path,
    download_hotpotqa: bool,
    chunk_words_count: int,
    chunks_per_query: int,
    query_repeats: int,
    max_examples: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    if max_requests <= 0:
        return []

    groups = load_hotpotqa_chunk_groups(
        hotpotqa_path,
        download_hotpotqa,
        max_examples,
        chunk_words_count,
        chunks_per_query,
        timeout_s,
    )
    group_order = hotpotqa_high_frequency_group_order(len(groups), max_requests)
    group_access_counts = [0 for _ in groups]
    hot_group_count = min(max(1, (len(groups) + 4) // 5), len(groups))
    rows: list[dict[str, Any]] = []

    for request_index, group_index in enumerate(group_order):
        group_row = groups[group_index]
        chunks = group_row["chunks"]
        reuse_key = "rag:hotpotqa:" + "|".join(row["chunk_id"] for row in chunks)
        repeat_index = group_access_counts[group_index] % max(1, query_repeats)
        group_access_counts[group_index] += 1
        query = str(group_row["question"])
        if repeat_index > 0:
            query = f"{query} Answer in a different concise wording. Variant {repeat_index}."
        prompt = build_rag_prompt(f"[TRACE_OBJECT hotpotqa_group_{group_index:04d}]", chunks, query)
        hot_or_tail = "hot" if group_index < hot_group_count else "tail"
        rows.append(
            {
                "request_id": f"rag_{request_index:06d}",
                "session_id": f"rag_session_{request_index:06d}",
                "source": "hotpotqa",
                "workload": "rag_chunk_reuse",
                "object_type": "rag_chunk_set",
                "reuse_key": reuse_key,
                "turn_index": repeat_index,
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "prompt_est_tokens": estimate_tokens(prompt),
                "dataset": "hotpotqa",
                "hotpotqa_example_id": group_row["example_id"],
                "hotpotqa_source_path": group_row["source_path"],
                "query": query,
                "answer": group_row["answer"],
                "chunks": chunks,
                "chunk_ids": [row["chunk_id"] for row in chunks],
                "doc_ids": sorted({row["doc_id"] for row in chunks}),
                "rag_group_index": group_index,
                "rag_group_temperature": hot_or_tail,
                "rag_total_group_count": len(groups),
                "rag_hot_group_count": hot_group_count,
                "rag_group_access_index": group_access_counts[group_index] - 1,
                "replay_source": "original_sharegpt_random_rag",
            }
        )
    return rows


def interleave_rag_random_slots(
    sharegpt_rows: Sequence[dict[str, Any]],
    rag_rows: Sequence[dict[str, Any]],
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    total_requests = len(sharegpt_rows) + len(rag_rows)
    rng = random.Random(random_seed)
    rag_positions = sorted(rng.sample(range(total_requests), len(rag_rows))) if rag_rows else []
    rag_position_set = set(rag_positions)
    rows: list[dict[str, Any]] = []
    sharegpt_pos = 0
    rag_pos = 0
    for request_index in range(total_requests):
        if request_index in rag_position_set:
            row = dict(rag_rows[rag_pos])
            rag_pos += 1
        else:
            row = dict(sharegpt_rows[sharegpt_pos])
            sharegpt_pos += 1
        row["trace_index"] = request_index
        rows.append(row)
    return rows, rag_positions


def temperature_of(row: dict[str, Any]) -> str:
    return str(row.get("temperature", "")).lower()


def prompt_first_line(row: dict[str, Any]) -> str:
    turns = row.get("turns", [])
    if not turns:
        return ""
    return str(turns[0].get("user", "")).splitlines()[0].strip()


def budget_ladder_order(
    sessions: list[dict[str, Any]],
    hot_objects: int,
    cold_scan_objects: int,
    probe_rounds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if hot_objects <= 0 or cold_scan_objects <= 0 or probe_rounds <= 0:
        raise ValueError("budget_ladder requires positive scan-hot-objects, scan-cold-objects, and scan-probe-rounds")

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
    required_cold = cold_scan_objects * probe_rounds
    if len(hot_keys) < hot_objects:
        raise RuntimeError(
            f"budget_ladder needs {hot_objects} hot objects with at least "
            f"{probe_rounds + 1} accesses; found {len(hot_keys)}"
        )
    if len(cold_once) < required_cold:
        raise RuntimeError(
            f"budget_ladder needs {required_cold} unique cold objects "
            f"({cold_scan_objects} per round * {probe_rounds} rounds); found {len(cold_once)}"
        )

    selected_hot_keys = hot_keys[:hot_objects]
    prefix: list[dict[str, Any]] = []
    used: set[int] = set()

    def add(row: dict[str, Any]) -> None:
        prefix.append(row)
        used.add(id(row))

    for key in selected_hot_keys:
        add(hot_by_key[key][0])

    cold_pos = 0
    for probe_index in range(probe_rounds):
        for row in cold_once[cold_pos:cold_pos + cold_scan_objects]:
            add(row)
        cold_pos += cold_scan_objects
        for key in selected_hot_keys:
            add(hot_by_key[key][probe_index + 1])

    ordered = prefix + [row for row in sessions if id(row) not in used]
    return ordered, prefix


def split_evenly(values: Sequence[Any], parts: int) -> list[list[Any]]:
    if parts <= 0:
        raise ValueError("parts must be > 0")
    base, extra = divmod(len(values), parts)
    chunks: list[list[Any]] = []
    pos = 0
    for index in range(parts):
        size = base + (1 if index < extra else 0)
        chunks.append(list(values[pos:pos + size]))
        pos += size
    return chunks


def segmented_ladder_order(
    sessions: list[dict[str, Any]],
    hot_objects: int,
    cold_scan_objects: int,
    probe_rounds: int,
    segments: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if segments <= 0:
        raise ValueError("segmented_ladder requires positive ladder-segments")
    if segments > hot_objects:
        raise ValueError("--ladder-segments must be <= --scan-hot-objects")
    if segments > cold_scan_objects:
        raise ValueError("--ladder-segments must be <= --scan-cold-objects")

    if hot_objects <= 0 or cold_scan_objects <= 0 or probe_rounds <= 0:
        raise ValueError("segmented_ladder requires positive scan-hot-objects, scan-cold-objects, and scan-probe-rounds")

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
    required_cold = cold_scan_objects * probe_rounds
    if len(hot_keys) < hot_objects:
        raise RuntimeError(
            f"segmented_ladder needs {hot_objects} hot objects with at least "
            f"{probe_rounds + 1} accesses; found {len(hot_keys)}"
        )
    if len(cold_once) < required_cold:
        raise RuntimeError(
            f"segmented_ladder needs {required_cold} unique cold objects "
            f"({cold_scan_objects} per round * {probe_rounds} rounds); found {len(cold_once)}"
        )

    selected_hot_keys = hot_keys[:hot_objects]
    hot_key_segments = split_evenly(selected_hot_keys, segments)
    prefix: list[dict[str, Any]] = []
    used: set[int] = set()

    def add(row: dict[str, Any]) -> None:
        prefix.append(row)
        used.add(id(row))

    for key in selected_hot_keys:
        add(hot_by_key[key][0])

    cold_pos = 0
    for probe_index in range(probe_rounds):
        round_cold = cold_once[cold_pos:cold_pos + cold_scan_objects]
        cold_pos += cold_scan_objects
        cold_segments = split_evenly(round_cold, segments)
        for cold_segment, hot_key_segment in zip(cold_segments, hot_key_segments):
            for row in cold_segment:
                add(row)
            for key in hot_key_segment:
                add(hot_by_key[key][probe_index + 1])

    ordered = prefix + [row for row in sessions if id(row) not in used]
    return ordered, prefix


def validate_args(args: argparse.Namespace) -> None:
    if args.reuse_schedule not in {"budget_ladder", "segmented_ladder", "original_sharegpt_random_rag"}:
        raise ValueError("unsupported --reuse-schedule")
    if args.reuse_schedule == "original_sharegpt_random_rag":
        return
    if args.hot_repeats < args.scan_probe_rounds + 1:
        raise ValueError("--hot-repeats must be >= --scan-probe-rounds + 1")
    if args.rag_hot_repeats < args.scan_probe_rounds + 1:
        raise ValueError("--rag-hot-repeats must be >= --scan-probe-rounds + 1")
    if args.reuse_schedule == "segmented_ladder":
        if args.ladder_segments > args.scan_hot_objects:
            raise ValueError("--ladder-segments must be <= --scan-hot-objects")
        if args.ladder_segments > args.scan_cold_objects:
            raise ValueError("--ladder-segments must be <= --scan-cold-objects")


def prompt_prefix_families(sessions: Sequence[dict[str, Any]]) -> set[str]:
    return {prompt_first_line(row) for row in sessions if prompt_first_line(row)}


def source_temperature_stats(sessions: Sequence[dict[str, Any]], source: str) -> dict[str, Any]:
    rows = [row for row in sessions if str(row.get("source")) == source]
    hot = [row for row in rows if row.get("temperature") == "hot"]
    cold = [row for row in rows if row.get("temperature") == "cold"]
    token_key = "prompt_est_tokens"
    return {
        "hot_objects": len({row["reuse_key"] for row in hot}),
        "cold_objects": len({row["reuse_key"] for row in cold}),
        "hot_accesses": len(hot),
        "cold_accesses": len(cold),
        "avg_hot_prompt_est_tokens": round(average([int(row[token_key]) for row in hot]), 2),
        "avg_cold_prompt_est_tokens": round(average([int(row[token_key]) for row in cold]), 2),
    }


def cleanup_old_trace_files(out_path: Path, summary_path: Path) -> list[str]:
    trace_dir = (REPO_ROOT / "data" / "edgekv_traces").resolve()
    out_path = out_path.resolve()
    summary_path = summary_path.resolve()
    if out_path.parent != trace_dir:
        return []

    keep = {out_path.name, summary_path.name}
    removed: list[str] = []
    for path in sorted(trace_dir.iterdir()):
        if not path.is_file() or path.name in keep:
            continue
        if path.suffix == ".jsonl" or path.name.endswith(".jsonl.summary.json") or path.name.endswith(".summary.json"):
            path.unlink()
            removed.append(str(path.relative_to(REPO_ROOT)))
    return removed


def build_summary(args: argparse.Namespace, sessions: Sequence[dict[str, Any]], prefix: Sequence[dict[str, Any]]) -> dict[str, Any]:
    prefix_hot_first_access: dict[str, int] = {}
    cold_scan_tokens_by_round: list[int] = []
    cold_tokens_this_round = 0
    hot_probe_seen = 0
    in_scan = False
    for row in prefix:
        temp = temperature_of(row)
        if temp == "hot":
            key = str(row["reuse_key"])
            prefix_hot_first_access.setdefault(key, int(row.get("prompt_est_tokens", 0)))
            if in_scan:
                hot_probe_seen += 1
                if hot_probe_seen == args.scan_hot_objects:
                    cold_scan_tokens_by_round.append(cold_tokens_this_round)
                    cold_tokens_this_round = 0
                    hot_probe_seen = 0
                    in_scan = False
        elif temp == "cold":
            in_scan = True
            cold_tokens_this_round += int(row.get("prompt_est_tokens", 0))

    all_hot = [row for row in sessions if row.get("temperature") == "hot"]
    all_cold = [row for row in sessions if row.get("temperature") == "cold"]
    summary = {
        "out": str(args.out.expanduser()),
        "source_mode": args.source_mode,
        "total_requests": len(sessions),
        "unique_prompt_prefix_families": len(prompt_prefix_families(sessions)),
        "sharegpt": source_temperature_stats(sessions, "sharegpt"),
        "rag": source_temperature_stats(sessions, "hotpotqa"),
        "avg_hot_prompt_est_tokens": round(average([int(row["prompt_est_tokens"]) for row in all_hot]), 2),
        "avg_cold_prompt_est_tokens": round(average([int(row["prompt_est_tokens"]) for row in all_cold]), 2),
        "estimated_hot_working_set_tokens": sum(prefix_hot_first_access.values()),
        "estimated_cold_scan_tokens_per_round": cold_scan_tokens_by_round,
        "budget_ladder_prefix_requests": len(prefix),
        "budget_ladder": {
            "reuse_schedule": args.reuse_schedule,
            "hot_objects": args.scan_hot_objects,
            "cold_scan_objects": args.scan_cold_objects,
            "probe_rounds": args.scan_probe_rounds,
            "ladder_segments": args.ladder_segments,
        },
        "expected_behavior": (
            "Low KV budgets should evict many primed hot prefixes during each unique cold scan; "
            "mid/high budgets should retain increasingly more hot object prefixes, raising prefix-cache hit rate."
        ),
    }
    return summary


def build_original_sharegpt_random_rag_summary(
    args: argparse.Namespace,
    sessions: Sequence[dict[str, Any]],
    rag_positions: Sequence[int],
) -> dict[str, Any]:
    sharegpt_rows = [row for row in sessions if row.get("source") == "sharegpt"]
    rag_rows = [row for row in sessions if row.get("source") == "hotpotqa"]
    rag_groups_used = len({row.get("rag_group_index") for row in rag_rows})
    rag_total_group_count = int(rag_rows[0].get("rag_total_group_count", rag_groups_used)) if rag_rows else 0
    rag_hot_group_count = int(rag_rows[0].get("rag_hot_group_count", 0)) if rag_rows else 0
    rag_hot_requests = sum(1 for row in rag_rows if row.get("rag_group_temperature") == "hot")
    rag_tail_requests = len(rag_rows) - rag_hot_requests
    return {
        "out": str(args.out.expanduser()),
        "reuse_schedule": args.reuse_schedule,
        "total_requests": len(sessions),
        "sharegpt_requests": len(sharegpt_rows),
        "rag_requests": len(rag_rows),
        "rag_share": round(len(rag_rows) / len(sessions), 6) if sessions else 0.0,
        "sharegpt_order": args.sharegpt_order,
        "random_seed": args.random_seed,
        "rag_insert_positions": list(rag_positions),
        "rag": {
            "group_count": rag_total_group_count,
            "groups_used": rag_groups_used,
            "hot_group_count": rag_hot_group_count,
            "hot_group_fraction": round(rag_hot_group_count / rag_total_group_count, 6) if rag_total_group_count else 0.0,
            "hot_requests": rag_hot_requests,
            "tail_requests": rag_tail_requests,
            "hot_request_fraction": round(rag_hot_requests / len(rag_rows), 6) if rag_rows else 0.0,
            "chunk_words_count": args.rag_chunk_words,
            "chunks_per_query": args.rag_chunks_per_query,
            "query_repeats": args.rag_query_repeats,
        },
        "avg_sharegpt_prompt_est_tokens": round(
            average([int(row["prompt_est_tokens"]) for row in sharegpt_rows]), 2
        ),
        "avg_rag_prompt_est_tokens": round(
            average([int(row["prompt_est_tokens"]) for row in rag_rows]), 2
        ),
        "trace_format": "flat_prompt_jsonl",
        "expected_behavior": (
            "ShareGPT follows source file order while HotpotQA/RAG requests are inserted at "
            "deterministic random slots; the first 20% of RAG chunk groups account for about "
            "80% of RAG requests."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a budget-sensitive H0/H1 pressure replay trace."
    )
    parser.add_argument("--sharegpt-path", type=Path, default=DEFAULT_SHAREGPT_TRACE_PATH)
    parser.add_argument("--hotpotqa-path", type=Path, default=DEFAULT_HOTPOTQA_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--source-mode",
        choices=("mixed", "sharegpt", "hotpotqa"),
        default="mixed",
        help="Select which source(s) to include in the hot/cold pressure trace.",
    )
    parser.add_argument("--sharegpt-groups", type=positive_int, default=96)
    parser.add_argument("--hot-ratio", type=bounded_ratio, default=0.20)
    parser.add_argument("--hot-repeats", type=positive_int, default=4)
    parser.add_argument("--hot-context-words", type=positive_int, default=300)
    parser.add_argument("--cold-context-words", type=positive_int, default=800)
    parser.add_argument("--min-context-words", type=positive_int, default=128)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--rag-requests", type=positive_int, default=128)
    parser.add_argument("--scan-hot-objects", type=positive_int, default=16)
    parser.add_argument("--scan-cold-objects", type=positive_int, default=24)
    parser.add_argument("--scan-probe-rounds", type=positive_int, default=3)
    parser.add_argument("--prompt-prefix-mode", choices=("object_marker", "unique_cold"), default="object_marker")
    parser.add_argument(
        "--reuse-schedule",
        choices=("budget_ladder", "segmented_ladder", "original_sharegpt_random_rag"),
        default="budget_ladder",
    )
    parser.add_argument("--ladder-segments", type=positive_int, default=3)
    parser.add_argument("--total-requests", type=positive_int, default=1024)
    parser.add_argument("--rag-share", type=bounded_ratio, default=0.20)
    parser.add_argument("--sharegpt-order", choices=("file", "longest"), default="file")
    parser.add_argument("--rag-chunk-words", type=positive_int, default=56)
    parser.add_argument("--rag-chunks-per-query", type=positive_int, default=2)
    parser.add_argument("--rag-query-repeats", type=positive_int, default=4)
    parser.add_argument("--hotpotqa-max-examples", type=positive_int, default=96)
    parser.add_argument("--rag-hot-ratio", type=bounded_ratio, default=0.20)
    parser.add_argument("--rag-hot-repeats", type=positive_int, default=4)
    parser.add_argument("--rag-hot-chunk-words", type=positive_int, default=120)
    parser.add_argument("--rag-cold-chunk-words", type=positive_int, default=320)
    parser.add_argument("--rag-hot-chunks-per-query", type=positive_int, default=1)
    parser.add_argument("--rag-cold-chunks-per-query", type=positive_int, default=3)
    parser.add_argument("--download-hotpotqa", action="store_true")
    parser.add_argument(
        "--keep-other-traces",
        action="store_true",
        help="Do not remove older .jsonl trace artifacts from data/edgekv_traces after writing this trace.",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    validate_args(args)
    return args


def main() -> None:
    args = parse_args()
    if args.reuse_schedule == "original_sharegpt_random_rag":
        rag_requests = int(round(args.total_requests * args.rag_share))
        rag_requests = min(args.total_requests, max(0, rag_requests))
        sharegpt_requests = args.total_requests - rag_requests
        sharegpt_rows = build_original_sharegpt_flat_rows(
            args.sharegpt_path.expanduser(),
            sharegpt_requests,
            args.sharegpt_order,
        )
        rag_rows = build_8020_rag_flat_rows(
            max_requests=rag_requests,
            hotpotqa_path=args.hotpotqa_path.expanduser(),
            download_hotpotqa=args.download_hotpotqa,
            chunk_words_count=args.rag_chunk_words,
            chunks_per_query=args.rag_chunks_per_query,
            query_repeats=args.rag_query_repeats,
            max_examples=args.hotpotqa_max_examples,
            timeout_s=args.timeout_s,
        )
        sessions, rag_positions = interleave_rag_random_slots(
            sharegpt_rows,
            rag_rows,
            args.random_seed,
        )
        write_jsonl(args.out.expanduser(), sessions)
        summary = build_original_sharegpt_random_rag_summary(args, sessions, rag_positions)
        summary_path = args.out.expanduser().with_suffix(args.out.expanduser().suffix + ".summary.json")
        removed_old_trace_files: list[str] = []
        if not args.keep_other_traces:
            removed_old_trace_files = cleanup_old_trace_files(args.out.expanduser(), summary_path)
        summary["removed_old_trace_files"] = removed_old_trace_files
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.source_mode == "hotpotqa":
        sessions, meta = build_hotqa_hierarchical_sessions(
            request_count=args.rag_requests,
            hotpotqa_path=args.hotpotqa_path.expanduser(),
            download_hotpotqa=args.download_hotpotqa,
            random_seed=args.random_seed,
            chunk_words_count=args.rag_hot_chunk_words,
            max_examples=args.hotpotqa_max_examples,
            timeout_s=args.timeout_s,
        )
        write_jsonl(args.out.expanduser(), sessions)
        summary_path = args.out.expanduser().with_suffix(args.out.expanduser().suffix + ".summary.json")
        summary = build_hotqa_hierarchical_summary(args.out.expanduser(), sessions, meta)
        summary["source_mode"] = args.source_mode
        summary["removed_old_trace_files"] = []
        if not args.keep_other_traces:
            summary["removed_old_trace_files"] = cleanup_old_trace_files(args.out.expanduser(), summary_path)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    sharegpt_sessions: list[dict[str, Any]] = []
    if args.source_mode in {"mixed", "sharegpt"}:
        sharegpt_sessions = build_hot_cold_sharegpt_sessions(
            args.sharegpt_path.expanduser(),
            args.sharegpt_groups,
            args.hot_ratio,
            args.hot_repeats,
            args.hot_context_words,
            args.cold_context_words,
            args.min_context_words,
            args.random_seed,
            args.prompt_prefix_mode,
        )

    rag_sessions: list[dict[str, Any]] = []
    if args.source_mode in {"mixed", "hotpotqa"}:
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
            prompt_prefix_mode=args.prompt_prefix_mode,
        )
    all_sessions = sharegpt_sessions + rag_sessions
    if args.reuse_schedule == "segmented_ladder":
        sessions, prefix = segmented_ladder_order(
            all_sessions,
            hot_objects=args.scan_hot_objects,
            cold_scan_objects=args.scan_cold_objects,
            probe_rounds=args.scan_probe_rounds,
            segments=args.ladder_segments,
        )
    else:
        sessions, prefix = budget_ladder_order(
            all_sessions,
            hot_objects=args.scan_hot_objects,
            cold_scan_objects=args.scan_cold_objects,
            probe_rounds=args.scan_probe_rounds,
        )
    write_replay_trace(args.out.expanduser(), sessions)
    summary = build_summary(args, sessions, prefix)
    summary_path = args.out.expanduser().with_suffix(args.out.expanduser().suffix + ".summary.json")
    removed_old_trace_files: list[str] = []
    if not args.keep_other_traces:
        removed_old_trace_files = cleanup_old_trace_files(args.out.expanduser(), summary_path)
    summary["removed_old_trace_files"] = removed_old_trace_files
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
