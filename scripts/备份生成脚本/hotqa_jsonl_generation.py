#!/usr/bin/env python3
"""Generate a prefix-cache-friendly HotpotQA replay trace.

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
    DEFAULT_HOTPOTQA_PATH,
    estimate_tokens,
    load_hotpotqa_chunk_groups,
    write_jsonl,
)


DEFAULT_OUT = (
    REPO_ROOT
    / "data"
    / "edgekv_traces"
    / "source_ablation"
    / "hotqa.jsonl"
)
DEFAULT_REQUESTS = 768
DEFAULT_RANDOM_SEED = 2026
DEFAULT_HOT_POOL_SIZE = 1
DEFAULT_WARM_POOL_SIZE = 300
DEFAULT_ZIPF_S = 0.8
DEFAULT_TAIL_WORDS = 8
DEFAULT_CHUNK_WORDS = 240
DEFAULT_MIN_CHUNK_WORDS = 240
DEFAULT_MAX_EXAMPLES = 200

# Single constant task, kept in the SHARED prefix (before the unique tail) so it
# never breaks prefix-cache matching. Per-request task/question rotation would
# inflate the always-miss region and depress the ceiling.
CONSTANT_TASK = "Answer the question using the retrieved HotpotQA context above."

# Deterministic filler vocabulary for padding the per-request unique tail to a
# fixed word budget. The uniqueness still comes from the session id, the filler
# only pads to a stable token length.
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


def bounded_ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
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


def chunk_row_from_group(group: dict[str, Any]) -> dict[str, Any]:
    chunks = group.get("chunks", [])
    if not chunks:
        raise ValueError("HotpotQA group has no chunks")
    chunk = dict(chunks[0])
    chunk["example_id"] = group.get("example_id", "")
    chunk["question"] = group.get("question", "")
    chunk["answer"] = group.get("answer", "")
    chunk["source_path"] = group.get("source_path", "")
    return chunk


def load_unique_hotqa_chunks(
    hotpotqa_path: Path,
    download_hotpotqa: bool,
    max_examples: int,
    chunk_words_count: int,
    min_chunk_words: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    """Load HotpotQA text and re-chunk it into UNIFORM, large prefix chunks.

    HotpotQA context paragraphs are intrinsically small (~50-150 words), so one
    paragraph per chunk yields tiny chunks where the unique tail + warm
    first-touch misses dominate and the prefix-cache hit ceiling is capped at
    ~0.76. To lift the ceiling we mirror ``sharedgpt_jsonl_generation.py``: stream
    all paragraph text in deterministic load order and re-slice into fixed
    ``chunk_words_count``-word chunks, dropping any trailing remainder shorter
    than ``min_chunk_words``. The content is still entirely HotpotQA-derived.
    """
    groups = load_hotpotqa_chunk_groups(
        hotpotqa_path,
        download_hotpotqa,
        max_examples,
        # Pull whole paragraphs (large window) so the loader does not pre-split;
        # we control the final chunk size ourselves below.
        max(chunk_words_count, min_chunk_words) * 4,
        1,
        timeout_s,
    )
    # Concatenate every unique paragraph's words in deterministic load order.
    words: list[str] = []
    seen_ids: set[str] = set()
    for group in groups:
        paragraph = chunk_row_from_group(group)
        paragraph_id = str(paragraph.get("chunk_id", ""))
        if not paragraph_id or paragraph_id in seen_ids:
            continue
        seen_ids.add(paragraph_id)
        words.extend(str(paragraph.get("text", "")).split())

    # Re-slice into uniform chunks; drop the trailing under-min remainder.
    chunks: list[dict[str, Any]] = []
    for start in range(0, len(words), max(chunk_words_count, 1)):
        piece = words[start:start + chunk_words_count]
        if len(piece) < min_chunk_words:
            continue
        chunk_id = f"hotqa:seg:{len(chunks):05d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": chunk_id,
                "title": "context",
                "text": " ".join(piece),
            }
        )
    return chunks


def format_prefix_chunk(tier: str, chunk: dict[str, Any]) -> str:
    return f"[{tier} {chunk['chunk_id']}] {chunk['title']}: {chunk['text']}"


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


def build_hotqa_prompt(
    global_chunks: Sequence[dict[str, Any]],
    warm_chunks: Sequence[dict[str, Any]],
    task: str,
    tail_text: str,
) -> str:
    sections: list[str] = ["Global Hot Prefix:"]
    sections.extend(format_prefix_chunk("GLOBAL", chunk) for chunk in global_chunks)
    sections.append("")
    sections.append("Warm Prefix:")
    sections.extend(format_prefix_chunk("WARM", chunk) for chunk in warm_chunks)
    sections.append("")
    sections.append(f"Task: {task}")
    sections.append(f"Q: {tail_text}")
    sections.append("Answer:")
    return "\n".join(sections)


def zipf_weights(pool_size: int, s: float) -> list[float]:
    return [1.0 / ((rank + 1) ** s) for rank in range(pool_size)]


def build_hotqa_hierarchical_sessions(
    request_count: int = DEFAULT_REQUESTS,
    hotpotqa_path: Path = DEFAULT_HOTPOTQA_PATH,
    download_hotpotqa: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    hot_pool_size: int = DEFAULT_HOT_POOL_SIZE,
    warm_pool_size: int = DEFAULT_WARM_POOL_SIZE,
    zipf_s: float = DEFAULT_ZIPF_S,
    tail_words: int = DEFAULT_TAIL_WORDS,
    chunk_words_count: int = DEFAULT_CHUNK_WORDS,
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    timeout_s: float = 120.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request_count <= 0:
        raise ValueError("request_count must be > 0")
    pool_size = hot_pool_size + warm_pool_size
    chunks = load_unique_hotqa_chunks(
        hotpotqa_path,
        download_hotpotqa,
        max_examples,
        chunk_words_count,
        min_chunk_words,
        timeout_s,
    )
    if len(chunks) < pool_size:
        raise RuntimeError(
            f"only built {len(chunks)} unique HotpotQA chunks; need {pool_size} "
            f"({hot_pool_size} hot + {warm_pool_size} warm)"
        )

    hot_pool = chunks[:hot_pool_size]
    warm_pool = chunks[hot_pool_size:hot_pool_size + warm_pool_size]

    rng = random.Random(random_seed)
    weights = zipf_weights(warm_pool_size, zipf_s)
    sessions: list[dict[str, Any]] = []

    for request_index in range(request_count):
        # HOT: entire hot pool, identical order, every request -> constant floor.
        selected_global = list(hot_pool)

        # WARM: exactly one chunk chosen by Zipf popularity, placed right after
        # the identical hot region so its blocks reuse on repeat picks.
        warm_index = rng.choices(range(warm_pool_size), weights=weights, k=1)[0]
        selected_warm = [warm_pool[warm_index]]

        task = CONSTANT_TASK
        session_id = f"hotqa_hier_{request_index:06d}"
        tail_text = build_unique_tail(session_id, request_index, tail_words)
        prompt = build_hotqa_prompt(selected_global, selected_warm, task, tail_text)

        selected_chunks = selected_global + selected_warm
        global_ids = [str(chunk["chunk_id"]) for chunk in selected_global]
        warm_ids = [str(chunk["chunk_id"]) for chunk in selected_warm]
        cold_ids: list[str] = []
        chunk_ids = global_ids + warm_ids
        doc_ids = sorted({str(chunk["doc_id"]) for chunk in selected_chunks})
        reuse_key = "hotqa:hot+warm:" + "|".join(global_ids) + "|" + "|".join(warm_ids)
        session = {
            "session_id": session_id,
            "source": "hotpotqa",
            "workload": "rag_chunk_reuse",
            "object_type": "rag_hierarchical_prefix",
            "reuse_key": reuse_key,
            "turns_format": "complete_prompt",
            "dataset": "hotpotqa",
            "task": task,
            "query": tail_text,
            "answer": "",
            "chunk_ids": chunk_ids,
            "doc_ids": doc_ids,
            "chunks": [
                {
                    "chunk_id": str(chunk["chunk_id"]),
                    "doc_id": str(chunk["doc_id"]),
                    "title": str(chunk["title"]),
                    "text": str(chunk["text"]),
                }
                for chunk in selected_chunks
            ],
            "prefix_layout": "hot_warm_tail",
            "global_prefix_ids": global_ids,
            "warm_prefix_ids": warm_ids,
            "cold_prefix_ids": cold_ids,
            "prefix_tiers": {
                "global": global_ids,
                "warm": warm_ids,
                "cold": cold_ids,
            },
            "warm_zipf_rank": warm_index,
            "tail_text": tail_text,
            "prompt_est_tokens": estimate_tokens(prompt),
            "prompt_is_complete": True,
            "turns": [{"i": 0, "user": prompt, "prompt_is_complete": True}],
        }
        sessions.append(session)

    meta = {
        "hot_pool_ids": [str(chunk["chunk_id"]) for chunk in hot_pool],
        "warm_pool_ids": [str(chunk["chunk_id"]) for chunk in warm_pool],
        "random_seed": random_seed,
        "hot_pool_size": hot_pool_size,
        "warm_pool_size": warm_pool_size,
        "zipf_s": zipf_s,
        "tail_words": tail_words,
        "chunk_words_count": chunk_words_count,
        "min_chunk_words": min_chunk_words,
        "max_examples": max_examples,
    }
    return sessions, meta


def build_hotqa_hierarchical_summary(
    out_path: Path,
    sessions: Sequence[dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    request_count = len(sessions)
    warm_requests = sum(1 for row in sessions if row.get("warm_prefix_ids"))
    prompt_tokens = [int(row.get("prompt_est_tokens", 0)) for row in sessions]
    warm_ranks = [int(row.get("warm_zipf_rank", 0)) for row in sessions]
    warm_rank_histogram: dict[str, int] = {}
    for rank in warm_ranks:
        key = str(rank)
        warm_rank_histogram[key] = warm_rank_histogram.get(key, 0) + 1
    return {
        "out": str(out_path),
        "source": "hotpotqa",
        "workload": "rag_chunk_reuse",
        "prefix_layout": "hot_warm_tail",
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
        "unique_selected_tier_chunks": {
            "hot": len({cid for row in sessions for cid in row.get("global_prefix_ids", [])}),
            "warm": len({cid for row in sessions for cid in row.get("warm_prefix_ids", [])}),
        },
        "warm_unique_used": len({cid for row in sessions for cid in row.get("warm_prefix_ids", [])}),
        "warm_zipf_rank_top10": {
            str(rank): warm_rank_histogram.get(str(rank), 0) for rank in range(10)
        },
        "unique_reuse_keys": len({str(row.get("reuse_key", "")) for row in sessions}),
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
        "random_seed": meta.get("random_seed"),
        "hot_pool_size": meta.get("hot_pool_size"),
        "warm_pool_size": meta.get("warm_pool_size"),
        "zipf_s": meta.get("zipf_s"),
        "tail_words": meta.get("tail_words"),
        "chunk_words_count": meta.get("chunk_words_count"),
        "min_chunk_words": meta.get("min_chunk_words"),
        "max_examples": meta.get("max_examples"),
        "hot_pool_ids": meta.get("hot_pool_ids", []),
        "warm_pool_ids": meta.get("warm_pool_ids", []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a prefix-cache-friendly HotpotQA replay trace."
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
    parser.add_argument("--hotpotqa-path", type=Path, default=DEFAULT_HOTPOTQA_PATH)
    parser.add_argument("--download-hotpotqa", action="store_true")
    # --global-pool-size is reused as the HOT pool size (default 1).
    parser.add_argument("--global-pool-size", type=positive_int, default=DEFAULT_HOT_POOL_SIZE)
    parser.add_argument("--warm-pool-size", type=positive_int, default=DEFAULT_WARM_POOL_SIZE)
    parser.add_argument("--zipf-s", type=positive_float, default=DEFAULT_ZIPF_S)
    parser.add_argument("--tail-words", type=positive_int, default=DEFAULT_TAIL_WORDS)
    parser.add_argument("--chunk-words", type=positive_int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--min-chunk-words", type=positive_int, default=DEFAULT_MIN_CHUNK_WORDS)
    parser.add_argument("--hotpotqa-max-examples", type=positive_int, default=DEFAULT_MAX_EXAMPLES)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    # Deprecated knobs from the old hierarchical layout: accepted but ignored so
    # existing command lines keep working.
    parser.add_argument("--cold-pool-size", type=positive_int, default=None,
                        help="Deprecated; ignored (cold tier removed).")
    parser.add_argument("--warm-request-ratio", type=bounded_ratio, default=None,
                        help="Deprecated; ignored (every request now has warm).")
    parser.add_argument("--cold-request-ratio", type=bounded_ratio, default=None,
                        help="Deprecated; ignored (cold tier removed).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out.expanduser()
    sessions, meta = build_hotqa_hierarchical_sessions(
        request_count=args.num_prompts or args.requests,
        hotpotqa_path=args.hotpotqa_path.expanduser(),
        download_hotpotqa=args.download_hotpotqa,
        random_seed=args.random_seed,
        hot_pool_size=args.global_pool_size,
        warm_pool_size=args.warm_pool_size,
        zipf_s=args.zipf_s,
        tail_words=args.tail_words,
        chunk_words_count=args.chunk_words,
        min_chunk_words=args.min_chunk_words,
        max_examples=args.hotpotqa_max_examples,
        timeout_s=args.timeout_s,
    )
    write_jsonl(out_path, sessions)
    summary = build_hotqa_hierarchical_summary(out_path, sessions, meta)
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
