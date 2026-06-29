#!/usr/bin/env python3
"""Generate a hierarchical HotpotQA prefix-reuse replay trace."""

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
DEFAULT_REQUESTS = 256
DEFAULT_RANDOM_SEED = 2026
DEFAULT_GLOBAL_POOL_SIZE = 5
DEFAULT_WARM_POOL_SIZE = 20
DEFAULT_COLD_POOL_SIZE = 100
DEFAULT_WARM_REQUEST_RATIO = 0.60
DEFAULT_COLD_REQUEST_RATIO = 0.20
DEFAULT_CHUNK_WORDS = 120
DEFAULT_MAX_EXAMPLES = 96


def positive_int(value: str) -> int:
    parsed = int(value)
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
    timeout_s: float,
) -> list[dict[str, Any]]:
    groups = load_hotpotqa_chunk_groups(
        hotpotqa_path,
        download_hotpotqa,
        max_examples,
        chunk_words_count,
        1,
        timeout_s,
    )
    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for group in groups:
        chunk = chunk_row_from_group(group)
        chunk_id = str(chunk.get("chunk_id", ""))
        if not chunk_id or chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        chunks.append(chunk)
    return chunks


def format_prefix_chunk(tier: str, chunk: dict[str, Any]) -> str:
    return f"[{tier} {chunk['chunk_id']}] {chunk['title']}: {chunk['text']}"


def build_hotqa_prompt(
    global_chunks: Sequence[dict[str, Any]],
    warm_chunks: Sequence[dict[str, Any]],
    cold_chunks: Sequence[dict[str, Any]],
    question: str,
) -> str:
    sections: list[str] = ["Global Hot Prefix:"]
    sections.extend(format_prefix_chunk("GLOBAL", chunk) for chunk in global_chunks)
    sections.append("")
    sections.append("Warm Prefix:")
    sections.extend(format_prefix_chunk("WARM", chunk) for chunk in warm_chunks)
    sections.append("")
    sections.append("Cold Prefix:")
    sections.extend(format_prefix_chunk("COLD", chunk) for chunk in cold_chunks)
    sections.append("")
    sections.append(f"Question: {question}")
    sections.append("Answer:")
    return "\n".join(sections)


def selected_request_indices(request_count: int, ratio: float, rng: random.Random) -> set[int]:
    selected_count = int(round(request_count * ratio))
    selected_count = max(0, min(request_count, selected_count))
    return set(rng.sample(range(request_count), selected_count)) if selected_count else set()


def build_hotqa_hierarchical_sessions(
    request_count: int = DEFAULT_REQUESTS,
    hotpotqa_path: Path = DEFAULT_HOTPOTQA_PATH,
    download_hotpotqa: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    global_pool_size: int = DEFAULT_GLOBAL_POOL_SIZE,
    warm_pool_size: int = DEFAULT_WARM_POOL_SIZE,
    cold_pool_size: int = DEFAULT_COLD_POOL_SIZE,
    warm_request_ratio: float = DEFAULT_WARM_REQUEST_RATIO,
    cold_request_ratio: float = DEFAULT_COLD_REQUEST_RATIO,
    chunk_words_count: int = DEFAULT_CHUNK_WORDS,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    timeout_s: float = 120.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if request_count <= 0:
        raise ValueError("request_count must be > 0")
    pool_size = global_pool_size + warm_pool_size + cold_pool_size
    chunks = load_unique_hotqa_chunks(
        hotpotqa_path,
        download_hotpotqa,
        max_examples,
        chunk_words_count,
        timeout_s,
    )
    if len(chunks) < pool_size:
        raise RuntimeError(
            f"only built {len(chunks)} unique HotpotQA chunks; need {pool_size} "
            f"({global_pool_size} global + {warm_pool_size} warm + {cold_pool_size} cold)"
        )

    global_pool = chunks[:global_pool_size]
    warm_pool = chunks[global_pool_size:global_pool_size + warm_pool_size]
    cold_pool = chunks[global_pool_size + warm_pool_size:pool_size]
    query_pool = chunks[pool_size:] or chunks

    rng = random.Random(random_seed)
    warm_requests = selected_request_indices(request_count, warm_request_ratio, rng)
    cold_requests = selected_request_indices(request_count, cold_request_ratio, rng)
    sessions: list[dict[str, Any]] = []

    for request_index in range(request_count):
        first_global = global_pool[request_index % len(global_pool)]
        selected_global = [first_global]
        if rng.random() < 0.5:
            selected_global.append(global_pool[(request_index + 1) % len(global_pool)])

        selected_warm: list[dict[str, Any]] = []
        if request_index in warm_requests:
            selected_warm.append(warm_pool[request_index % len(warm_pool)])

        selected_cold: list[dict[str, Any]] = []
        if request_index in cold_requests:
            selected_cold.append(cold_pool[request_index % len(cold_pool)])

        query_row = query_pool[request_index % len(query_pool)]
        question = str(query_row.get("question", "")).strip() or (
            f"Based on the retrieved context, answer request {request_index}."
        )
        answer = str(query_row.get("answer", "")).strip()
        prompt = build_hotqa_prompt(selected_global, selected_warm, selected_cold, question)
        selected_chunks = selected_global + selected_warm + selected_cold
        global_ids = [str(chunk["chunk_id"]) for chunk in selected_global]
        warm_ids = [str(chunk["chunk_id"]) for chunk in selected_warm]
        cold_ids = [str(chunk["chunk_id"]) for chunk in selected_cold]
        chunk_ids = global_ids + warm_ids + cold_ids
        doc_ids = sorted({str(chunk["doc_id"]) for chunk in selected_chunks})
        reuse_key = "hotqa:hierarchical:" + "|".join(chunk_ids)
        session = {
            "session_id": f"hotqa_hier_{request_index:06d}",
            "source": "hotpotqa",
            "workload": "rag_chunk_reuse",
            "object_type": "rag_hierarchical_prefix",
            "reuse_key": reuse_key,
            "turns_format": "complete_prompt",
            "dataset": "hotpotqa",
            "hotpotqa_example_id": str(query_row.get("example_id", "")),
            "hotpotqa_source_path": str(query_row.get("source_path", "")),
            "query": question,
            "answer": answer,
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
            "prefix_layout": "hierarchical",
            "global_prefix_ids": global_ids,
            "warm_prefix_ids": warm_ids,
            "cold_prefix_ids": cold_ids,
            "prefix_tiers": {
                "global": global_ids,
                "warm": warm_ids,
                "cold": cold_ids,
            },
            "prompt_est_tokens": estimate_tokens(prompt),
            "prompt_is_complete": True,
            "turns": [{"i": 0, "user": prompt, "prompt_is_complete": True}],
        }
        sessions.append(session)

    meta = {
        "global_pool_ids": [str(chunk["chunk_id"]) for chunk in global_pool],
        "warm_pool_ids": [str(chunk["chunk_id"]) for chunk in warm_pool],
        "cold_pool_ids": [str(chunk["chunk_id"]) for chunk in cold_pool],
        "random_seed": random_seed,
        "chunk_words_count": chunk_words_count,
        "max_examples": max_examples,
        "warm_request_ratio_target": warm_request_ratio,
        "cold_request_ratio_target": cold_request_ratio,
    }
    return sessions, meta


def build_hotqa_hierarchical_summary(
    out_path: Path,
    sessions: Sequence[dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    request_count = len(sessions)
    warm_requests = sum(1 for row in sessions if row.get("warm_prefix_ids"))
    cold_requests = sum(1 for row in sessions if row.get("cold_prefix_ids"))
    prompt_tokens = [int(row.get("prompt_est_tokens", 0)) for row in sessions]
    return {
        "out": str(out_path),
        "source": "hotpotqa",
        "workload": "rag_chunk_reuse",
        "prefix_layout": "hierarchical",
        "requests": request_count,
        "total_requests": request_count,
        "tier_pool_counts": {
            "global": len(meta.get("global_pool_ids", [])),
            "warm": len(meta.get("warm_pool_ids", [])),
            "cold": len(meta.get("cold_pool_ids", [])),
        },
        "tier_request_counts": {
            "global": request_count,
            "warm": warm_requests,
            "cold": cold_requests,
        },
        "tier_coverage": {
            "global": 1.0 if request_count else 0.0,
            "warm": round(warm_requests / request_count, 6) if request_count else 0.0,
            "cold": round(cold_requests / request_count, 6) if request_count else 0.0,
        },
        "unique_selected_tier_chunks": {
            "global": len({cid for row in sessions for cid in row.get("global_prefix_ids", [])}),
            "warm": len({cid for row in sessions for cid in row.get("warm_prefix_ids", [])}),
            "cold": len({cid for row in sessions for cid in row.get("cold_prefix_ids", [])}),
        },
        "avg_prompt_est_tokens": round(average(prompt_tokens), 2),
        "min_prompt_est_tokens": min(prompt_tokens) if prompt_tokens else 0,
        "max_prompt_est_tokens": max(prompt_tokens) if prompt_tokens else 0,
        "random_seed": meta.get("random_seed"),
        "chunk_words_count": meta.get("chunk_words_count"),
        "max_examples": meta.get("max_examples"),
        "global_pool_ids": meta.get("global_pool_ids", []),
        "warm_pool_ids": meta.get("warm_pool_ids", []),
        "cold_pool_size": len(meta.get("cold_pool_ids", [])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a HotpotQA hierarchical prefix replay trace."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--requests", type=positive_int, default=DEFAULT_REQUESTS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--hotpotqa-path", type=Path, default=DEFAULT_HOTPOTQA_PATH)
    parser.add_argument("--download-hotpotqa", action="store_true")
    parser.add_argument("--global-pool-size", type=positive_int, default=DEFAULT_GLOBAL_POOL_SIZE)
    parser.add_argument("--warm-pool-size", type=positive_int, default=DEFAULT_WARM_POOL_SIZE)
    parser.add_argument("--cold-pool-size", type=positive_int, default=DEFAULT_COLD_POOL_SIZE)
    parser.add_argument("--warm-request-ratio", type=bounded_ratio, default=DEFAULT_WARM_REQUEST_RATIO)
    parser.add_argument("--cold-request-ratio", type=bounded_ratio, default=DEFAULT_COLD_REQUEST_RATIO)
    parser.add_argument("--chunk-words", type=positive_int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--hotpotqa-max-examples", type=positive_int, default=DEFAULT_MAX_EXAMPLES)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out.expanduser()
    sessions, meta = build_hotqa_hierarchical_sessions(
        request_count=args.requests,
        hotpotqa_path=args.hotpotqa_path.expanduser(),
        download_hotpotqa=args.download_hotpotqa,
        random_seed=args.random_seed,
        global_pool_size=args.global_pool_size,
        warm_pool_size=args.warm_pool_size,
        cold_pool_size=args.cold_pool_size,
        warm_request_ratio=args.warm_request_ratio,
        cold_request_ratio=args.cold_request_ratio,
        chunk_words_count=args.chunk_words,
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
