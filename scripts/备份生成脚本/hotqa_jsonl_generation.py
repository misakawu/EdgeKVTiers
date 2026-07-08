#!/usr/bin/env python3
"""生成对 prefix-cache 友好的 HotpotQA replay trace。

每个请求的布局（严格保持位置稳定，使 vLLM 块级 prefix cache 可复用）：

    HOT  （每个请求都固定，决定下限）
    WARM （精确一个 chunk，按 Zipf 热度选择，推动命中率上升）
    TAIL （每请求唯一的短字符串，限制上限）

只有“选择哪个 warm chunk”是随机的；warm chunk 始终紧跟在相同 HOT 区域之后，
因此选择相同 warm chunk 的两个请求会共享完整的 `HOT + WARM` token 前缀并复用其 blocks。
unique tail 是唯一永不复用的区域，因此决定命中率上限缺口。
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

# 使用单个固定任务，并放在 SHARED 前缀中（位于 unique tail 之前），
# 避免破坏 prefix-cache 匹配。按请求轮换任务/问题会
# 扩大永远 miss 的区域，并压低命中率上限。
CONSTANT_TASK = "Answer the question using the retrieved HotpotQA context above."

# 使用确定性 filler 词表，把每个请求的 unique tail 填充到
# 固定词数预算。唯一性仍来自 session id，filler
# 只用于填充到稳定 token 长度。
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
    """加载 HotpotQA 文本，并重新切分为均匀的大 prefix chunks。

    HotpotQA context 段落天然较短（约 50-150 词），如果每段一个 chunk，
    会形成很小的 chunks，使 unique tail + warm 首次触碰 miss 占主导，
    prefix-cache 命中率上限被压到约 0.76。为抬高上限，这里对齐
    ``sharedgpt_jsonl_generation.py``：按确定性加载顺序流式拼接所有段落文本，
    再重新切成固定 ``chunk_words_count`` 词的 chunks，并丢弃短于
    ``min_chunk_words`` 的尾部余量。内容仍完全来自 HotpotQA。
    """
    groups = load_hotpotqa_chunk_groups(
        hotpotqa_path,
        download_hotpotqa,
        max_examples,
        # 拉取完整段落（大窗口），避免 loader 预先切分；
        # 最终 chunk 大小由下方逻辑自行控制。
        max(chunk_words_count, min_chunk_words) * 4,
        1,
        timeout_s,
    )
    # 按确定性加载顺序拼接每个唯一段落的词。
    words: list[str] = []
    seen_ids: set[str] = set()
    for group in groups:
        paragraph = chunk_row_from_group(group)
        paragraph_id = str(paragraph.get("chunk_id", ""))
        if not paragraph_id or paragraph_id in seen_ids:
            continue
        seen_ids.add(paragraph_id)
        words.extend(str(paragraph.get("text", "")).split())

    # 重新切分为均匀 chunks；丢弃末尾不足最小长度的余量。
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
    """每个请求唯一的短字符串（用于设定命中率上限缺口）。

    通过嵌入请求索引保证唯一性；剩余预算用确定性 filler 填充到稳定 token 长度。
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
        # HOT：每个请求都按相同顺序使用整个 hot pool，形成稳定下限。
        selected_global = list(hot_pool)

        # WARM：按 Zipf 热度精确选择一个 chunk，并放在
        # 相同 hot 区域之后，使其 block 在重复选择时复用。
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
    # 旧层次化布局遗留参数：接受但忽略，
    # 以保证现有命令行仍可运行。
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
