#!/usr/bin/env python3
"""Replay ShareGPT + RAG reuse prompts against a vLLM server.

This is the real-engine H0 runner: it assumes vLLM is already serving with
``--enable-prefix-caching`` and measures request TTFT, end-to-end latency, and
GPU memory peak while replaying prompts that share prefixes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
DEFAULT_HOTPOTQA_PATH = Path("/DATACENTER3/zhenxiang.wang/data/hotpotqa")
DEFAULT_REPLAY_TRACE_PATH = Path(
    "/DATACENTER3/zhenxiang.wang/data/edgekv_traces/h0_sharegpt_hotpotqa_200sessions.jsonl"
)
HOTPOTQA_HF_BASE_URL = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor"
HOTPOTQA_HF_FILES = (
    "validation-00000-of-00001.parquet",
    "train-00000-of-00002.parquet",
    "train-00001-of-00002.parquet",
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
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


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


def load_sharegpt_sessions(path: Path, max_sessions: int, order: str = "file") -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    sessions: List[dict] = []
    for raw_session_idx, row in enumerate(rows):
        conversations = row.get("conversations", [])
        if not isinstance(conversations, list):
            continue
        human_messages = [
            message_text(msg).strip()
            for msg in conversations
            if message_role(msg) in {"human", "user"} and message_text(msg).strip()
        ]
        if len(human_messages) < 2:
            continue
        session_id = str(row.get("id", f"sharegpt_{raw_session_idx:06d}"))
        turns = [
            {"i": i, "user": text}
            for i, text in enumerate(human_messages)
        ]
        sessions.append(
            {
                "session_id": session_id,
                "source": "sharegpt",
                "object_type": "sharegpt_session_prefix",
                "reuse_key": session_id,
                "source_index": raw_session_idx,
                "total_user_chars": sum(len(text) for text in human_messages),
                "turns": turns,
            }
        )
        if order == "file" and len(sessions) >= max_sessions:
            break
    if order == "longest":
        sessions.sort(key=lambda item: (int(item.get("total_user_chars", 0)), len(item.get("turns", []))), reverse=True)
    return sessions[:max_sessions]


def sharegpt_session_to_prompts(session: dict, tokenizer=None) -> List[dict]:
    prompts: List[dict] = []
    session_id = str(session["session_id"])
    transcript: List[str] = []
    for turn in session.get("turns", []):
        turn_index = int(turn.get("i", len(prompts)))
        user_text = str(turn.get("user", "")).strip()
        if not user_text:
            continue
        rag = turn.get("rag")
        rag_prefix = ""
        if isinstance(rag, dict):
            context_lines = [
                f"[{row['chunk_id']}] {row['title']}: {row['text']}"
                for row in rag.get("chunks", [])
            ]
            if context_lines:
                rag_prefix = "Retrieved context:\n" + "\n".join(context_lines) + "\n\n"
        prompt = rag_prefix + "\n".join(transcript + [f"User: {user_text}", "Assistant:"])
        row = {
            "request_id": f"{session_id}:turn:{turn_index:03d}",
            "session_id": session_id,
            "turn_index": turn_index,
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "prompt_est_tokens": estimate_tokens(prompt),
            "n_tokens": count_tokens(prompt, tokenizer),
            "workload": "sharegpt_session_prefix",
            "object_type": str(session.get("object_type", "sharegpt_session_prefix")),
            "reuse_key": str(session.get("reuse_key", session_id)),
            "replay_source": "frozen_replay_trace",
        }
        if isinstance(rag, dict):
            chunks = rag.get("chunks", [])
            row.update(
                {
                    "workload": "sharegpt_hotpotqa_weak_link",
                    "rag_reuse_key": str(rag.get("reuse_key", "")),
                    "rag_chunk_ids": [chunk.get("chunk_id", "") for chunk in chunks],
                    "rag_doc_ids": sorted({chunk.get("doc_id", "") for chunk in chunks}),
                    "dataset": rag.get("dataset", "hotpotqa"),
                    "hotpotqa_example_id": rag.get("hotpotqa_example_id", ""),
                    "hotpotqa_source_path": rag.get("hotpotqa_source_path", ""),
                    "rag_match_mode": rag.get("match_mode", "weak_round_robin"),
                }
            )
        prompts.append(row)
        transcript.append(f"User: {user_text}")
        assistant_text = str(turn.get("assistant", "")).strip()
        if assistant_text:
            transcript.append(f"Assistant: {assistant_text}")
    return prompts


def chunk_words(text: str, chunk_words_count: int) -> List[str]:
    words = (text or "").split()
    if not words:
        return []
    chunks = []
    for start in range(0, len(words), max(chunk_words_count, 1)):
        chunk = " ".join(words[start : start + chunk_words_count])
        if chunk:
            chunks.append(chunk)
    return chunks


def resolve_hotpotqa_files(path: Path, download: bool, timeout_s: float) -> List[Path]:
    if path.is_dir():
        files = sorted(path.glob("validation-*.parquet")) + sorted(path.glob("train-*.parquet"))
        if files:
            return files
    elif path.exists():
        return [path]
    if not download:
        raise FileNotFoundError(
            f"HotpotQA parquet/json file not found: {path}. Put Hugging Face distractor parquet files there, "
            "pass --download-hotpotqa, or set --hotpotqa-path."
        )
    target_dir = path if path.suffix == "" else path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    for filename in HOTPOTQA_HF_FILES:
        target = target_dir / filename
        if not target.exists():
            url = f"{HOTPOTQA_HF_BASE_URL}/{filename}"
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                target.write_bytes(resp.read())
        downloaded.append(target)
    return downloaded


def iter_hotpotqa_rows(path: Path) -> Iterable[dict]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            raise RuntimeError(
                "Reading HotpotQA parquet requires pyarrow. Install pyarrow in this environment "
                "or convert the Hugging Face parquet files to JSON/JSONL and pass --hotpotqa-path."
            ) from exc
        table = pq.read_table(path)
        for row in table.to_pylist():
            if isinstance(row, dict):
                yield row
        return

    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            rows = json.load(f)
            for row in rows:
                if isinstance(row, dict):
                    yield row
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def hotpot_context_items(row: dict) -> List[dict]:
    context = row.get("context", [])
    items: List[dict] = []
    if isinstance(context, list) and len(context) == 2 and all(isinstance(x, list) for x in context):
        titles, sentence_groups = context
        if len(titles) == len(sentence_groups):
            context = list(zip(titles, sentence_groups))
    if isinstance(context, dict):
        if "title" in context and "sentences" in context:
            titles = context.get("title", [])
            sentence_groups = context.get("sentences", [])
            if isinstance(titles, list) and isinstance(sentence_groups, list):
                context = list(zip(titles, sentence_groups))
            else:
                context = list(context.items())
        else:
            context = list(context.items())
    for index, item in enumerate(context):
        title = f"doc_{index:02d}"
        sentences: object = []
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title = str(item[0])
            sentences = item[1]
        elif isinstance(item, dict):
            title = str(item.get("title", title))
            sentences = item.get("sentences", item.get("text", []))
        if isinstance(sentences, str):
            text = sentences
        elif isinstance(sentences, list):
            text = " ".join(str(sentence) for sentence in sentences)
        else:
            text = str(sentences)
        text = " ".join(text.split())
        if text:
            items.append({"title": title, "text": text})
    return items


def load_hotpotqa_chunk_groups(
    path: Path,
    download: bool,
    max_examples: int,
    chunk_words_count: int,
    chunks_per_query: int,
    timeout_s: float,
) -> List[dict]:
    paths = resolve_hotpotqa_files(path, download, timeout_s)
    groups: List[dict] = []
    group_size = max(1, chunks_per_query)
    example_index = 0
    for source_path in paths:
        for row in iter_hotpotqa_rows(source_path):
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            context_items = hotpot_context_items(row)
            chunk_rows: List[dict] = []
            for doc_index, item in enumerate(context_items):
                title = item["title"]
                for chunk_index, text in enumerate(chunk_words(item["text"], chunk_words_count)):
                    chunk_rows.append(
                        {
                            "chunk_id": f"hotpot:{example_index:06d}:doc:{doc_index:02d}:chunk:{chunk_index:02d}",
                            "doc_id": f"hotpot:{example_index:06d}:doc:{doc_index:02d}",
                            "title": title,
                            "text": text,
                        }
                    )
            for start in range(0, len(chunk_rows), group_size):
                group = chunk_rows[start : start + group_size]
                if not group:
                    continue
                groups.append(
                    {
                        "example_id": str(row.get("_id", row.get("id", f"hotpot_{example_index:06d}"))),
                        "question": question,
                        "answer": answer,
                        "source_path": str(source_path),
                        "chunks": group,
                    }
                )
            example_index += 1
            if max_examples > 0 and example_index >= max_examples:
                break
        if max_examples > 0 and example_index >= max_examples:
            break
    if not groups:
        raise RuntimeError(f"HotpotQA files produced no RAG chunk groups: {path}")
    return groups


def build_rag_chunk_prompts(
    max_requests: int,
    hotpotqa_path: Path,
    download_hotpotqa: bool,
    tokenizer=None,
    chunk_words_count: int = 56,
    chunks_per_query: int = 2,
    query_repeats: int = 4,
    max_examples: int = 10,
    timeout_s: float = 120.0,
) -> List[dict]:
    """Build a deterministic HotpotQA RAG trace with cross-query chunk reuse.

    Each chunk set is placed at the beginning of the prompt and is queried with
    several different question suffixes. That shape lets vLLM prefix caching
    reuse the retrieved chunk prefix across requests.
    """

    prompts: List[dict] = []
    if max_requests <= 0:
        return prompts

    groups = load_hotpotqa_chunk_groups(
        hotpotqa_path,
        download_hotpotqa,
        max_examples,
        chunk_words_count,
        chunks_per_query,
        timeout_s,
    )
    request_index = 0
    while len(prompts) < max_requests:
        group_row = groups[request_index % len(groups)]
        group = group_row["chunks"]
        reuse_key = "rag:hotpotqa:" + "|".join(row["chunk_id"] for row in group)
        context_lines = []
        for row in group:
            context_lines.append(f"[{row['chunk_id']}] {row['title']}: {row['text']}")
        context_prefix = "Retrieved context:\n" + "\n".join(context_lines) + "\n\n"
        repeat_index = (request_index // len(groups)) % max(1, query_repeats)
        query = group_row["question"]
        if repeat_index > 0:
            query = f"{query} Answer in a different concise wording. Variant {repeat_index}."
        prompt = context_prefix + f"Question: {query}\nAnswer:"
        n_tokens = count_tokens(prompt, tokenizer)
        prompts.append(
            {
                "request_id": f"rag_{request_index:06d}",
                "session_id": f"rag_session_{request_index:06d}",
                "turn_index": repeat_index,
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "prompt_est_tokens": estimate_tokens(prompt),
                "n_tokens": n_tokens,
                "workload": "rag_chunk_reuse",
                "object_type": "rag_chunk_set",
                "reuse_key": reuse_key,
                "chunk_ids": [row["chunk_id"] for row in group],
                "doc_ids": sorted({row["doc_id"] for row in group}),
                "dataset": "hotpotqa",
                "hotpotqa_example_id": group_row["example_id"],
                "hotpotqa_source_path": group_row["source_path"],
                "query": query,
                "answer": group_row["answer"],
            }
        )
        request_index += 1
    return prompts[:max_requests]


def build_rag_sessions(
    max_requests: int,
    hotpotqa_path: Path,
    download_hotpotqa: bool,
    chunk_words_count: int = 56,
    chunks_per_query: int = 2,
    query_repeats: int = 4,
    max_examples: int = 10,
    timeout_s: float = 120.0,
) -> List[dict]:
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
    sessions: List[dict] = []
    request_count = 0
    for group_index, group_row in enumerate(groups):
        if request_count >= max_requests:
            break
        group = group_row["chunks"]
        reuse_key = "rag:hotpotqa:" + "|".join(row["chunk_id"] for row in group)
        turns = []
        for repeat_index in range(max(1, query_repeats)):
            if request_count >= max_requests:
                break
            query = group_row["question"]
            if repeat_index > 0:
                query = f"{query} Answer in a different concise wording. Variant {repeat_index}."
            turns.append({"i": repeat_index, "user": query})
            request_count += 1
        sessions.append(
            {
                "session_id": f"rag_hotpot_{group_index:06d}",
                "source": "hotpotqa",
                "object_type": "rag_chunk_set",
                "reuse_key": reuse_key,
                "dataset": "hotpotqa",
                "hotpotqa_example_id": group_row["example_id"],
                "hotpotqa_source_path": group_row["source_path"],
                "answer": group_row["answer"],
                "chunks": group,
                "turns": turns,
            }
        )
    return sessions


def rag_session_to_turn_rag(session: dict) -> dict:
    return {
        "reuse_key": session["reuse_key"],
        "dataset": session.get("dataset", "hotpotqa"),
        "hotpotqa_example_id": session.get("hotpotqa_example_id", ""),
        "hotpotqa_source_path": session.get("hotpotqa_source_path", ""),
        "chunks": session.get("chunks", []),
        "match_mode": "weak_round_robin",
    }


def attach_weak_rag_to_sharegpt(
    sharegpt_sessions: Sequence[dict],
    rag_sessions: Sequence[dict],
    max_rag_turns: int,
    repeat_each_rag: int = 2,
) -> List[dict]:
    linked = [json.loads(json.dumps(session, ensure_ascii=False)) for session in sharegpt_sessions]
    rag_units = [rag_session_to_turn_rag(session) for session in rag_sessions]
    if not linked or not rag_units or max_rag_turns <= 0:
        return linked
    assignments = 0
    rag_index = 0
    for session in linked:
        for turn in session.get("turns", []):
            if assignments >= max_rag_turns:
                return linked
            turn["rag"] = rag_units[rag_index % len(rag_units)]
            assignments += 1
            if repeat_each_rag <= 1 or assignments % repeat_each_rag == 0:
                rag_index += 1
    return linked


def rag_session_to_prompts(session: dict, tokenizer=None) -> List[dict]:
    chunks = session.get("chunks", [])
    context_lines = [
        f"[{row['chunk_id']}] {row['title']}: {row['text']}"
        for row in chunks
    ]
    context_prefix = "Retrieved context:\n" + "\n".join(context_lines) + "\n\n"
    prompts: List[dict] = []
    for turn in session.get("turns", []):
        turn_index = int(turn.get("i", len(prompts)))
        query = str(turn.get("user", "")).strip()
        if not query:
            continue
        prompt = context_prefix + f"Question: {query}\nAnswer:"
        prompts.append(
            {
                "request_id": f"{session['session_id']}:turn:{turn_index:03d}",
                "session_id": str(session["session_id"]),
                "turn_index": turn_index,
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "prompt_est_tokens": estimate_tokens(prompt),
                "n_tokens": count_tokens(prompt, tokenizer),
                "workload": "rag_chunk_reuse",
                "object_type": str(session.get("object_type", "rag_chunk_set")),
                "reuse_key": str(session["reuse_key"]),
                "chunk_ids": [row["chunk_id"] for row in chunks],
                "doc_ids": sorted({row["doc_id"] for row in chunks}),
                "dataset": session.get("dataset", "hotpotqa"),
                "hotpotqa_example_id": session.get("hotpotqa_example_id", ""),
                "hotpotqa_source_path": session.get("hotpotqa_source_path", ""),
                "query": query,
                "answer": session.get("answer", ""),
                "replay_source": "frozen_replay_trace",
            }
        )
    return prompts


def write_replay_trace(path: Path, sessions: Sequence[dict]) -> None:
    write_jsonl(path, sessions)


def load_replay_sessions(path: Path) -> List[dict]:
    sessions: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            rows = json.load(f)
            sessions.extend(row for row in rows if isinstance(row, dict))
        else:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        sessions.append(row)
    return sessions


def replay_sessions_to_prompts(sessions: Sequence[dict], tokenizer=None, max_requests: int = 0) -> List[dict]:
    prompts: List[dict] = []
    for session in sessions:
        source = str(session.get("source", "sharegpt")).lower()
        if source == "hotpotqa":
            session_prompts = rag_session_to_prompts(session, tokenizer)
        else:
            session_prompts = sharegpt_session_to_prompts(session, tokenizer)
        for item in session_prompts:
            prompts.append(item)
            if max_requests > 0 and len(prompts) >= max_requests:
                return prompts
    return prompts


def build_replay_sessions(args: argparse.Namespace) -> List[dict]:
    rag_requests = max(0, min(args.rag_requests, args.max_requests))
    if args.workload == "sharegpt":
        sharegpt_limit = args.max_sessions
        rag_requests = 0
    elif args.workload == "rag":
        sharegpt_limit = 0
        rag_requests = args.max_requests
    else:
        sharegpt_limit = args.max_sessions

    sharegpt_sessions = (
        load_sharegpt_sessions(
            Path(args.trace_path).expanduser(),
            sharegpt_limit,
            order=getattr(args, "sharegpt_order", "file"),
        )
        if sharegpt_limit > 0
        else []
    )
    rag_sessions = build_rag_sessions(
        rag_requests,
        args.hotpotqa_path,
        args.download_hotpotqa,
        chunk_words_count=args.rag_chunk_words,
        chunks_per_query=args.rag_chunks_per_query,
        query_repeats=args.rag_query_repeats,
        max_examples=args.hotpotqa_max_examples,
        timeout_s=args.timeout_s,
    )
    if args.workload == "sharegpt":
        return sharegpt_sessions
    if args.workload == "rag":
        return rag_sessions
    if getattr(args, "link_mode", "independent") == "weak":
        return attach_weak_rag_to_sharegpt(
            sharegpt_sessions,
            rag_sessions,
            max_rag_turns=rag_requests,
            repeat_each_rag=getattr(args, "weak_rag_repeat", 2),
        )

    mixed: List[dict] = []
    max_len = max(len(sharegpt_sessions), len(rag_sessions))
    for idx in range(max_len):
        if idx < len(sharegpt_sessions):
            mixed.append(sharegpt_sessions[idx])
        if idx < len(rag_sessions):
            mixed.append(rag_sessions[idx])
    return mixed


def mark_sharegpt_prompt(item: dict) -> dict:
    row = dict(item)
    row.setdefault("workload", "sharegpt_session_prefix")
    row.setdefault("object_type", "sharegpt_session_prefix")
    row.setdefault("reuse_key", str(row["session_id"]))
    return row


def interleave_prompts(primary: Sequence[dict], secondary: Sequence[dict], max_requests: int) -> List[dict]:
    mixed: List[dict] = []
    max_len = max(len(primary), len(secondary))
    for idx in range(max_len):
        if idx < len(primary):
            mixed.append(primary[idx])
            if len(mixed) >= max_requests:
                return mixed
        if idx < len(secondary):
            mixed.append(secondary[idx])
            if len(mixed) >= max_requests:
                return mixed
    return mixed[:max_requests]


def load_replay_prompts(args: argparse.Namespace, trace_path: Path, tokenizer=None) -> List[dict]:
    replay_trace = Path(getattr(args, "replay_trace", "") or "").expanduser()
    if str(replay_trace) not in {"", "."} and replay_trace.exists():
        return replay_sessions_to_prompts(
            load_replay_sessions(replay_trace),
            tokenizer,
            max_requests=args.max_requests,
        )

    rag_requests = max(0, min(args.rag_requests, args.max_requests))
    if args.workload == "sharegpt":
        sharegpt_limit = args.max_requests
        rag_requests = 0
    elif args.workload == "rag":
        sharegpt_limit = 0
        rag_requests = args.max_requests
    else:
        sharegpt_limit = max(args.max_requests - rag_requests, 0)

    sharegpt_prompts: List[dict] = []
    if sharegpt_limit > 0:
        sharegpt_prompts = [
            mark_sharegpt_prompt(item)
            for item in load_sharegpt_prompts(trace_path, args.max_sessions, sharegpt_limit, tokenizer)
        ]
    rag_prompts = build_rag_chunk_prompts(
        rag_requests,
        args.hotpotqa_path,
        args.download_hotpotqa,
        tokenizer,
        chunk_words_count=args.rag_chunk_words,
        chunks_per_query=args.rag_chunks_per_query,
        query_repeats=args.rag_query_repeats,
        max_examples=args.hotpotqa_max_examples,
        timeout_s=args.timeout_s,
    )
    if args.workload == "sharegpt":
        return sharegpt_prompts[: args.max_requests]
    if args.workload == "rag":
        return rag_prompts[: args.max_requests]
    return interleave_prompts(sharegpt_prompts, rag_prompts, args.max_requests)


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


def prepare_event_row(
    idx: int,
    item: dict,
    reuse_seen: set[str],
    args: argparse.Namespace,
    model: str,
    kv_mib_per_token: float,
) -> tuple[dict, str, bool]:
    policy_started = time.perf_counter()
    n_tokens = int(item["n_tokens"])
    overlength = n_tokens + args.max_tokens > args.max_model_len
    reuse_key = str(item.get("reuse_key", item.get("session_id", "")))
    workload = str(item.get("workload", "sharegpt_session_prefix"))
    trace_hit = reuse_key in reuse_seen
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
            "hit_source": "trace_side_reuse_key",
            "workload": workload,
            "object_type": item.get("object_type", workload),
            "reuse_key": reuse_key,
            "size_mb": round(size_mb, 6),
            "size_scope": "per_gpu_logical_full_kv_estimate",
            "kv_mib_per_token": round(kv_mib_per_token, 9),
            "t_policy_ms": round(t_policy_ms, 6),
            "model": model,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "temperature": args.temperature,
            "concurrency": args.concurrency,
        }
    )
    rag_reuse_key = str(item.get("rag_reuse_key", ""))
    if rag_reuse_key:
        row["rag_reuse_key"] = rag_reuse_key
        row["rag_hit"] = rag_reuse_key in reuse_seen
        row["rag_hit_source"] = "trace_side_rag_reuse_key"
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
    return row, reuse_key, overlength


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ShareGPT/RAG prompts against vLLM prefix caching server.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="", help="Served model name. Defaults to /v1/models first id.")
    parser.add_argument("--trace-path", default=str(DEFAULT_SHAREGPT_TRACE_PATH))
    parser.add_argument("--replay-trace", default=str(DEFAULT_REPLAY_TRACE_PATH))
    parser.add_argument("--workload", choices=("sharegpt", "rag", "mixed"), default="mixed")
    parser.add_argument("--hotpotqa-path", type=Path, default=DEFAULT_HOTPOTQA_PATH)
    parser.add_argument("--download-hotpotqa", action="store_true")
    parser.add_argument("--hotpotqa-max-examples", type=int, default=10)
    parser.add_argument("--max-sessions", type=int, default=20)
    parser.add_argument("--max-requests", type=int, default=40)
    parser.add_argument("--rag-requests", type=int, default=16)
    parser.add_argument("--rag-chunk-words", type=int, default=56)
    parser.add_argument("--rag-chunks-per-query", type=int, default=2)
    parser.add_argument("--rag-query-repeats", type=int, default=4)
    parser.add_argument("--sharegpt-order", choices=("file", "longest"), default="file")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=1)
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
    prompts = load_replay_prompts(args, trace_path, tokenizer)
    if not prompts:
        raise RuntimeError("no replay prompts were built from trace")

    monitor = GpuMemoryMonitor()
    monitor.start()
    events: List[dict] = []
    reuse_seen: set[str] = set()
    started = time.perf_counter()
    try:
        concurrency = max(1, int(args.concurrency))
        for batch_start in range(0, len(prompts), concurrency):
            batch = prompts[batch_start : batch_start + concurrency]
            prepared: List[tuple[int, dict, dict, str, bool]] = []
            for offset, item in enumerate(batch):
                idx = batch_start + offset
                row, reuse_key, overlength = prepare_event_row(
                    idx, item, reuse_seen, args, model, kv_mib_per_token
                )
                prepared.append((idx, item, row, reuse_key, overlength))

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures: dict[concurrent.futures.Future, tuple[int, dict]] = {}
                for idx, item, row, _reuse_key, overlength in prepared:
                    if overlength:
                        continue
                    future = executor.submit(
                        stream_completion,
                        args.endpoint,
                        model,
                        item["prompt"],
                        args.max_tokens,
                        args.temperature,
                        args.timeout_s,
                    )
                    futures[future] = (idx, row)
                for future, (_idx, row) in futures.items():
                    try:
                        ttft_ms, latency_ms, chunks, text = future.result()
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
                        row.update(
                            {
                                "ok": False,
                                "error": completion_error_message(exc),
                                "ttft_ms": 0.0,
                                "latency_ms": 0.0,
                            }
                        )

            for _idx, _item, row, reuse_key, _overlength in prepared:
                events.append(row)
                reuse_seen.add(reuse_key)
    finally:
        monitor.stop()
    elapsed_s = time.perf_counter() - started

    attempted = [row for row in events if not row.get("skipped")]
    measured = [row for row in attempted[args.warmup_requests :] if row.get("ok")]
    ttfts = [float(row["ttft_ms"]) for row in measured]
    latencies = [float(row["latency_ms"]) for row in measured]
    hit_rows = [row for row in measured if row.get("hit")]
    workload_counts: dict[str, int] = {}
    workload_hit_counts: dict[str, int] = {}
    for row in measured:
        workload = str(row.get("workload", "unknown"))
        workload_counts[workload] = workload_counts.get(workload, 0) + 1
        if row.get("hit"):
            workload_hit_counts[workload] = workload_hit_counts.get(workload, 0) + 1
    workload_hit_rates = {
        workload: round(workload_hit_counts.get(workload, 0) / max(count, 1), 6)
        for workload, count in workload_counts.items()
    }
    summary = {
        "experiment": "H0_vLLM_prefix_cache",
        "endpoint": args.endpoint,
        "model": model,
        "trace_source": str(trace_path),
        "replay_trace": str(args.replay_trace),
        "hotpotqa_source": str(args.hotpotqa_path),
        "workload": args.workload,
        "sessions_requested": args.max_sessions,
        "rag_requests_requested": args.rag_requests,
        "rag_chunk_words": args.rag_chunk_words,
        "rag_chunks_per_query": args.rag_chunks_per_query,
        "rag_query_repeats": args.rag_query_repeats,
        "sharegpt_order": args.sharegpt_order,
        "hotpotqa_max_examples": args.hotpotqa_max_examples,
        "requests_total": len(events),
        "requests_attempted": len(attempted),
        "requests_skipped_overlength": sum(1 for row in events if row.get("skipped")),
        "requests_measured": len(measured),
        "success_rate": round(sum(1 for row in attempted if row.get("ok")) / max(len(attempted), 1), 6),
        "warmup_requests": args.warmup_requests,
        "hit_rate": round(len(hit_rows) / max(len(measured), 1), 6),
        "hit_source": "trace_side_reuse_key",
        "workload_counts": workload_counts,
        "workload_hit_rates": workload_hit_rates,
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
        "concurrency": args.concurrency,
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
