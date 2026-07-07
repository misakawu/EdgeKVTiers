#!/usr/bin/env python3
"""Replay ShareGPT + RAG reuse prompts against a vLLM server.

This is the real-engine H0 runner: it assumes vLLM is already serving with
``--enable-prefix-caching`` and measures request TTFT, end-to-end latency, and
GPU memory peak while replaying prompts that share prefixes.
"""

from __future__ import annotations
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))   # 强制插入最前面
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List, Sequence

from edgekv_cop import COPProfiler, DEFAULT_BW_GBPS, DEFAULT_C_RE_MS_PER_TOKEN, DEFAULT_D_DESER_MS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAREGPT_TRACE_PATH = (
    REPO_ROOT / "data" / "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
)
DEFAULT_HOTPOTQA_PATH = REPO_ROOT / "data" / "hotpotqa"
DEFAULT_REPLAY_TRACE_PATH = (
    REPO_ROOT / "data" / "edgekv_traces" / "sharegpt_hotpotqa_session.jsonl"
)
# structured_conversation_v2 回放：session 不带 system_prompt 时用此默认系统提示，
# 与生成器 scripts/sharedgpt_structured_trace.py 保持一致，保证渲染确定性。
DEFAULT_SYSTEM_PROMPT = "You are a helpful, accurate assistant."
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


def lcp_token_count(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def token_prefix_hash(token_ids: Sequence[int], prefix_len: int | None = None) -> str:
    limit = len(token_ids) if prefix_len is None else max(0, min(prefix_len, len(token_ids)))
    payload = ",".join(str(int(tok)) for tok in token_ids[:limit]).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


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
        turns: List[dict] = []
        current_turn: dict | None = None
        for msg in conversations:
            role = message_role(msg)
            text = message_text(msg).strip()
            if not text:
                continue
            if role in {"human", "user"}:
                current_turn = {"i": len(turns), "user": text}
                turns.append(current_turn)
            elif role in {"gpt", "assistant"} and current_turn is not None and "assistant" not in current_turn:
                current_turn["assistant"] = text
        if len(turns) < 2:
            continue
        session_id = str(row.get("id", f"sharegpt_{raw_session_idx:06d}"))
        sessions.append(
            {
                "session_id": session_id,
                "source": "sharegpt",
                "object_type": "sharegpt_session_prefix",
                "reuse_key": session_id,
                "source_index": raw_session_idx,
                "total_user_chars": sum(len(str(turn.get("user", ""))) for turn in turns),
                "turns": turns,
            }
        )
        if order == "file" and len(sessions) >= max_sessions:
            break
    if order == "longest":
        sessions.sort(key=lambda item: (int(item.get("total_user_chars", 0)), len(item.get("turns", []))), reverse=True)
    return sessions[:max_sessions]


def build_sharegpt_cumulative_sessions(
    path: Path,
    max_turns: int,
    min_human_turns: int = 2,
    order: str = "file",
    max_prompt_est_tokens: int = 0,
) -> tuple[List[dict], dict]:
    """Build one JSONL row per ShareGPT session with cumulative prompts.

    ``max_turns`` caps the expanded request count, not the number of sessions.
    The last emitted session may be truncated to respect that cap.
    """
    if max_turns <= 0:
        raise ValueError("max_turns must be > 0")
    if min_human_turns <= 0:
        raise ValueError("min_human_turns must be > 0")
    if max_prompt_est_tokens < 0:
        raise ValueError("max_prompt_est_tokens must be >= 0")
    with path.open("r", encoding="utf-8") as f:
        raw_rows = json.load(f)
    if not isinstance(raw_rows, list):
        raise ValueError(f"ShareGPT input must be a JSON list: {path}")

    candidates: List[dict] = []
    skipped_non_list_conversations = 0
    skipped_min_human_turns = 0
    skipped_first_role = 0
    for raw_session_idx, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            skipped_non_list_conversations += 1
            continue
        conversations = raw.get("conversations", [])
        if not isinstance(conversations, list):
            skipped_non_list_conversations += 1
            continue

        first_role = ""
        turns: List[dict] = []
        current_turn: dict | None = None
        for msg in conversations:
            if not isinstance(msg, dict):
                continue
            role = message_role(msg)
            text = message_text(msg).strip()
            if not text:
                continue
            if not first_role:
                first_role = role
            if role in {"human", "user"}:
                current_turn = {"i": len(turns), "user": text}
                turns.append(current_turn)
            elif role in {"gpt", "assistant"} and current_turn is not None and "assistant" not in current_turn:
                current_turn["assistant"] = text

        if first_role not in {"human", "user"}:
            skipped_first_role += 1
            continue
        if len(turns) < min_human_turns:
            skipped_min_human_turns += 1
            continue
        session_id = str(raw.get("id", f"sharegpt_{raw_session_idx:06d}"))
        candidates.append(
            {
                "session_id": session_id,
                "source": "sharegpt",
                "object_type": "sharegpt_session_prefix",
                "reuse_key": session_id,
                "source_index": raw_session_idx,
                "total_user_chars": sum(len(str(turn.get("user", ""))) for turn in turns),
                "first_nonempty_role": first_role,
                "raw_turns": turns,
            }
        )

    if order == "longest":
        candidates.sort(
            key=lambda item: (
                int(item.get("total_user_chars", 0)),
                len(item.get("raw_turns", [])),
            ),
            reverse=True,
        )
    elif order != "file":
        raise ValueError(f"unsupported ShareGPT order: {order}")

    sessions: List[dict] = []
    written_turns = 0
    prompt_token_estimates: List[int] = []
    for candidate in candidates:
        if written_turns >= max_turns:
            break
        if max_turns - written_turns < min_human_turns:
            break
        transcript: List[str] = []
        cumulative_turns: List[dict] = []
        for turn in candidate["raw_turns"]:
            if written_turns + len(cumulative_turns) >= max_turns:
                break
            user_text = str(turn.get("user", "")).strip()
            if not user_text:
                continue
            prompt = "\n".join(transcript + [f"User: {user_text}", "Assistant:"])
            prompt_est_tokens = estimate_tokens(prompt)
            if max_prompt_est_tokens and prompt_est_tokens > max_prompt_est_tokens:
                break
            prompt_token_estimates.append(prompt_est_tokens)
            next_turn = dict(turn)
            next_turn["user"] = prompt
            next_turn["i"] = len(cumulative_turns)
            cumulative_turns.append(next_turn)
            transcript.append(f"User: {user_text}")
            assistant_text = str(turn.get("assistant", "")).strip()
            if assistant_text:
                transcript.append(f"Assistant: {assistant_text}")
        if len(cumulative_turns) >= min_human_turns:
            written_turns += len(cumulative_turns)
            sessions.append(
                {
                    "session_id": str(candidate["session_id"]),
                    "source": "sharegpt",
                    "workload": "sharegpt_session_prefix",
                    "object_type": "sharegpt_session_prefix",
                    "reuse_key": str(candidate["reuse_key"]),
                    "turns_format": "cumulative_user",
                    "source_index": int(candidate["source_index"]),
                    "total_user_chars": int(candidate["total_user_chars"]),
                    "first_nonempty_role": str(candidate["first_nonempty_role"]),
                    "turns": cumulative_turns,
                }
            )

    summary = {
        "sharegpt_path": str(path),
        "source": "sharegpt",
        "prefix_layout": "original_conversation_cumulative",
        "turns_format": "cumulative_user",
        "sessions": len(sessions),
        "written_requests": written_turns,
        "max_turns": max_turns,
        "min_human_turns": min_human_turns,
        "max_prompt_est_tokens": max_prompt_est_tokens,
        "order": order,
        "scanned_sessions": len(raw_rows),
        "eligible_sessions": len(candidates),
        "skipped_sessions": {
            "non_list_or_missing_conversations": skipped_non_list_conversations,
            "first_nonempty_role_not_human_or_user": skipped_first_role,
            "fewer_than_min_human_turns": skipped_min_human_turns,
        },
        "prompt_content": "cumulative_user_assistant_history",
        "assistant_text_preserved": True,
        "history_prefix_preserved": True,
        "prompt_est_tokens_avg": (
            sum(prompt_token_estimates) / len(prompt_token_estimates)
            if prompt_token_estimates else 0.0
        ),
        "prompt_est_tokens_min": min(prompt_token_estimates) if prompt_token_estimates else 0,
        "prompt_est_tokens_max": max(prompt_token_estimates) if prompt_token_estimates else 0,
    }
    return sessions, summary


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
        for key, value in session.items():
            if key not in row and key != "turns":
                row[key] = value
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


def hotpotqa_high_frequency_group_order(group_count: int, request_count: int) -> List[int]:
    """Return deterministic 80/20 group indices for HotpotQA chunk reuse.

    The first 20% of groups are treated as hot and occupy four out of every
    five request slots. The remaining groups are still exercised as the cold
    tail so coverage is preserved while reuse is intentionally skewed.
    """
    if group_count <= 0 or request_count <= 0:
        return []
    hot_count = max(1, (group_count + 4) // 5)
    hot_count = min(hot_count, group_count)
    hot_indices = list(range(hot_count))
    tail_indices = list(range(hot_count, group_count))
    order: List[int] = []
    hot_pos = 0
    tail_pos = 0
    for slot in range(request_count):
        use_tail = bool(tail_indices) and (slot + 1) % 5 == 0
        if use_tail:
            order.append(tail_indices[tail_pos % len(tail_indices)])
            tail_pos += 1
        else:
            order.append(hot_indices[hot_pos % len(hot_indices)])
            hot_pos += 1
    return order


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
    """Build a deterministic HotpotQA RAG trace with high-frequency chunk reuse.

    The retrieved chunk prefix stays at the beginning of each prompt, while the
    group schedule makes the first 20% of chunk groups account for about 80% of
    RAG requests. This gives LFU/LPE a stable hot/cold signal instead of the old
    near-uniform sequential reuse pattern.
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
    group_order = hotpotqa_high_frequency_group_order(len(groups), max_requests)
    group_access_counts = [0 for _ in groups]
    for request_index, group_index in enumerate(group_order):
        group_row = groups[group_index]
        group = group_row["chunks"]
        reuse_key = "rag:hotpotqa:" + "|".join(row["chunk_id"] for row in group)
        context_lines = []
        for row in group:
            context_lines.append(f"[{row['chunk_id']}] {row['title']}: {row['text']}")
        context_prefix = "Retrieved context:\n" + "\n".join(context_lines) + "\n\n"
        repeat_index = group_access_counts[group_index] % max(1, query_repeats)
        group_access_counts[group_index] += 1
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
    group_order = hotpotqa_high_frequency_group_order(len(groups), max_requests)
    group_access_counts = [0 for _ in groups]
    sessions: List[dict] = []
    session_seq = 0
    request_count = 0
    current_group_index: int | None = None
    current_session: dict | None = None

    def start_session(group_index: int, group_row: dict) -> dict:
        nonlocal session_seq
        group = group_row["chunks"]
        reuse_key = "rag:hotpotqa:" + "|".join(row["chunk_id"] for row in group)
        session = {
            "session_id": f"rag_hotpot_{session_seq:06d}_group_{group_index:06d}",
            "source": "hotpotqa",
            "object_type": "rag_chunk_set",
            "reuse_key": reuse_key,
            "dataset": "hotpotqa",
            "hotpotqa_example_id": group_row["example_id"],
            "hotpotqa_source_path": group_row["source_path"],
            "answer": group_row["answer"],
            "chunks": group,
            "turns": [],
        }
        session_seq += 1
        sessions.append(session)
        return session

    for group_index in group_order:
        group_row = groups[group_index]
        if (
            current_session is None
            or current_group_index != group_index
            or len(current_session.get("turns", [])) >= max(1, query_repeats)
        ):
            current_group_index = group_index
            current_session = start_session(group_index, group_row)
        repeat_index = group_access_counts[group_index] % max(1, query_repeats)
        query = group_row["question"]
        if repeat_index > 0:
            query = f"{query} Answer in a different concise wording. Variant {repeat_index}."
        current_session["turns"].append({"i": repeat_index, "user": query})
        group_access_counts[group_index] += 1
        request_count += 1
        if request_count >= max_requests:
            break
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
        row = {
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
        for key, value in session.items():
            if key not in row and key not in {"turns", "chunks"}:
                row[key] = value
        prompts.append(row)
    return prompts


def session_to_cumulative_user_session(session: dict) -> dict:
    row = dict(session)
    turns = session.get("turns", [])
    if not isinstance(turns, list) or row.get("turns_format") in {"cumulative_user", "complete_prompt"}:
        return row
    if str(row.get("source", "sharegpt")).lower() == "hotpotqa":
        return row

    transcript: List[str] = []
    cumulative_turns: List[dict] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        user_text = str(turn.get("user", "")).strip()
        if not user_text:
            continue
        next_turn = dict(turn)
        next_turn["user"] = "\n".join(transcript + [f"User: {user_text}", "Assistant:"])
        cumulative_turns.append(next_turn)
        transcript.append(f"User: {user_text}")
        assistant_text = str(turn.get("assistant", "")).strip()
        if assistant_text:
            transcript.append(f"Assistant: {assistant_text}")
    row["turns"] = cumulative_turns
    row["turns_format"] = "cumulative_user"
    return row


def write_replay_trace(path: Path, sessions: Sequence[dict]) -> None:
    write_jsonl(path, [session_to_cumulative_user_session(session) for session in sessions])


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


def is_flat_replay_prompt(row: dict) -> bool:
    return "prompt" in row and "turns" not in row


def flat_replay_prompt_to_session(row: dict, fallback_index: int) -> dict:
    session_id = str(row.get("session_id", f"flat_replay_{fallback_index:06d}"))
    turn_index = int(row.get("turn_index", row.get("i", 0)) or 0)
    user_text = str(row.get("prompt", "")).strip()
    session = {k: v for k, v in row.items() if k not in {"prompt", "turns", "turn_index"}}
    session.setdefault("session_id", session_id)
    session.setdefault("source", str(row.get("source", "flat_replay")))
    session.setdefault("object_type", str(row.get("object_type", "flat_prompt")))
    session.setdefault("reuse_key", str(row.get("reuse_key", session_id)))
    session["turns"] = [{"i": turn_index, "user": user_text, "prompt_is_complete": True}]
    return session


def complete_prompt_session_to_prompts(session: dict, tokenizer=None) -> List[dict]:
    prompts: List[dict] = []
    session_id = str(session["session_id"])
    for turn in session.get("turns", []):
        turn_index = int(turn.get("i", len(prompts)))
        prompt = str(turn.get("user", "")).strip()
        if not prompt:
            continue
        row = {
            "request_id": str(turn.get("request_id", f"{session_id}:turn:{turn_index:03d}")),
            "session_id": session_id,
            "turn_index": turn_index,
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "prompt_est_tokens": estimate_tokens(prompt),
            "n_tokens": count_tokens(prompt, tokenizer),
            "workload": str(session.get("workload", "sharegpt_session_prefix")),
            "object_type": str(session.get("object_type", "flat_prompt")),
            "reuse_key": str(session.get("reuse_key", session_id)),
            "replay_source": "frozen_replay_trace",
        }
        for key, value in session.items():
            if key not in row and key != "turns":
                row[key] = value
        prompts.append(row)
    return prompts


def token_delta_session_to_prompts(session: dict, tokenizer=None) -> List[dict]:
    session_id = str(session.get("session_id", session.get("session_group_id", ""))).strip()
    if not session_id:
        raise ValueError("token delta session missing session_id/session_group_id")
    if session.get("trace_format") != "sharegpt_token_delta_v1":
        raise ValueError(f"unsupported token trace format: {session.get('trace_format')!r}")
    turns = session.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"token delta session {session_id!r} must have nonempty turns list")

    prompts: List[dict] = []
    previous_ids: List[int] = []
    previous_turn_index = -1
    for offset, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"token delta session {session_id!r} has non-object turn")
        try:
            turn_index = int(turn.get("i", turn.get("turn_index", offset)))
        except (TypeError, ValueError):
            raise ValueError(f"token delta session {session_id!r} turn has invalid index")
        if turn_index <= previous_turn_index:
            raise ValueError(f"token delta session {session_id!r} turn_index must be strictly increasing")
        previous_turn_index = turn_index

        raw_ids = turn.get("prompt_token_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError(f"token delta session {session_id!r} turn {turn_index} has empty prompt_token_ids")
        try:
            token_ids = [int(tok) for tok in raw_ids]
        except (TypeError, ValueError):
            raise ValueError(f"token delta session {session_id!r} turn {turn_index} has non-integer token id")
        declared_count = int(turn.get("prompt_token_count", len(token_ids)) or 0)
        if declared_count != len(token_ids):
            raise ValueError(
                f"token delta session {session_id!r} turn {turn_index} prompt_token_count mismatch"
            )
        actual_lcp = lcp_token_count(previous_ids, token_ids) if previous_ids else 0
        declared_reused = int(turn.get("reused_prefix_token_count", actual_lcp) or 0)
        if declared_reused != actual_lcp:
            raise ValueError(
                f"token delta session {session_id!r} turn {turn_index} reused_prefix_token_count mismatch"
            )
        declared_new = int(turn.get("new_prefill_token_count", len(token_ids) - actual_lcp) or 0)
        if declared_new != len(token_ids) - actual_lcp:
            raise ValueError(
                f"token delta session {session_id!r} turn {turn_index} new_prefill_token_count mismatch"
            )

        row = {
            "request_id": str(turn.get("request_id", f"{session_id}:turn:{turn_index:03d}")),
            "session_id": session_id,
            "session_group_id": str(session.get("session_group_id", session_id)),
            "turn_index": turn_index,
            "prompt": str(turn.get("prompt", "")),
            "prompt_chars": len(str(turn.get("prompt", ""))),
            "prompt_est_tokens": len(token_ids),
            "n_tokens": len(token_ids),
            "prompt_token_ids": token_ids,
            "prompt_token_count": len(token_ids),
            "reused_prefix_token_count": actual_lcp,
            "new_prefill_token_count": len(token_ids) - actual_lcp,
            "token_prefix_hash": str(turn.get("prefix_hash", token_prefix_hash(token_ids))),
            "prefix_hash": str(turn.get("prefix_hash", token_prefix_hash(token_ids))),
            "system_prefix_token_count": int(session.get("system_prefix_token_count", 0) or 0),
            "workload": str(session.get("workload", "sharegpt_session_prefix")),
            "object_type": str(session.get("object_type", "sharegpt_token_delta")),
            "replay_source": "frozen_replay_trace",
            "history_format": "token_delta_chatml",
            "history_turns": turn_index + 1,
            "history_prompt_chars": len(str(turn.get("prompt", ""))),
            "replay_trace_format": "sharegpt_token_delta_v1",
            "tokenizer_name_or_path": str(session.get("tokenizer_name_or_path", "")),
            "chat_template_source": str(session.get("chat_template_source", "")),
            "system_fingerprint": str(session.get("system_fingerprint", "")),
        }
        if "delta_messages" in turn:
            row["delta_messages"] = turn["delta_messages"]
        for key, value in session.items():
            if key not in row and key != "turns":
                row[key] = value
        prompts.append(row)
        previous_ids = token_ids
    return prompts


def cumulative_user_session_to_prompts(session: dict, tokenizer=None) -> List[dict]:
    prompts: List[dict] = []
    session_id = str(session["session_id"])
    turns = session.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"cumulative_user session {session_id!r} must have nonempty turns list")
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError(f"cumulative_user session {session_id!r} has non-object turn")
        try:
            turn_index = int(turn.get("i"))
        except (TypeError, ValueError):
            raise ValueError(f"cumulative_user session {session_id!r} turn has invalid i")
        user_text = str(turn.get("user", "")).strip()
        if not user_text:
            raise ValueError(f"cumulative_user session {session_id!r} turn {turn_index} has empty user")
        rag = turn.get("rag")
        rag_prefix = ""
        if isinstance(rag, dict):
            context_lines = [
                f"[{row['chunk_id']}] {row['title']}: {row['text']}"
                for row in rag.get("chunks", [])
            ]
            if context_lines:
                rag_prefix = "Retrieved context:\n" + "\n".join(context_lines) + "\n\n"
        prompt = rag_prefix + user_text
        row = {
            "request_id": str(turn.get("request_id", f"{session_id}:turn:{turn_index:03d}")),
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
            "history_format": "cumulative_user",
            "history_turns": turn_index + 1,
            "history_prompt_chars": len(prompt),
            "replay_trace_format": "session_cumulative_user",
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
        for key, value in session.items():
            if key not in row and key != "turns":
                row[key] = value
        prompts.append(row)
    return prompts


def load_structured_conversation(session: dict, tokenizer=None) -> List[dict]:
    """structured_conversation_v2：动态累积历史 → apply_chat_template 渲染纯文本 Prompt。

    Trace 只存原始多轮消息（role/content/turn_index），渲染/分词职责全在此处。
    不产出 prompt_token_ids/prefix_hash：下游 run_cell 缺 prompt_token_ids 时会自动
    改喂 str(item['prompt'])，vLLM 内置前缀缓存对确定性渲染的纯文本同样命中。
    命中/复用统计一律走回放器真实执行，不再有 trace-side 模拟。
    """
    session_id = str(session.get("session_id", session.get("session_group_id", ""))).strip()
    if not session_id:
        raise ValueError("structured_conversation_v2 session missing session_id")
    messages = session.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"structured_conversation_v2 session {session_id!r} must have nonempty messages")
    if tokenizer is None or not getattr(tokenizer, "apply_chat_template", None):
        raise ValueError(
            f"structured_conversation_v2 session {session_id!r} requires a tokenizer with a chat template"
        )

    system_prompt = session.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        system_prompt = DEFAULT_SYSTEM_PROMPT
    history: List[dict] = [{"role": "system", "content": system_prompt}]
    temperature = str(session.get("temperature", ""))
    workload = str(session.get("workload", "sharegpt_session_prefix"))
    object_type = str(session.get("object_type", "sharegpt_session_prefix"))
    reuse_key = str(session.get("reuse_key", session_id))

    prompts: List[dict] = []
    user_turn = 0
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError(f"structured_conversation_v2 session {session_id!r} has non-object message")
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", ""))
        if role == "user":
            rendered = tokenizer.apply_chat_template(
                history + [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = rendered if isinstance(rendered, str) else str(rendered)
            row = {
                "request_id": f"{session_id}:turn:{user_turn:03d}",
                "session_id": session_id,
                "turn_index": user_turn,
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "prompt_est_tokens": estimate_tokens(prompt),
                # n_tokens 为 max_model_len 过滤所必需（无 prompt_token_ids 时是唯一长度依据）。
                "n_tokens": count_tokens(prompt, tokenizer),
                "reuse_key": reuse_key,
                "workload": workload,
                "object_type": object_type,
                "history_format": "structured_conversation_v2",
                "history_turns": user_turn + 1,
                "history_prompt_chars": len(prompt),
                "replay_trace_format": "structured_conversation_v2",
                "replay_source": "frozen_replay_trace",
                "temperature": temperature,
            }
            for key, value in session.items():
                if key not in row and key not in {"messages", "turns"}:
                    row[key] = value
            prompts.append(row)
            history.append({"role": "user", "content": content})
            user_turn += 1
        else:
            # assistant（及未来其它 role，如注入的检索上下文）只累积历史，不产出请求。
            history.append({"role": role, "content": content})
    if not prompts:
        raise ValueError(f"structured_conversation_v2 session {session_id!r} produced no user turns")
    return prompts


def replay_sessions_to_prompts(sessions: Sequence[dict], tokenizer=None, max_requests: int = 0) -> List[dict]:
    prompts: List[dict] = []
    for session_index, session in enumerate(sessions):
        if is_flat_replay_prompt(session):
            session = flat_replay_prompt_to_session(session, session_index)
        source = str(session.get("source", "sharegpt")).lower()
        turns_format = str(session.get("turns_format", ""))
        trace_format = str(session.get("trace_format", ""))
        if trace_format == "structured_conversation_v2":
            session_prompts = load_structured_conversation(session, tokenizer)
        elif trace_format == "sharegpt_token_delta_v1":
            session_prompts = token_delta_session_to_prompts(session, tokenizer)
        elif turns_format == "cumulative_user":
            session_prompts = cumulative_user_session_to_prompts(session, tokenizer)
        elif any(turn.get("prompt_is_complete") for turn in session.get("turns", [])):
            session_prompts = complete_prompt_session_to_prompts(session, tokenizer)
        elif source == "hotpotqa":
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
    cop: COPProfiler,
) -> tuple[dict, str, bool]:
    policy_started = time.perf_counter()
    n_tokens = int(item["n_tokens"])
    overlength = n_tokens + args.max_tokens > args.max_model_len
    reuse_key = str(item.get("reuse_key", item.get("session_id", "")))
    workload = str(item.get("workload", "sharegpt_session_prefix"))
    trace_hit = reuse_key in reuse_seen
    rag_reuse_key = str(item.get("rag_reuse_key", ""))
    rag_hit = bool(rag_reuse_key and rag_reuse_key in reuse_seen)
    event_name = "hit" if trace_hit or rag_hit else "miss"
    profile = cop.update_from_item(item, hit=bool(trace_hit or rag_hit), access_index=idx)
    size_mb = profile.size_mb
    t_policy_ms = (time.perf_counter() - policy_started) * 1000.0
    row = dict(item)
    row.pop("prompt", None)
    row.update(
        {
            "event_index": idx,
            "event": event_name,
            "hit": bool(trace_hit or rag_hit),
            "hit_source": "trace_side_rag_reuse_key" if rag_hit and not trace_hit else "trace_side_reuse_key",
            "workload": workload,
            "object_type": profile.object_type,
            "object_id": profile.object_id,
            "reuse_key": reuse_key,
            "size_mb": round(size_mb, 6),
            "size_scope": "per_gpu_logical_full_kv_estimate",
            "kv_mib_per_token": round(kv_mib_per_token, 9),
            "p_reuse": round(profile.p_reuse, 6),
            "c_recomp_ms": round(profile.c_recomp_ms, 6),
            "c_restore_ms": round(profile.c_restore_ms, 6),
            "risk_exp": round(profile.risk_exp, 6),
            "score": round(profile.score, 9),
            "score_source": "object_level_cop",
            "t_policy_ms": round(t_policy_ms, 6),
            "model": model,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "temperature": args.temperature,
            "concurrency": args.concurrency,
        }
    )
    if rag_reuse_key:
        row["rag_reuse_key"] = rag_reuse_key
        row["rag_hit"] = rag_hit
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
    parser.add_argument("--c-re-ms-per-token", type=float, default=float(os.environ.get("EDGEKV_C_RE_MS_PER_TOKEN", DEFAULT_C_RE_MS_PER_TOKEN)))
    parser.add_argument("--bw-gbps", type=float, default=float(os.environ.get("EDGEKV_BW_GBPS", DEFAULT_BW_GBPS)))
    parser.add_argument("--d-deser-ms", type=float, default=float(os.environ.get("EDGEKV_D_DESER_MS", DEFAULT_D_DESER_MS)))
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
    cop = COPProfiler(
        mu_kv_mb_per_token=kv_mib_per_token,
        c_re_ms_per_token=args.c_re_ms_per_token,
        bw_gbps=args.bw_gbps,
        d_deser_ms=args.d_deser_ms,
    )
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
                    idx, item, reuse_seen, args, model, kv_mib_per_token, cop
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
        "c_re_ms_per_token": round(args.c_re_ms_per_token, 9),
        "bw_gbps": round(args.bw_gbps, 9),
        "d_deser_ms": round(args.d_deser_ms, 9),
        **cop.summary(),
        "tokenizer_source": "transformers_auto" if tokenizer is not None else "whitespace_estimate",
    }

    write_jsonl(out_dir / "events.jsonl", events)
    write_csv(out_dir / "summary.csv", [summary])
    write_json(out_dir / "config.resolved.json", {"args": vars(args), "summary": summary})
    write_jsonl(out_dir / "gpu_memory_samples.jsonl", monitor.samples)
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
