#!/usr/bin/env python3
"""生成带模型 chat-template token 的 ShareGPT token-delta replay trace。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
H0_DIR = REPO_ROOT / "h0"
if str(H0_DIR) not in sys.path:
    sys.path.insert(0, str(H0_DIR))

from run_h0_vllm import DEFAULT_SHAREGPT_TRACE_PATH, lcp_token_count, token_prefix_hash  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

DEFAULT_MODEL = REPO_ROOT / "models" / "Qwen2.5-7B-Instruct"
DEFAULT_OUT = REPO_ROOT / "data" / "edgekv_traces" / "有效实验数据" / "sharedgpt_token_v1.jsonl"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, accurate assistant."


def message_role(message: dict[str, Any]) -> str:
    return str(message.get("from", message.get("role", ""))).lower()


def message_text(message: dict[str, Any]) -> str:
    value = message.get("value", message.get("content", ""))
    return value if isinstance(value, str) else str(value)


def normalize_role(role: str) -> str:
    role = role.lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    return role


def load_sharegpt_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"ShareGPT source must be a JSON array: {path}")
    return [row for row in rows if isinstance(row, dict)]


def conversation_length(row: dict[str, Any]) -> tuple[int, int]:
    messages = list(iter_clean_messages(row))
    user_count = sum(1 for msg in messages if msg["role"] == "user")
    total_chars = sum(len(msg["content"]) for msg in messages)
    return user_count, total_chars


def iter_clean_messages(row: dict[str, Any]) -> Iterable[dict[str, str]]:
    conversations = row.get("conversations", [])
    if not isinstance(conversations, list):
        return
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = normalize_role(message_role(message))
        if role not in {"user", "assistant"}:
            continue
        text = message_text(message).strip()
        if not text:
            continue
        yield {"role": role, "content": text}


def system_fingerprint(system_prompt: str, system_token_ids: Sequence[int]) -> str:
    payload = json.dumps(
        {"system_prompt": system_prompt, "system_token_ids": list(system_token_ids)},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered if isinstance(rendered, str) else str(rendered)


def build_prompt_token_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(token_ids, dict):
        token_ids = token_ids.get("input_ids", [])
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(tok) for tok in token_ids]


def build_session(
    row: dict[str, Any],
    source_index: int,
    tokenizer: Any,
    tokenizer_name: str,
    system_prompt: str,
    system_token_ids: list[int],
    include_debug_fields: bool = False,
) -> dict[str, Any] | None:
    messages = list(iter_clean_messages(row))
    user_count = sum(1 for msg in messages if msg["role"] == "user")
    if user_count < 2:
        return None

    session_id = str(row.get("id", f"sharegpt_{source_index:06d}"))
    history: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    previous_ids: list[int] = []
    turns: list[dict[str, Any]] = []
    user_turn = 0
    for msg in messages:
        if msg["role"] == "user":
            prompt_messages = history + [msg]
            prompt_ids = build_prompt_token_ids(tokenizer, prompt_messages)
            reused = lcp_token_count(previous_ids, prompt_ids) if previous_ids else 0
            turn = {
                "i": user_turn,
                "request_id": f"{session_id}:turn:{user_turn:03d}",
                "prompt_token_ids": prompt_ids,
                "prompt_token_count": len(prompt_ids),
                "reused_prefix_token_count": reused,
                "new_prefill_token_count": len(prompt_ids) - reused,
                "prefix_hash": token_prefix_hash(prompt_ids),
            }
            if include_debug_fields:
                turn["delta_messages"] = [msg]
                turn["prompt"] = build_prompt_text(tokenizer, prompt_messages)
            turns.append(turn)
            previous_ids = prompt_ids
            history.append(msg)
            user_turn += 1
        else:
            history.append(msg)

    if len(turns) < 2:
        return None
    session = {
        "trace_format": "sharegpt_token_delta_v1",
        "system_prefix_token_count": len(system_token_ids),
        "session_group_id": session_id,
        "session_id": session_id,
        "turns": turns,
    }
    if include_debug_fields:
        session.update(
            {
                "tokenizer_name_or_path": tokenizer_name,
                "chat_template_source": "tokenizer.apply_chat_template",
                "system_prompt": system_prompt,
                "system_fingerprint": system_fingerprint(system_prompt, system_token_ids),
                "source": "sharegpt",
                "workload": "sharegpt_session_prefix",
                "object_type": "sharegpt_token_delta",
                "source_index": source_index,
            }
        )
    return session


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sharegpt", type=Path, default=DEFAULT_SHAREGPT_TRACE_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-sessions", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max-requests", type=int, default=1536, help="0 means no limit")
    parser.add_argument("--order", choices=("file", "longest"), default="longest")
    parser.add_argument(
        "--include-debug-fields",
        action="store_true",
        help="Also write human-readable prompt/messages and provenance fields for inspection.",
    )
    args = parser.parse_args()

    tokenizer_name = str(args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(f"tokenizer has no chat_template: {tokenizer_name}")
    system_token_ids = build_prompt_token_ids(
        tokenizer,
        [{"role": "system", "content": args.system_prompt}],
        add_generation_prompt=False,
    )

    sessions: list[dict[str, Any]] = []
    request_count = 0
    rows = load_sharegpt_rows(args.sharegpt.expanduser())
    indexed_rows = list(enumerate(rows))
    if args.order == "longest":
        indexed_rows.sort(key=lambda item: conversation_length(item[1]), reverse=True)
    for source_index, row in indexed_rows:
        session = build_session(
            row,
            source_index,
            tokenizer,
            tokenizer_name,
            args.system_prompt,
            system_token_ids,
            include_debug_fields=args.include_debug_fields,
        )
        if session is None:
            continue
        sessions.append(session)
        request_count += len(session["turns"])
        if args.max_sessions > 0 and len(sessions) >= args.max_sessions:
            break
        if args.max_requests > 0 and request_count >= args.max_requests:
            break

    written = write_jsonl(args.out.expanduser(), sessions)
    summary = {
        "out": str(args.out),
        "sessions": written,
        "requests": sum(len(row["turns"]) for row in sessions),
        "trace_format": "sharegpt_token_delta_v1",
        "tokenizer_name_or_path": tokenizer_name,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
