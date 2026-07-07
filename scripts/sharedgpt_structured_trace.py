#!/usr/bin/env python3
"""Generate a structured_conversation_v2 ShareGPT replay trace.

与旧的 ``sharedgpt_token_trace.py`` 不同，这里 **只记录"发生了什么"**：原始
ShareGPT 多轮消息（role/content/turn_index），不含任何 ``prompt_token_ids`` /
``prefix_hash`` / ``reused_prefix_token_count``。累积历史拼装、apply_chat_template
渲染、分词与前缀复用统计全部移交给回放器（h0 loader + h1 replayer），因此本脚本
**不加载 tokenizer、不 apply_chat_template**，trace 与模型/模板彻底解耦。

``messages`` 结构天然为 RAG 预留扩展位：未来可在其中注入检索上下文消息
（如 ``{"role": "context", ...}``），无需改动格式。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
H0_DIR = REPO_ROOT / "h0"
if str(H0_DIR) not in sys.path:
    sys.path.insert(0, str(H0_DIR))

from run_h0_vllm import DEFAULT_SHAREGPT_TRACE_PATH  # noqa: E402

DEFAULT_OUT = (
    REPO_ROOT / "data" / "edgekv_traces" / "source_ablation" / "sharegpt_structured_v2.jsonl"
)
DEFAULT_SYSTEM_PROMPT = "You are a helpful, accurate assistant."
TRACE_FORMAT = "structured_conversation_v2"

# 长度分桶阈值（user 轮数）：把会话粗分为 hot/warm，作为下游 p_reuse_prior 的钩子。
# 纯 ShareGPT 下无真实 hot/cold 标注，这里用"多轮=更可能被复用"的朴素先验。
HOT_USER_TURN_THRESHOLD = 4


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


def conversation_length(row: dict[str, Any]) -> tuple[int, int]:
    messages = list(iter_clean_messages(row))
    user_count = sum(1 for msg in messages if msg["role"] == "user")
    total_chars = sum(len(msg["content"]) for msg in messages)
    return user_count, total_chars


def temperature_for(user_count: int) -> str:
    return "hot" if user_count >= HOT_USER_TURN_THRESHOLD else "warm"


def build_session(row: dict[str, Any], source_index: int) -> dict[str, Any] | None:
    """把一行 ShareGPT 会话转成 structured_conversation_v2 记录。"""
    raw_messages = list(iter_clean_messages(row))
    user_count = sum(1 for msg in raw_messages if msg["role"] == "user")
    if user_count < 2:
        return None

    session_id = str(row.get("id", f"sharegpt_{source_index:06d}"))
    # 按 user 轮次标注 turn_index：同一轮的 user 与其后紧邻的 assistant 共享 index。
    messages: list[dict[str, Any]] = []
    turn_index = -1
    for msg in raw_messages:
        if msg["role"] == "user":
            turn_index += 1
        messages.append(
            {
                "role": msg["role"],
                "content": msg["content"],
                "turn_index": max(turn_index, 0),
            }
        )

    if sum(1 for msg in messages if msg["role"] == "user") < 2:
        return None

    return {
        "session_id": session_id,
        "trace_format": TRACE_FORMAT,
        "source": "sharegpt",
        "system_prompt": None,
        "messages": messages,
        "reuse_key": session_id,
        "temperature": temperature_for(user_count),
        "workload": "sharegpt_session_prefix",
        "object_type": "sharegpt_session_prefix",
        "source_index": source_index,
    }


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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--system-prompt", default=None,
                        help="可选：写入每个 session 的 system_prompt；缺省为 null，由回放器用其默认值。")
    parser.add_argument("--max-sessions", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max-requests", type=int, default=1536, help="0 means no limit")
    parser.add_argument("--order", choices=("file", "longest"), default="longest")
    args = parser.parse_args()

    sessions: list[dict[str, Any]] = []
    request_count = 0
    rows = load_sharegpt_rows(args.sharegpt.expanduser())
    indexed_rows = list(enumerate(rows))
    if args.order == "longest":
        indexed_rows.sort(key=lambda item: conversation_length(item[1]), reverse=True)
    for source_index, row in indexed_rows:
        session = build_session(row, source_index)
        if session is None:
            continue
        if args.system_prompt is not None:
            session["system_prompt"] = args.system_prompt
        sessions.append(session)
        request_count += sum(1 for msg in session["messages"] if msg["role"] == "user")
        if args.max_sessions > 0 and len(sessions) >= args.max_sessions:
            break
        if args.max_requests > 0 and request_count >= args.max_requests:
            break

    written = write_jsonl(args.out.expanduser(), sessions)
    summary = {
        "out": str(args.out),
        "sessions": written,
        "requests": sum(
            sum(1 for msg in row["messages"] if msg["role"] == "user") for row in sessions
        ),
        "trace_format": TRACE_FORMAT,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
