#!/usr/bin/env python3
"""将 ShareGPT sessions 提取为累计上下文 replay trace。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
H0_DIR = REPO_ROOT / "h0"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(H0_DIR) not in sys.path:
    sys.path.insert(0, str(H0_DIR))

from run_h0_vllm import (  # noqa: E402
    DEFAULT_SHAREGPT_TRACE_PATH,
    build_sharegpt_cumulative_sessions,
    write_jsonl,
)


DEFAULT_OUT = REPO_ROOT / "data" / "edgekv_traces" / "sharedgpt_ordered.jsonl"
DEFAULT_MAX_TURNS = 256
DEFAULT_MIN_HUMAN_TURNS = 2


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def build_ordered_rows(
    sharegpt_path: Path,
    max_turns: int,
    min_human_turns: int,
    max_prompt_est_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, summary = build_sharegpt_cumulative_sessions(
        sharegpt_path,
        max_turns=max_turns,
        min_human_turns=min_human_turns,
        order="file",
        max_prompt_est_tokens=max_prompt_est_tokens,
    )
    summary["filter_rules"] = {
        "first_nonempty_role_in": ["human", "user"],
        "min_human_turns": min_human_turns,
        "prompt_content": "cumulative_user_assistant_history",
        "order": "original_file_order_then_dialogue_turn_order",
        "max_turns_caps_expanded_requests": True,
    }
    return rows, summary


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ShareGPT sessions as cumulative-context JSONL in original file order."
    )
    parser.add_argument(
        "--sharegpt-path",
        type=Path,
        default=DEFAULT_SHAREGPT_TRACE_PATH,
        help=f"ShareGPT JSON input path. Default: {DEFAULT_SHAREGPT_TRACE_PATH}",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output JSONL path. Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--max-turns",
        type=positive_int,
        default=DEFAULT_MAX_TURNS,
        help=f"Maximum expanded human/user requests. Default: {DEFAULT_MAX_TURNS}",
    )
    parser.add_argument(
        "--min-human-turns",
        type=positive_int,
        default=DEFAULT_MIN_HUMAN_TURNS,
        help=(
            "Minimum nonempty human/user turns required for a source session. "
            f"Default: {DEFAULT_MIN_HUMAN_TURNS}"
        ),
    )
    parser.add_argument(
        "--max-prompt-est-tokens",
        type=int,
        default=0,
        help="Drop the rest of a session once a cumulative prompt exceeds this estimate; 0 disables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = build_ordered_rows(
        args.sharegpt_path,
        max_turns=args.max_turns,
        min_human_turns=args.min_human_turns,
        max_prompt_est_tokens=args.max_prompt_est_tokens,
    )
    summary["out_path"] = str(args.out)
    summary_path = Path(str(args.out) + ".summary.json")

    write_jsonl(args.out, rows)
    write_summary(summary_path, summary)

    print(
        f"wrote {summary['written_requests']} requests in {len(rows)} sessions from {summary['eligible_sessions']} "
        f"eligible sessions to {args.out}"
    )
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
