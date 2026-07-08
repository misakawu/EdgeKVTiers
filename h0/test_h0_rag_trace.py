#!/usr/bin/env python3
# 测试数据契约：本仓库测试必须使用 JSONL replay trace 文件作为 workload 数据，不使用 vLLM 内置数据集/测试数据。
"""H0 RAG chunk 复用 trace 构造的冒烟测试。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from run_h0_vllm import (
    build_rag_chunk_prompts,
    build_replay_sessions,
    hotpot_context_items,
    hotpotqa_high_frequency_group_order,
    load_replay_prompts,
    replay_sessions_to_prompts,
    resolve_hotpotqa_files,
    write_replay_trace,
)


def write_hotpotqa_fixture(path: Path) -> None:
    rows = [
        {
            "_id": "hp_0",
            "question": "Which city hosted the described festival?",
            "answer": "Springfield",
            "context": [
                [
                    "Festival",
                    [
                        "The annual robotics festival was hosted in Springfield.",
                        "Teams presented edge serving demos and cache experiments.",
                    ],
                ],
                [
                    "Springfield",
                    [
                        "Springfield is a city used in this test fixture.",
                        "It appears in several questions to create reusable chunks.",
                    ],
                ],
            ],
        },
        {
            "_id": "hp_1",
            "question": "What system component was measured?",
            "answer": "KV cache",
            "context": [
                [
                    "Serving System",
                    [
                        "The serving system measured KV cache hit rate and TTFT.",
                        "The same retrieved context can be reused by multiple questions.",
                    ],
                ],
                [
                    "Metrics",
                    [
                        "Latency, memory peak, and policy overhead are logged.",
                        "These fields are useful for H0 trace validation.",
                    ],
                ],
            ],
        },
    ]
    path.write_text(__import__("json").dumps(rows), encoding="utf-8")


def test_rag_chunk_trace_reuses_chunk_sets() -> None:
    path = Path("/tmp/h0_hotpotqa_fixture.json")
    write_hotpotqa_fixture(path)
    rows = build_rag_chunk_prompts(
        max_requests=12,
        hotpotqa_path=path,
        download_hotpotqa=False,
        chunks_per_query=2,
    )
    assert len(rows) == 12
    assert {row["workload"] for row in rows} == {"rag_chunk_reuse"}
    assert {row["dataset"] for row in rows} == {"hotpotqa"}
    counts = Counter(row["reuse_key"] for row in rows)
    assert max(counts.values()) > 1
    assert sum(count - 1 for count in counts.values()) >= len(rows) // 2
    assert all(key.startswith("rag:hotpotqa:") for key in counts)
    assert all(row["chunk_ids"] for row in rows)


def test_hotpotqa_high_frequency_group_order_is_80_20() -> None:
    order = hotpotqa_high_frequency_group_order(group_count=10, request_count=100)
    counts = Counter(order)
    hot_requests = sum(counts[idx] for idx in range(2))
    cold_requests = sum(counts[idx] for idx in range(2, 10))
    assert len(order) == 100
    assert hot_requests == 80
    assert cold_requests == 20
    assert set(counts) == set(range(10))



def test_hotpotqa_directory_prefers_validation_then_train(tmp_path: Path) -> None:
    for name in [
        "train-00001-of-00002.parquet",
        "validation-00000-of-00001.parquet",
        "train-00000-of-00002.parquet",
    ]:
        (tmp_path / name).write_bytes(b"placeholder")
    files = resolve_hotpotqa_files(tmp_path, download=False, timeout_s=1.0)
    assert [path.name for path in files] == [
        "validation-00000-of-00001.parquet",
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ]


def test_hotpotqa_hf_context_shape() -> None:
    row = {
        "context": {
            "title": ["Doc A", "Doc B"],
            "sentences": [["A one.", "A two."], ["B one."]],
        }
    }
    items = hotpot_context_items(row)
    assert items == [
        {"title": "Doc A", "text": "A one. A two."},
        {"title": "Doc B", "text": "B one."},
    ]


def test_mixed_trace_interleaves_sharegpt_and_rag(tmp_path: Path) -> None:
    trace_path = tmp_path / "sharegpt.json"
    trace_path.write_text(
        """
[
  {"id":"sg_0","conversations":[
    {"from":"human","value":"first question"},
    {"from":"gpt","value":"first answer"},
    {"from":"human","value":"second question"}
  ]},
  {"id":"sg_1","conversations":[
    {"from":"human","value":"alpha"},
    {"from":"gpt","value":"beta"},
    {"from":"human","value":"gamma"}
  ]}
]
""".strip(),
        encoding="utf-8",
    )
    hotpotqa_path = tmp_path / "hotpotqa.json"
    write_hotpotqa_fixture(hotpotqa_path)
    args = argparse.Namespace(
        workload="mixed",
        max_requests=6,
        rag_requests=2,
        max_sessions=2,
        hotpotqa_path=hotpotqa_path,
        download_hotpotqa=False,
        hotpotqa_max_examples=2,
        rag_chunk_words=56,
        rag_chunks_per_query=2,
        rag_query_repeats=4,
        sharegpt_order="file",
        timeout_s=120.0,
    )
    rows = load_replay_prompts(args, trace_path)
    assert len(rows) == 6
    assert [row["workload"] for row in rows[:4]] == [
        "sharegpt_session_prefix",
        "rag_chunk_reuse",
        "sharegpt_session_prefix",
        "rag_chunk_reuse",
    ]
    assert {row["workload"] for row in rows} == {
        "sharegpt_session_prefix",
        "rag_chunk_reuse",
    }


def test_frozen_replay_trace_round_trip(tmp_path: Path) -> None:
    trace_path = tmp_path / "sharegpt.json"
    trace_path.write_text(
        """
[
  {"id":"sg_0","conversations":[
    {"from":"human","value":"first question"},
    {"from":"gpt","value":"first answer"},
    {"from":"human","value":"second question"}
  ]},
  {"id":"sg_1","conversations":[
    {"from":"human","value":"alpha"},
    {"from":"gpt","value":"beta"},
    {"from":"human","value":"gamma"}
  ]}
]
""".strip(),
        encoding="utf-8",
    )
    hotpotqa_path = tmp_path / "hotpotqa.json"
    write_hotpotqa_fixture(hotpotqa_path)
    args = argparse.Namespace(
        trace_path=str(trace_path),
        workload="mixed",
        max_requests=8,
        rag_requests=4,
        max_sessions=2,
        hotpotqa_path=hotpotqa_path,
        download_hotpotqa=False,
        hotpotqa_max_examples=1,
        rag_chunk_words=56,
        rag_chunks_per_query=2,
        rag_query_repeats=4,
        sharegpt_order="file",
        timeout_s=120.0,
    )
    sessions = build_replay_sessions(args)
    assert sessions[0]["source"] == "sharegpt"
    assert sessions[1]["source"] == "hotpotqa"
    assert sessions[0]["turns"] == [
        {"i": 0, "user": "first question", "assistant": "first answer"},
        {"i": 1, "user": "second question"},
    ]
    out = tmp_path / "replay.jsonl"
    write_replay_trace(out, sessions)
    first_line = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert set(first_line) >= {"session_id", "turns"}
    assert first_line["turns_format"] == "cumulative_user"
    assert first_line["turns"] == [
        {"i": 0, "user": "User: first question\nAssistant:", "assistant": "first answer"},
        {"i": 1, "user": "User: first question\nAssistant: first answer\nUser: second question\nAssistant:"},
    ]
    replay_args = argparse.Namespace(**vars(args), replay_trace=str(out))
    prompts = load_replay_prompts(replay_args, trace_path)
    assert prompts == load_replay_prompts(replay_args, trace_path)
    assert any(row["workload"] == "rag_chunk_reuse" for row in prompts)
    assert all(row["replay_source"] == "frozen_replay_trace" for row in prompts)


def test_legacy_flat_prompt_jsonl_still_loads(tmp_path: Path) -> None:
    trace_path = tmp_path / "unused_sharegpt.json"
    trace_path.write_text("[]", encoding="utf-8")
    replay_path = tmp_path / "legacy_flat.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "request_id": "legacy:turn:000",
                "session_id": "legacy",
                "turn_index": 0,
                "prompt": "User: legacy prompt\nAssistant:",
                "workload": "sharegpt_session_prefix",
                "reuse_key": "legacy",
                "n_tokens": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        replay_trace=str(replay_path),
        max_requests=8,
    )
    prompts = load_replay_prompts(args, trace_path)
    assert len(prompts) == 1
    assert prompts[0]["prompt"] == "User: legacy prompt\nAssistant:"
    assert prompts[0]["replay_source"] == "frozen_replay_trace"


def test_weak_linked_trace_attaches_rag_to_sharegpt_turns(tmp_path: Path) -> None:
    trace_path = tmp_path / "sharegpt.json"
    trace_path.write_text(
        """
[
  {"id":"sg_0","conversations":[
    {"from":"human","value":"first question"},
    {"from":"gpt","value":"first answer"},
    {"from":"human","value":"second question"}
  ]},
  {"id":"sg_1","conversations":[
    {"from":"human","value":"alpha"},
    {"from":"gpt","value":"beta"},
    {"from":"human","value":"gamma"}
  ]}
]
""".strip(),
        encoding="utf-8",
    )
    hotpotqa_path = tmp_path / "hotpotqa.json"
    write_hotpotqa_fixture(hotpotqa_path)
    args = argparse.Namespace(
        trace_path=str(trace_path),
        workload="mixed",
        max_requests=8,
        rag_requests=4,
        max_sessions=2,
        hotpotqa_path=hotpotqa_path,
        download_hotpotqa=False,
        hotpotqa_max_examples=1,
        rag_chunk_words=56,
        rag_chunks_per_query=2,
        rag_query_repeats=4,
        sharegpt_order="file",
        link_mode="weak",
        weak_rag_repeat=2,
        timeout_s=120.0,
    )
    sessions = build_replay_sessions(args)
    assert {row["source"] for row in sessions} == {"sharegpt"}
    rag_turns = [turn for row in sessions for turn in row["turns"] if "rag" in turn]
    assert len(rag_turns) == 4
    prompts = replay_sessions_to_prompts(sessions, max_requests=8)
    linked = [row for row in prompts if row["workload"] == "sharegpt_hotpotqa_weak_link"]
    assert linked
    assert linked[0]["prompt"].startswith("Retrieved context:")
    assert linked[0]["rag_reuse_key"].startswith("rag:hotpotqa:")


def test_rag_replay_preserves_temperature_prior_fields() -> None:
    sessions = [
        {
            "session_id": "rag_hot_0",
            "source": "hotpotqa",
            "object_type": "rag_hot_chunk_set",
            "reuse_key": "rag:hot:0",
            "dataset": "hotpotqa",
            "hotpotqa_example_id": "hp",
            "hotpotqa_source_path": "fixture",
            "answer": "answer",
            "chunks": [
                {
                    "chunk_id": "c0",
                    "doc_id": "d0",
                    "title": "Doc",
                    "text": "Reusable context text.",
                }
            ],
            "temperature": "hot",
            "p_reuse_prior": 0.95,
            "turns": [{"i": 0, "user": "Question?"}],
        }
    ]
    prompts = replay_sessions_to_prompts(sessions, max_requests=1)
    assert prompts[0]["temperature"] == "hot"
    assert prompts[0]["p_reuse_prior"] == 0.95
    assert prompts[0]["object_type"] == "rag_hot_chunk_set"


def test_cumulative_replay_preserves_temperature_prior_fields() -> None:
    sessions = [
        {
            "session_id": "sg_hot_0",
            "source": "sharegpt",
            "object_type": "sharegpt_hot_context",
            "reuse_key": "sg:hot:0",
            "turns_format": "cumulative_user",
            "temperature": "hot",
            "p_reuse_prior": 0.95,
            "turns": [{"i": 0, "user": "User: fixed context\nAssistant:"}],
        }
    ]
    prompts = replay_sessions_to_prompts(sessions, max_requests=1)
    assert prompts[0]["temperature"] == "hot"
    assert prompts[0]["p_reuse_prior"] == 0.95
    assert prompts[0]["object_type"] == "sharegpt_hot_context"


if __name__ == "__main__":
    test_rag_chunk_trace_reuses_chunk_sets()
    test_hotpotqa_high_frequency_group_order_is_80_20()
    test_mixed_trace_interleaves_sharegpt_and_rag(Path("/tmp"))
    test_frozen_replay_trace_round_trip(Path("/tmp"))
    test_legacy_flat_prompt_jsonl_still_loads(Path("/tmp"))
    test_weak_linked_trace_attaches_rag_to_sharegpt_turns(Path("/tmp"))
    test_rag_replay_preserves_temperature_prior_fields()
    test_cumulative_replay_preserves_temperature_prior_fields()
    print("h0 rag trace tests ok")
