#!/usr/bin/env python3
# TEST DATA CONTRACT: tests in this repository must use JSONL replay trace files as workload data. Do not use vLLM built-in datasets/test data.
"""Smoke tests for H1 reuse of the H0 mixed RAG replay trace."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

H1_DIR = Path(__file__).resolve().parent
REPO_ROOT = H1_DIR.parent
H0_DIR = REPO_ROOT / "h0"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(H1_DIR) not in sys.path:
    sys.path.insert(0, str(H1_DIR))
if str(H0_DIR) not in sys.path:
    sys.path.insert(0, str(H0_DIR))

import run_h1_vllm0110_real as h1
from edgekv_cop import COPProfiler
from test_h0_rag_trace import write_hotpotqa_fixture


def load_h1_sitecustomize():
    spec = importlib.util.spec_from_file_location("edgekv_h1_sitecustomize", H1_DIR / "sitecustomize.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBlock:
    def __init__(self, ref_cnt: int = 0, block_id: int = 0) -> None:
        self.ref_cnt = ref_cnt
        self.block_id = block_id


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


def write_sharegpt_fixture(path: Path) -> None:
    path.write_text(
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


def test_h1_defaults_use_repo_pressure_trace() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert h1.DEFAULT_SHAREGPT_TRACE_PATH == repo_root / "data" / "ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
    assert h1.DEFAULT_HOTPOTQA_PATH == repo_root / "data" / "hotpotqa"
    assert h1.DEFAULT_REPLAY_TRACE_PATH == repo_root / "data" / "edgekv_traces" / "sharegpt_hotpotqa_session.jsonl"


def test_step3_driver_uses_pressure_replay_not_vllm_builtin_dataset() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "h1" / "run_step3_budget_tiers.py").read_text(encoding="utf-8")
    assert "sharegpt_hotpotqa_session.jsonl" in source
    assert "H1_BENCH_DATASET" not in source
    assert "prefix_repetition" not in source



def test_h1_loads_h0_mixed_rag_trace(tmp_path: Path, monkeypatch) -> None:
    trace_path = tmp_path / "sharegpt.json"
    hotpotqa_path = tmp_path / "hotpotqa.json"
    write_sharegpt_fixture(trace_path)
    write_hotpotqa_fixture(hotpotqa_path)
    monkeypatch.setattr(h1.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())

    args = argparse.Namespace(
        model="models/Qwen2.5-7B-Instruct",
        trace_path=str(trace_path),
        replay_trace=str(tmp_path / "missing_replay.jsonl"),
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
        max_tokens=16,
        max_model_len=2048,
    )

    rows = h1.load_trace(args)
    assert len(rows) == 6
    assert {row["workload"] for row in rows} == {"sharegpt_session_prefix", "rag_chunk_reuse"}
    assert any(row.get("chunk_ids") for row in rows if row["workload"] == "rag_chunk_reuse")


def test_h1_loads_session_turns_jsonl_replay_trace(tmp_path: Path, monkeypatch) -> None:
    trace_path = tmp_path / "unused_sharegpt.json"
    trace_path.write_text("[]", encoding="utf-8")
    replay_path = tmp_path / "session_turns.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "session_id": "sg_0007",
                "turns_format": "cumulative_user",
                "turns": [
                    {"i": 0, "user": "User: first user input\nAssistant:"},
                    {
                        "i": 1,
                        "user": "User: first user input\nAssistant: first answer\nUser: second user input\nAssistant:",
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(h1.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    args = argparse.Namespace(
        model="models/Qwen2.5-7B-Instruct",
        trace_path=str(trace_path),
        replay_trace=str(replay_path),
        workload="mixed",
        max_requests=8,
        rag_requests=0,
        max_sessions=2,
        hotpotqa_path=tmp_path / "unused_hotpotqa.json",
        download_hotpotqa=False,
        hotpotqa_max_examples=0,
        rag_chunk_words=56,
        rag_chunks_per_query=2,
        rag_query_repeats=4,
        sharegpt_order="file",
        timeout_s=120.0,
        max_tokens=16,
        max_model_len=2048,
    )

    rows = h1.load_trace(args)
    assert [row["request_id"] for row in rows] == ["sg_0007:turn:000", "sg_0007:turn:001"]
    assert rows[0]["prompt"] == "User: first user input\nAssistant:"
    assert rows[1]["prompt"] == "User: first user input\nAssistant: first answer\nUser: second user input\nAssistant:"
    assert all(row["replay_source"] == "frozen_replay_trace" for row in rows)


def test_h1_trace_side_fields_include_lpe_score() -> None:
    item = {
        "request_id": "rag_0",
        "session_id": "rag_session_0",
        "turn_index": 1,
        "prompt": "Retrieved context...",
        "prompt_chars": 20,
        "n_tokens": 128,
        "workload": "rag_chunk_reuse",
        "object_type": "rag_chunk_set",
        "reuse_key": "rag:hotpotqa:a|b",
        "chunk_ids": ["a", "b"],
    }

    fields = h1.request_trace_fields(
        item,
        trace_hit=True,
        rag_hit=False,
        policy="h1_lpe",
        kv_mib_per_token=0.25,
        cop=h1.COPProfiler(mu_kv_mb_per_token=0.25),
        event_index=0,
    )
    assert fields["hit"] is True
    assert fields["hit_source"] == "trace_side_reuse_key"
    assert fields["object_id"] == "rag:hotpotqa:a|b"
    assert fields["p_reuse"] > 0.0
    assert fields["c_recomp_ms"] == 15.36
    assert fields["size_mb"] == 32.0
    assert abs(fields["score"] - (0.12 * fields["p_reuse"] / 0.25)) < 1e-6
    assert fields["score_source"] == "object_level_cop"
    assert fields["lpe_action"] == "score_evaluated"
    assert "prompt" not in fields


def test_h1_weak_link_rag_uses_rag_object_id() -> None:
    item = {
        "request_id": "sg_0:turn:001",
        "session_id": "sg_0",
        "turn_index": 1,
        "prompt": "Retrieved context...\nUser: second question\nAssistant:",
        "prompt_chars": 50,
        "n_tokens": 160,
        "workload": "sharegpt_hotpotqa_weak_link",
        "object_type": "sharegpt_session_prefix",
        "reuse_key": "sg_0",
        "rag_reuse_key": "rag:hotpotqa:a|b",
    }

    fields = h1.request_trace_fields(
        item,
        trace_hit=False,
        rag_hit=True,
        policy="h1_lpe",
        kv_mib_per_token=0.25,
        cop=h1.COPProfiler(mu_kv_mb_per_token=0.25),
        event_index=1,
    )
    assert fields["object_id"] == "rag:hotpotqa:a|b"
    assert fields["object_type"] == "rag_chunk_set"
    assert fields["hit"] is False
    assert fields["rag_hit"] is True
    assert fields["score_source"] == "object_level_cop"


def test_h1_trace_side_reuse_tracks_reuse_key() -> None:
    item = {"session_id": "s0", "reuse_key": "shared-prefix"}
    assert h1.trace_side_reuse(item, set()) == ("shared-prefix", False, "", False)
    assert h1.trace_side_reuse(item, {"shared-prefix"}) == ("shared-prefix", True, "", False)


def test_sitecustomize_infers_prefix_repetition_object_without_extra_args() -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()
    old_env = {
        key: os.environ.get(key)
        for key in (
            "H1_PREFIX_REPETITION_PREFIX_LEN",
            "EDGEKV_MU_KV_MB_PER_TOKEN",
            "EDGEKV_C_RE_MS_PER_TOKEN",
        )
    }
    os.environ["H1_PREFIX_REPETITION_PREFIX_LEN"] = "4"
    os.environ["EDGEKV_MU_KV_MB_PER_TOKEN"] = "0.25"
    os.environ["EDGEKV_C_RE_MS_PER_TOKEN"] = "0.12"

    class Request:
        all_token_ids = [11, 12, 13, 14, 99, 100]
        prompt_token_ids = all_token_ids
        request_id = "req-0"
        sampling_params = None

    try:
        object_id, object_type, n_tokens = sitecustomize._edgekv_infer_object_id(Request(), 0, 4)
        assert object_id.startswith("prefix:")
        assert object_type == "prefix_repetition_prefix"
        assert n_tokens == 4

        profile = sitecustomize._edgekv_profile_from_values(object_id, object_type, n_tokens)
        # No vLLM group registered -> logical size falls back to mu_kv * n_tokens
        # and is used as the (stable) score denominator; resident size is 0.
        assert profile["logical_size_mb"] == 0.25 * 4
        assert profile["size_mb"] == profile["logical_size_mb"]
        assert profile["resident_size_mb"] == 0.0
        assert profile["size_bytes"] == 0
        assert profile["score_size_source"] == "fallback_theoretical_mu_kv"
        assert profile["c_recomp_ms"] == 0.48
        assert profile["score"] == (
            profile["p_reuse"] * profile["c_recomp_ms"] / profile["logical_size_mb"]
        )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_sitecustomize_resident_size_uses_vllm_page_size_bytes() -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()

    class Spec:
        block_size = 16
        page_size_bytes = 393216

    class Pool:
        pass

    pool = Pool()
    sitecustomize._edgekv_init_pool_state(pool)
    sitecustomize._edgekv_register_kv_cache_group(0, Spec())
    profile = sitecustomize._edgekv_profile_from_values(
        "obj-a",
        "prefix_repetition_prefix",
        128,
        {"p_reuse": 0.5, "c_recomp_ms": 15.36},
    )
    sitecustomize._edgekv_set_block_object(pool, 0, 7, "obj-a")
    sitecustomize._edgekv_set_block_object(pool, 0, 8, "obj-a")

    assert profile["resident_block_count"] == 2
    assert profile["resident_block_count_by_group"] == {0: 2}
    assert profile["size_bytes"] == 2 * Spec.page_size_bytes
    # resident_size_mb is diagnostic and tracks resident block bytes.
    assert profile["resident_size_mb"] == (2 * Spec.page_size_bytes) / 1024 / 1024
    # logical (score) size is stable: n_tokens * per-token KV bytes from vLLM.
    bytes_per_token = Spec.page_size_bytes / Spec.block_size
    assert profile["bytes_per_token"] == bytes_per_token
    assert profile["logical_size_mb"] == 128 * bytes_per_token / 1024 / 1024
    assert profile["size_mb"] == profile["logical_size_mb"]
    assert profile["score_size_source"] == "vllm_kv_cache_spec_bytes_per_token"
    assert profile["score"] == (
        profile["p_reuse"] * profile["c_recomp_ms"] / profile["logical_size_mb"]
    )
    score_before = profile["score"]

    removed = sitecustomize._edgekv_drop_block_object(pool, 0, 7)
    assert removed == "obj-a"
    assert profile["resident_block_count"] == 1
    assert profile["size_bytes"] == Spec.page_size_bytes
    # resident_size_mb halves with the dropped block, but the logical score is
    # unchanged -- the score no longer inflates as resident blocks shrink.
    assert profile["resident_size_mb"] == Spec.page_size_bytes / 1024 / 1024
    assert profile["logical_size_mb"] == 128 * bytes_per_token / 1024 / 1024
    assert profile["score"] == score_before


def test_sitecustomize_resident_size_updates_incrementally_on_reassign() -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()

    class Spec:
        block_size = 16
        page_size_bytes = 100

    class Pool:
        pass

    pool = Pool()
    sitecustomize._edgekv_init_pool_state(pool)
    sitecustomize._edgekv_register_kv_cache_group(0, Spec())
    obj_a = sitecustomize._edgekv_profile_from_values("obj-a", "prefix", 16)
    obj_b = sitecustomize._edgekv_profile_from_values("obj-b", "prefix", 16)

    sitecustomize._edgekv_set_block_object(pool, 0, 1, "obj-a")
    sitecustomize._edgekv_set_block_object(pool, 0, 2, "obj-a")
    assert obj_a["resident_block_count"] == 2
    assert obj_a["size_bytes"] == 200

    sitecustomize._edgekv_set_block_object(pool, 0, 1, "obj-b")
    assert obj_a["resident_block_count"] == 1
    assert obj_a["size_bytes"] == 100
    assert obj_b["resident_block_count"] == 1
    assert obj_b["size_bytes"] == 100

    assert sitecustomize._edgekv_block_profile(pool, 1) is obj_b
    assert sitecustomize._edgekv_block_profile(pool, 2) is obj_a

    sitecustomize._edgekv_drop_block_object(pool, 0, 1)
    assert obj_b["resident_block_count"] == 0
    assert obj_b["size_bytes"] == 0
    assert sitecustomize._edgekv_block_profile(pool, 1) is None


def test_sitecustomize_lpe_reuse_uses_freq_recency_and_type_prior() -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()

    prefix = sitecustomize._edgekv_profile_from_values("prefix-hot", "prefix", 16)
    token_block = sitecustomize._edgekv_profile_from_values("block-cold", "token_block", 16)

    sitecustomize._edgekv_note_profile_access(prefix, hit=True)
    sitecustomize._edgekv_note_profile_access(token_block, hit=True)
    sitecustomize._edgekv_note_profile_access(token_block, hit=False)

    assert prefix["p_type"] == 0.80
    assert token_block["p_type"] == 0.45
    assert prefix["p_reuse"] > token_block["p_reuse"]
    assert "p_freq" in prefix
    assert "p_recency" in prefix


def test_sitecustomize_score_update_is_throttled(monkeypatch) -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()
    monkeypatch.setenv("H1_LPE_SCORE_UPDATE_INTERVAL", "4")

    profile = sitecustomize._edgekv_profile_from_values("prefix-hot", "prefix", 16)

    assert sitecustomize._edgekv_note_profile_access(profile, hit=True) is True
    first_score = profile["score"]
    assert sitecustomize._edgekv_note_profile_access(profile, hit=True) is False
    assert sitecustomize._edgekv_note_profile_access(profile, hit=True) is False
    assert profile["score"] == first_score
    assert sitecustomize._edgekv_note_profile_access(profile, hit=True) is True
    assert profile["score_update_seq"] == 4


def test_sitecustomize_token_block_inference_avoids_default_hash(monkeypatch) -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()
    monkeypatch.setenv("H1_PREFIX_REPETITION_PREFIX_LEN", "0")
    monkeypatch.delenv("H1_LPE_HASH_TOKEN_BLOCKS", raising=False)

    class Request:
        all_token_ids = [11, 12, 13, 14]
        prompt_token_ids = all_token_ids
        request_id = "req-0"
        sampling_params = None

    object_id, object_type, n_tokens = sitecustomize._edgekv_infer_object_id(Request(), 0, 4)
    assert object_id == "block:req-0:0:4"
    assert object_type == "token_block"
    assert n_tokens == 4


def test_sitecustomize_reorder_skips_low_pressure_without_scanning_queue(monkeypatch) -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()
    monkeypatch.setattr(sitecustomize, "_EDGEKV_GPU_POLICY_ENABLED", True)
    monkeypatch.setenv("H1_LPE_LIGHT_PATH", "1")
    monkeypatch.setenv("H1_LPE_PRESSURE_FREE_RATIO", "0.15")

    class Queue:
        num_free_blocks = 90

        def get_all_free_blocks(self):
            raise AssertionError("low-pressure path must not scan free queue")

    class Pool:
        enable_caching = True
        free_block_queue = Queue()
        num_gpu_blocks = 100

    pool = Pool()
    sitecustomize._edgekv_init_pool_state(pool)
    pool._edgekv_h1_queue_dirty = True

    sitecustomize._edgekv_reorder_free_queue(pool, num_blocks=1)
    stats = sitecustomize.get_edgekv_gpu_cache_stats()

    assert stats["free_queue_reorder_calls"] == 1
    assert stats["free_queue_reorder_skipped"] == 1
    assert stats["free_queue_reorder_blocks"] == 0


def test_sitecustomize_reorder_records_candidate_window(monkeypatch) -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()
    monkeypatch.setattr(sitecustomize, "_EDGEKV_GPU_POLICY_ENABLED", True)
    monkeypatch.setattr(sitecustomize, "_EDGEKV_GPU_POLICY_VALUE", "h1_lpe")
    monkeypatch.setenv("H1_LPE_LIGHT_PATH", "0")
    monkeypatch.setenv("H1_LPE_REORDER_MODE", "window")
    monkeypatch.setenv("H1_LPE_REORDER_WINDOW", "2")

    class Block:
        def __init__(self, block_id: int) -> None:
            self.block_id = block_id
            self.is_null = False
            self.prev_free_block = None
            self.next_free_block = None

    class Queue:
        """Faithful mini vLLM FreeKVCacheBlockQueue.

        The incremental reorder reads the head window and uses the prev/next
        link pointers as the O(1) free-queue membership signal, so the fake must
        model the doubly-linked list (fake head/tail, nulling pointers on
        remove, relinking on append_n) rather than just a Python list.
        """

        def __init__(self) -> None:
            self.fake_free_list_head = Block(-1)
            self.fake_free_list_tail = Block(-1)
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head
            self.num_free_blocks = 0
            self.append_n([Block(1), Block(2), Block(3)])

        def get_all_free_blocks(self):
            out = []
            node = self.fake_free_list_head.next_free_block
            while node is not None and node.next_free_block is not None:
                out.append(node)
                node = node.next_free_block
            return out

        def remove(self, block):
            block.prev_free_block.next_free_block = block.next_free_block
            block.next_free_block.prev_free_block = block.prev_free_block
            block.prev_free_block = block.next_free_block = None
            self.num_free_blocks -= 1

        def append_n(self, blocks):
            last = self.fake_free_list_tail.prev_free_block
            for block in blocks:
                block.prev_free_block = last
                last.next_free_block = block
                last = block
            last.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = last
            self.num_free_blocks += len(blocks)

    class Pool:
        enable_caching = True
        free_block_queue = Queue()

    pool = Pool()
    sitecustomize._edgekv_init_pool_state(pool)
    pool._edgekv_h1_scores.update({1: 10.0, 2: 1.0, 3: 0.0})

    sitecustomize._edgekv_reorder_free_queue(pool, num_blocks=1)
    stats = sitecustomize.get_edgekv_gpu_cache_stats()

    assert stats["free_queue_reorder_calls"] == 1
    assert stats["free_queue_reorder_blocks"] == 3
    assert stats["free_queue_reorder_window"] == 3
    assert stats["queue_reorders"] == 1


def _make_faithful_queue(sitecustomize, block_ids):
    """Build a faithful mini vLLM FreeKVCacheBlockQueue seeded with block_ids."""

    class Block:
        def __init__(self, block_id: int) -> None:
            self.block_id = block_id
            self.is_null = False
            self.prev_free_block = None
            self.next_free_block = None

    class Queue:
        def __init__(self) -> None:
            self.fake_free_list_head = Block(-1)
            self.fake_free_list_tail = Block(-1)
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head
            self.num_free_blocks = 0
            self.append_n([Block(bid) for bid in block_ids])

        def get_all_free_blocks(self):
            out = []
            node = self.fake_free_list_head.next_free_block
            while node is not None and node.next_free_block is not None:
                out.append(node)
                node = node.next_free_block
            return out

        def remove(self, block):
            block.prev_free_block.next_free_block = block.next_free_block
            block.next_free_block.prev_free_block = block.prev_free_block
            block.prev_free_block = block.next_free_block = None
            self.num_free_blocks -= 1

        def append_n(self, blocks):
            last = self.fake_free_list_tail.prev_free_block
            for block in blocks:
                block.prev_free_block = last
                last.next_free_block = block
                last = block
            last.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = last
            self.num_free_blocks += len(blocks)

    return Queue()


def test_sitecustomize_object_admission_marks_low_score_reject() -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()

    class Spec:
        block_size = 16
        page_size_bytes = 1600

    class Pool:
        pass

    pool = Pool()
    sitecustomize._edgekv_init_pool_state(pool)
    sitecustomize._edgekv_register_kv_cache_group(0, Spec())

    high = sitecustomize._edgekv_profile_from_values(
        "obj-high", "prefix", 16, {"p_reuse": 0.9, "c_recomp_ms": 100.0}
    )
    sitecustomize._edgekv_set_block_object(pool, 0, 1, "obj-high")
    # Empty cache -> the first resident object is admitted.
    assert sitecustomize._edgekv_evaluate_object_admission(high, False) == "accept"

    low = sitecustomize._edgekv_profile_from_values(
        "obj-low", "prefix", 16, {"p_reuse": 0.1, "c_recomp_ms": 1.0}
    )
    sitecustomize._edgekv_set_block_object(pool, 0, 2, "obj-low")
    assert low["score"] < high["score"]

    decision = sitecustomize._edgekv_evaluate_object_admission(low, False)
    assert decision == "reject"
    assert low["admission_decision"] == "reject"
    assert low["admission_rejected"] is True
    assert low["admission_seq"] >= 1
    assert low["admission_min_resident_score"] == high["score"]

    # Diagnostic reject must NOT drop the object's cached blocks.
    assert low["resident_block_count"] == 1
    assert sitecustomize._edgekv_block_profile(pool, 2) is low

    # A second decision on the same object is a no-op (decided once).
    assert sitecustomize._edgekv_evaluate_object_admission(low, False) is None

    stats = sitecustomize.get_edgekv_gpu_cache_stats()
    assert stats["admission_rejection_count"] == 1
    assert stats["admission_accept_count"] == 1
    assert stats["admission_mode"] == "diagnostic"


def test_sitecustomize_pinned_object_never_ranked_as_victim(monkeypatch) -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()
    monkeypatch.setattr(sitecustomize, "_EDGEKV_GPU_POLICY_VALUE", "h1_lpe")

    class Spec:
        block_size = 16
        page_size_bytes = 1600

    class Pool:
        pass

    pool = Pool()
    sitecustomize._edgekv_init_pool_state(pool)
    sitecustomize._edgekv_register_kv_cache_group(0, Spec())
    sitecustomize._edgekv_profile_from_values(
        "obj-a", "prefix", 16, {"p_reuse": 0.1, "c_recomp_ms": 1.0}
    )
    sitecustomize._edgekv_set_block_object(pool, 0, 5, "obj-a")

    normal_rank = sitecustomize._edgekv_rank_tuple(pool, 5)
    assert normal_rank[0] != float("inf")

    pool._edgekv_h1_pinned.add(5)
    pinned_rank = sitecustomize._edgekv_rank_tuple(pool, 5)
    assert pinned_rank[0] == float("inf")
    # A pinned block ranks strictly after any finite-score block, so the
    # lowest-first eviction reorder never selects it as a victim.
    assert pinned_rank > normal_rank


def test_sitecustomize_object_rank_evicts_rejected_before_accepted(monkeypatch) -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()
    monkeypatch.setattr(sitecustomize, "_EDGEKV_GPU_POLICY_VALUE", "h1_lpe")

    class Spec:
        block_size = 16
        page_size_bytes = 1600

    class Pool:
        pass

    pool = Pool()
    sitecustomize._edgekv_init_pool_state(pool)
    sitecustomize._edgekv_register_kv_cache_group(0, Spec())

    meta = {"p_reuse": 0.5, "c_recomp_ms": 10.0}
    accepted = sitecustomize._edgekv_profile_from_values("obj-acc", "prefix", 16, meta)
    rejected = sitecustomize._edgekv_profile_from_values("obj-rej", "prefix", 16, meta)
    sitecustomize._edgekv_set_block_object(pool, 0, 10, "obj-acc")
    sitecustomize._edgekv_set_block_object(pool, 0, 11, "obj-rej")

    # Same score, but one object is logically rejected.
    assert accepted["score"] == rejected["score"]
    rejected["admission_rejected"] = True

    rank_accepted = sitecustomize._edgekv_rank_tuple(pool, 10)
    rank_rejected = sitecustomize._edgekv_rank_tuple(pool, 11)
    assert rank_accepted[0] == rank_rejected[0]
    # At equal score the rejected object sorts first (evicted first).
    assert rank_rejected < rank_accepted
    assert rank_rejected[1] == 0
    assert rank_accepted[1] == 1


def test_sitecustomize_object_level_reorder_groups_blocks(monkeypatch) -> None:
    sitecustomize = load_h1_sitecustomize()
    sitecustomize.reset_edgekv_gpu_cache_stats()
    monkeypatch.setattr(sitecustomize, "_EDGEKV_GPU_POLICY_ENABLED", True)
    monkeypatch.setattr(sitecustomize, "_EDGEKV_GPU_POLICY_VALUE", "h1_lpe")
    monkeypatch.setattr(sitecustomize, "_EDGEKV_GPU_POLICY_IS_LPE", True)
    monkeypatch.setenv("H1_LPE_LIGHT_PATH", "0")
    monkeypatch.setenv("H1_LPE_REORDER_MODE", "window")
    monkeypatch.setenv("H1_LPE_REORDER_WINDOW", "16")

    class Spec:
        block_size = 16
        page_size_bytes = 1600

    class Pool:
        enable_caching = True

    pool = Pool()
    sitecustomize._edgekv_init_pool_state(pool)
    sitecustomize._edgekv_register_kv_cache_group(0, Spec())

    # Interleave two objects' blocks in the free queue: A owns {1,3}, B {2,4}.
    # A has the lower object score, so both A blocks must move to the front as a
    # contiguous group, ahead of B's contiguous group.
    pool.free_block_queue = _make_faithful_queue(sitecustomize, [1, 2, 3, 4])
    sitecustomize._edgekv_profile_from_values(
        "obj-a", "prefix", 16, {"p_reuse": 0.1, "c_recomp_ms": 1.0}
    )
    sitecustomize._edgekv_profile_from_values(
        "obj-b", "prefix", 16, {"p_reuse": 0.9, "c_recomp_ms": 100.0}
    )
    sitecustomize._edgekv_set_block_object(pool, 0, 1, "obj-a")
    sitecustomize._edgekv_set_block_object(pool, 0, 3, "obj-a")
    sitecustomize._edgekv_set_block_object(pool, 0, 2, "obj-b")
    sitecustomize._edgekv_set_block_object(pool, 0, 4, "obj-b")

    sitecustomize._edgekv_reorder_free_queue(pool, num_blocks=1)

    order = [block.block_id for block in pool.free_block_queue.get_all_free_blocks()]
    pos = {bid: idx for idx, bid in enumerate(order)}
    # Each object's blocks are contiguous...
    assert abs(pos[1] - pos[3]) == 1
    assert abs(pos[2] - pos[4]) == 1
    # ...and the low-score object A is ranked ahead of B.
    assert max(pos[1], pos[3]) < min(pos[2], pos[4])


def test_cop_profiler_updates_reuse_and_score() -> None:
    cop = COPProfiler(mu_kv_mb_per_token=0.25, c_re_ms_per_token=0.12)
    item = {
        "request_id": "req-0",
        "reuse_key": "block-a",
        "workload": "sharegpt_session_prefix",
        "object_type": "prefix",
        "n_tokens": 10,
    }
    profile = cop.update_from_item(item, hit=False, access_index=1)
    assert cop.get("block-a") is profile
    assert 0.0 < profile.p_reuse < 0.5
    first_reuse = profile.p_reuse
    assert round(profile.score, 6) == round(profile.p_reuse * 0.12 / 0.25, 6)

    profile = cop.update_from_item(item, hit=True, access_index=2)
    assert profile.p_reuse > first_reuse
    assert round(profile.score, 6) == round(profile.p_reuse * 0.12 / 0.25, 6)

    profile = cop.update_from_item(item, hit=False, access_index=3)
    assert 0.0 < profile.p_reuse < 1.0
    assert round(profile.score, 6) == round(profile.p_reuse * 0.12 / 0.25, 6)


def test_cop_profiler_prior_separates_hot_and_cold() -> None:
    cop = COPProfiler(mu_kv_mb_per_token=0.25, c_re_ms_per_token=0.12)
    hot = cop.update_from_item(
        {
            "reuse_key": "hot-a",
            "object_type": "sharegpt_hot_context",
            "n_tokens": 10,
            "p_reuse_prior": 0.95,
        },
        hit=False,
        access_index=1,
    )
    cold = cop.update_from_item(
        {
            "reuse_key": "cold-a",
            "object_type": "sharegpt_cold_context",
            "n_tokens": 10,
            "p_reuse_prior": 0.05,
        },
        hit=True,
        access_index=2,
    )
    assert hot.p_reuse_prior == 0.95
    assert cold.p_reuse_prior == 0.05
    assert hot.p_reuse > cold.p_reuse
    assert hot.score > cold.score


def test_h1_trace_fields_preserve_prior_temperature() -> None:
    item = {
        "request_id": "req-hot",
        "session_id": "req-hot",
        "prompt": "hot prompt",
        "n_tokens": 64,
        "workload": "sharegpt_session_prefix",
        "object_type": "sharegpt_hot_context",
        "reuse_key": "hot-key",
        "temperature": "hot",
        "p_reuse_prior": 0.95,
    }
    fields = h1.request_trace_fields(
        item,
        trace_hit=False,
        rag_hit=False,
        policy="h1_lpe",
        kv_mib_per_token=0.25,
        cop=h1.COPProfiler(mu_kv_mb_per_token=0.25),
        event_index=0,
    )
    assert fields["temperature"] == "hot"
    assert fields["p_reuse_prior"] == 0.95
    assert fields["p_reuse"] > 0.65


def test_sitecustomize_refresh_keeps_hot_prior_after_first_miss() -> None:
    sitecustomize = load_h1_sitecustomize()
    profile = {
        "object_id": "hot",
        "object_type": "sharegpt_hot_context",
        "n_tokens": 64,
        "size_mb": 16.0,
        "p_reuse_prior": 0.95,
        "access_count": 1,
        "hit_count": 0,
        "c_recomp_ms": 7.68,
    }
    sitecustomize._edgekv_refresh_profile_reuse(profile)
    assert profile["p_type"] == 0.90
    assert profile["p_reuse"] > 0.70

    cold_profile = {
        "object_id": "cold",
        "object_type": "sharegpt_cold_context",
        "n_tokens": 64,
        "size_mb": 16.0,
        "p_reuse_prior": 0.05,
        "access_count": 1,
        "hit_count": 1,
        "c_recomp_ms": 7.68,
    }
    sitecustomize._edgekv_refresh_profile_reuse(cold_profile)
    assert cold_profile["p_type"] == 0.10
    assert cold_profile["p_reuse"] < profile["p_reuse"]
