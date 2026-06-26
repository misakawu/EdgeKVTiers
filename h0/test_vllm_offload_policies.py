#!/usr/bin/env python3
# TEST DATA CONTRACT: tests in this repository must use JSONL replay trace files as workload data. Do not use vLLM built-in datasets/test data.
"""Legacy CPU offload policy smoke tests.

H0/H1 experiments are GPU-only. These tests are disabled by default so the
normal validation path cannot accidentally import or exercise CPU KV offload
code. Set ``EDGEKV_ENABLE_LEGACY_CPU_OFFLOAD_TESTS=1`` to run them explicitly.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("EDGEKV_ENABLE_LEGACY_CPU_OFFLOAD_TESTS") != "1",
    reason="legacy CPU KV offload tests are outside the GPU-only H0/H1 path",
)

if os.environ.get("EDGEKV_ENABLE_LEGACY_CPU_OFFLOAD_TESTS") == "1":
    from edgekv_v1_offload.vllm_offload_policies import (
        BlockStatus,
        CachePolicy,
        H1LFUCachePolicy,
        H1LPECachePolicy,
        H1LRUCachePolicy,
        _CACHE_POLICIES,
        register_h1_cache_policies,
    )


def _ready(block_id: int) -> BlockStatus:
    block = BlockStatus(block_id)
    block.ref_cnt = 0
    return block


def test_registered() -> None:
    registry = register_h1_cache_policies()
    assert registry["h1_lru"] is H1LRUCachePolicy
    assert registry["h1_lfu"] is H1LFUCachePolicy
    assert registry["h1_lpe"] is H1LPECachePolicy
    assert issubclass(H1LRUCachePolicy, CachePolicy)
    assert issubclass(H1LFUCachePolicy, CachePolicy)
    assert issubclass(H1LPECachePolicy, CachePolicy)
    assert _CACHE_POLICIES["h1_lpe"] is H1LPECachePolicy


def test_lru_atomic_failure_and_order() -> None:
    policy = H1LRUCachePolicy(cache_capacity=4)
    for key, block_id in [("a", 1), ("b", 2), ("c", 3)]:
        policy.insert(key, _ready(block_id))
    assert policy.evict(2, {"a", "b"}) is None
    assert list(policy.blocks) == ["a", "b", "c"]
    evicted = policy.evict(1, {"a"})
    assert evicted and evicted[0][0] == "b"
    assert list(policy.blocks) == ["a", "c"]


def test_lfu_evicts_lowest_frequency() -> None:
    policy = H1LFUCachePolicy(cache_capacity=4)
    for key, block_id in [("hot", 1), ("cold", 2), ("new", 3)]:
        policy.insert(key, _ready(block_id))
    policy.touch(["hot"])
    policy.touch(["hot"])
    evicted = policy.evict(1, set())
    assert evicted and evicted[0][0] == "cold"


def test_lpe_evicts_lowest_score_and_protects() -> None:
    policy = H1LPECachePolicy(cache_capacity=4)
    for key, block_id in [("low", 1), ("high", 2), ("unknown", 3)]:
        policy.insert(key, _ready(block_id))
    policy.set_score("low", p_reuse=0.1, c_recomp_ms=10, size_mb=10)
    policy.set_score("high", p_reuse=0.9, c_recomp_ms=100, size_mb=10)
    evicted = policy.evict(1, {"unknown"})
    assert evicted and evicted[0][0] == "low"
    assert "high" in policy.blocks


if __name__ == "__main__":
    if os.environ.get("EDGEKV_ENABLE_LEGACY_CPU_OFFLOAD_TESTS") != "1":
        print("legacy CPU KV offload tests skipped; set EDGEKV_ENABLE_LEGACY_CPU_OFFLOAD_TESTS=1 to run")
        raise SystemExit(0)
    test_registered()
    test_lru_atomic_failure_and_order()
    test_lfu_evicts_lowest_frequency()
    test_lpe_evicts_lowest_score_and_protects()
    print("h1 vllm offload policy tests ok")
