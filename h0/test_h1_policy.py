#!/usr/bin/env python3
"""Small unit smoke tests for H1 eviction policies."""

from __future__ import annotations

from edgekv_v1_offload.cache_policy import CachePolicy, LFUCachePolicy, LPECachePolicy, LRUCachePolicy
from edgekv_v1_offload.policy import H1Policy


def _observe(policy: H1Policy, object_id: str, size_mb: float, step: int = 0):
    return policy.observe_request(
        request_id=f"{object_id}:turn:{step:03d}",
        session_id=object_id,
        object_id=object_id,
        object_type="session_prefix",
        n_tokens=max(1, int(size_mb * 10)),
        size_mb=size_mb,
        hit=False,
    )


def test_cache_policy_subclasses() -> None:
    assert issubclass(LRUCachePolicy, CachePolicy)
    assert issubclass(LFUCachePolicy, CachePolicy)
    assert issubclass(LPECachePolicy, CachePolicy)


def test_lru_evicts_oldest() -> None:
    policy = H1Policy(policy="lru", gpu_budget_mb=10.0, c_re_ms_per_token=1.0)
    _observe(policy, "a", 5.0)
    _observe(policy, "b", 5.0)
    decisions = _observe(policy, "c", 5.0)
    assert any(row.action == "offload" and row.object_id == "a" for row in decisions)


def test_lfu_evicts_lowest_frequency() -> None:
    policy = H1Policy(policy="lfu", gpu_budget_mb=10.0, c_re_ms_per_token=1.0)
    _observe(policy, "a", 5.0)
    _observe(policy, "a", 5.0)
    _observe(policy, "b", 5.0)
    decisions = _observe(policy, "c", 5.0)
    assert any(row.action == "offload" and row.object_id == "b" for row in decisions)


def test_lpe_score_keeps_high_value_object() -> None:
    policy = H1Policy(policy="lpe-score", gpu_budget_mb=10.0, c_re_ms_per_token=1.0, theta_keep=0.5)
    _observe(policy, "hot", 5.0)
    _observe(policy, "hot", 5.0)
    _observe(policy, "cold", 5.0)
    decisions = _observe(policy, "new", 5.0)
    assert any(row.action == "drop" and row.object_id == "cold" for row in decisions)
    assert "hot" in policy.resident


def test_lpe_offloads_high_reuse_victim() -> None:
    policy = H1Policy(policy="lpe-score", gpu_budget_mb=10.0, c_re_ms_per_token=1.0, theta_keep=0.5)
    _observe(policy, "large", 10.0)
    _observe(policy, "large", 10.0)
    decisions = _observe(policy, "small", 5.0)
    assert any(row.action == "offload" and row.object_id == "large" for row in decisions)


if __name__ == "__main__":
    test_cache_policy_subclasses()
    test_lru_evicts_oldest()
    test_lfu_evicts_lowest_frequency()
    test_lpe_score_keeps_high_value_object()
    test_lpe_offloads_high_reuse_victim()
    print("h1 policy tests ok")
