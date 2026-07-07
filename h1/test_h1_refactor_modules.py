from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


class Pool:
    pass


def test_policy_factory_matches_existing_rank_semantics() -> None:
    from h1.policies.factory import create_policy

    pool = Pool()
    pool._edgekv_h1_freq = {7: 3}
    pool._edgekv_h1_recency = {7: 11}
    pool._edgekv_h1_pinned = {99}
    pool._edgekv_h1_scores = {5: 1.5}

    assert create_policy("vllm_default").name == "vllm_default"
    assert create_policy("h1_lru").rank_tuple(pool, 7) == (11.0, 0, 7)
    assert create_policy("h1_lfu").rank_tuple(pool, 7) == (3.0, 11, 7)
    assert create_policy("h1_lpe").rank_tuple(pool, 99) == (float("inf"), 1, 0)
    assert create_policy("h1_lpe").rank_tuple(pool, 5) == (1.5, 1, 5)
    assert create_policy("h1_lpe").needs_rank_state is True


def test_lpe_policy_orders_rejected_objects_before_accepted_with_same_score() -> None:
    from h1.policies.lpe import LPEPolicy

    accepted = {
        "object_id": "obj-a",
        "score": 4.0,
        "admission_rejected": False,
        "object_sort_key": 10,
    }
    rejected = {
        "object_id": "obj-b",
        "score": 4.0,
        "admission_rejected": True,
        "object_sort_key": 20,
    }
    pool = Pool()
    policy = LPEPolicy(block_profile=lambda _pool, block_id: rejected if block_id == 1 else accepted)

    assert policy.rank_tuple(pool, 1) < policy.rank_tuple(pool, 2)


def test_policy_lifecycle_methods_own_rank_state() -> None:
    from h1.policies.base import AccessInfo, AdmitInfo, EvictInfo
    from h1.policies.factory import create_policy

    pool = Pool()
    pool._edgekv_h1_seq = 0
    pool._edgekv_h1_scores = {}
    pool._edgekv_h1_p_reuse = {}
    pool._edgekv_h1_score_update_seq = {}
    pool._edgekv_h1_access_history = {}
    pool._edgekv_h1_pinned = set()
    pool._edgekv_h1_freq = {}
    pool._edgekv_h1_recency = {}

    policy = create_policy("h1_lfu")
    policy.on_admit(pool, 7, AdmitInfo(score=1.25, p_reuse=0.6, pinned=True))
    assert pool._edgekv_h1_freq[7] == 1
    assert pool._edgekv_h1_recency[7] == 1
    assert pool._edgekv_h1_scores[7] == 1.25
    assert 7 in pool._edgekv_h1_pinned

    policy.on_access(pool, 7, AccessInfo())
    assert pool._edgekv_h1_freq[7] == 2
    assert pool._edgekv_h1_recency[7] == 2
    assert policy.rank_tuple(pool, 7) == (2.0, 2, 7)

    policy.on_evict(pool, 7, EvictInfo())
    assert 7 not in pool._edgekv_h1_freq
    assert 7 not in pool._edgekv_h1_recency
    assert 7 not in pool._edgekv_h1_scores
    assert 7 not in pool._edgekv_h1_pinned


def test_lpe_policy_lifecycle_refreshes_profile_score_state() -> None:
    from h1.policies.base import AccessInfo, AdmitInfo
    from h1.policies.factory import create_policy

    pool = Pool()
    pool._edgekv_h1_seq = 0
    pool._edgekv_h1_scores = {}
    pool._edgekv_h1_p_reuse = {}
    pool._edgekv_h1_score_update_seq = {}
    pool._edgekv_h1_access_history = {}
    pool._edgekv_h1_pinned = set()
    pool._edgekv_h1_freq = {}
    pool._edgekv_h1_recency = {}

    profile = {"score": 3.5, "p_reuse": 0.75, "score_update_seq": 4}
    policy = create_policy("h1_lpe", block_profile=lambda _pool, _block_id: profile)
    policy.on_admit(pool, 5, AdmitInfo(score=1.0, p_reuse=0.2, profile=profile))
    assert pool._edgekv_h1_scores[5] == 3.5
    assert pool._edgekv_h1_p_reuse[5] == 0.75
    assert pool._edgekv_h1_score_update_seq[5] == 4

    profile.update({"score": 4.5, "p_reuse": 0.8, "score_update_seq": 8})
    policy.on_access(pool, 5, AccessInfo(profile=profile, refreshed=True))
    assert pool._edgekv_h1_freq[5] == 2
    assert pool._edgekv_h1_scores[5] == 4.5
    assert pool._edgekv_h1_p_reuse[5] == 0.8
    assert pool._edgekv_h1_score_update_seq[5] == 8


def test_gpu_stats_collector_snapshot_reset_and_flush(tmp_path: Path, monkeypatch) -> None:
    from h1.stats.collector import GpuStatsCollector

    stats_dir = tmp_path / "stats"
    monkeypatch.setenv("EDGEKV_H1_STATS_DIR", str(stats_dir))
    collector = GpuStatsCollector(policy_getter=lambda: "h1_lpe")
    collector.note("lookup_hits", 2)
    collector.note("native_queries", 10)
    collector.note("native_hits", 7)
    collector.note_policy_time_ns(2_000)

    snapshot = collector.snapshot()
    assert snapshot["lookup_total"] == 2
    assert snapshot["native_hit_rate"] == 0.7
    assert snapshot["hit_rate"] == 0.7
    assert snapshot["hit_source"] == "vllm_native_token_coverage"
    assert snapshot["policy"] == "h1_lpe"
    assert snapshot["policy_time_us_avg"] == 2.0

    collector.flush()
    written = list(stats_dir.glob("edgekv_gpu_stats_*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text(encoding="utf-8"))["policy"] == "h1_lpe"

    collector.reset()
    assert collector.snapshot()["lookup_total"] == 0


def test_runtime_config_parses_policy_and_monitor_path(tmp_path: Path, monkeypatch) -> None:
    from h1.runtime.config import RuntimeConfig

    stats_dir = tmp_path / "stats"
    monkeypatch.setenv("EDGEKV_H1_GPU_POLICY", " h1_lpe ")
    monkeypatch.setenv("EDGEKV_H1_STATS_DIR", str(stats_dir))
    monkeypatch.setenv("EDGEKV_H1_RUNTIME_MONITOR", "1")
    monkeypatch.setenv("H1_LPE_SCORE_UPDATE_INTERVAL", "4")

    config = RuntimeConfig.from_env()

    assert config.gpu_policy == "h1_lpe"
    assert config.policy_enabled is True
    assert config.policy_is_lpe is True
    assert config.policy_needs_rank_state is True
    assert config.lpe_monitor_path == stats_dir / "edgekv_lpe_runtime_monitor.jsonl"
    assert config.env_int("H1_LPE_SCORE_UPDATE_INTERVAL", 8) == 4


def test_runner_lpe_env_overrides_are_shared_and_policy_specific(tmp_path: Path) -> None:
    import h1._runner as runner

    lpe_env = runner.lpe_runtime_env_overrides("h1_lpe", tmp_path / "cell")
    lru_env = runner.lpe_runtime_env_overrides("h1_lru", tmp_path / "cell")

    assert lpe_env == {
        "EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES": "1",
        "EDGEKV_H1_RUNTIME_MONITOR": "1",
        "EDGEKV_H1_RUNTIME_MONITOR_PATH": str(tmp_path / "cell" / "runtime_monitor.jsonl"),
    }
    assert lru_env == {}


def test_runner_strips_vllm_per_batch_tqdm_but_keeps_engine_info() -> None:
    import h1._runner as runner

    assert runner._strip_vllm_batch_tqdm(
        "Adding requests: 100%|██████████| 8/8 [00:00<00:00, 244.19it/s]\n"
    ) == ""
    assert runner._strip_vllm_batch_tqdm(
        "Processed prompts:   0%|          | 0/8 [00:00<?, ?it/s]\n"
    ) == ""
    mixed = (
        "Processed prompts:   0%|          | 0/8 [00:00<?, ?it/s]"
        "INFO 07-01 17:56:49 [loggers.py:127] Prefix cache hit rate: 58.0%\n"
    )
    assert runner._strip_vllm_batch_tqdm(mixed).startswith("INFO 07-01")


def test_sitecustomize_rank_tuple_delegates_to_cached_policy_singleton() -> None:
    # sitecustomize 导入会 monkey-patch 全局 logging 等，用子进程隔离，避免污染 pytest 进程。
    # 同时验证：(1) 工厂实例被缓存为单例；(2) 委托后的 rank_tuple 与重构前逐分支等价。
    script = r"""
import sitecustomize as s
class Pool: pass
pool = Pool()
pool._edgekv_h1_freq = {7: 3}
pool._edgekv_h1_recency = {7: 11}
pool._edgekv_h1_pinned = {99}
pool._edgekv_h1_scores = {5: 1.5}
assert s._edgekv_policy_instance() is s._edgekv_policy_instance(), "policy not cached"
rt = lambda bid: s._edgekv_rank_tuple(pool, bid)
pol = s._edgekv_gpu_policy()
if pol == "h1_lfu":
    assert rt(7) == (3.0, 11, 7)
elif pol == "h1_lpe":
    assert rt(99) == (float("inf"), 1, 0)
    assert rt(5) == (1.5, 1, 5)
    # 打分委托：sitecustomize 重算结果须与 DefaultLPEScorer 逐字段相等
    from policies.lpe import DefaultLPEScorer
    prof = {"n_tokens": 100, "bytes_per_token": 2048.0, "p_reuse": 0.4, "c_recomp_ms": 12.0, "size_bytes": 4096}
    ref = dict(prof)
    s._edgekv_recompute_profile_score(prof)
    DefaultLPEScorer().recompute_score(ref)
    assert prof == ref, (prof, ref)
else:
    assert rt(7) == (11.0, 0, 7)
print("OK")
"""
    h1_dir = Path(__file__).resolve().parent
    for policy in ("h1_lru", "h1_lfu", "h1_lpe", "vllm_default"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(h1_dir),
            env={**os.environ, "EDGEKV_H1_GPU_POLICY": policy},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{policy}: {result.stderr}"
        assert "OK" in result.stdout, f"{policy}: {result.stdout}"


def test_default_lpe_scorer_matches_score_formula(monkeypatch) -> None:
    from h1.policies.lpe import DefaultLPEScorer

    scorer = DefaultLPEScorer()

    # 分支1：bytes_per_token>0，用 vLLM spec 字节
    p1 = {"n_tokens": 100, "bytes_per_token": 2048.0, "p_reuse": 0.4, "c_recomp_ms": 12.0, "size_bytes": 4096}
    scorer.recompute_score(p1)
    logical1 = 100 * 2048.0 / 1024.0 / 1024.0
    assert p1["logical_size_mb"] == logical1
    assert p1["size_mb"] == logical1
    assert p1["score_size_source"] == "vllm_kv_cache_spec_bytes_per_token"
    assert p1["size_source"] == "vllm_kv_cache_spec_bytes_per_token"
    assert p1["resident_size_mb"] == 4096 / 1024.0 / 1024.0
    assert p1["score"] == 0.4 * 12.0 / max(logical1, 1e-9)

    # 分支2：bytes_per_token=0，回退理论 mu_kv（默认 0.12）
    monkeypatch.delenv("EDGEKV_MU_KV_MB_PER_TOKEN", raising=False)
    p2 = {"n_tokens": 50, "bytes_per_token": 0.0, "p_reuse": 0.5, "c_recomp_ms": 6.0}
    scorer.recompute_score(p2)
    assert p2["logical_size_mb"] == 0.12 * 50
    assert p2["score_size_source"] == "fallback_theoretical_mu_kv"
    assert p2["score"] == 0.5 * 6.0 / max(0.12 * 50, 1e-9)

    # 分支3：logical_size_mb<=0 → score=0
    monkeypatch.setenv("EDGEKV_MU_KV_MB_PER_TOKEN", "0.0")
    p3 = {"n_tokens": 10, "bytes_per_token": 0.0, "p_reuse": 0.9, "c_recomp_ms": 9.0}
    scorer.recompute_score(p3)
    assert p3["logical_size_mb"] == 0.0
    assert p3["score"] == 0.0

    # 缺失字段用默认值（p_reuse=0.5, c_recomp_ms=0.0）
    monkeypatch.delenv("EDGEKV_MU_KV_MB_PER_TOKEN", raising=False)
    p4 = {"n_tokens": 20, "bytes_per_token": 1024.0}
    scorer.recompute_score(p4)
    assert p4["score"] == 0.5 * 0.0 / max(20 * 1024.0 / 1024.0 / 1024.0, 1e-9)
