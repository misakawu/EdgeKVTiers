from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable


DEFAULT_GPU_STATS: dict[str, int] = {
    "lookup_hits": 0,
    "lookup_misses": 0,
    "native_queries": 0,
    "native_hits": 0,
    "native_requests": 0,
    "touches": 0,
    "cached_blocks": 0,
    "evictions": 0,
    "queue_reorders": 0,
    "free_queue_reorder_calls": 0,
    "free_queue_reorder_blocks": 0,
    "free_queue_reorder_skipped": 0,
    "free_queue_reorder_window": 0,
    "admissions": 0,
    "admission_rejections": 0,
    "admission_accepts": 0,
    "evict_high_reuse": 0,
    "evict_drop": 0,
    "evicted_score_count": 0,
    "evicted_p_reuse_count": 0,
    "low_score_evictions": 0,
    "hot_prefix_evictions": 0,
    "pinned_eviction_attempts": 0,
    "policy_timing_samples": 0,
    "eviction_decision_timing_samples": 0,
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _stats_dir() -> Path | None:
    value = os.environ.get("EDGEKV_H1_STATS_DIR", "").strip()
    return Path(value) if value else None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _histogram(values: list[float], buckets: int = 10) -> dict[str, Any]:
    if not values:
        return {"min": 0.0, "max": 0.0, "buckets": []}
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {"min": float(low), "max": float(high), "buckets": [{"lo": float(low), "hi": float(high), "count": len(values)}]}
    bucket_count = max(int(buckets), 1)
    width = (high - low) / bucket_count
    counts = [0 for _ in range(bucket_count)]
    for value in values:
        counts[min(int((value - low) / width), bucket_count - 1)] += 1
    return {
        "min": float(low),
        "max": float(high),
        "buckets": [
            {"lo": float(low + idx * width), "hi": float(low + (idx + 1) * width), "count": count}
            for idx, count in enumerate(counts)
        ],
    }


class GpuStatsCollector:
    def __init__(
        self,
        *,
        policy_getter: Callable[[], str] | None = None,
        extra_snapshot: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.counters = dict(DEFAULT_GPU_STATS)
        self.policy_getter = policy_getter or (lambda: os.environ.get("EDGEKV_H1_GPU_POLICY", "vllm_default").strip() or "vllm_default")
        self.extra_snapshot = extra_snapshot
        self.updates = 0
        self.policy_time_ns = 0
        self.eviction_decision_time_ns = 0
        self.free_queue_reorder_time_ns = 0
        self.evicted_score_total = 0.0
        self.evicted_p_reuse_total = 0.0

    def note(self, key: str, amount: int = 1) -> None:
        self.counters[key] = int(self.counters.get(key, 0)) + int(amount)
        self.updates += 1
        flush_interval = max(_env_int("EDGEKV_H1_STATS_FLUSH_INTERVAL", 256), 1)
        if self.updates % flush_interval == 0:
            self.flush()

    def note_policy_time_ns(self, elapsed_ns: int) -> None:
        self.policy_time_ns += int(elapsed_ns)
        self.counters["policy_timing_samples"] += 1

    def note_eviction_decision_time_ns(self, elapsed_ns: int) -> None:
        elapsed_ns = int(elapsed_ns)
        self.eviction_decision_time_ns += elapsed_ns
        self.note_policy_time_ns(elapsed_ns)
        self.counters["eviction_decision_timing_samples"] += 1

    def note_reorder_time_ns(self, elapsed_ns: int) -> None:
        self.free_queue_reorder_time_ns += int(elapsed_ns)

    def reset(self) -> None:
        for key in list(self.counters):
            self.counters[key] = 0
        self.updates = 0
        self.policy_time_ns = 0
        self.eviction_decision_time_ns = 0
        self.free_queue_reorder_time_ns = 0
        self.evicted_score_total = 0.0
        self.evicted_p_reuse_total = 0.0

    def snapshot(self, *, profiles: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        stats: dict[str, Any] = dict(self.counters)
        lookups = int(stats.get("lookup_hits", 0)) + int(stats.get("lookup_misses", 0))
        stats["lookup_total"] = lookups
        stats["block_lookup_hit_rate"] = (stats["lookup_hits"] / lookups) if lookups else 0.0
        native_q = int(stats.get("native_queries", 0) or 0)
        native_h = int(stats.get("native_hits", 0) or 0)
        stats["native_hit_rate"] = (native_h / native_q) if native_q else 0.0
        if native_q:
            stats["hit_rate"] = stats["native_hit_rate"]
            stats["hit_source"] = "vllm_native_token_coverage"
        else:
            stats["hit_rate"] = stats["block_lookup_hit_rate"]
            stats["hit_source"] = "gpu_prefix_cache_block_lookup"
        stats["policy"] = self.policy_getter()
        stats["policy_time_ms_total"] = self.policy_time_ns / 1_000_000.0
        stats["policy_time_us_avg"] = (
            (self.policy_time_ns / 1000.0) / stats["policy_timing_samples"]
            if stats.get("policy_timing_samples") else 0.0
        )
        stats["eviction_decision_time_ms_total"] = self.eviction_decision_time_ns / 1_000_000.0
        stats["eviction_decision_time_us_avg"] = (
            (self.eviction_decision_time_ns / 1000.0) / stats["eviction_decision_timing_samples"]
            if stats.get("eviction_decision_timing_samples") else 0.0
        )
        stats["free_queue_reorder_time_ms"] = self.free_queue_reorder_time_ns / 1_000_000.0
        stats["evicted_score_avg"] = (
            self.evicted_score_total / stats["evicted_score_count"]
            if stats.get("evicted_score_count") else 0.0
        )
        stats["evicted_p_reuse_avg"] = (
            self.evicted_p_reuse_total / stats["evicted_p_reuse_count"]
            if stats.get("evicted_p_reuse_count") else 0.0
        )
        profiles = profiles or []
        scores = [float(profile.get("score", 0.0) or 0.0) for profile in profiles]
        p_reuses = [float(profile.get("p_reuse", 0.0) or 0.0) for profile in profiles]
        c_recomps = [float(profile.get("c_recomp_ms", 0.0) or 0.0) for profile in profiles]
        n_tokens_values = [float(profile.get("n_tokens", 0) or 0) for profile in profiles]
        stats.update({
            "score_min": min(scores) if scores else 0.0,
            "score_p50": _percentile(scores, 0.50),
            "score_p95": _percentile(scores, 0.95),
            "score_mean": sum(scores) / len(scores) if scores else 0.0,
            "score_std": _stddev(scores),
            "p_reuse_min": min(p_reuses) if p_reuses else 0.0,
            "p_reuse_p50": _percentile(p_reuses, 0.50),
            "p_reuse_p95": _percentile(p_reuses, 0.95),
            "p_reuse_std": _stddev(p_reuses),
            "c_recomp_ms_min": min(c_recomps) if c_recomps else 0.0,
            "c_recomp_ms_p50": _percentile(c_recomps, 0.50),
            "c_recomp_ms_p95": _percentile(c_recomps, 0.95),
            "n_tokens_min": min(n_tokens_values) if n_tokens_values else 0.0,
            "n_tokens_p50": _percentile(n_tokens_values, 0.50),
            "n_tokens_p95": _percentile(n_tokens_values, 0.95),
            "score_histogram": _histogram(scores),
            "p_reuse_histogram": _histogram(p_reuses),
            "lpe_profile_count": len(profiles),
            "avg_p_reuse": (sum(p_reuses) / len(p_reuses)) if p_reuses else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
            "cop_object_profile_count": len(profiles),
        })
        if self.extra_snapshot is not None:
            stats.update(self.extra_snapshot(stats))
        stats["pid"] = os.getpid()
        return stats

    def flush(self) -> None:
        stats_dir = _stats_dir()
        if stats_dir is None:
            return
        try:
            stats_dir.mkdir(parents=True, exist_ok=True)
            path = stats_dir / f"edgekv_gpu_stats_{os.getpid()}.json"
            tmp_path = path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(self.snapshot(), sort_keys=True), encoding="utf-8")
            tmp_path.replace(path)
        except Exception:
            pass
