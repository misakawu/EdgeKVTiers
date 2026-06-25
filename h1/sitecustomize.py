"""Local runtime patches for EdgeKVTiers experiment processes."""

from __future__ import annotations

import logging
import os
import atexit
import hashlib
import heapq
import json
import math
import time
from pathlib import Path
from typing import Any

_ORIGINAL_LOGGER_ERROR = logging.Logger.error


def _edgekv_logger_error(self: logging.Logger, msg: object, *args: object, **kwargs: object) -> None:
    """Downgrade vLLM FA2 probing noise on pre-Ampere GPUs.

    RTX 2080 Ti (SM 7.5) cannot use FlashAttention-2, but vLLM still probes it
    before selecting the configured Triton backend. The probe logs at ERROR even
    when execution can continue, which trips the H1 runner's fail-fast monitor.
    """
    if (
        self.name == 'vllm.attention.utils.fa_utils'
        and isinstance(msg, str)
        and msg.startswith('Cannot use FA version')
    ):
        self.warning(msg, *args, **kwargs)
        return
    _ORIGINAL_LOGGER_ERROR(self, msg, *args, **kwargs)


logging.Logger.error = _edgekv_logger_error

try:
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )
except Exception:
    pass


try:
    import prometheus_fastapi_instrumentator.routing as _pfi_routing

    _ORIGINAL_PFI_GET_ROUTE_NAME = _pfi_routing.get_route_name

    def _edgekv_get_route_name(request: Any) -> str:
        try:
            return str(_ORIGINAL_PFI_GET_ROUTE_NAME(request))
        except AttributeError as exc:
            if "'_IncludedRouter' object has no attribute 'path'" not in str(exc):
                raise
            scope = getattr(request, 'scope', {}) or {}
            return str(scope.get('path') or scope.get('root_path') or 'none')

    _pfi_routing.get_route_name = _edgekv_get_route_name
except Exception:
    pass


_EDGEKV_GPU_STATS: dict[str, int] = {
    'lookup_hits': 0,
    'lookup_misses': 0,
    'touches': 0,
    'cached_blocks': 0,
    'evictions': 0,
    'queue_reorders': 0,
    'free_queue_reorder_calls': 0,
    'free_queue_reorder_blocks': 0,
    'free_queue_reorder_skipped': 0,
    'free_queue_reorder_window': 0,
    'admissions': 0,
    'admission_rejections': 0,
    'evict_high_reuse': 0,
    'evict_drop': 0,
    'evicted_score_count': 0,
    'evicted_p_reuse_count': 0,
    'low_score_evictions': 0,
    'hot_prefix_evictions': 0,
    'pinned_eviction_attempts': 0,
    'policy_timing_samples': 0,
    'eviction_decision_timing_samples': 0,
}
_EDGEKV_GPU_LPE_PROFILES: dict[str, dict[str, Any]] = {}
_EDGEKV_GPU_BLOCK_OBJECTS: dict[tuple[int, int], str] = {}
_EDGEKV_GPU_BLOCK_ID_OBJECT_HINTS: dict[int, str] = {}
_EDGEKV_GPU_OBJECT_GROUP_BLOCK_COUNTS: dict[str, dict[int, int]] = {}
_EDGEKV_GPU_OBJECT_SIZE_BYTES: dict[str, int] = {}
_EDGEKV_GPU_GROUP_PAGE_BYTES: dict[int, int] = {}
_EDGEKV_GPU_GROUP_BLOCK_SIZE: dict[int, int] = {}
_EDGEKV_GPU_STATS_UPDATES = 0
_EDGEKV_GPU_POLICY_VALUE = os.environ.get('EDGEKV_H1_GPU_POLICY', 'vllm_default').strip() or 'vllm_default'
_EDGEKV_GPU_POLICY_ENABLED = _EDGEKV_GPU_POLICY_VALUE in {'h1_lru', 'h1_lfu', 'h1_lpe'}
_EDGEKV_GPU_POLICY_IS_LPE = _EDGEKV_GPU_POLICY_VALUE == 'h1_lpe'
_EDGEKV_GPU_POLICY_IS_LRU = _EDGEKV_GPU_POLICY_VALUE == 'h1_lru'
# Only LFU/LPE need per-block ranking state (freq/recency/scores) and free-queue
# reordering. Native vLLM free-queue order already equals LRU eviction order, so
# h1_lru keeps only the diagnostic counters and skips the per-access bookkeeping.
_EDGEKV_GPU_POLICY_NEEDS_RANK_STATE = _EDGEKV_GPU_POLICY_VALUE in {'h1_lfu', 'h1_lpe'}
_EDGEKV_GPU_PROFILE_POLICY_TIME = os.environ.get('EDGEKV_H1_PROFILE_POLICY_TIME', '1').strip().lower() not in {
    '',
    '0',
    'false',
    'no',
    'off',
}
_EDGEKV_GPU_POLICY_TIME_NS = 0
_EDGEKV_GPU_EVICTION_DECISION_TIME_NS = 0
_EDGEKV_GPU_FREE_QUEUE_REORDER_TIME_NS = 0
_EDGEKV_GPU_EVICTED_SCORE_TOTAL = 0.0
_EDGEKV_GPU_EVICTED_P_REUSE_TOTAL = 0.0
_EDGEKV_ENV_FLOAT_CACHE: dict[tuple[str, float], tuple[str | None, float]] = {}
_EDGEKV_ENV_INT_CACHE: dict[tuple[str, int], tuple[str | None, int]] = {}
_EDGEKV_ENV_BOOL_CACHE: dict[tuple[str, bool], tuple[str | None, bool]] = {}


def _edgekv_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    cache_key = (name, default)
    cached = _EDGEKV_ENV_FLOAT_CACHE.get(cache_key)
    if cached is not None and cached[0] == raw:
        return cached[1]
    try:
        value = float(raw if raw is not None else default)
    except (TypeError, ValueError):
        value = default
    _EDGEKV_ENV_FLOAT_CACHE[cache_key] = (raw, value)
    return value


def _edgekv_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    cache_key = (name, default)
    cached = _EDGEKV_ENV_INT_CACHE.get(cache_key)
    if cached is not None and cached[0] == raw:
        return cached[1]
    try:
        value = int(raw if raw is not None else default)
    except (TypeError, ValueError):
        value = default
    _EDGEKV_ENV_INT_CACHE[cache_key] = (raw, value)
    return value


def _edgekv_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    cache_key = (name, default)
    cached = _EDGEKV_ENV_BOOL_CACHE.get(cache_key)
    if cached is not None and cached[0] == value:
        return cached[1]
    parsed = default if value is None else value.strip().lower() not in {'', '0', 'false', 'no', 'off'}
    _EDGEKV_ENV_BOOL_CACHE[cache_key] = (value, parsed)
    return parsed


def _edgekv_stats_dir() -> Path | None:
    value = os.environ.get('EDGEKV_H1_STATS_DIR', '').strip()
    return Path(value) if value else None


def _edgekv_flush_gpu_cache_stats() -> None:
    stats_dir = _edgekv_stats_dir()
    if stats_dir is None:
        return
    try:
        stats_dir.mkdir(parents=True, exist_ok=True)
        path = stats_dir / f'edgekv_gpu_stats_{os.getpid()}.json'
        tmp_path = path.with_suffix('.json.tmp')
        tmp_path.write_text(
            json.dumps(get_edgekv_gpu_cache_stats(), sort_keys=True),
            encoding='utf-8',
        )
        tmp_path.replace(path)
    except Exception:
        pass


def _edgekv_note_gpu_stat(key: str, amount: int = 1) -> None:
    global _EDGEKV_GPU_STATS_UPDATES
    _EDGEKV_GPU_STATS[key] += amount
    _EDGEKV_GPU_STATS_UPDATES += 1
    flush_interval = max(_edgekv_env_int('EDGEKV_H1_STATS_FLUSH_INTERVAL', 256), 1)
    if _EDGEKV_GPU_STATS_UPDATES % flush_interval == 0:
        _edgekv_flush_gpu_cache_stats()


def _edgekv_note_policy_time(start_ns: int) -> None:
    global _EDGEKV_GPU_POLICY_TIME_NS
    if not _EDGEKV_GPU_PROFILE_POLICY_TIME or start_ns <= 0:
        return
    _EDGEKV_GPU_POLICY_TIME_NS += time.perf_counter_ns() - start_ns
    _EDGEKV_GPU_STATS['policy_timing_samples'] += 1


def _edgekv_note_eviction_decision_time(start_ns: int) -> None:
    global _EDGEKV_GPU_EVICTION_DECISION_TIME_NS
    if not _EDGEKV_GPU_PROFILE_POLICY_TIME or start_ns <= 0:
        return
    elapsed = time.perf_counter_ns() - start_ns
    _EDGEKV_GPU_EVICTION_DECISION_TIME_NS += elapsed
    _EDGEKV_GPU_POLICY_TIME_NS += elapsed
    _EDGEKV_GPU_STATS['eviction_decision_timing_samples'] += 1
    _EDGEKV_GPU_STATS['policy_timing_samples'] += 1


def _edgekv_note_reorder_time(start_ns: int) -> None:
    global _EDGEKV_GPU_FREE_QUEUE_REORDER_TIME_NS
    _EDGEKV_GPU_FREE_QUEUE_REORDER_TIME_NS += time.perf_counter_ns() - start_ns


def reset_edgekv_gpu_cache_stats() -> None:
    global _EDGEKV_GPU_STATS_UPDATES, _EDGEKV_GPU_POLICY_TIME_NS, _EDGEKV_GPU_EVICTION_DECISION_TIME_NS
    global _EDGEKV_GPU_FREE_QUEUE_REORDER_TIME_NS
    global _EDGEKV_GPU_EVICTED_SCORE_TOTAL, _EDGEKV_GPU_EVICTED_P_REUSE_TOTAL
    for key in _EDGEKV_GPU_STATS:
        _EDGEKV_GPU_STATS[key] = 0
    _EDGEKV_GPU_LPE_PROFILES.clear()
    _EDGEKV_GPU_BLOCK_OBJECTS.clear()
    _EDGEKV_GPU_BLOCK_ID_OBJECT_HINTS.clear()
    _EDGEKV_GPU_OBJECT_GROUP_BLOCK_COUNTS.clear()
    _EDGEKV_GPU_OBJECT_SIZE_BYTES.clear()
    _EDGEKV_GPU_STATS_UPDATES = 0
    _EDGEKV_GPU_POLICY_TIME_NS = 0
    _EDGEKV_GPU_EVICTION_DECISION_TIME_NS = 0
    _EDGEKV_GPU_FREE_QUEUE_REORDER_TIME_NS = 0
    _EDGEKV_GPU_EVICTED_SCORE_TOTAL = 0.0
    _EDGEKV_GPU_EVICTED_P_REUSE_TOTAL = 0.0
    _EDGEKV_ENV_FLOAT_CACHE.clear()
    _EDGEKV_ENV_INT_CACHE.clear()
    _EDGEKV_ENV_BOOL_CACHE.clear()
    _edgekv_flush_gpu_cache_stats()


def _edgekv_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    index = max(0, min(index, len(ordered) - 1))
    return float(ordered[index])


def _edgekv_stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _edgekv_histogram(values: list[float], buckets: int = 10) -> dict[str, Any]:
    if not values:
        return {'min': 0.0, 'max': 0.0, 'buckets': []}
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return {
            'min': float(low),
            'max': float(high),
            'buckets': [{'lo': float(low), 'hi': float(high), 'count': len(values)}],
        }
    bucket_count = max(int(buckets), 1)
    width = (high - low) / bucket_count
    counts = [0 for _ in range(bucket_count)]
    for value in values:
        index = min(int((value - low) / width), bucket_count - 1)
        counts[index] += 1
    return {
        'min': float(low),
        'max': float(high),
        'buckets': [
            {
                'lo': float(low + (idx * width)),
                'hi': float(low + ((idx + 1) * width)),
                'count': count,
            }
            for idx, count in enumerate(counts)
        ],
    }


def get_edgekv_gpu_cache_stats() -> dict[str, Any]:
    stats: dict[str, Any] = dict(_EDGEKV_GPU_STATS)
    lookups = stats['lookup_hits'] + stats['lookup_misses']
    stats['lookup_total'] = lookups
    stats['hit_rate'] = (stats['lookup_hits'] / lookups) if lookups else 0.0
    stats['policy'] = _edgekv_gpu_policy()
    stats['policy_time_ms_total'] = _EDGEKV_GPU_POLICY_TIME_NS / 1_000_000.0
    stats['policy_time_us_avg'] = (
        (_EDGEKV_GPU_POLICY_TIME_NS / 1000.0) / stats['policy_timing_samples']
        if stats.get('policy_timing_samples') else 0.0
    )
    stats['eviction_decision_time_ms_total'] = _EDGEKV_GPU_EVICTION_DECISION_TIME_NS / 1_000_000.0
    stats['eviction_decision_time_us_avg'] = (
        (_EDGEKV_GPU_EVICTION_DECISION_TIME_NS / 1000.0) / stats['eviction_decision_timing_samples']
        if stats.get('eviction_decision_timing_samples') else 0.0
    )
    stats['free_queue_reorder_time_ms'] = _EDGEKV_GPU_FREE_QUEUE_REORDER_TIME_NS / 1_000_000.0
    stats['evicted_score_avg'] = (
        _EDGEKV_GPU_EVICTED_SCORE_TOTAL / stats['evicted_score_count']
        if stats.get('evicted_score_count') else 0.0
    )
    stats['evicted_p_reuse_avg'] = (
        _EDGEKV_GPU_EVICTED_P_REUSE_TOTAL / stats['evicted_p_reuse_count']
        if stats.get('evicted_p_reuse_count') else 0.0
    )
    profiles = list(_EDGEKV_GPU_LPE_PROFILES.values())
    scores = [float(profile.get('score', 0.0) or 0.0) for profile in profiles]
    p_reuses = [float(profile.get('p_reuse', 0.0) or 0.0) for profile in profiles]
    c_recomps = [float(profile.get('c_recomp_ms', 0.0) or 0.0) for profile in profiles]
    n_tokens_values = [float(profile.get('n_tokens', 0) or 0) for profile in profiles]
    stats['score_min'] = min(scores) if scores else 0.0
    stats['score_p50'] = _edgekv_percentile(scores, 0.50)
    stats['score_p95'] = _edgekv_percentile(scores, 0.95)
    stats['score_mean'] = sum(scores) / len(scores) if scores else 0.0
    stats['score_std'] = _edgekv_stddev(scores)
    stats['p_reuse_min'] = min(p_reuses) if p_reuses else 0.0
    stats['p_reuse_p50'] = _edgekv_percentile(p_reuses, 0.50)
    stats['p_reuse_p95'] = _edgekv_percentile(p_reuses, 0.95)
    stats['p_reuse_std'] = _edgekv_stddev(p_reuses)
    stats['c_recomp_ms_min'] = min(c_recomps) if c_recomps else 0.0
    stats['c_recomp_ms_p50'] = _edgekv_percentile(c_recomps, 0.50)
    stats['c_recomp_ms_p95'] = _edgekv_percentile(c_recomps, 0.95)
    stats['c_recomp_ms_per_token'] = _edgekv_env_float('EDGEKV_C_RE_MS_PER_TOKEN', 0.12)
    stats['c_recomp_model'] = 'linear_c_re_ms_per_token_times_n_tokens'
    stats['n_tokens_min'] = min(n_tokens_values) if n_tokens_values else 0.0
    stats['n_tokens_p50'] = _edgekv_percentile(n_tokens_values, 0.50)
    stats['n_tokens_p95'] = _edgekv_percentile(n_tokens_values, 0.95)
    stats['eviction_granularity'] = 'vllm_prefix_cache_block'
    stats['score_histogram'] = _edgekv_histogram(scores)
    stats['p_reuse_histogram'] = _edgekv_histogram(p_reuses)
    stats['lpe_profile_count'] = len(profiles)
    stats['avg_p_reuse'] = (
        sum(float(profile.get('p_reuse', 0.0)) for profile in profiles) / len(profiles)
        if profiles else 0.0
    )
    stats['avg_score'] = (
        sum(float(profile.get('score', 0.0)) for profile in profiles) / len(profiles)
        if profiles else 0.0
    )
    stats['cop_object_profile_count'] = len(profiles)
    stats['block_object_mapping_count'] = len(_EDGEKV_GPU_BLOCK_OBJECTS)
    resident_size_bytes = sum(int(profile.get('size_bytes', 0) or 0) for profile in profiles)
    resident_blocks = sum(int(profile.get('resident_block_count', 0) or 0) for profile in profiles)
    page_sizes = list(_EDGEKV_GPU_GROUP_PAGE_BYTES.values())
    stats['cop_resident_size_bytes_total'] = resident_size_bytes
    stats['cop_resident_size_mb_total'] = resident_size_bytes / 1024 / 1024
    stats['cop_resident_size_mb_avg'] = (
        stats['cop_resident_size_mb_total'] / len(profiles) if profiles else 0.0
    )
    stats['cop_resident_block_count_total'] = resident_blocks
    stats['kv_cache_group_count'] = len(_EDGEKV_GPU_GROUP_PAGE_BYTES)
    stats['kv_cache_page_size_bytes_min'] = min(page_sizes) if page_sizes else 0
    stats['kv_cache_page_size_bytes_max'] = max(page_sizes) if page_sizes else 0
    if _edgekv_env_int('EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES', 0):
        stats['object_profiles'] = {
            str(profile.get('object_id', object_id)): {
                'object_type': str(profile.get('object_type', 'unknown')),
                'p_reuse': float(profile.get('p_reuse', 0.0) or 0.0),
                'score': float(profile.get('score', 0.0) or 0.0),
                'c_recomp_ms': float(profile.get('c_recomp_ms', 0.0) or 0.0),
                'size_bytes': int(profile.get('size_bytes', 0) or 0),
                'size_mb': float(profile.get('size_mb', 0.0) or 0.0),
                'resident_block_count': int(profile.get('resident_block_count', 0) or 0),
                'resident_block_count_by_group': dict(profile.get('resident_block_count_by_group', {}) or {}),
                'size_source': str(profile.get('size_source', 'unknown')),
            }
            for object_id, profile in _EDGEKV_GPU_LPE_PROFILES.items()
        }
    stats['pid'] = os.getpid()
    return stats


def _edgekv_gpu_policy() -> str:
    return _EDGEKV_GPU_POLICY_VALUE


def _edgekv_gpu_policy_enabled() -> bool:
    return _EDGEKV_GPU_POLICY_ENABLED


def _edgekv_gpu_policy_is_lpe() -> bool:
    return _EDGEKV_GPU_POLICY_IS_LPE


def _edgekv_gpu_policy_needs_rank_state() -> bool:
    return _EDGEKV_GPU_POLICY_NEEDS_RANK_STATE or _EDGEKV_GPU_POLICY_VALUE in {'h1_lfu', 'h1_lpe'}


def _edgekv_init_pool_state(pool: Any) -> None:
    if hasattr(pool, '_edgekv_h1_scores'):
        return
    pool._edgekv_h1_scores = {}
    pool._edgekv_h1_p_reuse = {}
    pool._edgekv_h1_score_update_seq = {}
    pool._edgekv_h1_access_history = {}
    pool._edgekv_h1_pinned = set()
    pool._edgekv_h1_block_objects = {}
    pool._edgekv_h1_freq = {}
    pool._edgekv_h1_recency = {}
    pool._edgekv_h1_blocks = {}
    pool._edgekv_h1_rank_heap = []
    pool._edgekv_h1_rank_version = {}
    pool._edgekv_h1_seq = 0
    pool._edgekv_h1_queue_dirty = True
    pool._edgekv_h1_recent_evictions = 0
    pool._edgekv_h1_last_reorder_seq = 0


def _edgekv_block_id(block: Any) -> int:
    return int(getattr(block, 'block_id'))


def _edgekv_block_key(kv_cache_group_id: int, block_id: int) -> tuple[int, int]:
    return (int(kv_cache_group_id), int(block_id))


def _edgekv_rank_tuple(pool: Any, block_id: int) -> tuple[float, int, int]:
    policy = _EDGEKV_GPU_POLICY_VALUE
    recency = int(getattr(pool, '_edgekv_h1_recency', {}).get(block_id, 0))
    if policy == 'h1_lfu':
        return (float(getattr(pool, '_edgekv_h1_freq', {}).get(block_id, 0)), recency, block_id)
    if policy == 'h1_lpe':
        return (float(getattr(pool, '_edgekv_h1_scores', {}).get(block_id, 0.0)), recency, block_id)
    return (float(recency), 0, block_id)


def _edgekv_heap_touch_block(pool: Any, block: Any) -> None:
    if not _edgekv_gpu_policy_needs_rank_state() or getattr(block, 'is_null', False):
        return
    _edgekv_init_pool_state(pool)
    block_id = _edgekv_block_id(block)
    pool._edgekv_h1_blocks[block_id] = block
    version = int(pool._edgekv_h1_rank_version.get(block_id, 0) or 0) + 1
    pool._edgekv_h1_rank_version[block_id] = version
    heapq.heappush(pool._edgekv_h1_rank_heap, (*_edgekv_rank_tuple(pool, block_id), version))
    pool._edgekv_h1_queue_dirty = True


def _edgekv_heap_drop_block(pool: Any, block_id: int) -> None:
    if not hasattr(pool, '_edgekv_h1_rank_version'):
        return
    pool._edgekv_h1_rank_version.pop(int(block_id), None)
    pool._edgekv_h1_blocks.pop(int(block_id), None)


def _edgekv_heap_valid_block(pool: Any, item: tuple[float, int, int, int]) -> Any | None:
    _score, _recency, block_id, version = item
    if int(getattr(pool, '_edgekv_h1_rank_version', {}).get(block_id, -1)) != int(version):
        return None
    block = getattr(pool, '_edgekv_h1_blocks', {}).get(block_id)
    if block is None or getattr(block, 'is_null', False):
        return None
    return block


def _edgekv_group_id_from_block_hash(block_hash: Any) -> int | None:
    if block_hash is None:
        return None
    try:
        raw = bytes(block_hash)
        if len(raw) < 4:
            return None
        return int.from_bytes(raw[-4:], 'big', signed=False)
    except Exception:
        return None


def _edgekv_register_kv_cache_group(kv_cache_group_id: int, kv_cache_spec: Any) -> None:
    try:
        group_id = int(kv_cache_group_id)
        page_size_bytes = int(getattr(kv_cache_spec, 'page_size_bytes'))
        block_size = int(getattr(kv_cache_spec, 'block_size'))
    except Exception:
        return
    if page_size_bytes <= 0 or block_size <= 0:
        return
    _EDGEKV_GPU_GROUP_PAGE_BYTES[group_id] = page_size_bytes
    _EDGEKV_GPU_GROUP_BLOCK_SIZE[group_id] = block_size


def _edgekv_iter_blocks(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        blocks: list[Any] = []
        for item in value:
            blocks.extend(_edgekv_iter_blocks(item))
        return blocks
    return [value]


def _edgekv_request_meta(request: Any) -> dict[str, Any]:
    sampling_params = getattr(request, 'sampling_params', None)
    extra_args = getattr(sampling_params, 'extra_args', None) or {}
    meta = extra_args.get('edgekv_h1', {})
    return meta if isinstance(meta, dict) else {}


def _edgekv_request_token_ids(request: Any) -> list[int]:
    cached = getattr(request, '_edgekv_h1_token_ids', None)
    if cached is not None:
        return cached
    token_ids = getattr(request, 'all_token_ids', None)
    if token_ids is None:
        token_ids = getattr(request, 'prompt_token_ids', None)
    try:
        cached = [int(token) for token in token_ids]
    except Exception:
        cached = []
    try:
        request._edgekv_h1_token_ids = cached
    except Exception:
        pass
    return cached


def _edgekv_hash_tokens(tokens: list[int]) -> str:
    payload = ','.join(str(token) for token in tokens).encode('utf-8')
    return hashlib.sha1(payload).hexdigest()[:16]


def _edgekv_infer_object_id(
    request: Any,
    block_start: int,
    block_end: int,
) -> tuple[str, str, int]:
    meta = _edgekv_request_meta(request)
    object_id = str(meta.get('object_id') or meta.get('reuse_key') or meta.get('request_id') or '').strip()
    object_type = str(meta.get('object_type') or meta.get('workload') or '').strip()
    n_tokens = int(meta.get('n_tokens', 0) or 0)
    if object_id:
        return object_id, object_type or 'request_meta', max(n_tokens, block_end, 1)

    prefix_len = _edgekv_env_int('H1_PREFIX_REPETITION_PREFIX_LEN', 0)
    tokens = _edgekv_request_token_ids(request) if prefix_len > 0 or _edgekv_env_bool('H1_LPE_HASH_TOKEN_BLOCKS', False) else []
    if prefix_len > 0 and len(tokens) >= prefix_len and block_start < prefix_len:
        cached = getattr(request, '_edgekv_h1_prefix_object', None)
        if cached is not None:
            return cached
        prefix_tokens = tokens[:prefix_len]
        cached = (
            f'prefix:{_edgekv_hash_tokens(prefix_tokens)}',
            'prefix_repetition_prefix',
            max(prefix_len, 1),
        )
        try:
            request._edgekv_h1_prefix_object = cached
        except Exception:
            pass
        return (
            cached[0],
            cached[1],
            cached[2],
        )

    if tokens:
        block_tokens = tokens[block_start:block_end]
        return (
            f'block:{_edgekv_hash_tokens(block_tokens)}',
            'token_block',
            max(len(block_tokens), 1),
        )

    request_id = str(getattr(request, 'request_id', '') or '').strip()
    return (
        f'block:{request_id or "unknown"}:{block_start}:{block_end}',
        'token_block',
        max(block_end - block_start, 1),
    )


def _edgekv_profile_from_values(
    object_id: str,
    object_type: str,
    n_tokens: int,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = meta or {}
    c_re = _edgekv_env_float('EDGEKV_C_RE_MS_PER_TOKEN', 0.12)
    n_tokens = max(int(n_tokens), 1)
    c_recomp_ms = float(meta.get('c_recomp_ms', 0.0) or 0.0) or (c_re * n_tokens)
    c_restore_ms = float(meta.get('c_restore_ms', 0.0) or 0.0)
    risk_exp = float(meta.get('risk_exp', 0.15) or 0.15)
    profile = _EDGEKV_GPU_LPE_PROFILES.setdefault(
        object_id,
        {
            'object_id': object_id,
            'object_type': object_type or 'unknown',
            'n_tokens': n_tokens,
            'p_reuse': float(meta.get('p_reuse', 0.5) or 0.5),
            'c_recomp_ms': c_recomp_ms,
            'c_restore_ms': c_restore_ms,
            'risk_exp': risk_exp,
            'size_bytes': 0,
            'size_mb': 0.0,
            'size_source': 'not_available_trace_only',
            'resident_block_count': 0,
            'resident_block_count_by_group': {},
            'score': 0.0,
            'access_count': 0,
            'hit_count': 0,
            'score_update_seq': 0,
        },
    )
    profile.update(
        {
            'object_type': object_type or str(profile.get('object_type', 'unknown')),
            'n_tokens': n_tokens,
            'c_recomp_ms': c_recomp_ms,
            'c_restore_ms': c_restore_ms,
            'risk_exp': risk_exp,
        }
    )
    if 'p_reuse' in meta:
        profile['p_reuse'] = float(meta.get('p_reuse', profile.get('p_reuse', 0.5)) or 0.5)
    _edgekv_recompute_profile_score(profile)
    return profile


def _edgekv_profile_from_meta(meta: dict[str, Any], block_size: int) -> dict[str, Any]:
    object_id = str(meta.get('object_id') or meta.get('reuse_key') or meta.get('request_id') or '').strip()
    if not object_id:
        object_id = 'unknown'
    n_tokens = max(int(meta.get('n_tokens', block_size) or block_size), 1)
    return _edgekv_profile_from_values(
        object_id,
        str(meta.get('object_type') or meta.get('workload') or 'unknown'),
        n_tokens,
        meta,
    )


def _edgekv_recompute_profile_score(profile: dict[str, Any]) -> None:
    size_mb = float(profile.get('size_mb', 0.0) or 0.0)
    if size_mb <= 0.0:
        profile['score'] = 0.0
        return
    profile['score'] = (
        float(profile.get('p_reuse', 0.5))
        * float(profile.get('c_recomp_ms', 0.0))
        / max(size_mb, 1e-9)
    )


def _edgekv_lpe_reuse_weights() -> tuple[float, float, float, float]:
    w_freq = _edgekv_env_float('H1_LPE_W_FREQ', 0.55)
    w_recency = _edgekv_env_float('H1_LPE_W_RECENCY', 0.30)
    w_type = _edgekv_env_float('H1_LPE_W_TYPE', 0.15)
    return w_freq, w_recency, w_type, max(w_freq + w_recency + w_type, 1e-9)


def _edgekv_clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _edgekv_object_type_prior(object_type: str) -> float:
    normalized = str(object_type or 'unknown').lower()
    if 'prefix' in normalized:
        return 0.80
    if 'session' in normalized:
        return 0.70
    if 'rag' in normalized or 'chunk' in normalized:
        return 0.60
    if 'token_block' in normalized or normalized == 'block':
        return 0.45
    return 0.50


def _edgekv_should_profile_object(object_type: str) -> bool:
    normalized = str(object_type or 'unknown').lower()
    if 'token_block' in normalized or normalized == 'block':
        return _edgekv_env_bool('H1_LPE_PROFILE_TOKEN_BLOCKS', False)
    return True


def _edgekv_refresh_profile_reuse(profile: dict[str, Any]) -> None:
    access_count = max(int(profile.get('access_count', 0) or 0), 1)
    hit_count = int(profile.get('hit_count', 0) or 0)
    misses = max(access_count - hit_count, 0)
    p_freq = _edgekv_clamp(hit_count / access_count, 0.0, 1.0)
    p_recency = 1.0 / (1.0 + math.log1p(misses))
    p_type = _edgekv_object_type_prior(str(profile.get('object_type', 'unknown')))
    w_freq, w_recency, w_type, weight_sum = _edgekv_lpe_reuse_weights()
    p_reuse = (
        (w_freq * p_freq)
        + (w_recency * p_recency)
        + (w_type * p_type)
    ) / weight_sum
    profile['p_freq'] = p_freq
    profile['p_recency'] = p_recency
    profile['p_type'] = p_type
    profile['p_reuse'] = _edgekv_clamp(p_reuse, 0.01, 0.99)
    _edgekv_recompute_profile_score(profile)


def _edgekv_apply_object_resident_delta(
    object_id: str,
    group_id: int,
    block_delta: int,
) -> None:
    object_id = str(object_id)
    group_id = int(group_id)
    if block_delta == 0:
        return
    group_counts = _EDGEKV_GPU_OBJECT_GROUP_BLOCK_COUNTS.setdefault(object_id, {})
    next_count = int(group_counts.get(group_id, 0) or 0) + int(block_delta)
    if next_count > 0:
        group_counts[group_id] = next_count
    else:
        group_counts.pop(group_id, None)
    if not group_counts:
        _EDGEKV_GPU_OBJECT_GROUP_BLOCK_COUNTS.pop(object_id, None)
    page_size_bytes = int(_EDGEKV_GPU_GROUP_PAGE_BYTES.get(group_id, 0) or 0)
    next_size = int(_EDGEKV_GPU_OBJECT_SIZE_BYTES.get(object_id, 0) or 0) + (
        int(block_delta) * page_size_bytes
    )
    if next_size > 0:
        _EDGEKV_GPU_OBJECT_SIZE_BYTES[object_id] = next_size
    else:
        _EDGEKV_GPU_OBJECT_SIZE_BYTES.pop(object_id, None)
        next_size = 0
    _edgekv_refresh_object_resident_profile(object_id)


def _edgekv_refresh_object_resident_profile(object_id: str) -> None:
    profile = _EDGEKV_GPU_LPE_PROFILES.get(str(object_id))
    if profile is None:
        return
    group_counts = _EDGEKV_GPU_OBJECT_GROUP_BLOCK_COUNTS.get(str(object_id), {})
    size_bytes = int(_EDGEKV_GPU_OBJECT_SIZE_BYTES.get(str(object_id), 0) or 0)
    profile['resident_block_count'] = sum(group_counts.values())
    profile['resident_block_count_by_group'] = dict(sorted(group_counts.items()))
    profile['size_bytes'] = int(size_bytes)
    profile['size_mb'] = float(size_bytes) / 1024.0 / 1024.0
    profile['size_source'] = (
        'vllm_kv_cache_spec_page_size_bytes'
        if size_bytes > 0 else 'vllm_kv_cache_spec_page_size_bytes_empty'
    )
    _edgekv_recompute_profile_score(profile)


def _edgekv_recompute_object_resident_size(object_id: str) -> None:
    _edgekv_refresh_object_resident_profile(object_id)


def _edgekv_set_block_object(pool: Any, kv_cache_group_id: int, block_id: int, object_id: str) -> None:
    key = _edgekv_block_key(kv_cache_group_id, block_id)
    object_id = str(object_id)
    old_object_id = _EDGEKV_GPU_BLOCK_OBJECTS.get(key)
    if old_object_id == object_id:
        _EDGEKV_GPU_BLOCK_ID_OBJECT_HINTS[int(block_id)] = object_id
        _edgekv_refresh_object_resident_profile(object_id)
        return
    getattr(pool, '_edgekv_h1_block_objects', {})[key] = object_id
    _EDGEKV_GPU_BLOCK_OBJECTS[key] = object_id
    _EDGEKV_GPU_BLOCK_ID_OBJECT_HINTS[int(block_id)] = object_id
    if old_object_id:
        _edgekv_apply_object_resident_delta(old_object_id, int(kv_cache_group_id), -1)
    _edgekv_apply_object_resident_delta(object_id, int(kv_cache_group_id), 1)


def _edgekv_drop_block_object(pool: Any, kv_cache_group_id: int | None, block_id: int) -> str | None:
    candidate_keys = (
        [_edgekv_block_key(kv_cache_group_id, block_id)]
        if kv_cache_group_id is not None
        else [
            key for key in list(_EDGEKV_GPU_BLOCK_OBJECTS)
            if key[1] == block_id
        ]
    )
    removed_object_id: str | None = None
    for key in candidate_keys:
        object_id = getattr(pool, '_edgekv_h1_block_objects', {}).pop(key, None)
        if object_id is None:
            object_id = _EDGEKV_GPU_BLOCK_OBJECTS.get(key)
        object_id = _EDGEKV_GPU_BLOCK_OBJECTS.pop(key, object_id)
        if object_id is not None:
            removed_object_id = str(object_id)
            _edgekv_apply_object_resident_delta(str(object_id), int(key[0]), -1)
    if not any(key[1] == block_id for key in _EDGEKV_GPU_BLOCK_OBJECTS):
        _EDGEKV_GPU_BLOCK_ID_OBJECT_HINTS.pop(int(block_id), None)
    return removed_object_id


def _edgekv_note_profile_access(profile: dict[str, Any], hit: bool) -> bool:
    profile['access_count'] = int(profile.get('access_count', 0) or 0) + 1
    if hit:
        profile['hit_count'] = int(profile.get('hit_count', 0) or 0) + 1
    profile['last_access_seq'] = int(profile.get('access_count', 0) or 0)
    interval = max(_edgekv_env_int('H1_LPE_SCORE_UPDATE_INTERVAL', 8), 1)
    access_count = int(profile.get('access_count', 0) or 0)
    if access_count == 1 or access_count % interval == 0:
        profile['score_update_seq'] = access_count
        _edgekv_refresh_profile_reuse(profile)
        return True
    return False


def _edgekv_block_profile(
    pool: Any,
    block_id: int,
    kv_cache_group_id: int | None = None,
) -> dict[str, Any] | None:
    object_id = None
    if kv_cache_group_id is not None:
        key = _edgekv_block_key(kv_cache_group_id, block_id)
        object_id = getattr(pool, '_edgekv_h1_block_objects', {}).get(key)
        if object_id is None:
            object_id = _EDGEKV_GPU_BLOCK_OBJECTS.get(key)
    if object_id is None:
        object_id = _EDGEKV_GPU_BLOCK_ID_OBJECT_HINTS.get(int(block_id))
    return _EDGEKV_GPU_LPE_PROFILES.get(str(object_id)) if object_id is not None else None


def _edgekv_free_queue_length(queue: Any) -> int:
    for attr in ('num_free_blocks', 'num_free_block', 'num_blocks'):
        value = getattr(queue, attr, None)
        if callable(value):
            try:
                return int(value())
            except Exception:
                continue
        if value is not None:
            try:
                return int(value)
            except Exception:
                continue
    try:
        return len(queue)
    except Exception:
        return 0


def _edgekv_queue_pressure(pool: Any, queue: Any, num_blocks: int | None = None) -> bool:
    if not _edgekv_env_bool('H1_LPE_LIGHT_PATH', True):
        return True
    if getattr(pool, '_edgekv_h1_queue_dirty', True) is False:
        _edgekv_note_gpu_stat('free_queue_reorder_skipped')
        return False
    free_count = _edgekv_free_queue_length(queue)
    if free_count <= 0:
        return True
    requested = max(int(num_blocks or 1), 1)
    if free_count <= requested:
        return True
    free_ratio_threshold = _edgekv_env_float('H1_LPE_PRESSURE_FREE_RATIO', 0.15)
    total_blocks = 0
    for attr in ('num_gpu_blocks', 'num_cpu_blocks', 'num_blocks'):
        value = getattr(pool, attr, None)
        if callable(value):
            try:
                total_blocks = int(value())
                break
            except Exception:
                continue
        if value is not None:
            try:
                total_blocks = int(value)
                break
            except Exception:
                continue
    if total_blocks > 0 and (free_count / max(total_blocks, 1)) <= free_ratio_threshold:
        return True
    eviction_window = max(_edgekv_env_int('H1_LPE_PRESSURE_EVICTION_WINDOW', 64), 1)
    if int(getattr(pool, '_edgekv_h1_recent_evictions', 0) or 0) >= eviction_window:
        return True
    _edgekv_note_gpu_stat('free_queue_reorder_skipped')
    return False


def _edgekv_reorder_free_queue(pool: Any, num_blocks: int | None = None) -> None:
    if not _EDGEKV_GPU_POLICY_ENABLED or not getattr(pool, 'enable_caching', False):
        return
    _edgekv_init_pool_state(pool)
    _edgekv_note_gpu_stat('free_queue_reorder_calls')
    queue = pool.free_block_queue
    mode = os.environ.get('H1_LPE_REORDER_MODE', 'window').strip().lower() or 'window'
    if mode == 'off' or not _edgekv_queue_pressure(pool, queue, num_blocks):
        return
    start_ns = time.perf_counter_ns()
    try:
        all_blocks = queue.get_all_free_blocks()
    except Exception:
        _edgekv_note_gpu_stat('free_queue_reorder_skipped')
        _edgekv_note_reorder_time(start_ns)
        return
    blocks = [block for block in all_blocks if not getattr(block, 'is_null', False)]
    if len(blocks) < 2:
        _edgekv_note_gpu_stat('free_queue_reorder_skipped')
        _edgekv_note_reorder_time(start_ns)
        return
    requested = max(int(num_blocks or 1), 1)
    base_window = max(_edgekv_env_int('H1_LPE_REORDER_WINDOW', 128), 2)
    window_size = max(base_window, requested * 4)
    select_count = len(blocks) if mode == 'full' else min(window_size, len(blocks))
    queue_block_ids = {_edgekv_block_id(block) for block in blocks}
    for block in blocks:
        block_id = _edgekv_block_id(block)
        pool._edgekv_h1_blocks[block_id] = block
        if block_id not in pool._edgekv_h1_rank_version:
            _edgekv_heap_touch_block(pool, block)
    selected: list[Any] = []
    selected_ids: set[int] = set()
    stale_pops = 0
    while pool._edgekv_h1_rank_heap and len(selected) < select_count:
        item = heapq.heappop(pool._edgekv_h1_rank_heap)
        block = _edgekv_heap_valid_block(pool, item)
        if block is None:
            stale_pops += 1
            continue
        block_id = _edgekv_block_id(block)
        if block_id not in queue_block_ids or block_id in selected_ids:
            stale_pops += 1
            continue
        selected.append(block)
        selected_ids.add(block_id)
    if len(selected) < select_count:
        # Heap can be sparse after a long low-pressure period. Re-seed missing
        # free blocks once, then continue with heap pops rather than sorting.
        for block in blocks:
            block_id = _edgekv_block_id(block)
            if block_id not in selected_ids:
                _edgekv_heap_touch_block(pool, block)
        while pool._edgekv_h1_rank_heap and len(selected) < select_count:
            item = heapq.heappop(pool._edgekv_h1_rank_heap)
            block = _edgekv_heap_valid_block(pool, item)
            if block is None:
                stale_pops += 1
                continue
            block_id = _edgekv_block_id(block)
            if block_id not in queue_block_ids or block_id in selected_ids:
                stale_pops += 1
                continue
            selected.append(block)
            selected_ids.add(block_id)
    if len(selected) < 2:
        _edgekv_note_gpu_stat('free_queue_reorder_skipped')
        _edgekv_note_reorder_time(start_ns)
        return
    _edgekv_note_gpu_stat('free_queue_reorder_blocks', len(selected))
    _EDGEKV_GPU_STATS['free_queue_reorder_window'] = max(
        int(_EDGEKV_GPU_STATS.get('free_queue_reorder_window', 0) or 0),
        len(selected),
    )
    current_prefix = [_edgekv_block_id(block) for block in blocks[:len(selected)]]
    selected_order = [_edgekv_block_id(block) for block in selected]
    if selected_order == current_prefix:
        for block in selected:
            _edgekv_heap_touch_block(pool, block)
        pool._edgekv_h1_queue_dirty = False
        _edgekv_note_reorder_time(start_ns)
        return
    for block in selected:
        queue.remove(block)
    queue.append_n(selected)
    for block in selected:
        _edgekv_heap_touch_block(pool, block)
    pool._edgekv_h1_queue_dirty = False
    pool._edgekv_h1_recent_evictions = 0
    _edgekv_note_gpu_stat('queue_reorders')
    _edgekv_note_reorder_time(start_ns)


def _install_edgekv_gpu_prefix_cache_patch() -> None:
    try:
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
    except Exception:
        return
    if not getattr(SingleTypeKVCacheManager, '_edgekv_h1_manager_patch_installed', False):
        original_manager_init = SingleTypeKVCacheManager.__init__

        def manager_init(
            self: Any,
            kv_cache_spec: Any,
            block_pool: Any,
            kv_cache_group_id: int,
            dcp_world_size: int = 1,
        ) -> None:
            original_manager_init(
                self,
                kv_cache_spec,
                block_pool,
                kv_cache_group_id,
                dcp_world_size,
            )
            _edgekv_register_kv_cache_group(kv_cache_group_id, kv_cache_spec)

        SingleTypeKVCacheManager.__init__ = manager_init
        SingleTypeKVCacheManager._edgekv_h1_manager_patch_installed = True

    if getattr(BlockPool, '_edgekv_h1_gpu_patch_installed', False):
        return

    original_get_cached_block = BlockPool.get_cached_block
    original_cache_full_blocks = BlockPool.cache_full_blocks
    original_get_new_blocks = BlockPool.get_new_blocks
    original_maybe_evict_cached_block = BlockPool._maybe_evict_cached_block
    original_touch = BlockPool.touch
    original_free_blocks = BlockPool.free_blocks

    def get_cached_block(self: Any, block_hash: Any, kv_cache_group_ids: list[int]) -> Any:
        result = original_get_cached_block(self, block_hash, kv_cache_group_ids)
        start_ns = time.perf_counter_ns() if _EDGEKV_GPU_PROFILE_POLICY_TIME else 0
        if _EDGEKV_GPU_POLICY_ENABLED and getattr(self, 'enable_caching', False):
            if result is None:
                _edgekv_note_gpu_stat('lookup_misses')
            else:
                _edgekv_note_gpu_stat('lookup_hits')
                if _EDGEKV_GPU_POLICY_IS_LPE:
                    _edgekv_init_pool_state(self)
                    result_blocks = _edgekv_iter_blocks(result)
                    for group_id, block in zip(kv_cache_group_ids, result_blocks):
                        if getattr(block, 'is_null', False):
                            continue
                        block_id = _edgekv_block_id(block)
                        profile = _edgekv_block_profile(self, block_id, int(group_id))
                        if profile is not None:
                            refreshed = _edgekv_note_profile_access(profile, hit=True)
                            if refreshed:
                                self._edgekv_h1_p_reuse[block_id] = float(profile.get('p_reuse', 0.5))
                                self._edgekv_h1_scores[block_id] = float(profile.get('score', 0.0))
                                self._edgekv_h1_score_update_seq[block_id] = int(
                                    profile.get('score_update_seq', 0) or 0
                                )
                                _edgekv_heap_touch_block(self, block)
        _edgekv_note_policy_time(start_ns)
        return result

    def cache_full_blocks(
        self: Any,
        request: Any,
        blocks: list[Any],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        if not _EDGEKV_GPU_POLICY_ENABLED or not getattr(self, 'enable_caching', False):
            original_cache_full_blocks(
                self,
                request,
                blocks,
                num_cached_blocks,
                num_full_blocks,
                block_size,
                kv_cache_group_id,
            )
            return
        new_full_blocks = list(blocks[num_cached_blocks:num_full_blocks])
        original_cache_full_blocks(
            self,
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            kv_cache_group_id,
        )
        start_ns = time.perf_counter_ns() if _EDGEKV_GPU_PROFILE_POLICY_TIME else 0
        if not _edgekv_gpu_policy_needs_rank_state():
            # h1_lru: native free-queue order already encodes LRU; only keep the
            # admission/cached_blocks diagnostics, skip per-block ranking state.
            for block in new_full_blocks:
                if getattr(block, 'is_null', False):
                    continue
                _edgekv_note_gpu_stat('cached_blocks')
                _edgekv_note_gpu_stat('admissions')
            _edgekv_note_policy_time(start_ns)
            return
        _edgekv_init_pool_state(self)
        meta = _edgekv_request_meta(request)
        pinned = bool(meta.get('is_pinned', False))
        cached_lpe_prefix_profile: dict[str, Any] | None = None
        cached_lpe_prefix_id = ''
        prefix_len = _edgekv_env_int('H1_PREFIX_REPETITION_PREFIX_LEN', 0) if _EDGEKV_GPU_POLICY_IS_LPE else 0
        for offset, block in enumerate(new_full_blocks):
            if getattr(block, 'is_null', False):
                continue
            block_id = _edgekv_block_id(block)
            self._edgekv_h1_seq += 1
            score = float(meta.get('score', 0.0) or 0.0)
            p_reuse = float(meta.get('p_reuse', 0.5) or 0.5)
            if _EDGEKV_GPU_POLICY_IS_LPE:
                block_index = num_cached_blocks + offset
                block_start = block_index * block_size
                block_end = (block_index + 1) * block_size
                if prefix_len > 0 and block_start < prefix_len and cached_lpe_prefix_profile is not None:
                    object_id = cached_lpe_prefix_id
                    profile = cached_lpe_prefix_profile
                else:
                    object_id, object_type, n_tokens = _edgekv_infer_object_id(
                        request,
                        block_start,
                        block_end,
                    )
                    if not _edgekv_should_profile_object(object_type):
                        object_id = ''
                        profile = None
                    else:
                        profile = _edgekv_profile_from_values(object_id, object_type, n_tokens, meta)
                    if prefix_len > 0 and block_start < prefix_len:
                        cached_lpe_prefix_id = object_id
                        cached_lpe_prefix_profile = profile
                if object_id:
                    _edgekv_set_block_object(self, kv_cache_group_id, block_id, object_id)
                if profile is not None:
                    score = float(profile.get('score', score) or score)
                    p_reuse = float(profile.get('p_reuse', p_reuse) or p_reuse)
                    self._edgekv_h1_p_reuse[block_id] = p_reuse
                    self._edgekv_h1_score_update_seq[block_id] = int(
                        profile.get('score_update_seq', 0) or 0
                    )
                if pinned:
                    self._edgekv_h1_pinned.add(block_id)
            self._edgekv_h1_scores[block_id] = score
            self._edgekv_h1_freq[block_id] = max(int(self._edgekv_h1_freq.get(block_id, 0)), 1)
            self._edgekv_h1_recency[block_id] = self._edgekv_h1_seq
            _edgekv_heap_touch_block(self, block)
            _edgekv_note_gpu_stat('cached_blocks')
            _edgekv_note_gpu_stat('admissions')
        _edgekv_note_policy_time(start_ns)

    def get_new_blocks(self: Any, num_blocks: int) -> list[Any]:
        if _edgekv_gpu_policy_needs_rank_state():
            decision_start_ns = time.perf_counter_ns() if _EDGEKV_GPU_PROFILE_POLICY_TIME else 0
            _edgekv_reorder_free_queue(self, num_blocks)
            _edgekv_note_eviction_decision_time(decision_start_ns)
        return original_get_new_blocks(self, num_blocks)

    def _maybe_evict_cached_block(self: Any, block: Any) -> bool:
        evicted = original_maybe_evict_cached_block(self, block)
        if not (evicted and _EDGEKV_GPU_POLICY_ENABLED):
            return evicted
        start_ns = time.perf_counter_ns() if _EDGEKV_GPU_PROFILE_POLICY_TIME else 0
        _edgekv_note_gpu_stat('evictions')
        if not _edgekv_gpu_policy_needs_rank_state():
            # h1_lru: only the eviction counter is needed for diagnostics.
            _edgekv_note_policy_time(start_ns)
            return evicted
        _edgekv_init_pool_state(self)
        block_id = _edgekv_block_id(block)
        score = float(self._edgekv_h1_scores.get(block_id, 0.0) or 0.0)
        p_reuse = float(self._edgekv_h1_p_reuse.get(block_id, 0.0) or 0.0)
        is_pinned = block_id in self._edgekv_h1_pinned
        if _EDGEKV_GPU_POLICY_IS_LPE:
            old_block_hash = getattr(block, 'block_hash', None)
            kv_cache_group_id = _edgekv_group_id_from_block_hash(old_block_hash)
            profile = _edgekv_block_profile(self, block_id, kv_cache_group_id)
            if profile is not None:
                score = float(profile.get('score', score) or score)
                p_reuse = float(profile.get('p_reuse', p_reuse) or p_reuse)
                object_type = str(profile.get('object_type', 'unknown'))
                if 'prefix' in object_type.lower() and p_reuse >= 0.7:
                    _edgekv_note_gpu_stat('hot_prefix_evictions')
            global _EDGEKV_GPU_EVICTED_SCORE_TOTAL, _EDGEKV_GPU_EVICTED_P_REUSE_TOTAL
            _EDGEKV_GPU_EVICTED_SCORE_TOTAL += score
            _EDGEKV_GPU_EVICTED_P_REUSE_TOTAL += p_reuse
            _edgekv_note_gpu_stat('evicted_score_count')
            _edgekv_note_gpu_stat('evicted_p_reuse_count')
            if score <= _edgekv_env_float('H1_LPE_LOW_SCORE_THRESHOLD', 0.0):
                _edgekv_note_gpu_stat('low_score_evictions')
            if is_pinned:
                _edgekv_note_gpu_stat('pinned_eviction_attempts')
            if p_reuse >= 0.5:
                _edgekv_note_gpu_stat('evict_high_reuse')
            else:
                _edgekv_note_gpu_stat('evict_drop')
            _edgekv_drop_block_object(self, kv_cache_group_id, block_id)
        self._edgekv_h1_scores.pop(block_id, None)
        self._edgekv_h1_p_reuse.pop(block_id, None)
        self._edgekv_h1_score_update_seq.pop(block_id, None)
        self._edgekv_h1_freq.pop(block_id, None)
        self._edgekv_h1_recency.pop(block_id, None)
        self._edgekv_h1_access_history.pop(block_id, None)
        self._edgekv_h1_pinned.discard(block_id)
        _edgekv_heap_drop_block(self, block_id)
        self._edgekv_h1_recent_evictions = int(getattr(self, '_edgekv_h1_recent_evictions', 0) or 0) + 1
        self._edgekv_h1_queue_dirty = True
        _edgekv_note_policy_time(start_ns)
        return evicted

    def touch(self: Any, blocks: tuple[list[Any], ...]) -> None:
        start_ns = time.perf_counter_ns() if _EDGEKV_GPU_PROFILE_POLICY_TIME else 0
        if _EDGEKV_GPU_POLICY_ENABLED:
            needs_state = _edgekv_gpu_policy_needs_rank_state()
            if needs_state:
                _edgekv_init_pool_state(self)
            for blocks_per_group in blocks:
                for block in blocks_per_group:
                    if getattr(block, 'is_null', False):
                        continue
                    _edgekv_note_gpu_stat('touches')
                    if needs_state:
                        block_id = _edgekv_block_id(block)
                        self._edgekv_h1_seq += 1
                        self._edgekv_h1_freq[block_id] = int(self._edgekv_h1_freq.get(block_id, 0)) + 1
                        self._edgekv_h1_recency[block_id] = self._edgekv_h1_seq
                        if _EDGEKV_GPU_POLICY_IS_LPE:
                            profile = _edgekv_block_profile(self, block_id)
                            if profile is not None:
                                score_update_seq = int(profile.get('score_update_seq', 0) or 0)
                                if self._edgekv_h1_score_update_seq.get(block_id) != score_update_seq:
                                    self._edgekv_h1_p_reuse[block_id] = float(profile.get('p_reuse', 0.5))
                                    self._edgekv_h1_scores[block_id] = float(profile.get('score', 0.0))
                                    self._edgekv_h1_score_update_seq[block_id] = score_update_seq
                        _edgekv_heap_touch_block(self, block)
        _edgekv_note_policy_time(start_ns)
        return original_touch(self, blocks)

    def free_blocks(self: Any, ordered_blocks: Any) -> None:
        original_free_blocks(self, ordered_blocks)
        if _edgekv_gpu_policy_needs_rank_state():
            _edgekv_init_pool_state(self)
            for block in _edgekv_iter_blocks(ordered_blocks):
                if not getattr(block, 'is_null', False):
                    _edgekv_heap_touch_block(self, block)
            self._edgekv_h1_queue_dirty = True

    BlockPool.get_cached_block = get_cached_block
    BlockPool.cache_full_blocks = cache_full_blocks
    BlockPool.get_new_blocks = get_new_blocks
    BlockPool._maybe_evict_cached_block = _maybe_evict_cached_block
    BlockPool.touch = touch
    BlockPool.free_blocks = free_blocks
    BlockPool._edgekv_h1_gpu_patch_installed = True


_install_edgekv_gpu_prefix_cache_patch()
atexit.register(_edgekv_flush_gpu_cache_stats)
