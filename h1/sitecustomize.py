"""EdgeKVTiers 实验进程的本地运行时补丁。"""

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

try:
    from h1.policies.base import AccessInfo, AdmitInfo, EvictInfo
    from h1.policies.factory import create_policy
    from h1.policies.lpe import object_sort_key as _policy_object_sort_key
    from h1.policies.lpe import DefaultLPEScorer as _DefaultLPEScorer
    from h1.runtime.config import RuntimeConfig
except Exception:
    try:
        from policies.base import AccessInfo, AdmitInfo, EvictInfo
        from policies.factory import create_policy
        from policies.lpe import object_sort_key as _policy_object_sort_key
        from policies.lpe import DefaultLPEScorer as _DefaultLPEScorer
        from runtime.config import RuntimeConfig
    except Exception:
        AccessInfo = None
        AdmitInfo = None
        EvictInfo = None
        create_policy = None
        _policy_object_sort_key = None
        _DefaultLPEScorer = None
        RuntimeConfig = None

_EDGEKV_DEFAULT_LPE_SCORER = None

_ORIGINAL_LOGGER_ERROR = logging.Logger.error


def _edgekv_logger_error(self: logging.Logger, msg: object, *args: object, **kwargs: object) -> None:
    """降低 pre-Ampere GPU 上 vLLM FA2 探测噪声的日志级别。

    RTX 2080 Ti（SM 7.5）不能使用 FlashAttention-2，但 vLLM 在选择已配置的
    Triton backend 之前仍会探测它。即使执行可以继续，该探测也会写 ERROR 日志，
    从而触发 H1 runner 的 fail-fast 监控。
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


def patch_edgekv_vllm_request_metrics() -> bool:
    """把 vLLM v1 请求级时间统计暴露到 RequestOutput.metrics。

    vLLM 0.11 会为日志/追踪保留 RequestStateStats，但 offline LLMEngine 路径不会
    把这些统计传入 RequestOutput。实验 runner 读取 RequestOutput.metrics，因此在
    vLLM 构造每个输出后挂上兼容的 RequestMetrics 对象。
    """
    try:
        from vllm.sequence import RequestMetrics
        from vllm.v1.engine.output_processor import RequestState
    except Exception:
        return False

    original = getattr(RequestState, '_new_request_output', None)
    if original is None:
        return False
    if getattr(original, '_edgekv_request_metrics_patch', False):
        return True

    def _edgekv_new_request_output(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = original(self, *args, **kwargs)
        try:
            stats = getattr(self, 'stats', None)
            if stats is None or getattr(output, 'metrics', None) is not None:
                return output
            scheduled_ts = float(getattr(stats, 'scheduled_ts', 0.0) or 0.0)
            queued_ts = float(getattr(stats, 'queued_ts', 0.0) or 0.0)
            first_token_ts = float(getattr(stats, 'first_token_ts', 0.0) or 0.0)
            last_token_ts = float(getattr(stats, 'last_token_ts', 0.0) or 0.0)
            time_in_queue = (
                max(0.0, scheduled_ts - queued_ts)
                if scheduled_ts > 0.0 and queued_ts > 0.0
                else None
            )
            output.metrics = RequestMetrics(
                arrival_time=float(getattr(stats, 'arrival_time', 0.0) or 0.0),
                last_token_time=last_token_ts,
                first_scheduled_time=scheduled_ts if scheduled_ts > 0.0 else None,
                first_token_time=first_token_ts if first_token_ts > 0.0 else None,
                time_in_queue=time_in_queue,
                finished_time=last_token_ts if getattr(output, 'finished', False) and last_token_ts > 0.0 else None,
            )
        except Exception:
            pass
        return output

    _edgekv_new_request_output._edgekv_request_metrics_patch = True
    RequestState._new_request_output = _edgekv_new_request_output
    return True


_EDGEKV_GPU_STATS: dict[str, int] = {
    'lookup_hits': 0,
    'lookup_misses': 0,
    # vLLM 原生 token 级前缀缓存覆盖率（queries=prompt tokens，
    # hits=已匹配/已计算 tokens）。上面的块级 lookup_hits/lookup_misses
    # 每个请求只计一次 miss（find_longest_cache_hit 在首次分叉处停止），因此其比值是
    # “前缀匹配长度效率”，会被共享前导前缀推高。native_hits/native_queries 才是真实
    # 缓存覆盖命中率，也是现在 hit_rate 报告的值。
    'native_queries': 0,
    'native_hits': 0,
    'native_requests': 0,
    'touches': 0,
    'cached_blocks': 0,
    'evictions': 0,
    'queue_reorders': 0,
    'free_queue_reorder_calls': 0,
    'free_queue_reorder_blocks': 0,
    'free_queue_reorder_skipped': 0,
    'free_queue_reorder_window': 0,
    'admissions': 0,
    # 对象级诊断准入（第一版）：逻辑上的接受/拒绝决策，不会阻止 vLLM 原生缓存。
    # admission_rejections 统计首次触碰 score <= 最低驻留对象 score 的对象；
    # admission_accepts 统计其余对象。
    'admission_rejections': 0,
    'admission_accepts': 0,
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
_EDGEKV_RUNTIME_CONFIG = RuntimeConfig.from_env() if RuntimeConfig is not None else None
_EDGEKV_GPU_POLICY_VALUE = (
    _EDGEKV_RUNTIME_CONFIG.gpu_policy
    if _EDGEKV_RUNTIME_CONFIG is not None
    else (os.environ.get('EDGEKV_H1_GPU_POLICY', 'vllm_default').strip() or 'vllm_default')
)
_EDGEKV_GPU_POLICY_ENABLED = (
    _EDGEKV_RUNTIME_CONFIG.policy_enabled
    if _EDGEKV_RUNTIME_CONFIG is not None
    else _EDGEKV_GPU_POLICY_VALUE in {'h1_lru', 'h1_lfu', 'h1_lpe'}
)
_EDGEKV_GPU_POLICY_IS_LPE = (
    _EDGEKV_RUNTIME_CONFIG.policy_is_lpe
    if _EDGEKV_RUNTIME_CONFIG is not None
    else _EDGEKV_GPU_POLICY_VALUE == 'h1_lpe'
)
_EDGEKV_GPU_POLICY_IS_LRU = (
    _EDGEKV_RUNTIME_CONFIG.policy_is_lru
    if _EDGEKV_RUNTIME_CONFIG is not None
    else _EDGEKV_GPU_POLICY_VALUE == 'h1_lru'
)
# 只有 LFU/LPE 需要块级排序状态（freq/recency/scores）和 free-queue 重排。
# vLLM 原生 free-queue 顺序已经等价于 LRU 驱逐顺序，因此 h1_lru 只保留诊断计数器，
# 跳过每次访问的状态维护。
_EDGEKV_GPU_POLICY_NEEDS_RANK_STATE = (
    _EDGEKV_RUNTIME_CONFIG.policy_needs_rank_state
    if _EDGEKV_RUNTIME_CONFIG is not None
    else _EDGEKV_GPU_POLICY_VALUE in {'h1_lfu', 'h1_lpe'}
)
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
_EDGEKV_LPE_MONITOR_SEQ = 0
_EDGEKV_LPE_ADMISSION_SEQ = 0
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


def _edgekv_lpe_monitor_path() -> Path | None:
    value = os.environ.get('EDGEKV_H1_RUNTIME_MONITOR_PATH', '').strip()
    if value:
        return Path(value)
    if not _edgekv_env_bool('EDGEKV_H1_RUNTIME_MONITOR', False):
        return None
    stats_dir = _edgekv_stats_dir()
    return (stats_dir / 'edgekv_lpe_runtime_monitor.jsonl') if stats_dir is not None else None


def _edgekv_record_lpe_monitor(
    lpe_action: str,
    *,
    block_id: int | None = None,
    kv_cache_group_id: int | None = None,
    profile: dict[str, Any] | None = None,
    hit: bool | None = None,
    **extra: Any,
) -> None:
    if not _EDGEKV_GPU_POLICY_IS_LPE:
        return
    path = _edgekv_lpe_monitor_path()
    if path is None:
        return
    global _EDGEKV_LPE_MONITOR_SEQ
    _EDGEKV_LPE_MONITOR_SEQ += 1
    c_re = _edgekv_env_float('EDGEKV_C_RE_MS_PER_TOKEN', 0.12)
    record: dict[str, Any] = {
        'seq': _EDGEKV_LPE_MONITOR_SEQ,
        'ts': time.time(),
        'pid': os.getpid(),
        'policy': _edgekv_gpu_policy(),
        'lpe_action': str(lpe_action),
        'hit': hit,
        'block_id': int(block_id) if block_id is not None else None,
        'kv_cache_group_id': int(kv_cache_group_id) if kv_cache_group_id is not None else None,
        'c_re': c_re,
        'c_re_ms_per_token': c_re,
    }
    if profile is not None:
        c_recomp_ms = float(profile.get('c_recomp_ms', 0.0) or 0.0)
        record.update({
            'object_id': str(profile.get('object_id', '')),
            'object_type': str(profile.get('object_type', 'unknown')),
            'n_tokens': int(profile.get('n_tokens', 0) or 0),
            'c_recomp': c_recomp_ms,
            'c_recomp_ms': c_recomp_ms,
            'p_reuse': float(profile.get('p_reuse', 0.0) or 0.0),
            'score': float(profile.get('score', 0.0) or 0.0),
            'size_mb': float(profile.get('size_mb', 0.0) or 0.0),
            'logical_size_mb': float(profile.get('logical_size_mb', 0.0) or 0.0),
            'resident_size_mb': float(profile.get('resident_size_mb', 0.0) or 0.0),
            'score_size_source': str(profile.get('score_size_source', 'unknown')),
            'admission_decision': str(profile.get('admission_decision', '') or ''),
            'admission_rejected': bool(profile.get('admission_rejected', False)),
            'resident_block_count': int(profile.get('resident_block_count', 0) or 0),
            'access_count': int(profile.get('access_count', 0) or 0),
            'hit_count': int(profile.get('hit_count', 0) or 0),
        })
    for key, value in extra.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            record[key] = value
        elif isinstance(value, (list, tuple)):
            record[key] = [item for item in value if isinstance(item, (str, int, float, bool)) or item is None]
        else:
            record[key] = str(value)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write('\n')
    except Exception:
        pass


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
    global _EDGEKV_GPU_EVICTION_DECISION_TIME_NS, _EDGEKV_GPU_POLICY_TIME_NS
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
    global _EDGEKV_GPU_FREE_QUEUE_REORDER_TIME_NS, _EDGEKV_LPE_MONITOR_SEQ, _EDGEKV_LPE_ADMISSION_SEQ
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
    _EDGEKV_LPE_MONITOR_SEQ = 0
    _EDGEKV_LPE_ADMISSION_SEQ = 0
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
    # 块级前缀匹配长度效率（仅诊断；由于 find_longest_cache_hit 每请求只记一次 miss，
    # 该值会被共享前导前缀推高）。
    stats['block_lookup_hit_rate'] = (stats['lookup_hits'] / lookups) if lookups else 0.0
    # vLLM 原生 token 级覆盖率 = 真实缓存命中率。
    native_q = stats.get('native_queries', 0)
    native_h = stats.get('native_hits', 0)
    stats['native_hit_rate'] = (native_h / native_q) if native_q else 0.0
    if native_q:
        stats['hit_rate'] = stats['native_hit_rate']
        stats['hit_source'] = 'vllm_native_token_coverage'
    else:
        stats['hit_rate'] = stats['block_lookup_hit_rate']
        stats['hit_source'] = 'gpu_prefix_cache_block_lookup'
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
    # 逻辑大小（score 分母）与驻留大小（诊断用途）的聚合值，方便读者确认 score 不再
    # 被残留块虚高。
    logical_sizes = [float(profile.get('logical_size_mb', 0.0) or 0.0) for profile in profiles]
    resident_sizes = [float(profile.get('resident_size_mb', 0.0) or 0.0) for profile in profiles]
    stats['cop_logical_size_mb_total'] = sum(logical_sizes)
    stats['cop_logical_size_mb_avg'] = (sum(logical_sizes) / len(profiles)) if profiles else 0.0
    stats['cop_resident_size_mb_from_profiles_total'] = sum(resident_sizes)
    score_size_source_counts: dict[str, int] = {}
    for profile in profiles:
        source = str(profile.get('score_size_source', 'unknown') or 'unknown')
        score_size_source_counts[source] = score_size_source_counts.get(source, 0) + 1
    stats['score_size_source_counts'] = dict(sorted(score_size_source_counts.items()))
    # 对象级诊断准入摘要（第一版）。
    stats['admission_mode'] = _edgekv_admission_mode()
    stats['admission_accept_count'] = int(stats.get('admission_accepts', 0) or 0)
    stats['admission_rejection_count'] = int(stats.get('admission_rejections', 0) or 0)
    admission_decisions = int(stats['admission_accept_count'] + stats['admission_rejection_count'])
    stats['admission_decision_count'] = admission_decisions
    stats['admission_rejection_rate'] = (
        stats['admission_rejection_count'] / admission_decisions if admission_decisions else 0.0
    )
    stats['kv_cache_group_count'] = len(_EDGEKV_GPU_GROUP_PAGE_BYTES)
    stats['kv_cache_page_size_bytes_min'] = min(page_sizes) if page_sizes else 0
    stats['kv_cache_page_size_bytes_max'] = max(page_sizes) if page_sizes else 0
    if _edgekv_env_int('EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES', 0):
        stats['object_profiles'] = {
            str(profile.get('object_id', object_id)): {
                'object_type': str(profile.get('object_type', 'unknown')),
                'p_reuse': float(profile.get('p_reuse', 0.0) or 0.0),
                'score': float(profile.get('score', 0.0) or 0.0),
                'n_tokens': int(profile.get('n_tokens', 0) or 0),
                'c_re': _edgekv_env_float('EDGEKV_C_RE_MS_PER_TOKEN', 0.12),
                'c_recomp': float(profile.get('c_recomp_ms', 0.0) or 0.0),
                'c_recomp_ms': float(profile.get('c_recomp_ms', 0.0) or 0.0),
                'size_bytes': int(profile.get('size_bytes', 0) or 0),
                'size_mb': float(profile.get('size_mb', 0.0) or 0.0),
                'logical_size_mb': float(profile.get('logical_size_mb', 0.0) or 0.0),
                'resident_size_mb': float(profile.get('resident_size_mb', 0.0) or 0.0),
                'bytes_per_token': float(profile.get('bytes_per_token', 0.0) or 0.0),
                'resident_block_count': int(profile.get('resident_block_count', 0) or 0),
                'resident_block_count_by_group': dict(profile.get('resident_block_count_by_group', {}) or {}),
                'size_source': str(profile.get('size_source', 'unknown')),
                'score_size_source': str(profile.get('score_size_source', 'unknown')),
                'admission_decision': str(profile.get('admission_decision', '') or ''),
                'admission_rejected': bool(profile.get('admission_rejected', False)),
                'admission_min_resident_score': profile.get('admission_min_resident_score', ''),
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


def _edgekv_object_sort_key(object_id: str) -> int:
    """生成稳定的对象级整数 key，使同一对象的块在排序中保持连续。

    对象级驱逐（h1_lpe）按所属对象的 score 给每个块排序。两个不同对象可能拥有相同
    score；如果没有对象级 tie-breaker，它们的块会按 block_id 交错，无法形成连续的
    驱逐组。对 object_id 做 hash 可得到确定且对象稳定的 key，使同一对象的块在
    rank 顺序中相邻。
    """
    if _policy_object_sort_key is not None:
        return int(_policy_object_sort_key(object_id))
    digest = hashlib.sha1(str(object_id).encode('utf-8')).hexdigest()[:12]
    return int(digest, 16)


_EDGEKV_POLICY_INSTANCE: Any = None
_EDGEKV_POLICY_INSTANCE_READY = False


def _edgekv_policy_instance() -> Any:
    """惰性获取并缓存本进程的策略实例（工厂只调用一次）。

    rank_tuple 位于驱逐热路径（heap push / compact / reorder），每次调用都新建
    策略实例会带来无谓的对象/闭包分配，并污染 policy_time 计时。进程内策略由
    EDGEKV_H1_GPU_POLICY 固定不变，因此缓存单例即可；block_profile 直接传函数
    引用（与旧 lambda 等价）。
    """
    global _EDGEKV_POLICY_INSTANCE, _EDGEKV_POLICY_INSTANCE_READY
    if not _EDGEKV_POLICY_INSTANCE_READY:
        _EDGEKV_POLICY_INSTANCE = (
            create_policy(_EDGEKV_GPU_POLICY_VALUE, block_profile=_edgekv_block_profile)
            if create_policy is not None
            else None
        )
        _EDGEKV_POLICY_INSTANCE_READY = True
    return _EDGEKV_POLICY_INSTANCE


def _edgekv_rank_tuple(pool: Any, block_id: int) -> tuple[float, ...]:
    policy = _edgekv_policy_instance()
    if policy is not None:
        return policy.rank_tuple(pool, int(block_id))
    recency = int(getattr(pool, '_edgekv_h1_recency', {}).get(block_id, 0))
    return (float(recency), 0, block_id)


def _edgekv_policy_on_admit(
    pool: Any,
    block_id: int,
    *,
    score: float = 0.0,
    p_reuse: float = 0.5,
    pinned: bool = False,
    profile: dict[str, Any] | None = None,
) -> None:
    policy = _edgekv_policy_instance()
    if policy is not None and AdmitInfo is not None:
        policy.on_admit(pool, int(block_id), AdmitInfo(score=score, p_reuse=p_reuse, pinned=pinned, profile=profile))
        return
    pool._edgekv_h1_seq += 1
    pool._edgekv_h1_scores[int(block_id)] = float(score)
    pool._edgekv_h1_freq[int(block_id)] = max(int(pool._edgekv_h1_freq.get(int(block_id), 0)), 1)
    pool._edgekv_h1_recency[int(block_id)] = pool._edgekv_h1_seq
    if pinned:
        pool._edgekv_h1_pinned.add(int(block_id))


def _edgekv_policy_on_access(
    pool: Any,
    block_id: int,
    *,
    profile: dict[str, Any] | None = None,
    refreshed: bool = False,
) -> None:
    policy = _edgekv_policy_instance()
    if policy is not None and AccessInfo is not None:
        policy.on_access(pool, int(block_id), AccessInfo(profile=profile, refreshed=refreshed))
        return
    pool._edgekv_h1_seq += 1
    pool._edgekv_h1_freq[int(block_id)] = int(pool._edgekv_h1_freq.get(int(block_id), 0)) + 1
    pool._edgekv_h1_recency[int(block_id)] = pool._edgekv_h1_seq


def _edgekv_policy_refresh_block_score(pool: Any, block_id: int, profile: dict[str, Any]) -> None:
    policy = _edgekv_policy_instance()
    if policy is not None:
        policy.refresh_block_score(pool, int(block_id), profile)
        return
    pool._edgekv_h1_p_reuse[int(block_id)] = float(profile.get('p_reuse', 0.5))
    pool._edgekv_h1_scores[int(block_id)] = float(profile.get('score', 0.0))
    pool._edgekv_h1_score_update_seq[int(block_id)] = int(profile.get('score_update_seq', 0) or 0)


def _edgekv_policy_on_evict(
    pool: Any,
    block_id: int,
    *,
    profile: dict[str, Any] | None = None,
    score: float = 0.0,
    p_reuse: float = 0.0,
) -> None:
    policy = _edgekv_policy_instance()
    if policy is not None and EvictInfo is not None:
        policy.on_evict(pool, int(block_id), EvictInfo(profile=profile, score=score, p_reuse=p_reuse))
        return
    pool._edgekv_h1_scores.pop(int(block_id), None)
    pool._edgekv_h1_p_reuse.pop(int(block_id), None)
    pool._edgekv_h1_score_update_seq.pop(int(block_id), None)
    pool._edgekv_h1_freq.pop(int(block_id), None)
    pool._edgekv_h1_recency.pop(int(block_id), None)
    pool._edgekv_h1_access_history.pop(int(block_id), None)
    pool._edgekv_h1_pinned.discard(int(block_id))


def _edgekv_heap_touch_block(pool: Any, block: Any) -> None:
    if not _edgekv_gpu_policy_needs_rank_state() or getattr(block, 'is_null', False):
        return
    _edgekv_init_pool_state(pool)
    block_id = _edgekv_block_id(block)
    pool._edgekv_h1_blocks[block_id] = block
    version = int(pool._edgekv_h1_rank_version.get(block_id, 0) or 0) + 1
    pool._edgekv_h1_rank_version[block_id] = version
    # 堆元素 = (rank_tuple, version, block_id)。rank_tuple 不透明，长度可能随策略变化
    # （LPE 是对象级）；将 block_id 放在 rank tuple 外，可统一作为最终身份/tie-break
    # 字段。
    heapq.heappush(
        pool._edgekv_h1_rank_heap,
        (_edgekv_rank_tuple(pool, block_id), version, block_id),
    )
    pool._edgekv_h1_queue_dirty = True


def _edgekv_heap_drop_block(pool: Any, block_id: int) -> None:
    if not hasattr(pool, '_edgekv_h1_rank_version'):
        return
    pool._edgekv_h1_rank_version.pop(int(block_id), None)
    pool._edgekv_h1_blocks.pop(int(block_id), None)


def _edgekv_heap_valid_block(pool: Any, item: tuple[Any, int, int]) -> Any | None:
    _rank_tuple, version, block_id = item
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
    p_reuse_prior = _edgekv_meta_p_reuse_prior(meta)
    initial_p_reuse = float(meta.get('p_reuse', 0.0) or 0.0)
    if initial_p_reuse <= 0.0:
        initial_p_reuse = p_reuse_prior if p_reuse_prior is not None else 0.5
    profile = _EDGEKV_GPU_LPE_PROFILES.setdefault(
        object_id,
        {
            'object_id': object_id,
            'object_type': object_type or 'unknown',
            'object_sort_key': _edgekv_object_sort_key(object_id),
            'n_tokens': n_tokens,
            'p_reuse': initial_p_reuse,
            'p_reuse_prior': p_reuse_prior if p_reuse_prior is not None else '',
            'c_recomp_ms': c_recomp_ms,
            'c_restore_ms': c_restore_ms,
            'risk_exp': risk_exp,
            'bytes_per_token': 0.0,
            'size_bytes': 0,
            'size_mb': 0.0,
            'size_source': 'not_available_trace_only',
            'logical_size_mb': 0.0,
            'resident_size_mb': 0.0,
            'resident_size_source': 'not_available_trace_only',
            'score_size_source': 'fallback_theoretical_mu_kv',
            'resident_block_count': 0,
            'resident_block_count_by_group': {},
            'score': 0.0,
            'access_count': 0,
            'hit_count': 0,
            'score_update_seq': 0,
            'admission_decision': '',
            'admission_rejected': False,
            'admission_seq': 0,
            'admission_min_resident_score': '',
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
    if p_reuse_prior is not None:
        profile['p_reuse_prior'] = p_reuse_prior
    if 'p_reuse' in meta:
        profile['p_reuse'] = float(meta.get('p_reuse', profile.get('p_reuse', 0.5)) or 0.5)
    elif p_reuse_prior is not None:
        profile['p_reuse'] = p_reuse_prior
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


def _edgekv_group_bytes_per_token(group_id: int) -> float:
    page = int(_EDGEKV_GPU_GROUP_PAGE_BYTES.get(int(group_id), 0) or 0)
    block_size = int(_EDGEKV_GPU_GROUP_BLOCK_SIZE.get(int(group_id), 0) or 0)
    if page > 0 and block_size > 0:
        return float(page) / float(block_size)
    return 0.0


def _edgekv_lpe_scorer():
    """返回当前生效的 LPE 打分器：LPE 模式用策略实例的 scorer，否则用模块级默认。"""
    global _EDGEKV_DEFAULT_LPE_SCORER
    policy = _edgekv_policy_instance()
    scorer = getattr(policy, 'scorer', None) if policy is not None else None
    if scorer is not None:
        return scorer
    if _DefaultLPEScorer is None:
        return None
    if _EDGEKV_DEFAULT_LPE_SCORER is None:
        _EDGEKV_DEFAULT_LPE_SCORER = _DefaultLPEScorer()
    return _EDGEKV_DEFAULT_LPE_SCORER


def _edgekv_recompute_profile_score(profile: dict[str, Any]) -> None:
    """重算 LPE score（委托打分器；算式见 policies/lpe.py DefaultLPEScorer）。"""
    scorer = _edgekv_lpe_scorer()
    if scorer is not None:
        scorer.recompute_score(profile)
        return
    # 兜底：policies 导入失败时的原始内联实现（与 DefaultLPEScorer 逐字一致）。
    n_tokens = max(float(profile.get('n_tokens', 1) or 1), 1.0)
    bytes_per_token = float(profile.get('bytes_per_token', 0.0) or 0.0)
    if bytes_per_token > 0.0:
        logical_size_mb = (n_tokens * bytes_per_token) / 1024.0 / 1024.0
        score_size_source = 'vllm_kv_cache_spec_bytes_per_token'
    else:
        logical_size_mb = _edgekv_env_float('EDGEKV_MU_KV_MB_PER_TOKEN', 0.12) * n_tokens
        score_size_source = 'fallback_theoretical_mu_kv'
    profile['logical_size_mb'] = logical_size_mb
    profile['score_size_source'] = score_size_source

    resident_bytes = int(profile.get('size_bytes', 0) or 0)
    profile['resident_size_mb'] = float(resident_bytes) / 1024.0 / 1024.0

    profile['size_mb'] = logical_size_mb
    profile['size_source'] = score_size_source

    if logical_size_mb <= 0.0:
        profile['score'] = 0.0
        return
    profile['score'] = (
        float(profile.get('p_reuse', 0.5))
        * float(profile.get('c_recomp_ms', 0.0))
        / max(logical_size_mb, 1e-9)
    )


def _edgekv_lpe_reuse_weights() -> tuple[float, float, float, float]:
    w_freq = _edgekv_env_float('H1_LPE_W_FREQ', 0.55)
    w_recency = _edgekv_env_float('H1_LPE_W_RECENCY', 0.30)
    w_type = _edgekv_env_float('H1_LPE_W_TYPE', 0.15)
    return w_freq, w_recency, w_type, max(w_freq + w_recency + w_type, 1e-9)


def _edgekv_p_reuse_prior_weight() -> float:
    return _edgekv_clamp(_edgekv_env_float('H1_LPE_W_PRIOR', 0.70), 0.0, 1.0)


def _edgekv_meta_p_reuse_prior(meta: dict[str, Any]) -> float | None:
    if 'p_reuse_prior' not in meta:
        return None
    try:
        return _edgekv_clamp(float(meta.get('p_reuse_prior')), 0.01, 0.99)
    except (TypeError, ValueError):
        return None


def _edgekv_clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _edgekv_object_type_prior(object_type: str) -> float:
    normalized = str(object_type or 'unknown').lower()
    if 'hot' in normalized:
        return 0.90
    if 'cold' in normalized:
        return 0.10
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
    p_reuse_prior = None
    if profile.get('p_reuse_prior', '') != '':
        try:
            p_reuse_prior = _edgekv_clamp(float(profile.get('p_reuse_prior')), 0.01, 0.99)
        except (TypeError, ValueError):
            p_reuse_prior = None
    if p_reuse_prior is not None:
        prior_weight = _edgekv_p_reuse_prior_weight()
        p_reuse = (prior_weight * p_reuse_prior) + ((1.0 - prior_weight) * p_reuse)
    profile['p_freq'] = p_freq
    profile['p_recency'] = p_recency
    profile['p_type'] = p_type
    if p_reuse_prior is not None:
        profile['p_reuse_prior'] = p_reuse_prior
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
    profile['resident_size_source'] = (
        'vllm_kv_cache_spec_page_size_bytes'
        if size_bytes > 0 else 'vllm_kv_cache_spec_page_size_bytes_empty'
    )
    # 一旦知道对象所在 group，就用 vLLM 真实的每 token KV 字节数确定逻辑大小；
    # 否则回退到理论 mu_kv。
    if float(profile.get('bytes_per_token', 0.0) or 0.0) <= 0.0:
        for gid in group_counts:
            bytes_per_token = _edgekv_group_bytes_per_token(gid)
            if bytes_per_token > 0.0:
                profile['bytes_per_token'] = bytes_per_token
                break
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


def _edgekv_admission_mode() -> str:
    """对象级准入模式。第一版只实现 'diagnostic'。

    diagnostic：计算接受/拒绝，但永不阻止 vLLM 原生缓存。
    strict：预留给第二版（对拒绝对象跳过/缩短原生缓存）；这里按 diagnostic 处理，
    因此在 strict 路径被明确构建并单独加门控之前，该标志不会产生效果。
    """
    value = os.environ.get('H1_LPE_ADMISSION_MODE', 'diagnostic').strip().lower() or 'diagnostic'
    return 'strict' if value == 'strict' else 'diagnostic'


def _edgekv_min_resident_object_score(exclude_object_id: str | None = None) -> float | None:
    """当前驻留对象（仍有块在缓存中）的最低 score。

    排除 ``exclude_object_id``（准入候选），使新对象与其他驻留对象比较，而不是与
    自己比较。当缓存中没有其他驻留对象时返回 None。
    """
    min_score: float | None = None
    for object_id, profile in _EDGEKV_GPU_LPE_PROFILES.items():
        if exclude_object_id is not None and object_id == exclude_object_id:
            continue
        if int(profile.get('resident_block_count', 0) or 0) <= 0:
            continue
        score = float(profile.get('score', 0.0) or 0.0)
        if min_score is None or score < min_score:
            min_score = score
    return min_score


def _edgekv_evaluate_object_admission(
    profile: dict[str, Any],
    pinned: bool,
) -> str | None:
    """对象级诊断准入决策（第一版）。

    每个对象首次准入时只决策一次：如果新对象 score 高于最低驻留对象 score
    （或缓存为空，或对象被 pinned）则接受，否则拒绝。该决策纯诊断用途，不会阻止
    vLLM 缓存 block；它只标记 profile，并影响对象级驱逐排序（相同 score 下拒绝对象
    先驱逐）。
    """
    if profile is None or profile.get('admission_decision'):
        return None
    global _EDGEKV_LPE_ADMISSION_SEQ
    _EDGEKV_LPE_ADMISSION_SEQ += 1
    object_id = str(profile.get('object_id', ''))
    new_score = float(profile.get('score', 0.0) or 0.0)
    min_resident = _edgekv_min_resident_object_score(exclude_object_id=object_id)
    if pinned or min_resident is None or new_score > min_resident:
        decision = 'accept'
    else:
        decision = 'reject'
    profile['admission_decision'] = decision
    profile['admission_seq'] = _EDGEKV_LPE_ADMISSION_SEQ
    profile['admission_min_resident_score'] = (
        float(min_resident) if min_resident is not None else ''
    )
    profile['admission_rejected'] = (decision == 'reject')
    if decision == 'reject':
        _edgekv_note_gpu_stat('admission_rejections')
    else:
        _edgekv_note_gpu_stat('admission_accepts')
    return decision


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


def _edgekv_block_in_free_queue(block: Any) -> bool:
    """vLLM free-block 链表的 O(1) 成员测试。

    vLLM 在 popleft_n()/remove() 时清空两个链表指针，并在 append()/append_n() 时
    重新链接，因此非空的 prev+next 指针对唯一标记一个 block 当前驻留在 free queue
    中。这让 reorder 能在不物化整个队列的情况下验证 heap 候选。
    """
    return (
        getattr(block, 'prev_free_block', None) is not None
        and getattr(block, 'next_free_block', None) is not None
    )


def _edgekv_free_queue_head_window(queue: Any, limit: int) -> list[Any]:
    """从 free queue 头部返回最多 ``limit`` 个真实 block。

    最多遍历 ``limit`` 个节点（fake tail 的 next_free_block=None），因此无论 free block
    总数多少，成本都是 O(limit)。它既用于把尚未追踪的头部 block 放入 rank heap，
    也用于检测 no-op move。
    """
    head = getattr(queue, 'fake_free_list_head', None)
    if head is None or limit <= 0:
        return []
    out: list[Any] = []
    node = getattr(head, 'next_free_block', None)
    while node is not None and getattr(node, 'next_free_block', None) is not None and len(out) < limit:
        if not getattr(node, 'is_null', False):
            out.append(node)
        node = node.next_free_block
    return out


def _edgekv_compact_rank_heap(pool: Any) -> None:
    """用 live version 重建 rank heap，丢弃陈旧的 lazy entry。

    每次 touch/cache/free 都会推入新的 heap entry，而不会删除旧 entry，因此 heap 会随
    访问次数增长。周期性压缩将 heappop 成本限制在被追踪（驻留）block 数量内，而不是
    总负载内，从而在吞吐上升时保持 reorder 延迟平稳。
    """
    versions = getattr(pool, '_edgekv_h1_rank_version', {})
    blocks = getattr(pool, '_edgekv_h1_blocks', {})
    new_heap: list[tuple[Any, int, int]] = []
    for block_id, version in versions.items():
        block = blocks.get(block_id)
        if block is None or getattr(block, 'is_null', False):
            continue
        new_heap.append((_edgekv_rank_tuple(pool, block_id), int(version), int(block_id)))
    heapq.heapify(new_heap)
    pool._edgekv_h1_rank_heap = new_heap


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
    free_count = _edgekv_free_queue_length(queue)
    if free_count < 2:
        _edgekv_note_gpu_stat('free_queue_reorder_skipped')
        _edgekv_note_reorder_time(start_ns)
        return
    requested = max(int(num_blocks or 1), 1)
    base_window = max(_edgekv_env_int('H1_LPE_REORDER_WINDOW', 128), 2)
    window_size = max(base_window, requested * 4)
    select_count = free_count if mode == 'full' else min(window_size, free_count)

    # 增量选择（D2）：rank heap 会在每次 cache/touch/free/evict 时保持更新，因此
    # reorder 不再扫描整个 free queue。这里只读取 O(select_count) 的头部窗口，用于把
    # 尚未进入 heap 的头部 block（例如从未触碰的原始 block）放入 heap，并检测 no-op；
    # 随后直接从 heap 弹出最低 rank 候选。每次调用总成本为
    # O(select_count + stale_pops)，与 free block 总数无关，替代之前的 O(num_free)
    # 全量重扫。
    head_window = _edgekv_free_queue_head_window(queue, select_count)
    for block in head_window:
        block_id = _edgekv_block_id(block)
        pool._edgekv_h1_blocks[block_id] = block
        if block_id not in pool._edgekv_h1_rank_version:
            _edgekv_heap_touch_block(pool, block)

    # 随负载上升，避免 lazy-deletion heap 被陈旧 entry 膨胀。
    compact_floor = max(_edgekv_env_int('H1_LPE_HEAP_COMPACT_MIN', 1024), 2)
    if len(pool._edgekv_h1_rank_heap) > max(compact_floor, 4 * len(pool._edgekv_h1_rank_version)):
        _edgekv_compact_rank_heap(pool)

    selected: list[Any] = []
    selected_ids: set[int] = set()
    stale_pops = 0
    heap = pool._edgekv_h1_rank_heap
    while heap and len(selected) < select_count:
        item = heapq.heappop(heap)
        block = _edgekv_heap_valid_block(pool, item)
        if block is None:
            stale_pops += 1
            continue
        block_id = _edgekv_block_id(block)
        if block_id in selected_ids or not _edgekv_block_in_free_queue(block):
            stale_pops += 1
            continue
        selected.append(block)
        selected_ids.add(block_id)
    if _EDGEKV_GPU_POLICY_IS_LPE:
        for rank, block in enumerate(selected):
            block_id = _edgekv_block_id(block)
            _edgekv_record_lpe_monitor(
                'reorder_candidate',
                block_id=block_id,
                profile=_edgekv_block_profile(pool, block_id),
                selected_rank=rank,
                hit=None,
            )
    if len(selected) < 2:
        # Heap 没有可用 free 候选；把已弹出的元素推回后直接返回，不回退到全队列扫描。
        for block in selected:
            _edgekv_heap_touch_block(pool, block)
        _edgekv_note_gpu_stat('free_queue_reorder_skipped')
        _edgekv_note_reorder_time(start_ns)
        return
    _edgekv_note_gpu_stat('free_queue_reorder_blocks', len(selected))
    _EDGEKV_GPU_STATS['free_queue_reorder_window'] = max(
        int(_EDGEKV_GPU_STATS.get('free_queue_reorder_window', 0) or 0),
        len(selected),
    )
    current_prefix = [_edgekv_block_id(block) for block in head_window[:len(selected)]]
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


def _install_edgekv_native_hit_rate_patch() -> None:
    """把 vLLM 原生 token 级前缀缓存覆盖率镜像到 edgekv stats。

    vLLM 只在 KVCacheManager.log_stats 开启时累积 queries/hits，因此这里包装
    get_computed_blocks 并无条件记录：native_queries += prompt tokens，
    native_hits += matched（computed）tokens。这是真实缓存命中率，不同于在首次分叉处
    停止的块级 lookup_hits/misses。
    """
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
    except Exception:
        return
    if getattr(KVCacheManager, '_edgekv_h1_native_hit_patch_installed', False):
        return
    original_get_computed_blocks = KVCacheManager.get_computed_blocks

    def get_computed_blocks(self: Any, request: Any) -> Any:
        result = original_get_computed_blocks(self, request)
        if _EDGEKV_GPU_POLICY_ENABLED and getattr(self, 'enable_caching', False):
            try:
                num_tokens = int(getattr(request, 'num_tokens', 0) or 0)
                _, num_new_computed_tokens = result
                _edgekv_note_gpu_stat('native_queries', num_tokens)
                _edgekv_note_gpu_stat('native_hits', int(num_new_computed_tokens or 0))
                _edgekv_note_gpu_stat('native_requests')
            except Exception:
                pass
        return result

    KVCacheManager.get_computed_blocks = get_computed_blocks
    KVCacheManager._edgekv_h1_native_hit_patch_installed = True


def _install_edgekv_gpu_prefix_cache_patch() -> None:
    _install_edgekv_native_hit_rate_patch()
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
                _edgekv_record_lpe_monitor(
                    'lookup_miss',
                    hit=False,
                    kv_cache_group_ids=[int(group_id) for group_id in kv_cache_group_ids],
                )
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
                            _edgekv_record_lpe_monitor(
                                'lookup_hit',
                                block_id=block_id,
                                kv_cache_group_id=int(group_id),
                                profile=profile,
                                hit=True,
                                score_refreshed=refreshed,
                            )
                            if refreshed:
                                _edgekv_policy_refresh_block_score(self, block_id, profile)
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
            # h1_lru：原生 free-queue 顺序已经编码 LRU；只保留
            # admission/cached_blocks 诊断，跳过块级 rank 状态。
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
        # object_id -> 本次调用准入的 block；循环后用于对每个新对象运行一次对象级诊断
        # 准入决策。
        admission_object_blocks: dict[str, list[Any]] = {}
        cached_lpe_prefix_profile: dict[str, Any] | None = None
        cached_lpe_prefix_id = ''
        prefix_len = _edgekv_env_int('H1_PREFIX_REPETITION_PREFIX_LEN', 0) if _EDGEKV_GPU_POLICY_IS_LPE else 0
        for offset, block in enumerate(new_full_blocks):
            if getattr(block, 'is_null', False):
                continue
            block_id = _edgekv_block_id(block)
            score = float(meta.get('score', 0.0) or 0.0)
            p_reuse = float(meta.get('p_reuse', 0.5) or 0.5)
            profile: dict[str, Any] | None = None
            object_id = ''
            object_type = ''
            n_tokens = 0
            block_index = num_cached_blocks + offset
            block_start = block_index * block_size
            block_end = (block_index + 1) * block_size
            if _EDGEKV_GPU_POLICY_IS_LPE:
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
                    admission_object_blocks.setdefault(object_id, []).append(block)
                    score = float(profile.get('score', score) or score)
                    p_reuse = float(profile.get('p_reuse', p_reuse) or p_reuse)
            _edgekv_policy_on_admit(
                self,
                block_id,
                score=score,
                p_reuse=p_reuse,
                pinned=pinned,
                profile=profile,
            )
            _edgekv_heap_touch_block(self, block)
            _edgekv_note_gpu_stat('cached_blocks')
            _edgekv_note_gpu_stat('admissions')
            _edgekv_record_lpe_monitor(
                'admit',
                block_id=block_id,
                kv_cache_group_id=int(kv_cache_group_id),
                profile=profile,
                hit=False,
                block_index=block_index,
                block_start=block_start,
                block_end=block_end,
                pinned=pinned,
                score=score,
                p_reuse=p_reuse,
            )
        # 对象级诊断准入：每个新准入对象只决策一次。第一版永不阻止 vLLM 缓存；
        # reject 只标记 profile 并重新 touch 该对象的 block，使被拒绝对象在相同 score
        # 下排在已接受对象前面。
        if _EDGEKV_GPU_POLICY_IS_LPE and admission_object_blocks:
            for object_id, object_blocks in admission_object_blocks.items():
                profile = _EDGEKV_GPU_LPE_PROFILES.get(object_id)
                if profile is None:
                    continue
                decision = _edgekv_evaluate_object_admission(profile, pinned)
                if decision is None:
                    continue
                if decision == 'reject':
                    for block in object_blocks:
                        if not getattr(block, 'is_null', False):
                            _edgekv_heap_touch_block(self, block)
                _edgekv_record_lpe_monitor(
                    'admit_accept' if decision == 'accept' else 'admit_reject',
                    profile=profile,
                    hit=False,
                    admission_mode=_edgekv_admission_mode(),
                    admission_decision=decision,
                    admission_seq=int(profile.get('admission_seq', 0) or 0),
                    admission_min_resident_score=profile.get('admission_min_resident_score', ''),
                    admission_block_count=len(object_blocks),
                    pinned=pinned,
                )
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
            # h1_lru：诊断只需要 eviction 计数器。
            _edgekv_note_policy_time(start_ns)
            return evicted
        _edgekv_init_pool_state(self)
        block_id = _edgekv_block_id(block)
        score = float(self._edgekv_h1_scores.get(block_id, 0.0) or 0.0)
        p_reuse = float(self._edgekv_h1_p_reuse.get(block_id, 0.0) or 0.0)
        is_pinned = block_id in self._edgekv_h1_pinned
        profile = None
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
            _edgekv_record_lpe_monitor(
                'evict',
                block_id=block_id,
                kv_cache_group_id=kv_cache_group_id,
                profile=profile,
                hit=False,
                evicted=True,
                is_pinned=is_pinned,
                score=score,
                p_reuse=p_reuse,
            )
            _edgekv_drop_block_object(self, kv_cache_group_id, block_id)
        _edgekv_policy_on_evict(self, block_id, profile=profile, score=score, p_reuse=p_reuse)
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
                        profile = None
                        refreshed = False
                        if _EDGEKV_GPU_POLICY_IS_LPE:
                            profile = _edgekv_block_profile(self, block_id)
                            if profile is not None:
                                score_update_seq = int(profile.get('score_update_seq', 0) or 0)
                                refreshed = self._edgekv_h1_score_update_seq.get(block_id) != score_update_seq
                                _edgekv_record_lpe_monitor(
                                    'touch',
                                    block_id=block_id,
                                    profile=profile,
                                    hit=None,
                                )
                        _edgekv_policy_on_access(self, block_id, profile=profile, refreshed=refreshed)
                        _edgekv_heap_touch_block(self, block)
        _edgekv_note_policy_time(start_ns)
        return original_touch(self, blocks)

    def free_blocks(self: Any, ordered_blocks: Any) -> None:
        needs_state = _edgekv_gpu_policy_needs_rank_state()
        if needs_state:
            # ordered_blocks 可能是一次性迭代器（例如 reversed(...)）；先物化，确保原始调用
            # 和我们的 touch 循环都能看到完整 block 列表。
            ordered_blocks = list(ordered_blocks)
        original_free_blocks(self, ordered_blocks)
        if needs_state:
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
