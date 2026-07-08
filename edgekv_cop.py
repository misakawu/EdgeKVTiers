"""EdgeKVTiers 实验使用的对象级缓存对象画像器。"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque


DEFAULT_C_RE_MS_PER_TOKEN = 0.12
DEFAULT_BW_GBPS = 1.0
DEFAULT_D_DESER_MS = 3.0
DEFAULT_P_REUSE_PRIOR_WEIGHT = 0.70


def p_reuse_prior_from_item(item: dict[str, Any]) -> float | None:
    for key in ('p_reuse_prior', 'reuse_prior'):
        if key not in item:
            continue
        try:
            prior = float(item.get(key))
        except (TypeError, ValueError):
            continue
        return max(0.01, min(0.99, prior))
    return None


def object_id_from_item(item: dict[str, Any]) -> str:
    workload = str(item.get('workload', '')).lower()
    keys = ('rag_reuse_key', 'reuse_key', 'session_id', 'request_id') if item.get('rag_reuse_key') or 'rag' in workload else (
        'reuse_key',
        'rag_reuse_key',
        'session_id',
        'request_id',
    )
    for key in keys:
        value = str(item.get(key, '')).strip()
        if value:
            return value
    return 'unknown'


def object_type_from_item(item: dict[str, Any]) -> str:
    workload = str(item.get('workload') or '').lower()
    if item.get('rag_reuse_key') or 'rag' in workload:
        return 'rag_chunk_set'
    return str(item.get('object_type') or item.get('workload') or 'unknown')


def expiry_risk(object_type: str, item: dict[str, Any] | None = None) -> float:
    if item and item.get('session_end'):
        return 0.85
    name = object_type.lower()
    if 'rag' in name or 'chunk' in name:
        return 0.35
    if 'system' in name or 'shared' in name:
        return 0.05
    return 0.15


@dataclass
class ObjectProfile:
    object_id: str
    object_type: str
    n_tokens: int
    mu_kv_mb_per_token: float
    c_re_ms_per_token: float = DEFAULT_C_RE_MS_PER_TOKEN
    bw_gbps: float = DEFAULT_BW_GBPS
    d_deser_ms: float = DEFAULT_D_DESER_MS
    risk_exp: float = 0.15
    access_history: Deque[bool] = field(default_factory=lambda: deque(maxlen=16))
    access_count: int = 0
    hit_count: int = 0
    last_access_index: int = 0
    last_access_ts: float = field(default_factory=time.time)
    p_reuse: float = 0.5
    p_reuse_prior: float | None = None
    p_reuse_prior_weight: float = DEFAULT_P_REUSE_PRIOR_WEIGHT
    size_mb: float = 0.0
    c_recomp_ms: float = 0.0
    c_restore_ms: float = 0.0
    score: float = 0.0

    def recompute(self) -> None:
        n_tokens = max(int(self.n_tokens), 1)
        mu_kv = max(float(self.mu_kv_mb_per_token), 1e-9)
        c_re = max(float(self.c_re_ms_per_token), 1e-9)
        bw_gbps = max(float(self.bw_gbps), 1e-9)
        self.size_mb = mu_kv * n_tokens
        self.c_recomp_ms = c_re * n_tokens
        # 1 GB/s 约等于 1000 MB/s，也就是 1 MB/ms。
        self.c_restore_ms = self.size_mb / bw_gbps + float(self.d_deser_ms)
        self.score = (float(self.p_reuse) * self.c_recomp_ms) / max(self.size_mb, 1e-9)

    def update_access(
        self,
        hit: bool,
        access_index: int | None = None,
        n_tokens: int | None = None,
    ) -> None:
        if n_tokens is not None:
            self.n_tokens = max(int(n_tokens), 1)
        self.access_count += 1
        if hit:
            self.hit_count += 1
        self.access_history.append(bool(hit))
        self.last_access_index = int(access_index if access_index is not None else self.access_count)
        self.last_access_ts = time.time()
        self.p_reuse = self.estimate_reuse()
        self.recompute()

    def estimate_reuse(self) -> float:
        if not self.access_history:
            return self.p_reuse
        recent_hits = sum(1 for value in self.access_history if value)
        recent_rate = recent_hits / max(len(self.access_history), 1)
        freq_rate = self.hit_count / max(self.access_count, 1)
        recency_bonus = 1.0 / (1.0 + math.log1p(max(self.access_count - self.hit_count, 0)))
        estimate = 0.55 * recent_rate + 0.35 * freq_rate + 0.10 * recency_bonus
        if self.p_reuse_prior is not None:
            weight = max(0.0, min(1.0, float(self.p_reuse_prior_weight)))
            estimate = (weight * float(self.p_reuse_prior)) + ((1.0 - weight) * estimate)
        return max(0.01, min(0.99, estimate))

    def to_dict(self) -> dict[str, Any]:
        return {
            'object_id': self.object_id,
            'object_type': self.object_type,
            'n_tokens': int(self.n_tokens),
            'size_mb': round(self.size_mb, 6),
            'p_reuse': round(self.p_reuse, 6),
            'p_reuse_prior': (
                round(float(self.p_reuse_prior), 6)
                if self.p_reuse_prior is not None else ''
            ),
            'c_recomp_ms': round(self.c_recomp_ms, 6),
            'c_restore_ms': round(self.c_restore_ms, 6),
            'risk_exp': round(self.risk_exp, 6),
            'score': round(self.score, 9),
            'access_count': self.access_count,
            'hit_count': self.hit_count,
            'last_access_index': self.last_access_index,
        }


class COPProfiler:
    """按 prefix/session/chunk 复用身份索引的对象级 COP 注册表。"""

    def __init__(
        self,
        mu_kv_mb_per_token: float,
        c_re_ms_per_token: float = DEFAULT_C_RE_MS_PER_TOKEN,
        bw_gbps: float = DEFAULT_BW_GBPS,
        d_deser_ms: float = DEFAULT_D_DESER_MS,
    ) -> None:
        self.mu_kv_mb_per_token = float(mu_kv_mb_per_token)
        self.c_re_ms_per_token = float(c_re_ms_per_token)
        self.bw_gbps = float(bw_gbps)
        self.d_deser_ms = float(d_deser_ms)
        self.profiles: dict[str, ObjectProfile] = {}

    def update_from_item(
        self,
        item: dict[str, Any],
        hit: bool,
        access_index: int,
        object_id: str | None = None,
    ) -> ObjectProfile:
        oid = object_id or object_id_from_item(item)
        otype = object_type_from_item(item)
        n_tokens = max(int(item.get('n_tokens', 1) or 1), 1)
        p_reuse_prior = p_reuse_prior_from_item(item)
        profile = self.profiles.get(oid)
        if profile is None:
            profile = ObjectProfile(
                object_id=oid,
                object_type=otype,
                n_tokens=n_tokens,
                mu_kv_mb_per_token=self.mu_kv_mb_per_token,
                c_re_ms_per_token=self.c_re_ms_per_token,
                bw_gbps=self.bw_gbps,
                d_deser_ms=self.d_deser_ms,
                risk_exp=expiry_risk(otype, item),
                p_reuse_prior=p_reuse_prior,
            )
            profile.recompute()
            self.profiles[oid] = profile
        else:
            profile.object_type = otype
            profile.risk_exp = expiry_risk(otype, item)
            if p_reuse_prior is not None:
                profile.p_reuse_prior = p_reuse_prior
        profile.update_access(hit=hit, access_index=access_index, n_tokens=n_tokens)
        return profile

    def get(self, object_id: str) -> ObjectProfile | None:
        return self.profiles.get(str(object_id))

    def summary(self) -> dict[str, Any]:
        profiles = list(self.profiles.values())
        if not profiles:
            return {
                'cop_object_profile_count': 0,
                'cop_avg_p_reuse': 0.0,
                'cop_avg_score': 0.0,
            }
        return {
            'cop_object_profile_count': len(profiles),
            'cop_avg_p_reuse': round(sum(p.p_reuse for p in profiles) / len(profiles), 6),
            'cop_avg_score': round(sum(p.score for p in profiles) / len(profiles), 9),
        }
