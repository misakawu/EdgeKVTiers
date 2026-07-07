from __future__ import annotations

import hashlib
import os
from typing import Any, Callable, Protocol

from .state import RankStatePolicy


def object_sort_key(object_id: str) -> int:
    digest = hashlib.sha1(str(object_id).encode("utf-8")).hexdigest()[:12]
    return int(digest, 16)


def _env_float(name: str, default: float) -> float:
    """与 sitecustomize._edgekv_env_float 等价的浮点解析（不含其缓存，不影响结果）。"""
    raw = os.environ.get(name)
    try:
        return float(raw if raw is not None else default)
    except (TypeError, ValueError):
        return default


class LPEScorer(Protocol):
    """LPE 打分策略接口 —— 后续研究的主要替换点。

    `recompute_score` 原地改写 profile 的 score 及其派生尺寸字段。
    """

    def recompute_score(self, profile: dict[str, Any]) -> None: ...


class DefaultLPEScorer:
    """默认 LPE 打分：score = p_reuse × c_recomp_ms / 逻辑KV大小(MB)。

    分母用对象的*逻辑*大小（由 n_tokens 派生的稳定量），不用当前驻留字节，
    避免部分驱逐后分母缩小导致 score 被数量级放大。逐字复刻自原
    sitecustomize._edgekv_recompute_profile_score，保证行为不变。
    """

    def __init__(self, mu_kv_mb_per_token_default: float = 0.12) -> None:
        self._mu_kv_default = mu_kv_mb_per_token_default

    def recompute_score(self, profile: dict[str, Any]) -> None:
        n_tokens = max(float(profile.get("n_tokens", 1) or 1), 1.0)
        bytes_per_token = float(profile.get("bytes_per_token", 0.0) or 0.0)
        if bytes_per_token > 0.0:
            logical_size_mb = (n_tokens * bytes_per_token) / 1024.0 / 1024.0
            score_size_source = "vllm_kv_cache_spec_bytes_per_token"
        else:
            logical_size_mb = _env_float("EDGEKV_MU_KV_MB_PER_TOKEN", self._mu_kv_default) * n_tokens
            score_size_source = "fallback_theoretical_mu_kv"
        profile["logical_size_mb"] = logical_size_mb
        profile["score_size_source"] = score_size_source

        resident_bytes = int(profile.get("size_bytes", 0) or 0)
        profile["resident_size_mb"] = float(resident_bytes) / 1024.0 / 1024.0

        # size_mb / size_source 镜像逻辑（score）大小，兼容旧版读取方。
        profile["size_mb"] = logical_size_mb
        profile["size_source"] = score_size_source

        if logical_size_mb <= 0.0:
            profile["score"] = 0.0
            return
        profile["score"] = (
            float(profile.get("p_reuse", 0.5))
            * float(profile.get("c_recomp_ms", 0.0))
            / max(logical_size_mb, 1e-9)
        )


class LPEPolicy(RankStatePolicy):
    name = "h1_lpe"
    needs_rank_state = True

    def __init__(
        self,
        block_profile: Callable[[Any, int], dict[str, Any] | None] | None = None,
        scorer: LPEScorer | None = None,
    ) -> None:
        self._block_profile = block_profile
        self.scorer: LPEScorer = scorer if scorer is not None else DefaultLPEScorer()

    def recompute_score(self, profile: dict[str, Any]) -> None:
        """委托打分器原地重算 profile 的 score（供 vLLM 钩子调用）。"""
        self.scorer.recompute_score(profile)

    def rank_tuple(self, pool: Any, block_id: int) -> tuple[float, ...]:
        block_id = int(block_id)
        if block_id in getattr(pool, "_edgekv_h1_pinned", set()):
            return (float("inf"), 1, 0)
        profile = self._block_profile(pool, block_id) if self._block_profile is not None else None
        if profile is None:
            score = float(getattr(pool, "_edgekv_h1_scores", {}).get(block_id, 0.0) or 0.0)
            return (score, 1, block_id)
        score = float(profile.get("score", 0.0) or 0.0)
        rejected_order = 0 if profile.get("admission_rejected") else 1
        obj_key = int(profile.get("object_sort_key") or object_sort_key(str(profile.get("object_id", ""))))
        return (score, rejected_order, obj_key)
