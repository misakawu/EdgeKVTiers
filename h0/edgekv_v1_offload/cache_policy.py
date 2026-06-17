#!/usr/bin/env python3
"""CachePolicy implementations for H1 lifecycle eviction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Iterable, Optional


@dataclass(frozen=True)
class EvictionPlan:
    object_id: str
    action: str
    reason: str


class CachePolicy(ABC):
    """Base class for H1 cache eviction policies.

    Subclasses only choose victims/actions. H1Policy owns object accounting,
    budget enforcement, and logging so the same policies can be reused by the
    shadow replay path and the vLLM V1 connector wrapper.
    """

    name = "base"

    def __init__(self, *, reserve_mb: float = 0.0) -> None:
        self.reserve_mb = max(float(reserve_mb), 0.0)

    def target_budget_mb(self, gpu_budget_mb: float) -> float:
        return max(float(gpu_budget_mb) - self.reserve_mb, 0.0)

    def should_evict(self, *, resident_mb: float, gpu_budget_mb: float) -> bool:
        return resident_mb > self.target_budget_mb(gpu_budget_mb)

    @abstractmethod
    def choose_victim(
        self,
        resident: Collection[object],
        *,
        exclude_object_id: Optional[str] = None,
    ) -> Optional[EvictionPlan]:
        """Return the next victim/action under memory pressure."""

    def _candidates(self, resident: Collection[object], *, exclude_object_id: Optional[str]) -> list[object]:
        candidates = [obj for obj in resident if getattr(obj, "object_id", None) != exclude_object_id]
        return candidates or list(resident)


class VllmDefaultCachePolicy(CachePolicy):
    name = "vllm-default"

    def choose_victim(
        self,
        resident: Collection[object],
        *,
        exclude_object_id: Optional[str] = None,
    ) -> Optional[EvictionPlan]:
        return None


class LRUCachePolicy(CachePolicy):
    name = "lru"

    def choose_victim(
        self,
        resident: Collection[object],
        *,
        exclude_object_id: Optional[str] = None,
    ) -> Optional[EvictionPlan]:
        candidates = self._candidates(resident, exclude_object_id=exclude_object_id)
        if not candidates:
            return None
        victim = min(candidates, key=lambda obj: (obj.last_access_step, obj.first_seen_step, obj.object_id))
        return EvictionPlan(victim.object_id, "offload", "lru-budget-pressure")


class LFUCachePolicy(CachePolicy):
    name = "lfu"

    def choose_victim(
        self,
        resident: Collection[object],
        *,
        exclude_object_id: Optional[str] = None,
    ) -> Optional[EvictionPlan]:
        candidates = self._candidates(resident, exclude_object_id=exclude_object_id)
        if not candidates:
            return None
        victim = min(candidates, key=lambda obj: (obj.access_count, obj.last_access_step, obj.object_id))
        return EvictionPlan(victim.object_id, "offload", "lfu-budget-pressure")


class LPECachePolicy(CachePolicy):
    """Lifecycle Policy Engine from module 6.4.

    LPE is the lifecycle fallback after TMS cannot free enough memory by tier
    downgrades. H1 is lifecycle-only, so all resident objects are treated as
    already eligible for LPE eviction. The score is unit-memory saved recompute
    value: p_reuse * c_recomp / size.
    """

    name = "lpe-score"

    def __init__(
        self,
        *,
        theta_keep: float = 0.5,
        reserve_mb: float = 0.0,
        pinned_object_ids: Optional[Iterable[str]] = None,
        feasible_offload: bool = True,
    ) -> None:
        super().__init__(reserve_mb=reserve_mb)
        self.theta_keep = float(theta_keep)
        self.pinned_object_ids = set(pinned_object_ids or [])
        self.feasible_offload = bool(feasible_offload)

    def choose_victim(
        self,
        resident: Collection[object],
        *,
        exclude_object_id: Optional[str] = None,
    ) -> Optional[EvictionPlan]:
        candidates = [
            obj
            for obj in self._candidates(resident, exclude_object_id=exclude_object_id)
            if obj.object_id not in self.pinned_object_ids and not getattr(obj, "pinned", False)
        ]
        if not candidates:
            return None
        victim = min(candidates, key=lambda obj: (obj.score, obj.last_access_step, obj.object_id))
        if victim.p_reuse >= self.theta_keep and self.feasible_offload:
            action = "offload"
            reason = "lpe-score-high-reuse-offload"
        else:
            action = "drop"
            reason = "lpe-score-low-reuse-drop"
        return EvictionPlan(victim.object_id, action, reason)


POLICY_CLASSES = {
    VllmDefaultCachePolicy.name: VllmDefaultCachePolicy,
    LRUCachePolicy.name: LRUCachePolicy,
    LFUCachePolicy.name: LFUCachePolicy,
    LPECachePolicy.name: LPECachePolicy,
}


def build_cache_policy(
    name: str,
    *,
    theta_keep: float = 0.5,
    reserve_mb: float = 0.0,
    pinned_object_ids: Optional[Iterable[str]] = None,
    feasible_offload: bool = True,
) -> CachePolicy:
    if name == LPECachePolicy.name:
        return LPECachePolicy(
            theta_keep=theta_keep,
            reserve_mb=reserve_mb,
            pinned_object_ids=pinned_object_ids,
            feasible_offload=feasible_offload,
        )
    try:
        cls = POLICY_CLASSES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported H1 cache policy {name!r}; expected one of {sorted(POLICY_CLASSES)}") from exc
    return cls(reserve_mb=reserve_mb)
