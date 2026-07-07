from __future__ import annotations

from typing import Any, Callable

from .base import AccessInfo, AdmitInfo, CachePolicy, EvictInfo
from .lfu import LFUPolicy
from .lpe import LPEPolicy
from .lru import LRUPolicy


class VllmDefaultPolicy:
    name = "vllm_default"
    needs_rank_state = False

    def rank_tuple(self, pool: Any, block_id: int) -> tuple[float, ...]:
        recency = int(getattr(pool, "_edgekv_h1_recency", {}).get(int(block_id), 0))
        return (float(recency), 0, int(block_id))

    def on_admit(self, pool: Any, block_id: int, info: AdmitInfo) -> None:
        return None

    def on_access(self, pool: Any, block_id: int, info: AccessInfo | None = None) -> None:
        return None

    def refresh_block_score(self, pool: Any, block_id: int, profile: dict[str, Any]) -> None:
        return None

    def on_evict(self, pool: Any, block_id: int, info: EvictInfo | None = None) -> None:
        return None


def create_policy(
    name: str | None,
    *,
    block_profile: Callable[[Any, int], dict[str, Any] | None] | None = None,
) -> CachePolicy:
    normalized = (name or "vllm_default").strip() or "vllm_default"
    if normalized == "vllm_default":
        return VllmDefaultPolicy()
    if normalized == "h1_lru":
        return LRUPolicy()
    if normalized == "h1_lfu":
        return LFUPolicy()
    if normalized == "h1_lpe":
        return LPEPolicy(block_profile=block_profile)
    raise ValueError(f"unknown EDGEKV_H1_GPU_POLICY: {normalized}")
