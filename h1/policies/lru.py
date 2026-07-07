from __future__ import annotations

from typing import Any

from .base import AccessInfo, AdmitInfo, EvictInfo


class LRUPolicy:
    name = "h1_lru"
    needs_rank_state = False

    def rank_tuple(self, pool: Any, block_id: int) -> tuple[float, ...]:
        recency = int(getattr(pool, "_edgekv_h1_recency", {}).get(block_id, 0))
        return (float(recency), 0, int(block_id))

    def on_admit(self, pool: Any, block_id: int, info: AdmitInfo) -> None:
        return None

    def on_access(self, pool: Any, block_id: int, info: AccessInfo | None = None) -> None:
        return None

    def refresh_block_score(self, pool: Any, block_id: int, profile: dict[str, Any]) -> None:
        return None

    def on_evict(self, pool: Any, block_id: int, info: EvictInfo | None = None) -> None:
        return None
