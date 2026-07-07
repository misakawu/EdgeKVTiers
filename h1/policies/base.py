from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AdmitInfo:
    score: float = 0.0
    p_reuse: float = 0.5
    pinned: bool = False
    profile: dict[str, Any] | None = None


@dataclass(frozen=True)
class AccessInfo:
    profile: dict[str, Any] | None = None
    refreshed: bool = False


@dataclass(frozen=True)
class EvictInfo:
    profile: dict[str, Any] | None = None
    score: float = 0.0
    p_reuse: float = 0.0


class CachePolicy(Protocol):
    name: str
    needs_rank_state: bool

    def rank_tuple(self, pool: Any, block_id: int) -> tuple[float, ...]:
        ...

    def on_admit(self, pool: Any, block_id: int, info: AdmitInfo) -> None:
        ...

    def on_access(self, pool: Any, block_id: int, info: AccessInfo | None = None) -> None:
        ...

    def refresh_block_score(self, pool: Any, block_id: int, profile: dict[str, Any]) -> None:
        ...

    def on_evict(self, pool: Any, block_id: int, info: EvictInfo | None = None) -> None:
        ...
