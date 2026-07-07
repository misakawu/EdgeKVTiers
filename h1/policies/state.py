from __future__ import annotations

from typing import Any

from .base import AccessInfo, AdmitInfo, EvictInfo


class RankStatePolicy:
    name = ""
    needs_rank_state = True

    def _next_seq(self, pool: Any) -> int:
        pool._edgekv_h1_seq = int(getattr(pool, "_edgekv_h1_seq", 0) or 0) + 1
        return int(pool._edgekv_h1_seq)

    def on_admit(self, pool: Any, block_id: int, info: AdmitInfo) -> None:
        block_id = int(block_id)
        seq = self._next_seq(pool)
        pool._edgekv_h1_scores[block_id] = float(info.score)
        pool._edgekv_h1_freq[block_id] = max(int(pool._edgekv_h1_freq.get(block_id, 0)), 1)
        pool._edgekv_h1_recency[block_id] = seq
        if info.profile is not None:
            self.refresh_block_score(pool, block_id, info.profile)
        if info.pinned:
            pool._edgekv_h1_pinned.add(block_id)

    def on_access(self, pool: Any, block_id: int, info: AccessInfo | None = None) -> None:
        block_id = int(block_id)
        seq = self._next_seq(pool)
        pool._edgekv_h1_freq[block_id] = int(pool._edgekv_h1_freq.get(block_id, 0)) + 1
        pool._edgekv_h1_recency[block_id] = seq
        if info is not None and info.profile is not None and info.refreshed:
            self.refresh_block_score(pool, block_id, info.profile)

    def refresh_block_score(self, pool: Any, block_id: int, profile: dict[str, Any]) -> None:
        block_id = int(block_id)
        pool._edgekv_h1_p_reuse[block_id] = float(profile.get("p_reuse", 0.5))
        pool._edgekv_h1_scores[block_id] = float(profile.get("score", 0.0))
        pool._edgekv_h1_score_update_seq[block_id] = int(profile.get("score_update_seq", 0) or 0)

    def on_evict(self, pool: Any, block_id: int, info: EvictInfo | None = None) -> None:
        block_id = int(block_id)
        pool._edgekv_h1_scores.pop(block_id, None)
        pool._edgekv_h1_p_reuse.pop(block_id, None)
        pool._edgekv_h1_score_update_seq.pop(block_id, None)
        pool._edgekv_h1_freq.pop(block_id, None)
        pool._edgekv_h1_recency.pop(block_id, None)
        pool._edgekv_h1_access_history.pop(block_id, None)
        pool._edgekv_h1_pinned.discard(block_id)
