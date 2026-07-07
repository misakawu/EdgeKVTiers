from __future__ import annotations

from typing import Any

from .state import RankStatePolicy


class LFUPolicy(RankStatePolicy):
    name = "h1_lfu"
    needs_rank_state = True

    def rank_tuple(self, pool: Any, block_id: int) -> tuple[float, ...]:
        block_id = int(block_id)
        recency = int(getattr(pool, "_edgekv_h1_recency", {}).get(block_id, 0))
        freq = int(getattr(pool, "_edgekv_h1_freq", {}).get(block_id, 0))
        return (float(freq), recency, block_id)
