#!/usr/bin/env python3
"""vLLM V1 CPU KV offload CachePolicy plugins for H1.

These classes target the real vLLM interface introduced under
``vllm.v1.kv_offload.cpu``.  The local fallback definitions keep the module
importable in older vLLM builds so unit tests can validate policy semantics
without requiring the offload-capable wheel.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any, Optional

try:  # vLLM >= 0.23.0
    from vllm.v1.kv_offload.base import OffloadKey
    from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
    from vllm.v1.kv_offload.cpu.manager import _CACHE_POLICIES
    HAS_VLLM_CPU_OFFLOAD = True
except Exception:  # import-safe fallback for the current vLLM 0.8.5 env
    OffloadKey = Any
    HAS_VLLM_CPU_OFFLOAD = False

    class BlockStatus:  # type: ignore[no-redef]
        def __init__(self, block_id: int):
            self.ref_cnt = -1
            self.block_id = block_id

        @property
        def is_ready(self) -> bool:
            return self.ref_cnt >= 0

    class CachePolicy:  # type: ignore[no-redef]
        def __init__(self, cache_capacity: int) -> None:
            raise NotImplementedError

        def get(self, key: OffloadKey) -> Optional[BlockStatus]:
            raise NotImplementedError

        def insert(self, key: OffloadKey, block: BlockStatus) -> None:
            raise NotImplementedError

        def remove(self, key: OffloadKey) -> None:
            raise NotImplementedError

        def touch(self, keys: Iterable[OffloadKey]) -> None:
            raise NotImplementedError

        def evict(
            self, n: int, protected: set[OffloadKey]
        ) -> Optional[list[tuple[OffloadKey, BlockStatus]]]:
            raise NotImplementedError

        def clear(self) -> None:
            raise NotImplementedError

    _CACHE_POLICIES: dict[str, type[CachePolicy]] = {}


class H1LRUCachePolicy(CachePolicy):
    """Atomic LRU policy for vLLM CPU offload blocks."""

    def __init__(self, cache_capacity: int) -> None:
        self.blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()

    def get(self, key: OffloadKey) -> Optional[BlockStatus]:
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block

    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.blocks.move_to_end(key)

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> Optional[list[tuple[OffloadKey, BlockStatus]]]:
        if n == 0:
            return []
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for key, block in self.blocks.items():
            if block.ref_cnt == 0 and key not in protected:
                candidates.append((key, block))
                if len(candidates) == n:
                    break
        if len(candidates) < n:
            return None
        for key, _ in candidates:
            del self.blocks[key]
        return candidates

    def clear(self) -> None:
        self.blocks.clear()


class H1LFUCachePolicy(CachePolicy):
    """LFU policy with LRU tie-break and atomic eviction."""

    def __init__(self, cache_capacity: int) -> None:
        self.blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()
        self.freq: dict[OffloadKey, int] = {}

    def get(self, key: OffloadKey) -> Optional[BlockStatus]:
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        self.freq.setdefault(key, 1)

    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self.freq.pop(key, None)

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.freq[key] = self.freq.get(key, 0) + 1
                self.blocks.move_to_end(key)

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> Optional[list[tuple[OffloadKey, BlockStatus]]]:
        if n == 0:
            return []
        eligible = [
            (key, block)
            for key, block in self.blocks.items()
            if block.ref_cnt == 0 and key not in protected
        ]
        if len(eligible) < n:
            return None
        selected = sorted(eligible, key=lambda item: (self.freq.get(item[0], 0), list(self.blocks).index(item[0])))[:n]
        for key, _ in selected:
            del self.blocks[key]
            self.freq.pop(key, None)
        return selected

    def clear(self) -> None:
        self.blocks.clear()
        self.freq.clear()


class H1LPECachePolicy(CachePolicy):
    """LPE policy from module 6.4: evict lowest unit-memory value first.

    vLLM's CPU offload CachePolicy only returns evicted blocks; it does not
    expose a separate drop/offload action at this layer. The LPE decision here
    maps to eviction from CPU offload storage. Scores can be supplied by a
    controller through ``set_score``; unseen keys fall back to 0 and are evicted
    before known high-value blocks.
    """

    def __init__(self, cache_capacity: int) -> None:
        self.blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()
        self.scores: dict[OffloadKey, float] = {}

    def set_score(self, key: OffloadKey, *, p_reuse: float, c_recomp_ms: float, size_mb: float) -> None:
        self.scores[key] = (float(p_reuse) * float(c_recomp_ms)) / max(float(size_mb), 1e-9)

    def get(self, key: OffloadKey) -> Optional[BlockStatus]:
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        self.scores.setdefault(key, 0.0)

    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self.scores.pop(key, None)

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.blocks.move_to_end(key)

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> Optional[list[tuple[OffloadKey, BlockStatus]]]:
        if n == 0:
            return []
        eligible = [
            (key, block, idx)
            for idx, (key, block) in enumerate(self.blocks.items())
            if block.ref_cnt == 0 and key not in protected
        ]
        if len(eligible) < n:
            return None
        selected = sorted(eligible, key=lambda item: (self.scores.get(item[0], 0.0), item[2]))[:n]
        result = [(key, block) for key, block, _ in selected]
        for key, _ in result:
            del self.blocks[key]
            self.scores.pop(key, None)
        return result

    def clear(self) -> None:
        self.blocks.clear()
        self.scores.clear()


def register_h1_cache_policies() -> dict[str, type[CachePolicy]]:
    """Register H1 policies into vLLM's CPU offload policy registry."""

    _CACHE_POLICIES["h1_lru"] = H1LRUCachePolicy
    _CACHE_POLICIES["h1_lfu"] = H1LFUCachePolicy
    _CACHE_POLICIES["h1_lpe"] = H1LPECachePolicy
    return _CACHE_POLICIES


register_h1_cache_policies()
