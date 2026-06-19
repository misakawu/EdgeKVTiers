#!/usr/bin/env python3
"""H1 cache policies for vLLM 0.11.0 KV OffloadingConnector.

vLLM 0.11.0 exposes CPU KV offload through ``OffloadingSpecFactory`` and a
monolithic ``LRUOffloadingManager``.  It does not expose the later
``vllm.v1.kv_offload.cpu.manager`` policy registry, so this module provides a
repo-local policy-aware manager that can be selected with
``kv_connector_extra_config.spec_module_path``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable
from typing import Optional

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (LoadStoreSpec, OffloadingEvent,
                                         OffloadingManager, PrepareStoreOutput)
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.cpu import CPUOffloadingSpec


class CachePolicy(ABC):
    """Eviction policy interface used by ``PolicyOffloadingManager``."""

    def __init__(self, cache_capacity: int) -> None:
        self.cache_capacity = cache_capacity

    @abstractmethod
    def get(self, key: BlockHash) -> Optional[BlockStatus]:
        pass

    @abstractmethod
    def insert(self, key: BlockHash, block: BlockStatus) -> None:
        pass

    @abstractmethod
    def touch(self, keys: Iterable[BlockHash]) -> None:
        pass

    @abstractmethod
    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> Optional[list[tuple[BlockHash, BlockStatus]]]:
        pass

    @abstractmethod
    def remove(self, key: BlockHash) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class H1LRUCachePolicy(CachePolicy):
    """LRU policy backed by insertion/access order."""

    def __init__(self, cache_capacity: int) -> None:
        super().__init__(cache_capacity)
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()

    def get(self, key: BlockHash) -> Optional[BlockStatus]:
        return self.blocks.get(key)

    def insert(self, key: BlockHash, block: BlockStatus) -> None:
        self.blocks[key] = block

    def touch(self, keys: Iterable[BlockHash]) -> None:
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.blocks.move_to_end(key)

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> Optional[list[tuple[BlockHash, BlockStatus]]]:
        if n == 0:
            return []

        candidates: list[tuple[BlockHash, BlockStatus]] = []
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

    def remove(self, key: BlockHash) -> None:
        del self.blocks[key]

    def clear(self) -> None:
        self.blocks.clear()


class H1LFUCachePolicy(CachePolicy):
    """LFU policy with LRU tie breaking."""

    def __init__(self, cache_capacity: int) -> None:
        super().__init__(cache_capacity)
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.freq: dict[BlockHash, int] = {}

    def get(self, key: BlockHash) -> Optional[BlockStatus]:
        return self.blocks.get(key)

    def insert(self, key: BlockHash, block: BlockStatus) -> None:
        self.blocks[key] = block
        self.freq.setdefault(key, 1)

    def touch(self, keys: Iterable[BlockHash]) -> None:
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.freq[key] = self.freq.get(key, 0) + 1
                self.blocks.move_to_end(key)

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> Optional[list[tuple[BlockHash, BlockStatus]]]:
        if n == 0:
            return []

        eligible: list[tuple[BlockHash, BlockStatus, int]] = [
            (key, block, idx)
            for idx, (key, block) in enumerate(self.blocks.items())
            if block.ref_cnt == 0 and key not in protected
        ]
        if len(eligible) < n:
            return None

        selected = sorted(
            eligible, key=lambda item: (self.freq.get(item[0], 0), item[2])
        )[:n]
        result = [(key, block) for key, block, _ in selected]
        for key, _ in result:
            del self.blocks[key]
            self.freq.pop(key, None)
        return result

    def remove(self, key: BlockHash) -> None:
        del self.blocks[key]
        self.freq.pop(key, None)

    def clear(self) -> None:
        self.blocks.clear()
        self.freq.clear()


class H1LPECachePolicy(CachePolicy):
    """LPE policy: evict the lowest expected value per memory unit first."""

    def __init__(self, cache_capacity: int) -> None:
        super().__init__(cache_capacity)
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.scores: dict[BlockHash, float] = {}

    def set_score(
        self,
        key: BlockHash,
        *,
        p_reuse: float,
        c_recomp_ms: float,
        size_mb: float,
    ) -> None:
        self.scores[key] = (
            float(p_reuse) * float(c_recomp_ms)
        ) / max(float(size_mb), 1e-9)

    def get(self, key: BlockHash) -> Optional[BlockStatus]:
        return self.blocks.get(key)

    def insert(self, key: BlockHash, block: BlockStatus) -> None:
        self.blocks[key] = block
        self.scores.setdefault(key, 0.0)

    def touch(self, keys: Iterable[BlockHash]) -> None:
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.blocks.move_to_end(key)

    def evict(
        self, n: int, protected: set[BlockHash]
    ) -> Optional[list[tuple[BlockHash, BlockStatus]]]:
        if n == 0:
            return []

        eligible: list[tuple[BlockHash, BlockStatus, int]] = [
            (key, block, idx)
            for idx, (key, block) in enumerate(self.blocks.items())
            if block.ref_cnt == 0 and key not in protected
        ]
        if len(eligible) < n:
            return None

        selected = sorted(
            eligible, key=lambda item: (self.scores.get(item[0], 0.0), item[2])
        )[:n]
        result = [(key, block) for key, block, _ in selected]
        for key, _ in result:
            del self.blocks[key]
            self.scores.pop(key, None)
        return result

    def remove(self, key: BlockHash) -> None:
        del self.blocks[key]
        self.scores.pop(key, None)

    def clear(self) -> None:
        self.blocks.clear()
        self.scores.clear()


_CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "h1_lru": H1LRUCachePolicy,
    "h1_lfu": H1LFUCachePolicy,
    "h1_lpe": H1LPECachePolicy,
}


class PolicyOffloadingManager(OffloadingManager):
    """vLLM 0.11.0 offloading manager with pluggable eviction policy."""

    def __init__(
        self,
        backend: Backend,
        cache_policy: str = "h1_lru",
        enable_events: bool = False,
    ) -> None:
        self.backend = backend
        policy_cls = _CACHE_POLICIES[cache_policy]
        self.policy = policy_cls(cache_capacity=backend.get_num_free_blocks())
        self.events: Optional[list[OffloadingEvent]] = (
            [] if enable_events else None
        )

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        hit_count = 0
        for block_hash in block_hashes:
            block = self.policy.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        block_hashes = list(block_hashes)
        blocks: list[BlockStatus] = []
        for block_hash in block_hashes:
            block = self.policy.get(block_hash)
            assert block is not None
            assert block.is_ready
            block.ref_cnt += 1
            blocks.append(block)
        return self.backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self.policy.touch(block_hashes)

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        for block_hash in block_hashes:
            block = self.policy.get(block_hash)
            assert block is not None
            assert block.ref_cnt > 0
            block.ref_cnt -= 1

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> Optional[PrepareStoreOutput]:
        block_hashes = list(block_hashes)
        block_hashes_to_store = [
            block_hash
            for block_hash in block_hashes
            if self.policy.get(block_hash) is None
        ]

        num_blocks_to_evict = (
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()
        )

        evicted: list[tuple[BlockHash, BlockStatus]] = []
        if num_blocks_to_evict > 0:
            evicted_or_none = self.policy.evict(
                num_blocks_to_evict, protected=set(block_hashes)
            )
            if evicted_or_none is None:
                return None
            evicted = evicted_or_none

        for _, block in evicted:
            self.backend.free(block)

        block_hashes_evicted = [block_hash for block_hash, _ in evicted]
        if block_hashes_evicted and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=block_hashes_evicted,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=True,
                )
            )

        blocks = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store)

        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.policy.insert(block_hash, block)

        store_spec = self.backend.get_load_store_spec(
            block_hashes_to_store, blocks
        )
        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=block_hashes_evicted,
        )

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        stored_block_hashes: list[BlockHash] = []
        if success:
            for block_hash in block_hashes:
                block = self.policy.get(block_hash)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self.policy.get(block_hash)
                if block is not None and not block.is_ready:
                    self.backend.free(block)
                    self.policy.remove(block_hash)

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored_block_hashes,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()


class H1CPUOffloadingSpec(CPUOffloadingSpec):
    """CPU offloading spec that swaps vLLM 0.11.0 LRU for H1 policies."""

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.cache_policy = self.extra_config.get("cache_policy", "h1_lru")
        if self.cache_policy not in _CACHE_POLICIES:
            supported = ", ".join(sorted(_CACHE_POLICIES))
            raise ValueError(
                f"Unsupported H1 cache_policy: {self.cache_policy}. "
                f"Supported policies: {supported}"
            )

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )
            self._manager = PolicyOffloadingManager(
                CPUBackend(
                    block_size=self.offloaded_block_size,
                    num_blocks=self.num_cpu_blocks,
                ),
                cache_policy=self.cache_policy,
                enable_events=enable_events,
            )
        return self._manager


__all__ = [
    "CachePolicy",
    "H1CPUOffloadingSpec",
    "H1LFUCachePolicy",
    "H1LPECachePolicy",
    "H1LRUCachePolicy",
    "PolicyOffloadingManager",
    "_CACHE_POLICIES",
]
