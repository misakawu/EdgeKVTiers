#!/usr/bin/env python3
"""Custom eviction policies for H1 V1 KV offload experiments."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass
class KVObjectState:
    object_id: str
    request_id: str
    session_id: str
    object_type: str
    n_tokens: int
    size_mb: float
    p_reuse: float = 0.0
    access_count: int = 0
    first_seen_step: int = 0
    last_access_step: int = 0
    c_recomp_ms: float = 0.0
    score: float = 0.0
    offloaded: bool = False


@dataclass
class PolicyDecision:
    policy: str
    request_id: str
    object_id: str
    object_type: str
    n_tokens: int
    size_mb: float
    p_reuse: float
    c_recomp_ms: float
    score: float
    action: str
    reason: str
    hit: bool
    resident_mb_before: float
    resident_mb_after: float
    gpu_budget_mb: float
    t_policy_ms: float
    ttft_ms: float = 0.0
    gpu_memory_mb: float = 0.0


class JsonlDecisionLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, decision: PolicyDecision) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(decision), ensure_ascii=False, sort_keys=True))
            f.write("\n")


class H1Policy:
    """Budgeted resident-set policy used by the H1 offload connector.

    The policy works at trace object granularity. The V1 connector can call the
    same admission/reuse methods when real request/block metadata is available;
    the runner also uses them for shadow decisions beside OpenAI replay logs.
    """

    SUPPORTED = {"vllm-default", "lru", "lfu", "lpe-score"}

    def __init__(
        self,
        *,
        policy: str,
        gpu_budget_mb: float,
        c_re_ms_per_token: float,
        freq_decay: float = 0.85,
        logger: Optional[JsonlDecisionLogger] = None,
    ) -> None:
        if policy not in self.SUPPORTED:
            raise ValueError(f"unsupported H1 policy {policy!r}; expected one of {sorted(self.SUPPORTED)}")
        self.policy = policy
        self.gpu_budget_mb = float(gpu_budget_mb)
        self.c_re_ms_per_token = float(c_re_ms_per_token)
        self.freq_decay = float(freq_decay)
        self.logger = logger
        self.step = 0
        self.resident: Dict[str, KVObjectState] = {}
        self.history: Dict[str, KVObjectState] = {}

    @property
    def resident_mb(self) -> float:
        return sum(obj.size_mb for obj in self.resident.values())

    def observe_request(
        self,
        *,
        request_id: str,
        session_id: str,
        object_id: str,
        object_type: str,
        n_tokens: int,
        size_mb: float,
        hit: bool,
        ttft_ms: float = 0.0,
        gpu_memory_mb: float = 0.0,
    ) -> list[PolicyDecision]:
        started = time.perf_counter()
        self.step += 1
        prior = self.history.get(object_id)
        access_count = (prior.access_count if prior else 0) + 1
        p_reuse = self._estimate_p_reuse(prior, access_count)
        c_recomp_ms = max(float(n_tokens), 0.0) * self.c_re_ms_per_token
        score = self._score(p_reuse, c_recomp_ms, size_mb)
        obj = KVObjectState(
            object_id=object_id,
            request_id=request_id,
            session_id=session_id,
            object_type=object_type,
            n_tokens=int(n_tokens),
            size_mb=float(size_mb),
            p_reuse=p_reuse,
            access_count=access_count,
            first_seen_step=prior.first_seen_step if prior else self.step,
            last_access_step=self.step,
            c_recomp_ms=c_recomp_ms,
            score=score,
            offloaded=False,
        )
        before = self.resident_mb
        self.history[object_id] = obj

        decisions: list[PolicyDecision] = []
        if self.policy == "vllm-default":
            decision = self._decision(
                obj,
                action="observe",
                reason="vllm-default-no-external-eviction",
                hit=hit,
                resident_mb_before=before,
                resident_mb_after=before,
                started=started,
                ttft_ms=ttft_ms,
                gpu_memory_mb=gpu_memory_mb,
            )
            self._log(decision)
            return [decision]

        self.resident[object_id] = obj
        after_admit = self.resident_mb
        admit = self._decision(
            obj,
            action="reuse" if hit else "admit",
            reason="object-hit" if hit else "new-object",
            hit=hit,
            resident_mb_before=before,
            resident_mb_after=after_admit,
            started=started,
            ttft_ms=ttft_ms,
            gpu_memory_mb=gpu_memory_mb,
        )
        decisions.append(admit)

        while self.resident_mb > self.gpu_budget_mb and self.resident:
            victim = self._choose_victim(exclude_object_id=object_id if len(self.resident) > 1 else None)
            if victim is None:
                break
            victim_before = self.resident_mb
            removed = self.resident.pop(victim.object_id)
            removed.offloaded = True
            self.history[removed.object_id] = removed
            evict = self._decision(
                removed,
                action="offload",
                reason=f"{self.policy}-budget-pressure",
                hit=False,
                resident_mb_before=victim_before,
                resident_mb_after=self.resident_mb,
                started=started,
                ttft_ms=ttft_ms,
                gpu_memory_mb=gpu_memory_mb,
            )
            decisions.append(evict)

        for decision in decisions:
            self._log(decision)
        return decisions

    def _estimate_p_reuse(self, prior: Optional[KVObjectState], access_count: int) -> float:
        if prior is None:
            return 0.25
        gap = max(self.step - prior.last_access_step, 1)
        recency = math.exp(-gap / 32.0)
        frequency = 1.0 - (self.freq_decay ** access_count)
        return min(0.99, max(0.01, 0.55 * recency + 0.45 * frequency))

    def _score(self, p_reuse: float, c_recomp_ms: float, size_mb: float) -> float:
        return (p_reuse * c_recomp_ms) / max(size_mb, 1e-9)

    def _choose_victim(self, *, exclude_object_id: Optional[str]) -> Optional[KVObjectState]:
        candidates = [obj for obj in self.resident.values() if obj.object_id != exclude_object_id]
        if not candidates:
            candidates = list(self.resident.values())
        if not candidates:
            return None
        if self.policy == "lru":
            return min(candidates, key=lambda obj: (obj.last_access_step, obj.first_seen_step, obj.object_id))
        if self.policy == "lfu":
            return min(candidates, key=lambda obj: (obj.access_count, obj.last_access_step, obj.object_id))
        if self.policy == "lpe-score":
            return min(candidates, key=lambda obj: (obj.score, obj.last_access_step, obj.object_id))
        return None

    def _decision(
        self,
        obj: KVObjectState,
        *,
        action: str,
        reason: str,
        hit: bool,
        resident_mb_before: float,
        resident_mb_after: float,
        started: float,
        ttft_ms: float,
        gpu_memory_mb: float,
    ) -> PolicyDecision:
        return PolicyDecision(
            policy=self.policy,
            request_id=obj.request_id,
            object_id=obj.object_id,
            object_type=obj.object_type,
            n_tokens=obj.n_tokens,
            size_mb=round(obj.size_mb, 6),
            p_reuse=round(obj.p_reuse, 6),
            c_recomp_ms=round(obj.c_recomp_ms, 6),
            score=round(obj.score, 6),
            action=action,
            reason=reason,
            hit=hit,
            resident_mb_before=round(resident_mb_before, 6),
            resident_mb_after=round(resident_mb_after, 6),
            gpu_budget_mb=round(self.gpu_budget_mb, 6),
            t_policy_ms=round((time.perf_counter() - started) * 1000.0, 6),
            ttft_ms=round(ttft_ms, 6),
            gpu_memory_mb=round(gpu_memory_mb, 6),
        )

    def _log(self, decision: PolicyDecision) -> None:
        if self.logger is not None:
            self.logger.write(decision)

    def snapshot(self) -> dict:
        return {
            "policy": self.policy,
            "gpu_budget_mb": self.gpu_budget_mb,
            "resident_mb": round(self.resident_mb, 6),
            "resident_objects": len(self.resident),
            "history_objects": len(self.history),
            "step": self.step,
        }


def decisions_to_rows(decisions: Iterable[PolicyDecision]) -> list[dict]:
    return [asdict(decision) for decision in decisions]
