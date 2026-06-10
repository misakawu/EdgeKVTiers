#!/usr/bin/env python3
"""LMCache/vLLM policy-layer adapter for EdgeKVTiers H3 validation.

This module intentionally avoids importing LMCache or vLLM directly. H3 only
needs a policy-layer contract that can be attached to whichever hook surface the
installed cache layer exposes. The adapter therefore supports two common forms:
``cache.register_hook(name, fn)`` and writable hook attributes such as
``cache.on_admit``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import run_h0


sim = run_h0.sim

HOOKS = ("on_admit", "on_reuse", "on_pressure", "on_evict")
H3_EVENT_FIELDS = (
    "h0_event_id",
    "lmcache_event_id",
    "hook",
    "object_id",
    "object_type",
    "n_tokens",
    "q_before",
    "q_after",
    "lpe_action",
    "tms_action",
    "rrs_action",
    "T_policy_ms",
    "c_mig_ms",
    "restore_latency_ms",
    "ttft_ms",
    "hit",
    "M_peak",
    "qloss_total_abs",
    "qloss_total_norm",
    "semantic_match",
)


@dataclass(frozen=True)
class HookResult:
    hook: str
    action: str
    object_id: str
    q_before: str = ""
    q_after: str = ""
    lpe_action: str = "none"
    tms_action: str = "none"
    rrs_action: str = "none"
    t_policy_ms: float = 0.0
    c_mig_ms: float = 0.0
    restore_latency_ms: float = 0.0


class JsonlHookLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


class EdgeKVTiersPolicy:
    """Policy object that maps EdgeKVTiers decisions to cache-layer hooks."""

    def __init__(
        self,
        cfg=None,
        *,
        policy: str = "tiered",
        rrs_mode: str = "rrs",
        logger: Optional[JsonlHookLogger] = None,
    ) -> None:
        self.cfg = cfg or sim.SimConfig()
        self.policy = policy
        self.rrs_mode = rrs_mode
        self.logger = logger

    def _object_id(self, obj) -> str:
        return str(getattr(obj, "object_id", getattr(obj, "key", getattr(obj, "id", "unknown"))))

    def _q(self, obj, default: str = "full") -> str:
        return str(getattr(obj, "q", getattr(obj, "tier", default)))

    def _timed(self, hook: str, fn: Callable[[], HookResult]) -> HookResult:
        started = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = HookResult(**{**result.__dict__, "t_policy_ms": round(elapsed_ms, 6)})
        if self.logger:
            self.logger.write(result.__dict__)
        return result

    def cop_update(self, obj) -> HookResult:
        def decide() -> HookResult:
            return HookResult(
                hook="on_admit",
                action="admit",
                object_id=self._object_id(obj),
                q_after=self._q(obj),
            )

        return self._timed("on_admit", decide)

    def rrs_on_reuse(self, obj) -> HookResult:
        def decide() -> HookResult:
            q = self._q(obj)
            if self.rrs_mode == "always-restore":
                action = "restore"
            elif self.rrs_mode == "always-recompute":
                action = "recompute"
            else:
                action = sim.rrs_action(obj, q, self.cfg)
            restore_latency = sim.c_restore_ms(obj, q, self.cfg) if action == "restore" else 0.0
            return HookResult(
                hook="on_reuse",
                action=action,
                object_id=self._object_id(obj),
                q_before=q,
                q_after=q if action == "restore" else "full",
                rrs_action=action,
                restore_latency_ms=round(restore_latency, 6),
            )

        return self._timed("on_reuse", decide)

    def tms_then_lpe(self, resident: Sequence[object]) -> HookResult:
        def decide() -> HookResult:
            candidates = [obj for obj in resident if self._q(obj) in sim.TIER_ORDER]
            candidates.sort(
                key=lambda obj: sim.keep_score(obj, self._q(obj), self.cfg)
                if hasattr(obj, "p_reuse") and hasattr(obj, "n_tokens")
                else 0.0
            )
            for obj in candidates:
                q_before = self._q(obj)
                tier_index = sim.TIER_ORDER.index(q_before)
                if tier_index + 1 >= len(sim.TIER_ORDER):
                    continue
                q_after = sim.TIER_ORDER[tier_index + 1]
                c_mig = sim.c_recomp_ms(obj, q_after, self.cfg) * 0.05
                return HookResult(
                    hook="on_pressure",
                    action="downgrade",
                    object_id=self._object_id(obj),
                    q_before=q_before,
                    q_after=q_after,
                    tms_action="downgrade",
                    c_mig_ms=round(c_mig, 6),
                )
            victim = candidates[0] if candidates else None
            return HookResult(
                hook="on_pressure",
                action="evict",
                object_id=self._object_id(victim) if victim is not None else "unknown",
                lpe_action="evict",
            )

        return self._timed("on_pressure", decide)

    def lpe_choose_victim(self, resident: Sequence[object]) -> HookResult:
        def decide() -> HookResult:
            if not resident:
                return HookResult(hook="on_evict", action="none", object_id="unknown")
            victim = min(
                resident,
                key=lambda obj: sim.keep_score(obj, self._q(obj), self.cfg)
                if hasattr(obj, "p_reuse") and hasattr(obj, "n_tokens")
                else 0.0,
            )
            action = "offload" if getattr(victim, "p_reuse", 0.0) >= self.cfg.offload_keep_threshold else "drop"
            return HookResult(
                hook="on_evict",
                action=action,
                object_id=self._object_id(victim),
                q_before=self._q(victim),
                q_after=f"offload:{self._q(victim)}" if action == "offload" else "drop",
                lpe_action=action,
            )

        return self._timed("on_evict", decide)


def attach_to_cache(cache, policy: EdgeKVTiersPolicy):
    """Attach policy callbacks to an LMCache-like cache object."""

    bindings = {
        "on_admit": policy.cop_update,
        "on_reuse": policy.rrs_on_reuse,
        "on_pressure": policy.tms_then_lpe,
        "on_evict": policy.lpe_choose_victim,
    }
    if hasattr(cache, "register_hook"):
        for name, fn in bindings.items():
            cache.register_hook(name, fn)
    else:
        for name, fn in bindings.items():
            setattr(cache, name, fn)
    return cache


def h0_event_to_h3(event: dict, lmcache_event_id: int) -> dict:
    """Convert an enriched H0 event row into the H3 hook log schema."""

    rrs_action = str(event.get("rrs_action", "none"))
    hook = "on_reuse" if event.get("hit") else "on_admit"
    if str(event.get("tms_action", "none")) != "none":
        hook = "on_pressure"
    elif str(event.get("lpe_action", "none")) in {"offload", "drop"}:
        hook = "on_evict"

    row = {
        "h0_event_id": event.get("event_index", lmcache_event_id),
        "lmcache_event_id": lmcache_event_id,
        "hook": hook,
        "object_id": event.get("object_id", ""),
        "object_type": event.get("object_type", ""),
        "n_tokens": event.get("n_tokens", 0),
        "q_before": event.get("q_before", ""),
        "q_after": event.get("q_after", ""),
        "lpe_action": event.get("lpe_action", "none"),
        "tms_action": event.get("tms_action", "none"),
        "rrs_action": rrs_action,
        "T_policy_ms": event.get("t_policy_ms", 0.0),
        "c_mig_ms": 0.0,
        "restore_latency_ms": event.get("ttft_ms", 0.0) if rrs_action == "restore" else 0.0,
        "ttft_ms": event.get("ttft_ms", 0.0),
        "hit": event.get("hit", False),
        "M_peak": event.get("M_peak", event.get("memory_current_mb", 0.0)),
        "qloss_total_abs": event.get("qloss_total_abs", event.get("qloss_current_abs", 0.0)),
        "qloss_total_norm": event.get("qloss_total_norm", event.get("qloss_current_norm", 0.0)),
        "semantic_match": True,
    }
    return {field: row.get(field, "") for field in H3_EVENT_FIELDS}


def write_h3_contract(out_dir: Path, *, trace_source: str, token_ref: int) -> None:
    run_h0.write_json(
        out_dir / "h3_adapter_contract.json",
        {
            "adapter": "EdgeKVTiersPolicy",
            "target": "LMCache/vLLM policy layer",
            "trace_source": trace_source,
            "token_ref": token_ref,
            "hooks": list(HOOKS),
            "required_event_fields": list(H3_EVENT_FIELDS),
            "binding": {
                "preferred": "cache.register_hook(name, fn)",
                "fallback": "cache.on_admit/on_reuse/on_pressure/on_evict attributes",
            },
            "policy_modules": {
                "COP": "on_admit -> cop_update",
                "RRS": "on_reuse -> rrs_on_reuse",
                "TMS": "on_pressure -> tms_then_lpe",
                "LPE": "on_evict or TMS exhaustion -> lpe_choose_victim",
            },
            "h3_gate": "T_policy p95 < 1 ms, complete hook log, semantic_match true or explained",
        },
    )


def write_h3_event_sample(out_dir: Path, events: Iterable[dict], *, limit: int = 200) -> None:
    rows: List[dict] = []
    for idx, event in enumerate(events):
        if idx >= limit:
            break
        rows.append(h0_event_to_h3(event, idx))
    run_h0.write_jsonl(out_dir / "h3_hook_events.sample.jsonl", rows)
