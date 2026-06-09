#!/usr/bin/env python3
"""Reusable analytical simulator toolkit for EdgeKVTiers pre-experiments.

The module intentionally has no command-line parser and no H1-H5 batch entry.
Experiment files should import this toolkit, build/load a trace, then call
``run_one`` or ``Simulator`` directly.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_SHAREGPT_PATH = Path(
    r"E:\DATASET\ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
)

POLICIES = ("lru", "lfu", "pgdsf", "score", "tiered", "static-int4", "static-sparse-k")
RRS_MODES = ("rrs", "always-restore", "always-recompute")
TIER_ORDER = ("full", "int8", "int4", "sparse_k")
TIERS = {
    "full": {"size_factor": 1.00, "qloss_per_token": 0.000, "restore_factor": 1.00},
    "int8": {"size_factor": 0.50, "qloss_per_token": 0.002, "restore_factor": 1.00},
    "int4": {"size_factor": 0.25, "qloss_per_token": 0.008, "restore_factor": 1.05},
    "sparse_k": {"size_factor": 0.20, "qloss_per_token": 0.015, "restore_factor": 1.10},
}


@dataclass(frozen=True)
class KVObject:
    object_id: str
    n_tokens: int
    reuse_key: str
    object_type: str
    p_reuse: float


@dataclass(frozen=True)
class Request:
    request_id: str
    object_id: str
    n_uncached: int
    arrival_ts: float
    reuse_key: str
    session_id: str = ""
    turn_index: int = 0
    admit_object_id: Optional[str] = None


@dataclass
class ResidentEntry:
    obj: KVObject
    q: str
    last_access: float
    access_count: int = 0


@dataclass(frozen=True)
class SimConfig:
    mu_kv_mb_per_token: float = 0.12
    c_re_ms_per_token: float = 0.12
    d_deser_ms: float = 3.0
    seed: int = 1
    m_budget_mb: float = 800.0
    bw_gbps: float = 8.0
    epsilon: float = 4.0
    offload_keep_threshold: float = 0.35
    trace_requests: int = 500
    warmup_requests: int = 50


@dataclass
class RunMetrics:
    policy: str
    rrs_mode: str
    m_budget_mb: float
    bw_gbps: float
    epsilon: float
    epsilon_abs: float
    epsilon_norm: float
    token_ref: int
    requests: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_mean_ms: float
    hit_rate: float
    resident_hit_rate: float
    offload_hit_rate: float
    recompute_ratio: float
    restore_ratio: float
    qloss_current: float
    qloss_peak: float
    qloss_current_abs: float
    qloss_peak_abs: float
    qloss_current_norm: float
    qloss_peak_norm: float
    memory_peak_mb: float
    migrations: int
    evictions: int
    offloads: int
    drops: int
    epsilon_ok: bool


def size_mb(obj: KVObject, q: str, cfg: SimConfig) -> float:
    return cfg.mu_kv_mb_per_token * obj.n_tokens * TIERS[q]["size_factor"]


def qloss(obj: KVObject, q: str) -> float:
    return obj.n_tokens * TIERS[q]["qloss_per_token"]


def token_ref_for_objects(objects: Sequence[KVObject]) -> int:
    return sum(obj.n_tokens for obj in objects)


def normalize_qloss(value: float, token_ref: int) -> float:
    return value / max(token_ref, 1)


def denormalize_epsilon(epsilon_norm: float, token_ref: int) -> float:
    return epsilon_norm * max(token_ref, 1)


def c_recomp_ms(obj: KVObject, q: str, cfg: SimConfig) -> float:
    return cfg.c_re_ms_per_token * obj.n_tokens * TIERS[q]["restore_factor"]


def c_restore_ms(obj: KVObject, q: str, cfg: SimConfig) -> float:
    # With BW in GB/s and size in MB, 1 GB/s is approximately 1 MB/ms.
    return size_mb(obj, q, cfg) / max(cfg.bw_gbps, 1e-9) + cfg.d_deser_ms


def keep_score(obj: KVObject, q: str, cfg: SimConfig) -> float:
    denom = max(size_mb(obj, q, cfg), 1e-9)
    return obj.p_reuse * c_recomp_ms(obj, q, cfg) / denom


def rrs_action(obj: KVObject, q: str, cfg: SimConfig) -> str:
    return "restore" if c_restore_ms(obj, q, cfg) <= c_recomp_ms(obj, q, cfg) else "recompute"


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def config_with(cfg: SimConfig, **updates: object) -> SimConfig:
    return replace(cfg, **updates)


def estimate_tokens(text: str) -> int:
    parts = re.findall(r"\w+|[^\w\s]", text or "", flags=re.UNICODE)
    return max(1, int(math.ceil(len(parts) * 1.15)))


def _message_role(message: dict) -> str:
    return str(message.get("from", message.get("role", ""))).lower()


def _message_text(message: dict) -> str:
    value = message.get("value", message.get("content", ""))
    return value if isinstance(value, str) else str(value)


_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "but",
    "can",
    "could",
    "for",
    "from",
    "have",
    "how",
    "into",
    "main",
    "more",
    "that",
    "the",
    "their",
    "then",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "write",
    "summarize",
    "explain",
}


def _topic_key(text: str) -> str:
    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", text or "")
        if w.lower() not in _STOPWORDS
    ]
    if not words:
        return "generic"
    counts: Dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return max(counts, key=lambda word: (counts[word], len(word), word))[:32]


def load_sharegpt_trace(
    path: Path = DEFAULT_SHAREGPT_PATH,
    max_sessions: int = 200,
    min_human_turns: int = 2,
    start_ts: float = 0.0,
) -> Tuple[List[KVObject], List[Request]]:
    """Convert ShareGPT V3 JSON conversations into prefix-cache requests.

    Each human turn accesses the previous session prefix when it exists and
    admits the extended prefix for the following turn. Token counts use a local
    heuristic so this loader stays independent of external tokenizers.
    """

    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    session_turns: List[dict] = []

    for raw_session_idx, row in enumerate(rows):
        conversations = row.get("conversations", [])
        if not isinstance(conversations, list):
            continue

        human_messages = [
            msg for msg in conversations if _message_role(msg) in {"human", "user"}
        ]
        if len(human_messages) < min_human_turns:
            continue

        session_id = str(row.get("id", f"sharegpt_{raw_session_idx:06d}"))
        prefix_tokens = 0
        turns: List[dict] = []
        human_turn_idx = 0

        for msg_idx, msg in enumerate(conversations):
            role = _message_role(msg)
            text_tokens = estimate_tokens(_message_text(msg))
            if role in {"human", "user"}:
                new_prefix_tokens = prefix_tokens + text_tokens
                turns.append(
                    {
                        "session_id": session_id,
                        "turn_index": human_turn_idx,
                        "human_text": _message_text(msg),
                        "human_tokens": text_tokens,
                        "prefix_tokens": new_prefix_tokens,
                        "human_turns": len(human_messages),
                    }
                )
                prefix_tokens = new_prefix_tokens
                human_turn_idx += 1
            elif msg_idx > 0 and role in {"gpt", "assistant"}:
                prefix_tokens += text_tokens

        if turns:
            session_turns.append({"session_id": session_id, "turns": turns})
        if len(session_turns) >= max_sessions:
            break

    object_specs: Dict[str, dict] = {}
    requests: List[Request] = []
    topic_counts: Dict[str, int] = {}
    ts = start_ts
    session_window = 32

    for batch_start in range(0, len(session_turns), session_window):
        batch = session_turns[batch_start : batch_start + session_window]
        max_turns = max((len(session["turns"]) for session in batch), default=0)
        for turn_index in range(max_turns):
            for session in batch:
                turns = session["turns"]
                if turn_index >= len(turns):
                    continue
                turn = turns[turn_index]
                session_id = turn["session_id"]
                topic = _topic_key(turn["human_text"])
                topic_bucket = sum(ord(ch) for ch in topic) % 24
                topic_id = f"rag:topic_bucket:{topic_bucket:02d}"
                topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
                object_specs.setdefault(
                    topic_id,
                    {
                        "n_tokens": max(48, min(220, estimate_tokens(turn["human_text"]) * 2)),
                        "reuse_key": topic,
                        "object_type": "rag_topic_chunk",
                        "uses": 0,
                        "base_reuse": 0.55,
                    },
                )
                object_specs[topic_id]["uses"] += 1

                ts += 0.25
                requests.append(
                    Request(
                        request_id=f"{session_id}:turn:{turn_index:03d}:rag",
                        object_id=topic_id,
                        n_uncached=max(4, turn["human_tokens"] // 10),
                        arrival_ts=ts,
                        reuse_key=topic,
                        session_id=session_id,
                        turn_index=turn_index,
                    )
                )

                current_prefix_id = f"{session_id}:session_prefix"
                previous_prefix_id = current_prefix_id
                future_uses = max(0, int(turn["human_turns"]) - turn_index - 1)
                prefix_spec = object_specs.setdefault(
                    current_prefix_id,
                    {
                        "n_tokens": 0,
                        "reuse_key": session_id,
                        "object_type": "sharegpt_session_prefix",
                        "uses": 0,
                        "base_reuse": 0.15,
                    },
                )
                prefix_spec["n_tokens"] = max(int(prefix_spec["n_tokens"]), int(turn["prefix_tokens"]))
                prefix_spec["uses"] += 1
                prefix_spec["base_reuse"] = max(
                    float(prefix_spec["base_reuse"]),
                    min(0.9, max(0.15, future_uses / max(int(turn["human_turns"]), 1))),
                )

                ts += 0.75
                requests.append(
                    Request(
                        request_id=f"{session_id}:turn:{turn_index:03d}:prefix",
                        object_id=previous_prefix_id,
                        n_uncached=int(turn["human_tokens"]),
                        arrival_ts=ts,
                        reuse_key=session_id,
                        session_id=session_id,
                        turn_index=turn_index,
                        admit_object_id=current_prefix_id,
                    )
                )

    max_uses = max((int(spec["uses"]) for spec in object_specs.values()), default=1)
    objects = [
        KVObject(
            object_id=object_id,
            n_tokens=int(spec["n_tokens"]),
            reuse_key=str(spec["reuse_key"]),
            object_type=str(spec["object_type"]),
            p_reuse=min(
                0.95,
                max(0.03, float(spec["base_reuse"]) + 0.35 * int(spec["uses"]) / max_uses),
            ),
        )
        for object_id, spec in object_specs.items()
    ]
    return objects, requests


def load_jsonl_trace(path: Path) -> Tuple[List[KVObject], List[Request]]:
    objects: Dict[str, KVObject] = {}
    requests: List[Request] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            object_id = str(row["object_id"])
            n_tokens = int(row.get("n_tokens", row.get("prompt_len", 128)))
            reuse_key = str(row.get("reuse_key", object_id))
            object_type = str(row.get("object_type", "external"))
            p_reuse = float(row.get("p_reuse", 0.5))
            objects.setdefault(
                object_id,
                KVObject(object_id, n_tokens, reuse_key, object_type, p_reuse),
            )

            admit_object_id = row.get("admit_object_id")
            if admit_object_id is not None:
                admit_object_id = str(admit_object_id)
                objects.setdefault(
                    admit_object_id,
                    KVObject(
                        admit_object_id,
                        int(row.get("admit_n_tokens", n_tokens)),
                        str(row.get("admit_reuse_key", reuse_key)),
                        str(row.get("admit_object_type", object_type)),
                        float(row.get("admit_p_reuse", p_reuse)),
                    ),
                )

            requests.append(
                Request(
                    request_id=str(row.get("request_id", f"req_{i:06d}")),
                    object_id=object_id,
                    n_uncached=int(row.get("n_uncached", max(1, n_tokens // 4))),
                    arrival_ts=float(row.get("arrival_ts", i)),
                    reuse_key=reuse_key,
                    session_id=str(row.get("session_id", "")),
                    turn_index=int(row.get("turn_index", 0)),
                    admit_object_id=admit_object_id,
                )
            )
    return list(objects.values()), requests


def make_synthetic_trace(
    seed: int = 1,
    n_requests: int = 500,
) -> Tuple[List[KVObject], List[Request]]:
    rng = random.Random(seed)
    specs = [
        ("prefix", 22, (48, 180), 0.78),
        ("session", 36, (64, 260), 0.55),
        ("rag", 52, (32, 160), 0.48),
        ("longtail", 90, (16, 96), 0.14),
    ]
    objects: List[KVObject] = []
    for object_type, count, token_range, base_reuse in specs:
        for i in range(count):
            n_tokens = rng.randint(*token_range)
            p_reuse = min(0.95, max(0.03, rng.gauss(base_reuse, 0.12)))
            object_id = f"{object_type}_{i:03d}"
            objects.append(KVObject(object_id, n_tokens, object_id, object_type, p_reuse))

    weights = [max(obj.p_reuse, 0.02) ** 1.6 for obj in objects]
    requests: List[Request] = []
    ts = 0.0
    while len(requests) < n_requests:
        obj = rng.choices(objects, weights=weights, k=1)[0]
        ts += rng.expovariate(1.0 / 3.0)
        n_uncached = rng.randint(4, max(8, obj.n_tokens // 5))
        requests.append(
            Request(
                request_id=f"req_{len(requests):06d}",
                object_id=obj.object_id,
                n_uncached=n_uncached,
                arrival_ts=ts,
                reuse_key=obj.reuse_key,
            )
        )
        if rng.random() < 0.18 and obj.object_type in {"prefix", "rag"}:
            for _ in range(rng.randint(1, 4)):
                if len(requests) >= n_requests:
                    break
                ts += rng.uniform(0.05, 0.8)
                requests.append(
                    Request(
                        request_id=f"req_{len(requests):06d}",
                        object_id=obj.object_id,
                        n_uncached=rng.randint(2, max(4, obj.n_tokens // 8)),
                        arrival_ts=ts,
                        reuse_key=obj.reuse_key,
                    )
                )
    return objects, requests


def load_default_trace(
    max_sessions: int = 200,
    fallback_to_synthetic: bool = True,
    cfg: Optional[SimConfig] = None,
) -> Tuple[List[KVObject], List[Request]]:
    if DEFAULT_SHAREGPT_PATH.exists():
        return load_sharegpt_trace(DEFAULT_SHAREGPT_PATH, max_sessions=max_sessions)
    if fallback_to_synthetic:
        cfg = cfg or SimConfig()
        return make_synthetic_trace(cfg.seed, cfg.trace_requests)
    raise FileNotFoundError(f"ShareGPT trace not found: {DEFAULT_SHAREGPT_PATH}")


class Simulator:
    def __init__(
        self,
        objects: Sequence[KVObject],
        requests: Sequence[Request],
        cfg: SimConfig,
        policy: str,
        rrs_mode: str = "rrs",
        emit_events: bool = False,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        if rrs_mode not in RRS_MODES:
            raise ValueError(f"unknown rrs_mode: {rrs_mode}")
        self.objects = {obj.object_id: obj for obj in objects}
        self.requests = list(requests)
        self.cfg = cfg
        self.policy = policy
        self.rrs_mode = rrs_mode
        self.emit_events = emit_events
        self.resident: Dict[str, ResidentEntry] = {}
        self.offloaded: Dict[str, ResidentEntry] = {}
        self.events: List[dict] = []
        self.token_ref = token_ref_for_objects(objects)
        self.migrations = 0
        self.evictions = 0
        self.offloads = 0
        self.drops = 0
        self.qloss_peak = 0.0
        self.memory_peak = 0.0

    def memory_used(self) -> float:
        return sum(size_mb(entry.obj, entry.q, self.cfg) for entry in self.resident.values())

    def quality_used(self) -> float:
        entries = list(self.resident.values()) + list(self.offloaded.values())
        return sum(qloss(entry.obj, entry.q) for entry in entries)

    def choose_initial_tier(self, obj: KVObject) -> str:
        if self.policy == "static-int4":
            return "int4"
        if self.policy == "static-sparse-k":
            return "sparse_k"
        return "full"

    def choose_rrs_action(self, obj: KVObject, q: str) -> str:
        if self.rrs_mode == "always-restore":
            return "restore"
        if self.rrs_mode == "always-recompute":
            return "recompute"
        return rrs_action(obj, q, self.cfg)

    def next_tier(self, q: str) -> Optional[str]:
        idx = TIER_ORDER.index(q)
        if idx + 1 >= len(TIER_ORDER):
            return None
        return TIER_ORDER[idx + 1]

    def downgrade_once(self) -> bool:
        candidates = [
            entry for entry in self.resident.values() if self.next_tier(entry.q) is not None
        ]
        candidates.sort(key=lambda entry: keep_score(entry.obj, entry.q, self.cfg))
        for entry in candidates:
            next_q = self.next_tier(entry.q)
            if next_q is None:
                continue
            delta_q = qloss(entry.obj, next_q) - qloss(entry.obj, entry.q)
            if self.quality_used() + delta_q <= self.cfg.epsilon + 1e-9:
                entry.q = next_q
                self.migrations += 1
                return True
        return False

    def victim_id(self) -> str:
        if self.policy == "lru":
            return min(self.resident, key=lambda oid: self.resident[oid].last_access)
        if self.policy == "lfu":
            return min(
                self.resident,
                key=lambda oid: (
                    self.resident[oid].access_count,
                    self.resident[oid].last_access,
                ),
            )
        if self.policy == "pgdsf":
            return min(
                self.resident,
                key=lambda oid: (
                    (self.resident[oid].access_count + 1)
                    * self.resident[oid].obj.p_reuse
                    * c_recomp_ms(self.resident[oid].obj, self.resident[oid].q, self.cfg)
                    / max(size_mb(self.resident[oid].obj, self.resident[oid].q, self.cfg), 1e-9)
                ),
            )
        return min(
            self.resident,
            key=lambda oid: keep_score(self.resident[oid].obj, self.resident[oid].q, self.cfg),
        )

    def relieve_pressure(self) -> None:
        while self.memory_used() > self.cfg.m_budget_mb + 1e-9 and self.resident:
            if self.policy == "tiered" and self.downgrade_once():
                continue
            victim_id = self.victim_id()
            victim = self.resident.pop(victim_id)
            self.evictions += 1
            if victim.obj.p_reuse >= self.cfg.offload_keep_threshold:
                self.offloaded[victim_id] = victim
                self.offloads += 1
            else:
                self.drops += 1

    def _miss_entry(self, obj: KVObject, req: Request) -> Tuple[ResidentEntry, float]:
        entry = ResidentEntry(obj, self.choose_initial_tier(obj), req.arrival_ts, 0)
        ttft = self.cfg.c_re_ms_per_token * max(obj.n_tokens, req.n_uncached)
        return entry, ttft

    def _lookup(self, req: Request) -> Tuple[ResidentEntry, float, bool, str, str]:
        obj = self.objects[req.object_id]
        base_ms = self.cfg.c_re_ms_per_token * req.n_uncached
        if req.object_id in self.resident:
            entry = self.resident[req.object_id]
            return entry, base_ms, True, entry.q, "none"
        if req.object_id in self.offloaded:
            entry = self.offloaded.pop(req.object_id)
            action = self.choose_rrs_action(obj, entry.q)
            if action == "restore":
                ttft = base_ms + c_restore_ms(obj, entry.q, self.cfg)
            else:
                ttft = base_ms + c_recomp_ms(obj, entry.q, self.cfg)
                entry.q = self.choose_initial_tier(obj)
            return entry, ttft, True, entry.q, action
        entry, ttft = self._miss_entry(obj, req)
        return entry, ttft, False, "miss", "recompute"

    def replay(self) -> RunMetrics:
        ttfts: List[float] = []
        hits = resident_hits = offload_hits = 0
        restores = recomputes = 0

        measured_hits = measured_resident_hits = measured_offload_hits = 0
        measured_restores = measured_recomputes = 0

        for req_idx, req in enumerate(self.requests):
            if req.object_id not in self.objects:
                raise KeyError(f"request references unknown object_id: {req.object_id}")
            entry, ttft, hit, q_before, action = self._lookup(req)

            if hit:
                hits += 1
                if action == "none":
                    resident_hits += 1
                else:
                    offload_hits += 1
            if action == "restore":
                restores += 1
            if action == "recompute":
                recomputes += 1

            entry.access_count += 1
            entry.last_access = req.arrival_ts
            self.resident[req.object_id] = entry

            admit_object_id = req.admit_object_id or req.object_id
            if admit_object_id not in self.objects:
                raise KeyError(f"request references unknown admit_object_id: {admit_object_id}")
            if admit_object_id != req.object_id:
                admit_obj = self.objects[admit_object_id]
                self.resident[admit_object_id] = ResidentEntry(
                    admit_obj,
                    self.choose_initial_tier(admit_obj),
                    req.arrival_ts,
                    1,
                )

            self.relieve_pressure()
            if admit_object_id in self.resident:
                q_after = self.resident[admit_object_id].q
            elif admit_object_id in self.offloaded:
                q_after = f"offload:{self.offloaded[admit_object_id].q}"
            else:
                q_after = "drop"

            current_quality = self.quality_used()
            current_memory = self.memory_used()
            self.qloss_peak = max(self.qloss_peak, current_quality)
            self.memory_peak = max(self.memory_peak, current_memory)
            if req_idx >= self.cfg.warmup_requests:
                ttfts.append(ttft)
                if hit:
                    measured_hits += 1
                    if action == "none":
                        measured_resident_hits += 1
                    else:
                        measured_offload_hits += 1
                if action == "restore":
                    measured_restores += 1
                if action == "recompute":
                    measured_recomputes += 1

            if self.emit_events:
                self.events.append(
                    {
                        "request_id": req.request_id,
                        "object_id": req.object_id,
                        "admit_object_id": admit_object_id,
                        "arrival_ts": req.arrival_ts,
                        "reuse_key": req.reuse_key,
                        "session_id": req.session_id,
                        "turn_index": req.turn_index,
                        "policy": self.policy,
                        "q_before": q_before,
                        "q_after": q_after,
                        "rrs_action": action,
                        "ttft_ms": round(ttft, 6),
                        "hit": hit,
                        "bw_gbps": self.cfg.bw_gbps,
                        "token_ref": self.token_ref,
                        "epsilon_abs": round(self.cfg.epsilon, 6),
                        "epsilon_norm": round(normalize_qloss(self.cfg.epsilon, self.token_ref), 9),
                        "qloss_current": round(current_quality, 6),
                        "qloss_current_abs": round(current_quality, 6),
                        "qloss_current_norm": round(
                            normalize_qloss(current_quality, self.token_ref),
                            9,
                        ),
                        "memory_current_mb": round(current_memory, 6),
                    }
                )

        n = max(len(ttfts), 1)
        qloss_current_abs = self.quality_used()
        qloss_peak_abs = self.qloss_peak
        epsilon_norm = normalize_qloss(self.cfg.epsilon, self.token_ref)
        return RunMetrics(
            policy=self.policy,
            rrs_mode=self.rrs_mode,
            m_budget_mb=self.cfg.m_budget_mb,
            bw_gbps=self.cfg.bw_gbps,
            epsilon=self.cfg.epsilon,
            epsilon_abs=self.cfg.epsilon,
            epsilon_norm=epsilon_norm,
            token_ref=self.token_ref,
            requests=len(ttfts),
            ttft_p50_ms=percentile(ttfts, 50),
            ttft_p95_ms=percentile(ttfts, 95),
            ttft_mean_ms=statistics.mean(ttfts) if ttfts else 0.0,
            hit_rate=measured_hits / n,
            resident_hit_rate=measured_resident_hits / n,
            offload_hit_rate=measured_offload_hits / n,
            recompute_ratio=measured_recomputes / n,
            restore_ratio=measured_restores / n,
            qloss_current=qloss_current_abs,
            qloss_peak=qloss_peak_abs,
            qloss_current_abs=qloss_current_abs,
            qloss_peak_abs=qloss_peak_abs,
            qloss_current_norm=normalize_qloss(qloss_current_abs, self.token_ref),
            qloss_peak_norm=normalize_qloss(qloss_peak_abs, self.token_ref),
            memory_peak_mb=self.memory_peak,
            migrations=self.migrations,
            evictions=self.evictions,
            offloads=self.offloads,
            drops=self.drops,
            epsilon_ok=self.qloss_peak <= self.cfg.epsilon + 1e-9,
        )


def run_one(
    objects: Sequence[KVObject],
    requests: Sequence[Request],
    cfg: Optional[SimConfig] = None,
    policy: str = "tiered",
    rrs_mode: str = "rrs",
    emit_events: bool = False,
) -> Tuple[RunMetrics, List[dict]]:
    sim = Simulator(objects, requests, cfg or SimConfig(), policy, rrs_mode, emit_events)
    metrics = sim.replay()
    return metrics, sim.events


def metrics_row(metrics: RunMetrics, experiment: str = "") -> dict:
    row = asdict(metrics)
    if experiment:
        row["experiment"] = experiment
    for key, value in list(row.items()):
        if isinstance(value, float):
            row[key] = round(value, 6)
    return row


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


__all__ = [
    "DEFAULT_SHAREGPT_PATH",
    "POLICIES",
    "RRS_MODES",
    "TIER_ORDER",
    "TIERS",
    "KVObject",
    "Request",
    "ResidentEntry",
    "SimConfig",
    "RunMetrics",
    "Simulator",
    "c_recomp_ms",
    "c_restore_ms",
    "config_with",
    "estimate_tokens",
    "keep_score",
    "load_default_trace",
    "load_jsonl_trace",
    "load_sharegpt_trace",
    "make_synthetic_trace",
    "metrics_row",
    "percentile",
    "denormalize_epsilon",
    "normalize_qloss",
    "qloss",
    "rrs_action",
    "run_one",
    "size_mb",
    "token_ref_for_objects",
    "write_csv",
    "write_json",
]
