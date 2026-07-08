#!/usr/bin/env python3
"""H0 同质/异质 LPE 行为的确定性解析探针。

启动命令：
    python h0/run_h0_hetero_probe.py

参数说明：
    --out-dir：输出目录根路径；每次运行会在其下创建时间戳目录。
    --seed：随机种子，控制 trace 与对象属性生成。
    --reps：每个场景/预算/策略重复次数。
    --trace-len：每条模拟 trace 的请求长度。
    --budgets：要扫描的缓存预算列表，表示可保留对象容量比例。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICIES = ("LRU", "LFU", "LPE-score")
SCENARIOS = ("homogeneous", "heterogeneous")
SUMMARY_COLUMNS = (
    "scenario",
    "budget",
    "policy",
    "rep",
    "hit_rate",
    "miss_cost",
    "p95_cost",
    "evictions",
    "cached_objects",
    "score_mean",
    "score_std",
    "p_reuse_mean",
    "p_reuse_std",
)


@dataclass(frozen=True)
class CacheObject:
    object_id: str
    object_type: str
    size: float
    recompute_cost: float
    size_factor: float
    recompute_factor: float


@dataclass
class ObjectState:
    freq: float = 0.0
    raw_count: int = 0
    last_access: int = -1
    p_reuse: float = 0.01
    score: float = 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    data = sorted(values)
    if not data:
        return 0.0
    pos = (len(data) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    if lo == hi:
        return data[lo]
    weight = pos - lo
    return data[lo] * (1.0 - weight) + data[hi] * weight


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return statistics.fmean(values), statistics.pstdev(values)


def build_objects(scenario: str) -> dict[str, CacheObject]:
    rng = random.Random(81031)
    objects: dict[str, CacheObject] = {}

    def add(prefix: str, count: int, object_type: str, base_size: float, size_factor: float, recompute_factor: float) -> None:
        for i in range(count):
            jitter = 0.88 + 0.24 * rng.random()
            size = base_size * size_factor * jitter
            recompute = base_size * recompute_factor * jitter
            object_id = f"{prefix}_{i:03d}"
            objects[object_id] = CacheObject(
                object_id=object_id,
                object_type=object_type,
                size=size,
                recompute_cost=recompute,
                size_factor=size_factor,
                recompute_factor=recompute_factor,
            )

    if scenario == "homogeneous":
        add("session", 36, "fp16_session_kv", 10.0, 1.0, 1.0)
        add("quant", 36, "int4_quant_block", 10.0, 1.0, 1.0)
        add("rag_s", 28, "rag_chunk", 6.0, 1.0, 1.0)
        add("rag_m", 20, "rag_chunk", 10.0, 1.0, 1.0)
        add("rag_l", 12, "rag_chunk", 16.0, 1.0, 1.0)
        add("cold", 120, "cold_object", 14.0, 1.0, 1.0)
    elif scenario == "heterogeneous":
        add("session", 36, "fp16_session_kv", 10.0, 1.0, 1.0)
        add("quant", 36, "int4_quant_block", 10.0, 0.25, 0.9)
        add("rag_s", 28, "rag_chunk", 6.0, 0.70, 0.85)
        add("rag_m", 20, "rag_chunk", 10.0, 1.10, 1.25)
        add("rag_l", 12, "rag_chunk", 16.0, 1.70, 2.20)
        add("cold", 120, "cold_object", 14.0, 1.40, 0.75)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return objects


def generate_trace(objects: dict[str, CacheObject], seed: int, rep: int, trace_len: int) -> list[str]:
    rng = random.Random(seed + rep * 7919)
    session_ids = sorted(k for k in objects if k.startswith("session_"))
    quant_ids = sorted(k for k in objects if k.startswith("quant_"))
    rag_hot = sorted(k for k in objects if k.startswith("rag_s_")) + sorted(k for k in objects if k.startswith("rag_m_"))[:10]
    rag_tail = sorted(k for k in objects if k.startswith("rag_l_")) + sorted(k for k in objects if k.startswith("rag_m_"))[10:]
    cold_ids = sorted(k for k in objects if k.startswith("cold_"))

    active_sessions = rng.sample(session_ids, 10)
    active_quant = rng.sample(quant_ids, 14)
    trace: list[str] = []
    for i in range(trace_len):
        phase = i % 60
        if phase < 22:
            trace.append(rng.choice(active_sessions if rng.random() < 0.80 else session_ids))
        elif phase < 42:
            trace.append(rng.choice(rag_hot if rng.random() < 0.82 else rag_tail))
        elif phase < 52:
            trace.append(rng.choice(active_quant if rng.random() < 0.76 else quant_ids))
        else:
            trace.append(rng.choice(cold_ids))
        if i and i % 240 == 0:
            active_sessions = rng.sample(session_ids, 10)
            active_quant = rng.sample(quant_ids, 14)
    return trace


def update_profile(state: ObjectState, now: int) -> None:
    if state.last_access < 0:
        gap = 64
        state.freq = 1.0
    else:
        gap = max(1, now - state.last_access)
        state.freq = state.freq * (0.985**gap) + 1.0
    state.raw_count += 1
    state.last_access = now
    recency = 1.0 / (1.0 + math.log1p(gap))
    frequency = state.freq / (state.freq + 3.0)
    state.p_reuse = max(0.001, min(0.999, 0.35 * recency + 0.65 * frequency))


def score_object(obj: CacheObject, state: ObjectState) -> float:
    return state.p_reuse * obj.recompute_cost / obj.size


def choose_victim(policy: str, cache: set[str], states: dict[str, ObjectState]) -> str:
    if policy == "LRU":
        return min(cache, key=lambda oid: states[oid].last_access)
    if policy == "LFU":
        return min(cache, key=lambda oid: (states[oid].raw_count, states[oid].last_access))
    if policy == "LPE-score":
        return min(cache, key=lambda oid: (states[oid].score, states[oid].last_access))
    raise ValueError(f"unknown policy: {policy}")


def simulate_policy(
    scenario: str,
    objects: dict[str, CacheObject],
    trace: Sequence[str],
    budget: float,
    policy: str,
    rep: int,
) -> tuple[dict, list[dict]]:
    capacity = budget * sum(obj.size for obj in objects.values())
    states = {oid: ObjectState() for oid in objects}
    cache: set[str] = set()
    cached_size = 0.0
    hits = 0
    evictions = 0
    costs: list[float] = []
    events: list[dict] = []

    for index, oid in enumerate(trace):
        obj = objects[oid]
        hit = oid in cache
        update_profile(states[oid], index)
        states[oid].score = score_object(obj, states[oid])
        evicted: list[str] = []

        if hit:
            hits += 1
            cost = obj.recompute_cost * 0.03
            action = "hit"
        else:
            cost = obj.recompute_cost
            cache.add(oid)
            cached_size += obj.size
            action = "admit"

        while cached_size > capacity and cache:
            victim = choose_victim(policy, cache, states)
            cache.remove(victim)
            cached_size -= objects[victim].size
            evictions += 1
            evicted.append(victim)
        if not hit and oid not in cache:
            action = "bypass"

        costs.append(cost)
        events.append(
            {
                "scenario": scenario,
                "budget": budget,
                "policy": policy,
                "rep": rep,
                "index": index,
                "object_id": oid,
                "object_type": obj.object_type,
                "hit": hit,
                "cost": cost,
                "size": obj.size,
                "recompute_cost": obj.recompute_cost,
                "p_reuse": states[oid].p_reuse,
                "score": states[oid].score,
                "action": action,
                "evicted": evicted,
                "cached_size": cached_size,
            }
        )

    score_mean, score_std = mean_std([states[oid].score for oid in cache])
    p_mean, p_std = mean_std([states[oid].p_reuse for oid in cache])
    miss_cost = sum(event["cost"] for event in events if not event["hit"])
    return (
        {
            "scenario": scenario,
            "budget": budget,
            "policy": policy,
            "rep": rep,
            "hit_rate": hits / len(trace),
            "miss_cost": miss_cost,
            "p95_cost": percentile(costs, 95),
            "evictions": evictions,
            "cached_objects": len(cache),
            "score_mean": score_mean,
            "score_std": score_std,
            "p_reuse_mean": p_mean,
            "p_reuse_std": p_std,
        },
        events,
    )


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def png_chunk(kind: bytes, data: bytes) -> bytes:
    payload = kind + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def write_png(path: Path, width: int, height: int, pixels: list[list[tuple[int, int, int]]]) -> None:
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for pixel in row:
            raw.extend(pixel)
    data = b"\x89PNG\r\n\x1a\n"
    data += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    data += png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def draw_line(pixels: list[list[tuple[int, int, int]]], x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        for yy in range(max(0, y - 1), min(len(pixels), y + 2)):
            for xx in range(max(0, x - 1), min(len(pixels[0]), x + 2)):
                pixels[yy][xx] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def write_line_plot(path: Path, rows: Sequence[dict], scenario: str, metric: str) -> None:
    data = [r for r in rows if r["scenario"] == scenario]
    budgets = sorted({float(r["budget"]) for r in data})
    colors = {"LRU": "#1f56a6", "LFU": "#2d824e", "LPE-score": "#be402a"}
    y_label = {
        "p95_cost": "p95 cost proxy",
        "hit_rate": "cache hit rate",
    }.get(metric, metric)

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=140)
    for policy in POLICIES:
        means: list[float] = []
        stds: list[float] = []
        for budget in budgets:
            vals = [
                float(r[metric])
                for r in data
                if r["policy"] == policy and math.isclose(float(r["budget"]), budget)
            ]
            if not vals:
                continue
            means.append(statistics.fmean(vals))
            stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        ax.plot(budgets[: len(means)], means, marker="o", linewidth=2.0, markersize=5.0, label=policy, color=colors[policy])
        if any(stds):
            lower = [m - s for m, s in zip(means, stds)]
            upper = [m + s for m, s in zip(means, stds)]
            ax.fill_between(budgets[: len(means)], lower, upper, color=colors[policy], alpha=0.12, linewidth=0)

    ax.set_title(f"{scenario}: budget vs {y_label}")
    ax.set_xlabel("cache budget fraction")
    ax.set_ylabel(y_label)
    ax.set_xticks(budgets)
    if metric == "hit_rate":
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="major", color="#d7dce2", linewidth=0.8, alpha=0.85)
    ax.legend(title="Policy", frameon=True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_scatter(path: Path, events: Sequence[dict], scenario: str) -> None:
    rows = [e for e in events if e["scenario"] == scenario and e["policy"] == "LPE-score" and e["rep"] == 0]
    rows = rows[:: max(1, len(rows) // 1600)]

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=140)
    colors = {
        "fp16_session_kv": "#1f56a6",
        "int4_quant_block": "#be402a",
        "rag_chunk": "#2d824e",
        "cold_object": "#7d8793",
    }
    for object_type in sorted({str(e["object_type"]) for e in rows}):
        typed = [e for e in rows if e["object_type"] == object_type]
        ax.scatter(
            [float(e["p_reuse"]) for e in typed],
            [float(e["score"]) for e in typed],
            s=13,
            alpha=0.58,
            label=object_type,
            color=colors.get(object_type, "#555555"),
            edgecolors="none",
        )

    ax.set_title(f"{scenario}: LPE score vs p_reuse")
    ax.set_xlabel("estimated p_reuse")
    ax.set_ylabel("LPE score = p_reuse * recompute_cost / size")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, which="major", color="#d7dce2", linewidth=0.8, alpha=0.85)
    ax.legend(title="Object type", frameon=True, markerscale=1.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_conclusion(path: Path, rows: Sequence[dict]) -> None:
    def avg(scenario: str, budget: float, policy: str, metric: str) -> float:
        vals = [
            float(r[metric])
            for r in rows
            if r["scenario"] == scenario and math.isclose(float(r["budget"]), budget) and r["policy"] == policy
        ]
        return statistics.fmean(vals) if vals else 0.0

    budgets = sorted({float(r["budget"]) for r in rows})
    effective: list[float] = []
    hetero_wins = 0
    hetero_lines: list[str] = []
    homo_lines: list[str] = []
    for budget in budgets:
        homo_lru = avg("homogeneous", budget, "LRU", "p95_cost")
        homo_lpe = avg("homogeneous", budget, "LPE-score", "p95_cost")
        homo_lines.append(f"| {budget:.2f} | {homo_lru:.6f} | {homo_lpe:.6f} | {homo_lpe - homo_lru:.6f} |")
        hit = avg("heterogeneous", budget, "LRU", "hit_rate")
        if 0.5 <= hit <= 0.85:
            effective.append(budget)
            lru = avg("heterogeneous", budget, "LRU", "p95_cost")
            lpe = avg("heterogeneous", budget, "LPE-score", "p95_cost")
            if lpe < lru:
                hetero_wins += 1
            hetero_lines.append(f"| {budget:.2f} | {hit:.6f} | {lru:.6f} | {lpe:.6f} | {lru - lpe:.6f} |")

    verdict = (
        "LPE score needs heterogeneous objects to become useful; the next step should move toward H4 quality/quantization evidence."
        if hetero_wins >= 2
        else "The current score formula should be reviewed before it is used as an H4 scheduling rule."
    )
    content = [
        "# H0 Heterogeneous Probe Conclusion",
        "",
        f"- Effective window budgets: {', '.join(f'{b:.2f}' for b in effective) or 'none'}",
        f"- Heterogeneous LPE wins in effective window: {hetero_wins}",
        f"- Verdict: {verdict}",
        "",
        "## Homogeneous Degeneration",
        "",
        "| budget | LRU p95_cost | LPE p95_cost | LPE-LRU |",
        "| ---: | ---: | ---: | ---: |",
        *homo_lines,
        "",
        "## Heterogeneous Effective Window",
        "",
        "| budget | LRU hit_rate | LRU p95_cost | LPE p95_cost | LRU-LPE |",
        "| ---: | ---: | ---: | ---: | ---: |",
        *(hetero_lines or ["| n/a | n/a | n/a | n/a | n/a |"]),
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    if out_dir.name == "h0_hetero_probe":
        out_dir = out_dir / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    event_rows: list[dict] = []
    for scenario in SCENARIOS:
        objects = build_objects(scenario)
        for rep in range(args.reps):
            trace = generate_trace(objects, args.seed, rep, args.trace_len)
            for budget in args.budgets:
                for policy in POLICIES:
                    summary, events = simulate_policy(scenario, objects, trace, budget, policy, rep)
                    summary_rows.append(summary)
                    event_rows.extend(events)

    write_csv(out_dir / "summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_jsonl(out_dir / "events.jsonl", event_rows)
    for scenario in SCENARIOS:
        write_line_plot(out_dir / f"{scenario}_p95_cost.png", summary_rows, scenario, "p95_cost")
        write_line_plot(out_dir / f"{scenario}_hit_rate.png", summary_rows, scenario, "hit_rate")
        write_scatter(out_dir / f"{scenario}_score_vs_p_reuse.png", event_rows, scenario)
    write_conclusion(out_dir / "conclusion.md", summary_rows)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="h0/results/h0_hetero_probe")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--trace-len", type=int, default=1800)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.35, 0.45, 0.60])
    return parser.parse_args()


def main() -> None:
    print(run(parse_args()))


if __name__ == "__main__":
    main()
