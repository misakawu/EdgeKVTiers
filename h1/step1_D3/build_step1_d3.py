#!/usr/bin/env python3
"""Build Step1 D3 artifacts from real LPE runtime monitor JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
STEP_DIR = ROOT / "h1" / "step1_D3"
DEFAULT_RUNTIME = STEP_DIR / "runtime"
ARCHIVE = ROOT / "h1" / "out" / "run_test"
OUT = STEP_DIR / "out"
BUCKETS = [("tight", "0.710"), ("mid", "0.720")]
MONITOR_NAMES = {"runtime_monitor.jsonl", "edgekv_lpe_runtime_monitor.jsonl"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSON in {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                row["_source_path"] = str(path)
                rows.append(row)
    return rows


def monitor_files_for(label: str, runtime_root: Path) -> list[Path]:
    roots = [runtime_root / label]
    if runtime_root.name == label:
        roots.append(runtime_root)
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            if path.name in MONITOR_NAMES or "monitor" in path.name:
                files.append(path)
    return sorted(set(files))


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def i(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "variance": 0.0, "std": 0.0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    def pct(q: float) -> float:
        idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
        return float(ordered[idx])
    mean = statistics.fmean(values)
    var = statistics.pvariance(values) if len(values) > 1 else 0.0
    return {"count": len(values), "mean": mean, "variance": var, "std": math.sqrt(var), "min": float(ordered[0]), "p50": pct(0.5), "p95": pct(0.95), "max": float(ordered[-1])}


def read_summary_row(label: str) -> dict[str, Any] | None:
    path = ARCHIVE / label / "step3_summary.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("policy") == "h1_lpe":
                return row
    return None


def collect_budget(label: str, budget: str, runtime_root: Path, allow_fallback: bool) -> dict[str, Any]:
    files = monitor_files_for(label, runtime_root)
    events: list[dict[str, Any]] = []
    for path in files:
        events.extend(load_jsonl(path))

    real_points = [
        {
            "seq": i(row, "seq"),
            "object_id": str(row.get("object_id", "")),
            "lpe_action": str(row.get("lpe_action", "")),
            "hit": row.get("hit"),
            "n_tokens": f(row, "n_tokens"),
            "c_recomp_ms": f(row, "c_recomp_ms", f(row, "c_recomp", 0.0)),
            "c_re": f(row, "c_re", f(row, "c_re_ms_per_token", 0.12)),
            "p_reuse": f(row, "p_reuse"),
            "score": f(row, "score"),
            "source_path": str(row.get("_source_path", "")),
        }
        for row in events
        if f(row, "n_tokens") > 0.0 and f(row, "c_recomp_ms", f(row, "c_recomp", 0.0)) > 0.0
    ]

    data_source = "runtime_monitor_jsonl"
    warning = ""
    if not real_points and allow_fallback:
        data_source = "aggregate_stats_fallback_not_runtime"
        warning = "No real runtime monitor records found; fallback is for debugging only."
    elif not real_points:
        searched = ", ".join(str(p) for p in files) or str(runtime_root / label)
        raise FileNotFoundError(f"no real LPE monitor records with n_tokens>0 and c_recomp>0 for {label}; searched {searched}")

    action_counts = Counter(str(row.get("lpe_action", "")) for row in events)
    hit_values = [row.get("hit") for row in events if row.get("hit") is not None]
    hit_counts = Counter(str(v) for v in hit_values)
    summary_row = read_summary_row(label)

    return {
        "label": label,
        "budget": budget,
        "data_source": data_source,
        "warning": warning,
        "monitor_files": [str(path) for path in files],
        "event_count": len(events),
        "real_point_count": len(real_points),
        "action_counts": dict(sorted(action_counts.items())),
        "hit_counts": dict(sorted(hit_counts.items())),
        "stats": {
            "n_tokens": stats([p["n_tokens"] for p in real_points]),
            "c_recomp_ms": stats([p["c_recomp_ms"] for p in real_points]),
            "c_re": stats([p["c_re"] for p in real_points]),
            "p_reuse": stats([p["p_reuse"] for p in real_points if p["p_reuse"] > 0.0]),
            "score": stats([p["score"] for p in real_points]),
        },
        "summary_csv_row": summary_row,
        "points": real_points,
    }


def plot(budgets: list[dict[str, Any]], out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = ["#1f77b4", "#d55e00"]
    for ax, bucket, color in zip(axes, budgets, colors, strict=True):
        points = bucket["points"]
        xs = [float(p["n_tokens"]) for p in points]
        ys = [float(p["c_recomp_ms"]) for p in points]
        ax.scatter(xs, ys, s=18, alpha=0.55, color=color, edgecolors="none")
        if xs:
            slope = bucket["stats"]["c_re"]["mean"] or 0.12
            x_max = max(xs)
            line_x = [x_max * k / 100.0 for k in range(101)]
            ax.plot(line_x, [slope * x for x in line_x], color="#222222", linewidth=1.4, label=f"c_re={slope:.3g} ms/token")
            ax.legend(loc="upper left", fontsize=8, frameon=False)
            ax.text(0.02, 0.95, f"events={bucket['event_count']}\npoints={len(points)}", transform=ax.transAxes, va="top", fontsize=9)
        ax.set_title(bucket["label"])
        ax.set_xlabel("object token n")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("c_recomp_ms")
    fig.suptitle("H1 Step1 D3: real LPE runtime c_recomp vs object token n")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(budgets: list[dict[str, Any]], out_md: Path) -> None:
    lines = ["# H1 Step1 D3 Runtime Monitor", ""]
    lines += ["数据来源：`EDGEKV_H1_RUNTIME_MONITOR=1` 写出的真实运行时 JSONL。图中点只来自 `n_tokens>0` 且 `c_recomp>0` 的事件。", ""]
    for bucket in budgets:
        ps = bucket["stats"]["p_reuse"]
        ss = bucket["stats"]["score"]
        lines += [
            f"## {bucket['label']}",
            f"- monitor_files: {', '.join(bucket['monitor_files']) or 'missing'}",
            f"- events: {bucket['event_count']}, plotted_points: {bucket['real_point_count']}",
            f"- lpe_action_counts: `{json.dumps(bucket['action_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- hit_counts: `{json.dumps(bucket['hit_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- p_reuse: mean={ps['mean']:.6g}, variance={ps['variance']:.6g}, std={ps['std']:.6g}",
            f"- score: mean={ss['mean']:.6g}, variance={ss['variance']:.6g}, std={ss['std']:.6g}",
            "",
        ]
    lines += [
        "## 计算口径",
        "`c_recomp_ms = c_re * n_tokens`，除非请求元信息显式传入 `c_recomp_ms`。默认 `c_re` 来自 `EDGEKV_C_RE_MS_PER_TOKEN`，当前默认值为 0.12 ms/token。",
        "`p_reuse` 由命中频率、未命中 recency 项和对象类型先验加权得到，随后 clamp 到 `[0.01, 0.99]`。",
        "`score = p_reuse * c_recomp_ms / size_mb`；当对象尚无可用 KV size 时 score 为 0。",
        "LPE 驱逐实际作用在 vLLM prefix cache block/free queue 上；profile 是对象级，驱逐候选和 `_maybe_evict_cached_block` 都是 block 粒度。",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--allow-fallback", action="store_true", help="debug only; do not use for final D3 evidence")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    budgets = [collect_budget(label, budget, args.runtime_root, args.allow_fallback) for label, budget in BUCKETS]
    plot(budgets, args.out / "step1_D3_c_recomp_vs_n.png")
    payload = {
        "generated_from": str(args.runtime_root),
        "data_policy": "runtime JSONL only unless --allow-fallback is passed",
        "budgets": budgets,
        "derived": {
            b["label"]: {
                "p_reuse_mean": b["stats"]["p_reuse"]["mean"],
                "p_reuse_variance": b["stats"]["p_reuse"]["variance"],
                "score_mean": b["stats"]["score"]["mean"],
                "score_variance": b["stats"]["score"]["variance"],
            }
            for b in budgets
        },
    }
    (args.out / "step1_D3_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_report(budgets, STEP_DIR / "step1_d3_report.md")


if __name__ == "__main__":
    main()
