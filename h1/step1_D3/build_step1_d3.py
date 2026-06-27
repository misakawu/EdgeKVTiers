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


def plot_histogram(budgets: list[dict[str, Any]], field: str, out_png: Path, title: str, xlabel: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = ["#2a9d8f", "#e76f51"]
    for ax, bucket, color in zip(axes, budgets, colors, strict=True):
        values = [float(p[field]) for p in bucket["points"] if float(p[field]) > 0.0]
        if values:
            ax.hist(values, bins=min(40, max(8, int(math.sqrt(len(values))))), color=color, alpha=0.82)
            stats_map = bucket["stats"][field]
            ax.axvline(stats_map["p50"], color="#222222", linewidth=1.2, label=f"p50={stats_map['p50']:.3g}")
            ax.axvline(stats_map["p95"], color="#555555", linewidth=1.0, linestyle="--", label=f"p95={stats_map['p95']:.3g}")
            ax.legend(loc="upper right", fontsize=8, frameon=False)
            ax.text(0.02, 0.95, f"count={len(values)}\nstd={stats_map['std']:.3g}", transform=ax.transAxes, va="top", fontsize=9)
        ax.set_title(bucket["label"])
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("event count")
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fmt_stats(values: dict[str, float]) -> str:
    return (
        f"count={int(values['count'])}, mean={values['mean']:.6g}, "
        f"variance={values['variance']:.6g}, std={values['std']:.6g}, "
        f"min={values['min']:.6g}, p50={values['p50']:.6g}, "
        f"p95={values['p95']:.6g}, max={values['max']:.6g}"
    )


def write_report(budgets: list[dict[str, Any]], out_md: Path) -> None:
    lines = [
        "# H1 Step1 D3 复盘：LPE 真实运行时监控",
        "",
        "## 结论",
        "- `c_recomp` 在实际代码中是对象 profile 字段 `c_recomp_ms`，默认按 `c_re * n_tokens` 线性计算。",
        "- KVCache 的 LPE 驱逐执行粒度是 vLLM prefix-cache block，不是对象整体。对象级 profile 只提供 `p_reuse/c_recomp/score` 给 block 排序和诊断。",
        "- `score = p_reuse * c_recomp_ms / size_mb`；`p_reuse` 由命中频率、miss recency 和对象类型先验加权得到。",
        "- 图和 summary 使用 `h1/step1_D3/runtime/{tight,mid}/**/runtime_monitor.jsonl` 的真实运行时事件，不从聚合 stats 复制散点。",
        "",
        "## 输出文件",
        "- 图片：`h1/step1_D3/out/step1_D3_c_recomp_vs_n.png`",
        "- 图片：`h1/step1_D3/out/step1_D3_score_histogram.png`",
        "- 图片：`h1/step1_D3/out/step1_D3_p_reuse_histogram.png`",
        "- summary：`h1/step1_D3/out/step1_D3_summary.json`",
        "- runtime 原始数据：`h1/step1_D3/runtime/{tight,mid}/**/runtime_monitor.jsonl`",
        "",
        "## 三行验收日志",
        "- `c_recomp/n_tokens`: 见 `c_recomp_ms` 与 `c_re` 统计，以及 `step1_D3_c_recomp_vs_n.png`。",
        "- `eviction_granularity`: LPE 执行粒度为 `vLLM prefix-cache block`，runtime 事件包含 `block_id/block_start/block_end`。",
        "- `score/p_reuse histogram`: 见 `step1_D3_score_histogram.png` 与 `step1_D3_p_reuse_histogram.png`。",
        "",
        "## 真实运行数据",
    ]
    for bucket in budgets:
        stats_map = bucket["stats"]
        lines.extend([
            f"### {bucket['label']}",
            f"- runtime monitor files: {len(bucket['monitor_files'])}",
            f"- events={bucket['event_count']}, plotted_points={bucket['real_point_count']}",
            f"- n_tokens: {_fmt_stats(stats_map['n_tokens'])}",
            f"- c_recomp_ms: {_fmt_stats(stats_map['c_recomp_ms'])}",
            f"- c_re: {_fmt_stats(stats_map['c_re'])}",
            f"- p_reuse: {_fmt_stats(stats_map['p_reuse'])}",
            f"- score: {_fmt_stats(stats_map['score'])}",
            f"- lpe_action_counts: `{json.dumps(bucket['action_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- hit_counts: `{json.dumps(bucket['hit_counts'], ensure_ascii=False, sort_keys=True)}`",
            "",
            "runtime files:",
        ])
        lines.extend(f"- `{path}`" for path in bucket["monitor_files"])
        lines.append("")
    lines.extend([
        "## 代码依据：runtime 监控字段如何写出",
        "位置：`h1/sitecustomize.py`，函数 `_edgekv_record_lpe_monitor()`，当前约第 231 行。",
        "",
        "`_edgekv_record_lpe_monitor()` 在 LPE 策略开启且设置 `EDGEKV_H1_RUNTIME_MONITOR_PATH` 后写 JSONL。`n_tokens/c_recomp/p_reuse/score` 来自真实运行中的 object profile；`lpe_action/hit/block_id` 来自当前 hook 事件。",
        "",
        "```python",
        "c_re = _edgekv_env_float('EDGEKV_C_RE_MS_PER_TOKEN', 0.12)",
        "record = {'lpe_action': str(lpe_action), 'hit': hit, 'block_id': block_id, 'c_re': c_re}",
        "if profile is not None:",
        "    c_recomp_ms = float(profile.get('c_recomp_ms', 0.0) or 0.0)",
        "    record.update({'n_tokens': int(profile.get('n_tokens', 0) or 0), 'c_recomp': c_recomp_ms, 'p_reuse': float(profile.get('p_reuse', 0.0) or 0.0), 'score': float(profile.get('score', 0.0) or 0.0)})",
        "```",
        "",
        "## 代码依据：c_recomp 如何计算",
        "位置：`h1/sitecustomize.py`，对象 profile 更新逻辑，当前约第 715 行。",
        "",
        "`c_recomp` 实际写出的值就是 `profile['c_recomp_ms']`。如果请求 meta 没有显式 `c_recomp_ms`，代码按 `c_re * n_tokens` 计算。",
        "",
        "```python",
        "c_re = _edgekv_env_float('EDGEKV_C_RE_MS_PER_TOKEN', 0.12)",
        "n_tokens = max(int(n_tokens), 1)",
        "c_recomp_ms = float(meta.get('c_recomp_ms', 0.0) or 0.0) or (c_re * n_tokens)",
        "profile.update({'n_tokens': n_tokens, 'c_recomp_ms': c_recomp_ms})",
        "```",
        "",
        "## 代码依据：p_reuse 与 score 如何计算",
        "位置：`h1/sitecustomize.py`，对象 profile 评分逻辑，当前约第 760-792 行。",
        "",
        "`p_reuse` 是命中比例、miss recency 项、对象类型先验三者加权；默认权重为 `0.55/0.30/0.15`。`score` 在对象有 resident size 后按收益/容量计算。",
        "",
        "```python",
        "p_freq = _edgekv_clamp(hit_count / access_count, 0.0, 1.0)",
        "p_recency = 1.0 / (1.0 + math.log1p(misses))",
        "p_type = _edgekv_object_type_prior(str(profile.get('object_type', 'unknown')))",
        "p_reuse = ((w_freq * p_freq) + (w_recency * p_recency) + (w_type * p_type)) / weight_sum",
        "profile['p_reuse'] = _edgekv_clamp(p_reuse, 0.01, 0.99)",
        "profile['score'] = profile['p_reuse'] * profile['c_recomp_ms'] / max(size_mb, 1e-9)",
        "```",
        "",
        "## 代码依据：LPE 按 block 驱逐",
        "位置：`h1/sitecustomize.py`，free queue 候选记录当前约第 1163 行；`_maybe_evict_cached_block()` hook 当前约第 1398 行。",
        "",
        "驱逐候选从 free queue/rank heap 中选出的是 `block`，真正 evict hook 是 `_maybe_evict_cached_block(self, block)`。因此执行粒度是 block；对象 profile 通过 block 映射提供分数。",
        "",
        "```python",
        "block = _edgekv_heap_valid_block(pool, item)",
        "block_id = _edgekv_block_id(block)",
        "_edgekv_record_lpe_monitor('reorder_candidate', block_id=block_id, profile=_edgekv_block_profile(pool, block_id))",
        "",
        "def _maybe_evict_cached_block(self, block):",
        "    evicted = original_maybe_evict_cached_block(self, block)",
        "    block_id = _edgekv_block_id(block)",
        "    profile = _edgekv_block_profile(self, block_id, kv_cache_group_id)",
        "    _edgekv_record_lpe_monitor('evict', block_id=block_id, profile=profile, hit=False, evicted=True)",
        "```",
        "",
        "## 生成脚本口径",
        "位置：`h1/step1_D3/build_step1_d3.py`，`collect_budget()` 中 `real_points` 过滤逻辑，当前约第 100 行。",
        "",
        "`build_step1_d3.py` 只保留 `n_tokens>0` 且 `c_recomp_ms/c_recomp>0` 的真实事件点；没有真实事件时默认报错。",
        "",
        "```python",
        "real_points = [p for row in events if f(row, 'n_tokens') > 0.0 and f(row, 'c_recomp_ms', f(row, 'c_recomp', 0.0)) > 0.0]",
        "```",
    ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--allow-fallback", action="store_true", help="debug only; do not use for final D3 evidence")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    budgets = [collect_budget(label, budget, args.runtime_root, args.allow_fallback) for label, budget in BUCKETS]
    plot(budgets, args.out / "step1_D3_c_recomp_vs_n.png")
    plot_histogram(
        budgets,
        "score",
        args.out / "step1_D3_score_histogram.png",
        "H1 Step1 D3: real LPE runtime score histogram",
        "score",
    )
    plot_histogram(
        budgets,
        "p_reuse",
        args.out / "step1_D3_p_reuse_histogram.png",
        "H1 Step1 D3: real LPE runtime p_reuse histogram",
        "p_reuse",
    )
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
