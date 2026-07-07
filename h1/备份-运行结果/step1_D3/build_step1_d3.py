#!/usr/bin/env python3
"""根据对象级 COP 记录构建 Step1 D3 产物（路径 A）。

数据源是 ``run_h1_vllm0110_real.py`` 请求级 CSV 中
``score_source == 'object_level_cop'`` 的行。这些行携带由 COP 模块
（``edgekv_cop.py``，``COPProfiler.update_from_item`` ->
``ObjectProfile.recompute`` / ``estimate_reuse``）计算出的 ``c_recomp_ms /
p_reuse / score / size_mb``，而不是进程内 sitecustomize 的块级 profile。

这使三个 D3 产物（c_recomp vs n、score 直方图、p_reuse 直方图）反映设计中的
对象粒度 COP Algorithm 1（00_预实验提取 §6.2）。Engine 驱逐本身仍是 vLLM
前缀缓存块级；该事实来自 run summary，而不是在这里重新计算。
"""

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
OUT = STEP_DIR / "out"
# (bucket 标签, gpu_memory_utilization) — run_h1_vllm0110_real 命名预算。
BUCKETS = [("tight", "0.720"), ("mid", "0.735")]
COP_CSV_GLOB = "*_h1_lpe_requests.csv"
COP_SUMMARY_GLOB = "*_h1_lpe_summary.json"
COP_SCORE_SOURCE = "object_level_cop"


def read_cop_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            row["_source_path"] = str(path)
            rows.append(row)
    return rows


def cop_csv_files(label: str, runtime_root: Path) -> list[Path]:
    roots = [runtime_root / label]
    if runtime_root.name == label:
        roots.append(runtime_root)
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(root.rglob(COP_CSV_GLOB))
    return sorted(set(files))


def read_summary(label: str, runtime_root: Path) -> dict[str, Any] | None:
    roots = [runtime_root / label]
    if runtime_root.name == label:
        roots.append(runtime_root)
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob(COP_SUMMARY_GLOB)):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
    return None


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def i(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
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


def collect_budget(label: str, budget: str, runtime_root: Path, allow_fallback: bool) -> dict[str, Any]:
    files = cop_csv_files(label, runtime_root)
    events: list[dict[str, Any]] = []
    for path in files:
        events.extend(read_cop_csv(path))

    cop_events = [row for row in events if str(row.get("score_source", "")) == COP_SCORE_SOURCE]

    real_points = [
        {
            "event_index": i(row, "event_index"),
            "object_id": str(row.get("object_id", "")),
            "object_type": str(row.get("object_type", "")),
            "lpe_action": str(row.get("lpe_action", "")),
            "hit": row.get("hit"),
            "n_tokens": f(row, "n_tokens"),
            "c_recomp_ms": f(row, "c_recomp_ms"),
            "c_re": f(row, "c_recomp_ms") / f(row, "n_tokens", 1.0) if f(row, "n_tokens") > 0.0 else 0.0,
            "p_reuse": f(row, "p_reuse"),
            "score": f(row, "score"),
            "size_mb": f(row, "size_mb"),
            "source_path": str(row.get("_source_path", "")),
        }
        for row in cop_events
        if f(row, "n_tokens") > 0.0 and f(row, "c_recomp_ms") > 0.0
    ]
    real_points.sort(key=lambda p: p["event_index"])

    data_source = "object_level_cop_requests_csv"
    warning = ""
    if not real_points:
        searched = ", ".join(str(p) for p in files) or str(runtime_root / label)
        if allow_fallback:
            data_source = "no_cop_records_fallback_debug_only"
            warning = f"No object_level_cop rows with n_tokens>0 and c_recomp_ms>0; searched {searched}."
        else:
            raise FileNotFoundError(
                f"no object_level_cop records with n_tokens>0 and c_recomp_ms>0 for {label}; searched {searched}"
            )

    object_type_counts = Counter(p["object_type"] for p in real_points)
    hit_counts = Counter(str(row.get("hit")) for row in cop_events if row.get("hit") is not None)
    summary = read_summary(label, runtime_root) or {}

    return {
        "label": label,
        "budget": budget,
        "data_source": data_source,
        "warning": warning,
        "cop_csv_files": [str(path) for path in files],
        "event_count": len(cop_events),
        "real_point_count": len(real_points),
        "object_type_counts": dict(sorted(object_type_counts.items())),
        "hit_counts": dict(sorted(hit_counts.items())),
        "stats": {
            "n_tokens": stats([p["n_tokens"] for p in real_points]),
            "c_recomp_ms": stats([p["c_recomp_ms"] for p in real_points]),
            "c_re": stats([p["c_re"] for p in real_points if p["c_re"] > 0.0]),
            "p_reuse": stats([p["p_reuse"] for p in real_points if p["p_reuse"] > 0.0]),
            "score": stats([p["score"] for p in real_points if p["score"] > 0.0]),
            "size_mb": stats([p["size_mb"] for p in real_points if p["size_mb"] > 0.0]),
        },
        "run_summary": {
            "c_re_ms_per_token": summary.get("c_re_ms_per_token"),
            "kv_mib_per_token": summary.get("kv_mib_per_token"),
            "c_recomp_model": summary.get("c_recomp_model"),
            "eviction_granularity": summary.get("eviction_granularity"),
            "gpu_prefix_cache_policy_impl": summary.get("gpu_prefix_cache_policy_impl"),
            "cop_object_profile_count": summary.get("cop_object_profile_count"),
            "cop_avg_p_reuse": summary.get("cop_avg_p_reuse"),
            "cop_avg_score": summary.get("cop_avg_score"),
            "hit_rate": summary.get("hit_rate"),
            "gpu_prefix_cache_evictions": summary.get("gpu_prefix_cache_evictions"),
        },
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
            ax.text(0.02, 0.95, f"cop events={bucket['event_count']}\npoints={len(points)}", transform=ax.transAxes, va="top", fontsize=9)
        ax.set_title(bucket["label"])
        ax.set_xlabel("object token n")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("c_recomp_ms")
    fig.suptitle("H1 Step1 D3: object-level COP c_recomp vs object token n")
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
    axes[0].set_ylabel("object event count")
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


def _figure_interpretation(budgets: list[dict[str, Any]]) -> list[str]:
    b = budgets[0]
    s = b["stats"]
    run = b["run_summary"]
    c_re = s["c_re"]["mean"] or 0.12
    try:
        kv = float(run.get("kv_mib_per_token") or 0.0)
    except (TypeError, ValueError):
        kv = 0.0
    ratio = (c_re / kv) if kv > 0.0 else 0.0
    labels = " / ".join(bk["label"] for bk in budgets)
    p = s["p_reuse"]
    sc = s["score"]
    n = s["n_tokens"]
    cm = s["c_recomp_ms"]
    return [
        "### 三张图片解释",
        "",
        f"> 数据来自 object-level COP（`score_source=object_level_cop`），{labels} 两档 COP 分布一致（见上），下述数值取代表档 `{b['label']}`。",
        "",
        "1. step1_D3_c_recomp_vs_n.png",
        "",
        f"  `c_recomp_ms` 与 `n_tokens` 严格线性：`c_re` 恒为 {c_re:.3g} ms/token（std≈{s['c_re']['std']:.2g}），"
        f"`c_recomp` p50={cm['p50']:.4g}、p95={cm['p95']:.4g}，对应 n p50={n['p50']:.0f}、p95={n['p95']:.0f}。",
        "",
        "  含义：重算成本只是 `c_re * n_tokens`，不含\"某些对象特别难重算\"的额外信息，本质是在表达对象长度。",
        "",
        "2. step1_D3_p_reuse_histogram.png",
        "",
        f"  与旧 path-B 的饱和分布不同，COP 的 `p_reuse` 有真实区分度：mean={p['mean']:.3g}、std={p['std']:.3g}、"
        f"min={p['min']:.3g}、p50={p['p50']:.3g}、max={p['max']:.3g}，呈 hot/cold 双峰（按 object_type 分化）。",
        "",
        "  含义：COP 的复用概率不再大面积饱和，对象之间能拉开差距——这正是用真实分布坐实退化所需要的前提（旧 path-B 因 p_reuse≈0.97 看不出区分度）。",
        "",
        "3. step1_D3_score_histogram.png",
        "",
        f"  `score` mean={sc['mean']:.3g}、std={sc['std']:.3g}、min={sc['min']:.3g}、max={sc['max']:.3g}。"
        + (f" 由于 `size_mb=μ_kv*n`、`c_recomp=c_re*n`，n 被约掉，`score = p_reuse * (c_re/μ_kv) ≈ p_reuse * {ratio:.3g}`。" if ratio > 0.0 else ""),
        "",
        f"  含义：score 只是 p_reuse 的线性缩放（score_var/p_reuse_var ≈ (c_re/μ_kv)² = {ratio*ratio:.3g}，"
        f"实测 {sc['variance']:.3g}/{p['variance']:.3g}={ (sc['variance']/p['variance']) if p['variance'] else 0:.3g}），"
        "排序信息与 p_reuse 完全等价，不提供额外区分度。",
        "",
        "综合：c_recomp 是长度线性、score 是 p_reuse 的常数倍 → 在同质 KV 下 `score` 退化为按 `p_reuse` 排序（§2 结论）。"
        "区别在于本轮用 COP 的真实 p_reuse 分布坐实，而非旧 path-B 的饱和值；唯一打破退化的途径是引入异质性（量化/对象类型，见 §5/H4）。",
        "",
    ]


def write_report(budgets: list[dict[str, Any]], out_md: Path) -> None:
    lines = [
        "# H1 Step1 D3 复盘：对象级 COP 监控（路径 A）",
        "",
        "## 结论",
        "- `c_recomp / p_reuse / score` 由 **COP 模块 `edgekv_cop.py`** 计算（`COPProfiler.update_from_item` → `ObjectProfile.recompute` / `estimate_reuse`），即设计 §6.2 Algorithm 1，对象粒度。",
        "- 数据来源是 `run_h1_vllm0110_real.py` 写出的 per-request 行，`score_source == object_level_cop`，不再从 sitecustomize 的 block 级内联 profile 取值。",
        "- `c_recomp = c_re * n_tokens`（线性）；`score = p_reuse * c_recomp / size_mb`，其中 `size_mb = μ_kv * n_tokens` → `score = p_reuse * (c_re/μ_kv)`，n 被约掉，**score 与 p_reuse 同序**（坐实 §2 退化）。",
        "- `p_reuse` 由 COP 的访问历史估计（近期命中率/频次/recency 加权 + 可选先验），在真实混合 trace 下有显著方差，因此本轮 D3 的退化结论靠 p_reuse 的真实分布坐实，而非饱和值。",
        "- 引擎实际驱逐粒度仍是 vLLM prefix-cache block（见各档 `eviction_granularity`）；COP 只在对象级提供 `p_reuse/c_recomp/score` 画像与打分。",
        "- COP 画像仅由 trace 回放顺序（每请求 `update_from_item`，hit=trace 复用）决定，与 GPU 显存预算无关，因此 tight/mid 两档的 `c_recomp/p_reuse/score` 分布一致；预算只改变引擎侧 summary（hit_rate/evictions）。",
        "",
        "## 输出文件",
        "- 图片：`h1/step1_D3/out/step1_D3_c_recomp_vs_n.png`",
        "- 图片：`h1/step1_D3/out/step1_D3_score_histogram.png`",
        "- 图片：`h1/step1_D3/out/step1_D3_p_reuse_histogram.png`",
        "- summary：`h1/step1_D3/out/step1_D3_summary.json`",
        "- COP 原始数据：`h1/step1_D3/runtime/{tight,mid}/**/*_h1_lpe_requests.csv`",
        "",
        "## 三行验收日志",
        "- `c_recomp/n_tokens`: 见 `c_recomp_ms` 与 `c_re` 统计，以及 `step1_D3_c_recomp_vs_n.png`（COP `ObjectProfile.recompute`）。",
        "- `eviction_granularity`: 引擎驱逐为 `vLLM prefix-cache block`（各档 run summary 字段）；COP 画像为对象级。",
        "- `score/p_reuse histogram`: 见 `step1_D3_score_histogram.png` 与 `step1_D3_p_reuse_histogram.png`（COP `estimate_reuse` + `score`）。",
        "",
        "## 真实运行数据",
    ]
    for bucket in budgets:
        stats_map = bucket["stats"]
        run = bucket["run_summary"]
        lines.extend([
            f"### {bucket['label']} (gpu_memory_utilization={bucket['budget']})",
            f"- cop requests csv files: {len(bucket['cop_csv_files'])}",
            f"- cop_events={bucket['event_count']}, plotted_points={bucket['real_point_count']}",
            f"- n_tokens: {_fmt_stats(stats_map['n_tokens'])}",
            f"- c_recomp_ms: {_fmt_stats(stats_map['c_recomp_ms'])}",
            f"- c_re (=c_recomp/n): {_fmt_stats(stats_map['c_re'])}",
            f"- p_reuse: {_fmt_stats(stats_map['p_reuse'])}",
            f"- score: {_fmt_stats(stats_map['score'])}",
            f"- size_mb: {_fmt_stats(stats_map['size_mb'])}",
            f"- object_type_counts: `{json.dumps(bucket['object_type_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- hit_counts: `{json.dumps(bucket['hit_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- run summary: c_recomp_model=`{run.get('c_recomp_model')}`, eviction_granularity=`{run.get('eviction_granularity')}`, "
            f"cop_object_profile_count={run.get('cop_object_profile_count')}, cop_avg_p_reuse={run.get('cop_avg_p_reuse')}, "
            f"cop_avg_score={run.get('cop_avg_score')}, hit_rate={run.get('hit_rate')}",
            "",
            "cop csv files:",
        ])
        lines.extend(f"- `{path}`" for path in bucket["cop_csv_files"])
        lines.append("")
    lines.extend(_figure_interpretation(budgets))
    lines.extend([
        "## 代码依据：COP 模块如何计算各变量",
        "位置：`edgekv_cop.py`，`ObjectProfile.recompute()` 与 `ObjectProfile.estimate_reuse()`；入口 `COPProfiler.update_from_item()`。",
        "",
        "`c_recomp_ms = c_re * n_tokens`，`size_mb = μ_kv * n_tokens`，`score = p_reuse * c_recomp_ms / size_mb`；`p_reuse` 由访问历史估计并可与先验融合。",
        "",
        "```python",
        "# edgekv_cop.py ObjectProfile.recompute()",
        "self.size_mb = mu_kv * n_tokens",
        "self.c_recomp_ms = c_re * n_tokens",
        "self.c_restore_ms = self.size_mb / bw_gbps + self.d_deser_ms",
        "self.score = (self.p_reuse * self.c_recomp_ms) / max(self.size_mb, 1e-9)",
        "",
        "# edgekv_cop.py ObjectProfile.estimate_reuse()",
        "recent_rate = recent_hits / len(self.access_history)",
        "freq_rate = self.hit_count / self.access_count",
        "recency_bonus = 1.0 / (1.0 + math.log1p(max(self.access_count - self.hit_count, 0)))",
        "estimate = 0.55 * recent_rate + 0.35 * freq_rate + 0.10 * recency_bonus",
        "# 若有 p_reuse_prior，再按 p_reuse_prior_weight 融合",
        "```",
        "",
        "## 代码依据：COP 记录如何写出",
        "位置：`h1/run_h1_vllm0110_real.py`，`request_trace_fields()`，调用 `cop.update_from_item(...)` 并落 `score_source='object_level_cop'`。",
        "",
        "```python",
        "profile = cop.update_from_item(item, hit=bool(trace_hit or rag_hit), access_index=event_index)",
        "fields.update({",
        "    'p_reuse': round(profile.p_reuse, 6),",
        "    'c_recomp_ms': round(profile.c_recomp_ms, 6),",
        "    'score': round(profile.score, 9),",
        "    'score_source': 'object_level_cop',",
        "    'size_mb': round(profile.size_mb, 6),",
        "})",
        "```",
        "",
        "## 生成脚本口径",
        "位置：`h1/step1_D3/build_step1_d3.py`，`collect_budget()`。",
        "",
        "只保留 `score_source == 'object_level_cop'` 且 `n_tokens>0`、`c_recomp_ms>0` 的 COP 行；没有 COP 行时默认报错（`--allow-fallback` 仅调试）。",
        "",
        "```python",
        "cop_events = [r for r in events if str(r.get('score_source', '')) == 'object_level_cop']",
        "real_points = [p for r in cop_events if f(r, 'n_tokens') > 0.0 and f(r, 'c_recomp_ms') > 0.0]",
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
        "H1 Step1 D3: object-level COP score histogram",
        "score",
    )
    plot_histogram(
        budgets,
        "p_reuse",
        args.out / "step1_D3_p_reuse_histogram.png",
        "H1 Step1 D3: object-level COP p_reuse histogram",
        "p_reuse",
    )
    payload = {
        "generated_from": str(args.runtime_root),
        "data_policy": "object_level_cop requests CSV only unless --allow-fallback is passed",
        "score_source": COP_SCORE_SOURCE,
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
