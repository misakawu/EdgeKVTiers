#!/usr/bin/env python3
# 测试数据契约：本仓库测试必须使用 JSONL replay trace 文件作为 workload 数据，不使用 vLLM 内置数据集/测试数据。
"""生成 H1 trace 候选，运行 LRU hit 扫描，并归档结果。

默认流程：

    python h1/run_find_hit.py

这会运行参数组 A-E。对每个组：
  1. 生成 data/edgekv_traces/sharegpt_hotpotqa_session.jsonl；
  2. 在预算 0.77、0.80、0.83、0.86、0.88、0.90 上运行 h1_lru；
  3. 写出包含 trace 摘要和曲线判断的 find_hit_report.csv/json；
  4. 用共享备份名备份 trace 和运行输出。

最终候选对比可使用 --groups E --policies h1_lru h1_lpe。
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import _runner as R
import run_step3_budget_tiers as step3


TRACE = Path("data/edgekv_traces/sharegpt_hotpotqa_session.jsonl")
BASE_OUT = Path("h1/out/find_hit")
TRACE_BACKUP_ROOT = Path("data/edgekv_traces/备份-实验数据")
RESULT_BACKUP_ROOT = Path("h1/备份-运行结果")
BUDGETS = ["0.77", "0.80", "0.83", "0.86", "0.88", "0.90"]
FIXED_TRACE_ARGS = {
    "scan_cold_objects": 24,
    "cold_context_words": 800,
    "scan_probe_rounds": 3,
    "hot_repeats": 4,
    "rag_hot_repeats": 4,
    "random_seed": 2026,
}


@dataclass(frozen=True)
class Group:
    name: str
    scan_hot_objects: int
    hot_context_words: int
    sharegpt_groups: int
    hot_ratio: float
    rag_requests: int


GROUPS: dict[str, Group] = {
    "A": Group("A", 70, 500, 200, 0.30, 160),
    "B": Group("B", 90, 500, 260, 0.32, 200),
    "C": Group("C", 110, 500, 320, 0.34, 240),
    "D": Group("D", 90, 700, 260, 0.32, 200),
    "E": Group("E", 110, 700, 320, 0.34, 240),
}


def _trace_summary_path(trace: Path) -> Path:
    return trace.with_suffix(trace.suffix + ".summary.json")


def _fmt_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _read_cell_metrics(summary_json: Path) -> dict[str, Any]:
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    p95 = float(data.get("ttft_proxy_p95_ms", 0.0) or 0.0)
    qwait_p95 = float(data.get("queue_wait_p95_ms", 0.0) or 0.0)
    ratio = data.get("queue_wait_p95_ratio")
    if ratio in (None, ""):
        ratio = (qwait_p95 / p95) if p95 > 0.0 else 0.0
    return {
        "ok": bool(data.get("ok", True)),
        "requests": int(data.get("requests", 0) or 0),
        "error": str(data.get("error", "") or ""),
        "hit_rate": float(data.get("hit_rate", 0.0) or 0.0),
        "p95_ttft_ms": p95,
        "queue_wait_p95_ms": qwait_p95,
        "qwait_ratio": float(ratio or 0.0),
    }


def _run_trace_generator(group: Group, trace: Path) -> None:
    args = [
        sys.executable,
        "scripts/optimize_h0_pressure_trace.py",
        "--out", str(trace),
        "--scan-hot-objects", str(group.scan_hot_objects),
        "--scan-cold-objects", str(FIXED_TRACE_ARGS["scan_cold_objects"]),
        "--scan-probe-rounds", str(FIXED_TRACE_ARGS["scan_probe_rounds"]),
        "--hot-repeats", str(FIXED_TRACE_ARGS["hot_repeats"]),
        "--rag-hot-repeats", str(FIXED_TRACE_ARGS["rag_hot_repeats"]),
        "--hot-context-words", str(group.hot_context_words),
        "--cold-context-words", str(FIXED_TRACE_ARGS["cold_context_words"]),
        "--sharegpt-groups", str(group.sharegpt_groups),
        "--hot-ratio", str(group.hot_ratio),
        "--rag-requests", str(group.rag_requests),
        "--random-seed", str(FIXED_TRACE_ARGS["random_seed"]),
    ]
    R.log(f"[find_hit] generate trace group={group.name}")
    subprocess.run(args, cwd=R.ROOT, check=True)


def _run_group(group: Group, budgets: list[str], policies: list[str], visible_devices: str,
               num_prompts: int, max_num_batched_tokens: int, force: bool) -> Path:
    group_out = BASE_OUT / group.name
    step3.run_step3(
        tier="sweep",
        base_out=group_out,
        budgets=budgets,
        policies=policies,
        num_prompts=num_prompts,
        visible_devices=visible_devices,
        max_num_batched_tokens=max_num_batched_tokens,
        no_finalize=True,
        keep_cells=True,
        force=force,
    )
    return group_out


def _collect(group_out: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_budget: dict[str, dict[str, dict[str, Any]]] = {}
    for summary_json in sorted((group_out / "sweep").glob("*/*/*_summary.json")):
        budget = summary_json.parent.parent.name
        policy = summary_json.parent.name
        by_budget.setdefault(budget, {})[policy] = _read_cell_metrics(summary_json)
    return by_budget


def _budget_value(budget: str) -> float:
    return float(budget)


def _judge_curve(by_budget: dict[str, dict[str, dict[str, Any]]], reference_policy: str) -> dict[str, Any]:
    points = [
        (budget, by_budget.get(budget, {}).get(reference_policy, {}).get("hit_rate"))
        for budget in BUDGETS
    ]
    present = [(b, float(h)) for b, h in points if h is not None]
    if len(present) < 2:
        return {
            "smooth": False,
            "slope": None,
            "cliff_budget": None,
            "backup_name": "xx_incomplete",
            "reason": "insufficient reference-policy points",
        }

    hits = [h for _, h in present]
    jumps = [
        (present[i][0], present[i + 1][0], present[i + 1][1] - present[i][1])
        for i in range(len(present) - 1)
    ]
    max_jump = max(jumps, key=lambda item: item[2]) if jumps else (None, None, 0.0)
    monotonic = all(jump >= -1e-6 for _, _, jump in jumps)
    first_budget, first_hit = present[0]
    last_budget, last_hit = present[-1]
    slope = (last_hit - first_hit) / (_budget_value(last_budget) - _budget_value(first_budget))
    distinct_levels = len({round(hit, 2) for hit in hits})

    hit_080 = by_budget.get("0.80", {}).get(reference_policy, {}).get("hit_rate")
    cliff_budget = None
    if hit_080 is not None and float(hit_080) >= 0.85:
        cliff_budget = "0.80"
    elif max_jump[2] > 0.25:
        cliff_budget = max_jump[1]

    smooth = (
        monotonic
        and 0.45 <= first_hit <= 0.55
        and 0.85 <= last_hit <= 0.95
        and distinct_levels >= 3
        and max_jump[2] <= 0.20
    )
    if smooth:
        backup_name = f"k_hit-{slope:.1f}"
        reason = "smooth monotonic LRU hit curve"
    else:
        if cliff_budget is None:
            cliff_budget = max_jump[1]
        backup_name = f"xx{cliff_budget}"
        reason = (
            f"not smooth: monotonic={monotonic}, first_hit={first_hit:.4f}, "
            f"last_hit={last_hit:.4f}, distinct_levels={distinct_levels}, "
            f"max_jump={max_jump[2]:.4f}"
        )

    return {
        "smooth": smooth,
        "slope": slope,
        "cliff_budget": cliff_budget,
        "backup_name": backup_name,
        "reason": reason,
        "monotonic": monotonic,
        "max_jump": max_jump[2],
        "distinct_levels": distinct_levels,
    }


def _params_payload(group: Group, policies: list[str], budgets: list[str], backup_name: str) -> dict[str, Any]:
    return {
        "group": asdict(group),
        "fixed_trace_args": FIXED_TRACE_ARGS,
        "budgets": budgets,
        "policies": policies,
        "backup_name": backup_name,
    }


def _write_params(group_out: Path, payload: dict[str, Any]) -> None:
    (group_out / "params.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"group={payload['group']['name']}",
        f"backup_name={payload['backup_name']}",
        f"budgets={' '.join(payload['budgets'])}",
        f"policies={' '.join(payload['policies'])}",
    ]
    for key, value in payload["fixed_trace_args"].items():
        lines.append(f"{key}={value}")
    for key, value in payload["group"].items():
        if key != "name":
            lines.append(f"{key}={value}")
    (group_out / "params.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report(group: Group, group_out: Path, by_budget: dict[str, dict[str, dict[str, Any]]],
            trace_summary: dict[str, Any], judgement: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for budget in sorted(by_budget, key=_budget_value):
        for policy in sorted(by_budget[budget]):
            metrics = by_budget[budget][policy]
            rows.append({
                "budget": budget,
                "policy": policy,
                "status": "ok" if metrics["ok"] and metrics["requests"] > 0 else "failed",
                "requests": metrics["requests"],
                "hit_rate": _fmt_float(metrics["hit_rate"]),
                "p95_ttft_ms": round(metrics["p95_ttft_ms"], 3),
                "queue_wait_p95_ms": round(metrics["queue_wait_p95_ms"], 3),
                "qwait_ratio": _fmt_float(metrics["qwait_ratio"]),
                "estimated_hot_working_set_tokens": trace_summary.get("estimated_hot_working_set_tokens"),
                "estimated_cold_scan_tokens_per_round": json.dumps(
                    trace_summary.get("estimated_cold_scan_tokens_per_round", []),
                    ensure_ascii=False,
                ),
                "smooth": judgement["smooth"],
                "slope": None if judgement["slope"] is None else round(judgement["slope"], 6),
                "cliff_budget": judgement["cliff_budget"],
                "backup_name": judgement["backup_name"],
                "error": metrics["error"],
            })

    csv_path = group_out / "find_hit_report.csv"
    fields = [
        "budget", "policy", "status", "requests", "hit_rate", "p95_ttft_ms",
        "queue_wait_p95_ms", "qwait_ratio", "estimated_hot_working_set_tokens",
        "estimated_cold_scan_tokens_per_round", "smooth", "slope",
        "cliff_budget", "backup_name", "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json_path = group_out / "find_hit_report.json"
    json_path.write_text(json.dumps({
        "group": group.name,
        "trace_summary": trace_summary,
        "judgement": judgement,
        "cells": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    R.log(f"[find_hit] wrote {csv_path} / {json_path}")


def _unique_backup_name(name: str, group_name: str, overwrite: bool) -> str:
    if overwrite:
        return name
    if not (TRACE_BACKUP_ROOT / name).exists() and not (RESULT_BACKUP_ROOT / name).exists():
        return name
    candidate = f"{name}-{group_name}"
    if not (TRACE_BACKUP_ROOT / candidate).exists() and not (RESULT_BACKUP_ROOT / candidate).exists():
        return candidate
    index = 2
    while True:
        candidate = f"{name}-{group_name}-{index}"
        if not (TRACE_BACKUP_ROOT / candidate).exists() and not (RESULT_BACKUP_ROOT / candidate).exists():
            return candidate
        index += 1


def _copytree_replace(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(dst)
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.copytree(src, dst)


def _backup(trace: Path, group_out: Path, backup_name: str, overwrite: bool) -> None:
    trace_backup = TRACE_BACKUP_ROOT / backup_name
    result_backup = RESULT_BACKUP_ROOT / backup_name
    trace_backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trace, trace_backup / trace.name)
    shutil.copy2(_trace_summary_path(trace), trace_backup / _trace_summary_path(trace).name)
    shutil.copy2(group_out / "params.json", trace_backup / "params.json")
    shutil.copy2(group_out / "params.txt", trace_backup / "params.txt")

    RESULT_BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    _copytree_replace(group_out, result_backup, overwrite=overwrite)
    shutil.copy2(_trace_summary_path(trace), result_backup / _trace_summary_path(trace).name)
    R.log(f"[find_hit] backup trace={trace_backup} result={result_backup}")


def _selected_groups(names: list[str]) -> list[Group]:
    selected: list[Group] = []
    for name in names:
        key = name.upper()
        if key not in GROUPS:
            raise SystemExit(f"unknown group {name!r}; choices: {' '.join(GROUPS)}")
        selected.append(GROUPS[key])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", nargs="+", default=list(GROUPS), help="parameter groups to run: A B C D E")
    parser.add_argument("--budgets", nargs="+", default=BUDGETS)
    parser.add_argument("--policies", nargs="+", default=["h1_lru"])
    parser.add_argument("--reference-policy", default="h1_lru")
    parser.add_argument("--visible-devices", default=step3.DEVICES)
    parser.add_argument("--num-prompts", type=int, default=step3.MAX_REQUESTS)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--skip-trace-generation", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun cells even if summary JSON exists")
    parser.add_argument("--overwrite-backups", action="store_true")
    args = parser.parse_args()

    for group in _selected_groups(args.groups):
        if not args.skip_trace_generation:
            _run_trace_generator(group, TRACE)
        if not TRACE.exists() or not _trace_summary_path(TRACE).exists():
            raise SystemExit(f"missing trace or summary for group {group.name}: {TRACE}")

        group_out = BASE_OUT / group.name
        group_out.mkdir(parents=True, exist_ok=True)
        if not args.skip_run:
            group_out = _run_group(
                group,
                args.budgets,
                args.policies,
                args.visible_devices,
                args.num_prompts,
                args.max_num_batched_tokens,
                args.force,
            )

        by_budget = _collect(group_out)
        judgement = _judge_curve(by_budget, args.reference_policy)
        backup_name = _unique_backup_name(judgement["backup_name"], group.name, args.overwrite_backups)
        judgement = {**judgement, "backup_name": backup_name}
        trace_summary = json.loads(_trace_summary_path(TRACE).read_text(encoding="utf-8"))
        params = _params_payload(group, args.policies, args.budgets, backup_name)
        if R.DRY_RUN:
            R.log(f"[dry-run] would write reports under {group_out}")
            R.log(f"[dry-run] would backup as {backup_name}")
        else:
            _write_params(group_out, params)
            _report(group, group_out, by_budget, trace_summary, judgement)
            _backup(TRACE, group_out, backup_name, overwrite=args.overwrite_backups)

        ref_hits = {
            budget: round(by_budget.get(budget, {}).get(args.reference_policy, {}).get("hit_rate", 0.0), 4)
            for budget in args.budgets
        }
        print(json.dumps({
            "group": group.name,
            "backup_name": backup_name,
            "smooth": judgement["smooth"],
            "cliff_budget": judgement["cliff_budget"],
            "slope": None if judgement["slope"] is None else round(judgement["slope"], 4),
            "reference_hits": ref_hits,
            "reason": judgement["reason"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
