#!/usr/bin/env python3
# 测试数据契约：本仓库测试必须使用 JSONL replay trace 文件作为 workload 数据，不使用 vLLM 内置数据集/测试数据。
"""H1 第二步「旋钮3：调 workload 复用率」扫描器。

在 budget 下扫受启动下限限制、batch 下扫只能去饱和后，本脚本固定已去饱和的
`replay_batch_size=8`，从 frozen pressure trace 派生一组低复用 trace。派生方式是按
`unique_fraction` 将部分请求的 prompt 开头注入 per-request 唯一 marker，同时改写
`reuse_key/session_id/rag_reuse_key`。vLLM GPU prefix cache 按 token 前缀命中，metadata-only 改写不会影响
真实 hit；默认直接改 prompt 前缀来降低复用率。

先用 LRU 找 `hit_rate=0.5~0.85 且 qwait/p95<0.50` 的有效窗口；找到后可用同一 trace、同一到达
序列、同一 batch_size 跑四策略。

示例：
    python h1/run_find_workload.py
    python h1/run_find_workload.py --unique-fractions 0.25 0.50 0.75 --policies h1_lru
    python h1/run_find_workload.py --unique-fractions 0.50 --four-policies --reps 3

启动命令：
    python h1/备份-启动器/run_find_workload.py

参数说明：
    --source-trace：用于派生低复用 trace 的源 JSONL。
    --trace-out：派生 trace 输出目录。
    --visible-devices：传给每个 cell 的 CUDA_VISIBLE_DEVICES。
    --budget：固定使用的 gpu_memory_utilization 档位。
    --batch-size：固定使用的 replay_batch_size。
    --num-prompts：每个 cell 回放请求数。
    --unique-fractions：要注入唯一前缀的请求比例列表。
    --policies：每个派生 trace 下运行的策略列表。
    --reference-policy：用于判定有效窗口的参考策略。
    --reps：每个 unique_fraction 重复运行次数。
    --four-policies：忽略 --policies，改跑 vllm_default/h1_lru/h1_lfu/h1_lpe。
    --metadata-only：只改 reuse metadata，不改 prompt 文本；用于验证 metadata 不影响真实 hit。
    --force：已有 summary JSON 时仍重跑 cell。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import _runner as R
import run_step3_budget_tiers as step3
from run_find_interval import HIT_HI, HIT_LO, QWAIT_RATIO_MAX, _in_window, _read_cell_metrics
from run_find_load import _mnbt_for_bs


BASE_OUT = Path("h1/out/find_workload")
TRACE_OUT = Path("data/edgekv_traces/h1_workload_sweep")
SOURCE_TRACE = step3.REPLAY_TRACE
BUDGET = "0.735"
BATCH_SIZE = 8
NUM_PROMPTS = 256
REFERENCE_POLICY = "h1_lru"
POLICIES = ["h1_lru"]
UNIQUE_FRACTIONS = [0.25, 0.50, 0.75, 0.90]
FOUR_POLICIES = ["vllm_default", "h1_lru", "h1_lfu", "h1_lpe"]


def _fraction_label(value: float) -> str:
    return f"uf{int(round(value * 1000)):03d}"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _unique_budget(total: int, unique_fraction: float) -> int:
    return max(0, min(total, int(round(total * unique_fraction))))


def _prepend_unique_prompt_prefix(item: dict[str, Any], marker: str) -> bool:
    prefix = f"Unique request marker: {marker}\n"
    if isinstance(item.get("prompt"), str):
        item["prompt"] = prefix + item["prompt"]
        return True
    turns = item.get("turns")
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, dict) and isinstance(turn.get("user"), str):
                turn["user"] = prefix + turn["user"]
                return True
    return False


def derive_trace(source: Path, out_dir: Path, unique_fraction: float, max_requests: int,
                 rewrite_prompt_prefix: bool = True) -> Path:
    """通过均匀抽样行并唯一化，创建确定性的低复用 trace。"""
    rows = _load_jsonl(source)[:max_requests]
    if not rows:
        raise RuntimeError(f"source trace is empty: {source}")
    unique_count = _unique_budget(len(rows), unique_fraction)
    selected = set()
    if unique_count:
        for pos in range(unique_count):
            selected.add((pos * len(rows)) // unique_count)

    derived: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        if index in selected:
            suffix = f"unique:{_fraction_label(unique_fraction)}:{index:06d}"
            original_session = str(item.get("session_id", ""))
            original_reuse = str(item.get("reuse_key", original_session))
            item["session_id"] = f"{original_session}:{suffix}" if original_session else suffix
            item["reuse_key"] = f"{original_reuse}:{suffix}" if original_reuse else suffix
            item["unique_fraction"] = round(unique_fraction, 6)
            item["reuse_transform"] = "per_request_unique"
            if item.get("rag_reuse_key"):
                item["rag_reuse_key"] = f"{item['rag_reuse_key']}:{suffix}"
            item["prompt_prefix_transform"] = "per_request_unique_marker" if rewrite_prompt_prefix else "metadata_only"
            if rewrite_prompt_prefix:
                item["prompt_prefix_rewritten"] = _prepend_unique_prompt_prefix(item, suffix)
        else:
            item["unique_fraction"] = round(unique_fraction, 6)
            item["reuse_transform"] = "preserve_reuse_key"
            item["prompt_prefix_transform"] = "preserve_prompt"
        derived.append(item)

    out_path = out_dir / f"{Path(source).stem}_{_fraction_label(unique_fraction)}.jsonl"
    _write_jsonl(out_path, derived)
    return out_path


def _tier(unique_fraction: float, rep: int) -> str:
    base = _fraction_label(unique_fraction)
    return base if rep == 0 else f"{base}_rep{rep + 1}"


def _run_fraction(unique_fraction: float, trace_path: Path, policies: list[str], reps: int,
                  budget: str, batch_size: int, visible_devices: str, num_prompts: int,
                  force: bool) -> None:
    for rep in range(max(1, reps)):
        step3.run_step3(
            tier=_tier(unique_fraction, rep),
            base_out=BASE_OUT,
            budgets=[budget],
            policies=policies,
            num_prompts=num_prompts,
            visible_devices=visible_devices,
            replay_trace=trace_path,
            replay_batch_size=batch_size,
            max_num_batched_tokens=_mnbt_for_bs(batch_size),
            no_finalize=True,
            keep_cells=True,
            force=force,
        )


def _collect(base_out: Path, budget: str) -> dict[str, dict[str, dict[str, dict]]]:
    by_knob: dict[str, dict[str, dict[str, dict]]] = {}
    for summary_json in sorted(base_out.glob(f"uf*/{budget}/*/*_summary.json")):
        tier = summary_json.parent.parent.parent.name
        knob = tier.split("_rep", 1)[0]
        rep = "rep1"
        if "_rep" in tier:
            rep = f"rep{tier.rsplit('_rep', 1)[1]}"
        policy = summary_json.parent.name
        try:
            by_knob.setdefault(knob, {}).setdefault(rep, {})[policy] = _read_cell_metrics(summary_json)
        except Exception as exc:  # noqa: BLE001
            R.log(f"[find_workload] 读取失败 {summary_json}: {exc}")
    return by_knob


def _policy_gain_pct(base: dict, other: dict) -> float | None:
    base_p95 = float(base.get("p95_ttft_ms", 0.0) or 0.0)
    if base_p95 <= 0.0:
        return None
    return (base_p95 - float(other.get("p95_ttft_ms", 0.0) or 0.0)) / base_p95 * 100.0


def _report(by_knob: dict[str, dict[str, dict[str, dict]]], reference_policy: str,
            out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for knob in sorted(by_knob):
        for rep in sorted(by_knob[knob]):
            policies = by_knob[knob][rep]
            for policy in sorted(policies):
                m = policies[policy]
                rows.append({
                    "workload_knob": knob,
                    "rep": rep,
                    "policy": policy,
                    "status": "ok" if m.get("ok", True) and int(m.get("requests", 0) or 0) > 0 else "failed",
                    "requests": int(m.get("requests", 0) or 0),
                    "hit_rate": round(float(m.get("hit_rate", 0.0) or 0.0), 6),
                    "p95_ttft_ms": round(float(m.get("p95_ttft_ms", 0.0) or 0.0), 3),
                    "queue_wait_p95_ms": round(float(m.get("queue_wait_p95_ms", 0.0) or 0.0), 3),
                    "qwait_ratio": round(float(m.get("qwait_ratio", 0.0) or 0.0), 6),
                    "in_window": _in_window(m),
                    "error": str(m.get("error", "") or ""),
                })

    recommended: list[dict[str, Any]] = []
    gains: list[dict[str, Any]] = []
    for knob in sorted(by_knob):
        for rep in sorted(by_knob[knob]):
            policies = by_knob[knob][rep]
            ref = policies.get(reference_policy)
            if ref and _in_window(ref):
                recommended.append({
                    "workload_knob": knob,
                    "rep": rep,
                    "ref_hit_rate": round(float(ref["hit_rate"]), 4),
                    "ref_qwait_ratio": round(float(ref["qwait_ratio"]), 4),
                })
            lru = policies.get("h1_lru")
            if lru:
                for policy in ("vllm_default", "h1_lfu", "h1_lpe"):
                    other = policies.get(policy)
                    if other:
                        gain = _policy_gain_pct(lru, other)
                        gains.append({
                            "workload_knob": knob,
                            "rep": rep,
                            "policy": policy,
                            "p95_gain_pct_vs_lru": round(gain, 3) if gain is not None else "",
                            "hit_rate_delta_vs_lru": round(float(other["hit_rate"]) - float(lru["hit_rate"]), 6),
                            "qwait_ratio": round(float(other["qwait_ratio"]), 6),
                            "in_window_ref": bool(ref and _in_window(ref)),
                        })

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "workload_knob", "rep", "policy", "status", "requests", "hit_rate",
        "p95_ttft_ms", "queue_wait_p95_ms", "qwait_ratio", "in_window", "error",
    ]
    with (out_dir / "find_workload_report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "find_workload_policy_gains.csv").open("w", encoding="utf-8", newline="") as f:
        fields_gain = ["workload_knob", "rep", "policy", "p95_gain_pct_vs_lru",
                       "hit_rate_delta_vs_lru", "qwait_ratio", "in_window_ref"]
        writer = csv.DictWriter(f, fieldnames=fields_gain)
        writer.writeheader()
        writer.writerows(gains)
    (out_dir / "find_workload_report.json").write_text(
        json.dumps(
            {
                "window": {"hit_lo": HIT_LO, "hit_hi": HIT_HI, "qwait_ratio_max": QWAIT_RATIO_MAX},
                "reference_policy": reference_policy,
                "cells": rows,
                "recommended_workload_knobs": recommended,
                "policy_gains": gains,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== H1 workload 扫描结果（窗口: hit∈[{:.2f},{:.2f}] 且 qwait/p95<{:.0%}）==="
          .format(HIT_LO, HIT_HI, QWAIT_RATIO_MAX))
    print(f"{'knob':>8} {'rep':>6} {'policy':>14} {'status':>8} {'hit':>8} {'p95':>10} {'qwait/p95':>10} {'win':>6}")
    for row in rows:
        print(f"{row['workload_knob']:>8} {row['rep']:>6} {row['policy']:>14} {row['status']:>8} "
              f"{row['hit_rate']:>8.4f} {row['p95_ttft_ms']:>10.1f} {row['qwait_ratio']:>10.4f} "
              f"{str(row['in_window']):>6}")
    print("\n--- 推荐 workload knob（参考策略 {} 落入有效窗口）---".format(reference_policy))
    if recommended:
        for item in recommended:
            print(f"  {item['workload_knob']} {item['rep']} hit={item['ref_hit_rate']} qwait/p95={item['ref_qwait_ratio']}")
    else:
        print("  （无）hit 仍偏高→继续提高 unique_fraction；hit 偏低→降低 unique_fraction。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-trace", type=Path, default=SOURCE_TRACE)
    parser.add_argument("--trace-out", type=Path, default=TRACE_OUT)
    parser.add_argument("--visible-devices", default=step3.DEVICES)
    parser.add_argument("--budget", default=BUDGET)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--unique-fractions", nargs="+", type=float, default=UNIQUE_FRACTIONS)
    parser.add_argument("--policies", nargs="+", default=POLICIES)
    parser.add_argument("--reference-policy", default=REFERENCE_POLICY)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--four-policies", action="store_true",
                        help="Shortcut: run vllm_default/h1_lru/h1_lfu/h1_lpe.")
    parser.add_argument("--metadata-only", action="store_true",
                        help="只改 reuse metadata，不改 prompt 前缀；用于证明 metadata 不影响 GPU prefix hit。")
    parser.add_argument("--force", action="store_true", help="即使 summary 已存在也重跑")
    args = parser.parse_args()

    policies = FOUR_POLICIES if args.four_policies else args.policies
    R.log(f"[find_workload] budget={args.budget} bs={args.batch_size} fractions={args.unique_fractions} "
          f"policies={policies} reps={args.reps}")
    for fraction in args.unique_fractions:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"unique_fraction must be in [0,1]: {fraction}")
        trace_path = derive_trace(
            args.source_trace,
            args.trace_out,
            fraction,
            args.num_prompts,
            rewrite_prompt_prefix=not args.metadata_only,
        )
        R.log(f"[find_workload] fraction={fraction:.3f} trace={trace_path}")
        _run_fraction(fraction, trace_path, policies, args.reps, args.budget, args.batch_size,
                      args.visible_devices, args.num_prompts, args.force)

    by_knob = _collect(BASE_OUT, args.budget)
    _report(by_knob, args.reference_policy, BASE_OUT)


if __name__ == "__main__":
    main()
