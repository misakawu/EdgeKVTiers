#!/usr/bin/env python3
# TEST DATA CONTRACT: tests in this repository must use JSONL replay trace files as workload data. Do not use vLLM built-in datasets/test data.
"""H1 第二步「定档」扫描器 v2 —— 在 0.77–0.90 区间用 LRU hit 粗网格最近点定三档。

背景（见 plan h1-trace-0-77hit0-5-budget-compare-step-resilient-snowflake）：新 trace
`data/edgekv_traces/sharegpt_hotpotqa_session.jsonl` 下 budget(=gpu_memory_utilization) 0.77
处 LRU hit≈0.5，往上扫到 0.90 才能覆盖 0.7/0.9。本脚本沿用 run_find_interval.py 的 cell 运行链，
**只做单轮粗扫**（删掉窗口/细扫逻辑），以 LRU 的 hit_rate 为基准，对 0.5/0.7/0.9 三个目标各取
「粗网格最近点」（|ref_hit − target| 最小的 budget）。

    # 默认粗扫 0.77/0.80/0.83/0.86/0.88/0.90，每档跑 h1_lru + h1_lpe
    python h1/run_find_interval_2.py

    # 指定档位
    python h1/run_find_interval_2.py --coarse-budgets 0.77 0.80 0.83 0.86 0.88 0.90

    # 连通性自检（不实跑）
    EDGEKV_DRY_RUN=1 python h1/run_find_interval_2.py --coarse-budgets 0.77 0.90

每档跑 h1_lru（参考基线，用于定档）+ h1_lpe。复用 run_step3_budget_tiers 的 cell 运行链；
按 budget 粒度 try/except 隔离失败，单档 OOM/拒服务只跳过该档、不中止整轮。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _runner as R
import run_step3_budget_tiers as step3

# ----------------------------------------------------------------------------- CONFIG
BASE_OUT = Path("h1/out/find_interval_2")
TIER = "sweep"                                    # 粗扫各档写到同一 tier 下，按 budget 子目录区分
POLICIES = ["h1_lru", "h1_lpe"]                   # 每档跑这两个策略
REFERENCE_POLICY = "h1_lru"                       # 用它的 hit_rate 定档（LRU≈vLLM 默认基线）
COARSE_BUDGETS = ["0.77", "0.80", "0.83", "0.86", "0.88", "0.90"]  # 0.77–0.90 粗网格（gpu_memory_utilization）
TARGET_HITS = [0.5, 0.7, 0.9]                     # 三个定档目标命中率
MAX_NUM_BATCHED_TOKENS = 4096                     # 沿用 run_test，避免放大显存压力
# --------------------------------------------------------------------------------------


def _fmt_budget(value: float) -> str:
    """把数值 budget 格式化成稳定、干净的目录名字符串（0.525 / 0.55）。"""
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _read_cell_metrics(summary_json: Path) -> dict:
    """从一个 cell 的 *_summary.json 读回定档所需指标。"""
    data = json.loads(summary_json.read_text(encoding="utf-8"))

    def fnum(key: str) -> float:
        return float(data.get(key, 0.0) or 0.0)

    p95 = fnum("ttft_proxy_p95_ms")
    qwait_p95 = fnum("queue_wait_p95_ms")
    ratio = data.get("queue_wait_p95_ratio")
    if ratio in (None, ""):
        ratio = (qwait_p95 / p95) if p95 > 0.0 else 0.0
    ok = bool(data.get("ok", True))
    requests = int(data.get("requests", 0) or 0)
    return {
        "ok": ok,
        "requests": requests,
        "error": str(data.get("error", "") or ""),
        "hit_rate": fnum("hit_rate"),
        "p95_ttft_ms": p95,
        "queue_wait_p95_ms": qwait_p95,
        "prefill_p95_ms": fnum("prefill_p95_ms"),
        "qwait_ratio": float(ratio or 0.0),
    }


def _run_budget(budget: str, policies: list[str], visible_devices: str,
                num_prompts: int, max_num_batched_tokens: int, force: bool) -> bool:
    """跑单个 budget（其下各策略）。失败（含 OOM 触发的 RuntimeError）只跳过该档。"""
    try:
        step3.run_step3(
            tier=TIER,
            base_out=BASE_OUT,
            budgets=[budget],
            policies=policies,
            num_prompts=num_prompts,
            visible_devices=visible_devices,
            max_num_batched_tokens=max_num_batched_tokens,
            no_finalize=True,      # 跳过 step3 自带 summarize/cleanup，保留 per-cell JSON
            keep_cells=True,
            force=force,
        )
        return True
    except Exception as exc:  # noqa: BLE001 — 单档失败需隔离，不能中止整轮
        R.log(f"[find_interval_2] budget={budget} 跳过（疑似 OOM/拒服务）: {exc}")
        return False


def _collect(tier_dir: Path) -> dict[str, dict[str, dict]]:
    """读回 tier 下所有已完成 cell：返回 by_budget[budget][policy] = metrics。"""
    by_budget: dict[str, dict[str, dict]] = {}
    for summary_json in sorted(tier_dir.glob("*/*/*_summary.json")):
        budget = summary_json.parent.parent.name
        policy = summary_json.parent.name
        try:
            by_budget.setdefault(budget, {})[policy] = _read_cell_metrics(summary_json)
        except Exception as exc:  # noqa: BLE001
            R.log(f"[find_interval_2] 读取失败 {summary_json}: {exc}")
    return by_budget


# 命名档→gpu_memory_utilization（与 run_h1_vllm0110_real.BUDGET_GPU_MEMORY_UTILIZATION 一致；
# 在此本地镜像一份，避免在编排进程里 import 该模块而触发 vllm 导入——vllm 只装在 cell 的
# conda 环境里，编排进程无法 import）。默认扫描用数值 budget，此表仅在传命名档时用于排序。
_NAMED_BUDGETS = {"super_tight": 0.710, "tight": 0.720, "mid": 0.735, "loose": 0.774}


def _budget_sort_key(budget: str) -> float:
    if budget in _NAMED_BUDGETS:
        return float(_NAMED_BUDGETS[budget])
    try:
        return float(budget)
    except ValueError:
        return 0.0


def _pick_tiers(by_budget: dict[str, dict[str, dict]], reference_policy: str,
                target_hits: list[float]) -> list[dict]:
    """对每个 target，在所有「ok 且 requests>0」的 budget 里取 |ref_hit − target| 最小者。

    返回每档 {target, budget, ref_hit, abs_delta, lpe_hit, lpe_vs_lru_p95_gain_pct}。
    无可用 budget 时返回的档 budget=None（如实标注数据不足）。
    """
    present = [
        (b, pol[reference_policy]["hit_rate"])
        for b, pol in by_budget.items()
        if reference_policy in pol
        and pol[reference_policy].get("ok", True)
        and int(pol[reference_policy].get("requests", 0) or 0) > 0
    ]
    tiers: list[dict] = []
    for target in target_hits:
        if not present:
            tiers.append({"target": target, "budget": None, "ref_hit": None,
                          "abs_delta": None, "lpe_hit": None,
                          "lpe_vs_lru_p95_gain_pct": None})
            continue
        b_best, h_best = min(present, key=lambda bh: abs(bh[1] - target))
        entry = {
            "target": target,
            "budget": b_best,
            "ref_hit": round(h_best, 6),
            "abs_delta": round(abs(h_best - target), 6),
            "lpe_hit": None,
            "lpe_vs_lru_p95_gain_pct": None,
        }
        lpe = by_budget.get(b_best, {}).get("h1_lpe")
        lru = by_budget.get(b_best, {}).get("h1_lru")
        if lpe:
            entry["lpe_hit"] = round(lpe["hit_rate"], 6)
        if lru and lpe and lru["p95_ttft_ms"] > 0:
            entry["lpe_vs_lru_p95_gain_pct"] = round(
                (lru["p95_ttft_ms"] - lpe["p95_ttft_ms"]) / lru["p95_ttft_ms"] * 100.0, 3)
        tiers.append(entry)
    return tiers


def _report(by_budget: dict[str, dict[str, dict]], reference_policy: str,
            target_hits: list[float], out_dir: Path) -> None:
    """打印 + 落盘（csv/json）每个 (budget,policy) 的指标，并报出三档最近点结果。"""
    budgets = sorted(by_budget, key=_budget_sort_key, reverse=True)
    rows: list[dict] = []
    failed: list[dict] = []
    for b in budgets:
        for pol in sorted(by_budget[b]):
            m = by_budget[b][pol]
            row = {
                "budget": b,
                "policy": pol,
                "status": "ok" if m.get("ok", True) and int(m.get("requests", 0) or 0) > 0 else "failed",
                "requests": int(m.get("requests", 0) or 0),
                "hit_rate": round(m["hit_rate"], 6),
                "p95_ttft_ms": round(m["p95_ttft_ms"], 3),
                "queue_wait_p95_ms": round(m["queue_wait_p95_ms"], 3),
                "qwait_ratio": round(m["qwait_ratio"], 6),
                "error": str(m.get("error", "") or ""),
            }
            rows.append(row)
            if row["status"] == "failed":
                failed.append(row)

    tiers = _pick_tiers(by_budget, reference_policy, target_hits)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "find_interval_2_report.csv"
    json_path = out_dir / "find_interval_2_report.json"
    fields = ["budget", "policy", "status", "requests", "hit_rate", "p95_ttft_ms",
              "queue_wait_p95_ms", "qwait_ratio", "error"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(
        {"reference_policy": reference_policy,
         "target_hits": target_hits,
         "coarse_budgets": budgets,
         "cells": rows,
         "failed_cells": failed,
         "tiers": tiers},
        ensure_ascii=False, indent=2), encoding="utf-8")

    R.log(f"[find_interval_2] wrote {csv_path} / {json_path}")
    print("\n=== H1 定档粗扫结果（参考策略 {}，目标 hit={}）==="
          .format(reference_policy, target_hits))
    print(f"{'budget':>8} {'policy':>14} {'status':>8} {'req':>5} {'hit_rate':>9} "
          f"{'p95_ttft':>10} {'qwait/p95':>10}")
    for r in rows:
        print(f"{r['budget']:>8} {r['policy']:>14} {r['status']:>8} {r['requests']:>5} "
              f"{r['hit_rate']:>9.4f} {r['p95_ttft_ms']:>10.1f} {r['qwait_ratio']:>10.4f}")
    if failed:
        print("\n--- 失败 cell（不参与定档）---")
        for r in failed:
            err = (r["error"][:120] + "...") if len(r["error"]) > 120 else r["error"]
            print(f"  budget={r['budget']} policy={r['policy']} requests={r['requests']} error={err}")

    print("\n--- 三档结果（target → 粗网格最近点）---")
    print(f"{'target':>7} {'budget':>8} {'LRU_hit':>9} {'LPE_hit':>9} {'|Δ|':>8} {'LPE_vs_LRU_p95':>15}")
    for t in tiers:
        if t["budget"] is None:
            print(f"{t['target']:>7.2f} {'(无)':>8} {'-':>9} {'-':>9} {'-':>8} {'-':>15}")
            continue
        lpe_hit = "-" if t["lpe_hit"] is None else f"{t['lpe_hit']:.4f}"
        gain = "-" if t["lpe_vs_lru_p95_gain_pct"] is None else f"{t['lpe_vs_lru_p95_gain_pct']:+.2f}%"
        print(f"{t['target']:>7.2f} {t['budget']:>8} {t['ref_hit']:>9.4f} {lpe_hit:>9} "
              f"{t['abs_delta']:>8.4f} {gain:>15}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visible-devices", default=step3.DEVICES)
    parser.add_argument("--num-prompts", type=int, default=step3.MAX_REQUESTS)
    parser.add_argument("--coarse-budgets", nargs="+", default=COARSE_BUDGETS,
                        help="粗扫档位（gpu_memory_utilization 数值或命名档 tight/mid/loose）")
    parser.add_argument("--policies", nargs="+", default=POLICIES)
    parser.add_argument("--reference-policy", default=REFERENCE_POLICY,
                        help="用于定档的参考策略（命中率基线）")
    parser.add_argument("--target-hits", nargs="+", type=float, default=TARGET_HITS,
                        help="定档目标命中率（默认 0.5 0.7 0.9）")
    parser.add_argument("--max-num-batched-tokens", type=int, default=MAX_NUM_BATCHED_TOKENS)
    parser.add_argument("--force", action="store_true", help="即使 summary 已存在也重跑该 cell")
    args = parser.parse_args()

    tier_dir = BASE_OUT / TIER

    # ---- 单轮粗扫（0.77–0.90）----
    R.log(f"[find_interval_2] coarse budgets={args.coarse_budgets} policies={args.policies}")
    for budget in args.coarse_budgets:
        _run_budget(budget, args.policies, args.visible_devices,
                    args.num_prompts, args.max_num_batched_tokens, args.force)
    by_budget = _collect(tier_dir)
    _report(by_budget, args.reference_policy, args.target_hits, BASE_OUT)

    # ---- 可选：顺带产出熟悉的 step3_summary.csv（best-effort）----
    try:
        R.summarize("summarize_step3_budget_tiers.py",
                    ["--out", str(tier_dir),
                     "--summary", str(tier_dir / "step3_summary.csv"),
                     "--request-rate", "0.0"])
    except Exception as exc:  # noqa: BLE001
        R.log(f"[find_interval_2] step3_summary.csv 生成跳过: {exc}")


if __name__ == "__main__":
    main()
