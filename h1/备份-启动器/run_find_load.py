#!/usr/bin/env python3
# TEST DATA CONTRACT: tests in this repository must use JSONL replay trace files as workload data. Do not use vLLM built-in datasets/test data.
"""H1 第二步「旋钮2：去饱和」负载扫描器（接续 run_find_interval 的发现）。

背景：在 3× RTX 2080 Ti（11GB）上，Qwen2.5-7B + TP=2 的低 budget 档无法启动，budget
（旋钮1）没有向下扫的空间；最新实测显示，能启动的 budget≈0.735 处命中率仍约 0.95，
高于窗口 [0.5,0.85]。本脚本只固定 budget 扫并发旋钮 `replay_batch_size`
（= vLLM max_num_seqs），用于把 qwait/p95 压到 < 0.50；若命中率仍偏高，说明当前 workload
是高复用死区，需要改 trace/workload，而不是继续扫 batch size。

    # 默认：budget=0.735，扫 replay_batch_size 16/8/4/2
    python h1/run_find_load.py

    # 自定义档位 / 负载
    python h1/run_find_load.py --budget 0.735 --batch-sizes 16 8 4 --num-prompts 256

    # 连通性自检（不实跑）
    EDGEKV_DRY_RUN=1 python h1/run_find_load.py --batch-sizes 8 --num-prompts 64

每档跑 h1_lru（参考基线，判窗口）+ h1_lpe。复用 run_step3_budget_tiers 的 cell 运行链，
用 tier=bs<N> 隔离各并发档目录（同 budget+policy 不会互相覆盖）。按档 try/except 隔离失败。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _runner as R
import run_step3_budget_tiers as step3
from run_find_interval import (
    HIT_HI,
    HIT_LO,
    QWAIT_RATIO_MAX,
    _in_window,
    _read_cell_metrics,
)

# ----------------------------------------------------------------------------- CONFIG
BASE_OUT = Path("h1/out/find_load")
BUDGET = "0.735"                          # 固定在引擎可启动的档；最新实测 hit 仍偏高，需要后续调 workload
POLICIES = ["h1_lru", "h1_lpe"]          # 每档跑这两个；lru 为窗口判定参考
REFERENCE_POLICY = "h1_lru"
BATCH_SIZES = [16, 8, 4, 2]              # 并发旋钮 = vLLM max_num_seqs，从默认 16 往下去饱和
NUM_PROMPTS = 256                        # 负载量；够大让 p95 有意义，又不至于太慢
MAX_NUM_BATCHED_TOKENS_CAP = 4096       # 上限；实际取 min(cap, max_model_len*bs)
# --------------------------------------------------------------------------------------


def _mnbt_for_bs(bs: int) -> int:
    """cell 约束 max_num_batched_tokens <= max_model_len*bs（且 >= bs）。按档取安全值。"""
    return max(bs, min(MAX_NUM_BATCHED_TOKENS_CAP, step3.MAX_MODEL_LEN * bs))


def _tier_for_bs(bs: int) -> str:
    return f"bs{bs}"


def _run_bs(bs: int, budget: str, policies: list[str], visible_devices: str,
            num_prompts: int, force: bool) -> bool:
    """跑单个并发档（其下各策略）。失败（含 OOM/拒服务）只跳过该档，不中止整轮。"""
    try:
        step3.run_step3(
            tier=_tier_for_bs(bs),
            base_out=BASE_OUT,
            budgets=[budget],
            policies=policies,
            num_prompts=num_prompts,
            visible_devices=visible_devices,
            replay_batch_size=bs,
            max_num_batched_tokens=_mnbt_for_bs(bs),
            no_finalize=True,
            keep_cells=True,
            force=force,
        )
        return True
    except Exception as exc:  # noqa: BLE001 — 单档失败需隔离
        R.log(f"[find_load] bs={bs} 跳过（疑似 OOM/拒服务）: {exc}")
        return False


def _collect(base_out: Path, budget: str) -> dict[int, dict[str, dict]]:
    """读回所有已完成 cell：by_bs[batch_size][policy] = metrics。"""
    by_bs: dict[int, dict[str, dict]] = {}
    for summary_json in sorted(base_out.glob(f"bs*/{budget}/*/*_summary.json")):
        tier = summary_json.parent.parent.parent.name  # bs<N>
        policy = summary_json.parent.name
        try:
            bs = int(tier[2:])
        except ValueError:
            continue
        try:
            by_bs.setdefault(bs, {})[policy] = _read_cell_metrics(summary_json)
        except Exception as exc:  # noqa: BLE001
            R.log(f"[find_load] 读取失败 {summary_json}: {exc}")
    return by_bs


def _report(by_bs: dict[int, dict[str, dict]], budget: str, reference_policy: str,
            out_dir: Path) -> None:
    """打印 + 落盘（csv/json）每个 (batch_size,policy) 的窗口判定与推荐工作点。"""
    sizes = sorted(by_bs, reverse=True)  # 并发从大到小（饱和→去饱和）
    rows: list[dict] = []
    for bs in sizes:
        for pol in sorted(by_bs[bs]):
            m = by_bs[bs][pol]
            rows.append({
                "batch_size": bs,
                "policy": pol,
                "hit_rate": round(m["hit_rate"], 6),
                "p95_ttft_ms": round(m["p95_ttft_ms"], 3),
                "queue_wait_p95_ms": round(m["queue_wait_p95_ms"], 3),
                "qwait_ratio": round(m["qwait_ratio"], 6),
                "in_window": _in_window(m),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "find_load_report.csv"
    json_path = out_dir / "find_load_report.json"
    fields = ["batch_size", "policy", "hit_rate", "p95_ttft_ms",
              "queue_wait_p95_ms", "qwait_ratio", "in_window"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # 推荐工作点：参考策略落入有效窗口的并发档；附该档 LPE vs LRU 的 p95 预判差
    recommended: list[dict] = []
    for bs in sizes:
        ref = by_bs[bs].get(reference_policy)
        if not ref or not _in_window(ref):
            continue
        entry = {"batch_size": bs, "ref_hit_rate": round(ref["hit_rate"], 4),
                 "ref_qwait_ratio": round(ref["qwait_ratio"], 4)}
        lru = by_bs[bs].get("h1_lru")
        lpe = by_bs[bs].get("h1_lpe")
        if lru and lpe and lru["p95_ttft_ms"] > 0:
            entry["lpe_vs_lru_p95_gain_pct"] = round(
                (lru["p95_ttft_ms"] - lpe["p95_ttft_ms"]) / lru["p95_ttft_ms"] * 100.0, 3)
        recommended.append(entry)

    json_path.write_text(json.dumps(
        {"budget": budget,
         "window": {"hit_lo": HIT_LO, "hit_hi": HIT_HI, "qwait_ratio_max": QWAIT_RATIO_MAX},
         "reference_policy": reference_policy,
         "cells": rows,
         "recommended_batch_sizes": recommended},
        ensure_ascii=False, indent=2), encoding="utf-8")

    R.log(f"[find_load] wrote {csv_path} / {json_path}")
    print("\n=== H1 负载扫描结果（budget={} 固定；窗口: hit∈[{:.2f},{:.2f}] 且 qwait/p95<{:.0%}）==="
          .format(budget, HIT_LO, HIT_HI, QWAIT_RATIO_MAX))
    print(f"{'batch_sz':>9} {'policy':>14} {'hit_rate':>9} {'p95_ttft':>10} "
          f"{'qwait/p95':>10} {'in_window':>10}")
    for r in rows:
        print(f"{r['batch_size']:>9} {r['policy']:>14} {r['hit_rate']:>9.4f} "
              f"{r['p95_ttft_ms']:>10.1f} {r['qwait_ratio']:>10.4f} "
              f"{str(r['in_window']):>10}")
    print("\n--- 推荐工作点（参考策略 {} 落入有效窗口的并发档）---".format(reference_policy))
    if recommended:
        for e in recommended:
            extra = ("  LPE vs LRU p95: {:+.2f}%".format(e["lpe_vs_lru_p95_gain_pct"])
                     if "lpe_vs_lru_p95_gain_pct" in e else "")
            print(f"  batch_size={e['batch_size']}  hit={e['ref_hit_rate']}  "
                  f"qwait/p95={e['ref_qwait_ratio']}{extra}")
    else:
        print("  （无）未命中窗口；qwait 仍偏高→再降 batch_size；hit 偏高(死区)→需调 trace/workload；"
              "hit 偏低→抬 batch_size 或调 trace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visible-devices", default=step3.DEVICES)
    parser.add_argument("--budget", default=BUDGET, help="固定的 gpu_memory_utilization 档")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES,
                        help="要扫的 replay_batch_size（=vLLM max_num_seqs 并发）")
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--policies", nargs="+", default=POLICIES)
    parser.add_argument("--reference-policy", default=REFERENCE_POLICY)
    parser.add_argument("--force", action="store_true", help="即使 summary 已存在也重跑")
    args = parser.parse_args()

    R.log(f"[find_load] budget={args.budget} batch_sizes={args.batch_sizes} "
          f"policies={args.policies} num_prompts={args.num_prompts}")
    for bs in args.batch_sizes:
        _run_bs(bs, args.budget, args.policies, args.visible_devices,
                args.num_prompts, args.force)

    by_bs = _collect(BASE_OUT, args.budget)
    _report(by_bs, args.budget, args.reference_policy, BASE_OUT)


if __name__ == "__main__":
    main()
