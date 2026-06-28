#!/usr/bin/env python3
# TEST DATA CONTRACT: tests in this repository must use JSONL replay trace files as workload data. Do not use vLLM built-in datasets/test data.
"""H1 第二步「寻找有效区间」扫描器（H1_复盘与重跑配置单.md §4）。

把缓存预算 gpu_memory_utilization **往下扫**，回读每档命中率与排队占比，自动判定哪些
budget 落进 **有效窗口**（命中率 0.5~0.85 且 queue_wait/p95 < 50%，即既非死区也非饱和区），
并给出推荐区间。采用「先粗后细」两轮网格：

    # 默认两轮（粗扫 0.30/0.40/0.50/0.60 → 自动在夹住窗口的相邻档间细扫）
    python h1/run_find_interval.py

    # 只跑粗扫、指定档位
    python h1/run_find_interval.py --coarse-budgets 0.50 0.60 --no-fine

    # 连通性自检（不实跑）
    EDGEKV_DRY_RUN=1 python h1/run_find_interval.py --coarse-budgets 0.50 --no-fine

每档跑 h1_lru（参考基线，用于判窗口）+ h1_lpe。复用 run_step3_budget_tiers 的 cell 运行链；
按 budget 粒度 try/except 隔离失败，单档 OOM/拒服务只跳过该档、不中止整轮。
本脚本只扫「旋钮1=budget」，到达率/批大小（旋钮2）留给后续；qwait_ratio 会逐档报告供判定饱和。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _runner as R
import run_step3_budget_tiers as step3

# ----------------------------------------------------------------------------- CONFIG
BASE_OUT = Path("h1/out/find_interval")
TIER = "sweep"                                    # 两轮都写到同一 tier 下，按 budget 子目录区分
POLICIES = ["h1_lru", "h1_lpe"]                   # 搜索阶段每档跑这两个策略
REFERENCE_POLICY = "h1_lru"                       # 用它的 hit_rate 判定窗口（LRU≈vLLM 默认基线）
COARSE_BUDGETS = ["0.30", "0.40", "0.50", "0.60"]  # 第二步默认下扫档（gpu_memory_utilization）
HIT_LO, HIT_HI = 0.50, 0.85                       # 有效窗口命中率区间
QWAIT_RATIO_MAX = 0.50                            # queue_wait/p95 上限（>=50% 视为饱和）
FINE_STEPS = 3                                    # 细扫在夹住窗口的相邻档间插入的中间档数
MAX_NUM_BATCHED_TOKENS = 4096                     # 沿用 run_test，避免放大显存压力
# --------------------------------------------------------------------------------------


def _fmt_budget(value: float) -> str:
    """把数值 budget 格式化成稳定、干净的目录名字符串（0.525 / 0.55）。"""
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _read_cell_metrics(summary_json: Path) -> dict:
    """从一个 cell 的 *_summary.json 读回区间判定所需指标。"""
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


def _in_window(m: dict) -> bool:
    if not m.get("ok", True) or int(m.get("requests", 0) or 0) <= 0:
        return False
    return (HIT_LO <= m["hit_rate"] <= HIT_HI) and (m["qwait_ratio"] < QWAIT_RATIO_MAX)


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
        R.log(f"[find_interval] budget={budget} 跳过（疑似 OOM/拒服务）: {exc}")
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
            R.log(f"[find_interval] 读取失败 {summary_json}: {exc}")
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


def _pick_refine_span(by_budget: dict[str, dict[str, dict]],
                      reference_policy: str) -> tuple[str, str] | None:
    """找出参考策略 hit_rate 夹住有效窗口的最紧相邻档对 (b_hi, b_lo)。

    缓存随 budget 下降变小、命中率单调下降；从死区(97%)往下，先在 0.85 处进入窗口。
    主规则：取相邻档对，高档 hit>=0.85>低档 hit（夹住上沿）。
    退而求其次：取任一与窗口 [0.5,0.85] 有交叠的相邻档对。
    都没有（全在窗上方=缓存仍太大 / 全在下方=太小 / 数据不足）→ None。
    """
    present = [
        (b, pol[reference_policy]["hit_rate"])
        for b, pol in by_budget.items()
        if reference_policy in pol
        and pol[reference_policy].get("ok", True)
        and int(pol[reference_policy].get("requests", 0) or 0) > 0
    ]
    if len(present) < 2:
        return None
    present.sort(key=lambda bh: _budget_sort_key(bh[0]), reverse=True)  # 按 budget 数值降序

    pairs = list(zip(present[:-1], present[1:]))  # (高档, 低档)
    # 主规则：夹住上沿 0.85
    for (b_hi, h_hi), (b_lo, h_lo) in pairs:
        if h_hi >= HIT_HI > h_lo:
            return (b_hi, b_lo)
    # 退而求其次：与窗口区间有交叠
    for (b_hi, h_hi), (b_lo, h_lo) in pairs:
        seg_lo, seg_hi = min(h_hi, h_lo), max(h_hi, h_lo)
        if seg_hi >= HIT_LO and seg_lo <= HIT_HI:
            return (b_hi, b_lo)
    return None


def _fine_budgets(b_hi: str, b_lo: str, steps: int, existing: set[str]) -> list[str]:
    """在 (b_lo, b_hi) 之间等距插入 steps 个数值档，去重已跑过的档。"""
    hi, lo = _budget_sort_key(b_hi), _budget_sort_key(b_lo)
    if hi <= lo or steps < 1:
        return []
    span = hi - lo
    out: list[str] = []
    for i in range(1, steps + 1):
        b = _fmt_budget(lo + span * i / (steps + 1))
        if b not in existing and b not in out:
            out.append(b)
    return out


def _report(by_budget: dict[str, dict[str, dict]], reference_policy: str,
            out_dir: Path) -> None:
    """打印 + 落盘（csv/json）每个 (budget,policy) 的窗口判定与推荐区间。"""
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
                "in_window": _in_window(m),
                "error": str(m.get("error", "") or ""),
            }
            rows.append(row)
            if row["status"] == "failed":
                failed.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "find_interval_report.csv"
    json_path = out_dir / "find_interval_report.json"
    fields = ["budget", "policy", "status", "requests", "hit_rate", "p95_ttft_ms",
              "queue_wait_p95_ms", "qwait_ratio", "in_window", "error"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # 推荐区间：参考策略落入有效窗口的 budget；附该档 LPE vs LRU 的 p95 预判差
    recommended: list[dict] = []
    for b in budgets:
        ref = by_budget[b].get(reference_policy)
        if not ref or not _in_window(ref):
            continue
        entry = {"budget": b, "ref_hit_rate": round(ref["hit_rate"], 4),
                 "ref_qwait_ratio": round(ref["qwait_ratio"], 4)}
        lru = by_budget[b].get("h1_lru")
        lpe = by_budget[b].get("h1_lpe")
        if lru and lpe and lru["p95_ttft_ms"] > 0:
            entry["lpe_vs_lru_p95_gain_pct"] = round(
                (lru["p95_ttft_ms"] - lpe["p95_ttft_ms"]) / lru["p95_ttft_ms"] * 100.0, 3)
        recommended.append(entry)

    json_path.write_text(json.dumps(
        {"window": {"hit_lo": HIT_LO, "hit_hi": HIT_HI, "qwait_ratio_max": QWAIT_RATIO_MAX},
         "reference_policy": reference_policy,
         "cells": rows,
         "failed_cells": failed,
         "recommended_budgets": recommended},
        ensure_ascii=False, indent=2), encoding="utf-8")

    R.log(f"[find_interval] wrote {csv_path} / {json_path}")
    print("\n=== H1 区间扫描结果（窗口: hit∈[{:.2f},{:.2f}] 且 qwait/p95<{:.0%}）==="
          .format(HIT_LO, HIT_HI, QWAIT_RATIO_MAX))
    print(f"{'budget':>8} {'policy':>14} {'status':>8} {'req':>5} {'hit_rate':>9} "
          f"{'p95_ttft':>10} {'qwait/p95':>10} {'in_window':>10}")
    for r in rows:
        print(f"{r['budget']:>8} {r['policy']:>14} {r['status']:>8} {r['requests']:>5} "
              f"{r['hit_rate']:>9.4f} {r['p95_ttft_ms']:>10.1f} {r['qwait_ratio']:>10.4f} "
              f"{str(r['in_window']):>10}")
    if failed:
        print("\n--- 失败 cell（不参与窗口判定）---")
        for r in failed:
            err = (r["error"][:120] + "...") if len(r["error"]) > 120 else r["error"]
            print(f"  budget={r['budget']} policy={r['policy']} requests={r['requests']} error={err}")
    print("\n--- 推荐 budget 区间（参考策略 {} 落入有效窗口）---".format(reference_policy))
    if recommended:
        for e in recommended:
            extra = ("  LPE vs LRU p95: {:+.2f}%".format(e["lpe_vs_lru_p95_gain_pct"])
                     if "lpe_vs_lru_p95_gain_pct" in e else "")
            print(f"  budget={e['budget']}  hit={e['ref_hit_rate']}  "
                  f"qwait/p95={e['ref_qwait_ratio']}{extra}")
    else:
        print("  （无）粗扫未命中窗口；建议调整 --coarse-budgets 范围（命中率全偏高→再降；全偏低→抬高）")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visible-devices", default=step3.DEVICES)
    parser.add_argument("--num-prompts", type=int, default=step3.MAX_REQUESTS)
    parser.add_argument("--coarse-budgets", nargs="+", default=COARSE_BUDGETS,
                        help="粗扫档位（gpu_memory_utilization 数值或命名档 tight/mid/loose）")
    parser.add_argument("--policies", nargs="+", default=POLICIES)
    parser.add_argument("--reference-policy", default=REFERENCE_POLICY,
                        help="用于判定窗口的参考策略（命中率基线）")
    parser.add_argument("--fine-steps", type=int, default=FINE_STEPS)
    parser.add_argument("--no-fine", action="store_true", help="只跑粗扫，不做第二轮细扫")
    parser.add_argument("--max-num-batched-tokens", type=int, default=MAX_NUM_BATCHED_TOKENS)
    parser.add_argument("--force", action="store_true", help="即使 summary 已存在也重跑该 cell")
    args = parser.parse_args()

    tier_dir = BASE_OUT / TIER

    # ---- Round 1: 粗扫 ----
    R.log(f"[find_interval] Round1 coarse budgets={args.coarse_budgets} policies={args.policies}")
    for budget in args.coarse_budgets:
        _run_budget(budget, args.policies, args.visible_devices,
                    args.num_prompts, args.max_num_batched_tokens, args.force)
    by_budget = _collect(tier_dir)
    _report(by_budget, args.reference_policy, BASE_OUT)

    # ---- Round 2: 细扫（在夹住窗口的相邻粗扫档之间）----
    if not args.no_fine:
        span = _pick_refine_span(by_budget, args.reference_policy)
        if span is None:
            R.log("[find_interval] 粗扫未夹住有效窗口，跳过细扫（建议调整 --coarse-budgets 范围）")
        else:
            b_hi, b_lo = span
            existing = set(by_budget)
            fine = _fine_budgets(b_hi, b_lo, args.fine_steps, existing)
            R.log(f"[find_interval] Round2 fine 区间=({b_lo},{b_hi}) budgets={fine}")
            for budget in fine:
                _run_budget(budget, args.policies, args.visible_devices,
                            args.num_prompts, args.max_num_batched_tokens, args.force)
            by_budget = _collect(tier_dir)
            _report(by_budget, args.reference_policy, BASE_OUT)

    # ---- 可选：顺带产出熟悉的 step3_summary.csv（best-effort）----
    try:
        R.summarize("summarize_step3_budget_tiers.py",
                    ["--out", str(tier_dir),
                     "--summary", str(tier_dir / "step3_summary.csv"),
                     "--request-rate", "0.0"])
    except Exception as exc:  # noqa: BLE001
        R.log(f"[find_interval] step3_summary.csv 生成跳过: {exc}")


if __name__ == "__main__":
    main()
