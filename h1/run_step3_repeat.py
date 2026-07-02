#!/usr/bin/env python3
"""在 pressure replay trace 上运行 Step3 可重复多次测量协议。

每个工作点都是 4 策略对比（h1_lru / h1_lfu / vllm_default / h1_lpe），重复 N 次
后用中位数削弱单次运行波动。cell 复用 run_step3_budget_tiers 中的 run_step3()，
该函数现在驱动真实 H0 pressure replay，而不是 vLLM 内置 benchmark 数据集。

    python h1/run_step3_repeat.py
    python h1/run_step3_repeat.py --reps 5 --force

所有配置都在下方 CONFIG 块内，无需环境变量。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _runner as R
import run_step3_budget_tiers as step3

# ----------------------------------------------------------------------------- 配置
DEVICES = "0,1"
REPS = 3
POLICIES = ["h1_lru", "h1_lfu", "vllm_default", "h1_lpe"]
BASE = Path("h1/out/step3_repeat")

# pressure trace 上的工作点：(标签, 预算档, 最大 replay 请求数)。
WORK_POINTS = [
    ("pressure_tight_n400", "tight", 400),
    ("pressure_mid_n400", "mid", 400),
    ("pressure_loose_n400", "loose", 400),
]
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visible-devices", default=DEVICES)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--force", action="store_true", help="rerun cells even if aggregate.csv exists")
    ap.add_argument("--keep-cells", action="store_true", help="retain per-rep outputs and logs")
    args = ap.parse_args()

    for label, budget, nprompts in WORK_POINTS:
        for rep in range(1, args.reps + 1):
            R.log(f"[protocol] point={label} budget={budget} n={nprompts} rep={rep}/{args.reps}")
            step3.run_step3(
                tier=label,
                base_out=BASE / label / f"rep{rep}",
                budgets=[budget],
                policies=POLICIES,
                num_prompts=nprompts,
                visible_devices=args.visible_devices,
                no_finalize=True,
                force=args.force,
                keep_cells=True,
            )

    summary_csv = BASE / "step3_repeat_summary.csv"
    R.log(f"[summary] writing {summary_csv}")
    R.summarize("summarize_step3_repeat.py",
                ["--base", str(BASE), "--summary", str(summary_csv)])

    R.log("[validate-d3] writing d3_validation.json for LPE cells")
    R.validate_d3_for_lpe_cells(BASE)

    R.cleanup_dirs(BASE, keep=args.keep_cells)
    R.log(f"[done] summary: {summary_csv}")
    R.log("REPEAT_PROTOCOL_DONE")


if __name__ == "__main__":
    main()
