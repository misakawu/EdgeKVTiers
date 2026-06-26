#!/usr/bin/env python3
"""Step3 repeatable multi-measurement protocol on the pressure replay trace.

Each work point is a 4-policy comparison (h1_lru / h1_lfu / vllm_default / h1_lpe)
repeated N times so a median removes single-run variance. The cells reuse
run_step3() from run_step3_budget_tiers, which now drives the real H0 pressure
replay instead of vLLM's built-in benchmark datasets.

    python h1/run_step3_repeat.py
    python h1/run_step3_repeat.py --reps 5 --force

All configuration lives in the CONFIG block below; no env vars are required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _runner as R
import run_step3_budget_tiers as step3

# ----------------------------------------------------------------------------- CONFIG
DEVICES = "0,1"
REPS = 3
POLICIES = ["h1_lru", "h1_lfu", "vllm_default", "h1_lpe"]
BASE = Path("h1/out/step3_repeat")

# Work points: (label, budget tier, max replay requests) on the pressure trace.
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

    R.cleanup_dirs(BASE, keep=args.keep_cells)
    R.log(f"[done] summary: {summary_csv}")
    R.log("REPEAT_PROTOCOL_DONE")


if __name__ == "__main__":
    main()
