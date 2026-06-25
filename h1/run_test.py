#!/usr/bin/env python3
"""Quick H1 smoke launcher: LRU/LPE once on tight and mid.

Runs two small Step3 serving-bench tiers without visualization:

  tight: gpu_memory_utilization=0.710
  mid:   gpu_memory_utilization=0.720

Each tier runs h1_lru and h1_lpe once, summarizes CSV outputs, and keeps the
cell directories so diagnostics such as metrics.txt and lpe_diagnostics.json
remain available.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import run_step3_budget_tiers as step3


BASE = Path("h1/out/run_test")
POLICIES = ["h1_lru", "h1_lpe"]
TIERS = [
    ("tight", "0.710"),
    ("mid", "0.720"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-devices", default=step3.DEVICES)
    parser.add_argument("--num-prompts", type=int, default=step3.NUM_PROMPTS)
    parser.add_argument("--force", action="store_true", help="rerun cells even if aggregate.csv exists")
    parser.add_argument("--base-out", default=str(BASE))
    args = parser.parse_args()

    base_out = Path(args.base_out)
    for tier, budget in TIERS:
        step3.run_step3(
            tier=tier,
            base_out=base_out,
            budgets=[budget],
            policies=POLICIES,
            num_prompts=args.num_prompts,
            visible_devices=args.visible_devices,
            force=args.force,
            keep_cells=True,
        )


if __name__ == "__main__":
    main()
