#!/usr/bin/env python3
# TEST DATA CONTRACT: tests in this repository must use JSONL replay trace files as workload data. Do not use vLLM built-in datasets/test data.
"""Quick H1 smoke launcher: LRU/LPE once on tight and mid pressure replay.

Runs two small Step3 pressure-replay tiers. Each tier runs h1_lru and h1_lpe
once, summarizes CSV outputs, and keeps the cell directories for diagnostics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import run_step3_budget_tiers as step3


BASE = Path("h1/out/run_test")
POLICIES = ["h1_lru", "h1_lpe"]
TIERS = [
    ("tight", "tight"),
    ("mid", "mid"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-devices", default=step3.DEVICES)
    parser.add_argument("--num-prompts", type=int, default=step3.MAX_REQUESTS)
    parser.add_argument("--force", action="store_true", help="rerun cells even if summary JSON exists")
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
