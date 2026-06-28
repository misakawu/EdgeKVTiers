#!/usr/bin/env python3
# TEST DATA CONTRACT: tests in this repository must use JSONL replay trace files as workload data. Do not use vLLM built-in datasets/test data.
"""H1 Step B policy comparison: LRU/LPE across three numeric memory budgets.

Runs one Step3 tier whose budget axis is the three numeric gpu_memory_utilization
values 0.735 / 0.75 / 0.77 (tight / mid / loosen). Each budget runs h1_lru and
h1_lpe once on the frozen sharegpt_hotpotqa_session.jsonl trace, then summarizes
all six cells into a single CSV so the budget-vs-p95 main figure can be built.

Knobs are pinned to match the documented find_load run: replay_batch_size=16,
max_num_batched_tokens=4096, same arrival order. This is a 1-rep smoke to confirm
all six cells start and produce hit_rate before escalating to reps>=3.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import run_step3_budget_tiers as step3

OUT_DIR = Path("h1/out")
POLICIES = ["h1_lru", "h1_lpe"]
TIER = "budget_compare"
# Numeric budgets (gpu_memory_utilization) resolved via float() in resolve_budget;
# intentionally NOT the named tight/mid/loose buckets (which are 0.720/0.735/0.774).
BUDGETS = ["0.735", "0.75", "0.77"]  # tight / mid / loosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-devices", default=step3.DEVICES)
    parser.add_argument("--num-prompts", type=int, default=step3.MAX_REQUESTS)
    parser.add_argument("--force", action="store_true", help="rerun cells even if summary JSON exists")
    parser.add_argument("--out-dir", default="run_test")
    args = parser.parse_args()

    base_out = OUT_DIR / args.out_dir
    step3.run_step3(
        tier=TIER,
        base_out=base_out,
        budgets=BUDGETS,
        policies=POLICIES,
        num_prompts=args.num_prompts,
        visible_devices=args.visible_devices,
        force=args.force,
        keep_cells=True,
        max_num_batched_tokens=4096,
    )


if __name__ == "__main__":
    main()
