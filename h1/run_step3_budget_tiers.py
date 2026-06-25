#!/usr/bin/env python3
"""Step3 budget-tier calibration / policy comparison (replaces run_step3_budget_tiers.sh).

One big task: on the workload fixed by Step2 (prefix_repetition, p8/pl512/s128,
output_len=1, num_prompts=200, rr=18.5), scan gpu_memory_utilization budgets and
compare policies (LRU / LFU / vLLM-default / LPE) by p95 TTFT. Each cell runs
through h1/run_h1_policy_serving_bench.sh.

`run_step3()` is importable and reused by run_step3_repeat.py (which calls it with
no_finalize=True so all reps are kept until it aggregates the median itself).

    python h1/run_step3_budget_tiers.py
    python h1/run_step3_budget_tiers.py --budgets "0.710 0.720" --policies "h1_lru h1_lpe"

All configuration lives in the CONFIG block below; no env vars are required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _runner as R

# ----------------------------------------------------------------------------- CONFIG
DEVICES = "0,1"
TIER = "tight"
BASE_OUT = Path("h1/out/step3")

# Budgets (== gpu_memory_utilization values) and policies to compare.
BUDGETS = ["0.710", "0.720", "0.730"]
POLICIES = ["h1_lru"]

# Fixed workload (Step2-selected operating point).
NUM_PROMPTS = 200
REQUEST_RATE = 18.5
NUM_PREFIXES = 8
PREFIX_LEN = 512
SUFFIX_LEN = 128
OUTPUT_LEN = 1

# Batching / concurrency (same as Step2 formal tier).
MAX_CONCURRENCY = 64
MAX_NUM_SEQS = 4
MAX_NUM_BATCHED_TOKENS = 4096
STATS_FLUSH_INTERVAL = 2048
PROFILE_POLICY_TIME = 0

# LPE knobs (same as Step2 defaults).
LPE_LIGHT_PATH = 1
LPE_REORDER_MODE = "window"
LPE_REORDER_WINDOW = 128
LPE_PRESSURE_FREE_RATIO = 0.15
LPE_PRESSURE_EVICTION_WINDOW = 64
# --------------------------------------------------------------------------------------


def cell_env(policy, budget, num_prompts, request_rate) -> dict:
    return {
        "H1_BENCH_DATASET": "prefix_repetition",
        "H1_GPU_POLICY": policy,
        "H1_BENCH_NUM_PROMPTS": num_prompts,
        "H1_BENCH_REQUEST_RATE": request_rate,
        "H1_PREFIX_REPETITION_NUM_PREFIXES": NUM_PREFIXES,
        "H1_PREFIX_REPETITION_PREFIX_LEN": PREFIX_LEN,
        "H1_PREFIX_REPETITION_SUFFIX_LEN": SUFFIX_LEN,
        "H1_PREFIX_REPETITION_OUTPUT_LEN": OUTPUT_LEN,
        "H1_GPU_MEMORY_UTILIZATION": budget,
        "H1_BENCH_MAX_CONCURRENCY": MAX_CONCURRENCY,
        "H1_VLLM_MAX_NUM_SEQS": MAX_NUM_SEQS,
        "H1_VLLM_MAX_NUM_BATCHED_TOKENS": MAX_NUM_BATCHED_TOKENS,
        "EDGEKV_H1_STATS_FLUSH_INTERVAL": STATS_FLUSH_INTERVAL,
        "EDGEKV_H1_PROFILE_POLICY_TIME": PROFILE_POLICY_TIME,
        "H1_LPE_LIGHT_PATH": LPE_LIGHT_PATH,
        "H1_LPE_REORDER_MODE": LPE_REORDER_MODE,
        "H1_LPE_REORDER_WINDOW": LPE_REORDER_WINDOW,
        "H1_LPE_PRESSURE_FREE_RATIO": LPE_PRESSURE_FREE_RATIO,
        "H1_LPE_PRESSURE_EVICTION_WINDOW": LPE_PRESSURE_EVICTION_WINDOW,
    }


def run_step3(*, tier=TIER, base_out=BASE_OUT, budgets=BUDGETS, policies=POLICIES,
              num_prompts=NUM_PROMPTS, request_rate=REQUEST_RATE,
              visible_devices=DEVICES, no_finalize=False, force=False,
              keep_cells=False) -> Path:
    """Run one tier's budget x policy matrix. Returns the tier directory.

    With no_finalize=True the summary/cleanup is deferred (used by the repeat
    protocol). Otherwise summarizes into <tier_dir>/step3_summary.csv and drops
    the per-cell outputs unless keep_cells.
    """
    base_out = Path(base_out)
    tier_dir = base_out / tier
    log_dir = tier_dir / "logs"
    if not R.DRY_RUN:
        log_dir.mkdir(parents=True, exist_ok=True)

    R.log(f"[step3] tier={tier} out={tier_dir} budgets={budgets} policies={policies}")
    R.log(f"[step3] workload: prompts={num_prompts} rr={request_rate} prefixes={NUM_PREFIXES} "
          f"prefix_len={PREFIX_LEN} suffix_len={SUFFIX_LEN} output_len={OUTPUT_LEN}")
    for budget in budgets:
        for policy in policies:
            out_dir = tier_dir / budget / policy
            R.log(f"[run] tier={tier} budget={budget} policy={policy} prompts={num_prompts} "
                  f"rr={request_rate} prefixes={NUM_PREFIXES} prefix_len={PREFIX_LEN} "
                  f"suffix_len={SUFFIX_LEN}")
            R.run_bench_cell(
                out_dir, visible_devices,
                cell_env(policy, budget, num_prompts, request_rate),
                log_file=log_dir / f"{budget}_{policy}.log", echo=False, force=force,
            )
    R.log(f"[step3] done tier={tier}")

    if no_finalize:
        R.log(f"[step3] finalize deferred (no_finalize=True) tier={tier}")
        return tier_dir

    summary_csv = tier_dir / "step3_summary.csv"
    R.log(f"[summary] writing {summary_csv}")
    R.summarize("summarize_step3_budget_tiers.py",
                ["--out", str(tier_dir), "--summary", str(summary_csv),
                 "--request-rate", str(request_rate)])
    R.cleanup_dirs(tier_dir, keep=keep_cells)
    R.log(f"[done] summary: {summary_csv}")
    return tier_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visible-devices", default=DEVICES)
    ap.add_argument("--tier", default=TIER)
    ap.add_argument("--budgets", default=" ".join(BUDGETS))
    ap.add_argument("--policies", default=" ".join(POLICIES))
    ap.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    ap.add_argument("--no-finalize", action="store_true", help="defer summary/cleanup (for repeat)")
    ap.add_argument("--force", action="store_true", help="rerun cells even if aggregate.csv exists")
    ap.add_argument("--keep-cells", action="store_true", help="retain per-cell outputs and logs")
    args = ap.parse_args()

    run_step3(
        tier=args.tier,
        budgets=args.budgets.split(),
        policies=args.policies.split(),
        num_prompts=args.num_prompts,
        visible_devices=args.visible_devices,
        no_finalize=args.no_finalize,
        force=args.force,
        keep_cells=args.keep_cells,
    )


if __name__ == "__main__":
    main()
