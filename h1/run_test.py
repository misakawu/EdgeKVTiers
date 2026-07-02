#!/usr/bin/env python3
# TEST DATA CONTRACT: tests in this repository must use JSONL replay trace files as workload data. Do not use vLLM built-in datasets/test data.
"""Replay-trace policy comparison over the default three budget tiers.

Runs one Step3 tier whose budget axis is the three numeric gpu_memory_utilization
values 0.75 / 0.825 / 0.95. Each budget runs all four cache policies by default:
LPE, LRU, LFU, and vLLM default. Low budgets evict primed hot prefixes more
aggressively; higher budgets retain more, raising the prefix-cache hit rate.
max_model_len stays at 1024 so the low budget can initialize on the local 2x11GiB
GPUs.

Defaults target the ShareGPT trace; pass --replay-trace / --tier / --num-prompts
to retarget (e.g. to fall back to the frozen HotQA ws2 trace).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import run_step3_budget_tiers as step3

OUT_DIR = Path("h1/out")
POLICIES = ["h1_lpe", "h1_lru", "h1_lfu", "vllm_default"]
TIER = "sharedgpt_v5"
# Numeric budgets (gpu_memory_utilization) resolved via float() in resolve_budget;
# intentionally NOT the named tight/mid/loose buckets.
# BUDGETS = ["0.75", "0.80", "0.85", "0.90", "0.95"]
BUDGETS = ["0.75", "0.825", "0.95"]
REPLAY_TRACE = Path("data/edgekv_traces/有效实验数据/sharedgpt_v5.jsonl")
# sharedgpt_v5 pressure trace holds 1536 requests (matches config.json trace_size).
NUM_PROMPTS = 1536
REPLAY_BATCH_SIZE = 8
MAX_MODEL_LEN = 1024
MAX_NUM_BATCHED_TOKENS = MAX_MODEL_LEN * REPLAY_BATCH_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-devices", default=step3.DEVICES)
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--tier", default=TIER)
    parser.add_argument("--replay-trace", type=Path, default=REPLAY_TRACE)
    parser.add_argument("--budgets", default=" ".join(BUDGETS))
    parser.add_argument("--policies", default=" ".join(POLICIES))
    parser.add_argument("--force", action="store_true", help="rerun cells even if summary JSON exists")
    parser.add_argument("--out-dir", default="LPE结果复现")
    args = parser.parse_args()

    base_out = OUT_DIR / args.out_dir
    step3.run_step3(
        tier=args.tier,
        base_out=base_out,
        budgets=args.budgets.split(),
        policies=args.policies.split(),
        num_prompts=args.num_prompts,
        visible_devices=args.visible_devices,
        force=args.force,
        keep_cells=True,
        replay_trace=args.replay_trace,
        replay_batch_size=REPLAY_BATCH_SIZE,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_model_len=MAX_MODEL_LEN,
    )


if __name__ == "__main__":
    main()
