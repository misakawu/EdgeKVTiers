#!/usr/bin/env python3
# TEST DATA CONTRACT: tests in this repository must use JSONL replay trace files as workload data. Do not use vLLM built-in datasets/test data.
"""HotQA ws2 xLRU quick budget sweep.

Runs one Step3 tier whose budget axis is the five numeric gpu_memory_utilization
values 0.75 / 0.80 / 0.85 / 0.90 / 0.95. Each budget runs h1_lru once on the
frozen HotQA ws2 replay trace. This is the quick 1-rep sweep used by
hotqa生成fix.md after lowering max_model_len to 1024 so the 0.75/0.80 budgets
can initialize on the local 2x11GiB GPUs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import run_step3_budget_tiers as step3

OUT_DIR = Path("h1/out")
POLICIES = ["h1_lru"]
TIER = "ws2_lru_budget5_maxlen1024"
# Numeric budgets (gpu_memory_utilization) resolved via float() in resolve_budget;
# intentionally NOT the named tight/mid/loose buckets.
BUDGETS = ["0.75", "0.80", "0.85", "0.90", "0.95"]
REPLAY_TRACE = Path("data/edgekv_traces/source_ablation/hotqa_ws2.jsonl")
REPLAY_BATCH_SIZE = 8
MAX_MODEL_LEN = 1024
MAX_NUM_BATCHED_TOKENS = MAX_MODEL_LEN * REPLAY_BATCH_SIZE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-devices", default=step3.DEVICES)
    parser.add_argument("--num-prompts", type=int, default=step3.MAX_REQUESTS)
    parser.add_argument("--force", action="store_true", help="rerun cells even if summary JSON exists")
    parser.add_argument("--out-dir", default="hotqa三级trace_有效窗口/run_test_ws2_lru_budget5_maxlen1024")
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
        replay_trace=REPLAY_TRACE,
        replay_batch_size=REPLAY_BATCH_SIZE,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_model_len=MAX_MODEL_LEN,
    )


if __name__ == "__main__":
    main()
