#!/usr/bin/env python3
# 测试数据契约：本仓库测试必须使用 JSONL replay trace 文件作为 workload 数据，不使用 vLLM 内置数据集/测试数据。
"""在默认三档预算上运行 replay-trace 策略对比。

运行一个 Step3 tier，其预算轴为三个数值型 gpu_memory_utilization：
0.75 / 0.825 / 0.95。每档预算默认运行四种缓存策略：LPE、LRU、LFU 和 vLLM
默认策略。低预算会更激进地驱逐已预热的 hot prefix；高预算保留更多内容，从而
提高前缀缓存命中率。max_model_len 保持为 1024，使低预算能在本地 2x11GiB
GPU 上初始化。

默认目标是 ShareGPT trace；可传入 --replay-trace / --tier / --num-prompts 重定向
（例如退回到冻结的 HotQA ws2 trace）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import run_step3_budget_tiers as step3

OUT_DIR = Path("h1/out")
POLICIES = ["h1_lpe", "h1_lru", "h1_lfu", "vllm_default"]
TIER = "sharedgpt_v5"
# 数值型预算（gpu_memory_utilization）会在 resolve_budget 中通过 float() 解析；
# 有意不使用 tight/mid/loose 这些命名档。
# BUDGETS = ["0.75", "0.80", "0.85", "0.90", "0.95"]
BUDGETS = ["0.75", "0.825", "0.95"]
REPLAY_TRACE = Path("data/edgekv_traces/有效实验数据/sharedgpt_v5.jsonl")
# sharedgpt_v5 pressure trace 包含 1536 个请求（匹配 config.json trace_size）。
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
