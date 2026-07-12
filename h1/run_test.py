#!/usr/bin/env python3
# 测试数据契约：本仓库测试必须使用 JSONL replay trace 文件作为 workload 数据，不使用 vLLM 内置数据集/测试数据。
"""在默认三档预算上运行 replay-trace 策略对比。

运行一个 Step3 tier，其预算轴为三个数值型 gpu_memory_utilization：
0.75 / 0.825 / 0.95。每档预算默认运行四种缓存策略：LPE、LRU、LFU 和 vLLM
默认策略。低预算会更激进地驱逐已预热的 hot prefix；高预算保留更多内容，从而
提高前缀缓存命中率。max_model_len 保持为 2048，对齐三档 batch 测试口径。

默认目标是 ShareGPT trace；可传入 --replay-trace / --tier / --num-prompts 重定向
（例如退回到冻结的 HotQA ws2 trace）。

启动命令：
    python h1/run_test.py

参数说明：
    --visible-devices：传给每个 cell 的 CUDA_VISIBLE_DEVICES。
    --num-prompts：每个 cell 回放请求数。
    --tier：输出 tier 前缀；实际目录会追加 replay_batch_size。
    --replay-trace：JSONL replay trace 输入路径。
    --budgets：要扫描的 gpu_memory_utilization 档位，空格分隔。
    --policies：要运行的缓存策略，空格分隔。
    --replay-batch-size：回放批大小，对应 vLLM max_num_seqs 压力。
    --batch-order：批内请求排序方式；本脚本默认 round_robin。
    --max-model-len：vLLM max_model_len。
    --max-num-batched-tokens：vLLM max_num_batched_tokens。
    --workload：回放类型，sharegpt/rag/mixed。
    --rag-requests：mixed/rag workload 中 RAG 请求数。
    --hotpotqa-max-examples：加载 HotpotQA 的最大样本数。
    --force：已有 summary JSON 时仍重跑 cell。
    --out-dir：保留参数；当前实现固定写入 h1/out。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import run_step3_budget_tiers as step3

OUT_DIR = Path("h1/out")
# POLICIES = ["h1_lpe", "h1_lru", "h1_lfu", "vllm_default"]
POLICIES = ["h1_lru", "h1_lpe"]
# 数值型预算（gpu_memory_utilization）会在 resolve_budget 中通过 float() 解析；
# 有意不使用 tight/mid/loose 这些命名档。
# BUDGETS = ["0.75", "0.8", "0.85", "0.9"]
BUDGETS = ["0.9","0.85","0.8","0.75"]
# BUDGETS = ["0.75"]
REPLAY_TRACE = Path("data/edgekv_traces/source_ablation/sharegpt_256_original_order.jsonl")
# structured_conversation_v2 trace 默认包含 1536 个请求（匹配 config.json trace_size）。
NUM_PROMPTS = 1536
REPLAY_BATCH_SIZE = 8
TIER = "LPE_FIX"
MAX_MODEL_LEN = 2048
MAX_NUM_BATCHED_TOKENS = 8192
BATCH_ORDER = "round_robin"
WORKLOAD = "sharegpt"
RAG_REQUESTS = 0
HOTPOTQA_MAX_EXAMPLES = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-devices", default="1,2")
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--tier", default=TIER)
    parser.add_argument("--replay-trace", type=Path, default=REPLAY_TRACE)
    parser.add_argument("--budgets", default=" ".join(BUDGETS))
    parser.add_argument("--policies", default=" ".join(POLICIES))
    parser.add_argument("--replay-batch-size", type=int, default=REPLAY_BATCH_SIZE)
    parser.add_argument("--batch-order", choices=("round_robin",), default=BATCH_ORDER)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--max-num-batched-tokens", type=int, default=MAX_NUM_BATCHED_TOKENS)
    parser.add_argument("--workload", choices=("sharegpt", "rag", "mixed"), default=WORKLOAD)
    parser.add_argument("--rag-requests", type=int, default=RAG_REQUESTS)
    parser.add_argument("--hotpotqa-max-examples", type=int, default=HOTPOTQA_MAX_EXAMPLES)
    parser.add_argument("--force", action="store_true", help="rerun cells even if summary JSON exists")
    parser.add_argument("--out-dir", default="sharegpt_structured_v2")
    args = parser.parse_args()

    # 原始写法：base_out = OUT_DIR / args.out_dir
    base_out = OUT_DIR
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
        replay_batch_size=args.replay_batch_size,
        batch_order=args.batch_order,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        workload=args.workload,
        rag_requests=args.rag_requests,
        hotpotqa_max_examples=args.hotpotqa_max_examples,
    )


if __name__ == "__main__":
    main()
