#!/usr/bin/env python3
"""Step3 on the real ShareGPT+HotpotQA mixed workload (real replay harness).

Mirrors run_step3_repeat.py on the real mixed pressure replay
(``h1/run_h1_vllm0110_real.py``): 4 policies x 3 GPU
memory budgets x N reps, cross-rep median, 4-panel visualization. The two real
datasets are frozen once into a replay trace (h0/build_h0_replay_trace.py) and reused
by every cell so the large ShareGPT JSON is parsed only once.

  budgets: tight/mid/loose -> gpu_memory_utilization 0.710/0.735/0.774
  policies: vllm_default / h1_lru / h1_lfu / h1_lpe

Each cell runs as one isolated conda process (single policy+budget) for clean GPU
memory between cells, like the per-cell isolation of the serving-bench path.

    python h1/run_step3_real.py --batch-sweep 8 16 32 64 --visible-devices 0,1
    python h1/run_step3_real.py --visible-devices 0,1 --replay-batch-size 32
    python h1/run_step3_real.py --reps 1 --budgets tight --policies h1_lru h1_lpe --max-requests 64
    EDGEKV_DRY_RUN=1 python h1/run_step3_real.py   # print commands only

All configuration lives in the CONFIG block below; no env vars are required.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import _runner as R

# ----------------------------------------------------------------------------- CONFIG
DEVICES = "0,1"  # real harness requires exactly two GPUs (tensor_parallel_size=2)
REPS = 3
POLICIES = ["vllm_default", "h1_lru", "h1_lfu", "h1_lpe"]
BUDGETS = ["tight", "mid", "loose"]
BASE = Path("h1/out/step3_real")
TRACE = Path("data/edgekv_traces/h0_sharegpt_hotpotqa_200sessions_pressure.jsonl")

# Repo-local real datasets (== run_h1_vllm0110_real.py defaults).
SHAREGPT_PATH = "data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
HOTPOTQA_PATH = "data/hotpotqa"

# Replay / workload knobs (aligned with h1/run_h1_vllm_real.sh).
WORKLOAD = "mixed"
SHAREGPT_ORDER = "longest"
MAX_SESSIONS = 200
MAX_REQUESTS = 1024
RAG_REQUESTS = 100
HOTPOTQA_MAX_EXAMPLES = 5
RAG_CHUNK_WORDS = 56
RAG_CHUNKS_PER_QUERY = 2
RAG_QUERY_REPEATS = 4
MAX_TOKENS = 16
MAX_MODEL_LEN = 2048
# Default to a conservative pressure point. Use --batch-sweep 8 16 32 64 to
# choose the final point from queue_wait_p95_ms / ttft_proxy_p95_ms and policy
# separation instead of making batch size 64 the main conclusion by default.
REPLAY_BATCH_SIZE = 32
DEFAULT_BATCH_SWEEP = [8, 16, 32, 64]
DEFAULT_BATCH_ORDER = "original"
DEFAULT_WARMUP_BATCHES = 0
TENSOR_PARALLEL_SIZE = 2

# run_h1_vllm0110_real.py imports edgekv_cop from the repo root, so "." must be on
# PYTHONPATH (run_real_cell hardcodes only "h1:h0"; env_overrides override it).
PYTHONPATH = ".:h1:h0"
# --------------------------------------------------------------------------------------


def build_trace(args: argparse.Namespace) -> None:
    """Freeze the mixed ShareGPT+HotpotQA replay trace once from the two JSON inputs."""
    trace_path = Path(args.replay_trace)
    if trace_path.exists() and not args.force:
        R.log(f"[trace] reuse existing {trace_path}")
        return
    cmd = [
        "conda", "run", "--no-capture-output", "-n", R.CONDA_ENV,
        "python", "h0/build_h0_replay_trace.py",
        "--trace-path", args.sharegpt_path,
        "--hotpotqa-path", args.hotpotqa_path,
        "--workload", WORKLOAD,
        "--sharegpt-order", SHAREGPT_ORDER,
        "--max-sessions", str(MAX_SESSIONS),
        "--max-requests", str(MAX_REQUESTS),
        "--rag-requests", str(RAG_REQUESTS),
        "--hotpotqa-max-examples", str(HOTPOTQA_MAX_EXAMPLES),
        "--rag-chunk-words", str(RAG_CHUNK_WORDS),
        "--rag-chunks-per-query", str(RAG_CHUNKS_PER_QUERY),
        "--rag-query-repeats", str(RAG_QUERY_REPEATS),
        "--out", str(trace_path),
    ]
    R.log(f"[trace] building {trace_path} from ShareGPT+HotpotQA")
    if R.DRY_RUN:
        R.log(f"[dry-run] cmd: {' '.join(cmd)}")
        return
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": PYTHONPATH}
    subprocess.run(cmd, cwd=R.ROOT, env=env, check=True)


def cell_args(cell_dir: Path, budget: str, policy: str, replay_batch_size: int,
              args: argparse.Namespace) -> list[str]:
    return [
        "--out", str(cell_dir),
        "--dtype", "float16",
        "--policies", policy,
        "--budgets", budget,
        "--workload", WORKLOAD,
        "--replay-trace", str(args.replay_trace),
        "--hotpotqa-path", args.hotpotqa_path,
        "--hotpotqa-max-examples", str(HOTPOTQA_MAX_EXAMPLES),
        "--max-sessions", str(MAX_SESSIONS),
        "--max-requests", str(args.max_requests),
        "--rag-requests", str(RAG_REQUESTS),
        "--rag-chunk-words", str(RAG_CHUNK_WORDS),
        "--rag-chunks-per-query", str(RAG_CHUNKS_PER_QUERY),
        "--rag-query-repeats", str(RAG_QUERY_REPEATS),
        "--sharegpt-order", SHAREGPT_ORDER,
        "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-tokens", str(MAX_TOKENS),
        "--replay-batch-size", str(replay_batch_size),
        "--batch-order", args.batch_order,
        "--warmup-batches", str(args.warmup_batches),
        "--visible-devices", args.visible_devices,
    ]


def summary_matches_config(summary_json: Path, replay_batch_size: int,
                           args: argparse.Namespace) -> bool:
    try:
        data = json.loads(summary_json.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        int(data.get("replay_batch_size", -1)) == replay_batch_size
        and str(data.get("batch_order", DEFAULT_BATCH_ORDER)) == args.batch_order
        and int(data.get("warmup_batches", DEFAULT_WARMUP_BATCHES)) == args.warmup_batches
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visible-devices", default=DEVICES)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--budgets", nargs="+", default=BUDGETS)
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--max-requests", type=int, default=MAX_REQUESTS)
    ap.add_argument("--replay-batch-size", type=int, default=REPLAY_BATCH_SIZE,
                    help="concurrency per cell (drives max_num_seqs / max_num_batched_tokens)")
    ap.add_argument("--batch-sweep", nargs="+", type=int, default=None,
                    help="并发扫描:固定单一 budget,对这些 replay_batch_size 各跑一轮 "
                         "(recommended: --batch-sweep 8 16 32 64),用 queue_wait_p95_ms "
                         "/ ttft_proxy_p95_ms 和策略差异选工作点")
    ap.add_argument("--recommended-batch-sweep", action="store_true",
                    help=f"use the recommended sweep sizes: {' '.join(map(str, DEFAULT_BATCH_SWEEP))}")
    ap.add_argument("--batch-order", choices=("original", "length_bucket"),
                    default=DEFAULT_BATCH_ORDER,
                    help="request order before batching; length_bucket groups similar prompt lengths")
    ap.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES,
                    help="run this many synthetic warmup batches before measured replay")
    ap.add_argument("--sharegpt-path", default=SHAREGPT_PATH)
    ap.add_argument("--hotpotqa-path", default=HOTPOTQA_PATH)
    ap.add_argument("--replay-trace", default=str(TRACE))
    ap.add_argument("--force", action="store_true", help="rerun cells (and rebuild trace) even if outputs exist")
    ap.add_argument("--keep-cells", action="store_true", help="retain per-rep cell outputs and logs")
    args = ap.parse_args()

    if not R.DRY_RUN:
        BASE.mkdir(parents=True, exist_ok=True)

    build_trace(args)

    if args.recommended_batch_sweep:
        args.batch_sweep = DEFAULT_BATCH_SWEEP

    # Default mode: full budget x policy matrix at a single concurrency. Sweep mode
    # (--batch-sweep): fix one budget (first of --budgets, default tight) and add the
    # batch size as the swept dimension so we can locate where the budget starts to bite.
    sweeping = bool(args.batch_sweep)
    batch_sizes = args.batch_sweep if sweeping else [args.replay_batch_size]
    budgets = [args.budgets[0]] if sweeping else args.budgets
    if sweeping:
        R.log(f"[mode] batch-sweep on budget={budgets[0]} sizes={batch_sizes} "
              f"(ttft_proxy_ms is a batch-latency proxy, grows with concurrency)")

    total = args.reps * len(budgets) * len(args.policies) * len(batch_sizes)
    idx = 0
    cell_suffix_parts = []
    if args.batch_order != DEFAULT_BATCH_ORDER:
        cell_suffix_parts.append(args.batch_order)
    if args.warmup_batches != DEFAULT_WARMUP_BATCHES:
        cell_suffix_parts.append(f"warm{args.warmup_batches}")
    cell_suffix = ("_" + "_".join(cell_suffix_parts)) if cell_suffix_parts else ""
    for rep in range(1, args.reps + 1):
        for budget in budgets:
            for policy in args.policies:
                for bs in batch_sizes:
                    idx += 1
                    cell_name = (
                        f"{budget}_{policy}_bs{bs}{cell_suffix}"
                        if sweeping else f"{budget}_{policy}{cell_suffix}"
                    )
                    cell_dir = BASE / f"rep{rep}" / cell_name
                    # harness writes {budget}_{policy}_summary.json (prefix is budget+policy only)
                    summary_json = cell_dir / f"{budget}_{policy}_summary.json"
                    R.log(f"[protocol {idx}/{total}] rep={rep} budget={budget} policy={policy} bs={bs}")
                    if (
                        summary_json.exists()
                        and not args.force
                        and summary_matches_config(summary_json, bs, args)
                    ):
                        R.log(f"[skip] {summary_json} already exists with matching config")
                        continue
                    if not R.DRY_RUN:
                        cell_dir.mkdir(parents=True, exist_ok=True)
                    rc = R.run_real_cell(
                        cell_dir, args.visible_devices,
                        cell_args(cell_dir, budget, policy, bs, args),
                        # Inject the GPU policy from the LAUNCH environment so every vLLM
                        # subprocess (EngineCore/Workers) sees it when sitecustomize is
                        # imported. build_llm's late os.environ assignment is racy and
                        # leaves some workers on vllm_default (lookup_total=0). Mirrors
                        # run_h1_policy_serving_bench.sh which exports EDGEKV_H1_GPU_POLICY.
                        {"PYTHONPATH": PYTHONPATH, "EDGEKV_H1_GPU_POLICY": policy},
                        log_file=cell_dir / "cell.log",
                    )
                    if rc != 0:
                        R.log(f"[warn] cell rc={rc} rep={rep} budget={budget} policy={policy} bs={bs} (continuing)")

    summary_csv = BASE / "step3_real_summary.csv"
    R.log(f"[summary] writing {summary_csv}")
    R.summarize("summarize_step3_real.py", ["--base", str(BASE), "--summary", str(summary_csv)])

    R.log("[visualize] rendering step3_real_scenarios.png/.pdf")
    R.summarize("visualize_step3_real.py", ["--summary", str(summary_csv)])

    if not args.keep_cells:
        R.cleanup_dirs(BASE, keep=False, only=[f"rep{r}" for r in range(1, args.reps + 1)])
    R.log(f"[done] summary: {summary_csv}")
    R.log("STEP3_REAL_DONE")


if __name__ == "__main__":
    main()
