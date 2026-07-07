#!/usr/bin/env python3
"""在真实 ShareGPT+HotpotQA 混合工作负载上运行第三步（真实回放执行器）。

在真实混合压力回放（``h1/run_h1_vllm0110_real.py``）上对齐
run_step3_repeat.py：4 种策略 x 3 档 GPU 显存预算 x N 个重复轮次，跨轮次取中位数，
并生成 4 面板可视化。两个真实数据集会先冻结成一条回放轨迹
（h0/build_h0_replay_trace.py），供所有实验单元复用，避免反复解析大型 ShareGPT JSON。

  budgets: tight/mid/loose -> gpu_memory_utilization 0.710/0.735/0.774
  policies: vllm_default / h1_lru / h1_lfu / h1_lpe

每个实验单元都作为隔离 conda 进程运行（单策略 + 单预算），从而在实验单元之间清理 GPU
显存状态，类似服务压测路径的逐单元隔离。

    python h1/run_step3_real.py --batch-sweep 8 16 32 64 --visible-devices 0,1
    python h1/run_step3_real.py --visible-devices 0,1 --replay-batch-size 32
    python h1/run_step3_real.py --reps 1 --budgets tight --policies h1_lru h1_lpe --max-requests 64
    EDGEKV_DRY_RUN=1 python h1/run_step3_real.py   # 只打印命令

所有配置都在下方 CONFIG 块内，无需环境变量。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import _runner as R

# ----------------------------------------------------------------------------- 配置
DEVICES = "0,1"  # 真实回放执行器需要正好两张 GPU（tensor_parallel_size=2）
REPS = 3
POLICIES = ["vllm_default", "h1_lru", "h1_lfu", "h1_lpe"]
BUDGETS = ["tight", "mid", "loose"]
BASE = Path("h1/out/step3_real")
TRACE = Path("data/edgekv_traces/sharegpt_hotpotqa_session.jsonl")

# 仓库内真实数据集路径（等于 run_h1_vllm0110_real.py 默认值）。
SHAREGPT_PATH = "data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
HOTPOTQA_PATH = "data/hotpotqa"

# 回放/工作负载开关（与 h1/run_h1_vllm_real.sh 对齐）。
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
MAX_NUM_BATCHED_TOKENS = 8192
# 默认使用保守压力点。可用 --batch-sweep 8 16 32 64 根据 queue_wait_p95_ms /
# ttft_proxy_p95_ms 和策略区分度选择最终工作点，避免默认把批大小 64
# 作为主要结论。
REPLAY_BATCH_SIZE = 32
DEFAULT_BATCH_SWEEP = [8, 16, 32, 64]
DEFAULT_BATCH_ORDER = "original"
DEFAULT_WARMUP_BATCHES = 0
TENSOR_PARALLEL_SIZE = 2

# run_h1_vllm0110_real.py 会从仓库根目录导入 edgekv_cop，因此 PYTHONPATH 必须
# 包含 "."（run_real_cell 默认只写死 "h1:h0"；这里通过 env_overrides 覆盖）。
PYTHONPATH = ".:h1:h0"
# --------------------------------------------------------------------------------------


def build_trace(args: argparse.Namespace) -> None:
    """从两个 JSON 输入一次性冻结混合 ShareGPT+HotpotQA replay trace。"""
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
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
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
        and int(data.get("max_num_batched_tokens", -1)) == args.max_num_batched_tokens
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
    ap.add_argument("--max-num-batched-tokens", type=int, default=MAX_NUM_BATCHED_TOKENS,
                    help="vLLM scheduler token cap; keep fixed while sweeping replay batch size")
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

    # 默认模式：在单一并发下运行完整 budget x policy 矩阵。扫描模式
    # (--batch-sweep)：固定一个 budget（--budgets 的第一个值，默认 tight），并把
    # 批大小作为扫描维度，以定位预算压力开始显现的位置。
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
                    # 执行器写出 {budget}_{policy}_summary.json（前缀只包含 budget+policy）
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
                        # 从启动环境注入 GPU 策略，确保每个 vLLM 子进程
                        # （EngineCore/Workers）导入 sitecustomize 时都能看到。build_llm
                        # 中较晚的 os.environ 赋值存在竞态，可能让部分 worker 留在
                        # vllm_default（lookup_total=0）。这里对齐导出
                        # EDGEKV_H1_GPU_POLICY 的 run_h1_policy_serving_bench.sh。
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

    R.log("[validate-d3] writing d3_validation.json for LPE cells")
    R.validate_d3_for_lpe_cells(BASE)

    if not args.keep_cells:
        R.cleanup_dirs(BASE, keep=False, only=[f"rep{r}" for r in range(1, args.reps + 1)])
    R.log(f"[done] summary: {summary_csv}")
    R.log("STEP3_REAL_DONE")


if __name__ == "__main__":
    main()
