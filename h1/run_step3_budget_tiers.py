#!/usr/bin/env python3
"""在 pressure replay trace 上运行 Step3 预算档校准/策略对比。

运行 H1 real replay harness，而不是 vLLM 内置 benchmark 数据集。workload 默认
使用冻结的 H0 ShareGPT+HotpotQA pressure trace，因此 Step3 与 H1 其他部分使用
同样偏斜、高频的 HotpotQA chunk 复用。

    python h1/run_step3_budget_tiers.py
    python h1/run_step3_budget_tiers.py --budgets tight mid --policies h1_lru h1_lpe

所有配置都在下方 CONFIG 块内，无需环境变量。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _runner as R

# ----------------------------------------------------------------------------- 配置
DEVICES = "0,1"
TIER = "tight"
BASE_OUT = Path("h1/out/step3")

BUDGETS = ["tight", "mid", "loose"]
POLICIES = ["h1_lru", "h1_lfu", "vllm_default", "h1_lpe"]

WORKLOAD = "mixed"
REPLAY_TRACE = Path("data/edgekv_traces/sharegpt_hotpotqa_session.jsonl")
HOTPOTQA_PATH = "data/hotpotqa"
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
# REPLAY_BATCH_SIZE = 64
REPLAY_BATCH_SIZE=16
# MAX_NUM_BATCHED_TOKENS = 64
MAX_NUM_BATCHED_TOKENS = 32768   
TENSOR_PARALLEL_SIZE = 2
PYTHONPATH = ".:h1:h0"
# --------------------------------------------------------------------------------------


def cell_args(cell_dir: Path, budget: str, policy: str, max_requests: int,
              replay_batch_size: int, replay_trace: Path, hotpotqa_path: str,
              visible_devices: str, max_num_batched_tokens: int,
              max_model_len: int) -> list[str]:
    return [
        "--out", str(cell_dir),
        "--dtype", "float16",
        "--policies", policy,
        "--budgets", budget,
        "--workload", WORKLOAD,
        "--replay-trace", str(replay_trace),
        "--hotpotqa-path", hotpotqa_path,
        "--hotpotqa-max-examples", str(HOTPOTQA_MAX_EXAMPLES),
        "--max-sessions", str(MAX_SESSIONS),
        "--max-requests", str(max_requests),
        "--rag-requests", str(RAG_REQUESTS),
        "--rag-chunk-words", str(RAG_CHUNK_WORDS),
        "--rag-chunks-per-query", str(RAG_CHUNKS_PER_QUERY),
        "--rag-query-repeats", str(RAG_QUERY_REPEATS),
        "--sharegpt-order", SHAREGPT_ORDER,
        "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
        "--max-model-len", str(max_model_len),
        "--max-tokens", str(MAX_TOKENS),
        "--replay-batch-size", str(replay_batch_size),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--visible-devices", visible_devices,
    ]


def run_step3(*, tier=TIER, base_out=BASE_OUT, budgets=BUDGETS, policies=POLICIES,
              num_prompts=MAX_REQUESTS, request_rate=0.0, visible_devices=DEVICES,
              replay_trace=REPLAY_TRACE, replay_batch_size=REPLAY_BATCH_SIZE,
              max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
              max_model_len=MAX_MODEL_LEN,
              hotpotqa_path=HOTPOTQA_PATH, no_finalize=False, force=False,
              keep_cells=False) -> Path:
    """在 pressure replay trace 上运行单个 tier 的 budget x policy 矩阵。"""
    base_out = Path(base_out)
    replay_trace = Path(replay_trace)
    tier_dir = base_out / tier
    log_dir = tier_dir / "logs"
    if not R.DRY_RUN:
        log_dir.mkdir(parents=True, exist_ok=True)

    R.log(f"[step3] tier={tier} out={tier_dir} budgets={budgets} policies={policies}")
    R.log(f"[step3] workload=pressure_replay trace={replay_trace} max_requests={num_prompts} bs={replay_batch_size}")
    for budget in budgets:
        for policy in policies:
            out_dir = tier_dir / budget / policy
            summary_json = out_dir / f"{budget}_{policy}_summary.json"
            R.log(f"[run] tier={tier} budget={budget} policy={policy} max_requests={num_prompts}")
            if summary_json.exists() and not force:
                R.log(f"[skip] {summary_json} already exists")
                continue
            if not R.DRY_RUN:
                out_dir.mkdir(parents=True, exist_ok=True)
            env_overrides = {
                "PYTHONPATH": PYTHONPATH,
                "EDGEKV_H1_GPU_POLICY": policy,
                "EDGEKV_H1_PROFILE_POLICY_TIME": "1",
            }
            rc = R.run_real_cell(
                out_dir,
                visible_devices,
                cell_args(out_dir, budget, policy, num_prompts, replay_batch_size,
                          replay_trace, hotpotqa_path, visible_devices,
                          max_num_batched_tokens, max_model_len),
                env_overrides,
                log_file=log_dir / f"{budget}_{policy}.log",
            )
            if rc != 0:
                raise RuntimeError(f"real replay cell failed (rc={rc}): {out_dir}")
            if summary_json.exists():
                try:
                    summary = json.loads(summary_json.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise RuntimeError(f"cannot read real replay summary: {summary_json}") from exc
                if summary.get("ok") is False:
                    err = summary.get("error", "summary marked ok=false")
                    raise RuntimeError(f"real replay cell failed: {out_dir}: {err}")
    R.log(f"[step3] done tier={tier}")

    if no_finalize:
        R.log(f"[step3] finalize deferred (no_finalize=True) tier={tier}")
        return tier_dir

    summary_csv = tier_dir / "step3_summary.csv"
    R.log(f"[summary] writing {summary_csv}")
    R.summarize("summarize_step3_budget_tiers.py",
                ["--out", str(tier_dir), "--summary", str(summary_csv),
                 "--request-rate", str(request_rate)])
    R.cleanup_dirs(tier_dir, keep=keep_cells, extra=[log_dir])
    R.log(f"[done] summary: {summary_csv}")
    return tier_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--visible-devices", default=DEVICES)
    ap.add_argument("--tier", default=TIER)
    ap.add_argument("--base-out", default=str(BASE_OUT))
    ap.add_argument("--budgets", default=" ".join(BUDGETS))
    ap.add_argument("--policies", default=" ".join(POLICIES))
    ap.add_argument("--num-prompts", type=int, default=MAX_REQUESTS)
    ap.add_argument("--replay-trace", default=str(REPLAY_TRACE))
    ap.add_argument("--replay-batch-size", type=int, default=REPLAY_BATCH_SIZE)
    ap.add_argument("--max-num-batched-tokens", type=int, default=MAX_NUM_BATCHED_TOKENS)
    ap.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    ap.add_argument("--hotpotqa-path", default=HOTPOTQA_PATH)
    ap.add_argument("--no-finalize", action="store_true", help="defer summary/cleanup (for repeat)")
    ap.add_argument("--force", action="store_true", help="rerun cells even if summary JSON exists")
    ap.add_argument("--keep-cells", action="store_true", help="retain per-cell outputs and logs")
    args = ap.parse_args()

    run_step3(
        tier=args.tier,
        base_out=Path(args.base_out),
        budgets=args.budgets.split(),
        policies=args.policies.split(),
        num_prompts=args.num_prompts,
        visible_devices=args.visible_devices,
        replay_trace=Path(args.replay_trace),
        replay_batch_size=args.replay_batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        hotpotqa_path=args.hotpotqa_path,
        no_finalize=args.no_finalize,
        force=args.force,
        keep_cells=args.keep_cells,
    )


if __name__ == "__main__":
    main()
