#!/usr/bin/env python3
"""按 run_test.py 默认配置运行 10x4 矩阵，每个 cell 重复三次后取平均。

该启动器沿用 h1/run_test.py 的默认实验配置，但把每次重复单独保存：

    h1/out/run_all_cell_3x/rep1/<tier>/<budget>/<policy>/
    h1/out/run_all_cell_3x/rep2/<tier>/<budget>/<policy>/
    h1/out/run_all_cell_3x/rep3/<tier>/<budget>/<policy>/

最终 CSV 会对每个 budget/policy cell 的已完成重复取平均值。

启动命令：
    python h1/run_all_cell_3x.py

参数说明：
    --visible-devices：传给每个 cell 的 CUDA_VISIBLE_DEVICES。
    --reps：每个 budget/policy cell 重复运行次数，默认 3。
    --num-prompts：每个 cell 回放请求数。
    --tier：输出 tier 前缀；实际目录会追加 replay_batch_size。
    --replay-trace：JSONL replay trace 输入路径。
    --budgets：要扫描的 gpu_memory_utilization 档位，空格分隔。
    --policies：要运行的缓存策略，空格分隔。
    --replay-batch-size：回放批大小，对应 vLLM max_num_seqs 压力。
    --batch-order：批内请求排序方式；本启动器默认 round_robin。
    --max-model-len：vLLM max_model_len。
    --max-num-batched-tokens：vLLM max_num_batched_tokens。
    --workload：回放类型，sharegpt/rag/mixed。
    --rag-requests：mixed/rag workload 中 RAG 请求数。
    --hotpotqa-max-examples：加载 HotpotQA 的最大样本数。
    --base-out：三次重复的根输出目录。
    --summary：跨重复取平均后的 CSV 输出路径。
    --force：已有 summary JSON 时仍重跑 cell。
    --keep-cells：兼容参数；本脚本始终保留各次重复输出。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import _runner as R
import run_step3_budget_tiers as step3
import run_test
from summarize_step3_budget_tiers import METRICS, as_float, row_from_real_summary


BASE_OUT = Path("h1/out/run_all_cell_3x")
REPS = 3


def remove_failed_summaries(tier_dir: Path, budgets: list[str], policies: list[str]) -> None:
    """删除 ok=false 的 cell summary，使断点续跑能自动重跑失败 cell。"""
    for budget in budgets:
        for policy in policies:
            summary_json = tier_dir / budget / policy / f"{budget}_{policy}_summary.json"
            if not summary_json.exists():
                continue
            try:
                payload = json.loads(summary_json.read_text(encoding="utf-8"))
            except Exception:
                R.log(f"[warn] unreadable summary will be rerun: {summary_json}")
                summary_json.unlink()
                continue
            if payload.get("ok") is False:
                R.log(f"[rerun] removing failed summary: {summary_json}")
                summary_json.unlink()


def mean_row(values: dict[str, list[float]]) -> dict[str, str]:
    row: dict[str, str] = {}
    for metric in METRICS:
        vals = values.get(metric, [])
        row[metric] = f"{statistics.fmean(vals):.6f}" if vals else ""
    return row


def write_rep_summary(
    *,
    tier_dir: Path,
    budgets: list[str],
    policies: list[str],
    summary_csv: Path | None = None,
    request_rate: float = 0.0,
) -> None:
    """写出单个 rep 的 step3_summary.csv，字段对齐 summarize_step3_budget_tiers.py。"""
    rows: list[dict[str, str]] = []
    by_budget: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)

    for budget in budgets:
        for policy in policies:
            summary_json = tier_dir / budget / policy / f"{budget}_{policy}_summary.json"
            if not summary_json.exists():
                R.log(f"[warn] missing rep summary: {summary_json}")
                continue
            source = row_from_real_summary(summary_json)
            if not source:
                R.log(f"[warn] unreadable rep summary: {summary_json}")
                continue
            by_budget[budget][policy] = source
            throughput = as_float(source, "request_throughput")
            row = {
                "budget": budget,
                "policy": policy,
                "saturated": str(bool(request_rate and throughput < 0.8 * request_rate)).lower(),
            }
            for metric in METRICS:
                row[metric] = source.get(metric, "")
            rows.append(row)

    for row in rows:
        row.update(
            {
                "p95_ttft_gain_pct_vs_lru": "",
                "mean_ttft_gain_pct_vs_lru": "",
                "hit_rate_delta_vs_lru": "",
                "eviction_delta_vs_lru": "",
            }
        )
        if row["policy"] != "h1_lpe":
            continue
        lru = by_budget.get(row["budget"], {}).get("h1_lru")
        if not lru:
            continue
        lru_p95 = as_float(lru, "p95_ttft_ms")
        lpe_p95 = as_float(row, "p95_ttft_ms")
        lru_mean = as_float(lru, "mean_ttft_ms")
        lpe_mean = as_float(row, "mean_ttft_ms")
        row["p95_ttft_gain_pct_vs_lru"] = (
            f"{((lru_p95 - lpe_p95) / lru_p95 * 100.0):.6f}" if lru_p95 else ""
        )
        row["mean_ttft_gain_pct_vs_lru"] = (
            f"{((lru_mean - lpe_mean) / lru_mean * 100.0):.6f}" if lru_mean else ""
        )
        row["hit_rate_delta_vs_lru"] = (
            f'{(as_float(row, "hit_rate") - as_float(lru, "hit_rate")):.6f}'
        )
        row["eviction_delta_vs_lru"] = (
            f'{(as_float(row, "gpu_prefix_cache_evictions") - as_float(lru, "gpu_prefix_cache_evictions")):.6f}'
        )

    fields = [
        "budget",
        "policy",
        *METRICS,
        "p95_ttft_gain_pct_vs_lru",
        "mean_ttft_gain_pct_vs_lru",
        "hit_rate_delta_vs_lru",
        "eviction_delta_vs_lru",
        "saturated",
    ]
    summary_csv = summary_csv or tier_dir / "step3_summary.csv"
    if R.DRY_RUN:
        R.log(f"[dry-run] rep summary: {summary_csv} ({len(rows)} rows)")
        return
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    R.log(f"[done] rep summary: {summary_csv} ({len(rows)} rows)")


def write_average_summary(
    *,
    base_out: Path,
    tier: str,
    budgets: list[str],
    policies: list[str],
    reps: int,
    summary_csv: Path,
) -> None:
    collected: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for rep in range(1, reps + 1):
        tier_dir = base_out / f"rep{rep}" / tier
        for budget in budgets:
            for policy in policies:
                summary_json = tier_dir / budget / policy / f"{budget}_{policy}_summary.json"
                if not summary_json.exists():
                    R.log(f"[warn] missing rep summary: {summary_json}")
                    continue
                try:
                    payload = json.loads(summary_json.read_text(encoding="utf-8"))
                except Exception:
                    R.log(f"[warn] unreadable rep summary: {summary_json}")
                    continue
                if payload.get("ok") is False:
                    R.log(f"[warn] failed rep summary ignored: {summary_json}")
                    continue
                source = row_from_real_summary(summary_json)
                if not source:
                    R.log(f"[warn] unreadable rep summary: {summary_json}")
                    continue
                for metric in METRICS:
                    collected[(budget, policy)][metric].append(as_float(source, metric))

    rows: list[dict[str, str]] = []
    by_budget: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for budget in budgets:
        for policy in policies:
            values = collected.get((budget, policy), {})
            reps_completed = max((len(vals) for vals in values.values()), default=0)
            row = {
                "budget": budget,
                "policy": policy,
                "reps": str(reps_completed),
                **mean_row(values),
            }
            rows.append(row)
            by_budget[budget][policy] = row

    for row in rows:
        row.update(
            {
                "p95_ttft_gain_pct_vs_lru": "",
                "mean_ttft_gain_pct_vs_lru": "",
                "hit_rate_delta_vs_lru": "",
                "eviction_delta_vs_lru": "",
            }
        )
        if row["policy"] != "h1_lpe":
            continue
        lru = by_budget.get(row["budget"], {}).get("h1_lru")
        if not lru:
            continue
        lru_p95 = as_float(lru, "p95_ttft_ms")
        lpe_p95 = as_float(row, "p95_ttft_ms")
        lru_mean = as_float(lru, "mean_ttft_ms")
        lpe_mean = as_float(row, "mean_ttft_ms")
        row["p95_ttft_gain_pct_vs_lru"] = (
            f"{((lru_p95 - lpe_p95) / lru_p95 * 100.0):.6f}" if lru_p95 else ""
        )
        row["mean_ttft_gain_pct_vs_lru"] = (
            f"{((lru_mean - lpe_mean) / lru_mean * 100.0):.6f}" if lru_mean else ""
        )
        row["hit_rate_delta_vs_lru"] = (
            f'{(as_float(row, "hit_rate") - as_float(lru, "hit_rate")):.6f}'
        )
        row["eviction_delta_vs_lru"] = (
            f'{(as_float(row, "gpu_prefix_cache_evictions") - as_float(lru, "gpu_prefix_cache_evictions")):.6f}'
        )

    if R.DRY_RUN:
        R.log(f"[dry-run] average summary: {summary_csv} ({len(rows)} rows)")
        return

    fields = [
        "budget",
        "policy",
        "reps",
        *METRICS,
        "p95_ttft_gain_pct_vs_lru",
        "mean_ttft_gain_pct_vs_lru",
        "hit_rate_delta_vs_lru",
        "eviction_delta_vs_lru",
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    R.log(f"[done] average summary: {summary_csv} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-devices", default="1,2")
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--num-prompts", type=int, default=run_test.NUM_PROMPTS)
    parser.add_argument("--tier", default=run_test.TIER)
    parser.add_argument("--replay-trace", type=Path, default=run_test.REPLAY_TRACE)
    parser.add_argument("--budgets", default=" ".join(run_test.BUDGETS))
    parser.add_argument("--policies", default=" ".join(run_test.POLICIES))
    parser.add_argument("--replay-batch-size", type=int, default=run_test.REPLAY_BATCH_SIZE)
    parser.add_argument("--batch-order", choices=("round_robin",), default=run_test.BATCH_ORDER)
    parser.add_argument("--max-model-len", type=int, default=run_test.MAX_MODEL_LEN)
    parser.add_argument("--max-num-batched-tokens", type=int, default=run_test.MAX_NUM_BATCHED_TOKENS)
    parser.add_argument("--workload", choices=("sharegpt", "rag", "mixed"), default=run_test.WORKLOAD)
    parser.add_argument("--rag-requests", type=int, default=run_test.RAG_REQUESTS)
    parser.add_argument("--hotpotqa-max-examples", type=int, default=run_test.HOTPOTQA_MAX_EXAMPLES)
    parser.add_argument("--base-out", type=Path, default=BASE_OUT)
    parser.add_argument("--summary", type=Path, default=BASE_OUT / "run_all_cell_3x_average.csv")
    parser.add_argument("--force", action="store_true", help="即使 summary JSON 已存在也重新运行 cell")
    parser.add_argument("--keep-cells", action="store_true", help="兼容参数；本脚本始终保留各次重复输出")
    args = parser.parse_args()

    budgets = args.budgets.split()
    policies = args.policies.split()
    tier = args.tier + str(args.replay_batch_size)

    R.log(
        f"[protocol] run_test defaults: tier={tier} reps={args.reps} "
        f"cells={len(budgets)}x{len(policies)} out={args.base_out}"
    )
    for rep in range(1, args.reps + 1):
        R.log(f"[protocol] rep={rep}/{args.reps}")
        tier_dir = args.base_out / f"rep{rep}" / tier
        remove_failed_summaries(tier_dir, budgets, policies)
        step3.run_step3(
            tier=tier,
            base_out=args.base_out / f"rep{rep}",
            budgets=budgets,
            policies=policies,
            num_prompts=args.num_prompts,
            visible_devices=args.visible_devices,
            force=args.force,
            keep_cells=True,
            no_finalize=True,
            replay_trace=args.replay_trace,
            replay_batch_size=args.replay_batch_size,
            batch_order=args.batch_order,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
            workload=args.workload,
            rag_requests=args.rag_requests,
            hotpotqa_max_examples=args.hotpotqa_max_examples,
        )
        write_rep_summary(tier_dir=tier_dir, budgets=budgets, policies=policies)

    write_average_summary(
        base_out=args.base_out,
        tier=tier,
        budgets=budgets,
        policies=policies,
        reps=args.reps,
        summary_csv=args.summary,
    )


if __name__ == "__main__":
    main()
