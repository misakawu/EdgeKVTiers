#!/usr/bin/env python3
"""Aggregate H1 run_all_cell reps into one visualization-ready CSV.

The output is a long table with one row per tier/budget/policy/metric:

    h1/out/run_all_cell_ci/h1_visualization_data.csv

For each metric group, samples are collected across reps. The interval
workflow follows h1/数据处理.md: mark IQR outliers without dropping them,
use log-space statistics for skewed positive samples, compute both Student-t
and bootstrap intervals, prefer t-CI when it agrees with bootstrap, and fall
back to bootstrap median CI when the mean interval is too wide.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


METRICS: tuple[str, ...] = (
    "p95_ttft_ms",
    "p50_ttft_ms",
    "mean_ttft_ms",
    "hit_rate",
    "prefill_ms",
    "prefill_p95_ms",
    "policy_time_us_avg",
    "eviction_decision_time_us_avg",
    "gpu_memory_peak_mib",
)

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "p95_ttft_ms": ("p95_ttft_ms", "ttft_proxy_p95_ms", "latency_p95_ms"),
    "p50_ttft_ms": ("p50_ttft_ms", "ttft_proxy_p50_ms"),
    "mean_ttft_ms": ("mean_ttft_ms", "latency_mean_ms"),
    "hit_rate": ("hit_rate", "native_hit_rate", "gpu_block_lookup_hit_rate", "block_lookup_hit_rate"),
    "prefill_ms": ("prefill_ms",),
    "prefill_p95_ms": ("prefill_p95_ms",),
    "policy_time_us_avg": ("policy_time_us_avg",),
    "eviction_decision_time_us_avg": ("eviction_decision_time_us_avg",),
    "gpu_memory_peak_mib": ("gpu_memory_peak_mib",),
}

UNITS: dict[str, str] = {
    "p95_ttft_ms": "ms",
    "p50_ttft_ms": "ms",
    "mean_ttft_ms": "ms",
    "hit_rate": "ratio",
    "prefill_ms": "ms",
    "prefill_p95_ms": "ms",
    "policy_time_us_avg": "us",
    "eviction_decision_time_us_avg": "us",
    "gpu_memory_peak_mib": "MiB",
}

T_CRITICAL_95_TWO_SIDED = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    50: 2.009,
    60: 2.000,
    80: 1.990,
    100: 1.984,
    120: 1.980,
}

BASELINE_POLICIES = ("h1_lru", "h1_lfu", "vllm_default")
OUTPUT_COLUMNS = (
    "tier",
    "budget",
    "policy",
    "metric",
    "n",
    "value",
    "ci95_low",
    "ci95_high",
    "ci95_half_width",
    "unit",
    "method",
    "warning",
    "run_roots",
    "reps",
    "baseline_policy",
    "gain_pct_vs_lru",
    "gain_pct_vs_lfu",
    "gain_pct_vs_vllm_default",
    "pass_h1_vs_lru",
)


@dataclass(frozen=True)
class SourceCsv:
    run_root: str
    rep: str
    tier: str
    path: Path
    kind: str


@dataclass(frozen=True)
class SampleRow:
    run_root: str
    rep: str
    tier: str
    budget: str
    policy: str
    row: dict[str, str]
    source_path: Path
    source_kind: str


@dataclass(frozen=True)
class StatResult:
    n: int
    value: float | None
    ci_low: float | None
    ci_high: float | None
    method: str
    warning: str


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def metric_value(row: dict[str, str], metric: str) -> float | None:
    for field in FIELD_ALIASES[metric]:
        value = parse_float(row.get(field))
        if value is not None:
            return value
    return None


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def fmt_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def sort_budget(value: str) -> tuple[int, float | str]:
    parsed = parse_float(value)
    return (0, parsed) if parsed is not None else (1, value)


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[int(pos)]
    weight = pos - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def t_critical_95_two_sided(df: int) -> float:
    if df <= 0:
        return math.nan
    if df in T_CRITICAL_95_TWO_SIDED:
        return T_CRITICAL_95_TWO_SIDED[df]
    for cutoff in (40, 50, 60, 80, 100, 120):
        if df <= cutoff:
            return T_CRITICAL_95_TWO_SIDED[cutoff]
    return 1.96


def stable_seed(seed: int, parts: tuple[str, ...]) -> int:
    digest = hashlib.sha256(("\0".join(parts)).encode("utf-8")).hexdigest()
    return (seed + int(digest[:16], 16)) % (2**32)


def bootstrap_interval(
    values: list[float],
    iters: int,
    seed: int,
    reducer,
) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    estimates: list[float] = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(reducer(sample))
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def iqr_outlier_warning(values: list[float]) -> str | None:
    if len(values) < 4:
        return None
    ordered = sorted(values)
    q1 = percentile(ordered, 0.25)
    q3 = percentile(ordered, 0.75)
    iqr = q3 - q1
    low_fence = q1 - 1.5 * iqr
    high_fence = q3 + 1.5 * iqr
    count = sum(1 for value in values if value < low_fence or value > high_fence)
    return f"iqr_outliers={count}" if count else None


def compute_stats(
    values: list[float],
    min_samples: int,
    bootstrap_iters: int,
    seed: int,
) -> StatResult:
    warnings: list[str] = []
    n = len(values)
    if n < min_samples:
        warnings.append(f"n<{min_samples}")
    if n == 0:
        return StatResult(n, None, None, None, "no_samples", ";".join(warnings))

    outlier_warning = iqr_outlier_warning(values)
    if outlier_warning:
        warnings.append(outlier_warning)

    if n == 1:
        return StatResult(n, values[0], values[0], values[0], "single_sample", ";".join(warnings))

    center_mean = statistics.fmean(values)
    center_median = statistics.median(values)
    use_log = (
        center_median != 0
        and all(value > 0 for value in values)
        and (center_mean > 1.2 * center_median or center_mean < 0.8 * center_median)
    )
    work_values = [math.log(value) for value in values] if use_log else list(values)
    if use_log:
        warnings.append("log_transform=skewed")

    work_mean = statistics.fmean(work_values)
    work_std = statistics.stdev(work_values)
    se = work_std / math.sqrt(n)
    half_width = t_critical_95_two_sided(n - 1) * se
    t_low = work_mean - half_width
    t_high = work_mean + half_width

    boot_low, boot_high = bootstrap_interval(
        work_values,
        bootstrap_iters,
        seed,
        statistics.fmean,
    )

    if use_log:
        value = math.exp(work_mean)
        t_ci = (math.exp(t_low), math.exp(t_high))
        boot_ci = (math.exp(boot_low), math.exp(boot_high))
    else:
        value = work_mean
        t_ci = (t_low, t_high)
        boot_ci = (boot_low, boot_high)

    denom = abs(boot_ci[0])
    lower_rel_diff = abs(t_ci[0] - boot_ci[0]) / denom if denom > 1e-12 else abs(t_ci[0] - boot_ci[0])
    if lower_rel_diff < 0.20:
        ci_low, ci_high = t_ci
        method = "t_ci_bootstrap_verified_log" if use_log else "t_ci_bootstrap_verified"
    else:
        ci_low, ci_high = boot_ci
        method = "bootstrap_mean_log" if use_log else "bootstrap_mean"
        warnings.append("t_bootstrap_lower_diff>=20pct")

    if ci_high - ci_low > 2.0 * abs(value):
        median_value = statistics.median(values)
        median_low, median_high = bootstrap_interval(
            values,
            bootstrap_iters,
            seed + 1,
            statistics.median,
        )
        value = median_value
        ci_low, ci_high = median_low, median_high
        method = "bootstrap_median"
        warnings.append("wide_interval_use_median")

    return StatResult(n, value, ci_low, ci_high, method, ";".join(warnings))


def tier_dirs_for_rep(rep_dir: Path, tier: str | None) -> tuple[list[Path], list[str]]:
    warnings: list[str] = []
    if tier:
        return [rep_dir / tier], warnings

    candidates = sorted(
        child
        for child in rep_dir.iterdir()
        if child.is_dir()
        and (
            (child / "step3_summary.csv").exists()
            or any(child.glob("*/*/h1_vllm_real_summary.csv"))
        )
    )
    if not candidates:
        warnings.append(f"no tier data under {rep_dir}")
    elif len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        warnings.append(f"ambiguous tiers under {rep_dir}: {names}; use --tier")
        return [], warnings
    return candidates, warnings


def discover_sources(base_glob: str, tier: str | None) -> tuple[list[SourceCsv], list[str]]:
    warnings: list[str] = []
    sources: list[SourceCsv] = []
    roots = [Path(item) for item in sorted(glob.glob(base_glob)) if Path(item).is_dir()]
    if not roots:
        return [], [f"base glob matched no directories: {base_glob}"]

    for root in roots:
        for rep_dir in sorted(root.glob("rep*")):
            if not rep_dir.is_dir():
                continue
            tier_dirs, tier_warnings = tier_dirs_for_rep(rep_dir, tier)
            warnings.extend(tier_warnings)
            for tier_dir in tier_dirs:
                if not tier_dir.exists():
                    warnings.append(f"missing tier dir: {tier_dir}")
                    continue
                cell_csvs = sorted(tier_dir.glob("*/*/h1_vllm_real_summary.csv"))
                if cell_csvs:
                    for path in cell_csvs:
                        sources.append(SourceCsv(str(root), rep_dir.name, tier_dir.name, path, "cell"))
                    continue
                summary = tier_dir / "step3_summary.csv"
                if summary.exists():
                    sources.append(SourceCsv(str(root), rep_dir.name, tier_dir.name, summary, "step3"))
                else:
                    warnings.append(f"missing h1_vllm_real_summary.csv and step3_summary.csv under {tier_dir}")

    if not sources:
        warnings.append("no source CSV files found")
    return sources, warnings


def read_samples(sources: list[SourceCsv]) -> tuple[list[SampleRow], list[str]]:
    samples: list[SampleRow] = []
    warnings: list[str] = []
    for source in sources:
        try:
            with source.path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception as exc:
            warnings.append(f"unreadable csv: {source.path}: {exc}")
            continue

        for index, row in enumerate(rows, start=2):
            budget = row.get("budget") or (source.path.parents[1].name if source.kind == "cell" else "")
            policy = row.get("policy") or (source.path.parent.name if source.kind == "cell" else "")
            if not budget or not policy:
                warnings.append(f"missing budget/policy at {source.path}:{index}")
                continue
            samples.append(
                SampleRow(
                    run_root=source.run_root,
                    rep=source.rep,
                    tier=source.tier,
                    budget=budget,
                    policy=policy,
                    row=row,
                    source_path=source.path,
                    source_kind=source.kind,
                )
            )
    return samples, warnings


def gain_pct(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def aggregate(
    samples: list[SampleRow],
    min_samples: int,
    bootstrap_iters: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[str]]:
    warnings: list[str] = []
    groups: dict[tuple[str, str, str, str], list[SampleRow]] = defaultdict(list)
    for sample in samples:
        for metric in METRICS:
            groups[(sample.tier, sample.budget, sample.policy, metric)].append(sample)

    p95_values: dict[tuple[str, str, str], float | None] = {}
    rows_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    group_keys = sorted(groups, key=lambda key: (key[0], sort_budget(key[1]), key[2], key[3]))
    for key in group_keys:
        tier, budget, policy, metric = key
        sample_rows = groups[key]
        values: list[float] = []
        missing_count = 0
        for sample in sample_rows:
            value = metric_value(sample.row, metric)
            if value is None:
                missing_count += 1
            else:
                values.append(value)

        stat_seed = stable_seed(seed, key)
        stat = compute_stats(values, min_samples, bootstrap_iters, stat_seed)
        warning_items = [item for item in (stat.warning, f"missing={missing_count}" if missing_count else "") if item]
        warning = ";".join(warning_items)
        if warning:
            warnings.append(f"{tier},{budget},{policy},{metric}: {warning}")

        if metric == "p95_ttft_ms":
            p95_values[(tier, budget, policy)] = stat.value

        run_roots = " ".join(sorted({sample.run_root for sample in sample_rows}))
        reps = " ".join(sorted({sample.rep for sample in sample_rows}))
        row = {
            "tier": tier,
            "budget": budget,
            "policy": policy,
            "metric": metric,
            "n": str(stat.n),
            "value": fmt(stat.value),
            "ci95_low": fmt(stat.ci_low),
            "ci95_high": fmt(stat.ci_high),
            "ci95_half_width": fmt((stat.ci_high - stat.ci_low) / 2.0 if stat.ci_low is not None and stat.ci_high is not None else None),
            "unit": UNITS[metric],
            "method": stat.method,
            "warning": warning,
            "run_roots": run_roots,
            "reps": reps,
            "baseline_policy": "",
            "gain_pct_vs_lru": "",
            "gain_pct_vs_lfu": "",
            "gain_pct_vs_vllm_default": "",
            "pass_h1_vs_lru": "",
        }
        rows_by_key[key] = row

    for (tier, budget, policy, metric), row in rows_by_key.items():
        if policy != "h1_lpe" or metric != "p95_ttft_ms":
            continue
        lpe_value = p95_values.get((tier, budget, "h1_lpe"))
        gains = {
            baseline: gain_pct(lpe_value, p95_values.get((tier, budget, baseline)))
            for baseline in BASELINE_POLICIES
        }
        row["baseline_policy"] = " ".join(BASELINE_POLICIES)
        row["gain_pct_vs_lru"] = fmt(gains["h1_lru"])
        row["gain_pct_vs_lfu"] = fmt(gains["h1_lfu"])
        row["gain_pct_vs_vllm_default"] = fmt(gains["vllm_default"])
        row["pass_h1_vs_lru"] = fmt_bool(gains["h1_lru"] is not None and gains["h1_lru"] >= 20.0)
        for baseline, value in gains.items():
            if value is None:
                warnings.append(f"{tier},{budget},h1_lpe,p95_ttft_ms: missing gain baseline {baseline}")

    rows = [rows_by_key[key] for key in group_keys]
    return rows, warnings


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-glob",
        default="h1/out/run_all_cell_3x_rep*",
        help="run_all_cell root glob. Default: h1/out/run_all_cell_3x_rep*",
    )
    parser.add_argument(
        "--tier",
        default=None,
        help="Tier directory name, for example sharegpt_batch_8. If omitted, each rep must contain one tier.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("h1/out/run_all_cell_ci/h1_visualization_data.csv"),
        help="Output CSV path. Default: h1/out/run_all_cell_ci/h1_visualization_data.csv",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=10000,
        help="Bootstrap iterations per metric group. Default: 10000.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260711,
        help="Base seed for deterministic bootstrap resampling.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum valid samples before adding a warning. Default: 1.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any warning is produced.",
    )
    args = parser.parse_args()

    if args.bootstrap_iters < 1:
        parser.error("--bootstrap-iters must be >= 1")
    if args.min_samples < 1:
        parser.error("--min-samples must be >= 1")

    sources, warnings = discover_sources(args.base_glob, args.tier)
    samples, read_warnings = read_samples(sources)
    warnings.extend(read_warnings)
    rows, aggregate_warnings = aggregate(samples, args.min_samples, args.bootstrap_iters, args.bootstrap_seed)
    warnings.extend(aggregate_warnings)
    write_csv(args.out, rows)

    print(f"wrote {args.out} ({len(rows)} rows, {len(sources)} source CSVs, {len(samples)} samples)")
    if warnings:
        unique_warnings = list(dict.fromkeys(warnings))
        print(f"warnings: {len(unique_warnings)}", file=sys.stderr)
        for warning in unique_warnings[:40]:
            print(f"[warn] {warning}", file=sys.stderr)
        if len(unique_warnings) > 40:
            print(f"[warn] ... {len(unique_warnings) - 40} more", file=sys.stderr)
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
