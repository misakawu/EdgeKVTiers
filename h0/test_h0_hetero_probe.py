#!/usr/bin/env python3
"""H0 异质解析探针测试。"""

from __future__ import annotations

import csv
import tempfile
from argparse import Namespace
from pathlib import Path

import run_h0_hetero_probe as probe


def test_homogeneous_score_orders_like_p_reuse() -> None:
    objects = probe.build_objects("homogeneous")
    states = {}
    for index, oid in enumerate(sorted(objects)[:24]):
        state = probe.ObjectState()
        for step in range(index % 5 + 1):
            probe.update_profile(state, step + index * 10)
        state.score = probe.score_object(objects[oid], state)
        states[oid] = state

    for left, left_state in states.items():
        for right, right_state in states.items():
            p_delta = left_state.p_reuse - right_state.p_reuse
            score_delta = left_state.score - right_state.score
            assert p_delta * score_delta >= -1e-12, (left, right, p_delta, score_delta)


def test_all_policies_run_on_fixed_trace() -> None:
    objects = probe.build_objects("heterogeneous")
    trace = probe.generate_trace(objects, seed=20260701, rep=0, trace_len=240)
    for policy in probe.POLICIES:
        summary, events = probe.simulate_policy("heterogeneous", objects, trace, 0.35, policy, 0)
        assert 0.0 <= summary["hit_rate"] <= 1.0
        assert summary["p95_cost"] > 0.0
        assert summary["cached_objects"] > 0
        assert events
        assert {"score", "p_reuse", "hit", "object_type"}.issubset(events[0])


def test_heterogeneous_int4_has_higher_recompute_per_size_than_fp16() -> None:
    objects = probe.build_objects("heterogeneous")
    fp16 = [obj.recompute_cost / obj.size for obj in objects.values() if obj.object_type == "fp16_session_kv"]
    int4 = [obj.recompute_cost / obj.size for obj in objects.values() if obj.object_type == "int4_quant_block"]
    assert min(int4) > max(fp16)


def test_summary_csv_schema() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        args = Namespace(
            out_dir=str(Path(tmp) / "h0_hetero_probe"),
            seed=20260701,
            reps=1,
            trace_len=180,
            budgets=[0.25],
        )
        out_dir = probe.run(args)
        with (out_dir / "summary.csv").open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert reader.fieldnames == list(probe.SUMMARY_COLUMNS)
        assert len(rows) == len(probe.SCENARIOS) * len(probe.POLICIES)
        assert (out_dir / "events.jsonl").exists()
        assert (out_dir / "homogeneous_p95_cost.png").stat().st_size > 0
        assert (out_dir / "heterogeneous_hit_rate.png").stat().st_size > 0
        assert (out_dir / "conclusion.md").stat().st_size > 0


if __name__ == "__main__":
    test_homogeneous_score_orders_like_p_reuse()
    test_all_policies_run_on_fixed_trace()
    test_heterogeneous_int4_has_higher_recompute_per_size_than_fp16()
    test_summary_csv_schema()
    print("h0 hetero probe tests passed")
