#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${1:-h1/step1_D3/runtime}"
VISIBLE_DEVICES="${2:-${H1_VISIBLE_DEVICES:-0,1}}"

run_bucket() {
  local label="$1"
  local gpu_util="$2"
  local port="$3"
  local cell_dir="$OUT_ROOT/$label"

  mkdir -p "$cell_dir"
  : > "$cell_dir/runtime_monitor.jsonl"

  echo "[step1_D3] START label=$label gpu_memory_utilization=$gpu_util monitor=$cell_dir/runtime_monitor.jsonl"
  EDGEKV_H1_RUNTIME_MONITOR=1 \
  EDGEKV_H1_RUNTIME_MONITOR_PATH="$cell_dir/runtime_monitor.jsonl" \
  EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES=1 \
  EDGEKV_H1_STATS_FLUSH_INTERVAL="${EDGEKV_H1_STATS_FLUSH_INTERVAL:-32}" \
  H1_GPU_POLICY=h1_lpe \
  H1_GPU_MEMORY_UTILIZATION="$gpu_util" \
  H1_SERVE_PORT="$port" \
  H1_PREFIX_REPETITION_NUM_PREFIXES="${H1_PREFIX_REPETITION_NUM_PREFIXES:-8}" \
  H1_PREFIX_REPETITION_PREFIX_LEN="${H1_PREFIX_REPETITION_PREFIX_LEN:-512}" \
  H1_PREFIX_REPETITION_SUFFIX_LEN="${H1_PREFIX_REPETITION_SUFFIX_LEN:-128}" \
  H1_PREFIX_REPETITION_OUTPUT_LEN="${H1_PREFIX_REPETITION_OUTPUT_LEN:-1}" \
  H1_BENCH_NUM_PROMPTS="${H1_BENCH_NUM_PROMPTS:-128}" \
  bash h1/run_h1_policy_serving_bench.sh "$cell_dir/bench" "$VISIBLE_DEVICES"
  echo "[step1_D3] DONE label=$label"
}

run_bucket tight "${H1_STEP1_D3_TIGHT_GPU_UTIL:-0.710}" "${H1_STEP1_D3_TIGHT_PORT:-8110}"
run_bucket mid "${H1_STEP1_D3_MID_GPU_UTIL:-0.720}" "${H1_STEP1_D3_MID_PORT:-8111}"

python3 h1/step1_D3/build_step1_d3.py --runtime-root "$OUT_ROOT"
echo "[step1_D3] artifacts: h1/step1_D3/out/step1_D3_c_recomp_vs_n.png h1/step1_D3/out/step1_D3_summary.json h1/step1_D3/step1_d3_report.md"
