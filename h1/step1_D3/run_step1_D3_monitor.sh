#!/usr/bin/env bash
# H1 Step1 D3 — object-level COP monitor (path A).
#
# Drives h1/run_h1_vllm0110_real.py directly (policy h1_lpe) so that the monitored
# c_recomp / p_reuse / score come from the COP module (edgekv_cop.py,
# score_source=object_level_cop), NOT from sitecustomize's block-level inline
# profiles. build_step1_d3.py then reads the per-request COP CSV.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${1:-h1/step1_D3/runtime}"
VISIBLE_DEVICES="${2:-${H1_VISIBLE_DEVICES:-0,1}}"

CONDA_ENV="${H1_CONDA_ENV:-edgekv-vllm0110}"
MODEL="${H1_MODEL:-models/Qwen2.5-7B-Instruct}"
REPLAY_TRACE="${H1_REPLAY_TRACE:-data/edgekv_traces/h0_sharegpt_hotpotqa_200sessions_pressure.jsonl}"
HOTPOTQA_PATH="${H1_HOTPOTQA_PATH:-data/hotpotqa}"
NUM_PROMPTS="${H1_BENCH_NUM_PROMPTS:-128}"
REPLAY_BATCH_SIZE="${H1_REPLAY_BATCH_SIZE:-1}"
MAX_MODEL_LEN="${H1_MAX_MODEL_LEN:-2048}"
MAX_TOKENS="${H1_MAX_TOKENS:-16}"
TENSOR_PARALLEL_SIZE="${H1_TENSOR_PARALLEL_SIZE:-2}"
ATTENTION_BACKEND="${H1_ATTENTION_BACKEND:-TRITON_ATTN}"

run_bucket() {
  local label="$1"      # bucket dir name
  local budget="$2"     # run_h1_vllm0110_real named budget (tight=0.720, mid=0.735)
  local cell_dir="$OUT_ROOT/$label"

  mkdir -p "$cell_dir"
  local log="$cell_dir/run.log"
  : > "$log"

  echo "[step1_D3] START label=$label budget=$budget out=$cell_dir" | tee -a "$log"
  PYTHONPATH=.:h1:h0 \
  CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" \
  VLLM_USE_V1=1 \
  VLLM_ATTENTION_BACKEND="$ATTENTION_BACKEND" \
  VLLM_NO_USAGE_STATS=1 \
  EDGEKV_H1_GPU_POLICY=h1_lpe \
  EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES=1 \
  EDGEKV_C_RE_MS_PER_TOKEN="${EDGEKV_C_RE_MS_PER_TOKEN:-0.12}" \
  EDGEKV_BW_GBPS="${EDGEKV_BW_GBPS:-1.0}" \
  EDGEKV_D_DESER_MS="${EDGEKV_D_DESER_MS:-3.0}" \
  conda run --no-capture-output -n "$CONDA_ENV" \
    python h1/run_h1_vllm0110_real.py \
      --out "$cell_dir" \
      --model "$MODEL" \
      --dtype float16 \
      --policies h1_lpe \
      --budgets "$budget" \
      --workload mixed \
      --replay-trace "$REPLAY_TRACE" \
      --hotpotqa-path "$HOTPOTQA_PATH" \
      --max-requests "$NUM_PROMPTS" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-tokens "$MAX_TOKENS" \
      --replay-batch-size "$REPLAY_BATCH_SIZE" \
      --visible-devices "$VISIBLE_DEVICES" \
    2>&1 | tee -a "$log"
  echo "[step1_D3] DONE label=$label" | tee -a "$log"
}

run_bucket tight "${H1_STEP1_D3_TIGHT_BUDGET:-tight}"
run_bucket mid "${H1_STEP1_D3_MID_BUDGET:-mid}"

python3 h1/step1_D3/build_step1_d3.py --runtime-root "$OUT_ROOT"
echo "[step1_D3] artifacts: h1/step1_D3/out/step1_D3_c_recomp_vs_n.png h1/step1_D3/out/step1_D3_summary.json h1/step1_D3/step1_d3_report.md"
