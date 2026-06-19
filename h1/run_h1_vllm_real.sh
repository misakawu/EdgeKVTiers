#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${1:-out/h1_vllm_real_$(date +%Y%m%d_%H%M%S)}"
VISIBLE_DEVICES="${2:-${H1_VISIBLE_DEVICES:-0,1}}"
ATTENTION_BACKEND="${H1_ATTENTION_BACKEND:-TRITON_ATTN}"
LOG_FILE="$OUT_DIR/run.log"
PID_FILE="$OUT_DIR/runner.pid"
LOCK_DIR="/tmp/edgekv_h1_vllm_real.lock"
SUMMARY_AGG="$OUT_DIR/h1_vllm_real_summary.aggregate.csv"
REQUESTS_AGG="$OUT_DIR/h1_vllm_real_requests.aggregate.csv"
SUMMARY_FINAL="$OUT_DIR/h1_vllm_real_summary.csv"
REQUESTS_FINAL="$OUT_DIR/h1_vllm_real_requests.csv"

POLICIES=(vllm_default h1_lru h1_lfu h1_lpe)
BUDGETS=(tight mid loose)
TOTAL=$(( ${#POLICIES[@]} * ${#BUDGETS[@]} ))

IFS=',' read -r -a GPU_IDS <<< "$VISIBLE_DEVICES"
if [[ "${#GPU_IDS[@]}" -ne 2 ]]; then
  echo "[run] expected exactly two GPU ids, got '$VISIBLE_DEVICES'" >&2
  echo "[run] usage: $0 [out_dir] [gpu_ids_csv], e.g. $0 out/h1_vllm_real_final 1,2" >&2
  exit 2
fi
TENSOR_PARALLEL_SIZE=2

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$$" > "$LOCK_DIR/pid"
    return 0
  fi
  local old_pid=""
  if [[ -f "$LOCK_DIR/pid" ]]; then
    old_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  fi
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[run] another H1 run is active: pid=$old_pid lock=$LOCK_DIR" >&2
    exit 1
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
  echo "$$" > "$LOCK_DIR/pid"
}

mkdir -p "$OUT_DIR"
: > "$LOG_FILE"
rm -f "$SUMMARY_AGG" "$REQUESTS_AGG"

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  if [[ -s "$PID_FILE" ]]; then
    local runner_pgid
    runner_pgid="$(cat "$PID_FILE")"
    if kill -0 "$runner_pgid" 2>/dev/null; then
      echo "[cleanup] stopping H1 process group pgid=$runner_pgid" | tee -a "$LOG_FILE"
      kill -TERM -- "-$runner_pgid" 2>/dev/null || true
      sleep 10
      kill -KILL -- "-$runner_pgid" 2>/dev/null || true
    fi
  fi

  rm -rf "$LOCK_DIR"
  echo "[cleanup] GPU processes after cleanup:" | tee -a "$LOG_FILE"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>&1 | tee -a "$LOG_FILE" || true
  exit "$exit_code"
}

append_csv() {
  local src="$1"
  local dst="$2"
  if [[ ! -s "$src" ]]; then
    return 0
  fi
  if [[ ! -s "$dst" ]]; then
    cat "$src" > "$dst"
  else
    tail -n +2 "$src" >> "$dst"
  fi
}

print_cell_summary() {
  local csv_file="$1"
  python - "$csv_file" <<'PY'
import csv
import sys
path = sys.argv[1]
try:
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
except FileNotFoundError:
    print('summary=missing')
    raise SystemExit(0)
if not rows:
    print('summary=empty')
    raise SystemExit(0)
row = rows[-1]
print(
    'ok={ok} requests={requests} p95_ms={p95} hit_rate={hit} peak_mib={peak} error={error}'.format(
        ok=row.get('ok', ''),
        requests=row.get('requests', ''),
        p95=row.get('ttft_proxy_p95_ms', ''),
        hit=row.get('hit_rate', ''),
        peak=row.get('gpu_memory_peak_mib', ''),
        error=(row.get('error', '') or '').replace('\n', ' ')[:180],
    )
)
PY
}

wait_gpu_idle() {
  local waited=0
  while true; do
    local busy
    busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$VISIBLE_DEVICES" 2>/dev/null | awk 'NF {print}' | wc -l)"
    if [[ "$busy" == "0" ]]; then
      return 0
    fi
    echo "[wait] GPU $VISIBLE_DEVICES busy with $busy compute process(es); waited=${waited}s" | tee -a "$LOG_FILE"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>&1 | tee -a "$LOG_FILE" || true
    sleep 30
    waited=$((waited + 30))
  done
}

run_cell() {
  local policy="$1"
  local budget="$2"
  local index="$3"
  local started
  started="$(date +%s)"

  wait_gpu_idle
  echo "[progress $index/$TOTAL] START budget=$budget policy=$policy" | tee -a "$LOG_FILE"

  setsid env \
    PYTHONPATH=h1:h0 \
    CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" \
    VLLM_USE_V1=1 \
    VLLM_ATTENTION_BACKEND="$ATTENTION_BACKEND" \
    conda run -n edgekv-vllm0110 python h1/run_h1_vllm0110_real.py \
      --out "$OUT_DIR" \
      --dtype float16 \
      --policies "$policy" \
      --budgets "$budget" \
      --max-sessions 200 \
      --max-requests 200 \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --max-model-len 2048 \
      --max-tokens 8 \
      --replay-batch-size 2 \
      --num-cpu-blocks 256 \
      --offload-block-size 16 \
      --visible-devices "$VISIBLE_DEVICES" \
      > >(tee -a "$LOG_FILE") 2>&1 &

  echo "$!" > "$PID_FILE"
  if ! wait "$(cat "$PID_FILE")"; then
    echo "[progress $index/$TOTAL] PROCESS_FAILED budget=$budget policy=$policy" | tee -a "$LOG_FILE"
  fi
  : > "$PID_FILE"

  append_csv "$SUMMARY_FINAL" "$SUMMARY_AGG"
  append_csv "$REQUESTS_FINAL" "$REQUESTS_AGG"

  local elapsed
  elapsed=$(( $(date +%s) - started ))
  local cell_summary
  cell_summary="$(print_cell_summary "$SUMMARY_FINAL")"
  echo "[progress $index/$TOTAL] DONE budget=$budget policy=$policy elapsed=${elapsed}s $cell_summary" | tee -a "$LOG_FILE"
  echo "[progress $index/$TOTAL] GPU process snapshot:" | tee -a "$LOG_FILE"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>&1 | tee -a "$LOG_FILE" || true
}

trap cleanup EXIT INT TERM
acquire_lock

echo "[run] output directory: $OUT_DIR" | tee -a "$LOG_FILE"
echo "[run] H1 real vLLM matrix: 4 policies x 3 GPU memory budgets x ShareGPT first 200 requests" | tee -a "$LOG_FILE"
echo "[run] execution mode: one vLLM process per policy/budget cell with CSV aggregation" | tee -a "$LOG_FILE"
echo "[run] attention backend: VLLM_ATTENTION_BACKEND=$ATTENTION_BACKEND" | tee -a "$LOG_FILE"
echo "[run] CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES tensor_parallel_size=$TENSOR_PARALLEL_SIZE" | tee -a "$LOG_FILE"

idx=0
for budget in "${BUDGETS[@]}"; do
  for policy in "${POLICIES[@]}"; do
    idx=$((idx + 1))
    run_cell "$policy" "$budget" "$idx"
  done
done

if [[ -s "$SUMMARY_AGG" ]]; then
  mv "$SUMMARY_AGG" "$SUMMARY_FINAL"
fi
if [[ -s "$REQUESTS_AGG" ]]; then
  mv "$REQUESTS_AGG" "$REQUESTS_FINAL"
fi

echo "[run] summary CSV: $SUMMARY_FINAL" | tee -a "$LOG_FILE"
echo "[run] request CSV: $REQUESTS_FINAL" | tee -a "$LOG_FILE"
echo "[run] completed all $TOTAL cells" | tee -a "$LOG_FILE"
