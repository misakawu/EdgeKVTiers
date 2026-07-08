#!/usr/bin/env bash
# 启动命令：
#   bash h1/run_h1_policy_serving_bench.sh [OUT_DIR] [VISIBLE_DEVICES]
#
# 位置参数说明：
#   OUT_DIR：cell 输出目录；不传时写入带时间戳的 h1/out/h1_policy_pressure_replay_*。
#   VISIBLE_DEVICES：传给 CUDA_VISIBLE_DEVICES；不传时读取 H1_VISIBLE_DEVICES，默认 0,1。
#
# 环境变量说明：
#   H1_GPU_POLICY：缓存策略，默认 h1_lru。
#   H1_GPU_MEMORY_UTILIZATION：vLLM gpu_memory_utilization；会映射为 tight/mid/loose 或读取 H1_BUDGET。
#   H1_BUDGET：当 H1_GPU_MEMORY_UTILIZATION 不是内置映射值时使用的 budget 名。
#   H1_MODEL：本地模型路径。
#   H1_REPLAY_TRACE：JSONL replay trace 输入路径。
#   H1_HOTPOTQA_PATH：本地 HotpotQA 数据目录。
#   H1_BENCH_NUM_PROMPTS：回放请求数。
#   H1_REPLAY_BATCH_SIZE / H1_BENCH_MAX_CONCURRENCY：回放批大小，对应 vLLM max_num_seqs。
#   H1_MAX_MODEL_LEN：vLLM max_model_len。
#   H1_MAX_TOKENS：每个请求生成 token 上限。
#   H1_TENSOR_PARALLEL_SIZE：vLLM tensor parallel size。
#   H1_ATTENTION_BACKEND：vLLM attention backend。
#   EDGEKV_H1_PROFILE_POLICY_TIME：是否记录策略耗时。
#   EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES：是否输出对象级画像。
#   EDGEKV_H1_RUNTIME_MONITOR：是否启用 LPE 运行时监控。
#   EDGEKV_H1_RUNTIME_MONITOR_PATH：运行时监控 JSONL 输出路径。
#   EDGEKV_C_RE_MS_PER_TOKEN / EDGEKV_BW_GBPS / EDGEKV_D_DESER_MS：COP 成本模型参数。
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${1:-h1/out/h1_policy_pressure_replay_$(date +%Y%m%d_%H%M%S)}"
VISIBLE_DEVICES="${2:-${H1_VISIBLE_DEVICES:-0,1}}"
POLICY="${H1_GPU_POLICY:-h1_lru}"
GPU_UTIL="${H1_GPU_MEMORY_UTILIZATION:-0.710}"
MODEL="${H1_MODEL:-models/Qwen2.5-7B-Instruct}"
REPLAY_TRACE="${H1_REPLAY_TRACE:-data/edgekv_traces/sharegpt_hotpotqa_session.jsonl}"
HOTPOTQA_PATH="${H1_HOTPOTQA_PATH:-data/hotpotqa}"
NUM_PROMPTS="${H1_BENCH_NUM_PROMPTS:-128}"
REPLAY_BATCH_SIZE="${H1_REPLAY_BATCH_SIZE:-${H1_BENCH_MAX_CONCURRENCY:-1}}"
MAX_MODEL_LEN="${H1_MAX_MODEL_LEN:-2048}"
MAX_TOKENS="${H1_MAX_TOKENS:-16}"
TENSOR_PARALLEL_SIZE="${H1_TENSOR_PARALLEL_SIZE:-2}"
ATTENTION_BACKEND="${H1_ATTENTION_BACKEND:-TRITON_ATTN}"
EDGEKV_H1_PROFILE_POLICY_TIME="${EDGEKV_H1_PROFILE_POLICY_TIME:-1}"
EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES="${EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES:-}"
EDGEKV_H1_RUNTIME_MONITOR="${EDGEKV_H1_RUNTIME_MONITOR:-}"
EDGEKV_H1_RUNTIME_MONITOR_PATH="${EDGEKV_H1_RUNTIME_MONITOR_PATH:-$OUT_DIR/runtime_monitor.jsonl}"

if [[ "$POLICY" == "h1_lpe" ]]; then
  EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES="${EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES:-1}"
  EDGEKV_H1_RUNTIME_MONITOR="${EDGEKV_H1_RUNTIME_MONITOR:-1}"
else
  EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES="${EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES:-0}"
  EDGEKV_H1_RUNTIME_MONITOR="${EDGEKV_H1_RUNTIME_MONITOR:-0}"
fi

case "$GPU_UTIL" in
  0.710) BUDGET="tight" ;;
  0.735) BUDGET="mid" ;;
  0.774) BUDGET="loose" ;;
  *) BUDGET="${H1_BUDGET:-tight}" ;;
esac

mkdir -p "$OUT_DIR"
MAIN_LOG="$OUT_DIR/run.log"
: > "$MAIN_LOG"

echo "[run] pressure replay wrapper out=$OUT_DIR" | tee -a "$MAIN_LOG"
echo "[run] policy=$POLICY budget=$BUDGET trace=$REPLAY_TRACE prompts=$NUM_PROMPTS batch=$REPLAY_BATCH_SIZE devices=$VISIBLE_DEVICES" | tee -a "$MAIN_LOG"

PYTHONPATH=.:h1:h0 \
CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" \
VLLM_USE_V1=1 \
VLLM_ATTENTION_BACKEND="$ATTENTION_BACKEND" \
VLLM_NO_USAGE_STATS=1 \
EDGEKV_H1_GPU_POLICY="$POLICY" \
EDGEKV_H1_STATS_DIR="$OUT_DIR/edgekv_gpu_stats" \
EDGEKV_H1_PROFILE_POLICY_TIME="$EDGEKV_H1_PROFILE_POLICY_TIME" \
EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES="$EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES" \
EDGEKV_H1_RUNTIME_MONITOR="$EDGEKV_H1_RUNTIME_MONITOR" \
EDGEKV_H1_RUNTIME_MONITOR_PATH="$EDGEKV_H1_RUNTIME_MONITOR_PATH" \
EDGEKV_C_RE_MS_PER_TOKEN="${EDGEKV_C_RE_MS_PER_TOKEN:-0.12}" \
EDGEKV_BW_GBPS="${EDGEKV_BW_GBPS:-1.0}" \
EDGEKV_D_DESER_MS="${EDGEKV_D_DESER_MS:-3.0}" \
conda run --no-capture-output -n edgekv-vllm0110 \
  python h1/run_h1_vllm0110_real.py \
    --out "$OUT_DIR" \
    --model "$MODEL" \
    --dtype float16 \
    --policies "$POLICY" \
    --budgets "$BUDGET" \
    --workload mixed \
    --replay-trace "$REPLAY_TRACE" \
    --hotpotqa-path "$HOTPOTQA_PATH" \
    --max-requests "$NUM_PROMPTS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --replay-batch-size "$REPLAY_BATCH_SIZE" \
    --visible-devices "$VISIBLE_DEVICES" \
  2>&1 | tee -a "$MAIN_LOG"

echo "[run] summary: $OUT_DIR/h1_vllm_real_summary.csv" | tee -a "$MAIN_LOG"
