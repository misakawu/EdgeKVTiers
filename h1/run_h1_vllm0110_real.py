#!/usr/bin/env python3
"""在真实 vLLM 0.11.0 GPU 前缀缓存上运行 H1 0.5.1 实验。

实验形态来自 00_预实验提取 0.5.1：
- 四种策略：LRU、LFU、vLLM 默认、LPE-score。
- 三档显存预算。
- 默认使用同一条 H0 ShareGPT + HotpotQA RAG 混合 replay trace。
- 记录 p95 TTFT 代理指标、命中率和 GPU 显存峰值。

说明：
- H1 策略通过仓库内的 ``h1/sitecustomize.py`` 运行时补丁接入 vLLM v1
  GPU 前缀缓存块驱逐流程。实验只使用 GPU KV cache，不配置 vLLM CPU KV
  offload。
- vLLM offline ``LLM.generate`` 在可用时记录请求级调度指标。
  ``ttft_proxy_ms`` 优先使用 ``RequestOutput.metrics`` 中的请求级首 token
  时间；旧版本不可用时退回到批次 wall-clock 延迟代理。``queue_wait_ms``
  和 ``prefill_ms`` 也来自 ``RequestOutput.metrics``。
- summary 中的命中率来自真实 GPU 前缀缓存块 lookup 统计。请求级行仍保留
  H0 trace 侧的复用代理字段。

启动命令：
    conda run --no-capture-output -n edgekv-vllm0110 python h1/run_h1_vllm0110_real.py

参数说明：
    --out：单次运行输出目录。
    --model：本地模型路径或 vLLM 可加载的模型名。
    --dtype：模型权重 dtype。
    --trace-path：ShareGPT 原始 JSON 路径；未使用 replay trace 时生成 ShareGPT workload。
    --replay-trace：JSONL replay trace 输入路径。
    --workload：回放类型，sharegpt/rag/mixed。
    --hotpotqa-path：本地 HotpotQA 数据目录。
    --download-hotpotqa：缺少 HotpotQA parquet 时允许下载。
    --hotpotqa-max-examples：加载 HotpotQA 的最大样本数。
    --rag-requests：mixed/rag workload 中 RAG 请求数。
    --rag-chunk-words：每个 RAG chunk 的词数。
    --rag-chunks-per-query：每个 RAG query 拼接的 chunk 数。
    --rag-query-repeats：每组 RAG query 重复次数。
    --sharegpt-order：ShareGPT session 选择顺序，file 或 longest。
    --timeout-s：请求/下载相关操作超时时间。
    --policies：要运行的缓存策略列表。
    --budgets：预算档位，可用命名档或数值 gpu_memory_utilization。
    --max-sessions：最多加载的 ShareGPT session 数。
    --max-requests：最多回放请求数。
    --tensor-parallel-size：vLLM tensor parallel size。
    --max-model-len：vLLM max_model_len。
    --max-tokens：每个请求生成 token 上限。
    --replay-batch-size：回放批大小，对应 vLLM max_num_seqs。
    --batch-order：批内请求排序方式，original/length_bucket/round_robin。
    --warmup-batches：正式计量前跳过的 warmup batch 数。
    --max-num-batched-tokens：vLLM scheduler/profile token 上限。
    --c-re-ms-per-token：COP 重算成本估计，单位 ms/token。
    --bw-gbps：COP restore 带宽估计，单位 GB/s。
    --d-deser-ms：COP 反序列化固定开销，单位 ms。
    --visible-devices：设置 CUDA_VISIBLE_DEVICES。
    --stats-dir：EdgeKV GPU stats 输出目录；为空时使用 cell 输出目录下的默认位置。
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from edgekv_cop import COPProfiler, DEFAULT_BW_GBPS, DEFAULT_C_RE_MS_PER_TOKEN, DEFAULT_D_DESER_MS

REPO_ROOT = Path(__file__).resolve().parents[1]
H0_DIR = REPO_ROOT / 'h0'
if str(H0_DIR) not in sys.path:
    sys.path.insert(0, str(H0_DIR))
DEFAULT_SHAREGPT_TRACE_PATH = (
    REPO_ROOT / 'data' / 'ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
)
DEFAULT_HOTPOTQA_PATH = REPO_ROOT / 'data' / 'hotpotqa'
DEFAULT_REPLAY_TRACE_PATH = (
    REPO_ROOT / 'data' / 'edgekv_traces' / 'sharegpt_hotpotqa_session.jsonl'
)

from run_h0_vllm import (
    kv_size_mib_per_token,
    load_model_config,
    load_replay_prompts,
)

from transformers import AutoTokenizer, PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens
    )

from vllm import LLM, SamplingParams


POLICIES = ('vllm_default', 'h1_lru', 'h1_lfu', 'h1_lpe')
BUDGET_GPU_MEMORY_UTILIZATION = {
    'super_tight': 0.710,  # 极端/饱和预算（等于旧 tight 值）
    'tight': 0.720,
    'mid': 0.735,
    'loose': 0.774,
}
MODEL_PRECISION_BITS = 16
MODEL_MEMORY_HEADROOM = 1.2
DEFAULT_MODEL_PARAM_BYTES = 2.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return values[lo]
    w = pos - lo
    return values[lo] * (1.0 - w) + values[hi] * w


def request_output_timing_ms(output: Any) -> dict[str, float | str | bool]:
    """提取 vLLM RequestMetrics 时间字段，缺失时回退为 0。

    vLLM 以秒保存时间戳/耗时。vLLM 0.11 在 ``RequestMetrics`` 上使用
    ``first_scheduled_time`` 和 ``time_in_queue``；仓库内 sitecustomize 补丁
    也可能暴露 v1 内部的 ``queued_ts/scheduled_ts``。这里保持防御式解析，
    让新旧版本都能输出 CSV 列，而不是中断实验。
    """
    metrics = getattr(output, 'metrics', None)
    if metrics is None:
        return {
            'queue_wait_ms': 0.0,
            'prefill_ms': 0.0,
            'ttft_ms': 0.0,
            'd4_metrics_available': False,
            'd4_metrics_source': 'request_output.metrics_missing',
        }

    def ts(name: str) -> float | None:
        value = getattr(metrics, name, None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    arrival = ts('arrival_time')
    queued = ts('queued_ts')
    scheduled = ts('first_scheduled_time')
    if scheduled is None:
        scheduled = ts('scheduled_time')
    if scheduled is None:
        scheduled = ts('scheduled_ts')
    first_token = ts('first_token_time')
    if first_token is None:
        first_token = ts('first_token_ts')
    time_in_queue = ts('time_in_queue')

    if time_in_queue is not None:
        queue_wait_ms = max(0.0, time_in_queue * 1000.0)
    elif queued is not None and scheduled is not None:
        queue_wait_ms = max(0.0, (scheduled - queued) * 1000.0)
    elif arrival is not None and scheduled is not None:
        queue_wait_ms = max(0.0, (scheduled - arrival) * 1000.0)
    else:
        queue_wait_ms = 0.0
    prefill_ms = (
        max(0.0, (first_token - scheduled) * 1000.0)
        if scheduled is not None and first_token is not None
        else 0.0
    )
    ttft_ms = 0.0
    if arrival is not None and first_token is not None:
        ttft_ms = max(0.0, (first_token - arrival) * 1000.0)
    if ttft_ms <= 0.0 and queued is not None and first_token is not None:
        ttft_ms = max(0.0, (first_token - queued) * 1000.0)
    if ttft_ms <= 0.0 and time_in_queue is not None and prefill_ms > 0.0:
        ttft_ms = max(0.0, time_in_queue * 1000.0) + prefill_ms
    if ttft_ms <= 0.0 and scheduled is not None and first_token is not None:
        ttft_ms = max(0.0, (first_token - scheduled) * 1000.0)
    available = (
        (time_in_queue is not None or (queued is not None and scheduled is not None)
         or (arrival is not None and scheduled is not None))
        and scheduled is not None
        and first_token is not None
    )
    missing = [
        name
        for name, value in (
            ('queue_wait_source', time_in_queue if time_in_queue is not None else queued if queued is not None else arrival),
            ('first_scheduled_time', scheduled),
            ('first_token_time', first_token),
        )
        if value is None
    ]
    return {
        'queue_wait_ms': queue_wait_ms,
        'prefill_ms': prefill_ms,
        'ttft_ms': ttft_ms,
        'd4_metrics_available': available,
        'd4_metrics_source': (
            'request_output.metrics'
            if available
            else f"request_output.metrics_missing:{','.join(missing)}"
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write('\n')


def estimate_tokens(text: str) -> int:
    return max(1, int(len((text or '').split()) * 1.35))


def count_tokens(text: str, tokenizer: Any) -> int:
    if tokenizer is None:
        return estimate_tokens(text)
    return len(tokenizer.encode(text, add_special_tokens=False))


def message_role(message: dict[str, Any]) -> str:
    return str(message.get('from', message.get('role', ''))).lower()


def message_text(message: dict[str, Any]) -> str:
    value = message.get('value', message.get('content', ''))
    return value if isinstance(value, str) else str(value)


def load_sharegpt_prompts(
    path: Path,
    max_sessions: int,
    max_requests: int,
    tokenizer: Any = None,
) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        rows = json.load(f)

    prompts: list[dict[str, Any]] = []
    sessions = 0
    for raw_session_idx, row in enumerate(rows):
        conversations = row.get('conversations', [])
        if not isinstance(conversations, list):
            continue
        human_count = sum(
            1 for msg in conversations if message_role(msg) in {'human', 'user'}
        )
        if human_count < 2:
            continue

        session_id = str(row.get('id', f'sharegpt_{raw_session_idx:06d}'))
        transcript: list[str] = []
        human_turn = 0
        for msg in conversations:
            role = message_role(msg)
            text = message_text(msg).strip()
            if not text:
                continue
            if role in {'human', 'user'}:
                prompt = '\n'.join(transcript + [f'User: {text}', 'Assistant:'])
                n_tokens = count_tokens(prompt, tokenizer)
                prompts.append(
                    {
                        'request_id': f'{session_id}:turn:{human_turn:03d}',
                        'session_id': session_id,
                        'turn_index': human_turn,
                        'prompt': prompt,
                        'prompt_chars': len(prompt),
                        'prompt_est_tokens': estimate_tokens(prompt),
                        'n_tokens': n_tokens,
                    }
                )
                transcript.append(f'User: {text}')
                human_turn += 1
                if len(prompts) >= max_requests:
                    return prompts
            elif role in {'gpt', 'assistant'}:
                transcript.append(f'Assistant: {text}')
        sessions += 1
        if sessions >= max_sessions:
            break
    return prompts[:max_requests]


def model_parameter_count(model: str) -> float:
    model_path = Path(model)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    index_path = model_path / 'model.safetensors.index.json'
    if not index_path.exists():
        raise FileNotFoundError(
            f'cannot derive model parameter count: missing {index_path}'
        )
    with index_path.open(encoding='utf-8') as f:
        index = json.load(f)
    total_size = float(index.get('metadata', {}).get('total_size', 0.0))
    if total_size <= 0:
        raise ValueError(f'cannot derive model parameter count from {index_path}')
    return total_size / DEFAULT_MODEL_PARAM_BYTES


def visible_gpu_total_mib(visible_devices: str) -> float:
    ids = [item.strip() for item in visible_devices.split(',') if item.strip()]
    if not ids:
        raise ValueError('visible_devices must contain at least one GPU id')
    proc = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=memory.total',
            '--format=csv,noheader,nounits',
            '-i',
            ','.join(ids),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            'failed to query visible GPU memory with nvidia-smi: '
            f'{proc.stderr.strip()}'
        )
    totals = [float(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
    if len(totals) != len(ids):
        raise RuntimeError(
            f'nvidia-smi returned {len(totals)} GPU memory rows for {len(ids)} visible devices'
        )
    return min(totals)


def resolve_budget(
    args: argparse.Namespace,
    budget_name: str,
    param_count: float,
    per_gpu_total_mib: float,
) -> dict[str, Any]:
    base_bytes = param_count * (MODEL_PRECISION_BITS / 8.0) * MODEL_MEMORY_HEADROOM
    if budget_name in BUDGET_GPU_MEMORY_UTILIZATION:
        gpu_memory_utilization = BUDGET_GPU_MEMORY_UTILIZATION[budget_name]
    else:
        try:
            gpu_memory_utilization = float(budget_name)
        except ValueError as exc:
            known = ', '.join(sorted(BUDGET_GPU_MEMORY_UTILIZATION))
            raise ValueError(
                f'unknown budget={budget_name!r}; use one of {{{known}}} or a '
                'numeric gpu_memory_utilization such as 0.60'
            ) from exc
    per_gpu_target_mib = gpu_memory_utilization * per_gpu_total_mib
    target_mib = per_gpu_target_mib * max(args.tensor_parallel_size, 1)
    if not 0.0 < gpu_memory_utilization < 1.0:
        raise ValueError(
            f'budget={budget_name} resolves to gpu_memory_utilization='
            f'{gpu_memory_utilization:.6f}; target={target_mib:.3f} MiB, '
            f'per_gpu_target={per_gpu_target_mib:.3f} MiB, '
            f'per_gpu_total={per_gpu_total_mib:.3f} MiB'
        )
    return {
        'budget': budget_name,
        'model_precision_bits': MODEL_PRECISION_BITS,
        'model_parameter_count': param_count,
        'model_memory_headroom': MODEL_MEMORY_HEADROOM,
        'base_model_memory_mib': base_bytes / (1024.0 * 1024.0),
        'target_memory_mib': target_mib,
        'per_gpu_target_memory_mib': per_gpu_target_mib,
        'per_gpu_total_memory_mib': per_gpu_total_mib,
        'gpu_memory_utilization': gpu_memory_utilization,
    }


class GpuMemoryMonitor:
    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        'nvidia-smi',
                        '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader,nounits',
                    ],
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                ts = time.time()
                for line in proc.stdout.splitlines():
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) != 4:
                        continue
                    self.samples.append(
                        {
                            'ts': ts,
                            'gpu_index': int(parts[0]),
                            'memory_used_mib': float(parts[1]),
                            'memory_total_mib': float(parts[2]),
                            'gpu_util_pct': float(parts[3]),
                        }
                    )
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def peak_mib(self, visible_devices: str | None = None) -> float:
        if not self.samples:
            return 0.0
        if not visible_devices:
            return max(float(row['memory_used_mib']) for row in self.samples)
        ids = {int(x) for x in visible_devices.split(',') if x.strip().isdigit()}
        rows = [row for row in self.samples if int(row['gpu_index']) in ids]
        return max((float(row['memory_used_mib']) for row in rows), default=0.0)


def load_trace(args: argparse.Namespace) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = load_replay_prompts(
        args,
        Path(args.trace_path),
        tokenizer=tokenizer,
    )
    filtered: list[dict[str, Any]] = []
    for original_index, item in enumerate(prompts):
        n_tokens = int(item['n_tokens'])
        if n_tokens + args.max_tokens <= args.max_model_len:
            item = dict(item)
            item['n_tokens'] = n_tokens
            item['_trace_original_index'] = original_index
            filtered.append(item)
        if len(filtered) >= args.max_requests:
            break
    return filtered


def item_token_count(item: dict[str, Any]) -> int:
    for key in ('n_tokens', 'prompt_est_tokens'):
        try:
            value = int(item.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return estimate_tokens(str(item.get('prompt', '')))


def length_bucket_id(n_tokens: int) -> int:
    """按 2 的幂范围生成粗粒度 token 长度桶。"""
    return max(1, int(n_tokens)).bit_length() - 1


def order_trace_for_batches(
    trace: list[dict[str, Any]],
    batch_order: str,
    replay_batch_size: int | None = None,
) -> list[dict[str, Any]]:
    if batch_order == 'original':
        return list(trace)
    if batch_order == 'length_bucket':
        indexed = [
            (length_bucket_id(item_token_count(item)), idx, item)
            for idx, item in enumerate(trace)
        ]
        indexed.sort(key=lambda row: (row[0], row[1]))
        return [item for _, _, item in indexed]
    if batch_order == 'round_robin':
        window_size = (
            replay_batch_size
            if replay_batch_size and replay_batch_size > 0
            else max(1, len(trace))
        )
        sessions: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        session_order: list[str] = []
        orphans: list[dict[str, Any]] = []
        for idx, item in enumerate(trace):
            session_id = str(item.get('session_id', '') or '')
            if not session_id:
                orphans.append(item)
                continue
            if session_id not in sessions:
                sessions[session_id] = []
                session_order.append(session_id)
            try:
                original_index = int(item.get('_trace_original_index', idx))
            except (TypeError, ValueError):
                original_index = idx
            sessions[session_id].append((original_index, item))

        ordered_sessions = [
            [item for _, item in sorted(sessions[session_id], key=lambda row: row[0])]
            for session_id in session_order
        ]
        ordered: list[dict[str, Any]] = []
        for window_start in range(0, len(ordered_sessions), window_size):
            window = ordered_sessions[window_start:window_start + window_size]
            offset = 0
            while True:
                emitted = False
                for session_items in window:
                    if offset < len(session_items):
                        ordered.append(session_items[offset])
                        emitted = True
                if not emitted:
                    break
                offset += 1
        ordered.extend(orphans)
        return ordered
    raise ValueError(f'unknown batch_order={batch_order!r}')


def replay_batches(trace: list[dict[str, Any]], replay_batch_size: int) -> list[tuple[int, list[dict[str, Any]]]]:
    batches: list[tuple[int, list[dict[str, Any]]]] = []
    current: list[dict[str, Any]] = []
    current_start = 0
    current_sessions: set[str] = set()
    for index, item in enumerate(trace):
        session_id = str(item.get('session_id', ''))
        must_split = bool(session_id and session_id in current_sessions)
        if current and (len(current) >= replay_batch_size or must_split):
            batches.append((current_start, current))
            current = []
            current_sessions = set()
            current_start = index
        if not current:
            current_start = index
        current.append(item)
        if session_id:
            current_sessions.add(session_id)
    if current:
        batches.append((current_start, current))
    return batches


def llm_generate_no_tqdm(llm: LLM, prompts: list[Any], sampling: list[SamplingParams]) -> Any:
    generate = llm.generate
    try:
        if 'use_tqdm' in inspect.signature(generate).parameters:
            return generate(prompts, sampling, use_tqdm=False)
    except (TypeError, ValueError):
        pass
    try:
        return generate(prompts, sampling, use_tqdm=False)
    except TypeError:
        return generate(prompts, sampling)


def request_trace_fields(
    item: dict[str, Any],
    policy: str,
    kv_mib_per_token: float,
    cop: COPProfiler,
    event_index: int,
) -> dict[str, Any]:
    """产出对象成本列（COP）与引擎 meta 所需字段。

    **不再产出任何 trace-side 模拟命中**：命中/复用统计只来自回放器真实执行
    （sitecustomize 的 native_hits/native_queries）。COP 仅用于对象成本估计
    （c_recomp/c_restore/risk_exp/size/object_id）与 p_reuse_prior seed；
    update_from_item 的 hit 恒传 False，在线 p_reuse 由引擎从真实访问自算。
    """
    workload = str(item.get('workload', 'sharegpt_session_prefix'))
    reuse_key = str(item.get('reuse_key', item.get('session_id', '')))
    profile = cop.update_from_item(
        item,
        hit=False,
        access_index=event_index,
    )
    fields = {k: v for k, v in item.items() if k not in {'prompt', '_trace_original_index'}}
    fields.update(
        {
            'workload': workload,
            'object_type': profile.object_type,
            'object_id': profile.object_id,
            'reuse_key': reuse_key,
            'p_reuse_prior': (
                round(float(profile.p_reuse_prior), 6)
                if profile.p_reuse_prior is not None else item.get('p_reuse_prior', '')
            ),
            'c_recomp_ms': round(profile.c_recomp_ms, 6),
            'c_restore_ms': round(profile.c_restore_ms, 6),
            'risk_exp': round(profile.risk_exp, 6),
            'score_source': 'object_level_cop',
            'lpe_action': 'score_evaluated' if policy == 'h1_lpe' else '',
            'size_mb': round(profile.size_mb, 6),
            'size_scope': 'per_gpu_logical_full_kv_estimate',
            'kv_mib_per_token': round(kv_mib_per_token, 9),
            'trace_original_index': item.get('_trace_original_index', event_index),
        }
    )
    return fields


def resolved_max_num_batched_tokens(args: argparse.Namespace) -> int:
    upper_bound = args.max_model_len * args.replay_batch_size
    default_value = min(8192, upper_bound)
    value = args.max_num_batched_tokens
    if value is None:
        return default_value
    if value < args.replay_batch_size:
        raise ValueError(
            'max_num_batched_tokens must be >= replay_batch_size '
            f'({value} < {args.replay_batch_size})'
        )
    if value > upper_bound:
        raise ValueError(
            'max_num_batched_tokens must be <= max_model_len * replay_batch_size '
            f'({value} > {upper_bound})'
        )
    return value


def build_llm(args: argparse.Namespace, policy: str, gpu_memory_utilization: float) -> LLM:
    os.environ['EDGEKV_H1_GPU_POLICY'] = policy
    return LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        max_num_seqs=args.replay_batch_size,
        max_num_batched_tokens=resolved_max_num_batched_tokens(args),
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        swap_space=0,
        cpu_offload_gb=0,
        disable_log_stats=False,
    )


def release_llm(llm: LLM | None) -> None:
    if llm is not None:
        shutdown = getattr(llm, 'shutdown', None)
        if callable(shutdown):
            shutdown()
        del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


def reset_after_warmup(llm: LLM, stats_dir: Path) -> None:
    reset_prefix_cache = getattr(llm, 'reset_prefix_cache', None)
    if callable(reset_prefix_cache):
        try:
            reset_prefix_cache()
        except Exception:
            pass
    try:
        import sitecustomize

        reset_stats = getattr(sitecustomize, 'reset_edgekv_gpu_cache_stats', None)
        if callable(reset_stats):
            reset_stats()
    except Exception:
        pass
    if stats_dir.exists():
        shutil.rmtree(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)


def run_cell(
    args: argparse.Namespace,
    trace: list[dict[str, Any]],
    policy: str,
    budget_name: str,
    budget: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    llm: LLM | None = None
    rows: list[dict[str, Any]] = []
    batch_queue_spans: list[float] = []
    monitor = GpuMemoryMonitor()
    started = time.time()
    ok = False
    error = ''
    gpu_memory_utilization = float(budget['gpu_memory_utilization'])
    model_config = load_model_config(args.model)
    kv_mib_per_token = kv_size_mib_per_token(model_config, args.tensor_parallel_size)
    os.environ.setdefault('EDGEKV_H1_PROFILE_POLICY_TIME', '1')
    if policy == 'h1_lpe':
        os.environ.setdefault('EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES', '1')
        os.environ.setdefault('EDGEKV_H1_RUNTIME_MONITOR', '1')
        monitor_path = (
            out_dir / 'runtime_monitor.jsonl'
            if len(args.budgets) == 1 and len(args.policies) == 1
            else out_dir / f'{budget_name}_{policy}_runtime_monitor.jsonl'
        )
        os.environ.setdefault('EDGEKV_H1_RUNTIME_MONITOR_PATH', str(monitor_path))
    os.environ['EDGEKV_MU_KV_MB_PER_TOKEN'] = str(kv_mib_per_token)
    os.environ['EDGEKV_C_RE_MS_PER_TOKEN'] = str(args.c_re_ms_per_token)
    cop = COPProfiler(
        mu_kv_mb_per_token=kv_mib_per_token,
        c_re_ms_per_token=args.c_re_ms_per_token,
        bw_gbps=args.bw_gbps,
        d_deser_ms=args.d_deser_ms,
    )
    stats_dir = Path(args.stats_dir) / f'{budget_name}_{policy}'
    if stats_dir.exists():
        shutil.rmtree(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    os.environ['EDGEKV_H1_STATS_DIR'] = str(stats_dir)
    try:
        import sitecustomize

        patch_metrics = getattr(sitecustomize, 'patch_edgekv_vllm_request_metrics', None)
        if callable(patch_metrics):
            patch_metrics()
        reset_stats = getattr(sitecustomize, 'reset_edgekv_gpu_cache_stats', None)
        if callable(reset_stats):
            reset_stats()
    except Exception:
        pass
    try:
        monitor.start()
        llm = build_llm(args, policy, gpu_memory_utilization)
        replay_trace = order_trace_for_batches(
            trace,
            args.batch_order,
            args.replay_batch_size,
        )
        batches = replay_batches(replay_trace, args.replay_batch_size)
        print(
            f"[cell] policy={policy} budget={budget_name} requests={len(replay_trace)} "
            f"batch_size={args.replay_batch_size} batches={len(batches)} trace={args.replay_trace}",
            flush=True,
        )
        if args.warmup_batches > 0 and replay_trace:
            warmup_count = min(len(replay_trace), args.replay_batch_size, 8)
            warmup_prompts = [
                f'Warmup request {i}. Reply with one short sentence.'
                for i in range(warmup_count)
            ]
            warmup_sampling = [
                SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
                for _ in warmup_prompts
            ]
            for _ in range(args.warmup_batches):
                llm_generate_no_tqdm(llm, warmup_prompts, warmup_sampling)
            reset_after_warmup(llm, stats_dir)
        progress_every = max(1, len(batches) // 10) if batches else 1
        for batch_number, (batch_start, batch) in enumerate(batches, start=1):
            if batch_number == 1 or batch_number == len(batches) or batch_number % progress_every == 0:
                print(
                    f"[replay] policy={policy} budget={budget_name} "
                    f"batch={batch_number}/{len(batches)} done_requests={len(rows)}",
                    flush=True,
                )
            prompts = [str(item['prompt']) for item in batch]
            trace_fields_batch = [
                request_trace_fields(
                    item,
                    policy,
                    kv_mib_per_token,
                    cop,
                    batch_start + offset,
                )
                for offset, item in enumerate(batch)
            ]
            sampling = [
                SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    extra_args={
                        'edgekv_h1': {
                            'policy': policy,
                            'request_id': trace_fields.get('request_id', ''),
                            'object_id': trace_fields.get('object_id', ''),
                            'reuse_key': trace_fields.get('reuse_key', ''),
                            'object_type': trace_fields.get('object_type', ''),
                            'workload': trace_fields.get('workload', ''),
                            'n_tokens': trace_fields.get('n_tokens', 0),
                            # 不再传伪造的 p_reuse/score：只传 p_reuse_prior 作先验 seed，
                            # 在线 p_reuse 与 score 由引擎从真实访问/命中自算。
                            'p_reuse_prior': trace_fields.get('p_reuse_prior', ''),
                            'temperature': trace_fields.get('temperature', ''),
                            'c_recomp_ms': trace_fields.get('c_recomp_ms', 0.0),
                            'c_restore_ms': trace_fields.get('c_restore_ms', 0.0),
                            'risk_exp': trace_fields.get('risk_exp', 0.0),
                            'size_mb': trace_fields.get('size_mb', 0.0),
                        }
                    },
                )
                for trace_fields in trace_fields_batch
            ]
            t0 = time.perf_counter()
            outputs = llm_generate_no_tqdm(llm, prompts, sampling)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            batch_rows: list[dict[str, Any]] = []
            for offset, (item, output, trace_fields) in enumerate(
                zip(batch, outputs, trace_fields_batch)
            ):
                text = output.outputs[0].text
                timing = request_output_timing_ms(output)
                timing_ttft_ms = float(timing.get('ttft_ms', 0.0) or 0.0)
                metrics_available = bool(timing['d4_metrics_available'])
                ttft_proxy_ms = (
                    timing_ttft_ms
                    if metrics_available and timing_ttft_ms > 0.0
                    else latency_ms
                )
                idx = batch_start + offset
                row = {
                    **trace_fields,
                    'experiment': 'H1_vLLM_real_0_5_1',
                    'policy': policy,
                    'budget': budget_name,
                    'gpu_memory_utilization': gpu_memory_utilization,
                    'model_precision_bits': budget['model_precision_bits'],
                    'budget_target_memory_mib': round(float(budget['target_memory_mib']), 6),
                    'budget_per_gpu_target_memory_mib': round(float(budget['per_gpu_target_memory_mib']), 6),
                    'event_index': idx,
                    'batch_start_index': batch_start,
                    'batch_order': args.batch_order,
                    'replay_batch_size': args.replay_batch_size,
                    'max_tokens': args.max_tokens,
                    'latency_ms': round(latency_ms, 6),
                    'ttft_proxy_ms': round(ttft_proxy_ms, 6),
                    'queue_wait_ms': round(float(timing['queue_wait_ms']), 6),
                    'prefill_ms': round(float(timing['prefill_ms']), 6),
                    'd4_metrics_available': metrics_available,
                    'd4_metrics_source': str(timing['d4_metrics_source']),
                    'output_chars': len(text),
                    'model': args.model,
                    'tensor_parallel_size': args.tensor_parallel_size,
                    'kv_cache_location': 'gpu',
                }
                if item.get('rag_reuse_key'):
                    row['rag_reuse_key'] = str(item.get('rag_reuse_key', ''))
                batch_rows.append(row)
            batch_queue_values = [
                float(row.get('queue_wait_ms', 0.0) or 0.0)
                for row in batch_rows
                if row.get('d4_metrics_available')
            ]
            batch_queue_span_ms = (
                max(batch_queue_values) - min(batch_queue_values)
                if batch_queue_values else 0.0
            )
            batch_queue_spans.append(batch_queue_span_ms)
            for row in batch_rows:
                row['batch_queue_span_ms'] = round(batch_queue_span_ms, 6)
                rows.append(row)
        ok = True
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        rows.append(
            {
                'experiment': 'H1_vLLM_real_0_5_1',
                'policy': policy,
                'budget': budget_name,
                'ok': False,
                'error': error,
            }
        )
    finally:
        monitor.stop()
        release_llm(llm)

    valid = [row for row in rows if 'ttft_proxy_ms' in row]
    ttft_values = [float(row['ttft_proxy_ms']) for row in valid]
    d4_metrics_available_count = sum(1 for row in valid if row.get('d4_metrics_available'))
    d4_valid = [row for row in valid if row.get('d4_metrics_available')]
    queue_wait_values = [float(row.get('queue_wait_ms', 0.0) or 0.0) for row in d4_valid]
    prefill_values = [float(row.get('prefill_ms', 0.0) or 0.0) for row in d4_valid]
    queue_wait_ratio_values = [
        float(row.get('queue_wait_ms', 0.0) or 0.0)
        / float(row.get('ttft_proxy_ms', 0.0) or 0.0)
        for row in d4_valid
        if float(row.get('ttft_proxy_ms', 0.0) or 0.0) > 0.0
    ]
    d4_queue_wait_p95_ms = percentile(queue_wait_values, 95)
    d4_prefill_p95_ms = percentile(prefill_values, 95)
    batch_queue_span_p95_ms = percentile(batch_queue_spans, 95)
    d4_ttft_p95_ms = percentile(ttft_values, 95)
    gpu_cache_stats: dict[str, Any] = {}
    try:
        import sitecustomize

        get_stats = getattr(sitecustomize, 'get_edgekv_gpu_cache_stats', None)
        if callable(get_stats):
            gpu_cache_stats = dict(get_stats())
    except Exception:
        gpu_cache_stats = {}
    gpu_cache_stats = merge_gpu_cache_stats(gpu_cache_stats, stats_dir)
    gpu_lookup_total = int(gpu_cache_stats.get('lookup_total', 0) or 0)
    gpu_lookup_hits = int(gpu_cache_stats.get('lookup_hits', 0) or 0)
    gpu_hit_rate = float(gpu_cache_stats.get('hit_rate', 0.0) or 0.0)
    workload_counts: dict[str, int] = {}
    replay_trace_formats = sorted({str(row.get('replay_trace_format', 'structured_conversation_v2')) for row in valid})
    history_formats = sorted({str(row.get('history_format', 'structured_conversation_v2')) for row in valid})
    structured_sessions = len({
        str(row.get('session_id', ''))
        for row in valid
        if row.get('session_id')
    })
    structured_turns = len(valid)
    for row in valid:
        workload = str(row.get('workload', 'unknown'))
        workload_counts[workload] = workload_counts.get(workload, 0) + 1
    summary = {
        'experiment': 'H1_vLLM_real_0_5_1',
        'policy': policy,
        'budget': budget_name,
        'gpu_memory_utilization': gpu_memory_utilization,
        'model_precision_bits': budget['model_precision_bits'],
        'budget_base_model_memory_mib': round(float(budget['base_model_memory_mib']), 6),
        'budget_target_memory_mib': round(float(budget['target_memory_mib']), 6),
        'budget_per_gpu_target_memory_mib': round(float(budget['per_gpu_target_memory_mib']), 6),
        'budget_per_gpu_total_memory_mib': round(float(budget['per_gpu_total_memory_mib']), 6),
        'model_parameter_count': round(float(budget['model_parameter_count']), 6),
        'model_memory_headroom': budget['model_memory_headroom'],
        'ok': ok,
        'error': error,
        'requests': len(valid),
        'sessions_requested': args.max_sessions,
        'rag_requests_requested': args.rag_requests,
        'trace_requests_available': len(trace),
        'replay_trace': args.replay_trace,
        'replay_trace_format': ",".join(replay_trace_formats) if replay_trace_formats else "unknown",
        'history_format': ",".join(history_formats) if history_formats else "",
        'structured_sessions': structured_sessions,
        'structured_turns': structured_turns,
        'vllm_generate_tqdm': False,
        'replay_progress_log': 'task_level',
        'workload': args.workload,
        'hotpotqa_source': str(args.hotpotqa_path),
        'rag_chunk_words': args.rag_chunk_words,
        'rag_chunks_per_query': args.rag_chunks_per_query,
        'rag_query_repeats': args.rag_query_repeats,
        'sharegpt_order': args.sharegpt_order,
        'hotpotqa_max_examples': args.hotpotqa_max_examples,
        'ttft_proxy_p50_ms': round(percentile(ttft_values, 50), 6),
        'ttft_proxy_p95_ms': round(d4_ttft_p95_ms, 6),
        'latency_p95_ms': round(d4_ttft_p95_ms, 6),
        'latency_mean_ms': round(statistics.mean(ttft_values), 6) if ttft_values else 0.0,
        'queue_wait_ms': round(statistics.mean(queue_wait_values), 6) if queue_wait_values else 0.0,
        'prefill_ms': round(statistics.mean(prefill_values), 6) if prefill_values else 0.0,
        'queue_wait_ratio_mean': round(statistics.mean(queue_wait_ratio_values), 6) if queue_wait_ratio_values else 0.0,
        'queue_wait_p95_ms': round(d4_queue_wait_p95_ms, 6),
        'prefill_p95_ms': round(d4_prefill_p95_ms, 6),
        'batch_queue_span_p95_ms': round(batch_queue_span_p95_ms, 6),
        'queue_wait_p95_ratio': round(d4_queue_wait_p95_ms / d4_ttft_p95_ms, 6) if d4_ttft_p95_ms else 0.0,
        'prefill_p95_ratio': round(d4_prefill_p95_ms / d4_ttft_p95_ms, 6) if d4_ttft_p95_ms else 0.0,
        'd4_metrics_available_count': d4_metrics_available_count,
        'd4_metrics_available_ratio': round(d4_metrics_available_count / max(len(valid), 1), 6),
        # 主命中率指向真实缓存命中口径（native_hits/native_queries），与 trace-side 模拟无关。
        'hit_rate': round(float(gpu_cache_stats.get('native_hit_rate', 0.0) or 0.0), 6),
        'hit_source': 'gpu_prefix_cache_native',
        'native_hit_rate': round(float(gpu_cache_stats.get('native_hit_rate', 0.0) or 0.0), 6),
        # 块级 lookup 命中率为附带口径（matched blocks / total blocks），仅供参考。
        'block_lookup_hit_rate': round(float(gpu_cache_stats.get('block_lookup_hit_rate', 0.0) or 0.0), 6),
        'gpu_block_lookup_hit_rate': round(gpu_hit_rate, 6),
        'gpu_prefix_cache_native_queries': int(gpu_cache_stats.get('native_queries', 0) or 0),
        'gpu_prefix_cache_native_hits': int(gpu_cache_stats.get('native_hits', 0) or 0),
        'gpu_prefix_cache_native_requests': int(gpu_cache_stats.get('native_requests', 0) or 0),
        'gpu_prefix_cache_lookup_total': gpu_lookup_total,
        'gpu_prefix_cache_lookup_hits': gpu_lookup_hits,
        'gpu_prefix_cache_lookup_misses': int(gpu_cache_stats.get('lookup_misses', 0) or 0),
        'gpu_prefix_cache_evictions': int(gpu_cache_stats.get('evictions', 0) or 0),
        'gpu_prefix_cache_cached_blocks': int(gpu_cache_stats.get('cached_blocks', 0) or 0),
        'gpu_prefix_cache_touches': int(gpu_cache_stats.get('touches', 0) or 0),
        'gpu_prefix_cache_queue_reorders': int(gpu_cache_stats.get('queue_reorders', 0) or 0),
        'free_queue_reorder_calls': int(gpu_cache_stats.get('free_queue_reorder_calls', 0) or 0),
        'free_queue_reorder_blocks': int(gpu_cache_stats.get('free_queue_reorder_blocks', 0) or 0),
        'free_queue_reorder_skipped': int(gpu_cache_stats.get('free_queue_reorder_skipped', 0) or 0),
        'free_queue_reorder_window': int(gpu_cache_stats.get('free_queue_reorder_window', 0) or 0),
        'free_queue_reorder_time_ms': round(float(gpu_cache_stats.get('free_queue_reorder_time_ms', 0.0) or 0.0), 6),
        'policy_time_us_avg': round(float(gpu_cache_stats.get('policy_time_us_avg', 0.0) or 0.0), 6),
        'eviction_decision_time_us_avg': round(
            float(gpu_cache_stats.get('eviction_decision_time_us_avg', 0.0) or 0.0),
            6,
        ),
        'avg_p_reuse': round(float(gpu_cache_stats.get('avg_p_reuse', 0.0) or 0.0), 6),
        'avg_score': round(float(gpu_cache_stats.get('avg_score', 0.0) or 0.0), 9),
        'score_p50': round(float(gpu_cache_stats.get('score_p50', 0.0) or 0.0), 9),
        'score_p95': round(float(gpu_cache_stats.get('score_p95', 0.0) or 0.0), 9),
        'score_std': round(float(gpu_cache_stats.get('score_std', 0.0) or 0.0), 9),
        'p_reuse_std': round(float(gpu_cache_stats.get('p_reuse_std', 0.0) or 0.0), 6),
        'c_recomp_ms_p50': round(float(gpu_cache_stats.get('c_recomp_ms_p50', 0.0) or 0.0), 6),
        'c_recomp_model': str(gpu_cache_stats.get('c_recomp_model', '')),
        'eviction_granularity': str(gpu_cache_stats.get('eviction_granularity', '')),
        'lpe_profile_count': int(gpu_cache_stats.get('lpe_profile_count', 0) or 0),
        'admission_count': int(gpu_cache_stats.get('admissions', 0) or 0),
        'admission_rejection_count': int(gpu_cache_stats.get('admission_rejections', 0) or 0),
        'eviction_count': int(gpu_cache_stats.get('evictions', 0) or 0),
        'high_reuse_eviction_count': int(gpu_cache_stats.get('evict_high_reuse', 0) or 0)
        + int(gpu_cache_stats.get('evict_offload', 0) or 0),
        'drop_count': int(gpu_cache_stats.get('evict_drop', 0) or 0),
        'evicted_score_avg': round(float(gpu_cache_stats.get('evicted_score_avg', 0.0) or 0.0), 9),
        'evicted_p_reuse_avg': round(float(gpu_cache_stats.get('evicted_p_reuse_avg', 0.0) or 0.0), 6),
        'low_score_evictions': int(gpu_cache_stats.get('low_score_evictions', 0) or 0),
        'hot_prefix_evictions': int(gpu_cache_stats.get('hot_prefix_evictions', 0) or 0),
        'gpu_prefix_cache_policy_impl': 'sitecustomize.BlockPool.free_block_queue',
        'hit_rate_source': 'real_gpu_prefix_cache_native',
        'workload_counts': workload_counts,
        'gpu_memory_peak_mib': round(monitor.peak_mib(args.visible_devices), 6),
        'elapsed_s': round(time.time() - started, 3),
        'model': args.model,
        'tensor_parallel_size': args.tensor_parallel_size,
        'kv_cache_location': 'gpu',
        'kv_mib_per_token': round(kv_mib_per_token, 9),
        'c_re_ms_per_token': round(args.c_re_ms_per_token, 9),
        'bw_gbps': round(args.bw_gbps, 9),
        'd_deser_ms': round(args.d_deser_ms, 9),
        **cop.summary(),
        'replay_batch_size': args.replay_batch_size,
        'batch_order': args.batch_order,
        'warmup_batches': args.warmup_batches,
        'max_num_batched_tokens': resolved_max_num_batched_tokens(args),
        'ttft_note': 'ttft_proxy_ms uses per-request RequestOutput.metrics first-token timing when available; if metrics are unavailable or non-positive it falls back to offline LLM.generate batch latency',
    }
    if valid:
        print(
            f"[cell-done] policy={policy} budget={budget_name} requests={len(valid)} "
            f"elapsed_s={summary['elapsed_s']} native_hit_rate={summary['native_hit_rate']} "
            f"block_lookup_hit_rate={summary['block_lookup_hit_rate']} "
            f"ttft_p95_ms={summary['ttft_proxy_p95_ms']} "
            f"gpu_memory_peak_mib={summary['gpu_memory_peak_mib']}",
            flush=True,
        )
    return rows, summary, monitor.samples


def plot_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    ok_rows = [row for row in summaries if row.get('ok')]
    if not ok_rows:
        return
    policies = list(dict.fromkeys(str(row['policy']) for row in ok_rows))
    budgets = list(dict.fromkeys(str(row['budget']) for row in ok_rows))
    plt.figure(figsize=(8, 4.8))
    for policy in policies:
        y = []
        x = []
        for budget in budgets:
            match = [row for row in ok_rows if row['policy'] == policy and row['budget'] == budget]
            if match:
                x.append(budget)
                y.append(float(match[0]['ttft_proxy_p95_ms']))
        plt.plot(x, y, marker='o', label=policy)
    plt.ylabel('p95 TTFT proxy / latency (ms)')
    plt.xlabel('GPU memory budget')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def merge_gpu_cache_stats(parent_stats: dict[str, Any], stats_dir: Path) -> dict[str, Any]:
    int_keys = {
        'lookup_hits',
        'lookup_misses',
        'native_queries',
        'native_hits',
        'native_requests',
        'touches',
        'cached_blocks',
        'evictions',
        'queue_reorders',
        'admissions',
        'admission_rejections',
        'evict_high_reuse',
        'evict_offload',
        'evict_drop',
        'free_queue_reorder_calls',
        'free_queue_reorder_blocks',
        'free_queue_reorder_skipped',
        'free_queue_reorder_window',
        'low_score_evictions',
        'hot_prefix_evictions',
        'policy_timing_samples',
        'eviction_decision_timing_samples',
        'lpe_profile_count',
    }
    sums = {key: int(parent_stats.get(key, 0) or 0) for key in int_keys}
    float_keys = {
        'policy_time_ms_total',
        'eviction_decision_time_ms_total',
        'eviction_decision_time_us_avg',
        'free_queue_reorder_time_ms',
        'score_std',
        'p_reuse_std',
        'c_recomp_ms_p50',
    }
    float_sums = {key: float(parent_stats.get(key, 0.0) or 0.0) for key in float_keys}
    profile_count = max(int(parent_stats.get('lpe_profile_count', 0) or 0), 0)
    weighted_p_reuse = float(parent_stats.get('avg_p_reuse', 0.0) or 0.0) * profile_count
    weighted_score = float(parent_stats.get('avg_score', 0.0) or 0.0) * profile_count
    evicted_score_count = int(parent_stats.get('evicted_score_count', 0) or 0)
    evicted_p_reuse_count = int(parent_stats.get('evicted_p_reuse_count', 0) or 0)
    weighted_evicted_score = float(parent_stats.get('evicted_score_avg', 0.0) or 0.0) * evicted_score_count
    weighted_evicted_p_reuse = float(parent_stats.get('evicted_p_reuse_avg', 0.0) or 0.0) * evicted_p_reuse_count
    stat_files = 0
    if stats_dir.exists():
        for path in stats_dir.glob('edgekv_gpu_stats_*.json'):
            try:
                row = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                continue
            stat_files += 1
            for key in int_keys:
                sums[key] += int(row.get(key, 0) or 0)
            for key in float_keys:
                float_sums[key] += float(row.get(key, 0.0) or 0.0)
            count = int(row.get('lpe_profile_count', 0) or 0)
            weighted_p_reuse += float(row.get('avg_p_reuse', 0.0) or 0.0) * count
            weighted_score += float(row.get('avg_score', 0.0) or 0.0) * count
            row_evicted_score_count = int(row.get('evicted_score_count', 0) or 0)
            row_evicted_p_reuse_count = int(row.get('evicted_p_reuse_count', 0) or 0)
            weighted_evicted_score += float(row.get('evicted_score_avg', 0.0) or 0.0) * row_evicted_score_count
            weighted_evicted_p_reuse += float(row.get('evicted_p_reuse_avg', 0.0) or 0.0) * row_evicted_p_reuse_count

    lookups = sums['lookup_hits'] + sums['lookup_misses']
    result: dict[str, Any] = dict(parent_stats)
    result.update(sums)
    result.update(float_sums)
    result['lookup_total'] = lookups
    # 块级前缀匹配长度效率（仅诊断）。
    result['block_lookup_hit_rate'] = (sums['lookup_hits'] / lookups) if lookups else 0.0
    # vLLM 原生 token 级覆盖率 = 真实缓存命中率（优先使用）。
    native_q = sums['native_queries']
    native_h = sums['native_hits']
    result['native_hit_rate'] = (native_h / native_q) if native_q else 0.0
    if native_q:
        result['hit_rate'] = result['native_hit_rate']
        result['hit_source'] = 'vllm_native_token_coverage'
    else:
        result['hit_rate'] = result['block_lookup_hit_rate']
        result['hit_source'] = 'gpu_prefix_cache_block_lookup'
    result['avg_p_reuse'] = weighted_p_reuse / sums['lpe_profile_count'] if sums['lpe_profile_count'] else 0.0
    result['avg_score'] = weighted_score / sums['lpe_profile_count'] if sums['lpe_profile_count'] else 0.0
    result['policy_time_us_avg'] = (
        (float_sums['policy_time_ms_total'] * 1000.0) / sums['policy_timing_samples']
        if sums.get('policy_timing_samples') else 0.0
    )
    result['eviction_decision_time_us_avg'] = (
        (float_sums['eviction_decision_time_ms_total'] * 1000.0)
        / sums['eviction_decision_timing_samples']
        if sums.get('eviction_decision_timing_samples') else 0.0
    )
    result['evicted_score_avg'] = (
        weighted_evicted_score / sums['evicted_score_count']
        if sums.get('evicted_score_count') else 0.0
    )
    result['evicted_p_reuse_avg'] = (
        weighted_evicted_p_reuse / sums['evicted_p_reuse_count']
        if sums.get('evicted_p_reuse_count') else 0.0
    )
    result['stats_files'] = stat_files
    result['stats_dir'] = str(stats_dir)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='h1/out/h1_vllm_real')
    parser.add_argument('--model', default='models/Qwen2.5-7B-Instruct')
    parser.add_argument('--dtype', default='float16')
    parser.add_argument('--trace-path', default=str(DEFAULT_SHAREGPT_TRACE_PATH))
    parser.add_argument('--replay-trace', default=str(DEFAULT_REPLAY_TRACE_PATH))
    parser.add_argument('--workload', choices=('sharegpt', 'rag', 'mixed'), default='mixed')
    parser.add_argument('--hotpotqa-path', type=Path, default=DEFAULT_HOTPOTQA_PATH)
    parser.add_argument('--download-hotpotqa', action='store_true')
    parser.add_argument('--hotpotqa-max-examples', type=int, default=5)
    parser.add_argument('--rag-requests', type=int, default=100)
    parser.add_argument('--rag-chunk-words', type=int, default=56)
    parser.add_argument('--rag-chunks-per-query', type=int, default=2)
    parser.add_argument('--rag-query-repeats', type=int, default=4)
    parser.add_argument('--sharegpt-order', choices=('file', 'longest'), default='longest')
    parser.add_argument('--timeout-s', type=float, default=120.0)
    parser.add_argument('--policies', nargs='+', default=list(POLICIES), choices=POLICIES)
    parser.add_argument(
        '--budgets',
        nargs='+',
        default=list(BUDGET_GPU_MEMORY_UTILIZATION),
        help='Budget names (tight/mid/loose) or numeric gpu_memory_utilization values such as 0.60.',
    )
    parser.add_argument('--max-sessions', type=int, default=200)
    parser.add_argument('--max-requests', type=int, default=1024)
    parser.add_argument('--tensor-parallel-size', type=int, default=2)
    parser.add_argument('--max-model-len', type=int, default=2048)
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--replay-batch-size', type=int, default=1)
    parser.add_argument(
        '--batch-order',
        choices=('original', 'length_bucket', 'round_robin'),
        default='original',
        help=(
            'Order requests before batching. original preserves trace order; '
            'length_bucket groups similar prompt lengths while preserving order '
            'within each length bucket; round_robin interleaves active session '
            'windows sized by replay batch size so a batch tends to contain one '
            'request from each session.'
        ),
    )
    parser.add_argument(
        '--warmup-batches',
        type=int,
        default=0,
        help='Run this many small warmup batches before measured replay; not written to CSV.',
    )
    parser.add_argument(
        '--max-num-batched-tokens',
        type=int,
        default=None,
        help=(
            'vLLM scheduler/profile token cap. Defaults to '
            'min(8192, max_model_len * replay_batch_size) so increasing '
            'replay batch size does not also inflate the prefill token budget.'
        ),
    )
    parser.add_argument('--c-re-ms-per-token', type=float, default=float(os.environ.get('EDGEKV_C_RE_MS_PER_TOKEN', DEFAULT_C_RE_MS_PER_TOKEN)))
    parser.add_argument('--bw-gbps', type=float, default=float(os.environ.get('EDGEKV_BW_GBPS', DEFAULT_BW_GBPS)))
    parser.add_argument('--d-deser-ms', type=float, default=float(os.environ.get('EDGEKV_D_DESER_MS', DEFAULT_D_DESER_MS)))
    parser.add_argument('--visible-devices', default='0,1')
    parser.add_argument('--stats-dir', default='')
    args = parser.parse_args()
    if args.warmup_batches < 0:
        parser.error('--warmup-batches must be >= 0')
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.stats_dir:
        args.stats_dir = str(out_dir / 'edgekv_gpu_stats')

    trace = load_trace(args)
    if not trace:
        raise RuntimeError('ShareGPT trace produced no runnable prompts')
    write_jsonl(out_dir / 'trace_used.jsonl', trace)

    param_count = model_parameter_count(args.model)
    per_gpu_total_mib = visible_gpu_total_mib(args.visible_devices)
    budgets = {
        name: resolve_budget(args, name, param_count, per_gpu_total_mib)
        for name in args.budgets
    }

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for budget in args.budgets:
        for policy in args.policies:
            rows, summary, samples = run_cell(args, trace, policy, budget, budgets[budget])
            all_rows.extend(rows)
            summaries.append(summary)
            prefix = f'{budget}_{policy}'
            write_csv(out_dir / f'{prefix}_requests.csv', rows)
            write_csv(out_dir / f'{prefix}_gpu_memory_samples.csv', samples)
            (out_dir / f'{prefix}_summary.json').write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
            )

    write_csv(out_dir / 'h1_vllm_real_requests.csv', all_rows)
    write_csv(out_dir / 'h1_vllm_real_summary.csv', summaries)
    (out_dir / 'config.json').write_text(
        json.dumps(
            {
                **vars(args),
                'budget_formula': 'fixed per-budget gpu_memory_utilization calibrated for H1 pressure',
                'budget_gpu_memory_utilization': BUDGET_GPU_MEMORY_UTILIZATION,
                'model_precision_bits': MODEL_PRECISION_BITS,
                'resolved_budgets': budgets,
                'trace_size': len(trace),
                'policies': args.policies,
                'trace_builder': 'h0.load_replay_prompts',
                'vllm_generate_tqdm': False,
                'replay_progress_log': 'task_level',
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding='utf-8',
    )
    plot_summary(out_dir / 'h1_memory_budget_vs_p95.png', summaries)
    print(json.dumps({'out_dir': str(out_dir), 'trace_size': len(trace), 'summaries': summaries}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
