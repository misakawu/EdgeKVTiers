#!/usr/bin/env python3
"""Run H1 0.5.1 on real vLLM 0.11.0 OffloadingConnector.

Experiment shape from 00_预实验提取 0.5.1:
- four policies: LRU, LFU, vLLM default, LPE-score
- three memory budgets
- the same ShareGPT multi-turn trace, first 200 sessions by default
- record p95 TTFT proxy, hit rate, and GPU memory peak

Notes:
- vLLM offline ``LLM.generate`` does not expose first-token streaming timing, so
  this runner records per-request end-to-end latency as ``ttft_proxy_ms``. For a
  true TTFT value, run through OpenAI streaming server mode.
- hit rate is the same trace-side session-prefix reuse proxy used by H0.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAREGPT_TRACE_PATH = Path(
    '/DATACENTER3/zhenxiang.wang/data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
)

from transformers import AutoTokenizer, PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens
    )

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


POLICIES = ('vllm_default', 'h1_lru', 'h1_lfu', 'h1_lpe')
BUDGET_GPU_MEMORY_UTILIZATION = {
    'tight': 0.710,
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
    gpu_memory_utilization = BUDGET_GPU_MEMORY_UTILIZATION[budget_name]
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
    prompts = load_sharegpt_prompts(
        Path(args.trace_path),
        max_sessions=args.max_sessions,
        max_requests=args.max_requests,
        tokenizer=tokenizer,
    )
    filtered: list[dict[str, Any]] = []
    for item in prompts:
        if int(item['n_tokens']) + args.max_tokens <= args.max_model_len:
            filtered.append(item)
        if len(filtered) >= args.max_requests:
            break
    return filtered


def build_llm(args: argparse.Namespace, policy: str, gpu_memory_utilization: float) -> LLM:
    if policy == 'vllm_default':
        # vLLM 0.11.0 built-in CPUOffloadingSpec uses its own LRUOffloadingManager.
        extra = {
            'spec_name': 'CPUOffloadingSpec',
            'num_cpu_blocks': args.num_cpu_blocks,
            'block_size': args.offload_block_size,
        }
    else:
        extra = {
            'spec_name': 'H1CPUOffloadingSpec',
            'spec_module_path': 'edgekv_v1_offload.vllm0110_offload',
            'cache_policy': policy,
            'num_cpu_blocks': args.num_cpu_blocks,
            'block_size': args.offload_block_size,
        }
    kv_transfer_config = KVTransferConfig(
        kv_connector='OffloadingConnector',
        kv_role='kv_both',
        kv_connector_extra_config=extra,
    )
    return LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=args.max_model_len,
        kv_transfer_config=kv_transfer_config,
        enforce_eager=True,
        max_num_seqs=args.replay_batch_size,
        max_num_batched_tokens=args.max_model_len * args.replay_batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
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


def run_cell(
    args: argparse.Namespace,
    trace: list[dict[str, Any]],
    policy: str,
    budget_name: str,
    budget: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    llm: LLM | None = None
    rows: list[dict[str, Any]] = []
    monitor = GpuMemoryMonitor()
    started = time.time()
    ok = False
    error = ''
    session_seen: set[str] = set()
    gpu_memory_utilization = float(budget['gpu_memory_utilization'])
    try:
        monitor.start()
        llm = build_llm(args, policy, gpu_memory_utilization)
        sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
        for batch_start in range(0, len(trace), args.replay_batch_size):
            batch = trace[batch_start:batch_start + args.replay_batch_size]
            prompts = [str(item['prompt']) for item in batch]
            trace_hits = [
                item['session_id'] in session_seen and int(item['turn_index']) > 0
                for item in batch
            ]
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            for offset, (item, output, trace_hit) in enumerate(zip(batch, outputs, trace_hits)):
                text = output.outputs[0].text
                idx = batch_start + offset
                row = {
                    'experiment': 'H1_vLLM_real_0_5_1',
                    'policy': policy,
                    'budget': budget_name,
                    'gpu_memory_utilization': gpu_memory_utilization,
                    'model_precision_bits': budget['model_precision_bits'],
                    'budget_target_memory_mib': round(float(budget['target_memory_mib']), 6),
                    'budget_per_gpu_target_memory_mib': round(float(budget['per_gpu_target_memory_mib']), 6),
                    'event_index': idx,
                    'batch_start_index': batch_start,
                    'replay_batch_size': args.replay_batch_size,
                    'request_id': item['request_id'],
                    'session_id': item['session_id'],
                    'turn_index': item['turn_index'],
                    'hit': trace_hit,
                    'hit_source': 'trace_side_session_prefix_reuse',
                    'n_tokens': item['n_tokens'],
                    'prompt_chars': item['prompt_chars'],
                    'max_tokens': args.max_tokens,
                    'latency_ms': round(latency_ms, 6),
                    'ttft_proxy_ms': round(latency_ms, 6),
                    'output_chars': len(text),
                    'model': args.model,
                    'tensor_parallel_size': args.tensor_parallel_size,
                    'num_cpu_blocks': args.num_cpu_blocks,
                    'offload_block_size': args.offload_block_size,
                }
                rows.append(row)
            for item in batch:
                session_seen.add(str(item['session_id']))
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
    hit_count = sum(1 for row in valid if row.get('hit'))
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
        'trace_requests_available': len(trace),
        'ttft_proxy_p50_ms': round(percentile(ttft_values, 50), 6),
        'ttft_proxy_p95_ms': round(percentile(ttft_values, 95), 6),
        'latency_p95_ms': round(percentile(ttft_values, 95), 6),
        'latency_mean_ms': round(statistics.mean(ttft_values), 6) if ttft_values else 0.0,
        'hit_rate': round(hit_count / max(len(valid), 1), 6),
        'gpu_memory_peak_mib': round(monitor.peak_mib(args.visible_devices), 6),
        'elapsed_s': round(time.time() - started, 3),
        'model': args.model,
        'tensor_parallel_size': args.tensor_parallel_size,
        'num_cpu_blocks': args.num_cpu_blocks,
        'offload_block_size': args.offload_block_size,
        'replay_batch_size': args.replay_batch_size,
        'ttft_note': 'offline LLM.generate latency is used as TTFT proxy; with replay_batch_size>1 each request receives its concurrent batch latency',
    }
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='out/h1_vllm_real')
    parser.add_argument('--model', default='models/Qwen2.5-7B-Instruct')
    parser.add_argument('--dtype', default='float16')
    parser.add_argument('--trace-path', default=str(DEFAULT_SHAREGPT_TRACE_PATH))
    parser.add_argument('--policies', nargs='+', default=list(POLICIES), choices=POLICIES)
    parser.add_argument('--budgets', nargs='+', default=list(BUDGET_GPU_MEMORY_UTILIZATION), choices=list(BUDGET_GPU_MEMORY_UTILIZATION))
    parser.add_argument('--max-sessions', type=int, default=200)
    parser.add_argument('--max-requests', type=int, default=200)
    parser.add_argument('--tensor-parallel-size', type=int, default=2)
    parser.add_argument('--max-model-len', type=int, default=1024)
    parser.add_argument('--max-tokens', type=int, default=8)
    parser.add_argument('--replay-batch-size', type=int, default=1)
    parser.add_argument('--num-cpu-blocks', type=int, default=256)
    parser.add_argument('--offload-block-size', type=int, default=16)
    parser.add_argument('--visible-devices', default='0,1')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    plot_summary(out_dir / 'h1_memory_budget_vs_p95.png', summaries)
    print(json.dumps({'out_dir': str(out_dir), 'trace_size': len(trace), 'summaries': summaries}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
