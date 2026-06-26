"""Shared helpers for the h1 run_*.py experiment drivers.

Each run_*.py file holds the full configuration of one "big task" at the top and
uses these helpers to execute its experiment matrix. The helpers wrap the lower
level executors that are *not* rewritten:

  - h1/run_h1_policy_serving_bench.sh   (compat wrapper around pressure replay)
  - h1/run_h1_vllm0110_real.py          (the skewed-reuse replay harness)

and the summarize_*.py aggregators, then clean up per-cell intermediates.

Set EDGEKV_DRY_RUN=1 to print the commands / env that would run without executing
them -- used to diff a run_*.py against the .sh it replaces.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Repo root (parent of h1/). All commands run with this as cwd, mirroring the
# `cd "$ROOT_DIR"` at the top of every original .sh.
ROOT = Path(__file__).resolve().parents[1]

DRY_RUN = os.environ.get("EDGEKV_DRY_RUN", "0") == "1"

# Lower-level executors (relative to ROOT).
PRESSURE_REPLAY_WRAPPER = "h1/run_h1_policy_serving_bench.sh"
SERVING_BENCH = PRESSURE_REPLAY_WRAPPER
REAL_HARNESS = "h1/run_h1_vllm0110_real.py"
CONDA_ENV = "edgekv-vllm0110"


def log(msg: str) -> None:
    print(msg, flush=True)


def _run_streaming(cmd: list[str], *, env: dict[str, str] | None,
                   log_file: Path | None, echo: bool) -> int:
    """Run cmd from ROOT, streaming combined stdout/stderr to the console (when
    `echo`) and to `log_file` (when given). Returns the process exit code.

    Replicates the original scripts' `| tee logfile` (echo=True) and `> logfile 2>&1`
    (echo=False) behaviors.
    """
    if DRY_RUN:
        log(f"[dry-run] cmd: {' '.join(cmd)}")
        if log_file is not None:
            log(f"[dry-run] log: {log_file}")
        return 0

    fh = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = log_file.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
            if fh is not None:
                fh.write(line)
                fh.flush()
        return proc.wait()
    finally:
        if fh is not None:
            fh.close()


def run_bench_cell(out_dir: Path, visible_devices: str, env_overrides: dict[str, str],
                   *, log_file: Path | None = None, echo: bool = True,
                   force: bool = False) -> bool:
    """Run one pressure-replay cell via run_h1_policy_serving_bench.sh.

    Resumable: if out_dir/aggregate.csv already exists and not force, skip and
    return False. Returns True if the cell was executed. Raises on non-zero exit
    (matches the original `set -e`).
    """
    out_dir = Path(out_dir)
    if (out_dir / "aggregate.csv").exists() and not force:
        log(f"[skip] {out_dir} already has aggregate.csv")
        return False

    env = {**os.environ, **{k: str(v) for k, v in env_overrides.items()}}
    cmd = ["bash", SERVING_BENCH, str(out_dir), visible_devices]
    rc = _run_streaming(cmd, env=env, log_file=log_file, echo=echo)
    if rc != 0:
        raise RuntimeError(f"pressure replay wrapper cell failed (rc={rc}): {out_dir}")
    return True


def run_real_cell(out_dir: Path, visible_devices: str, cli_args: list[str],
                  env_overrides: dict[str, str], *, log_file: Path,
                  attention_backend: str = "TRITON_ATTN") -> int:
    """Run one skewed-reuse replay cell via run_h1_vllm0110_real.py (conda env).

    Mirrors run_skewed_reuse_experiment.sh: foreground execution, env injected,
    output redirected to log_file (no console echo). Returns the exit code; the
    caller decides whether a non-zero code is fatal (the .sh only warned).
    """
    env = {
        **os.environ,
        "PYTHONPATH": "h1:h0",
        "CUDA_VISIBLE_DEVICES": visible_devices,
        "VLLM_USE_V1": "1",
        "VLLM_ATTENTION_BACKEND": attention_backend,
        "VLLM_NO_USAGE_STATS": "1",
        **{k: str(v) for k, v in env_overrides.items()},
    }
    cmd = [
        "conda", "run", "--no-capture-output", "-n", CONDA_ENV,
        "python", REAL_HARNESS, *cli_args,
    ]
    return _run_streaming(cmd, env=env, log_file=log_file, echo=False)


def summarize(script_name: str, args: list[str]) -> None:
    """Run a summarize_*.py aggregator: python3 h1/<script_name> <args...>."""
    cmd = [sys.executable, str(Path("h1") / script_name), *args]
    if DRY_RUN:
        log(f"[dry-run] summarize: {' '.join(cmd)}")
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def cleanup_dirs(base_out: Path, *, keep: bool, only: list[str] | None = None,
                 extra: list[Path] | None = None) -> None:
    """Drop per-cell intermediates under base_out unless keep is True.

    only=None  -> remove every immediate subdirectory of base_out (find -maxdepth 1 -type d).
    only=[...] -> remove only those named subdirectories (single_parameter's precise cleanup).
    extra      -> additional explicit paths to remove (e.g. a logs dir).
    """
    if keep:
        log("[keep] retaining per-cell outputs and logs (keep_cells=True)")
        return
    log("[cleanup] removing per-cell outputs and logs (pass --keep-cells to retain)")
    if DRY_RUN:
        return
    base_out = Path(base_out)
    if only is not None:
        for name in only:
            shutil.rmtree(base_out / name, ignore_errors=True)
    else:
        if base_out.is_dir():
            for child in base_out.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
    for path in extra or []:
        shutil.rmtree(path, ignore_errors=True)
