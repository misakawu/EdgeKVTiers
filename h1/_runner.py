"""h1 run_*.py 实验驱动共用辅助函数。

每个 run_*.py 文件都在顶部保存一个“大任务”的完整配置，并使用这些辅助函数
执行实验矩阵。这里包装的底层执行器暂不重写：

  - h1/run_h1_policy_serving_bench.sh   （pressure replay 兼容包装器）
  - h1/run_h1_vllm0110_real.py          （偏斜复用 replay harness）

随后调用 summarize_*.py 汇总器，并清理每个 cell 的中间产物。

设置 EDGEKV_DRY_RUN=1 时只打印将要执行的命令/环境变量，不实际运行；用于对比
run_*.py 与其替代的 .sh 脚本。
"""
from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path

# 仓库根目录（h1/ 的父目录）。所有命令都以此为 cwd，等价于原始 .sh 顶部的
# `cd "$ROOT_DIR"`。
ROOT = Path(__file__).resolve().parents[1]

DRY_RUN = os.environ.get("EDGEKV_DRY_RUN", "0") == "1"

# 底层执行器路径（相对 ROOT）。
PRESSURE_REPLAY_WRAPPER = "h1/run_h1_policy_serving_bench.sh"
SERVING_BENCH = PRESSURE_REPLAY_WRAPPER
REAL_HARNESS = "h1/run_h1_vllm0110_real.py"
CONDA_ENV = "edgekv-vllm0110"


def log(msg: str) -> None:
    print(msg, flush=True)


def _run_streaming(cmd: list[str], *, env: dict[str, str] | None,
                   log_file: Path | None, echo: bool) -> int:
    """从 ROOT 运行 cmd，并流式转发合并后的 stdout/stderr。

    `echo` 为真时同步输出到控制台；提供 `log_file` 时同步写入日志文件。
    返回进程退出码。行为对应原脚本的 `| tee logfile`（echo=True）和
    `> logfile 2>&1`（echo=False）。
    """
    if DRY_RUN:
        log(f"[dry-run] cmd: {' '.join(cmd)}")
        if env is not None:
            interesting = {
                key: env[key]
                for key in sorted(env)
                if key.startswith("EDGEKV_H1_") or key in {"PYTHONPATH", "CUDA_VISIBLE_DEVICES"}
            }
            for key, value in interesting.items():
                log(f"[dry-run] env {key}={value}")
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
    """通过 run_h1_policy_serving_bench.sh 运行一个 pressure-replay cell。

    支持断点续跑：如果 out_dir/aggregate.csv 已存在且未指定 force，则跳过并
    返回 False。实际执行 cell 时返回 True。非零退出码会抛异常，匹配原来的
    `set -e` 行为。
    """
    out_dir = Path(out_dir)
    if (out_dir / "aggregate.csv").exists() and not force:
        log(f"[skip] {out_dir} already has aggregate.csv")
        return False

    env_defaults = {"EDGEKV_H1_PROFILE_POLICY_TIME": "1"}
    policy = str(env_overrides.get("EDGEKV_H1_GPU_POLICY", os.environ.get("H1_GPU_POLICY", "")))
    if policy == "h1_lpe":
        env_defaults.update({
            "EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES": "1",
            "EDGEKV_H1_RUNTIME_MONITOR": "1",
            "EDGEKV_H1_RUNTIME_MONITOR_PATH": str(out_dir / "runtime_monitor.jsonl"),
        })
    env = {**os.environ, **env_defaults, **{k: str(v) for k, v in env_overrides.items()}}
    cmd = ["bash", SERVING_BENCH, str(out_dir), visible_devices]
    rc = _run_streaming(cmd, env=env, log_file=log_file, echo=echo)
    if rc != 0:
        raise RuntimeError(f"pressure replay wrapper cell failed (rc={rc}): {out_dir}")
    return True


def run_real_cell(out_dir: Path, visible_devices: str, cli_args: list[str],
                  env_overrides: dict[str, str], *, log_file: Path,
                  attention_backend: str = "TRITON_ATTN") -> int:
    """通过 run_h1_vllm0110_real.py 在 conda 环境里运行一个偏斜复用 cell。

    对齐 run_skewed_reuse_experiment.sh：前台执行、注入环境变量、输出重定向到
    log_file（不回显到控制台）。返回退出码；调用者决定非零退出码是否致命
    （原 .sh 只发警告）。
    """
    out_dir = Path(out_dir)
    env_defaults = {"EDGEKV_H1_PROFILE_POLICY_TIME": "1"}
    policy = str(env_overrides.get("EDGEKV_H1_GPU_POLICY", ""))
    if policy == "h1_lpe":
        env_defaults.update({
            "EDGEKV_H1_STATS_INCLUDE_OBJECT_PROFILES": "1",
            "EDGEKV_H1_RUNTIME_MONITOR": "1",
            "EDGEKV_H1_RUNTIME_MONITOR_PATH": str(out_dir / "runtime_monitor.jsonl"),
        })
    env = {
        **os.environ,
        "PYTHONPATH": "h1:h0",
        "CUDA_VISIBLE_DEVICES": visible_devices,
        "VLLM_USE_V1": "1",
        "VLLM_ATTENTION_BACKEND": attention_backend,
        "VLLM_NO_USAGE_STATS": "1",
        **env_defaults,
        **{k: str(v) for k, v in env_overrides.items()},
    }
    cmd = [
        "conda", "run", "--no-capture-output", "-n", CONDA_ENV,
        "python", REAL_HARNESS, *cli_args,
    ]
    return _run_streaming(cmd, env=env, log_file=log_file, echo=False)


def summarize(script_name: str, args: list[str]) -> None:
    """运行 summarize_*.py 汇总器：python3 h1/<script_name> <args...>。"""
    cmd = [sys.executable, str(Path("h1") / script_name), *args]
    if DRY_RUN:
        log(f"[dry-run] summarize: {' '.join(cmd)}")
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def validate_d3_for_lpe_cells(base_out: Path) -> None:
    """为 base_out 下每个保留的 LPE cell 写入 d3_validation.json。"""
    base_out = Path(base_out)
    if DRY_RUN:
        log(f"[dry-run] validate-d3 under {base_out}")
        return
    seen: set[Path] = set()
    cell_dirs: list[Path] = []
    for summary_json in sorted(base_out.glob("**/*_h1_lpe_summary.json")):
        cell_dir = summary_json.parent
        if cell_dir not in seen:
            seen.add(cell_dir)
            cell_dirs.append(cell_dir)
    manifest: list[dict[str, object]] = []
    for cell_dir in cell_dirs:
        stats_dirs = [
            path
            for path in sorted((cell_dir / "edgekv_gpu_stats").glob("*_h1_lpe"))
            if path.is_dir()
        ]
        if not stats_dirs and (cell_dir / "edgekv_gpu_stats").is_dir():
            stats_dirs = [cell_dir / "edgekv_gpu_stats"]
        if not stats_dirs:
            log(f"[warn] no D3 stats dir found for {cell_dir}")
            continue
        out_json = cell_dir / "d3_validation.json"
        cmd = [
            sys.executable,
            str(Path("h1") / "validate_d3.py"),
            str(stats_dirs[0]),
            "--out-json",
            str(out_json),
        ]
        log(f"[validate-d3] {cell_dir}")
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            log(f"[warn] validate_d3 failed rc={rc}: {cell_dir}")
            manifest.append({"cell_dir": str(cell_dir), "stats_dir": str(stats_dirs[0]), "ok": False, "rc": rc})
            continue
        try:
            payload = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        manifest.append({
            "cell_dir": str(cell_dir),
            "stats_dir": str(stats_dirs[0]),
            "out_json": str(out_json),
            "ok": True,
            "result": payload,
        })
    if manifest:
        (base_out / "d3_validation_summary.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def cleanup_dirs(base_out: Path, *, keep: bool, only: list[str] | None = None,
                 extra: list[Path] | None = None) -> None:
    """除非 keep 为 True，否则删除 base_out 下每个 cell 的中间目录。

    only=None  -> 删除 base_out 的所有一级子目录（等价 find -maxdepth 1 -type d）。
    only=[...] -> 只删除这些具名子目录（用于 single_parameter 的精确清理）。
    extra      -> 额外指定要删除的路径（例如日志目录）。
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
