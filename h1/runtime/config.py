from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


@dataclass(frozen=True)
class RuntimeConfig:
    gpu_policy: str = "vllm_default"
    profile_policy_time: bool = True
    stats_dir: Path | None = None
    runtime_monitor_enabled: bool = False
    runtime_monitor_path: Path | None = None
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        policy = os.environ.get("EDGEKV_H1_GPU_POLICY", "vllm_default").strip() or "vllm_default"
        stats_value = os.environ.get("EDGEKV_H1_STATS_DIR", "").strip()
        stats_dir = Path(stats_value) if stats_value else None
        monitor_path_value = os.environ.get("EDGEKV_H1_RUNTIME_MONITOR_PATH", "").strip()
        monitor_enabled = env_bool("EDGEKV_H1_RUNTIME_MONITOR", False)
        if monitor_path_value:
            monitor_path = Path(monitor_path_value)
        elif monitor_enabled and stats_dir is not None:
            monitor_path = stats_dir / "edgekv_lpe_runtime_monitor.jsonl"
        else:
            monitor_path = None
        return cls(
            gpu_policy=policy,
            profile_policy_time=env_bool("EDGEKV_H1_PROFILE_POLICY_TIME", True),
            stats_dir=stats_dir,
            runtime_monitor_enabled=monitor_enabled,
            runtime_monitor_path=monitor_path,
            env=dict(os.environ),
        )

    @property
    def policy_enabled(self) -> bool:
        return self.gpu_policy in {"h1_lru", "h1_lfu", "h1_lpe"}

    @property
    def policy_is_lpe(self) -> bool:
        return self.gpu_policy == "h1_lpe"

    @property
    def policy_is_lru(self) -> bool:
        return self.gpu_policy == "h1_lru"

    @property
    def policy_needs_rank_state(self) -> bool:
        return self.gpu_policy in {"h1_lfu", "h1_lpe"}

    @property
    def lpe_monitor_path(self) -> Path | None:
        return self.runtime_monitor_path

    def env_float(self, name: str, default: float) -> float:
        raw = self.env.get(name)
        try:
            return float(raw if raw is not None else default)
        except (TypeError, ValueError):
            return default

    def env_int(self, name: str, default: int) -> int:
        raw = self.env.get(name)
        try:
            return int(raw if raw is not None else default)
        except (TypeError, ValueError):
            return default

    def env_bool(self, name: str, default: bool) -> bool:
        value = self.env.get(name)
        if value is None:
            return default
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
