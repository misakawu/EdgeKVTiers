#!/usr/bin/env python3
"""Runtime checks for vLLM V1 KV connector support."""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class V1OffloadEnvStatus:
    ok: bool
    vllm_version: str
    lmcache_version: str
    has_v1_connector_base: bool
    has_lmcache_v1_connector: bool
    message: str


def _version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"not-importable: {exc!r}"
    return str(getattr(module, "__version__", "unknown"))


def check_v1_offload_env() -> V1OffloadEnvStatus:
    vllm_version = _version("vllm")
    lmcache_version = _version("lmcache")
    has_base = _has_module("vllm.distributed.kv_transfer.kv_connector.v1.base")
    has_lmcache = _has_module("lmcache.integration.vllm.lmcache_connector_v1") or _has_module(
        "lmcache.integration.vllm.lmcache_connector_v1_085"
    )
    ok = has_base and has_lmcache and not vllm_version.startswith("not-importable")
    if ok:
        message = "vLLM V1 KV connector and LMCache V1 adapter are importable."
    else:
        message = (
            "vLLM V1 KV connector is not available in this environment. "
            "Upgrade h3-lmcache-blog to a vLLM/LMCache pair that exposes "
            "vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1."
        )
    return V1OffloadEnvStatus(
        ok=ok,
        vllm_version=vllm_version,
        lmcache_version=lmcache_version,
        has_v1_connector_base=has_base,
        has_lmcache_v1_connector=has_lmcache,
        message=message,
    )


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def write_env_status(path: Path, status: Optional[V1OffloadEnvStatus] = None) -> V1OffloadEnvStatus:
    status = status or check_v1_offload_env()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(status), f, indent=2, ensure_ascii=False, sort_keys=True)
    return status
