#!/usr/bin/env python3
"""vLLM V1 connector wrapper for H1 custom eviction policies.

This module is import-safe on older vLLM versions. The real connector class is
created only when vLLM exposes KVConnectorBase_V1.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .policy import H1Policy, JsonlDecisionLogger


def _load_lmcache_connector_class():
    try:
        from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic

        return LMCacheConnectorV1Dynamic
    except Exception:
        from lmcache.integration.vllm.lmcache_connector_v1_085 import LMCacheConnectorV1Dynamic

        return LMCacheConnectorV1Dynamic


def _request_id(request: Any) -> str:
    return str(getattr(request, "request_id", getattr(request, "id", "unknown")))


def _prompt_token_count(request: Any) -> int:
    prompt_token_ids = getattr(request, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        return len(prompt_token_ids)
    prompt = getattr(request, "prompt", "")
    if isinstance(prompt, str):
        return max(1, len(prompt.split()))
    return 1


def _session_id(request: Any) -> str:
    req_id = _request_id(request)
    return req_id.split(":turn:", 1)[0] if ":turn:" in req_id else req_id


def build_edgekv_connector_class():
    """Return a KVConnectorBase_V1 subclass for upgraded vLLM environments."""

    base_cls = _load_lmcache_connector_class()

    class EdgeKVH1Connector(base_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            policy_name = os.environ.get("EDGEKV_H1_POLICY", "lpe-score")
            budget_mb = float(os.environ.get("EDGEKV_H1_GPU_BUDGET_MB", "2048"))
            c_re = float(os.environ.get("EDGEKV_H1_C_RE_MS_PER_TOKEN", "0.08"))
            log_path = os.environ.get("EDGEKV_H1_POLICY_LOG", "")
            logger = JsonlDecisionLogger(Path(log_path)) if log_path else None
            self.edgekv_policy = H1Policy(
                policy=policy_name,
                gpu_budget_mb=budget_mb,
                c_re_ms_per_token=c_re,
                logger=logger,
            )
            self.edgekv_kv_mib_per_token = float(os.environ.get("EDGEKV_H1_KV_MIB_PER_TOKEN", "0.02734375"))

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            matched = super().get_num_new_matched_tokens(request, num_computed_tokens)
            n_tokens = _prompt_token_count(request)
            matched_tokens = matched[0] if isinstance(matched, tuple) else matched
            hit = bool(matched_tokens and int(matched_tokens) > 0)
            self.edgekv_policy.observe_request(
                request_id=_request_id(request),
                session_id=_session_id(request),
                object_id=_session_id(request),
                object_type="session_prefix",
                n_tokens=n_tokens,
                size_mb=n_tokens * self.edgekv_kv_mib_per_token,
                hit=hit,
            )
            return matched

        def request_finished(self, request, block_ids) -> tuple[bool, Optional[dict[str, Any]]]:
            if hasattr(super(), "request_finished"):
                return super().request_finished(request, block_ids)
            return False, None

    return EdgeKVH1Connector


EdgeKVH1Connector = build_edgekv_connector_class()
