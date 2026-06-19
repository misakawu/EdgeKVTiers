"""Local runtime patches for EdgeKVTiers experiment processes."""

from __future__ import annotations

import logging

_ORIGINAL_LOGGER_ERROR = logging.Logger.error


def _edgekv_logger_error(self: logging.Logger, msg: object, *args: object, **kwargs: object) -> None:
    """Downgrade vLLM FA2 probing noise on pre-Ampere GPUs.

    RTX 2080 Ti (SM 7.5) cannot use FlashAttention-2, but vLLM still probes it
    before selecting the configured Triton backend. The probe logs at ERROR even
    when execution can continue, which trips the H1 runner's fail-fast monitor.
    """
    if (
        self.name == 'vllm.attention.utils.fa_utils'
        and isinstance(msg, str)
        and msg.startswith('Cannot use FA version')
    ):
        self.warning(msg, *args, **kwargs)
        return
    _ORIGINAL_LOGGER_ERROR(self, msg, *args, **kwargs)


logging.Logger.error = _edgekv_logger_error
