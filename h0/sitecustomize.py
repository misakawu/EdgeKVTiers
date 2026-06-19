"""Runtime compatibility patches for H0 vLLM server entrypoints."""

try:
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )
except Exception:
    pass
