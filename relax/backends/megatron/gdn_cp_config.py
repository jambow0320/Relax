# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from typing import Optional


def _validate_linear_cp_mode(args: Namespace, config: Optional[object] = None) -> None:
    """Validate the mode name, then GDN-specific flags once the model is known.

    Geometry-dependent rejections (e.g. explicit `headwise` on heads not
    divisible by `tp*max_cp`) can only be checked once the real GDN head counts
    are known, which happens in MCore's `TransformerConfig.__post_init__` gate
    -- not here. Bridge calls this again with the provider's actual config.
    """
    model_config = config if config is not None else args
    mode = getattr(model_config, "linear_cp_mode", "chunkwise")
    allowed_modes = {"headwise", "chunkwise", "all_gather"}
    if mode not in allowed_modes:
        raise ValueError(
            f"--linear-cp-mode must be one of {sorted(allowed_modes)!r}; got {mode!r}. Resolve 'auto' before construction."
        )

    # Bridge determines the attention variant from the HF checkpoint. The default
    # linear_cp_mode on an ordinary-attention model does not make it a GDN model.
    if getattr(model_config, "experimental_attention_variant", None) != "gated_delta_net":
        return

    cp_may_exceed_one = (
        getattr(args, "dynamic_context_parallel", False) or getattr(model_config, "context_parallel_size", 1) > 1
    )
    if cp_may_exceed_one and getattr(args, "allgather_cp", False):
        raise ValueError(
            "GDN CP requires zig-zag THD packing in every linear_cp_mode; --allgather-cp uses "
            "contiguous per-rank packing and is incompatible with GDN CP>1. --allgather-cp is a "
            "data/attention packing flag, separate from --linear-cp-mode=all_gather."
        )

    if mode == "chunkwise" and cp_may_exceed_one and getattr(model_config, "deterministic_mode", False):
        raise ValueError(
            "--linear-cp-mode=chunkwise does not support --deterministic-mode while CP>1 may occur: "
            "the deterministic torch reference path only accepts cp_context=None. "
            "Packed GDN inputs also do not support deterministic mode in the other CP modes."
        )
