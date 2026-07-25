# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import gc

import torch
import torch.distributed as dist

from relax.utils import device as device_utils
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


# Per-step memory-usage debug prints on the colocate model-switch path
# (sleep / wake_up / update_weights) each issue a CUDA mem query + an all-rank
# log line, and some also force synchronize()+empty_cache() via
# ``clear_before_print``. That is pure observability overhead sitting on the
# switch critical path (~10-13% of step time in colocate multimodal). Gate it so
# throughput-sensitive runs can turn it off. Default True preserves the existing
# behavior; flip via ``--no-log-memory-usage``.
_LOG_MEMORY_USAGE = True


def set_memory_logging(enabled: bool) -> None:
    """Enable/disable ``print_memory`` (debug observability on the switch path)."""
    global _LOG_MEMORY_USAGE
    _LOG_MEMORY_USAGE = enabled


def clear_memory(clear_host_memory: bool = False):
    device_utils.synchronize()
    gc.collect()
    device_utils.empty_cache()
    if clear_host_memory:
        if device_utils.is_npu_available:
            torch.npu.host_empty_cache()
        else:
            torch._C._host_emptyCache()


def available_memory():
    dev = device_utils.current_device()
    free, total = device_utils.mem_get_info(dev)
    return {
        "device": str(dev),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(device_utils.memory_allocated(dev)),
        "reserved_GB": _byte_to_gb(device_utils.memory_reserved(dev)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if not _LOG_MEMORY_USAGE:
        return None

    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
