# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real NCCL layout round-trip regression for GDN context parallelism.

Checks token-exact zigzag/contiguous conversion for packed THD inputs and an
SBHD round trip. Requires two CUDA devices and the patched Megatron-LM.

Run with:
    pytest tests/backends/megatron/test_gdn_cp_layout_gpu.py
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.multiprocessing as mp


WORLD_SIZE = 2

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"requires {WORLD_SIZE} CUDA devices",
)


def _has_backport() -> bool:
    try:
        import megatron.core.context_parallel_layout  # noqa: F401
    except ImportError:
        return False
    return True


needs_backport = pytest.mark.skipif(not _has_backport(), reason="requires patched Megatron-LM")


def _init_dist(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    torch.cuda.set_device(rank)
    import torch.distributed as dist

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def _worker_layout_round_trip(rank, world_size, _spec, _unused):
    """zigzag -> contiguous -> zigzag over a real CP group must be token-exact.

    This is the collective-level version of RFC 5.1: it drives the actual
    ``all_to_all`` in ``context_parallel_layout``, for packed THD (several
    unequal-length samples) and for SBHD.
    """
    _init_dist(rank, world_size)
    import torch.distributed as dist
    from megatron.core.context_parallel_layout import (
        contiguous_to_zigzag_chunks,
        get_thd_context_parallel_rank_indices,
        zigzag_to_contiguous_chunks,
    )

    device = torch.device("cuda", rank)
    cp_group = dist.new_group(list(range(world_size)))

    # --- packed THD, three samples of different lengths ---
    lengths = [2 * world_size * f for f in (5, 1, 3)]
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device=device, dtype=torch.int32)
    total = int(cu[-1])
    # Row t is (t, t+1e6, t+2e6): a permuted token is impossible to miss.
    full = (
        torch.arange(total, dtype=torch.float64, device=device).unsqueeze(1)
        + torch.arange(3, dtype=torch.float64, device=device).unsqueeze(0) * 1e6
    )

    zig_idx = get_thd_context_parallel_rank_indices(cu, world_size, rank, "zigzag")
    con_idx = get_thd_context_parallel_rank_indices(cu, world_size, rank, "contiguous")
    local_zig = full[zig_idx]

    got_con = zigzag_to_contiguous_chunks(local_zig, cp_group, seq_dim=0, cu_seqlens=cu)
    assert torch.equal(got_con, full[con_idx]), f"rank {rank}: THD zigzag->contiguous is wrong"
    got_zig = contiguous_to_zigzag_chunks(got_con, cp_group=cp_group, seq_dim=0, cu_seqlens=cu)
    assert torch.equal(got_zig, local_zig), f"rank {rank}: THD round trip is not identity"

    # --- SBHD (chunk-level swap, no cu_seqlens) ---
    seq_local = 2 * world_size * 4
    sbhd = torch.arange(seq_local * 2 * 3, dtype=torch.float64, device=device).reshape(seq_local, 2, 3) + rank * 1e9
    swapped = zigzag_to_contiguous_chunks(sbhd, cp_group, seq_dim=0)
    back = contiguous_to_zigzag_chunks(swapped, cp_group=cp_group, seq_dim=0)
    assert torch.equal(back, sbhd), f"rank {rank}: SBHD round trip is not identity"

    dist.barrier()
    dist.destroy_process_group()


def _spawn(fn, spec, port, extra=None):
    _spawn_world(fn, WORLD_SIZE, spec, port, extra=extra)


def _spawn_world(fn, world_size, spec, port, extra=None):
    os.environ["MASTER_PORT"] = str(port)
    mp.spawn(
        fn,
        args=(world_size, extra if extra is not None else spec, None),
        nprocs=world_size,
        join=True,
    )


@needs_backport
def test_layout_round_trip_over_real_cp_group():
    """RFC 5.1 at the collective level: the layout swap is a pure
    permutation."""
    _spawn(_worker_layout_round_trip, "n/a", 29544)
