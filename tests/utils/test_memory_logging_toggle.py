# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU-only tests for the print_memory observability toggle.

The colocate model-switch path (sleep / wake_up / update_weights) calls
print_memory() several times per step. set_memory_logging(False) must turn those
calls into true no-ops so throughput-sensitive runs pay no CUDA query / all-rank
log / clear_before_print cost. No GPU or process group is required.
"""

from __future__ import annotations

import relax.utils.memory_utils as memory_utils


def test_memory_logging_toggle_disabled_is_noop(monkeypatch):
    calls = {"clear": 0, "available": 0}

    def _fail_clear(*args, **kwargs):
        calls["clear"] += 1

    def _fail_available(*args, **kwargs):
        calls["available"] += 1
        return {}

    monkeypatch.setattr(memory_utils, "clear_memory", _fail_clear)
    monkeypatch.setattr(memory_utils, "available_memory", _fail_available)

    memory_utils.set_memory_logging(False)
    try:
        # Even with clear_before_print=True, the disabled path must not touch the
        # device (no clear_memory) nor query memory (no available_memory).
        assert memory_utils.print_memory("switch", clear_before_print=True) is None
        assert calls == {"clear": 0, "available": 0}
    finally:
        memory_utils.set_memory_logging(True)


def test_memory_logging_toggle_enabled_queries(monkeypatch):
    calls = {"clear": 0, "available": 0}

    monkeypatch.setattr(memory_utils, "clear_memory", lambda *a, **k: calls.__setitem__("clear", calls["clear"] + 1))
    monkeypatch.setattr(
        memory_utils, "available_memory", lambda *a, **k: (calls.__setitem__("available", 1) or {"x": 1})
    )
    monkeypatch.setattr(memory_utils.dist, "get_rank", lambda: 0)

    memory_utils.set_memory_logging(True)
    info = memory_utils.print_memory("switch", clear_before_print=True)
    assert info == {"x": 1}
    assert calls == {"clear": 1, "available": 1}
