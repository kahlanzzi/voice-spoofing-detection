"""Compute benchmark policy tests; no GPU or model is used."""
from __future__ import annotations

import pytest

from src.preprocessing import PreprocessingError
from src.preprocessing.benchmark import choose_batch_size


def test_choose_largest_batch_with_20_percent_headroom():
    gib = 1024 ** 3
    rows = [
        {"batch_size": 16, "status": "ok", "peak_allocated_bytes": 2 * gib},
        {"batch_size": 32, "status": "ok", "peak_allocated_bytes": 5 * gib},
        {"batch_size": 64, "status": "ok", "peak_allocated_bytes": 7 * gib},
        {"batch_size": 128, "status": "oom", "peak_allocated_bytes": None},
    ]
    assert choose_batch_size(rows, 8 * gib, 0.8) == 32


def test_choose_batch_fails_when_nothing_fits():
    with pytest.raises(PreprocessingError, match="No candidate"):
        choose_batch_size(
            [{"batch_size": 16, "status": "ok", "peak_allocated_bytes": 100}],
            total_vram_bytes=100, fraction=0.5,
        )
