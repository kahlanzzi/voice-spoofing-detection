"""RawBoost checks use a generated two-tone signal; no project audio."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.augmentation import AugmentationError, load_augmentation_config
from src.augmentation.rawboost import rawboost_algorithm_1


CONFIG = load_augmentation_config(
    Path(__file__).resolve().parents[2] / "configs" / "augmentation.yaml"
)


def _tone(samples=4096):
    t = np.arange(samples, dtype=np.float64) / 16000
    return (0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.08 * np.sin(2 * np.pi * 997 * t)).astype(np.float32)


def test_matches_pinned_official_reference_on_synthetic_signal():
    """Golden generated with pinned MIT RawBoost.py plus NumPy-2 scalar adaptation.

    Source commit: 4f161a8b4d0d9f4a8431509ddd86c645869ef6c4.
    The adaptation changes size-one arrays to ``.item()`` only; the official
    equations, random draw order and scipy signal operations are unchanged.
    """
    result = rawboost_algorithm_1(_tone(), CONFIG.rawboost, np.random.RandomState(2026))
    # SciPy's FIR implementation can differ by a few last bits across operating
    # systems.  Hashing every sample after eight-decimal quantisation keeps the
    # whole-output oracle strict while remaining identical on Windows/Linux.
    quantized = np.round(result.astype(np.float64), decimals=8)
    digest = hashlib.sha256(quantized.astype("<f8").tobytes()).hexdigest()
    assert digest == "26372e7c512a4985cf841f167d9635cf670ebd2f1036b8696e4c682f37f15e73"
    assert result[:4] == pytest.approx(
        [0.1379072368144989, 0.16940143704414368,
         0.1898445039987564, 0.20473429560661316], abs=1e-7,
    )


def test_rawboost_is_seed_deterministic_and_peak_safe():
    a = rawboost_algorithm_1(_tone(), CONFIG.rawboost, np.random.RandomState(7))
    b = rawboost_algorithm_1(_tone(), CONFIG.rawboost, np.random.RandomState(7))
    c = rawboost_algorithm_1(_tone(), CONFIG.rawboost, np.random.RandomState(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.dtype == np.float32 and a.shape == _tone().shape
    assert np.max(np.abs(a)) <= 1.0


def test_rawboost_refuses_wrong_rng_and_bad_waveform():
    with pytest.raises(AugmentationError, match="RandomState"):
        rawboost_algorithm_1(_tone(), CONFIG.rawboost, np.random.default_rng(1))
    with pytest.raises(AugmentationError, match="NaN"):
        rawboost_algorithm_1(np.array([0.0, np.nan]), CONFIG.rawboost,
                             np.random.RandomState(1))
