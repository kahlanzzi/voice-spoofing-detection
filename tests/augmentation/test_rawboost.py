"""RawBoost checks use a generated two-tone signal; no project audio."""
from __future__ import annotations

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
    # FIR kernels may differ by a few last bits across operating systems and
    # CPU libraries.  Distributed landmarks plus global statistics check the
    # complete signal numerically without requiring byte-identical floats.
    indices = np.linspace(0, result.size - 1, 33, dtype=int)
    expected_landmarks = np.array(
        [
            0.1379072368144989, -0.03506878390908241, 0.024861598387360573,
            -0.080895334482193, 0.05777749419212341, -0.11987072229385376,
            0.0879165381193161, -0.15018104016780853, 0.11358733475208282,
            -0.17080670595169067, 0.13273751735687256, -0.18141022324562073,
            0.14332546293735504, -0.1819698065519333, 0.14404919743537903,
            -0.17248930037021637, 0.1348365992307663, -0.1530219465494156,
            0.11671709269285202, -0.12397148460149765, 0.09127115458250046,
            -0.08640597015619278, 0.06019766256213188, -0.042117275297641754,
            0.025315016508102417, 0.006581911351531744, -0.011212407611310482,
            0.05702850595116615, -0.046884935349226, 0.10607069730758667,
            -0.07929994910955429, 0.14977672696113586, 0.059496838599443436,
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        result[indices], expected_landmarks, rtol=2e-6, atol=2e-7,
    )
    observed_stats = np.array(
        [result.mean(), result.std(), result.min(), result.max(), np.linalg.norm(result)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        observed_stats,
        [2.3283064365386963e-10, 0.12294656038284302,
         -0.18360553681850433, 0.22265107929706573, 7.868579864501953],
        rtol=2e-6,
        atol=2e-7,
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
