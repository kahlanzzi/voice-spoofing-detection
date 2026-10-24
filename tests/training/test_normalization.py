"""Per-Mel-bin train-only standardization (D42): synthetic log-Mel arrays only, seed 2026."""
from __future__ import annotations

import numpy as np
import pytest

import src.training.normalization as normalization
from src.training.normalization import FEATURE_SHAPE, N_MELS, apply_bin_stats, fit_bin_stats


def _logmel(n: int = 6, offset: float = 0.0, seed: int = 2026) -> np.ndarray:
    """(n, 1, 64, 197) float32 with a different mean and scale per Mel bin, as a real log-Mel has."""
    rng = np.random.default_rng(seed)
    bin_mean = np.linspace(-12.0, 2.0, N_MELS)[None, None, :, None]
    bin_scale = np.linspace(0.5, 4.0, N_MELS)[None, None, :, None]
    x = rng.normal(0.0, 1.0, (n, *FEATURE_SHAPE)) * bin_scale + bin_mean + offset
    return x.astype(np.float32)


def test_fit_bin_stats_are_per_bin_over_clips_and_frames():
    x = _logmel()
    copy = x.copy()
    mean, std = fit_bin_stats(x)
    assert mean.dtype == std.dtype == np.float32
    assert mean.shape == std.shape == (N_MELS,)
    # Reference: every value of bin b over all clips and frames, float64, ddof 0.
    per_bin = x[:, 0].transpose(1, 0, 2).reshape(N_MELS, -1).astype(np.float64)
    np.testing.assert_allclose(mean, per_bin.mean(axis=1), rtol=1e-6)
    np.testing.assert_allclose(std, per_bin.std(axis=1, ddof=0), rtol=1e-6)
    np.testing.assert_array_equal(x, copy)  # fitting does not touch the input


def test_apply_never_refits_and_leaves_stats_unchanged(monkeypatch):
    train, val = _logmel(), _logmel(n=3, offset=3.0, seed=7)
    mean, std = fit_bin_stats(train)
    mean_copy, std_copy = mean.copy(), std.copy()
    calls = []
    real_fit = normalization.fit_bin_stats
    monkeypatch.setattr(normalization, "fit_bin_stats",
                        lambda *a, **k: calls.append(a) or real_fit(*a, **k))

    z_train = apply_bin_stats(train, mean, std)
    z_val = apply_bin_stats(val, mean, std)

    assert calls == []  # apply never fits
    np.testing.assert_array_equal(mean, mean_copy)
    np.testing.assert_array_equal(std, std_copy)
    assert z_train.dtype == z_val.dtype == np.float32
    expected = (val - mean[None, None, :, None]) / std[None, None, :, None]
    np.testing.assert_allclose(z_val, expected, rtol=1e-6, atol=1e-6)
    # Train becomes centred per bin; val keeps its offset because its own statistics are never used.
    assert np.abs(z_train.mean(axis=(0, 1, 3))).max() < 1e-4
    assert (z_val.mean(axis=(0, 1, 3)) > 0.5).all()


@pytest.mark.parametrize("bad", [
    np.zeros((4, 64, 197), np.float32),        # missing channel axis
    np.zeros((4, 1, 63, 197), np.float32),     # wrong Mel bins
    np.zeros((4, 1, 64, 196), np.float32),     # wrong frames
    np.zeros((4, 2, 64, 197), np.float32),     # wrong channels
    np.zeros((1, 1, 64, 197), np.float32),     # n < 2
    _logmel().astype(np.float64),              # not float32
])
def test_fit_refuses_wrong_shape_dtype_or_too_few_clips(bad):
    with pytest.raises(ValueError):
        fit_bin_stats(bad)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_fit_and_apply_refuse_non_finite(value):
    x = _logmel()
    mean, std = fit_bin_stats(x)
    x[2, 0, 10, 50] = value
    with pytest.raises(ValueError, match="NaN or infinite"):
        fit_bin_stats(x)
    with pytest.raises(ValueError, match="NaN or infinite"):
        apply_bin_stats(x, mean, std)


def test_fit_refuses_zero_std_bin():
    x = _logmel()
    x[:, 0, 5, :] = -3.0  # one constant bin cannot be standardized
    with pytest.raises(ValueError, match=r"constant Mel bins .* \[5\]"):
        fit_bin_stats(x)


def test_apply_refuses_bad_stats_or_features():
    x = _logmel()
    mean, std = fit_bin_stats(x)
    zero = std.copy()
    zero[0] = 0.0
    with pytest.raises(ValueError, match="std must be > 0"):
        apply_bin_stats(x, mean, zero)
    with pytest.raises(ValueError):
        apply_bin_stats(x, mean[:63], std)
    with pytest.raises(ValueError):
        apply_bin_stats(x, mean.astype(np.float64), std)
    with pytest.raises(ValueError):
        apply_bin_stats(x[:, :, :32], mean, std)
    with pytest.raises(ValueError):
        apply_bin_stats(x[:0], mean, std)  # n >= 1
