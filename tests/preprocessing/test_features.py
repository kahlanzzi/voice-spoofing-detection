"""Feature tests use generated tones only; no dataset audio."""
from __future__ import annotations

import numpy as np
import pytest
import librosa

from src.preprocessing import (
    PreprocessingError,
    extract_features,
    load_profile,
    log_mel_spectrogram,
    mfcc_summary,
)


@pytest.fixture(scope="module")
def profile():
    from pathlib import Path
    return load_profile(Path(__file__).resolve().parents[2] / "configs" / "preprocessing.yaml")


def _tone(sr=16000):
    t = np.arange(2 * sr, dtype=np.float32) / sr
    return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _nyquist_edge_probe(sr=16000):
    """Synthetic speech-band tone plus energy near the old 8 kHz Mel edge."""
    t = np.arange(2 * sr, dtype=np.float32) / sr
    return (
        0.2 * np.sin(2 * np.pi * 440 * t)
        + 0.02 * np.sin(2 * np.pi * 7900 * t)
    ).astype(np.float32)


def test_feature_shapes_dtype_finite_and_deterministic(profile):
    prepared, log_mel, mfcc = extract_features(_tone(), 16000, profile)
    assert prepared.waveform.shape == (32000,)
    assert log_mel.shape == (1, 64, 197)
    assert mfcc.shape == (40,)
    assert log_mel.dtype == np.float32 and mfcc.dtype == np.float32
    assert np.isfinite(log_mel).all() and np.isfinite(mfcc).all()
    _, log_mel_2, mfcc_2 = extract_features(_tone(), 16000, profile)
    assert np.array_equal(log_mel, log_mel_2)
    assert np.array_equal(mfcc, mfcc_2)


def test_40_44_1_48khz_stereo_synthetic_invariance(profile):
    reference = _nyquist_edge_probe()
    _, base_log_mel, base_mfcc = extract_features(reference, 16000, profile)
    for sr in (40000, 44100, 48000):
        # D18 converts the same clip to another rate/stereo and then sends that
        # conversion through the common profile; it does not synthesize a new clip.
        converted = librosa.resample(
            reference, orig_sr=16000, target_sr=sr, res_type="soxr_hq",
            fix=True, scale=False,
        ).astype(np.float32)
        stereo = np.column_stack((converted, converted))
        _, log_mel, mfcc = extract_features(stereo, sr, profile)
        assert np.linalg.norm(log_mel - base_log_mel) / np.linalg.norm(base_log_mel) < 0.01
        assert np.linalg.norm(mfcc - base_mfcc) / np.linalg.norm(base_mfcc) < 0.01


def test_feature_functions_require_exact_model_ready_waveform(profile):
    with pytest.raises(PreprocessingError, match="shape"):
        log_mel_spectrogram(np.zeros(10, dtype=np.float32), profile)
    with pytest.raises(PreprocessingError, match="NaN"):
        bad = np.zeros(32000, dtype=np.float32)
        bad[0] = np.nan
        mfcc_summary(bad, profile)
