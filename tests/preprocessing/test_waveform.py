"""Waveform preprocessing tests use generated signals only; no dataset audio."""
from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from src.preprocessing import PreprocessingError, load_profile, preprocess_waveform
from src.preprocessing.waveform import speech_mask


@pytest.fixture(scope="module")
def profile():
    from pathlib import Path
    return load_profile(Path(__file__).resolve().parents[2] / "configs" / "preprocessing.yaml")


def _tone(sr=16000, seconds=2.0, amplitude=0.2):
    t = np.arange(round(sr * seconds), dtype=np.float32) / sr
    return amplitude * np.sin(2 * np.pi * 440 * t).astype(np.float32)


def test_mono_average_resample_shape_dtype_and_determinism(profile):
    mono = _tone(48000)
    stereo = np.column_stack((mono, mono * 0.5))
    first = preprocess_waveform(stereo, 48000, profile)
    second = preprocess_waveform(stereo, 48000, profile)
    assert first.waveform.shape == (32000,)
    assert first.waveform.dtype == np.float32
    assert np.array_equal(first.waveform, second.waveform)
    assert first.speech_fraction == 1.0


def test_channel_axis_zero_is_supported(profile):
    mono = _tone()
    channels_first = np.stack((mono, mono))
    result = preprocess_waveform(channels_first, 16000, profile, channel_axis=0)
    assert np.allclose(result.waveform, mono)


def test_silence_is_trimmed_then_center_padded(profile):
    signal = np.concatenate((np.zeros(3200), _tone(seconds=1.6), np.zeros(3200))).astype(np.float32)
    result = preprocess_waveform(signal, 16000, profile)
    assert result.trimmed is True
    assert result.waveform.shape == (32000,)
    assert np.count_nonzero(result.waveform[:3000]) == 0
    assert np.count_nonzero(result.waveform[-3000:]) == 0


def test_all_silent_is_preserved_and_processable(profile):
    result = preprocess_waveform(np.zeros(32000, dtype=np.float32), 16000, profile)
    assert result.speech_fraction == 0.0
    assert result.trimmed is False
    assert np.array_equal(result.waveform, np.zeros(32000, dtype=np.float32))


def test_center_crop_is_deterministic(profile):
    long = np.arange(33000, dtype=np.float32) / 33000
    result = preprocess_waveform(long, 16000, profile)
    assert result.waveform.shape == (32000,)
    assert result.waveform[0] == long[500]


def test_speech_rms_normalization_is_peak_safe(profile):
    loudness = dict(profile.loudness)
    loudness["mode"] = "speech_rms"
    normalized_profile = replace(profile, loudness=MappingProxyType(loudness))
    result = preprocess_waveform(_tone(amplitude=0.001), 16000, normalized_profile)
    assert np.max(np.abs(result.waveform)) <= float(profile.loudness["peak_ceiling"]) + 1e-6
    assert np.max(np.abs(result.waveform)) > 0.001


@pytest.mark.parametrize(
    "waveform, sample_rate, message",
    [
        (np.array([], dtype=np.float32), 16000, "empty"),
        (np.array([0.0, np.nan], dtype=np.float32), 16000, "NaN"),
        (np.zeros((2, 2, 2), dtype=np.float32), 16000, "one- or two-dimensional"),
        (np.zeros(10, dtype=np.float32), 0, "positive integer"),
    ],
)
def test_invalid_waveforms_fail_loudly(profile, waveform, sample_rate, message):
    with pytest.raises(PreprocessingError, match=message):
        preprocess_waveform(waveform, sample_rate, profile)


def test_speech_mask_uses_20ms_frames_and_absolute_threshold(profile):
    frame = np.ones(320, dtype=np.float32) * 10 ** (-40 / 20)
    quiet = np.ones(320, dtype=np.float32) * 10 ** (-60 / 20)
    mask = speech_mask(np.concatenate((quiet, frame, quiet)), profile)
    assert mask.tolist() == [False, True, False]
