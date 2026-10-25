"""D35 integration coverage: preprocessing is called once per complete window."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.evaluation.external_run as external_run
from src.preprocessing import load_profile, preprocess_waveform


@pytest.fixture(scope="module")
def profile():
    return load_profile(Path(__file__).resolve().parents[2] / "configs" / "preprocessing.yaml")


def _tone(samples: int, amplitude: float = 0.2) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / 16000.0
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _window(silence_each_side: int) -> np.ndarray:
    voiced = 32000 - 2 * silence_each_side
    return np.concatenate(
        (
            np.zeros(silence_each_side, dtype=np.float32),
            _tone(voiced),
            np.zeros(silence_each_side, dtype=np.float32),
        )
    )


def test_every_complete_window_uses_real_preprocessing_before_scoring(monkeypatch, profile):
    windows = (_window(4000), _window(8000), _window(12000))
    tail = np.zeros(4000, dtype=np.float32)
    recording = np.concatenate((*windows, tail))
    preprocessed_inputs: list[np.ndarray] = []
    scorer_inputs: list[np.ndarray] = []
    real_preprocess = preprocess_waveform

    def wrapped_preprocess(raw, sample_rate, supplied_profile, **kwargs):
        preprocessed_inputs.append(np.array(raw, copy=True))
        return real_preprocess(raw, sample_rate, supplied_profile, **kwargs)

    monkeypatch.setattr(external_run, "preprocess_waveform", wrapped_preprocess)

    def fake_scorer(prepared):
        scorer_inputs.append(np.array(prepared, copy=True))
        return float(np.mean(prepared))

    result = external_run.score_windows(recording, 16000, profile, fake_scorer)

    # Three complete windows are scored; the 0.25 s remainder is dropped by Protocol section 8.
    assert result.n_complete_windows == 3
    assert result.n_dropped_samples == len(tail)
    assert result.n_filter_dropped == 1
    assert tuple(row.window_index for row in result.filtered_scores) == (0, 1)
    assert len(preprocessed_inputs) == len(scorer_inputs) == 3

    # A full-recording trim before splitting would change these raw inputs.
    for actual, expected in zip(preprocessed_inputs, windows):
        np.testing.assert_array_equal(actual, expected)
    for raw, actual in zip(windows, scorer_inputs):
        expected = real_preprocess(raw, 16000, profile).waveform
        np.testing.assert_array_equal(actual, expected)

    assert tuple(row.padding_samples for row in result.scores) == (7680, 16000, 23680)


def test_external_config_is_path_driven_and_strict():
    config = external_run.load_external_config(
        Path(__file__).resolve().parents[2] / "configs" / "external_run.yaml"
    )
    assert config.dataset == "DEEP-VOICE"
    assert config.profile_path == Path("configs/preprocessing.yaml")
