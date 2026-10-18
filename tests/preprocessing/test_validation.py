"""Stage 6 validation helpers use synthetic arrays and temporary files only."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import PreprocessingError, load_profile
from src.preprocessing.validation import (
    cosine_similarity,
    relative_l2,
    resolve_audio,
    shortcut_screen,
    shortcut_stats,
)


@pytest.fixture(scope="module")
def profile():
    from pathlib import Path
    return load_profile(Path(__file__).resolve().parents[2] / "configs" / "preprocessing.yaml")


def test_shortcut_stats_and_screen_find_planted_loudness_cue(profile):
    quiet = np.ones(32000, dtype=np.float32) * 0.01
    loud = np.ones(32000, dtype=np.float32) * 0.3
    rows = []
    for label, subgroup, waveform in (
        ("REAL", "REAL", quiet), ("FAKE", "FAKE_mp3", loud),
        ("REAL", "REAL", quiet), ("FAKE", "FAKE_other", loud),
    ):
        rows.append({"label": label, "subgroup": subgroup, "split": "train", **shortcut_stats(waveform, profile)})
    screen = shortcut_screen(pd.DataFrame(rows), profile)
    rms = screen[(screen.scope == "train") & (screen.comparison == "FAKE_vs_REAL") & (screen.feature == "rms_dbfs")].iloc[0]
    assert rms.auc_separability == 1.0
    assert rms.flag == "HIGH"


def test_similarity_metrics_have_exact_identity_and_detect_change():
    expected = np.array([1.0, 2.0, 3.0])
    assert relative_l2(expected, expected) == 0.0
    assert cosine_similarity(expected, expected) == pytest.approx(1.0)
    assert relative_l2(expected + 1.0, expected) > 0
    assert cosine_similarity(expected[::-1], expected) < 1.0


def test_resolve_audio_rejects_traversal_and_missing(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    (root / "ok.wav").write_bytes(b"not decoded in this test")
    assert resolve_audio(root, "ok.wav") == (root / "ok.wav").resolve()
    with pytest.raises(PreprocessingError, match="unsafe"):
        resolve_audio(root, "../outside.wav")
    with pytest.raises(PreprocessingError, match="missing"):
        resolve_audio(root, "missing.wav")
