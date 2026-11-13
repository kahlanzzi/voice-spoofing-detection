from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.augmentation import (
    AugmentationError,
    AugmentationResources,
    Augmenter,
    load_augmentation_config,
)


CONFIG = load_augmentation_config(
    Path(__file__).resolve().parents[2] / "configs" / "augmentation.yaml"
)


def _tone(samples=3200):
    t = np.arange(samples, dtype=np.float32) / 16000
    return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class FakeCodec:
    def round_trip(self, waveform, option):
        return np.asarray(waveform, dtype=np.float32) * np.float32(0.5)


@pytest.fixture
def augmenter():
    rng = np.random.default_rng(11)
    resources = AugmentationResources(
        noise_waveforms=(rng.normal(size=5000).astype(np.float32),),
        rir_waveforms=(np.array([1.0, 0.3, 0.1], dtype=np.float32),),
    )
    return Augmenter(CONFIG, resources, FakeCodec())


def test_same_seed_is_byte_identical_and_label_independent(augmenter):
    real = augmenter.augment(_tone(), split="train", label="REAL", seed=19)
    fake = augmenter.augment(_tone(), split="train", label="FAKE", seed=19)
    assert real.family == fake.family
    assert np.array_equal(real.waveform, fake.waveform)
    assert dict(real.parameters) == dict(fake.parameters)


@pytest.mark.parametrize("split", ["val-A", "val-B", "test", "selection"])
def test_non_training_splits_are_refused(augmenter, split):
    with pytest.raises(AugmentationError, match="forbids augmentation"):
        augmenter.augment(_tone(), split=split, label="REAL", seed=1)


def test_clean_probability_and_five_families_are_uniform(augmenter):
    families = Counter()
    for seed in range(20000):
        rng = np.random.RandomState(seed)
        if float(rng.uniform()) >= 0.8:
            families["clean"] += 1
        else:
            families[augmenter.sample_family(rng)] += 1
    assert set(families) == {"clean", *CONFIG.families}
    assert abs(families["clean"] / 20000 - 0.2) < 0.015
    for family in CONFIG.families:
        assert abs(families[family] / 20000 - 0.16) < 0.015


def test_seven_codec_options_are_uniform(augmenter):
    counts = Counter(
        augmenter.sample_codec_option(np.random.RandomState(seed))["id"]
        for seed in range(14000)
    )
    assert set(counts) == {item["id"] for item in CONFIG.codec_options}
    for count in counts.values():
        assert abs(count / 14000 - 1 / 7) < 0.012


def test_gain_never_exceeds_minus_one_dbfs(augmenter):
    loud = np.ones(3200, dtype=np.float32)
    result, parameters = augmenter._gain(loud, np.random.RandomState(4))
    ceiling = 10 ** (-1 / 20)
    assert parameters["peak_limited"] is True
    assert float(np.max(np.abs(result))) <= ceiling + 1e-7


@pytest.mark.parametrize("family", ["recorded_noise", "reverberation", "codec", "gain", "rawboost"])
def test_each_family_preserves_shape_dtype_and_finiteness(augmenter, monkeypatch, family):
    monkeypatch.setattr(augmenter, "sample_family", lambda rng: family)
    # Seed 0 has first draw < 0.8, so the selected family is applied.
    result = augmenter.augment(_tone(), split="train", label="REAL", seed=0)
    assert result.family == family
    assert result.waveform.shape == _tone().shape
    assert result.waveform.dtype == np.float32
    assert np.isfinite(result.waveform).all()


def test_unpinned_default_codec_backend_refuses_execution(monkeypatch):
    augmenter = Augmenter(CONFIG)
    monkeypatch.setattr(augmenter, "sample_family", lambda rng: "codec")
    with pytest.raises(AugmentationError, match="pins must be filled by V1b"):
        augmenter.augment(_tone(), split="train", label="REAL", seed=0)
