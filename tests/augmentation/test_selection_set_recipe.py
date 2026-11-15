"""Synthetic byte-level checks for notebook 07's extracted selection recipe."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import soundfile as sf

from src.augmentation.selection_set_recipe import (
    CONDITIONS,
    apply_condition,
    derived_seed,
    fit_noise,
    waveform_sha256,
)


class FakeCodec:
    def round_trip(self, waveform, option):
        gain = {"opus_16": 0.75, "narrowband_mulaw": 0.5}[option["id"]]
        return np.asarray(waveform * gain, dtype=np.float32)


@pytest.fixture
def recipe_inputs(tmp_path):
    t = np.arange(32000, dtype=np.float64) / 16000.0
    source = (0.2 * np.sin(2 * np.pi * 317.0 * t)).astype(np.float32)
    noise = np.linspace(-0.3, 0.2, 40003, dtype=np.float32)
    rir = np.column_stack((
        np.array([1.0, 0.2, -0.1, 0.05], dtype=np.float32),
        np.array([0.8, -0.3, 0.15, 0.02], dtype=np.float32),
    ))
    noise_root, rir_root = tmp_path / "noise", tmp_path / "rir"
    noise_root.mkdir(); rir_root.mkdir()
    sf.write(noise_root / "noise.wav", noise, 16000, subtype="FLOAT")
    sf.write(rir_root / "rir.wav", rir, 16000, subtype="FLOAT")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    noise_rows = [{"rel_path": "noise.wav", "sha256": sha(noise_root / "noise.wav"),
                   "half": "selection"}]
    rir_rows = [{"rel_path": "rir.wav", "sha256": sha(rir_root / "rir.wav"),
                 "half": "selection"}]
    options = {name: {"id": name} for name in ("opus_16", "narrowband_mulaw")}
    return source, noise_rows, rir_rows, noise_root, rir_root, options


def test_recipe_matches_frozen_synthetic_known_answers(recipe_inputs):
    source, noise, rirs, noise_root, rir_root, options = recipe_inputs
    expected = {
        # Filled from notebook-07 arithmetic and intentionally frozen as byte-level oracles.
        "clean": "91060378e8b9faae4ee30652e7f07e00ea176639183808ed7c355e7cd8649bb5",
        "recorded_noise_10db": "e16931386f347f548a98c672aaa1274eefa6f2b0aab34dbceed65b495c64de32",
        "reverberation": "24a709688cd5f987a63085f128c7b6fbd93f187eac3ca8b0e8e6f90dd72947f5",
        "opus_16": "8a6ff8f9598e3c070c5338d0dde7f170584802f75d5595e28d19d2be2a30bc9f",
        "narrowband_mulaw": "01b06d782c286ab28a7893de486f428203241cae561bf9584b1154fc95b51002",
    }
    actual = {}
    for condition in CONDITIONS:
        result = apply_condition(
            source, "FAKE/synthetic.wav", condition,
            noise_inventory=noise, rir_inventory=rirs, noise_root=noise_root, rirs_root=rir_root,
            codec_backend=FakeCodec(), codec_options=options,
        )
        actual[condition] = waveform_sha256(result["output"])
        assert result["condition_seed"] == derived_seed("FAKE/synthetic.wav", condition)
    assert actual == expected


def test_fit_noise_and_refusals(recipe_inputs):
    rng = np.random.RandomState(4)
    fitted, meta = fit_noise(np.array([1, 2], dtype=np.float32), 5, rng)
    assert fitted.tolist() == [1, 2, 1, 2, 1]
    assert meta == {"fit": "tile", "repeats": 3, "start_frame": 0}
    source, noise, rirs, noise_root, rir_root, options = recipe_inputs
    bad = [{**noise[0], "half": "training"}]
    with pytest.raises(ValueError, match="expected 'selection'"):
        apply_condition(
            source, "x.wav", "recorded_noise_10db",
            noise_inventory=bad, rir_inventory=rirs, noise_root=noise_root, rirs_root=rir_root,
            codec_backend=FakeCodec(), codec_options=options,
        )
    (noise_root / "noise.wav").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="differs from audited inventory"):
        apply_condition(
            source, "x.wav", "recorded_noise_10db",
            noise_inventory=noise, rir_inventory=rirs, noise_root=noise_root, rirs_root=rir_root,
            codec_backend=FakeCodec(), codec_options=options,
        )
