"""Synthetic V1c training-half augmentation-resource cache checks."""
from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.augmentation.resource_cache import (
    JSON_NAME,
    NPZ_NAME,
    build_training_resources,
    load_training_resources,
    training_rir_channel,
)

HEX = "a" * 64
COMMIT = "b" * 40


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path, *, zero_noise=False):
    noise_root, rir_root = tmp_path / "noise", tmp_path / "rir"
    noise_root.mkdir(parents=True); rir_root.mkdir(parents=True)
    noise_rows, rir_rows = [], []
    for i in range(2):
        values = np.zeros(20, dtype=np.float32) if zero_noise and i == 0 else np.linspace(
            -0.2 + i * 0.01, 0.3, 20 + i, dtype=np.float32
        )
        path = noise_root / f"n{i}.wav"
        sf.write(path, values, 16000, subtype="FLOAT")
        noise_rows.append({"rel_path": path.name, "sha256": _sha(path), "half": "training"})
    for i in range(2):
        values = np.column_stack((np.linspace(0.1, 0.5, 8 + i, dtype=np.float32),
                                  np.linspace(-0.4, 0.2, 8 + i, dtype=np.float32)))
        path = rir_root / f"r{i}.wav"
        sf.write(path, values, 16000, subtype="FLOAT")
        rir_rows.append({"rel_path": path.name, "sha256": _sha(path), "half": "training"})
    return noise_root, rir_root, pd.DataFrame(noise_rows), pd.DataFrame(rir_rows)


def _build(tmp_path, **kw):
    roots = _inputs(tmp_path, **kw)
    out = tmp_path / "out"
    meta = build_training_resources(
        *roots, out, noise_inventory_sha256=HEX, rir_inventory_sha256="c" * 64,
        noise_archive_sha256="d" * 64, rir_archive_sha256="e" * 64,
        code_commit=COMMIT, expected_noise_count=2, expected_rir_count=2,
    )
    return out, meta


def test_build_load_and_read_only_views(tmp_path):
    out, meta = _build(tmp_path)
    resources, loaded = load_training_resources(out, expected_noise_count=2, expected_rir_count=2)
    assert loaded == meta and len(resources.noise_waveforms) == len(resources.rir_waveforms) == 2
    assert meta["n_all_zero_noise"] == meta["n_all_zero_rir"] == 0
    assert not resources.noise_waveforms[0].flags.writeable
    assert training_rir_channel("r0.wav", 1) == 0
    assert training_rir_channel("r0.wav", 8) == training_rir_channel("r0.wav", 8)


def test_refuses_wrong_half_hash_and_zero(tmp_path):
    noise_root, rir_root, noise, rirs = _inputs(tmp_path)
    noise.loc[0, "half"] = "selection"
    with pytest.raises(ValueError, match="must contain 2"):
        build_training_resources(
            noise_root, rir_root, noise, rirs, tmp_path / "bad-half",
            noise_inventory_sha256=HEX, rir_inventory_sha256="c" * 64,
            noise_archive_sha256="d" * 64, rir_archive_sha256="e" * 64,
            code_commit=COMMIT, expected_noise_count=2, expected_rir_count=2,
        )
    noise.loc[0, "half"] = "training"
    noise.loc[0, "sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from inventory"):
        build_training_resources(
            noise_root, rir_root, noise, rirs, tmp_path / "bad-hash",
            noise_inventory_sha256=HEX, rir_inventory_sha256="c" * 64,
            noise_archive_sha256="d" * 64, rir_archive_sha256="e" * 64,
            code_commit=COMMIT, expected_noise_count=2, expected_rir_count=2,
        )
    with pytest.raises(ValueError, match="all-zero"):
        _build(tmp_path / "zero", zero_noise=True)


def test_loader_detects_tampering(tmp_path):
    out, _ = _build(tmp_path)
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    blob = bytearray((copy / NPZ_NAME).read_bytes())
    blob[-20] ^= 1
    (copy / NPZ_NAME).write_bytes(blob)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_training_resources(copy, expected_noise_count=2, expected_rir_count=2)
    copy2 = tmp_path / "copy2"
    shutil.copytree(out, copy2)
    meta = json.loads((copy2 / JSON_NAME).read_text(encoding="utf-8"))
    meta["n_all_zero_noise"] = 1
    (copy2 / JSON_NAME).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="all-zero"):
        load_training_resources(copy2, expected_noise_count=2, expected_rir_count=2)
