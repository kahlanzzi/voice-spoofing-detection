"""Synthetic builder checks for the frozen V1b recipe to V1c S conversion."""
from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import src.augmentation.selection_set_build as build_module
from src.augmentation.selection_set_build import (
    RECIPE_COLUMNS,
    STAGE8_VAL_A_NPZ_SHA256,
    check_stage8_reference,
    load_recipe,
    materialize_arrays,
    official_runtime_record,
)
from src.augmentation.selection_set_recipe import CONDITIONS, apply_condition, waveform_sha256
from src.preprocessing.profile import load_profile


class FakeCodec:
    def round_trip(self, waveform, option):
        return np.asarray(waveform * (0.8 if option["id"] == "opus_16" else 0.6), dtype=np.float32)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path):
    root, noise_root, rir_root = tmp_path / "for", tmp_path / "noise", tmp_path / "rir"
    for path in (root / "FAKE", noise_root, rir_root):
        path.mkdir(parents=True)
    t = np.arange(32000, dtype=np.float64) / 16000
    source = (0.15 * np.sin(2 * np.pi * 231 * t)).astype(np.float32)
    source_path = root / "FAKE" / "one.wav"
    sf.write(source_path, source, 16000, subtype="FLOAT")
    sf.write(noise_root / "n.wav", np.linspace(-0.2, 0.3, 33000, dtype=np.float32),
             16000, subtype="FLOAT")
    sf.write(rir_root / "r.wav", np.array([1.0, 0.2, -0.1], dtype=np.float32),
             16000, subtype="FLOAT")
    noise = [{"rel_path": "n.wav", "sha256": _sha(noise_root / "n.wav"), "half": "selection"}]
    rirs = [{"rel_path": "r.wav", "sha256": _sha(rir_root / "r.wav"), "half": "selection"}]
    options = {name: {"id": name} for name in ("opus_16", "narrowband_mulaw")}
    decoded = sf.read(source_path, dtype="float32")[0]
    rows = []
    for condition in CONDITIONS:
        result = apply_condition(
            decoded, "FAKE/one.wav", condition,
            noise_inventory=noise, rir_inventory=rirs, noise_root=noise_root, rirs_root=rir_root,
            codec_backend=FakeCodec(), codec_options=options,
        )
        rows.append({
            "source_path": "FAKE/one.wav", "label": "FAKE", "subgroup": "FAKE_other",
            "group": 1, "source_file_sha256": _sha(source_path), "condition": condition,
            "condition_seed": result["condition_seed"], "resource_path": result["resource_path"],
            "resource_file_sha256": result["resource_file_sha256"],
            "resource_channel": result["resource_channel"],
            "parameters_json": json.dumps(result["parameters"], sort_keys=True, separators=(",", ":")),
            "sample_rate_hz": 16000, "samples": 32000,
            "waveform_float32_sha256": waveform_sha256(result["output"]),
        })
    recipe_path = tmp_path / "recipe.csv"
    pd.DataFrame(rows, columns=RECIPE_COLUMNS).to_csv(recipe_path, index=False)
    f2 = pd.DataFrame([{
        "path": "FAKE/one.wav", "label": "FAKE", "subgroup": "FAKE_other", "group": 1,
        "split": "val-A", "hash": _sha(source_path),
    }])
    return root, noise_root, rir_root, noise, rirs, options, recipe_path, f2


def test_materializes_all_conditions_after_hash_checks(tmp_path):
    root, noise_root, rir_root, noise, rirs, options, recipe_path, f2 = _case(tmp_path)
    recipe = load_recipe(recipe_path, expected_sources=1)
    profile = load_profile(Path(__file__).resolve().parents[2] / "configs" / "preprocessing.yaml")
    arrays = materialize_arrays(
        root, f2, recipe, noise_inventory=noise, rir_inventory=rirs,
        musan_noise_root=noise_root, rirs_root=rir_root, codec_backend=FakeCodec(),
        codec_options=options, profile=profile, expected_sources=1,
    )
    assert list(arrays) == list(CONDITIONS)
    for values in arrays.values():
        assert values["waveforms"].shape == (1, 32000)
        assert values["log_mel"].shape == (1, 1, 64, 197)
        assert values["labels"].tolist() == [1]


def test_refuses_recipe_waveform_drift(tmp_path):
    root, noise_root, rir_root, noise, rirs, options, recipe_path, f2 = _case(tmp_path)
    recipe = load_recipe(recipe_path, expected_sources=1)
    recipe.loc[recipe["condition"] == "clean", "waveform_float32_sha256"] = "0" * 64
    profile = load_profile(Path(__file__).resolve().parents[2] / "configs" / "preprocessing.yaml")
    with pytest.raises(ValueError, match="waveform SHA-256"):
        materialize_arrays(
            root, f2, recipe, noise_inventory=noise, rir_inventory=rirs,
            musan_noise_root=noise_root, rirs_root=rir_root, codec_backend=FakeCodec(),
            codec_options=options, profile=profile, expected_sources=1,
        )


def test_requires_official_label_counts(tmp_path, monkeypatch):
    *_, recipe_path, _ = _case(tmp_path)
    monkeypatch.setattr(build_module, "EXPECTED_SOURCES", 1)
    monkeypatch.setattr(build_module, "EXPECTED_COUNTS", {"REAL": 1})
    monkeypatch.setattr(build_module, "RECIPE_SHA256", _sha(recipe_path))
    with pytest.raises(ValueError, match="label counts"):
        load_recipe(recipe_path, expected_sources=1)


def test_requires_approved_stage8_val_a_hash():
    approved = (None, None, None, {"npz_sha256": STAGE8_VAL_A_NPZ_SHA256})
    assert check_stage8_reference(approved) is approved
    with pytest.raises(ValueError, match="differs from approved"):
        check_stage8_reference((None, None, None, {"npz_sha256": "0" * 64}))


def test_condition_fields_pin_recipe_and_policies():
    fields = build_module._condition_fields()
    assert [item["name"] for item in fields] == list(CONDITIONS)
    assert all(item["parameters"]["recipe_file_sha256"] == build_module.RECIPE_SHA256
               for item in fields)
    assert fields[2]["parameters"]["channel_policy"] == "seeded_single_channel"
    assert fields[3]["parameters"]["codec_option"] == "opus_16"
    assert fields[4]["parameters"]["codec_option"] == "narrowband_mulaw"


def test_official_runtime_record_requires_d58_environment(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _index: "Tesla T4")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.11.0+cu130", cuda=cuda))
    monkeypatch.setattr(build_module.platform, "python_version", lambda: "3.13.16")
    monkeypatch.setattr(build_module.importlib.metadata, "version", lambda name: "0.11.0")
    codec = SimpleNamespace(version_text=lambda: "ffmpeg version 6.1.1-3ubuntu5\nmore")
    record = official_runtime_record(codec, "2026.10")
    assert record["gpu"] == "Tesla T4"
    assert record["ffmpeg_version_line"] == "ffmpeg version 6.1.1-3ubuntu5"
    with pytest.raises(ValueError, match="Colab runtime"):
        official_runtime_record(codec, "2026.11")
