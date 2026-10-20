"""MFCC cache (Stage 7 step 1): synthetic WAV files and a synthetic manifest only, seed 2026.

Convention: label 1 = spoof (FAKE), 0 = bonafide (REAL).
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import src.features.mfcc_cache as cache_module
from src.features.mfcc_cache import build_mfcc_cache, load_mfcc_cache
from src.preprocessing import PreprocessingError, extract_features, load_profile

REPO = Path(__file__).resolve().parents[2]
PROVENANCE = {
    "f2_manifest_sha256": "a" * 64,
    "profile_config_sha256": "b" * 64,
    "code_commit": "c" * 40,
}


@pytest.fixture(scope="module")
def profile():
    return load_profile(REPO / "configs" / "preprocessing.yaml")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def corpus(tmp_path):
    """Seven clips (tones, noise, one 22.05 kHz stereo) over three splits plus a test row."""
    rng = np.random.default_rng(2026)
    root = tmp_path / "for"
    specs = [  # (path, label, split, sample_rate, kind)
        ("train/real/b.wav", "REAL", "train", 16000, "tone"),
        ("train/fake/a.wav", "FAKE", "train", 16000, "noise"),
        ("train/real/a.wav", "REAL", "train", 22050, "stereo"),
        ("train/fake/c.wav", "FAKE", "train", 16000, "tone"),
        ("val/a.wav", "REAL", "val-A", 16000, "noise"),
        ("val/b.wav", "FAKE", "val-A", 16000, "tone"),
        ("valb/a.wav", "FAKE", "val-B", 16000, "noise"),
        ("test/a.wav", "REAL", "test", 16000, "tone"),
    ]
    rows = []
    for i, (rel, label, split, rate, kind) in enumerate(specs):
        n = int(rate * (1.5 + 0.2 * i))
        t = np.arange(n) / rate
        if kind == "noise":
            data = 0.1 * rng.standard_normal(n)
        else:
            data = 0.3 * np.sin(2 * np.pi * (220 + 50 * i) * t)
        if kind == "stereo":
            data = np.stack([data, 0.5 * data], axis=1)
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, data.astype(np.float32), rate, subtype="PCM_16")
        rows.append({
            "path": rel, "label": label, "subgroup": "REAL" if label == "REAL" else "FAKE_other",
            "group": f"g{i}", "split": split, "hash": _sha(path),
        })
    return root, pd.DataFrame(rows, dtype=str)


def _build(corpus, profile, split, out):
    root, manifest = corpus
    return build_mfcc_cache(root, manifest, profile, split, out, **PROVENANCE)


def test_shapes_dtypes_labels_and_path_order(corpus, profile, tmp_path):
    meta = _build(corpus, profile, "train", tmp_path / "cache")
    features, labels, paths, loaded = load_mfcc_cache(tmp_path / "cache", "train", profile)
    assert features.shape == (4, 40) and features.dtype == np.float32
    assert labels.dtype == np.int8 and paths.dtype.kind == "U"
    assert paths.tolist() == [
        "train/fake/a.wav", "train/fake/c.wav", "train/real/a.wav", "train/real/b.wav"
    ]
    assert labels.tolist() == [1, 1, 0, 0]
    assert loaded == meta
    assert (meta["n_rows"], meta["n_spoof"], meta["n_bonafide"]) == (4, 2, 2)
    assert meta["profile_fingerprint"] == profile.fingerprint()
    assert sorted((tmp_path / "cache").iterdir()) == [
        tmp_path / "cache" / "train.json", tmp_path / "cache" / "train.npz"
    ]


@pytest.mark.parametrize("split", ["val-A", "val-B"])
def test_validation_splits_are_cached(corpus, profile, tmp_path, split):
    _build(corpus, profile, split, tmp_path / split)
    _, labels, paths, meta = load_mfcc_cache(tmp_path / split, split, profile)
    root, manifest = corpus
    assert paths.tolist() == sorted(manifest.loc[manifest.split == split, "path"])
    assert meta["n_rows"] == len(labels)


@pytest.mark.parametrize("split", ["test", "Test", "val", "", None])
def test_test_and_unknown_splits_raise_before_reading(corpus, profile, tmp_path, monkeypatch, split):
    calls = []
    monkeypatch.setattr(cache_module, "read_verified_audio", lambda *a: calls.append(a))
    monkeypatch.setattr(cache_module, "sha256_file", lambda *a: calls.append(a))
    with pytest.raises(ValueError, match="never cached"):
        _build(corpus, profile, split, tmp_path / "cache")
    with pytest.raises(ValueError, match="never cached"):
        load_mfcc_cache(tmp_path / "cache", split, profile)
    assert calls == []
    assert not (tmp_path / "cache").exists()


def test_cli_refuses_test_split_before_git_or_inputs(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cache_module, "code_version", lambda: calls.append("git"))
    monkeypatch.setattr(cache_module, "load_f2_manifest", lambda p: calls.append("manifest"))
    with pytest.raises(ValueError, match="never cached"):
        cache_module.run(tmp_path, tmp_path / "m.csv", tmp_path / "p.yaml", "test", tmp_path / "o")
    with pytest.raises(SystemExit):  # argparse choices exclude "test"
        cache_module.main(["--root", "r", "--manifest", "m", "--profile", "p",
                           "--split", "test", "--out", "o"])
    assert calls == []


def test_sample_hash_mismatch_raises_and_writes_nothing(corpus, profile, tmp_path):
    root, manifest = corpus
    manifest = manifest.copy()
    manifest.loc[manifest.path == "train/real/b.wav", "hash"] = "0" * 64  # last row in order
    out = tmp_path / "cache"
    with pytest.raises(PreprocessingError, match="SHA-256 mismatch"):
        build_mfcc_cache(root, manifest, profile, "train", out, **PROVENANCE)
    assert not out.exists()


def test_existing_out_dir_is_refused(corpus, profile, tmp_path, monkeypatch):
    out = tmp_path / "cache"
    out.mkdir()
    calls = []
    monkeypatch.setattr(cache_module, "read_verified_audio", lambda *a: calls.append(a))
    with pytest.raises(ValueError, match="already exists"):
        _build(corpus, profile, "train", out)
    assert calls == [] and list(out.iterdir()) == []


def test_invalid_label_and_provenance_raise(corpus, profile, tmp_path):
    root, manifest = corpus
    bad = manifest.copy()
    bad.loc[0, "label"] = "SPOOF"
    with pytest.raises(ValueError, match="FAKE or REAL"):
        build_mfcc_cache(root, bad, profile, "train", tmp_path / "a", **PROVENANCE)
    with pytest.raises(ValueError, match="code_commit"):
        build_mfcc_cache(root, manifest, profile, "train", tmp_path / "b",
                         **{**PROVENANCE, "code_commit": "dirty"})
    with pytest.raises(ValueError, match="no rows"):
        build_mfcc_cache(root, manifest[manifest.split != "val-B"], profile, "val-B",
                         tmp_path / "c", **PROVENANCE)
    assert not any((tmp_path / name).exists() for name in "abc")


def test_two_builds_are_byte_identical(corpus, profile, tmp_path):
    _build(corpus, profile, "train", tmp_path / "one")
    _build(corpus, profile, "train", tmp_path / "two")
    for name in ("train.npz", "train.json"):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()
    text = (tmp_path / "one" / "train.json").read_text(encoding="utf-8")
    meta = json.loads(text)
    assert list(meta) == sorted(meta)
    assert meta["npz_sha256"] == _sha(tmp_path / "one" / "train.npz")
    assert meta["schema_version"] == 1 and meta["code_commit"] == "c" * 40


def test_npz_has_no_object_arrays(corpus, profile, tmp_path):
    _build(corpus, profile, "train", tmp_path / "cache")
    with np.load(tmp_path / "cache" / "train.npz", allow_pickle=False) as npz:
        assert sorted(npz.files) == ["features", "labels", "paths"]
        assert all(npz[k].dtype != object and not npz[k].dtype.hasobject for k in npz.files)


def test_features_equal_extract_features(corpus, profile, tmp_path):
    root, _ = corpus
    _build(corpus, profile, "train", tmp_path / "cache")
    features, _, paths, _ = load_mfcc_cache(tmp_path / "cache", "train", profile)
    for row, rel in zip(features, paths):
        waveform, rate = sf.read(root / rel, dtype="float32", always_2d=False)
        np.testing.assert_array_equal(row, extract_features(waveform, rate, profile)[2])


def _rewrite_npz(path: Path, **changes) -> None:
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays.update(changes)
    buffer = io.BytesIO()
    np.savez(buffer, **{k: v for k, v in arrays.items() if v is not None})
    path.write_bytes(buffer.getvalue())


def _rehash(out: Path) -> None:
    json_path = out / "train.json"
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    meta["npz_sha256"] = _sha(out / "train.npz")
    json_path.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")


def test_load_refuses_tampered_npz(corpus, profile, tmp_path):
    out = tmp_path / "cache"
    _build(corpus, profile, "train", out)
    with np.load(out / "train.npz", allow_pickle=False) as npz:
        features = npz["features"].copy()
    features[0, 0] += 1.0
    _rewrite_npz(out / "train.npz", features=features)
    with pytest.raises(ValueError, match="npz SHA-256 check failed"):
        load_mfcc_cache(out, "train", profile)


@pytest.mark.parametrize("change, message", [
    (lambda m: m.update(extra=1), "metadata keys check failed"),
    (lambda m: m.pop("code_commit"), "metadata keys check failed"),
    (lambda m: m.update(n_split=m.pop("split")), "metadata keys check failed"),
    (lambda m: m.update(split="val-A"), "split is"),
    (lambda m: m.update(schema_version=2), "schema_version"),
    (lambda m: m.update(n_spoof=3), "n_spoof"),
])
def test_load_refuses_changed_json(corpus, profile, tmp_path, change, message):
    out = tmp_path / "cache"
    _build(corpus, profile, "train", out)
    meta = json.loads((out / "train.json").read_text(encoding="utf-8"))
    change(meta)
    (out / "train.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_mfcc_cache(out, "train", profile)


def test_load_refuses_other_profile_fingerprint(corpus, profile, tmp_path):
    out = tmp_path / "cache"
    _build(corpus, profile, "train", out)
    other = dataclasses.replace(profile, features={**profile.features, "fmax_hz": 8000.0})
    assert other.fingerprint() != profile.fingerprint()
    with pytest.raises(ValueError, match="profile fingerprint check failed"):
        load_mfcc_cache(out, "train", other)


@pytest.mark.parametrize("change, message", [
    ({"labels": np.array([1, 1, 0, 0], dtype=np.int64)}, "labels check failed"),
    ({"labels": np.array([1, 2, 0, 0], dtype=np.int8)}, "labels check failed"),
    ({"features": np.zeros((4, 39), dtype=np.float32)}, "features check failed"),
    ({"features": np.zeros((4, 40), dtype=np.float64)}, "features check failed"),
    ({"paths": np.array(["d", "c", "b", "a"])}, "paths check failed"),
    ({"paths": None}, "npz entries check failed"),
])
def test_load_refuses_bad_arrays_even_with_matching_hash(corpus, profile, tmp_path, change, message):
    out = tmp_path / "cache"
    _build(corpus, profile, "train", out)
    _rewrite_npz(out / "train.npz", **change)
    _rehash(out)
    with pytest.raises(ValueError, match=message):
        load_mfcc_cache(out, "train", profile)


def test_load_refuses_object_array(corpus, profile, tmp_path):
    out = tmp_path / "cache"
    _build(corpus, profile, "train", out)
    buffer = io.BytesIO()
    np.savez(buffer, features=np.zeros((4, 40), dtype=np.float32),
             labels=np.array([1, 1, 0, 0], dtype=np.int8),
             paths=np.array(["a", "b", "c", "d"], dtype=object))
    (out / "train.npz").write_bytes(buffer.getvalue())
    _rehash(out)
    with pytest.raises(ValueError):
        load_mfcc_cache(out, "train", profile)


def test_load_refuses_missing_file_and_corrupt_npz(corpus, profile, tmp_path):
    out = tmp_path / "cache"
    _build(corpus, profile, "train", out)
    with pytest.raises(ValueError, match="missing cache file"):
        load_mfcc_cache(out, "val-A", profile)
    (out / "train.npz").write_bytes(b"not a zip")
    _rehash(out)
    with pytest.raises(ValueError, match="npz read check failed"):
        load_mfcc_cache(out, "train", profile)


def test_cli_records_provenance_from_files(corpus, profile, tmp_path, monkeypatch):
    root, manifest = corpus
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    monkeypatch.setattr(cache_module, "code_version",
                        lambda: {"git_commit": "d" * 40, "git_dirty": False})
    monkeypatch.setattr(cache_module, "load_f2_manifest",
                        lambda p: pd.read_csv(p, dtype=str, keep_default_na=False))
    profile_path = REPO / "configs" / "preprocessing.yaml"
    cache_module.main(["--root", str(root), "--manifest", str(manifest_path),
                       "--profile", str(profile_path), "--split", "val-A",
                       "--out", str(tmp_path / "out")])
    _, _, _, meta = load_mfcc_cache(tmp_path / "out", "val-A", profile)
    assert meta["code_commit"] == "d" * 40
    assert meta["f2_manifest_sha256"] == _sha(manifest_path)
    assert meta["profile_config_sha256"] == _sha(profile_path)


def test_cli_refuses_dirty_worktree_before_reading(monkeypatch, tmp_path):
    def dirty():
        raise PreprocessingError("requires a clean Git worktree")
    calls = []
    monkeypatch.setattr(cache_module, "code_version", dirty)
    monkeypatch.setattr(cache_module, "load_f2_manifest", lambda p: calls.append(p))
    monkeypatch.setattr(cache_module, "load_profile", lambda p: calls.append(p))
    with pytest.raises(PreprocessingError, match="clean Git worktree"):
        cache_module.run(tmp_path, tmp_path / "m.csv", tmp_path / "p.yaml", "train", tmp_path / "o")
    assert calls == [] and not (tmp_path / "o").exists()
