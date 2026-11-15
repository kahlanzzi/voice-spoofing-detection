"""Raw train waveform cache (Track V, D50 (1)): round trip, determinism, refusals. Synthetic WAVs only."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import src.features.raw_train_cache as raw_module
from src.features.raw_train_cache import (
    JSON_NAME,
    NPZ_NAME,
    build_raw_train_cache,
    load_raw_train_cache,
)
from src.preprocessing.validation import read_verified_audio
from tests._v2_fakes import COMMIT, sha, write_dataset


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    base = tmp_path_factory.mktemp("raw")
    root, manifest, manifest_path = write_dataset(base)
    f2 = sha(manifest_path)
    out = base / "cache"
    meta = build_raw_train_cache(root, manifest, "train", out, f2_manifest_sha256=f2, code_commit=COMMIT)
    return {"base": base, "root": root, "manifest": manifest, "f2": f2, "out": out, "meta": meta}


def _build(data, out, **kw):
    args = {"f2_manifest_sha256": data["f2"], "code_commit": COMMIT, **kw}
    return build_raw_train_cache(data["root"], data["manifest"], "train", out, **args)


def test_round_trip_matches_decoded_audio(data):
    cache = load_raw_train_cache(data["out"], expected_f2_manifest_sha256=data["f2"])
    rows = data["manifest"][data["manifest"].split == "train"].sort_values("path")
    assert len(cache) == 24 and cache.paths.tolist() == rows["path"].tolist()
    assert cache.groups.tolist() == rows["group"].tolist()
    assert cache.labels.tolist() == rows["label"].map({"FAKE": 1, "REAL": 0}).tolist()
    for i, (_, row) in enumerate(rows.iterrows()):
        decoded, rate = read_verified_audio(data["root"], row)
        assert rate == 16000 and np.array_equal(cache.clip(i), decoded)
    lengths = np.diff(cache.offsets)
    assert lengths.min() >= 24000 and lengths.max() <= 40000  # 1.5-2.5 s
    assert set(cache.meta) == set(raw_module.JSON_KEYS)
    assert cache.meta["total_samples"] == cache.samples.size == int(cache.offsets[-1])
    assert (cache.meta["n_spoof"], cache.meta["n_bonafide"]) == (12, 12)
    with pytest.raises(ValueError):
        cache.samples[0] = 1.0  # read-only views
    with pytest.raises(IndexError):
        cache.clip(24)


def test_rebuild_is_byte_identical(data, tmp_path):
    meta = _build(data, tmp_path / "again")
    assert meta == data["meta"]
    assert (tmp_path / "again" / NPZ_NAME).read_bytes() == (data["out"] / NPZ_NAME).read_bytes()


def test_refuses_other_split_before_reading(data, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a file was read")

    monkeypatch.setattr(raw_module, "read_verified_audio", boom)
    for split in ("val-A", "val-B", "external"):
        with pytest.raises(ValueError, match="split must be 'train'"):
            build_raw_train_cache(data["root"], data["manifest"], split, tmp_path / split,
                                  f2_manifest_sha256=data["f2"], code_commit=COMMIT)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("rate, channels, match", [(22050, 1, "sample rate"), (16000, 2, "one channel")])
def test_refuses_non_16k_or_stereo(tmp_path, rate, channels, match):
    root, manifest, manifest_path = write_dataset(tmp_path, sizes={"train": 4}, rate=rate,
                                                  channels=channels)
    with pytest.raises(ValueError, match=match):
        build_raw_train_cache(root, manifest, "train", tmp_path / "out",
                              f2_manifest_sha256=sha(manifest_path), code_commit=COMMIT)
    assert not (tmp_path / "out").exists()


def test_refuses_sample_hash_mismatch(tmp_path):
    root, manifest, manifest_path = write_dataset(tmp_path, sizes={"train": 4})
    first = root / manifest.iloc[0]["path"]
    data, rate = sf.read(first, dtype="float32")
    sf.write(first, data * 0.5, rate, subtype="PCM_16")
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        build_raw_train_cache(root, manifest, "train", tmp_path / "out",
                              f2_manifest_sha256=sha(manifest_path), code_commit=COMMIT)
    assert not (tmp_path / "out").exists()


def test_refuses_existing_out_dir_and_repo_out_dir(data, tmp_path):
    (tmp_path / "exists").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _build(data, tmp_path / "exists")
    with pytest.raises(ValueError, match="inside the repository"):
        _build(data, raw_module.REPO_ROOT / "data" / "raw_cache_should_not_exist")
    assert not (raw_module.REPO_ROOT / "data" / "raw_cache_should_not_exist").exists()


def test_refuses_bad_provenance(data, tmp_path):
    with pytest.raises(ValueError, match="lowercase hex"):
        _build(data, tmp_path / "a", code_commit="x")
    with pytest.raises(ValueError, match="lowercase hex"):
        _build(data, tmp_path / "b", f2_manifest_sha256="0" * 63)


def _copy(data, tmp_path) -> Path:
    out = tmp_path / "copy"
    shutil.copytree(data["out"], out)
    return out


def test_loader_detects_tampered_npz(data, tmp_path):
    out = _copy(data, tmp_path)
    blob = bytearray((out / NPZ_NAME).read_bytes())
    blob[-100] ^= 0xFF
    (out / NPZ_NAME).write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="npz SHA-256 check failed"):
        load_raw_train_cache(out, expected_f2_manifest_sha256=data["f2"])


@pytest.mark.parametrize("key, value, match", [
    ("split", "val-A", "split is"),
    ("sample_rate", 22050, "sample_rate"),
    ("n_spoof", 11, "n_spoof"),
    ("f2_manifest_sha256", "e" * 64, "F2 manifest check failed"),
    ("extra", 1, "metadata keys"),
])
def test_loader_refuses_bad_metadata(data, tmp_path, key, value, match):
    out = _copy(data, tmp_path)
    meta = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    meta[key] = value
    (out / JSON_NAME).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_raw_train_cache(out, expected_f2_manifest_sha256=data["f2"])


def _rewrite_npz(out, **changes):
    with np.load(out / NPZ_NAME) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays.update(changes)
    raw_module._write_npz(out / NPZ_NAME, arrays)
    meta = json.loads((out / JSON_NAME).read_text(encoding="utf-8"))
    meta["npz_sha256"] = sha(out / NPZ_NAME)
    (out / JSON_NAME).write_text(json.dumps(meta), encoding="utf-8")
    return arrays


@pytest.mark.parametrize("change, match", [
    (lambda a: {"samples": a["samples"].astype(np.float64)}, "samples check"),
    (lambda a: {"samples": np.where(np.arange(a["samples"].size) == 5, np.nan, a["samples"]).astype(np.float32)},
     "NaN"),
    (lambda a: {"offsets": a["offsets"][::-1].copy()}, "offsets check"),
    (lambda a: {"offsets": a["offsets"].astype(np.int32)}, "offsets check"),
    (lambda a: {"labels": a["labels"].astype(np.int64)}, "labels check"),
    (lambda a: {"paths": a["paths"][::-1].copy()}, "sorted"),
    (lambda a: {"groups": a["groups"][:3].copy()}, "groups check"),
])
def test_loader_refuses_bad_arrays(data, tmp_path, change, match):
    out = _copy(data, tmp_path)
    with np.load(out / NPZ_NAME) as npz:
        arrays = {k: npz[k] for k in npz.files}
    _rewrite_npz(out, **change(arrays))
    with pytest.raises(ValueError, match=match):
        load_raw_train_cache(out, expected_f2_manifest_sha256=data["f2"])


def test_cli_verify(data, capsys, monkeypatch):
    # verify checks the approved F2 hash by default, so a synthetic cache is refused (fail closed)
    with pytest.raises(ValueError, match="F2 manifest check failed"):
        raw_module.main(["verify", "--dir", str(data["out"])])
    monkeypatch.setattr(raw_module.load_raw_train_cache, "__kwdefaults__",
                        {"expected_f2_manifest_sha256": data["f2"]})
    raw_module.main(["verify", "--dir", str(data["out"])])
    assert json.loads(capsys.readouterr().out) == data["meta"]
